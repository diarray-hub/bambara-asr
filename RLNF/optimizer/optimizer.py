
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from ..loss.ppo_loss import PPOLoss 
from ..PPO.critic_network import CriticModel
from ..utils.rollout import _blank_index, _seq_logprob_ctc, _ensure_log_softmax, _mean_mean, ppo_group_statistics, _normalize_adv, _same_num_hypotheses


class PPOOptimizer:
    """
    Core PPO optimizer handling actor and critic updates with CTC sequence log-probs.
    """
    def __init__(
        self,
        actor: EncDecCTCModel | EncDecCTCModelBPE,
        critic: CriticModel,
        clip_eps: float = 0.2,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-4,
        K_updates: int = 4,
        entropy_coef: float = 0.0,
        device: torch.device = torch.device("cpu"),
        amp: bool = False,
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.K_updates = K_updates
        self.entropy_coef = entropy_coef
        self.amp = amp

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.ppo_loss = PPOLoss(clip_eps)

        #self.actor.train()
        #self.critic.train()
        #self._blank_idx = _blank_index(self.actor)
        actor_for_blank = self.actor.module if hasattr(self.actor, "module") else self.actor
        self._blank_idx = _blank_index(actor_for_blank)

        # If you're on CUDA with bf16/fp16 AMP, this will be enabled upstream when amp=True
        # (PyTorch 2.4+ has torch.amp; older uses torch.cuda.amp — keep to your env)
        self._scaler = torch.amp.GradScaler(enabled=amp)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform K PPO updates on the provided batch.

        Expected batch keys (CPU tensors unless noted):
            audio_batch: [B, C, T]
            audio_lengths: [B]
            targets: [B, Lmax]
            target_lengths: [B]
            input_lengths: [B]
            log_probs_old: [B]
            reward: [B]
            values: [B]
        """
        # Move once per epoch to device
        audio = batch["audio_batch"].to(self.device, non_blocking=True)
        a_len = batch["audio_lengths"].to(self.device, non_blocking=True)
        targets = batch["targets"].to(self.device, non_blocking=True)
        t_len = batch["target_lengths"].to(self.device, non_blocking=True)
        in_len = batch["input_lengths"].to(self.device, non_blocking=True)

        logp_old = batch["log_probs_old"].to(self.device, non_blocking=True).detach()
        reward = batch["reward"].to(self.device, non_blocking=True)
        values_old = batch["values"].to(self.device, non_blocking=True)
        
        indexes = batch["indexes"].to(device=self.device)

        # Old advantages (on-policy): use stored old values for stability
        #adv = self._normalize_adv(reward - values_old, indexes = indexes).detach()
        
        adv, rewards_centered, mode = ppo_group_statistics(reward=reward, values_old=values_old, indexes=indexes)
        
        
        if not _same_num_hypotheses(indexes=indexes) :

            #adv_mean, adv_std = _mean_mean(adv, indexes)
            logp_old_mean, _ = _mean_mean(logp_old, indexes)
            rewards_mean, _ = _mean_mean(reward, indexes)
            values_old_mean, _ = _mean_mean(values_old, indexes)
            
        else :
            #adv_mean, adv_std = adv.mean(), adv.std(unbiased=False)
            logp_old_mean = logp_old.mean()
            rewards_mean = reward.mean()
            values_old_mean = values_old.mean()

        self.actor.train()
        self.critic.train()

        for _ in range(self.K_updates):
            
            self.opt_actor.zero_grad(set_to_none=True)
            self.opt_critic.zero_grad(set_to_none=True)

            # autocast only when actually on GPU & amp=True
            with torch.amp.autocast(device_type="cuda", enabled=self.amp):
                # === Actor forward ===
                out = self.actor(processed_signal = audio, processed_signal_length  = a_len)
                
                if not isinstance(out, (list, tuple)) or len(out) < 2:
                    raise RuntimeError("Unexpected ASR forward() return; expected (logits_or_logp, enc_len, ...).")
                logits_or_logp3d, in_len_new = out[0], out[1]

                # === Ensure log-probs every time ===
                logp3d_new = _ensure_log_softmax(logits_or_logp3d)

                # === Finite check on actor outputs (early-exit if broken) ===
                if not torch.isfinite(logp3d_new).all():
                    # Reduce LR and skip the step; return diagnostics so the trainer can log it.
                    for g in self.opt_actor.param_groups:
                        
                        g["lr"] = max(g["lr"] * 0.5, 1e-7)
                    return {
                        "actor_loss": float("nan"),
                        "critic_loss": float("nan"),
                        "mean_value": float("nan"),
                        "ratio_mean": float("nan"),
                        "frac_clipped": 0.0,
                        "adv_mean": float(adv.mean().cpu()),
                        "adv_std": float(adv.std().cpu()),
                        "logp_old_mean": float(logp_old_mean.cpu()),
                        "logp_new_mean": float("nan"),
                        "reward_mean": float(rewards_mean.cpu()),
                        "V_hat_mean": float(values_old_mean.cpu()),
                        
                    }

                # === Clamp/validate lengths and mask invalid rows ===
                T = logp3d_new.size(1)
                in_len_use = in_len_new.clamp(min=1, max=T).int()
                t_len_use = t_len.clamp(min=1).int()
                valid = (t_len_use <= in_len_use)

                # Compute sequence log-prob only on valid rows; keep others ratio=1 (no learning)
                if valid.any():
                    logp_new_valid = _seq_logprob_ctc(
                        logp3d_new[valid], in_len_use[valid], targets[valid], t_len_use[valid], self._blank_idx
                    )
                    logp_new = torch.zeros_like(logp_old)
                    logp_new[valid] = logp_new_valid
                    logp_new[~valid] = logp_old[~valid]
                else:
                    logp_new = logp_old.clone()

                # Critic forward
                V_hat = self.critic(audio = audio)  # [B]

                # PPO actor loss (uses old log-prob & normalized advantage)
                # Zero-out advantages for invalid rows to fully mask them
                adv_use = adv * valid.to(adv.dtype)
                loss_actor = self.ppo_loss(logp_old, logp_new, adv_use, indexes)

                # Optional entropy bonus
                if self.entropy_coef > 0:
                    ent = -(logp3d_new.exp() * logp3d_new).sum(dim=(1, 2)).mean()
                    loss_actor = loss_actor - self.entropy_coef * ent

                #rewards_centered = rewards_means[indexes]
                # Critic loss vs actual reward
                loss_critic = F.mse_loss(V_hat, rewards_centered)
                
                if not _same_num_hypotheses(indexes=indexes) :
                    
                    logp_new_mean, _ = _mean_mean(logp_new, indexes)
                
                else :
                    logp_new_mean =  logp_new.mean()
                    

                # Diagnostics
                with torch.no_grad():
                    ratio = torch.exp(logp_new - logp_old)
                    
                    # Ignore invalid rows for clipping stat
                    ratio_valid = ratio[valid] if valid.any() else ratio
                    
                    if not _same_num_hypotheses(indexes=indexes):
                        # Mean-of-means par audio
                        ratio_means, ratio_mean, ratio_std = _mean_mean(ratio_valid, indexes, only=False)
                        ratio_for_diag = ratio_means[indexes] 
                        frac_clipped = ((ratio_for_diag > 1.0 + self.ppo_loss.clip_eps) |
                                        (ratio_for_diag < 1.0 - self.ppo_loss.clip_eps)).float().mean().item()
                        ratio_mean_diag = ratio_mean
                    else:
                        # Même hypothèse → simple moyenne
                        ratio_mean_diag = ratio.mean().item()
                        if ratio_valid.numel() > 0:
                            frac_clipped = ((ratio_valid > 1.0 + self.ppo_loss.clip_eps) |
                                            (ratio_valid < 1.0 - self.ppo_loss.clip_eps)).float().mean().item()
                        else:
                            frac_clipped = 0.0
                            
                    diag = {
                        "adv_mean": float(adv.mean().cpu()),
                        "adv_std": float(adv.std().cpu()),
                        "ratio_mean": float(ratio_mean_diag),
                        "frac_clipped": float(frac_clipped),
                        "logp_old_mean": float(logp_old_mean.cpu()),
                        "logp_new_mean": float(logp_new_mean.cpu()),
                        "reward_mean": float(rewards_mean.cpu()),
                        "V_hat_mean": float(V_hat.mean().cpu()),
                    }

            # === Backprop + Grad clipping (AMP-aware) ===
            total_loss = loss_actor + 0.5 * loss_critic
            if self._scaler.is_enabled():
                self._scaler.scale(total_loss).backward()
                # Unscale before clipping
                self._scaler.unscale_(self.opt_actor)
                self._scaler.unscale_(self.opt_critic)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self._scaler.step(self.opt_actor)
                self._scaler.step(self.opt_critic)
                self._scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_actor.step()
                self.opt_critic.step()

        # Return last-iter losses + key diagnostics
        out_stats = {
            "actor_loss": float(loss_actor.detach().cpu().item()),
            "critic_loss": float(loss_critic.detach().cpu().item()),
            "mean_value": float(V_hat.detach().mean().cpu().item()),
            "mode" : float(mode)
        }
        out_stats.update(diag)
        return out_stats
