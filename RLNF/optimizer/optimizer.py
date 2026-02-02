import torch
import torch.nn as nn
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from typing import Dict
from ..utils.rollout import _seq_logprob_ctc, _blank_index, _ensure_log_softmax, _normalize_adv, _group_statistics

class SCSTOptimizer:
    """
    Self-Critical Sequence Training (SCST) for ASR with CTC.
    Handles multiple hypotheses per audio and baseline computation (max or mean).
    """
    def __init__(
        self,
        actor: EncDecCTCModel | EncDecCTCModelBPE,
        lr: float = 1e-5,
        alpha : float = 1.0,
        beta : float = 0.5,
        gamma : float = 0.2,
        ent_coeff : float = 0.01,
        regularize : bool = False,
        baseline: str = "max",  # "max" or "mean" or "greedy"
        device: torch.device = torch.device("cpu"),
        amp: bool = False
    ):
        self.actor = actor.to(device)
        self.device = device
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.baseline = baseline
        self.blank_idx = _blank_index(actor)
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ent_coeff = ent_coeff
        self.regularize = regularize

    @staticmethod
    def _compute_baseline(reward: torch.Tensor, indexes: torch.Tensor, method: str = "max",
                          greedy_reward : torch.Tensor = None) -> torch.Tensor:
        """
        Compute baseline per audio.
        """

        baseline = torch.zeros_like(reward)

        if method == "greedy" :

            if greedy_reward is None or len(greedy_reward) != len(indexes.unique()) : 
                raise ValueError("greedy_rewards required for method='greedy'")
            
            for i, uid in enumerate(indexes.unique()) : 
                mask = indexes == uid
                baseline[mask] = greedy_reward[i]

        else : 

            for uid in indexes.unique():
                mask = indexes == uid
                if method == "max":
                    baseline[mask] = reward[mask].max()
                elif method == "mean":
                    baseline[mask] = reward[mask].mean()
                else:
                    raise ValueError(f"Unknown baseline method: {method}")
                
        return baseline

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute SCST loss and update actor.

        Expected batch keys:
            - audio_batch: [B_audio, C, T]
            - audio_lengths: [B_audio]
            - targets: [B_hyp, Lmax] (padded)
            - target_lengths: [B_hyp]
            - input_lengths: [B_hyp] (from CTC)
            - log_probs_old: [B_hyp] (CTC log-probs per hypothesis)
            - reward: [B_hyp]
            - indexes: [B_hyp] mapping each hypothesis to its audio
        """
        audio = batch["audio_batch"].to(self.device)
        audio_lens = batch["audio_lengths"].to(self.device)
        targets = batch["targets"].to(self.device)
        target_lens = batch["target_lengths"].to(self.device)
        input_lens = batch["input_lengths"].to(self.device)
        reward = batch["reward"].to(self.device)
        indexes = batch["indexes"].to(self.device)

        wers = batch["wers"].to(self.device)
        cers = batch["cers"].to(self.device)

        wers_greedy = batch["wers"].to(self.device)
        cers_greedy = batch["cers"].to(self.device)

        greedy_rewards = batch["greedy"].to(self.device)

        self.actor.train()
        self.opt.zero_grad()

        # === mixed precision context ===
        with torch.amp.autocast(device_type="cuda", enabled=self.amp):
            # Forward pass through ASR model
            out = self.actor(processed_signal=audio, processed_signal_length=audio_lens)
            if isinstance(out, (list, tuple)):
                logits, _ = out[0], out[1]
            else:
                raise RuntimeError("Expected tuple (logits, enc_len) from actor forward")

            log_probs = _ensure_log_softmax(logits)

            # Sequence log-prob per hypothesis (CTC)
            seq_logp = _seq_logprob_ctc(log_probs, input_lens, targets, target_lens, self.blank_idx)

           

            baseline_wers = self._compute_baseline(wers, indexes, method=self.baseline, greedy_reward=greedy_rewards)
            baseline_cers = self._compute_baseline(cers, indexes, method=self.baseline, greedy_rewards=greedy_rewards)

            wer_c = _group_statistics(-wers, baseline_wers,indexes)
            cer_c = _group_statistics(-cers, baseline_cers,indexes)

            #wer_c = wer_c / (wer_c.abs().max(dim=0, keepdim=True)[0] + 1e-8)
            #cer_c = cer_c / (cer_c.abs().max(dim=0, keepdim=True)[0] + 1e-8)


            reward_p = self.alpha * reward + self.beta * wer_c.tanh() - self.gamma * cer_c.tanh()


            # Compute baseline per audio
            baseline_p = self._compute_baseline(reward_p, indexes, self.baseline, greedy_reward=greedy_rewards)
            
            # Advantage = reward - baseline
            advantage = _group_statistics(reward=reward_p, values_old=baseline_p, indexes=indexes)

            # Diagnostics BEFORE detach
            adv_mean = advantage.mean().item()
            adv_std = advantage.std(unbiased=False).item()
            seq_logp_mean = seq_logp.mean().item()
            seq_logp_std = seq_logp.std(unbiased=False).item()

            if self.regularize :

                probs = log_probs.exp()
                entropy_per_step = -(probs * log_probs).sum(dim=-1)
                entropy = entropy_per_step.mean()
                loss = -(advantage * seq_logp).mean() - self.ent_coeff * entropy

            else : 

                loss = -(advantage * seq_logp).mean() 

        # Backprop with GradScaler (AMP aware)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()

        # Grad-norm diagnostic (sample few params)
        total_grad_norm = 0.0
        cnt = 0
        for p in self.actor.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.detach().norm().item()
                cnt += 1
        avg_grad_norm = (total_grad_norm / cnt) if cnt else 0.0

        return {
            "scst_loss": float(loss.detach().cpu()),
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "seq_logp_mean": seq_logp_mean,
            "seq_logp_std": seq_logp_std,
            "reward_mean": float(reward.mean().cpu()),
            "reward_p_mean": float(reward_p.mean().cpu()),
            "baseline_p_mean": float(baseline_p.mean().cpu()),
            "avg_grad_norm": avg_grad_norm,
            "cos(wers)_mean" : wers.cos().mean().cpu(),
            "sin(cers)_mean" : cers.sin().mean().cpu(),
            "cer_c_mean" : cer_c.tanh().mean().cpu(),
            "wer_c_mean" : wer_c.tanh().mean().cpu()
        }
