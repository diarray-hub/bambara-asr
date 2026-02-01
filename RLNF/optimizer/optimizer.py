import torch
import torch.nn as nn
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from typing import Dict
from ..utils.rollout import _seq_logprob_ctc, _blank_index, _ensure_log_softmax, ppo_group_statistics, _same_num_hypotheses, _mean_mean

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
        beta : float = 0.2,
        gamma : float = 0.1,
        baseline: str = "max",  # "max" or "mean"
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

    @staticmethod
    def _compute_baseline(reward: torch.Tensor, indexes: torch.Tensor, method: str = "max") -> torch.Tensor:
        """
        Compute baseline per audio.
        """

        if _same_num_hypotheses(indexes) :

            reward = reward

        else : 
            reward, _, _ = _mean_mean(reward, indexes, False)

            reward = reward["indexes"]

        baseline = torch.zeros_like(reward)

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

            #if not _same_num_hypotheses(indexes) :
            #    seq_logp, _, _ = _mean_mean(seq_logp, indexes, False)
            #    seq_logp = seq_logp["indexes"]

            reward_p = self.alpha * reward + self.beta * wers.cos() - self.gamma * cers.sin()

            # Compute baseline per audio
            baseline_p = self._compute_baseline(reward_p, indexes, self.baseline)
            baseline = self._compute_baseline(reward, indexes, self.baseline)

            # Advantage = reward - baseline
            advantage = reward_p - baseline_p
            adv = reward - baseline

            ADV = ppo_group_statistics(reward, baseline, indexes)
            ADV_p = ppo_group_statistics(reward_p, baseline_p, indexes)


            # SCST loss
            loss = -(advantage.detach() * seq_logp).mean()

            loss_ = -(adv.detach() * seq_logp).mean()

            l_ = -(ADV.detach() * seq_logp).mean()
            l = -(ADV_p.detach() * seq_logp).mean()


        # Backprop with GradScaler (AMP aware)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()

        return {
            "scst_loss": loss.item(),
            "reward_mean": reward.mean().item(),
            "reward_p" : reward_p.mean().item(),
            "baseline_p_mean": baseline_p.mean().item(),
            "adv_p_mean": advantage.mean().item(),
            "baseline_mean": baseline.mean().item(),
            "adv_mean": adv.mean().item(),
            "loss_" : loss_.item(),
            "ADV" :ADV.mean().item(),
            "ADV_p" : ADV_p.mean().item(),
            "l_n_p" : l.item(),
            "l_n" : l_.item()
        }
