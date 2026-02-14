import torch
import torch.nn as nn
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from typing import Dict
from ..utils.rollout import _seq_logprob_ctc, _blank_index, _ensure_log_softmax

class DistillationOptimizer:
    """
    Simple CTC distillation: maximize log-prob of teacher targets.
    """
    def __init__(
        self,
        student: EncDecCTCModel | EncDecCTCModelBPE,
        lr: float = 1e-5,
        device: torch.device = torch.device("cpu"),
        amp: bool = False
    ):
        self.student = student.to(device)
        self.device = device
        self.opt = torch.optim.Adam(self.student.parameters(), lr=lr)
        self.blank_idx = _blank_index(student)
        self.amp = amp
        self.scaler = torch.amp.GradScaler(enabled=amp)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute distillation loss on selected teacher hypotheses.
        Expected batch keys:
            - audio: [B, C, T]
            - audio_len: [B]
            - targets: [B, Lmax]
            - target_lengths: [B]
        """
        
        audio = batch["audio"].to(self.device)
        audio_lens = batch["audio_len"].to(self.device)
        targets = batch["targets"].to(self.device)
        target_lens = batch["target_lengths"].to(self.device)
        input_lenght = batch["input_lenght"].to(self.device)
        scores = batch["score"]
        sample_weight = batch["sample_weight"].to(self.device)  # [B]


        self.student.train()
        self.opt.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=self.amp):
            # Forward student
            out = self.student(processed_signal=audio, processed_signal_length=audio_lens)
            if isinstance(out, (list, tuple)):
                logits, _ = out[0], out[1]
            else:
                raise RuntimeError("Expected tuple (logits, enc_len) from student forward")

            log_probs = _ensure_log_softmax(logits)

            # Sequence log-prob per sample
            seq_logp = _seq_logprob_ctc(
                log_probs_btv=log_probs,
                input_lengths_b=input_lenght,
                targets_padded_bl=targets,
                target_lengths_b=target_lens,
                blank_idx=self.blank_idx
            )

            # Maximize log-prob => minimize negative log-prob
            loss = -(sample_weight * seq_logp).mean()

        # Backprop with GradScaler
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()

        return {
            "distill_loss": float(loss.detach().cpu()),
            "seq_logp_mean": seq_logp.mean().item(),
            "seq_logp_std": seq_logp.std(unbiased=False).item(),
            "score_mean" : scores.mean().item(),
            "score_std" : scores.std(unbiased=False).item()
        }
