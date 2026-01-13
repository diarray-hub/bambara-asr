import torch
import torch.nn as nn
from RLNF.utils.rollout import _mean_mean

def _clip_surrogate(ratio: torch.Tensor, adv: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Compute clipped surrogate objective term per sample: min(ratio * adv,
    clipped_ratio * adv).
    """
    clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
    return torch.minimum(ratio * adv, clipped_ratio * adv)

class PPOLoss(nn.Module):
    """
    Proximal Policy Optimization (PPO) clipped surrogate loss for on-policy RL.

    Loss = -mean(min(ratio*adv, clipped_ratio*adv))
    where ratio = exp(logp_new - logp_old).
    """
    def __init__(self, clip_eps: float = 0.2):
        """
        Args:
            clip_eps: clipping epsilon (e.g., 0.1 or 0.2)
        """
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        logp_old: torch.Tensor,
        logp_new: torch.Tensor,
        advantages: torch.Tensor,
        indexes : torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PPO clipped loss.

        Args:
            logp_old: [B] log-prob under old policy
            logp_new: [B] log-prob under new policy
            advantages: [B] estimated advantages

        Returns:
            loss: scalar tensor
        """
        # probability ratio
        ratio = torch.exp(logp_new - logp_old)
        # clipped surrogate
        clipped_obj = _clip_surrogate(ratio, advantages, self.clip_eps)
        
        loss, _ = _mean_mean(clipped_obj, indexes)
        
        # PPO loss is negative of the objective
        return -loss
