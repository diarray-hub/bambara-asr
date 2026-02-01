from ..Rewards.reward_model import RewardModel, masked_mean_pooling
from ..Rewards.reward_config import RewardConfig
import torch.nn.functional as F

class CriticModel(RewardModel) :
    
    def __init__(self, config : RewardConfig):
        
        super().__init__(config)
        
        self.combined_dim = self.cfg.audio_conv_channels
        
        
        self.head = self.build_head(self.combined_dim)
        
    def forward(self, audio, labels=None, reward=None, **kwargs):
        
        audio_enc = self.audio_encoder(audio).squeeze(-1)
        
        pred = self.head(audio_enc).squeeze(-1)
        
        if reward is None :
            
            return pred
        
        return F.mse_loss(pred, reward)


        
        
        
        
