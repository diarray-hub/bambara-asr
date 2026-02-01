import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import PreTrainedModel
from .reward_config import RewardConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from ..utils.dot_dict import DotDict
import torch.nn.functional as F 


def masked_mean_pooling(outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    outputs: [B, T, D]
    mask: [B, T] (1 = token réel, 0 = padding)
    """
    mask = mask.unsqueeze(-1).float()  # [B, T, 1]
    masked_outputs = outputs * mask
    summed = masked_outputs.sum(dim=1)  # [B, D]
    denom = mask.sum(dim=1).clamp(min=1.0)  # éviter /0
    return summed / denom

class RewardModel(PreTrainedModel):
    
    config_class = RewardConfig
    
    """
    Reward model that predicts a human-assigned score in [0,1] from audio and text.
      - Uses a lightweight CNN + BiLSTM encoder (custom)
    
    """

    def __init__(self, config: RewardConfig):
        super().__init__(config=config)
        self.config = config
        
        self.cfg = DotDict(config.pretrained_config)
        
        # -------- Audio Encoder --------
        convs = []
        in_ch = 64 #self.cfg.n_mel
        for _ in range(self.cfg.audio_conv_layers):
            convs += [
                nn.Conv1d(in_ch, self.cfg.audio_conv_channels, kernel_size=self.cfg.kernel_size, stride=self.cfg.stride, padding=self.cfg.padding),
                nn.BatchNorm1d(self.cfg.audio_conv_channels),
                nn.ReLU(inplace=True),
            ]
            in_ch = self.cfg.audio_conv_channels
             
                
        self.audio_encoder = nn.Sequential(*convs, nn.AdaptiveAvgPool1d(1))
        audio_dim = self.cfg.audio_conv_channels


        # -------- Text Encoder --------
        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.embed_dim,
                                      padding_idx=self.cfg.pad_token_id)
        self.lstm = nn.LSTM(
            self.cfg.embed_dim,
            self.cfg.lstm_hidden,
            num_layers=self.cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # -------- Regression Head --------
        self.combined_dim = audio_dim + 2 * self.cfg.lstm_hidden
        self.head = self.build_head(self.combined_dim, use_sigmoid=True)
        
    """   def build_head(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, self.cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden, self.cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.head_hidden, 1),
            nn.Sigmoid() ,
        ) """
        
    def build_head(self, in_dim, use_sigmoid: bool = False):
        layers = [
            nn.Linear(in_dim, self.cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden, self.cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.head_hidden, 1),
        ]

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)


    def forward(self, audio, audio_len, text, text_attention_mask, labels=None, score=None, **kwargs):
        
        
        # Audio encoder
        audio_enc = self.audio_encoder(audio).squeeze(-1)

        # -------- Text encoder  -------- 
        emb = self.embedding(text)
        lengths = text_attention_mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        text_enc, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, L, 2*H]
        text_enc = masked_mean_pooling(text_enc, text_attention_mask)


        # -------- Combine -------- 
        combined = torch.cat([audio_enc, text_enc], dim=1)
        
        logits = self.head(combined).squeeze(-1)
        
        # -------- Loss --------
        target = labels if labels is not None else score
        loss = None
        if target is not None:
            loss = nn.MSELoss()(logits, target)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )