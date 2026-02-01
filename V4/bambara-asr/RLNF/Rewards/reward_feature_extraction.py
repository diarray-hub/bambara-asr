
import json
import os
from typing import Tuple
from torch import Tensor
import torch
from transformers import AutoFeatureExtractor
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModel

class RewardFeatureExtractor:
     def __init__(self, asr_model):
          
         self.model = asr_model

     def __call__(self, audios: Tensor, audios_lens: Tensor) -> Tuple[Tensor, Tensor]:
          
        """
        audios: (B, T)
        audios_lens: (B,)
        return: (features, features_lens)
        """
        
        device = audios.device

        try:
            self.model.preprocessor.featurizer.fb = (
                self.model.preprocessor.featurizer.fb.to(device)
            )
        except Exception:
            pass

        return self.model.preprocessor(input_signal=audios, length=audios_lens)
   
     @classmethod
     def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
          from nemo.collections.asr.models import EncDecCTCModel
          import os

          nemo_file = os.path.join(pretrained_model_name_or_path, "asr_model.nemo")

          if not os.path.isfile(nemo_file):
               raise FileNotFoundError(
                    f"Expected NeMo model at {nemo_file}, but it was not found."
               )

          asr_model = EncDecCTCModel.restore_from(nemo_file)
          return cls(asr_model)


     def save_pretrained(self, save_directory):
          
        os.makedirs(save_directory, exist_ok=True)
        nemo_file = os.path.join(save_directory, "asr_model.nemo")
        
        self.model.save_to(nemo_file)
        # Optionally write a tiny descriptor for the featurizer (sample rate etc.)
        try:
            cfg = {
                "sample_rate": getattr(self.model.preprocessor, "_sample_rate", None)
            }
            with open(os.path.join(save_directory, "feature_extractor_config.json"), "w") as f:
                json.dump(cfg, f)
        except Exception:
            pass

# register so transformers can find it (optional but consistent)
#AutoFeatureExtractor.register("RewardFeatureExtractor", RewardFeatureExtractor)

AutoFeatureExtractor.register("RewardFeatureExtractor", RewardFeatureExtractor)

        
        