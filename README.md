# bambara-asr

**Project Overview**

This repository contains a collection of tools, experiments, and utilities for building and fine-tuning automatic speech recognition (ASR) systems—particularly for low-resource languages like Bambara—and for experimenting with reinforcement learning from human feedback (RLHF) techniques in ASR training.

**License**

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Getting Started

1. **Install requirements** (for RLNF toolkit & core dependencies):

Before you try to experiment with Reinforcement Learning from Nouhoum Feedback

   ```bash
   git clone https://github.com/diarray-hub/bambara-asr.git --branch=rlnf-v3-gpu
  cd bambara-asr
   pip install .
   ```

OR 
```bash
pip install git+https://github.com/diarray-hub/bambara-asr.git@rlnf-v3-gpu
```

## How to use this package:
**want to train a reward model : coming soon.....**

**want to test the reward model**

```python
import torch
from RLNF.Rewards.reward_model import RewardModel
from RLNF.Rewards.reward_processor import RewardModelProcessor
from RLNF.Rewards.reward_feature_extraction import RewardFeatureExtractor
from transformers import T5Tokenizer
from nemo.collections.asr.models import EncDecCTCModel

audios = ["1.wav", "2.wav"]
texts = ["kelen", "fila."]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer : T5Tokenizer = T5Tokenizer.from_pretrained("RobotsMali/reward-model")
asr_model : EncDecCTCModel= EncDecCTCModel.from_pretrained("RobotsMali/stt-bm-quartznet15x5-V0")
feature_extractor : RewardFeatureExtractor = RewardFeatureExtractor(asr_model)

processor : RewardModelProcessor = RewardModelProcessor(feature_extractor, tokenizer)

model : RewardModel = RewardModel.from_pretrained("RobotsMali/reward-model")

model.eval()
model.to(device)
    
out = processor(audios=audios, texts=texts)    
out = {k: v.to(device) if torch.is_tensor(v) else v for k, v in out.items()}


with torch.no_grad() :
  preds = model(**out).logits
    
    
for i, (t, val) in enumerate(zip(texts, preds)):
  print(f"Audio : {audios[i]:<10} | Text: {t:<10} | Score: {val.item() * 100:.4f}")


```
**want to train a RLNF model : coming soon....**

coming soon......


**want to test the RLNF model**

coming soon......







