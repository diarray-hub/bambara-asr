---
language:
- bm
library_name: nemo
datasets:
- RobotsMali/an-be-kalan-bench
thumbnail: null
tags:
- automatic-speech-recognition
- speech
- audio
- Transducer
- TDT
- FastConformer
- Conformer
- pytorch
- Bambara
- NeMo
license: cc-by-4.0
base_model: RobotsMali/soloni-114m-tdt-ctc-v2
model-index:
- name: soloni-be-kalan-v0
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: An be kalan Children's Reading Benchmark
      type: RobotsMali/an-be-kalan-bench
      split: test
      args:
        language: bm
    metrics:
    - name: Test WER
      type: wer
      value: 22.0
    - name: Test CER
      type: cer
      value: 8.0
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
---

# Soloni Be Kalan (TDT-CTC 114M)

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--TDT-blue#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-114M-green#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-bm-orange#model-badge)](#datasets)

`soloni-be-kalan-v0` is a domain-specific fine-tuned version of [`RobotsMali/soloni-114m-tdt-ctc-v2`](https://huggingface.co/RobotsMali/soloni-114m-tdt-ctc-v2). This model was adapted specifically for Bambara educational materials and child speech applications. The model was fine-tuned using **NVIDIA NeMo** and supports **both TDT (Token-and-Duration Transducer) and CTC (Connectionist Temporal Classification) decoding**.

## **🚨 Important Note**

This model, along with its associated resources, is part of an **ongoing research effort** by the RobotsMali AI4D Lab. Users should be aware that:

*
**Early Childhood Performance Gap:** While this model significantly reduces the baseline error rate on early childhood speech (<10 years) from 56% down to 29% WER, physiological features unique to young children (unformed acoustic profiles, erratic speech rates) continue to present an out-of-domain challenge compared to older cohorts.


*
**Structural Dependencies:** The model performs exceptionally well on fluid, sequential storytelling text (achieving down to 7% WER) , but faces structural limitations on short, highly repetitive, sparse token arrays (e.g., *Kuloriw* or *Jate*) where language model prior biases dominate.



## NVIDIA NeMo: Training

To fine-tune or run inference with this model, you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend installing it alongside a compatible PyTorch environment.

```bash
pip install nemo-toolkit['asr']

```

## How to Use This Model

### Load Model with NeMo

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="RobotsMali/soloni-be-kalan-v0")

```

### NeMo 2.5.0 Compatibility

This checkpoint was created with NeMo 2.5.0. Loading it with newer NeMo versions (observed with 2.7.x) can fail because the strict decoding schema expects `key_phrase_items_list`; see [NVIDIA-NeMo/Speech#15658](https://github.com/NVIDIA-NeMo/Speech/issues/15658). This workaround was tested with Python 3.12:

```python
from pathlib import Path

from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf

model_name = "RobotsMali/soloni-be-kalan-v0"
cfg = ASRModel.from_pretrained(model_name, return_config=True)
OmegaConf.set_struct(cfg, False)

for decoder in ("greedy", "beam"):
    boosting_tree = OmegaConf.select(cfg, f"decoding.{decoder}.boosting_tree")
    if boosting_tree is not None:
        boosting_tree.key_phrase_items_list = None

config_path = Path("patched_config.yaml").resolve()
OmegaConf.save(cfg, config_path)
asr_model = ASRModel.from_pretrained(
    model_name=model_name,
    override_config_path=str(config_path),
    strict=False,
)
config_path.unlink()
```

### Transcribe Audio

```python
# Accepts 16 kHz mono-channel audio wav files (no need to resample manually if your audio isn't 16kHz, the preprocessor will do)
asr_model.transcribe(['sample_child_reading.wav'])

```

If you encounter a `RuntimeError: CUDA error: invalid argument` due to GPU compatibility with CUDA Graphs in the TDT decoder, disable it in your configuration before transcribing:

```python
decoding_cfg = asr_model.cfg.decoding
decoding_cfg.greedy.use_cuda_graph_decoder = False
asr_model.change_decoding_strategy(decoding_cfg=decoding_cfg)

```

## Model Architecture

This model utilizes the **Hybrid FastConformer-TDT-CTC** architecture with 114 million parameters. FastConformer optimizes the standard Conformer model with 8x depthwise-separable convolutional downsampling. It features two independent but jointly trained decoders: an auto-regressive TDT decoder (default branch) and a convolutional decoder optimized via CTC loss.

## Training & Fine-Tuning Configurations

Fine-tuning was executed under strict low-resource optimization constraints:

*
**Optimization Framework:** Regularized using an **Early Stopping mechanism** with a **15-epoch patience window** based on a validation set matching the benchmark distribution.

*
**Convergence Behavior:** Due to high acoustic data density acting as a natural regularizer, training safely concluded at **epoch 20**, successfully preventing the vocabulary collapse and lexical overfitting typical of small, pristine speech corpora.

*
**Augmentation Strategy:** This configuration explicitly omitted synthetic spectral masking (`SpecAugment=None`) , demonstrating that physical voice variance from natural human speakers acts as a superior regularizer than artificial noise injection in this specific domain.

## Dataset

The model was fine-tuned on the combined **Main + Duplicate** expanded subset (totaling **45.6 hours**) of the [RobotsMali/an-be-kalan-bench](https://huggingface.co/datasets/RobotsMali/an-be-kalan-bench) dataset.

*
**Main Split (1.6h):** Clean readings of 22 GAIFE project books recorded by 8 unique speakers.

*
**Duplicate Split (44h):** A highly dense, multi-speaker redundant corpus featuring natural human speech variations (pitch, accent, child speech dynamics) reading the identical source literature.

## Performance

Performance is disaggregated below across overall results, specific age cohorts, and distinctive book structures using Word Error Rate (WER) and Character Error Rate (CER).

### Overall Evaluation

| Model | Decoding Branch | WER (%) ↓ | CER (%) ↓ | Status |
| --- | --- | --- | --- | --- |
| **soloni-be-kalan** | CTC  | **22.0%** | **8.0%** | <br>*Newly deployed model in An be Kalan app* |
| *soloni-114m-tdt-ctc-v2 (Base)* | CTC | *42.0%* | *15.0%* | <br>*Pre-trained Baseline Reference* |

### Demographic Cohort Breakdown

| Age Cohort | Utterance Count | Baseline WER (%) | Fine-Tuned WER (%) | Key Insights |
| --- | --- | --- | --- | --- |
| **Early Childhood (<10 yrs)** | 93 | 56.0% | **29.0%** | Remains the single largest acoustic error cluster.|
| **Target Cohort (10-15 yrs)** | 527 |0.39 | **22.0%** | Majority representation; stable acoustic profiles.|

## License

This model is released under the **CC-BY-4.0** license.

