import json
import os
import numpy as np
import torchaudio
import torch
from transformers import (
    ProcessorMixin,
    T5Tokenizer,
)

from .reward_feature_extraction import RewardFeatureExtractor
from typing import Union, List

class RewardModelProcessor(ProcessorMixin):
    """
    Processor for RewardModel that handles audio and text preprocessing.
    """

    feature_extractor_class = "RewardFeatureExtractor"
    tokenizer_class = "T5Tokenizer"

    def __init__(self, feature_extractor: RewardFeatureExtractor, tokenizer: T5Tokenizer):
        super().__init__(feature_extractor, tokenizer)
        
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        self.asr_model = self.feature_extractor.model
        
        self.sample_rate = getattr(self.feature_extractor.model.preprocessor, "_sample_rate", 16_000)

        self.audio_fn = self._nemo_audio_pipeline

    def _load_waveform(self, wav_input: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        
        """
        Load a waveform from a file path, numpy array, or torch tensor.
        Always returns a mono float32 torch tensor.

        Args:
            wav_input (str | np.ndarray | torch.Tensor): Input audio data or file path.

        Returns:
            torch.Tensor: Waveform tensor (mono, float32).
        """
        
        if isinstance(wav_input, str):
            wav, sr = torchaudio.load(wav_input)
            if self.sample_rate is not None and sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        elif isinstance(wav_input, np.ndarray):
            wav = torch.from_numpy(wav_input)
        elif isinstance(wav_input, torch.Tensor):
            wav = wav_input
        else:
            raise TypeError(f"Unsupported type: {type(wav_input)}")

        # make mono if multi-channel
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        return wav.float()

    def _nemo_audio_pipeline(self, audios):
        
        """
        Return tensors [B, T] and their length.
        """
        
        if not isinstance(audios, (list, tuple)):
            audios = [audios]

        waveforms = []
        lengths = []
        for wav_input in audios:
            wav = self._load_waveform(wav_input)
            waveforms.append(wav)
            lengths.append(wav.shape[-1])

        padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        lengths = torch.tensor(lengths, dtype=torch.long)
        return padded, lengths

    def __call__(self, audios: list, texts: list):
        
        """
        Process a batch of audio files + text strings
        Returns a dict containing:
          - audio features
          - audio lengths
          - tokenized text (input_ids, attention_mask)
          - nemo_audio and nemo_audio_length : for nemo model
        """
        
        audio_feats, audio_len = self.audio_fn(audios)
        audio_batch, audio_batch_len = self.feature_extractor(audio_feats, audio_len)

        text_batch = self.tokenizer(texts, padding=True, return_tensors="pt", return_attention_mask=True)

        return {
            "audio": audio_batch,
            "audio_len": audio_batch_len,
            "text": text_batch["input_ids"],
            "text_attention_mask": text_batch["attention_mask"],
            "_audio": audio_feats,  # raw waveform batch
        }

    # ---------------------------
    # Custom serialization to avoid deepcopy of non-serializable model objects
    # ---------------------------
    def to_dict(self):
        """
        Return a small serializable dict describing this processor (no heavy objects).
        This is used by our custom save_pretrained to write a JSON config.
        """
        return {
            "feature_extractor_class": self.feature_extractor_class,
            "tokenizer_class": self.tokenizer_class,
            "sample_rate": self.sample_rate,
        }

    def save_pretrained(self, save_directory: str):
        """
        Save tokenizer and feature_extractor (which itself saves the .nemo file).
        Write a small JSON config for the processor.
        """
        os.makedirs(save_directory, exist_ok=True)

        self.tokenizer.save_pretrained(save_directory)

        try:
            self.feature_extractor.save_pretrained(save_directory)
        except Exception as e:
            
            print("Warning: feature_extractor.save_pretrained failed:", e)


        config = self.to_dict()
        with open(os.path.join(save_directory, "reward_processor_config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, save_directory: str, *args, **kwargs):
        """
        Reconstruct RewardModelProcessor from saved directory.
        Expects:
          - tokenizer saved via tokenizer.save_pretrained(save_directory)
          - feature extractor saved via RewardFeatureExtractor.save_pretrained(save_directory)
          - config in reward_processor_config.json
        """
        
        tokenizer = T5Tokenizer.from_pretrained(save_directory)

        fe = RewardFeatureExtractor.from_pretrained(save_directory)

        
    
        proc = cls(fe, tokenizer)
        
        cfg_path = os.path.join(save_directory, "reward_processor_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            proc.sample_rate = cfg.get("sample_rate", proc.sample_rate)
        return proc