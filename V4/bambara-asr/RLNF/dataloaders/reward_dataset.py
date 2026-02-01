import torch
import torchaudio
from ..Rewards.reward_processor import RewardModelProcessor


class RewardDataCollator:
    """
    Collate function for the RewardModel.
    Prepares batches of Mel-spectrograms + text and applies selected augmentations
    (noise, time/freq masking, SpecAugment, time stretch) if augment=True.
    """

    def __init__(
        self,
        processor: RewardModelProcessor,
        augment: bool = True,
        augmentations: list = None,
        freq_mask_param: int = 35,
        time_mask_param: int = 20,
        noise_level: float = 0.05,
        stretch_rate: float = 1.2,
        n_time_masks: int = 2,
        n_freq_masks : int = 2,
        snr : float = 20.0
    ):
        """
        Args:
            processor: HuggingFace processor (tokenizer + feature extractor)
            augment: apply augmentations during training
            augmentations: list of augmentation names to apply
                           e.g. ["noise", "time_mask", "freq_mask", "specaug"]
            freq_mask_param: width of frequency masking (SpecAugment)
            time_mask_param: width of time masking (SpecAugment)
            noise_level: std of Gaussian noise
            stretch_rate: time stretch factor (>1 = faster, <1 = slower)
        """
        self.processor = processor
        self.augment = augment
        self.noise_level = noise_level
        self.stretch_rate = stretch_rate
        self.snr = snr

        self.augmentations = augmentations or ["noise", "time_mask", "freq_mask", "specaug"]


        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.specaug = torchaudio.transforms.SpecAugment(
            time_mask_param=time_mask_param,
            freq_mask_param=freq_mask_param,
            n_time_masks=n_time_masks,
            n_freq_masks=n_freq_masks
        )
        self.time_stretch = torchaudio.transforms.TimeStretch()
        
        self.add_noise = torchaudio.transforms.AddNoise()

    # === Augmentations ===
    def apply_noise(self, mel):
        
        mel = mel.unsqueeze(0)
        noise = torch.randn_like(mel) * self.noise_level
        snr = torch.tensor([self.snr])
        return self.add_noise(mel, noise, snr).squeeze(0)
        
    def apply_time_mask(self, mel):
        return self.time_mask(mel.unsqueeze(0)).squeeze(0)

    def apply_freq_mask(self, mel):
        return self.freq_mask(mel.unsqueeze(0)).squeeze(0)

    def apply_specaug(self, mel):
        return self.specaug(mel.unsqueeze(0)).squeeze(0)


    # === Collate ===
    def __call__(self, batch: list) -> dict:
        """
        Converts a raw batch into model-ready tensors.
        Applies all selected augmentations if self.augment=True.
        """
        mels, texts, scores = [], [], []

        for x in batch:
            mel = torch.tensor(x["audio"]["array"], dtype=torch.float32)
            text = x["text"]

            score = x.get("score")
            if score is not None:
                score = score / 100.0
                
            # Original
            mels.append(mel)
            texts.append(text)
            scores.append(score)

            if self.augment:
                for aug_name in self.augmentations:
                    aug_fn = getattr(self, f"apply_{aug_name}", None)
                    if aug_fn is None:
                        raise ValueError(f"Unknown augmentation: {aug_name}")
                    mel_aug = aug_fn(mel)
                    mels.append(mel_aug)
                    texts.append(text)
                    scores.append(score)

        batch_dict = self.processor(
            mels,
            texts
        )

        if scores and all( s is not None for s in scores) :
            batch_dict["score"] = torch.tensor(scores, dtype=torch.float32)
            batch_dict["labels"] = batch_dict["score"]
            
        return batch_dict
