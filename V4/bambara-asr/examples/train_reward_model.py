from datasets import load_dataset

from RLNF.Rewards.reward_config import RewardConfig
from RLNF.dataloaders.reward_dataset import RewardDataCollator
from RLNF.Rewards.reward_feature_extraction import RewardFeatureExtractor
from RLNF.Rewards.reward_model import RewardModel
from RLNF.Rewards.reward_processor import RewardModelProcessor
from RLNF.utils.dot_dict import DotDict

from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, T5Tokenizer

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

ds = load_dataset("RobotsMali/transcription-scorer")
train_ds = ds["train"]
eval_ds = ds["test"]


def compute_metrics(eval_pred, tolerance=0.1):
    preds, labels = eval_pred
    preds = np.squeeze(preds)
    labels = np.squeeze(labels)

    mse = mean_squared_error(labels, preds)
    r2 = r2_score(labels, preds)
    pearson_corr, _ = pearsonr(labels, preds)

    correct = np.abs(preds - labels) <= tolerance
    accuracy = correct.mean()

    return {
        "mse": mse,
        "r2": r2,
        "pearson": pearson_corr,
        "accuracy": accuracy,
    }


path = "RobotsMali/reward-model"

#uncomment this part if you do not want to use our configuration(eg: you want to change some parameters.....)
""" tokenizer_path = "./tokenizer/tokenizer.model"
config = RewardConfig(tokenizer_path=tokenizer_path)
cfg = DotDict(config.pretrained_config) #make a dot dict
feature_extractor = RewardFeatureExtractor(#feature_size=cfg.n_mel, sampling_rate=cfg.sample_rate, 
                                            #n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                                           #chunk_length=cfg.chunk_length,dither=cfg.dither,
                                           #return_attention_mask=cfg.return_attention_mask,
                                           #padding_value=cfg.padding_value
                                           ) # uncomment if you want to change anything


#we use T5Tokenizer cause it handles sentencepice tokenizers.
tokenizer = T5Tokenizer(vocab_file = config.tokenizer_path, legacy=False)  


#we make our processor and saved it for next using.
processor = RewardModelProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
processor.save_pretrained("reward-model")
config.save_pretrained("reward-model") """


#comment the 2 lines below if you  uncomment lines above.
config = RewardConfig.from_pretrained(path)
processor = RewardModelProcessor.from_pretrained(path)

collator_train = RewardDataCollator(processor, augment=True) #we do data augmentation 
collator_eval = RewardDataCollator(processor, augment=False ) #here we don't

reward_model = RewardModel(config=config) #initialize our reward model.

#make your training args
training_args = TrainingArguments(...) 


#custom class to avoid data augmentation on eval dataset 
class TrainerWithEvalCollator(Trainer):
    def __init__(self, eval_collator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_collator_custom = eval_collator

    def get_eval_dataloader(self, eval_dataset=None):
        dl = super().get_eval_dataloader(eval_dataset)
        dl.collate_fn = self.eval_collator_custom
        return dl


early_stopping = EarlyStoppingCallback(early_stopping_patience=10) 


trainer = TrainerWithEvalCollator(
    model=reward_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator_train,  
    eval_collator=collator_eval,  
    compute_metrics=compute_metrics,
    #callbacks=[early_stopping],
)



trainer.train()
trainer.log_metrics("test", trainer.evaluate())
trainer.save_metrics("test", trainer.evaluate())

