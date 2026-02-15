from typing import Dict
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE

import wandb
import datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from nemo.collections.asr.metrics.wer import word_error_rate

from ..utils.rollout import collect_batch, _wer_cer, decode_batch

from ..dataloaders.reward_dataset import RewardDataCollator
from ..Rewards.reward_processor import RewardModelProcessor
from ..Rewards.reward_model import RewardModel
from ..optimizer.optimizer import SCSTOptimizer


class RLNFTrainerSCST:
    """
    RLNF Trainer for SCST (Self-Critical Sequence Training) — Single & Multi-GPU (DDP) compatible.
    """

    def __init__(
        self,
        asr_model: EncDecCTCModel | EncDecCTCModelBPE,
        reward_model: RewardModel,
        dataset: datasets,
        processor: RewardModelProcessor,
        device: torch.device,
        wandb_logging: bool = True,
        wandb_project: str = "Bambara-RLNF",
        run_name: str = "rlnf-scst",
        batch_size: int = 16,
        epochs: int = 3,
        actor_lr: float = 1e-5,
        val_every: int = 200,
        alpha : float = 1.0,
        beta : float = 0.2,
        gamma : float = 0.1,
        num_workers: int = 2,
        ent_coeff : float = 0.01,
        p_lr : float = 1e-3,
        pin_memory: bool = True,
        only : bool = False,
        regularize : bool = False,
        amp: bool = False,
        save_dir: str = "checkpoints",
        save_best_by: str = "val/wer",
        save_best_mode: str = "min",
        baseline : str = "max",
        use_lm: bool = True,
        trainable : bool = False,
        beam_size: int = 4,
        resume_from_checkpoint: str | None = None,
        best_external_wer: float = float("inf"),  # Nouveau paramètre
    ):
        # ================= DDP =================
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main = self.rank == 0

        self.device = device
        self.processor = processor
        self.reward_model = reward_model
        self.epochs = epochs
        self.val_every = val_every
        self.current_epoch = 0
        self.batch_size = batch_size
        self.use_lm = use_lm
        self.beam_size = beam_size
        self.resume_from_checkpoint = resume_from_checkpoint
        self.global_step = 0

        # ================= SAVE =================
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_best_by = save_best_by
        self.save_best_mode = save_best_mode
        self.best_val = float("inf") if save_best_mode == "min" else -float("inf")
        self.best_scst_loss = float("inf")

        # Initialisation avec WER externe
        if best_external_wer != float("inf"):
            self.best_val = best_external_wer

        # ================= LOGGING =================
        self.tb_writer = SummaryWriter(f"tb_logs/{run_name}") if self.is_main else None
        self._use_wandb = wandb_logging and self.is_main
        if self._use_wandb:
            wandb.init(project=wandb_project, name=run_name)

        # ================= DATA =================
        collate_fn = RewardDataCollator(processor=processor, augment=False)
        train_sampler = DistributedSampler(dataset["train"]) if self.is_distributed else None
        val_sampler = DistributedSampler(dataset["test"], shuffle=False) if self.is_distributed else None

        self.train_loader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.val_loader = DataLoader(
            dataset["test"],
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # ================= SCST =================
        self.scst = SCSTOptimizer(
            actor=asr_model,
            lr=actor_lr,
            p_lr=p_lr,
            trainable=trainable,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            baseline=baseline,
            ent_coeff=ent_coeff,
            regularize=regularize,
            device=device,
            only=only,
            amp=amp
        )

    # =====================================================
    # TRAIN
    # =====================================================
    def train(self):
        if self.resume_from_checkpoint is not None:
            self.load_checkpoint(self.resume_from_checkpoint)

        global_step = self.global_step

        try:
            for epoch in range(self.current_epoch, self.epochs):
                self.current_epoch = epoch
                if self.is_distributed:
                    self.train_loader.sampler.set_epoch(epoch)

                pbar = tqdm(
                    self.train_loader,
                    disable=not self.is_main,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                )

                for batch in pbar:
                    actor = self.scst.actor.module if self.is_distributed else self.scst.actor

                    # === Collect batch ===
                    batch_dict = collect_batch(
                        batch=batch,
                        asr_model=actor,
                        reward_model=self.reward_model,
                        processor=self.processor,
                        device=self.device,
                        use_lm=self.use_lm,
                        beam_size=self.beam_size
                    )

                    # === Reward sync across GPUs ===
                    if self.is_distributed:
                        batch_dict["reward"] = batch_dict["reward"].to(self.device)
                        torch.distributed.all_reduce(batch_dict["reward"], op=torch.distributed.ReduceOp.SUM)
                        batch_dict["reward"] /= self.world_size

                    # === SCST update ===
                    stats = self.scst.update(batch_dict)

                    if self.is_main:
                        scst_loss = stats["scst_loss"]
                        if abs(scst_loss) < abs(self.best_scst_loss):
                            self.best_scst_loss = abs(scst_loss)
                            self.save_actor_by_loss(global_step, scst_loss)

                        # Logging
                        if self._use_wandb and wandb.run is not None:
                            wandb.log({f"train/{k}": v for k, v in stats.items()}, step=global_step)
                        for k, v in stats.items():
                            self.tb_writer.add_scalar(f"train/{k}", v, global_step)

                        pbar.set_postfix({
                            "scst_loss": f"{scst_loss:.4f}",
                            "reward_mean": f"{stats['reward_mean']:.4f}"
                        })

                    global_step += 1

                    # Validation
                    do_val = self.val_every > 0 and global_step % self.val_every == 0
                    if self.is_distributed:
                        val_tensor = torch.tensor(int(do_val), device=self.device)
                        dist.broadcast(val_tensor, src=0)
                        do_val = bool(val_tensor.item())
                    if do_val:
                        self.validate(global_step)
                        self.save_checkpoint(global_step)

                # End of epoch validation
                self.validate(global_step, end_of_epoch=True)

        finally:
            if self.is_main:
                self.save_final()
                if self._use_wandb:
                    wandb.finish()
                self.tb_writer.close()

    # =====================================================
    # VALIDATION
    # =====================================================
    def validate(self, step: int, end_of_epoch: bool = False):

        actor = self.scst.actor.module if self.is_distributed else self.scst.actor

        if actor.cfg.decoding.strategy == "pyctcdecode" :
            decoding_cfg = actor.cfg.decoding
            decoding_cfg.strategy = "greedy_batch"
            decoding_cfg.beam.return_best_hypothesis = True

            actor.change_decoding_strategy(decoding_cfg)

        actor.eval()

        wers, cers, rewards = [], [], []

        with torch.no_grad():

            pbar_val = tqdm(self.val_loader, leave=False, disable=not self.is_main, desc=f"Validation at step {step}")
            for batch in pbar_val:
               
               audio = batch["audio"].to(self.device)
               audio_len = batch["audio_len"].to(self.device)

               
               out = actor(processed_signal = audio, processed_signal_length = audio_len)
               logits, enc_len = out[0], out[1]

               log_probs = logits.log_softmax(dim=-1)

               preds = decode_batch(
                   log_probs=log_probs,
                   enc_len=enc_len,
                   asr_model=actor,
                   use_lm=False
               )

               refs = self.processor.tokenizer.batch_decode(
                   batch["text"],
                   skip_special_tokens=True
               )

               re = self.processor.tokenizer.batch_encode_plus(preds, return_attention_mask=True, padding=True, return_tensors="pt")

               reward_model_input = {
                   "audio": audio,
                   "audio_len": audio_len,
                   "text": re["input_ids"],
                   "text_attention_mask": re["attention_mask"],
                   }
               
               reward_model_input = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in reward_model_input.items()}
               
               reward = self.reward_model(**reward_model_input)
               
               wers.append(word_error_rate(preds, refs))
               cers.append(word_error_rate(preds, refs, use_cer=True))

            # Rewards mean
               rewards.append(reward)

                #if self._use_wandb:
                #    wandb.log({
                #        "val/wer": sum(wers)/len(wers),
                #        "val/cer": sum(cers)/len(cers),
                #        "val/reward": sum(rewards)/len(rewards),
                #        "step": step,
                #        "reward" : sum(rewards)/len(rewards)
                #    })


            # =======================
            # Aggregate metrics
            # =======================
            mean_wer = sum(wers) / len(wers)
            mean_cer = sum(cers) / len(cers)
            mean_reward = sum(rewards) / len(rewards)

            # ---- WandB logging ----
            if self._use_wandb:

                wandb.log({
                    "val/wer": mean_wer,
                    "val/cer": mean_cer,
                    "val/reward": mean_reward,
                    "val/best_wer": self.best_val,
                }, step=step)

            # =======================
            # Save if WER improved
            # =======================
            improved = (
                (self.save_best_mode == "min" and mean_wer < self.best_val)
                or
                (self.save_best_mode == "max" and mean_wer > self.best_val)
            )

            if improved:
                if self.is_main:
                    old = self.best_val
                    self.best_val = mean_wer
                    
                    # Save checkpoint + actor
                    self.save_best(step)

                    if self._use_wandb:
                        wandb.log({
                            "val/wer_improved": 1,
                            "val/wer_delta": old - mean_wer,
                        }, step=step)
            else:
                if self.is_main and self._use_wandb:
                    wandb.log({
                        "val/wer_improved": 0,
                    }, step=step)


        actor.train()

    # =====================================================
    # SAVE / LOAD
    # =====================================================
    def save_best(self, step: int):
        actor = self.scst.actor.module if self.is_distributed else self.scst.actor
        path = os.path.join(self.save_dir, f"best_actor_step{step}.nemo")
        try:
            actor.save_to(path)
            print(f"[save_best] actor saved -> {path}")
        except Exception as e:
            torch.save(actor.state_dict(), path.replace(".nemo", ".pt"))

    def save_final(self):
        actor = self.scst.actor.module if self.is_distributed else self.scst.actor
        actor.save_to("actor_final.nemo")

    def save_checkpoint(self, step: int, name="last"):
        if not self.is_main:
            return
        actor = self.scst.actor.module if self.is_distributed else self.scst.actor
        ckpt = {
            "actor": actor.state_dict(),
            "epoch": self.current_epoch,
            "global_step": step,
            "best_val": self.best_val,
            "best_scst_loss": self.best_scst_loss,
        }
        if self.scst.trainable:
            ckpt.update({
                "alpha": self.scst.alpha.detach().cpu(),
                "beta": self.scst.beta.detach().cpu(),
                "gamma": self.scst.gamma.detach().cpu(),
                "opt_coeff": self.scst.opt_coeff.state_dict() if self.scst.opt_coeff else None,
            })
        path = os.path.join(self.save_dir, f"checkpoint_step{step}.pt")
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):

        map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank} if self.is_distributed else self.device
        ckpt = torch.load(path, map_location=map_location)

        actor = self.scst.actor.module if self.is_distributed else self.scst.actor
        actor.load_state_dict(ckpt["actor"])

        self.current_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val = ckpt.get("best_val", self.best_val)
        self.best_scst_loss = ckpt.get("best_scst_loss", self.best_scst_loss)

        if self.scst.trainable and "alpha" in ckpt:

            self.scst.alpha.data = ckpt["alpha"].to(self.device)
            self.scst.beta.data = ckpt["beta"].to(self.device)
            self.scst.gamma.data = ckpt["gamma"].to(self.device)

            if self.scst.opt_coeff and ckpt.get("opt_coeff"):

                self.scst.opt_coeff.load_state_dict(ckpt["opt_coeff"])

        if self.is_main:
            print(f"[checkpoint] resumed from {path} | epoch={self.current_epoch}, step={self.global_step}")

    def save_actor_by_loss(self, step: int, scst_loss: float):
        actor = self.scst.actor.module if self.is_distributed else self.scst.actor
        path = os.path.join(self.save_dir, f"best_actor_by_loss_step{step}_loss{scst_loss:.4f}.nemo")
        try:
            actor.save_to(path)
        except Exception as e:
            torch.save(actor.state_dict(), path.replace(".nemo", ".pt"))
    
