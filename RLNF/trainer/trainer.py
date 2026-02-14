from typing import Dict
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE

import wandb
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from RLNF.utils.rollout import collect_batch
from ..dataloaders.reward_dataset import RewardDataCollator
from ..Rewards.reward_processor import RewardModelProcessor
from ..Rewards.reward_model import RewardModel
from ..optimizer.optimizer import DistillationOptimizer
from nemo.collections.asr.metrics.wer import word_error_rate


class RLNFTrainerDistill:
    """
    Trainer pour distillation ASR avec CTC.
    Compatible Single & Multi-GPU (DDP).
    """

    def __init__(
        self,
        student_model: EncDecCTCModel | EncDecCTCModelBPE,
        teacher_model: EncDecCTCModel | EncDecCTCModelBPE,
        dataset,
        processor: RewardModelProcessor,
        device: torch.device,
        wandb_logging: bool = True,
        wandb_project: str = "Bambara-RLNF",
        run_name: str = "rlnf-distill",
        batch_size: int = 16,
        epochs: int = 3,
        lr: float = 1e-5,
        val_every: int = 200,
        num_workers: int = 2,
        amp: bool = False,
        save_dir: str = "checkpoints",
        save_best_by: str = "val/loss",
        save_best_mode: str = "min",
        best_external_wer: float = float("inf"),
        resume_from_checkpoint: str | None = None,
    ):
        # ================= DDP =================
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main = self.rank == 0

        self.device = device
        self.processor = processor
        self.teacher_model = teacher_model.to(device)
        self.epochs = epochs
        self.val_every = val_every
        self.current_epoch = 0
        self.batch_size = batch_size
        self.resume_from_checkpoint = resume_from_checkpoint
        self.global_step = 0

        # ================= SAVE =================
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_best_by = save_best_by
        self.save_best_mode = save_best_mode
        self.best_val = float("inf") if save_best_mode == "min" else -float("inf")
        if best_external_wer != float("inf"): self.best_wer = best_external_wer

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
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            dataset["test"],
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

        # ================= DISTILLATION =================
        self.optimizer = DistillationOptimizer(
            student=student_model,
            teacher=self.teacher_model,
            lr=lr,
            device=device,
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
                    student = self.optimizer.student.module if self.is_distributed else self.optimizer.student

                    # === Collect batch ===
                    batch_dict = collect_batch(
                        batch=batch,
                        asr_model=student,
                        reward_model=self.teacher_model,  # teacher used to provide targets
                        processor=self.processor,
                        device=self.device,
                        use_lm=False,
                        beam_size=4
                    )

                    # === Distillation update ===
                    stats = self.optimizer.update(batch_dict)

                    if self.is_main:
                        loss = stats["distill_loss"]
                        #if loss < self.best_val:
                        #    self.best_val = loss
                        #    self.save_student_best(global_step, loss)

                        if self._use_wandb:
                            wandb.log({f"train/{k}": v for k, v in stats.items()}, step=global_step)
                        if self.tb_writer is not None:
                            for k, v in stats.items():
                                self.tb_writer.add_scalar(f"train/{k}", v, global_step)

                        pbar.set_postfix({"distill_loss": f"{loss:.4f}"})

                    global_step += 1

                    # Validation
                    if self.val_every > 0 and global_step % self.val_every == 0:
                        self.validate(global_step)
                        self.save_checkpoint(global_step)

                # End of epoch validation
                self.validate(global_step, end_of_epoch=True)

        finally:
            if self.is_main:
                self.save_final()
                if self._use_wandb:
                    wandb.finish()
                if self.tb_writer is not None:
                    self.tb_writer.close()

    # =====================================================
    # VALIDATION
    # =====================================================
    def validate(self, step: int, end_of_epoch: bool = False):
        student = self.optimizer.student.module if self.is_distributed else self.optimizer.student
        student.eval()

        losses = []
        wers = []
        cers = []

        with torch.no_grad():
            pbar_val = tqdm(self.val_loader, leave=False, disable=not self.is_main, desc=f"Validation at step {step}")
            for batch in pbar_val:
                val_dict = collect_batch(
                    batch=batch,
                    asr_model=student,
                    reward_model=self.teacher_model,
                    processor=self.processor,
                    device=self.device,
                    use_lm=False,
                    beam_size=4
                )

                stats_val = self.optimizer.update(val_dict)
                losses.append(stats_val["distill_loss"])
                wers.append(word_error_rate(val_dict["greedy_text"], batch["text"]))
                cers.append(word_error_rate(val_dict["greedy_text"], batch["text"], use_cer=True))


        mean_loss = sum(losses) / len(losses)
        mean_wer = sum(wers) / len(wers)
        mean_cer = sum(cers) / len(cers)

        if self.is_main:
            if self._use_wandb:
                wandb.log({"val/distill_loss": mean_loss,
                           "val/wer" : mean_wer ,
                           "val/cer" : mean_cer}, step=step)

            improved = self.save_best_mode == "min" and mean_loss < self.best_val
            improved_wer = self.save_best_mode == "min" and mean_wer < self.best_wer
            
            if improved_wer:
                self.best_wer = mean_wer
                self.save_student_best_wer(step, mean_wer)

            if improved:
                self.best_val = mean_loss
                self.save_student_best(step, mean_loss)

        student.train()

    # =====================================================
    # SAVE / LOAD
    # =====================================================
    def save_student_best(self, step: int, loss: float):
        student = self.optimizer.student.module if self.is_distributed else self.optimizer.student
        path = os.path.join(self.save_dir, f"best_student_step{step}_loss{loss:.4f}.nemo")
        student.save_to(path)

    def save_student_best_wer(self, step: int, wer: float):
        student = self.optimizer.student.module if self.is_distributed else self.optimizer.student
        path = os.path.join(self.save_dir, f"best_student_step{step}_wer{wer:.4f}.nemo")
        student.save_to(path)

    def save_final(self):
        student = self.optimizer.student.module if self.is_distributed else self.optimizer.student
        student.save_to(os.path.join(self.save_dir, "student_final.nemo"))

    def save_checkpoint(self, step: int, name="last"):
        if not self.is_main:
            return
        student = self.optimizer.student.module if self.is_distributed else self.optimizer.student
        ckpt = {
            "student": student.state_dict(),
            "epoch": self.current_epoch,
            "global_step": step,
            "best_val": self.best_val,
        }
        path = os.path.join(self.save_dir, f"checkpoint_step{step}.pt")
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank} if self.is_distributed else self.device
        ckpt = torch.load(path, map_location=map_location)
        student = self.optimizer.student.module if self.is_distributed else self.optimizer.student
        student.load_state_dict(ckpt["student"])
        self.current_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val = ckpt.get("best_val", self.best_val)
