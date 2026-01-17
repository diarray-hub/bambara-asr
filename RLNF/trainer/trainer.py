from typing import Dict
import os
import contextlib

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE

import wandb
import datasets

from ..dataloaders.reward_dataset import RewardDataCollator
from ..Rewards.reward_processor import RewardModelProcessor
from ..utils.rollout import collect_batch, _mean_mean, _same_num_hypotheses
from ..optimizer.optimizer import PPOOptimizer
from ..PPO.critic_network import CriticModel
from ..Rewards.reward_model import RewardModel

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class RLNFTrainer:
    """
    RLNF Trainer — Single GPU & Multi-GPU (DDP) SAFE
    """

    def __init__(
        self,
        asr_model: EncDecCTCModel | EncDecCTCModelBPE,
        reward_model: RewardModel,
        critic_model: CriticModel,
        dataset: datasets,
        processor: RewardModelProcessor,
        device: torch.device,
        wandb_logging: bool = True,
        wandb_project: str = "Bambara-RLNF",
        run_name: str = "rlnf-exp",
        batch_size: int = 16,
        epochs: int = 3,
        K_updates: int = 4,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-4,
        clip_eps: float = 0.2,
        val_every: int = 200,
        num_workers: int = 2,
        pin_memory: bool = True,
        amp: bool = False,
        save_dir: str = "checkpoints",
        save_best_by: str = "val/wer",
        save_best_mode: str = "min",
        use_lm : bool = True,
        beam_size : int = 4,
        resume_from_checkpoint: str | None = None,
        
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
        
        self.best_actor_loss = float("inf")


        # ================= LOGGING =================
        self.tb_writer = SummaryWriter(f"tb_logs/{run_name}") if self.is_main else None
        self._use_wandb = wandb_logging and self.is_main

        if self._use_wandb:
            wandb.init(
                project=wandb_project,
                name=run_name,
            )

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

        # ================= PPO =================
        self.ppo = PPOOptimizer(
            actor=asr_model,
            critic=critic_model,
            clip_eps=clip_eps,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            K_updates=K_updates,
            device=device,
            amp=amp,
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
                    actor = self.ppo.actor.module if self.is_distributed else self.ppo.actor
                    critic = self.ppo.critic.module if self.is_distributed else self.ppo.critic

                    batch_dict = collect_batch(
                        batch=batch,
                        asr_model=actor,
                        reward_model=self.reward_model,
                        critic=critic,
                        processor=self.processor,
                        device=self.device,
                        use_lm=self.use_lm,
                        beam_size=self.beam_size
                    )

                    # ===== reward sync =====
                    if self.is_distributed:
                       
                        batch_dict["reward"] = batch_dict["reward"].to(self.device)
                        dist.all_reduce(batch_dict["reward"], op=dist.ReduceOp.SUM)
                        batch_dict["reward"] /= self.world_size


                    stats = self.ppo.update(batch_dict)

                    if self.is_main:
                        
                        actor_loss = stats["actor_loss"]
                        
                        if actor_loss < self.best_actor_loss :
                            self.best_actor_loss = actor_loss
                            self.save_actor_by_loss(global_step, actor_loss)
                            
                            
                        
                        if self._use_wandb and wandb.run is not None:
                            wandb.log({f"train/{k}": v for k, v in stats.items()}, step=global_step)

                        for k, v in stats.items():
                            self.tb_writer.add_scalar(f"train/{k}", v, global_step)
                            
                            
                        pbar.set_postfix_str(f"""actor_loss :{stats['actor_loss']:.3f}
                            critic_loss : {stats['critic_loss']:.3f},
                            V: {stats['mean_value']:.3f},
                            adv : {stats['adv_mean']:.3f},
                            clip%: {100*stats['frac_clipped']:.1f},
                            reward:{stats['reward_mean']:.3f},
                            ratio_mean:{stats['ratio_mean']:.3f},
                            frac_clipped:{stats['frac_clipped']:.3f}""")

                        pbar.set_postfix({
                            "actor_loss": f"{stats['actor_loss']:.3f}",
                            "critic_loss": f"{stats['critic_loss']:.3f}",
                            "V": f"{stats['mean_value']:.3f}",
                            "adv": f"{stats['adv_mean']:.3f}",
                            "clip%": f"{100*stats['frac_clipped']:.1f}",
                            "reward" : f"{stats['reward_mean']:.3f}",
                            "ratio_mean": f"{stats['ratio_mean']:.3f}",
                            "frac_clipped": f"{stats['frac_clipped']:.3f}"
                        
                    })

                    global_step += 1
                    
                    if self.is_distributed:
                        #dist.barrier()
                        step_tensor = torch.tensor(global_step, device=self.device)
                        dist.broadcast(step_tensor, src=0)
                        global_step = step_tensor.item()
                        
                    do_val = self.val_every > 0 and global_step % self.val_every == 0
                    if self.is_distributed:
                        val_tensor = torch.tensor(int(do_val), device=self.device)
                        dist.broadcast(val_tensor, src=0)
                        do_val = bool(val_tensor.item())

                    # ---- validation pour tous les ranks (rank 0 log) ----
                    if do_val:
                        self.validate(global_step)
                        self.save_checkpoint(global_step)
                        

                    #if self.val_every > 0 and global_step % self.val_every == 0:
                    #    self.validate(global_step)
                    #    self.save_checkpoint(global_step)
                        
                    #if self.is_distributed:
                    #    dist.barrier()

                #if self.is_distributed:
                #    dist.barrier()

                self.validate(global_step, end_of_epoch=True)

                #if self.is_distributed:
                #    dist.barrier()

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
        actor = self.ppo.actor.module if self.is_distributed else self.ppo.actor
        critic = self.ppo.critic.module if self.is_distributed else self.ppo.critic

        actor.eval()
        critic.eval()

        wers, cers, rewards, values = [], [], [], []

        with torch.no_grad():
            pbar_val = tqdm(
                self.val_loader,
                leave=False,
                disable=not self.is_main,
                desc=f"Validation at step {step}"
            )
            
            for batch in pbar_val:
               

                actor.spec_augmentation = None
                actor.sample_rate = 16000
                actor.preprocessor.featurizer.to(self.device)
                
                audio = [aud for aud in batch["_audio"]]
                hyps = actor.transcribe(audio, batch_size=8)
                
                #hyp_texts = [[h.text for h in hyp] for hyp in hyps] #[h.text for h in hyps]
                hyp_texts_flat = [h for hyps_per_audio in hyps for h in hyps_per_audio]

                # Dupliquer refs pour correspondre aux hypothèses
                refs_flat = []
                for i, hyps_per_audio in enumerate(hyps):
                    n_hyps = len(hyps_per_audio)
                    refs_flat.extend([refs[i]] * n_hyps)

                refs = self.processor.tokenizer.batch_decode(
                    batch["text"], skip_special_tokens=True
                )
                
                print(refs_flat)
                print(hyp_texts_flat)

                # metrics per batch
                batch_wer = word_error_rate(hyp_texts_flat, refs_flat)
                batch_cer = word_error_rate(hyp_texts_flat, refs_flat, use_cer=True)
                wers.append(batch_wer)
                cers.append(batch_cer)

                val_dict = collect_batch(
                    batch=batch,
                    asr_model=actor,
                    reward_model=self.reward_model,
                    critic=critic,
                    processor=self.processor,
                    device=self.device,
                    use_lm=self.use_lm,
                    beam_size=self.beam_size
                )

                batch_reward = ( val_dict["reward"].mean() 
                                if _same_num_hypotheses(indexes=val_dict["indexes"]) 
                                else _mean_mean(val_dict["reward"], indexes=val_dict["indexes"])[0]
                            )
                
                batch_value = val_dict["values"].mean()
                
                rewards.append(batch_reward)
                values.append(batch_value)

            # affichage live par batch
            if self.is_main:
                
                string = f"""WER :{sum(wers)/len(wers):.4f}, 
                        CER: {sum(cers)/len(cers):.4f},
                        Reward: {sum(rewards)/len(rewards):.3f},
                        Value: {sum(values)/len(values):.3f}"""
                        
                pbar_val.set_postfix_str(string)
                
                pbar_val.set_postfix({
                    "WER": f"{sum(wers)/len(wers):.4f}",
                    "CER": f"{sum(cers)/len(cers):.4f}",
                    "Reward": f"{sum(rewards)/len(rewards):.3f}",
                    "Value": f"{sum(values)/len(values):.3f}",
                })
                
                

            # ===== reduce metrics across GPUs =====
            t = torch.tensor([
                sum(wers),
                sum(cers),
                sum([r.item() for r in rewards]),
                sum([v.item() for v in values]),
                len(wers)
            ], device=self.device)

            if self.is_distributed:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

            total = t[-1].item()
            to_log = {
                "val/wer": (t[0]/total).item(),
                "val/cer": (t[1]/total).item(),
                "val/reward": (t[2]/total).item(),
                "val/value": (t[3]/total).item(),
            }

            if end_of_epoch:
                to_log["epoch"] = self.current_epoch

            cur = to_log[self.save_best_by]
            if (
                (self.save_best_mode == "min" and cur < self.best_val)
                or (self.save_best_mode == "max" and cur > self.best_val)
            ):
                self.best_val = cur
                self.save_best(step)

            if self.is_main:
                if self._use_wandb:
                    wandb.log(to_log, step=step)
                for k, v in to_log.items():
                    self.tb_writer.add_scalar(k, v, step)

        actor.train()
        critic.train()


    # =====================================================
    # SAVE
    # =====================================================
    def save_best(self, step: int):
        
        
        actor = self.ppo.actor.module if hasattr(self.ppo.actor, "module") else self.ppo.actor
        critic = self.ppo.critic.module if hasattr(self.ppo.critic, "module") else self.ppo.critic

        actor_path = os.path.join(self.save_dir, f"best_step{step}_actor.nemo")
        critic_dir = os.path.join(self.save_dir, f"best_step{step}_critic")
        fallback_actor = os.path.join(self.save_dir, f"best_step{step}_actor_state.pt")
        fallback_critic = os.path.join(self.save_dir, f"best_step{step}_critic_state.pt")

       
        try:
            actor.save_to(actor_path)
            print(f"[save_best] actor saved -> {actor_path}")
        except Exception as e:
            print(f"[save_best] Warning: actor.save_to failed: {e}. Attempting state_dict fallback.")
            try:
                torch.save(actor.state_dict(), fallback_actor)
                print(f"[save_best] actor.state_dict saved -> {fallback_actor}")
            except Exception as e2:
                print(f"[save_best] ERROR: Could not save actor state_dict: {e2}")

      
        try:
            critic.save_pretrained(critic_dir)
            print(f"[save_best] critic saved -> {critic_dir}")
        except Exception as e:
            print(f"[save_best] Warning: critic.save_pretrained failed: {e}. Attempting state_dict fallback.")
            try:
                torch.save(critic.state_dict(), fallback_critic)
                print(f"[save_best] critic.state_dict saved -> {fallback_critic}")
            except Exception as e2:
                print(f"[save_best] ERROR: Could not save critic state_dict: {e2}")

    def save_final(self):
        actor = self.ppo.actor.module if self.is_distributed else self.ppo.actor
        critic = self.ppo.critic.module if self.is_distributed else self.ppo.critic

        actor.save_to("actor_final.nemo")
        critic.save_pretrained("critic_final")
        
    def save_checkpoint(self, step: int, name = "last"):
        if not self.is_main:
            return

        actor = self.ppo.actor.module if self.is_distributed else self.ppo.actor
        critic = self.ppo.critic.module if self.is_distributed else self.ppo.critic

        ckpt = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_optim": self.ppo.opt_actor.state_dict(),
            "critic_optim": self.ppo.opt_critic.state_dict(),
            "epoch": self.current_epoch,
            "global_step": step,
            "best_val": self.best_val,
            "best_actor_loss": self.best_actor_loss, 
        }

        path = os.path.join(self.save_dir, f"checkpoint_step{step}.pt")
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        
        map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank} if self.is_distributed else self.device
        ckpt = torch.load(path, map_location=map_location)

        actor = self.ppo.actor.module if self.is_distributed else self.ppo.actor
        critic = self.ppo.critic.module if self.is_distributed else self.ppo.critic

        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])

        self.ppo.opt_actor.load_state_dict(ckpt["actor_optim"])
        self.ppo.opt_critic.load_state_dict(ckpt["critic_optim"])

        self.current_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val = ckpt.get("best_val", self.best_val)
        self.best_actor_loss = ckpt.get("best_actor_loss", self.best_actor_loss)


        if self.is_main:
            
            print(
                f"[checkpoint] resumed from {path} | "
                f"epoch={self.current_epoch}, step={self.global_step}"
            )
            
    def save_actor_by_loss(self, step: int, actor_loss: float):
        
        actor = self.ppo.actor.module if self.is_distributed else self.ppo.actor

        path = os.path.join(
            self.save_dir,
            f"best_actor_by_loss_step{step}_loss{actor_loss:.4f}.nemo"
        )

        try:
            actor.save_to(path)
            print(f"[save_actor_by_loss] actor saved -> {path}")
        except Exception as e:
            torch.save(actor.state_dict(), path.replace(".nemo", ".pt"))

