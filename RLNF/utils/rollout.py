# rlnf/ppo/rollout.py
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from torch.nn.utils.rnn import pad_sequence
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from ..Rewards.reward_model import RewardModel
from ..PPO.critic_network import CriticModel
from ..Rewards.reward_processor import RewardModelProcessor
import torch.nn.functional as F
import importlib.resources as rsc
import RLNF.ressources


def _mean_mean(T: torch.Tensor, indexes : torch.Tensor, only = True) -> Tuple[torch.Tensor] :
    
    """
    Compute a 'mean of means' aggregation over grouped samples.

    This function is designed for settings where multiple trajectories
    (e.g. decoded hypotheses) belong to the same higher-level sample
    (e.g. one audio), and we want each higher-level sample to contribute
    equally, regardless of how many trajectories it has.

    Args:
        T:
            Tensor of shape [B], containing per-trajectory values.
            In PPO, this is typically the clipped surrogate objective
            min(ratio * adv, clipped_ratio * adv).

        indexes:
            Tensor of shape [B], where indexes[b] indicates the group
            (e.g. audio id) to which T[b] belongs.
            All entries with the same index are averaged together.

        only:
            If True:
                return (global_mean, global_std)
                where statistics are computed over group-level means.
            If False:
                return (group_means, global_mean, global_std)

    Returns:
        If only=True:
            mean_of_means:
                Scalar tensor, average of per-group means.
            std_of_means:
                Scalar tensor, standard deviation of per-group means
                (unbiased=False).

        If only=False:
            group_means:
                Tensor of shape [N_groups], mean value for each group.
            mean_of_means:
                Scalar tensor.
            std_of_means:
                Scalar tensor.
    """
    unique_ids = torch.unique(indexes)
    means = torch.empty_like(unique_ids, dtype=torch.float)
    
    for i, uid in enumerate(unique_ids) :
        
        means[i] = T[indexes == uid].mean()
        
    if only :
    
        return means.mean(), means.std(unbiased=False)
    
    return means, means.mean(), means.std(unbiased=False)

def _normalize_adv(adv: torch.Tensor, indexes : torch.Tensor, eps : float = 1e-8) -> torch.Tensor:
    
    """
        Normalize advantages independently for each group.

        This function performs per-group (e.g. per-audio) advantage normalization.
        All trajectories belonging to the same group are normalized using
        that group's mean and standard deviation.

        This is important in settings where each higher-level sample (audio)
        produces a variable number of trajectories (e.g. multiple decoded texts),
        and we want normalization to respect group boundaries.

        Args:
            adv:
                Tensor of shape [B], containing raw advantage values
                (e.g. A = R - V).

            indexes:
                Tensor of shape [B], where indexes[b] indicates the group
                (e.g. audio id) to which adv[b] belongs.

            eps:
                Small constant added to the denominator for numerical stability.

        Returns:
            A_norm:
                Tensor of shape [B], containing normalized advantages.
                For each group i:
                    mean(A_norm[indexes == i]) ≈ 0
                    std(A_norm[indexes == i]) ≈ 1
    """
             
    N = indexes.max().item() + 1
    
    A_norm = torch.empty_like(adv)
    
    for i in range(N) : 
        
        A_i = adv[indexes == i]
        
        mean = A_i.mean()
        std = A_i.std(unbiased=False)
        
        A_norm[indexes == i ] = (A_i - mean) / (std + eps)
        
    return A_norm
    
def _same_num_hypotheses(indexes: torch.Tensor) -> bool:
    """
    Check whether all groups (audios) have the same number of hypotheses.
    """
    _, counts = torch.unique(indexes, return_counts=True)
    return torch.all(counts == counts[0]).item()

def ppo_group_statistics(
    reward: torch.Tensor,
    values_old: torch.Tensor,
    indexes: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Compute advantages and critic targets depending on hypothesis structure.

    Args:
        reward:
            Tensor [B], reward per hypothesis.
        values_old:
            Tensor [B], critic predictions (broadcasted if per-audio).
        indexes:
            Tensor [B], audio id for each hypothesis.
        eps:
            Numerical stability constant.

    Returns:
        adv:
            Normalized advantages [B].
        critic_target:
            Target values for critic regression [B].
        mode:
            String describing which strategy was used.
    """

    same_hypo = _same_num_hypotheses(indexes)

    # --------------------------------------------------
    # CASE 1: Same number of hypotheses per audio
    # --------------------------------------------------
    if same_hypo:
        # Classic PPO
        adv = reward - values_old
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + eps)

        critic_target = reward
        mode = "flat_hypotheses"

    # --------------------------------------------------
    # CASE 2: Variable number of hypotheses per audio
    # --------------------------------------------------
    else:
        # Advantage normalized per audio
        adv = reward - values_old
        adv = _normalize_adv(adv, indexes, eps=eps)

        # Critic target = mean reward per audio
        rewards_means, _, _ = _mean_mean(reward, indexes, only=False)
        critic_target = rewards_means[indexes]

        mode = "grouped_by_audio"

    return adv.detach(), critic_target.detach(), mode

@torch.no_grad()
def decode_batch(
    log_probs: torch.Tensor,
    enc_len: torch.Tensor,
    asr_model: EncDecCTCModel | EncDecCTCModelBPE,
    return_hypotheses: bool = False,
    use_lm :  bool = True
) -> List[str]:
    
    """
    Decode a batch of CTC log-probs [B, T, V] to text using NeMo's decoder.
    """
    if use_lm :
        
        kenlm_path = rsc.files(RLNF.ressources) / "5gram_bambara.bin"
        kenlm_path = str(kenlm_path) 
        
        decoding_cfg = asr_model.cfg.decoding
        decoding_cfg.strategy = "pyctcdecode"
        decoding_cfg.beam.beam_size = 16           
        decoding_cfg.beam.return_best_hypothesis = False
        decoding_cfg.ngram_lm_model = kenlm_path  
        decoding_cfg.ngram_lm_alpha = 0.5        
        decoding_cfg.beam.beta = 1.5
        decoding_cfg.beam.search_type = "pyctcdecode"

        asr_model.change_decoding_strategy(decoding_cfg)
    
    
    if hasattr(asr_model.decoding, "ctc_decoder_predictions_tensor"):
        hyps = asr_model.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=log_probs, decoder_lengths=enc_len, fold_consecutive=False,return_hypotheses=return_hypotheses
        )
    else:
        raise AttributeError("Only CTC models are supported for now.")
    
    return [[h.text for h in hyp] for hyp in hyps] if isinstance(hyps, list) else hyps


def _blank_index(asr_model: EncDecCTCModel) -> int:
    return len(asr_model.decoder.vocabulary)  # QuartzNet-style: blank is last index

def _ensure_log_softmax(logits_btv: torch.Tensor) -> torch.Tensor:
    # If already log-probs: logsumexp ≈ 0
    lse = torch.logsumexp(logits_btv.detach(), dim=-1)
    if torch.allclose(lse, torch.zeros_like(lse), atol=1e-3, rtol=1e-3):
        return logits_btv
    return logits_btv.log_softmax(dim=-1)

def _encode_texts_for_ctc(
    asr_model: EncDecCTCModel | EncDecCTCModelBPE,
    texts: List[str],
) -> Tuple[List[List[int]], List[int]]:
    """
    Map decoded strings back to label indices (no blanks), consistent with actor's output vocab.
    - For BPE models: use asr_model.tokenizer.text_to_ids()
    - For char-level CTC: map each character via decoder.vocabulary
    Returns (list_of_id_lists, list_of_lengths)
    """
    ids_list: List[List[int]] = []
    lens_list: List[int] = []

    # Prefer tokenizer if present (BPE)
    tok = getattr(asr_model, "tokenizer", None)
    if tok is not None:
        for t in texts:
            ids = tok.text_to_ids(t) if hasattr(tok, "text_to_ids") else tok.encode(t)
            # ensure non-empty for CTC
            if len(ids) == 0:
                # pick a safe non-blank symbol; take index 1 if blank=0 else 0
                blank = _blank_index(asr_model)
                fallback = 1 if blank == 0 else 0
                ids = [fallback]
            ids_list.append(ids)
            lens_list.append(len(ids))
        return ids_list, lens_list

    # Char-level fallback via vocabulary
    if hasattr(asr_model, "decoder") and hasattr(asr_model.decoder, "vocabulary"):
        vocab: List[str] = asr_model.decoder.vocabulary
        sym2idx = {s: i for i, s in enumerate(vocab)}
        for t in texts:
            # direct per-character mapping
            ids = []
            for ch in t:
                if ch in sym2idx:
                    ids.append(sym2idx[ch])
                # If char not in vocab, you could skip or map to a fallback; here we skip.
            if len(ids) == 0:
                blank = _blank_index(asr_model)
                fallback = 1 if blank == 0 else 0
                ids = [fallback]
            ids_list.append(ids)
            lens_list.append(len(ids))
        return ids_list, lens_list

    raise RuntimeError("Could not derive CTC targets: no tokenizer and no vocabulary found.")


def _pack_targets_1d(targets_padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert a padded [B, Lmax] int tensor + [B] lengths into 1D concat targets for CTCLoss.
    """
    parts = []
    for i in range(targets_padded.size(0)):
        li = int(lengths[i].item())
        parts.append(targets_padded[i, :li])
    return torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.long, device=targets_padded.device)


def _seq_logprob_ctc(
    log_probs_btv: torch.Tensor,  # [B, T, V] log-probs
    input_lengths_b: torch.Tensor,  # [B] lengths in time-steps
    targets_padded_bl: torch.Tensor,  # [B, Lmax] label ids
    target_lengths_b: torch.Tensor,  # [B] target lengths
    blank_idx: int,
) -> torch.Tensor:
    """
    Compute per-sample sequence log-prob: log P(y|x) using CTC forward-backward.
    """
    # CTCLoss expects [T, B, V] and 1D concatenated targets
    log_probs_tbv = log_probs_btv.permute(1, 0, 2).float()
    flat_targets_1d = _pack_targets_1d(targets_padded_bl, target_lengths_b).to(log_probs_btv.device)

    ctc = nn.CTCLoss(blank=blank_idx, reduction="none", zero_infinity=True)
    nll = ctc(log_probs_tbv, flat_targets_1d, input_lengths_b.int(), target_lengths_b.int())  # [B]
    return -nll  # [B], sequence log-prob


@torch.no_grad()
def collect_batch(
    asr_model: EncDecCTCModel | EncDecCTCModelBPE,
    reward_model: RewardModel,
    critic: CriticModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    processor : RewardModelProcessor
) -> Dict[str, torch.Tensor | List[str]]:
    """
    One on-policy rollout over a single mini-batch for PPO.
    Stores ONLY CPU tensors needed for PPO; avoids time-major tensors.
    """
    
    # for nemo
    audios = batch["audio"].to(device)
    audio_lens = batch["audio_len"].to(device)    

    # Eval/no-grad rollout
    asr_model.eval()
    reward_model.eval()
    critic.eval()
    
    reward_model.to(device)
    critic.to(device)
    asr_model.to(device)
    
    R = [] #Rewards.
    V = [] #Values.
    K = [] #numbers of trajectories of each audios.
  
    TARGETS = []
    TARGET_LENS =[]
    AUDIO = []
    AUDIO_LENS = []
    INPUT_LENS = []
  

    LOG_OLD = []

    with torch.no_grad():
        # Forward actor -> CTC log-probs and encoded lengths
        # NeMo CTC typically returns (log_probs[B,T,V], enc_len[B], greedy_ids[B,T]) or similar tuple
        

        out = asr_model(processed_signal=audios, processed_signal_length=audio_lens)
        
        #out = asr_model.forward(input_signal=audio, input_signal_length=audio_lens)
        # Be tolerant to output tuple structure
        if isinstance(out, (list, tuple)):
            logits_or_logp3d = out[0]
            enc_len = out[1]
        else:
            raise RuntimeError("Unexpected ASR forward() return; expected (log_probs, enc_len, ...).")
        
        # === ensure log-probs for both decoding & CTCLoss ===
        log_probs3d = _ensure_log_softmax(logits_or_logp3d)

        # Decode to text (for reward model & diagnostics)
        transcriptions = decode_batch(log_probs3d, enc_len, asr_model)

        
        for i, tra in enumerate(transcriptions) :
        
            tran = processor.tokenizer.batch_encode_plus(tra, return_attention_mask=True, padding=True, return_tensors="pt")
            
            tgt_lists, tgt_lens_list = _encode_texts_for_ctc(asr_model, tra)
            tgt_tensors = [torch.tensor(x, dtype=torch.long) for x in tgt_lists]
            tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=0).to(device=device)  # [B, Lmax]
            tgt_lens = torch.tensor(tgt_lens_list, dtype=torch.long, device=device)              # [B]
            
            K_i = len(tra)
        

            audio_i = audios[i]              # [T, F]
            audio = audio_i.unsqueeze(0).repeat(K_i, 1, 1)   # [K_i, T, F]

            audio_len = audio_lens[i].expand(K_i)

            
            reward_model_input = {
                "audio": audio,
                "audio_len": audio_len,
                "text": tran["input_ids"],
                "text_attention_mask": tran["attention_mask"],
            }

            critic_model_input = {
                "audio": audio
            }
            
            reward_model_input = {k: v.to(device) if torch.is_tensor(v) else v for k, v in reward_model_input.items()}
            critic_model_input = {k: v.to(device) if torch.is_tensor(v) else v for k, v in critic_model_input.items()}

            reward = reward_model(**reward_model_input).logits
            values = critic(**critic_model_input)
            
            
            # 5. Log-prob CTC (old policy)
            logp_i = log_probs3d[i]          # [T_enc, V]
            len_i  = enc_len[i]              # scalar

            logp_i = logp_i.unsqueeze(0).repeat(K_i, 1, 1)   # [K_i, T_enc, V]
            len_i  = len_i.expand(K_i) 
            
            blank_idx = _blank_index(asr_model)
                    
            logp_old = _seq_logprob_ctc(logp_i, len_i, tgt_padded, tgt_lens, blank_idx).detach()

            
            K.append(K_i)
            
            TARGETS.append(tgt_padded)
            TARGET_LENS.append(tgt_lens)
            AUDIO.append(audio)
            AUDIO_LENS.append(audio_len)
            INPUT_LENS.append(len_i)
            
            R.append(reward)
            V.append(values)
            
            LOG_OLD.append(logp_old)
            
        
        
        
        Lmax = max(t.size(1) for t in TARGETS)

        TARGETS_padded = [
            F.pad(t, (0, Lmax - t.size(1))) for t in TARGETS
        ]
        
        
        audio_all = torch.cat(AUDIO, dim=0)              # [K_i, T, F]
        audio_lens_all = torch.cat(AUDIO_LENS, dim=0)    # [K_i]

        targets_all = torch.cat(TARGETS_padded, dim=0)          # [K_i, Lmax]
        target_lens_all = torch.cat(TARGET_LENS, dim=0)  # [K_i]

        input_lens_all = torch.cat(INPUT_LENS, dim=0)    # [K_i]

        logp_old_all = torch.cat(LOG_OLD, dim=0)          # [K_i]
        reward_all = torch.cat(R, dim=0)                  # [K_i]
        values_all = torch.cat(V, dim=0)
        
        indexes = torch.repeat_interleave(torch.arange(len(K)), torch.tensor(K, dtype=torch.long))
        
    # Return CPU payload only; keep raw text too (tiny memory footprint)
    return {
        "audio_batch": audio_all.cpu(),
        "audio_lengths": audio_lens_all.cpu(),
        "targets": targets_all.cpu(),          # [K_i, Lmax] (for PPO update)
        "target_lengths": target_lens_all.cpu(),     # [K_i]
        "input_lengths": input_lens_all.cpu(),       # [K_i] (time steps at CTC head)
        "log_probs_old": logp_old_all.cpu(),      # [K_i]
        "reward": reward_all.cpu(),               # [K_i]
        "values": values_all.cpu(),               # [K_i]
        "texts": transcriptions,              # keep raw strings for reward/debug
        "indexes" : indexes.cpu(),
        #
        # reward model text batch is not needed after reward is computed; not stored
    }
