# rlnf/ppo/rollout.py
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from torch.nn.utils.rnn import pad_sequence
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from ..Rewards.reward_model import RewardModel
from ..Rewards.reward_processor import RewardModelProcessor
import torch.nn.functional as F
import importlib.resources as rsc
import RLNF.ressources
from nemo.collections.asr.metrics.wer import word_error_rate



@torch.no_grad()
def decode_batch(
    log_probs: torch.Tensor,
    enc_len: torch.Tensor,
    asr_model: EncDecCTCModel | EncDecCTCModelBPE,
    return_hypotheses: bool = False,
    use_lm :  bool = True ,
    beam_size : int = 4
) -> List[str]:
    
    """
    Decode a batch of CTC log-probs [B, T, V] to text using NeMo's decoder.
    """

    asr = asr_model
    decoding_cfg = asr.cfg.decoding

    if use_lm :
        
        kenlm_path = rsc.files(RLNF.ressources) / "5gram_bambara.bin"
        kenlm_path = str(kenlm_path) 
        
        
        decoding_cfg.strategy = "pyctcdecode"
        decoding_cfg.beam.beam_size = beam_size           
        decoding_cfg.beam.return_best_hypothesis = False
        decoding_cfg.ngram_lm_model = kenlm_path  
        decoding_cfg.ngram_lm_alpha = 0.5        
        decoding_cfg.beam.beta = 1.5
        decoding_cfg.beam.search_type = "pyctcdecode"
    
    else :
        decoding_cfg.strategy = "greedy_batch"
        decoding_cfg.beam.return_best_hypothesis = True

    asr.change_decoding_strategy(decoding_cfg)
    
    
    if hasattr(asr.decoding, "ctc_decoder_predictions_tensor"):
        hyps = asr.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=log_probs, decoder_lengths=enc_len, fold_consecutive=False,return_hypotheses=return_hypotheses
        )
    else:
        raise AttributeError("Only CTC models are supported for now.")
    
    if isinstance(hyps, list) and len(hyps) > 0 and hasattr(hyps[0], "text"):
        return [h.text for h in hyps]
    else:
        return [[h.text for h in hyp] for hyp in hyps]




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
    #critic: CriticModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    processor : RewardModelProcessor, 
    use_lm : bool = True,
    beam_size : int = 4,
    alpha  = 1.0,
    beta = 0.3,
    temperature : float = 1.0
) -> Dict[str, torch.Tensor | List[str]]:
   
    
    asr_model.eval()
    reward_model.eval()
   
    audios = batch["audio"].to(device)
    audio_lens = batch["audio_len"].to(device)    

    
    reward_model.to(device)
    asr_model.to(device)
    
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

    greedy_trans = decode_batch(log_probs3d, enc_len, asr_model, use_lm=False) 

    if not use_lm :

        return {"greedy_trans" :  greedy_trans}
   

    transcriptions = decode_batch(log_probs3d, enc_len, asr_model, use_lm=use_lm,beam_size=beam_size)
    
    FINAL_AUDIO = []
    FINAL_AUDIO_LENS = []
    FINAL_TARGETS = []
    FINAL_TARGET_LENS = []
    FINAL_LENS = []
    SCORES = []
    FINAL_SAMPLE_WEIGHT = []
    REWARDS = [ ]

    for i, tra in enumerate(transcriptions) :
    
        tran = processor.tokenizer.batch_encode_plus(tra, return_attention_mask=True, padding=True, return_tensors="pt")
        
        tgt_lists, tgt_lens_list = _encode_texts_for_ctc(asr_model, tra)
        tgt_tensors = [torch.tensor(x, dtype=torch.long) for x in tgt_lists]
        tgt_pad = pad_sequence(tgt_tensors, batch_first=True, padding_value=0).to(device=device)  # [B, Lmax]
        tgt_lens = torch.tensor(tgt_lens_list, dtype=torch.long, device=device)  

        audio_i = audios[i].unsqueeze(0).repeat(len(tra), 1, 1)              # [T, F]
        audio_len = audio_lens[i].expand(len(tra))

            
        reward_model_input = {
            "audio": audio_i,
            "audio_len": audio_len,
            "text": tran["input_ids"],
            "text_attention_mask": tran["attention_mask"],
        }

            
        reward_model_input = {k: v.to(device) if torch.is_tensor(v) else v for k, v in reward_model_input.items()}

        rewards = reward_model(**reward_model_input).logits
        rewards = (rewards - rewards.mean()) #/ (rewards.std() + 1e-8)
       
        #reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        
    
        logp_i = log_probs3d[i].unsqueeze(0).repeat(len(tra), 1, 1)
        len_i = enc_len[i].expand(len(tra))

        logp_ctc = _seq_logprob_ctc(
            logp_i,
            len_i,
            tgt_pad,
            tgt_lens,
            _blank_index(asr_model)
        ).detach()

        logp_ctc = (logp_ctc - logp_ctc.mean()) #/ (logp_ctc.std() + 1e-8)

        scores = alpha * logp_ctc + beta * rewards

        weights = torch.softmax(scores / temperature, dim=0).detach()

        best = scores.argmax().item()

        SCORES.append(scores.mean())

        FINAL_AUDIO.append(audios[i].unsqueeze(0))
        FINAL_AUDIO_LENS.append(audio_lens[i].unsqueeze(0))
        FINAL_TARGETS.append(tgt_pad[best].unsqueeze(0))
        FINAL_TARGET_LENS.append(tgt_lens[best].unsqueeze(0))
        FINAL_LENS.append(len_i[best].unsqueeze(0))

        FINAL_SAMPLE_WEIGHT.append(weights[best])
        REWARDS.append(rewards.mean())

        Lmax = max(t.size(1) for t in FINAL_TARGETS)

        TARGETS_padded = [
            F.pad(t, (0, Lmax - t.size(1))) for t in FINAL_TARGETS
        ]

    return {
        "audio": torch.cat(FINAL_AUDIO).cpu(),
        "audio_len": torch.cat(FINAL_AUDIO_LENS).cpu(),
        "targets": torch.cat(TARGETS_padded).cpu(),
        "target_lengths": torch.cat(FINAL_TARGET_LENS).cpu(),
        "input_lenght" : torch.cat(FINAL_LENS).cpu(),
        "score" : torch.stack(SCORES).cpu(),
        "reward" : torch.stack(REWARDS).cpu(),
        "greedy_trans" : greedy_trans,
        "sample_weight" : torch.stack(FINAL_SAMPLE_WEIGHT).cpu()
    }
        

