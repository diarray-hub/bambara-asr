"""Shared deterministic scoring and NeMo inference for benchmark review scripts."""

from __future__ import annotations

import json
import tempfile
import unicodedata
from pathlib import Path

SOLONI = "RobotsMali/soloni-be-kalan-v0"
QUARTZNET = "RobotsMali/anbekalanNet"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize_text(value: str) -> str:
    """Case-fold, remove punctuation, and collapse whitespace for common scoring."""
    value = unicodedata.normalize("NFC", value).casefold()
    value = "".join(" " if unicodedata.category(char).startswith("P") else char for char in value)
    return " ".join(value.split())


def edit_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, reference_item in enumerate(reference, 1):
        current = [index]
        for column, hypothesis_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def score(reference: str, hypothesis: str) -> dict[str, int]:
    reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    words = reference.split()
    if not words or not reference:
        raise ValueError("Empty reference after scoring normalization")
    return {
        "word_errors": edit_distance(words, hypothesis.split()),
        "reference_words": len(words),
        "character_errors": edit_distance(reference, hypothesis),
        "reference_characters": len(reference),
    }


def cohort(row: dict) -> str | None:
    age = row.get("speaker_age")
    if age is None or age == "":
        return None
    age = int(age)
    if age < 10:
        return "under_10"
    if 10 <= age <= 15:
        return "age_10_15"
    if 16 <= age <= 20:
        return "age_16_20"
    return None


def load_model(repo: str, device: str):
    """Load NeMo checkpoint, patch post-2.5 decoding schemas, select Soloni CTC."""
    import torch
    import nemo
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf
    from packaging.version import Version

    config = ASRModel.from_pretrained(repo, return_config=True)
    if Version(nemo.__version__) >= Version("2.7.0"):
        OmegaConf.set_struct(config, False)
        for decoder in ("greedy", "beam"):
            boosting_tree = OmegaConf.select(config, f"decoding.{decoder}.boosting_tree")
            if boosting_tree is not None:
                boosting_tree.key_phrase_items_list = None
    with tempfile.TemporaryDirectory() as directory:
        config_path = Path(directory) / "config.yaml"
        OmegaConf.save(config, config_path)
        model = ASRModel.from_pretrained(repo, override_config_path=str(config_path), strict=False)
    model.eval()
    model.to(torch.device(device))
    if repo == SOLONI:
        model.change_decoding_strategy(decoder_type="ctc", verbose=False)
    return model


def transcribe(model, paths: list[str], batch_size: int, dropout: bool = False, seed: int = 0) -> list[str]:
    import torch

    dropouts = [
        module for module in model.encoder.modules()
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0
    ]
    if dropout and not dropouts:
        raise ValueError("No nonzero encoder dropout modules are available")
    original_forward = model._transcribe_forward

    if dropout:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        def dropout_forward(*args, **kwargs):
            # NeMo transcribe() has just called model.eval(); keep its training flag
            # false so SpecAugment and BatchNorm stay in their evaluation paths.
            for module in dropouts:
                module.train()
            return original_forward(*args, **kwargs)

        model._transcribe_forward = dropout_forward
    try:
        with torch.inference_mode():
            output = model.transcribe(paths, batch_size=batch_size, num_workers=0, verbose=False)
    finally:
        model._transcribe_forward = original_forward
        model.eval()
    return [item if isinstance(item, str) else item.text for item in output]


def deterministic_predictions(model, rows: list[dict], data_dir: Path, cache: Path, batch_size: int) -> list[dict]:
    if cache.exists():
        predictions = read_jsonl(cache)
        if [row["id"] for row in predictions] != [row["id"] for row in rows]:
            raise ValueError(f"Prediction cache does not match manifest: {cache}")
        return predictions
    paths = [str(data_dir / row["audio_path"]) for row in rows]
    hypotheses = transcribe(model, paths, batch_size)
    if len(hypotheses) != len(rows):
        raise ValueError("NeMo returned the wrong number of transcriptions")
    predictions = [{"id": row["id"], "hypothesis": hypothesis} for row, hypothesis in zip(rows, hypotheses)]
    write_jsonl(cache, predictions)
    return predictions
