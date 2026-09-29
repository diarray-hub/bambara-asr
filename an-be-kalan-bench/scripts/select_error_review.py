#!/usr/bin/env python3
"""Cache Soloni CTC predictions and export 30 positive-CER cases for human review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

from eval_common import SOLONI, deterministic_predictions, load_model, read_jsonl, score, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("an-be-kalan-bench/.evaluation/data"))
    parser.add_argument("--results-dir", type=Path, default=Path("an-be-kalan-bench/.evaluation/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("an-be-kalan-bench/.evaluation/error_review"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_cases(cases: list[dict]) -> list[dict]:
    positive = [item for item in cases if item["character_errors"] > 0]
    if len(positive) < 30:
        raise ValueError(f"Need at least 30 positive-CER errors, found {len(positive)}")
    low = sorted(positive, key=lambda item: (item["cer"], item["id"]))[:10]
    remaining = [item for item in positive if item["id"] not in {case["id"] for case in low}]
    high = sorted(remaining, key=lambda item: (-item["cer"], item["id"]))[:10]
    remaining = [item for item in remaining if item["id"] not in {case["id"] for case in high}]
    median = statistics.median(item["cer"] for item in positive)
    middle = sorted(remaining, key=lambda item: (abs(item["cer"] - median), item["id"]))[:10]
    return [{**item, "band": label} for label, group in
            (("highest", high), ("middle", middle), ("lowest_positive", low)) for item in group]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    data_dir, results_dir, output_dir = args.data_dir.resolve(), args.results_dir.resolve(), args.output_dir.resolve()
    rows = read_jsonl(data_dir / "manifest.jsonl")
    if not rows:
        raise SystemExit("Empty test manifest")
    results_dir.mkdir(parents=True, exist_ok=True)
    identity_path = results_dir / "cache_identity.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        current_hash = hashlib.sha256((data_dir / "manifest.jsonl").read_bytes()).hexdigest()
        if identity.get("manifest_sha256") != current_hash:
            raise ValueError("Results cache was made from a different test manifest")
    cache = results_dir / "soloni_deterministic.jsonl"
    if cache.exists():
        predictions = read_jsonl(cache)
        if [item["id"] for item in predictions] != [row["id"] for row in rows]:
            raise ValueError("Cached Soloni predictions do not match the test manifest")
    else:
        import torch
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        model = load_model(SOLONI, device)
        predictions = deterministic_predictions(model, rows, data_dir, cache, args.batch_size)

    cases = []
    for row, prediction in zip(rows, predictions):
        counts = score(row["text"], prediction["hypothesis"])
        cases.append({**row, "hypothesis": prediction["hypothesis"], **counts,
                      "cer": counts["character_errors"] / counts["reference_characters"],
                      "wer": counts["word_errors"] / counts["reference_words"]})
    selected = choose_cases(cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    exported = []
    for item in selected:
        source = data_dir / item["audio_path"]
        destination = audio_dir / f"{item['id']}{source.suffix}"
        shutil.copyfile(source, destination)
        exported.append({"band": item["band"], "id": item["id"],
                         "audio_path": str(Path("audio") / destination.name),
                         "book": item["book"], "sentence_id": item["sentence_id"],
                         "speaker_age": item["speaker_age"], "speaker_gender": item["speaker_gender"],
                         "speaker_id": item["speaker_id"], "reference": item["text"],
                         "hypothesis": item["hypothesis"], "wer": item["wer"], "cer": item["cer"],
                         "word_errors": item["word_errors"], "character_errors": item["character_errors"],
                         "error_category": "", "review_notes": ""})
    fields = list(exported[0])
    with (output_dir / "review_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exported)
    write_jsonl(output_dir / "review_cases.jsonl", exported)
    dataset = json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8"))
    (output_dir / "README.md").write_text(
        "# Soloni error review subset\n\n"
        f"Model: `{SOLONI}` (CTC branch). Dataset: `{dataset['dataset']}` revision `{dataset['revision']}`.\n\n"
        "The 30 cases contain 10 highest, 10 closest to the median, and 10 lowest positive utterance CER values. "
        "Ties break by stable row ID. These cases deliberately sample errors and do not estimate population error frequencies. "
        "Fill `error_category` and `review_notes` after listening to the copied audio and checking the reference transcript.\n",
        encoding="utf-8",
    )
    print(f"Saved {len(exported)} review cases to {output_dir}")


if __name__ == "__main__":
    main()
