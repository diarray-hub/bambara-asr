#!/usr/bin/env python3
"""Download the pinned An bɛ kalan main/test split and its dataset card."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


DEFAULT_OUTPUT = Path("an-be-kalan-bench/.evaluation/data")
DATASET = "RobotsMali/an-be-kalan-bench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default="main", help="HF revision or commit SHA")
    return parser.parse_args()


def main() -> None:
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    args = parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(DATASET, revision=args.revision)
    revision = info.sha
    metadata_path = root / "dataset_info.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old["revision"] != revision:
            raise SystemExit(f"{root} contains revision {old['revision']}; choose a new output directory")

    card = hf_hub_download(DATASET, "README.md", repo_type="dataset", revision=revision)
    shutil.copyfile(card, root / "DATASET_CARD.md")
    dataset = load_dataset(DATASET, "main", split="test", revision=revision)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    audio_dir = root / "audio"
    audio_dir.mkdir(exist_ok=True)
    rows = []
    for index, record in enumerate(dataset):
        audio = record["audio"]
        suffix = Path(audio.get("path") or "").suffix.lower() or ".wav"
        if suffix not in {".wav", ".flac", ".mp3", ".ogg"}:
            suffix = ".wav"
        relative_path = Path("audio") / f"{index:04d}{suffix}"
        path = root / relative_path
        data = audio.get("bytes")
        if not path.exists():
            if data is not None:
                path.write_bytes(data)
            elif audio.get("path") and Path(audio["path"]).is_file():
                shutil.copyfile(audio["path"], path)
            else:
                raise ValueError(f"No audio bytes or readable path for row {index}")
        if path.stat().st_size == 0:
            raise ValueError(f"Empty audio file: {path}")
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if data is not None and file_hash != hashlib.sha256(data).hexdigest():
            raise ValueError(f"Cached audio differs from pinned dataset row {index}: {path}")
        rows.append(
            {
                "id": f"main-test-{index:04d}",
                "source_index": index,
                "audio_path": str(relative_path),
                "audio_sha256": file_hash,
                "text": record["text"],
                "book": record.get("BookTitle"),
                "sentence_id": record.get("sentenceID"),
                "speaker_age": record.get("speakerAge"),
                "speaker_gender": record.get("speakerGender"),
                "speaker_id": record.get("speakerID"),
                "duration": record.get("duration"),
            }
        )

    manifest = root / "manifest.jsonl"
    temporary = root / "manifest.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(manifest)
    metadata_path.write_text(
        json.dumps({"dataset": DATASET, "revision": revision, "config": "main", "split": "test", "rows": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(rows)} test rows from {DATASET}@{revision} to {root}")


if __name__ == "__main__":
    main()
