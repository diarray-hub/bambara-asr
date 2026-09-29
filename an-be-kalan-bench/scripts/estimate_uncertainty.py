#!/usr/bin/env python3
"""Book-cluster bootstrap and optional Monte Carlo dropout for released Exp3 ASR models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from eval_common import (
    QUARTZNET,
    SOLONI,
    cohort,
    deterministic_predictions,
    load_model,
    read_jsonl,
    score,
    transcribe,
    write_jsonl,
)

MODEL_IDS = {"soloni": SOLONI, "quartznet": QUARTZNET}
COHORTS = ("overall", "under_10", "age_10_15")
METRICS = (("wer", "word_errors", "reference_words"), ("cer", "character_errors", "reference_characters"))
DEFAULT_DATA = Path("an-be-kalan-bench/.evaluation/data")
DEFAULT_OUTPUT = Path("an-be-kalan-bench/.evaluation/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=("bootstrap", "mc-dropout", "both"), default="both")
    parser.add_argument("--runs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    point = (len(ordered) - 1) * probability
    lower = int(point)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (point - lower)


def metric_value(scores: list[dict], numerator: str, denominator: str) -> float:
    bottom = sum(item[denominator] for item in scores)
    if not bottom:
        raise ValueError("No reference units in metric denominator")
    return sum(item[numerator] for item in scores) / bottom


def selections(rows: list[dict]) -> dict[str, list[int]]:
    output = {name: [] for name in COHORTS}
    for index, row in enumerate(rows):
        output["overall"].append(index)
        name = cohort(row)
        if name:
            output[name].append(index)
    return output


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(method: str, model: str, group: str, metric: str, estimate: float | None,
              values: list[float], n: int, clusters: int, note: str = "") -> dict:
    return {
        "method": method, "model": model, "cohort": group, "metric": metric,
        "estimate": estimate if estimate is not None else "",
        "lower_95": percentile(values, 0.025) if values else "",
        "upper_95": percentile(values, 0.975) if values else "",
        "n_utterances": n, "n_books": clusters, "runs": len(values), "note": note,
    }


def run_bootstrap(rows: list[dict], deterministic: dict[str, list[dict]], runs: int,
                  seed: int) -> tuple[list[dict], list[dict]]:
    scores = {
        model: [score(row["text"], prediction["hypothesis"]) for row, prediction in zip(rows, predictions)]
        for model, predictions in deterministic.items()
    }
    selected = selections(rows)
    summaries, replicates = [], []
    for group, indices in selected.items():
        by_book = defaultdict(list)
        for index in indices:
            by_book[rows[index]["book"]].append(index)
        books = sorted(by_book)
        if not indices or len(books) < 2:
            for metric, numerator, denominator in METRICS:
                points = {
                    model: metric_value([model_scores[index] for index in indices], numerator, denominator)
                    for model, model_scores in scores.items()
                } if indices else {model: None for model in MODEL_IDS}
                points["soloni_minus_quartznet"] = (
                    points["soloni"] - points["quartznet"] if indices else None
                )
                for model, point in points.items():
                    summaries.append(summarize("book_bootstrap", model, group, metric, point, [], len(indices), len(books), "too few distinct books"))
            continue
        draws = random.Random(seed + COHORTS.index(group))
        distributions = defaultdict(list)
        for run in range(runs):
            sampled = [index for _ in books for index in by_book[draws.choice(books)]]
            for metric, numerator, denominator in METRICS:
                values = {
                    model: metric_value([model_scores[index] for index in sampled], numerator, denominator)
                    for model, model_scores in scores.items()
                }
                values["soloni_minus_quartznet"] = values["soloni"] - values["quartznet"]
                for model, value in values.items():
                    distributions[(model, metric)].append(value)
                    replicates.append({"method": "book_bootstrap", "run": run, "model": model,
                                       "cohort": group, "metric": metric, "value": value})
        for metric, numerator, denominator in METRICS:
            points = {
                model: metric_value([model_scores[index] for index in indices], numerator, denominator)
                for model, model_scores in scores.items()
            }
            points["soloni_minus_quartznet"] = points["soloni"] - points["quartznet"]
            for model, point in points.items():
                summaries.append(summarize("book_bootstrap", model, group, metric, point,
                                           distributions[(model, metric)], len(indices), len(books),
                                           "paired book resampling" if model == "soloni_minus_quartznet" else ""))
    return summaries, replicates


def run_dropout(rows: list[dict], data_dir: Path, output_dir: Path, runs: int,
                seed: int, batch_size: int, device: str,
                deterministic: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    selected = selections(rows)
    paths = [str(data_dir / row["audio_path"]) for row in rows]
    summaries, replicates = [], []
    # QuartzNet's published Jasper config has dropout=0.0 in every block.
    for group, indices in selected.items():
        books = {rows[index]["book"] for index in indices}
        for metric, _, _ in METRICS:
            summaries.append(summarize("mc_dropout", "quartznet", group, metric, None, [],
                                       len(indices), len(books), "unavailable: released model has zero dropout"))

    model = load_model(SOLONI, device)
    import torch
    positive = [module for module in model.encoder.modules()
                if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0]
    if not positive:
        raise RuntimeError("Soloni checkpoint has no nonzero encoder dropout; verify model revision")
    mc_dir = output_dir / "mc_dropout" / "soloni"
    mc_dir.mkdir(parents=True, exist_ok=True)
    distributions = defaultdict(list)
    for run in range(runs):
        cache = mc_dir / f"pass_{run:04d}.jsonl"
        if cache.exists():
            predictions = read_jsonl(cache)
            if [item["id"] for item in predictions] != [row["id"] for row in rows]:
                raise ValueError(f"Stale MC dropout cache: {cache}")
            hypotheses = [item["hypothesis"] for item in predictions]
        else:
            hypotheses = transcribe(model, paths, batch_size, dropout=True, seed=seed + run)
            if len(hypotheses) != len(rows):
                raise ValueError("Incomplete MC dropout pass")
            write_jsonl(cache, [{"id": row["id"], "hypothesis": text}
                               for row, text in zip(rows, hypotheses)])
        scored = [score(row["text"], hypothesis) for row, hypothesis in zip(rows, hypotheses)]
        for group, indices in selected.items():
            if not indices:
                continue
            subset = [scored[index] for index in indices]
            for metric, numerator, denominator in METRICS:
                value = metric_value(subset, numerator, denominator)
                distributions[(group, metric)].append(value)
                replicates.append({"method": "mc_dropout", "run": run, "model": "soloni",
                                   "cohort": group, "metric": metric, "value": value})
        if (run + 1) % 32 == 0 or run + 1 == runs:
            print(f"Soloni MC dropout: {run + 1}/{runs} passes", flush=True)
    for group, indices in selected.items():
        books = {rows[index]["book"] for index in indices}
        for metric, numerator, denominator in METRICS:
            point_scores = [score(rows[index]["text"], deterministic["soloni"][index]["hypothesis"])
                            for index in indices]
            point = metric_value(point_scores, numerator, denominator) if indices else None
            values = distributions[(group, metric)]
            note = "predictive pass spread; not a dataset confidence interval"
            if values and max(values) == min(values):
                note += "; no observed metric variation"
            summaries.append(summarize("mc_dropout", "soloni", group, metric, point, values,
                                       len(indices), len(books), note))
    return summaries, replicates


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.batch_size < 1:
        raise SystemExit("--runs and --batch-size must be positive")
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    rows = read_jsonl(data_dir / "manifest.jsonl")
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise SystemExit("Manifest is empty or has duplicate IDs")
    for row in rows:
        if not (data_dir / row["audio_path"]).is_file():
            raise SystemExit(f"Missing audio: {row['audio_path']}")
    data_info = json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256((data_dir / "manifest.jsonl").read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    import nemo
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    from huggingface_hub import HfApi
    api = HfApi()
    model_revisions = {}
    for model, repo in MODEL_IDS.items():
        try:
            model_revisions[model] = api.model_info(repo).sha
        except Exception:
            model_revisions[model] = "unavailable"
    identity_path = output_dir / "cache_identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        for model, revision in model_revisions.items():
            if revision == "unavailable" or previous.get("model_revisions", {}).get(model) == "unavailable":
                model_revisions[model] = previous.get("model_revisions", {}).get(model, revision)
    identity = {"dataset_revision": data_info["revision"], "manifest_sha256": manifest_hash,
                "models": MODEL_IDS, "model_revisions": model_revisions,
                "scoring": "NFC casefold punctuation-to-space whitespace collapse",
                "seed": args.seed, "batch_size": args.batch_size, "device": device,
                "nemo_version": nemo.__version__, "torch_version": torch.__version__}
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise SystemExit("Existing output cache has a different dataset, model, scoring rule, or seed")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    deterministic = {}
    for model, repo in MODEL_IDS.items():
        cache = output_dir / f"{model}_deterministic.jsonl"
        if cache.exists():
            predictions = read_jsonl(cache)
            if [item["id"] for item in predictions] != [row["id"] for row in rows]:
                raise ValueError(f"Stale deterministic cache: {cache}")
        else:
            loaded = load_model(repo, device)
            predictions = deterministic_predictions(loaded, rows, data_dir, cache, args.batch_size)
            del loaded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        deterministic[model] = predictions

    summaries, replicates = [], []
    if args.method in {"bootstrap", "both"}:
        part_summaries, part_replicates = run_bootstrap(rows, deterministic, args.runs, args.seed)
        summaries.extend(part_summaries)
        replicates.extend(part_replicates)
    if args.method in {"mc-dropout", "both"}:
        part_summaries, part_replicates = run_dropout(
            rows, data_dir, output_dir, args.runs, args.seed, args.batch_size, device, deterministic
        )
        summaries.extend(part_summaries)
        replicates.extend(part_replicates)

    write_csv(output_dir / "summary.csv",
              ["method", "model", "cohort", "metric", "estimate", "lower_95", "upper_95",
               "n_utterances", "n_books", "runs", "note"], summaries)
    write_csv(output_dir / "replicates.csv", ["method", "run", "model", "cohort", "metric", "value"], replicates)
    per_utterance = []
    for index, row in enumerate(rows):
        for model, predictions in deterministic.items():
            hypothesis = predictions[index]["hypothesis"]
            per_utterance.append({"id": row["id"], "model": model, "book": row["book"],
                                  "speaker_age": row.get("speaker_age"), "reference": row["text"],
                                  "hypothesis": hypothesis, **score(row["text"], hypothesis)})
    write_csv(output_dir / "per_utterance.csv",
              ["id", "model", "book", "speaker_age", "reference", "hypothesis", "word_errors",
               "reference_words", "character_errors", "reference_characters"], per_utterance)
    published = {"soloni": (0.22, 0.08), "quartznet": (0.40, 0.15)}
    lines = ["# An bɛ kalan uncertainty report", "", f"Dataset: `{data_info['dataset']}` revision `{data_info['revision']}`; {len(rows)} utterances.",
             f"Models: `{SOLONI}` revision `{model_revisions['soloni']}` (CTC branch) and `{QUARTZNET}` revision `{model_revisions['quartznet']}` (greedy CTC).",
             f"Scoring: {identity['scoring']}; corpus error rate = total edit errors / total reference units.",
             f"Method: {args.method}; {args.runs} replicates; seed {args.seed}; device {device}; NeMo {nemo.__version__}; PyTorch {torch.__version__}.", "",
             "## Results", "", "| Method | Cohort | Metric | Model | Estimate | 95% range | Utterances | Books | Note |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for item in summaries:
        estimate = f"{item['estimate']:.3f}" if item["estimate"] != "" else "—"
        interval = (f"{item['lower_95']:.3f} to {item['upper_95']:.3f}"
                    if item["lower_95"] != "" else "—")
        lines.append(f"| {item['method']} | {item['cohort']} | {item['metric']} | {item['model']} | "
                     f"{estimate} | {interval} | {item['n_utterances']} | {item['n_books']} | {item['note']} |")
    lines.extend(["", "Full outputs: `summary.csv`, `replicates.csv`, and `per_utterance.csv`.", ""])
    for model in MODEL_IDS:
        values = [item for item in per_utterance if item["model"] == model]
        wer = metric_value(values, "word_errors", "reference_words")
        cer = metric_value(values, "character_errors", "reference_characters")
        expected_wer, expected_cer = published[model]
        lines.append(f"- {model}: WER {wer:.4f}, CER {cer:.4f}; paper rounded values {expected_wer:.2f}/{expected_cer:.2f}."
                     + (" **Investigate scoring or checkpoint mismatch.**" if abs(wer - expected_wer) > .015 or abs(cer - expected_cer) > .015 else ""))
    lines.extend(["", "## Interpretation", "",
                  "Book-cluster intervals resample the observed test books. Their small number, especially within age cohorts, limits precision.",
                  "The same test manifest supplied duration-filtered validation recordings for checkpoint selection; these are conditional, descriptive intervals rather than evidence from an untouched held-out test.",
                  "Monte Carlo dropout ranges describe variability across stochastic predictions for a fixed dataset, not sampling confidence intervals. QuartzNet's released checkpoint has zero dropout and has no valid dropout estimate.",
                  "No inference about other Soloni experiment variants is possible from these two released checkpoints alone.", ""])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
