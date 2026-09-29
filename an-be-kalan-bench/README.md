# An Bɛ Kalan ASR Benchmark

Experiments for adapting Bambara ASR to children's speech and readings from RobotsMali's GAIFE educational books. The released [`soloni-be-kalan-v0`](https://huggingface.co/RobotsMali/soloni-be-kalan-v0) checkpoint is based on `soloni-114m-tdt-ctc-v2` and evaluated on [`RobotsMali/an-be-kalan-bench`](https://huggingface.co/datasets/RobotsMali/an-be-kalan-bench).

## Contents

- `config/quartznet/` and `config/soloni/`: experiment-specific NeMo YAML files.
- `scripts/train.py`: shared ASR fine-tuning entry point.
- `scripts/test.py`: evaluates `.nemo` archives and optional Lightning checkpoints.
- `scripts/hf_to_nemo_asr.py`: prepares NeMo JSONL manifests from Hugging Face data.
- `requirements.txt`: pinned NeMo 2.5.0 training environment.

## Reproduce an Experiment

Run from the repository root and adjust local manifest/checkpoint paths in the selected config:

```bash
pip install -r an-be-kalan-bench/requirements.txt
python an-be-kalan-bench/scripts/train.py \
  --config an-be-kalan-bench/config/soloni/soloni-be-kalan-exp4.yaml
python an-be-kalan-bench/scripts/test.py --help
```

The dataset contains a small `main` split and a much larger `duplicate` split with repeated book text read by many speakers. Preserve book-level separation when creating evaluation splits; otherwise repeated sentences can inflate results. See the dataset and model cards for cohort-level limitations and reported WER/CER.

## Statistical review on a cloud machine

The review evidence and draft paper sections are in [`review/`](review/). The released Exp3 model configurations are saved in `an-be-kalan-bench/review/model_configs/`. The scripts were smoke-tested with Python 3.12 and NeMo 3.0.0; NeMo 2.5.0 is pinned for the original training environment but is not required by these evaluation scripts. Install a compatible NeMo ASR environment and `review-requirements.txt` for the dataset download. The scripts patch a known decoding-schema difference when newer NeMo loads the Soloni checkpoint. In the tested NeMo 3.0.0 environment, model loading required `NUMBA_DISABLE_JIT=1` to avoid a Numba `no locator available` error.

Run from the repository root:

```bash
pip install -r an-be-kalan-bench/review-requirements.txt
python an-be-kalan-bench/scripts/prepare_test_set.py
NUMBA_DISABLE_JIT=1 python an-be-kalan-bench/scripts/estimate_uncertainty.py --method both --runs 1024
NUMBA_DISABLE_JIT=1 python an-be-kalan-bench/scripts/select_error_review.py
```

The first command stores the pinned `main/test` audio and card under `an-be-kalan-bench/.evaluation/data/`. The second caches deterministic transcriptions, resamples test books for paired WER/CER intervals, and runs 1,024 stochastic Soloni passes with encoder dropout active; it writes CSVs and `REPORT.md` under `.evaluation/results/`. QuartzNet's released checkpoint has zero dropout, so the dropout output explicitly marks that method unavailable for QuartzNet. The third command exports 30 positive-CER Soloni cases and copied audio under `.evaluation/error_review/`. These directories are ignored by Git. Copy the result and error-review directories back for the later paper revision.

The test manifest also supplied duration-filtered validation recordings during model selection. Treat the intervals as descriptive for these selected checkpoints and this benchmark, not as independent held-out-test evidence. The released test data currently has 17 distinct book titles, although the dataset card lists 11; the analysis uses observed titles as clusters and records cluster counts.
