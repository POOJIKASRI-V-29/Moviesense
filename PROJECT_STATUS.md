# MovieSense — Project Status

Statuses: **VERIFIED** (backed by execution/tests/artefacts) · **IN PROGRESS** · **BLOCKED** · **NOT STARTED**
Last updated: 2026-10-15

## Summary

| Area | Status | Evidence |
|------|--------|----------|
| Report audit, requirements, plan | VERIFIED | `PROJECT_REQUIREMENTS.md`, `IMPLEMENTATION_PLAN.md` |
| Dataset download (CMU Movie Summary Corpus) | VERIFIED | `scripts/download_data.sh`; archive size check 48,002,242 B passed; SHA-256 in `data/raw/MovieSummaries.tar.gz.sha256` |
| Preprocessing, labels, group-aware split, metadata store | VERIFIED | `scripts/build_dataset.py` ran in 95 s; `reports/data_report.json` |
| Data unit tests | VERIFIED | `pytest tests/test_data.py` → 15 passed |
| TF-IDF + LogReg baseline (validation) | VERIFIED | `logs/train_baseline.log`, `reports/baseline_validation.json`, `models/baseline/` |
| Retrieval index + evaluation | BLOCKED (iCloud eviction) | code: `scripts/build_index.py`, `src/moviesense/retrieval/`; first run stalled on I/O, see below |
| Explanations (baseline linear attributions + deletion test) | IN PROGRESS — written, not run | `src/moviesense/explain/linear.py`, `scripts/eval_explanations.py` |
| FastAPI backend | IN PROGRESS — written, not run | `src/moviesense/api/`, `src/moviesense/service.py`, `tests/test_api.py` (17 tests) |
| React frontend | IN PROGRESS — written, not installed/built | `frontend/` |
| DistilBERT model | NOT STARTED (training) — Colab workflow written, not run | `notebooks/train_distilbert_colab.ipynb`, `scripts/train_distilbert.py`, `src/moviesense/models/transformer.py` |
| Final test-set evaluation | NOT STARTED | `scripts/evaluate.py` written |
| README, `.env.example`, Dockerfile, run scripts | IN PROGRESS — written, not verified | |

## Verified facts

### Data (`reports/data_report.json`)
- Raw: 42,306 plots · 81,741 movie records · 448,740 cast rows with an actor name (64,258 movies).
- Joined plots↔metadata: 42,207 (99 plots had no metadata).
- Genre map v1.0 (`configs/genre_map.yaml`): 195 of 363 Freebase genres mapped to 19 coarse targets,
  covering 79.1% of label occurrences. Production-format/regional labels are deliberately unmapped.
- Cleaning: 2,627 dropped (no target genre), 281 dropped (< 20 tokens; 1st percentile = 21),
  7 exact duplicates → **39,292 movies**, 2.42 genres per movie on average.
- Title leakage check: 21.9% of summaries contain their own title (≥ 5% rule) → titles masked.
- Groups: 36,755 (1,963 multi-movie; 35 near-duplicate pairs at TF-IDF cosine ≥ 0.80).
- Split (seed 42): train 27,574 (70.2%) · val 5,867 (14.9%) · test 5,851 (14.9%); every genre in all splits.

### Baseline validation (`reports/baseline_validation.json`) — validation split only, not test
- Search over C ∈ {1, 4, 16} × class_weight ∈ {balanced, none}; selection criterion is val macro-F1
  averaged over the two single-sentence conditions. Best: **C = 1.0, class_weight = none**.
- 3 seeds (42, 13, 7; seeds change sampled training windows):
  single-sentence val macro-F1 **0.4135 ± 0.0021**; full-summary val macro-F1 ≈ 0.581–0.583.
- Saved: `models/baseline/model.joblib` (47 MB, seed 42), `config.json` (labels + thresholds),
  `preds_seed{42,13,7}.npz` (val/test probabilities for the final paired comparison).

## Current blocker — iCloud eviction (BLOCKED)
- `~/Documents` is synced by iCloud Drive. With the disk ~99% full, macOS evicted **12,662 project files**
  to iCloud (12,599 in `.venv`, plus all of `data/raw`, `data/processed`, `models/`), leaving placeholders.
  Files are not lost, but every read triggers a slow download.
- Evidence: the first `build_index.py` run sat 11 min blocked in `read()` on a numpy `.pyc` inside `.venv`
  (stack sample + `find -flags +dataless`); it was stopped before writing any output.
- Agreed fix (user, 2026-10-15): free ≥ 5 GB, then move the project to `~/Projects/MovieSense`
  (outside iCloud sync). Waiting for free space; nothing has been moved yet.

## Known issues / risks
- **Disk:** ~1.2 GB free. macOS swap files live on the same volume and grew to 1.8 GB during baseline
  training, so memory-heavy jobs consume disk. PyTorch/Transformers are not installed locally by
  instruction; DistilBERT is to be trained on Colab.
- Code written while blocked (API, frontend, Colab workflow, evaluation scripts) has only been
  syntax-checked (`py_compile`); none of it has been executed yet.
- `scripts/build_dataset.py` now writes the gzip with `mtime=0` so rebuilds are byte-identical. The current
  `dataset.jsonl.gz` was written before this change; data content and methodology are unchanged.
- Threshold tuning and model selection both use validation (pooled conditions); validation metrics are
  therefore optimistic. Test metrics will be reported separately and once.
