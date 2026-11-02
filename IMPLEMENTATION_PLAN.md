# MovieSense — Implementation Plan

Ordered for a working, honest MVP first. Maps to the report's milestones (Table 12).

| # | Step | Report milestone | Output |
|---|------|------------------|--------|
| 1 | Audit report, inspect environment | — | `PROJECT_REQUIREMENTS.md`, this plan, `PROJECT_STATUS.md` |
| 2 | Download corpus from the official CMU URL into `data/raw/` | 1. Data | raw files + checksum |
| 3 | Parse, clean, join, dedupe; EDA of genres/lengths; write `configs/genre_map.yaml` | 1. Data | `data/processed/movies.parquet`, `reports/eda.json` |
| 4 | Near-duplicate + title grouping; group-aware iterative stratified 70/15/15 split; title-leakage check | 2. Splits | `data/processed/splits/{train,val,test}.txt` |
| 5 | Short-text windows; TF-IDF + OvR LogReg; hyperparameter search on val; per-genre thresholds; 3 seeds | 3. Baseline | `models/baseline/`, `reports/baseline_*.json` |
| 6 | SQLite metadata store (movies, genres, cast — names only) | 5. Retrieval | `data/processed/moviesense.db` |
| 7 | Retrieval index (TF-IDF vectors first, DistilBERT embeddings once trained), τ_match, P@k/Jaccard@k | 5. Retrieval | `models/*/index.npy`, `reports/retrieval_*.json` |
| 8 | Explanations: exact linear attributions for the baseline; Captum IG for DistilBERT; deletion test | 6. Explanations | `reports/explain_*.json` |
| 9 | FastAPI service: `/health`, `/api/analyze`, `/api/predict`, `/api/recommend`, `/api/movies/...`, `/api/models` | 8. Demo | `src/moviesense/api/` |
| 10 | React + Vite + Tailwind UI wired to the API | 8. Demo | `frontend/` |
| 11 | DistilBERT fine-tune (MPS on Apple M4, or a free GPU), thresholds, artefact bundle, corpus embedding | 4. Transformer | `models/distilbert/` |
| 12 | Final test-set evaluation, bootstrap CIs, H1 test | 7. Evaluation | `reports/final_comparison.json`, `RESULTS.md` |
| 13 | pytest suite, frontend build, end-to-end check, README, `.env.example`, Dockerfile | — | passing checks |

## Environment notes (observed 2026-10-15)

- macOS, Apple M4, 16 GB RAM; Python 3.13 (Anaconda) available; Node 22; git.
- **Disk: ~2.3 GB free (99% full).** PyTorch + Transformers + DistilBERT weights + a fine-tuned copy
  need roughly 1–1.3 GB, so the transformer step is the tightest point. Pip/npm caches are disabled
  for installs and the raw tarball is deleted after extraction.
- No CUDA GPU. Apple MPS is used if PyTorch is installed; Colab-compatible CLI flags are provided.

## Decisions the report leaves open (to be documented as they are made)

- Minimum plot length, genre mapping table, near-duplicate threshold, title masking — set after EDA.
- Threshold tuning set: validation predictions pooled across full-summary and single-sentence
  conditions, so one threshold set serves both input lengths in the app.
- Retrieval vectors: TF-IDF cosine is used until the DistilBERT encoder exists; both are measured (§10.3).
