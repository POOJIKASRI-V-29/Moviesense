# MovieSense

Sentence-based movie genre classification with evidence-based cast recommendation, built from the
design in `project-report.pdf` (Team 28, G2+TG2). Given one plot sentence or a short synopsis, MovieSense:

1. predicts every applicable genre (19 coarse genres, per-genre thresholds tuned on validation),
2. explains each predicted genre with word-level attributions,
3. retrieves the most similar movies from 42,306 real plot summaries,
4. shows cast **only** from `character.metadata.tsv` records of matched/similar films; it never generates names.

Current status, verified numbers and blockers: **[PROJECT_STATUS.md](PROJECT_STATUS.md)**.
Requirements traced to the report: [PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md).

## Architecture

```
data/raw (CMU corpus) ─► scripts/build_dataset.py ─► data/processed/{dataset.jsonl.gz, splits/, moviesense.db}
                                   │
       ┌───────────────────────────┼─────────────────────────────┐
scripts/train_baseline.py   scripts/build_index.py      notebooks/train_distilbert_colab.ipynb
TF-IDF + OvR LogReg         plot-vector index + τ_match  DistilBERT + IG (GPU, Colab)
       └──────────────┬────────────┴─────────────────────────────┘
              src/moviesense/service.py  ──►  FastAPI (src/moviesense/api)  ──►  React UI (frontend/)
```

| Path | Purpose |
|------|---------|
| `src/moviesense/data/` | parsing, normalisation/validation, genre mapping, group-aware iterative stratified split, short-text windows |
| `src/moviesense/models/` | baseline, DistilBERT model, metrics, threshold tuning, bootstrap CIs |
| `src/moviesense/retrieval/` | SQLite metadata store (names only), exact cosine index, retrieval metrics |
| `src/moviesense/explain/` | linear attributions (baseline); IG lives in `models/transformer.py` |
| `src/moviesense/api/` | FastAPI app and schemas |
| `configs/genre_map.yaml` | documented Freebase → coarse genre mapping |
| `reports/` | every reported number comes from a JSON file here |

## Setup

Requirements: Python ≥ 3.10, Node ≥ 18, ~1 GB free disk for the baseline (more for DistilBERT).
Do not keep the project in an iCloud-synced folder when disk space is low; macOS evicts files and every read stalls.

```bash
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir -r requirements.txt
./scripts/download_data.sh                       # official CMU archive, 48 MB, size-checked
.venv/bin/python scripts/build_dataset.py        # join, clean, labels, splits, SQLite store (~2 min)
.venv/bin/python scripts/train_baseline.py       # 6-config search on val + 3 seeds (~10 min)
.venv/bin/python scripts/build_index.py --encoder tfidf
.venv/bin/python scripts/eval_explanations.py --model baseline
.venv/bin/python scripts/evaluate.py             # final test-split metrics → reports/final_comparison.json
```

## Run

```bash
./scripts/run_backend.sh                          # http://127.0.0.1:8000/docs
cd frontend && npm install && npm run dev         # http://localhost:5173 (proxies /api to :8000)
```

API: `GET /health`, `GET /api/models`, `POST /api/analyze`, `POST /api/predict`, `POST /api/recommend`,
`GET /api/recommend/by-title`, `GET /api/movies/search`, `GET /api/movies/{wiki_id}`,
`GET /api/movies/{wiki_id}/similar`. Settings: see `.env.example`.

## Tests

```bash
.venv/bin/python -m pytest -q                     # data/unit tests + API tests against real artefacts
cd frontend && npm run build                      # frontend build check
```

## DistilBERT (proposed model)

Trained on a free Colab GPU because the local machine has no CUDA and very little disk:

1. `./scripts/package_colab_bundle.sh` → `dist/moviesense_colab_bundle.zip` (code + the *existing* processed
   dataset and split IDs, so DistilBERT uses exactly the baseline's splits and windows).
2. Open `notebooks/train_distilbert_colab.ipynb` in Colab (T4 GPU), upload the bundle, run all cells.
3. `./scripts/install_distilbert_artifacts.sh ~/Downloads/moviesense_distilbert_artifacts.zip`
   (refuses artefacts trained on a different dataset hash), then
   `.venv/bin/pip install --no-cache-dir -r requirements-ml.txt` and
   `MOVIESENSE_MODEL=distilbert MOVIESENSE_RETRIEVAL=distilbert ./scripts/run_backend.sh`.

## Methodology notes and deviations from the report

- **Genre set.** 19 coarse genres; production-format and regional labels (World cinema, Black-and-white,
  Indie, Short Film, Bollywood, …) are unmapped because a plot sentence cannot evidence them.
- **Title masking.** 21.9% of summaries contain their own title (≥ 5% rule), so titles are removed from
  training text (§7.7).
- **Thresholds.** Tuned per genre on validation predictions pooled over full/first/random-sentence
  conditions, so one threshold set serves both input lengths in the app. Same procedure for both models.
- **Baseline explanations.** Exact linear attributions (coefficient × TF-IDF value). For a linear logit
  with a zero baseline this equals Integrated Gradients. DistilBERT uses Captum IG as specified.
- **Retrieval.** TF-IDF cosine until the DistilBERT encoder is available; both are evaluated (§10.3).
- **Storage format.** gzipped JSONL rather than parquet (avoids a ~100 MB pyarrow install on a full disk).
- **Seeds.** Baseline: 3 seeds. DistilBERT: 1 seed by default on Colab; 3 seeds supported (`--seeds 42 13 7`).
- **Poster-branch comparison.** Qualitative only (report Table 3); no posters are available.

## Data licence

CMU Movie Summary Corpus — D. Bamman, B. O'Connor, N. A. Smith, *Learning Latent Personas of Film Characters*,
ACL 2013. <https://www.cs.cmu.edu/~ark/personas/>. Released under CC BY-SA; derived data must keep this licence.
Raw and processed data are not committed. Actor attributes (DOB, gender, height, ethnicity) are never loaded.
# Moviesense
