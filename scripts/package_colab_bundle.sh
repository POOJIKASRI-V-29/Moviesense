#!/usr/bin/env bash
# Package the code + the EXISTING processed dataset and split definitions for Colab training.
# Shipping the processed data (not re-deriving it) guarantees DistilBERT sees the exact same
# train/val/test movies, labels and windows as the baseline. Output: dist/moviesense_colab_bundle.zip
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for f in data/processed/dataset.jsonl.gz data/processed/labels.json data/processed/splits/train.txt \
         reports/data_report.json models/baseline/preds_seed42.npz; do
  [[ -f "$f" ]] || { echo "missing $f — run the local pipeline first" >&2; exit 1; }
done
mkdir -p dist
rm -f dist/moviesense_colab_bundle.zip
zip -qr dist/moviesense_colab_bundle.zip \
  src scripts configs notebooks requirements.txt requirements-ml.txt pyproject.toml \
  reports/data_report.json reports/baseline_validation.json \
  data/processed/dataset.jsonl.gz data/processed/labels.json data/processed/splits \
  models/baseline/preds_seed*.npz \
  -x '*/__pycache__/*' '*.pyc'
# Record split checksums so the notebook can prove the splits are unchanged.
shasum -a 256 data/processed/splits/*.txt data/processed/dataset.jsonl.gz
ls -lh dist/moviesense_colab_bundle.zip
