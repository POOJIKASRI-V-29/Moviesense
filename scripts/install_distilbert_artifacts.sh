#!/usr/bin/env bash
# Install the artefacts produced by notebooks/train_distilbert_colab.ipynb.
#   ./scripts/install_distilbert_artifacts.sh ~/Downloads/moviesense_distilbert_artifacts.zip
# Verifies that the model was trained on the same dataset version as the local pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP="${1:?usage: $0 path/to/moviesense_distilbert_artifacts.zip}"
cd "$ROOT"
unzip -o -q "$ZIP" -d "$ROOT"   # writes models/distilbert/ and reports/*distilbert*, final_comparison.json
PY="${PYTHON:-.venv/bin/python}"
"$PY" - <<'EOF'
import json
local = json.load(open("reports/data_report.json"))["hashes"]["dataset"]
trained = json.load(open("models/distilbert/config.json"))["meta"]["dataset_hash"]
print("local dataset hash  :", local)
print("model dataset hash  :", trained)
if local != trained:
    raise SystemExit("MISMATCH: the model was trained on a different dataset version")
print("OK: DistilBERT artefacts match the local dataset and splits.")
EOF
cat <<'EOF'
Next:
  .venv/bin/pip install --no-cache-dir -r requirements-ml.txt     # ~0.5 GB (torch, transformers, captum)
  MOVIESENSE_MODEL=distilbert MOVIESENSE_RETRIEVAL=distilbert ./scripts/run_backend.sh
EOF
