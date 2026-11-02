"""Central paths and constants. Override the project root with MOVIESENSE_ROOT."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MOVIESENSE_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SPLITS_DIR = DATA_PROCESSED / "splits"
CONFIGS = ROOT / "configs"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

DATASET_FILE = DATA_PROCESSED / "dataset.jsonl.gz"  # classification set (joined, cleaned, labelled)
DB_FILE = DATA_PROCESSED / "moviesense.db"  # SQLite metadata store
GENRE_MAP_FILE = CONFIGS / "genre_map.yaml"

SEED = 42
MAX_INPUT_CHARS = 2000  # report §9.2 stage 2
