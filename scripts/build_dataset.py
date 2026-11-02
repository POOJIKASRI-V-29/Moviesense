"""Phase 1-2 of the report's milestones: parse, join, clean, label, group, split, build store.

Usage:
    python scripts/build_dataset.py --eda     # write reports/eda_freebase_genres.json only
    python scripts/build_dataset.py           # full pipeline
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moviesense.config import (DATA_PROCESSED, DATA_RAW, DATASET_FILE, DB_FILE, GENRE_MAP_FILE,
                               REPORTS, SEED, SPLITS_DIR)
from moviesense.data.labels import GenreMapper
from moviesense.data.loaders import load_cast, load_movies, load_plots
from moviesense.data.splitting import (assert_no_group_leakage, build_groups,
                                       group_stratified_split, near_duplicate_pairs)
from moviesense.data.text import text_hash
from moviesense.retrieval.store import build_store

NEAR_DUP_THRESHOLD = 0.80
TITLE_MASK_RATE = 0.05  # mask titles in plots if they appear in >= 5% of summaries (§7.7)


def pct(a):
    return [float(x) for x in np.percentile(a, [1, 5, 25, 50, 75, 95, 99])]


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def title_regex(title: str):
    t = title.strip()
    if len(t) < 3 or not re.search(r"[A-Za-z]", t):
        return None
    return re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.IGNORECASE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda", action="store_true")
    ap.add_argument("--min-tokens", type=int, default=None)
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    plots, movies, cast = load_plots(), load_movies(), load_cast()
    stats: dict = {"raw": {"plots": len(plots), "movies": len(movies), "cast_rows": len(cast),
                           "movies_with_cast": int(cast["wiki_id"].nunique())}}
    print("raw:", stats["raw"])

    joined = plots.merge(movies, on="wiki_id", how="inner")
    stats["join"] = {"plots_without_metadata": int((~plots.wiki_id.isin(movies.wiki_id)).sum()),
                     "joined": len(joined),
                     "joined_without_genres": int((joined.genres_fb.map(len) == 0).sum())}
    joined["n_tokens"] = joined["plot"].str.split().map(len)
    stats["plot_tokens_percentiles_1_5_25_50_75_95_99"] = pct(joined["n_tokens"])

    fb_counts = Counter(g for gs in joined["genres_fb"] for g in gs)
    if args.eda:
        (REPORTS / "eda_freebase_genres.json").write_text(json.dumps(
            {"stats": stats, "n_distinct": len(fb_counts),
             "counts": fb_counts.most_common()}, indent=1))
        print(json.dumps(stats, indent=1))
        print("distinct freebase genres:", len(fb_counts))
        return

    # ---- Labels (§7.5) ----
    mapper = GenreMapper(GENRE_MAP_FILE)
    joined["genres"] = joined["genres_fb"].map(mapper.map)
    covered = sum(c for g, c in fb_counts.items() if g.lower() in mapper.mapping)
    stats["genre_map"] = {"version": mapper.version, "freebase_genres_mapped": len(
        [g for g in fb_counts if g.lower() in mapper.mapping]),
        "freebase_genres_total": len(fb_counts),
        "label_occurrences_covered": round(covered / sum(fb_counts.values()), 4)}

    # ---- Cleaning (§7.4) ----
    n0 = len(joined)
    joined = joined[joined["genres"].map(len) > 0]
    n_no_genre = n0 - len(joined)
    min_tokens = args.min_tokens or 20
    short = joined["n_tokens"] < min_tokens
    joined = joined[~short]
    joined["hash"] = joined["plot"].map(text_hash)
    dup = joined.duplicated("hash", keep="first")
    joined = joined[~dup].reset_index(drop=True)
    stats["cleaning"] = {"dropped_no_target_genre": int(n_no_genre),
                         "min_tokens": min_tokens, "dropped_short": int(short.sum()),
                         "dropped_exact_duplicates": int(dup.sum()), "remaining": len(joined)}
    print("cleaning:", stats["cleaning"])

    # ---- Prune rare targets: support counted on all remaining rows before split ----
    support = Counter(g for gs in joined["genres"] for g in gs)
    kept = [g for g in mapper.targets if support[g] >= mapper.min_support]
    dropped = [g for g in mapper.targets if g not in kept]
    joined["genres"] = joined["genres"].map(lambda gs: [g for g in gs if g in kept])
    joined = joined[joined["genres"].map(len) > 0].reset_index(drop=True)
    stats["labels"] = {"targets": kept, "dropped_below_min_support": dropped,
                       "min_support": mapper.min_support,
                       "support": {g: support[g] for g in kept},
                       "labels_per_movie_mean": float(joined["genres"].map(len).mean())}

    # ---- Title leakage check (§7.7) ----
    hits = 0
    regs = [title_regex(t) for t in joined["title"]]
    for r, p in zip(regs, joined["plot"]):
        if r is not None and r.search(p):
            hits += 1
    rate = hits / len(joined)
    mask = rate >= TITLE_MASK_RATE
    if mask:
        joined["plot"] = [r.sub(" ", p) if r is not None else p for r, p in zip(regs, joined["plot"])]
        joined["plot"] = joined["plot"].str.replace(r"\s+", " ", regex=True).str.strip()
    stats["title_leakage"] = {"summaries_containing_own_title": hits, "rate": round(rate, 4),
                              "mask_threshold": TITLE_MASK_RATE, "titles_masked": bool(mask)}
    print("title leakage:", stats["title_leakage"])

    # ---- Groups and split (§7.6) ----
    pairs = near_duplicate_pairs(joined["plot"].tolist(), NEAR_DUP_THRESHOLD)
    joined["group"] = build_groups(joined["title"].tolist(), pairs)
    labels = kept
    Y = np.array([[int(g in gs) for g in labels] for gs in joined["genres"]], dtype=np.int8)
    joined["split"] = group_stratified_split(Y, joined["group"].to_numpy(), SEED)
    assert_no_group_leakage(joined)
    gsize = joined.groupby("group").size()
    split_counts = joined["split"].value_counts().to_dict()
    per_split_support = {s: {g: int(Y[(joined.split == s).to_numpy(), i].sum())
                             for i, g in enumerate(labels)} for s in ("train", "val", "test")}
    missing = {s: [g for g, c in d.items() if c == 0] for s, d in per_split_support.items()}
    stats["split"] = {"near_dup_threshold": NEAR_DUP_THRESHOLD, "near_dup_pairs": len(pairs),
                      "groups": int(gsize.size), "multi_movie_groups": int((gsize > 1).sum()),
                      "largest_group": int(gsize.max()), "counts": split_counts,
                      "fractions": {k: round(v / len(joined), 4) for k, v in split_counts.items()},
                      "genres_missing_from_a_split": missing,
                      "per_split_support": per_split_support, "seed": SEED}
    print("split:", split_counts, "missing:", missing)
    assert not any(missing.values()), "every target genre must appear in all splits (Table 12)"

    # ---- Save (IDs only in split files) ----
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        (SPLITS_DIR / f"{s}.txt").write_text(
            "\n".join(map(str, sorted(joined.loc[joined.split == s, "wiki_id"]))) + "\n")
    cols = ["wiki_id", "title", "year", "plot", "genres", "genres_fb", "group", "split", "n_tokens"]
    # mtime=0 keeps the gzip header constant so rebuilds are byte-identical (stable dataset hash)
    with io.TextIOWrapper(gzip.GzipFile(DATASET_FILE, "wb", mtime=0), encoding="utf-8") as fh:
        for rec in joined[cols].to_dict("records"):
            rec["year"] = None if pd.isna(rec["year"]) else int(rec["year"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (DATA_PROCESSED / "labels.json").write_text(json.dumps(labels))

    raw_hash = hashlib.sha256("".join(file_sha256(DATA_RAW / f) for f in
                                      ("plot_summaries.txt", "movie.metadata.tsv",
                                       "character.metadata.tsv")).encode()).hexdigest()[:16]
    data_hash = hashlib.sha256(DATASET_FILE.read_bytes()).hexdigest()[:16]
    stats["hashes"] = {"raw": raw_hash, "dataset": data_hash}

    # ---- Metadata store (all metadata rows; coarse genres where mappable) ----
    coarse = {int(w): [g for g in mapper.map(gs) if g in kept]
              for w, gs in zip(movies["wiki_id"], movies["genres_fb"])}
    build_store(DB_FILE, movies, plots, cast, coarse,
                {"labels": json.dumps(labels), "dataset_hash": data_hash, "raw_hash": raw_hash,
                 "genre_map_version": mapper.version,
                 "source": "CMU Movie Summary Corpus (Bamman, O'Connor & Smith, ACL 2013), CC BY-SA"})
    stats["store"] = {"movies": len(movies), "cast_rows": len(cast)}
    (REPORTS / "data_report.json").write_text(json.dumps(stats, indent=1))
    print("wrote", DATASET_FILE, DB_FILE, REPORTS / "data_report.json")


if __name__ == "__main__":
    main()
