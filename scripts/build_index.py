"""Milestone 5: plot-vector index over every plot summary + retrieval evaluation (report §13.3).

    python scripts/build_index.py --encoder tfidf
    python scripts/build_index.py --encoder distilbert   # needs models/distilbert (train first)

Index: all plot summaries in the corpus (42,306). The encoder is unsupervised for TF-IDF
(no labels used) and the fine-tuned classifier encoder for DistilBERT.
Evaluation uses validation movies only:
  * genre-overlap Precision@k / Jaccard@k, query excluded, index restricted to labelled movies;
  * tau_match: top-1 cast-attribution precision for single-sentence queries from real summaries
    (correct = top-1 is the query's own film); chosen as the lowest threshold with >= 95% precision.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moviesense.config import DB_FILE, MODELS, REPORTS
from moviesense.data.augment import eval_condition_texts
from moviesense.data.dataset import load_splits
from moviesense.data.text import normalise
from moviesense.retrieval.evaluate import calibrate_tau_match, genre_overlap_metrics
from moviesense.retrieval.index import TfidfEncoder, VectorIndex

# Hand-written, fictional premises used to check that tau_match does not fire on unseen plots.
FICTIONAL_QUERIES = [
    "A young detective investigates a series of mysterious murders while uncovering a conspiracy "
    "involving his closest friend.",
    "A retired baker discovers that her sourdough starter can predict the weather.",
    "Two rival beekeepers in rural Norway fall in love during a honey festival.",
    "An astronaut stranded on a moon of Neptune teaches a robot to play chess.",
    "A teenage violinist joins a heist crew to steal a cursed instrument from a museum.",
]


def load_all_plots():
    con = sqlite3.connect(DB_FILE)
    rows = con.execute("SELECT m.wiki_id, m.plot FROM movies m WHERE m.has_plot=1 "
                       "ORDER BY m.wiki_id").fetchall()
    genres = {}
    for w, g in con.execute("SELECT wiki_id, genre FROM movie_genres"):
        genres.setdefault(w, set()).add(g)
    con.close()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    return ids, [normalise(r[1]) for r in rows], [frozenset(genres.get(int(w), ())) for w in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["tfidf", "distilbert"], default="tfidf")
    ap.add_argument("--max-queries", type=int, default=2000)
    args = ap.parse_args()

    ids, plots, genre_sets = load_all_plots()
    print(f"indexing {len(ids)} plots with {args.encoder}")
    if args.encoder == "tfidf":
        enc = TfidfEncoder().fit(plots)
        out_dir = MODELS / "retrieval_tfidf"
        out_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(enc.vectorizer, out_dir / "encoder.joblib", compress=3)
    else:
        from moviesense.models.transformer import TransformerModel
        enc = TransformerModel.load(MODELS / "distilbert")
        out_dir = MODELS / "distilbert"
    vecs = enc.encode(plots)
    index = VectorIndex(ids, vecs, args.encoder, genre_sets)

    splits = load_splits()
    val = splits.val.sample(n=min(args.max_queries, len(splits.val)), random_state=0)
    report = {"encoder": args.encoder, "index_size": len(ids), "conditions": {}}

    labelled = np.array([len(g) > 0 for g in genre_sets])
    eval_index = VectorIndex(ids[labelled], vecs[np.where(labelled)[0]], args.encoder,
                             [g for g, l in zip(genre_sets, labelled) if l])
    top1_sims, top1_ok = [], []
    for cond in ("full", "first", "random"):
        q = enc.encode(eval_condition_texts(val, cond))
        report["conditions"][cond] = genre_overlap_metrics(
            eval_index, q, val["wiki_id"].tolist(), val["genres"].tolist())
        if cond != "full":  # tau_match from single-sentence queries (query film IS in the index)
            for i, wid in enumerate(val["wiki_id"]):
                (n, s), = index.search(q[i:i + 1], k=1)
                top1_sims.append(s)
                top1_ok.append(n == int(wid))
        print(cond, report["conditions"][cond], flush=True)

    cal = calibrate_tau_match(np.array(top1_sims), np.array(top1_ok))
    fq = enc.encode(FICTIONAL_QUERIES)
    fictional = [{"query": t, "top1_similarity": index.search(fq[i:i + 1], k=1)[0][1]}
                 for i, t in enumerate(FICTIONAL_QUERIES)]
    cal["fictional_queries"] = fictional
    cal["fictional_false_matches"] = int(sum(f["top1_similarity"] >= cal["tau_match"]
                                             for f in fictional))
    cal["top1_accuracy_single_sentence"] = float(np.mean(top1_ok))
    report["tau_match"] = cal
    print("tau_match", cal["tau_match"], "top1 acc", cal["top1_accuracy_single_sentence"],
          "fictional false matches", cal["fictional_false_matches"])

    index.save(out_dir, {"tau_match": cal["tau_match"]})
    (REPORTS / f"retrieval_{args.encoder}.json").write_text(json.dumps(report, indent=1))
    print("saved", out_dir)


if __name__ == "__main__":
    main()
