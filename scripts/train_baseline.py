"""Milestone 3: TF-IDF + one-vs-rest logistic regression baseline (report §13.2).

* Vocabulary fitted on the training split only (full summaries + 1-3 sentence windows).
* Hyperparameters chosen on validation macro-F1 (mean of the single-sentence conditions).
* Per-genre thresholds tuned on validation predictions pooled across conditions.
* Final config re-trained with three seeds (seeds change the sampled windows);
  val/test probabilities are saved for the paired comparison in scripts/evaluate.py.
  The test split plays no part in any choice made here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moviesense.config import MODELS, REPORTS
from moviesense.data.augment import eval_condition_texts, training_texts
from moviesense.data.dataset import EVAL_CONDITIONS, PRIMARY_CONDITIONS, load_splits
from moviesense.models.baseline import BaselineModel
from moviesense.models.metrics import apply_thresholds, classification_report, tune_thresholds

OUT = MODELS / "baseline"


def evaluate_val(model, splits, Yv, val_texts):
    probs = {c: model.predict_proba(val_texts[c]) for c in EVAL_CONDITIONS}
    taus = tune_thresholds(np.vstack([Yv] * len(EVAL_CONDITIONS)),
                           np.vstack([probs[c] for c in EVAL_CONDITIONS]))
    reports = {c: classification_report(Yv, apply_thresholds(probs[c], taus), splits.labels)
               for c in EVAL_CONDITIONS}
    primary = float(np.mean([reports[c]["macro_f1"] for c in PRIMARY_CONDITIONS]))
    return taus, reports, primary, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 7])
    ap.add_argument("--grid-C", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    ap.add_argument("--grid-cw", nargs="+", default=["balanced", "none"])
    args = ap.parse_args()

    splits = load_splits()
    Ytr, Yv, Yte = splits.Y(splits.train), splits.Y(splits.val), splits.Y(splits.test)
    val_texts = {c: eval_condition_texts(splits.val, c) for c in EVAL_CONDITIONS}
    test_texts = {c: eval_condition_texts(splits.test, c) for c in EVAL_CONDITIONS}
    print(f"labels={len(splits.labels)} train={len(splits.train)} val={len(splits.val)} "
          f"test={len(splits.test)}")

    def expand(seed):
        tt = training_texts(splits.train, args.windows, seed)
        row = {w: i for i, w in enumerate(splits.train["wiki_id"])}
        return tt["text"].tolist(), Ytr[[row[w] for w in tt["wiki_id"]]]

    # ---- Hyperparameter search on validation (seed 42) ----
    texts, Y = expand(args.seeds[0])
    print(f"training texts: {len(texts)} (full + {args.windows} windows per movie)")
    search = []
    for C, cw in itertools.product(args.grid_C, args.grid_cw):
        t0 = time.time()
        m = BaselineModel.fit(texts, Y, splits.labels, C=C, class_weight=None if cw == "none" else cw)
        _, reports, primary, _ = evaluate_val(m, splits, Yv, val_texts)
        rec = {"C": C, "class_weight": cw, "val_primary_macro_f1": round(primary, 4),
               "val_full_macro_f1": round(reports["full"]["macro_f1"], 4),
               "fit_seconds": round(time.time() - t0, 1)}
        print(rec, flush=True)
        search.append(rec)
    best = max(search, key=lambda r: r["val_primary_macro_f1"])
    print("best:", best)

    # ---- Final config, three seeds ----
    runs = []
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        texts, Y = expand(seed)
        cw = None if best["class_weight"] == "none" else best["class_weight"]
        m = BaselineModel.fit(texts, Y, splits.labels, C=best["C"], class_weight=cw)
        taus, reports, primary, vprobs = evaluate_val(m, splits, Yv, val_texts)
        tprobs = {c: m.predict_proba(test_texts[c]) for c in EVAL_CONDITIONS}
        np.savez_compressed(OUT / f"preds_seed{seed}.npz", thresholds=taus,
                            **{f"val_{c}": vprobs[c] for c in EVAL_CONDITIONS},
                            **{f"test_{c}": tprobs[c] for c in EVAL_CONDITIONS})
        runs.append({"seed": seed, "val_primary_macro_f1": primary,
                     "val": {c: {k: v for k, v in r.items() if k != "per_genre_f1"}
                             for c, r in reports.items()},
                     "val_per_genre_f1_first": reports["first"]["per_genre_f1"]})
        print(f"seed {seed}: val primary macro-F1 {primary:.4f}", flush=True)
        if seed == args.seeds[0]:
            m.thresholds = taus
            m.meta.update({"seed": seed, "windows_per_movie": args.windows,
                           "val_primary_macro_f1": primary})
            m.save(OUT)

    vals = [r["val_primary_macro_f1"] for r in runs]
    report = {"model": "tfidf_logreg", "search": search, "best": best, "runs": runs,
              "val_primary_macro_f1_mean": float(np.mean(vals)),
              "val_primary_macro_f1_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
              "threshold_tuning": "per genre, on validation predictions pooled across "
                                  "full/first/random conditions"}
    (REPORTS / "baseline_validation.json").write_text(json.dumps(report, indent=1))
    print("saved", OUT, REPORTS / "baseline_validation.json")


if __name__ == "__main__":
    main()
