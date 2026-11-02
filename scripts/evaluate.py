"""Milestone 7: final test-split evaluation and model comparison (report §13.1, §13.2).

Reads the saved val/test probabilities (`models/<name>/preds_seed*.npz`, written by the training
scripts together with validation-tuned thresholds). No choice is made on test here: thresholds and
hyperparameters were fixed on validation. Reports, per model and condition, the mean ± sd over
seeds of every §13.1 metric, a bootstrap 95% CI of macro-F1 for the primary seed, and — when both
models exist — H1: paired bootstrap CI of the DistilBERT − TF-IDF macro-F1 difference on the
single-sentence conditions (accepted only if the CI excludes zero).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moviesense.config import MODELS, REPORTS
from moviesense.data.dataset import EVAL_CONDITIONS, PRIMARY_CONDITIONS, load_splits
from moviesense.models.metrics import apply_thresholds, bootstrap_macro_f1, classification_report

MODEL_DIRS = {"tfidf_logreg": MODELS / "baseline", "distilbert": MODELS / "distilbert"}
PRIMARY_SEED = 42
SCALARS = ["macro_f1", "micro_f1", "macro_precision", "macro_recall", "micro_precision",
           "micro_recall", "hamming_loss"]


def main():
    splits = load_splits()
    Yte = splits.Y(splits.test)
    labels = splits.labels
    out = {"n_test": int(len(Yte)), "labels": labels, "models": {}, "summary": {}}
    primary_preds = {}
    for name, d in MODEL_DIRS.items():
        files = sorted(d.glob("preds_seed*.npz"))
        if not files:
            print(f"skip {name}: no predictions in {d}")
            continue
        per_seed = {}
        for f in files:
            seed = int(f.stem.replace("preds_seed", ""))
            z = np.load(f)
            taus = z["thresholds"]
            per_seed[seed] = {}
            for c in EVAL_CONDITIONS:
                pred = apply_thresholds(z[f"test_{c}"], taus)
                per_seed[seed][c] = classification_report(Yte, pred, labels)
                if seed == PRIMARY_SEED:
                    primary_preds.setdefault(name, {})[c] = pred
        agg = {}
        for c in EVAL_CONDITIONS:
            agg[c] = {m: {"mean": float(np.mean([per_seed[s][c][m] for s in per_seed])),
                          "std": float(np.std([per_seed[s][c][m] for s in per_seed], ddof=1))
                          if len(per_seed) > 1 else 0.0} for m in SCALARS}
            if name in primary_preds:
                agg[c]["macro_f1_bootstrap_primary_seed"] = bootstrap_macro_f1(
                    Yte, [primary_preds[name][c]])["models"][0]
            agg[c]["per_genre_f1_primary_seed"] = per_seed.get(PRIMARY_SEED, next(iter(per_seed.values())))[c]["per_genre_f1"]
        out["models"][name] = {"seeds": sorted(per_seed), "conditions": agg}
        out["summary"][name] = {c: {"macro_f1": round(agg[c]["macro_f1"]["mean"], 4),
                                    "macro_f1_std": round(agg[c]["macro_f1"]["std"], 4),
                                    "micro_f1": round(agg[c]["micro_f1"]["mean"], 4)}
                                for c in EVAL_CONDITIONS}
        print(name, json.dumps(out["summary"][name]))

    if {"tfidf_logreg", "distilbert"} <= set(primary_preds):
        h1 = {}
        for c in PRIMARY_CONDITIONS:
            b = bootstrap_macro_f1(Yte, [primary_preds["tfidf_logreg"][c],
                                         primary_preds["distilbert"][c]], n_boot=1000)
            h1[c] = b["differences_vs_first"][0] | {"accept_H1": b["differences_vs_first"][0]["excludes_zero"]
                                                    and b["differences_vs_first"][0]["mean"] > 0}
        out["H1_distilbert_minus_tfidf"] = h1
        print("H1:", json.dumps(h1))
    else:
        out["H1_distilbert_minus_tfidf"] = "not tested: DistilBERT predictions not available"

    (REPORTS / "final_comparison.json").write_text(json.dumps(out, indent=1))
    print("wrote", REPORTS / "final_comparison.json")


if __name__ == "__main__":
    main()
