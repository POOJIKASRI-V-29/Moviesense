"""Milestone 6: explanation faithfulness via a deletion test (report §9.4, §13.3).

For validation single-sentence inputs: take the top predicted genre, delete the top-n attributed
words, and compare the drop in that genre's probability with deleting n random words from the
same sentence (averaged over several random draws). Faithful attributions should give larger drops.

    python scripts/eval_explanations.py --model baseline
    python scripts/eval_explanations.py --model distilbert
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moviesense.config import MODELS, REPORTS
from moviesense.data.augment import eval_condition_texts
from moviesense.data.dataset import load_splits
from moviesense.service import load_classifier

_TOKEN = re.compile(r"\S+")


def remove_positions(text: str, positions: set[int]) -> str:
    return " ".join(t for i, t in enumerate(_TOKEN.findall(text)) if i not in positions)


def word_positions(text: str, words: set[str]) -> set[int]:
    out = set()
    for i, tok in enumerate(_TOKEN.findall(text)):
        if any(w in words for w in re.findall(r"\w+", tok.lower())):
            out.add(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="baseline")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--random-draws", type=int, default=5)
    args = ap.parse_args()

    clf = load_classifier(args.model, MODELS)
    splits = load_splits()
    val = splits.val.sample(n=min(args.n, len(splits.val)), random_state=1)
    texts = eval_condition_texts(val, "first")
    rng = np.random.default_rng(0)
    probs = clf.predict_proba(texts)
    rows = []
    for text, p in zip(texts, probs):
        n_tok = len(_TOKEN.findall(text))
        if n_tok < args.top + 3:
            continue
        g = int(np.argmax(p))
        label = clf.labels[g]
        exp = clf.explain(text, [label])[label]
        top_words = {w["word"].lower() for w in exp["top_words"][:args.top]}
        pos_top = word_positions(text, top_words)
        if not pos_top:
            continue
        k = len(pos_top)
        variants = [remove_positions(text, pos_top)]
        for _ in range(args.random_draws):
            variants.append(remove_positions(text, set(rng.choice(n_tok, size=k, replace=False))))
        pv = clf.predict_proba(variants)[:, g]
        rows.append({"base": float(p[g]), "drop_top": float(p[g] - pv[0]),
                     "drop_random": float(np.mean(p[g] - pv[1:]))})
    d_top = np.array([r["drop_top"] for r in rows])
    d_rand = np.array([r["drop_random"] for r in rows])
    diff = d_top - d_rand
    boot = [rng.choice(diff, size=len(diff)).mean() for _ in range(1000)]
    report = {"model": args.model, "n_sentences": len(rows), "words_deleted": args.top,
              "mean_drop_top_attributed": float(d_top.mean()),
              "mean_drop_random": float(d_rand.mean()),
              "mean_difference": float(diff.mean()),
              "difference_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
              "fraction_top_drop_greater": float((diff > 0).mean()),
              "condition": "validation, first sentence, top predicted genre"}
    print(json.dumps(report, indent=1))
    (REPORTS / f"explain_{args.model}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
