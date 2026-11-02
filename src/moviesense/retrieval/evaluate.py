"""Retrieval metrics (report §13.3) and the tau_match calibration."""

from __future__ import annotations

import numpy as np


def genre_overlap_metrics(index, query_vecs, query_ids, query_genres, ks=(5, 10)) -> dict:
    """Genre-overlap Precision@k and mean Jaccard@k; the query is excluded from its own list."""
    gold = {int(w): frozenset(g) for w, g in zip(index.wiki_ids, index.genre_sets)}
    kmax = max(ks)
    p = {k: [] for k in ks}
    j = {k: [] for k in ks}
    for qi, (wid, qg) in enumerate(zip(query_ids, query_genres)):
        qg = frozenset(qg)
        nbrs = index.search(query_vecs[qi:qi + 1], k=kmax, exclude_ids={int(wid)})
        for k in ks:
            top = [gold[n] for n, _ in nbrs[:k]]
            p[k].append(np.mean([len(qg & g) > 0 for g in top]))
            j[k].append(np.mean([len(qg & g) / len(qg | g) for g in top]))
    return {f"precision@{k}": float(np.mean(p[k])) for k in ks} | \
           {f"jaccard@{k}": float(np.mean(j[k])) for k in ks} | {"n_queries": len(query_ids)}


def calibrate_tau_match(top1_sims: np.ndarray, top1_correct: np.ndarray, target_precision=0.95
                        ) -> dict:
    """Smallest threshold whose 'input match' precision is >= target (conservative, §9.2)."""
    grid = np.round(np.arange(0.05, 1.0, 0.01), 2)
    rows = []
    chosen = None
    for t in grid:
        sel = top1_sims >= t
        n = int(sel.sum())
        prec = float(top1_correct[sel].mean()) if n else float("nan")
        rows.append({"tau": float(t), "declared": n, "precision": prec,
                     "coverage": n / len(top1_sims)})
        if chosen is None and n >= 20 and prec >= target_precision:
            chosen = float(t)
    if chosen is None:
        chosen = 1.01  # never declare a match if precision target is unreachable
    return {"tau_match": chosen, "target_precision": target_precision, "curve": rows}
