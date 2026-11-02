"""Multi-label metrics, threshold tuning and bootstrap CIs (report §13.1, §13.2, §10.1 line 13)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def tune_thresholds(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray = THRESHOLD_GRID
                    ) -> np.ndarray:
    """tau_g = argmax_t F1_g(val, t), independently per genre."""
    taus = np.empty(y_true.shape[1])
    for g in range(y_true.shape[1]):
        yt = y_true[:, g].astype(bool)
        p = probs[:, g]
        preds = p[None, :] >= grid[:, None]  # (T, N)
        tp = (preds & yt).sum(1)
        fp = (preds & ~yt).sum(1)
        fn = (~preds & yt).sum(1)
        f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        taus[g] = grid[int(np.argmax(f1))]
    return taus


def apply_thresholds(probs: np.ndarray, taus: np.ndarray, fallback_top1: bool = True) -> np.ndarray:
    """Binary predictions; rows with no genre above threshold get their argmax (report §9.2)."""
    pred = (probs >= taus[None, :]).astype(np.int8)
    if fallback_top1:
        empty = pred.sum(1) == 0
        pred[np.where(empty)[0], probs[empty].argmax(1)] = 1
    return pred


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    kw = dict(zero_division=0)
    per_genre_f1 = f1_score(y_true, y_pred, average=None, **kw)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", **kw)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", **kw)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", **kw)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", **kw)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", **kw)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "per_genre_f1": {g: float(f) for g, f in zip(labels, per_genre_f1)},
        "n": int(y_true.shape[0]),
    }


def _macro_f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean(-1)


def bootstrap_macro_f1(y_true: np.ndarray, preds: list[np.ndarray], n_boot: int = 1000,
                       seed: int = 0) -> dict:
    """Bootstrap 95% CI of macro-F1 for each prediction matrix and for pairwise differences
    (preds[i] - preds[0]); resampling rows jointly so the comparison is paired."""
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    yt = y_true.astype(bool)
    stats = np.empty((n_boot, len(preds)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, p in enumerate(preds):
            pb, tb = p[idx].astype(bool), yt[idx]
            tp = (pb & tb).sum(0)
            fp = (pb & ~tb).sum(0)
            fn = (~pb & tb).sum(0)
            stats[b, k] = _macro_f1_from_counts(tp, fp, fn)
    out = {"models": [], "differences_vs_first": []}
    for k in range(len(preds)):
        lo, hi = np.percentile(stats[:, k], [2.5, 97.5])
        out["models"].append({"mean": float(stats[:, k].mean()), "ci95": [float(lo), float(hi)]})
        if k:
            d = stats[:, k] - stats[:, 0]
            dlo, dhi = np.percentile(d, [2.5, 97.5])
            out["differences_vs_first"].append({"mean": float(d.mean()),
                                                "ci95": [float(dlo), float(dhi)],
                                                "excludes_zero": bool(dlo > 0 or dhi < 0)})
    return out
