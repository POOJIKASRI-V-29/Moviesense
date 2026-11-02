"""Word-level attributions for the TF-IDF + logistic-regression baseline (report §9.4).

For a linear logit z_g = w_g . x + b_g, Integrated Gradients with an all-zero input baseline is
exactly w_g * x (the path integral of a constant gradient), so these attributions are the IG
attributions of the baseline model, computed in closed form. Each n-gram's contribution is split
equally over its words, then summed per word of the user's input.
"""

from __future__ import annotations

import re

import numpy as np

_WORD = re.compile(r"\b\w\w+\b", re.UNICODE)  # matches the vectoriser's default token pattern


def explain_linear(model, text: str, genres: list[str], top_k: int = 8) -> dict[str, dict]:
    vec = model.vectorizer
    x = vec.transform([text]).tocsr()
    idx, vals = x.indices, x.data
    names = vec.get_feature_names_out()
    coef = model.coef_matrix()
    preprocess = vec.build_preprocessor()

    # Words of the original text, in order, with their normalised form.
    words = [(m.group(0), preprocess(m.group(0))) for m in _WORD.finditer(text)]
    out: dict[str, dict] = {}
    for g in genres:
        gi = model.labels.index(g)
        contrib = coef[gi, idx] * vals
        word_score: dict[str, float] = {}
        for j, c in zip(idx, contrib):
            parts = names[j].split(" ")
            for p in parts:
                word_score[p] = word_score.get(p, 0.0) + float(c) / len(parts)
        order = np.argsort(-contrib)
        top_features = [{"token": str(names[idx[j]]), "score": round(float(contrib[j]), 4)}
                        for j in order[:top_k] if contrib[j] > 0]
        top_words = sorted(((w, s) for w, s in word_score.items() if s > 0), key=lambda t: -t[1])
        out[g] = {
            "top_features": top_features,
            "top_words": [{"word": w, "score": round(s, 4)} for w, s in top_words[:top_k]],
            "word_scores": [{"word": orig, "score": round(word_score.get(norm, 0.0), 4)}
                            for orig, norm in words],
        }
    return out


def delete_words(text: str, words: set[str], preprocess) -> str:
    kept = [m.group(0) for m in re.finditer(r"\S+", text)
            if not any(preprocess(w) in words for w in _WORD.findall(m.group(0)))]
    return " ".join(kept)
