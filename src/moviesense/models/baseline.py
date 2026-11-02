"""TF-IDF (word 1-2-grams, vocabulary fitted on train only) + one-vs-rest logistic regression
(report §13.2). Explanations are exact for this linear model: the contribution of each n-gram to
a genre's logit is coefficient x TF-IDF value."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def make_vectorizer(max_features: int = 300_000) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True,
                           max_features=max_features, strip_accents="unicode", dtype=np.float32)


@dataclass
class BaselineModel:
    labels: list[str]
    vectorizer: TfidfVectorizer
    clf: OneVsRestClassifier
    thresholds: np.ndarray
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, texts: list[str], Y: np.ndarray, labels: list[str], C: float = 4.0,
            class_weight: str | None = "balanced", n_jobs: int = -1) -> "BaselineModel":
        vec = make_vectorizer()
        X = vec.fit_transform(texts)
        lr = LogisticRegression(C=C, solver="liblinear", class_weight=class_weight, max_iter=200)
        clf = OneVsRestClassifier(lr, n_jobs=n_jobs).fit(X, Y)
        return cls(labels, vec, clf, np.full(len(labels), 0.5),
                   {"C": C, "class_weight": class_weight, "vocab_size": len(vec.vocabulary_)})

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict_proba(self.vectorizer.transform(texts))

    def coef_matrix(self) -> np.ndarray:
        return np.vstack([e.coef_.ravel() for e in self.clf.estimators_])  # (G, V)

    def explain(self, text: str, genres: list[str], top_k: int = 8) -> dict[str, list[dict]]:
        """Top positive n-gram contributions to each requested genre's logit."""
        x = self.vectorizer.transform([text]).tocsr()
        idx, vals = x.indices, x.data
        names = self.vectorizer.get_feature_names_out()
        coef = self.coef_matrix()
        out = {}
        for g in genres:
            gi = self.labels.index(g)
            contrib = coef[gi, idx] * vals
            order = np.argsort(-contrib)
            items = [{"token": str(names[idx[j]]), "score": float(contrib[j])}
                     for j in order[:top_k] if contrib[j] > 0]
            out[g] = items
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "clf": self.clf}, path / "model.joblib",
                    compress=3)
        (path / "config.json").write_text(json.dumps(
            {"labels": self.labels, "thresholds": self.thresholds.tolist(), "meta": self.meta},
            indent=2))

    @classmethod
    def load(cls, path: Path) -> "BaselineModel":
        obj = joblib.load(path / "model.joblib")
        cfg = json.loads((path / "config.json").read_text())
        return cls(cfg["labels"], obj["vectorizer"], obj["clf"], np.array(cfg["thresholds"]),
                   cfg.get("meta", {}))
