"""Exact cosine nearest-neighbour search over plot vectors (report §9.2 stage 7, Table 8).

Two encoders share this interface:
  * TfidfEncoder  — sparse TF-IDF vectors (available without a GPU; the MVP retrieval space)
  * DistilBERT mean-pooled vectors from the fine-tuned encoder (see models/transformer.py)
Both are L2-normalised so a dot product is cosine similarity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class TfidfEncoder:
    name = "tfidf"

    def __init__(self, vectorizer: TfidfVectorizer | None = None):
        self.vectorizer = vectorizer or TfidfVectorizer(
            stop_words="english", min_df=2, max_df=0.5, sublinear_tf=True,
            strip_accents="unicode", dtype=np.float32)

    def fit(self, texts: list[str]) -> "TfidfEncoder":
        self.vectorizer.fit(texts)
        return self

    def encode(self, texts: list[str]):
        return normalize(self.vectorizer.transform(texts))


@dataclass
class VectorIndex:
    wiki_ids: np.ndarray  # (N,)
    vectors: object  # (N, D) dense float32 or sparse CSR, rows L2-normalised
    encoder_name: str
    genre_sets: list[frozenset] | None = None  # coarse genres per row, for optional filtering

    def search(self, query_vec, k: int = 10, exclude_ids: set[int] | None = None,
               genre_filter: set[str] | None = None) -> list[tuple[int, float]]:
        sims = self.vectors @ query_vec.T
        sims = np.asarray(sims.todense() if sp.issparse(sims) else sims).ravel()
        order = np.argsort(-sims)
        out = []
        for i in order:
            wid = int(self.wiki_ids[i])
            if exclude_ids and wid in exclude_ids:
                continue
            if genre_filter and self.genre_sets is not None and not (self.genre_sets[i] & genre_filter):
                continue
            out.append((wid, float(sims[i])))
            if len(out) >= k:
                break
        return out

    def vector_of(self, wiki_id: int):
        pos = np.where(self.wiki_ids == wiki_id)[0]
        if not len(pos):
            return None
        return self.vectors[pos[0]:pos[0] + 1]

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "index_ids.npy", self.wiki_ids)
        if sp.issparse(self.vectors):
            sp.save_npz(path / "index_vectors.npz", self.vectors.tocsr())
        else:
            np.save(path / "index_vectors.npy", self.vectors.astype(np.float16))
        joblib.dump([sorted(g) for g in (self.genre_sets or [])], path / "index_genres.joblib")
        (path / "index_meta.json").write_text(json.dumps({"encoder": self.encoder_name,
                                                          **(extra or {})}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        ids = np.load(path / "index_ids.npy")
        if (path / "index_vectors.npz").exists():
            vecs = sp.load_npz(path / "index_vectors.npz").tocsr()
        else:
            vecs = np.load(path / "index_vectors.npy").astype(np.float32)
        genres = [frozenset(g) for g in joblib.load(path / "index_genres.joblib")] or None
        meta = json.loads((path / "index_meta.json").read_text())
        return cls(ids, vecs, meta["encoder"], genres)
