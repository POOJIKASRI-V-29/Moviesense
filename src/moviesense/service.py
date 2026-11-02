"""Inference pipeline (report §9.2, §10.2): validate -> classify -> retrieve -> cast -> explain.

Cast is never predicted (§9.3): names come only from the metadata store for retrieved records,
each tied to its source wiki_id. The input itself gets cast only when its top neighbour clears
tau_match (calibrated on validation, see reports/retrieval_*.json); otherwise "unavailable".
"""

from __future__ import annotations

import difflib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from moviesense.config import DB_FILE, MODELS, REPORTS
from moviesense.data.text import validate_user_text
from moviesense.retrieval.index import TfidfEncoder, VectorIndex
from moviesense.retrieval.store import MetadataStore

SOURCE = "CMU Movie Summary Corpus (Bamman, O'Connor & Smith, ACL 2013), CC BY-SA"
NEAR_THRESHOLD_RATIO = 0.5  # non-predicted genres with p >= 0.5 * tau_g are shown as near-threshold
NEAR_THRESHOLD_MAX = 3


class UnknownTitleError(LookupError):
    def __init__(self, title: str, suggestions: list[dict]):
        super().__init__(title)
        self.title = title
        self.suggestions = suggestions


@dataclass
class Classifier:
    name: str
    labels: list[str]
    thresholds: np.ndarray
    impl: object

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.impl.predict_proba(texts)

    def explain(self, text: str, genres: list[str]) -> dict:
        if self.name == "baseline":
            from moviesense.explain.linear import explain_linear
            return explain_linear(self.impl, text, genres)
        return self.impl.explain(text, genres)


def load_classifier(name: str, models_dir: Path) -> Classifier:
    if name == "baseline":
        from moviesense.models.baseline import BaselineModel
        m = BaselineModel.load(models_dir / "baseline")
        return Classifier("baseline", m.labels, m.thresholds, m)
    if name == "distilbert":
        from moviesense.models.transformer import TransformerModel
        m = TransformerModel.load(models_dir / "distilbert")
        return Classifier("distilbert", m.labels, m.thresholds, m)
    raise ValueError(f"unknown model {name!r}")


class MovieSenseService:
    def __init__(self, models_dir: Path = MODELS, db_path: Path = DB_FILE,
                 model: str | None = None, retrieval: str | None = None):
        model = model or os.environ.get("MOVIESENSE_MODEL", "baseline")
        retrieval = retrieval or os.environ.get("MOVIESENSE_RETRIEVAL", "tfidf")
        self.models_dir = Path(models_dir)
        self.store = MetadataStore(db_path)
        self.classifier = load_classifier(model, self.models_dir)

        if retrieval == "tfidf":
            idx_dir = self.models_dir / "retrieval_tfidf"
            self._query_encoder = TfidfEncoder(joblib.load(idx_dir / "encoder.joblib"))
        elif retrieval == "distilbert":
            idx_dir = self.models_dir / "distilbert"
            if self.classifier.name == "distilbert":
                self._query_encoder = self.classifier.impl
            else:
                from moviesense.models.transformer import TransformerModel
                self._query_encoder = TransformerModel.load(idx_dir)
        else:
            raise ValueError(f"unknown retrieval encoder {retrieval!r}")
        self.retrieval_name = retrieval
        self.index = VectorIndex.load(idx_dir)
        self.tau_match = float(json.loads((idx_dir / "index_meta.json").read_text())["tau_match"])
        self._titles = None

    # ---------- classification ----------
    def predict(self, text: str, explain: bool = True) -> dict:
        text = validate_user_text(text)
        p = self.classifier.predict_proba([text])[0]
        taus = self.classifier.thresholds
        labels = self.classifier.labels
        predicted = [i for i in np.argsort(-p) if p[i] >= taus[i]]
        low_conf = not predicted
        if low_conf:
            predicted = [int(np.argmax(p))]
        near = [i for i in np.argsort(-p) if i not in predicted and p[i] >= NEAR_THRESHOLD_RATIO * taus[i]]
        result = {
            "input": text,
            "model": self.classifier.name,
            "genres": [{"label": labels[i], "score": round(float(p[i]), 4),
                        "threshold": round(float(taus[i]), 2)} for i in predicted],
            "low_confidence": low_conf,
            "near_threshold": [{"label": labels[i], "score": round(float(p[i]), 4),
                                "threshold": round(float(taus[i]), 2)}
                               for i in near[:NEAR_THRESHOLD_MAX]],
            "all_scores": {labels[i]: round(float(p[i]), 4) for i in range(len(labels))},
        }
        if explain:
            result["explanation"] = self.classifier.explain(text, [labels[i] for i in predicted])
        return result

    # ---------- retrieval ----------
    def _neighbours(self, query_vec, k: int, exclude: set[int] | None, genre_filter) -> list[dict]:
        hits = self.index.search(query_vec, k=k, exclude_ids=exclude, genre_filter=genre_filter)
        ids = [w for w, _ in hits]
        movies = self.store.movies(ids)
        cast = self.store.cast(ids)
        out = []
        for rank, (w, s) in enumerate(hits, 1):
            m = movies.get(w, {})
            plot = m.get("plot") or ""
            out.append({"rank": rank, "wiki_id": w, "title": m.get("title"), "year": m.get("year"),
                        "similarity": round(s, 4), "genres": m.get("genres", []),
                        "plot_excerpt": plot[:300] + ("…" if len(plot) > 300 else ""),
                        "cast": cast.get(w, []),
                        "cast_available": bool(cast.get(w)),
                        "source": SOURCE})
        return out

    def recommend_for_text(self, text: str, k: int = 5, genre_filter: list[str] | None = None
                           ) -> dict:
        text = validate_user_text(text)
        q = self._query_encoder.encode([text])
        neighbours = self._neighbours(q, k, None, set(genre_filter) if genre_filter else None)
        top = self.index.search(q, k=1)  # match test ignores any genre filter
        input_match, input_cast = None, None
        if top and top[0][1] >= self.tau_match:
            w, s = top[0]
            m = self.store.movies([w]).get(w, {})
            input_match = {"wiki_id": w, "title": m.get("title"), "year": m.get("year"),
                           "similarity": round(s, 4), "source": SOURCE}
            input_cast = self.store.cast([w], limit_per_movie=20)[w] or None
        cast_msg = None
        if input_match is None:
            best = top[0][1] if top else 0.0
            cast_msg = (f"unavailable: no matching record (max similarity {best:.2f} below "
                        f"threshold {self.tau_match:.2f})")
        elif not input_cast:
            cast_msg = "unavailable: matched record has no cast in character.metadata.tsv"
        return {"similar_movies": neighbours, "input_match": input_match,
                "input_cast": input_cast if input_cast else cast_msg,
                "tau_match": self.tau_match, "retrieval_encoder": self.retrieval_name,
                "genre_filter": sorted(genre_filter) if genre_filter else None}

    def analyze(self, text: str, k: int = 5, use_genre_filter: bool = False,
                explain: bool = True) -> dict:
        pred = self.predict(text, explain=explain)
        gf = [g["label"] for g in pred["genres"]] if use_genre_filter else None
        rec = self.recommend_for_text(pred["input"], k=k, genre_filter=gf)
        return {**pred, **rec, "illustrative": False}

    def recommend_for_title(self, title: str, k: int = 5) -> dict:
        title = (title or "").strip()
        if not title:
            raise UnknownTitleError(title, [])
        matches = [m for m in self.store.find_title(title, limit=10) if m["has_plot"]]
        if not matches:
            raise UnknownTitleError(title, self.suggest_titles(title))
        movie = matches[0]
        vec = self.index.vector_of(movie["wiki_id"])
        if vec is None:
            raise UnknownTitleError(title, self.suggest_titles(title))
        details = self.store.movies([movie["wiki_id"]])[movie["wiki_id"]]
        return {"query_movie": {"wiki_id": movie["wiki_id"], "title": details["title"],
                                "year": details["year"], "genres": details["genres"],
                                "source": SOURCE},
                "other_matches": [{"wiki_id": m["wiki_id"], "title": m["title"], "year": m["year"]}
                                  for m in matches[1:6]],
                "similar_movies": self._neighbours(vec, k, {movie["wiki_id"]}, None),
                "retrieval_encoder": self.retrieval_name}

    def recommend_for_id(self, wiki_id: int, k: int = 5) -> dict | None:
        vec = self.index.vector_of(wiki_id)
        if vec is None:
            return None
        d = self.store.movies([wiki_id])[wiki_id]
        return {"query_movie": {"wiki_id": wiki_id, "title": d["title"], "year": d["year"],
                                "genres": d["genres"], "source": SOURCE},
                "other_matches": [],
                "similar_movies": self._neighbours(vec, k, {wiki_id}, None),
                "retrieval_encoder": self.retrieval_name}

    def suggest_titles(self, title: str, n: int = 5) -> list[dict]:
        if self._titles is None:
            rows = self.store._q("SELECT wiki_id, title, year FROM movies WHERE has_plot=1")
            self._titles = {r["title"].lower(): dict(r) for r in rows}
        close = difflib.get_close_matches(title.lower(), list(self._titles), n=n, cutoff=0.6)
        return [self._titles[c] for c in close]

    def movie(self, wiki_id: int) -> dict | None:
        m = self.store.movies([wiki_id]).get(wiki_id)
        if not m:
            return None
        m["cast"] = self.store.cast([wiki_id], limit_per_movie=30)[wiki_id]
        m["source"] = SOURCE
        return m

    def info(self) -> dict:
        def read(name):
            p = REPORTS / name
            return json.loads(p.read_text()) if p.exists() else None
        val = read("baseline_validation.json") if self.classifier.name == "baseline" else \
            read("distilbert_validation.json")
        final = read("final_comparison.json")
        return {"model": self.classifier.name, "labels": self.classifier.labels,
                "thresholds": {g: round(float(t), 2) for g, t in
                               zip(self.classifier.labels, self.classifier.thresholds)},
                "retrieval_encoder": self.retrieval_name, "tau_match": self.tau_match,
                "index_size": int(len(self.index.wiki_ids)), "store": self.store.counts(),
                "validation": None if val is None else {
                    k: val.get(k) for k in ("best", "val_primary_macro_f1_mean",
                                            "val_primary_macro_f1_std")},
                "test": None if final is None else final.get("summary"),
                "source": SOURCE}
