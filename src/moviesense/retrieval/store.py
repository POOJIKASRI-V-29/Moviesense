"""SQLite metadata store: movies, genres, cast (report §9.1, Table 11).

All queries are parameterised (§15.2). Cast holds only actor and character names (§7.2).
Every metadata row is kept, including movies without a plot or genres, for title/cast lookup.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    wiki_id      INTEGER PRIMARY KEY,
    freebase_id  TEXT,
    title        TEXT NOT NULL,
    title_norm   TEXT NOT NULL,
    year         INTEGER,
    runtime      REAL,
    languages    TEXT,
    countries    TEXT,
    has_plot     INTEGER NOT NULL DEFAULT 0,
    plot         TEXT
);
CREATE INDEX IF NOT EXISTS idx_movies_title_norm ON movies(title_norm);
CREATE TABLE IF NOT EXISTS movie_genres_freebase (wiki_id INTEGER, genre TEXT);
CREATE TABLE IF NOT EXISTS movie_genres (wiki_id INTEGER, genre TEXT);
CREATE INDEX IF NOT EXISTS idx_mg ON movie_genres(wiki_id);
CREATE INDEX IF NOT EXISTS idx_mgf ON movie_genres_freebase(wiki_id);
CREATE TABLE IF NOT EXISTS cast_members (
    wiki_id        INTEGER NOT NULL,
    actor_name     TEXT NOT NULL,
    character_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_cast_wiki ON cast_members(wiki_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def build_store(db_path: Path, movies: pd.DataFrame, plots: pd.DataFrame, cast: pd.DataFrame,
                coarse_genres: dict[int, list[str]], meta: dict[str, str]) -> None:
    from moviesense.data.text import normalise_title

    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    plot_by_id = dict(zip(plots["wiki_id"], plots["plot"]))
    con.executemany(
        "INSERT OR REPLACE INTO movies VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(int(r.wiki_id), r.freebase_id, r.title, normalise_title(r.title),
          None if pd.isna(r.year) else int(r.year),
          None if pd.isna(r.runtime) else float(r.runtime),
          ", ".join(r.languages), ", ".join(r.countries),
          int(r.wiki_id in plot_by_id), plot_by_id.get(r.wiki_id))
         for r in movies.itertuples()])
    con.executemany("INSERT INTO movie_genres_freebase VALUES (?,?)",
                    [(int(w), g) for w, gs in zip(movies["wiki_id"], movies["genres_fb"])
                     for g in gs])
    con.executemany("INSERT INTO movie_genres VALUES (?,?)",
                    [(int(w), g) for w, gs in coarse_genres.items() for g in gs])
    con.executemany("INSERT INTO cast_members VALUES (?,?,?)",
                    cast[["wiki_id", "actor_name", "character_name"]].itertuples(index=False))
    con.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)", list(meta.items()))
    con.commit()
    con.close()


class MetadataStore:
    """Read-only access used by the API. One connection per call keeps it thread-safe."""

    def __init__(self, db_path: Path):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Metadata store not found: {db_path}")
        self.db_path = str(db_path)

    def _q(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()

    def movies(self, wiki_ids: list[int]) -> dict[int, dict]:
        if not wiki_ids:
            return {}
        ph = ",".join("?" * len(wiki_ids))
        rows = self._q(f"SELECT wiki_id, title, year, runtime, languages, countries, plot "
                       f"FROM movies WHERE wiki_id IN ({ph})", tuple(wiki_ids))
        out = {r["wiki_id"]: dict(r) for r in rows}
        for r in self._q(f"SELECT wiki_id, genre FROM movie_genres WHERE wiki_id IN ({ph})",
                         tuple(wiki_ids)):
            out[r["wiki_id"]].setdefault("genres", []).append(r["genre"])
        for r in self._q(f"SELECT wiki_id, genre FROM movie_genres_freebase "
                         f"WHERE wiki_id IN ({ph})", tuple(wiki_ids)):
            out[r["wiki_id"]].setdefault("genres_freebase", []).append(r["genre"])
        for v in out.values():
            v.setdefault("genres", [])
            v.setdefault("genres_freebase", [])
        return out

    def cast(self, wiki_ids: list[int], limit_per_movie: int = 10) -> dict[int, list[dict]]:
        if not wiki_ids:
            return {}
        ph = ",".join("?" * len(wiki_ids))
        rows = self._q(f"SELECT wiki_id, actor_name, character_name FROM cast_members "
                       f"WHERE wiki_id IN ({ph}) ORDER BY rowid", tuple(wiki_ids))
        out: dict[int, list[dict]] = {w: [] for w in wiki_ids}
        for r in rows:
            if len(out[r["wiki_id"]]) < limit_per_movie:
                out[r["wiki_id"]].append({"actor": r["actor_name"],
                                          "character": r["character_name"] or None})
        return out

    def find_title(self, title: str, limit: int = 10) -> list[dict]:
        """Exact normalised-title matches first, then prefix/substring matches."""
        from moviesense.data.text import normalise_title

        key = normalise_title(title)
        if not key:
            return []
        rows = self._q(
            "SELECT wiki_id, title, year, has_plot, "
            "CASE WHEN title_norm = ? THEN 0 WHEN title_norm LIKE ? ESCAPE '\\' THEN 1 ELSE 2 END "
            "AS rank FROM movies WHERE title_norm LIKE ? ESCAPE '\\' "
            "ORDER BY rank, has_plot DESC, year DESC LIMIT ?",
            (key, _like_escape(key) + "%", "%" + _like_escape(key) + "%", limit))
        return [dict(r) for r in rows]

    def meta(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self._q("SELECT key, value FROM meta")}

    def counts(self) -> dict[str, int]:
        return {
            "movies": self._q("SELECT COUNT(*) c FROM movies")[0]["c"],
            "movies_with_plot": self._q("SELECT COUNT(*) c FROM movies WHERE has_plot=1")[0]["c"],
            "cast_rows": self._q("SELECT COUNT(*) c FROM cast_members")[0]["c"],
        }


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
