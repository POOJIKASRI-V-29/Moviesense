"""API integration tests against the real artefacts (baseline model, TF-IDF index, SQLite store)."""

import sqlite3

import pytest

from moviesense.config import DB_FILE, MODELS

pytestmark = pytest.mark.skipif(
    not ((MODELS / "baseline" / "model.joblib").exists()
         and (MODELS / "retrieval_tfidf" / "index_meta.json").exists() and DB_FILE.exists()),
    reason="artefacts missing: run build_dataset.py, train_baseline.py, build_index.py first")

FICTIONAL = ("A retired baker discovers that her sourdough starter can predict the weather "
             "and becomes a reluctant celebrity in her small town.")
DETECTIVE = ("A young detective investigates a series of mysterious murders while uncovering a "
             "conspiracy involving his closest friend.")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from moviesense.api import main
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def labels(client):
    return client.get("/api/models").json()["labels"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model"] == "baseline"


def test_models_info(client):
    info = client.get("/api/models").json()
    assert len(info["labels"]) == 19 and set(info["thresholds"]) == set(info["labels"])
    assert info["index_size"] > 40_000 and info["store"]["movies"] == 81_741


def test_analyze_returns_real_predictions_and_grounded_cast(client, labels):
    r = client.post("/api/analyze", json={"text": DETECTIVE, "k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["genres"] and all(g["label"] in labels for g in body["genres"])
    assert all(0.0 <= v <= 1.0 for v in body["all_scores"].values())
    assert len(body["similar_movies"]) == 5
    sims = [m["similarity"] for m in body["similar_movies"]]
    assert sims == sorted(sims, reverse=True)
    con = sqlite3.connect(DB_FILE)
    for m in body["similar_movies"]:
        assert con.execute("SELECT title FROM movies WHERE wiki_id=?", (m["wiki_id"],)
                           ).fetchone()[0] == m["title"]
        for c in m["cast"]:  # every actor name must exist in the store for that exact movie
            assert con.execute("SELECT 1 FROM cast_members WHERE wiki_id=? AND actor_name=?",
                               (m["wiki_id"], c["actor"])).fetchone()
    con.close()
    assert body["illustrative"] is False


def test_fictional_input_has_unavailable_cast(client):
    body = client.post("/api/analyze", json={"text": FICTIONAL}).json()
    assert body["input_match"] is None
    assert isinstance(body["input_cast"], str) and body["input_cast"].startswith("unavailable")


def test_explanation_words_come_from_input(client):
    body = client.post("/api/predict", json={"text": DETECTIVE}).json()
    assert set(body["explanation"]) == {g["label"] for g in body["genres"]}
    input_lower = DETECTIVE.lower()
    for exp in body["explanation"].values():
        assert all(w["word"].lower() in input_lower for w in exp["top_words"])
        assert [w["word"] for w in exp["word_scores"]][:3] == ["young", "detective", "investigates"]


def test_genre_filter(client):
    body = client.post("/api/analyze", json={"text": DETECTIVE, "use_genre_filter": True}).json()
    predicted = {g["label"] for g in body["genres"]}
    assert all(set(m["genres"]) & predicted for m in body["similar_movies"])


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {"text": "1234 5678 !!"},
                                     {"text": "a" * 2001}, {}, {"text": "ok plot", "k": 0},
                                     {"text": "ok plot", "k": 50}])
def test_validation_errors(client, payload):
    assert client.post("/api/analyze", json=payload).status_code == 422


def test_recommend_unknown_genre_rejected(client):
    r = client.post("/api/recommend", json={"text": DETECTIVE, "genres": ["Nonsense"]})
    assert r.status_code == 422


def test_recommend_by_known_title_excludes_itself(client):
    r = client.get("/api/recommend/by-title", params={"title": "Ghosts of Mars", "k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query_movie"]["title"] == "Ghosts of Mars"
    assert body["query_movie"]["wiki_id"] not in {m["wiki_id"] for m in body["similar_movies"]}
    assert len(body["similar_movies"]) == 5


def test_recommend_by_unknown_title_gives_404_with_suggestions(client):
    r = client.get("/api/recommend/by-title", params={"title": "Ghosts of Marz"})
    assert r.status_code == 404
    assert any(s["title"] == "Ghosts of Mars" for s in r.json()["suggestions"])
    r = client.get("/api/recommend/by-title", params={"title": "qqqzzzxxyy"})
    assert r.status_code == 404 and r.json()["suggestions"] == []


def test_movie_lookup(client):
    m = client.get("/api/movies/975900").json()  # Ghosts of Mars (corpus README example row)
    assert m["title"] == "Ghosts of Mars" and any(c["actor"] == "Ice Cube" for c in m["cast"])
    assert client.get("/api/movies/999999999").status_code == 404


def test_cors_allows_frontend_origin(client):
    r = client.options("/api/analyze", headers={"Origin": "http://localhost:5173",
                                                "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_rate_limit(client, monkeypatch):
    from moviesense.api import main
    monkeypatch.setattr(main, "RATE_LIMIT", 2)
    main._hits.clear()
    codes = [client.get("/api/models").status_code for _ in range(3)]
    main._hits.clear()
    assert codes == [200, 200, 429]


def test_similar_by_id(client):
    body = client.get("/api/movies/975900/similar", params={"k": 3}).json()
    assert body["query_movie"]["title"] == "Ghosts of Mars" and len(body["similar_movies"]) == 3
    assert 975900 not in {m["wiki_id"] for m in body["similar_movies"]}
    assert client.get("/api/movies/999999999/similar").status_code == 404
