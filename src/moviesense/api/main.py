"""FastAPI service for MovieSense.

Run:  uvicorn moviesense.api.main:app --port 8000   (from the repo root, with src on PYTHONPATH)
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from moviesense import __version__
from moviesense.api.schemas import (AnalyzeRequest, AnalyzeResponse, ErrorResponse,
                                    PredictRequest, PredictResponse, RecommendRequest,
                                    RecommendResponse, TitleRecommendResponse)
from moviesense.data.text import InputValidationError
from moviesense.service import MovieSenseService, UnknownTitleError

log = logging.getLogger("moviesense.api")
logging.basicConfig(level=os.environ.get("MOVIESENSE_LOG_LEVEL", "INFO"))

STATE: dict = {"service": None, "error": None, "loaded_at": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    try:
        STATE["service"] = MovieSenseService()
        STATE["loaded_at"] = time.time()
        log.info("models loaded in %.1fs", time.time() - t0)
    except Exception as exc:  # keep /health reachable and explain what is missing
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        log.error("failed to load models: %s", STATE["error"])
    yield


app = FastAPI(title="MovieSense API", version=__version__, lifespan=lifespan,
              description="Sentence-based movie genre classification with evidence-based cast "
                          "recommendation. Data: CMU Movie Summary Corpus (CC BY-SA).")

origins = [o.strip() for o in os.environ.get(
    "MOVIESENSE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])

# ---- Simple per-client sliding-window rate limit (report §15.2) ----
RATE_LIMIT = int(os.environ.get("MOVIESENSE_RATE_LIMIT_PER_MINUTE", "60"))
_hits: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


@app.middleware("http")
async def rate_limit_and_log(request: Request, call_next):
    if request.url.path.startswith("/api/") and RATE_LIMIT > 0:
        client = request.client.host if request.client else "unknown"
        now = time.time()
        with _lock:
            q = _hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= RATE_LIMIT:
                return JSONResponse({"detail": "Rate limit exceeded; try again shortly."}, 429)
            q.append(now)
    t0 = time.time()
    response = await call_next(request)
    # No raw user text is logged — only path, status and latency.
    log.info("%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code,
             (time.time() - t0) * 1000)
    return response


def svc() -> MovieSenseService:
    if STATE["service"] is None:
        raise HTTPException(503, f"Models not loaded: {STATE['error']}")
    return STATE["service"]


@app.exception_handler(InputValidationError)
async def _invalid(_: Request, exc: InputValidationError):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.get("/health")
def health():
    s = STATE["service"]
    return {"status": "ok" if s else "degraded", "version": __version__,
            "model": s.classifier.name if s else None,
            "retrieval_encoder": s.retrieval_name if s else None,
            "error": STATE["error"]}


@app.get("/api/models")
def models_info():
    return svc().info()


@app.post("/api/predict", response_model=PredictResponse,
          responses={422: {"model": ErrorResponse}})
def predict(req: PredictRequest):
    return svc().predict(req.text, explain=req.explain)


@app.post("/api/recommend", response_model=RecommendResponse,
          responses={422: {"model": ErrorResponse}})
def recommend(req: RecommendRequest):
    s = svc()
    if req.genres:
        unknown = set(req.genres) - set(s.classifier.labels)
        if unknown:
            raise HTTPException(422, f"Unknown genres: {sorted(unknown)}")
    return s.recommend_for_text(req.text, k=req.k, genre_filter=req.genres)


@app.post("/api/analyze", response_model=AnalyzeResponse,
          responses={422: {"model": ErrorResponse}})
def analyze(req: AnalyzeRequest):
    return svc().analyze(req.text, k=req.k, use_genre_filter=req.use_genre_filter,
                         explain=req.explain)


@app.get("/api/recommend/by-title", response_model=TitleRecommendResponse,
         responses={404: {"model": ErrorResponse}})
def recommend_by_title(title: str = Query(..., min_length=1, max_length=200),
                       k: int = Query(5, ge=1, le=20)):
    try:
        return svc().recommend_for_title(title, k=k)
    except UnknownTitleError as e:
        return JSONResponse({"detail": f"No movie with a plot summary matches '{e.title}'.",
                             "suggestions": e.suggestions}, status_code=404)


@app.get("/api/movies/search")
def search_movies(q: str = Query(..., min_length=1, max_length=200),
                  limit: int = Query(10, ge=1, le=50)):
    return {"results": svc().store.find_title(q, limit=limit)}


@app.get("/api/movies/{wiki_id}/similar", response_model=TitleRecommendResponse,
         responses={404: {"model": ErrorResponse}})
def similar_to_movie(wiki_id: int, k: int = Query(5, ge=1, le=20)):
    out = svc().recommend_for_id(wiki_id, k=k)
    if out is None:
        raise HTTPException(404, f"No indexed plot summary for wiki_id {wiki_id}")
    return out


@app.get("/api/movies/{wiki_id}")
def movie(wiki_id: int):
    m = svc().movie(wiki_id)
    if m is None:
        raise HTTPException(404, f"No movie with wiki_id {wiki_id}")
    return m
