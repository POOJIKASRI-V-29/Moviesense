from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

RAW_TEXT_LIMIT = 10_000  # hard payload bound; the 2,000-char rule applies after normalisation


class TextRequest(BaseModel):
    text: str = Field(..., max_length=RAW_TEXT_LIMIT,
                      description="Plot sentence or short synopsis (<= 2,000 chars after normalisation)",
                      examples=["A young detective investigates a series of mysterious murders "
                                "while uncovering a conspiracy involving his closest friend."])


class PredictRequest(TextRequest):
    explain: bool = True


class AnalyzeRequest(TextRequest):
    k: int = Field(5, ge=1, le=20, description="Number of similar movies")
    use_genre_filter: bool = Field(False, description="Only return neighbours sharing a predicted genre")
    explain: bool = True


class RecommendRequest(TextRequest):
    k: int = Field(5, ge=1, le=20)
    genres: list[str] | None = Field(None, description="Optional genre filter")


class GenreScore(BaseModel):
    label: str
    score: float
    threshold: float


class CastMember(BaseModel):
    actor: str
    character: str | None = None


class SimilarMovie(BaseModel):
    rank: int
    wiki_id: int
    title: str | None
    year: int | None
    similarity: float
    genres: list[str]
    plot_excerpt: str
    cast: list[CastMember]
    cast_available: bool
    source: str


class InputMatch(BaseModel):
    wiki_id: int
    title: str | None
    year: int | None
    similarity: float
    source: str


class PredictResponse(BaseModel):
    input: str
    model: str
    genres: list[GenreScore]
    low_confidence: bool
    near_threshold: list[GenreScore]
    all_scores: dict[str, float]
    explanation: dict[str, Any] | None = None


class RecommendResponse(BaseModel):
    similar_movies: list[SimilarMovie]
    input_match: InputMatch | None
    input_cast: list[CastMember] | str
    tau_match: float
    retrieval_encoder: str
    genre_filter: list[str] | None


class AnalyzeResponse(PredictResponse, RecommendResponse):
    illustrative: bool = False


class TitleRecommendResponse(BaseModel):
    query_movie: dict[str, Any]
    other_matches: list[dict[str, Any]]
    similar_movies: list[SimilarMovie]
    retrieval_encoder: str


class ErrorResponse(BaseModel):
    detail: str
    suggestions: list[dict[str, Any]] | None = None
