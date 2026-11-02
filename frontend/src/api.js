// Thin client for the MovieSense FastAPI backend. No predictions are produced client-side.
const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

function describe(body, status) {
  if (!body) return `Request failed (${status})`;
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) {
    return body.detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
  }
  return `Request failed (${status})`;
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new ApiError("Cannot reach the MovieSense API. Is the backend running on port 8000?", 0);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(describe(body, res.status), res.status, body);
  return body;
}

export const api = {
  health: () => request("/health"),
  models: () => request("/api/models"),
  analyze: (text, k, useGenreFilter) =>
    request("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ text, k, use_genre_filter: useGenreFilter, explain: true }),
    }),
  searchTitles: (q) => request(`/api/movies/search?q=${encodeURIComponent(q)}&limit=8`),
  byTitle: (title, k) =>
    request(`/api/recommend/by-title?title=${encodeURIComponent(title)}&k=${k}`),
  similarTo: (wikiId, k) => request(`/api/movies/${wikiId}/similar?k=${k}`),
};
