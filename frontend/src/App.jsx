import { useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import Explanation from "./components/Explanation.jsx";
import GenreScores from "./components/GenreScores.jsx";
import InputCast from "./components/InputCast.jsx";
import SimilarMovies from "./components/SimilarMovies.jsx";

const MAX_CHARS = 2000;
// Example inputs for users to try. They are inputs only; every output comes from the backend.
const EXAMPLES = [
  "A young detective investigates a series of mysterious murders while uncovering a conspiracy involving his closest friend.",
  "Two lonely strangers meet on a train across Europe and fall in love over a single night.",
  "A group of astronauts must stop a rogue AI aboard their ship before it reaches Earth.",
  "A family moves into an old farmhouse where something in the cellar begins to whisper at night.",
];

function ErrorBox({ error, onSuggestion }) {
  if (!error) return null;
  const suggestions = error.body?.suggestions || [];
  return (
    <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
      <p>{error.message}</p>
      {suggestions.length > 0 && (
        <p className="mt-2">
          Did you mean:{" "}
          {suggestions.map((s, i) => (
            <button key={s.wiki_id} className="underline underline-offset-2" onClick={() => onSuggestion?.(s)}>
              {s.title}{s.year ? ` (${s.year})` : ""}{i < suggestions.length - 1 ? ", " : ""}
            </button>
          ))}
        </p>
      )}
    </div>
  );
}

function Spinner() {
  return <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" aria-hidden />;
}

function PlotTab({ onMoreLike }) {
  const [text, setText] = useState("");
  const [k, setK] = useState(5);
  const [filter, setFilter] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const trimmed = text.trim();

  async function submit(e) {
    e?.preventDefault();
    if (!trimmed) {
      setError({ message: "Please enter a plot sentence or short synopsis." });
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setResult(await api.analyze(text, k, filter));
    } catch (err) {
      setError(err);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <form onSubmit={submit} className="card space-y-4">
        <label htmlFor="plot" className="label block">Describe a movie plot</label>
        <textarea
          id="plot"
          rows={4}
          className="input resize-y"
          placeholder="e.g. A retired hitman is pulled back in for one last job in Hong Kong…"
          value={text}
          maxLength={MAX_CHARS}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => (e.metaKey || e.ctrlKey) && e.key === "Enter" && submit(e)}
        />
        <div className="flex flex-wrap items-center justify-between gap-3 text-sm">
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button key={ex} type="button" className="chip hover:border-amber-600 hover:text-amber-800 dark:hover:text-amber-400" onClick={() => setText(ex)}>
                {ex.slice(0, 38)}…
              </button>
            ))}
          </div>
          <span className={`tabular-nums ${text.length > MAX_CHARS * 0.9 ? "text-amber-700" : "text-stone-400"}`}>
            {text.length}/{MAX_CHARS}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-t border-stone-100 pt-4 dark:border-stone-800">
          <label className="flex items-center gap-2 text-sm">
            Similar movies
            <select className="rounded-lg border border-stone-300 bg-white px-2 py-1 dark:border-stone-700 dark:bg-stone-950" value={k} onChange={(e) => setK(Number(e.target.value))}>
              {[3, 5, 8, 10].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" className="h-4 w-4 accent-amber-700" checked={filter} onChange={(e) => setFilter(e.target.checked)} />
            Only movies sharing a predicted genre
          </label>
          <button type="submit" className="btn-primary ml-auto" disabled={loading || !trimmed}>
            {loading && <Spinner />} {loading ? "Analysing…" : "Analyse plot"}
          </button>
        </div>
      </form>

      <ErrorBox error={error} />

      {result && (
        <div className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-5">
            <div className="lg:col-span-2"><GenreScores result={result} /></div>
            <div className="lg:col-span-3"><Explanation result={result} /></div>
          </div>
          <InputCast result={result} />
          <SimilarMovies
            title="Similar movies from the CMU corpus"
            subtitle={`Top ${result.similar_movies.length} by cosine similarity of plot vectors (${result.retrieval_encoder})${result.genre_filter ? `, filtered to ${result.genre_filter.join(", ")}` : ""}.`}
            movies={result.similar_movies}
            onMoreLike={onMoreLike}
          />
        </div>
      )}
    </div>
  );
}

function TitleTab({ seed }) {
  const [query, setQuery] = useState("");
  const [options, setOptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const timer = useRef();

  useEffect(() => {
    if (seed) loadById(seed.wiki_id, seed.title);
  }, [seed]); // eslint-disable-line react-hooks/exhaustive-deps

  function onType(v) {
    setQuery(v);
    clearTimeout(timer.current);
    if (v.trim().length < 2) return setOptions([]);
    timer.current = setTimeout(() => {
      api.searchTitles(v.trim()).then((r) => setOptions(r.results)).catch(() => setOptions([]));
    }, 200);
  }

  async function run(promise) {
    setLoading(true);
    setError(null);
    setOptions([]);
    try {
      setResult(await promise);
    } catch (err) {
      setError(err);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }
  const loadById = (id, title) => {
    if (title) setQuery(title);
    return run(api.similarTo(id, 6));
  };
  const submit = (e) => {
    e.preventDefault();
    if (!query.trim()) return setError({ message: "Please enter a movie title." });
    run(api.byTitle(query.trim(), 6));
  };

  return (
    <div className="space-y-6">
      <form onSubmit={submit} className="card relative space-y-3">
        <label htmlFor="title" className="label block">Find movies like a known title</label>
        <div className="flex gap-3">
          <input id="title" className="input" autoComplete="off" placeholder="e.g. Ghosts of Mars" value={query} onChange={(e) => onType(e.target.value)} />
          <button className="btn-primary" disabled={loading}>{loading ? <Spinner /> : "Search"}</button>
        </div>
        {options.length > 0 && (
          <ul className="absolute left-5 right-5 z-10 mt-1 max-h-72 overflow-auto rounded-xl border border-stone-200 bg-white shadow-lg dark:border-stone-700 dark:bg-stone-900">
            {options.map((o) => (
              <li key={o.wiki_id}>
                <button
                  type="button"
                  disabled={!o.has_plot}
                  className="flex w-full justify-between px-4 py-2 text-left text-sm hover:bg-stone-50 disabled:opacity-50 dark:hover:bg-stone-800"
                  onClick={() => loadById(o.wiki_id, o.title)}
                >
                  <span>{o.title} {o.year && <span className="text-stone-500">({o.year})</span>}</span>
                  {!o.has_plot && <span className="text-xs text-stone-500">no plot summary</span>}
                </button>
              </li>
            ))}
          </ul>
        )}
        <p className="text-xs text-stone-500">Only the 42,306 movies with a plot summary in the corpus can be used as a query.</p>
      </form>
      <ErrorBox error={error} onSuggestion={(s) => loadById(s.wiki_id, s.title)} />
      {result && (
        <SimilarMovies
          title={`Movies similar to ${result.query_movie.title}${result.query_movie.year ? ` (${result.query_movie.year})` : ""}`}
          subtitle={`Query genres: ${result.query_movie.genres.join(", ") || "none mapped"}. The query film itself is excluded.`}
          movies={result.similar_movies}
          onMoreLike={(m) => loadById(m.wiki_id, m.title)}
        />
      )}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("plot");
  const [info, setInfo] = useState(null);
  const [offline, setOffline] = useState(false);
  const [seed, setSeed] = useState(null);

  useEffect(() => {
    api.models().then(setInfo).catch(() => setOffline(true));
  }, []);

  const moreLike = (m) => {
    setSeed({ wiki_id: m.wiki_id, title: m.title, t: Date.now() });
    setTab("title");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const val = info?.validation;
  return (
    <div className="mx-auto max-w-6xl px-4 pb-16 sm:px-6">
      <header className="flex flex-wrap items-end justify-between gap-4 py-8">
        <div>
          <h1 className="font-display text-4xl font-semibold tracking-tight">MovieSense</h1>
          <p className="mt-1 max-w-xl text-stone-600 dark:text-stone-400">
            Genres, explanations and similar films from a single plot sentence, with cast drawn only from real records.
          </p>
        </div>
        <div className="text-right text-xs text-stone-500" aria-live="polite">
          {offline && <span className="chip border-red-300 text-red-700">API offline</span>}
          {info && (
            <div className="space-y-0.5">
              <div><span className="chip">model: {info.model}</span> <span className="chip">retrieval: {info.retrieval_encoder}</span></div>
              {val?.val_primary_macro_f1_mean != null && (
                <div>Validation macro-F1 (single sentence): {val.val_primary_macro_f1_mean.toFixed(3)} ± {val.val_primary_macro_f1_std.toFixed(3)}</div>
              )}
              <div>{info.index_size.toLocaleString()} indexed plots · {info.labels.length} genres</div>
            </div>
          )}
        </div>
      </header>

      <nav className="mb-6 flex gap-1 rounded-xl bg-stone-200/60 p-1 text-sm font-medium dark:bg-stone-800/60 sm:w-fit" role="tablist">
        {[["plot", "Analyse a plot"], ["title", "Similar to a title"]].map(([id, label]) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
            className={`flex-1 rounded-lg px-4 py-2 transition sm:flex-none ${tab === id ? "bg-white shadow-sm dark:bg-stone-900" : "text-stone-600 hover:text-stone-900 dark:text-stone-400"}`}
          >
            {label}
          </button>
        ))}
      </nav>

      <main>
        <div hidden={tab !== "plot"}><PlotTab onMoreLike={moreLike} /></div>
        <div hidden={tab !== "title"}><TitleTab seed={seed} /></div>
      </main>

      <footer className="mt-16 border-t border-stone-200 pt-6 text-xs leading-relaxed text-stone-500 dark:border-stone-800">
        Data: CMU Movie Summary Corpus (Bamman, O'Connor &amp; Smith, ACL 2013), Wikipedia plots and Freebase metadata
        as of November 2012, released under CC BY-SA. Genre labels are noisy and coverage skews to English-language films.
        Predictions are statistical estimates; cast names are never generated.
      </footer>
    </div>
  );
}
