import { useState } from "react";

function ScoreBar({ label, score, threshold, predicted }) {
  const pct = Math.round(score * 100);
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-sm">
        <span className={predicted ? "font-semibold" : "text-stone-600 dark:text-stone-400"}>{label}</span>
        <span className="tabular-nums text-stone-600 dark:text-stone-400">
          {score.toFixed(2)}
          {threshold !== undefined && <span className="ml-1 text-xs text-stone-400">/ τ {threshold.toFixed(2)}</span>}
        </span>
      </div>
      <div className="relative h-2.5 rounded-full bg-stone-100 dark:bg-stone-800" role="presentation">
        <div
          className={`h-2.5 rounded-full ${predicted ? "bg-amber-600" : "bg-stone-400 dark:bg-stone-600"}`}
          style={{ width: `${Math.max(pct, 1)}%` }}
        />
        {threshold !== undefined && (
          <div
            className="absolute -top-1 w-0.5 bg-stone-900 dark:bg-stone-100"
            style={{ left: `${threshold * 100}%`, height: "18px" }}
            title={`Decision threshold ${threshold.toFixed(2)}`}
          />
        )}
      </div>
    </div>
  );
}

export default function GenreScores({ result }) {
  const [showAll, setShowAll] = useState(false);
  const all = Object.entries(result.all_scores).sort((a, b) => b[1] - a[1]);
  const predicted = new Set(result.genres.map((g) => g.label));
  return (
    <section className="card" aria-labelledby="genres-h">
      <div className="mb-4 flex items-center justify-between">
        <h2 id="genres-h" className="label">Predicted genres</h2>
        <span className="chip">model: {result.model}</span>
      </div>
      {result.low_confidence && (
        <p className="mb-3 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:bg-amber-950 dark:text-amber-200">
          Low confidence: no genre cleared its threshold, so the single highest-scoring genre is shown.
        </p>
      )}
      <div className="space-y-3">
        {result.genres.map((g) => (
          <ScoreBar key={g.label} {...g} predicted />
        ))}
      </div>
      {result.near_threshold.length > 0 && (
        <div className="mt-5">
          <h3 className="label mb-2">Near threshold (not predicted)</h3>
          <div className="space-y-3">
            {result.near_threshold.map((g) => (
              <ScoreBar key={g.label} {...g} predicted={false} />
            ))}
          </div>
        </div>
      )}
      <button className="btn-ghost mt-4 -ml-2.5" onClick={() => setShowAll((s) => !s)} aria-expanded={showAll}>
        {showAll ? "Hide" : "Show"} all {all.length} genre scores
      </button>
      {showAll && (
        <ul className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-3">
          {all.map(([g, s]) => (
            <li key={g} className="flex justify-between gap-2">
              <span className={predicted.has(g) ? "font-semibold" : ""}>{g}</span>
              <span className="tabular-nums text-stone-500">{s.toFixed(3)}</span>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-4 text-xs text-stone-500">
        Scores are independent sigmoid probabilities. A genre is predicted when its score reaches its
        per-genre threshold τ, tuned on the validation split.
      </p>
    </section>
  );
}
