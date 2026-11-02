import { useEffect, useState } from "react";

// Map each attributed word back onto the original text (in order), so the input is shown verbatim.
function alignWords(text, wordScores) {
  const segments = [];
  const lower = text.toLowerCase();
  let cursor = 0;
  for (const { word, score } of wordScores) {
    const at = lower.indexOf(word.toLowerCase(), cursor);
    if (at < 0) continue;
    if (at > cursor) segments.push({ text: text.slice(cursor, at), score: null });
    segments.push({ text: text.slice(at, at + word.length), score });
    cursor = at + word.length;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), score: null });
  return segments;
}

function Highlighted({ text, wordScores }) {
  const maxAbs = Math.max(1e-9, ...wordScores.map((w) => Math.abs(w.score)));
  return (
    <p className="text-lg leading-9">
      {alignWords(text, wordScores).map((seg, i) => {
        if (seg.score === null || seg.score === 0) return <span key={i}>{seg.text}</span>;
        const a = Math.min(1, Math.abs(seg.score) / maxAbs);
        const bg = seg.score > 0 ? `rgba(217, 119, 6, ${0.12 + 0.6 * a})` : `rgba(71, 85, 105, ${0.08 + 0.3 * a})`;
        return (
          <mark
            key={i}
            className="rounded px-0.5 text-inherit"
            style={{ backgroundColor: bg }}
            title={`attribution ${seg.score > 0 ? "+" : ""}${seg.score.toFixed(3)}`}
          >
            {seg.text}
          </mark>
        );
      })}
    </p>
  );
}

export default function Explanation({ result }) {
  const genres = Object.keys(result.explanation || {});
  const [active, setActive] = useState(genres[0]);
  useEffect(() => setActive(genres[0]), [result]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!genres.length || !active || !result.explanation[active]) return null;
  const exp = result.explanation[active];
  const isIG = exp.method === "integrated_gradients";
  return (
    <section className="card" aria-labelledby="explain-h">
      <h2 id="explain-h" className="label mb-3">Why these genres?</h2>
      <div className="mb-4 flex flex-wrap gap-2" role="tablist">
        {genres.map((g) => (
          <button
            key={g}
            role="tab"
            aria-selected={g === active}
            onClick={() => setActive(g)}
            className={`rounded-full px-3 py-1 text-sm font-medium transition ${
              g === active
                ? "bg-amber-700 text-white"
                : "bg-stone-100 text-stone-700 hover:bg-stone-200 dark:bg-stone-800 dark:text-stone-300"
            }`}
          >
            {g}
          </button>
        ))}
      </div>
      <Highlighted text={result.input} wordScores={exp.word_scores} />
      {exp.top_words.length > 0 ? (
        <div className="mt-4">
          <h3 className="label mb-2">Most influential words for {active}</h3>
          <ol className="flex flex-wrap gap-2">
            {exp.top_words.map((w) => (
              <li key={w.word} className="chip gap-1.5">
                <span className="font-medium">{w.word}</span>
                <span className="tabular-nums text-stone-400">+{w.score.toFixed(2)}</span>
              </li>
            ))}
          </ol>
        </div>
      ) : (
        <p className="mt-4 text-sm text-stone-500">
          No word in the input pushes this genre up; the score comes from the model's intercept (its prior for {active}).
        </p>
      )}
      <p className="mt-4 text-xs text-stone-500">
        {isIG
          ? `Integrated Gradients (${exp.n_steps} steps) on the DistilBERT embeddings, summed to words. `
          : "Exact attributions for the linear TF-IDF model: coefficient × TF-IDF weight of each n-gram, split across its words (equal to Integrated Gradients for a linear model). "}
        Orange words raise the genre's score; grey words lower it. Attributions show what the model used,
        not whether that reasoning is correct.
      </p>
    </section>
  );
}
