import { CastList } from "./MovieCard.jsx";

export default function InputCast({ result }) {
  const match = result.input_match;
  return (
    <section className="card" aria-labelledby="cast-h">
      <h2 id="cast-h" className="label mb-3">Cast for your plot</h2>
      {match ? (
        <>
          <p className="mb-3 text-sm">
            Your text closely matches the plot of{" "}
            <strong>
              {match.title} {match.year ? `(${match.year})` : ""}
            </strong>{" "}
            (similarity {match.similarity.toFixed(2)} ≥ τ<sub>match</sub> {result.tau_match.toFixed(2)}), so its recorded cast is shown.
          </p>
          {Array.isArray(result.input_cast) ? (
            <CastList cast={result.input_cast} limit={12} />
          ) : (
            <p className="text-sm italic text-stone-500">{result.input_cast}</p>
          )}
          <p className="mt-3 text-xs text-stone-500">Source record: wiki_id {match.wiki_id}</p>
        </>
      ) : (
        <>
          <p className="text-base font-medium">Unavailable</p>
          <p className="mt-1 text-sm text-stone-600 dark:text-stone-400">{result.input_cast}.</p>
          <p className="mt-3 text-xs text-stone-500">
            MovieSense never generates actor names. Cast is only shown when it comes from a matched record in the
            CMU metadata; the similar movies below list the cast of those films, not a predicted cast for your plot.
          </p>
        </>
      )}
    </section>
  );
}
