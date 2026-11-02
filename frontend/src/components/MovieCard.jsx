export function CastList({ cast, limit = 6 }) {
  if (!cast || cast.length === 0) {
    return <p className="text-sm italic text-stone-500">No cast listed in character.metadata.tsv for this record.</p>;
  }
  return (
    <ul className="flex flex-wrap gap-x-3 gap-y-1 text-sm">
      {cast.slice(0, limit).map((c, i) => (
        <li key={`${c.actor}-${i}`}>
          <span className="font-medium">{c.actor}</span>
          {c.character && <span className="text-stone-500"> as {c.character}</span>}
        </li>
      ))}
      {cast.length > limit && <li className="text-stone-400">+{cast.length - limit} more</li>}
    </ul>
  );
}

export default function MovieCard({ movie, onMoreLike }) {
  return (
    <article className="card flex flex-col gap-3">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-display text-lg font-semibold leading-tight">
            <span className="mr-2 text-stone-400">{movie.rank}.</span>
            {movie.title || "Untitled record"}
            {movie.year && <span className="ml-1 font-sans text-base font-normal text-stone-500">({movie.year})</span>}
          </h3>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {movie.genres.map((g) => (
              <span key={g} className="chip">{g}</span>
            ))}
          </div>
        </div>
        <div className="shrink-0 text-right">
          <div className="text-lg font-semibold tabular-nums">{movie.similarity.toFixed(2)}</div>
          <div className="text-xs text-stone-500">cosine</div>
        </div>
      </header>
      <p className="text-sm leading-relaxed text-stone-600 dark:text-stone-400">{movie.plot_excerpt}</p>
      <div>
        <h4 className="label mb-1">Cast of this movie</h4>
        <CastList cast={movie.cast} />
      </div>
      <footer className="mt-auto flex items-center justify-between border-t border-stone-100 pt-3 text-xs text-stone-500 dark:border-stone-800">
        <span>Source: CMU corpus · wiki_id {movie.wiki_id}</span>
        {onMoreLike && (
          <button className="btn-ghost" onClick={() => onMoreLike(movie)}>
            More like this →
          </button>
        )}
      </footer>
    </article>
  );
}
