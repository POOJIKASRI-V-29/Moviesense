import MovieCard from "./MovieCard.jsx";

export default function SimilarMovies({ title, subtitle, movies, onMoreLike }) {
  return (
    <section aria-labelledby="similar-h">
      <div className="mb-3">
        <h2 id="similar-h" className="label">{title}</h2>
        {subtitle && <p className="mt-1 text-sm text-stone-500">{subtitle}</p>}
      </div>
      {movies.length === 0 ? (
        <p className="card text-sm text-stone-500">No similar movies found with the current filter.</p>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {movies.map((m) => (
            <MovieCard key={m.wiki_id} movie={m} onMoreLike={onMoreLike} />
          ))}
        </div>
      )}
    </section>
  );
}
