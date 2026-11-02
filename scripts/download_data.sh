#!/usr/bin/env bash
# Download the CMU Movie Summary Corpus (Bamman, O'Connor & Smith, ACL 2013).
# Official page: https://www.cs.cmu.edu/~ark/personas/  — licence: CC BY-SA.
# Only the three files MovieSense needs are kept; the tarball is removed afterwards.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
URL="http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
EXPECTED_BYTES=48002242

mkdir -p "$RAW"
if [[ -f "$RAW/plot_summaries.txt" && -f "$RAW/movie.metadata.tsv" && -f "$RAW/character.metadata.tsv" ]]; then
  echo "Corpus already present in $RAW"
  exit 0
fi

PART="$RAW/MovieSummaries.tar.gz.part"
echo "Downloading $URL ..."
# Resumable: a slow or dropped connection is retried and continues from the partial file.
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  SIZE=$( [[ -f "$PART" ]] && wc -c < "$PART" | tr -d ' ' || echo 0)
  [[ "$SIZE" == "$EXPECTED_BYTES" ]] && break
  echo "attempt $attempt (have $SIZE bytes)"
  curl -fL -C - --speed-limit 2000 --speed-time 30 -o "$PART" "$URL" || sleep 3
done

SIZE=$(wc -c < "$PART" | tr -d ' ')
if [[ "$SIZE" != "$EXPECTED_BYTES" ]]; then
  echo "Unexpected archive size: $SIZE (expected $EXPECTED_BYTES)" >&2
  exit 1
fi
shasum -a 256 "$PART" | awk '{print $1}' > "$RAW/MovieSummaries.tar.gz.sha256"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
tar -xzf "$PART" -C "$TMP" \
  MovieSummaries/plot_summaries.txt MovieSummaries/movie.metadata.tsv \
  MovieSummaries/character.metadata.tsv MovieSummaries/README.txt
mv "$TMP"/MovieSummaries/* "$RAW/"
rm -f "$PART"
echo "Done. Files in $RAW:"
ls -la "$RAW"
