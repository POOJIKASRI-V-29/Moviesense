# MovieSense — Project Requirements

Extracted from `project-report.pdf` (Team 28, Slot G2+TG2, Poojikasri V., 24BDS0139, October 2026).
Section numbers below refer to the report. Nothing here is invented; where the report leaves a
choice open, the choice made during implementation is marked **[Impl. decision]**.

## 1. Purpose

A text branch for the existing poster-based "Movie Genre Classification and Cast Recognition"
project. The only required input is a **plot sentence or short synopsis** (§1, §3).

## 2. Functional requirements (§3, §9.2, §10.2)

| ID | Requirement | Report ref |
|----|-------------|-----------|
| F1 | Predict every applicable genre from a fixed genre set, with a confidence score per genre | §3.1, §9.2 stage 4 |
| F2 | Per-genre decision thresholds τ_g tuned on validation; if none passes, return the top genre flagged "low confidence" | §5.3, §9.2 stage 4 |
| F3 | Show near-threshold genres for transparency | §9.2 stage 5, §11 |
| F4 | Retrieve the k most similar movies by cosine similarity over embedded plot summaries, optional genre filter | §3.2, §9.2 stages 6–7 |
| F5 | Cast is **retrieved, never predicted**: return cast only from `character.metadata.tsv` rows of matched/neighbour records, each tied to its source `wiki_id` | §3.3, §9.3 |
| F6 | Declare an input match only when cosine ≥ τ_match (set conservatively on validation); otherwise input cast = "unavailable" | §9.2 stage 7, §13.3 |
| F7 | Token-level explanations per predicted genre (Integrated Gradients for DistilBERT, aggregated to words); attention weights are not used | §9.4 |
| F8 | Structured JSON output: genres, scores, near-threshold, input_match, input_cast, similar_movies (with cast and source id), explanation | §11 |
| F9 | Input validation: Unicode NFC, control-char removal, whitespace collapse; reject empty, > 2,000 chars, or no alphabetic content | §9.2 stage 2 |

## 3. Data requirements (§7)

- **Dataset:** CMU Movie Summary Corpus (Bamman, O'Connor, Smith, ACL 2013).
  Official source: <https://www.cs.cmu.edu/~ark/personas/> → `MovieSummaries.tar.gz`
  (48,002,242 bytes, verified via HTTP HEAD on 2026-10-15). Licence: CC BY-SA.
- **Files used:** `plot_summaries.txt`, `movie.metadata.tsv`, `character.metadata.tsv`.
  The CoreNLP supplement (628 MB) is **not** required.
- **Data minimisation:** from character metadata load only wiki_id, character name, actor name.
  Actor DOB, gender, height, ethnicity, age are **not** loaded (§7.2, §15.2).
- **Join:** inner join summaries ↔ metadata on Wikipedia movie ID for the classification set;
  all metadata rows remain available to the metadata store (§7.3).
- **Cleaning (§7.4):** drop missing genres (kept in store); drop plots below a minimum token count
  chosen from the length distribution; exact-duplicate plots removed by hashing normalised text;
  near-duplicates/remakes grouped by embedding cosine above a high threshold; duplicate titles kept,
  disambiguated by year; documented genre mapping table merges Freebase synonyms; rows without an
  actor name not shown as cast.
- **Labels (§7.5):** coarse target set (~15–25 genres) defined after EDA; rare genres dropped or
  merged; multi-hot encoding with `MultiLabelBinarizer`, fixed and versioned column order.
- **Splits (§7.6):** 70/15/15, iterative multi-label stratification, group-aware (group =
  normalised title ∪ near-duplicate cluster), fixed seed, split ID lists saved.
- **Leakage (§7.7):** TF-IDF vocabulary fitted on train only; thresholds on validation only;
  test used once for final numbers; query excluded from its own neighbours; measure how often titles
  appear in summaries and mask titles if frequent.
- **Short-text augmentation (§7.8):** each training example also presented as random windows of 1–3
  sentences inheriting the movie's labels.

## 4. Models (§5, §10, §13.2)

- **Baseline:** TF-IDF (word 1–2-grams, vocab fitted on train only) + one-vs-rest Logistic
  Regression. Same splits, labels, inputs (incl. short windows), and threshold tuning as the
  transformer.
- **Proposed:** DistilBERT encoder, mean pooling over non-padding tokens → Linear(768, G) → sigmoid.
  BCEWithLogits with `pos_weight`, AdamW, linear warm-up, early stopping on validation macro-F1,
  per-genre thresholds `τ_g = argmax_t F1_g(val, t)`. Max 512 tokens.
- **Artefact:** versioned bundle of tokenizer, encoder weights, head, label list, thresholds, data hash.

## 5. Retrieval and store (§9.1 T7, §10.3)

- Embed every plot with the trained encoder (Sentence-BERT kept as fallback option).
- Exact cosine search in memory (~42k vectors); FAISS not needed.
- Metadata store: SQLite (movies, genres, cast), parameterised queries.

## 6. Evaluation (§13)

- Classification: micro-F1, **macro-F1 (primary)**, micro/macro precision & recall, per-genre F1,
  Hamming loss.
- Conditions: full summary, first sentence, random sentence. **Single-sentence is primary** (§7.8).
- Variance: three seeds per model (mean ± sd), bootstrap 95% CI on test macro-F1.
- Hypothesis H1: DistilBERT macro-F1 > TF-IDF on single-sentence inputs; accept only if the CI of the
  difference excludes zero.
- Retrieval: genre-overlap Precision@k, mean Jaccard@k (query excluded), cast-attribution precision
  (sets τ_match).
- Explanations: deletion test — removing top-attributed words should reduce the genre score more
  than removing random words.
- Poster-branch comparison: qualitative only (Table 3) unless posters are available for test films.

## 7. Non-functional / ethical (§15.2)

- Never fabricate cast. Every cast name traceable to a source wiki_id.
- API: length limits and validation, parameterised SQL, rate limiting, no logging of raw user text.
- CC BY-SA attribution preserved in code, docs and derived data. Raw data not committed.

## 8. Technology stack (§12, Table 11)

Python 3, pandas, NumPy, Hugging Face Transformers, PyTorch, scikit-learn, iterative-stratification,
Captum, SQLite, FastAPI (+ Docker optional).
**[Impl. decision]** The report does not specify a frontend; per the user's instructions a
React + Vite + Tailwind CSS UI is added on top of the FastAPI service.

## 9. Out of scope (stated by the report)

- Late fusion with the poster branch (future work, §15.3).
- Numeric comparison with the poster branch (no shared posters available).
- PostgreSQL (SQLite is the prototype store).
