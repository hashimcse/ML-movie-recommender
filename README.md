# CineMatch

A movie recommendation app built with Python, Streamlit, and a content-based TF-IDF model. Start with a film you love, discover similar movies, narrow the results, and save a shortlist for later.

## Run locally

From this `code` folder, with Python 3.14 (the tested version):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. Open the local URL printed by Streamlit, usually http://localhost:8501.

Both TMDB 5000 CSVs are included. The first launch builds `data/cinematch.npz` automatically if it is missing. Subsequent launches load that artifact. No API key or external network connection is needed for search, recommendations, filters, or the illustrated covers.

## What you can do

- Search partial titles, accents, and common spelling mistakes. Remakes remain separate films, identified by TMDB ID and displayed with a year.
- Discover related films with explanations such as a shared director, cast member, genre, or theme.
- Filter by genre, minimum dataset rating, and maximum runtime. Filters are applied before selecting the top results.
- Use **More like this** to follow a movie trail, or **Surprise me** for a pick that respects your filters.
- Save and remove films in a watchlist without losing your recommendations. The watchlist lasts for the current Streamlit browser session; download the CSV to keep it after a refresh, disconnect, or server restart.
- Read synopses, cast, directors, and TMDB links on each card.

## Optional official posters

Set `TMDB_API_KEY` as a server environment variable, put it in `.streamlit/secrets.toml`, or enter a key under **Poster settings**. Do not commit credentials. Example secrets file:

```toml
TMDB_API_KEY = "your-api-key"
```

Requests happen server-side with bounded timeouts, up to six requests in parallel, and a 15-minute cache. Only public TMDB image URLs reach the browser. Failed or missing poster responses use the original illustrated covers. Recommendations still work when TMDB is unavailable. Movie data is from TMDB; this app is not endorsed or certified by TMDB.

## Train or rebuild

```powershell
python train_model.py
```

Custom input/output paths are supported:

```powershell
python train_model.py --movies path/to/tmdb_5000_movies.csv --credits path/to/tmdb_5000_credits.csv --output data/cinematch.npz
```

Run training again after changing the CSVs. The app notices a replaced model on its next rerun. Missing files or required columns produce actionable errors.

The pipeline joins credits by **movie ID**, not title, and retains films with missing optional metadata. It makes separate TF-IDF feature blocks for overview (weight 1.0), genres (1.1), keywords (1.4), the top three cast members (1.0), and directors (1.2), then normalizes the concatenated sparse matrix. Recommendations use cosine similarity for one selected movie at a time; there is no full dense movie-by-movie matrix. Self-matches and zero-similarity results are excluded. Ties are deterministic.

The catalog and feature matrix are written atomically into one compressed NumPy artifact, loaded with `allow_pickle=False`. The bundled 4,803-film dataset produces an artifact of approximately 2.6 MB. Previous `movies.pkl`, `similarity.pkl`, and `recommendations.pkl` files are no longer used; existing copies are left untouched. The old notebook is retained as a historical experiment, not the production training pipeline.

**Limitations:** this historical dataset does not include all films or new releases. Ratings are dataset snapshots, not live ratings. Similarity describes shared movie content; it is not a probability that a user will like a movie. Discovery uses a Bayesian-adjusted rating with a 200-vote prior to avoid promoting films based on very few votes. This is not collaborative filtering and has not been evaluated against user preference data.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The suite checks ID joins and remakes, sparse artifact roundtrips, search, filter ordering, malformed optional metadata, poster failures, escaped titles, CSV export, and Streamlit interactions including watchlist persistence across reruns. Poster requests are mocked; tests do not need a TMDB key.

## Hosting

Build command:

```sh
pip install -r requirements.txt && python train_model.py
```

Start command (use `code` as the service root):

```sh
python -m streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT:-8501}
```

The health endpoint is `/_stcore/health`. `.streamlit/config.toml` supplies the theme and headless configuration. The existing `procfile` contains an equivalent command; hosts that auto-detect a Procfile require the filename `Procfile` with a capital P. `setup.sh` builds the model and no longer writes configuration to the home directory or disables CORS. Keep Streamlit's CORS and XSRF protections enabled.

## Project layout

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit pages, controls, session state, and poster caching |
| `recommender.py` | CSV preparation, search, ranking, filters, artifact I/O |
| `train_model.py` | Reproducible CLI training entry point |
| `posters.py` | Optional, defensive TMDB poster requests |
| `ui.py` | Escaped HTML presentation and watchlist CSV export |
| `assets/style.css` | Responsive cinema-inspired interface |
| `tests/` | Model and app regression tests |
