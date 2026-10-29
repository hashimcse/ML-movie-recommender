"""CineMatch — thoughtful movie discovery. Run: python -m streamlit run app.py"""

from concurrent.futures import ThreadPoolExecutor
from html import escape
import os
import random

import streamlit as st

from posters import fetch_poster
from recommender import BASE_DIR, MODEL_PATH, Catalog, Match, Movie, load_catalog
from train_model import build_model
from ui import hero_html, movie_meta, poster_html, watchlist_csv

st.set_page_config(page_title="CineMatch · Find your next favorite", page_icon="✦", layout="wide")
st.markdown(f"<style>{(BASE_DIR / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_catalog(model_modified: int) -> Catalog:
    return load_catalog() if MODEL_PATH.exists() else build_model()


@st.cache_data(ttl=900, max_entries=128, show_spinner=False)
def poster_batch(movie_ids: tuple[int, ...], api_key: str) -> dict[int, str | None]:
    if not api_key:
        return {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        urls = list(pool.map(lambda movie_id: fetch_poster(movie_id, api_key), movie_ids))
    return dict(zip(movie_ids, urls))


def configured_key() -> str:
    if st.session_state.get("poster_key", "").strip():
        return st.session_state.poster_key.strip()
    if os.environ.get("TMDB_API_KEY"):
        return os.environ["TMDB_API_KEY"].strip()
    try:
        return str(st.secrets.get("TMDB_API_KEY", "")).strip()
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return ""


def choose_movie(movie_id: int) -> None:
    st.session_state.selected_id = movie_id
    st.session_state.query = ""
    st.session_state.view = "Discover"


def clear_selection() -> None:
    st.session_state.selected_id = None


def start_over() -> None:
    clear_selection()
    st.session_state.query = ""


def reset_filters() -> None:
    st.session_state.genre = "All genres"
    st.session_state.min_rating = 0
    st.session_state.runtime = 0


def toggle_watchlist(movie_id: int) -> None:
    saved = list(st.session_state.watchlist)
    if movie_id in saved:
        saved.remove(movie_id)
    else:
        saved.append(movie_id)
    st.session_state.watchlist = saved


def empty_state(title: str, message: str, symbol: str = "✦") -> None:
    st.markdown(f'<div class="empty-state"><div class="empty-symbol">{escape(symbol)}</div>'
                f'<h3>{escape(title)}</h3><p>{escape(message)}</p></div>', unsafe_allow_html=True)


def section(title: str, subtitle: str, kicker: str = "THE SHORTLIST", count: str = "") -> None:
    st.markdown(f'<div class="section-heading"><div><div class="section-kicker">{escape(kicker)}</div>'
                f'<h2>{escape(title)}</h2><p>{escape(subtitle)}</p></div>'
                + (f'<span class="section-count">{escape(count)}</span>' if count else "")
                + '</div>', unsafe_allow_html=True)


def render_cards(movies: list[Movie], api_key: str, prefix: str,
                 matches: dict[int, Match] | None = None) -> None:
    posters = poster_batch(tuple(movie.movie_id for movie in movies), api_key)
    for start in range(0, len(movies), 6):
        columns = st.columns(6, gap="small")
        for column, movie in zip(columns, movies[start:start + 6]):
            match = (matches or {}).get(movie.movie_id)
            with column, st.container(key=f"film_{prefix}_{movie.movie_id}"):
                st.markdown(poster_html(movie, posters.get(movie.movie_id)), unsafe_allow_html=True)
                st.markdown(f'<div class="film-title">{escape(movie.title)}</div>'
                            f'<div class="film-meta">{escape(movie_meta(movie))}</div>'
                            f'<div class="film-rating"><span>{f"★ {movie.rating:.1f}" if movie.vote_count else "Not rated"}</span>'
                            f'<small>{movie.vote_count:,} ratings</small></div>', unsafe_allow_html=True)
                if match:
                    st.markdown('<div class="match-reason">' + '<br>'.join(escape(reason) for reason in match.reasons)
                                + '</div>', unsafe_allow_html=True)
                with st.expander("About this film"):
                    st.write(movie.overview or "A synopsis is not available for this film.")
                    if movie.directors:
                        st.caption("Director: " + ", ".join(movie.directors))
                    if movie.cast:
                        st.caption("Starring: " + ", ".join(movie.cast))
                    if match:
                        st.caption(f"Content similarity: {match.score:.0%}. This compares movie features; it is not a predicted rating.")
                    st.markdown(f"[View on TMDB ↗](https://www.themoviedb.org/movie/{movie.movie_id})")
                st.button("More like this ↗", key=f"more_{prefix}_{movie.movie_id}",
                          on_click=choose_movie, args=(movie.movie_id,), width="stretch")
                saved = movie.movie_id in st.session_state.watchlist
                st.button("✓ Saved · remove" if saved else "+ Watchlist", key=f"save_{prefix}_{movie.movie_id}",
                          on_click=toggle_watchlist, args=(movie.movie_id,), width="stretch",
                          help=f"{'Remove' if saved else 'Save'} {movie.title} {'from' if saved else 'to'} your watchlist")


def filter_values() -> dict:
    return {"genre": None if st.session_state.genre == "All genres" else st.session_state.genre,
            "min_rating": st.session_state.min_rating,
            "max_runtime": st.session_state.runtime or None}


def surprise(catalog: Catalog) -> None:
    candidates = catalog.discover(**filter_values(), limit=100)
    candidates = [movie for movie in candidates if movie.movie_id != st.session_state.selected_id]
    if candidates:
        choose_movie(random.choice(candidates).movie_id)
    else:
        st.session_state.surprise_empty = True


def render_filters(catalog: Catalog) -> None:
    with st.container(key="filter_panel"):
        genre, rating, runtime, reset = st.columns([2.2, 1.5, 1.5, 1])
        with genre:
            st.selectbox("Genre", ["All genres", *catalog.genres], key="genre")
        with rating:
            st.selectbox("Minimum rating", [0, 6, 7, 8], key="min_rating",
                         format_func=lambda value: "Any rating" if not value else f"{value}.0 and up")
        with runtime:
            st.selectbox("Time for a movie", [0, 90, 120, 150], key="runtime",
                         format_func=lambda value: "Any length" if not value else f"{value} min or less")
        with reset:
            st.button("Reset", key="reset_filters", on_click=reset_filters, width="stretch")


def render_discover(catalog: Catalog, api_key: str) -> None:
    if st.session_state.selected_id is None and not st.session_state.query:
        st.markdown(hero_html(), unsafe_allow_html=True)
    with st.container(key="search_panel"):
        search_column, random_column = st.columns([4.7, 1.3])
        with search_column:
            query = st.text_input("Start with a movie you love", key="query", on_change=clear_selection,
                                  placeholder="Try Interstellar, The Dark Knight, or a favorite of your own…")
        with random_column:
            st.button("Surprise me ✦", key="surprise", on_click=surprise, args=(catalog,), width="stretch")
    if st.session_state.pop("surprise_empty", False):
        st.info("No new films fit these filters. Try another genre or reset your filters.")
    if query.strip():
        hits = catalog.search(query)
        section("Which one did you have in mind?", "Choose a film to discover its kindred spirits.", "FIND YOUR FILM")
        if hits:
            for start in range(0, len(hits), 3):
                for column, movie in zip(st.columns(3), hits[start:start + 3]):
                    with column:
                        st.button(movie.label, key=f"pick_{movie.movie_id}", on_click=choose_movie,
                                  args=(movie.movie_id,), width="stretch")
            st.caption("Partial titles and spelling slips are welcome. Years help distinguish remakes.")
        else:
            empty_state("That one’s not in our collection.", "Try a shorter title or a different film. Our catalog is the TMDB 5000 collection.", "⌕")
        st.button("Back to discovery", on_click=start_over)
        return

    selected = catalog.by_id.get(st.session_state.selected_id)
    if selected:
        st.markdown(f'<div class="selected-film"><div class="eyebrow">BECAUSE YOU LOVE</div>'
                    f'<h2>{escape(selected.title)}</h2><div class="film-meta">{escape(movie_meta(selected))}</div>'
                    f'<p>{escape(selected.overview or "Let’s find more films like this one.")}</p></div>', unsafe_allow_html=True)
        back, save, _ = st.columns([1.3, 1.3, 3])
        with back:
            st.button("← Explore all films", on_click=start_over, width="stretch")
        with save:
            saved = selected.movie_id in st.session_state.watchlist
            st.button("✓ Saved · remove" if saved else "+ Save this film", key="save_selected",
                      on_click=toggle_watchlist, args=(selected.movie_id,), width="stretch")
    render_filters(catalog)
    filters = filter_values()
    if selected:
        matches = catalog.recommend(selected.movie_id, **filters, limit=6)
        section("Keep the feeling going.", "Connected by story, genre, and the people behind the camera.",
                "CHOSEN FOR YOU", f"{len(matches)} matches")
        if matches:
            render_cards([match.movie for match in matches], api_key, "matches",
                         {match.movie.movie_id: match for match in matches})
        else:
            empty_state("Let’s widen the lens.", "No similar films fit all your filters. Reset them to see more possibilities.")
    else:
        filtered = any(filters.values())
        featured_ids = [27205, 120467, 244786, 76341, 129, 13]
        picks = (catalog.discover(**filters, limit=6) if filtered else
                 [catalog.by_id[movie_id] for movie_id in featured_ids if movie_id in catalog.by_id])
        if not picks and not filtered:
            picks = catalog.discover(limit=6)
        section("A good place to start.", "Find your kind of film, then follow the trail." if not filtered else
                "Highly rated films that fit your mood and your evening.", "WORTH YOUR EVENING", f"{len(picks)} films")
        if picks:
            render_cards(picks, api_key, "discover")
        else:
            empty_state("Nothing here just yet.", "Try a different genre, a lower rating, or a little more time.")


def render_watchlist(catalog: Catalog, api_key: str) -> None:
    saved = [catalog.by_id[movie_id] for movie_id in st.session_state.watchlist if movie_id in catalog.by_id]
    section("For a future movie night.", "All the films you’ve set aside, in one little collection.",
            "YOUR WATCHLIST", f"{len(saved)} saved")
    if saved:
        st.download_button("Download watchlist ↓", watchlist_csv(saved), "cinematch-watchlist.csv", "text/csv")
        st.caption("Your list stays with this browser session. Download it to keep a copy before you leave.")
        render_cards(saved, api_key, "watchlist")
    else:
        empty_state("Good movies are worth saving.", "Tap + Watchlist on any film. Your next movie night will thank you.", "+")
        st.button("Discover a film ↗", on_click=lambda: st.session_state.update(view="Discover"), type="primary")


def main() -> None:
    defaults = {"selected_id": None, "query": "", "view": "Discover", "watchlist": [],
                "genre": "All genres", "min_rating": 0, "runtime": 0}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.markdown('<div class="masthead"><div class="brand"><span class="brand-mark">✦</span> cinematch</div>'
                '<div class="masthead-note">A FILM FOR EVERY FEELING.</div></div>', unsafe_allow_html=True)
    try:
        with st.spinner("Setting the scene…"):
            catalog = get_catalog(MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.exists() else 0)
    except (OSError, ValueError) as exc:
        st.error("The movie collection could not be loaded. Rebuild it with `python train_model.py`.")
        with st.expander("Setup details"):
            st.code(str(exc))
        st.stop()
    if st.session_state.selected_id not in catalog.by_id:
        st.session_state.selected_id = None
    watchlist_label = f"Watchlist ({len(st.session_state.watchlist)})"
    st.radio("Navigation", ["Discover", "Watchlist"], horizontal=True, key="view",
             format_func=lambda value: watchlist_label if value == "Watchlist" else value,
             label_visibility="collapsed")
    api_key = configured_key()
    if st.session_state.view == "Watchlist":
        render_watchlist(catalog, api_key)
    else:
        render_discover(catalog, api_key)
    st.write("")
    with st.expander("A peek behind the curtain"):
        st.write(f"CineMatch compares plot descriptions, genres, keywords, the top three cast members, and directors across {len(catalog.movies):,} films. Start with one you love, and we’ll find movies with similar features.")
        st.caption("Ratings and film details come from the historical TMDB 5000 dataset, not live rankings. This collection does not include every movie or new release. Discovery ratings account for vote counts. Recommendations are based on content, not your viewing history.")
    with st.expander("Poster settings"):
        st.caption("Our illustrated covers work offline. Add an optional TMDB API key for official posters. It is used only by the server to request movie images.")
        st.text_input("TMDB API key", type="password", key="poster_key", placeholder="Optional")
        if api_key:
            st.caption("Poster requests are enabled. Illustrated covers are used if TMDB is unavailable.")
    st.markdown(f'<footer class="site-footer"><span>✦ CINEMATCH &nbsp; / &nbsp; MADE FOR MOVIE NIGHTS</span>'
                f'<span>{len(catalog.movies):,} films · Movie metadata from TMDB · Not endorsed by TMDB</span></footer>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
