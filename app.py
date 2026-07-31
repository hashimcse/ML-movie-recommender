"""
app.py
------
Streamlit frontend for the movie recommender.

Run from PyCharm's terminal (make sure train_model.py has been run first,
so movies.pkl and similarity.pkl exist in this folder):

    streamlit run app.py

This opens the app in your browser automatically.
"""

import difflib
import os
import pickle

import requests
import streamlit as st

# TMDB API key: checked in this order, no key needs to be hardcoded in this file:
#   1. TMDB_API_KEY environment variable (if you set one on your machine)
#   2. Typed into the sidebar box when the app is running
# Get a free key at https://www.themoviedb.org/settings/api
DEFAULT_API_KEY = os.environ.get("TMDB_API_KEY", "")

# Always look for the pickle files next to THIS script, regardless of
# what folder Streamlit/PyCharm was launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_PKL = os.path.join(BASE_DIR, "movies.pkl")
SIMILARITY_PKL = os.path.join(BASE_DIR, "similarity.pkl")


@st.cache_resource
def load_data():
    if not (os.path.exists(MOVIES_PKL) and os.path.exists(SIMILARITY_PKL)):
        st.error(
            "Model files not found (movies.pkl / similarity.pkl).\n\n"
            f"Expected them in: {BASE_DIR}\n\n"
            "Run `python train_model.py` from that same folder first "
            "(with tmdb_5000_movies.csv and tmdb_5000_credits.csv present), "
            "then restart this app."
        )
        st.stop()
    movies = pickle.load(open(MOVIES_PKL, "rb"))
    similarity = pickle.load(open(SIMILARITY_PKL, "rb"))
    return movies, similarity


def fetch_poster(movie_id, api_key):
    if not api_key:
        return "https://placehold.co/300x450?text=No+Poster"
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        data = requests.get(url, timeout=5).json()
        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
    except Exception:
        pass
    return "https://placehold.co/300x450?text=No+Poster"


def filter_titles(query, all_titles, limit=10):
    """
    Live search: returns a ranked list of movie titles matching the query,
    used to populate a dropdown that updates as the user types.
    """
    query_clean = query.strip().lower()
    if not query_clean:
        return []

    title_lookup = {t.strip().lower(): t for t in all_titles}

    # 1) Substring matches first (e.g. "batman" -> "Batman Begins", "Batman Returns")
    substring_hits = [t for t in all_titles if query_clean in t.lower()]
    substring_hits.sort(key=len)  # shortest/closest titles first

    if substring_hits:
        return substring_hits[:limit]

    # 2) Fall back to fuzzy matches for typos (e.g. "the dark knihgt")
    close = difflib.get_close_matches(
        query_clean, [t.lower() for t in all_titles], n=limit, cutoff=0.5
    )
    return [title_lookup[c] for c in close]


def recommend(movie, movies, similarity, api_key):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]

    top_5 = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)[1:6]

    names, posters = [], []
    for idx, score in top_5:
        row = movies.iloc[idx]
        names.append(row.title)
        posters.append(fetch_poster(row.movie_id, api_key))
    return names, posters


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommender System")
st.write("Start typing a movie name, then pick from the matching options.")

with st.sidebar:
    st.subheader("Settings")
    api_key_override = st.text_input(
        "TMDB API key (optional override, for posters)",
        value="",
        type="password",
        help="Leave blank to use the app's configured key automatically. "
             "Only fill this in if you want to use your OWN key instead "
             "of the deployed one.",
    )
    # Never pre-fill this field with the real key — Streamlit sends the
    # widget's default value to the browser even for password fields, so
    # pre-filling it would leak the key to anyone viewing the page's
    # network traffic. The real key is used silently below instead.
    api_key = api_key_override or DEFAULT_API_KEY

movies, similarity = load_data()
movie_titles = movies["title"].values

query = st.text_input("Search for a movie", placeholder="e.g. bat, dark knihgt, avngers...")

selected_movie = None
if query.strip():
    matches = filter_titles(query, movie_titles)
    if matches:
        selected_movie = st.selectbox(
            f"{len(matches)} match(es) found — select one",
            matches,
        )
    else:
        st.warning(
            f"No movie matching **{query}** was found in the dataset. "
            "Check the spelling or try a shorter/partial name."
        )
else:
    st.caption("Start typing above to see matching movies appear here.")

if selected_movie and st.button("Recommend", type="primary"):
    st.subheader(f"Because you searched for: {selected_movie}")
    names, posters = recommend(selected_movie, movies, similarity, api_key)

    cols = st.columns(5)
    for col, name, poster in zip(cols, names, posters):
        with col:
            st.image(poster, use_container_width=True)
            st.caption(name)
