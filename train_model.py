"""
train_model.py
----------------
Builds the movie recommendation model from the TMDB 5000 dataset
(tmdb_5000_movies.csv and tmdb_5000_credits.csv must be in the same folder
as this script) and saves the trained artifacts as pickle files:

    movies.pkl        -> cleaned dataframe (movie_id, title, tags)
    similarity.pkl     -> cosine similarity matrix

Run this ONCE (from PyCharm's terminal, or by right-clicking > Run):
    python train_model.py

After it finishes, run the frontend with:
    streamlit run app.py
"""

import ast
import os
import pickle

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_CSV = os.path.join(BASE_DIR, "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join(BASE_DIR, "tmdb_5000_credits.csv")


def convert(obj):
    """Turn a JSON-like string of {id, name} dicts into a list of names."""
    return [i["name"] for i in ast.literal_eval(obj)]


def convert_cast(obj, limit=3):
    """Keep only the first `limit` cast member names."""
    names = []
    for i, item in enumerate(ast.literal_eval(obj)):
        if i == limit:
            break
        names.append(item["name"])
    return names


def fetch_director(obj):
    """Pull out the director's name from the crew list."""
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            return [i["name"]]
    return []


def stem_text(text, stemmer):
    return " ".join(stemmer.stem(word) for word in text.split())


def build_model():
    if not (os.path.exists(MOVIES_CSV) and os.path.exists(CREDITS_CSV)):
        raise FileNotFoundError(
            "Couldn't find tmdb_5000_movies.csv / tmdb_5000_credits.csv in:\n"
            f"  {BASE_DIR}\n"
            "Download them and place both CSVs in the same folder as this script."
        )

    print("Loading datasets...")
    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    movies = movies.merge(credits, on="title")
    movies = movies[["genres", "movie_id", "keywords", "overview", "title", "cast", "crew"]]
    movies.dropna(inplace=True)

    print("Extracting genres, keywords, cast, and director...")
    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    # remove spaces inside multi-word names/tags so "Sam Worthington" -> "SamWorthington"
    for col in ["genres", "keywords", "cast", "crew"]:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    movies["tags"] = (
        movies["overview"] + movies["keywords"] + movies["genres"] + movies["cast"] + movies["crew"]
    )

    new_df = movies[["movie_id", "title", "tags"]].copy()
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())

    print("Stemming tags...")
    ps = PorterStemmer()
    new_df["tags"] = new_df["tags"].apply(lambda x: stem_text(x, ps))

    print("Vectorizing and computing similarity matrix (this may take a moment)...")
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    new_df.reset_index(drop=True, inplace=True)

    print("Saving movies.pkl and similarity.pkl...")
    pickle.dump(new_df, open(os.path.join(BASE_DIR, "movies.pkl"), "wb"))
    pickle.dump(similarity, open(os.path.join(BASE_DIR, "similarity.pkl"), "wb"))

    print(f"Done! Model built for {len(new_df)} movies.")


if __name__ == "__main__":
    build_model()
