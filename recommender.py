"""Content-based movie discovery, independent of the Streamlit interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

MODEL_VERSION = 1
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "data" / "cinematch.npz"


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"\w+", text))


def _records(value: object) -> list[dict]:
    """Missing or malformed optional metadata should not discard a movie."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _names(value: object) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        item["name"].strip() for item in _records(value)
        if isinstance(item.get("name"), str) and item["name"].strip()
    ))


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: object, default: float = 0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    overview: str = ""
    year: int | None = None
    genres: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    cast: tuple[str, ...] = ()
    directors: tuple[str, ...] = ()
    runtime: int = 0
    rating: float = 0
    vote_count: int = 0
    popularity: float = 0

    @property
    def label(self) -> str:
        return f"{self.title} ({self.year})" if self.year else self.title


@dataclass(frozen=True)
class Match:
    movie: Movie
    score: float
    reasons: tuple[str, ...]


def read_movies(movies_path: Path, credits_path: Path) -> list[Movie]:
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    for frame, required, name in (
        (movies, {"id", "title", "overview", "genres", "keywords"}, "Movies"),
        (credits, {"movie_id", "cast", "crew"}, "Credits"),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} CSV is missing columns: {', '.join(sorted(missing))}")
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    credits["movie_id"] = pd.to_numeric(credits["movie_id"], errors="coerce")
    movies = movies.dropna(subset=["id"]).drop_duplicates("id")
    credits = credits.dropna(subset=["movie_id"]).drop_duplicates("movie_id")
    # Titles are not unique: remakes must retain their own cast and crew.
    merged = movies.merge(
        credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id",
        how="left", validate="one_to_one",
    )
    result = []
    for row in merged.to_dict("records"):
        title = _text(row["title"])
        movie_id = _number(row["id"])
        if not title or movie_id <= 0 or not movie_id.is_integer():
            continue
        date = _text(row.get("release_date"))
        year = int(date[:4]) if re.match(r"^\d{4}-", date) else None
        directors = tuple(dict.fromkeys(
            item["name"].strip() for item in _records(row.get("crew"))
            if item.get("job") == "Director" and _text(item.get("name"))
        ))
        result.append(Movie(
            movie_id=int(movie_id), title=title, overview=_text(row.get("overview")),
            year=year, genres=_names(row.get("genres")), keywords=_names(row.get("keywords")),
            cast=_names(row.get("cast"))[:3], directors=directors,
            runtime=max(0, int(_number(row.get("runtime")))),
            rating=min(10, max(0, _number(row.get("vote_average")))),
            vote_count=max(0, int(_number(row.get("vote_count")))),
            popularity=max(0, _number(row.get("popularity"))),
        ))
    if len(result) < 2:
        raise ValueError("At least two movies with valid IDs and titles are required.")
    return result


def feature_matrix(movies: list[Movie]) -> sparse.csr_matrix:
    blocks = []
    # Separate vocabularies preserve whole genres and names. TF-IDF downweights
    # ubiquitous terms; weights control each field's contribution.
    for field, weight in (("overview", 1.0), ("genres", 1.1), ("keywords", 1.4),
                          ("cast", 1.0), ("directors", 1.2)):
        if field == "overview":
            documents = [movie.overview for movie in movies]
            vectorizer = TfidfVectorizer(
                stop_words="english", strip_accents="unicode", max_features=12000,
                sublinear_tf=True, dtype=np.float32,
            )
        else:
            documents = ["|".join(getattr(movie, field)).casefold() for movie in movies]
            vectorizer = TfidfVectorizer(token_pattern=r"[^|]+", lowercase=False, dtype=np.float32)
        try:
            blocks.append(vectorizer.fit_transform(documents) * weight)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
    if not blocks:
        raise ValueError("The dataset has no usable movie descriptions or metadata.")
    return normalize(sparse.hstack(blocks, format="csr"), copy=False).tocsr()


class Catalog:
    def __init__(self, movies: list[Movie], features: sparse.csr_matrix):
        if not movies or features.shape[0] != len(movies) or features.shape[1] == 0:
            raise ValueError("Model features do not match the movie catalog. Rebuild the model.")
        if len({movie.movie_id for movie in movies}) != len(movies):
            raise ValueError("Movie IDs must be unique.")
        if not np.isfinite(features.data).all():
            raise ValueError("Model contains invalid feature values. Rebuild the model.")
        features.check_format(full_check=True)
        self.movies = movies
        self.features = features
        self.by_id = {movie.movie_id: movie for movie in movies}
        self.positions = {movie.movie_id: index for index, movie in enumerate(movies)}
        self.titles = [normalize_title(movie.title) for movie in movies]
        self.genres = sorted({genre for movie in movies for genre in movie.genres})
        self.mean_rating = sum(movie.rating for movie in movies) / len(movies)

    def search(self, query: str, limit: int = 9) -> list[Movie]:
        query = normalize_title(query)
        if not query or limit <= 0:
            return []
        hits = []
        for movie, title in zip(self.movies, self.titles):
            if title == query:
                rank = 3.0
            elif query in title:
                rank = 2.0 + len(query) / max(1, len(title))
            else:
                ratio = SequenceMatcher(None, query, title).ratio()
                if ratio < 0.62 or len(query) < 3:
                    continue
                rank = ratio
            hits.append((rank, movie))
        hits.sort(key=lambda hit: (-hit[0], -hit[1].popularity, hit[1].movie_id))
        return [movie for _, movie in hits[:limit]]

    def quality(self, movie: Movie) -> float:
        # Bayesian rating prevents a single 10/10 vote dominating discovery.
        return (movie.vote_count * movie.rating + 200 * self.mean_rating) / (movie.vote_count + 200)

    @staticmethod
    def eligible(movie: Movie, genre: str | None, min_rating: float, max_runtime: int | None) -> bool:
        return ((not genre or genre in movie.genres) and movie.rating >= min_rating
                and (max_runtime is None or 0 < movie.runtime <= max_runtime))

    def discover(self, genre: str | None = None, min_rating: float = 0,
                 max_runtime: int | None = None, limit: int = 12) -> list[Movie]:
        movies = [movie for movie in self.movies if self.eligible(movie, genre, min_rating, max_runtime)]
        return sorted(movies, key=lambda movie: (-self.quality(movie), -movie.popularity, movie.movie_id))[:max(0, limit)]

    def recommend(self, movie_id: int, genre: str | None = None, min_rating: float = 0,
                  max_runtime: int | None = None, limit: int = 6) -> list[Match]:
        if movie_id not in self.positions or limit <= 0:
            return []
        seed = self.by_id[movie_id]
        position = self.positions[movie_id]
        # One sparse row against the catalog, never an N x N dense matrix.
        scores = (self.features @ self.features.getrow(position).T).toarray().ravel()
        candidates = [index for index, movie in enumerate(self.movies)
                      if movie.movie_id != movie_id and scores[index] > 0
                      and self.eligible(movie, genre, min_rating, max_runtime)]
        candidates.sort(key=lambda index: (-float(scores[index]), -self.quality(self.movies[index]), self.movies[index].movie_id))
        return [Match(self.movies[index], min(1.0, float(scores[index])), match_reasons(seed, self.movies[index]))
                for index in candidates[:limit]]


def match_reasons(seed: Movie, movie: Movie) -> tuple[str, ...]:
    reasons = []
    for field, prefix in (("directors", "Directed by "), ("cast", "Also stars "),
                          ("keywords", "Shared theme: "), ("genres", "Both ")):
        shared = sorted(set(getattr(seed, field)) & set(getattr(movie, field)))
        if shared:
            reasons.append(prefix + ", ".join(shared[:2]))
    return tuple(reasons[:2]) or ("Similar story and language",)


def save_catalog(catalog: Catalog, path: Path = MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"version": MODEL_VERSION, "movies": [asdict(movie) for movie in catalog.movies]}, ensure_ascii=False)
    # One atomic artifact cannot pair an old catalog with new features.
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(handle, "wb") as output:
            np.savez_compressed(output, metadata=np.array(metadata), data=catalog.features.data,
                                indices=catalog.features.indices, indptr=catalog.features.indptr,
                                shape=np.array(catalog.features.shape))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_catalog(path: Path = MODEL_PATH) -> Catalog:
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata["version"] != MODEL_VERSION:
                raise ValueError("Unsupported model version. Rebuild with train_model.py.")
            movies = []
            for record in metadata["movies"]:
                for field in ("genres", "keywords", "cast", "directors"):
                    record[field] = tuple(record[field])
                movies.append(Movie(**record))
            features = sparse.csr_matrix((archive["data"], archive["indices"], archive["indptr"]), shape=tuple(archive["shape"]))
        return Catalog(movies, features)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid model artifact. Rebuild with train_model.py.") from exc
