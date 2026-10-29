"""Optional TMDB images. Recommendations always work without a network."""

import re

import requests


def fetch_poster(movie_id: int, api_key: str) -> str | None:
    if not api_key or movie_id <= 0:
        return None
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(movie_id)}",
            params={"api_key": api_key, "language": "en-US"}, timeout=(2, 3),
        )
        response.raise_for_status()
        payload = response.json()
        path = payload.get("poster_path") if isinstance(payload, dict) else None
        if isinstance(path, str) and re.fullmatch(r"/[a-zA-Z0-9_\-]+\.(?:jpg|png|webp)", path):
            return "https://image.tmdb.org/t/p/w500" + path
    except (requests.RequestException, ValueError):
        pass
    return None
