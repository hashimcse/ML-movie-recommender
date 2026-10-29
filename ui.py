"""Small, escaped presentation helpers for CineMatch."""

import csv
from html import escape
from io import StringIO

from recommender import Movie


def movie_meta(movie: Movie) -> str:
    return " · ".join(filter(None, [str(movie.year or ""),
                                    f"{movie.runtime} min" if movie.runtime else "",
                                    " / ".join(movie.genres[:2])]))


def poster_html(movie: Movie, poster_url: str | None = None) -> str:
    """Original typographic covers remain useful when TMDB is unavailable."""
    title = escape(movie.title)
    variant = movie.movie_id % 5
    artwork = (
        f'<div class="cover cover-{variant}" role="img" aria-label="Illustrated cover for {title}">'
        '<span class="cover-edition">THE CINEMATCH COLLECTION</span>'
        '<div class="cover-art" aria-hidden="true"><i></i><b></b><em></em></div>'
        f'<div class="cover-copy"><span>{escape(movie.genres[0] if movie.genres else "CINEMA")}</span>'
        f'<strong>{title}</strong><small>{movie.year or "A FILM TO DISCOVER"}</small></div></div>'
    )
    if poster_url:
        # The illustrated cover also sits behind the image if the image host fails.
        return (f'<div class="poster-wrap">{artwork}'
                f'<img class="real-poster" src="{escape(poster_url, quote=True)}" '
                f'alt="{title} poster" loading="lazy" referrerpolicy="no-referrer"></div>')
    return artwork


def hero_html() -> str:
    covers = [
        Movie(157336, "Interstellar", year=2014, genres=("Science Fiction",)),
        Movie(120467, "The Grand Budapest Hotel", year=2014, genres=("Comedy",)),
        Movie(27205, "Inception", year=2010, genres=("Thriller",)),
    ]
    return (
        '<section class="hero"><div class="hero-copy">'
        '<div class="eyebrow"><span class="live-dot"></span> GOOD FILMS. LESS SCROLLING.</div>'
        '<h1>Life’s too short<br>for <em>bad movies.</em></h1>'
        '<p>Start with a film you love. Discover the stories, worlds,<br class="desktop-break"> '
        'and unexpected favorites waiting on the other side.</p>'
        '<div class="hero-note"><span>✦</span> A little movie magic, matched to your taste.</div>'
        '</div><div class="hero-gallery" aria-hidden="true">'
        + "".join(f'<div class="hero-print print-{i}">{poster_html(movie)}</div>' for i, movie in enumerate(covers))
        + '<div class="gallery-caption">YOUR NEXT FAVORITE IS OUT THERE. ↗</div></div></section>'
    )


def watchlist_csv(movies: list[Movie]) -> str:
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["TMDB ID", "Title", "Year", "Genres", "Runtime (min)", "Rating", "TMDB URL"])
    for movie in movies:
        # Prevent titles from being interpreted as formulas by spreadsheet apps.
        title = movie.title
        if title.startswith(("=", "+", "-", "@", "\t", "\r")):
            title = "'" + title
        writer.writerow([movie.movie_id, title, movie.year or "", " / ".join(movie.genres),
                         movie.runtime or "", movie.rating or "",
                         f"https://www.themoviedb.org/movie/{movie.movie_id}"])
    return output.getvalue()
