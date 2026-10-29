"""Build the same compact, safe model artifact used by the application."""

import argparse
from pathlib import Path
from time import perf_counter

from recommender import BASE_DIR, MODEL_PATH, Catalog, feature_matrix, read_movies, save_catalog


def build_model(movies_path: Path = BASE_DIR / "tmdb_5000_movies.csv",
                credits_path: Path = BASE_DIR / "tmdb_5000_credits.csv",
                output_path: Path = MODEL_PATH) -> Catalog:
    for path in (movies_path, credits_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Dataset not found: {path}. Supply both TMDB 5000 CSV files.")
    movies = read_movies(movies_path, credits_path)
    catalog = Catalog(movies, feature_matrix(movies))
    save_catalog(catalog, output_path)
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies", type=Path, default=BASE_DIR / "tmdb_5000_movies.csv")
    parser.add_argument("--credits", type=Path, default=BASE_DIR / "tmdb_5000_credits.csv")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    started = perf_counter()
    try:
        catalog = build_model(args.movies, args.credits, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Could not build the model: {exc}\n")
    print(f"Built {len(catalog.movies):,} movies in {perf_counter() - started:.1f}s.")
    print(f"Saved {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MB).")


if __name__ == "__main__":
    main()
