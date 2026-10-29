import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests
from scipy import sparse

from posters import fetch_poster
from recommender import Catalog, Movie, feature_matrix, load_catalog, read_movies, save_catalog
from ui import poster_html, watchlist_csv


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.movies = [
            Movie(90, "Orbit", "An astronaut explores a distant planet", 2010,
                  ("Science Fiction",), ("space",), ("A Person",), ("D Director",), 120, 8, 1000),
            Movie(12, "Orbit", "An astronaut explores a distant planet", 2020,
                  ("Science Fiction",), ("space",), ("A Person",), ("D Director",), 130, 6, 100),
            Movie(47, "Amélie", "A cook explores a distant planet", 2001,
                  ("Comedy",), ("space",), (), (), 90, 7.5, 800),
            Movie(3, "Kitchen", "A cook opens a restaurant", 1999,
                  ("Comedy",), ("food",), (), (), 0, 8.5, 1),
        ]
        self.catalog = Catalog(self.movies, feature_matrix(self.movies))

    def test_remakes_are_independent_and_seed_is_excluded(self):
        matches = self.catalog.recommend(90)
        self.assertEqual(matches[0].movie.movie_id, 12)
        self.assertNotIn(90, [match.movie.movie_id for match in matches])
        self.assertTrue(all(0 < match.score <= 1 for match in matches))
        self.assertIn("Directed by D Director", matches[0].reasons)
        self.assertEqual(self.catalog.recommend(12)[0].movie.movie_id, 90)

    def test_filters_are_applied_before_limiting_results(self):
        matches = self.catalog.recommend(90, genre="Comedy", min_rating=7, max_runtime=90, limit=1)
        self.assertEqual([match.movie.movie_id for match in matches], [47])
        self.assertEqual(self.catalog.recommend(90, genre="Horror"), [])
        self.assertEqual(self.catalog.recommend(-1), [])
        self.assertEqual(self.catalog.recommend(90, limit=0), [])

    def test_search_handles_accents_typos_and_duplicate_titles(self):
        self.assertEqual(self.catalog.search(" AMELIE ")[0].movie_id, 47)
        self.assertEqual(self.catalog.search("Orbt")[0].title, "Orbit")
        self.assertEqual({movie.movie_id for movie in self.catalog.search("orbit")}, {90, 12})
        self.assertEqual(self.catalog.search("   "), [])
        self.assertEqual(self.catalog.search("zzzzzzzzzzzzzzzzzz"), [])

    def test_discovery_accounts_for_votes_and_missing_runtime(self):
        self.assertLess(self.catalog.quality(self.movies[3]), self.catalog.quality(self.movies[0]))
        self.assertEqual([movie.movie_id for movie in self.catalog.discover(max_runtime=90)], [47])

    def test_artifact_roundtrip_preserves_ids_and_rankings_without_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            save_catalog(self.catalog, path)
            actual = load_catalog(path)
            self.assertEqual(actual.movies, self.catalog.movies)
            self.assertEqual(actual.recommend(90), self.catalog.recommend(90))
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(all(archive[key].dtype.kind != "O" for key in archive.files))

    def test_invalid_models_and_empty_features_are_rejected(self):
        with self.assertRaises(ValueError):
            Catalog(self.movies, sparse.csr_matrix((3, 4)))
        with self.assertRaises(ValueError):
            Catalog([self.movies[0], self.movies[0]], sparse.eye(2).tocsr())
        with self.assertRaises(ValueError):
            feature_matrix([Movie(1, "Empty"), Movie(2, "Also empty")])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.npz"
            np.savez(path, metadata=np.array(json.dumps({"version": -1})))
            with self.assertRaisesRegex(ValueError, "Unsupported model version"):
                load_catalog(path)

    def test_csv_join_uses_ids_and_keeps_missing_optional_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            movies_path, credits_path = Path(folder) / "movies.csv", Path(folder) / "credits.csv"
            pd.DataFrame([
                {"id": 1, "title": "The Same", "genres": "[]", "keywords": "bad json", "overview": None},
                {"id": 2, "title": "The Same", "genres": "[]", "keywords": "[]", "overview": "A plot"},
                {"id": 3, "title": "No credits", "genres": "[]", "keywords": "[]", "overview": "Another plot"},
            ]).to_csv(movies_path, index=False)
            pd.DataFrame([
                {"movie_id": 2, "title": "The Same", "cast": '[{"name":"Second Actor"}]', "crew": "[]"},
                {"movie_id": 1, "title": "The Same", "cast": '[{"name":"First Actor"}]', "crew": "[]"},
            ]).to_csv(credits_path, index=False)
            movies = read_movies(movies_path, credits_path)
            self.assertEqual(len(movies), 3)
            self.assertEqual(movies[0].cast, ("First Actor",))
            self.assertEqual(movies[1].cast, ("Second Actor",))
            self.assertEqual(movies[0].overview, "")
            self.assertEqual(movies[2].cast, ())

    def test_titles_are_escaped_and_watchlist_is_csv_safe(self):
        movie = replace(self.movies[0], title='<script>alert("hi")</script>')
        self.assertNotIn("<script>", poster_html(movie))
        movie = replace(movie, title="=SUM(1,2)")
        rows = list(csv.reader(io.StringIO(watchlist_csv([movie]))))
        self.assertEqual(rows[1][1], "'=SUM(1,2)")


class PosterTests(unittest.TestCase):
    @patch("posters.requests.get")
    def test_no_key_never_uses_network(self, get):
        self.assertIsNone(fetch_poster(1, ""))
        get.assert_not_called()

    @patch("posters.requests.get")
    def test_valid_response_uses_only_public_image_url(self, get):
        get.return_value = Mock(json=lambda: {"poster_path": "/image-123.jpg"})
        self.assertEqual(fetch_poster(1, "secret"), "https://image.tmdb.org/t/p/w500/image-123.jpg")
        get.return_value.raise_for_status.assert_called_once()
        self.assertEqual(get.call_args.kwargs["timeout"], (2, 3))

    @patch("posters.requests.get")
    def test_failures_and_unexpected_payloads_fall_back(self, get):
        for payload in ({"poster_path": None}, [], {"poster_path": "https://untrusted.example/a.jpg"}):
            get.return_value = Mock(json=lambda: payload)
            self.assertIsNone(fetch_poster(1, "secret"))
        get.side_effect = requests.Timeout()
        self.assertIsNone(fetch_poster(1, "secret"))
        get.side_effect = None
        get.return_value = Mock()
        get.return_value.raise_for_status.side_effect = requests.HTTPError()
        self.assertIsNone(fetch_poster(1, "secret"))


if __name__ == "__main__":
    unittest.main()
