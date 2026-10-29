"""Exercise real Streamlit reruns against the bundled movie collection."""

from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("posters.requests.get", side_effect=AssertionError("Tests must work offline"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(self.app.exception)

    def test_typo_search_select_and_save_survive_reruns(self):
        app = self.app
        app.text_input(key="query").set_value("interstelar").run()
        self.assertTrue(any(button.key == "pick_157336" for button in app.button))
        app.button(key="pick_157336").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_id"], 157336)
        recommendations = [button.key for button in app.button if str(button.key).startswith("more_matches_")]
        self.assertEqual(len(recommendations), 6)
        app.button(key="save_selected").click().run()
        self.assertEqual(app.session_state["watchlist"], [157336])
        self.assertEqual([button.key for button in app.button if str(button.key).startswith("more_matches_")], recommendations)
        app.selectbox(key="min_rating").select(8).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_id"], 157336)
        app.button(key="reset_filters").click().run()
        self.assertEqual(app.selectbox(key="min_rating").value, 0)
        app.radio(key="view").set_value("Watchlist").run()
        self.assertTrue(any(button.key == "save_watchlist_157336" for button in app.button))
        app.button(key="save_watchlist_157336").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["watchlist"], [])

    def test_more_like_this_and_surprise_navigate(self):
        app = self.app
        app.button(key="more_discover_27205").click().run()
        self.assertEqual(app.session_state["selected_id"], 27205)
        app.button(key="surprise").click().run()
        self.assertFalse(app.exception)
        self.assertNotEqual(app.session_state["selected_id"], 27205)

    def test_empty_search_has_recovery(self):
        app = self.app
        app.text_input(key="query").set_value("zzzzzzzzzzzzzzzzzzzz").run()
        self.assertFalse(app.exception)
        self.assertFalse(any(str(button.key).startswith("pick_") for button in app.button))
        next(button for button in app.button if button.label == "Back to discovery").click().run()
        self.assertEqual(app.text_input(key="query").value, "")
        self.assertTrue(any(button.key == "more_discover_27205" for button in app.button))


if __name__ == "__main__":
    unittest.main()
