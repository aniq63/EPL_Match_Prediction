"""Tests for the gameweek prediction pipeline (helpers + feature builders)."""

import pandas as pd
import pytest
from datetime import datetime

from src.services import prediction_pipeline as pp
from src.services.prediction_pipeline import (
    PredictionPipeline,
    get_current_season,
    get_last5_all_matches,
    get_last5_home_matches,
    get_last5_away_matches,
    FEATURE_COLUMNS,
)


class TestGetCurrentSeason:
    def test_returns_two_year_string(self):
        season = get_current_season()
        assert season.count("/") == 1
        a, b = season.split("/")
        assert int(a) + 1 == int(b)

    @pytest.mark.parametrize(
        "month,expected_offset",
        [
            (1, -1),  # Jan => previous_year/current_year
            (7, -1),  # Jul => previous_year/current_year
            (8, 0),  # Aug => current_year/next_year
            (12, 0),  # Dec => current_year/next_year
        ],
    )
    def test_season_boundary_logic(self, monkeypatch, month, expected_offset):
        class FakeNow(datetime):
            @classmethod
            def now(cls):
                return datetime(2025, month, 15)

        monkeypatch.setattr(pp, "datetime", FakeNow)
        season = get_current_season()
        a, b = season.split("/")
        assert int(a) == 2025 + expected_offset


def _make_history_df():
    """A history DataFrame with the columns used by the last-5 helpers."""
    n = 10
    return pd.DataFrame(
        {
            "date": pd.to_datetime(pd.date_range("2025-01-01", periods=n, freq="D")),
            "home_team": ["Arsenal"] * n,
            "away_team": ["Chelsea"] * n,
            "home_goals": [2, 1, 3, 0, 1, 2, 1, 0, 3, 2],
            "away_goals": [1, 1, 0, 2, 1, 0, 1, 2, 1, 0],
            "home_xg": [1.5] * n,
            "away_xg": [1.0] * n,
            "home_ppda": [10.0] * n,
            "away_ppda": [11.0] * n,
            "home_deep_completions": [20] * n,
            "away_deep_completions": [10] * n,
            "home_points": [3, 1, 3, 0, 1, 3, 1, 0, 3, 3],
            "away_points": [0, 1, 0, 3, 1, 0, 1, 3, 0, 0],
            "result": ["Win", "Draw", "Win", "Lose", "Draw",
                       "Win", "Draw", "Lose", "Win", "Win"],
        }
    )


class TestLast5Helpers:
    def test_get_last5_all_matches_returns_five(self):
        df = _make_history_df()
        out = get_last5_all_matches(df, "Arsenal")
        assert len(out) == 5

    def test_get_last5_all_adds_unified_columns(self):
        df = _make_history_df()
        out = get_last5_all_matches(df, "Arsenal")
        for col in ["_goals", "_goals_con", "_xg", "_ppda", "_deep", "_points"]:
            assert col in out.columns

    def test_get_last5_all_raises_for_unknown_team(self):
        df = _make_history_df()
        with pytest.raises(ValueError):
            get_last5_all_matches(df, "Nonexistent FC")

    def test_get_last5_home_matches_returns_five(self):
        df = _make_history_df()
        out = get_last5_home_matches(df, "Arsenal")
        assert len(out) == 5
        assert (out["home_team"] == "Arsenal").all()

    def test_get_last5_away_matches_returns_five(self):
        df = _make_history_df()
        out = get_last5_away_matches(df, "Chelsea")
        assert len(out) == 5
        assert (out["away_team"] == "Chelsea").all()


@pytest.fixture
def pipeline():
    """Instance used only for pure feature-building methods.

    Avoids PredictionPipeline.__init__ (which calls DagsHub/MLflow
    `_connect_to_mlflow` and makes network calls) by constructing the
    object via __new__ and setting only the attrs the tested methods use.
    """
    obj = PredictionPipeline.__new__(PredictionPipeline)
    obj.stage = "Production"
    obj.model_name = "AdaBoostClassifier"
    obj.model_uri = "models:/AdaBoostClassifier/Production"
    return obj


@pytest.fixture
def mock_slices():
    """Slices with deterministic last-5 values for both teams."""
    rng = __import__("numpy").random.default_rng(0)

    def _slice(n, points, goals, xg, ppda, deep, con, points_col):
        return pd.DataFrame(
            {
                "date": pd.to_datetime(pd.date_range("2025-01-01", periods=n, freq="D")),
                "_goals": [goals] * n,
                "_goals_con": [con] * n,
                "_xg": [xg] * n,
                "_ppda": [ppda] * n,
                "_deep": [deep] * n,
                "_points": [points] * n,
                points_col: [points] * n,
            }
        )

    return {
        # home_all / away_all only need the _-prefixed unified columns
        "home_all": _slice(5, points=3, goals=2.0, xg=1.5, ppda=10.0, deep=20, con=0.6, points_col="_points"),
        "away_all": _slice(5, points=1, goals=1.0, xg=0.8, ppda=12.0, deep=8, con=1.2, points_col="_points"),
        # venue slices need the raw home_points / away_points columns
        "home_h5": _slice(5, points=3, goals=2.0, xg=1.5, ppda=10.0, deep=20, con=0.6, points_col="home_points"),
        "away_a5": _slice(5, points=0, goals=1.0, xg=0.8, ppda=12.0, deep=8, con=1.2, points_col="away_points"),
    }


class TestBuildFeatures:
    def test_build_home_features_computes_means(self, pipeline, mock_slices):
        feats = pipeline.build_home_features(mock_slices["home_all"])
        assert feats["goals_avg"] == pytest.approx(2.0)
        assert feats["points_sum"] == 15
        assert feats["xg_avg"] == pytest.approx(1.5)

    def test_build_away_features_computes_means(self, pipeline, mock_slices):
        feats = pipeline.build_away_features(mock_slices["away_all"])
        assert feats["points_sum"] == 5

    def test_build_venue_features_counts(self, pipeline, mock_slices):
        feats = pipeline.build_venue_features(
            mock_slices["home_h5"], mock_slices["away_a5"]
        )
        assert feats["home_venue_wins"] == 5  # all points=3
        assert feats["away_venue_wins"] == 0  # all points=0
        assert feats["home_venue_advantage"] == pytest.approx(1.0)
        assert feats["venue_wins_diff"] == 5


class TestBuildPredictionRow:
    def test_returns_27_features_in_exact_order(self, pipeline, mock_slices):
        home = pipeline.build_home_features(mock_slices["home_all"])
        away = pipeline.build_away_features(mock_slices["away_all"])
        venue = pipeline.build_venue_features(mock_slices["home_h5"], mock_slices["away_a5"])

        row = pipeline.build_prediction_row(home, away, venue)
        assert row.shape == (1, 27)
        assert list(row.columns) == FEATURE_COLUMNS
        assert row["home_advantage"].iloc[0] == 1

    def test_differential_features_are_home_minus_away(self, pipeline, mock_slices):
        home = pipeline.build_home_features(mock_slices["home_all"])
        away = pipeline.build_away_features(mock_slices["away_all"])
        venue = pipeline.build_venue_features(mock_slices["home_h5"], mock_slices["away_a5"])

        row = pipeline.build_prediction_row(home, away, venue)
        assert row["points_diff_last5"].iloc[0] == 15 - 5
        assert row["goal_diff_avg5"].iloc[0] == pytest.approx(2.0 - 1.0)
        assert row["xg_diff_avg5"].iloc[0] == pytest.approx(1.5 - 0.8)


class TestMapEspnName:
    def test_maps_known_team(self, pipeline):
        assert pipeline.map_espn_name("Brighton & Hove Albion") == "Brighton"
        assert pipeline.map_espn_name("Tottenham Hotspur") == "Tottenham"

    def test_returns_none_for_unknown_team(self, pipeline):
        assert pipeline.map_espn_name("Some Nonexistent Club") is None

    def test_direct_mapping_stays_identity_for_full_names(self, pipeline):
        assert pipeline.map_espn_name("Arsenal") == "Arsenal"
        assert pipeline.map_espn_name("Liverpool") == "Liverpool"
