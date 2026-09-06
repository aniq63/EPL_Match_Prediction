"""Tests for the feature engineering pipeline (the core 27-feature logic)."""

import pandas as pd
import pytest

from config.constants import INPUT_FEATURES
from src.feature_engineering.feature_enginnering import FeatureEngineering, RowTracker


class TestRowTracker:
    def test_asserts_if_after_without_before(self):
        tracker = RowTracker()
        with pytest.raises(AssertionError):
            tracker.after(pd.DataFrame({"x": [1]}))

    def test_tracks_dropped_rows(self):
        tracker = RowTracker()
        df = pd.DataFrame({"x": [1, 2, 3]})
        tracker.before("step", df)
        tracker.after(df.iloc[:2], note="n")
        entry = tracker.log[-1]
        assert entry["step"] == "step"
        assert entry["before"] == 3
        assert entry["after"] == 2
        assert entry["dropped"] == 1
        assert entry["note"] == "n"


class TestFeatureEngineeringInit:
    def test_accepts_dataframe(self, raw_matches):
        fe = FeatureEngineering(raw_matches)
        assert fe.df is not raw_matches  # defensive copy

    def test_rejects_non_dataframe(self):
        with pytest.raises(ValueError):
            FeatureEngineering([1, 2, 3])


class TestBasicFeatures:
    def test_creates_result_column(self, raw_matches):
        fe = FeatureEngineering(raw_matches)
        out = fe.basic_features().df
        assert "result" in out.columns
        assert set(out["result"].unique()) <= {"Win", "Draw", "Lose"}

    def test_result_is_correctly_classified(self):
        df = pd.DataFrame(
            {
                "date": ["2024-08-10", "2024-08-17", "2024-08-24"],
                "home_goals": [3, 2, 0],
                "away_goals": [1, 2, 1],
            }
        )
        out = FeatureEngineering(df).basic_features().df
        assert out["result"].tolist() == ["Win", "Draw", "Lose"]

    def test_creates_goals_conceded_columns(self, raw_matches_small):
        out = FeatureEngineering(raw_matches_small).basic_features().df
        assert (out["home_goals_conceded"] == out["away_goals"]).all()
        assert (out["away_goals_conceded"] == out["home_goals"]).all()

    def test_sorts_by_date(self, raw_matches):
        fe = FeatureEngineering(raw_matches).basic_features()
        assert fe.df["date"].is_monotonic_increasing


class TestRollingFeatures:
    def test_creates_average_columns(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(pd.date_range("2025-01-01", periods=20, freq="D")),
                "home_team": ["A"] * 20,
                "away_team": ["B"] * 20,
                "home_goals": [1] * 20,
                "away_goals": [0] * 20,
                "home_xg": [1.5] * 20,
                "away_xg": [0.5] * 20,
                "home_ppda": [10.0] * 20,
                "away_ppda": [10.0] * 20,
                "home_deep_completions": [20] * 20,
                "away_deep_completions": [10] * 20,
                "home_goals_conceded": [0] * 20,
                "away_goals_conceded": [1] * 20,
            }
        )
        out = FeatureEngineering(df).basic_features().rolling_features().df
        for col in [
            "home_goals_avg_last5", "away_goals_avg_last5",
            "home_goals_conceded_avg_last5", "away_goals_conceded_avg_last5",
            "home_xg_avg_last5", "away_xg_avg_last5",
            "home_ppda_avg_last5", "away_ppda_avg_last5",
            "home_deep_completions_avg_last5", "away_deep_completions_avg_last5",
        ]:
            assert col in out.columns

    def test_rolling_average_uses_last_five_closed_left(self):
        # home team scores 1,2,3,4,5,6 across six matches
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(pd.date_range("2025-01-01", periods=6, freq="W")),
                "home_team": ["A"] * 6,
                "away_team": ["B"] * 6,
                "home_goals": [1, 2, 3, 4, 5, 6],
                "away_goals": [0] * 6,
                "home_xg": [1.0] * 6,
                "away_xg": [0.5] * 6,
                "home_ppda": [10.0] * 6,
                "away_ppda": [11.0] * 6,
                "home_deep_completions": [20] * 6,
                "away_deep_completions": [21] * 6,
                "home_goals_conceded": [0] * 6,
                "away_goals_conceded": [1] * 6,
            }
        )
        out = FeatureEngineering(df).basic_features().rolling_features().df
        # Rolling uses closed="left", so at the 6th match the window is the
        # PREVIOUS 5 home games = [1,2,3,4,5] -> mean 3.0 (excludes current).
        last = out.iloc[-1]
        assert last["home_goals_avg_last5"] == pytest.approx(3.0)

    def test_drops_early_rows_without_5_matches(self, raw_matches):
        fe = FeatureEngineering(raw_matches).basic_features()
        before = len(fe.df)
        out = fe.rolling_features().df
        assert len(out) <= before


class TestFullPipeline:
    def test_pipeline_produces_all_27_features(self, raw_matches):
        out = FeatureEngineering(raw_matches).run()
        missing = [c for c in INPUT_FEATURES if c not in out.columns]
        assert not missing, f"Missing feature columns: {missing}"

    def test_pipeline_keeps_original_columns(self, raw_matches):
        out = FeatureEngineering(raw_matches).run()
        for col in raw_matches.columns:
            assert col in out.columns

    def test_pipeline_drops_early_games(self, raw_matches):
        # Rolling windows need >=5 prior games; some rows must be dropped
        out = FeatureEngineering(raw_matches).run()
        assert len(out) <= len(raw_matches)
        assert len(out) > 0

    def test_pipeline_sorted_by_date(self, raw_matches):
        out = FeatureEngineering(raw_matches).run()
        assert out["date"].is_monotonic_increasing

    def test_home_advantage_always_one(self, raw_matches):
        out = FeatureEngineering(raw_matches).run()
        assert (out["home_advantage"] == 1).all()

    def test_derived_diff_features_present(self, raw_matches):
        out = FeatureEngineering(raw_matches).run()
        for col in [
            "points_diff_last5", "goal_diff_avg5", "xg_diff_avg5",
            "x_defense_diff", "ppda_diff_avg5", "deep_comp_diff_avg5",
            "venue_wins_diff", "home_venue_advantage", "home_advantage",
        ]:
            assert col in out.columns

    def test_pipeline_has_no_nan_in_features(self, raw_matches):
        out = FeatureEngineering(raw_matches).run()
        assert out[INPUT_FEATURES].isna().sum().sum() == 0

    def test_pipeline_returns_copy_not_shared(self, raw_matches):
        original_rows = len(raw_matches)
        FeatureEngineering(raw_matches).run()
        assert len(raw_matches) == original_rows  # input unmodified
