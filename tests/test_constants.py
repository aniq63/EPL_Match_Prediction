"""Tests for the config/constants module."""

import pytest

from config import constants


class TestInputFeatures:
    def test_exactly_twenty_seven_features(self):
        assert len(constants.INPUT_FEATURES) == 27

    def test_no_duplicate_features(self):
        assert len(constants.INPUT_FEATURES) == len(set(constants.INPUT_FEATURES))

    def test_all_features_are_strings(self):
        assert all(isinstance(f, str) for f in constants.INPUT_FEATURES)

    def test_home_advantage_is_boolean_feature(self):
        assert "home_advantage" in constants.INPUT_FEATURES


class TestResultClasses:
    def test_classes_are_home_perspective(self):
        assert set(constants.RESULT_CLASSES) == {"Win", "Draw", "Lose"}

    def test_no_duplicate_classes(self):
        assert len(constants.RESULT_CLASSES) == len(set(constants.RESULT_CLASSES))


class TestTrainingConfig:
    def test_random_state_is_int_for_reproducibility(self):
        assert isinstance(constants.RANDOM_STATE, int)

    def test_cv_folds_positive(self):
        assert constants.CV_FOLDS >= 2

    def test_n_iter_positive(self):
        assert constants.N_ITER > 0

    def test_scoring_is_f1_macro(self):
        assert constants.SCORING == "f1_macro"

    def test_model_name(self):
        assert constants.MODEL_NAME == "AdaBoostClassifier"


class TestETLConfig:
    def test_required_columns_present(self):
        expected = {
            "date", "home_team", "away_team", "home_goals", "away_goals",
            "home_xg", "away_xg", "home_ppda", "away_ppda",
            "home_deep_completions", "away_deep_completions",
            "home_points", "away_points",
        }
        assert expected.issubset(set(constants.ETL_REQUIRED_COLUMNS))

    def test_base_season(self):
        assert constants.ETL_BASE_SEASON >= 2023

    def test_data_source_league(self):
        assert constants.DATA_SOURCE_LEAGUE == "ENG-Premier League"
