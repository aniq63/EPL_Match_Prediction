"""Tests for the ETL data transformation module."""

import pandas as pd
import pytest

from config.constants import ETL_COLUMNS_TO_DROP
from src.etl.data_transformation import DataTransformation


def _df_with_drop_cols(**extra):
    df = pd.DataFrame(
        {
            "date": ["2024-08-10"],
            "home_team": ["Arsenal"],
            "league_id": [1],
            "season_id": [1],
            "game_id": [1],
            "home_team_id": [1],
            "home_np_xg_difference": [0.1],
        }
    )
    for k, v in extra.items():
        df[k] = v
    return df


class TestDropUselessColumns:
    def test_drops_configured_columns(self):
        df = _df_with_drop_cols()
        out = DataTransformation(df).drop_useless_columns(df.copy())
        for col in ETL_COLUMNS_TO_DROP:
            assert col not in out.columns

    def test_keeps_data_columns(self):
        df = _df_with_drop_cols()
        out = DataTransformation(df).drop_useless_columns(df.copy())
        assert "date" in out.columns
        assert "home_team" in out.columns
        assert "home_goals" not in out.columns  # absent from source anyway

    def test_ignores_missing_columns_without_error(self):
        df = pd.DataFrame({"date": ["2024-08-10"], "home_team": ["Arsenal"]})
        out = DataTransformation(df).drop_useless_columns(df.copy())
        # Should not raise even though none of the drop columns exist
        assert set(out.columns) == {"date", "home_team"}

    def test_drop_identity_preserves_row_count(self):
        df = _df_with_drop_cols()
        out = DataTransformation(df).drop_useless_columns(df.copy())
        assert len(out) == len(df)


class TestTransformPlData:
    def test_returns_dataframe_without_drop_columns(self):
        df = pd.DataFrame(
            {
                "date": ["2024-08-10", "2024-08-17"],
                "home_team": ["Arsenal", "Chelsea"],
                "league_id": [1, 1],
                "home_goals": [2, 1],
            }
        )
        transformer = DataTransformation(df)
        out = transformer.transform_pl_data()
        assert isinstance(out, pd.DataFrame)
        assert "league_id" not in out.columns
        assert "home_goals" in out.columns
        assert len(out) == 2

    def test_does_not_mutate_input(self):
        df = pd.DataFrame(
            {"date": ["2024-08-10"], "home_team": ["Arsenal"], "league_id": [1]}
        )
        original = df.copy()
        DataTransformation(df).transform_pl_data()
        pd.testing.assert_frame_equal(df, original)
