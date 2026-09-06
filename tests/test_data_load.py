"""Tests for the ETL data load validation logic (no DB connection needed)."""

import pandas as pd
import pytest

from config.constants import ETL_REQUIRED_COLUMNS
from src.etl.data_load import DataLoad


def _valid_df():
    return pd.DataFrame({col: [0] for col in ETL_REQUIRED_COLUMNS})


class TestValidateColumns:
    def test_passes_when_all_columns_present(self):
        assert DataLoad(_valid_df()).validate_columns() is True

    def test_raises_when_column_missing(self):
        df = _valid_df().drop(columns=["home_goals"])
        with pytest.raises(Exception) as exc:
            DataLoad(df).validate_columns()
        assert "home_goals" in str(exc.value)

    def test_raises_when_multiple_columns_missing(self):
        df = _valid_df().drop(columns=["home_goals", "away_xg", "date"])
        with pytest.raises(Exception) as exc:
            DataLoad(df).validate_columns()
        assert "home_goals" in str(exc.value)
        assert "away_xg" in str(exc.value)

    def test_empty_columns(self):
        with pytest.raises(Exception):
            DataLoad(pd.DataFrame()).validate_columns()

    def test_required_columns_constant_is_list_of_strings(self):
        assert isinstance(ETL_REQUIRED_COLUMNS, list)
        assert all(isinstance(c, str) for c in ETL_REQUIRED_COLUMNS)
