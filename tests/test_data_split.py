"""Tests for the data split utility."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from config.constants import TEST_SIZE_WEEKS
from src.utils.data_split import DataSplitter, get_split_date


class TestGetSplitDate:
    def test_returns_datetime(self):
        assert isinstance(get_split_date(), datetime)

    def test_is_exactly_test_size_weeks_ago(self):
        split = get_split_date()
        expected = datetime.now() - timedelta(weeks=TEST_SIZE_WEEKS)
        # Allow ~5 seconds drift from the now() sampled inside
        assert abs((split - expected).total_seconds()) < 10


class TestDataSplitter:
    def _make_df(self, dates):
        return pd.DataFrame({"date": pd.to_datetime(dates), "home_team": ["A"] * len(dates)})

    def test_raises_if_no_date_column(self):
        df = pd.DataFrame({"home_team": ["A"]})
        with pytest.raises(Exception):
            DataSplitter(df).split()

    def test_dates_are_temporally_disjoint(self):
        split_date = get_split_date()
        df = self._make_df(
            pd.to_datetime(
                [
                    split_date - timedelta(days=30),
                    split_date - timedelta(days=10),
                    split_date + timedelta(days=1),
                    split_date + timedelta(days=5),
                ]
            )
        )
        train, test = DataSplitter(df).split()
        # Train contains everything before split_date, test everything >= split_date
        assert (train["date"] < split_date).all()
        assert (test["date"] >= split_date).all()

    def test_split_is_exhaustive_for_future_dates(self):
        split_date = get_split_date()
        df = self._make_df([split_date + timedelta(days=1), split_date + timedelta(days=5)])
        train, test = DataSplitter(df).split()
        assert len(train) == 0
        assert len(test) == 2

    def test_split_all_past(self):
        split_date = get_split_date()
        df = self._make_df([split_date - timedelta(days=60)])
        train, test = DataSplitter(df).split()
        assert len(train) == 1
        assert len(test) == 0

    def test_stores_defensive_copy(self):
        df = pd.DataFrame({"date": ["2024-08-10", "2025-01-01"], "home_team": ["A", "B"]})
        splitter = DataSplitter(df)
        # Mutating the internal copy must not affect the caller's frame
        splitter.df["home_team"] = "MUTATED"
        assert df["home_team"].tolist() == ["A", "B"]

    def test_does_not_mutate_input_dataframe(self):
        split_date = get_split_date()
        df = self._make_df([split_date - timedelta(days=5)])
        original = df.copy()
        DataSplitter(df).split()
        pd.testing.assert_frame_equal(df, original)
