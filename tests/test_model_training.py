"""Tests for the ModelTrainer component."""

import numpy as np
import pandas as pd
import pytest

from config.constants import INPUT_FEATURES, RESULT_CLASSES
from src.components.model_training import ModelTrainer


def _make_df(n, seed=0, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(pd.date_range(start=start, periods=n, freq="D"))
    rows = []
    for i in range(n):
        row = {"date": dates[i]}
        for f in INPUT_FEATURES:
            row[f] = float(rng.uniform(0, 5))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["result"] = rng.choice(RESULT_CLASSES, size=n)
    return df


@pytest.fixture
def train_df():
    return _make_df(40)


@pytest.fixture
def test_df():
    return _make_df(10, seed=1)


class TestInit:
    def test_stores_dataframe_copies(self, train_df, test_df):
        t = ModelTrainer(train_df, test_df)
        assert t.train_df is not train_df
        assert t.test_df is not test_df

    def test_label_encoder_fitted_to_expected_classes(self, train_df, test_df):
        t = ModelTrainer(train_df, test_df)
        assert set(t.le.classes_) == set(RESULT_CLASSES)


class TestValidateFeatures:
    def test_passes_when_all_features_present(self, train_df, test_df):
        t = ModelTrainer(train_df, test_df)
        t._validate_features(train_df, "train_df")  # should not raise

    def test_raises_on_missing_feature(self, train_df, test_df):
        missing_df = train_df.drop(columns=["home_advantage"])
        t = ModelTrainer(missing_df, test_df)
        with pytest.raises(ValueError) as exc:
            t._validate_features(missing_df, "train_df")
        assert "home_advantage" in str(exc.value)


class TestPrepareData:
    def test_returns_encoded_shapes(self, train_df, test_df):
        t = ModelTrainer(train_df, test_df)
        X_train, y_train, X_test, y_test = t._prepare_data()
        assert X_train.shape == (len(train_df), len(INPUT_FEATURES))
        assert X_test.shape == (len(test_df), len(INPUT_FEATURES))
        assert len(y_train) == len(train_df)
        assert len(y_test) == len(test_df)

    def test_encoded_labels_in_range(self, train_df, test_df):
        t = ModelTrainer(train_df, test_df)
        _, y_train, _, _ = t._prepare_data()
        assert set(np.unique(y_train)).issubset({0, 1, 2})

    def test_raises_when_result_column_missing(self, train_df, test_df):
        bad = train_df.drop(columns=["result"])
        t = ModelTrainer(bad, test_df)
        with pytest.raises(KeyError):
            t._prepare_data()


class TestBuildFinalModel:
    def test_model_params_recorded(self, train_df, test_df):
        t = ModelTrainer(train_df, test_df)
        best = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "estimator__max_depth": 3,
            "estimator__min_samples_leaf": 2,
            "estimator__min_samples_split": 2,
            "estimator__max_features": "sqrt",
        }
        model = t._build_final_model(best)
        assert t.model_params["n_estimators"] == 100
        assert t.model_params["max_depth"] == 3
        assert t.model_params["model_name"] == "AdaBoostClassifier"


class TestTrain:
    def test_train_returns_complete_dict(self, train_df, test_df, monkeypatch):
        t = ModelTrainer(train_df, test_df)

        class FakeSearch:
            best_params_ = {
                "n_estimators": 50,
                "learning_rate": 0.1,
                "estimator__max_depth": 3,
                "estimator__min_samples_leaf": 2,
                "estimator__min_samples_split": 2,
                "estimator__max_features": "sqrt",
            }

        monkeypatch.setattr(t, "_run_hyperparameter_search", lambda X, y: FakeSearch())
        result = t.train()
        assert set(result.keys()) == {
            "model_name", "model", "params", "label_encoder", "X_test", "y_test",
        }
        assert result["model_name"] == "AdaBoostClassifier"
        assert result["model"].n_estimators == 50

    def test_model_is_fitted(self, train_df, test_df, monkeypatch):
        t = ModelTrainer(train_df, test_df)

        class FakeSearch:
            best_params_ = {
                "n_estimators": 10,
                "learning_rate": 0.1,
                "estimator__max_depth": 3,
                "estimator__min_samples_leaf": 2,
                "estimator__min_samples_split": 2,
                "estimator__max_features": "sqrt",
            }

        monkeypatch.setattr(t, "_run_hyperparameter_search", lambda X, y: FakeSearch())
        result = t.train()
        from sklearn.ensemble import AdaBoostClassifier
        assert isinstance(result["model"], AdaBoostClassifier)
        assert hasattr(result["model"], "classes_")
