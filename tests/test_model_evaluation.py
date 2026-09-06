"""Tests for the ModelEvaluator component (without MLflow network calls)."""

import numpy as np
import pandas as pd
import pytest

from config.constants import INPUT_FEATURES
from src.components.model_evaluation import ModelEvaluator


def _make_feature_df(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f: rng.uniform(0, 5, n) for f in INPUT_FEATURES}, index=range(n)
    )


def _make_trained_model():
    """Train a tiny AdaBoost so predictions and probas are meaningful."""
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.preprocessing import LabelEncoder

    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame({f: rng.uniform(0, 5, n) for f in INPUT_FEATURES})
    y_vals = rng.choice(["Win", "Draw", "Lose"], size=n)
    le = LabelEncoder()
    y = le.fit_transform(y_vals)
    model = AdaBoostClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, le


def _make_model_dict(model, le, X_test, y_test):
    return {
        "model_name": "AdaBoostClassifier",
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "label_encoder": le,
        "params": {
            "input_features": list(INPUT_FEATURES),
            "n_estimators": 10,
            "model_name": "AdaBoostClassifier",
        },
    }


@pytest.fixture
def model_dict():
    model, le = _make_trained_model()
    X_test = _make_feature_df(20, seed=5)
    rng = np.random.default_rng(5)
    y_test = np.asarray(rng.integers(0, 3, size=20))
    return _make_model_dict(model, le, X_test, y_test)


class TestInit:
    def test_raises_on_missing_key(self):
        with pytest.raises(Exception):
            ModelEvaluator({})

    def test_has_results_empty_before_evaluate(self, model_dict):
        ev = ModelEvaluator(model_dict)
        assert ev._results == {}

    def test_run_name_defaults_to_model_name(self, model_dict):
        ev = ModelEvaluator(model_dict)
        assert ev.run_name == "AdaBoostClassifier"


class TestComputeMetrics:
    def test_returns_all_expected_keys(self, model_dict):
        ev = ModelEvaluator(model_dict)
        ev.model.predict  # model must be fitted for evaluate
        y_pred = ev.model.predict(ev.X_test)
        metrics = ev._compute_metrics(y_pred)
        expected = {
            "accuracy", "precision", "recall", "f1_score",
            "precision_Draw", "recall_Draw", "f1_Draw",
            "precision_Lose", "recall_Lose", "f1_Lose",
            "precision_Win", "recall_Win", "f1_Win",
        }
        assert set(metrics.keys()) == expected


class TestEvaluate:
    def test_evaluate_returns_full_results(self, model_dict):
        ev = ModelEvaluator(model_dict)
        res = ev.evaluate()
        assert "metrics" in res
        assert res["confusion_matrix"].shape == (3, 3)
        assert isinstance(res["feature_importances"], pd.Series)
        assert len(res["y_pred"]) == len(ev.X_test)
        assert res["y_pred_proba"].shape == (len(ev.X_test), 3)
        assert "accuracy" in res["metrics"]
        assert 0.0 <= res["metrics"]["accuracy"] <= 1.0

    def test_evaluate_populates_results_cache(self, model_dict):
        ev = ModelEvaluator(model_dict)
        ev.evaluate()
        assert ev._results != {}

    def test_feature_importance_index_matches_features(self, model_dict):
        ev = ModelEvaluator(model_dict)
        res = ev.evaluate()
        assert set(res["feature_importances"].index) == set(INPUT_FEATURES)


class TestBuildEvaluationFigure:
    def test_returns_matplotlib_figure(self, model_dict):
        ev = ModelEvaluator(model_dict)
        res = ev.evaluate()
        fig = ev._build_evaluation_figure(
            res["confusion_matrix"], res["feature_importances"]
        )
        assert fig is not None


def _fixed_training_model():
    """A tiny fitted model for a fast smoke test of the full evaluate path."""
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(1)
    X = pd.DataFrame({f: rng.uniform(0, 5, 80) for f in INPUT_FEATURES})
    le = LabelEncoder()
    y = le.fit_transform(rng.choice(["Win", "Draw", "Lose"], size=80))
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=1),
        n_estimators=5,
        random_state=1,
    )
    model.fit(X, y)
    return model, le
