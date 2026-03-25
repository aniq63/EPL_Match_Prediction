import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform

from src.utils.logger import logging
from src.utils.exception import MyException

from config.constants import (
    INPUT_FEATURES,
    MODEL_NAME,
    RESULT_CLASSES,
    N_ITER,
    CV_FOLDS,
    RANDOM_STATE,
    SCORING,
)

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Constants used in training
# ============================================================

# MODEL_NAME      = "AdaBoostClassifier"
# RESULT_CLASSES  = ["Win", "Draw", "Lose"]
# N_ITER          = 20
# CV_FOLDS        = 5
# RANDOM_STATE    = 42
# SCORING         = "accuracy"

PARAM_DIST = {
    "estimator__max_depth":        randint(1, 8),
    "estimator__min_samples_leaf": randint(1, 20),
    "estimator__min_samples_split":randint(2, 20),
    "estimator__max_features":     ["sqrt", "log2", None],
    "n_estimators":                randint(100, 600),
    "learning_rate":               uniform(0.01, 0.49),
}


# ============================================================
# ModelTrainer Class
# ============================================================

class ModelTrainer:
    """
    Trains an AdaBoostClassifier (with DecisionTree base estimator) on
    EPL match data using RandomizedSearchCV + TimeSeriesSplit for tuning.

    Args:
        train_df (pd.DataFrame): Training split of the transformed dataset.
        test_df  (pd.DataFrame): Test split of the transformed dataset.
        input_features (list):  Feature column names. Defaults to INPUT_FEATURES constant.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        input_features: list = INPUT_FEATURES,
    ):
        self.train_df       = train_df.copy()
        self.test_df        = test_df.copy()
        self.input_features = input_features
        self.le             = LabelEncoder()
        self.le.fit(RESULT_CLASSES)
        self.model          = None
        self.model_params   = {}

        logging.info(
            f"ModelTrainer initialized | "
            f"train={len(train_df)} rows, test={len(test_df)} rows, "
            f"features={len(input_features)}"
        )

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _validate_features(self, df: pd.DataFrame, name: str):
        """Check that all required feature columns exist in the dataframe."""
        missing = [f for f in self.input_features if f not in df.columns]
        if missing:
            raise ValueError(
                f"[{name}] Missing feature columns: {missing}"
            )

    def _prepare_data(self):
        """Validate and encode labels; returns X_train, y_train, X_test, y_test."""
        logging.info("Validating feature columns...")
        self._validate_features(self.train_df, "train_df")
        self._validate_features(self.test_df, "test_df")

        X_train = self.train_df[self.input_features]
        y_train = self.le.transform(self.train_df["result"])

        X_test  = self.test_df[self.input_features]
        y_test  = self.le.transform(self.test_df["result"])

        logging.info(
            f"Data prepared | X_train={X_train.shape}, X_test={X_test.shape}"
        )
        logging.info(f"Label classes: {list(self.le.classes_)}")
        return X_train, y_train, X_test, y_test

    def _run_hyperparameter_search(self, X_train, y_train):
        """Run RandomizedSearchCV and return the search object."""
        logging.info(
            f"Starting RandomizedSearchCV | "
            f"n_iter={N_ITER}, cv_folds={CV_FOLDS}, scoring={SCORING}"
        )

        search_base = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            # algorithm="SAMME",
            random_state=RANDOM_STATE,
        )

        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

        random_search = RandomizedSearchCV(
            estimator=search_base,
            param_distributions=PARAM_DIST,
            n_iter=N_ITER,
            scoring=SCORING,
            cv=tscv,
            n_jobs=-1,
            verbose=1,
            random_state=RANDOM_STATE,
            refit=True,
            return_train_score=True,
        )

        random_search.fit(X_train, y_train)

        logging.info(
            f"Hyperparameter search complete | "
            f"Best CV score: {random_search.best_score_:.4f}"
        )
        logging.info(f"Best params: {random_search.best_params_}")
        return random_search

    def _build_final_model(self, best_params: dict):
        """Construct and return the final AdaBoostClassifier from best params."""
        bp = best_params
        tuned_tree = DecisionTreeClassifier(
            max_depth=bp["estimator__max_depth"],
            min_samples_leaf=bp["estimator__min_samples_leaf"],
            min_samples_split=bp["estimator__min_samples_split"],
            max_features=bp["estimator__max_features"],
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )

        model = AdaBoostClassifier(
            estimator=tuned_tree,
            n_estimators=bp["n_estimators"],
            learning_rate=bp["learning_rate"],
            # algorithm="SAMME",
            random_state=RANDOM_STATE,
        )

        # Store all final params for evaluation / reproducibility
        self.model_params = {
            "model_name":           MODEL_NAME,
            "n_estimators":         bp["n_estimators"],
            "learning_rate":        bp["learning_rate"],
            "max_depth":             bp["estimator__max_depth"],
            "min_samples_leaf":     bp["estimator__min_samples_leaf"],
            "min_samples_split":    bp["estimator__min_samples_split"],
            "max_features":         bp["estimator__max_features"],
            "algorithm":            "SAMME",
            "random_state":         RANDOM_STATE,
            "n_iter_search":        N_ITER,
            "cv_folds":             CV_FOLDS,
            "scoring":              SCORING,
            "input_features":       self.input_features,
            "label_classes":        list(self.le.classes_),
        }

        return model

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def train(self) -> dict:
        """
        Run the full training pipeline:
          1. Validate & prepare data
          2. RandomizedSearchCV for hyperparameter tuning
          3. Build and fit final model

        Returns:
            dict with keys:
              - "model_name"    (str)
              - "model"         (fitted AdaBoostClassifier)
              - "params"        (dict of all model + search params)
              - "label_encoder" (fitted LabelEncoder)
              - "X_test"        (pd.DataFrame, for evaluation)
              - "y_test"        (np.ndarray, encoded labels, for evaluation)
        """
        try:
            logging.info("=" * 60)
            logging.info(f"MODEL TRAINING START | {MODEL_NAME}")
            logging.info("=" * 60)

            X_train, y_train, X_test, y_test = self._prepare_data()

            # Hyperparameter search
            random_search = self._run_hyperparameter_search(X_train, y_train)

            # Build final model from best params
            self.model = self._build_final_model(random_search.best_params_)

            # Fit final model on full training set
            logging.info(f"Fitting final model on {len(X_train)} training rows...")
            self.model.fit(X_train, y_train)

            logging.info("=" * 60)
            logging.info("MODEL TRAINING COMPLETE")
            logging.info(f"Final params: {self.model_params}")
            logging.info("=" * 60)

            return {
                "model_name":    MODEL_NAME,
                "model":         self.model,
                "params":        self.model_params,
                "label_encoder": self.le,
                "X_test":        X_test,
                "y_test":        y_test,
            }

        except ValueError as ve:
            logging.error(f"Validation error during training: {ve}")
            raise MyException(ve, sys)
        except Exception as e:
            logging.error(f"Unexpected error during training: {e}")
            raise MyException(e, sys)