"""
tests.test_models
=================
Test suite for the Aletheia model layer.

Coverage
--------
Unit tests — ModelTrainer
    - Raises TypeError for a non-BaseModel argument
    - Raises ValueError for test_size outside (0.0, 1.0)
    - Raises ValueError for empty X
    - Raises ValueError for mismatched X/y lengths
    - Returns a frozen TrainingResult with correct model_name
    - train_rows + validation_rows equals total dataset size
    - Model artefact written to disk when model_output_path is provided

Unit tests — LightGBMModel
    - fit() returns self (method chaining)
    - is_fitted is False before fit; True after fit
    - predict() returns a numpy ndarray of correct length
    - predict() on unfitted model raises RuntimeError
    - predict() with empty DataFrame raises ValueError
    - feature_importance returns a DataFrame with 'feature' and 'importance' columns
    - feature_importance sorted descending by importance
    - feature_importance on unfitted model raises RuntimeError
    - save() creates the artefact file on disk
    - save() on unfitted model raises RuntimeError
    - load() restores the model; is_fitted becomes True; predict() works
    - load() on a missing path raises FileNotFoundError
    - fit() with X_val / y_val succeeds; is_fitted True
    - fit() with X_val but no y_val raises ValueError

Unit tests — ModelPredictor
    - Raises TypeError for a non-BaseModel argument
    - predict() on unfitted model raises RuntimeError
    - predict() with empty DataFrame raises ValueError
    - predict() returns a DataFrame with a 'prediction' column
    - Output length matches input length
    - Metadata columns (date, platform, campaign_id, campaign_name) propagated
    - prediction dtype is float64
    - load_model() restores and enables inference
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aletheia.models.lightgbm_model import LightGBMModel
from aletheia.models.predictor import ModelPredictor
from aletheia.models.trainer import ModelTrainer, TrainingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_feature_df(n_rows: int = 200, n_features: int = 10, seed: int = 42) -> pd.DataFrame:
    """Synthetic numeric feature DataFrame for model tests."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"feature_{i:02d}": rng.standard_normal(n_rows) for i in range(n_features)}
    )


def _make_target(n_rows: int = 200, seed: int = 99) -> pd.Series:
    """Synthetic target vector uniformly distributed in [100, 1000]."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(100.0, 1000.0, n_rows), name="revenue")


@pytest.fixture(scope="module")
def X() -> pd.DataFrame:
    """200-row synthetic feature matrix with 10 numeric columns."""
    return _make_feature_df(n_rows=200, n_features=10)


@pytest.fixture(scope="module")
def y(X) -> pd.Series:
    """200-element synthetic target vector aligned with X."""
    return _make_target(n_rows=len(X))


@pytest.fixture(scope="module")
def fitted_model(X, y) -> LightGBMModel:
    """LightGBMModel fitted on synthetic data with minimal boosting rounds."""
    model = LightGBMModel(num_boost_round=20, early_stopping_rounds=5)
    model.fit(X, y)
    return model


# ===========================================================================
# 1. ModelTrainer
# ===========================================================================

class TestModelTrainer:

    def test_non_basemodel_raises_type_error(self):
        with pytest.raises(TypeError, match="BaseModel"):
            ModelTrainer(model="not_a_model")  # type: ignore[arg-type]

    def test_test_size_above_one_raises(self):
        with pytest.raises(ValueError, match="test_size"):
            ModelTrainer(model=LightGBMModel(), test_size=1.5)

    def test_test_size_zero_raises(self):
        with pytest.raises(ValueError, match="test_size"):
            ModelTrainer(model=LightGBMModel(), test_size=0.0)

    def test_empty_x_raises(self, y):
        trainer = ModelTrainer(model=LightGBMModel(), test_size=0.2)
        with pytest.raises(ValueError, match="empty"):
            trainer.train(pd.DataFrame(), y)

    def test_mismatched_xy_lengths_raise(self, X):
        trainer = ModelTrainer(model=LightGBMModel(), test_size=0.2)
        short_y = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="lengths"):
            trainer.train(X, short_y)

    def test_returns_training_result(self, X, y):
        result = ModelTrainer(
            model=LightGBMModel(num_boost_round=10), test_size=0.2
        ).train(X, y)
        assert isinstance(result, TrainingResult)

    def test_training_result_is_frozen(self, X, y):
        result = ModelTrainer(
            model=LightGBMModel(num_boost_round=10), test_size=0.2
        ).train(X, y)
        with pytest.raises((AttributeError, TypeError)):
            result.rmse = 0.0  # type: ignore[misc]

    def test_model_name_in_result(self, X, y):
        result = ModelTrainer(
            model=LightGBMModel(num_boost_round=10), test_size=0.2
        ).train(X, y)
        assert result.model_name == "lightgbm"

    def test_row_counts_sum_to_total(self, X, y):
        result = ModelTrainer(
            model=LightGBMModel(num_boost_round=10),
            test_size=0.2,
            random_state=42,
        ).train(X, y)
        assert result.train_rows + result.validation_rows == len(X)

    def test_rmse_non_negative(self, X, y):
        result = ModelTrainer(
            model=LightGBMModel(num_boost_round=10), test_size=0.2
        ).train(X, y)
        assert result.rmse >= 0.0

    def test_model_saved_when_path_given(self, X, y, tmp_path):
        model = LightGBMModel(num_boost_round=10)
        output_path = tmp_path / "model.txt"
        ModelTrainer(model=model, test_size=0.2).train(
            X, y, model_output_path=output_path
        )
        assert output_path.exists()


# ===========================================================================
# 2. LightGBMModel
# ===========================================================================

class TestLightGBMModel:

    def test_fit_returns_self(self, X, y):
        model = LightGBMModel(num_boost_round=10)
        assert model.fit(X, y) is model

    def test_is_fitted_false_before_fit(self):
        assert LightGBMModel().is_fitted is False

    def test_is_fitted_true_after_fit(self, fitted_model):
        assert fitted_model.is_fitted is True

    def test_predict_returns_ndarray(self, fitted_model, X):
        assert isinstance(fitted_model.predict(X), np.ndarray)

    def test_predict_length_matches_input(self, fitted_model, X):
        assert len(fitted_model.predict(X)) == len(X)

    def test_predict_unfitted_raises_runtime_error(self, X):
        with pytest.raises(RuntimeError, match="not been fitted"):
            LightGBMModel().predict(X)

    def test_predict_empty_raises_value_error(self, fitted_model):
        with pytest.raises(ValueError, match="empty"):
            fitted_model.predict(pd.DataFrame())

    def test_feature_importance_returns_dataframe(self, fitted_model):
        assert isinstance(fitted_model.feature_importance, pd.DataFrame)

    def test_feature_importance_has_required_columns(self, fitted_model):
        fi = fitted_model.feature_importance
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_feature_importance_sorted_descending(self, fitted_model):
        importances = fitted_model.feature_importance["importance"].tolist()
        assert importances == sorted(importances, reverse=True)

    def test_feature_importance_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = LightGBMModel().feature_importance

    def test_save_creates_file(self, fitted_model, tmp_path):
        path = tmp_path / "lgbm.txt"
        fitted_model.save(path)
        assert path.exists()

    def test_save_unfitted_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="not been fitted"):
            LightGBMModel().save(tmp_path / "lgbm.txt")

    def test_load_roundtrip(self, fitted_model, X, tmp_path):
        """Saved model must produce predictions after being loaded into a fresh instance."""
        path = tmp_path / "lgbm_roundtrip.txt"
        fitted_model.save(path)

        loaded = LightGBMModel()
        loaded.load(path)

        assert loaded.is_fitted is True
        preds = loaded.predict(X)
        assert len(preds) == len(X)

    def test_load_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LightGBMModel().load(tmp_path / "does_not_exist.txt")

    def test_fit_with_validation_set(self, X, y):
        """Passing X_val and y_val must not raise and must result in a fitted model."""
        model = LightGBMModel(num_boost_round=10, early_stopping_rounds=5)
        model.fit(
            X.iloc[:160], y.iloc[:160],
            X_val=X.iloc[160:], y_val=y.iloc[160:],
        )
        assert model.is_fitted is True

    def test_fit_x_val_without_y_val_raises(self, X, y):
        with pytest.raises(ValueError):
            LightGBMModel(num_boost_round=10).fit(
                X.iloc[:160], y.iloc[:160],
                X_val=X.iloc[160:],
            )

    def test_name_class_attribute(self):
        assert LightGBMModel.name == "lightgbm"

    def test_repr_contains_status(self):
        model = LightGBMModel()
        assert "unfitted" in repr(model)
        assert "lightgbm" in repr(model)


# ===========================================================================
# 3. ModelPredictor
# ===========================================================================

class TestModelPredictor:

    def test_non_basemodel_raises_type_error(self):
        with pytest.raises(TypeError, match="BaseModel"):
            ModelPredictor(model="not_a_model")  # type: ignore[arg-type]

    def test_unfitted_model_raises_on_predict(self, X):
        with pytest.raises(RuntimeError, match="not been fitted"):
            ModelPredictor(model=LightGBMModel()).predict(X)

    def test_empty_x_raises_value_error(self, fitted_model):
        with pytest.raises(ValueError, match="empty"):
            ModelPredictor(model=fitted_model).predict(pd.DataFrame())

    def test_returns_dataframe(self, fitted_model, X):
        result = ModelPredictor(model=fitted_model).predict(X)
        assert isinstance(result, pd.DataFrame)

    def test_prediction_column_present(self, fitted_model, X):
        result = ModelPredictor(model=fitted_model).predict(X)
        assert "prediction" in result.columns

    def test_output_length_matches_input(self, fitted_model, X):
        result = ModelPredictor(model=fitted_model).predict(X)
        assert len(result) == len(X)

    def test_prediction_dtype_float64(self, fitted_model, X):
        result = ModelPredictor(model=fitted_model).predict(X)
        assert result["prediction"].dtype == np.float64

    def test_metadata_columns_propagated(self, fitted_model, X):
        """Canonical metadata columns present in X must appear in the output."""
        X_with_meta = X.copy()
        X_with_meta["date"] = pd.date_range("2024-01-01", periods=len(X))
        X_with_meta["platform"] = "google_ads"
        X_with_meta["campaign_id"] = "c001"
        X_with_meta["campaign_name"] = "Brand Campaign"

        result = ModelPredictor(model=fitted_model).predict(X_with_meta)

        assert "date" in result.columns
        assert "platform" in result.columns
        assert "campaign_id" in result.columns
        assert "campaign_name" in result.columns
        assert "prediction" in result.columns

    def test_load_model_enables_inference(self, fitted_model, X, tmp_path):
        """load_model() must make a fresh predictor ready to serve predictions."""
        path = tmp_path / "predictor_model.txt"
        fitted_model.save(path)

        predictor = ModelPredictor(model=LightGBMModel())
        predictor.load_model(path)

        result = predictor.predict(X)
        assert "prediction" in result.columns
        assert len(result) == len(X)
