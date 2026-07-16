"""
tests.test_evaluation
=====================
Test suite for the Aletheia evaluation layer.

Coverage
--------
Unit tests — RegressionEvaluator
    - Returns frozen RegressionMetrics dataclass
    - RMSE equals 10.0 for uniform prediction errors of 10
    - MAE equals 10.0 for uniform prediction errors of 10
    - Perfect predictions yield RMSE=0 and R²=1
    - Empty arrays raise ValueError
    - Mismatched lengths raise ValueError
    - pd.Series inputs accepted

Unit tests — RegressionPlotter
    - All four plot methods return matplotlib Figure objects
    - plot_actual_vs_predicted accepts pd.Series inputs
    - Empty inputs raise ValueError
    - Mismatched lengths raise ValueError
    - plot_feature_importance with empty DataFrame raises ValueError
    - plot_feature_importance with missing columns raises ValueError

Integration tests — EvaluationReporter
    - evaluate() returns frozen EvaluationReport
    - Metrics match direct RegressionEvaluator output
    - All three base figures are Figure instances
    - feature_importance_fig is None when not supplied
    - feature_importance_fig is a Figure when supplied
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from aletheia.evaluation.metrics import RegressionEvaluator, RegressionMetrics
from aletheia.evaluation.plots import RegressionPlotter
from aletheia.evaluation.reporter import EvaluationReport, EvaluationReporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def known_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic y_true / y_pred pair with uniform errors of exactly 10.

    RMSE = MAE = 10.0 for this dataset.
    """
    y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0, 510.0])
    return y_true, y_pred


@pytest.fixture(scope="module")
def feature_importance_df() -> pd.DataFrame:
    """Minimal valid feature importance DataFrame with four features."""
    return pd.DataFrame({
        "feature": ["spend", "clicks", "impressions", "ctr"],
        "importance": [0.45, 0.30, 0.15, 0.10],
    })


# ===========================================================================
# 1. RegressionEvaluator
# ===========================================================================

class TestRegressionEvaluator:

    def test_returns_regression_metrics_instance(self, known_arrays):
        y_true, y_pred = known_arrays
        result = RegressionEvaluator().evaluate(y_true, y_pred)
        assert isinstance(result, RegressionMetrics)

    def test_result_is_frozen(self, known_arrays):
        y_true, y_pred = known_arrays
        result = RegressionEvaluator().evaluate(y_true, y_pred)
        with pytest.raises((AttributeError, TypeError)):
            result.rmse = 0.0  # type: ignore[misc]

    def test_rmse_correct_for_uniform_errors(self, known_arrays):
        """All errors are exactly 10 — RMSE must equal 10.0."""
        y_true, y_pred = known_arrays
        result = RegressionEvaluator().evaluate(y_true, y_pred)
        assert result.rmse == pytest.approx(10.0, rel=1e-6)

    def test_mae_correct_for_uniform_errors(self, known_arrays):
        """All absolute errors are exactly 10 — MAE must equal 10.0."""
        y_true, y_pred = known_arrays
        result = RegressionEvaluator().evaluate(y_true, y_pred)
        assert result.mae == pytest.approx(10.0, rel=1e-6)

    def test_perfect_predictions_rmse_zero(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        result = RegressionEvaluator().evaluate(y, y)
        assert result.rmse == pytest.approx(0.0, abs=1e-9)

    def test_perfect_predictions_r2_one(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        result = RegressionEvaluator().evaluate(y, y)
        assert result.r2 == pytest.approx(1.0, abs=1e-9)

    def test_r2_at_most_one(self, known_arrays):
        y_true, y_pred = known_arrays
        result = RegressionEvaluator().evaluate(y_true, y_pred)
        assert result.r2 <= 1.0

    def test_mape_non_negative(self, known_arrays):
        y_true, y_pred = known_arrays
        result = RegressionEvaluator().evaluate(y_true, y_pred)
        assert result.mape >= 0.0

    def test_pandas_series_accepted(self):
        y_true = pd.Series([100.0, 200.0, 300.0])
        y_pred = pd.Series([110.0, 190.0, 310.0])
        result = RegressionEvaluator().evaluate(y_true, y_pred)
        assert isinstance(result, RegressionMetrics)

    def test_empty_y_true_raises(self):
        with pytest.raises(ValueError):
            RegressionEvaluator().evaluate(np.array([]), np.array([1.0]))

    def test_empty_y_pred_raises(self):
        with pytest.raises(ValueError):
            RegressionEvaluator().evaluate(np.array([1.0]), np.array([]))

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="lengths"):
            RegressionEvaluator().evaluate(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0]),
            )


# ===========================================================================
# 2. RegressionPlotter
# ===========================================================================

class TestRegressionPlotter:

    def test_actual_vs_predicted_returns_figure(self, known_arrays):
        y_true, y_pred = known_arrays
        fig = RegressionPlotter().plot_actual_vs_predicted(y_true, y_pred)
        assert isinstance(fig, Figure)

    def test_residuals_returns_figure(self, known_arrays):
        y_true, y_pred = known_arrays
        fig = RegressionPlotter().plot_residuals(y_true, y_pred)
        assert isinstance(fig, Figure)

    def test_prediction_error_returns_figure(self, known_arrays):
        y_true, y_pred = known_arrays
        fig = RegressionPlotter().plot_prediction_error(y_true, y_pred)
        assert isinstance(fig, Figure)

    def test_feature_importance_returns_figure(self, feature_importance_df):
        fig = RegressionPlotter().plot_feature_importance(feature_importance_df)
        assert isinstance(fig, Figure)

    def test_actual_vs_predicted_accepts_pandas_series(self):
        y_true = pd.Series([100.0, 200.0, 300.0])
        y_pred = pd.Series([105.0, 195.0, 305.0])
        fig = RegressionPlotter().plot_actual_vs_predicted(y_true, y_pred)
        assert isinstance(fig, Figure)

    def test_actual_vs_predicted_empty_raises(self):
        with pytest.raises(ValueError):
            RegressionPlotter().plot_actual_vs_predicted(
                np.array([]), np.array([])
            )

    def test_residuals_mismatched_raises(self):
        with pytest.raises(ValueError, match="Lengths"):
            RegressionPlotter().plot_residuals(
                np.array([1.0, 2.0]),
                np.array([1.0]),
            )

    def test_prediction_error_mismatched_raises(self):
        with pytest.raises(ValueError, match="Lengths"):
            RegressionPlotter().plot_prediction_error(
                np.array([1.0, 2.0]),
                np.array([1.0]),
            )

    def test_feature_importance_empty_raises(self):
        with pytest.raises(ValueError):
            RegressionPlotter().plot_feature_importance(pd.DataFrame())

    def test_feature_importance_missing_importance_column_raises(self):
        df = pd.DataFrame({"feature": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="importance"):
            RegressionPlotter().plot_feature_importance(df)

    def test_feature_importance_missing_feature_column_raises(self):
        df = pd.DataFrame({"importance": [0.5, 0.3, 0.2]})
        with pytest.raises(ValueError, match="importance"):
            RegressionPlotter().plot_feature_importance(df)


# ===========================================================================
# 3. Integration — EvaluationReporter
# ===========================================================================

class TestEvaluationReporter:

    def test_returns_evaluation_report(self, known_arrays):
        y_true, y_pred = known_arrays
        report = EvaluationReporter().evaluate(y_true, y_pred)
        assert isinstance(report, EvaluationReport)

    def test_report_is_frozen(self, known_arrays):
        y_true, y_pred = known_arrays
        report = EvaluationReporter().evaluate(y_true, y_pred)
        with pytest.raises((AttributeError, TypeError)):
            report.metrics = None  # type: ignore[misc]

    def test_metrics_matches_direct_evaluator(self, known_arrays):
        """Reporter metrics must be identical to direct RegressionEvaluator output."""
        y_true, y_pred = known_arrays
        direct = RegressionEvaluator().evaluate(y_true, y_pred)
        report = EvaluationReporter().evaluate(y_true, y_pred)
        assert report.metrics.rmse == pytest.approx(direct.rmse, rel=1e-9)
        assert report.metrics.mae == pytest.approx(direct.mae, rel=1e-9)
        assert report.metrics.r2 == pytest.approx(direct.r2, rel=1e-9)

    def test_actual_vs_predicted_fig_is_figure(self, known_arrays):
        y_true, y_pred = known_arrays
        report = EvaluationReporter().evaluate(y_true, y_pred)
        assert isinstance(report.actual_vs_predicted_fig, Figure)

    def test_residuals_fig_is_figure(self, known_arrays):
        y_true, y_pred = known_arrays
        report = EvaluationReporter().evaluate(y_true, y_pred)
        assert isinstance(report.residuals_fig, Figure)

    def test_prediction_error_fig_is_figure(self, known_arrays):
        y_true, y_pred = known_arrays
        report = EvaluationReporter().evaluate(y_true, y_pred)
        assert isinstance(report.prediction_error_fig, Figure)

    def test_feature_importance_fig_none_by_default(self, known_arrays):
        y_true, y_pred = known_arrays
        report = EvaluationReporter().evaluate(y_true, y_pred)
        assert report.feature_importance_fig is None

    def test_feature_importance_fig_present_when_supplied(
        self, known_arrays, feature_importance_df
    ):
        y_true, y_pred = known_arrays
        report = EvaluationReporter().evaluate(
            y_true, y_pred, feature_importance=feature_importance_df
        )
        assert isinstance(report.feature_importance_fig, Figure)

    def test_pandas_series_inputs_accepted(self):
        y_true = pd.Series([100.0, 200.0, 300.0, 400.0])
        y_pred = pd.Series([105.0, 195.0, 305.0, 395.0])
        report = EvaluationReporter().evaluate(y_true, y_pred)
        assert isinstance(report, EvaluationReport)
