"""
aletheia.evaluation.metrics
============================
Reusable regression evaluation metrics for the Aletheia ML pipeline.

Responsibilities
----------------
- Accepting ground-truth and predicted arrays (NumPy or Pandas).
- Computing RMSE, MAE, R², and MAPE via scikit-learn.
- Returning results as an immutable, inspectable :class:`RegressionMetrics`
  dataclass.

Design notes
------------
- All metric computations are delegated to :mod:`sklearn.metrics`.  No
  formulas are implemented manually; this guards against subtle numerical
  errors and keeps the module aligned with the broader scientific Python
  ecosystem.
- RMSE is derived from ``sqrt(mean_squared_error(...))`` rather than
  ``mean_squared_error(..., squared=False)`` (deprecated in sklearn ≥ 1.4)
  or ``root_mean_squared_error`` (added in sklearn 1.4) so that the module
  works correctly across a wide range of installed sklearn versions.
- Both :class:`numpy.ndarray` and :class:`pandas.Series` inputs are accepted
  and normalised to NumPy arrays before any computation, preventing
  index-alignment surprises that can occur when two Series with different
  indices are passed.
- This module is strictly for regression evaluation.  Classification metrics,
  confusion matrices, and plotting are explicitly out of scope.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegressionMetrics:
    """
    Immutable container for regression evaluation results.

    All metrics are computed on the validation or test split and summarise
    the quality of a model's predictions against ground-truth target values.

    Attributes
    ----------
    rmse : float
        Root Mean Squared Error.  Lower is better.  Expressed in the same
        units as the target variable.
    mae : float
        Mean Absolute Error.  Lower is better.  More robust to outliers than
        RMSE because it does not square individual errors.
    r2 : float
        Coefficient of determination (R²).  A value of 1.0 indicates a
        perfect fit; 0.0 indicates the model performs no better than
        predicting the mean; negative values indicate the model is worse
        than the mean baseline.
    mape : float
        Mean Absolute Percentage Error.  Expressed as a decimal fraction
        (e.g. ``0.10`` = 10 %).  Undefined (and excluded from logging) when
        any true value is zero.
    """

    rmse: float
    mae: float
    r2: float
    mape: float

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"RegressionMetrics("
            f"RMSE={self.rmse:.4f}, "
            f"MAE={self.mae:.4f}, "
            f"R²={self.r2:.4f}, "
            f"MAPE={self.mape * 100:.2f}%)"
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RegressionEvaluator:
    """
    Stateless evaluator that computes regression metrics from predictions.

    The evaluator has no internal state beyond optional construction-time
    configuration.  It does not train models, generate predictions, or
    produce plots.  Its single responsibility is to transform a pair of
    ground-truth / prediction arrays into a :class:`RegressionMetrics`
    dataclass.

    All metric computations are delegated to :mod:`sklearn.metrics`.

    Example
    -------
    .. code-block:: python

        import numpy as np
        from aletheia.evaluation.metrics import RegressionEvaluator

        evaluator = RegressionEvaluator()
        metrics = evaluator.evaluate(y_true=np.array([1, 2, 3]),
                                     y_pred=np.array([1.1, 1.9, 3.2]))
        print(metrics)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> RegressionMetrics:
        """
        Compute RMSE, MAE, R², and MAPE for the provided arrays.

        Parameters
        ----------
        y_true : pd.Series | np.ndarray
            Ground-truth target values.  Shape ``(n_samples,)``.
        y_pred : pd.Series | np.ndarray
            Predicted values corresponding to ``y_true``.
            Shape ``(n_samples,)``.

        Returns
        -------
        RegressionMetrics
            Frozen dataclass containing all computed metrics.

        Raises
        ------
        ValueError
            If either array is empty, or if their lengths differ.
        """
        y_true_arr, y_pred_arr = self._validate_and_coerce(y_true, y_pred)

        logger.debug(
            "[RegressionEvaluator] Evaluating %d samples.", len(y_true_arr)
        )

        rmse: float = math.sqrt(
            float(mean_squared_error(y_true_arr, y_pred_arr))
        )
        mae: float = float(mean_absolute_error(y_true_arr, y_pred_arr))
        r2: float = float(r2_score(y_true_arr, y_pred_arr))
        mape: float = float(
            mean_absolute_percentage_error(y_true_arr, y_pred_arr)
        )

        metrics = RegressionMetrics(rmse=rmse, mae=mae, r2=r2, mape=mape)

        logger.info(
            "[RegressionEvaluator] RMSE=%.4f | MAE=%.4f | R²=%.4f | MAPE=%.2f%%",
            metrics.rmse,
            metrics.mae,
            metrics.r2,
            metrics.mape * 100,
        )

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_coerce(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate inputs and normalise them to flat NumPy float64 arrays.

        Converting both inputs to NumPy before passing them to sklearn
        prevents index-alignment issues that can occur when two
        :class:`pandas.Series` with mismatched indices are supplied.

        Parameters
        ----------
        y_true : pd.Series | np.ndarray
            Ground-truth array to validate and coerce.
        y_pred : pd.Series | np.ndarray
            Predicted array to validate and coerce.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A pair of one-dimensional ``float64`` NumPy arrays ready for
            metric computation.

        Raises
        ------
        ValueError
            If either array is empty, or if their lengths differ.
        """
        # Coerce to ndarray — works for both Series and ndarray inputs.
        y_true_arr = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()

        if y_true_arr.size == 0:
            raise ValueError(
                "[RegressionEvaluator] y_true must not be empty."
            )
        if y_pred_arr.size == 0:
            raise ValueError(
                "[RegressionEvaluator] y_pred must not be empty."
            )
        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError(
                f"[RegressionEvaluator] y_true and y_pred lengths must match. "
                f"Got y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}."
            )

        return y_true_arr, y_pred_arr
