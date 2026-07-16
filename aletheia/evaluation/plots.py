"""
aletheia.evaluation.plots
=========================
Visualization utilities for regression model evaluation.

Responsibilities
----------------
- Generating actual vs. predicted value line plots.
- Generating prediction error scatter plots (actual vs. predicted with y=x line).
- Generating residual scatter plots (residuals vs. predicted with y=0 line).
- Generating sorted feature importance horizontal bar plots.

Design notes
------------
- This module relies strictly on Matplotlib for plotting to ensure minimal
  dependencies and maximum flexibility. Seaborn is explicitly avoided.
- To prevent side effects in multi-threaded or notebook environments, all
  plotting functions create a new Figure and Axes object locally using
  ``plt.subplots()`` and return the Figure object directly.
- Neither ``plt.show()`` nor ``Figure.savefig()`` is called. The caller has
  full control over displaying, saving, or embedding the generated figures.
- Input validation is performed on all arguments. Mismatched lengths, empty
  data, or missing columns in DataFrames will raise a ``ValueError``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class RegressionPlotter:
    """
    Utility class for visualizing regression model performance.

    All plotting methods generate a new figure, apply professional styling,
    axis labels, grids, and return the Matplotlib Figure object to the caller.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_actual_vs_predicted(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> Figure:
        """
        Generate a line plot comparing actual vs. predicted values over time/samples.

        Parameters
        ----------
        y_true : pd.Series | np.ndarray
            Ground-truth target values.
        y_pred : pd.Series | np.ndarray
            Predicted target values.

        Returns
        -------
        Figure
            The Matplotlib Figure object containing the plot.

        Raises
        ------
        ValueError
            If inputs are empty or have mismatched lengths.
        """
        y_true_arr, y_pred_arr, index = self._validate_and_coerce(y_true, y_pred)

        logger.info("[RegressionPlotter] Plotting actual vs. predicted values.")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(index, y_true_arr, label="Actual", color="#1f77b4", linewidth=2, alpha=0.8)
        ax.plot(index, y_pred_arr, label="Predicted", color="#ff7f0e", linewidth=1.5, linestyle="--", alpha=0.9)

        ax.set_title("Actual vs. Predicted Values")
        ax.set_xlabel("Date / Sample Index" if isinstance(y_true, pd.Series) and isinstance(y_true.index, pd.DatetimeIndex) else "Sample Index")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.6)

        fig.tight_layout()
        return fig

    def plot_residuals(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> Figure:
        """
        Generate a residual plot showing residuals vs. predicted values.

        The standard residual definition is actual - predicted (y - y_hat).

        Parameters
        ----------
        y_true : pd.Series | np.ndarray
            Ground-truth target values.
        y_pred : pd.Series | np.ndarray
            Predicted target values.

        Returns
        -------
        Figure
            The Matplotlib Figure object containing the plot.

        Raises
        ------
        ValueError
            If inputs are empty or have mismatched lengths.
        """
        y_true_arr, y_pred_arr, _ = self._validate_and_coerce(y_true, y_pred)
        residuals = y_true_arr - y_pred_arr

        logger.info("[RegressionPlotter] Plotting residuals.")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(y_pred_arr, residuals, color="#2ca02c", alpha=0.6, edgecolors="none")
        ax.axhline(y=0.0, color="red", linestyle="--", linewidth=1.5, alpha=0.8)

        ax.set_title("Residuals vs. Predicted Values")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals (Actual - Predicted)")
        ax.grid(True, linestyle=":", alpha=0.6)

        fig.tight_layout()
        return fig

    def plot_prediction_error(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> Figure:
        """
        Generate a prediction error scatter plot (Actual vs. Predicted with y=x line).

        Parameters
        ----------
        y_true : pd.Series | np.ndarray
            Ground-truth target values.
        y_pred : pd.Series | np.ndarray
            Predicted target values.

        Returns
        -------
        Figure
            The Matplotlib Figure object containing the plot.

        Raises
        ------
        ValueError
            If inputs are empty or have mismatched lengths.
        """
        y_true_arr, y_pred_arr, _ = self._validate_and_coerce(y_true, y_pred)

        logger.info("[RegressionPlotter] Plotting prediction error.")

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(y_true_arr, y_pred_arr, color="#9467bd", alpha=0.6, edgecolors="none")

        # Identity line (y=x)
        min_val = min(y_true_arr.min(), y_pred_arr.min())
        max_val = max(y_true_arr.max(), y_pred_arr.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1.5, alpha=0.7)

        ax.set_title("Prediction Error Plot")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", alpha=0.6)

        fig.tight_layout()
        return fig

    def plot_feature_importance(
        self,
        feature_importance: pd.DataFrame,
    ) -> Figure:
        """
        Generate a horizontal bar plot showing feature importances sorted descending.

        Parameters
        ----------
        feature_importance : pd.DataFrame
            DataFrame containing 'feature' and 'importance' columns.

        Returns
        -------
        Figure
            The Matplotlib Figure object containing the plot.

        Raises
        ------
        ValueError
            If the DataFrame is empty or missing required columns.
        """
        if feature_importance.empty:
            raise ValueError("[RegressionPlotter] feature_importance DataFrame must not be empty.")

        required_cols = {"feature", "importance"}
        if not required_cols.issubset(feature_importance.columns):
            raise ValueError(
                f"[RegressionPlotter] feature_importance DataFrame must contain columns {required_cols}. "
                f"Got: {list(feature_importance.columns)}"
            )

        logger.info("[RegressionPlotter] Plotting feature importance.")

        # Sort descending and take top features for readability
        df_sorted = feature_importance.sort_values(by="importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(4, len(df_sorted) * 0.4)))

        y_pos = np.arange(len(df_sorted))
        ax.barh(y_pos, df_sorted["importance"], color="#bcbd22", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted["feature"])

        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance / Attribution Score")
        ax.set_ylabel("Features")
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_coerce(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | pd.Index]:
        """
        Validate input arrays and coerce to flat numpy arrays, returning index.

        Parameters
        ----------
        y_true : pd.Series | np.ndarray
            Ground-truth target values.
        y_pred : pd.Series | np.ndarray
            Predicted target values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray | pd.Index]
            Flattened y_true, y_pred, and index/sequence to plot against.

        Raises
        ------
        ValueError
            If inputs are empty or lengths do not match.
        """
        y_true_arr = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()

        if y_true_arr.size == 0 or y_pred_arr.size == 0:
            raise ValueError("[RegressionPlotter] Input arrays must not be empty.")

        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError(
                f"[RegressionPlotter] Lengths of y_true and y_pred must match. "
                f"Got y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}"
            )

        # Get index if y_true is a pandas Series, otherwise use a simple range
        if isinstance(y_true, pd.Series):
            index = y_true.index
        else:
            index = np.arange(len(y_true_arr))

        return y_true_arr, y_pred_arr, index
