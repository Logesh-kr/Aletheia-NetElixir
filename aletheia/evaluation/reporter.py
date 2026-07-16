"""
aletheia.evaluation.reporter
============================
Combines metric computation and plot generation into a single evaluation session.

Responsibilities
----------------
- Accepting ground-truth and predicted arrays alongside an optional feature
  importance DataFrame.
- Delegating all metric computation to
  :class:`~aletheia.evaluation.metrics.RegressionEvaluator`.
- Delegating all plot generation to
  :class:`~aletheia.evaluation.plots.RegressionPlotter`.
- Packaging everything into an immutable :class:`EvaluationReport` dataclass
  returned to the caller.

Design notes
------------
- The reporter is a thin orchestration layer.  It owns no metric formulas and
  no plotting logic; those responsibilities live exclusively in
  :mod:`aletheia.evaluation.metrics` and :mod:`aletheia.evaluation.plots`.
- :class:`EvaluationReport` stores Matplotlib :class:`~matplotlib.figure.Figure`
  objects, not rendered image bytes or file paths.  Callers decide whether to
  display, save, or embed the figures — this class never calls ``plt.show()``
  or writes to disk.
- ``feature_importance_fig`` is ``None`` when no feature importance DataFrame
  is supplied, so the reporter degrades gracefully for models that do not
  expose feature attributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from aletheia.evaluation.metrics import RegressionEvaluator, RegressionMetrics
from aletheia.evaluation.plots import RegressionPlotter

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationReport:
    """
    Immutable container holding all outputs of a single evaluation run.

    Attributes
    ----------
    metrics : RegressionMetrics
        Computed RMSE, MAE, R², and MAPE for the evaluated predictions.
    actual_vs_predicted_fig : Figure
        Line plot of actual vs. predicted values over the sample sequence.
    residuals_fig : Figure
        Scatter plot of residuals (actual − predicted) vs. predicted values.
    prediction_error_fig : Figure
        Scatter plot of actual vs. predicted with the identity line (y = x).
    feature_importance_fig : Figure | None
        Horizontal bar chart of feature importances sorted descending.
        ``None`` when no feature importance DataFrame was supplied.
    """

    metrics: RegressionMetrics
    actual_vs_predicted_fig: "Figure"
    residuals_fig: "Figure"
    prediction_error_fig: "Figure"
    feature_importance_fig: "Figure | None"

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"EvaluationReport("
            f"RMSE={self.metrics.rmse:.4f}, "
            f"MAE={self.metrics.mae:.4f}, "
            f"R²={self.metrics.r2:.4f}, "
            f"MAPE={self.metrics.mape * 100:.2f}%, "
            f"feature_importance={'yes' if self.feature_importance_fig is not None else 'no'})"
        )


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class EvaluationReporter:
    """
    Orchestrates metric computation and plot generation for a regression model.

    The reporter delegates all computation to :class:`RegressionEvaluator`
    and all visualisation to :class:`RegressionPlotter`.  Its sole
    responsibility is to coordinate these two concerns and return a single
    :class:`EvaluationReport`.

    Example
    -------
    .. code-block:: python

        import numpy as np
        from aletheia.evaluation.reporter import EvaluationReporter

        reporter = EvaluationReporter()
        report = reporter.evaluate(
            y_true=np.array([100.0, 200.0, 300.0]),
            y_pred=np.array([110.0, 195.0, 290.0]),
        )
        print(report.metrics)
        report.actual_vs_predicted_fig.savefig("avp.png")
    """

    def __init__(self) -> None:
        self._evaluator = RegressionEvaluator()
        self._plotter = RegressionPlotter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        feature_importance: pd.DataFrame | None = None,
    ) -> EvaluationReport:
        """
        Compute metrics and generate all evaluation plots.

        Parameters
        ----------
        y_true : pd.Series | np.ndarray
            Ground-truth target values.  Shape ``(n_samples,)``.
        y_pred : pd.Series | np.ndarray
            Predicted target values corresponding to ``y_true``.
            Shape ``(n_samples,)``.
        feature_importance : pd.DataFrame | None
            Optional DataFrame with ``"feature"`` and ``"importance"`` columns
            used to generate the feature importance bar chart.  Pass ``None``
            (default) to omit the importance plot from the report.

        Returns
        -------
        EvaluationReport
            Frozen dataclass containing computed metrics and all Matplotlib
            figures.

        Raises
        ------
        ValueError
            If ``y_true`` or ``y_pred`` are empty, or their lengths differ.
        ValueError
            If ``feature_importance`` is provided but is empty or missing the
            required ``"feature"`` / ``"importance"`` columns.
        """
        logger.info("[EvaluationReporter] Starting evaluation.")

        metrics = self._evaluator.evaluate(y_true, y_pred)

        n_samples = len(np.asarray(y_true).ravel())
        logger.info(
            "[EvaluationReporter] Generating evaluation plots for %d samples.",
            n_samples,
        )

        actual_vs_predicted_fig = self._plotter.plot_actual_vs_predicted(y_true, y_pred)
        residuals_fig = self._plotter.plot_residuals(y_true, y_pred)
        prediction_error_fig = self._plotter.plot_prediction_error(y_true, y_pred)

        feature_importance_fig = None
        if feature_importance is not None:
            feature_importance_fig = self._plotter.plot_feature_importance(
                feature_importance
            )
            logger.info(
                "[EvaluationReporter] Feature importance plot generated "
                "(%d features).",
                len(feature_importance),
            )

        report = EvaluationReport(
            metrics=metrics,
            actual_vs_predicted_fig=actual_vs_predicted_fig,
            residuals_fig=residuals_fig,
            prediction_error_fig=prediction_error_fig,
            feature_importance_fig=feature_importance_fig,
        )

        logger.info("[EvaluationReporter] Evaluation complete — %s", report)
        return report
