"""
aletheia.models.trainer
=======================
Orchestrates the end-to-end model training pipeline for Aletheia.

Responsibilities
----------------
- Input validation
- Train / validation splitting via scikit-learn
- Delegating fit and predict to the model interface (``BaseModel``)
- Computing evaluation metrics (RMSE, MAE, R²) via scikit-learn
- Optionally persisting the trained model artefact
- Returning a structured :class:`TrainingResult` dataclass

Design notes
------------
- The trainer is intentionally model-agnostic: it depends only on the
  ``BaseModel`` abstract interface, so it works identically with LightGBM,
  CatBoost, or any future model that inherits from ``BaseModel``.
- No ML logic lives here.  Hyperparameter tuning, cross-validation, and
  plotting are explicitly out of scope; each belongs in a dedicated layer.
- Metrics are computed exclusively via ``sklearn.metrics`` — no manual
  formula implementations.
- ``TrainingResult`` is a frozen dataclass so that callers receive an
  immutable, inspectable record of what happened during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from aletheia.models.base import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingResult:
    """
    Immutable record of a single training run produced by :class:`ModelTrainer`.

    Attributes
    ----------
    model_name : str
        The :attr:`~aletheia.models.base.BaseModel.name` of the trained model.
    train_rows : int
        Number of samples used for training.
    validation_rows : int
        Number of samples used for validation / evaluation.
    rmse : float
        Root Mean Squared Error on the validation split.
    mae : float
        Mean Absolute Error on the validation split.
    r2 : float
        Coefficient of determination (R²) on the validation split.
    """

    model_name: str
    train_rows: int
    validation_rows: int
    rmse: float
    mae: float
    r2: float

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"TrainingResult("
            f"model={self.model_name!r}, "
            f"train_rows={self.train_rows}, "
            f"val_rows={self.validation_rows}, "
            f"RMSE={self.rmse:.4f}, "
            f"MAE={self.mae:.4f}, "
            f"R²={self.r2:.4f})"
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Orchestrates model training across the Aletheia ML pipeline.

    The trainer is model-agnostic: it accepts any concrete subclass of
    :class:`~aletheia.models.base.BaseModel` and drives the full pipeline::

        Dataset → Train/Val Split → fit() → predict() → evaluate() → [save()]

    It does **not** implement any ML logic, perform hyperparameter tuning,
    run cross-validation, or produce plots.

    Parameters
    ----------
    model : BaseModel
        A concrete model instance (e.g. :class:`~aletheia.models.lightgbm_model.LightGBMModel`).
        The trainer will call :meth:`~aletheia.models.base.BaseModel.fit` and
        :meth:`~aletheia.models.base.BaseModel.predict` on this object.
    test_size : float
        Fraction of the dataset reserved for the validation split.  Must be
        in the open interval ``(0.0, 1.0)``.  Defaults to ``0.2`` (20 %).
    random_state : int
        Random seed forwarded to :func:`sklearn.model_selection.train_test_split`
        to ensure reproducible splits.  Defaults to ``42``.

    Raises
    ------
    TypeError
        If ``model`` is not an instance of :class:`~aletheia.models.base.BaseModel`.
    ValueError
        If ``test_size`` is not in the open interval ``(0.0, 1.0)``.
    """

    def __init__(
        self,
        model: BaseModel,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        if not isinstance(model, BaseModel):
            raise TypeError(
                f"model must be an instance of BaseModel, got {type(model).__name__!r}."
            )
        if not (0.0 < test_size < 1.0):
            raise ValueError(
                f"test_size must be in (0.0, 1.0), got {test_size!r}."
            )

        self._model: BaseModel = model
        self._test_size: float = test_size
        self._random_state: int = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_output_path: str | Path | None = None,
        fit_kwargs: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """
        Execute the full training pipeline and return evaluation results.

        Pipeline
        --------
        1. Validate ``X`` and ``y``.
        2. Split into train and validation sets using
           :func:`sklearn.model_selection.train_test_split`.
        3. Call :meth:`~aletheia.models.base.BaseModel.fit` on the training split.
        4. Generate predictions on the validation split.
        5. Compute RMSE, MAE, and R² via :mod:`sklearn.metrics`.
        6. Optionally save the trained model if ``model_output_path`` is given.
        7. Return a :class:`TrainingResult`.

        Parameters
        ----------
        X : pd.DataFrame
            Full feature matrix.  Shape ``(n_samples, n_features)``.
        y : pd.Series
            Full target vector.  Shape ``(n_samples,)``.  Must be aligned
            with ``X`` (same index).
        model_output_path : str | Path | None
            If provided, the trained model is persisted to this path by
            calling :meth:`~aletheia.models.base.BaseModel.save`.  The
            parent directory is created automatically if it does not exist.
            Pass ``None`` (default) to skip saving.
        fit_kwargs : dict[str, Any] | None
            Optional keyword arguments forwarded to
            :meth:`~aletheia.models.base.BaseModel.fit` (e.g. LightGBM
            callbacks).  Pass ``None`` to use no extra arguments.

        Returns
        -------
        TrainingResult
            Immutable record containing evaluation metrics and split sizes.

        Raises
        ------
        ValueError
            If ``X`` or ``y`` is empty, or if their lengths do not match.
        """
        self._validate_inputs(X, y)

        logger.info(
            "[ModelTrainer] Starting training pipeline for model=%r "
            "(test_size=%.2f, random_state=%d).",
            self._model.name,
            self._test_size,
            self._random_state,
        )

        # ------------------------------------------------------------------
        # Step 1: Split
        # ------------------------------------------------------------------
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self._test_size,
            random_state=self._random_state,
        )

        logger.info(
            "[ModelTrainer] Split complete — train: %d rows, validation: %d rows.",
            len(X_train),
            len(X_val),
        )

        # ------------------------------------------------------------------
        # Step 2: Fit
        # ------------------------------------------------------------------
        logger.info("[ModelTrainer] Fitting model=%r …", self._model.name)

        self._model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            **(fit_kwargs or {}),
        )

        logger.info("[ModelTrainer] Model fit complete.")

        # ------------------------------------------------------------------
        # Step 3: Predict on validation
        # ------------------------------------------------------------------
        y_pred: np.ndarray = self._model.predict(X_val)

        # ------------------------------------------------------------------
        # Step 4: Evaluate
        # ------------------------------------------------------------------
        result = self._evaluate(
            y_true=y_val.to_numpy(),
            y_pred=y_pred,
            train_rows=len(X_train),
            validation_rows=len(X_val),
        )

        logger.info(
            "[ModelTrainer] Evaluation — RMSE: %.4f | MAE: %.4f | R²: %.4f",
            result.rmse,
            result.mae,
            result.r2,
        )

        # ------------------------------------------------------------------
        # Step 5: Save (optional)
        # ------------------------------------------------------------------
        if model_output_path is not None:
            output_path = Path(model_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._model.save(output_path)
            logger.info("[ModelTrainer] Model saved to: %s", output_path)

        logger.info("[ModelTrainer] Training pipeline complete.")
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Raise informative errors for common input problems before any
        expensive computation begins.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to validate.
        y : pd.Series
            Target vector to validate.

        Raises
        ------
        ValueError
            If ``X`` is empty, ``y`` is empty, or their lengths do not match.
        """
        if X.empty:
            raise ValueError("[ModelTrainer] X must not be empty.")
        if y.empty:
            raise ValueError("[ModelTrainer] y must not be empty.")
        if len(X) != len(y):
            raise ValueError(
                f"[ModelTrainer] X and y lengths must match. "
                f"Got X={len(X)}, y={len(y)}."
            )

    def _evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        train_rows: int,
        validation_rows: int,
    ) -> TrainingResult:
        """
        Compute evaluation metrics and package them into a :class:`TrainingResult`.

        All metric computations are delegated to :mod:`sklearn.metrics`.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth target values from the validation split.
        y_pred : np.ndarray
            Model predictions corresponding to ``y_true``.
        train_rows : int
            Number of samples in the training split (recorded in the result).
        validation_rows : int
            Number of samples in the validation split (recorded in the result).

        Returns
        -------
        TrainingResult
            Populated result dataclass.
        """
        rmse: float = float(
            mean_squared_error(y_true, y_pred, squared=False)
            if _sklearn_supports_squared_false()
            else np.sqrt(mean_squared_error(y_true, y_pred))
        )
        mae: float = float(mean_absolute_error(y_true, y_pred))
        r2: float = float(r2_score(y_true, y_pred))

        return TrainingResult(
            model_name=self._model.name,
            train_rows=train_rows,
            validation_rows=validation_rows,
            rmse=rmse,
            mae=mae,
            r2=r2,
        )


# ---------------------------------------------------------------------------
# Internal compatibility helper
# ---------------------------------------------------------------------------

def _sklearn_supports_squared_false() -> bool:
    """
    Return ``True`` if the installed scikit-learn version supports the
    ``squared=False`` argument in :func:`~sklearn.metrics.mean_squared_error`.

    The ``squared`` parameter was added in scikit-learn 0.22.2 and removed
    in favour of :func:`~sklearn.metrics.root_mean_squared_error` in 1.4.
    This helper ensures RMSE is computed correctly across all supported
    scikit-learn versions without raising a ``TypeError``.
    """
    import inspect
    return "squared" in inspect.signature(mean_squared_error).parameters
