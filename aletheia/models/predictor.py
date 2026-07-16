"""
aletheia.models.predictor
=========================
Inference-only layer for the Aletheia ML pipeline.

Responsibilities
----------------
- Loading a previously trained model artefact via the ``BaseModel`` interface.
- Validating the input feature DataFrame before inference.
- Delegating prediction to the model and returning a structured result.

Design notes
------------
- The predictor is intentionally separated from training concerns.  It holds
  no knowledge of how a model was trained, what metrics it achieved, or how
  features were engineered.  Those responsibilities belong to
  :mod:`aletheia.models.trainer` and the feature pipeline, respectively.
- The class is model-agnostic: it depends solely on the
  :class:`~aletheia.models.base.BaseModel` abstract interface and therefore
  works identically with LightGBM, CatBoost, or any future model subclass.
- The input DataFrame is never mutated.  All output is a fresh DataFrame
  built from the prediction array.
- Metadata columns (``campaign_id``, ``campaign_name``, ``platform``,
  ``date``) are propagated to the output when present so that downstream
  consumers can join predictions back to business context without re-fetching
  the source data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from aletheia.models.base import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columns that, when present in the input, are carried through to the output.
# ---------------------------------------------------------------------------

#: Canonical metadata columns preserved in the prediction output DataFrame
#: when they are present in the input ``X``.  Order is significant: they will
#: appear before ``prediction`` in the returned DataFrame.
_METADATA_COLUMNS: list[str] = [
    "date",
    "platform",
    "campaign_id",
    "campaign_name",
]


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class ModelPredictor:
    """
    Inference-only wrapper around a trained :class:`~aletheia.models.base.BaseModel`.

    The predictor handles a single concern: given an already-trained (or
    loaded) model and a feature DataFrame, produce a prediction DataFrame.
    It does not train, evaluate, tune, or persist models.

    Typical usage
    -------------
    .. code-block:: python

        from aletheia.models.lightgbm_model import LightGBMModel
        from aletheia.models.predictor import ModelPredictor

        model = LightGBMModel()
        predictor = ModelPredictor(model)
        predictor.load_model("artefacts/lgbm_revenue.txt")

        predictions_df = predictor.predict(feature_df)

    Parameters
    ----------
    model : BaseModel
        A concrete model instance.  The model does not need to be fitted at
        construction time; it may be loaded afterwards via :meth:`load_model`.

    Raises
    ------
    TypeError
        If ``model`` is not an instance of :class:`~aletheia.models.base.BaseModel`.
    """

    def __init__(self, model: BaseModel) -> None:
        if not isinstance(model, BaseModel):
            raise TypeError(
                f"model must be an instance of BaseModel, "
                f"got {type(model).__name__!r}."
            )
        self._model: BaseModel = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, model_path: str | Path) -> None:
        """
        Load a serialised model artefact from disk into the predictor.

        Delegates entirely to :meth:`~aletheia.models.base.BaseModel.load`
        so that the predictor remains agnostic of the serialisation format.
        After this call the predictor is ready to serve predictions.

        Parameters
        ----------
        model_path : str | Path
            Path to the model artefact written by
            :meth:`~aletheia.models.base.BaseModel.save`.

        Raises
        ------
        FileNotFoundError
            If ``model_path`` does not exist on disk.
        ValueError
            If the file cannot be deserialised as a valid model artefact.
        """
        model_path = Path(model_path)
        logger.info(
            "[ModelPredictor] Loading model=%r from: %s",
            self._model.name,
            model_path,
        )
        self._model.load(model_path)
        logger.info(
            "[ModelPredictor] Model=%r loaded successfully.",
            self._model.name,
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate revenue predictions for the given feature matrix.

        The input DataFrame is never modified.  Any metadata columns
        (``date``, ``platform``, ``campaign_id``, ``campaign_name``) that
        are present in ``X`` are carried through to the output so that
        callers can join predictions back to their original business context.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.  Must contain all columns the model was trained
            on.  Shape ``(n_samples, n_features)``.  Metadata columns may
            also be present; they are ignored by the model but preserved in
            the output.

        Returns
        -------
        pd.DataFrame
            A **new** DataFrame with the following column layout:

            - Any of ``date``, ``platform``, ``campaign_id``,
              ``campaign_name`` that were present in ``X`` (in that order).
            - ``prediction`` — the model's output as ``float64``.

            The index is reset to a contiguous integer range.

        Raises
        ------
        RuntimeError
            If the model has not been fitted or loaded before calling this
            method.
        ValueError
            If ``X`` is empty.
        """
        self._validate_input(X)

        logger.info(
            "[ModelPredictor] Running inference with model=%r on %d rows.",
            self._model.name,
            len(X),
        )

        raw_predictions: np.ndarray = self._model.predict(X)

        result_df = self._build_output(X, raw_predictions)

        logger.info(
            "[ModelPredictor] Inference complete — %d predictions generated.",
            len(result_df),
        )
        return result_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Raise a clear error for the most common input problems.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to validate.

        Raises
        ------
        ValueError
            If ``X`` is empty.
        RuntimeError
            If the underlying model has not been fitted or loaded.
        """
        if X.empty:
            raise ValueError("[ModelPredictor] Input DataFrame X must not be empty.")

        if not self._model.is_fitted:
            raise RuntimeError(
                f"[ModelPredictor] Model={self._model.name!r} has not been fitted. "
                "Call load_model() or train the model before calling predict()."
            )

    def _build_output(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        """
        Assemble the output DataFrame from metadata columns and predictions.

        The input DataFrame ``X`` is **not** mutated.  Metadata columns that
        are absent from ``X`` are silently skipped rather than raising, so
        the predictor degrades gracefully for feature-only DataFrames.

        Parameters
        ----------
        X : pd.DataFrame
            Original input (used only to extract metadata columns).
        predictions : np.ndarray
            Raw prediction array returned by the model.  Shape ``(n_samples,)``.

        Returns
        -------
        pd.DataFrame
            Output DataFrame with metadata columns (where present) followed
            by ``prediction``.
        """
        present_metadata: list[str] = [
            col for col in _METADATA_COLUMNS if col in X.columns
        ]

        output_parts: dict[str, object] = {}

        for col in present_metadata:
            output_parts[col] = X[col].to_numpy()

        output_parts["prediction"] = predictions.astype(np.float64)

        return pd.DataFrame(output_parts).reset_index(drop=True)
