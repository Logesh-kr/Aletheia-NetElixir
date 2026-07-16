"""
aletheia.models.base
====================
Abstract base class defining the common interface for all Aletheia ML models.

Design notes
------------
- All concrete model classes (LightGBM, CatBoost, future models) must inherit
  from :class:`BaseModel` and implement every abstract method.
- The interface is intentionally minimal and stable: fit, predict, save, load.
  Framework-specific hyperparameter logic belongs in each concrete subclass.
- :meth:`save` and :meth:`load` accept a :class:`pathlib.Path` so callers are
  not forced to deal with raw string paths; str inputs are also accepted and
  are coerced internally by subclasses.
- :meth:`predict` returns a plain NumPy array to remain framework-agnostic.
  Downstream consumers (trainers, evaluators) always receive a consistent type.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all Aletheia forecasting models.

    Every concrete model (LightGBM, CatBoost, future architectures) must
    inherit from this class and implement all abstract methods.  The contract
    enforced here ensures that the :mod:`aletheia.models.trainer` and
    :mod:`aletheia.models.predictor` layers can interact with any model
    through a single, stable interface without branching on model type.

    Subclasses are responsible for:
        - Storing and exposing their trained artefact via instance state.
        - Persisting the artefact to disk in :meth:`save` and restoring it
          in :meth:`load`.
        - Accepting arbitrary keyword arguments in :meth:`fit` so that
          framework-specific options (e.g. ``eval_set``, ``callbacks``) can
          be passed without changing the shared interface.

    Attributes
    ----------
    name : str
        Human-readable identifier for this model type.  Set by each concrete
        subclass as a class-level attribute (e.g. ``"lightgbm"``).
    _is_fitted : bool
        Internal flag set to ``True`` after a successful :meth:`fit` call.
        Subclasses should respect this flag and may raise if :meth:`predict`
        is called before fitting.
    """

    #: Short identifier used in logging and artefact file names.
    #: Must be overridden by every concrete subclass.
    name: str = ""

    def __init__(self) -> None:
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> "BaseModel":
        """
        Train the model on the provided feature matrix and target vector.

        Parameters
        ----------
        X_train:
            Feature matrix for the training split.  Shape ``(n_samples,
            n_features)``.
        y_train:
            Target vector for the training split.  Shape ``(n_samples,)``.
        X_val:
            Optional feature matrix for an in-training validation split.
            Frameworks that support early stopping (e.g. LightGBM, CatBoost)
            will use this to determine the optimal number of rounds.
        y_val:
            Optional target vector corresponding to ``X_val``.
        **kwargs:
            Framework-specific keyword arguments forwarded directly to the
            underlying library's training routine (e.g. ``callbacks``,
            ``verbose_eval``).

        Returns
        -------
        BaseModel
            The fitted model instance (``self``), enabling method chaining.

        Raises
        ------
        ValueError
            If the training data is empty or feature dimensions are
            inconsistent.
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for the given feature matrix.

        Parameters
        ----------
        X:
            Feature matrix.  Must have the same column set as the training
            data supplied to :meth:`fit`.  Shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            One-dimensional array of predicted values.  Shape ``(n_samples,)``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit` has been successfully executed.
        ValueError
            If ``X`` is empty or has a mismatched feature set.
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Persist the trained model artefact to disk.

        The serialisation format is left to each concrete subclass
        (e.g. ``joblib`` for scikit-learn-compatible models, the native
        LightGBM ``model.save_model`` API, etc.).  The caller is responsible
        for ensuring the parent directory exists before calling this method.

        Parameters
        ----------
        path:
            Destination file path.  The file extension convention is
            subclass-specific (e.g. ``.txt`` for LightGBM, ``.cbm`` for
            CatBoost).

        Raises
        ------
        RuntimeError
            If called before :meth:`fit` has been successfully executed.
        OSError
            If the destination path is not writable.
        """
        ...

    @abstractmethod
    def load(self, path: str | Path) -> "BaseModel":
        """
        Restore a previously saved model artefact from disk.

        After a successful load the instance must be in the same state as if
        :meth:`fit` had been called; i.e. :attr:`_is_fitted` must be ``True``
        and :meth:`predict` must work without further configuration.

        Parameters
        ----------
        path:
            Path to the serialised model artefact written by :meth:`save`.

        Returns
        -------
        BaseModel
            The loaded model instance (``self``), enabling method chaining.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist on disk.
        ValueError
            If the file at ``path`` cannot be deserialised as a valid model
            artefact of this type.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """
        Return ``True`` if the model has been trained via :meth:`fit` or
        restored via :meth:`load`.
        """
        return self._is_fitted

    def _assert_is_fitted(self) -> None:
        """
        Raise :class:`RuntimeError` if the model has not yet been fitted.

        Concrete subclasses should call this at the top of :meth:`predict`
        and :meth:`save` to surface a clear error message rather than an
        obscure attribute error from the underlying framework.

        Raises
        ------
        RuntimeError
            If :attr:`_is_fitted` is ``False``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"[{self.name}] Model has not been fitted. "
                "Call fit() or load() before predict() or save()."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"{self.__class__.__name__}(name={self.name!r}, status={status!r})"
