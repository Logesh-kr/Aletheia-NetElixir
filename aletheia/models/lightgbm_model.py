"""
aletheia.models.lightgbm_model
==============================
LightGBM implementation of :class:`~aletheia.models.base.BaseModel`.

Design notes
------------
- The native ``lightgbm.train`` API is used instead of the scikit-learn
  wrapper.  This gives full access to LightGBM callbacks, custom objectives,
  and the ``Booster`` object without scikit-learn overhead.
- Hyperparameters are passed as a plain ``dict`` and merged with safe defaults
  at fit-time so that callers only need to specify overrides.
- Early stopping is handled via LightGBM's built-in callback rather than a
  manual loop; the optimal number of rounds is stored on the ``Booster`` and
  is automatically used at predict-time.
- Feature names are captured from the training ``pd.DataFrame`` columns so
  that :attr:`feature_importance` and column-order validation are always
  consistent with what the model was trained on.
- The model is persisted with ``Booster.save_model`` (LightGBM's native text
  format), which is version-stable and does not rely on ``pickle``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from aletheia.models.base import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

#: Safe defaults used when the caller does not supply a full parameter dict.
#: Any key supplied via ``params`` in :meth:`LightGBMModel.fit` overrides the
#: corresponding default.
_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
}

#: Default number of boosting rounds used when no validation set is provided
#: (i.e. early stopping cannot be applied).
_DEFAULT_NUM_BOOST_ROUND: int = 500

#: Default number of early-stopping rounds when a validation set is provided.
_DEFAULT_EARLY_STOPPING_ROUNDS: int = 50


# ---------------------------------------------------------------------------
# LightGBM model
# ---------------------------------------------------------------------------

class LightGBMModel(BaseModel):
    """
    Gradient-boosted tree model backed by LightGBM.

    Uses the native ``lightgbm.train`` API for maximum flexibility.  All
    hyperparameters are configurable at fit-time; sensible defaults are
    applied for any parameter that is not explicitly provided.

    Parameters
    ----------
    params : dict[str, Any] | None
        LightGBM hyperparameter overrides.  Merged with :data:`_DEFAULT_PARAMS`
        at fit-time.  Pass ``None`` (default) to use the built-in defaults
        without modification.
    num_boost_round : int
        Maximum number of boosting iterations.  When a validation set is
        supplied, early stopping may halt training before this limit.
    early_stopping_rounds : int
        Number of consecutive rounds without improvement on the validation
        metric that trigger early stopping.  Only active when ``X_val`` and
        ``y_val`` are passed to :meth:`fit`.
    seed : int
        Random seed for reproducibility.  Forwarded to LightGBM via the
        ``seed`` parameter key.

    Attributes
    ----------
    name : str
        Model identifier — always ``"lightgbm"``.
    params_ : dict[str, Any]
        Resolved hyperparameter dict used during the last :meth:`fit` call.
    feature_names_ : list[str]
        Ordered list of feature column names captured from the training
        ``DataFrame``.  Populated after :meth:`fit` or :meth:`load`.
    booster_ : lgb.Booster | None
        Trained LightGBM ``Booster`` object.  ``None`` until the model is
        fitted or loaded.
    """

    name: str = "lightgbm"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_boost_round: int = _DEFAULT_NUM_BOOST_ROUND,
        early_stopping_rounds: int = _DEFAULT_EARLY_STOPPING_ROUNDS,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._init_params: dict[str, Any] = params or {}
        self._num_boost_round: int = num_boost_round
        self._early_stopping_rounds: int = early_stopping_rounds
        self._seed: int = seed

        self.params_: dict[str, Any] = {}
        self.feature_names_: list[str] = []
        self.booster_: lgb.Booster | None = None

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> "LightGBMModel":
        """
        Train a LightGBM booster on the provided data.

        Parameters
        ----------
        X_train:
            Training feature matrix.  Shape ``(n_samples, n_features)``.
        y_train:
            Training target vector.  Shape ``(n_samples,)``.
        X_val:
            Optional validation feature matrix used for early stopping.
        y_val:
            Optional validation target vector.  Required when ``X_val`` is
            provided.
        **kwargs:
            Additional keyword arguments forwarded directly to
            ``lightgbm.train`` (e.g. ``feval``, ``init_model``).

        Returns
        -------
        LightGBMModel
            The fitted model instance (``self``).

        Raises
        ------
        ValueError
            If ``X_train`` is empty, or if ``X_val`` is provided without
            ``y_val`` (or vice versa).
        """
        if X_train.empty:
            raise ValueError(f"[{self.name}] X_train must not be empty.")

        if (X_val is None) != (y_val is None):
            raise ValueError(
                f"[{self.name}] X_val and y_val must both be provided or both omitted."
            )

        # Resolve final hyperparameters: defaults → caller overrides → seed
        self.params_ = {**_DEFAULT_PARAMS, **self._init_params, "seed": self._seed}

        # Capture feature names before constructing LightGBM datasets so they
        # are available even if predict is called with a renamed DataFrame.
        self.feature_names_ = list(X_train.columns)

        logger.info(
            "[%s] Building training dataset (%d rows, %d features).",
            self.name,
            len(X_train),
            len(self.feature_names_),
        )

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_names_,
            free_raw_data=True,
        )

        callbacks: list[Any] = []
        valid_sets: list[lgb.Dataset] = [train_data]
        valid_names: list[str] = ["train"]

        if X_val is not None and y_val is not None:
            logger.info(
                "[%s] Validation set provided (%d rows). Early stopping after %d rounds.",
                self.name,
                len(X_val),
                self._early_stopping_rounds,
            )
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
                feature_name=self.feature_names_,
                free_raw_data=True,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self._early_stopping_rounds,
                    verbose=False,
                )
            )

        callbacks.append(lgb.log_evaluation(period=50))

        logger.info("[%s] Starting training (max rounds: %d).", self.name, self._num_boost_round)

        self.booster_ = lgb.train(
            params=self.params_,
            train_set=train_data,
            num_boost_round=self._num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
            **kwargs,
        )

        self._is_fitted = True

        best_iteration = self.booster_.best_iteration
        logger.info(
            "[%s] Training complete. Best iteration: %s.",
            self.name,
            best_iteration if best_iteration > 0 else self._num_boost_round,
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for the given feature matrix.

        Parameters
        ----------
        X:
            Feature matrix.  Must contain all columns present in
            :attr:`feature_names_`.  Shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            One-dimensional array of predicted values.  Shape ``(n_samples,)``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If ``X`` is empty or is missing required feature columns.
        """
        self._assert_is_fitted()

        if X.empty:
            raise ValueError(f"[{self.name}] X must not be empty.")

        missing = sorted(set(self.feature_names_) - set(X.columns))
        if missing:
            raise ValueError(
                f"[{self.name}] Prediction data is missing features: {missing}."
            )

        predictions: np.ndarray = self.booster_.predict(  # type: ignore[union-attr]
            X[self.feature_names_],
            num_iteration=self.booster_.best_iteration,  # type: ignore[union-attr]
        )

        logger.debug("[%s] Predicted %d samples.", self.name, len(predictions))
        return predictions

    def save(self, path: str | Path) -> None:
        """
        Persist the trained booster to disk using LightGBM's native text format.

        Parameters
        ----------
        path:
            Destination file path.  The ``.txt`` extension is conventional for
            LightGBM model files but is not enforced.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        OSError
            If the destination directory is not writable.
        """
        self._assert_is_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.booster_.save_model(str(path))  # type: ignore[union-attr]
        logger.info("[%s] Model saved to: %s", self.name, path)

    def load(self, path: str | Path) -> "LightGBMModel":
        """
        Restore a previously saved LightGBM booster from disk.

        Parameters
        ----------
        path:
            Path to a model file written by :meth:`save`.

        Returns
        -------
        LightGBMModel
            The loaded model instance (``self``).

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the file cannot be loaded as a valid LightGBM model.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"[{self.name}] Model file not found: {path}"
            )

        try:
            self.booster_ = lgb.Booster(model_file=str(path))
        except Exception as exc:
            raise ValueError(
                f"[{self.name}] Failed to load model from {path}: {exc}"
            ) from exc

        self.feature_names_ = self.booster_.feature_name()
        self._is_fitted = True

        logger.info(
            "[%s] Model loaded from: %s (%d features).",
            self.name,
            path,
            len(self.feature_names_),
        )
        return self

    # ------------------------------------------------------------------
    # Additional properties
    # ------------------------------------------------------------------

    @property
    def feature_importance(self) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances sorted in descending order.

        Uses LightGBM's ``gain``-based importance metric, which measures the
        total reduction in loss attributed to each feature across all trees.
        This is generally more informative than split-count importance for
        regression tasks.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``["feature", "importance"]``, sorted by
            ``importance`` descending.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._assert_is_fitted()

        importances = self.booster_.feature_importance(importance_type="gain")  # type: ignore[union-attr]

        df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return df
