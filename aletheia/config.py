"""
aletheia.config
===============
Centralised runtime configuration for the Aletheia ML pipeline.

Responsibilities
----------------
- Defining default hyperparameters for LightGBM training.
- Defining default train/validation split parameters.
- Providing canonical path configuration for data directories and model artefacts.
- Exposing a single :class:`AletheiaConfig` dataclass as the authoritative
  configuration contract consumed by :class:`~aletheia.pipeline.AletheiaPipeline`.

Design notes
------------
- :class:`AletheiaConfig` is a plain mutable dataclass so callers can override
  specific fields at construction time without subclassing.
- All path fields are coerced to :class:`pathlib.Path` in ``__post_init__``.
  String inputs are accepted everywhere for ergonomic use from scripts.
- Derived paths (e.g. per-platform CSV paths, model output path) are exposed
  as read-only properties computed from the base directory fields.
- Hyperparameter defaults mirror those in
  :data:`~aletheia.models.lightgbm_model._DEFAULT_PARAMS` but can be
  selectively overridden.  The LightGBM model merges config params on top of
  its own defaults at fit-time.
- This module has zero dependencies on other Aletheia modules so it can be
  imported anywhere in the package without introducing circular imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

#: Default LightGBM hyperparameter overrides applied at pipeline level.
_DEFAULT_LGBM_PARAMS: dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AletheiaConfig:
    """
    Centralised runtime configuration for the Aletheia pipeline.

    All fields carry production-appropriate defaults.  Override only the
    fields relevant to your environment; unchanged fields retain their
    defaults.

    Parameters
    ----------
    data_dir : str | Path
        Root directory containing raw platform CSV exports.
        Defaults to ``./data``.
    models_dir : str | Path
        Directory where trained model artefacts are written.
        Defaults to ``./artefacts/models``.
    google_ads_filename : str
        Filename for the Google Ads CSV inside ``data_dir``.
        Defaults to ``"google_ads.csv"``.
    meta_ads_filename : str
        Filename for the Meta Ads CSV inside ``data_dir``.
        Defaults to ``"meta_ads.csv"``.
    bing_ads_filename : str
        Filename for the Microsoft/Bing Ads CSV inside ``data_dir``.
        Defaults to ``"bing_ads.csv"``.
    target_column : str
        Name of the target variable in the feature DataFrame.
        Defaults to ``"revenue"``.
    test_size : float
        Fraction of data reserved for the validation split.
        Must be in the open interval ``(0.0, 1.0)``.  Defaults to ``0.2``.
    random_state : int
        Random seed forwarded to the train/validation splitter and the
        LightGBM booster for reproducibility.  Defaults to ``42``.
    num_boost_round : int
        Maximum number of LightGBM boosting iterations.  Defaults to ``500``.
    early_stopping_rounds : int
        Rounds without improvement on the validation metric before early
        stopping triggers.  Defaults to ``50``.
    lgbm_params : dict[str, Any]
        LightGBM hyperparameter overrides merged on top of the model's
        built-in defaults at fit-time.
    drop_feature_columns : list[str]
        Columns present in the feature DataFrame that must be excluded
        before the matrix is passed to the model.  Typically non-numeric
        metadata and the target column itself.

    Attributes
    ----------
    google_ads_path : Path
        Resolved path ``data_dir / google_ads_filename``.
    meta_ads_path : Path
        Resolved path ``data_dir / meta_ads_filename``.
    bing_ads_path : Path
        Resolved path ``data_dir / bing_ads_filename``.
    model_output_path : Path
        Resolved path ``models_dir / "lgbm_revenue.txt"``.

    Raises
    ------
    ValueError
        If ``test_size`` is not in the open interval ``(0.0, 1.0)``.
    """

    # ------------------------------------------------------------------
    # Directory configuration
    # ------------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("artefacts") / "models")

    # ------------------------------------------------------------------
    # Platform CSV filenames
    # ------------------------------------------------------------------
    google_ads_filename: str = "google_ads.csv"
    meta_ads_filename: str = "meta_ads.csv"
    bing_ads_filename: str = "bing_ads.csv"

    # ------------------------------------------------------------------
    # Modelling
    # ------------------------------------------------------------------
    target_column: str = "revenue"
    test_size: float = 0.2
    random_state: int = 42
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    lgbm_params: dict[str, Any] = field(
        default_factory=lambda: dict(_DEFAULT_LGBM_PARAMS)
    )

    # ------------------------------------------------------------------
    # Feature handling
    # ------------------------------------------------------------------
    #: Non-predictive columns removed from the feature matrix before
    #: model training and inference.  String calendar features produced
    #: by the feature pipeline are also excluded here to ensure the
    #: matrix contains only numeric inputs.
    drop_feature_columns: list[str] = field(
        default_factory=lambda: [
            "date",
            "platform",
            "campaign_id",
            "campaign_name",
            "revenue",       # target — must never be a feature
            "day_name",      # string; non-numeric calendar feature
            "month_name",    # string; non-numeric calendar feature
        ]
    )

    # ------------------------------------------------------------------
    # Post-init coercions and validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.models_dir = Path(self.models_dir)

        if not (0.0 < self.test_size < 1.0):
            raise ValueError(
                f"[AletheiaConfig] test_size must be in (0.0, 1.0), "
                f"got {self.test_size!r}."
            )

        logger.debug(
            "[AletheiaConfig] Initialised — data_dir=%s | models_dir=%s | "
            "target=%s | test_size=%.2f | num_boost_round=%d.",
            self.data_dir,
            self.models_dir,
            self.target_column,
            self.test_size,
            self.num_boost_round,
        )

    # ------------------------------------------------------------------
    # Derived path properties
    # ------------------------------------------------------------------

    @property
    def google_ads_path(self) -> Path:
        """Resolved path to the Google Ads CSV export."""
        return self.data_dir / self.google_ads_filename

    @property
    def meta_ads_path(self) -> Path:
        """Resolved path to the Meta Ads CSV export."""
        return self.data_dir / self.meta_ads_filename

    @property
    def bing_ads_path(self) -> Path:
        """Resolved path to the Microsoft/Bing Ads CSV export."""
        return self.data_dir / self.bing_ads_filename

    @property
    def model_output_path(self) -> Path:
        """Resolved path for saving the trained LightGBM model artefact."""
        return self.models_dir / "lgbm_revenue.txt"

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AletheiaConfig("
            f"data_dir={str(self.data_dir)!r}, "
            f"target_column={self.target_column!r}, "
            f"test_size={self.test_size}, "
            f"num_boost_round={self.num_boost_round})"
        )
