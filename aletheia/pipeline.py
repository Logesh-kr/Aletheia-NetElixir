"""
aletheia.pipeline
=================
End-to-end orchestration pipeline for the Aletheia AI-powered marketing copilot.

Responsibilities
----------------
- Accepting platform CSV paths and a runtime :class:`~aletheia.config.AletheiaConfig`.
- Orchestrating the complete pipeline:

      Ingestion → Feature Engineering → Model Training → Evaluation → [Save]

- Returning an immutable :class:`PipelineResult` containing the trained model,
  training metrics, and a full :class:`~aletheia.evaluation.reporter.EvaluationReport`.
- Optionally persisting the trained model artefact to the configured path.

Design notes
------------
- This module is the single programmatic entry point for running Aletheia
  end-to-end.  It imports from every other layer but owns no ML or data logic.
- All layer-specific concerns (schema validation, feature transforms,
  hyperparameter handling) remain strictly in their respective modules.
- Rows where the target column is ``NaN`` are dropped before model training.
  This is expected for Meta Ads rows which carry no native revenue signal.
- The feature matrix passed to the model excludes non-numeric columns and all
  columns listed in :attr:`~aletheia.config.AletheiaConfig.drop_feature_columns`.
  No additional manual column exclusion is needed at the call site.
- A CLI wrapper that parses command-line arguments and invokes
  :meth:`AletheiaPipeline.run` belongs in a dedicated entry-point script and
  is explicitly out of scope for this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from aletheia.config import AletheiaConfig
from aletheia.evaluation.reporter import EvaluationReport, EvaluationReporter
from aletheia.features.pipeline import FeaturePipeline
from aletheia.ingestion import IngestionPipeline
from aletheia.models.lightgbm_model import LightGBMModel
from aletheia.models.trainer import ModelTrainer, TrainingResult

if TYPE_CHECKING:
    from aletheia.models.base import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineResult:
    """
    Immutable record of a complete Aletheia pipeline run.

    Attributes
    ----------
    model : BaseModel
        The fitted model instance produced by the training stage.
    training_result : TrainingResult
        Metrics and split sizes from :class:`~aletheia.models.trainer.ModelTrainer`,
        computed on the held-out validation split.
    evaluation_report : EvaluationReport
        Full evaluation report — metrics and Matplotlib figures — produced by
        :class:`~aletheia.evaluation.reporter.EvaluationReporter` on the
        complete non-null target dataset.
    unified_df : pd.DataFrame
        Unified canonical DataFrame output by the ingestion stage.
    feature_df : pd.DataFrame
        Feature-engineered DataFrame passed to the model training stage.
    """

    model: "BaseModel"
    training_result: TrainingResult
    evaluation_report: EvaluationReport
    unified_df: pd.DataFrame
    feature_df: pd.DataFrame

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"PipelineResult("
            f"model={self.model.name!r}, "
            f"train_rows={self.training_result.train_rows}, "
            f"val_rows={self.training_result.validation_rows}, "
            f"RMSE={self.training_result.rmse:.4f}, "
            f"R²={self.training_result.r2:.4f})"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AletheiaPipeline:
    """
    End-to-end pipeline for the Aletheia AI-powered marketing copilot.

    Orchestrates the full flow::

        Ingestion → Feature Engineering → Training → Evaluation → [Save]

    Each stage is fully delegated to its own module.  This class coordinates
    the sequence and passes data between layers without owning any logic.

    Parameters
    ----------
    config : AletheiaConfig | None
        Runtime configuration.  Pass ``None`` to use the default
        :class:`~aletheia.config.AletheiaConfig` settings.

    Example
    -------
    .. code-block:: python

        from aletheia.pipeline import AletheiaPipeline
        from aletheia.config import AletheiaConfig

        config = AletheiaConfig(data_dir="data/", test_size=0.2)
        result = AletheiaPipeline(config).run(
            google_ads_path="data/google.csv",
            meta_ads_path="data/meta.csv",
            bing_ads_path="data/bing.csv",
            save_model=True,
        )
        print(result)
    """

    def __init__(self, config: AletheiaConfig | None = None) -> None:
        self._config: AletheiaConfig = config or AletheiaConfig()
        self._ingestion_pipeline = IngestionPipeline()
        self._feature_pipeline = FeaturePipeline()
        self._reporter = EvaluationReporter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        google_ads_path: str | Path | None = None,
        meta_ads_path: str | Path | None = None,
        bing_ads_path: str | Path | None = None,
        save_model: bool = False,
    ) -> PipelineResult:
        """
        Execute the full end-to-end Aletheia pipeline.

        Pipeline stages
        ---------------
        1. **Ingestion** — load and validate platform CSVs into a unified
           canonical DataFrame via :class:`~aletheia.ingestion.IngestionPipeline`.
        2. **Feature engineering** — transform the canonical DataFrame into a
           model-ready feature matrix via
           :class:`~aletheia.features.pipeline.FeaturePipeline`.
        3. **Preparation** — drop non-numeric and metadata columns; remove rows
           where the target is ``NaN`` (e.g. all Meta Ads rows); separate ``X``
           from ``y``.
        4. **Training** — fit a :class:`~aletheia.models.lightgbm_model.LightGBMModel`
           via :class:`~aletheia.models.trainer.ModelTrainer`, which handles
           the train/validation split internally.
        5. **Evaluation** — generate metrics and plots for the full non-null
           dataset via :class:`~aletheia.evaluation.reporter.EvaluationReporter`.
        6. **Persistence** (optional) — save the trained model artefact when
           ``save_model=True``.

        Parameters
        ----------
        google_ads_path : str | Path | None
            Path to the Google Ads CSV export.  ``None`` to skip this platform.
        meta_ads_path : str | Path | None
            Path to the Meta Ads CSV export.  ``None`` to skip this platform.
        bing_ads_path : str | Path | None
            Path to the Microsoft/Bing Ads CSV export.  ``None`` to skip.
        save_model : bool
            When ``True``, the trained model is persisted to
            :attr:`~aletheia.config.AletheiaConfig.model_output_path`.
            Defaults to ``False``.

        Returns
        -------
        PipelineResult
            Immutable record containing the trained model, training metrics,
            full evaluation report, and intermediate DataFrames.

        Raises
        ------
        ValueError
            If no platform paths are provided, if the target column is absent
            from the feature DataFrame, or if no rows remain after dropping
            null-target rows.
        FileNotFoundError
            If a provided platform CSV path does not exist on disk.
        """
        logger.info(
            "[AletheiaPipeline] Starting pipeline run. config=%r",
            self._config,
        )

        # ------------------------------------------------------------------
        # Stage 1: Ingestion
        # ------------------------------------------------------------------
        logger.info("[AletheiaPipeline] Stage 1 — Ingestion.")
        unified_df = self._ingestion_pipeline.run(
            google_ads_path=google_ads_path,
            meta_ads_path=meta_ads_path,
            bing_ads_path=bing_ads_path,
        )
        logger.info(
            "[AletheiaPipeline] Ingestion complete — %d rows, %d campaigns.",
            len(unified_df),
            unified_df["campaign_id"].nunique(),
        )

        # ------------------------------------------------------------------
        # Stage 2: Feature engineering
        # ------------------------------------------------------------------
        logger.info("[AletheiaPipeline] Stage 2 — Feature engineering.")
        feature_df = self._feature_pipeline.transform(unified_df)
        logger.info(
            "[AletheiaPipeline] Feature engineering complete — "
            "%d rows, %d columns.",
            len(feature_df),
            len(feature_df.columns),
        )

        # ------------------------------------------------------------------
        # Stage 3: Prepare model inputs
        # ------------------------------------------------------------------
        logger.info("[AletheiaPipeline] Stage 3 — Preparing model inputs.")
        X, y = self._prepare_model_inputs(feature_df)
        logger.info(
            "[AletheiaPipeline] Model inputs ready — %d rows, %d features.",
            len(X),
            len(X.columns),
        )

        # ------------------------------------------------------------------
        # Stage 4: Training
        # ------------------------------------------------------------------
        logger.info("[AletheiaPipeline] Stage 4 — Training.")
        model = LightGBMModel(
            params=self._config.lgbm_params,
            num_boost_round=self._config.num_boost_round,
            early_stopping_rounds=self._config.early_stopping_rounds,
            seed=self._config.random_state,
        )
        trainer = ModelTrainer(
            model=model,
            test_size=self._config.test_size,
            random_state=self._config.random_state,
        )
        model_output_path: Path | None = (
            self._config.model_output_path if save_model else None
        )
        training_result = trainer.train(
            X=X,
            y=y,
            model_output_path=model_output_path,
        )
        logger.info(
            "[AletheiaPipeline] Training complete — "
            "RMSE=%.4f | MAE=%.4f | R²=%.4f.",
            training_result.rmse,
            training_result.mae,
            training_result.r2,
        )

        # ------------------------------------------------------------------
        # Stage 5: Evaluation
        # ------------------------------------------------------------------
        logger.info("[AletheiaPipeline] Stage 5 — Evaluation.")
        y_pred = model.predict(X)

        feature_importance: pd.DataFrame | None = None
        if hasattr(model, "feature_importance"):
            try:
                feature_importance = model.feature_importance
            except RuntimeError:
                pass

        evaluation_report = self._reporter.evaluate(
            y_true=y,
            y_pred=y_pred,
            feature_importance=feature_importance,
        )
        logger.info("[AletheiaPipeline] Evaluation complete.")

        logger.info(
            "[AletheiaPipeline] Pipeline run complete. %s",
            training_result,
        )

        return PipelineResult(
            model=model,
            training_result=training_result,
            evaluation_report=evaluation_report,
            unified_df=unified_df,
            feature_df=feature_df,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_model_inputs(
        self,
        feature_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separate the target vector from the feature matrix and drop
        non-predictive columns.

        Rows where the target column is ``NaN`` are dropped before splitting.
        This is required for platforms (e.g. Meta Ads) that produce no native
        revenue signal.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature-engineered DataFrame from :class:`FeaturePipeline`.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            ``(X, y)`` — numeric feature matrix and target vector aligned on
            the same index.

        Raises
        ------
        ValueError
            If the target column is absent from ``feature_df``, or if no rows
            remain after dropping null-target rows.
        """
        target = self._config.target_column

        if target not in feature_df.columns:
            raise ValueError(
                f"[AletheiaPipeline] Target column '{target}' not found in "
                f"feature DataFrame.  Available columns: "
                f"{sorted(feature_df.columns.tolist())}"
            )

        before = len(feature_df)
        feature_df = feature_df.dropna(subset=[target]).reset_index(drop=True)
        dropped = before - len(feature_df)

        if dropped > 0:
            logger.info(
                "[AletheiaPipeline] Dropped %d rows with null target '%s'. "
                "%d rows remaining.",
                dropped,
                target,
                len(feature_df),
            )

        if feature_df.empty:
            raise ValueError(
                f"[AletheiaPipeline] No rows remain after dropping null "
                f"values for target column '{target}'."
            )

        drop_cols = [
            col
            for col in self._config.drop_feature_columns
            if col in feature_df.columns
        ]
        X = feature_df.drop(columns=drop_cols)

        # Retain only numeric columns — string calendar features not removed
        # by drop_feature_columns are excluded here as a safety net.
        X = X.select_dtypes(include="number")

        y: pd.Series = feature_df[target].astype("float64")

        logger.debug(
            "[AletheiaPipeline] Feature matrix: %d rows × %d columns.",
            len(X),
            len(X.columns),
        )

        return X, y
