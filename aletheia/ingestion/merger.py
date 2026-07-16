"""
aletheia.ingestion.merger
==========================
Orchestrates the end-to-end ingestion pipeline:

    1. Load each platform CSV via its dedicated connector.
    2. Validate each normalised DataFrame individually.
    3. Concatenate into a single canonical DataFrame.
    4. Run a final cross-platform validation pass.
    5. Sort by ``(date, platform, campaign_id)`` and reset the index.

Usage
-----
::

    from aletheia.ingestion import IngestionPipeline

    pipeline = IngestionPipeline()

    df = pipeline.run(
        google_ads_path="data/google_ads.csv",
        meta_ads_path="data/meta_ads.csv",
        bing_ads_path="data/bing_ads.csv",
    )

    # df is a clean, unified pandas DataFrame with these columns:
    # date | platform | campaign_id | campaign_name | spend | revenue |
    # conversions | clicks | impressions

Design notes
------------
- Each platform is loaded independently; a failure in one platform does
  *not* silently suppress data from another.  Errors propagate immediately
  with a clear platform tag so the caller knows exactly which file failed.
- All three paths are optional.  At least one must be provided.
- Sorting is deterministic: ``(date ASC, platform ASC, campaign_id ASC)``.
  This ensures reproducible row order for snapshot testing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import CANONICAL_COLUMNS
from .bing_ads import BingAdsConnector
from .google_ads import GoogleAdsConnector
from .meta_ads import MetaAdsConnector
from .validator import CanonicalValidator

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrate loading, normalising, and merging ad platform CSV data into a
    single canonical :class:`pandas.DataFrame`.

    Parameters
    ----------
    None — all configuration is passed per-run to :meth:`run`.

    Examples
    --------
    Load all three platforms::

        pipeline = IngestionPipeline()
        df = pipeline.run(
            google_ads_path="data/google.csv",
            meta_ads_path="data/meta.csv",
            bing_ads_path="data/bing.csv",
        )

    Load only Google and Bing (Meta omitted)::

        df = pipeline.run(
            google_ads_path="data/google.csv",
            bing_ads_path="data/bing.csv",
        )
    """

    def __init__(self) -> None:
        self._connectors: dict[str, GoogleAdsConnector | MetaAdsConnector | BingAdsConnector] = {
            "google_ads": GoogleAdsConnector(),
            "meta_ads": MetaAdsConnector(),
            "microsoft_ads": BingAdsConnector(),
        }
        self._validator = CanonicalValidator()

    def run(
        self,
        google_ads_path: str | Path | None = None,
        meta_ads_path: str | Path | None = None,
        bing_ads_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Load platform CSVs, normalise, merge, and validate.

        Parameters
        ----------
        google_ads_path:
            Path to the Google Ads daily performance CSV.  ``None`` to skip.
        meta_ads_path:
            Path to the Meta Ads Insights CSV.  ``None`` to skip.
        bing_ads_path:
            Path to the Microsoft/Bing Ads performance report CSV.  ``None`` to skip.

        Returns
        -------
        pd.DataFrame
            Unified canonical DataFrame with columns:
            ``date, platform, campaign_id, campaign_name, spend, revenue,
            conversions, clicks, impressions``.
            Sorted by ``(date ASC, platform ASC, campaign_id ASC)``.
            Index is reset to ``RangeIndex``.

        Raises
        ------
        ValueError
            If *all* paths are ``None``.
        FileNotFoundError
            If a provided path does not exist on disk.
        ValueError
            If a platform CSV is missing required source columns.
        """
        path_map: dict[str, str | Path | None] = {
            "google_ads": google_ads_path,
            "meta_ads": meta_ads_path,
            "microsoft_ads": bing_ads_path,
        }

        if all(v is None for v in path_map.values()):
            raise ValueError(
                "IngestionPipeline.run() requires at least one platform path. "
                "All three (google_ads_path, meta_ads_path, bing_ads_path) were None."
            )

        frames: list[pd.DataFrame] = []

        for platform_key, path in path_map.items():
            if path is None:
                logger.info("[Pipeline] Skipping %s — no path provided.", platform_key)
                continue

            connector = self._connectors[platform_key]

            try:
                platform_df = connector.load(path)
                platform_df = self._validator.validate(platform_df)
            except (FileNotFoundError, ValueError) as exc:
                # Re-raise with additional context; caller must handle.
                logger.error(
                    "[Pipeline] Failed to load %s from '%s': %s",
                    platform_key,
                    path,
                    exc,
                )
                raise

            frames.append(platform_df)
            logger.info(
                "[Pipeline] %s: %d rows ingested.",
                platform_key,
                len(platform_df),
            )

        # Concatenate all platform frames
        unified_df = pd.concat(frames, ignore_index=True)
        logger.info(
            "[Pipeline] Concatenated %d rows from %d platform(s).",
            len(unified_df),
            len(frames),
        )

        # Cross-platform validation pass
        unified_df = self._validator.validate(unified_df)

        # Final sort + column order enforcement
        unified_df = self._finalise(unified_df)

        logger.info(
            "[Pipeline] Ingestion complete. "
            "rows=%d | platforms=%s | campaigns=%d | date_range=[%s, %s]",
            len(unified_df),
            sorted(unified_df["platform"].unique().tolist()),
            unified_df["campaign_id"].nunique(),
            unified_df["date"].min().date(),
            unified_df["date"].max().date(),
        )

        return unified_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finalise(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort the unified DataFrame and enforce canonical column order.

        Sorting by ``(date, platform, campaign_id)`` ensures deterministic
        row order regardless of the order platforms were loaded.
        """
        return (
            df
            .sort_values(["date", "platform", "campaign_id"], ascending=True)
            .reset_index(drop=True)
            [CANONICAL_COLUMNS]
        )
