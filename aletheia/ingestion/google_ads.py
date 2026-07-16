"""
aletheia.ingestion.google_ads
==============================
Normaliser for Google Ads CSV exports.

Source field mapping
--------------------
+-------------------------------+-------------------+----------------------------------------------+
| Source column                 | Canonical column  | Transformation                               |
+===============================+===================+==============================================+
| segments_date                 | date              | Parse to datetime64[ns]                      |
| campaign_name                 | campaign_name     | Strip whitespace                             |
| (derived)                     | campaign_id       | MD5(google_ads::campaign_name)[:16]          |
| metrics_impressions           | impressions       | Cast to int64                                |
| metrics_clicks                | clicks            | Cast to int64                                |
| metrics_cost_micros           | spend             | Divide by 1_000_000 → float64               |
| metrics_conversions           | conversions       | Cast to float64                              |
| metrics_conversions_value     | revenue           | Cast to float64; 0 → NaN (no revenue signal) |
+-------------------------------+-------------------+----------------------------------------------+

Notes
-----
- Google Ads stores monetary values as integer micros (1 unit = 1/1,000,000 of
  the account currency).  All cost and value fields must be divided by
  ``MICROS_DIVISOR = 1_000_000`` to obtain human-readable currency amounts.

- ``metrics_conversions_value = 0`` means the conversion tracking pixel did not
  record a value, *not* that revenue was genuinely zero.  We convert these to
  ``NaN`` so they are treated as missing data rather than zero-revenue events,
  which would corrupt ROAS calculations.

- ``ctr``, ``cpc``, ``cpm`` are intentionally not ingested.  They are
  platform-computed ratios that will be derived from canonical primitives to
  ensure cross-platform consistency.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseAdConnector

logger = logging.getLogger(__name__)

#: Divisor to convert Google Ads micro-units to standard currency.
MICROS_DIVISOR: int = 1_000_000


class GoogleAdsConnector(BaseAdConnector):
    """
    Normalise a Google Ads daily performance CSV export into the canonical schema.

    Expected source columns
    -----------------------
    ``segments_date``, ``campaign_name``, ``metrics_impressions``,
    ``metrics_clicks``, ``metrics_cost_micros``, ``metrics_conversions``,
    ``metrics_conversions_value``

    Examples
    --------
    >>> connector = GoogleAdsConnector()
    >>> df = connector.load("data/google_ads.csv")
    >>> df.dtypes["spend"]
    dtype('float64')
    """

    PLATFORM_NAME: str = "google_ads"

    @property
    def required_source_columns(self) -> list[str]:
        return [
            "segments_date",
            "campaign_name",
            "metrics_impressions",
            "metrics_clicks",
            "metrics_cost_micros",
            "metrics_conversions",
            "metrics_conversions_value",
        ]

    def _map_to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remap Google Ads source fields to the canonical schema.

        All values arrive as ``str`` from the base class CSV loader.
        Numeric coercion is handled downstream by
        :meth:`BaseAdConnector._apply_shared_transforms`.
        """
        out = pd.DataFrame()

        # --- Identity ---
        out["date"] = df["segments_date"]
        out["campaign_name"] = df["campaign_name"]
        out["campaign_id"] = out["campaign_name"].apply(
            lambda name: self._make_campaign_id(self.PLATFORM_NAME, name)
        )

        # --- Reach metrics ---
        out["impressions"] = df["metrics_impressions"]
        out["clicks"] = df["metrics_clicks"]

        # --- Spend: micros → currency ---
        # Convert string → numeric first, then divide.  errors="coerce" turns
        # unparseable strings into NaN rather than raising.
        out["spend"] = (
            pd.to_numeric(df["metrics_cost_micros"], errors="coerce")
            / MICROS_DIVISOR
        )
        # --- Conversion metrics ---
        out["conversions"] = df["metrics_conversions"]

        # --- Revenue: keep zero values, convert invalid negatives to NaN ---
        # Zero revenue can be a legitimate business outcome and should be preserved.
        # Negative revenue values are treated as invalid and converted to NaN.
        revenue_raw = pd.to_numeric(df["metrics_conversions_value"], errors="coerce")

        out["revenue"] = revenue_raw.where(
            revenue_raw >= 0,
            other=float("nan")
        )

        negative_revenue_count = (revenue_raw < 0).sum()
        if negative_revenue_count > 0:
            logger.debug(
                "[%s] Converted %d negative revenue rows to NaN.",
                self.PLATFORM_NAME,
                negative_revenue_count,
            )

        return out
