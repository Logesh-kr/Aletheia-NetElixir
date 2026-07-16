"""
aletheia.ingestion.meta_ads
============================
Normaliser for Meta Ads (Facebook/Instagram) CSV exports.

Source field mapping
--------------------
+----------------+------------------+-----------------------------------------------+
| Source column  | Canonical column | Transformation                                |
+================+==================+===============================================+
| date_start     | date             | Parse to datetime64[ns]                       |
| campaign_name  | campaign_name    | Strip whitespace                              |
| (derived)      | campaign_id      | MD5(meta_ads::campaign_name)[:16]             |
| impressions    | impressions      | Cast to int64                                 |
| clicks         | clicks           | Cast to int64                                 |
| spend          | spend            | Cast to float64                               |
| conversion     | conversions      | Cast to float64                               |
| (none)         | revenue          | Always NaN — no native revenue in Meta CSV   |
+----------------+------------------+-----------------------------------------------+

Notes
-----
Revenue handling
    Meta's standard Insights API does not expose a ``purchase_roas`` or
    ``conversion_values`` field at the campaign level in a CSV export.
    Revenue must be sourced from:

    1. The Meta Conversions API (server-side events with ``value`` attached)
    2. A cross-platform AOV proxy (Tier 2 imputation in the Aletheia strategy)

    In this ingestion module, ``revenue`` is always set to ``NaN`` to signal
    that downstream imputation logic must handle it.  This is a deliberate
    design choice: storing 0 would corrupt ROAS distributions and model
    training sets.

Dropped columns
    ``ctr``, ``cpc``, ``cpm`` are platform-computed ratios derived from
    ``clicks``, ``impressions``, and ``spend``.  They are *not* ingested
    because Aletheia derives them from canonical primitives to ensure
    cross-platform consistency.  Ingesting platform-specific computed values
    would introduce subtle discrepancies (e.g. Meta rounds differently than
    the Aletheia formula).

Date column
    The source uses ``date_start`` rather than ``date_stop``.  If the report
    is daily-grain, ``date_start == date_stop``.  If a multi-day report is
    accidentally loaded (i.e. ``date_start != date_stop``), the connector
    uses ``date_start`` and logs a warning so the issue is visible.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseAdConnector

logger = logging.getLogger(__name__)


class MetaAdsConnector(BaseAdConnector):
    """
    Normalise a Meta Ads (Facebook/Instagram) Insights CSV export into the
    canonical schema.

    Expected source columns
    -----------------------
    ``date_start``, ``campaign_name``, ``impressions``, ``clicks``,
    ``spend``, ``conversion``

    Optional source columns
    -----------------------
    ``date_stop`` — used only to validate daily grain; not ingested.

    Examples
    --------
    >>> connector = MetaAdsConnector()
    >>> df = connector.load("data/meta_ads.csv")
    >>> df["revenue"].isna().all()
    True
    """

    PLATFORM_NAME: str = "meta_ads"

    @property
    def required_source_columns(self) -> list[str]:
        return [
            "date_start",
            "campaign_name",
            "impressions",
            "clicks",
            "spend",
            "conversion",
        ]

    def _map_to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remap Meta Ads source fields to the canonical schema.

        Revenue is explicitly set to ``NaN`` for all rows — this is
        intentional and expected.  See module docstring for rationale.
        """
        out = pd.DataFrame()

        # --- Date: validate daily grain if date_stop is present ---
        if "date_stop" in df.columns:
            mismatched = (df["date_start"] != df["date_stop"]).sum()
            if mismatched > 0:
                logger.warning(
                    "[%s] %d rows have date_start != date_stop — "
                    "report may not be daily grain.  Using date_start.",
                    self.PLATFORM_NAME,
                    mismatched,
                )
        out["date"] = df["date_start"]

        # --- Identity ---
        out["campaign_name"] = df["campaign_name"]
        out["campaign_id"] = out["campaign_name"].apply(
            lambda name: self._make_campaign_id(self.PLATFORM_NAME, name)
        )

        # --- Reach metrics ---
        out["impressions"] = df["impressions"]
        out["clicks"] = df["clicks"]

        # --- Spend ---
        out["spend"] = df["spend"]

        # --- Conversion count ---
        out["conversions"] = df["conversion"]

        # --- Revenue: intentionally absent from Meta CSV exports ---
        # NaN signals "no native revenue signal" to the downstream
        # imputation pipeline.  Do NOT substitute 0 here.
        out["revenue"] = float("nan")

        logger.debug(
            "[%s] Revenue set to NaN for all %d rows — "
            "downstream imputation required.",
            self.PLATFORM_NAME,
            len(out),
        )

        return out
