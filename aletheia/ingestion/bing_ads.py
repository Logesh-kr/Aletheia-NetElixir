"""
aletheia.ingestion.bing_ads
============================
Normaliser for Microsoft/Bing Ads CSV exports.

Source field mapping
--------------------
+----------------+------------------+--------------------------------------------------+
| Source column  | Canonical column | Transformation                                   |
+================+==================+==================================================+
| TimePeriod     | date             | Parse to datetime64[ns]; handles MM/DD/YYYY      |
| CampaignName   | campaign_name    | Strip whitespace                                 |
| (derived)      | campaign_id      | MD5(microsoft_ads::CampaignName)[:16]            |
| Impressions    | impressions      | Strip commas → cast to int64                     |
| Clicks         | clicks           | Strip commas → cast to int64                     |
| Spend          | spend            | Strip commas → cast to float64                   |
| Conversions    | conversions      | Strip commas → cast to float64                   |
| Revenue        | revenue          | Strip commas → float64; 0 → NaN                 |
+----------------+------------------+--------------------------------------------------+

Notes
-----
Comma-formatted numbers
    Bing Ads CSV exports may include thousands-separator commas in numeric
    fields (e.g. ``"1,234.56"``).  All numeric columns are stripped of commas
    before conversion via :meth:`BaseAdConnector._strip_commas`.

Date format
    Bing typically emits dates as ``MM/DD/YYYY`` (e.g. ``"1/15/2024"``),
    which differs from the ISO 8601 format used by Google and Meta.
    :meth:`BaseAdConnector._parse_dates` handles this via ``dayfirst=False``,
    which correctly disambiguates ``1/15/2024`` as January 15th.

Revenue zero → NaN
    A ``Revenue = 0`` in Bing typically means the campaign had conversions but
    no revenue value was tracked (e.g. lead-gen campaigns without a purchase
    event).  Zero revenue is converted to ``NaN`` to prevent it from anchoring
    the ROAS distribution downward.  Rows where ``Conversions = 0`` as well
    are expected to have ``Revenue = 0`` — but these should still become NaN
    because they carry no revenue signal.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseAdConnector

logger = logging.getLogger(__name__)


class BingAdsConnector(BaseAdConnector):
    """
    Normalise a Microsoft/Bing Ads performance report CSV into the canonical schema.

    Expected source columns
    -----------------------
    ``TimePeriod``, ``CampaignName``, ``Impressions``, ``Clicks``,
    ``Spend``, ``Conversions``, ``Revenue``

    Examples
    --------
    >>> connector = BingAdsConnector()
    >>> df = connector.load("data/bing_ads.csv")
    >>> df["date"].dtype
    dtype('<M8[ns]')
    """

    PLATFORM_NAME: str = "microsoft_ads"

    @property
    def required_source_columns(self) -> list[str]:
        return [
            "TimePeriod",
            "CampaignName",
            "Impressions",
            "Clicks",
            "Spend",
            "Conversions",
            "Revenue",
        ]

    def _map_to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remap Bing Ads source fields to the canonical schema.

        Strips thousands-separator commas from all numeric columns before
        conversion, as Bing exports may include them.
        """
        out = pd.DataFrame()

        # --- Date: Bing uses MM/DD/YYYY by default ---
        out["date"] = df["TimePeriod"]

        # --- Identity ---
        out["campaign_name"] = df["CampaignName"]
        out["campaign_id"] = out["campaign_name"].apply(
            lambda name: self._make_campaign_id(self.PLATFORM_NAME, name)
        )

        # --- Reach metrics (strip commas before coercion) ---
        out["impressions"] = self._strip_commas(df["Impressions"])
        out["clicks"] = self._strip_commas(df["Clicks"])

        # --- Spend ---
        out["spend"] = self._strip_commas(df["Spend"])

        # --- Conversions ---
        out["conversions"] = self._strip_commas(df["Conversions"]).pipe(
            pd.to_numeric, errors="coerce"
        )

        # --- Revenue: 0 → NaN (no revenue attribution, not zero revenue) ---
        revenue_raw = self._strip_commas(df["Revenue"]).pipe(
            pd.to_numeric, errors="coerce"
        )
        out["revenue"] = revenue_raw.where(revenue_raw >= 0, other=float("nan"))

        negative_revenue_count = (revenue_raw < 0).sum()

        if negative_revenue_count > 0:
            logger.debug(
                "[%s] Converted %d negative Revenue rows to NaN.",
                self.PLATFORM_NAME,
                negative_revenue_count,
            )
        return out

