"""
aletheia.features.marketing_features
====================================

Marketing KPI feature engineering.

Derives standard digital marketing metrics from the canonical schema.

Generated Features
------------------
- ctr
- cpc
- cpm
- conversion_rate
- roas
- revenue_per_click
- revenue_per_impression
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseFeatureTransformer


class MarketingFeatureTransformer(BaseFeatureTransformer):
    """
    Generate marketing KPIs from canonical campaign metrics.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add marketing performance features.

        Parameters
        ----------
        df : pd.DataFrame
            Canonical dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe with marketing KPI features.
        """
        df = df.copy()

        required = [
            "spend",
            "revenue",
            "clicks",
            "impressions",
            "conversions",
        ]

        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"MarketingFeatureTransformer missing required columns: {missing}"
            )

        # Avoid divide-by-zero
        clicks = df["clicks"].replace(0, np.nan)
        impressions = df["impressions"].replace(0, np.nan)
        spend = df["spend"].replace(0, np.nan)

        # Click Through Rate
        df["ctr"] = df["clicks"] / impressions

        # Cost Per Click
        df["cpc"] = spend / clicks

        # Cost Per Mille (1000 impressions)
        df["cpm"] = (spend / impressions) * 1000

        # Conversion Rate
        df["conversion_rate"] = df["conversions"] / clicks

        # Return On Ad Spend
        df["roas"] = df["revenue"] / spend

        # Revenue Per Click
        df["revenue_per_click"] = df["revenue"] / clicks

        # Revenue Per Impression
        df["revenue_per_impression"] = df["revenue"] / impressions

        return df