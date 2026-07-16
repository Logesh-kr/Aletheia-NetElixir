"""
aletheia.features.lag_features
==============================

Lag-based feature engineering.

Creates historical lag features for forecasting models by shifting
metrics within each (platform, campaign_id) group.

Generated Features
------------------
- spend_lag_1
- spend_lag_7
- revenue_lag_1
- revenue_lag_7
- clicks_lag_1
- clicks_lag_7
- conversions_lag_1
- conversions_lag_7
"""

from __future__ import annotations

import pandas as pd

from .base import BaseFeatureTransformer


class LagFeatureTransformer(BaseFeatureTransformer):
    """
    Generate lag features for campaign-level forecasting.

    Lag features are created independently for every
    (platform, campaign_id) combination to prevent data leakage
    across campaigns.
    """

    LAG_PERIODS = [1, 7]

    FEATURE_COLUMNS = [
        "spend",
        "revenue",
        "clicks",
        "conversions",
    ]

    GROUP_COLUMNS = [
        "platform",
        "campaign_id",
    ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag features.

        Parameters
        ----------
        df : pd.DataFrame
            Canonical dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe with lag features.
        """
        df = df.copy()

        required = self.GROUP_COLUMNS + ["date"] + self.FEATURE_COLUMNS

        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"LagFeatureTransformer missing required columns: {missing}"
            )

        # Always sort before shifting
        df = df.sort_values(
            self.GROUP_COLUMNS + ["date"]
        ).reset_index(drop=True)

        grouped = df.groupby(
            self.GROUP_COLUMNS,
            sort=False,
        )

        for column in self.FEATURE_COLUMNS:
            for lag in self.LAG_PERIODS:
                df[f"{column}_lag_{lag}"] = grouped[column].shift(lag)

        return df