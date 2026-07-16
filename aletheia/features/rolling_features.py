"""
aletheia.features.rolling_features
==================================

Rolling window feature engineering.

Creates rolling statistical features for forecasting models by
computing historical trends within each (platform, campaign_id).

To prevent data leakage, all rolling statistics are calculated
using only past observations via shift(1).

Generated Features
------------------
For each metric:
- rolling_mean_7
- rolling_mean_14
- rolling_std_7
- rolling_std_14
- rolling_min_7
- rolling_min_14
- rolling_max_7
- rolling_max_14
"""

from __future__ import annotations

import pandas as pd

from .base import BaseFeatureTransformer


class RollingFeatureTransformer(BaseFeatureTransformer):
    """
    Generate rolling statistical features for campaign-level forecasting.

    Rolling statistics are computed independently for every
    (platform, campaign_id) combination using only historical
    observations to avoid data leakage.
    """

    GROUP_COLUMNS = [
        "platform",
        "campaign_id",
    ]

    FEATURE_COLUMNS = [
        "spend",
        "revenue",
        "clicks",
        "conversions",
    ]

    WINDOWS = [7, 14]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling statistical features.

        Parameters
        ----------
        df : pd.DataFrame
            Feature dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe with rolling statistical features.
        """
        df = df.copy()

        required = self.GROUP_COLUMNS + ["date"] + self.FEATURE_COLUMNS

        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"RollingFeatureTransformer missing required columns: {missing}"
            )

        # Ensure chronological order before rolling calculations
        df = (
            df.sort_values(self.GROUP_COLUMNS + ["date"])
              .reset_index(drop=True)
        )

        grouped = df.groupby(self.GROUP_COLUMNS, sort=False)

        for column in self.FEATURE_COLUMNS:

            for window in self.WINDOWS:

                # Rolling Mean (historical only)
                df[f"{column}_rolling_mean_{window}"] = grouped[column].transform(
                    lambda s: s.shift(1).rolling(
                        window=window,
                        min_periods=1,
                    ).mean()
                )

                # Rolling Standard Deviation
                df[f"{column}_rolling_std_{window}"] = (
                    grouped[column]
                    .transform(
                        lambda s: s.shift(1).rolling(
                            window=window,
                            min_periods=1,
                        ).std()
                    )
                    .fillna(0)
                )

                # Rolling Minimum
                df[f"{column}_rolling_min_{window}"] = grouped[column].transform(
                    lambda s: s.shift(1).rolling(
                        window=window,
                        min_periods=1,
                    ).min()
                )

                # Rolling Maximum
                df[f"{column}_rolling_max_{window}"] = grouped[column].transform(
                    lambda s: s.shift(1).rolling(
                        window=window,
                        min_periods=1,
                    ).max()
                )

        return df