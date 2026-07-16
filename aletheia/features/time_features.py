"""
aletheia.features.time_features
===============================

Time-based feature engineering.

Extracts calendar features from the canonical `date` column to help
forecasting models learn recurring temporal patterns.

Generated Features
------------------
- day_of_week
- day_name
- month
- month_name
- quarter
- week_of_year
- day_of_month
- day_of_year
- is_weekend
"""

from __future__ import annotations

import pandas as pd

from .base import BaseFeatureTransformer


class TimeFeatureTransformer(BaseFeatureTransformer):
    """
    Generate calendar-based features from the canonical date column.

    The input dataframe is never modified in-place.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.

        Parameters
        ----------
        df : pd.DataFrame
            Canonical dataframe.

        Returns
        -------
        pd.DataFrame
            Copy of dataframe with additional calendar features.
        """
        df = df.copy()

        if "date" not in df.columns:
            raise ValueError(
                "TimeFeatureTransformer requires a 'date' column."
            )

        # Ensure datetime dtype
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Calendar features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_name"] = df["date"].dt.day_name()

        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.month_name()

        df["quarter"] = df["date"].dt.quarter

        df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int64")

        df["day_of_month"] = df["date"].dt.day
        df["day_of_year"] = df["date"].dt.dayofyear

        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int64")

        return df