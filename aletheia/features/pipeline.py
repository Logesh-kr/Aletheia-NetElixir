"""
aletheia.features.pipeline
==========================

Feature engineering pipeline for Aletheia.

Applies all feature transformers in sequence to produce the
final feature dataframe used by forecasting models.
"""

from __future__ import annotations

import pandas as pd
from .rolling_features import RollingFeatureTransformer
from .base import BaseFeatureTransformer
from .lag_features import LagFeatureTransformer
from .marketing_features import MarketingFeatureTransformer
from .time_features import TimeFeatureTransformer


class FeaturePipeline:
    """
    Sequential feature engineering pipeline.

    Every transformer follows the BaseFeatureTransformer
    interface and is executed in order.
    """

    def __init__(self) -> None:
        self._transformers: list[BaseFeatureTransformer] = [
            TimeFeatureTransformer(),
            MarketingFeatureTransformer(),
            LagFeatureTransformer(),
            RollingFeatureTransformer(),
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature transformers.

        Parameters
        ----------
        df : pd.DataFrame
            Canonical dataframe.

        Returns
        -------
        pd.DataFrame
            Feature engineered dataframe.
        """
        df = df.copy()

        for transformer in self._transformers:
            df = transformer.transform(df)

        return df