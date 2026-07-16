"""
aletheia.features.base
======================

Base abstractions for feature engineering.

Every feature transformer in Aletheia inherits from
BaseFeatureTransformer and implements a single method:

    transform(df) -> pd.DataFrame

This guarantees a consistent interface across the entire
feature engineering pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFeatureTransformer(ABC):
    """
    Base class for all feature engineering modules.

    Every transformer receives a canonical DataFrame,
    adds or modifies feature columns, and returns a new
    DataFrame without mutating the input.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a canonical dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Canonical dataframe produced by the ingestion pipeline.

        Returns
        -------
        pd.DataFrame
            Dataframe with additional engineered features.
        """
        raise NotImplementedError