"""
aletheia.ingestion
==================
Data ingestion and normalization pipeline for Aletheia.

Loads raw CSV exports from Google Ads, Meta Ads, and Microsoft/Bing Ads,
normalises each into the canonical schema, and merges them into a single
unified pandas DataFrame.

Public API
----------
    from aletheia.ingestion import IngestionPipeline

    pipeline = IngestionPipeline()
    df = pipeline.run(
        google_ads_path="data/google.csv",
        meta_ads_path="data/meta.csv",
        bing_ads_path="data/bing.csv",
    )
"""

from .merger import IngestionPipeline
from .google_ads import GoogleAdsConnector
from .meta_ads import MetaAdsConnector
from .bing_ads import BingAdsConnector

__all__ = [
    "IngestionPipeline",
    "GoogleAdsConnector",
    "MetaAdsConnector",
    "BingAdsConnector",
]
