"""
tests.test_ingestion
====================
Comprehensive test suite for the Aletheia ingestion pipeline.

Coverage
--------
Unit tests
    - GoogleAdsConnector: micros conversion, zero-revenue preserved, schema
    - MetaAdsConnector: revenue always NaN, ctr/cpc/cpm dropped, schema
    - BingAdsConnector: comma-formatted numbers, MM/DD/YYYY dates, zero-revenue preserved
    - BaseAdConnector: missing source column raises ValueError
    - CanonicalValidator: negative value sanitisation (negative values -> 0),
      null drop behaviour

Integration tests
    - IngestionPipeline.run() with all three platforms
    - IngestionPipeline.run() with a single platform
    - IngestionPipeline.run() with no paths raises ValueError
    - Row counts, date range, platform values, canonical column set
    - Revenue preservation across connectors

Schema conformance tests
    - All canonical columns present
    - Correct dtypes on every column
    - date column is datetime64[ns]
    - platform values are expected strings
    - spend and impressions are non-negative after validation
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import pytest

from aletheia.ingestion import (
    BingAdsConnector,
    GoogleAdsConnector,
    IngestionPipeline,
    MetaAdsConnector,
)
from aletheia.ingestion.base import CANONICAL_COLUMNS
from aletheia.ingestion.validator import CanonicalValidator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"

GOOGLE_CSV = FIXTURES_DIR / "google_ads_sample.csv"
META_CSV = FIXTURES_DIR / "meta_ads_sample.csv"
BING_CSV = FIXTURES_DIR / "bing_ads_sample.csv"


@pytest.fixture(scope="module")
def google_df() -> pd.DataFrame:
    """Normalised Google Ads DataFrame from sample fixture."""
    return GoogleAdsConnector().load(GOOGLE_CSV)


@pytest.fixture(scope="module")
def meta_df() -> pd.DataFrame:
    """Normalised Meta Ads DataFrame from sample fixture."""
    return MetaAdsConnector().load(META_CSV)


@pytest.fixture(scope="module")
def bing_df() -> pd.DataFrame:
    """Normalised Bing Ads DataFrame from sample fixture."""
    return BingAdsConnector().load(BING_CSV)


@pytest.fixture(scope="module")
def unified_df() -> pd.DataFrame:
    """Unified DataFrame from all three platform fixtures."""
    pipeline = IngestionPipeline()
    return pipeline.run(
        google_ads_path=GOOGLE_CSV,
        meta_ads_path=META_CSV,
        bing_ads_path=BING_CSV,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def assert_canonical_schema(df: pd.DataFrame) -> None:
    """Assert all canonical columns are present and correctly typed."""
    assert list(df.columns) == CANONICAL_COLUMNS, (
        f"Column order mismatch.\nExpected: {CANONICAL_COLUMNS}\nGot: {df.columns.tolist()}"
    )
    assert df["date"].dtype == "datetime64[ns]", "date must be datetime64[ns]"
    assert df["spend"].dtype == "float64", "spend must be float64"
    assert df["revenue"].dtype == "float64", "revenue must be float64"
    assert df["conversions"].dtype == "float64", "conversions must be float64"
    assert df["clicks"].dtype == "int64", "clicks must be int64"
    assert df["impressions"].dtype == "int64", "impressions must be int64"


# ===========================================================================
# 1. Google Ads Connector
# ===========================================================================

class TestGoogleAdsConnector:

    def test_canonical_schema(self, google_df):
        assert_canonical_schema(google_df)

    def test_platform_label(self, google_df):
        assert (google_df["platform"] == "google_ads").all()

    def test_row_count(self, google_df):
        # Sample fixture has 9 rows (including 1 paused campaign)
        assert len(google_df) == 9

    def test_micros_to_spend_conversion(self, google_df):
        """45,000,000 micros must equal 45.00 spend."""
        brand_jan15 = google_df[
            (google_df["campaign_name"] == "Brand - Search") &
            (google_df["date"] == pd.Timestamp("2024-01-15"))
        ]
        assert len(brand_jan15) == 1
        assert brand_jan15["spend"].iloc[0] == pytest.approx(45.00, rel=1e-6)

    def test_another_micros_conversion(self, google_df):
        """32,000,000 micros must equal 32.00 spend."""
        row = google_df[
            (google_df["campaign_name"] == "Generic - Shopping") &
            (google_df["date"] == pd.Timestamp("2024-01-15"))
        ]
        assert row["spend"].iloc[0] == pytest.approx(32.00, rel=1e-6)

    def test_zero_revenue_preserved(self, google_df):
        """Paused Campaign has metrics_conversions_value=0; revenue should remain 0."""
        paused = google_df[google_df["campaign_name"] == "Paused Campaign"]
        assert len(paused) == 1
        assert paused["revenue"].iloc[0] == pytest.approx(0.0)

    def test_positive_revenue_preserved(self, google_df):
        """Non-zero revenue values must be preserved as-is."""
        row = google_df[
            (google_df["campaign_name"] == "Brand - Search") &
            (google_df["date"] == pd.Timestamp("2024-01-15"))
        ]
        assert row["revenue"].iloc[0] == pytest.approx(450.00, rel=1e-6)

    def test_campaign_id_is_deterministic(self, google_df):
        """Same campaign name must always produce the same campaign_id."""
        ids = google_df[google_df["campaign_name"] == "Brand - Search"]["campaign_id"].unique()
        assert len(ids) == 1

    def test_campaign_id_differs_across_campaigns(self, google_df):
        """Different campaign names must produce different campaign_ids."""
        id_brand = google_df[google_df["campaign_name"] == "Brand - Search"]["campaign_id"].iloc[0]
        id_generic = google_df[google_df["campaign_name"] == "Generic - Shopping"]["campaign_id"].iloc[0]
        assert id_brand != id_generic

    def test_dates_are_datetime(self, google_df):
        assert google_df["date"].dtype == "datetime64[ns]"

    def test_clicks_are_integer(self, google_df):
        assert google_df["clicks"].dtype == "int64"

    def test_missing_column_raises(self, tmp_path):
        """CSV missing a required column must raise ValueError."""
        bad_csv = tmp_path / "bad_google.csv"
        bad_csv.write_text(
            "segments_date,campaign_name,metrics_clicks\n"
            "2024-01-15,Test,100\n"
        )
        with pytest.raises(ValueError, match="Missing required source columns"):
            GoogleAdsConnector().load(bad_csv)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            GoogleAdsConnector().load("/nonexistent/path/google.csv")


# ===========================================================================
# 2. Meta Ads Connector
# ===========================================================================

class TestMetaAdsConnector:

    def test_canonical_schema(self, meta_df):
        assert_canonical_schema(meta_df)

    def test_platform_label(self, meta_df):
        assert (meta_df["platform"] == "meta_ads").all()

    def test_revenue_always_nan(self, meta_df):
        """Every Meta row must have revenue = NaN — no native revenue signal."""
        assert meta_df["revenue"].isna().all(), (
            "Meta revenue must be NaN for all rows. "
            f"Found non-null: {meta_df['revenue'].dropna()}"
        )

    def test_ctr_not_in_output(self, meta_df):
        """ctr is a platform-computed ratio and must not appear in output."""
        assert "ctr" not in meta_df.columns

    def test_cpc_not_in_output(self, meta_df):
        assert "cpc" not in meta_df.columns

    def test_cpm_not_in_output(self, meta_df):
        assert "cpm" not in meta_df.columns

    def test_spend_is_float(self, meta_df):
        assert meta_df["spend"].dtype == "float64"

    def test_spend_values_correct(self, meta_df):
        row = meta_df[
            (meta_df["campaign_name"] == "Retargeting - Dynamic") &
            (meta_df["date"] == pd.Timestamp("2024-01-15"))
        ]
        assert row["spend"].iloc[0] == pytest.approx(78.50, rel=1e-6)

    def test_row_count(self, meta_df):
        # Sample fixture has 9 rows
        assert len(meta_df) == 9

    def test_dates_normalised(self, meta_df):
        assert meta_df["date"].dtype == "datetime64[ns]"

    def test_missing_column_raises(self, tmp_path):
        bad_csv = tmp_path / "bad_meta.csv"
        bad_csv.write_text(
            "date_start,campaign_name,impressions\n"
            "2024-01-15,Test,1000\n"
        )
        with pytest.raises(ValueError, match="Missing required source columns"):
            MetaAdsConnector().load(bad_csv)


# ===========================================================================
# 3. Bing Ads Connector
# ===========================================================================

class TestBingAdsConnector:

    def test_canonical_schema(self, bing_df):
        assert_canonical_schema(bing_df)

    def test_platform_label(self, bing_df):
        assert (bing_df["platform"] == "microsoft_ads").all()

    def test_mmddyyyy_date_parsed(self, bing_df):
        """1/15/2024 must be parsed as January 15, 2024."""
        dates = bing_df["date"].dt.date.astype(str).unique()
        assert "2024-01-15" in dates

    def test_comma_formatted_impressions(self, bing_df):
        """'1,600' must be parsed as integer 1600."""
        row = bing_df[
            (bing_df["campaign_name"] == "Generic Shopping") &
            (bing_df["date"] == pd.Timestamp("2024-01-16"))
        ]
        assert len(row) == 1
        assert row["impressions"].iloc[0] == 1600

    def test_comma_formatted_spend(self, bing_df):
        """'2,800' impressions row from Jan-17 Brand Search."""
        row = bing_df[
            (bing_df["campaign_name"] == "Brand Search") &
            (bing_df["date"] == pd.Timestamp("2024-01-17"))
        ]
        assert row["impressions"].iloc[0] == 2800

    def test_zero_revenue_preserved(self, bing_df):
        """Revenue=0.00 in source should remain 0."""
        row = bing_df[
            (bing_df["campaign_name"] == "Generic Shopping") &
            (bing_df["date"] == pd.Timestamp("2024-01-16"))
        ]
        assert len(row) == 1
        assert row["revenue"].iloc[0] == pytest.approx(0.0)

    def test_positive_revenue_preserved(self, bing_df):
        row = bing_df[
            (bing_df["campaign_name"] == "Brand Search") &
            (bing_df["date"] == pd.Timestamp("2024-01-15"))
        ]
        assert row["revenue"].iloc[0] == pytest.approx(420.00, rel=1e-6)

    def test_row_count(self, bing_df):
        # Sample fixture has 9 rows
        assert len(bing_df) == 9

    def test_missing_column_raises(self, tmp_path):
        bad_csv = tmp_path / "bad_bing.csv"
        bad_csv.write_text(
            "TimePeriod,CampaignName,Clicks\n"
            "1/15/2024,Test,50\n"
        )
        with pytest.raises(ValueError, match="Missing required source columns"):
            BingAdsConnector().load(bad_csv)


# ===========================================================================
# 4. Canonical Validator
# ===========================================================================

class TestCanonicalValidator:

    def _make_valid_df(self) -> pd.DataFrame:
        """Minimal valid canonical DataFrame for validator tests."""
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-15", "2024-01-16"]),
            "platform": ["google_ads", "meta_ads"],
            "campaign_id": ["abc123", "def456"],
            "campaign_name": ["Campaign A", "Campaign B"],
            "spend": [45.0, 78.5],
            "revenue": [450.0, float("nan")],
            "conversions": [8.0, 12.0],
            "clicks": [120, 210],
            "impressions": [3500, 15000],
        })

    def test_valid_df_passes(self):
        df = self._make_valid_df()
        result = CanonicalValidator().validate(df)
        assert len(result) == 2

    def test_negative_spend_set_to_zero(self):
        df = self._make_valid_df()
        df.loc[0, "spend"] = -10.0
        result = CanonicalValidator().validate(df)
        assert result.loc[0, "spend"] == 0
        
    def test_negative_clicks_set_to_zero(self):
        df = self._make_valid_df()
        df.loc[0, "clicks"] = -5
        result = CanonicalValidator().validate(df)
        assert result.loc[0, "clicks"] == 0

    def test_null_spend_row_dropped(self):
        df = self._make_valid_df()
        df.loc[0, "spend"] = float("nan")
        result = CanonicalValidator().validate(df)
        assert len(result) == 1

    def test_null_campaign_name_row_dropped(self):
        df = self._make_valid_df()
        df.loc[1, "campaign_name"] = float("nan")
        result = CanonicalValidator().validate(df)
        assert len(result) == 1

    def test_revenue_null_not_dropped(self):
        """NaN revenue must NOT cause row removal — it is expected for Meta."""
        df = self._make_valid_df()
        # Both rows: one has revenue, one doesn't
        result = CanonicalValidator().validate(df)
        assert len(result) == 2

    def test_missing_canonical_column_raises(self):
        df = self._make_valid_df().drop(columns=["impressions"])
        with pytest.raises(ValueError, match="Canonical columns missing"):
            CanonicalValidator().validate(df)


# ===========================================================================
# 5. Integration — IngestionPipeline
# ===========================================================================

class TestIngestionPipeline:

    def test_all_canonical_columns_present(self, unified_df):
        assert unified_df.columns.tolist() == CANONICAL_COLUMNS

    def test_three_platforms_present(self, unified_df):
        platforms = set(unified_df["platform"].unique())
        assert platforms == {"google_ads", "meta_ads", "microsoft_ads"}

    def test_total_row_count(self, unified_df):
        # Google: 9, Meta: 9, Bing: 9 → 27 total
        assert len(unified_df) == 27

    def test_sorted_by_date_platform_campaign(self, unified_df):
        """DataFrame must be sorted by (date, platform, campaign_id) ascending."""
        expected = unified_df.sort_values(
            ["date", "platform", "campaign_id"]
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(unified_df, expected)

    def test_date_range_spans_three_days(self, unified_df):
        assert unified_df["date"].min() == pd.Timestamp("2024-01-15")
        assert unified_df["date"].max() == pd.Timestamp("2024-01-17")

    def test_google_spend_correct_in_unified(self, unified_df):
        """Validate micros conversion is preserved after merge."""
        row = unified_df[
            (unified_df["platform"] == "google_ads") &
            (unified_df["campaign_name"] == "Brand - Search") &
            (unified_df["date"] == pd.Timestamp("2024-01-15"))
        ]
        assert row["spend"].iloc[0] == pytest.approx(45.00, rel=1e-6)

    def test_meta_revenue_nan_in_unified(self, unified_df):
        """All Meta rows in the unified DF must still have NaN revenue."""
        meta_rows = unified_df[unified_df["platform"] == "meta_ads"]
        assert meta_rows["revenue"].isna().all()

    def test_bing_zero_revenue_preserved(self, unified_df):
        """Bing rows with Revenue=0.00 in source should remain 0 in the unified DF."""
        row = unified_df[
            (unified_df["platform"] == "microsoft_ads") &
            (unified_df["campaign_name"] == "Generic Shopping") &
            (unified_df["date"] == pd.Timestamp("2024-01-16"))
        ]
        assert len(row) == 1
        assert row["revenue"].iloc[0] == pytest.approx(0.0)

    def test_no_paths_raises(self):
        with pytest.raises(ValueError, match="at least one platform path"):
            IngestionPipeline().run()

    def test_single_platform_run(self):
        """Pipeline must work with only one platform provided."""
        df = IngestionPipeline().run(google_ads_path=GOOGLE_CSV)
        assert (df["platform"] == "google_ads").all()
        assert len(df) == 9

    def test_file_not_found_propagates(self):
        with pytest.raises(FileNotFoundError):
            IngestionPipeline().run(google_ads_path="/no/such/file.csv")

    def test_schema_conformance_after_merge(self, unified_df):
        assert_canonical_schema(unified_df)

    def test_no_negative_spend_after_merge(self, unified_df):
        assert (unified_df["spend"].dropna() >= 0).all()

    def test_no_negative_impressions_after_merge(self, unified_df):
        assert (unified_df["impressions"] >= 0).all()

    def test_campaign_ids_differ_across_platforms(self, unified_df):
        """
        Campaign with same name on different platforms must have different IDs
        because campaign_id is hashed from (platform + campaign_name).
        """
        # "Brand - Search" exists on Google; "Brand Search" on Bing
        # They differ by name — but even same names differ by platform prefix.
        google_ids = set(
            unified_df[unified_df["platform"] == "google_ads"]["campaign_id"]
        )
        bing_ids = set(
            unified_df[unified_df["platform"] == "microsoft_ads"]["campaign_id"]
        )
        # No ID collision across platforms
        assert google_ids.isdisjoint(bing_ids)
