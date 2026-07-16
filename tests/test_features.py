"""
tests.test_features
===================
Test suite for the Aletheia feature engineering pipeline.

Coverage
--------
Unit tests — TimeFeatureTransformer
    - Calendar features generated: day_of_week, month, quarter, week_of_year,
      day_of_month, day_of_year, is_weekend
    - is_weekend correct for Monday (0) and Saturday (1)
    - Row count preserved; input not mutated
    - Missing 'date' column raises ValueError

Unit tests — MarketingFeatureTransformer
    - KPI features generated: ctr, cpc, cpm, roas, conversion_rate,
      revenue_per_click, revenue_per_impression
    - ctr value correct for known inputs
    - Zero impressions/clicks/spend produce NaN (no ZeroDivisionError)
    - Missing required column raises ValueError
    - Row count preserved; input not mutated

Unit tests — LagFeatureTransformer
    - Lag features generated: spend_lag_1, spend_lag_7, revenue_lag_1, etc.
    - First observation per campaign has NaN lag_1
    - lag_1 on day 2 equals day 1 value
    - No cross-campaign data leakage
    - Row count preserved; missing column raises ValueError

Unit tests — RollingFeatureTransformer
    - Rolling features generated: spend_rolling_mean_7, rolling_std_7, etc.
    - No future leakage — rolling mean on day 2 equals day 1 spend
    - rolling_std filled with 0 (no NaN)
    - Row count preserved; missing column raises ValueError

Integration tests — FeaturePipeline
    - Output has more columns than input
    - All original canonical columns preserved
    - Input not mutated; row count preserved
    - Time, marketing, lag, and rolling features all present
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from aletheia.features.time_features import TimeFeatureTransformer
from aletheia.features.marketing_features import MarketingFeatureTransformer
from aletheia.features.lag_features import LagFeatureTransformer
from aletheia.features.rolling_features import RollingFeatureTransformer
from aletheia.features.pipeline import FeaturePipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canonical_df(n_days: int = 10, n_campaigns: int = 2) -> pd.DataFrame:
    """Build a minimal canonical DataFrame for feature engineering tests."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for campaign_idx in range(n_campaigns):
        for date in dates:
            rows.append({
                "date": date,
                "platform": "google_ads",
                "campaign_id": f"campaign_{campaign_idx:02d}",
                "campaign_name": f"Campaign {campaign_idx}",
                "spend": float(10 + campaign_idx + date.day),
                "revenue": float(100 + campaign_idx * 10 + date.day),
                "conversions": float(1 + date.day % 3),
                "clicks": int(50 + date.day),
                "impressions": int(1000 + date.day * 10),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def canonical_df() -> pd.DataFrame:
    """Canonical DataFrame with 2 campaigns × 10 days = 20 rows."""
    return _make_canonical_df(n_days=10, n_campaigns=2)


# ===========================================================================
# 1. TimeFeatureTransformer
# ===========================================================================

class TestTimeFeatureTransformer:

    def test_day_of_week_present(self, canonical_df):
        result = TimeFeatureTransformer().transform(canonical_df)
        assert "day_of_week" in result.columns

    def test_month_present(self, canonical_df):
        result = TimeFeatureTransformer().transform(canonical_df)
        assert "month" in result.columns

    def test_quarter_present(self, canonical_df):
        result = TimeFeatureTransformer().transform(canonical_df)
        assert "quarter" in result.columns

    def test_week_of_year_present(self, canonical_df):
        result = TimeFeatureTransformer().transform(canonical_df)
        assert "week_of_year" in result.columns

    def test_day_of_month_present(self, canonical_df):
        result = TimeFeatureTransformer().transform(canonical_df)
        assert "day_of_month" in result.columns

    def test_day_of_year_present(self, canonical_df):
        result = TimeFeatureTransformer().transform(canonical_df)
        assert "day_of_year" in result.columns

    def test_is_weekend_present(self, canonical_df):
        result = TimeFeatureTransformer().transform(canonical_df)
        assert "is_weekend" in result.columns

    def test_is_weekend_binary(self, canonical_df):
        result = TimeFeatureTransformer().transform(canonical_df)
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_is_weekend_monday_is_zero(self):
        """2024-01-01 is a Monday — is_weekend must be 0."""
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "platform": ["google_ads"],
            "campaign_id": ["c1"],
            "campaign_name": ["Campaign 1"],
            "spend": [10.0],
            "revenue": [100.0],
            "conversions": [1.0],
            "clicks": [50],
            "impressions": [1000],
        })
        result = TimeFeatureTransformer().transform(df)
        assert result["is_weekend"].iloc[0] == 0

    def test_is_weekend_saturday_is_one(self):
        """2024-01-06 is a Saturday — is_weekend must be 1."""
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-06")],
            "platform": ["google_ads"],
            "campaign_id": ["c1"],
            "campaign_name": ["Campaign 1"],
            "spend": [10.0],
            "revenue": [100.0],
            "conversions": [1.0],
            "clicks": [50],
            "impressions": [1000],
        })
        result = TimeFeatureTransformer().transform(df)
        assert result["is_weekend"].iloc[0] == 1

    def test_row_count_preserved(self, canonical_df):
        result = TimeFeatureTransformer().transform(canonical_df)
        assert len(result) == len(canonical_df)

    def test_input_not_mutated(self, canonical_df):
        original_cols = canonical_df.columns.tolist()
        TimeFeatureTransformer().transform(canonical_df)
        assert canonical_df.columns.tolist() == original_cols

    def test_missing_date_column_raises(self):
        df = pd.DataFrame({"spend": [10.0], "revenue": [100.0]})
        with pytest.raises(ValueError, match="date"):
            TimeFeatureTransformer().transform(df)


# ===========================================================================
# 2. MarketingFeatureTransformer
# ===========================================================================

class TestMarketingFeatureTransformer:

    def test_ctr_present(self, canonical_df):
        result = MarketingFeatureTransformer().transform(canonical_df)
        assert "ctr" in result.columns

    def test_cpc_present(self, canonical_df):
        result = MarketingFeatureTransformer().transform(canonical_df)
        assert "cpc" in result.columns

    def test_cpm_present(self, canonical_df):
        result = MarketingFeatureTransformer().transform(canonical_df)
        assert "cpm" in result.columns

    def test_roas_present(self, canonical_df):
        result = MarketingFeatureTransformer().transform(canonical_df)
        assert "roas" in result.columns

    def test_conversion_rate_present(self, canonical_df):
        result = MarketingFeatureTransformer().transform(canonical_df)
        assert "conversion_rate" in result.columns

    def test_revenue_per_click_present(self, canonical_df):
        result = MarketingFeatureTransformer().transform(canonical_df)
        assert "revenue_per_click" in result.columns

    def test_revenue_per_impression_present(self, canonical_df):
        result = MarketingFeatureTransformer().transform(canonical_df)
        assert "revenue_per_impression" in result.columns

    def test_ctr_value_correct(self):
        """ctr = clicks / impressions = 50 / 1000 = 0.05."""
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "platform": ["google_ads"],
            "campaign_id": ["c1"],
            "campaign_name": ["Campaign 1"],
            "spend": [100.0],
            "revenue": [500.0],
            "conversions": [5.0],
            "clicks": [50],
            "impressions": [1000],
        })
        result = MarketingFeatureTransformer().transform(df)
        assert result["ctr"].iloc[0] == pytest.approx(0.05, rel=1e-6)

    def test_zero_impressions_yields_nan(self):
        """Zero impressions must produce NaN in ctr and cpm, not ZeroDivisionError."""
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "platform": ["google_ads"],
            "campaign_id": ["c1"],
            "campaign_name": ["Campaign 1"],
            "spend": [10.0],
            "revenue": [0.0],
            "conversions": [0.0],
            "clicks": [0],
            "impressions": [0],
        })
        result = MarketingFeatureTransformer().transform(df)
        assert pd.isna(result["ctr"].iloc[0])
        assert pd.isna(result["cpm"].iloc[0])

    def test_missing_column_raises(self):
        df = pd.DataFrame({"spend": [10.0], "clicks": [5]})
        with pytest.raises(ValueError, match="missing required columns"):
            MarketingFeatureTransformer().transform(df)

    def test_row_count_preserved(self, canonical_df):
        result = MarketingFeatureTransformer().transform(canonical_df)
        assert len(result) == len(canonical_df)

    def test_input_not_mutated(self, canonical_df):
        original_cols = canonical_df.columns.tolist()
        MarketingFeatureTransformer().transform(canonical_df)
        assert canonical_df.columns.tolist() == original_cols


# ===========================================================================
# 3. LagFeatureTransformer
# ===========================================================================

class TestLagFeatureTransformer:

    def test_spend_lag_1_present(self, canonical_df):
        result = LagFeatureTransformer().transform(canonical_df)
        assert "spend_lag_1" in result.columns

    def test_spend_lag_7_present(self, canonical_df):
        result = LagFeatureTransformer().transform(canonical_df)
        assert "spend_lag_7" in result.columns

    def test_revenue_lag_1_present(self, canonical_df):
        result = LagFeatureTransformer().transform(canonical_df)
        assert "revenue_lag_1" in result.columns

    def test_revenue_lag_7_present(self, canonical_df):
        result = LagFeatureTransformer().transform(canonical_df)
        assert "revenue_lag_7" in result.columns

    def test_first_row_per_campaign_has_nan_lag(self):
        """The first observation per (platform, campaign_id) group must have NaN lag_1."""
        df = _make_canonical_df(n_days=5, n_campaigns=1)
        result = (
            LagFeatureTransformer()
            .transform(df)
            .sort_values(["campaign_id", "date"])
            .reset_index(drop=True)
        )
        assert pd.isna(result.loc[0, "spend_lag_1"])

    def test_lag_1_correct_value(self):
        """spend_lag_1 on day 2 must equal spend on day 1."""
        df = _make_canonical_df(n_days=5, n_campaigns=1)
        result = (
            LagFeatureTransformer()
            .transform(df)
            .sort_values(["campaign_id", "date"])
            .reset_index(drop=True)
        )
        assert result.loc[1, "spend_lag_1"] == pytest.approx(
            result.loc[0, "spend"], rel=1e-6
        )

    def test_no_cross_campaign_leakage(self):
        """First row of every campaign must have NaN lag regardless of other campaigns."""
        df = _make_canonical_df(n_days=5, n_campaigns=3)
        result = LagFeatureTransformer().transform(df)
        first_per_campaign = (
            result.sort_values(["campaign_id", "date"])
            .groupby("campaign_id")
            .head(1)
            .reset_index(drop=True)
        )
        assert first_per_campaign["spend_lag_1"].isna().all()

    def test_row_count_preserved(self, canonical_df):
        result = LagFeatureTransformer().transform(canonical_df)
        assert len(result) == len(canonical_df)

    def test_missing_column_raises(self):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "spend": [10.0],
        })
        with pytest.raises(ValueError, match="missing required columns"):
            LagFeatureTransformer().transform(df)


# ===========================================================================
# 4. RollingFeatureTransformer
# ===========================================================================

class TestRollingFeatureTransformer:

    def test_spend_rolling_mean_7_present(self, canonical_df):
        result = RollingFeatureTransformer().transform(canonical_df)
        assert "spend_rolling_mean_7" in result.columns

    def test_spend_rolling_mean_14_present(self, canonical_df):
        result = RollingFeatureTransformer().transform(canonical_df)
        assert "spend_rolling_mean_14" in result.columns

    def test_spend_rolling_std_7_present(self, canonical_df):
        result = RollingFeatureTransformer().transform(canonical_df)
        assert "spend_rolling_std_7" in result.columns

    def test_spend_rolling_min_7_present(self, canonical_df):
        result = RollingFeatureTransformer().transform(canonical_df)
        assert "spend_rolling_min_7" in result.columns

    def test_spend_rolling_max_7_present(self, canonical_df):
        result = RollingFeatureTransformer().transform(canonical_df)
        assert "spend_rolling_max_7" in result.columns

    def test_no_future_leakage(self):
        """Rolling mean on day 2 must equal day 1 spend (only 1 past observation)."""
        df = _make_canonical_df(n_days=14, n_campaigns=1)
        result = (
            RollingFeatureTransformer()
            .transform(df)
            .sort_values(["campaign_id", "date"])
            .reset_index(drop=True)
        )
        expected = result.loc[0, "spend"]
        assert result.loc[1, "spend_rolling_mean_7"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_rolling_std_no_nan(self, canonical_df):
        """rolling_std is filled with 0 for insufficient window — NaN must not appear."""
        result = RollingFeatureTransformer().transform(canonical_df)
        assert not result["spend_rolling_std_7"].isna().any()

    def test_row_count_preserved(self, canonical_df):
        result = RollingFeatureTransformer().transform(canonical_df)
        assert len(result) == len(canonical_df)

    def test_missing_column_raises(self):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "spend": [10.0],
        })
        with pytest.raises(ValueError, match="missing required columns"):
            RollingFeatureTransformer().transform(df)


# ===========================================================================
# 5. Integration — FeaturePipeline
# ===========================================================================

class TestFeaturePipeline:

    def test_output_has_more_columns_than_input(self, canonical_df):
        result = FeaturePipeline().transform(canonical_df)
        assert len(result.columns) > len(canonical_df.columns)

    def test_all_canonical_columns_preserved(self, canonical_df):
        result = FeaturePipeline().transform(canonical_df)
        for col in canonical_df.columns:
            assert col in result.columns, (
                f"Canonical column '{col}' missing from FeaturePipeline output."
            )

    def test_input_not_mutated(self, canonical_df):
        original_shape = canonical_df.shape
        original_cols = canonical_df.columns.tolist()
        FeaturePipeline().transform(canonical_df)
        assert canonical_df.shape == original_shape
        assert canonical_df.columns.tolist() == original_cols

    def test_row_count_preserved(self, canonical_df):
        result = FeaturePipeline().transform(canonical_df)
        assert len(result) == len(canonical_df)

    def test_time_features_present(self, canonical_df):
        result = FeaturePipeline().transform(canonical_df)
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "quarter" in result.columns

    def test_marketing_features_present(self, canonical_df):
        result = FeaturePipeline().transform(canonical_df)
        assert "ctr" in result.columns
        assert "roas" in result.columns
        assert "cpc" in result.columns

    def test_lag_features_present(self, canonical_df):
        result = FeaturePipeline().transform(canonical_df)
        assert "spend_lag_1" in result.columns
        assert "revenue_lag_7" in result.columns
        assert "clicks_lag_1" in result.columns

    def test_rolling_features_present(self, canonical_df):
        result = FeaturePipeline().transform(canonical_df)
        assert "spend_rolling_mean_7" in result.columns
        assert "revenue_rolling_mean_14" in result.columns
        assert "spend_rolling_std_7" in result.columns
