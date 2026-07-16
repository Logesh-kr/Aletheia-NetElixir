"""
aletheia.ingestion.validator
=============================
Post-normalisation validation of the canonical unified DataFrame.

Validation philosophy
---------------------
The validator enforces **data quality contracts** on the canonical schema
after all platform normalisation is complete.  It distinguishes between:

- **Critical failures** — missing required columns or rows that cannot be
  used at all (dropped or raised).
- **Recoverable anomalies** — rows with suspicious but non-fatal values
  (sanitised in place with a warning).
- **Expected nulls** — ``revenue = NaN`` for Meta rows is normal and never
  triggers a warning.

Design notes
------------
- The validator *never* raises on revenue nulls.  Meta revenue is
  intentionally absent; raising here would break ingestion for any pipeline
  that includes Meta data.
- Negative spend/clicks/impressions are sanitised to ``NaN`` rather than
  dropped, because the row may still be useful for conversion or revenue
  analysis.
- Future datetime rows trigger a warning, not a drop — it is valid to
  pre-load a campaign CSV that includes scheduled future-dated rows.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import CANONICAL_COLUMNS

logger = logging.getLogger(__name__)

#: Columns that must not be null after normalisation.
#: Revenue is intentionally excluded — Meta rows always have NaN revenue.
REQUIRED_NON_NULL: list[str] = [
    "date",
    "platform",
    "campaign_id",
    "campaign_name",
    "spend",
    "conversions",
    "clicks",
    "impressions",
]

#: Numeric columns where negative values are anomalous.
NON_NEGATIVE_COLUMNS: list[str] = [
    "spend",
    "conversions",
    "clicks",
    "impressions",
]


class CanonicalValidator:
    """
    Validate a normalised canonical DataFrame and return a cleaned copy.

    Usage
    -----
    ::

        validator = CanonicalValidator()
        clean_df = validator.validate(df)

    Raises
    ------
    ValueError
        If required canonical columns are absent from the DataFrame.
    """

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all validation checks and return a cleaned DataFrame.

        Parameters
        ----------
        df:
            Normalised canonical DataFrame (output of one or more connectors
            after concatenation).

        Returns
        -------
        pd.DataFrame
            Cleaned copy of ``df`` — may have fewer rows if null-required-
            column rows were dropped.

        Raises
        ------
        ValueError
            If any column in :data:`CANONICAL_COLUMNS` is absent.
        """
        self._check_schema(df)
        df = df.copy()
        df = self._enforce_required_non_null(df)
        df = self._sanitise_negative_values(df)
        df = self._audit_revenue_nulls(df)
        self._check_date_range(df)
        self._log_summary(df)
        return df

    # ------------------------------------------------------------------
    # Validation steps
    # ------------------------------------------------------------------

    def _check_schema(self, df: pd.DataFrame) -> None:
        """
        Raise if any canonical column is absent.

        This is a hard failure — a missing column indicates a broken
        connector, not a data quality issue.
        """
        missing = sorted(set(CANONICAL_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(
                f"[Validator] Canonical columns missing from DataFrame: {missing}. "
                "Check connector output."
            )

    def _enforce_required_non_null(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where a required column is null and log the count.

        ``revenue`` is intentionally excluded from this check — null
        revenue for Meta rows is expected and handled by the imputation
        pipeline.
        """
        for col in REQUIRED_NON_NULL:
            null_mask = df[col].isna()
            null_count = null_mask.sum()
            if null_count > 0:
                logger.warning(
                    "[Validator] '%s' has %d null value(s) — %d row(s) dropped.",
                    col,
                    null_count,
                    null_count,
                )
                df = df[~null_mask]
        return df

    def _sanitise_negative_values(self, df: pd.DataFrame) -> pd.DataFrame:

        """Set negative values in numeric columns to 0.Negative spend, clicks, conversions, or impressions are considered
        source-data anomalies. Replacing them with 0 prevents NaN values
        from propagating into downstream feature engineering and modeling.
        """
        for col in NON_NEGATIVE_COLUMNS:
            if col not in df.columns:
                continue

            neg_mask = df[col] < 0
            neg_count = neg_mask.sum()

            if neg_count > 0:
                logger.warning(
                    "[Validator] '%s' has %d negative value(s) — setting to 0.",
                    col,
                    neg_count,
                )

                df.loc[neg_mask, col] = 0

        return df

    def _audit_revenue_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Log a per-platform breakdown of revenue nulls.

        This is an informational audit step, not a corrective one.
        """
        revenue_nulls = df["revenue"].isna().sum()
        total = len(df)
        if revenue_nulls > 0:
            # Break down by platform for actionable diagnostics
            null_by_platform = (
                df[df["revenue"].isna()]
                .groupby("platform")
                .size()
                .to_dict()
            )
            logger.info(
                "[Validator] revenue is NaN for %d/%d rows (%.1f%%). "
                "Breakdown by platform: %s. "
                "Meta NaN is expected; others require imputation review.",
                revenue_nulls,
                total,
                100.0 * revenue_nulls / total if total > 0 else 0,
                null_by_platform,
            )
        return df

    def _check_date_range(self, df: pd.DataFrame) -> None:
        """
        Log the date range and warn if future-dated rows are present.
        """
        
        if df.empty:
            logger.warning("[Validator] DataFrame is empty after validation.")
            return
        nat_count = df["date"].isna().sum()

        if nat_count > 0:
            logger.warning("[Validator] %d rows contain invalid dates.",nat_count)

        min_date = df["date"].min()
        max_date = df["date"].max()
        today = pd.Timestamp.today().normalize()

        future_count = (df["date"] > today).sum()
        if future_count > 0:
            logger.warning(
                "[Validator] %d row(s) have future dates (max: %s). "
                "Verify source data is not pre-loaded scheduled data.",
                future_count,
                max_date.date(),
            )

        logger.info(
            "[Validator] Date range: %s → %s (%d calendar days).",
            min_date.date(),
            max_date.date(),
            (max_date - min_date).days + 1,
        )

    def _log_summary(self, df: pd.DataFrame) -> None:
        """
        Emit a structured summary log after all validation passes.
        """
        logger.info(
            "[Validator] Validation complete. "
            "rows=%d | platforms=%s | campaigns=%d | revenue_null=%d",
            len(df),
            sorted(df["platform"].unique().tolist()),
            df["campaign_id"].nunique(),
            df["revenue"].isna().sum(),
        )
