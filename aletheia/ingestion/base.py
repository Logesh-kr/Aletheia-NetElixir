"""
aletheia.ingestion.base
=======================
Abstract base class and canonical schema definition for all platform connectors.

Design notes
------------
- All CSVs are read as ``dtype=str`` to prevent silent Pandas coercions on
  mixed-format columns (e.g. Bing date strings, comma-formatted numbers).
- Each subclass implements :meth:`_map_to_canonical`, which is responsible only
  for *field remapping*. Type coercion and column-order enforcement are handled
  once in :meth:`_apply_shared_transforms` on the base class.
- ``campaign_id`` is a deterministic MD5 hash of ``platform::campaign_name``.
  Replace with the native platform campaign ID when integrating live APIs.
"""


from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

#: Ordered list of columns every normalised dataframe must expose.
CANONICAL_COLUMNS: list[str] = [
    "date",
    "platform",
    "campaign_id",
    "campaign_name",
    "spend",
    "revenue",
    "conversions",
    "clicks",
    "impressions",
]

#: Target dtypes applied by :meth:`BaseAdConnector._apply_shared_transforms`.
#: ``revenue`` is intentionally float64 (nullable via NaN) rather than
#: pandas nullable Int64 — downstream Pandas operations behave more
#: predictably with NaN than pd.NA for numeric columns.
CANONICAL_DTYPES: dict[str, str] = {
    "date": "datetime64[ns]",
    "platform": "object",
    "campaign_id": "object",
    "campaign_name": "object",
    "spend": "float64",
    "revenue": "float64",       # NaN where no native revenue signal exists
    "conversions": "float64",   # float because fractional conversions are valid
    "clicks": "int64",
    "impressions": "int64",
}


# ---------------------------------------------------------------------------
# Abstract base connector
# ---------------------------------------------------------------------------

class BaseAdConnector(ABC):
    """
    Abstract base for Google Ads, Meta Ads, and Microsoft/Bing Ads connectors.

    Subclasses must implement:
        - :attr:`PLATFORM_NAME` — string identifier (e.g. ``"google_ads"``)
        - :attr:`required_source_columns` — list of columns that must exist in
          the raw CSV before normalisation begins
        - :meth:`_map_to_canonical` — field remapping logic; returns a DataFrame
          that *at minimum* contains every column in :data:`CANONICAL_COLUMNS`

    The base class handles:
        - File loading (``dtype=str`` for safety)
        - Source-column presence validation
        - Shared type coercions and column ordering
        - Deterministic ``campaign_id`` generation
        - Logging at each stage
    """

    #: Unique identifier stamped into the ``platform`` column.  Must be set
    #: by every concrete subclass as a class-level attribute.
    PLATFORM_NAME: str = ""

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def required_source_columns(self) -> list[str]:
        """
        Columns that *must* be present in the raw source CSV.

        Checked before :meth:`_map_to_canonical` is called; raises
        :class:`ValueError` if any are absent.
        """
        ...

    @abstractmethod
    def _map_to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remap platform-specific column names to the canonical schema.

        Parameters
        ----------
        df:
            Raw CSV as a DataFrame (all columns are ``str`` dtype at this point).

        Returns
        -------
        pd.DataFrame
            DataFrame containing every column in :data:`CANONICAL_COLUMNS`.
            Values may still be string or mixed-type; coercion happens in the
            base class :meth:`_apply_shared_transforms`.
        """
        ...

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> pd.DataFrame:
        """
        Load a platform CSV and return a normalised canonical DataFrame.

        Parameters
        ----------
        path:
            Absolute or relative path to the platform CSV file.

        Returns
        -------
        pd.DataFrame
            Normalised DataFrame with columns as defined in
            :data:`CANONICAL_COLUMNS` and types as in :data:`CANONICAL_DTYPES`.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If any column listed in :attr:`required_source_columns` is absent
            from the raw CSV.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"[{self.PLATFORM_NAME}] File not found: {path}"
            )

        logger.info("[%s] Loading: %s", self.PLATFORM_NAME, path)

        # Read everything as str to prevent silent Pandas coercions
        raw_df = pd.read_csv(path, dtype=str)

        self._validate_source_columns(raw_df)
        mapped_df = self._map_to_canonical(raw_df)
        canonical_df = self._apply_shared_transforms(mapped_df)

        logger.info(
            "[%s] Loaded %d rows, %d campaigns.",
            self.PLATFORM_NAME,
            len(canonical_df),
            canonical_df["campaign_id"].nunique(),
        )
        return canonical_df

    # ------------------------------------------------------------------
    # Shared transforms (applied to every platform after mapping)
    # ------------------------------------------------------------------

    def _apply_shared_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce canonical column presence, ordering, and dtypes.

        This method is intentionally platform-agnostic.  Any
        platform-specific logic belongs in :meth:`_map_to_canonical`.
        """
        # Ensure every canonical column exists (fill missing with NaN)
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA

        # Select and order canonical columns only
        df = df[CANONICAL_COLUMNS].copy()

        # --- Date ---
        df["date"] = self._parse_dates(df["date"])

        # --- String columns ---
        df["platform"] = self.PLATFORM_NAME
        df["campaign_id"] = df["campaign_id"].astype(str)
        df["campaign_name"] = df["campaign_name"].astype(str).str.strip()

        # --- Monetary / count columns ---
        df["spend"] = pd.to_numeric(df["spend"], errors="coerce").astype("float64")
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").astype("float64")

        # Conversions: fractional values are valid (view-through attribution)
        df["conversions"] = pd.to_numeric(df["conversions"], errors="coerce").astype("float64")

        # Clicks / impressions: fill parse failures with 0 (not NaN) to keep
        # these columns integer-safe; a row with unparseable clicks is still
        # useful for revenue / spend analysis.
        df["clicks"] = (
            pd.to_numeric(df["clicks"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
        df["impressions"] = (
            pd.to_numeric(df["impressions"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )

        return df

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_source_columns(self, df: pd.DataFrame) -> None:
        """
        Raise :class:`ValueError` if any required source columns are absent.
        """
        missing = sorted(set(self.required_source_columns) - set(df.columns))
        if missing:
            raise ValueError(
                f"[{self.PLATFORM_NAME}] Missing required source columns: {missing}. "
                f"Found: {sorted(df.columns.tolist())}"
            )

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_dates(series: pd.Series) -> pd.Series:
        """
        Robustly parse a string Series into ``datetime64[ns]``.

        Handles mixed formats (ISO 8601, MM/DD/YYYY, DD-Mon-YYYY, etc.)
        by delegating to ``pd.to_datetime`` with ``dayfirst=False`` and
        ``format="mixed"`` (Pandas >= 2.0).  Falls back to the legacy
        ``infer_datetime_format=True`` for older Pandas versions.
        """
        try:
            # Pandas >= 2.0 — preferred; explicit and unambiguous.
            # Cast to datetime64[ns] because pd.to_datetime with format="mixed"
            # returns datetime64[us] in Pandas 2.x, which would fail dtype checks.
            return pd.to_datetime(
                series,
                format="mixed",
                errors="coerce",
                dayfirst=False,
            ).astype("datetime64[ns]")
        except TypeError:
            # Pandas < 2.0 fallback
            return pd.to_datetime(
                series,
                infer_datetime_format=True,
                errors="coerce",
                dayfirst=False,
            ).astype("datetime64[ns]")

    @staticmethod
    def _make_campaign_id(platform: str, campaign_name: str) -> str:
        """
        Generate a stable, deterministic surrogate campaign ID.

        Uses the first 16 hex characters of MD5(``platform::campaign_name``).
        This is sufficient for deduplication within Aletheia; it is *not*
        cryptographically secure and is not intended to be.

        Replace with the native integer/string campaign ID when ingesting
        from live platform APIs rather than CSV exports.

        Parameters
        ----------
        platform:
            Platform identifier (e.g. ``"google_ads"``).
        campaign_name:
            Human-readable campaign name as it appears in the source CSV.

        Returns
        -------
        str
            16-character lowercase hex string.
        """
        key = f"{platform}::{campaign_name}".encode("utf-8")
        return hashlib.md5(key, usedforsecurity=False).hexdigest()[:16]

    @staticmethod
    def _strip_commas(series: pd.Series) -> pd.Series:
        """
        Remove thousands-separator commas from a string Series.

        Used for Bing Ads, which emits numbers like ``"1,234.56"``.
        """
        return series.str.replace(",", "", regex=False)
