# -*- coding: utf-8 -*-
"""
Spend Extractor - AGENT-DATA-003: ERP/Finance Connector
========================================================

Extracts and analyses vendor spend data from ERP connections. Generates
deterministic simulated spend records for testing and development, and
provides summarisation, trend analysis, top-vendor analysis, and
category-level breakdowns.

Supports:
    - Multi-vendor spend extraction with date-range filtering
    - Vendor ID and cost-center filtering
    - Spend summarisation by vendor, category, and period
    - Top-N vendor ranking by total spend
    - Spend-by-category breakdowns
    - Monthly and quarterly trend analysis
    - Record filtering by amount, vendor, and category
    - Deterministic hash-based simulated data (no randomness)
    - Thread-safe statistics counters

Zero-Hallucination Guarantees:
    - All simulated data is deterministic (hash-seeded, not random)
    - All aggregations are pure arithmetic on extracted records
    - No LLM involvement in spend extraction or summarisation
    - SHA-256 provenance hashes for audit trails

Example:
    >>> from greenlang.erp_connector.spend_extractor import SpendExtractor
    >>> extractor = SpendExtractor()
    >>> records = extractor.extract_spend(
    ...     connection_id="conn-abc123",
    ...     start_date=date(2024, 1, 1),
    ...     end_date=date(2024, 3, 31),
    ... )
    >>> summary = extractor.summarize_spend(records)
    >>> print(summary.total_spend_usd)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-003 ERP/Finance Connector
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Layer 1 imports
from greenlang.agents.data.erp_connector_agent import (
    SpendCategory,
    SpendRecord,
    TransactionType,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SpendSummary",
    "SpendExtractor",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _hash_int(seed: str, modulus: int) -> int:
    """Deterministic integer from a seed string.

    Args:
        seed: String to hash.
        modulus: Upper bound (exclusive) for the result.

    Returns:
        Integer in [0, modulus).
    """
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulus


def _hash_float(seed: str, low: float, high: float) -> float:
    """Deterministic float in [low, high] from a seed string.

    Args:
        seed: String to hash.
        low: Minimum value (inclusive).
        high: Maximum value (inclusive).

    Returns:
        Float in [low, high].
    """
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    return round(low + fraction * (high - low), 2)


# ---------------------------------------------------------------------------
# Simulated vendor catalogue
# ---------------------------------------------------------------------------

_VENDOR_CATALOGUE: List[Tuple[str, str, SpendCategory]] = [
    ("V001", "Steel Supplier Inc", SpendCategory.DIRECT_MATERIALS),
    ("V002", "Packaging Corp", SpendCategory.INDIRECT_MATERIALS),
    ("V003", "Consulting Partners", SpendCategory.PROFESSIONAL_SERVICES),
    ("V004", "Freight Services LLC", SpendCategory.TRANSPORT),
    ("V005", "Energy Provider", SpendCategory.ENERGY),
    ("V006", "IT Solutions", SpendCategory.IT_SERVICES),
    ("V007", "Travel Agency", SpendCategory.TRAVEL),
    ("V008", "Equipment Supplier", SpendCategory.CAPITAL_EQUIPMENT),
    ("V009", "Office Supplies", SpendCategory.INDIRECT_MATERIALS),
    ("V010", "Facility Services", SpendCategory.FACILITIES),
]

_COST_CENTERS = ["CC001", "CC002", "CC003", "CC004"]
_GL_ACCOUNTS = ["5000", "5100", "5200", "6000", "6100"]
_TRANSACTION_TYPES = [
    TransactionType.INVOICE,
    TransactionType.PURCHASE_ORDER,
    TransactionType.GOODS_RECEIPT,
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SpendSummary(BaseModel):
    """Aggregated spend summary."""

    total_spend_usd: float = Field(
        ..., description="Total spend in USD",
    )
    record_count: int = Field(
        ..., ge=0, description="Number of spend records",
    )
    vendor_count: int = Field(
        ..., ge=0, description="Unique vendor count",
    )
    by_vendor: Dict[str, float] = Field(
        default_factory=dict,
        description="Spend totals by vendor ID",
    )
    by_category: Dict[str, float] = Field(
        default_factory=dict,
        description="Spend totals by spend category",
    )
    by_period: Dict[str, float] = Field(
        default_factory=dict,
        description="Spend totals by YYYY-MM period",
    )
    date_range_start: Optional[date] = Field(
        None, description="Earliest transaction date",
    )
    date_range_end: Optional[date] = Field(
        None, description="Latest transaction date",
    )
    provenance_hash: Optional[str] = Field(
        None, description="SHA-256 provenance hash",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# SpendExtractor
# ---------------------------------------------------------------------------


class SpendExtractor:
    """Vendor spend data extractor with deterministic simulation.

    Extracts spend records from ERP connections (simulated mode) and
    provides aggregation, trend analysis, and filtering utilities.
    All simulated data is deterministic via SHA-256-seeded generation.

    Attributes:
        _config: Configuration dictionary.
        _lock: Threading lock for statistics.
        _stats: Extraction statistics counters.

    Example:
        >>> ext = SpendExtractor()
        >>> records = ext.extract_spend("conn-abc", date(2024,1,1), date(2024,1,31))
        >>> assert len(records) > 0
        >>> summary = ext.summarize_spend(records)
        >>> assert summary.total_spend_usd > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SpendExtractor.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``records_per_day``: int (default 10)
                - ``min_amount``: float (default 100.0)
                - ``max_amount``: float (default 50000.0)
                - ``default_currency``: str (default "USD")
        """
        self._config = config or {}
        self._records_per_day: int = self._config.get("records_per_day", 10)
        self._min_amount: float = self._config.get("min_amount", 100.0)
        self._max_amount: float = self._config.get("max_amount", 50000.0)
        self._default_currency: str = self._config.get(
            "default_currency", "USD",
        )
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "records_extracted": 0,
            "total_spend_usd": 0.0,
            "extractions_count": 0,
            "errors": 0,
        }
        logger.info(
            "SpendExtractor initialised: records_per_day=%d, "
            "amount_range=[%.2f, %.2f]",
            self._records_per_day,
            self._min_amount,
            self._max_amount,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_spend(
        self,
        connection_id: str,
        start_date: date,
        end_date: date,
        vendor_ids: Optional[List[str]] = None,
        cost_centers: Optional[List[str]] = None,
        company_codes: Optional[List[str]] = None,
    ) -> List[SpendRecord]:
        """Extract spend records for a date range.

        Generates deterministic simulated spend data and applies
        optional vendor and cost-center filters.

        Args:
            connection_id: ERP connection identifier (used as seed).
            start_date: Extraction start date (inclusive).
            end_date: Extraction end date (inclusive).
            vendor_ids: Optional vendor ID filter list.
            cost_centers: Optional cost-center filter list.
            company_codes: Optional company-code filter list (reserved).

        Returns:
            List of SpendRecord objects.

        Raises:
            ValueError: If start_date > end_date.
        """
        start = time.monotonic()

        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must be <= end_date ({end_date})"
            )

        # Generate simulated data
        records = self._simulate_spend_data(
            connection_id, start_date, end_date, vendor_ids, cost_centers,
        )

        # Compute total spend
        total_spend = sum(r.amount_usd or r.amount for r in records)

        with self._lock:
            self._stats["records_extracted"] += len(records)
            self._stats["total_spend_usd"] += total_spend
            self._stats["extractions_count"] += 1

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Extracted %d spend records for %s: period=%s to %s, "
            "total=%.2f USD (%.1f ms)",
            len(records), connection_id, start_date, end_date,
            total_spend, elapsed_ms,
        )
        return records

    def summarize_spend(
        self,
        records: List[SpendRecord],
    ) -> SpendSummary:
        """Summarize spend records by vendor, category, and period.

        Args:
            records: List of SpendRecord objects.

        Returns:
            SpendSummary with aggregated totals.
        """
        if not records:
            return SpendSummary(
                total_spend_usd=0.0,
                record_count=0,
                vendor_count=0,
            )

        by_vendor: Dict[str, float] = defaultdict(float)
        by_category: Dict[str, float] = defaultdict(float)
        by_period: Dict[str, float] = defaultdict(float)
        dates: List[date] = []

        for r in records:
            amount = r.amount_usd if r.amount_usd is not None else r.amount
            by_vendor[r.vendor_id] += amount
            cat_key = r.spend_category.value if r.spend_category else "other"
            by_category[cat_key] += amount
            period_key = r.transaction_date.strftime("%Y-%m")
            by_period[period_key] += amount
            dates.append(r.transaction_date)

        total_spend = sum(by_vendor.values())

        provenance_hash = self._compute_provenance(
            "summarize", str(len(records)), str(total_spend),
        )

        return SpendSummary(
            total_spend_usd=round(total_spend, 2),
            record_count=len(records),
            vendor_count=len(by_vendor),
            by_vendor={k: round(v, 2) for k, v in by_vendor.items()},
            by_category={k: round(v, 2) for k, v in by_category.items()},
            by_period={k: round(v, 2) for k, v in sorted(by_period.items())},
            date_range_start=min(dates),
            date_range_end=max(dates),
            provenance_hash=provenance_hash,
        )

    def get_top_vendors(
        self,
        records: List[SpendRecord],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get the top N vendors by total spend.

        Args:
            records: List of SpendRecord objects.
            top_n: Number of top vendors to return (default 10).

        Returns:
            List of dicts with vendor_id, vendor_name, total_spend,
            record_count, and rank.
        """
        vendor_data: Dict[str, Dict[str, Any]] = {}

        for r in records:
            amount = r.amount_usd if r.amount_usd is not None else r.amount
            if r.vendor_id not in vendor_data:
                vendor_data[r.vendor_id] = {
                    "vendor_id": r.vendor_id,
                    "vendor_name": r.vendor_name,
                    "total_spend": 0.0,
                    "record_count": 0,
                }
            vendor_data[r.vendor_id]["total_spend"] += amount
            vendor_data[r.vendor_id]["record_count"] += 1

        # Sort by total spend descending
        sorted_vendors = sorted(
            vendor_data.values(),
            key=lambda v: v["total_spend"],
            reverse=True,
        )

        results: List[Dict[str, Any]] = []
        for rank, vendor in enumerate(sorted_vendors[:top_n], start=1):
            results.append({
                "rank": rank,
                "vendor_id": vendor["vendor_id"],
                "vendor_name": vendor["vendor_name"],
                "total_spend": round(vendor["total_spend"], 2),
                "record_count": vendor["record_count"],
            })

        return results

    def get_spend_by_category(
        self,
        records: List[SpendRecord],
    ) -> Dict[str, float]:
        """Get total spend broken down by spend category.

        Args:
            records: List of SpendRecord objects.

        Returns:
            Dictionary of category -> total spend (USD).
        """
        by_category: Dict[str, float] = defaultdict(float)

        for r in records:
            amount = r.amount_usd if r.amount_usd is not None else r.amount
            cat_key = r.spend_category.value if r.spend_category else "other"
            by_category[cat_key] += amount

        return {k: round(v, 2) for k, v in by_category.items()}

    def get_spend_trend(
        self,
        records: List[SpendRecord],
        granularity: str = "monthly",
    ) -> List[Dict[str, Any]]:
        """Get spend trend over time.

        Groups spend records by month or quarter and returns an ordered
        list of period buckets with totals and record counts.

        Args:
            records: List of SpendRecord objects.
            granularity: "monthly" or "quarterly" (default "monthly").

        Returns:
            List of dicts with period, total_spend, record_count,
            ordered chronologically.
        """
        buckets: Dict[str, Dict[str, Any]] = {}

        for r in records:
            amount = r.amount_usd if r.amount_usd is not None else r.amount
            period_key = self._get_period_key(
                r.transaction_date, granularity,
            )
            if period_key not in buckets:
                buckets[period_key] = {
                    "period": period_key,
                    "total_spend": 0.0,
                    "record_count": 0,
                }
            buckets[period_key]["total_spend"] += amount
            buckets[period_key]["record_count"] += 1

        # Round and sort
        result: List[Dict[str, Any]] = []
        for key in sorted(buckets.keys()):
            entry = buckets[key]
            entry["total_spend"] = round(entry["total_spend"], 2)
            result.append(entry)

        return result

    def filter_records(
        self,
        records: List[SpendRecord],
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        vendors: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> List[SpendRecord]:
        """Filter spend records by amount, vendor, and/or category.

        Args:
            records: List of SpendRecord objects.
            min_amount: Minimum amount filter (inclusive).
            max_amount: Maximum amount filter (inclusive).
            vendors: Vendor ID filter list.
            categories: Spend category value filter list.

        Returns:
            Filtered list of SpendRecord objects.
        """
        filtered = records

        if min_amount is not None:
            filtered = [
                r for r in filtered
                if (r.amount_usd or r.amount) >= min_amount
            ]

        if max_amount is not None:
            filtered = [
                r for r in filtered
                if (r.amount_usd or r.amount) <= max_amount
            ]

        if vendors is not None:
            vendor_set = set(vendors)
            filtered = [r for r in filtered if r.vendor_id in vendor_set]

        if categories is not None:
            cat_set = set(categories)
            filtered = [
                r for r in filtered
                if r.spend_category is not None
                and r.spend_category.value in cat_set
            ]

        logger.debug(
            "Filtered %d -> %d records", len(records), len(filtered),
        )
        return filtered

    def get_statistics(self) -> Dict[str, Any]:
        """Return extraction statistics.

        Returns:
            Dictionary of counter values.
        """
        with self._lock:
            return {
                "records_extracted": self._stats["records_extracted"],
                "total_spend_usd": round(
                    self._stats["total_spend_usd"], 2,
                ),
                "extractions_count": self._stats["extractions_count"],
                "errors": self._stats["errors"],
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Simulated data generation
    # ------------------------------------------------------------------

    def _simulate_spend_data(
        self,
        connection_id: str,
        start_date: date,
        end_date: date,
        vendor_ids: Optional[List[str]],
        cost_centers: Optional[List[str]],
    ) -> List[SpendRecord]:
        """Generate deterministic simulated spend data.

        Uses SHA-256-based seeding to produce repeatable records for
        each day in the date range. Each record's attributes are
        derived from a composite seed of connection_id, date, and
        index to guarantee determinism.

        Args:
            connection_id: Used as part of the deterministic seed.
            start_date: First date for simulated records.
            end_date: Last date for simulated records.
            vendor_ids: Optional vendor filter (post-generation).
            cost_centers: Optional cost-center filter (post-generation).

        Returns:
            List of SpendRecord objects.
        """
        records: List[SpendRecord] = []
        current_date = start_date
        day_index = 0

        while current_date <= end_date:
            # Deterministic records per day
            day_seed = f"{connection_id}:{current_date.isoformat()}"
            num_records = self._records_per_day + _hash_int(
                f"{day_seed}:count", 6,
            )  # 10 to 15 records per day

            for i in range(num_records):
                record_seed = f"{day_seed}:{i}"
                record = self._generate_single_record(
                    record_seed, current_date, day_index, i,
                )
                records.append(record)

            current_date += timedelta(days=1)
            day_index += 1

        # Apply filters
        if vendor_ids:
            vendor_set = set(vendor_ids)
            records = [r for r in records if r.vendor_id in vendor_set]

        if cost_centers:
            cc_set = set(cost_centers)
            records = [r for r in records if r.cost_center in cc_set]

        return records

    def _generate_single_record(
        self,
        seed: str,
        transaction_date: date,
        day_index: int,
        record_index: int,
    ) -> SpendRecord:
        """Generate a single deterministic SpendRecord.

        Args:
            seed: Composite seed string for determinism.
            transaction_date: Date for this record.
            day_index: Day offset from start.
            record_index: Record index within the day.

        Returns:
            SpendRecord with all fields populated.
        """
        # Select vendor
        vendor_idx = _hash_int(f"{seed}:vendor", len(_VENDOR_CATALOGUE))
        vendor_id, vendor_name, spend_cat = _VENDOR_CATALOGUE[vendor_idx]

        # Determine amount
        amount = _hash_float(
            f"{seed}:amount", self._min_amount, self._max_amount,
        )

        # Transaction type
        tx_idx = _hash_int(f"{seed}:txtype", len(_TRANSACTION_TYPES))
        tx_type = _TRANSACTION_TYPES[tx_idx]

        # Cost center and GL account
        cc_idx = _hash_int(f"{seed}:cc", len(_COST_CENTERS))
        gl_idx = _hash_int(f"{seed}:gl", len(_GL_ACCOUNTS))

        # Record ID is deterministic
        record_id_hash = hashlib.sha256(
            seed.encode("utf-8"),
        ).hexdigest()[:8].upper()

        return SpendRecord(
            record_id=f"SPD-{record_id_hash}",
            transaction_type=tx_type,
            transaction_date=transaction_date,
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            amount=amount,
            currency=self._default_currency,
            amount_usd=amount,
            description=f"Purchase from {vendor_name}",
            cost_center=_COST_CENTERS[cc_idx],
            gl_account=_GL_ACCOUNTS[gl_idx],
            spend_category=spend_cat,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_period_key(
        self,
        d: date,
        granularity: str,
    ) -> str:
        """Get a period key for a date at the specified granularity.

        Args:
            d: Date to classify.
            granularity: "monthly" or "quarterly".

        Returns:
            Period key string (e.g. "2024-01" or "2024-Q1").
        """
        if granularity == "quarterly":
            quarter = (d.month - 1) // 3 + 1
            return f"{d.year}-Q{quarter}"
        return d.strftime("%Y-%m")

    def _compute_provenance(self, *parts: str) -> str:
        """Compute SHA-256 provenance hash from parts.

        Args:
            *parts: Strings to include in the hash.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        combined = json.dumps(
            {"parts": list(parts), "timestamp": _utcnow().isoformat()},
            sort_keys=True,
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
