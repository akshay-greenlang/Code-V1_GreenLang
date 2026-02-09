# -*- coding: utf-8 -*-
"""
Spend Analytics Engine - AGENT-DATA-009: Spend Data Categorizer
=================================================================

Provides spend analytics, hotspot identification, Pareto/ABC analysis,
concentration metrics, intensity calculations, trend analysis,
variance analysis, and industry benchmarking for categorised spend data.

Supports:
    - Aggregation by Scope 3 category, vendor, and time period
    - ABC / Pareto analysis (80/20 rule)
    - Emissions hotspot identification (top-N emitters)
    - Herfindahl-Hirschman Index (HHI) and concentration ratios
    - Emissions intensity by category (kgCO2e per USD)
    - Year-over-year and quarter-over-quarter trend analysis
    - Variance analysis against baseline periods
    - Industry benchmarking (configurable benchmarks)
    - Thread-safe in-memory storage
    - SHA-256 provenance hashes on all analytics

Zero-Hallucination Guarantees:
    - All analytics are pure arithmetic aggregations
    - No LLM or ML model in analytics or reporting
    - HHI and concentration ratios are standard formulae
    - SHA-256 provenance hashes for audit trails

Example:
    >>> from greenlang.spend_categorizer.spend_analytics import SpendAnalyticsEngine
    >>> engine = SpendAnalyticsEngine()
    >>> hotspots = engine.identify_hotspots(records, top_n=5)
    >>> pareto = engine.pareto_analysis(records)
    >>> print(pareto["a_class_count"], pareto["a_class_percentage"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-009 Spend Data Categorizer (GL-DATA-SUP-002)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "SpendAggregate",
    "HotspotResult",
    "TrendDataPoint",
    "SpendAnalyticsEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "agg") -> str:
    """Generate a unique identifier with a prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Industry benchmarks (kgCO2e per USD of revenue, by industry)
# ---------------------------------------------------------------------------

_INDUSTRY_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "technology": {"intensity_kgco2e_per_usd": 0.08, "cat1_share": 0.55, "cat2_share": 0.10, "cat6_share": 0.12},
    "manufacturing": {"intensity_kgco2e_per_usd": 0.45, "cat1_share": 0.60, "cat2_share": 0.15, "cat4_share": 0.10},
    "financial_services": {"intensity_kgco2e_per_usd": 0.05, "cat1_share": 0.40, "cat6_share": 0.25, "cat15_share": 0.20},
    "healthcare": {"intensity_kgco2e_per_usd": 0.18, "cat1_share": 0.55, "cat2_share": 0.12, "cat5_share": 0.08},
    "retail": {"intensity_kgco2e_per_usd": 0.22, "cat1_share": 0.65, "cat4_share": 0.15, "cat9_share": 0.08},
    "energy": {"intensity_kgco2e_per_usd": 1.20, "cat1_share": 0.35, "cat3_share": 0.30, "cat11_share": 0.15},
    "construction": {"intensity_kgco2e_per_usd": 0.52, "cat1_share": 0.50, "cat2_share": 0.25, "cat4_share": 0.10},
    "transportation": {"intensity_kgco2e_per_usd": 0.65, "cat1_share": 0.30, "cat3_share": 0.25, "cat4_share": 0.20},
    "agriculture": {"intensity_kgco2e_per_usd": 0.85, "cat1_share": 0.45, "cat3_share": 0.15, "cat5_share": 0.10},
    "professional_services": {"intensity_kgco2e_per_usd": 0.06, "cat1_share": 0.50, "cat6_share": 0.20, "cat7_share": 0.10},
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SpendAggregate(BaseModel):
    """Aggregated spend and emissions for a group."""

    group_key: str = Field(..., description="Group identifier (category, vendor, period)")
    group_type: str = Field(default="category", description="Aggregation type")
    record_count: int = Field(default=0, ge=0, description="Number of records")
    total_spend_usd: float = Field(default=0.0, description="Total spend in USD")
    total_emissions_kgco2e: float = Field(default=0.0, description="Total emissions kgCO2e")
    total_emissions_tco2e: float = Field(default=0.0, description="Total emissions tCO2e")
    avg_spend_usd: float = Field(default=0.0, description="Average spend per record")
    avg_emissions_kgco2e: float = Field(default=0.0, description="Average emissions per record")
    intensity_kgco2e_per_usd: float = Field(default=0.0, description="Emissions intensity")
    share_of_spend: float = Field(default=0.0, description="Percentage share of total spend")
    share_of_emissions: float = Field(default=0.0, description="Percentage share of total emissions")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = {"extra": "forbid"}


class HotspotResult(BaseModel):
    """Emissions hotspot identification result."""

    rank: int = Field(..., ge=1, description="Hotspot rank (1=highest)")
    group_key: str = Field(..., description="Hotspot identifier")
    group_type: str = Field(default="vendor", description="Hotspot type")
    total_spend_usd: float = Field(default=0.0, description="Total spend")
    total_emissions_kgco2e: float = Field(default=0.0, description="Total emissions")
    share_of_emissions: float = Field(default=0.0, description="Percentage of total emissions")
    cumulative_share: float = Field(default=0.0, description="Cumulative share up to this rank")
    intensity_kgco2e_per_usd: float = Field(default=0.0, description="Emissions intensity")
    record_count: int = Field(default=0, description="Number of records")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = {"extra": "forbid"}


class TrendDataPoint(BaseModel):
    """A single data point in a trend analysis."""

    period: str = Field(..., description="Period label (e.g. 2024-Q1, 2024-01)")
    period_index: int = Field(default=0, description="Numeric period index")
    total_spend_usd: float = Field(default=0.0, description="Period spend")
    total_emissions_kgco2e: float = Field(default=0.0, description="Period emissions")
    record_count: int = Field(default=0, description="Period record count")
    spend_change_pct: Optional[float] = Field(None, description="Spend change % vs prior period")
    emissions_change_pct: Optional[float] = Field(None, description="Emissions change % vs prior period")
    intensity_kgco2e_per_usd: float = Field(default=0.0, description="Period intensity")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# SpendAnalyticsEngine
# ---------------------------------------------------------------------------


class SpendAnalyticsEngine:
    """Spend analytics and hotspot identification engine.

    Provides aggregation, Pareto analysis, concentration metrics,
    trend analysis, variance analysis, and industry benchmarking
    for categorised spend records.

    All analytics are deterministic arithmetic computations.

    Attributes:
        _config: Configuration dictionary.
        _lock: Threading lock for thread-safe mutations.
        _stats: Cumulative analytics statistics.

    Example:
        >>> engine = SpendAnalyticsEngine()
        >>> aggs = engine.aggregate_by_category(records)
        >>> hotspots = engine.identify_hotspots(records, top_n=5)
        >>> pareto = engine.pareto_analysis(records)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SpendAnalyticsEngine.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``default_top_n``: int (default 10)
                - ``pareto_threshold``: float (default 0.8)
                - ``industry``: str (default "manufacturing")
        """
        self._config = config or {}
        self._default_top_n: int = self._config.get("default_top_n", 10)
        self._pareto_threshold: float = self._config.get("pareto_threshold", 0.8)
        self._default_industry: str = self._config.get("industry", "manufacturing")
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "analyses_performed": 0,
            "records_analysed": 0,
            "by_analysis_type": {},
            "errors": 0,
        }
        logger.info(
            "SpendAnalyticsEngine initialised: top_n=%d, pareto=%.2f, "
            "industry=%s",
            self._default_top_n,
            self._pareto_threshold,
            self._default_industry,
        )

    # ------------------------------------------------------------------
    # Public API - Aggregation
    # ------------------------------------------------------------------

    def aggregate_by_category(
        self,
        records: List[Dict[str, Any]],
        group_by: str = "scope3_category",
    ) -> List[SpendAggregate]:
        """Aggregate records by Scope 3 category or custom field.

        Args:
            records: List of spend record dicts with ``amount_usd``,
                ``emissions_kgco2e``, and the ``group_by`` field.
            group_by: Field name to group by (default "scope3_category").

        Returns:
            List of SpendAggregate objects sorted by emissions descending.
        """
        return self._aggregate(records, group_by, "category")

    def aggregate_by_vendor(
        self,
        records: List[Dict[str, Any]],
        top_n: int = 20,
    ) -> List[SpendAggregate]:
        """Aggregate records by vendor.

        Args:
            records: List of spend record dicts.
            top_n: Number of top vendors to return.

        Returns:
            List of SpendAggregate objects sorted by spend descending.
        """
        aggs = self._aggregate(records, "vendor_name", "vendor")
        return aggs[:top_n]

    def aggregate_by_period(
        self,
        records: List[Dict[str, Any]],
        timeframe: str = "quarterly",
    ) -> List[SpendAggregate]:
        """Aggregate records by time period.

        Args:
            records: List of spend record dicts with ``transaction_date``.
            timeframe: Aggregation level: "monthly", "quarterly", "yearly".

        Returns:
            List of SpendAggregate objects sorted by period ascending.
        """
        # Assign period labels
        for rec in records:
            rec["_period"] = self._assign_period(
                rec.get("transaction_date", ""), timeframe,
            )

        aggs = self._aggregate(records, "_period", "period")
        # Sort ascending by period label
        aggs.sort(key=lambda a: a.group_key)
        return aggs

    # ------------------------------------------------------------------
    # Public API - Pareto / ABC Analysis
    # ------------------------------------------------------------------

    def pareto_analysis(
        self,
        records: List[Dict[str, Any]],
        threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """Perform ABC / Pareto analysis on spend records.

        Classifies vendors into A (top contributors to threshold),
        B (next 15%), and C (remaining) classes.

        Args:
            records: List of spend record dicts.
            threshold: Cumulative share threshold for A-class (default 0.8).

        Returns:
            Dict with class counts, percentages, and vendor lists.
        """
        start = time.monotonic()
        threshold = threshold or self._pareto_threshold

        # Aggregate by vendor
        vendor_spend: Dict[str, float] = defaultdict(float)
        vendor_emissions: Dict[str, float] = defaultdict(float)
        for rec in records:
            vendor = str(rec.get("vendor_name", "Unknown"))
            vendor_spend[vendor] += float(rec.get("amount_usd", 0) or 0)
            vendor_emissions[vendor] += float(rec.get("emissions_kgco2e", 0) or 0)

        total_spend = sum(vendor_spend.values())
        if total_spend == 0:
            return self._empty_pareto()

        # Sort vendors by spend descending
        sorted_vendors = sorted(
            vendor_spend.items(), key=lambda x: x[1], reverse=True,
        )

        a_class: List[Dict[str, Any]] = []
        b_class: List[Dict[str, Any]] = []
        c_class: List[Dict[str, Any]] = []
        cumulative = 0.0

        for vendor, spend in sorted_vendors:
            share = spend / total_spend
            cumulative += share
            entry = {
                "vendor": vendor,
                "spend_usd": round(spend, 2),
                "emissions_kgco2e": round(vendor_emissions.get(vendor, 0), 4),
                "share": round(share, 4),
                "cumulative": round(cumulative, 4),
            }

            if cumulative <= threshold:
                a_class.append(entry)
            elif cumulative <= threshold + 0.15:
                b_class.append(entry)
            else:
                c_class.append(entry)

        total_vendors = len(sorted_vendors)
        provenance = self._compute_provenance(
            "pareto", total_spend, len(records), total_vendors,
        )

        self._record_analysis("pareto", len(records))

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Pareto analysis: %d vendors, A=%d B=%d C=%d (%.1f ms)",
            total_vendors, len(a_class), len(b_class), len(c_class), elapsed,
        )

        return {
            "total_spend_usd": round(total_spend, 2),
            "total_vendors": total_vendors,
            "threshold": threshold,
            "a_class_count": len(a_class),
            "a_class_spend_usd": round(sum(v["spend_usd"] for v in a_class), 2),
            "a_class_percentage": round(len(a_class) / total_vendors * 100, 2) if total_vendors else 0,
            "a_class": a_class,
            "b_class_count": len(b_class),
            "b_class_spend_usd": round(sum(v["spend_usd"] for v in b_class), 2),
            "b_class": b_class,
            "c_class_count": len(c_class),
            "c_class_spend_usd": round(sum(v["spend_usd"] for v in c_class), 2),
            "c_class": c_class,
            "provenance_hash": provenance,
        }

    # ------------------------------------------------------------------
    # Public API - Hotspot identification
    # ------------------------------------------------------------------

    def identify_hotspots(
        self,
        records: List[Dict[str, Any]],
        top_n: int = 10,
    ) -> List[HotspotResult]:
        """Identify the top emissions hotspots.

        Groups by Scope 3 category and returns the highest-emitting
        categories ranked by total emissions.

        Args:
            records: List of spend record dicts.
            top_n: Number of hotspots to return.

        Returns:
            List of HotspotResult objects ranked by emissions.
        """
        start = time.monotonic()
        top_n = top_n or self._default_top_n

        # Aggregate by category
        groups: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"spend": 0.0, "emissions": 0.0, "count": 0},
        )
        for rec in records:
            key = str(rec.get("scope3_category", rec.get("category_name", "Unknown")))
            groups[key]["spend"] += float(rec.get("amount_usd", 0) or 0)
            groups[key]["emissions"] += float(rec.get("emissions_kgco2e", 0) or 0)
            groups[key]["count"] += 1

        total_emissions = sum(g["emissions"] for g in groups.values())
        if total_emissions == 0:
            return []

        # Sort by emissions descending
        sorted_groups = sorted(
            groups.items(), key=lambda x: x[1]["emissions"], reverse=True,
        )

        hotspots: List[HotspotResult] = []
        cumulative = 0.0
        for rank, (key, data) in enumerate(sorted_groups[:top_n], 1):
            share = data["emissions"] / total_emissions
            cumulative += share
            intensity = (
                data["emissions"] / data["spend"]
                if data["spend"] > 0 else 0.0
            )

            provenance = self._compute_provenance(
                f"hotspot-{rank}", data["emissions"], data["count"], rank,
            )

            hotspots.append(HotspotResult(
                rank=rank,
                group_key=key,
                group_type="scope3_category",
                total_spend_usd=round(data["spend"], 2),
                total_emissions_kgco2e=round(data["emissions"], 4),
                share_of_emissions=round(share, 4),
                cumulative_share=round(cumulative, 4),
                intensity_kgco2e_per_usd=round(intensity, 6),
                record_count=data["count"],
                provenance_hash=provenance,
            ))

        self._record_analysis("hotspot", len(records))
        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Identified %d hotspots from %d records (%.1f ms)",
            len(hotspots), len(records), elapsed,
        )
        return hotspots

    # ------------------------------------------------------------------
    # Public API - Concentration metrics
    # ------------------------------------------------------------------

    def calculate_concentration(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate supplier concentration metrics.

        Computes HHI (Herfindahl-Hirschman Index) and concentration
        ratios (CR4, CR8, CR20) based on vendor spend shares.

        HHI interpretation:
        - < 1500: Unconcentrated
        - 1500-2500: Moderately concentrated
        - > 2500: Highly concentrated

        Args:
            records: List of spend record dicts.

        Returns:
            Dict with HHI, CR4, CR8, CR20, and interpretation.
        """
        vendor_spend: Dict[str, float] = defaultdict(float)
        for rec in records:
            vendor = str(rec.get("vendor_name", "Unknown"))
            vendor_spend[vendor] += float(rec.get("amount_usd", 0) or 0)

        total_spend = sum(vendor_spend.values())
        if total_spend == 0:
            return {"hhi": 0, "cr4": 0.0, "cr8": 0.0, "cr20": 0.0, "interpretation": "no_data"}

        # Calculate market shares (0-100 scale for HHI)
        shares = sorted(
            [(v / total_spend) * 100 for v in vendor_spend.values()],
            reverse=True,
        )

        # HHI = sum of squared shares (0-10000 scale)
        hhi = round(sum(s * s for s in shares), 2)

        # Concentration ratios
        cr4 = round(sum(shares[:4]) / 100, 4) if len(shares) >= 4 else round(sum(shares) / 100, 4)
        cr8 = round(sum(shares[:8]) / 100, 4) if len(shares) >= 8 else round(sum(shares) / 100, 4)
        cr20 = round(sum(shares[:20]) / 100, 4) if len(shares) >= 20 else round(sum(shares) / 100, 4)

        # Interpretation
        if hhi < 1500:
            interpretation = "unconcentrated"
        elif hhi < 2500:
            interpretation = "moderately_concentrated"
        else:
            interpretation = "highly_concentrated"

        provenance = self._compute_provenance(
            "concentration", total_spend, len(vendor_spend), hhi,
        )

        self._record_analysis("concentration", len(records))

        return {
            "hhi": hhi,
            "hhi_max": 10000,
            "cr4": cr4,
            "cr8": cr8,
            "cr20": cr20,
            "vendor_count": len(vendor_spend),
            "total_spend_usd": round(total_spend, 2),
            "interpretation": interpretation,
            "provenance_hash": provenance,
        }

    # ------------------------------------------------------------------
    # Public API - Intensity
    # ------------------------------------------------------------------

    def calculate_intensity(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate emissions intensity by category.

        Intensity = total_emissions / total_spend for each category.

        Args:
            records: List of spend record dicts.

        Returns:
            Dict with per-category and overall intensity metrics.
        """
        categories: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"spend": 0.0, "emissions": 0.0, "count": 0},
        )
        for rec in records:
            cat = str(rec.get("scope3_category", rec.get("category_name", "Unknown")))
            categories[cat]["spend"] += float(rec.get("amount_usd", 0) or 0)
            categories[cat]["emissions"] += float(rec.get("emissions_kgco2e", 0) or 0)
            categories[cat]["count"] += 1

        total_spend = sum(c["spend"] for c in categories.values())
        total_emissions = sum(c["emissions"] for c in categories.values())
        overall_intensity = (
            round(total_emissions / total_spend, 6) if total_spend > 0 else 0.0
        )

        by_category: List[Dict[str, Any]] = []
        for cat, data in sorted(categories.items()):
            intensity = (
                round(data["emissions"] / data["spend"], 6)
                if data["spend"] > 0 else 0.0
            )
            by_category.append({
                "category": cat,
                "spend_usd": round(data["spend"], 2),
                "emissions_kgco2e": round(data["emissions"], 4),
                "intensity_kgco2e_per_usd": intensity,
                "record_count": int(data["count"]),
            })

        # Sort by intensity descending
        by_category.sort(key=lambda x: x["intensity_kgco2e_per_usd"], reverse=True)

        self._record_analysis("intensity", len(records))

        return {
            "overall_intensity_kgco2e_per_usd": overall_intensity,
            "total_spend_usd": round(total_spend, 2),
            "total_emissions_kgco2e": round(total_emissions, 4),
            "category_count": len(categories),
            "by_category": by_category,
            "provenance_hash": self._compute_provenance(
                "intensity", total_spend, total_emissions, len(categories),
            ),
        }

    # ------------------------------------------------------------------
    # Public API - Trend analysis
    # ------------------------------------------------------------------

    def trend_analysis(
        self,
        records: List[Dict[str, Any]],
        periods: int = 4,
    ) -> List[TrendDataPoint]:
        """Perform period-over-period trend analysis.

        Groups records by period (quarterly by default) and calculates
        spend and emissions changes between consecutive periods.

        Args:
            records: List of spend record dicts with ``transaction_date``.
            periods: Number of recent periods to include.

        Returns:
            List of TrendDataPoint objects in chronological order.
        """
        start = time.monotonic()

        # Group by quarterly period
        period_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"spend": 0.0, "emissions": 0.0, "count": 0},
        )
        for rec in records:
            period = self._assign_period(
                rec.get("transaction_date", ""), "quarterly",
            )
            period_data[period]["spend"] += float(rec.get("amount_usd", 0) or 0)
            period_data[period]["emissions"] += float(rec.get("emissions_kgco2e", 0) or 0)
            period_data[period]["count"] += 1

        # Sort periods and take the most recent N
        sorted_periods = sorted(period_data.keys())
        if periods > 0:
            sorted_periods = sorted_periods[-periods:]

        trend: List[TrendDataPoint] = []
        for idx, period in enumerate(sorted_periods):
            data = period_data[period]
            spend = data["spend"]
            emissions = data["emissions"]
            intensity = round(emissions / spend, 6) if spend > 0 else 0.0

            spend_change: Optional[float] = None
            emissions_change: Optional[float] = None
            if idx > 0:
                prev_data = period_data[sorted_periods[idx - 1]]
                if prev_data["spend"] > 0:
                    spend_change = round(
                        (spend - prev_data["spend"]) / prev_data["spend"] * 100, 2,
                    )
                if prev_data["emissions"] > 0:
                    emissions_change = round(
                        (emissions - prev_data["emissions"]) / prev_data["emissions"] * 100, 2,
                    )

            provenance = self._compute_provenance(
                f"trend-{period}", spend, emissions, idx,
            )

            trend.append(TrendDataPoint(
                period=period,
                period_index=idx,
                total_spend_usd=round(spend, 2),
                total_emissions_kgco2e=round(emissions, 4),
                record_count=data["count"],
                spend_change_pct=spend_change,
                emissions_change_pct=emissions_change,
                intensity_kgco2e_per_usd=intensity,
                provenance_hash=provenance,
            ))

        self._record_analysis("trend", len(records))
        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Trend analysis: %d periods from %d records (%.1f ms)",
            len(trend), len(records), elapsed,
        )
        return trend

    # ------------------------------------------------------------------
    # Public API - Variance analysis
    # ------------------------------------------------------------------

    def variance_analysis(
        self,
        records: List[Dict[str, Any]],
        baseline_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare current period to a baseline period.

        Calculates absolute and percentage variances in spend,
        emissions, intensity, and vendor count.

        Args:
            records: Current period records.
            baseline_records: Baseline period records.

        Returns:
            Dict with variances and category-level comparisons.
        """
        start = time.monotonic()

        current = self._summarise_records(records)
        baseline = self._summarise_records(baseline_records)

        spend_var = current["total_spend"] - baseline["total_spend"]
        emissions_var = current["total_emissions"] - baseline["total_emissions"]

        spend_pct = (
            round(spend_var / baseline["total_spend"] * 100, 2)
            if baseline["total_spend"] > 0 else 0.0
        )
        emissions_pct = (
            round(emissions_var / baseline["total_emissions"] * 100, 2)
            if baseline["total_emissions"] > 0 else 0.0
        )

        # Category-level variances
        all_cats = set(current["by_category"].keys()) | set(baseline["by_category"].keys())
        category_variances: List[Dict[str, Any]] = []
        for cat in sorted(all_cats):
            cur = current["by_category"].get(cat, {"spend": 0.0, "emissions": 0.0})
            base = baseline["by_category"].get(cat, {"spend": 0.0, "emissions": 0.0})
            category_variances.append({
                "category": cat,
                "current_spend_usd": round(cur["spend"], 2),
                "baseline_spend_usd": round(base["spend"], 2),
                "spend_variance_usd": round(cur["spend"] - base["spend"], 2),
                "current_emissions_kgco2e": round(cur["emissions"], 4),
                "baseline_emissions_kgco2e": round(base["emissions"], 4),
                "emissions_variance_kgco2e": round(cur["emissions"] - base["emissions"], 4),
            })

        provenance = self._compute_provenance(
            "variance", spend_var, emissions_var, len(category_variances),
        )
        self._record_analysis("variance", len(records) + len(baseline_records))

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Variance analysis: spend=%.2f%%, emissions=%.2f%% (%.1f ms)",
            spend_pct, emissions_pct, elapsed,
        )

        return {
            "current_spend_usd": round(current["total_spend"], 2),
            "baseline_spend_usd": round(baseline["total_spend"], 2),
            "spend_variance_usd": round(spend_var, 2),
            "spend_variance_pct": spend_pct,
            "current_emissions_kgco2e": round(current["total_emissions"], 4),
            "baseline_emissions_kgco2e": round(baseline["total_emissions"], 4),
            "emissions_variance_kgco2e": round(emissions_var, 4),
            "emissions_variance_pct": emissions_pct,
            "current_record_count": len(records),
            "baseline_record_count": len(baseline_records),
            "category_variances": category_variances,
            "provenance_hash": provenance,
        }

    # ------------------------------------------------------------------
    # Public API - Benchmarking
    # ------------------------------------------------------------------

    def benchmark(
        self,
        records: List[Dict[str, Any]],
        industry: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Benchmark spend emissions against industry averages.

        Args:
            records: List of spend record dicts.
            industry: Industry identifier (defaults to configured industry).

        Returns:
            Dict with benchmark comparison and gap analysis.
        """
        industry = (industry or self._default_industry).lower()
        bench = _INDUSTRY_BENCHMARKS.get(industry)
        if bench is None:
            logger.warning("No benchmark data for industry: %s", industry)
            bench = _INDUSTRY_BENCHMARKS.get("manufacturing", {})

        total_spend = sum(float(r.get("amount_usd", 0) or 0) for r in records)
        total_emissions = sum(float(r.get("emissions_kgco2e", 0) or 0) for r in records)
        actual_intensity = (
            round(total_emissions / total_spend, 6) if total_spend > 0 else 0.0
        )
        bench_intensity = bench.get("intensity_kgco2e_per_usd", 0.25)

        gap = round(actual_intensity - bench_intensity, 6)
        gap_pct = (
            round(gap / bench_intensity * 100, 2)
            if bench_intensity > 0 else 0.0
        )

        performance = "below_average"
        if actual_intensity <= bench_intensity * 0.8:
            performance = "leader"
        elif actual_intensity <= bench_intensity:
            performance = "above_average"
        elif actual_intensity <= bench_intensity * 1.2:
            performance = "below_average"
        else:
            performance = "laggard"

        provenance = self._compute_provenance(
            "benchmark", total_spend, total_emissions, actual_intensity,
        )
        self._record_analysis("benchmark", len(records))

        return {
            "industry": industry,
            "actual_intensity_kgco2e_per_usd": actual_intensity,
            "benchmark_intensity_kgco2e_per_usd": bench_intensity,
            "intensity_gap": gap,
            "intensity_gap_pct": gap_pct,
            "performance_rating": performance,
            "total_spend_usd": round(total_spend, 2),
            "total_emissions_kgco2e": round(total_emissions, 4),
            "record_count": len(records),
            "benchmark_source": _INDUSTRY_BENCHMARKS,
            "provenance_hash": provenance,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return cumulative analytics statistics.

        Returns:
            Dictionary with analysis counters and breakdowns.
        """
        with self._lock:
            stats = dict(self._stats)
            stats["by_analysis_type"] = dict(self._stats["by_analysis_type"])
        stats["industry_benchmarks_available"] = list(_INDUSTRY_BENCHMARKS.keys())
        return stats

    # ------------------------------------------------------------------
    # Internal - Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        records: List[Dict[str, Any]],
        group_by: str,
        group_type: str,
    ) -> List[SpendAggregate]:
        """Aggregate records by a specified field.

        Args:
            records: List of records.
            group_by: Field to group by.
            group_type: Group type label.

        Returns:
            List of SpendAggregate objects sorted by emissions descending.
        """
        start = time.monotonic()

        groups: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"spend": 0.0, "emissions": 0.0, "count": 0},
        )
        for rec in records:
            key = str(rec.get(group_by, "Unknown"))
            groups[key]["spend"] += float(rec.get("amount_usd", 0) or 0)
            groups[key]["emissions"] += float(rec.get("emissions_kgco2e", 0) or 0)
            groups[key]["count"] += 1

        total_spend = sum(g["spend"] for g in groups.values())
        total_emissions = sum(g["emissions"] for g in groups.values())

        result: List[SpendAggregate] = []
        for key, data in groups.items():
            spend = data["spend"]
            emissions = data["emissions"]
            count = data["count"]
            intensity = round(emissions / spend, 6) if spend > 0 else 0.0

            provenance = self._compute_provenance(
                f"agg-{group_type}-{key}", spend, emissions, count,
            )

            result.append(SpendAggregate(
                group_key=key,
                group_type=group_type,
                record_count=count,
                total_spend_usd=round(spend, 2),
                total_emissions_kgco2e=round(emissions, 4),
                total_emissions_tco2e=round(emissions / 1000, 6),
                avg_spend_usd=round(spend / count, 2) if count > 0 else 0.0,
                avg_emissions_kgco2e=round(emissions / count, 4) if count > 0 else 0.0,
                intensity_kgco2e_per_usd=intensity,
                share_of_spend=round(spend / total_spend, 4) if total_spend > 0 else 0.0,
                share_of_emissions=round(emissions / total_emissions, 4) if total_emissions > 0 else 0.0,
                provenance_hash=provenance,
            ))

        # Sort by emissions descending
        result.sort(key=lambda a: a.total_emissions_kgco2e, reverse=True)

        self._record_analysis(f"aggregate_{group_type}", len(records))

        elapsed = (time.monotonic() - start) * 1000
        logger.debug(
            "Aggregated %d records by %s into %d groups (%.1f ms)",
            len(records), group_by, len(result), elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Internal - Helpers
    # ------------------------------------------------------------------

    def _assign_period(self, date_str: Any, timeframe: str) -> str:
        """Assign a period label to a date string.

        Args:
            date_str: Date string or value.
            timeframe: "monthly", "quarterly", or "yearly".

        Returns:
            Period label (e.g. "2024-Q1", "2024-01", "2024").
        """
        if not date_str:
            return "Unknown"

        text = str(date_str).strip()[:10]
        try:
            dt = datetime.fromisoformat(text)
        except (ValueError, TypeError):
            return "Unknown"

        if timeframe == "monthly":
            return f"{dt.year}-{dt.month:02d}"
        elif timeframe == "quarterly":
            quarter = (dt.month - 1) // 3 + 1
            return f"{dt.year}-Q{quarter}"
        else:
            return str(dt.year)

    def _summarise_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a quick summary for variance analysis.

        Args:
            records: List of records.

        Returns:
            Dict with totals and by-category breakdown.
        """
        total_spend = 0.0
        total_emissions = 0.0
        by_cat: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"spend": 0.0, "emissions": 0.0},
        )
        for rec in records:
            spend = float(rec.get("amount_usd", 0) or 0)
            emissions = float(rec.get("emissions_kgco2e", 0) or 0)
            cat = str(rec.get("scope3_category", "Unknown"))
            total_spend += spend
            total_emissions += emissions
            by_cat[cat]["spend"] += spend
            by_cat[cat]["emissions"] += emissions

        return {
            "total_spend": total_spend,
            "total_emissions": total_emissions,
            "by_category": dict(by_cat),
        }

    def _empty_pareto(self) -> Dict[str, Any]:
        """Return an empty Pareto result.

        Returns:
            Dict with zero values.
        """
        return {
            "total_spend_usd": 0.0,
            "total_vendors": 0,
            "threshold": self._pareto_threshold,
            "a_class_count": 0,
            "a_class_spend_usd": 0.0,
            "a_class_percentage": 0.0,
            "a_class": [],
            "b_class_count": 0,
            "b_class_spend_usd": 0.0,
            "b_class": [],
            "c_class_count": 0,
            "c_class_spend_usd": 0.0,
            "c_class": [],
            "provenance_hash": "",
        }

    def _record_analysis(self, analysis_type: str, record_count: int) -> None:
        """Record an analysis in statistics.

        Args:
            analysis_type: Type of analysis performed.
            record_count: Number of records analysed.
        """
        with self._lock:
            self._stats["analyses_performed"] += 1
            self._stats["records_analysed"] += record_count
            at_counts = self._stats["by_analysis_type"]
            at_counts[analysis_type] = at_counts.get(analysis_type, 0) + 1

    def _compute_provenance(
        self,
        entity_id: str,
        value1: Any,
        value2: Any,
        value3: Any,
    ) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            entity_id: Entity identifier.
            value1: First value.
            value2: Second value.
            value3: Third value.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        payload = json.dumps({
            "entity_id": entity_id,
            "v1": str(value1),
            "v2": str(value2),
            "v3": str(value3),
            "timestamp": _utcnow().isoformat(),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
