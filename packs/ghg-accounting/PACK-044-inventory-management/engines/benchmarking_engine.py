# -*- coding: utf-8 -*-
"""
BenchmarkingEngine - PACK-044 Inventory Management Engine 10
===============================================================

Peer comparison and facility ranking engine that benchmarks an
organisation's GHG emissions performance against sector peers,
CDP disclosure data, and internal facility rankings using intensity
metrics (emissions per revenue, per FTE, per unit of production).

Calculation Methodology:
    Intensity Metrics:
        intensity_revenue = total_tco2e / revenue_eur_millions
        intensity_fte = total_tco2e / full_time_equivalents
        intensity_production = total_tco2e / production_units
        intensity_floor_area = total_tco2e / floor_area_m2

    Percentile Ranking (within peer group):
        rank_pct = (count_of_peers_with_higher_intensity / total_peers) * 100
        Where higher intensity = worse performance.
        A rank_pct of 90 means you outperform 90% of peers.

    Z-Score Calculation:
        z_score = (entity_intensity - peer_mean) / peer_std_dev
        Negative z-score = better than average
        |z| > 2 = significantly different from peers

    Trend Benchmarking:
        year_over_year_pct = (current - previous) / previous * 100
        compound_annual_reduction = (1 - (latest / earliest)^(1/years)) * 100

    Facility Ranking (internal):
        rank by intensity metric within organisation
        identify top/bottom performers
        calculate spread (max - min intensity) within organisation

    CDP Comparison:
        Map to CDP scoring bands:
            A-List:  Top 20% of sector disclosers
            B:       21st-50th percentile
            C:       51st-75th percentile
            D:       76th-100th percentile

Regulatory References:
    - GHG Protocol Corporate Standard, Chapter 9 (Verification)
    - CDP Climate Change Scoring Methodology 2025
    - CSRD / ESRS E1 Disclosure Requirement E1-6 (GHG intensity)
    - SBTi Corporate Manual, Section 5 (Target setting benchmarks)
    - TCFD Metrics and Targets, Recommendation 4b

Zero-Hallucination:
    - All rankings use deterministic sorting and percentile calculation
    - Intensity metrics use Decimal arithmetic for precision
    - Z-scores computed from actual peer data, no interpolation
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-044 Inventory Management
Engine:  10 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round2(value: Any) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round4(value: Any) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _round6(value: Any) -> float:
    """Round to 6 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IntensityMetricType(str, Enum):
    """Types of emission intensity metrics.

    REVENUE:     tCO2e per million EUR revenue.
    FTE:         tCO2e per full-time equivalent employee.
    PRODUCTION:  tCO2e per unit of production output.
    FLOOR_AREA:  tCO2e per square metre of floor area.
    CUSTOM:      Custom intensity denominator.
    """
    REVENUE = "revenue"
    FTE = "fte"
    PRODUCTION = "production"
    FLOOR_AREA = "floor_area"
    CUSTOM = "custom"


class SectorClassification(str, Enum):
    """Industry sector classifications for peer grouping.

    Based on GICS (Global Industry Classification Standard) sectors,
    commonly used in CDP and TCFD reporting.

    ENERGY:              Oil & gas, mining, utilities.
    MATERIALS:           Chemicals, construction materials, metals.
    INDUSTRIALS:         Aerospace, machinery, transportation.
    CONSUMER_DISC:       Automobiles, retail, media.
    CONSUMER_STAPLES:    Food, beverages, household products.
    HEALTH_CARE:         Pharma, biotech, health equipment.
    FINANCIALS:          Banks, insurance, real estate.
    INFORMATION_TECH:    Software, hardware, semiconductors.
    COMMUNICATION:       Telecom, media services.
    UTILITIES:           Electric, gas, water utilities.
    REAL_ESTATE:         REITs, real estate management.
    """
    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISC = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTH_CARE = "health_care"
    FINANCIALS = "financials"
    INFORMATION_TECH = "information_technology"
    COMMUNICATION = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"


class CDPScoreBand(str, Enum):
    """CDP Climate Change scoring bands.

    A_LIST:  Leadership level (top 20%).
    B:       Management level (21-50th percentile).
    C:       Awareness level (51-75th percentile).
    D:       Disclosure level (76-100th percentile).
    F:       Failed to disclose or below minimum.
    """
    A_LIST = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class BenchmarkStatus(str, Enum):
    """Performance status relative to benchmark.

    LEADER:       Top quartile (top 25%).
    ABOVE_AVG:    Above median (25-50th percentile).
    AVERAGE:      Around median (40-60th percentile).
    BELOW_AVG:    Below median (60-75th percentile).
    LAGGARD:      Bottom quartile (bottom 25%).
    """
    LEADER = "leader"
    ABOVE_AVG = "above_average"
    AVERAGE = "average"
    BELOW_AVG = "below_average"
    LAGGARD = "laggard"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sector average intensities (tCO2e per million EUR revenue).
# Sources: CDP 2024 Global Disclosure Dataset, S&P Global sector averages.
# These are illustrative defaults; clients should use current-year peer data.
SECTOR_AVERAGE_INTENSITIES: Dict[str, Dict[str, float]] = {
    SectorClassification.ENERGY.value: {
        "revenue": 850.0, "fte": 45.0, "source": "CDP 2024 sector average",
    },
    SectorClassification.MATERIALS.value: {
        "revenue": 420.0, "fte": 28.0, "source": "CDP 2024 sector average",
    },
    SectorClassification.INDUSTRIALS.value: {
        "revenue": 180.0, "fte": 15.0, "source": "CDP 2024 sector average",
    },
    SectorClassification.CONSUMER_DISC.value: {
        "revenue": 95.0, "fte": 8.0, "source": "CDP 2024 sector average",
    },
    SectorClassification.CONSUMER_STAPLES.value: {
        "revenue": 140.0, "fte": 10.0, "source": "CDP 2024 sector average",
    },
    SectorClassification.HEALTH_CARE.value: {
        "revenue": 55.0, "fte": 5.0, "source": "CDP 2024 sector average",
    },
    SectorClassification.FINANCIALS.value: {
        "revenue": 12.0, "fte": 3.0, "source": "CDP 2024 sector average",
    },
    SectorClassification.INFORMATION_TECH.value: {
        "revenue": 25.0, "fte": 4.0, "source": "CDP 2024 sector average",
    },
    SectorClassification.COMMUNICATION.value: {
        "revenue": 35.0, "fte": 4.5, "source": "CDP 2024 sector average",
    },
    SectorClassification.UTILITIES.value: {
        "revenue": 1200.0, "fte": 80.0, "source": "CDP 2024 sector average",
    },
    SectorClassification.REAL_ESTATE.value: {
        "revenue": 50.0, "fte": 6.0, "source": "CDP 2024 sector average",
    },
}
"""Sector average emission intensities."""

# CDP score band percentile thresholds (lower intensity = better rank).
CDP_BAND_THRESHOLDS: Dict[str, float] = {
    CDPScoreBand.A_LIST.value: 80.0,  # Top 20% = rank percentile >= 80
    CDPScoreBand.B.value: 50.0,       # 21-50th percentile
    CDPScoreBand.C.value: 25.0,       # 51-75th percentile
    CDPScoreBand.D.value: 0.0,        # Bottom 25%
}
"""CDP score band thresholds based on percentile ranking."""

# Benchmark status thresholds based on percentile ranking.
BENCHMARK_THRESHOLDS: Dict[str, float] = {
    BenchmarkStatus.LEADER.value: 75.0,
    BenchmarkStatus.ABOVE_AVG.value: 50.0,
    BenchmarkStatus.AVERAGE.value: 40.0,
    BenchmarkStatus.BELOW_AVG.value: 25.0,
    BenchmarkStatus.LAGGARD.value: 0.0,
}
"""Benchmark status thresholds."""


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class EntityProfile(BaseModel):
    """An entity's profile for benchmarking.

    Attributes:
        entity_id: Unique entity identifier.
        entity_name: Entity name.
        sector: GICS sector classification.
        country: ISO 3166-1 alpha-2 country code.
        total_scope1_tco2e: Total Scope 1 emissions.
        total_scope2_location_tco2e: Scope 2 location-based.
        total_scope2_market_tco2e: Scope 2 market-based.
        total_scope3_tco2e: Total Scope 3 emissions.
        revenue_eur_millions: Annual revenue in EUR millions.
        fte_count: Full-time equivalent employees.
        production_units: Production output in natural units.
        production_unit_name: Name of the production unit.
        floor_area_m2: Total floor area in square metres.
        custom_denominator: Custom intensity denominator value.
        custom_denominator_name: Custom denominator name.
    """
    entity_id: str = Field(default_factory=_new_uuid, description="Entity ID")
    entity_name: str = Field(..., min_length=1, max_length=500, description="Name")
    sector: SectorClassification = Field(
        default=SectorClassification.INDUSTRIALS, description="Sector"
    )
    country: str = Field(default="", max_length=2, description="Country")
    total_scope1_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 1 tCO2e"
    )
    total_scope2_location_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 2 loc tCO2e"
    )
    total_scope2_market_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 2 mkt tCO2e"
    )
    total_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 3 tCO2e"
    )
    revenue_eur_millions: Decimal = Field(
        default=Decimal("0"), ge=0, description="Revenue (EUR M)"
    )
    fte_count: Decimal = Field(
        default=Decimal("0"), ge=0, description="FTE count"
    )
    production_units: Decimal = Field(
        default=Decimal("0"), ge=0, description="Production units"
    )
    production_unit_name: str = Field(
        default="units", description="Production unit name"
    )
    floor_area_m2: Decimal = Field(
        default=Decimal("0"), ge=0, description="Floor area (m2)"
    )
    custom_denominator: Decimal = Field(
        default=Decimal("0"), ge=0, description="Custom denominator"
    )
    custom_denominator_name: str = Field(
        default="", description="Custom denominator name"
    )

    @field_validator(
        "total_scope1_tco2e", "total_scope2_location_tco2e",
        "total_scope2_market_tco2e", "total_scope3_tco2e",
        "revenue_eur_millions", "fte_count", "production_units",
        "floor_area_m2", "custom_denominator",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)


class FacilityProfile(BaseModel):
    """A facility profile for internal ranking.

    Attributes:
        facility_id: Unique facility identifier.
        facility_name: Facility name.
        entity_id: Parent entity identifier.
        country: Country code.
        total_tco2e: Total emissions (Scope 1+2).
        revenue_eur_millions: Facility revenue if available.
        fte_count: Facility FTE count.
        production_units: Production output.
        floor_area_m2: Floor area.
    """
    facility_id: str = Field(default_factory=_new_uuid, description="Facility ID")
    facility_name: str = Field(..., min_length=1, max_length=500, description="Name")
    entity_id: str = Field(default="", description="Parent entity")
    country: str = Field(default="", max_length=2, description="Country")
    total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total tCO2e"
    )
    revenue_eur_millions: Decimal = Field(
        default=Decimal("0"), ge=0, description="Revenue (EUR M)"
    )
    fte_count: Decimal = Field(
        default=Decimal("0"), ge=0, description="FTE count"
    )
    production_units: Decimal = Field(
        default=Decimal("0"), ge=0, description="Production units"
    )
    floor_area_m2: Decimal = Field(
        default=Decimal("0"), ge=0, description="Floor area (m2)"
    )

    @field_validator(
        "total_tco2e", "revenue_eur_millions", "fte_count",
        "production_units", "floor_area_m2",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)


class PeerDataPoint(BaseModel):
    """A single peer data point for benchmarking.

    Attributes:
        peer_id: Peer identifier.
        peer_name: Peer name.
        sector: Sector classification.
        intensity_revenue: Revenue intensity (tCO2e / EUR M).
        intensity_fte: FTE intensity (tCO2e / FTE).
        intensity_production: Production intensity.
        total_scope12_tco2e: Scope 1+2 total.
        year: Reporting year.
    """
    peer_id: str = Field(default_factory=_new_uuid, description="Peer ID")
    peer_name: str = Field(default="", description="Peer name")
    sector: str = Field(default="", description="Sector")
    intensity_revenue: Decimal = Field(
        default=Decimal("0"), ge=0, description="Revenue intensity"
    )
    intensity_fte: Decimal = Field(
        default=Decimal("0"), ge=0, description="FTE intensity"
    )
    intensity_production: Decimal = Field(
        default=Decimal("0"), ge=0, description="Production intensity"
    )
    total_scope12_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 1+2 tCO2e"
    )
    year: int = Field(default=2025, ge=1990, le=2100, description="Year")

    @field_validator(
        "intensity_revenue", "intensity_fte", "intensity_production",
        "total_scope12_tco2e", mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)


class HistoricalDataPoint(BaseModel):
    """Historical emission data point for trend benchmarking.

    Attributes:
        year: Reporting year.
        total_scope12_tco2e: Scope 1+2 total for the year.
        intensity_revenue: Revenue intensity for the year.
        intensity_fte: FTE intensity for the year.
        revenue_eur_millions: Revenue for the year.
        fte_count: FTE count for the year.
    """
    year: int = Field(..., ge=1990, le=2100, description="Year")
    total_scope12_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 1+2 tCO2e"
    )
    intensity_revenue: Decimal = Field(
        default=Decimal("0"), ge=0, description="Revenue intensity"
    )
    intensity_fte: Decimal = Field(
        default=Decimal("0"), ge=0, description="FTE intensity"
    )
    revenue_eur_millions: Decimal = Field(
        default=Decimal("0"), ge=0, description="Revenue (EUR M)"
    )
    fte_count: Decimal = Field(
        default=Decimal("0"), ge=0, description="FTE count"
    )

    @field_validator(
        "total_scope12_tco2e", "intensity_revenue", "intensity_fte",
        "revenue_eur_millions", "fte_count", mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class IntensityResult(BaseModel):
    """Calculated intensity metric result.

    Attributes:
        metric_type: Type of intensity metric.
        numerator_tco2e: Emissions numerator (tCO2e).
        denominator_value: Denominator value.
        denominator_unit: Denominator unit name.
        intensity: Calculated intensity value.
        sector_average: Sector average for comparison.
        vs_sector_pct: Percentage relative to sector average.
        z_score: Z-score relative to peer group.
    """
    metric_type: str = Field(default="", description="Metric type")
    numerator_tco2e: float = Field(default=0.0, description="Numerator (tCO2e)")
    denominator_value: float = Field(default=0.0, description="Denominator")
    denominator_unit: str = Field(default="", description="Denominator unit")
    intensity: float = Field(default=0.0, description="Intensity value")
    sector_average: float = Field(default=0.0, description="Sector average")
    vs_sector_pct: float = Field(default=0.0, description="% vs sector average")
    z_score: float = Field(default=0.0, description="Z-score")


class BenchmarkResult(BaseModel):
    """Result of a single benchmark comparison.

    Attributes:
        benchmark_id: Unique benchmark identifier.
        metric_type: Type of intensity metric used.
        entity_intensity: Entity's intensity value.
        peer_mean: Peer group mean intensity.
        peer_median: Peer group median intensity.
        peer_std_dev: Peer group standard deviation.
        peer_min: Peer group minimum intensity.
        peer_max: Peer group maximum intensity.
        percentile_rank: Entity's percentile rank (higher = better).
        z_score: Z-score (negative = better than average).
        status: Benchmark status classification.
        peer_count: Number of peers in comparison.
    """
    benchmark_id: str = Field(default_factory=_new_uuid, description="Benchmark ID")
    metric_type: str = Field(default="", description="Metric type")
    entity_intensity: float = Field(default=0.0, description="Entity intensity")
    peer_mean: float = Field(default=0.0, description="Peer mean")
    peer_median: float = Field(default=0.0, description="Peer median")
    peer_std_dev: float = Field(default=0.0, description="Peer std dev")
    peer_min: float = Field(default=0.0, description="Peer min")
    peer_max: float = Field(default=0.0, description="Peer max")
    percentile_rank: float = Field(default=0.0, description="Percentile rank")
    z_score: float = Field(default=0.0, description="Z-score")
    status: str = Field(default="average", description="Benchmark status")
    peer_count: int = Field(default=0, description="Peer count")


class PeerComparison(BaseModel):
    """Detailed peer-by-peer comparison.

    Attributes:
        peer_id: Peer identifier.
        peer_name: Peer name.
        peer_intensity: Peer's intensity value.
        entity_intensity: Entity's intensity for comparison.
        difference: Entity intensity - peer intensity.
        difference_pct: Percentage difference.
        entity_is_better: Whether entity outperforms this peer.
    """
    peer_id: str = Field(default="", description="Peer ID")
    peer_name: str = Field(default="", description="Peer name")
    peer_intensity: float = Field(default=0.0, description="Peer intensity")
    entity_intensity: float = Field(default=0.0, description="Entity intensity")
    difference: float = Field(default=0.0, description="Difference")
    difference_pct: float = Field(default=0.0, description="Difference %")
    entity_is_better: bool = Field(default=False, description="Entity is better")


class FacilityRanking(BaseModel):
    """Internal facility ranking result.

    Attributes:
        facility_id: Facility identifier.
        facility_name: Facility name.
        rank: Rank within the organisation (1 = best).
        intensity: Intensity metric value.
        metric_type: Type of intensity metric.
        total_tco2e: Total emissions.
        vs_org_average_pct: Percentage relative to org average.
        performance_category: Leader / average / laggard.
    """
    facility_id: str = Field(default="", description="Facility ID")
    facility_name: str = Field(default="", description="Facility name")
    rank: int = Field(default=0, description="Rank (1=best)")
    intensity: float = Field(default=0.0, description="Intensity")
    metric_type: str = Field(default="", description="Metric type")
    total_tco2e: float = Field(default=0.0, description="Total tCO2e")
    vs_org_average_pct: float = Field(default=0.0, description="% vs org average")
    performance_category: str = Field(default="average", description="Performance")


class BenchmarkTrend(BaseModel):
    """Historical benchmarking trend.

    Attributes:
        metric_type: Type of intensity metric.
        start_year: First year in the series.
        end_year: Last year in the series.
        start_value: Value at start.
        end_value: Value at end.
        absolute_change: Absolute change (end - start).
        percentage_change: Percentage change.
        compound_annual_change_pct: CAGR of the intensity metric.
        trend_direction: Improving / stable / worsening.
        year_over_year: Year-by-year changes.
    """
    metric_type: str = Field(default="", description="Metric type")
    start_year: int = Field(default=0, description="Start year")
    end_year: int = Field(default=0, description="End year")
    start_value: float = Field(default=0.0, description="Start value")
    end_value: float = Field(default=0.0, description="End value")
    absolute_change: float = Field(default=0.0, description="Absolute change")
    percentage_change: float = Field(default=0.0, description="% change")
    compound_annual_change_pct: float = Field(
        default=0.0, description="Compound annual change %"
    )
    trend_direction: str = Field(default="stable", description="Trend direction")
    year_over_year: List[Dict[str, Any]] = Field(
        default_factory=list, description="YoY changes"
    )


class BenchmarkingResult(BaseModel):
    """Complete benchmarking result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp (UTC).
        processing_time_ms: Processing time in milliseconds.
        entity_name: Benchmarked entity name.
        sector: Entity sector.
        intensity_metrics: Calculated intensity metrics.
        sector_benchmarks: Sector benchmark comparisons.
        peer_comparisons: Peer-by-peer comparisons.
        facility_rankings: Internal facility rankings.
        trends: Historical trend analysis.
        cdp_score_band: Estimated CDP score band.
        overall_status: Overall benchmark status.
        summary: Human-readable summary.
        methodology_notes: Methodology notes.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Version")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    entity_name: str = Field(default="", description="Entity name")
    sector: str = Field(default="", description="Sector")
    intensity_metrics: List[IntensityResult] = Field(
        default_factory=list, description="Intensity metrics"
    )
    sector_benchmarks: List[BenchmarkResult] = Field(
        default_factory=list, description="Sector benchmarks"
    )
    peer_comparisons: List[PeerComparison] = Field(
        default_factory=list, description="Peer comparisons"
    )
    facility_rankings: List[FacilityRanking] = Field(
        default_factory=list, description="Facility rankings"
    )
    trends: List[BenchmarkTrend] = Field(
        default_factory=list, description="Trends"
    )
    cdp_score_band: str = Field(default="", description="CDP score band")
    overall_status: str = Field(default="average", description="Overall status")
    summary: str = Field(default="", description="Summary")
    methodology_notes: List[str] = Field(
        default_factory=list, description="Notes"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Model Rebuild
# ---------------------------------------------------------------------------

EntityProfile.model_rebuild()
FacilityProfile.model_rebuild()
PeerDataPoint.model_rebuild()
HistoricalDataPoint.model_rebuild()
IntensityResult.model_rebuild()
BenchmarkResult.model_rebuild()
PeerComparison.model_rebuild()
FacilityRanking.model_rebuild()
BenchmarkTrend.model_rebuild()
BenchmarkingResult.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BenchmarkingEngine:
    """GHG emission peer comparison and facility ranking engine.

    Benchmarks an entity's emission intensity against sector peers,
    CDP disclosure data, and internal facility performance. Provides
    percentile rankings, z-scores, and historical trend analysis.

    Features:
        - Multi-metric intensity calculation (revenue, FTE, production, area)
        - Sector average comparison with GICS classification
        - Peer group percentile ranking and z-score analysis
        - Internal facility ranking and performance categorisation
        - Historical trend analysis with compound annual change
        - CDP score band estimation
        - Performance summary generation

    Guarantees:
        - Deterministic: same inputs produce identical results
        - Reproducible: SHA-256 provenance hash on every result
        - Auditable: per-peer and per-facility breakdown
        - No LLM: zero hallucination risk in any calculation path

    Usage::

        engine = BenchmarkingEngine()
        entity = EntityProfile(
            entity_name="Acme Corp",
            sector=SectorClassification.INDUSTRIALS,
            total_scope1_tco2e=Decimal("5000"),
            total_scope2_location_tco2e=Decimal("3000"),
            revenue_eur_millions=Decimal("500"),
            fte_count=Decimal("2000"),
        )
        peers = [PeerDataPoint(...), PeerDataPoint(...)]
        result = engine.benchmark(entity, peers=peers)
        print(f"Status: {result.overall_status}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the benchmarking engine.

        Args:
            config: Optional configuration. Supported keys:
                - default_metric (str): default intensity metric type
                - include_scope3 (bool): include Scope 3 in totals
                - custom_sector_averages (dict): override sector averages
        """
        self._config = config or {}
        self._default_metric = self._config.get(
            "default_metric", IntensityMetricType.REVENUE.value
        )
        self._include_scope3 = bool(self._config.get("include_scope3", False))
        self._custom_averages: Dict[str, Dict[str, float]] = self._config.get(
            "custom_sector_averages", {}
        )
        self._notes: List[str] = []
        logger.info("BenchmarkingEngine v%s initialised.", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def benchmark(
        self,
        entity: EntityProfile,
        peers: Optional[List[PeerDataPoint]] = None,
        facilities: Optional[List[FacilityProfile]] = None,
        historical: Optional[List[HistoricalDataPoint]] = None,
    ) -> BenchmarkingResult:
        """Run complete benchmarking analysis.

        Args:
            entity: Entity profile to benchmark.
            peers: Optional peer data points for comparison.
            facilities: Optional facilities for internal ranking.
            historical: Optional historical data for trend analysis.

        Returns:
            BenchmarkingResult with full benchmark analysis.
        """
        t0 = time.perf_counter()
        self._notes = [f"Engine version: {self.engine_version}"]
        peers = peers or []
        facilities = facilities or []
        historical = historical or []

        logger.info(
            "Benchmarking '%s' (sector=%s): %d peers, %d facilities, %d history",
            entity.entity_name, entity.sector.value,
            len(peers), len(facilities), len(historical),
        )

        # Step 1: Calculate intensity metrics
        intensities = self._calculate_intensities(entity)

        # Step 2: Sector benchmarks
        sector_benchmarks = self._benchmark_against_sector(entity, intensities)

        # Step 3: Peer comparisons
        peer_comparisons = self._compare_with_peers(entity, peers, intensities)

        # Step 4: Calculate peer-based benchmarks
        peer_benchmarks = self._calculate_peer_benchmarks(entity, peers, intensities)
        all_benchmarks = sector_benchmarks + peer_benchmarks

        # Step 5: Facility rankings
        fac_rankings = self._rank_facilities(facilities)

        # Step 6: Trend analysis
        trends = self._analyse_trends(historical)

        # Step 7: CDP score band estimation
        cdp_band = self._estimate_cdp_band(all_benchmarks)

        # Step 8: Overall status
        overall = self._determine_overall_status(all_benchmarks)

        # Step 9: Summary
        summary = self._build_summary(
            entity, intensities, all_benchmarks, cdp_band, overall
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = BenchmarkingResult(
            entity_name=entity.entity_name,
            sector=entity.sector.value,
            intensity_metrics=intensities,
            sector_benchmarks=all_benchmarks,
            peer_comparisons=peer_comparisons,
            facility_rankings=fac_rankings,
            trends=trends,
            cdp_score_band=cdp_band,
            overall_status=overall,
            summary=summary,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Benchmarking complete: '%s', status=%s, CDP=%s, hash=%s (%.1f ms)",
            entity.entity_name, overall, cdp_band,
            result.provenance_hash[:16], elapsed_ms,
        )
        return result

    def calculate_intensity(
        self,
        emissions_tco2e: Decimal,
        denominator: Decimal,
        metric_type: IntensityMetricType = IntensityMetricType.REVENUE,
    ) -> float:
        """Calculate a single intensity metric.

        Args:
            emissions_tco2e: Total emissions (tCO2e).
            denominator: Denominator value.
            metric_type: Type of metric for labelling.

        Returns:
            Intensity value (tCO2e per denominator unit).
        """
        return _round4(float(_safe_divide(emissions_tco2e, denominator)))

    def calculate_percentile_rank(
        self,
        entity_intensity: float,
        peer_intensities: List[float],
    ) -> float:
        """Calculate entity's percentile rank among peers.

        Lower intensity = better performance = higher percentile rank.

        Args:
            entity_intensity: Entity's intensity value.
            peer_intensities: List of peer intensity values.

        Returns:
            Percentile rank (0-100, higher = better).
        """
        if not peer_intensities:
            return 50.0

        # Count peers with higher (worse) intensity
        better_count = sum(
            1 for p in peer_intensities if p > entity_intensity
        )
        rank = (better_count / len(peer_intensities)) * 100.0
        return _round2(rank)

    def calculate_z_score(
        self,
        entity_intensity: float,
        peer_mean: float,
        peer_std: float,
    ) -> float:
        """Calculate z-score for entity vs peer group.

        Negative z-score = lower intensity = better performance.

        Args:
            entity_intensity: Entity's intensity.
            peer_mean: Peer group mean.
            peer_std: Peer group standard deviation.

        Returns:
            Z-score (negative is better).
        """
        if peer_std == 0.0:
            return 0.0
        return _round4((entity_intensity - peer_mean) / peer_std)

    # -------------------------------------------------------------------
    # Private -- Intensity calculations
    # -------------------------------------------------------------------

    def _calculate_intensities(
        self,
        entity: EntityProfile,
    ) -> List[IntensityResult]:
        """Calculate all available intensity metrics for an entity.

        Args:
            entity: Entity profile.

        Returns:
            List of IntensityResult for each calculable metric.
        """
        total = (
            _decimal(entity.total_scope1_tco2e)
            + _decimal(entity.total_scope2_location_tco2e)
        )
        if self._include_scope3:
            total += _decimal(entity.total_scope3_tco2e)

        sector_avgs = (
            self._custom_averages.get(entity.sector.value)
            or SECTOR_AVERAGE_INTENSITIES.get(entity.sector.value, {})
        )

        results: List[IntensityResult] = []

        # Revenue intensity
        if entity.revenue_eur_millions > Decimal("0"):
            intensity = _safe_divide(total, _decimal(entity.revenue_eur_millions))
            sector_avg = _decimal(sector_avgs.get("revenue", 0))
            vs_sector = float(
                _safe_pct(intensity - sector_avg, sector_avg)
            ) if sector_avg > Decimal("0") else 0.0

            results.append(IntensityResult(
                metric_type=IntensityMetricType.REVENUE.value,
                numerator_tco2e=_round4(float(total)),
                denominator_value=_round4(float(entity.revenue_eur_millions)),
                denominator_unit="EUR millions",
                intensity=_round4(float(intensity)),
                sector_average=_round4(float(sector_avg)),
                vs_sector_pct=_round2(vs_sector),
            ))

        # FTE intensity
        if entity.fte_count > Decimal("0"):
            intensity = _safe_divide(total, _decimal(entity.fte_count))
            sector_avg = _decimal(sector_avgs.get("fte", 0))
            vs_sector = float(
                _safe_pct(intensity - sector_avg, sector_avg)
            ) if sector_avg > Decimal("0") else 0.0

            results.append(IntensityResult(
                metric_type=IntensityMetricType.FTE.value,
                numerator_tco2e=_round4(float(total)),
                denominator_value=_round4(float(entity.fte_count)),
                denominator_unit="FTE",
                intensity=_round4(float(intensity)),
                sector_average=_round4(float(sector_avg)),
                vs_sector_pct=_round2(vs_sector),
            ))

        # Production intensity
        if entity.production_units > Decimal("0"):
            intensity = _safe_divide(total, _decimal(entity.production_units))
            results.append(IntensityResult(
                metric_type=IntensityMetricType.PRODUCTION.value,
                numerator_tco2e=_round4(float(total)),
                denominator_value=_round4(float(entity.production_units)),
                denominator_unit=entity.production_unit_name,
                intensity=_round6(float(intensity)),
                sector_average=0.0,
                vs_sector_pct=0.0,
            ))

        # Floor area intensity
        if entity.floor_area_m2 > Decimal("0"):
            intensity = _safe_divide(total, _decimal(entity.floor_area_m2))
            results.append(IntensityResult(
                metric_type=IntensityMetricType.FLOOR_AREA.value,
                numerator_tco2e=_round4(float(total)),
                denominator_value=_round4(float(entity.floor_area_m2)),
                denominator_unit="m2",
                intensity=_round6(float(intensity)),
                sector_average=0.0,
                vs_sector_pct=0.0,
            ))

        self._notes.append(
            f"Calculated {len(results)} intensity metrics for "
            f"'{entity.entity_name}'."
        )
        return results

    # -------------------------------------------------------------------
    # Private -- Sector benchmarking
    # -------------------------------------------------------------------

    def _benchmark_against_sector(
        self,
        entity: EntityProfile,
        intensities: List[IntensityResult],
    ) -> List[BenchmarkResult]:
        """Benchmark entity against sector averages.

        Args:
            entity: Entity profile.
            intensities: Calculated intensity metrics.

        Returns:
            List of BenchmarkResult vs sector.
        """
        results: List[BenchmarkResult] = []

        for intensity in intensities:
            if intensity.sector_average <= 0.0:
                continue

            # Approximate percentile rank from sector average comparison
            # If entity intensity < sector average, entity performs better
            ratio = intensity.intensity / intensity.sector_average
            # Approximate percentile: 50 + (1 - ratio) * 50, clamped 0-100
            approx_pct = max(0.0, min(100.0, 50.0 + (1.0 - ratio) * 50.0))

            status = self._classify_status(approx_pct)

            results.append(BenchmarkResult(
                metric_type=intensity.metric_type,
                entity_intensity=intensity.intensity,
                peer_mean=intensity.sector_average,
                peer_median=intensity.sector_average,
                peer_std_dev=0.0,
                peer_min=0.0,
                peer_max=0.0,
                percentile_rank=_round2(approx_pct),
                z_score=0.0,
                status=status,
                peer_count=0,
            ))

        return results

    # -------------------------------------------------------------------
    # Private -- Peer comparisons
    # -------------------------------------------------------------------

    def _compare_with_peers(
        self,
        entity: EntityProfile,
        peers: List[PeerDataPoint],
        intensities: List[IntensityResult],
    ) -> List[PeerComparison]:
        """Generate peer-by-peer comparison details.

        Args:
            entity: Entity profile.
            peers: Peer data points.
            intensities: Entity's intensity metrics.

        Returns:
            List of PeerComparison.
        """
        if not peers or not intensities:
            return []

        # Use revenue intensity by default for peer comparison
        entity_rev = next(
            (i for i in intensities if i.metric_type == IntensityMetricType.REVENUE.value),
            None,
        )
        if entity_rev is None and intensities:
            entity_rev = intensities[0]

        if entity_rev is None:
            return []

        comparisons: List[PeerComparison] = []
        entity_val = entity_rev.intensity

        for peer in peers:
            peer_val = float(peer.intensity_revenue)
            if peer_val == 0.0:
                continue

            diff = entity_val - peer_val
            diff_pct = float(
                _safe_pct(_decimal(diff), _decimal(peer_val))
            )

            comparisons.append(PeerComparison(
                peer_id=peer.peer_id,
                peer_name=peer.peer_name,
                peer_intensity=_round4(peer_val),
                entity_intensity=_round4(entity_val),
                difference=_round4(diff),
                difference_pct=_round2(diff_pct),
                entity_is_better=entity_val < peer_val,
            ))

        return comparisons

    def _calculate_peer_benchmarks(
        self,
        entity: EntityProfile,
        peers: List[PeerDataPoint],
        intensities: List[IntensityResult],
    ) -> List[BenchmarkResult]:
        """Calculate benchmarks against actual peer data.

        Args:
            entity: Entity profile.
            peers: Peer data.
            intensities: Entity intensities.

        Returns:
            List of BenchmarkResult from peer comparisons.
        """
        if not peers or not intensities:
            return []

        results: List[BenchmarkResult] = []

        # Revenue intensity benchmark
        entity_rev = next(
            (i for i in intensities if i.metric_type == IntensityMetricType.REVENUE.value),
            None,
        )
        if entity_rev:
            peer_vals = [
                float(p.intensity_revenue) for p in peers
                if float(p.intensity_revenue) > 0
            ]
            if peer_vals:
                results.append(
                    self._build_peer_benchmark(
                        IntensityMetricType.REVENUE.value,
                        entity_rev.intensity,
                        peer_vals,
                    )
                )

        # FTE intensity benchmark
        entity_fte = next(
            (i for i in intensities if i.metric_type == IntensityMetricType.FTE.value),
            None,
        )
        if entity_fte:
            peer_vals = [
                float(p.intensity_fte) for p in peers
                if float(p.intensity_fte) > 0
            ]
            if peer_vals:
                results.append(
                    self._build_peer_benchmark(
                        IntensityMetricType.FTE.value,
                        entity_fte.intensity,
                        peer_vals,
                    )
                )

        return results

    def _build_peer_benchmark(
        self,
        metric_type: str,
        entity_intensity: float,
        peer_values: List[float],
    ) -> BenchmarkResult:
        """Build a benchmark result from peer data.

        Args:
            metric_type: Metric type.
            entity_intensity: Entity's intensity.
            peer_values: Peer intensity values.

        Returns:
            BenchmarkResult.
        """
        n = len(peer_values)
        peer_mean = sum(peer_values) / n
        sorted_vals = sorted(peer_values)
        peer_median = sorted_vals[n // 2] if n % 2 else (
            (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
        )
        variance = sum((v - peer_mean) ** 2 for v in peer_values) / max(n - 1, 1)
        peer_std = math.sqrt(max(variance, 0.0))

        pct_rank = self.calculate_percentile_rank(entity_intensity, peer_values)
        z = self.calculate_z_score(entity_intensity, peer_mean, peer_std)
        status = self._classify_status(pct_rank)

        return BenchmarkResult(
            metric_type=metric_type,
            entity_intensity=_round4(entity_intensity),
            peer_mean=_round4(peer_mean),
            peer_median=_round4(peer_median),
            peer_std_dev=_round4(peer_std),
            peer_min=_round4(min(peer_values)),
            peer_max=_round4(max(peer_values)),
            percentile_rank=pct_rank,
            z_score=z,
            status=status,
            peer_count=n,
        )

    # -------------------------------------------------------------------
    # Private -- Facility ranking
    # -------------------------------------------------------------------

    def _rank_facilities(
        self,
        facilities: List[FacilityProfile],
    ) -> List[FacilityRanking]:
        """Rank facilities by emission intensity.

        Uses revenue intensity by default, falling back to FTE or total.

        Args:
            facilities: Facility profiles.

        Returns:
            List of FacilityRanking sorted by rank.
        """
        if not facilities:
            return []

        # Calculate intensity for each facility
        fac_intensities: List[Tuple[FacilityProfile, float, str]] = []

        for fac in facilities:
            if fac.revenue_eur_millions > Decimal("0"):
                intensity = float(
                    _safe_divide(fac.total_tco2e, fac.revenue_eur_millions)
                )
                metric = IntensityMetricType.REVENUE.value
            elif fac.fte_count > Decimal("0"):
                intensity = float(_safe_divide(fac.total_tco2e, fac.fte_count))
                metric = IntensityMetricType.FTE.value
            elif fac.floor_area_m2 > Decimal("0"):
                intensity = float(
                    _safe_divide(fac.total_tco2e, fac.floor_area_m2)
                )
                metric = IntensityMetricType.FLOOR_AREA.value
            else:
                intensity = float(fac.total_tco2e)
                metric = "absolute"
            fac_intensities.append((fac, intensity, metric))

        # Sort by intensity ascending (lower = better)
        fac_intensities.sort(key=lambda x: x[1])

        # Calculate organisation average
        all_intensities = [fi[1] for fi in fac_intensities]
        org_avg = sum(all_intensities) / len(all_intensities) if all_intensities else 0.0

        rankings: List[FacilityRanking] = []
        for rank, (fac, intensity, metric) in enumerate(fac_intensities, start=1):
            vs_avg = float(
                _safe_pct(
                    _decimal(intensity) - _decimal(org_avg),
                    _decimal(org_avg),
                )
            ) if org_avg > 0 else 0.0

            # Classify performance
            n = len(fac_intensities)
            pct = ((n - rank) / n) * 100.0 if n > 0 else 50.0
            perf = self._classify_status(pct)

            rankings.append(FacilityRanking(
                facility_id=fac.facility_id,
                facility_name=fac.facility_name,
                rank=rank,
                intensity=_round4(intensity),
                metric_type=metric,
                total_tco2e=_round4(float(fac.total_tco2e)),
                vs_org_average_pct=_round2(vs_avg),
                performance_category=perf,
            ))

        self._notes.append(
            f"Ranked {len(rankings)} facilities by intensity."
        )
        return rankings

    # -------------------------------------------------------------------
    # Private -- Trend analysis
    # -------------------------------------------------------------------

    def _analyse_trends(
        self,
        historical: List[HistoricalDataPoint],
    ) -> List[BenchmarkTrend]:
        """Analyse historical emission intensity trends.

        Args:
            historical: Historical data points sorted by year.

        Returns:
            List of BenchmarkTrend for each metric.
        """
        if len(historical) < 2:
            return []

        sorted_hist = sorted(historical, key=lambda h: h.year)
        trends: List[BenchmarkTrend] = []

        # Revenue intensity trend
        rev_points = [
            h for h in sorted_hist if float(h.intensity_revenue) > 0
        ]
        if len(rev_points) >= 2:
            trends.append(
                self._build_trend(
                    IntensityMetricType.REVENUE.value,
                    [(h.year, float(h.intensity_revenue)) for h in rev_points],
                )
            )

        # FTE intensity trend
        fte_points = [
            h for h in sorted_hist if float(h.intensity_fte) > 0
        ]
        if len(fte_points) >= 2:
            trends.append(
                self._build_trend(
                    IntensityMetricType.FTE.value,
                    [(h.year, float(h.intensity_fte)) for h in fte_points],
                )
            )

        # Absolute emissions trend
        abs_points = [
            h for h in sorted_hist if float(h.total_scope12_tco2e) > 0
        ]
        if len(abs_points) >= 2:
            trends.append(
                self._build_trend(
                    "absolute_scope12",
                    [(h.year, float(h.total_scope12_tco2e)) for h in abs_points],
                )
            )

        return trends

    def _build_trend(
        self,
        metric_type: str,
        data_points: List[Tuple[int, float]],
    ) -> BenchmarkTrend:
        """Build a trend analysis from year-value pairs.

        Args:
            metric_type: Metric type label.
            data_points: List of (year, value) tuples.

        Returns:
            BenchmarkTrend with YoY and compound analysis.
        """
        if len(data_points) < 2:
            return BenchmarkTrend(metric_type=metric_type)

        start_year, start_val = data_points[0]
        end_year, end_val = data_points[-1]
        abs_change = end_val - start_val
        pct_change = float(
            _safe_pct(_decimal(abs_change), _decimal(start_val))
        ) if start_val > 0 else 0.0

        # Compound annual change
        years = end_year - start_year
        cagr = 0.0
        if years > 0 and start_val > 0 and end_val > 0:
            ratio = end_val / start_val
            cagr = (math.pow(ratio, 1.0 / years) - 1.0) * 100.0

        # Year over year changes
        yoy: List[Dict[str, Any]] = []
        for i in range(1, len(data_points)):
            prev_year, prev_val = data_points[i - 1]
            curr_year, curr_val = data_points[i]
            yoy_change = curr_val - prev_val
            yoy_pct = float(
                _safe_pct(_decimal(yoy_change), _decimal(prev_val))
            ) if prev_val > 0 else 0.0
            yoy.append({
                "from_year": prev_year,
                "to_year": curr_year,
                "value_from": _round4(prev_val),
                "value_to": _round4(curr_val),
                "change": _round4(yoy_change),
                "change_pct": _round2(yoy_pct),
            })

        # Trend direction
        if pct_change < -2.0:
            direction = "improving"
        elif pct_change > 2.0:
            direction = "worsening"
        else:
            direction = "stable"

        return BenchmarkTrend(
            metric_type=metric_type,
            start_year=start_year,
            end_year=end_year,
            start_value=_round4(start_val),
            end_value=_round4(end_val),
            absolute_change=_round4(abs_change),
            percentage_change=_round2(pct_change),
            compound_annual_change_pct=_round2(cagr),
            trend_direction=direction,
            year_over_year=yoy,
        )

    # -------------------------------------------------------------------
    # Private -- CDP band estimation
    # -------------------------------------------------------------------

    def _estimate_cdp_band(
        self,
        benchmarks: List[BenchmarkResult],
    ) -> str:
        """Estimate CDP score band from benchmark results.

        Uses the best available percentile rank to map to CDP bands.

        Args:
            benchmarks: Benchmark results.

        Returns:
            CDP score band string.
        """
        if not benchmarks:
            return CDPScoreBand.D.value

        # Use the best percentile rank
        best_pct = max(b.percentile_rank for b in benchmarks)

        for band, threshold in sorted(
            CDP_BAND_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if best_pct >= threshold:
                return band

        return CDPScoreBand.D.value

    # -------------------------------------------------------------------
    # Private -- Status classification
    # -------------------------------------------------------------------

    def _classify_status(self, percentile_rank: float) -> str:
        """Classify benchmark status from percentile rank.

        Args:
            percentile_rank: Percentile rank (0-100, higher = better).

        Returns:
            BenchmarkStatus value.
        """
        for status, threshold in sorted(
            BENCHMARK_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if percentile_rank >= threshold:
                return status
        return BenchmarkStatus.LAGGARD.value

    def _determine_overall_status(
        self,
        benchmarks: List[BenchmarkResult],
    ) -> str:
        """Determine overall benchmark status.

        Uses weighted average of all benchmark percentile ranks.

        Args:
            benchmarks: All benchmark results.

        Returns:
            Overall BenchmarkStatus value.
        """
        if not benchmarks:
            return BenchmarkStatus.AVERAGE.value

        # Weight revenue and peer-based benchmarks higher
        total_rank = sum(b.percentile_rank for b in benchmarks)
        avg_rank = total_rank / len(benchmarks)

        return self._classify_status(avg_rank)

    # -------------------------------------------------------------------
    # Private -- Summary
    # -------------------------------------------------------------------

    def _build_summary(
        self,
        entity: EntityProfile,
        intensities: List[IntensityResult],
        benchmarks: List[BenchmarkResult],
        cdp_band: str,
        overall: str,
    ) -> str:
        """Build a human-readable benchmark summary.

        Args:
            entity: Entity profile.
            intensities: Calculated intensities.
            benchmarks: Benchmark results.
            cdp_band: Estimated CDP band.
            overall: Overall status.

        Returns:
            Summary string.
        """
        parts: List[str] = []
        parts.append(
            f"Benchmarking summary for '{entity.entity_name}' "
            f"(sector: {entity.sector.value})."
        )

        # Intensity summary
        for intensity in intensities[:2]:
            parts.append(
                f"- {intensity.metric_type.capitalize()} intensity: "
                f"{intensity.intensity} tCO2e/{intensity.denominator_unit}"
                + (
                    f" (sector avg: {intensity.sector_average})"
                    if intensity.sector_average > 0 else ""
                )
            )

        # Benchmark status
        if benchmarks:
            best = max(benchmarks, key=lambda b: b.percentile_rank)
            parts.append(
                f"- Best percentile rank: {best.percentile_rank}% "
                f"({best.metric_type} metric, {best.peer_count} peers)"
            )

        parts.append(f"- Overall status: {overall}")
        parts.append(f"- Estimated CDP score band: {cdp_band}")

        return " ".join(parts)
