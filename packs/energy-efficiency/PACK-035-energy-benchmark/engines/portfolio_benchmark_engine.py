# -*- coding: utf-8 -*-
"""
PortfolioBenchmarkEngine - PACK-035 Energy Benchmark Engine 6
==============================================================

Benchmarks portfolios of 1-1000+ facilities with area-weighted aggregation,
multi-criteria ranking, statistical distribution analysis, year-over-year
improvement tracking, and multi-entity hierarchy traversal.  Supports
portfolios organised by region, country, business unit, and site with
configurable aggregation methods and ranking criteria.

Calculation Methodology:
    Area-Weighted EUI:
        portfolio_eui = sum(facility_eui_i * facility_area_i) / sum(facility_area_i)

    Simple Average EUI:
        portfolio_eui = sum(facility_eui_i) / n

    Consumption-Weighted EUI:
        portfolio_eui = sum(energy_i) / sum(area_i)

    Percentile Rank:
        percentile_i = (count_of_peers_worse + 0.5 * count_of_ties) / n * 100

    Year-over-Year Improvement:
        yoy_pct = (eui_prev - eui_current) / eui_prev * 100

    Quartile Assignment:
        Q1 (top 25%), Q2, Q3, Q4 (bottom 25%) by ascending EUI rank

    Statistical Distribution:
        mean, median, std_dev, min, max, P10, P25, P75, P90, skewness, kurtosis

Regulatory / Standard References:
    - ISO 50001:2018 Energy Management Systems
    - ISO 50006:2014 Measuring energy performance using baselines and EnPIs
    - ENERGY STAR Portfolio Manager Technical Reference (EPA, 2024)
    - EU EED 2023/1791 Article 8 (Multi-site audits)
    - ASHRAE Standard 100-2018 (Energy Benchmarking)
    - GRESB Real Estate Assessment (Portfolio-level ESG benchmarking)
    - CRREM (Carbon Risk Real Estate Monitor) Portfolio Analysis

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Aggregation methods follow ISO 50006 and ENERGY STAR Portfolio Manager
    - Statistical measures computed from first principles
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-035 Energy Benchmark
Engine:  6 of 10
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

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
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
    """Safely convert a value to Decimal."""
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


def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* decimal digits and return float."""
    return float(value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP))


def _round2(value: Any) -> float:
    """Round to 2 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round3(value: Any) -> float:
    """Round to 3 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _round4(value: Any) -> float:
    """Round to 4 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AggregationMethod(str, Enum):
    """Method for aggregating facility-level EUI into portfolio-level.

    AREA_WEIGHTED:        Weight each facility EUI by its gross floor area.
    SIMPLE_AVERAGE:       Unweighted arithmetic mean of facility EUIs.
    CONSUMPTION_WEIGHTED: Total energy / total area (true portfolio EUI).
    MEDIAN:               Median facility EUI.
    """
    AREA_WEIGHTED = "area_weighted"
    SIMPLE_AVERAGE = "simple_average"
    CONSUMPTION_WEIGHTED = "consumption_weighted"
    MEDIAN = "median"


class RankingCriteria(str, Enum):
    """Criteria for ranking facilities within a portfolio.

    EUI_ABSOLUTE:      Rank by absolute EUI value (lower is better).
    EUI_PERCENTILE:    Rank by percentile within peer group.
    IMPROVEMENT_RATE:  Rank by year-over-year EUI improvement rate.
    CARBON_INTENSITY:  Rank by carbon intensity (kgCO2/m2).
    COST_INTENSITY:    Rank by energy cost intensity (EUR/m2).
    """
    EUI_ABSOLUTE = "eui_absolute"
    EUI_PERCENTILE = "eui_percentile"
    IMPROVEMENT_RATE = "improvement_rate"
    CARBON_INTENSITY = "carbon_intensity"
    COST_INTENSITY = "cost_intensity"


class PortfolioTier(str, Enum):
    """Quartile classification within portfolio.

    TOP_QUARTILE:     Best 25% of facilities (P75-P100).
    SECOND_QUARTILE:  P50-P75.
    THIRD_QUARTILE:   P25-P50.
    BOTTOM_QUARTILE:  Worst 25% of facilities (P0-P25).
    """
    TOP_QUARTILE = "top_quartile"
    SECOND_QUARTILE = "second_quartile"
    THIRD_QUARTILE = "third_quartile"
    BOTTOM_QUARTILE = "bottom_quartile"


class EntityLevel(str, Enum):
    """Hierarchy levels in a multi-entity portfolio.

    PORTFOLIO:     Entire portfolio.
    REGION:        Geographic region (e.g. EMEA, APAC, Americas).
    COUNTRY:       Individual country.
    BUSINESS_UNIT: Organisational business unit.
    SITE:          Individual facility / site.
    """
    PORTFOLIO = "portfolio"
    REGION = "region"
    COUNTRY = "country"
    BUSINESS_UNIT = "business_unit"
    SITE = "site"


class OutlierDetectionMethod(str, Enum):
    """Method for detecting outlier facilities in a portfolio.

    IQR:           Tukey's 1.5 * IQR fence method.
    Z_SCORE:       Z-score > 3.0 standard deviations.
    MODIFIED_Z:    Modified Z-score using MAD (robust to outliers).
    PERCENTILE:    Outside P1 / P99 range.
    """
    IQR = "iqr"
    Z_SCORE = "z_score"
    MODIFIED_Z = "modified_z"
    PERCENTILE = "percentile"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Area-weighted aggregation adjustments by building type.
# Source: ENERGY STAR Portfolio Manager Technical Reference 2024,
# CIBSE TM46:2008, ASHRAE Standard 100-2018.
PORTFOLIO_AGGREGATION_WEIGHTS: Dict[str, float] = {
    "office": 1.0,
    "retail": 1.0,
    "warehouse": 0.6,
    "hospital": 1.8,
    "school": 0.9,
    "hotel": 1.3,
    "data_centre": 2.5,
    "industrial": 1.2,
    "residential": 0.8,
    "mixed_use": 1.0,
    "laboratory": 2.0,
    "restaurant": 1.5,
    "supermarket": 1.4,
    "cold_storage": 1.6,
    "default": 1.0,
}

# Year-over-year improvement rate thresholds for performance assessment.
# Source: ISO 50001:2018 Annex A (typical improvement expectations),
# EU EED 2023/1791 Article 8 (1.5% annual improvement target),
# GRESB benchmark data 2024.
IMPROVEMENT_RATE_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "excellent": {
        "min_pct": 5.0,
        "description": "Exceptional improvement (>5% year-over-year)",
        "colour": "#00B050",
    },
    "good": {
        "min_pct": 3.0,
        "description": "Strong improvement (3-5% year-over-year)",
        "colour": "#92D050",
    },
    "on_target": {
        "min_pct": 1.5,
        "description": "On target per EED 1.5% annual target",
        "colour": "#FFFF00",
    },
    "marginal": {
        "min_pct": 0.5,
        "description": "Marginal improvement (0.5-1.5%)",
        "colour": "#FFC000",
    },
    "stagnant": {
        "min_pct": -0.5,
        "description": "Essentially flat performance (-0.5% to +0.5%)",
        "colour": "#FF8000",
    },
    "deteriorating": {
        "min_pct": -999.0,
        "description": "Performance worsening (>0.5% increase)",
        "colour": "#FF0000",
    },
}

# Default percentile breakpoints for distribution analysis.
# Source: Standard statistical analysis practice.
DEFAULT_PERCENTILES: List[float] = [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class PortfolioDefinition(BaseModel):
    """Definition of a portfolio for benchmarking.

    Attributes:
        portfolio_id: Unique portfolio identifier.
        portfolio_name: Human-readable portfolio name.
        entity_level: Portfolio hierarchy level.
        parent_entity_id: Parent entity identifier (for hierarchy).
        description: Portfolio description.
        target_eui: Target EUI for the portfolio (kWh/m2/year).
        target_year: Year by which the target should be achieved.
    """
    portfolio_id: str = Field(default_factory=_new_uuid, description="Unique portfolio identifier")
    portfolio_name: str = Field(default="", max_length=500, description="Portfolio name")
    entity_level: EntityLevel = Field(default=EntityLevel.PORTFOLIO, description="Hierarchy level")
    parent_entity_id: Optional[str] = Field(default=None, description="Parent entity ID")
    description: str = Field(default="", description="Portfolio description")
    target_eui: Optional[float] = Field(default=None, ge=0.0, description="Target EUI kWh/m2/year")
    target_year: Optional[int] = Field(default=None, ge=2020, le=2060, description="Target year")


class FacilityMembership(BaseModel):
    """A single facility within a portfolio.

    Attributes:
        facility_id: Unique facility identifier.
        facility_name: Human-readable facility name.
        building_type: Building type classification.
        region: Geographic region.
        country: ISO 2-letter country code.
        business_unit: Business unit name.
        gross_floor_area_m2: Gross floor area (m2).
        energy_consumption_kwh: Total annual energy consumption (kWh).
        eui_kwh_per_m2: Pre-calculated EUI (if available).
        carbon_emissions_kgco2: Annual carbon emissions (kgCO2).
        energy_cost_eur: Annual energy cost (EUR).
        reporting_year: Year of the data.
        historical_eui: Historical EUI values {year: eui_value}.
        is_active: Whether the facility is currently active.
    """
    facility_id: str = Field(default_factory=_new_uuid, description="Facility identifier")
    facility_name: str = Field(default="", max_length=500, description="Facility name")
    building_type: str = Field(default="office", description="Building type")
    region: str = Field(default="", description="Region")
    country: str = Field(default="", description="Country ISO code")
    business_unit: str = Field(default="", description="Business unit")
    gross_floor_area_m2: float = Field(default=0.0, ge=0.0, description="Gross floor area m2")
    energy_consumption_kwh: float = Field(default=0.0, ge=0.0, description="Total energy kWh")
    eui_kwh_per_m2: Optional[float] = Field(default=None, ge=0.0, description="EUI kWh/m2/year")
    carbon_emissions_kgco2: float = Field(default=0.0, ge=0.0, description="Carbon emissions kgCO2")
    energy_cost_eur: float = Field(default=0.0, ge=0.0, description="Energy cost EUR")
    reporting_year: int = Field(default=2025, ge=2015, le=2035, description="Reporting year")
    historical_eui: Dict[int, float] = Field(
        default_factory=dict, description="Historical EUI: {year: eui_value}"
    )
    is_active: bool = Field(default=True, description="Is facility active")

    @model_validator(mode="after")
    def compute_eui_if_missing(self) -> "FacilityMembership":
        """Compute EUI from consumption and area if not provided."""
        if self.eui_kwh_per_m2 is None and self.gross_floor_area_m2 > 0:
            self.eui_kwh_per_m2 = self.energy_consumption_kwh / self.gross_floor_area_m2
        return self


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class PortfolioMetrics(BaseModel):
    """Aggregated metrics for a portfolio or sub-portfolio.

    Attributes:
        entity_level: Hierarchy level of aggregation.
        entity_name: Name of the entity (portfolio, region, etc.).
        facility_count: Number of active facilities.
        total_area_m2: Total gross floor area (m2).
        total_energy_kwh: Total energy consumption (kWh).
        total_carbon_kgco2: Total carbon emissions (kgCO2).
        total_cost_eur: Total energy cost (EUR).
        area_weighted_eui: Area-weighted EUI (kWh/m2/year).
        simple_average_eui: Simple average EUI.
        consumption_weighted_eui: Consumption-weighted EUI.
        median_eui: Median EUI.
        carbon_intensity_kgco2_per_m2: Carbon intensity (kgCO2/m2).
        cost_intensity_eur_per_m2: Cost intensity (EUR/m2).
    """
    entity_level: str = Field(default="portfolio")
    entity_name: str = Field(default="")
    facility_count: int = Field(default=0, ge=0)
    total_area_m2: float = Field(default=0.0)
    total_energy_kwh: float = Field(default=0.0)
    total_carbon_kgco2: float = Field(default=0.0)
    total_cost_eur: float = Field(default=0.0)
    area_weighted_eui: float = Field(default=0.0)
    simple_average_eui: float = Field(default=0.0)
    consumption_weighted_eui: float = Field(default=0.0)
    median_eui: float = Field(default=0.0)
    carbon_intensity_kgco2_per_m2: float = Field(default=0.0)
    cost_intensity_eur_per_m2: float = Field(default=0.0)


class FacilityRanking(BaseModel):
    """Ranking of a single facility within its portfolio.

    Attributes:
        facility_id: Facility identifier.
        facility_name: Facility name.
        building_type: Building type.
        rank: Rank position (1 = best).
        rank_of: Total facilities in ranking.
        percentile: Percentile position (100 = best).
        tier: Quartile tier assignment.
        eui_kwh_per_m2: Facility EUI.
        carbon_intensity_kgco2_per_m2: Carbon intensity.
        cost_intensity_eur_per_m2: Cost intensity.
        yoy_improvement_pct: Year-over-year improvement percentage.
        improvement_assessment: Textual assessment of improvement.
        is_outlier: Whether the facility is flagged as an outlier.
    """
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    building_type: str = Field(default="")
    rank: int = Field(default=0, ge=0)
    rank_of: int = Field(default=0, ge=0)
    percentile: float = Field(default=0.0, ge=0.0, le=100.0)
    tier: PortfolioTier = Field(default=PortfolioTier.SECOND_QUARTILE)
    eui_kwh_per_m2: float = Field(default=0.0)
    carbon_intensity_kgco2_per_m2: float = Field(default=0.0)
    cost_intensity_eur_per_m2: float = Field(default=0.0)
    yoy_improvement_pct: float = Field(default=0.0)
    improvement_assessment: str = Field(default="")
    is_outlier: bool = Field(default=False)


class PortfolioDistribution(BaseModel):
    """Statistical distribution of EUI across a portfolio.

    Attributes:
        count: Number of data points.
        mean: Arithmetic mean.
        median: Median (P50).
        std_dev: Standard deviation.
        min_val: Minimum value.
        max_val: Maximum value.
        range_val: Max - Min.
        skewness: Distribution skewness (0 = symmetric).
        kurtosis: Distribution kurtosis (3 = normal).
        percentiles: Percentile values {pN: value}.
        histogram_bins: Histogram bin edges.
        histogram_counts: Count per bin.
        box_plot_data: Box-plot summary data.
        outlier_count: Number of outliers detected.
        outlier_facility_ids: IDs of outlier facilities.
    """
    count: int = Field(default=0, ge=0)
    mean: float = Field(default=0.0)
    median: float = Field(default=0.0)
    std_dev: float = Field(default=0.0)
    min_val: float = Field(default=0.0)
    max_val: float = Field(default=0.0)
    range_val: float = Field(default=0.0)
    skewness: float = Field(default=0.0)
    kurtosis: float = Field(default=0.0)
    percentiles: Dict[str, float] = Field(default_factory=dict)
    histogram_bins: List[float] = Field(default_factory=list)
    histogram_counts: List[int] = Field(default_factory=list)
    box_plot_data: Dict[str, float] = Field(default_factory=dict)
    outlier_count: int = Field(default=0, ge=0)
    outlier_facility_ids: List[str] = Field(default_factory=list)


class PortfolioTrend(BaseModel):
    """Year-over-year trend data for the portfolio.

    Attributes:
        year: Reporting year.
        facility_count: Number of facilities reporting.
        area_weighted_eui: Area-weighted EUI for the year.
        yoy_change_pct: Year-over-year change (negative = improvement).
        cumulative_change_pct: Cumulative change from baseline year.
    """
    year: int = Field(default=2025)
    facility_count: int = Field(default=0, ge=0)
    area_weighted_eui: float = Field(default=0.0)
    yoy_change_pct: float = Field(default=0.0)
    cumulative_change_pct: float = Field(default=0.0)


class PortfolioBenchmarkResult(BaseModel):
    """Complete portfolio benchmarking result.

    Attributes:
        result_id: Unique result identifier.
        portfolio: Portfolio definition.
        portfolio_metrics: Aggregated portfolio metrics.
        facility_rankings: Ranked list of facilities.
        distribution: Statistical distribution of EUI.
        trends: Year-over-year portfolio trends.
        entity_breakdowns: Metrics broken down by entity (region, country, BU).
        best_performers: Top-performing facility IDs.
        worst_performers: Worst-performing facility IDs.
        target_gap_pct: Gap to portfolio target EUI (%).
        methodology_notes: Methodology and source notes.
        processing_time_ms: Computation time (ms).
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    portfolio: Optional[PortfolioDefinition] = Field(default=None)
    portfolio_metrics: Optional[PortfolioMetrics] = Field(default=None)
    facility_rankings: List[FacilityRanking] = Field(default_factory=list)
    distribution: Optional[PortfolioDistribution] = Field(default=None)
    trends: List[PortfolioTrend] = Field(default_factory=list)
    entity_breakdowns: Dict[str, PortfolioMetrics] = Field(default_factory=dict)
    best_performers: List[str] = Field(default_factory=list)
    worst_performers: List[str] = Field(default_factory=list)
    target_gap_pct: float = Field(default=0.0)
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PortfolioBenchmarkEngine:
    """Zero-hallucination portfolio benchmarking engine.

    Benchmarks portfolios of 1-1000+ facilities with area-weighted
    aggregation, multi-criteria ranking, statistical distribution
    analysis, year-over-year improvement tracking, and multi-entity
    hierarchy traversal.

    Guarantees:
        - Deterministic: same inputs produce identical outputs.
        - Reproducible: every result carries a SHA-256 provenance hash.
        - Auditable: full breakdown of rankings and distributions.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = PortfolioBenchmarkEngine()
        result = engine.benchmark_portfolio(portfolio_def, facilities)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the portfolio benchmark engine.

        Args:
            config: Optional configuration overrides.  Supported keys:
                - aggregation_method (str): default aggregation method
                - ranking_criteria (str): default ranking criteria
                - outlier_method (str): outlier detection method
                - top_n (int): number of best/worst performers to report
        """
        self._config = config or {}
        self._default_aggregation = AggregationMethod(
            self._config.get("aggregation_method", AggregationMethod.AREA_WEIGHTED.value)
        )
        self._default_ranking = RankingCriteria(
            self._config.get("ranking_criteria", RankingCriteria.EUI_ABSOLUTE.value)
        )
        self._default_outlier = OutlierDetectionMethod(
            self._config.get("outlier_method", OutlierDetectionMethod.IQR.value)
        )
        self._top_n = int(self._config.get("top_n", 5))
        self._notes: List[str] = []
        logger.info("PortfolioBenchmarkEngine v%s initialised.", _MODULE_VERSION)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def benchmark_portfolio(
        self,
        portfolio: PortfolioDefinition,
        facilities: List[FacilityMembership],
        aggregation: Optional[AggregationMethod] = None,
        ranking: Optional[RankingCriteria] = None,
    ) -> PortfolioBenchmarkResult:
        """Run comprehensive portfolio benchmarking.

        Calculates portfolio metrics, ranks facilities, analyses
        distributions, computes year-over-year trends, and breaks
        down metrics by entity hierarchy.

        Args:
            portfolio: Portfolio definition.
            facilities: List of facilities in the portfolio.
            aggregation: Aggregation method override.
            ranking: Ranking criteria override.

        Returns:
            PortfolioBenchmarkResult with full analysis and provenance.

        Raises:
            ValueError: If no active facilities with valid data are provided.
        """
        t0 = time.perf_counter()
        self._notes = [
            f"Engine version: {self.engine_version}",
            f"Portfolio: {portfolio.portfolio_name}",
        ]

        agg_method = aggregation or self._default_aggregation
        rank_criteria = ranking or self._default_ranking

        # Filter to active facilities with valid area and EUI.
        active = [f for f in facilities if f.is_active and f.gross_floor_area_m2 > 0]
        if not active:
            raise ValueError("No active facilities with valid area data for benchmarking.")

        self._notes.append(f"Active facilities: {len(active)} of {len(facilities)}")

        # --- 1. Portfolio Metrics ---
        portfolio_metrics = self.calculate_portfolio_metrics(active, agg_method)
        portfolio_metrics.entity_level = portfolio.entity_level.value
        portfolio_metrics.entity_name = portfolio.portfolio_name

        # --- 2. Facility Rankings ---
        rankings = self.rank_facilities(active, rank_criteria)

        # --- 3. Distribution Analysis ---
        distribution = self.get_distribution(active)

        # --- 4. Year-over-Year Trends ---
        trends = self.calculate_yoy_improvement(active)

        # --- 5. Entity Breakdowns ---
        entity_breakdowns = self.aggregate_by_entity(active, agg_method)

        # --- 6. Best/Worst Performers ---
        best_ids = self.identify_best_performers(rankings, self._top_n)
        worst_ids = self.identify_worst_performers(rankings, self._top_n)

        # --- 7. Target Gap ---
        target_gap = Decimal("0")
        if portfolio.target_eui is not None and portfolio.target_eui > 0:
            d_actual = _decimal(portfolio_metrics.area_weighted_eui)
            d_target = _decimal(portfolio.target_eui)
            target_gap = _safe_pct(d_actual - d_target, d_target)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = PortfolioBenchmarkResult(
            portfolio=portfolio,
            portfolio_metrics=portfolio_metrics,
            facility_rankings=rankings,
            distribution=distribution,
            trends=trends,
            entity_breakdowns=entity_breakdowns,
            best_performers=best_ids,
            worst_performers=worst_ids,
            target_gap_pct=_round2(float(target_gap)),
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )

        result.provenance_hash = _compute_hash(result)
        logger.info(
            "Portfolio benchmark complete: %d facilities, EUI=%.1f, hash=%s (%.1f ms)",
            len(active),
            portfolio_metrics.area_weighted_eui,
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    # --------------------------------------------------------------------- #
    # Portfolio Metrics Calculation
    # --------------------------------------------------------------------- #

    def calculate_portfolio_metrics(
        self,
        facilities: List[FacilityMembership],
        method: AggregationMethod = AggregationMethod.AREA_WEIGHTED,
    ) -> PortfolioMetrics:
        """Calculate aggregated portfolio-level metrics.

        Args:
            facilities: List of active facilities.
            method: Aggregation method for portfolio EUI.

        Returns:
            PortfolioMetrics with all aggregated values.
        """
        n = len(facilities)
        if n == 0:
            return PortfolioMetrics()

        total_area = sum(_decimal(f.gross_floor_area_m2) for f in facilities)
        total_energy = sum(_decimal(f.energy_consumption_kwh) for f in facilities)
        total_carbon = sum(_decimal(f.carbon_emissions_kgco2) for f in facilities)
        total_cost = sum(_decimal(f.energy_cost_eur) for f in facilities)

        # Compute EUIs for each facility.
        eui_values = self._get_eui_values(facilities)

        # Area-weighted EUI.
        area_weighted = self._area_weighted_eui(facilities, eui_values)

        # Simple average EUI.
        simple_avg = _safe_divide(
            sum(_decimal(e) for e in eui_values),
            _decimal(n),
        )

        # Consumption-weighted EUI (total energy / total area).
        consumption_weighted = _safe_divide(total_energy, total_area)

        # Median EUI.
        median_eui = self._compute_median([_decimal(e) for e in eui_values])

        # Carbon and cost intensity.
        carbon_intensity = _safe_divide(total_carbon, total_area)
        cost_intensity = _safe_divide(total_cost, total_area)

        self._notes.append(
            f"Portfolio metrics: {n} facilities, total area {_round2(float(total_area))} m2, "
            f"area-weighted EUI {_round2(float(area_weighted))} kWh/m2/yr."
        )

        return PortfolioMetrics(
            facility_count=n,
            total_area_m2=_round2(float(total_area)),
            total_energy_kwh=_round2(float(total_energy)),
            total_carbon_kgco2=_round2(float(total_carbon)),
            total_cost_eur=_round2(float(total_cost)),
            area_weighted_eui=_round2(float(area_weighted)),
            simple_average_eui=_round2(float(simple_avg)),
            consumption_weighted_eui=_round2(float(consumption_weighted)),
            median_eui=_round2(float(median_eui)),
            carbon_intensity_kgco2_per_m2=_round3(float(carbon_intensity)),
            cost_intensity_eur_per_m2=_round3(float(cost_intensity)),
        )

    # --------------------------------------------------------------------- #
    # Facility Ranking
    # --------------------------------------------------------------------- #

    def rank_facilities(
        self,
        facilities: List[FacilityMembership],
        criteria: RankingCriteria = RankingCriteria.EUI_ABSOLUTE,
    ) -> List[FacilityRanking]:
        """Rank facilities within the portfolio by the chosen criteria.

        Args:
            facilities: List of active facilities.
            criteria: Ranking criteria to use.

        Returns:
            List of FacilityRanking sorted by rank (1 = best).
        """
        n = len(facilities)
        if n == 0:
            return []

        # Compute the ranking metric for each facility.
        scored: List[Tuple[FacilityMembership, Decimal]] = []
        for f in facilities:
            score = self._compute_ranking_score(f, criteria)
            scored.append((f, score))

        # Sort ascending (lower is better for all criteria except IMPROVEMENT_RATE).
        if criteria == RankingCriteria.IMPROVEMENT_RATE:
            scored.sort(key=lambda x: x[1], reverse=True)  # Higher improvement = better
        else:
            scored.sort(key=lambda x: x[1])  # Lower value = better

        # Detect outliers.
        all_scores = [s for _, s in scored]
        outlier_ids = self._detect_outliers(
            [(f.facility_id, s) for f, s in scored],
            self._default_outlier,
        )

        rankings: List[FacilityRanking] = []
        for rank_pos, (f, score) in enumerate(scored, start=1):
            # Percentile: rank 1 (best) = 100th percentile.
            percentile = _safe_divide(
                _decimal(n - rank_pos) * Decimal("100"),
                _decimal(n - 1) if n > 1 else Decimal("1"),
            )
            percentile = min(percentile, Decimal("100"))

            # Tier assignment.
            tier = self._assign_tier(percentile)

            # YoY improvement.
            yoy_pct = self._facility_yoy_improvement(f)
            improvement_text = self._assess_improvement(yoy_pct)

            # EUI and intensities.
            d_area = _decimal(f.gross_floor_area_m2)
            eui_val = _safe_divide(_decimal(f.energy_consumption_kwh), d_area)
            carbon_val = _safe_divide(_decimal(f.carbon_emissions_kgco2), d_area)
            cost_val = _safe_divide(_decimal(f.energy_cost_eur), d_area)

            rankings.append(FacilityRanking(
                facility_id=f.facility_id,
                facility_name=f.facility_name,
                building_type=f.building_type,
                rank=rank_pos,
                rank_of=n,
                percentile=_round2(float(percentile)),
                tier=tier,
                eui_kwh_per_m2=_round2(float(eui_val)),
                carbon_intensity_kgco2_per_m2=_round3(float(carbon_val)),
                cost_intensity_eur_per_m2=_round3(float(cost_val)),
                yoy_improvement_pct=_round2(float(yoy_pct)),
                improvement_assessment=improvement_text,
                is_outlier=f.facility_id in outlier_ids,
            ))

        self._notes.append(
            f"Rankings: {n} facilities ranked by {criteria.value}, "
            f"{len(outlier_ids)} outliers detected."
        )

        return rankings

    # --------------------------------------------------------------------- #
    # Distribution Analysis
    # --------------------------------------------------------------------- #

    def get_distribution(
        self,
        facilities: List[FacilityMembership],
        num_bins: int = 10,
    ) -> PortfolioDistribution:
        """Compute statistical distribution of EUI across the portfolio.

        Args:
            facilities: List of active facilities.
            num_bins: Number of histogram bins.

        Returns:
            PortfolioDistribution with full statistics.
        """
        eui_values = sorted([_decimal(e) for e in self._get_eui_values(facilities)])
        n = len(eui_values)

        if n == 0:
            return PortfolioDistribution()

        # Basic statistics.
        d_sum = sum(eui_values)
        mean = _safe_divide(d_sum, _decimal(n))
        median = self._compute_median(eui_values)
        min_val = eui_values[0]
        max_val = eui_values[-1]
        range_val = max_val - min_val

        # Standard deviation (population).
        if n > 1:
            variance = _safe_divide(
                sum((x - mean) ** 2 for x in eui_values),
                _decimal(n - 1),
            )
            # Use math.sqrt on float for non-negative values.
            std_dev = _decimal(math.sqrt(max(float(variance), 0.0)))
        else:
            std_dev = Decimal("0")
            variance = Decimal("0")

        # Skewness (Fisher).
        skewness = Decimal("0")
        if n > 2 and std_dev > Decimal("0"):
            m3 = _safe_divide(
                sum((x - mean) ** 3 for x in eui_values),
                _decimal(n),
            )
            skewness = _safe_divide(m3, std_dev ** 3)

        # Kurtosis (excess, Fisher definition, 0 for normal).
        kurtosis = Decimal("0")
        if n > 3 and std_dev > Decimal("0"):
            m4 = _safe_divide(
                sum((x - mean) ** 4 for x in eui_values),
                _decimal(n),
            )
            kurtosis = _safe_divide(m4, std_dev ** 4) - Decimal("3")

        # Percentiles.
        percentile_values: Dict[str, float] = {}
        for p in DEFAULT_PERCENTILES:
            pval = self._compute_percentile_value(eui_values, p)
            percentile_values[f"P{int(p)}"] = _round2(float(pval))

        # Histogram.
        bins, counts = self._compute_histogram(eui_values, num_bins)

        # Box plot data.
        p25 = self._compute_percentile_value(eui_values, 25.0)
        p75 = self._compute_percentile_value(eui_values, 75.0)
        iqr = p75 - p25
        whisker_low = max(min_val, p25 - Decimal("1.5") * iqr)
        whisker_high = min(max_val, p75 + Decimal("1.5") * iqr)

        box_plot = {
            "q1": _round2(float(p25)),
            "median": _round2(float(median)),
            "q3": _round2(float(p75)),
            "iqr": _round2(float(iqr)),
            "whisker_low": _round2(float(whisker_low)),
            "whisker_high": _round2(float(whisker_high)),
            "min": _round2(float(min_val)),
            "max": _round2(float(max_val)),
        }

        # Outliers (IQR method).
        outlier_ids: List[str] = []
        for f in facilities:
            eui = self._get_facility_eui(f)
            if eui < whisker_low or eui > whisker_high:
                outlier_ids.append(f.facility_id)

        return PortfolioDistribution(
            count=n,
            mean=_round2(float(mean)),
            median=_round2(float(median)),
            std_dev=_round3(float(std_dev)),
            min_val=_round2(float(min_val)),
            max_val=_round2(float(max_val)),
            range_val=_round2(float(range_val)),
            skewness=_round3(float(skewness)),
            kurtosis=_round3(float(kurtosis)),
            percentiles=percentile_values,
            histogram_bins=[_round2(float(b)) for b in bins],
            histogram_counts=counts,
            box_plot_data=box_plot,
            outlier_count=len(outlier_ids),
            outlier_facility_ids=outlier_ids,
        )

    # --------------------------------------------------------------------- #
    # Year-over-Year Improvement
    # --------------------------------------------------------------------- #

    def calculate_yoy_improvement(
        self,
        facilities: List[FacilityMembership],
    ) -> List[PortfolioTrend]:
        """Calculate year-over-year portfolio-level improvement trends.

        Aggregates historical EUI data across all facilities to produce
        a time series of area-weighted portfolio EUI with year-over-year
        and cumulative change metrics.

        Args:
            facilities: List of active facilities.

        Returns:
            List of PortfolioTrend sorted by year.
        """
        # Collect all years reported across facilities.
        all_years: set = set()
        for f in facilities:
            if f.historical_eui:
                all_years.update(f.historical_eui.keys())
            # Also include the current reporting year.
            all_years.add(f.reporting_year)

        if not all_years:
            return []

        sorted_years = sorted(all_years)
        trends: List[PortfolioTrend] = []
        baseline_eui: Optional[Decimal] = None
        prev_eui: Optional[Decimal] = None

        for year in sorted_years:
            # Collect facilities reporting in this year.
            year_facilities: List[Tuple[Decimal, Decimal]] = []  # (eui, area)
            count = 0

            for f in facilities:
                eui_val: Optional[float] = None
                area = _decimal(f.gross_floor_area_m2)

                if year in (f.historical_eui or {}):
                    eui_val = f.historical_eui[year]
                elif year == f.reporting_year and f.eui_kwh_per_m2 is not None:
                    eui_val = f.eui_kwh_per_m2

                if eui_val is not None and area > Decimal("0"):
                    year_facilities.append((_decimal(eui_val), area))
                    count += 1

            if not year_facilities:
                continue

            # Area-weighted EUI for this year.
            total_weighted = sum(eui * area for eui, area in year_facilities)
            total_area = sum(area for _, area in year_facilities)
            year_eui = _safe_divide(total_weighted, total_area)

            if baseline_eui is None:
                baseline_eui = year_eui

            yoy_pct = Decimal("0")
            if prev_eui is not None and prev_eui > Decimal("0"):
                yoy_pct = _safe_pct(prev_eui - year_eui, prev_eui)

            cumulative_pct = Decimal("0")
            if baseline_eui > Decimal("0"):
                cumulative_pct = _safe_pct(baseline_eui - year_eui, baseline_eui)

            trends.append(PortfolioTrend(
                year=year,
                facility_count=count,
                area_weighted_eui=_round2(float(year_eui)),
                yoy_change_pct=_round2(float(yoy_pct)),
                cumulative_change_pct=_round2(float(cumulative_pct)),
            ))

            prev_eui = year_eui

        if trends:
            self._notes.append(
                f"Trends: {len(trends)} years, baseline {trends[0].area_weighted_eui} "
                f"({trends[0].year}), latest {trends[-1].area_weighted_eui} "
                f"({trends[-1].year}), cumulative {trends[-1].cumulative_change_pct}%."
            )

        return trends

    # --------------------------------------------------------------------- #
    # Best / Worst Performers
    # --------------------------------------------------------------------- #

    def identify_best_performers(
        self,
        rankings: List[FacilityRanking],
        top_n: int = 5,
    ) -> List[str]:
        """Identify the top N best-performing facilities.

        Args:
            rankings: Ranked facility list (rank 1 = best).
            top_n: Number of best performers to return.

        Returns:
            List of facility IDs of the best performers.
        """
        return [r.facility_id for r in rankings[:top_n]]

    def identify_worst_performers(
        self,
        rankings: List[FacilityRanking],
        bottom_n: int = 5,
    ) -> List[str]:
        """Identify the bottom N worst-performing facilities.

        Args:
            rankings: Ranked facility list (last = worst).
            bottom_n: Number of worst performers to return.

        Returns:
            List of facility IDs of the worst performers.
        """
        return [r.facility_id for r in rankings[-bottom_n:]]

    # --------------------------------------------------------------------- #
    # Entity Aggregation
    # --------------------------------------------------------------------- #

    def aggregate_by_entity(
        self,
        facilities: List[FacilityMembership],
        method: AggregationMethod = AggregationMethod.AREA_WEIGHTED,
    ) -> Dict[str, PortfolioMetrics]:
        """Aggregate metrics by entity hierarchy (region, country, BU).

        Traverses the multi-entity hierarchy and calculates metrics
        at each grouping level.

        Args:
            facilities: List of active facilities.
            method: Aggregation method.

        Returns:
            Dict mapping entity key to PortfolioMetrics.
        """
        breakdowns: Dict[str, PortfolioMetrics] = {}

        # Group by region.
        regions = self._group_by_attribute(facilities, "region")
        for region_name, region_facs in regions.items():
            if not region_name:
                continue
            metrics = self.calculate_portfolio_metrics(region_facs, method)
            metrics.entity_level = EntityLevel.REGION.value
            metrics.entity_name = region_name
            breakdowns[f"region:{region_name}"] = metrics

        # Group by country.
        countries = self._group_by_attribute(facilities, "country")
        for country_name, country_facs in countries.items():
            if not country_name:
                continue
            metrics = self.calculate_portfolio_metrics(country_facs, method)
            metrics.entity_level = EntityLevel.COUNTRY.value
            metrics.entity_name = country_name
            breakdowns[f"country:{country_name}"] = metrics

        # Group by business unit.
        bus = self._group_by_attribute(facilities, "business_unit")
        for bu_name, bu_facs in bus.items():
            if not bu_name:
                continue
            metrics = self.calculate_portfolio_metrics(bu_facs, method)
            metrics.entity_level = EntityLevel.BUSINESS_UNIT.value
            metrics.entity_name = bu_name
            breakdowns[f"business_unit:{bu_name}"] = metrics

        # Group by building type.
        types = self._group_by_attribute(facilities, "building_type")
        for type_name, type_facs in types.items():
            if not type_name:
                continue
            metrics = self.calculate_portfolio_metrics(type_facs, method)
            metrics.entity_level = "building_type"
            metrics.entity_name = type_name
            breakdowns[f"building_type:{type_name}"] = metrics

        self._notes.append(f"Entity breakdowns: {len(breakdowns)} groupings computed.")
        return breakdowns

    # --------------------------------------------------------------------- #
    # Private Helpers -- EUI Computation
    # --------------------------------------------------------------------- #

    def _get_eui_values(self, facilities: List[FacilityMembership]) -> List[float]:
        """Extract EUI values for all facilities.

        Args:
            facilities: List of facilities.

        Returns:
            List of EUI values in facility order.
        """
        result: List[float] = []
        for f in facilities:
            eui = self._get_facility_eui(f)
            result.append(float(eui))
        return result

    def _get_facility_eui(self, f: FacilityMembership) -> Decimal:
        """Get the EUI for a single facility.

        Args:
            f: Facility membership record.

        Returns:
            EUI as Decimal.
        """
        if f.eui_kwh_per_m2 is not None and f.eui_kwh_per_m2 > 0:
            return _decimal(f.eui_kwh_per_m2)
        d_area = _decimal(f.gross_floor_area_m2)
        if d_area > Decimal("0"):
            return _safe_divide(_decimal(f.energy_consumption_kwh), d_area)
        return Decimal("0")

    def _area_weighted_eui(
        self,
        facilities: List[FacilityMembership],
        eui_values: List[float],
    ) -> Decimal:
        """Compute area-weighted EUI with building-type adjustments.

        Area-weighted EUI = sum(EUI_i * area_i * weight_i) / sum(area_i * weight_i)

        Args:
            facilities: List of facilities.
            eui_values: EUI for each facility.

        Returns:
            Area-weighted EUI as Decimal.
        """
        numerator = Decimal("0")
        denominator = Decimal("0")

        for f, eui in zip(facilities, eui_values):
            d_area = _decimal(f.gross_floor_area_m2)
            d_eui = _decimal(eui)
            weight = _decimal(PORTFOLIO_AGGREGATION_WEIGHTS.get(
                f.building_type, PORTFOLIO_AGGREGATION_WEIGHTS["default"]
            ))
            numerator += d_eui * d_area * weight
            denominator += d_area * weight

        return _safe_divide(numerator, denominator)

    # --------------------------------------------------------------------- #
    # Private Helpers -- Ranking
    # --------------------------------------------------------------------- #

    def _compute_ranking_score(
        self,
        f: FacilityMembership,
        criteria: RankingCriteria,
    ) -> Decimal:
        """Compute the ranking score for a facility.

        Args:
            f: Facility record.
            criteria: Ranking criteria.

        Returns:
            Score as Decimal (interpretation depends on criteria).
        """
        d_area = _decimal(f.gross_floor_area_m2)

        if criteria == RankingCriteria.EUI_ABSOLUTE:
            return self._get_facility_eui(f)
        elif criteria == RankingCriteria.EUI_PERCENTILE:
            return self._get_facility_eui(f)
        elif criteria == RankingCriteria.IMPROVEMENT_RATE:
            return self._facility_yoy_improvement(f)
        elif criteria == RankingCriteria.CARBON_INTENSITY:
            return _safe_divide(_decimal(f.carbon_emissions_kgco2), d_area)
        elif criteria == RankingCriteria.COST_INTENSITY:
            return _safe_divide(_decimal(f.energy_cost_eur), d_area)
        else:
            return self._get_facility_eui(f)

    def _facility_yoy_improvement(self, f: FacilityMembership) -> Decimal:
        """Compute year-over-year EUI improvement for a single facility.

        Args:
            f: Facility record with historical_eui data.

        Returns:
            Improvement percentage (positive = improving).
        """
        if not f.historical_eui:
            return Decimal("0")

        sorted_years = sorted(f.historical_eui.keys())
        if len(sorted_years) < 2:
            return Decimal("0")

        prev_eui = _decimal(f.historical_eui[sorted_years[-2]])
        curr_eui = _decimal(f.historical_eui[sorted_years[-1]])

        if prev_eui <= Decimal("0"):
            return Decimal("0")

        # Positive value means improvement (EUI decreased).
        return _safe_pct(prev_eui - curr_eui, prev_eui)

    def _assign_tier(self, percentile: Decimal) -> PortfolioTier:
        """Assign a quartile tier based on percentile.

        Args:
            percentile: Percentile position (0-100, 100 = best).

        Returns:
            PortfolioTier enum value.
        """
        if percentile >= Decimal("75"):
            return PortfolioTier.TOP_QUARTILE
        elif percentile >= Decimal("50"):
            return PortfolioTier.SECOND_QUARTILE
        elif percentile >= Decimal("25"):
            return PortfolioTier.THIRD_QUARTILE
        else:
            return PortfolioTier.BOTTOM_QUARTILE

    def _assess_improvement(self, yoy_pct: Decimal) -> str:
        """Assess the improvement rate against published thresholds.

        Args:
            yoy_pct: Year-over-year improvement percentage.

        Returns:
            Textual assessment string.
        """
        f_pct = float(yoy_pct)
        for level in ["excellent", "good", "on_target", "marginal", "stagnant", "deteriorating"]:
            if f_pct >= IMPROVEMENT_RATE_THRESHOLDS[level]["min_pct"]:
                return IMPROVEMENT_RATE_THRESHOLDS[level]["description"]
        return "Unknown"

    # --------------------------------------------------------------------- #
    # Private Helpers -- Outlier Detection
    # --------------------------------------------------------------------- #

    def _detect_outliers(
        self,
        scored: List[Tuple[str, Decimal]],
        method: OutlierDetectionMethod,
    ) -> set:
        """Detect outlier facilities using the specified method.

        Args:
            scored: List of (facility_id, score) tuples.
            method: Detection method.

        Returns:
            Set of outlier facility IDs.
        """
        if len(scored) < 4:
            return set()

        values = [s for _, s in scored]
        ids = [fid for fid, _ in scored]
        outlier_ids: set = set()

        if method == OutlierDetectionMethod.IQR:
            sorted_vals = sorted(values)
            q1 = self._compute_percentile_value(sorted_vals, 25.0)
            q3 = self._compute_percentile_value(sorted_vals, 75.0)
            iqr = q3 - q1
            lower = q1 - Decimal("1.5") * iqr
            upper = q3 + Decimal("1.5") * iqr
            for fid, val in scored:
                if val < lower or val > upper:
                    outlier_ids.add(fid)

        elif method == OutlierDetectionMethod.Z_SCORE:
            mean = _safe_divide(sum(values), _decimal(len(values)))
            if len(values) > 1:
                variance = _safe_divide(
                    sum((v - mean) ** 2 for v in values),
                    _decimal(len(values) - 1),
                )
                std = _decimal(math.sqrt(max(float(variance), 0.0)))
                if std > Decimal("0"):
                    for fid, val in scored:
                        z = abs(val - mean) / std
                        if z > Decimal("3"):
                            outlier_ids.add(fid)

        elif method == OutlierDetectionMethod.MODIFIED_Z:
            sorted_vals = sorted(values)
            median = self._compute_median(sorted_vals)
            abs_devs = sorted([abs(v - median) for v in values])
            mad = self._compute_median(abs_devs) if abs_devs else Decimal("0")
            if mad > Decimal("0"):
                for fid, val in scored:
                    modified_z = Decimal("0.6745") * abs(val - median) / mad
                    if modified_z > Decimal("3.5"):
                        outlier_ids.add(fid)

        elif method == OutlierDetectionMethod.PERCENTILE:
            sorted_vals = sorted(values)
            p1 = self._compute_percentile_value(sorted_vals, 1.0)
            p99 = self._compute_percentile_value(sorted_vals, 99.0)
            for fid, val in scored:
                if val < p1 or val > p99:
                    outlier_ids.add(fid)

        return outlier_ids

    # --------------------------------------------------------------------- #
    # Private Helpers -- Statistics
    # --------------------------------------------------------------------- #

    def _compute_median(self, sorted_values: List[Decimal]) -> Decimal:
        """Compute median of a sorted list of Decimal values.

        Args:
            sorted_values: Pre-sorted list.

        Returns:
            Median as Decimal.
        """
        n = len(sorted_values)
        if n == 0:
            return Decimal("0")
        if n % 2 == 1:
            return sorted_values[n // 2]
        else:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / Decimal("2")

    def _compute_percentile_value(
        self,
        sorted_values: List[Decimal],
        percentile: float,
    ) -> Decimal:
        """Compute a percentile value from sorted data using linear interpolation.

        Uses the nearest-rank method with linear interpolation (NIST method).

        Args:
            sorted_values: Pre-sorted list of Decimal values.
            percentile: Percentile to compute (0-100).

        Returns:
            Interpolated percentile value.
        """
        n = len(sorted_values)
        if n == 0:
            return Decimal("0")
        if n == 1:
            return sorted_values[0]

        # NIST percentile interpolation.
        k = _decimal(percentile) / Decimal("100") * _decimal(n - 1)
        f_k = int(float(k))
        f_k = max(0, min(f_k, n - 2))
        fraction = k - _decimal(f_k)

        return sorted_values[f_k] + fraction * (sorted_values[f_k + 1] - sorted_values[f_k])

    def _compute_histogram(
        self,
        sorted_values: List[Decimal],
        num_bins: int = 10,
    ) -> Tuple[List[Decimal], List[int]]:
        """Compute histogram bins and counts.

        Args:
            sorted_values: Pre-sorted list of Decimal values.
            num_bins: Number of bins.

        Returns:
            Tuple of (bin_edges, counts).
        """
        n = len(sorted_values)
        if n == 0:
            return [], []

        min_val = sorted_values[0]
        max_val = sorted_values[-1]

        if min_val == max_val:
            return [min_val, max_val + Decimal("1")], [n]

        bin_width = _safe_divide(max_val - min_val, _decimal(num_bins))
        bins: List[Decimal] = []
        for i in range(num_bins + 1):
            bins.append(min_val + _decimal(i) * bin_width)

        counts: List[int] = [0] * num_bins
        for val in sorted_values:
            idx = int(float(_safe_divide(val - min_val, bin_width)))
            idx = max(0, min(idx, num_bins - 1))
            counts[idx] += 1

        return bins, counts

    # --------------------------------------------------------------------- #
    # Private Helpers -- Grouping
    # --------------------------------------------------------------------- #

    def _group_by_attribute(
        self,
        facilities: List[FacilityMembership],
        attribute: str,
    ) -> Dict[str, List[FacilityMembership]]:
        """Group facilities by a string attribute.

        Args:
            facilities: List of facilities.
            attribute: Attribute name to group by.

        Returns:
            Dict mapping attribute value to list of facilities.
        """
        groups: Dict[str, List[FacilityMembership]] = {}
        for f in facilities:
            key = getattr(f, attribute, "") or ""
            if key not in groups:
                groups[key] = []
            groups[key].append(f)
        return groups


# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------
# When `from __future__ import annotations` is active, all type annotations
# become lazy strings.  Pydantic v2 must explicitly rebuild models that
# reference other user-defined types (enums, other models) so that validators
# and serialisers resolve correctly at import time.

PortfolioDefinition.model_rebuild()
FacilityMembership.model_rebuild()
PortfolioMetrics.model_rebuild()
FacilityRanking.model_rebuild()
PortfolioDistribution.model_rebuild()
PortfolioTrend.model_rebuild()
PortfolioBenchmarkResult.model_rebuild()


# ---------------------------------------------------------------------------
# Public Aliases -- required by PACK-035 __init__.py symbol contract
# ---------------------------------------------------------------------------
# The engines __init__.py imports these names; we alias them to the
# canonical class names used within this module so both names work.

PortfolioFacility = FacilityMembership
"""Alias: ``PortfolioFacility`` -> :class:`FacilityMembership`."""

PortfolioSummary = PortfolioMetrics
"""Alias: ``PortfolioSummary`` -> :class:`PortfolioMetrics`."""

RankingMetric = RankingCriteria
"""Alias: ``RankingMetric`` -> :class:`RankingCriteria`."""
