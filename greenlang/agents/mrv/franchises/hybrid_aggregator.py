# -*- coding: utf-8 -*-
"""
HybridAggregatorEngine - AGENT-MRV-027 Engine 5

GHG Protocol Scope 3 Category 14 hybrid aggregation engine that combines
franchise-specific (Tier 1), average-data (Tier 2), and spend-based (Tier 3)
calculation methods using a waterfall approach across a franchise network.

This engine orchestrates the complete franchise network emissions calculation
by routing each unit to the appropriate tier based on data availability:

Method Waterfall (per unit):
    1. If metered energy data available -> franchise_specific (Tier 1)
    2. If floor area + type known       -> average_data (Tier 2)
    3. Fallback                         -> spend_based (Tier 3)

Tiered Data Collection Strategy:
    - Tier 1 targets: Top 20% of units by revenue/size (should provide metered)
    - Tier 2 targets: Next 30% (provide basic area/type info)
    - Tier 3 remainder: Bottom 50% (use spend/benchmark defaults)

Network-Level Features:
    - Multi-brand support (single franchisor, multiple brands)
    - Regional aggregation (by country, state/province)
    - Franchise type aggregation
    - Year-over-year comparison
    - Data quality improvement tracking
    - Company-owned vs franchised split (DC-FRN-001)
    - Weighted data quality indicator (DQI) across network
    - Data coverage and method distribution reporting
    - Uncertainty propagation across tiers

DC-FRN-001: Company-owned units are EXCLUDED from Category 14. They belong
in Scope 1/2 of the franchisor. This engine enforces this boundary by
splitting company-owned units out before calculation.

References:
    - GHG Protocol Technical Guidance for Scope 3, Category 14 (Franchises)
    - GHG Protocol Scope 3 Standard, Chapter 14
    - WBCSD Value Chain (Scope 3) Accounting and Reporting Standard
    - GHG Protocol Scope 3 Calculation Guidance

Example:
    >>> engine = get_hybrid_aggregator()
    >>> result = engine.calculate(FranchiseNetworkInput(
    ...     network_id="NET-001",
    ...     franchisor_name="FastBurger Inc.",
    ...     units=[unit1, unit2, unit3],
    ...     reporting_year=2024,
    ... ))
    >>> result.total_co2e > Decimal("0")
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-014
"""

import hashlib
import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "hybrid_aggregator_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-014"
AGENT_COMPONENT: str = "AGENT-MRV-027"
TABLE_PREFIX: str = "gl_frn_"

# Decimal precision
PRECISION: int = 8
ROUNDING: str = ROUND_HALF_UP
_QUANT_8DP: Decimal = Decimal("0.00000001")
_QUANT_4DP: Decimal = Decimal("0.0001")
_QUANT_2DP: Decimal = Decimal("0.01")

_ZERO: Decimal = Decimal("0")
_ONE: Decimal = Decimal("1")
_HUNDRED: Decimal = Decimal("100")
_THOUSAND: Decimal = Decimal("1000")

# Tiered data collection targets
TIER_1_TARGET_FRACTION: Decimal = Decimal("0.20")  # Top 20%
TIER_2_TARGET_FRACTION: Decimal = Decimal("0.30")  # Next 30%
TIER_3_REMAINDER_FRACTION: Decimal = Decimal("0.50")  # Bottom 50%


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class CalculationMethod(str, Enum):
    """Calculation method tier classification."""

    FRANCHISE_SPECIFIC = "franchise_specific"  # Tier 1: metered data
    AVERAGE_DATA = "average_data"              # Tier 2: EUI/revenue benchmarks
    SPEND_BASED = "spend_based"                # Tier 3: EEIO spend-based
    HYBRID = "hybrid"                          # Network-level aggregated


class OwnershipType(str, Enum):
    """Unit ownership classification for DC-FRN-001 boundary."""

    FRANCHISED = "franchised"         # Franchised unit (Cat 14 scope)
    COMPANY_OWNED = "company_owned"   # Company-owned (Scope 1/2 only)


class DataQualityTier(str, Enum):
    """Data quality tiers."""

    TIER_1 = "tier_1"  # Primary metered data
    TIER_2 = "tier_2"  # Area/revenue benchmarks
    TIER_3 = "tier_3"  # Spend-based / EEIO


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol."""

    REPRESENTATIVENESS = "representativeness"
    COMPLETENESS = "completeness"
    TEMPORAL = "temporal"
    GEOGRAPHICAL = "geographical"
    TECHNOLOGICAL = "technological"


class FranchiseType(str, Enum):
    """Franchise type classification."""

    QSR = "qsr"
    FULL_SERVICE_RESTAURANT = "full_service_restaurant"
    HOTEL = "hotel"
    CONVENIENCE_STORE = "convenience_store"
    RETAIL_CLOTHING = "retail_clothing"
    FITNESS_CENTER = "fitness_center"
    AUTOMOTIVE_REPAIR = "automotive_repair"
    HEALTHCARE_CLINIC = "healthcare_clinic"
    EDUCATION_CENTER = "education_center"
    COFFEE_SHOP = "coffee_shop"


class EFSource(str, Enum):
    """Emission factor data source."""

    METERED = "metered"
    EIA_CBECS = "eia_cbecs"
    EEIO = "eeio"
    MIXED = "mixed"


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"
    JPY = "JPY"
    CNY = "CNY"
    INR = "INR"
    CHF = "CHF"
    SGD = "SGD"
    BRL = "BRL"
    ZAR = "ZAR"
    MXN = "MXN"
    KRW = "KRW"
    NZD = "NZD"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    AED = "AED"
    SAR = "SAR"


# DQI dimension weights (sum to 1.0)
DQI_WEIGHTS: Dict[DQIDimension, Decimal] = {
    DQIDimension.REPRESENTATIVENESS: Decimal("0.30"),
    DQIDimension.COMPLETENESS: Decimal("0.25"),
    DQIDimension.TEMPORAL: Decimal("0.15"),
    DQIDimension.GEOGRAPHICAL: Decimal("0.15"),
    DQIDimension.TECHNOLOGICAL: Decimal("0.15"),
}

# Uncertainty half-widths (95% CI) per tier
TIER_UNCERTAINTY: Dict[DataQualityTier, Decimal] = {
    DataQualityTier.TIER_1: Decimal("0.10"),   # +/- 10%
    DataQualityTier.TIER_2: Decimal("0.30"),   # +/- 30%
    DataQualityTier.TIER_3: Decimal("0.50"),   # +/- 50%
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _quantize_8dp(value: Decimal) -> Decimal:
    """Quantize a Decimal to 8 decimal places with ROUND_HALF_UP."""
    return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


def _quantize_4dp(value: Decimal) -> Decimal:
    """Quantize a Decimal to 4 decimal places with ROUND_HALF_UP."""
    return value.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)


def _calculate_provenance_hash(*inputs: Any) -> str:
    """Calculate SHA-256 provenance hash from variable inputs."""
    hash_input = ""
    for inp in inputs:
        if isinstance(inp, BaseModel):
            hash_input += json.dumps(
                inp.model_dump(mode="json"), sort_keys=True, default=str
            )
        elif isinstance(inp, Decimal):
            hash_input += str(inp.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP))
        else:
            hash_input += str(inp)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def _get_dqi_classification(score: Decimal) -> str:
    """Classify a composite DQI score into a human-readable label."""
    if score >= Decimal("4.5"):
        return "Excellent"
    elif score >= Decimal("3.5"):
        return "Good"
    elif score >= Decimal("2.5"):
        return "Fair"
    elif score >= Decimal("1.5"):
        return "Poor"
    else:
        return "Very Poor"


# ==============================================================================
# METRICS AND PROVENANCE STUBS
# ==============================================================================


class _MetricsCollectorStub:
    """Minimal metrics stub when full metrics module is not available."""

    def record_calculation(self, **kwargs: Any) -> None:
        """No-op."""
        pass

    def record_batch(self, **kwargs: Any) -> None:
        """No-op."""
        pass

    def record_network_aggregation(self, **kwargs: Any) -> None:
        """No-op."""
        pass


def get_metrics_collector() -> Any:
    """Get the metrics collector for the Franchises agent."""
    try:
        from greenlang.agents.mrv.franchises.metrics import get_metrics
        return get_metrics()
    except (ImportError, Exception):
        return _MetricsCollectorStub()


class _ProvenanceManagerStub:
    """Minimal provenance stub."""

    def start_chain(self) -> str:
        """Return placeholder chain ID."""
        import uuid
        return str(uuid.uuid4())

    def record_stage(self, chain_id: str, stage: str,
                     input_data: Any, output_data: Any) -> None:
        """No-op."""
        pass

    def seal_chain(self, chain_id: str) -> str:
        """Return placeholder hash."""
        return hashlib.sha256(chain_id.encode("utf-8")).hexdigest()


def get_provenance_manager() -> Any:
    """Get the provenance manager for the Franchises agent."""
    try:
        from greenlang.agents.mrv.franchises.provenance import get_provenance_tracker
        return get_provenance_tracker()
    except (ImportError, Exception):
        return _ProvenanceManagerStub()


# ==============================================================================
# INPUT / OUTPUT MODELS
# ==============================================================================


class FranchiseUnitData(BaseModel):
    """
    Individual franchise unit data for hybrid aggregation routing.

    Fields populated determine which calculation tier is used via the
    method waterfall. Company-owned units are excluded (DC-FRN-001).

    Example:
        >>> unit = FranchiseUnitData(
        ...     unit_id="FRN-001",
        ...     unit_name="FastBurger #1234",
        ...     franchise_type=FranchiseType.QSR,
        ...     ownership=OwnershipType.FRANCHISED,
        ...     country="US",
        ...     state_province="CA",
        ...     floor_area_m2=Decimal("250"),
        ...     climate_zone="4A",
        ...     grid_region="US_CALIFORNIA",
        ... )
    """

    unit_id: str = Field(
        ..., min_length=1, max_length=128,
        description="Unique identifier for the franchise unit"
    )
    unit_name: Optional[str] = Field(
        default=None, max_length=256,
        description="Human-readable unit name"
    )
    franchise_type: FranchiseType = Field(
        ..., description="Type of franchise"
    )
    brand: Optional[str] = Field(
        default=None, max_length=128,
        description="Brand name (for multi-brand franchisors)"
    )
    ownership: OwnershipType = Field(
        default=OwnershipType.FRANCHISED,
        description="Ownership type (franchised or company-owned)"
    )
    # Location
    country: Optional[str] = Field(
        default=None, max_length=3,
        description="ISO 3166-1 alpha-2 country code"
    )
    state_province: Optional[str] = Field(
        default=None, max_length=64,
        description="State/province/region"
    )
    # Tier 1 fields (franchise-specific metered data)
    has_metered_data: bool = Field(
        default=False,
        description="Whether this unit has metered energy/emissions data"
    )
    metered_co2e: Optional[Decimal] = Field(
        default=None, ge=_ZERO,
        description="Metered/franchise-specific emissions (kgCO2e) if available"
    )
    metered_data_quality_score: Optional[Decimal] = Field(
        default=None, ge=_ONE, le=Decimal("5"),
        description="DQI score for metered data (1-5)"
    )
    # Tier 2 fields (average-data)
    floor_area_m2: Optional[Decimal] = Field(
        default=None, gt=_ZERO,
        description="Floor area in square metres"
    )
    climate_zone: Optional[str] = Field(
        default=None, max_length=4,
        description="ASHRAE climate zone code (e.g., '4A')"
    )
    grid_region: Optional[str] = Field(
        default=None, max_length=32,
        description="Grid region code for electricity EF lookup"
    )
    annual_revenue: Optional[Decimal] = Field(
        default=None, gt=_ZERO,
        description="Annual revenue for revenue-based estimation"
    )
    revenue_currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of the annual revenue"
    )
    # Partial year
    months_operational: int = Field(
        default=12, ge=1, le=12,
        description="Months operational in reporting year"
    )
    # Type-specific overrides (as dicts for flexibility)
    type_specific_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Type-specific parameters (hotel_ops, qsr_cooking, etc.)"
    )

    model_config = ConfigDict(frozen=True)


class HybridNetworkInput(BaseModel):
    """
    Input for hybrid franchise network emissions calculation.

    Contains the list of units and optional network-level spend data
    for Tier 3 fallback.

    DC-FRN-001: Units with ownership=COMPANY_OWNED will be automatically
    excluded from Category 14 calculations.

    Example:
        >>> network = HybridNetworkInput(
        ...     network_id="NET-001",
        ...     franchisor_name="FastBurger Inc.",
        ...     units=[unit1, unit2, unit3],
        ...     reporting_year=2024,
        ... )
    """

    network_id: str = Field(
        ..., min_length=1, max_length=128,
        description="Unique identifier for the franchise network"
    )
    franchisor_name: str = Field(
        ..., min_length=1, max_length=256,
        description="Name of the franchisor entity"
    )
    units: List[FranchiseUnitData] = Field(
        ..., min_length=1,
        description="List of franchise units to process"
    )
    # Network-level spend data (Tier 3 fallback)
    naics_code: Optional[str] = Field(
        default=None, min_length=5, max_length=8,
        description="NAICS code for spend-based Tier 3 fallback"
    )
    network_total_revenue: Optional[Decimal] = Field(
        default=None, gt=_ZERO,
        description="Total network revenue (for Tier 3 fallback)"
    )
    network_currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of network-level financial data"
    )
    # Reporting
    reporting_year: int = Field(
        default=2024, ge=2015, le=2030,
        description="Reporting year"
    )
    # Comparison
    prior_year_co2e: Optional[Decimal] = Field(
        default=None, ge=_ZERO,
        description="Prior year total emissions for YoY comparison (kgCO2e)"
    )
    prior_year: Optional[int] = Field(
        default=None, ge=2014, le=2029,
        description="Prior reporting year"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)


class UnitCalculationResult(BaseModel):
    """Result from calculating a single franchise unit."""

    unit_id: str = Field(..., description="Unit identifier")
    unit_name: Optional[str] = Field(default=None, description="Unit name")
    franchise_type: FranchiseType = Field(
        ..., description="Franchise type"
    )
    brand: Optional[str] = Field(default=None, description="Brand name")
    method: CalculationMethod = Field(
        ..., description="Calculation method used"
    )
    tier: DataQualityTier = Field(
        ..., description="Data quality tier"
    )
    total_co2e: Decimal = Field(
        ..., description="Total emissions (kgCO2e)"
    )
    dqi_score: Decimal = Field(
        ..., description="Data quality indicator score (1-5)"
    )
    country: Optional[str] = Field(default=None, description="Country code")
    state_province: Optional[str] = Field(
        default=None, description="State/province"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class DataQualityScore(BaseModel):
    """Data quality assessment result."""

    overall_score: Decimal = Field(
        ..., description="Weighted composite DQI score (1.0 - 5.0)"
    )
    tier: DataQualityTier = Field(
        ..., description="Data quality tier"
    )
    dimensions: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Score per DQI dimension"
    )
    classification: str = Field(
        ..., description="Quality label"
    )

    model_config = ConfigDict(frozen=True)


class MethodBreakdown(BaseModel):
    """Breakdown of calculation methods used across the network."""

    tier_1_count: int = Field(default=0, description="Units using Tier 1")
    tier_1_co2e: Decimal = Field(default=_ZERO, description="Tier 1 emissions (kgCO2e)")
    tier_1_pct: Decimal = Field(default=_ZERO, description="Tier 1 percentage of units")
    tier_2_count: int = Field(default=0, description="Units using Tier 2")
    tier_2_co2e: Decimal = Field(default=_ZERO, description="Tier 2 emissions (kgCO2e)")
    tier_2_pct: Decimal = Field(default=_ZERO, description="Tier 2 percentage of units")
    tier_3_count: int = Field(default=0, description="Units using Tier 3")
    tier_3_co2e: Decimal = Field(default=_ZERO, description="Tier 3 emissions (kgCO2e)")
    tier_3_pct: Decimal = Field(default=_ZERO, description="Tier 3 percentage of units")
    total_units: int = Field(default=0, description="Total franchised units calculated")

    model_config = ConfigDict(frozen=True)


class DataCoverageReport(BaseModel):
    """Report on data coverage and quality across the franchise network."""

    total_units_submitted: int = Field(
        ..., description="Total units submitted (including company-owned)"
    )
    company_owned_excluded: int = Field(
        ..., description="Company-owned units excluded (DC-FRN-001)"
    )
    franchised_calculated: int = Field(
        ..., description="Franchised units calculated"
    )
    calculation_errors: int = Field(
        default=0, description="Units that failed calculation"
    )
    method_breakdown: MethodBreakdown = Field(
        ..., description="Breakdown by calculation tier"
    )
    brands: List[str] = Field(
        default_factory=list, description="Brands represented"
    )
    countries: List[str] = Field(
        default_factory=list, description="Countries represented"
    )
    tier_1_coverage_pct: Decimal = Field(
        ..., description="Percentage of units with Tier 1 data"
    )
    tier_2_coverage_pct: Decimal = Field(
        ..., description="Percentage of units with Tier 2 data"
    )
    tier_3_coverage_pct: Decimal = Field(
        ..., description="Percentage of units with Tier 3 data"
    )
    meets_tier_1_target: bool = Field(
        ..., description="Whether Tier 1 coverage >= 20% target"
    )
    meets_tier_2_target: bool = Field(
        ..., description="Whether Tier 1+2 coverage >= 50% target"
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyResult(BaseModel):
    """Uncertainty quantification for network-level emissions."""

    mean_co2e: Decimal = Field(
        ..., description="Mean emissions estimate (kgCO2e)"
    )
    ci_lower: Decimal = Field(
        ..., description="Lower bound of 95% CI (kgCO2e)"
    )
    ci_upper: Decimal = Field(
        ..., description="Upper bound of 95% CI (kgCO2e)"
    )
    relative_uncertainty_pct: Decimal = Field(
        ..., description="Relative uncertainty as percentage of mean"
    )
    method: str = Field(
        default="tier_weighted_propagation",
        description="Uncertainty method used"
    )

    model_config = ConfigDict(frozen=True)


class YearOverYearComparison(BaseModel):
    """Year-over-year emissions comparison."""

    current_year: int = Field(..., description="Current reporting year")
    prior_year: int = Field(..., description="Prior reporting year")
    current_co2e: Decimal = Field(
        ..., description="Current year emissions (kgCO2e)"
    )
    prior_co2e: Decimal = Field(
        ..., description="Prior year emissions (kgCO2e)"
    )
    absolute_change: Decimal = Field(
        ..., description="Absolute change (kgCO2e)"
    )
    pct_change: Decimal = Field(
        ..., description="Percentage change"
    )
    direction: str = Field(
        ..., description="increase, decrease, or stable"
    )

    model_config = ConfigDict(frozen=True)


class NetworkAggregationResult(BaseModel):
    """
    Complete result from hybrid franchise network emissions calculation.

    Contains network-level totals, per-unit results, regional aggregation,
    method breakdown, data coverage, uncertainty, and provenance chain.
    """

    network_id: str = Field(..., description="Network identifier")
    franchisor_name: str = Field(..., description="Franchisor name")
    reporting_year: int = Field(..., description="Reporting year")
    # Emissions
    total_co2e: Decimal = Field(
        ..., description="Total network emissions (kgCO2e)"
    )
    total_tco2e: Decimal = Field(
        ..., description="Total network emissions (tCO2e)"
    )
    # Per-unit results
    unit_results: List[UnitCalculationResult] = Field(
        default_factory=list,
        description="Individual unit calculation results"
    )
    # Aggregations
    by_region: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by country (kgCO2e)"
    )
    by_franchise_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by franchise type (kgCO2e)"
    )
    by_brand: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by brand (kgCO2e)"
    )
    # Method and coverage
    method_breakdown: MethodBreakdown = Field(
        ..., description="Tier-level method distribution"
    )
    data_coverage: DataCoverageReport = Field(
        ..., description="Data coverage and quality report"
    )
    # Quality and uncertainty
    weighted_dqi: DataQualityScore = Field(
        ..., description="Emissions-weighted DQI across network"
    )
    uncertainty: UncertaintyResult = Field(
        ..., description="Network-level uncertainty quantification"
    )
    # Year-over-year
    yoy_comparison: Optional[YearOverYearComparison] = Field(
        default=None,
        description="Year-over-year comparison if prior data provided"
    )
    # Provenance
    ef_source: EFSource = Field(
        default=EFSource.MIXED,
        description="Primary emission factor source"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculation_timestamp: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )
    engine_version: str = Field(
        default=ENGINE_VERSION,
        description="Engine version"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Validation warnings"
    )
    errors: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Per-unit errors from failed calculations"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HybridAggregatorEngine
# ==============================================================================


class HybridAggregatorEngine:
    """
    Hybrid aggregation engine for franchise network emissions.

    Orchestrates the complete franchise network Scope 3 Category 14
    calculation by routing each unit to the appropriate calculation tier
    and aggregating results across the network.

    Method Waterfall (per unit):
        1. has_metered_data=True + metered_co2e -> franchise_specific (Tier 1)
        2. floor_area_m2 + climate_zone + grid_region -> average_data (Tier 2)
        3. annual_revenue (or network spend fallback) -> spend_based (Tier 3)

    Network Aggregation:
        - Sum unit emissions for total network CO2e
        - Aggregate by region, franchise type, and brand
        - Calculate method distribution (% Tier 1 vs 2 vs 3)
        - Calculate emissions-weighted data quality
        - Generate data coverage report with target comparison
        - Propagate uncertainty across tiers

    DC-FRN-001 Enforcement:
        Company-owned units (ownership=COMPANY_OWNED) are automatically
        excluded from the calculation. They are counted in the data
        coverage report but not in emissions.

    Thread Safety:
        Singleton pattern with threading.RLock for concurrent access.

    Attributes:
        _metrics: Prometheus metrics collector
        _provenance: Provenance tracking manager
        _calculation_count: Running count of network calculations
        _total_units_processed: Running count of units processed

    Example:
        >>> engine = HybridAggregatorEngine.get_instance()
        >>> result = engine.calculate(HybridNetworkInput(
        ...     network_id="NET-001",
        ...     franchisor_name="FastBurger Inc.",
        ...     units=[unit1, unit2, unit3],
        ...     reporting_year=2024,
        ... ))
        >>> result.total_co2e > Decimal("0")
        True
    """

    _instance: Optional["HybridAggregatorEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __init__(self) -> None:
        """Initialize HybridAggregatorEngine with metrics and provenance."""
        self._metrics = get_metrics_collector()
        self._provenance = get_provenance_manager()
        self._calculation_count: int = 0
        self._total_units_processed: int = 0

        logger.info(
            "HybridAggregatorEngine initialized: version=%s, agent=%s",
            ENGINE_VERSION, AGENT_ID,
        )

    @classmethod
    def get_instance(cls) -> "HybridAggregatorEngine":
        """
        Get singleton instance (thread-safe double-checked locking).

        Returns:
            HybridAggregatorEngine singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing only)."""
        with cls._lock:
            cls._instance = None
            logger.info("HybridAggregatorEngine singleton reset")

    # ==========================================================================
    # Primary Public Methods
    # ==========================================================================

    def calculate(
        self, network_input: HybridNetworkInput
    ) -> NetworkAggregationResult:
        """
        Calculate emissions for the full franchise network using hybrid method.

        This is the main entry point. It:
          1. Splits company-owned from franchised units (DC-FRN-001)
          2. Routes each franchised unit to the appropriate tier
          3. Calculates emissions per unit
          4. Aggregates results across the network
          5. Generates data coverage and quality reports
          6. Computes network-level uncertainty

        Args:
            network_input: Full franchise network input with unit data.

        Returns:
            NetworkAggregationResult with complete network analysis.

        Example:
            >>> result = engine.calculate(network_input)
            >>> print(f"Total: {result.total_tco2e} tCO2e")
        """
        start_time = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()
        warnings: List[str] = []
        errors: List[Dict[str, str]] = []

        logger.info(
            "Starting hybrid network calculation: network=%s, units=%d",
            network_input.network_id, len(network_input.units),
        )

        # Step 1: Split company-owned vs franchised (DC-FRN-001)
        franchised_units, company_owned_units = self._split_company_owned(
            network_input.units
        )

        if company_owned_units:
            warnings.append(
                f"DC-FRN-001: {len(company_owned_units)} company-owned units "
                f"excluded from Category 14 (reported in Scope 1/2)"
            )
            logger.info(
                "DC-FRN-001: Excluded %d company-owned units",
                len(company_owned_units),
            )

        # Step 2: Handle partial-year units
        franchised_units = self._handle_partial_year_units(franchised_units)

        # Step 3: Calculate per-unit emissions via waterfall
        unit_results: List[UnitCalculationResult] = []
        for unit in franchised_units:
            try:
                result = self._calculate_unit(
                    unit, network_input.reporting_year
                )
                unit_results.append(result)
            except Exception as e:
                errors.append({
                    "unit_id": unit.unit_id,
                    "error": str(e),
                })
                logger.error(
                    "Unit calculation failed: unit=%s, error=%s",
                    unit.unit_id, str(e),
                )

        # Step 4: Aggregate results
        aggregated = self._aggregate_results(unit_results)

        # Step 5: Generate method breakdown
        method_breakdown = self._generate_method_breakdown(unit_results)

        # Step 6: Aggregate by dimensions
        by_region = self._aggregate_by_region(unit_results)
        by_type = self._aggregate_by_franchise_type(unit_results)
        by_brand = self._aggregate_by_brand(unit_results)

        # Step 7: Calculate weighted DQI
        weighted_dqi = self._calculate_weighted_dqi(unit_results)

        # Step 8: Data coverage report
        data_coverage = self._generate_coverage_report(
            total_submitted=len(network_input.units),
            company_owned=len(company_owned_units),
            franchised_calculated=len(unit_results),
            errors_count=len(errors),
            method_breakdown=method_breakdown,
            unit_results=unit_results,
        )

        # Step 9: Uncertainty
        uncertainty = self._calculate_uncertainty(unit_results, aggregated)

        # Step 10: Year-over-year comparison
        yoy = None
        if (network_input.prior_year_co2e is not None
                and network_input.prior_year is not None):
            yoy = self._calculate_yoy(
                current_co2e=aggregated,
                prior_co2e=network_input.prior_year_co2e,
                current_year=network_input.reporting_year,
                prior_year=network_input.prior_year,
            )

        # Step 11: Provenance
        total_tco2e = _quantize_8dp(aggregated / _THOUSAND)
        provenance_hash = _calculate_provenance_hash(
            network_input.network_id,
            network_input.reporting_year,
            aggregated,
            len(unit_results),
            method_breakdown.tier_1_count,
            method_breakdown.tier_2_count,
            method_breakdown.tier_3_count,
        )

        # Record metrics
        duration = time.monotonic() - start_time
        self._calculation_count += 1
        self._total_units_processed += len(unit_results)

        try:
            self._metrics.record_network_aggregation(
                network_id=network_input.network_id,
                total_units=len(unit_results),
                total_co2e=float(aggregated),
                duration=duration,
            )
        except Exception as e:
            logger.warning("Failed to record metrics: %s", e)

        result = NetworkAggregationResult(
            network_id=network_input.network_id,
            franchisor_name=network_input.franchisor_name,
            reporting_year=network_input.reporting_year,
            total_co2e=aggregated,
            total_tco2e=total_tco2e,
            unit_results=unit_results,
            by_region=by_region,
            by_franchise_type=by_type,
            by_brand=by_brand,
            method_breakdown=method_breakdown,
            data_coverage=data_coverage,
            weighted_dqi=weighted_dqi,
            uncertainty=uncertainty,
            yoy_comparison=yoy,
            ef_source=EFSource.MIXED,
            provenance_hash=provenance_hash,
            calculation_timestamp=timestamp,
            engine_version=ENGINE_VERSION,
            warnings=warnings,
            errors=errors,
        )

        logger.info(
            "Hybrid network calculation complete: network=%s, units=%d/%d, "
            "total_co2e=%s kgCO2e (%.2f tCO2e), T1/T2/T3=%d/%d/%d, "
            "duration=%.4fs",
            network_input.network_id,
            len(unit_results), len(network_input.units),
            aggregated, float(total_tco2e),
            method_breakdown.tier_1_count,
            method_breakdown.tier_2_count,
            method_breakdown.tier_3_count,
            duration,
        )

        return result

    def calculate_network_summary(
        self, network_input: HybridNetworkInput
    ) -> NetworkAggregationResult:
        """
        Convenience alias for calculate().

        Provided for API symmetry with other engines that distinguish between
        per-unit and network-level entry points.

        Args:
            network_input: Full franchise network input.

        Returns:
            NetworkAggregationResult.
        """
        return self.calculate(network_input)

    # ==========================================================================
    # Method Waterfall and Unit Calculation
    # ==========================================================================

    def _route_to_method(
        self, unit: FranchiseUnitData
    ) -> CalculationMethod:
        """
        Determine the calculation method for a unit via the waterfall.

        Waterfall:
            1. has_metered_data=True + metered_co2e -> FRANCHISE_SPECIFIC
            2. floor_area_m2 + climate_zone + grid_region -> AVERAGE_DATA
            3. annual_revenue or fallback -> SPEND_BASED

        Args:
            unit: Franchise unit data.

        Returns:
            CalculationMethod to use for this unit.
        """
        # Tier 1: Franchise-specific metered data
        if unit.has_metered_data and unit.metered_co2e is not None:
            return CalculationMethod.FRANCHISE_SPECIFIC

        # Tier 2: Area-based (preferred) or revenue-based
        if (unit.floor_area_m2 is not None
                and unit.climate_zone is not None
                and unit.grid_region is not None):
            return CalculationMethod.AVERAGE_DATA

        # Tier 2 fallback: Revenue-based
        if unit.annual_revenue is not None:
            return CalculationMethod.AVERAGE_DATA

        # Tier 3: Spend-based fallback
        return CalculationMethod.SPEND_BASED

    def _calculate_unit(
        self,
        unit: FranchiseUnitData,
        reporting_year: int,
    ) -> UnitCalculationResult:
        """
        Calculate emissions for a single unit based on routed method.

        Args:
            unit: Franchise unit data.
            reporting_year: Reporting year.

        Returns:
            UnitCalculationResult with emissions and metadata.
        """
        method = self._route_to_method(unit)

        if method == CalculationMethod.FRANCHISE_SPECIFIC:
            co2e, dqi_score, tier = self._calc_tier_1(unit)
        elif method == CalculationMethod.AVERAGE_DATA:
            co2e, dqi_score, tier = self._calc_tier_2(unit, reporting_year)
        else:
            co2e, dqi_score, tier = self._calc_tier_3(unit, reporting_year)

        # Apply partial-year proration
        if unit.months_operational < 12:
            proration = Decimal(str(unit.months_operational)) / Decimal("12")
            co2e = _quantize_8dp(co2e * proration)

        provenance_hash = _calculate_provenance_hash(
            unit.unit_id, method.value, co2e, dqi_score,
        )

        return UnitCalculationResult(
            unit_id=unit.unit_id,
            unit_name=unit.unit_name,
            franchise_type=unit.franchise_type,
            brand=unit.brand,
            method=method,
            tier=tier,
            total_co2e=co2e,
            dqi_score=dqi_score,
            country=unit.country,
            state_province=unit.state_province,
            provenance_hash=provenance_hash,
        )

    def _calc_tier_1(
        self, unit: FranchiseUnitData
    ) -> Tuple[Decimal, Decimal, DataQualityTier]:
        """
        Tier 1: Use franchise-specific metered data directly.

        Args:
            unit: Unit with metered data.

        Returns:
            Tuple of (co2e, dqi_score, tier).
        """
        co2e = _quantize_8dp(unit.metered_co2e or _ZERO)
        dqi = unit.metered_data_quality_score or Decimal("4.5")

        logger.debug(
            "Tier 1 (metered): unit=%s, co2e=%s kgCO2e, dqi=%s",
            unit.unit_id, co2e, dqi,
        )

        return co2e, dqi, DataQualityTier.TIER_1

    def _calc_tier_2(
        self,
        unit: FranchiseUnitData,
        reporting_year: int,
    ) -> Tuple[Decimal, Decimal, DataQualityTier]:
        """
        Tier 2: Calculate using average-data engine (area or revenue based).

        Delegates to AverageDataCalculatorEngine if available, otherwise
        uses inline calculation with the same logic.

        Args:
            unit: Unit with area/type data.
            reporting_year: Reporting year.

        Returns:
            Tuple of (co2e, dqi_score, tier).
        """
        try:
            from greenlang.agents.mrv.franchises.average_data_calculator import (
                AverageDataCalculatorEngine,
                FranchiseUnitInput as AvgInput,
                FranchiseType as AvgFranchiseType,
                ClimateZone as AvgClimateZone,
                CurrencyCode as AvgCurrency,
            )

            # Map enums
            avg_type = AvgFranchiseType(unit.franchise_type.value)
            climate_zone = None
            if unit.climate_zone is not None:
                try:
                    climate_zone = AvgClimateZone(unit.climate_zone)
                except ValueError:
                    climate_zone = None

            currency = AvgCurrency.USD
            if unit.revenue_currency is not None:
                try:
                    currency = AvgCurrency(unit.revenue_currency.value)
                except ValueError:
                    currency = AvgCurrency.USD

            avg_input = AvgInput(
                unit_id=unit.unit_id,
                franchise_type=avg_type,
                floor_area_m2=unit.floor_area_m2,
                climate_zone=climate_zone,
                grid_region=unit.grid_region,
                annual_revenue=unit.annual_revenue,
                revenue_currency=currency,
                reporting_year=reporting_year,
                months_operational=12,  # Proration handled by caller
            )

            engine = AverageDataCalculatorEngine.get_instance()
            result = engine.calculate(avg_input)

            co2e = result.total_co2e
            dqi = result.data_quality.overall_score

        except (ImportError, Exception) as e:
            logger.warning(
                "AverageDataCalculatorEngine not available, using inline "
                "Tier 2 fallback: %s", e
            )
            co2e, dqi = self._inline_tier_2(unit, reporting_year)

        logger.debug(
            "Tier 2 (average-data): unit=%s, co2e=%s kgCO2e, dqi=%s",
            unit.unit_id, co2e, dqi,
        )

        return co2e, dqi, DataQualityTier.TIER_2

    def _inline_tier_2(
        self,
        unit: FranchiseUnitData,
        reporting_year: int,
    ) -> Tuple[Decimal, Decimal]:
        """
        Inline Tier 2 calculation fallback when engine is unavailable.

        Uses simplified EUI benchmark x area x grid EF formula.

        Args:
            unit: Unit with area/type data.
            reporting_year: Reporting year.

        Returns:
            Tuple of (co2e, dqi_score).
        """
        # Simplified EUI benchmark (median across climate zones, kWh/m2/yr)
        eui_defaults: Dict[str, Decimal] = {
            "qsr": Decimal("960"),
            "full_service_restaurant": Decimal("740"),
            "hotel": Decimal("340"),
            "convenience_store": Decimal("640"),
            "retail_clothing": Decimal("250"),
            "fitness_center": Decimal("450"),
            "automotive_repair": Decimal("270"),
            "healthcare_clinic": Decimal("400"),
            "education_center": Decimal("280"),
            "coffee_shop": Decimal("680"),
        }

        if unit.floor_area_m2 is not None:
            eui = eui_defaults.get(unit.franchise_type.value, Decimal("400"))
            total_kwh = _quantize_8dp(unit.floor_area_m2 * eui)
            # Assume 60% electricity, 40% fuel; global average grid EF
            grid_ef = Decimal("0.4360")
            gas_ef = Decimal("0.1837")
            elec_co2e = _quantize_8dp(total_kwh * Decimal("0.60") * grid_ef)
            fuel_co2e = _quantize_8dp(total_kwh * Decimal("0.40") * gas_ef)
            co2e = _quantize_8dp(elec_co2e + fuel_co2e)
            dqi = Decimal("2.5")
        elif unit.annual_revenue is not None:
            # Revenue-based fallback
            intensity_defaults: Dict[str, Decimal] = {
                "qsr": Decimal("0.1850"),
                "full_service_restaurant": Decimal("0.1620"),
                "hotel": Decimal("0.0980"),
                "convenience_store": Decimal("0.0850"),
                "retail_clothing": Decimal("0.0420"),
                "fitness_center": Decimal("0.0730"),
                "automotive_repair": Decimal("0.0560"),
                "healthcare_clinic": Decimal("0.0490"),
                "education_center": Decimal("0.0380"),
                "coffee_shop": Decimal("0.1550"),
            }
            intensity = intensity_defaults.get(
                unit.franchise_type.value, Decimal("0.10")
            )
            co2e = _quantize_8dp(unit.annual_revenue * intensity)
            dqi = Decimal("2.0")
        else:
            co2e = _ZERO
            dqi = Decimal("1.0")

        return co2e, dqi

    def _calc_tier_3(
        self,
        unit: FranchiseUnitData,
        reporting_year: int,
    ) -> Tuple[Decimal, Decimal, DataQualityTier]:
        """
        Tier 3: Spend-based fallback using EEIO factors.

        Uses revenue if available, otherwise applies a default per-unit
        estimate based on franchise type.

        Args:
            unit: Unit with limited data.
            reporting_year: Reporting year.

        Returns:
            Tuple of (co2e, dqi_score, tier).
        """
        # Default per-unit annual emissions by type (kgCO2e) for units
        # with no financial data at all
        type_defaults: Dict[str, Decimal] = {
            "qsr": Decimal("85000"),
            "full_service_restaurant": Decimal("120000"),
            "hotel": Decimal("450000"),
            "convenience_store": Decimal("65000"),
            "retail_clothing": Decimal("35000"),
            "fitness_center": Decimal("55000"),
            "automotive_repair": Decimal("28000"),
            "healthcare_clinic": Decimal("32000"),
            "education_center": Decimal("18000"),
            "coffee_shop": Decimal("45000"),
        }

        if unit.annual_revenue is not None:
            # Use revenue x EEIO-like intensity
            naics_map = {
                "qsr": "722513", "full_service_restaurant": "722511",
                "hotel": "721110", "convenience_store": "445120",
                "retail_clothing": "448140", "fitness_center": "713940",
                "automotive_repair": "811111", "healthcare_clinic": "621111",
                "education_center": "611691", "coffee_shop": "722513",
            }
            eeio_defaults = {
                "722511": Decimal("0.3920"), "722513": Decimal("0.3680"),
                "721110": Decimal("0.1490"), "445120": Decimal("0.2350"),
                "448140": Decimal("0.1780"), "713940": Decimal("0.2080"),
                "811111": Decimal("0.1950"), "621111": Decimal("0.1620"),
                "611691": Decimal("0.1350"),
            }
            naics = naics_map.get(
                unit.franchise_type.value, "722513"
            )
            eeio = eeio_defaults.get(naics, Decimal("0.20"))
            co2e = _quantize_8dp(unit.annual_revenue * eeio)
            dqi = Decimal("1.5")
        else:
            # Use default per-unit estimate
            co2e = type_defaults.get(
                unit.franchise_type.value, Decimal("50000")
            )
            co2e = _quantize_8dp(co2e)
            dqi = Decimal("1.0")

        logger.debug(
            "Tier 3 (spend-based): unit=%s, co2e=%s kgCO2e, dqi=%s",
            unit.unit_id, co2e, dqi,
        )

        return co2e, dqi, DataQualityTier.TIER_3

    # ==========================================================================
    # DC-FRN-001: Company-Owned Split
    # ==========================================================================

    def _split_company_owned(
        self, units: List[FranchiseUnitData]
    ) -> Tuple[List[FranchiseUnitData], List[FranchiseUnitData]]:
        """
        Split units into franchised and company-owned (DC-FRN-001).

        Company-owned units belong in Scope 1/2 of the franchisor and must
        NOT be included in Scope 3 Category 14.

        Args:
            units: All submitted units.

        Returns:
            Tuple of (franchised_units, company_owned_units).
        """
        franchised: List[FranchiseUnitData] = []
        company_owned: List[FranchiseUnitData] = []

        for unit in units:
            if unit.ownership == OwnershipType.COMPANY_OWNED:
                company_owned.append(unit)
            else:
                franchised.append(unit)

        return franchised, company_owned

    # ==========================================================================
    # Partial Year Handling
    # ==========================================================================

    def _handle_partial_year_units(
        self, units: List[FranchiseUnitData]
    ) -> List[FranchiseUnitData]:
        """
        Process partial-year units.

        Currently a passthrough; partial-year proration is applied during
        per-unit calculation. This method logs warnings for short periods.

        Args:
            units: Franchised units.

        Returns:
            Same list (modified in future versions if needed).
        """
        for unit in units:
            if unit.months_operational < 12:
                logger.info(
                    "Partial-year unit: %s (%d months operational)",
                    unit.unit_id, unit.months_operational,
                )

        return units

    # ==========================================================================
    # Aggregation Methods
    # ==========================================================================

    def _aggregate_results(
        self, results: List[UnitCalculationResult]
    ) -> Decimal:
        """
        Sum total emissions across all unit results.

        Args:
            results: Per-unit calculation results.

        Returns:
            Total emissions in kgCO2e.
        """
        total = _ZERO
        for r in results:
            total = total + r.total_co2e
        return _quantize_8dp(total)

    def _generate_method_breakdown(
        self, results: List[UnitCalculationResult]
    ) -> MethodBreakdown:
        """
        Generate breakdown of calculation methods used.

        Args:
            results: Per-unit calculation results.

        Returns:
            MethodBreakdown with counts, emissions, and percentages per tier.
        """
        t1_count = 0
        t1_co2e = _ZERO
        t2_count = 0
        t2_co2e = _ZERO
        t3_count = 0
        t3_co2e = _ZERO

        for r in results:
            if r.tier == DataQualityTier.TIER_1:
                t1_count += 1
                t1_co2e = t1_co2e + r.total_co2e
            elif r.tier == DataQualityTier.TIER_2:
                t2_count += 1
                t2_co2e = t2_co2e + r.total_co2e
            else:
                t3_count += 1
                t3_co2e = t3_co2e + r.total_co2e

        total = len(results)
        if total == 0:
            total = 1  # Avoid division by zero

        total_dec = Decimal(str(total))

        return MethodBreakdown(
            tier_1_count=t1_count,
            tier_1_co2e=_quantize_8dp(t1_co2e),
            tier_1_pct=_quantize_4dp(
                Decimal(str(t1_count)) / total_dec * _HUNDRED
            ),
            tier_2_count=t2_count,
            tier_2_co2e=_quantize_8dp(t2_co2e),
            tier_2_pct=_quantize_4dp(
                Decimal(str(t2_count)) / total_dec * _HUNDRED
            ),
            tier_3_count=t3_count,
            tier_3_co2e=_quantize_8dp(t3_co2e),
            tier_3_pct=_quantize_4dp(
                Decimal(str(t3_count)) / total_dec * _HUNDRED
            ),
            total_units=len(results),
        )

    def _apply_network_adjustments(
        self,
        total_emissions: Decimal,
        network_info: HybridNetworkInput,
    ) -> Decimal:
        """
        Apply any network-level adjustments to total emissions.

        Currently a passthrough. Reserved for future adjustments such as
        network efficiency factors or shared infrastructure deductions.

        Args:
            total_emissions: Raw total from unit aggregation.
            network_info: Network input data.

        Returns:
            Adjusted total emissions (kgCO2e).
        """
        return total_emissions

    def _aggregate_by_region(
        self, results: List[UnitCalculationResult]
    ) -> Dict[str, Decimal]:
        """
        Aggregate emissions by country code.

        Args:
            results: Per-unit results.

        Returns:
            Dict mapping country code to total emissions (kgCO2e).
        """
        by_region: Dict[str, Decimal] = {}
        for r in results:
            key = r.country or "UNKNOWN"
            current = by_region.get(key, _ZERO)
            by_region[key] = _quantize_8dp(current + r.total_co2e)
        return dict(sorted(by_region.items()))

    def _aggregate_by_franchise_type(
        self, results: List[UnitCalculationResult]
    ) -> Dict[str, Decimal]:
        """
        Aggregate emissions by franchise type.

        Args:
            results: Per-unit results.

        Returns:
            Dict mapping franchise type to total emissions (kgCO2e).
        """
        by_type: Dict[str, Decimal] = {}
        for r in results:
            key = r.franchise_type.value
            current = by_type.get(key, _ZERO)
            by_type[key] = _quantize_8dp(current + r.total_co2e)
        return dict(sorted(by_type.items()))

    def _aggregate_by_brand(
        self, results: List[UnitCalculationResult]
    ) -> Dict[str, Decimal]:
        """
        Aggregate emissions by brand name.

        Args:
            results: Per-unit results.

        Returns:
            Dict mapping brand to total emissions (kgCO2e).
        """
        by_brand: Dict[str, Decimal] = {}
        for r in results:
            key = r.brand or "DEFAULT"
            current = by_brand.get(key, _ZERO)
            by_brand[key] = _quantize_8dp(current + r.total_co2e)
        return dict(sorted(by_brand.items()))

    # ==========================================================================
    # Data Quality
    # ==========================================================================

    def _calculate_weighted_dqi(
        self, results: List[UnitCalculationResult]
    ) -> DataQualityScore:
        """
        Calculate emissions-weighted DQI across the network.

        Each unit's DQI score is weighted by its share of total emissions,
        giving higher-emitting units more influence on the overall quality.

        Args:
            results: Per-unit results.

        Returns:
            DataQualityScore with weighted overall assessment.
        """
        if not results:
            return DataQualityScore(
                overall_score=Decimal("1.0"),
                tier=DataQualityTier.TIER_3,
                dimensions={},
                classification="Very Poor",
            )

        total_co2e = _ZERO
        weighted_sum = _ZERO

        for r in results:
            total_co2e = total_co2e + r.total_co2e
            weighted_sum = weighted_sum + (r.dqi_score * r.total_co2e)

        if total_co2e > _ZERO:
            overall = _quantize_4dp(weighted_sum / total_co2e)
        else:
            # Equal weight if all emissions are zero
            if results:
                total_dqi = sum(r.dqi_score for r in results)
                overall = _quantize_4dp(
                    total_dqi / Decimal(str(len(results)))
                )
            else:
                overall = Decimal("1.0")

        # Determine tier from weighted score
        if overall >= Decimal("4.0"):
            tier = DataQualityTier.TIER_1
        elif overall >= Decimal("2.5"):
            tier = DataQualityTier.TIER_2
        else:
            tier = DataQualityTier.TIER_3

        classification = _get_dqi_classification(overall)

        # Approximate per-dimension scores based on tier distribution
        dimensions = self._estimate_dimension_scores(results, overall)

        return DataQualityScore(
            overall_score=overall,
            tier=tier,
            dimensions=dimensions,
            classification=classification,
        )

    def _estimate_dimension_scores(
        self,
        results: List[UnitCalculationResult],
        overall: Decimal,
    ) -> Dict[str, Decimal]:
        """
        Estimate per-dimension DQI scores from overall and tier distribution.

        Args:
            results: Per-unit results.
            overall: Overall weighted DQI score.

        Returns:
            Dict of dimension name to estimated score.
        """
        # Base dimension scores from overall with slight variation
        base = overall
        return {
            DQIDimension.REPRESENTATIVENESS.value: min(
                _quantize_4dp(base * Decimal("0.95")), Decimal("5.0")
            ),
            DQIDimension.COMPLETENESS.value: min(
                _quantize_4dp(base * Decimal("1.05")), Decimal("5.0")
            ),
            DQIDimension.TEMPORAL.value: min(
                _quantize_4dp(base * Decimal("1.00")), Decimal("5.0")
            ),
            DQIDimension.GEOGRAPHICAL.value: min(
                _quantize_4dp(base * Decimal("0.90")), Decimal("5.0")
            ),
            DQIDimension.TECHNOLOGICAL.value: min(
                _quantize_4dp(base * Decimal("0.95")), Decimal("5.0")
            ),
        }

    # ==========================================================================
    # Data Coverage Report
    # ==========================================================================

    def _generate_coverage_report(
        self,
        total_submitted: int,
        company_owned: int,
        franchised_calculated: int,
        errors_count: int,
        method_breakdown: MethodBreakdown,
        unit_results: List[UnitCalculationResult],
    ) -> DataCoverageReport:
        """
        Generate data coverage and quality report for the network.

        Compares actual tier distribution against target thresholds:
          - Tier 1 target: >= 20% of units
          - Tier 1 + Tier 2 target: >= 50% of units

        Args:
            total_submitted: Total units submitted.
            company_owned: Company-owned units excluded.
            franchised_calculated: Franchised units successfully calculated.
            errors_count: Number of calculation errors.
            method_breakdown: Method tier distribution.
            unit_results: Per-unit calculation results.

        Returns:
            DataCoverageReport with coverage metrics and target comparisons.
        """
        # Extract unique brands and countries
        brands: Set[str] = set()
        countries: Set[str] = set()
        for r in unit_results:
            if r.brand:
                brands.add(r.brand)
            if r.country:
                countries.add(r.country)

        tier_1_pct = method_breakdown.tier_1_pct
        tier_2_pct = method_breakdown.tier_2_pct
        tier_3_pct = method_breakdown.tier_3_pct

        meets_t1 = tier_1_pct >= (TIER_1_TARGET_FRACTION * _HUNDRED)
        meets_t2 = (tier_1_pct + tier_2_pct) >= (
            (TIER_1_TARGET_FRACTION + TIER_2_TARGET_FRACTION) * _HUNDRED
        )

        return DataCoverageReport(
            total_units_submitted=total_submitted,
            company_owned_excluded=company_owned,
            franchised_calculated=franchised_calculated,
            calculation_errors=errors_count,
            method_breakdown=method_breakdown,
            brands=sorted(brands),
            countries=sorted(countries),
            tier_1_coverage_pct=tier_1_pct,
            tier_2_coverage_pct=tier_2_pct,
            tier_3_coverage_pct=tier_3_pct,
            meets_tier_1_target=meets_t1,
            meets_tier_2_target=meets_t2,
        )

    def _calculate_data_coverage(
        self, results: List[UnitCalculationResult]
    ) -> Dict[str, Any]:
        """
        Calculate simple data coverage statistics.

        Args:
            results: Per-unit results.

        Returns:
            Dict with coverage statistics.
        """
        total = len(results)
        if total == 0:
            return {"total": 0, "tier_1": 0, "tier_2": 0, "tier_3": 0}

        t1 = sum(1 for r in results if r.tier == DataQualityTier.TIER_1)
        t2 = sum(1 for r in results if r.tier == DataQualityTier.TIER_2)
        t3 = sum(1 for r in results if r.tier == DataQualityTier.TIER_3)

        return {
            "total": total,
            "tier_1": t1,
            "tier_2": t2,
            "tier_3": t3,
            "tier_1_pct": float(t1 / total * 100),
            "tier_2_pct": float(t2 / total * 100),
            "tier_3_pct": float(t3 / total * 100),
        }

    # ==========================================================================
    # Uncertainty
    # ==========================================================================

    def _calculate_uncertainty(
        self,
        results: List[UnitCalculationResult],
        total_co2e: Decimal,
    ) -> UncertaintyResult:
        """
        Calculate network-level uncertainty by propagating tier uncertainties.

        Uses sum-in-quadrature for combining independent uncertainty sources
        across tiers:
            sigma_total = sqrt(sum(sigma_i^2))
        where sigma_i = co2e_tier_i x uncertainty_half_width_tier_i

        Args:
            results: Per-unit results.
            total_co2e: Total network emissions.

        Returns:
            UncertaintyResult with confidence interval.
        """
        if total_co2e <= _ZERO or not results:
            return UncertaintyResult(
                mean_co2e=_ZERO,
                ci_lower=_ZERO,
                ci_upper=_ZERO,
                relative_uncertainty_pct=_ZERO,
                method="tier_weighted_propagation",
            )

        # Sum in quadrature per tier
        variance_sum = 0.0
        for r in results:
            tier_unc = TIER_UNCERTAINTY.get(r.tier, Decimal("0.50"))
            sigma_i = float(r.total_co2e * tier_unc)
            variance_sum += sigma_i ** 2

        sigma_total = Decimal(str(math.sqrt(variance_sum)))
        sigma_total = _quantize_8dp(sigma_total)

        ci_lower = _quantize_8dp(total_co2e - sigma_total)
        ci_upper = _quantize_8dp(total_co2e + sigma_total)

        # Ensure lower bound is non-negative
        if ci_lower < _ZERO:
            ci_lower = _ZERO

        relative_pct = _ZERO
        if total_co2e > _ZERO:
            relative_pct = _quantize_4dp(
                sigma_total / total_co2e * _HUNDRED
            )

        return UncertaintyResult(
            mean_co2e=total_co2e,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            relative_uncertainty_pct=relative_pct,
            method="tier_weighted_propagation",
        )

    # ==========================================================================
    # Year-over-Year Comparison
    # ==========================================================================

    def _calculate_yoy(
        self,
        current_co2e: Decimal,
        prior_co2e: Decimal,
        current_year: int,
        prior_year: int,
    ) -> YearOverYearComparison:
        """
        Calculate year-over-year emissions comparison.

        Args:
            current_co2e: Current year emissions (kgCO2e).
            prior_co2e: Prior year emissions (kgCO2e).
            current_year: Current reporting year.
            prior_year: Prior reporting year.

        Returns:
            YearOverYearComparison with absolute and percentage change.
        """
        absolute_change = _quantize_8dp(current_co2e - prior_co2e)

        pct_change = _ZERO
        if prior_co2e > _ZERO:
            pct_change = _quantize_4dp(
                absolute_change / prior_co2e * _HUNDRED
            )

        if absolute_change > _ZERO:
            direction = "increase"
        elif absolute_change < _ZERO:
            direction = "decrease"
        else:
            direction = "stable"

        return YearOverYearComparison(
            current_year=current_year,
            prior_year=prior_year,
            current_co2e=current_co2e,
            prior_co2e=prior_co2e,
            absolute_change=absolute_change,
            pct_change=pct_change,
            direction=direction,
        )

    # ==========================================================================
    # Operational Stats
    # ==========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Return operational statistics for this engine.

        Returns:
            Dictionary with calculation counts and configuration.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "network_calculations": self._calculation_count,
            "total_units_processed": self._total_units_processed,
            "tier_targets": {
                "tier_1": float(TIER_1_TARGET_FRACTION),
                "tier_2": float(TIER_2_TARGET_FRACTION),
                "tier_3": float(TIER_3_REMAINDER_FRACTION),
            },
        }


# ==============================================================================
# MODULE-LEVEL SINGLETON ACCESSOR
# ==============================================================================


_engine_instance: Optional[HybridAggregatorEngine] = None
_engine_lock: threading.RLock = threading.RLock()


def get_hybrid_aggregator() -> HybridAggregatorEngine:
    """
    Get the singleton HybridAggregatorEngine instance.

    Thread-safe accessor for the global engine instance.

    Returns:
        HybridAggregatorEngine singleton instance.

    Example:
        >>> engine = get_hybrid_aggregator()
        >>> result = engine.calculate(network_input)
    """
    global _engine_instance

    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = HybridAggregatorEngine.get_instance()

    return _engine_instance


def reset_hybrid_aggregator() -> None:
    """
    Reset the singleton engine instance (for testing only).

    Convenience function that resets both the module-level and class-level
    singletons. Should only be called in test teardown.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    HybridAggregatorEngine.reset_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Constants
    "ENGINE_ID",
    "ENGINE_VERSION",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "TABLE_PREFIX",
    "TIER_1_TARGET_FRACTION",
    "TIER_2_TARGET_FRACTION",
    "TIER_3_REMAINDER_FRACTION",
    # Enumerations
    "CalculationMethod",
    "OwnershipType",
    "DataQualityTier",
    "DQIDimension",
    "FranchiseType",
    "EFSource",
    "CurrencyCode",
    # Reference data
    "DQI_WEIGHTS",
    "TIER_UNCERTAINTY",
    # Input models
    "FranchiseUnitData",
    "HybridNetworkInput",
    # Output models
    "UnitCalculationResult",
    "DataQualityScore",
    "MethodBreakdown",
    "DataCoverageReport",
    "UncertaintyResult",
    "YearOverYearComparison",
    "NetworkAggregationResult",
    # Engine class
    "HybridAggregatorEngine",
    # Module-level accessors
    "get_hybrid_aggregator",
    "reset_hybrid_aggregator",
]
