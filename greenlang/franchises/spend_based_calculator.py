# -*- coding: utf-8 -*-
"""
SpendBasedCalculatorEngine - AGENT-MRV-027 Engine 4

GHG Protocol Scope 3 Category 14 Tier 3 spend-based emissions calculator
using EEIO (Environmentally Extended Input-Output) factors applied to
franchise revenue, royalty income, and per-unit averages.

This engine provides the lowest-tier fallback estimation method for franchise
emissions when neither franchise-specific metered data (Tier 1) nor area/
revenue benchmarks (Tier 2) are available. It implements three approaches:

1. **Total Franchise Revenue Approach**:
   E_total = total_franchise_revenue x EEIO_factor(NAICS_code)
   Uses aggregate revenue across the franchise network.

2. **Royalty-Based Approach**:
   implied_revenue = total_royalty_income / royalty_rate
   E_total = implied_revenue x EEIO_factor(NAICS_code)
   Useful when only royalty/franchise fee data is available.

3. **Per-Unit Average Approach**:
   E_per_unit = average_unit_revenue x EEIO_factor(NAICS_code)
   E_total = E_per_unit x franchised_unit_count
   Useful when aggregate data is unavailable but per-unit averages exist.

All approaches include:
   - NAICS-code-specific EEIO factor lookup (9 franchise sectors)
   - Multi-currency conversion (20 currencies)
   - CPI deflation to base year (2021)
   - Optional margin removal to isolate service cost from profit
   - SHA-256 provenance hashing for audit trail
   - 5-dimension data quality scoring (DQI)

DC-FRN-001: Company-owned units MUST be excluded. Revenue data should
reflect only franchised (not company-operated) unit revenues.

References:
    - GHG Protocol Technical Guidance for Scope 3, Category 14
    - US EPA USEEIO v2.0 Supply Chain Emission Factors
    - US BLS CPI-U (Consumer Price Index for All Urban Consumers)
    - US Census Bureau NAICS Classification System

Example:
    >>> engine = get_spend_based_calculator()
    >>> result = engine.calculate(FranchiseNetworkInput(
    ...     network_id="NET-001",
    ...     franchisor_name="FastBurger Inc.",
    ...     naics_code="722513",
    ...     total_franchise_revenue=Decimal("500000000"),
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
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict, field_validator

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "spend_based_calculator_engine"
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


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class SpendApproach(str, Enum):
    """Spend-based calculation approach."""

    TOTAL_REVENUE = "total_revenue"         # Total franchise revenue x EEIO
    ROYALTY_BASED = "royalty_based"          # Royalty income / rate x EEIO
    PER_UNIT_AVERAGE = "per_unit_average"   # Avg unit revenue x units x EEIO


class CalculationMethod(str, Enum):
    """Top-level calculation method classification."""

    FRANCHISE_SPECIFIC = "franchise_specific"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"
    HYBRID = "hybrid"


class EFSource(str, Enum):
    """Emission factor data source."""

    EEIO = "eeio"           # EPA USEEIO v2.0
    EXIOBASE = "exiobase"   # Exiobase 3.0 MRIO
    DEFRA = "defra"         # UK DEFRA conversion factors
    CUSTOM = "custom"       # Organization-specific factors


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


# ==============================================================================
# REFERENCE DATA TABLES
# ==============================================================================


# EEIO factors for franchise sectors (kgCO2e per USD revenue)
# Source: US EPA USEEIO v2.0 / Exiobase 3 cross-validated
# Keys are NAICS codes; values include factor and metadata
FRANCHISE_EEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    "722511": {
        "name": "Full-service restaurants",
        "description": "Sit-down restaurants, casual/fine dining franchises",
        "ef": Decimal("0.3920"),
        "ef_unit": "kgCO2e/USD",
        "source": "EPA USEEIO v2.0",
        "year": 2021,
        "examples": ["Applebee's", "Olive Garden", "Chili's", "IHOP"],
    },
    "722513": {
        "name": "Limited-service restaurants (QSR)",
        "description": "Fast food, quick-service, drive-through restaurants",
        "ef": Decimal("0.3680"),
        "ef_unit": "kgCO2e/USD",
        "source": "EPA USEEIO v2.0",
        "year": 2021,
        "examples": ["McDonald's", "Subway", "KFC", "Domino's", "Taco Bell"],
    },
    "721110": {
        "name": "Hotels and motels",
        "description": "Hotels, motels, and accommodation franchises",
        "ef": Decimal("0.1490"),
        "ef_unit": "kgCO2e/USD",
        "source": "EPA USEEIO v2.0",
        "year": 2021,
        "examples": ["Marriott", "Hilton", "Holiday Inn", "Best Western"],
    },
    "445120": {
        "name": "Convenience stores",
        "description": "Convenience stores, gas station shops",
        "ef": Decimal("0.2350"),
        "ef_unit": "kgCO2e/USD",
        "source": "EPA USEEIO v2.0",
        "year": 2021,
        "examples": ["7-Eleven", "Circle K", "Wawa"],
    },
    "448140": {
        "name": "Family clothing stores",
        "description": "Retail clothing and apparel franchises",
        "ef": Decimal("0.1780"),
        "ef_unit": "kgCO2e/USD",
        "source": "EPA USEEIO v2.0",
        "year": 2021,
        "examples": ["Gap", "H&M", "Zara", "Uniqlo"],
    },
    "713940": {
        "name": "Fitness and recreational centers",
        "description": "Gyms, fitness clubs, sports recreation franchises",
        "ef": Decimal("0.2080"),
        "ef_unit": "kgCO2e/USD",
        "source": "EPA USEEIO v2.0",
        "year": 2021,
        "examples": ["Planet Fitness", "Anytime Fitness", "Gold's Gym"],
    },
    "811111": {
        "name": "General automotive repair",
        "description": "Auto repair, maintenance, oil change franchises",
        "ef": Decimal("0.1950"),
        "ef_unit": "kgCO2e/USD",
        "source": "EPA USEEIO v2.0",
        "year": 2021,
        "examples": ["Jiffy Lube", "Meineke", "Midas", "Pep Boys"],
    },
    "621111": {
        "name": "Offices of physicians",
        "description": "Healthcare, urgent care, and medical franchises",
        "ef": Decimal("0.1620"),
        "ef_unit": "kgCO2e/USD",
        "source": "EPA USEEIO v2.0",
        "year": 2021,
        "examples": ["AFC Urgent Care", "CareSpot", "MinuteClinic"],
    },
    "611691": {
        "name": "Exam prep and tutoring",
        "description": "Education, tutoring, test preparation franchises",
        "ef": Decimal("0.1350"),
        "ef_unit": "kgCO2e/USD",
        "source": "EPA USEEIO v2.0",
        "year": 2021,
        "examples": ["Kumon", "Mathnasium", "Sylvan Learning"],
    },
}

# Franchise type to NAICS code mapping (for convenience)
FRANCHISE_TYPE_NAICS: Dict[str, str] = {
    "qsr": "722513",
    "full_service_restaurant": "722511",
    "hotel": "721110",
    "convenience_store": "445120",
    "retail_clothing": "448140",
    "fitness_center": "713940",
    "automotive_repair": "811111",
    "healthcare_clinic": "621111",
    "education_center": "611691",
    "coffee_shop": "722513",  # Coffee shops map to limited-service restaurants
}

# Default margin rates by sector (profit margin to remove from revenue)
SECTOR_MARGIN_RATES: Dict[str, Decimal] = {
    "722511": Decimal("0.06"),  # Full-service restaurants: 6% margin
    "722513": Decimal("0.08"),  # QSR / limited-service: 8% margin
    "721110": Decimal("0.12"),  # Hotels: 12% margin
    "445120": Decimal("0.04"),  # Convenience stores: 4% margin
    "448140": Decimal("0.10"),  # Retail clothing: 10% margin
    "713940": Decimal("0.15"),  # Fitness centers: 15% margin
    "811111": Decimal("0.12"),  # Automotive repair: 12% margin
    "621111": Decimal("0.08"),  # Healthcare clinics: 8% margin
    "611691": Decimal("0.18"),  # Education/tutoring: 18% margin
}

# Default margin rate when sector-specific is not available
DEFAULT_MARGIN_RATE: Decimal = Decimal("0.10")

# Typical royalty rates by franchise sector
TYPICAL_ROYALTY_RATES: Dict[str, Decimal] = {
    "722511": Decimal("0.05"),  # Full-service: 5%
    "722513": Decimal("0.06"),  # QSR: 6%
    "721110": Decimal("0.05"),  # Hotels: 5%
    "445120": Decimal("0.04"),  # Convenience: 4%
    "448140": Decimal("0.05"),  # Retail: 5%
    "713940": Decimal("0.07"),  # Fitness: 7%
    "811111": Decimal("0.06"),  # Automotive: 6%
    "621111": Decimal("0.06"),  # Healthcare: 6%
    "611691": Decimal("0.08"),  # Education: 8%
}

# Currency exchange rates to USD (mid-market, approximate)
CURRENCY_RATES: Dict[CurrencyCode, Decimal] = {
    CurrencyCode.USD: Decimal("1.0"),
    CurrencyCode.EUR: Decimal("1.0850"),
    CurrencyCode.GBP: Decimal("1.2650"),
    CurrencyCode.CAD: Decimal("0.7410"),
    CurrencyCode.AUD: Decimal("0.6520"),
    CurrencyCode.JPY: Decimal("0.006667"),
    CurrencyCode.CNY: Decimal("0.1378"),
    CurrencyCode.INR: Decimal("0.01198"),
    CurrencyCode.CHF: Decimal("1.1280"),
    CurrencyCode.SGD: Decimal("0.7440"),
    CurrencyCode.BRL: Decimal("0.1990"),
    CurrencyCode.ZAR: Decimal("0.05340"),
    CurrencyCode.MXN: Decimal("0.05680"),
    CurrencyCode.KRW: Decimal("0.000741"),
    CurrencyCode.NZD: Decimal("0.6090"),
    CurrencyCode.SEK: Decimal("0.09530"),
    CurrencyCode.NOK: Decimal("0.09280"),
    CurrencyCode.DKK: Decimal("0.1455"),
    CurrencyCode.AED: Decimal("0.2723"),
    CurrencyCode.SAR: Decimal("0.2666"),
}

# CPI deflators (base year 2021 = 1.0)
CPI_DEFLATORS: Dict[int, Decimal] = {
    2015: Decimal("0.8490"),
    2016: Decimal("0.8597"),
    2017: Decimal("0.8781"),
    2018: Decimal("0.8997"),
    2019: Decimal("0.9153"),
    2020: Decimal("0.9271"),
    2021: Decimal("1.0000"),
    2022: Decimal("1.0800"),
    2023: Decimal("1.1152"),
    2024: Decimal("1.1490"),
    2025: Decimal("1.1780"),
}

# DQI dimension weights (sum to 1.0)
DQI_WEIGHTS: Dict[DQIDimension, Decimal] = {
    DQIDimension.REPRESENTATIVENESS: Decimal("0.30"),
    DQIDimension.COMPLETENESS: Decimal("0.25"),
    DQIDimension.TEMPORAL: Decimal("0.15"),
    DQIDimension.GEOGRAPHICAL: Decimal("0.15"),
    DQIDimension.TECHNOLOGICAL: Decimal("0.15"),
}

# Uncertainty range for Tier 3 spend-based (half-width of 95% CI)
TIER_3_UNCERTAINTY: Decimal = Decimal("0.50")  # +/- 50%


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
    """
    Calculate SHA-256 provenance hash from variable inputs.

    Args:
        *inputs: Variable number of input objects to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
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
        """No-op metric recording."""
        pass

    def record_batch(self, **kwargs: Any) -> None:
        """No-op metric recording."""
        pass


def get_metrics_collector() -> Any:
    """Get the metrics collector for the Franchises agent."""
    try:
        from greenlang.franchises.metrics import get_metrics
        return get_metrics()
    except (ImportError, Exception):
        return _MetricsCollectorStub()


class _ProvenanceManagerStub:
    """Minimal provenance stub."""

    def start_chain(self) -> str:
        """Return a placeholder chain ID."""
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
        from greenlang.franchises.provenance import get_provenance_tracker
        return get_provenance_tracker()
    except (ImportError, Exception):
        return _ProvenanceManagerStub()


# ==============================================================================
# INPUT / OUTPUT MODELS
# ==============================================================================


class DataQualityScore(BaseModel):
    """Data quality assessment result."""

    overall_score: Decimal = Field(
        ..., description="Weighted composite DQI score (1.0 - 5.0)"
    )
    tier: DataQualityTier = Field(
        ..., description="Data quality tier classification"
    )
    dimensions: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Score per DQI dimension"
    )
    classification: str = Field(
        ..., description="Quality label: Excellent/Good/Fair/Poor/Very Poor"
    )

    model_config = ConfigDict(frozen=True)


class FranchiseNetworkInput(BaseModel):
    """
    Input for spend-based franchise network emissions calculation.

    At least one of total_franchise_revenue, total_royalty_income, or
    (average_unit_revenue + franchised_unit_count) must be provided.

    DC-FRN-001: Revenue data MUST exclude company-owned units.

    Example:
        >>> network = FranchiseNetworkInput(
        ...     network_id="NET-001",
        ...     franchisor_name="FastBurger Inc.",
        ...     naics_code="722513",
        ...     total_franchise_revenue=Decimal("500000000"),
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
    naics_code: str = Field(
        ..., min_length=5, max_length=8,
        description="NAICS code for EEIO factor lookup"
    )
    # Approach 1: Total revenue
    total_franchise_revenue: Optional[Decimal] = Field(
        default=None, gt=_ZERO,
        description="Total revenue across all franchised units (not company-owned)"
    )
    # Approach 2: Royalty-based
    total_royalty_income: Optional[Decimal] = Field(
        default=None, gt=_ZERO,
        description="Total royalty/franchise fee income"
    )
    royalty_rate: Optional[Decimal] = Field(
        default=None, gt=_ZERO, le=_ONE,
        description="Royalty rate as fraction (e.g., 0.06 for 6%)"
    )
    # Approach 3: Per-unit average
    average_unit_revenue: Optional[Decimal] = Field(
        default=None, gt=_ZERO,
        description="Average annual revenue per franchised unit"
    )
    franchised_unit_count: Optional[int] = Field(
        default=None, ge=1,
        description="Number of franchised units (excluding company-owned)"
    )
    # Common fields
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of all monetary values"
    )
    reporting_year: int = Field(
        default=2024, ge=2015, le=2030,
        description="Reporting year for CPI deflation"
    )
    enable_margin_removal: bool = Field(
        default=True,
        description="Whether to strip profit margin from revenue"
    )
    custom_margin_rate: Optional[Decimal] = Field(
        default=None, ge=_ZERO, le=_ONE,
        description="Custom margin rate override (if None, uses sector default)"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy isolation"
    )

    model_config = ConfigDict(frozen=True)


class NetworkAggregationResult(BaseModel):
    """
    Result from spend-based franchise network emissions calculation.

    Contains total emissions, approach used, intermediate values, and
    full provenance chain.
    """

    network_id: str = Field(..., description="Franchise network identifier")
    franchisor_name: str = Field(..., description="Franchisor name")
    naics_code: str = Field(..., description="NAICS code used")
    naics_name: str = Field(..., description="NAICS sector name")
    approach: SpendApproach = Field(
        ..., description="Spend-based approach applied"
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.SPEND_BASED,
        description="Top-level calculation method"
    )
    # Revenue and factor details
    revenue_original: Decimal = Field(
        ..., description="Original revenue in source currency"
    )
    revenue_usd: Decimal = Field(
        ..., description="Revenue converted to USD"
    )
    revenue_deflated_usd: Decimal = Field(
        ..., description="Revenue in base-year (2021) USD"
    )
    revenue_after_margin: Decimal = Field(
        ..., description="Revenue after margin removal (if enabled)"
    )
    margin_rate: Decimal = Field(
        ..., description="Margin rate applied (0 if disabled)"
    )
    eeio_factor: Decimal = Field(
        ..., description="EEIO factor applied (kgCO2e/USD)"
    )
    # Emissions
    total_co2e: Decimal = Field(
        ..., description="Total emissions (kgCO2e)"
    )
    total_tco2e: Decimal = Field(
        ..., description="Total emissions (tCO2e)"
    )
    # Per-unit (if applicable)
    per_unit_co2e: Optional[Decimal] = Field(
        default=None, description="Per-unit emissions if unit count known (kgCO2e)"
    )
    franchised_unit_count: Optional[int] = Field(
        default=None, description="Number of franchised units"
    )
    # Quality and provenance
    data_quality: DataQualityScore = Field(
        ..., description="Data quality assessment"
    )
    uncertainty_lower: Decimal = Field(
        ..., description="Lower bound of 95% CI (kgCO2e)"
    )
    uncertainty_upper: Decimal = Field(
        ..., description="Upper bound of 95% CI (kgCO2e)"
    )
    ef_source: EFSource = Field(
        default=EFSource.EEIO,
        description="Emission factor source"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculation_timestamp: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )
    engine_version: str = Field(
        default=ENGINE_VERSION,
        description="Engine version that produced this result"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Validation warnings (non-fatal)"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# SpendBasedCalculatorEngine
# ==============================================================================


class SpendBasedCalculatorEngine:
    """
    Tier 3 spend-based emissions calculator for franchise networks.

    Implements three EEIO-based approaches for estimating GHG Protocol
    Scope 3 Category 14 (Franchises) emissions using financial data.

    Approaches:
        1. Total Revenue: Aggregate franchise revenue x EEIO factor
        2. Royalty-Based: Royalty income / rate x EEIO factor
        3. Per-Unit Average: Avg unit revenue x unit count x EEIO factor

    Calculation Pipeline:
        1. Validate input (NAICS code exists, financial data provided)
        2. Select approach based on available data
        3. Convert currency to USD
        4. Apply CPI deflation to base year (2021)
        5. Optionally remove profit margin
        6. Look up EEIO factor by NAICS code
        7. Calculate emissions: adjusted_revenue x EEIO_factor
        8. Assess data quality (Tier 3)
        9. Record provenance hash and metrics

    Thread Safety:
        Singleton pattern with threading.RLock for concurrent access.

    Data Quality:
        Spend-based estimates are Tier 3 (lowest accuracy). The GHG Protocol
        recommends limiting spend-based to the bottom 50% of units and
        progressively upgrading to Tier 2 or Tier 1 data.

    DC-FRN-001:
        Input revenue MUST exclude company-owned units. Only franchised
        unit revenue should be included.

    Attributes:
        _metrics: Prometheus metrics collector
        _provenance: Provenance tracking manager
        _calculation_count: Running count of calculations performed

    Example:
        >>> engine = SpendBasedCalculatorEngine.get_instance()
        >>> result = engine.calculate(FranchiseNetworkInput(
        ...     network_id="NET-001",
        ...     franchisor_name="FastBurger Inc.",
        ...     naics_code="722513",
        ...     total_franchise_revenue=Decimal("500000000"),
        ...     reporting_year=2024,
        ... ))
        >>> result.total_co2e > Decimal("0")
        True
    """

    _instance: Optional["SpendBasedCalculatorEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __init__(self) -> None:
        """Initialize SpendBasedCalculatorEngine with metrics and provenance."""
        self._metrics = get_metrics_collector()
        self._provenance = get_provenance_manager()
        self._calculation_count: int = 0
        self._batch_count: int = 0

        logger.info(
            "SpendBasedCalculatorEngine initialized: version=%s, agent=%s, "
            "naics_sectors=%d, currencies=%d",
            ENGINE_VERSION, AGENT_ID,
            len(FRANCHISE_EEIO_FACTORS),
            len(CURRENCY_RATES),
        )

    @classmethod
    def get_instance(cls) -> "SpendBasedCalculatorEngine":
        """
        Get singleton instance (thread-safe double-checked locking).

        Returns:
            SpendBasedCalculatorEngine singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (for testing only).

        Thread Safety:
            Protected by the class-level RLock.
        """
        with cls._lock:
            cls._instance = None
            logger.info("SpendBasedCalculatorEngine singleton reset")

    # ==========================================================================
    # Primary Public Methods
    # ==========================================================================

    def calculate(
        self, network_input: FranchiseNetworkInput
    ) -> NetworkAggregationResult:
        """
        Calculate spend-based emissions for a franchise network.

        Automatically selects the best approach based on available data:
          1. total_franchise_revenue -> Total Revenue approach
          2. total_royalty_income + royalty_rate -> Royalty-Based approach
          3. average_unit_revenue + franchised_unit_count -> Per-Unit Average

        Args:
            network_input: Validated franchise network financial data.

        Returns:
            NetworkAggregationResult with total emissions, provenance.

        Raises:
            ValueError: If NAICS code not found in FRANCHISE_EEIO_FACTORS.
            ValueError: If insufficient financial data for any approach.

        Example:
            >>> result = engine.calculate(FranchiseNetworkInput(
            ...     network_id="NET-001",
            ...     franchisor_name="FastBurger Inc.",
            ...     naics_code="722513",
            ...     total_franchise_revenue=Decimal("500000000"),
            ... ))
        """
        start_time = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Validate inputs
        warnings = self._validate_spend_data(network_input)

        # Validate NAICS code
        self._validate_naics_code(network_input.naics_code)

        # Determine and execute approach
        approach, implied_revenue = self._select_approach(network_input)

        logger.debug(
            "Selected approach=%s for network=%s, implied_revenue=%s %s",
            approach.value, network_input.network_id,
            implied_revenue, network_input.currency.value,
        )

        # Convert currency to USD
        revenue_usd = self._apply_currency_conversion(
            implied_revenue, network_input.currency, network_input.reporting_year
        )

        # Apply CPI deflation
        revenue_deflated = self._apply_cpi_deflation(
            revenue_usd, network_input.reporting_year
        )

        # Apply margin removal
        margin_rate = _ZERO
        revenue_after_margin = revenue_deflated
        if network_input.enable_margin_removal:
            margin_rate = self._get_margin_rate(
                network_input.naics_code, network_input.custom_margin_rate
            )
            revenue_after_margin = self._apply_margin_removal(
                revenue_deflated, margin_rate
            )

        # Look up EEIO factor
        eeio_factor = self._get_eeio_factor(network_input.naics_code)
        naics_name = self._get_naics_name(network_input.naics_code)

        # Calculate emissions
        total_co2e = _quantize_8dp(revenue_after_margin * eeio_factor)
        total_tco2e = _quantize_8dp(total_co2e / Decimal("1000"))

        # Calculate per-unit if unit count available
        per_unit_co2e = None
        unit_count = network_input.franchised_unit_count
        if unit_count is not None and unit_count > 0:
            per_unit_co2e = _quantize_8dp(
                total_co2e / Decimal(str(unit_count))
            )

        # Assess data quality
        dq = self._assess_data_quality(network_input, approach)

        # Uncertainty bounds
        unc_lower = _quantize_8dp(total_co2e * (_ONE - TIER_3_UNCERTAINTY))
        unc_upper = _quantize_8dp(total_co2e * (_ONE + TIER_3_UNCERTAINTY))

        # Provenance hash
        provenance_hash = _calculate_provenance_hash(
            network_input, approach.value, revenue_usd,
            revenue_deflated, revenue_after_margin,
            eeio_factor, total_co2e,
        )

        # Record metrics
        duration = time.monotonic() - start_time
        self._record_metrics(
            network_input=network_input,
            approach=approach,
            co2e=total_co2e,
            duration=duration,
        )
        self._calculation_count += 1

        result = NetworkAggregationResult(
            network_id=network_input.network_id,
            franchisor_name=network_input.franchisor_name,
            naics_code=network_input.naics_code,
            naics_name=naics_name,
            approach=approach,
            method=CalculationMethod.SPEND_BASED,
            revenue_original=implied_revenue,
            revenue_usd=revenue_usd,
            revenue_deflated_usd=revenue_deflated,
            revenue_after_margin=revenue_after_margin,
            margin_rate=margin_rate,
            eeio_factor=eeio_factor,
            total_co2e=total_co2e,
            total_tco2e=total_tco2e,
            per_unit_co2e=per_unit_co2e,
            franchised_unit_count=unit_count,
            data_quality=dq,
            uncertainty_lower=unc_lower,
            uncertainty_upper=unc_upper,
            ef_source=EFSource.EEIO,
            provenance_hash=provenance_hash,
            calculation_timestamp=timestamp,
            engine_version=ENGINE_VERSION,
            warnings=warnings,
        )

        logger.info(
            "Spend-based calculation complete: network=%s, approach=%s, "
            "naics=%s, total_co2e=%s kgCO2e (%.2f tCO2e), duration=%.4fs",
            network_input.network_id, approach.value,
            network_input.naics_code,
            total_co2e, float(total_tco2e), duration,
        )

        return result

    def calculate_revenue_based(
        self,
        total_revenue: Decimal,
        naics_code: str,
        currency: CurrencyCode = CurrencyCode.USD,
        reporting_year: int = 2024,
        enable_margin_removal: bool = True,
        custom_margin_rate: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate emissions using total franchise revenue.

        Formula:
            revenue_usd = total_revenue x currency_rate
            deflated = revenue_usd / CPI_deflator(year)
            adjusted = deflated x (1 - margin_rate)
            co2e = adjusted x EEIO_factor(naics_code)

        Args:
            total_revenue: Total franchise revenue.
            naics_code: NAICS code for EEIO lookup.
            currency: Currency of the revenue (default: USD).
            reporting_year: Year for CPI deflation.
            enable_margin_removal: Whether to strip profit margin.
            custom_margin_rate: Override margin rate.

        Returns:
            Total emissions in kgCO2e.

        Raises:
            ValueError: If NAICS code not in FRANCHISE_EEIO_FACTORS.
        """
        self._validate_naics_code(naics_code)

        revenue_usd = self._apply_currency_conversion(
            total_revenue, currency, reporting_year
        )
        revenue_deflated = self._apply_cpi_deflation(
            revenue_usd, reporting_year
        )

        if enable_margin_removal:
            margin_rate = self._get_margin_rate(naics_code, custom_margin_rate)
            revenue_deflated = self._apply_margin_removal(
                revenue_deflated, margin_rate
            )

        eeio_factor = self._get_eeio_factor(naics_code)
        co2e = _quantize_8dp(revenue_deflated * eeio_factor)

        logger.debug(
            "Revenue-based: revenue=%s %s, deflated=%s USD, "
            "eeio=%s, co2e=%s kgCO2e",
            total_revenue, currency.value, revenue_deflated,
            eeio_factor, co2e,
        )

        return co2e

    def calculate_royalty_based(
        self,
        royalty_income: Decimal,
        royalty_rate: Decimal,
        naics_code: str,
        currency: CurrencyCode = CurrencyCode.USD,
        reporting_year: int = 2024,
        enable_margin_removal: bool = True,
        custom_margin_rate: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate emissions by deriving implied revenue from royalty data.

        Formula:
            implied_revenue = royalty_income / royalty_rate
            (then same as revenue-based approach)

        Args:
            royalty_income: Total royalty/franchise fee income.
            royalty_rate: Royalty rate as fraction (e.g., 0.06 for 6%).
            naics_code: NAICS code for EEIO lookup.
            currency: Currency of the royalty income.
            reporting_year: Year for CPI deflation.
            enable_margin_removal: Whether to strip profit margin.
            custom_margin_rate: Override margin rate.

        Returns:
            Total emissions in kgCO2e.

        Raises:
            ValueError: If royalty_rate is zero or negative.
            ValueError: If NAICS code not in FRANCHISE_EEIO_FACTORS.
        """
        if royalty_rate <= _ZERO:
            raise ValueError(
                f"Royalty rate must be positive, got {royalty_rate}"
            )

        # Derive implied revenue
        implied_revenue = _quantize_8dp(royalty_income / royalty_rate)

        logger.debug(
            "Royalty-based: royalty=%s, rate=%s, implied_revenue=%s %s",
            royalty_income, royalty_rate, implied_revenue, currency.value,
        )

        return self.calculate_revenue_based(
            total_revenue=implied_revenue,
            naics_code=naics_code,
            currency=currency,
            reporting_year=reporting_year,
            enable_margin_removal=enable_margin_removal,
            custom_margin_rate=custom_margin_rate,
        )

    def calculate_per_unit_average(
        self,
        avg_revenue: Decimal,
        unit_count: int,
        naics_code: str,
        currency: CurrencyCode = CurrencyCode.USD,
        reporting_year: int = 2024,
        enable_margin_removal: bool = True,
        custom_margin_rate: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate emissions using per-unit average revenue.

        Formula:
            total_revenue = avg_revenue x unit_count
            (then same as revenue-based approach)

        Args:
            avg_revenue: Average annual revenue per franchised unit.
            unit_count: Number of franchised units (DC-FRN-001 excluded).
            naics_code: NAICS code for EEIO lookup.
            currency: Currency of the revenue.
            reporting_year: Year for CPI deflation.
            enable_margin_removal: Whether to strip profit margin.
            custom_margin_rate: Override margin rate.

        Returns:
            Total emissions in kgCO2e.

        Raises:
            ValueError: If unit_count < 1.
            ValueError: If NAICS code not in FRANCHISE_EEIO_FACTORS.
        """
        if unit_count < 1:
            raise ValueError(
                f"Franchised unit count must be >= 1, got {unit_count}"
            )

        total_revenue = _quantize_8dp(
            avg_revenue * Decimal(str(unit_count))
        )

        logger.debug(
            "Per-unit average: avg=%s x %d units = %s %s total",
            avg_revenue, unit_count, total_revenue, currency.value,
        )

        return self.calculate_revenue_based(
            total_revenue=total_revenue,
            naics_code=naics_code,
            currency=currency,
            reporting_year=reporting_year,
            enable_margin_removal=enable_margin_removal,
            custom_margin_rate=custom_margin_rate,
        )

    # ==========================================================================
    # EEIO Factor Methods
    # ==========================================================================

    def _get_eeio_factor(self, naics_code: str) -> Decimal:
        """
        Look up EEIO emission factor by NAICS code.

        Args:
            naics_code: NAICS industry code string.

        Returns:
            EEIO factor in kgCO2e per USD.

        Raises:
            ValueError: If NAICS code not found.
        """
        entry = FRANCHISE_EEIO_FACTORS.get(naics_code)
        if entry is None:
            raise ValueError(
                f"NAICS code '{naics_code}' not found in FRANCHISE_EEIO_FACTORS. "
                f"Available: {sorted(FRANCHISE_EEIO_FACTORS.keys())}"
            )
        return entry["ef"]

    def _get_naics_name(self, naics_code: str) -> str:
        """Get the descriptive name for a NAICS code."""
        entry = FRANCHISE_EEIO_FACTORS.get(naics_code)
        if entry is None:
            return f"Unknown ({naics_code})"
        return entry["name"]

    def _validate_naics_code(self, naics_code: str) -> None:
        """Validate that the NAICS code exists in our EEIO factors."""
        if naics_code not in FRANCHISE_EEIO_FACTORS:
            raise ValueError(
                f"NAICS code '{naics_code}' not found in FRANCHISE_EEIO_FACTORS. "
                f"Available: {sorted(FRANCHISE_EEIO_FACTORS.keys())}"
            )

    # ==========================================================================
    # Currency and Deflation Methods
    # ==========================================================================

    def _apply_currency_conversion(
        self,
        amount: Decimal,
        currency: CurrencyCode,
        year: int,
    ) -> Decimal:
        """
        Convert an amount to USD using stored exchange rates.

        Args:
            amount: Amount in source currency.
            currency: Source currency code.
            year: Reporting year (currently rates are static).

        Returns:
            Amount in USD.

        Raises:
            ValueError: If currency not found in CURRENCY_RATES.
        """
        if currency == CurrencyCode.USD:
            return amount

        rate = CURRENCY_RATES.get(currency)
        if rate is None:
            raise ValueError(
                f"Currency '{currency.value}' not found in CURRENCY_RATES"
            )

        converted = _quantize_8dp(amount * rate)

        logger.debug(
            "Currency conversion: %s %s x %s = %s USD",
            amount, currency.value, rate, converted,
        )

        return converted

    def _apply_cpi_deflation(
        self,
        amount_usd: Decimal,
        reporting_year: int,
        base_year: int = 2021,
    ) -> Decimal:
        """
        Apply CPI deflation to normalize to base year USD.

        Args:
            amount_usd: Nominal amount in USD.
            reporting_year: Year of the financial data.
            base_year: Base year for deflation (default 2021).

        Returns:
            Real (base-year) amount in USD.

        Raises:
            ValueError: If year not found in CPI_DEFLATORS.
        """
        if reporting_year == base_year:
            return amount_usd

        year_deflator = CPI_DEFLATORS.get(reporting_year)
        if year_deflator is None:
            raise ValueError(
                f"CPI deflator not available for year {reporting_year}. "
                f"Available: {sorted(CPI_DEFLATORS.keys())}"
            )

        base_deflator = CPI_DEFLATORS.get(base_year)
        if base_deflator is None:
            raise ValueError(
                f"CPI deflator not available for base year {base_year}"
            )

        deflated = _quantize_8dp(amount_usd * base_deflator / year_deflator)

        logger.debug(
            "CPI deflation: %s (%d) -> %s (%d), factor=%s/%s",
            amount_usd, reporting_year, deflated, base_year,
            base_deflator, year_deflator,
        )

        return deflated

    def _apply_margin_removal(
        self,
        revenue: Decimal,
        margin_rate: Decimal,
    ) -> Decimal:
        """
        Remove profit margin from revenue to isolate service cost.

        The EEIO factor is calibrated against purchaser prices, so removing
        the margin provides a more accurate estimate of the emissions
        associated with producing the service.

        Args:
            revenue: Revenue amount in USD.
            margin_rate: Profit margin rate (0.0-1.0).

        Returns:
            Revenue after margin removal.
        """
        if margin_rate <= _ZERO:
            return revenue

        adjusted = _quantize_8dp(revenue * (_ONE - margin_rate))

        logger.debug(
            "Margin removal: %s x (1 - %s) = %s USD",
            revenue, margin_rate, adjusted,
        )

        return adjusted

    def _get_margin_rate(
        self,
        naics_code: str,
        custom_rate: Optional[Decimal],
    ) -> Decimal:
        """
        Get the margin rate for a NAICS code.

        Uses custom rate if provided, otherwise falls back to sector default,
        then to global default.

        Args:
            naics_code: NAICS code for sector lookup.
            custom_rate: Override margin rate (optional).

        Returns:
            Margin rate as Decimal fraction.
        """
        if custom_rate is not None:
            return custom_rate

        sector_rate = SECTOR_MARGIN_RATES.get(naics_code)
        if sector_rate is not None:
            return sector_rate

        return DEFAULT_MARGIN_RATE

    # ==========================================================================
    # Approach Selection and Validation
    # ==========================================================================

    def _select_approach(
        self, network_input: FranchiseNetworkInput
    ) -> Tuple[SpendApproach, Decimal]:
        """
        Select the best spend-based approach and return implied revenue.

        Priority:
            1. Total Revenue (most direct)
            2. Royalty-Based (requires royalty rate)
            3. Per-Unit Average (requires unit count)

        Args:
            network_input: Network financial input data.

        Returns:
            Tuple of (approach, implied_revenue_in_source_currency).

        Raises:
            ValueError: If no approach can be selected.
        """
        # Approach 1: Total Revenue
        if network_input.total_franchise_revenue is not None:
            return (
                SpendApproach.TOTAL_REVENUE,
                network_input.total_franchise_revenue,
            )

        # Approach 2: Royalty-Based
        if (network_input.total_royalty_income is not None
                and network_input.royalty_rate is not None):
            implied = _quantize_8dp(
                network_input.total_royalty_income / network_input.royalty_rate
            )
            return (SpendApproach.ROYALTY_BASED, implied)

        # Approach 2b: Royalty without explicit rate -- use typical rate
        if network_input.total_royalty_income is not None:
            typical_rate = TYPICAL_ROYALTY_RATES.get(
                network_input.naics_code, Decimal("0.05")
            )
            implied = _quantize_8dp(
                network_input.total_royalty_income / typical_rate
            )
            logger.warning(
                "No royalty_rate provided for network=%s; using typical "
                "rate %s for NAICS %s",
                network_input.network_id, typical_rate,
                network_input.naics_code,
            )
            return (SpendApproach.ROYALTY_BASED, implied)

        # Approach 3: Per-Unit Average
        if (network_input.average_unit_revenue is not None
                and network_input.franchised_unit_count is not None):
            implied = _quantize_8dp(
                network_input.average_unit_revenue
                * Decimal(str(network_input.franchised_unit_count))
            )
            return (SpendApproach.PER_UNIT_AVERAGE, implied)

        raise ValueError(
            f"Insufficient financial data for spend-based calculation "
            f"(network={network_input.network_id}). Provide at least one of: "
            f"total_franchise_revenue, total_royalty_income, or "
            f"(average_unit_revenue + franchised_unit_count)"
        )

    def _validate_spend_data(
        self, network_input: FranchiseNetworkInput
    ) -> List[str]:
        """
        Validate spend data and return a list of non-fatal warnings.

        Args:
            network_input: Network financial input.

        Returns:
            List of warning messages (empty if no issues).
        """
        warnings: List[str] = []

        # Check for unusually high or low revenues
        if network_input.total_franchise_revenue is not None:
            rev = network_input.total_franchise_revenue
            if rev > Decimal("100000000000"):
                warnings.append(
                    f"Very high revenue ({rev}) -- verify this is franchise-only "
                    f"revenue (DC-FRN-001: exclude company-owned)"
                )

        # Check royalty rate reasonableness
        if network_input.royalty_rate is not None:
            rate = network_input.royalty_rate
            if rate > Decimal("0.20"):
                warnings.append(
                    f"Unusually high royalty rate ({rate}); typical is 4-8%"
                )
            elif rate < Decimal("0.01"):
                warnings.append(
                    f"Unusually low royalty rate ({rate}); typical is 4-8%"
                )

        # Check unit count reasonableness
        if network_input.franchised_unit_count is not None:
            count = network_input.franchised_unit_count
            if count > 50000:
                warnings.append(
                    f"Very high unit count ({count}) -- verify this excludes "
                    f"company-owned units (DC-FRN-001)"
                )

        return warnings

    # ==========================================================================
    # Data Quality Assessment
    # ==========================================================================

    def _assess_data_quality(
        self,
        network_input: FranchiseNetworkInput,
        approach: SpendApproach,
    ) -> DataQualityScore:
        """
        Assess data quality for Tier 3 spend-based estimates.

        Spend-based methods typically score lower than Tier 2 benchmarks
        due to the use of aggregate financial data and EEIO factors.

        Args:
            network_input: Network input data.
            approach: Spend approach used.

        Returns:
            DataQualityScore with dimension scores and overall assessment.
        """
        dimensions: Dict[str, Decimal] = {}

        # Representativeness: EEIO factors are industry averages
        if approach == SpendApproach.TOTAL_REVENUE:
            dimensions[DQIDimension.REPRESENTATIVENESS.value] = Decimal("2.0")
        elif approach == SpendApproach.ROYALTY_BASED:
            dimensions[DQIDimension.REPRESENTATIVENESS.value] = Decimal("1.5")
        else:
            dimensions[DQIDimension.REPRESENTATIVENESS.value] = Decimal("1.5")

        # Completeness: How much data was provided?
        completeness = Decimal("1.5")
        if network_input.total_franchise_revenue is not None:
            completeness = completeness + Decimal("0.5")
        if network_input.franchised_unit_count is not None:
            completeness = completeness + Decimal("0.5")
        if network_input.total_royalty_income is not None:
            completeness = completeness + Decimal("0.5")
        completeness = min(completeness, Decimal("5.0"))
        dimensions[DQIDimension.COMPLETENESS.value] = completeness

        # Temporal: EEIO factors are for 2021; more recent years get higher score
        if network_input.reporting_year >= 2023:
            dimensions[DQIDimension.TEMPORAL.value] = Decimal("3.0")
        elif network_input.reporting_year >= 2021:
            dimensions[DQIDimension.TEMPORAL.value] = Decimal("2.5")
        else:
            dimensions[DQIDimension.TEMPORAL.value] = Decimal("2.0")

        # Geographical: EEIO factors are US-based
        if network_input.currency == CurrencyCode.USD:
            dimensions[DQIDimension.GEOGRAPHICAL.value] = Decimal("2.5")
        else:
            dimensions[DQIDimension.GEOGRAPHICAL.value] = Decimal("1.5")

        # Technological: Generic EEIO = low technology specificity
        dimensions[DQIDimension.TECHNOLOGICAL.value] = Decimal("1.5")

        # Calculate weighted overall
        overall = _ZERO
        for dim_name, dim_score in dimensions.items():
            dim_enum = DQIDimension(dim_name)
            weight = DQI_WEIGHTS.get(dim_enum, Decimal("0.20"))
            overall = overall + (dim_score * weight)
        overall = _quantize_4dp(overall)

        # Spend-based is always Tier 3
        tier = DataQualityTier.TIER_3
        classification = _get_dqi_classification(overall)

        return DataQualityScore(
            overall_score=overall,
            tier=tier,
            dimensions=dimensions,
            classification=classification,
        )

    # ==========================================================================
    # Lookup Helpers
    # ==========================================================================

    def get_available_naics_codes(self) -> List[Dict[str, Any]]:
        """
        Return all available NAICS codes with sector names and EEIO factors.

        Returns:
            List of dicts with naics_code, name, ef, and examples.

        Example:
            >>> codes = engine.get_available_naics_codes()
            >>> len(codes)
            9
        """
        result = []
        for naics_code, data in sorted(FRANCHISE_EEIO_FACTORS.items()):
            result.append({
                "naics_code": naics_code,
                "name": data["name"],
                "description": data["description"],
                "ef": float(data["ef"]),
                "ef_unit": data["ef_unit"],
                "source": data["source"],
                "examples": data["examples"],
            })
        return result

    def get_franchise_type_mapping(self) -> Dict[str, str]:
        """
        Return mapping from franchise type to NAICS code.

        Returns:
            Dict mapping franchise type string to NAICS code.
        """
        return dict(FRANCHISE_TYPE_NAICS)

    def get_stats(self) -> Dict[str, Any]:
        """
        Return operational statistics for this engine.

        Returns:
            Dictionary with calculation counts and configuration info.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculation_count": self._calculation_count,
            "batch_count": self._batch_count,
            "naics_sectors": len(FRANCHISE_EEIO_FACTORS),
            "currencies": len(CURRENCY_RATES),
        }

    # ==========================================================================
    # Metrics Recording
    # ==========================================================================

    def _record_metrics(
        self,
        network_input: FranchiseNetworkInput,
        approach: SpendApproach,
        co2e: Decimal,
        duration: float,
    ) -> None:
        """Record calculation metrics to the metrics collector."""
        try:
            self._metrics.record_calculation(
                method="spend_based",
                approach=approach.value,
                naics_code=network_input.naics_code,
                status="success",
                duration=duration,
                co2e=float(co2e),
            )
        except Exception as e:
            logger.warning("Failed to record metrics: %s", e)


# ==============================================================================
# MODULE-LEVEL SINGLETON ACCESSOR
# ==============================================================================


_engine_instance: Optional[SpendBasedCalculatorEngine] = None
_engine_lock: threading.RLock = threading.RLock()


def get_spend_based_calculator() -> SpendBasedCalculatorEngine:
    """
    Get the singleton SpendBasedCalculatorEngine instance.

    Thread-safe accessor for the global engine instance.

    Returns:
        SpendBasedCalculatorEngine singleton instance.

    Example:
        >>> engine = get_spend_based_calculator()
        >>> result = engine.calculate(network_input)
    """
    global _engine_instance

    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = SpendBasedCalculatorEngine.get_instance()

    return _engine_instance


def reset_spend_based_calculator() -> None:
    """
    Reset the singleton engine instance (for testing only).

    Convenience function that resets both the module-level and class-level
    singletons. Should only be called in test teardown.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    SpendBasedCalculatorEngine.reset_instance()


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
    # Enumerations
    "SpendApproach",
    "CalculationMethod",
    "EFSource",
    "DataQualityTier",
    "DQIDimension",
    "CurrencyCode",
    # Reference data
    "FRANCHISE_EEIO_FACTORS",
    "FRANCHISE_TYPE_NAICS",
    "SECTOR_MARGIN_RATES",
    "DEFAULT_MARGIN_RATE",
    "TYPICAL_ROYALTY_RATES",
    "CURRENCY_RATES",
    "CPI_DEFLATORS",
    "DQI_WEIGHTS",
    "TIER_3_UNCERTAINTY",
    # Input models
    "FranchiseNetworkInput",
    # Output models
    "DataQualityScore",
    "NetworkAggregationResult",
    # Engine class
    "SpendBasedCalculatorEngine",
    # Module-level accessors
    "get_spend_based_calculator",
    "reset_spend_based_calculator",
]
