# -*- coding: utf-8 -*-
"""
DebtInvestmentCalculatorEngine - Corporate bonds, business loans, and project finance.

This module implements the DebtInvestmentCalculatorEngine for AGENT-MRV-028
(Investments, GHG Protocol Scope 3 Category 15). It provides thread-safe
singleton calculations for financed emissions from debt instruments using
PCAF (Partnership for Carbon Accounting Financials) methodology.

Supported Debt Instruments:
    - Corporate bonds (EVIC-based attribution)
    - Business loans (EVIC-based attribution)
    - Project finance (project-cost-based attribution)
    - Green bonds (earmarked project emissions)
    - Transition bonds (trajectory tracking)
    - Revolving credit facilities (average outstanding)
    - Syndicated loans (share of syndication)

Calculation Formulas:
    Corporate Bonds & Business Loans:
        attribution_factor = outstanding_amount / EVIC
        financed_emissions = attribution_factor x company_emissions (Scope 1 + Scope 2)

    Project Finance:
        attribution_factor = outstanding_amount / total_project_cost
        financed_emissions = attribution_factor x project_lifetime_emissions / project_lifetime_years

PCAF Data Quality Hierarchy (Score 1-5):
    Score 1: Verified company/project emissions
    Score 2: Unverified reported emissions
    Score 3: Physical activity data + emission factors
    Score 4: Revenue-based EEIO
    Score 5: Sector/asset average

Thread Safety:
    Uses __new__ singleton pattern with threading.RLock for thread-safe
    instantiation. All mutable state is protected by locks.

Example:
    >>> engine = DebtInvestmentCalculatorEngine()
    >>> from decimal import Decimal
    >>> from greenlang.investments.debt_investment_calculator import (
    ...     DebtInvestmentInput, DebtInstrumentType
    ... )
    >>> inp = DebtInvestmentInput(
    ...     instrument_type=DebtInstrumentType.CORPORATE_BOND,
    ...     outstanding_amount=Decimal("10000000"),
    ...     borrower_evic=Decimal("500000000"),
    ...     borrower_scope1_co2e=Decimal("50000"),
    ...     borrower_scope2_co2e=Decimal("30000"),
    ...     reporting_year=2024,
    ... )
    >>> result = engine.calculate(inp)
    >>> result.financed_emissions_co2e > Decimal("0")
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-015
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-015"
AGENT_COMPONENT: str = "AGENT-MRV-028"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_inv_"

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Quantization constant: 8 decimal places
_QUANT_8DP = Decimal("0.00000001")

# Quantization constant: 2 decimal places (for display)
_QUANT_2DP = Decimal("0.01")

# Zero constant
_ZERO = Decimal("0")

# One constant
_ONE = Decimal("1")

# Maximum attribution factor (cannot exceed 100%)
_MAX_ATTRIBUTION = Decimal("1")

# SHA-256 encoding
_ENCODING = "utf-8"


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class DebtInstrumentType(str, Enum):
    """Type of debt instrument for financed emissions calculation."""

    CORPORATE_BOND = "corporate_bond"
    BUSINESS_LOAN = "business_loan"
    PROJECT_FINANCE = "project_finance"
    GREEN_BOND = "green_bond"
    TRANSITION_BOND = "transition_bond"
    REVOLVING_CREDIT = "revolving_credit"
    SYNDICATED_LOAN = "syndicated_loan"
    TERM_LOAN = "term_loan"
    CREDIT_LINE = "credit_line"


class PCAFDataQuality(int, Enum):
    """PCAF data quality score (1 = best, 5 = worst)."""

    SCORE_1 = 1  # Verified company/project emissions
    SCORE_2 = 2  # Unverified reported emissions
    SCORE_3 = 3  # Physical activity data + EFs
    SCORE_4 = 4  # Revenue-based EEIO
    SCORE_5 = 5  # Sector/asset average


class ProjectPhase(str, Enum):
    """Project phase for project finance emissions."""

    CONSTRUCTION = "construction"
    OPERATIONAL = "operational"
    DECOMMISSIONING = "decommissioning"
    MIXED = "mixed"


class ProjectType(str, Enum):
    """Project type for project finance emissions."""

    RENEWABLE_ENERGY = "renewable_energy"
    FOSSIL_FUEL = "fossil_fuel"
    INFRASTRUCTURE = "infrastructure"
    REAL_ESTATE = "real_estate"
    MANUFACTURING = "manufacturing"
    MINING = "mining"
    TRANSPORT = "transport"
    WATER_WASTE = "water_waste"
    TELECOMMUNICATIONS = "telecommunications"
    OTHER = "other"


class GreenBondCategory(str, Enum):
    """Green bond use-of-proceeds category per ICMA Green Bond Principles."""

    RENEWABLE_ENERGY = "renewable_energy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    POLLUTION_PREVENTION = "pollution_prevention"
    SUSTAINABLE_WATER = "sustainable_water"
    GREEN_BUILDINGS = "green_buildings"
    CLEAN_TRANSPORTATION = "clean_transportation"
    BIODIVERSITY = "biodiversity"
    CLIMATE_ADAPTATION = "climate_adaptation"
    OTHER = "other"


class EmissionsDataSource(str, Enum):
    """Source of borrower/project emissions data."""

    CDP_DISCLOSURE = "cdp_disclosure"
    ANNUAL_REPORT = "annual_report"
    SUSTAINABILITY_REPORT = "sustainability_report"
    THIRD_PARTY_VERIFIED = "third_party_verified"
    ESTIMATED_PHYSICAL = "estimated_physical"
    EEIO_REVENUE = "eeio_revenue"
    SECTOR_AVERAGE = "sector_average"
    PROJECT_EIA = "project_eia"
    DIRECT_MEASUREMENT = "direct_measurement"


class ComplianceFramework(str, Enum):
    """Regulatory/reporting frameworks for investments compliance."""

    GHG_PROTOCOL = "ghg_protocol"
    PCAF = "pcaf"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI_FI = "sbti_fi"
    SB_253 = "sb_253"
    TCFD = "tcfd"
    NZBA = "nzba"


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for monetary calculations."""

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
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    HKD = "HKD"
    KRW = "KRW"
    NZD = "NZD"
    MXN = "MXN"
    TRY = "TRY"


# ==============================================================================
# SECTOR AVERAGE EMISSION INTENSITIES (tCO2e per million USD revenue)
# Source: PCAF Database / EXIOBASE / EPA EEIO v2.0
# ==============================================================================

SECTOR_EMISSION_INTENSITIES: Dict[str, Dict[str, Any]] = {
    "oil_gas_extraction": {
        "intensity_tco2e_per_m_usd": Decimal("850.0"),
        "naics": "211",
        "description": "Oil and gas extraction",
    },
    "coal_mining": {
        "intensity_tco2e_per_m_usd": Decimal("1200.0"),
        "naics": "2121",
        "description": "Coal mining",
    },
    "electric_power": {
        "intensity_tco2e_per_m_usd": Decimal("1050.0"),
        "naics": "2211",
        "description": "Electric power generation",
    },
    "chemicals": {
        "intensity_tco2e_per_m_usd": Decimal("420.0"),
        "naics": "325",
        "description": "Chemical manufacturing",
    },
    "cement": {
        "intensity_tco2e_per_m_usd": Decimal("1800.0"),
        "naics": "32731",
        "description": "Cement and concrete",
    },
    "iron_steel": {
        "intensity_tco2e_per_m_usd": Decimal("780.0"),
        "naics": "3311",
        "description": "Iron and steel mills",
    },
    "aluminum": {
        "intensity_tco2e_per_m_usd": Decimal("650.0"),
        "naics": "3313",
        "description": "Alumina and aluminum",
    },
    "automotive": {
        "intensity_tco2e_per_m_usd": Decimal("180.0"),
        "naics": "3361",
        "description": "Motor vehicle manufacturing",
    },
    "food_beverage": {
        "intensity_tco2e_per_m_usd": Decimal("350.0"),
        "naics": "311",
        "description": "Food manufacturing",
    },
    "paper_pulp": {
        "intensity_tco2e_per_m_usd": Decimal("510.0"),
        "naics": "322",
        "description": "Paper manufacturing",
    },
    "pharmaceuticals": {
        "intensity_tco2e_per_m_usd": Decimal("95.0"),
        "naics": "3254",
        "description": "Pharmaceutical manufacturing",
    },
    "technology": {
        "intensity_tco2e_per_m_usd": Decimal("45.0"),
        "naics": "334",
        "description": "Computer and electronic products",
    },
    "financial_services": {
        "intensity_tco2e_per_m_usd": Decimal("12.0"),
        "naics": "52",
        "description": "Finance and insurance",
    },
    "real_estate": {
        "intensity_tco2e_per_m_usd": Decimal("85.0"),
        "naics": "53",
        "description": "Real estate",
    },
    "healthcare": {
        "intensity_tco2e_per_m_usd": Decimal("75.0"),
        "naics": "62",
        "description": "Health care",
    },
    "retail_trade": {
        "intensity_tco2e_per_m_usd": Decimal("55.0"),
        "naics": "44-45",
        "description": "Retail trade",
    },
    "telecommunications": {
        "intensity_tco2e_per_m_usd": Decimal("38.0"),
        "naics": "517",
        "description": "Telecommunications",
    },
    "transportation": {
        "intensity_tco2e_per_m_usd": Decimal("320.0"),
        "naics": "48-49",
        "description": "Transportation and warehousing",
    },
    "agriculture": {
        "intensity_tco2e_per_m_usd": Decimal("580.0"),
        "naics": "111-112",
        "description": "Agriculture",
    },
    "construction": {
        "intensity_tco2e_per_m_usd": Decimal("210.0"),
        "naics": "23",
        "description": "Construction",
    },
    "general_average": {
        "intensity_tco2e_per_m_usd": Decimal("200.0"),
        "naics": "ALL",
        "description": "Cross-sector average",
    },
}


# Currency exchange rates to USD (mid-market approximations)
CURRENCY_RATES_TO_USD: Dict[CurrencyCode, Decimal] = {
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
    CurrencyCode.SEK: Decimal("0.09250"),
    CurrencyCode.NOK: Decimal("0.09150"),
    CurrencyCode.DKK: Decimal("0.1450"),
    CurrencyCode.HKD: Decimal("0.1282"),
    CurrencyCode.KRW: Decimal("0.000745"),
    CurrencyCode.NZD: Decimal("0.6050"),
    CurrencyCode.MXN: Decimal("0.05820"),
    CurrencyCode.TRY: Decimal("0.03120"),
}


# Project type default emission intensities (tCO2e per million USD project cost per year)
PROJECT_TYPE_DEFAULT_INTENSITIES: Dict[ProjectType, Decimal] = {
    ProjectType.RENEWABLE_ENERGY: Decimal("15.0"),
    ProjectType.FOSSIL_FUEL: Decimal("950.0"),
    ProjectType.INFRASTRUCTURE: Decimal("180.0"),
    ProjectType.REAL_ESTATE: Decimal("85.0"),
    ProjectType.MANUFACTURING: Decimal("350.0"),
    ProjectType.MINING: Decimal("620.0"),
    ProjectType.TRANSPORT: Decimal("280.0"),
    ProjectType.WATER_WASTE: Decimal("120.0"),
    ProjectType.TELECOMMUNICATIONS: Decimal("40.0"),
    ProjectType.OTHER: Decimal("200.0"),
}


# PCAF uncertainty ranges by data quality score (half-width of 95% CI as fraction)
PCAF_UNCERTAINTY_RANGES: Dict[PCAFDataQuality, Decimal] = {
    PCAFDataQuality.SCORE_1: Decimal("0.05"),
    PCAFDataQuality.SCORE_2: Decimal("0.15"),
    PCAFDataQuality.SCORE_3: Decimal("0.30"),
    PCAFDataQuality.SCORE_4: Decimal("0.45"),
    PCAFDataQuality.SCORE_5: Decimal("0.60"),
}


# Double-counting rules for investments
# DC-INV-001: Do not double-count corporate bonds and equity in same company
# DC-INV-002: Do not double-count project finance with direct Scope 1/2
# DC-INV-003: Do not double-count syndicated loans across syndication members
# DC-INV-004: Sovereign bond emissions already include corporate - flag overlap
# DC-INV-005: Green bond project emissions vs general corporate emissions
DC_RULES: Dict[str, str] = {
    "DC-INV-001": (
        "Corporate bonds and listed equity in same company: use combined "
        "outstanding, not separate attribution"
    ),
    "DC-INV-002": (
        "Project finance emissions must not overlap with Scope 1/2 direct "
        "operational emissions of the same projects"
    ),
    "DC-INV-003": (
        "Syndicated loan: each syndicate member reports only their share, "
        "total must not exceed 100% of borrower emissions"
    ),
    "DC-INV-004": (
        "Sovereign bond country emissions include corporate emissions; "
        "flag potential overlap with corporate holdings in same country"
    ),
    "DC-INV-005": (
        "Green bond project-level emissions should be used instead of "
        "corporate-level when use-of-proceeds is earmarked"
    ),
}


# ==============================================================================
# INPUT MODELS
# ==============================================================================


class DebtInvestmentInput(BaseModel):
    """
    Input for corporate bond or business loan financed emissions calculation.

    Uses EVIC-based attribution:
        attribution_factor = outstanding_amount / borrower_evic
        financed_emissions = attribution_factor x (borrower_scope1 + borrower_scope2)

    Example:
        >>> inp = DebtInvestmentInput(
        ...     instrument_type=DebtInstrumentType.CORPORATE_BOND,
        ...     outstanding_amount=Decimal("10000000"),
        ...     borrower_evic=Decimal("500000000"),
        ...     borrower_scope1_co2e=Decimal("50000"),
        ...     borrower_scope2_co2e=Decimal("30000"),
        ...     reporting_year=2024,
        ... )
    """

    instrument_type: DebtInstrumentType = Field(
        ..., description="Type of debt instrument"
    )
    outstanding_amount: Decimal = Field(
        ..., gt=0, description="Outstanding loan/bond amount in currency units"
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of outstanding amount",
    )
    borrower_name: Optional[str] = Field(
        default=None, description="Borrower/issuer name"
    )
    borrower_sector: Optional[str] = Field(
        default=None, description="Borrower sector for EEIO fallback"
    )
    borrower_evic: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Enterprise Value Including Cash (EVIC) of borrower"
    )
    borrower_total_debt: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Total outstanding debt of borrower (fallback denominator)"
    )
    borrower_scope1_co2e: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Borrower Scope 1 emissions in tCO2e"
    )
    borrower_scope2_co2e: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Borrower Scope 2 emissions in tCO2e"
    )
    borrower_scope3_co2e: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Borrower Scope 3 emissions in tCO2e (optional, for full value chain)"
    )
    borrower_revenue: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Borrower annual revenue for EEIO fallback (in currency)"
    )
    borrower_verified: bool = Field(
        default=False,
        description="Whether borrower emissions are third-party verified"
    )
    emissions_data_source: Optional[EmissionsDataSource] = Field(
        default=None,
        description="Source of emissions data"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030,
        description="Reporting year for the calculation"
    )
    maturity_date: Optional[str] = Field(
        default=None,
        description="Bond/loan maturity date (ISO 8601)"
    )
    is_green_bond: bool = Field(
        default=False,
        description="Whether this is a green bond with earmarked proceeds"
    )
    green_bond_category: Optional[GreenBondCategory] = Field(
        default=None,
        description="Green bond use-of-proceeds category"
    )
    green_bond_project_emissions: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Project-specific emissions for green bond (tCO2e)"
    )
    is_revolving: bool = Field(
        default=False,
        description="Whether this is a revolving credit facility"
    )
    total_facility_amount: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Total facility size for revolving credit"
    )
    average_outstanding: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Average outstanding balance for revolving credit"
    )
    syndication_share: Optional[Decimal] = Field(
        default=None, gt=0, le=1,
        description="Share of syndicated loan (0-1)"
    )
    total_syndication_amount: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Total syndicated loan amount"
    )
    include_scope3: bool = Field(
        default=False,
        description="Include borrower Scope 3 in financed emissions (optional)"
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("outstanding_amount")
    def validate_outstanding(cls, v: Decimal) -> Decimal:
        """Validate outstanding amount is positive."""
        if v <= _ZERO:
            raise ValueError(f"Outstanding amount must be positive, got {v}")
        return v

    @validator("syndication_share")
    def validate_syndication_share(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate syndication share is between 0 and 1."""
        if v is not None and (v <= _ZERO or v > _ONE):
            raise ValueError(
                f"Syndication share must be in (0, 1], got {v}"
            )
        return v


class ProjectFinanceInput(BaseModel):
    """
    Input for project finance financed emissions calculation.

    Uses project-cost-based attribution:
        attribution_factor = outstanding_amount / total_project_cost
        financed_emissions = attribution_factor x project_lifetime_emissions / lifetime_years

    Example:
        >>> inp = ProjectFinanceInput(
        ...     outstanding_amount=Decimal("25000000"),
        ...     total_project_cost=Decimal("100000000"),
        ...     project_lifetime_emissions=Decimal("500000"),
        ...     project_lifetime_years=25,
        ...     project_type=ProjectType.RENEWABLE_ENERGY,
        ...     project_phase=ProjectPhase.OPERATIONAL,
        ...     reporting_year=2024,
        ... )
    """

    outstanding_amount: Decimal = Field(
        ..., gt=0, description="Outstanding financing amount"
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of monetary amounts",
    )
    total_project_cost: Decimal = Field(
        ..., gt=0, description="Total project cost (denominator for attribution)"
    )
    project_name: Optional[str] = Field(
        default=None, description="Project name/identifier"
    )
    project_type: ProjectType = Field(
        default=ProjectType.OTHER,
        description="Type of project",
    )
    project_phase: ProjectPhase = Field(
        default=ProjectPhase.OPERATIONAL,
        description="Current project phase",
    )
    project_lifetime_years: int = Field(
        ..., gt=0, le=100,
        description="Project expected lifetime in years"
    )
    project_lifetime_emissions: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Total project lifetime emissions in tCO2e"
    )
    annual_operational_emissions: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual operational emissions in tCO2e (for operational phase)"
    )
    construction_phase_emissions: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Total construction phase emissions in tCO2e"
    )
    construction_years: Optional[int] = Field(
        default=None, gt=0,
        description="Number of construction years"
    )
    borrower_verified: bool = Field(
        default=False,
        description="Whether project emissions are third-party verified"
    )
    emissions_data_source: Optional[EmissionsDataSource] = Field(
        default=None,
        description="Source of emissions data"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030,
        description="Reporting year"
    )
    is_renewable: bool = Field(
        default=False,
        description="Whether this is a renewable energy project"
    )
    syndication_share: Optional[Decimal] = Field(
        default=None, gt=0, le=1,
        description="Share of syndicated project financing (0-1)"
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("total_project_cost")
    def validate_project_cost(cls, v: Decimal) -> Decimal:
        """Validate total project cost is positive."""
        if v <= _ZERO:
            raise ValueError(
                f"Total project cost must be positive, got {v}"
            )
        return v


# ==============================================================================
# RESULT MODELS
# ==============================================================================


class InvestmentCalculationResult(BaseModel):
    """
    Result from a debt investment financed emissions calculation.

    Contains the full attribution breakdown, PCAF quality score,
    uncertainty range, and SHA-256 provenance hash.
    """

    instrument_type: str = Field(
        ..., description="Type of debt instrument"
    )
    outstanding_amount: Decimal = Field(
        ..., description="Outstanding amount used for attribution"
    )
    outstanding_amount_usd: Decimal = Field(
        ..., description="Outstanding amount converted to USD"
    )
    attribution_factor: Decimal = Field(
        ..., description="Attribution factor (outstanding / denominator)"
    )
    denominator_type: str = Field(
        ..., description="Type of denominator (EVIC, total_debt, project_cost)"
    )
    denominator_value: Decimal = Field(
        ..., description="Denominator value used"
    )
    company_emissions_co2e: Decimal = Field(
        ..., description="Total company/project emissions used (tCO2e)"
    )
    financed_emissions_co2e: Decimal = Field(
        ..., description="Financed emissions (tCO2e)"
    )
    financed_scope1_co2e: Decimal = Field(
        default=_ZERO,
        description="Financed Scope 1 emissions (tCO2e)"
    )
    financed_scope2_co2e: Decimal = Field(
        default=_ZERO,
        description="Financed Scope 2 emissions (tCO2e)"
    )
    financed_scope3_co2e: Decimal = Field(
        default=_ZERO,
        description="Financed Scope 3 emissions if included (tCO2e)"
    )
    pcaf_quality_score: int = Field(
        ..., ge=1, le=5,
        description="PCAF data quality score (1=best, 5=worst)"
    )
    uncertainty_lower_co2e: Decimal = Field(
        ..., description="Lower bound of 95% CI (tCO2e)"
    )
    uncertainty_upper_co2e: Decimal = Field(
        ..., description="Upper bound of 95% CI (tCO2e)"
    )
    emissions_data_source: str = Field(
        ..., description="Source of emissions data"
    )
    calculation_method: str = Field(
        ..., description="Calculation method used"
    )
    dc_rules_applied: List[str] = Field(
        default_factory=list,
        description="Double-counting rules checked"
    )
    dc_warnings: List[str] = Field(
        default_factory=list,
        description="Double-counting warnings"
    )
    reporting_year: int = Field(
        ..., description="Reporting year"
    )
    processing_time_ms: Decimal = Field(
        ..., description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HASH UTILITIES
# ==============================================================================


def _serialize_for_hash(obj: Any) -> str:
    """
    Serialize object to deterministic JSON for hashing.

    Args:
        obj: Object to serialize.

    Returns:
        Deterministic JSON string.
    """

    def _default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "model_dump"):
            return o.model_dump(mode="json")
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=_default)


def _compute_hash(*inputs: Any) -> str:
    """
    Compute SHA-256 hash from variable inputs.

    Args:
        *inputs: Objects to hash.

    Returns:
        Hex SHA-256 hash string.
    """
    combined = ""
    for inp in inputs:
        combined += _serialize_for_hash(inp)
    return hashlib.sha256(combined.encode(_ENCODING)).hexdigest()


# ==============================================================================
# ENGINE CLASS
# ==============================================================================


class DebtInvestmentCalculatorEngine:
    """
    Thread-safe singleton engine for debt investment financed emissions.

    Implements PCAF methodology for corporate bonds, business loans,
    project finance, green bonds, revolving credit, and syndicated loans.
    All arithmetic uses Python Decimal with ROUND_HALF_UP quantization
    to 8 decimal places for regulatory precision.

    This engine is ZERO-HALLUCINATION: all calculations are deterministic
    Python arithmetic. No LLM calls are used for any numeric computation.

    Thread Safety:
        Uses __new__ singleton pattern with threading.RLock. The
        _calculation_count attribute is protected by a dedicated lock.

    Attributes:
        _calculation_count: Total number of calculations performed.

    Example:
        >>> engine = DebtInvestmentCalculatorEngine()
        >>> result = engine.calculate(debt_input)
        >>> assert result.provenance_hash  # SHA-256 hash present
    """

    _instance: Optional["DebtInvestmentCalculatorEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "DebtInvestmentCalculatorEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the debt investment calculator engine."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._calculation_count: int = 0
        self._count_lock: threading.RLock = threading.RLock()

        logger.info(
            "DebtInvestmentCalculatorEngine initialized: agent=%s, version=%s",
            AGENT_ID,
            VERSION,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_count(self) -> int:
        """Increment and return the calculation counter thread-safely."""
        with self._count_lock:
            self._calculation_count += 1
            return self._calculation_count

    def _quantize(self, value: Decimal) -> Decimal:
        """Quantize a Decimal value to 8 decimal places."""
        return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

    def _quantize_2dp(self, value: Decimal) -> Decimal:
        """Quantize a Decimal value to 2 decimal places."""
        return value.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)

    def _convert_to_usd(self, amount: Decimal, currency: CurrencyCode) -> Decimal:
        """
        Convert amount from given currency to USD.

        Args:
            amount: Amount in source currency.
            currency: Source currency code.

        Returns:
            Amount in USD, quantized to 8 decimal places.

        Raises:
            ValueError: If currency not found.
        """
        rate = CURRENCY_RATES_TO_USD.get(currency)
        if rate is None:
            raise ValueError(
                f"Currency '{currency.value}' not found in exchange rate table"
            )
        return self._quantize(amount * rate)

    # =========================================================================
    # ATTRIBUTION FACTOR
    # =========================================================================

    def _calculate_attribution_factor(
        self,
        outstanding: Decimal,
        denominator: Decimal,
    ) -> Decimal:
        """
        Calculate the attribution factor as outstanding / denominator.

        The result is capped at 1.0 (100%) since a single lender cannot
        finance more than the entire entity.

        Args:
            outstanding: Outstanding amount (numerator).
            denominator: EVIC, total debt, or project cost (denominator).

        Returns:
            Attribution factor in [0, 1], quantized to 8 decimal places.

        Raises:
            ValueError: If denominator is zero or negative.
        """
        if denominator <= _ZERO:
            raise ValueError(
                f"Denominator must be positive, got {denominator}"
            )

        raw_factor = outstanding / denominator
        capped = min(raw_factor, _MAX_ATTRIBUTION)
        result = self._quantize(capped)

        logger.debug(
            "Attribution factor: %s / %s = %s (capped=%s)",
            outstanding, denominator, raw_factor, result,
        )

        return result

    # =========================================================================
    # PCAF DATA QUALITY
    # =========================================================================

    def _determine_pcaf_quality(
        self,
        has_scope1: bool,
        has_scope2: bool,
        verified: bool,
        data_source: Optional[EmissionsDataSource],
        has_revenue: bool,
        has_sector: bool,
    ) -> PCAFDataQuality:
        """
        Determine PCAF data quality score based on available data.

        Score 1: Verified emissions (third-party)
        Score 2: Unverified reported emissions
        Score 3: Physical activity data + emission factors
        Score 4: Revenue-based EEIO
        Score 5: Sector average only

        Args:
            has_scope1: Whether Scope 1 emissions are available.
            has_scope2: Whether Scope 2 emissions are available.
            verified: Whether data is third-party verified.
            data_source: Emissions data source.
            has_revenue: Whether revenue data is available.
            has_sector: Whether sector classification is available.

        Returns:
            PCAFDataQuality enum value.
        """
        if has_scope1 and has_scope2 and verified:
            return PCAFDataQuality.SCORE_1

        if has_scope1 and has_scope2 and not verified:
            return PCAFDataQuality.SCORE_2

        if data_source in (
            EmissionsDataSource.ESTIMATED_PHYSICAL,
            EmissionsDataSource.DIRECT_MEASUREMENT,
        ):
            return PCAFDataQuality.SCORE_3

        if has_revenue and has_sector:
            return PCAFDataQuality.SCORE_4

        return PCAFDataQuality.SCORE_5

    def _determine_pcaf_quality_project(
        self,
        has_project_emissions: bool,
        verified: bool,
        data_source: Optional[EmissionsDataSource],
        project_phase: ProjectPhase,
    ) -> PCAFDataQuality:
        """
        Determine PCAF data quality score for project finance.

        Args:
            has_project_emissions: Whether project emissions are available.
            verified: Whether data is third-party verified.
            data_source: Emissions data source.
            project_phase: Current project phase.

        Returns:
            PCAFDataQuality enum value.
        """
        if has_project_emissions and verified:
            return PCAFDataQuality.SCORE_1

        if has_project_emissions and not verified:
            return PCAFDataQuality.SCORE_2

        if data_source == EmissionsDataSource.PROJECT_EIA:
            return PCAFDataQuality.SCORE_3

        if data_source == EmissionsDataSource.ESTIMATED_PHYSICAL:
            return PCAFDataQuality.SCORE_3

        if project_phase == ProjectPhase.CONSTRUCTION:
            # Construction phase typically has estimated data
            return PCAFDataQuality.SCORE_4

        return PCAFDataQuality.SCORE_5

    # =========================================================================
    # UNCERTAINTY
    # =========================================================================

    def _calculate_uncertainty(
        self,
        financed_emissions: Decimal,
        quality_score: PCAFDataQuality,
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate uncertainty bounds based on PCAF quality score.

        Uses the half-width of the 95% confidence interval.

        Args:
            financed_emissions: Central estimate of financed emissions.
            quality_score: PCAF data quality score.

        Returns:
            Tuple of (lower_bound, upper_bound) in tCO2e.
        """
        half_width = PCAF_UNCERTAINTY_RANGES.get(
            quality_score, Decimal("0.60")
        )
        delta = self._quantize(financed_emissions * half_width)
        lower = self._quantize(max(financed_emissions - delta, _ZERO))
        upper = self._quantize(financed_emissions + delta)
        return lower, upper

    # =========================================================================
    # ANNUALIZE PROJECT EMISSIONS
    # =========================================================================

    def _annualize_project_emissions(
        self,
        lifetime_emissions: Decimal,
        lifetime_years: int,
    ) -> Decimal:
        """
        Annualize project lifetime emissions.

        Args:
            lifetime_emissions: Total project lifetime emissions in tCO2e.
            lifetime_years: Project lifetime in years.

        Returns:
            Annual emissions in tCO2e.

        Raises:
            ValueError: If lifetime_years is zero or negative.
        """
        if lifetime_years <= 0:
            raise ValueError(
                f"Lifetime years must be positive, got {lifetime_years}"
            )
        return self._quantize(
            lifetime_emissions / Decimal(str(lifetime_years))
        )

    # =========================================================================
    # REVOLVING CREDIT
    # =========================================================================

    def _handle_revolving_credit(
        self,
        avg_outstanding: Decimal,
        total_facility: Decimal,
    ) -> Decimal:
        """
        Calculate effective outstanding for revolving credit facilities.

        For revolving credit, PCAF recommends using the average outstanding
        balance over the reporting period rather than the committed amount.

        Args:
            avg_outstanding: Average outstanding balance during the period.
            total_facility: Total facility commitment.

        Returns:
            Effective outstanding amount (capped at total facility).
        """
        if avg_outstanding > total_facility:
            logger.warning(
                "Average outstanding %s exceeds total facility %s; "
                "capping at facility amount",
                avg_outstanding, total_facility,
            )
            return self._quantize(total_facility)
        return self._quantize(avg_outstanding)

    # =========================================================================
    # GREEN BOND HANDLING
    # =========================================================================

    def _handle_green_bond(
        self,
        input_data: DebtInvestmentInput,
    ) -> InvestmentCalculationResult:
        """
        Handle green bond calculation with project-specific emissions.

        Green bonds with earmarked use-of-proceeds should use project-level
        emissions rather than corporate-level emissions (DC-INV-005).

        Args:
            input_data: DebtInvestmentInput with green bond fields.

        Returns:
            InvestmentCalculationResult with project-specific emissions.

        Raises:
            ValueError: If required green bond fields are missing.
        """
        start_time = time.monotonic()

        if input_data.green_bond_project_emissions is None:
            raise ValueError(
                "Green bond requires green_bond_project_emissions "
                "when is_green_bond=True"
            )

        # Convert to USD
        outstanding_usd = self._convert_to_usd(
            input_data.outstanding_amount, input_data.currency
        )

        # For green bonds, denominator is the total bond issuance.
        # Use EVIC if available, else total_debt, else outstanding as proxy.
        if input_data.borrower_evic is not None:
            denominator = self._convert_to_usd(
                input_data.borrower_evic, input_data.currency
            )
            denominator_type = "evic"
        elif input_data.borrower_total_debt is not None:
            denominator = self._convert_to_usd(
                input_data.borrower_total_debt, input_data.currency
            )
            denominator_type = "total_debt"
        else:
            # Green bond uses outstanding as proxy
            denominator = outstanding_usd
            denominator_type = "outstanding_proxy"

        attribution = self._calculate_attribution_factor(
            outstanding_usd, denominator
        )

        # Use green bond project emissions
        project_co2e = input_data.green_bond_project_emissions
        financed_co2e = self._quantize(attribution * project_co2e)

        # Determine PCAF quality
        pcaf_score = self._determine_pcaf_quality(
            has_scope1=False,
            has_scope2=False,
            verified=input_data.borrower_verified,
            data_source=input_data.emissions_data_source,
            has_revenue=False,
            has_sector=False,
        )
        # Green bonds with specific project data get at least Score 2
        if input_data.green_bond_project_emissions is not None:
            if input_data.borrower_verified:
                pcaf_score = PCAFDataQuality.SCORE_1
            else:
                pcaf_score = PCAFDataQuality.SCORE_2

        lower, upper = self._calculate_uncertainty(financed_co2e, pcaf_score)

        duration_ms = Decimal(str((time.monotonic() - start_time) * 1000))

        provenance = _compute_hash(
            input_data.model_dump(mode="json"),
            str(attribution),
            str(financed_co2e),
            str(pcaf_score.value),
        )

        return InvestmentCalculationResult(
            instrument_type=DebtInstrumentType.GREEN_BOND.value,
            outstanding_amount=input_data.outstanding_amount,
            outstanding_amount_usd=outstanding_usd,
            attribution_factor=attribution,
            denominator_type=denominator_type,
            denominator_value=denominator,
            company_emissions_co2e=project_co2e,
            financed_emissions_co2e=financed_co2e,
            financed_scope1_co2e=_ZERO,
            financed_scope2_co2e=_ZERO,
            financed_scope3_co2e=_ZERO,
            pcaf_quality_score=pcaf_score.value,
            uncertainty_lower_co2e=lower,
            uncertainty_upper_co2e=upper,
            emissions_data_source=(
                input_data.emissions_data_source.value
                if input_data.emissions_data_source
                else "green_bond_project"
            ),
            calculation_method="green_bond_project_specific",
            dc_rules_applied=["DC-INV-005"],
            dc_warnings=[],
            reporting_year=input_data.reporting_year,
            processing_time_ms=self._quantize_2dp(duration_ms),
            provenance_hash=provenance,
        )

    # =========================================================================
    # VALIDATE INPUT
    # =========================================================================

    def _validate_debt_input(
        self,
        input_data: DebtInvestmentInput,
    ) -> List[str]:
        """
        Validate debt investment input for completeness and consistency.

        Args:
            input_data: DebtInvestmentInput to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        if input_data.outstanding_amount <= _ZERO:
            errors.append("outstanding_amount must be positive")

        if input_data.is_revolving:
            if input_data.total_facility_amount is None:
                errors.append(
                    "Revolving credit requires total_facility_amount"
                )
            if input_data.average_outstanding is None:
                errors.append(
                    "Revolving credit requires average_outstanding"
                )

        if input_data.instrument_type == DebtInstrumentType.SYNDICATED_LOAN:
            if input_data.syndication_share is None:
                errors.append(
                    "Syndicated loan requires syndication_share"
                )
            if input_data.total_syndication_amount is None:
                errors.append(
                    "Syndicated loan requires total_syndication_amount"
                )

        if input_data.is_green_bond:
            if input_data.green_bond_project_emissions is None:
                errors.append(
                    "Green bond requires green_bond_project_emissions"
                )

        # Check that we have at least one way to compute emissions
        has_direct = (
            input_data.borrower_scope1_co2e is not None
            or input_data.borrower_scope2_co2e is not None
        )
        has_revenue = input_data.borrower_revenue is not None
        has_sector = input_data.borrower_sector is not None
        has_green = (
            input_data.is_green_bond
            and input_data.green_bond_project_emissions is not None
        )

        if not has_direct and not has_revenue and not has_sector and not has_green:
            errors.append(
                "At least one of borrower_scope1/2_co2e, "
                "borrower_revenue+sector, or green_bond_project_emissions "
                "is required"
            )

        return errors

    # =========================================================================
    # CORE CALCULATION: CORPORATE BOND
    # =========================================================================

    def calculate_corporate_bond(
        self,
        input_data: DebtInvestmentInput,
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for a corporate bond.

        Formula:
            attribution_factor = outstanding_amount / EVIC
            financed_emissions = attribution_factor x (scope1 + scope2)

        Falls back to total_debt denominator if EVIC not available,
        then to revenue-based EEIO, then to sector average.

        Args:
            input_data: DebtInvestmentInput with bond details.

        Returns:
            InvestmentCalculationResult with financed emissions.

        Raises:
            ValueError: If required data is missing.
        """
        return self._calculate_evic_based(
            input_data,
            instrument_label=DebtInstrumentType.CORPORATE_BOND.value,
        )

    # =========================================================================
    # CORE CALCULATION: BUSINESS LOAN
    # =========================================================================

    def calculate_business_loan(
        self,
        input_data: DebtInvestmentInput,
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for a business loan.

        Uses same EVIC-based attribution as corporate bonds, but may have
        different data quality paths (loans to smaller borrowers typically
        have less disclosure).

        Args:
            input_data: DebtInvestmentInput with loan details.

        Returns:
            InvestmentCalculationResult with financed emissions.

        Raises:
            ValueError: If required data is missing.
        """
        return self._calculate_evic_based(
            input_data,
            instrument_label=DebtInstrumentType.BUSINESS_LOAN.value,
        )

    # =========================================================================
    # INTERNAL EVIC-BASED CALCULATION
    # =========================================================================

    def _calculate_evic_based(
        self,
        input_data: DebtInvestmentInput,
        instrument_label: str,
    ) -> InvestmentCalculationResult:
        """
        Internal EVIC-based financed emissions calculation.

        Shared logic for corporate bonds and business loans.

        The method selects the best available data path:
            1. EVIC + reported emissions (Score 1-2)
            2. Total debt + reported emissions (Score 2-3)
            3. Revenue-based EEIO (Score 4)
            4. Sector average (Score 5)

        Args:
            input_data: DebtInvestmentInput.
            instrument_label: Label for the instrument type.

        Returns:
            InvestmentCalculationResult.
        """
        start_time = time.monotonic()
        calc_number = self._increment_count()

        logger.info(
            "Debt calculation #%d: type=%s, outstanding=%s %s",
            calc_number,
            instrument_label,
            input_data.outstanding_amount,
            input_data.currency.value,
        )

        # Handle special instrument types
        if input_data.is_green_bond:
            return self._handle_green_bond(input_data)

        # Handle revolving credit outstanding
        effective_outstanding = input_data.outstanding_amount
        if input_data.is_revolving:
            if (
                input_data.average_outstanding is not None
                and input_data.total_facility_amount is not None
            ):
                effective_outstanding = self._handle_revolving_credit(
                    input_data.average_outstanding,
                    input_data.total_facility_amount,
                )

        # Handle syndicated loan
        if (
            input_data.instrument_type == DebtInstrumentType.SYNDICATED_LOAN
            and input_data.syndication_share is not None
        ):
            effective_outstanding = self._quantize(
                effective_outstanding * input_data.syndication_share
            )

        # Convert to USD
        outstanding_usd = self._convert_to_usd(
            effective_outstanding, input_data.currency
        )

        # Determine denominator and attribution
        denominator_type: str
        denominator_value: Decimal
        if input_data.borrower_evic is not None:
            denominator_value = self._convert_to_usd(
                input_data.borrower_evic, input_data.currency
            )
            denominator_type = "evic"
        elif input_data.borrower_total_debt is not None:
            denominator_value = self._convert_to_usd(
                input_data.borrower_total_debt, input_data.currency
            )
            denominator_type = "total_debt"
        elif input_data.borrower_revenue is not None:
            # Revenue-based path: use revenue as proxy denominator
            denominator_value = self._convert_to_usd(
                input_data.borrower_revenue, input_data.currency
            )
            denominator_type = "revenue_proxy"
        else:
            # Sector average path: use outstanding as denominator
            denominator_value = outstanding_usd
            denominator_type = "outstanding_proxy"

        attribution = self._calculate_attribution_factor(
            outstanding_usd, denominator_value
        )

        # Determine company emissions
        scope1 = input_data.borrower_scope1_co2e or _ZERO
        scope2 = input_data.borrower_scope2_co2e or _ZERO
        scope3 = (
            input_data.borrower_scope3_co2e
            if input_data.include_scope3 and input_data.borrower_scope3_co2e
            else _ZERO
        )

        calculation_method: str
        data_source_label: str

        if scope1 > _ZERO or scope2 > _ZERO:
            # Direct emissions data available
            company_emissions = scope1 + scope2 + scope3
            calculation_method = "reported_emissions"
            data_source_label = (
                input_data.emissions_data_source.value
                if input_data.emissions_data_source
                else "reported"
            )
        elif (
            input_data.borrower_revenue is not None
            and input_data.borrower_sector is not None
        ):
            # Revenue-based EEIO fallback
            revenue_usd = self._convert_to_usd(
                input_data.borrower_revenue, input_data.currency
            )
            revenue_m_usd = self._quantize(revenue_usd / Decimal("1000000"))
            sector_data = SECTOR_EMISSION_INTENSITIES.get(
                input_data.borrower_sector,
                SECTOR_EMISSION_INTENSITIES["general_average"],
            )
            intensity = sector_data["intensity_tco2e_per_m_usd"]
            company_emissions = self._quantize(revenue_m_usd * intensity)
            scope1 = _ZERO
            scope2 = _ZERO
            calculation_method = "eeio_revenue"
            data_source_label = "eeio_revenue"
            # For EEIO, use revenue as denominator for attribution
            if denominator_type == "revenue_proxy":
                attribution = _ONE  # Revenue-based: full emissions attributed
        elif input_data.borrower_sector is not None:
            # Sector average fallback (Score 5)
            sector_data = SECTOR_EMISSION_INTENSITIES.get(
                input_data.borrower_sector,
                SECTOR_EMISSION_INTENSITIES["general_average"],
            )
            intensity = sector_data["intensity_tco2e_per_m_usd"]
            outstanding_m_usd = self._quantize(
                outstanding_usd / Decimal("1000000")
            )
            company_emissions = self._quantize(outstanding_m_usd * intensity)
            scope1 = _ZERO
            scope2 = _ZERO
            calculation_method = "sector_average"
            data_source_label = "sector_average"
            attribution = _ONE  # Sector average: full amount attributed
        else:
            # General average fallback
            intensity = SECTOR_EMISSION_INTENSITIES["general_average"][
                "intensity_tco2e_per_m_usd"
            ]
            outstanding_m_usd = self._quantize(
                outstanding_usd / Decimal("1000000")
            )
            company_emissions = self._quantize(outstanding_m_usd * intensity)
            scope1 = _ZERO
            scope2 = _ZERO
            calculation_method = "general_average"
            data_source_label = "sector_average"
            attribution = _ONE

        # Calculate financed emissions
        financed_emissions = self._quantize(attribution * company_emissions)
        financed_s1 = self._quantize(attribution * scope1)
        financed_s2 = self._quantize(attribution * scope2)
        financed_s3 = self._quantize(attribution * scope3)

        # Determine PCAF quality
        pcaf_score = self._determine_pcaf_quality(
            has_scope1=(input_data.borrower_scope1_co2e is not None),
            has_scope2=(input_data.borrower_scope2_co2e is not None),
            verified=input_data.borrower_verified,
            data_source=input_data.emissions_data_source,
            has_revenue=(input_data.borrower_revenue is not None),
            has_sector=(input_data.borrower_sector is not None),
        )

        # Calculate uncertainty
        lower, upper = self._calculate_uncertainty(
            financed_emissions, pcaf_score
        )

        # Double-counting checks
        dc_applied: List[str] = ["DC-INV-001"]
        dc_warnings: List[str] = []
        if input_data.instrument_type == DebtInstrumentType.SYNDICATED_LOAN:
            dc_applied.append("DC-INV-003")

        # Processing time
        duration_ms = Decimal(str((time.monotonic() - start_time) * 1000))

        # Provenance hash
        provenance = _compute_hash(
            input_data.model_dump(mode="json"),
            str(attribution),
            str(company_emissions),
            str(financed_emissions),
            str(pcaf_score.value),
        )

        result = InvestmentCalculationResult(
            instrument_type=instrument_label,
            outstanding_amount=effective_outstanding,
            outstanding_amount_usd=outstanding_usd,
            attribution_factor=attribution,
            denominator_type=denominator_type,
            denominator_value=denominator_value,
            company_emissions_co2e=company_emissions,
            financed_emissions_co2e=financed_emissions,
            financed_scope1_co2e=financed_s1,
            financed_scope2_co2e=financed_s2,
            financed_scope3_co2e=financed_s3,
            pcaf_quality_score=pcaf_score.value,
            uncertainty_lower_co2e=lower,
            uncertainty_upper_co2e=upper,
            emissions_data_source=data_source_label,
            calculation_method=calculation_method,
            dc_rules_applied=dc_applied,
            dc_warnings=dc_warnings,
            reporting_year=input_data.reporting_year,
            processing_time_ms=self._quantize_2dp(duration_ms),
            provenance_hash=provenance,
        )

        logger.info(
            "Debt calculation #%d complete: type=%s, attribution=%.6f, "
            "financed_co2e=%s tCO2e, pcaf=%d, method=%s, "
            "duration=%.2fms, provenance=%s...%s",
            calc_number,
            instrument_label,
            float(attribution),
            financed_emissions,
            pcaf_score.value,
            calculation_method,
            float(duration_ms),
            provenance[:8],
            provenance[-8:],
        )

        return result

    # =========================================================================
    # PROJECT FINANCE
    # =========================================================================

    def calculate_project_finance(
        self,
        input_data: ProjectFinanceInput,
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for project finance.

        Formula:
            attribution = outstanding / total_project_cost
            annual_emissions = lifetime_emissions / lifetime_years
            financed_emissions = attribution x annual_emissions

        For construction phase, uses estimated emissions from EIA/project plan.
        For operational phase, uses actual operational emissions.
        Renewable energy projects may have near-zero operational emissions
        but material construction emissions.

        Args:
            input_data: ProjectFinanceInput with project details.

        Returns:
            InvestmentCalculationResult with financed emissions.

        Raises:
            ValueError: If required fields are missing.
        """
        start_time = time.monotonic()
        calc_number = self._increment_count()

        logger.info(
            "Project finance calculation #%d: type=%s, phase=%s, "
            "outstanding=%s %s, project_cost=%s",
            calc_number,
            input_data.project_type.value,
            input_data.project_phase.value,
            input_data.outstanding_amount,
            input_data.currency.value,
            input_data.total_project_cost,
        )

        # Convert to USD
        outstanding_usd = self._convert_to_usd(
            input_data.outstanding_amount, input_data.currency
        )
        project_cost_usd = self._convert_to_usd(
            input_data.total_project_cost, input_data.currency
        )

        # Handle syndication
        effective_outstanding = outstanding_usd
        if input_data.syndication_share is not None:
            effective_outstanding = self._quantize(
                outstanding_usd * input_data.syndication_share
            )

        # Attribution factor
        attribution = self._calculate_attribution_factor(
            effective_outstanding, project_cost_usd
        )

        # Determine annual emissions
        annual_emissions: Decimal
        calculation_method: str
        data_source_label: str

        if input_data.annual_operational_emissions is not None:
            # Direct operational emissions available
            annual_emissions = input_data.annual_operational_emissions
            calculation_method = "operational_reported"
            data_source_label = (
                input_data.emissions_data_source.value
                if input_data.emissions_data_source
                else "operational_reported"
            )
        elif input_data.project_lifetime_emissions is not None:
            # Lifetime emissions available, annualize
            annual_emissions = self._annualize_project_emissions(
                input_data.project_lifetime_emissions,
                input_data.project_lifetime_years,
            )
            calculation_method = "lifetime_annualized"
            data_source_label = (
                input_data.emissions_data_source.value
                if input_data.emissions_data_source
                else "lifetime_estimate"
            )
        elif (
            input_data.project_phase == ProjectPhase.CONSTRUCTION
            and input_data.construction_phase_emissions is not None
            and input_data.construction_years is not None
        ):
            # Construction phase with EIA data
            annual_emissions = self._annualize_project_emissions(
                input_data.construction_phase_emissions,
                input_data.construction_years,
            )
            calculation_method = "construction_eia"
            data_source_label = "project_eia"
        else:
            # Fallback to project type default intensity
            project_cost_m_usd = self._quantize(
                project_cost_usd / Decimal("1000000")
            )
            default_intensity = PROJECT_TYPE_DEFAULT_INTENSITIES.get(
                input_data.project_type,
                PROJECT_TYPE_DEFAULT_INTENSITIES[ProjectType.OTHER],
            )
            annual_emissions = self._quantize(
                project_cost_m_usd * default_intensity
            )
            calculation_method = "project_type_default"
            data_source_label = "sector_average"

        # Calculate financed emissions
        financed_emissions = self._quantize(attribution * annual_emissions)

        # Determine PCAF quality
        pcaf_score = self._determine_pcaf_quality_project(
            has_project_emissions=(
                input_data.project_lifetime_emissions is not None
                or input_data.annual_operational_emissions is not None
            ),
            verified=input_data.borrower_verified,
            data_source=input_data.emissions_data_source,
            project_phase=input_data.project_phase,
        )

        # Uncertainty
        lower, upper = self._calculate_uncertainty(
            financed_emissions, pcaf_score
        )

        # Double-counting rules
        dc_applied: List[str] = ["DC-INV-002"]
        dc_warnings: List[str] = []

        if input_data.syndication_share is not None:
            dc_applied.append("DC-INV-003")

        # Processing time
        duration_ms = Decimal(str((time.monotonic() - start_time) * 1000))

        # Provenance hash
        provenance = _compute_hash(
            input_data.model_dump(mode="json"),
            str(attribution),
            str(annual_emissions),
            str(financed_emissions),
            str(pcaf_score.value),
        )

        result = InvestmentCalculationResult(
            instrument_type="project_finance",
            outstanding_amount=input_data.outstanding_amount,
            outstanding_amount_usd=effective_outstanding,
            attribution_factor=attribution,
            denominator_type="project_cost",
            denominator_value=project_cost_usd,
            company_emissions_co2e=annual_emissions,
            financed_emissions_co2e=financed_emissions,
            financed_scope1_co2e=financed_emissions,
            financed_scope2_co2e=_ZERO,
            financed_scope3_co2e=_ZERO,
            pcaf_quality_score=pcaf_score.value,
            uncertainty_lower_co2e=lower,
            uncertainty_upper_co2e=upper,
            emissions_data_source=data_source_label,
            calculation_method=calculation_method,
            dc_rules_applied=dc_applied,
            dc_warnings=dc_warnings,
            reporting_year=input_data.reporting_year,
            processing_time_ms=self._quantize_2dp(duration_ms),
            provenance_hash=provenance,
        )

        logger.info(
            "Project finance #%d complete: type=%s, phase=%s, "
            "attribution=%.6f, annual_co2e=%s, financed_co2e=%s tCO2e, "
            "pcaf=%d, method=%s, duration=%.2fms",
            calc_number,
            input_data.project_type.value,
            input_data.project_phase.value,
            float(attribution),
            annual_emissions,
            financed_emissions,
            pcaf_score.value,
            calculation_method,
            float(duration_ms),
        )

        return result

    # =========================================================================
    # UNIFIED CALCULATE
    # =========================================================================

    def calculate(
        self,
        input_data: Union[DebtInvestmentInput, ProjectFinanceInput],
    ) -> InvestmentCalculationResult:
        """
        Unified calculation entry point for all debt instrument types.

        Routes to the appropriate calculator based on input type and
        instrument_type field.

        Args:
            input_data: DebtInvestmentInput or ProjectFinanceInput.

        Returns:
            InvestmentCalculationResult with financed emissions.

        Raises:
            ValueError: If input validation fails.
            TypeError: If input type is unsupported.
        """
        if isinstance(input_data, ProjectFinanceInput):
            return self.calculate_project_finance(input_data)

        if isinstance(input_data, DebtInvestmentInput):
            # Validate input
            errors = self._validate_debt_input(input_data)
            if errors:
                raise ValueError(
                    f"Input validation failed: {'; '.join(errors)}"
                )

            if input_data.is_green_bond:
                return self._handle_green_bond(input_data)

            if input_data.instrument_type == DebtInstrumentType.CORPORATE_BOND:
                return self.calculate_corporate_bond(input_data)

            if input_data.instrument_type == DebtInstrumentType.BUSINESS_LOAN:
                return self.calculate_business_loan(input_data)

            if input_data.instrument_type == DebtInstrumentType.PROJECT_FINANCE:
                raise TypeError(
                    "Use ProjectFinanceInput for project finance calculations"
                )

            # Default: treat as generic debt instrument
            return self._calculate_evic_based(
                input_data,
                instrument_label=input_data.instrument_type.value,
            )

        raise TypeError(
            f"Unsupported input type: {type(input_data).__name__}. "
            f"Expected DebtInvestmentInput or ProjectFinanceInput."
        )

    # =========================================================================
    # BATCH CALCULATION
    # =========================================================================

    def calculate_batch(
        self,
        inputs: List[Union[DebtInvestmentInput, ProjectFinanceInput]],
    ) -> List[InvestmentCalculationResult]:
        """
        Calculate financed emissions for a batch of debt investments.

        Processes each input independently. Failed calculations are logged
        but do not halt the batch.

        Args:
            inputs: List of DebtInvestmentInput or ProjectFinanceInput.

        Returns:
            List of InvestmentCalculationResult for successful calculations.
        """
        start_time = time.monotonic()
        results: List[InvestmentCalculationResult] = []
        error_count = 0

        logger.info(
            "Batch debt calculation started: %d inputs", len(inputs)
        )

        for i, inp in enumerate(inputs):
            try:
                result = self.calculate(inp)
                results.append(result)
            except Exception as exc:
                error_count += 1
                logger.error(
                    "Batch debt #%d failed: %s",
                    i + 1,
                    exc,
                    exc_info=True,
                )

        total_duration = time.monotonic() - start_time

        logger.info(
            "Batch debt calculation complete: %d/%d succeeded, "
            "%d errors, duration=%.3fs",
            len(results),
            len(inputs),
            error_count,
            total_duration,
        )

        return results

    # =========================================================================
    # SUMMARY AND STATS
    # =========================================================================

    def get_calculation_count(self) -> int:
        """Get the total number of calculations performed."""
        with self._count_lock:
            return self._calculation_count

    def get_engine_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the engine state.

        Returns:
            Dict with engine metadata and statistics.
        """
        return {
            "engine": "DebtInvestmentCalculatorEngine",
            "agent_id": AGENT_ID,
            "version": VERSION,
            "calculation_count": self.get_calculation_count(),
            "supported_instruments": [t.value for t in DebtInstrumentType],
            "supported_project_types": [t.value for t in ProjectType],
            "pcaf_quality_levels": [s.value for s in PCAFDataQuality],
            "sectors_available": len(SECTOR_EMISSION_INTENSITIES),
            "currencies_supported": len(CURRENCY_RATES_TO_USD),
            "dc_rules": list(DC_RULES.keys()),
        }

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        Warning: This method is intended for test fixtures only.
        Do not call in production code.
        """
        with cls._lock:
            cls._instance = None


# =============================================================================
# MODULE-LEVEL ACCESSOR
# =============================================================================

_calculator_instance: Optional[DebtInvestmentCalculatorEngine] = None
_calculator_lock: threading.RLock = threading.RLock()


def get_debt_investment_calculator() -> DebtInvestmentCalculatorEngine:
    """
    Get the singleton DebtInvestmentCalculatorEngine instance.

    Returns:
        DebtInvestmentCalculatorEngine singleton.
    """
    global _calculator_instance
    with _calculator_lock:
        if _calculator_instance is None:
            _calculator_instance = DebtInvestmentCalculatorEngine()
        return _calculator_instance


def reset_debt_investment_calculator() -> None:
    """
    Reset the module-level calculator instance (for testing only).

    Warning: Intended for test fixtures only.
    """
    global _calculator_instance
    with _calculator_lock:
        _calculator_instance = None
    DebtInvestmentCalculatorEngine.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    # Enums
    "DebtInstrumentType",
    "PCAFDataQuality",
    "ProjectPhase",
    "ProjectType",
    "GreenBondCategory",
    "EmissionsDataSource",
    "ComplianceFramework",
    "CurrencyCode",
    # Constants
    "SECTOR_EMISSION_INTENSITIES",
    "CURRENCY_RATES_TO_USD",
    "PROJECT_TYPE_DEFAULT_INTENSITIES",
    "PCAF_UNCERTAINTY_RANGES",
    "DC_RULES",
    # Input models
    "DebtInvestmentInput",
    "ProjectFinanceInput",
    # Result models
    "InvestmentCalculationResult",
    # Engine
    "DebtInvestmentCalculatorEngine",
    "get_debt_investment_calculator",
    "reset_debt_investment_calculator",
]
