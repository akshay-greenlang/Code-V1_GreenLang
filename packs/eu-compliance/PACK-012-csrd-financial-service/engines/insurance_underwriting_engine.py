# -*- coding: utf-8 -*-
"""
InsuranceUnderwritingEngine - PACK-012 CSRD Financial Service Engine 2
========================================================================

PCAF Part C "follow the risk" methodology for calculating emissions
associated with insurance underwriting portfolios.  Unlike financed
emissions (which follow the money), insurance-associated emissions
follow the insured risk via premium-based attribution.

Lines of Business Supported:
    1. Commercial Motor
    2. Personal Motor
    3. Commercial Property
    4. General Liability
    5. Project Insurance
    6. Treaty Reinsurance

Core Formulas:
    Premium Share        = Insurer Premium / Total Market Premium
    Gross Emissions      = Premium Share * Insured Emissions (tCO2e)
    Net Emissions        = Gross - Reinsurance Cession
    Reinsurance Adj      = Gross * (Ceded Premium / Written Premium)
    Claims Emissions     = Claims Paid * Emission Factor per Claim
    Intensity            = Total Emissions / Total Written Premium (tCO2e / EUR M)

Regulatory References:
    - PCAF Insurance-Associated Emissions Standard (Part C, 2022)
    - PCAF Global GHG Accounting & Reporting Standard (2nd Ed.)
    - ESRS E1-6 (Insurance-associated GHG emissions)
    - Solvency II (line of business classification)
    - NACE Rev. 2 (industry sector classification)

Zero-Hallucination:
    - All calculations use deterministic Python arithmetic
    - Premium-based attribution is pure ratio computation
    - Reinsurance adjustments use contractual cession percentages
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-012 CSRD Financial Service
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if denominator is zero.

    Returns:
        Result of division or default value.
    """
    if denominator == 0.0:
        return default
    return numerator / denominator

def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        Percentage or 0.0 on zero denominator.
    """
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0

def _round_val(value: float, places: int = 4) -> float:
    """Round a float to specified decimal places."""
    return round(value, places)

def _round_decimal(value: float, places: int = 4) -> float:
    """Round using Decimal for regulatory precision."""
    d = Decimal(str(value))
    q = Decimal("0." + "0" * places)
    return float(d.quantize(q, rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InsuranceLine(str, Enum):
    """Lines of business for insurance underwriting emissions."""
    COMMERCIAL_MOTOR = "commercial_motor"
    PERSONAL_MOTOR = "personal_motor"
    COMMERCIAL_PROPERTY = "commercial_property"
    GENERAL_LIABILITY = "general_liability"
    PROJECT_INSURANCE = "project_insurance"
    TREATY_REINSURANCE = "treaty_reinsurance"

class NACESector(str, Enum):
    """NACE Rev. 2 high-level sector classifications for insured entities."""
    AGRICULTURE = "A"
    MINING = "B"
    MANUFACTURING = "C"
    ELECTRICITY_GAS = "D"
    WATER_WASTE = "E"
    CONSTRUCTION = "F"
    WHOLESALE_RETAIL = "G"
    TRANSPORT_STORAGE = "H"
    ACCOMMODATION_FOOD = "I"
    ICT = "J"
    FINANCIAL_INSURANCE = "K"
    REAL_ESTATE = "L"
    PROFESSIONAL_SCIENTIFIC = "M"
    ADMINISTRATIVE = "N"
    PUBLIC_ADMIN = "O"
    EDUCATION = "P"
    HEALTH_SOCIAL = "Q"
    ARTS = "R"
    OTHER_SERVICES = "S"
    HOUSEHOLD = "T"
    EXTRATERRITORIAL = "U"

class DataQualityLevel(str, Enum):
    """PCAF data quality score levels for insurance (1 = best, 5 = worst)."""
    SCORE_1 = "score_1"  # Insured entity reported & verified emissions
    SCORE_2 = "score_2"  # Insured entity reported, unverified emissions
    SCORE_3 = "score_3"  # Estimated from physical activity data
    SCORE_4 = "score_4"  # Estimated from economic/premium data
    SCORE_5 = "score_5"  # Sector average estimates

    @property
    def numeric(self) -> int:
        """Return integer score value."""
        return int(self.value.split("_")[1])

class ReinsuranceType(str, Enum):
    """Types of reinsurance arrangements."""
    QUOTA_SHARE = "quota_share"
    SURPLUS = "surplus"
    EXCESS_OF_LOSS = "excess_of_loss"
    STOP_LOSS = "stop_loss"
    FACULTATIVE = "facultative"
    NONE = "none"

class EmissionCalculationMethod(str, Enum):
    """Method used to calculate insurance-associated emissions."""
    PREMIUM_BASED = "premium_based"
    CLAIMS_BASED = "claims_based"
    HYBRID = "hybrid"
    SECTOR_AVERAGE = "sector_average"

# ---------------------------------------------------------------------------
# Default Sector Emission Intensity Factors
# ---------------------------------------------------------------------------

# tCO2e per EUR M of insured premium (industry averages)
SECTOR_EMISSION_INTENSITY: Dict[str, float] = {
    NACESector.AGRICULTURE.value: 2450.0,
    NACESector.MINING.value: 3200.0,
    NACESector.MANUFACTURING.value: 1800.0,
    NACESector.ELECTRICITY_GAS.value: 4500.0,
    NACESector.WATER_WASTE.value: 1200.0,
    NACESector.CONSTRUCTION.value: 950.0,
    NACESector.WHOLESALE_RETAIL.value: 320.0,
    NACESector.TRANSPORT_STORAGE.value: 2800.0,
    NACESector.ACCOMMODATION_FOOD.value: 280.0,
    NACESector.ICT.value: 150.0,
    NACESector.FINANCIAL_INSURANCE.value: 80.0,
    NACESector.REAL_ESTATE.value: 400.0,
    NACESector.PROFESSIONAL_SCIENTIFIC.value: 120.0,
    NACESector.ADMINISTRATIVE.value: 100.0,
    NACESector.PUBLIC_ADMIN.value: 200.0,
    NACESector.EDUCATION.value: 90.0,
    NACESector.HEALTH_SOCIAL.value: 180.0,
    NACESector.ARTS.value: 70.0,
    NACESector.OTHER_SERVICES.value: 150.0,
    NACESector.HOUSEHOLD.value: 250.0,
    NACESector.EXTRATERRITORIAL.value: 100.0,
}

# Vehicle emission factors (tCO2e per vehicle per year, average)
VEHICLE_EMISSION_FACTORS: Dict[str, float] = {
    "passenger_car_petrol": 2.3,
    "passenger_car_diesel": 2.1,
    "passenger_car_hybrid": 1.4,
    "passenger_car_ev": 0.5,
    "light_commercial": 3.8,
    "heavy_commercial": 12.5,
    "bus_coach": 18.0,
    "motorcycle": 0.8,
}

# Property emission factors (tCO2e per EUR M insured value per year)
PROPERTY_EMISSION_FACTORS: Dict[str, float] = {
    "office": 35.0,
    "retail": 45.0,
    "warehouse": 25.0,
    "industrial": 85.0,
    "residential": 20.0,
    "mixed_use": 40.0,
    "data_center": 120.0,
}

# ---------------------------------------------------------------------------
# Pydantic Data Models
# ---------------------------------------------------------------------------

class ReinsuranceAdjustment(BaseModel):
    """Reinsurance cession adjustment for gross-to-net calculation.

    Captures the reinsurance arrangement details needed to compute
    the net emissions (after ceding risk to reinsurers).

    Attributes:
        arrangement_id: Unique arrangement identifier.
        reinsurance_type: Type of reinsurance arrangement.
        ceded_premium: Premium ceded to reinsurer (EUR).
        written_premium: Total written premium (EUR).
        cession_pct: Cession percentage (ceded/written * 100).
        reinsurer_name: Name of the reinsurer.
        treaty_year: Treaty year.
        provenance_hash: SHA-256 provenance hash.
    """
    arrangement_id: str = Field(
        default_factory=_new_uuid, description="Unique arrangement identifier",
    )
    reinsurance_type: ReinsuranceType = Field(
        default=ReinsuranceType.NONE,
        description="Type of reinsurance arrangement",
    )
    ceded_premium: float = Field(
        default=0.0, ge=0.0, description="Premium ceded to reinsurer (EUR)",
    )
    written_premium: float = Field(
        default=0.0, ge=0.0, description="Total written premium (EUR)",
    )
    cession_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Cession percentage",
    )
    reinsurer_name: str = Field(default="", description="Reinsurer name")
    treaty_year: int = Field(default=2024, description="Treaty year")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @model_validator(mode="after")
    def _compute_cession(self) -> "ReinsuranceAdjustment":
        """Auto-compute cession percentage if not provided."""
        if self.cession_pct <= 0.0 and self.written_premium > 0.0:
            self.cession_pct = (self.ceded_premium / self.written_premium) * 100.0
        return self

class ClaimsEmissions(BaseModel):
    """Claims-linked emissions for a policy or line of business.

    Tracks emissions attributable to insurance claims, which provide
    an alternative or supplementary measurement to premium-based.

    Attributes:
        claims_id: Unique identifier.
        total_claims_paid: Total claims paid in EUR.
        claims_count: Number of claims.
        emission_factor_per_claim: Average tCO2e per claim.
        total_claims_emissions: Total claims-linked emissions (tCO2e).
        methodology: Methodology used for claims emission calculation.
        provenance_hash: SHA-256 provenance hash.
    """
    claims_id: str = Field(
        default_factory=_new_uuid, description="Unique identifier",
    )
    total_claims_paid: float = Field(
        default=0.0, ge=0.0, description="Total claims paid (EUR)",
    )
    claims_count: int = Field(default=0, ge=0, description="Number of claims")
    emission_factor_per_claim: float = Field(
        default=0.0, ge=0.0,
        description="Average tCO2e per claim event",
    )
    total_claims_emissions: float = Field(
        default=0.0, ge=0.0,
        description="Total claims-linked emissions (tCO2e)",
    )
    methodology: str = Field(
        default="", description="Methodology for claims emissions",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class PolicyData(BaseModel):
    """Input data for a single insurance policy or risk.

    Contains all underwriting and risk data needed to compute
    insurance-associated emissions under PCAF Part C.

    Attributes:
        policy_id: Unique policy identifier.
        policyholder_name: Name of the policyholder.
        line_of_business: Insurance line classification.
        nace_sector: NACE sector of the insured activity.
        written_premium: Gross written premium (EUR).
        earned_premium: Earned premium for the period (EUR).
        insured_value: Total insured value (EUR).
        total_market_premium: Total market premium for this risk segment.
        insured_scope1: Insured entity Scope 1 emissions (tCO2e).
        insured_scope2: Insured entity Scope 2 emissions (tCO2e).
        insured_scope3: Insured entity Scope 3 emissions (tCO2e).
        insured_revenue: Insured entity annual revenue (EUR).
        vehicle_type: Vehicle type (for motor lines).
        vehicle_count: Number of vehicles insured.
        property_type: Property type (for property lines).
        property_area_sqm: Property floor area (sqm).
        project_type: Project type (for project insurance).
        project_total_emissions: Total project emissions (tCO2e).
        data_quality_score: PCAF data quality score (1-5).
        country: Country of risk (ISO 3166-1).
        reinsurance: Reinsurance arrangement details.
        claims_data: Claims-linked emission data.
        policy_start_date: Policy inception date.
        policy_end_date: Policy expiry date.
        calculation_method: Preferred emission calculation method.
    """
    policy_id: str = Field(default_factory=_new_uuid, description="Unique policy ID")
    policyholder_name: str = Field(default="", description="Policyholder name")
    line_of_business: InsuranceLine = Field(description="Insurance line classification")
    nace_sector: str = Field(default="", description="NACE sector of insured activity")
    written_premium: float = Field(
        default=0.0, ge=0.0, description="Gross written premium (EUR)",
    )
    earned_premium: float = Field(
        default=0.0, ge=0.0, description="Earned premium (EUR)",
    )
    insured_value: float = Field(
        default=0.0, ge=0.0, description="Total insured value (EUR)",
    )
    total_market_premium: float = Field(
        default=0.0, ge=0.0,
        description="Total market premium for this risk segment (EUR)",
    )
    # Insured entity emissions
    insured_scope1: float = Field(
        default=0.0, ge=0.0, description="Insured entity Scope 1 (tCO2e)",
    )
    insured_scope2: float = Field(
        default=0.0, ge=0.0, description="Insured entity Scope 2 (tCO2e)",
    )
    insured_scope3: float = Field(
        default=0.0, ge=0.0, description="Insured entity Scope 3 (tCO2e)",
    )
    insured_revenue: float = Field(
        default=0.0, ge=0.0, description="Insured entity annual revenue (EUR)",
    )
    # Motor-specific
    vehicle_type: str = Field(default="", description="Vehicle type classification")
    vehicle_count: int = Field(default=0, ge=0, description="Number of vehicles insured")
    # Property-specific
    property_type: str = Field(default="", description="Property type classification")
    property_area_sqm: float = Field(
        default=0.0, ge=0.0, description="Property floor area (sqm)",
    )
    # Project-specific
    project_type: str = Field(default="", description="Project type")
    project_total_emissions: float = Field(
        default=0.0, ge=0.0, description="Total project lifetime emissions (tCO2e)",
    )
    data_quality_score: int = Field(
        default=5, ge=1, le=5, description="PCAF data quality score (1-5)",
    )
    country: str = Field(default="", description="Country of risk (ISO 3166)")
    reinsurance: Optional[ReinsuranceAdjustment] = Field(
        default=None, description="Reinsurance arrangement details",
    )
    claims_data: Optional[ClaimsEmissions] = Field(
        default=None, description="Claims-linked emission data",
    )
    policy_start_date: str = Field(default="", description="Policy inception (YYYY-MM-DD)")
    policy_end_date: str = Field(default="", description="Policy expiry (YYYY-MM-DD)")
    calculation_method: EmissionCalculationMethod = Field(
        default=EmissionCalculationMethod.PREMIUM_BASED,
        description="Preferred emission calculation method",
    )

class PolicyEmissionsResult(BaseModel):
    """Emissions result for a single insurance policy.

    Contains gross and net emissions, premium share attribution,
    and data quality assessment.

    Attributes:
        policy_id: Policy identifier.
        policyholder_name: Policyholder name.
        line_of_business: Insurance line.
        premium_share: Premium share attribution factor.
        gross_emissions: Gross attributed emissions (tCO2e).
        reinsurance_cession_emissions: Emissions ceded to reinsurers.
        net_emissions: Net emissions after reinsurance (tCO2e).
        claims_emissions: Claims-linked emissions (tCO2e).
        emission_intensity: tCO2e per EUR M written premium.
        written_premium: Gross written premium (EUR).
        data_quality_score: PCAF data quality score.
        calculation_method: Method used for calculation.
        methodology_note: Note on the calculation methodology.
        provenance_hash: SHA-256 provenance hash.
    """
    policy_id: str = Field(default="", description="Policy identifier")
    policyholder_name: str = Field(default="", description="Policyholder name")
    line_of_business: InsuranceLine = Field(description="Insurance line")
    premium_share: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Premium share attribution factor",
    )
    gross_emissions: float = Field(
        default=0.0, description="Gross attributed emissions (tCO2e)",
    )
    reinsurance_cession_emissions: float = Field(
        default=0.0, description="Emissions ceded to reinsurers (tCO2e)",
    )
    net_emissions: float = Field(
        default=0.0, description="Net emissions after reinsurance (tCO2e)",
    )
    claims_emissions: float = Field(
        default=0.0, description="Claims-linked emissions (tCO2e)",
    )
    emission_intensity: float = Field(
        default=0.0, description="tCO2e per EUR M written premium",
    )
    written_premium: float = Field(
        default=0.0, description="Gross written premium (EUR)",
    )
    data_quality_score: int = Field(
        default=5, ge=1, le=5, description="PCAF data quality score",
    )
    calculation_method: EmissionCalculationMethod = Field(
        default=EmissionCalculationMethod.PREMIUM_BASED,
        description="Method used for calculation",
    )
    methodology_note: str = Field(
        default="", description="Methodology note",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class LineOfBusinessResult(BaseModel):
    """Aggregated results for a single line of business.

    Attributes:
        line_of_business: Insurance line.
        policy_count: Number of policies.
        total_written_premium: Total GWP (EUR).
        total_gross_emissions: Total gross emissions (tCO2e).
        total_net_emissions: Total net emissions (tCO2e).
        total_claims_emissions: Total claims emissions (tCO2e).
        total_reinsurance_cession: Total reinsurance cession (tCO2e).
        weighted_emission_intensity: Premium-weighted intensity.
        avg_data_quality_score: Average DQ score.
        weight_in_portfolio_pct: Weight of this LoB in total portfolio.
    """
    line_of_business: InsuranceLine = Field(description="Insurance line")
    policy_count: int = Field(default=0, ge=0, description="Number of policies")
    total_written_premium: float = Field(
        default=0.0, description="Total GWP (EUR)",
    )
    total_gross_emissions: float = Field(
        default=0.0, description="Total gross emissions (tCO2e)",
    )
    total_net_emissions: float = Field(
        default=0.0, description="Total net emissions (tCO2e)",
    )
    total_claims_emissions: float = Field(
        default=0.0, description="Total claims emissions (tCO2e)",
    )
    total_reinsurance_cession: float = Field(
        default=0.0, description="Total reinsurance cession (tCO2e)",
    )
    weighted_emission_intensity: float = Field(
        default=0.0, description="Premium-weighted emission intensity",
    )
    avg_data_quality_score: float = Field(
        default=5.0, ge=1.0, le=5.0, description="Average DQ score",
    )
    weight_in_portfolio_pct: float = Field(
        default=0.0, description="Weight of LoB in portfolio (%)",
    )

class SectorBreakdown(BaseModel):
    """Emissions breakdown by NACE industry sector.

    Attributes:
        nace_sector: NACE sector code.
        sector_name: Human-readable sector name.
        policy_count: Number of policies in this sector.
        total_written_premium: Total GWP for this sector.
        total_gross_emissions: Gross emissions for this sector.
        total_net_emissions: Net emissions for this sector.
        emission_intensity: Intensity for this sector.
        weight_pct: Weight in total portfolio.
    """
    nace_sector: str = Field(description="NACE sector code")
    sector_name: str = Field(default="", description="Sector name")
    policy_count: int = Field(default=0, ge=0, description="Number of policies")
    total_written_premium: float = Field(default=0.0, description="Total GWP (EUR)")
    total_gross_emissions: float = Field(default=0.0, description="Gross emissions (tCO2e)")
    total_net_emissions: float = Field(default=0.0, description="Net emissions (tCO2e)")
    emission_intensity: float = Field(default=0.0, description="Intensity (tCO2e/EUR M)")
    weight_pct: float = Field(default=0.0, description="Weight in portfolio (%)")

class UnderwritingEmissionsResult(BaseModel):
    """Complete underwriting portfolio emissions result.

    Top-level result containing gross/net totals, line-of-business
    breakdown, sector breakdown, and full audit trail.

    Attributes:
        result_id: Unique result identifier.
        reporting_year: Reporting year.
        total_written_premium: Total portfolio GWP (EUR).
        total_gross_emissions: Total gross emissions (tCO2e).
        total_net_emissions: Total net emissions (tCO2e).
        total_reinsurance_cession: Total reinsurance cession (tCO2e).
        total_claims_emissions: Total claims emissions (tCO2e).
        portfolio_emission_intensity: Portfolio-level intensity.
        weighted_avg_data_quality: Portfolio weighted avg DQ.
        lob_breakdown: Breakdown by line of business.
        sector_breakdown: Breakdown by NACE sector.
        policy_results: Individual policy results.
        total_policies: Total policies processed.
        methodology_notes: Methodology notes for disclosure.
        processing_time_ms: Processing time (ms).
        engine_version: Engine version string.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    reporting_year: int = Field(default=2024, description="Reporting year")
    total_written_premium: float = Field(
        default=0.0, description="Total portfolio GWP (EUR)",
    )
    total_gross_emissions: float = Field(
        default=0.0, description="Total gross emissions (tCO2e)",
    )
    total_net_emissions: float = Field(
        default=0.0, description="Total net emissions (tCO2e)",
    )
    total_reinsurance_cession: float = Field(
        default=0.0, description="Total reinsurance cession (tCO2e)",
    )
    total_claims_emissions: float = Field(
        default=0.0, description="Total claims emissions (tCO2e)",
    )
    portfolio_emission_intensity: float = Field(
        default=0.0, description="Portfolio intensity (tCO2e / EUR M GWP)",
    )
    weighted_avg_data_quality: float = Field(
        default=5.0, ge=1.0, le=5.0,
        description="Portfolio weighted average DQ score",
    )
    lob_breakdown: List[LineOfBusinessResult] = Field(
        default_factory=list, description="Breakdown by line of business",
    )
    sector_breakdown: List[SectorBreakdown] = Field(
        default_factory=list, description="Breakdown by NACE sector",
    )
    policy_results: List[PolicyEmissionsResult] = Field(
        default_factory=list, description="Individual policy results",
    )
    total_policies: int = Field(default=0, ge=0, description="Total policies processed")
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology notes",
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class UnderwritingConfig(BaseModel):
    """Configuration for the InsuranceUnderwritingEngine.

    Controls emission calculation methods, sector intensities,
    reinsurance treatment, and reporting parameters.

    Attributes:
        reporting_year: Reporting year.
        include_scope3: Whether to include Scope 3 of insured entities.
        default_calculation_method: Default emission calculation method.
        sector_emission_intensities: Override sector intensity factors.
        vehicle_emission_factors: Override vehicle emission factors.
        property_emission_factors: Override property emission factors.
        apply_reinsurance_adjustment: Whether to compute net emissions.
        default_reinsurance_cession_pct: Default cession % if unknown.
        include_claims_emissions: Whether to include claims emissions.
        min_premium_for_inclusion: Minimum premium to include policy.
        precision_decimal_places: Decimal places for rounding.
    """
    reporting_year: int = Field(default=2024, description="Reporting year")
    include_scope3: bool = Field(
        default=False,
        description="Include Scope 3 of insured entities",
    )
    default_calculation_method: EmissionCalculationMethod = Field(
        default=EmissionCalculationMethod.PREMIUM_BASED,
        description="Default emission calculation method",
    )
    sector_emission_intensities: Dict[str, float] = Field(
        default_factory=lambda: dict(SECTOR_EMISSION_INTENSITY),
        description="Sector emission intensity factors (tCO2e / EUR M premium)",
    )
    vehicle_emission_factors: Dict[str, float] = Field(
        default_factory=lambda: dict(VEHICLE_EMISSION_FACTORS),
        description="Vehicle emission factors (tCO2e / vehicle / year)",
    )
    property_emission_factors: Dict[str, float] = Field(
        default_factory=lambda: dict(PROPERTY_EMISSION_FACTORS),
        description="Property emission factors (tCO2e / EUR M insured value)",
    )
    apply_reinsurance_adjustment: bool = Field(
        default=True,
        description="Whether to compute net (after reinsurance) emissions",
    )
    default_reinsurance_cession_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Default reinsurance cession percentage if not specified",
    )
    include_claims_emissions: bool = Field(
        default=True,
        description="Whether to include claims-linked emissions",
    )
    min_premium_for_inclusion: float = Field(
        default=0.0, ge=0.0,
        description="Minimum premium to include a policy in calculations",
    )
    precision_decimal_places: int = Field(
        default=4, ge=0, le=10,
        description="Decimal places for rounding",
    )

# ---------------------------------------------------------------------------
# NACE sector name lookup
# ---------------------------------------------------------------------------

NACE_SECTOR_NAMES: Dict[str, str] = {
    "A": "Agriculture, forestry and fishing",
    "B": "Mining and quarrying",
    "C": "Manufacturing",
    "D": "Electricity, gas, steam and air conditioning supply",
    "E": "Water supply, sewerage, waste management",
    "F": "Construction",
    "G": "Wholesale and retail trade",
    "H": "Transportation and storage",
    "I": "Accommodation and food service",
    "J": "Information and communication",
    "K": "Financial and insurance activities",
    "L": "Real estate activities",
    "M": "Professional, scientific and technical",
    "N": "Administrative and support services",
    "O": "Public administration and defence",
    "P": "Education",
    "Q": "Human health and social work",
    "R": "Arts, entertainment and recreation",
    "S": "Other service activities",
    "T": "Activities of households",
    "U": "Activities of extraterritorial organisations",
}

# ---------------------------------------------------------------------------
# Model rebuilds for forward references
# ---------------------------------------------------------------------------

ReinsuranceAdjustment.model_rebuild()
ClaimsEmissions.model_rebuild()
PolicyData.model_rebuild()
PolicyEmissionsResult.model_rebuild()
LineOfBusinessResult.model_rebuild()
SectorBreakdown.model_rebuild()
UnderwritingEmissionsResult.model_rebuild()
UnderwritingConfig.model_rebuild()

# ---------------------------------------------------------------------------
# InsuranceUnderwritingEngine
# ---------------------------------------------------------------------------

class InsuranceUnderwritingEngine:
    """
    Insurance underwriting emissions engine implementing PCAF Part C.

    Calculates emissions associated with insurance underwriting using
    the "follow the risk" methodology.  Premium-based attribution
    determines the insurer's share of insured entities' emissions.
    Supports gross and net (reinsurance) calculations, claims-linked
    emissions, and breakdowns by line of business and NACE sector.

    Zero-Hallucination Guarantees:
        - All calculations use deterministic Python arithmetic
        - Premium shares are pure ratios of premium data
        - Reinsurance adjustments use contractual cession percentages
        - SHA-256 provenance hash on every result
        - No LLM involvement in any calculation path

    Example:
        >>> config = UnderwritingConfig(reporting_year=2024)
        >>> engine = InsuranceUnderwritingEngine(config)
        >>> policies = [PolicyData(
        ...     line_of_business=InsuranceLine.COMMERCIAL_MOTOR,
        ...     written_premium=500_000,
        ...     total_market_premium=50_000_000,
        ...     insured_scope1=10_000,
        ...     nace_sector="H",
        ... )]
        >>> result = engine.calculate_underwriting_emissions(policies)
        >>> assert result.total_gross_emissions > 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize InsuranceUnderwritingEngine.

        Args:
            config: Optional UnderwritingConfig or dict.
        """
        if config and isinstance(config, dict):
            self.config = UnderwritingConfig(**config)
        elif config and isinstance(config, UnderwritingConfig):
            self.config = config
        else:
            self.config = UnderwritingConfig()

        self._policies: List[PolicyData] = []
        self._policy_results: Dict[str, PolicyEmissionsResult] = {}

        logger.info(
            "InsuranceUnderwritingEngine initialized (version=%s, year=%d)",
            _MODULE_VERSION,
            self.config.reporting_year,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_underwriting_emissions(
        self,
        policies: List[PolicyData],
    ) -> UnderwritingEmissionsResult:
        """Calculate insurance-associated emissions for the portfolio.

        Pipeline:
        1. Filter policies by minimum premium
        2. Calculate emissions per policy (premium or claims based)
        3. Apply reinsurance adjustments
        4. Aggregate by line of business
        5. Aggregate by NACE sector
        6. Compute portfolio totals and intensity

        Args:
            policies: List of PolicyData for each insured risk.

        Returns:
            UnderwritingEmissionsResult with full breakdown.

        Raises:
            ValueError: If policies list is empty.
        """
        start = utcnow()

        if not policies:
            raise ValueError("Policies list cannot be empty")

        self._policies = policies
        self._policy_results = {}

        logger.info(
            "Calculating underwriting emissions for %d policies",
            len(policies),
        )

        # Step 1: Filter by minimum premium
        filtered = [
            p for p in policies
            if p.written_premium >= self.config.min_premium_for_inclusion
        ]

        # Step 2: Process each policy
        all_results: List[PolicyEmissionsResult] = []
        for policy in filtered:
            result = self._process_single_policy(policy)
            self._policy_results[policy.policy_id] = result
            all_results.append(result)

        # Step 3: Aggregate by line of business
        lob_breakdown = self._aggregate_by_lob(all_results)

        # Step 4: Aggregate by sector
        sector_breakdown = self._aggregate_by_sector(all_results, filtered)

        # Step 5: Portfolio totals
        total_gwp = sum(r.written_premium for r in all_results)
        total_gross = _round_val(sum(r.gross_emissions for r in all_results))
        total_net = _round_val(sum(r.net_emissions for r in all_results))
        total_ri = _round_val(sum(r.reinsurance_cession_emissions for r in all_results))
        total_claims = _round_val(sum(r.claims_emissions for r in all_results))

        # Portfolio intensity
        gwp_m = total_gwp / 1_000_000.0 if total_gwp > 0 else 0.0
        portfolio_intensity = _safe_divide(total_gross, gwp_m)

        # Weighted avg data quality
        weighted_dq = self._compute_weighted_dq(all_results, total_gwp)

        # Methodology notes
        notes = self._generate_methodology_notes(all_results, filtered, policies)

        end = utcnow()
        processing_ms = (end - start).total_seconds() * 1000.0

        result = UnderwritingEmissionsResult(
            reporting_year=self.config.reporting_year,
            total_written_premium=_round_val(total_gwp, 2),
            total_gross_emissions=total_gross,
            total_net_emissions=total_net,
            total_reinsurance_cession=total_ri,
            total_claims_emissions=total_claims,
            portfolio_emission_intensity=_round_val(portfolio_intensity),
            weighted_avg_data_quality=_round_val(weighted_dq, 2),
            lob_breakdown=lob_breakdown,
            sector_breakdown=sector_breakdown,
            policy_results=all_results,
            total_policies=len(all_results),
            methodology_notes=notes,
            processing_time_ms=_round_val(processing_ms, 2),
        )

        result.provenance_hash = _compute_hash(result)
        logger.info(
            "Underwriting emissions: gross=%.2f net=%.2f tCO2e (%d policies)",
            result.total_gross_emissions,
            result.total_net_emissions,
            result.total_policies,
        )
        return result

    def calculate_single_policy(
        self, policy: PolicyData,
    ) -> PolicyEmissionsResult:
        """Calculate emissions for a single insurance policy.

        Args:
            policy: PolicyData for the insured risk.

        Returns:
            PolicyEmissionsResult with attribution and provenance.
        """
        return self._process_single_policy(policy)

    def compute_premium_share(
        self, policy: PolicyData,
    ) -> float:
        """Compute the premium share attribution factor for a policy.

        Premium Share = Insurer Written Premium / Total Market Premium

        Args:
            policy: PolicyData with premium information.

        Returns:
            Premium share (0.0 to 1.0).
        """
        return self._compute_premium_share(policy)

    def compute_reinsurance_adjustment(
        self,
        gross_emissions: float,
        reinsurance: ReinsuranceAdjustment,
    ) -> Tuple[float, float]:
        """Compute reinsurance cession and net emissions.

        Args:
            gross_emissions: Gross attributed emissions.
            reinsurance: Reinsurance arrangement details.

        Returns:
            Tuple of (cession_emissions, net_emissions).
        """
        return self._apply_reinsurance(gross_emissions, reinsurance)

    # ------------------------------------------------------------------
    # Internal: Single Policy Processing
    # ------------------------------------------------------------------

    def _process_single_policy(self, policy: PolicyData) -> PolicyEmissionsResult:
        """Process a single insurance policy.

        Determines the appropriate calculation method and computes
        gross emissions, reinsurance adjustments, and net emissions.

        Args:
            policy: PolicyData input.

        Returns:
            PolicyEmissionsResult.
        """
        method = policy.calculation_method

        # Determine insured entity emissions
        insured_emissions = self._get_insured_emissions(policy)

        # Compute gross emissions based on method
        if method == EmissionCalculationMethod.PREMIUM_BASED:
            gross, note = self._calculate_premium_based(policy, insured_emissions)
        elif method == EmissionCalculationMethod.CLAIMS_BASED:
            gross, note = self._calculate_claims_based(policy)
        elif method == EmissionCalculationMethod.HYBRID:
            gross, note = self._calculate_hybrid(policy, insured_emissions)
        elif method == EmissionCalculationMethod.SECTOR_AVERAGE:
            gross, note = self._calculate_sector_average(policy)
        else:
            gross, note = self._calculate_premium_based(policy, insured_emissions)

        gross = _round_val(gross)

        # Reinsurance adjustment
        cession = 0.0
        net = gross
        if self.config.apply_reinsurance_adjustment and policy.reinsurance:
            cession, net = self._apply_reinsurance(gross, policy.reinsurance)

        # Claims emissions (supplementary)
        claims_em = 0.0
        if self.config.include_claims_emissions and policy.claims_data:
            claims_em = _round_val(policy.claims_data.total_claims_emissions)

        # Premium share for reference
        premium_share = self._compute_premium_share(policy)

        # Emission intensity
        gwp_m = policy.written_premium / 1_000_000.0 if policy.written_premium > 0 else 0.0
        intensity = _safe_divide(gross, gwp_m)

        result = PolicyEmissionsResult(
            policy_id=policy.policy_id,
            policyholder_name=policy.policyholder_name,
            line_of_business=policy.line_of_business,
            premium_share=_round_val(premium_share, 6),
            gross_emissions=gross,
            reinsurance_cession_emissions=_round_val(cession),
            net_emissions=_round_val(net),
            claims_emissions=claims_em,
            emission_intensity=_round_val(intensity),
            written_premium=_round_val(policy.written_premium, 2),
            data_quality_score=policy.data_quality_score,
            calculation_method=method,
            methodology_note=note,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Emission Calculation Methods
    # ------------------------------------------------------------------

    def _get_insured_emissions(self, policy: PolicyData) -> float:
        """Get total insured entity emissions (deterministic).

        Args:
            policy: PolicyData with emission data.

        Returns:
            Total insured emissions (tCO2e).
        """
        total = policy.insured_scope1 + policy.insured_scope2
        if self.config.include_scope3:
            total += policy.insured_scope3
        return total

    def _compute_premium_share(self, policy: PolicyData) -> float:
        """Compute premium share attribution factor.

        Premium Share = Written Premium / Total Market Premium

        If total market premium is not provided, premium share
        defaults to 1.0 (full attribution).

        Args:
            policy: PolicyData.

        Returns:
            Premium share (0.0 to 1.0).
        """
        if policy.total_market_premium > 0:
            share = _safe_divide(
                policy.written_premium, policy.total_market_premium
            )
            return min(share, 1.0)
        return 1.0

    def _calculate_premium_based(
        self,
        policy: PolicyData,
        insured_emissions: float,
    ) -> Tuple[float, str]:
        """Calculate emissions using premium-based attribution.

        Gross Emissions = Premium Share * Insured Entity Emissions

        If insured emissions are not available, falls back to
        sector intensity * premium.

        Args:
            policy: PolicyData.
            insured_emissions: Total insured entity emissions.

        Returns:
            Tuple of (gross_emissions, methodology_note).
        """
        premium_share = self._compute_premium_share(policy)

        if insured_emissions > 0:
            gross = premium_share * insured_emissions
            note = (
                f"Premium-based: share={premium_share:.6f} * "
                f"insured_emissions={insured_emissions:.2f}"
            )
            return gross, note

        # Fallback: use sector intensity
        return self._calculate_sector_average(policy)

    def _calculate_claims_based(
        self,
        policy: PolicyData,
    ) -> Tuple[float, str]:
        """Calculate emissions using claims-linked data.

        Uses claims data directly if available, otherwise falls
        back to sector average.

        Args:
            policy: PolicyData with claims data.

        Returns:
            Tuple of (gross_emissions, methodology_note).
        """
        if policy.claims_data and policy.claims_data.total_claims_emissions > 0:
            gross = policy.claims_data.total_claims_emissions
            note = (
                f"Claims-based: {policy.claims_data.claims_count} claims, "
                f"total={gross:.2f} tCO2e"
            )
            return gross, note

        # Fallback
        return self._calculate_sector_average(policy)

    def _calculate_hybrid(
        self,
        policy: PolicyData,
        insured_emissions: float,
    ) -> Tuple[float, str]:
        """Calculate emissions using hybrid (premium + claims) method.

        Averages the premium-based and claims-based estimates when
        both are available. Otherwise uses whichever is available.

        Args:
            policy: PolicyData.
            insured_emissions: Total insured entity emissions.

        Returns:
            Tuple of (gross_emissions, methodology_note).
        """
        premium_gross, _ = self._calculate_premium_based(policy, insured_emissions)
        claims_gross, _ = self._calculate_claims_based(policy)

        if premium_gross > 0 and claims_gross > 0:
            gross = (premium_gross + claims_gross) / 2.0
            note = (
                f"Hybrid: avg(premium={premium_gross:.2f}, "
                f"claims={claims_gross:.2f})"
            )
            return gross, note
        elif premium_gross > 0:
            return premium_gross, "Hybrid fallback to premium-based"
        elif claims_gross > 0:
            return claims_gross, "Hybrid fallback to claims-based"
        else:
            return self._calculate_sector_average(policy)

    def _calculate_sector_average(
        self,
        policy: PolicyData,
    ) -> Tuple[float, str]:
        """Calculate emissions using sector average intensity.

        Uses NACE sector emission intensity factors applied to
        the written premium.

        Falls back to line-of-business specific calculations for
        motor and property lines.

        Args:
            policy: PolicyData.

        Returns:
            Tuple of (gross_emissions, methodology_note).
        """
        # Line-specific calculations
        if policy.line_of_business in (
            InsuranceLine.COMMERCIAL_MOTOR, InsuranceLine.PERSONAL_MOTOR,
        ):
            return self._calculate_motor_emissions(policy)

        if policy.line_of_business == InsuranceLine.COMMERCIAL_PROPERTY:
            return self._calculate_property_emissions(policy)

        if policy.line_of_business == InsuranceLine.PROJECT_INSURANCE:
            return self._calculate_project_emissions(policy)

        # Generic sector average
        nace = policy.nace_sector.upper()[:1] if policy.nace_sector else ""
        intensity = self.config.sector_emission_intensities.get(nace, 150.0)
        gwp_m = policy.written_premium / 1_000_000.0
        gross = intensity * gwp_m
        note = (
            f"Sector average: NACE={nace}, intensity={intensity:.1f} tCO2e/EUR M, "
            f"GWP_M={gwp_m:.4f}"
        )
        return gross, note

    def _calculate_motor_emissions(
        self,
        policy: PolicyData,
    ) -> Tuple[float, str]:
        """Calculate motor insurance emissions.

        Uses vehicle count and type-specific emission factors when
        available; otherwise falls back to premium-based sector average.

        Args:
            policy: PolicyData for motor line.

        Returns:
            Tuple of (gross_emissions, methodology_note).
        """
        if policy.vehicle_count > 0 and policy.vehicle_type:
            factor = self.config.vehicle_emission_factors.get(
                policy.vehicle_type, 2.3  # default to petrol car
            )
            premium_share = self._compute_premium_share(policy)
            gross = premium_share * policy.vehicle_count * factor
            note = (
                f"Motor: {policy.vehicle_count} vehicles x "
                f"{factor:.1f} tCO2e/vehicle x share={premium_share:.6f}"
            )
            return gross, note

        # Fallback to sector intensity for transport
        intensity = self.config.sector_emission_intensities.get("H", 2800.0)
        gwp_m = policy.written_premium / 1_000_000.0
        gross = intensity * gwp_m
        note = f"Motor sector average: intensity={intensity:.1f}, GWP_M={gwp_m:.4f}"
        return gross, note

    def _calculate_property_emissions(
        self,
        policy: PolicyData,
    ) -> Tuple[float, str]:
        """Calculate commercial property insurance emissions.

        Uses property type and insured value when available.

        Args:
            policy: PolicyData for property line.

        Returns:
            Tuple of (gross_emissions, methodology_note).
        """
        if policy.insured_value > 0 and policy.property_type:
            factor = self.config.property_emission_factors.get(
                policy.property_type, 40.0  # default mixed use
            )
            premium_share = self._compute_premium_share(policy)
            value_m = policy.insured_value / 1_000_000.0
            gross = premium_share * factor * value_m
            note = (
                f"Property: type={policy.property_type}, "
                f"value_M={value_m:.2f}, factor={factor:.1f}, "
                f"share={premium_share:.6f}"
            )
            return gross, note

        # Fallback
        intensity = self.config.sector_emission_intensities.get("L", 400.0)
        gwp_m = policy.written_premium / 1_000_000.0
        gross = intensity * gwp_m
        note = f"Property sector average: intensity={intensity:.1f}, GWP_M={gwp_m:.4f}"
        return gross, note

    def _calculate_project_emissions(
        self,
        policy: PolicyData,
    ) -> Tuple[float, str]:
        """Calculate project insurance emissions.

        Uses project total emissions with premium share attribution.

        Args:
            policy: PolicyData for project insurance line.

        Returns:
            Tuple of (gross_emissions, methodology_note).
        """
        if policy.project_total_emissions > 0:
            premium_share = self._compute_premium_share(policy)
            gross = premium_share * policy.project_total_emissions
            note = (
                f"Project: total_emissions={policy.project_total_emissions:.2f}, "
                f"share={premium_share:.6f}"
            )
            return gross, note

        # Fallback to construction sector
        intensity = self.config.sector_emission_intensities.get("F", 950.0)
        gwp_m = policy.written_premium / 1_000_000.0
        gross = intensity * gwp_m
        note = f"Project sector average: intensity={intensity:.1f}, GWP_M={gwp_m:.4f}"
        return gross, note

    # ------------------------------------------------------------------
    # Internal: Reinsurance Adjustment
    # ------------------------------------------------------------------

    def _apply_reinsurance(
        self,
        gross_emissions: float,
        reinsurance: ReinsuranceAdjustment,
    ) -> Tuple[float, float]:
        """Apply reinsurance cession adjustment to get net emissions.

        Net Emissions = Gross - (Gross * Cession%)

        Args:
            gross_emissions: Gross attributed emissions.
            reinsurance: Reinsurance arrangement details.

        Returns:
            Tuple of (cession_emissions, net_emissions).
        """
        cession_pct = reinsurance.cession_pct
        if cession_pct <= 0.0:
            cession_pct = self.config.default_reinsurance_cession_pct

        cession = gross_emissions * (cession_pct / 100.0)
        net = gross_emissions - cession
        return _round_val(cession), _round_val(net)

    # ------------------------------------------------------------------
    # Internal: Aggregation
    # ------------------------------------------------------------------

    def _aggregate_by_lob(
        self,
        results: List[PolicyEmissionsResult],
    ) -> List[LineOfBusinessResult]:
        """Aggregate policy results by line of business.

        Args:
            results: List of PolicyEmissionsResult.

        Returns:
            List of LineOfBusinessResult.
        """
        groups: Dict[InsuranceLine, List[PolicyEmissionsResult]] = defaultdict(list)
        for r in results:
            groups[r.line_of_business].append(r)

        total_gwp = sum(r.written_premium for r in results)

        breakdowns: List[LineOfBusinessResult] = []
        for lob, group in groups.items():
            lob_gwp = sum(r.written_premium for r in group)
            lob_gross = sum(r.gross_emissions for r in group)
            lob_net = sum(r.net_emissions for r in group)
            lob_claims = sum(r.claims_emissions for r in group)
            lob_ri = sum(r.reinsurance_cession_emissions for r in group)

            gwp_m = lob_gwp / 1_000_000.0 if lob_gwp > 0 else 0.0
            intensity = _safe_divide(lob_gross, gwp_m)

            dq_scores = [r.data_quality_score for r in group]
            avg_dq = sum(dq_scores) / len(dq_scores) if dq_scores else 5.0

            weight = _safe_pct(lob_gwp, total_gwp)

            breakdowns.append(LineOfBusinessResult(
                line_of_business=lob,
                policy_count=len(group),
                total_written_premium=_round_val(lob_gwp, 2),
                total_gross_emissions=_round_val(lob_gross),
                total_net_emissions=_round_val(lob_net),
                total_claims_emissions=_round_val(lob_claims),
                total_reinsurance_cession=_round_val(lob_ri),
                weighted_emission_intensity=_round_val(intensity),
                avg_data_quality_score=_round_val(avg_dq, 2),
                weight_in_portfolio_pct=_round_val(weight, 2),
            ))

        return breakdowns

    def _aggregate_by_sector(
        self,
        results: List[PolicyEmissionsResult],
        policies: List[PolicyData],
    ) -> List[SectorBreakdown]:
        """Aggregate results by NACE sector.

        Args:
            results: List of PolicyEmissionsResult.
            policies: Original PolicyData (for sector info).

        Returns:
            List of SectorBreakdown.
        """
        # Build policy_id -> result and policy_id -> policy maps
        result_map: Dict[str, PolicyEmissionsResult] = {
            r.policy_id: r for r in results
        }
        policy_map: Dict[str, PolicyData] = {
            p.policy_id: p for p in policies
        }

        # Group by NACE sector
        sector_groups: Dict[str, List[Tuple[PolicyData, PolicyEmissionsResult]]] = defaultdict(list)
        for pid, res in result_map.items():
            pol = policy_map.get(pid)
            if pol:
                nace = pol.nace_sector.upper()[:1] if pol.nace_sector else "UNKNOWN"
                sector_groups[nace].append((pol, res))

        total_gwp = sum(r.written_premium for r in results)

        breakdowns: List[SectorBreakdown] = []
        for nace, group in sorted(sector_groups.items()):
            gwp = sum(r.written_premium for _, r in group)
            gross = sum(r.gross_emissions for _, r in group)
            net = sum(r.net_emissions for _, r in group)
            gwp_m = gwp / 1_000_000.0 if gwp > 0 else 0.0
            intensity = _safe_divide(gross, gwp_m)
            weight = _safe_pct(gwp, total_gwp)
            name = NACE_SECTOR_NAMES.get(nace, "Unknown sector")

            breakdowns.append(SectorBreakdown(
                nace_sector=nace,
                sector_name=name,
                policy_count=len(group),
                total_written_premium=_round_val(gwp, 2),
                total_gross_emissions=_round_val(gross),
                total_net_emissions=_round_val(net),
                emission_intensity=_round_val(intensity),
                weight_pct=_round_val(weight, 2),
            ))

        return breakdowns

    def _compute_weighted_dq(
        self,
        results: List[PolicyEmissionsResult],
        total_gwp: float,
    ) -> float:
        """Compute premium-weighted average data quality score.

        Args:
            results: List of PolicyEmissionsResult.
            total_gwp: Total GWP for weights.

        Returns:
            Weighted average DQ score (1.0-5.0).
        """
        if total_gwp <= 0:
            return 5.0

        weighted_sum = 0.0
        for r in results:
            weight = r.written_premium / total_gwp
            weighted_sum += weight * r.data_quality_score

        return max(1.0, min(5.0, weighted_sum))

    # ------------------------------------------------------------------
    # Internal: Methodology Notes
    # ------------------------------------------------------------------

    def _generate_methodology_notes(
        self,
        results: List[PolicyEmissionsResult],
        filtered: List[PolicyData],
        original: List[PolicyData],
    ) -> List[str]:
        """Generate methodology disclosure notes.

        Args:
            results: Policy emission results.
            filtered: Filtered policies (after minimum premium).
            original: Original input policies.

        Returns:
            List of methodology note strings.
        """
        notes: List[str] = [
            "Methodology: PCAF Insurance-Associated Emissions Standard (Part C)",
            f"Reporting year: {self.config.reporting_year}",
            f"Total policies submitted: {len(original)}",
            f"Policies included (after premium filter): {len(filtered)}",
            f"Scope 3 included: {self.config.include_scope3}",
            f"Reinsurance adjustment: {self.config.apply_reinsurance_adjustment}",
        ]

        # Method distribution
        method_counts: Dict[str, int] = defaultdict(int)
        for r in results:
            method_counts[r.calculation_method.value] += 1
        for method, count in sorted(method_counts.items()):
            notes.append(f"Calculation method {method}: {count} policy(ies)")

        # LoB distribution
        lob_counts: Dict[str, int] = defaultdict(int)
        for r in results:
            lob_counts[r.line_of_business.value] += 1
        for lob, count in sorted(lob_counts.items()):
            notes.append(f"Line of business {lob}: {count} policy(ies)")

        return notes
