"""
GL-013: California SB 253 Climate Disclosure Agent

This module implements the SB253 Climate Corporate Data Accountability Act
compliance agent for calculating and reporting GHG emissions.

The agent supports:
- Scope 1 direct emissions (stationary combustion, mobile combustion, process, fugitive)
- Scope 2 indirect emissions (location-based and market-based methods)
- Scope 3 value chain emissions (all 15 GHG Protocol categories)
- CARB portal filing format generation
- Third-party assurance package preparation

Regulatory Context:
- California SB 253 (Climate Corporate Data Accountability Act)
- Applies to companies with >$1B revenue doing business in California
- First Scope 1&2 reports due: June 30, 2026
- First Scope 3 reports due: June 30, 2027
- Enforcement agency: California Air Resources Board (CARB)

Example:
    >>> agent = SB253DisclosureAgent()
    >>> result = agent.run(SB253ReportInput(
    ...     company_info=CompanyInfo(
    ...         company_name="Acme Corp",
    ...         ein="12-3456789",
    ...         total_revenue_usd=2000000000,
    ...         california_revenue_usd=500000000,
    ...         naics_code="331110"
    ...     ),
    ...     fiscal_year=2025,
    ...     scope1_sources=[...],
    ...     scope2_sources=[...],
    ...     scope3_data={...}
    ... ))
    >>> print(f"Total emissions: {result.total_emissions_mtco2e} MTCO2e")
"""

import hashlib
import json
import logging
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator, root_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class OrganizationalBoundary(str, Enum):
    """GHG Protocol organizational boundary approaches."""
    EQUITY_SHARE = "equity_share"
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"


class FuelType(str, Enum):
    """Supported fuel types for Scope 1 calculations."""
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    PROPANE = "propane"
    FUEL_OIL_2 = "fuel_oil_2"
    COAL = "coal"
    LPG = "lpg"


class FuelUnit(str, Enum):
    """Fuel quantity units."""
    THERMS = "therms"
    GALLONS = "gallons"
    MMBTU = "MMBtu"
    KWH = "kWh"
    TONS = "tons"
    KG = "kg"
    LITERS = "liters"
    SCF = "scf"
    CCF = "ccf"


class SourceCategory(str, Enum):
    """Scope 1 emission source categories."""
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"


class Scope2Method(str, Enum):
    """Scope 2 calculation methods."""
    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"


class Scope3Category(int, Enum):
    """GHG Protocol Scope 3 categories."""
    PURCHASED_GOODS_SERVICES = 1
    CAPITAL_GOODS = 2
    FUEL_ENERGY_ACTIVITIES = 3
    UPSTREAM_TRANSPORTATION = 4
    WASTE_GENERATED = 5
    BUSINESS_TRAVEL = 6
    EMPLOYEE_COMMUTING = 7
    UPSTREAM_LEASED_ASSETS = 8
    DOWNSTREAM_TRANSPORTATION = 9
    PROCESSING_SOLD_PRODUCTS = 10
    USE_OF_SOLD_PRODUCTS = 11
    END_OF_LIFE_TREATMENT = 12
    DOWNSTREAM_LEASED_ASSETS = 13
    FRANCHISES = 14
    INVESTMENTS = 15


class CalculationMethod(str, Enum):
    """Scope 3 calculation methods."""
    SPEND_BASED = "spend_based"
    SUPPLIER_SPECIFIC = "supplier_specific"
    AVERAGE_DATA = "average_data"
    HYBRID = "hybrid"


class DataQualityScore(int, Enum):
    """GHG Protocol data quality indicators (1=best, 5=worst)."""
    VERY_GOOD = 1
    GOOD = 2
    FAIR = 3
    POOR = 4
    VERY_POOR = 5


class GWPSet(str, Enum):
    """IPCC Global Warming Potential assessment reports."""
    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"


class AssuranceLevel(str, Enum):
    """Third-party assurance levels."""
    LIMITED = "limited"
    REASONABLE = "reasonable"


class RefrigerantType(str, Enum):
    """Common refrigerants for fugitive emissions."""
    R134A = "R-134a"
    R410A = "R-410A"
    R22 = "R-22"
    R404A = "R-404A"
    R407C = "R-407C"
    R507A = "R-507A"
    R32 = "R-32"
    CO2 = "CO2"
    AMMONIA = "ammonia"


# =============================================================================
# INPUT MODELS
# =============================================================================

class CompanyInfo(BaseModel):
    """Company identification and applicability data."""

    company_name: str = Field(..., min_length=1, description="Legal company name")
    ein: str = Field(..., pattern=r"^[0-9]{2}-[0-9]{7}$", description="Employer Identification Number")
    total_revenue_usd: float = Field(..., ge=1_000_000_000, description="Total annual revenue (must be >$1B)")
    california_revenue_usd: Optional[float] = Field(None, ge=0, description="California revenue")
    naics_code: str = Field(..., pattern=r"^[0-9]{6}$", description="6-digit NAICS code")
    organizational_boundary: OrganizationalBoundary = Field(
        OrganizationalBoundary.OPERATIONAL_CONTROL,
        description="Organizational boundary approach"
    )
    parent_company: Optional[str] = Field(None, description="Parent company name if subsidiary")
    california_facilities: Optional[int] = Field(None, ge=0, description="Number of CA facilities")

    @validator('total_revenue_usd')
    def validate_sb253_threshold(cls, v):
        """Validate company meets SB 253 $1B revenue threshold."""
        if v < 1_000_000_000:
            raise ValueError("SB 253 applies only to companies with >$1B annual revenue")
        return v


class FacilityInfo(BaseModel):
    """Facility location and operational data."""

    facility_id: str = Field(..., description="Unique facility identifier")
    facility_name: str = Field(..., description="Facility name")
    egrid_subregion: str = Field(..., description="EPA eGRID subregion code")
    california_facility: bool = Field(..., description="Whether facility is in California")
    address_city: str = Field(..., description="City")
    address_state: str = Field(..., description="State code")
    address_zip: str = Field(..., description="ZIP code")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")


class Scope1Source(BaseModel):
    """Scope 1 emission source data."""

    facility_id: str = Field(..., description="Facility identifier")
    source_category: SourceCategory = Field(..., description="Emission source category")
    fuel_type: FuelType = Field(..., description="Fuel type")
    quantity: float = Field(..., ge=0, description="Fuel quantity consumed")
    unit: FuelUnit = Field(..., description="Unit of measurement")
    source_description: Optional[str] = Field(None, description="Description of emission source")

    # For fugitive emissions
    refrigerant_type: Optional[RefrigerantType] = Field(None, description="Refrigerant type for fugitive emissions")
    refrigerant_charge_kg: Optional[float] = Field(None, ge=0, description="Total refrigerant charge")
    refrigerant_loss_kg: Optional[float] = Field(None, ge=0, description="Annual refrigerant loss")


class Scope2Source(BaseModel):
    """Scope 2 electricity consumption data."""

    facility_id: str = Field(..., description="Facility identifier")
    kwh: float = Field(..., ge=0, description="Electricity consumption in kWh")
    egrid_subregion: str = Field(..., description="EPA eGRID subregion")
    renewable_percentage: float = Field(0.0, ge=0, le=100, description="% from renewable sources")
    has_ppa: bool = Field(False, description="Has Power Purchase Agreement")
    ppa_emissions_factor: Optional[float] = Field(None, ge=0, description="PPA emission factor (kg CO2e/kWh)")
    has_recs: bool = Field(False, description="Has Renewable Energy Certificates")
    rec_mwh: Optional[float] = Field(None, ge=0, description="RECs in MWh")
    utility_name: Optional[str] = Field(None, description="Utility provider name")
    utility_emission_factor: Optional[float] = Field(None, ge=0, description="Utility-specific factor (kg CO2e/kWh)")


class Scope3CategoryData(BaseModel):
    """Activity data for a single Scope 3 category."""

    category: Scope3Category = Field(..., description="Scope 3 category number")
    calculation_method: CalculationMethod = Field(..., description="Calculation method used")
    data_quality_score: DataQualityScore = Field(DataQualityScore.FAIR, description="Data quality indicator")

    # Spend-based data
    spend_usd: Optional[float] = Field(None, ge=0, description="Total spend in USD")
    naics_code: Optional[str] = Field(None, description="NAICS code for spend category")

    # Activity-based data
    activity_data: Optional[Dict[str, Any]] = Field(None, description="Category-specific activity data")

    # Supplier-specific data
    supplier_emissions_kgco2e: Optional[float] = Field(None, ge=0, description="Primary supplier emissions")

    # Metadata
    notes: Optional[str] = Field(None, description="Calculation notes")


class Scope3Data(BaseModel):
    """Complete Scope 3 value chain data for all 15 categories."""

    purchased_goods_services: Optional[Scope3CategoryData] = Field(None, description="Category 1")
    capital_goods: Optional[Scope3CategoryData] = Field(None, description="Category 2")
    fuel_energy_activities: Optional[Scope3CategoryData] = Field(None, description="Category 3")
    upstream_transportation: Optional[Scope3CategoryData] = Field(None, description="Category 4")
    waste_generated: Optional[Scope3CategoryData] = Field(None, description="Category 5")
    business_travel: Optional[Scope3CategoryData] = Field(None, description="Category 6")
    employee_commuting: Optional[Scope3CategoryData] = Field(None, description="Category 7")
    upstream_leased_assets: Optional[Scope3CategoryData] = Field(None, description="Category 8")
    downstream_transportation: Optional[Scope3CategoryData] = Field(None, description="Category 9")
    processing_sold_products: Optional[Scope3CategoryData] = Field(None, description="Category 10")
    use_of_sold_products: Optional[Scope3CategoryData] = Field(None, description="Category 11")
    end_of_life_treatment: Optional[Scope3CategoryData] = Field(None, description="Category 12")
    downstream_leased_assets: Optional[Scope3CategoryData] = Field(None, description="Category 13")
    franchises: Optional[Scope3CategoryData] = Field(None, description="Category 14")
    investments: Optional[Scope3CategoryData] = Field(None, description="Category 15")


class ReportingPeriod(BaseModel):
    """Reporting period configuration."""

    fiscal_year: int = Field(..., ge=2025, description="Fiscal year")
    start_date: date = Field(..., description="Period start date")
    end_date: date = Field(..., description="Period end date")

    @root_validator(skip_on_failure=True)
    def validate_dates(cls, values):
        """Validate date range."""
        start = values.get('start_date')
        end = values.get('end_date')
        if start and end and start >= end:
            raise ValueError("start_date must be before end_date")
        return values


class SB253ReportInput(BaseModel):
    """
    Complete input model for SB 253 Climate Disclosure Report.

    Attributes:
        company_info: Company identification and applicability data
        fiscal_year: Reporting fiscal year
        reporting_period: Optional custom reporting period
        facilities: List of facility data
        scope1_sources: Scope 1 emission sources
        scope2_sources: Scope 2 electricity consumption
        scope3_data: Scope 3 value chain data
        gwp_set: Global Warming Potential assessment report to use
        include_scope3: Whether to include Scope 3 (required from 2027)
        assurance_level: Third-party assurance level
    """

    company_info: CompanyInfo = Field(..., description="Company identification data")
    fiscal_year: int = Field(..., ge=2025, description="Reporting fiscal year")
    reporting_period: Optional[ReportingPeriod] = Field(None, description="Custom reporting period")

    facilities: List[FacilityInfo] = Field(default_factory=list, description="Facility data")
    scope1_sources: List[Scope1Source] = Field(default_factory=list, description="Scope 1 sources")
    scope2_sources: List[Scope2Source] = Field(default_factory=list, description="Scope 2 sources")
    scope3_data: Optional[Scope3Data] = Field(None, description="Scope 3 data")

    gwp_set: GWPSet = Field(GWPSet.AR5, description="GWP assessment report")
    include_scope3: bool = Field(False, description="Include Scope 3 emissions")
    assurance_level: AssuranceLevel = Field(AssuranceLevel.LIMITED, description="Assurance level")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('fiscal_year')
    def validate_fiscal_year(cls, v):
        """Validate fiscal year for SB 253 timeline."""
        if v < 2025:
            raise ValueError("SB 253 reporting starts from fiscal year 2025")
        return v


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class EmissionBreakdown(BaseModel):
    """Emission breakdown by gas type."""

    co2_kg: float = Field(..., description="CO2 emissions in kg")
    ch4_kg: float = Field(..., description="CH4 emissions in kg")
    n2o_kg: float = Field(..., description="N2O emissions in kg")
    co2e_kg: float = Field(..., description="Total CO2-equivalent in kg")
    co2e_mtco2e: float = Field(..., description="Total CO2-equivalent in metric tons")


class Scope1Result(BaseModel):
    """Scope 1 emissions calculation result."""

    total_emissions: EmissionBreakdown = Field(..., description="Total Scope 1 emissions")
    stationary_combustion: EmissionBreakdown = Field(..., description="Stationary combustion emissions")
    mobile_combustion: EmissionBreakdown = Field(..., description="Mobile combustion emissions")
    process_emissions: EmissionBreakdown = Field(..., description="Process emissions")
    fugitive_emissions: EmissionBreakdown = Field(..., description="Fugitive emissions")

    emissions_by_facility: Dict[str, float] = Field(..., description="Emissions by facility (kgCO2e)")
    emissions_by_fuel: Dict[str, float] = Field(..., description="Emissions by fuel type (kgCO2e)")

    emission_factors_used: List[Dict[str, Any]] = Field(..., description="Emission factors applied")


class Scope2Result(BaseModel):
    """Scope 2 emissions calculation result."""

    location_based: EmissionBreakdown = Field(..., description="Location-based emissions")
    market_based: EmissionBreakdown = Field(..., description="Market-based emissions")

    emissions_by_facility: Dict[str, Dict[str, float]] = Field(
        ..., description="Emissions by facility (location and market)"
    )
    emissions_by_grid: Dict[str, float] = Field(..., description="Emissions by eGRID subregion")

    total_electricity_mwh: float = Field(..., description="Total electricity consumption in MWh")
    renewable_mwh: float = Field(..., description="Renewable electricity in MWh")
    renewable_percentage: float = Field(..., description="Overall renewable percentage")

    egrid_factors_used: List[Dict[str, Any]] = Field(..., description="eGRID factors applied")


class Scope3CategoryResult(BaseModel):
    """Result for a single Scope 3 category."""

    category_number: int = Field(..., ge=1, le=15, description="Category number")
    category_name: str = Field(..., description="Category name")
    co2e_kg: float = Field(..., ge=0, description="Emissions in kgCO2e")
    co2e_mtco2e: float = Field(..., ge=0, description="Emissions in MTCO2e")

    calculation_method: str = Field(..., description="Method used")
    data_quality_score: int = Field(..., ge=1, le=5, description="Data quality indicator")
    uncertainty_percentage: float = Field(..., description="Uncertainty range %")

    emission_factor_source: str = Field(..., description="Emission factor source")
    is_relevant: bool = Field(True, description="Whether category is relevant")
    exclusion_reason: Optional[str] = Field(None, description="Reason if excluded")


class Scope3Result(BaseModel):
    """Complete Scope 3 emissions result."""

    total_emissions_kgco2e: float = Field(..., description="Total Scope 3 in kgCO2e")
    total_emissions_mtco2e: float = Field(..., description="Total Scope 3 in MTCO2e")

    categories: List[Scope3CategoryResult] = Field(..., description="Results by category")

    upstream_total_mtco2e: float = Field(..., description="Total upstream (Cat 1-8)")
    downstream_total_mtco2e: float = Field(..., description="Total downstream (Cat 9-15)")

    top_categories: List[Dict[str, Any]] = Field(..., description="Top emission categories")
    data_quality_summary: Dict[str, Any] = Field(..., description="Overall data quality")


class AssurancePackage(BaseModel):
    """Third-party assurance documentation."""

    assurance_level: str = Field(..., description="Limited or reasonable")
    standards_applied: List[str] = Field(..., description="Assurance standards (ISAE 3410, etc.)")

    methodology_notes: str = Field(..., description="Detailed calculation methodology")
    data_sources: List[Dict[str, str]] = Field(..., description="Data source documentation")

    evidence_index: List[Dict[str, Any]] = Field(..., description="Evidence documentation index")
    control_activities: List[str] = Field(..., description="Internal control activities")

    ready_for_verification: bool = Field(..., description="Ready for third-party verification")
    completeness_score: float = Field(..., ge=0, le=100, description="Documentation completeness %")


class CARBFilingData(BaseModel):
    """CARB portal submission data."""

    filing_id: str = Field(..., description="Unique filing identifier")
    submission_format: str = Field("CARB_SB253_V1", description="Submission format version")

    company_section: Dict[str, Any] = Field(..., description="Company information section")
    scope1_section: Dict[str, Any] = Field(..., description="Scope 1 data section")
    scope2_section: Dict[str, Any] = Field(..., description="Scope 2 data section")
    scope3_section: Optional[Dict[str, Any]] = Field(None, description="Scope 3 data section")

    methodology_section: Dict[str, Any] = Field(..., description="Methodology documentation")
    verification_section: Dict[str, Any] = Field(..., description="Verification documentation")

    xml_ready: bool = Field(..., description="Ready for XML export")
    pdf_ready: bool = Field(..., description="Ready for PDF generation")


class ProvenanceRecord(BaseModel):
    """Provenance tracking record."""

    operation: str = Field(..., description="Operation performed")
    timestamp: datetime = Field(..., description="Operation timestamp")
    input_hash: str = Field(..., description="SHA-256 of inputs")
    output_hash: str = Field(..., description="SHA-256 of outputs")
    tool_name: str = Field(..., description="Tool/calculator used")
    parameters: Dict[str, Any] = Field(..., description="Operation parameters")


class SB253ReportOutput(BaseModel):
    """
    Complete output model for SB 253 Climate Disclosure Report.

    Includes all emissions calculations, CARB filing data, and assurance package.
    """

    # Identification
    report_id: str = Field(..., description="Unique report identifier")
    company_name: str = Field(..., description="Company name")
    fiscal_year: int = Field(..., description="Reporting fiscal year")
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Emissions Results
    scope1_emissions: Scope1Result = Field(..., description="Scope 1 results")
    scope2_emissions: Scope2Result = Field(..., description="Scope 2 results")
    scope3_emissions: Optional[Scope3Result] = Field(None, description="Scope 3 results")

    # Totals
    total_scope1_mtco2e: float = Field(..., description="Total Scope 1 in MTCO2e")
    total_scope2_location_mtco2e: float = Field(..., description="Total Scope 2 location-based in MTCO2e")
    total_scope2_market_mtco2e: float = Field(..., description="Total Scope 2 market-based in MTCO2e")
    total_scope3_mtco2e: Optional[float] = Field(None, description="Total Scope 3 in MTCO2e")
    total_emissions_mtco2e: float = Field(..., description="Total emissions (Scope 1+2) in MTCO2e")

    # CARB Filing
    carb_filing: CARBFilingData = Field(..., description="CARB portal filing data")

    # Assurance
    assurance_package: AssurancePackage = Field(..., description="Assurance documentation")
    assurance_status: str = Field(..., description="READY, PENDING, or INCOMPLETE")

    # Provenance
    provenance_chain: List[ProvenanceRecord] = Field(..., description="Complete audit trail")
    provenance_hash: str = Field(..., description="SHA-256 hash of complete provenance")

    # Metadata
    gwp_set_used: str = Field(..., description="GWP assessment report used")
    processing_time_ms: float = Field(..., description="Total processing time")
    validation_status: str = Field(..., description="PASS or FAIL")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors if any")


# =============================================================================
# EMISSION FACTOR MODELS
# =============================================================================

class Scope1EmissionFactor(BaseModel):
    """Scope 1 emission factor with provenance."""

    fuel_type: str
    co2_factor: float  # kg CO2 per unit
    ch4_factor: float  # kg CH4 per unit
    n2o_factor: float  # kg N2O per unit
    unit: str
    source: str
    year: int


class EGridFactor(BaseModel):
    """EPA eGRID emission factor."""

    subregion: str
    subregion_name: str
    co2_lb_per_mwh: float
    ch4_lb_per_mwh: float
    n2o_lb_per_mwh: float
    source: str
    year: int


class EEIOFactor(BaseModel):
    """EPA USEEIO emission factor for spend-based Scope 3."""

    naics_code: str
    sector_name: str
    factor_kgco2e_per_usd: float
    source: str
    year: int


# =============================================================================
# SB253 DISCLOSURE AGENT
# =============================================================================

class SB253DisclosureAgent:
    """
    GL-013: California SB 253 Climate Disclosure Agent.

    This agent calculates and reports greenhouse gas emissions
    in compliance with California's Climate Corporate Data
    Accountability Act (SB 253).

    Key Features:
    - Scope 1 direct emissions calculation (stationary, mobile, process, fugitive)
    - Scope 2 indirect emissions (location-based and market-based methods)
    - Scope 3 value chain emissions (all 15 GHG Protocol categories)
    - CARB portal filing format generation
    - Third-party assurance package preparation
    - Complete SHA-256 provenance tracking

    Zero-Hallucination Guarantee:
    - All numeric calculations use deterministic formulas
    - Emission factors sourced from EPA, CARB, GHG Protocol
    - No LLM inference in calculation path
    - Complete audit trail for regulatory compliance

    Attributes:
        VERSION: Agent version
        AGENT_ID: Unique agent identifier
        gwp_values: Global Warming Potential values by GHG
        scope1_factors: Scope 1 emission factors by fuel type
        egrid_factors: EPA eGRID factors by subregion
        eeio_factors: EPA USEEIO factors for Scope 3

    Example:
        >>> agent = SB253DisclosureAgent()
        >>> input_data = SB253ReportInput(
        ...     company_info=CompanyInfo(
        ...         company_name="Test Corp",
        ...         ein="12-3456789",
        ...         total_revenue_usd=2_000_000_000,
        ...         naics_code="331110"
        ...     ),
        ...     fiscal_year=2025,
        ...     scope1_sources=[...],
        ...     scope2_sources=[...]
        ... )
        >>> result = agent.run(input_data)
        >>> assert result.validation_status == "PASS"
    """

    AGENT_ID = "regulatory/sb253_disclosure_v1"
    VERSION = "1.0.0"
    DESCRIPTION = "California SB 253 Climate Disclosure compliance agent"

    # Scope 3 Category Names
    SCOPE3_CATEGORY_NAMES: Dict[int, str] = {
        1: "Purchased Goods and Services",
        2: "Capital Goods",
        3: "Fuel- and Energy-Related Activities",
        4: "Upstream Transportation and Distribution",
        5: "Waste Generated in Operations",
        6: "Business Travel",
        7: "Employee Commuting",
        8: "Upstream Leased Assets",
        9: "Downstream Transportation and Distribution",
        10: "Processing of Sold Products",
        11: "Use of Sold Products",
        12: "End-of-Life Treatment of Sold Products",
        13: "Downstream Leased Assets",
        14: "Franchises",
        15: "Investments",
    }

    # Global Warming Potentials (100-year)
    GWP_VALUES: Dict[str, Dict[str, float]] = {
        "AR4": {"CO2": 1.0, "CH4": 25.0, "N2O": 298.0},
        "AR5": {"CO2": 1.0, "CH4": 28.0, "N2O": 265.0},
        "AR6": {"CO2": 1.0, "CH4": 27.9, "N2O": 273.0},
    }

    # Scope 1 Emission Factors (EPA 2024)
    # Format: fuel_type -> {unit: (CO2, CH4, N2O) in kg per unit}
    SCOPE1_EMISSION_FACTORS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
        "natural_gas": {
            "therms": (5.31, 0.00005, 0.0000097),
            "ccf": (5.31, 0.00005, 0.0000097),
            "MMBtu": (53.06, 0.001, 0.0001),
            "scf": (0.0531, 0.0000005, 0.000000097),
        },
        "diesel": {
            "gallons": (10.21, 0.00041, 0.00041),
            "liters": (2.697, 0.000108, 0.000108),
        },
        "gasoline": {
            "gallons": (8.78, 0.00036, 0.00018),
            "liters": (2.319, 0.000095, 0.000048),
        },
        "propane": {
            "gallons": (5.72, 0.00003, 0.00059),
            "liters": (1.511, 0.000008, 0.000156),
        },
        "fuel_oil_2": {
            "gallons": (10.16, 0.00041, 0.00008),
            "liters": (2.684, 0.000108, 0.000021),
        },
        "coal": {
            "tons": (2406.0, 0.011, 0.016),
            "kg": (2.406, 0.000011, 0.000016),
        },
        "lpg": {
            "gallons": (5.68, 0.00003, 0.00059),
            "liters": (1.501, 0.000008, 0.000156),
        },
    }

    # Refrigerant GWP Values (AR5)
    REFRIGERANT_GWP: Dict[str, float] = {
        "R-134a": 1430.0,
        "R-410A": 2088.0,
        "R-22": 1810.0,
        "R-404A": 3922.0,
        "R-407C": 1774.0,
        "R-507A": 3985.0,
        "R-32": 675.0,
        "CO2": 1.0,
        "ammonia": 0.0,
    }

    # EPA eGRID Subregional Factors (2022 data, lb/MWh)
    # California subregions emphasized for SB 253
    EGRID_FACTORS: Dict[str, EGridFactor] = {
        "CAMX": EGridFactor(
            subregion="CAMX",
            subregion_name="WECC California",
            co2_lb_per_mwh=472.0,
            ch4_lb_per_mwh=0.035,
            n2o_lb_per_mwh=0.006,
            source="EPA eGRID 2022",
            year=2022,
        ),
        "NWPP": EGridFactor(
            subregion="NWPP",
            subregion_name="WECC Northwest",
            co2_lb_per_mwh=627.0,
            ch4_lb_per_mwh=0.054,
            n2o_lb_per_mwh=0.009,
            source="EPA eGRID 2022",
            year=2022,
        ),
        "AZNM": EGridFactor(
            subregion="AZNM",
            subregion_name="WECC Southwest",
            co2_lb_per_mwh=789.0,
            ch4_lb_per_mwh=0.058,
            n2o_lb_per_mwh=0.010,
            source="EPA eGRID 2022",
            year=2022,
        ),
        "ERCT": EGridFactor(
            subregion="ERCT",
            subregion_name="ERCOT Texas",
            co2_lb_per_mwh=870.0,
            ch4_lb_per_mwh=0.075,
            n2o_lb_per_mwh=0.010,
            source="EPA eGRID 2022",
            year=2022,
        ),
        "RFCW": EGridFactor(
            subregion="RFCW",
            subregion_name="RFC West",
            co2_lb_per_mwh=992.0,
            ch4_lb_per_mwh=0.091,
            n2o_lb_per_mwh=0.015,
            source="EPA eGRID 2022",
            year=2022,
        ),
        "RFCE": EGridFactor(
            subregion="RFCE",
            subregion_name="RFC East",
            co2_lb_per_mwh=653.0,
            ch4_lb_per_mwh=0.047,
            n2o_lb_per_mwh=0.008,
            source="EPA eGRID 2022",
            year=2022,
        ),
        "SRMW": EGridFactor(
            subregion="SRMW",
            subregion_name="SERC Midwest",
            co2_lb_per_mwh=1356.0,
            ch4_lb_per_mwh=0.128,
            n2o_lb_per_mwh=0.020,
            source="EPA eGRID 2022",
            year=2022,
        ),
        "SRSO": EGridFactor(
            subregion="SRSO",
            subregion_name="SERC South",
            co2_lb_per_mwh=822.0,
            ch4_lb_per_mwh=0.065,
            n2o_lb_per_mwh=0.010,
            source="EPA eGRID 2022",
            year=2022,
        ),
        "NEWE": EGridFactor(
            subregion="NEWE",
            subregion_name="NPCC New England",
            co2_lb_per_mwh=482.0,
            ch4_lb_per_mwh=0.048,
            n2o_lb_per_mwh=0.006,
            source="EPA eGRID 2022",
            year=2022,
        ),
        "NYUP": EGridFactor(
            subregion="NYUP",
            subregion_name="NPCC Upstate NY",
            co2_lb_per_mwh=299.0,
            ch4_lb_per_mwh=0.019,
            n2o_lb_per_mwh=0.003,
            source="EPA eGRID 2022",
            year=2022,
        ),
        "NYCW": EGridFactor(
            subregion="NYCW",
            subregion_name="NPCC NYC/Westchester",
            co2_lb_per_mwh=562.0,
            ch4_lb_per_mwh=0.039,
            n2o_lb_per_mwh=0.006,
            source="EPA eGRID 2022",
            year=2022,
        ),
        "MROW": EGridFactor(
            subregion="MROW",
            subregion_name="MRO West",
            co2_lb_per_mwh=1014.0,
            ch4_lb_per_mwh=0.092,
            n2o_lb_per_mwh=0.014,
            source="EPA eGRID 2022",
            year=2022,
        ),
        "RMPA": EGridFactor(
            subregion="RMPA",
            subregion_name="WECC Rockies",
            co2_lb_per_mwh=1022.0,
            ch4_lb_per_mwh=0.085,
            n2o_lb_per_mwh=0.014,
            source="EPA eGRID 2022",
            year=2022,
        ),
    }

    # EPA USEEIO Factors by NAICS (kgCO2e per USD spend)
    EEIO_FACTORS: Dict[str, EEIOFactor] = {
        "331110": EEIOFactor(
            naics_code="331110",
            sector_name="Iron and Steel Mills",
            factor_kgco2e_per_usd=0.95,
            source="EPA USEEIO v2.0",
            year=2024,
        ),
        "336111": EEIOFactor(
            naics_code="336111",
            sector_name="Automobile Manufacturing",
            factor_kgco2e_per_usd=0.42,
            source="EPA USEEIO v2.0",
            year=2024,
        ),
        "325110": EEIOFactor(
            naics_code="325110",
            sector_name="Petrochemical Manufacturing",
            factor_kgco2e_per_usd=1.15,
            source="EPA USEEIO v2.0",
            year=2024,
        ),
        "311000": EEIOFactor(
            naics_code="311000",
            sector_name="Food Manufacturing",
            factor_kgco2e_per_usd=0.58,
            source="EPA USEEIO v2.0",
            year=2024,
        ),
        "334111": EEIOFactor(
            naics_code="334111",
            sector_name="Electronic Computer Manufacturing",
            factor_kgco2e_per_usd=0.35,
            source="EPA USEEIO v2.0",
            year=2024,
        ),
        "541500": EEIOFactor(
            naics_code="541500",
            sector_name="Computer Systems Design",
            factor_kgco2e_per_usd=0.12,
            source="EPA USEEIO v2.0",
            year=2024,
        ),
        "481000": EEIOFactor(
            naics_code="481000",
            sector_name="Air Transportation",
            factor_kgco2e_per_usd=0.75,
            source="EPA USEEIO v2.0",
            year=2024,
        ),
        "484000": EEIOFactor(
            naics_code="484000",
            sector_name="Truck Transportation",
            factor_kgco2e_per_usd=0.52,
            source="EPA USEEIO v2.0",
            year=2024,
        ),
        "DEFAULT": EEIOFactor(
            naics_code="DEFAULT",
            sector_name="Average All Sectors",
            factor_kgco2e_per_usd=0.40,
            source="EPA USEEIO v2.0",
            year=2024,
        ),
    }

    # Business Travel Factors (kgCO2e per passenger-km)
    TRAVEL_FACTORS: Dict[str, float] = {
        "air_short_haul": 0.255,   # <1500 km
        "air_medium_haul": 0.156,  # 1500-4000 km
        "air_long_haul": 0.195,    # >4000 km
        "rail": 0.041,
        "car_rental": 0.171,
        "taxi": 0.203,
        "bus": 0.089,
    }

    # Employee Commuting Factors (kgCO2e per passenger-km)
    COMMUTING_FACTORS: Dict[str, float] = {
        "car_alone": 0.171,
        "car_carpool": 0.086,
        "public_transit": 0.089,
        "rail": 0.041,
        "bus": 0.089,
        "bicycle": 0.0,
        "walking": 0.0,
        "remote": 0.002,  # Home office electricity
    }

    # Waste Treatment Factors (kgCO2e per kg waste)
    WASTE_FACTORS: Dict[str, float] = {
        "landfill_mixed": 0.586,
        "landfill_organic": 0.700,
        "incineration": 0.021,
        "recycling": -0.500,  # Credit for avoided primary production
        "composting": 0.010,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SB253 Disclosure Agent.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []

        logger.info(f"SB253DisclosureAgent initialized (version {self.VERSION})")

    def run(self, input_data: SB253ReportInput) -> SB253ReportOutput:
        """
        Execute the SB 253 climate disclosure calculation.

        This method performs complete GHG inventory calculation:
        - Scope 1: Direct emissions from owned/controlled sources
        - Scope 2: Indirect emissions from purchased electricity
        - Scope 3: Value chain emissions (if enabled)

        All calculations follow zero-hallucination principles:
        - Deterministic formulas from GHG Protocol
        - EPA/CARB emission factors
        - Complete provenance tracking

        Args:
            input_data: Validated SB 253 input data

        Returns:
            Complete disclosure report with CARB filing data

        Raises:
            ValueError: If input validation fails
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []

        logger.info(
            f"Starting SB 253 disclosure calculation for {input_data.company_info.company_name}, "
            f"FY{input_data.fiscal_year}"
        )

        try:
            # Step 1: Validate applicability
            self._validate_sb253_applicability(input_data)
            self._track_provenance(
                "applicability_check",
                {"company": input_data.company_info.company_name},
                {"applicable": True},
                "sb253_validator"
            )

            # Step 2: Calculate Scope 1 emissions
            scope1_result = self._calculate_scope1(
                input_data.scope1_sources,
                input_data.gwp_set
            )
            self._track_provenance(
                "scope1_calculation",
                {"sources_count": len(input_data.scope1_sources)},
                {"total_mtco2e": scope1_result.total_emissions.co2e_mtco2e},
                "scope1_calculator"
            )

            # Step 3: Calculate Scope 2 emissions (dual reporting)
            scope2_result = self._calculate_scope2(
                input_data.scope2_sources,
                input_data.gwp_set
            )
            self._track_provenance(
                "scope2_calculation",
                {"sources_count": len(input_data.scope2_sources)},
                {
                    "location_based_mtco2e": scope2_result.location_based.co2e_mtco2e,
                    "market_based_mtco2e": scope2_result.market_based.co2e_mtco2e
                },
                "scope2_calculator"
            )

            # Step 4: Calculate Scope 3 emissions (if required)
            scope3_result = None
            if input_data.include_scope3 and input_data.scope3_data:
                scope3_result = self._calculate_scope3(
                    input_data.scope3_data,
                    input_data.company_info
                )
                self._track_provenance(
                    "scope3_calculation",
                    {"include_scope3": True},
                    {"total_mtco2e": scope3_result.total_emissions_mtco2e},
                    "scope3_calculator"
                )

            # Step 5: Calculate totals
            total_scope1 = scope1_result.total_emissions.co2e_mtco2e
            total_scope2_location = scope2_result.location_based.co2e_mtco2e
            total_scope2_market = scope2_result.market_based.co2e_mtco2e
            total_scope3 = scope3_result.total_emissions_mtco2e if scope3_result else None

            # Total emissions = Scope 1 + Scope 2 (location-based)
            total_emissions = total_scope1 + total_scope2_location
            if total_scope3:
                total_emissions += total_scope3

            # Step 6: Generate CARB filing data
            carb_filing = self._generate_carb_filing(
                input_data,
                scope1_result,
                scope2_result,
                scope3_result
            )

            # Step 7: Generate assurance package
            assurance_package = self._generate_assurance_package(
                input_data,
                scope1_result,
                scope2_result,
                scope3_result
            )

            # Step 8: Calculate final provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Step 9: Determine validation status
            validation_status = "PASS" if not self._validation_errors else "FAIL"
            assurance_status = "READY" if assurance_package.ready_for_verification else "PENDING"

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Generate report ID
            report_id = f"SB253-{input_data.company_info.ein}-FY{input_data.fiscal_year}"

            output = SB253ReportOutput(
                report_id=report_id,
                company_name=input_data.company_info.company_name,
                fiscal_year=input_data.fiscal_year,

                scope1_emissions=scope1_result,
                scope2_emissions=scope2_result,
                scope3_emissions=scope3_result,

                total_scope1_mtco2e=round(total_scope1, 4),
                total_scope2_location_mtco2e=round(total_scope2_location, 4),
                total_scope2_market_mtco2e=round(total_scope2_market, 4),
                total_scope3_mtco2e=round(total_scope3, 4) if total_scope3 else None,
                total_emissions_mtco2e=round(total_emissions, 4),

                carb_filing=carb_filing,
                assurance_package=assurance_package,
                assurance_status=assurance_status,

                provenance_chain=[
                    ProvenanceRecord(
                        operation=step["operation"],
                        timestamp=step["timestamp"],
                        input_hash=step["input_hash"],
                        output_hash=step["output_hash"],
                        tool_name=step["tool_name"],
                        parameters=step.get("parameters", {})
                    )
                    for step in self._provenance_steps
                ],
                provenance_hash=provenance_hash,

                gwp_set_used=input_data.gwp_set.value,
                processing_time_ms=round(processing_time, 2),
                validation_status=validation_status,
                validation_errors=self._validation_errors,
            )

            logger.info(
                f"SB 253 calculation complete: {total_emissions:.2f} MTCO2e total, "
                f"Scope 1: {total_scope1:.2f}, Scope 2: {total_scope2_location:.2f} "
                f"(duration: {processing_time:.2f}ms)"
            )

            return output

        except Exception as e:
            logger.error(f"SB 253 calculation failed: {str(e)}", exc_info=True)
            raise

    # =========================================================================
    # SCOPE 1 CALCULATIONS
    # =========================================================================

    def _calculate_scope1(
        self,
        sources: List[Scope1Source],
        gwp_set: GWPSet
    ) -> Scope1Result:
        """
        Calculate Scope 1 direct emissions.

        ZERO-HALLUCINATION: Uses deterministic formula:
        CO2e = (CO2 * GWP_CO2) + (CH4 * GWP_CH4) + (N2O * GWP_N2O)

        Categories:
        - Stationary combustion (boilers, furnaces, heaters)
        - Mobile combustion (fleet vehicles)
        - Process emissions (chemical/industrial processes)
        - Fugitive emissions (refrigerants, methane leaks)

        Args:
            sources: List of Scope 1 emission sources
            gwp_set: Global Warming Potential set to use

        Returns:
            Scope1Result with emissions breakdown
        """
        logger.info(f"Calculating Scope 1 emissions for {len(sources)} sources")

        gwp = self.GWP_VALUES[gwp_set.value]

        # Initialize accumulators
        stationary = {"co2": 0.0, "ch4": 0.0, "n2o": 0.0}
        mobile = {"co2": 0.0, "ch4": 0.0, "n2o": 0.0}
        process = {"co2": 0.0, "ch4": 0.0, "n2o": 0.0}
        fugitive = {"co2": 0.0, "ch4": 0.0, "n2o": 0.0}

        emissions_by_facility: Dict[str, float] = {}
        emissions_by_fuel: Dict[str, float] = {}
        factors_used: List[Dict[str, Any]] = []

        for source in sources:
            # Get emission factors
            fuel_factors = self.SCOPE1_EMISSION_FACTORS.get(source.fuel_type.value, {})
            unit_factors = fuel_factors.get(source.unit.value)

            if not unit_factors:
                # Try to convert units
                unit_factors = self._get_converted_factors(source.fuel_type.value, source.unit.value)

            if unit_factors:
                co2_factor, ch4_factor, n2o_factor = unit_factors

                # ZERO-HALLUCINATION CALCULATION
                # emissions = quantity * emission_factor
                co2_kg = source.quantity * co2_factor
                ch4_kg = source.quantity * ch4_factor
                n2o_kg = source.quantity * n2o_factor

                # Calculate CO2e
                co2e_kg = (co2_kg * gwp["CO2"]) + (ch4_kg * gwp["CH4"]) + (n2o_kg * gwp["N2O"])

                # Categorize by source type
                if source.source_category == SourceCategory.STATIONARY_COMBUSTION:
                    stationary["co2"] += co2_kg
                    stationary["ch4"] += ch4_kg
                    stationary["n2o"] += n2o_kg
                elif source.source_category == SourceCategory.MOBILE_COMBUSTION:
                    mobile["co2"] += co2_kg
                    mobile["ch4"] += ch4_kg
                    mobile["n2o"] += n2o_kg
                elif source.source_category == SourceCategory.PROCESS_EMISSIONS:
                    process["co2"] += co2_kg
                    process["ch4"] += ch4_kg
                    process["n2o"] += n2o_kg

                # Track by facility and fuel
                emissions_by_facility[source.facility_id] = (
                    emissions_by_facility.get(source.facility_id, 0) + co2e_kg
                )
                emissions_by_fuel[source.fuel_type.value] = (
                    emissions_by_fuel.get(source.fuel_type.value, 0) + co2e_kg
                )

                factors_used.append({
                    "fuel_type": source.fuel_type.value,
                    "co2_factor": co2_factor,
                    "ch4_factor": ch4_factor,
                    "n2o_factor": n2o_factor,
                    "unit": source.unit.value,
                    "source": "EPA GHG Emission Factors Hub 2024",
                })

            # Handle fugitive emissions (refrigerants)
            if (source.source_category == SourceCategory.FUGITIVE_EMISSIONS and
                source.refrigerant_type and source.refrigerant_loss_kg):

                refrigerant_gwp = self.REFRIGERANT_GWP.get(source.refrigerant_type.value, 0)

                # ZERO-HALLUCINATION: CO2e = refrigerant_loss * GWP
                fugitive_co2e = source.refrigerant_loss_kg * refrigerant_gwp
                fugitive["co2"] += fugitive_co2e  # Report as CO2e

                emissions_by_facility[source.facility_id] = (
                    emissions_by_facility.get(source.facility_id, 0) + fugitive_co2e
                )

                factors_used.append({
                    "refrigerant": source.refrigerant_type.value,
                    "gwp": refrigerant_gwp,
                    "source": f"IPCC {gwp_set.value}",
                })

        # Create emission breakdowns
        def create_breakdown(data: Dict[str, float]) -> EmissionBreakdown:
            co2e = (data["co2"] * gwp["CO2"]) + (data["ch4"] * gwp["CH4"]) + (data["n2o"] * gwp["N2O"])
            return EmissionBreakdown(
                co2_kg=round(data["co2"], 4),
                ch4_kg=round(data["ch4"], 6),
                n2o_kg=round(data["n2o"], 6),
                co2e_kg=round(co2e, 4),
                co2e_mtco2e=round(co2e / 1000, 4),
            )

        # Calculate totals
        total_data = {
            "co2": stationary["co2"] + mobile["co2"] + process["co2"] + fugitive["co2"],
            "ch4": stationary["ch4"] + mobile["ch4"] + process["ch4"] + fugitive["ch4"],
            "n2o": stationary["n2o"] + mobile["n2o"] + process["n2o"] + fugitive["n2o"],
        }

        return Scope1Result(
            total_emissions=create_breakdown(total_data),
            stationary_combustion=create_breakdown(stationary),
            mobile_combustion=create_breakdown(mobile),
            process_emissions=create_breakdown(process),
            fugitive_emissions=create_breakdown(fugitive),
            emissions_by_facility={k: round(v, 2) for k, v in emissions_by_facility.items()},
            emissions_by_fuel={k: round(v, 2) for k, v in emissions_by_fuel.items()},
            emission_factors_used=factors_used,
        )

    def _get_converted_factors(
        self,
        fuel_type: str,
        unit: str
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get emission factors with unit conversion.

        Converts between common units (e.g., gallons to liters).
        """
        fuel_factors = self.SCOPE1_EMISSION_FACTORS.get(fuel_type, {})

        # Unit conversion map
        conversions = {
            ("gallons", "liters"): 3.78541,
            ("liters", "gallons"): 0.264172,
            ("therms", "MMBtu"): 0.1,
            ("MMBtu", "therms"): 10.0,
            ("tons", "kg"): 907.185,
            ("kg", "tons"): 0.001102,
        }

        for base_unit, factors in fuel_factors.items():
            conversion_key = (base_unit, unit)
            if conversion_key in conversions:
                conv_factor = conversions[conversion_key]
                return (
                    factors[0] / conv_factor,
                    factors[1] / conv_factor,
                    factors[2] / conv_factor,
                )

        return None

    # =========================================================================
    # SCOPE 2 CALCULATIONS
    # =========================================================================

    def _calculate_scope2(
        self,
        sources: List[Scope2Source],
        gwp_set: GWPSet
    ) -> Scope2Result:
        """
        Calculate Scope 2 indirect emissions.

        Implements dual reporting as required by GHG Protocol and SB 253:
        - Location-based: Uses EPA eGRID subregional factors
        - Market-based: Uses contractual instruments (PPAs, RECs, utility factors)

        ZERO-HALLUCINATION: Uses deterministic formula:
        CO2e = kWh * (emission_factor_lb/MWh / 2.20462 / 1000)

        Args:
            sources: List of Scope 2 electricity sources
            gwp_set: Global Warming Potential set to use

        Returns:
            Scope2Result with location and market-based emissions
        """
        logger.info(f"Calculating Scope 2 emissions for {len(sources)} sources")

        gwp = self.GWP_VALUES[gwp_set.value]

        # Initialize accumulators
        location_total = {"co2": 0.0, "ch4": 0.0, "n2o": 0.0}
        market_total = {"co2": 0.0, "ch4": 0.0, "n2o": 0.0}

        emissions_by_facility: Dict[str, Dict[str, float]] = {}
        emissions_by_grid: Dict[str, float] = {}

        total_kwh = 0.0
        renewable_kwh = 0.0

        egrid_factors_used: List[Dict[str, Any]] = []

        for source in sources:
            total_kwh += source.kwh
            renewable_kwh += source.kwh * (source.renewable_percentage / 100)

            # Get eGRID factors
            egrid = self.EGRID_FACTORS.get(source.egrid_subregion)
            if not egrid:
                egrid = self.EGRID_FACTORS.get("CAMX")  # Default to California
                logger.warning(f"Using CAMX factors for unknown subregion: {source.egrid_subregion}")

            # Convert lb/MWh to kg/kWh: divide by 2.20462 (lb to kg) and 1000 (MWh to kWh)
            lb_to_kg = 0.453592
            mwh_to_kwh = 1000.0

            co2_kg_per_kwh = egrid.co2_lb_per_mwh * lb_to_kg / mwh_to_kwh
            ch4_kg_per_kwh = egrid.ch4_lb_per_mwh * lb_to_kg / mwh_to_kwh
            n2o_kg_per_kwh = egrid.n2o_lb_per_mwh * lb_to_kg / mwh_to_kwh

            # LOCATION-BASED CALCULATION (ZERO-HALLUCINATION)
            # emissions = kWh * emission_factor
            loc_co2 = source.kwh * co2_kg_per_kwh
            loc_ch4 = source.kwh * ch4_kg_per_kwh
            loc_n2o = source.kwh * n2o_kg_per_kwh

            location_total["co2"] += loc_co2
            location_total["ch4"] += loc_ch4
            location_total["n2o"] += loc_n2o

            # MARKET-BASED CALCULATION
            # Adjust for RECs, PPAs, and utility-specific factors
            market_kwh = source.kwh

            # Subtract RECs (zero emissions for REC-covered electricity)
            if source.has_recs and source.rec_mwh:
                rec_kwh = source.rec_mwh * 1000
                market_kwh = max(0, market_kwh - rec_kwh)

            # Apply renewable percentage reduction
            market_kwh *= (1 - source.renewable_percentage / 100)

            # Use utility-specific factor if available
            if source.utility_emission_factor is not None:
                mkt_co2 = market_kwh * source.utility_emission_factor
                mkt_ch4 = 0.0  # Utility factors usually only report CO2
                mkt_n2o = 0.0
            elif source.has_ppa and source.ppa_emissions_factor is not None:
                mkt_co2 = market_kwh * source.ppa_emissions_factor
                mkt_ch4 = 0.0
                mkt_n2o = 0.0
            else:
                # Fall back to location-based factors
                mkt_co2 = market_kwh * co2_kg_per_kwh
                mkt_ch4 = market_kwh * ch4_kg_per_kwh
                mkt_n2o = market_kwh * n2o_kg_per_kwh

            market_total["co2"] += mkt_co2
            market_total["ch4"] += mkt_ch4
            market_total["n2o"] += mkt_n2o

            # Track by facility
            loc_co2e = (loc_co2 * gwp["CO2"]) + (loc_ch4 * gwp["CH4"]) + (loc_n2o * gwp["N2O"])
            mkt_co2e = (mkt_co2 * gwp["CO2"]) + (mkt_ch4 * gwp["CH4"]) + (mkt_n2o * gwp["N2O"])

            emissions_by_facility[source.facility_id] = {
                "location_based": round(loc_co2e, 2),
                "market_based": round(mkt_co2e, 2),
            }

            # Track by grid
            emissions_by_grid[source.egrid_subregion] = (
                emissions_by_grid.get(source.egrid_subregion, 0) + loc_co2e
            )

            egrid_factors_used.append({
                "subregion": egrid.subregion,
                "subregion_name": egrid.subregion_name,
                "co2_lb_per_mwh": egrid.co2_lb_per_mwh,
                "ch4_lb_per_mwh": egrid.ch4_lb_per_mwh,
                "n2o_lb_per_mwh": egrid.n2o_lb_per_mwh,
                "source": egrid.source,
            })

        # Create emission breakdowns
        def create_breakdown(data: Dict[str, float]) -> EmissionBreakdown:
            co2e = (data["co2"] * gwp["CO2"]) + (data["ch4"] * gwp["CH4"]) + (data["n2o"] * gwp["N2O"])
            return EmissionBreakdown(
                co2_kg=round(data["co2"], 4),
                ch4_kg=round(data["ch4"], 6),
                n2o_kg=round(data["n2o"], 6),
                co2e_kg=round(co2e, 4),
                co2e_mtco2e=round(co2e / 1000, 4),
            )

        renewable_pct = (renewable_kwh / total_kwh * 100) if total_kwh > 0 else 0

        return Scope2Result(
            location_based=create_breakdown(location_total),
            market_based=create_breakdown(market_total),
            emissions_by_facility=emissions_by_facility,
            emissions_by_grid={k: round(v, 2) for k, v in emissions_by_grid.items()},
            total_electricity_mwh=round(total_kwh / 1000, 2),
            renewable_mwh=round(renewable_kwh / 1000, 2),
            renewable_percentage=round(renewable_pct, 2),
            egrid_factors_used=egrid_factors_used,
        )

    # =========================================================================
    # SCOPE 3 CALCULATIONS - ALL 15 CATEGORIES
    # =========================================================================

    def _calculate_scope3(
        self,
        scope3_data: Scope3Data,
        company_info: CompanyInfo
    ) -> Scope3Result:
        """
        Calculate Scope 3 value chain emissions for all 15 categories.

        Categories:
        1. Purchased goods and services
        2. Capital goods
        3. Fuel- and energy-related activities
        4. Upstream transportation and distribution
        5. Waste generated in operations
        6. Business travel
        7. Employee commuting
        8. Upstream leased assets
        9. Downstream transportation and distribution
        10. Processing of sold products
        11. Use of sold products
        12. End-of-life treatment of sold products
        13. Downstream leased assets
        14. Franchises
        15. Investments

        Args:
            scope3_data: Complete Scope 3 input data
            company_info: Company information

        Returns:
            Scope3Result with all category emissions
        """
        logger.info("Calculating Scope 3 emissions for all 15 categories")

        categories_results: List[Scope3CategoryResult] = []

        # Category mapping
        category_map = {
            1: scope3_data.purchased_goods_services,
            2: scope3_data.capital_goods,
            3: scope3_data.fuel_energy_activities,
            4: scope3_data.upstream_transportation,
            5: scope3_data.waste_generated,
            6: scope3_data.business_travel,
            7: scope3_data.employee_commuting,
            8: scope3_data.upstream_leased_assets,
            9: scope3_data.downstream_transportation,
            10: scope3_data.processing_sold_products,
            11: scope3_data.use_of_sold_products,
            12: scope3_data.end_of_life_treatment,
            13: scope3_data.downstream_leased_assets,
            14: scope3_data.franchises,
            15: scope3_data.investments,
        }

        for cat_num in range(1, 16):
            cat_data = category_map.get(cat_num)
            cat_name = self.SCOPE3_CATEGORY_NAMES[cat_num]

            if cat_data:
                result = self._calculate_scope3_category(cat_num, cat_data)
            else:
                # Category not provided - mark as not calculated
                result = Scope3CategoryResult(
                    category_number=cat_num,
                    category_name=cat_name,
                    co2e_kg=0.0,
                    co2e_mtco2e=0.0,
                    calculation_method="not_calculated",
                    data_quality_score=5,
                    uncertainty_percentage=100.0,
                    emission_factor_source="N/A",
                    is_relevant=True,
                    exclusion_reason="No data provided",
                )

            categories_results.append(result)

        # Calculate totals
        total_kg = sum(r.co2e_kg for r in categories_results)
        total_mt = total_kg / 1000

        # Upstream (1-8) vs Downstream (9-15)
        upstream_mt = sum(
            r.co2e_mtco2e for r in categories_results
            if r.category_number <= 8
        )
        downstream_mt = sum(
            r.co2e_mtco2e for r in categories_results
            if r.category_number > 8
        )

        # Top categories
        sorted_cats = sorted(categories_results, key=lambda x: x.co2e_kg, reverse=True)
        top_categories = [
            {
                "category": c.category_number,
                "name": c.category_name,
                "emissions_mtco2e": c.co2e_mtco2e,
                "percentage": round(c.co2e_kg / total_kg * 100, 1) if total_kg > 0 else 0,
            }
            for c in sorted_cats[:5]
        ]

        # Data quality summary
        dq_scores = [r.data_quality_score for r in categories_results if r.co2e_kg > 0]
        avg_dq = sum(dq_scores) / len(dq_scores) if dq_scores else 5

        data_quality_summary = {
            "average_score": round(avg_dq, 1),
            "categories_with_data": len(dq_scores),
            "categories_missing_data": 15 - len(dq_scores),
            "high_quality_count": sum(1 for s in dq_scores if s <= 2),
            "low_quality_count": sum(1 for s in dq_scores if s >= 4),
        }

        return Scope3Result(
            total_emissions_kgco2e=round(total_kg, 2),
            total_emissions_mtco2e=round(total_mt, 4),
            categories=categories_results,
            upstream_total_mtco2e=round(upstream_mt, 4),
            downstream_total_mtco2e=round(downstream_mt, 4),
            top_categories=top_categories,
            data_quality_summary=data_quality_summary,
        )

    def _calculate_scope3_category(
        self,
        category_number: int,
        data: Scope3CategoryData
    ) -> Scope3CategoryResult:
        """
        Calculate emissions for a specific Scope 3 category.

        ZERO-HALLUCINATION: Uses deterministic formulas by category:
        - Spend-based: emissions = spend * EEIO_factor
        - Activity-based: emissions = activity * factor
        - Supplier-specific: uses provided emissions directly

        Args:
            category_number: Scope 3 category (1-15)
            data: Category-specific input data

        Returns:
            Scope3CategoryResult with calculated emissions
        """
        cat_name = self.SCOPE3_CATEGORY_NAMES[category_number]
        co2e_kg = 0.0
        factor_source = "EPA USEEIO v2.0"

        # Category-specific calculations
        if data.calculation_method == CalculationMethod.SPEND_BASED:
            # SPEND-BASED CALCULATION
            # CO2e = spend_USD * EEIO_factor
            if data.spend_usd and data.spend_usd > 0:
                naics = data.naics_code or "DEFAULT"
                eeio = self.EEIO_FACTORS.get(naics, self.EEIO_FACTORS["DEFAULT"])
                co2e_kg = data.spend_usd * eeio.factor_kgco2e_per_usd
                factor_source = eeio.source

        elif data.calculation_method == CalculationMethod.SUPPLIER_SPECIFIC:
            # Use supplier-provided emissions
            if data.supplier_emissions_kgco2e:
                co2e_kg = data.supplier_emissions_kgco2e
                factor_source = "Supplier-specific data"

        elif data.calculation_method in [CalculationMethod.AVERAGE_DATA, CalculationMethod.HYBRID]:
            # Category-specific activity calculations
            activity = data.activity_data or {}

            if category_number == 6:  # Business Travel
                co2e_kg = self._calculate_business_travel(activity)
                factor_source = "DEFRA/EPA Travel Factors 2024"

            elif category_number == 7:  # Employee Commuting
                co2e_kg = self._calculate_employee_commuting(activity)
                factor_source = "EPA Commuting Factors 2024"

            elif category_number == 5:  # Waste
                co2e_kg = self._calculate_waste_emissions(activity)
                factor_source = "EPA Waste Factors 2024"

            elif category_number in [4, 9]:  # Transportation
                co2e_kg = self._calculate_transportation(activity)
                factor_source = "GLEC Framework 2024"

            else:
                # Fall back to spend-based if available
                if data.spend_usd and data.spend_usd > 0:
                    eeio = self.EEIO_FACTORS.get("DEFAULT")
                    co2e_kg = data.spend_usd * eeio.factor_kgco2e_per_usd

        # Calculate uncertainty based on data quality
        uncertainty_map = {
            1: 10.0,
            2: 25.0,
            3: 50.0,
            4: 75.0,
            5: 100.0,
        }
        uncertainty = uncertainty_map.get(data.data_quality_score.value, 50.0)

        return Scope3CategoryResult(
            category_number=category_number,
            category_name=cat_name,
            co2e_kg=round(co2e_kg, 2),
            co2e_mtco2e=round(co2e_kg / 1000, 4),
            calculation_method=data.calculation_method.value,
            data_quality_score=data.data_quality_score.value,
            uncertainty_percentage=uncertainty,
            emission_factor_source=factor_source,
            is_relevant=True,
        )

    def _calculate_business_travel(self, activity: Dict[str, Any]) -> float:
        """
        Calculate Category 6: Business Travel emissions.

        ZERO-HALLUCINATION:
        CO2e = SUM(distance_km * passenger_factor)
        """
        co2e = 0.0

        # Air travel by haul type
        short_haul_km = activity.get("short_haul_miles", 0) * 1.60934
        medium_haul_km = activity.get("medium_haul_miles", 0) * 1.60934
        long_haul_km = activity.get("long_haul_miles", 0) * 1.60934

        co2e += short_haul_km * self.TRAVEL_FACTORS["air_short_haul"]
        co2e += medium_haul_km * self.TRAVEL_FACTORS["air_medium_haul"]
        co2e += long_haul_km * self.TRAVEL_FACTORS["air_long_haul"]

        # Rail travel
        rail_km = activity.get("rail_km", 0)
        co2e += rail_km * self.TRAVEL_FACTORS["rail"]

        # Car rental
        car_km = activity.get("car_rental_km", 0)
        co2e += car_km * self.TRAVEL_FACTORS["car_rental"]

        return co2e

    def _calculate_employee_commuting(self, activity: Dict[str, Any]) -> float:
        """
        Calculate Category 7: Employee Commuting emissions.

        ZERO-HALLUCINATION:
        CO2e = employees * avg_commute_km * working_days * mode_factor
        """
        employees = activity.get("employees", 0)
        avg_commute_miles = activity.get("avg_commute_miles", 0)
        avg_commute_km = avg_commute_miles * 1.60934
        working_days = activity.get("working_days", 220)

        # Modal split (default assumptions if not provided)
        car_pct = activity.get("car_percentage", 76) / 100
        transit_pct = activity.get("transit_percentage", 5) / 100
        remote_pct = activity.get("remote_percentage", 15) / 100
        other_pct = 1 - car_pct - transit_pct - remote_pct

        total_km = employees * avg_commute_km * 2 * working_days  # Round trip

        co2e = 0.0
        co2e += total_km * car_pct * self.COMMUTING_FACTORS["car_alone"]
        co2e += total_km * transit_pct * self.COMMUTING_FACTORS["public_transit"]
        co2e += employees * working_days * remote_pct * self.COMMUTING_FACTORS["remote"]

        return co2e

    def _calculate_waste_emissions(self, activity: Dict[str, Any]) -> float:
        """
        Calculate Category 5: Waste Generated emissions.

        ZERO-HALLUCINATION:
        CO2e = waste_kg * treatment_factor
        """
        co2e = 0.0

        for waste_type, factor in self.WASTE_FACTORS.items():
            waste_kg = activity.get(f"{waste_type}_kg", 0)
            co2e += waste_kg * factor

        # Generic waste if no breakdown
        total_waste_kg = activity.get("total_waste_kg", 0)
        if total_waste_kg > 0 and co2e == 0:
            co2e = total_waste_kg * self.WASTE_FACTORS["landfill_mixed"]

        return co2e

    def _calculate_transportation(self, activity: Dict[str, Any]) -> float:
        """
        Calculate Category 4/9: Transportation emissions.

        ZERO-HALLUCINATION:
        CO2e = tonne_km * transport_factor
        """
        co2e = 0.0

        # Transport factors (kgCO2e per tonne-km)
        transport_factors = {
            "road": 0.089,
            "rail": 0.028,
            "sea": 0.016,
            "air": 0.602,
        }

        for mode, factor in transport_factors.items():
            tonne_km = activity.get(f"{mode}_tonne_km", 0)
            co2e += tonne_km * factor

        return co2e

    # =========================================================================
    # CARB FILING GENERATION
    # =========================================================================

    def _generate_carb_filing(
        self,
        input_data: SB253ReportInput,
        scope1: Scope1Result,
        scope2: Scope2Result,
        scope3: Optional[Scope3Result]
    ) -> CARBFilingData:
        """
        Generate CARB portal submission data.

        Creates structured data conforming to CARB's SB 253 filing requirements.
        """
        filing_id = f"CARB-{input_data.company_info.ein}-{input_data.fiscal_year}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        company_section = {
            "legal_name": input_data.company_info.company_name,
            "ein": input_data.company_info.ein,
            "naics_code": input_data.company_info.naics_code,
            "total_revenue_usd": input_data.company_info.total_revenue_usd,
            "california_revenue_usd": input_data.company_info.california_revenue_usd,
            "organizational_boundary": input_data.company_info.organizational_boundary.value,
            "reporting_year": input_data.fiscal_year,
        }

        scope1_section = {
            "total_mtco2e": scope1.total_emissions.co2e_mtco2e,
            "stationary_combustion_mtco2e": scope1.stationary_combustion.co2e_mtco2e,
            "mobile_combustion_mtco2e": scope1.mobile_combustion.co2e_mtco2e,
            "process_emissions_mtco2e": scope1.process_emissions.co2e_mtco2e,
            "fugitive_emissions_mtco2e": scope1.fugitive_emissions.co2e_mtco2e,
            "co2_kg": scope1.total_emissions.co2_kg,
            "ch4_kg": scope1.total_emissions.ch4_kg,
            "n2o_kg": scope1.total_emissions.n2o_kg,
            "emission_factors": scope1.emission_factors_used,
        }

        scope2_section = {
            "location_based_mtco2e": scope2.location_based.co2e_mtco2e,
            "market_based_mtco2e": scope2.market_based.co2e_mtco2e,
            "total_electricity_mwh": scope2.total_electricity_mwh,
            "renewable_mwh": scope2.renewable_mwh,
            "renewable_percentage": scope2.renewable_percentage,
            "egrid_factors_used": scope2.egrid_factors_used,
        }

        scope3_section = None
        if scope3:
            scope3_section = {
                "total_mtco2e": scope3.total_emissions_mtco2e,
                "upstream_mtco2e": scope3.upstream_total_mtco2e,
                "downstream_mtco2e": scope3.downstream_total_mtco2e,
                "categories": [
                    {
                        "number": c.category_number,
                        "name": c.category_name,
                        "mtco2e": c.co2e_mtco2e,
                        "method": c.calculation_method,
                        "data_quality": c.data_quality_score,
                    }
                    for c in scope3.categories
                ],
            }

        methodology_section = {
            "ghg_protocol_version": "Corporate Standard (2015)",
            "scope_2_guidance": "Scope 2 Guidance (2015)",
            "scope_3_standard": "Corporate Value Chain Standard (2011)",
            "gwp_set": input_data.gwp_set.value,
            "emission_factor_sources": [
                "EPA GHG Emission Factors Hub 2024",
                "EPA eGRID 2022",
                "EPA USEEIO v2.0",
            ],
        }

        verification_section = {
            "assurance_level": input_data.assurance_level.value,
            "assurance_standards": ["ISAE 3410", "AT-C Section 105"],
            "verification_deadline": "2026-06-30" if input_data.fiscal_year == 2025 else f"{input_data.fiscal_year + 1}-06-30",
        }

        return CARBFilingData(
            filing_id=filing_id,
            company_section=company_section,
            scope1_section=scope1_section,
            scope2_section=scope2_section,
            scope3_section=scope3_section,
            methodology_section=methodology_section,
            verification_section=verification_section,
            xml_ready=True,
            pdf_ready=True,
        )

    # =========================================================================
    # ASSURANCE PACKAGE GENERATION
    # =========================================================================

    def _generate_assurance_package(
        self,
        input_data: SB253ReportInput,
        scope1: Scope1Result,
        scope2: Scope2Result,
        scope3: Optional[Scope3Result]
    ) -> AssurancePackage:
        """
        Generate third-party assurance documentation package.

        Creates comprehensive documentation for ISAE 3410 verification.
        """
        methodology_notes = f"""
California SB 253 Climate Disclosure - Calculation Methodology

1. ORGANIZATIONAL BOUNDARY
   Approach: {input_data.company_info.organizational_boundary.value}
   Scope: All facilities under {input_data.company_info.organizational_boundary.value}

2. SCOPE 1 METHODOLOGY
   - Stationary Combustion: EPA emission factors (GHG Factors Hub 2024)
   - Mobile Combustion: EPA emission factors by fuel type
   - Fugitive Emissions: IPCC {input_data.gwp_set.value} GWP values
   - Formula: CO2e = Activity * EF * GWP

3. SCOPE 2 METHODOLOGY
   - Location-based: EPA eGRID 2022 subregional factors
   - Market-based: Contractual instruments (PPAs, RECs, utility-specific)
   - Formula: CO2e = kWh * EF (converted from lb/MWh to kg/kWh)

4. SCOPE 3 METHODOLOGY
   - Spend-based: EPA USEEIO v2.0 emission factors
   - Activity-based: Category-specific factors (DEFRA, GLEC)
   - Data quality scoring per GHG Protocol guidance

5. GLOBAL WARMING POTENTIALS
   - Source: IPCC {input_data.gwp_set.value}
   - CO2: 1, CH4: {self.GWP_VALUES[input_data.gwp_set.value]['CH4']}, N2O: {self.GWP_VALUES[input_data.gwp_set.value]['N2O']}

6. UNCERTAINTY ASSESSMENT
   - Scope 1: +/- 5% (measurement-based)
   - Scope 2: +/- 10% (eGRID factor uncertainty)
   - Scope 3: +/- 50% (average for spend-based)
        """.strip()

        data_sources = [
            {"name": "Fuel consumption records", "type": "primary", "period": str(input_data.fiscal_year)},
            {"name": "Electricity invoices", "type": "primary", "period": str(input_data.fiscal_year)},
            {"name": "EPA eGRID database", "type": "secondary", "version": "2022"},
            {"name": "EPA GHG Emission Factors Hub", "type": "secondary", "version": "2024"},
            {"name": "EPA USEEIO model", "type": "secondary", "version": "v2.0"},
        ]

        evidence_index = [
            {"id": "E001", "description": "Scope 1 fuel consumption records", "location": "ERP System"},
            {"id": "E002", "description": "Scope 2 electricity invoices", "location": "Accounts Payable"},
            {"id": "E003", "description": "Refrigerant tracking logs", "location": "Facilities Management"},
            {"id": "E004", "description": "Fleet mileage records", "location": "Fleet Management"},
            {"id": "E005", "description": "Scope 3 spend data", "location": "Procurement System"},
        ]

        control_activities = [
            "Monthly fuel consumption reconciliation",
            "Quarterly electricity usage review",
            "Annual refrigerant inventory audit",
            "Fleet mileage verification",
            "Emission factor update review",
            "Calculation methodology documentation",
            "Data quality assessment",
        ]

        # Calculate completeness score
        completeness_items = [
            len(input_data.scope1_sources) > 0,
            len(input_data.scope2_sources) > 0,
            scope3 is not None,
            len(input_data.facilities) > 0,
        ]
        completeness_score = (sum(completeness_items) / len(completeness_items)) * 100

        return AssurancePackage(
            assurance_level=input_data.assurance_level.value,
            standards_applied=["ISAE 3410", "ISAE 3000", "AT-C Section 105"],
            methodology_notes=methodology_notes,
            data_sources=data_sources,
            evidence_index=evidence_index,
            control_activities=control_activities,
            ready_for_verification=completeness_score >= 75,
            completeness_score=round(completeness_score, 1),
        )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_sb253_applicability(self, input_data: SB253ReportInput) -> None:
        """Validate company meets SB 253 applicability requirements."""
        # Revenue threshold
        if input_data.company_info.total_revenue_usd < 1_000_000_000:
            self._validation_errors.append(
                f"Company revenue ${input_data.company_info.total_revenue_usd:,.0f} "
                f"is below SB 253 $1B threshold"
            )

        # Scope 3 timeline check
        if input_data.fiscal_year >= 2027 and not input_data.include_scope3:
            self._validation_errors.append(
                "Scope 3 reporting is mandatory for fiscal year 2027 onwards"
            )

    # =========================================================================
    # PROVENANCE TRACKING
    # =========================================================================

    def _track_provenance(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tool_name: str
    ) -> None:
        """Track a calculation step for audit trail."""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        output_str = json.dumps(outputs, sort_keys=True, default=str)

        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(input_str.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode()).hexdigest(),
            "tool_name": tool_name,
            "parameters": inputs,
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of complete provenance chain."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": [
                {
                    "operation": s["operation"],
                    "input_hash": s["input_hash"],
                    "output_hash": s["output_hash"],
                }
                for s in self._provenance_steps
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_scope3_categories(self) -> List[Dict[str, Any]]:
        """Get list of Scope 3 categories with descriptions."""
        return [
            {"number": num, "name": name}
            for num, name in self.SCOPE3_CATEGORY_NAMES.items()
        ]

    def get_egrid_subregions(self) -> List[Dict[str, str]]:
        """Get list of supported eGRID subregions."""
        return [
            {"code": code, "name": factor.subregion_name}
            for code, factor in self.EGRID_FACTORS.items()
        ]

    def get_supported_fuels(self) -> List[str]:
        """Get list of supported fuel types."""
        return list(self.SCOPE1_EMISSION_FACTORS.keys())


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "regulatory/sb253_disclosure_v1",
    "name": "SB 253 Climate Disclosure Agent",
    "version": "1.0.0",
    "summary": "California SB 253 Climate Corporate Data Accountability Act compliance agent",
    "tags": ["sb253", "california", "ghg-protocol", "scope1", "scope2", "scope3", "carb"],
    "owners": ["regulatory-team"],
    "compute": {
        "entrypoint": "python://agents.gl_013_sb253_disclosure.agent:SB253DisclosureAgent",
        "deterministic": True,
    },
    "factors": [
        {"ref": "ef://epa/ghg-hub/2024"},
        {"ref": "ef://epa/egrid/2022"},
        {"ref": "ef://epa/useeio/v2.0"},
        {"ref": "ef://ipcc/gwp/ar5"},
    ],
    "provenance": {
        "ef_version_pin": "2024-Q4",
        "gwp_set": "AR5",
        "enable_audit": True,
    },
    "regulatory": {
        "regulation": "California SB 253",
        "effective_date": "2024-01-01",
        "first_scope_12_report": "2026-06-30",
        "first_scope_3_report": "2027-06-30",
        "enforcement_agency": "CARB",
    },
}
