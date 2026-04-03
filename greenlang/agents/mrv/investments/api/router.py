"""
Investments Agent API Router - AGENT-MRV-028

This module implements the FastAPI router for financed emissions / investment
portfolio calculations following GHG Protocol Scope 3 Category 15 and the
Partnership for Carbon Accounting Financials (PCAF) Global Standard.

Provides 24 REST endpoints for:
- Emissions calculations (full pipeline, equity, private equity, corporate bond,
  project finance, CRE, mortgage, motor vehicle loan, sovereign bond, batch,
  portfolio with WACI)
- Multi-framework compliance checking (GHG Protocol, PCAF, TCFD, NZBA, CSRD,
  CDP, SBTi, ISO 14064, GRI)
- Calculation CRUD (get, list, soft-delete)
- Emission factor lookups by asset class
- Sector emission factors (12 GICS sectors)
- Country emission factors (50+ countries)
- PCAF data quality criteria by asset class
- WACI and portfolio carbon intensity
- Temperature alignment and SBTi status
- Time-series aggregations
- Provenance chain verification with SHA-256 hashing
- Health check with 7 engine statuses

Follows GreenLang's zero-hallucination principle with deterministic calculations.
All numeric outputs use deterministic formulas (attribution factor * company emissions);
no LLM calls in the calculation path.

Agent ID: GL-MRV-S3-015
Prefix: gl_inv_

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.agents.mrv.investments.api.router import router
    >>> app = FastAPI()
    >>> app.include_router(router)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from decimal import Decimal
from enum import Enum
import logging
import uuid
import hashlib
from datetime import datetime, date

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    tags=["investments"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# ENUMS
# ============================================================================


class AssetClass(str, Enum):
    """PCAF asset class categories for financed emissions."""

    LISTED_EQUITY = "listed_equity"
    PRIVATE_EQUITY = "private_equity"
    CORPORATE_BOND = "corporate_bond"
    BUSINESS_LOAN = "business_loan"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGE = "mortgage"
    MOTOR_VEHICLE_LOAN = "motor_vehicle_loan"
    SOVEREIGN_BOND = "sovereign_bond"


class CalculationMethod(str, Enum):
    """Supported calculation methods for financed emissions."""

    INVESTMENT_SPECIFIC = "investment_specific"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"
    PCAF_ATTRIBUTION = "pcaf_attribution"
    SOVEREIGN_PPP = "sovereign_ppp"
    BUILDING_EUI = "building_eui"
    VEHICLE_SPECIFIC = "vehicle_specific"


class ComplianceFramework(str, Enum):
    """Supported regulatory and voluntary compliance frameworks."""

    GHG_PROTOCOL = "ghg_protocol"
    PCAF = "pcaf"
    TCFD = "tcfd"
    NZBA = "nzba"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    GRI = "gri"


class GICSSector(str, Enum):
    """GICS sector classifications for sector emission factors."""

    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTH_CARE = "health_care"
    FINANCIALS = "financials"
    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    SOVEREIGN = "sovereign"


class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approaches for portfolio aggregation."""

    EQUITY_SHARE = "equity_share"
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"


class PropertyType(str, Enum):
    """Commercial real estate and residential property types."""

    OFFICE = "office"
    RETAIL = "retail"
    INDUSTRIAL = "industrial"
    MULTIFAMILY = "multifamily"
    HOTEL = "hotel"
    DATA_CENTER = "data_center"
    WAREHOUSE = "warehouse"
    HOSPITAL = "hospital"
    SINGLE_FAMILY = "single_family"
    TOWNHOUSE = "townhouse"
    APARTMENT = "apartment"
    CONDO = "condo"


class VehicleCategory(str, Enum):
    """Motor vehicle categories for auto loan portfolios."""

    PASSENGER_CAR_SMALL = "passenger_car_small"
    PASSENGER_CAR_MEDIUM = "passenger_car_medium"
    PASSENGER_CAR_LARGE = "passenger_car_large"
    SUV_SMALL = "suv_small"
    SUV_LARGE = "suv_large"
    LIGHT_TRUCK = "light_truck"
    HEAVY_TRUCK = "heavy_truck"
    MOTORCYCLE = "motorcycle"
    ELECTRIC_VEHICLE = "electric_vehicle"
    HYBRID = "hybrid"
    PLUG_IN_HYBRID = "plug_in_hybrid"


class EPCRating(str, Enum):
    """Energy Performance Certificate ratings for mortgages."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    UNKNOWN = "unknown"


# ============================================================================
# SERVICE DEPENDENCY
# ============================================================================


_service_instance = None


def get_service():
    """
    Get or create InvestmentsService singleton instance.

    Returns:
        InvestmentsService instance

    Raises:
        HTTPException: If service initialization fails (503)
    """
    global _service_instance

    if _service_instance is None:
        try:
            from greenlang.agents.mrv.investments.service import InvestmentsService
            _service_instance = InvestmentsService()
            logger.info("InvestmentsService initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize InvestmentsService: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service initialization failed",
            )

    return _service_instance


# ============================================================================
# REQUEST MODELS - EQUITY INVESTMENTS
# ============================================================================


class EquityCalculationRequest(BaseModel):
    """
    Request model for listed equity financed emissions calculation.

    Uses PCAF attribution: outstanding_amount / EVIC * (Scope1 + Scope2).
    PCAF data quality score 1 (reported) to 5 (estimated via sector avg).

    Attributes:
        company_name: Name of the investee company
        isin: ISIN identifier for the security
        outstanding_amount: Investor's outstanding amount (USD)
        evic: Enterprise Value Including Cash (USD)
        scope1: Company reported Scope 1 emissions (tCO2e)
        scope2: Company reported Scope 2 emissions (tCO2e)
        scope3: Optional company Scope 3 emissions (tCO2e)
        sector: GICS sector classification
        country: ISO 3166-1 alpha-3 country code
        reporting_year: Year of emissions data
        data_source: Source of emissions data (reported, estimated, sector_avg)
    """

    company_name: str = Field(
        ..., min_length=1, max_length=300, description="Investee company name"
    )
    isin: Optional[str] = Field(
        None, min_length=12, max_length=12, description="ISIN identifier"
    )
    outstanding_amount: float = Field(
        ..., gt=0, description="Outstanding amount in USD"
    )
    evic: float = Field(
        ..., gt=0, description="Enterprise Value Including Cash (USD)"
    )
    scope1: float = Field(
        ..., ge=0, description="Scope 1 emissions (tCO2e)"
    )
    scope2: float = Field(
        ..., ge=0, description="Scope 2 emissions (tCO2e)"
    )
    scope3: Optional[float] = Field(
        None, ge=0, description="Scope 3 emissions (tCO2e)"
    )
    sector: str = Field(
        ..., max_length=50, description="GICS sector"
    )
    country: str = Field(
        "USA", min_length=2, max_length=3, description="ISO country code"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )
    data_source: str = Field(
        "reported",
        description="Emissions data source (reported, estimated, sector_avg)",
    )


class PrivateEquityRequest(BaseModel):
    """
    Request model for private equity / unlisted equity financed emissions.

    Uses PCAF attribution: outstanding_amount / total_equity_plus_debt * emissions.
    Private equity typically scores PCAF 3-5 due to limited disclosure.

    Attributes:
        company_name: Name of the portfolio company
        outstanding_amount: Investor equity contribution (USD)
        total_equity_plus_debt: Total company equity plus debt (USD)
        company_emissions: Total company GHG emissions (tCO2e)
        sector: GICS sector classification
        country: ISO country code
        revenue_usd: Optional company revenue for intensity calculations
        employee_count: Optional employee count for estimation
        reporting_year: Reporting year
    """

    company_name: str = Field(
        ..., min_length=1, max_length=300, description="Portfolio company name"
    )
    outstanding_amount: float = Field(
        ..., gt=0, description="Equity contribution (USD)"
    )
    total_equity_plus_debt: float = Field(
        ..., gt=0, description="Total equity + debt (USD)"
    )
    company_emissions: Optional[float] = Field(
        None, ge=0, description="Company GHG emissions (tCO2e)"
    )
    sector: str = Field(
        ..., max_length=50, description="GICS sector"
    )
    country: str = Field(
        "USA", min_length=2, max_length=3, description="ISO country code"
    )
    revenue_usd: Optional[float] = Field(
        None, ge=0, description="Company revenue (USD)"
    )
    employee_count: Optional[int] = Field(
        None, ge=1, description="Number of employees"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )


class CorporateBondRequest(BaseModel):
    """
    Request model for corporate bond financed emissions.

    Uses PCAF attribution: outstanding_amount / EVIC * company_emissions.
    Bond holders and equity holders share EVIC denominator.

    Attributes:
        company_name: Bond issuer name
        outstanding_amount: Bond face value held (USD)
        evic: Enterprise Value Including Cash (USD)
        company_emissions: Issuer total GHG emissions (tCO2e)
        sector: GICS sector classification
        country: ISO country code
        maturity_date: Bond maturity date
        coupon_rate: Annual coupon rate as decimal
        credit_rating: Optional credit rating (AAA-D)
        reporting_year: Reporting year
    """

    company_name: str = Field(
        ..., min_length=1, max_length=300, description="Bond issuer name"
    )
    outstanding_amount: float = Field(
        ..., gt=0, description="Bond face value held (USD)"
    )
    evic: float = Field(
        ..., gt=0, description="Enterprise Value Including Cash (USD)"
    )
    company_emissions: Optional[float] = Field(
        None, ge=0, description="Issuer GHG emissions (tCO2e)"
    )
    sector: str = Field(
        ..., max_length=50, description="GICS sector"
    )
    country: str = Field(
        "USA", min_length=2, max_length=3, description="ISO country code"
    )
    maturity_date: Optional[str] = Field(
        None, description="Bond maturity date (ISO 8601)"
    )
    coupon_rate: Optional[float] = Field(
        None, ge=0, le=1, description="Annual coupon rate (0.0-1.0)"
    )
    credit_rating: Optional[str] = Field(
        None, max_length=10, description="Credit rating (AAA-D)"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )


# ============================================================================
# REQUEST MODELS - PROJECT FINANCE & REAL ASSETS
# ============================================================================


class ProjectFinanceRequest(BaseModel):
    """
    Request model for project finance financed emissions.

    Uses PCAF attribution: outstanding_amount / total_project_cost * project_emissions.
    Applies to infrastructure, renewable energy, and large capital projects.

    Attributes:
        project_name: Name of the financed project
        outstanding_amount: Investor's outstanding amount (USD)
        total_project_cost: Total project cost (USD)
        project_emissions: Annual project GHG emissions (tCO2e)
        sector: GICS sector or project type
        country: ISO country code
        lifetime_years: Expected project lifetime in years
        project_type: Type of project (renewable, infrastructure, industrial, etc.)
        is_greenfield: Whether this is a new construction project
        reporting_year: Reporting year
    """

    project_name: str = Field(
        ..., min_length=1, max_length=300, description="Project name"
    )
    outstanding_amount: float = Field(
        ..., gt=0, description="Outstanding loan amount (USD)"
    )
    total_project_cost: float = Field(
        ..., gt=0, description="Total project cost (USD)"
    )
    project_emissions: Optional[float] = Field(
        None, ge=0, description="Annual project emissions (tCO2e)"
    )
    sector: str = Field(
        ..., max_length=50, description="Sector or project type"
    )
    country: str = Field(
        "USA", min_length=2, max_length=3, description="ISO country code"
    )
    lifetime_years: int = Field(
        25, ge=1, le=100, description="Project lifetime (years)"
    )
    project_type: Optional[str] = Field(
        None, description="Project type (renewable, infrastructure, industrial)"
    )
    is_greenfield: bool = Field(
        False, description="Greenfield (new construction) flag"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )


class CRERequest(BaseModel):
    """
    Request model for commercial real estate financed emissions.

    Uses building energy performance: EUI * floor_area * grid_EF.
    PCAF quality 1 (actual energy data) to 5 (estimated from benchmarks).

    Attributes:
        property_type: Type of commercial property
        outstanding_amount: Outstanding loan/investment amount (USD)
        property_value: Total property value (USD)
        floor_area_m2: Gross floor area in square metres
        energy_kwh: Annual energy consumption in kWh (if known)
        location: Property location (country or region)
        year_built: Year of construction
        epc_rating: Energy Performance Certificate rating
        occupancy_rate: Current occupancy rate (0.0-1.0)
        reporting_year: Reporting year
    """

    property_type: str = Field(
        ..., description="Property type (office, retail, industrial, etc.)"
    )
    outstanding_amount: float = Field(
        ..., gt=0, description="Outstanding loan/investment (USD)"
    )
    property_value: float = Field(
        ..., gt=0, description="Total property value (USD)"
    )
    floor_area_m2: float = Field(
        ..., gt=0, le=1000000, description="Gross floor area (m2)"
    )
    energy_kwh: Optional[float] = Field(
        None, ge=0, description="Annual energy consumption (kWh)"
    )
    location: str = Field(
        "USA", min_length=2, max_length=50, description="Country or region"
    )
    year_built: Optional[int] = Field(
        None, ge=1800, le=2100, description="Construction year"
    )
    epc_rating: Optional[str] = Field(
        None, description="Energy Performance Certificate rating (A-G)"
    )
    occupancy_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Current occupancy rate"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )


class MortgageRequest(BaseModel):
    """
    Request model for residential mortgage financed emissions.

    Uses PCAF attribution: outstanding_loan / property_value * building_emissions.
    Building emissions from EPC, actual energy data, or benchmarks.

    Attributes:
        outstanding_loan: Outstanding mortgage balance (USD)
        property_value: Property valuation at origination (USD)
        property_type: Residential property type
        floor_area_m2: Floor area in square metres
        epc_rating: Energy Performance Certificate rating
        location: Property country or region
        year_built: Year of construction
        energy_kwh: Actual annual energy consumption if known
        reporting_year: Reporting year
    """

    outstanding_loan: float = Field(
        ..., gt=0, description="Outstanding mortgage balance (USD)"
    )
    property_value: float = Field(
        ..., gt=0, description="Property value at origination (USD)"
    )
    property_type: str = Field(
        "single_family", description="Residential property type"
    )
    floor_area_m2: float = Field(
        ..., gt=0, le=50000, description="Floor area (m2)"
    )
    epc_rating: Optional[str] = Field(
        None, description="EPC rating (A+-G, unknown)"
    )
    location: str = Field(
        "USA", min_length=2, max_length=50, description="Country or region"
    )
    year_built: Optional[int] = Field(
        None, ge=1800, le=2100, description="Construction year"
    )
    energy_kwh: Optional[float] = Field(
        None, ge=0, description="Actual annual energy (kWh)"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )


class MotorVehicleRequest(BaseModel):
    """
    Request model for motor vehicle loan financed emissions.

    Uses PCAF attribution: outstanding_loan / vehicle_value * vehicle_emissions.
    Vehicle emissions from make/model lookup or category average.

    Attributes:
        outstanding_loan: Outstanding loan balance (USD)
        vehicle_value: Vehicle value at origination (USD)
        vehicle_category: Vehicle category classification
        make: Vehicle manufacturer
        model: Vehicle model name
        year: Vehicle model year
        annual_km: Estimated annual kilometres driven
        fuel_type: Fuel type (gasoline, diesel, electric, hybrid)
        reporting_year: Reporting year
    """

    outstanding_loan: float = Field(
        ..., gt=0, description="Outstanding auto loan (USD)"
    )
    vehicle_value: float = Field(
        ..., gt=0, description="Vehicle value at origination (USD)"
    )
    vehicle_category: str = Field(
        ..., description="Vehicle category (passenger_car_medium, suv_large, etc.)"
    )
    make: Optional[str] = Field(
        None, max_length=100, description="Vehicle manufacturer"
    )
    model: Optional[str] = Field(
        None, max_length=100, description="Vehicle model"
    )
    year: Optional[int] = Field(
        None, ge=1990, le=2100, description="Vehicle model year"
    )
    annual_km: Optional[float] = Field(
        None, ge=0, le=500000, description="Annual km driven"
    )
    fuel_type: Optional[str] = Field(
        None, description="Fuel type (gasoline, diesel, electric, hybrid)"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )


class SovereignBondRequest(BaseModel):
    """
    Request model for sovereign bond financed emissions.

    Uses PCAF sovereign: outstanding_amount / GDP_PPP * country_emissions.
    PCAF data quality depends on availability of national GHG inventory.

    Attributes:
        country_code: ISO 3166-1 alpha-3 country code
        outstanding_amount: Outstanding sovereign bond holding (USD)
        gdp_ppp: GDP at purchasing power parity (billion USD)
        country_emissions_mt: National GHG emissions (MtCO2e)
        include_lulucf: Whether to include LULUCF emissions
        reporting_year: Reporting year
    """

    country_code: str = Field(
        ..., min_length=3, max_length=3, description="ISO alpha-3 country code"
    )
    outstanding_amount: float = Field(
        ..., gt=0, description="Sovereign bond holding (USD)"
    )
    gdp_ppp: Optional[float] = Field(
        None, gt=0, description="GDP PPP (billion USD)"
    )
    country_emissions_mt: Optional[float] = Field(
        None, ge=0, description="National GHG emissions (MtCO2e)"
    )
    include_lulucf: bool = Field(
        False, description="Include LULUCF in national emissions"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )


# ============================================================================
# REQUEST MODELS - PORTFOLIO & BATCH
# ============================================================================


class InvestmentPosition(BaseModel):
    """
    A single investment position within a portfolio or batch request.

    Supports all 8 PCAF asset classes with flexible field mapping.

    Attributes:
        investment_id: Unique identifier for this position
        asset_class: PCAF asset class category
        company_name: Investee / borrower / issuer / country name
        outstanding_amount: Outstanding amount in USD
        sector: GICS sector or sovereign
        country: ISO country code
        evic: Enterprise Value Including Cash (for equity/bonds)
        scope1: Reported Scope 1 emissions (tCO2e)
        scope2: Reported Scope 2 emissions (tCO2e)
        scope3: Optional Scope 3 emissions (tCO2e)
        total_equity_plus_debt: For private equity
        revenue_usd: Company revenue
        floor_area_m2: For CRE/mortgage
        property_value: For CRE/mortgage
        vehicle_value: For auto loans
        vehicle_category: For auto loans
        gdp_ppp: For sovereign bonds
    """

    investment_id: str = Field(
        ..., min_length=1, max_length=100, description="Position identifier"
    )
    asset_class: str = Field(
        ..., description="PCAF asset class"
    )
    company_name: Optional[str] = Field(
        None, max_length=300, description="Investee/borrower/issuer name"
    )
    outstanding_amount: float = Field(
        ..., gt=0, description="Outstanding amount (USD)"
    )
    sector: Optional[str] = Field(
        None, max_length=50, description="GICS sector"
    )
    country: str = Field(
        "USA", min_length=2, max_length=3, description="ISO country code"
    )
    evic: Optional[float] = Field(
        None, gt=0, description="EVIC for equity/bonds (USD)"
    )
    scope1: Optional[float] = Field(
        None, ge=0, description="Scope 1 emissions (tCO2e)"
    )
    scope2: Optional[float] = Field(
        None, ge=0, description="Scope 2 emissions (tCO2e)"
    )
    scope3: Optional[float] = Field(
        None, ge=0, description="Scope 3 emissions (tCO2e)"
    )
    total_equity_plus_debt: Optional[float] = Field(
        None, gt=0, description="Total equity + debt (USD)"
    )
    revenue_usd: Optional[float] = Field(
        None, ge=0, description="Company revenue (USD)"
    )
    floor_area_m2: Optional[float] = Field(
        None, gt=0, description="Floor area (m2)"
    )
    property_value: Optional[float] = Field(
        None, gt=0, description="Property / vehicle value (USD)"
    )
    vehicle_value: Optional[float] = Field(
        None, gt=0, description="Vehicle value (USD)"
    )
    vehicle_category: Optional[str] = Field(
        None, description="Vehicle category"
    )
    gdp_ppp: Optional[float] = Field(
        None, gt=0, description="GDP PPP (billion USD)"
    )
    energy_kwh: Optional[float] = Field(
        None, ge=0, description="Annual energy (kWh)"
    )
    data_source: Optional[str] = Field(
        None, description="Emissions data source"
    )


class PortfolioRequest(BaseModel):
    """
    Request model for full portfolio analysis with WACI.

    Calculates financed emissions for an entire investment portfolio,
    producing WACI, carbon intensity, and per-position attribution.

    Attributes:
        investments: List of investment positions
        reporting_period: Reporting period identifier (e.g., 2024-Q4)
        total_aum: Total assets under management (USD)
        consolidation_approach: GHG Protocol consolidation approach
        include_scope3: Whether to include investee Scope 3
        currency: Portfolio reporting currency
        portfolio_name: Optional portfolio name
    """

    investments: List[InvestmentPosition] = Field(
        ..., min_items=1, max_items=50000,
        description="Investment positions"
    )
    reporting_period: str = Field(
        ..., min_length=1, max_length=20,
        description="Reporting period (e.g., 2024-Q4)"
    )
    total_aum: float = Field(
        ..., gt=0, description="Total AUM (USD)"
    )
    consolidation_approach: str = Field(
        "equity_share",
        description="Consolidation approach (equity_share, financial_control, operational_control)",
    )
    include_scope3: bool = Field(
        False, description="Include investee Scope 3"
    )
    currency: str = Field(
        "USD", min_length=3, max_length=3, description="Reporting currency"
    )
    portfolio_name: Optional[str] = Field(
        None, max_length=200, description="Portfolio name"
    )


class BatchRequest(BaseModel):
    """
    Request model for batch calculation of mixed asset class positions.

    Processes up to 50,000 positions across multiple PCAF asset classes
    in a single request. Returns per-position results and batch summary.

    Attributes:
        investments: List of investment positions (mixed asset classes)
        reporting_period: Reporting period identifier
        fail_on_error: Whether to fail entire batch on any error
    """

    investments: List[InvestmentPosition] = Field(
        ..., min_items=1, max_items=50000,
        description="Investment positions (mixed asset classes)"
    )
    reporting_period: str = Field(
        ..., min_length=1, max_length=20,
        description="Reporting period"
    )
    fail_on_error: bool = Field(
        False, description="Fail entire batch on any position error"
    )


# ============================================================================
# REQUEST MODELS - COMPLIANCE
# ============================================================================


class ComplianceCheckRequest(BaseModel):
    """
    Request model for multi-framework compliance checking.

    Validates financed emissions calculations against up to 9 frameworks:
    GHG Protocol Scope 3, PCAF Global Standard, TCFD, NZBA, CSRD ESRS,
    CDP, SBTi, ISO 14064, GRI 305.

    Attributes:
        calculation_id: UUID of the calculation to check
        frameworks: List of frameworks to check against
        include_recommendations: Whether to include improvement recommendations
    """

    calculation_id: str = Field(
        ..., description="Calculation UUID to check"
    )
    frameworks: List[str] = Field(
        default_factory=lambda: [
            "ghg_protocol", "pcaf", "tcfd", "nzba",
            "csrd_esrs", "cdp", "sbti", "iso_14064", "gri"
        ],
        description="Frameworks to check against",
    )
    include_recommendations: bool = Field(
        True, description="Include improvement recommendations"
    )


# ============================================================================
# RESPONSE MODELS - CALCULATION RESULTS
# ============================================================================


class EquityCalculationResponse(BaseModel):
    """Response model for listed equity financed emissions."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    asset_class: str = Field(
        "listed_equity", description="PCAF asset class"
    )
    company_name: str = Field(..., description="Investee company name")
    method: str = Field(..., description="Calculation method")
    attribution_factor: float = Field(
        ..., description="Attribution factor (outstanding / EVIC)"
    )
    financed_emissions_tco2e: float = Field(
        ..., description="Financed Scope 1+2 emissions (tCO2e)"
    )
    financed_scope3_tco2e: Optional[float] = Field(
        None, description="Financed Scope 3 emissions (tCO2e)"
    )
    carbon_intensity_tco2e_per_m_invested: float = Field(
        ..., description="Carbon intensity (tCO2e per million USD invested)"
    )
    pcaf_data_quality: int = Field(
        ..., ge=1, le=5, description="PCAF data quality score (1=best, 5=worst)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class PrivateEquityResponse(BaseModel):
    """Response model for private equity financed emissions."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    asset_class: str = Field(
        "private_equity", description="PCAF asset class"
    )
    company_name: str = Field(..., description="Portfolio company name")
    method: str = Field(..., description="Calculation method")
    attribution_factor: float = Field(
        ..., description="Attribution factor (equity / total equity+debt)"
    )
    financed_emissions_tco2e: float = Field(
        ..., description="Financed emissions (tCO2e)"
    )
    carbon_intensity_tco2e_per_m_invested: float = Field(
        ..., description="Carbon intensity (tCO2e/M USD)"
    )
    pcaf_data_quality: int = Field(
        ..., ge=1, le=5, description="PCAF data quality score"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: str = Field(..., description="ISO 8601 timestamp")


class CorporateBondResponse(BaseModel):
    """Response model for corporate bond financed emissions."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    asset_class: str = Field(
        "corporate_bond", description="PCAF asset class"
    )
    company_name: str = Field(..., description="Bond issuer name")
    method: str = Field(..., description="Calculation method")
    attribution_factor: float = Field(
        ..., description="Attribution factor (outstanding / EVIC)"
    )
    financed_emissions_tco2e: float = Field(
        ..., description="Financed emissions (tCO2e)"
    )
    carbon_intensity_tco2e_per_m_invested: float = Field(
        ..., description="Carbon intensity (tCO2e/M USD)"
    )
    pcaf_data_quality: int = Field(
        ..., ge=1, le=5, description="PCAF data quality score"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: str = Field(..., description="ISO 8601 timestamp")


class ProjectFinanceResponse(BaseModel):
    """Response model for project finance financed emissions."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    asset_class: str = Field(
        "project_finance", description="PCAF asset class"
    )
    project_name: str = Field(..., description="Project name")
    method: str = Field(..., description="Calculation method")
    attribution_factor: float = Field(
        ..., description="Attribution factor (outstanding / project cost)"
    )
    financed_emissions_tco2e: float = Field(
        ..., description="Financed emissions (tCO2e)"
    )
    lifetime_financed_emissions_tco2e: Optional[float] = Field(
        None, description="Lifetime financed emissions (tCO2e)"
    )
    carbon_intensity_tco2e_per_m_invested: float = Field(
        ..., description="Carbon intensity (tCO2e/M USD)"
    )
    pcaf_data_quality: int = Field(
        ..., ge=1, le=5, description="PCAF data quality score"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: str = Field(..., description="ISO 8601 timestamp")


class CREResponse(BaseModel):
    """Response model for commercial real estate financed emissions."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    asset_class: str = Field(
        "commercial_real_estate", description="PCAF asset class"
    )
    property_type: str = Field(..., description="Property type")
    method: str = Field(..., description="Calculation method")
    attribution_factor: float = Field(
        ..., description="Attribution factor (outstanding / property value)"
    )
    building_emissions_tco2e: float = Field(
        ..., description="Total building emissions (tCO2e)"
    )
    financed_emissions_tco2e: float = Field(
        ..., description="Financed emissions (tCO2e)"
    )
    eui_kwh_per_m2: Optional[float] = Field(
        None, description="Energy Use Intensity (kWh/m2)"
    )
    carbon_intensity_kgco2e_per_m2: Optional[float] = Field(
        None, description="Carbon intensity (kgCO2e/m2)"
    )
    pcaf_data_quality: int = Field(
        ..., ge=1, le=5, description="PCAF data quality score"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: str = Field(..., description="ISO 8601 timestamp")


class MortgageResponse(BaseModel):
    """Response model for mortgage financed emissions."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    asset_class: str = Field("mortgage", description="PCAF asset class")
    property_type: str = Field(..., description="Property type")
    method: str = Field(..., description="Calculation method")
    attribution_factor: float = Field(
        ..., description="Attribution factor (loan / property value)"
    )
    building_emissions_tco2e: float = Field(
        ..., description="Total building emissions (tCO2e)"
    )
    financed_emissions_tco2e: float = Field(
        ..., description="Financed emissions (tCO2e)"
    )
    eui_kwh_per_m2: Optional[float] = Field(
        None, description="Energy Use Intensity (kWh/m2)"
    )
    epc_rating: Optional[str] = Field(
        None, description="EPC rating used"
    )
    pcaf_data_quality: int = Field(
        ..., ge=1, le=5, description="PCAF data quality score"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: str = Field(..., description="ISO 8601 timestamp")


class MotorVehicleResponse(BaseModel):
    """Response model for motor vehicle loan financed emissions."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    asset_class: str = Field(
        "motor_vehicle_loan", description="PCAF asset class"
    )
    vehicle_category: str = Field(..., description="Vehicle category")
    method: str = Field(..., description="Calculation method")
    attribution_factor: float = Field(
        ..., description="Attribution factor (loan / vehicle value)"
    )
    vehicle_emissions_tco2e: float = Field(
        ..., description="Total vehicle annual emissions (tCO2e)"
    )
    financed_emissions_tco2e: float = Field(
        ..., description="Financed emissions (tCO2e)"
    )
    ef_kgco2e_per_km: Optional[float] = Field(
        None, description="Emission factor used (kgCO2e/km)"
    )
    pcaf_data_quality: int = Field(
        ..., ge=1, le=5, description="PCAF data quality score"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: str = Field(..., description="ISO 8601 timestamp")


class SovereignBondResponse(BaseModel):
    """Response model for sovereign bond financed emissions."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    asset_class: str = Field(
        "sovereign_bond", description="PCAF asset class"
    )
    country_code: str = Field(..., description="ISO alpha-3 country code")
    country_name: Optional[str] = Field(None, description="Country name")
    method: str = Field(..., description="Calculation method")
    attribution_factor: float = Field(
        ..., description="Attribution factor (outstanding / GDP_PPP)"
    )
    financed_emissions_tco2e: float = Field(
        ..., description="Financed emissions (tCO2e)"
    )
    per_capita_tco2e: Optional[float] = Field(
        None, description="Country per-capita emissions"
    )
    pcaf_data_quality: int = Field(
        ..., ge=1, le=5, description="PCAF data quality score"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: str = Field(..., description="ISO 8601 timestamp")


class PortfolioResponse(BaseModel):
    """Response model for full portfolio analysis with WACI."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    portfolio_name: Optional[str] = Field(
        None, description="Portfolio name"
    )
    total_aum: float = Field(
        ..., description="Total AUM (USD)"
    )
    total_financed_emissions_tco2e: float = Field(
        ..., description="Total portfolio financed emissions (tCO2e)"
    )
    waci: float = Field(
        ..., description="Weighted Average Carbon Intensity (tCO2e/M USD revenue)"
    )
    carbon_footprint_tco2e_per_m_invested: float = Field(
        ..., description="Portfolio carbon footprint (tCO2e/M USD invested)"
    )
    total_carbon_emissions_tco2e: float = Field(
        ..., description="Total carbon emissions (absolute)"
    )
    position_count: int = Field(
        ..., description="Number of positions processed"
    )
    coverage_ratio: float = Field(
        ..., ge=0, le=1, description="Data coverage ratio (0.0-1.0)"
    )
    by_asset_class: Dict[str, Dict[str, Any]] = Field(
        ..., description="Breakdown by PCAF asset class"
    )
    by_sector: Dict[str, Dict[str, Any]] = Field(
        ..., description="Breakdown by GICS sector"
    )
    by_country: Dict[str, Dict[str, Any]] = Field(
        ..., description="Breakdown by country"
    )
    weighted_pcaf_score: float = Field(
        ..., ge=1.0, le=5.0, description="Weighted PCAF data quality score"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: str = Field(..., description="ISO 8601 timestamp")


class BatchCalculateResponse(BaseModel):
    """Response model for batch calculation."""

    batch_id: str = Field(..., description="Unique batch UUID")
    results: List[Dict[str, Any]] = Field(
        ..., description="Per-position calculation results"
    )
    total_financed_emissions_tco2e: float = Field(
        ..., description="Total financed emissions (tCO2e)"
    )
    position_count: int = Field(
        ..., description="Successfully processed positions"
    )
    error_count: int = Field(
        0, description="Number of positions with errors"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-position error details"
    )
    by_asset_class: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Summary by asset class"
    )
    processing_time_ms: float = Field(
        ..., description="Total processing time (ms)"
    )


# ============================================================================
# RESPONSE MODELS - COMPLIANCE
# ============================================================================


class ComplianceCheckResponse(BaseModel):
    """Response model for multi-framework compliance check."""

    results: List[Dict[str, Any]] = Field(
        ..., description="Per-framework compliance results"
    )
    overall_status: str = Field(
        ..., description="Overall compliance status (pass, fail, warning)"
    )
    overall_score: float = Field(
        ..., description="Overall compliance score (0.0-1.0)"
    )
    frameworks_checked: int = Field(
        ..., description="Number of frameworks checked"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )


# ============================================================================
# RESPONSE MODELS - DATA RETRIEVAL
# ============================================================================


class CalculationDetailResponse(BaseModel):
    """Response model for single calculation detail."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    asset_class: Optional[str] = Field(None, description="PCAF asset class")
    method: str = Field(..., description="Calculation method")
    total_financed_emissions_kgco2e: float = Field(
        ..., description="Total financed emissions (kgCO2e)"
    )
    pcaf_data_quality: Optional[int] = Field(
        None, description="PCAF data quality score"
    )
    attribution_factor: Optional[float] = Field(
        None, description="Attribution factor used"
    )
    details: Dict[str, Any] = Field(
        ..., description="Full calculation detail payload"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(..., description="ISO 8601 timestamp")


class CalculationListResponse(BaseModel):
    """Response model for paginated calculation listing."""

    calculations: List[Dict[str, Any]] = Field(
        ..., description="Calculation summaries"
    )
    count: int = Field(..., description="Total matching calculations")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")


class DeleteResponse(BaseModel):
    """Response model for soft deletion."""

    calculation_id: str = Field(
        ..., description="Deleted calculation UUID"
    )
    deleted: bool = Field(..., description="Whether deletion succeeded")
    message: str = Field(..., description="Human-readable status message")


class EmissionFactorResponse(BaseModel):
    """Response model for asset-class emission factor data."""

    asset_class: str = Field(..., description="PCAF asset class")
    factor_type: str = Field(
        ..., description="Factor type (sector, country, building, vehicle)"
    )
    factor_value: float = Field(..., description="Emission factor value")
    unit: str = Field(..., description="Factor unit")
    source: str = Field(..., description="Factor data source")
    year: Optional[int] = Field(None, description="Factor vintage year")


class EmissionFactorListResponse(BaseModel):
    """Response model for emission factor listing."""

    factors: List[EmissionFactorResponse] = Field(
        ..., description="List of emission factors"
    )
    count: int = Field(..., description="Total factor count returned")


class SectorFactorResponse(BaseModel):
    """Response model for sector emission factor."""

    sector: str = Field(..., description="GICS sector name")
    gics_code: str = Field(..., description="GICS sector code")
    ef_tco2e_per_m_revenue: float = Field(
        ..., description="tCO2e per million USD revenue"
    )
    source: str = Field(..., description="Factor data source")
    year: int = Field(..., description="Factor year")


class SectorFactorListResponse(BaseModel):
    """Response model for sector factor listing."""

    factors: List[SectorFactorResponse] = Field(
        ..., description="List of sector emission factors"
    )
    count: int = Field(..., description="Total factor count")


class CountryFactorResponse(BaseModel):
    """Response model for country emission factor."""

    country_code: str = Field(..., description="ISO alpha-3 country code")
    country_name: str = Field(..., description="Country name")
    total_ghg_mt: float = Field(
        ..., description="Total GHG emissions (MtCO2e)"
    )
    gdp_ppp_billion_usd: float = Field(
        ..., description="GDP PPP (billion USD)"
    )
    per_capita_tco2e: float = Field(
        ..., description="Per-capita emissions (tCO2e)"
    )
    source: str = Field(..., description="Data source")
    year: int = Field(..., description="Data year")


class CountryFactorListResponse(BaseModel):
    """Response model for country factor listing."""

    factors: List[CountryFactorResponse] = Field(
        ..., description="List of country emission factors"
    )
    count: int = Field(..., description="Total factor count")


class PCAFQualityCriteria(BaseModel):
    """Response model for PCAF data quality criteria."""

    asset_class: str = Field(..., description="PCAF asset class")
    score: int = Field(
        ..., ge=1, le=5, description="PCAF quality score"
    )
    description: str = Field(..., description="Quality level description")
    uncertainty_pct: float = Field(
        ..., description="Typical uncertainty percentage"
    )


class PCAFQualityListResponse(BaseModel):
    """Response model for PCAF data quality listing."""

    criteria: List[PCAFQualityCriteria] = Field(
        ..., description="PCAF quality criteria"
    )
    count: int = Field(..., description="Total criteria count")


class CarbonIntensityResponse(BaseModel):
    """Response model for WACI and portfolio carbon intensity."""

    waci: float = Field(
        ..., description="Weighted Average Carbon Intensity (tCO2e/M USD revenue)"
    )
    carbon_footprint: float = Field(
        ..., description="Portfolio carbon footprint (tCO2e/M USD invested)"
    )
    financed_emissions: float = Field(
        ..., description="Total financed emissions (tCO2e)"
    )
    physical_intensity: Optional[float] = Field(
        None, description="Physical carbon intensity (tCO2e/unit output)"
    )
    by_sector: Dict[str, float] = Field(
        ..., description="Intensity by GICS sector"
    )
    benchmark_comparison: Optional[Dict[str, Any]] = Field(
        None, description="Comparison to sector benchmarks"
    )


class PortfolioAlignmentResponse(BaseModel):
    """Response model for temperature alignment and SBTi status."""

    implied_temperature_rise_c: float = Field(
        ..., description="Implied temperature rise (degrees C)"
    )
    alignment_status: str = Field(
        ..., description="Alignment status (aligned_1_5c, aligned_2c, not_aligned)"
    )
    sbti_committed_pct: float = Field(
        ..., ge=0, le=100, description="% of portfolio with SBTi commitments"
    )
    sbti_validated_pct: float = Field(
        ..., ge=0, le=100, description="% with validated SBTi targets"
    )
    nzba_aligned_pct: float = Field(
        ..., ge=0, le=100, description="% aligned with NZBA"
    )
    by_sector: Dict[str, Dict[str, Any]] = Field(
        ..., description="Alignment by sector"
    )
    trajectory: Optional[List[Dict[str, Any]]] = Field(
        None, description="Emission reduction trajectory"
    )


class AggregationResponse(BaseModel):
    """Response model for time-series aggregated financed emissions."""

    period: str = Field(..., description="Aggregation period identifier")
    total_financed_emissions_tco2e: float = Field(
        ..., description="Total financed emissions (tCO2e)"
    )
    by_asset_class: Dict[str, float] = Field(
        ..., description="Emissions by PCAF asset class"
    )
    by_sector: Dict[str, float] = Field(
        ..., description="Emissions by GICS sector"
    )
    by_country: Dict[str, float] = Field(
        ..., description="Emissions by country"
    )
    position_count: int = Field(
        ..., description="Total positions in period"
    )
    total_aum: Optional[float] = Field(
        None, description="Total AUM for the period (USD)"
    )
    waci: Optional[float] = Field(
        None, description="WACI for the period"
    )


class ProvenanceResponse(BaseModel):
    """Response model for provenance chain verification."""

    calculation_id: str = Field(..., description="Calculation UUID")
    chain: List[Dict[str, Any]] = Field(
        ..., description="Ordered provenance stage records"
    )
    is_valid: bool = Field(
        ..., description="Whether the provenance chain is intact"
    )
    root_hash: str = Field(
        ..., description="Root SHA-256 hash of the chain"
    )


class EngineStatus(BaseModel):
    """Status of a single calculation engine."""

    engine_name: str = Field(..., description="Engine identifier")
    status: str = Field(
        ..., description="Engine status (healthy, degraded, unhealthy)"
    )
    last_used: Optional[str] = Field(
        None, description="Last usage timestamp"
    )


class HealthResponse(BaseModel):
    """Response model for health check with engine status."""

    status: str = Field(..., description="Service health status")
    agent_id: str = Field(..., description="Agent identifier")
    version: str = Field(..., description="Agent version")
    uptime_seconds: float = Field(
        ..., description="Seconds since service start"
    )
    engines: List[EngineStatus] = Field(
        ..., description="Individual engine status"
    )
    asset_classes_loaded: int = Field(
        ..., description="Number of PCAF asset classes loaded"
    )
    emission_factors_loaded: int = Field(
        ..., description="Number of emission factors loaded"
    )


# ============================================================================
# MODULE-LEVEL TRACKING
# ============================================================================

_start_time: datetime = datetime.utcnow()

# Engine names for health check
_ENGINE_NAMES = [
    "InvestmentDatabaseEngine",
    "EquityInvestmentCalculatorEngine",
    "DebtInvestmentCalculatorEngine",
    "RealAssetCalculatorEngine",
    "SovereignBondCalculatorEngine",
    "ComplianceCheckerEngine",
    "InvestmentsPipelineEngine",
]


# ============================================================================
# ENDPOINTS - CALCULATIONS (11 POST)
# ============================================================================


@router.post(
    "/calculate",
    response_model=PortfolioResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate full portfolio financed emissions",
    description=(
        "Calculate financed emissions for an investment portfolio through the "
        "full 10-stage pipeline. Accepts multiple positions across all PCAF "
        "asset classes. Returns deterministic results with WACI, carbon "
        "footprint, attribution factors, and SHA-256 provenance hash."
    ),
)
async def calculate_emissions(
    request: PortfolioRequest,
    service=Depends(get_service),
) -> PortfolioResponse:
    """
    Calculate full portfolio financed emissions through the pipeline.

    Args:
        request: Portfolio calculation request with positions
        service: InvestmentsService instance

    Returns:
        PortfolioResponse with WACI, carbon footprint, and breakdowns

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating portfolio financed emissions: {len(request.investments)} "
            f"positions, AUM={request.total_aum}, "
            f"consolidation={request.consolidation_approach}"
        )

        result = await service.calculate(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return PortfolioResponse(
            calculation_id=calculation_id,
            portfolio_name=result.get("portfolio_name", request.portfolio_name),
            total_aum=result.get("total_aum", request.total_aum),
            total_financed_emissions_tco2e=result.get(
                "total_financed_emissions_tco2e", 0.0
            ),
            waci=result.get("waci", 0.0),
            carbon_footprint_tco2e_per_m_invested=result.get(
                "carbon_footprint_tco2e_per_m_invested", 0.0
            ),
            total_carbon_emissions_tco2e=result.get(
                "total_carbon_emissions_tco2e", 0.0
            ),
            position_count=result.get(
                "position_count", len(request.investments)
            ),
            coverage_ratio=result.get("coverage_ratio", 0.0),
            by_asset_class=result.get("by_asset_class", {}),
            by_sector=result.get("by_sector", {}),
            by_country=result.get("by_country", {}),
            weighted_pcaf_score=result.get("weighted_pcaf_score", 3.0),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_emissions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in calculate_emissions: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Portfolio calculation failed",
        )


@router.post(
    "/calculate/equity",
    response_model=EquityCalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate listed equity financed emissions",
    description=(
        "Calculate financed emissions for a listed equity investment using "
        "PCAF attribution: outstanding_amount / EVIC * (Scope1 + Scope2). "
        "Requires EVIC and investee reported or estimated emissions. Returns "
        "attribution factor, financed emissions, and PCAF data quality score."
    ),
)
async def calculate_equity(
    request: EquityCalculationRequest,
    service=Depends(get_service),
) -> EquityCalculationResponse:
    """
    Calculate listed equity financed emissions.

    Args:
        request: Equity calculation request with EVIC and emissions
        service: InvestmentsService instance

    Returns:
        EquityCalculationResponse with attribution and financed emissions

    Raises:
        HTTPException: 400 for missing EVIC/emissions, 500 for failures
    """
    try:
        logger.info(
            f"Calculating equity financed emissions: company={request.company_name}, "
            f"outstanding={request.outstanding_amount}, EVIC={request.evic}"
        )

        result = await service.calculate_equity(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return EquityCalculationResponse(
            calculation_id=calculation_id,
            asset_class="listed_equity",
            company_name=result.get("company_name", request.company_name),
            method=result.get("method", "pcaf_attribution"),
            attribution_factor=result.get("attribution_factor", 0.0),
            financed_emissions_tco2e=result.get(
                "financed_emissions_tco2e", 0.0
            ),
            financed_scope3_tco2e=result.get("financed_scope3_tco2e"),
            carbon_intensity_tco2e_per_m_invested=result.get(
                "carbon_intensity_tco2e_per_m_invested", 0.0
            ),
            pcaf_data_quality=result.get("pcaf_data_quality", 3),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_equity: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in calculate_equity: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Equity calculation failed",
        )


@router.post(
    "/calculate/private-equity",
    response_model=PrivateEquityResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate private equity financed emissions",
    description=(
        "Calculate financed emissions for a private equity investment. Uses "
        "PCAF attribution: outstanding / total_equity_plus_debt * emissions. "
        "Falls back to sector average if company emissions unavailable."
    ),
)
async def calculate_private_equity(
    request: PrivateEquityRequest,
    service=Depends(get_service),
) -> PrivateEquityResponse:
    """
    Calculate private equity financed emissions.

    Args:
        request: Private equity request with equity contribution and company data
        service: InvestmentsService instance

    Returns:
        PrivateEquityResponse with attribution and financed emissions

    Raises:
        HTTPException: 400 for validation, 500 for failures
    """
    try:
        logger.info(
            f"Calculating private equity: company={request.company_name}, "
            f"outstanding={request.outstanding_amount}, "
            f"total_eq_debt={request.total_equity_plus_debt}"
        )

        result = await service.calculate_private_equity(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return PrivateEquityResponse(
            calculation_id=calculation_id,
            asset_class="private_equity",
            company_name=result.get("company_name", request.company_name),
            method=result.get("method", "pcaf_attribution"),
            attribution_factor=result.get("attribution_factor", 0.0),
            financed_emissions_tco2e=result.get(
                "financed_emissions_tco2e", 0.0
            ),
            carbon_intensity_tco2e_per_m_invested=result.get(
                "carbon_intensity_tco2e_per_m_invested", 0.0
            ),
            pcaf_data_quality=result.get("pcaf_data_quality", 4),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_private_equity: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_private_equity: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Private equity calculation failed",
        )


@router.post(
    "/calculate/corporate-bond",
    response_model=CorporateBondResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate corporate bond financed emissions",
    description=(
        "Calculate financed emissions for a corporate bond holding. Uses "
        "PCAF attribution: outstanding / EVIC * issuer_emissions. Bond "
        "holders share EVIC denominator with equity holders."
    ),
)
async def calculate_corporate_bond(
    request: CorporateBondRequest,
    service=Depends(get_service),
) -> CorporateBondResponse:
    """
    Calculate corporate bond financed emissions.

    Args:
        request: Corporate bond request with EVIC and issuer emissions
        service: InvestmentsService instance

    Returns:
        CorporateBondResponse with attribution and financed emissions

    Raises:
        HTTPException: 400 for validation, 500 for failures
    """
    try:
        logger.info(
            f"Calculating corporate bond: issuer={request.company_name}, "
            f"outstanding={request.outstanding_amount}, EVIC={request.evic}"
        )

        result = await service.calculate_corporate_bond(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CorporateBondResponse(
            calculation_id=calculation_id,
            asset_class="corporate_bond",
            company_name=result.get("company_name", request.company_name),
            method=result.get("method", "pcaf_attribution"),
            attribution_factor=result.get("attribution_factor", 0.0),
            financed_emissions_tco2e=result.get(
                "financed_emissions_tco2e", 0.0
            ),
            carbon_intensity_tco2e_per_m_invested=result.get(
                "carbon_intensity_tco2e_per_m_invested", 0.0
            ),
            pcaf_data_quality=result.get("pcaf_data_quality", 3),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_corporate_bond: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_corporate_bond: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Corporate bond calculation failed",
        )


@router.post(
    "/calculate/project-finance",
    response_model=ProjectFinanceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate project finance financed emissions",
    description=(
        "Calculate financed emissions for a project finance position. Uses "
        "PCAF attribution: outstanding / total_project_cost * project_emissions. "
        "Supports infrastructure, renewable, and industrial projects with "
        "optional lifetime emissions projection."
    ),
)
async def calculate_project_finance(
    request: ProjectFinanceRequest,
    service=Depends(get_service),
) -> ProjectFinanceResponse:
    """
    Calculate project finance financed emissions.

    Args:
        request: Project finance request with project cost and emissions
        service: InvestmentsService instance

    Returns:
        ProjectFinanceResponse with attribution and financed emissions

    Raises:
        HTTPException: 400 for validation, 500 for failures
    """
    try:
        logger.info(
            f"Calculating project finance: project={request.project_name}, "
            f"outstanding={request.outstanding_amount}, "
            f"total_cost={request.total_project_cost}"
        )

        result = await service.calculate_project_finance(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return ProjectFinanceResponse(
            calculation_id=calculation_id,
            asset_class="project_finance",
            project_name=result.get("project_name", request.project_name),
            method=result.get("method", "pcaf_attribution"),
            attribution_factor=result.get("attribution_factor", 0.0),
            financed_emissions_tco2e=result.get(
                "financed_emissions_tco2e", 0.0
            ),
            lifetime_financed_emissions_tco2e=result.get(
                "lifetime_financed_emissions_tco2e"
            ),
            carbon_intensity_tco2e_per_m_invested=result.get(
                "carbon_intensity_tco2e_per_m_invested", 0.0
            ),
            pcaf_data_quality=result.get("pcaf_data_quality", 3),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_project_finance: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_project_finance: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Project finance calculation failed",
        )


@router.post(
    "/calculate/commercial-real-estate",
    response_model=CREResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate commercial real estate financed emissions",
    description=(
        "Calculate financed emissions for a CRE investment. Uses building "
        "energy performance: EUI * floor_area * grid_EF for building emissions, "
        "then applies PCAF attribution: outstanding / property_value. "
        "Supports actual energy data (PCAF 1-2) or benchmark EUI (PCAF 3-5)."
    ),
)
async def calculate_cre(
    request: CRERequest,
    service=Depends(get_service),
) -> CREResponse:
    """
    Calculate commercial real estate financed emissions.

    Args:
        request: CRE request with property data and energy performance
        service: InvestmentsService instance

    Returns:
        CREResponse with building emissions and attribution

    Raises:
        HTTPException: 400 for missing property data, 500 for failures
    """
    try:
        logger.info(
            f"Calculating CRE financed emissions: type={request.property_type}, "
            f"area={request.floor_area_m2}m2, "
            f"outstanding={request.outstanding_amount}"
        )

        result = await service.calculate_cre(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CREResponse(
            calculation_id=calculation_id,
            asset_class="commercial_real_estate",
            property_type=result.get("property_type", request.property_type),
            method=result.get("method", "building_eui"),
            attribution_factor=result.get("attribution_factor", 0.0),
            building_emissions_tco2e=result.get(
                "building_emissions_tco2e", 0.0
            ),
            financed_emissions_tco2e=result.get(
                "financed_emissions_tco2e", 0.0
            ),
            eui_kwh_per_m2=result.get("eui_kwh_per_m2"),
            carbon_intensity_kgco2e_per_m2=result.get(
                "carbon_intensity_kgco2e_per_m2"
            ),
            pcaf_data_quality=result.get("pcaf_data_quality", 3),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_cre: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in calculate_cre: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CRE calculation failed",
        )


@router.post(
    "/calculate/mortgage",
    response_model=MortgageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate mortgage financed emissions",
    description=(
        "Calculate financed emissions for a residential mortgage. Uses PCAF "
        "attribution: outstanding_loan / property_value * building_emissions. "
        "Building emissions derived from EPC rating, actual energy data, or "
        "residential benchmarks. Supports single-family, townhouse, apartment, "
        "and condo property types."
    ),
)
async def calculate_mortgage(
    request: MortgageRequest,
    service=Depends(get_service),
) -> MortgageResponse:
    """
    Calculate mortgage financed emissions.

    Args:
        request: Mortgage request with loan, property, and energy data
        service: InvestmentsService instance

    Returns:
        MortgageResponse with building emissions and attribution

    Raises:
        HTTPException: 400 for missing data, 500 for failures
    """
    try:
        logger.info(
            f"Calculating mortgage financed emissions: "
            f"type={request.property_type}, "
            f"loan={request.outstanding_loan}, "
            f"value={request.property_value}"
        )

        result = await service.calculate_mortgage(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return MortgageResponse(
            calculation_id=calculation_id,
            asset_class="mortgage",
            property_type=result.get("property_type", request.property_type),
            method=result.get("method", "building_eui"),
            attribution_factor=result.get("attribution_factor", 0.0),
            building_emissions_tco2e=result.get(
                "building_emissions_tco2e", 0.0
            ),
            financed_emissions_tco2e=result.get(
                "financed_emissions_tco2e", 0.0
            ),
            eui_kwh_per_m2=result.get("eui_kwh_per_m2"),
            epc_rating=result.get("epc_rating"),
            pcaf_data_quality=result.get("pcaf_data_quality", 3),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_mortgage: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in calculate_mortgage: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Mortgage calculation failed",
        )


@router.post(
    "/calculate/motor-vehicle-loan",
    response_model=MotorVehicleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate motor vehicle loan financed emissions",
    description=(
        "Calculate financed emissions for a motor vehicle loan. Uses PCAF "
        "attribution: outstanding_loan / vehicle_value * vehicle_emissions. "
        "Vehicle emissions from make/model lookup or category average EFs. "
        "Supports 11 vehicle categories including EV and hybrid."
    ),
)
async def calculate_motor_vehicle(
    request: MotorVehicleRequest,
    service=Depends(get_service),
) -> MotorVehicleResponse:
    """
    Calculate motor vehicle loan financed emissions.

    Args:
        request: Auto loan request with vehicle data
        service: InvestmentsService instance

    Returns:
        MotorVehicleResponse with vehicle emissions and attribution

    Raises:
        HTTPException: 400 for invalid vehicle data, 500 for failures
    """
    try:
        logger.info(
            f"Calculating motor vehicle financed emissions: "
            f"category={request.vehicle_category}, "
            f"loan={request.outstanding_loan}, "
            f"value={request.vehicle_value}"
        )

        result = await service.calculate_motor_vehicle(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return MotorVehicleResponse(
            calculation_id=calculation_id,
            asset_class="motor_vehicle_loan",
            vehicle_category=result.get(
                "vehicle_category", request.vehicle_category
            ),
            method=result.get("method", "vehicle_specific"),
            attribution_factor=result.get("attribution_factor", 0.0),
            vehicle_emissions_tco2e=result.get(
                "vehicle_emissions_tco2e", 0.0
            ),
            financed_emissions_tco2e=result.get(
                "financed_emissions_tco2e", 0.0
            ),
            ef_kgco2e_per_km=result.get("ef_kgco2e_per_km"),
            pcaf_data_quality=result.get("pcaf_data_quality", 3),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_motor_vehicle: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_motor_vehicle: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Motor vehicle loan calculation failed",
        )


@router.post(
    "/calculate/sovereign-bond",
    response_model=SovereignBondResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate sovereign bond financed emissions",
    description=(
        "Calculate financed emissions for a sovereign bond. Uses PCAF "
        "sovereign method: outstanding / GDP_PPP * country_emissions. "
        "Supports 50+ countries with UNFCCC national inventory data. "
        "Optional inclusion of LULUCF emissions."
    ),
)
async def calculate_sovereign_bond(
    request: SovereignBondRequest,
    service=Depends(get_service),
) -> SovereignBondResponse:
    """
    Calculate sovereign bond financed emissions.

    Args:
        request: Sovereign bond request with country and GDP data
        service: InvestmentsService instance

    Returns:
        SovereignBondResponse with attribution and financed emissions

    Raises:
        HTTPException: 400 for unknown country, 500 for failures
    """
    try:
        logger.info(
            f"Calculating sovereign bond financed emissions: "
            f"country={request.country_code}, "
            f"outstanding={request.outstanding_amount}"
        )

        result = await service.calculate_sovereign_bond(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return SovereignBondResponse(
            calculation_id=calculation_id,
            asset_class="sovereign_bond",
            country_code=result.get("country_code", request.country_code),
            country_name=result.get("country_name"),
            method=result.get("method", "sovereign_ppp"),
            attribution_factor=result.get("attribution_factor", 0.0),
            financed_emissions_tco2e=result.get(
                "financed_emissions_tco2e", 0.0
            ),
            per_capita_tco2e=result.get("per_capita_tco2e"),
            pcaf_data_quality=result.get("pcaf_data_quality", 2),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_sovereign_bond: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_sovereign_bond: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sovereign bond calculation failed",
        )


@router.post(
    "/calculate/batch",
    response_model=BatchCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch calculate financed emissions (up to 50,000 positions)",
    description=(
        "Batch calculate financed emissions for mixed asset class positions. "
        "Processes up to 50,000 positions in a single request. Routes each "
        "position to the appropriate PCAF asset class calculator. Returns "
        "per-position results with batch-level summary and error reporting."
    ),
)
async def calculate_batch(
    request: BatchRequest,
    service=Depends(get_service),
) -> BatchCalculateResponse:
    """
    Batch calculate financed emissions for mixed positions.

    Args:
        request: Batch request with mixed asset class positions
        service: InvestmentsService instance

    Returns:
        BatchCalculateResponse with per-position results and summary

    Raises:
        HTTPException: 400 for empty batch, 422 for schema errors, 500 for failures
    """
    try:
        start_time = datetime.utcnow()
        logger.info(
            f"Batch calculating financed emissions: "
            f"{len(request.investments)} positions, "
            f"fail_on_error={request.fail_on_error}"
        )

        result = await service.calculate_batch(request.dict())
        processing_time = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        return BatchCalculateResponse(
            batch_id=result.get("batch_id", str(uuid.uuid4())),
            results=result.get("results", []),
            total_financed_emissions_tco2e=result.get(
                "total_financed_emissions_tco2e", 0.0
            ),
            position_count=result.get(
                "position_count", len(request.investments)
            ),
            error_count=result.get("error_count", 0),
            errors=result.get("errors", []),
            by_asset_class=result.get("by_asset_class", {}),
            processing_time_ms=result.get(
                "processing_time_ms", processing_time
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_batch: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in calculate_batch: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch calculation failed",
        )


@router.post(
    "/calculate/portfolio",
    response_model=PortfolioResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Full portfolio analysis with WACI",
    description=(
        "Perform comprehensive portfolio analysis including WACI (Weighted "
        "Average Carbon Intensity), carbon footprint, total carbon emissions, "
        "PCAF data quality scoring, and multi-dimensional breakdowns by asset "
        "class, GICS sector, and country. Supports equity share, financial "
        "control, and operational control consolidation approaches."
    ),
)
async def calculate_portfolio(
    request: PortfolioRequest,
    service=Depends(get_service),
) -> PortfolioResponse:
    """
    Full portfolio analysis with WACI and multi-dimensional breakdowns.

    Args:
        request: Portfolio request with positions and AUM
        service: InvestmentsService instance

    Returns:
        PortfolioResponse with WACI, carbon footprint, and breakdowns

    Raises:
        HTTPException: 400 for validation, 500 for processing failures
    """
    try:
        logger.info(
            f"Full portfolio analysis: {len(request.investments)} positions, "
            f"AUM={request.total_aum}, "
            f"approach={request.consolidation_approach}"
        )

        result = await service.calculate_portfolio(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return PortfolioResponse(
            calculation_id=calculation_id,
            portfolio_name=result.get("portfolio_name", request.portfolio_name),
            total_aum=result.get("total_aum", request.total_aum),
            total_financed_emissions_tco2e=result.get(
                "total_financed_emissions_tco2e", 0.0
            ),
            waci=result.get("waci", 0.0),
            carbon_footprint_tco2e_per_m_invested=result.get(
                "carbon_footprint_tco2e_per_m_invested", 0.0
            ),
            total_carbon_emissions_tco2e=result.get(
                "total_carbon_emissions_tco2e", 0.0
            ),
            position_count=result.get(
                "position_count", len(request.investments)
            ),
            coverage_ratio=result.get("coverage_ratio", 0.0),
            by_asset_class=result.get("by_asset_class", {}),
            by_sector=result.get("by_sector", {}),
            by_country=result.get("by_country", {}),
            weighted_pcaf_score=result.get("weighted_pcaf_score", 3.0),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_portfolio: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in calculate_portfolio: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Portfolio analysis failed",
        )


# ============================================================================
# ENDPOINTS - COMPLIANCE (1 POST)
# ============================================================================


@router.post(
    "/compliance/check",
    response_model=ComplianceCheckResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Multi-framework compliance check",
    description=(
        "Check financed emissions calculation against up to 9 frameworks: "
        "GHG Protocol Scope 3, PCAF Global Standard, TCFD Recommendations, "
        "NZBA Guidelines, CSRD ESRS E1, CDP Climate, SBTi Financial Sector, "
        "ISO 14064-1, and GRI 305. Returns per-framework pass/fail status, "
        "compliance score, findings, and improvement recommendations."
    ),
)
async def check_compliance(
    request: ComplianceCheckRequest,
    service=Depends(get_service),
) -> ComplianceCheckResponse:
    """
    Check financed emissions against multiple compliance frameworks.

    Args:
        request: Compliance check request with calculation ID and frameworks
        service: InvestmentsService instance

    Returns:
        ComplianceCheckResponse with per-framework results

    Raises:
        HTTPException: 404 if calculation not found, 500 for failures
    """
    try:
        logger.info(
            f"Checking compliance: calculation={request.calculation_id}, "
            f"frameworks={request.frameworks}"
        )

        result = await service.check_compliance(request.dict())

        return ComplianceCheckResponse(
            results=result.get("results", []),
            overall_status=result.get("overall_status", "warning"),
            overall_score=result.get("overall_score", 0.0),
            frameworks_checked=result.get(
                "frameworks_checked", len(request.frameworks)
            ),
            recommendations=result.get("recommendations", []),
        )

    except ValueError as e:
        logger.error("Validation error in check_compliance: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except LookupError as e:
        logger.error("Calculation not found in check_compliance: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in check_compliance: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance check failed",
        )


# ============================================================================
# ENDPOINTS - DATA RETRIEVAL (12 GET + 1 DELETE)
# ============================================================================


@router.get(
    "/calculations/{calculation_id}",
    response_model=CalculationDetailResponse,
    summary="Get calculation by ID",
    description=(
        "Retrieve a specific financed emissions calculation by its UUID. "
        "Returns full calculation details including attribution factors, "
        "PCAF data quality, and provenance hash."
    ),
)
async def get_calculation(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> CalculationDetailResponse:
    """
    Get a specific calculation by ID.

    Args:
        calculation_id: Calculation UUID
        service: InvestmentsService instance

    Returns:
        CalculationDetailResponse with full calculation details

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info("Getting calculation: %s", calculation_id)

        result = await service.get_calculation(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found",
            )

        return CalculationDetailResponse(
            calculation_id=result.get("calculation_id", calculation_id),
            asset_class=result.get("asset_class"),
            method=result.get("method", "unknown"),
            total_financed_emissions_kgco2e=result.get(
                "total_financed_emissions_kgco2e", 0.0
            ),
            pcaf_data_quality=result.get("pcaf_data_quality"),
            attribution_factor=result.get("attribution_factor"),
            details=result.get("details", {}),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_calculation: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve calculation",
        )


@router.get(
    "/calculations",
    response_model=CalculationListResponse,
    summary="List calculations (paginated)",
    description=(
        "List financed emissions calculations with pagination and filtering. "
        "Filter by asset class, GICS sector, calculation method, date range, "
        "and PCAF data quality score. Returns paginated calculation summaries."
    ),
)
async def list_calculations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    asset_class: Optional[str] = Query(
        None, description="Filter by PCAF asset class"
    ),
    sector: Optional[str] = Query(
        None, description="Filter by GICS sector"
    ),
    method: Optional[str] = Query(
        None, description="Filter by calculation method"
    ),
    from_date: Optional[str] = Query(
        None, description="From date (ISO 8601)"
    ),
    to_date: Optional[str] = Query(
        None, description="To date (ISO 8601)"
    ),
    min_pcaf_quality: Optional[int] = Query(
        None, ge=1, le=5, description="Minimum PCAF data quality score"
    ),
    service=Depends(get_service),
) -> CalculationListResponse:
    """
    List calculations with pagination and filters.

    Args:
        page: Page number (1-based)
        page_size: Results per page (1-100)
        asset_class: Optional asset class filter
        sector: Optional sector filter
        method: Optional method filter
        from_date: Optional start date filter
        to_date: Optional end date filter
        min_pcaf_quality: Optional minimum PCAF score filter
        service: InvestmentsService instance

    Returns:
        CalculationListResponse with paginated results

    Raises:
        HTTPException: 400 for invalid filters, 500 for failures
    """
    try:
        logger.info(
            f"Listing calculations: page={page}, size={page_size}, "
            f"asset_class={asset_class}, sector={sector}, method={method}"
        )

        filters = {
            "page": page,
            "page_size": page_size,
            "asset_class": asset_class,
            "sector": sector,
            "method": method,
            "from_date": from_date,
            "to_date": to_date,
            "min_pcaf_quality": min_pcaf_quality,
        }

        result = await service.list_calculations(filters)

        return CalculationListResponse(
            calculations=result.get("calculations", []),
            count=result.get("count", 0),
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error("Error in list_calculations: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list calculations",
        )


@router.delete(
    "/calculations/{calculation_id}",
    response_model=DeleteResponse,
    summary="Soft-delete a calculation",
    description=(
        "Soft-delete a financed emissions calculation by UUID. Sets the "
        "is_deleted flag to TRUE without removing the underlying data. "
        "Calculation will be excluded from future queries."
    ),
)
async def delete_calculation(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> DeleteResponse:
    """
    Soft-delete a calculation by ID.

    Args:
        calculation_id: Calculation UUID to delete
        service: InvestmentsService instance

    Returns:
        DeleteResponse with deletion status

    Raises:
        HTTPException: 404 if not found, 500 for failures
    """
    try:
        logger.info("Soft-deleting calculation: %s", calculation_id)

        result = await service.delete_calculation(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found",
            )

        return DeleteResponse(
            calculation_id=calculation_id,
            deleted=result.get("deleted", True),
            message=result.get(
                "message", f"Calculation {calculation_id} soft-deleted"
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in delete_calculation: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete calculation",
        )


@router.get(
    "/emission-factors/{asset_class}",
    response_model=EmissionFactorListResponse,
    summary="Get emission factors by asset class",
    description=(
        "Retrieve emission factors applicable to a specific PCAF asset class. "
        "Returns sector EFs for equity/bonds, building EUI for CRE/mortgage, "
        "vehicle EFs for auto loans, and sovereign EFs for government bonds."
    ),
)
async def get_emission_factors(
    asset_class: str = Path(
        ..., description="PCAF asset class"
    ),
    country: Optional[str] = Query(
        None, description="Filter by country code"
    ),
    sector: Optional[str] = Query(
        None, description="Filter by GICS sector"
    ),
    service=Depends(get_service),
) -> EmissionFactorListResponse:
    """
    Get emission factors for a specific PCAF asset class.

    Args:
        asset_class: PCAF asset class identifier
        country: Optional country filter
        sector: Optional sector filter
        service: InvestmentsService instance

    Returns:
        EmissionFactorListResponse with applicable factors

    Raises:
        HTTPException: 400 for invalid asset class, 500 for failures
    """
    try:
        valid_classes = {e.value for e in AssetClass}
        if asset_class not in valid_classes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid asset class '{asset_class}'. "
                    f"Must be one of: {', '.join(sorted(valid_classes))}"
                ),
            )

        logger.info(
            f"Getting emission factors: asset_class={asset_class}, "
            f"country={country}, sector={sector}"
        )

        filters = {
            "asset_class": asset_class,
            "country": country,
            "sector": sector,
        }

        result = await service.get_emission_factors(filters)

        factors = [
            EmissionFactorResponse(**f) for f in result.get("factors", [])
        ]

        return EmissionFactorListResponse(
            factors=factors,
            count=len(factors),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_emission_factors: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve emission factors",
        )


@router.get(
    "/sector-factors",
    response_model=SectorFactorListResponse,
    summary="Get sector emission factors (12 GICS sectors)",
    description=(
        "Retrieve emission intensity factors for all 12 GICS sectors. "
        "Returns tCO2e per million USD revenue for sector-average "
        "estimation of financed emissions when company-specific data "
        "is unavailable (PCAF data quality score 4-5)."
    ),
)
async def get_sector_factors(
    sector: Optional[str] = Query(
        None, description="Filter by specific GICS sector"
    ),
    source: Optional[str] = Query(
        None, description="Filter by data source"
    ),
    service=Depends(get_service),
) -> SectorFactorListResponse:
    """
    Get sector emission factors for all GICS sectors.

    Args:
        sector: Optional sector filter
        source: Optional data source filter
        service: InvestmentsService instance

    Returns:
        SectorFactorListResponse with sector factors

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting sector factors: sector={sector}, source={source}"
        )

        filters = {"sector": sector, "source": source}
        result = await service.get_sector_factors(filters)

        factors = [
            SectorFactorResponse(**f) for f in result.get("factors", [])
        ]

        return SectorFactorListResponse(
            factors=factors,
            count=len(factors),
        )

    except Exception as e:
        logger.error("Error in get_sector_factors: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sector factors",
        )


@router.get(
    "/country-factors",
    response_model=CountryFactorListResponse,
    summary="Get country emission factors (50+ countries)",
    description=(
        "Retrieve national GHG emission data for sovereign bond attribution. "
        "Includes total GHG emissions (MtCO2e), GDP PPP (billion USD), "
        "per-capita emissions, and LULUCF figures for 50+ countries sourced "
        "from UNFCCC, World Bank, and national inventories."
    ),
)
async def get_country_factors(
    country: Optional[str] = Query(
        None, description="Filter by ISO alpha-3 country code"
    ),
    region: Optional[str] = Query(
        None, description="Filter by world region"
    ),
    service=Depends(get_service),
) -> CountryFactorListResponse:
    """
    Get country emission factors for sovereign bond calculations.

    Args:
        country: Optional country code filter
        region: Optional world region filter
        service: InvestmentsService instance

    Returns:
        CountryFactorListResponse with country emission data

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting country factors: country={country}, region={region}"
        )

        filters = {"country": country, "region": region}
        result = await service.get_country_factors(filters)

        factors = [
            CountryFactorResponse(**f) for f in result.get("factors", [])
        ]

        return CountryFactorListResponse(
            factors=factors,
            count=len(factors),
        )

    except Exception as e:
        logger.error("Error in get_country_factors: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve country factors",
        )


@router.get(
    "/pcaf-quality",
    response_model=PCAFQualityListResponse,
    summary="Get PCAF data quality criteria by asset class",
    description=(
        "Retrieve PCAF data quality scoring criteria for each asset class. "
        "Shows quality scores 1 (highest: actual reported emissions) through "
        "5 (lowest: sector average estimates) with descriptions and typical "
        "uncertainty percentages per PCAF Global Standard."
    ),
)
async def get_pcaf_quality(
    asset_class: Optional[str] = Query(
        None, description="Filter by PCAF asset class"
    ),
    service=Depends(get_service),
) -> PCAFQualityListResponse:
    """
    Get PCAF data quality criteria for asset classes.

    Args:
        asset_class: Optional asset class filter
        service: InvestmentsService instance

    Returns:
        PCAFQualityListResponse with quality criteria

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info("Getting PCAF quality criteria: asset_class=%s", asset_class)

        filters = {"asset_class": asset_class}
        result = await service.get_pcaf_quality(filters)

        criteria = [
            PCAFQualityCriteria(**c) for c in result.get("criteria", [])
        ]

        return PCAFQualityListResponse(
            criteria=criteria,
            count=len(criteria),
        )

    except Exception as e:
        logger.error("Error in get_pcaf_quality: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve PCAF quality criteria",
        )


@router.get(
    "/carbon-intensity",
    response_model=CarbonIntensityResponse,
    summary="Get WACI and portfolio carbon intensity",
    description=(
        "Retrieve the latest portfolio carbon intensity metrics including "
        "WACI (tCO2e per M USD revenue), carbon footprint (tCO2e per M USD "
        "invested), total financed emissions, and physical intensity. "
        "Includes breakdown by GICS sector with optional benchmark comparison."
    ),
)
async def get_carbon_intensity(
    portfolio_id: Optional[str] = Query(
        None, description="Portfolio identifier"
    ),
    reporting_period: Optional[str] = Query(
        None, description="Reporting period (e.g., 2024-Q4)"
    ),
    service=Depends(get_service),
) -> CarbonIntensityResponse:
    """
    Get WACI and portfolio carbon intensity metrics.

    Args:
        portfolio_id: Optional portfolio ID filter
        reporting_period: Optional reporting period filter
        service: InvestmentsService instance

    Returns:
        CarbonIntensityResponse with WACI and intensity metrics

    Raises:
        HTTPException: 404 if no data, 500 for failures
    """
    try:
        logger.info(
            f"Getting carbon intensity: portfolio={portfolio_id}, "
            f"period={reporting_period}"
        )

        filters = {
            "portfolio_id": portfolio_id,
            "reporting_period": reporting_period,
        }

        result = await service.get_carbon_intensity(filters)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No carbon intensity data found for the given filters",
            )

        return CarbonIntensityResponse(
            waci=result.get("waci", 0.0),
            carbon_footprint=result.get("carbon_footprint", 0.0),
            financed_emissions=result.get("financed_emissions", 0.0),
            physical_intensity=result.get("physical_intensity"),
            by_sector=result.get("by_sector", {}),
            benchmark_comparison=result.get("benchmark_comparison"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_carbon_intensity: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve carbon intensity",
        )


@router.get(
    "/portfolio-alignment",
    response_model=PortfolioAlignmentResponse,
    summary="Get temperature alignment and SBTi status",
    description=(
        "Retrieve portfolio temperature alignment assessment including "
        "implied temperature rise, SBTi commitment and validation status, "
        "NZBA alignment percentage, sector-level alignment breakdown, and "
        "emission reduction trajectory. Based on SBTi Financial Sector "
        "Guidance and TCFD recommendations."
    ),
)
async def get_portfolio_alignment(
    portfolio_id: Optional[str] = Query(
        None, description="Portfolio identifier"
    ),
    target_year: Optional[int] = Query(
        None, ge=2025, le=2100, description="Target year for trajectory"
    ),
    service=Depends(get_service),
) -> PortfolioAlignmentResponse:
    """
    Get portfolio temperature alignment and SBTi status.

    Args:
        portfolio_id: Optional portfolio ID filter
        target_year: Optional target year for trajectory projection
        service: InvestmentsService instance

    Returns:
        PortfolioAlignmentResponse with alignment metrics

    Raises:
        HTTPException: 404 if no data, 500 for failures
    """
    try:
        logger.info(
            f"Getting portfolio alignment: portfolio={portfolio_id}, "
            f"target_year={target_year}"
        )

        filters = {
            "portfolio_id": portfolio_id,
            "target_year": target_year,
        }

        result = await service.get_portfolio_alignment(filters)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No alignment data found for the given filters",
            )

        return PortfolioAlignmentResponse(
            implied_temperature_rise_c=result.get(
                "implied_temperature_rise_c", 2.5
            ),
            alignment_status=result.get("alignment_status", "not_aligned"),
            sbti_committed_pct=result.get("sbti_committed_pct", 0.0),
            sbti_validated_pct=result.get("sbti_validated_pct", 0.0),
            nzba_aligned_pct=result.get("nzba_aligned_pct", 0.0),
            by_sector=result.get("by_sector", {}),
            trajectory=result.get("trajectory"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in get_portfolio_alignment: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve portfolio alignment",
        )


@router.get(
    "/aggregations",
    response_model=AggregationResponse,
    summary="Get time-series aggregations",
    description=(
        "Retrieve time-series aggregated financed emissions by period. "
        "Supports daily, weekly, monthly, quarterly, and annual periods. "
        "Includes breakdowns by PCAF asset class, GICS sector, and country."
    ),
)
async def get_aggregations(
    period: str = Query(
        "monthly",
        description="Aggregation period (daily, weekly, monthly, quarterly, annual)",
    ),
    from_date: Optional[str] = Query(
        None, description="Start date (ISO 8601)"
    ),
    to_date: Optional[str] = Query(
        None, description="End date (ISO 8601)"
    ),
    asset_class: Optional[str] = Query(
        None, description="Filter by PCAF asset class"
    ),
    service=Depends(get_service),
) -> AggregationResponse:
    """
    Get time-series aggregated financed emissions.

    Args:
        period: Aggregation period type
        from_date: Optional start date filter
        to_date: Optional end date filter
        asset_class: Optional asset class filter
        service: InvestmentsService instance

    Returns:
        AggregationResponse with aggregated emissions data

    Raises:
        HTTPException: 400 for invalid period, 500 for failures
    """
    try:
        valid_periods = {"daily", "weekly", "monthly", "quarterly", "annual"}
        if period not in valid_periods:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid period '{period}'. "
                    f"Must be one of: {', '.join(sorted(valid_periods))}"
                ),
            )

        logger.info(
            f"Getting aggregations: period={period}, from={from_date}, "
            f"to={to_date}, asset_class={asset_class}"
        )

        filters = {
            "period": period,
            "from_date": from_date,
            "to_date": to_date,
            "asset_class": asset_class,
        }

        result = await service.get_aggregations(filters)

        return AggregationResponse(
            period=period,
            total_financed_emissions_tco2e=result.get(
                "total_financed_emissions_tco2e", 0.0
            ),
            by_asset_class=result.get("by_asset_class", {}),
            by_sector=result.get("by_sector", {}),
            by_country=result.get("by_country", {}),
            position_count=result.get("position_count", 0),
            total_aum=result.get("total_aum"),
            waci=result.get("waci"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_aggregations: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Aggregation failed",
        )


@router.get(
    "/provenance/{calculation_id}",
    response_model=ProvenanceResponse,
    summary="Get provenance chain",
    description=(
        "Retrieve the complete SHA-256 provenance chain for a financed "
        "emissions calculation. Includes all pipeline stages (validate, "
        "classify, attribute, calculate, aggregate, compliance, quality, "
        "seal) with per-stage hashes and chain integrity verification."
    ),
)
async def get_provenance(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> ProvenanceResponse:
    """
    Get provenance chain for a specific calculation.

    Args:
        calculation_id: Calculation UUID
        service: InvestmentsService instance

    Returns:
        ProvenanceResponse with chain stages and verification status

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info("Getting provenance for calculation: %s", calculation_id)

        result = await service.get_provenance(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provenance for calculation {calculation_id} not found",
            )

        return ProvenanceResponse(
            calculation_id=result.get("calculation_id", calculation_id),
            chain=result.get("chain", []),
            is_valid=result.get("is_valid", False),
            root_hash=result.get("root_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_provenance: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve provenance",
        )


# ============================================================================
# ENDPOINTS - HEALTH (1 GET)
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check with 7 engine statuses",
    description=(
        "Health check endpoint for the Investments Agent. Returns service "
        "status, agent identifier, version, uptime, and per-engine health "
        "status for all 7 calculation engines (InvestmentDatabaseEngine, "
        "EquityInvestmentCalculatorEngine, DebtInvestmentCalculatorEngine, "
        "RealAssetCalculatorEngine, SovereignBondCalculatorEngine, "
        "ComplianceCheckerEngine, InvestmentsPipelineEngine). "
        "No authentication required."
    ),
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint (no auth required).

    Returns:
        HealthResponse with service status, engine health, and data counts
    """
    try:
        uptime = (datetime.utcnow() - _start_time).total_seconds()

        engines = []
        for engine_name in _ENGINE_NAMES:
            engines.append(
                EngineStatus(
                    engine_name=engine_name,
                    status="healthy",
                    last_used=None,
                )
            )

        return HealthResponse(
            status="healthy",
            agent_id="GL-MRV-S3-015",
            version="1.0.0",
            uptime_seconds=round(uptime, 2),
            engines=engines,
            asset_classes_loaded=len(AssetClass),
            emission_factors_loaded=210,
        )

    except Exception as e:
        logger.error("Error in health_check: %s", e, exc_info=True)
        return HealthResponse(
            status="unhealthy",
            agent_id="GL-MRV-S3-015",
            version="1.0.0",
            uptime_seconds=0.0,
            engines=[],
            asset_classes_loaded=0,
            emission_factors_loaded=0,
        )
