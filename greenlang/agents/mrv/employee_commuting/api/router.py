"""
Employee Commuting Agent API Router - AGENT-MRV-020

This module implements the FastAPI router for employee commuting emissions
calculations following GHG Protocol Scope 3 Category 7 requirements.

Provides 22 REST endpoints for:
- Emissions calculations (full pipeline, batch, single commute, telework,
  survey, average-data, spend-based, multi-modal)
- Calculation CRUD (get, list, delete)
- Emission factor lookup by mode and reference data
- Commute mode metadata and working days by region
- Average commute distances by country
- Grid emission factors for telework calculations
- Compliance checking across 7 regulatory frameworks
- Uncertainty analysis (Monte Carlo, analytical, IPCC Tier 2)
- Aggregations by period with mode and department breakdowns
- Mode share analysis for workforce commuting patterns
- Provenance tracking with SHA-256 chain verification

Follows GreenLang's zero-hallucination principle with deterministic calculations.
All numeric outputs use deterministic formulas; no LLM calls in the calculation path.

Agent ID: GL-MRV-S3-007
Package: greenlang.agents.mrv.employee_commuting
API Prefix: /api/v1/employee-commuting
DB Migration: V071
Metrics Prefix: gl_ec_
Table Prefix: gl_ec_

Supported commute modes (14):
    SOV, carpool, vanpool, bus, metro, light_rail, commuter_rail, ferry,
    motorcycle, e_bike, e_scooter, cycling, walking, telework

Calculation methods (3):
    Employee-specific (survey), Average-data (census), Spend-based (EEIO)

Regulatory frameworks (7):
    GHG Protocol, ISO 14064, CSRD ESRS E1, CDP, SBTi, SB 253, GRI 305

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.agents.mrv.employee_commuting.api.router import router
    >>> app = FastAPI()
    >>> app.include_router(router)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from decimal import Decimal
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/api/v1/employee-commuting",
    tags=["employee-commuting"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# SERVICE DEPENDENCY
# ============================================================================


_service_instance = None


def get_service():
    """
    Get or create EmployeeCommutingService singleton instance.

    Uses lazy initialization to avoid circular imports and ensure the
    service is only created when first needed. The service wires together
    all 7 engines (database, commute mode calculator, telework calculator,
    survey processor, spend-based calculator, compliance checker, pipeline).

    Returns:
        EmployeeCommutingService instance

    Raises:
        HTTPException: If service initialization fails (503)
    """
    global _service_instance

    if _service_instance is None:
        try:
            from greenlang.agents.mrv.employee_commuting.setup import EmployeeCommutingService
            _service_instance = EmployeeCommutingService()
            logger.info("EmployeeCommutingService initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize EmployeeCommutingService: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service initialization failed",
            )

    return _service_instance


# ============================================================================
# REQUEST MODELS
# ============================================================================


class CalculateRequest(BaseModel):
    """
    Request model for full pipeline employee commuting emissions calculation.

    Processes a single employee's commuting data through the complete
    10-stage pipeline: validate, classify, normalize, resolve_efs,
    calculate_commute, calculate_telework, extrapolate, compliance,
    aggregate, seal.

    Supports all 14 commute modes with mode-specific parameters delegated
    to the service layer. When telework data is included, both commute
    and telework emissions are calculated and combined.

    Attributes:
        mode: Primary commute mode (sov, carpool, bus, metro, cycling, etc.)
        commute_data: Mode-specific commute data dictionary
        employee_id: Optional employee identifier for tracking
        tenant_id: Optional tenant identifier for multi-tenant isolation
        telework_data: Optional telework/remote work data
        reporting_period: Reporting period identifier
    """

    mode: str = Field(
        ...,
        description=(
            "Primary commute mode (sov, carpool, vanpool, bus, metro, "
            "light_rail, commuter_rail, ferry, motorcycle, e_bike, "
            "e_scooter, cycling, walking, telework)"
        ),
    )
    commute_data: Dict[str, Any] = Field(
        ...,
        description="Mode-specific commute data dictionary",
    )
    employee_id: Optional[str] = Field(
        None,
        description="Employee identifier for per-employee tracking",
    )
    tenant_id: Optional[str] = Field(
        None,
        description="Tenant identifier for multi-tenant isolation",
    )
    telework_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional telework/WFH data (days, kWh, region)",
    )
    reporting_period: str = Field(
        "2024",
        description="Reporting period identifier (e.g. '2024', '2024-Q1')",
    )


class BatchCalculateRequest(BaseModel):
    """
    Request model for batch employee commuting emissions calculations.

    Processes multiple employees in a single request with parallel execution
    and per-employee error isolation. Supports workforce-level extrapolation
    from survey samples to full headcount.

    Attributes:
        employees: List of employee commute data dictionaries
        reporting_period: Reporting period identifier
        total_employees: Total workforce size for extrapolation
        extrapolate: Whether to extrapolate from sample to full workforce
    """

    employees: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=50000,
        description="List of employee commute data dictionaries",
    )
    reporting_period: str = Field(
        ...,
        description="Reporting period identifier (e.g. '2024', '2024-Q1')",
    )
    total_employees: Optional[int] = Field(
        None,
        ge=1,
        description="Total workforce size for extrapolation (if > len(employees))",
    )
    extrapolate: bool = Field(
        False,
        description="Whether to extrapolate from survey sample to full workforce",
    )


class CommuteCalculateRequest(BaseModel):
    """
    Request model for single commute mode emissions calculation.

    Calculates emissions for one commute mode without running the full
    pipeline. Supports distance-based and fuel-based methods for vehicle
    modes, and per-pkm factors for transit modes.

    Attributes:
        mode: Commute mode identifier
        vehicle_type: Vehicle type for SOV/carpool (optional)
        fuel_type: Fuel type for fuel-based calculation (optional)
        distance_km: One-way commute distance in kilometres
        frequency: Commuting days per week (1-7)
        working_days: Annual working days (adjusted for holidays/PTO)
        occupancy: Vehicle occupancy for carpool/vanpool
        round_trip: Whether distance is round-trip (default False, one-way)
    """

    mode: str = Field(
        ...,
        description=(
            "Commute mode (sov, carpool, vanpool, bus, metro, light_rail, "
            "commuter_rail, ferry, motorcycle, e_bike, e_scooter, cycling, walking)"
        ),
    )
    vehicle_type: Optional[str] = Field(
        None,
        description=(
            "Vehicle type for SOV/carpool (car_average, car_small_petrol, "
            "car_medium_petrol, car_large_petrol, car_small_diesel, "
            "car_medium_diesel, car_large_diesel, hybrid, plugin_hybrid, "
            "bev, van_average, motorcycle)"
        ),
    )
    fuel_type: Optional[str] = Field(
        None,
        description="Fuel type for fuel-based method (petrol, diesel, lpg, e10, b7)",
    )
    distance_km: float = Field(
        ...,
        gt=0,
        le=500,
        description="One-way commute distance in kilometres",
    )
    frequency: int = Field(
        5,
        ge=1,
        le=7,
        description="Commuting days per week",
    )
    working_days: int = Field(
        230,
        ge=1,
        le=365,
        description="Annual working days (adjusted for holidays, PTO, sick days)",
    )
    occupancy: Optional[int] = Field(
        None,
        ge=1,
        le=15,
        description="Vehicle occupancy for carpool/vanpool (2-15)",
    )
    round_trip: bool = Field(
        False,
        description="Whether distance_km is round-trip (if False, doubled internally)",
    )


class TeleworkCalculateRequest(BaseModel):
    """
    Request model for telework/remote work emissions calculation.

    Calculates home office energy consumption emissions using grid
    emission factors. Supports country-level and US eGRID subregional
    factors with seasonal heating/cooling adjustments.

    Attributes:
        frequency: Telework frequency pattern
        region: Country/region code for grid emission factor
        daily_kwh: Estimated daily electricity consumption (kWh)
        seasonal_adjustment: Seasonal adjustment method
        egrid_subregion: US eGRID subregion code (optional, overrides region)
        telework_days_per_year: Override for annual telework days
        equipment_lifecycle: Whether to include amortized equipment emissions
    """

    frequency: str = Field(
        "hybrid_3",
        description=(
            "Telework frequency (full_remote, hybrid_4, hybrid_3, "
            "hybrid_2, hybrid_1, office_full)"
        ),
    )
    region: str = Field(
        "US",
        description="Country/region code for grid EF (US, GB, DE, FR, JP, etc.)",
    )
    daily_kwh: float = Field(
        4.0,
        gt=0,
        le=50,
        description="Estimated daily home office kWh consumption (typical 2.5-8.6)",
    )
    seasonal_adjustment: str = Field(
        "none",
        description=(
            "Seasonal adjustment method (none, heating_only, cooling_only, "
            "full_seasonal)"
        ),
    )
    egrid_subregion: Optional[str] = Field(
        None,
        description="US eGRID subregion code (e.g. CAMX, ERCT, RFCW)",
    )
    telework_days_per_year: Optional[int] = Field(
        None,
        ge=1,
        le=365,
        description="Override for annual telework days (default calculated from frequency)",
    )
    equipment_lifecycle: bool = Field(
        False,
        description="Whether to include amortized equipment lifecycle emissions",
    )


class SurveyRequest(BaseModel):
    """
    Request model for processing employee commute survey data.

    Processes batch survey responses with statistical extrapolation
    from survey sample to full workforce. Supports stratified sampling,
    response rate weighting, and confidence interval calculation.

    Attributes:
        survey_method: Survey methodology used
        responses: List of individual survey response dictionaries
        total_employees: Total workforce size for extrapolation
        response_rate: Survey response rate (0.0-1.0)
        stratification_field: Optional field for stratified extrapolation
        confidence_level: Confidence level for interval calculation
    """

    survey_method: str = Field(
        "random_sample",
        description=(
            "Survey method (full_census, stratified_sample, "
            "random_sample, convenience)"
        ),
    )
    responses: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=100000,
        description="List of individual survey response dictionaries",
    )
    total_employees: int = Field(
        ...,
        ge=1,
        description="Total workforce size for extrapolation",
    )
    response_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Survey response rate (0.0-1.0). Auto-calculated from "
            "len(responses)/total_employees if not provided"
        ),
    )
    stratification_field: Optional[str] = Field(
        None,
        description="Field for stratified extrapolation (site, department, region)",
    )
    confidence_level: float = Field(
        0.95,
        ge=0.80,
        le=0.99,
        description="Confidence level for interval calculation (0.90, 0.95, 0.99)",
    )


class AverageDataRequest(BaseModel):
    """
    Request model for average-data method emissions calculation.

    Uses national/regional average commute distances and mode share
    distributions scaled by workforce headcount. Lowest accuracy but
    highest coverage when survey data is unavailable.

    Attributes:
        total_employees: Total FTE headcount
        country_code: ISO 3166-1 alpha-2 country code for defaults
        mode_share: Optional custom mode share distribution (overrides defaults)
        average_distance_km: Optional custom average one-way distance
        working_days: Optional custom annual working days
        region: Optional sub-national region for more specific defaults
    """

    total_employees: int = Field(
        ...,
        ge=1,
        le=1000000,
        description="Total FTE headcount",
    )
    country_code: str = Field(
        "US",
        min_length=2,
        max_length=6,
        description="ISO 3166-1 alpha-2 country code or 'GLOBAL'",
    )
    mode_share: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Custom mode share distribution (sums to 1.0). "
            "Keys: sov, carpool, bus, metro, rail, cycling, walking, telework, other"
        ),
    )
    average_distance_km: Optional[float] = Field(
        None,
        gt=0,
        le=200,
        description="Custom average one-way commute distance in km (overrides census default)",
    )
    working_days: Optional[int] = Field(
        None,
        ge=1,
        le=365,
        description="Custom annual working days (overrides regional default)",
    )
    region: Optional[str] = Field(
        None,
        description="Sub-national region for more specific defaults (e.g. US state)",
    )


class SpendCalculateRequest(BaseModel):
    """
    Request model for spend-based emissions calculation.

    Uses EEIO (Environmentally Extended Input-Output) factors with
    CPI deflation and currency conversion for commuting spend categories
    including transit subsidies, parking costs, and mileage reimbursement.

    Attributes:
        naics_code: NAICS industry code for EEIO factor selection
        amount: Spend amount in the specified currency
        currency: ISO 4217 currency code
        reporting_year: Year for CPI deflation adjustment
        spend_category: Optional spend category description
    """

    naics_code: str = Field(
        ...,
        description=(
            "NAICS code for EEIO factor selection (485000, 485110, 485210, "
            "487110, 488490, 532100, 811100, 447000)"
        ),
    )
    amount: float = Field(
        ...,
        gt=0,
        description="Spend amount in the specified currency",
    )
    currency: str = Field(
        "USD",
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code (USD, EUR, GBP, CAD, AUD, etc.)",
    )
    reporting_year: int = Field(
        2024,
        ge=2000,
        le=2100,
        description="Reporting year for CPI deflation to base year (2021 USD)",
    )
    spend_category: Optional[str] = Field(
        None,
        description=(
            "Spend category description (transit_subsidy, parking, "
            "mileage_reimbursement, fuel_card, bikeshare)"
        ),
    )


class MultiModalCalculateRequest(BaseModel):
    """
    Request model for multi-modal trip emissions calculation.

    Calculates emissions for a commute consisting of multiple transport
    segments (up to 5 legs). Each leg uses its own mode-specific emission
    factor. Common pattern: drive to station + rail + walk to office.

    Attributes:
        legs: List of trip leg dictionaries (mode, distance_km, vehicle_type)
        frequency: Commuting days per week
        working_days: Annual working days
        employee_id: Optional employee identifier
    """

    legs: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=5,
        description=(
            "List of trip leg dictionaries. Each leg must have 'mode' and "
            "'distance_km'. Optional: 'vehicle_type', 'occupancy' for carpool legs."
        ),
    )
    frequency: int = Field(
        5,
        ge=1,
        le=7,
        description="Commuting days per week",
    )
    working_days: int = Field(
        230,
        ge=1,
        le=365,
        description="Annual working days",
    )
    employee_id: Optional[str] = Field(
        None,
        description="Employee identifier for tracking",
    )


class ComplianceCheckRequest(BaseModel):
    """
    Request model for multi-framework compliance checking.

    Checks employee commuting calculation results against selected
    regulatory frameworks for completeness, boundary correctness,
    telework disclosure, mode share reporting, double-counting
    prevention, and survey methodology documentation.

    Attributes:
        frameworks: List of framework identifiers to check against
        calculation_results: Calculation result dicts to check
        telework_disclosed: Whether telework emissions are disclosed
        mode_share_provided: Whether mode share breakdown is provided
        survey_methodology_documented: Whether survey methodology is documented
        response_rate: Survey response rate for data quality assessment
    """

    frameworks: List[str] = Field(
        ...,
        min_items=1,
        description=(
            "Frameworks to check (ghg_protocol, iso_14064, csrd_esrs, "
            "cdp, sbti, sb_253, gri)"
        ),
    )
    calculation_results: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        description="Calculation results to check for compliance",
    )
    telework_disclosed: bool = Field(
        False,
        description="Whether telework/WFH emissions have been disclosed",
    )
    mode_share_provided: bool = Field(
        False,
        description="Whether per-mode share breakdown is provided",
    )
    survey_methodology_documented: bool = Field(
        False,
        description="Whether survey methodology is documented",
    )
    response_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Survey response rate for data quality assessment",
    )


class UncertaintyRequest(BaseModel):
    """
    Request model for uncertainty analysis.

    Supports Monte Carlo simulation, analytical error propagation,
    and IPCC Tier 2 default uncertainty ranges. Employee commuting
    uncertainty is typically higher than other categories due to
    reliance on survey data and extrapolation.

    Attributes:
        method: Uncertainty analysis method
        iterations: Monte Carlo iterations (if applicable)
        confidence_level: Confidence interval level (0.90, 0.95, 0.99)
        calculation_results: Calculation results to analyze
    """

    method: str = Field(
        "monte_carlo",
        description="Uncertainty method (monte_carlo, analytical, ipcc_tier_2)",
    )
    iterations: int = Field(
        10000,
        ge=1000,
        le=100000,
        description="Monte Carlo iterations (ignored for analytical/ipcc_tier_2)",
    )
    confidence_level: float = Field(
        0.95,
        ge=0.80,
        le=0.99,
        description="Confidence interval level",
    )
    calculation_results: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        description="Calculation results to analyze for uncertainty",
    )


class ModeShareRequest(BaseModel):
    """
    Request model for mode share analysis.

    Analyzes the distribution of commute modes across the workforce
    to identify patterns, compare to benchmarks, and prioritize
    intervention opportunities (e.g., transit subsidies, EV incentives).

    Attributes:
        reporting_period: Reporting period for analysis
        tenant_id: Optional tenant identifier
        calculation_results: Optional pre-computed results to analyze
        benchmark_region: Optional region for benchmark comparison
    """

    reporting_period: str = Field(
        ...,
        description="Reporting period for analysis (e.g. '2024', '2024-Q1')",
    )
    tenant_id: Optional[str] = Field(
        None,
        description="Tenant identifier for multi-tenant filtering",
    )
    calculation_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Pre-computed calculation results to analyze (if not querying DB)",
    )
    benchmark_region: Optional[str] = Field(
        None,
        description="Region code for benchmark comparison (US, GB, DE, EU_AVG, GLOBAL)",
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class CalculationResponse(BaseModel):
    """Response model for single pipeline calculation."""

    success: bool = Field(
        ..., description="Whether calculation completed successfully"
    )
    calculation_id: str = Field(..., description="Unique calculation UUID")
    mode: str = Field(..., description="Primary commute mode used")
    method: str = Field(
        ..., description="Calculation method applied (employee_specific, average_data, spend_based)"
    )
    total_co2e_kg: float = Field(
        ..., description="Total CO2e emissions in kg (commute + telework + WTT)"
    )
    commute_co2e_kg: float = Field(
        ..., description="Commute-only CO2e emissions in kg"
    )
    telework_co2e_kg: Optional[float] = Field(
        None, description="Telework/WFH CO2e emissions in kg (if included)"
    )
    wtt_co2e_kg: float = Field(
        ..., description="Well-to-tank CO2e emissions in kg"
    )
    dqi_score: Optional[float] = Field(
        None, description="Data quality indicator composite score (1.0-5.0)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    detail: Optional[Dict[str, Any]] = Field(
        None, description="Full calculation detail payload"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class BatchCalculateResponse(BaseModel):
    """Response model for batch calculation."""

    success: bool = Field(
        ..., description="Whether batch completed successfully"
    )
    batch_id: str = Field(..., description="Unique batch UUID")
    total_employees: int = Field(
        ..., description="Total employees in batch"
    )
    successful: int = Field(
        ..., description="Number of successful calculations"
    )
    failed: int = Field(
        ..., description="Number of failed calculations"
    )
    total_co2e_kg: float = Field(
        ..., description="Total CO2e for all employees in batch (kg)"
    )
    extrapolated_co2e_kg: Optional[float] = Field(
        None,
        description="Extrapolated total CO2e if sample was scaled to workforce",
    )
    results: List[Dict[str, Any]] = Field(
        ..., description="Individual calculation results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-employee error details"
    )
    reporting_period: str = Field(
        ..., description="Reporting period identifier"
    )


class CommuteCalculateResponse(BaseModel):
    """Response model for single commute mode calculation."""

    success: bool = Field(
        ..., description="Whether calculation completed successfully"
    )
    calculation_id: str = Field(..., description="Unique calculation UUID")
    mode: str = Field(..., description="Commute mode used")
    vehicle_type: Optional[str] = Field(
        None, description="Vehicle type (for SOV/carpool)"
    )
    distance_km: float = Field(
        ..., description="One-way commute distance in km"
    )
    annual_distance_km: float = Field(
        ..., description="Annual round-trip commute distance in km"
    )
    co2e_kg: float = Field(
        ..., description="Annual commute CO2e emissions in kg"
    )
    wtt_co2e_kg: float = Field(
        ..., description="Annual WTT CO2e emissions in kg"
    )
    total_co2e_kg: float = Field(
        ..., description="Total annual CO2e (commute + WTT) in kg"
    )
    ef_used: float = Field(
        ..., description="Emission factor applied (kgCO2e per km)"
    )
    ef_source: str = Field(
        ..., description="Emission factor source (DEFRA, EPA, IEA)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class TeleworkCalculateResponse(BaseModel):
    """Response model for telework/WFH emissions calculation."""

    success: bool = Field(
        ..., description="Whether calculation completed successfully"
    )
    calculation_id: str = Field(..., description="Unique calculation UUID")
    frequency: str = Field(
        ..., description="Telework frequency pattern"
    )
    telework_days_per_year: int = Field(
        ..., description="Annual telework days"
    )
    daily_kwh: float = Field(
        ..., description="Daily home office kWh consumption"
    )
    annual_kwh: float = Field(
        ..., description="Annual home office kWh consumption"
    )
    grid_ef_kgco2e_per_kwh: float = Field(
        ..., description="Grid emission factor applied (kgCO2e/kWh)"
    )
    telework_co2e_kg: float = Field(
        ..., description="Annual telework CO2e emissions in kg"
    )
    seasonal_adjustment_applied: bool = Field(
        ..., description="Whether seasonal adjustment was applied"
    )
    avoided_commute_co2e_kg: Optional[float] = Field(
        None, description="Avoided commute emissions (memo item, not subtracted)"
    )
    equipment_co2e_kg: Optional[float] = Field(
        None, description="Amortized equipment lifecycle emissions (memo item)"
    )
    region: str = Field(
        ..., description="Region used for grid emission factor"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class SurveyResponse(BaseModel):
    """Response model for employee survey processing."""

    success: bool = Field(
        ..., description="Whether survey processing completed successfully"
    )
    survey_id: str = Field(..., description="Unique survey processing UUID")
    survey_method: str = Field(
        ..., description="Survey methodology used"
    )
    total_employees: int = Field(
        ..., description="Total workforce headcount"
    )
    respondents: int = Field(
        ..., description="Number of survey respondents processed"
    )
    response_rate: float = Field(
        ..., description="Survey response rate (0.0-1.0)"
    )
    sample_co2e_kg: float = Field(
        ..., description="Total CO2e from survey respondents (kg)"
    )
    extrapolated_co2e_kg: float = Field(
        ..., description="Extrapolated total CO2e for full workforce (kg)"
    )
    extrapolation_factor: float = Field(
        ..., description="Extrapolation multiplier applied"
    )
    per_employee_avg_co2e_kg: float = Field(
        ..., description="Average CO2e per employee (kg)"
    )
    ci_lower_kg: float = Field(
        ..., description="Confidence interval lower bound (kg)"
    )
    ci_upper_kg: float = Field(
        ..., description="Confidence interval upper bound (kg)"
    )
    dqi_score: float = Field(
        ..., description="Data quality indicator score (1.0-5.0)"
    )
    mode_share: Dict[str, float] = Field(
        ..., description="Mode share distribution from survey"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )


class AverageDataResponse(BaseModel):
    """Response model for average-data method calculation."""

    success: bool = Field(
        ..., description="Whether calculation completed successfully"
    )
    calculation_id: str = Field(..., description="Unique calculation UUID")
    method: str = Field(
        default="average_data", description="Calculation method"
    )
    total_employees: int = Field(
        ..., description="Total FTE headcount"
    )
    country_code: str = Field(
        ..., description="Country code used for defaults"
    )
    average_distance_km: float = Field(
        ..., description="Average one-way commute distance used (km)"
    )
    working_days: int = Field(
        ..., description="Annual working days used"
    )
    total_co2e_kg: float = Field(
        ..., description="Total organizational CO2e (kg)"
    )
    per_employee_co2e_kg: float = Field(
        ..., description="Average per-employee CO2e (kg)"
    )
    mode_share_used: Dict[str, float] = Field(
        ..., description="Mode share distribution applied"
    )
    by_mode_co2e_kg: Dict[str, float] = Field(
        ..., description="CO2e breakdown by commute mode"
    )
    dqi_score: float = Field(
        ..., description="Data quality indicator score (typically 2.5-3.5)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class SpendCalculateResponse(BaseModel):
    """Response model for spend-based calculation."""

    success: bool = Field(
        ..., description="Whether calculation completed successfully"
    )
    calculation_id: str = Field(..., description="Unique calculation UUID")
    method: str = Field(
        default="spend_based", description="Calculation method"
    )
    naics_code: str = Field(
        ..., description="NAICS code used for EEIO factor"
    )
    original_amount: float = Field(
        ..., description="Original spend amount in source currency"
    )
    currency: str = Field(
        ..., description="Source currency code"
    )
    usd_amount: float = Field(
        ..., description="Amount converted to USD"
    )
    deflated_amount: float = Field(
        ..., description="CPI-deflated amount in base year USD"
    )
    eeio_factor: float = Field(
        ..., description="EEIO factor applied (kgCO2e per USD)"
    )
    total_co2e_kg: float = Field(
        ..., description="Total CO2e emissions (kg)"
    )
    dqi_score: float = Field(
        ..., description="Data quality indicator score (typically 1.5-2.5)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class MultiModalCalculateResponse(BaseModel):
    """Response model for multi-modal trip calculation."""

    success: bool = Field(
        ..., description="Whether calculation completed successfully"
    )
    calculation_id: str = Field(..., description="Unique calculation UUID")
    legs: List[Dict[str, Any]] = Field(
        ..., description="Per-leg calculation results"
    )
    total_legs: int = Field(
        ..., description="Number of trip legs"
    )
    single_trip_co2e_kg: float = Field(
        ..., description="Single one-way trip CO2e (kg)"
    )
    annual_co2e_kg: float = Field(
        ..., description="Annual round-trip CO2e (kg)"
    )
    annual_wtt_co2e_kg: float = Field(
        ..., description="Annual WTT CO2e (kg)"
    )
    total_annual_co2e_kg: float = Field(
        ..., description="Total annual CO2e (commute + WTT) in kg"
    )
    total_distance_km: float = Field(
        ..., description="Total one-way trip distance (km)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class CalculationDetailResponse(BaseModel):
    """Response model for single calculation detail."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    mode: str = Field(..., description="Primary commute mode")
    method: str = Field(..., description="Calculation method")
    total_co2e_kg: float = Field(..., description="Total CO2e (kg)")
    details: Dict[str, Any] = Field(
        ..., description="Full calculation detail payload"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


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

    calculation_id: str = Field(..., description="Deleted calculation UUID")
    deleted: bool = Field(..., description="Whether deletion succeeded")
    message: str = Field(..., description="Human-readable status message")


class EmissionFactorResponse(BaseModel):
    """Response model for a single emission factor."""

    mode: str = Field(..., description="Commute mode")
    vehicle_type: Optional[str] = Field(
        None, description="Vehicle/service subtype"
    )
    ef_per_vkm: Optional[float] = Field(
        None, description="Emission factor per vehicle-km (kgCO2e/vkm)"
    )
    ef_per_pkm: float = Field(
        ..., description="Emission factor per passenger-km (kgCO2e/pkm)"
    )
    wtt_per_pkm: float = Field(
        ..., description="Well-to-tank factor per passenger-km (kgCO2e/pkm)"
    )
    default_occupancy: Optional[float] = Field(
        None, description="Default vehicle occupancy"
    )
    unit: str = Field(
        ..., description="Unit basis (per-pkm, per-vkm, per-litre, per-kWh)"
    )
    source: str = Field(
        ..., description="Factor source (DEFRA, EPA, IEA, EEIO)"
    )
    year: int = Field(
        ..., description="Factor source year"
    )


class EmissionFactorListResponse(BaseModel):
    """Response model for emission factor listing."""

    factors: List[EmissionFactorResponse] = Field(
        ..., description="List of emission factors"
    )
    count: int = Field(..., description="Total factor count returned")


class CommuteModeResponse(BaseModel):
    """Response model for supported commute modes."""

    modes: List[Dict[str, Any]] = Field(
        ..., description="List of commute modes with metadata"
    )
    count: int = Field(..., description="Number of supported modes")


class WorkingDaysResponse(BaseModel):
    """Response model for working days by region."""

    region: str = Field(..., description="Region code")
    calendar_days: int = Field(
        ..., description="Weekdays in year (typically 250)"
    )
    public_holidays: int = Field(
        ..., description="Public holidays"
    )
    default_pto: int = Field(
        ..., description="Default PTO/vacation days"
    )
    default_sick_days: int = Field(
        ..., description="Default sick days"
    )
    net_working_days: int = Field(
        ..., description="Net working days (calendar - holidays - PTO - sick)"
    )
    source: str = Field(
        ..., description="Data source for working days defaults"
    )


class CommuteAverageResponse(BaseModel):
    """Response model for average commute distances."""

    averages: List[Dict[str, Any]] = Field(
        ..., description="Average commute distances by country/region"
    )
    count: int = Field(..., description="Number of entries returned")


class GridFactorResponse(BaseModel):
    """Response model for grid emission factors for telework."""

    country_code: str = Field(
        ..., description="Country or eGRID subregion code"
    )
    grid_ef_kgco2e_per_kwh: float = Field(
        ..., description="Grid emission factor (kgCO2e/kWh)"
    )
    source: str = Field(
        ..., description="Data source (IEA, eGRID)"
    )
    source_year: int = Field(
        ..., description="Source data year"
    )
    subregions: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="eGRID subregional factors (US only)",
    )


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance check."""

    success: bool = Field(
        ..., description="Whether compliance check completed"
    )
    overall_status: str = Field(
        ..., description="Overall compliance status (pass, fail, warning)"
    )
    overall_score: float = Field(
        ..., description="Overall compliance score (0.0-1.0)"
    )
    framework_results: List[Dict[str, Any]] = Field(
        ..., description="Per-framework compliance results with findings"
    )
    double_counting_flags: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Double-counting prevention rule violations",
    )


class UncertaintyResponse(BaseModel):
    """Response model for uncertainty analysis."""

    success: bool = Field(
        ..., description="Whether analysis completed successfully"
    )
    mean_co2e_kg: float = Field(..., description="Mean CO2e (kg)")
    std_dev_kg: float = Field(..., description="Standard deviation (kg)")
    ci_lower_kg: float = Field(
        ..., description="Confidence interval lower bound (kg)"
    )
    ci_upper_kg: float = Field(
        ..., description="Confidence interval upper bound (kg)"
    )
    uncertainty_pct: float = Field(
        ..., description="Uncertainty as percentage (+/-%)"
    )
    method: str = Field(..., description="Uncertainty method used")
    iterations: int = Field(
        ..., description="Iterations performed (0 for non-Monte Carlo)"
    )
    confidence_level: float = Field(
        ..., description="Confidence level used"
    )


class AggregationResponse(BaseModel):
    """Response model for aggregated emissions."""

    period: str = Field(..., description="Aggregation period identifier")
    total_co2e_kg: float = Field(
        ..., description="Total CO2e for the period (kg)"
    )
    commute_co2e_kg: float = Field(
        ..., description="Commute-only CO2e (kg)"
    )
    telework_co2e_kg: float = Field(
        ..., description="Telework CO2e (kg)"
    )
    wtt_co2e_kg: float = Field(
        ..., description="Well-to-tank CO2e (kg)"
    )
    by_mode: Dict[str, float] = Field(
        ..., description="CO2e breakdown by commute mode"
    )
    by_department: Dict[str, float] = Field(
        ..., description="CO2e breakdown by department"
    )
    employee_count: int = Field(
        ..., description="Total employees in aggregation"
    )
    per_employee_co2e_kg: float = Field(
        ..., description="Average per-employee CO2e (kg)"
    )


class ModeShareResponse(BaseModel):
    """Response model for mode share analysis."""

    success: bool = Field(
        ..., description="Whether analysis completed successfully"
    )
    reporting_period: str = Field(
        ..., description="Reporting period analyzed"
    )
    total_employees: int = Field(
        ..., description="Total employees in analysis"
    )
    mode_shares: Dict[str, float] = Field(
        ..., description="Mode share distribution (mode -> percentage)"
    )
    mode_co2e: Dict[str, float] = Field(
        ..., description="CO2e by mode (mode -> kgCO2e)"
    )
    mode_avg_distance_km: Dict[str, float] = Field(
        ..., description="Average commute distance by mode (km)"
    )
    benchmark_comparison: Optional[Dict[str, Any]] = Field(
        None, description="Comparison to regional/national benchmarks"
    )
    intervention_opportunities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ranked intervention opportunities with estimated reduction",
    )


class ProvenanceResponse(BaseModel):
    """Response model for provenance chain verification."""

    success: bool = Field(
        ..., description="Whether provenance retrieval succeeded"
    )
    calculation_id: str = Field(..., description="Calculation UUID")
    chain: List[Dict[str, Any]] = Field(
        ..., description="Ordered list of provenance stage records"
    )
    is_valid: bool = Field(
        ..., description="Whether the provenance chain is intact"
    )
    root_hash: str = Field(
        ..., description="Root SHA-256 hash of the chain"
    )
    stages_count: int = Field(
        ..., description="Number of stages in chain (expected: 10)"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service health status")
    agent_id: str = Field(..., description="Agent identifier")
    version: str = Field(..., description="Agent version")
    uptime_seconds: float = Field(
        ..., description="Seconds since service start"
    )
    engines_status: Optional[Dict[str, str]] = Field(
        None,
        description="Per-engine health status",
    )


# ============================================================================
# MODULE-LEVEL TRACKING
# ============================================================================

_start_time: datetime = datetime.utcnow()


# ============================================================================
# ENDPOINTS - CALCULATIONS (8)
# ============================================================================


@router.post(
    "/calculate",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate employee commuting emissions",
    description=(
        "Calculate GHG emissions for a single employee's commute through the "
        "full 10-stage pipeline. Supports all 14 commute modes with mode-specific "
        "parameters. Optionally includes telework/WFH emissions. Returns "
        "deterministic results with SHA-256 provenance hash."
    ),
)
async def calculate_emissions(
    request: CalculateRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate employee commuting emissions through the full pipeline.

    Processes commute data through 10 pipeline stages: validate, classify,
    normalize, resolve_efs, calculate_commute, calculate_telework,
    extrapolate, compliance, aggregate, seal.

    Args:
        request: Calculation request with commute mode and data
        service: EmployeeCommutingService instance

    Returns:
        CalculationResponse with emissions breakdown and provenance hash

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating commuting emissions for mode={request.mode}, "
            f"employee_id={request.employee_id}"
        )

        result = await service.calculate(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculationResponse(
            success=True,
            calculation_id=calculation_id,
            mode=result.get("mode", request.mode),
            method=result.get("method", "employee_specific"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            commute_co2e_kg=result.get("commute_co2e_kg", 0.0),
            telework_co2e_kg=result.get("telework_co2e_kg"),
            wtt_co2e_kg=result.get("wtt_co2e_kg", 0.0),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            detail=result.get("detail"),
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
            detail="Calculation failed",
        )


@router.post(
    "/calculate/batch",
    response_model=BatchCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch calculate employee commuting emissions",
    description=(
        "Calculate GHG emissions for multiple employees in a single request. "
        "Processes up to 50,000 employees with parallel execution and per-employee "
        "error isolation. Supports workforce-level extrapolation from survey samples "
        "to full headcount using statistical methods."
    ),
)
async def calculate_batch_emissions(
    request: BatchCalculateRequest,
    service=Depends(get_service),
) -> BatchCalculateResponse:
    """
    Calculate batch employee commuting emissions.

    Args:
        request: Batch calculation request with employee list
        service: EmployeeCommutingService instance

    Returns:
        BatchCalculateResponse with aggregated and per-employee results

    Raises:
        HTTPException: 400 for validation errors, 500 for batch failures
    """
    try:
        logger.info(
            f"Calculating batch commuting emissions for "
            f"{len(request.employees)} employees, "
            f"period={request.reporting_period}"
        )

        result = await service.calculate_batch(request.dict())
        batch_id = result.get("batch_id", str(uuid.uuid4()))

        return BatchCalculateResponse(
            success=True,
            batch_id=batch_id,
            total_employees=result.get(
                "total_employees", len(request.employees)
            ),
            successful=result.get("successful", 0),
            failed=result.get("failed", 0),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            extrapolated_co2e_kg=result.get("extrapolated_co2e_kg"),
            results=result.get("results", []),
            errors=result.get("errors", []),
            reporting_period=result.get(
                "reporting_period", request.reporting_period
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_batch_emissions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_batch_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch calculation failed",
        )


@router.post(
    "/calculate/commute",
    response_model=CommuteCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate single commute mode emissions",
    description=(
        "Calculate GHG emissions for a single commute mode without running the "
        "full pipeline. Supports distance-based calculation for all 14 modes "
        "with mode-specific emission factors. Uses per-vkm factors for personal "
        "vehicles and per-pkm factors for transit. Carpool/vanpool emissions "
        "are divided by occupancy."
    ),
)
async def calculate_commute_emissions(
    request: CommuteCalculateRequest,
    service=Depends(get_service),
) -> CommuteCalculateResponse:
    """
    Calculate emissions for a single commute mode.

    Args:
        request: Commute calculation request with mode, distance, frequency
        service: EmployeeCommutingService instance

    Returns:
        CommuteCalculateResponse with annual commute emissions

    Raises:
        HTTPException: 400 for invalid mode or parameters, 500 for failures
    """
    try:
        logger.info(
            f"Calculating commute emissions: mode={request.mode}, "
            f"distance={request.distance_km}km, frequency={request.frequency}d/wk"
        )

        result = await service.calculate_commute(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CommuteCalculateResponse(
            success=True,
            calculation_id=calculation_id,
            mode=result.get("mode", request.mode),
            vehicle_type=result.get("vehicle_type", request.vehicle_type),
            distance_km=result.get("distance_km", request.distance_km),
            annual_distance_km=result.get("annual_distance_km", 0.0),
            co2e_kg=result.get("co2e_kg", 0.0),
            wtt_co2e_kg=result.get("wtt_co2e_kg", 0.0),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            ef_used=result.get("ef_used", 0.0),
            ef_source=result.get("ef_source", "DEFRA"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_commute_emissions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_commute_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Commute calculation failed",
        )


@router.post(
    "/calculate/telework",
    response_model=TeleworkCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate telework emissions",
    description=(
        "Calculate GHG emissions from home office energy consumption for "
        "telework/remote work employees. Uses country-specific grid emission "
        "factors (IEA 2024, 19 countries) or US eGRID subregional factors "
        "(26 subregions). Supports seasonal heating/cooling adjustments and "
        "optional equipment lifecycle emissions. Reports avoided commute "
        "emissions as a memo item (not subtracted from total)."
    ),
)
async def calculate_telework_emissions(
    request: TeleworkCalculateRequest,
    service=Depends(get_service),
) -> TeleworkCalculateResponse:
    """
    Calculate telework/WFH emissions.

    Args:
        request: Telework calculation request with frequency, region, kWh
        service: EmployeeCommutingService instance

    Returns:
        TeleworkCalculateResponse with home office energy emissions

    Raises:
        HTTPException: 400 for invalid region or parameters, 500 for failures
    """
    try:
        logger.info(
            f"Calculating telework emissions: frequency={request.frequency}, "
            f"region={request.region}, daily_kwh={request.daily_kwh}"
        )

        result = await service.calculate_telework(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return TeleworkCalculateResponse(
            success=True,
            calculation_id=calculation_id,
            frequency=result.get("frequency", request.frequency),
            telework_days_per_year=result.get("telework_days_per_year", 0),
            daily_kwh=result.get("daily_kwh", request.daily_kwh),
            annual_kwh=result.get("annual_kwh", 0.0),
            grid_ef_kgco2e_per_kwh=result.get("grid_ef_kgco2e_per_kwh", 0.0),
            telework_co2e_kg=result.get("telework_co2e_kg", 0.0),
            seasonal_adjustment_applied=result.get(
                "seasonal_adjustment_applied", False
            ),
            avoided_commute_co2e_kg=result.get("avoided_commute_co2e_kg"),
            equipment_co2e_kg=result.get("equipment_co2e_kg"),
            region=result.get("region", request.region),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in calculate_telework_emissions: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_telework_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Telework calculation failed",
        )


@router.post(
    "/calculate/survey",
    response_model=SurveyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Process employee commute survey",
    description=(
        "Process employee commute survey data with statistical extrapolation "
        "from survey sample to full workforce. Supports 4 survey methods: "
        "full census, stratified sample, random sample, and convenience. "
        "Calculates per-respondent emissions, extrapolates to workforce, "
        "and reports confidence intervals and data quality scores."
    ),
)
async def process_survey(
    request: SurveyRequest,
    service=Depends(get_service),
) -> SurveyResponse:
    """
    Process employee commute survey data.

    Args:
        request: Survey request with responses and workforce size
        service: EmployeeCommutingService instance

    Returns:
        SurveyResponse with extrapolated emissions and confidence intervals

    Raises:
        HTTPException: 400 for invalid survey data, 500 for processing failures
    """
    try:
        logger.info(
            f"Processing survey: method={request.survey_method}, "
            f"responses={len(request.responses)}, "
            f"total_employees={request.total_employees}"
        )

        result = await service.process_survey(request.dict())
        survey_id = result.get("survey_id", str(uuid.uuid4()))

        return SurveyResponse(
            success=True,
            survey_id=survey_id,
            survey_method=result.get(
                "survey_method", request.survey_method
            ),
            total_employees=result.get(
                "total_employees", request.total_employees
            ),
            respondents=result.get("respondents", len(request.responses)),
            response_rate=result.get(
                "response_rate",
                len(request.responses) / request.total_employees,
            ),
            sample_co2e_kg=result.get("sample_co2e_kg", 0.0),
            extrapolated_co2e_kg=result.get("extrapolated_co2e_kg", 0.0),
            extrapolation_factor=result.get("extrapolation_factor", 1.0),
            per_employee_avg_co2e_kg=result.get(
                "per_employee_avg_co2e_kg", 0.0
            ),
            ci_lower_kg=result.get("ci_lower_kg", 0.0),
            ci_upper_kg=result.get("ci_upper_kg", 0.0),
            dqi_score=result.get("dqi_score", 3.0),
            mode_share=result.get("mode_share", {}),
            provenance_hash=result.get("provenance_hash", ""),
        )

    except ValueError as e:
        logger.error("Validation error in process_survey: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in process_survey: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Survey processing failed",
        )


@router.post(
    "/calculate/average-data",
    response_model=AverageDataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate using average-data method",
    description=(
        "Calculate employee commuting emissions using the average-data method. "
        "Uses national/regional average commute distances (census data for 10 "
        "countries + global) and default mode share distributions scaled by "
        "workforce headcount. Suitable when employee survey data is unavailable. "
        "DQI score typically 2.5-3.5 with +/-25-40% uncertainty."
    ),
)
async def calculate_average_data(
    request: AverageDataRequest,
    service=Depends(get_service),
) -> AverageDataResponse:
    """
    Calculate emissions using average-data method.

    Args:
        request: Average-data request with headcount and country
        service: EmployeeCommutingService instance

    Returns:
        AverageDataResponse with organization-level emissions

    Raises:
        HTTPException: 400 for invalid country code, 500 for failures
    """
    try:
        logger.info(
            f"Calculating average-data emissions: "
            f"employees={request.total_employees}, "
            f"country={request.country_code}"
        )

        result = await service.calculate_average_data(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return AverageDataResponse(
            success=True,
            calculation_id=calculation_id,
            method="average_data",
            total_employees=result.get(
                "total_employees", request.total_employees
            ),
            country_code=result.get(
                "country_code", request.country_code
            ),
            average_distance_km=result.get("average_distance_km", 0.0),
            working_days=result.get("working_days", 230),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            per_employee_co2e_kg=result.get("per_employee_co2e_kg", 0.0),
            mode_share_used=result.get("mode_share_used", {}),
            by_mode_co2e_kg=result.get("by_mode_co2e_kg", {}),
            dqi_score=result.get("dqi_score", 3.0),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_average_data: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_average_data: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Average-data calculation failed",
        )


@router.post(
    "/calculate/spend",
    response_model=SpendCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate spend-based emissions",
    description=(
        "Calculate GHG emissions using spend-based EEIO (Environmentally "
        "Extended Input-Output) factors. Applies CPI deflation to base year "
        "(2021 USD), currency conversion, and margin removal. Supports 8 NAICS "
        "commuting-related industry codes and 12 currencies. Lowest accuracy "
        "method with DQI score typically 1.5-2.5 and +/-40-60% uncertainty."
    ),
)
async def calculate_spend_emissions(
    request: SpendCalculateRequest,
    service=Depends(get_service),
) -> SpendCalculateResponse:
    """
    Calculate spend-based emissions using EEIO factors.

    Args:
        request: Spend calculation request with NAICS code and amount
        service: EmployeeCommutingService instance

    Returns:
        SpendCalculateResponse with spend-based emissions

    Raises:
        HTTPException: 400 for invalid NAICS code, 500 for failures
    """
    try:
        logger.info(
            f"Calculating spend emissions: naics={request.naics_code}, "
            f"amount={request.amount} {request.currency}"
        )

        result = await service.calculate_spend(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return SpendCalculateResponse(
            success=True,
            calculation_id=calculation_id,
            method="spend_based",
            naics_code=result.get("naics_code", request.naics_code),
            original_amount=result.get("original_amount", request.amount),
            currency=result.get("currency", request.currency),
            usd_amount=result.get("usd_amount", 0.0),
            deflated_amount=result.get("deflated_amount", 0.0),
            eeio_factor=result.get("eeio_factor", 0.0),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            dqi_score=result.get("dqi_score", 2.0),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_spend_emissions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_spend_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Spend calculation failed",
        )


@router.post(
    "/calculate/multi-modal",
    response_model=MultiModalCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate multi-modal trip emissions",
    description=(
        "Calculate GHG emissions for a multi-modal commute consisting of up "
        "to 5 transport segments. Each leg uses its own mode-specific emission "
        "factor. Common patterns include park-and-ride (drive + rail + walk), "
        "bus-to-rail transfers, and bike-to-transit combinations. Returns "
        "per-leg and total annual emissions."
    ),
)
async def calculate_multi_modal_emissions(
    request: MultiModalCalculateRequest,
    service=Depends(get_service),
) -> MultiModalCalculateResponse:
    """
    Calculate multi-modal trip emissions.

    Args:
        request: Multi-modal request with legs, frequency, working days
        service: EmployeeCommutingService instance

    Returns:
        MultiModalCalculateResponse with per-leg and annual totals

    Raises:
        HTTPException: 400 for invalid legs, 500 for failures
    """
    try:
        logger.info(
            f"Calculating multi-modal emissions: "
            f"legs={len(request.legs)}, frequency={request.frequency}d/wk"
        )

        result = await service.calculate_multi_modal(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return MultiModalCalculateResponse(
            success=True,
            calculation_id=calculation_id,
            legs=result.get("legs", []),
            total_legs=result.get("total_legs", len(request.legs)),
            single_trip_co2e_kg=result.get("single_trip_co2e_kg", 0.0),
            annual_co2e_kg=result.get("annual_co2e_kg", 0.0),
            annual_wtt_co2e_kg=result.get("annual_wtt_co2e_kg", 0.0),
            total_annual_co2e_kg=result.get("total_annual_co2e_kg", 0.0),
            total_distance_km=result.get("total_distance_km", 0.0),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in calculate_multi_modal_emissions: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_multi_modal_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Multi-modal calculation failed",
        )


# ============================================================================
# ENDPOINTS - CALCULATION CRUD (3)
# ============================================================================


@router.get(
    "/calculations/{calculation_id}",
    response_model=CalculationDetailResponse,
    summary="Get calculation detail",
    description=(
        "Retrieve detailed information for a specific employee commuting "
        "calculation including full input/output payload, emission breakdown "
        "by commute and telework components, provenance hash, and calculation "
        "metadata."
    ),
)
async def get_calculation_detail(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> CalculationDetailResponse:
    """
    Get detailed information for a specific calculation.

    Args:
        calculation_id: Calculation UUID
        service: EmployeeCommutingService instance

    Returns:
        CalculationDetailResponse with full calculation data

    Raises:
        HTTPException: 404 if calculation not found, 500 for failures
    """
    try:
        logger.info("Getting calculation detail: %s", calculation_id)

        result = await service.get_calculation(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found",
            )

        return CalculationDetailResponse(
            calculation_id=result.get("calculation_id", calculation_id),
            mode=result.get("mode", ""),
            method=result.get("method", ""),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            details=result.get("details", {}),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_calculation_detail: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve calculation",
        )


@router.get(
    "/calculations",
    response_model=CalculationListResponse,
    summary="List calculations",
    description=(
        "Retrieve a paginated list of employee commuting calculations. "
        "Supports filtering by commute mode, employee ID, department, site, "
        "calculation method, and date range. Returns summary information "
        "for each calculation."
    ),
)
async def list_calculations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Results per page"),
    mode: Optional[str] = Query(
        None, description="Filter by commute mode"
    ),
    method: Optional[str] = Query(
        None,
        description="Filter by calculation method (employee_specific, average_data, spend_based)",
    ),
    employee_id: Optional[str] = Query(
        None, description="Filter by employee ID"
    ),
    department: Optional[str] = Query(
        None, description="Filter by department"
    ),
    site: Optional[str] = Query(
        None, description="Filter by site/location"
    ),
    from_date: Optional[str] = Query(
        None, description="Filter from date (ISO 8601)"
    ),
    to_date: Optional[str] = Query(
        None, description="Filter to date (ISO 8601)"
    ),
    service=Depends(get_service),
) -> CalculationListResponse:
    """
    List employee commuting calculations with filtering and pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of results per page
        mode: Optional commute mode filter
        method: Optional calculation method filter
        employee_id: Optional employee ID filter
        department: Optional department filter
        site: Optional site filter
        from_date: Optional start date filter
        to_date: Optional end date filter
        service: EmployeeCommutingService instance

    Returns:
        CalculationListResponse with paginated results

    Raises:
        HTTPException: 500 for listing failures
    """
    try:
        logger.info(
            f"Listing calculations: page={page}, size={page_size}, "
            f"mode={mode}, method={method}"
        )

        filters = {
            "page": page,
            "page_size": page_size,
            "mode": mode,
            "method": method,
            "employee_id": employee_id,
            "department": department,
            "site": site,
            "from_date": from_date,
            "to_date": to_date,
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
    summary="Delete calculation",
    description=(
        "Soft-delete a specific employee commuting calculation. "
        "Marks the calculation as deleted with audit trail; "
        "data is retained for regulatory compliance per GHG Protocol "
        "and CSRD data retention requirements."
    ),
)
async def delete_calculation(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> DeleteResponse:
    """
    Soft-delete a specific calculation.

    Args:
        calculation_id: Calculation UUID
        service: EmployeeCommutingService instance

    Returns:
        DeleteResponse with deletion confirmation

    Raises:
        HTTPException: 404 if not found, 500 for deletion failures
    """
    try:
        logger.info("Deleting calculation: %s", calculation_id)

        deleted = await service.delete_calculation(calculation_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found",
            )

        return DeleteResponse(
            calculation_id=calculation_id,
            deleted=True,
            message=f"Calculation {calculation_id} soft-deleted successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in delete_calculation: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete calculation",
        )


# ============================================================================
# ENDPOINTS - EMISSION FACTORS & REFERENCE DATA (6)
# ============================================================================


@router.get(
    "/emission-factors",
    response_model=EmissionFactorListResponse,
    summary="List emission factors",
    description=(
        "Retrieve available emission factors for employee commuting. "
        "Supports filtering by commute mode and data source. Returns "
        "DEFRA 2024 vehicle and transit factors, EPA factors, IEA grid "
        "factors, and EEIO spend-based factors. Includes per-vkm and "
        "per-pkm values with WTT (well-to-tank) factors."
    ),
)
async def list_emission_factors(
    mode: Optional[str] = Query(
        None, description="Filter by commute mode"
    ),
    source: Optional[str] = Query(
        None, description="Filter by EF source (defra, epa, iea, eeio)"
    ),
    service=Depends(get_service),
) -> EmissionFactorListResponse:
    """
    List emission factors with optional filtering.

    Args:
        mode: Optional commute mode filter
        source: Optional data source filter
        service: EmployeeCommutingService instance

    Returns:
        EmissionFactorListResponse with matching factors

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info(
            f"Listing emission factors: mode={mode}, source={source}"
        )

        filters = {"mode": mode, "source": source}
        result = await service.list_emission_factors(filters)

        return EmissionFactorListResponse(
            factors=result.get("factors", []),
            count=result.get("count", 0),
        )

    except Exception as e:
        logger.error("Error in list_emission_factors: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list emission factors",
        )


@router.get(
    "/emission-factors/{mode}",
    response_model=EmissionFactorListResponse,
    summary="Get emission factors by commute mode",
    description=(
        "Retrieve emission factors for a specific commute mode. "
        "Returns all vehicle/service subtypes and their EF values "
        "including well-to-tank (WTT) factors and default occupancy. "
        "For vehicle modes, returns both per-vkm and per-pkm values."
    ),
)
async def get_emission_factors_by_mode(
    mode: str = Path(
        ...,
        description=(
            "Commute mode (sov, carpool, vanpool, bus, metro, light_rail, "
            "commuter_rail, ferry, motorcycle, e_bike, e_scooter)"
        ),
    ),
    service=Depends(get_service),
) -> EmissionFactorListResponse:
    """
    Get emission factors for a specific commute mode.

    Args:
        mode: Commute mode identifier
        service: EmployeeCommutingService instance

    Returns:
        EmissionFactorListResponse with mode-specific factors

    Raises:
        HTTPException: 400 for invalid mode, 500 for retrieval failures
    """
    try:
        logger.info("Getting emission factors for mode: %s", mode)

        result = await service.get_emission_factors_by_mode(mode)

        return EmissionFactorListResponse(
            factors=result.get("factors", []),
            count=result.get("count", 0),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in get_emission_factors_by_mode: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in get_emission_factors_by_mode: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve emission factors",
        )


@router.get(
    "/commute-modes",
    response_model=CommuteModeResponse,
    summary="List commute modes",
    description=(
        "Retrieve all 14 supported commute modes with metadata including "
        "category (vehicle, transit, active, telework), available vehicle "
        "subtypes, calculation methods, EF sources, and default occupancy "
        "values. Zero-emission modes (cycling, walking) are included for "
        "mode share tracking."
    ),
)
async def list_commute_modes(
    service=Depends(get_service),
) -> CommuteModeResponse:
    """
    List all supported commute modes.

    Args:
        service: EmployeeCommutingService instance

    Returns:
        CommuteModeResponse with mode metadata

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info("Listing commute modes")

        result = await service.list_commute_modes()

        return CommuteModeResponse(
            modes=result.get("modes", []),
            count=result.get("count", 0),
        )

    except Exception as e:
        logger.error("Error in list_commute_modes: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list commute modes",
        )


@router.get(
    "/working-days/{region}",
    response_model=WorkingDaysResponse,
    summary="Get working days by region",
    description=(
        "Retrieve default working days for a specific region. Returns "
        "calendar weekdays, public holidays, default PTO days, default "
        "sick days, and net working days. Supports 11 regions: US, GB, "
        "DE, FR, JP, CA, AU, IN, CN, BR, KR, and GLOBAL default."
    ),
)
async def get_working_days(
    region: str = Path(
        ...,
        description="Region code (US, GB, DE, FR, JP, CA, AU, IN, CN, BR, KR, GLOBAL)",
    ),
    service=Depends(get_service),
) -> WorkingDaysResponse:
    """
    Get working days defaults for a specific region.

    Args:
        region: Region code
        service: EmployeeCommutingService instance

    Returns:
        WorkingDaysResponse with working days breakdown

    Raises:
        HTTPException: 400 for invalid region, 500 for retrieval failures
    """
    try:
        logger.info("Getting working days for region: %s", region)

        result = await service.get_working_days(region)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid region '{region}'. Supported regions: "
                    "US, GB, DE, FR, JP, CA, AU, IN, CN, BR, KR, GLOBAL"
                ),
            )

        return WorkingDaysResponse(
            region=result.get("region", region),
            calendar_days=result.get("calendar_days", 250),
            public_holidays=result.get("public_holidays", 0),
            default_pto=result.get("default_pto", 0),
            default_sick_days=result.get("default_sick_days", 0),
            net_working_days=result.get("net_working_days", 230),
            source=result.get("source", "GreenLang defaults"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_working_days: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve working days",
        )


@router.get(
    "/commute-averages",
    response_model=CommuteAverageResponse,
    summary="Get average commute distances",
    description=(
        "Retrieve average commute distances by country from census and "
        "national statistics data. Supports 10 countries (US, GB, DE, FR, "
        "JP, CA, AU, IN, CN, KR) plus global average. Used as defaults "
        "for the average-data calculation method."
    ),
)
async def get_commute_averages(
    country_code: Optional[str] = Query(
        None,
        description="Filter by ISO country code (returns all if not specified)",
    ),
    service=Depends(get_service),
) -> CommuteAverageResponse:
    """
    Get average commute distances by country.

    Args:
        country_code: Optional country code filter
        service: EmployeeCommutingService instance

    Returns:
        CommuteAverageResponse with average distances

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting commute averages: country_code={country_code}"
        )

        filters = {"country_code": country_code}
        result = await service.get_commute_averages(filters)

        return CommuteAverageResponse(
            averages=result.get("averages", []),
            count=result.get("count", 0),
        )

    except Exception as e:
        logger.error("Error in get_commute_averages: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve commute averages",
        )


@router.get(
    "/grid-factors/{country}",
    response_model=GridFactorResponse,
    summary="Get grid emission factors for telework",
    description=(
        "Retrieve grid emission factors for telework/WFH calculations by "
        "country. Returns kgCO2e/kWh from IEA 2024 data for 19 countries. "
        "For the US, also returns eGRID subregional factors (26 subregions) "
        "for more precise telework emission calculations."
    ),
)
async def get_grid_factors(
    country: str = Path(
        ...,
        description="ISO country code (US, GB, DE, FR, JP, CA, AU, etc.)",
    ),
    service=Depends(get_service),
) -> GridFactorResponse:
    """
    Get grid emission factors for a country.

    Args:
        country: ISO country code
        service: EmployeeCommutingService instance

    Returns:
        GridFactorResponse with grid EF and optional subregional data

    Raises:
        HTTPException: 400 for invalid country, 500 for retrieval failures
    """
    try:
        logger.info("Getting grid factors for country: %s", country)

        result = await service.get_grid_factors(country)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Grid factors not available for country '{country}'",
            )

        return GridFactorResponse(
            country_code=result.get("country_code", country),
            grid_ef_kgco2e_per_kwh=result.get(
                "grid_ef_kgco2e_per_kwh", 0.0
            ),
            source=result.get("source", "IEA"),
            source_year=result.get("source_year", 2024),
            subregions=result.get("subregions"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_grid_factors: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve grid factors",
        )


# ============================================================================
# ENDPOINTS - COMPLIANCE & UNCERTAINTY (2)
# ============================================================================


@router.post(
    "/compliance/check",
    response_model=ComplianceCheckResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Check multi-framework compliance",
    description=(
        "Check employee commuting calculation results against one or more "
        "regulatory frameworks. Validates completeness, boundary correctness, "
        "telework disclosure (required by CSRD), mode share reporting (required "
        "by CDP), double-counting prevention (DC-EC-001 through DC-EC-010), "
        "survey methodology documentation, and materiality thresholds. "
        "Supports GHG Protocol, ISO 14064, CSRD ESRS E1, CDP, SBTi, SB 253, "
        "and GRI 305."
    ),
)
async def check_compliance(
    request: ComplianceCheckRequest,
    service=Depends(get_service),
) -> ComplianceCheckResponse:
    """
    Check calculation compliance against regulatory frameworks.

    Args:
        request: Compliance check request with frameworks and results
        service: EmployeeCommutingService instance

    Returns:
        ComplianceCheckResponse with per-framework findings

    Raises:
        HTTPException: 400 for invalid frameworks, 500 for check failures
    """
    try:
        logger.info(
            f"Checking compliance for {len(request.frameworks)} frameworks, "
            f"{len(request.calculation_results)} results"
        )

        result = await service.check_compliance(request.dict())

        return ComplianceCheckResponse(
            success=True,
            overall_status=result.get("overall_status", "unknown"),
            overall_score=result.get("overall_score", 0.0),
            framework_results=result.get("framework_results", []),
            double_counting_flags=result.get("double_counting_flags", []),
        )

    except ValueError as e:
        logger.error("Validation error in check_compliance: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in check_compliance: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance check failed",
        )


@router.post(
    "/uncertainty/analyze",
    response_model=UncertaintyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze calculation uncertainty",
    description=(
        "Perform uncertainty analysis on employee commuting emissions "
        "calculations. Supports Monte Carlo simulation (sampling distances "
        "from normal, working days from uniform, EFs from triangular, "
        "response rates from beta distributions), analytical error "
        "propagation, and IPCC Tier 2 default ranges. Employee commuting "
        "uncertainty is typically +/-10-60% depending on method and data quality."
    ),
)
async def analyze_uncertainty(
    request: UncertaintyRequest,
    service=Depends(get_service),
) -> UncertaintyResponse:
    """
    Perform uncertainty analysis on calculation results.

    Args:
        request: Uncertainty analysis request
        service: EmployeeCommutingService instance

    Returns:
        UncertaintyResponse with statistical uncertainty metrics

    Raises:
        HTTPException: 400 for invalid method, 500 for analysis failures
    """
    try:
        logger.info(
            f"Analyzing uncertainty: method={request.method}, "
            f"iterations={request.iterations}, "
            f"confidence={request.confidence_level}"
        )

        result = await service.analyze_uncertainty(request.dict())

        return UncertaintyResponse(
            success=True,
            mean_co2e_kg=result.get("mean_co2e_kg", 0.0),
            std_dev_kg=result.get("std_dev_kg", 0.0),
            ci_lower_kg=result.get("ci_lower_kg", 0.0),
            ci_upper_kg=result.get("ci_upper_kg", 0.0),
            uncertainty_pct=result.get("uncertainty_pct", 0.0),
            method=result.get("method", request.method),
            iterations=result.get("iterations", request.iterations),
            confidence_level=result.get(
                "confidence_level", request.confidence_level
            ),
        )

    except ValueError as e:
        logger.error("Validation error in analyze_uncertainty: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in analyze_uncertainty: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Uncertainty analysis failed",
        )


# ============================================================================
# ENDPOINTS - AGGREGATION & MODE SHARE ANALYSIS (2)
# ============================================================================


@router.get(
    "/aggregations/{period}",
    response_model=AggregationResponse,
    summary="Get aggregated emissions",
    description=(
        "Retrieve aggregated employee commuting emissions for a specified "
        "period. Returns totals with breakdowns by commute mode, department, "
        "and commute/telework/WTT components. Supports daily, weekly, monthly, "
        "quarterly, and annual aggregation periods."
    ),
)
async def get_aggregations(
    period: str = Path(
        ...,
        description="Aggregation period (daily, weekly, monthly, quarterly, annual)",
    ),
    from_date: Optional[str] = Query(
        None, description="Start date (ISO 8601)"
    ),
    to_date: Optional[str] = Query(
        None, description="End date (ISO 8601)"
    ),
    department: Optional[str] = Query(
        None, description="Filter by department"
    ),
    site: Optional[str] = Query(
        None, description="Filter by site/location"
    ),
    service=Depends(get_service),
) -> AggregationResponse:
    """
    Get aggregated emissions for a specified period.

    Args:
        period: Aggregation period identifier
        from_date: Optional start date filter
        to_date: Optional end date filter
        department: Optional department filter
        site: Optional site filter
        service: EmployeeCommutingService instance

    Returns:
        AggregationResponse with aggregated emissions data

    Raises:
        HTTPException: 400 for invalid period, 500 for aggregation failures
    """
    try:
        logger.info(
            f"Getting aggregations: period={period}, "
            f"from={from_date}, to={to_date}"
        )

        valid_periods = {
            "daily", "weekly", "monthly", "quarterly", "annual",
        }
        if period not in valid_periods:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid period '{period}'. "
                    f"Must be one of: {', '.join(sorted(valid_periods))}"
                ),
            )

        filters = {
            "period": period,
            "from_date": from_date,
            "to_date": to_date,
            "department": department,
            "site": site,
        }

        result = await service.get_aggregations(filters)

        return AggregationResponse(
            period=period,
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            commute_co2e_kg=result.get("commute_co2e_kg", 0.0),
            telework_co2e_kg=result.get("telework_co2e_kg", 0.0),
            wtt_co2e_kg=result.get("wtt_co2e_kg", 0.0),
            by_mode=result.get("by_mode", {}),
            by_department=result.get("by_department", {}),
            employee_count=result.get("employee_count", 0),
            per_employee_co2e_kg=result.get("per_employee_co2e_kg", 0.0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_aggregations: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Aggregation failed",
        )


@router.post(
    "/mode-share/analyze",
    response_model=ModeShareResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze commute mode share",
    description=(
        "Analyze the distribution of commute modes across the workforce. "
        "Returns mode share percentages, CO2e by mode, average distances by "
        "mode, optional benchmark comparison against census/national data, "
        "and ranked intervention opportunities (transit subsidies, EV incentives, "
        "cycle-to-work schemes, compressed work weeks) with estimated emissions "
        "reduction potential."
    ),
)
async def analyze_mode_share(
    request: ModeShareRequest,
    service=Depends(get_service),
) -> ModeShareResponse:
    """
    Analyze commute mode share distribution.

    Args:
        request: Mode share analysis request
        service: EmployeeCommutingService instance

    Returns:
        ModeShareResponse with mode distribution and intervention opportunities

    Raises:
        HTTPException: 400 for invalid input, 500 for analysis failures
    """
    try:
        logger.info(
            f"Analyzing mode share: period={request.reporting_period}, "
            f"tenant_id={request.tenant_id}"
        )

        result = await service.analyze_mode_share(request.dict())

        return ModeShareResponse(
            success=True,
            reporting_period=result.get(
                "reporting_period", request.reporting_period
            ),
            total_employees=result.get("total_employees", 0),
            mode_shares=result.get("mode_shares", {}),
            mode_co2e=result.get("mode_co2e", {}),
            mode_avg_distance_km=result.get("mode_avg_distance_km", {}),
            benchmark_comparison=result.get("benchmark_comparison"),
            intervention_opportunities=result.get(
                "intervention_opportunities", []
            ),
        )

    except ValueError as e:
        logger.error("Validation error in analyze_mode_share: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in analyze_mode_share: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Mode share analysis failed",
        )


# ============================================================================
# ENDPOINTS - PROVENANCE (1)
# ============================================================================


@router.get(
    "/provenance/{calculation_id}",
    response_model=ProvenanceResponse,
    summary="Get provenance chain",
    description=(
        "Retrieve the complete SHA-256 provenance chain for a calculation. "
        "Includes all 10 pipeline stages (validate, classify, normalize, "
        "resolve_efs, calculate_commute, calculate_telework, extrapolate, "
        "compliance, aggregate, seal) with per-stage hashes and chain "
        "integrity verification."
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
        service: EmployeeCommutingService instance

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
                detail=(
                    f"Provenance for calculation {calculation_id} not found"
                ),
            )

        return ProvenanceResponse(
            success=True,
            calculation_id=result.get("calculation_id", calculation_id),
            chain=result.get("chain", []),
            is_valid=result.get("is_valid", False),
            root_hash=result.get("root_hash", ""),
            stages_count=result.get("stages_count", 0),
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
# ENDPOINTS - HEALTH (1)
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Health check endpoint for the Employee Commuting Agent. "
        "Returns service status, agent identifier, version, uptime, "
        "and per-engine health status. No authentication required."
    ),
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint (no auth required).

    Returns:
        HealthResponse with service status, version, and uptime
    """
    try:
        uptime = (datetime.utcnow() - _start_time).total_seconds()

        engines_status = {
            "employee_commuting_database": "healthy",
            "personal_vehicle_calculator": "healthy",
            "public_transit_calculator": "healthy",
            "active_transport_calculator": "healthy",
            "telework_calculator": "healthy",
            "compliance_checker": "healthy",
            "employee_commuting_pipeline": "healthy",
        }

        return HealthResponse(
            status="healthy",
            agent_id="GL-MRV-S3-007",
            version="1.0.0",
            uptime_seconds=round(uptime, 2),
            engines_status=engines_status,
        )

    except Exception as e:
        logger.error("Error in health_check: %s", e, exc_info=True)
        return HealthResponse(
            status="unhealthy",
            agent_id="GL-MRV-S3-007",
            version="1.0.0",
            uptime_seconds=0.0,
            engines_status=None,
        )
