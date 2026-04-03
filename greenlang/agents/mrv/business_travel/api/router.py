"""
Business Travel Agent API Router - AGENT-MRV-019

This module implements the FastAPI router for business travel emissions
calculations following GHG Protocol Scope 3 Category 6 requirements.

Provides 22 REST endpoints (20 core + health + stats) for:
- Emissions calculations (single, batch, flight, rail, road, hotel, spend)
- Emission factor lookup and transport mode metadata
- Airport database search
- Cabin class multiplier reference
- Compliance checking across 7 regulatory frameworks
- Uncertainty analysis (Monte Carlo, analytical, IPCC Tier 2)
- Aggregations by period, mode, and department
- Hot-spot analysis for route and mode optimization
- Provenance tracking with SHA-256 chain verification
- Health and statistics monitoring

Follows GreenLang's zero-hallucination principle with deterministic calculations.
All numeric outputs use deterministic formulas; no LLM calls in the calculation path.

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.agents.mrv.business_travel.api.router import router
    >>> app = FastAPI()
    >>> app.include_router(router)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
from pydantic import Field
from typing import List, Optional, Dict, Any
from decimal import Decimal
import logging
import uuid
from datetime import datetime
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/api/v1/business-travel",
    tags=["business-travel"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# SERVICE DEPENDENCY
# ============================================================================


_service_instance = None


def get_service():
    """
    Get or create BusinessTravelService singleton instance.

    Returns:
        BusinessTravelService instance

    Raises:
        HTTPException: If service initialization fails (503)
    """
    global _service_instance

    if _service_instance is None:
        try:
            from greenlang.agents.mrv.business_travel.service import BusinessTravelService
            _service_instance = BusinessTravelService()
            logger.info("BusinessTravelService initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize BusinessTravelService: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service initialization failed",
            )

    return _service_instance


# ============================================================================
# REQUEST MODELS
# ============================================================================


class CalculateRequest(GreenLangBase):
    """
    Request model for full pipeline business travel emissions calculation.

    Supports all transport modes with mode-specific parameters delegated
    to the service layer pipeline.

    Attributes:
        mode: Transport mode (air, rail, road, bus, taxi, ferry, motorcycle, hotel)
        trip_data: Mode-specific trip data dictionary
        trip_purpose: Purpose of the business trip
        department: Optional department for allocation
        cost_center: Optional cost center for allocation
    """

    mode: str = Field(
        ...,
        description="Transport mode (air, rail, road, bus, taxi, ferry, motorcycle, hotel)",
    )
    trip_data: Dict[str, Any] = Field(
        ...,
        description="Mode-specific trip data dictionary",
    )
    trip_purpose: str = Field(
        "business",
        description="Trip purpose (business, conference, client_visit, training, other)",
    )
    department: Optional[str] = Field(
        None,
        description="Department for cost allocation",
    )
    cost_center: Optional[str] = Field(
        None,
        description="Cost center for cost allocation",
    )


class BatchCalculateRequest(GreenLangBase):
    """
    Request model for batch business travel emissions calculations.

    Processes multiple trips in a single request with parallel execution
    and per-trip error isolation.

    Attributes:
        trips: List of trip data dictionaries (each must include 'mode')
        reporting_period: Reporting period identifier (e.g. '2024-Q1')
    """

    trips: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="List of trip data dictionaries",
    )
    reporting_period: str = Field(
        ...,
        description="Reporting period identifier (e.g. '2024-Q1')",
    )


class FlightCalculateRequest(GreenLangBase):
    """
    Request model for air travel emissions calculation.

    Uses great-circle distance between IATA airport pairs, DEFRA 2024
    distance-band emission factors, cabin class multipliers, and
    optional radiative forcing (RF) uplift.

    Attributes:
        origin_iata: Origin airport IATA code (3-letter)
        destination_iata: Destination airport IATA code (3-letter)
        cabin_class: Aircraft cabin class
        passengers: Number of passengers
        round_trip: Whether the trip is round-trip
        rf_option: Radiative forcing reporting option
    """

    origin_iata: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Origin airport IATA code",
    )
    destination_iata: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Destination airport IATA code",
    )
    cabin_class: str = Field(
        "economy",
        description="Cabin class (economy, premium_economy, business, first)",
    )
    passengers: int = Field(
        1,
        ge=1,
        le=500,
        description="Number of passengers",
    )
    round_trip: bool = Field(
        False,
        description="Whether the trip is round-trip (doubles distance)",
    )
    rf_option: str = Field(
        "with_rf",
        description="Radiative forcing option (with_rf, without_rf, both)",
    )


class RailCalculateRequest(GreenLangBase):
    """
    Request model for rail travel emissions calculation.

    Supports 8 rail types with DEFRA 2024 per-passenger-km factors
    including well-to-tank (WTT) emissions.

    Attributes:
        rail_type: Type of rail service
        distance_km: Travel distance in kilometres
        passengers: Number of passengers
    """

    rail_type: str = Field(
        ...,
        description=(
            "Rail type (national, international, light_rail, underground, "
            "eurostar, high_speed, us_intercity, us_commuter)"
        ),
    )
    distance_km: float = Field(
        ...,
        gt=0,
        description="Travel distance in kilometres",
    )
    passengers: int = Field(
        1,
        ge=1,
        le=500,
        description="Number of passengers",
    )


class RoadCalculateRequest(GreenLangBase):
    """
    Request model for road transport emissions calculation.

    Supports distance-based (per vehicle-km) and fuel-based (per litre)
    calculation methods with 13 vehicle types.

    Attributes:
        vehicle_type: Road vehicle type (optional, defaults to car_average)
        distance_km: Travel distance in kilometres (for distance-based)
        fuel_type: Fuel type (for fuel-based calculation)
        litres: Fuel consumed in litres (for fuel-based calculation)
    """

    vehicle_type: Optional[str] = Field(
        None,
        description="Vehicle type (car_average, car_small_petrol, hybrid, taxi_regular, etc.)",
    )
    distance_km: Optional[float] = Field(
        None,
        gt=0,
        description="Travel distance in kilometres (distance-based method)",
    )
    fuel_type: Optional[str] = Field(
        None,
        description="Fuel type (petrol, diesel, lpg, cng, e85) for fuel-based method",
    )
    litres: Optional[float] = Field(
        None,
        gt=0,
        description="Fuel consumed in litres (fuel-based method)",
    )


class HotelCalculateRequest(GreenLangBase):
    """
    Request model for hotel accommodation emissions calculation.

    Uses country-specific room-night emission factors with hotel class
    adjustments from DEFRA 2024 and Cornell/STR benchmarks.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code or 'GLOBAL'
        room_nights: Number of room nights
        hotel_class: Hotel class/tier affecting emissions intensity
    """

    country_code: str = Field(
        "GLOBAL",
        min_length=2,
        max_length=6,
        description="ISO country code or 'GLOBAL' for global average",
    )
    room_nights: int = Field(
        ...,
        ge=1,
        le=365,
        description="Number of room nights",
    )
    hotel_class: str = Field(
        "standard",
        description="Hotel class (budget, standard, upscale, luxury)",
    )


class SpendCalculateRequest(GreenLangBase):
    """
    Request model for spend-based emissions calculation.

    Uses EEIO (Environmentally Extended Input-Output) factors with
    CPI deflation and margin removal for spend-to-emissions conversion.

    Attributes:
        naics_code: NAICS industry code for EEIO factor selection
        amount: Spend amount in the specified currency
        currency: ISO 4217 currency code
        reporting_year: Year for CPI deflation adjustment
    """

    naics_code: str = Field(
        ...,
        description="NAICS code for EEIO factor selection",
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
        description="ISO 4217 currency code",
    )
    reporting_year: int = Field(
        2024,
        ge=2000,
        le=2100,
        description="Reporting year for CPI deflation",
    )


class ComplianceCheckRequest(GreenLangBase):
    """
    Request model for multi-framework compliance checking.

    Checks calculation results against selected regulatory frameworks
    for completeness, boundary correctness, and disclosure requirements.

    Attributes:
        frameworks: List of framework identifiers to check against
        calculation_results: List of calculation result dicts to check
        rf_disclosed: Whether radiative forcing has been disclosed
        mode_breakdown_provided: Whether per-mode breakdown is provided
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
    rf_disclosed: bool = Field(
        False,
        description="Whether radiative forcing has been disclosed",
    )
    mode_breakdown_provided: bool = Field(
        False,
        description="Whether per-mode breakdown is provided",
    )


class UncertaintyRequest(GreenLangBase):
    """
    Request model for uncertainty analysis.

    Supports Monte Carlo simulation, analytical error propagation,
    and IPCC Tier 2 default uncertainty ranges.

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


class HotSpotRequest(GreenLangBase):
    """
    Request model for hot-spot analysis.

    Identifies top emission sources by route, mode, and department
    with Pareto-based reduction opportunity ranking.

    Attributes:
        calculation_results: Calculation results to analyze
        top_n: Number of top hot-spots to return
    """

    calculation_results: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        description="Calculation results to analyze for hot-spots",
    )
    top_n: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of top hot-spots to return",
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class CalculateResponse(GreenLangBase):
    """Response model for single pipeline calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    mode: str = Field(..., description="Transport mode used")
    method: str = Field(..., description="Calculation method applied")
    total_co2e_kg: float = Field(..., description="Total CO2e emissions in kg")
    co2e_without_rf_kg: Optional[float] = Field(
        None, description="CO2e without radiative forcing (air only)"
    )
    co2e_with_rf_kg: Optional[float] = Field(
        None, description="CO2e with radiative forcing (air only)"
    )
    wtt_co2e_kg: float = Field(
        ..., description="Well-to-tank CO2e emissions in kg"
    )
    dqi_score: Optional[float] = Field(
        None, description="Data quality indicator score (1-5)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class BatchCalculateResponse(GreenLangBase):
    """Response model for batch calculation."""

    batch_id: str = Field(..., description="Unique batch UUID")
    results: List[Dict[str, Any]] = Field(
        ..., description="Individual calculation results"
    )
    total_co2e_kg: float = Field(
        ..., description="Total CO2e for all trips in batch"
    )
    count: int = Field(..., description="Number of successful calculations")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-trip error details"
    )
    reporting_period: str = Field(
        ..., description="Reporting period identifier"
    )


class FlightCalculateResponse(GreenLangBase):
    """Response model for flight-specific calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    origin_iata: str = Field(..., description="Origin airport IATA code")
    destination_iata: str = Field(
        ..., description="Destination airport IATA code"
    )
    distance_km: float = Field(
        ..., description="Great-circle distance in km"
    )
    distance_band: str = Field(
        ..., description="DEFRA distance band (domestic, short_haul, long_haul)"
    )
    cabin_class: str = Field(..., description="Cabin class used")
    class_multiplier: float = Field(
        ..., description="Cabin class multiplier applied"
    )
    co2e_without_rf_kg: float = Field(
        ..., description="CO2e without radiative forcing in kg"
    )
    co2e_with_rf_kg: float = Field(
        ..., description="CO2e with radiative forcing in kg"
    )
    wtt_co2e_kg: float = Field(
        ..., description="Well-to-tank CO2e in kg"
    )
    total_co2e_kg: float = Field(
        ..., description="Total CO2e based on rf_option selection"
    )
    rf_option: str = Field(
        ..., description="Radiative forcing option applied"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )


class EmissionFactorResponse(GreenLangBase):
    """Response model for a single emission factor."""

    mode: str = Field(..., description="Transport mode")
    vehicle_type: Optional[str] = Field(
        None, description="Vehicle/service subtype"
    )
    ef_value: float = Field(
        ..., description="Emission factor value (kgCO2e per unit)"
    )
    wtt_value: float = Field(
        ..., description="Well-to-tank factor value (kgCO2e per unit)"
    )
    unit: str = Field(
        ..., description="Unit basis (per-pkm, per-vkm, per-room-night, etc.)"
    )
    source: str = Field(
        ..., description="Factor source (DEFRA, ICAO, EPA, EEIO, etc.)"
    )


class EmissionFactorListResponse(GreenLangBase):
    """Response model for emission factor listing."""

    factors: List[EmissionFactorResponse] = Field(
        ..., description="List of emission factors"
    )
    count: int = Field(..., description="Total factor count returned")


class AirportResponse(GreenLangBase):
    """Response model for a single airport."""

    iata_code: str = Field(..., description="IATA 3-letter airport code")
    name: str = Field(..., description="Airport name")
    city: Optional[str] = Field(None, description="City name")
    country_code: str = Field(
        ..., description="ISO 3166-1 alpha-2 country code"
    )
    latitude: float = Field(..., description="Latitude in decimal degrees")
    longitude: float = Field(..., description="Longitude in decimal degrees")


class AirportListResponse(GreenLangBase):
    """Response model for airport search results."""

    airports: List[AirportResponse] = Field(
        ..., description="Matching airports"
    )
    count: int = Field(..., description="Number of airports returned")


class TransportModeResponse(GreenLangBase):
    """Response model for supported transport modes."""

    modes: List[Dict[str, Any]] = Field(
        ..., description="List of transport modes with metadata"
    )


class CabinClassResponse(GreenLangBase):
    """Response model for cabin class multipliers."""

    classes: List[Dict[str, Any]] = Field(
        ..., description="Cabin classes with multipliers"
    )


class ComplianceCheckResponse(GreenLangBase):
    """Response model for compliance check."""

    results: List[Dict[str, Any]] = Field(
        ..., description="Per-framework compliance results"
    )
    overall_status: str = Field(
        ..., description="Overall compliance status (pass, fail, warning)"
    )
    overall_score: float = Field(
        ..., description="Overall compliance score (0.0-1.0)"
    )


class UncertaintyResponse(GreenLangBase):
    """Response model for uncertainty analysis."""

    mean: float = Field(..., description="Mean CO2e (kg)")
    std_dev: float = Field(..., description="Standard deviation (kg)")
    ci_lower: float = Field(
        ..., description="Confidence interval lower bound (kg)"
    )
    ci_upper: float = Field(
        ..., description="Confidence interval upper bound (kg)"
    )
    method: str = Field(..., description="Uncertainty method used")
    iterations: int = Field(
        ..., description="Iterations performed (0 for non-Monte Carlo)"
    )


class AggregationResponse(GreenLangBase):
    """Response model for aggregated emissions."""

    period: str = Field(..., description="Aggregation period identifier")
    total_co2e_kg: float = Field(
        ..., description="Total CO2e for the period (kg)"
    )
    by_mode: Dict[str, float] = Field(
        ..., description="CO2e breakdown by transport mode"
    )
    by_department: Dict[str, float] = Field(
        ..., description="CO2e breakdown by department"
    )
    trip_count: int = Field(..., description="Total number of trips")


class CalculationListResponse(GreenLangBase):
    """Response model for paginated calculation listing."""

    calculations: List[Dict[str, Any]] = Field(
        ..., description="Calculation summaries"
    )
    count: int = Field(..., description="Total matching calculations")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")


class CalculationDetailResponse(GreenLangBase):
    """Response model for single calculation detail."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    mode: str = Field(..., description="Transport mode")
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


class DeleteResponse(GreenLangBase):
    """Response model for soft deletion."""

    calculation_id: str = Field(..., description="Deleted calculation UUID")
    deleted: bool = Field(..., description="Whether deletion succeeded")
    message: str = Field(..., description="Human-readable status message")


class ProvenanceResponse(GreenLangBase):
    """Response model for provenance chain verification."""

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


class HotSpotResponse(GreenLangBase):
    """Response model for hot-spot analysis."""

    top_routes: List[Dict[str, Any]] = Field(
        ..., description="Top emission routes (origin-destination pairs)"
    )
    top_modes: Dict[str, float] = Field(
        ..., description="CO2e totals by transport mode"
    )
    reduction_opportunities: List[Dict[str, Any]] = Field(
        ...,
        description="Ranked reduction opportunities with estimated savings",
    )


class HealthResponse(GreenLangBase):
    """Response model for health check."""

    status: str = Field(..., description="Service health status")
    agent_id: str = Field(..., description="Agent identifier")
    version: str = Field(..., description="Agent version")
    uptime_seconds: float = Field(
        ..., description="Seconds since service start"
    )


class StatsResponse(GreenLangBase):
    """Response model for agent statistics."""

    total_calculations: int = Field(
        ..., description="Total calculations processed"
    )
    total_co2e_kg: float = Field(
        ..., description="Total CO2e across all calculations (kg)"
    )
    total_flights: int = Field(
        ..., description="Total air travel calculations"
    )
    total_ground_trips: int = Field(
        ..., description="Total ground transport calculations"
    )
    total_hotel_nights: int = Field(
        ..., description="Total hotel night calculations"
    )


# ============================================================================
# MODULE-LEVEL TRACKING
# ============================================================================

_start_time: datetime = datetime.utcnow()


# ============================================================================
# ENDPOINTS - CALCULATIONS (7)
# ============================================================================


@router.post(
    "/calculate",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate business travel emissions",
    description=(
        "Calculate GHG emissions for a single business trip through the full "
        "10-stage pipeline. Supports all transport modes with mode-specific "
        "parameters. Returns deterministic results with SHA-256 provenance hash."
    ),
)
async def calculate_emissions(
    request: CalculateRequest,
    service=Depends(get_service),
) -> CalculateResponse:
    """
    Calculate business travel emissions through the full pipeline.

    Args:
        request: Calculation request with mode and trip data
        service: BusinessTravelService instance

    Returns:
        CalculateResponse with emissions and provenance hash

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating emissions for mode={request.mode}, "
            f"purpose={request.trip_purpose}"
        )

        result = await service.calculate(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculateResponse(
            calculation_id=calculation_id,
            mode=result.get("mode", request.mode),
            method=result.get("method", "distance_based"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2e_without_rf_kg=result.get("co2e_without_rf_kg"),
            co2e_with_rf_kg=result.get("co2e_with_rf_kg"),
            wtt_co2e_kg=result.get("wtt_co2e_kg", 0.0),
            dqi_score=result.get("dqi_score"),
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
            detail="Calculation failed",
        )


@router.post(
    "/calculate/batch",
    response_model=BatchCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch calculate business travel emissions",
    description=(
        "Calculate GHG emissions for multiple business trips in a single "
        "request. Processes up to 10,000 trips with parallel execution and "
        "per-trip error isolation. Returns aggregated totals with individual "
        "results and any per-trip errors."
    ),
)
async def calculate_batch_emissions(
    request: BatchCalculateRequest,
    service=Depends(get_service),
) -> BatchCalculateResponse:
    """
    Calculate batch business travel emissions.

    Args:
        request: Batch calculation request with trip list
        service: BusinessTravelService instance

    Returns:
        BatchCalculateResponse with aggregated and per-trip results

    Raises:
        HTTPException: 400 for validation errors, 500 for batch failures
    """
    try:
        logger.info(
            f"Calculating batch emissions for {len(request.trips)} trips, "
            f"period={request.reporting_period}"
        )

        result = await service.calculate_batch(request.dict())
        batch_id = result.get("batch_id", str(uuid.uuid4()))

        return BatchCalculateResponse(
            batch_id=batch_id,
            results=result.get("results", []),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            count=result.get("count", 0),
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
    "/calculate/flight",
    response_model=FlightCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate air travel emissions",
    description=(
        "Calculate GHG emissions for a single flight using great-circle "
        "distance between IATA airport pairs, DEFRA 2024 distance-band "
        "emission factors, cabin class multipliers, and optional radiative "
        "forcing (RF) uplift. Returns both with-RF and without-RF values."
    ),
)
async def calculate_flight_emissions(
    request: FlightCalculateRequest,
    service=Depends(get_service),
) -> FlightCalculateResponse:
    """
    Calculate air travel emissions for an airport-pair flight.

    Args:
        request: Flight calculation request with IATA codes
        service: BusinessTravelService instance

    Returns:
        FlightCalculateResponse with distance, band, and emissions

    Raises:
        HTTPException: 400 for invalid IATA codes, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating flight emissions: "
            f"{request.origin_iata}->{request.destination_iata}, "
            f"cabin={request.cabin_class}, rf={request.rf_option}"
        )

        result = await service.calculate_flight(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return FlightCalculateResponse(
            calculation_id=calculation_id,
            origin_iata=result.get("origin_iata", request.origin_iata),
            destination_iata=result.get(
                "destination_iata", request.destination_iata
            ),
            distance_km=result.get("distance_km", 0.0),
            distance_band=result.get("distance_band", "unknown"),
            cabin_class=result.get("cabin_class", request.cabin_class),
            class_multiplier=result.get("class_multiplier", 1.0),
            co2e_without_rf_kg=result.get("co2e_without_rf_kg", 0.0),
            co2e_with_rf_kg=result.get("co2e_with_rf_kg", 0.0),
            wtt_co2e_kg=result.get("wtt_co2e_kg", 0.0),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            rf_option=result.get("rf_option", request.rf_option),
            provenance_hash=result.get("provenance_hash", ""),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_flight_emissions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_flight_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Flight calculation failed",
        )


@router.post(
    "/calculate/rail",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate rail travel emissions",
    description=(
        "Calculate GHG emissions for rail travel using DEFRA 2024 "
        "per-passenger-km factors. Supports 8 rail types including national, "
        "international, light rail, underground, Eurostar, high-speed, and "
        "US intercity/commuter rail."
    ),
)
async def calculate_rail_emissions(
    request: RailCalculateRequest,
    service=Depends(get_service),
) -> CalculateResponse:
    """
    Calculate rail travel emissions.

    Args:
        request: Rail calculation request with type and distance
        service: BusinessTravelService instance

    Returns:
        CalculateResponse with rail emissions

    Raises:
        HTTPException: 400 for invalid rail type, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating rail emissions: type={request.rail_type}, "
            f"distance={request.distance_km}km"
        )

        result = await service.calculate_rail(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculateResponse(
            calculation_id=calculation_id,
            mode="rail",
            method=result.get("method", "distance_based"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2e_without_rf_kg=None,
            co2e_with_rf_kg=None,
            wtt_co2e_kg=result.get("wtt_co2e_kg", 0.0),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_rail_emissions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_rail_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Rail calculation failed",
        )


@router.post(
    "/calculate/road",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate road transport emissions",
    description=(
        "Calculate GHG emissions for road transport. Supports distance-based "
        "(per vehicle-km) and fuel-based (per litre) methods with 13 vehicle "
        "types including petrol/diesel cars, hybrids, BEVs, taxis, and "
        "motorcycles. DEFRA 2024 emission factors."
    ),
)
async def calculate_road_emissions(
    request: RoadCalculateRequest,
    service=Depends(get_service),
) -> CalculateResponse:
    """
    Calculate road transport emissions.

    Args:
        request: Road calculation request with vehicle type and distance/fuel
        service: BusinessTravelService instance

    Returns:
        CalculateResponse with road emissions

    Raises:
        HTTPException: 400 for missing distance/fuel data, 500 for failures
    """
    try:
        logger.info(
            f"Calculating road emissions: vehicle={request.vehicle_type}, "
            f"distance={request.distance_km}km"
        )

        result = await service.calculate_road(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculateResponse(
            calculation_id=calculation_id,
            mode="road",
            method=result.get("method", "distance_based"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2e_without_rf_kg=None,
            co2e_with_rf_kg=None,
            wtt_co2e_kg=result.get("wtt_co2e_kg", 0.0),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_road_emissions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_road_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Road calculation failed",
        )


@router.post(
    "/calculate/hotel",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate hotel accommodation emissions",
    description=(
        "Calculate GHG emissions for hotel accommodation using country-specific "
        "room-night emission factors. Supports 16 countries plus global average "
        "with 4 hotel class tiers (budget, standard, upscale, luxury). "
        "DEFRA 2024 and Cornell/STR benchmark factors."
    ),
)
async def calculate_hotel_emissions(
    request: HotelCalculateRequest,
    service=Depends(get_service),
) -> CalculateResponse:
    """
    Calculate hotel accommodation emissions.

    Args:
        request: Hotel calculation request with country and room nights
        service: BusinessTravelService instance

    Returns:
        CalculateResponse with hotel emissions

    Raises:
        HTTPException: 400 for invalid country code, 500 for failures
    """
    try:
        logger.info(
            f"Calculating hotel emissions: country={request.country_code}, "
            f"nights={request.room_nights}, class={request.hotel_class}"
        )

        result = await service.calculate_hotel(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculateResponse(
            calculation_id=calculation_id,
            mode="hotel",
            method=result.get("method", "average_data"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2e_without_rf_kg=None,
            co2e_with_rf_kg=None,
            wtt_co2e_kg=result.get("wtt_co2e_kg", 0.0),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_hotel_emissions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_hotel_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hotel calculation failed",
        )


@router.post(
    "/calculate/spend",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate spend-based emissions",
    description=(
        "Calculate GHG emissions using spend-based EEIO (Environmentally "
        "Extended Input-Output) factors. Applies CPI deflation to base year, "
        "currency conversion to USD, and margin removal. Supports 10 NAICS "
        "travel-related industry codes and 12 currencies."
    ),
)
async def calculate_spend_emissions(
    request: SpendCalculateRequest,
    service=Depends(get_service),
) -> CalculateResponse:
    """
    Calculate spend-based emissions using EEIO factors.

    Args:
        request: Spend calculation request with NAICS code and amount
        service: BusinessTravelService instance

    Returns:
        CalculateResponse with spend-based emissions

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

        return CalculateResponse(
            calculation_id=calculation_id,
            mode="spend",
            method=result.get("method", "spend_based"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2e_without_rf_kg=None,
            co2e_with_rf_kg=None,
            wtt_co2e_kg=result.get("wtt_co2e_kg", 0.0),
            dqi_score=result.get("dqi_score"),
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


# ============================================================================
# ENDPOINTS - CALCULATION CRUD (3)
# ============================================================================


@router.get(
    "/calculations/{calculation_id}",
    response_model=CalculationDetailResponse,
    summary="Get calculation detail",
    description=(
        "Retrieve detailed information for a specific business travel "
        "calculation including full input/output payload, provenance hash, "
        "and calculation metadata."
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
        service: BusinessTravelService instance

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
        "Retrieve a paginated list of business travel calculations. "
        "Supports filtering by mode, department, and date range. "
        "Returns summary information for each calculation."
    ),
)
async def list_calculations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Results per page"),
    mode: Optional[str] = Query(None, description="Filter by transport mode"),
    department: Optional[str] = Query(
        None, description="Filter by department"
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
    List business travel calculations with filtering and pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of results per page
        mode: Optional transport mode filter
        department: Optional department filter
        from_date: Optional start date filter
        to_date: Optional end date filter
        service: BusinessTravelService instance

    Returns:
        CalculationListResponse with paginated results

    Raises:
        HTTPException: 500 for listing failures
    """
    try:
        logger.info(
            f"Listing calculations: page={page}, size={page_size}, "
            f"mode={mode}"
        )

        filters = {
            "page": page,
            "page_size": page_size,
            "mode": mode,
            "department": department,
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
        "Soft-delete a specific business travel calculation. "
        "Marks the calculation as deleted with audit trail; "
        "data is retained for regulatory compliance."
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
        service: BusinessTravelService instance

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
# ENDPOINTS - EMISSION FACTORS & METADATA (5)
# ============================================================================


@router.get(
    "/emission-factors",
    response_model=EmissionFactorListResponse,
    summary="List emission factors",
    description=(
        "Retrieve available emission factors for business travel. "
        "Supports filtering by transport mode and data source. "
        "Returns DEFRA 2024, ICAO, EPA, and EEIO factors."
    ),
)
async def list_emission_factors(
    mode: Optional[str] = Query(
        None, description="Filter by transport mode"
    ),
    source: Optional[str] = Query(
        None, description="Filter by EF source (defra, icao, epa, eeio)"
    ),
    service=Depends(get_service),
) -> EmissionFactorListResponse:
    """
    List emission factors with optional filtering.

    Args:
        mode: Optional transport mode filter
        source: Optional data source filter
        service: BusinessTravelService instance

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
    summary="Get emission factors by mode",
    description=(
        "Retrieve emission factors for a specific transport mode. "
        "Returns all vehicle/service subtypes and their EF values "
        "including well-to-tank (WTT) factors."
    ),
)
async def get_emission_factors_by_mode(
    mode: str = Path(
        ...,
        description="Transport mode (air, rail, road, bus, taxi, ferry, motorcycle, hotel)",
    ),
    service=Depends(get_service),
) -> EmissionFactorListResponse:
    """
    Get emission factors for a specific transport mode.

    Args:
        mode: Transport mode identifier
        service: BusinessTravelService instance

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
    "/airports",
    response_model=AirportListResponse,
    summary="Search airports",
    description=(
        "Search the airport database by IATA code, name, or city. "
        "Returns matching airports with coordinates for great-circle "
        "distance calculation. Database includes 50 major airports."
    ),
)
async def search_airports(
    q: Optional[str] = Query(
        None,
        min_length=2,
        description="Search query (IATA code, airport name, or city)",
    ),
    country_code: Optional[str] = Query(
        None, description="Filter by ISO country code"
    ),
    service=Depends(get_service),
) -> AirportListResponse:
    """
    Search airports by query string or country code.

    Args:
        q: Search query (IATA code, name, or city)
        country_code: Optional country code filter
        service: BusinessTravelService instance

    Returns:
        AirportListResponse with matching airports

    Raises:
        HTTPException: 500 for search failures
    """
    try:
        logger.info(
            f"Searching airports: q={q}, country={country_code}"
        )

        filters = {"q": q, "country_code": country_code}
        result = await service.search_airports(filters)

        return AirportListResponse(
            airports=result.get("airports", []),
            count=result.get("count", 0),
        )

    except Exception as e:
        logger.error("Error in search_airports: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Airport search failed",
        )


@router.get(
    "/transport-modes",
    response_model=TransportModeResponse,
    summary="List transport modes",
    description=(
        "Retrieve all supported transport modes with metadata including "
        "available vehicle subtypes, calculation methods, and EF sources."
    ),
)
async def list_transport_modes(
    service=Depends(get_service),
) -> TransportModeResponse:
    """
    List all supported transport modes.

    Args:
        service: BusinessTravelService instance

    Returns:
        TransportModeResponse with mode metadata

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info("Listing transport modes")

        result = await service.list_transport_modes()

        return TransportModeResponse(
            modes=result.get("modes", []),
        )

    except Exception as e:
        logger.error("Error in list_transport_modes: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list transport modes",
        )


@router.get(
    "/cabin-classes",
    response_model=CabinClassResponse,
    summary="List cabin classes with multipliers",
    description=(
        "Retrieve all aircraft cabin classes with their emissions "
        "multipliers relative to economy class (DEFRA 2024). "
        "Economy=1.0, Premium Economy=1.6, Business=2.9, First=4.0."
    ),
)
async def list_cabin_classes(
    service=Depends(get_service),
) -> CabinClassResponse:
    """
    List cabin classes with multipliers.

    Args:
        service: BusinessTravelService instance

    Returns:
        CabinClassResponse with class multiplier data

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info("Listing cabin classes")

        result = await service.list_cabin_classes()

        return CabinClassResponse(
            classes=result.get("classes", []),
        )

    except Exception as e:
        logger.error("Error in list_cabin_classes: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list cabin classes",
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
        "Check business travel calculation results against one or more "
        "regulatory frameworks. Validates completeness, boundary correctness, "
        "radiative forcing disclosure, mode breakdown, and DQI requirements. "
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
        service: BusinessTravelService instance

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
            results=result.get("results", []),
            overall_status=result.get("overall_status", "unknown"),
            overall_score=result.get("overall_score", 0.0),
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
        "Perform uncertainty analysis on business travel emissions "
        "calculations. Supports Monte Carlo simulation, analytical error "
        "propagation, and IPCC Tier 2 default ranges. Returns mean, "
        "standard deviation, and confidence intervals."
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
        service: BusinessTravelService instance

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
            mean=result.get("mean", 0.0),
            std_dev=result.get("std_dev", 0.0),
            ci_lower=result.get("ci_lower", 0.0),
            ci_upper=result.get("ci_upper", 0.0),
            method=result.get("method", request.method),
            iterations=result.get("iterations", request.iterations),
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
# ENDPOINTS - AGGREGATION & ANALYSIS (2)
# ============================================================================


@router.get(
    "/aggregations/{period}",
    response_model=AggregationResponse,
    summary="Get aggregated emissions",
    description=(
        "Retrieve aggregated business travel emissions for a specified "
        "period. Returns totals with breakdowns by transport mode and "
        "department. Supports daily, weekly, monthly, quarterly, and "
        "annual aggregation periods."
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
    service=Depends(get_service),
) -> AggregationResponse:
    """
    Get aggregated emissions for a specified period.

    Args:
        period: Aggregation period identifier
        from_date: Optional start date filter
        to_date: Optional end date filter
        department: Optional department filter
        service: BusinessTravelService instance

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
        }

        result = await service.get_aggregations(filters)

        return AggregationResponse(
            period=period,
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            by_mode=result.get("by_mode", {}),
            by_department=result.get("by_department", {}),
            trip_count=result.get("trip_count", 0),
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
    "/hot-spots/analyze",
    response_model=HotSpotResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze emission hot-spots",
    description=(
        "Identify top emission sources from business travel data using "
        "Pareto analysis. Returns top routes by CO2e, mode-level totals, "
        "and ranked reduction opportunities with estimated savings from "
        "mode-switching or policy changes."
    ),
)
async def analyze_hot_spots(
    request: HotSpotRequest,
    service=Depends(get_service),
) -> HotSpotResponse:
    """
    Analyze emission hot-spots for reduction opportunities.

    Args:
        request: Hot-spot analysis request with calculation results
        service: BusinessTravelService instance

    Returns:
        HotSpotResponse with top routes, modes, and reduction opportunities

    Raises:
        HTTPException: 400 for invalid input, 500 for analysis failures
    """
    try:
        logger.info(
            f"Analyzing hot-spots: {len(request.calculation_results)} results, "
            f"top_n={request.top_n}"
        )

        result = await service.analyze_hot_spots(request.dict())

        return HotSpotResponse(
            top_routes=result.get("top_routes", []),
            top_modes=result.get("top_modes", {}),
            reduction_opportunities=result.get(
                "reduction_opportunities", []
            ),
        )

    except ValueError as e:
        logger.error("Validation error in analyze_hot_spots: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in analyze_hot_spots: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hot-spot analysis failed",
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
        "resolve_efs, calculate_flights, calculate_ground, allocate, "
        "compliance, aggregate, seal) with per-stage hashes and verification."
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
        service: BusinessTravelService instance

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
# ENDPOINTS - HEALTH & STATS (2)
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Health check endpoint for the Business Travel Agent. "
        "Returns service status, agent identifier, version, and uptime. "
        "No authentication required."
    ),
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint (no auth required).

    Returns:
        HealthResponse with service status and uptime
    """
    try:
        uptime = (datetime.utcnow() - _start_time).total_seconds()

        return HealthResponse(
            status="healthy",
            agent_id="GL-MRV-S3-006",
            version="1.0.0",
            uptime_seconds=round(uptime, 2),
        )

    except Exception as e:
        logger.error("Error in health_check: %s", e, exc_info=True)
        return HealthResponse(
            status="unhealthy",
            agent_id="GL-MRV-S3-006",
            version="1.0.0",
            uptime_seconds=0.0,
        )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Agent statistics",
    description=(
        "Retrieve aggregate statistics for the Business Travel Agent "
        "including total calculations, total CO2e, and breakdowns by "
        "transport category (flights, ground trips, hotel nights)."
    ),
)
async def get_stats(
    service=Depends(get_service),
) -> StatsResponse:
    """
    Get aggregate statistics for the Business Travel Agent.

    Args:
        service: BusinessTravelService instance

    Returns:
        StatsResponse with aggregate statistics

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info("Getting agent statistics")

        result = await service.get_stats()

        return StatsResponse(
            total_calculations=result.get("total_calculations", 0),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            total_flights=result.get("total_flights", 0),
            total_ground_trips=result.get("total_ground_trips", 0),
            total_hotel_nights=result.get("total_hotel_nights", 0),
        )

    except Exception as e:
        logger.error("Error in get_stats: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics",
        )
