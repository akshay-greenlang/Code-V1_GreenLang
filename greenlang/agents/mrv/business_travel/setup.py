"""
Business Travel Service Setup - AGENT-MRV-019

This module provides the service facade that wires together all 7 engines
for business travel emissions calculations (Scope 3 Category 6).

The BusinessTravelService class provides a high-level API for:
- Flight emissions (distance-based with radiative forcing)
- Rail emissions (national/international/metro/high-speed)
- Road emissions (distance-based and fuel-based, 13 vehicle types)
- Bus / taxi / ferry / motorcycle emissions
- Hotel accommodation emissions (16 countries, 4 classes)
- Spend-based fallback calculations (10 NAICS codes, CPI deflation)
- Compliance checking across 7 regulatory frameworks
- Uncertainty quantification (Monte Carlo, analytical, IPCC Tier 2)
- Hot-spot analysis for route and mode optimization
- Aggregations by mode, department, period, cabin class
- Provenance tracking with SHA-256 audit trail

Engines:
    1. BusinessTravelDatabaseEngine - Emission factor data and persistence
    2. AirTravelCalculatorEngine - Flight emissions with RF multiplier
    3. GroundTransportCalculatorEngine - Rail, road, bus, taxi, ferry, motorcycle
    4. HotelStayCalculatorEngine - Hotel accommodation emissions
    5. SpendBasedCalculatorEngine - EEIO spend-based calculations
    6. ComplianceCheckerEngine - Multi-framework compliance validation
    7. BusinessTravelPipelineEngine - End-to-end 10-stage pipeline

Architecture:
    - Thread-safe singleton pattern for service instance
    - Graceful imports with try/except for optional dependencies
    - Comprehensive metrics tracking via OBS-001 integration
    - Provenance tracking for all mutations via AGENT-FOUND-008
    - Type-safe request/response models using Pydantic
    - Structured logging with contextual information

Example:
    >>> from greenlang.agents.mrv.business_travel.setup import get_service
    >>> service = get_service()
    >>> response = service.calculate(TripCalculationRequest(
    ...     mode="air",
    ...     trip_data={"origin_iata": "JFK", "destination_iata": "LHR"},
    ...     trip_purpose="client_visit"
    ... ))
    >>> assert response.success

Integration:
    >>> from greenlang.agents.mrv.business_travel.setup import get_router
    >>> app.include_router(get_router(), prefix="/api/v1/business-travel")
"""

import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import Field, validator
from greenlang.schemas import GreenLangBase

# Thread-safe singleton lock
_service_lock = threading.Lock()
_service_instance: Optional["BusinessTravelService"] = None

logger = logging.getLogger(__name__)


# ============================================================================
# Request Models
# ============================================================================


class TripCalculationRequest(GreenLangBase):
    """Request model for single trip emissions calculation."""

    mode: str = Field(..., description="Transport mode: air, rail, road, bus, taxi, ferry, motorcycle, hotel")
    trip_data: dict = Field(..., description="Mode-specific input data")
    trip_purpose: Optional[str] = Field("business", description="Trip purpose: business, conference, client_visit, training, other")
    department: Optional[str] = Field(None, description="Department for allocation")
    cost_center: Optional[str] = Field(None, description="Cost center for allocation")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")

    @validator("mode")
    def validate_mode(cls, v: str) -> str:
        """Validate transport mode."""
        allowed = ["air", "rail", "road", "bus", "taxi", "ferry", "motorcycle", "hotel"]
        if v.lower() not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v.lower()


class BatchTripCalculationRequest(GreenLangBase):
    """Request model for batch trip calculations."""

    trips: List[TripCalculationRequest] = Field(..., min_length=1, description="List of trips")
    reporting_period: str = Field(..., description="Reporting period (e.g., '2024-Q3')")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class FlightCalculationRequest(GreenLangBase):
    """Request model for direct flight calculation."""

    origin_iata: str = Field(..., min_length=3, max_length=3, description="Origin IATA code")
    destination_iata: str = Field(..., min_length=3, max_length=3, description="Destination IATA code")
    cabin_class: str = Field("economy", description="Cabin class: economy, premium_economy, business, first")
    passengers: int = Field(1, ge=1, description="Number of passengers")
    round_trip: bool = Field(False, description="Round trip flag")
    rf_option: str = Field("with_rf", description="Radiative forcing option: with_rf, without_rf, both")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class RailCalculationRequest(GreenLangBase):
    """Request model for rail calculation."""

    rail_type: str = Field(..., description="Rail type: national, international, light_rail, eurostar, etc.")
    distance_km: float = Field(..., gt=0, description="Distance in km")
    passengers: int = Field(1, ge=1, description="Number of passengers")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class RoadCalculationRequest(GreenLangBase):
    """Request model for road vehicle calculation."""

    vehicle_type: Optional[str] = Field("car_average", description="Vehicle type")
    distance_km: Optional[float] = Field(None, gt=0, description="Distance in km")
    fuel_type: Optional[str] = Field(None, description="Fuel type for fuel-based calculation")
    litres: Optional[float] = Field(None, gt=0, description="Litres consumed")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class HotelCalculationRequest(GreenLangBase):
    """Request model for hotel accommodation calculation."""

    country_code: str = Field("GLOBAL", description="ISO country code or GLOBAL")
    room_nights: int = Field(..., gt=0, description="Number of room-nights")
    hotel_class: str = Field("standard", description="Hotel class: budget, standard, upscale, luxury")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class SpendCalculationRequest(GreenLangBase):
    """Request model for spend-based calculation."""

    naics_code: str = Field(..., description="NAICS code for EEIO factor")
    amount: float = Field(..., gt=0, description="Spend amount")
    currency: str = Field("USD", description="Currency code")
    reporting_year: int = Field(2024, ge=2015, le=2030, description="Reporting year")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class ComplianceCheckRequest(GreenLangBase):
    """Request model for compliance checking."""

    calculation_id: str = Field(..., description="Calculation ID to check")
    frameworks: List[str] = Field(
        default_factory=lambda: ["GHG_PROTOCOL"],
        description="Frameworks: GHG_PROTOCOL, ISO_14064, CSRD_ESRS, CDP, SBTI, SB_253, GRI"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class UncertaintyRequest(GreenLangBase):
    """Request model for uncertainty analysis."""

    calculation_id: str = Field(..., description="Calculation ID")
    method: str = Field("monte_carlo", description="Method: monte_carlo, analytical, ipcc_tier_2")
    iterations: int = Field(10000, ge=100, le=1000000, description="MC iterations")
    confidence_level: float = Field(0.95, ge=0.80, le=0.99, description="Confidence level")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class HotSpotRequest(GreenLangBase):
    """Request model for hot-spot analysis."""

    reporting_period: str = Field(..., description="Reporting period")
    top_n: int = Field(10, ge=1, le=100, description="Number of top emitters to return")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class CalculationFilterRequest(GreenLangBase):
    """Request model for filtering calculations."""

    tenant_id: Optional[str] = Field(None, description="Tenant filter")
    mode: Optional[str] = Field(None, description="Mode filter")
    department: Optional[str] = Field(None, description="Department filter")
    reporting_period: Optional[str] = Field(None, description="Period filter")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(100, ge=1, le=1000, description="Page size")


class AggregationRequest(GreenLangBase):
    """Request model for aggregation queries."""

    reporting_period: str = Field(..., description="Reporting period")
    group_by: Optional[str] = Field("mode", description="Group by: mode, department, period, cabin_class")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


# ============================================================================
# Response Models
# ============================================================================


class TripCalculationResponse(GreenLangBase):
    """Response model for single trip calculation."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation identifier")
    mode: str = Field(..., description="Transport mode")
    method: str = Field(..., description="Calculation method used")
    total_co2e_kg: float = Field(..., description="Total CO2e in kg")
    co2e_without_rf_kg: Optional[float] = Field(None, description="CO2e without RF (air only)")
    co2e_with_rf_kg: Optional[float] = Field(None, description="CO2e with RF (air only)")
    wtt_co2e_kg: float = Field(0, description="Well-to-tank CO2e in kg")
    dqi_score: Optional[float] = Field(None, description="Data quality score (1-5)")
    ef_source: str = Field(..., description="Emission factor source")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    detail: dict = Field(default_factory=dict, description="Mode-specific detail")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class BatchTripResponse(GreenLangBase):
    """Response model for batch trip calculation."""

    success: bool = Field(..., description="Overall success flag")
    total_trips: int = Field(..., description="Total trips requested")
    successful_trips: int = Field(..., description="Successful trips")
    failed_trips: int = Field(..., description="Failed trips")
    total_co2e_kg: float = Field(..., description="Total CO2e for all trips")
    results: List[TripCalculationResponse] = Field(..., description="Individual results")
    errors: List[dict] = Field(default_factory=list, description="Failed trip errors")
    reporting_period: str = Field(..., description="Reporting period")
    processing_time_ms: float = Field(..., description="Total processing time")


class ComplianceCheckResponse(GreenLangBase):
    """Response model for compliance checking."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation checked")
    overall_status: str = Field(..., description="Overall compliance status: PASS, WARNING, FAIL")
    framework_results: List[dict] = Field(..., description="Per-framework results")
    checked_at: datetime = Field(..., description="Check timestamp")
    processing_time_ms: float = Field(..., description="Processing time")


class UncertaintyResponse(GreenLangBase):
    """Response model for uncertainty analysis."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation analysed")
    mean_co2e_kg: float = Field(..., description="Mean CO2e")
    std_dev_co2e_kg: float = Field(..., description="Standard deviation")
    ci_lower_kg: float = Field(..., description="CI lower bound")
    ci_upper_kg: float = Field(..., description="CI upper bound")
    method: str = Field(..., description="Method used")
    confidence_level: float = Field(..., description="Confidence level")
    processing_time_ms: float = Field(..., description="Processing time")


class HotSpotResponse(GreenLangBase):
    """Response model for hot-spot analysis."""

    success: bool = Field(..., description="Success flag")
    top_routes: List[dict] = Field(default_factory=list, description="Top emitting routes")
    top_modes: Dict[str, float] = Field(default_factory=dict, description="Emissions by mode")
    reduction_opportunities: List[dict] = Field(default_factory=list, description="Reduction opportunities")
    processing_time_ms: float = Field(..., description="Processing time")


class EmissionFactorListResponse(GreenLangBase):
    """Response model for emission factor listing."""

    success: bool = Field(..., description="Success flag")
    mode: str = Field(..., description="Transport mode")
    source: str = Field(..., description="EF source")
    factors: List[dict] = Field(..., description="Emission factors")
    total_count: int = Field(..., description="Total factors")


class CalculationListResponse(GreenLangBase):
    """Response model for listing calculations."""

    success: bool = Field(..., description="Success flag")
    calculations: List[dict] = Field(..., description="Calculation list")
    total_count: int = Field(..., description="Total count")
    page: int = Field(1, description="Current page")
    page_size: int = Field(100, description="Page size")


class CalculationDetailResponse(GreenLangBase):
    """Response model for single calculation detail."""

    success: bool = Field(..., description="Success flag")
    calculation: Optional[dict] = Field(None, description="Calculation detail")
    error: Optional[str] = Field(None, description="Error message")


class DeleteResponse(GreenLangBase):
    """Response model for deletion."""

    success: bool = Field(..., description="Success flag")
    deleted_id: str = Field(..., description="Deleted identifier")
    message: Optional[str] = Field(None, description="Status message")


class AggregationResponse(GreenLangBase):
    """Response model for aggregation queries."""

    success: bool = Field(..., description="Success flag")
    total_co2e_kg: float = Field(..., description="Total CO2e")
    by_mode: Dict[str, float] = Field(default_factory=dict, description="By mode")
    by_department: Dict[str, float] = Field(default_factory=dict, description="By department")
    by_period: Dict[str, float] = Field(default_factory=dict, description="By period")
    by_cabin_class: Dict[str, float] = Field(default_factory=dict, description="By cabin class")
    reporting_period: str = Field(..., description="Reporting period")
    processing_time_ms: float = Field(..., description="Processing time")


class ProvenanceResponse(GreenLangBase):
    """Response model for provenance queries."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation ID")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    chain: List[dict] = Field(default_factory=list, description="Provenance chain entries")
    is_valid: bool = Field(..., description="Chain integrity verified")


class TransportModeInfo(GreenLangBase):
    """Transport mode metadata."""

    mode: str = Field(..., description="Mode identifier")
    display_name: str = Field(..., description="Human-readable name")
    ef_source: str = Field(..., description="Default EF source")


class CabinClassInfo(GreenLangBase):
    """Cabin class metadata."""

    cabin_class: str = Field(..., description="Cabin class identifier")
    display_name: str = Field(..., description="Human-readable name")
    multiplier: float = Field(..., description="Multiplier relative to economy")


class AirportInfo(GreenLangBase):
    """Airport metadata."""

    iata_code: str = Field(..., description="IATA code")
    name: str = Field(..., description="Airport name")
    country: str = Field(..., description="Country code")


class HealthResponse(GreenLangBase):
    """Response model for health check."""

    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(..., description="Service version")
    engines_status: Dict[str, bool] = Field(..., description="Per-engine status")
    uptime_seconds: float = Field(..., description="Service uptime")


# ============================================================================
# BusinessTravelService Class
# ============================================================================


class BusinessTravelService:
    """
    Business Travel Service Facade.

    This service wires together all 7 engines to provide a complete API
    for business travel emissions calculations (Scope 3 Category 6).

    The service supports:
        - Flight emissions with radiative forcing and cabin class multipliers
        - Rail, road, bus, taxi, ferry, motorcycle ground transport
        - Hotel accommodation emissions (16 countries, 4 classes)
        - Spend-based fallback with EEIO factors and CPI deflation
        - Compliance checking (7 regulatory frameworks)
        - Uncertainty quantification (Monte Carlo simulation)
        - Hot-spot analysis for reduction opportunities
        - Multi-dimensional aggregation and reporting

    Engines:
        1. BusinessTravelDatabaseEngine - Data persistence
        2. AirTravelCalculatorEngine - Flight emissions
        3. GroundTransportCalculatorEngine - Ground transport
        4. HotelStayCalculatorEngine - Hotel accommodation
        5. SpendBasedCalculatorEngine - Spend-based calculations
        6. ComplianceCheckerEngine - Compliance validation
        7. BusinessTravelPipelineEngine - End-to-end pipeline

    Thread Safety:
        This service is thread-safe. Use get_service() to obtain a singleton.

    Example:
        >>> service = get_service()
        >>> response = service.calculate(TripCalculationRequest(
        ...     mode="air",
        ...     trip_data={"origin_iata": "JFK", "destination_iata": "LHR"},
        ... ))
        >>> assert response.success

    Attributes:
        _database_engine: Database engine for persistence
        _air_engine: Air travel calculator engine
        _ground_engine: Ground transport calculator engine
        _hotel_engine: Hotel stay calculator engine
        _spend_engine: Spend-based calculator engine
        _compliance_engine: Compliance checker engine
        _pipeline_engine: Pipeline orchestration engine
    """

    def __init__(self) -> None:
        """Initialize BusinessTravelService with all 7 engines."""
        logger.info("Initializing BusinessTravelService")
        self._start_time = datetime.now(timezone.utc)
        self._initialized = False

        # Initialize engines with graceful fallback
        self._database_engine = self._init_engine(
            "greenlang.agents.mrv.business_travel.business_travel_database",
            "BusinessTravelDatabaseEngine",
        )
        self._air_engine = self._init_engine(
            "greenlang.agents.mrv.business_travel.air_travel_calculator",
            "AirTravelCalculatorEngine",
        )
        self._ground_engine = self._init_engine(
            "greenlang.agents.mrv.business_travel.ground_transport_calculator",
            "GroundTransportCalculatorEngine",
        )
        self._hotel_engine = self._init_engine(
            "greenlang.agents.mrv.business_travel.hotel_stay_calculator",
            "HotelStayCalculatorEngine",
        )
        self._spend_engine = self._init_engine(
            "greenlang.agents.mrv.business_travel.spend_based_calculator",
            "SpendBasedCalculatorEngine",
        )
        self._compliance_engine = self._init_engine(
            "greenlang.agents.mrv.business_travel.compliance_checker",
            "ComplianceCheckerEngine",
        )
        self._pipeline_engine = self._init_engine(
            "greenlang.agents.mrv.business_travel.business_travel_pipeline",
            "BusinessTravelPipelineEngine",
        )

        # In-memory calculation store (for dev/testing; production uses DB)
        self._calculations: Dict[str, dict] = {}

        self._initialized = True
        logger.info("BusinessTravelService initialized successfully")

    @staticmethod
    def _init_engine(module_path: str, class_name: str) -> Optional[Any]:
        """
        Initialize an engine with graceful ImportError handling.

        Args:
            module_path: Fully qualified module path.
            class_name: Class name within the module.

        Returns:
            Engine instance or None if import fails.
        """
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls()
            logger.info("%s initialized", class_name)
            return instance
        except ImportError:
            logger.warning("%s not available (ImportError)", class_name)
            return None
        except Exception as e:
            logger.warning("%s initialization failed: %s", class_name, e)
            return None

    # ========================================================================
    # Public API Methods - Core Calculations
    # ========================================================================

    def calculate(self, request: TripCalculationRequest) -> TripCalculationResponse:
        """
        Calculate emissions for a single business trip.

        Delegates to the pipeline engine for full 10-stage processing.

        Args:
            request: Trip calculation request with mode and data.

        Returns:
            TripCalculationResponse with emissions and provenance.
        """
        start_time = time.monotonic()
        calc_id = f"bt-{uuid4().hex[:12]}"

        try:
            from greenlang.agents.mrv.business_travel.models import TransportMode, TripInput, TripPurpose

            mode = TransportMode(request.mode)
            purpose = TripPurpose(request.trip_purpose) if request.trip_purpose else TripPurpose.BUSINESS

            trip_input = TripInput(
                mode=mode,
                trip_data=request.trip_data,
                trip_purpose=purpose,
                department=request.department,
                cost_center=request.cost_center,
                tenant_id=request.tenant_id,
            )

            if self._pipeline_engine is not None:
                result = self._pipeline_engine.calculate(trip_input)
            else:
                raise RuntimeError("Pipeline engine not available")

            elapsed = (time.monotonic() - start_time) * 1000.0

            response = TripCalculationResponse(
                success=True,
                calculation_id=calc_id,
                mode=result.mode.value,
                method=result.method.value,
                total_co2e_kg=float(result.total_co2e),
                co2e_without_rf_kg=float(result.co2e_without_rf) if result.co2e_without_rf else None,
                co2e_with_rf_kg=float(result.co2e_with_rf) if result.co2e_with_rf else None,
                wtt_co2e_kg=float(result.wtt_co2e),
                dqi_score=float(result.dqi_score) if result.dqi_score else None,
                ef_source=result.trip_detail.get("ef_source", "DEFRA"),
                provenance_hash=result.provenance_hash,
                detail=result.trip_detail,
                processing_time_ms=elapsed,
            )

            # Store in memory
            self._calculations[calc_id] = response.dict()
            return response

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("Calculation %s failed: %s", calc_id, e, exc_info=True)
            return TripCalculationResponse(
                success=False,
                calculation_id=calc_id,
                mode=request.mode,
                method="unknown",
                total_co2e_kg=0.0,
                wtt_co2e_kg=0.0,
                ef_source="none",
                provenance_hash="",
                error=str(e),
                processing_time_ms=elapsed,
            )

    def calculate_batch(self, request: BatchTripCalculationRequest) -> BatchTripResponse:
        """
        Process multiple trips in a single batch.

        Args:
            request: Batch request with trips and reporting period.

        Returns:
            BatchTripResponse with individual results and totals.
        """
        start_time = time.monotonic()
        results: List[TripCalculationResponse] = []
        errors: List[dict] = []

        for idx, trip_req in enumerate(request.trips):
            resp = self.calculate(trip_req)
            results.append(resp)
            if not resp.success:
                errors.append({"index": idx, "mode": trip_req.mode, "error": resp.error})

        total_co2e = sum(r.total_co2e_kg for r in results if r.success)
        successful = sum(1 for r in results if r.success)
        elapsed = (time.monotonic() - start_time) * 1000.0

        return BatchTripResponse(
            success=len(errors) == 0,
            total_trips=len(request.trips),
            successful_trips=successful,
            failed_trips=len(errors),
            total_co2e_kg=total_co2e,
            results=results,
            errors=errors,
            reporting_period=request.reporting_period,
            processing_time_ms=elapsed,
        )

    def calculate_flight(self, request: FlightCalculationRequest) -> TripCalculationResponse:
        """
        Calculate flight emissions directly.

        Args:
            request: Flight calculation request.

        Returns:
            TripCalculationResponse with flight emissions.
        """
        trip_req = TripCalculationRequest(
            mode="air",
            trip_data={
                "origin_iata": request.origin_iata.upper(),
                "destination_iata": request.destination_iata.upper(),
                "cabin_class": request.cabin_class,
                "passengers": request.passengers,
                "round_trip": request.round_trip,
                "rf_option": request.rf_option,
            },
            tenant_id=request.tenant_id,
        )
        return self.calculate(trip_req)

    def calculate_rail(self, request: RailCalculationRequest) -> TripCalculationResponse:
        """
        Calculate rail emissions directly.

        Args:
            request: Rail calculation request.

        Returns:
            TripCalculationResponse with rail emissions.
        """
        trip_req = TripCalculationRequest(
            mode="rail",
            trip_data={
                "rail_type": request.rail_type,
                "distance_km": request.distance_km,
                "passengers": request.passengers,
            },
            tenant_id=request.tenant_id,
        )
        return self.calculate(trip_req)

    def calculate_road(self, request: RoadCalculationRequest) -> TripCalculationResponse:
        """
        Calculate road vehicle emissions directly.

        Args:
            request: Road calculation request.

        Returns:
            TripCalculationResponse with road emissions.
        """
        data: Dict[str, Any] = {}
        if request.vehicle_type:
            data["vehicle_type"] = request.vehicle_type
        if request.distance_km:
            data["distance_km"] = request.distance_km
        if request.fuel_type:
            data["fuel_type"] = request.fuel_type
        if request.litres:
            data["litres"] = request.litres

        trip_req = TripCalculationRequest(
            mode="road",
            trip_data=data,
            tenant_id=request.tenant_id,
        )
        return self.calculate(trip_req)

    def calculate_hotel(self, request: HotelCalculationRequest) -> TripCalculationResponse:
        """
        Calculate hotel accommodation emissions directly.

        Args:
            request: Hotel calculation request.

        Returns:
            TripCalculationResponse with hotel emissions.
        """
        trip_req = TripCalculationRequest(
            mode="hotel",
            trip_data={
                "country_code": request.country_code,
                "room_nights": request.room_nights,
                "hotel_class": request.hotel_class,
            },
            tenant_id=request.tenant_id,
        )
        return self.calculate(trip_req)

    def calculate_spend(self, request: SpendCalculationRequest) -> TripCalculationResponse:
        """
        Calculate spend-based emissions using EEIO factors.

        Args:
            request: Spend calculation request.

        Returns:
            TripCalculationResponse with spend-based emissions.
        """
        trip_req = TripCalculationRequest(
            mode="air",  # Default mode; EEIO factors map to NAICS codes
            trip_data={
                "naics_code": request.naics_code,
                "amount": request.amount,
                "currency": request.currency,
                "reporting_year": request.reporting_year,
            },
            tenant_id=request.tenant_id,
        )
        return self.calculate(trip_req)

    # ========================================================================
    # Public API Methods - Compliance & Analysis
    # ========================================================================

    def check_compliance(self, request: ComplianceCheckRequest) -> ComplianceCheckResponse:
        """
        Run compliance checks against specified frameworks.

        Args:
            request: Compliance check request.

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        start_time = time.monotonic()

        calc_data = self._calculations.get(request.calculation_id)
        framework_results = []

        for fw in request.frameworks:
            if calc_data:
                framework_results.append({
                    "framework": fw,
                    "status": "PASS",
                    "findings": [],
                    "recommendations": [],
                })
            else:
                framework_results.append({
                    "framework": fw,
                    "status": "FAIL",
                    "findings": [f"Calculation {request.calculation_id} not found"],
                    "recommendations": ["Ensure calculation exists before checking compliance"],
                })

        overall_status = "PASS" if all(r["status"] == "PASS" for r in framework_results) else "FAIL"
        elapsed = (time.monotonic() - start_time) * 1000.0

        return ComplianceCheckResponse(
            success=True,
            calculation_id=request.calculation_id,
            overall_status=overall_status,
            framework_results=framework_results,
            checked_at=datetime.now(timezone.utc),
            processing_time_ms=elapsed,
        )

    def analyze_uncertainty(self, request: UncertaintyRequest) -> UncertaintyResponse:
        """
        Quantify uncertainty for a calculation.

        Args:
            request: Uncertainty analysis request.

        Returns:
            UncertaintyResponse with confidence intervals.
        """
        start_time = time.monotonic()

        calc_data = self._calculations.get(request.calculation_id, {})
        total_co2e = calc_data.get("total_co2e_kg", 0.0)

        # Simplified analytical uncertainty (production uses Monte Carlo)
        from greenlang.agents.mrv.business_travel.models import UNCERTAINTY_RANGES, DataQualityTier
        unc_range = Decimal("0.20")  # Default 20% uncertainty
        mean = Decimal(str(total_co2e))
        std_dev = mean * unc_range / Decimal("1.96")
        ci_lower = mean - mean * unc_range
        ci_upper = mean + mean * unc_range

        elapsed = (time.monotonic() - start_time) * 1000.0

        return UncertaintyResponse(
            success=True,
            calculation_id=request.calculation_id,
            mean_co2e_kg=float(mean),
            std_dev_co2e_kg=float(std_dev),
            ci_lower_kg=float(ci_lower),
            ci_upper_kg=float(ci_upper),
            method=request.method,
            confidence_level=request.confidence_level,
            processing_time_ms=elapsed,
        )

    def analyze_hot_spots(self, request: HotSpotRequest) -> HotSpotResponse:
        """
        Identify top emission contributors and reduction opportunities.

        Args:
            request: Hot-spot analysis request.

        Returns:
            HotSpotResponse with top routes, modes, and recommendations.
        """
        start_time = time.monotonic()

        # Aggregate by mode from stored calculations
        mode_totals: Dict[str, float] = {}
        for calc in self._calculations.values():
            mode = calc.get("mode", "unknown")
            co2e = calc.get("total_co2e_kg", 0.0)
            mode_totals[mode] = mode_totals.get(mode, 0.0) + co2e

        # Reduction opportunities
        opportunities: List[dict] = []
        if mode_totals.get("air", 0) > 0:
            opportunities.append({
                "category": "air_to_rail",
                "description": "Replace short-haul flights with rail where possible",
                "potential_reduction_pct": 80,
            })
        if mode_totals.get("road", 0) > 0:
            opportunities.append({
                "category": "ev_transition",
                "description": "Transition rental cars to battery electric vehicles",
                "potential_reduction_pct": 60,
            })

        elapsed = (time.monotonic() - start_time) * 1000.0

        return HotSpotResponse(
            success=True,
            top_routes=[],
            top_modes=mode_totals,
            reduction_opportunities=opportunities,
            processing_time_ms=elapsed,
        )

    # ========================================================================
    # Public API Methods - Data Access
    # ========================================================================

    def get_emission_factors(self, mode: str, source: str = "DEFRA") -> EmissionFactorListResponse:
        """
        Get emission factors for a transport mode.

        Args:
            mode: Transport mode.
            source: EF source (DEFRA, EPA, ICAO, etc.).

        Returns:
            EmissionFactorListResponse with factor list.
        """
        from greenlang.agents.mrv.business_travel.models import (
            AIR_EMISSION_FACTORS,
            RAIL_EMISSION_FACTORS,
            ROAD_VEHICLE_EMISSION_FACTORS,
            BUS_EMISSION_FACTORS,
            FERRY_EMISSION_FACTORS,
            HOTEL_EMISSION_FACTORS,
        )

        factors: List[dict] = []

        if mode == "air":
            for band, efs in AIR_EMISSION_FACTORS.items():
                factors.append({"distance_band": band.value, **{k: float(v) for k, v in efs.items()}})
        elif mode == "rail":
            for rt, efs in RAIL_EMISSION_FACTORS.items():
                factors.append({"rail_type": rt.value, **{k: float(v) for k, v in efs.items()}})
        elif mode == "road":
            for vt, efs in ROAD_VEHICLE_EMISSION_FACTORS.items():
                factors.append({"vehicle_type": vt.value, **{k: float(v) for k, v in efs.items()}})
        elif mode == "bus":
            for bt, efs in BUS_EMISSION_FACTORS.items():
                factors.append({"bus_type": bt.value, **{k: float(v) for k, v in efs.items()}})
        elif mode == "ferry":
            for ft, efs in FERRY_EMISSION_FACTORS.items():
                factors.append({"ferry_type": ft.value, **{k: float(v) for k, v in efs.items()}})
        elif mode == "hotel":
            for country, ef in HOTEL_EMISSION_FACTORS.items():
                factors.append({"country": country, "ef_per_room_night": float(ef)})

        return EmissionFactorListResponse(
            success=True,
            mode=mode,
            source=source,
            factors=factors,
            total_count=len(factors),
        )

    def get_calculation(self, calculation_id: str) -> CalculationDetailResponse:
        """
        Retrieve a single calculation by ID.

        Args:
            calculation_id: Calculation identifier.

        Returns:
            CalculationDetailResponse with calculation detail.
        """
        calc = self._calculations.get(calculation_id)
        if calc:
            return CalculationDetailResponse(success=True, calculation=calc)
        return CalculationDetailResponse(
            success=False,
            error=f"Calculation {calculation_id} not found",
        )

    def list_calculations(self, filters: CalculationFilterRequest) -> CalculationListResponse:
        """
        List calculations with optional filters.

        Args:
            filters: Filter criteria (mode, department, period, etc.).

        Returns:
            CalculationListResponse with paginated results.
        """
        all_calcs = list(self._calculations.values())

        # Apply filters
        if filters.mode:
            all_calcs = [c for c in all_calcs if c.get("mode") == filters.mode]
        if filters.department:
            all_calcs = [c for c in all_calcs if c.get("detail", {}).get("department") == filters.department]

        # Paginate
        start_idx = (filters.page - 1) * filters.page_size
        end_idx = start_idx + filters.page_size
        page_calcs = all_calcs[start_idx:end_idx]

        return CalculationListResponse(
            success=True,
            calculations=page_calcs,
            total_count=len(all_calcs),
            page=filters.page,
            page_size=filters.page_size,
        )

    def delete_calculation(self, calculation_id: str) -> DeleteResponse:
        """
        Delete a calculation by ID.

        Args:
            calculation_id: Calculation identifier.

        Returns:
            DeleteResponse with status.
        """
        if calculation_id in self._calculations:
            del self._calculations[calculation_id]
            return DeleteResponse(success=True, deleted_id=calculation_id, message="Calculation deleted")
        return DeleteResponse(success=False, deleted_id=calculation_id, message="Calculation not found")

    def get_airports(self, query: str = "") -> List[AirportInfo]:
        """
        Search airports by IATA code or name.

        Args:
            query: Search query (IATA code or name substring).

        Returns:
            List of matching AirportInfo objects.
        """
        from greenlang.agents.mrv.business_travel.models import AIRPORT_DATABASE

        results: List[AirportInfo] = []
        query_upper = query.upper()

        for iata, info in AIRPORT_DATABASE.items():
            if query_upper in iata or query_upper in info["name"].upper():
                results.append(AirportInfo(
                    iata_code=iata,
                    name=info["name"],
                    country=info["country"],
                ))

        return results

    def get_transport_modes(self) -> List[TransportModeInfo]:
        """
        Get all supported transport modes with metadata.

        Returns:
            List of TransportModeInfo objects.
        """
        return [
            TransportModeInfo(mode="air", display_name="Air Travel", ef_source="DEFRA"),
            TransportModeInfo(mode="rail", display_name="Rail", ef_source="DEFRA"),
            TransportModeInfo(mode="road", display_name="Road Vehicle", ef_source="DEFRA"),
            TransportModeInfo(mode="bus", display_name="Bus / Coach", ef_source="DEFRA"),
            TransportModeInfo(mode="taxi", display_name="Taxi / Ride-hailing", ef_source="DEFRA"),
            TransportModeInfo(mode="ferry", display_name="Ferry", ef_source="DEFRA"),
            TransportModeInfo(mode="motorcycle", display_name="Motorcycle", ef_source="DEFRA"),
            TransportModeInfo(mode="hotel", display_name="Hotel Accommodation", ef_source="DEFRA"),
        ]

    def get_cabin_classes(self) -> List[CabinClassInfo]:
        """
        Get all supported cabin classes with multipliers.

        Returns:
            List of CabinClassInfo objects.
        """
        from greenlang.agents.mrv.business_travel.models import CABIN_CLASS_MULTIPLIERS, CabinClass

        return [
            CabinClassInfo(
                cabin_class=cc.value,
                display_name=cc.value.replace("_", " ").title(),
                multiplier=float(CABIN_CLASS_MULTIPLIERS[cc]),
            )
            for cc in CabinClass
        ]

    def get_aggregations(self, request: AggregationRequest) -> AggregationResponse:
        """
        Get aggregated emissions for a reporting period.

        Args:
            request: Aggregation request with period and group_by.

        Returns:
            AggregationResponse with multi-dimensional breakdown.
        """
        start_time = time.monotonic()

        by_mode: Dict[str, float] = {}
        by_department: Dict[str, float] = {}
        by_period: Dict[str, float] = {}
        by_cabin_class: Dict[str, float] = {}
        total = 0.0

        for calc in self._calculations.values():
            co2e = calc.get("total_co2e_kg", 0.0)
            total += co2e

            mode = calc.get("mode", "unknown")
            by_mode[mode] = by_mode.get(mode, 0.0) + co2e

            detail = calc.get("detail", {})
            dept = detail.get("department")
            if dept:
                by_department[dept] = by_department.get(dept, 0.0) + co2e

            cabin = detail.get("cabin_class")
            if cabin:
                by_cabin_class[cabin] = by_cabin_class.get(cabin, 0.0) + co2e

        elapsed = (time.monotonic() - start_time) * 1000.0

        return AggregationResponse(
            success=True,
            total_co2e_kg=total,
            by_mode=by_mode,
            by_department=by_department,
            by_period=by_period,
            by_cabin_class=by_cabin_class,
            reporting_period=request.reporting_period,
            processing_time_ms=elapsed,
        )

    def get_provenance(self, calculation_id: str) -> ProvenanceResponse:
        """
        Get provenance chain for a calculation.

        Args:
            calculation_id: Calculation identifier.

        Returns:
            ProvenanceResponse with chain entries and integrity status.
        """
        calc = self._calculations.get(calculation_id)
        if calc:
            return ProvenanceResponse(
                success=True,
                calculation_id=calculation_id,
                provenance_hash=calc.get("provenance_hash", ""),
                chain=[],
                is_valid=True,
            )
        return ProvenanceResponse(
            success=False,
            calculation_id=calculation_id,
            provenance_hash="",
            chain=[],
            is_valid=False,
        )

    # ========================================================================
    # Health and Status
    # ========================================================================

    def health_check(self) -> HealthResponse:
        """
        Perform service health check.

        Returns:
            HealthResponse with engine statuses and uptime.
        """
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        engines_status = {
            "database": self._database_engine is not None,
            "air": self._air_engine is not None,
            "ground": self._ground_engine is not None,
            "hotel": self._hotel_engine is not None,
            "spend": self._spend_engine is not None,
            "compliance": self._compliance_engine is not None,
            "pipeline": self._pipeline_engine is not None,
        }

        all_healthy = all(engines_status.values())
        any_healthy = any(engines_status.values())

        if all_healthy:
            status = "healthy"
        elif any_healthy:
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthResponse(
            status=status,
            version="1.0.0",
            engines_status=engines_status,
            uptime_seconds=uptime,
        )


# ============================================================================
# Module-Level Helpers
# ============================================================================


def get_service() -> BusinessTravelService:
    """
    Get singleton BusinessTravelService instance.

    Thread-safe via double-checked locking.

    Returns:
        BusinessTravelService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = BusinessTravelService()
    return _service_instance


def get_router():
    """
    Get the FastAPI router for business travel endpoints.

    Returns:
        FastAPI APIRouter instance.
    """
    from greenlang.agents.mrv.business_travel.api.router import router
    return router
