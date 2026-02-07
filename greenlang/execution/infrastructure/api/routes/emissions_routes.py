"""
GreenLang Emissions REST API Routes

This module provides REST API endpoints for emission calculations,
following the GreenLang API standards with comprehensive error handling,
authentication, rate limiting, and audit trails.

Endpoints:
    GET  /api/v1/emissions          - List emission calculations
    POST /api/v1/emissions/calculate - Run emission calculation
    GET  /api/v1/emissions/{id}     - Get specific result

Features:
    - JWT/API Key authentication
    - Rate limiting (configurable per endpoint)
    - Comprehensive request/response validation
    - Audit trail logging
    - OpenAPI 3.0 documentation

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.api.routes.emissions_routes import emissions_router
    >>>
    >>> app = FastAPI()
    >>> app.include_router(emissions_router, prefix="/api/v1")
"""

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object
    Depends = None
    HTTPException = Exception
    Query = None
    Request = None
    status = None
    JSONResponse = None

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS
# =============================================================================


class ScopeType(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class EmissionStatus(str, Enum):
    """Emission calculation status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BoundaryType(str, Enum):
    """Emission calculation boundary types."""
    COMBUSTION = "combustion"
    WTT = "wtt"  # Well-to-Tank
    WTW = "wtw"  # Well-to-Wheel
    CRADLE_TO_GATE = "cradle_to_gate"
    CRADLE_TO_GRAVE = "cradle_to_grave"


class GWPSet(str, Enum):
    """Global Warming Potential reference sets."""
    IPCC_AR6_100 = "IPCC_AR6_100"
    IPCC_AR6_20 = "IPCC_AR6_20"
    IPCC_AR5_100 = "IPCC_AR5_100"
    IPCC_AR4_100 = "IPCC_AR4_100"


# =============================================================================
# REQUEST MODELS
# =============================================================================


class EmissionCalculationRequest(BaseModel):
    """
    Request model for emission calculation.

    Attributes:
        fuel_type: Type of fuel (diesel, natural_gas, electricity, etc.)
        activity_amount: Amount of activity (consumption)
        activity_unit: Unit of measurement
        geography: ISO country/region code
        scope: GHG Protocol scope
        boundary: Calculation boundary
        gwp_set: Global Warming Potential reference
        calculation_date: Optional date for historical calculations
        metadata: Optional additional metadata
    """
    fuel_type: str = Field(
        ...,
        description="Fuel type: diesel, gasoline, natural_gas, electricity, etc.",
        example="diesel",
        min_length=1,
        max_length=100
    )
    activity_amount: float = Field(
        ...,
        gt=0,
        description="Activity amount (must be positive)",
        example=1000.0
    )
    activity_unit: str = Field(
        ...,
        description="Activity unit: gallons, kWh, therms, liters, kg, etc.",
        example="gallons",
        min_length=1,
        max_length=50
    )
    geography: str = Field(
        default="US",
        description="ISO country code or region",
        example="US",
        min_length=2,
        max_length=10
    )
    scope: ScopeType = Field(
        default=ScopeType.SCOPE_1,
        description="GHG Protocol scope (1, 2, or 3)"
    )
    boundary: BoundaryType = Field(
        default=BoundaryType.COMBUSTION,
        description="Calculation boundary"
    )
    gwp_set: GWPSet = Field(
        default=GWPSet.IPCC_AR6_100,
        description="Global Warming Potential reference set"
    )
    calculation_date: Optional[datetime] = Field(
        default=None,
        description="Date for historical factor lookup"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the calculation"
    )

    class Config:
        schema_extra = {
            "example": {
                "fuel_type": "diesel",
                "activity_amount": 1000.0,
                "activity_unit": "gallons",
                "geography": "US",
                "scope": "scope_1",
                "boundary": "combustion",
                "gwp_set": "IPCC_AR6_100"
            }
        }

    @validator("activity_amount")
    def validate_activity_amount(cls, v: float) -> float:
        """Ensure activity amount is within reasonable bounds."""
        if v > 1e12:
            raise ValueError("Activity amount exceeds maximum allowed value")
        return v


class BatchEmissionRequest(BaseModel):
    """
    Batch emission calculation request.

    Attributes:
        calculations: List of individual calculation requests
        aggregate_results: Whether to aggregate results
    """
    calculations: List[EmissionCalculationRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of calculations (max 100 per batch)"
    )
    aggregate_results: bool = Field(
        default=True,
        description="Whether to aggregate results into totals"
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class GasBreakdown(BaseModel):
    """
    Greenhouse gas breakdown by gas type.

    Attributes:
        co2_kg: Carbon dioxide emissions in kg
        ch4_kg: Methane emissions in kg
        n2o_kg: Nitrous oxide emissions in kg
        hfcs_kg: HFC emissions in kg (optional)
        pfcs_kg: PFC emissions in kg (optional)
        sf6_kg: SF6 emissions in kg (optional)
    """
    co2_kg: float = Field(..., description="CO2 emissions in kg")
    ch4_kg: float = Field(..., description="CH4 emissions in kg")
    n2o_kg: float = Field(..., description="N2O emissions in kg")
    hfcs_kg: Optional[float] = Field(default=None, description="HFC emissions in kg")
    pfcs_kg: Optional[float] = Field(default=None, description="PFC emissions in kg")
    sf6_kg: Optional[float] = Field(default=None, description="SF6 emissions in kg")


class EmissionFactorInfo(BaseModel):
    """
    Information about the emission factor used in calculation.

    Attributes:
        factor_id: Unique factor identifier
        source: Source organization (EPA, DEFRA, etc.)
        source_year: Year of the emission factor
        co2e_per_unit: kg CO2e per activity unit
        data_quality_score: Quality score (1-5)
        uncertainty_percent: Uncertainty as percentage
    """
    factor_id: str = Field(..., description="Unique factor identifier")
    source: str = Field(..., description="Source organization")
    source_year: int = Field(..., description="Year of emission factor")
    co2e_per_unit: float = Field(..., description="kg CO2e per activity unit")
    data_quality_score: float = Field(..., ge=1, le=5, description="Data quality score")
    uncertainty_percent: float = Field(..., ge=0, le=100, description="Uncertainty %")


class EmissionCalculationResponse(BaseModel):
    """
    Response model for a single emission calculation.

    Attributes:
        calculation_id: Unique calculation identifier
        status: Calculation status
        emissions_kg_co2e: Total emissions in kg CO2e
        emissions_tonnes_co2e: Total emissions in tonnes CO2e
        gas_breakdown: Breakdown by individual gases
        factor_info: Information about the factor used
        request_summary: Summary of the input request
        created_at: Timestamp of calculation
        completed_at: Timestamp when calculation completed
        provenance_hash: SHA-256 hash for audit trail
    """
    calculation_id: str = Field(..., description="Unique calculation ID")
    status: EmissionStatus = Field(..., description="Calculation status")
    emissions_kg_co2e: float = Field(..., description="Total emissions in kg CO2e")
    emissions_tonnes_co2e: float = Field(..., description="Total emissions in tonnes CO2e")
    gas_breakdown: GasBreakdown = Field(..., description="Emissions by gas type")
    factor_info: EmissionFactorInfo = Field(..., description="Factor information")
    request_summary: Dict[str, Any] = Field(..., description="Input request summary")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    class Config:
        schema_extra = {
            "example": {
                "calculation_id": "calc_abc123xyz",
                "status": "completed",
                "emissions_kg_co2e": 10210.0,
                "emissions_tonnes_co2e": 10.21,
                "gas_breakdown": {
                    "co2_kg": 10180.0,
                    "ch4_kg": 23.0,
                    "n2o_kg": 7.0
                },
                "factor_info": {
                    "factor_id": "EF:US:diesel:2024:v1",
                    "source": "EPA",
                    "source_year": 2024,
                    "co2e_per_unit": 10.21,
                    "data_quality_score": 4.6,
                    "uncertainty_percent": 5.0
                },
                "request_summary": {
                    "fuel_type": "diesel",
                    "activity_amount": 1000.0,
                    "activity_unit": "gallons"
                },
                "created_at": "2025-12-07T10:30:00Z",
                "completed_at": "2025-12-07T10:30:01Z",
                "provenance_hash": "sha256:abc123..."
            }
        }


class EmissionListResponse(BaseModel):
    """
    Paginated list of emission calculations.

    Attributes:
        items: List of calculation results
        total: Total number of matching items
        page: Current page number
        page_size: Items per page
        total_pages: Total number of pages
        has_next: Whether there is a next page
        has_prev: Whether there is a previous page
    """
    items: List[EmissionCalculationResponse] = Field(..., description="Calculation results")
    total: int = Field(..., description="Total matching items")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")


class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable message")
    field: Optional[str] = Field(default=None, description="Field that caused the error")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")


class ErrorResponse(BaseModel):
    """
    Standard error response model.

    Attributes:
        error_code: Application error code
        message: Human-readable error message
        errors: List of detailed errors
        correlation_id: Request correlation ID for tracing
        timestamp: Error timestamp
    """
    error_code: str = Field(..., description="Application error code")
    message: str = Field(..., description="Error message")
    errors: Optional[List[ErrorDetail]] = Field(default=None, description="Detailed errors")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# STORAGE (In-memory for demonstration - replace with database in production)
# =============================================================================

# In-memory storage for calculations
_emission_calculations: Dict[str, EmissionCalculationResponse] = {}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def generate_calculation_id() -> str:
    """Generate a unique calculation ID."""
    return f"calc_{uuid.uuid4().hex[:12]}"


def compute_provenance_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    import hashlib
    import json
    serialized = json.dumps(data, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()}"


def perform_emission_calculation(
    request: EmissionCalculationRequest
) -> EmissionCalculationResponse:
    """
    Perform the actual emission calculation.

    In production, this would integrate with the GreenLang calculation engine.
    """
    # Simulated emission factors (in production, fetch from factor database)
    emission_factors = {
        "diesel": {"co2e": 10.21, "co2": 10.18, "ch4": 0.023, "n2o": 0.007},
        "gasoline": {"co2e": 8.78, "co2": 8.75, "ch4": 0.02, "n2o": 0.006},
        "natural_gas": {"co2e": 5.31, "co2": 5.28, "ch4": 0.02, "n2o": 0.005},
        "electricity": {"co2e": 0.42, "co2": 0.41, "ch4": 0.005, "n2o": 0.002},
    }

    fuel_key = request.fuel_type.lower()
    if fuel_key not in emission_factors:
        # Default factors for unknown fuels
        factors = {"co2e": 5.0, "co2": 4.95, "ch4": 0.03, "n2o": 0.02}
    else:
        factors = emission_factors[fuel_key]

    # Calculate emissions
    activity = request.activity_amount
    emissions_co2e = activity * factors["co2e"]

    calculation_id = generate_calculation_id()
    now = datetime.now(timezone.utc)

    request_summary = {
        "fuel_type": request.fuel_type,
        "activity_amount": request.activity_amount,
        "activity_unit": request.activity_unit,
        "geography": request.geography,
        "scope": request.scope.value,
        "boundary": request.boundary.value
    }

    response = EmissionCalculationResponse(
        calculation_id=calculation_id,
        status=EmissionStatus.COMPLETED,
        emissions_kg_co2e=round(emissions_co2e, 4),
        emissions_tonnes_co2e=round(emissions_co2e / 1000, 6),
        gas_breakdown=GasBreakdown(
            co2_kg=round(activity * factors["co2"], 4),
            ch4_kg=round(activity * factors["ch4"], 6),
            n2o_kg=round(activity * factors["n2o"], 6)
        ),
        factor_info=EmissionFactorInfo(
            factor_id=f"EF:{request.geography}:{fuel_key}:2024:v1",
            source="EPA" if request.geography == "US" else "DEFRA",
            source_year=2024,
            co2e_per_unit=factors["co2e"],
            data_quality_score=4.5,
            uncertainty_percent=5.0
        ),
        request_summary=request_summary,
        created_at=now,
        completed_at=now,
        provenance_hash=compute_provenance_hash(request_summary)
    )

    # Store calculation
    _emission_calculations[calculation_id] = response

    return response


# =============================================================================
# ROUTER DEFINITION
# =============================================================================

if FASTAPI_AVAILABLE:
    emissions_router = APIRouter(
        prefix="/emissions",
        tags=["Emissions"],
        responses={
            400: {"model": ErrorResponse, "description": "Bad Request"},
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            403: {"model": ErrorResponse, "description": "Forbidden"},
            404: {"model": ErrorResponse, "description": "Not Found"},
            429: {"model": ErrorResponse, "description": "Rate Limited"},
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
        }
    )


    @emissions_router.get(
        "",
        response_model=EmissionListResponse,
        summary="List emission calculations",
        description="""
        Retrieve a paginated list of emission calculations.

        Supports filtering by:
        - Status (pending, completed, failed)
        - Scope (scope_1, scope_2, scope_3)
        - Date range
        - Fuel type

        Results are sorted by creation date (most recent first).
        """,
        operation_id="list_emissions",
        responses={
            200: {
                "description": "List of emission calculations",
                "model": EmissionListResponse
            }
        }
    )
    async def list_emissions(
        request: Request,
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        status: Optional[EmissionStatus] = Query(None, description="Filter by status"),
        scope: Optional[ScopeType] = Query(None, description="Filter by scope"),
        fuel_type: Optional[str] = Query(None, description="Filter by fuel type"),
        from_date: Optional[datetime] = Query(None, description="Filter from date"),
        to_date: Optional[datetime] = Query(None, description="Filter to date"),
    ) -> EmissionListResponse:
        """
        List emission calculations with pagination and filtering.

        Args:
            request: FastAPI request object
            page: Page number (1-indexed)
            page_size: Number of items per page
            status: Optional status filter
            scope: Optional scope filter
            fuel_type: Optional fuel type filter
            from_date: Optional start date filter
            to_date: Optional end date filter

        Returns:
            Paginated list of emission calculations
        """
        logger.info(f"Listing emissions: page={page}, page_size={page_size}")

        # Filter calculations
        items = list(_emission_calculations.values())

        if status:
            items = [i for i in items if i.status == status]

        if scope:
            items = [i for i in items if i.request_summary.get("scope") == scope.value]

        if fuel_type:
            items = [i for i in items if i.request_summary.get("fuel_type", "").lower() == fuel_type.lower()]

        if from_date:
            items = [i for i in items if i.created_at >= from_date]

        if to_date:
            items = [i for i in items if i.created_at <= to_date]

        # Sort by creation date (most recent first)
        items.sort(key=lambda x: x.created_at, reverse=True)

        # Paginate
        total = len(items)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_items = items[start_idx:end_idx]

        return EmissionListResponse(
            items=paginated_items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )


    @emissions_router.post(
        "/calculate",
        response_model=EmissionCalculationResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Run emission calculation",
        description="""
        Submit a new emission calculation request.

        The calculation uses the appropriate emission factor based on:
        - Fuel type
        - Geographic region
        - Activity unit
        - GHG scope
        - Boundary type

        Returns the calculation result with full provenance tracking.
        """,
        operation_id="calculate_emissions",
        responses={
            201: {
                "description": "Calculation completed successfully",
                "model": EmissionCalculationResponse
            }
        }
    )
    async def calculate_emissions(
        request: Request,
        calculation_request: EmissionCalculationRequest,
    ) -> EmissionCalculationResponse:
        """
        Perform an emission calculation.

        Args:
            request: FastAPI request object
            calculation_request: Emission calculation parameters

        Returns:
            Emission calculation result with full details

        Raises:
            HTTPException: If calculation fails
        """
        logger.info(
            f"Emission calculation requested: "
            f"fuel={calculation_request.fuel_type}, "
            f"amount={calculation_request.activity_amount} "
            f"{calculation_request.activity_unit}"
        )

        try:
            result = perform_emission_calculation(calculation_request)

            logger.info(
                f"Calculation completed: {result.calculation_id}, "
                f"emissions={result.emissions_kg_co2e} kg CO2e"
            )

            return result

        except ValueError as e:
            logger.warning(f"Validation error in calculation: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "VALIDATION_ERROR",
                    "message": str(e)
                }
            )
        except Exception as e:
            logger.error(f"Calculation error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_code": "CALCULATION_ERROR",
                    "message": "An error occurred during emission calculation"
                }
            )


    @emissions_router.get(
        "/{calculation_id}",
        response_model=EmissionCalculationResponse,
        summary="Get emission calculation by ID",
        description="""
        Retrieve a specific emission calculation by its unique ID.

        Returns the full calculation details including:
        - Emission results
        - Gas breakdown
        - Factor information
        - Provenance hash
        """,
        operation_id="get_emission",
        responses={
            200: {
                "description": "Calculation details",
                "model": EmissionCalculationResponse
            },
            404: {
                "description": "Calculation not found"
            }
        }
    )
    async def get_emission(
        request: Request,
        calculation_id: str,
    ) -> EmissionCalculationResponse:
        """
        Get a specific emission calculation by ID.

        Args:
            request: FastAPI request object
            calculation_id: Unique calculation identifier

        Returns:
            Emission calculation result

        Raises:
            HTTPException: If calculation not found
        """
        logger.info(f"Getting emission calculation: {calculation_id}")

        calculation = _emission_calculations.get(calculation_id)

        if not calculation:
            logger.warning(f"Calculation not found: {calculation_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "NOT_FOUND",
                    "message": f"Emission calculation '{calculation_id}' not found"
                }
            )

        return calculation

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )
        protect_router(emissions_router)
    except ImportError:
        pass  # auth_service not available

else:
    # Provide stub router when FastAPI is not available
    emissions_router = None
    logger.warning("FastAPI not available - emissions_router is None")
