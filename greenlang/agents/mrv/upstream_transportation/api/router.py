"""
Upstream Transportation & Distribution API Router - AGENT-MRV-017

This module implements the FastAPI router for upstream transportation and distribution
emissions calculations following GHG Protocol Scope 3 Category 4 requirements.

Provides 20 REST endpoints for:
- Emissions calculations (single and batch)
- Transport chain management
- Emission factor lookup and custom factors
- Auto-classification
- Compliance checking
- Uncertainty analysis
- Aggregations and hot-spot analysis
- Export functionality
- Health and statistics

Follows GreenLang's zero-hallucination principle with deterministic calculations.

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.agents.mrv.upstream_transportation.api.router import router
    >>> app = FastAPI()
    >>> app.include_router(router)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
from typing import Optional, List, Dict, Any
from decimal import Decimal
from datetime import datetime, date
from uuid import UUID
import logging

from pydantic import BaseModel, Field, validator, constr

from greenlang.agents.mrv.upstream_transportation.service import UpstreamTransportationService
from greenlang.agents.mrv.upstream_transportation.models import (
    TransportMode,
    VehicleType,
    CalculationMethod,
    UncertaintyMethod,
    TransportChain,
    EmissionFactor,
    CalculationResult,
)

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/api/v1/upstream-transportation",
    tags=["upstream-transportation"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# REQUEST MODELS
# ============================================================================


class CalculateRequest(BaseModel):
    """
    Request model for single upstream transportation emissions calculation.

    Attributes:
        tenant_id: Tenant identifier for multi-tenancy
        calculation_id: Optional UUID for idempotency
        mode: Transport mode (ROAD, RAIL, AIR, SEA, PIPELINE, MULTIMODAL)
        vehicle_type: Type of vehicle used
        distance_km: Transportation distance in kilometers
        mass_tonnes: Mass transported in metric tonnes
        method: Calculation method (DISTANCE_BASED, SPEND_BASED, FUEL_BASED, SUPPLIER_SPECIFIC)
        fuel_type: Optional fuel type for fuel-based method
        fuel_amount_l: Optional fuel amount in liters
        spend_usd: Optional spend amount in USD
        emission_factor_id: Optional custom emission factor ID
        metadata: Additional metadata for audit trail
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculation_id: Optional[UUID] = Field(
        None, description="Optional UUID for idempotency"
    )
    mode: TransportMode = Field(..., description="Transport mode")
    vehicle_type: Optional[VehicleType] = Field(
        None, description="Vehicle type (required for most modes)"
    )
    distance_km: Optional[Decimal] = Field(
        None, ge=0, description="Distance in kilometers"
    )
    mass_tonnes: Optional[Decimal] = Field(
        None, ge=0, description="Mass in metric tonnes"
    )
    method: CalculationMethod = Field(
        CalculationMethod.DISTANCE_BASED, description="Calculation method"
    )
    fuel_type: Optional[str] = Field(None, description="Fuel type for fuel-based method")
    fuel_amount_l: Optional[Decimal] = Field(
        None, ge=0, description="Fuel amount in liters"
    )
    spend_usd: Optional[Decimal] = Field(
        None, ge=0, description="Spend amount in USD"
    )
    emission_factor_id: Optional[UUID] = Field(
        None, description="Custom emission factor ID"
    )
    origin_country: Optional[str] = Field(None, max_length=3, description="ISO 3166-1 alpha-3 origin country code")
    destination_country: Optional[str] = Field(None, max_length=3, description="ISO 3166-1 alpha-3 destination country code")
    year: int = Field(..., ge=1990, le=2100, description="Calculation year")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator('distance_km')
    def validate_distance_for_method(cls, v, values):
        """Validate distance is provided for distance-based method."""
        if values.get('method') == CalculationMethod.DISTANCE_BASED and v is None:
            raise ValueError("distance_km required for DISTANCE_BASED method")
        return v

    @validator('fuel_amount_l')
    def validate_fuel_for_method(cls, v, values):
        """Validate fuel amount is provided for fuel-based method."""
        if values.get('method') == CalculationMethod.FUEL_BASED and v is None:
            raise ValueError("fuel_amount_l required for FUEL_BASED method")
        return v

    @validator('spend_usd')
    def validate_spend_for_method(cls, v, values):
        """Validate spend is provided for spend-based method."""
        if values.get('method') == CalculationMethod.SPEND_BASED and v is None:
            raise ValueError("spend_usd required for SPEND_BASED method")
        return v


class BatchCalculateRequest(BaseModel):
    """
    Request model for batch upstream transportation emissions calculations.

    Attributes:
        tenant_id: Tenant identifier
        calculations: List of calculation requests
        batch_id: Optional batch identifier
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculations: List[CalculateRequest] = Field(
        ..., min_items=1, max_items=10000, description="List of calculations"
    )
    batch_id: Optional[UUID] = Field(None, description="Optional batch identifier")


class TransportChainCreateRequest(BaseModel):
    """
    Request model for creating a multi-modal transport chain.

    Attributes:
        tenant_id: Tenant identifier
        chain_name: Name of the transport chain
        legs: List of transport legs in sequence
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    chain_name: constr(min_length=1, max_length=200) = Field(
        ..., description="Transport chain name"
    )
    legs: List[CalculateRequest] = Field(
        ..., min_items=1, max_items=50, description="Transport legs in sequence"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class EmissionFactorCreateRequest(BaseModel):
    """
    Request model for creating custom emission factors.

    Attributes:
        tenant_id: Tenant identifier
        mode: Transport mode
        vehicle_type: Vehicle type
        factor_name: Name of the emission factor
        co2_kg_per_tkm: CO2 emissions per tonne-kilometer
        ch4_kg_per_tkm: CH4 emissions per tonne-kilometer
        n2o_kg_per_tkm: N2O emissions per tonne-kilometer
        source: Source of the emission factor
        year: Applicable year
        region: Optional region code
        metadata: Additional metadata
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    mode: TransportMode = Field(..., description="Transport mode")
    vehicle_type: VehicleType = Field(..., description="Vehicle type")
    factor_name: constr(min_length=1, max_length=200) = Field(
        ..., description="Emission factor name"
    )
    co2_kg_per_tkm: Decimal = Field(..., ge=0, description="CO2 kg per tonne-km")
    ch4_kg_per_tkm: Decimal = Field(0, ge=0, description="CH4 kg per tonne-km")
    n2o_kg_per_tkm: Decimal = Field(0, ge=0, description="N2O kg per tonne-km")
    source: constr(min_length=1, max_length=500) = Field(
        ..., description="Emission factor source"
    )
    year: int = Field(..., ge=1990, le=2100, description="Applicable year")
    region: Optional[str] = Field(None, max_length=10, description="Region code")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ClassifyRequest(BaseModel):
    """
    Request model for auto-classifying transportation activities.

    Attributes:
        tenant_id: Tenant identifier
        description: Text description of the transport activity
        additional_context: Additional context for classification
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    description: constr(min_length=1, max_length=2000) = Field(
        ..., description="Transport activity description"
    )
    additional_context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context for classification"
    )


class ComplianceCheckRequest(BaseModel):
    """
    Request model for compliance checking.

    Attributes:
        tenant_id: Tenant identifier
        calculation_id: Calculation ID to check
        frameworks: List of regulatory frameworks to check against
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculation_id: UUID = Field(..., description="Calculation ID to check")
    frameworks: List[str] = Field(
        ["GHG_PROTOCOL", "ISO_14083", "GLEC"],
        description="Regulatory frameworks"
    )


class UncertaintyRequest(BaseModel):
    """
    Request model for uncertainty analysis.

    Attributes:
        tenant_id: Tenant identifier
        calculation_id: Calculation ID to analyze
        method: Uncertainty analysis method
        confidence_level: Confidence level (0.90, 0.95, 0.99)
        monte_carlo_iterations: Number of iterations for Monte Carlo
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculation_id: UUID = Field(..., description="Calculation ID")
    method: UncertaintyMethod = Field(
        UncertaintyMethod.TIER_1, description="Uncertainty method"
    )
    confidence_level: Decimal = Field(
        Decimal("0.95"), ge=Decimal("0.80"), le=Decimal("0.99"),
        description="Confidence level"
    )
    monte_carlo_iterations: int = Field(
        10000, ge=1000, le=100000, description="Monte Carlo iterations"
    )


class AggregationRequest(BaseModel):
    """
    Request model for aggregated emissions queries.

    Attributes:
        tenant_id: Tenant identifier
        group_by: Fields to group by
        from_date: Start date for aggregation
        to_date: End date for aggregation
        filters: Additional filters
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    group_by: List[str] = Field(
        ["mode"], description="Fields to group by"
    )
    from_date: Optional[date] = Field(None, description="Start date")
    to_date: Optional[date] = Field(None, description="End date")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")


class HotSpotRequest(BaseModel):
    """
    Request model for hot-spot analysis.

    Attributes:
        tenant_id: Tenant identifier
        analysis_type: Type of hot-spot analysis
        top_n: Number of top hot-spots to return
        from_date: Start date for analysis
        to_date: End date for analysis
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    analysis_type: str = Field(
        "emissions", description="Analysis type (emissions, distance, cost)"
    )
    top_n: int = Field(10, ge=1, le=100, description="Number of hot-spots")
    from_date: Optional[date] = Field(None, description="Start date")
    to_date: Optional[date] = Field(None, description="End date")


class ExportRequest(BaseModel):
    """
    Request model for exporting calculations.

    Attributes:
        tenant_id: Tenant identifier
        format: Export format (CSV, JSON, XLSX, PDF)
        calculation_ids: Optional list of specific calculation IDs
        from_date: Start date for export
        to_date: End date for export
        include_uncertainty: Include uncertainty data
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    format: str = Field("CSV", description="Export format")
    calculation_ids: Optional[List[UUID]] = Field(
        None, description="Specific calculation IDs"
    )
    from_date: Optional[date] = Field(None, description="Start date")
    to_date: Optional[date] = Field(None, description="End date")
    include_uncertainty: bool = Field(False, description="Include uncertainty data")

    @validator('format')
    def validate_format(cls, v):
        """Validate export format."""
        allowed = ["CSV", "JSON", "XLSX", "PDF"]
        if v.upper() not in allowed:
            raise ValueError(f"Format must be one of {allowed}")
        return v.upper()


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class CalculateResponse(BaseModel):
    """Response model for single calculation."""

    calculation_id: UUID = Field(..., description="Calculation ID")
    tenant_id: str = Field(..., description="Tenant ID")
    mode: str = Field(..., description="Transport mode")
    vehicle_type: Optional[str] = Field(None, description="Vehicle type")
    method: str = Field(..., description="Calculation method")
    distance_km: Optional[Decimal] = Field(None, description="Distance")
    mass_tonnes: Optional[Decimal] = Field(None, description="Mass")
    co2_kg: Decimal = Field(..., description="CO2 emissions in kg")
    ch4_kg: Decimal = Field(..., description="CH4 emissions in kg")
    n2o_kg: Decimal = Field(..., description="N2O emissions in kg")
    co2e_kg: Decimal = Field(..., description="CO2e emissions in kg")
    emission_factor_source: str = Field(..., description="Emission factor source")
    calculation_timestamp: datetime = Field(..., description="Calculation timestamp")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")


class BatchCalculateResponse(BaseModel):
    """Response model for batch calculations."""

    batch_id: UUID = Field(..., description="Batch ID")
    tenant_id: str = Field(..., description="Tenant ID")
    total_calculations: int = Field(..., description="Total calculations")
    successful: int = Field(..., description="Successful calculations")
    failed: int = Field(..., description="Failed calculations")
    total_co2e_kg: Decimal = Field(..., description="Total CO2e emissions in kg")
    results: List[CalculateResponse] = Field(..., description="Individual results")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class TransportChainResponse(BaseModel):
    """Response model for transport chain creation."""

    chain_id: UUID = Field(..., description="Chain ID")
    tenant_id: str = Field(..., description="Tenant ID")
    chain_name: str = Field(..., description="Chain name")
    total_legs: int = Field(..., description="Number of legs")
    total_distance_km: Decimal = Field(..., description="Total distance")
    total_co2e_kg: Decimal = Field(..., description="Total CO2e emissions")
    legs: List[CalculateResponse] = Field(..., description="Leg calculations")
    created_at: datetime = Field(..., description="Creation timestamp")


class TransportChainListResponse(BaseModel):
    """Response model for transport chain listing."""

    chains: List[TransportChainResponse] = Field(..., description="Transport chains")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Page limit")
    offset: int = Field(..., description="Page offset")


class TransportChainDetailResponse(TransportChainResponse):
    """Response model for transport chain detail."""

    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")


class EmissionFactorResponse(BaseModel):
    """Response model for emission factor."""

    factor_id: UUID = Field(..., description="Factor ID")
    tenant_id: str = Field(..., description="Tenant ID")
    mode: str = Field(..., description="Transport mode")
    vehicle_type: str = Field(..., description="Vehicle type")
    factor_name: str = Field(..., description="Factor name")
    co2_kg_per_tkm: Decimal = Field(..., description="CO2 kg/tkm")
    ch4_kg_per_tkm: Decimal = Field(..., description="CH4 kg/tkm")
    n2o_kg_per_tkm: Decimal = Field(..., description="N2O kg/tkm")
    source: str = Field(..., description="Source")
    year: int = Field(..., description="Year")
    region: Optional[str] = Field(None, description="Region")
    created_at: datetime = Field(..., description="Creation timestamp")


class EmissionFactorListResponse(BaseModel):
    """Response model for emission factor listing."""

    factors: List[EmissionFactorResponse] = Field(..., description="Emission factors")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Page limit")
    offset: int = Field(..., description="Page offset")


class EmissionFactorDetailResponse(EmissionFactorResponse):
    """Response model for emission factor detail."""

    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")


class ClassificationResponse(BaseModel):
    """Response model for classification."""

    tenant_id: str = Field(..., description="Tenant ID")
    description: str = Field(..., description="Input description")
    predicted_mode: str = Field(..., description="Predicted transport mode")
    predicted_vehicle_type: Optional[str] = Field(None, description="Predicted vehicle type")
    confidence: Decimal = Field(..., ge=0, le=1, description="Confidence score")
    alternatives: List[Dict[str, Any]] = Field(
        default_factory=list, description="Alternative classifications"
    )
    classification_timestamp: datetime = Field(..., description="Timestamp")


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance check."""

    check_id: UUID = Field(..., description="Check ID")
    tenant_id: str = Field(..., description="Tenant ID")
    calculation_id: UUID = Field(..., description="Calculation ID")
    frameworks: List[str] = Field(..., description="Checked frameworks")
    overall_status: str = Field(..., description="Overall compliance status")
    findings: List[Dict[str, Any]] = Field(..., description="Compliance findings")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    check_timestamp: datetime = Field(..., description="Check timestamp")


class ComplianceDetailResponse(ComplianceCheckResponse):
    """Response model for compliance detail."""

    detailed_findings: Dict[str, Any] = Field(..., description="Detailed findings")


class UncertaintyResponse(BaseModel):
    """Response model for uncertainty analysis."""

    analysis_id: UUID = Field(..., description="Analysis ID")
    tenant_id: str = Field(..., description="Tenant ID")
    calculation_id: UUID = Field(..., description="Calculation ID")
    method: str = Field(..., description="Uncertainty method")
    confidence_level: Decimal = Field(..., description="Confidence level")
    co2e_mean_kg: Decimal = Field(..., description="Mean CO2e")
    co2e_std_kg: Decimal = Field(..., description="Standard deviation CO2e")
    co2e_lower_bound_kg: Decimal = Field(..., description="Lower bound CO2e")
    co2e_upper_bound_kg: Decimal = Field(..., description="Upper bound CO2e")
    relative_uncertainty_pct: Decimal = Field(..., description="Relative uncertainty %")
    analysis_timestamp: datetime = Field(..., description="Analysis timestamp")


class AggregationResponse(BaseModel):
    """Response model for aggregations."""

    tenant_id: str = Field(..., description="Tenant ID")
    group_by: List[str] = Field(..., description="Grouped by fields")
    from_date: Optional[date] = Field(None, description="Start date")
    to_date: Optional[date] = Field(None, description="End date")
    aggregations: List[Dict[str, Any]] = Field(..., description="Aggregated data")
    total_co2e_kg: Decimal = Field(..., description="Total CO2e")
    total_calculations: int = Field(..., description="Total calculations")


class HotSpotResponse(BaseModel):
    """Response model for hot-spot analysis."""

    tenant_id: str = Field(..., description="Tenant ID")
    analysis_type: str = Field(..., description="Analysis type")
    top_n: int = Field(..., description="Number of hot-spots")
    hot_spots: List[Dict[str, Any]] = Field(..., description="Hot-spot data")
    total_co2e_kg: Decimal = Field(..., description="Total CO2e")
    analysis_timestamp: datetime = Field(..., description="Analysis timestamp")


class ExportResponse(BaseModel):
    """Response model for export."""

    export_id: UUID = Field(..., description="Export ID")
    tenant_id: str = Field(..., description="Tenant ID")
    format: str = Field(..., description="Export format")
    file_url: str = Field(..., description="Download URL")
    record_count: int = Field(..., description="Number of records")
    file_size_bytes: int = Field(..., description="File size in bytes")
    expires_at: datetime = Field(..., description="URL expiration")
    created_at: datetime = Field(..., description="Creation timestamp")


class CalculationListResponse(BaseModel):
    """Response model for calculation listing."""

    calculations: List[CalculateResponse] = Field(..., description="Calculations")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Page limit")
    offset: int = Field(..., description="Page offset")


class CalculationDetailResponse(CalculateResponse):
    """Response model for calculation detail."""

    uncertainty_data: Optional[Dict[str, Any]] = Field(None, description="Uncertainty data")
    compliance_status: Optional[Dict[str, Any]] = Field(None, description="Compliance status")


class DeleteResponse(BaseModel):
    """Response model for deletion."""

    deleted: bool = Field(..., description="Deletion status")
    calculation_id: UUID = Field(..., description="Deleted calculation ID")
    timestamp: datetime = Field(..., description="Deletion timestamp")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(..., description="Check timestamp")
    database_connected: bool = Field(..., description="Database connection status")
    cache_connected: bool = Field(..., description="Cache connection status")


class StatsResponse(BaseModel):
    """Response model for statistics."""

    total_calculations: int = Field(..., description="Total calculations")
    total_co2e_kg: Decimal = Field(..., description="Total CO2e emissions")
    calculations_by_mode: Dict[str, int] = Field(..., description="Calculations by mode")
    calculations_by_method: Dict[str, int] = Field(..., description="Calculations by method")
    average_distance_km: Decimal = Field(..., description="Average distance")
    average_co2e_per_tkm: Decimal = Field(..., description="Average CO2e per tkm")
    timestamp: datetime = Field(..., description="Stats timestamp")


# ============================================================================
# SERVICE DEPENDENCY
# ============================================================================


_service_instance: Optional[UpstreamTransportationService] = None


def get_service() -> UpstreamTransportationService:
    """
    Get or create UpstreamTransportationService instance.

    Returns:
        UpstreamTransportationService instance

    Raises:
        HTTPException: If service initialization fails
    """
    global _service_instance

    if _service_instance is None:
        try:
            _service_instance = UpstreamTransportationService()
            logger.info("UpstreamTransportationService initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize UpstreamTransportationService: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service initialization failed"
            )

    return _service_instance


# ============================================================================
# ENDPOINTS - CALCULATIONS (5)
# ============================================================================


@router.post(
    "/calculate",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate upstream transportation emissions",
    description=(
        "Calculate GHG emissions for a single upstream transportation activity. "
        "Supports multiple transport modes (road, rail, air, sea, pipeline, multimodal) "
        "and calculation methods (distance-based, spend-based, fuel-based, supplier-specific). "
        "Returns deterministic results with provenance hash for audit trail."
    ),
)
async def calculate_emissions(
    request: CalculateRequest,
    service: UpstreamTransportationService = Depends(get_service),
) -> CalculateResponse:
    """
    Calculate upstream transportation emissions for a single activity.

    Args:
        request: Calculation request with activity data
        service: UpstreamTransportationService instance

    Returns:
        CalculateResponse with emissions and metadata

    Raises:
        HTTPException: If calculation fails or validation errors occur
    """
    try:
        logger.info(
            f"Calculating emissions for tenant {request.tenant_id}, "
            f"mode {request.mode}, method {request.method}"
        )

        result = await service.calculate(request.dict())

        return CalculateResponse(**result)

    except ValueError as e:
        logger.error("Validation error in calculate_emissions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Error in calculate_emissions: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Calculation failed"
        )


@router.post(
    "/calculate/batch",
    response_model=BatchCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate batch upstream transportation emissions",
    description=(
        "Calculate GHG emissions for multiple upstream transportation activities in a single request. "
        "Processes up to 10,000 calculations with parallel execution. "
        "Returns aggregated results with individual calculation details and error handling."
    ),
)
async def calculate_batch_emissions(
    request: BatchCalculateRequest,
    service: UpstreamTransportationService = Depends(get_service),
) -> BatchCalculateResponse:
    """
    Calculate upstream transportation emissions for multiple activities.

    Args:
        request: Batch calculation request
        service: UpstreamTransportationService instance

    Returns:
        BatchCalculateResponse with aggregated results

    Raises:
        HTTPException: If batch calculation fails
    """
    try:
        logger.info(
            f"Calculating batch emissions for tenant {request.tenant_id}, "
            f"count {len(request.calculations)}"
        )

        result = await service.calculate_batch(request.dict())

        return BatchCalculateResponse(**result)

    except ValueError as e:
        logger.error("Validation error in calculate_batch_emissions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Error in calculate_batch_emissions: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch calculation failed"
        )


@router.get(
    "/calculations",
    response_model=CalculationListResponse,
    summary="List upstream transportation calculations",
    description=(
        "Retrieve a paginated list of upstream transportation calculations. "
        "Supports filtering by tenant, mode, method, and date range. "
        "Returns summary information for each calculation."
    ),
)
async def list_calculations(
    tenant_id: str = Query(..., description="Tenant identifier"),
    mode: Optional[TransportMode] = Query(None, description="Filter by transport mode"),
    method: Optional[CalculationMethod] = Query(None, description="Filter by calculation method"),
    from_date: Optional[date] = Query(None, description="Filter from date"),
    to_date: Optional[date] = Query(None, description="Filter to date"),
    limit: int = Query(100, ge=1, le=1000, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    service: UpstreamTransportationService = Depends(get_service),
) -> CalculationListResponse:
    """
    List upstream transportation calculations with filtering and pagination.

    Args:
        tenant_id: Tenant identifier
        mode: Optional transport mode filter
        method: Optional calculation method filter
        from_date: Optional start date filter
        to_date: Optional end date filter
        limit: Maximum number of results
        offset: Number of results to skip
        service: UpstreamTransportationService instance

    Returns:
        CalculationListResponse with paginated results

    Raises:
        HTTPException: If listing fails
    """
    try:
        logger.info("Listing calculations for tenant %s", tenant_id)

        filters = {
            "tenant_id": tenant_id,
            "mode": mode.value if mode else None,
            "method": method.value if method else None,
            "from_date": from_date,
            "to_date": to_date,
            "limit": limit,
            "offset": offset,
        }

        result = await service.list_calculations(filters)

        return CalculationListResponse(**result)

    except Exception as e:
        logger.error("Error in list_calculations: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list calculations"
        )


@router.get(
    "/calculations/{calculation_id}",
    response_model=CalculationDetailResponse,
    summary="Get upstream transportation calculation detail",
    description=(
        "Retrieve detailed information for a specific upstream transportation calculation. "
        "Includes emissions data, uncertainty information, and compliance status."
    ),
)
async def get_calculation_detail(
    calculation_id: UUID = Path(..., description="Calculation ID"),
    tenant_id: str = Query(..., description="Tenant identifier"),
    service: UpstreamTransportationService = Depends(get_service),
) -> CalculationDetailResponse:
    """
    Get detailed information for a specific calculation.

    Args:
        calculation_id: Calculation UUID
        tenant_id: Tenant identifier
        service: UpstreamTransportationService instance

    Returns:
        CalculationDetailResponse with full calculation data

    Raises:
        HTTPException: If calculation not found or access denied
    """
    try:
        logger.info("Getting calculation detail %s for tenant %s", calculation_id, tenant_id)

        result = await service.get_calculation(calculation_id, tenant_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found"
            )

        return CalculationDetailResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_calculation_detail: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve calculation"
        )


@router.delete(
    "/calculations/{calculation_id}",
    response_model=DeleteResponse,
    summary="Delete upstream transportation calculation",
    description=(
        "Delete a specific upstream transportation calculation. "
        "Requires tenant ownership. Soft delete with audit trail."
    ),
)
async def delete_calculation(
    calculation_id: UUID = Path(..., description="Calculation ID"),
    tenant_id: str = Query(..., description="Tenant identifier"),
    service: UpstreamTransportationService = Depends(get_service),
) -> DeleteResponse:
    """
    Delete a specific calculation.

    Args:
        calculation_id: Calculation UUID
        tenant_id: Tenant identifier
        service: UpstreamTransportationService instance

    Returns:
        DeleteResponse with deletion confirmation

    Raises:
        HTTPException: If calculation not found or deletion fails
    """
    try:
        logger.info("Deleting calculation %s for tenant %s", calculation_id, tenant_id)

        deleted = await service.delete_calculation(calculation_id, tenant_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found"
            )

        return DeleteResponse(
            deleted=True,
            calculation_id=calculation_id,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in delete_calculation: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete calculation"
        )


# ============================================================================
# ENDPOINTS - TRANSPORT CHAINS (3)
# ============================================================================


@router.post(
    "/transport-chains",
    response_model=TransportChainResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create multi-modal transport chain",
    description=(
        "Create a multi-modal transport chain with multiple sequential legs. "
        "Calculates emissions for each leg and provides total emissions across the chain. "
        "Useful for complex supply chain transportation scenarios."
    ),
)
async def create_transport_chain(
    request: TransportChainCreateRequest,
    service: UpstreamTransportationService = Depends(get_service),
) -> TransportChainResponse:
    """
    Create a multi-modal transport chain.

    Args:
        request: Transport chain creation request
        service: UpstreamTransportationService instance

    Returns:
        TransportChainResponse with chain details and leg calculations

    Raises:
        HTTPException: If chain creation fails
    """
    try:
        logger.info(
            f"Creating transport chain '{request.chain_name}' for tenant {request.tenant_id}"
        )

        result = await service.create_transport_chain(request.dict())

        return TransportChainResponse(**result)

    except ValueError as e:
        logger.error("Validation error in create_transport_chain: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Error in create_transport_chain: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transport chain creation failed"
        )


@router.get(
    "/transport-chains",
    response_model=TransportChainListResponse,
    summary="List transport chains",
    description=(
        "Retrieve a paginated list of transport chains for a tenant. "
        "Returns summary information for each chain including total emissions."
    ),
)
async def list_transport_chains(
    tenant_id: str = Query(..., description="Tenant identifier"),
    limit: int = Query(100, ge=1, le=1000, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    service: UpstreamTransportationService = Depends(get_service),
) -> TransportChainListResponse:
    """
    List transport chains with pagination.

    Args:
        tenant_id: Tenant identifier
        limit: Maximum number of results
        offset: Number of results to skip
        service: UpstreamTransportationService instance

    Returns:
        TransportChainListResponse with paginated results

    Raises:
        HTTPException: If listing fails
    """
    try:
        logger.info("Listing transport chains for tenant %s", tenant_id)

        result = await service.list_transport_chains(tenant_id, limit, offset)

        return TransportChainListResponse(**result)

    except Exception as e:
        logger.error("Error in list_transport_chains: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list transport chains"
        )


@router.get(
    "/transport-chains/{chain_id}",
    response_model=TransportChainDetailResponse,
    summary="Get transport chain detail",
    description=(
        "Retrieve detailed information for a specific transport chain. "
        "Includes all leg calculations and metadata."
    ),
)
async def get_transport_chain_detail(
    chain_id: UUID = Path(..., description="Chain ID"),
    tenant_id: str = Query(..., description="Tenant identifier"),
    service: UpstreamTransportationService = Depends(get_service),
) -> TransportChainDetailResponse:
    """
    Get detailed information for a specific transport chain.

    Args:
        chain_id: Chain UUID
        tenant_id: Tenant identifier
        service: UpstreamTransportationService instance

    Returns:
        TransportChainDetailResponse with full chain data

    Raises:
        HTTPException: If chain not found or access denied
    """
    try:
        logger.info("Getting transport chain detail %s for tenant %s", chain_id, tenant_id)

        result = await service.get_transport_chain(chain_id, tenant_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transport chain {chain_id} not found"
            )

        return TransportChainDetailResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_transport_chain_detail: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transport chain"
        )


# ============================================================================
# ENDPOINTS - EMISSION FACTORS (3)
# ============================================================================


@router.get(
    "/emission-factors",
    response_model=EmissionFactorListResponse,
    summary="List emission factors",
    description=(
        "Retrieve available emission factors for upstream transportation. "
        "Supports filtering by transport mode and vehicle type. "
        "Includes both standard and custom emission factors."
    ),
)
async def list_emission_factors(
    tenant_id: str = Query(..., description="Tenant identifier"),
    mode: Optional[TransportMode] = Query(None, description="Filter by transport mode"),
    vehicle_type: Optional[VehicleType] = Query(None, description="Filter by vehicle type"),
    limit: int = Query(100, ge=1, le=1000, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    service: UpstreamTransportationService = Depends(get_service),
) -> EmissionFactorListResponse:
    """
    List emission factors with filtering and pagination.

    Args:
        tenant_id: Tenant identifier
        mode: Optional transport mode filter
        vehicle_type: Optional vehicle type filter
        limit: Maximum number of results
        offset: Number of results to skip
        service: UpstreamTransportationService instance

    Returns:
        EmissionFactorListResponse with paginated results

    Raises:
        HTTPException: If listing fails
    """
    try:
        logger.info("Listing emission factors for tenant %s", tenant_id)

        filters = {
            "tenant_id": tenant_id,
            "mode": mode.value if mode else None,
            "vehicle_type": vehicle_type.value if vehicle_type else None,
            "limit": limit,
            "offset": offset,
        }

        result = await service.list_emission_factors(filters)

        return EmissionFactorListResponse(**result)

    except Exception as e:
        logger.error("Error in list_emission_factors: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list emission factors"
        )


@router.get(
    "/emission-factors/{factor_id}",
    response_model=EmissionFactorDetailResponse,
    summary="Get emission factor detail",
    description=(
        "Retrieve detailed information for a specific emission factor. "
        "Includes source, applicability, and metadata."
    ),
)
async def get_emission_factor_detail(
    factor_id: UUID = Path(..., description="Factor ID"),
    tenant_id: str = Query(..., description="Tenant identifier"),
    service: UpstreamTransportationService = Depends(get_service),
) -> EmissionFactorDetailResponse:
    """
    Get detailed information for a specific emission factor.

    Args:
        factor_id: Factor UUID
        tenant_id: Tenant identifier
        service: UpstreamTransportationService instance

    Returns:
        EmissionFactorDetailResponse with full factor data

    Raises:
        HTTPException: If factor not found or access denied
    """
    try:
        logger.info("Getting emission factor detail %s for tenant %s", factor_id, tenant_id)

        result = await service.get_emission_factor(factor_id, tenant_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Emission factor {factor_id} not found"
            )

        return EmissionFactorDetailResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_emission_factor_detail: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve emission factor"
        )


@router.post(
    "/emission-factors/custom",
    response_model=EmissionFactorResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create custom emission factor",
    description=(
        "Create a custom emission factor for tenant-specific transportation scenarios. "
        "Requires source documentation and year of applicability. "
        "Custom factors override standard factors for the tenant."
    ),
)
async def create_custom_emission_factor(
    request: EmissionFactorCreateRequest,
    service: UpstreamTransportationService = Depends(get_service),
) -> EmissionFactorResponse:
    """
    Create a custom emission factor.

    Args:
        request: Emission factor creation request
        service: UpstreamTransportationService instance

    Returns:
        EmissionFactorResponse with created factor

    Raises:
        HTTPException: If creation fails or validation errors occur
    """
    try:
        logger.info(
            f"Creating custom emission factor '{request.factor_name}' for tenant {request.tenant_id}"
        )

        result = await service.create_emission_factor(request.dict())

        return EmissionFactorResponse(**result)

    except ValueError as e:
        logger.error("Validation error in create_custom_emission_factor: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Error in create_custom_emission_factor: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Emission factor creation failed"
        )


# ============================================================================
# ENDPOINTS - CLASSIFICATION (1)
# ============================================================================


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Auto-classify transportation activity",
    description=(
        "Automatically classify transportation activities based on text descriptions. "
        "Uses ML model to predict transport mode and vehicle type. "
        "Returns confidence scores and alternative classifications."
    ),
)
async def classify_activity(
    request: ClassifyRequest,
    service: UpstreamTransportationService = Depends(get_service),
) -> ClassificationResponse:
    """
    Auto-classify a transportation activity.

    Args:
        request: Classification request with activity description
        service: UpstreamTransportationService instance

    Returns:
        ClassificationResponse with predicted classifications

    Raises:
        HTTPException: If classification fails
    """
    try:
        logger.info("Classifying activity for tenant %s", request.tenant_id)

        result = await service.classify_activity(request.dict())

        return ClassificationResponse(**result)

    except Exception as e:
        logger.error("Error in classify_activity: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Classification failed"
        )


# ============================================================================
# ENDPOINTS - COMPLIANCE (2)
# ============================================================================


@router.post(
    "/compliance/check",
    response_model=ComplianceCheckResponse,
    summary="Check calculation compliance",
    description=(
        "Check calculation compliance against regulatory frameworks. "
        "Supports GHG Protocol Scope 3, ISO 14083, GLEC Framework, and EN 16258. "
        "Returns compliance status, findings, and recommendations."
    ),
)
async def check_compliance(
    request: ComplianceCheckRequest,
    service: UpstreamTransportationService = Depends(get_service),
) -> ComplianceCheckResponse:
    """
    Check calculation compliance against regulatory frameworks.

    Args:
        request: Compliance check request
        service: UpstreamTransportationService instance

    Returns:
        ComplianceCheckResponse with compliance findings

    Raises:
        HTTPException: If compliance check fails
    """
    try:
        logger.info(
            f"Checking compliance for calculation {request.calculation_id}, "
            f"tenant {request.tenant_id}"
        )

        result = await service.check_compliance(request.dict())

        return ComplianceCheckResponse(**result)

    except ValueError as e:
        logger.error("Validation error in check_compliance: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Error in check_compliance: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance check failed"
        )


@router.get(
    "/compliance/{check_id}",
    response_model=ComplianceDetailResponse,
    summary="Get compliance check detail",
    description=(
        "Retrieve detailed compliance check results. "
        "Includes framework-specific findings and remediation guidance."
    ),
)
async def get_compliance_detail(
    check_id: UUID = Path(..., description="Check ID"),
    tenant_id: str = Query(..., description="Tenant identifier"),
    service: UpstreamTransportationService = Depends(get_service),
) -> ComplianceDetailResponse:
    """
    Get detailed compliance check results.

    Args:
        check_id: Check UUID
        tenant_id: Tenant identifier
        service: UpstreamTransportationService instance

    Returns:
        ComplianceDetailResponse with detailed findings

    Raises:
        HTTPException: If check not found or access denied
    """
    try:
        logger.info("Getting compliance detail %s for tenant %s", check_id, tenant_id)

        result = await service.get_compliance_check(check_id, tenant_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Compliance check {check_id} not found"
            )

        return ComplianceDetailResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_compliance_detail: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance check"
        )


# ============================================================================
# ENDPOINTS - UNCERTAINTY (1)
# ============================================================================


@router.post(
    "/uncertainty",
    response_model=UncertaintyResponse,
    summary="Analyze calculation uncertainty",
    description=(
        "Perform uncertainty analysis on emissions calculations. "
        "Supports Tier 1 (default factors), Tier 2 (custom factors), "
        "and Monte Carlo simulation methods. "
        "Returns confidence intervals and relative uncertainty."
    ),
)
async def analyze_uncertainty(
    request: UncertaintyRequest,
    service: UpstreamTransportationService = Depends(get_service),
) -> UncertaintyResponse:
    """
    Perform uncertainty analysis on a calculation.

    Args:
        request: Uncertainty analysis request
        service: UpstreamTransportationService instance

    Returns:
        UncertaintyResponse with uncertainty metrics

    Raises:
        HTTPException: If analysis fails
    """
    try:
        logger.info(
            f"Analyzing uncertainty for calculation {request.calculation_id}, "
            f"tenant {request.tenant_id}, method {request.method}"
        )

        result = await service.analyze_uncertainty(request.dict())

        return UncertaintyResponse(**result)

    except ValueError as e:
        logger.error("Validation error in analyze_uncertainty: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Error in analyze_uncertainty: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Uncertainty analysis failed"
        )


# ============================================================================
# ENDPOINTS - AGGREGATIONS (2)
# ============================================================================


@router.get(
    "/aggregations",
    response_model=AggregationResponse,
    summary="Get aggregated emissions",
    description=(
        "Retrieve aggregated emissions data grouped by specified dimensions. "
        "Supports grouping by mode, method, vehicle type, origin, destination, etc. "
        "Useful for reporting and analysis."
    ),
)
async def get_aggregations(
    tenant_id: str = Query(..., description="Tenant identifier"),
    group_by: List[str] = Query(["mode"], description="Fields to group by"),
    from_date: Optional[date] = Query(None, description="Start date"),
    to_date: Optional[date] = Query(None, description="End date"),
    service: UpstreamTransportationService = Depends(get_service),
) -> AggregationResponse:
    """
    Get aggregated emissions data.

    Args:
        tenant_id: Tenant identifier
        group_by: Fields to group by
        from_date: Optional start date filter
        to_date: Optional end date filter
        service: UpstreamTransportationService instance

    Returns:
        AggregationResponse with aggregated data

    Raises:
        HTTPException: If aggregation fails
    """
    try:
        logger.info("Getting aggregations for tenant %s, group_by %s", tenant_id, group_by)

        filters = {
            "tenant_id": tenant_id,
            "group_by": group_by,
            "from_date": from_date,
            "to_date": to_date,
        }

        result = await service.get_aggregations(filters)

        return AggregationResponse(**result)

    except Exception as e:
        logger.error("Error in get_aggregations: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Aggregation failed"
        )


@router.get(
    "/hot-spots",
    response_model=HotSpotResponse,
    summary="Identify emissions hot-spots",
    description=(
        "Identify top emissions hot-spots across transportation activities. "
        "Analyzes by emissions, distance, or cost. "
        "Useful for prioritizing reduction efforts."
    ),
)
async def get_hot_spots(
    tenant_id: str = Query(..., description="Tenant identifier"),
    analysis_type: str = Query("emissions", description="Analysis type"),
    top_n: int = Query(10, ge=1, le=100, description="Number of hot-spots"),
    from_date: Optional[date] = Query(None, description="Start date"),
    to_date: Optional[date] = Query(None, description="End date"),
    service: UpstreamTransportationService = Depends(get_service),
) -> HotSpotResponse:
    """
    Identify emissions hot-spots.

    Args:
        tenant_id: Tenant identifier
        analysis_type: Type of analysis (emissions, distance, cost)
        top_n: Number of hot-spots to return
        from_date: Optional start date filter
        to_date: Optional end date filter
        service: UpstreamTransportationService instance

    Returns:
        HotSpotResponse with hot-spot data

    Raises:
        HTTPException: If analysis fails
    """
    try:
        logger.info(
            f"Getting hot-spots for tenant {tenant_id}, "
            f"type {analysis_type}, top {top_n}"
        )

        request_data = {
            "tenant_id": tenant_id,
            "analysis_type": analysis_type,
            "top_n": top_n,
            "from_date": from_date,
            "to_date": to_date,
        }

        result = await service.get_hot_spots(request_data)

        return HotSpotResponse(**result)

    except Exception as e:
        logger.error("Error in get_hot_spots: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hot-spot analysis failed"
        )


# ============================================================================
# ENDPOINTS - EXPORT (1)
# ============================================================================


@router.post(
    "/export",
    response_model=ExportResponse,
    summary="Export calculations",
    description=(
        "Export calculations to various formats (CSV, JSON, XLSX, PDF). "
        "Supports filtering by date range or specific calculation IDs. "
        "Returns download URL with expiration."
    ),
)
async def export_calculations(
    request: ExportRequest,
    service: UpstreamTransportationService = Depends(get_service),
) -> ExportResponse:
    """
    Export calculations to specified format.

    Args:
        request: Export request with format and filters
        service: UpstreamTransportationService instance

    Returns:
        ExportResponse with download URL

    Raises:
        HTTPException: If export fails
    """
    try:
        logger.info(
            f"Exporting calculations for tenant {request.tenant_id}, "
            f"format {request.format}"
        )

        result = await service.export_calculations(request.dict())

        return ExportResponse(**result)

    except ValueError as e:
        logger.error("Validation error in export_calculations: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Error in export_calculations: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Export failed"
        )


# ============================================================================
# ENDPOINTS - HEALTH (2)
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Check service health status. "
        "Verifies database and cache connectivity. "
        "Returns service version and status."
    ),
)
async def health_check(
    service: UpstreamTransportationService = Depends(get_service),
) -> HealthResponse:
    """
    Perform health check.

    Args:
        service: UpstreamTransportationService instance

    Returns:
        HealthResponse with service status
    """
    try:
        health_data = await service.health_check()

        return HealthResponse(**health_data)

    except Exception as e:
        logger.error("Error in health_check: %s", e, exc_info=True)
        return HealthResponse(
            status="unhealthy",
            service="upstream-transportation",
            version="1.0.0",
            timestamp=datetime.utcnow(),
            database_connected=False,
            cache_connected=False,
        )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get service statistics",
    description=(
        "Retrieve service statistics including total calculations, emissions, "
        "and breakdowns by mode and method. "
        "Useful for monitoring and reporting."
    ),
)
async def get_stats(
    tenant_id: Optional[str] = Query(None, description="Optional tenant filter"),
    service: UpstreamTransportationService = Depends(get_service),
) -> StatsResponse:
    """
    Get service statistics.

    Args:
        tenant_id: Optional tenant identifier for filtering
        service: UpstreamTransportationService instance

    Returns:
        StatsResponse with service statistics

    Raises:
        HTTPException: If stats retrieval fails
    """
    try:
        logger.info("Getting stats for tenant %s", tenant_id or 'all')

        stats_data = await service.get_stats(tenant_id)

        return StatsResponse(**stats_data)

    except Exception as e:
        logger.error("Error in get_stats: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )
