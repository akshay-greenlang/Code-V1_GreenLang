"""
GL-EUDR-002: FastAPI Router

REST API endpoints for the Geolocation Collector Agent.
All endpoints require authentication and support multi-tenancy.

Endpoints:
- POST /plots - Submit a new plot
- POST /plots/validate - Validate coordinates without storing
- POST /plots/bulk - Bulk upload plots (async)
- GET /plots - List plots with filtering
- GET /plots/{plot_id} - Get a specific plot
- POST /plots/{plot_id}/validate - Re-validate existing plot
- POST /plots/{plot_id}/enrich - Enrich plot with geocoding
- GET /bulk/{job_id} - Get bulk upload job status
"""

import logging
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field

from .agent import (
    GeolocationCollectorAgent,
    GeolocationInput,
    GeolocationOutput,
    GeolocationValidator,
    PointCoordinates,
    PolygonCoordinates,
    Plot,
    PlotValidationHistory,
    BulkUploadJob,
    CommodityType,
    GeometryType,
    ValidationStatus,
    CollectionMethod,
    OperationType,
    BulkUploadFormat,
    BulkJobStatus,
)

# Import auth components (assuming same pattern as GL-EUDR-001)
try:
    from ..gl_eudr_001_supply_chain_mapper.auth import (
        User,
        UserRole,
        Permission,
        get_current_user,
        require_permissions,
        require_role,
        ResourceOwnershipVerifier,
        PIIMasker,
        RateLimiter,
    )
    from ..gl_eudr_001_supply_chain_mapper.audit import (
        AuditLogger,
        AuditContext,
        AuditAction,
        get_audit_logger,
    )
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    # Provide mock implementations for standalone testing
    class User:
        user_id = uuid.uuid4()
        email = "test@example.com"
        role = "analyst"
        organization_id = uuid.uuid4()
        permissions = []

    def get_current_user():
        return User()

    def require_permissions(*args):
        def decorator(func):
            return func
        return decorator

    class Permission:
        PLOT_READ = "plot:read"
        PLOT_WRITE = "plot:write"
        PLOT_DELETE = "plot:delete"


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["geolocation"])


# =============================================================================
# AGENT MANAGER (Thread-safe singleton)
# =============================================================================

class AgentManager:
    """Thread-safe manager for agent instances per organization."""
    _instances: Dict[UUID, GeolocationCollectorAgent] = {}

    @classmethod
    def get_agent(cls, organization_id: UUID) -> GeolocationCollectorAgent:
        """Get or create agent for organization."""
        if organization_id not in cls._instances:
            cls._instances[organization_id] = GeolocationCollectorAgent()
        return cls._instances[organization_id]


def get_agent(current_user: User = Depends(get_current_user)) -> GeolocationCollectorAgent:
    """Dependency to get agent for current user's organization."""
    return AgentManager.get_agent(current_user.organization_id)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PointCoordinateRequest(BaseModel):
    """Point coordinate input."""
    type: str = Field("point", const=True)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class PolygonCoordinateRequest(BaseModel):
    """Polygon coordinate input."""
    type: str = Field("polygon", const=True)
    coordinates: List[List[float]] = Field(
        ...,
        min_items=4,
        description="Array of [longitude, latitude] pairs"
    )


class PlotSubmissionRequest(BaseModel):
    """Request to submit a new plot."""
    supplier_id: UUID
    external_id: Optional[str] = None
    coordinates: Union[PointCoordinateRequest, PolygonCoordinateRequest]
    country_code: str = Field(..., pattern=r"^[A-Z]{2}$")
    commodity: CommodityType
    declared_area_hectares: Optional[float] = Field(None, gt=0)
    collection_method: CollectionMethod = CollectionMethod.API
    collection_device: Optional[str] = None
    collection_accuracy_m: Optional[float] = Field(None, ge=0)
    collection_date: Optional[str] = None
    collected_by: Optional[str] = None
    crop_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidateCoordinatesRequest(BaseModel):
    """Request to validate coordinates without storing."""
    coordinates: Union[PointCoordinateRequest, PolygonCoordinateRequest]
    country_code: str = Field(..., pattern=r"^[A-Z]{2}$")
    commodity: CommodityType
    declared_area_hectares: Optional[float] = Field(None, gt=0)
    collection_accuracy_m: Optional[float] = Field(None, ge=0)


class UpdatePlotRequest(BaseModel):
    """Request to update a plot."""
    external_id: Optional[str] = None
    crop_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ValidationErrorResponse(BaseModel):
    """Validation error in response."""
    code: str
    message: str
    severity: str
    coordinate: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationResultResponse(BaseModel):
    """Validation result response."""
    valid: bool
    status: str
    errors: List[ValidationErrorResponse] = []
    warnings: List[ValidationErrorResponse] = []
    computed: Dict[str, Any] = Field(default_factory=dict)


class PlotResponse(BaseModel):
    """Plot response."""
    plot_id: UUID
    supplier_id: UUID
    external_id: Optional[str] = None
    geometry_type: str
    coordinates: Dict[str, Any]
    centroid: Optional[Dict[str, float]] = None
    bounding_box: Optional[List[float]] = None
    area_hectares: Optional[float] = None
    perimeter_km: Optional[float] = None
    country_code: str
    admin_level_1: Optional[str] = None
    admin_level_2: Optional[str] = None
    admin_level_3: Optional[str] = None
    commodity: str
    crop_type: Optional[str] = None
    validation_status: str
    error_count: int = 0
    warning_count: int = 0
    precision_lat: Optional[int] = None
    precision_lon: Optional[int] = None
    last_validated_at: Optional[datetime] = None
    collection_method: str
    collection_accuracy_m: Optional[float] = None
    in_protected_area: bool = False
    protected_area_name: Optional[str] = None
    in_urban_area: bool = False
    biome: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class PlotListResponse(BaseModel):
    """Response for plot listing."""
    plots: List[PlotResponse]
    total_count: int
    limit: int
    offset: int


class BulkUploadResponse(BaseModel):
    """Response for bulk upload initiation."""
    job_id: UUID
    status: str
    message: str


class BulkJobStatusResponse(BaseModel):
    """Response for bulk job status."""
    job_id: UUID
    status: str
    total_count: int
    processed_count: int
    valid_count: int
    invalid_count: int
    warning_count: int
    progress_percent: float
    report_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def convert_coordinates_to_model(
    coords: Union[PointCoordinateRequest, PolygonCoordinateRequest]
) -> Union[PointCoordinates, PolygonCoordinates]:
    """Convert request coordinates to agent model."""
    if isinstance(coords, PointCoordinateRequest) or coords.type == "point":
        return PointCoordinates(
            latitude=coords.latitude,
            longitude=coords.longitude
        )
    else:
        return PolygonCoordinates(coordinates=coords.coordinates)


def plot_to_response(plot: Plot) -> PlotResponse:
    """Convert Plot model to response."""
    # Convert coordinates to dict
    if isinstance(plot.coordinates, PointCoordinates):
        coords_dict = {
            "type": "point",
            "latitude": plot.coordinates.latitude,
            "longitude": plot.coordinates.longitude
        }
    else:
        coords_dict = {
            "type": "polygon",
            "coordinates": plot.coordinates.coordinates
        }

    # Convert centroid
    centroid_dict = None
    if plot.centroid:
        centroid_dict = {
            "latitude": plot.centroid.latitude,
            "longitude": plot.centroid.longitude
        }

    return PlotResponse(
        plot_id=plot.plot_id,
        supplier_id=plot.supplier_id,
        external_id=plot.external_id,
        geometry_type=plot.geometry_type.value if hasattr(plot.geometry_type, 'value') else plot.geometry_type,
        coordinates=coords_dict,
        centroid=centroid_dict,
        bounding_box=plot.bounding_box,
        area_hectares=float(plot.area_hectares) if plot.area_hectares else None,
        perimeter_km=float(plot.perimeter_km) if plot.perimeter_km else None,
        country_code=plot.country_code,
        admin_level_1=plot.admin_level_1,
        admin_level_2=plot.admin_level_2,
        admin_level_3=plot.admin_level_3,
        commodity=plot.commodity.value if hasattr(plot.commodity, 'value') else plot.commodity,
        crop_type=plot.crop_type,
        validation_status=plot.validation_status.value if hasattr(plot.validation_status, 'value') else plot.validation_status,
        error_count=len(plot.validation_errors),
        warning_count=len(plot.validation_warnings),
        precision_lat=plot.precision_lat,
        precision_lon=plot.precision_lon,
        last_validated_at=plot.last_validated_at,
        collection_method=plot.collection_method.value if hasattr(plot.collection_method, 'value') else plot.collection_method,
        collection_accuracy_m=plot.collection_accuracy_m,
        in_protected_area=plot.in_protected_area,
        protected_area_name=plot.protected_area_name,
        in_urban_area=plot.in_urban_area,
        biome=plot.biome,
        created_at=plot.created_at,
        updated_at=plot.updated_at
    )


def validation_result_to_response(result) -> ValidationResultResponse:
    """Convert ValidationResult to response."""
    errors = []
    for e in result.errors:
        errors.append(ValidationErrorResponse(
            code=e.code.value if hasattr(e.code, 'value') else e.code,
            message=e.message,
            severity=e.severity.value if hasattr(e.severity, 'value') else e.severity,
            coordinate=list(e.coordinate) if e.coordinate else None,
            metadata=e.metadata
        ))

    warnings = []
    for w in result.warnings:
        warnings.append(ValidationErrorResponse(
            code=w.code.value if hasattr(w.code, 'value') else w.code,
            message=w.message,
            severity=w.severity.value if hasattr(w.severity, 'value') else w.severity,
            coordinate=list(w.coordinate) if w.coordinate else None,
            metadata=w.metadata
        ))

    return ValidationResultResponse(
        valid=result.valid,
        status=result.status.value if hasattr(result.status, 'value') else result.status,
        errors=errors,
        warnings=warnings,
        computed=result.metadata
    )


# =============================================================================
# RATE LIMITER
# =============================================================================

# Rate limiters for expensive operations
_validation_limiter = RateLimiter(max_requests=100, window_seconds=60) if AUTH_AVAILABLE else None
_bulk_limiter = RateLimiter(max_requests=10, window_seconds=3600) if AUTH_AVAILABLE else None


def check_validation_rate_limit(user: User):
    """Check rate limit for validation operations."""
    if _validation_limiter:
        allowed, remaining = _validation_limiter.check(str(user.user_id))
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for validation operations"
            )


def check_bulk_rate_limit(user: User):
    """Check rate limit for bulk operations."""
    if _bulk_limiter:
        allowed, remaining = _bulk_limiter.check(str(user.user_id))
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for bulk uploads"
            )


# =============================================================================
# VALIDATION ENDPOINTS
# =============================================================================

@router.post(
    "/plots/validate",
    response_model=ValidationResultResponse,
    summary="Validate coordinates without storing",
    description="Validate plot coordinates against EUDR requirements without persisting."
)
async def validate_coordinates(
    request: Request,
    body: ValidateCoordinatesRequest,
    current_user: User = Depends(get_current_user),
    agent: GeolocationCollectorAgent = Depends(get_agent)
):
    """
    Validate coordinates without storing.

    Performs deterministic validation:
    - Precision check (>= 6 decimal places)
    - Range check (-90 <= lat <= 90, -180 <= lon <= 180)
    - Country boundary check
    - Water body exclusion
    - Polygon geometry validation (if applicable)
    """
    check_validation_rate_limit(current_user)

    coordinates = convert_coordinates_to_model(body.coordinates)

    input_data = GeolocationInput(
        operation=OperationType.VALIDATE_COORDINATES,
        coordinates=coordinates,
        country_code=body.country_code,
        commodity=body.commodity,
        declared_area_hectares=body.declared_area_hectares,
        collection_accuracy_m=body.collection_accuracy_m
    )

    result = agent.run(input_data)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.errors[0] if result.errors else "Validation failed"
        )

    return validation_result_to_response(result.validation_result)


# =============================================================================
# PLOT CRUD ENDPOINTS
# =============================================================================

@router.post(
    "/plots",
    response_model=PlotResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a new plot",
    description="Submit and validate a new production plot."
)
async def submit_plot(
    request: Request,
    body: PlotSubmissionRequest,
    current_user: User = Depends(get_current_user),
    agent: GeolocationCollectorAgent = Depends(get_agent)
):
    """
    Submit a new production plot.

    The plot will be validated and stored with its validation status.
    Invalid plots are stored with INVALID status for remediation.
    Plots with warnings are stored with NEEDS_REVIEW status.
    """
    check_validation_rate_limit(current_user)

    coordinates = convert_coordinates_to_model(body.coordinates)

    input_data = GeolocationInput(
        operation=OperationType.SUBMIT_PLOT,
        coordinates=coordinates,
        country_code=body.country_code,
        commodity=body.commodity,
        declared_area_hectares=body.declared_area_hectares,
        supplier_id=body.supplier_id,
        collection_method=body.collection_method,
        collection_accuracy_m=body.collection_accuracy_m
    )

    result = agent.run(input_data)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.errors[0] if result.errors else "Failed to submit plot"
        )

    return plot_to_response(result.plot)


@router.get(
    "/plots",
    response_model=PlotListResponse,
    summary="List plots",
    description="List plots with optional filtering."
)
async def list_plots(
    request: Request,
    supplier_id: Optional[UUID] = Query(None, description="Filter by supplier"),
    commodity: Optional[CommodityType] = Query(None, description="Filter by commodity"),
    validation_status: Optional[ValidationStatus] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: User = Depends(get_current_user),
    agent: GeolocationCollectorAgent = Depends(get_agent)
):
    """List plots with optional filtering and pagination."""
    input_data = GeolocationInput(
        operation=OperationType.LIST_PLOTS,
        supplier_id=supplier_id,
        commodity=commodity,
        validation_status=validation_status,
        limit=limit,
        offset=offset
    )

    result = agent.run(input_data)

    return PlotListResponse(
        plots=[plot_to_response(p) for p in result.plots],
        total_count=result.total_count,
        limit=limit,
        offset=offset
    )


@router.get(
    "/plots/{plot_id}",
    response_model=PlotResponse,
    summary="Get plot by ID",
    description="Retrieve a specific plot by its ID."
)
async def get_plot(
    request: Request,
    plot_id: UUID,
    current_user: User = Depends(get_current_user),
    agent: GeolocationCollectorAgent = Depends(get_agent)
):
    """Get a specific plot by ID."""
    input_data = GeolocationInput(
        operation=OperationType.GET_PLOT,
        plot_id=plot_id
    )

    result = agent.run(input_data)

    if not result.success or not result.plot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot {plot_id} not found"
        )

    return plot_to_response(result.plot)


@router.post(
    "/plots/{plot_id}/validate",
    response_model=ValidationResultResponse,
    summary="Re-validate plot",
    description="Re-run validation on an existing plot."
)
async def revalidate_plot(
    request: Request,
    plot_id: UUID,
    current_user: User = Depends(get_current_user),
    agent: GeolocationCollectorAgent = Depends(get_agent)
):
    """Re-validate an existing plot."""
    check_validation_rate_limit(current_user)

    input_data = GeolocationInput(
        operation=OperationType.REVALIDATE_PLOT,
        plot_id=plot_id
    )

    result = agent.run(input_data)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.errors[0] if result.errors else f"Plot {plot_id} not found"
        )

    return validation_result_to_response(result.validation_result)


@router.post(
    "/plots/{plot_id}/enrich",
    response_model=PlotResponse,
    summary="Enrich plot data",
    description="Enrich plot with geocoding and administrative data."
)
async def enrich_plot(
    request: Request,
    plot_id: UUID,
    current_user: User = Depends(get_current_user),
    agent: GeolocationCollectorAgent = Depends(get_agent)
):
    """Enrich plot with geocoding and additional data."""
    input_data = GeolocationInput(
        operation=OperationType.ENRICH_PLOT,
        plot_id=plot_id
    )

    result = agent.run(input_data)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.errors[0] if result.errors else f"Plot {plot_id} not found"
        )

    return plot_to_response(result.plot)


@router.get(
    "/plots/{plot_id}/history",
    response_model=List[Dict[str, Any]],
    summary="Get validation history",
    description="Get validation history for a plot."
)
async def get_plot_validation_history(
    request: Request,
    plot_id: UUID,
    current_user: User = Depends(get_current_user),
    agent: GeolocationCollectorAgent = Depends(get_agent)
):
    """Get validation history for a plot."""
    history = agent.get_validation_history(plot_id)

    return [
        {
            "validation_id": str(h.validation_id),
            "plot_id": str(h.plot_id),
            "validation_date": h.validation_date.isoformat(),
            "status": h.status.value if hasattr(h.status, 'value') else h.status,
            "error_count": len(h.errors),
            "warning_count": len(h.warnings),
            "validated_by": h.validated_by,
            "validation_method": h.validation_method
        }
        for h in history
    ]


# =============================================================================
# BULK UPLOAD ENDPOINTS
# =============================================================================

@router.post(
    "/plots/bulk",
    response_model=BulkUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Bulk upload plots",
    description="Upload multiple plots from a file (async processing)."
)
async def bulk_upload_plots(
    request: Request,
    file: UploadFile = File(..., description="File containing plot data"),
    format: BulkUploadFormat = Form(..., description="File format"),
    supplier_id: UUID = Form(..., description="Supplier ID for all plots"),
    current_user: User = Depends(get_current_user),
    agent: GeolocationCollectorAgent = Depends(get_agent)
):
    """
    Upload multiple plots from a file.

    Supported formats:
    - CSV: Columns for latitude, longitude, country_code, commodity, etc.
    - GeoJSON: FeatureCollection with Point or Polygon features
    - KML: Standard KML format with placemarks
    - Shapefile: ZIP containing .shp, .shx, .dbf files

    Returns a job ID for tracking progress.
    """
    check_bulk_rate_limit(current_user)

    # Save file temporarily
    file_path = f"/tmp/bulk_upload_{uuid.uuid4()}_{file.filename}"

    input_data = GeolocationInput(
        operation=OperationType.BULK_UPLOAD,
        file_path=file_path,
        file_format=format,
        supplier_id=supplier_id
    )

    result = agent.run(input_data)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.errors[0] if result.errors else "Failed to initiate bulk upload"
        )

    return BulkUploadResponse(
        job_id=result.bulk_job.job_id,
        status=result.bulk_job.status.value,
        message="Upload accepted for processing. Poll /bulk/{job_id} for status."
    )


@router.get(
    "/bulk/{job_id}",
    response_model=BulkJobStatusResponse,
    summary="Get bulk upload status",
    description="Get the status of a bulk upload job."
)
async def get_bulk_job_status(
    request: Request,
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    agent: GeolocationCollectorAgent = Depends(get_agent)
):
    """Get bulk upload job status."""
    job = agent.get_bulk_job_status(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bulk job {job_id} not found"
        )

    progress = 0.0
    if job.total_count > 0:
        progress = (job.processed_count / job.total_count) * 100

    return BulkJobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        total_count=job.total_count,
        processed_count=job.processed_count,
        valid_count=job.valid_count,
        invalid_count=job.invalid_count,
        warning_count=job.warning_count,
        progress_percent=round(progress, 2),
        report_url=job.report_url,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


# =============================================================================
# STATISTICS ENDPOINTS
# =============================================================================

@router.get(
    "/stats",
    response_model=Dict[str, Any],
    summary="Get validation statistics",
    description="Get summary statistics for plot validation."
)
async def get_statistics(
    request: Request,
    supplier_id: Optional[UUID] = Query(None, description="Filter by supplier"),
    commodity: Optional[CommodityType] = Query(None, description="Filter by commodity"),
    current_user: User = Depends(get_current_user),
    agent: GeolocationCollectorAgent = Depends(get_agent)
):
    """Get validation statistics."""
    plots = agent.get_all_plots()

    # Apply filters
    if supplier_id:
        plots = [p for p in plots if p.supplier_id == supplier_id]
    if commodity:
        plots = [p for p in plots if p.commodity == commodity]

    total = len(plots)
    valid = sum(1 for p in plots if p.validation_status == ValidationStatus.VALID)
    invalid = sum(1 for p in plots if p.validation_status == ValidationStatus.INVALID)
    needs_review = sum(1 for p in plots if p.validation_status == ValidationStatus.NEEDS_REVIEW)
    pending = sum(1 for p in plots if p.validation_status == ValidationStatus.PENDING)

    # Count by commodity
    by_commodity = {}
    for p in plots:
        comm = p.commodity.value if hasattr(p.commodity, 'value') else p.commodity
        by_commodity[comm] = by_commodity.get(comm, 0) + 1

    # Count by geometry type
    points = sum(1 for p in plots if p.geometry_type == GeometryType.POINT)
    polygons = sum(1 for p in plots if p.geometry_type == GeometryType.POLYGON)

    return {
        "total_plots": total,
        "validation_status": {
            "valid": valid,
            "invalid": invalid,
            "needs_review": needs_review,
            "pending": pending
        },
        "validation_rate": round(valid / total * 100, 2) if total > 0 else 0,
        "by_commodity": by_commodity,
        "by_geometry": {
            "point": points,
            "polygon": polygons
        }
    }


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="Health check",
    description="Check API health status."
)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent": "GL-EUDR-002",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }
