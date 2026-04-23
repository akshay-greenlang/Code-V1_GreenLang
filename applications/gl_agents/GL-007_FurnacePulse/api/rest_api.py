"""
GL-007 FurnacePulse - REST API

FastAPI-based REST endpoints for industrial furnace monitoring,
predictive maintenance, and NFPA 86 compliance management.

Endpoints:
- GET /health - Health check
- GET /status - Agent status with KPIs
- GET /furnaces/{furnace_id}/kpis - Current efficiency KPIs
- GET /furnaces/{furnace_id}/hotspots - Active hotspot alerts
- GET /furnaces/{furnace_id}/tmt - Current TMT readings
- GET /furnaces/{furnace_id}/compliance - NFPA 86 compliance status
- GET /furnaces/{furnace_id}/rul - RUL predictions for components
- POST /furnaces/{furnace_id}/evidence - Generate evidence package
- GET /alerts - List active alerts with filtering
- POST /alerts/{alert_id}/acknowledge - Acknowledge alert
- GET /explain/{prediction_id} - Get SHAP/LIME explanation
- GET /metrics - Prometheus metrics endpoint

Author: GreenLang API Team
Version: 1.0.0
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import logging
import uuid
import hashlib
import io

from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    Depends,
    Query,
    Path,
    Header,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .middleware import (
    RBACMiddleware,
    AuditLoggingMiddleware,
    RequestIDMiddleware,
    RateLimitMiddleware,
    ErrorHandlingMiddleware,
    ProvenanceMiddleware,
    UserRole,
    get_current_user,
    require_roles,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class ComponentType(str, Enum):
    """Furnace component types for RUL predictions."""
    RADIANT_TUBE = "radiant_tube"
    BURNER = "burner"
    REFRACTORY = "refractory"
    THERMOCOUPLE = "thermocouple"
    DAMPER = "damper"
    FAN = "fan"
    HEAT_EXCHANGER = "heat_exchanger"
    CONTROL_VALVE = "control_valve"


class ComplianceStatus(str, Enum):
    """NFPA 86 compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    WAIVER_GRANTED = "waiver_granted"


class ExplanationType(str, Enum):
    """XAI explanation method types."""
    SHAP = "shap"
    LIME = "lime"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    COUNTERFACTUAL = "counterfactual"


# =============================================================================
# Request/Response Models
# =============================================================================

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Component health statuses")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2025-01-15T10:30:00Z",
                "components": {
                    "database": "ok",
                    "redis": "ok",
                    "ml_engine": "ok",
                    "telemetry_pipeline": "ok"
                }
            }
        }


class AgentStatusResponse(BaseModel):
    """Response model for agent status endpoint."""
    agent_id: str = Field(..., description="Agent identifier")
    agent_name: str = Field(..., description="Agent display name")
    status: str = Field(..., description="Agent operational status")
    uptime_seconds: float = Field(..., description="Agent uptime in seconds")
    active_furnaces: int = Field(..., description="Number of monitored furnaces")
    active_alerts: int = Field(..., description="Number of active alerts")
    last_prediction_at: Optional[datetime] = Field(None, description="Last prediction timestamp")
    kpis: Dict[str, float] = Field(..., description="Key performance indicators")


class TMTReading(BaseModel):
    """Tube Metal Temperature reading from a sensor."""
    sensor_id: str = Field(..., description="Sensor identifier")
    position_x: float = Field(..., description="X position in furnace (meters)")
    position_y: float = Field(..., description="Y position in furnace (meters)")
    position_z: float = Field(..., description="Z position in furnace (meters)")
    temperature_c: float = Field(..., description="Temperature reading in Celsius")
    timestamp: datetime = Field(..., description="Reading timestamp")
    quality_score: float = Field(..., ge=0, le=1, description="Data quality score (0-1)")
    is_valid: bool = Field(..., description="Whether reading passed validation")


class TMTReadingsResponse(BaseModel):
    """Response model for TMT readings endpoint."""
    furnace_id: str = Field(..., description="Furnace identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    readings: List[TMTReading] = Field(..., description="List of TMT readings")
    avg_temperature_c: float = Field(..., description="Average tube temperature")
    max_temperature_c: float = Field(..., description="Maximum tube temperature")
    min_temperature_c: float = Field(..., description="Minimum tube temperature")
    gradient_max_c_m: float = Field(..., description="Maximum temperature gradient (C/m)")
    sensor_count: int = Field(..., description="Total sensor count")
    valid_sensor_count: int = Field(..., description="Valid sensor count")


class HotspotAlert(BaseModel):
    """Hotspot alert model."""
    alert_id: str = Field(..., description="Alert identifier")
    furnace_id: str = Field(..., description="Furnace identifier")
    cluster_id: str = Field(..., description="Hotspot cluster identifier")
    severity: AlertSeverity = Field(..., description="Alert severity")
    center_x: float = Field(..., description="Hotspot center X position (meters)")
    center_y: float = Field(..., description="Hotspot center Y position (meters)")
    center_z: float = Field(..., description="Hotspot center Z position (meters)")
    radius_m: float = Field(..., description="Hotspot radius (meters)")
    peak_temperature_c: float = Field(..., description="Peak temperature in hotspot")
    avg_temperature_c: float = Field(..., description="Average temperature in hotspot")
    temperature_delta_c: float = Field(..., description="Delta from normal temperature")
    affected_sensors: List[str] = Field(..., description="List of affected sensor IDs")
    detected_at: datetime = Field(..., description="Detection timestamp")
    status: AlertStatus = Field(..., description="Alert status")
    recommended_action: str = Field(..., description="Recommended action")


class HotspotsResponse(BaseModel):
    """Response model for hotspots endpoint."""
    furnace_id: str = Field(..., description="Furnace identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    hotspots: List[HotspotAlert] = Field(..., description="Active hotspot alerts")
    total_hotspots: int = Field(..., description="Total hotspot count")
    critical_count: int = Field(..., description="Critical hotspot count")
    high_count: int = Field(..., description="High severity hotspot count")


class EfficiencyKPI(BaseModel):
    """Efficiency KPI model."""
    kpi_id: str = Field(..., description="KPI identifier")
    name: str = Field(..., description="KPI name")
    value: float = Field(..., description="KPI value")
    unit: str = Field(..., description="Unit of measurement")
    target: Optional[float] = Field(None, description="Target value")
    threshold_low: Optional[float] = Field(None, description="Low threshold")
    threshold_high: Optional[float] = Field(None, description="High threshold")
    status: str = Field(..., description="KPI status (normal/warning/critical)")
    trend: str = Field(..., description="Trend direction (up/down/stable)")
    timestamp: datetime = Field(..., description="Measurement timestamp")


class EfficiencyKPIsResponse(BaseModel):
    """Response model for efficiency KPIs endpoint."""
    furnace_id: str = Field(..., description="Furnace identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    kpis: List[EfficiencyKPI] = Field(..., description="List of efficiency KPIs")
    overall_efficiency_pct: float = Field(..., description="Overall thermal efficiency (%)")
    fuel_consumption_kg_h: float = Field(..., description="Fuel consumption (kg/h)")
    excess_air_pct: float = Field(..., description="Excess air percentage")
    stack_loss_pct: float = Field(..., description="Stack heat loss percentage")
    co2_emissions_kg_h: float = Field(..., description="CO2 emissions (kg/h)")


class NFPA86Requirement(BaseModel):
    """NFPA 86 requirement model."""
    requirement_id: str = Field(..., description="Requirement identifier (e.g., 8.4.1)")
    section: str = Field(..., description="NFPA 86 section")
    description: str = Field(..., description="Requirement description")
    status: ComplianceStatus = Field(..., description="Compliance status")
    last_verified_at: Optional[datetime] = Field(None, description="Last verification timestamp")
    evidence_ids: List[str] = Field(default_factory=list, description="Associated evidence IDs")
    notes: Optional[str] = Field(None, description="Compliance notes")


class ComplianceStatusResponse(BaseModel):
    """Response model for compliance status endpoint."""
    furnace_id: str = Field(..., description="Furnace identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    overall_status: ComplianceStatus = Field(..., description="Overall compliance status")
    compliance_score_pct: float = Field(..., description="Compliance score (0-100)")
    total_requirements: int = Field(..., description="Total NFPA 86 requirements")
    compliant_count: int = Field(..., description="Compliant requirement count")
    non_compliant_count: int = Field(..., description="Non-compliant requirement count")
    pending_count: int = Field(..., description="Pending review count")
    requirements: List[NFPA86Requirement] = Field(..., description="Requirement statuses")
    next_audit_due: Optional[datetime] = Field(None, description="Next audit due date")


class RULPrediction(BaseModel):
    """Remaining Useful Life prediction model."""
    prediction_id: str = Field(..., description="Prediction identifier")
    component_id: str = Field(..., description="Component identifier")
    component_type: ComponentType = Field(..., description="Component type")
    component_name: str = Field(..., description="Component display name")
    rul_days: float = Field(..., description="Predicted RUL in days")
    rul_hours: float = Field(..., description="Predicted RUL in operating hours")
    confidence_lower_days: float = Field(..., description="Lower confidence bound (days)")
    confidence_upper_days: float = Field(..., description="Upper confidence bound (days)")
    confidence_level: float = Field(..., description="Confidence level (e.g., 0.95)")
    failure_probability_30d: float = Field(..., description="Failure probability in 30 days")
    failure_mode: str = Field(..., description="Predicted failure mode")
    health_score: float = Field(..., ge=0, le=100, description="Component health score (0-100)")
    degradation_rate: float = Field(..., description="Degradation rate (units/day)")
    last_maintenance: Optional[datetime] = Field(None, description="Last maintenance date")
    recommended_action: str = Field(..., description="Recommended maintenance action")
    predicted_at: datetime = Field(..., description="Prediction timestamp")


class RULPredictionsResponse(BaseModel):
    """Response model for RUL predictions endpoint."""
    furnace_id: str = Field(..., description="Furnace identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    predictions: List[RULPrediction] = Field(..., description="Component RUL predictions")
    components_at_risk: int = Field(..., description="Components with RUL < 30 days")
    avg_health_score: float = Field(..., description="Average component health score")
    next_maintenance_due: Optional[datetime] = Field(None, description="Next recommended maintenance")
    maintenance_window_start: Optional[datetime] = Field(None, description="Optimal maintenance window start")
    maintenance_window_end: Optional[datetime] = Field(None, description="Optimal maintenance window end")


class EvidenceGenerateRequest(BaseModel):
    """Request model for evidence package generation."""
    package_type: str = Field(
        "compliance",
        description="Package type: compliance, incident, maintenance"
    )
    start_date: datetime = Field(..., description="Evidence start date")
    end_date: datetime = Field(..., description="Evidence end date")
    include_telemetry: bool = Field(True, description="Include telemetry data")
    include_alerts: bool = Field(True, description="Include alerts")
    include_predictions: bool = Field(True, description="Include ML predictions")
    include_maintenance: bool = Field(True, description="Include maintenance records")
    requirement_ids: Optional[List[str]] = Field(None, description="Specific NFPA 86 requirements")
    format: str = Field("json", description="Output format: json, pdf, zip")


class EvidencePackageResponse(BaseModel):
    """Response model for evidence package generation."""
    package_id: str = Field(..., description="Evidence package identifier")
    furnace_id: str = Field(..., description="Furnace identifier")
    package_type: str = Field(..., description="Package type")
    status: str = Field(..., description="Generation status")
    created_at: datetime = Field(..., description="Creation timestamp")
    download_url: Optional[str] = Field(None, description="Download URL when ready")
    sha256_hash: Optional[str] = Field(None, description="Package SHA-256 hash")
    size_bytes: Optional[int] = Field(None, description="Package size in bytes")
    expires_at: Optional[datetime] = Field(None, description="Download URL expiration")


class Alert(BaseModel):
    """General alert model."""
    alert_id: str = Field(..., description="Alert identifier")
    furnace_id: str = Field(..., description="Furnace identifier")
    alert_type: str = Field(..., description="Alert type code")
    severity: AlertSeverity = Field(..., description="Alert severity")
    status: AlertStatus = Field(..., description="Alert status")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    source: str = Field(..., description="Alert source (system/component)")
    created_at: datetime = Field(..., description="Creation timestamp")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment timestamp")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    resolved_by: Optional[str] = Field(None, description="User who resolved")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AlertsResponse(BaseModel):
    """Response model for alerts listing endpoint."""
    alerts: List[Alert] = Field(..., description="List of alerts")
    total: int = Field(..., description="Total alert count")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")
    has_more: bool = Field(..., description="Whether more pages exist")


class AlertAcknowledgeRequest(BaseModel):
    """Request model for alert acknowledgment."""
    notes: Optional[str] = Field(None, description="Acknowledgment notes")
    assign_to: Optional[str] = Field(None, description="Assign to user ID")


class AlertAcknowledgeResponse(BaseModel):
    """Response model for alert acknowledgment."""
    alert_id: str = Field(..., description="Alert identifier")
    status: AlertStatus = Field(..., description="New alert status")
    acknowledged_at: datetime = Field(..., description="Acknowledgment timestamp")
    acknowledged_by: str = Field(..., description="User who acknowledged")


class FeatureImportance(BaseModel):
    """Feature importance for explainability."""
    feature_name: str = Field(..., description="Feature name")
    importance_value: float = Field(..., description="Importance value")
    direction: str = Field(..., description="Impact direction (positive/negative)")
    contribution_pct: float = Field(..., description="Contribution percentage")


class ExplanationResponse(BaseModel):
    """Response model for prediction explanation endpoint."""
    prediction_id: str = Field(..., description="Prediction identifier")
    prediction_type: str = Field(..., description="Prediction type")
    prediction_value: float = Field(..., description="Predicted value")
    explanation_method: ExplanationType = Field(..., description="Explanation method used")
    feature_importance: List[FeatureImportance] = Field(..., description="Feature importances")
    summary: str = Field(..., description="Human-readable summary")
    key_drivers: List[str] = Field(..., description="Key prediction drivers")
    confidence_score: float = Field(..., description="Explanation confidence")
    counterfactuals: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Counterfactual explanations"
    )
    generated_at: datetime = Field(..., description="Generation timestamp")
    computation_hash: str = Field(..., description="Computation provenance hash")


class MetricsResponse(BaseModel):
    """Response model for Prometheus metrics."""
    content_type: str = Field("text/plain", description="Content type")


class User(BaseModel):
    """User model for authentication."""
    user_id: str = Field(..., description="User identifier")
    email: str = Field(..., description="User email")
    roles: List[UserRole] = Field(..., description="User roles")
    tenant_id: str = Field(..., description="Tenant identifier")


# =============================================================================
# In-memory stores (replace with database in production)
# =============================================================================

job_store: Dict[str, Dict[str, Any]] = {}
alert_store: Dict[str, Alert] = {}
prediction_store: Dict[str, Dict[str, Any]] = {}

# Agent start time for uptime calculation
AGENT_START_TIME = datetime.now(timezone.utc)


# =============================================================================
# Dependencies
# =============================================================================

def get_furnace_service():
    """Get furnace service instance."""
    # In production, return actual service instance
    return None


def get_alert_service():
    """Get alert service instance."""
    return None


def get_compliance_service():
    """Get compliance service instance."""
    return None


def get_prediction_service():
    """Get prediction service instance."""
    return None


# =============================================================================
# Lifespan Handler
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger.info("GL-007 FurnacePulse API starting up")

    # Initialize services, connections, ML models
    # await init_database()
    # await init_redis()
    # await load_ml_models()

    yield

    # Cleanup
    logger.info("GL-007 FurnacePulse API shutting down")
    # await close_database()
    # await close_redis()


# =============================================================================
# Create FastAPI Application
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="GL-007 FurnacePulse API",
        description=(
            "Industrial Furnace Monitoring API - Real-time TMT monitoring, "
            "predictive maintenance, efficiency optimization, and NFPA 86 compliance."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/api/v1/openapi.json",
        openapi_tags=[
            {"name": "Health", "description": "Service health and status"},
            {"name": "Furnaces", "description": "Furnace monitoring endpoints"},
            {"name": "Alerts", "description": "Alert management"},
            {"name": "Explainability", "description": "XAI explanations for predictions"},
            {"name": "Metrics", "description": "Prometheus metrics"},
        ],
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://*.greenlang.io", "http://localhost:*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )

    # Add custom middleware (order matters - last added is first executed)
    app.add_middleware(ProvenanceMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=120, burst_size=20)
    app.add_middleware(AuditLoggingMiddleware, log_request_body=False)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RBACMiddleware, require_auth=False)  # Set True for production

    # Include API router
    app.include_router(router, prefix="/api/v1")

    return app


# =============================================================================
# API Router
# =============================================================================

from fastapi import APIRouter

router = APIRouter()


# -----------------------------------------------------------------------------
# Health Endpoints
# -----------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check service health status and component availability.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for load balancers and monitoring.

    Returns:
        Service health status with component details.
    """
    # Check component health (implement actual checks in production)
    components = {
        "database": "ok",
        "redis": "ok",
        "ml_engine": "ok",
        "telemetry_pipeline": "ok",
        "nfpa_compliance": "ok",
    }

    # Determine overall status
    all_healthy = all(v == "ok" for v in components.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc),
        components=components,
    )


@router.get(
    "/status",
    response_model=AgentStatusResponse,
    tags=["Health"],
    summary="Agent status",
    description="Get detailed agent status with KPIs.",
)
async def get_agent_status(
    current_user: Optional[User] = Depends(get_current_user),
) -> AgentStatusResponse:
    """
    Get detailed agent operational status and KPIs.

    Returns:
        Agent status including uptime, monitored furnaces, and KPIs.
    """
    now = datetime.now(timezone.utc)
    uptime = (now - AGENT_START_TIME).total_seconds()

    return AgentStatusResponse(
        agent_id="GL-007",
        agent_name="FurnacePulse",
        status="operational",
        uptime_seconds=uptime,
        active_furnaces=12,
        active_alerts=len([a for a in alert_store.values() if a.status == AlertStatus.ACTIVE]),
        last_prediction_at=now - timedelta(minutes=5),
        kpis={
            "avg_efficiency_pct": 87.5,
            "avg_availability_pct": 98.2,
            "mtbf_hours": 2160.0,
            "predictions_today": 1440,
            "alerts_resolved_today": 23,
            "compliance_score_pct": 96.5,
        },
    )


# -----------------------------------------------------------------------------
# Furnace Endpoints
# -----------------------------------------------------------------------------

@router.get(
    "/furnaces/{furnace_id}/kpis",
    response_model=EfficiencyKPIsResponse,
    tags=["Furnaces"],
    summary="Get efficiency KPIs",
    description="Get current efficiency KPIs for a specific furnace.",
)
async def get_furnace_kpis(
    furnace_id: str = Path(..., description="Furnace identifier", example="FRN-001"),
    current_user: Optional[User] = Depends(get_current_user),
) -> EfficiencyKPIsResponse:
    """
    Get current efficiency KPIs for a furnace.

    Args:
        furnace_id: Furnace identifier.
        current_user: Authenticated user.

    Returns:
        Efficiency KPIs including thermal efficiency, fuel consumption, and emissions.

    Raises:
        HTTPException: 404 if furnace not found.
    """
    now = datetime.now(timezone.utc)

    # Mock KPI data (replace with actual service call)
    kpis = [
        EfficiencyKPI(
            kpi_id="kpi-001",
            name="Thermal Efficiency",
            value=87.5,
            unit="%",
            target=90.0,
            threshold_low=80.0,
            threshold_high=95.0,
            status="normal",
            trend="stable",
            timestamp=now,
        ),
        EfficiencyKPI(
            kpi_id="kpi-002",
            name="Fuel Consumption Rate",
            value=125.3,
            unit="kg/h",
            target=120.0,
            threshold_low=100.0,
            threshold_high=150.0,
            status="warning",
            trend="up",
            timestamp=now,
        ),
        EfficiencyKPI(
            kpi_id="kpi-003",
            name="Excess Air Ratio",
            value=12.5,
            unit="%",
            target=10.0,
            threshold_low=5.0,
            threshold_high=20.0,
            status="normal",
            trend="down",
            timestamp=now,
        ),
        EfficiencyKPI(
            kpi_id="kpi-004",
            name="Stack Temperature",
            value=285.0,
            unit="C",
            target=275.0,
            threshold_low=200.0,
            threshold_high=350.0,
            status="normal",
            trend="stable",
            timestamp=now,
        ),
    ]

    return EfficiencyKPIsResponse(
        furnace_id=furnace_id,
        timestamp=now,
        kpis=kpis,
        overall_efficiency_pct=87.5,
        fuel_consumption_kg_h=125.3,
        excess_air_pct=12.5,
        stack_loss_pct=8.2,
        co2_emissions_kg_h=385.7,
    )


@router.get(
    "/furnaces/{furnace_id}/hotspots",
    response_model=HotspotsResponse,
    tags=["Furnaces"],
    summary="Get active hotspots",
    description="Get active hotspot alerts detected in the furnace.",
)
async def get_furnace_hotspots(
    furnace_id: str = Path(..., description="Furnace identifier", example="FRN-001"),
    severity: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
    current_user: Optional[User] = Depends(get_current_user),
) -> HotspotsResponse:
    """
    Get active hotspot alerts for a furnace.

    Args:
        furnace_id: Furnace identifier.
        severity: Optional severity filter.
        current_user: Authenticated user.

    Returns:
        Active hotspot alerts with location and severity details.
    """
    now = datetime.now(timezone.utc)

    # Mock hotspot data
    hotspots = [
        HotspotAlert(
            alert_id="hs-001",
            furnace_id=furnace_id,
            cluster_id="cluster-001",
            severity=AlertSeverity.HIGH,
            center_x=2.5,
            center_y=1.2,
            center_z=0.8,
            radius_m=0.35,
            peak_temperature_c=925.0,
            avg_temperature_c=895.0,
            temperature_delta_c=45.0,
            affected_sensors=["TMT-023", "TMT-024", "TMT-025"],
            detected_at=now - timedelta(minutes=15),
            status=AlertStatus.ACTIVE,
            recommended_action="Inspect radiant tube section B2 for refractory degradation",
        ),
        HotspotAlert(
            alert_id="hs-002",
            furnace_id=furnace_id,
            cluster_id="cluster-002",
            severity=AlertSeverity.MEDIUM,
            center_x=4.1,
            center_y=0.8,
            center_z=1.5,
            radius_m=0.25,
            peak_temperature_c=875.0,
            avg_temperature_c=855.0,
            temperature_delta_c=25.0,
            affected_sensors=["TMT-045", "TMT-046"],
            detected_at=now - timedelta(hours=2),
            status=AlertStatus.ACKNOWLEDGED,
            recommended_action="Monitor zone C1 - scheduled inspection in 48 hours",
        ),
    ]

    # Apply severity filter
    if severity:
        hotspots = [h for h in hotspots if h.severity == severity]

    return HotspotsResponse(
        furnace_id=furnace_id,
        timestamp=now,
        hotspots=hotspots,
        total_hotspots=len(hotspots),
        critical_count=sum(1 for h in hotspots if h.severity == AlertSeverity.CRITICAL),
        high_count=sum(1 for h in hotspots if h.severity == AlertSeverity.HIGH),
    )


@router.get(
    "/furnaces/{furnace_id}/tmt",
    response_model=TMTReadingsResponse,
    tags=["Furnaces"],
    summary="Get TMT readings",
    description="Get current Tube Metal Temperature readings from all sensors.",
)
async def get_furnace_tmt(
    furnace_id: str = Path(..., description="Furnace identifier", example="FRN-001"),
    zone: Optional[str] = Query(None, description="Filter by zone ID"),
    valid_only: bool = Query(True, description="Return only valid readings"),
    current_user: Optional[User] = Depends(get_current_user),
) -> TMTReadingsResponse:
    """
    Get current TMT readings for a furnace.

    Args:
        furnace_id: Furnace identifier.
        zone: Optional zone filter.
        valid_only: Whether to return only valid readings.
        current_user: Authenticated user.

    Returns:
        TMT readings with statistics and quality metrics.
    """
    now = datetime.now(timezone.utc)

    # Mock TMT readings
    import random
    readings = []
    for i in range(48):  # 48 sensors
        temp = 850.0 + random.uniform(-30, 50)
        is_valid = random.random() > 0.05  # 95% valid

        readings.append(TMTReading(
            sensor_id=f"TMT-{i+1:03d}",
            position_x=(i % 8) * 0.5,
            position_y=(i // 8) * 0.5,
            position_z=0.0 if i < 24 else 1.5,
            temperature_c=temp,
            timestamp=now,
            quality_score=0.98 if is_valid else 0.45,
            is_valid=is_valid,
        ))

    if valid_only:
        readings = [r for r in readings if r.is_valid]

    temps = [r.temperature_c for r in readings]

    return TMTReadingsResponse(
        furnace_id=furnace_id,
        timestamp=now,
        readings=readings,
        avg_temperature_c=sum(temps) / len(temps) if temps else 0,
        max_temperature_c=max(temps) if temps else 0,
        min_temperature_c=min(temps) if temps else 0,
        gradient_max_c_m=25.5,  # Mock value
        sensor_count=48,
        valid_sensor_count=len(readings),
    )


@router.get(
    "/furnaces/{furnace_id}/compliance",
    response_model=ComplianceStatusResponse,
    tags=["Furnaces"],
    summary="Get NFPA 86 compliance",
    description="Get NFPA 86 compliance status and requirement details.",
    dependencies=[Depends(require_roles([UserRole.SAFETY, UserRole.ADMIN]))],
)
async def get_furnace_compliance(
    furnace_id: str = Path(..., description="Furnace identifier", example="FRN-001"),
    section: Optional[str] = Query(None, description="Filter by NFPA 86 section"),
    current_user: Optional[User] = Depends(get_current_user),
) -> ComplianceStatusResponse:
    """
    Get NFPA 86 compliance status for a furnace.

    Args:
        furnace_id: Furnace identifier.
        section: Optional NFPA 86 section filter.
        current_user: Authenticated user (requires safety or admin role).

    Returns:
        Compliance status with requirement details.
    """
    now = datetime.now(timezone.utc)

    # Mock NFPA 86 requirements
    requirements = [
        NFPA86Requirement(
            requirement_id="8.4.1",
            section="8.4 Safety Controls",
            description="Temperature limiting devices shall be provided",
            status=ComplianceStatus.COMPLIANT,
            last_verified_at=now - timedelta(days=15),
            evidence_ids=["EVD-001", "EVD-002"],
            notes="Verified during Q4 2024 audit",
        ),
        NFPA86Requirement(
            requirement_id="8.5.2",
            section="8.5 Combustion Safeguards",
            description="Flame detection system required for Class A furnaces",
            status=ComplianceStatus.COMPLIANT,
            last_verified_at=now - timedelta(days=15),
            evidence_ids=["EVD-003"],
        ),
        NFPA86Requirement(
            requirement_id="8.6.1",
            section="8.6 Purge Requirements",
            description="Minimum 4 volume changes purge before ignition",
            status=ComplianceStatus.COMPLIANT,
            last_verified_at=now - timedelta(days=30),
            evidence_ids=["EVD-004", "EVD-005"],
        ),
        NFPA86Requirement(
            requirement_id="9.2.1",
            section="9.2 Ventilation",
            description="Adequate ventilation for combustion air supply",
            status=ComplianceStatus.PENDING_REVIEW,
            last_verified_at=now - timedelta(days=90),
            notes="Scheduled for review in next audit cycle",
        ),
    ]

    if section:
        requirements = [r for r in requirements if section in r.section]

    compliant = sum(1 for r in requirements if r.status == ComplianceStatus.COMPLIANT)
    non_compliant = sum(1 for r in requirements if r.status == ComplianceStatus.NON_COMPLIANT)
    pending = sum(1 for r in requirements if r.status == ComplianceStatus.PENDING_REVIEW)

    return ComplianceStatusResponse(
        furnace_id=furnace_id,
        timestamp=now,
        overall_status=ComplianceStatus.COMPLIANT if non_compliant == 0 else ComplianceStatus.NON_COMPLIANT,
        compliance_score_pct=100.0 * compliant / len(requirements) if requirements else 0,
        total_requirements=len(requirements),
        compliant_count=compliant,
        non_compliant_count=non_compliant,
        pending_count=pending,
        requirements=requirements,
        next_audit_due=now + timedelta(days=90),
    )


@router.get(
    "/furnaces/{furnace_id}/rul",
    response_model=RULPredictionsResponse,
    tags=["Furnaces"],
    summary="Get RUL predictions",
    description="Get Remaining Useful Life predictions for furnace components.",
)
async def get_furnace_rul(
    furnace_id: str = Path(..., description="Furnace identifier", example="FRN-001"),
    component_type: Optional[ComponentType] = Query(None, description="Filter by component type"),
    at_risk_only: bool = Query(False, description="Return only at-risk components (RUL < 30 days)"),
    current_user: Optional[User] = Depends(get_current_user),
) -> RULPredictionsResponse:
    """
    Get RUL predictions for furnace components.

    Args:
        furnace_id: Furnace identifier.
        component_type: Optional component type filter.
        at_risk_only: Whether to return only at-risk components.
        current_user: Authenticated user.

    Returns:
        RUL predictions with confidence intervals and maintenance recommendations.
    """
    now = datetime.now(timezone.utc)

    # Mock RUL predictions
    predictions = [
        RULPrediction(
            prediction_id="rul-001",
            component_id="RT-001",
            component_type=ComponentType.RADIANT_TUBE,
            component_name="Radiant Tube Zone A1",
            rul_days=145.5,
            rul_hours=3492.0,
            confidence_lower_days=120.0,
            confidence_upper_days=175.0,
            confidence_level=0.95,
            failure_probability_30d=0.02,
            failure_mode="Wall thinning due to oxidation",
            health_score=82.5,
            degradation_rate=0.12,
            last_maintenance=now - timedelta(days=180),
            recommended_action="Schedule inspection during next planned outage",
            predicted_at=now,
        ),
        RULPrediction(
            prediction_id="rul-002",
            component_id="BRN-003",
            component_type=ComponentType.BURNER,
            component_name="Burner Unit 3",
            rul_days=25.0,
            rul_hours=600.0,
            confidence_lower_days=18.0,
            confidence_upper_days=35.0,
            confidence_level=0.95,
            failure_probability_30d=0.65,
            failure_mode="Igniter electrode erosion",
            health_score=45.0,
            degradation_rate=0.85,
            last_maintenance=now - timedelta(days=365),
            recommended_action="URGENT: Schedule burner replacement within 2 weeks",
            predicted_at=now,
        ),
        RULPrediction(
            prediction_id="rul-003",
            component_id="TC-012",
            component_type=ComponentType.THERMOCOUPLE,
            component_name="Thermocouple Zone B2",
            rul_days=210.0,
            rul_hours=5040.0,
            confidence_lower_days=180.0,
            confidence_upper_days=250.0,
            confidence_level=0.95,
            failure_probability_30d=0.01,
            failure_mode="Drift due to contamination",
            health_score=92.0,
            degradation_rate=0.05,
            last_maintenance=now - timedelta(days=90),
            recommended_action="No immediate action required",
            predicted_at=now,
        ),
    ]

    # Apply filters
    if component_type:
        predictions = [p for p in predictions if p.component_type == component_type]

    if at_risk_only:
        predictions = [p for p in predictions if p.rul_days < 30]

    at_risk_count = sum(1 for p in predictions if p.rul_days < 30)
    avg_health = sum(p.health_score for p in predictions) / len(predictions) if predictions else 0

    # Find next maintenance
    next_maintenance = None
    if predictions:
        soonest = min(predictions, key=lambda p: p.rul_days)
        if soonest.rul_days < 60:
            next_maintenance = now + timedelta(days=max(0, soonest.rul_days - 7))

    return RULPredictionsResponse(
        furnace_id=furnace_id,
        timestamp=now,
        predictions=predictions,
        components_at_risk=at_risk_count,
        avg_health_score=avg_health,
        next_maintenance_due=next_maintenance,
        maintenance_window_start=now + timedelta(days=14) if next_maintenance else None,
        maintenance_window_end=now + timedelta(days=21) if next_maintenance else None,
    )


@router.post(
    "/furnaces/{furnace_id}/evidence",
    response_model=EvidencePackageResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Furnaces"],
    summary="Generate evidence package",
    description="Generate an evidence package for compliance or audit purposes.",
    dependencies=[Depends(require_roles([UserRole.SAFETY, UserRole.ADMIN]))],
)
async def generate_evidence_package(
    furnace_id: str = Path(..., description="Furnace identifier", example="FRN-001"),
    request: EvidenceGenerateRequest = ...,
    background_tasks: BackgroundTasks = None,
    current_user: Optional[User] = Depends(get_current_user),
) -> EvidencePackageResponse:
    """
    Generate an evidence package for a furnace.

    Args:
        furnace_id: Furnace identifier.
        request: Evidence package generation request.
        background_tasks: Background task handler.
        current_user: Authenticated user (requires safety or admin role).

    Returns:
        Evidence package metadata with status.
    """
    now = datetime.now(timezone.utc)
    package_id = f"EVP-{uuid.uuid4().hex[:12].upper()}"

    # Store job for background processing
    job_store[package_id] = {
        "status": "pending",
        "furnace_id": furnace_id,
        "request": request.model_dump(),
        "created_at": now,
        "created_by": current_user.user_id if current_user else "anonymous",
    }

    # In production, add background task to generate package
    # background_tasks.add_task(generate_evidence_task, package_id)

    return EvidencePackageResponse(
        package_id=package_id,
        furnace_id=furnace_id,
        package_type=request.package_type,
        status="processing",
        created_at=now,
        download_url=None,  # Will be populated when ready
        sha256_hash=None,
        size_bytes=None,
        expires_at=now + timedelta(days=7),
    )


# -----------------------------------------------------------------------------
# Alert Endpoints
# -----------------------------------------------------------------------------

@router.get(
    "/alerts",
    response_model=AlertsResponse,
    tags=["Alerts"],
    summary="List alerts",
    description="List alerts with filtering and pagination.",
)
async def list_alerts(
    furnace_id: Optional[str] = Query(None, description="Filter by furnace ID"),
    severity: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
    status: Optional[AlertStatus] = Query(None, description="Filter by status"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    current_user: Optional[User] = Depends(get_current_user),
) -> AlertsResponse:
    """
    List alerts with filtering and pagination.

    Args:
        furnace_id: Optional furnace ID filter.
        severity: Optional severity filter.
        status: Optional status filter.
        alert_type: Optional alert type filter.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        page: Page number.
        page_size: Page size.
        current_user: Authenticated user.

    Returns:
        Paginated list of alerts.
    """
    now = datetime.now(timezone.utc)

    # Mock alerts (replace with database query)
    all_alerts = [
        Alert(
            alert_id="ALT-001",
            furnace_id="FRN-001",
            alert_type="HOTSPOT_DETECTED",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            title="High temperature hotspot detected in Zone B2",
            description="TMT readings show sustained temperature 45C above normal in cluster-001",
            source="HotspotDetector",
            created_at=now - timedelta(minutes=15),
            metadata={"cluster_id": "cluster-001", "peak_temp_c": 925.0},
        ),
        Alert(
            alert_id="ALT-002",
            furnace_id="FRN-001",
            alert_type="RUL_WARNING",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.ACTIVE,
            title="Burner Unit 3 requires urgent maintenance",
            description="RUL prediction indicates 65% failure probability within 30 days",
            source="RULPredictor",
            created_at=now - timedelta(hours=2),
            metadata={"component_id": "BRN-003", "rul_days": 25.0},
        ),
        Alert(
            alert_id="ALT-003",
            furnace_id="FRN-002",
            alert_type="EFFICIENCY_DROP",
            severity=AlertSeverity.MEDIUM,
            status=AlertStatus.ACKNOWLEDGED,
            title="Thermal efficiency below target",
            description="Efficiency dropped to 82.5% (target: 90%)",
            source="EfficiencyCalculator",
            created_at=now - timedelta(hours=6),
            acknowledged_at=now - timedelta(hours=4),
            acknowledged_by="operator@example.com",
            metadata={"efficiency_pct": 82.5, "target_pct": 90.0},
        ),
    ]

    # Apply filters
    filtered = all_alerts
    if furnace_id:
        filtered = [a for a in filtered if a.furnace_id == furnace_id]
    if severity:
        filtered = [a for a in filtered if a.severity == severity]
    if status:
        filtered = [a for a in filtered if a.status == status]
    if alert_type:
        filtered = [a for a in filtered if a.alert_type == alert_type]
    if start_date:
        filtered = [a for a in filtered if a.created_at >= start_date]
    if end_date:
        filtered = [a for a in filtered if a.created_at <= end_date]

    # Pagination
    total = len(filtered)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = filtered[start_idx:end_idx]

    return AlertsResponse(
        alerts=paginated,
        total=total,
        page=page,
        page_size=page_size,
        has_more=end_idx < total,
    )


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertAcknowledgeResponse,
    tags=["Alerts"],
    summary="Acknowledge alert",
    description="Acknowledge an active alert.",
    dependencies=[Depends(require_roles([UserRole.OPERATOR, UserRole.ENGINEER, UserRole.SAFETY, UserRole.ADMIN]))],
)
async def acknowledge_alert(
    alert_id: str = Path(..., description="Alert identifier", example="ALT-001"),
    request: AlertAcknowledgeRequest = ...,
    current_user: Optional[User] = Depends(get_current_user),
) -> AlertAcknowledgeResponse:
    """
    Acknowledge an alert.

    Args:
        alert_id: Alert identifier.
        request: Acknowledgment request with optional notes.
        current_user: Authenticated user.

    Returns:
        Updated alert status.

    Raises:
        HTTPException: 404 if alert not found.
    """
    now = datetime.now(timezone.utc)

    # In production, update in database
    # For mock, just return success
    user_id = current_user.user_id if current_user else "anonymous"

    return AlertAcknowledgeResponse(
        alert_id=alert_id,
        status=AlertStatus.ACKNOWLEDGED,
        acknowledged_at=now,
        acknowledged_by=user_id,
    )


# -----------------------------------------------------------------------------
# Explainability Endpoints
# -----------------------------------------------------------------------------

@router.get(
    "/explain/{prediction_id}",
    response_model=ExplanationResponse,
    tags=["Explainability"],
    summary="Get prediction explanation",
    description="Get SHAP/LIME explanation for a ML prediction.",
)
async def get_prediction_explanation(
    prediction_id: str = Path(..., description="Prediction identifier", example="rul-001"),
    method: ExplanationType = Query(ExplanationType.SHAP, description="Explanation method"),
    current_user: Optional[User] = Depends(get_current_user),
) -> ExplanationResponse:
    """
    Get explanation for a ML prediction.

    Args:
        prediction_id: Prediction identifier.
        method: Explanation method (SHAP, LIME, etc.).
        current_user: Authenticated user.

    Returns:
        Feature importances and explanation summary.

    Raises:
        HTTPException: 404 if prediction not found.
    """
    now = datetime.now(timezone.utc)

    # Mock explanation (replace with actual XAI computation)
    feature_importance = [
        FeatureImportance(
            feature_name="operating_hours_since_maintenance",
            importance_value=0.35,
            direction="positive",
            contribution_pct=35.0,
        ),
        FeatureImportance(
            feature_name="avg_temperature_delta",
            importance_value=0.25,
            direction="positive",
            contribution_pct=25.0,
        ),
        FeatureImportance(
            feature_name="vibration_rms",
            importance_value=0.18,
            direction="positive",
            contribution_pct=18.0,
        ),
        FeatureImportance(
            feature_name="fuel_consumption_trend",
            importance_value=0.12,
            direction="negative",
            contribution_pct=12.0,
        ),
        FeatureImportance(
            feature_name="ambient_temperature",
            importance_value=0.10,
            direction="negative",
            contribution_pct=10.0,
        ),
    ]

    # Generate computation hash for provenance
    hash_input = f"{prediction_id}:{method.value}:{now.isoformat()}"
    computation_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    return ExplanationResponse(
        prediction_id=prediction_id,
        prediction_type="rul_prediction",
        prediction_value=25.0,
        explanation_method=method,
        feature_importance=feature_importance,
        summary=(
            "The RUL prediction of 25 days is primarily driven by the high number of "
            "operating hours since last maintenance (35% contribution) and elevated "
            "temperature delta readings (25% contribution). The burner has been operating "
            "for 8,760 hours since the last overhaul, significantly exceeding the "
            "recommended 6,000 hour service interval."
        ),
        key_drivers=[
            "Operating hours 46% above service interval",
            "Temperature delta trending upward over last 14 days",
            "Vibration levels increased 15% in last 7 days",
        ],
        confidence_score=0.87,
        counterfactuals=[
            {
                "scenario": "If maintenance performed 30 days ago",
                "new_rul_days": 180,
                "feature_changes": {"operating_hours_since_maintenance": -720},
            },
            {
                "scenario": "If temperature delta reduced by 20%",
                "new_rul_days": 45,
                "feature_changes": {"avg_temperature_delta": -8.5},
            },
        ],
        generated_at=now,
        computation_hash=computation_hash,
    )


# -----------------------------------------------------------------------------
# Metrics Endpoint
# -----------------------------------------------------------------------------

@router.get(
    "/metrics",
    tags=["Metrics"],
    summary="Prometheus metrics",
    description="Prometheus-compatible metrics endpoint.",
)
async def get_metrics() -> Response:
    """
    Get Prometheus-compatible metrics.

    Returns:
        Prometheus metrics in text format.
    """
    # Mock Prometheus metrics
    metrics = """
# HELP furnacepulse_active_alerts Number of active alerts
# TYPE furnacepulse_active_alerts gauge
furnacepulse_active_alerts{severity="critical"} 1
furnacepulse_active_alerts{severity="high"} 2
furnacepulse_active_alerts{severity="medium"} 3
furnacepulse_active_alerts{severity="low"} 5

# HELP furnacepulse_furnace_efficiency_percent Current furnace efficiency
# TYPE furnacepulse_furnace_efficiency_percent gauge
furnacepulse_furnace_efficiency_percent{furnace_id="FRN-001"} 87.5
furnacepulse_furnace_efficiency_percent{furnace_id="FRN-002"} 82.3
furnacepulse_furnace_efficiency_percent{furnace_id="FRN-003"} 91.2

# HELP furnacepulse_predictions_total Total predictions made
# TYPE furnacepulse_predictions_total counter
furnacepulse_predictions_total{type="rul"} 14420
furnacepulse_predictions_total{type="hotspot"} 8640
furnacepulse_predictions_total{type="efficiency"} 28800

# HELP furnacepulse_api_requests_total Total API requests
# TYPE furnacepulse_api_requests_total counter
furnacepulse_api_requests_total{method="GET",endpoint="/health"} 12500
furnacepulse_api_requests_total{method="GET",endpoint="/furnaces/kpis"} 8430
furnacepulse_api_requests_total{method="GET",endpoint="/alerts"} 3250

# HELP furnacepulse_api_request_duration_seconds API request duration
# TYPE furnacepulse_api_request_duration_seconds histogram
furnacepulse_api_request_duration_seconds_bucket{le="0.01"} 45000
furnacepulse_api_request_duration_seconds_bucket{le="0.05"} 52000
furnacepulse_api_request_duration_seconds_bucket{le="0.1"} 54000
furnacepulse_api_request_duration_seconds_bucket{le="0.5"} 55000
furnacepulse_api_request_duration_seconds_bucket{le="1.0"} 55200
furnacepulse_api_request_duration_seconds_bucket{le="+Inf"} 55250
furnacepulse_api_request_duration_seconds_sum 1250.5
furnacepulse_api_request_duration_seconds_count 55250

# HELP furnacepulse_compliance_score_percent NFPA 86 compliance score
# TYPE furnacepulse_compliance_score_percent gauge
furnacepulse_compliance_score_percent{furnace_id="FRN-001"} 96.5
furnacepulse_compliance_score_percent{furnace_id="FRN-002"} 100.0
furnacepulse_compliance_score_percent{furnace_id="FRN-003"} 92.0
"""

    return Response(
        content=metrics.strip(),
        media_type="text/plain; charset=utf-8",
    )


# =============================================================================
# Create Application Instance
# =============================================================================

app = create_app()
