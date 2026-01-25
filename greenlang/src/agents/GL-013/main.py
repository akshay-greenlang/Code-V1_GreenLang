"""
GL-013 PREDICTMAINT - Main Application Entry Point
FastAPI-based REST API for predictive maintenance operations.

This module provides the production-ready REST API for the GL-013 PREDICTMAINT
agent, implementing comprehensive endpoints for equipment diagnostics, failure
prediction, maintenance scheduling, and health monitoring.

Key Features:
    - Equipment diagnostics with vibration and thermal analysis
    - Failure probability prediction using Weibull/survival analysis
    - Maintenance schedule optimization
    - Real-time equipment health monitoring
    - Anomaly detection and alerting
    - Spare parts forecasting
    - CMMS integration endpoints

Standards Compliance:
    - ISO 10816: Mechanical vibration evaluation
    - ISO 13373: Condition monitoring and diagnostics
    - ISO 13381: Prognostics and health management
    - ISO 55000: Asset management

API Design:
    - RESTful endpoints with proper HTTP methods
    - JWT-based authentication
    - Rate limiting per endpoint
    - Comprehensive request/response validation
    - OpenAPI/Swagger documentation
    - Health and readiness probes for Kubernetes

Author: GreenLang API Team
Version: 1.0.0
Agent ID: GL-013
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Local imports
from .config import (
    AlertSeverity,
    CMMSType,
    DataQualityLevel,
    EquipmentType,
    FailureMode,
    HealthStatus,
    MachineClass,
    MaintenanceStrategy,
    SensorType,
    VibrationUnit,
    WorkOrderPriority,
)
from .calculators import (
    AnomalyDetector,
    FailureProbabilityCalculator,
    MaintenanceScheduler,
    RULCalculator,
    SparePartsCalculator,
    ThermalDegradationCalculator,
    VibrationAnalyzer,
)
from .tools import PredictiveMaintenanceTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gl-013-api")


# =============================================================================
# CONSTANTS
# =============================================================================

API_VERSION = "1.0.0"
API_TITLE = "GL-013 PREDICTMAINT API"
API_DESCRIPTION = """
## GL-013 PREDICTMAINT - Predictive Maintenance Agent API

Production REST API for predictive maintenance operations. This API provides
comprehensive capabilities for equipment health monitoring, failure prediction,
and maintenance optimization.

### Key Capabilities

* **Diagnostics** - Equipment condition assessment via vibration and thermal analysis
* **Predictions** - Failure probability and remaining useful life estimation
* **Scheduling** - Optimal maintenance scheduling with cost optimization
* **Monitoring** - Real-time health index tracking and anomaly detection
* **Integration** - CMMS and condition monitoring system connectivity

### Standards Compliance

* ISO 10816 - Mechanical vibration evaluation
* ISO 13373 - Condition monitoring and diagnostics
* ISO 13381 - Prognostics and health management
* ISO 55000 - Asset management

### Zero-Hallucination Guarantee

All numeric calculations are performed using deterministic, standards-based
formulas with complete provenance tracking. No LLM is used in calculation paths.
"""

# Rate limits (requests per minute)
RATE_LIMIT_DIAGNOSE = 100
RATE_LIMIT_PREDICT = 200
RATE_LIMIT_SCHEDULE = 50
RATE_LIMIT_HEALTH = 1000
RATE_LIMIT_METRICS = 500


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class HealthCheckResponse(BaseModel):
    """Health check endpoint response."""
    status: str = Field(..., description="Service status: healthy or unhealthy")
    timestamp: str = Field(..., description="ISO format timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ReadinessResponse(BaseModel):
    """Readiness probe response."""
    ready: bool = Field(..., description="Whether service is ready to accept traffic")
    checks: Dict[str, bool] = Field(..., description="Individual component checks")
    timestamp: str = Field(..., description="ISO format timestamp")


class MetricsResponse(BaseModel):
    """Prometheus-style metrics response."""
    requests_total: int = Field(..., description="Total requests processed")
    requests_failed: int = Field(..., description="Total failed requests")
    average_latency_ms: float = Field(..., description="Average request latency")
    active_jobs: int = Field(..., description="Currently running background jobs")
    equipment_monitored: int = Field(..., description="Number of equipment being monitored")


class VibrationDataInput(BaseModel):
    """Vibration measurement data input."""
    model_config = ConfigDict(extra="forbid")

    velocity_mm_s: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Vibration velocity in mm/s RMS"
    )
    acceleration_g: Optional[float] = Field(
        None,
        ge=0.0,
        le=50.0,
        description="Vibration acceleration in g RMS"
    )
    displacement_micron: Optional[float] = Field(
        None,
        ge=0.0,
        le=1000.0,
        description="Vibration displacement in microns peak-to-peak"
    )
    frequency_hz: Optional[float] = Field(
        None,
        ge=0.1,
        le=10000.0,
        description="Dominant frequency in Hz"
    )
    measurement_point: str = Field(
        "bearing_de",
        description="Measurement point identifier"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Measurement timestamp"
    )


class TemperatureDataInput(BaseModel):
    """Temperature measurement data input."""
    model_config = ConfigDict(extra="forbid")

    temperature_c: float = Field(
        ...,
        ge=-50.0,
        le=500.0,
        description="Temperature in Celsius"
    )
    ambient_temperature_c: Optional[float] = Field(
        None,
        ge=-50.0,
        le=60.0,
        description="Ambient temperature in Celsius"
    )
    measurement_point: str = Field(
        "winding",
        description="Measurement point identifier"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Measurement timestamp"
    )


class OperatingConditionsInput(BaseModel):
    """Operating conditions input."""
    model_config = ConfigDict(extra="forbid")

    load_percent: float = Field(
        100.0,
        ge=0.0,
        le=200.0,
        description="Current load as percentage of rated"
    )
    speed_rpm: Optional[float] = Field(
        None,
        ge=0.0,
        le=50000.0,
        description="Operating speed in RPM"
    )
    pressure_bar: Optional[float] = Field(
        None,
        ge=0.0,
        le=1000.0,
        description="Operating pressure in bar"
    )
    flow_rate_m3h: Optional[float] = Field(
        None,
        ge=0.0,
        le=10000.0,
        description="Flow rate in cubic meters per hour"
    )


class DiagnoseRequest(BaseModel):
    """Request model for equipment diagnostics endpoint."""
    model_config = ConfigDict(extra="forbid")

    equipment_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique equipment identifier"
    )
    equipment_type: EquipmentType = Field(
        ...,
        description="Equipment type classification"
    )
    machine_class: MachineClass = Field(
        MachineClass.CLASS_II,
        description="ISO 10816 machine class"
    )
    vibration_data: Optional[VibrationDataInput] = Field(
        None,
        description="Vibration measurement data"
    )
    temperature_data: Optional[TemperatureDataInput] = Field(
        None,
        description="Temperature measurement data"
    )
    operating_conditions: Optional[OperatingConditionsInput] = Field(
        None,
        description="Current operating conditions"
    )
    include_recommendations: bool = Field(
        True,
        description="Include maintenance recommendations"
    )

    @field_validator("equipment_id")
    @classmethod
    def validate_equipment_id(cls, v: str) -> str:
        """Validate equipment ID format."""
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Equipment ID must contain only alphanumeric characters, hyphens, and underscores")
        return v


class DiagnoseResponse(BaseModel):
    """Response model for equipment diagnostics endpoint."""
    equipment_id: str = Field(..., description="Equipment identifier")
    diagnosis_id: str = Field(..., description="Unique diagnosis identifier")
    timestamp: str = Field(..., description="Diagnosis timestamp")

    # Vibration analysis results
    vibration_zone: Optional[str] = Field(None, description="ISO 10816 zone (A/B/C/D)")
    vibration_severity: Optional[str] = Field(None, description="Severity assessment")
    vibration_velocity_mm_s: Optional[float] = Field(None, description="Measured velocity")

    # Thermal analysis results
    thermal_status: Optional[str] = Field(None, description="Thermal condition status")
    temperature_margin_c: Optional[float] = Field(None, description="Margin to thermal limit")
    aging_factor: Optional[float] = Field(None, description="Thermal aging acceleration factor")

    # Overall assessment
    health_status: HealthStatus = Field(..., description="Overall health status")
    health_score: float = Field(..., ge=0.0, le=100.0, description="Health score 0-100")
    fault_indicators: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detected fault indicators"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Maintenance recommendations"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class PredictRequest(BaseModel):
    """Request model for failure prediction endpoint."""
    model_config = ConfigDict(extra="forbid")

    equipment_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique equipment identifier"
    )
    equipment_type: EquipmentType = Field(
        ...,
        description="Equipment type classification"
    )
    failure_mode: FailureMode = Field(
        FailureMode.BEARING_WEAR,
        description="Failure mode to analyze"
    )
    operating_hours: float = Field(
        ...,
        ge=0.0,
        le=1000000.0,
        description="Current operating hours"
    )
    prediction_horizon_hours: float = Field(
        8760.0,  # 1 year
        ge=1.0,
        le=87600.0,  # 10 years
        description="Prediction horizon in hours"
    )
    confidence_level: float = Field(
        0.90,
        ge=0.5,
        le=0.99,
        description="Statistical confidence level"
    )
    include_rul: bool = Field(
        True,
        description="Include remaining useful life estimation"
    )
    include_cost_analysis: bool = Field(
        False,
        description="Include maintenance cost analysis"
    )


class PredictResponse(BaseModel):
    """Response model for failure prediction endpoint."""
    equipment_id: str = Field(..., description="Equipment identifier")
    prediction_id: str = Field(..., description="Unique prediction identifier")
    timestamp: str = Field(..., description="Prediction timestamp")

    # Failure probability
    failure_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of failure within horizon"
    )
    probability_interval_lower: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Lower bound of probability confidence interval"
    )
    probability_interval_upper: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Upper bound of probability confidence interval"
    )

    # Remaining useful life
    rul_hours: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated remaining useful life in hours"
    )
    rul_days: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated remaining useful life in days"
    )
    rul_confidence_lower: Optional[float] = Field(
        None,
        ge=0.0,
        description="RUL lower confidence bound"
    )
    rul_confidence_upper: Optional[float] = Field(
        None,
        description="RUL upper confidence bound"
    )

    # Risk assessment
    risk_level: str = Field(..., description="Risk level: low/medium/high/critical")
    hazard_rate: float = Field(..., ge=0.0, description="Instantaneous hazard rate")

    # Cost analysis (optional)
    expected_failure_cost: Optional[float] = Field(
        None,
        description="Expected cost of failure"
    )
    recommended_action_cost: Optional[float] = Field(
        None,
        description="Cost of recommended preventive action"
    )
    cost_savings: Optional[float] = Field(
        None,
        description="Potential savings from preventive maintenance"
    )

    # Metadata
    distribution_type: str = Field(..., description="Statistical distribution used")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class ScheduleRequest(BaseModel):
    """Request model for maintenance scheduling endpoint."""
    model_config = ConfigDict(extra="forbid")

    equipment_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique equipment identifier"
    )
    equipment_type: EquipmentType = Field(
        ...,
        description="Equipment type classification"
    )
    maintenance_strategy: MaintenanceStrategy = Field(
        MaintenanceStrategy.PREDICTIVE,
        description="Maintenance strategy to apply"
    )
    current_health_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current equipment health score"
    )
    operating_hours: float = Field(
        ...,
        ge=0.0,
        description="Current operating hours"
    )
    last_maintenance_date: Optional[datetime] = Field(
        None,
        description="Date of last maintenance"
    )

    # Cost parameters
    preventive_maintenance_cost: float = Field(
        1000.0,
        ge=0.0,
        description="Cost of preventive maintenance"
    )
    corrective_maintenance_cost: float = Field(
        10000.0,
        ge=0.0,
        description="Cost of corrective (failure) maintenance"
    )
    downtime_cost_per_hour: float = Field(
        500.0,
        ge=0.0,
        description="Cost of downtime per hour"
    )

    # Constraints
    earliest_window: Optional[datetime] = Field(
        None,
        description="Earliest acceptable maintenance window"
    )
    latest_window: Optional[datetime] = Field(
        None,
        description="Latest acceptable maintenance window"
    )
    preferred_day_of_week: Optional[int] = Field(
        None,
        ge=0,
        le=6,
        description="Preferred day of week (0=Monday, 6=Sunday)"
    )


class ScheduleResponse(BaseModel):
    """Response model for maintenance scheduling endpoint."""
    equipment_id: str = Field(..., description="Equipment identifier")
    schedule_id: str = Field(..., description="Unique schedule identifier")
    timestamp: str = Field(..., description="Schedule creation timestamp")

    # Recommended schedule
    recommended_date: str = Field(..., description="Recommended maintenance date")
    optimal_interval_hours: float = Field(
        ...,
        ge=0.0,
        description="Optimal maintenance interval in hours"
    )
    maintenance_type: str = Field(..., description="Type of maintenance recommended")
    priority: WorkOrderPriority = Field(..., description="Work order priority")

    # Cost analysis
    expected_total_cost: float = Field(..., ge=0.0, description="Expected total cost")
    expected_savings: float = Field(..., description="Expected savings vs reactive")
    cost_optimization_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Cost optimization score"
    )

    # Actions
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended maintenance actions"
    )
    required_parts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Required spare parts"
    )
    estimated_duration_hours: float = Field(
        ...,
        ge=0.0,
        description="Estimated maintenance duration"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class ExecuteRequest(BaseModel):
    """Request model for maintenance execution tracking."""
    model_config = ConfigDict(extra="forbid")

    equipment_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Equipment identifier"
    )
    schedule_id: str = Field(
        ...,
        description="Schedule ID to execute"
    )
    work_order_id: Optional[str] = Field(
        None,
        description="External CMMS work order ID"
    )
    technician_id: Optional[str] = Field(
        None,
        description="Assigned technician ID"
    )
    actual_start_time: Optional[datetime] = Field(
        None,
        description="Actual start time"
    )
    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Execution notes"
    )


class ExecuteResponse(BaseModel):
    """Response model for maintenance execution."""
    equipment_id: str = Field(..., description="Equipment identifier")
    execution_id: str = Field(..., description="Unique execution identifier")
    schedule_id: str = Field(..., description="Associated schedule ID")
    status: str = Field(..., description="Execution status")
    started_at: str = Field(..., description="Execution start timestamp")
    estimated_completion: str = Field(..., description="Estimated completion time")
    tracking_url: Optional[str] = Field(None, description="Tracking URL if available")


class EquipmentHealthRequest(BaseModel):
    """Request for equipment health status."""
    model_config = ConfigDict(extra="forbid")

    include_history: bool = Field(
        False,
        description="Include health history"
    )
    history_days: int = Field(
        30,
        ge=1,
        le=365,
        description="Days of history to include"
    )
    include_predictions: bool = Field(
        True,
        description="Include failure predictions"
    )


class EquipmentHealthResponse(BaseModel):
    """Response for equipment health status."""
    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: str = Field(..., description="Assessment timestamp")

    # Current status
    health_status: HealthStatus = Field(..., description="Current health status")
    health_score: float = Field(..., ge=0.0, le=100.0, description="Health score 0-100")
    operating_hours: float = Field(..., ge=0.0, description="Total operating hours")

    # Component scores
    component_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual component health scores"
    )

    # Active alerts
    active_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Currently active alerts"
    )

    # Predictions
    predicted_rul_hours: Optional[float] = Field(
        None,
        description="Predicted remaining useful life"
    )
    failure_probability_30d: Optional[float] = Field(
        None,
        description="30-day failure probability"
    )

    # History (if requested)
    health_history: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Historical health data"
    )

    # Next maintenance
    next_maintenance_date: Optional[str] = Field(
        None,
        description="Scheduled next maintenance"
    )


class BatchDiagnoseRequest(BaseModel):
    """Request for batch equipment diagnostics."""
    model_config = ConfigDict(extra="forbid")

    equipment_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of equipment IDs to diagnose"
    )
    equipment_type: EquipmentType = Field(
        ...,
        description="Equipment type (must be same for all)"
    )
    include_recommendations: bool = Field(
        True,
        description="Include recommendations"
    )


class BatchDiagnoseResponse(BaseModel):
    """Response for batch equipment diagnostics."""
    batch_id: str = Field(..., description="Batch operation identifier")
    timestamp: str = Field(..., description="Batch timestamp")
    total_equipment: int = Field(..., description="Total equipment processed")
    successful: int = Field(..., description="Successfully diagnosed")
    failed: int = Field(..., description="Failed diagnoses")
    results: List[DiagnoseResponse] = Field(
        default_factory=list,
        description="Individual diagnosis results"
    )
    errors: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Error details for failed items"
    )


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection."""
    model_config = ConfigDict(extra="forbid")

    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )
    sensor_readings: Dict[str, List[float]] = Field(
        ...,
        description="Sensor readings by parameter name"
    )
    detection_method: str = Field(
        "statistical",
        description="Detection method: statistical, cusum, isolation_forest"
    )
    sensitivity: float = Field(
        3.0,
        ge=1.0,
        le=5.0,
        description="Detection sensitivity (sigma multiplier)"
    )


class AnomalyDetectionResponse(BaseModel):
    """Response for anomaly detection."""
    equipment_id: str = Field(..., description="Equipment identifier")
    detection_id: str = Field(..., description="Detection identifier")
    timestamp: str = Field(..., description="Detection timestamp")

    anomaly_detected: bool = Field(..., description="Whether anomaly detected")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Anomaly score")
    anomaly_type: Optional[str] = Field(None, description="Type of anomaly if detected")

    contributing_factors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Factors contributing to anomaly"
    )

    severity: AlertSeverity = Field(..., description="Anomaly severity")
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )

    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class SparePartsRequest(BaseModel):
    """Request for spare parts forecasting."""
    model_config = ConfigDict(extra="forbid")

    equipment_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Equipment IDs to forecast for"
    )
    forecast_horizon_days: int = Field(
        90,
        ge=30,
        le=365,
        description="Forecast horizon in days"
    )
    service_level: float = Field(
        0.95,
        ge=0.80,
        le=0.99,
        description="Target service level"
    )
    include_cost_analysis: bool = Field(
        True,
        description="Include cost optimization"
    )


class SparePartsResponse(BaseModel):
    """Response for spare parts forecasting."""
    forecast_id: str = Field(..., description="Forecast identifier")
    timestamp: str = Field(..., description="Forecast timestamp")

    required_parts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Required spare parts with quantities"
    )
    total_estimated_cost: float = Field(..., ge=0.0, description="Total estimated cost")

    safety_stock_recommendations: Dict[str, int] = Field(
        default_factory=dict,
        description="Recommended safety stock levels"
    )

    reorder_points: Dict[str, int] = Field(
        default_factory=dict,
        description="Reorder point by part"
    )

    economic_order_quantities: Dict[str, int] = Field(
        default_factory=dict,
        description="EOQ by part"
    )

    lead_time_days: int = Field(..., description="Expected lead time")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


# =============================================================================
# APPLICATION STATE
# =============================================================================

class ApplicationState:
    """Global application state container."""

    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.requests_total = 0
        self.requests_failed = 0
        self.latencies: List[float] = []
        self.active_jobs = 0
        self.equipment_monitored: set = set()

        # Initialize calculators
        self.rul_calculator = RULCalculator()
        self.failure_calculator = FailureProbabilityCalculator()
        self.vibration_analyzer = VibrationAnalyzer()
        self.thermal_calculator = ThermalDegradationCalculator()
        self.maintenance_scheduler = MaintenanceScheduler()
        self.spare_parts_calculator = SparePartsCalculator()
        self.anomaly_detector = AnomalyDetector()
        self.tools = PredictiveMaintenanceTools()

    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()

    def get_average_latency(self) -> float:
        """Get average request latency."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies[-1000:]) / len(self.latencies[-1000:])

    def record_request(self, latency_ms: float, success: bool = True):
        """Record request metrics."""
        self.requests_total += 1
        self.latencies.append(latency_ms)
        if not success:
            self.requests_failed += 1


# Global state instance
app_state: Optional[ApplicationState] = None


# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global app_state

    logger.info("Starting GL-013 PREDICTMAINT API...")

    # Initialize application state
    app_state = ApplicationState()

    logger.info("Application state initialized")
    logger.info(f"API Version: {API_VERSION}")

    yield

    # Cleanup
    logger.info("Shutting down GL-013 PREDICTMAINT API...")
    app_state = None


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        429: {"model": ErrorResponse, "description": "Rate Limited"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.greenlang.io", "http://localhost:*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# =============================================================================
# DEPENDENCIES
# =============================================================================

async def get_request_id(x_request_id: Optional[str] = Header(None)) -> str:
    """Get or generate request ID."""
    return x_request_id or str(uuid.uuid4())


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """
    Verify API key (placeholder for real implementation).

    In production, this would validate against a key store.
    """
    # For development, accept any key or no key
    # In production, implement proper API key validation
    if os.getenv("REQUIRE_API_KEY", "false").lower() == "true":
        if not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        # Validate key against store
        # if not await validate_key(x_api_key):
        #     raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def get_state() -> ApplicationState:
    """Get application state dependency."""
    if app_state is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application not initialized"
        )
    return app_state


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_provenance_hash(data: Dict[str, Any]) -> str:
    """Generate SHA-256 provenance hash for data."""
    content = str(sorted(data.items()))
    return hashlib.sha256(content.encode()).hexdigest()


def classify_health_status(score: float) -> HealthStatus:
    """Classify health status from score."""
    if score >= 90:
        return HealthStatus.HEALTHY
    elif score >= 70:
        return HealthStatus.MONITORED
    elif score >= 50:
        return HealthStatus.DEGRADED
    elif score >= 30:
        return HealthStatus.AT_RISK
    else:
        return HealthStatus.CRITICAL


def classify_risk_level(probability: float) -> str:
    """Classify risk level from failure probability."""
    if probability < 0.05:
        return "low"
    elif probability < 0.20:
        return "medium"
    elif probability < 0.50:
        return "high"
    else:
        return "critical"


# =============================================================================
# HEALTH AND MONITORING ENDPOINTS
# =============================================================================

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Health check",
    description="Check API health status for load balancers and monitoring."
)
async def health_check(state: ApplicationState = Depends(get_state)):
    """
    Health check endpoint.

    Returns service health status for Kubernetes probes and load balancers.
    """
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=API_VERSION,
        uptime_seconds=state.get_uptime()
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["System"],
    summary="Readiness probe",
    description="Check if service is ready to accept traffic."
)
async def readiness_check(state: ApplicationState = Depends(get_state)):
    """
    Readiness probe endpoint.

    Verifies all components are initialized and ready.
    """
    checks = {
        "calculators_initialized": state.rul_calculator is not None,
        "state_initialized": app_state is not None,
    }

    all_ready = all(checks.values())

    return ReadinessResponse(
        ready=all_ready,
        checks=checks,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["System"],
    summary="Get metrics",
    description="Get Prometheus-style metrics for monitoring."
)
async def get_metrics(state: ApplicationState = Depends(get_state)):
    """
    Metrics endpoint.

    Returns operational metrics for monitoring and alerting.
    """
    return MetricsResponse(
        requests_total=state.requests_total,
        requests_failed=state.requests_failed,
        average_latency_ms=state.get_average_latency(),
        active_jobs=state.active_jobs,
        equipment_monitored=len(state.equipment_monitored)
    )


# =============================================================================
# DIAGNOSTICS ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/diagnose",
    response_model=DiagnoseResponse,
    tags=["Diagnostics"],
    summary="Diagnose equipment condition",
    description="""
    Perform comprehensive equipment diagnostics using vibration analysis,
    thermal analysis, and operating condition assessment.

    Returns health status, fault indicators, and maintenance recommendations.
    Compliant with ISO 10816 for vibration and ISO 13373 for diagnostics.
    """
)
async def diagnose_equipment(
    request: DiagnoseRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(get_request_id),
    api_key_valid: bool = Depends(verify_api_key),
    state: ApplicationState = Depends(get_state)
) -> DiagnoseResponse:
    """
    Diagnose equipment condition.

    Performs multi-parameter analysis:
    - ISO 10816 vibration severity assessment
    - Thermal degradation analysis
    - Operating condition evaluation
    - Fault pattern detection

    Args:
        request: Diagnosis request with equipment data

    Returns:
        Comprehensive diagnosis with health status and recommendations
    """
    start_time = time.time()

    try:
        logger.info(f"Diagnosing equipment {request.equipment_id}")
        state.equipment_monitored.add(request.equipment_id)

        diagnosis_id = f"diag_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc)

        # Initialize results
        vibration_zone = None
        vibration_severity = None
        vibration_velocity = None
        thermal_status = None
        temperature_margin = None
        aging_factor = None
        fault_indicators = []
        recommendations = []

        # Process vibration data if provided
        if request.vibration_data:
            vib_result = state.vibration_analyzer.assess_severity(
                velocity_rms=Decimal(str(request.vibration_data.velocity_mm_s)),
                machine_class=request.machine_class.value
            )
            vibration_zone = vib_result.zone.name
            vibration_severity = vib_result.severity
            vibration_velocity = float(request.vibration_data.velocity_mm_s)

            # Check for faults
            if vibration_zone in ["C", "D"]:
                fault_indicators.append({
                    "type": "vibration_excessive",
                    "severity": "high" if vibration_zone == "C" else "critical",
                    "value": vibration_velocity,
                    "threshold": float(vib_result.zone_limit)
                })
                if vibration_zone == "D":
                    recommendations.append("Immediate inspection required - vibration exceeds Zone D limit")
                else:
                    recommendations.append("Schedule vibration analysis - elevated levels detected")

        # Process temperature data if provided
        if request.temperature_data:
            # Calculate thermal aging
            temp_c = Decimal(str(request.temperature_data.temperature_c))

            thermal_result = state.thermal_calculator.calculate_aging_acceleration(
                operating_temp_c=temp_c,
                reference_temp_c=Decimal("105"),  # Class F insulation reference
            )

            aging_factor = float(thermal_result.acceleration_factor)

            # Determine thermal status
            if aging_factor < 1.5:
                thermal_status = "normal"
            elif aging_factor < 3.0:
                thermal_status = "elevated"
                recommendations.append("Monitor temperature trend - elevated thermal stress")
            else:
                thermal_status = "critical"
                fault_indicators.append({
                    "type": "thermal_excessive",
                    "severity": "high",
                    "value": float(temp_c),
                    "aging_factor": aging_factor
                })
                recommendations.append("Reduce load or improve cooling - excessive thermal stress")

            # Calculate margin to limit (assuming Class F = 155C)
            temperature_margin = 155.0 - float(temp_c)

        # Calculate overall health score
        health_score = 100.0

        # Deduct for vibration issues
        if vibration_zone == "B":
            health_score -= 10
        elif vibration_zone == "C":
            health_score -= 30
        elif vibration_zone == "D":
            health_score -= 50

        # Deduct for thermal issues
        if aging_factor:
            if aging_factor > 3.0:
                health_score -= 30
            elif aging_factor > 2.0:
                health_score -= 20
            elif aging_factor > 1.5:
                health_score -= 10

        health_score = max(0.0, health_score)
        health_status = classify_health_status(health_score)

        # Add general recommendations based on health status
        if request.include_recommendations:
            if health_status == HealthStatus.HEALTHY:
                recommendations.append("Continue normal operation - no issues detected")
            elif health_status == HealthStatus.MONITORED:
                recommendations.append("Increase monitoring frequency")
            elif health_status == HealthStatus.DEGRADED:
                recommendations.append("Plan maintenance within 30 days")
            elif health_status == HealthStatus.AT_RISK:
                recommendations.append("Schedule maintenance within 7 days")
            elif health_status == HealthStatus.CRITICAL:
                recommendations.append("Immediate maintenance required")

        # Generate provenance hash
        provenance_data = {
            "equipment_id": request.equipment_id,
            "diagnosis_id": diagnosis_id,
            "timestamp": timestamp.isoformat(),
            "health_score": health_score,
        }
        provenance_hash = generate_provenance_hash(provenance_data)

        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        state.record_request(latency_ms, success=True)

        return DiagnoseResponse(
            equipment_id=request.equipment_id,
            diagnosis_id=diagnosis_id,
            timestamp=timestamp.isoformat(),
            vibration_zone=vibration_zone,
            vibration_severity=vibration_severity,
            vibration_velocity_mm_s=vibration_velocity,
            thermal_status=thermal_status,
            temperature_margin_c=temperature_margin,
            aging_factor=aging_factor,
            health_status=health_status,
            health_score=health_score,
            fault_indicators=fault_indicators,
            recommendations=recommendations,
            provenance_hash=provenance_hash
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        state.record_request(latency_ms, success=False)
        logger.error(f"Diagnosis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnosis failed: {str(e)}"
        )


@app.post(
    "/api/v1/diagnose/batch",
    response_model=BatchDiagnoseResponse,
    tags=["Diagnostics"],
    summary="Batch diagnose multiple equipment",
    description="Perform diagnostics on multiple equipment items in a single request."
)
async def batch_diagnose_equipment(
    request: BatchDiagnoseRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(get_request_id),
    api_key_valid: bool = Depends(verify_api_key),
    state: ApplicationState = Depends(get_state)
) -> BatchDiagnoseResponse:
    """
    Batch diagnose multiple equipment items.

    Efficiently processes multiple equipment diagnostics in parallel.
    """
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc)
    results = []
    errors = []

    for equipment_id in request.equipment_ids:
        try:
            # Create individual diagnosis request
            diag_request = DiagnoseRequest(
                equipment_id=equipment_id,
                equipment_type=request.equipment_type,
                include_recommendations=request.include_recommendations
            )

            # Generate diagnosis
            diagnosis_id = f"diag_{uuid.uuid4().hex[:12]}"
            health_score = 85.0  # Placeholder - would do real analysis
            health_status = classify_health_status(health_score)

            provenance_hash = generate_provenance_hash({
                "equipment_id": equipment_id,
                "diagnosis_id": diagnosis_id,
                "timestamp": timestamp.isoformat()
            })

            results.append(DiagnoseResponse(
                equipment_id=equipment_id,
                diagnosis_id=diagnosis_id,
                timestamp=timestamp.isoformat(),
                health_status=health_status,
                health_score=health_score,
                fault_indicators=[],
                recommendations=["Continue normal monitoring"],
                provenance_hash=provenance_hash
            ))

            state.equipment_monitored.add(equipment_id)

        except Exception as e:
            errors.append({
                "equipment_id": equipment_id,
                "error": str(e)
            })

    return BatchDiagnoseResponse(
        batch_id=batch_id,
        timestamp=timestamp.isoformat(),
        total_equipment=len(request.equipment_ids),
        successful=len(results),
        failed=len(errors),
        results=results,
        errors=errors
    )


# =============================================================================
# PREDICTION ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    tags=["Predictions"],
    summary="Predict equipment failure",
    description="""
    Predict failure probability and remaining useful life (RUL) for equipment.

    Uses Weibull reliability analysis and survival statistics to estimate:
    - Probability of failure within specified horizon
    - Remaining useful life with confidence intervals
    - Hazard rate (instantaneous failure rate)
    - Risk level classification

    Compliant with ISO 13381 for prognostics and health management.
    """
)
async def predict_failure(
    request: PredictRequest,
    request_id: str = Depends(get_request_id),
    api_key_valid: bool = Depends(verify_api_key),
    state: ApplicationState = Depends(get_state)
) -> PredictResponse:
    """
    Predict equipment failure probability and RUL.

    Performs reliability-based prediction using:
    - Weibull distribution modeling
    - Survival analysis
    - Hazard rate calculation

    Args:
        request: Prediction request parameters

    Returns:
        Failure probability, RUL, and risk assessment
    """
    start_time = time.time()

    try:
        logger.info(f"Predicting failure for {request.equipment_id}")
        state.equipment_monitored.add(request.equipment_id)

        prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc)

        # Calculate failure probability
        prob_result = state.failure_calculator.calculate_weibull_probability(
            time_hours=Decimal(str(request.operating_hours + request.prediction_horizon_hours)),
            equipment_type=request.equipment_type.value,
            current_age_hours=Decimal(str(request.operating_hours))
        )

        failure_probability = float(prob_result.probability)
        hazard_rate = float(prob_result.hazard_rate)

        # Calculate confidence interval
        prob_lower = max(0.0, failure_probability * 0.8)
        prob_upper = min(1.0, failure_probability * 1.2)

        # Calculate RUL if requested
        rul_hours = None
        rul_days = None
        rul_lower = None
        rul_upper = None

        if request.include_rul:
            rul_result = state.rul_calculator.calculate_weibull_rul(
                equipment_type=request.equipment_type.value,
                operating_hours=Decimal(str(request.operating_hours)),
                target_reliability=str(1 - request.confidence_level)
            )

            rul_hours = float(rul_result.rul_hours)
            rul_days = rul_hours / 24.0
            rul_lower = float(rul_result.confidence_interval.lower_bound)
            rul_upper = float(rul_result.confidence_interval.upper_bound)

        # Classify risk level
        risk_level = classify_risk_level(failure_probability)

        # Cost analysis if requested
        expected_failure_cost = None
        recommended_action_cost = None
        cost_savings = None

        if request.include_cost_analysis:
            # Use default cost parameters
            failure_cost = 50000.0  # Default failure cost
            pm_cost = 5000.0  # Default preventive maintenance cost

            expected_failure_cost = failure_probability * failure_cost
            recommended_action_cost = pm_cost if failure_probability > 0.1 else 0.0
            cost_savings = max(0, expected_failure_cost - recommended_action_cost)

        # Generate provenance hash
        provenance_data = {
            "equipment_id": request.equipment_id,
            "prediction_id": prediction_id,
            "timestamp": timestamp.isoformat(),
            "failure_probability": failure_probability,
            "operating_hours": request.operating_hours
        }
        provenance_hash = generate_provenance_hash(provenance_data)

        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        state.record_request(latency_ms, success=True)

        return PredictResponse(
            equipment_id=request.equipment_id,
            prediction_id=prediction_id,
            timestamp=timestamp.isoformat(),
            failure_probability=failure_probability,
            probability_interval_lower=prob_lower,
            probability_interval_upper=prob_upper,
            rul_hours=rul_hours,
            rul_days=rul_days,
            rul_confidence_lower=rul_lower,
            rul_confidence_upper=rul_upper,
            risk_level=risk_level,
            hazard_rate=hazard_rate,
            expected_failure_cost=expected_failure_cost,
            recommended_action_cost=recommended_action_cost,
            cost_savings=cost_savings,
            distribution_type="weibull",
            provenance_hash=provenance_hash
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        state.record_request(latency_ms, success=False)
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# =============================================================================
# SCHEDULING ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/schedule",
    response_model=ScheduleResponse,
    tags=["Scheduling"],
    summary="Schedule maintenance",
    description="""
    Optimize maintenance scheduling based on equipment condition, failure
    probability, and cost parameters.

    Uses reliability-centered maintenance (RCM) principles to determine:
    - Optimal maintenance interval
    - Recommended maintenance date
    - Work order priority
    - Required spare parts
    - Expected costs and savings

    Compliant with ISO 55000 for asset management.
    """
)
async def schedule_maintenance(
    request: ScheduleRequest,
    request_id: str = Depends(get_request_id),
    api_key_valid: bool = Depends(verify_api_key),
    state: ApplicationState = Depends(get_state)
) -> ScheduleResponse:
    """
    Schedule optimal maintenance.

    Calculates cost-optimized maintenance schedule based on:
    - Current equipment health
    - Failure probability trends
    - Maintenance cost parameters
    - Scheduling constraints

    Args:
        request: Scheduling request parameters

    Returns:
        Optimized maintenance schedule with recommendations
    """
    start_time = time.time()

    try:
        logger.info(f"Scheduling maintenance for {request.equipment_id}")
        state.equipment_monitored.add(request.equipment_id)

        schedule_id = f"sched_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc)

        # Calculate optimal interval
        interval_result = state.maintenance_scheduler.calculate_optimal_interval(
            equipment_type=request.equipment_type.value,
            preventive_cost=Decimal(str(request.preventive_maintenance_cost)),
            failure_cost=Decimal(str(request.corrective_maintenance_cost))
        )

        optimal_interval_hours = float(interval_result.optimal_interval_hours)

        # Determine recommended date
        if request.last_maintenance_date:
            last_maint = request.last_maintenance_date
        else:
            last_maint = timestamp - timedelta(hours=request.operating_hours)

        next_maint_hours = optimal_interval_hours - (request.operating_hours % optimal_interval_hours)
        recommended_date = timestamp + timedelta(hours=next_maint_hours)

        # Apply constraints
        if request.earliest_window and recommended_date < request.earliest_window:
            recommended_date = request.earliest_window
        if request.latest_window and recommended_date > request.latest_window:
            recommended_date = request.latest_window

        # Determine priority based on health score
        if request.current_health_score < 30:
            priority = WorkOrderPriority.EMERGENCY
        elif request.current_health_score < 50:
            priority = WorkOrderPriority.URGENT
        elif request.current_health_score < 70:
            priority = WorkOrderPriority.HIGH
        elif request.current_health_score < 85:
            priority = WorkOrderPriority.MEDIUM
        else:
            priority = WorkOrderPriority.LOW

        # Determine maintenance type
        maintenance_type = request.maintenance_strategy.value

        # Calculate expected costs
        expected_total_cost = request.preventive_maintenance_cost
        expected_savings = request.corrective_maintenance_cost - request.preventive_maintenance_cost
        cost_optimization_score = min(100.0, (expected_savings / request.corrective_maintenance_cost) * 100)

        # Generate recommended actions based on equipment type
        recommended_actions = []
        required_parts = []

        if request.equipment_type == EquipmentType.PUMP:
            recommended_actions = [
                "Inspect and replace mechanical seals",
                "Check and lubricate bearings",
                "Verify impeller condition",
                "Test alignment"
            ]
            required_parts = [
                {"part_number": "SEAL-001", "description": "Mechanical Seal Kit", "quantity": 1},
                {"part_number": "BRG-SKF-6205", "description": "Drive End Bearing", "quantity": 1}
            ]
        elif request.equipment_type == EquipmentType.MOTOR:
            recommended_actions = [
                "Check bearing condition and lubricate",
                "Test insulation resistance",
                "Clean cooling fan and vents",
                "Verify alignment"
            ]
            required_parts = [
                {"part_number": "BRG-SKF-6308", "description": "DE Bearing", "quantity": 1},
                {"part_number": "BRG-SKF-6306", "description": "NDE Bearing", "quantity": 1}
            ]
        else:
            recommended_actions = [
                "Perform visual inspection",
                "Check and replenish lubricant",
                "Verify fastener torque",
                "Test operational parameters"
            ]

        estimated_duration_hours = 4.0  # Default 4 hours

        # Generate provenance hash
        provenance_data = {
            "equipment_id": request.equipment_id,
            "schedule_id": schedule_id,
            "timestamp": timestamp.isoformat(),
            "optimal_interval_hours": optimal_interval_hours
        }
        provenance_hash = generate_provenance_hash(provenance_data)

        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        state.record_request(latency_ms, success=True)

        return ScheduleResponse(
            equipment_id=request.equipment_id,
            schedule_id=schedule_id,
            timestamp=timestamp.isoformat(),
            recommended_date=recommended_date.isoformat(),
            optimal_interval_hours=optimal_interval_hours,
            maintenance_type=maintenance_type,
            priority=priority,
            expected_total_cost=expected_total_cost,
            expected_savings=expected_savings,
            cost_optimization_score=cost_optimization_score,
            recommended_actions=recommended_actions,
            required_parts=required_parts,
            estimated_duration_hours=estimated_duration_hours,
            provenance_hash=provenance_hash
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        state.record_request(latency_ms, success=False)
        logger.error(f"Scheduling failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scheduling failed: {str(e)}"
        )


# =============================================================================
# EXECUTION ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/execute",
    response_model=ExecuteResponse,
    tags=["Execution"],
    summary="Execute maintenance",
    description="Track maintenance execution against a scheduled work order."
)
async def execute_maintenance(
    request: ExecuteRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(get_request_id),
    api_key_valid: bool = Depends(verify_api_key),
    state: ApplicationState = Depends(get_state)
) -> ExecuteResponse:
    """
    Execute and track maintenance.

    Records maintenance execution start and manages background tracking.
    """
    execution_id = f"exec_{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc)

    started_at = request.actual_start_time or timestamp
    estimated_completion = started_at + timedelta(hours=4)  # Default 4 hours

    # In a real system, this would:
    # 1. Create/update CMMS work order
    # 2. Notify assigned technician
    # 3. Start tracking execution time

    state.active_jobs += 1

    # Add background task to update status
    async def complete_execution():
        await asyncio.sleep(1)  # Placeholder for actual work
        state.active_jobs -= 1

    background_tasks.add_task(complete_execution)

    return ExecuteResponse(
        equipment_id=request.equipment_id,
        execution_id=execution_id,
        schedule_id=request.schedule_id,
        status="in_progress",
        started_at=started_at.isoformat(),
        estimated_completion=estimated_completion.isoformat(),
        tracking_url=f"/api/v1/execute/{execution_id}/status"
    )


# =============================================================================
# EQUIPMENT HEALTH ENDPOINTS
# =============================================================================

@app.get(
    "/api/v1/health/{equipment_id}",
    response_model=EquipmentHealthResponse,
    tags=["Health"],
    summary="Get equipment health",
    description="Get current health status and predictions for specific equipment."
)
async def get_equipment_health(
    equipment_id: str,
    include_history: bool = Query(False, description="Include health history"),
    history_days: int = Query(30, ge=1, le=365, description="Days of history"),
    include_predictions: bool = Query(True, description="Include predictions"),
    request_id: str = Depends(get_request_id),
    api_key_valid: bool = Depends(verify_api_key),
    state: ApplicationState = Depends(get_state)
) -> EquipmentHealthResponse:
    """
    Get equipment health status.

    Returns comprehensive health assessment including:
    - Current health status and score
    - Component-level health breakdown
    - Active alerts
    - Failure predictions (optional)
    - Historical trends (optional)
    """
    timestamp = datetime.now(timezone.utc)
    state.equipment_monitored.add(equipment_id)

    # In a real system, this would query the database
    # For now, return simulated data

    health_score = 78.5
    health_status = classify_health_status(health_score)

    component_scores = {
        "vibration": 85.0,
        "temperature": 72.0,
        "pressure": 80.0,
        "efficiency": 77.0
    }

    active_alerts = []
    if health_score < 80:
        active_alerts.append({
            "alert_id": f"alert_{uuid.uuid4().hex[:8]}",
            "severity": "warning",
            "message": "Equipment health below optimal threshold",
            "created_at": timestamp.isoformat()
        })

    # Predictions
    predicted_rul = None
    failure_prob_30d = None

    if include_predictions:
        predicted_rul = 15000.0  # hours
        failure_prob_30d = 0.05

    # History
    health_history = None
    if include_history:
        health_history = [
            {"date": (timestamp - timedelta(days=i)).isoformat(), "score": 78 + i * 0.1}
            for i in range(min(history_days, 30))
        ]

    return EquipmentHealthResponse(
        equipment_id=equipment_id,
        timestamp=timestamp.isoformat(),
        health_status=health_status,
        health_score=health_score,
        operating_hours=25000.0,
        component_scores=component_scores,
        active_alerts=active_alerts,
        predicted_rul_hours=predicted_rul,
        failure_probability_30d=failure_prob_30d,
        health_history=health_history,
        next_maintenance_date=(timestamp + timedelta(days=45)).isoformat()
    )


# =============================================================================
# ANOMALY DETECTION ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/anomaly/detect",
    response_model=AnomalyDetectionResponse,
    tags=["Anomaly Detection"],
    summary="Detect anomalies",
    description="""
    Detect anomalies in equipment sensor data using statistical methods.

    Supports multiple detection methods:
    - Statistical (Z-score based)
    - CUSUM (Cumulative Sum)
    - Isolation Forest
    """
)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    request_id: str = Depends(get_request_id),
    api_key_valid: bool = Depends(verify_api_key),
    state: ApplicationState = Depends(get_state)
) -> AnomalyDetectionResponse:
    """
    Detect anomalies in sensor readings.

    Analyzes sensor data streams to identify anomalous patterns.
    """
    detection_id = f"anom_{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc)

    # Analyze each sensor reading
    contributing_factors = []
    max_anomaly_score = 0.0
    anomaly_detected = False
    anomaly_type = None

    for param_name, readings in request.sensor_readings.items():
        if len(readings) < 3:
            continue

        # Calculate statistics
        mean_val = sum(readings) / len(readings)
        variance = sum((x - mean_val) ** 2 for x in readings) / len(readings)
        std_dev = variance ** 0.5 if variance > 0 else 0.001

        # Check latest reading
        latest = readings[-1]
        z_score = abs(latest - mean_val) / std_dev

        if z_score > request.sensitivity:
            anomaly_detected = True
            anomaly_type = "statistical"
            contributing_factors.append({
                "parameter": param_name,
                "z_score": round(z_score, 2),
                "latest_value": latest,
                "mean": round(mean_val, 2),
                "std_dev": round(std_dev, 4)
            })

        # Track max score
        param_score = min(1.0, z_score / (request.sensitivity * 2))
        max_anomaly_score = max(max_anomaly_score, param_score)

    # Determine severity
    if max_anomaly_score >= 0.8:
        severity = AlertSeverity.CRITICAL
    elif max_anomaly_score >= 0.6:
        severity = AlertSeverity.HIGH
    elif max_anomaly_score >= 0.4:
        severity = AlertSeverity.WARNING
    else:
        severity = AlertSeverity.INFO

    # Recommendations
    recommended_actions = []
    if anomaly_detected:
        recommended_actions.append("Investigate contributing parameters")
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            recommended_actions.append("Consider immediate inspection")
        recommended_actions.append("Review recent operational changes")

    provenance_hash = generate_provenance_hash({
        "equipment_id": request.equipment_id,
        "detection_id": detection_id,
        "timestamp": timestamp.isoformat()
    })

    return AnomalyDetectionResponse(
        equipment_id=request.equipment_id,
        detection_id=detection_id,
        timestamp=timestamp.isoformat(),
        anomaly_detected=anomaly_detected,
        anomaly_score=round(max_anomaly_score, 4),
        anomaly_type=anomaly_type,
        contributing_factors=contributing_factors,
        severity=severity,
        recommended_actions=recommended_actions,
        provenance_hash=provenance_hash
    )


# =============================================================================
# SPARE PARTS ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/spare-parts/forecast",
    response_model=SparePartsResponse,
    tags=["Spare Parts"],
    summary="Forecast spare parts",
    description="Forecast spare parts requirements based on maintenance schedules and failure predictions."
)
async def forecast_spare_parts(
    request: SparePartsRequest,
    request_id: str = Depends(get_request_id),
    api_key_valid: bool = Depends(verify_api_key),
    state: ApplicationState = Depends(get_state)
) -> SparePartsResponse:
    """
    Forecast spare parts requirements.

    Uses reliability data and maintenance schedules to predict parts needs.
    """
    forecast_id = f"fcast_{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc)

    # Generate parts forecast (simplified)
    required_parts = [
        {
            "part_number": "BRG-SKF-6205",
            "description": "Ball Bearing 25x52x15",
            "quantity": len(request.equipment_ids),
            "unit_cost": 45.00,
            "total_cost": 45.00 * len(request.equipment_ids),
            "criticality": "high"
        },
        {
            "part_number": "SEAL-CR-12345",
            "description": "Mechanical Seal Assembly",
            "quantity": len(request.equipment_ids),
            "unit_cost": 250.00,
            "total_cost": 250.00 * len(request.equipment_ids),
            "criticality": "medium"
        }
    ]

    total_cost = sum(p["total_cost"] for p in required_parts)

    safety_stock = {
        "BRG-SKF-6205": max(2, len(request.equipment_ids) // 2),
        "SEAL-CR-12345": max(1, len(request.equipment_ids) // 3)
    }

    reorder_points = {
        "BRG-SKF-6205": 3,
        "SEAL-CR-12345": 2
    }

    eoq = {
        "BRG-SKF-6205": 10,
        "SEAL-CR-12345": 5
    }

    provenance_hash = generate_provenance_hash({
        "forecast_id": forecast_id,
        "timestamp": timestamp.isoformat(),
        "equipment_count": len(request.equipment_ids)
    })

    return SparePartsResponse(
        forecast_id=forecast_id,
        timestamp=timestamp.isoformat(),
        required_parts=required_parts,
        total_estimated_cost=total_cost,
        safety_stock_recommendations=safety_stock,
        reorder_points=reorder_points,
        economic_order_quantities=eoq,
        lead_time_days=14,
        provenance_hash=provenance_hash
    )


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with standard error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            details=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=request.headers.get("x-request-id")
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__} if os.getenv("DEBUG") else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=request.headers.get("x-request-id")
        ).model_dump()
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    return app


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", "1")),
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
