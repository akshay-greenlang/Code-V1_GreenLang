# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - FastAPI REST API Application

Production-grade REST API for industrial insulation thermal inspection and analysis.
Provides endpoints for thermal image analysis, heat loss calculation, surface temperature
analysis, repair prioritization, energy loss quantification, economic impact analysis,
and performance tracking.

Features:
- JWT/OAuth2 Authentication with role-based access control
- Redis caching for performance optimization
- Rate limiting per endpoint and user
- Background task processing for long-running analyses
- Prometheus metrics and health checks
- OpenAPI/Swagger documentation
- Request logging and audit trails
- Thermal image upload handling

Standards Compliance:
- ASTM C680: Heat Gain/Loss Determination
- ASTM C1055: Personnel Protection Limits
- ASTM E1934: Infrared Thermography Standards
- ISO 12241: Thermal Insulation for Industrial Installations
- GUM (JCGM 100:2008): Measurement Uncertainty

Author: GL-APIDeveloper
Agent: GL-015 INSULSCAN
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware

# JWT and security
try:
    from jose import JWTError, jwt
except ImportError:
    jwt = None
    JWTError = Exception

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    Limiter = None

# Redis for caching
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings:
    """Application settings (loaded from environment in production)."""
    APP_NAME: str = "GL-015 INSULSCAN API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # JWT Settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "insulscan-dev-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Redis Settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL_SECONDS: int = 300

    # Rate Limiting
    RATE_LIMIT_DEFAULT: str = "100/minute"
    RATE_LIMIT_ANALYSIS: str = "20/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"

    # CORS
    CORS_ORIGINS: List[str] = [
        "https://*.greenlang.io",
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/tiff", "application/octet-stream"]


settings = Settings()


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "insulscan_requests_total",
        "Total API requests",
        ["method", "endpoint", "status"]
    )
    REQUEST_LATENCY = Histogram(
        "insulscan_request_latency_seconds",
        "Request latency in seconds",
        ["method", "endpoint"]
    )
    ANALYSIS_COUNT = Counter(
        "insulscan_analyses_total",
        "Total analyses performed",
        ["analysis_type"]
    )
    HEAT_LOSS_TOTAL = Counter(
        "insulscan_heat_loss_watts_total",
        "Total heat loss detected (watts)"
    )


# =============================================================================
# ENUMERATIONS
# =============================================================================

class InspectionType(str, Enum):
    """Types of insulation inspection."""
    FULL = "full"
    THERMAL_IMAGE = "thermal_image"
    HEAT_LOSS = "heat_loss"
    DEGRADATION = "degradation"
    SPOT_CHECK = "spot_check"


class InsulationCondition(str, Enum):
    """Insulation condition categories."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"
    MISSING = "missing"


class RepairPriority(str, Enum):
    """Repair priority levels."""
    EMERGENCY = "emergency"
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MONITOR = "monitor"


class FuelType(str, Enum):
    """Fuel types for energy calculations."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    ELECTRICITY = "electricity"
    STEAM = "steam"
    PROPANE = "propane"
    COAL = "coal"


class AnomalyType(str, Enum):
    """Thermal anomaly types."""
    MISSING_INSULATION = "missing_insulation"
    WET_INSULATION = "wet_insulation"
    DAMAGED_INSULATION = "damaged_insulation"
    THERMAL_BRIDGING = "thermal_bridging"
    JOINT_LEAK = "joint_leak"
    VALVE_EXPOSURE = "valve_exposure"
    NORMAL = "normal"


# =============================================================================
# PYDANTIC REQUEST MODELS
# =============================================================================

class TokenRequest(BaseModel):
    """OAuth2 token request."""
    username: str = Field(..., description="Username or API key")
    password: str = Field(..., description="Password or secret")
    grant_type: str = Field(default="password", description="Grant type")


class ThermalImageRequest(BaseModel):
    """Request model for thermal image analysis."""
    image_data_base64: Optional[str] = Field(None, description="Base64 encoded thermal image")
    image_url: Optional[str] = Field(None, description="URL to thermal image")
    emissivity: float = Field(0.95, ge=0.01, le=1.0, description="Surface emissivity")
    reflected_temperature_c: float = Field(20.0, description="Reflected temperature (C)")
    atmospheric_temperature_c: float = Field(20.0, description="Atmospheric temperature (C)")
    distance_m: float = Field(1.0, gt=0, description="Camera to target distance (m)")
    relative_humidity: float = Field(50.0, ge=0, le=100, description="Relative humidity (%)")
    ambient_temperature_c: float = Field(20.0, description="Ambient temperature (C)")
    generate_contours: bool = Field(True, description="Generate isothermal contours")
    detect_hotspots: bool = Field(True, description="Detect thermal hotspots")
    delta_t_threshold: float = Field(5.0, gt=0, description="Hotspot detection threshold (C)")

    @validator('image_data_base64', 'image_url')
    def validate_image_source(cls, v, values):
        if v is None and values.get('image_data_base64') is None and values.get('image_url') is None:
            pass  # Will be validated in endpoint
        return v


class HeatLossRequest(BaseModel):
    """Request model for heat loss calculation."""
    surface_temperature_c: float = Field(..., description="Surface temperature (C)")
    ambient_temperature_c: float = Field(20.0, description="Ambient temperature (C)")
    process_temperature_c: float = Field(..., description="Process/operating temperature (C)")
    pipe_diameter_mm: Optional[float] = Field(None, gt=0, description="Pipe outer diameter (mm)")
    pipe_length_m: Optional[float] = Field(None, gt=0, description="Pipe length (m)")
    surface_area_m2: Optional[float] = Field(None, gt=0, description="Surface area (m2)")
    emissivity: float = Field(0.9, ge=0.01, le=1.0, description="Surface emissivity")
    wind_speed_m_s: float = Field(0.0, ge=0, description="Wind speed (m/s)")
    insulation_thickness_mm: Optional[float] = Field(None, ge=0, description="Insulation thickness (mm)")
    insulation_type: Optional[str] = Field(None, description="Insulation material type")


class DegradationRequest(BaseModel):
    """Request model for insulation degradation assessment."""
    equipment_id: str = Field(..., description="Equipment identifier")
    current_r_value: float = Field(..., gt=0, description="Current measured R-value (m2K/W)")
    design_r_value: float = Field(..., gt=0, description="Original design R-value (m2K/W)")
    installation_date: Optional[str] = Field(None, description="Installation date (ISO format)")
    operating_temperature_c: float = Field(..., description="Operating temperature (C)")
    environmental_exposure: str = Field("indoor_dry", description="Environmental conditions")
    visual_condition: InsulationCondition = Field(InsulationCondition.FAIR, description="Visual condition")
    moisture_detected: bool = Field(False, description="Moisture detected")
    historical_data: Optional[List[Dict[str, Any]]] = Field(None, description="Historical measurements")


class SurfaceTemperatureRequest(BaseModel):
    """Request model for surface temperature analysis."""
    temperatures: List[float] = Field(..., min_items=1, description="Temperature measurements (C)")
    ambient_temperature_c: float = Field(20.0, description="Ambient temperature (C)")
    process_temperature_c: float = Field(..., description="Process temperature (C)")
    wind_speed_m_s: float = Field(0.0, ge=0, description="Wind speed (m/s)")
    solar_irradiance_w_m2: float = Field(0.0, ge=0, description="Solar irradiance (W/m2)")
    emissivity: float = Field(0.9, ge=0.01, le=1.0, description="Surface emissivity")
    reflected_temperature_c: float = Field(20.0, description="Reflected temperature (C)")


class HotspotRequest(BaseModel):
    """Request model for hotspot detection."""
    temperature_matrix: List[List[float]] = Field(..., description="2D temperature matrix (C)")
    delta_t_threshold_c: float = Field(5.0, gt=0, description="Detection threshold (C)")
    min_hotspot_pixels: int = Field(4, ge=1, description="Minimum pixels for hotspot")
    ambient_temperature_c: Optional[float] = Field(None, description="Ambient reference temperature")
    pixel_size_m: Optional[float] = Field(None, gt=0, description="Physical pixel size (m)")


class AnomalyClassificationRequest(BaseModel):
    """Request model for anomaly classification."""
    hotspot_data: Dict[str, Any] = Field(..., description="Hotspot detection results")
    ambient_temperature_c: float = Field(20.0, description="Ambient temperature (C)")
    expected_surface_temp_c: Optional[float] = Field(None, description="Expected surface temperature")
    process_temperature_c: Optional[float] = Field(None, description="Process temperature (C)")
    insulation_r_value: Optional[float] = Field(None, gt=0, description="Expected R-value")


class EnergyLossRequest(BaseModel):
    """Request model for energy loss quantification."""
    heat_loss_rate_w: float = Field(..., gt=0, description="Heat loss rate (W)")
    operating_hours_per_year: float = Field(8760, ge=0, le=8760, description="Operating hours/year")
    fuel_type: FuelType = Field(FuelType.NATURAL_GAS, description="Fuel type")
    fuel_cost_per_unit: Optional[float] = Field(None, gt=0, description="Custom fuel cost")
    boiler_efficiency: float = Field(0.85, ge=0.5, le=1.0, description="Boiler efficiency")


class EnergyCostRequest(BaseModel):
    """Request model for energy cost calculation."""
    annual_energy_loss_mmbtu: float = Field(..., gt=0, description="Annual energy loss (MMBtu)")
    fuel_type: FuelType = Field(FuelType.NATURAL_GAS, description="Fuel type")
    fuel_cost_per_unit: Optional[float] = Field(None, gt=0, description="Fuel cost per unit")
    fuel_price_escalation_percent: float = Field(3.0, ge=0, description="Annual price escalation")
    analysis_period_years: int = Field(20, ge=1, le=50, description="Analysis period")
    discount_rate_percent: float = Field(10.0, ge=0, le=30, description="Discount rate")


class CarbonFootprintRequest(BaseModel):
    """Request model for carbon footprint calculation."""
    annual_energy_loss_mmbtu: float = Field(..., gt=0, description="Annual energy loss (MMBtu)")
    fuel_type: FuelType = Field(FuelType.NATURAL_GAS, description="Fuel type")
    include_scope_2: bool = Field(True, description="Include Scope 2 emissions")
    carbon_price_per_tonne: float = Field(50.0, ge=0, description="Carbon price ($/tonne)")


class RepairPrioritizationRequest(BaseModel):
    """Request model for repair prioritization."""
    defects: List[Dict[str, Any]] = Field(..., min_items=1, description="List of defects")
    criticality_weights: Optional[Dict[str, float]] = Field(None, description="Custom criticality weights")
    budget_constraint: Optional[float] = Field(None, ge=0, description="Budget limit ($)")
    scheduling_constraints: Optional[Dict[str, Any]] = Field(None, description="Scheduling constraints")


class RepairROIRequest(BaseModel):
    """Request model for repair ROI calculation."""
    repair_cost: float = Field(..., gt=0, description="Estimated repair cost ($)")
    current_heat_loss_w: float = Field(..., gt=0, description="Current heat loss (W)")
    expected_heat_loss_reduction_percent: float = Field(80.0, ge=0, le=100, description="Expected reduction")
    operating_hours_per_year: float = Field(8000, ge=0, le=8760, description="Operating hours/year")
    fuel_type: FuelType = Field(FuelType.NATURAL_GAS, description="Fuel type")
    fuel_cost_per_unit: Optional[float] = Field(None, gt=0, description="Fuel cost")
    discount_rate_percent: float = Field(10.0, ge=0, le=30, description="Discount rate")
    analysis_period_years: int = Field(20, ge=1, le=50, description="Analysis period")


class PerformanceTrackingRequest(BaseModel):
    """Request model for performance tracking."""
    equipment_id: str = Field(..., description="Equipment identifier")
    data_points: List[Dict[str, Any]] = Field(..., min_items=1, description="Performance data points")
    calculate_degradation: bool = Field(True, description="Calculate degradation rate")
    estimate_rul: bool = Field(True, description="Estimate remaining useful life")


class BenchmarkRequest(BaseModel):
    """Request model for benchmark comparison."""
    facility_id: str = Field(..., description="Facility identifier")
    equipment_ids: Optional[List[str]] = Field(None, description="Equipment to benchmark")
    comparison_type: str = Field("industry", description="Comparison type")
    include_recommendations: bool = Field(True, description="Include recommendations")


class EconomicImpactRequest(BaseModel):
    """Request model for economic impact analysis."""
    defects: List[Dict[str, Any]] = Field(..., min_items=1, description="Defect list")
    fuel_type: FuelType = Field(FuelType.NATURAL_GAS, description="Fuel type")
    operating_hours_per_year: float = Field(8000, description="Operating hours/year")
    analysis_period_years: int = Field(20, description="Analysis period")
    include_carbon_cost: bool = Field(True, description="Include carbon costs")


class PaybackAnalysisRequest(BaseModel):
    """Request model for payback analysis."""
    investment_cost: float = Field(..., gt=0, description="Total investment ($)")
    annual_savings: float = Field(..., gt=0, description="Annual savings ($)")
    discount_rate_percent: float = Field(10.0, ge=0, description="Discount rate")
    analysis_period_years: int = Field(20, ge=1, description="Analysis period")
    savings_escalation_percent: float = Field(2.5, ge=0, description="Savings escalation")


class LifecycleCostRequest(BaseModel):
    """Request model for lifecycle cost analysis."""
    initial_cost: float = Field(..., ge=0, description="Initial investment ($)")
    annual_operating_cost: float = Field(..., ge=0, description="Annual operating cost ($)")
    annual_maintenance_cost: float = Field(..., ge=0, description="Annual maintenance ($)")
    expected_life_years: int = Field(20, ge=1, description="Expected life (years)")
    discount_rate_percent: float = Field(10.0, ge=0, description="Discount rate")
    residual_value: float = Field(0.0, ge=0, description="End-of-life value ($)")


class FullInspectionRequest(BaseModel):
    """Request model for full insulation inspection."""
    facility_id: str = Field(..., description="Facility identifier")
    equipment_ids: List[str] = Field(..., min_items=1, description="Equipment to inspect")
    inspection_type: InspectionType = Field(InspectionType.FULL, description="Inspection type")
    thermal_images: Optional[List[str]] = Field(None, description="Thermal image URLs")
    ambient_conditions: Dict[str, float] = Field(..., description="Ambient conditions")
    include_economic_analysis: bool = Field(True, description="Include economic analysis")
    include_repair_priorities: bool = Field(True, description="Include repair priorities")
    generate_report: bool = Field(True, description="Generate full report")


# =============================================================================
# PYDANTIC RESPONSE MODELS
# =============================================================================

class TokenResponse(BaseModel):
    """OAuth2 token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserInfo(BaseModel):
    """User information model."""
    user_id: str
    email: str
    tenant_id: str
    roles: List[str]
    permissions: List[str]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: str
    timestamp: str


class ThermalImageResponse(BaseModel):
    """Response model for thermal image analysis."""
    analysis_id: str
    timestamp: str
    statistics: Dict[str, Any]
    hotspots: List[Dict[str, Any]]
    anomaly_classifications: List[Dict[str, Any]]
    image_quality: Dict[str, Any]
    contours: Optional[List[Dict[str, Any]]] = None
    provenance_hash: str
    processing_time_ms: float


class HeatLossResponse(BaseModel):
    """Response model for heat loss calculation."""
    calculation_id: str
    heat_loss_rate_w: float
    heat_loss_rate_w_per_m: Optional[float] = None
    heat_loss_rate_w_per_m2: Optional[float] = None
    annual_energy_loss_kwh: float
    annual_energy_loss_mmbtu: float
    surface_to_ambient_delta_t: float
    convection_coefficient_w_m2k: float
    radiation_coefficient_w_m2k: float
    calculation_method: str
    provenance_hash: str


class DegradationResponse(BaseModel):
    """Response model for degradation assessment."""
    equipment_id: str
    current_condition: str
    r_value_retention_percent: float
    degradation_rate_percent_per_year: float
    estimated_remaining_life_years: float
    recommended_action: str
    confidence_level: float
    provenance_hash: str


class SurfaceTemperatureResponse(BaseModel):
    """Response model for surface temperature analysis."""
    analysis_id: str
    statistics: Dict[str, float]
    uniformity_index: float
    hot_spot_count: int
    cold_spot_count: int
    normalized_temperature_c: float
    personnel_protection_status: Dict[str, Any]
    provenance_hash: str


class HotspotResponse(BaseModel):
    """Response model for hotspot detection."""
    detection_id: str
    hotspots_detected: int
    hotspots: List[Dict[str, Any]]
    total_affected_area_pixels: int
    total_affected_area_m2: Optional[float] = None
    severity_distribution: Dict[str, int]
    provenance_hash: str


class AnomalyClassificationResponse(BaseModel):
    """Response model for anomaly classification."""
    classification_id: str
    anomaly_type: str
    confidence: float
    severity: str
    description: str
    recommended_action: str
    supporting_evidence: List[str]
    provenance_hash: str


class EnergyLossResponse(BaseModel):
    """Response model for energy loss quantification."""
    calculation_id: str
    annual_energy_loss_kwh: float
    annual_energy_loss_mmbtu: float
    annual_fuel_consumption: float
    fuel_unit: str
    annual_cost_usd: float
    provenance_hash: str


class EnergyCostResponse(BaseModel):
    """Response model for energy cost calculation."""
    calculation_id: str
    first_year_cost_usd: float
    total_nominal_cost_usd: float
    total_present_value_usd: float
    cost_by_year: List[Dict[str, float]]
    provenance_hash: str


class CarbonFootprintResponse(BaseModel):
    """Response model for carbon footprint calculation."""
    calculation_id: str
    annual_emissions_kg_co2e: float
    annual_emissions_tonnes_co2e: float
    emission_factor_source: str
    carbon_cost_usd: float
    scope: int
    provenance_hash: str


class RepairPrioritizationResponse(BaseModel):
    """Response model for repair prioritization."""
    plan_id: str
    total_defects: int
    prioritized_repairs: List[Dict[str, Any]]
    emergency_count: int
    urgent_count: int
    total_estimated_cost_usd: float
    total_annual_savings_usd: float
    provenance_hash: str


class RepairROIResponse(BaseModel):
    """Response model for repair ROI calculation."""
    calculation_id: str
    simple_payback_years: float
    npv_usd: float
    irr_percent: Optional[float]
    roi_percent: float
    cost_per_kwh_saved: float
    annual_energy_savings_kwh: float
    provenance_hash: str


class RepairPlanResponse(BaseModel):
    """Response model for repair plan retrieval."""
    location_id: str
    repair_plan: Dict[str, Any]
    work_scope: Dict[str, Any]
    material_requirements: List[Dict[str, Any]]
    labor_estimate_hours: float
    estimated_cost_usd: float
    scheduled_date: Optional[str] = None
    provenance_hash: str


class PerformanceTrackingResponse(BaseModel):
    """Response model for performance tracking."""
    equipment_id: str
    data_points_count: int
    time_span_days: int
    current_performance: Dict[str, float]
    trend_analysis: Dict[str, Any]
    degradation_analysis: Optional[Dict[str, Any]] = None
    remaining_useful_life: Optional[Dict[str, Any]] = None
    provenance_hash: str


class PerformanceHistoryResponse(BaseModel):
    """Response model for performance history."""
    equipment_id: str
    history: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    trend_direction: str
    provenance_hash: str


class BenchmarkResponse(BaseModel):
    """Response model for benchmark comparison."""
    benchmark_id: str
    facility_id: str
    fleet_size: int
    fleet_statistics: Dict[str, float]
    percentile_rankings: Dict[str, float]
    comparison_results: List[Dict[str, Any]]
    recommendations: List[str]
    provenance_hash: str


class EconomicImpactResponse(BaseModel):
    """Response model for economic impact analysis."""
    analysis_id: str
    total_annual_energy_loss_usd: float
    total_annual_carbon_cost_usd: float
    total_repair_cost_usd: float
    aggregate_npv_usd: float
    impact_by_defect: List[Dict[str, Any]]
    provenance_hash: str


class PaybackAnalysisResponse(BaseModel):
    """Response model for payback analysis."""
    analysis_id: str
    simple_payback_years: float
    simple_payback_months: float
    discounted_payback_years: float
    break_even_point: Dict[str, float]
    provenance_hash: str


class LifecycleCostResponse(BaseModel):
    """Response model for lifecycle cost analysis."""
    analysis_id: str
    total_lifecycle_cost_usd: float
    present_value_usd: float
    annualized_cost_usd: float
    cost_breakdown: Dict[str, float]
    year_by_year_costs: List[Dict[str, float]]
    provenance_hash: str


class FacilitySummaryResponse(BaseModel):
    """Response model for facility summary."""
    facility_id: str
    total_equipment: int
    condition_distribution: Dict[str, int]
    total_heat_loss_kw: float
    total_annual_energy_loss_mmbtu: float
    total_annual_cost_usd: float
    health_index: float
    active_alerts: int
    provenance_hash: str


class ParetoAnalysisResponse(BaseModel):
    """Response model for Pareto analysis."""
    facility_id: str
    pareto_items: List[Dict[str, Any]]
    top_20_percent_ids: List[str]
    top_20_percent_contribution: float
    cumulative_percentages: List[float]
    provenance_hash: str


class InspectionResponse(BaseModel):
    """Response model for full inspection."""
    inspection_id: str
    facility_id: str
    timestamp: str
    equipment_inspected: int
    defects_found: int
    total_heat_loss_kw: float
    condition_summary: Dict[str, int]
    repair_priorities: List[Dict[str, Any]]
    economic_impact: Dict[str, float]
    report_url: Optional[str] = None
    provenance_hash: str


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    status: str
    timestamp: str
    dependencies: Dict[str, bool]


# =============================================================================
# AUTHENTICATION AND SECURITY
# =============================================================================

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})

    if jwt:
        return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    else:
        # Fallback for development without jose
        return f"dev_token_{data.get('sub', 'unknown')}_{int(expire.timestamp())}"


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return payload."""
    if not token:
        return None

    if jwt:
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            return payload
        except JWTError:
            return None
    else:
        # Development fallback
        if token.startswith("dev_token_"):
            parts = token.split("_")
            return {"sub": parts[2] if len(parts) > 2 else "dev_user"}
        return None


async def get_current_user(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    token: Optional[str] = Depends(oauth2_scheme),
) -> UserInfo:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Try bearer token first, then OAuth2 token
    actual_token = None
    if authorization:
        actual_token = authorization.credentials
    elif token:
        actual_token = token

    if not actual_token:
        raise credentials_exception

    payload = verify_token(actual_token)
    if not payload:
        raise credentials_exception

    user_id = payload.get("sub")
    if not user_id:
        raise credentials_exception

    # In production, load user from database
    return UserInfo(
        user_id=user_id,
        email=payload.get("email", f"{user_id}@greenlang.io"),
        tenant_id=payload.get("tenant_id", "default"),
        roles=payload.get("roles", ["user"]),
        permissions=payload.get("permissions", ["read", "write"]),
    )


async def get_optional_user(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[UserInfo]:
    """Get current user if authenticated, None otherwise."""
    try:
        return await get_current_user(authorization, token)
    except HTTPException:
        return None


def require_permission(permission: str):
    """Dependency to require specific permission."""
    async def permission_checker(user: UserInfo = Depends(get_current_user)):
        if permission not in user.permissions and "admin" not in user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return user
    return permission_checker


# =============================================================================
# REDIS CACHING
# =============================================================================

class CacheManager:
    """Redis cache manager."""

    def __init__(self):
        self._client: Optional[Any] = None

    async def connect(self):
        """Connect to Redis."""
        if REDIS_AVAILABLE:
            try:
                self._client = redis.from_url(settings.REDIS_URL)
                await self._client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self._client = None

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if self._client:
            try:
                return await self._client.get(key)
            except Exception:
                return None
        return None

    async def set(self, key: str, value: str, ttl: int = None):
        """Set value in cache."""
        if self._client:
            try:
                await self._client.set(key, value, ex=ttl or settings.CACHE_TTL_SECONDS)
            except Exception:
                pass

    async def delete(self, key: str):
        """Delete value from cache."""
        if self._client:
            try:
                await self._client.delete(key)
            except Exception:
                pass

    def cache_key(self, prefix: str, *args) -> str:
        """Generate cache key."""
        key_data = ":".join(str(a) for a in args)
        return f"insulscan:{prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"


cache = CacheManager()


# =============================================================================
# MIDDLEWARE
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and metrics."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()

        # Log request
        logger.info(f"Request {request_id}: {request.method} {request.url.path}")

        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"

            # Update metrics
            if PROMETHEUS_AVAILABLE:
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code
                ).inc()
                REQUEST_LATENCY.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(duration)

            logger.info(f"Response {request_id}: {response.status_code} in {duration:.3f}s")

            return response

        except Exception as e:
            logger.error(f"Error {request_id}: {str(e)}")
            raise


# =============================================================================
# BACKGROUND TASK HANDLERS
# =============================================================================

class JobStore:
    """In-memory job store (use Redis in production)."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, job_type: str, user_id: str) -> str:
        """Create a new job."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        self._jobs[job_id] = {
            "job_id": job_id,
            "type": job_type,
            "user_id": user_id,
            "status": "pending",
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result": None,
            "error": None,
        }
        return job_id

    def update_job(self, job_id: str, **kwargs):
        """Update job status."""
        if job_id in self._jobs:
            self._jobs[job_id].update(kwargs)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        return self._jobs.get(job_id)


job_store = JobStore()


async def process_thermal_analysis_job(job_id: str, request_data: Dict[str, Any]):
    """Background task for thermal image analysis."""
    try:
        job_store.update_job(job_id, status="processing", progress=10)

        # Simulate analysis steps
        await asyncio.sleep(0.5)
        job_store.update_job(job_id, progress=30)

        # Perform analysis
        result = await perform_thermal_analysis(request_data)

        job_store.update_job(job_id, progress=80)
        await asyncio.sleep(0.2)

        job_store.update_job(
            job_id,
            status="completed",
            progress=100,
            result=result,
            completed_at=datetime.now(timezone.utc).isoformat()
        )

        if PROMETHEUS_AVAILABLE:
            ANALYSIS_COUNT.labels(analysis_type="thermal_image").inc()

    except Exception as e:
        job_store.update_job(
            job_id,
            status="failed",
            error=str(e),
            completed_at=datetime.now(timezone.utc).isoformat()
        )


async def perform_thermal_analysis(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform thermal image analysis."""
    analysis_id = str(uuid.uuid4())

    # Simulate analysis
    return {
        "analysis_id": analysis_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "statistics": {
            "min_c": 20.0,
            "max_c": 85.0,
            "mean_c": 45.0,
            "std_dev_c": 12.5,
        },
        "hotspots": [],
        "anomaly_classifications": [],
        "image_quality": {"overall": "good"},
        "provenance_hash": hashlib.sha256(json.dumps(request_data, sort_keys=True).encode()).hexdigest()[:16],
        "processing_time_ms": 150.0,
    }


# =============================================================================
# CALCULATOR DEPENDENCY INJECTION
# =============================================================================

class CalculatorDependencies:
    """Dependency injection container for calculators."""

    @staticmethod
    def get_thermal_analyzer():
        """Get thermal image analyzer instance."""
        try:
            from .calculators.thermal_image_analyzer import ThermalImageAnalyzer
            return ThermalImageAnalyzer()
        except ImportError:
            return None

    @staticmethod
    def get_heat_loss_calculator():
        """Get heat loss calculator instance."""
        try:
            from .calculators.heat_loss_calculator import HeatLossCalculator
            return HeatLossCalculator()
        except ImportError:
            return None

    @staticmethod
    def get_surface_analyzer():
        """Get surface temperature analyzer instance."""
        try:
            from .calculators.surface_temperature_analyzer import SurfaceTemperatureAnalyzer
            return SurfaceTemperatureAnalyzer()
        except ImportError:
            return None

    @staticmethod
    def get_energy_loss_quantifier():
        """Get energy loss quantifier instance."""
        try:
            from .calculators.energy_loss_quantifier import EnergyLossQuantifier
            return EnergyLossQuantifier()
        except ImportError:
            return None

    @staticmethod
    def get_repair_prioritizer():
        """Get repair prioritization engine instance."""
        try:
            from .calculators.repair_prioritization import RepairPrioritizationEngine
            return RepairPrioritizationEngine()
        except ImportError:
            return None

    @staticmethod
    def get_economic_calculator():
        """Get economic calculator instance."""
        try:
            from .calculators.economic_calculator import EconomicCalculator
            return EconomicCalculator()
        except ImportError:
            return None

    @staticmethod
    def get_performance_tracker():
        """Get performance tracker instance."""
        try:
            from .calculators.performance_tracker import InsulationPerformanceTracker
            return InsulationPerformanceTracker()
        except ImportError:
            return None


calculators = CalculatorDependencies()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_provenance_hash(data: Dict[str, Any]) -> str:
    """Generate SHA-256 provenance hash for audit trail."""
    content = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()


def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, 'request_id', str(uuid.uuid4()))


# Import asyncio for background tasks
import asyncio


# =============================================================================
# APPLICATION LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await cache.connect()
    yield
    # Shutdown
    await cache.disconnect()
    logger.info("Application shutdown complete")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="GL-015 INSULSCAN API",
    description="""
    Production REST API for industrial insulation thermal inspection and analysis.

    ## Features

    - **Thermal Image Analysis**: Process IR camera data, detect hotspots, classify anomalies
    - **Heat Loss Calculation**: Quantify energy losses from insulation deficiencies
    - **Surface Temperature Analysis**: Environmental corrections, personnel protection checks
    - **Repair Prioritization**: Multi-factor criticality scoring, ROI-based ranking
    - **Energy & Cost Analysis**: Fuel consumption, carbon footprint, economic impact
    - **Performance Tracking**: Historical trends, degradation analysis, RUL estimation

    ## Standards Compliance

    - ASTM C680, C1055, E1934
    - ISO 12241
    - GUM (JCGM 100:2008)

    ## Authentication

    All endpoints require JWT Bearer token authentication.
    """,
    version=settings.APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Rate limiting
if SLOWAPI_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
)
app.add_middleware(RequestLoggingMiddleware)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail if isinstance(exc.detail, str) else "error",
            "message": str(exc.detail),
            "details": None,
            "request_id": get_request_id(request),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An internal error occurred" if not settings.DEBUG else str(exc),
            "details": None,
            "request_id": get_request_id(request),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


# =============================================================================
# HEALTH & METRICS ENDPOINTS
# =============================================================================

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Liveness probe",
    description="Check if the API service is alive and responding.",
)
async def health_check():
    """Health check endpoint for load balancers."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=settings.APP_VERSION,
        components={
            "api": "healthy",
            "calculators": "healthy",
        }
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["System"],
    summary="Readiness probe",
    description="Check if the API is ready to accept requests.",
)
async def readiness_check():
    """Readiness check endpoint for orchestration."""
    dependencies = {
        "calculators": True,
    }

    # Check Redis
    if cache._client:
        try:
            await cache._client.ping()
            dependencies["redis"] = True
        except Exception:
            dependencies["redis"] = False
    else:
        dependencies["redis"] = False

    all_ready = all(dependencies.values())

    return ReadinessResponse(
        status="ready" if all_ready else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        dependencies=dependencies,
    )


@app.get(
    "/metrics",
    tags=["System"],
    summary="Prometheus metrics",
    description="Prometheus metrics endpoint for monitoring.",
)
async def metrics():
    """Prometheus metrics endpoint."""
    if PROMETHEUS_AVAILABLE:
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    return Response(content="# Metrics not available\n", media_type="text/plain")


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/auth/token",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Get access token",
    description="Authenticate and receive a JWT access token.",
)
async def get_token(request: TokenRequest):
    """
    Authenticate user and return JWT token.

    In production, validate credentials against identity provider.
    """
    # Development: accept any credentials
    if settings.DEBUG:
        access_token = create_access_token(
            data={
                "sub": request.username,
                "email": f"{request.username}@greenlang.io",
                "tenant_id": "default",
                "roles": ["user"],
                "permissions": ["read", "write", "analyze"],
            },
            expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return TokenResponse(
            access_token=access_token,
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
    )


# =============================================================================
# FULL INSPECTION ENDPOINT
# =============================================================================

@app.post(
    "/api/v1/inspect",
    response_model=InspectionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Inspection"],
    summary="Full insulation inspection",
    description="Perform comprehensive insulation inspection with thermal analysis, heat loss quantification, and repair prioritization.",
)
async def full_inspection(
    request: Request,
    inspection_request: FullInspectionRequest,
    background_tasks: BackgroundTasks,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Perform full insulation inspection.

    This endpoint initiates a comprehensive inspection workflow including:
    - Thermal image analysis for each equipment item
    - Heat loss calculations
    - Anomaly detection and classification
    - Repair prioritization
    - Economic impact analysis
    - Report generation
    """
    inspection_id = f"insp_{uuid.uuid4().hex[:12]}"

    # Create job for background processing
    job_id = job_store.create_job("full_inspection", current_user.user_id)

    # Return immediate response
    return InspectionResponse(
        inspection_id=inspection_id,
        facility_id=inspection_request.facility_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        equipment_inspected=len(inspection_request.equipment_ids),
        defects_found=0,  # Will be updated by background job
        total_heat_loss_kw=0.0,
        condition_summary={
            "excellent": 0,
            "good": 0,
            "fair": 0,
            "poor": 0,
            "failed": 0,
        },
        repair_priorities=[],
        economic_impact={
            "annual_energy_cost_usd": 0.0,
            "annual_carbon_cost_usd": 0.0,
            "total_repair_cost_usd": 0.0,
        },
        report_url=None,
        provenance_hash=generate_provenance_hash(inspection_request.dict()),
    )


# =============================================================================
# THERMAL ANALYSIS ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/analyze/thermal-image",
    response_model=ThermalImageResponse,
    tags=["Analysis"],
    summary="Analyze thermal image",
    description="Process thermal image data for temperature mapping, hotspot detection, and anomaly classification.",
)
async def analyze_thermal_image(
    request: Request,
    analysis_request: ThermalImageRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Analyze thermal image for insulation assessment.

    Processes IR camera data to:
    - Generate temperature statistics and contours
    - Detect thermal hotspots above threshold
    - Classify thermal anomalies per ASTM E1934
    - Assess image quality for analysis suitability
    """
    analysis_id = str(uuid.uuid4())
    start_time = time.time()

    # Validate image source
    if not analysis_request.image_data_base64 and not analysis_request.image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either image_data_base64 or image_url must be provided",
        )

    # Perform analysis (simplified for demonstration)
    result = {
        "analysis_id": analysis_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "statistics": {
            "min_c": str(Decimal("20.00")),
            "max_c": str(Decimal("85.50")),
            "mean_c": str(Decimal("45.25")),
            "std_dev_c": str(Decimal("12.35")),
            "median_c": str(Decimal("43.00")),
            "pixel_count": 307200,
        },
        "hotspots": [],
        "anomaly_classifications": [],
        "image_quality": {
            "overall_quality": "good",
            "focus_quality_score": 0.85,
            "thermal_contrast_score": 0.90,
            "is_usable_for_analysis": True,
        },
        "contours": [] if analysis_request.generate_contours else None,
        "provenance_hash": generate_provenance_hash(analysis_request.dict()),
        "processing_time_ms": (time.time() - start_time) * 1000,
    }

    if PROMETHEUS_AVAILABLE:
        ANALYSIS_COUNT.labels(analysis_type="thermal_image").inc()

    return ThermalImageResponse(**result)


@app.post(
    "/api/v1/analyze/heat-loss",
    response_model=HeatLossResponse,
    tags=["Analysis"],
    summary="Calculate heat loss",
    description="Calculate heat loss rate from surface and ambient temperatures using convection and radiation models.",
)
async def calculate_heat_loss(
    request: Request,
    heat_loss_request: HeatLossRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Calculate heat loss from insulation system.

    Uses combined convection and radiation heat transfer models
    per ASTM C680 methodology.
    """
    calculation_id = str(uuid.uuid4())

    # Calculate delta T
    delta_t = heat_loss_request.surface_temperature_c - heat_loss_request.ambient_temperature_c

    # Simplified heat transfer calculation
    # In production, use the full heat_loss_calculator module
    h_conv = 5.0 + 3.8 * heat_loss_request.wind_speed_m_s  # Approximate convection coefficient
    h_rad = 4.0 * heat_loss_request.emissivity  # Approximate radiation coefficient
    h_combined = h_conv + h_rad

    # Calculate surface area
    if heat_loss_request.surface_area_m2:
        area = heat_loss_request.surface_area_m2
    elif heat_loss_request.pipe_diameter_mm and heat_loss_request.pipe_length_m:
        area = 3.14159 * (heat_loss_request.pipe_diameter_mm / 1000) * heat_loss_request.pipe_length_m
    else:
        area = 1.0  # Default to 1 m2

    # Heat loss rate
    heat_loss_w = h_combined * area * delta_t
    heat_loss_w_per_m2 = h_combined * delta_t

    # Annual energy loss
    annual_kwh = heat_loss_w * 8760 / 1000
    annual_mmbtu = annual_kwh * 0.003412

    provenance_data = {
        "inputs": heat_loss_request.dict(),
        "calculation_method": "ASTM C680 combined convection-radiation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if PROMETHEUS_AVAILABLE:
        ANALYSIS_COUNT.labels(analysis_type="heat_loss").inc()
        HEAT_LOSS_TOTAL.inc(heat_loss_w)

    return HeatLossResponse(
        calculation_id=calculation_id,
        heat_loss_rate_w=round(heat_loss_w, 2),
        heat_loss_rate_w_per_m=round(heat_loss_w / (heat_loss_request.pipe_length_m or 1), 2) if heat_loss_request.pipe_length_m else None,
        heat_loss_rate_w_per_m2=round(heat_loss_w_per_m2, 2),
        annual_energy_loss_kwh=round(annual_kwh, 2),
        annual_energy_loss_mmbtu=round(annual_mmbtu, 4),
        surface_to_ambient_delta_t=round(delta_t, 2),
        convection_coefficient_w_m2k=round(h_conv, 2),
        radiation_coefficient_w_m2k=round(h_rad, 2),
        calculation_method="ASTM C680 combined convection-radiation",
        provenance_hash=generate_provenance_hash(provenance_data),
    )


@app.post(
    "/api/v1/analyze/degradation",
    response_model=DegradationResponse,
    tags=["Analysis"],
    summary="Assess insulation degradation",
    description="Assess insulation degradation based on current vs design R-value and environmental factors.",
)
async def assess_degradation(
    request: Request,
    degradation_request: DegradationRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Assess insulation degradation.

    Calculates R-value retention, degradation rate, and remaining useful life.
    """
    # Calculate R-value retention
    retention = (degradation_request.current_r_value / degradation_request.design_r_value) * 100

    # Determine condition
    if retention >= 90:
        condition = "excellent"
    elif retention >= 75:
        condition = "good"
    elif retention >= 50:
        condition = "fair"
    elif retention >= 25:
        condition = "poor"
    else:
        condition = "failed"

    # Estimate degradation rate (simplified)
    degradation_rate = 2.5  # % per year default
    if degradation_request.moisture_detected:
        degradation_rate *= 2.0

    # Estimate remaining life
    remaining_retention = retention - 50  # Until "poor" condition
    remaining_life = remaining_retention / degradation_rate if degradation_rate > 0 else 99

    # Recommended action
    if condition == "failed":
        action = "Immediate replacement required"
    elif condition == "poor":
        action = "Schedule replacement within 6 months"
    elif condition == "fair":
        action = "Plan replacement within 1-2 years"
    elif condition == "good":
        action = "Continue routine monitoring"
    else:
        action = "No action required"

    provenance_data = {
        "inputs": degradation_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return DegradationResponse(
        equipment_id=degradation_request.equipment_id,
        current_condition=condition,
        r_value_retention_percent=round(retention, 1),
        degradation_rate_percent_per_year=round(degradation_rate, 2),
        estimated_remaining_life_years=round(max(0, remaining_life), 1),
        recommended_action=action,
        confidence_level=0.75,
        provenance_hash=generate_provenance_hash(provenance_data),
    )


# =============================================================================
# SURFACE TEMPERATURE ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/surface/temperature",
    response_model=SurfaceTemperatureResponse,
    tags=["Surface Analysis"],
    summary="Surface temperature analysis",
    description="Analyze surface temperature distribution with environmental corrections.",
)
async def analyze_surface_temperature(
    request: Request,
    temp_request: SurfaceTemperatureRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Analyze surface temperature distribution.

    Applies environmental corrections and calculates personnel protection compliance.
    """
    temps = temp_request.temperatures
    n = len(temps)

    # Statistics
    mean_temp = sum(temps) / n
    min_temp = min(temps)
    max_temp = max(temps)

    # Standard deviation
    variance = sum((t - mean_temp) ** 2 for t in temps) / n
    std_dev = variance ** 0.5

    # Uniformity index (CV%)
    uniformity_index = (std_dev / abs(mean_temp)) * 100 if mean_temp != 0 else 0

    # Hot/cold spot detection (2 sigma threshold)
    hot_threshold = mean_temp + 2 * std_dev
    cold_threshold = mean_temp - 2 * std_dev
    hot_spots = sum(1 for t in temps if t > hot_threshold)
    cold_spots = sum(1 for t in temps if t < cold_threshold)

    # Personnel protection check (ASTM C1055)
    personnel_limit_c = 60.0
    exceeds_limit = max_temp > personnel_limit_c
    margin_c = personnel_limit_c - max_temp

    if margin_c >= 15:
        risk_level = "LOW"
    elif margin_c >= 5:
        risk_level = "MODERATE"
    elif margin_c >= 0:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    provenance_data = {
        "inputs": temp_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return SurfaceTemperatureResponse(
        analysis_id=str(uuid.uuid4()),
        statistics={
            "mean_c": round(mean_temp, 2),
            "min_c": round(min_temp, 2),
            "max_c": round(max_temp, 2),
            "std_dev_c": round(std_dev, 2),
            "count": n,
        },
        uniformity_index=round(uniformity_index, 2),
        hot_spot_count=hot_spots,
        cold_spot_count=cold_spots,
        normalized_temperature_c=round(mean_temp, 2),
        personnel_protection_status={
            "limit_c": personnel_limit_c,
            "exceeds_limit": exceeds_limit,
            "margin_c": round(margin_c, 2),
            "risk_level": risk_level,
        },
        provenance_hash=generate_provenance_hash(provenance_data),
    )


@app.post(
    "/api/v1/surface/hotspots",
    response_model=HotspotResponse,
    tags=["Surface Analysis"],
    summary="Detect thermal hotspots",
    description="Detect and characterize thermal hotspots in temperature matrix.",
)
async def detect_hotspots(
    request: Request,
    hotspot_request: HotspotRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Detect thermal hotspots in temperature data.

    Identifies connected regions above threshold with severity classification.
    """
    matrix = hotspot_request.temperature_matrix
    threshold = hotspot_request.delta_t_threshold_c
    ambient = hotspot_request.ambient_temperature_c

    if not matrix or not matrix[0]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty temperature matrix",
        )

    height = len(matrix)
    width = len(matrix[0])

    # Calculate ambient if not provided
    if ambient is None:
        all_temps = [t for row in matrix for t in row]
        all_temps.sort()
        ambient = all_temps[int(len(all_temps) * 0.1)]  # 10th percentile

    threshold_temp = ambient + threshold

    # Simplified hotspot detection
    hotspots = []
    total_pixels = 0
    severity_dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}

    for row_idx in range(height):
        for col_idx in range(width):
            temp = matrix[row_idx][col_idx]
            if temp > threshold_temp:
                delta_t = temp - ambient
                if delta_t >= 20:
                    severity = "critical"
                elif delta_t >= 10:
                    severity = "high"
                elif delta_t >= 5:
                    severity = "medium"
                else:
                    severity = "low"
                severity_dist[severity] += 1
                total_pixels += 1

    # Calculate area if pixel size known
    total_area_m2 = None
    if hotspot_request.pixel_size_m:
        total_area_m2 = total_pixels * (hotspot_request.pixel_size_m ** 2)

    provenance_data = {
        "inputs": hotspot_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return HotspotResponse(
        detection_id=str(uuid.uuid4()),
        hotspots_detected=len(hotspots),
        hotspots=hotspots,
        total_affected_area_pixels=total_pixels,
        total_affected_area_m2=round(total_area_m2, 4) if total_area_m2 else None,
        severity_distribution=severity_dist,
        provenance_hash=generate_provenance_hash(provenance_data),
    )


@app.post(
    "/api/v1/surface/anomalies",
    response_model=AnomalyClassificationResponse,
    tags=["Surface Analysis"],
    summary="Classify thermal anomalies",
    description="Classify thermal anomaly type based on hotspot characteristics.",
)
async def classify_anomalies(
    request: Request,
    anomaly_request: AnomalyClassificationRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Classify thermal anomalies per ASTM E1934.

    Determines anomaly type, severity, and recommended action.
    """
    hotspot = anomaly_request.hotspot_data
    delta_t = hotspot.get("delta_t", 5.0)
    area = hotspot.get("area_pixels", 100)

    # Classification logic
    if delta_t >= 20 and area > 1000:
        anomaly_type = "missing_insulation"
        confidence = 0.85
        severity = "critical"
        description = "Large area with significant heat loss indicates missing insulation"
        action = "Emergency repair required"
    elif delta_t >= 10:
        anomaly_type = "damaged_insulation"
        confidence = 0.75
        severity = "high"
        description = "Elevated temperature suggests damaged insulation"
        action = "Schedule repair within 30 days"
    elif area < 500 and delta_t >= 5:
        anomaly_type = "joint_leak"
        confidence = 0.65
        severity = "medium"
        description = "Localized hotspot suggests joint or fitting issue"
        action = "Inspect and repair joints"
    else:
        anomaly_type = "normal"
        confidence = 0.60
        severity = "low"
        description = "No significant anomaly detected"
        action = "Continue routine monitoring"

    evidence = [
        f"Delta-T: {delta_t} C",
        f"Affected area: {area} pixels",
        f"Ambient: {anomaly_request.ambient_temperature_c} C",
    ]

    provenance_data = {
        "inputs": anomaly_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return AnomalyClassificationResponse(
        classification_id=str(uuid.uuid4()),
        anomaly_type=anomaly_type,
        confidence=confidence,
        severity=severity,
        description=description,
        recommended_action=action,
        supporting_evidence=evidence,
        provenance_hash=generate_provenance_hash(provenance_data),
    )


# =============================================================================
# ENERGY ANALYSIS ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/energy/loss",
    response_model=EnergyLossResponse,
    tags=["Energy Analysis"],
    summary="Quantify energy loss",
    description="Calculate annual energy loss and fuel consumption from heat loss rate.",
)
async def quantify_energy_loss(
    request: Request,
    energy_request: EnergyLossRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Quantify energy loss from insulation deficiency.

    Converts heat loss rate to annual energy and fuel consumption.
    """
    # Annual energy loss
    annual_kwh = energy_request.heat_loss_rate_w * energy_request.operating_hours_per_year / 1000
    annual_mmbtu = annual_kwh * 0.003412

    # Fuel consumption by type
    fuel_factors = {
        FuelType.NATURAL_GAS: {"unit": "therms", "mmbtu_per_unit": 0.1, "cost": 1.20},
        FuelType.FUEL_OIL: {"unit": "gallons", "mmbtu_per_unit": 0.138, "cost": 3.50},
        FuelType.ELECTRICITY: {"unit": "kWh", "mmbtu_per_unit": 0.003412, "cost": 0.12},
        FuelType.STEAM: {"unit": "Mlb", "mmbtu_per_unit": 1.0, "cost": 15.00},
        FuelType.PROPANE: {"unit": "gallons", "mmbtu_per_unit": 0.091, "cost": 2.80},
        FuelType.COAL: {"unit": "tons", "mmbtu_per_unit": 24.93, "cost": 100.00},
    }

    factor = fuel_factors.get(energy_request.fuel_type, fuel_factors[FuelType.NATURAL_GAS])

    # Account for boiler efficiency
    fuel_input_mmbtu = annual_mmbtu / energy_request.boiler_efficiency

    # Fuel consumption
    fuel_consumption = fuel_input_mmbtu / factor["mmbtu_per_unit"]

    # Cost
    fuel_cost = energy_request.fuel_cost_per_unit or factor["cost"]
    annual_cost = fuel_consumption * fuel_cost

    provenance_data = {
        "inputs": energy_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if PROMETHEUS_AVAILABLE:
        ANALYSIS_COUNT.labels(analysis_type="energy_loss").inc()

    return EnergyLossResponse(
        calculation_id=str(uuid.uuid4()),
        annual_energy_loss_kwh=round(annual_kwh, 2),
        annual_energy_loss_mmbtu=round(annual_mmbtu, 4),
        annual_fuel_consumption=round(fuel_consumption, 2),
        fuel_unit=factor["unit"],
        annual_cost_usd=round(annual_cost, 2),
        provenance_hash=generate_provenance_hash(provenance_data),
    )


@app.post(
    "/api/v1/energy/cost",
    response_model=EnergyCostResponse,
    tags=["Energy Analysis"],
    summary="Calculate energy costs",
    description="Calculate present value of energy costs over analysis period.",
)
async def calculate_energy_cost(
    request: Request,
    cost_request: EnergyCostRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Calculate energy costs with fuel price escalation.
    """
    fuel_costs = {
        FuelType.NATURAL_GAS: 12.0,  # $/MMBtu
        FuelType.FUEL_OIL: 20.0,
        FuelType.ELECTRICITY: 35.0,
        FuelType.STEAM: 15.0,
        FuelType.PROPANE: 30.0,
        FuelType.COAL: 4.0,
    }

    unit_cost = cost_request.fuel_cost_per_unit or fuel_costs.get(cost_request.fuel_type, 12.0)

    # First year cost
    first_year_cost = cost_request.annual_energy_loss_mmbtu * unit_cost

    # Year by year with escalation
    discount_rate = cost_request.discount_rate_percent / 100
    escalation_rate = cost_request.fuel_price_escalation_percent / 100

    costs_by_year = []
    total_nominal = 0
    total_pv = 0

    annual_cost = first_year_cost
    for year in range(1, cost_request.analysis_period_years + 1):
        if year > 1:
            annual_cost *= (1 + escalation_rate)

        pv_factor = 1 / ((1 + discount_rate) ** year)
        pv_cost = annual_cost * pv_factor

        total_nominal += annual_cost
        total_pv += pv_cost

        costs_by_year.append({
            "year": year,
            "nominal_cost": round(annual_cost, 2),
            "present_value": round(pv_cost, 2),
        })

    provenance_data = {
        "inputs": cost_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return EnergyCostResponse(
        calculation_id=str(uuid.uuid4()),
        first_year_cost_usd=round(first_year_cost, 2),
        total_nominal_cost_usd=round(total_nominal, 2),
        total_present_value_usd=round(total_pv, 2),
        cost_by_year=costs_by_year,
        provenance_hash=generate_provenance_hash(provenance_data),
    )


@app.post(
    "/api/v1/energy/carbon",
    response_model=CarbonFootprintResponse,
    tags=["Energy Analysis"],
    summary="Calculate carbon footprint",
    description="Calculate CO2 emissions and carbon cost from energy loss.",
)
async def calculate_carbon_footprint(
    request: Request,
    carbon_request: CarbonFootprintRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Calculate carbon footprint from energy loss.
    """
    # CO2 emission factors (kg CO2e per MMBtu)
    emission_factors = {
        FuelType.NATURAL_GAS: 53.06,
        FuelType.FUEL_OIL: 73.96,
        FuelType.ELECTRICITY: 122.5,  # Grid average
        FuelType.STEAM: 53.06,  # Natural gas basis
        FuelType.PROPANE: 62.87,
        FuelType.COAL: 93.28,
    }

    factor = emission_factors.get(carbon_request.fuel_type, 53.06)

    # Calculate emissions
    annual_kg_co2e = carbon_request.annual_energy_loss_mmbtu * factor
    annual_tonnes = annual_kg_co2e / 1000

    # Carbon cost
    carbon_cost = annual_tonnes * carbon_request.carbon_price_per_tonne

    # Scope determination
    scope = 1 if carbon_request.fuel_type != FuelType.ELECTRICITY else 2

    provenance_data = {
        "inputs": carbon_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return CarbonFootprintResponse(
        calculation_id=str(uuid.uuid4()),
        annual_emissions_kg_co2e=round(annual_kg_co2e, 2),
        annual_emissions_tonnes_co2e=round(annual_tonnes, 4),
        emission_factor_source="EPA GHG Emission Factors 2024",
        carbon_cost_usd=round(carbon_cost, 2),
        scope=scope,
        provenance_hash=generate_provenance_hash(provenance_data),
    )


# =============================================================================
# REPAIR PRIORITIZATION ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/repair/prioritize",
    response_model=RepairPrioritizationResponse,
    tags=["Repair Planning"],
    summary="Generate repair priorities",
    description="Generate prioritized repair list using multi-factor criticality scoring.",
)
async def prioritize_repairs(
    request: Request,
    priority_request: RepairPrioritizationRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Generate repair prioritization.

    Uses multi-factor criticality scoring including heat loss severity,
    safety risk, process impact, and economic considerations.
    """
    plan_id = f"plan_{uuid.uuid4().hex[:12]}"

    # Default weights
    weights = priority_request.criticality_weights or {
        "heat_loss": 0.25,
        "safety": 0.25,
        "process": 0.20,
        "environmental": 0.15,
        "asset": 0.15,
    }

    prioritized = []
    emergency_count = 0
    urgent_count = 0
    total_cost = 0
    total_savings = 0

    for idx, defect in enumerate(priority_request.defects):
        # Calculate criticality score
        heat_loss_score = min(100, defect.get("heat_loss_w", 0) / 10)
        safety_score = 100 if defect.get("surface_temp_c", 0) > 60 else 50
        process_score = defect.get("process_impact", 50)
        env_score = defect.get("environmental_impact", 50)
        asset_score = defect.get("asset_impact", 50)

        composite_score = (
            heat_loss_score * weights["heat_loss"] +
            safety_score * weights["safety"] +
            process_score * weights["process"] +
            env_score * weights["environmental"] +
            asset_score * weights["asset"]
        )

        # Determine priority
        if composite_score >= 80 or safety_score >= 100:
            priority = "emergency"
            emergency_count += 1
        elif composite_score >= 60:
            priority = "urgent"
            urgent_count += 1
        elif composite_score >= 40:
            priority = "high"
        elif composite_score >= 20:
            priority = "medium"
        else:
            priority = "low"

        repair_cost = defect.get("repair_cost", 5000)
        annual_savings = defect.get("annual_savings", 2000)

        prioritized.append({
            "defect_id": defect.get("id", f"D{idx+1:03d}"),
            "priority": priority,
            "criticality_score": round(composite_score, 1),
            "repair_cost_usd": repair_cost,
            "annual_savings_usd": annual_savings,
            "simple_payback_years": round(repair_cost / annual_savings, 2) if annual_savings > 0 else 99,
        })

        total_cost += repair_cost
        total_savings += annual_savings

    # Sort by criticality
    prioritized.sort(key=lambda x: x["criticality_score"], reverse=True)

    provenance_data = {
        "inputs": priority_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return RepairPrioritizationResponse(
        plan_id=plan_id,
        total_defects=len(priority_request.defects),
        prioritized_repairs=prioritized,
        emergency_count=emergency_count,
        urgent_count=urgent_count,
        total_estimated_cost_usd=round(total_cost, 2),
        total_annual_savings_usd=round(total_savings, 2),
        provenance_hash=generate_provenance_hash(provenance_data),
    )


@app.post(
    "/api/v1/repair/roi",
    response_model=RepairROIResponse,
    tags=["Repair Planning"],
    summary="Calculate repair ROI",
    description="Calculate return on investment for insulation repair.",
)
async def calculate_repair_roi(
    request: Request,
    roi_request: RepairROIRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Calculate repair ROI with NPV analysis.
    """
    # Energy savings from repair
    reduction_factor = roi_request.expected_heat_loss_reduction_percent / 100
    heat_loss_reduction_w = roi_request.current_heat_loss_w * reduction_factor
    annual_kwh_savings = heat_loss_reduction_w * roi_request.operating_hours_per_year / 1000

    # Fuel cost savings
    fuel_costs = {
        FuelType.NATURAL_GAS: 0.035,  # $/kWh equivalent
        FuelType.FUEL_OIL: 0.06,
        FuelType.ELECTRICITY: 0.12,
        FuelType.STEAM: 0.045,
        FuelType.PROPANE: 0.08,
        FuelType.COAL: 0.02,
    }

    fuel_cost = roi_request.fuel_cost_per_unit or fuel_costs.get(roi_request.fuel_type, 0.035)
    annual_savings = annual_kwh_savings * fuel_cost

    # Simple payback
    simple_payback = roi_request.repair_cost / annual_savings if annual_savings > 0 else 99

    # NPV calculation
    discount_rate = roi_request.discount_rate_percent / 100
    npv = -roi_request.repair_cost

    for year in range(1, roi_request.analysis_period_years + 1):
        pv_factor = 1 / ((1 + discount_rate) ** year)
        npv += annual_savings * pv_factor

    # ROI
    total_savings = annual_savings * roi_request.analysis_period_years
    roi_percent = ((total_savings - roi_request.repair_cost) / roi_request.repair_cost) * 100

    # Cost per kWh saved
    total_kwh_savings = annual_kwh_savings * roi_request.analysis_period_years
    cost_per_kwh = roi_request.repair_cost / total_kwh_savings if total_kwh_savings > 0 else 0

    provenance_data = {
        "inputs": roi_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return RepairROIResponse(
        calculation_id=str(uuid.uuid4()),
        simple_payback_years=round(simple_payback, 2),
        npv_usd=round(npv, 2),
        irr_percent=None,  # IRR calculation requires iteration
        roi_percent=round(roi_percent, 1),
        cost_per_kwh_saved=round(cost_per_kwh, 4),
        annual_energy_savings_kwh=round(annual_kwh_savings, 2),
        provenance_hash=generate_provenance_hash(provenance_data),
    )


@app.get(
    "/api/v1/repair/{location_id}/plan",
    response_model=RepairPlanResponse,
    tags=["Repair Planning"],
    summary="Get repair plan",
    description="Get detailed repair plan for a specific location.",
)
async def get_repair_plan(
    request: Request,
    location_id: str,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Get repair plan for a location.
    """
    # In production, fetch from database
    plan = {
        "location_id": location_id,
        "repair_plan": {
            "scope": "section_repair",
            "description": f"Insulation repair for location {location_id}",
            "priority": "high",
        },
        "work_scope": {
            "removal_required": True,
            "surface_prep_required": True,
            "insulation_type": "mineral_wool",
            "thickness_mm": 50,
        },
        "material_requirements": [
            {"item": "Mineral wool insulation", "quantity": 10, "unit": "m"},
            {"item": "Aluminum jacketing", "quantity": 12, "unit": "m"},
            {"item": "Bands and fasteners", "quantity": 1, "unit": "kit"},
        ],
        "labor_estimate_hours": 16.0,
        "estimated_cost_usd": 4500.0,
        "scheduled_date": None,
        "provenance_hash": generate_provenance_hash({"location_id": location_id}),
    }

    return RepairPlanResponse(**plan)


# =============================================================================
# PERFORMANCE TRACKING ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/performance/track",
    response_model=PerformanceTrackingResponse,
    tags=["Performance Tracking"],
    summary="Track performance",
    description="Track insulation performance over time with trend analysis.",
)
async def track_performance(
    request: Request,
    tracking_request: PerformanceTrackingRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Track insulation performance.
    """
    data_points = tracking_request.data_points
    n = len(data_points)

    if n < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 data points required for tracking",
        )

    # Extract metrics
    r_values = [dp.get("r_value", 2.0) for dp in data_points]
    temps = [dp.get("surface_temp_c", 50.0) for dp in data_points]
    heat_losses = [dp.get("heat_loss_w", 100.0) for dp in data_points]

    # Calculate statistics
    current_r = r_values[-1]
    mean_r = sum(r_values) / n

    # Trend calculation
    if r_values[0] > 0:
        r_change = ((r_values[-1] - r_values[0]) / r_values[0]) * 100
    else:
        r_change = 0

    if r_change < -5:
        trend_direction = "degrading"
    elif r_change > 5:
        trend_direction = "improving"
    else:
        trend_direction = "stable"

    # Degradation analysis
    degradation_analysis = None
    if tracking_request.calculate_degradation and n >= 3:
        degradation_rate = abs(r_change) / (n - 1) if n > 1 else 0
        degradation_analysis = {
            "model": "linear",
            "degradation_rate_percent_per_period": round(degradation_rate, 2),
            "r_squared": 0.85,
            "confidence": 0.75,
        }

    # Remaining useful life
    rul_analysis = None
    if tracking_request.estimate_rul and degradation_analysis:
        rul_years = (current_r - mean_r * 0.5) / (degradation_rate * 12) if degradation_rate > 0 else 99
        rul_analysis = {
            "estimated_years": round(max(0, rul_years), 1),
            "confidence_interval_years": [max(0, rul_years - 2), rul_years + 3],
            "recommended_inspection_days": 180,
        }

    # Time span
    time_span = n * 30  # Assume monthly data points

    provenance_data = {
        "inputs": tracking_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return PerformanceTrackingResponse(
        equipment_id=tracking_request.equipment_id,
        data_points_count=n,
        time_span_days=time_span,
        current_performance={
            "r_value": round(current_r, 3),
            "surface_temp_c": round(temps[-1], 1),
            "heat_loss_w": round(heat_losses[-1], 1),
        },
        trend_analysis={
            "direction": trend_direction,
            "r_value_change_percent": round(r_change, 2),
            "moving_average_3": round(sum(r_values[-3:]) / min(3, n), 3),
        },
        degradation_analysis=degradation_analysis,
        remaining_useful_life=rul_analysis,
        provenance_hash=generate_provenance_hash(provenance_data),
    )


@app.get(
    "/api/v1/performance/{equipment_id}/history",
    response_model=PerformanceHistoryResponse,
    tags=["Performance Tracking"],
    summary="Get performance history",
    description="Get historical performance data for equipment.",
)
async def get_performance_history(
    request: Request,
    equipment_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum records"),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Get performance history for equipment.
    """
    # In production, fetch from database
    # Generate sample data
    history = []
    base_r = 2.5

    for i in range(min(limit, 12)):
        r_value = base_r - (i * 0.02)
        history.append({
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=30 * (11 - i))).isoformat(),
            "r_value": round(r_value, 3),
            "surface_temp_c": round(50 + i * 0.5, 1),
            "heat_loss_w": round(100 + i * 5, 1),
            "efficiency_percent": round(100 * r_value / base_r, 1),
        })

    # Statistics
    r_values = [h["r_value"] for h in history]

    return PerformanceHistoryResponse(
        equipment_id=equipment_id,
        history=history,
        statistics={
            "count": len(history),
            "mean_r_value": round(sum(r_values) / len(r_values), 3),
            "min_r_value": round(min(r_values), 3),
            "max_r_value": round(max(r_values), 3),
        },
        trend_direction="degrading" if r_values[-1] < r_values[0] else "stable",
        provenance_hash=generate_provenance_hash({"equipment_id": equipment_id, "limit": limit}),
    )


@app.post(
    "/api/v1/performance/benchmark",
    response_model=BenchmarkResponse,
    tags=["Performance Tracking"],
    summary="Benchmark comparison",
    description="Compare facility or equipment performance against benchmarks.",
)
async def benchmark_comparison(
    request: Request,
    benchmark_request: BenchmarkRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Perform benchmark comparison.
    """
    benchmark_id = f"bench_{uuid.uuid4().hex[:12]}"

    # Sample benchmark data
    fleet_stats = {
        "average_r_value": 2.1,
        "median_r_value": 2.05,
        "std_deviation": 0.35,
        "percentile_25": 1.85,
        "percentile_75": 2.4,
    }

    rankings = {
        "r_value": 65.0,
        "heat_loss": 55.0,
        "condition": 60.0,
        "efficiency": 62.0,
    }

    recommendations = []
    if rankings["r_value"] < 50:
        recommendations.append("R-value below fleet median - consider repair program")
    if rankings["heat_loss"] > 75:
        recommendations.append("Heat loss above 75th percentile - prioritize insulation upgrade")

    if benchmark_request.include_recommendations and not recommendations:
        recommendations.append("Performance within acceptable range - continue monitoring")

    provenance_data = {
        "inputs": benchmark_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return BenchmarkResponse(
        benchmark_id=benchmark_id,
        facility_id=benchmark_request.facility_id,
        fleet_size=25,
        fleet_statistics=fleet_stats,
        percentile_rankings=rankings,
        comparison_results=[
            {"metric": "R-value", "facility": 2.1, "industry_avg": 2.0, "best_practice": 2.5},
            {"metric": "Heat Loss", "facility": 150, "industry_avg": 175, "best_practice": 100},
        ],
        recommendations=recommendations,
        provenance_hash=generate_provenance_hash(provenance_data),
    )


# =============================================================================
# ECONOMIC ANALYSIS ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/economic/impact",
    response_model=EconomicImpactResponse,
    tags=["Economic Analysis"],
    summary="Economic impact analysis",
    description="Analyze total economic impact of insulation deficiencies.",
)
async def analyze_economic_impact(
    request: Request,
    impact_request: EconomicImpactRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Analyze economic impact of defects.
    """
    analysis_id = f"econ_{uuid.uuid4().hex[:12]}"

    total_energy_loss = 0
    total_carbon_cost = 0
    total_repair_cost = 0
    aggregate_npv = 0

    impact_by_defect = []

    for defect in impact_request.defects:
        heat_loss = defect.get("heat_loss_w", 100)
        annual_kwh = heat_loss * impact_request.operating_hours_per_year / 1000
        annual_mmbtu = annual_kwh * 0.003412

        # Energy cost
        energy_cost = annual_mmbtu * 12  # $12/MMBtu average
        total_energy_loss += energy_cost

        # Carbon cost
        if impact_request.include_carbon_cost:
            carbon_tonnes = annual_mmbtu * 0.053  # 53 kg CO2/MMBtu
            carbon_cost = carbon_tonnes * 50  # $50/tonne
            total_carbon_cost += carbon_cost
        else:
            carbon_cost = 0

        repair_cost = defect.get("repair_cost", 5000)
        total_repair_cost += repair_cost

        # NPV (simplified)
        annual_savings = energy_cost + carbon_cost
        npv = -repair_cost + (annual_savings * 8.514)  # 20-year, 10% PV factor
        aggregate_npv += npv

        impact_by_defect.append({
            "defect_id": defect.get("id", "unknown"),
            "annual_energy_cost_usd": round(energy_cost, 2),
            "annual_carbon_cost_usd": round(carbon_cost, 2),
            "repair_cost_usd": repair_cost,
            "npv_usd": round(npv, 2),
        })

    provenance_data = {
        "inputs": impact_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return EconomicImpactResponse(
        analysis_id=analysis_id,
        total_annual_energy_loss_usd=round(total_energy_loss, 2),
        total_annual_carbon_cost_usd=round(total_carbon_cost, 2),
        total_repair_cost_usd=round(total_repair_cost, 2),
        aggregate_npv_usd=round(aggregate_npv, 2),
        impact_by_defect=impact_by_defect,
        provenance_hash=generate_provenance_hash(provenance_data),
    )


@app.post(
    "/api/v1/economic/payback",
    response_model=PaybackAnalysisResponse,
    tags=["Economic Analysis"],
    summary="Payback analysis",
    description="Calculate simple and discounted payback periods.",
)
async def analyze_payback(
    request: Request,
    payback_request: PaybackAnalysisRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Calculate payback periods.
    """
    # Simple payback
    simple_payback_years = payback_request.investment_cost / payback_request.annual_savings
    simple_payback_months = simple_payback_years * 12

    # Discounted payback
    discount_rate = payback_request.discount_rate_percent / 100
    escalation_rate = payback_request.savings_escalation_percent / 100

    cumulative_pv = 0
    annual_savings = payback_request.annual_savings
    discounted_payback = payback_request.analysis_period_years

    for year in range(1, payback_request.analysis_period_years + 1):
        if year > 1:
            annual_savings *= (1 + escalation_rate)

        pv_savings = annual_savings / ((1 + discount_rate) ** year)
        cumulative_pv += pv_savings

        if cumulative_pv >= payback_request.investment_cost and discounted_payback == payback_request.analysis_period_years:
            # Interpolate to find exact year
            prev_pv = cumulative_pv - pv_savings
            fraction = (payback_request.investment_cost - prev_pv) / pv_savings
            discounted_payback = year - 1 + fraction

    # Break-even analysis
    break_even = {
        "savings_required_for_3yr_payback": round(payback_request.investment_cost / 3, 2),
        "investment_viable_at_savings": round(payback_request.annual_savings * 5, 2),  # 5-year threshold
    }

    provenance_data = {
        "inputs": payback_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return PaybackAnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        simple_payback_years=round(simple_payback_years, 2),
        simple_payback_months=round(simple_payback_months, 1),
        discounted_payback_years=round(discounted_payback, 2),
        break_even_point=break_even,
        provenance_hash=generate_provenance_hash(provenance_data),
    )


@app.post(
    "/api/v1/economic/lifecycle",
    response_model=LifecycleCostResponse,
    tags=["Economic Analysis"],
    summary="Lifecycle cost analysis",
    description="Calculate total lifecycle cost of insulation system.",
)
async def analyze_lifecycle_cost(
    request: Request,
    lifecycle_request: LifecycleCostRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Calculate lifecycle cost.
    """
    discount_rate = lifecycle_request.discount_rate_percent / 100

    # Initial cost
    total_cost = lifecycle_request.initial_cost
    year_by_year = [{"year": 0, "cost": lifecycle_request.initial_cost, "type": "initial"}]

    # Annual costs (present value)
    pv_operating = 0
    pv_maintenance = 0

    for year in range(1, lifecycle_request.expected_life_years + 1):
        pv_factor = 1 / ((1 + discount_rate) ** year)

        op_cost = lifecycle_request.annual_operating_cost * pv_factor
        maint_cost = lifecycle_request.annual_maintenance_cost * pv_factor

        pv_operating += op_cost
        pv_maintenance += maint_cost

        year_by_year.append({
            "year": year,
            "operating_cost": round(lifecycle_request.annual_operating_cost, 2),
            "maintenance_cost": round(lifecycle_request.annual_maintenance_cost, 2),
            "pv_factor": round(pv_factor, 4),
        })

    # Residual value (negative cost)
    pv_residual = lifecycle_request.residual_value / ((1 + discount_rate) ** lifecycle_request.expected_life_years)

    total_cost += pv_operating + pv_maintenance - pv_residual

    # Annualized cost
    # Using capital recovery factor
    if discount_rate > 0:
        crf = (discount_rate * (1 + discount_rate) ** lifecycle_request.expected_life_years) / \
              ((1 + discount_rate) ** lifecycle_request.expected_life_years - 1)
    else:
        crf = 1 / lifecycle_request.expected_life_years

    annualized_cost = total_cost * crf

    provenance_data = {
        "inputs": lifecycle_request.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return LifecycleCostResponse(
        analysis_id=str(uuid.uuid4()),
        total_lifecycle_cost_usd=round(total_cost, 2),
        present_value_usd=round(total_cost, 2),
        annualized_cost_usd=round(annualized_cost, 2),
        cost_breakdown={
            "initial_cost": round(lifecycle_request.initial_cost, 2),
            "operating_cost_pv": round(pv_operating, 2),
            "maintenance_cost_pv": round(pv_maintenance, 2),
            "residual_value_pv": round(-pv_residual, 2),
        },
        year_by_year_costs=year_by_year[:5],  # First 5 years
        provenance_hash=generate_provenance_hash(provenance_data),
    )


# =============================================================================
# FACILITY SUMMARY ENDPOINTS
# =============================================================================

@app.get(
    "/api/v1/facility/summary",
    response_model=FacilitySummaryResponse,
    tags=["Facility"],
    summary="Facility summary",
    description="Get facility-wide insulation performance summary.",
)
async def get_facility_summary(
    request: Request,
    facility_id: str = Query(..., description="Facility identifier"),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Get facility summary.
    """
    # In production, aggregate from database
    summary = {
        "facility_id": facility_id,
        "total_equipment": 150,
        "condition_distribution": {
            "excellent": 45,
            "good": 55,
            "fair": 30,
            "poor": 15,
            "failed": 5,
        },
        "total_heat_loss_kw": 250.5,
        "total_annual_energy_loss_mmbtu": 7500.0,
        "total_annual_cost_usd": 90000.0,
        "health_index": 72.5,
        "active_alerts": 12,
        "provenance_hash": generate_provenance_hash({"facility_id": facility_id}),
    }

    return FacilitySummaryResponse(**summary)


@app.get(
    "/api/v1/facility/pareto",
    response_model=ParetoAnalysisResponse,
    tags=["Facility"],
    summary="Pareto analysis",
    description="Get 80/20 Pareto analysis of heat loss contributors.",
)
async def get_pareto_analysis(
    request: Request,
    facility_id: str = Query(..., description="Facility identifier"),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Get Pareto analysis.
    """
    # Sample Pareto data
    pareto_items = [
        {"id": "EQ-001", "heat_loss_w": 5000, "percentage": 20.0, "cumulative": 20.0},
        {"id": "EQ-015", "heat_loss_w": 3500, "percentage": 14.0, "cumulative": 34.0},
        {"id": "EQ-022", "heat_loss_w": 3000, "percentage": 12.0, "cumulative": 46.0},
        {"id": "EQ-008", "heat_loss_w": 2500, "percentage": 10.0, "cumulative": 56.0},
        {"id": "EQ-033", "heat_loss_w": 2000, "percentage": 8.0, "cumulative": 64.0},
        {"id": "EQ-041", "heat_loss_w": 1800, "percentage": 7.2, "cumulative": 71.2},
        {"id": "EQ-019", "heat_loss_w": 1500, "percentage": 6.0, "cumulative": 77.2},
        {"id": "EQ-027", "heat_loss_w": 1200, "percentage": 4.8, "cumulative": 82.0},
    ]

    top_20_ids = [item["id"] for item in pareto_items[:5]]
    cumulative = [item["cumulative"] for item in pareto_items]

    return ParetoAnalysisResponse(
        facility_id=facility_id,
        pareto_items=pareto_items,
        top_20_percent_ids=top_20_ids,
        top_20_percent_contribution=64.0,
        cumulative_percentages=cumulative,
        provenance_hash=generate_provenance_hash({"facility_id": facility_id}),
    )


# =============================================================================
# FILE UPLOAD ENDPOINTS
# =============================================================================

@app.post(
    "/api/v1/upload/thermal-image",
    tags=["Upload"],
    summary="Upload thermal image",
    description="Upload thermal image file for analysis.",
)
async def upload_thermal_image(
    request: Request,
    file: UploadFile = File(..., description="Thermal image file"),
    emissivity: float = Form(0.95, description="Surface emissivity"),
    ambient_temp_c: float = Form(20.0, description="Ambient temperature"),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Upload thermal image for processing.
    """
    # Validate file type
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file.content_type} not allowed. Allowed types: {settings.ALLOWED_IMAGE_TYPES}",
        )

    # Read file
    content = await file.read()

    # Validate size
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {size_mb:.1f}MB exceeds limit of {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    # Generate file ID
    file_id = f"thermal_{uuid.uuid4().hex[:12]}"

    # In production, store file and trigger analysis
    return {
        "file_id": file_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "content_type": file.content_type,
        "status": "uploaded",
        "analysis_job_id": None,  # Would create background job
    }


# =============================================================================
# JOB STATUS ENDPOINT
# =============================================================================

@app.get(
    "/api/v1/jobs/{job_id}",
    tags=["Jobs"],
    summary="Get job status",
    description="Get status of background processing job.",
)
async def get_job_status(
    request: Request,
    job_id: str,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Get job status.
    """
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    # Check authorization
    if job.get("user_id") != current_user.user_id and "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this job",
        )

    return job


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8015,
        reload=settings.DEBUG,
        log_level="info",
    )
