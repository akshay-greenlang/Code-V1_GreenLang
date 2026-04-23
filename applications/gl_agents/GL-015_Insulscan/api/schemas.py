"""
GL-015 INSULSCAN - API Schemas

Pydantic v2 models for REST API request/response validation.

Provides:
- AnalyzeInsulationRequest/Response for insulation analysis
- BatchAnalysisRequest/Response for batch processing
- HotSpotDetectionRequest/Response for thermal image analysis
- RepairRecommendationResponse for repair recommendations
- HealthResponse for system health checks
- Prometheus metrics response

All responses include:
- computation_hash for traceability (SHA-256)
- timestamp (UTC)
- agent_version
- request_id
- warnings array

Zero-Hallucination Principle:
    All thermal calculations are performed by the deterministic engine.
    The LLM is used only for natural language explanations and summaries.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import hashlib
import json


# =============================================================================
# Constants
# =============================================================================

AGENT_VERSION = "1.0.0"
AGENT_ID = "GL-015"
AGENT_NAME = "INSULSCAN"


# =============================================================================
# Enums
# =============================================================================

class InsulationType(str, Enum):
    """Insulation material type classification."""
    MINERAL_WOOL = "mineral_wool"
    FIBERGLASS = "fiberglass"
    CALCIUM_SILICATE = "calcium_silicate"
    CERAMIC_FIBER = "ceramic_fiber"
    FOAM_GLASS = "foam_glass"
    POLYURETHANE = "polyurethane"
    AEROGEL = "aerogel"
    PERLITE = "perlite"
    VERMICULITE = "vermiculite"
    UNKNOWN = "unknown"


class ConditionRating(str, Enum):
    """Insulation condition rating classification."""
    EXCELLENT = "excellent"      # 90-100% effective
    GOOD = "good"               # 75-89% effective
    FAIR = "fair"               # 50-74% effective
    POOR = "poor"               # 25-49% effective
    CRITICAL = "critical"       # <25% effective or immediate attention


class DegradationMechanism(str, Enum):
    """Insulation degradation mechanism types."""
    MOISTURE_INGRESS = "moisture_ingress"
    THERMAL_CYCLING = "thermal_cycling"
    MECHANICAL_DAMAGE = "mechanical_damage"
    UV_DEGRADATION = "uv_degradation"
    CHEMICAL_ATTACK = "chemical_attack"
    AGING = "aging"
    COMPRESSION = "compression"
    SETTLING = "settling"
    CORROSION_UNDER_INSULATION = "corrosion_under_insulation"


class RepairPriority(str, Enum):
    """Repair priority classification."""
    CRITICAL = "critical"       # Immediate action required
    HIGH = "high"               # Action within 1 week
    MEDIUM = "medium"           # Action within 1 month
    LOW = "low"                 # Schedule during next maintenance
    MONITOR = "monitor"         # No action, continue monitoring


class HotSpotSeverity(str, Enum):
    """Hot spot severity classification."""
    SEVERE = "severe"           # >30C above ambient or design
    MODERATE = "moderate"       # 15-30C above
    MILD = "mild"               # 5-15C above
    NORMAL = "normal"           # Within acceptable range


# =============================================================================
# Base Response Model
# =============================================================================

class BaseAPIResponse(BaseModel):
    """Base response model with standard traceability fields."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "computation_hash": "sha256:a1b2c3d4e5f6...",
                "timestamp": "2025-12-27T10:30:00Z",
                "agent_version": "1.0.0",
                "agent_id": "GL-015",
                "request_id": "req_abc123",
                "warnings": []
            }
        }
    )

    computation_hash: str = Field(
        ...,
        description="SHA-256 hash of computation inputs for traceability"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of computation"
    )
    agent_version: str = Field(
        default=AGENT_VERSION,
        description="Version of the GL-015 agent"
    )
    agent_id: str = Field(
        default=AGENT_ID,
        description="Agent identifier"
    )
    request_id: str = Field(
        ...,
        description="Unique request identifier for tracing"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages from computation"
    )


# =============================================================================
# Analyze Insulation Models
# =============================================================================

class AnalyzeInsulationRequest(BaseModel):
    """Request model for single asset insulation analysis."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "asset_id": "PIPE-HX-001",
                "include_recommendations": True,
                "include_explanations": True,
                "measurement_data": {
                    "surface_temperature_C": 85.0,
                    "ambient_temperature_C": 25.0,
                    "design_surface_temp_C": 45.0,
                    "insulation_thickness_mm": 50.0
                }
            }
        }
    )

    asset_id: str = Field(
        ...,
        description="Unique asset identifier",
        min_length=1,
        max_length=128
    )
    include_recommendations: bool = Field(
        True,
        description="Include repair/maintenance recommendations in response"
    )
    include_explanations: bool = Field(
        False,
        description="Include natural language explanations for results"
    )
    measurement_data: Optional[Dict[str, float]] = Field(
        None,
        description="Optional real-time measurement data to override stored values"
    )
    analysis_depth: str = Field(
        "standard",
        description="Analysis depth: quick, standard, comprehensive"
    )

    @field_validator("analysis_depth")
    @classmethod
    def validate_analysis_depth(cls, v: str) -> str:
        """Validate analysis depth value."""
        valid_depths = ["quick", "standard", "comprehensive"]
        if v not in valid_depths:
            raise ValueError(f"analysis_depth must be one of: {valid_depths}")
        return v


class InsulationAnalysisResult(BaseModel):
    """Detailed insulation analysis results."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "effectiveness_percent": 72.5,
                "heat_loss_W_m2": 125.5,
                "heat_loss_design_W_m2": 45.0,
                "surface_temperature_C": 85.0,
                "ambient_temperature_C": 25.0,
                "condition_rating": "fair",
                "degradation_mechanisms": ["moisture_ingress", "aging"],
                "estimated_age_years": 12.5,
                "remaining_useful_life_years": 3.2
            }
        }
    )

    effectiveness_percent: float = Field(
        ...,
        description="Insulation effectiveness percentage (0-100)",
        ge=0.0,
        le=100.0
    )
    heat_loss_W_m2: float = Field(
        ...,
        description="Current heat loss in W/m2"
    )
    heat_loss_design_W_m2: float = Field(
        ...,
        description="Design heat loss in W/m2"
    )
    excess_heat_loss_W_m2: float = Field(
        ...,
        description="Excess heat loss above design in W/m2"
    )
    surface_temperature_C: float = Field(
        ...,
        description="Measured surface temperature in Celsius"
    )
    ambient_temperature_C: float = Field(
        ...,
        description="Ambient temperature in Celsius"
    )
    process_temperature_C: Optional[float] = Field(
        None,
        description="Internal process temperature in Celsius"
    )
    condition_rating: ConditionRating = Field(
        ...,
        description="Overall condition rating"
    )
    degradation_mechanisms: List[DegradationMechanism] = Field(
        default_factory=list,
        description="Detected degradation mechanisms"
    )
    insulation_type: InsulationType = Field(
        InsulationType.UNKNOWN,
        description="Type of insulation material"
    )
    insulation_thickness_mm: Optional[float] = Field(
        None,
        description="Insulation thickness in mm"
    )
    thermal_conductivity_W_mK: Optional[float] = Field(
        None,
        description="Effective thermal conductivity in W/(m*K)"
    )
    estimated_age_years: Optional[float] = Field(
        None,
        description="Estimated insulation age in years"
    )
    remaining_useful_life_years: Optional[float] = Field(
        None,
        description="Estimated remaining useful life in years"
    )
    energy_loss_annual_kWh: Optional[float] = Field(
        None,
        description="Annual energy loss in kWh"
    )
    energy_cost_annual_usd: Optional[float] = Field(
        None,
        description="Annual energy cost in USD"
    )


class AnalyzeInsulationResponse(BaseAPIResponse):
    """Response model for insulation analysis."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "computation_hash": "sha256:abc123...",
                "timestamp": "2025-12-27T10:30:00Z",
                "agent_version": "1.0.0",
                "agent_id": "GL-015",
                "request_id": "req_xyz789",
                "warnings": [],
                "asset_id": "PIPE-HX-001",
                "analysis_result": {
                    "effectiveness_percent": 72.5,
                    "heat_loss_W_m2": 125.5,
                    "heat_loss_design_W_m2": 45.0,
                    "excess_heat_loss_W_m2": 80.5,
                    "surface_temperature_C": 85.0,
                    "ambient_temperature_C": 25.0,
                    "condition_rating": "fair"
                },
                "processing_time_ms": 156.5
            }
        }
    )

    asset_id: str = Field(
        ...,
        description="Analyzed asset identifier"
    )
    analysis_result: InsulationAnalysisResult = Field(
        ...,
        description="Detailed analysis results"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )
    recommendations: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Repair/maintenance recommendations if requested"
    )
    explanations: Optional[Dict[str, str]] = Field(
        None,
        description="Natural language explanations if requested"
    )


# =============================================================================
# Batch Analysis Models
# =============================================================================

class BatchAnalysisRequest(BaseModel):
    """Request model for batch insulation analysis."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "asset_ids": ["PIPE-001", "PIPE-002", "TANK-001"],
                "parallel": True,
                "include_recommendations": True,
                "priority_filter": None
            }
        }
    )

    asset_ids: List[str] = Field(
        ...,
        description="List of asset identifiers to analyze",
        min_length=1,
        max_length=100
    )
    parallel: bool = Field(
        True,
        description="Process assets in parallel for faster results"
    )
    include_recommendations: bool = Field(
        True,
        description="Include recommendations for each asset"
    )
    priority_filter: Optional[List[RepairPriority]] = Field(
        None,
        description="Only return results for specified priority levels"
    )
    max_concurrent: int = Field(
        10,
        description="Maximum concurrent analyses (when parallel=True)",
        ge=1,
        le=50
    )


class BatchAnalysisItemResult(BaseModel):
    """Result for a single asset in batch analysis."""

    asset_id: str = Field(
        ...,
        description="Asset identifier"
    )
    success: bool = Field(
        ...,
        description="Whether analysis completed successfully"
    )
    analysis_result: Optional[InsulationAnalysisResult] = Field(
        None,
        description="Analysis result if successful"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if analysis failed"
    )
    recommendations: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Recommendations if requested and successful"
    )


class BatchAnalysisResponse(BaseAPIResponse):
    """Response model for batch insulation analysis."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "computation_hash": "sha256:batch123...",
                "timestamp": "2025-12-27T10:30:00Z",
                "agent_version": "1.0.0",
                "agent_id": "GL-015",
                "request_id": "req_batch456",
                "warnings": [],
                "total_requested": 3,
                "total_processed": 3,
                "total_successful": 2,
                "total_failed": 1,
                "processing_time_ms": 1250.5
            }
        }
    )

    total_requested: int = Field(
        ...,
        description="Total number of assets requested"
    )
    total_processed: int = Field(
        ...,
        description="Total number of assets processed"
    )
    total_successful: int = Field(
        ...,
        description="Number of successful analyses"
    )
    total_failed: int = Field(
        ...,
        description="Number of failed analyses"
    )
    results: List[BatchAnalysisItemResult] = Field(
        ...,
        description="Results for each asset"
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )
    summary_statistics: Optional[Dict[str, Any]] = Field(
        None,
        description="Summary statistics across all analyzed assets"
    )


# =============================================================================
# Hot Spot Detection Models
# =============================================================================

class CalibrationParams(BaseModel):
    """Thermal image calibration parameters."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "emissivity": 0.95,
                "reflected_temperature_C": 25.0,
                "distance_m": 3.0,
                "relative_humidity_percent": 50.0,
                "atmospheric_temperature_C": 25.0
            }
        }
    )

    emissivity: float = Field(
        0.95,
        description="Surface emissivity (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    reflected_temperature_C: float = Field(
        25.0,
        description="Reflected apparent temperature in Celsius"
    )
    distance_m: float = Field(
        3.0,
        description="Distance from camera to target in meters",
        gt=0.0
    )
    relative_humidity_percent: float = Field(
        50.0,
        description="Relative humidity percentage",
        ge=0.0,
        le=100.0
    )
    atmospheric_temperature_C: float = Field(
        25.0,
        description="Atmospheric temperature in Celsius"
    )
    transmission: float = Field(
        1.0,
        description="Transmission coefficient for external optics",
        ge=0.0,
        le=1.0
    )


class HotSpotDetectionRequest(BaseModel):
    """Request model for hot spot detection from thermal images."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "thermal_image_data": "base64_encoded_image_data...",
                "calibration_params": {
                    "emissivity": 0.95,
                    "reflected_temperature_C": 25.0,
                    "distance_m": 3.0
                },
                "asset_id": "PIPE-HX-001",
                "detection_threshold_C": 5.0
            }
        }
    )

    thermal_image_data: str = Field(
        ...,
        description="Base64-encoded thermal image data (FLIR, TIFF, or JPEG format)",
        min_length=100
    )
    calibration_params: CalibrationParams = Field(
        default_factory=CalibrationParams,
        description="Thermal camera calibration parameters"
    )
    asset_id: Optional[str] = Field(
        None,
        description="Associated asset identifier"
    )
    detection_threshold_C: float = Field(
        5.0,
        description="Temperature threshold above ambient for hot spot detection",
        gt=0.0
    )
    reference_temperature_C: Optional[float] = Field(
        None,
        description="Reference/design temperature for comparison"
    )
    image_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional image metadata (timestamp, location, etc.)"
    )

    @field_validator("thermal_image_data")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate base64 encoding."""
        import base64
        try:
            # Try to decode to verify it's valid base64
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("thermal_image_data must be valid base64-encoded data")


class DetectedHotSpot(BaseModel):
    """Details of a detected hot spot."""

    hotspot_id: str = Field(
        ...,
        description="Unique identifier for this hot spot"
    )
    location_x_percent: float = Field(
        ...,
        description="X location as percentage of image width (0-100)",
        ge=0.0,
        le=100.0
    )
    location_y_percent: float = Field(
        ...,
        description="Y location as percentage of image height (0-100)",
        ge=0.0,
        le=100.0
    )
    max_temperature_C: float = Field(
        ...,
        description="Maximum temperature in hot spot region"
    )
    avg_temperature_C: float = Field(
        ...,
        description="Average temperature in hot spot region"
    )
    delta_T_C: float = Field(
        ...,
        description="Temperature difference from reference/ambient"
    )
    area_percent: float = Field(
        ...,
        description="Hot spot area as percentage of image",
        ge=0.0,
        le=100.0
    )
    severity: HotSpotSeverity = Field(
        ...,
        description="Hot spot severity classification"
    )
    estimated_heat_loss_W: Optional[float] = Field(
        None,
        description="Estimated heat loss from this hot spot"
    )
    bounding_box: Optional[Dict[str, float]] = Field(
        None,
        description="Bounding box coordinates (x1, y1, x2, y2 as percentages)"
    )


class HotSpotDetectionResponse(BaseAPIResponse):
    """Response model for hot spot detection."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "computation_hash": "sha256:hotspot123...",
                "timestamp": "2025-12-27T10:30:00Z",
                "agent_version": "1.0.0",
                "agent_id": "GL-015",
                "request_id": "req_hotspot789",
                "warnings": [],
                "asset_id": "PIPE-HX-001",
                "total_hotspots_detected": 3,
                "processing_time_ms": 450.5
            }
        }
    )

    asset_id: Optional[str] = Field(
        None,
        description="Associated asset identifier"
    )
    total_hotspots_detected: int = Field(
        ...,
        description="Total number of hot spots detected"
    )
    hotspots: List[DetectedHotSpot] = Field(
        default_factory=list,
        description="List of detected hot spots"
    )
    image_temperature_range: Dict[str, float] = Field(
        ...,
        description="Temperature range in image (min, max, avg)"
    )
    ambient_temperature_C: float = Field(
        ...,
        description="Detected or provided ambient temperature"
    )
    total_estimated_heat_loss_W: Optional[float] = Field(
        None,
        description="Total estimated heat loss from all hot spots"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )
    annotated_image_base64: Optional[str] = Field(
        None,
        description="Base64-encoded annotated image with hot spots marked"
    )


# =============================================================================
# Asset Condition Models
# =============================================================================

class AssetConditionResponse(BaseAPIResponse):
    """Response model for asset condition endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "computation_hash": "sha256:condition123...",
                "timestamp": "2025-12-27T10:30:00Z",
                "agent_version": "1.0.0",
                "agent_id": "GL-015",
                "request_id": "req_condition456",
                "warnings": [],
                "asset_id": "PIPE-HX-001",
                "condition_rating": "fair",
                "effectiveness_percent": 72.5,
                "last_inspection_date": "2025-12-01T00:00:00Z"
            }
        }
    )

    asset_id: str = Field(
        ...,
        description="Asset identifier"
    )
    condition_rating: ConditionRating = Field(
        ...,
        description="Current condition rating"
    )
    effectiveness_percent: float = Field(
        ...,
        description="Current effectiveness percentage"
    )
    last_inspection_date: Optional[datetime] = Field(
        None,
        description="Date of last inspection"
    )
    last_analysis_date: Optional[datetime] = Field(
        None,
        description="Date of last analysis"
    )
    trend: str = Field(
        "stable",
        description="Condition trend: improving, stable, degrading"
    )
    days_until_next_inspection: Optional[int] = Field(
        None,
        description="Recommended days until next inspection"
    )
    active_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of active issues/concerns"
    )


# =============================================================================
# Historical Data Models
# =============================================================================

class HistoricalDataPoint(BaseModel):
    """Single historical data point."""

    timestamp: datetime = Field(
        ...,
        description="Data point timestamp"
    )
    effectiveness_percent: Optional[float] = Field(
        None,
        description="Effectiveness at this time"
    )
    surface_temperature_C: Optional[float] = Field(
        None,
        description="Surface temperature at this time"
    )
    heat_loss_W_m2: Optional[float] = Field(
        None,
        description="Heat loss at this time"
    )
    condition_rating: Optional[ConditionRating] = Field(
        None,
        description="Condition rating at this time"
    )
    data_source: str = Field(
        "measurement",
        description="Data source: measurement, calculation, estimate"
    )


class AssetHistoryResponse(BaseAPIResponse):
    """Response model for asset historical data."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "computation_hash": "sha256:history123...",
                "timestamp": "2025-12-27T10:30:00Z",
                "agent_version": "1.0.0",
                "agent_id": "GL-015",
                "request_id": "req_history789",
                "warnings": [],
                "asset_id": "PIPE-HX-001",
                "data_points_count": 100,
                "time_range_days": 30
            }
        }
    )

    asset_id: str = Field(
        ...,
        description="Asset identifier"
    )
    start_date: datetime = Field(
        ...,
        description="Start of time range"
    )
    end_date: datetime = Field(
        ...,
        description="End of time range"
    )
    data_points_count: int = Field(
        ...,
        description="Number of data points returned"
    )
    time_range_days: float = Field(
        ...,
        description="Time range in days"
    )
    resolution: str = Field(
        "1h",
        description="Data resolution"
    )
    data_points: List[HistoricalDataPoint] = Field(
        default_factory=list,
        description="Historical data points"
    )
    statistics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Statistical summary (min, max, avg, std) per metric"
    )
    trend_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Trend analysis results"
    )


# =============================================================================
# Repair Recommendation Models
# =============================================================================

class RepairRecommendation(BaseModel):
    """Single repair recommendation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendation_id": "rec_001",
                "asset_id": "PIPE-HX-001",
                "priority": "high",
                "category": "insulation_replacement",
                "title": "Replace degraded pipe insulation",
                "description": "Mineral wool insulation shows significant moisture damage",
                "estimated_cost_usd": 2500.0,
                "estimated_savings_annual_usd": 1200.0,
                "payback_period_years": 2.1,
                "recommended_timeframe": "Within 2 weeks"
            }
        }
    )

    recommendation_id: str = Field(
        ...,
        description="Unique recommendation identifier"
    )
    asset_id: str = Field(
        ...,
        description="Associated asset identifier"
    )
    priority: RepairPriority = Field(
        ...,
        description="Repair priority level"
    )
    category: str = Field(
        ...,
        description="Recommendation category"
    )
    title: str = Field(
        ...,
        description="Short recommendation title"
    )
    description: str = Field(
        ...,
        description="Detailed recommendation description"
    )
    rationale: str = Field(
        ...,
        description="Reasoning behind the recommendation"
    )
    estimated_cost_usd: float = Field(
        ...,
        description="Estimated repair cost in USD",
        ge=0.0
    )
    estimated_savings_annual_usd: float = Field(
        ...,
        description="Estimated annual savings in USD",
        ge=0.0
    )
    payback_period_years: Optional[float] = Field(
        None,
        description="Simple payback period in years"
    )
    roi_percent: Optional[float] = Field(
        None,
        description="Return on investment percentage"
    )
    recommended_timeframe: str = Field(
        ...,
        description="Recommended timeframe for action"
    )
    required_materials: Optional[List[str]] = Field(
        None,
        description="List of required materials"
    )
    estimated_labor_hours: Optional[float] = Field(
        None,
        description="Estimated labor hours"
    )
    confidence: float = Field(
        ...,
        description="Recommendation confidence (0-1)",
        ge=0.0,
        le=1.0
    )


class RepairRecommendationRequest(BaseModel):
    """Request model for generating repair recommendations."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "asset_ids": ["PIPE-001", "PIPE-002"],
                "budget_limit_usd": 10000.0,
                "priority_filter": ["critical", "high"],
                "include_roi_analysis": True
            }
        }
    )

    asset_ids: Optional[List[str]] = Field(
        None,
        description="Specific asset IDs (None for all assets)"
    )
    budget_limit_usd: Optional[float] = Field(
        None,
        description="Maximum budget constraint in USD",
        gt=0.0
    )
    priority_filter: Optional[List[RepairPriority]] = Field(
        None,
        description="Filter by priority levels"
    )
    include_roi_analysis: bool = Field(
        True,
        description="Include ROI analysis in recommendations"
    )
    optimization_objective: str = Field(
        "cost_benefit",
        description="Optimization objective: cost_benefit, energy_savings, risk_reduction"
    )


class RepairRecommendationResponse(BaseAPIResponse):
    """Response model for repair recommendations."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "computation_hash": "sha256:repair123...",
                "timestamp": "2025-12-27T10:30:00Z",
                "agent_version": "1.0.0",
                "agent_id": "GL-015",
                "request_id": "req_repair456",
                "warnings": [],
                "total_recommendations": 5,
                "total_cost_usd": 15000.0,
                "total_savings_annual_usd": 8500.0
            }
        }
    )

    recommendations: List[RepairRecommendation] = Field(
        ...,
        description="List of repair recommendations"
    )
    total_recommendations: int = Field(
        ...,
        description="Total number of recommendations"
    )
    total_cost_usd: float = Field(
        ...,
        description="Total estimated cost of all recommendations"
    )
    total_savings_annual_usd: float = Field(
        ...,
        description="Total annual savings if all recommendations implemented"
    )
    overall_payback_period_years: Optional[float] = Field(
        None,
        description="Overall payback period for all recommendations"
    )
    priority_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of recommendations by priority"
    )
    budget_utilization_percent: Optional[float] = Field(
        None,
        description="Budget utilization percentage if budget limit specified"
    )


# =============================================================================
# Health and Metrics Models
# =============================================================================

class ComponentHealth(BaseModel):
    """Health status of a system component."""

    name: str = Field(
        ...,
        description="Component name"
    )
    status: str = Field(
        ...,
        description="Status: healthy, degraded, unhealthy"
    )
    latency_ms: Optional[float] = Field(
        None,
        description="Component response latency in ms"
    )
    message: Optional[str] = Field(
        None,
        description="Status message"
    )
    last_check: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last health check timestamp"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "agent_id": "GL-015",
                "timestamp": "2025-12-27T10:30:00Z",
                "uptime_seconds": 86400.0,
                "component_statuses": [
                    {
                        "name": "database",
                        "status": "healthy",
                        "latency_ms": 5.2
                    },
                    {
                        "name": "thermal_analyzer",
                        "status": "healthy",
                        "latency_ms": 12.5
                    }
                ]
            }
        }
    )

    status: str = Field(
        ...,
        description="Overall health status: healthy, degraded, unhealthy"
    )
    version: str = Field(
        AGENT_VERSION,
        description="Agent version"
    )
    agent_id: str = Field(
        AGENT_ID,
        description="Agent identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp"
    )
    uptime_seconds: float = Field(
        ...,
        description="Application uptime in seconds"
    )
    component_statuses: List[ComponentHealth] = Field(
        default_factory=list,
        description="Individual component health statuses"
    )
    memory_usage_mb: Optional[float] = Field(
        None,
        description="Current memory usage in MB"
    )
    active_requests: Optional[int] = Field(
        None,
        description="Number of active requests"
    )


class MetricValue(BaseModel):
    """Single Prometheus metric value."""

    name: str = Field(
        ...,
        description="Metric name"
    )
    value: float = Field(
        ...,
        description="Metric value"
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Metric labels"
    )
    metric_type: str = Field(
        "gauge",
        description="Metric type: gauge, counter, histogram"
    )
    help_text: Optional[str] = Field(
        None,
        description="Metric help text"
    )


class MetricsResponse(BaseModel):
    """Prometheus metrics response."""

    metrics: List[MetricValue] = Field(
        default_factory=list,
        description="List of metrics"
    )

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus exposition format."""
        lines = []
        for metric in self.metrics:
            # Add HELP if available
            if metric.help_text:
                lines.append(f"# HELP {metric.name} {metric.help_text}")
            # Add TYPE
            lines.append(f"# TYPE {metric.name} {metric.metric_type}")
            # Add metric value with labels
            labels_str = ""
            if metric.labels:
                label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
                labels_str = "{" + ",".join(label_pairs) + "}"
            lines.append(f"{metric.name}{labels_str} {metric.value}")
        return "\n".join(lines)


# =============================================================================
# Error Response Models
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: Optional[str] = Field(
        None,
        description="Field that caused the error"
    )
    message: str = Field(
        ...,
        description="Error message"
    )
    code: str = Field(
        ...,
        description="Error code"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "validation_error",
                "message": "Invalid request parameters",
                "details": [
                    {
                        "field": "asset_id",
                        "message": "Asset not found",
                        "code": "asset_not_found"
                    }
                ],
                "request_id": "req_abc123",
                "timestamp": "2025-12-27T10:30:00Z"
            }
        }
    )

    error: str = Field(
        ...,
        description="Error type"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[List[ErrorDetail]] = Field(
        None,
        description="Detailed error information"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for tracing"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )


# =============================================================================
# Utility Functions
# =============================================================================

def compute_hash(data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 hash of input data for traceability.

    Args:
        data: Dictionary of computation inputs

    Returns:
        SHA-256 hash string prefixed with 'sha256:'
    """
    # Sort keys for deterministic hashing
    serialized = json.dumps(data, sort_keys=True, default=str)
    hash_value = hashlib.sha256(serialized.encode()).hexdigest()
    return f"sha256:{hash_value}"


def create_response_with_hash(
    response_class: type,
    input_data: Dict[str, Any],
    request_id: str,
    **kwargs
) -> BaseAPIResponse:
    """
    Create a response with computed hash.

    Args:
        response_class: Response model class
        input_data: Input data to hash
        request_id: Request identifier
        **kwargs: Response field values

    Returns:
        Response instance with computed hash
    """
    computation_hash = compute_hash(input_data)
    return response_class(
        computation_hash=computation_hash,
        request_id=request_id,
        **kwargs
    )


# =============================================================================
# Export all models
# =============================================================================

__all__ = [
    # Enums
    "InsulationType",
    "ConditionRating",
    "DegradationMechanism",
    "RepairPriority",
    "HotSpotSeverity",
    # Base
    "BaseAPIResponse",
    # Analyze Insulation
    "AnalyzeInsulationRequest",
    "InsulationAnalysisResult",
    "AnalyzeInsulationResponse",
    # Batch Analysis
    "BatchAnalysisRequest",
    "BatchAnalysisItemResult",
    "BatchAnalysisResponse",
    # Hot Spot Detection
    "CalibrationParams",
    "HotSpotDetectionRequest",
    "DetectedHotSpot",
    "HotSpotDetectionResponse",
    # Asset Condition
    "AssetConditionResponse",
    # Historical Data
    "HistoricalDataPoint",
    "AssetHistoryResponse",
    # Repair Recommendations
    "RepairRecommendation",
    "RepairRecommendationRequest",
    "RepairRecommendationResponse",
    # Health & Metrics
    "ComponentHealth",
    "HealthResponse",
    "MetricValue",
    "MetricsResponse",
    # Error
    "ErrorDetail",
    "ErrorResponse",
    # Utilities
    "compute_hash",
    "create_response_with_hash",
    # Constants
    "AGENT_VERSION",
    "AGENT_ID",
    "AGENT_NAME",
]
