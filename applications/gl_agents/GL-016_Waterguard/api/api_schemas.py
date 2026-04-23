"""
GL-016_Waterguard API Schemas

Pydantic models for API request/response validation and serialization.
Implements comprehensive schemas for water chemistry optimization endpoints.

Author: GL-APIDeveloper
Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# Enumerations
# =============================================================================

class OperatingMode(str, Enum):
    """Operating mode for the cooling tower system."""
    NORMAL = "normal"
    CONSERVATION = "conservation"
    HIGH_LOAD = "high_load"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


class RecommendationPriority(str, Enum):
    """Priority level for recommendations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendationType(str, Enum):
    """Type of optimization recommendation."""
    BLOWDOWN_ADJUSTMENT = "blowdown_adjustment"
    DOSING_RATE_CHANGE = "dosing_rate_change"
    COC_TARGET_UPDATE = "coc_target_update"
    MAINTENANCE_ALERT = "maintenance_alert"
    COMPLIANCE_WARNING = "compliance_warning"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"


class RecommendationStatus(str, Enum):
    """Status of a recommendation."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    EXPIRED = "expired"


class ComplianceStatus(str, Enum):
    """Compliance status for constraints."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


class HealthStatus(str, Enum):
    """Health status for system components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# =============================================================================
# Base Models
# =============================================================================

class BaseAPIModel(BaseModel):
    """Base model with common configuration."""

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
        schema_extra = {}


class TimestampedModel(BaseAPIModel):
    """Model with timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# =============================================================================
# Chemistry Models
# =============================================================================

class ChemistryReading(BaseAPIModel):
    """Single chemistry parameter reading."""
    parameter: str = Field(..., description="Parameter name (e.g., pH, conductivity)")
    value: float = Field(..., description="Current measured value")
    unit: str = Field(..., description="Unit of measurement")
    min_limit: Optional[float] = Field(None, description="Minimum acceptable limit")
    max_limit: Optional[float] = Field(None, description="Maximum acceptable limit")
    target: Optional[float] = Field(None, description="Target setpoint")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_within_limits(self) -> bool:
        """Check if value is within acceptable limits."""
        if self.min_limit is not None and self.value < self.min_limit:
            return False
        if self.max_limit is not None and self.value > self.max_limit:
            return False
        return True


class ChemistryStateResponse(BaseAPIModel):
    """Complete water chemistry state response."""
    tower_id: str = Field(..., description="Cooling tower identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Primary parameters
    ph: float = Field(..., ge=0, le=14, description="pH level (0-14)")
    conductivity: float = Field(..., ge=0, description="Conductivity in uS/cm")
    tds: float = Field(..., ge=0, description="Total Dissolved Solids in ppm")
    cycles_of_concentration: float = Field(..., ge=1, description="Cycles of concentration")

    # Secondary parameters
    alkalinity: Optional[float] = Field(None, ge=0, description="Alkalinity in ppm as CaCO3")
    hardness: Optional[float] = Field(None, ge=0, description="Total hardness in ppm as CaCO3")
    chloride: Optional[float] = Field(None, ge=0, description="Chloride concentration in ppm")
    silica: Optional[float] = Field(None, ge=0, description="Silica concentration in ppm")
    temperature: Optional[float] = Field(None, description="Water temperature in Celsius")

    # Indices
    langelier_saturation_index: Optional[float] = Field(None, description="LSI value")
    ryznar_stability_index: Optional[float] = Field(None, description="RSI value")

    # Detailed readings
    readings: List[ChemistryReading] = Field(default_factory=list)

    # Status
    overall_status: ComplianceStatus = Field(
        default=ComplianceStatus.UNKNOWN,
        description="Overall chemistry compliance status"
    )
    parameters_out_of_spec: List[str] = Field(
        default_factory=list,
        description="List of parameters outside acceptable limits"
    )

    class Config:
        schema_extra = {
            "example": {
                "tower_id": "tower-001",
                "timestamp": "2025-01-15T10:30:00Z",
                "ph": 7.8,
                "conductivity": 1500.0,
                "tds": 1200.0,
                "cycles_of_concentration": 4.5,
                "alkalinity": 120.0,
                "hardness": 200.0,
                "temperature": 32.5,
                "langelier_saturation_index": 0.5,
                "overall_status": "compliant",
                "parameters_out_of_spec": []
            }
        }


# =============================================================================
# Optimization Models
# =============================================================================

class OptimizationRequest(BaseAPIModel):
    """Request to trigger optimization cycle."""
    tower_id: str = Field(..., description="Cooling tower identifier")
    operating_mode: OperatingMode = Field(
        default=OperatingMode.NORMAL,
        description="Current operating mode"
    )
    force_optimization: bool = Field(
        default=False,
        description="Force optimization even if recently run"
    )
    target_coc: Optional[float] = Field(
        None,
        ge=1.0,
        le=10.0,
        description="Target cycles of concentration (1-10)"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional optimization constraints"
    )

    class Config:
        schema_extra = {
            "example": {
                "tower_id": "tower-001",
                "operating_mode": "normal",
                "force_optimization": False,
                "target_coc": 5.0,
                "constraints": {
                    "max_blowdown_rate": 50.0,
                    "min_water_savings": 10.0
                }
            }
        }


class OptimizationResult(BaseAPIModel):
    """Result of a single optimization calculation."""
    parameter: str = Field(..., description="Optimized parameter name")
    current_value: float = Field(..., description="Current parameter value")
    recommended_value: float = Field(..., description="Recommended new value")
    change_percent: float = Field(..., description="Percentage change")
    impact_score: float = Field(..., ge=0, le=100, description="Impact score (0-100)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level (0-1)")


class OptimizationResponse(BaseAPIModel):
    """Response from optimization engine."""
    optimization_id: str = Field(..., description="Unique optimization run identifier")
    tower_id: str = Field(..., description="Cooling tower identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(..., description="Optimization status")

    # Optimization results
    results: List[OptimizationResult] = Field(default_factory=list)

    # Key recommendations
    recommended_coc: float = Field(..., description="Recommended cycles of concentration")
    recommended_blowdown_rate: float = Field(..., description="Recommended blowdown rate (gpm)")
    recommended_dosing_rates: Dict[str, float] = Field(
        default_factory=dict,
        description="Recommended dosing rates by chemical type"
    )

    # Projected savings
    projected_water_savings_percent: float = Field(..., description="Projected water savings %")
    projected_energy_savings_percent: float = Field(..., description="Projected energy savings %")
    projected_chemical_savings_percent: float = Field(..., description="Projected chemical savings %")

    # Execution details
    execution_time_ms: float = Field(..., description="Optimization execution time in ms")
    model_version: str = Field(..., description="ML model version used")
    recommendations_generated: int = Field(..., description="Number of recommendations generated")

    class Config:
        schema_extra = {
            "example": {
                "optimization_id": "opt-20250115-001",
                "tower_id": "tower-001",
                "timestamp": "2025-01-15T10:30:00Z",
                "status": "completed",
                "results": [],
                "recommended_coc": 5.2,
                "recommended_blowdown_rate": 12.5,
                "recommended_dosing_rates": {
                    "inhibitor": 2.5,
                    "biocide": 0.5,
                    "dispersant": 1.0
                },
                "projected_water_savings_percent": 15.0,
                "projected_energy_savings_percent": 8.0,
                "projected_chemical_savings_percent": 12.0,
                "execution_time_ms": 245.5,
                "model_version": "v2.1.0",
                "recommendations_generated": 3
            }
        }


# =============================================================================
# Recommendation Models
# =============================================================================

class RecommendationResponse(BaseAPIModel):
    """Single recommendation response."""
    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    tower_id: str = Field(..., description="Cooling tower identifier")
    type: RecommendationType = Field(..., description="Type of recommendation")
    priority: RecommendationPriority = Field(..., description="Priority level")
    status: RecommendationStatus = Field(..., description="Current status")

    # Recommendation details
    title: str = Field(..., description="Short recommendation title")
    description: str = Field(..., description="Detailed description")
    action_required: str = Field(..., description="Required action")

    # Technical details
    current_value: Optional[float] = Field(None, description="Current parameter value")
    recommended_value: Optional[float] = Field(None, description="Recommended value")
    parameter: Optional[str] = Field(None, description="Affected parameter")
    unit: Optional[str] = Field(None, description="Unit of measurement")

    # Impact assessment
    impact_score: float = Field(..., ge=0, le=100, description="Impact score (0-100)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level (0-1)")
    projected_savings: Optional[float] = Field(None, description="Projected savings")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    approved_at: Optional[datetime] = Field(None, description="Approval timestamp")
    approved_by: Optional[str] = Field(None, description="User who approved")

    # Explainability
    reasoning: Optional[str] = Field(None, description="AI reasoning for recommendation")
    supporting_data: Optional[Dict[str, Any]] = Field(None, description="Supporting data")

    class Config:
        schema_extra = {
            "example": {
                "recommendation_id": "rec-001",
                "tower_id": "tower-001",
                "type": "blowdown_adjustment",
                "priority": "medium",
                "status": "pending",
                "title": "Increase Blowdown Rate",
                "description": "Conductivity trending high. Increase blowdown to maintain COC target.",
                "action_required": "Increase blowdown rate from 10 gpm to 12.5 gpm",
                "current_value": 10.0,
                "recommended_value": 12.5,
                "parameter": "blowdown_rate",
                "unit": "gpm",
                "impact_score": 75.0,
                "confidence": 0.92,
                "projected_savings": 500.0,
                "created_at": "2025-01-15T10:30:00Z",
                "reasoning": "Based on 24-hour conductivity trend and COC analysis"
            }
        }


class RecommendationListResponse(BaseAPIModel):
    """List of recommendations response."""
    tower_id: str = Field(..., description="Cooling tower identifier")
    total_count: int = Field(..., description="Total number of recommendations")
    pending_count: int = Field(..., description="Number of pending recommendations")
    recommendations: List[RecommendationResponse] = Field(default_factory=list)

    # Summary by priority
    critical_count: int = Field(default=0)
    high_count: int = Field(default=0)
    medium_count: int = Field(default=0)
    low_count: int = Field(default=0)


class RecommendationApprovalRequest(BaseAPIModel):
    """Request to approve or reject a recommendation."""
    approved: bool = Field(..., description="Whether to approve the recommendation")
    operator_notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="Operator notes or reason for decision"
    )
    modified_value: Optional[float] = Field(
        None,
        description="Modified value if different from recommendation"
    )
    schedule_implementation: Optional[datetime] = Field(
        None,
        description="Schedule implementation for later"
    )

    class Config:
        schema_extra = {
            "example": {
                "approved": True,
                "operator_notes": "Approved after shift supervisor review",
                "modified_value": None,
                "schedule_implementation": None
            }
        }


class RecommendationApprovalResponse(BaseAPIModel):
    """Response after processing recommendation approval."""
    recommendation_id: str = Field(..., description="Recommendation identifier")
    status: RecommendationStatus = Field(..., description="Updated status")
    approved: bool = Field(..., description="Whether approved")
    approved_at: datetime = Field(default_factory=datetime.utcnow)
    approved_by: str = Field(..., description="User who approved")
    implementation_scheduled: Optional[datetime] = Field(None)
    message: str = Field(..., description="Result message")


# =============================================================================
# Blowdown Models
# =============================================================================

class BlowdownStatusResponse(BaseAPIModel):
    """Blowdown system status response."""
    tower_id: str = Field(..., description="Cooling tower identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Current status
    blowdown_active: bool = Field(..., description="Whether blowdown is currently active")
    current_rate_gpm: float = Field(..., ge=0, description="Current blowdown rate in GPM")
    target_rate_gpm: float = Field(..., ge=0, description="Target blowdown rate in GPM")

    # COC tracking
    current_coc: float = Field(..., ge=1, description="Current cycles of concentration")
    target_coc: float = Field(..., ge=1, description="Target cycles of concentration")
    coc_deviation: float = Field(..., description="Deviation from target COC")

    # Daily statistics
    total_blowdown_today_gallons: float = Field(..., ge=0, description="Total blowdown today")
    blowdown_events_today: int = Field(..., ge=0, description="Number of blowdown events")
    average_duration_minutes: float = Field(..., ge=0, description="Average event duration")

    # Setpoints
    conductivity_setpoint: float = Field(..., ge=0, description="Conductivity setpoint uS/cm")
    high_conductivity_alarm: float = Field(..., ge=0, description="High conductivity alarm")
    low_conductivity_alarm: float = Field(..., ge=0, description="Low conductivity alarm")

    # Valve status
    valve_position_percent: float = Field(..., ge=0, le=100, description="Valve position %")
    valve_status: str = Field(..., description="Valve status (open/closed/modulating)")

    class Config:
        schema_extra = {
            "example": {
                "tower_id": "tower-001",
                "timestamp": "2025-01-15T10:30:00Z",
                "blowdown_active": True,
                "current_rate_gpm": 12.5,
                "target_rate_gpm": 12.5,
                "current_coc": 4.8,
                "target_coc": 5.0,
                "coc_deviation": -0.2,
                "total_blowdown_today_gallons": 1500.0,
                "blowdown_events_today": 8,
                "average_duration_minutes": 15.0,
                "conductivity_setpoint": 1500.0,
                "high_conductivity_alarm": 2000.0,
                "low_conductivity_alarm": 800.0,
                "valve_position_percent": 25.0,
                "valve_status": "modulating"
            }
        }


# =============================================================================
# Dosing Models
# =============================================================================

class DosingChannel(BaseAPIModel):
    """Single dosing channel status."""
    channel_id: str = Field(..., description="Channel identifier")
    chemical_type: str = Field(..., description="Type of chemical")
    chemical_name: str = Field(..., description="Chemical product name")

    # Status
    active: bool = Field(..., description="Whether dosing is active")
    current_rate_ml_hr: float = Field(..., ge=0, description="Current dosing rate mL/hr")
    target_rate_ml_hr: float = Field(..., ge=0, description="Target dosing rate mL/hr")

    # Tank levels
    tank_level_percent: float = Field(..., ge=0, le=100, description="Tank level %")
    tank_capacity_liters: float = Field(..., ge=0, description="Tank capacity in liters")
    estimated_days_remaining: float = Field(..., ge=0, description="Days until empty")

    # Daily usage
    daily_usage_ml: float = Field(..., ge=0, description="Daily usage in mL")
    monthly_usage_liters: float = Field(..., ge=0, description="Monthly usage in liters")

    # Pump status
    pump_status: str = Field(..., description="Pump status")
    last_calibration: Optional[datetime] = Field(None, description="Last calibration date")


class DosingStatusResponse(BaseAPIModel):
    """Complete dosing system status response."""
    tower_id: str = Field(..., description="Cooling tower identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Overall status
    system_status: str = Field(..., description="Overall system status")
    active_channels: int = Field(..., ge=0, description="Number of active channels")
    total_channels: int = Field(..., ge=0, description="Total number of channels")

    # Individual channels
    channels: List[DosingChannel] = Field(default_factory=list)

    # Alerts
    low_chemical_alerts: List[str] = Field(default_factory=list)
    pump_fault_alerts: List[str] = Field(default_factory=list)

    # Cost tracking
    daily_chemical_cost: float = Field(..., ge=0, description="Daily chemical cost USD")
    monthly_chemical_cost: float = Field(..., ge=0, description="Monthly chemical cost USD")

    class Config:
        schema_extra = {
            "example": {
                "tower_id": "tower-001",
                "timestamp": "2025-01-15T10:30:00Z",
                "system_status": "operational",
                "active_channels": 3,
                "total_channels": 4,
                "channels": [],
                "low_chemical_alerts": [],
                "pump_fault_alerts": [],
                "daily_chemical_cost": 45.50,
                "monthly_chemical_cost": 1365.00
            }
        }


# =============================================================================
# Compliance Models
# =============================================================================

class ConstraintStatus(BaseAPIModel):
    """Status of a single compliance constraint."""
    constraint_id: str = Field(..., description="Constraint identifier")
    constraint_name: str = Field(..., description="Constraint name")
    category: str = Field(..., description="Constraint category")

    # Current status
    status: ComplianceStatus = Field(..., description="Compliance status")
    current_value: float = Field(..., description="Current value")
    limit_value: float = Field(..., description="Limit value")
    margin_percent: float = Field(..., description="Margin to limit %")

    # Violation details
    in_violation: bool = Field(..., description="Whether currently in violation")
    violation_count_24h: int = Field(default=0, description="Violations in last 24h")
    last_violation: Optional[datetime] = Field(None, description="Last violation timestamp")


class ComplianceReportResponse(BaseAPIModel):
    """Comprehensive compliance status report."""
    tower_id: str = Field(..., description="Cooling tower identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    report_period_hours: int = Field(default=24, description="Report period in hours")

    # Overall compliance
    overall_status: ComplianceStatus = Field(..., description="Overall compliance status")
    compliance_score: float = Field(..., ge=0, le=100, description="Compliance score (0-100)")

    # Constraint details
    total_constraints: int = Field(..., description="Total constraints monitored")
    compliant_constraints: int = Field(..., description="Compliant constraints")
    warning_constraints: int = Field(..., description="Constraints in warning")
    violated_constraints: int = Field(..., description="Violated constraints")

    # Detailed constraints
    constraints: List[ConstraintStatus] = Field(default_factory=list)

    # Violation summary
    total_violations_24h: int = Field(default=0, description="Total violations in 24h")
    critical_violations: List[str] = Field(default_factory=list)

    # Regulatory tracking
    discharge_permit_status: str = Field(..., description="Discharge permit status")
    next_compliance_review: Optional[datetime] = Field(None)

    class Config:
        schema_extra = {
            "example": {
                "tower_id": "tower-001",
                "timestamp": "2025-01-15T10:30:00Z",
                "report_period_hours": 24,
                "overall_status": "compliant",
                "compliance_score": 98.5,
                "total_constraints": 15,
                "compliant_constraints": 14,
                "warning_constraints": 1,
                "violated_constraints": 0,
                "constraints": [],
                "total_violations_24h": 0,
                "critical_violations": [],
                "discharge_permit_status": "valid",
                "next_compliance_review": "2025-06-01T00:00:00Z"
            }
        }


# =============================================================================
# Savings Models
# =============================================================================

class SavingsMetric(BaseAPIModel):
    """Single savings metric."""
    metric_name: str = Field(..., description="Metric name")
    baseline_value: float = Field(..., description="Baseline value")
    current_value: float = Field(..., description="Current value")
    savings_value: float = Field(..., description="Savings value")
    savings_percent: float = Field(..., description="Savings percentage")
    unit: str = Field(..., description="Unit of measurement")
    monetary_value: Optional[float] = Field(None, description="Monetary value USD")


class SavingsReportResponse(BaseAPIModel):
    """Comprehensive savings report."""
    tower_id: str = Field(..., description="Cooling tower identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    period_start: datetime = Field(..., description="Report period start")
    period_end: datetime = Field(..., description="Report period end")

    # Water savings
    water_baseline_gallons: float = Field(..., description="Water baseline gallons")
    water_actual_gallons: float = Field(..., description="Water actual gallons")
    water_savings_gallons: float = Field(..., description="Water savings gallons")
    water_savings_percent: float = Field(..., description="Water savings %")
    water_cost_savings: float = Field(..., description="Water cost savings USD")

    # Energy savings
    energy_baseline_kwh: float = Field(..., description="Energy baseline kWh")
    energy_actual_kwh: float = Field(..., description="Energy actual kWh")
    energy_savings_kwh: float = Field(..., description="Energy savings kWh")
    energy_savings_percent: float = Field(..., description="Energy savings %")
    energy_cost_savings: float = Field(..., description="Energy cost savings USD")

    # Chemical savings
    chemical_baseline_cost: float = Field(..., description="Chemical baseline cost USD")
    chemical_actual_cost: float = Field(..., description="Chemical actual cost USD")
    chemical_savings: float = Field(..., description="Chemical savings USD")
    chemical_savings_percent: float = Field(..., description="Chemical savings %")

    # Emissions reduction
    co2_baseline_kg: float = Field(..., description="CO2 baseline kg")
    co2_actual_kg: float = Field(..., description="CO2 actual kg")
    co2_reduction_kg: float = Field(..., description="CO2 reduction kg")
    co2_reduction_percent: float = Field(..., description="CO2 reduction %")

    # Total savings
    total_cost_savings: float = Field(..., description="Total cost savings USD")
    total_savings_percent: float = Field(..., description="Total savings %")

    # Detailed metrics
    metrics: List[SavingsMetric] = Field(default_factory=list)

    # Projections
    projected_annual_savings: float = Field(..., description="Projected annual savings USD")
    projected_annual_water_savings: float = Field(..., description="Projected annual water gallons")
    projected_annual_co2_reduction: float = Field(..., description="Projected annual CO2 reduction kg")

    class Config:
        schema_extra = {
            "example": {
                "tower_id": "tower-001",
                "timestamp": "2025-01-15T10:30:00Z",
                "period_start": "2025-01-01T00:00:00Z",
                "period_end": "2025-01-15T00:00:00Z",
                "water_baseline_gallons": 100000,
                "water_actual_gallons": 85000,
                "water_savings_gallons": 15000,
                "water_savings_percent": 15.0,
                "water_cost_savings": 750.0,
                "energy_baseline_kwh": 50000,
                "energy_actual_kwh": 46000,
                "energy_savings_kwh": 4000,
                "energy_savings_percent": 8.0,
                "energy_cost_savings": 400.0,
                "chemical_baseline_cost": 2000,
                "chemical_actual_cost": 1760,
                "chemical_savings": 240,
                "chemical_savings_percent": 12.0,
                "co2_baseline_kg": 25000,
                "co2_actual_kg": 21250,
                "co2_reduction_kg": 3750,
                "co2_reduction_percent": 15.0,
                "total_cost_savings": 1390.0,
                "total_savings_percent": 11.6,
                "metrics": [],
                "projected_annual_savings": 33360.0,
                "projected_annual_water_savings": 360000,
                "projected_annual_co2_reduction": 90000
            }
        }


# =============================================================================
# Health Check Models
# =============================================================================

class ComponentHealth(BaseAPIModel):
    """Health status of a single component."""
    component: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Health status")
    message: Optional[str] = Field(None, description="Status message")
    latency_ms: Optional[float] = Field(None, description="Response latency in ms")
    last_check: datetime = Field(default_factory=datetime.utcnow)


class HealthCheckResponse(BaseAPIModel):
    """API health check response."""
    status: HealthStatus = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")

    # Component health
    components: List[ComponentHealth] = Field(default_factory=list)

    # System metrics
    cpu_usage_percent: Optional[float] = Field(None, description="CPU usage %")
    memory_usage_percent: Optional[float] = Field(None, description="Memory usage %")
    active_connections: Optional[int] = Field(None, description="Active connections")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-15T10:30:00Z",
                "version": "1.0.0",
                "uptime_seconds": 86400.0,
                "components": [
                    {"component": "database", "status": "healthy", "latency_ms": 5.2},
                    {"component": "redis", "status": "healthy", "latency_ms": 1.1},
                    {"component": "ml_engine", "status": "healthy", "latency_ms": 15.5}
                ],
                "cpu_usage_percent": 25.0,
                "memory_usage_percent": 45.0,
                "active_connections": 150
            }
        }


# =============================================================================
# Error Models
# =============================================================================

class ErrorDetail(BaseAPIModel):
    """Detailed error information."""
    field: Optional[str] = Field(None, description="Field with error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseAPIModel):
    """Standard API error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed errors")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Request validation failed",
                "details": [
                    {"field": "tower_id", "message": "Tower ID not found", "code": "not_found"}
                ],
                "request_id": "req-abc123",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


# =============================================================================
# Pagination Models
# =============================================================================

class PaginationParams(BaseAPIModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$", description="Sort order")


class PaginatedResponse(BaseAPIModel):
    """Base paginated response."""
    total: int = Field(..., description="Total items")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_previous: bool = Field(..., description="Has previous page")
