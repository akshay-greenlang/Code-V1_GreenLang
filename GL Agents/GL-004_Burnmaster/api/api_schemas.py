"""
GL-004 BURNMASTER API Schemas

Pydantic request/response schemas for the Burner Optimization API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class OperatingMode(str, Enum):
    NORMAL = "normal"
    ECO = "eco"
    HIGH_EFFICIENCY = "high_efficiency"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"


class BurnerState(str, Enum):
    RUNNING = "running"
    IDLE = "idle"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class RecommendationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendationStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    EXPIRED = "expired"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class OptimizationState(str, Enum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    APPLYING = "applying"
    MONITORING = "monitoring"
    ERROR = "error"


class UserRole(str, Enum):
    OPERATOR = "operator"
    ENGINEER = "engineer"
    ADMIN = "admin"


class BurnerMetrics(BaseModel):
    firing_rate: float = Field(..., ge=0, le=100, description="Firing rate percentage")
    fuel_flow_rate: float = Field(..., ge=0, description="Fuel flow rate (kg/h)")
    air_flow_rate: float = Field(..., ge=0, description="Air flow rate (m3/h)")
    combustion_air_temp: float = Field(..., description="Combustion air temperature (C)")
    flue_gas_temp: float = Field(..., description="Flue gas temperature (C)")
    oxygen_level: float = Field(..., ge=0, le=25, description="O2 level (%)")
    co_level: float = Field(..., ge=0, description="CO level (ppm)")
    nox_level: float = Field(..., ge=0, description="NOx level (ppm)")
    efficiency: float = Field(..., ge=0, le=100, description="Combustion efficiency (%)")
    heat_output: float = Field(..., ge=0, description="Heat output (MW)")


class BurnerStatusResponse(BaseModel):
    unit_id: str = Field(..., description="Unique unit identifier")
    name: str = Field(..., description="Unit display name")
    state: BurnerState = Field(..., description="Current operational state")
    mode: OperatingMode = Field(..., description="Current operating mode")
    metrics: BurnerMetrics = Field(..., description="Real-time metrics")
    uptime_hours: float = Field(..., ge=0, description="Hours since last startup")
    last_maintenance: Optional[datetime] = Field(None, description="Last maintenance timestamp")
    next_maintenance: Optional[datetime] = Field(None, description="Next scheduled maintenance")
    active_alerts_count: int = Field(default=0, ge=0, description="Number of active alerts")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Status timestamp")


class KPIValue(BaseModel):
    name: str = Field(..., description="KPI name")
    value: float = Field(..., description="Current value")
    unit: str = Field(..., description="Unit of measurement")
    target: Optional[float] = Field(None, description="Target value")
    trend: Optional[str] = Field(None, description="Trend direction: up, down, stable")
    change_percent: Optional[float] = Field(None, description="Change from previous period (%)")


class EmissionsKPIs(BaseModel):
    co2_emissions: KPIValue = Field(..., description="CO2 emissions")
    nox_emissions: KPIValue = Field(..., description="NOx emissions")
    co_emissions: KPIValue = Field(..., description="CO emissions")
    particulate_matter: KPIValue = Field(..., description="Particulate matter")
    carbon_intensity: KPIValue = Field(..., description="Carbon intensity")


class EfficiencyKPIs(BaseModel):
    thermal_efficiency: KPIValue = Field(..., description="Thermal efficiency")
    combustion_efficiency: KPIValue = Field(..., description="Combustion efficiency")
    fuel_utilization: KPIValue = Field(..., description="Fuel utilization rate")
    heat_recovery: KPIValue = Field(..., description="Heat recovery rate")
    overall_equipment_effectiveness: KPIValue = Field(..., description="OEE")


class OperationalKPIs(BaseModel):
    availability: KPIValue = Field(..., description="Equipment availability")
    reliability: KPIValue = Field(..., description="Equipment reliability")
    mean_time_between_failures: KPIValue = Field(..., description="MTBF")
    mean_time_to_repair: KPIValue = Field(..., description="MTTR")
    capacity_utilization: KPIValue = Field(..., description="Capacity utilization")


class KPIDashboardResponse(BaseModel):
    unit_id: str = Field(..., description="Unit identifier")
    period_start: datetime = Field(..., description="Reporting period start")
    period_end: datetime = Field(..., description="Reporting period end")
    emissions: EmissionsKPIs = Field(..., description="Emissions KPIs")
    efficiency: EfficiencyKPIs = Field(..., description="Efficiency KPIs")
    operational: OperationalKPIs = Field(..., description="Operational KPIs")
    overall_score: float = Field(..., ge=0, le=100, description="Overall performance score")
    comparison_baseline: Optional[str] = Field(None, description="Baseline for comparison")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RecommendationAction(BaseModel):
    action_id: str = Field(..., description="Action identifier")
    description: str = Field(..., description="Action description")
    parameter: str = Field(..., description="Parameter to adjust")
    current_value: float = Field(..., description="Current parameter value")
    recommended_value: float = Field(..., description="Recommended parameter value")
    unit: str = Field(..., description="Unit of measurement")
    auto_applicable: bool = Field(default=False, description="Can be auto-applied")


class RecommendationImpact(BaseModel):
    efficiency_improvement: Optional[float] = Field(None, description="Expected efficiency improvement (%)")
    emissions_reduction: Optional[float] = Field(None, description="Expected emissions reduction (%)")
    cost_savings: Optional[float] = Field(None, description="Expected cost savings ($/day)")
    energy_savings: Optional[float] = Field(None, description="Expected energy savings (kWh/day)")
    confidence_level: float = Field(..., ge=0, le=1, description="Confidence in prediction")


class RecommendationResponse(BaseModel):
    recommendation_id: str = Field(..., description="Unique recommendation ID")
    unit_id: str = Field(..., description="Target unit ID")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    priority: RecommendationPriority = Field(..., description="Priority level")
    status: RecommendationStatus = Field(..., description="Current status")
    category: str = Field(..., description="Recommendation category")
    actions: List[RecommendationAction] = Field(..., description="Recommended actions")
    impact: RecommendationImpact = Field(..., description="Expected impact")
    reasoning: str = Field(..., description="AI reasoning explanation")
    model_version: str = Field(..., description="Optimization model version")
    valid_until: datetime = Field(..., description="Recommendation validity period")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accepted_at: Optional[datetime] = Field(None)
    implemented_at: Optional[datetime] = Field(None)
    accepted_by: Optional[str] = Field(None, description="User who accepted")


class AcceptRecommendationRequest(BaseModel):
    auto_implement: bool = Field(default=False, description="Auto-implement if possible")
    scheduled_time: Optional[datetime] = Field(None, description="Schedule implementation")
    notes: Optional[str] = Field(None, max_length=500, description="Operator notes")
    override_safety_check: bool = Field(default=False, description="Override safety checks (admin only)")


class AcceptRecommendationResponse(BaseModel):
    recommendation_id: str = Field(..., description="Recommendation ID")
    status: RecommendationStatus = Field(..., description="New status")
    implementation_status: str = Field(..., description="Implementation status")
    scheduled_time: Optional[datetime] = Field(None)
    estimated_completion: Optional[datetime] = Field(None)
    accepted_by: str = Field(..., description="User who accepted")
    accepted_at: datetime = Field(default_factory=datetime.utcnow)


class OptimizationMetrics(BaseModel):
    recommendations_generated: int = Field(..., ge=0)
    recommendations_accepted: int = Field(..., ge=0)
    recommendations_implemented: int = Field(..., ge=0)
    average_confidence: float = Field(..., ge=0, le=1)
    total_savings_achieved: float = Field(..., ge=0, description="Total savings ($)")
    efficiency_improvement: float = Field(..., description="Efficiency improvement (%)")


class OptimizationStatusResponse(BaseModel):
    unit_id: str = Field(..., description="Unit identifier")
    state: OptimizationState = Field(..., description="Current optimization state")
    is_active: bool = Field(..., description="Whether optimization is active")
    last_analysis: Optional[datetime] = Field(None)
    next_analysis: Optional[datetime] = Field(None)
    analysis_interval_minutes: int = Field(..., ge=1)
    metrics: OptimizationMetrics = Field(..., description="Performance metrics")
    active_model: str = Field(..., description="Active optimization model")
    model_accuracy: float = Field(..., ge=0, le=1, description="Model accuracy score")
    data_quality_score: float = Field(..., ge=0, le=1, description="Input data quality")
    constraints_active: List[str] = Field(default=[], description="Active constraints")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModeChangeRequest(BaseModel):
    new_mode: OperatingMode = Field(..., description="Target operating mode")
    reason: str = Field(..., min_length=10, max_length=500, description="Reason for change")
    scheduled_time: Optional[datetime] = Field(None, description="Schedule mode change")
    transition_duration_minutes: Optional[int] = Field(None, ge=1, le=60, description="Transition duration")
    notify_operators: bool = Field(default=True, description="Send notifications")


class ModeChangeResponse(BaseModel):
    unit_id: str = Field(..., description="Unit identifier")
    previous_mode: OperatingMode = Field(..., description="Previous mode")
    new_mode: OperatingMode = Field(..., description="New mode")
    status: str = Field(..., description="Change status")
    scheduled_time: Optional[datetime] = Field(None)
    estimated_completion: Optional[datetime] = Field(None)
    transition_steps: List[str] = Field(default=[], description="Transition steps")
    changed_by: str = Field(..., description="User who initiated")
    changed_at: datetime = Field(default_factory=datetime.utcnow)


class HistoryRequest(BaseModel):
    start_time: datetime = Field(..., description="Start of time range")
    end_time: datetime = Field(..., description="End of time range")
    metrics: List[str] = Field(default=["all"], description="Metrics to retrieve")
    resolution: str = Field(default="1h", description="Data resolution: 1m, 5m, 15m, 1h, 1d")
    aggregation: str = Field(default="avg", description="Aggregation: avg, min, max, sum")


class HistoryDataPoint(BaseModel):
    timestamp: datetime = Field(..., description="Data point timestamp")
    values: Dict[str, float] = Field(..., description="Metric values")


class HistoryResponse(BaseModel):
    unit_id: str = Field(..., description="Unit identifier")
    start_time: datetime = Field(..., description="Actual start time")
    end_time: datetime = Field(..., description="Actual end time")
    resolution: str = Field(..., description="Data resolution")
    metrics: List[str] = Field(..., description="Included metrics")
    data_points: List[HistoryDataPoint] = Field(..., description="Historical data")
    total_points: int = Field(..., ge=0)
    statistics: Dict[str, Dict[str, float]] = Field(default={}, description="Summary statistics")


class AlertResponse(BaseModel):
    alert_id: str = Field(..., description="Unique alert ID")
    unit_id: str = Field(..., description="Affected unit ID")
    severity: AlertSeverity = Field(..., description="Alert severity")
    status: AlertStatus = Field(..., description="Alert status")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed description")
    source: str = Field(..., description="Alert source/trigger")
    metric_name: Optional[str] = Field(None, description="Related metric")
    metric_value: Optional[float] = Field(None, description="Triggering value")
    threshold: Optional[float] = Field(None, description="Threshold value")
    recommended_action: Optional[str] = Field(None, description="Recommended action")
    acknowledged_by: Optional[str] = Field(None)
    acknowledged_at: Optional[datetime] = Field(None)
    resolved_at: Optional[datetime] = Field(None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AlertAcknowledgeRequest(BaseModel):
    notes: Optional[str] = Field(None, max_length=500, description="Acknowledgement notes")
    suppress_duration_minutes: Optional[int] = Field(None, ge=1, le=1440, description="Suppress for duration")


class ServiceHealth(BaseModel):
    name: str = Field(..., description="Service name")
    status: str = Field(..., description="Status: healthy, degraded, unhealthy")
    latency_ms: Optional[float] = Field(None, description="Response latency")
    last_check: datetime = Field(default_factory=datetime.utcnow)
    message: Optional[str] = Field(None, description="Status message")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment name")
    uptime_seconds: float = Field(..., ge=0, description="Service uptime")
    services: List[ServiceHealth] = Field(default=[], description="Component health")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Message type")
    unit_id: str = Field(..., description="Unit identifier")
    data: Any = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
