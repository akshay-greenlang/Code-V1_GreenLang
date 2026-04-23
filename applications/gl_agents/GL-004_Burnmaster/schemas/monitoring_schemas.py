"""
Monitoring Schemas - Data models for monitoring and alerting.

This module defines Pydantic models for alerts, alert levels, KPI dashboards,
and health status monitoring. These schemas support real-time monitoring
and operational awareness for combustion systems.

Example:
    >>> from monitoring_schemas import Alert, AlertLevel, KPIDashboard
    >>> alert = Alert(
    ...     level=AlertLevel.WARNING,
    ...     message="O2 approaching high limit",
    ...     parameter="o2_percent"
    ... )
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, computed_field
import hashlib


class AlertLevel(str, Enum):
    """
    Alert severity level classification.

    INFO: Informational message, no action required
    WARNING: Attention needed, potential issue developing
    CRITICAL: Immediate attention required, significant issue
    EMERGENCY: Emergency condition, immediate response required
    """
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(str, Enum):
    """Category of alert."""
    PROCESS = "process"           # Process conditions
    EQUIPMENT = "equipment"       # Equipment status
    SAFETY = "safety"            # Safety-related
    EMISSIONS = "emissions"       # Emissions compliance
    PERFORMANCE = "performance"   # Performance degradation
    COMMUNICATION = "communication"  # Communication failures
    SYSTEM = "system"            # System/software issues


class AlertState(str, Enum):
    """Current state of an alert."""
    ACTIVE = "active"           # Alert is currently active
    ACKNOWLEDGED = "acknowledged"  # Alert has been acknowledged
    CLEARED = "cleared"         # Alert condition has cleared
    SHELVED = "shelved"         # Alert is temporarily shelved
    SUPPRESSED = "suppressed"   # Alert is suppressed


class ComponentStatus(str, Enum):
    """Health status of a component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class TrendDirection(str, Enum):
    """Direction of a metric trend."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


class Alert(BaseModel):
    """
    Alert record for monitoring system.

    Represents an alert or alarm condition with full lifecycle tracking
    including acknowledgement and clearing.

    Attributes:
        id: Unique alert identifier
        level: Severity level of the alert
        message: Human-readable alert message
        timestamp: When the alert was generated
        acknowledged: Whether alert has been acknowledged
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    alert_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Unique alert identifier")
    level: AlertLevel = Field(..., description="Alert severity level")
    category: AlertCategory = Field(default=AlertCategory.PROCESS, description="Alert category")
    state: AlertState = Field(default=AlertState.ACTIVE, description="Current alert state")

    # Alert content
    message: str = Field(..., min_length=1, max_length=500, description="Human-readable alert message")
    description: str = Field(default="", max_length=2000, description="Detailed alert description")
    parameter: Optional[str] = Field(default=None, max_length=100, description="Related parameter name")
    value: Optional[float] = Field(default=None, description="Value that triggered the alert")
    threshold: Optional[float] = Field(default=None, description="Threshold that was exceeded")
    unit: str = Field(default="", max_length=50, description="Engineering unit")

    # Location/source
    equipment_id: Optional[str] = Field(default=None, max_length=100, description="Related equipment identifier")
    source_system: str = Field(default="BURNMASTER", max_length=100, description="System that generated the alert")
    tag: Optional[str] = Field(default=None, max_length=100, description="Related tag/point")

    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When alert was generated")
    acknowledged_at: Optional[datetime] = Field(default=None, description="When alert was acknowledged")
    cleared_at: Optional[datetime] = Field(default=None, description="When alert condition cleared")

    # Acknowledgement
    acknowledged: bool = Field(default=False, description="Whether alert has been acknowledged")
    acknowledged_by: Optional[str] = Field(default=None, max_length=100, description="User who acknowledged")
    acknowledgement_comment: Optional[str] = Field(default=None, max_length=500, description="Acknowledgement comment")

    # Priority and escalation
    priority: int = Field(default=5, ge=1, le=10, description="Alert priority (1=highest)")
    escalation_level: int = Field(default=0, ge=0, le=5, description="Current escalation level")
    escalate_after_minutes: Optional[int] = Field(default=None, ge=0, description="Minutes before escalation")

    # Suppression
    suppressed: bool = Field(default=False, description="Whether alert is suppressed")
    suppression_reason: Optional[str] = Field(default=None, max_length=500, description="Reason for suppression")
    suppression_expires: Optional[datetime] = Field(default=None, description="When suppression expires")

    # Related information
    related_alerts: List[str] = Field(default_factory=list, description="IDs of related alerts")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")

    # Occurrence tracking
    occurrence_count: int = Field(default=1, ge=1, description="Number of occurrences")
    first_occurrence: Optional[datetime] = Field(default=None, description="First occurrence time")

    @computed_field
    @property
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.state == AlertState.ACTIVE

    @computed_field
    @property
    def duration_minutes(self) -> float:
        """Calculate alert duration in minutes."""
        end_time = self.cleared_at or datetime.utcnow()
        return (end_time - self.timestamp).total_seconds() / 60.0

    @computed_field
    @property
    def needs_escalation(self) -> bool:
        """Check if alert needs escalation."""
        if self.escalate_after_minutes is None or self.acknowledged:
            return False
        minutes_active = (datetime.utcnow() - self.timestamp).total_seconds() / 60.0
        return minutes_active > self.escalate_after_minutes

    def acknowledge(self, user: str, comment: Optional[str] = None) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()
        self.acknowledgement_comment = comment
        self.state = AlertState.ACKNOWLEDGED

    def clear(self) -> None:
        """Clear the alert."""
        self.cleared_at = datetime.utcnow()
        self.state = AlertState.CLEARED

    def shelve(self, duration_minutes: int, reason: str) -> None:
        """Shelve the alert temporarily."""
        self.state = AlertState.SHELVED
        self.suppressed = True
        self.suppression_reason = reason
        self.suppression_expires = datetime.utcnow() + timedelta(minutes=duration_minutes)


class AlertSummary(BaseModel):
    """Summary of alert statistics."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Summary timestamp")
    period_hours: float = Field(default=24.0, description="Period covered by summary")

    # Counts by level
    info_count: int = Field(default=0, ge=0, description="Number of INFO alerts")
    warning_count: int = Field(default=0, ge=0, description="Number of WARNING alerts")
    critical_count: int = Field(default=0, ge=0, description="Number of CRITICAL alerts")
    emergency_count: int = Field(default=0, ge=0, description="Number of EMERGENCY alerts")

    # Counts by state
    active_count: int = Field(default=0, ge=0, description="Number of active alerts")
    acknowledged_count: int = Field(default=0, ge=0, description="Number of acknowledged alerts")
    unacknowledged_count: int = Field(default=0, ge=0, description="Number of unacknowledged alerts")

    # Top alerts
    top_alert_parameters: List[str] = Field(default_factory=list, description="Most frequent alert parameters")
    top_alert_equipment: List[str] = Field(default_factory=list, description="Equipment with most alerts")

    # Response metrics
    mean_time_to_acknowledge_minutes: Optional[float] = Field(default=None, description="Average acknowledgement time")
    mean_time_to_clear_minutes: Optional[float] = Field(default=None, description="Average clearing time")

    @computed_field
    @property
    def total_count(self) -> int:
        """Total alert count."""
        return self.info_count + self.warning_count + self.critical_count + self.emergency_count


class MetricValue(BaseModel):
    """Single metric value with context."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    name: str = Field(..., min_length=1, max_length=100, description="Metric name")
    value: float = Field(..., description="Current metric value")
    unit: str = Field(default="", max_length=50, description="Engineering unit")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Measurement timestamp")

    # Context
    target: Optional[float] = Field(default=None, description="Target value")
    min_acceptable: Optional[float] = Field(default=None, description="Minimum acceptable value")
    max_acceptable: Optional[float] = Field(default=None, description="Maximum acceptable value")

    # Trend
    trend_direction: TrendDirection = Field(default=TrendDirection.UNKNOWN, description="Trend direction")
    change_percent_1h: Optional[float] = Field(default=None, description="Percentage change in last hour")
    change_percent_24h: Optional[float] = Field(default=None, description="Percentage change in last 24 hours")

    # Status
    is_at_target: bool = Field(default=True, description="Whether value is at target")
    is_acceptable: bool = Field(default=True, description="Whether value is acceptable")

    @computed_field
    @property
    def deviation_from_target(self) -> Optional[float]:
        """Calculate deviation from target."""
        if self.target is not None:
            return self.value - self.target
        return None

    @computed_field
    @property
    def deviation_percent(self) -> Optional[float]:
        """Calculate percentage deviation from target."""
        if self.target is not None and self.target != 0:
            return ((self.value - self.target) / self.target) * 100.0
        return None


class KPIDashboard(BaseModel):
    """
    Key Performance Indicator dashboard for combustion system.

    Aggregates key metrics for operational monitoring including
    fuel intensity, emissions, stability, and efficiency.

    Attributes:
        fuel_intensity: Fuel consumption intensity metric
        emissions: Emissions metrics (CO2, NOx, CO)
        stability: Combustion stability score
        efficiency: Thermal efficiency percentage
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    dashboard_id: str = Field(default="main", description="Dashboard identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Dashboard update timestamp")
    burner_id: Optional[str] = Field(default=None, max_length=50, description="Associated burner ID")

    # Fuel metrics
    fuel_intensity_mj_per_unit: MetricValue = Field(..., description="Fuel intensity per production unit")
    fuel_flow_kg_per_s: MetricValue = Field(..., description="Current fuel flow rate")
    fuel_cost_per_hour: Optional[MetricValue] = Field(default=None, description="Fuel cost rate")

    # Emissions metrics
    co2_kg_per_hour: MetricValue = Field(..., description="CO2 emissions rate")
    nox_kg_per_hour: MetricValue = Field(..., description="NOx emissions rate")
    co_ppm: MetricValue = Field(..., description="CO concentration")
    nox_ppm: MetricValue = Field(..., description="NOx concentration")
    o2_percent: MetricValue = Field(..., description="Flue gas O2")

    # Stability metrics
    stability_score: MetricValue = Field(..., description="Overall stability score (0-1)")
    flame_stability: MetricValue = Field(..., description="Flame stability metric")

    # Efficiency metrics
    thermal_efficiency_percent: MetricValue = Field(..., description="Thermal efficiency")
    combustion_efficiency_percent: MetricValue = Field(..., description="Combustion efficiency")

    # Operational metrics
    operating_load_percent: MetricValue = Field(..., description="Current operating load")
    excess_air_percent: MetricValue = Field(..., description="Excess air percentage")

    # System health
    data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Data quality score")
    last_optimization_timestamp: Optional[datetime] = Field(default=None, description="Last optimization run")

    # Summary statistics
    uptime_hours_24h: Optional[float] = Field(default=None, ge=0.0, le=24.0, description="Uptime in last 24 hours")
    efficiency_average_24h: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Average efficiency (24h)")
    fuel_consumed_24h_mj: Optional[float] = Field(default=None, ge=0.0, description="Total fuel consumed (24h)")
    co2_emitted_24h_kg: Optional[float] = Field(default=None, ge=0.0, description="Total CO2 emitted (24h)")

    @computed_field
    @property
    def overall_status(self) -> str:
        """Calculate overall system status."""
        if self.stability_score.value < 0.5:
            return "critical"
        elif self.stability_score.value < 0.7:
            return "warning"
        elif self.thermal_efficiency_percent.value < 85.0:
            return "suboptimal"
        return "optimal"

    def get_all_metrics(self) -> Dict[str, MetricValue]:
        """Get all metrics as a dictionary."""
        metrics = {}
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, MetricValue):
                metrics[field_name] = value
        return metrics


class HealthCheck(BaseModel):
    """Individual health check result."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    check_name: str = Field(..., min_length=1, max_length=100, description="Health check name")
    status: ComponentStatus = Field(..., description="Check result status")
    message: str = Field(default="", max_length=500, description="Status message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    duration_ms: Optional[float] = Field(default=None, ge=0.0, description="Check duration")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional check details")


class HealthStatus(BaseModel):
    """
    Health status of a system component.

    Captures the health state of a component with diagnostic
    information and last check timestamp.

    Attributes:
        component: Name of the component
        status: Current health status
        last_check: When health was last checked
        message: Status message or error details
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    component: str = Field(..., min_length=1, max_length=100, description="Component name")
    component_type: str = Field(default="unknown", max_length=50, description="Type of component")
    status: ComponentStatus = Field(..., description="Current health status")

    # Check results
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last health check timestamp")
    next_check: Optional[datetime] = Field(default=None, description="Next scheduled check")
    message: str = Field(default="", max_length=500, description="Status message or error")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional status details")

    # Health checks
    health_checks: List[HealthCheck] = Field(default_factory=list, description="Individual health check results")

    # Metrics
    uptime_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Uptime percentage")
    response_time_ms: Optional[float] = Field(default=None, ge=0.0, description="Response time")
    error_rate_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Error rate")
    last_error: Optional[str] = Field(default=None, max_length=500, description="Last error message")
    last_error_time: Optional[datetime] = Field(default=None, description="Last error timestamp")

    # Version info
    version: Optional[str] = Field(default=None, max_length=50, description="Component version")
    config_version: Optional[str] = Field(default=None, max_length=50, description="Configuration version")

    @computed_field
    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == ComponentStatus.HEALTHY

    @computed_field
    @property
    def is_operational(self) -> bool:
        """Check if component is operational (healthy or degraded)."""
        return self.status in [ComponentStatus.HEALTHY, ComponentStatus.DEGRADED]

    @computed_field
    @property
    def check_age_minutes(self) -> float:
        """Calculate how old the last check is."""
        return (datetime.utcnow() - self.last_check).total_seconds() / 60.0


class SystemHealthSummary(BaseModel):
    """Summary of overall system health."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Summary timestamp")
    system_name: str = Field(default="BURNMASTER", max_length=100, description="System name")

    # Overall status
    overall_status: ComponentStatus = Field(..., description="Overall system status")
    operational: bool = Field(default=True, description="Whether system is operational")

    # Component health
    components: List[HealthStatus] = Field(default_factory=list, description="Individual component health")
    healthy_count: int = Field(default=0, ge=0, description="Number of healthy components")
    degraded_count: int = Field(default=0, ge=0, description="Number of degraded components")
    unhealthy_count: int = Field(default=0, ge=0, description="Number of unhealthy components")

    # Key metrics
    cpu_usage_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="CPU usage")
    memory_usage_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Memory usage")
    disk_usage_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Disk usage")
    network_latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Network latency")

    # Alerts
    active_alerts: int = Field(default=0, ge=0, description="Number of active alerts")
    critical_alerts: int = Field(default=0, ge=0, description="Number of critical alerts")

    # Message
    status_message: str = Field(default="", max_length=500, description="Overall status message")
    issues: List[str] = Field(default_factory=list, description="Current issues")

    @computed_field
    @property
    def total_components(self) -> int:
        """Total number of components."""
        return self.healthy_count + self.degraded_count + self.unhealthy_count

    @computed_field
    @property
    def health_percentage(self) -> float:
        """Calculate health percentage."""
        if self.total_components == 0:
            return 100.0
        return (self.healthy_count / self.total_components) * 100.0


class MonitoringConfig(BaseModel):
    """Configuration for monitoring system."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    config_id: str = Field(default="default", description="Configuration identifier")

    # Alert settings
    alert_retention_days: int = Field(default=90, ge=1, le=365, description="Days to retain alerts")
    auto_clear_info_alerts_hours: Optional[int] = Field(default=24, ge=1, description="Auto-clear INFO alerts after hours")
    escalation_enabled: bool = Field(default=True, description="Enable alert escalation")
    max_escalation_level: int = Field(default=3, ge=1, le=5, description="Maximum escalation level")

    # Health check settings
    health_check_interval_s: float = Field(default=60.0, ge=10.0, le=3600.0, description="Health check interval")
    stale_check_threshold_minutes: float = Field(default=5.0, ge=1.0, description="Minutes before check is stale")

    # Dashboard settings
    dashboard_refresh_interval_s: float = Field(default=5.0, ge=1.0, le=60.0, description="Dashboard refresh interval")
    trend_window_hours: int = Field(default=24, ge=1, le=168, description="Hours of trend data to show")

    # Notification settings
    notification_enabled: bool = Field(default=True, description="Enable notifications")
    notification_endpoints: List[str] = Field(default_factory=list, description="Notification endpoints")


__all__ = [
    "AlertLevel",
    "AlertCategory",
    "AlertState",
    "ComponentStatus",
    "TrendDirection",
    "Alert",
    "AlertSummary",
    "MetricValue",
    "KPIDashboard",
    "HealthCheck",
    "HealthStatus",
    "SystemHealthSummary",
    "MonitoringConfig",
]
