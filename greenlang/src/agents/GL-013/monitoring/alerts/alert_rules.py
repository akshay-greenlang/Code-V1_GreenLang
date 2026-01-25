"""
GL-013 PREDICTMAINT - Alert Rules Configuration

Comprehensive predictive maintenance alerting configuration with rules
for equipment health, failure prediction, vibration analysis, temperature
monitoring, and system performance.

Key Features:
    - 35+ predefined alert rules covering all maintenance scenarios
    - Severity levels: INFO, WARNING, HIGH, CRITICAL, EMERGENCY
    - PromQL expressions for Prometheus integration
    - Configurable thresholds and durations
    - Rich annotations for alert context

Alert Categories:
    - Equipment Health: Health index degradation alerts
    - Failure Prediction: RUL and failure probability alerts
    - Vibration Analysis: ISO 10816 zone violations
    - Temperature: Thermal limit exceedances
    - Anomaly Detection: ML-based anomaly alerts
    - Maintenance: Scheduling and overdue alerts
    - System: Performance and integration alerts

Standards Compliance:
    - ISO 10816: Vibration severity thresholds
    - IEC 60085: Temperature limits by insulation class
    - ISO 13381: Prognostics alert thresholds

Example:
    >>> from gl_013.monitoring.alerts.alert_rules import ALERT_RULES, AlertSeverity
    >>> critical_rules = [r for r in ALERT_RULES if r.severity == AlertSeverity.CRITICAL]
    >>> print(f"Critical rules: {len(critical_rules)}")

Author: GL-MonitoringEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import List, Dict, Optional, Callable, Any, Union


# =============================================================================
# ENUMS
# =============================================================================

class AlertSeverity(str, Enum):
    """
    Alert severity levels following standard incident management practices.

    INFO: Informational, no action required
    WARNING: Potential issue, monitor closely
    HIGH: Significant issue, action required soon
    CRITICAL: Major issue, immediate action required
    EMERGENCY: System or safety critical, immediate escalation
    """
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(str, Enum):
    """Alert categories for grouping and routing."""
    EQUIPMENT_HEALTH = "equipment_health"
    FAILURE_PREDICTION = "failure_prediction"
    VIBRATION = "vibration"
    TEMPERATURE = "temperature"
    ANOMALY = "anomaly"
    MAINTENANCE = "maintenance"
    INTEGRATION = "integration"
    SYSTEM = "system"


class NotificationChannel(str, Enum):
    """Notification channels for alert delivery."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"
    OPSGENIE = "opsgenie"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class AlertThreshold:
    """
    Alert threshold configuration.

    Attributes:
        warning: Warning threshold value
        critical: Critical threshold value
        emergency: Emergency threshold value (optional)
    """
    warning: float
    critical: float
    emergency: Optional[float] = None


@dataclass
class AlertRule:
    """
    Prometheus alert rule definition.

    Attributes:
        name: Unique alert name (PascalCase)
        description: Human-readable description
        severity: Alert severity level
        category: Alert category for grouping
        condition: PromQL expression for alert condition
        for_duration: Duration condition must be true before firing
        labels: Additional labels for routing and identification
        annotations: Rich context for alert handlers
        runbook_url: Link to runbook for this alert
        enabled: Whether this rule is active
        notification_channels: Channels to notify for this alert
    """
    name: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    condition: str
    for_duration: timedelta
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    runbook_url: Optional[str] = None
    enabled: bool = True
    notification_channels: List[NotificationChannel] = field(default_factory=list)

    def to_prometheus_rule(self) -> Dict[str, Any]:
        """
        Convert to Prometheus alert rule format.

        Returns:
            Dictionary in Prometheus alerting rules format
        """
        rule = {
            "alert": self.name,
            "expr": self.condition,
            "for": f"{int(self.for_duration.total_seconds())}s",
            "labels": {
                "severity": self.severity.value,
                "category": self.category.value,
                "agent": "gl013",
                **self.labels
            },
            "annotations": {
                "description": self.description,
                **self.annotations
            }
        }

        if self.runbook_url:
            rule["annotations"]["runbook_url"] = self.runbook_url

        return rule

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category.value,
            "condition": self.condition,
            "for_duration_seconds": int(self.for_duration.total_seconds()),
            "labels": self.labels,
            "annotations": self.annotations,
            "runbook_url": self.runbook_url,
            "enabled": self.enabled,
            "notification_channels": [c.value for c in self.notification_channels]
        }


@dataclass
class AlertGroup:
    """
    Group of related alert rules.

    Attributes:
        name: Group name
        interval: Evaluation interval
        rules: List of alert rules in this group
    """
    name: str
    interval: timedelta
    rules: List[AlertRule]

    def to_prometheus_group(self) -> Dict[str, Any]:
        """
        Convert to Prometheus rule group format.

        Returns:
            Dictionary in Prometheus rule group format
        """
        return {
            "name": self.name,
            "interval": f"{int(self.interval.total_seconds())}s",
            "rules": [rule.to_prometheus_rule() for rule in self.rules if rule.enabled]
        }


# =============================================================================
# ALERT RULES DEFINITIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Equipment Health Alerts
# -----------------------------------------------------------------------------

EQUIPMENT_HEALTH_RULES = [
    AlertRule(
        name="EquipmentHealthCritical",
        description="Equipment health index has dropped below critical threshold (25%), indicating imminent failure risk.",
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.EQUIPMENT_HEALTH,
        condition="gl013_equipment_health_index < 25",
        for_duration=timedelta(minutes=5),
        labels={
            "team": "maintenance",
            "priority": "P1",
            "impact": "high"
        },
        annotations={
            "summary": "Equipment {{ $labels.equipment_id }} health critical at {{ $value | printf \"%.1f\" }}%",
            "description": "Health index for {{ $labels.equipment_id }} ({{ $labels.equipment_type }}) has fallen to {{ $value | printf \"%.1f\" }}%. Immediate inspection and maintenance required.",
            "action": "Initiate emergency maintenance procedure. Check vibration, temperature, and lubrication."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/equipment-health-critical",
        notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SLACK]
    ),
    AlertRule(
        name="EquipmentHealthWarning",
        description="Equipment health index has dropped below warning threshold (50%), indicating degradation.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.EQUIPMENT_HEALTH,
        condition="gl013_equipment_health_index < 50 and gl013_equipment_health_index >= 25",
        for_duration=timedelta(minutes=15),
        labels={
            "team": "maintenance",
            "priority": "P2",
            "impact": "medium"
        },
        annotations={
            "summary": "Equipment {{ $labels.equipment_id }} health degraded to {{ $value | printf \"%.1f\" }}%",
            "description": "Health index for {{ $labels.equipment_id }} is degrading. Schedule maintenance within 7 days.",
            "action": "Review maintenance history and schedule inspection."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/equipment-health-warning",
        notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
    ),
    AlertRule(
        name="EquipmentHealthDeclining",
        description="Equipment health index is declining rapidly.",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.EQUIPMENT_HEALTH,
        condition="delta(gl013_equipment_health_index[1h]) < -5",
        for_duration=timedelta(minutes=30),
        labels={
            "team": "maintenance",
            "priority": "P2",
            "impact": "medium"
        },
        annotations={
            "summary": "Equipment {{ $labels.equipment_id }} health declining rapidly",
            "description": "Health index dropped by {{ $value | printf \"%.1f\" }}% in the last hour. Investigate root cause.",
            "action": "Check recent operational changes and sensor readings."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/equipment-health-declining",
        notification_channels=[NotificationChannel.SLACK]
    ),
]

# -----------------------------------------------------------------------------
# Failure Prediction Alerts
# -----------------------------------------------------------------------------

FAILURE_PREDICTION_RULES = [
    AlertRule(
        name="RULCriticallyLow",
        description="Remaining Useful Life is critically low (less than 7 days).",
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.FAILURE_PREDICTION,
        condition="gl013_equipment_rul_days < 7",
        for_duration=timedelta(minutes=5),
        labels={
            "team": "maintenance",
            "priority": "P1",
            "impact": "high"
        },
        annotations={
            "summary": "Equipment {{ $labels.equipment_id }} RUL critical: {{ $value | printf \"%.1f\" }} days remaining",
            "description": "Predicted failure within {{ $value | printf \"%.1f\" }} days for {{ $labels.equipment_id }}. Immediate maintenance required.",
            "action": "Schedule emergency maintenance. Prepare spare parts and backup equipment."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/rul-critical",
        notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SLACK, NotificationChannel.EMAIL]
    ),
    AlertRule(
        name="RULLow",
        description="Remaining Useful Life is low (less than 30 days).",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.FAILURE_PREDICTION,
        condition="gl013_equipment_rul_days < 30 and gl013_equipment_rul_days >= 7",
        for_duration=timedelta(minutes=15),
        labels={
            "team": "maintenance",
            "priority": "P2",
            "impact": "medium"
        },
        annotations={
            "summary": "Equipment {{ $labels.equipment_id }} RUL low: {{ $value | printf \"%.1f\" }} days remaining",
            "description": "Schedule maintenance for {{ $labels.equipment_id }} within {{ $value | printf \"%.0f\" }} days.",
            "action": "Plan maintenance window and order spare parts if needed."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/rul-low",
        notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
    ),
    AlertRule(
        name="HighFailureProbability30Day",
        description="30-day failure probability exceeds 50%.",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.FAILURE_PREDICTION,
        condition="gl013_failure_probability_30d > 0.5",
        for_duration=timedelta(minutes=10),
        labels={
            "team": "maintenance",
            "priority": "P1",
            "impact": "high"
        },
        annotations={
            "summary": "Equipment {{ $labels.equipment_id }} has {{ $value | printf \"%.0f\" }}% chance of failure in 30 days",
            "description": "High failure probability detected. Recommend immediate inspection and preventive maintenance.",
            "action": "Review failure mode analysis and schedule appropriate maintenance."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/high-failure-probability",
        notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
    ),
    AlertRule(
        name="FailureProbabilityIncreasing",
        description="Failure probability is increasing rapidly.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.FAILURE_PREDICTION,
        condition="delta(gl013_failure_probability_30d[24h]) > 0.1",
        for_duration=timedelta(hours=1),
        labels={
            "team": "maintenance",
            "priority": "P2",
            "impact": "medium"
        },
        annotations={
            "summary": "Failure probability increasing for {{ $labels.equipment_id }}",
            "description": "Failure probability increased by {{ $value | printf \"%.0f\" }}% in 24 hours.",
            "action": "Investigate degradation factors and adjust maintenance schedule."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
    AlertRule(
        name="ReliabilityBelowThreshold",
        description="Equipment reliability has dropped below acceptable threshold.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.FAILURE_PREDICTION,
        condition="gl013_equipment_reliability < 0.7",
        for_duration=timedelta(minutes=30),
        labels={
            "team": "maintenance",
            "priority": "P2"
        },
        annotations={
            "summary": "Reliability below threshold for {{ $labels.equipment_id }}: {{ $value | printf \"%.2f\" }}",
            "description": "Current reliability R(t) = {{ $value | printf \"%.2f\" }} is below the 0.7 threshold.",
            "action": "Review reliability model and maintenance history."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
]

# -----------------------------------------------------------------------------
# Vibration Analysis Alerts
# -----------------------------------------------------------------------------

VIBRATION_RULES = [
    AlertRule(
        name="VibrationZoneDanger",
        description="Vibration level has reached ISO 10816 Zone D (Danger).",
        severity=AlertSeverity.EMERGENCY,
        category=AlertCategory.VIBRATION,
        condition="gl013_vibration_zone == 4",
        for_duration=timedelta(minutes=1),
        labels={
            "team": "maintenance",
            "priority": "P0",
            "impact": "critical",
            "safety": "true"
        },
        annotations={
            "summary": "DANGER: {{ $labels.equipment_id }} vibration in Zone D - STOP IMMEDIATELY",
            "description": "ISO 10816 Zone D violation. Machine may cause damage. Immediate shutdown required.",
            "action": "STOP MACHINE IMMEDIATELY. Do not restart until inspection complete."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/vibration-zone-d",
        notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SMS, NotificationChannel.SLACK]
    ),
    AlertRule(
        name="VibrationZoneAlert",
        description="Vibration level has reached ISO 10816 Zone C (Alert).",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.VIBRATION,
        condition="gl013_vibration_zone == 3",
        for_duration=timedelta(minutes=5),
        labels={
            "team": "maintenance",
            "priority": "P1",
            "impact": "high"
        },
        annotations={
            "summary": "Alert: {{ $labels.equipment_id }} vibration in Zone C",
            "description": "ISO 10816 Zone C - Short-term operation only. Schedule maintenance within 48 hours.",
            "action": "Increase monitoring frequency. Plan maintenance window."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/vibration-zone-c",
        notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SLACK]
    ),
    AlertRule(
        name="VibrationZoneMarginLow",
        description="Margin to next vibration zone is critically low.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.VIBRATION,
        condition="gl013_vibration_zone_margin_mm_s < 0.5",
        for_duration=timedelta(minutes=15),
        labels={
            "team": "maintenance",
            "priority": "P2"
        },
        annotations={
            "summary": "Low margin to next vibration zone for {{ $labels.equipment_id }}",
            "description": "Only {{ $value | printf \"%.2f\" }} mm/s margin to next zone boundary.",
            "action": "Monitor closely and investigate root cause of vibration increase."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
    AlertRule(
        name="VibrationTrendingUp",
        description="Vibration levels are trending upward.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.VIBRATION,
        condition="gl013_vibration_trend_rate > 0.1",
        for_duration=timedelta(hours=2),
        labels={
            "team": "maintenance",
            "priority": "P2"
        },
        annotations={
            "summary": "Vibration trending up for {{ $labels.equipment_id }}",
            "description": "Vibration increasing at {{ $value | printf \"%.2f\" }} mm/s per day.",
            "action": "Investigate cause. Check bearing condition and alignment."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
    AlertRule(
        name="BearingFaultDetected",
        description="Bearing fault signature detected in vibration spectrum.",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.VIBRATION,
        condition="gl013_bearing_fault_frequency_energy{fault_type=~\"BPFO|BPFI\"} > 0.5",
        for_duration=timedelta(minutes=10),
        labels={
            "team": "maintenance",
            "priority": "P1",
            "component": "bearing"
        },
        annotations={
            "summary": "Bearing fault detected on {{ $labels.equipment_id }} ({{ $labels.fault_type }})",
            "description": "Elevated energy at {{ $labels.fault_type }} frequency indicates {{ $labels.bearing_id }} degradation.",
            "action": "Schedule bearing replacement. Order replacement bearing."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/bearing-fault",
        notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
    ),
    AlertRule(
        name="HighVibrationVelocity",
        description="Vibration velocity exceeds threshold.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.VIBRATION,
        condition="gl013_vibration_velocity_mm_s > 4.5",
        for_duration=timedelta(minutes=10),
        labels={
            "team": "maintenance",
            "priority": "P2"
        },
        annotations={
            "summary": "High vibration on {{ $labels.equipment_id }}: {{ $value | printf \"%.2f\" }} mm/s",
            "description": "Vibration velocity at {{ $labels.measurement_point }} exceeds 4.5 mm/s threshold.",
            "action": "Check machine balance, alignment, and bearing condition."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
]

# -----------------------------------------------------------------------------
# Temperature Alerts
# -----------------------------------------------------------------------------

TEMPERATURE_RULES = [
    AlertRule(
        name="TemperatureCritical",
        description="Equipment temperature has exceeded critical threshold.",
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.TEMPERATURE,
        condition="gl013_temperature_celsius > 120",
        for_duration=timedelta(minutes=2),
        labels={
            "team": "maintenance",
            "priority": "P1",
            "impact": "high",
            "safety": "true"
        },
        annotations={
            "summary": "Critical temperature on {{ $labels.equipment_id }}: {{ $value | printf \"%.1f\" }}C",
            "description": "Temperature at {{ $labels.sensor_location }} has exceeded 120C. Risk of damage or fire.",
            "action": "Reduce load or shut down immediately. Investigate cooling system."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/temperature-critical",
        notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SLACK]
    ),
    AlertRule(
        name="TemperatureHigh",
        description="Equipment temperature is elevated.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.TEMPERATURE,
        condition="gl013_temperature_celsius > 100 and gl013_temperature_celsius <= 120",
        for_duration=timedelta(minutes=10),
        labels={
            "team": "maintenance",
            "priority": "P2"
        },
        annotations={
            "summary": "High temperature on {{ $labels.equipment_id }}: {{ $value | printf \"%.1f\" }}C",
            "description": "Temperature elevated. Check cooling and ventilation.",
            "action": "Verify cooling system operation. Reduce load if possible."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
    AlertRule(
        name="ThermalLifeLow",
        description="Thermal life remaining is critically low.",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.TEMPERATURE,
        condition="gl013_thermal_life_remaining_hours < 2000",
        for_duration=timedelta(minutes=30),
        labels={
            "team": "maintenance",
            "priority": "P1"
        },
        annotations={
            "summary": "Low thermal life for {{ $labels.equipment_id }}: {{ $value | printf \"%.0f\" }} hours",
            "description": "Thermal life consumed significantly. Insulation degradation likely.",
            "action": "Plan winding replacement or rewind. Monitor temperature closely."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/thermal-life-low",
        notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
    ),
    AlertRule(
        name="ThermalLifeConsumedHigh",
        description="More than 80% of thermal life has been consumed.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.TEMPERATURE,
        condition="gl013_thermal_life_consumed_percent > 80",
        for_duration=timedelta(hours=1),
        labels={
            "team": "maintenance",
            "priority": "P2"
        },
        annotations={
            "summary": "High thermal life consumption for {{ $labels.equipment_id }}: {{ $value | printf \"%.0f\" }}%",
            "description": "Thermal life {{ $value | printf \"%.0f\" }}% consumed. Plan for replacement.",
            "action": "Schedule replacement during next planned outage."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
    AlertRule(
        name="HighTemperatureRise",
        description="Temperature rise above ambient is excessive.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.TEMPERATURE,
        condition="gl013_temperature_delta_ambient_celsius > 60",
        for_duration=timedelta(minutes=15),
        labels={
            "team": "maintenance",
            "priority": "P2"
        },
        annotations={
            "summary": "High temperature rise on {{ $labels.equipment_id }}: {{ $value | printf \"%.1f\" }}C above ambient",
            "description": "Temperature rise exceeds 60C. Check load and cooling.",
            "action": "Reduce load if overloaded. Check cooling airflow."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
]

# -----------------------------------------------------------------------------
# Anomaly Detection Alerts
# -----------------------------------------------------------------------------

ANOMALY_RULES = [
    AlertRule(
        name="AnomalyCritical",
        description="Critical anomaly detected by ML model.",
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.ANOMALY,
        condition="gl013_anomaly_score > 0.9",
        for_duration=timedelta(minutes=5),
        labels={
            "team": "maintenance",
            "priority": "P1"
        },
        annotations={
            "summary": "Critical anomaly on {{ $labels.equipment_id }}: score {{ $value | printf \"%.2f\" }}",
            "description": "ML model detected critical anomaly. Immediate investigation required.",
            "action": "Review sensor data and recent operational changes."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/anomaly-critical",
        notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SLACK]
    ),
    AlertRule(
        name="AnomalyHigh",
        description="High anomaly score detected.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.ANOMALY,
        condition="gl013_anomaly_score > 0.7 and gl013_anomaly_score <= 0.9",
        for_duration=timedelta(minutes=15),
        labels={
            "team": "maintenance",
            "priority": "P2"
        },
        annotations={
            "summary": "Anomaly detected on {{ $labels.equipment_id }}: score {{ $value | printf \"%.2f\" }}",
            "description": "Unusual behavior detected. Investigation recommended.",
            "action": "Compare current readings with historical baselines."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
    AlertRule(
        name="MultipleAnomaliesActive",
        description="Multiple critical anomalies are active across equipment.",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.ANOMALY,
        condition='sum(gl013_anomalies_active{severity="critical"}) > 3',
        for_duration=timedelta(minutes=10),
        labels={
            "team": "maintenance",
            "priority": "P1"
        },
        annotations={
            "summary": "Multiple critical anomalies: {{ $value }} active",
            "description": "Systemic issue possible. Review all anomalies immediately.",
            "action": "Investigate for common cause. Check shared systems."
        },
        notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SLACK]
    ),
]

# -----------------------------------------------------------------------------
# Maintenance Scheduling Alerts
# -----------------------------------------------------------------------------

MAINTENANCE_RULES = [
    AlertRule(
        name="MaintenanceOverdue",
        description="Scheduled maintenance is overdue.",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.MAINTENANCE,
        condition="gl013_maintenance_tasks_overdue > 0",
        for_duration=timedelta(hours=1),
        labels={
            "team": "maintenance",
            "priority": "P1"
        },
        annotations={
            "summary": "{{ $value }} maintenance tasks overdue",
            "description": "Scheduled maintenance has not been completed on time.",
            "action": "Review overdue tasks and reschedule immediately."
        },
        notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
    ),
    AlertRule(
        name="EmergencyMaintenanceRequired",
        description="Emergency maintenance has been scheduled.",
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.MAINTENANCE,
        condition='increase(gl013_maintenance_tasks_scheduled_total{urgency="emergency"}[1h]) > 0',
        for_duration=timedelta(minutes=1),
        labels={
            "team": "maintenance",
            "priority": "P0"
        },
        annotations={
            "summary": "Emergency maintenance scheduled for {{ $labels.equipment_type }}",
            "description": "Emergency maintenance task created. Immediate action required.",
            "action": "Mobilize maintenance team. Prepare required resources."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/emergency-maintenance",
        notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SMS, NotificationChannel.SLACK]
    ),
    AlertRule(
        name="SparePartsShortage",
        description="Spare parts inventory is insufficient for upcoming maintenance.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.MAINTENANCE,
        condition="gl013_spare_parts_required > gl013_spare_parts_available",
        for_duration=timedelta(hours=4),
        labels={
            "team": "procurement",
            "priority": "P2"
        },
        annotations={
            "summary": "Spare parts shortage: {{ $labels.part_category }}",
            "description": "Required: {{ $value }}, Available: {{ $value }}. Order additional parts.",
            "action": "Create purchase requisition for required parts."
        },
        notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
    ),
    AlertRule(
        name="MaintenanceLeadTimeCritical",
        description="Maintenance lead time is critically short.",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.MAINTENANCE,
        condition="gl013_maintenance_lead_time_days < 3",
        for_duration=timedelta(hours=1),
        labels={
            "team": "maintenance",
            "priority": "P1"
        },
        annotations={
            "summary": "Maintenance due soon for {{ $labels.equipment_id }}: {{ $value | printf \"%.0f\" }} days",
            "description": "Less than 3 days until scheduled maintenance.",
            "action": "Confirm maintenance resources and schedule."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
]

# -----------------------------------------------------------------------------
# Integration and System Alerts
# -----------------------------------------------------------------------------

INTEGRATION_RULES = [
    AlertRule(
        name="ConnectorDisconnected",
        description="Integration connector has lost connection.",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.INTEGRATION,
        condition="gl013_connector_status == 0",
        for_duration=timedelta(minutes=5),
        labels={
            "team": "platform",
            "priority": "P1"
        },
        annotations={
            "summary": "{{ $labels.connector_type }} connector disconnected from {{ $labels.endpoint }}",
            "description": "Integration connector is not connected. Data flow interrupted.",
            "action": "Check network connectivity and endpoint availability."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/connector-disconnected",
        notification_channels=[NotificationChannel.SLACK]
    ),
    AlertRule(
        name="ConnectorDegraded",
        description="Integration connector is in degraded state.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.INTEGRATION,
        condition="gl013_connector_status == 2",
        for_duration=timedelta(minutes=10),
        labels={
            "team": "platform",
            "priority": "P2"
        },
        annotations={
            "summary": "{{ $labels.connector_type }} connector degraded",
            "description": "Connector experiencing issues. Some operations may fail.",
            "action": "Review connector logs and error rates."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
    AlertRule(
        name="HighConnectorLatency",
        description="Connector latency is abnormally high.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.INTEGRATION,
        condition="histogram_quantile(0.95, rate(gl013_connector_latency_seconds_bucket[5m])) > 5",
        for_duration=timedelta(minutes=10),
        labels={
            "team": "platform",
            "priority": "P2"
        },
        annotations={
            "summary": "High latency on {{ $labels.connector_type }} connector",
            "description": "P95 latency exceeds 5 seconds. Performance degradation.",
            "action": "Check endpoint performance and network."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
    AlertRule(
        name="DataSyncLag",
        description="Data synchronization is lagging significantly.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.INTEGRATION,
        condition="gl013_data_sync_lag_seconds > 300",
        for_duration=timedelta(minutes=15),
        labels={
            "team": "platform",
            "priority": "P2"
        },
        annotations={
            "summary": "Data sync lag: {{ $value | printf \"%.0f\" }} seconds for {{ $labels.data_source }}",
            "description": "Sync lag exceeds 5 minutes. Data may be stale.",
            "action": "Check sync process and source system availability."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
]

SYSTEM_RULES = [
    AlertRule(
        name="HighOperationLatency",
        description="Operation latency is abnormally high.",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.SYSTEM,
        condition="histogram_quantile(0.95, rate(gl013_operation_latency_seconds_bucket[5m])) > 10",
        for_duration=timedelta(minutes=10),
        labels={
            "team": "platform",
            "priority": "P2"
        },
        annotations={
            "summary": "High latency for {{ $labels.operation_type }} operations",
            "description": "P95 latency exceeds 10 seconds. System performance degraded.",
            "action": "Check system resources and database performance."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
    AlertRule(
        name="HighErrorRate",
        description="Error rate is elevated.",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.SYSTEM,
        condition='sum(rate(gl013_operations_total{status="failure"}[5m])) / sum(rate(gl013_operations_total[5m])) > 0.1',
        for_duration=timedelta(minutes=5),
        labels={
            "team": "platform",
            "priority": "P1"
        },
        annotations={
            "summary": "High error rate: {{ $value | printf \"%.1f\" }}%",
            "description": "Operation failure rate exceeds 10%.",
            "action": "Review error logs and recent deployments."
        },
        notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SLACK]
    ),
    AlertRule(
        name="LowCacheHitRate",
        description="Cache hit rate is below threshold.",
        severity=AlertSeverity.INFO,
        category=AlertCategory.SYSTEM,
        condition="gl013_cache_hit_rate < 50",
        for_duration=timedelta(hours=1),
        labels={
            "team": "platform",
            "priority": "P3"
        },
        annotations={
            "summary": "Low cache hit rate: {{ $value | printf \"%.1f\" }}%",
            "description": "Cache effectiveness reduced. May impact performance.",
            "action": "Review cache configuration and access patterns."
        },
        notification_channels=[NotificationChannel.SLACK]
    ),
    AlertRule(
        name="ProvenanceValidationFailures",
        description="Provenance validation failures detected.",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.SYSTEM,
        condition="increase(gl013_provenance_validation_failures_total[1h]) > 0",
        for_duration=timedelta(minutes=5),
        labels={
            "team": "platform",
            "priority": "P1",
            "compliance": "true"
        },
        annotations={
            "summary": "Provenance validation failures detected",
            "description": "Data provenance integrity compromised. Audit trail affected.",
            "action": "Investigate validation failures. Check for data corruption."
        },
        runbook_url="https://runbooks.greenlang.io/gl013/provenance-validation",
        notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SLACK]
    ),
]


# =============================================================================
# COMBINED ALERT RULES LIST
# =============================================================================

ALERT_RULES: List[AlertRule] = (
    EQUIPMENT_HEALTH_RULES +
    FAILURE_PREDICTION_RULES +
    VIBRATION_RULES +
    TEMPERATURE_RULES +
    ANOMALY_RULES +
    MAINTENANCE_RULES +
    INTEGRATION_RULES +
    SYSTEM_RULES
)


# =============================================================================
# ALERT GROUPS
# =============================================================================

ALERT_GROUPS: List[AlertGroup] = [
    AlertGroup(
        name="gl013_equipment_health",
        interval=timedelta(seconds=30),
        rules=EQUIPMENT_HEALTH_RULES
    ),
    AlertGroup(
        name="gl013_failure_prediction",
        interval=timedelta(seconds=30),
        rules=FAILURE_PREDICTION_RULES
    ),
    AlertGroup(
        name="gl013_vibration",
        interval=timedelta(seconds=15),
        rules=VIBRATION_RULES
    ),
    AlertGroup(
        name="gl013_temperature",
        interval=timedelta(seconds=15),
        rules=TEMPERATURE_RULES
    ),
    AlertGroup(
        name="gl013_anomaly",
        interval=timedelta(seconds=30),
        rules=ANOMALY_RULES
    ),
    AlertGroup(
        name="gl013_maintenance",
        interval=timedelta(minutes=1),
        rules=MAINTENANCE_RULES
    ),
    AlertGroup(
        name="gl013_integration",
        interval=timedelta(seconds=30),
        rules=INTEGRATION_RULES
    ),
    AlertGroup(
        name="gl013_system",
        interval=timedelta(seconds=30),
        rules=SYSTEM_RULES
    ),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_rules_by_severity(severity: AlertSeverity) -> List[AlertRule]:
    """
    Get all alert rules with the specified severity.

    Args:
        severity: AlertSeverity to filter by

    Returns:
        List of AlertRule objects matching the severity
    """
    return [rule for rule in ALERT_RULES if rule.severity == severity]


def get_rules_by_category(category: AlertCategory) -> List[AlertRule]:
    """
    Get all alert rules in the specified category.

    Args:
        category: AlertCategory to filter by

    Returns:
        List of AlertRule objects matching the category
    """
    return [rule for rule in ALERT_RULES if rule.category == category]


def get_enabled_rules() -> List[AlertRule]:
    """
    Get all enabled alert rules.

    Returns:
        List of enabled AlertRule objects
    """
    return [rule for rule in ALERT_RULES if rule.enabled]


def export_prometheus_rules() -> Dict[str, Any]:
    """
    Export all alert rules in Prometheus alerting rules format.

    Returns:
        Dictionary suitable for YAML serialization
    """
    return {
        "groups": [group.to_prometheus_group() for group in ALERT_GROUPS]
    }


def get_rule_by_name(name: str) -> Optional[AlertRule]:
    """
    Get an alert rule by its name.

    Args:
        name: Alert rule name

    Returns:
        AlertRule if found, None otherwise
    """
    for rule in ALERT_RULES:
        if rule.name == name:
            return rule
    return None


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "AlertSeverity",
    "AlertCategory",
    "NotificationChannel",
    # Data classes
    "AlertThreshold",
    "AlertRule",
    "AlertGroup",
    # Rule collections
    "ALERT_RULES",
    "ALERT_GROUPS",
    "EQUIPMENT_HEALTH_RULES",
    "FAILURE_PREDICTION_RULES",
    "VIBRATION_RULES",
    "TEMPERATURE_RULES",
    "ANOMALY_RULES",
    "MAINTENANCE_RULES",
    "INTEGRATION_RULES",
    "SYSTEM_RULES",
    # Helper functions
    "get_rules_by_severity",
    "get_rules_by_category",
    "get_enabled_rules",
    "export_prometheus_rules",
    "get_rule_by_name",
]
