"""
GL-003 UNIFIEDSTEAM - Event Publisher

Publishes system events, alarms, recommendations, and operational changes
to Kafka and notification systems.

Event Types:
- Alarms: Process and equipment alarms with severity levels
- Recommendations: Optimization recommendations with impact estimates
- Maintenance Events: Predictive maintenance alerts
- Setpoint Changes: Operational setpoint modification events
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class AlarmSeverity(Enum):
    """Alarm severity levels (ISA-18.2 compliant)."""
    DIAGNOSTIC = "diagnostic"  # Informational
    LOW = "low"  # Minor issue, no immediate action
    MEDIUM = "medium"  # Significant issue, action recommended
    HIGH = "high"  # Serious issue, action required
    CRITICAL = "critical"  # Safety/environmental, immediate action


class AlarmState(Enum):
    """Alarm states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    CLEARED = "cleared"
    SUPPRESSED = "suppressed"
    SHELVED = "shelved"


class AlarmCategory(Enum):
    """Alarm categories."""
    PROCESS = "process"  # Process variable alarms
    EQUIPMENT = "equipment"  # Equipment health alarms
    SAFETY = "safety"  # Safety system alarms
    ENVIRONMENTAL = "environmental"  # Environmental compliance
    QUALITY = "quality"  # Data quality alarms
    SYSTEM = "system"  # System/infrastructure alarms


class RecommendationType(Enum):
    """Types of optimization recommendations."""
    SETPOINT_CHANGE = "setpoint_change"  # Change operating setpoint
    MAINTENANCE = "maintenance"  # Schedule maintenance
    OPERATIONAL = "operational"  # Operational procedure change
    EQUIPMENT = "equipment"  # Equipment repair/replacement
    SCHEDULING = "scheduling"  # Production scheduling
    LOAD_SHIFT = "load_shift"  # Shift load between equipment


class RecommendationPriority(Enum):
    """Recommendation priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MaintenanceType(Enum):
    """Types of maintenance events."""
    PREDICTIVE = "predictive"  # Based on condition monitoring
    PREVENTIVE = "preventive"  # Scheduled maintenance
    CORRECTIVE = "corrective"  # Fix known issue
    EMERGENCY = "emergency"  # Urgent unplanned


@dataclass
class Alarm:
    """Process or equipment alarm."""
    alarm_id: str
    tag: str
    severity: AlarmSeverity
    category: AlarmCategory
    title: str
    description: str
    timestamp: datetime

    # State
    state: AlarmState = AlarmState.ACTIVE

    # Values
    value: Optional[float] = None
    limit: Optional[float] = None
    deviation: Optional[float] = None
    unit: str = ""

    # Context
    asset_id: str = ""
    asset_type: str = ""
    area: str = ""
    site: str = ""

    # Associated data
    associated_tags: List[str] = field(default_factory=list)
    associated_values: Dict[str, float] = field(default_factory=dict)

    # Actions
    recommended_action: str = ""
    escalation_time_s: int = 300  # Time before escalation

    # Acknowledgment
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    acknowledgment_comment: str = ""

    # Clearing
    cleared_at: Optional[datetime] = None
    auto_clear: bool = True

    def to_dict(self) -> Dict:
        return {
            "alarm_id": self.alarm_id,
            "tag": self.tag,
            "severity": self.severity.value,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state.value,
            "value": self.value,
            "limit": self.limit,
            "deviation": self.deviation,
            "unit": self.unit,
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "area": self.area,
            "site": self.site,
            "recommended_action": self.recommended_action,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")


@dataclass
class Recommendation:
    """Optimization recommendation."""
    recommendation_id: str
    recommendation_type: RecommendationType
    priority: RecommendationPriority
    title: str
    description: str
    timestamp: datetime

    # Target
    target_asset_id: str = ""
    target_asset_type: str = ""
    target_parameter: str = ""

    # Change details
    current_value: Optional[float] = None
    recommended_value: Optional[float] = None
    unit: str = ""
    change_percent: float = 0.0

    # Impact estimates
    estimated_savings_pct: float = 0.0
    estimated_savings_value: float = 0.0
    savings_unit: str = ""  # $/hr, MMBTU/hr, etc.
    payback_period_hours: float = 0.0
    co2_reduction_kg_hr: float = 0.0

    # Confidence
    confidence: float = 0.0
    model_version: str = ""
    calculation_method: str = ""

    # Supporting data
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    causal_factors: List[str] = field(default_factory=list)

    # Status
    status: str = "pending"  # pending, accepted, rejected, implemented, expired
    expires_at: Optional[datetime] = None

    # Implementation
    implemented_at: Optional[datetime] = None
    implemented_by: Optional[str] = None
    actual_value: Optional[float] = None
    actual_savings: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "recommendation_id": self.recommendation_id,
            "recommendation_type": self.recommendation_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "target_asset_id": self.target_asset_id,
            "target_asset_type": self.target_asset_type,
            "target_parameter": self.target_parameter,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "unit": self.unit,
            "change_percent": round(self.change_percent, 2),
            "estimated_savings_pct": round(self.estimated_savings_pct, 2),
            "estimated_savings_value": round(self.estimated_savings_value, 2),
            "savings_unit": self.savings_unit,
            "payback_period_hours": round(self.payback_period_hours, 2),
            "co2_reduction_kg_hr": round(self.co2_reduction_kg_hr, 2),
            "confidence": round(self.confidence, 3),
            "model_version": self.model_version,
            "status": self.status,
            "causal_factors": self.causal_factors,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")


@dataclass
class MaintenanceEvent:
    """Predictive/preventive maintenance event."""
    event_id: str
    maintenance_type: MaintenanceType
    severity: AlarmSeverity
    title: str
    description: str
    timestamp: datetime

    # Target equipment
    asset_id: str = ""
    asset_type: str = ""
    asset_name: str = ""

    # Prediction details
    failure_probability: float = 0.0
    remaining_useful_life_hours: Optional[float] = None
    degradation_rate: float = 0.0

    # Condition indicators
    condition_indicators: Dict[str, float] = field(default_factory=dict)
    anomaly_history: List[Dict] = field(default_factory=list)

    # Recommended actions
    recommended_actions: List[str] = field(default_factory=list)
    required_parts: List[str] = field(default_factory=list)
    estimated_downtime_hours: float = 0.0
    estimated_cost: float = 0.0

    # Scheduling
    recommended_date: Optional[datetime] = None
    deadline: Optional[datetime] = None
    scheduled_date: Optional[datetime] = None

    # Status
    status: str = "pending"  # pending, scheduled, in_progress, completed, cancelled

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "maintenance_type": self.maintenance_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "asset_name": self.asset_name,
            "failure_probability": round(self.failure_probability, 3),
            "remaining_useful_life_hours": self.remaining_useful_life_hours,
            "condition_indicators": self.condition_indicators,
            "recommended_actions": self.recommended_actions,
            "estimated_downtime_hours": self.estimated_downtime_hours,
            "status": self.status,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")


@dataclass
class SetpointChange:
    """Operational setpoint change event."""
    change_id: str
    timestamp: datetime

    # Target
    tag: str
    asset_id: str = ""
    asset_type: str = ""
    parameter_name: str = ""

    # Values
    previous_value: float = 0.0
    new_value: float = 0.0
    unit: str = ""
    change_percent: float = 0.0

    # Source
    source: str = ""  # manual, optimizer, schedule, emergency
    requested_by: str = ""  # user or system
    recommendation_id: Optional[str] = None

    # Validation
    validated: bool = False
    validation_result: str = ""
    constraints_checked: List[str] = field(default_factory=list)

    # Impact
    expected_impact: Dict[str, float] = field(default_factory=dict)

    # Execution
    executed: bool = False
    executed_at: Optional[datetime] = None
    execution_result: str = ""

    # Rollback
    can_rollback: bool = True
    rollback_deadline: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "change_id": self.change_id,
            "timestamp": self.timestamp.isoformat(),
            "tag": self.tag,
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "parameter_name": self.parameter_name,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "unit": self.unit,
            "change_percent": round(self.change_percent, 2),
            "source": self.source,
            "requested_by": self.requested_by,
            "recommendation_id": self.recommendation_id,
            "validated": self.validated,
            "executed": self.executed,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")


class EventPublisher:
    """
    Publishes steam system events to Kafka and notification systems.

    Handles alarms, recommendations, maintenance events, and setpoint changes
    with appropriate routing, persistence, and notification.

    Example:
        publisher = EventPublisher(kafka_producer)

        # Publish alarm
        await publisher.publish_alarm(Alarm(
            alarm_id="ALM001",
            tag="HEADER.PRESSURE",
            severity=AlarmSeverity.HIGH,
            category=AlarmCategory.PROCESS,
            title="High Header Pressure",
            description="Steam header pressure exceeds high limit",
            timestamp=datetime.now(timezone.utc),
            value=165.0,
            limit=150.0,
        ))

        # Publish recommendation
        await publisher.publish_recommendation(Recommendation(
            recommendation_id="REC001",
            recommendation_type=RecommendationType.SETPOINT_CHANGE,
            priority=RecommendationPriority.MEDIUM,
            title="Reduce Excess Air",
            description="Reduce boiler excess air to improve efficiency",
            timestamp=datetime.now(timezone.utc),
            target_asset_id="BOILER1",
            current_value=25.0,
            recommended_value=20.0,
            estimated_savings_pct=2.5,
        ))
    """

    def __init__(
        self,
        kafka_producer: Optional[Any] = None,
        notification_service: Optional[Any] = None,
    ) -> None:
        """
        Initialize event publisher.

        Args:
            kafka_producer: SteamKafkaProducer instance
            notification_service: Optional notification service
        """
        self._producer = kafka_producer
        self._notification_service = notification_service

        # Active alarms tracking
        self._active_alarms: Dict[str, Alarm] = {}

        # Recommendation tracking
        self._active_recommendations: Dict[str, Recommendation] = {}

        # Statistics
        self._stats = {
            "alarms_published": 0,
            "recommendations_published": 0,
            "maintenance_events_published": 0,
            "setpoint_changes_published": 0,
            "notifications_sent": 0,
        }

        # Callbacks
        self._on_alarm: Optional[Callable[[Alarm], None]] = None
        self._on_recommendation: Optional[Callable[[Recommendation], None]] = None

        logger.info("EventPublisher initialized")

    def set_alarm_callback(self, callback: Callable[[Alarm], None]) -> None:
        """Set callback for alarm events."""
        self._on_alarm = callback

    def set_recommendation_callback(self, callback: Callable[[Recommendation], None]) -> None:
        """Set callback for recommendation events."""
        self._on_recommendation = callback

    async def publish_alarm(
        self,
        alarm: Alarm,
        site: str = "default",
        area: str = "util",
    ) -> None:
        """
        Publish alarm to Kafka and notification system.

        Args:
            alarm: Alarm to publish
            site: Site identifier
            area: Process area
        """
        # Track active alarm
        if alarm.state == AlarmState.ACTIVE:
            self._active_alarms[alarm.alarm_id] = alarm
        elif alarm.state == AlarmState.CLEARED:
            self._active_alarms.pop(alarm.alarm_id, None)

        # Publish to Kafka
        if self._producer:
            from .kafka_producer import EventData
            event = EventData(
                event_id=alarm.alarm_id,
                event_type="alarm",
                category=alarm.category.value,
                title=alarm.title,
                description=alarm.description,
                timestamp=alarm.timestamp,
                source_asset_id=alarm.asset_id,
                severity=alarm.severity.value,
                associated_tags=alarm.associated_tags,
                associated_values=alarm.associated_values,
            )
            await self._producer.publish_event(site, area, event)

        # Send notification for high/critical
        if alarm.severity in [AlarmSeverity.HIGH, AlarmSeverity.CRITICAL]:
            await self._send_notification(
                title=f"[{alarm.severity.value.upper()}] {alarm.title}",
                message=alarm.description,
                severity=alarm.severity.value,
            )

        # Callback
        if self._on_alarm:
            self._on_alarm(alarm)

        self._stats["alarms_published"] += 1
        logger.info(f"Published alarm: {alarm.alarm_id} ({alarm.severity.value})")

    async def publish_recommendation(
        self,
        recommendation: Recommendation,
        site: str = "default",
        area: str = "util",
    ) -> None:
        """
        Publish optimization recommendation.

        Args:
            recommendation: Recommendation to publish
            site: Site identifier
            area: Process area
        """
        # Track active recommendation
        self._active_recommendations[recommendation.recommendation_id] = recommendation

        # Publish to Kafka
        if self._producer:
            from .kafka_producer import RecommendationData
            rec_data = RecommendationData(
                recommendation_id=recommendation.recommendation_id,
                recommendation_type=recommendation.recommendation_type.value,
                title=recommendation.title,
                description=recommendation.description,
                timestamp=recommendation.timestamp,
                target_asset_id=recommendation.target_asset_id,
                target_asset_type=recommendation.target_asset_type,
                current_value=recommendation.current_value,
                recommended_value=recommendation.recommended_value,
                parameter_name=recommendation.target_parameter,
                unit=recommendation.unit,
                estimated_savings_pct=recommendation.estimated_savings_pct,
                estimated_savings_value=recommendation.estimated_savings_value,
                savings_unit=recommendation.savings_unit,
                payback_hours=recommendation.payback_period_hours,
                priority=recommendation.priority.value,
                confidence=recommendation.confidence,
                status=recommendation.status,
            )
            await self._producer.publish_recommendation(site, area, rec_data)

        # Send notification for high priority
        if recommendation.priority in [RecommendationPriority.HIGH, RecommendationPriority.CRITICAL]:
            await self._send_notification(
                title=f"[{recommendation.priority.value.upper()}] {recommendation.title}",
                message=f"{recommendation.description}\nEstimated savings: {recommendation.estimated_savings_pct:.1f}%",
                severity=recommendation.priority.value,
            )

        # Callback
        if self._on_recommendation:
            self._on_recommendation(recommendation)

        self._stats["recommendations_published"] += 1
        logger.info(f"Published recommendation: {recommendation.recommendation_id}")

    async def publish_maintenance_event(
        self,
        event: MaintenanceEvent,
        site: str = "default",
        area: str = "util",
    ) -> None:
        """
        Publish maintenance event.

        Args:
            event: Maintenance event to publish
            site: Site identifier
            area: Process area
        """
        # Publish to Kafka
        if self._producer:
            from .kafka_producer import EventData
            kafka_event = EventData(
                event_id=event.event_id,
                event_type="maintenance",
                category="equipment",
                title=event.title,
                description=event.description,
                timestamp=event.timestamp,
                source_asset_id=event.asset_id,
                source_asset_type=event.asset_type,
                severity=event.severity.value,
                associated_values=event.condition_indicators,
            )
            await self._producer.publish_event(site, area, kafka_event)

        # Send notification
        if event.maintenance_type == MaintenanceType.EMERGENCY:
            await self._send_notification(
                title=f"[EMERGENCY] Maintenance Required: {event.asset_name}",
                message=event.description,
                severity="critical",
            )
        elif event.severity in [AlarmSeverity.HIGH, AlarmSeverity.CRITICAL]:
            await self._send_notification(
                title=f"Maintenance Alert: {event.asset_name}",
                message=f"{event.description}\nRUL: {event.remaining_useful_life_hours:.0f} hours" if event.remaining_useful_life_hours else event.description,
                severity=event.severity.value,
            )

        self._stats["maintenance_events_published"] += 1
        logger.info(f"Published maintenance event: {event.event_id}")

    async def publish_setpoint_change(
        self,
        change: SetpointChange,
        site: str = "default",
        area: str = "util",
    ) -> None:
        """
        Publish setpoint change event.

        Args:
            change: Setpoint change to publish
            site: Site identifier
            area: Process area
        """
        # Publish to Kafka
        if self._producer:
            from .kafka_producer import EventData
            event = EventData(
                event_id=change.change_id,
                event_type="setpoint_change",
                category="process",
                title=f"Setpoint Change: {change.parameter_name}",
                description=f"{change.previous_value} -> {change.new_value} {change.unit}",
                timestamp=change.timestamp,
                source_asset_id=change.asset_id,
                severity="info",
                associated_tags=[change.tag],
                associated_values={
                    "previous_value": change.previous_value,
                    "new_value": change.new_value,
                    "change_percent": change.change_percent,
                },
            )
            await self._producer.publish_event(site, area, event)

        self._stats["setpoint_changes_published"] += 1
        logger.info(f"Published setpoint change: {change.tag} {change.previous_value} -> {change.new_value}")

    async def acknowledge_alarm(
        self,
        alarm_id: str,
        user: str,
        comment: str = "",
    ) -> bool:
        """
        Acknowledge an alarm.

        Args:
            alarm_id: Alarm to acknowledge
            user: User acknowledging
            comment: Optional comment

        Returns:
            True if acknowledged successfully
        """
        if alarm_id not in self._active_alarms:
            return False

        alarm = self._active_alarms[alarm_id]
        alarm.state = AlarmState.ACKNOWLEDGED
        alarm.acknowledged_by = user
        alarm.acknowledged_at = datetime.now(timezone.utc)
        alarm.acknowledgment_comment = comment

        # Publish update
        await self.publish_alarm(alarm)

        logger.info(f"Alarm {alarm_id} acknowledged by {user}")
        return True

    async def clear_alarm(self, alarm_id: str) -> bool:
        """
        Clear an alarm.

        Args:
            alarm_id: Alarm to clear

        Returns:
            True if cleared successfully
        """
        if alarm_id not in self._active_alarms:
            return False

        alarm = self._active_alarms[alarm_id]
        alarm.state = AlarmState.CLEARED
        alarm.cleared_at = datetime.now(timezone.utc)

        # Publish update
        await self.publish_alarm(alarm)

        # Remove from active
        self._active_alarms.pop(alarm_id, None)

        logger.info(f"Alarm {alarm_id} cleared")
        return True

    async def update_recommendation_status(
        self,
        recommendation_id: str,
        status: str,
        user: Optional[str] = None,
        actual_value: Optional[float] = None,
    ) -> bool:
        """
        Update recommendation status.

        Args:
            recommendation_id: Recommendation to update
            status: New status
            user: User making update
            actual_value: Actual value if implemented

        Returns:
            True if updated successfully
        """
        if recommendation_id not in self._active_recommendations:
            return False

        rec = self._active_recommendations[recommendation_id]
        rec.status = status

        if status == "implemented":
            rec.implemented_at = datetime.now(timezone.utc)
            rec.implemented_by = user
            rec.actual_value = actual_value

        # Publish update
        await self.publish_recommendation(rec)

        logger.info(f"Recommendation {recommendation_id} status updated to {status}")
        return True

    async def _send_notification(
        self,
        title: str,
        message: str,
        severity: str,
    ) -> None:
        """Send notification through notification service."""
        if self._notification_service:
            try:
                # await self._notification_service.send(title, message, severity)
                self._stats["notifications_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
        else:
            # Log notification
            logger.info(f"Notification: [{severity}] {title} - {message}")
            self._stats["notifications_sent"] += 1

    def get_active_alarms(self) -> List[Alarm]:
        """Get all active alarms."""
        return list(self._active_alarms.values())

    def get_active_recommendations(self) -> List[Recommendation]:
        """Get all active recommendations."""
        return [r for r in self._active_recommendations.values() if r.status == "pending"]

    def get_alarm_summary(self) -> Dict[str, int]:
        """Get alarm count by severity."""
        summary = {severity.value: 0 for severity in AlarmSeverity}
        for alarm in self._active_alarms.values():
            summary[alarm.severity.value] += 1
        return summary

    def get_statistics(self) -> Dict:
        """Get publisher statistics."""
        return {
            **self._stats,
            "active_alarms": len(self._active_alarms),
            "active_recommendations": len([r for r in self._active_recommendations.values() if r.status == "pending"]),
        }


def create_alarm_from_anomaly(
    anomaly: Any,
    asset_id: str = "",
    asset_type: str = "",
) -> Alarm:
    """
    Create alarm from detected anomaly.

    Args:
        anomaly: Anomaly from stream processor
        asset_id: Asset identifier
        asset_type: Asset type

    Returns:
        Alarm instance
    """
    # Map anomaly severity to alarm severity
    severity_map = {
        "low": AlarmSeverity.LOW,
        "medium": AlarmSeverity.MEDIUM,
        "high": AlarmSeverity.HIGH,
        "critical": AlarmSeverity.CRITICAL,
    }

    return Alarm(
        alarm_id=anomaly.anomaly_id,
        tag=anomaly.tag,
        severity=severity_map.get(anomaly.severity, AlarmSeverity.MEDIUM),
        category=AlarmCategory.PROCESS,
        title=f"Anomaly Detected: {anomaly.anomaly_type.value}",
        description=anomaly.description,
        timestamp=anomaly.timestamp,
        value=anomaly.observed_value,
        limit=anomaly.expected_value,
        deviation=anomaly.deviation,
        asset_id=asset_id,
        asset_type=asset_type,
        recommended_action=f"Investigate {anomaly.anomaly_type.value} anomaly on {anomaly.tag}",
    )
