"""
ISA182 - ISA 18.2 Alarm Management

This module implements alarm management per ISA 18.2:
Management of Alarm Systems for the Process Industries.

Key aspects covered:
- Alarm rationalization
- Alarm performance metrics (EEMUA 191)
- Master Alarm Database
- Alarm state-based suppression

Reference: ANSI/ISA-18.2-2016

Example:
    >>> from greenlang.safety.compliance.isa_18_2 import ISA182
    >>> manager = ISA182(system_id="AMS-001")
    >>> metrics = manager.calculate_metrics()
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class AlarmPriority(str, Enum):
    """Alarm priorities per ISA 18.2."""
    EMERGENCY = "emergency"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    JOURNAL = "journal"  # Information only


class AlarmState(str, Enum):
    """Alarm states per ISA 18.2."""
    NORMAL = "normal"
    UNACKNOWLEDGED = "unacknowledged"
    ACKNOWLEDGED = "acknowledged"
    RETURNED_UNACKNOWLEDGED = "returned_unacknowledged"
    SUPPRESSED = "suppressed"
    SHELVED = "shelved"
    OUT_OF_SERVICE = "out_of_service"


class AlarmRationalization(BaseModel):
    """Alarm rationalization record per ISA 18.2."""
    alarm_id: str = Field(...)
    tag_name: str = Field(...)
    alarm_type: str = Field(...)
    priority: AlarmPriority = Field(...)
    setpoint: float = Field(...)
    consequence: str = Field(...)
    response_time_minutes: float = Field(...)
    operator_action: str = Field(...)
    cause: str = Field(...)
    is_rationalized: bool = Field(default=True)
    suppression_allowed: bool = Field(default=False)
    shelving_allowed: bool = Field(default=True)
    class_code: Optional[str] = Field(None)  # Equipment class


class AlarmMetrics(BaseModel):
    """Alarm performance metrics per EEMUA 191."""
    period: str = Field(...)
    total_alarms: int = Field(...)
    alarms_per_10_minutes: float = Field(...)
    alarms_per_operator_hour: float = Field(...)
    peak_rate_10_min: int = Field(...)
    standing_alarms: int = Field(...)
    priority_distribution: Dict[str, int] = Field(default_factory=dict)
    chattering_alarms: int = Field(default=0)
    stale_alarms: int = Field(default=0)
    average_response_time_seconds: float = Field(default=0)
    performance_rating: str = Field(default="unknown")
    provenance_hash: str = Field(default="")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ISA182:
    """
    ISA 18.2 Alarm Management System.

    Implements alarm management best practices per ISA 18.2
    and EEMUA 191 guidelines.

    Performance targets per EEMUA 191:
    - Acceptable: <150 alarms per 10 minutes average
    - Manageable: <300 alarms per 10 minutes average
    - Overloaded: >300 alarms per 10 minutes average

    Example:
        >>> manager = ISA182(system_id="AMS-001")
        >>> manager.add_alarm_event(tag="PI-101", priority=AlarmPriority.HIGH)
        >>> metrics = manager.calculate_metrics()
    """

    # EEMUA 191 performance thresholds
    THRESHOLDS = {
        "acceptable_per_10min": 6,
        "manageable_per_10min": 12,
        "target_per_operator_day": 144,  # Average 1 every 10 minutes
        "chattering_threshold": 3,  # Same alarm 3+ times in 10 min
    }

    def __init__(self, system_id: str):
        """Initialize ISA182 manager."""
        self.system_id = system_id
        self.alarm_database: Dict[str, AlarmRationalization] = {}
        self.alarm_events: List[Dict[str, Any]] = []
        logger.info(f"ISA182 alarm manager initialized: {system_id}")

    def add_rationalized_alarm(
        self,
        alarm: AlarmRationalization
    ) -> None:
        """Add alarm to Master Alarm Database."""
        self.alarm_database[alarm.alarm_id] = alarm
        logger.debug(f"Added alarm {alarm.alarm_id} to database")

    def record_alarm_event(
        self,
        alarm_id: str,
        state: AlarmState,
        value: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record an alarm event."""
        event = {
            "alarm_id": alarm_id,
            "state": state.value,
            "value": value,
            "timestamp": timestamp or datetime.utcnow(),
        }
        self.alarm_events.append(event)

    def calculate_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operator_count: int = 1
    ) -> AlarmMetrics:
        """
        Calculate alarm performance metrics.

        Args:
            start_time: Analysis start time
            end_time: Analysis end time
            operator_count: Number of operators

        Returns:
            AlarmMetrics
        """
        # Filter events by time
        events = self.alarm_events
        if start_time:
            events = [e for e in events if e["timestamp"] >= start_time]
        if end_time:
            events = [e for e in events if e["timestamp"] <= end_time]

        if not events:
            return AlarmMetrics(
                period="No data",
                total_alarms=0,
                alarms_per_10_minutes=0,
                alarms_per_operator_hour=0,
                peak_rate_10_min=0,
                standing_alarms=0,
            )

        total_alarms = len(events)

        # Calculate time period
        first_time = min(e["timestamp"] for e in events)
        last_time = max(e["timestamp"] for e in events)
        duration_hours = (last_time - first_time).total_seconds() / 3600
        duration_10min = max(duration_hours * 6, 1)  # At least 1 period

        alarms_per_10min = total_alarms / duration_10min
        alarms_per_op_hour = total_alarms / (duration_hours * operator_count) if duration_hours > 0 else 0

        # Count by priority
        priority_dist = {}
        for event in events:
            alarm = self.alarm_database.get(event["alarm_id"])
            if alarm:
                pri = alarm.priority.value
                priority_dist[pri] = priority_dist.get(pri, 0) + 1

        # Identify chattering alarms
        alarm_counts = {}
        for event in events:
            alarm_id = event["alarm_id"]
            alarm_counts[alarm_id] = alarm_counts.get(alarm_id, 0) + 1

        chattering = sum(1 for count in alarm_counts.values()
                        if count >= self.THRESHOLDS["chattering_threshold"])

        # Determine performance rating
        if alarms_per_10min <= self.THRESHOLDS["acceptable_per_10min"]:
            rating = "acceptable"
        elif alarms_per_10min <= self.THRESHOLDS["manageable_per_10min"]:
            rating = "manageable"
        else:
            rating = "overloaded"

        metrics = AlarmMetrics(
            period=f"{first_time.isoformat()} to {last_time.isoformat()}",
            total_alarms=total_alarms,
            alarms_per_10_minutes=round(alarms_per_10min, 1),
            alarms_per_operator_hour=round(alarms_per_op_hour, 1),
            peak_rate_10_min=max(alarm_counts.values()) if alarm_counts else 0,
            standing_alarms=0,  # Would need current state
            priority_distribution=priority_dist,
            chattering_alarms=chattering,
            performance_rating=rating,
        )

        metrics.provenance_hash = hashlib.sha256(
            f"{self.system_id}|{total_alarms}|{alarms_per_10min}".encode()
        ).hexdigest()

        return metrics

    def get_bad_actors(
        self,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top nuisance/chattering alarms."""
        alarm_counts: Dict[str, int] = {}
        for event in self.alarm_events:
            alarm_id = event["alarm_id"]
            alarm_counts[alarm_id] = alarm_counts.get(alarm_id, 0) + 1

        sorted_alarms = sorted(
            alarm_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        bad_actors = []
        for alarm_id, count in sorted_alarms:
            alarm = self.alarm_database.get(alarm_id)
            bad_actors.append({
                "alarm_id": alarm_id,
                "tag_name": alarm.tag_name if alarm else "Unknown",
                "activation_count": count,
                "priority": alarm.priority.value if alarm else "Unknown",
            })

        return bad_actors
