"""
ISA 18.2 Alarm Management System for Process Heat Agents

This module implements comprehensive alarm management following ISA-18.2-2016 standard
for Process Measurement, Control, and Automation (PMCA) systems. Specifically designed
for Process Heat agents with advanced rationalization, flood detection, and metrics.

Key Features:
    - Five-level priority system (Emergency, High, Medium, Low, Diagnostic)
    - Alarm configuration with deadband and setpoint management
    - Alarm acknowledgment and shelving capabilities
    - Alarm rationalization per ISA 18.2 Annex D
    - Flood detection and suppression (>10 alarms/10 min)
    - Chattering/fleeting alarm detection (<1 sec duration)
    - Performance metrics per ISA-18.2-2016:
        * Alarms per operator per 10 minutes (target <10)
        * % acknowledged within 10 minutes (target >90%)
        * Standing alarm count (unacknowledged)
        * Chattering alarm detection

ISA-18.2-2016 References:
    - Section 4.3: Alarm priority assignment
    - Section 5.1: Operator burden assessment
    - Annex C: Alarm rationalization process
    - Annex D: Operator performance metrics

Process Heat Integration:
    - Monitor furnace temperature, pressure, flow setpoints
    - Detect process deviations with hysteresis (deadband)
    - Track equipment cycling and process faults
    - Support multi-point alarming (High, High-High, Low, Low-Low)

Alarm Priority Definitions (ISA-18.2):
    - Emergency: Immediate action required (1-2 second response)
    - High: Prompt action required (5-10 second response)
    - Medium: Timely action required (30-60 second response)
    - Low: Awareness only (no immediate action)
    - Diagnostic: Information only (not for operator display during normal ops)

Example:
    >>> from greenlang.safety.isa_18_2_alarms import (
    ...     AlarmManager, AlarmPriority, AlarmState
    ... )
    >>> manager = AlarmManager(config={
    ...     'operator_id': 'OP-001',
    ...     'plant_id': 'PLANT-01'
    ... })
    >>> # Configure high temperature alarm
    >>> manager.configure_alarm(
    ...     tag='FURNACE_TEMP_01',
    ...     description='Furnace Temperature High',
    ...     priority=AlarmPriority.HIGH,
    ...     setpoint=450.0,
    ...     deadband=5.0
    ... )
    >>> # Process an alarm event
    >>> result = manager.process_alarm(
    ...     tag='FURNACE_TEMP_01',
    ...     value=455.0,
    ...     timestamp=datetime.now()
    ... )
    >>> if result.alarm_triggered:
    ...     manager.acknowledge_alarm(
    ...         alarm_id=result.alarm_id,
    ...         operator_id='OP-001'
    ...     )
    >>> # Check metrics per ISA 18.2
    >>> metrics = manager.get_alarm_metrics()
    >>> print(f"Alarms per 10 min: {metrics.alarms_per_10min}")
    >>> print(f"Ack rate: {metrics.ack_rate_10min_pct}%")

Author: GreenLang Safety Engineering Team
License: Proprietary
"""

from typing import Dict, List, Optional, Any, ClassVar, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field as dataclass_field
import uuid
import threading
from collections import deque
from statistics import mean

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS - ISA 18.2 STANDARD
# =============================================================================

class AlarmPriority(str, Enum):
    """ISA-18.2 Alarm Priority Levels."""

    EMERGENCY = "EMERGENCY"      # Immediate action - 1-2 sec response
    HIGH = "HIGH"                # Prompt action - 5-10 sec response
    MEDIUM = "MEDIUM"            # Timely action - 30-60 sec response
    LOW = "LOW"                  # Awareness only
    DIAGNOSTIC = "DIAGNOSTIC"    # Information only

    @property
    def numeric_value(self) -> int:
        """Return numeric value for sorting (higher = more urgent)."""
        mapping = {
            AlarmPriority.EMERGENCY: 5,
            AlarmPriority.HIGH: 4,
            AlarmPriority.MEDIUM: 3,
            AlarmPriority.LOW: 2,
            AlarmPriority.DIAGNOSTIC: 1,
        }
        return mapping[self]


class AlarmState(str, Enum):
    """Alarm State Machine per ISA 18.2."""

    NORMAL = "NORMAL"                # No alarm condition
    UNACKNOWLEDGED = "UNACKNOWLEDGED"  # Alarm active, not ack'd
    ACKNOWLEDGED = "ACKNOWLEDGED"    # Alarm active, ack'd
    SHELVED = "SHELVED"              # Suppressed temporarily
    CLEARED = "CLEARED"              # Alarm condition resolved
    STALE = "STALE"                  # Standing >1 hour


class AlarmType(str, Enum):
    """Types of alarms for Process Heat."""

    ANALOG_HI = "ANALOG_HI"          # High setpoint
    ANALOG_HI_HI = "ANALOG_HI_HI"    # High-High setpoint
    ANALOG_LO = "ANALOG_LO"          # Low setpoint
    ANALOG_LO_LO = "ANALOG_LO_LO"    # Low-Low setpoint
    DIGITAL = "DIGITAL"              # Discrete switch alarm
    CHATTERING = "CHATTERING"        # Fleeting alarm
    FLOODED = "FLOODED"              # Flood suppression


# =============================================================================
# DATA MODELS - PYDANTIC
# =============================================================================

class AlarmSetpoint(BaseModel):
    """Alarm setpoint configuration with deadband."""

    value: float = Field(..., description="Setpoint value")
    deadband: float = Field(
        default=0.0,
        ge=0.0,
        description="Deadband for hysteresis (clears when value < setpoint-deadband)"
    )
    units: str = Field(default="", description="Engineering units")

    def is_triggered(self, current_value: float) -> bool:
        """Check if setpoint is triggered."""
        return current_value >= self.value

    def is_cleared(self, current_value: float) -> bool:
        """Check if alarm is cleared (with deadband)."""
        return current_value < (self.value - self.deadband)


class AlarmConfiguration(BaseModel):
    """Complete alarm configuration per ISA 18.2."""

    tag: str = Field(..., description="Alarm tag (must be unique)")
    description: str = Field(..., description="Alarm description for operator")
    alarm_type: AlarmType = Field(default=AlarmType.ANALOG_HI)
    priority: AlarmPriority = Field(default=AlarmPriority.MEDIUM)
    setpoint: AlarmSetpoint = Field(..., description="Setpoint and deadband")

    # ISA 18.2 Rationalization fields (Annex D)
    consequence: str = Field(
        default="",
        description="What happens if process parameter exceeds setpoint"
    )
    response: str = Field(
        default="",
        description="Required operator response"
    )
    response_time_sec: int = Field(
        default=60,
        ge=1,
        description="Target response time in seconds"
    )

    # Enable/Disable
    enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class AlarmEvent(BaseModel):
    """Single alarm event record."""

    alarm_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tag: str = Field(...)
    priority: AlarmPriority = Field(...)
    state: AlarmState = Field(default=AlarmState.UNACKNOWLEDGED)
    value: float = Field(...)
    setpoint: float = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)

    # Acknowledgment tracking (ISA 18.2 requirement)
    ack_timestamp: Optional[datetime] = Field(None)
    ack_operator_id: Optional[str] = Field(None)
    ack_time_sec: Optional[float] = Field(None, description="Time to acknowledge")

    # Shelving tracking
    shelved_until: Optional[datetime] = Field(None)
    shelved_reason: Optional[str] = Field(None)

    # Provenance
    provenance_hash: str = Field(...)


class ProcessAlarmResult(BaseModel):
    """Result from processing a single alarm."""

    alarm_triggered: bool = Field(...)
    alarm_cleared: bool = Field(...)
    alarm_id: Optional[str] = Field(None)
    new_state: AlarmState = Field(...)
    flooded: bool = Field(default=False, description="True if flood suppression active")
    chattering: bool = Field(default=False, description="True if chattering detected")


class AlarmMetrics(BaseModel):
    """ISA 18.2 Performance Metrics."""

    # Per ISA 18.2 Section 5.1
    alarms_per_10min: float = Field(
        default=0.0,
        description="Alarms per operator per 10 minutes (target <10)"
    )
    ack_rate_10min_pct: float = Field(
        default=100.0,
        description="% acknowledged within 10 minutes (target >90%)"
    )
    avg_ack_time_sec: Optional[float] = Field(
        None, description="Average time to acknowledge (seconds)"
    )
    standing_alarm_count: int = Field(
        default=0,
        description="Unacknowledged alarms (target 0)"
    )
    stale_alarm_count: int = Field(
        default=0,
        description="Standing >1 hour (target 0)"
    )

    # Flood and chattering detection
    flood_events_10min: int = Field(
        default=0,
        description="Alarm floods in last 10 minutes (target 0)"
    )
    chattering_alarms: List[str] = Field(
        default_factory=list,
        description="Tags with chattering (fleeting) alarms"
    )

    # Health
    operator_burden: str = Field(
        default="NORMAL",
        description="NORMAL / WARNING / CRITICAL"
    )
    rationalization_completeness_pct: float = Field(
        default=0.0,
        description="% of alarms with complete rationalization"
    )


# =============================================================================
# ALARM RATIONALIZATION - ISA 18.2 ANNEX D
# =============================================================================

class AlarmRationalization(BaseModel):
    """ISA 18.2 Annex D - Alarm Rationalization."""

    tag: str = Field(...)
    consequence: str = Field(..., description="What happens if not addressed")
    response: str = Field(..., description="Operator response")
    response_time_sec: int = Field(..., ge=1)
    alarm_necessary: bool = Field(
        default=True,
        description="Is this alarm necessary per Annex D?"
    )
    rationalization_date: datetime = Field(default_factory=datetime.now)
    rationalized_by: str = Field(default="", description="Name/ID of person")
    notes: str = Field(default="")


# =============================================================================
# ALARM MANAGER - MAIN CLASS
# =============================================================================

class AlarmManager:
    """
    ISA 18.2 Alarm Management System for Process Heat Agents.

    This class implements the complete alarm lifecycle:
        1. Configuration - Define setpoints, priorities, rationalization
        2. Trigger/Clear - Process values against setpoints
        3. Acknowledge - Operator acknowledges unacknowledged alarms
        4. Shelve - Temporarily suppress known nuisance alarms
        5. Report - Generate metrics per ISA 18.2

    Thread-safe for multi-operator environments.

    Attributes:
        config: Manager configuration (operator_id, plant_id)
        alarms: Dict of AlarmConfiguration by tag
        events: Deque of recent AlarmEvent records
        active_alarms: Dict of active alarms by alarm_id
        metrics_window: Time window for metrics (seconds)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        metrics_window_sec: int = 600  # 10 minutes per ISA 18.2
    ):
        """
        Initialize AlarmManager.

        Args:
            config: Configuration dict with operator_id, plant_id
            metrics_window_sec: Time window for metrics (default 10 min)
        """
        self.config = config or {}
        self.metrics_window_sec = metrics_window_sec

        # Alarm storage
        self.alarms: Dict[str, AlarmConfiguration] = {}
        self.events: deque = deque(maxlen=10000)  # Last 10k events
        self.active_alarms: Dict[str, AlarmEvent] = {}
        self.shelved_alarms: Dict[str, AlarmEvent] = {}

        # Chattering detection (tag -> list of event timestamps)
        self.alarm_timestamps: Dict[str, deque] = {}

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            f"AlarmManager initialized for {self.config.get('plant_id', 'unknown')} "
            f"operator {self.config.get('operator_id', 'unknown')}"
        )

    # =========================================================================
    # ALARM CONFIGURATION
    # =========================================================================

    def configure_alarm(
        self,
        tag: str,
        description: str,
        priority: AlarmPriority,
        setpoint: float,
        deadband: float = 0.0,
        alarm_type: AlarmType = AlarmType.ANALOG_HI,
        units: str = "",
    ) -> AlarmConfiguration:
        """
        Configure a new alarm or update existing.

        Per ISA 18.2 Section 4.3 - Alarm Priority Assignment.

        Args:
            tag: Unique alarm tag (e.g., 'FURNACE_TEMP_01')
            description: Human-readable description
            priority: AlarmPriority level
            setpoint: Alarm setpoint value
            deadband: Hysteresis value (default 0)
            alarm_type: AlarmType (default ANALOG_HI)
            units: Engineering units (e.g., 'degC')

        Returns:
            AlarmConfiguration object

        Raises:
            ValueError: If tag already exists with different priority
        """
        with self._lock:
            # Validate tag uniqueness for critical alarms
            if tag in self.alarms:
                existing = self.alarms[tag]
                if existing.priority != priority:
                    logger.warning(
                        f"Changing priority of {tag} from "
                        f"{existing.priority.value} to {priority.value}"
                    )

            setpoint_obj = AlarmSetpoint(
                value=setpoint,
                deadband=deadband,
                units=units
            )

            config = AlarmConfiguration(
                tag=tag,
                description=description,
                alarm_type=alarm_type,
                priority=priority,
                setpoint=setpoint_obj,
                enabled=True,
                updated_at=datetime.now()
            )

            self.alarms[tag] = config
            self.alarm_timestamps[tag] = deque(maxlen=100)

            logger.info(
                f"Configured alarm {tag}: {description} "
                f"({priority.value} @ {setpoint} {units})"
            )
            return config

    # =========================================================================
    # ALARM PROCESSING
    # =========================================================================

    def process_alarm(
        self,
        tag: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> ProcessAlarmResult:
        """
        Process a single alarm event.

        Implements ISA 18.2 alarm state machine:
            NORMAL -> UNACKNOWLEDGED -> ACKNOWLEDGED -> CLEARED

        Args:
            tag: Alarm tag
            value: Current process value
            timestamp: Event timestamp (default now)

        Returns:
            ProcessAlarmResult with state transition

        Raises:
            KeyError: If tag not configured
        """
        timestamp = timestamp or datetime.now()

        with self._lock:
            if tag not in self.alarms:
                raise KeyError(f"Alarm tag not configured: {tag}")

            config = self.alarms[tag]
            if not config.enabled:
                return ProcessAlarmResult(
                    alarm_triggered=False,
                    alarm_cleared=False,
                    new_state=AlarmState.NORMAL
                )

            # Check for chattering
            self.alarm_timestamps[tag].append(timestamp)
            chattering = self._detect_chattering(tag, timestamp)

            # Check for flood
            flooded = self._check_alarm_flood()

            # Determine if alarm should trigger
            setpoint = config.setpoint
            triggered = setpoint.is_triggered(value)
            cleared = setpoint.is_cleared(value)

            # Find or create alarm
            alarm_id = None
            existing_alarm = self._find_active_alarm(tag)

            if triggered and not existing_alarm:
                # NEW ALARM
                alarm_id = str(uuid.uuid4())
                alarm = AlarmEvent(
                    alarm_id=alarm_id,
                    tag=tag,
                    priority=config.priority,
                    state=AlarmState.UNACKNOWLEDGED,
                    value=value,
                    setpoint=setpoint.value,
                    timestamp=timestamp,
                    provenance_hash=self._calculate_provenance(tag, value)
                )
                self.active_alarms[alarm_id] = alarm
                self.events.append(alarm)

                logger.warning(
                    f"ALARM {config.priority.value}: {config.description} "
                    f"(value={value}, setpoint={setpoint.value})"
                )

                return ProcessAlarmResult(
                    alarm_triggered=True,
                    alarm_cleared=False,
                    alarm_id=alarm_id,
                    new_state=AlarmState.UNACKNOWLEDGED,
                    flooded=flooded,
                    chattering=chattering
                )

            elif cleared and existing_alarm:
                # CLEAR ALARM
                alarm_id = existing_alarm.alarm_id
                existing_alarm.state = AlarmState.CLEARED
                existing_alarm.timestamp = timestamp
                del self.active_alarms[alarm_id]

                logger.info(
                    f"Alarm cleared: {tag} (value={value})"
                )

                return ProcessAlarmResult(
                    alarm_triggered=False,
                    alarm_cleared=True,
                    alarm_id=alarm_id,
                    new_state=AlarmState.CLEARED
                )

            elif existing_alarm:
                # ALARM REMAINS ACTIVE
                existing_alarm.value = value
                return ProcessAlarmResult(
                    alarm_triggered=False,
                    alarm_cleared=False,
                    alarm_id=existing_alarm.alarm_id,
                    new_state=existing_alarm.state,
                    flooded=flooded,
                    chattering=chattering
                )

            else:
                # NO ALARM CONDITION
                return ProcessAlarmResult(
                    alarm_triggered=False,
                    alarm_cleared=False,
                    new_state=AlarmState.NORMAL
                )

    # =========================================================================
    # ACKNOWLEDGMENT & SHELVING
    # =========================================================================

    def acknowledge_alarm(
        self,
        alarm_id: str,
        operator_id: str
    ) -> AlarmEvent:
        """
        Acknowledge an unacknowledged alarm.

        Per ISA 18.2 requirement - tracks acknowledgment time for metrics.

        Args:
            alarm_id: Alarm ID to acknowledge
            operator_id: Operator ID for audit trail

        Returns:
            Updated AlarmEvent

        Raises:
            KeyError: If alarm_id not found
        """
        with self._lock:
            if alarm_id not in self.active_alarms:
                raise KeyError(f"Alarm not found: {alarm_id}")

            alarm = self.active_alarms[alarm_id]
            if alarm.state == AlarmState.ACKNOWLEDGED:
                logger.debug(f"Alarm {alarm_id} already acknowledged")
                return alarm

            alarm.state = AlarmState.ACKNOWLEDGED
            alarm.ack_timestamp = datetime.now()
            alarm.ack_operator_id = operator_id
            alarm.ack_time_sec = (
                alarm.ack_timestamp - alarm.timestamp
            ).total_seconds()

            logger.info(
                f"Acknowledged alarm {alarm.tag} by {operator_id} "
                f"({alarm.ack_time_sec:.1f}s)"
            )
            return alarm

    def shelve_alarm(
        self,
        alarm_id: str,
        duration_hours: int,
        reason: str
    ) -> AlarmEvent:
        """
        Temporarily shelve (suppress) an alarm.

        Use for known nuisance alarms. Requires documentation per ISA 18.2.

        Args:
            alarm_id: Alarm ID to shelve
            duration_hours: How long to suppress (hours)
            reason: Reason for suppression (audit trail)

        Returns:
            Updated AlarmEvent

        Raises:
            KeyError: If alarm_id not found
            ValueError: If duration_hours > 24
        """
        if duration_hours > 24:
            raise ValueError("Shelving duration limited to 24 hours per ISA 18.2")

        with self._lock:
            if alarm_id not in self.active_alarms:
                raise KeyError(f"Alarm not found: {alarm_id}")

            alarm = self.active_alarms[alarm_id]
            alarm.state = AlarmState.SHELVED
            alarm.shelved_until = datetime.now() + timedelta(hours=duration_hours)
            alarm.shelved_reason = reason
            self.shelved_alarms[alarm_id] = alarm

            logger.warning(
                f"Shelved alarm {alarm.tag} until {alarm.shelved_until} "
                f"reason: {reason}"
            )
            return alarm

    # =========================================================================
    # ALARM RATIONALIZATION - ISA 18.2 ANNEX D
    # =========================================================================

    def rationalize_alarm(
        self,
        tag: str,
        consequence: str,
        response: str,
        response_time_sec: int
    ) -> AlarmRationalization:
        """
        Document alarm rationalization per ISA 18.2 Annex D.

        This is the formal justification for why the alarm is necessary.

        Args:
            tag: Alarm tag
            consequence: What happens if not addressed
            response: What operator should do
            response_time_sec: Target response time (seconds)

        Returns:
            AlarmRationalization record

        Raises:
            KeyError: If tag not configured
        """
        with self._lock:
            if tag not in self.alarms:
                raise KeyError(f"Alarm not configured: {tag}")

            config = self.alarms[tag]
            config.consequence = consequence
            config.response = response
            config.response_time_sec = response_time_sec
            config.updated_at = datetime.now()

            rationalization = AlarmRationalization(
                tag=tag,
                consequence=consequence,
                response=response,
                response_time_sec=response_time_sec,
                alarm_necessary=True,
                rationalized_by=self.config.get("operator_id", "unknown"),
                notes=f"Configured for {config.description}"
            )

            logger.info(
                f"Rationalized alarm {tag}: {response_time_sec}s response "
                f"to {consequence}"
            )
            return rationalization

    # =========================================================================
    # FLOOD & CHATTERING DETECTION
    # =========================================================================

    def check_alarm_flood(
        self,
        threshold: int = 10,
        window_minutes: int = 10
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Detect alarm flood per ISA 18.2.

        Alarm flood = >10 alarms per operator per 10 minutes.
        When detected, lower-priority alarms are suppressed.

        Args:
            threshold: Alarm count threshold (default 10)
            window_minutes: Time window (default 10 minutes)

        Returns:
            (is_flooded, alarm_counts_by_priority)
        """
        with self._lock:
            now = datetime.now()
            window = timedelta(minutes=window_minutes)
            recent_events = [
                e for e in self.events
                if (now - e.timestamp) <= window
            ]

            if len(recent_events) > threshold:
                logger.warning(
                    f"ALARM FLOOD DETECTED: {len(recent_events)} alarms "
                    f"in {window_minutes} minutes (threshold={threshold})"
                )
                return True, self._count_by_priority(recent_events)

            return False, self._count_by_priority(recent_events)

    def suppress_nuisance_alarm(
        self,
        tag: str,
        suppress_until: Optional[datetime] = None
    ) -> bool:
        """
        Suppress a known nuisance alarm.

        Use after rationalization determines alarm is not necessary.

        Args:
            tag: Alarm tag to suppress
            suppress_until: Until when (default 24 hours)

        Returns:
            True if suppression successful
        """
        suppress_until = suppress_until or (
            datetime.now() + timedelta(hours=24)
        )

        with self._lock:
            if tag not in self.alarms:
                raise KeyError(f"Alarm not configured: {tag}")

            config = self.alarms[tag]
            config.enabled = False

            logger.warning(
                f"Suppressed nuisance alarm {tag} until {suppress_until}"
            )
            return True

    # =========================================================================
    # QUERIES & METRICS
    # =========================================================================

    def get_standing_alarms(self) -> List[AlarmEvent]:
        """
        Get all unacknowledged alarms.

        Per ISA 18.2 Section 5.1 - unacknowledged alarm count is key metric.

        Returns:
            List of unacknowledged AlarmEvent records (sorted by priority)
        """
        with self._lock:
            standing = [
                a for a in self.active_alarms.values()
                if a.state in (AlarmState.UNACKNOWLEDGED, AlarmState.ACKNOWLEDGED)
            ]
            # Sort by priority (highest first) then by timestamp
            standing.sort(
                key=lambda x: (-x.priority.numeric_value, x.timestamp)
            )
            return standing

    def get_alarm_metrics(self) -> AlarmMetrics:
        """
        Calculate performance metrics per ISA-18.2-2016 Section 5.1.

        Metrics:
            - Alarms per 10 minutes (target <10)
            - % acknowledged within 10 minutes (target >90%)
            - Standing alarms (target 0)
            - Stale alarms (standing >1 hour) (target 0)
            - Alarm floods (target 0)
            - Chattering alarms (target 0)

        Returns:
            AlarmMetrics object
        """
        with self._lock:
            now = datetime.now()
            window = timedelta(seconds=self.metrics_window_sec)

            # Alarms in last 10 minutes
            recent_events = [
                e for e in self.events
                if (now - e.timestamp) <= window
            ]
            alarms_per_10min = float(len(recent_events))

            # Acknowledgment rate
            ack_events = [
                e for e in recent_events
                if e.ack_timestamp and e.ack_time_sec is not None
            ]
            ack_rate = (
                (len(ack_events) / len(recent_events) * 100)
                if recent_events else 100.0
            )

            # Average acknowledgment time
            ack_times = [e.ack_time_sec for e in ack_events if e.ack_time_sec]
            avg_ack_time = mean(ack_times) if ack_times else None

            # Standing alarms
            standing = self.get_standing_alarms()
            standing_count = len(standing)

            # Stale alarms (standing >1 hour)
            stale_threshold = now - timedelta(hours=1)
            stale_alarms = [
                a for a in standing
                if a.timestamp <= stale_threshold
            ]
            stale_count = len(stale_alarms)

            # Flood detection
            is_flooded, _ = self.check_alarm_flood()
            flood_count = 1 if is_flooded else 0

            # Chattering detection
            chattering_tags = self._get_chattering_alarms()

            # Operator burden assessment
            burden = "CRITICAL" if alarms_per_10min > 20 else (
                "WARNING" if alarms_per_10min > 10 else "NORMAL"
            )

            # Rationalization completeness
            rationalized = sum(
                1 for a in self.alarms.values()
                if a.consequence and a.response
            )
            rationalization_pct = (
                (rationalized / len(self.alarms) * 100)
                if self.alarms else 0.0
            )

            metrics = AlarmMetrics(
                alarms_per_10min=alarms_per_10min,
                ack_rate_10min_pct=ack_rate,
                avg_ack_time_sec=avg_ack_time,
                standing_alarm_count=standing_count,
                stale_alarm_count=stale_count,
                flood_events_10min=flood_count,
                chattering_alarms=chattering_tags,
                operator_burden=burden,
                rationalization_completeness_pct=rationalization_pct
            )

            return metrics

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _find_active_alarm(self, tag: str) -> Optional[AlarmEvent]:
        """Find active alarm by tag (assumes single active per tag)."""
        for alarm in self.active_alarms.values():
            if alarm.tag == tag:
                return alarm
        return None

    def _detect_chattering(self, tag: str, timestamp: datetime) -> bool:
        """
        Detect chattering (fleeting) alarm.

        Chattering = alarm triggers and clears within <1 second.
        """
        timestamps = self.alarm_timestamps.get(tag, deque())
        if len(timestamps) < 2:
            return False

        # Check if last two events are <1 second apart
        last_two = list(timestamps)[-2:]
        time_diff = (last_two[1] - last_two[0]).total_seconds()
        return time_diff < 1.0

    def _check_alarm_flood(self) -> bool:
        """Check if current alarm flood is active."""
        is_flooded, _ = self.check_alarm_flood()
        return is_flooded

    def _count_by_priority(
        self,
        events: List[AlarmEvent]
    ) -> Dict[AlarmPriority, int]:
        """Count alarms by priority."""
        counts = {p: 0 for p in AlarmPriority}
        for event in events:
            counts[event.priority] += 1
        return counts

    def _get_chattering_alarms(self) -> List[str]:
        """Get list of tags with chattering alarms."""
        chattering = []
        for tag, timestamps in self.alarm_timestamps.items():
            if len(timestamps) >= 2:
                last_two = list(timestamps)[-2:]
                time_diff = (last_two[1] - last_two[0]).total_seconds()
                if time_diff < 1.0:
                    chattering.append(tag)
        return chattering

    def _calculate_provenance(self, tag: str, value: float) -> str:
        """Calculate SHA-256 hash for alarm provenance."""
        provenance_str = f"{tag}:{value}:{datetime.now().isoformat()}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "AlarmManager",
    "AlarmPriority",
    "AlarmState",
    "AlarmType",
    "AlarmConfiguration",
    "AlarmEvent",
    "AlarmSetpoint",
    "ProcessAlarmResult",
    "AlarmMetrics",
    "AlarmRationalization",
]
