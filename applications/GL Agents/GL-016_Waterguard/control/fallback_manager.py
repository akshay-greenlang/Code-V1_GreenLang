# -*- coding: utf-8 -*-
"""
GL-016 Waterguard Fallback Manager Module

Degraded operation handling with fixed baseline schedules, trigger conditions,
and automatic recovery.

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import hashlib
import logging
import threading

logger = logging.getLogger(__name__)


class FallbackTrigger(str, Enum):
    """Fallback trigger conditions."""
    DATA_QUALITY_LOSS = "data_quality_loss"
    COMM_FAILURE = "comm_failure"
    ANALYZER_FAULT = "analyzer_fault"
    SENSOR_FAULT = "sensor_fault"
    CALCULATION_ERROR = "calculation_error"
    SAFETY_INTERLOCK = "safety_interlock"
    MANUAL_TRIGGER = "manual_trigger"
    WATCHDOG_TIMEOUT = "watchdog_timeout"


class RecoveryCondition(BaseModel):
    """Condition for automatic recovery from fallback."""
    trigger: FallbackTrigger
    min_recovery_time_seconds: float = Field(default=300.0, ge=0)
    max_recovery_time_seconds: float = Field(default=3600.0, ge=0)
    required_good_readings: int = Field(default=10, ge=1)
    auto_recovery_enabled: bool = Field(default=True)


class BaselineSchedule(BaseModel):
    """Baseline schedule entry for fallback operation."""
    parameter_name: str = Field(...)
    value: float = Field(...)
    time_of_day_start: str = Field(default="00:00")
    time_of_day_end: str = Field(default="23:59")
    day_of_week: Optional[List[int]] = Field(default=None)
    description: str = Field(default="")


class FallbackSchedule(BaseModel):
    """Complete fallback schedule."""
    schedule_id: str = Field(...)
    schedules: List[BaselineSchedule] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)
    created_time: datetime = Field(default_factory=datetime.now)
    modified_time: datetime = Field(default_factory=datetime.now)


class FallbackState(BaseModel):
    """Current fallback manager state."""
    active: bool = Field(default=False)
    trigger: Optional[FallbackTrigger] = Field(default=None)
    trigger_time: Optional[datetime] = Field(default=None)
    active_schedule_id: Optional[str] = Field(default=None)
    recovery_eligible: bool = Field(default=False)
    good_reading_count: int = Field(default=0)
    last_check_time: datetime = Field(default_factory=datetime.now)


class FallbackConfig(BaseModel):
    """Fallback manager configuration."""
    schedules: List[FallbackSchedule] = Field(default_factory=list)
    recovery_conditions: List[RecoveryCondition] = Field(default_factory=list)
    default_blowdown_percent: float = Field(default=10.0, ge=0, le=100)
    default_dosing_percent: float = Field(default=50.0, ge=0, le=100)
    check_interval_seconds: float = Field(default=10.0, ge=1.0)
    enable_auto_recovery: bool = Field(default=True)


class FallbackManager:
    """Fallback mode management with baseline schedules."""

    def __init__(self, config: FallbackConfig, alert_callback: Optional[Callable] = None):
        self.config = config
        self._lock = threading.RLock()
        self._state = FallbackState()
        self._schedules = {s.schedule_id: s for s in config.schedules}
        self._recovery_conditions = {c.trigger: c for c in config.recovery_conditions}
        self._alert_callback = alert_callback
        self._trigger_history: List[Dict[str, Any]] = []
        logger.info("FallbackManager initialized")

    def trigger_fallback(self, trigger: FallbackTrigger, reason: str = "") -> bool:
        """Trigger fallback mode."""
        with self._lock:
            if self._state.active:
                logger.info(f"Fallback already active, ignoring trigger: {trigger.value}")
                return False

            self._state.active = True
            self._state.trigger = trigger
            self._state.trigger_time = datetime.now()
            self._state.recovery_eligible = False
            self._state.good_reading_count = 0

            # Select appropriate schedule
            self._state.active_schedule_id = self._select_schedule()

            # Log trigger
            self._trigger_history.append({
                "trigger": trigger.value,
                "reason": reason,
                "timestamp": datetime.now(),
                "schedule_id": self._state.active_schedule_id
            })

            logger.warning(f"Fallback triggered: {trigger.value} - {reason}")

            if self._alert_callback:
                self._alert_callback({
                    "type": "fallback_triggered",
                    "trigger": trigger.value,
                    "reason": reason
                })

            return True

    def check_recovery(self, data_quality_good: bool = False) -> bool:
        """Check if recovery from fallback is possible."""
        with self._lock:
            if not self._state.active:
                return True

            if not self.config.enable_auto_recovery:
                return False

            trigger = self._state.trigger
            condition = self._recovery_conditions.get(trigger)

            if not condition or not condition.auto_recovery_enabled:
                return False

            # Check minimum recovery time
            if self._state.trigger_time:
                elapsed = (datetime.now() - self._state.trigger_time).total_seconds()
                if elapsed < condition.min_recovery_time_seconds:
                    return False

            # Check good readings
            if data_quality_good:
                self._state.good_reading_count += 1
            else:
                self._state.good_reading_count = 0

            if self._state.good_reading_count >= condition.required_good_readings:
                self._state.recovery_eligible = True
                return True

            return False

    def recover_from_fallback(self, force: bool = False) -> bool:
        """Recover from fallback mode."""
        with self._lock:
            if not self._state.active:
                return True

            if not force and not self._state.recovery_eligible:
                logger.warning("Recovery not eligible, use force=True to override")
                return False

            prev_trigger = self._state.trigger
            self._state.active = False
            self._state.trigger = None
            self._state.trigger_time = None
            self._state.active_schedule_id = None
            self._state.recovery_eligible = False
            self._state.good_reading_count = 0

            logger.info(f"Recovered from fallback (trigger was: {prev_trigger.value if prev_trigger else 'None'})")

            if self._alert_callback:
                self._alert_callback({
                    "type": "fallback_recovered",
                    "previous_trigger": prev_trigger.value if prev_trigger else None
                })

            return True

    def get_fallback_setpoints(self) -> Dict[str, float]:
        """Get current fallback setpoints based on schedule."""
        with self._lock:
            if not self._state.active:
                return {}

            schedule_id = self._state.active_schedule_id
            if not schedule_id:
                # Use defaults
                return {
                    "blowdown_percent": self.config.default_blowdown_percent,
                    "dosing_percent": self.config.default_dosing_percent
                }

            schedule = self._schedules.get(schedule_id)
            if not schedule:
                return {}

            now = datetime.now()
            current_time = now.strftime("%H:%M")
            current_dow = now.weekday()

            setpoints = {}
            for entry in schedule.schedules:
                if self._is_schedule_active(entry, current_time, current_dow):
                    setpoints[entry.parameter_name] = entry.value

            return setpoints

    def get_state(self) -> FallbackState:
        """Get current fallback state."""
        with self._lock:
            return self._state.copy()

    def is_active(self) -> bool:
        """Check if fallback is active."""
        return self._state.active

    def get_trigger_history(self, hours: float = 24.0) -> List[Dict[str, Any]]:
        """Get trigger history."""
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            return [t for t in self._trigger_history if t["timestamp"] > cutoff]

    def add_schedule(self, schedule: FallbackSchedule) -> bool:
        """Add a fallback schedule."""
        with self._lock:
            self._schedules[schedule.schedule_id] = schedule
            logger.info(f"Added fallback schedule: {schedule.schedule_id}")
            return True

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a fallback schedule."""
        with self._lock:
            if schedule_id in self._schedules:
                del self._schedules[schedule_id]
                logger.info(f"Removed fallback schedule: {schedule_id}")
                return True
            return False

    def _select_schedule(self) -> Optional[str]:
        """Select the appropriate fallback schedule."""
        # Select highest priority schedule
        if not self._schedules:
            return None

        schedules = sorted(self._schedules.values(), key=lambda s: s.priority)
        return schedules[0].schedule_id if schedules else None

    def _is_schedule_active(self, entry: BaselineSchedule, current_time: str, current_dow: int) -> bool:
        """Check if a schedule entry is active for current time."""
        # Check day of week
        if entry.day_of_week is not None:
            if current_dow not in entry.day_of_week:
                return False

        # Check time of day
        if entry.time_of_day_start <= current_time <= entry.time_of_day_end:
            return True

        return False
