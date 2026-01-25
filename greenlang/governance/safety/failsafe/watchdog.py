"""
Watchdog - Watchdog Timer Framework

This module implements watchdog timer functionality for Safety Instrumented
Systems per IEC 61511. Watchdog timers are critical for detecting:
- Software hangs
- CPU failures
- Communication failures
- Logic solver malfunctions

Reference: IEC 61508-2 Clause 7.4.3, IEC 61511-1 Clause 11.6

Example:
    >>> from greenlang.safety.failsafe.watchdog import Watchdog, WatchdogConfig
    >>> config = WatchdogConfig(timeout_ms=1000, action_on_timeout="trip")
    >>> watchdog = Watchdog(config)
    >>> watchdog.start()
    >>> watchdog.kick()  # Must call periodically
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import threading
import time
import uuid

logger = logging.getLogger(__name__)


class WatchdogState(str, Enum):
    """Watchdog timer states."""

    STOPPED = "stopped"
    RUNNING = "running"
    EXPIRED = "expired"
    DISABLED = "disabled"


class TimeoutAction(str, Enum):
    """Action to take on watchdog timeout."""

    TRIP = "trip"  # Initiate safety trip
    ALARM = "alarm"  # Generate alarm only
    RESET = "reset"  # Attempt system reset
    LOG = "log"  # Log only
    CALLBACK = "callback"  # Call registered callback


class WatchdogConfig(BaseModel):
    """Configuration for watchdog timer."""

    watchdog_id: str = Field(
        default_factory=lambda: f"WD-{uuid.uuid4().hex[:6].upper()}",
        description="Watchdog identifier"
    )
    timeout_ms: float = Field(
        default=1000.0,
        gt=0,
        description="Timeout period in milliseconds"
    )
    action_on_timeout: TimeoutAction = Field(
        default=TimeoutAction.TRIP,
        description="Action on timeout"
    )
    auto_restart: bool = Field(
        default=False,
        description="Auto-restart after timeout"
    )
    min_kick_interval_ms: float = Field(
        default=0,
        ge=0,
        description="Minimum interval between kicks (anti-rapid-kick)"
    )
    max_kick_interval_ms: Optional[float] = Field(
        None,
        description="Maximum interval between kicks"
    )
    description: str = Field(
        default="",
        description="Watchdog description"
    )


class WatchdogEvent(BaseModel):
    """Record of a watchdog event."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Event identifier"
    )
    watchdog_id: str = Field(
        ...,
        description="Watchdog identifier"
    )
    event_type: str = Field(
        ...,
        description="Event type (start, kick, timeout, stop)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )
    time_since_last_kick_ms: Optional[float] = Field(
        None,
        description="Time since last kick"
    )
    state_before: WatchdogState = Field(
        ...,
        description="State before event"
    )
    state_after: WatchdogState = Field(
        ...,
        description="State after event"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event details"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Watchdog:
    """
    Watchdog Timer Implementation.

    Implements a watchdog timer for SIS health monitoring.
    Features:
    - Configurable timeout periods
    - Multiple timeout actions
    - Event logging for audit trail
    - Thread-safe operation
    - Kick interval validation

    The watchdog follows fail-safe principles:
    - Timeout triggers safe action
    - Missing kicks detected
    - Complete event history

    Attributes:
        config: WatchdogConfig settings
        state: Current watchdog state
        events: List of recorded events

    Example:
        >>> watchdog = Watchdog(WatchdogConfig(timeout_ms=500))
        >>> watchdog.start()
        >>> # Must call kick() before timeout
        >>> watchdog.kick()
    """

    def __init__(
        self,
        config: WatchdogConfig,
        timeout_callback: Optional[Callable] = None
    ):
        """
        Initialize Watchdog.

        Args:
            config: WatchdogConfig with settings
            timeout_callback: Optional callback function on timeout
        """
        self.config = config
        self.timeout_callback = timeout_callback
        self.state = WatchdogState.STOPPED
        self.events: List[WatchdogEvent] = []

        self._last_kick_time: Optional[datetime] = None
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._kick_count = 0
        self._timeout_count = 0
        self._start_time: Optional[datetime] = None

        logger.info(
            f"Watchdog initialized: {config.watchdog_id}, "
            f"timeout={config.timeout_ms}ms"
        )

    def start(self) -> bool:
        """
        Start the watchdog timer.

        Returns:
            True if started successfully
        """
        with self._lock:
            if self.state == WatchdogState.RUNNING:
                logger.warning(f"Watchdog {self.config.watchdog_id} already running")
                return False

            state_before = self.state
            self.state = WatchdogState.RUNNING
            self._start_time = datetime.utcnow()
            self._last_kick_time = datetime.utcnow()

            # Start timer
            self._start_timer()

            # Log event
            self._log_event("start", state_before, self.state)

            logger.info(f"Watchdog {self.config.watchdog_id} started")
            return True

    def stop(self) -> bool:
        """
        Stop the watchdog timer.

        Returns:
            True if stopped successfully
        """
        with self._lock:
            if self.state == WatchdogState.STOPPED:
                return True

            state_before = self.state

            # Cancel timer
            if self._timer:
                self._timer.cancel()
                self._timer = None

            self.state = WatchdogState.STOPPED

            # Log event
            self._log_event("stop", state_before, self.state)

            logger.info(f"Watchdog {self.config.watchdog_id} stopped")
            return True

    def kick(self) -> bool:
        """
        Kick (reset) the watchdog timer.

        Must be called periodically to prevent timeout.

        Returns:
            True if kick accepted
        """
        with self._lock:
            if self.state != WatchdogState.RUNNING:
                logger.warning(
                    f"Cannot kick watchdog {self.config.watchdog_id}: "
                    f"state={self.state.value}"
                )
                return False

            now = datetime.utcnow()

            # Check minimum kick interval
            if self._last_kick_time and self.config.min_kick_interval_ms > 0:
                interval = (now - self._last_kick_time).total_seconds() * 1000
                if interval < self.config.min_kick_interval_ms:
                    logger.warning(
                        f"Kick too fast: {interval}ms < "
                        f"{self.config.min_kick_interval_ms}ms"
                    )
                    # Still accept but log warning

            # Calculate time since last kick
            time_since_last = None
            if self._last_kick_time:
                time_since_last = (now - self._last_kick_time).total_seconds() * 1000

            # Reset timer
            if self._timer:
                self._timer.cancel()
            self._start_timer()

            self._last_kick_time = now
            self._kick_count += 1

            # Log event
            self._log_event(
                "kick",
                self.state,
                self.state,
                {"time_since_last_ms": time_since_last}
            )

            return True

    def get_status(self) -> Dict[str, Any]:
        """
        Get current watchdog status.

        Returns:
            Status dictionary
        """
        with self._lock:
            now = datetime.utcnow()

            time_since_last_kick = None
            time_remaining = None

            if self._last_kick_time:
                elapsed = (now - self._last_kick_time).total_seconds() * 1000
                time_since_last_kick = elapsed
                time_remaining = max(0, self.config.timeout_ms - elapsed)

            return {
                "watchdog_id": self.config.watchdog_id,
                "state": self.state.value,
                "timeout_ms": self.config.timeout_ms,
                "time_since_last_kick_ms": time_since_last_kick,
                "time_remaining_ms": time_remaining,
                "kick_count": self._kick_count,
                "timeout_count": self._timeout_count,
                "start_time": self._start_time.isoformat() if self._start_time else None,
                "action_on_timeout": self.config.action_on_timeout.value,
            }

    def get_event_history(
        self,
        limit: int = 100
    ) -> List[WatchdogEvent]:
        """
        Get watchdog event history.

        Args:
            limit: Maximum events to return

        Returns:
            List of WatchdogEvent objects
        """
        return self.events[-limit:]

    def _start_timer(self) -> None:
        """Start internal timer thread."""
        timeout_seconds = self.config.timeout_ms / 1000.0
        self._timer = threading.Timer(timeout_seconds, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    def _on_timeout(self) -> None:
        """Handle watchdog timeout."""
        with self._lock:
            if self.state != WatchdogState.RUNNING:
                return

            state_before = self.state
            self.state = WatchdogState.EXPIRED
            self._timeout_count += 1

            # Calculate time since last kick
            time_since_last = None
            if self._last_kick_time:
                time_since_last = (
                    datetime.utcnow() - self._last_kick_time
                ).total_seconds() * 1000

            # Log event
            self._log_event(
                "timeout",
                state_before,
                self.state,
                {"time_since_last_ms": time_since_last}
            )

            logger.warning(
                f"Watchdog {self.config.watchdog_id} TIMEOUT! "
                f"Action: {self.config.action_on_timeout.value}"
            )

        # Execute timeout action (outside lock)
        self._execute_timeout_action()

        # Auto-restart if configured
        if self.config.auto_restart:
            with self._lock:
                self.state = WatchdogState.RUNNING
                self._last_kick_time = datetime.utcnow()
                self._start_timer()
                logger.info(f"Watchdog {self.config.watchdog_id} auto-restarted")

    def _execute_timeout_action(self) -> None:
        """Execute configured timeout action."""
        action = self.config.action_on_timeout

        if action == TimeoutAction.TRIP:
            logger.critical(
                f"Watchdog {self.config.watchdog_id}: TRIP action triggered"
            )
            # In production, this would trigger safety shutdown
            if self.timeout_callback:
                self.timeout_callback("TRIP")

        elif action == TimeoutAction.ALARM:
            logger.error(
                f"Watchdog {self.config.watchdog_id}: ALARM generated"
            )
            if self.timeout_callback:
                self.timeout_callback("ALARM")

        elif action == TimeoutAction.RESET:
            logger.warning(
                f"Watchdog {self.config.watchdog_id}: RESET requested"
            )
            if self.timeout_callback:
                self.timeout_callback("RESET")

        elif action == TimeoutAction.LOG:
            logger.warning(
                f"Watchdog {self.config.watchdog_id}: Timeout logged"
            )

        elif action == TimeoutAction.CALLBACK:
            if self.timeout_callback:
                self.timeout_callback("TIMEOUT")
            else:
                logger.warning(
                    f"Watchdog {self.config.watchdog_id}: "
                    "No callback registered for CALLBACK action"
                )

    def _log_event(
        self,
        event_type: str,
        state_before: WatchdogState,
        state_after: WatchdogState,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log watchdog event."""
        event = WatchdogEvent(
            watchdog_id=self.config.watchdog_id,
            event_type=event_type,
            state_before=state_before,
            state_after=state_after,
            details=details or {},
        )

        if details and "time_since_last_ms" in details:
            event.time_since_last_kick_ms = details["time_since_last_ms"]

        self.events.append(event)

        # Trim history if too long
        if len(self.events) > 10000:
            self.events = self.events[-5000:]

    def get_provenance_hash(self) -> str:
        """Get provenance hash for current state."""
        provenance_str = (
            f"{self.config.watchdog_id}|"
            f"{self.state.value}|"
            f"{self._kick_count}|"
            f"{self._timeout_count}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


class WatchdogManager:
    """
    Manager for multiple watchdog timers.

    Provides centralized management of watchdog timers
    for a safety system.
    """

    def __init__(self):
        """Initialize WatchdogManager."""
        self.watchdogs: Dict[str, Watchdog] = {}
        logger.info("WatchdogManager initialized")

    def create_watchdog(
        self,
        config: WatchdogConfig,
        timeout_callback: Optional[Callable] = None
    ) -> Watchdog:
        """
        Create and register a new watchdog.

        Args:
            config: WatchdogConfig for new watchdog
            timeout_callback: Optional timeout callback

        Returns:
            Created Watchdog instance
        """
        watchdog = Watchdog(config, timeout_callback)
        self.watchdogs[config.watchdog_id] = watchdog
        return watchdog

    def get_watchdog(self, watchdog_id: str) -> Optional[Watchdog]:
        """Get watchdog by ID."""
        return self.watchdogs.get(watchdog_id)

    def start_all(self) -> int:
        """
        Start all watchdogs.

        Returns:
            Number of watchdogs started
        """
        count = 0
        for watchdog in self.watchdogs.values():
            if watchdog.start():
                count += 1
        return count

    def stop_all(self) -> int:
        """
        Stop all watchdogs.

        Returns:
            Number of watchdogs stopped
        """
        count = 0
        for watchdog in self.watchdogs.values():
            if watchdog.stop():
                count += 1
        return count

    def get_status_all(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all watchdogs."""
        return {
            wd_id: wd.get_status()
            for wd_id, wd in self.watchdogs.items()
        }

    def get_expired_watchdogs(self) -> List[str]:
        """Get list of expired watchdog IDs."""
        return [
            wd_id for wd_id, wd in self.watchdogs.items()
            if wd.state == WatchdogState.EXPIRED
        ]
