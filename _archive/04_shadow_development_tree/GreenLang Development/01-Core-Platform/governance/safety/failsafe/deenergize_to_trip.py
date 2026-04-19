"""
DeenergizeToTrip - De-energize-to-Trip Implementation

This module implements the de-energize-to-trip fail-safe pattern
per IEC 61511. This is the fundamental principle that:
- Loss of power = Safe state
- No energy needed to achieve safe state
- Fail-safe by design

Reference: IEC 61511-1 Clause 11.5.3

Example:
    >>> from greenlang.safety.failsafe.deenergize_to_trip import DeenergizeToTrip
    >>> trip = DeenergizeToTrip()
    >>> trip.add_channel("XV-001", safe_state="CLOSED")
    >>> result = trip.execute_trip()
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime
import time
import uuid

logger = logging.getLogger(__name__)


class ChannelState(str, Enum):
    """State of a trip channel."""

    ENERGIZED = "energized"  # Normal operation
    DE_ENERGIZED = "de_energized"  # Safe state (tripped)
    UNKNOWN = "unknown"  # State unknown
    FAULT = "fault"  # Fault detected


class TripInitiator(str, Enum):
    """Source of trip initiation."""

    AUTOMATIC = "automatic"  # Logic solver initiated
    MANUAL = "manual"  # Manual pushbutton
    WATCHDOG = "watchdog"  # Watchdog timeout
    POWER_LOSS = "power_loss"  # Power failure
    COMMUNICATION_LOSS = "communication_loss"  # Comm failure
    EXTERNAL = "external"  # External system


class TripChannel(BaseModel):
    """Trip channel definition."""

    channel_id: str = Field(
        ...,
        description="Channel identifier"
    )
    equipment_tag: str = Field(
        ...,
        description="Associated equipment tag"
    )
    description: str = Field(
        default="",
        description="Channel description"
    )
    safe_state: str = Field(
        ...,
        description="Safe state (e.g., 'CLOSED', 'OPEN', 'OFF')"
    )
    current_state: ChannelState = Field(
        default=ChannelState.ENERGIZED,
        description="Current channel state"
    )
    response_time_ms: float = Field(
        default=1000.0,
        gt=0,
        description="Expected response time (ms)"
    )
    is_fail_safe: bool = Field(
        default=True,
        description="Is channel fail-safe by design"
    )
    last_trip_time: Optional[datetime] = Field(
        None,
        description="Last trip timestamp"
    )
    trip_count: int = Field(
        default=0,
        description="Total trip count"
    )


class TripResult(BaseModel):
    """Result of trip execution."""

    trip_id: str = Field(
        default_factory=lambda: f"TRIP-{uuid.uuid4().hex[:8].upper()}",
        description="Trip identifier"
    )
    initiator: TripInitiator = Field(
        ...,
        description="Trip initiator"
    )
    initiated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Trip initiation time"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Trip completion time"
    )
    duration_ms: Optional[float] = Field(
        None,
        description="Trip duration (ms)"
    )
    channels_tripped: List[str] = Field(
        default_factory=list,
        description="Channels that tripped"
    )
    channels_failed: List[str] = Field(
        default_factory=list,
        description="Channels that failed to trip"
    )
    all_channels_safe: bool = Field(
        default=False,
        description="All channels in safe state"
    )
    response_time_met: bool = Field(
        default=True,
        description="Response time requirement met"
    )
    channel_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed channel results"
    )
    safe_state_verified: bool = Field(
        default=False,
        description="Safe state verified"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DeenergizeToTrip:
    """
    De-energize-to-Trip Controller.

    Implements the fundamental fail-safe principle where loss of
    energy results in safe state. This is the gold standard for
    SIS design per IEC 61511.

    Features:
    - Multi-channel trip management
    - Response time tracking
    - State verification
    - Complete audit trail

    The controller follows fail-safe principles:
    - De-energize = Safe
    - No power = Safe
    - Fail closed (for isolation valves)

    Attributes:
        channels: Dict of TripChannel definitions
        max_response_time_ms: Maximum allowed response time

    Example:
        >>> controller = DeenergizeToTrip()
        >>> controller.add_channel("XV-001", safe_state="CLOSED")
        >>> result = controller.execute_trip(TripInitiator.AUTOMATIC)
    """

    def __init__(
        self,
        max_response_time_ms: float = 1000.0,
        verify_safe_state: bool = True,
        state_reader: Optional[Callable[[str], ChannelState]] = None
    ):
        """
        Initialize DeenergizeToTrip controller.

        Args:
            max_response_time_ms: Maximum response time requirement
            verify_safe_state: Verify safe state after trip
            state_reader: Callback to read actual channel state
        """
        self.channels: Dict[str, TripChannel] = {}
        self.max_response_time_ms = max_response_time_ms
        self.verify_safe_state = verify_safe_state
        self.state_reader = state_reader or self._default_state_reader
        self.trip_history: List[TripResult] = []

        logger.info(
            f"DeenergizeToTrip initialized, max response: {max_response_time_ms}ms"
        )

    def add_channel(
        self,
        channel_id: str,
        equipment_tag: str,
        safe_state: str,
        description: str = "",
        response_time_ms: float = 1000.0,
        is_fail_safe: bool = True
    ) -> TripChannel:
        """
        Add a trip channel.

        Args:
            channel_id: Channel identifier
            equipment_tag: Associated equipment tag
            safe_state: Safe state definition
            description: Channel description
            response_time_ms: Expected response time
            is_fail_safe: Is channel fail-safe by design

        Returns:
            Created TripChannel
        """
        if not is_fail_safe:
            logger.warning(
                f"Channel {channel_id} is NOT fail-safe. "
                "Consider design change."
            )

        channel = TripChannel(
            channel_id=channel_id,
            equipment_tag=equipment_tag,
            description=description,
            safe_state=safe_state,
            response_time_ms=response_time_ms,
            is_fail_safe=is_fail_safe,
        )

        self.channels[channel_id] = channel

        logger.info(f"Trip channel added: {channel_id} -> {safe_state}")

        return channel

    def execute_trip(
        self,
        initiator: TripInitiator = TripInitiator.AUTOMATIC,
        channel_ids: Optional[List[str]] = None,
        force: bool = False
    ) -> TripResult:
        """
        Execute de-energize-to-trip.

        Args:
            initiator: Source of trip
            channel_ids: Specific channels to trip (all if None)
            force: Force trip even if already de-energized

        Returns:
            TripResult with execution details
        """
        initiated_at = datetime.utcnow()

        logger.warning(
            f"TRIP INITIATED by {initiator.value} at {initiated_at.isoformat()}"
        )

        # Determine channels to trip
        if channel_ids:
            channels_to_trip = {
                cid: ch for cid, ch in self.channels.items()
                if cid in channel_ids
            }
        else:
            channels_to_trip = self.channels.copy()

        channels_tripped = []
        channels_failed = []
        channel_details = []
        max_time = 0.0

        # Execute trip on each channel
        for channel_id, channel in channels_to_trip.items():
            start_time = time.time()

            # De-energize channel
            success, actual_time = self._deenergize_channel(
                channel,
                force=force
            )

            elapsed_ms = (time.time() - start_time) * 1000
            max_time = max(max_time, elapsed_ms)

            detail = {
                "channel_id": channel_id,
                "equipment_tag": channel.equipment_tag,
                "safe_state": channel.safe_state,
                "success": success,
                "response_time_ms": elapsed_ms,
                "response_time_met": elapsed_ms <= channel.response_time_ms,
            }

            if success:
                channels_tripped.append(channel_id)
                channel.current_state = ChannelState.DE_ENERGIZED
                channel.last_trip_time = datetime.utcnow()
                channel.trip_count += 1
            else:
                channels_failed.append(channel_id)
                detail["failure_reason"] = "De-energize failed"

            channel_details.append(detail)

        completed_at = datetime.utcnow()
        duration_ms = (completed_at - initiated_at).total_seconds() * 1000

        # Verify safe state if enabled
        safe_state_verified = False
        if self.verify_safe_state and not channels_failed:
            safe_state_verified = self._verify_all_safe()

        # Check overall response time
        response_time_met = max_time <= self.max_response_time_ms

        # Build result
        result = TripResult(
            initiator=initiator,
            initiated_at=initiated_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            channels_tripped=channels_tripped,
            channels_failed=channels_failed,
            all_channels_safe=len(channels_failed) == 0,
            response_time_met=response_time_met,
            channel_details=channel_details,
            safe_state_verified=safe_state_verified,
        )

        # Calculate provenance
        result.provenance_hash = self._calculate_provenance(result)

        # Store in history
        self.trip_history.append(result)

        logger.warning(
            f"TRIP COMPLETED: {len(channels_tripped)}/{len(channels_to_trip)} "
            f"channels in {duration_ms:.0f}ms"
        )

        if channels_failed:
            logger.error(f"TRIP FAILURES: {channels_failed}")

        return result

    def reset_channel(
        self,
        channel_id: str,
        reset_by: str
    ) -> bool:
        """
        Reset (re-energize) a tripped channel.

        Note: Reset should only be done after confirming safe to restart.

        Args:
            channel_id: Channel to reset
            reset_by: Person authorizing reset

        Returns:
            True if reset successful
        """
        if channel_id not in self.channels:
            logger.error(f"Channel not found: {channel_id}")
            return False

        channel = self.channels[channel_id]

        if channel.current_state != ChannelState.DE_ENERGIZED:
            logger.warning(
                f"Channel {channel_id} is not de-energized: {channel.current_state}"
            )

        # Re-energize
        channel.current_state = ChannelState.ENERGIZED

        logger.info(f"Channel {channel_id} reset by {reset_by}")

        return True

    def reset_all(self, reset_by: str) -> Dict[str, bool]:
        """
        Reset all tripped channels.

        Args:
            reset_by: Person authorizing reset

        Returns:
            Dict of channel_id to reset success
        """
        results = {}
        for channel_id in self.channels:
            results[channel_id] = self.reset_channel(channel_id, reset_by)

        logger.info(f"All channels reset by {reset_by}")

        return results

    def get_channel_status(
        self,
        channel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get status of channel(s).

        Args:
            channel_id: Specific channel (all if None)

        Returns:
            Status dictionary
        """
        if channel_id:
            if channel_id not in self.channels:
                return {"error": f"Channel not found: {channel_id}"}
            channels = {channel_id: self.channels[channel_id]}
        else:
            channels = self.channels

        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_channels": len(channels),
            "channels_energized": sum(
                1 for ch in channels.values()
                if ch.current_state == ChannelState.ENERGIZED
            ),
            "channels_de_energized": sum(
                1 for ch in channels.values()
                if ch.current_state == ChannelState.DE_ENERGIZED
            ),
            "channels_faulted": sum(
                1 for ch in channels.values()
                if ch.current_state == ChannelState.FAULT
            ),
            "channel_details": [
                {
                    "channel_id": ch.channel_id,
                    "equipment_tag": ch.equipment_tag,
                    "state": ch.current_state.value,
                    "safe_state": ch.safe_state,
                    "trip_count": ch.trip_count,
                    "last_trip": ch.last_trip_time.isoformat() if ch.last_trip_time else None,
                }
                for ch in channels.values()
            ]
        }

        return status

    def check_fail_safe_integrity(self) -> Dict[str, Any]:
        """
        Check fail-safe integrity of all channels.

        Returns:
            Integrity check results
        """
        issues = []
        warnings = []

        for channel in self.channels.values():
            # Check if channel is truly fail-safe
            if not channel.is_fail_safe:
                issues.append({
                    "channel_id": channel.channel_id,
                    "issue": "Channel is not fail-safe by design",
                    "severity": "HIGH"
                })

            # Check response time
            if channel.response_time_ms > self.max_response_time_ms:
                warnings.append({
                    "channel_id": channel.channel_id,
                    "warning": f"Response time {channel.response_time_ms}ms exceeds max {self.max_response_time_ms}ms",
                })

            # Check for unknown states
            if channel.current_state == ChannelState.UNKNOWN:
                issues.append({
                    "channel_id": channel.channel_id,
                    "issue": "Channel state unknown - verify connectivity",
                    "severity": "MEDIUM"
                })

        return {
            "check_timestamp": datetime.utcnow().isoformat(),
            "total_channels": len(self.channels),
            "fail_safe_count": sum(1 for ch in self.channels.values() if ch.is_fail_safe),
            "issues": issues,
            "warnings": warnings,
            "integrity_ok": len(issues) == 0,
        }

    def get_trip_history(
        self,
        limit: int = 100
    ) -> List[TripResult]:
        """Get trip history."""
        return self.trip_history[-limit:]

    def _deenergize_channel(
        self,
        channel: TripChannel,
        force: bool = False
    ) -> tuple:
        """
        De-energize a channel (internal).

        In production, this would interface with actual I/O.

        Args:
            channel: Channel to de-energize
            force: Force even if already de-energized

        Returns:
            Tuple of (success, actual_time_ms)
        """
        if channel.current_state == ChannelState.DE_ENERGIZED and not force:
            return True, 0.0

        # Simulate de-energization
        # In production, this would write to actual output
        start = time.time()

        # Simulate response time (use fraction of expected)
        simulated_time = channel.response_time_ms * 0.5 / 1000.0
        time.sleep(min(simulated_time, 0.1))  # Cap at 100ms for simulation

        actual_time = (time.time() - start) * 1000

        logger.debug(
            f"De-energized {channel.channel_id} in {actual_time:.1f}ms"
        )

        return True, actual_time

    def _verify_all_safe(self) -> bool:
        """Verify all channels are in safe state."""
        for channel in self.channels.values():
            actual_state = self.state_reader(channel.channel_id)
            if actual_state != ChannelState.DE_ENERGIZED:
                logger.warning(
                    f"Channel {channel.channel_id} not in safe state: {actual_state}"
                )
                return False
        return True

    def _default_state_reader(self, channel_id: str) -> ChannelState:
        """Default state reader (returns stored state)."""
        if channel_id in self.channels:
            return self.channels[channel_id].current_state
        return ChannelState.UNKNOWN

    def _calculate_provenance(self, result: TripResult) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{result.trip_id}|"
            f"{result.initiator.value}|"
            f"{len(result.channels_tripped)}|"
            f"{result.all_channels_safe}|"
            f"{result.completed_at.isoformat() if result.completed_at else ''}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
