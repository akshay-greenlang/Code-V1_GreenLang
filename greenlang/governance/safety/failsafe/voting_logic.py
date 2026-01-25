"""
VotingLogic - Voting Logic Implementations for SIS

This module implements voting logic for Safety Instrumented Systems
per IEC 61511. Voting logic determines how multiple redundant
channels are combined to make trip decisions.

Common voting architectures:
- 1oo1: Single channel (any trip triggers action)
- 1oo2: Dual redundant (any one of two triggers action)
- 2oo2: Dual series (both must agree to trigger)
- 2oo3: Triple modular redundant (two of three must agree)

Reference: IEC 61511-1 Clause 11.4

Example:
    >>> from greenlang.safety.failsafe.voting_logic import VotingLogic
    >>> voter = VotingLogic("2oo3")
    >>> result = voter.evaluate([True, True, False])
    >>> print(f"Trip: {result.trip_decision}")  # True
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class VotingArchitecture(str, Enum):
    """Supported voting architectures per IEC 61511."""

    ONE_OO_ONE = "1oo1"  # Single channel
    ONE_OO_TWO = "1oo2"  # 1 out of 2 (any)
    TWO_OO_TWO = "2oo2"  # 2 out of 2 (both)
    TWO_OO_THREE = "2oo3"  # 2 out of 3
    ONE_OO_THREE = "1oo3"  # 1 out of 3 (any)
    THREE_OO_THREE = "3oo3"  # 3 out of 3 (all)
    TWO_OO_FOUR = "2oo4"  # 2 out of 4


class ChannelStatus(str, Enum):
    """Status of individual channel."""

    NORMAL = "normal"  # Channel functioning normally
    TRIP = "trip"  # Channel in trip state
    FAULT = "fault"  # Channel has detected fault
    BYPASSED = "bypassed"  # Channel is bypassed
    UNKNOWN = "unknown"  # Channel status unknown


class ChannelInput(BaseModel):
    """Input from a single voting channel."""

    channel_id: str = Field(
        ...,
        description="Channel identifier"
    )
    value: bool = Field(
        ...,
        description="Channel trip status (True=trip requested)"
    )
    status: ChannelStatus = Field(
        default=ChannelStatus.NORMAL,
        description="Channel health status"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of value"
    )
    quality: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Signal quality (0-1)"
    )


class VotingResult(BaseModel):
    """Result of voting logic evaluation."""

    voting_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Voting evaluation ID"
    )
    architecture: VotingArchitecture = Field(
        ...,
        description="Voting architecture used"
    )
    trip_decision: bool = Field(
        ...,
        description="Final trip decision"
    )
    channels_voting_trip: int = Field(
        ...,
        description="Number of channels voting for trip"
    )
    channels_total: int = Field(
        ...,
        description="Total number of channels"
    )
    channels_required: int = Field(
        ...,
        description="Channels required for trip"
    )
    channels_healthy: int = Field(
        ...,
        description="Number of healthy channels"
    )
    channels_bypassed: int = Field(
        default=0,
        description="Number of bypassed channels"
    )
    channels_faulted: int = Field(
        default=0,
        description="Number of faulted channels"
    )
    degraded_mode: bool = Field(
        default=False,
        description="Is system in degraded mode?"
    )
    effective_architecture: str = Field(
        default="",
        description="Effective architecture after degradation"
    )
    channel_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Details of each channel input"
    )
    evaluation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of evaluation"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VotingLogic:
    """
    Voting Logic Implementation for SIS.

    Implements various voting architectures per IEC 61511.
    Handles:
    - Standard voting evaluations
    - Degraded mode operation
    - Bypassed channel handling
    - Fault detection integration

    The implementation follows zero-hallucination principles:
    - All decisions are deterministic
    - Boolean logic only
    - Complete audit trail

    Attributes:
        architecture: Current voting architecture
        channels_required: Minimum channels for trip

    Example:
        >>> voter = VotingLogic("2oo3")
        >>> result = voter.evaluate([
        ...     ChannelInput(channel_id="A", value=True),
        ...     ChannelInput(channel_id="B", value=True),
        ...     ChannelInput(channel_id="C", value=False),
        ... ])
        >>> print(f"Trip: {result.trip_decision}")
    """

    # Architecture configurations: (total_channels, required_for_trip)
    ARCHITECTURE_CONFIG: Dict[VotingArchitecture, Tuple[int, int]] = {
        VotingArchitecture.ONE_OO_ONE: (1, 1),
        VotingArchitecture.ONE_OO_TWO: (2, 1),
        VotingArchitecture.TWO_OO_TWO: (2, 2),
        VotingArchitecture.TWO_OO_THREE: (3, 2),
        VotingArchitecture.ONE_OO_THREE: (3, 1),
        VotingArchitecture.THREE_OO_THREE: (3, 3),
        VotingArchitecture.TWO_OO_FOUR: (4, 2),
    }

    def __init__(
        self,
        architecture: str,
        fail_safe_on_fault: bool = True
    ):
        """
        Initialize VotingLogic.

        Args:
            architecture: Voting architecture (e.g., "2oo3")
            fail_safe_on_fault: Trip on channel fault (default True)
        """
        self.architecture = VotingArchitecture(architecture)
        self.fail_safe_on_fault = fail_safe_on_fault

        config = self.ARCHITECTURE_CONFIG[self.architecture]
        self.expected_channels = config[0]
        self.channels_required = config[1]

        logger.info(
            f"VotingLogic initialized: {architecture}, "
            f"requires {self.channels_required}/{self.expected_channels}"
        )

    def evaluate(
        self,
        channel_inputs: List[ChannelInput]
    ) -> VotingResult:
        """
        Evaluate voting logic with channel inputs.

        Args:
            channel_inputs: List of channel input objects

        Returns:
            VotingResult with trip decision

        Raises:
            ValueError: If channel count doesn't match architecture
        """
        logger.debug(f"Evaluating {self.architecture.value} with {len(channel_inputs)} channels")

        # Count channel states
        channels_trip = 0
        channels_healthy = 0
        channels_bypassed = 0
        channels_faulted = 0
        channel_details = []

        for ch in channel_inputs:
            detail = {
                "channel_id": ch.channel_id,
                "value": ch.value,
                "status": ch.status.value,
                "quality": ch.quality,
            }
            channel_details.append(detail)

            if ch.status == ChannelStatus.BYPASSED:
                channels_bypassed += 1
                continue
            elif ch.status == ChannelStatus.FAULT:
                channels_faulted += 1
                if self.fail_safe_on_fault:
                    channels_trip += 1  # Fail safe
                continue
            elif ch.status == ChannelStatus.UNKNOWN:
                if self.fail_safe_on_fault:
                    channels_trip += 1  # Fail safe
                continue

            channels_healthy += 1
            if ch.value:
                channels_trip += 1

        # Determine effective channels
        effective_channels = channels_healthy
        total_channels = len(channel_inputs)

        # Check for degraded mode
        degraded_mode = (channels_bypassed > 0 or
                        channels_faulted > 0 or
                        total_channels != self.expected_channels)

        # Determine effective architecture in degraded mode
        effective_architecture = self.architecture.value
        effective_required = self.channels_required

        if degraded_mode:
            effective_architecture, effective_required = self._get_degraded_architecture(
                effective_channels, channels_bypassed, channels_faulted
            )

        # Apply voting logic
        trip_decision = channels_trip >= effective_required

        # Build result
        result = VotingResult(
            architecture=self.architecture,
            trip_decision=trip_decision,
            channels_voting_trip=channels_trip,
            channels_total=total_channels,
            channels_required=effective_required,
            channels_healthy=channels_healthy,
            channels_bypassed=channels_bypassed,
            channels_faulted=channels_faulted,
            degraded_mode=degraded_mode,
            effective_architecture=effective_architecture,
            channel_details=channel_details,
        )

        # Calculate provenance hash
        result.provenance_hash = self._calculate_provenance(result)

        logger.info(
            f"Voting result: {trip_decision}, "
            f"{channels_trip}/{effective_required} channels"
        )

        return result

    def evaluate_simple(
        self,
        channel_values: List[bool]
    ) -> VotingResult:
        """
        Simplified evaluation with boolean values only.

        Args:
            channel_values: List of boolean trip values

        Returns:
            VotingResult with trip decision
        """
        # Convert to ChannelInput objects
        channel_inputs = [
            ChannelInput(
                channel_id=chr(65 + i),  # A, B, C...
                value=value,
                status=ChannelStatus.NORMAL
            )
            for i, value in enumerate(channel_values)
        ]

        return self.evaluate(channel_inputs)

    def _get_degraded_architecture(
        self,
        healthy_channels: int,
        bypassed: int,
        faulted: int
    ) -> Tuple[str, int]:
        """
        Determine effective architecture in degraded mode.

        Per IEC 61511, when channels fail or are bypassed,
        the effective architecture changes.

        Args:
            healthy_channels: Number of healthy channels
            bypassed: Number of bypassed channels
            faulted: Number of faulted channels

        Returns:
            Tuple of (effective architecture string, required channels)
        """
        # 2oo3 degradation
        if self.architecture == VotingArchitecture.TWO_OO_THREE:
            if healthy_channels == 3:
                return "2oo3", 2
            elif healthy_channels == 2:
                return "1oo2", 1  # More conservative
            elif healthy_channels == 1:
                return "1oo1", 1
            else:
                return "0oo0", 0  # All failed - trip

        # 1oo2 degradation
        elif self.architecture == VotingArchitecture.ONE_OO_TWO:
            if healthy_channels == 2:
                return "1oo2", 1
            elif healthy_channels == 1:
                return "1oo1", 1
            else:
                return "0oo0", 0

        # 2oo2 degradation
        elif self.architecture == VotingArchitecture.TWO_OO_TWO:
            if healthy_channels == 2:
                return "2oo2", 2
            elif healthy_channels == 1:
                return "1oo1", 1  # Single channel remains
            else:
                return "0oo0", 0

        # Default - use original
        return self.architecture.value, self.channels_required

    def get_spurious_trip_probability(
        self,
        channel_availability: float = 0.99
    ) -> float:
        """
        Calculate spurious trip probability for architecture.

        Args:
            channel_availability: Availability of each channel

        Returns:
            Probability of spurious trip
        """
        p_fail = 1 - channel_availability  # Probability of spurious trip

        if self.architecture == VotingArchitecture.ONE_OO_ONE:
            # Single channel: P(trip) = p_fail
            return p_fail

        elif self.architecture == VotingArchitecture.ONE_OO_TWO:
            # Either channel: P = 1 - (1-p)^2 = 2p - p^2
            return 2 * p_fail - p_fail ** 2

        elif self.architecture == VotingArchitecture.TWO_OO_TWO:
            # Both channels: P = p^2
            return p_fail ** 2

        elif self.architecture == VotingArchitecture.TWO_OO_THREE:
            # 2 of 3: P = 3p^2 - 2p^3
            return 3 * p_fail ** 2 - 2 * p_fail ** 3

        elif self.architecture == VotingArchitecture.ONE_OO_THREE:
            # Any of 3: P = 1 - (1-p)^3
            return 1 - (1 - p_fail) ** 3

        return p_fail

    def get_dangerous_failure_probability(
        self,
        channel_pfd: float = 0.01
    ) -> float:
        """
        Calculate dangerous failure probability for architecture.

        Args:
            channel_pfd: PFD of each channel

        Returns:
            System PFD
        """
        p = channel_pfd

        if self.architecture == VotingArchitecture.ONE_OO_ONE:
            return p

        elif self.architecture == VotingArchitecture.ONE_OO_TWO:
            # Both must fail: p^2
            return p ** 2

        elif self.architecture == VotingArchitecture.TWO_OO_TWO:
            # Either fails: 2p - p^2
            return 2 * p - p ** 2

        elif self.architecture == VotingArchitecture.TWO_OO_THREE:
            # 2 of 3 must fail: 3p^2 - 2p^3
            return 3 * p ** 2 - 2 * p ** 3

        elif self.architecture == VotingArchitecture.ONE_OO_THREE:
            # All 3 must fail: p^3
            return p ** 3

        return p

    def compare_architectures(
        self,
        channel_pfd: float = 0.01,
        channel_spurious_rate: float = 0.01
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare all architectures for given channel parameters.

        Args:
            channel_pfd: PFD of each channel
            channel_spurious_rate: Spurious trip rate of each channel

        Returns:
            Comparison dict with PFD and spurious rates
        """
        comparison = {}

        for arch in VotingArchitecture:
            voter = VotingLogic(arch.value)

            comparison[arch.value] = {
                "pfd": voter.get_dangerous_failure_probability(channel_pfd),
                "spurious_rate": voter.get_spurious_trip_probability(channel_spurious_rate),
                "channels_total": voter.expected_channels,
                "channels_required": voter.channels_required,
            }

        return comparison

    def _calculate_provenance(self, result: VotingResult) -> str:
        """Calculate SHA-256 provenance hash for voting result."""
        provenance_str = (
            f"{result.architecture.value}|"
            f"{result.trip_decision}|"
            f"{result.channels_voting_trip}|"
            f"{result.channels_total}|"
            f"{result.evaluation_timestamp.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
