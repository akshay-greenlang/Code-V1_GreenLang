# -*- coding: utf-8 -*-
"""
Redundant Sensor Voting Logic for GL-005 CombustionSense
========================================================

Implements safety-critical sensor voting schemes per IEC 61508.

Voting Architectures:
    - 1oo1 (1 out of 1): Single sensor, no redundancy
    - 1oo2 (1 out of 2): Trip if any sensor detects hazard
    - 2oo2 (2 out of 2): Trip only if both sensors agree
    - 1oo2D (1 out of 2 with diagnostics): Enhanced with self-diagnostics
    - 2oo3 (2 out of 3): Trip if 2 of 3 agree (outlier rejection)

Safety Integrity Level (SIL) Requirements:
    - SIL 1: 1oo1 or 1oo2
    - SIL 2: 1oo2 or 2oo3
    - SIL 3: 2oo3 with diagnostics
    - SIL 4: Redundant 2oo3 with diversity

Reference Standards:
    - IEC 61508: Functional Safety
    - IEC 61511: Safety Instrumented Systems for Process Industry
    - ISA-84.01: Safety Instrumented Functions

Author: GL-SafetyEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import statistics
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class VotingArchitecture(Enum):
    """Safety voting architectures per IEC 61508."""
    ONE_OO_ONE = "1oo1"      # Single sensor
    ONE_OO_TWO = "1oo2"      # Any one of two
    TWO_OO_TWO = "2oo2"      # Both must agree
    ONE_OO_TWO_D = "1oo2D"   # One of two with diagnostics
    TWO_OO_THREE = "2oo3"    # Two of three
    TWO_OO_FOUR = "2oo4"     # Two of four


class VoteResult(Enum):
    """Result of voting operation."""
    SAFE = "safe"               # Normal operation
    TRIP = "trip"               # Safety trip required
    INCONCLUSIVE = "inconclusive"  # Cannot determine
    DEGRADED = "degraded"       # Operating with reduced redundancy


class DiagnosticStatus(Enum):
    """Sensor diagnostic status."""
    OK = "ok"
    WARNING = "warning"
    FAULT = "fault"
    UNKNOWN = "unknown"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SensorInput:
    """Input from a single sensor for voting."""
    sensor_id: str
    value: float
    timestamp: datetime
    quality: str = "GOOD"
    diagnostic_status: DiagnosticStatus = DiagnosticStatus.OK


@dataclass
class VotingOutput:
    """Output from voting logic."""
    result: VoteResult
    voted_value: Optional[float]
    confidence: float           # 0.0 to 1.0
    participating_sensors: List[str]
    rejected_sensors: List[str]
    agreement_percent: float
    provenance_hash: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TripCondition:
    """Definition of a trip condition."""
    parameter: str
    high_trip: Optional[float] = None
    low_trip: Optional[float] = None
    rate_trip: Optional[float] = None  # Rate of change trip
    delay_seconds: float = 0.0


@dataclass
class VotingGroupConfig:
    """Configuration for a voting group."""
    group_id: str
    parameter: str
    architecture: VotingArchitecture
    sensor_ids: List[str]
    tolerance_percent: float = 10.0
    trip_conditions: Optional[TripCondition] = None


# =============================================================================
# VOTING LOGIC IMPLEMENTATION
# =============================================================================

class RedundantSensorVoter:
    """
    Implements redundant sensor voting for safety-critical applications.

    Features:
        - Multiple voting architectures
        - Automatic outlier rejection
        - Diagnostic integration
        - Degraded mode operation
        - Complete audit trail
    """

    def __init__(self):
        self.voting_groups: Dict[str, VotingGroupConfig] = {}
        self.sensor_diagnostics: Dict[str, DiagnosticStatus] = {}
        self.vote_history: List[VotingOutput] = []

    def configure_group(self, config: VotingGroupConfig) -> None:
        """
        Configure a voting group.

        Args:
            config: Voting group configuration
        """
        # Validate configuration
        min_sensors = {
            VotingArchitecture.ONE_OO_ONE: 1,
            VotingArchitecture.ONE_OO_TWO: 2,
            VotingArchitecture.TWO_OO_TWO: 2,
            VotingArchitecture.ONE_OO_TWO_D: 2,
            VotingArchitecture.TWO_OO_THREE: 3,
            VotingArchitecture.TWO_OO_FOUR: 4,
        }

        required = min_sensors.get(config.architecture, 1)
        if len(config.sensor_ids) < required:
            raise ValueError(
                f"Architecture {config.architecture.value} requires at least "
                f"{required} sensors, got {len(config.sensor_ids)}"
            )

        self.voting_groups[config.group_id] = config
        logger.info(f"Configured voting group {config.group_id} with {config.architecture.value}")

    def process_inputs(
        self,
        group_id: str,
        inputs: List[SensorInput]
    ) -> VotingOutput:
        """
        Process sensor inputs and perform voting.

        Args:
            group_id: ID of voting group
            inputs: List of sensor inputs

        Returns:
            VotingOutput with voted result
        """
        if group_id not in self.voting_groups:
            raise ValueError(f"Unknown voting group: {group_id}")

        config = self.voting_groups[group_id]

        # Filter inputs to only those in the group
        valid_inputs = [i for i in inputs if i.sensor_id in config.sensor_ids]

        # Update diagnostics
        for inp in valid_inputs:
            self.sensor_diagnostics[inp.sensor_id] = inp.diagnostic_status

        # Execute voting based on architecture
        if config.architecture == VotingArchitecture.ONE_OO_ONE:
            output = self._vote_1oo1(config, valid_inputs)
        elif config.architecture == VotingArchitecture.ONE_OO_TWO:
            output = self._vote_1oo2(config, valid_inputs)
        elif config.architecture == VotingArchitecture.TWO_OO_TWO:
            output = self._vote_2oo2(config, valid_inputs)
        elif config.architecture == VotingArchitecture.ONE_OO_TWO_D:
            output = self._vote_1oo2d(config, valid_inputs)
        elif config.architecture == VotingArchitecture.TWO_OO_THREE:
            output = self._vote_2oo3(config, valid_inputs)
        elif config.architecture == VotingArchitecture.TWO_OO_FOUR:
            output = self._vote_2oo4(config, valid_inputs)
        else:
            output = self._vote_generic(config, valid_inputs)

        # Check trip conditions
        if config.trip_conditions and output.voted_value is not None:
            output = self._check_trip_conditions(output, config.trip_conditions)

        # Store in history
        self.vote_history.append(output)
        if len(self.vote_history) > 1000:
            self.vote_history = self.vote_history[-1000:]

        return output

    def _vote_1oo1(
        self,
        config: VotingGroupConfig,
        inputs: List[SensorInput]
    ) -> VotingOutput:
        """Single sensor voting (no redundancy)."""
        if not inputs:
            return self._create_inconclusive_output(config, inputs, "No inputs")

        inp = inputs[0]

        if inp.quality != "GOOD":
            return VotingOutput(
                result=VoteResult.DEGRADED,
                voted_value=inp.value,
                confidence=0.5,
                participating_sensors=[inp.sensor_id],
                rejected_sensors=[],
                agreement_percent=100.0,
                provenance_hash=self._calculate_hash(inputs),
            )

        return VotingOutput(
            result=VoteResult.SAFE,
            voted_value=inp.value,
            confidence=0.8,  # Lower confidence for single sensor
            participating_sensors=[inp.sensor_id],
            rejected_sensors=[],
            agreement_percent=100.0,
            provenance_hash=self._calculate_hash(inputs),
        )

    def _vote_1oo2(
        self,
        config: VotingGroupConfig,
        inputs: List[SensorInput]
    ) -> VotingOutput:
        """
        1 out of 2 voting - Trip if ANY sensor detects hazard.

        This is more conservative (safer) - trips on single sensor.
        """
        good_inputs = [i for i in inputs if i.quality == "GOOD"]

        if len(good_inputs) < 1:
            return self._create_inconclusive_output(config, inputs, "No good inputs")

        values = [i.value for i in good_inputs]

        # Use maximum value (most conservative for high trips)
        # or minimum value (most conservative for low trips)
        voted_value = max(values)  # Conservative for high limit

        return VotingOutput(
            result=VoteResult.SAFE,
            voted_value=voted_value,
            confidence=0.9,
            participating_sensors=[i.sensor_id for i in good_inputs],
            rejected_sensors=[i.sensor_id for i in inputs if i not in good_inputs],
            agreement_percent=self._calculate_agreement(values, config.tolerance_percent),
            provenance_hash=self._calculate_hash(inputs),
        )

    def _vote_2oo2(
        self,
        config: VotingGroupConfig,
        inputs: List[SensorInput]
    ) -> VotingOutput:
        """
        2 out of 2 voting - Both must agree for trip.

        Reduces spurious trips but requires both sensors working.
        """
        good_inputs = [i for i in inputs if i.quality == "GOOD"]

        if len(good_inputs) < 2:
            return VotingOutput(
                result=VoteResult.DEGRADED,
                voted_value=good_inputs[0].value if good_inputs else None,
                confidence=0.5,
                participating_sensors=[i.sensor_id for i in good_inputs],
                rejected_sensors=[i.sensor_id for i in inputs if i not in good_inputs],
                agreement_percent=0.0,
                provenance_hash=self._calculate_hash(inputs),
            )

        values = [i.value for i in good_inputs]
        agreement = self._calculate_agreement(values, config.tolerance_percent)

        if agreement >= 90.0:
            return VotingOutput(
                result=VoteResult.SAFE,
                voted_value=statistics.mean(values),
                confidence=1.0,
                participating_sensors=[i.sensor_id for i in good_inputs],
                rejected_sensors=[],
                agreement_percent=agreement,
                provenance_hash=self._calculate_hash(inputs),
            )
        else:
            # Disagreement - cannot determine
            return VotingOutput(
                result=VoteResult.INCONCLUSIVE,
                voted_value=None,
                confidence=0.3,
                participating_sensors=[i.sensor_id for i in good_inputs],
                rejected_sensors=[],
                agreement_percent=agreement,
                provenance_hash=self._calculate_hash(inputs),
            )

    def _vote_1oo2d(
        self,
        config: VotingGroupConfig,
        inputs: List[SensorInput]
    ) -> VotingOutput:
        """
        1 out of 2 with diagnostics - Uses diagnostic info to select best.
        """
        good_inputs = [i for i in inputs
                      if i.quality == "GOOD" and i.diagnostic_status == DiagnosticStatus.OK]

        if not good_inputs:
            # Fall back to any available input
            good_inputs = [i for i in inputs if i.quality == "GOOD"]

        if not good_inputs:
            return self._create_inconclusive_output(config, inputs, "No healthy inputs")

        # If both healthy, average them
        if len(good_inputs) >= 2:
            values = [i.value for i in good_inputs]
            voted_value = statistics.mean(values)
            confidence = 1.0
        else:
            voted_value = good_inputs[0].value
            confidence = 0.7

        return VotingOutput(
            result=VoteResult.SAFE,
            voted_value=voted_value,
            confidence=confidence,
            participating_sensors=[i.sensor_id for i in good_inputs],
            rejected_sensors=[i.sensor_id for i in inputs if i not in good_inputs],
            agreement_percent=self._calculate_agreement(
                [i.value for i in good_inputs], config.tolerance_percent
            ) if len(good_inputs) > 1 else 100.0,
            provenance_hash=self._calculate_hash(inputs),
        )

    def _vote_2oo3(
        self,
        config: VotingGroupConfig,
        inputs: List[SensorInput]
    ) -> VotingOutput:
        """
        2 out of 3 voting - Use median, reject outlier.

        Most common architecture for SIL 2/3 applications.
        """
        good_inputs = [i for i in inputs if i.quality == "GOOD"]

        if len(good_inputs) < 2:
            return VotingOutput(
                result=VoteResult.DEGRADED,
                voted_value=good_inputs[0].value if good_inputs else None,
                confidence=0.5,
                participating_sensors=[i.sensor_id for i in good_inputs],
                rejected_sensors=[i.sensor_id for i in inputs if i not in good_inputs],
                agreement_percent=100.0 if len(good_inputs) == 1 else 0.0,
                provenance_hash=self._calculate_hash(inputs),
            )

        values = [i.value for i in good_inputs]

        # Use median to reject outlier
        voted_value = statistics.median(values)

        # Identify rejected sensor (furthest from median)
        rejected = []
        participating = []
        for inp in good_inputs:
            deviation = abs(inp.value - voted_value)
            tolerance = abs(voted_value * config.tolerance_percent / 100) if voted_value != 0 else 1.0
            if deviation > tolerance and len(good_inputs) == 3:
                rejected.append(inp.sensor_id)
            else:
                participating.append(inp.sensor_id)

        # If no sensors rejected, all participate
        if not rejected:
            participating = [i.sensor_id for i in good_inputs]

        agreement = self._calculate_agreement(values, config.tolerance_percent)

        return VotingOutput(
            result=VoteResult.SAFE if agreement > 50 else VoteResult.DEGRADED,
            voted_value=voted_value,
            confidence=min(agreement / 100, 1.0),
            participating_sensors=participating,
            rejected_sensors=rejected + [i.sensor_id for i in inputs if i not in good_inputs],
            agreement_percent=agreement,
            provenance_hash=self._calculate_hash(inputs),
        )

    def _vote_2oo4(
        self,
        config: VotingGroupConfig,
        inputs: List[SensorInput]
    ) -> VotingOutput:
        """2 out of 4 voting - Use middle two values."""
        good_inputs = [i for i in inputs if i.quality == "GOOD"]

        if len(good_inputs) < 2:
            return self._create_inconclusive_output(config, inputs, "Insufficient inputs")

        values = sorted([i.value for i in good_inputs])

        # Use middle two values (reject highest and lowest)
        if len(values) >= 4:
            middle_values = values[1:3]  # indices 1 and 2
        else:
            middle_values = values

        voted_value = statistics.mean(middle_values)

        return VotingOutput(
            result=VoteResult.SAFE,
            voted_value=voted_value,
            confidence=0.95,
            participating_sensors=[i.sensor_id for i in good_inputs],
            rejected_sensors=[i.sensor_id for i in inputs if i not in good_inputs],
            agreement_percent=self._calculate_agreement(
                [i.value for i in good_inputs], config.tolerance_percent
            ),
            provenance_hash=self._calculate_hash(inputs),
        )

    def _vote_generic(
        self,
        config: VotingGroupConfig,
        inputs: List[SensorInput]
    ) -> VotingOutput:
        """Generic voting fallback."""
        good_inputs = [i for i in inputs if i.quality == "GOOD"]

        if not good_inputs:
            return self._create_inconclusive_output(config, inputs, "No good inputs")

        values = [i.value for i in good_inputs]
        voted_value = statistics.mean(values)

        return VotingOutput(
            result=VoteResult.SAFE,
            voted_value=voted_value,
            confidence=0.7,
            participating_sensors=[i.sensor_id for i in good_inputs],
            rejected_sensors=[i.sensor_id for i in inputs if i not in good_inputs],
            agreement_percent=self._calculate_agreement(values, config.tolerance_percent),
            provenance_hash=self._calculate_hash(inputs),
        )

    def _check_trip_conditions(
        self,
        output: VotingOutput,
        conditions: TripCondition
    ) -> VotingOutput:
        """Check if voted value triggers trip condition."""
        value = output.voted_value

        if value is None:
            return output

        trip = False

        if conditions.high_trip is not None and value >= conditions.high_trip:
            trip = True
            logger.warning(f"High trip triggered: {value} >= {conditions.high_trip}")

        if conditions.low_trip is not None and value <= conditions.low_trip:
            trip = True
            logger.warning(f"Low trip triggered: {value} <= {conditions.low_trip}")

        if trip:
            output.result = VoteResult.TRIP

        return output

    def _calculate_agreement(self, values: List[float], tolerance: float) -> float:
        """Calculate percentage agreement between sensors."""
        if len(values) < 2:
            return 100.0

        mean = statistics.mean(values)
        if mean == 0:
            return 100.0 if all(v == 0 for v in values) else 0.0

        max_deviation = max(abs(v - mean) / abs(mean) * 100 for v in values)
        agreement = max(0, 100 - (max_deviation / tolerance * 100))

        return min(100.0, agreement)

    def _create_inconclusive_output(
        self,
        config: VotingGroupConfig,
        inputs: List[SensorInput],
        reason: str
    ) -> VotingOutput:
        """Create an inconclusive voting output."""
        return VotingOutput(
            result=VoteResult.INCONCLUSIVE,
            voted_value=None,
            confidence=0.0,
            participating_sensors=[],
            rejected_sensors=[i.sensor_id for i in inputs],
            agreement_percent=0.0,
            provenance_hash=self._calculate_hash(inputs),
        )

    def _calculate_hash(self, inputs: List[SensorInput]) -> str:
        """Calculate provenance hash."""
        data = "|".join(f"{i.sensor_id}:{i.value}:{i.timestamp.isoformat()}" for i in inputs)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get_group_status(self, group_id: str) -> Dict[str, Any]:
        """Get current status of voting group."""
        if group_id not in self.voting_groups:
            return {"error": f"Unknown group: {group_id}"}

        config = self.voting_groups[group_id]

        sensor_statuses = {}
        for sid in config.sensor_ids:
            sensor_statuses[sid] = {
                "diagnostic": self.sensor_diagnostics.get(sid, DiagnosticStatus.UNKNOWN).value,
            }

        return {
            "group_id": group_id,
            "architecture": config.architecture.value,
            "sensors": sensor_statuses,
            "recent_votes": len([v for v in self.vote_history[-10:]
                                if v.participating_sensors and
                                any(s in config.sensor_ids for s in v.participating_sensors)]),
        }


if __name__ == "__main__":
    # Example usage
    voter = RedundantSensorVoter()

    # Configure O2 monitoring with 2oo3 voting
    o2_config = VotingGroupConfig(
        group_id="O2_SAFETY",
        parameter="O2",
        architecture=VotingArchitecture.TWO_OO_THREE,
        sensor_ids=["O2-A", "O2-B", "O2-C"],
        tolerance_percent=5.0,
        trip_conditions=TripCondition(
            parameter="O2",
            low_trip=1.0,   # Trip if O2 < 1%
            high_trip=15.0, # Trip if O2 > 15%
        ),
    )

    voter.configure_group(o2_config)

    # Process inputs
    inputs = [
        SensorInput("O2-A", 3.5, datetime.now()),
        SensorInput("O2-B", 3.6, datetime.now()),
        SensorInput("O2-C", 3.4, datetime.now()),
    ]

    result = voter.process_inputs("O2_SAFETY", inputs)
    print(f"Vote result: {result.result.value}, Value: {result.voted_value}, Confidence: {result.confidence}")
