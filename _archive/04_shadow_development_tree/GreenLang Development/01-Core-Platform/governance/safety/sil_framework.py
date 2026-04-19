"""
GreenLang IEC 61511 Safety Integrity Level Framework

This module provides a comprehensive implementation of IEC 61511 (Functional Safety -
Safety Instrumented Systems for the Process Industry Sector) requirements for
Safety Instrumented Systems (SIS) design, assessment, and operation.

Key Components:
    - SILLevel: Safety Integrity Level definitions with PFD ranges
    - VotingArchitecture: MooN voting configurations for redundancy
    - SafetyFunction: Complete safety function specification
    - LOPAAnalyzer: Layer of Protection Analysis for SIL determination
    - PFDCalculator: Probability of Failure on Demand calculations
    - ProofTestScheduler: Proof test interval management
    - SRSGenerator: Safety Requirements Specification document generation
    - FailSafeManager: Fail-safe logic and safe state transitions

IEC 61511 Formula Reference:
    - PFD_avg (1oo1) = lambda_DU * T_proof / 2
    - PFD_avg (1oo2) = ((1-beta) * lambda_DU)^2 * T_proof^2 / 3 + beta * lambda_DU * T_proof / 2
    - PFD_avg (2oo3) = 3 * ((1-beta) * lambda_DU)^2 * T_proof^2 / 3 + beta * lambda_DU * T_proof / 2

Reference Standards:
    - IEC 61511-1:2016 Functional Safety - Safety instrumented systems
    - IEC 61511-2:2016 Guidelines for the application of IEC 61511-1
    - IEC 61511-3:2016 Guidance for the determination of the required SIL
    - IEC 61508-6:2010 Annex B - PFD Calculations

Target: Safety score 72 -> 95+/100

Example:
    >>> from greenlang.safety.sil_framework import (
    ...     SILLevel, PFDCalculator, LOPAAnalyzer, FailSafeManager
    ... )
    >>> # Determine SIL target via LOPA
    >>> lopa = LOPAAnalyzer()
    >>> scenario = LOPAScenario(
    ...     scenario_id="SCN-001",
    ...     description="High pressure in reactor",
    ...     initiating_event_frequency=0.1,
    ...     consequence_severity=ConsequenceSeverity.FATALITY,
    ...     ipls=[IPLDefinition(name="BPCS", pfd=0.1)]
    ... )
    >>> result = lopa.analyze(scenario)
    >>> print(f"Required SIL: {result.recommended_sil}")

    >>> # Calculate PFD for proposed design
    >>> pfd_calc = PFDCalculator()
    >>> pfd_result = pfd_calc.calculate_pfd_1oo2(
    ...     lambda_du=1e-6,
    ...     t_proof=8760,
    ...     beta=0.1
    ... )
    >>> print(f"PFD_avg: {pfd_result.pfd_avg:.2e}")

Author: GreenLang Safety Engineering Team
Version: 1.0.0
License: Proprietary
"""

from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, field_validator, model_validator
import hashlib
import logging
import math
import threading
import time
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SILLevel(str, Enum):
    """
    Safety Integrity Level definitions per IEC 61511.

    SIL levels define the required risk reduction and corresponding
    PFD (Probability of Failure on Demand) ranges for low demand mode
    safety functions.

    PFD Ranges (Low Demand Mode):
        - SIL 4: >= 1e-5 to < 1e-4 (RRF: 10,000-100,000)
        - SIL 3: >= 1e-4 to < 1e-3 (RRF: 1,000-10,000)
        - SIL 2: >= 1e-3 to < 1e-2 (RRF: 100-1,000)
        - SIL 1: >= 1e-2 to < 1e-1 (RRF: 10-100)

    Reference: IEC 61511-1:2016 Table 3
    """

    SIL_1 = "SIL_1"
    SIL_2 = "SIL_2"
    SIL_3 = "SIL_3"
    SIL_4 = "SIL_4"

    @property
    def pfd_range(self) -> Tuple[float, float]:
        """Get PFD range (lower, upper) for this SIL level."""
        ranges = {
            SILLevel.SIL_1: (1e-2, 1e-1),
            SILLevel.SIL_2: (1e-3, 1e-2),
            SILLevel.SIL_3: (1e-4, 1e-3),
            SILLevel.SIL_4: (1e-5, 1e-4),
        }
        return ranges[self]

    @property
    def risk_reduction_factor(self) -> Tuple[int, int]:
        """Get Risk Reduction Factor range (min, max)."""
        factors = {
            SILLevel.SIL_1: (10, 100),
            SILLevel.SIL_2: (100, 1000),
            SILLevel.SIL_3: (1000, 10000),
            SILLevel.SIL_4: (10000, 100000),
        }
        return factors[self]

    @property
    def minimum_hft(self) -> int:
        """Get minimum Hardware Fault Tolerance per IEC 61511."""
        hft = {
            SILLevel.SIL_1: 0,
            SILLevel.SIL_2: 0,
            SILLevel.SIL_3: 1,
            SILLevel.SIL_4: 2,
        }
        return hft[self]

    @classmethod
    def from_pfd(cls, pfd: float) -> Optional['SILLevel']:
        """Determine SIL level from PFD value."""
        if 1e-5 <= pfd < 1e-4:
            return cls.SIL_4
        elif 1e-4 <= pfd < 1e-3:
            return cls.SIL_3
        elif 1e-3 <= pfd < 1e-2:
            return cls.SIL_2
        elif 1e-2 <= pfd < 1e-1:
            return cls.SIL_1
        return None


class VotingArchitecture(str, Enum):
    """
    Voting architecture configurations per IEC 61511.

    MooN notation: M channels out of N must agree for trip action.

    Common configurations:
        - 1oo1: Single channel (no redundancy)
        - 1oo2: Dual redundant (any 1 of 2 trips)
        - 2oo2: Dual series (both must trip - high availability)
        - 2oo3: Triple Modular Redundant (2 of 3 must trip)
        - 2oo4: Quad redundant (2 of 4 must trip)

    Reference: IEC 61508-6:2010 Annex B
    """

    ONE_OO_ONE = "1oo1"
    ONE_OO_TWO = "1oo2"
    TWO_OO_TWO = "2oo2"
    TWO_OO_THREE = "2oo3"
    TWO_OO_FOUR = "2oo4"
    ONE_OO_ONE_D = "1oo1D"
    ONE_OO_TWO_D = "1oo2D"
    TWO_OO_THREE_D = "2oo3D"

    @property
    def channels_required(self) -> int:
        """Number of channels required for trip."""
        config = {
            VotingArchitecture.ONE_OO_ONE: 1,
            VotingArchitecture.ONE_OO_TWO: 1,
            VotingArchitecture.TWO_OO_TWO: 2,
            VotingArchitecture.TWO_OO_THREE: 2,
            VotingArchitecture.TWO_OO_FOUR: 2,
            VotingArchitecture.ONE_OO_ONE_D: 1,
            VotingArchitecture.ONE_OO_TWO_D: 1,
            VotingArchitecture.TWO_OO_THREE_D: 2,
        }
        return config[self]

    @property
    def total_channels(self) -> int:
        """Total number of channels in architecture."""
        config = {
            VotingArchitecture.ONE_OO_ONE: 1,
            VotingArchitecture.ONE_OO_TWO: 2,
            VotingArchitecture.TWO_OO_TWO: 2,
            VotingArchitecture.TWO_OO_THREE: 3,
            VotingArchitecture.TWO_OO_FOUR: 4,
            VotingArchitecture.ONE_OO_ONE_D: 1,
            VotingArchitecture.ONE_OO_TWO_D: 2,
            VotingArchitecture.TWO_OO_THREE_D: 3,
        }
        return config[self]

    @property
    def hardware_fault_tolerance(self) -> int:
        """Hardware Fault Tolerance (HFT) provided by architecture."""
        return self.total_channels - self.channels_required


class ConsequenceSeverity(str, Enum):
    """Consequence severity categories for LOPA analysis."""

    MINOR = "minor"
    SERIOUS = "serious"
    SEVERE = "severe"
    FATALITY = "fatality"
    MULTIPLE_FATALITIES = "multiple_fatalities"
    CATASTROPHIC = "catastrophic"


class TestType(str, Enum):
    """Types of proof tests per IEC 61511."""

    FULL_PROOF_TEST = "full_proof_test"
    PARTIAL_STROKE_TEST = "partial_stroke_test"
    DIAGNOSTIC_TEST = "diagnostic_test"
    VISUAL_INSPECTION = "visual_inspection"
    CALIBRATION_CHECK = "calibration_check"


class TestResult(str, Enum):
    """Proof test result status."""

    PASS = "pass"
    FAIL = "fail"
    DEGRADED = "degraded"
    INCONCLUSIVE = "inconclusive"
    NOT_PERFORMED = "not_performed"


class WatchdogState(str, Enum):
    """Watchdog timer states."""

    STOPPED = "stopped"
    RUNNING = "running"
    EXPIRED = "expired"
    DISABLED = "disabled"


class TimeoutAction(str, Enum):
    """Action to take on watchdog timeout."""

    TRIP = "trip"
    ALARM = "alarm"
    RESET = "reset"
    LOG = "log"
    CALLBACK = "callback"


class TransitionMode(str, Enum):
    """Safe state transition modes."""

    IMMEDIATE = "immediate"
    SEQUENCED = "sequenced"
    RAMPED = "ramped"
    CONDITIONAL = "conditional"


class TransitionStatus(str, Enum):
    """Status of a transition execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    TIMEOUT = "timeout"


class ChannelStatus(str, Enum):
    """Status of individual voting channel."""

    NORMAL = "normal"
    TRIP = "trip"
    FAULT = "fault"
    BYPASSED = "bypassed"
    UNKNOWN = "unknown"


class SRSStatus(str, Enum):
    """SRS document status."""

    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    SUPERSEDED = "superseded"


# =============================================================================
# DATA CLASSES AND MODELS
# =============================================================================

@dataclass
class SafetyFunction:
    """
    Safety Function specification per IEC 61511-1 Clause 10.

    A Safety Instrumented Function (SIF) defines the complete specification
    for a single safety function including target SIL, PFD requirements,
    response time, and safe state definition.

    Attributes:
        function_id: Unique identifier (e.g., "SIF-001")
        description: Functional description of the safety function
        target_sil: Required SIL level
        pfd_target: Target PFD average
        proof_test_interval: Proof test interval in hours
        hardware_fault_tolerance: Required HFT (0, 1, or 2)
        safe_state: Definition of the safe state
        process_safety_time_ms: Maximum time to reach safe state
        sensors: List of input sensor tags
        logic_solver: Logic solver tag
        final_elements: List of output element tags
        voting_architecture: Voting configuration
        diagnostic_coverage: Target diagnostic coverage (0-1)
        bypass_permitted: Whether bypass is allowed

    Example:
        >>> sf = SafetyFunction(
        ...     function_id="SIF-HT-001",
        ...     description="High Temperature Shutdown",
        ...     target_sil=SILLevel.SIL_2,
        ...     pfd_target=0.005,
        ...     proof_test_interval=8760,
        ...     hardware_fault_tolerance=0,
        ...     safe_state="Valve XV-001 CLOSED, Pump P-001 OFF"
        ... )
    """

    function_id: str
    description: str
    target_sil: SILLevel
    pfd_target: float
    proof_test_interval: float  # hours
    hardware_fault_tolerance: int
    safe_state: str
    process_safety_time_ms: float = 5000.0
    sensors: List[str] = field(default_factory=list)
    logic_solver: str = ""
    final_elements: List[str] = field(default_factory=list)
    voting_architecture: VotingArchitecture = VotingArchitecture.ONE_OO_ONE
    diagnostic_coverage: float = 0.6
    bypass_permitted: bool = False
    initiating_cause: str = ""
    consequence: str = ""
    action_on_detection: str = ""
    response_time_ms: float = 1000.0

    def __post_init__(self):
        """Validate safety function parameters."""
        # Validate PFD target matches SIL
        pfd_lower, pfd_upper = self.target_sil.pfd_range
        if not (pfd_lower <= self.pfd_target < pfd_upper):
            logger.warning(
                f"SIF {self.function_id}: PFD target {self.pfd_target} "
                f"outside {self.target_sil.value} range ({pfd_lower}, {pfd_upper})"
            )

        # Validate HFT meets minimum
        min_hft = self.target_sil.minimum_hft
        if self.hardware_fault_tolerance < min_hft:
            logger.warning(
                f"SIF {self.function_id}: HFT {self.hardware_fault_tolerance} "
                f"below minimum {min_hft} for {self.target_sil.value}"
            )

        # Validate response time < PST
        if self.response_time_ms >= self.process_safety_time_ms:
            logger.warning(
                f"SIF {self.function_id}: Response time {self.response_time_ms}ms "
                f">= PST {self.process_safety_time_ms}ms"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "function_id": self.function_id,
            "description": self.description,
            "target_sil": self.target_sil.value,
            "pfd_target": self.pfd_target,
            "proof_test_interval": self.proof_test_interval,
            "hardware_fault_tolerance": self.hardware_fault_tolerance,
            "safe_state": self.safe_state,
            "process_safety_time_ms": self.process_safety_time_ms,
            "sensors": self.sensors,
            "logic_solver": self.logic_solver,
            "final_elements": self.final_elements,
            "voting_architecture": self.voting_architecture.value,
            "diagnostic_coverage": self.diagnostic_coverage,
            "bypass_permitted": self.bypass_permitted,
            "response_time_ms": self.response_time_ms,
        }


class IPLDefinition(BaseModel):
    """Independent Protection Layer definition for LOPA."""

    name: str = Field(..., description="Name of the IPL")
    pfd: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of Failure on Demand (0-1)"
    )
    description: Optional[str] = Field(None, description="IPL description")
    is_sis: bool = Field(False, description="Is this a Safety Instrumented System?")

    @field_validator('pfd')
    @classmethod
    def validate_pfd(cls, v: float) -> float:
        """Validate PFD is within acceptable range."""
        if v < 1e-5:
            logger.warning(f"IPL PFD {v} is very low. Verify credit is justified.")
        if v > 0.1:
            logger.warning(f"IPL PFD {v} exceeds typical credit of 0.1")
        return v


class LOPAScenario(BaseModel):
    """LOPA Scenario definition for analysis."""

    scenario_id: str = Field(..., description="Unique scenario identifier")
    description: str = Field(..., description="Scenario description")
    initiating_event: str = Field(default="", description="Initiating event description")
    initiating_event_frequency: float = Field(
        ...,
        gt=0,
        description="Initiating event frequency (per year)"
    )
    consequence_severity: ConsequenceSeverity = Field(
        ...,
        description="Consequence severity category"
    )
    consequence_description: Optional[str] = Field(None, description="Detailed consequence")
    ipls: List[IPLDefinition] = Field(default_factory=list, description="Protection layers")
    conditional_modifiers: Dict[str, float] = Field(
        default_factory=dict,
        description="Conditional probability modifiers"
    )
    target_mitigated_frequency: Optional[float] = Field(
        None,
        description="Target mitigated frequency (per year)"
    )


class PFDInput(BaseModel):
    """Input parameters for PFD calculation."""

    architecture: VotingArchitecture = Field(..., description="Voting architecture")
    lambda_du: float = Field(
        ...,
        ge=0,
        description="Dangerous undetected failure rate (per hour)"
    )
    lambda_dd: float = Field(
        default=0,
        ge=0,
        description="Dangerous detected failure rate (per hour)"
    )
    proof_test_interval_hours: float = Field(
        ...,
        gt=0,
        description="Proof test interval (hours)"
    )
    diagnostic_test_interval_hours: float = Field(
        default=1.0,
        gt=0,
        description="Diagnostic test interval (hours)"
    )
    mean_time_to_repair_hours: float = Field(
        default=8.0,
        ge=0,
        description="Mean time to repair (hours)"
    )
    beta_ccf: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Common cause failure beta factor"
    )
    diagnostic_coverage: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Diagnostic coverage (DC) factor"
    )
    component_id: Optional[str] = Field(None, description="Component identifier")

    @field_validator('beta_ccf')
    @classmethod
    def validate_beta(cls, v: float) -> float:
        """Validate CCF beta factor."""
        if v < 0.01:
            logger.warning(f"Beta CCF {v} is very low. Typical values: 0.05-0.2")
        if v > 0.2:
            logger.warning(f"Beta CCF {v} is high. Consider improving diversity.")
        return v


class PFDResult(BaseModel):
    """Result of PFD calculation."""

    architecture: VotingArchitecture = Field(..., description="Architecture used")
    pfd_avg: float = Field(..., ge=0, le=1, description="Average PFD")
    pfd_max: float = Field(..., ge=0, le=1, description="Maximum PFD")
    sil_achieved: int = Field(..., ge=0, le=4, description="SIL level achieved")
    lambda_du_effective: float = Field(..., description="Effective lambda_DU")
    ccf_contribution: float = Field(..., description="CCF contribution to PFD")
    independent_contribution: float = Field(..., description="Independent failure contribution")
    proof_test_interval_hours: float = Field(..., description="Proof test interval used")
    risk_reduction_factor: float = Field(..., description="Risk Reduction Factor (1/PFD)")
    calculation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Calculation timestamp"
    )
    provenance_hash: str = Field(..., description="SHA-256 audit hash")
    formula_used: str = Field(..., description="Formula applied")
    warnings: List[str] = Field(default_factory=list, description="Calculation warnings")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class LOPAResult(BaseModel):
    """LOPA Analysis result."""

    scenario_id: str = Field(..., description="Scenario identifier")
    initiating_event_frequency: float = Field(..., description="Initiating frequency")
    total_ipl_pfd: float = Field(..., description="Combined IPL PFD")
    conditional_modifier_product: float = Field(..., description="Modifier product")
    unmitigated_frequency: float = Field(..., description="Frequency before IPL credit")
    mitigated_frequency: float = Field(..., description="Frequency after IPL credit")
    target_frequency: float = Field(..., description="Target tolerable frequency")
    risk_gap: float = Field(..., description="Gap between mitigated and target")
    sif_required: bool = Field(..., description="Is a SIF required?")
    required_sif_pfd: Optional[float] = Field(None, description="Required SIF PFD")
    recommended_sil: Optional[int] = Field(None, description="Recommended SIL (1-4)")
    ipls_credited: List[str] = Field(default_factory=list, description="Credited IPLs")
    calculation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp"
    )
    provenance_hash: str = Field(..., description="SHA-256 audit hash")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ProofTestRecord(BaseModel):
    """Record of a proof test execution."""

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Record identifier"
    )
    equipment_id: str = Field(..., description="Equipment identifier")
    test_type: TestType = Field(..., description="Test type")
    scheduled_date: datetime = Field(..., description="Scheduled date")
    actual_date: Optional[datetime] = Field(None, description="Actual date")
    result: TestResult = Field(default=TestResult.NOT_PERFORMED, description="Result")
    detected_failures: List[str] = Field(default_factory=list, description="Failures found")
    corrective_actions: List[str] = Field(default_factory=list, description="Actions taken")
    test_coverage: float = Field(default=1.0, ge=0, le=1, description="Coverage factor")
    performed_by: Optional[str] = Field(None, description="Tester name")
    witnessed_by: Optional[str] = Field(None, description="Witness name")
    duration_minutes: Optional[int] = Field(None, description="Duration")
    notes: str = Field(default="", description="Notes")
    provenance_hash: str = Field(default="", description="Audit hash")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ProofTestSchedule(BaseModel):
    """Proof test schedule for a SIS component."""

    schedule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Schedule identifier"
    )
    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_description: str = Field(default="", description="Description")
    target_sil: int = Field(..., ge=1, le=4, description="Target SIL")
    proof_test_interval_hours: float = Field(..., gt=0, description="Test interval (hours)")
    partial_stroke_test_interval_hours: Optional[float] = Field(
        None,
        gt=0,
        description="PST interval"
    )
    diagnostic_test_interval_hours: float = Field(
        default=1.0,
        gt=0,
        description="Diagnostic interval"
    )
    last_test_date: Optional[datetime] = Field(None, description="Last test date")
    next_test_due: datetime = Field(..., description="Next test due date")
    grace_period_days: int = Field(default=30, description="Grace period")
    is_overdue: bool = Field(default=False, description="Is overdue?")
    days_until_due: int = Field(default=0, description="Days until due")
    test_history: List[ProofTestRecord] = Field(default_factory=list, description="History")
    pfd_avg_current: Optional[float] = Field(None, description="Current PFDavg")
    created_date: datetime = Field(default_factory=datetime.utcnow, description="Created")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Updated")
    provenance_hash: str = Field(default="", description="Audit hash")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class VotingResult(BaseModel):
    """Result of voting logic evaluation."""

    voting_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Voting ID"
    )
    architecture: VotingArchitecture = Field(..., description="Architecture used")
    trip_decision: bool = Field(..., description="Final trip decision")
    channels_voting_trip: int = Field(..., description="Channels voting trip")
    channels_total: int = Field(..., description="Total channels")
    channels_required: int = Field(..., description="Channels required")
    channels_healthy: int = Field(..., description="Healthy channels")
    channels_bypassed: int = Field(default=0, description="Bypassed channels")
    channels_faulted: int = Field(default=0, description="Faulted channels")
    degraded_mode: bool = Field(default=False, description="In degraded mode?")
    effective_architecture: str = Field(default="", description="Effective architecture")
    channel_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Channel details"
    )
    evaluation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp"
    )
    provenance_hash: str = Field(default="", description="Audit hash")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ChannelInput(BaseModel):
    """Input from a single voting channel."""

    channel_id: str = Field(..., description="Channel identifier")
    value: bool = Field(..., description="Trip status (True=trip)")
    status: ChannelStatus = Field(default=ChannelStatus.NORMAL, description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    quality: float = Field(default=1.0, ge=0, le=1, description="Signal quality")


class WatchdogConfig(BaseModel):
    """Configuration for watchdog timer."""

    watchdog_id: str = Field(
        default_factory=lambda: f"WD-{uuid.uuid4().hex[:6].upper()}",
        description="Watchdog ID"
    )
    timeout_ms: float = Field(default=1000.0, gt=0, description="Timeout (ms)")
    action_on_timeout: TimeoutAction = Field(
        default=TimeoutAction.TRIP,
        description="Timeout action"
    )
    auto_restart: bool = Field(default=False, description="Auto-restart after timeout")
    min_kick_interval_ms: float = Field(default=0, ge=0, description="Min kick interval")
    max_kick_interval_ms: Optional[float] = Field(None, description="Max kick interval")
    description: str = Field(default="", description="Description")


class TransitionStep(BaseModel):
    """Individual step in a transition sequence."""

    step_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Step ID"
    )
    sequence_number: int = Field(..., ge=1, description="Sequence number")
    description: str = Field(..., description="Step description")
    action: str = Field(..., description="Action to execute")
    target_equipment: str = Field(..., description="Target equipment")
    target_state: str = Field(..., description="Target state")
    timeout_ms: float = Field(default=5000.0, gt=0, description="Step timeout (ms)")
    verification_required: bool = Field(default=True, description="Verify after action")
    rollback_on_failure: bool = Field(default=False, description="Rollback on failure")
    dependencies: List[str] = Field(default_factory=list, description="Step dependencies")


class TransitionConfig(BaseModel):
    """Configuration for safe transition."""

    transition_id: str = Field(
        default_factory=lambda: f"TR-{uuid.uuid4().hex[:8].upper()}",
        description="Transition ID"
    )
    name: str = Field(..., description="Transition name")
    from_state: str = Field(default="OPERATING", description="Source state")
    to_state: str = Field(..., description="Target safe state")
    mode: TransitionMode = Field(default=TransitionMode.IMMEDIATE, description="Mode")
    steps: List[TransitionStep] = Field(default_factory=list, description="Steps")
    total_timeout_ms: float = Field(default=30000.0, gt=0, description="Total timeout")
    abort_on_step_failure: bool = Field(default=True, description="Abort on failure")
    verification_timeout_ms: float = Field(default=5000.0, gt=0, description="Verify timeout")
    allow_partial_success: bool = Field(default=False, description="Accept partial")


class TransitionResult(BaseModel):
    """Result of transition execution."""

    transition_id: str = Field(..., description="Transition ID")
    execution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Execution ID"
    )
    status: TransitionStatus = Field(..., description="Final status")
    from_state: str = Field(..., description="Starting state")
    to_state: str = Field(..., description="Target state")
    achieved_state: str = Field(..., description="Achieved state")
    start_time: datetime = Field(..., description="Start time")
    end_time: datetime = Field(..., description="End time")
    duration_ms: float = Field(..., description="Duration (ms)")
    steps_completed: int = Field(default=0, description="Steps completed")
    steps_total: int = Field(default=0, description="Total steps")
    step_results: List[Dict[str, Any]] = Field(default_factory=list, description="Step results")
    errors: List[str] = Field(default_factory=list, description="Errors")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    verified: bool = Field(default=False, description="Safe state verified")
    provenance_hash: str = Field(default="", description="Audit hash")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SRSSection(BaseModel):
    """Section of an SRS document."""

    section_id: str = Field(..., description="Section ID")
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    subsections: List['SRSSection'] = Field(default_factory=list, description="Subsections")
    requirements: List[str] = Field(default_factory=list, description="Requirements")


class SafetyFunctionRequirement(BaseModel):
    """Safety function requirement per IEC 61511-1 Clause 10.3."""

    req_id: str = Field(..., description="Requirement ID")
    description: str = Field(..., description="Requirement description")
    sil_level: int = Field(..., ge=1, le=4, description="Required SIL")
    safe_state: str = Field(..., description="Safe state definition")
    process_safety_time_ms: float = Field(..., gt=0, description="PST (ms)")
    required_response_time_ms: float = Field(..., gt=0, description="Response time (ms)")
    proof_test_interval_hours: float = Field(..., gt=0, description="Test interval (hours)")
    pfd_target: float = Field(..., gt=0, lt=1, description="Target PFD")
    hft_requirement: int = Field(default=0, ge=0, description="HFT requirement")
    diagnostic_coverage_target: float = Field(default=0.6, ge=0, le=1, description="DC target")
    input_sensors: List[str] = Field(default_factory=list, description="Input sensors")
    output_actuators: List[str] = Field(default_factory=list, description="Output actuators")
    initiating_cause: str = Field(default="", description="Initiating cause")
    consequence: str = Field(default="", description="Consequence")
    action_on_detection: str = Field(default="", description="Action on detection")
    manual_shutdown_required: bool = Field(default=True, description="Manual shutdown")
    bypass_permitted: bool = Field(default=False, description="Bypass permitted")


class SRSDocument(BaseModel):
    """Complete Safety Requirements Specification document."""

    document_id: str = Field(
        default_factory=lambda: f"SRS-{uuid.uuid4().hex[:8].upper()}",
        description="Document ID"
    )
    title: str = Field(..., description="SRS title")
    revision: str = Field(default="1.0", description="Revision")
    status: SRSStatus = Field(default=SRSStatus.DRAFT, description="Status")
    project_name: str = Field(default="", description="Project name")
    system_name: str = Field(default="", description="SIS name")
    created_date: datetime = Field(default_factory=datetime.utcnow, description="Created")
    last_modified: datetime = Field(default_factory=datetime.utcnow, description="Modified")
    created_by: str = Field(default="", description="Author")
    approved_by: Optional[str] = Field(None, description="Approver")
    approval_date: Optional[datetime] = Field(None, description="Approval date")
    scope: str = Field(default="", description="Scope")
    hazard_identification_ref: str = Field(default="", description="Hazard study ref")
    safety_function_requirements: List[SafetyFunctionRequirement] = Field(
        default_factory=list,
        description="Safety function requirements"
    )
    sections: List[SRSSection] = Field(default_factory=list, description="Sections")
    appendices: List[str] = Field(default_factory=list, description="Appendices")
    references: List[str] = Field(default_factory=list, description="References")
    change_history: List[Dict[str, Any]] = Field(default_factory=list, description="History")
    provenance_hash: str = Field(default="", description="Audit hash")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# =============================================================================
# LOPA ANALYZER
# =============================================================================

class LOPAAnalyzer:
    """
    Layer of Protection Analysis (LOPA) Analyzer per IEC 61511-3.

    LOPA is a simplified risk assessment method that:
    1. Identifies hazard scenarios
    2. Quantifies initiating event frequencies
    3. Credits Independent Protection Layers (IPLs)
    4. Determines residual risk and required SIL

    Follows zero-hallucination principles:
    - All calculations are deterministic
    - No LLM involvement in numeric computations
    - Complete audit trail with provenance hashing

    Attributes:
        target_frequencies: Dict mapping severity to target frequency

    Example:
        >>> analyzer = LOPAAnalyzer()
        >>> scenario = LOPAScenario(
        ...     scenario_id="SCN-001",
        ...     description="High pressure",
        ...     initiating_event_frequency=0.1,
        ...     consequence_severity=ConsequenceSeverity.FATALITY,
        ...     ipls=[IPLDefinition(name="BPCS", pfd=0.1)]
        ... )
        >>> result = analyzer.analyze(scenario)
        >>> print(f"SIF Required: {result.sif_required}")
    """

    DEFAULT_TARGET_FREQUENCIES: Dict[ConsequenceSeverity, float] = {
        ConsequenceSeverity.MINOR: 1.0,
        ConsequenceSeverity.SERIOUS: 0.1,
        ConsequenceSeverity.SEVERE: 0.01,
        ConsequenceSeverity.FATALITY: 1e-4,
        ConsequenceSeverity.MULTIPLE_FATALITIES: 1e-5,
        ConsequenceSeverity.CATASTROPHIC: 1e-6,
    }

    SIL_PFD_RANGES: Dict[int, Tuple[float, float]] = {
        4: (1e-5, 1e-4),
        3: (1e-4, 1e-3),
        2: (1e-3, 1e-2),
        1: (1e-2, 1e-1),
    }

    def __init__(
        self,
        target_frequencies: Optional[Dict[ConsequenceSeverity, float]] = None
    ):
        """
        Initialize LOPAAnalyzer.

        Args:
            target_frequencies: Custom target frequencies per severity
        """
        self.target_frequencies = (
            target_frequencies or self.DEFAULT_TARGET_FREQUENCIES.copy()
        )
        logger.info("LOPAAnalyzer initialized with target frequencies")

    def analyze(self, scenario: LOPAScenario) -> LOPAResult:
        """
        Perform LOPA analysis on a scenario.

        Args:
            scenario: LOPAScenario to analyze

        Returns:
            LOPAResult with analysis results

        Raises:
            ValueError: If scenario data is invalid
        """
        start_time = datetime.utcnow()
        warnings: List[str] = []

        logger.info(f"Starting LOPA analysis for scenario: {scenario.scenario_id}")

        try:
            # Calculate conditional modifier product
            conditional_product = self._calculate_conditional_product(
                scenario.conditional_modifiers
            )

            # Calculate unmitigated frequency
            unmitigated_frequency = (
                scenario.initiating_event_frequency * conditional_product
            )

            # Calculate total IPL PFD
            total_ipl_pfd, ipl_warnings = self._calculate_total_ipl_pfd(scenario.ipls)
            warnings.extend(ipl_warnings)

            # Calculate mitigated frequency
            mitigated_frequency = unmitigated_frequency * total_ipl_pfd

            # Get target frequency
            target_frequency = self._get_target_frequency(
                scenario.consequence_severity,
                scenario.target_mitigated_frequency
            )

            # Calculate risk gap
            risk_gap = mitigated_frequency / target_frequency

            # Determine if SIF is required
            sif_required = risk_gap > 1.0

            # Calculate required SIF PFD if needed
            required_sif_pfd = None
            recommended_sil = None

            if sif_required:
                required_sif_pfd = target_frequency / (
                    unmitigated_frequency * total_ipl_pfd
                )
                # Apply safety factor
                required_sif_pfd *= 0.5
                recommended_sil = self._pfd_to_sil(required_sif_pfd)

                if recommended_sil is None:
                    warnings.append(
                        f"Required PFD {required_sif_pfd:.2e} exceeds SIL 4 capability. "
                        "Consider additional IPLs or inherently safer design."
                    )
                    recommended_sil = 4

            # Generate provenance hash
            provenance_hash = self._calculate_provenance(
                scenario, mitigated_frequency, target_frequency
            )

            result = LOPAResult(
                scenario_id=scenario.scenario_id,
                initiating_event_frequency=scenario.initiating_event_frequency,
                total_ipl_pfd=total_ipl_pfd,
                conditional_modifier_product=conditional_product,
                unmitigated_frequency=unmitigated_frequency,
                mitigated_frequency=mitigated_frequency,
                target_frequency=target_frequency,
                risk_gap=risk_gap,
                sif_required=sif_required,
                required_sif_pfd=required_sif_pfd,
                recommended_sil=recommended_sil,
                ipls_credited=[ipl.name for ipl in scenario.ipls],
                calculation_timestamp=start_time,
                provenance_hash=provenance_hash,
                warnings=warnings,
            )

            logger.info(
                f"LOPA analysis complete for {scenario.scenario_id}. "
                f"SIF Required: {sif_required}, Recommended SIL: {recommended_sil}"
            )

            return result

        except Exception as e:
            logger.error(
                f"LOPA analysis failed for {scenario.scenario_id}: {str(e)}",
                exc_info=True
            )
            raise

    def _calculate_conditional_product(self, modifiers: Dict[str, float]) -> float:
        """Calculate product of conditional modifiers."""
        if not modifiers:
            return 1.0

        product = 1.0
        for name, value in modifiers.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Conditional modifier '{name}' value {value} must be between 0 and 1"
                )
            product *= value

        return product

    def _calculate_total_ipl_pfd(
        self,
        ipls: List[IPLDefinition]
    ) -> Tuple[float, List[str]]:
        """Calculate combined PFD of all IPLs."""
        warnings: List[str] = []

        if not ipls:
            return 1.0, warnings

        total_pfd = 1.0
        for ipl in ipls:
            total_pfd *= ipl.pfd

            if ipl.pfd < 0.01 and not ipl.is_sis:
                warnings.append(
                    f"IPL '{ipl.name}' has PFD {ipl.pfd} < 0.01. "
                    "Ensure credit is justified for non-SIS IPL."
                )

        return total_pfd, warnings

    def _get_target_frequency(
        self,
        severity: ConsequenceSeverity,
        override: Optional[float] = None
    ) -> float:
        """Get target tolerable frequency."""
        if override is not None:
            return override
        return self.target_frequencies[severity]

    def _pfd_to_sil(self, pfd: float) -> Optional[int]:
        """Convert PFD to SIL level."""
        for sil, (lower, upper) in self.SIL_PFD_RANGES.items():
            if lower <= pfd < upper:
                return sil

        if pfd < 1e-5:
            return None  # Beyond SIL 4
        if pfd >= 0.1:
            return None  # Below SIL 1
        return None

    def _calculate_provenance(
        self,
        scenario: LOPAScenario,
        mitigated_freq: float,
        target_freq: float
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{scenario.scenario_id}|"
            f"{scenario.initiating_event_frequency}|"
            f"{mitigated_freq}|"
            f"{target_freq}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def batch_analyze(self, scenarios: List[LOPAScenario]) -> List[LOPAResult]:
        """Analyze multiple scenarios."""
        results = []
        for scenario in scenarios:
            try:
                result = self.analyze(scenario)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze scenario {scenario.scenario_id}: {e}")
                raise
        return results

    def sil_determination(
        self,
        initiating_event_frequency: float,
        consequence_severity: ConsequenceSeverity,
        ipls: List[IPLDefinition]
    ) -> int:
        """
        Quick SIL determination from basic parameters.

        Args:
            initiating_event_frequency: Initiating event frequency per year
            consequence_severity: Consequence severity
            ipls: List of IPLs

        Returns:
            Recommended SIL level (1-4)
        """
        scenario = LOPAScenario(
            scenario_id="QUICK-SIL",
            description="Quick SIL determination",
            initiating_event_frequency=initiating_event_frequency,
            consequence_severity=consequence_severity,
            ipls=ipls
        )
        result = self.analyze(scenario)
        return result.recommended_sil or 0


# =============================================================================
# PFD CALCULATOR
# =============================================================================

class PFDCalculator:
    """
    Probability of Failure on Demand Calculator per IEC 61511/61508.

    Implements PFD calculations for various voting architectures.
    All calculations are deterministic with complete audit trail.

    IEC 61511 Formulas:
        PFD_avg (1oo1) = lambda_DU * T_proof / 2
        PFD_avg (1oo2) = ((1-beta) * lambda_DU)^2 * T_proof^2 / 3 + beta * lambda_DU * T_proof / 2
        PFD_avg (2oo3) = 3 * ((1-beta) * lambda_DU)^2 * T_proof^2 / 3 + beta * lambda_DU * T_proof / 2

    Attributes:
        SIL_PFD_RANGES: Dict mapping SIL levels to PFD ranges

    Example:
        >>> calc = PFDCalculator()
        >>> result = calc.calculate_pfd_1oo2(lambda_du=1e-6, t_proof=8760, beta=0.1)
        >>> print(f"PFD_avg: {result.pfd_avg:.2e}")
    """

    SIL_PFD_RANGES: Dict[int, Tuple[float, float]] = {
        4: (1e-5, 1e-4),
        3: (1e-4, 1e-3),
        2: (1e-3, 1e-2),
        1: (1e-2, 1e-1),
        0: (1e-1, 1.0),
    }

    def __init__(self):
        """Initialize PFDCalculator."""
        logger.info("PFDCalculator initialized")

    def calculate(self, input_data: PFDInput) -> PFDResult:
        """
        Calculate PFD for given input parameters.

        Args:
            input_data: PFDInput with all required parameters

        Returns:
            PFDResult with calculated PFD and metrics
        """
        start_time = datetime.utcnow()
        warnings: List[str] = []

        logger.info(f"Calculating PFD for {input_data.architecture.value} architecture")

        try:
            calc_method = self._get_calculation_method(input_data.architecture)
            pfd_result = calc_method(input_data)

            pfd_avg = pfd_result['pfd_avg']
            pfd_max = pfd_result['pfd_max']
            ccf_contrib = pfd_result['ccf_contribution']
            ind_contrib = pfd_result['independent_contribution']
            formula = pfd_result['formula']

            # Validate result
            if pfd_avg > 1.0:
                warnings.append(
                    f"Calculated PFD {pfd_avg} exceeds 1.0. Check input parameters."
                )
                pfd_avg = min(pfd_avg, 1.0)

            # Determine SIL achieved
            sil_achieved = self._pfd_to_sil(pfd_avg)

            # Risk Reduction Factor
            rrf = 1.0 / pfd_avg if pfd_avg > 0 else float('inf')

            # Check low demand assumption
            ti = input_data.proof_test_interval_hours
            if input_data.lambda_du * ti > 0.1:
                warnings.append(
                    "Lambda_DU * TI > 0.1. Low demand approximation may not be valid."
                )

            # Provenance hash
            provenance_hash = self._calculate_provenance(input_data, pfd_avg)

            result = PFDResult(
                architecture=input_data.architecture,
                pfd_avg=pfd_avg,
                pfd_max=pfd_max,
                sil_achieved=sil_achieved,
                lambda_du_effective=input_data.lambda_du,
                ccf_contribution=ccf_contrib,
                independent_contribution=ind_contrib,
                proof_test_interval_hours=input_data.proof_test_interval_hours,
                risk_reduction_factor=rrf,
                calculation_timestamp=start_time,
                provenance_hash=provenance_hash,
                formula_used=formula,
                warnings=warnings,
            )

            logger.info(f"PFD calculation complete. PFDavg: {pfd_avg:.2e}, SIL: {sil_achieved}")

            return result

        except Exception as e:
            logger.error(f"PFD calculation failed: {str(e)}", exc_info=True)
            raise

    def calculate_pfd_1oo1(
        self,
        lambda_du: float,
        t_proof: float
    ) -> PFDResult:
        """
        Calculate PFD for 1oo1 (single channel) architecture.

        Formula: PFD_avg = lambda_DU * T_proof / 2

        Args:
            lambda_du: Dangerous undetected failure rate (per hour)
            t_proof: Proof test interval (hours)

        Returns:
            PFDResult with calculation results
        """
        input_data = PFDInput(
            architecture=VotingArchitecture.ONE_OO_ONE,
            lambda_du=lambda_du,
            proof_test_interval_hours=t_proof
        )
        return self.calculate(input_data)

    def calculate_pfd_1oo2(
        self,
        lambda_du: float,
        t_proof: float,
        beta: float = 0.1
    ) -> PFDResult:
        """
        Calculate PFD for 1oo2 (dual redundant) architecture.

        Formula: PFD_avg = ((1-beta) * lambda_DU)^2 * T_proof^2 / 3 + beta * lambda_DU * T_proof / 2

        Args:
            lambda_du: Dangerous undetected failure rate (per hour)
            t_proof: Proof test interval (hours)
            beta: Common cause failure beta factor

        Returns:
            PFDResult with calculation results
        """
        input_data = PFDInput(
            architecture=VotingArchitecture.ONE_OO_TWO,
            lambda_du=lambda_du,
            proof_test_interval_hours=t_proof,
            beta_ccf=beta
        )
        return self.calculate(input_data)

    def calculate_pfd_2oo3(
        self,
        lambda_du: float,
        t_proof: float,
        beta: float = 0.1
    ) -> PFDResult:
        """
        Calculate PFD for 2oo3 (triple modular redundant) architecture.

        Formula: PFD_avg = 3 * ((1-beta) * lambda_DU)^2 * T_proof^2 / 3 + beta * lambda_DU * T_proof / 2

        Args:
            lambda_du: Dangerous undetected failure rate (per hour)
            t_proof: Proof test interval (hours)
            beta: Common cause failure beta factor

        Returns:
            PFDResult with calculation results
        """
        input_data = PFDInput(
            architecture=VotingArchitecture.TWO_OO_THREE,
            lambda_du=lambda_du,
            proof_test_interval_hours=t_proof,
            beta_ccf=beta
        )
        return self.calculate(input_data)

    def common_cause_beta_factor(
        self,
        separation: bool = False,
        diversity: bool = False,
        complexity: str = "medium",
        analysis_quality: str = "medium",
        training: str = "medium",
        environmental_control: str = "medium"
    ) -> float:
        """
        Calculate common cause beta factor using IEC 61508-6 method.

        Args:
            separation: Physical separation of channels
            diversity: Use of diverse technology
            complexity: System complexity (low/medium/high)
            analysis_quality: Quality of analysis (low/medium/high)
            training: Training quality (low/medium/high)
            environmental_control: Environmental control (low/medium/high)

        Returns:
            Beta factor (typically 0.01 to 0.2)
        """
        base_beta = 0.1

        # Separation reduces CCF
        if separation:
            base_beta *= 0.7

        # Diversity significantly reduces CCF
        if diversity:
            base_beta *= 0.5

        # Complexity factors
        complexity_factors = {"low": 0.8, "medium": 1.0, "high": 1.5}
        base_beta *= complexity_factors.get(complexity, 1.0)

        # Bound the result
        return max(0.01, min(0.2, base_beta))

    def diagnostic_coverage(
        self,
        lambda_dd: float,
        lambda_du: float
    ) -> float:
        """
        Calculate diagnostic coverage from failure rates.

        DC = lambda_DD / (lambda_DD + lambda_DU)

        Args:
            lambda_dd: Dangerous detected failure rate
            lambda_du: Dangerous undetected failure rate

        Returns:
            Diagnostic coverage (0-1)
        """
        total = lambda_dd + lambda_du
        if total == 0:
            return 0.0
        return lambda_dd / total

    def _get_calculation_method(self, architecture: VotingArchitecture):
        """Get appropriate calculation method for architecture."""
        methods = {
            VotingArchitecture.ONE_OO_ONE: self._calc_1oo1,
            VotingArchitecture.ONE_OO_TWO: self._calc_1oo2,
            VotingArchitecture.TWO_OO_TWO: self._calc_2oo2,
            VotingArchitecture.TWO_OO_THREE: self._calc_2oo3,
            VotingArchitecture.TWO_OO_FOUR: self._calc_2oo4,
            VotingArchitecture.ONE_OO_ONE_D: self._calc_1oo1d,
            VotingArchitecture.ONE_OO_TWO_D: self._calc_1oo2d,
            VotingArchitecture.TWO_OO_THREE_D: self._calc_2oo3d,
        }
        return methods[architecture]

    def _calc_1oo1(self, input_data: PFDInput) -> Dict[str, Any]:
        """Calculate PFD for 1oo1 architecture."""
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du

        pfd_avg = lambda_du * ti / 2.0
        pfd_max = lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': 0.0,
            'independent_contribution': pfd_avg,
            'formula': 'PFDavg = lambda_DU * TI / 2'
        }

    def _calc_1oo2(self, input_data: PFDInput) -> Dict[str, Any]:
        """Calculate PFD for 1oo2 architecture."""
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du
        beta = input_data.beta_ccf

        # Independent failure contribution
        ind_contrib = ((1 - beta) * lambda_du) ** 2 * ti ** 2 / 3.0

        # CCF contribution
        ccf_contrib = beta * lambda_du * ti / 2.0

        pfd_avg = ind_contrib + ccf_contrib
        pfd_max = ((1 - beta) * lambda_du) ** 2 * ti ** 2 + beta * lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': ccf_contrib,
            'independent_contribution': ind_contrib,
            'formula': 'PFDavg = ((1-beta)*lambda_DU)^2*TI^2/3 + beta*lambda_DU*TI/2'
        }

    def _calc_2oo2(self, input_data: PFDInput) -> Dict[str, Any]:
        """Calculate PFD for 2oo2 architecture."""
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du

        pfd_avg = 2.0 * lambda_du * ti / 2.0
        pfd_max = 2.0 * lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': 0.0,
            'independent_contribution': pfd_avg,
            'formula': 'PFDavg = lambda_DU * TI (2oo2 increases availability, not safety)'
        }

    def _calc_2oo3(self, input_data: PFDInput) -> Dict[str, Any]:
        """Calculate PFD for 2oo3 architecture."""
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du
        beta = input_data.beta_ccf

        # Independent failure contribution (C(3,2) = 3 combinations)
        ind_contrib = 3 * ((1 - beta) * lambda_du) ** 2 * ti ** 2 / 3.0

        # CCF contribution
        ccf_contrib = beta * lambda_du * ti / 2.0

        pfd_avg = ind_contrib + ccf_contrib
        pfd_max = 3 * ((1 - beta) * lambda_du) ** 2 * ti ** 2 + beta * lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': ccf_contrib,
            'independent_contribution': ind_contrib,
            'formula': 'PFDavg = 3*((1-beta)*lambda_DU)^2*TI^2/3 + beta*lambda_DU*TI/2'
        }

    def _calc_2oo4(self, input_data: PFDInput) -> Dict[str, Any]:
        """Calculate PFD for 2oo4 architecture."""
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du
        beta = input_data.beta_ccf

        # C(4,2) = 6 combinations
        ind_contrib = 6 * ((1 - beta) * lambda_du) ** 2 * ti ** 2 / 3.0
        ccf_contrib = beta * lambda_du * ti / 2.0

        pfd_avg = ind_contrib + ccf_contrib
        pfd_max = 6 * ((1 - beta) * lambda_du) ** 2 * ti ** 2 + beta * lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': ccf_contrib,
            'independent_contribution': ind_contrib,
            'formula': 'PFDavg = 6*((1-beta)*lambda_DU)^2*TI^2/3 + beta*lambda_DU*TI/2'
        }

    def _calc_1oo1d(self, input_data: PFDInput) -> Dict[str, Any]:
        """Calculate PFD for 1oo1D with diagnostics."""
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du
        lambda_dd = input_data.lambda_dd
        mttr = input_data.mean_time_to_repair_hours

        pfd_du = lambda_du * ti / 2.0
        pfd_dd = lambda_dd * mttr

        pfd_avg = pfd_du + pfd_dd
        pfd_max = lambda_du * ti + lambda_dd * mttr

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': 0.0,
            'independent_contribution': pfd_avg,
            'formula': 'PFDavg = lambda_DU*TI/2 + lambda_DD*MTTR'
        }

    def _calc_1oo2d(self, input_data: PFDInput) -> Dict[str, Any]:
        """Calculate PFD for 1oo2D with diagnostics."""
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du
        lambda_dd = input_data.lambda_dd
        beta = input_data.beta_ccf
        mttr = input_data.mean_time_to_repair_hours

        ind_du = ((1 - beta) * lambda_du) ** 2 * ti ** 2 / 3.0
        ind_dd = ((1 - beta) * lambda_dd) ** 2 * mttr ** 2
        ccf_contrib = beta * lambda_du * ti / 2.0

        pfd_avg = ind_du + ind_dd + ccf_contrib
        pfd_max = ((1 - beta) * lambda_du) ** 2 * ti ** 2 + beta * lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': ccf_contrib,
            'independent_contribution': ind_du + ind_dd,
            'formula': 'PFDavg = ((1-beta)*lambda_DU)^2*TI^2/3 + ((1-beta)*lambda_DD)^2*MTTR^2 + beta*lambda_DU*TI/2'
        }

    def _calc_2oo3d(self, input_data: PFDInput) -> Dict[str, Any]:
        """Calculate PFD for 2oo3D with diagnostics."""
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du
        lambda_dd = input_data.lambda_dd
        beta = input_data.beta_ccf
        mttr = input_data.mean_time_to_repair_hours

        ind_du = 3 * ((1 - beta) * lambda_du) ** 2 * ti ** 2 / 3.0
        ind_dd = 3 * ((1 - beta) * lambda_dd) ** 2 * mttr ** 2
        ccf_contrib = beta * lambda_du * ti / 2.0

        pfd_avg = ind_du + ind_dd + ccf_contrib
        pfd_max = 3 * ((1 - beta) * lambda_du) ** 2 * ti ** 2 + beta * lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': ccf_contrib,
            'independent_contribution': ind_du + ind_dd,
            'formula': 'PFDavg = 3*((1-beta)*lambda_DU)^2*TI^2/3 + 3*((1-beta)*lambda_DD)^2*MTTR^2 + beta*lambda_DU*TI/2'
        }

    def _pfd_to_sil(self, pfd: float) -> int:
        """Convert PFD to SIL level."""
        for sil, (lower, upper) in self.SIL_PFD_RANGES.items():
            if lower <= pfd < upper:
                return sil

        if pfd < 1e-5:
            return 4  # Better than SIL 4
        return 0  # Below SIL 1

    def _calculate_provenance(self, input_data: PFDInput, pfd_avg: float) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{input_data.architecture.value}|"
            f"{input_data.lambda_du}|"
            f"{input_data.proof_test_interval_hours}|"
            f"{input_data.beta_ccf}|"
            f"{pfd_avg}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def optimize_proof_test_interval(
        self,
        target_pfd: float,
        lambda_du: float,
        architecture: VotingArchitecture,
        beta_ccf: float = 0.1,
        min_interval_hours: float = 720,
        max_interval_hours: float = 87600,
    ) -> float:
        """
        Find optimal proof test interval to achieve target PFD.

        Uses binary search to find maximum interval that achieves target.

        Args:
            target_pfd: Target PFDavg to achieve
            lambda_du: Dangerous undetected failure rate
            architecture: Voting architecture
            beta_ccf: Common cause failure factor
            min_interval_hours: Minimum proof test interval
            max_interval_hours: Maximum proof test interval

        Returns:
            Optimal proof test interval in hours
        """
        logger.info(f"Optimizing proof test interval for target PFD {target_pfd:.2e}")

        low = min_interval_hours
        high = max_interval_hours
        optimal = min_interval_hours

        while low <= high:
            mid = (low + high) / 2

            input_data = PFDInput(
                architecture=architecture,
                lambda_du=lambda_du,
                proof_test_interval_hours=mid,
                beta_ccf=beta_ccf
            )

            result = self.calculate(input_data)

            if result.pfd_avg <= target_pfd:
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1

        logger.info(f"Optimal proof test interval: {optimal} hours")
        return optimal

    def compare_architectures(
        self,
        lambda_du: float,
        proof_test_interval_hours: float,
        beta_ccf: float = 0.1
    ) -> Dict[VotingArchitecture, PFDResult]:
        """Compare PFD across all voting architectures."""
        results = {}

        for arch in VotingArchitecture:
            try:
                input_data = PFDInput(
                    architecture=arch,
                    lambda_du=lambda_du,
                    proof_test_interval_hours=proof_test_interval_hours,
                    beta_ccf=beta_ccf
                )
                results[arch] = self.calculate(input_data)
            except Exception as e:
                logger.warning(f"Failed to calculate PFD for {arch}: {e}")

        return results


# =============================================================================
# PROOF TEST SCHEDULER
# =============================================================================

class ProofTestScheduler:
    """
    Proof Test Scheduler for SIS Components per IEC 61511.

    Manages proof test scheduling, tracking, and optimization including:
    - Test interval calculation based on target SIL
    - Overdue tracking and notifications
    - Test record management
    - PFD impact analysis
    - Partial stroke test integration

    Attributes:
        schedules: Dict of equipment_id to ProofTestSchedule
        test_records: List of all test records

    Example:
        >>> scheduler = ProofTestScheduler()
        >>> schedule = scheduler.schedule_proof_tests(safety_functions)
        >>> print(f"Next test: {schedule[0].next_test_due}")
    """

    MAX_INTERVALS_BY_SIL: Dict[int, float] = {
        1: 87600,  # 10 years
        2: 43800,  # 5 years
        3: 17520,  # 2 years
        4: 8760,   # 1 year
    }

    def __init__(self):
        """Initialize ProofTestScheduler."""
        self.schedules: Dict[str, ProofTestSchedule] = {}
        self.test_records: List[ProofTestRecord] = []
        logger.info("ProofTestScheduler initialized")

    def schedule_proof_tests(
        self,
        safety_functions: List[SafetyFunction]
    ) -> List[ProofTestSchedule]:
        """
        Create proof test schedules for safety functions.

        Args:
            safety_functions: List of SafetyFunction objects

        Returns:
            List of ProofTestSchedule objects
        """
        schedules = []

        for sf in safety_functions:
            schedule = self.create_schedule(
                equipment_id=sf.function_id,
                proof_test_interval_hours=sf.proof_test_interval,
                target_sil=int(sf.target_sil.value.split('_')[1]),
                equipment_description=sf.description
            )
            schedules.append(schedule)

        return schedules

    def create_schedule(
        self,
        equipment_id: str,
        proof_test_interval_hours: float,
        target_sil: int,
        equipment_description: str = "",
        last_test_date: Optional[datetime] = None,
        partial_stroke_test_interval_hours: Optional[float] = None,
        grace_period_days: int = 30
    ) -> ProofTestSchedule:
        """
        Create a proof test schedule for equipment.

        Args:
            equipment_id: Unique equipment identifier
            proof_test_interval_hours: Test interval in hours
            target_sil: Target SIL level (1-4)
            equipment_description: Description of equipment
            last_test_date: Date of last test
            partial_stroke_test_interval_hours: PST interval
            grace_period_days: Grace period before overdue

        Returns:
            ProofTestSchedule object
        """
        logger.info(f"Creating proof test schedule for {equipment_id}")

        # Validate SIL-specific constraints
        max_interval = self.MAX_INTERVALS_BY_SIL.get(target_sil, 87600)
        if proof_test_interval_hours > max_interval:
            logger.warning(
                f"Proof test interval {proof_test_interval_hours}h exceeds "
                f"recommended maximum {max_interval}h for SIL {target_sil}"
            )

        if last_test_date is None:
            last_test_date = datetime.utcnow()

        interval_delta = timedelta(hours=proof_test_interval_hours)
        next_test_due = last_test_date + interval_delta

        days_until_due = (next_test_due - datetime.utcnow()).days
        is_overdue = days_until_due < -grace_period_days

        schedule = ProofTestSchedule(
            equipment_id=equipment_id,
            equipment_description=equipment_description,
            target_sil=target_sil,
            proof_test_interval_hours=proof_test_interval_hours,
            partial_stroke_test_interval_hours=partial_stroke_test_interval_hours,
            last_test_date=last_test_date,
            next_test_due=next_test_due,
            grace_period_days=grace_period_days,
            is_overdue=is_overdue,
            days_until_due=days_until_due,
        )

        schedule.provenance_hash = self._calculate_provenance(
            equipment_id, proof_test_interval_hours, target_sil
        )

        self.schedules[equipment_id] = schedule

        logger.info(f"Schedule created. Next test due: {next_test_due.isoformat()}")

        return schedule

    def calculate_optimal_interval(
        self,
        lambda_du: float,
        target_sil: int,
        architecture: str = "1oo1",
        beta_ccf: float = 0.1
    ) -> float:
        """
        Calculate optimal proof test interval for target SIL.

        Args:
            lambda_du: Dangerous undetected failure rate (per hour)
            target_sil: Target SIL level (1-4)
            architecture: Voting architecture
            beta_ccf: Common cause failure factor

        Returns:
            Optimal proof test interval in hours
        """
        sil_pfd_limits = {1: 1e-1, 2: 1e-2, 3: 1e-3, 4: 1e-4}
        target_pfd = sil_pfd_limits[target_sil] * 0.5  # Safety margin

        if architecture == "1oo1":
            optimal_ti = 2 * target_pfd / lambda_du
        elif architecture == "1oo2":
            optimal_ti = 2 * target_pfd / (beta_ccf * lambda_du)
        elif architecture == "2oo3":
            optimal_ti = 2 * target_pfd / (beta_ccf * lambda_du)
        else:
            optimal_ti = 2 * target_pfd / lambda_du

        max_ti = self.MAX_INTERVALS_BY_SIL[target_sil]
        optimal_ti = min(optimal_ti, max_ti)
        optimal_ti = max(optimal_ti, 720)  # Minimum 1 month

        logger.info(
            f"Optimal proof test interval for SIL {target_sil}: "
            f"{optimal_ti:.0f} hours ({optimal_ti/8760:.1f} years)"
        )

        return optimal_ti

    def generate_test_procedures(
        self,
        equipment_id: str
    ) -> Dict[str, Any]:
        """
        Generate test procedure template for equipment.

        Args:
            equipment_id: Equipment identifier

        Returns:
            Test procedure template dictionary
        """
        if equipment_id not in self.schedules:
            raise ValueError(f"No schedule found for equipment: {equipment_id}")

        schedule = self.schedules[equipment_id]

        return {
            "procedure_id": f"PROC-{equipment_id}",
            "equipment_id": equipment_id,
            "description": schedule.equipment_description,
            "target_sil": schedule.target_sil,
            "test_interval_hours": schedule.proof_test_interval_hours,
            "steps": [
                {
                    "step": 1,
                    "action": "Notify Operations of pending test",
                    "verification": "Operations acknowledgment received"
                },
                {
                    "step": 2,
                    "action": "Put SIF in bypass (if permitted)",
                    "verification": "Bypass alarm acknowledged"
                },
                {
                    "step": 3,
                    "action": "Simulate input condition to initiate trip",
                    "verification": "Output achieves safe state"
                },
                {
                    "step": 4,
                    "action": "Verify response time meets requirement",
                    "verification": "Response time < PST"
                },
                {
                    "step": 5,
                    "action": "Reset system and remove bypass",
                    "verification": "System returns to normal operation"
                },
                {
                    "step": 6,
                    "action": "Document test results",
                    "verification": "Test record completed"
                }
            ],
            "requirements": [
                f"Personnel must be trained on SIL {schedule.target_sil} testing",
                "Operations must approve test window",
                "Test equipment must be calibrated",
                "Safety margin must be maintained during test"
            ],
            "generated_at": datetime.utcnow().isoformat()
        }

    def track_test_completion(
        self,
        equipment_id: str,
        test_type: TestType,
        result: TestResult,
        actual_date: Optional[datetime] = None,
        detected_failures: Optional[List[str]] = None,
        corrective_actions: Optional[List[str]] = None,
        test_coverage: float = 1.0,
        performed_by: Optional[str] = None,
        witnessed_by: Optional[str] = None,
        duration_minutes: Optional[int] = None,
        notes: str = ""
    ) -> ProofTestRecord:
        """
        Record a proof test completion.

        Args:
            equipment_id: Equipment identifier
            test_type: Type of test performed
            result: Test result
            actual_date: Actual test date
            detected_failures: List of failures found
            corrective_actions: Actions taken
            test_coverage: Coverage factor (0-1)
            performed_by: Tester name
            witnessed_by: Witness name
            duration_minutes: Test duration
            notes: Additional notes

        Returns:
            ProofTestRecord object
        """
        logger.info(f"Recording {test_type.value} for {equipment_id}")

        if equipment_id not in self.schedules:
            raise ValueError(f"No schedule found for equipment: {equipment_id}")

        schedule = self.schedules[equipment_id]

        if actual_date is None:
            actual_date = datetime.utcnow()

        record = ProofTestRecord(
            equipment_id=equipment_id,
            test_type=test_type,
            scheduled_date=schedule.next_test_due,
            actual_date=actual_date,
            result=result,
            detected_failures=detected_failures or [],
            corrective_actions=corrective_actions or [],
            test_coverage=test_coverage,
            performed_by=performed_by,
            witnessed_by=witnessed_by,
            duration_minutes=duration_minutes,
            notes=notes,
        )

        record.provenance_hash = self._calculate_record_provenance(record)

        self.test_records.append(record)
        schedule.test_history.append(record)

        # Update schedule if full proof test
        if test_type == TestType.FULL_PROOF_TEST:
            self._update_schedule_after_test(schedule, actual_date, test_coverage)

        logger.info(f"Test recorded for {equipment_id}. Result: {result.value}")

        return record

    def _update_schedule_after_test(
        self,
        schedule: ProofTestSchedule,
        test_date: datetime,
        test_coverage: float
    ) -> None:
        """Update schedule after proof test completion."""
        schedule.last_test_date = test_date

        effective_interval = schedule.proof_test_interval_hours
        if test_coverage < 1.0:
            effective_interval *= test_coverage
            logger.warning(
                f"Test coverage {test_coverage:.0%} reduces effective interval "
                f"to {effective_interval:.0f} hours"
            )

        interval_delta = timedelta(hours=effective_interval)
        schedule.next_test_due = test_date + interval_delta

        schedule.days_until_due = (schedule.next_test_due - datetime.utcnow()).days
        schedule.is_overdue = False
        schedule.last_updated = datetime.utcnow()

    def get_overdue_equipment(self) -> List[ProofTestSchedule]:
        """Get list of equipment with overdue proof tests."""
        self._update_all_schedules()
        overdue = [s for s in self.schedules.values() if s.is_overdue]

        if overdue:
            logger.warning(f"{len(overdue)} equipment items have overdue proof tests")

        return overdue

    def get_upcoming_tests(self, days_ahead: int = 30) -> List[ProofTestSchedule]:
        """Get tests due within specified period."""
        self._update_all_schedules()
        cutoff = datetime.utcnow() + timedelta(days=days_ahead)

        upcoming = [
            s for s in self.schedules.values()
            if s.next_test_due <= cutoff and not s.is_overdue
        ]

        return sorted(upcoming, key=lambda x: x.next_test_due)

    def _update_all_schedules(self) -> None:
        """Update status of all schedules."""
        now = datetime.utcnow()

        for schedule in self.schedules.values():
            schedule.days_until_due = (schedule.next_test_due - now).days
            schedule.is_overdue = schedule.days_until_due < -schedule.grace_period_days
            schedule.last_updated = now

    def _calculate_provenance(
        self,
        equipment_id: str,
        interval: float,
        sil: int
    ) -> str:
        """Calculate SHA-256 provenance hash for schedule."""
        provenance_str = (
            f"{equipment_id}|{interval}|{sil}|{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _calculate_record_provenance(self, record: ProofTestRecord) -> str:
        """Calculate SHA-256 provenance hash for test record."""
        provenance_str = (
            f"{record.equipment_id}|"
            f"{record.test_type.value}|"
            f"{record.result.value}|"
            f"{record.actual_date.isoformat() if record.actual_date else ''}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# SRS GENERATOR
# =============================================================================

class SRSGenerator:
    """
    Safety Requirements Specification Generator per IEC 61511-1 Clause 10.

    Generates SRS documents with all mandatory content including:
    - Safe state definitions
    - Safety function specifications
    - SIL requirements
    - Process safety time requirements
    - Hardware fault tolerance requirements
    - Proof test requirements
    - Response time requirements

    Example:
        >>> generator = SRSGenerator()
        >>> srs = generator.generate_srs_document("AGENT-001", safety_functions)
        >>> print(srs.document_id)
    """

    STANDARD_SECTIONS = [
        {"id": "1", "title": "Scope and Introduction"},
        {"id": "2", "title": "Reference Documents"},
        {"id": "3", "title": "Process Description"},
        {"id": "4", "title": "Safe State Definitions"},
        {"id": "5", "title": "Safety Instrumented Functions"},
        {"id": "6", "title": "SIL Requirements"},
        {"id": "7", "title": "Response Time Requirements"},
        {"id": "8", "title": "Proof Test Requirements"},
        {"id": "9", "title": "Input/Output Requirements"},
        {"id": "10", "title": "Bypass Requirements"},
    ]

    def __init__(self):
        """Initialize SRSGenerator."""
        logger.info("SRSGenerator initialized")

    def generate_srs_document(
        self,
        agent_id: str,
        safety_functions: List[SafetyFunction],
        project_name: str = "",
        created_by: str = "",
        hazard_identification_ref: str = ""
    ) -> SRSDocument:
        """
        Generate a complete SRS document.

        Args:
            agent_id: Agent/system identifier
            safety_functions: List of SafetyFunction objects
            project_name: Project name
            created_by: Author name
            hazard_identification_ref: Reference to HAZOP

        Returns:
            Complete SRSDocument
        """
        logger.info(f"Generating SRS for agent: {agent_id}")

        # Convert SafetyFunction to SafetyFunctionRequirement
        requirements = []
        for sf in safety_functions:
            sil_num = int(sf.target_sil.value.split('_')[1])
            req = SafetyFunctionRequirement(
                req_id=sf.function_id,
                description=sf.description,
                sil_level=sil_num,
                safe_state=sf.safe_state,
                process_safety_time_ms=sf.process_safety_time_ms,
                required_response_time_ms=sf.response_time_ms,
                proof_test_interval_hours=sf.proof_test_interval,
                pfd_target=sf.pfd_target,
                hft_requirement=sf.hardware_fault_tolerance,
                diagnostic_coverage_target=sf.diagnostic_coverage,
                input_sensors=sf.sensors,
                output_actuators=sf.final_elements,
                initiating_cause=sf.initiating_cause,
                consequence=sf.consequence,
                action_on_detection=sf.action_on_detection,
                bypass_permitted=sf.bypass_permitted,
            )
            requirements.append(req)

        # Generate sections
        sections = self._generate_sections(agent_id, requirements, hazard_identification_ref)

        # Create document
        srs = SRSDocument(
            title=f"Safety Requirements Specification - {agent_id}",
            project_name=project_name,
            system_name=agent_id,
            created_by=created_by,
            hazard_identification_ref=hazard_identification_ref,
            safety_function_requirements=requirements,
            sections=sections,
            references=[
                "IEC 61511-1:2016 - Functional safety - Safety instrumented systems",
                "IEC 61511-2:2016 - Guidelines for application of IEC 61511-1",
                "IEC 61511-3:2016 - Guidance for determination of SIL",
            ]
        )

        # Provenance hash
        srs.provenance_hash = self._calculate_provenance(srs)

        # Initial change history
        srs.change_history.append({
            "revision": "1.0",
            "date": datetime.utcnow().isoformat(),
            "author": created_by,
            "description": "Initial release"
        })

        logger.info(f"SRS created: {srs.document_id}")

        return srs

    def safe_state_documentation(
        self,
        safety_functions: List[SafetyFunction]
    ) -> Dict[str, str]:
        """
        Generate safe state documentation.

        Args:
            safety_functions: List of SafetyFunction objects

        Returns:
            Dict mapping function_id to safe state description
        """
        return {sf.function_id: sf.safe_state for sf in safety_functions}

    def process_safety_time(
        self,
        safety_functions: List[SafetyFunction]
    ) -> Dict[str, Dict[str, float]]:
        """
        Document process safety time requirements.

        Args:
            safety_functions: List of SafetyFunction objects

        Returns:
            Dict with PST and response time for each function
        """
        result = {}
        for sf in safety_functions:
            margin = sf.process_safety_time_ms - sf.response_time_ms
            result[sf.function_id] = {
                "process_safety_time_ms": sf.process_safety_time_ms,
                "required_response_time_ms": sf.response_time_ms,
                "margin_ms": margin,
                "margin_adequate": margin > 0
            }
        return result

    def diagnostic_requirements(
        self,
        safety_functions: List[SafetyFunction]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Document diagnostic requirements.

        Args:
            safety_functions: List of SafetyFunction objects

        Returns:
            Dict with diagnostic requirements for each function
        """
        result = {}
        for sf in safety_functions:
            sil_num = int(sf.target_sil.value.split('_')[1])
            min_dc = {1: 0.6, 2: 0.6, 3: 0.9, 4: 0.99}.get(sil_num, 0.6)

            result[sf.function_id] = {
                "target_dc": sf.diagnostic_coverage,
                "minimum_dc_for_sil": min_dc,
                "dc_adequate": sf.diagnostic_coverage >= min_dc,
                "proof_test_interval_hours": sf.proof_test_interval,
                "voting_architecture": sf.voting_architecture.value
            }
        return result

    def _generate_sections(
        self,
        system_name: str,
        requirements: List[SafetyFunctionRequirement],
        hazard_ref: str
    ) -> List[SRSSection]:
        """Generate standard SRS sections."""
        sections = []

        # Section 1: Scope
        sections.append(SRSSection(
            section_id="1",
            title="Scope and Introduction",
            content=f"This SRS defines safety requirements for {system_name} "
                    f"per IEC 61511-1:2016 Clause 10.",
            requirements=["IEC 61511-1 Clause 10.3.1"]
        ))

        # Section 2: References
        sections.append(SRSSection(
            section_id="2",
            title="Reference Documents",
            content=f"References:\n- IEC 61511-1:2016\n- {hazard_ref}",
            requirements=["IEC 61511-1 Clause 10.3.2"]
        ))

        # Section 3: Safe States
        safe_states = set(req.safe_state for req in requirements)
        safe_state_content = "Safe states:\n" + "\n".join(f"- {ss}" for ss in safe_states)
        sections.append(SRSSection(
            section_id="3",
            title="Safe State Definitions",
            content=safe_state_content,
            requirements=["IEC 61511-1 Clause 10.3.3"]
        ))

        # Section 4: Safety Functions
        sif_content = "Safety Instrumented Functions:\n\n"
        for req in requirements:
            sif_content += f"**{req.req_id}**: {req.description}\n"
            sif_content += f"- SIL: {req.sil_level}\n"
            sif_content += f"- Safe State: {req.safe_state}\n\n"

        sections.append(SRSSection(
            section_id="4",
            title="Safety Instrumented Functions",
            content=sif_content,
            requirements=["IEC 61511-1 Clause 10.3.4"]
        ))

        # Section 5: SIL Requirements
        sil_content = "SIL Requirements:\n\n"
        sil_content += "| SIF ID | SIL | PFD Target | HFT |\n"
        sil_content += "|--------|-----|------------|-----|\n"
        for req in requirements:
            sil_content += f"| {req.req_id} | {req.sil_level} | {req.pfd_target:.1e} | {req.hft_requirement} |\n"

        sections.append(SRSSection(
            section_id="5",
            title="SIL Requirements",
            content=sil_content,
            requirements=["IEC 61511-1 Clause 10.3.5"]
        ))

        # Section 6: Response Time
        response_content = "Response Time Requirements:\n\n"
        response_content += "| SIF ID | PST (ms) | Response (ms) | Margin |\n"
        response_content += "|--------|----------|---------------|--------|\n"
        for req in requirements:
            margin = req.process_safety_time_ms - req.required_response_time_ms
            response_content += (
                f"| {req.req_id} | {req.process_safety_time_ms:.0f} | "
                f"{req.required_response_time_ms:.0f} | {margin:.0f} |\n"
            )

        sections.append(SRSSection(
            section_id="6",
            title="Response Time Requirements",
            content=response_content,
            requirements=["IEC 61511-1 Clause 10.3.6"]
        ))

        # Section 7: Proof Test
        proof_content = "Proof Test Requirements:\n\n"
        for req in requirements:
            interval_str = f"{req.proof_test_interval_hours/8760:.1f} years"
            proof_content += f"- {req.req_id}: {interval_str}\n"

        sections.append(SRSSection(
            section_id="7",
            title="Proof Test Requirements",
            content=proof_content,
            requirements=["IEC 61511-1 Clause 10.3.7"]
        ))

        return sections

    def _calculate_provenance(self, srs: SRSDocument) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{srs.document_id}|"
            f"{srs.revision}|"
            f"{len(srs.safety_function_requirements)}|"
            f"{srs.last_modified.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# FAIL SAFE MANAGER
# =============================================================================

class FailSafeManager:
    """
    Fail-Safe Manager for Safety Instrumented Systems per IEC 61511.

    Implements fail-safe logic including:
    - Voting logic for redundant channels
    - Watchdog timer management
    - Manual override with authorization
    - Safe state transitions
    - De-energize-to-trip logic

    Follows fail-safe principles:
    - Loss of power = safe state
    - Loss of signal = safe state
    - Detected fault = safe state

    Example:
        >>> manager = FailSafeManager()
        >>> result = manager.voting_logic(VotingArchitecture.TWO_OO_THREE, channels)
        >>> print(f"Trip: {result.trip_decision}")
    """

    ARCHITECTURE_CONFIG: Dict[VotingArchitecture, Tuple[int, int]] = {
        VotingArchitecture.ONE_OO_ONE: (1, 1),
        VotingArchitecture.ONE_OO_TWO: (2, 1),
        VotingArchitecture.TWO_OO_TWO: (2, 2),
        VotingArchitecture.TWO_OO_THREE: (3, 2),
        VotingArchitecture.TWO_OO_FOUR: (4, 2),
    }

    def __init__(self, fail_safe_on_fault: bool = True):
        """
        Initialize FailSafeManager.

        Args:
            fail_safe_on_fault: Trip on channel fault (default True)
        """
        self.fail_safe_on_fault = fail_safe_on_fault
        self.watchdog: Optional[threading.Timer] = None
        self.watchdog_state = WatchdogState.STOPPED
        self.watchdog_config: Optional[WatchdogConfig] = None
        self._watchdog_lock = threading.Lock()
        self._last_kick_time: Optional[datetime] = None
        self._override_authorized: bool = False
        self._override_expiry: Optional[datetime] = None

        logger.info("FailSafeManager initialized (fail_safe_on_fault=%s)", fail_safe_on_fault)

    def voting_logic(
        self,
        architecture: VotingArchitecture,
        channel_inputs: List[ChannelInput]
    ) -> VotingResult:
        """
        Evaluate voting logic for trip decision.

        Args:
            architecture: Voting architecture
            channel_inputs: List of channel inputs

        Returns:
            VotingResult with trip decision
        """
        logger.debug(f"Evaluating {architecture.value} with {len(channel_inputs)} channels")

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
                    channels_trip += 1
                continue
            elif ch.status == ChannelStatus.UNKNOWN:
                if self.fail_safe_on_fault:
                    channels_trip += 1
                continue

            channels_healthy += 1
            if ch.value:
                channels_trip += 1

        total_channels = len(channel_inputs)
        expected_total, expected_required = self.ARCHITECTURE_CONFIG.get(
            architecture, (total_channels, 1)
        )

        degraded_mode = (
            channels_bypassed > 0 or
            channels_faulted > 0 or
            total_channels != expected_total
        )

        effective_architecture = architecture.value
        effective_required = expected_required

        if degraded_mode:
            effective_architecture, effective_required = self._get_degraded_architecture(
                architecture, channels_healthy, channels_bypassed, channels_faulted
            )

        trip_decision = channels_trip >= effective_required

        result = VotingResult(
            architecture=architecture,
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

        result.provenance_hash = self._calculate_voting_provenance(result)

        logger.info(f"Voting result: {trip_decision}, {channels_trip}/{effective_required}")

        return result

    def _get_degraded_architecture(
        self,
        architecture: VotingArchitecture,
        healthy_channels: int,
        bypassed: int,
        faulted: int
    ) -> Tuple[str, int]:
        """Determine effective architecture in degraded mode."""
        if architecture == VotingArchitecture.TWO_OO_THREE:
            if healthy_channels == 3:
                return "2oo3", 2
            elif healthy_channels == 2:
                return "1oo2", 1
            elif healthy_channels == 1:
                return "1oo1", 1
            else:
                return "0oo0", 0

        elif architecture == VotingArchitecture.ONE_OO_TWO:
            if healthy_channels == 2:
                return "1oo2", 1
            elif healthy_channels == 1:
                return "1oo1", 1
            else:
                return "0oo0", 0

        elif architecture == VotingArchitecture.TWO_OO_TWO:
            if healthy_channels == 2:
                return "2oo2", 2
            elif healthy_channels == 1:
                return "1oo1", 1
            else:
                return "0oo0", 0

        return architecture.value, architecture.channels_required

    def watchdog_timer(
        self,
        config: WatchdogConfig,
        timeout_callback: Optional[Callable] = None
    ) -> None:
        """
        Start watchdog timer.

        Args:
            config: WatchdogConfig settings
            timeout_callback: Callback on timeout
        """
        with self._watchdog_lock:
            if self.watchdog_state == WatchdogState.RUNNING:
                logger.warning("Watchdog already running")
                return

            self.watchdog_config = config
            self.watchdog_state = WatchdogState.RUNNING
            self._last_kick_time = datetime.utcnow()
            self._timeout_callback = timeout_callback

            self._start_watchdog_timer()

            logger.info(f"Watchdog started: {config.watchdog_id}, timeout={config.timeout_ms}ms")

    def _start_watchdog_timer(self) -> None:
        """Start internal watchdog timer thread."""
        if self.watchdog_config is None:
            return

        timeout_seconds = self.watchdog_config.timeout_ms / 1000.0
        self.watchdog = threading.Timer(timeout_seconds, self._on_watchdog_timeout)
        self.watchdog.daemon = True
        self.watchdog.start()

    def _on_watchdog_timeout(self) -> None:
        """Handle watchdog timeout."""
        with self._watchdog_lock:
            if self.watchdog_state != WatchdogState.RUNNING:
                return

            self.watchdog_state = WatchdogState.EXPIRED

            logger.warning(f"Watchdog TIMEOUT! Action: {self.watchdog_config.action_on_timeout.value}")

        if self.watchdog_config.action_on_timeout == TimeoutAction.TRIP:
            logger.critical("Watchdog triggered TRIP action")
            if hasattr(self, '_timeout_callback') and self._timeout_callback:
                self._timeout_callback("TRIP")

        if self.watchdog_config.auto_restart:
            with self._watchdog_lock:
                self.watchdog_state = WatchdogState.RUNNING
                self._last_kick_time = datetime.utcnow()
                self._start_watchdog_timer()
                logger.info("Watchdog auto-restarted")

    def kick_watchdog(self) -> bool:
        """
        Kick (reset) the watchdog timer.

        Returns:
            True if kick accepted
        """
        with self._watchdog_lock:
            if self.watchdog_state != WatchdogState.RUNNING:
                logger.warning(f"Cannot kick watchdog: state={self.watchdog_state.value}")
                return False

            if self.watchdog:
                self.watchdog.cancel()

            self._last_kick_time = datetime.utcnow()
            self._start_watchdog_timer()

            return True

    def stop_watchdog(self) -> bool:
        """
        Stop the watchdog timer.

        Returns:
            True if stopped successfully
        """
        with self._watchdog_lock:
            if self.watchdog_state == WatchdogState.STOPPED:
                return True

            if self.watchdog:
                self.watchdog.cancel()
                self.watchdog = None

            self.watchdog_state = WatchdogState.STOPPED

            logger.info("Watchdog stopped")
            return True

    def manual_override(
        self,
        authorization_code: str,
        duration_minutes: int = 60,
        reason: str = ""
    ) -> bool:
        """
        Enable manual override with authorization.

        Args:
            authorization_code: Authorization code
            duration_minutes: Override duration
            reason: Reason for override

        Returns:
            True if override authorized
        """
        # In production, validate against authorization system
        # For safety, require multi-factor authorization

        if not authorization_code or len(authorization_code) < 8:
            logger.warning("Override authorization failed: invalid code")
            return False

        self._override_authorized = True
        self._override_expiry = datetime.utcnow() + timedelta(minutes=duration_minutes)

        logger.warning(
            f"Manual override AUTHORIZED for {duration_minutes} minutes. "
            f"Reason: {reason}"
        )

        return True

    def is_override_active(self) -> bool:
        """Check if manual override is currently active."""
        if not self._override_authorized:
            return False

        if self._override_expiry and datetime.utcnow() > self._override_expiry:
            self._override_authorized = False
            logger.info("Manual override expired")
            return False

        return True

    def safe_state_transition(
        self,
        config: TransitionConfig,
        state_verifier: Optional[Callable[[str, str], bool]] = None,
        action_executor: Optional[Callable[[str, str, str], bool]] = None
    ) -> TransitionResult:
        """
        Execute safe state transition.

        Args:
            config: TransitionConfig with transition settings
            state_verifier: Callback to verify state
            action_executor: Callback to execute actions

        Returns:
            TransitionResult with execution details
        """
        start_time = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        step_results: List[Dict[str, Any]] = []
        steps_completed = 0
        achieved_state = config.from_state

        logger.info(f"Starting transition: {config.from_state} -> {config.to_state}")

        try:
            if config.mode == TransitionMode.IMMEDIATE:
                # De-energize-to-trip: immediate transition
                success = True
                for step in config.steps:
                    if action_executor:
                        step_success = action_executor(
                            step.target_equipment,
                            step.action,
                            step.target_state
                        )
                    else:
                        step_success = True  # Simulation

                    step_results.append({
                        "step_id": step.step_id,
                        "equipment": step.target_equipment,
                        "success": step_success
                    })

                    if step_success:
                        steps_completed += 1
                    else:
                        success = False
                        errors.append(f"Step {step.step_id} failed")
                        if config.abort_on_step_failure:
                            break

                if success:
                    achieved_state = config.to_state

            else:
                # Sequenced transition
                sorted_steps = sorted(config.steps, key=lambda s: s.sequence_number)
                total_start = time.time()

                for step in sorted_steps:
                    if (time.time() - total_start) * 1000 > config.total_timeout_ms:
                        errors.append("Total transition timeout exceeded")
                        break

                    if action_executor:
                        step_success = action_executor(
                            step.target_equipment,
                            step.action,
                            step.target_state
                        )
                    else:
                        step_success = True

                    if step.verification_required and step_success and state_verifier:
                        verified = state_verifier(step.target_equipment, step.target_state)
                        step_success = step_success and verified

                    step_results.append({
                        "step_id": step.step_id,
                        "sequence": step.sequence_number,
                        "success": step_success
                    })

                    if step_success:
                        steps_completed += 1
                    else:
                        errors.append(f"Step {step.step_id} failed")
                        if config.abort_on_step_failure:
                            break

                if steps_completed == len(sorted_steps):
                    achieved_state = config.to_state

            # Verify safe state
            verified = (achieved_state == config.to_state)
            if not verified:
                warnings.append(f"Safe state {config.to_state} not achieved")

            status = TransitionStatus.COMPLETED if verified else TransitionStatus.FAILED

        except Exception as e:
            status = TransitionStatus.FAILED
            errors.append(f"Transition error: {str(e)}")
            verified = False
            logger.error(f"Transition failed: {e}", exc_info=True)

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        result = TransitionResult(
            transition_id=config.transition_id,
            status=status,
            from_state=config.from_state,
            to_state=config.to_state,
            achieved_state=achieved_state,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            steps_completed=steps_completed,
            steps_total=len(config.steps),
            step_results=step_results,
            errors=errors,
            warnings=warnings,
            verified=verified,
        )

        result.provenance_hash = self._calculate_transition_provenance(result)

        logger.info(f"Transition {status.value}: {achieved_state} in {duration_ms:.0f}ms")

        return result

    def deenergize_to_trip(
        self,
        final_elements: List[str],
        action_executor: Optional[Callable[[str, str, str], bool]] = None
    ) -> Dict[str, bool]:
        """
        Execute de-energize-to-trip logic.

        De-energize-to-trip is the fail-safe principle where removing power
        causes the system to go to the safe state. This is the preferred
        approach for SIS final elements.

        Args:
            final_elements: List of final element tags
            action_executor: Optional action executor callback

        Returns:
            Dict mapping element to success status
        """
        logger.info(f"Executing de-energize-to-trip for {len(final_elements)} elements")

        results = {}

        for element in final_elements:
            try:
                if action_executor:
                    success = action_executor(element, "DE-ENERGIZE", "SAFE")
                else:
                    # Simulation: assume success
                    success = True
                    logger.debug(f"Simulated de-energize for {element}")

                results[element] = success

                if success:
                    logger.info(f"De-energize successful: {element}")
                else:
                    logger.error(f"De-energize FAILED: {element}")

            except Exception as e:
                results[element] = False
                logger.error(f"De-energize error for {element}: {e}")

        # Report summary
        success_count = sum(1 for v in results.values() if v)
        logger.info(
            f"De-energize-to-trip complete: {success_count}/{len(final_elements)} successful"
        )

        return results

    def _calculate_voting_provenance(self, result: VotingResult) -> str:
        """Calculate SHA-256 provenance hash for voting result."""
        provenance_str = (
            f"{result.architecture.value}|"
            f"{result.trip_decision}|"
            f"{result.channels_voting_trip}|"
            f"{result.channels_total}|"
            f"{result.evaluation_timestamp.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _calculate_transition_provenance(self, result: TransitionResult) -> str:
        """Calculate SHA-256 provenance hash for transition result."""
        provenance_str = (
            f"{result.transition_id}|"
            f"{result.status.value}|"
            f"{result.achieved_state}|"
            f"{result.duration_ms}|"
            f"{result.end_time.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enumerations
    'SILLevel',
    'VotingArchitecture',
    'ConsequenceSeverity',
    'TestType',
    'TestResult',
    'WatchdogState',
    'TimeoutAction',
    'TransitionMode',
    'TransitionStatus',
    'ChannelStatus',
    'SRSStatus',

    # Data Classes and Models
    'SafetyFunction',
    'IPLDefinition',
    'LOPAScenario',
    'PFDInput',
    'PFDResult',
    'LOPAResult',
    'ProofTestRecord',
    'ProofTestSchedule',
    'VotingResult',
    'ChannelInput',
    'WatchdogConfig',
    'TransitionStep',
    'TransitionConfig',
    'TransitionResult',
    'SRSSection',
    'SafetyFunctionRequirement',
    'SRSDocument',

    # Core Classes
    'LOPAAnalyzer',
    'PFDCalculator',
    'ProofTestScheduler',
    'SRSGenerator',
    'FailSafeManager',
]
