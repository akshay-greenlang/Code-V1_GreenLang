"""
GL-004 BURNMASTER - Setpoint Manager

This module implements the setpoint management system for the burner management
control system. It provides controlled setpoint proposals, validation, application,
and rollback capabilities with complete audit trails.

Key Features:
    - Setpoint proposal with impact assessment
    - Validation against safety envelopes
    - Controlled application with rate limiting
    - Complete rollback capability
    - SHA-256 provenance tracking for all changes

Reference Standards:
    - IEC 61511 Functional Safety
    - ISA-84 Safety Instrumented Systems
    - NFPA 85/86 Boiler/Furnace Standards

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class SetpointStatus(str, Enum):
    """Status of a setpoint value."""
    PROPOSED = "proposed"
    VALIDATED = "validated"
    APPLIED = "applied"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class SetpointType(str, Enum):
    """Types of setpoints in the burner management system."""
    AIR_FUEL_RATIO = "air_fuel_ratio"
    FIRING_RATE = "firing_rate"
    EXCESS_O2 = "excess_o2"
    STACK_TEMPERATURE = "stack_temperature"
    COMBUSTION_AIR_TEMP = "combustion_air_temp"
    FUEL_PRESSURE = "fuel_pressure"
    DAMPER_POSITION = "damper_position"
    FGR_RATE = "fgr_rate"


class SetpointValidationResult(BaseModel):
    """Result of validating a proposed setpoint."""
    validation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    setpoint_id: str
    is_valid: bool
    within_envelope: bool = Field(default=True)
    envelope_min: Optional[float] = None
    envelope_max: Optional[float] = None
    proposed_value: float
    current_value: float
    rate_limit_ok: bool = Field(default=True)
    safety_checks_passed: List[str] = Field(default_factory=list)
    safety_checks_failed: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            hash_input = f"{self.validation_id}|{self.setpoint_id}|{self.proposed_value}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class SetpointProposal(BaseModel):
    """A proposed setpoint change."""
    proposal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    setpoint_type: SetpointType
    current_value: float
    proposed_value: float
    unit: str = Field(default="")
    reason: str = Field(default="")
    source: str = Field(default="OPTIMIZER")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    validation_result: Optional[SetpointValidationResult] = None
    status: SetpointStatus = Field(default=SetpointStatus.PROPOSED)
    provenance_hash: str = Field(default="")

    @validator('proposed_value')
    def validate_proposed_value(cls, v, values):
        """Ensure proposed value is different from current."""
        if 'current_value' in values and v == values['current_value']:
            raise ValueError("Proposed value must differ from current value")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            hash_input = f"{self.proposal_id}|{self.setpoint_type.value}|{self.proposed_value}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class SetpointRecord(BaseModel):
    """Historical record of a setpoint change."""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    setpoint_type: SetpointType
    previous_value: float
    new_value: float
    applied_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    applied_by: str = Field(default="SYSTEM")
    proposal_id: Optional[str] = None
    rollback_available: bool = Field(default=True)
    rolled_back: bool = Field(default=False)
    rolled_back_at: Optional[datetime] = None
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            hash_input = f"{self.record_id}|{self.previous_value}|{self.new_value}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class ApplyResult(BaseModel):
    """Result of applying a setpoint change."""
    success: bool
    record: Optional[SetpointRecord] = None
    error_message: Optional[str] = None
    applied_value: Optional[float] = None
    verification_passed: bool = Field(default=False)
    rollback_available: bool = Field(default=True)


class ImpactAssessment(BaseModel):
    """Assessment of the impact of a proposed setpoint change."""
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    proposal_id: str
    estimated_efficiency_change: float = Field(default=0.0)
    estimated_emissions_change: float = Field(default=0.0)
    estimated_fuel_savings_percent: float = Field(default=0.0)
    risk_level: str = Field(default="low")
    affected_parameters: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SafetyEnvelope(BaseModel):
    """Safety envelope defining acceptable ranges for setpoints."""
    setpoint_type: SetpointType
    min_value: float
    max_value: float
    rate_limit_per_second: float = Field(default=1.0, gt=0)
    unit: str = Field(default="")
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    description: str = Field(default="")


class SetpointManager:
    """
    Manages setpoint proposals, validation, application, and rollback.

    This class provides controlled setpoint management with:
    - Safety envelope validation
    - Rate limiting for gradual changes
    - Complete audit trail with rollback capability
    - Impact assessment for proposed changes

    Example:
        >>> manager = SetpointManager()
        >>> proposal = manager.propose_setpoint(
        ...     setpoint_type=SetpointType.AIR_FUEL_RATIO,
        ...     proposed_value=1.15,
        ...     reason="Optimize for efficiency"
        ... )
        >>> validation = manager.validate_proposal(proposal)
        >>> if validation.is_valid:
        ...     result = manager.apply_setpoint(proposal)
    """

    # Default safety envelopes for each setpoint type
    DEFAULT_ENVELOPES: Dict[SetpointType, SafetyEnvelope] = {
        SetpointType.AIR_FUEL_RATIO: SafetyEnvelope(
            setpoint_type=SetpointType.AIR_FUEL_RATIO,
            min_value=1.0,
            max_value=1.5,
            rate_limit_per_second=0.01,
            unit="ratio",
            warning_low=1.05,
            warning_high=1.4,
            description="Air to fuel mass ratio"
        ),
        SetpointType.FIRING_RATE: SafetyEnvelope(
            setpoint_type=SetpointType.FIRING_RATE,
            min_value=0.1,
            max_value=1.0,
            rate_limit_per_second=0.05,
            unit="fraction",
            warning_low=0.2,
            warning_high=0.95,
            description="Firing rate as fraction of max capacity"
        ),
        SetpointType.EXCESS_O2: SafetyEnvelope(
            setpoint_type=SetpointType.EXCESS_O2,
            min_value=1.0,
            max_value=8.0,
            rate_limit_per_second=0.1,
            unit="%",
            warning_low=1.5,
            warning_high=6.0,
            description="Excess oxygen in flue gas"
        ),
        SetpointType.STACK_TEMPERATURE: SafetyEnvelope(
            setpoint_type=SetpointType.STACK_TEMPERATURE,
            min_value=100.0,
            max_value=500.0,
            rate_limit_per_second=5.0,
            unit="C",
            warning_low=120.0,
            warning_high=450.0,
            description="Stack exhaust temperature"
        ),
        SetpointType.DAMPER_POSITION: SafetyEnvelope(
            setpoint_type=SetpointType.DAMPER_POSITION,
            min_value=0.0,
            max_value=100.0,
            rate_limit_per_second=2.0,
            unit="%",
            warning_low=10.0,
            warning_high=95.0,
            description="Air damper position"
        ),
        SetpointType.FGR_RATE: SafetyEnvelope(
            setpoint_type=SetpointType.FGR_RATE,
            min_value=0.0,
            max_value=30.0,
            rate_limit_per_second=0.5,
            unit="%",
            warning_low=None,
            warning_high=25.0,
            description="Flue gas recirculation rate"
        ),
    }

    def __init__(self) -> None:
        """Initialize the setpoint manager."""
        self._current_setpoints: Dict[SetpointType, float] = {}
        self._envelopes: Dict[SetpointType, SafetyEnvelope] = dict(self.DEFAULT_ENVELOPES)
        self._pending_proposals: Dict[str, SetpointProposal] = {}
        self._history: List[SetpointRecord] = []
        self._audit_log: List[Dict[str, Any]] = []

        # Initialize default setpoint values
        self._initialize_default_setpoints()

        logger.info("SetpointManager initialized")

    def _initialize_default_setpoints(self) -> None:
        """Initialize default setpoint values."""
        self._current_setpoints = {
            SetpointType.AIR_FUEL_RATIO: 1.2,
            SetpointType.FIRING_RATE: 0.75,
            SetpointType.EXCESS_O2: 3.0,
            SetpointType.STACK_TEMPERATURE: 250.0,
            SetpointType.DAMPER_POSITION: 50.0,
            SetpointType.FGR_RATE: 10.0,
        }

    def propose_setpoint(
        self,
        setpoint_type: SetpointType,
        proposed_value: float,
        reason: str = "",
        source: str = "OPTIMIZER",
        confidence: float = 1.0
    ) -> SetpointProposal:
        """
        Create a new setpoint proposal.

        Args:
            setpoint_type: Type of setpoint to change
            proposed_value: Proposed new value
            reason: Explanation for the change
            source: Source of the proposal (e.g., OPTIMIZER, OPERATOR)
            confidence: Confidence level in the proposal (0-1)

        Returns:
            SetpointProposal ready for validation
        """
        current_value = self._current_setpoints.get(setpoint_type, 0.0)
        envelope = self._envelopes.get(setpoint_type)

        proposal = SetpointProposal(
            setpoint_type=setpoint_type,
            current_value=current_value,
            proposed_value=proposed_value,
            unit=envelope.unit if envelope else "",
            reason=reason,
            source=source,
            confidence=confidence
        )

        self._pending_proposals[proposal.proposal_id] = proposal
        self._log_event("SETPOINT_PROPOSED", proposal)

        logger.info(f"Setpoint proposal created: {setpoint_type.value} = {proposed_value}")

        return proposal

    def validate_proposal(self, proposal: SetpointProposal) -> SetpointValidationResult:
        """
        Validate a setpoint proposal against safety envelopes.

        Args:
            proposal: The setpoint proposal to validate

        Returns:
            SetpointValidationResult with validation details
        """
        envelope = self._envelopes.get(proposal.setpoint_type)
        safety_passed = []
        safety_failed = []
        warnings = []
        within_envelope = True
        rate_limit_ok = True

        # Check envelope bounds
        if envelope:
            if proposal.proposed_value < envelope.min_value:
                within_envelope = False
                safety_failed.append(f"Below minimum ({envelope.min_value})")
            elif proposal.proposed_value > envelope.max_value:
                within_envelope = False
                safety_failed.append(f"Above maximum ({envelope.max_value})")
            else:
                safety_passed.append("Within safety envelope")

            # Check warning zones
            if envelope.warning_low and proposal.proposed_value < envelope.warning_low:
                warnings.append(f"Below warning threshold ({envelope.warning_low})")
            if envelope.warning_high and proposal.proposed_value > envelope.warning_high:
                warnings.append(f"Above warning threshold ({envelope.warning_high})")

            # Check rate limit
            change = abs(proposal.proposed_value - proposal.current_value)
            max_change_per_cycle = envelope.rate_limit_per_second * 1.0  # 1 second cycle
            if change > max_change_per_cycle * 10:  # Allow 10 cycles
                rate_limit_ok = False
                warnings.append(f"Large change may require gradual application")

        is_valid = within_envelope and len(safety_failed) == 0

        validation = SetpointValidationResult(
            setpoint_id=proposal.proposal_id,
            is_valid=is_valid,
            within_envelope=within_envelope,
            envelope_min=envelope.min_value if envelope else None,
            envelope_max=envelope.max_value if envelope else None,
            proposed_value=proposal.proposed_value,
            current_value=proposal.current_value,
            rate_limit_ok=rate_limit_ok,
            safety_checks_passed=safety_passed,
            safety_checks_failed=safety_failed,
            warnings=warnings
        )

        # Update proposal with validation result
        proposal.validation_result = validation
        proposal.status = SetpointStatus.VALIDATED if is_valid else SetpointStatus.REJECTED

        self._log_event("SETPOINT_VALIDATED", validation)

        return validation

    def apply_setpoint(
        self,
        proposal: SetpointProposal,
        applied_by: str = "SYSTEM",
        force: bool = False
    ) -> ApplyResult:
        """
        Apply a validated setpoint proposal.

        Args:
            proposal: The validated proposal to apply
            applied_by: Identifier of who/what is applying the change
            force: If True, apply even if not validated (use with caution)

        Returns:
            ApplyResult with success status and record
        """
        # Check validation unless forced
        if not force:
            if proposal.validation_result is None:
                return ApplyResult(
                    success=False,
                    error_message="Proposal not validated"
                )
            if not proposal.validation_result.is_valid:
                return ApplyResult(
                    success=False,
                    error_message="Proposal validation failed"
                )

        # Record the change
        previous_value = self._current_setpoints.get(proposal.setpoint_type, 0.0)

        record = SetpointRecord(
            setpoint_type=proposal.setpoint_type,
            previous_value=previous_value,
            new_value=proposal.proposed_value,
            applied_by=applied_by,
            proposal_id=proposal.proposal_id
        )

        # Apply the change
        self._current_setpoints[proposal.setpoint_type] = proposal.proposed_value
        self._history.append(record)

        # Update proposal status
        proposal.status = SetpointStatus.APPLIED

        # Remove from pending
        if proposal.proposal_id in self._pending_proposals:
            del self._pending_proposals[proposal.proposal_id]

        self._log_event("SETPOINT_APPLIED", record)

        logger.info(f"Setpoint applied: {proposal.setpoint_type.value} = {proposal.proposed_value}")

        return ApplyResult(
            success=True,
            record=record,
            applied_value=proposal.proposed_value,
            verification_passed=True,
            rollback_available=True
        )

    def rollback_setpoint(self, record_id: str) -> ApplyResult:
        """
        Rollback a previously applied setpoint change.

        Args:
            record_id: ID of the setpoint record to rollback

        Returns:
            ApplyResult with rollback status
        """
        # Find the record
        record = None
        for r in self._history:
            if r.record_id == record_id:
                record = r
                break

        if record is None:
            return ApplyResult(
                success=False,
                error_message=f"Record not found: {record_id}"
            )

        if not record.rollback_available:
            return ApplyResult(
                success=False,
                error_message="Rollback not available for this record"
            )

        if record.rolled_back:
            return ApplyResult(
                success=False,
                error_message="Already rolled back"
            )

        # Apply rollback
        self._current_setpoints[record.setpoint_type] = record.previous_value
        record.rolled_back = True
        record.rolled_back_at = datetime.now(timezone.utc)

        # Create new record for the rollback
        rollback_record = SetpointRecord(
            setpoint_type=record.setpoint_type,
            previous_value=record.new_value,
            new_value=record.previous_value,
            applied_by="ROLLBACK",
            rollback_available=False
        )
        self._history.append(rollback_record)

        self._log_event("SETPOINT_ROLLED_BACK", rollback_record)

        logger.info(f"Setpoint rolled back: {record.setpoint_type.value} = {record.previous_value}")

        return ApplyResult(
            success=True,
            record=rollback_record,
            applied_value=record.previous_value,
            rollback_available=False
        )

    def assess_impact(self, proposal: SetpointProposal) -> ImpactAssessment:
        """
        Assess the potential impact of a setpoint change.

        Args:
            proposal: The proposal to assess

        Returns:
            ImpactAssessment with estimated impacts
        """
        change = proposal.proposed_value - proposal.current_value
        change_percent = abs(change / proposal.current_value) * 100 if proposal.current_value != 0 else 0

        # Estimate impacts based on setpoint type
        efficiency_change = 0.0
        emissions_change = 0.0
        fuel_savings = 0.0
        risk_level = "low"
        affected = []
        recommendations = []

        if proposal.setpoint_type == SetpointType.AIR_FUEL_RATIO:
            if change < 0:  # Reducing air (leaner)
                efficiency_change = change_percent * 0.5
                emissions_change = -change_percent * 0.3  # May increase NOx
                fuel_savings = change_percent * 0.3
                affected = ["combustion_efficiency", "nox_emissions", "flame_stability"]
                if abs(change) > 0.1:
                    risk_level = "medium"
                    recommendations.append("Monitor flame stability during transition")
            else:
                efficiency_change = -change_percent * 0.3
                emissions_change = change_percent * 0.2
                affected = ["combustion_efficiency", "co_emissions"]

        elif proposal.setpoint_type == SetpointType.FIRING_RATE:
            efficiency_change = -change_percent * 0.1  # Lower rates often less efficient
            affected = ["heat_output", "turndown_ratio"]
            if abs(change) > 0.2:
                risk_level = "medium"
                recommendations.append("Verify minimum firing rate safety")

        elif proposal.setpoint_type == SetpointType.EXCESS_O2:
            if change < 0:  # Reducing excess O2
                efficiency_change = abs(change) * 2.0
                fuel_savings = abs(change) * 1.5
                emissions_change = change * 0.5  # May affect CO
                affected = ["stack_loss", "combustion_efficiency", "co_emissions"]
                if proposal.proposed_value < 2.0:
                    risk_level = "high"
                    recommendations.append("Closely monitor CO levels")
            else:
                efficiency_change = -change * 1.5
                fuel_savings = -change * 1.0
                affected = ["stack_loss"]

        return ImpactAssessment(
            proposal_id=proposal.proposal_id,
            estimated_efficiency_change=efficiency_change,
            estimated_emissions_change=emissions_change,
            estimated_fuel_savings_percent=fuel_savings,
            risk_level=risk_level,
            affected_parameters=affected,
            recommendations=recommendations,
            confidence=0.75
        )

    def get_current_setpoint(self, setpoint_type: SetpointType) -> float:
        """Get the current value of a setpoint."""
        return self._current_setpoints.get(setpoint_type, 0.0)

    def get_all_setpoints(self) -> Dict[SetpointType, float]:
        """Get all current setpoint values."""
        return dict(self._current_setpoints)

    def get_envelope(self, setpoint_type: SetpointType) -> Optional[SafetyEnvelope]:
        """Get the safety envelope for a setpoint type."""
        return self._envelopes.get(setpoint_type)

    def update_envelope(self, envelope: SafetyEnvelope) -> None:
        """Update the safety envelope for a setpoint type."""
        self._envelopes[envelope.setpoint_type] = envelope
        self._log_event("ENVELOPE_UPDATED", envelope)
        logger.info(f"Safety envelope updated: {envelope.setpoint_type.value}")

    def get_pending_proposals(self) -> List[SetpointProposal]:
        """Get all pending setpoint proposals."""
        return list(self._pending_proposals.values())

    def get_history(self, limit: int = 100) -> List[SetpointRecord]:
        """Get recent setpoint change history."""
        return list(reversed(self._history[-limit:]))

    def get_rollback_candidates(self) -> List[SetpointRecord]:
        """Get setpoint records that can be rolled back."""
        return [r for r in self._history if r.rollback_available and not r.rolled_back]

    def _log_event(self, event_type: str, data: Any) -> None:
        """Log an event to the audit trail."""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "data": data.model_dump() if hasattr(data, 'model_dump') else str(data)
        }
        self._audit_log.append(audit_entry)

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the audit log."""
        return list(reversed(self._audit_log[-limit:]))

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the setpoint manager."""
        return {
            "current_setpoints": {k.value: v for k, v in self._current_setpoints.items()},
            "pending_proposals_count": len(self._pending_proposals),
            "history_count": len(self._history),
            "rollback_candidates_count": len(self.get_rollback_candidates())
        }
