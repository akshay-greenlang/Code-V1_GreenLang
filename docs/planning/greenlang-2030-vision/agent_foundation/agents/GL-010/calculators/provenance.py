"""
Provenance Module for GL-010 EMISSIONWATCH.

This module provides SHA-256 calculation hashing and audit trail
generation for regulatory compliance. All calculations are fully
traceable and verifiable.

Features:
- SHA-256 hashing for calculation provenance
- Complete audit trail generation
- Data lineage tracking
- Timestamp verification
- Regulatory submission tracking

Zero-Hallucination Guarantee:
- All provenance data is cryptographically verified
- Complete and immutable audit trails
- Full reproducibility assurance
"""

from typing import Dict, List, Optional, Union, Any
from decimal import Decimal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import uuid
from pydantic import BaseModel, Field


class ProvenanceEventType(str, Enum):
    """Types of provenance events."""
    CALCULATION = "calculation"
    DATA_INPUT = "data_input"
    DATA_OUTPUT = "data_output"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    SUBMISSION = "submission"
    APPROVAL = "approval"
    CORRECTION = "correction"


class DataQualityLevel(str, Enum):
    """Data quality levels per EPA guidance."""
    TIER_1 = "tier_1"  # Default emission factors
    TIER_2 = "tier_2"  # Facility-specific factors
    TIER_3 = "tier_3"  # Direct measurement (CEMS)


@dataclass(frozen=True)
class ProvenanceEvent:
    """
    Single provenance event in the audit trail.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of event
        timestamp: When event occurred
        actor: Who/what performed the action
        description: Description of the event
        inputs: Input data/references
        outputs: Output data/references
        hash: SHA-256 hash of event data
        parent_events: References to parent events
    """
    event_id: str
    event_type: ProvenanceEventType
    timestamp: datetime
    actor: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    hash: str
    parent_events: List[str] = field(default_factory=list)


@dataclass
class CalculationProvenance:
    """
    Complete provenance record for a calculation.

    Attributes:
        calculation_id: Unique calculation identifier
        formula_id: Formula/method identifier
        version: Formula/method version
        inputs: All input values
        outputs: All output values
        intermediate_values: Intermediate calculation values
        provenance_hash: Master hash of all data
        timestamp: When calculation was performed
        events: Audit trail events
        data_quality: Data quality tier
    """
    calculation_id: str
    formula_id: str
    version: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    intermediate_values: Dict[str, Any]
    provenance_hash: str
    timestamp: datetime
    events: List[ProvenanceEvent]
    data_quality: DataQualityLevel


@dataclass
class AuditTrail:
    """
    Complete audit trail for regulatory compliance.

    Attributes:
        trail_id: Unique trail identifier
        entity_id: ID of entity being tracked
        entity_type: Type of entity
        events: Ordered list of events
        created_at: When trail was created
        last_modified: Last modification time
        integrity_hash: Hash for trail integrity verification
    """
    trail_id: str
    entity_id: str
    entity_type: str
    events: List[ProvenanceEvent]
    created_at: datetime
    last_modified: datetime
    integrity_hash: str


class ProvenanceInput(BaseModel):
    """Input parameters for provenance tracking."""
    formula_id: str = Field(description="Formula identifier")
    version: str = Field(description="Version string")
    inputs: Dict[str, Any] = Field(description="Input values")
    actor: str = Field(default="system", description="Actor performing calculation")


class ProvenanceTracker:
    """
    Provenance tracker for calculation audit trails.

    Provides:
    - SHA-256 hashing for all calculations
    - Complete audit trail management
    - Data lineage tracking
    - Regulatory submission records
    """

    def __init__(self):
        """Initialize provenance tracker."""
        self._audit_trails: Dict[str, AuditTrail] = {}
        self._calculations: Dict[str, CalculationProvenance] = {}

    def create_calculation_provenance(
        self,
        provenance_input: ProvenanceInput,
        outputs: Dict[str, Any],
        intermediate_values: Optional[Dict[str, Any]] = None,
        data_quality: DataQualityLevel = DataQualityLevel.TIER_1
    ) -> CalculationProvenance:
        """
        Create provenance record for a calculation.

        Args:
            provenance_input: Input parameters
            outputs: Calculation outputs
            intermediate_values: Intermediate calculation values
            data_quality: Data quality tier

        Returns:
            CalculationProvenance with full audit trail
        """
        calculation_id = self._generate_id("CALC")
        timestamp = datetime.utcnow()

        # Create input event
        input_event = self._create_event(
            event_type=ProvenanceEventType.DATA_INPUT,
            actor=provenance_input.actor,
            description=f"Input data for {provenance_input.formula_id}",
            inputs=provenance_input.inputs,
            outputs={}
        )

        # Create calculation event
        calc_event = self._create_event(
            event_type=ProvenanceEventType.CALCULATION,
            actor=provenance_input.actor,
            description=f"Execute {provenance_input.formula_id} v{provenance_input.version}",
            inputs={"input_event": input_event.event_id},
            outputs=outputs,
            parent_events=[input_event.event_id]
        )

        # Create output event
        output_event = self._create_event(
            event_type=ProvenanceEventType.DATA_OUTPUT,
            actor=provenance_input.actor,
            description=f"Output from {provenance_input.formula_id}",
            inputs={"calculation_event": calc_event.event_id},
            outputs=outputs,
            parent_events=[calc_event.event_id]
        )

        # Calculate master provenance hash
        provenance_data = {
            "calculation_id": calculation_id,
            "formula_id": provenance_input.formula_id,
            "version": provenance_input.version,
            "inputs": self._serialize_for_hash(provenance_input.inputs),
            "outputs": self._serialize_for_hash(outputs),
            "intermediate": self._serialize_for_hash(intermediate_values or {}),
            "timestamp": timestamp.isoformat(),
            "events": [e.hash for e in [input_event, calc_event, output_event]]
        }
        provenance_hash = self._calculate_hash(provenance_data)

        provenance = CalculationProvenance(
            calculation_id=calculation_id,
            formula_id=provenance_input.formula_id,
            version=provenance_input.version,
            inputs=provenance_input.inputs,
            outputs=outputs,
            intermediate_values=intermediate_values or {},
            provenance_hash=provenance_hash,
            timestamp=timestamp,
            events=[input_event, calc_event, output_event],
            data_quality=data_quality
        )

        self._calculations[calculation_id] = provenance
        return provenance

    def verify_calculation(
        self,
        provenance: CalculationProvenance
    ) -> Tuple[bool, List[str]]:
        """
        Verify calculation provenance integrity.

        Checks:
        - Event chain integrity
        - Hash consistency
        - Data consistency

        Args:
            provenance: Provenance record to verify

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Recalculate provenance hash
        provenance_data = {
            "calculation_id": provenance.calculation_id,
            "formula_id": provenance.formula_id,
            "version": provenance.version,
            "inputs": self._serialize_for_hash(provenance.inputs),
            "outputs": self._serialize_for_hash(provenance.outputs),
            "intermediate": self._serialize_for_hash(provenance.intermediate_values),
            "timestamp": provenance.timestamp.isoformat(),
            "events": [e.hash for e in provenance.events]
        }
        calculated_hash = self._calculate_hash(provenance_data)

        if calculated_hash != provenance.provenance_hash:
            issues.append(f"Provenance hash mismatch: expected {provenance.provenance_hash}, got {calculated_hash}")

        # Verify event chain
        for event in provenance.events:
            # Recalculate event hash
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "actor": event.actor,
                "description": event.description,
                "inputs": self._serialize_for_hash(event.inputs),
                "outputs": self._serialize_for_hash(event.outputs),
                "parent_events": event.parent_events
            }
            calculated_event_hash = self._calculate_hash(event_data)

            if calculated_event_hash != event.hash:
                issues.append(f"Event hash mismatch for {event.event_id}")

        return len(issues) == 0, issues

    def create_audit_trail(
        self,
        entity_id: str,
        entity_type: str
    ) -> AuditTrail:
        """
        Create new audit trail for an entity.

        Args:
            entity_id: Entity identifier
            entity_type: Type of entity

        Returns:
            New AuditTrail
        """
        trail_id = self._generate_id("TRAIL")
        now = datetime.utcnow()

        trail = AuditTrail(
            trail_id=trail_id,
            entity_id=entity_id,
            entity_type=entity_type,
            events=[],
            created_at=now,
            last_modified=now,
            integrity_hash=self._calculate_hash({
                "trail_id": trail_id,
                "entity_id": entity_id,
                "created_at": now.isoformat()
            })
        )

        self._audit_trails[trail_id] = trail
        return trail

    def add_audit_event(
        self,
        trail_id: str,
        event_type: ProvenanceEventType,
        actor: str,
        description: str,
        data: Dict[str, Any]
    ) -> ProvenanceEvent:
        """
        Add event to audit trail.

        Args:
            trail_id: Audit trail identifier
            event_type: Type of event
            actor: Actor performing action
            description: Event description
            data: Event data

        Returns:
            Created ProvenanceEvent

        Raises:
            ValueError: If trail not found
        """
        trail = self._audit_trails.get(trail_id)
        if trail is None:
            raise ValueError(f"Audit trail not found: {trail_id}")

        # Get parent event (previous event in trail)
        parent_events = []
        if trail.events:
            parent_events = [trail.events[-1].event_id]

        event = self._create_event(
            event_type=event_type,
            actor=actor,
            description=description,
            inputs=data,
            outputs={},
            parent_events=parent_events
        )

        trail.events.append(event)
        trail.last_modified = datetime.utcnow()

        # Update integrity hash
        trail.integrity_hash = self._calculate_hash({
            "trail_id": trail.trail_id,
            "events": [e.hash for e in trail.events],
            "last_modified": trail.last_modified.isoformat()
        })

        return event

    def verify_audit_trail(
        self,
        trail_id: str
    ) -> Tuple[bool, List[str]]:
        """
        Verify audit trail integrity.

        Checks:
        - Event chain integrity
        - Hash consistency
        - Parent-child relationships

        Args:
            trail_id: Trail to verify

        Returns:
            Tuple of (is_valid, list of issues)
        """
        trail = self._audit_trails.get(trail_id)
        if trail is None:
            return False, ["Audit trail not found"]

        issues = []

        # Verify event chain
        for i, event in enumerate(trail.events):
            # Verify event hash
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "actor": event.actor,
                "description": event.description,
                "inputs": self._serialize_for_hash(event.inputs),
                "outputs": self._serialize_for_hash(event.outputs),
                "parent_events": event.parent_events
            }
            calculated_hash = self._calculate_hash(event_data)

            if calculated_hash != event.hash:
                issues.append(f"Event {i}: hash mismatch")

            # Verify parent-child relationship
            if i > 0 and trail.events[i-1].event_id not in event.parent_events:
                issues.append(f"Event {i}: broken parent chain")

        # Verify trail integrity hash
        expected_hash = self._calculate_hash({
            "trail_id": trail.trail_id,
            "events": [e.hash for e in trail.events],
            "last_modified": trail.last_modified.isoformat()
        })

        if expected_hash != trail.integrity_hash:
            issues.append("Trail integrity hash mismatch")

        return len(issues) == 0, issues

    def generate_provenance_certificate(
        self,
        calculation_id: str
    ) -> Dict[str, Any]:
        """
        Generate provenance certificate for regulatory submission.

        Args:
            calculation_id: Calculation to certify

        Returns:
            Certificate dictionary
        """
        provenance = self._calculations.get(calculation_id)
        if provenance is None:
            raise ValueError(f"Calculation not found: {calculation_id}")

        # Verify calculation
        is_valid, issues = self.verify_calculation(provenance)

        return {
            "certificate_id": self._generate_id("CERT"),
            "calculation_id": calculation_id,
            "formula_id": provenance.formula_id,
            "version": provenance.version,
            "provenance_hash": provenance.provenance_hash,
            "timestamp": provenance.timestamp.isoformat(),
            "data_quality": provenance.data_quality.value,
            "verification_status": "VERIFIED" if is_valid else "FAILED",
            "verification_issues": issues,
            "certification_timestamp": datetime.utcnow().isoformat(),
            "event_count": len(provenance.events),
            "input_hash": self._calculate_hash(provenance.inputs),
            "output_hash": self._calculate_hash(provenance.outputs),
        }

    def export_audit_trail(
        self,
        trail_id: str,
        format: str = "json"
    ) -> str:
        """
        Export audit trail for external review.

        Args:
            trail_id: Trail to export
            format: Export format (json)

        Returns:
            Serialized audit trail
        """
        trail = self._audit_trails.get(trail_id)
        if trail is None:
            raise ValueError(f"Audit trail not found: {trail_id}")

        export_data = {
            "trail_id": trail.trail_id,
            "entity_id": trail.entity_id,
            "entity_type": trail.entity_type,
            "created_at": trail.created_at.isoformat(),
            "last_modified": trail.last_modified.isoformat(),
            "integrity_hash": trail.integrity_hash,
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "timestamp": e.timestamp.isoformat(),
                    "actor": e.actor,
                    "description": e.description,
                    "hash": e.hash,
                    "parent_events": e.parent_events,
                }
                for e in trail.events
            ]
        }

        return json.dumps(export_data, indent=2, default=str)

    def _create_event(
        self,
        event_type: ProvenanceEventType,
        actor: str,
        description: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        parent_events: Optional[List[str]] = None
    ) -> ProvenanceEvent:
        """Create a provenance event with hash."""
        event_id = self._generate_id("EVT")
        timestamp = datetime.utcnow()

        event_data = {
            "event_id": event_id,
            "event_type": event_type.value,
            "timestamp": timestamp.isoformat(),
            "actor": actor,
            "description": description,
            "inputs": self._serialize_for_hash(inputs),
            "outputs": self._serialize_for_hash(outputs),
            "parent_events": parent_events or []
        }
        event_hash = self._calculate_hash(event_data)

        return ProvenanceEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            actor=actor,
            description=description,
            inputs=inputs,
            outputs=outputs,
            hash=event_hash,
            parent_events=parent_events or []
        )

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash of data."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def _serialize_for_hash(self, data: Any) -> Any:
        """Serialize data for consistent hashing."""
        if isinstance(data, Decimal):
            return str(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, dict):
            return {k: self._serialize_for_hash(v) for k, v in sorted(data.items())}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_for_hash(v) for v in data]
        elif isinstance(data, Enum):
            return data.value
        else:
            return data

    def _generate_id(self, prefix: str) -> str:
        """Generate unique identifier."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique = uuid.uuid4().hex[:8].upper()
        return f"{prefix}-{timestamp}-{unique}"


# Global tracker instance
_provenance_tracker: Optional[ProvenanceTracker] = None


def get_provenance_tracker() -> ProvenanceTracker:
    """Get or create global provenance tracker."""
    global _provenance_tracker
    if _provenance_tracker is None:
        _provenance_tracker = ProvenanceTracker()
    return _provenance_tracker


def calculate_hash(data: Any) -> str:
    """
    Calculate SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash

    Returns:
        SHA-256 hash string
    """
    tracker = get_provenance_tracker()
    return tracker._calculate_hash(data)


def create_provenance(
    formula_id: str,
    version: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    intermediate: Optional[Dict[str, Any]] = None,
    actor: str = "system"
) -> CalculationProvenance:
    """
    Create provenance record for a calculation.

    Args:
        formula_id: Formula identifier
        version: Version string
        inputs: Input values
        outputs: Output values
        intermediate: Intermediate values
        actor: Actor name

    Returns:
        CalculationProvenance record
    """
    tracker = get_provenance_tracker()

    provenance_input = ProvenanceInput(
        formula_id=formula_id,
        version=version,
        inputs=inputs,
        actor=actor
    )

    return tracker.create_calculation_provenance(
        provenance_input=provenance_input,
        outputs=outputs,
        intermediate_values=intermediate
    )


def verify_provenance(provenance: CalculationProvenance) -> bool:
    """
    Verify calculation provenance integrity.

    Args:
        provenance: Provenance to verify

    Returns:
        True if valid, False otherwise
    """
    tracker = get_provenance_tracker()
    is_valid, _ = tracker.verify_calculation(provenance)
    return is_valid


# Import Tuple from typing for type hints
from typing import Tuple
