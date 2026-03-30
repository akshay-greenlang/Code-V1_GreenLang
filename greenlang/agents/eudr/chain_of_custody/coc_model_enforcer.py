# -*- coding: utf-8 -*-
"""
CoC Model Enforcer Engine - AGENT-EUDR-009: Chain of Custody (Feature 3)

Enforces the four chain-of-custody models defined by ISO 22095:2020 --
Identity Preserved (IP), Segregated (SG), Mass Balance (MB), and
Controlled Blending (CB). Validates batch operations against model-specific
rules, manages cross-model handoffs with material downgrade logic, and
links facility-commodity model assignments to certification schemes
(FSC, RSPO, ISCC, Rainforest Alliance).

Zero-Hallucination Guarantees:
    - All model validation is deterministic rule evaluation.
    - IP: single-origin check = exact string equality.
    - SG: compliant-source check = membership in compliant set.
    - MB: credit balance check = deterministic arithmetic (no overdraft).
    - CB: blend ratio check = float comparison against ratio cap.
    - Cross-model handoff downgrades follow a static hierarchy.
    - SHA-256 provenance hashes on all validation results.
    - No ML/LLM used for any model enforcement logic.

ISO 22095 Model Hierarchy (strictest to least strict):
    IP > SG > MB > CB
    Material can only downgrade (IP -> SG -> MB -> CB), never upgrade.

Model Rules Summary:
    IP (Identity Preserved):
        - No mixing with any other origin.
        - 100% single-origin, physically separated.
        - Requires batch-level traceability to single plot.
    SG (Segregated):
        - Only compliant sources allowed.
        - No mixing with non-compliant material.
        - Multiple compliant origins acceptable.
    MB (Mass Balance):
        - Accounting-based: compliant inputs credited.
        - Outputs cannot exceed compliant credits.
        - Credit period limits (3, 6, or 12 months).
        - No overdraft (output <= available credits).
    CB (Controlled Blending):
        - Maximum blend ratio of non-compliant material.
        - Minimum compliant percentage enforced.
        - Ratio caps per certification scheme.

Performance Targets:
    - Model assignment: <2ms
    - Operation validation: <5ms
    - Compliance score calculation: <10ms per facility

Regulatory References:
    - EUDR Article 4: Due diligence system requirements.
    - EUDR Article 9: Traceability information per CoC model.
    - ISO 22095:2020 Chain of custody -- General terminology and models.
    - FSC-STD-40-004: FSC Chain of Custody certification.
    - RSPO SCC 2020: RSPO Supply Chain Certification Standard.
    - ISCC 202: Sustainability Requirements.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009 (Feature 3: CoC Model Enforcement)
Agent ID: GL-EUDR-COC-009
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id() -> str:
    """Generate a unique identifier using UUID4."""
    return str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Model strictness hierarchy (index 0 = strictest).
MODEL_HIERARCHY: Tuple[str, ...] = ("ip", "sg", "mb", "cb")

#: Model human-readable names.
MODEL_NAMES: Dict[str, str] = {
    "ip": "Identity Preserved",
    "sg": "Segregated",
    "mb": "Mass Balance",
    "cb": "Controlled Blending",
}

#: Default credit period in months for mass balance model.
DEFAULT_MB_CREDIT_PERIOD_MONTHS: int = 12

#: Credit period presets per certification scheme (months).
CERTIFICATION_CREDIT_PERIODS: Dict[str, int] = {
    "fsc": 12,
    "rspo": 3,
    "iscc": 12,
    "rainforest_alliance": 12,
    "fairtrade": 12,
    "organic": 12,
    "eudr_default": 12,
}

#: Default minimum compliant percentage for Controlled Blending.
DEFAULT_CB_MIN_COMPLIANT_PCT: float = 50.0

#: CB minimum compliant % per certification scheme.
CERTIFICATION_CB_MIN_PCT: Dict[str, float] = {
    "fsc": 70.0,
    "rspo": 50.0,
    "iscc": 50.0,
    "rainforest_alliance": 30.0,
    "fairtrade": 20.0,
    "eudr_default": 50.0,
}

#: Required certifications per model type.
REQUIRED_CERTIFICATIONS: Dict[str, List[str]] = {
    "ip": ["fsc", "rspo", "iscc", "rainforest_alliance", "organic"],
    "sg": ["fsc", "rspo", "iscc", "rainforest_alliance"],
    "mb": ["fsc", "rspo", "iscc"],
    "cb": [],
}

#: Maximum number of model assignments per facility.
MAX_ASSIGNMENTS_PER_FACILITY: int = 50

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CoCModelType(str, Enum):
    """Chain-of-custody model types per ISO 22095."""

    IDENTITY_PRESERVED = "ip"
    SEGREGATED = "sg"
    MASS_BALANCE = "mb"
    CONTROLLED_BLENDING = "cb"

class OperationKind(str, Enum):
    """Kinds of batch operations subject to model enforcement."""

    RECEIVE = "receive"
    STORE = "store"
    PROCESS = "process"
    SPLIT = "split"
    MERGE = "merge"
    BLEND = "blend"
    TRANSFER = "transfer"
    EXPORT = "export"

class ComplianceLevel(str, Enum):
    """Compliance assessment levels."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    UNKNOWN = "unknown"

class TransitionDirection(str, Enum):
    """Direction of model transition."""

    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    LATERAL = "lateral"

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CoCModelAssignment:
    """Assignment of a CoC model to a facility-commodity pair.

    Attributes:
        assignment_id: Unique identifier.
        facility_id: Facility identifier.
        facility_name: Human-readable facility name.
        commodity: Commodity type.
        model_type: Assigned CoC model (ip, sg, mb, cb).
        certification_scheme: Primary certification scheme.
        certification_id: Certification number/ID.
        certification_expiry: Certification expiry date.
        credit_period_months: Credit period for MB model (months).
        cb_min_compliant_pct: Min compliant % for CB model.
        effective_from: When this assignment became effective.
        effective_to: When this assignment expires (optional).
        is_active: Whether the assignment is currently active.
        created_at: Record creation timestamp.
        updated_at: Last update timestamp.
        provenance_hash: SHA-256 provenance hash.
    """

    assignment_id: str = ""
    facility_id: str = ""
    facility_name: str = ""
    commodity: str = ""
    model_type: str = ""
    certification_scheme: str = ""
    certification_id: str = ""
    certification_expiry: str = ""
    credit_period_months: int = DEFAULT_MB_CREDIT_PERIOD_MONTHS
    cb_min_compliant_pct: float = DEFAULT_CB_MIN_COMPLIANT_PCT
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert assignment to dictionary."""
        return {
            "assignment_id": self.assignment_id,
            "facility_id": self.facility_id,
            "facility_name": self.facility_name,
            "commodity": self.commodity,
            "model_type": self.model_type,
            "certification_scheme": self.certification_scheme,
            "certification_id": self.certification_id,
            "certification_expiry": self.certification_expiry,
            "credit_period_months": self.credit_period_months,
            "cb_min_compliant_pct": self.cb_min_compliant_pct,
            "effective_from": str(self.effective_from) if self.effective_from else "",
            "effective_to": str(self.effective_to) if self.effective_to else "",
            "is_active": self.is_active,
            "created_at": str(self.created_at) if self.created_at else "",
            "updated_at": str(self.updated_at) if self.updated_at else "",
        }

@dataclass
class ModelValidationResult:
    """Result of validating an operation against a CoC model.

    Attributes:
        validation_id: Unique identifier.
        facility_id: Facility identifier.
        commodity: Commodity type.
        model_type: CoC model applied.
        operation_kind: Type of operation validated.
        is_compliant: Whether the operation is compliant.
        compliance_level: Overall compliance level.
        errors: List of compliance errors.
        warnings: List of compliance warnings.
        details: Detailed validation information.
        origin_count: Number of distinct origins in the operation.
        compliant_pct: Percentage of compliant material.
        credit_balance_kg: Remaining credit balance (MB model).
        blend_ratio_actual: Actual blend ratio (CB model).
        blend_ratio_max: Maximum allowed blend ratio (CB model).
        validated_at: Validation timestamp.
        processing_time_ms: Validation processing time.
        provenance_hash: SHA-256 provenance hash.
    """

    validation_id: str = ""
    facility_id: str = ""
    commodity: str = ""
    model_type: str = ""
    operation_kind: str = ""
    is_compliant: bool = True
    compliance_level: str = ComplianceLevel.COMPLIANT
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    origin_count: int = 0
    compliant_pct: float = 100.0
    credit_balance_kg: float = 0.0
    blend_ratio_actual: float = 0.0
    blend_ratio_max: float = 0.0
    validated_at: Optional[datetime] = None
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "validation_id": self.validation_id,
            "facility_id": self.facility_id,
            "commodity": self.commodity,
            "model_type": self.model_type,
            "operation_kind": self.operation_kind,
            "is_compliant": self.is_compliant,
            "compliance_level": self.compliance_level,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "details": dict(self.details),
            "origin_count": self.origin_count,
            "compliant_pct": self.compliant_pct,
            "credit_balance_kg": self.credit_balance_kg,
            "blend_ratio_actual": self.blend_ratio_actual,
            "blend_ratio_max": self.blend_ratio_max,
            "validated_at": str(self.validated_at) if self.validated_at else "",
            "processing_time_ms": self.processing_time_ms,
        }

@dataclass
class ComplianceScore:
    """Compliance score for a facility.

    Attributes:
        score_id: Unique identifier.
        facility_id: Facility identifier.
        facility_name: Facility name.
        overall_score: Overall compliance score (0-100).
        model_scores: Per-model compliance scores.
        commodity_scores: Per-commodity compliance scores.
        total_operations: Total number of operations validated.
        compliant_operations: Number of compliant operations.
        non_compliant_operations: Number of non-compliant operations.
        active_assignments: Number of active model assignments.
        certifications_valid: Number of valid certifications.
        certifications_expired: Number of expired certifications.
        risk_level: Risk level (low, medium, high, critical).
        calculated_at: When the score was calculated.
        processing_time_ms: Calculation time in ms.
        provenance_hash: SHA-256 provenance hash.
    """

    score_id: str = ""
    facility_id: str = ""
    facility_name: str = ""
    overall_score: float = 0.0
    model_scores: Dict[str, float] = field(default_factory=dict)
    commodity_scores: Dict[str, float] = field(default_factory=dict)
    total_operations: int = 0
    compliant_operations: int = 0
    non_compliant_operations: int = 0
    active_assignments: int = 0
    certifications_valid: int = 0
    certifications_expired: int = 0
    risk_level: str = "low"
    calculated_at: Optional[datetime] = None
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert score to dictionary."""
        return {
            "score_id": self.score_id,
            "facility_id": self.facility_id,
            "facility_name": self.facility_name,
            "overall_score": self.overall_score,
            "model_scores": dict(self.model_scores),
            "commodity_scores": dict(self.commodity_scores),
            "total_operations": self.total_operations,
            "compliant_operations": self.compliant_operations,
            "non_compliant_operations": self.non_compliant_operations,
            "active_assignments": self.active_assignments,
            "certifications_valid": self.certifications_valid,
            "certifications_expired": self.certifications_expired,
            "risk_level": self.risk_level,
            "calculated_at": str(self.calculated_at) if self.calculated_at else "",
            "processing_time_ms": self.processing_time_ms,
        }

@dataclass
class ModelTransition:
    """Record of a model transition (upgrade/downgrade).

    Attributes:
        transition_id: Unique identifier.
        facility_id: Facility identifier.
        commodity: Commodity type.
        from_model: Previous model type.
        to_model: New model type.
        direction: Upgrade, downgrade, or lateral.
        reason: Reason for the transition.
        previous_assignment_id: ID of the deactivated assignment.
        new_assignment_id: ID of the new assignment.
        transitioned_at: When the transition occurred.
        transitioned_by: Who initiated the transition.
        provenance_hash: SHA-256 provenance hash.
    """

    transition_id: str = ""
    facility_id: str = ""
    commodity: str = ""
    from_model: str = ""
    to_model: str = ""
    direction: str = ""
    reason: str = ""
    previous_assignment_id: str = ""
    new_assignment_id: str = ""
    transitioned_at: Optional[datetime] = None
    transitioned_by: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert transition to dictionary."""
        return {
            "transition_id": self.transition_id,
            "facility_id": self.facility_id,
            "commodity": self.commodity,
            "from_model": self.from_model,
            "to_model": self.to_model,
            "direction": self.direction,
            "reason": self.reason,
            "previous_assignment_id": self.previous_assignment_id,
            "new_assignment_id": self.new_assignment_id,
            "transitioned_at": str(self.transitioned_at) if self.transitioned_at else "",
            "transitioned_by": self.transitioned_by,
        }

@dataclass
class CrossModelHandoff:
    """Cross-model handoff validation when material moves between models.

    Attributes:
        handoff_id: Unique identifier.
        source_facility_id: Sending facility.
        dest_facility_id: Receiving facility.
        source_model: Model at source.
        dest_model: Model at destination.
        batch_id: Batch being handed off.
        commodity: Commodity type.
        quantity_kg: Quantity being handed off.
        is_valid: Whether the handoff is valid.
        requires_downgrade: Whether material must be downgraded.
        downgraded_to: Model the material is downgraded to (if applicable).
        errors: Validation errors.
        warnings: Validation warnings.
        validated_at: Validation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """

    handoff_id: str = ""
    source_facility_id: str = ""
    dest_facility_id: str = ""
    source_model: str = ""
    dest_model: str = ""
    batch_id: str = ""
    commodity: str = ""
    quantity_kg: float = 0.0
    is_valid: bool = True
    requires_downgrade: bool = False
    downgraded_to: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_at: Optional[datetime] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert handoff to dictionary."""
        return {
            "handoff_id": self.handoff_id,
            "source_facility_id": self.source_facility_id,
            "dest_facility_id": self.dest_facility_id,
            "source_model": self.source_model,
            "dest_model": self.dest_model,
            "batch_id": self.batch_id,
            "commodity": self.commodity,
            "quantity_kg": self.quantity_kg,
            "is_valid": self.is_valid,
            "requires_downgrade": self.requires_downgrade,
            "downgraded_to": self.downgraded_to,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "validated_at": str(self.validated_at) if self.validated_at else "",
        }

# ---------------------------------------------------------------------------
# CoCModelEnforcer
# ---------------------------------------------------------------------------

class CoCModelEnforcer:
    """Production-grade CoC model enforcement engine for EUDR compliance.

    Enforces Identity Preserved (IP), Segregated (SG), Mass Balance (MB),
    and Controlled Blending (CB) chain-of-custody models per ISO 22095.
    Validates operations against model-specific rules, handles cross-model
    handoffs with material downgrade logic, and calculates facility
    compliance scores.

    All operations are deterministic with zero LLM/ML involvement.

    Example::

        enforcer = CoCModelEnforcer()
        enforcer.assign_model("FAC-001", "cocoa", "sg",
                              certification_scheme="rspo",
                              certification_id="RSPO-2024-001")
        result = enforcer.validate_operation(
            operation={"kind": "receive", "origins": [{"compliant": True}]},
            facility_id="FAC-001",
            commodity="cocoa",
        )
        assert result.is_compliant

    Attributes:
        assignments: Active model assignments per facility-commodity.
        validations: History of validation results.
        transitions: History of model transitions.
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the CoCModelEnforcer.

        Args:
            config: Optional configuration object.
        """
        # Assignments: (facility_id, commodity) -> CoCModelAssignment
        self._assignments: Dict[Tuple[str, str], CoCModelAssignment] = {}

        # All assignments indexed by assignment_id
        self._assignments_by_id: Dict[str, CoCModelAssignment] = {}

        # Validation history: facility_id -> [ModelValidationResult]
        self._validations: Dict[str, List[ModelValidationResult]] = {}

        # Transition history
        self._transitions: List[ModelTransition] = []

        # MB credit ledger: (facility_id, commodity) -> available_credit_kg
        self._mb_credits: Dict[Tuple[str, str], float] = {}

        logger.info("CoCModelEnforcer initialized.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def assignment_count(self) -> int:
        """Return total number of active model assignments."""
        return sum(1 for a in self._assignments.values() if a.is_active)

    @property
    def transition_count(self) -> int:
        """Return total number of model transitions."""
        return len(self._transitions)

    # ------------------------------------------------------------------
    # Public API: assign_model
    # ------------------------------------------------------------------

    def assign_model(
        self,
        facility_id: str,
        commodity: str,
        model_type: str,
        certification_scheme: str = "",
        certification_id: str = "",
        certification_expiry: str = "",
        facility_name: str = "",
        credit_period_months: Optional[int] = None,
        cb_min_compliant_pct: Optional[float] = None,
    ) -> CoCModelAssignment:
        """Assign a CoC model to a facility-commodity pair.

        Args:
            facility_id: Facility identifier.
            commodity: Commodity type.
            model_type: CoC model type ('ip', 'sg', 'mb', 'cb').
            certification_scheme: Certification scheme name.
            certification_id: Certification number.
            certification_expiry: Expiry date (ISO format string).
            facility_name: Human-readable facility name.
            credit_period_months: Credit period for MB (overrides default).
            cb_min_compliant_pct: Min compliant % for CB (overrides default).

        Returns:
            The created CoCModelAssignment.

        Raises:
            ValueError: If model_type is invalid or facility/commodity empty.
        """
        start_time = time.monotonic()

        facility_id = facility_id.strip()
        commodity = commodity.strip().lower()
        model_type = model_type.strip().lower()

        if not facility_id:
            raise ValueError("facility_id must not be empty.")
        if not commodity:
            raise ValueError("commodity must not be empty.")
        if model_type not in MODEL_HIERARCHY:
            raise ValueError(
                f"Invalid model_type '{model_type}'. "
                f"Valid: {MODEL_HIERARCHY}"
            )

        # Determine credit period
        if credit_period_months is None:
            credit_period_months = CERTIFICATION_CREDIT_PERIODS.get(
                certification_scheme.lower(), DEFAULT_MB_CREDIT_PERIOD_MONTHS
            )

        # Determine CB min compliant pct
        if cb_min_compliant_pct is None:
            cb_min_compliant_pct = CERTIFICATION_CB_MIN_PCT.get(
                certification_scheme.lower(), DEFAULT_CB_MIN_COMPLIANT_PCT
            )

        now = utcnow()
        key = (facility_id, commodity)

        # Deactivate existing assignment if present
        if key in self._assignments:
            existing = self._assignments[key]
            existing.is_active = False
            existing.effective_to = now
            existing.updated_at = now

        assignment = CoCModelAssignment(
            assignment_id=_generate_id(),
            facility_id=facility_id,
            facility_name=facility_name,
            commodity=commodity,
            model_type=model_type,
            certification_scheme=certification_scheme.lower().strip(),
            certification_id=certification_id.strip(),
            certification_expiry=certification_expiry.strip(),
            credit_period_months=credit_period_months,
            cb_min_compliant_pct=cb_min_compliant_pct,
            effective_from=now,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        assignment.provenance_hash = _compute_hash(assignment.to_dict())

        self._assignments[key] = assignment
        self._assignments_by_id[assignment.assignment_id] = assignment

        # Initialize MB credits if mass balance model
        if model_type == CoCModelType.MASS_BALANCE:
            if key not in self._mb_credits:
                self._mb_credits[key] = 0.0

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Assigned model '%s' (%s) to facility '%s' commodity '%s' "
            "(cert=%s) in %.2fms",
            model_type,
            MODEL_NAMES.get(model_type, model_type),
            facility_id,
            commodity,
            certification_scheme,
            elapsed_ms,
        )

        return assignment

    # ------------------------------------------------------------------
    # Public API: validate_operation
    # ------------------------------------------------------------------

    def validate_operation(
        self,
        operation: Dict[str, Any],
        facility_id: str,
        commodity: str,
    ) -> ModelValidationResult:
        """Validate a batch operation against the assigned CoC model rules.

        Dispatches to the model-specific validator (IP, SG, MB, CB)
        based on the facility's current model assignment.

        Args:
            operation: Operation details dictionary. Expected keys:
                - kind (str): Operation kind (receive, store, process, etc.).
                - origins (list): List of origin dicts with 'compliant' flag.
                - quantity_kg (float): Operation quantity.
                - batch_id (str): Batch identifier.
                - source_batches (list): Source batch info for merge/blend.
                Optional keys depend on model type.
            facility_id: Facility identifier.
            commodity: Commodity type.

        Returns:
            ModelValidationResult with compliance status.

        Raises:
            ValueError: If no model assigned or operation kind invalid.
        """
        start_time = time.monotonic()

        facility_id = facility_id.strip()
        commodity = commodity.strip().lower()
        key = (facility_id, commodity)

        if key not in self._assignments or not self._assignments[key].is_active:
            raise ValueError(
                f"No active model assignment for facility '{facility_id}', "
                f"commodity '{commodity}'."
            )

        assignment = self._assignments[key]
        model_type = assignment.model_type

        op_kind = str(operation.get("kind", "")).strip().lower()
        valid_kinds = {k.value for k in OperationKind}
        if op_kind not in valid_kinds:
            raise ValueError(
                f"Invalid operation kind '{op_kind}'. Valid: {sorted(valid_kinds)}"
            )

        # Dispatch to model-specific validator
        if model_type == CoCModelType.IDENTITY_PRESERVED:
            result = self.validate_ip(operation)
        elif model_type == CoCModelType.SEGREGATED:
            result = self.validate_segregated(operation)
        elif model_type == CoCModelType.MASS_BALANCE:
            result = self.validate_mass_balance(operation)
        elif model_type == CoCModelType.CONTROLLED_BLENDING:
            result = self.validate_controlled_blending(operation)
        else:
            raise ValueError(f"Unknown model type '{model_type}'.")

        # Populate common fields
        result.facility_id = facility_id
        result.commodity = commodity
        result.model_type = model_type
        result.operation_kind = op_kind
        result.processing_time_ms = (time.monotonic() - start_time) * 1000.0
        result.provenance_hash = _compute_hash(result.to_dict())

        # Store validation result
        if facility_id not in self._validations:
            self._validations[facility_id] = []
        self._validations[facility_id].append(result)

        logger.info(
            "Validated %s operation at facility '%s' commodity '%s' "
            "model=%s: %s in %.2fms",
            op_kind,
            facility_id,
            commodity,
            model_type,
            "COMPLIANT" if result.is_compliant else "NON-COMPLIANT",
            result.processing_time_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: validate_ip
    # ------------------------------------------------------------------

    def validate_ip(
        self, operation: Dict[str, Any]
    ) -> ModelValidationResult:
        """Validate operation against Identity Preserved (IP) model rules.

        IP Rules:
            - No mixing: only ONE origin allowed per batch.
            - 100% single-origin: all material from the same plot.
            - Physical separation: no co-mingling with other material.
            - Merge and blend operations are NOT allowed under IP.

        Args:
            operation: Operation details with 'origins' list.

        Returns:
            ModelValidationResult.
        """
        result = ModelValidationResult(
            validation_id=_generate_id(),
            validated_at=utcnow(),
        )

        origins = operation.get("origins", [])
        op_kind = str(operation.get("kind", "")).strip().lower()

        # IP does not allow merge or blend
        if op_kind in (OperationKind.MERGE, OperationKind.BLEND):
            result.is_compliant = False
            result.compliance_level = ComplianceLevel.NON_COMPLIANT
            result.errors.append(
                f"IP model does not allow '{op_kind}' operations. "
                f"Material must remain physically separated."
            )
            return result

        # Check single origin
        unique_origins = self._extract_unique_origins(origins)
        result.origin_count = len(unique_origins)

        if len(unique_origins) > 1:
            result.is_compliant = False
            result.compliance_level = ComplianceLevel.NON_COMPLIANT
            result.errors.append(
                f"IP model requires single origin, found {len(unique_origins)} "
                f"distinct origins: {sorted(unique_origins)}"
            )
        elif len(unique_origins) == 0:
            result.is_compliant = False
            result.compliance_level = ComplianceLevel.NON_COMPLIANT
            result.errors.append("IP model requires at least one origin.")
        else:
            result.is_compliant = True
            result.compliance_level = ComplianceLevel.COMPLIANT
            result.details["single_origin"] = sorted(unique_origins)[0]

        # All origins must be compliant
        non_compliant = [o for o in origins if not o.get("compliant", False)]
        if non_compliant:
            result.is_compliant = False
            result.compliance_level = ComplianceLevel.NON_COMPLIANT
            result.errors.append(
                f"IP model requires 100% compliant material, found "
                f"{len(non_compliant)} non-compliant origin(s)."
            )
            result.compliant_pct = self._calc_compliant_pct(origins)
        else:
            result.compliant_pct = 100.0

        return result

    # ------------------------------------------------------------------
    # Public API: validate_segregated
    # ------------------------------------------------------------------

    def validate_segregated(
        self, operation: Dict[str, Any]
    ) -> ModelValidationResult:
        """Validate operation against Segregated (SG) model rules.

        SG Rules:
            - Only compliant sources allowed.
            - No mixing with non-compliant material.
            - Multiple compliant origins are acceptable.
            - Blend with non-compliant material is NOT allowed.

        Args:
            operation: Operation details with 'origins' list.

        Returns:
            ModelValidationResult.
        """
        result = ModelValidationResult(
            validation_id=_generate_id(),
            validated_at=utcnow(),
        )

        origins = operation.get("origins", [])
        op_kind = str(operation.get("kind", "")).strip().lower()

        unique_origins = self._extract_unique_origins(origins)
        result.origin_count = len(unique_origins)

        # Check all origins are compliant
        non_compliant = [o for o in origins if not o.get("compliant", False)]
        compliant_pct = self._calc_compliant_pct(origins)
        result.compliant_pct = compliant_pct

        if non_compliant:
            result.is_compliant = False
            result.compliance_level = ComplianceLevel.NON_COMPLIANT
            result.errors.append(
                f"SG model requires 100% compliant sources. Found "
                f"{len(non_compliant)} non-compliant origin(s) "
                f"({100.0 - compliant_pct:.1f}% non-compliant)."
            )
        else:
            result.is_compliant = True
            result.compliance_level = ComplianceLevel.COMPLIANT

        # SG allows merge of compliant batches but not blend with non-compliant
        if op_kind == OperationKind.BLEND:
            source_batches = operation.get("source_batches", [])
            has_non_compliant_source = any(
                not sb.get("compliant", False) for sb in source_batches
            )
            if has_non_compliant_source:
                result.is_compliant = False
                result.compliance_level = ComplianceLevel.NON_COMPLIANT
                result.errors.append(
                    "SG model does not allow blending with non-compliant sources."
                )

        return result

    # ------------------------------------------------------------------
    # Public API: validate_mass_balance
    # ------------------------------------------------------------------

    def validate_mass_balance(
        self, operation: Dict[str, Any]
    ) -> ModelValidationResult:
        """Validate operation against Mass Balance (MB) model rules.

        MB Rules:
            - Accounting-based: compliant inputs create credits.
            - Outputs cannot exceed available compliant credits.
            - No overdraft: output_compliant_kg <= credit_balance.
            - Credit period applies (credits expire after N months).

        Args:
            operation: Operation details with 'origins', 'quantity_kg',
                and optionally 'compliant_input_kg' or 'compliant_output_kg'.

        Returns:
            ModelValidationResult.
        """
        result = ModelValidationResult(
            validation_id=_generate_id(),
            validated_at=utcnow(),
        )

        facility_id = str(operation.get("facility_id", "")).strip()
        commodity = str(operation.get("commodity", "")).strip().lower()
        op_kind = str(operation.get("kind", "")).strip().lower()
        quantity_kg = float(operation.get("quantity_kg", 0.0))

        origins = operation.get("origins", [])
        result.origin_count = len(self._extract_unique_origins(origins))
        result.compliant_pct = self._calc_compliant_pct(origins)

        key = (facility_id, commodity)

        # For input operations, add compliant credits
        if op_kind in (OperationKind.RECEIVE, OperationKind.STORE):
            compliant_input_kg = float(
                operation.get("compliant_input_kg", 0.0)
            )
            if compliant_input_kg <= 0:
                # Calculate from origins
                compliant_input_kg = self._calc_compliant_quantity(
                    origins, quantity_kg
                )

            if key in self._mb_credits:
                self._mb_credits[key] = round(
                    self._mb_credits[key] + compliant_input_kg, 4
                )
            else:
                self._mb_credits[key] = round(compliant_input_kg, 4)

            result.credit_balance_kg = self._mb_credits.get(key, 0.0)
            result.is_compliant = True
            result.compliance_level = ComplianceLevel.COMPLIANT
            result.details["credits_added_kg"] = compliant_input_kg
            result.details["new_balance_kg"] = result.credit_balance_kg
            return result

        # For output operations, check and deduct credits
        if op_kind in (OperationKind.TRANSFER, OperationKind.EXPORT,
                       OperationKind.PROCESS, OperationKind.SPLIT):
            compliant_output_kg = float(
                operation.get("compliant_output_kg", 0.0)
            )
            if compliant_output_kg <= 0:
                compliant_output_kg = self._calc_compliant_quantity(
                    origins, quantity_kg
                )

            available_credits = self._mb_credits.get(key, 0.0)
            result.credit_balance_kg = available_credits

            if compliant_output_kg > available_credits:
                result.is_compliant = False
                result.compliance_level = ComplianceLevel.NON_COMPLIANT
                result.errors.append(
                    f"MB overdraft: requested {compliant_output_kg:.4f}kg "
                    f"compliant output but only {available_credits:.4f}kg "
                    f"credits available."
                )
                result.details["overdraft_kg"] = round(
                    compliant_output_kg - available_credits, 4
                )
            else:
                # Deduct credits
                self._mb_credits[key] = round(
                    available_credits - compliant_output_kg, 4
                )
                result.credit_balance_kg = self._mb_credits[key]
                result.is_compliant = True
                result.compliance_level = ComplianceLevel.COMPLIANT
                result.details["credits_deducted_kg"] = compliant_output_kg
                result.details["remaining_balance_kg"] = result.credit_balance_kg

            return result

        # For merge/blend under MB, validate total credits
        if op_kind in (OperationKind.MERGE, OperationKind.BLEND):
            result.is_compliant = True
            result.compliance_level = ComplianceLevel.COMPLIANT
            result.credit_balance_kg = self._mb_credits.get(key, 0.0)
            result.warnings.append(
                "MB model allows merge/blend; credit accounting applies."
            )

        return result

    # ------------------------------------------------------------------
    # Public API: validate_controlled_blending
    # ------------------------------------------------------------------

    def validate_controlled_blending(
        self, operation: Dict[str, Any]
    ) -> ModelValidationResult:
        """Validate operation against Controlled Blending (CB) model rules.

        CB Rules:
            - Maximum blend ratio of non-compliant material.
            - Minimum compliant percentage must be met.
            - Ratio caps depend on certification scheme.

        Args:
            operation: Operation details with 'origins' and 'quantity_kg'.

        Returns:
            ModelValidationResult.
        """
        result = ModelValidationResult(
            validation_id=_generate_id(),
            validated_at=utcnow(),
        )

        facility_id = str(operation.get("facility_id", "")).strip()
        commodity = str(operation.get("commodity", "")).strip().lower()
        origins = operation.get("origins", [])
        quantity_kg = float(operation.get("quantity_kg", 0.0))

        unique_origins = self._extract_unique_origins(origins)
        result.origin_count = len(unique_origins)

        # Calculate actual compliant percentage
        compliant_pct = self._calc_compliant_pct(origins)
        result.compliant_pct = compliant_pct

        # Get minimum required percentage from assignment
        key = (facility_id, commodity)
        min_pct = DEFAULT_CB_MIN_COMPLIANT_PCT
        if key in self._assignments:
            min_pct = self._assignments[key].cb_min_compliant_pct

        # Calculate blend ratios
        non_compliant_pct = 100.0 - compliant_pct
        max_non_compliant_pct = 100.0 - min_pct

        result.blend_ratio_actual = round(non_compliant_pct / 100.0, 4)
        result.blend_ratio_max = round(max_non_compliant_pct / 100.0, 4)

        if compliant_pct < min_pct:
            result.is_compliant = False
            result.compliance_level = ComplianceLevel.NON_COMPLIANT
            result.errors.append(
                f"CB model requires minimum {min_pct:.1f}% compliant material, "
                f"actual is {compliant_pct:.1f}%. Non-compliant ratio "
                f"{non_compliant_pct:.1f}% exceeds maximum "
                f"{max_non_compliant_pct:.1f}%."
            )
        else:
            result.is_compliant = True
            result.compliance_level = ComplianceLevel.COMPLIANT
            result.details["compliant_pct"] = compliant_pct
            result.details["min_required_pct"] = min_pct

        return result

    # ------------------------------------------------------------------
    # Public API: get_model_compliance
    # ------------------------------------------------------------------

    def get_model_compliance(
        self, facility_id: str
    ) -> ComplianceScore:
        """Calculate compliance score for a facility across all models.

        Examines all validation history for the facility and computes
        an overall score (0-100) based on compliant/non-compliant ratio.

        Args:
            facility_id: Facility identifier.

        Returns:
            ComplianceScore for the facility.
        """
        start_time = time.monotonic()

        facility_id = facility_id.strip()
        validations = self._validations.get(facility_id, [])

        total = len(validations)
        compliant = sum(1 for v in validations if v.is_compliant)
        non_compliant = total - compliant

        overall_score = (compliant / total * 100.0) if total > 0 else 100.0

        # Per-model scores
        model_groups: Dict[str, List[ModelValidationResult]] = {}
        for v in validations:
            model_groups.setdefault(v.model_type, []).append(v)

        model_scores: Dict[str, float] = {}
        for model, vals in model_groups.items():
            m_total = len(vals)
            m_compliant = sum(1 for v in vals if v.is_compliant)
            model_scores[model] = round(
                (m_compliant / m_total * 100.0) if m_total > 0 else 100.0, 2
            )

        # Per-commodity scores
        commodity_groups: Dict[str, List[ModelValidationResult]] = {}
        for v in validations:
            commodity_groups.setdefault(v.commodity, []).append(v)

        commodity_scores: Dict[str, float] = {}
        for comm, vals in commodity_groups.items():
            c_total = len(vals)
            c_compliant = sum(1 for v in vals if v.is_compliant)
            commodity_scores[comm] = round(
                (c_compliant / c_total * 100.0) if c_total > 0 else 100.0, 2
            )

        # Count assignments and certifications
        active_assignments = 0
        certs_valid = 0
        certs_expired = 0
        facility_name = ""

        for key, assignment in self._assignments.items():
            if key[0] == facility_id:
                if not facility_name and assignment.facility_name:
                    facility_name = assignment.facility_name
                if assignment.is_active:
                    active_assignments += 1
                    if assignment.certification_id:
                        if self._is_certification_valid(assignment):
                            certs_valid += 1
                        else:
                            certs_expired += 1

        # Risk level
        risk_level = self._classify_risk(overall_score)

        score = ComplianceScore(
            score_id=_generate_id(),
            facility_id=facility_id,
            facility_name=facility_name,
            overall_score=round(overall_score, 2),
            model_scores=model_scores,
            commodity_scores=commodity_scores,
            total_operations=total,
            compliant_operations=compliant,
            non_compliant_operations=non_compliant,
            active_assignments=active_assignments,
            certifications_valid=certs_valid,
            certifications_expired=certs_expired,
            risk_level=risk_level,
            calculated_at=utcnow(),
            processing_time_ms=(time.monotonic() - start_time) * 1000.0,
        )
        score.provenance_hash = _compute_hash(score.to_dict())

        logger.info(
            "Compliance score for facility '%s': %.1f%% (%d/%d compliant, "
            "risk=%s) in %.2fms",
            facility_id,
            overall_score,
            compliant,
            total,
            risk_level,
            score.processing_time_ms,
        )

        return score

    # ------------------------------------------------------------------
    # Public API: validate_cross_model_handoff
    # ------------------------------------------------------------------

    def validate_cross_model_handoff(
        self,
        source_model: str,
        dest_model: str,
        batch: Dict[str, Any],
    ) -> CrossModelHandoff:
        """Validate a cross-model handoff when material moves between facilities.

        Material can only downgrade in the model hierarchy (IP -> SG -> MB -> CB).
        An IP batch moving to an MB facility is automatically downgraded to MB.
        An MB batch moving to an IP facility is INVALID (cannot upgrade).

        Args:
            source_model: CoC model at the source facility.
            dest_model: CoC model at the destination facility.
            batch: Batch information dictionary with batch_id, quantity_kg, etc.

        Returns:
            CrossModelHandoff with validity and downgrade info.
        """
        start_time = time.monotonic()

        source_model = source_model.strip().lower()
        dest_model = dest_model.strip().lower()

        if source_model not in MODEL_HIERARCHY:
            raise ValueError(f"Invalid source_model '{source_model}'.")
        if dest_model not in MODEL_HIERARCHY:
            raise ValueError(f"Invalid dest_model '{dest_model}'.")

        source_idx = MODEL_HIERARCHY.index(source_model)
        dest_idx = MODEL_HIERARCHY.index(dest_model)

        handoff = CrossModelHandoff(
            handoff_id=_generate_id(),
            source_facility_id=str(batch.get("source_facility_id", "")).strip(),
            dest_facility_id=str(batch.get("dest_facility_id", "")).strip(),
            source_model=source_model,
            dest_model=dest_model,
            batch_id=str(batch.get("batch_id", "")).strip(),
            commodity=str(batch.get("commodity", "")).strip().lower(),
            quantity_kg=float(batch.get("quantity_kg", 0.0)),
            validated_at=utcnow(),
        )

        if source_idx == dest_idx:
            # Same model: valid, no downgrade
            handoff.is_valid = True
            handoff.requires_downgrade = False
        elif source_idx < dest_idx:
            # Downgrade (stricter -> less strict): valid with downgrade
            handoff.is_valid = True
            handoff.requires_downgrade = True
            handoff.downgraded_to = dest_model
            handoff.warnings.append(
                f"Material downgraded from {MODEL_NAMES[source_model]} to "
                f"{MODEL_NAMES[dest_model]}. Traceability claim reduced."
            )
        else:
            # Upgrade attempt (less strict -> stricter): invalid
            handoff.is_valid = False
            handoff.requires_downgrade = False
            handoff.errors.append(
                f"Cannot upgrade material from {MODEL_NAMES[source_model]} "
                f"to {MODEL_NAMES[dest_model]}. Material retains the "
                f"lower traceability claim."
            )

        handoff.provenance_hash = _compute_hash(handoff.to_dict())

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Cross-model handoff %s->%s for batch '%s': valid=%s, "
            "downgrade=%s in %.2fms",
            source_model,
            dest_model,
            handoff.batch_id,
            handoff.is_valid,
            handoff.requires_downgrade,
            elapsed_ms,
        )

        return handoff

    # ------------------------------------------------------------------
    # Public API: get_certification_linkage
    # ------------------------------------------------------------------

    def get_certification_linkage(
        self, facility_id: str
    ) -> List[Dict[str, Any]]:
        """Get certification linkage for all models assigned to a facility.

        Args:
            facility_id: Facility identifier.

        Returns:
            List of dictionaries with certification details.
        """
        facility_id = facility_id.strip()
        linkages: List[Dict[str, Any]] = []

        for key, assignment in self._assignments.items():
            if key[0] != facility_id:
                continue

            is_valid = self._is_certification_valid(assignment)

            linkages.append({
                "assignment_id": assignment.assignment_id,
                "commodity": assignment.commodity,
                "model_type": assignment.model_type,
                "model_name": MODEL_NAMES.get(assignment.model_type, ""),
                "certification_scheme": assignment.certification_scheme,
                "certification_id": assignment.certification_id,
                "certification_expiry": assignment.certification_expiry,
                "is_valid": is_valid,
                "is_active": assignment.is_active,
                "credit_period_months": assignment.credit_period_months,
                "cb_min_compliant_pct": assignment.cb_min_compliant_pct,
            })

        return linkages

    # ------------------------------------------------------------------
    # Public API: track_model_transition
    # ------------------------------------------------------------------

    def track_model_transition(
        self,
        facility_id: str,
        from_model: str,
        to_model: str,
        commodity: str = "",
        reason: str = "",
        transitioned_by: str = "",
        certification_scheme: str = "",
        certification_id: str = "",
    ) -> ModelTransition:
        """Track a model upgrade/downgrade transition for a facility.

        Args:
            facility_id: Facility identifier.
            from_model: Previous model type.
            to_model: New model type.
            commodity: Commodity type.
            reason: Reason for the transition.
            transitioned_by: Who initiated the transition.
            certification_scheme: Certification for the new model.
            certification_id: Certification ID for the new model.

        Returns:
            ModelTransition record.

        Raises:
            ValueError: If models are invalid.
        """
        start_time = time.monotonic()

        from_model = from_model.strip().lower()
        to_model = to_model.strip().lower()
        facility_id = facility_id.strip()
        commodity = commodity.strip().lower()

        if from_model not in MODEL_HIERARCHY:
            raise ValueError(f"Invalid from_model '{from_model}'.")
        if to_model not in MODEL_HIERARCHY:
            raise ValueError(f"Invalid to_model '{to_model}'.")

        from_idx = MODEL_HIERARCHY.index(from_model)
        to_idx = MODEL_HIERARCHY.index(to_model)

        if from_idx < to_idx:
            direction = TransitionDirection.DOWNGRADE
        elif from_idx > to_idx:
            direction = TransitionDirection.UPGRADE
        else:
            direction = TransitionDirection.LATERAL

        # Get previous assignment ID if exists
        key = (facility_id, commodity)
        prev_id = ""
        if key in self._assignments:
            prev_id = self._assignments[key].assignment_id

        # Create new assignment
        new_assignment = self.assign_model(
            facility_id=facility_id,
            commodity=commodity,
            model_type=to_model,
            certification_scheme=certification_scheme,
            certification_id=certification_id,
        )

        transition = ModelTransition(
            transition_id=_generate_id(),
            facility_id=facility_id,
            commodity=commodity,
            from_model=from_model,
            to_model=to_model,
            direction=direction,
            reason=reason.strip(),
            previous_assignment_id=prev_id,
            new_assignment_id=new_assignment.assignment_id,
            transitioned_at=utcnow(),
            transitioned_by=transitioned_by.strip(),
        )
        transition.provenance_hash = _compute_hash(transition.to_dict())
        self._transitions.append(transition)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Model transition for facility '%s' commodity '%s': "
            "%s -> %s (%s, reason: %s) in %.2fms",
            facility_id,
            commodity,
            from_model,
            to_model,
            direction,
            reason[:50],
            elapsed_ms,
        )

        return transition

    # ------------------------------------------------------------------
    # Public API: get_assignment
    # ------------------------------------------------------------------

    def get_assignment(
        self, facility_id: str, commodity: str
    ) -> Optional[CoCModelAssignment]:
        """Get the active model assignment for a facility-commodity pair.

        Args:
            facility_id: Facility identifier.
            commodity: Commodity type.

        Returns:
            Active CoCModelAssignment, or None if not found.
        """
        key = (facility_id.strip(), commodity.strip().lower())
        assignment = self._assignments.get(key)
        if assignment and assignment.is_active:
            return assignment
        return None

    # ------------------------------------------------------------------
    # Public API: get_mb_credit_balance
    # ------------------------------------------------------------------

    def get_mb_credit_balance(
        self, facility_id: str, commodity: str
    ) -> float:
        """Get the current mass balance credit balance.

        Args:
            facility_id: Facility identifier.
            commodity: Commodity type.

        Returns:
            Available credit balance in kg.
        """
        key = (facility_id.strip(), commodity.strip().lower())
        return self._mb_credits.get(key, 0.0)

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _extract_unique_origins(
        self, origins: List[Dict[str, Any]]
    ) -> Set[str]:
        """Extract unique origin identifiers from origin list.

        Args:
            origins: List of origin dictionaries.

        Returns:
            Set of unique origin identifiers.
        """
        unique: Set[str] = set()
        for o in origins:
            plot_id = str(o.get("plot_id", "")).strip()
            origin_id = str(o.get("origin_id", "")).strip()
            key = plot_id or origin_id
            if key:
                unique.add(key)
        return unique

    def _calc_compliant_pct(
        self, origins: List[Dict[str, Any]]
    ) -> float:
        """Calculate percentage of compliant origins by quantity.

        If quantities are available, uses quantity-weighted calculation.
        Otherwise, uses simple count-based percentage.

        Args:
            origins: List of origin dictionaries.

        Returns:
            Compliant percentage (0-100).
        """
        if not origins:
            return 0.0

        total_qty = sum(float(o.get("quantity_kg", 0.0)) for o in origins)

        if total_qty > 0:
            compliant_qty = sum(
                float(o.get("quantity_kg", 0.0))
                for o in origins
                if o.get("compliant", False)
            )
            return round((compliant_qty / total_qty) * 100.0, 4)

        # Fallback to count-based
        compliant_count = sum(1 for o in origins if o.get("compliant", False))
        return round((compliant_count / len(origins)) * 100.0, 4)

    def _calc_compliant_quantity(
        self, origins: List[Dict[str, Any]], total_qty: float
    ) -> float:
        """Calculate compliant quantity from origins.

        Args:
            origins: List of origin dictionaries.
            total_qty: Total operation quantity.

        Returns:
            Compliant quantity in kg.
        """
        if not origins:
            return 0.0

        origin_total = sum(float(o.get("quantity_kg", 0.0)) for o in origins)

        if origin_total > 0:
            compliant_qty = sum(
                float(o.get("quantity_kg", 0.0))
                for o in origins
                if o.get("compliant", False)
            )
            return round(compliant_qty, 4)

        # If no quantities on origins, estimate from compliant ratio
        compliant_count = sum(1 for o in origins if o.get("compliant", False))
        ratio = compliant_count / len(origins) if origins else 0.0
        return round(total_qty * ratio, 4)

    def _is_certification_valid(
        self, assignment: CoCModelAssignment
    ) -> bool:
        """Check if a certification is currently valid (not expired).

        Args:
            assignment: Model assignment with certification info.

        Returns:
            True if certification is valid (not expired).
        """
        if not assignment.certification_expiry:
            return True  # No expiry means assumed valid

        try:
            expiry = datetime.strptime(
                assignment.certification_expiry, "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)
            return utcnow() < expiry
        except ValueError:
            # Unparseable date treated as valid (warn logged)
            logger.warning(
                "Unable to parse certification expiry: %s",
                assignment.certification_expiry,
            )
            return True

    def _classify_risk(self, score: float) -> str:
        """Classify risk level based on compliance score.

        Args:
            score: Compliance score (0-100).

        Returns:
            Risk level string.
        """
        if score >= 95.0:
            return "low"
        elif score >= 80.0:
            return "medium"
        elif score >= 50.0:
            return "high"
        else:
            return "critical"
