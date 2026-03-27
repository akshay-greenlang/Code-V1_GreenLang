"""
PACK-050 GHG Consolidation Pack - Consolidation Audit Engine
====================================================================

Provides a complete audit trail for the GHG consolidation process.
Tracks every consolidation step, performs reconciliation between
bottom-up and top-down totals, validates completeness, manages
sign-off workflows, and generates assurance-ready documentation.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 7): Managing
      inventory quality - verification and audit trail.
    - ISO 14064-1:2018 (Clause 9): Documentation and retention
      requirements for GHG inventories.
    - ISO 14064-3:2019: Requirements for validation and
      verification of GHG assertions.
    - ISAE 3410: Assurance engagements on GHG statements -
      documentation requirements.
    - ESRS E1-6: Assurance readiness for GHG disclosures.
    - SOC 2 Type II: Completeness and accuracy controls.

Calculation Methodology:
    Reconciliation Variance:
        variance = bottom_up_total - top_down_total
        variance_pct = (variance / top_down_total) * 100
        is_material = abs(variance_pct) > materiality_threshold

    Completeness Score:
        completeness = entities_reported / entities_in_boundary * 100

Capabilities:
    - Track every consolidation step (data receipt, equity
      adjustment, elimination, manual adjustment)
    - Bottom-up vs top-down reconciliation
    - Variance analysis with materiality assessment
    - Completeness checks (all entities, all scopes)
    - Sign-off tracking (entity-level and group-level)
    - Assurance-ready documentation package generation
    - Audit finding recording and resolution tracking
    - Materiality assessment for variances

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-050 GHG Consolidation
Engine:  10 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash, excluding volatile fields."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("created_at", "updated_at", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert any value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Divide safely, returning *default* when denominator is zero."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round2(value: Any) -> Decimal:
    """Round a value to two decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AuditStepType(str, Enum):
    """Types of consolidation steps tracked in the audit trail."""
    DATA_RECEIPT = "DATA_RECEIPT"
    DATA_VALIDATION = "DATA_VALIDATION"
    EQUITY_ADJUSTMENT = "EQUITY_ADJUSTMENT"
    INTERCOMPANY_ELIMINATION = "INTERCOMPANY_ELIMINATION"
    MANUAL_ADJUSTMENT = "MANUAL_ADJUSTMENT"
    SCOPE_RECLASSIFICATION = "SCOPE_RECLASSIFICATION"
    BASE_YEAR_RESTATEMENT = "BASE_YEAR_RESTATEMENT"
    RECONCILIATION = "RECONCILIATION"
    COMPLETENESS_CHECK = "COMPLETENESS_CHECK"
    SIGN_OFF = "SIGN_OFF"
    REPORT_GENERATION = "REPORT_GENERATION"
    ASSURANCE_PACKAGE = "ASSURANCE_PACKAGE"


class FindingSeverity(str, Enum):
    """Severity levels for audit findings."""
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    OBSERVATION = "OBSERVATION"
    IMPROVEMENT = "IMPROVEMENT"


class FindingStatus(str, Enum):
    """Status of an audit finding."""
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    ACCEPTED = "ACCEPTED"


class SignOffLevel(str, Enum):
    """Level at which sign-off is performed."""
    ENTITY = "ENTITY"
    REGION = "REGION"
    BUSINESS_UNIT = "BUSINESS_UNIT"
    GROUP = "GROUP"
    EXTERNAL_ASSURANCE = "EXTERNAL_ASSURANCE"


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_MATERIALITY_THRESHOLD_PCT = Decimal("5")
DEFAULT_COMPLETENESS_TARGET_PCT = Decimal("95")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    """A single audit trail entry for a consolidation step.

    Records what happened, when, by whom, and the before/after
    state of the affected data.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    entry_id: str = Field(
        default_factory=_new_uuid,
        description="Unique audit entry identifier.",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year.",
    )
    step_type: str = Field(
        ...,
        description="Type of consolidation step.",
    )
    entity_id: Optional[str] = Field(
        None,
        description="Entity affected (if applicable).",
    )
    description: str = Field(
        ...,
        description="Human-readable description of what occurred.",
    )
    before_value: Optional[str] = Field(
        None,
        description="Value before the step (JSON or string).",
    )
    after_value: Optional[str] = Field(
        None,
        description="Value after the step (JSON or string).",
    )
    impact_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Impact of this step in tCO2e.",
    )
    performed_by: Optional[str] = Field(
        None,
        description="User or system that performed the step.",
    )
    evidence_reference: Optional[str] = Field(
        None,
        description="Reference to supporting documentation.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of the audit entry.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )

    @field_validator("impact_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

    @field_validator("step_type")
    @classmethod
    def _validate_step_type(cls, v: str) -> str:
        valid = {st.value for st in AuditStepType}
        if v.upper() not in valid:
            logger.warning("Audit step type '%s' not standard; accepted.", v)
        return v.upper()


class ReconciliationResult(BaseModel):
    """Reconciliation between bottom-up and top-down totals.

    Compares the sum of all entity totals (after adjustments and
    eliminations) against an independent top-down figure.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    reconciliation_id: str = Field(
        default_factory=_new_uuid,
        description="Unique reconciliation identifier.",
    )
    reporting_year: int = Field(
        ...,
        description="Reporting year.",
    )
    bottom_up_total: Decimal = Field(
        ...,
        description="Sum of entity totals (tCO2e).",
    )
    top_down_total: Decimal = Field(
        ...,
        description="Independent top-down total (tCO2e).",
    )
    variance: Decimal = Field(
        default=Decimal("0"),
        description="Absolute variance (bottom_up - top_down).",
    )
    variance_pct: Decimal = Field(
        default=Decimal("0"),
        description="Variance as percentage of top-down.",
    )
    is_material: bool = Field(
        default=False,
        description="Whether the variance exceeds materiality threshold.",
    )
    materiality_threshold_pct: Decimal = Field(
        default=DEFAULT_MATERIALITY_THRESHOLD_PCT,
        description="Materiality threshold used.",
    )
    reconciling_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Known reconciling items.",
    )
    unexplained_variance: Decimal = Field(
        default=Decimal("0"),
        description="Variance not explained by reconciling items.",
    )
    status: str = Field(
        default="PENDING",
        description="RECONCILED, PARTIALLY_RECONCILED, or UNRECONCILED.",
    )
    notes: Optional[str] = Field(
        None,
        description="Reconciliation notes.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
    )
    provenance_hash: str = Field(default="")

    @field_validator(
        "bottom_up_total", "top_down_total", "variance",
        "variance_pct", "materiality_threshold_pct",
        "unexplained_variance", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))


class CompletenessCheck(BaseModel):
    """Completeness assessment for the consolidation."""
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    check_id: str = Field(
        default_factory=_new_uuid,
        description="Unique check identifier.",
    )
    reporting_year: int = Field(
        ...,
        description="Reporting year.",
    )
    entities_in_boundary: int = Field(
        ...,
        ge=0,
        description="Total entities in the boundary.",
    )
    entities_reported: int = Field(
        ...,
        ge=0,
        description="Entities that have submitted data.",
    )
    completeness_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of entities reported.",
    )
    target_pct: Decimal = Field(
        default=DEFAULT_COMPLETENESS_TARGET_PCT,
        description="Target completeness percentage.",
    )
    meets_target: bool = Field(
        default=False,
        description="Whether completeness meets the target.",
    )
    missing_entities: List[str] = Field(
        default_factory=list,
        description="Entity IDs that have not reported.",
    )
    scope_coverage: Dict[str, bool] = Field(
        default_factory=dict,
        description="Whether each scope is covered (S1, S2, S3).",
    )
    scopes_missing: List[str] = Field(
        default_factory=list,
        description="Scopes with incomplete coverage.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
    )
    provenance_hash: str = Field(default="")

    @field_validator("completeness_pct", "target_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))


class SignOff(BaseModel):
    """Sign-off record for entity or group level approval."""
    model_config = ConfigDict(validate_default=True)

    signoff_id: str = Field(
        default_factory=_new_uuid,
        description="Unique sign-off identifier.",
    )
    reporting_year: int = Field(
        ...,
        description="Reporting year.",
    )
    level: str = Field(
        ...,
        description="Sign-off level (ENTITY, GROUP, etc.).",
    )
    entity_id: Optional[str] = Field(
        None,
        description="Entity ID (for entity-level sign-offs).",
    )
    signer: str = Field(
        ...,
        description="Person signing off.",
    )
    role: Optional[str] = Field(
        None,
        description="Role of the signer.",
    )
    comments: Optional[str] = Field(
        None,
        description="Sign-off comments.",
    )
    signed_at: datetime = Field(
        default_factory=_utcnow,
        description="When the sign-off was performed.",
    )
    provenance_hash: str = Field(default="")

    @field_validator("level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        valid = {l.value for l in SignOffLevel}
        if v.upper() not in valid:
            logger.warning("Sign-off level '%s' not standard; accepted.", v)
        return v.upper()


class AuditFinding(BaseModel):
    """An audit finding identified during consolidation review."""
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    finding_id: str = Field(
        default_factory=_new_uuid,
        description="Unique finding identifier.",
    )
    reporting_year: int = Field(
        ...,
        description="Reporting year.",
    )
    severity: str = Field(
        ...,
        description="Finding severity.",
    )
    title: str = Field(
        ...,
        description="Short title of the finding.",
    )
    description: str = Field(
        ...,
        description="Detailed description.",
    )
    entity_id: Optional[str] = Field(
        None,
        description="Affected entity (if applicable).",
    )
    impact_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Estimated emission impact.",
    )
    recommendation: Optional[str] = Field(
        None,
        description="Recommended corrective action.",
    )
    status: str = Field(
        default=FindingStatus.OPEN.value,
        description="Current status of the finding.",
    )
    assigned_to: Optional[str] = Field(
        None,
        description="Person responsible for resolution.",
    )
    due_date: Optional[date] = Field(
        None,
        description="Target resolution date.",
    )
    resolution_notes: Optional[str] = Field(
        None,
        description="Notes on how the finding was resolved.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
    )
    provenance_hash: str = Field(default="")

    @field_validator("impact_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

    @field_validator("severity")
    @classmethod
    def _validate_severity(cls, v: str) -> str:
        valid = {s.value for s in FindingSeverity}
        if v.upper() not in valid:
            raise ValueError(
                f"Invalid severity '{v}'. Must be one of {sorted(valid)}."
            )
        return v.upper()


class AssurancePackage(BaseModel):
    """Assurance-ready documentation package.

    Contains all the documentation needed for external assurance
    engagement (ISAE 3410 / ISO 14064-3).
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    package_id: str = Field(
        default_factory=_new_uuid,
        description="Unique package identifier.",
    )
    reporting_year: int = Field(
        ...,
        description="Reporting year.",
    )
    organisation_name: str = Field(
        default="",
        description="Organisation name.",
    )
    total_audit_entries: int = Field(default=0)
    total_findings: int = Field(default=0)
    open_findings: int = Field(default=0)
    critical_findings: int = Field(default=0)
    total_signoffs: int = Field(default=0)
    entity_signoffs: int = Field(default=0)
    group_signoffs: int = Field(default=0)
    reconciliation_status: str = Field(default="PENDING")
    completeness_pct: Decimal = Field(default=Decimal("0"))
    consolidated_total_tco2e: Decimal = Field(default=Decimal("0"))
    audit_trail_summary: List[Dict[str, Any]] = Field(default_factory=list)
    findings_summary: List[Dict[str, Any]] = Field(default_factory=list)
    signoff_summary: List[Dict[str, Any]] = Field(default_factory=list)
    is_assurance_ready: bool = Field(
        default=False,
        description="Whether all assurance prerequisites are met.",
    )
    assurance_readiness_checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Individual assurance readiness checks.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
    )
    provenance_hash: str = Field(default="")

    @field_validator(
        "completeness_pct", "consolidated_total_tco2e", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ConsolidationAuditEngine:
    """Provides a complete audit trail for GHG consolidation.

    Tracks every step, performs reconciliation and completeness
    checks, manages sign-offs and findings, and generates
    assurance-ready documentation.

    Attributes:
        _entries: List of all audit entries.
        _reconciliations: Dict mapping reconciliation_id.
        _completeness_checks: Dict mapping check_id.
        _signoffs: List of all sign-offs.
        _findings: Dict mapping finding_id to AuditFinding.
        _packages: Dict mapping package_id to AssurancePackage.

    Example:
        >>> engine = ConsolidationAuditEngine()
        >>> engine.record_step(
        ...     reporting_year=2025,
        ...     step_type="DATA_RECEIPT",
        ...     description="Received data from Entity A",
        ...     entity_id="ENT-A",
        ... )
        >>> recon = engine.reconcile(
        ...     reporting_year=2025,
        ...     bottom_up=Decimal("50000"),
        ...     top_down=Decimal("49500"),
        ... )
        >>> assert recon.within_tolerance
    """

    def __init__(self) -> None:
        """Initialise the ConsolidationAuditEngine."""
        self._entries: List[AuditEntry] = []
        self._reconciliations: Dict[str, ReconciliationResult] = {}
        self._completeness_checks: Dict[str, CompletenessCheck] = {}
        self._signoffs: List[SignOff] = []
        self._findings: Dict[str, AuditFinding] = {}
        self._packages: Dict[str, AssurancePackage] = {}
        logger.info(
            "ConsolidationAuditEngine v%s initialised.", _MODULE_VERSION
        )

    # ------------------------------------------------------------------
    # Audit Trail Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        reporting_year: int,
        step_type: str,
        description: str,
        entity_id: Optional[str] = None,
        before_value: Optional[str] = None,
        after_value: Optional[str] = None,
        impact_tco2e: Union[Decimal, str, int, float] = "0",
        performed_by: Optional[str] = None,
        evidence_reference: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Record a consolidation step in the audit trail.

        Args:
            reporting_year: Reporting year.
            step_type: Type of step (from AuditStepType).
            description: What happened.
            entity_id: Affected entity.
            before_value: State before (JSON string).
            after_value: State after (JSON string).
            impact_tco2e: Emission impact.
            performed_by: Who performed it.
            evidence_reference: Supporting doc reference.
            metadata: Additional key-value data.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            reporting_year=reporting_year,
            step_type=step_type,
            entity_id=entity_id,
            description=description,
            before_value=before_value,
            after_value=after_value,
            impact_tco2e=_decimal(impact_tco2e),
            performed_by=performed_by,
            evidence_reference=evidence_reference,
            metadata=metadata or {},
        )
        entry.provenance_hash = _compute_hash(entry)
        self._entries.append(entry)

        logger.info(
            "Audit step recorded: %s for year %d (entity=%s).",
            step_type, reporting_year, entity_id or "N/A",
        )
        return entry

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(
        self,
        reporting_year: int,
        bottom_up: Union[Decimal, str, int, float],
        top_down: Union[Decimal, str, int, float],
        materiality_threshold_pct: Optional[
            Union[Decimal, str, int, float]
        ] = None,
        reconciling_items: Optional[List[Dict[str, Any]]] = None,
    ) -> ReconciliationResult:
        """Reconcile bottom-up entity totals against top-down total.

        Args:
            reporting_year: Reporting year.
            bottom_up: Sum of entity totals.
            top_down: Independent top-down total.
            materiality_threshold_pct: Materiality threshold.
            reconciling_items: Known reconciling items.

        Returns:
            ReconciliationResult with variance analysis.
        """
        bu = _decimal(bottom_up)
        td = _decimal(top_down)
        threshold = _decimal(
            materiality_threshold_pct
            if materiality_threshold_pct is not None
            else DEFAULT_MATERIALITY_THRESHOLD_PCT
        )

        variance = _round2(bu - td)
        variance_pct = _round2(
            _safe_divide(abs(variance), td) * Decimal("100")
        ) if td != Decimal("0") else Decimal("0")

        is_material = abs(variance_pct) > threshold

        # Calculate explained variance
        items = reconciling_items or []
        explained = sum(
            (_decimal(item.get("amount", "0")) for item in items),
            Decimal("0"),
        )
        unexplained = _round2(variance - explained)

        # Determine status
        if abs(variance_pct) <= Decimal("1"):
            status = "RECONCILED"
        elif abs(_round2(
            _safe_divide(abs(unexplained), td) * Decimal("100")
        ) if td != Decimal("0") else Decimal("0")) <= threshold:
            status = "PARTIALLY_RECONCILED"
        else:
            status = "UNRECONCILED"

        result = ReconciliationResult(
            reporting_year=reporting_year,
            bottom_up_total=bu,
            top_down_total=td,
            variance=variance,
            variance_pct=variance_pct,
            is_material=is_material,
            materiality_threshold_pct=threshold,
            reconciling_items=items,
            unexplained_variance=unexplained,
            status=status,
        )
        result.provenance_hash = _compute_hash(result)
        self._reconciliations[result.reconciliation_id] = result

        # Record in audit trail
        self.record_step(
            reporting_year=reporting_year,
            step_type=AuditStepType.RECONCILIATION.value,
            description=(
                f"Reconciliation: BU={bu}, TD={td}, "
                f"variance={variance} ({variance_pct}%), "
                f"status={status}."
            ),
            impact_tco2e=str(variance),
        )

        logger.info(
            "Reconciliation: BU=%s, TD=%s, variance=%s (%s%%), "
            "material=%s, status=%s.",
            bu, td, variance, variance_pct, is_material, status,
        )
        return result

    # ------------------------------------------------------------------
    # Completeness
    # ------------------------------------------------------------------

    def check_completeness(
        self,
        reporting_year: int,
        entities_in_boundary: List[str],
        entities_reported: List[str],
        scope_coverage: Optional[Dict[str, bool]] = None,
        target_pct: Optional[Union[Decimal, str, int, float]] = None,
    ) -> CompletenessCheck:
        """Check reporting completeness.

        Args:
            reporting_year: Reporting year.
            entities_in_boundary: All entity IDs in boundary.
            entities_reported: Entity IDs that submitted data.
            scope_coverage: Dict with S1/S2/S3 coverage bools.
            target_pct: Target completeness percentage.

        Returns:
            CompletenessCheck with detailed assessment.
        """
        boundary_set = set(entities_in_boundary)
        reported_set = set(entities_reported)
        missing = sorted(boundary_set - reported_set)

        total = len(boundary_set)
        reported = len(reported_set & boundary_set)
        completeness_pct = _round2(
            _safe_divide(_decimal(reported), _decimal(total)) * Decimal("100")
        ) if total > 0 else Decimal("0")

        tgt = _decimal(
            target_pct if target_pct is not None
            else DEFAULT_COMPLETENESS_TARGET_PCT
        )
        meets_target = completeness_pct >= tgt

        # Scope coverage
        coverage = scope_coverage or {
            "scope1": True, "scope2": True, "scope3": False
        }
        scopes_missing = [
            scope for scope, covered in coverage.items()
            if not covered
        ]

        check = CompletenessCheck(
            reporting_year=reporting_year,
            entities_in_boundary=total,
            entities_reported=reported,
            completeness_pct=completeness_pct,
            target_pct=tgt,
            meets_target=meets_target,
            missing_entities=missing,
            scope_coverage=coverage,
            scopes_missing=scopes_missing,
        )
        check.provenance_hash = _compute_hash(check)
        self._completeness_checks[check.check_id] = check

        # Record in audit trail
        self.record_step(
            reporting_year=reporting_year,
            step_type=AuditStepType.COMPLETENESS_CHECK.value,
            description=(
                f"Completeness: {reported}/{total} entities ({completeness_pct}%), "
                f"target={tgt}%, meets_target={meets_target}."
            ),
        )

        logger.info(
            "Completeness check: %d/%d entities (%s%%), "
            "meets target=%s, %d missing.",
            reported, total, completeness_pct, meets_target, len(missing),
        )
        return check

    # ------------------------------------------------------------------
    # Sign-Off
    # ------------------------------------------------------------------

    def record_signoff(
        self,
        reporting_year: int,
        level: str,
        signer: str,
        entity_id: Optional[str] = None,
        role: Optional[str] = None,
        comments: Optional[str] = None,
    ) -> SignOff:
        """Record a sign-off at entity or group level.

        Args:
            reporting_year: Reporting year.
            level: Sign-off level (ENTITY, GROUP, etc.).
            signer: Person signing off.
            entity_id: Entity (for entity-level).
            role: Signer's role.
            comments: Sign-off comments.

        Returns:
            The created SignOff record.
        """
        signoff = SignOff(
            reporting_year=reporting_year,
            level=level,
            entity_id=entity_id,
            signer=signer,
            role=role,
            comments=comments,
        )
        signoff.provenance_hash = _compute_hash(signoff)
        self._signoffs.append(signoff)

        # Record in audit trail
        self.record_step(
            reporting_year=reporting_year,
            step_type=AuditStepType.SIGN_OFF.value,
            description=(
                f"{level} sign-off by '{signer}' "
                f"(entity={entity_id or 'GROUP'})."
            ),
            entity_id=entity_id,
            performed_by=signer,
        )

        logger.info(
            "Sign-off recorded: level=%s, signer='%s', entity=%s.",
            level, signer, entity_id or "GROUP",
        )
        return signoff

    # ------------------------------------------------------------------
    # Audit Findings
    # ------------------------------------------------------------------

    def record_finding(
        self,
        reporting_year: int,
        severity: str,
        title: str,
        description: str,
        entity_id: Optional[str] = None,
        impact_tco2e: Union[Decimal, str, int, float] = "0",
        recommendation: Optional[str] = None,
        assigned_to: Optional[str] = None,
        due_date: Optional[date] = None,
    ) -> AuditFinding:
        """Record an audit finding.

        Args:
            reporting_year: Reporting year.
            severity: Finding severity.
            title: Short title.
            description: Detailed description.
            entity_id: Affected entity.
            impact_tco2e: Estimated impact.
            recommendation: Corrective action recommendation.
            assigned_to: Responsible person.
            due_date: Target resolution date.

        Returns:
            The created AuditFinding.
        """
        finding = AuditFinding(
            reporting_year=reporting_year,
            severity=severity,
            title=title,
            description=description,
            entity_id=entity_id,
            impact_tco2e=_decimal(impact_tco2e),
            recommendation=recommendation,
            status=FindingStatus.OPEN.value,
            assigned_to=assigned_to,
            due_date=due_date,
        )
        finding.provenance_hash = _compute_hash(finding)
        self._findings[finding.finding_id] = finding

        logger.info(
            "Finding recorded: '%s' (%s), severity=%s, entity=%s.",
            title, finding.finding_id, severity, entity_id or "N/A",
        )
        return finding

    def update_finding_status(
        self,
        finding_id: str,
        new_status: str,
        resolution_notes: Optional[str] = None,
    ) -> AuditFinding:
        """Update the status of an audit finding.

        Args:
            finding_id: The finding to update.
            new_status: New status value.
            resolution_notes: Notes on the resolution.

        Returns:
            Updated AuditFinding.

        Raises:
            KeyError: If finding not found.
        """
        if finding_id not in self._findings:
            raise KeyError(f"Finding '{finding_id}' not found.")

        valid = {s.value for s in FindingStatus}
        if new_status.upper() not in valid:
            raise ValueError(
                f"Invalid status '{new_status}'. Must be one of {sorted(valid)}."
            )

        finding = self._findings[finding_id]
        data = finding.model_dump()
        data["status"] = new_status.upper()
        data["updated_at"] = _utcnow()
        if resolution_notes:
            data["resolution_notes"] = resolution_notes

        updated = AuditFinding(**data)
        updated.provenance_hash = _compute_hash(updated)
        self._findings[finding_id] = updated

        logger.info(
            "Finding '%s' updated: status=%s.", finding_id, new_status
        )
        return updated

    # ------------------------------------------------------------------
    # Assurance Package
    # ------------------------------------------------------------------

    def generate_assurance_package(
        self,
        reporting_year: int,
        organisation_name: str = "",
        consolidated_total_tco2e: Union[Decimal, str, int, float] = "0",
    ) -> AssurancePackage:
        """Generate an assurance-ready documentation package.

        Compiles audit trail, findings, sign-offs, reconciliation,
        and completeness checks into a single package.

        Args:
            reporting_year: Reporting year.
            organisation_name: Organisation name.
            consolidated_total_tco2e: Final consolidated total.

        Returns:
            AssurancePackage with readiness assessment.
        """
        logger.info(
            "Generating assurance package for year %d.", reporting_year
        )

        # Audit trail summary
        year_entries = [
            e for e in self._entries
            if e.reporting_year == reporting_year
        ]
        trail_summary = self._summarise_audit_trail(year_entries)

        # Findings summary
        year_findings = [
            f for f in self._findings.values()
            if f.reporting_year == reporting_year
        ]
        findings_summary = self._summarise_findings(year_findings)
        open_findings = sum(
            1 for f in year_findings
            if f.status in (FindingStatus.OPEN.value, FindingStatus.IN_PROGRESS.value)
        )
        critical_findings = sum(
            1 for f in year_findings
            if f.severity == FindingSeverity.CRITICAL.value
            and f.status != FindingStatus.CLOSED.value
        )

        # Sign-off summary
        year_signoffs = [
            s for s in self._signoffs
            if s.reporting_year == reporting_year
        ]
        signoff_summary = self._summarise_signoffs(year_signoffs)
        entity_so = sum(
            1 for s in year_signoffs
            if s.level == SignOffLevel.ENTITY.value
        )
        group_so = sum(
            1 for s in year_signoffs
            if s.level == SignOffLevel.GROUP.value
        )

        # Reconciliation status
        year_recons = [
            r for r in self._reconciliations.values()
            if r.reporting_year == reporting_year
        ]
        recon_status = "PENDING"
        if year_recons:
            latest = year_recons[-1]
            recon_status = latest.status

        # Completeness
        year_checks = [
            c for c in self._completeness_checks.values()
            if c.reporting_year == reporting_year
        ]
        completeness_pct = Decimal("0")
        if year_checks:
            completeness_pct = year_checks[-1].completeness_pct

        # Assurance readiness checks
        readiness = {
            "audit_trail_exists": len(year_entries) > 0,
            "reconciliation_done": recon_status != "PENDING",
            "reconciliation_acceptable": recon_status in (
                "RECONCILED", "PARTIALLY_RECONCILED"
            ),
            "no_critical_findings": critical_findings == 0,
            "all_findings_resolved": open_findings == 0,
            "group_signoff_done": group_so > 0,
            "completeness_above_target": completeness_pct >= DEFAULT_COMPLETENESS_TARGET_PCT,
        }
        is_ready = all(readiness.values())

        package = AssurancePackage(
            reporting_year=reporting_year,
            organisation_name=organisation_name,
            total_audit_entries=len(year_entries),
            total_findings=len(year_findings),
            open_findings=open_findings,
            critical_findings=critical_findings,
            total_signoffs=len(year_signoffs),
            entity_signoffs=entity_so,
            group_signoffs=group_so,
            reconciliation_status=recon_status,
            completeness_pct=completeness_pct,
            consolidated_total_tco2e=_decimal(consolidated_total_tco2e),
            audit_trail_summary=trail_summary,
            findings_summary=findings_summary,
            signoff_summary=signoff_summary,
            is_assurance_ready=is_ready,
            assurance_readiness_checks=readiness,
        )
        package.provenance_hash = _compute_hash(package)
        self._packages[package.package_id] = package

        # Record in audit trail
        self.record_step(
            reporting_year=reporting_year,
            step_type=AuditStepType.ASSURANCE_PACKAGE.value,
            description=(
                f"Assurance package generated: ready={is_ready}, "
                f"{len(year_entries)} entries, {len(year_findings)} findings."
            ),
        )

        logger.info(
            "Assurance package '%s': ready=%s, entries=%d, "
            "findings=%d (open=%d, critical=%d), signoffs=%d.",
            package.package_id, is_ready, len(year_entries),
            len(year_findings), open_findings, critical_findings,
            len(year_signoffs),
        )
        return package

    def _summarise_audit_trail(
        self,
        entries: List[AuditEntry],
    ) -> List[Dict[str, Any]]:
        """Summarise audit trail by step type.

        Args:
            entries: Audit entries to summarise.

        Returns:
            List of dicts with step_type, count, and total_impact.
        """
        by_type: Dict[str, Dict[str, Any]] = {}
        for e in entries:
            if e.step_type not in by_type:
                by_type[e.step_type] = {
                    "step_type": e.step_type,
                    "count": 0,
                    "total_impact_tco2e": Decimal("0"),
                }
            by_type[e.step_type]["count"] += 1
            by_type[e.step_type]["total_impact_tco2e"] += e.impact_tco2e

        result: List[Dict[str, Any]] = []
        for vals in by_type.values():
            result.append({
                "step_type": vals["step_type"],
                "count": vals["count"],
                "total_impact_tco2e": str(
                    _round2(vals["total_impact_tco2e"])
                ),
            })
        return result

    def _summarise_findings(
        self,
        findings: List[AuditFinding],
    ) -> List[Dict[str, Any]]:
        """Summarise findings by severity and status.

        Args:
            findings: Findings to summarise.

        Returns:
            List of summary dicts.
        """
        by_severity: Dict[str, Dict[str, int]] = {}
        for f in findings:
            if f.severity not in by_severity:
                by_severity[f.severity] = {}
            by_severity[f.severity][f.status] = (
                by_severity[f.severity].get(f.status, 0) + 1
            )

        result: List[Dict[str, Any]] = []
        for sev, statuses in sorted(by_severity.items()):
            result.append({
                "severity": sev,
                "by_status": statuses,
                "total": sum(statuses.values()),
            })
        return result

    def _summarise_signoffs(
        self,
        signoffs: List[SignOff],
    ) -> List[Dict[str, Any]]:
        """Summarise sign-offs by level.

        Args:
            signoffs: Sign-offs to summarise.

        Returns:
            List of summary dicts.
        """
        by_level: Dict[str, int] = {}
        for s in signoffs:
            by_level[s.level] = by_level.get(s.level, 0) + 1

        return [
            {"level": level, "count": count}
            for level, count in sorted(by_level.items())
        ]

    # ------------------------------------------------------------------
    # Audit Trail Retrieval
    # ------------------------------------------------------------------

    def get_audit_trail(
        self,
        reporting_year: Optional[int] = None,
        entity_id: Optional[str] = None,
        step_type: Optional[str] = None,
    ) -> List[AuditEntry]:
        """Retrieve audit trail entries with optional filters.

        Args:
            reporting_year: Filter by year.
            entity_id: Filter by entity.
            step_type: Filter by step type.

        Returns:
            List of matching AuditEntry records.
        """
        results = list(self._entries)

        if reporting_year is not None:
            results = [e for e in results if e.reporting_year == reporting_year]
        if entity_id is not None:
            results = [e for e in results if e.entity_id == entity_id]
        if step_type is not None:
            results = [e for e in results if e.step_type == step_type.upper()]

        logger.info("Audit trail query: %d entries returned.", len(results))
        return results

    def get_reconciliation(
        self,
        reconciliation_id: str,
    ) -> ReconciliationResult:
        """Retrieve a reconciliation by ID.

        Args:
            reconciliation_id: The reconciliation ID.

        Returns:
            ReconciliationResult.

        Raises:
            KeyError: If not found.
        """
        if reconciliation_id not in self._reconciliations:
            raise KeyError(
                f"Reconciliation '{reconciliation_id}' not found."
            )
        return self._reconciliations[reconciliation_id]

    def get_findings(
        self,
        reporting_year: Optional[int] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[AuditFinding]:
        """Retrieve findings with optional filters.

        Args:
            reporting_year: Filter by year.
            severity: Filter by severity.
            status: Filter by status.

        Returns:
            List of matching AuditFindings.
        """
        results = list(self._findings.values())

        if reporting_year is not None:
            results = [f for f in results if f.reporting_year == reporting_year]
        if severity is not None:
            results = [f for f in results if f.severity == severity.upper()]
        if status is not None:
            results = [f for f in results if f.status == status.upper()]

        return results

    def get_signoffs(
        self,
        reporting_year: Optional[int] = None,
        level: Optional[str] = None,
    ) -> List[SignOff]:
        """Retrieve sign-offs with optional filters.

        Args:
            reporting_year: Filter by year.
            level: Filter by level.

        Returns:
            List of matching SignOff records.
        """
        results = list(self._signoffs)

        if reporting_year is not None:
            results = [s for s in results if s.reporting_year == reporting_year]
        if level is not None:
            results = [s for s in results if s.level == level.upper()]

        return results

    def get_package(self, package_id: str) -> AssurancePackage:
        """Retrieve an assurance package by ID.

        Args:
            package_id: The package ID.

        Returns:
            AssurancePackage.

        Raises:
            KeyError: If not found.
        """
        if package_id not in self._packages:
            raise KeyError(f"Assurance package '{package_id}' not found.")
        return self._packages[package_id]
