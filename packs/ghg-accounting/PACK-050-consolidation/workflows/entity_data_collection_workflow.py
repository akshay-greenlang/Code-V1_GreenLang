# -*- coding: utf-8 -*-
"""
Entity Data Collection Workflow
====================================

5-phase workflow for collecting entity-level GHG data across a multi-entity
corporate group within PACK-050 GHG Consolidation Pack.

Phases:
    1. EntityAssignment          -- Assign data stewards to each entity
                                    in the consolidation boundary.
    2. DataRequestDistribution   -- Distribute data request templates to
                                    entity contacts with scope/category
                                    requirements.
    3. SubmissionCollection      -- Collect and validate submitted GHG data
                                    from entity stewards.
    4. ValidationReview          -- Review submitted data for quality,
                                    completeness, and consistency.
    5. GapResolution             -- Identify and resolve data gaps through
                                    follow-up requests or estimation.

Regulatory Basis:
    GHG Protocol Corporate Standard (Ch. 6) -- Identifying and calculating
    GHG Protocol Corporate Standard (Ch. 7) -- Managing inventory quality
    ISO 14064-1:2018 (Cl. 5.2-5.4) -- Quantification
    CSRD / ESRS E1 -- Climate change data requirements

Author: GreenLang Team
Version: 50.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DataCollectionPhase(str, Enum):
    ENTITY_ASSIGNMENT = "entity_assignment"
    DATA_REQUEST_DISTRIBUTION = "data_request_distribution"
    SUBMISSION_COLLECTION = "submission_collection"
    VALIDATION_REVIEW = "validation_review"
    GAP_RESOLUTION = "gap_resolution"


class SubmissionStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    VALIDATED = "validated"
    REJECTED = "rejected"
    APPROVED = "approved"


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class GapResolutionMethod(str, Enum):
    FOLLOW_UP = "follow_up"
    ESTIMATION = "estimation"
    PROXY_DATA = "proxy_data"
    PRIOR_YEAR = "prior_year"
    EXCLUSION = "exclusion"


class EmissionScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"


# =============================================================================
# REFERENCE DATA
# =============================================================================

MINIMUM_COMPLETENESS_PCT = Decimal("80.0")

REQUIRED_SCOPES = [
    EmissionScope.SCOPE_1,
    EmissionScope.SCOPE_2_LOCATION,
    EmissionScope.SCOPE_2_MARKET,
]

SCOPE_3_CATEGORIES = [
    "cat_1_purchased_goods",
    "cat_2_capital_goods",
    "cat_3_fuel_energy",
    "cat_4_upstream_transport",
    "cat_5_waste",
    "cat_6_business_travel",
    "cat_7_employee_commuting",
]


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    phase_name: str = Field(...)
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class StewardAssignment(BaseModel):
    """Data steward assignment for an entity."""
    entity_id: str = Field(...)
    entity_name: str = Field("")
    steward_name: str = Field("")
    steward_email: str = Field("")
    steward_role: str = Field("data_steward")
    backup_steward_name: str = Field("")
    backup_steward_email: str = Field("")
    assigned_at: str = Field(default_factory=lambda: _utcnow().isoformat())
    deadline: str = Field("")


class DataRequest(BaseModel):
    """Data request template distributed to an entity."""
    request_id: str = Field(default_factory=_new_uuid)
    entity_id: str = Field(...)
    entity_name: str = Field("")
    scopes_required: List[str] = Field(default_factory=list)
    categories_required: List[str] = Field(default_factory=list)
    reporting_period_start: str = Field("")
    reporting_period_end: str = Field("")
    submission_deadline: str = Field("")
    template_version: str = Field("1.0")
    distributed_at: str = Field(default_factory=lambda: _utcnow().isoformat())
    status: SubmissionStatus = Field(SubmissionStatus.NOT_STARTED)


class EntitySubmission(BaseModel):
    """Submitted GHG data from an entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    submission_id: str = Field(default_factory=_new_uuid)
    entity_id: str = Field(...)
    entity_name: str = Field("")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    scope_3_categories: Dict[str, Decimal] = Field(default_factory=dict)
    data_quality_score: Decimal = Field(Decimal("0"))
    completeness_pct: Decimal = Field(Decimal("0"))
    submitted_by: str = Field("")
    submitted_at: str = Field("")
    status: SubmissionStatus = Field(SubmissionStatus.SUBMITTED)
    provenance_hash: str = Field("")


class ValidationFinding(BaseModel):
    """A validation finding for an entity submission."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    finding_id: str = Field(default_factory=_new_uuid)
    entity_id: str = Field(...)
    scope: str = Field("")
    severity: ValidationSeverity = Field(ValidationSeverity.WARNING)
    rule_code: str = Field("")
    message: str = Field("")
    current_value: str = Field("")
    expected_range: str = Field("")


class DataGap(BaseModel):
    """An identified data gap for an entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    gap_id: str = Field(default_factory=_new_uuid)
    entity_id: str = Field(...)
    entity_name: str = Field("")
    scope: str = Field("")
    category: str = Field("")
    gap_description: str = Field("")
    resolution_method: GapResolutionMethod = Field(GapResolutionMethod.FOLLOW_UP)
    estimated_value_tco2e: Decimal = Field(Decimal("0"))
    is_resolved: bool = Field(False)
    resolution_notes: str = Field("")


class EntityDataCollectionInput(BaseModel):
    """Input for the entity data collection workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    organisation_id: str = Field(...)
    reporting_year: int = Field(...)
    reporting_period_start: str = Field("")
    reporting_period_end: str = Field("")
    submission_deadline: str = Field("")
    entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Entities in boundary"
    )
    steward_assignments: List[Dict[str, Any]] = Field(
        default_factory=list, description="Steward assignment overrides"
    )
    submissions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Entity GHG data submissions"
    )
    require_scope_3: bool = Field(False)
    minimum_completeness_pct: Decimal = Field(MINIMUM_COMPLETENESS_PCT)
    skip_phases: List[str] = Field(default_factory=list)


class EntityDataCollectionResult(BaseModel):
    """Output from the entity data collection workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflow_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    steward_assignments: List[StewardAssignment] = Field(default_factory=list)
    data_requests: List[DataRequest] = Field(default_factory=list)
    submissions: List[EntitySubmission] = Field(default_factory=list)
    validation_findings: List[ValidationFinding] = Field(default_factory=list)
    data_gaps: List[DataGap] = Field(default_factory=list)
    total_entities: int = Field(0)
    entities_submitted: int = Field(0)
    entities_approved: int = Field(0)
    overall_completeness_pct: Decimal = Field(Decimal("0"))
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class EntityDataCollectionWorkflow:
    """
    5-phase entity data collection workflow for GHG consolidation.

    Assigns data stewards, distributes collection templates, validates
    submissions, and resolves gaps with full SHA-256 provenance.

    Example:
        >>> wf = EntityDataCollectionWorkflow()
        >>> inp = EntityDataCollectionInput(
        ...     organisation_id="ORG-001", reporting_year=2025,
        ...     entities=[{"entity_id": "E1", "entity_name": "Sub A"}],
        ...     submissions=[{"entity_id": "E1", "scope_1_tco2e": "1000"}],
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_ORDER: List[DataCollectionPhase] = [
        DataCollectionPhase.ENTITY_ASSIGNMENT,
        DataCollectionPhase.DATA_REQUEST_DISTRIBUTION,
        DataCollectionPhase.SUBMISSION_COLLECTION,
        DataCollectionPhase.VALIDATION_REVIEW,
        DataCollectionPhase.GAP_RESOLUTION,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._entities: List[Dict[str, Any]] = []
        self._assignments: Dict[str, StewardAssignment] = {}
        self._requests: Dict[str, DataRequest] = {}
        self._submissions: Dict[str, EntitySubmission] = {}

    def execute(self, input_data: EntityDataCollectionInput) -> EntityDataCollectionResult:
        """Execute the full 5-phase entity data collection workflow."""
        start = _utcnow()
        result = EntityDataCollectionResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        phase_methods = {
            DataCollectionPhase.ENTITY_ASSIGNMENT: self._phase_entity_assignment,
            DataCollectionPhase.DATA_REQUEST_DISTRIBUTION: self._phase_data_request_distribution,
            DataCollectionPhase.SUBMISSION_COLLECTION: self._phase_submission_collection,
            DataCollectionPhase.VALIDATION_REVIEW: self._phase_validation_review,
            DataCollectionPhase.GAP_RESOLUTION: self._phase_gap_resolution,
        }

        for idx, phase in enumerate(self.PHASE_ORDER, 1):
            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.SKIPPED,
                ))
                continue

            phase_start = _utcnow()
            try:
                phase_out = phase_methods[phase](input_data, result)
                elapsed = (_utcnow() - phase_start).total_seconds()
                ph_hash = _compute_hash(str(phase_out))
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                    outputs=phase_out, provenance_hash=ph_hash,
                ))
            except Exception as exc:
                elapsed = (_utcnow() - phase_start).total_seconds()
                logger.error("Phase %s failed: %s", phase.value, exc, exc_info=True)
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.FAILED, duration_seconds=elapsed,
                    errors=[str(exc)],
                ))
                result.status = WorkflowStatus.FAILED
                result.errors.append(f"Phase {phase.value} failed: {exc}")
                break

        if result.status != WorkflowStatus.FAILED:
            result.status = WorkflowStatus.COMPLETED

        end = _utcnow()
        result.completed_at = end.isoformat()
        result.duration_seconds = (end - start).total_seconds()
        result.provenance_hash = _compute_hash(
            f"{result.workflow_id}|{result.organisation_id}|"
            f"{result.entities_approved}|{result.completed_at}"
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- ENTITY ASSIGNMENT
    # -----------------------------------------------------------------

    def _phase_entity_assignment(
        self, input_data: EntityDataCollectionInput, result: EntityDataCollectionResult,
    ) -> Dict[str, Any]:
        """Assign data stewards to each entity in the consolidation boundary."""
        logger.info("Phase 1 -- Entity Assignment: %d entities", len(input_data.entities))
        self._entities = input_data.entities
        result.total_entities = len(input_data.entities)

        # Build steward override lookup
        override_lookup: Dict[str, Dict[str, Any]] = {}
        for sa in input_data.steward_assignments:
            eid = sa.get("entity_id", "")
            if eid:
                override_lookup[eid] = sa

        assignments: Dict[str, StewardAssignment] = {}
        auto_assigned = 0
        manual_assigned = 0

        for entity in input_data.entities:
            eid = entity.get("entity_id", _new_uuid())
            override = override_lookup.get(eid, {})

            steward_name = override.get("steward_name", entity.get("contact_name", ""))
            steward_email = override.get("steward_email", entity.get("contact_email", ""))

            if not steward_name:
                steward_name = f"Steward-{eid[:8]}"
                auto_assigned += 1
            else:
                manual_assigned += 1

            assignment = StewardAssignment(
                entity_id=eid,
                entity_name=entity.get("entity_name", ""),
                steward_name=steward_name,
                steward_email=steward_email,
                steward_role=override.get("steward_role", "data_steward"),
                backup_steward_name=override.get("backup_steward_name", ""),
                backup_steward_email=override.get("backup_steward_email", ""),
                deadline=input_data.submission_deadline,
            )
            assignments[eid] = assignment

        self._assignments = assignments
        result.steward_assignments = list(assignments.values())

        logger.info("Assignment: %d manual, %d auto-assigned", manual_assigned, auto_assigned)
        return {
            "entities_assigned": len(assignments),
            "manual_assigned": manual_assigned,
            "auto_assigned": auto_assigned,
        }

    # -----------------------------------------------------------------
    # PHASE 2 -- DATA REQUEST DISTRIBUTION
    # -----------------------------------------------------------------

    def _phase_data_request_distribution(
        self, input_data: EntityDataCollectionInput, result: EntityDataCollectionResult,
    ) -> Dict[str, Any]:
        """Distribute data request templates to entity contacts."""
        logger.info("Phase 2 -- Data Request Distribution")

        scopes_required = [s.value for s in REQUIRED_SCOPES]
        categories_required: List[str] = []
        if input_data.require_scope_3:
            scopes_required.append(EmissionScope.SCOPE_3.value)
            categories_required = SCOPE_3_CATEGORIES.copy()

        period_start = input_data.reporting_period_start or f"{input_data.reporting_year}-01-01"
        period_end = input_data.reporting_period_end or f"{input_data.reporting_year}-12-31"
        deadline = input_data.submission_deadline or f"{input_data.reporting_year + 1}-03-31"

        requests: Dict[str, DataRequest] = {}

        for entity in self._entities:
            eid = entity.get("entity_id", "")
            request = DataRequest(
                entity_id=eid,
                entity_name=entity.get("entity_name", ""),
                scopes_required=scopes_required,
                categories_required=categories_required,
                reporting_period_start=period_start,
                reporting_period_end=period_end,
                submission_deadline=deadline,
            )
            requests[eid] = request

        self._requests = requests
        result.data_requests = list(requests.values())

        logger.info("Distributed %d data requests", len(requests))
        return {
            "requests_distributed": len(requests),
            "scopes_required": scopes_required,
            "scope_3_required": input_data.require_scope_3,
            "deadline": deadline,
        }

    # -----------------------------------------------------------------
    # PHASE 3 -- SUBMISSION COLLECTION
    # -----------------------------------------------------------------

    def _phase_submission_collection(
        self, input_data: EntityDataCollectionInput, result: EntityDataCollectionResult,
    ) -> Dict[str, Any]:
        """Collect and parse submitted GHG data from entity stewards."""
        logger.info("Phase 3 -- Submission Collection: %d submissions", len(input_data.submissions))
        submissions: Dict[str, EntitySubmission] = {}

        for raw in input_data.submissions:
            eid = raw.get("entity_id", "")
            if not eid:
                result.warnings.append(f"Submission missing entity_id: {raw}")
                continue

            s1 = self._dec(raw.get("scope_1_tco2e", "0"))
            s2l = self._dec(raw.get("scope_2_location_tco2e", "0"))
            s2m = self._dec(raw.get("scope_2_market_tco2e", "0"))
            s3 = self._dec(raw.get("scope_3_tco2e", "0"))

            # Parse scope 3 categories
            s3_cats: Dict[str, Decimal] = {}
            for cat in SCOPE_3_CATEGORIES:
                cat_val = raw.get(cat)
                if cat_val is not None:
                    s3_cats[cat] = self._dec(cat_val)

            # Compute completeness
            completeness = self._calculate_completeness(s1, s2l, s2m, s3, input_data.require_scope_3)

            # Data quality based on completeness and source
            quality = self._calculate_data_quality(completeness, raw)

            prov_input = f"{eid}|{float(s1)}|{float(s2l)}|{float(s2m)}|{float(s3)}|{_utcnow().isoformat()}"
            prov_hash = _compute_hash(prov_input)

            submission = EntitySubmission(
                entity_id=eid,
                entity_name=raw.get("entity_name", ""),
                scope_1_tco2e=s1,
                scope_2_location_tco2e=s2l,
                scope_2_market_tco2e=s2m,
                scope_3_tco2e=s3,
                scope_3_categories=s3_cats,
                data_quality_score=quality,
                completeness_pct=completeness,
                submitted_by=raw.get("submitted_by", ""),
                submitted_at=raw.get("submitted_at", _utcnow().isoformat()),
                status=SubmissionStatus.SUBMITTED,
                provenance_hash=prov_hash,
            )
            submissions[eid] = submission

        self._submissions = submissions
        result.submissions = list(submissions.values())
        result.entities_submitted = len(submissions)

        # Check for missing entities
        submitted_ids = set(submissions.keys())
        all_ids = {e.get("entity_id", "") for e in self._entities}
        missing = all_ids - submitted_ids
        if missing:
            result.warnings.append(f"{len(missing)} entities have not submitted data")

        logger.info("Collected %d submissions, %d missing", len(submissions), len(missing))
        return {
            "submissions_received": len(submissions),
            "missing_submissions": len(missing),
            "average_completeness_pct": float(self._avg_completeness(submissions)),
        }

    def _calculate_completeness(
        self, s1: Decimal, s2l: Decimal, s2m: Decimal, s3: Decimal, require_s3: bool
    ) -> Decimal:
        """Calculate data completeness as percentage."""
        fields_required = 3  # S1 + S2L + S2M
        filled = sum(1 for v in [s1, s2l, s2m] if v > Decimal("0"))
        if require_s3:
            fields_required += 1
            if s3 > Decimal("0"):
                filled += 1
        pct = (Decimal(str(filled)) / Decimal(str(fields_required)) * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return pct

    def _calculate_data_quality(self, completeness: Decimal, raw: Dict[str, Any]) -> Decimal:
        """Calculate deterministic data quality score 0-100."""
        score = completeness * Decimal("0.6")
        # Bonus for methodology reference
        if raw.get("methodology"):
            score += Decimal("15")
        # Bonus for evidence documentation
        if raw.get("evidence_ref"):
            score += Decimal("15")
        # Bonus for third-party verification
        if raw.get("verified"):
            score += Decimal("10")
        return min(score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), Decimal("100"))

    def _avg_completeness(self, submissions: Dict[str, EntitySubmission]) -> Decimal:
        """Calculate average completeness across submissions."""
        if not submissions:
            return Decimal("0")
        total = sum(s.completeness_pct for s in submissions.values())
        return (total / Decimal(str(len(submissions)))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    # -----------------------------------------------------------------
    # PHASE 4 -- VALIDATION REVIEW
    # -----------------------------------------------------------------

    def _phase_validation_review(
        self, input_data: EntityDataCollectionInput, result: EntityDataCollectionResult,
    ) -> Dict[str, Any]:
        """Review submitted data for quality, completeness, and consistency."""
        logger.info("Phase 4 -- Validation Review: %d submissions", len(self._submissions))
        findings: List[ValidationFinding] = []
        approved = 0
        rejected = 0

        for eid, sub in self._submissions.items():
            entity_findings = self._validate_submission(sub, input_data)
            findings.extend(entity_findings)

            error_count = sum(1 for f in entity_findings if f.severity == ValidationSeverity.ERROR)
            if error_count > 0:
                sub.status = SubmissionStatus.REJECTED
                rejected += 1
            elif sub.completeness_pct >= input_data.minimum_completeness_pct:
                sub.status = SubmissionStatus.APPROVED
                approved += 1
            else:
                sub.status = SubmissionStatus.REJECTED
                rejected += 1
                findings.append(ValidationFinding(
                    entity_id=eid,
                    severity=ValidationSeverity.ERROR,
                    rule_code="COMPLETENESS_BELOW_THRESHOLD",
                    message=(
                        f"Completeness {sub.completeness_pct}% below "
                        f"threshold {input_data.minimum_completeness_pct}%"
                    ),
                ))

        result.validation_findings = findings
        result.entities_approved = approved
        result.submissions = list(self._submissions.values())

        logger.info("Validation: %d approved, %d rejected, %d findings",
                     approved, rejected, len(findings))
        return {
            "entities_reviewed": len(self._submissions),
            "approved": approved,
            "rejected": rejected,
            "total_findings": len(findings),
            "errors": sum(1 for f in findings if f.severity == ValidationSeverity.ERROR),
            "warnings": sum(1 for f in findings if f.severity == ValidationSeverity.WARNING),
        }

    def _validate_submission(
        self, sub: EntitySubmission, input_data: EntityDataCollectionInput
    ) -> List[ValidationFinding]:
        """Run validation rules on a single submission."""
        findings: List[ValidationFinding] = []

        # Rule: Scope 1 should not be negative
        if sub.scope_1_tco2e < Decimal("0"):
            findings.append(ValidationFinding(
                entity_id=sub.entity_id, scope="scope_1",
                severity=ValidationSeverity.ERROR,
                rule_code="NEGATIVE_EMISSIONS",
                message="Scope 1 emissions cannot be negative",
                current_value=str(sub.scope_1_tco2e),
            ))

        # Rule: Scope 2 location should not be negative
        if sub.scope_2_location_tco2e < Decimal("0"):
            findings.append(ValidationFinding(
                entity_id=sub.entity_id, scope="scope_2_location",
                severity=ValidationSeverity.ERROR,
                rule_code="NEGATIVE_EMISSIONS",
                message="Scope 2 location emissions cannot be negative",
                current_value=str(sub.scope_2_location_tco2e),
            ))

        # Rule: Scope 1 zero warning
        if sub.scope_1_tco2e == Decimal("0"):
            findings.append(ValidationFinding(
                entity_id=sub.entity_id, scope="scope_1",
                severity=ValidationSeverity.WARNING,
                rule_code="ZERO_EMISSIONS",
                message="Scope 1 emissions reported as zero -- confirm if correct",
            ))

        # Rule: Scope 2 market should be close to location
        if sub.scope_2_location_tco2e > Decimal("0") and sub.scope_2_market_tco2e > Decimal("0"):
            ratio = sub.scope_2_market_tco2e / sub.scope_2_location_tco2e
            if ratio > Decimal("3.0") or ratio < Decimal("0.1"):
                findings.append(ValidationFinding(
                    entity_id=sub.entity_id, scope="scope_2",
                    severity=ValidationSeverity.WARNING,
                    rule_code="MARKET_LOCATION_DIVERGENCE",
                    message=f"Market/location ratio {ratio:.2f} is unusual",
                    current_value=f"market={sub.scope_2_market_tco2e}, location={sub.scope_2_location_tco2e}",
                ))

        return findings

    # -----------------------------------------------------------------
    # PHASE 5 -- GAP RESOLUTION
    # -----------------------------------------------------------------

    def _phase_gap_resolution(
        self, input_data: EntityDataCollectionInput, result: EntityDataCollectionResult,
    ) -> Dict[str, Any]:
        """Identify and resolve data gaps through follow-up or estimation."""
        logger.info("Phase 5 -- Gap Resolution")
        gaps: List[DataGap] = []

        # Identify entities with no submission
        submitted_ids = set(self._submissions.keys())
        for entity in self._entities:
            eid = entity.get("entity_id", "")
            ename = entity.get("entity_name", "")

            if eid not in submitted_ids:
                gap = DataGap(
                    entity_id=eid,
                    entity_name=ename,
                    scope="all",
                    gap_description="No data submitted for entity",
                    resolution_method=GapResolutionMethod.FOLLOW_UP,
                )
                gaps.append(gap)
                continue

            sub = self._submissions[eid]

            # Check scope-level gaps
            if sub.scope_1_tco2e == Decimal("0"):
                gaps.append(DataGap(
                    entity_id=eid, entity_name=ename, scope="scope_1",
                    gap_description="Scope 1 data is zero or missing",
                    resolution_method=GapResolutionMethod.ESTIMATION,
                ))
            if sub.scope_2_location_tco2e == Decimal("0"):
                gaps.append(DataGap(
                    entity_id=eid, entity_name=ename, scope="scope_2_location",
                    gap_description="Scope 2 location data is zero or missing",
                    resolution_method=GapResolutionMethod.ESTIMATION,
                ))

            # Check rejected submissions
            if sub.status == SubmissionStatus.REJECTED:
                gaps.append(DataGap(
                    entity_id=eid, entity_name=ename, scope="all",
                    gap_description="Submission rejected during validation",
                    resolution_method=GapResolutionMethod.FOLLOW_UP,
                ))

        result.data_gaps = gaps

        # Calculate overall completeness
        all_ids = {e.get("entity_id", "") for e in self._entities}
        if all_ids:
            approved_ids = {
                eid for eid, sub in self._submissions.items()
                if sub.status == SubmissionStatus.APPROVED
            }
            result.overall_completeness_pct = (
                Decimal(str(len(approved_ids))) / Decimal(str(len(all_ids))) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        resolved = sum(1 for g in gaps if g.is_resolved)
        logger.info("Gap resolution: %d gaps found, %d resolved", len(gaps), resolved)
        return {
            "total_gaps": len(gaps),
            "resolved_gaps": resolved,
            "unresolved_gaps": len(gaps) - resolved,
            "overall_completeness_pct": float(result.overall_completeness_pct),
            "gap_methods": self._count_gap_methods(gaps),
        }

    def _count_gap_methods(self, gaps: List[DataGap]) -> Dict[str, int]:
        """Count gaps by resolution method."""
        counts: Dict[str, int] = {}
        for g in gaps:
            k = g.resolution_method.value
            counts[k] = counts.get(k, 0) + 1
        return counts

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    def _dec(self, value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "EntityDataCollectionWorkflow",
    "EntityDataCollectionInput",
    "EntityDataCollectionResult",
    "DataCollectionPhase",
    "SubmissionStatus",
    "ValidationSeverity",
    "GapResolutionMethod",
    "EmissionScope",
    "StewardAssignment",
    "DataRequest",
    "EntitySubmission",
    "ValidationFinding",
    "DataGap",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
