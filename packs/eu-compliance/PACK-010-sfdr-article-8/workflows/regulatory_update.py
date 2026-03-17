# -*- coding: utf-8 -*-
"""
Regulatory Update Workflow
=============================

Three-phase workflow for SFDR regulatory change management for Article 8
financial products. Orchestrates change detection, impact assessment, and
migration planning into a single auditable pipeline.

Regulatory Context:
    The SFDR regulatory landscape is evolving with:
    - SFDR 2.0 (Level 1 review): Proposed reclassification of products
      into categories (sustainability-focused, transition, ESG collection,
      unclassified) to replace Article 8/9 framework.
    - ESMA guidance updates: Regular Q&A publications clarifying
      interpretation of existing requirements.
    - Level 2 RTS amendments: Technical standards updates addressing
      PAI calculation methodologies, taxonomy alignment disclosure,
      and pre-contractual/periodic template changes.
    - EU Taxonomy updates: New delegated acts covering additional
      environmental objectives, activity eligibility changes.
    - CSRD alignment: Double materiality linkages affecting SFDR disclosures.

    Key Monitoring Sources:
    - European Commission SFDR publications
    - ESMA (European Securities and Markets Authority) guidance
    - EBA (European Banking Authority) opinions
    - National Competent Authority guidance
    - EU Official Journal (legislative changes)

Phases:
    1. ChangeDetection - Monitor SFDR amendments, ESMA guidance updates,
       Q&A publications, Level 2 RTS changes, SFDR 2.0 developments
    2. ImpactAssessment - Evaluate impact on current disclosures, flag
       required disclosure updates, assess new data requirements
    3. MigrationPlanning - Plan disclosure updates with timeline, assign
       responsibilities, create implementation roadmap

Author: GreenLang Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITIES
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time with timezone info."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class ChangeSource(str, Enum):
    """Source of regulatory change."""
    EU_COMMISSION = "EU_COMMISSION"
    ESMA = "ESMA"
    EBA = "EBA"
    NCA = "NCA"
    EU_OFFICIAL_JOURNAL = "EU_OFFICIAL_JOURNAL"
    TAXONOMY_PLATFORM = "TAXONOMY_PLATFORM"


class ChangeCategory(str, Enum):
    """Category of regulatory change."""
    LEVEL_1_AMENDMENT = "LEVEL_1_AMENDMENT"
    LEVEL_2_RTS = "LEVEL_2_RTS"
    GUIDANCE_UPDATE = "GUIDANCE_UPDATE"
    QA_PUBLICATION = "QA_PUBLICATION"
    TAXONOMY_UPDATE = "TAXONOMY_UPDATE"
    SFDR_2_DEVELOPMENT = "SFDR_2_DEVELOPMENT"


class ImpactLevel(str, Enum):
    """Impact level of a regulatory change."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class MigrationStatus(str, Enum):
    """Status of a migration task."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"


# =============================================================================
# DATA MODELS - SHARED
# =============================================================================


class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=_utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# DATA MODELS - REGULATORY UPDATE
# =============================================================================


class RegulatoryChange(BaseModel):
    """A detected regulatory change."""
    change_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Change title")
    description: str = Field(default="", description="Change description")
    source: ChangeSource = Field(default=ChangeSource.ESMA)
    category: ChangeCategory = Field(default=ChangeCategory.GUIDANCE_UPDATE)
    publication_date: str = Field(default="", description="YYYY-MM-DD")
    effective_date: Optional[str] = Field(
        None, description="Effective date YYYY-MM-DD"
    )
    reference_url: Optional[str] = Field(None)
    reference_document: Optional[str] = Field(None)
    affected_articles: List[str] = Field(
        default_factory=list,
        description="Affected SFDR articles"
    )
    affected_annexes: List[str] = Field(
        default_factory=list,
        description="Affected annexes (II, III, IV)"
    )


class CurrentDisclosureState(BaseModel):
    """Current state of disclosures for impact assessment."""
    annex_ii_version: Optional[str] = Field(None)
    annex_ii_last_updated: Optional[str] = Field(None)
    annex_iii_version: Optional[str] = Field(None)
    annex_iii_last_updated: Optional[str] = Field(None)
    annex_iv_version: Optional[str] = Field(None)
    annex_iv_last_updated: Optional[str] = Field(None)
    pai_indicators_count: int = Field(default=18)
    taxonomy_alignment_disclosed: bool = Field(default=False)
    sfdr_classification: str = Field(default="ARTICLE_8")


class RegulatoryUpdateInput(BaseModel):
    """Input configuration for the regulatory update workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    review_date: str = Field(..., description="Review date YYYY-MM-DD")
    detected_changes: List[RegulatoryChange] = Field(
        default_factory=list, description="Detected regulatory changes"
    )
    current_disclosure_state: Optional[CurrentDisclosureState] = Field(
        None, description="Current disclosure state for impact assessment"
    )
    monitoring_sources: List[str] = Field(
        default_factory=lambda: [
            ChangeSource.EU_COMMISSION.value,
            ChangeSource.ESMA.value,
            ChangeSource.EU_OFFICIAL_JOURNAL.value,
        ],
        description="Sources to monitor for changes"
    )
    last_review_date: Optional[str] = Field(
        None, description="Date of last regulatory review"
    )
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("review_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate review date is valid ISO format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("review_date must be YYYY-MM-DD format")
        return v


class RegulatoryUpdateResult(WorkflowResult):
    """Complete result from the regulatory update workflow."""
    product_name: str = Field(default="")
    changes_detected: int = Field(default=0)
    critical_changes: int = Field(default=0)
    high_impact_changes: int = Field(default=0)
    disclosures_requiring_update: int = Field(default=0)
    migration_tasks_created: int = Field(default=0)
    estimated_effort_days: float = Field(default=0.0)
    earliest_deadline: Optional[str] = Field(None)


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================


class ChangeDetectionPhase:
    """
    Phase 1: Change Detection.

    Monitors SFDR amendments, ESMA guidance updates, Q&A publications,
    Level 2 RTS changes, and SFDR 2.0 developments.
    """

    PHASE_NAME = "change_detection"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute change detection phase."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            detected_changes = config.get("detected_changes", [])
            monitoring_sources = config.get("monitoring_sources", [])
            last_review = config.get("last_review_date")

            outputs["monitoring_sources"] = monitoring_sources
            outputs["last_review_date"] = last_review
            outputs["review_date"] = config.get("review_date", "")

            # Classify and organize detected changes
            changes_by_category: Dict[str, List[Dict[str, Any]]] = {}
            changes_by_source: Dict[str, int] = {}

            for change in detected_changes:
                category = change.get(
                    "category", ChangeCategory.GUIDANCE_UPDATE.value
                )
                source = change.get("source", ChangeSource.ESMA.value)

                if category not in changes_by_category:
                    changes_by_category[category] = []
                changes_by_category[category].append(change)

                changes_by_source[source] = (
                    changes_by_source.get(source, 0) + 1
                )

            outputs["total_changes_detected"] = len(detected_changes)
            outputs["changes_by_category"] = {
                k: len(v) for k, v in changes_by_category.items()
            }
            outputs["changes_by_source"] = changes_by_source
            outputs["detected_changes"] = detected_changes

            # Identify changes since last review
            new_since_review: List[Dict[str, Any]] = []
            if last_review:
                for change in detected_changes:
                    pub_date = change.get("publication_date", "")
                    if pub_date and pub_date > last_review:
                        new_since_review.append(change)
            else:
                new_since_review = detected_changes

            outputs["new_since_last_review"] = len(new_since_review)
            outputs["new_changes"] = new_since_review

            # Flag urgent changes
            urgent_changes = [
                c for c in detected_changes
                if c.get("category") in (
                    ChangeCategory.LEVEL_1_AMENDMENT.value,
                    ChangeCategory.LEVEL_2_RTS.value,
                )
            ]
            outputs["urgent_changes_count"] = len(urgent_changes)

            if urgent_changes:
                warnings.append(
                    f"{len(urgent_changes)} legislative/RTS change(s) "
                    f"detected requiring attention"
                )

            # SFDR 2.0 tracking
            sfdr2_changes = [
                c for c in detected_changes
                if c.get("category") == ChangeCategory.SFDR_2_DEVELOPMENT.value
            ]
            outputs["sfdr_2_developments"] = len(sfdr2_changes)

            if sfdr2_changes:
                outputs["sfdr_2_status"] = (
                    "Active developments detected. Monitor for transition "
                    "timeline from Article 8/9 to new categorization."
                )
            else:
                outputs["sfdr_2_status"] = (
                    "No new SFDR 2.0 developments detected."
                )

            status = PhaseStatus.COMPLETED
            records = len(detected_changes)

        except Exception as exc:
            logger.error("ChangeDetection failed: %s", exc, exc_info=True)
            errors.append(f"Change detection failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )


class ImpactAssessmentPhase:
    """
    Phase 2: Impact Assessment.

    Evaluates the impact of detected changes on current disclosures,
    flags required updates, and assesses new data requirements.
    """

    PHASE_NAME = "impact_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute impact assessment phase."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            detection_output = context.get_phase_output("change_detection")
            changes = detection_output.get("detected_changes", [])
            disclosure_state = config.get("current_disclosure_state", {})

            impact_assessments: List[Dict[str, Any]] = []
            affected_disclosures: Dict[str, List[str]] = {
                "ANNEX_II": [],
                "ANNEX_III": [],
                "ANNEX_IV": [],
                "PAI_STATEMENT": [],
            }
            new_data_requirements: List[Dict[str, Any]] = []

            for change in changes:
                # Determine impact level
                category = change.get(
                    "category", ChangeCategory.GUIDANCE_UPDATE.value
                )
                impact_level = self._assess_impact_level(category, change)

                # Identify affected disclosures
                affected_annexes = change.get("affected_annexes", [])
                if not affected_annexes:
                    affected_annexes = self._infer_affected_annexes(
                        category, change
                    )

                for annex in affected_annexes:
                    if annex in affected_disclosures:
                        affected_disclosures[annex].append(
                            change.get("title", "")
                        )

                # Check for new data requirements
                data_reqs = self._check_data_requirements(change)
                new_data_requirements.extend(data_reqs)

                # Disclosure update needed?
                update_required = impact_level in (
                    ImpactLevel.CRITICAL.value,
                    ImpactLevel.HIGH.value,
                    ImpactLevel.MEDIUM.value,
                )

                impact_assessments.append({
                    "change_id": change.get("change_id", ""),
                    "title": change.get("title", ""),
                    "category": category,
                    "impact_level": impact_level,
                    "affected_annexes": affected_annexes,
                    "update_required": update_required,
                    "effective_date": change.get("effective_date"),
                    "new_data_requirements": data_reqs,
                    "assessment_notes": self._generate_assessment_notes(
                        category, impact_level, change
                    ),
                })

            outputs["impact_assessments"] = impact_assessments
            outputs["affected_disclosures"] = {
                k: list(set(v)) for k, v in affected_disclosures.items()
            }
            outputs["new_data_requirements"] = new_data_requirements

            # Count impacts by level
            impact_counts = {level.value: 0 for level in ImpactLevel}
            for assessment in impact_assessments:
                level = assessment.get(
                    "impact_level", ImpactLevel.INFORMATIONAL.value
                )
                impact_counts[level] = impact_counts.get(level, 0) + 1
            outputs["impact_distribution"] = impact_counts

            # Disclosures requiring update
            disclosures_needing_update = sum(
                1 for k, v in affected_disclosures.items() if len(v) > 0
            )
            outputs["disclosures_requiring_update"] = (
                disclosures_needing_update
            )

            critical_count = impact_counts.get(
                ImpactLevel.CRITICAL.value, 0
            )
            high_count = impact_counts.get(ImpactLevel.HIGH.value, 0)

            if critical_count > 0:
                warnings.append(
                    f"{critical_count} CRITICAL impact change(s) requiring "
                    f"immediate disclosure updates"
                )
            if high_count > 0:
                warnings.append(
                    f"{high_count} HIGH impact change(s) requiring "
                    f"disclosure updates"
                )

            status = PhaseStatus.COMPLETED
            records = len(changes)

        except Exception as exc:
            logger.error("ImpactAssessment failed: %s", exc, exc_info=True)
            errors.append(f"Impact assessment failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _assess_impact_level(
        self, category: str, change: Dict[str, Any]
    ) -> str:
        """Assess impact level based on change category."""
        impact_map = {
            ChangeCategory.LEVEL_1_AMENDMENT.value: ImpactLevel.CRITICAL.value,
            ChangeCategory.LEVEL_2_RTS.value: ImpactLevel.HIGH.value,
            ChangeCategory.TAXONOMY_UPDATE.value: ImpactLevel.HIGH.value,
            ChangeCategory.GUIDANCE_UPDATE.value: ImpactLevel.MEDIUM.value,
            ChangeCategory.QA_PUBLICATION.value: ImpactLevel.LOW.value,
            ChangeCategory.SFDR_2_DEVELOPMENT.value: ImpactLevel.MEDIUM.value,
        }
        return impact_map.get(category, ImpactLevel.INFORMATIONAL.value)

    def _infer_affected_annexes(
        self, category: str, change: Dict[str, Any]
    ) -> List[str]:
        """Infer which annexes are affected by a change."""
        affected = change.get("affected_articles", [])
        annexes = []

        if any("6" in a or "8" in a for a in affected):
            annexes.append("ANNEX_II")
        if any("10" in a for a in affected):
            annexes.append("ANNEX_III")
        if any("11" in a for a in affected):
            annexes.append("ANNEX_IV")
        if any("4" in a or "7" in a for a in affected):
            annexes.append("PAI_STATEMENT")

        # If no specific articles, assume all for legislative changes
        if not annexes and category in (
            ChangeCategory.LEVEL_1_AMENDMENT.value,
            ChangeCategory.LEVEL_2_RTS.value,
        ):
            annexes = ["ANNEX_II", "ANNEX_III", "ANNEX_IV"]

        return annexes

    def _check_data_requirements(
        self, change: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check if a change introduces new data requirements."""
        requirements = []
        category = change.get("category", "")
        title = change.get("title", "").lower()

        if "pai" in title or "indicator" in title:
            requirements.append({
                "type": "pai_data",
                "description": "New or modified PAI indicator requirements",
                "source_change": change.get("title", ""),
            })
        if "taxonomy" in title:
            requirements.append({
                "type": "taxonomy_data",
                "description": "Updated taxonomy eligibility/alignment data",
                "source_change": change.get("title", ""),
            })

        return requirements

    def _generate_assessment_notes(
        self, category: str, impact: str, change: Dict[str, Any]
    ) -> str:
        """Generate human-readable assessment notes."""
        if impact == ImpactLevel.CRITICAL.value:
            return (
                f"Legislative amendment requires immediate review of all "
                f"affected disclosures. Consult legal counsel."
            )
        if impact == ImpactLevel.HIGH.value:
            return (
                f"Technical standards update requires disclosure revision. "
                f"Plan update within effective date timeline."
            )
        if impact == ImpactLevel.MEDIUM.value:
            return (
                f"Guidance update may require interpretation changes. "
                f"Review applicability to current disclosures."
            )
        return (
            f"Informational update. Monitor for future developments."
        )


class MigrationPlanningPhase:
    """
    Phase 3: Migration Planning.

    Plans disclosure updates with timeline, assigns responsibilities,
    and creates an implementation roadmap.
    """

    PHASE_NAME = "migration_planning"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute migration planning phase."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            impact_output = context.get_phase_output("impact_assessment")
            assessments = impact_output.get("impact_assessments", [])
            affected_disclosures = impact_output.get(
                "affected_disclosures", {}
            )
            data_requirements = impact_output.get(
                "new_data_requirements", []
            )

            # Generate migration tasks
            migration_tasks: List[Dict[str, Any]] = []
            total_effort_days = 0.0

            for assessment in assessments:
                if not assessment.get("update_required", False):
                    continue

                impact = assessment.get(
                    "impact_level", ImpactLevel.INFORMATIONAL.value
                )
                affected = assessment.get("affected_annexes", [])
                effective_date = assessment.get("effective_date")

                # Estimate effort per impact level
                effort_map = {
                    ImpactLevel.CRITICAL.value: 10.0,
                    ImpactLevel.HIGH.value: 5.0,
                    ImpactLevel.MEDIUM.value: 2.0,
                    ImpactLevel.LOW.value: 0.5,
                }
                effort = effort_map.get(impact, 1.0)

                for annex in affected:
                    task = {
                        "task_id": str(uuid.uuid4()),
                        "change_id": assessment.get("change_id", ""),
                        "change_title": assessment.get("title", ""),
                        "disclosure_type": annex,
                        "impact_level": impact,
                        "description": (
                            f"Update {annex} disclosure in response to: "
                            f"{assessment.get('title', '')}"
                        ),
                        "estimated_effort_days": effort,
                        "deadline": effective_date,
                        "owner": self._assign_owner(annex, impact),
                        "status": MigrationStatus.NOT_STARTED.value,
                        "dependencies": [],
                    }
                    migration_tasks.append(task)
                    total_effort_days += effort

            # Add data sourcing tasks
            for req in data_requirements:
                task = {
                    "task_id": str(uuid.uuid4()),
                    "change_id": "",
                    "change_title": req.get("source_change", ""),
                    "disclosure_type": "DATA_SOURCING",
                    "impact_level": ImpactLevel.HIGH.value,
                    "description": req.get("description", ""),
                    "estimated_effort_days": 3.0,
                    "deadline": None,
                    "owner": "data_team",
                    "status": MigrationStatus.NOT_STARTED.value,
                    "dependencies": [],
                }
                migration_tasks.append(task)
                total_effort_days += 3.0

            # Sort by deadline (earliest first, None last)
            migration_tasks.sort(
                key=lambda t: t.get("deadline") or "9999-12-31"
            )

            outputs["migration_tasks"] = migration_tasks
            outputs["total_tasks"] = len(migration_tasks)
            outputs["total_effort_days"] = round(total_effort_days, 1)

            # Earliest deadline
            deadlines = [
                t.get("deadline") for t in migration_tasks
                if t.get("deadline")
            ]
            outputs["earliest_deadline"] = (
                min(deadlines) if deadlines else None
            )

            # Build implementation roadmap
            roadmap_phases: List[Dict[str, Any]] = []

            # Phase 1: Critical/urgent updates
            critical_tasks = [
                t for t in migration_tasks
                if t.get("impact_level") == ImpactLevel.CRITICAL.value
            ]
            if critical_tasks:
                roadmap_phases.append({
                    "phase": "Immediate Actions",
                    "timeline": "0-2 weeks",
                    "tasks": len(critical_tasks),
                    "effort_days": sum(
                        t.get("estimated_effort_days", 0)
                        for t in critical_tasks
                    ),
                })

            # Phase 2: High-priority updates
            high_tasks = [
                t for t in migration_tasks
                if t.get("impact_level") == ImpactLevel.HIGH.value
            ]
            if high_tasks:
                roadmap_phases.append({
                    "phase": "High Priority Updates",
                    "timeline": "2-6 weeks",
                    "tasks": len(high_tasks),
                    "effort_days": sum(
                        t.get("estimated_effort_days", 0)
                        for t in high_tasks
                    ),
                })

            # Phase 3: Medium/low updates
            remaining_tasks = [
                t for t in migration_tasks
                if t.get("impact_level") in (
                    ImpactLevel.MEDIUM.value, ImpactLevel.LOW.value
                )
            ]
            if remaining_tasks:
                roadmap_phases.append({
                    "phase": "Standard Updates",
                    "timeline": "6-12 weeks",
                    "tasks": len(remaining_tasks),
                    "effort_days": sum(
                        t.get("estimated_effort_days", 0)
                        for t in remaining_tasks
                    ),
                })

            outputs["implementation_roadmap"] = roadmap_phases

            # Resource requirements
            owners: Dict[str, int] = {}
            for task in migration_tasks:
                owner = task.get("owner", "unassigned")
                owners[owner] = owners.get(owner, 0) + 1
            outputs["resource_requirements"] = owners

            if critical_tasks:
                warnings.append(
                    f"{len(critical_tasks)} critical migration task(s) "
                    f"require immediate attention"
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("MigrationPlanning failed: %s", exc, exc_info=True)
            errors.append(f"Migration planning failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

    def _assign_owner(self, disclosure_type: str, impact: str) -> str:
        """Assign default owner for a migration task."""
        if impact == ImpactLevel.CRITICAL.value:
            return "head_of_compliance"
        if disclosure_type in ("ANNEX_II", "ANNEX_III", "ANNEX_IV"):
            return "compliance_officer"
        if disclosure_type == "DATA_SOURCING":
            return "data_team"
        return "compliance_officer"


# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================


class RegulatoryUpdateWorkflow:
    """
    Three-phase regulatory update workflow for SFDR Article 8.

    Orchestrates change detection, impact assessment, and migration
    planning for regulatory change management.

    Example:
        >>> wf = RegulatoryUpdateWorkflow()
        >>> input_data = RegulatoryUpdateInput(
        ...     organization_id="org-123",
        ...     product_name="Green Bond Fund",
        ...     review_date="2026-03-15",
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "regulatory_update"

    PHASE_ORDER = [
        "change_detection",
        "impact_assessment",
        "migration_planning",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize the regulatory update workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "change_detection": ChangeDetectionPhase(),
            "impact_assessment": ImpactAssessmentPhase(),
            "migration_planning": MigrationPlanningPhase(),
        }

    async def run(
        self, input_data: RegulatoryUpdateInput
    ) -> RegulatoryUpdateResult:
        """Execute the complete 3-phase regulatory update workflow."""
        started_at = _utcnow()
        logger.info(
            "Starting regulatory update workflow %s for org=%s product=%s",
            self.workflow_id, input_data.organization_id,
            input_data.product_name,
        )

        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(phase_name, f"Starting: {phase_name}", pct)
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, phase_result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, phase_result.status)

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised unhandled exception: %s",
                    phase_name, exc, exc_info=True,
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    started_at=_utcnow(),
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
            )

        completed_at = _utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )

        return RegulatoryUpdateResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            product_name=summary.get("product_name", ""),
            changes_detected=summary.get("changes_detected", 0),
            critical_changes=summary.get("critical_changes", 0),
            high_impact_changes=summary.get("high_impact_changes", 0),
            disclosures_requiring_update=summary.get(
                "disclosures_requiring_update", 0
            ),
            migration_tasks_created=summary.get(
                "migration_tasks_created", 0
            ),
            estimated_effort_days=summary.get(
                "estimated_effort_days", 0.0
            ),
            earliest_deadline=summary.get("earliest_deadline"),
        )

    def _build_config(
        self, input_data: RegulatoryUpdateInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        config = input_data.model_dump()
        if input_data.detected_changes:
            config["detected_changes"] = [
                c.model_dump() for c in input_data.detected_changes
            ]
            for c in config["detected_changes"]:
                c["source"] = c["source"].value if isinstance(
                    c["source"], ChangeSource
                ) else c["source"]
                c["category"] = c["category"].value if isinstance(
                    c["category"], ChangeCategory
                ) else c["category"]
        if input_data.current_disclosure_state:
            config["current_disclosure_state"] = (
                input_data.current_disclosure_state.model_dump()
            )
        return config

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        config = context.config
        detection = context.get_phase_output("change_detection")
        impact = context.get_phase_output("impact_assessment")
        migration = context.get_phase_output("migration_planning")

        impact_dist = impact.get("impact_distribution", {})

        return {
            "product_name": config.get("product_name", ""),
            "changes_detected": detection.get(
                "total_changes_detected", 0
            ),
            "critical_changes": impact_dist.get(
                ImpactLevel.CRITICAL.value, 0
            ),
            "high_impact_changes": impact_dist.get(
                ImpactLevel.HIGH.value, 0
            ),
            "disclosures_requiring_update": impact.get(
                "disclosures_requiring_update", 0
            ),
            "migration_tasks_created": migration.get("total_tasks", 0),
            "estimated_effort_days": migration.get(
                "total_effort_days", 0.0
            ),
            "earliest_deadline": migration.get("earliest_deadline"),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
