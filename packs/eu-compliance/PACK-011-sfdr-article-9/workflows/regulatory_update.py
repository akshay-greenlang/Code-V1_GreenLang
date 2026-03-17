# -*- coding: utf-8 -*-
"""
Regulatory Update Workflow
================================================

Three-phase workflow for monitoring and responding to SFDR regulatory
changes affecting Article 9 products. Orchestrates change detection,
impact assessment, and migration planning into a single auditable pipeline.

Regulatory Context:
    Per EU SFDR Regulation 2019/2088 and Delegated Regulation 2022/1288 (RTS):
    - The SFDR regulatory framework is subject to ongoing evolution through
      Level 2 RTS amendments, Q&A updates, and ESA guidance.
    - Article 9 products face the highest compliance burden and are most
      affected by regulatory changes.
    - Key regulatory change areas include: SFDR review (Level 1 amendments),
      RTS updates (disclosure templates, PAI definitions), Taxonomy Regulation
      amendments (new environmental objectives, technical screening criteria),
      and NCA supervisory convergence measures.
    - Products must track regulatory changes, assess impact on existing
      disclosures and processes, and implement migration plans with clear
      timelines aligned to regulatory transition periods.

Phases:
    1. ChangeDetection - Detect and catalog applicable regulatory changes
    2. ImpactAssessment - Assess impact of each change on the product
    3. MigrationPlanning - Create migration plan with timelines and actions

Author: GreenLang Team
Version: 1.0.0
"""

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


class ChangeType(str, Enum):
    """Type of regulatory change."""
    LEVEL_1_AMENDMENT = "LEVEL_1_AMENDMENT"
    RTS_UPDATE = "RTS_UPDATE"
    TAXONOMY_AMENDMENT = "TAXONOMY_AMENDMENT"
    NCA_GUIDANCE = "NCA_GUIDANCE"
    ESA_QA = "ESA_QA"
    DELEGATED_ACT = "DELEGATED_ACT"
    TECHNICAL_STANDARD = "TECHNICAL_STANDARD"


class ImpactLevel(str, Enum):
    """Impact level of a regulatory change on the product."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MigrationPriority(str, Enum):
    """Migration action priority."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


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


class RegulatoryUpdateInput(BaseModel):
    """Input configuration for the regulatory update workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    product_isin: Optional[str] = Field(None, description="ISIN if applicable")
    assessment_date: str = Field(
        ..., description="Assessment date YYYY-MM-DD"
    )
    current_sfdr_version: str = Field(
        default="2022/1288",
        description="Current RTS version in use"
    )
    regulatory_changes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of regulatory changes to assess"
    )
    affected_disclosures: List[str] = Field(
        default_factory=list,
        description="Current disclosure documents that may be affected"
    )
    affected_processes: List[str] = Field(
        default_factory=list,
        description="Current processes that may need updating"
    )
    compliance_deadline: Optional[str] = Field(
        None, description="Regulatory compliance deadline YYYY-MM-DD"
    )
    available_resources: Dict[str, int] = Field(
        default_factory=dict,
        description="Available team resources (role: FTE count)"
    )
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("assessment_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date is valid ISO format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be YYYY-MM-DD format")
        return v

    @field_validator("compliance_deadline")
    @classmethod
    def validate_deadline_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate deadline is valid ISO format if provided."""
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError(
                    "compliance_deadline must be YYYY-MM-DD format"
                )
        return v


class RegulatoryUpdateResult(WorkflowResult):
    """Complete result from the regulatory update workflow."""
    product_name: str = Field(default="")
    changes_detected: int = Field(default=0)
    changes_applicable: int = Field(default=0)
    high_impact_changes: int = Field(default=0)
    critical_impact_changes: int = Field(default=0)
    disclosures_affected: int = Field(default=0)
    processes_affected: int = Field(default=0)
    migration_actions: int = Field(default=0)
    urgent_actions: int = Field(default=0)
    estimated_effort_days: float = Field(default=0.0)


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================


class ChangeDetectionPhase:
    """
    Phase 1: Change Detection.

    Detects and catalogs regulatory changes applicable to the Article 9
    product, classifying each by type, source, effective date, and
    preliminary relevance to the product's configuration.
    """

    PHASE_NAME = "change_detection"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute change detection phase.

        Args:
            context: Workflow context with regulatory change inputs.

        Returns:
            PhaseResult with cataloged regulatory changes.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            product_name = config.get("product_name", "")
            changes_input = config.get("regulatory_changes", [])
            current_version = config.get(
                "current_sfdr_version", "2022/1288"
            )

            outputs["product_name"] = product_name
            outputs["current_sfdr_version"] = current_version

            # Catalog and classify each change
            cataloged_changes: List[Dict[str, Any]] = []

            for idx, change in enumerate(changes_input):
                change_id = change.get(
                    "change_id", f"RC-{idx + 1:03d}"
                )
                change_type = change.get(
                    "change_type", ChangeType.RTS_UPDATE.value
                )
                title = change.get("title", f"Change {idx + 1}")
                description = change.get("description", "")
                source = change.get("source", "EU Official Journal")
                effective_date = change.get("effective_date", "")
                transition_period_days = change.get(
                    "transition_period_days", 180
                )
                affected_articles = change.get(
                    "affected_articles", []
                )

                # Check if change is applicable to Article 9
                is_applicable = (
                    "9" in affected_articles
                    or "all" in affected_articles
                    or not affected_articles  # Empty = applies to all
                )

                # Determine preliminary scope
                scope_areas = change.get("scope_areas", [])
                if not scope_areas:
                    scope_areas = self._infer_scope(change_type, title)

                cataloged = {
                    "change_id": change_id,
                    "change_type": change_type,
                    "title": title,
                    "description": description,
                    "source": source,
                    "effective_date": effective_date,
                    "transition_period_days": transition_period_days,
                    "affected_articles": affected_articles,
                    "is_applicable": is_applicable,
                    "scope_areas": scope_areas,
                    "cataloged_at": _utcnow().isoformat(),
                }
                cataloged_changes.append(cataloged)

            applicable_count = sum(
                1 for c in cataloged_changes if c["is_applicable"]
            )

            outputs["cataloged_changes"] = cataloged_changes
            outputs["total_changes"] = len(cataloged_changes)
            outputs["applicable_changes"] = applicable_count
            outputs["non_applicable_changes"] = (
                len(cataloged_changes) - applicable_count
            )
            outputs["change_types_summary"] = {}
            for c in cataloged_changes:
                ct = c["change_type"]
                outputs["change_types_summary"][ct] = (
                    outputs["change_types_summary"].get(ct, 0) + 1
                )

            if applicable_count == 0:
                warnings.append(
                    "No applicable regulatory changes detected for "
                    "this Article 9 product"
                )

            status = PhaseStatus.COMPLETED
            records = len(cataloged_changes)

        except Exception as exc:
            logger.error(
                "ChangeDetection failed: %s", exc, exc_info=True
            )
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

    def _infer_scope(
        self, change_type: str, title: str
    ) -> List[str]:
        """Infer scope areas from change type and title."""
        scope = []
        title_lower = title.lower()

        if "disclosure" in title_lower or "template" in title_lower:
            scope.append("disclosures")
        if "pai" in title_lower or "adverse" in title_lower:
            scope.append("pai_indicators")
        if "taxonomy" in title_lower:
            scope.append("taxonomy_alignment")
        if "benchmark" in title_lower:
            scope.append("benchmarks")
        if "dnsh" in title_lower:
            scope.append("dnsh_assessment")
        if "data" in title_lower:
            scope.append("data_requirements")

        if change_type == ChangeType.LEVEL_1_AMENDMENT.value:
            scope.append("core_regulation")
        elif change_type == ChangeType.TAXONOMY_AMENDMENT.value:
            scope.append("taxonomy_alignment")

        return scope if scope else ["general"]


class ImpactAssessmentPhase:
    """
    Phase 2: Impact Assessment.

    Assesses the impact of each applicable regulatory change on the
    product's disclosures, processes, data requirements, and systems.
    Assigns impact levels and identifies specific areas requiring changes.
    """

    PHASE_NAME = "impact_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute impact assessment phase.

        Args:
            context: Workflow context with cataloged changes.

        Returns:
            PhaseResult with impact assessment for each change.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            detection_output = context.get_phase_output(
                "change_detection"
            )
            cataloged = detection_output.get("cataloged_changes", [])
            affected_disclosures = config.get(
                "affected_disclosures", []
            )
            affected_processes = config.get("affected_processes", [])

            impact_assessments: List[Dict[str, Any]] = []
            high_impact_count = 0
            critical_impact_count = 0
            all_affected_disclosures: List[str] = []
            all_affected_processes: List[str] = []

            for change in cataloged:
                if not change.get("is_applicable", False):
                    continue

                change_id = change["change_id"]
                change_type = change["change_type"]
                scope_areas = change.get("scope_areas", [])

                # Assess impact level based on change type and scope
                impact_level = self._assess_impact_level(
                    change_type, scope_areas
                )

                # Identify affected disclosures
                disc_affected = []
                for disc in affected_disclosures:
                    for scope in scope_areas:
                        if self._disclosure_affected(disc, scope):
                            disc_affected.append(disc)
                            break

                all_affected_disclosures.extend(disc_affected)

                # Identify affected processes
                proc_affected = []
                for proc in affected_processes:
                    for scope in scope_areas:
                        if self._process_affected(proc, scope):
                            proc_affected.append(proc)
                            break

                all_affected_processes.extend(proc_affected)

                # Estimate effort
                effort_days = self._estimate_effort(
                    impact_level, len(disc_affected), len(proc_affected)
                )

                if impact_level == ImpactLevel.HIGH.value:
                    high_impact_count += 1
                elif impact_level == ImpactLevel.CRITICAL.value:
                    critical_impact_count += 1
                    high_impact_count += 1

                assessment = {
                    "change_id": change_id,
                    "title": change["title"],
                    "impact_level": impact_level,
                    "scope_areas": scope_areas,
                    "disclosures_affected": disc_affected,
                    "processes_affected": proc_affected,
                    "estimated_effort_days": effort_days,
                    "key_impacts": self._describe_impacts(
                        change_type, scope_areas
                    ),
                    "transition_period_days": change.get(
                        "transition_period_days", 180
                    ),
                }
                impact_assessments.append(assessment)

            # Deduplicate affected items
            unique_disclosures = list(set(all_affected_disclosures))
            unique_processes = list(set(all_affected_processes))
            total_effort = sum(
                a["estimated_effort_days"] for a in impact_assessments
            )

            outputs["impact_assessments"] = impact_assessments
            outputs["changes_assessed"] = len(impact_assessments)
            outputs["high_impact_changes"] = high_impact_count
            outputs["critical_impact_changes"] = critical_impact_count
            outputs["disclosures_affected"] = unique_disclosures
            outputs["disclosures_affected_count"] = len(unique_disclosures)
            outputs["processes_affected"] = unique_processes
            outputs["processes_affected_count"] = len(unique_processes)
            outputs["total_estimated_effort_days"] = round(
                total_effort, 1
            )

            if critical_impact_count > 0:
                warnings.append(
                    f"{critical_impact_count} critical-impact regulatory "
                    f"change(s) require immediate attention"
                )

            status = PhaseStatus.COMPLETED
            records = len(impact_assessments)

        except Exception as exc:
            logger.error(
                "ImpactAssessment failed: %s", exc, exc_info=True
            )
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
        self, change_type: str, scope_areas: List[str]
    ) -> str:
        """Assess impact level based on change type and scope."""
        # Level 1 amendments are always high or critical impact
        if change_type == ChangeType.LEVEL_1_AMENDMENT.value:
            return ImpactLevel.CRITICAL.value

        # Taxonomy amendments affecting alignment are high impact
        if change_type == ChangeType.TAXONOMY_AMENDMENT.value:
            if "taxonomy_alignment" in scope_areas:
                return ImpactLevel.HIGH.value

        # RTS updates with disclosure scope are medium-high
        if change_type == ChangeType.RTS_UPDATE.value:
            if "disclosures" in scope_areas:
                return ImpactLevel.HIGH.value
            if "pai_indicators" in scope_areas:
                return ImpactLevel.MEDIUM.value

        # NCA guidance and Q&A are typically lower impact
        if change_type in (
            ChangeType.NCA_GUIDANCE.value,
            ChangeType.ESA_QA.value,
        ):
            return ImpactLevel.LOW.value

        return ImpactLevel.MEDIUM.value

    def _disclosure_affected(
        self, disclosure: str, scope: str
    ) -> bool:
        """Check if a disclosure document is affected by scope area."""
        scope_disclosure_map = {
            "disclosures": True,
            "pai_indicators": "pai" in disclosure.lower(),
            "taxonomy_alignment": "taxonomy" in disclosure.lower(),
            "benchmarks": "benchmark" in disclosure.lower(),
            "dnsh_assessment": "dnsh" in disclosure.lower(),
            "core_regulation": True,
        }
        return scope_disclosure_map.get(scope, False)

    def _process_affected(
        self, process: str, scope: str
    ) -> bool:
        """Check if a process is affected by scope area."""
        scope_process_map = {
            "data_requirements": "data" in process.lower(),
            "pai_indicators": "pai" in process.lower(),
            "taxonomy_alignment": "taxonomy" in process.lower(),
            "benchmarks": "benchmark" in process.lower(),
            "dnsh_assessment": "dnsh" in process.lower(),
            "core_regulation": True,
        }
        return scope_process_map.get(scope, False)

    def _estimate_effort(
        self,
        impact_level: str,
        disclosures_count: int,
        processes_count: int,
    ) -> float:
        """Estimate migration effort in person-days."""
        base_effort = {
            ImpactLevel.NONE.value: 0.0,
            ImpactLevel.LOW.value: 2.0,
            ImpactLevel.MEDIUM.value: 5.0,
            ImpactLevel.HIGH.value: 15.0,
            ImpactLevel.CRITICAL.value: 30.0,
        }
        effort = base_effort.get(impact_level, 5.0)
        effort += disclosures_count * 2.0
        effort += processes_count * 3.0
        return round(effort, 1)

    def _describe_impacts(
        self, change_type: str, scope_areas: List[str]
    ) -> List[str]:
        """Generate human-readable impact descriptions."""
        impacts = []

        if "disclosures" in scope_areas:
            impacts.append(
                "Disclosure templates may require updates"
            )
        if "pai_indicators" in scope_areas:
            impacts.append(
                "PAI indicator definitions or calculation "
                "methodology changes"
            )
        if "taxonomy_alignment" in scope_areas:
            impacts.append(
                "Taxonomy alignment criteria or thresholds "
                "may change"
            )
        if "benchmarks" in scope_areas:
            impacts.append(
                "Benchmark methodology or designation "
                "requirements updated"
            )
        if "core_regulation" in scope_areas:
            impacts.append(
                "Core SFDR regulation changes affecting "
                "Article 9 classification"
            )

        return impacts if impacts else ["General regulatory update"]


class MigrationPlanningPhase:
    """
    Phase 3: Migration Planning.

    Creates a structured migration plan with specific actions, timelines,
    resource requirements, and dependencies for implementing required
    regulatory changes.
    """

    PHASE_NAME = "migration_planning"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute migration planning phase.

        Args:
            context: Workflow context with impact assessment results.

        Returns:
            PhaseResult with migration plan and action items.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            impact_output = context.get_phase_output(
                "impact_assessment"
            )
            assessments = impact_output.get(
                "impact_assessments", []
            )
            compliance_deadline = config.get("compliance_deadline")
            available_resources = config.get(
                "available_resources", {}
            )

            migration_actions: List[Dict[str, Any]] = []

            for assessment in assessments:
                change_id = assessment["change_id"]
                title = assessment["title"]
                impact_level = assessment["impact_level"]
                disclosures = assessment.get(
                    "disclosures_affected", []
                )
                processes = assessment.get(
                    "processes_affected", []
                )
                transition_days = assessment.get(
                    "transition_period_days", 180
                )

                # Determine priority based on impact and deadline
                if impact_level == ImpactLevel.CRITICAL.value:
                    priority = MigrationPriority.URGENT.value
                elif impact_level == ImpactLevel.HIGH.value:
                    priority = MigrationPriority.HIGH.value
                elif impact_level == ImpactLevel.MEDIUM.value:
                    priority = MigrationPriority.MEDIUM.value
                else:
                    priority = MigrationPriority.LOW.value

                # Generate disclosure update actions
                for disc in disclosures:
                    migration_actions.append({
                        "action_id": str(uuid.uuid4()),
                        "change_id": change_id,
                        "change_title": title,
                        "priority": priority,
                        "action_type": "disclosure_update",
                        "description": (
                            f"Update disclosure '{disc}' per "
                            f"regulatory change: {title}"
                        ),
                        "responsible_team": "Compliance",
                        "estimated_days": 2.0,
                        "deadline": compliance_deadline,
                        "transition_period_days": transition_days,
                        "dependencies": [],
                        "status": "NOT_STARTED",
                    })

                # Generate process update actions
                for proc in processes:
                    migration_actions.append({
                        "action_id": str(uuid.uuid4()),
                        "change_id": change_id,
                        "change_title": title,
                        "priority": priority,
                        "action_type": "process_update",
                        "description": (
                            f"Update process '{proc}' per "
                            f"regulatory change: {title}"
                        ),
                        "responsible_team": "Operations",
                        "estimated_days": 3.0,
                        "deadline": compliance_deadline,
                        "transition_period_days": transition_days,
                        "dependencies": [],
                        "status": "NOT_STARTED",
                    })

                # Generate system/data update action if needed
                if impact_level in (
                    ImpactLevel.HIGH.value,
                    ImpactLevel.CRITICAL.value,
                ):
                    migration_actions.append({
                        "action_id": str(uuid.uuid4()),
                        "change_id": change_id,
                        "change_title": title,
                        "priority": priority,
                        "action_type": "system_update",
                        "description": (
                            f"Update system configuration and data "
                            f"pipelines for: {title}"
                        ),
                        "responsible_team": "Technology",
                        "estimated_days": 5.0,
                        "deadline": compliance_deadline,
                        "transition_period_days": transition_days,
                        "dependencies": [],
                        "status": "NOT_STARTED",
                    })

                # Generate testing action
                if impact_level != ImpactLevel.NONE.value:
                    migration_actions.append({
                        "action_id": str(uuid.uuid4()),
                        "change_id": change_id,
                        "change_title": title,
                        "priority": priority,
                        "action_type": "testing_validation",
                        "description": (
                            f"Test and validate all changes "
                            f"related to: {title}"
                        ),
                        "responsible_team": "QA",
                        "estimated_days": 2.0,
                        "deadline": compliance_deadline,
                        "transition_period_days": transition_days,
                        "dependencies": [
                            "disclosure_update",
                            "process_update",
                        ],
                        "status": "NOT_STARTED",
                    })

            # Sort by priority
            priority_order = {
                MigrationPriority.URGENT.value: 0,
                MigrationPriority.HIGH.value: 1,
                MigrationPriority.MEDIUM.value: 2,
                MigrationPriority.LOW.value: 3,
            }
            migration_actions.sort(
                key=lambda a: priority_order.get(
                    a.get("priority", "LOW"), 99
                )
            )

            # Calculate totals
            urgent_count = sum(
                1 for a in migration_actions
                if a["priority"] == MigrationPriority.URGENT.value
            )
            total_effort = sum(
                a["estimated_days"] for a in migration_actions
            )

            # Resource feasibility check
            total_fte = sum(available_resources.values())
            if total_effort > 0 and total_fte > 0:
                required_weeks = total_effort / (total_fte * 5)
                feasible = required_weeks <= 12
            else:
                required_weeks = 0.0
                feasible = True

            outputs["migration_actions"] = migration_actions
            outputs["actions_count"] = len(migration_actions)
            outputs["urgent_actions"] = urgent_count
            outputs["total_estimated_effort_days"] = round(
                total_effort, 1
            )
            outputs["estimated_weeks"] = round(required_weeks, 1)
            outputs["resource_feasible"] = feasible
            outputs["available_resources"] = available_resources
            outputs["compliance_deadline"] = compliance_deadline
            outputs["action_types_summary"] = {
                "disclosure_update": sum(
                    1 for a in migration_actions
                    if a["action_type"] == "disclosure_update"
                ),
                "process_update": sum(
                    1 for a in migration_actions
                    if a["action_type"] == "process_update"
                ),
                "system_update": sum(
                    1 for a in migration_actions
                    if a["action_type"] == "system_update"
                ),
                "testing_validation": sum(
                    1 for a in migration_actions
                    if a["action_type"] == "testing_validation"
                ),
            }
            outputs["generated_at"] = _utcnow().isoformat()

            if not feasible:
                warnings.append(
                    f"Migration plan requires {required_weeks:.1f} weeks "
                    f"with current resources ({total_fte} FTE). "
                    f"Additional resources may be needed."
                )

            if urgent_count > 0:
                warnings.append(
                    f"{urgent_count} urgent migration action(s) "
                    f"require immediate scheduling"
                )

            status = PhaseStatus.COMPLETED
            records = len(migration_actions)

        except Exception as exc:
            logger.error(
                "MigrationPlanning failed: %s", exc, exc_info=True
            )
            errors.append(f"Migration planning failed: {str(exc)}")
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


# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================


class RegulatoryUpdateWorkflow:
    """
    Three-phase regulatory change management workflow for Article 9.

    Orchestrates the complete regulatory update pipeline from change
    detection through impact assessment and migration planning.
    Supports checkpoint/resume and phase skipping.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered mapping of phase name to executor instance.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = RegulatoryUpdateWorkflow()
        >>> input_data = RegulatoryUpdateInput(
        ...     organization_id="org-123",
        ...     product_name="Climate Solutions Fund",
        ...     assessment_date="2026-03-01",
        ...     regulatory_changes=[
        ...         {
        ...             "title": "SFDR RTS Amendment 2026",
        ...             "change_type": "RTS_UPDATE",
        ...             "affected_articles": ["9"],
        ...         }
        ...     ],
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
        """
        Initialize the regulatory update workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
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
        """
        Execute the complete 3-phase regulatory update workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            RegulatoryUpdateResult with per-phase details and summary.
        """
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
                logger.info(
                    "Phase '%s' already completed, skipping",
                    phase_name,
                )
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(
                phase_name, f"Starting: {phase_name}", pct
            )
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(
                        phase_name, phase_result.outputs
                    )
                    context.mark_phase(
                        phase_name, PhaseStatus.COMPLETED
                    )
                else:
                    context.mark_phase(
                        phase_name, phase_result.status
                    )
                    if phase_name == "change_detection":
                        overall_status = WorkflowStatus.FAILED
                        logger.error(
                            "Critical phase '%s' failed, aborting",
                            phase_name,
                        )
                        break

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
                WorkflowStatus.COMPLETED if all_ok
                else WorkflowStatus.PARTIAL
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
        logger.info(
            "Regulatory update workflow %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
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
            changes_applicable=summary.get("changes_applicable", 0),
            high_impact_changes=summary.get(
                "high_impact_changes", 0
            ),
            critical_impact_changes=summary.get(
                "critical_impact_changes", 0
            ),
            disclosures_affected=summary.get(
                "disclosures_affected", 0
            ),
            processes_affected=summary.get(
                "processes_affected", 0
            ),
            migration_actions=summary.get("migration_actions", 0),
            urgent_actions=summary.get("urgent_actions", 0),
            estimated_effort_days=summary.get(
                "estimated_effort_days", 0.0
            ),
        )

    def _build_config(
        self, input_data: RegulatoryUpdateInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        return input_data.model_dump()

    def _build_summary(
        self, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        detection = context.get_phase_output("change_detection")
        impact = context.get_phase_output("impact_assessment")
        migration = context.get_phase_output("migration_planning")

        return {
            "product_name": detection.get("product_name", ""),
            "changes_detected": detection.get("total_changes", 0),
            "changes_applicable": detection.get(
                "applicable_changes", 0
            ),
            "high_impact_changes": impact.get(
                "high_impact_changes", 0
            ),
            "critical_impact_changes": impact.get(
                "critical_impact_changes", 0
            ),
            "disclosures_affected": impact.get(
                "disclosures_affected_count", 0
            ),
            "processes_affected": impact.get(
                "processes_affected_count", 0
            ),
            "migration_actions": migration.get("actions_count", 0),
            "urgent_actions": migration.get("urgent_actions", 0),
            "estimated_effort_days": migration.get(
                "total_estimated_effort_days", 0.0
            ),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug(
                    "Progress callback failed for phase=%s", phase
                )
