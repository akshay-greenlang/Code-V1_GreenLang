# -*- coding: utf-8 -*-
"""
Stakeholder Engagement Workflow
====================================

5-phase workflow for stakeholder engagement within PACK-015 Double
Materiality Pack. Implements stakeholder identification, mapping,
consultation, synthesis, and validation per ESRS 1 sections 22-23
and EFRAG IG-1 guidance on affected stakeholder groups.

Phases:
    1. StakeholderIdentification   -- Identify affected stakeholder groups
    2. StakeholderMapping          -- Build influence vs impact priority matrix
    3. ConsultationExecution       -- Record and process consultation data
    4. FindingSynthesis            -- Synthesize across stakeholder groups
    5. Validation                  -- Validate completeness per ESRS 1

Author: GreenLang Team
Version: 15.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class StakeholderCategory(str, Enum):
    """Stakeholder group categories per ESRS 1."""
    EMPLOYEES = "employees"
    WORKERS_VALUE_CHAIN = "workers_value_chain"
    LOCAL_COMMUNITIES = "local_communities"
    CONSUMERS = "consumers"
    INVESTORS = "investors"
    REGULATORS = "regulators"
    NGOS = "ngos"
    SUPPLIERS = "suppliers"
    CUSTOMERS = "customers"
    INDIGENOUS_PEOPLES = "indigenous_peoples"
    CIVIL_SOCIETY = "civil_society"
    BUSINESS_PARTNERS = "business_partners"


class EngagementMethod(str, Enum):
    """Method used for stakeholder consultation."""
    SURVEY = "survey"
    INTERVIEW = "interview"
    FOCUS_GROUP = "focus_group"
    WORKSHOP = "workshop"
    PUBLIC_HEARING = "public_hearing"
    WRITTEN_SUBMISSION = "written_submission"
    ADVISORY_PANEL = "advisory_panel"
    GRIEVANCE_MECHANISM = "grievance_mechanism"


class PriorityLevel(str, Enum):
    """Stakeholder priority based on influence-impact matrix."""
    KEY_PLAYER = "key_player"           # High influence, high impact
    KEEP_SATISFIED = "keep_satisfied"   # High influence, low impact
    KEEP_INFORMED = "keep_informed"     # Low influence, high impact
    MONITOR = "monitor"                 # Low influence, low impact


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class Stakeholder(BaseModel):
    """An identified stakeholder group or entity."""
    stakeholder_id: str = Field(default_factory=lambda: f"sh-{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="Stakeholder name or group label")
    category: StakeholderCategory = Field(...)
    description: str = Field(default="")
    influence_score: float = Field(default=3.0, ge=0.0, le=5.0)
    impact_score: float = Field(default=3.0, ge=0.0, le=5.0)
    priority: PriorityLevel = Field(default=PriorityLevel.MONITOR)
    esrs_topics_relevant: List[str] = Field(default_factory=list)
    contact_info: str = Field(default="")
    region: str = Field(default="")


class ConsultationRecord(BaseModel):
    """Record of a stakeholder consultation event."""
    consultation_id: str = Field(default_factory=lambda: f"con-{uuid.uuid4().hex[:8]}")
    stakeholder_id: str = Field(default="")
    stakeholder_name: str = Field(default="")
    method: EngagementMethod = Field(default=EngagementMethod.SURVEY)
    date: str = Field(default="", description="ISO date string")
    participants_count: int = Field(default=0, ge=0)
    topics_discussed: List[str] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    matters_raised: List[str] = Field(default_factory=list)
    satisfaction_score: float = Field(default=0.0, ge=0.0, le=5.0)
    evidence_reference: str = Field(default="")


class StakeholderFinding(BaseModel):
    """Synthesized finding from stakeholder consultations."""
    finding_id: str = Field(default_factory=lambda: f"sf-{uuid.uuid4().hex[:8]}")
    topic: str = Field(default="", description="Sustainability topic")
    esrs_topic: str = Field(default="")
    description: str = Field(default="")
    stakeholder_groups_count: int = Field(default=0, ge=0)
    frequency: int = Field(default=0, ge=0, description="Times raised across consultations")
    priority_score: float = Field(default=0.0, ge=0.0, le=5.0)
    supporting_stakeholders: List[str] = Field(default_factory=list)


class ValidationCheck(BaseModel):
    """ESRS 1 completeness validation check."""
    check_id: str = Field(default="")
    check_name: str = Field(default="")
    esrs_reference: str = Field(default="")
    passed: bool = Field(default=False)
    details: str = Field(default="")


class StakeholderEngagementInput(BaseModel):
    """Input data model for StakeholderEngagementWorkflow."""
    stakeholders: List[Stakeholder] = Field(
        default_factory=list, description="Identified stakeholders"
    )
    consultations: List[ConsultationRecord] = Field(
        default_factory=list, description="Consultation records"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    required_categories: List[StakeholderCategory] = Field(
        default_factory=lambda: [
            StakeholderCategory.EMPLOYEES,
            StakeholderCategory.WORKERS_VALUE_CHAIN,
            StakeholderCategory.LOCAL_COMMUNITIES,
            StakeholderCategory.CONSUMERS,
        ],
        description="Minimum required stakeholder categories per ESRS 1",
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class StakeholderEngagementResult(BaseModel):
    """Complete result from stakeholder engagement workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="stakeholder_engagement")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    stakeholders_identified: int = Field(default=0, ge=0)
    consultations_recorded: int = Field(default=0, ge=0)
    total_participants: int = Field(default=0, ge=0)
    synthesized_findings: List[StakeholderFinding] = Field(default_factory=list)
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    validation_passed: bool = Field(default=False)
    category_coverage: Dict[str, int] = Field(default_factory=dict)
    priority_distribution: Dict[str, int] = Field(default_factory=dict)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# ESRS 1 REQUIRED STAKEHOLDER CATEGORIES
# =============================================================================

ESRS1_STAKEHOLDER_REQUIREMENTS = {
    "affected_stakeholders": [
        "employees", "workers_value_chain", "local_communities",
        "consumers", "indigenous_peoples",
    ],
    "users_of_statements": [
        "investors", "regulators",
    ],
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class StakeholderEngagementWorkflow:
    """
    5-phase stakeholder engagement workflow.

    Implements stakeholder identification, influence-impact matrix mapping,
    consultation recording, cross-group synthesis, and ESRS 1 sections
    22-23 completeness validation. Ensures all required stakeholder
    categories are covered and findings are prioritized.

    Zero-hallucination: priority scoring uses deterministic influence x impact
    matrix. No LLM in numeric paths.

    Example:
        >>> wf = StakeholderEngagementWorkflow()
        >>> inp = StakeholderEngagementInput(stakeholders=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.validation_passed
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize StakeholderEngagementWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._stakeholders: List[Stakeholder] = []
        self._consultations: List[ConsultationRecord] = []
        self._findings: List[StakeholderFinding] = []
        self._validation_checks: List[ValidationCheck] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[StakeholderEngagementInput] = None,
        stakeholders: Optional[List[Stakeholder]] = None,
        consultations: Optional[List[ConsultationRecord]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> StakeholderEngagementResult:
        """
        Execute the 5-phase stakeholder engagement workflow.

        Args:
            input_data: Full input model (preferred).
            stakeholders: Stakeholder records (fallback).
            consultations: Consultation records (fallback).
            config: Configuration overrides.

        Returns:
            StakeholderEngagementResult with findings, validation, coverage.
        """
        if input_data is None:
            input_data = StakeholderEngagementInput(
                stakeholders=stakeholders or [],
                consultations=consultations or [],
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting stakeholder engagement %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(
                await self._phase_stakeholder_identification(input_data)
            )
            phase_results.append(
                await self._phase_stakeholder_mapping(input_data)
            )
            phase_results.append(
                await self._phase_consultation_execution(input_data)
            )
            phase_results.append(
                await self._phase_finding_synthesis(input_data)
            )
            phase_results.append(
                await self._phase_validation(input_data)
            )
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error(
                "Stakeholder engagement workflow failed: %s", exc, exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        category_cov: Dict[str, int] = {}
        for sh in self._stakeholders:
            category_cov[sh.category.value] = category_cov.get(sh.category.value, 0) + 1
        priority_dist: Dict[str, int] = {}
        for sh in self._stakeholders:
            priority_dist[sh.priority.value] = priority_dist.get(sh.priority.value, 0) + 1

        all_passed = all(vc.passed for vc in self._validation_checks) if self._validation_checks else False

        result = StakeholderEngagementResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            stakeholders_identified=len(self._stakeholders),
            consultations_recorded=len(self._consultations),
            total_participants=sum(c.participants_count for c in self._consultations),
            synthesized_findings=self._findings,
            validation_checks=self._validation_checks,
            validation_passed=all_passed,
            category_coverage=category_cov,
            priority_distribution=priority_dist,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Stakeholder engagement %s completed in %.2fs: %d stakeholders, %d findings",
            self.workflow_id, elapsed, len(self._stakeholders), len(self._findings),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Stakeholder Identification
    # -------------------------------------------------------------------------

    async def _phase_stakeholder_identification(
        self, input_data: StakeholderEngagementInput,
    ) -> PhaseResult:
        """Identify affected stakeholder groups."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._stakeholders = list(input_data.stakeholders)

        category_counts: Dict[str, int] = {}
        for sh in self._stakeholders:
            category_counts[sh.category.value] = (
                category_counts.get(sh.category.value, 0) + 1
            )

        outputs["stakeholders_identified"] = len(self._stakeholders)
        outputs["categories_represented"] = len(category_counts)
        outputs["category_distribution"] = category_counts

        # Check required categories
        present_cats = set(category_counts.keys())
        required_cats = set(c.value for c in input_data.required_categories)
        missing_cats = required_cats - present_cats
        if missing_cats:
            warnings.append(
                f"Missing required stakeholder categories: {', '.join(sorted(missing_cats))}"
            )
        outputs["missing_required_categories"] = sorted(list(missing_cats))

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 StakeholderIdentification: %d stakeholders, %d categories",
            len(self._stakeholders), len(category_counts),
        )
        return PhaseResult(
            phase_name="stakeholder_identification", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Stakeholder Mapping
    # -------------------------------------------------------------------------

    async def _phase_stakeholder_mapping(
        self, input_data: StakeholderEngagementInput,
    ) -> PhaseResult:
        """Build influence vs impact priority matrix."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        for sh in self._stakeholders:
            sh.priority = self._classify_priority(sh.influence_score, sh.impact_score)

        priority_counts: Dict[str, int] = {}
        for sh in self._stakeholders:
            priority_counts[sh.priority.value] = (
                priority_counts.get(sh.priority.value, 0) + 1
            )

        outputs["priority_distribution"] = priority_counts
        outputs["key_players"] = [
            sh.name for sh in self._stakeholders
            if sh.priority == PriorityLevel.KEY_PLAYER
        ]
        outputs["total_mapped"] = len(self._stakeholders)

        if not outputs["key_players"]:
            warnings.append(
                "No key players identified in stakeholder matrix. "
                "Review influence and impact scores."
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 StakeholderMapping: %d mapped, %d key players",
            len(self._stakeholders), len(outputs["key_players"]),
        )
        return PhaseResult(
            phase_name="stakeholder_mapping", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _classify_priority(
        self, influence: float, impact: float, threshold: float = 3.0,
    ) -> PriorityLevel:
        """Classify stakeholder priority based on influence-impact matrix."""
        if influence >= threshold and impact >= threshold:
            return PriorityLevel.KEY_PLAYER
        elif influence >= threshold and impact < threshold:
            return PriorityLevel.KEEP_SATISFIED
        elif influence < threshold and impact >= threshold:
            return PriorityLevel.KEEP_INFORMED
        else:
            return PriorityLevel.MONITOR

    # -------------------------------------------------------------------------
    # Phase 3: Consultation Execution
    # -------------------------------------------------------------------------

    async def _phase_consultation_execution(
        self, input_data: StakeholderEngagementInput,
    ) -> PhaseResult:
        """Record and process consultation data."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._consultations = list(input_data.consultations)
        total_participants = sum(c.participants_count for c in self._consultations)

        method_counts: Dict[str, int] = {}
        for c in self._consultations:
            method_counts[c.method.value] = method_counts.get(c.method.value, 0) + 1

        # Check coverage: which stakeholders have consultations
        consulted_ids = set(c.stakeholder_id for c in self._consultations if c.stakeholder_id)
        all_ids = set(sh.stakeholder_id for sh in self._stakeholders)
        unconsulted = all_ids - consulted_ids

        outputs["consultations_recorded"] = len(self._consultations)
        outputs["total_participants"] = total_participants
        outputs["method_distribution"] = method_counts
        outputs["stakeholders_consulted"] = len(consulted_ids)
        outputs["stakeholders_unconsulted"] = len(unconsulted)
        outputs["coverage_pct"] = round(
            (len(consulted_ids) / len(all_ids) * 100)
            if all_ids else 0.0, 1,
        )

        if unconsulted:
            warnings.append(
                f"{len(unconsulted)} stakeholder(s) have no recorded consultation"
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ConsultationExecution: %d consultations, %d participants",
            len(self._consultations), total_participants,
        )
        return PhaseResult(
            phase_name="consultation_execution", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Finding Synthesis
    # -------------------------------------------------------------------------

    async def _phase_finding_synthesis(
        self, input_data: StakeholderEngagementInput,
    ) -> PhaseResult:
        """Synthesize findings across stakeholder groups."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._findings = []

        # Aggregate matters raised across consultations
        matter_details: Dict[str, Dict[str, Any]] = {}
        for consultation in self._consultations:
            for matter in consultation.matters_raised:
                if matter not in matter_details:
                    matter_details[matter] = {
                        "frequency": 0,
                        "stakeholder_groups": set(),
                        "topics": set(),
                    }
                matter_details[matter]["frequency"] += 1
                matter_details[matter]["stakeholder_groups"].add(
                    consultation.stakeholder_name or consultation.stakeholder_id
                )
                for topic in consultation.topics_discussed:
                    matter_details[matter]["topics"].add(topic)

        # Create findings sorted by frequency
        sorted_matters = sorted(
            matter_details.items(), key=lambda x: x[1]["frequency"], reverse=True,
        )
        for matter_name, details in sorted_matters:
            groups_count = len(details["stakeholder_groups"])
            frequency = details["frequency"]
            # Priority score: normalized frequency x group coverage
            max_freq = sorted_matters[0][1]["frequency"] if sorted_matters else 1
            priority = min(
                (frequency / max_freq) * 3.0 + (groups_count / max(len(self._stakeholders), 1)) * 2.0,
                5.0,
            )

            self._findings.append(StakeholderFinding(
                topic=matter_name,
                description=f"Raised {frequency} time(s) by {groups_count} stakeholder group(s)",
                stakeholder_groups_count=groups_count,
                frequency=frequency,
                priority_score=round(priority, 2),
                supporting_stakeholders=sorted(list(details["stakeholder_groups"])),
            ))

        outputs["findings_synthesized"] = len(self._findings)
        outputs["unique_matters_raised"] = len(matter_details)
        outputs["top_findings"] = [
            {"topic": f.topic, "priority": f.priority_score}
            for f in self._findings[:10]
        ]

        if not self._findings:
            warnings.append(
                "No findings synthesized; check consultation matters_raised data"
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 FindingSynthesis: %d findings from %d matters",
            len(self._findings), len(matter_details),
        )
        return PhaseResult(
            phase_name="finding_synthesis", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Validation
    # -------------------------------------------------------------------------

    async def _phase_validation(
        self, input_data: StakeholderEngagementInput,
    ) -> PhaseResult:
        """Validate completeness per ESRS 1 sections 22-23."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._validation_checks = []

        # Check 1: Affected stakeholder categories coverage
        present_cats = set(sh.category.value for sh in self._stakeholders)
        required_cats = set(c.value for c in input_data.required_categories)
        cats_covered = required_cats.issubset(present_cats)
        self._validation_checks.append(ValidationCheck(
            check_id="vc_001",
            check_name="Required stakeholder categories",
            esrs_reference="ESRS 1, para 22",
            passed=cats_covered,
            details=f"Required: {sorted(required_cats)}, Present: {sorted(present_cats)}",
        ))

        # Check 2: At least one consultation per key player
        key_player_ids = set(
            sh.stakeholder_id for sh in self._stakeholders
            if sh.priority == PriorityLevel.KEY_PLAYER
        )
        consulted_ids = set(c.stakeholder_id for c in self._consultations)
        key_players_consulted = key_player_ids.issubset(consulted_ids)
        self._validation_checks.append(ValidationCheck(
            check_id="vc_002",
            check_name="Key player consultation coverage",
            esrs_reference="ESRS 1, para 22-23",
            passed=key_players_consulted or len(key_player_ids) == 0,
            details=(
                f"Key players: {len(key_player_ids)}, "
                f"Consulted: {len(key_player_ids & consulted_ids)}"
            ),
        ))

        # Check 3: Findings documented
        has_findings = len(self._findings) > 0
        self._validation_checks.append(ValidationCheck(
            check_id="vc_003",
            check_name="Stakeholder findings documented",
            esrs_reference="ESRS 1, para 23",
            passed=has_findings,
            details=f"Findings count: {len(self._findings)}",
        ))

        # Check 4: Multiple engagement methods used
        methods_used = set(c.method.value for c in self._consultations)
        multi_method = len(methods_used) >= 2
        self._validation_checks.append(ValidationCheck(
            check_id="vc_004",
            check_name="Diverse engagement methods",
            esrs_reference="ESRS 1, para 22",
            passed=multi_method or len(self._consultations) == 0,
            details=f"Methods used: {sorted(methods_used)}",
        ))

        # Check 5: Evidence references provided
        with_evidence = sum(1 for c in self._consultations if c.evidence_reference)
        evidence_coverage = (
            with_evidence / len(self._consultations) * 100
            if self._consultations else 0.0
        )
        self._validation_checks.append(ValidationCheck(
            check_id="vc_005",
            check_name="Evidence documentation",
            esrs_reference="ESRS 1, Appendix A",
            passed=evidence_coverage >= 50.0,
            details=f"Evidence coverage: {evidence_coverage:.1f}%",
        ))

        all_passed = all(vc.passed for vc in self._validation_checks)
        pass_count = sum(1 for vc in self._validation_checks if vc.passed)

        outputs["checks_total"] = len(self._validation_checks)
        outputs["checks_passed"] = pass_count
        outputs["checks_failed"] = len(self._validation_checks) - pass_count
        outputs["overall_validation"] = "PASS" if all_passed else "FAIL"

        if not all_passed:
            failed_names = [vc.check_name for vc in self._validation_checks if not vc.passed]
            warnings.append(f"Validation failed for: {', '.join(failed_names)}")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 Validation: %d/%d checks passed, overall=%s",
            pass_count, len(self._validation_checks),
            "PASS" if all_passed else "FAIL",
        )
        return PhaseResult(
            phase_name="validation", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: StakeholderEngagementResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
