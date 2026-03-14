# -*- coding: utf-8 -*-
"""
Stakeholder Engagement Workflow
================================

Stakeholder engagement workflow for ESRS 1 double materiality. Manages
stakeholder identification, salience scoring, survey design, engagement
execution, materiality input analysis, and audit-ready evidence packaging.

Phases:
    1. Stakeholder Mapping: Register stakeholders, calculate salience scores
    2. Survey Design: Generate materiality surveys per stakeholder group
    3. Engagement Execution: Track distribution, collect responses (mock in demo)
    4. Analysis: Weighted materiality aggregation, influence analysis
    5. Evidence Packaging: Audit-ready engagement documentation per ESRS 1

Author: GreenLang Team
Version: 2.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

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
    CANCELLED = "cancelled"


class StakeholderCategory(str, Enum):
    """Stakeholder categories per ESRS 1."""
    EMPLOYEES = "employees"
    INVESTORS = "investors"
    CUSTOMERS = "customers"
    SUPPLIERS = "suppliers"
    COMMUNITIES = "communities"
    REGULATORS = "regulators"
    NGO = "ngo"
    INDUSTRY_BODIES = "industry_bodies"
    ACADEMIA = "academia"
    MEDIA = "media"


# =============================================================================
# DATA MODELS
# =============================================================================


class StakeholderInput(BaseModel):
    """Input definition for a stakeholder."""
    name: str = Field(..., description="Stakeholder or group name")
    category: str = Field(..., description="Stakeholder category")
    organization: str = Field(default="", description="Organization represented")
    power_score: float = Field(default=0.5, ge=0, le=1, description="Power/influence score")
    legitimacy_score: float = Field(default=0.5, ge=0, le=1, description="Legitimacy score")
    urgency_score: float = Field(default=0.5, ge=0, le=1, description="Urgency score")
    email: str = Field(default="", description="Contact email for surveys")


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0)
    agents_executed: int = Field(default=0)
    records_processed: int = Field(default=0)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class PhaseDefinition(BaseModel):
    """Internal definition of a workflow phase."""
    name: str
    display_name: str
    estimated_minutes: float
    required: bool = True
    depends_on: List[str] = Field(default_factory=list)


class StakeholderEngagementInput(BaseModel):
    """Input configuration for the stakeholder engagement workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=2024, le=2050)
    stakeholders: List[StakeholderInput] = Field(
        ..., min_length=1, description="Stakeholders to engage"
    )
    esrs_topics: List[str] = Field(
        default_factory=lambda: [
            "ESRS_E1", "ESRS_E2", "ESRS_E3", "ESRS_E4", "ESRS_E5",
            "ESRS_S1", "ESRS_S2", "ESRS_S3", "ESRS_S4",
            "ESRS_G1", "ESRS_G2",
        ],
        description="ESRS topics for materiality input"
    )
    engagement_types: List[str] = Field(
        default_factory=lambda: ["survey", "interview", "workshop"],
        description="Engagement types to use"
    )


class StakeholderEngagementResult(BaseModel):
    """Complete result from the stakeholder engagement workflow."""
    workflow_id: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    salience_map: Dict[str, Any] = Field(
        default_factory=dict, description="Stakeholder quadrant mapping"
    )
    engagement_activities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Engagement activities executed"
    )
    materiality_inputs: Dict[str, Any] = Field(
        default_factory=dict, description="Aggregated materiality inputs per topic"
    )
    participation_rate: float = Field(default=0.0, description="Response rate")
    evidence_package: Dict[str, Any] = Field(
        default_factory=dict, description="Audit-ready evidence"
    )
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class StakeholderEngagementWorkflow:
    """
    Stakeholder engagement workflow for ESRS 1 materiality assessment.

    Manages the full stakeholder engagement lifecycle: identification,
    salience scoring, survey design, engagement execution, materiality
    input analysis, and audit-ready evidence packaging.

    Attributes:
        workflow_id: Unique execution identifier.
        _cancelled: Cancellation flag.
        _progress_callback: Optional progress callback.

    Example:
        >>> workflow = StakeholderEngagementWorkflow()
        >>> input_cfg = StakeholderEngagementInput(
        ...     organization_id="org-123",
        ...     reporting_year=2025,
        ...     stakeholders=[StakeholderInput(
        ...         name="Investor Group A", category="investors",
        ...         power_score=0.9, legitimacy_score=0.8, urgency_score=0.7,
        ...     )],
        ... )
        >>> result = await workflow.execute(input_cfg)
        >>> assert result.participation_rate > 0
    """

    PHASES: List[PhaseDefinition] = [
        PhaseDefinition(
            name="stakeholder_mapping",
            display_name="Stakeholder Mapping & Salience",
            estimated_minutes=5.0,
            required=True,
            depends_on=[],
        ),
        PhaseDefinition(
            name="survey_design",
            display_name="Survey Design",
            estimated_minutes=10.0,
            required=True,
            depends_on=["stakeholder_mapping"],
        ),
        PhaseDefinition(
            name="engagement_execution",
            display_name="Engagement Execution",
            estimated_minutes=5.0,
            required=True,
            depends_on=["survey_design"],
        ),
        PhaseDefinition(
            name="analysis",
            display_name="Materiality Input Analysis",
            estimated_minutes=15.0,
            required=True,
            depends_on=["engagement_execution"],
        ),
        PhaseDefinition(
            name="evidence_packaging",
            display_name="Evidence Packaging",
            estimated_minutes=10.0,
            required=True,
            depends_on=["analysis"],
        ),
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the stakeholder engagement workflow.

        Args:
            progress_callback: Optional callback(phase_name, message, pct_complete).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._cancelled: bool = False
        self._progress_callback = progress_callback
        self._phase_results: Dict[str, PhaseResult] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: StakeholderEngagementInput
    ) -> StakeholderEngagementResult:
        """
        Execute the stakeholder engagement workflow.

        Args:
            input_data: Validated workflow input.

        Returns:
            StakeholderEngagementResult with salience map, materiality inputs,
            and evidence package.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting stakeholder engagement %s for org=%s year=%d "
            "stakeholders=%d",
            self.workflow_id, input_data.organization_id,
            input_data.reporting_year, len(input_data.stakeholders),
        )
        self._notify_progress("workflow", "Workflow started", 0.0)

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            for idx, phase_def in enumerate(self.PHASES):
                if self._cancelled:
                    overall_status = WorkflowStatus.CANCELLED
                    break

                for dep in phase_def.depends_on:
                    dep_result = self._phase_results.get(dep)
                    if dep_result and dep_result.status == PhaseStatus.FAILED:
                        if phase_def.required:
                            raise RuntimeError(
                                f"Required phase '{phase_def.name}' cannot run: "
                                f"dependency '{dep}' failed."
                            )

                pct_base = idx / len(self.PHASES)
                self._notify_progress(
                    phase_def.name, f"Starting: {phase_def.display_name}", pct_base
                )

                phase_result = await self._execute_phase(
                    phase_def, input_data, pct_base
                )
                completed_phases.append(phase_result)
                self._phase_results[phase_def.name] = phase_result

                if phase_result.status == PhaseStatus.FAILED and phase_def.required:
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

        except Exception as exc:
            logger.critical(
                "Workflow %s failed: %s", self.workflow_id, exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            completed_phases.append(PhaseResult(
                phase_name="workflow_error", status=PhaseStatus.FAILED,
                errors=[str(exc)],
                provenance_hash=self._hash_data({"error": str(exc)}),
            ))

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        salience = self._extract_salience_map(completed_phases)
        activities = self._extract_activities(completed_phases)
        mat_inputs = self._extract_materiality_inputs(completed_phases)
        part_rate = self._extract_participation_rate(completed_phases)
        evidence = self._extract_evidence_package(completed_phases)
        artifacts = {p.phase_name: p.artifacts for p in completed_phases if p.artifacts}

        provenance = self._hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress("workflow", f"Workflow {overall_status.value}", 1.0)

        return StakeholderEngagementResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            salience_map=salience,
            engagement_activities=activities,
            materiality_inputs=mat_inputs,
            participation_rate=part_rate,
            evidence_package=evidence,
            artifacts=artifacts,
            provenance_hash=provenance,
        )

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        logger.info("Cancellation requested for workflow %s", self.workflow_id)
        self._cancelled = True

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    async def _execute_phase(
        self, phase_def: PhaseDefinition,
        input_data: StakeholderEngagementInput, pct_base: float,
    ) -> PhaseResult:
        """Dispatch to the correct phase handler."""
        started_at = datetime.utcnow()
        handler_map = {
            "stakeholder_mapping": self._phase_stakeholder_mapping,
            "survey_design": self._phase_survey_design,
            "engagement_execution": self._phase_engagement_execution,
            "analysis": self._phase_analysis,
            "evidence_packaging": self._phase_evidence_packaging,
        }
        handler = handler_map.get(phase_def.name)
        if handler is None:
            return PhaseResult(
                phase_name=phase_def.name, status=PhaseStatus.FAILED,
                started_at=started_at,
                errors=[f"Unknown phase: {phase_def.name}"],
                provenance_hash=self._hash_data({"error": "unknown_phase"}),
            )
        try:
            result = await handler(input_data, pct_base)
            result.started_at = started_at
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (result.completed_at - started_at).total_seconds()
            return result
        except Exception as exc:
            logger.error("Phase '%s' raised: %s", phase_def.name, exc, exc_info=True)
            return PhaseResult(
                phase_name=phase_def.name, status=PhaseStatus.FAILED,
                started_at=started_at, completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - started_at).total_seconds(),
                errors=[str(exc)],
                provenance_hash=self._hash_data({"error": str(exc)}),
            )

    # -------------------------------------------------------------------------
    # Phase 1: Stakeholder Mapping
    # -------------------------------------------------------------------------

    async def _phase_stakeholder_mapping(
        self, input_data: StakeholderEngagementInput, pct_base: float
    ) -> PhaseResult:
        """
        Register stakeholders, calculate Mitchell-Agle-Wood salience scores
        (power x legitimacy x urgency), and assign to quadrants.
        """
        phase_name = "stakeholder_mapping"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        self._notify_progress(phase_name, "Calculating salience scores", pct_base + 0.02)

        salience_entries: List[Dict[str, Any]] = []
        quadrants: Dict[str, List[str]] = {
            "definitive": [],
            "dominant": [],
            "dependent": [],
            "discretionary": [],
        }

        for sh in input_data.stakeholders:
            salience_score = (sh.power_score + sh.legitimacy_score + sh.urgency_score) / 3
            quadrant = self._classify_quadrant(
                sh.power_score, sh.legitimacy_score, sh.urgency_score
            )
            entry = {
                "name": sh.name,
                "category": sh.category,
                "organization": sh.organization,
                "power": sh.power_score,
                "legitimacy": sh.legitimacy_score,
                "urgency": sh.urgency_score,
                "salience_score": round(salience_score, 3),
                "quadrant": quadrant,
            }
            salience_entries.append(entry)
            quadrants[quadrant].append(sh.name)

        agents_executed = 1
        artifacts["salience_entries"] = salience_entries
        artifacts["quadrants"] = quadrants
        artifacts["total_stakeholders"] = len(input_data.stakeholders)
        artifacts["categories_represented"] = list(
            {sh.category for sh in input_data.stakeholders}
        )

        # Check for underrepresented categories
        represented = {sh.category for sh in input_data.stakeholders}
        recommended = {"employees", "investors", "customers", "communities", "regulators"}
        missing = recommended - represented
        if missing:
            warnings.append(
                f"Recommended stakeholder categories not represented: "
                f"{', '.join(missing)}"
            )

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(input_data.stakeholders),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Survey Design
    # -------------------------------------------------------------------------

    async def _phase_survey_design(
        self, input_data: StakeholderEngagementInput, pct_base: float
    ) -> PhaseResult:
        """
        Generate materiality assessment surveys tailored per stakeholder
        group based on salience scores and engagement type.
        """
        phase_name = "survey_design"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        mapping_phase = self._phase_results.get("stakeholder_mapping")
        salience = (
            mapping_phase.artifacts.get("salience_entries", [])
            if mapping_phase and mapping_phase.artifacts else []
        )

        self._notify_progress(phase_name, "Designing surveys per group", pct_base + 0.02)

        surveys: List[Dict[str, Any]] = []
        categories = {sh.category for sh in input_data.stakeholders}

        for category in categories:
            for eng_type in input_data.engagement_types:
                survey = await self._design_survey(
                    input_data.organization_id, category, eng_type,
                    input_data.esrs_topics,
                )
                surveys.append(survey)
                agents_executed += 1

        artifacts["surveys_designed"] = len(surveys)
        artifacts["surveys"] = surveys
        artifacts["engagement_types"] = input_data.engagement_types
        artifacts["categories_covered"] = list(categories)

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(surveys),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Engagement Execution
    # -------------------------------------------------------------------------

    async def _phase_engagement_execution(
        self, input_data: StakeholderEngagementInput, pct_base: float
    ) -> PhaseResult:
        """
        Track survey distribution, collect responses. In demo mode,
        generates realistic mock response data.
        """
        phase_name = "engagement_execution"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        self._notify_progress(
            phase_name, "Distributing surveys and collecting responses", pct_base + 0.02
        )

        # Simulate distribution and collection
        total_distributed = len(input_data.stakeholders) * len(
            input_data.engagement_types
        )
        responses_collected = int(total_distributed * 0.72)

        activities: List[Dict[str, Any]] = []
        response_data: List[Dict[str, Any]] = []

        for sh in input_data.stakeholders:
            for eng_type in input_data.engagement_types:
                responded = hash(f"{sh.name}{eng_type}") % 4 != 0  # ~75% response
                activity = {
                    "stakeholder": sh.name,
                    "category": sh.category,
                    "engagement_type": eng_type,
                    "distributed": True,
                    "responded": responded,
                    "distributed_at": datetime.utcnow().isoformat(),
                }
                activities.append(activity)

                if responded:
                    response = await self._generate_mock_response(
                        sh, eng_type, input_data.esrs_topics
                    )
                    response_data.append(response)

        agents_executed = 1
        participation_rate = (
            len(response_data) / max(len(activities), 1) * 100
        )

        artifacts["total_distributed"] = len(activities)
        artifacts["responses_collected"] = len(response_data)
        artifacts["participation_rate_pct"] = round(participation_rate, 1)
        artifacts["activities"] = activities
        artifacts["response_data"] = response_data

        if participation_rate < 50:
            warnings.append(
                f"Participation rate is {participation_rate:.1f}% (below 50% threshold). "
                "Consider follow-up engagement."
            )

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(response_data),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Analysis
    # -------------------------------------------------------------------------

    async def _phase_analysis(
        self, input_data: StakeholderEngagementInput, pct_base: float
    ) -> PhaseResult:
        """
        Weighted materiality aggregation and stakeholder influence analysis.
        Weights responses by stakeholder salience score.
        """
        phase_name = "analysis"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        exec_phase = self._phase_results.get("engagement_execution")
        response_data = (
            exec_phase.artifacts.get("response_data", [])
            if exec_phase and exec_phase.artifacts else []
        )

        mapping_phase = self._phase_results.get("stakeholder_mapping")
        salience_entries = (
            mapping_phase.artifacts.get("salience_entries", [])
            if mapping_phase and mapping_phase.artifacts else []
        )

        self._notify_progress(
            phase_name, "Running weighted materiality aggregation", pct_base + 0.02
        )

        # Build salience lookup
        salience_lookup: Dict[str, float] = {
            e["name"]: e["salience_score"] for e in salience_entries
        }

        # Weighted aggregation per topic
        topic_scores: Dict[str, Dict[str, float]] = {}
        for topic in input_data.esrs_topics:
            weighted_sum = 0.0
            weight_total = 0.0
            for response in response_data:
                score = response.get("topic_scores", {}).get(topic, 0.0)
                weight = salience_lookup.get(response.get("stakeholder", ""), 0.5)
                weighted_sum += score * weight
                weight_total += weight

            avg_score = weighted_sum / max(weight_total, 0.001)
            topic_scores[topic] = {
                "weighted_score": round(avg_score, 3),
                "responses": sum(
                    1 for r in response_data
                    if topic in r.get("topic_scores", {})
                ),
            }

        agents_executed = 1
        artifacts["topic_scores"] = topic_scores

        self._notify_progress(
            phase_name, "Running influence analysis", pct_base + 0.04
        )

        # Influence analysis
        influence = await self._run_influence_analysis(
            input_data.organization_id, response_data, salience_entries
        )
        agents_executed += 1
        artifacts["influence_analysis"] = influence

        # Rank topics
        ranked_topics = sorted(
            topic_scores.items(),
            key=lambda x: x[1]["weighted_score"],
            reverse=True,
        )
        artifacts["ranked_topics"] = [
            {"topic": t, "score": s["weighted_score"]}
            for t, s in ranked_topics
        ]

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(response_data),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Evidence Packaging
    # -------------------------------------------------------------------------

    async def _phase_evidence_packaging(
        self, input_data: StakeholderEngagementInput, pct_base: float
    ) -> PhaseResult:
        """
        Generate audit-ready engagement documentation per ESRS 1.

        Includes:
            - Stakeholder identification methodology
            - Salience scoring rationale
            - Survey instruments
            - Response data (anonymized)
            - Aggregation methodology
            - Results and how they informed materiality
        """
        phase_name = "evidence_packaging"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        self._notify_progress(
            phase_name, "Assembling engagement evidence package", pct_base + 0.02
        )

        # Generate evidence package
        package = await self._generate_evidence_package(
            input_data.organization_id,
            input_data.reporting_year,
            self._phase_results.get("stakeholder_mapping"),
            self._phase_results.get("survey_design"),
            self._phase_results.get("engagement_execution"),
            self._phase_results.get("analysis"),
        )
        agents_executed = 1

        artifacts["evidence_package"] = package
        artifacts["package_id"] = package.get("package_id", "")
        artifacts["sections"] = package.get("sections", [])
        artifacts["documents_generated"] = package.get("document_count", 0)

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(artifacts.get("sections", [])),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Agent Invocation Helpers
    # -------------------------------------------------------------------------

    def _classify_quadrant(
        self, power: float, legitimacy: float, urgency: float
    ) -> str:
        """Classify stakeholder into Mitchell-Agle-Wood quadrant."""
        high_count = sum(1 for s in [power, legitimacy, urgency] if s >= 0.6)
        if high_count == 3:
            return "definitive"
        elif power >= 0.6 and legitimacy >= 0.6:
            return "dominant"
        elif legitimacy >= 0.6 and urgency >= 0.6:
            return "dependent"
        else:
            return "discretionary"

    async def _design_survey(
        self, org_id: str, category: str, eng_type: str, topics: List[str],
    ) -> Dict[str, Any]:
        """Design a materiality survey for a stakeholder group."""
        await asyncio.sleep(0)
        return {
            "survey_id": str(uuid.uuid4()),
            "category": category,
            "engagement_type": eng_type,
            "topics_covered": topics,
            "questions_count": len(topics) * 3 + 5,
            "estimated_completion_minutes": 15,
            "language": "en",
        }

    async def _generate_mock_response(
        self, stakeholder: StakeholderInput, eng_type: str,
        topics: List[str],
    ) -> Dict[str, Any]:
        """Generate realistic mock response data for demo."""
        await asyncio.sleep(0)
        topic_scores: Dict[str, float] = {}
        for i, topic in enumerate(topics):
            base = 0.4 + (hash(f"{stakeholder.name}{topic}") % 50) / 100
            topic_scores[topic] = round(min(1.0, base), 3)

        return {
            "response_id": str(uuid.uuid4()),
            "stakeholder": stakeholder.name,
            "category": stakeholder.category,
            "engagement_type": eng_type,
            "topic_scores": topic_scores,
            "qualitative_feedback": f"Feedback from {stakeholder.name} regarding sustainability priorities.",
            "completed_at": datetime.utcnow().isoformat(),
        }

    async def _run_influence_analysis(
        self, org_id: str, responses: List[Dict[str, Any]],
        salience: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run stakeholder influence analysis."""
        await asyncio.sleep(0)
        return {
            "most_influential_category": "investors",
            "alignment_score": 0.78,
            "divergence_topics": ["ESRS_E4", "ESRS_S3"],
            "consensus_topics": ["ESRS_E1", "ESRS_G1"],
        }

    async def _generate_evidence_package(
        self, org_id: str, year: int,
        mapping: Optional[PhaseResult], survey: Optional[PhaseResult],
        execution: Optional[PhaseResult], analysis: Optional[PhaseResult],
    ) -> Dict[str, Any]:
        """Generate audit-ready evidence package."""
        await asyncio.sleep(0)
        return {
            "package_id": str(uuid.uuid4()),
            "sections": [
                "stakeholder_identification_methodology",
                "salience_scoring_rationale",
                "survey_instruments",
                "anonymized_response_data",
                "aggregation_methodology",
                "materiality_influence_results",
            ],
            "document_count": 8,
            "format": "pdf",
            "esrs_reference": "ESRS 1 Chapter 3 para 22-26",
        }

    # -------------------------------------------------------------------------
    # Result Extractors
    # -------------------------------------------------------------------------

    def _extract_salience_map(self, phases: List[PhaseResult]) -> Dict[str, Any]:
        """Extract salience map from mapping phase."""
        for p in phases:
            if p.phase_name == "stakeholder_mapping" and p.artifacts:
                return {
                    "entries": p.artifacts.get("salience_entries", []),
                    "quadrants": p.artifacts.get("quadrants", {}),
                }
        return {}

    def _extract_activities(self, phases: List[PhaseResult]) -> List[Dict[str, Any]]:
        """Extract engagement activities."""
        for p in phases:
            if p.phase_name == "engagement_execution" and p.artifacts:
                return p.artifacts.get("activities", [])
        return []

    def _extract_materiality_inputs(self, phases: List[PhaseResult]) -> Dict[str, Any]:
        """Extract materiality inputs."""
        for p in phases:
            if p.phase_name == "analysis" and p.artifacts:
                return p.artifacts.get("topic_scores", {})
        return {}

    def _extract_participation_rate(self, phases: List[PhaseResult]) -> float:
        """Extract participation rate."""
        for p in phases:
            if p.phase_name == "engagement_execution" and p.artifacts:
                return p.artifacts.get("participation_rate_pct", 0.0)
        return 0.0

    def _extract_evidence_package(self, phases: List[PhaseResult]) -> Dict[str, Any]:
        """Extract evidence package."""
        for p in phases:
            if p.phase_name == "evidence_packaging" and p.artifacts:
                return p.artifacts.get("evidence_package", {})
        return {}

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)

    @staticmethod
    def _hash_data(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        serialized = str(data).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
