# -*- coding: utf-8 -*-
"""
Materiality Assessment Workflow
===============================

Standalone double materiality assessment workflow per ESRS 1 Chapter 3.
Can run independently from the annual reporting cycle (e.g. mid-year refresh,
board-mandated reassessment, or M&A due diligence).

Implements the EFRAG IG-1 guidance on double materiality with:
    - Impact materiality: severity (scale x scope x irremediability) + likelihood
    - Financial materiality: magnitude x likelihood over time horizons
    - Combined double materiality matrix with configurable thresholds
    - Human-in-the-loop review and approval workflow
    - Full audit trail for assurance readiness

Steps:
    1. Company context collection (sector, value chain, stakeholders)
    2. Impact materiality scoring (severity x scope x irremediability)
    3. Financial materiality scoring (magnitude x likelihood)
    4. Double materiality matrix generation
    5. Material topic prioritization
    6. Human review queue with approval workflow
    7. Documentation for auditors

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class MaterialityStepStatus(str, Enum):
    """Status of a materiality workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    AWAITING_REVIEW = "awaiting_review"


class MaterialityDimension(str, Enum):
    """Which materiality dimension a topic qualifies under."""
    IMPACT_ONLY = "impact_only"
    FINANCIAL_ONLY = "financial_only"
    DOUBLE = "double"
    NOT_MATERIAL = "not_material"


class TimeHorizon(str, Enum):
    """ESRS time horizons for materiality assessment."""
    SHORT_TERM = "short_term"     # <= 1 year
    MEDIUM_TERM = "medium_term"   # 1-5 years
    LONG_TERM = "long_term"       # > 5 years


class ReviewDecision(str, Enum):
    """Human reviewer decision for a material topic."""
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    PENDING = "pending"


# =============================================================================
# DATA MODELS
# =============================================================================


class StakeholderGroup(BaseModel):
    """Definition of a stakeholder group for impact assessment."""
    name: str = Field(..., description="Stakeholder group name")
    category: str = Field(
        ..., description="Category: employees, communities, customers, investors, regulators, suppliers"
    )
    engagement_method: str = Field(
        default="survey", description="How stakeholders were engaged"
    )
    weight: float = Field(default=1.0, ge=0.0, le=5.0, description="Relative weight in assessment")


class MaterialityAssessmentInput(BaseModel):
    """Input configuration for the materiality assessment workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    assessment_year: int = Field(..., ge=2024, le=2050, description="Assessment year")
    sector_codes: List[str] = Field(
        ..., min_length=1, description="NACE sector codes (at least one)"
    )
    value_chain_description: str = Field(
        default="", description="Free-text description of value chain"
    )
    stakeholder_groups: List[StakeholderGroup] = Field(
        default_factory=lambda: [
            StakeholderGroup(name="Employees", category="employees"),
            StakeholderGroup(name="Investors", category="investors"),
            StakeholderGroup(name="Local Communities", category="communities"),
            StakeholderGroup(name="Customers", category="customers"),
            StakeholderGroup(name="Regulators", category="regulators"),
        ],
        description="Stakeholder groups to consider"
    )
    esrs_topics: List[str] = Field(
        default_factory=lambda: [
            "E1_climate_change", "E2_pollution", "E3_water_marine",
            "E4_biodiversity", "E5_circular_economy",
            "S1_own_workforce", "S2_value_chain_workers",
            "S3_affected_communities", "S4_consumers",
            "G1_business_conduct", "G2_corporate_culture",
        ],
        description="ESRS topics to assess"
    )
    impact_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Threshold for impact materiality"
    )
    financial_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Threshold for financial materiality"
    )
    include_sector_agnostic: bool = Field(
        default=True, description="Include ESRS 2 cross-cutting (always material)"
    )
    reviewer_emails: List[str] = Field(
        default_factory=list, description="Email addresses for human review queue"
    )

    @field_validator("sector_codes")
    @classmethod
    def validate_sector_codes(cls, v: List[str]) -> List[str]:
        """Validate NACE sector codes are non-empty strings."""
        for code in v:
            if not code.strip():
                raise ValueError("Sector codes must be non-empty strings")
        return v


class ImpactScore(BaseModel):
    """Impact materiality score for a single topic."""
    topic_id: str = Field(..., description="ESRS topic identifier")
    scale: float = Field(..., ge=0, le=1, description="Scale of impact (0-1)")
    scope: float = Field(..., ge=0, le=1, description="Scope/breadth of impact (0-1)")
    irremediability: float = Field(..., ge=0, le=1, description="Irremediability (0-1)")
    likelihood: float = Field(..., ge=0, le=1, description="Likelihood of occurrence (0-1)")
    severity_score: float = Field(..., ge=0, le=1, description="Combined severity (scale x scope x irremediability)")
    final_score: float = Field(..., ge=0, le=1, description="Final impact materiality score")
    time_horizon: TimeHorizon = Field(..., description="Relevant time horizon")
    rationale: str = Field(default="", description="Rationale for scoring")


class FinancialScore(BaseModel):
    """Financial materiality score for a single topic."""
    topic_id: str = Field(..., description="ESRS topic identifier")
    magnitude: float = Field(..., ge=0, le=1, description="Financial magnitude (0-1)")
    likelihood: float = Field(..., ge=0, le=1, description="Likelihood (0-1)")
    final_score: float = Field(..., ge=0, le=1, description="Final financial materiality score")
    time_horizon: TimeHorizon = Field(..., description="Relevant time horizon")
    risk_or_opportunity: str = Field(default="risk", description="risk or opportunity")
    rationale: str = Field(default="", description="Rationale for scoring")


class MaterialTopic(BaseModel):
    """A topic that has passed the materiality threshold."""
    topic_id: str = Field(...)
    topic_name: str = Field(...)
    esrs_standard: str = Field(...)
    dimension: MaterialityDimension = Field(...)
    impact_score: float = Field(default=0.0)
    financial_score: float = Field(default=0.0)
    combined_score: float = Field(default=0.0)
    priority_rank: int = Field(default=0)
    review_status: ReviewDecision = Field(default=ReviewDecision.PENDING)
    reviewer_notes: str = Field(default="")


class StepResult(BaseModel):
    """Result from a single workflow step."""
    step_name: str = Field(...)
    status: MaterialityStepStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class MaterialityAssessmentResult(BaseModel):
    """Complete result from the materiality assessment workflow."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    status: MaterialityStepStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    steps: List[StepResult] = Field(default_factory=list)
    material_topics: List[MaterialTopic] = Field(default_factory=list)
    non_material_topics: List[str] = Field(default_factory=list)
    matrix: Dict[str, Any] = Field(default_factory=dict)
    review_queue_id: Optional[str] = Field(None)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# TOPIC REGISTRY
# =============================================================================

ESRS_TOPIC_REGISTRY: Dict[str, Dict[str, str]] = {
    "E1_climate_change": {"name": "Climate Change", "standard": "ESRS E1"},
    "E2_pollution": {"name": "Pollution", "standard": "ESRS E2"},
    "E3_water_marine": {"name": "Water and Marine Resources", "standard": "ESRS E3"},
    "E4_biodiversity": {"name": "Biodiversity and Ecosystems", "standard": "ESRS E4"},
    "E5_circular_economy": {"name": "Resource Use and Circular Economy", "standard": "ESRS E5"},
    "S1_own_workforce": {"name": "Own Workforce", "standard": "ESRS S1"},
    "S2_value_chain_workers": {"name": "Workers in the Value Chain", "standard": "ESRS S2"},
    "S3_affected_communities": {"name": "Affected Communities", "standard": "ESRS S3"},
    "S4_consumers": {"name": "Consumers and End-users", "standard": "ESRS S4"},
    "G1_business_conduct": {"name": "Business Conduct", "standard": "ESRS G1"},
    "G2_corporate_culture": {"name": "Corporate Culture", "standard": "ESRS G2"},
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class MaterialityAssessmentWorkflow:
    """
    Standalone double materiality assessment per ESRS 1 Chapter 3.

    Follows EFRAG Implementation Guidance IG-1 for identifying material topics
    across both impact and financial dimensions. Produces a prioritized list
    of material topics with full audit documentation for assurance readiness.

    Attributes:
        workflow_id: Unique execution identifier.
        _cancelled: Cancellation flag for cooperative shutdown.
        _progress_callback: Optional callback for step progress updates.

    Example:
        >>> wf = MaterialityAssessmentWorkflow()
        >>> inp = MaterialityAssessmentInput(
        ...     organization_id="org-123",
        ...     assessment_year=2025,
        ...     sector_codes=["C20.1"],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert len(result.material_topics) > 0
    """

    STEPS = [
        "context_collection",
        "impact_scoring",
        "financial_scoring",
        "matrix_generation",
        "topic_prioritization",
        "human_review",
        "audit_documentation",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the materiality assessment workflow.

        Args:
            progress_callback: Optional callback(step_name, message, pct_complete).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._cancelled: bool = False
        self._progress_callback = progress_callback
        self._step_results: Dict[str, StepResult] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: MaterialityAssessmentInput
    ) -> MaterialityAssessmentResult:
        """
        Execute the double materiality assessment workflow.

        Args:
            input_data: Validated materiality assessment input.

        Returns:
            MaterialityAssessmentResult with material topics, matrix, review queue.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting materiality assessment %s for org=%s year=%d",
            self.workflow_id, input_data.organization_id, input_data.assessment_year,
        )
        self._notify("workflow", "Materiality assessment started", 0.0)

        completed_steps: List[StepResult] = []
        overall_status = MaterialityStepStatus.RUNNING
        material_topics: List[MaterialTopic] = []
        non_material: List[str] = []
        matrix: Dict[str, Any] = {}
        review_queue_id: Optional[str] = None

        step_handlers = [
            ("context_collection", self._step_context_collection),
            ("impact_scoring", self._step_impact_scoring),
            ("financial_scoring", self._step_financial_scoring),
            ("matrix_generation", self._step_matrix_generation),
            ("topic_prioritization", self._step_topic_prioritization),
            ("human_review", self._step_human_review),
            ("audit_documentation", self._step_audit_documentation),
        ]

        try:
            for idx, (step_name, handler) in enumerate(step_handlers):
                if self._cancelled:
                    overall_status = MaterialityStepStatus.SKIPPED
                    break

                pct = idx / len(step_handlers)
                self._notify(step_name, f"Starting: {step_name}", pct)
                step_started = datetime.utcnow()

                try:
                    step_result = await handler(input_data, pct)
                    step_result.started_at = step_started
                    step_result.completed_at = datetime.utcnow()
                    step_result.duration_seconds = (
                        step_result.completed_at - step_started
                    ).total_seconds()
                except Exception as exc:
                    logger.error("Step '%s' failed: %s", step_name, exc, exc_info=True)
                    step_result = StepResult(
                        step_name=step_name,
                        status=MaterialityStepStatus.FAILED,
                        started_at=step_started,
                        completed_at=datetime.utcnow(),
                        duration_seconds=(datetime.utcnow() - step_started).total_seconds(),
                        errors=[str(exc)],
                        provenance_hash=self._hash({"error": str(exc)}),
                    )

                completed_steps.append(step_result)
                self._step_results[step_name] = step_result

                # Collect outputs for final result
                if step_name == "matrix_generation" and step_result.artifacts:
                    matrix = step_result.artifacts.get("matrix", {})
                if step_name == "topic_prioritization" and step_result.artifacts:
                    raw = step_result.artifacts.get("material_topics", [])
                    material_topics = [
                        MaterialTopic(**t) for t in raw if isinstance(t, dict)
                    ]
                    non_material = step_result.artifacts.get("non_material_topics", [])
                if step_name == "human_review" and step_result.artifacts:
                    review_queue_id = step_result.artifacts.get("review_queue_id")

                if step_result.status == MaterialityStepStatus.FAILED:
                    overall_status = MaterialityStepStatus.FAILED
                    break

            if overall_status == MaterialityStepStatus.RUNNING:
                # If human review is awaiting, set overall to awaiting
                review_step = self._step_results.get("human_review")
                if review_step and review_step.status == MaterialityStepStatus.AWAITING_REVIEW:
                    overall_status = MaterialityStepStatus.AWAITING_REVIEW
                else:
                    overall_status = MaterialityStepStatus.COMPLETED

        except Exception as exc:
            logger.critical(
                "Materiality assessment %s failed: %s", self.workflow_id, exc, exc_info=True
            )
            overall_status = MaterialityStepStatus.FAILED

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        metrics = {
            "total_topics_assessed": len(input_data.esrs_topics),
            "material_topics": len(material_topics),
            "non_material_topics": len(non_material),
            "steps_completed": sum(
                1 for s in completed_steps
                if s.status in (MaterialityStepStatus.COMPLETED, MaterialityStepStatus.AWAITING_REVIEW)
            ),
        }
        artifacts = {s.step_name: s.artifacts for s in completed_steps if s.artifacts}
        provenance = self._hash({
            "workflow_id": self.workflow_id,
            "steps": [s.provenance_hash for s in completed_steps],
        })

        self._notify("workflow", f"Assessment {overall_status.value}", 1.0)

        return MaterialityAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            steps=completed_steps,
            material_topics=material_topics,
            non_material_topics=non_material,
            matrix=matrix,
            review_queue_id=review_queue_id,
            metrics=metrics,
            artifacts=artifacts,
            provenance_hash=provenance,
        )

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        logger.info("Cancellation requested for assessment %s", self.workflow_id)
        self._cancelled = True

    # -------------------------------------------------------------------------
    # Step 1: Company Context Collection
    # -------------------------------------------------------------------------

    async def _step_context_collection(
        self, input_data: MaterialityAssessmentInput, pct_base: float
    ) -> StepResult:
        """
        Collect company context: sector profile, value chain mapping,
        stakeholder identification, and geographic footprint.

        Agents invoked:
            - greenlang.agents.intelligence (sector context analysis)
            - greenlang.agents.data.erp_connector_agent (org structure data)
        """
        step_name = "context_collection"
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Analyzing sector profile", pct_base + 0.02)

        # Sector profile lookup
        sector_profile = await self._analyze_sector_profile(
            input_data.organization_id, input_data.sector_codes
        )
        artifacts["sector_profile"] = sector_profile

        self._notify(step_name, "Mapping value chain", pct_base + 0.04)

        # Value chain mapping
        value_chain = await self._map_value_chain(
            input_data.organization_id, input_data.value_chain_description
        )
        artifacts["value_chain"] = value_chain

        # Stakeholder mapping
        artifacts["stakeholder_groups"] = [
            sg.model_dump() for sg in input_data.stakeholder_groups
        ]
        artifacts["topics_to_assess"] = input_data.esrs_topics

        # Sector-specific materiality hints (from EFRAG sector standards)
        sector_hints = await self._get_sector_materiality_hints(input_data.sector_codes)
        artifacts["sector_hints"] = sector_hints

        provenance = self._hash(artifacts)

        return StepResult(
            step_name=step_name,
            status=MaterialityStepStatus.COMPLETED,
            artifacts=artifacts,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 2: Impact Materiality Scoring
    # -------------------------------------------------------------------------

    async def _step_impact_scoring(
        self, input_data: MaterialityAssessmentInput, pct_base: float
    ) -> StepResult:
        """
        Score impact materiality for each ESRS topic.

        Per ESRS 1 AR16, impact severity = scale x scope x irremediability.
        For potential impacts, likelihood is also factored in.

        Agents invoked:
            - greenlang.agents.intelligence (AI-based impact classification)
            - greenlang.agents.foundation.assumptions_registry (log assumptions)
        """
        step_name = "impact_scoring"
        artifacts: Dict[str, Any] = {}
        warnings: List[str] = []

        context = self._step_results.get("context_collection")
        sector_hints = {}
        if context and context.artifacts:
            sector_hints = context.artifacts.get("sector_hints", {})

        self._notify(step_name, "Scoring impact materiality for all topics", pct_base + 0.02)

        impact_scores: List[Dict[str, Any]] = []

        for topic_id in input_data.esrs_topics:
            score = await self._score_single_impact(
                input_data.organization_id, topic_id,
                input_data.stakeholder_groups, sector_hints,
            )
            impact_scores.append(score.model_dump())

        # Log assumptions for audit trail
        await self._log_scoring_assumptions(
            input_data.organization_id, "impact", impact_scores
        )

        artifacts["impact_scores"] = impact_scores
        artifacts["methodology"] = "ESRS_1_AR16_severity_x_likelihood"
        artifacts["topics_scored"] = len(impact_scores)

        above_threshold = sum(
            1 for s in impact_scores if s.get("final_score", 0) >= input_data.impact_threshold
        )
        artifacts["above_threshold"] = above_threshold

        if above_threshold == 0:
            warnings.append(
                "No topics reached impact materiality threshold. "
                "Consider lowering threshold or reviewing scoring inputs."
            )

        provenance = self._hash(artifacts)

        return StepResult(
            step_name=step_name,
            status=MaterialityStepStatus.COMPLETED,
            artifacts=artifacts,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 3: Financial Materiality Scoring
    # -------------------------------------------------------------------------

    async def _step_financial_scoring(
        self, input_data: MaterialityAssessmentInput, pct_base: float
    ) -> StepResult:
        """
        Score financial materiality for each ESRS topic.

        Per ESRS 1 AR17, financial materiality considers the magnitude of
        potential financial effects and their likelihood across time horizons.

        Agents invoked:
            - greenlang.agents.intelligence (AI-based financial risk scoring)
            - greenlang.agents.finance (financial data context)
        """
        step_name = "financial_scoring"
        artifacts: Dict[str, Any] = {}
        warnings: List[str] = []

        self._notify(step_name, "Scoring financial materiality for all topics", pct_base + 0.02)

        financial_scores: List[Dict[str, Any]] = []

        for topic_id in input_data.esrs_topics:
            score = await self._score_single_financial(
                input_data.organization_id, topic_id
            )
            financial_scores.append(score.model_dump())

        await self._log_scoring_assumptions(
            input_data.organization_id, "financial", financial_scores
        )

        artifacts["financial_scores"] = financial_scores
        artifacts["methodology"] = "ESRS_1_AR17_magnitude_x_likelihood"
        artifacts["topics_scored"] = len(financial_scores)

        above_threshold = sum(
            1 for s in financial_scores if s.get("final_score", 0) >= input_data.financial_threshold
        )
        artifacts["above_threshold"] = above_threshold

        provenance = self._hash(artifacts)

        return StepResult(
            step_name=step_name,
            status=MaterialityStepStatus.COMPLETED,
            artifacts=artifacts,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 4: Double Materiality Matrix
    # -------------------------------------------------------------------------

    async def _step_matrix_generation(
        self, input_data: MaterialityAssessmentInput, pct_base: float
    ) -> StepResult:
        """
        Generate the double materiality matrix by combining impact and
        financial scores for each topic.
        """
        step_name = "matrix_generation"
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Building double materiality matrix", pct_base + 0.02)

        impact_step = self._step_results.get("impact_scoring")
        financial_step = self._step_results.get("financial_scoring")

        impact_map: Dict[str, float] = {}
        if impact_step and impact_step.artifacts:
            for s in impact_step.artifacts.get("impact_scores", []):
                impact_map[s["topic_id"]] = s.get("final_score", 0.0)

        financial_map: Dict[str, float] = {}
        if financial_step and financial_step.artifacts:
            for s in financial_step.artifacts.get("financial_scores", []):
                financial_map[s["topic_id"]] = s.get("final_score", 0.0)

        matrix_topics: List[Dict[str, Any]] = []
        for topic_id in input_data.esrs_topics:
            imp = impact_map.get(topic_id, 0.0)
            fin = financial_map.get(topic_id, 0.0)
            combined = max(imp, fin)  # ESRS: material if either dimension qualifies

            # Determine dimension
            impact_material = imp >= input_data.impact_threshold
            financial_material = fin >= input_data.financial_threshold
            if impact_material and financial_material:
                dimension = MaterialityDimension.DOUBLE
            elif impact_material:
                dimension = MaterialityDimension.IMPACT_ONLY
            elif financial_material:
                dimension = MaterialityDimension.FINANCIAL_ONLY
            else:
                dimension = MaterialityDimension.NOT_MATERIAL

            topic_info = ESRS_TOPIC_REGISTRY.get(topic_id, {})
            matrix_topics.append({
                "topic_id": topic_id,
                "topic_name": topic_info.get("name", topic_id),
                "esrs_standard": topic_info.get("standard", ""),
                "impact_score": round(imp, 4),
                "financial_score": round(fin, 4),
                "combined_score": round(combined, 4),
                "dimension": dimension.value,
                "is_material": dimension != MaterialityDimension.NOT_MATERIAL,
            })

        artifacts["matrix"] = {
            "topics": matrix_topics,
            "impact_threshold": input_data.impact_threshold,
            "financial_threshold": input_data.financial_threshold,
            "methodology": "ESRS_1_double_materiality",
        }

        provenance = self._hash(artifacts)

        return StepResult(
            step_name=step_name,
            status=MaterialityStepStatus.COMPLETED,
            artifacts=artifacts,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 5: Topic Prioritization
    # -------------------------------------------------------------------------

    async def _step_topic_prioritization(
        self, input_data: MaterialityAssessmentInput, pct_base: float
    ) -> StepResult:
        """
        Rank material topics by combined score and assign priority ranks.
        Separate material from non-material topics.
        """
        step_name = "topic_prioritization"
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Prioritizing material topics", pct_base + 0.02)

        matrix_step = self._step_results.get("matrix_generation")
        matrix_topics = []
        if matrix_step and matrix_step.artifacts:
            matrix_topics = matrix_step.artifacts.get("matrix", {}).get("topics", [])

        # Separate material vs non-material
        material_list: List[Dict[str, Any]] = []
        non_material_list: List[str] = []

        for t in matrix_topics:
            if t.get("is_material", False):
                material_list.append(t)
            else:
                non_material_list.append(t.get("topic_id", ""))

        # Sort by combined score descending
        material_list.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        # Assign priority ranks
        for rank, topic in enumerate(material_list, start=1):
            topic["priority_rank"] = rank
            topic["review_status"] = ReviewDecision.PENDING.value
            topic["reviewer_notes"] = ""

        # Add ESRS 2 cross-cutting if configured (always material by regulation)
        if input_data.include_sector_agnostic:
            esrs2_entry = {
                "topic_id": "ESRS_2_general",
                "topic_name": "General Disclosures (ESRS 2)",
                "esrs_standard": "ESRS 2",
                "impact_score": 1.0,
                "financial_score": 1.0,
                "combined_score": 1.0,
                "dimension": MaterialityDimension.DOUBLE.value,
                "is_material": True,
                "priority_rank": 0,
                "review_status": ReviewDecision.APPROVED.value,
                "reviewer_notes": "ESRS 2 is always material by regulation.",
            }
            material_list.insert(0, esrs2_entry)

        artifacts["material_topics"] = material_list
        artifacts["non_material_topics"] = non_material_list
        artifacts["total_material"] = len(material_list)
        artifacts["total_non_material"] = len(non_material_list)

        provenance = self._hash(artifacts)

        return StepResult(
            step_name=step_name,
            status=MaterialityStepStatus.COMPLETED,
            artifacts=artifacts,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 6: Human Review Queue
    # -------------------------------------------------------------------------

    async def _step_human_review(
        self, input_data: MaterialityAssessmentInput, pct_base: float
    ) -> StepResult:
        """
        Queue the materiality results for human review and approval.
        Creates a review task for each topic requiring sign-off.
        """
        step_name = "human_review"
        artifacts: Dict[str, Any] = {}
        warnings: List[str] = []

        self._notify(step_name, "Creating human review queue", pct_base + 0.02)

        prio_step = self._step_results.get("topic_prioritization")
        material_topics = []
        if prio_step and prio_step.artifacts:
            material_topics = prio_step.artifacts.get("material_topics", [])

        # Create review queue
        review_queue_id = await self._create_review_queue(
            input_data.organization_id,
            material_topics,
            input_data.reviewer_emails,
        )
        artifacts["review_queue_id"] = review_queue_id
        artifacts["topics_for_review"] = len(material_topics)
        artifacts["reviewers_notified"] = len(input_data.reviewer_emails)

        if not input_data.reviewer_emails:
            warnings.append(
                "No reviewer emails configured. Topics will remain in pending state."
            )

        # Non-material topics also need documented justification
        non_material = []
        if prio_step and prio_step.artifacts:
            non_material = prio_step.artifacts.get("non_material_topics", [])
        if non_material:
            justification_queue_id = await self._create_justification_queue(
                input_data.organization_id, non_material
            )
            artifacts["justification_queue_id"] = justification_queue_id
            artifacts["topics_requiring_justification"] = len(non_material)

        provenance = self._hash(artifacts)

        # Status is AWAITING_REVIEW since human action is needed
        return StepResult(
            step_name=step_name,
            status=MaterialityStepStatus.AWAITING_REVIEW,
            artifacts=artifacts,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 7: Audit Documentation
    # -------------------------------------------------------------------------

    async def _step_audit_documentation(
        self, input_data: MaterialityAssessmentInput, pct_base: float
    ) -> StepResult:
        """
        Generate audit-ready documentation for the materiality assessment.
        Includes methodology, scoring rationale, assumptions, and evidence.

        Agents invoked:
            - greenlang.agents.foundation.citations_agent (evidence linking)
            - greenlang.agents.foundation.assumptions_registry (assumptions doc)
            - greenlang.agents.reporting.assurance_preparation_agent (evidence pkg)
        """
        step_name = "audit_documentation"
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Generating audit documentation", pct_base + 0.02)

        # Collect all step provenance hashes for the audit trail
        audit_trail = {
            step_name: {
                "provenance_hash": result.provenance_hash,
                "status": result.status.value,
                "duration_seconds": result.duration_seconds,
            }
            for step_name, result in self._step_results.items()
        }

        # Generate methodology documentation
        methodology_doc = await self._generate_methodology_document(
            input_data.organization_id,
            input_data.assessment_year,
            input_data.impact_threshold,
            input_data.financial_threshold,
        )
        artifacts["methodology_document_id"] = methodology_doc.get("document_id", "")

        # Generate assumptions log
        assumptions_doc = await self._generate_assumptions_document(
            input_data.organization_id
        )
        artifacts["assumptions_document_id"] = assumptions_doc.get("document_id", "")

        # Generate evidence package
        evidence_pkg = await self._generate_evidence_package(
            input_data.organization_id, audit_trail
        )
        artifacts["evidence_package_id"] = evidence_pkg.get("package_id", "")
        artifacts["audit_trail"] = audit_trail
        artifacts["ready_for_assurance"] = True

        provenance = self._hash(artifacts)

        return StepResult(
            step_name=step_name,
            status=MaterialityStepStatus.COMPLETED,
            artifacts=artifacts,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Agent Invocation Helpers
    # -------------------------------------------------------------------------

    async def _analyze_sector_profile(
        self, org_id: str, sector_codes: List[str]
    ) -> Dict[str, Any]:
        """Analyze sector profile for materiality context."""
        await asyncio.sleep(0)
        return {"sectors": sector_codes, "high_impact_sector": False}

    async def _map_value_chain(
        self, org_id: str, description: str
    ) -> Dict[str, Any]:
        """Map the organization's value chain stages."""
        await asyncio.sleep(0)
        return {
            "stages": ["upstream_supply", "own_operations", "downstream_use", "end_of_life"],
        }

    async def _get_sector_materiality_hints(
        self, sector_codes: List[str]
    ) -> Dict[str, Any]:
        """Get sector-specific materiality hints from EFRAG guidance."""
        await asyncio.sleep(0)
        return {"likely_material": [], "likely_not_material": []}

    async def _score_single_impact(
        self, org_id: str, topic_id: str,
        stakeholders: List[StakeholderGroup],
        sector_hints: Dict[str, Any],
    ) -> ImpactScore:
        """Score a single topic on impact materiality."""
        await asyncio.sleep(0)
        return ImpactScore(
            topic_id=topic_id,
            scale=0.0, scope=0.0, irremediability=0.0, likelihood=0.0,
            severity_score=0.0, final_score=0.0,
            time_horizon=TimeHorizon.MEDIUM_TERM,
            rationale="Scoring pending; requires agent integration.",
        )

    async def _score_single_financial(
        self, org_id: str, topic_id: str
    ) -> FinancialScore:
        """Score a single topic on financial materiality."""
        await asyncio.sleep(0)
        return FinancialScore(
            topic_id=topic_id,
            magnitude=0.0, likelihood=0.0, final_score=0.0,
            time_horizon=TimeHorizon.MEDIUM_TERM,
            rationale="Scoring pending; requires agent integration.",
        )

    async def _log_scoring_assumptions(
        self, org_id: str, dimension: str, scores: List[Dict[str, Any]]
    ) -> None:
        """Log scoring assumptions via greenlang.agents.foundation.assumptions_registry."""
        await asyncio.sleep(0)

    async def _create_review_queue(
        self, org_id: str, topics: List[Dict[str, Any]], reviewers: List[str]
    ) -> str:
        """Create a human review queue for materiality topics."""
        await asyncio.sleep(0)
        return str(uuid.uuid4())

    async def _create_justification_queue(
        self, org_id: str, non_material_topics: List[str]
    ) -> str:
        """Create queue for justifying non-material topic exclusions."""
        await asyncio.sleep(0)
        return str(uuid.uuid4())

    async def _generate_methodology_document(
        self, org_id: str, year: int, impact_threshold: float, fin_threshold: float
    ) -> Dict[str, Any]:
        """Generate methodology documentation for auditors."""
        await asyncio.sleep(0)
        return {"document_id": str(uuid.uuid4())}

    async def _generate_assumptions_document(
        self, org_id: str
    ) -> Dict[str, Any]:
        """Generate assumptions documentation."""
        await asyncio.sleep(0)
        return {"document_id": str(uuid.uuid4())}

    async def _generate_evidence_package(
        self, org_id: str, audit_trail: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate evidence package for assurance."""
        await asyncio.sleep(0)
        return {"package_id": str(uuid.uuid4())}

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _notify(self, step: str, message: str, pct: float) -> None:
        """Send progress notification."""
        if self._progress_callback:
            try:
                self._progress_callback(step, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for step=%s", step)

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
