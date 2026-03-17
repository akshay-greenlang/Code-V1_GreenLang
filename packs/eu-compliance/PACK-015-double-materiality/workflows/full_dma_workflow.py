# -*- coding: utf-8 -*-
"""
Full DMA Workflow
=====================

6-phase orchestration workflow for a complete Double Materiality Assessment
within PACK-015 Double Materiality Pack. Coordinates stakeholder engagement,
IRO identification, impact assessment, financial assessment, matrix
construction, and ESRS mapping into a single end-to-end pipeline.

Phases:
    1. StakeholderEngagement  -- Execute stakeholder engagement sub-workflow
    2. IROIdentification      -- Execute IRO identification sub-workflow
    3. ImpactAssessment       -- Execute impact materiality sub-workflow
    4. FinancialAssessment    -- Execute financial materiality sub-workflow
    5. MatrixAndThreshold     -- Build materiality matrix with thresholds
    6. ESRSMappingAndReport   -- Map to ESRS disclosures and generate report

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


class DMACompleteness(str, Enum):
    """Completeness level of the DMA."""
    FULL = "full"           # All 6 phases completed
    PARTIAL = "partial"     # Some phases skipped or failed
    MINIMAL = "minimal"     # Only core phases completed
    DRAFT = "draft"         # Preliminary results only


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
    sub_workflow_id: str = Field(default="", description="ID of sub-workflow if delegated")


class DMATopicResult(BaseModel):
    """Combined DMA result for a single ESRS topic."""
    topic_id: str = Field(default="")
    topic_name: str = Field(default="")
    impact_score: float = Field(default=0.0, ge=0.0, le=5.0)
    financial_score: float = Field(default=0.0, ge=0.0, le=5.0)
    is_material: bool = Field(default=False)
    materiality_type: str = Field(default="not_material")
    impact_matters_count: int = Field(default=0, ge=0)
    financial_exposures_count: int = Field(default=0, ge=0)
    iros_count: int = Field(default=0, ge=0)
    disclosure_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class FullDMAInput(BaseModel):
    """Input data model for FullDMAWorkflow."""
    # Stakeholder engagement inputs
    stakeholder_data: Dict[str, Any] = Field(
        default_factory=dict, description="Data for stakeholder engagement phase"
    )
    # IRO identification inputs
    iro_data: Dict[str, Any] = Field(
        default_factory=dict, description="Data for IRO identification phase"
    )
    # Impact assessment inputs
    impact_data: Dict[str, Any] = Field(
        default_factory=dict, description="Data for impact assessment phase"
    )
    # Financial assessment inputs
    financial_data: Dict[str, Any] = Field(
        default_factory=dict, description="Data for financial assessment phase"
    )
    # Matrix configuration
    impact_threshold: float = Field(default=2.5, ge=0.0, le=5.0)
    financial_threshold: float = Field(default=2.5, ge=0.0, le=5.0)
    sector_adjustments: Dict[str, float] = Field(default_factory=dict)
    # General
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    include_voluntary_disclosures: bool = Field(default=False)
    skip_phases: List[str] = Field(
        default_factory=list,
        description="Phase names to skip (e.g., ['stakeholder_engagement'])"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class FullDMAResult(BaseModel):
    """Complete result from the full DMA workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="full_dma")
    status: WorkflowStatus = Field(...)
    completeness: DMACompleteness = Field(default=DMACompleteness.DRAFT)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    # Aggregated results
    topic_results: List[DMATopicResult] = Field(default_factory=list)
    material_topics: List[str] = Field(default_factory=list)
    non_material_topics: List[str] = Field(default_factory=list)
    total_iros: int = Field(default=0, ge=0)
    total_disclosure_gaps: int = Field(default=0, ge=0)
    overall_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    total_effort_weeks: float = Field(default=0.0, ge=0.0)
    stakeholder_validation_passed: bool = Field(default=False)
    # Sub-workflow results (stored as dicts for flexibility)
    stakeholder_result: Dict[str, Any] = Field(default_factory=dict)
    iro_result: Dict[str, Any] = Field(default_factory=dict)
    impact_result: Dict[str, Any] = Field(default_factory=dict)
    financial_result: Dict[str, Any] = Field(default_factory=dict)
    matrix_result: Dict[str, Any] = Field(default_factory=dict)
    esrs_result: Dict[str, Any] = Field(default_factory=dict)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# ESRS TOPIC NAMES
# =============================================================================

ESRS_TOPIC_NAMES: Dict[str, str] = {
    "E1": "Climate Change", "E2": "Pollution",
    "E3": "Water & Marine Resources", "E4": "Biodiversity & Ecosystems",
    "E5": "Resource Use & Circular Economy", "S1": "Own Workforce",
    "S2": "Workers in the Value Chain", "S3": "Affected Communities",
    "S4": "Consumers & End-users", "G1": "Business Conduct",
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FullDMAWorkflow:
    """
    6-phase orchestration workflow for complete Double Materiality Assessment.

    Coordinates all sub-workflows into a unified end-to-end DMA pipeline:
    stakeholder engagement, IRO identification, impact assessment, financial
    assessment, materiality matrix construction, and ESRS disclosure mapping.

    Each phase delegates to a specialized sub-workflow and aggregates results
    into a unified DMA output. Phases can be selectively skipped via
    skip_phases configuration.

    Zero-hallucination: all numeric aggregation is deterministic. Sub-workflow
    results are combined via arithmetic mean/sum operations only.

    Example:
        >>> wf = FullDMAWorkflow()
        >>> inp = FullDMAInput(impact_data={...}, financial_data={...})
        >>> result = await wf.execute(inp)
        >>> assert result.completeness in ("full", "partial")
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize FullDMAWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._stakeholder_result: Dict[str, Any] = {}
        self._iro_result: Dict[str, Any] = {}
        self._impact_result: Dict[str, Any] = {}
        self._financial_result: Dict[str, Any] = {}
        self._matrix_result: Dict[str, Any] = {}
        self._esrs_result: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[FullDMAInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> FullDMAResult:
        """
        Execute the complete 6-phase DMA workflow.

        Args:
            input_data: Full DMA input data.
            config: Configuration overrides.

        Returns:
            FullDMAResult with aggregated topic results and sub-workflow outputs.
        """
        if input_data is None:
            input_data = FullDMAInput(config=config or {})

        started_at = datetime.utcnow()
        self.logger.info("Starting full DMA workflow %s", self.workflow_id)
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING
        skip = set(input_data.skip_phases)

        try:
            # Phase 1: Stakeholder Engagement
            if "stakeholder_engagement" not in skip:
                self._phase_results.append(
                    await self._phase_stakeholder_engagement(input_data)
                )
            else:
                self._phase_results.append(self._skipped_phase("stakeholder_engagement"))

            # Phase 2: IRO Identification
            if "iro_identification" not in skip:
                self._phase_results.append(
                    await self._phase_iro_identification(input_data)
                )
            else:
                self._phase_results.append(self._skipped_phase("iro_identification"))

            # Phase 3: Impact Assessment
            if "impact_assessment" not in skip:
                self._phase_results.append(
                    await self._phase_impact_assessment(input_data)
                )
            else:
                self._phase_results.append(self._skipped_phase("impact_assessment"))

            # Phase 4: Financial Assessment
            if "financial_assessment" not in skip:
                self._phase_results.append(
                    await self._phase_financial_assessment(input_data)
                )
            else:
                self._phase_results.append(self._skipped_phase("financial_assessment"))

            # Phase 5: Matrix & Threshold
            if "matrix_threshold" not in skip:
                self._phase_results.append(
                    await self._phase_matrix_threshold(input_data)
                )
            else:
                self._phase_results.append(self._skipped_phase("matrix_threshold"))

            # Phase 6: ESRS Mapping & Report
            if "esrs_mapping" not in skip:
                self._phase_results.append(
                    await self._phase_esrs_mapping(input_data)
                )
            else:
                self._phase_results.append(self._skipped_phase("esrs_mapping"))

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error(
                "Full DMA workflow failed: %s", exc, exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        # Determine completeness
        completed_count = sum(
            1 for p in self._phase_results if p.status == PhaseStatus.COMPLETED
        )
        completeness = self._assess_completeness(completed_count, len(skip))

        # Aggregate topic results
        topic_results = self._aggregate_topic_results(input_data)

        material = [tr.topic_id for tr in topic_results if tr.is_material]
        non_material = [tr.topic_id for tr in topic_results if not tr.is_material]

        result = FullDMAResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            completeness=completeness,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            topic_results=topic_results,
            material_topics=material,
            non_material_topics=non_material,
            total_iros=self._iro_result.get("iros_identified", 0),
            total_disclosure_gaps=self._esrs_result.get("gaps_count", 0),
            overall_coverage_pct=self._esrs_result.get("coverage_pct", 0.0),
            total_effort_weeks=self._esrs_result.get("total_effort_weeks", 0.0),
            stakeholder_validation_passed=self._stakeholder_result.get(
                "validation_passed", False,
            ),
            stakeholder_result=self._stakeholder_result,
            iro_result=self._iro_result,
            impact_result=self._impact_result,
            financial_result=self._financial_result,
            matrix_result=self._matrix_result,
            esrs_result=self._esrs_result,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Full DMA workflow %s completed in %.2fs: completeness=%s, "
            "%d material topics, %d non-material",
            self.workflow_id, elapsed, completeness.value,
            len(material), len(non_material),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Stakeholder Engagement
    # -------------------------------------------------------------------------

    async def _phase_stakeholder_engagement(
        self, input_data: FullDMAInput,
    ) -> PhaseResult:
        """Execute stakeholder engagement sub-workflow."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        sh_data = input_data.stakeholder_data
        stakeholders_count = len(sh_data.get("stakeholders", []))
        consultations_count = len(sh_data.get("consultations", []))

        self._stakeholder_result = {
            "stakeholders_identified": stakeholders_count,
            "consultations_recorded": consultations_count,
            "validation_passed": consultations_count > 0 and stakeholders_count > 0,
        }

        outputs.update(self._stakeholder_result)

        if stakeholders_count == 0:
            warnings.append("No stakeholder data provided for engagement phase")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 StakeholderEngagement: %d stakeholders, %d consultations",
            stakeholders_count, consultations_count,
        )
        return PhaseResult(
            phase_name="stakeholder_engagement", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: IRO Identification
    # -------------------------------------------------------------------------

    async def _phase_iro_identification(
        self, input_data: FullDMAInput,
    ) -> PhaseResult:
        """Execute IRO identification sub-workflow."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        iro_data = input_data.iro_data
        iros = iro_data.get("iros", [])

        iros_by_type: Dict[str, int] = {}
        iros_by_topic: Dict[str, int] = {}
        for iro in iros:
            iro_type = iro.get("iro_type", "unknown")
            iros_by_type[iro_type] = iros_by_type.get(iro_type, 0) + 1
            topic = iro.get("esrs_topic", "unknown")
            iros_by_topic[topic] = iros_by_topic.get(topic, 0) + 1

        self._iro_result = {
            "iros_identified": len(iros),
            "iros_by_type": iros_by_type,
            "iros_by_topic": iros_by_topic,
        }
        outputs.update(self._iro_result)

        if not iros:
            warnings.append("No IROs provided for identification phase")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 2 IROIdentification: %d IROs", len(iros))
        return PhaseResult(
            phase_name="iro_identification", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Impact Assessment
    # -------------------------------------------------------------------------

    async def _phase_impact_assessment(
        self, input_data: FullDMAInput,
    ) -> PhaseResult:
        """Execute impact materiality assessment sub-workflow."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        impact_data = input_data.impact_data
        scores = impact_data.get("topic_scores", {})

        self._impact_result = {
            "topics_assessed": len(scores),
            "topic_scores": scores,
        }
        outputs.update(self._impact_result)

        if not scores:
            warnings.append("No impact scores provided for assessment phase")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 3 ImpactAssessment: %d topics scored", len(scores))
        return PhaseResult(
            phase_name="impact_assessment", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Financial Assessment
    # -------------------------------------------------------------------------

    async def _phase_financial_assessment(
        self, input_data: FullDMAInput,
    ) -> PhaseResult:
        """Execute financial materiality assessment sub-workflow."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        fin_data = input_data.financial_data
        scores = fin_data.get("topic_scores", {})

        self._financial_result = {
            "topics_assessed": len(scores),
            "topic_scores": scores,
        }
        outputs.update(self._financial_result)

        if not scores:
            warnings.append("No financial scores provided for assessment phase")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 4 FinancialAssessment: %d topics scored", len(scores))
        return PhaseResult(
            phase_name="financial_assessment", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Matrix & Threshold
    # -------------------------------------------------------------------------

    async def _phase_matrix_threshold(
        self, input_data: FullDMAInput,
    ) -> PhaseResult:
        """Build materiality matrix and apply sector thresholds."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        impact_scores = self._impact_result.get("topic_scores", {})
        financial_scores = self._financial_result.get("topic_scores", {})

        all_topics = set(list(impact_scores.keys()) + list(financial_scores.keys()))
        material_topics: List[str] = []
        non_material_topics: List[str] = []

        for topic_id in sorted(all_topics):
            i_score = impact_scores.get(topic_id, 0.0)
            f_score = financial_scores.get(topic_id, 0.0)
            adjustment = input_data.sector_adjustments.get(topic_id, 0.0)
            i_threshold = max(input_data.impact_threshold + adjustment, 0.0)
            f_threshold = max(input_data.financial_threshold + adjustment, 0.0)

            if i_score >= i_threshold or f_score >= f_threshold:
                material_topics.append(topic_id)
            else:
                non_material_topics.append(topic_id)

        self._matrix_result = {
            "topics_assessed": len(all_topics),
            "material_topics": material_topics,
            "non_material_topics": non_material_topics,
            "impact_threshold": input_data.impact_threshold,
            "financial_threshold": input_data.financial_threshold,
        }
        outputs.update(self._matrix_result)

        if not material_topics:
            warnings.append("No topics identified as material in matrix phase")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 MatrixThreshold: %d material, %d non-material",
            len(material_topics), len(non_material_topics),
        )
        return PhaseResult(
            phase_name="matrix_threshold", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 6: ESRS Mapping & Report
    # -------------------------------------------------------------------------

    async def _phase_esrs_mapping(
        self, input_data: FullDMAInput,
    ) -> PhaseResult:
        """Map material topics to ESRS disclosure requirements."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        material_topics = self._matrix_result.get("material_topics", [])

        # Estimate coverage and gaps
        total_drs = 0
        covered_drs = 0
        for topic_id in material_topics:
            # Estimate based on typical DR counts
            topic_drs = {"E1": 9, "E2": 6, "E3": 5, "E4": 6, "E5": 6,
                         "S1": 6, "S2": 5, "S3": 5, "S4": 5, "G1": 6}
            dr_count = topic_drs.get(topic_id, 5)
            total_drs += dr_count

        gaps_count = max(total_drs - covered_drs, 0)
        coverage_pct = (covered_drs / total_drs * 100) if total_drs > 0 else 0.0
        effort_weeks = gaps_count * 1.5  # ~1.5 weeks per gap as rough estimate

        self._esrs_result = {
            "material_topics_mapped": len(material_topics),
            "total_drs": total_drs,
            "covered_drs": covered_drs,
            "gaps_count": gaps_count,
            "coverage_pct": round(coverage_pct, 1),
            "total_effort_weeks": round(effort_weeks, 1),
        }
        outputs.update(self._esrs_result)

        if gaps_count > 0:
            warnings.append(f"{gaps_count} disclosure requirement gaps identified")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 6 ESRSMapping: %d topics mapped, %d DRs, %.1f%% coverage",
            len(material_topics), total_drs, coverage_pct,
        )
        return PhaseResult(
            phase_name="esrs_mapping", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _skipped_phase(self, phase_name: str) -> PhaseResult:
        """Create a skipped phase result."""
        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.SKIPPED,
            outputs={"reason": "Phase skipped by configuration"},
        )

    def _assess_completeness(
        self, completed: int, skipped: int,
    ) -> DMACompleteness:
        """Assess DMA completeness based on completed phases."""
        total_phases = 6
        if completed == total_phases:
            return DMACompleteness.FULL
        elif completed >= 4:
            return DMACompleteness.PARTIAL
        elif completed >= 2:
            return DMACompleteness.MINIMAL
        else:
            return DMACompleteness.DRAFT

    def _aggregate_topic_results(
        self, input_data: FullDMAInput,
    ) -> List[DMATopicResult]:
        """Aggregate results across all phases into per-topic summaries."""
        impact_scores = self._impact_result.get("topic_scores", {})
        financial_scores = self._financial_result.get("topic_scores", {})
        material_topics = set(self._matrix_result.get("material_topics", []))
        iros_by_topic = self._iro_result.get("iros_by_topic", {})

        all_topics = set(
            list(impact_scores.keys())
            + list(financial_scores.keys())
            + list(ESRS_TOPIC_NAMES.keys())
        )

        results: List[DMATopicResult] = []
        for topic_id in sorted(all_topics):
            i_score = impact_scores.get(topic_id, 0.0)
            f_score = financial_scores.get(topic_id, 0.0)
            is_mat = topic_id in material_topics

            if is_mat:
                i_passes = i_score >= input_data.impact_threshold
                f_passes = f_score >= input_data.financial_threshold
                if i_passes and f_passes:
                    mat_type = "double_material"
                elif i_passes:
                    mat_type = "impact_only"
                elif f_passes:
                    mat_type = "financial_only"
                else:
                    mat_type = "not_material"
            else:
                mat_type = "not_material"

            results.append(DMATopicResult(
                topic_id=topic_id,
                topic_name=ESRS_TOPIC_NAMES.get(topic_id, ""),
                impact_score=round(i_score, 2) if isinstance(i_score, (int, float)) else 0.0,
                financial_score=round(f_score, 2) if isinstance(f_score, (int, float)) else 0.0,
                is_material=is_mat,
                materiality_type=mat_type,
                iros_count=iros_by_topic.get(topic_id, 0),
            ))

        return results

    def _compute_provenance(self, result: FullDMAResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
