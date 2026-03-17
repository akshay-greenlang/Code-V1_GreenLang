# -*- coding: utf-8 -*-
"""
Materiality Matrix Workflow
================================

3-phase workflow for constructing the double materiality matrix within
PACK-015 Double Materiality Pack. Aggregates impact and financial scores,
generates a 2x2 matrix with four quadrants, and applies sector-specific
thresholds per ESRS 1 / EFRAG IG-1.

Phases:
    1. ScoreAggregation      -- Combine impact and financial scores per topic
    2. MatrixGeneration      -- Build 2x2 matrix with quadrant classification
    3. ThresholdApplication  -- Apply sector-specific materiality thresholds

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


class MatrixQuadrant(str, Enum):
    """Double materiality matrix quadrants."""
    DOUBLE_MATERIAL = "double_material"           # High impact + High financial
    IMPACT_MATERIAL_ONLY = "impact_material_only"  # High impact + Low financial
    FINANCIAL_MATERIAL_ONLY = "financial_material_only"  # Low impact + High financial
    NOT_MATERIAL = "not_material"                   # Low impact + Low financial


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


class TopicScore(BaseModel):
    """Aggregated scores for a single ESRS topic."""
    topic_id: str = Field(default="", description="ESRS topic code")
    topic_name: str = Field(default="")
    impact_score: float = Field(default=0.0, ge=0.0, le=5.0)
    financial_score: float = Field(default=0.0, ge=0.0, le=5.0)
    impact_matter_count: int = Field(default=0, ge=0)
    financial_exposure_count: int = Field(default=0, ge=0)


class MatrixEntry(BaseModel):
    """An entry in the materiality matrix."""
    topic_id: str = Field(default="", description="ESRS topic code")
    topic_name: str = Field(default="")
    impact_score: float = Field(default=0.0, ge=0.0, le=5.0)
    financial_score: float = Field(default=0.0, ge=0.0, le=5.0)
    quadrant: MatrixQuadrant = Field(default=MatrixQuadrant.NOT_MATERIAL)
    is_material: bool = Field(default=False)
    x_position_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="X position for visualization (financial axis, 0-100%)"
    )
    y_position_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Y position for visualization (impact axis, 0-100%)"
    )


class ThresholdResult(BaseModel):
    """Result of threshold application to a topic."""
    topic_id: str = Field(default="")
    topic_name: str = Field(default="")
    impact_threshold: float = Field(default=0.0)
    financial_threshold: float = Field(default=0.0)
    impact_passes: bool = Field(default=False)
    financial_passes: bool = Field(default=False)
    final_materiality: str = Field(
        default="not_material",
        description="double_material|impact_only|financial_only|not_material",
    )


class MaterialityMatrixInput(BaseModel):
    """Input data model for MaterialityMatrixWorkflow."""
    topic_scores: List[TopicScore] = Field(
        default_factory=list, description="Pre-computed topic scores"
    )
    impact_threshold: float = Field(
        default=2.5, ge=0.0, le=5.0,
        description="Minimum impact score for materiality"
    )
    financial_threshold: float = Field(
        default=2.5, ge=0.0, le=5.0,
        description="Minimum financial score for materiality"
    )
    sector_adjustments: Dict[str, float] = Field(
        default_factory=dict,
        description="Topic-specific threshold adjustments by sector",
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    prior_year_matrix: List[MatrixEntry] = Field(
        default_factory=list, description="Previous year matrix for comparison"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class MaterialityMatrixResult(BaseModel):
    """Complete result from materiality matrix workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="materiality_matrix")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    topics_assessed: int = Field(default=0, ge=0)
    matrix_entries: List[MatrixEntry] = Field(default_factory=list)
    threshold_results: List[ThresholdResult] = Field(default_factory=list)
    quadrant_distribution: Dict[str, int] = Field(default_factory=dict)
    material_topics: List[str] = Field(default_factory=list)
    non_material_topics: List[str] = Field(default_factory=list)
    year_over_year_changes: List[Dict[str, Any]] = Field(default_factory=list)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# REFERENCE DATA
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


class MaterialityMatrixWorkflow:
    """
    3-phase materiality matrix construction workflow.

    Aggregates impact and financial assessment scores per ESRS topic,
    generates a 2x2 double materiality matrix with four quadrants
    (double material, impact-only, financial-only, not material),
    and applies configurable sector-specific thresholds.

    Zero-hallucination: all quadrant classification and threshold logic
    is purely deterministic arithmetic. No LLM in numeric paths.

    Example:
        >>> wf = MaterialityMatrixWorkflow()
        >>> inp = MaterialityMatrixInput(topic_scores=[...])
        >>> result = await wf.execute(inp)
        >>> assert len(result.material_topics) >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize MaterialityMatrixWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._topic_scores: List[TopicScore] = []
        self._matrix_entries: List[MatrixEntry] = []
        self._threshold_results: List[ThresholdResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[MaterialityMatrixInput] = None,
        topic_scores: Optional[List[TopicScore]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> MaterialityMatrixResult:
        """
        Execute the 3-phase materiality matrix workflow.

        Args:
            input_data: Full input model (preferred).
            topic_scores: Topic scores (fallback).
            config: Configuration overrides.

        Returns:
            MaterialityMatrixResult with matrix entries, thresholds, quadrants.
        """
        if input_data is None:
            input_data = MaterialityMatrixInput(
                topic_scores=topic_scores or [],
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting materiality matrix %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(await self._phase_score_aggregation(input_data))
            phase_results.append(await self._phase_matrix_generation(input_data))
            phase_results.append(await self._phase_threshold_application(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error(
                "Materiality matrix workflow failed: %s", exc, exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        quadrant_dist: Dict[str, int] = {}
        for entry in self._matrix_entries:
            quadrant_dist[entry.quadrant.value] = (
                quadrant_dist.get(entry.quadrant.value, 0) + 1
            )

        material_topics = [
            e.topic_id for e in self._matrix_entries if e.is_material
        ]
        non_material_topics = [
            e.topic_id for e in self._matrix_entries if not e.is_material
        ]

        # Year-over-year comparison
        yoy_changes = self._compute_yoy_changes(input_data)

        result = MaterialityMatrixResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            topics_assessed=len(self._topic_scores),
            matrix_entries=self._matrix_entries,
            threshold_results=self._threshold_results,
            quadrant_distribution=quadrant_dist,
            material_topics=material_topics,
            non_material_topics=non_material_topics,
            year_over_year_changes=yoy_changes,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Materiality matrix %s completed in %.2fs: %d material, %d non-material",
            self.workflow_id, elapsed, len(material_topics), len(non_material_topics),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Score Aggregation
    # -------------------------------------------------------------------------

    async def _phase_score_aggregation(
        self, input_data: MaterialityMatrixInput,
    ) -> PhaseResult:
        """Combine impact and financial scores per topic."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._topic_scores = list(input_data.topic_scores)

        if self._topic_scores:
            impact_scores = [ts.impact_score for ts in self._topic_scores]
            financial_scores = [ts.financial_score for ts in self._topic_scores]
            outputs["topics_count"] = len(self._topic_scores)
            outputs["avg_impact_score"] = round(
                sum(impact_scores) / len(impact_scores), 3,
            )
            outputs["avg_financial_score"] = round(
                sum(financial_scores) / len(financial_scores), 3,
            )
            outputs["max_impact_score"] = round(max(impact_scores), 3)
            outputs["max_financial_score"] = round(max(financial_scores), 3)
        else:
            outputs["topics_count"] = 0
            outputs["avg_impact_score"] = 0.0
            outputs["avg_financial_score"] = 0.0
            outputs["max_impact_score"] = 0.0
            outputs["max_financial_score"] = 0.0
            warnings.append("No topic scores provided for aggregation")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 ScoreAggregation: %d topics aggregated",
            len(self._topic_scores),
        )
        return PhaseResult(
            phase_name="score_aggregation", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Matrix Generation
    # -------------------------------------------------------------------------

    async def _phase_matrix_generation(
        self, input_data: MaterialityMatrixInput,
    ) -> PhaseResult:
        """Build 2x2 materiality matrix with quadrant assignments."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._matrix_entries = []
        impact_threshold = input_data.impact_threshold
        financial_threshold = input_data.financial_threshold

        for ts in self._topic_scores:
            quadrant = self._classify_quadrant(
                ts.impact_score, ts.financial_score,
                impact_threshold, financial_threshold,
            )
            is_material = quadrant != MatrixQuadrant.NOT_MATERIAL

            # Normalize to 0-100% for visualization
            x_pct = min(ts.financial_score / 5.0 * 100.0, 100.0)
            y_pct = min(ts.impact_score / 5.0 * 100.0, 100.0)

            self._matrix_entries.append(MatrixEntry(
                topic_id=ts.topic_id,
                topic_name=ts.topic_name or ESRS_TOPIC_NAMES.get(ts.topic_id, ""),
                impact_score=ts.impact_score,
                financial_score=ts.financial_score,
                quadrant=quadrant,
                is_material=is_material,
                x_position_pct=round(x_pct, 1),
                y_position_pct=round(y_pct, 1),
            ))

        quadrant_counts: Dict[str, int] = {}
        for entry in self._matrix_entries:
            quadrant_counts[entry.quadrant.value] = (
                quadrant_counts.get(entry.quadrant.value, 0) + 1
            )

        outputs["matrix_entries"] = len(self._matrix_entries)
        outputs["quadrant_distribution"] = quadrant_counts
        outputs["impact_threshold"] = impact_threshold
        outputs["financial_threshold"] = financial_threshold

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 MatrixGeneration: %d entries placed in matrix",
            len(self._matrix_entries),
        )
        return PhaseResult(
            phase_name="matrix_generation", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _classify_quadrant(
        self,
        impact: float,
        financial: float,
        impact_threshold: float,
        financial_threshold: float,
    ) -> MatrixQuadrant:
        """Classify a topic into a matrix quadrant based on thresholds."""
        impact_passes = impact >= impact_threshold
        financial_passes = financial >= financial_threshold

        if impact_passes and financial_passes:
            return MatrixQuadrant.DOUBLE_MATERIAL
        elif impact_passes and not financial_passes:
            return MatrixQuadrant.IMPACT_MATERIAL_ONLY
        elif not impact_passes and financial_passes:
            return MatrixQuadrant.FINANCIAL_MATERIAL_ONLY
        else:
            return MatrixQuadrant.NOT_MATERIAL

    # -------------------------------------------------------------------------
    # Phase 3: Threshold Application
    # -------------------------------------------------------------------------

    async def _phase_threshold_application(
        self, input_data: MaterialityMatrixInput,
    ) -> PhaseResult:
        """Apply sector-specific materiality thresholds."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._threshold_results = []

        base_impact_threshold = input_data.impact_threshold
        base_financial_threshold = input_data.financial_threshold
        adjustments = input_data.sector_adjustments

        for entry in self._matrix_entries:
            # Apply sector adjustment if available
            adjustment = adjustments.get(entry.topic_id, 0.0)
            adj_impact_threshold = max(base_impact_threshold + adjustment, 0.0)
            adj_financial_threshold = max(base_financial_threshold + adjustment, 0.0)

            impact_passes = entry.impact_score >= adj_impact_threshold
            financial_passes = entry.financial_score >= adj_financial_threshold

            if impact_passes and financial_passes:
                final = "double_material"
            elif impact_passes:
                final = "impact_only"
            elif financial_passes:
                final = "financial_only"
            else:
                final = "not_material"

            # Update matrix entry materiality based on adjusted thresholds
            entry.is_material = final != "not_material"
            if impact_passes and financial_passes:
                entry.quadrant = MatrixQuadrant.DOUBLE_MATERIAL
            elif impact_passes:
                entry.quadrant = MatrixQuadrant.IMPACT_MATERIAL_ONLY
            elif financial_passes:
                entry.quadrant = MatrixQuadrant.FINANCIAL_MATERIAL_ONLY
            else:
                entry.quadrant = MatrixQuadrant.NOT_MATERIAL

            self._threshold_results.append(ThresholdResult(
                topic_id=entry.topic_id,
                topic_name=entry.topic_name,
                impact_threshold=round(adj_impact_threshold, 2),
                financial_threshold=round(adj_financial_threshold, 2),
                impact_passes=impact_passes,
                financial_passes=financial_passes,
                final_materiality=final,
            ))

        material_count = sum(1 for tr in self._threshold_results if tr.final_materiality != "not_material")
        outputs["threshold_results_count"] = len(self._threshold_results)
        outputs["material_after_thresholds"] = material_count
        outputs["non_material_after_thresholds"] = len(self._threshold_results) - material_count
        outputs["sector_adjustments_applied"] = len(adjustments)

        if material_count == 0:
            warnings.append(
                "No topics are material after threshold application. "
                "Review thresholds and sector adjustments."
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ThresholdApplication: %d material after sector thresholds",
            material_count,
        )
        return PhaseResult(
            phase_name="threshold_application", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_yoy_changes(
        self, input_data: MaterialityMatrixInput,
    ) -> List[Dict[str, Any]]:
        """Compute year-over-year changes between current and prior matrix."""
        if not input_data.prior_year_matrix:
            return []

        prior_lookup = {
            e.topic_id: e for e in input_data.prior_year_matrix
        }
        changes: List[Dict[str, Any]] = []

        for current in self._matrix_entries:
            prior = prior_lookup.get(current.topic_id)
            if prior:
                impact_delta = round(current.impact_score - prior.impact_score, 2)
                financial_delta = round(current.financial_score - prior.financial_score, 2)
                quadrant_changed = current.quadrant != prior.quadrant

                if impact_delta != 0 or financial_delta != 0 or quadrant_changed:
                    changes.append({
                        "topic_id": current.topic_id,
                        "topic_name": current.topic_name,
                        "impact_delta": impact_delta,
                        "financial_delta": financial_delta,
                        "prior_quadrant": prior.quadrant.value if hasattr(prior.quadrant, "value") else str(prior.quadrant),
                        "current_quadrant": current.quadrant.value,
                        "quadrant_changed": quadrant_changed,
                    })

        return changes

    def _compute_provenance(self, result: MaterialityMatrixResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
