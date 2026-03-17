# -*- coding: utf-8 -*-
"""
DMA Update Workflow
========================

4-phase workflow for annual DMA refresh within PACK-015 Double Materiality
Pack. Detects regulatory and business changes, re-assesses affected
sustainability matters, computes deltas against the prior DMA, and
publishes the updated assessment.

Phases:
    1. ChangeDetection    -- Detect regulatory/business/market changes
    2. ReAssessment       -- Re-score changed matters
    3. DeltaAnalysis      -- Compare with previous DMA
    4. UpdatePublication   -- Publish updated DMA with audit trail

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


class ChangeType(str, Enum):
    """Types of changes triggering DMA update."""
    REGULATORY = "regulatory"
    BUSINESS_MODEL = "business_model"
    MARKET = "market"
    STAKEHOLDER = "stakeholder"
    SECTOR = "sector"
    SCIENTIFIC = "scientific"
    GEOGRAPHIC = "geographic"


class ChangeSeverity(str, Enum):
    """Severity of a detected change."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PublicationStatus(str, Enum):
    """Status of the DMA update publication."""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    PUBLISHED = "published"


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


class DetectedChange(BaseModel):
    """A detected change that may affect materiality assessment."""
    change_id: str = Field(default_factory=lambda: f"chg-{uuid.uuid4().hex[:8]}")
    change_type: ChangeType = Field(...)
    severity: ChangeSeverity = Field(default=ChangeSeverity.MEDIUM)
    description: str = Field(default="")
    affected_topics: List[str] = Field(default_factory=list)
    effective_date: str = Field(default="", description="ISO date")
    source: str = Field(default="")
    requires_rescoring: bool = Field(default=True)


class PriorDMARecord(BaseModel):
    """Record from the previous DMA for comparison."""
    topic_id: str = Field(default="")
    topic_name: str = Field(default="")
    prior_impact_score: float = Field(default=0.0, ge=0.0, le=5.0)
    prior_financial_score: float = Field(default=0.0, ge=0.0, le=5.0)
    prior_materiality_type: str = Field(default="not_material")
    prior_year: int = Field(default=2024)


class ReAssessedTopic(BaseModel):
    """A topic that has been re-assessed after change detection."""
    topic_id: str = Field(default="")
    topic_name: str = Field(default="")
    prior_impact_score: float = Field(default=0.0, ge=0.0, le=5.0)
    new_impact_score: float = Field(default=0.0, ge=0.0, le=5.0)
    prior_financial_score: float = Field(default=0.0, ge=0.0, le=5.0)
    new_financial_score: float = Field(default=0.0, ge=0.0, le=5.0)
    change_triggers: List[str] = Field(default_factory=list)
    rescoring_justification: str = Field(default="")


class DeltaEntry(BaseModel):
    """Delta between current and prior DMA for a topic."""
    topic_id: str = Field(default="")
    topic_name: str = Field(default="")
    impact_delta: float = Field(default=0.0)
    financial_delta: float = Field(default=0.0)
    prior_materiality: str = Field(default="not_material")
    current_materiality: str = Field(default="not_material")
    materiality_changed: bool = Field(default=False)
    direction: str = Field(
        default="unchanged",
        description="upgraded|downgraded|unchanged|new|removed",
    )


class DMAUpdateInput(BaseModel):
    """Input data model for DMAUpdateWorkflow."""
    detected_changes: List[DetectedChange] = Field(
        default_factory=list, description="Pre-identified changes"
    )
    prior_dma: List[PriorDMARecord] = Field(
        default_factory=list, description="Previous DMA results"
    )
    current_scores: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Current topic scores: {topic_id: {impact: X, financial: Y}}",
    )
    impact_threshold: float = Field(default=2.5, ge=0.0, le=5.0)
    financial_threshold: float = Field(default=2.5, ge=0.0, le=5.0)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    prior_year: int = Field(default=2024, ge=2019, le=2049)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    auto_publish: bool = Field(default=False)
    config: Dict[str, Any] = Field(default_factory=dict)


class DMAUpdateResult(BaseModel):
    """Complete result from DMA update workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="dma_update")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    changes_detected: int = Field(default=0, ge=0)
    topics_rescored: int = Field(default=0, ge=0)
    deltas: List[DeltaEntry] = Field(default_factory=list)
    materiality_changes: int = Field(default=0, ge=0)
    topics_upgraded: int = Field(default=0, ge=0)
    topics_downgraded: int = Field(default=0, ge=0)
    publication_status: PublicationStatus = Field(default=PublicationStatus.DRAFT)
    reporting_year: int = Field(default=2025)
    prior_year: int = Field(default=2024)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class DMAUpdateWorkflow:
    """
    4-phase DMA update workflow for annual refresh.

    Detects regulatory, business, and market changes that affect
    materiality, re-scores affected topics, computes deltas against
    the prior year DMA, and publishes the updated assessment with
    full audit trail.

    Zero-hallucination: all delta computations and re-scoring use
    deterministic arithmetic. Change detection is input-driven, not
    predicted. No LLM in numeric paths.

    Example:
        >>> wf = DMAUpdateWorkflow()
        >>> inp = DMAUpdateInput(prior_dma=[...], detected_changes=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.materiality_changes >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize DMAUpdateWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._changes: List[DetectedChange] = []
        self._reassessed: List[ReAssessedTopic] = []
        self._deltas: List[DeltaEntry] = []
        self._publication_status: PublicationStatus = PublicationStatus.DRAFT
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[DMAUpdateInput] = None,
        detected_changes: Optional[List[DetectedChange]] = None,
        prior_dma: Optional[List[PriorDMARecord]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> DMAUpdateResult:
        """
        Execute the 4-phase DMA update workflow.

        Args:
            input_data: Full input model (preferred).
            detected_changes: Change records (fallback).
            prior_dma: Prior DMA records (fallback).
            config: Configuration overrides.

        Returns:
            DMAUpdateResult with deltas, materiality changes, publication status.
        """
        if input_data is None:
            input_data = DMAUpdateInput(
                detected_changes=detected_changes or [],
                prior_dma=prior_dma or [],
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting DMA update %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(await self._phase_change_detection(input_data))
            phase_results.append(await self._phase_reassessment(input_data))
            phase_results.append(await self._phase_delta_analysis(input_data))
            phase_results.append(await self._phase_update_publication(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error(
                "DMA update workflow failed: %s", exc, exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        mat_changes = sum(1 for d in self._deltas if d.materiality_changed)
        upgraded = sum(1 for d in self._deltas if d.direction == "upgraded")
        downgraded = sum(1 for d in self._deltas if d.direction == "downgraded")

        result = DMAUpdateResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            changes_detected=len(self._changes),
            topics_rescored=len(self._reassessed),
            deltas=self._deltas,
            materiality_changes=mat_changes,
            topics_upgraded=upgraded,
            topics_downgraded=downgraded,
            publication_status=self._publication_status,
            reporting_year=input_data.reporting_year,
            prior_year=input_data.prior_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "DMA update %s completed in %.2fs: %d changes, %d rescored, "
            "%d materiality changes",
            self.workflow_id, elapsed, len(self._changes),
            len(self._reassessed), mat_changes,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Change Detection
    # -------------------------------------------------------------------------

    async def _phase_change_detection(
        self, input_data: DMAUpdateInput,
    ) -> PhaseResult:
        """Detect regulatory, business, and market changes."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._changes = list(input_data.detected_changes)

        type_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}
        affected_topics: set = set()

        for change in self._changes:
            type_counts[change.change_type.value] = (
                type_counts.get(change.change_type.value, 0) + 1
            )
            severity_counts[change.severity.value] = (
                severity_counts.get(change.severity.value, 0) + 1
            )
            for topic in change.affected_topics:
                affected_topics.add(topic)

        outputs["changes_detected"] = len(self._changes)
        outputs["type_distribution"] = type_counts
        outputs["severity_distribution"] = severity_counts
        outputs["affected_topics"] = sorted(list(affected_topics))
        outputs["topics_requiring_rescore"] = sum(
            1 for c in self._changes if c.requires_rescoring
        )

        critical_count = severity_counts.get("critical", 0)
        if critical_count > 0:
            warnings.append(f"{critical_count} critical change(s) detected requiring immediate attention")

        if not self._changes:
            warnings.append("No changes detected; DMA update may not be necessary")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 ChangeDetection: %d changes, %d critical",
            len(self._changes), critical_count,
        )
        return PhaseResult(
            phase_name="change_detection", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Re-Assessment
    # -------------------------------------------------------------------------

    async def _phase_reassessment(
        self, input_data: DMAUpdateInput,
    ) -> PhaseResult:
        """Re-score affected sustainability matters."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._reassessed = []

        # Determine which topics need re-scoring
        topics_to_rescore: set = set()
        change_triggers: Dict[str, List[str]] = {}
        for change in self._changes:
            if change.requires_rescoring:
                for topic in change.affected_topics:
                    topics_to_rescore.add(topic)
                    if topic not in change_triggers:
                        change_triggers[topic] = []
                    change_triggers[topic].append(change.description or change.change_type.value)

        # Prior DMA lookup
        prior_lookup = {p.topic_id: p for p in input_data.prior_dma}
        current_scores = input_data.current_scores

        for topic_id in sorted(topics_to_rescore):
            prior = prior_lookup.get(topic_id)
            current = current_scores.get(topic_id, {})

            prior_impact = prior.prior_impact_score if prior else 0.0
            prior_financial = prior.prior_financial_score if prior else 0.0
            new_impact = current.get("impact", prior_impact)
            new_financial = current.get("financial", prior_financial)

            # Apply change severity adjustments
            topic_changes = [
                c for c in self._changes
                if topic_id in c.affected_topics and c.requires_rescoring
            ]
            adjustment = self._compute_change_adjustment(topic_changes)
            new_impact = min(new_impact + adjustment, 5.0)
            new_financial = min(new_financial + adjustment, 5.0)

            self._reassessed.append(ReAssessedTopic(
                topic_id=topic_id,
                topic_name=prior.topic_name if prior else "",
                prior_impact_score=prior_impact,
                new_impact_score=round(new_impact, 2),
                prior_financial_score=prior_financial,
                new_financial_score=round(new_financial, 2),
                change_triggers=change_triggers.get(topic_id, []),
                rescoring_justification=(
                    f"Re-scored due to {len(topic_changes)} change(s): "
                    f"adjustment={adjustment:+.2f}"
                ),
            ))

        outputs["topics_rescored"] = len(self._reassessed)
        outputs["topics_unchanged"] = len(input_data.prior_dma) - len(self._reassessed)
        outputs["rescore_reasons"] = {
            ra.topic_id: ra.change_triggers for ra in self._reassessed
        }

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 ReAssessment: %d topics rescored", len(self._reassessed),
        )
        return PhaseResult(
            phase_name="reassessment", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _compute_change_adjustment(
        self, changes: List[DetectedChange],
    ) -> float:
        """Compute score adjustment based on change severity."""
        if not changes:
            return 0.0

        severity_adjustments: Dict[str, float] = {
            "critical": 0.50,
            "high": 0.30,
            "medium": 0.15,
            "low": 0.05,
        }

        total = 0.0
        for change in changes:
            total += severity_adjustments.get(change.severity.value, 0.0)

        # Cap adjustment at 1.0
        return min(total, 1.0)

    # -------------------------------------------------------------------------
    # Phase 3: Delta Analysis
    # -------------------------------------------------------------------------

    async def _phase_delta_analysis(
        self, input_data: DMAUpdateInput,
    ) -> PhaseResult:
        """Compare updated DMA with previous year."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._deltas = []

        prior_lookup = {p.topic_id: p for p in input_data.prior_dma}
        reassessed_lookup = {r.topic_id: r for r in self._reassessed}
        current_scores = input_data.current_scores

        # Process all topics from prior DMA
        all_topics = set(prior_lookup.keys()) | set(current_scores.keys())

        for topic_id in sorted(all_topics):
            prior = prior_lookup.get(topic_id)
            reassessed = reassessed_lookup.get(topic_id)
            current = current_scores.get(topic_id, {})

            if reassessed:
                new_impact = reassessed.new_impact_score
                new_financial = reassessed.new_financial_score
            else:
                new_impact = current.get("impact", prior.prior_impact_score if prior else 0.0)
                new_financial = current.get("financial", prior.prior_financial_score if prior else 0.0)

            prior_impact = prior.prior_impact_score if prior else 0.0
            prior_financial = prior.prior_financial_score if prior else 0.0

            impact_delta = round(new_impact - prior_impact, 2)
            financial_delta = round(new_financial - prior_financial, 2)

            # Determine materiality status
            prior_mat = prior.prior_materiality_type if prior else "not_material"
            current_mat = self._determine_materiality(
                new_impact, new_financial,
                input_data.impact_threshold, input_data.financial_threshold,
            )

            mat_changed = prior_mat != current_mat
            direction = self._determine_direction(prior_mat, current_mat, prior is None)

            self._deltas.append(DeltaEntry(
                topic_id=topic_id,
                topic_name=prior.topic_name if prior else "",
                impact_delta=impact_delta,
                financial_delta=financial_delta,
                prior_materiality=prior_mat,
                current_materiality=current_mat,
                materiality_changed=mat_changed,
                direction=direction,
            ))

        mat_changes = sum(1 for d in self._deltas if d.materiality_changed)
        upgraded = sum(1 for d in self._deltas if d.direction == "upgraded")
        downgraded = sum(1 for d in self._deltas if d.direction == "downgraded")

        outputs["topics_analyzed"] = len(self._deltas)
        outputs["materiality_changes"] = mat_changes
        outputs["topics_upgraded"] = upgraded
        outputs["topics_downgraded"] = downgraded
        outputs["topics_unchanged"] = len(self._deltas) - mat_changes
        outputs["avg_impact_delta"] = round(
            sum(d.impact_delta for d in self._deltas) / max(len(self._deltas), 1), 3,
        )
        outputs["avg_financial_delta"] = round(
            sum(d.financial_delta for d in self._deltas) / max(len(self._deltas), 1), 3,
        )

        if mat_changes > 0:
            warnings.append(
                f"{mat_changes} topic(s) changed materiality status "
                f"({upgraded} upgraded, {downgraded} downgraded)"
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 DeltaAnalysis: %d deltas, %d materiality changes",
            len(self._deltas), mat_changes,
        )
        return PhaseResult(
            phase_name="delta_analysis", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _determine_materiality(
        self,
        impact: float,
        financial: float,
        impact_threshold: float,
        financial_threshold: float,
    ) -> str:
        """Determine materiality type from scores and thresholds."""
        i_passes = impact >= impact_threshold
        f_passes = financial >= financial_threshold
        if i_passes and f_passes:
            return "double_material"
        elif i_passes:
            return "impact_only"
        elif f_passes:
            return "financial_only"
        else:
            return "not_material"

    def _determine_direction(
        self, prior_mat: str, current_mat: str, is_new: bool,
    ) -> str:
        """Determine materiality change direction."""
        if is_new:
            return "new"
        if prior_mat == current_mat:
            return "unchanged"

        mat_rank = {
            "not_material": 0,
            "impact_only": 1,
            "financial_only": 1,
            "double_material": 2,
        }
        prior_rank = mat_rank.get(prior_mat, 0)
        current_rank = mat_rank.get(current_mat, 0)

        if current_rank > prior_rank:
            return "upgraded"
        elif current_rank < prior_rank:
            return "downgraded"
        else:
            return "changed"  # Same rank but different type

    # -------------------------------------------------------------------------
    # Phase 4: Update Publication
    # -------------------------------------------------------------------------

    async def _phase_update_publication(
        self, input_data: DMAUpdateInput,
    ) -> PhaseResult:
        """Publish updated DMA with audit trail."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Build publication record
        publication_record = {
            "workflow_id": self.workflow_id,
            "reporting_year": input_data.reporting_year,
            "prior_year": input_data.prior_year,
            "entity_id": input_data.entity_id,
            "entity_name": input_data.entity_name,
            "changes_detected": len(self._changes),
            "topics_rescored": len(self._reassessed),
            "materiality_changes": sum(1 for d in self._deltas if d.materiality_changed),
            "published_at": datetime.utcnow().isoformat(),
        }

        if input_data.auto_publish:
            self._publication_status = PublicationStatus.PUBLISHED
        else:
            self._publication_status = PublicationStatus.DRAFT

        # Compute publication provenance
        pub_hash = self._hash_dict(publication_record)

        outputs["publication_status"] = self._publication_status.value
        outputs["publication_record"] = publication_record
        outputs["publication_provenance"] = pub_hash
        outputs["auto_published"] = input_data.auto_publish

        if not input_data.auto_publish:
            warnings.append(
                "DMA update saved as draft; manual review and approval required"
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 UpdatePublication: status=%s",
            self._publication_status.value,
        )
        return PhaseResult(
            phase_name="update_publication", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=pub_hash,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: DMAUpdateResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
