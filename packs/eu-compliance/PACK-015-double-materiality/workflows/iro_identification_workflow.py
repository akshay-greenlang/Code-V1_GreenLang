# -*- coding: utf-8 -*-
"""
IRO Identification Workflow
================================

4-phase workflow for Impacts, Risks, and Opportunities (IRO) identification
within PACK-015 Double Materiality Pack. Maps value chain stages, discovers
IROs per ESRS topic, classifies them, and prioritizes by composite scoring.

Phases:
    1. ValueChainMapping    -- Map value chain stages (upstream/own/downstream)
    2. IRODiscovery         -- Identify IROs per ESRS topic
    3. IROClassification    -- Classify as impact, risk, or opportunity
    4. IROPrioritization    -- Score and prioritize IROs

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


class ValueChainStage(str, Enum):
    """Value chain stages per ESRS 1."""
    RAW_MATERIALS = "raw_materials"
    MANUFACTURING = "manufacturing"
    DISTRIBUTION = "distribution"
    OWN_OPERATIONS = "own_operations"
    PRODUCT_USE = "product_use"
    END_OF_LIFE = "end_of_life"
    FINANCE = "finance"


class IROType(str, Enum):
    """Classification of an IRO."""
    IMPACT_ACTUAL_NEGATIVE = "impact_actual_negative"
    IMPACT_POTENTIAL_NEGATIVE = "impact_potential_negative"
    IMPACT_ACTUAL_POSITIVE = "impact_actual_positive"
    IMPACT_POTENTIAL_POSITIVE = "impact_potential_positive"
    RISK = "risk"
    OPPORTUNITY = "opportunity"


class IROStatus(str, Enum):
    """Processing status of an IRO."""
    IDENTIFIED = "identified"
    CLASSIFIED = "classified"
    SCORED = "scored"
    PRIORITIZED = "prioritized"


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


class ValueChainActivity(BaseModel):
    """An activity within a value chain stage."""
    activity_id: str = Field(default_factory=lambda: f"vca-{uuid.uuid4().hex[:8]}")
    stage: ValueChainStage = Field(...)
    name: str = Field(..., description="Activity name")
    description: str = Field(default="")
    geography: str = Field(default="", description="Geographic scope")
    stakeholders_affected: List[str] = Field(default_factory=list)
    esrs_topics_relevant: List[str] = Field(default_factory=list)


class IRORecord(BaseModel):
    """An Impacts, Risks, or Opportunities record."""
    iro_id: str = Field(default_factory=lambda: f"iro-{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="IRO name")
    description: str = Field(default="")
    esrs_topic: str = Field(default="", description="ESRS topic code (E1-G1)")
    sub_topic: str = Field(default="")
    iro_type: IROType = Field(default=IROType.RISK)
    status: IROStatus = Field(default=IROStatus.IDENTIFIED)
    value_chain_stages: List[ValueChainStage] = Field(default_factory=list)
    affected_stakeholders: List[str] = Field(default_factory=list)
    source: str = Field(default="", description="Data source or reference")
    severity_score: float = Field(default=0.0, ge=0.0, le=5.0)
    financial_score: float = Field(default=0.0, ge=0.0, le=5.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=5.0)
    priority_rank: int = Field(default=0, ge=0)


class IROIdentificationInput(BaseModel):
    """Input data model for IROIdentificationWorkflow."""
    value_chain_activities: List[ValueChainActivity] = Field(
        default_factory=list, description="Value chain activity records"
    )
    initial_iros: List[IRORecord] = Field(
        default_factory=list, description="Pre-identified IROs"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    sector_code: str = Field(default="", description="NACE sector code")
    prioritization_weights: Dict[str, float] = Field(
        default_factory=lambda: {"severity": 0.50, "financial": 0.50}
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class IROIdentificationResult(BaseModel):
    """Complete result from IRO identification workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="iro_identification")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    value_chain_stages_mapped: int = Field(default=0, ge=0)
    activities_mapped: int = Field(default=0, ge=0)
    iros_identified: int = Field(default=0, ge=0)
    iros_by_type: Dict[str, int] = Field(default_factory=dict)
    iros_by_topic: Dict[str, int] = Field(default_factory=dict)
    prioritized_iros: List[IRORecord] = Field(default_factory=list)
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

# Default IRO scoring: stage relevance weights
STAGE_RELEVANCE: Dict[str, float] = {
    "raw_materials": 0.85,
    "manufacturing": 0.90,
    "distribution": 0.70,
    "own_operations": 1.00,
    "product_use": 0.75,
    "end_of_life": 0.65,
    "finance": 0.50,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class IROIdentificationWorkflow:
    """
    4-phase IRO identification and prioritization workflow.

    Maps value chain activities, discovers IROs per ESRS topic, classifies
    them as impacts (actual/potential, positive/negative), risks, or
    opportunities, and prioritizes using composite severity + financial
    scoring.

    Zero-hallucination: all scoring uses deterministic weighted formulas.
    No LLM in numeric paths.

    Example:
        >>> wf = IROIdentificationWorkflow()
        >>> inp = IROIdentificationInput(initial_iros=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.iros_identified > 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize IROIdentificationWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._activities: List[ValueChainActivity] = []
        self._iros: List[IRORecord] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[IROIdentificationInput] = None,
        value_chain_activities: Optional[List[ValueChainActivity]] = None,
        initial_iros: Optional[List[IRORecord]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> IROIdentificationResult:
        """
        Execute the 4-phase IRO identification workflow.

        Args:
            input_data: Full input model (preferred).
            value_chain_activities: Value chain activities (fallback).
            initial_iros: Pre-identified IROs (fallback).
            config: Configuration overrides.

        Returns:
            IROIdentificationResult with classified and prioritized IROs.
        """
        if input_data is None:
            input_data = IROIdentificationInput(
                value_chain_activities=value_chain_activities or [],
                initial_iros=initial_iros or [],
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting IRO identification %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(await self._phase_value_chain_mapping(input_data))
            phase_results.append(await self._phase_iro_discovery(input_data))
            phase_results.append(await self._phase_iro_classification(input_data))
            phase_results.append(await self._phase_iro_prioritization(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error(
                "IRO identification workflow failed: %s", exc, exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        iros_by_type: Dict[str, int] = {}
        iros_by_topic: Dict[str, int] = {}
        for iro in self._iros:
            iros_by_type[iro.iro_type.value] = iros_by_type.get(iro.iro_type.value, 0) + 1
            iros_by_topic[iro.esrs_topic] = iros_by_topic.get(iro.esrs_topic, 0) + 1

        stages_mapped = len(set(a.stage.value for a in self._activities))

        result = IROIdentificationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            value_chain_stages_mapped=stages_mapped,
            activities_mapped=len(self._activities),
            iros_identified=len(self._iros),
            iros_by_type=iros_by_type,
            iros_by_topic=iros_by_topic,
            prioritized_iros=self._iros,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "IRO identification %s completed in %.2fs: %d IROs across %d stages",
            self.workflow_id, elapsed, len(self._iros), stages_mapped,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Value Chain Mapping
    # -------------------------------------------------------------------------

    async def _phase_value_chain_mapping(
        self, input_data: IROIdentificationInput,
    ) -> PhaseResult:
        """Map value chain stages and activities."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._activities = list(input_data.value_chain_activities)

        stage_counts: Dict[str, int] = {}
        stage_topics: Dict[str, set] = {}
        for activity in self._activities:
            stage_key = activity.stage.value
            stage_counts[stage_key] = stage_counts.get(stage_key, 0) + 1
            if stage_key not in stage_topics:
                stage_topics[stage_key] = set()
            for topic in activity.esrs_topics_relevant:
                stage_topics[stage_key].add(topic)

        outputs["activities_mapped"] = len(self._activities)
        outputs["stages_covered"] = len(stage_counts)
        outputs["stage_distribution"] = stage_counts
        outputs["topics_by_stage"] = {
            k: sorted(list(v)) for k, v in stage_topics.items()
        }

        # Check for upstream/downstream coverage
        all_stages = set(s.value for s in ValueChainStage)
        covered = set(stage_counts.keys())
        uncovered = all_stages - covered
        if uncovered:
            warnings.append(
                f"Value chain stages not covered: {', '.join(sorted(uncovered))}"
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 ValueChainMapping: %d activities across %d stages",
            len(self._activities), len(stage_counts),
        )
        return PhaseResult(
            phase_name="value_chain_mapping", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: IRO Discovery
    # -------------------------------------------------------------------------

    async def _phase_iro_discovery(
        self, input_data: IROIdentificationInput,
    ) -> PhaseResult:
        """Identify IROs per ESRS topic from value chain activities."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._iros = list(input_data.initial_iros)

        # Cross-reference activities with topics to discover additional IROs
        existing_topics = set(iro.esrs_topic for iro in self._iros)
        activity_topics: set = set()
        for activity in self._activities:
            for topic in activity.esrs_topics_relevant:
                activity_topics.add(topic)

        new_topics = activity_topics - existing_topics

        outputs["pre_existing_iros"] = len(input_data.initial_iros)
        outputs["total_iros_after_discovery"] = len(self._iros)
        outputs["topics_from_activities"] = sorted(list(activity_topics))
        outputs["new_topics_discovered"] = sorted(list(new_topics))

        if not self._iros:
            warnings.append("No IROs identified; provide initial_iros or value chain data")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 IRODiscovery: %d IROs, %d topics from activities",
            len(self._iros), len(activity_topics),
        )
        return PhaseResult(
            phase_name="iro_discovery", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: IRO Classification
    # -------------------------------------------------------------------------

    async def _phase_iro_classification(
        self, input_data: IROIdentificationInput,
    ) -> PhaseResult:
        """Classify IROs as impact, risk, or opportunity."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        type_counts: Dict[str, int] = {}
        for iro in self._iros:
            iro.status = IROStatus.CLASSIFIED
            type_counts[iro.iro_type.value] = type_counts.get(iro.iro_type.value, 0) + 1

        impacts = sum(
            v for k, v in type_counts.items() if k.startswith("impact_")
        )
        risks = type_counts.get("risk", 0)
        opportunities = type_counts.get("opportunity", 0)

        outputs["classification_distribution"] = type_counts
        outputs["total_impacts"] = impacts
        outputs["total_risks"] = risks
        outputs["total_opportunities"] = opportunities
        outputs["iros_classified"] = len(self._iros)

        if impacts == 0 and risks == 0 and opportunities == 0:
            warnings.append("No IROs classified; check iro_type assignments")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 IROClassification: %d impacts, %d risks, %d opportunities",
            impacts, risks, opportunities,
        )
        return PhaseResult(
            phase_name="iro_classification", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: IRO Prioritization
    # -------------------------------------------------------------------------

    async def _phase_iro_prioritization(
        self, input_data: IROIdentificationInput,
    ) -> PhaseResult:
        """Score and prioritize IROs by composite severity + financial score."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        weights = input_data.prioritization_weights
        w_sev = weights.get("severity", 0.50)
        w_fin = weights.get("financial", 0.50)
        w_sum = w_sev + w_fin

        for iro in self._iros:
            # Apply value chain stage relevance to severity
            stage_factor = max(
                (STAGE_RELEVANCE.get(s.value, 0.5) for s in iro.value_chain_stages),
                default=0.5,
            )
            adjusted_severity = iro.severity_score * stage_factor

            # Composite score
            if w_sum > 0:
                iro.composite_score = round(
                    (adjusted_severity * w_sev + iro.financial_score * w_fin) / w_sum, 2,
                )
            else:
                iro.composite_score = 0.0
            iro.status = IROStatus.PRIORITIZED

        # Rank by composite score
        self._iros.sort(key=lambda i: i.composite_score, reverse=True)
        for rank, iro in enumerate(self._iros, start=1):
            iro.priority_rank = rank

        if self._iros:
            composites = [i.composite_score for i in self._iros]
            outputs["max_composite"] = round(max(composites), 3)
            outputs["min_composite"] = round(min(composites), 3)
            outputs["average_composite"] = round(
                sum(composites) / len(composites), 3,
            )
        else:
            outputs["max_composite"] = 0.0
            outputs["min_composite"] = 0.0
            outputs["average_composite"] = 0.0

        outputs["iros_prioritized"] = len(self._iros)
        outputs["top_5_iros"] = [
            {"rank": i.priority_rank, "name": i.name, "score": i.composite_score}
            for i in self._iros[:5]
        ]

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 IROPrioritization: %d IROs ranked", len(self._iros),
        )
        return PhaseResult(
            phase_name="iro_prioritization", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: IROIdentificationResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
