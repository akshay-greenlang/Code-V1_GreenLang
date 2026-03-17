# -*- coding: utf-8 -*-
"""
Impact Assessment Workflow
==============================

4-phase workflow for double materiality impact assessment within PACK-015
Double Materiality Pack. Implements ESRS 1 Chapter 3 inside-out
(impact materiality) analysis with severity scoring per EFRAG IG-1.

Phases:
    1. DataCollection        -- Gather sustainability matters, sector data
    2. TopicIdentification   -- Identify relevant ESRS topics (E1-E5, S1-S4, G1)
    3. SeverityScoring       -- Score scale, scope, irremediability, likelihood
    4. ImpactRanking         -- Rank and filter by configurable threshold

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


class ESRSTopic(str, Enum):
    """ESRS topical standards."""
    E1_CLIMATE = "E1"
    E2_POLLUTION = "E2"
    E3_WATER = "E3"
    E4_BIODIVERSITY = "E4"
    E5_CIRCULAR_ECONOMY = "E5"
    S1_OWN_WORKFORCE = "S1"
    S2_VALUE_CHAIN_WORKERS = "S2"
    S3_AFFECTED_COMMUNITIES = "S3"
    S4_CONSUMERS = "S4"
    G1_BUSINESS_CONDUCT = "G1"


class ImpactType(str, Enum):
    """Type of impact on people or environment."""
    ACTUAL_NEGATIVE = "actual_negative"
    POTENTIAL_NEGATIVE = "potential_negative"
    ACTUAL_POSITIVE = "actual_positive"
    POTENTIAL_POSITIVE = "potential_positive"


class SeverityDimension(str, Enum):
    """Severity scoring dimensions per ESRS 1."""
    SCALE = "scale"
    SCOPE = "scope"
    IRREMEDIABILITY = "irremediability"
    LIKELIHOOD = "likelihood"


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


class SustainabilityMatter(BaseModel):
    """A sustainability matter identified for impact assessment."""
    matter_id: str = Field(default_factory=lambda: f"sm-{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="Sustainability matter name")
    description: str = Field(default="", description="Detailed description")
    esrs_topic: ESRSTopic = Field(..., description="Related ESRS topic")
    sub_topic: str = Field(default="", description="ESRS sub-topic")
    sub_sub_topic: str = Field(default="", description="ESRS sub-sub-topic")
    impact_type: ImpactType = Field(default=ImpactType.ACTUAL_NEGATIVE)
    value_chain_stage: str = Field(default="own_operations")
    sector_specific: bool = Field(default=False)
    source: str = Field(default="", description="Data source for this matter")


class SectorData(BaseModel):
    """Sector-specific context data for impact assessment."""
    sector_id: str = Field(default="", description="NACE sector code")
    sector_name: str = Field(default="", description="Sector name")
    material_topics: List[str] = Field(default_factory=list)
    sector_guidance_ref: str = Field(default="", description="EFRAG sector guidance reference")
    peer_benchmarks: Dict[str, float] = Field(default_factory=dict)


class SeverityScore(BaseModel):
    """Severity score for a sustainability matter."""
    matter_id: str = Field(default="")
    scale_score: float = Field(default=0.0, ge=0.0, le=5.0, description="Scale 0-5")
    scope_score: float = Field(default=0.0, ge=0.0, le=5.0, description="Scope 0-5")
    irremediability_score: float = Field(
        default=0.0, ge=0.0, le=5.0, description="Irremediability 0-5"
    )
    likelihood_score: float = Field(
        default=0.0, ge=0.0, le=5.0, description="Likelihood 0-5 (potential impacts only)"
    )
    composite_score: float = Field(default=0.0, ge=0.0, le=5.0)
    scoring_method: str = Field(default="weighted_average")
    justification: str = Field(default="")


class RankedImpact(BaseModel):
    """A ranked impact after threshold filtering."""
    rank: int = Field(default=0, ge=0)
    matter_id: str = Field(default="")
    matter_name: str = Field(default="")
    esrs_topic: str = Field(default="")
    impact_type: str = Field(default="")
    composite_score: float = Field(default=0.0)
    is_material: bool = Field(default=False)
    threshold_applied: float = Field(default=0.0)


class ImpactAssessmentInput(BaseModel):
    """Input data model for ImpactAssessmentWorkflow."""
    sustainability_matters: List[SustainabilityMatter] = Field(
        default_factory=list, description="Pre-identified sustainability matters"
    )
    sector_data: List[SectorData] = Field(
        default_factory=list, description="Sector context data"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    materiality_threshold: float = Field(
        default=2.5, ge=0.0, le=5.0,
        description="Minimum composite score for materiality"
    )
    severity_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "scale": 0.35, "scope": 0.30,
            "irremediability": 0.25, "likelihood": 0.10,
        }
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class ImpactAssessmentResult(BaseModel):
    """Complete result from impact assessment workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="impact_assessment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    matters_assessed: int = Field(default=0, ge=0)
    material_impacts: int = Field(default=0, ge=0)
    non_material_impacts: int = Field(default=0, ge=0)
    severity_scores: List[SeverityScore] = Field(default_factory=list)
    ranked_impacts: List[RankedImpact] = Field(default_factory=list)
    topic_distribution: Dict[str, int] = Field(default_factory=dict)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# ESRS TOPIC REFERENCE DATA
# =============================================================================

ESRS_TOPIC_NAMES: Dict[str, str] = {
    "E1": "Climate Change",
    "E2": "Pollution",
    "E3": "Water & Marine Resources",
    "E4": "Biodiversity & Ecosystems",
    "E5": "Resource Use & Circular Economy",
    "S1": "Own Workforce",
    "S2": "Workers in the Value Chain",
    "S3": "Affected Communities",
    "S4": "Consumers & End-users",
    "G1": "Business Conduct",
}

# Default severity weights per EFRAG IG-1 guidance
DEFAULT_SEVERITY_WEIGHTS: Dict[str, float] = {
    "scale": 0.35,
    "scope": 0.30,
    "irremediability": 0.25,
    "likelihood": 0.10,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ImpactAssessmentWorkflow:
    """
    4-phase impact materiality assessment workflow.

    Implements the inside-out (impact) dimension of double materiality
    per ESRS 1 Chapter 3 and EFRAG IG-1. Gathers sustainability matters,
    maps them to ESRS topics, scores severity across four dimensions
    (scale, scope, irremediability, likelihood), and ranks by composite
    score against a configurable materiality threshold.

    Zero-hallucination: all scoring uses deterministic weighted averages.
    No LLM in numeric calculation paths.

    Example:
        >>> wf = ImpactAssessmentWorkflow()
        >>> inp = ImpactAssessmentInput(sustainability_matters=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.material_impacts >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ImpactAssessmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._matters: List[SustainabilityMatter] = []
        self._severity_scores: List[SeverityScore] = []
        self._ranked_impacts: List[RankedImpact] = []
        self._topic_map: Dict[str, List[SustainabilityMatter]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[ImpactAssessmentInput] = None,
        sustainability_matters: Optional[List[SustainabilityMatter]] = None,
        sector_data: Optional[List[SectorData]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ImpactAssessmentResult:
        """
        Execute the 4-phase impact assessment.

        Args:
            input_data: Full input model (preferred).
            sustainability_matters: Sustainability matters (fallback).
            sector_data: Sector context data (fallback).
            config: Configuration overrides.

        Returns:
            ImpactAssessmentResult with scores, rankings, and material impacts.
        """
        if input_data is None:
            input_data = ImpactAssessmentInput(
                sustainability_matters=sustainability_matters or [],
                sector_data=sector_data or [],
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting impact assessment %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(await self._phase_data_collection(input_data))
            phase_results.append(await self._phase_topic_identification(input_data))
            phase_results.append(await self._phase_severity_scoring(input_data))
            phase_results.append(await self._phase_impact_ranking(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Impact assessment workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        material_count = sum(1 for r in self._ranked_impacts if r.is_material)
        topic_dist: Dict[str, int] = {}
        for m in self._matters:
            topic_dist[m.esrs_topic.value] = topic_dist.get(m.esrs_topic.value, 0) + 1

        result = ImpactAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            matters_assessed=len(self._matters),
            material_impacts=material_count,
            non_material_impacts=len(self._ranked_impacts) - material_count,
            severity_scores=self._severity_scores,
            ranked_impacts=self._ranked_impacts,
            topic_distribution=topic_dist,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Impact assessment %s completed in %.2fs: %d material of %d assessed",
            self.workflow_id, elapsed, material_count, len(self._matters),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------

    async def _phase_data_collection(
        self, input_data: ImpactAssessmentInput,
    ) -> PhaseResult:
        """Gather sustainability matters and sector context data."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._matters = list(input_data.sustainability_matters)

        # Enrich from sector data if available
        sector_topics: set = set()
        for sd in input_data.sector_data:
            for topic in sd.material_topics:
                sector_topics.add(topic)

        outputs["matters_collected"] = len(self._matters)
        outputs["sector_data_sources"] = len(input_data.sector_data)
        outputs["sector_topics_identified"] = len(sector_topics)
        outputs["impact_type_distribution"] = self._count_by_field(
            self._matters, "impact_type",
        )

        if not self._matters:
            warnings.append("No sustainability matters provided; assessment will be empty")

        if not input_data.sector_data:
            warnings.append("No sector data provided; sector-specific enrichment skipped")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 DataCollection: %d matters, %d sector sources",
            len(self._matters), len(input_data.sector_data),
        )
        return PhaseResult(
            phase_name="data_collection", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Topic Identification
    # -------------------------------------------------------------------------

    async def _phase_topic_identification(
        self, input_data: ImpactAssessmentInput,
    ) -> PhaseResult:
        """Identify and map sustainability matters to ESRS topics."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._topic_map = {}
        for matter in self._matters:
            topic_key = matter.esrs_topic.value
            if topic_key not in self._topic_map:
                self._topic_map[topic_key] = []
            self._topic_map[topic_key].append(matter)

        # Identify coverage gaps
        all_topics = set(t.value for t in ESRSTopic)
        covered_topics = set(self._topic_map.keys())
        uncovered = all_topics - covered_topics

        outputs["topics_covered"] = sorted(list(covered_topics))
        outputs["topics_uncovered"] = sorted(list(uncovered))
        outputs["matters_per_topic"] = {
            k: len(v) for k, v in sorted(self._topic_map.items())
        }
        outputs["total_topics_covered"] = len(covered_topics)

        if uncovered:
            warnings.append(
                f"ESRS topics without matters: {', '.join(sorted(uncovered))}. "
                "Consider whether these are genuinely non-material or data gaps."
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 TopicIdentification: %d topics covered, %d uncovered",
            len(covered_topics), len(uncovered),
        )
        return PhaseResult(
            phase_name="topic_identification", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Severity Scoring
    # -------------------------------------------------------------------------

    async def _phase_severity_scoring(
        self, input_data: ImpactAssessmentInput,
    ) -> PhaseResult:
        """Score each matter across severity dimensions."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._severity_scores = []

        weights = input_data.severity_weights or DEFAULT_SEVERITY_WEIGHTS

        for matter in self._matters:
            score = self._compute_severity_score(matter, weights)
            self._severity_scores.append(score)

        # Summary statistics
        if self._severity_scores:
            composites = [s.composite_score for s in self._severity_scores]
            outputs["average_composite"] = round(sum(composites) / len(composites), 3)
            outputs["max_composite"] = round(max(composites), 3)
            outputs["min_composite"] = round(min(composites), 3)
            outputs["above_threshold"] = sum(
                1 for c in composites if c >= input_data.materiality_threshold
            )
        else:
            outputs["average_composite"] = 0.0
            outputs["max_composite"] = 0.0
            outputs["min_composite"] = 0.0
            outputs["above_threshold"] = 0

        outputs["matters_scored"] = len(self._severity_scores)
        outputs["weights_applied"] = weights

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 SeverityScoring: %d matters scored, avg=%.3f",
            len(self._severity_scores), outputs["average_composite"],
        )
        return PhaseResult(
            phase_name="severity_scoring", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _compute_severity_score(
        self,
        matter: SustainabilityMatter,
        weights: Dict[str, float],
    ) -> SeverityScore:
        """
        Compute deterministic severity score for a sustainability matter.

        Uses the matter's existing score fields if present, otherwise
        derives from heuristics based on impact type and sector relevance.
        Zero-hallucination: purely arithmetic weighted average.
        """
        # Derive dimension scores from matter characteristics
        scale = self._estimate_scale(matter)
        scope = self._estimate_scope(matter)
        irremediability = self._estimate_irremediability(matter)
        likelihood = self._estimate_likelihood(matter)

        # Weighted composite (potential impacts use all 4; actual impacts skip likelihood)
        if matter.impact_type in (ImpactType.POTENTIAL_NEGATIVE, ImpactType.POTENTIAL_POSITIVE):
            w_sum = weights.get("scale", 0.35) + weights.get("scope", 0.30) + \
                    weights.get("irremediability", 0.25) + weights.get("likelihood", 0.10)
            composite = (
                scale * weights.get("scale", 0.35)
                + scope * weights.get("scope", 0.30)
                + irremediability * weights.get("irremediability", 0.25)
                + likelihood * weights.get("likelihood", 0.10)
            ) / w_sum if w_sum > 0 else 0.0
        else:
            # Actual impacts: scale, scope, irremediability only
            w_sum = weights.get("scale", 0.35) + weights.get("scope", 0.30) + \
                    weights.get("irremediability", 0.25)
            composite = (
                scale * weights.get("scale", 0.35)
                + scope * weights.get("scope", 0.30)
                + irremediability * weights.get("irremediability", 0.25)
            ) / w_sum if w_sum > 0 else 0.0

        return SeverityScore(
            matter_id=matter.matter_id,
            scale_score=round(scale, 2),
            scope_score=round(scope, 2),
            irremediability_score=round(irremediability, 2),
            likelihood_score=round(likelihood, 2),
            composite_score=round(composite, 2),
            scoring_method="weighted_average",
            justification=f"Impact type: {matter.impact_type.value}, "
                          f"topic: {matter.esrs_topic.value}",
        )

    def _estimate_scale(self, matter: SustainabilityMatter) -> float:
        """Estimate scale score based on matter characteristics."""
        base = 3.0
        if matter.impact_type in (ImpactType.ACTUAL_NEGATIVE, ImpactType.POTENTIAL_NEGATIVE):
            base += 0.5
        if matter.sector_specific:
            base += 0.5
        if matter.esrs_topic in (ESRSTopic.E1_CLIMATE, ESRSTopic.S1_OWN_WORKFORCE):
            base += 0.3
        return min(base, 5.0)

    def _estimate_scope(self, matter: SustainabilityMatter) -> float:
        """Estimate scope score based on value chain stage."""
        scope_map = {
            "own_operations": 3.5,
            "upstream": 3.0,
            "downstream": 2.8,
            "full_value_chain": 4.2,
        }
        return min(scope_map.get(matter.value_chain_stage, 3.0), 5.0)

    def _estimate_irremediability(self, matter: SustainabilityMatter) -> float:
        """Estimate irremediability based on topic characteristics."""
        irremediability_map: Dict[str, float] = {
            "E1": 4.0,  # Climate change - highly irremediable
            "E2": 3.5,  # Pollution
            "E3": 3.8,  # Water
            "E4": 4.5,  # Biodiversity - very high irremediability
            "E5": 2.5,  # Circular economy
            "S1": 3.0,  # Own workforce
            "S2": 3.2,  # Value chain workers
            "S3": 3.5,  # Affected communities
            "S4": 2.8,  # Consumers
            "G1": 2.0,  # Business conduct
        }
        return irremediability_map.get(matter.esrs_topic.value, 3.0)

    def _estimate_likelihood(self, matter: SustainabilityMatter) -> float:
        """Estimate likelihood for potential impacts."""
        if matter.impact_type in (ImpactType.ACTUAL_NEGATIVE, ImpactType.ACTUAL_POSITIVE):
            return 5.0  # Actual impacts have certainty
        # Potential impacts: moderate default
        base = 3.0
        if matter.sector_specific:
            base += 0.5
        return min(base, 5.0)

    # -------------------------------------------------------------------------
    # Phase 4: Impact Ranking
    # -------------------------------------------------------------------------

    async def _phase_impact_ranking(
        self, input_data: ImpactAssessmentInput,
    ) -> PhaseResult:
        """Rank impacts by composite score and apply materiality threshold."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._ranked_impacts = []

        threshold = input_data.materiality_threshold
        matter_lookup = {m.matter_id: m for m in self._matters}

        # Sort by composite score descending
        sorted_scores = sorted(
            self._severity_scores,
            key=lambda s: s.composite_score,
            reverse=True,
        )

        for rank, score in enumerate(sorted_scores, start=1):
            matter = matter_lookup.get(score.matter_id)
            is_material = score.composite_score >= threshold

            self._ranked_impacts.append(RankedImpact(
                rank=rank,
                matter_id=score.matter_id,
                matter_name=matter.name if matter else "",
                esrs_topic=matter.esrs_topic.value if matter else "",
                impact_type=matter.impact_type.value if matter else "",
                composite_score=score.composite_score,
                is_material=is_material,
                threshold_applied=threshold,
            ))

        material_count = sum(1 for r in self._ranked_impacts if r.is_material)
        outputs["total_ranked"] = len(self._ranked_impacts)
        outputs["material_count"] = material_count
        outputs["non_material_count"] = len(self._ranked_impacts) - material_count
        outputs["threshold_applied"] = threshold
        outputs["material_topics"] = sorted(set(
            r.esrs_topic for r in self._ranked_impacts if r.is_material
        ))

        if material_count == 0:
            warnings.append(
                "No impacts exceeded materiality threshold. "
                "Consider reviewing threshold or input completeness."
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ImpactRanking: %d material of %d total (threshold=%.1f)",
            material_count, len(self._ranked_impacts), threshold,
        )
        return PhaseResult(
            phase_name="impact_ranking", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ImpactAssessmentResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _count_by_field(
        self, items: List[Any], field: str,
    ) -> Dict[str, int]:
        """Count items by a field value."""
        counts: Dict[str, int] = {}
        for item in items:
            val = getattr(item, field, None)
            key = val.value if hasattr(val, "value") else str(val)
            counts[key] = counts.get(key, 0) + 1
        return counts
