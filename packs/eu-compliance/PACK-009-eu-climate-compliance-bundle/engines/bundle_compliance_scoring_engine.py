# -*- coding: utf-8 -*-
"""
BundleComplianceScoringEngine - PACK-009 EU Climate Compliance Bundle Engine 7

Computes weighted compliance scores across all 4 EU regulations in the bundle
(CSRD, CBAM, EU Taxonomy, EUDR). Includes maturity assessment (5 levels per
regulation), risk-adjusted scoring with deadline proximity boosting, heatmap
data generation, improvement recommendations, and benchmark comparison.

Capabilities:
    1. Weighted compliance score calculation per regulation
    2. Industry-configurable regulation weights
    3. Five-level maturity assessment per regulation
    4. Risk-adjusted scoring (deadline proximity, severity multiplier)
    5. Heatmap data generation for visual dashboards
    6. Improvement recommendations based on gaps
    7. Benchmark comparison against industry averages

Scoring Method:
    raw_score     = (items_compliant / items_assessed) * 100
    risk_adj      = raw_score * (1 - risk_penalty)
    deadline_adj  = risk_adj * (1 + near_deadline_boost) if near_deadline
    weighted      = deadline_adj * regulation_weight / sum(weights)
    overall       = sum(weighted_i for all i)

Zero-Hallucination:
    - All scores use deterministic arithmetic formulae
    - Maturity levels derived from threshold tables, not ML
    - Recommendations from a fixed rule-based catalogue
    - SHA-256 provenance hash on all results

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-009 EU Climate Compliance Bundle
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    elif isinstance(data, list):
        serializable = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in data
        ]
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning *default* when denominator is zero."""
    if denominator == 0.0:
        return default
    return numerator / denominator


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a value within [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MaturityLevel(int, Enum):
    """Five-level maturity scale."""
    INITIAL = 1
    DEVELOPING = 2
    DEFINED = 3
    MANAGED = 4
    OPTIMIZED = 5


class RiskSeverity(str, Enum):
    """Risk severity categories."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class HeatmapStatus(str, Enum):
    """Status for heatmap cells."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------


DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "default": {"CSRD": 0.30, "CBAM": 0.25, "EU_TAXONOMY": 0.25, "EUDR": 0.20},
    "manufacturing": {"CSRD": 0.25, "CBAM": 0.35, "EU_TAXONOMY": 0.20, "EUDR": 0.20},
    "agriculture": {"CSRD": 0.20, "CBAM": 0.15, "EU_TAXONOMY": 0.20, "EUDR": 0.45},
    "financial_services": {"CSRD": 0.35, "CBAM": 0.10, "EU_TAXONOMY": 0.45, "EUDR": 0.10},
    "energy": {"CSRD": 0.25, "CBAM": 0.35, "EU_TAXONOMY": 0.30, "EUDR": 0.10},
    "construction": {"CSRD": 0.30, "CBAM": 0.30, "EU_TAXONOMY": 0.30, "EUDR": 0.10},
    "retail": {"CSRD": 0.25, "CBAM": 0.20, "EU_TAXONOMY": 0.20, "EUDR": 0.35},
    "food_beverage": {"CSRD": 0.20, "CBAM": 0.15, "EU_TAXONOMY": 0.20, "EUDR": 0.45},
}


MATURITY_DEFINITIONS: Dict[str, Dict[int, Dict[str, str]]] = {
    "CSRD": {
        1: {"name": "Initial", "description": "Ad-hoc sustainability reporting with no structured process. ESRS awareness is minimal."},
        2: {"name": "Developing", "description": "Double materiality assessment initiated. Basic ESRS gap analysis underway. Some data collection processes established."},
        3: {"name": "Defined", "description": "Structured ESRS reporting process defined. Data collection pipelines operational. First CSRD report drafted."},
        4: {"name": "Managed", "description": "CSRD reporting integrated into financial reporting cycle. Assurance-ready data. Automated XBRL tagging."},
        5: {"name": "Optimized", "description": "Fully assured CSRD report. Continuous improvement. Leading practice disclosures exceeding requirements."},
    },
    "CBAM": {
        1: {"name": "Initial", "description": "Limited awareness of CBAM obligations. No import-level emission tracking."},
        2: {"name": "Developing", "description": "CBAM goods categories identified. Quarterly reporting initiated with default values."},
        3: {"name": "Defined", "description": "Supplier-specific emission data collection underway. Transitional declarations filed on time."},
        4: {"name": "Managed", "description": "Automated CBAM reporting. Certificate management active. Supplier data largely verified."},
        5: {"name": "Optimized", "description": "Full precursor chain visibility. Cost-optimised certificate strategy. NCA audit-ready."},
    },
    "EU_TAXONOMY": {
        1: {"name": "Initial", "description": "Basic awareness of EU Taxonomy. No activity screening performed."},
        2: {"name": "Developing", "description": "Economic activities screened for eligibility. Initial KPI calculations (turnover/CapEx/OpEx)."},
        3: {"name": "Defined", "description": "Substantial contribution and DNSH assessments underway. Minimum safeguards evaluation started."},
        4: {"name": "Managed", "description": "Full alignment assessment complete. Taxonomy KPIs integrated into financial reporting."},
        5: {"name": "Optimized", "description": "Taxonomy-aligned strategy drives investment decisions. Continuous activity monitoring and optimisation."},
    },
    "EUDR": {
        1: {"name": "Initial", "description": "Minimal awareness of EUDR requirements. No commodity traceability in place."},
        2: {"name": "Developing", "description": "Relevant commodities identified. Initial supply chain mapping and geolocation collection started."},
        3: {"name": "Defined", "description": "Risk assessment framework in place. Due diligence system operational. Satellite monitoring initiated."},
        4: {"name": "Managed", "description": "Full supply chain traceability for all relevant commodities. Automated DDS submission. Risk mitigation active."},
        5: {"name": "Optimized", "description": "Real-time deforestation monitoring. Proactive supplier engagement. NCA inspection-ready with complete audit trail."},
    },
}

INDUSTRY_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "default": {"CSRD": 55.0, "CBAM": 45.0, "EU_TAXONOMY": 40.0, "EUDR": 35.0},
    "manufacturing": {"CSRD": 60.0, "CBAM": 55.0, "EU_TAXONOMY": 45.0, "EUDR": 40.0},
    "agriculture": {"CSRD": 50.0, "CBAM": 35.0, "EU_TAXONOMY": 35.0, "EUDR": 55.0},
    "financial_services": {"CSRD": 65.0, "CBAM": 30.0, "EU_TAXONOMY": 60.0, "EUDR": 25.0},
    "energy": {"CSRD": 60.0, "CBAM": 60.0, "EU_TAXONOMY": 55.0, "EUDR": 30.0},
}


# Upcoming compliance deadlines (month/year) for deadline-proximity boost
_REGULATION_DEADLINES: Dict[str, str] = {
    "CSRD": "2026-07",
    "CBAM": "2026-01",
    "EU_TAXONOMY": "2026-07",
    "EUDR": "2026-12",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ScoringConfig(BaseModel):
    """Configuration for the BundleComplianceScoringEngine."""

    regulation_weights: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS["default"]),
        description="Weight per regulation (must sum to 1.0)",
    )
    risk_multiplier: float = Field(
        default=1.0, ge=0.5, le=3.0,
        description="Multiplier applied to risk penalties",
    )
    maturity_levels: int = Field(
        default=5, ge=3, le=7,
        description="Number of maturity levels in the model",
    )
    near_deadline_boost: float = Field(
        default=0.10, ge=0.0, le=0.50,
        description="Score boost factor when a regulation deadline is within 6 months",
    )
    industry: str = Field(
        default="default",
        description="Industry sector for weight and benchmark selection",
    )
    risk_flags_threshold: float = Field(
        default=60.0, ge=0.0, le=100.0,
        description="Score below which a risk flag is raised",
    )

    @field_validator("regulation_weights")
    @classmethod
    def _validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
        return v


# ---------------------------------------------------------------------------
# Data Models - Inputs
# ---------------------------------------------------------------------------


class RegulationInput(BaseModel):
    """Input data for scoring a single regulation."""

    regulation: str = Field(..., description="Regulation identifier (CSRD, CBAM, EU_TAXONOMY, EUDR)")
    items_assessed: int = Field(default=0, ge=0, description="Total items assessed")
    items_compliant: int = Field(default=0, ge=0, description="Items compliant")
    items_non_compliant: int = Field(default=0, ge=0, description="Items non-compliant")
    items_pending: int = Field(default=0, ge=0, description="Items pending")
    evidence_count: int = Field(default=0, ge=0, description="Number of evidence artefacts")
    risk_flags: List[str] = Field(default_factory=list, description="Risk flag descriptions")
    near_deadline: bool = Field(default=False, description="Is a compliance deadline approaching?")
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Data quality 0-100")
    gaps: List[str] = Field(default_factory=list, description="Identified compliance gaps")


# ---------------------------------------------------------------------------
# Data Models - Outputs
# ---------------------------------------------------------------------------


class MaturityAssessment(BaseModel):
    """Maturity assessment for a single regulation."""

    regulation: str = Field(..., description="Regulation identifier")
    level: int = Field(default=1, ge=1, le=5, description="Maturity level 1-5")
    level_name: str = Field(default="Initial", description="Maturity level name")
    description: str = Field(default="", description="Maturity level description")
    evidence_count: int = Field(default=0, ge=0, description="Number of evidence artefacts")
    gaps_remaining: int = Field(default=0, ge=0, description="Number of open gaps")
    next_level_requirements: List[str] = Field(
        default_factory=list, description="What is needed to reach the next level"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class ComplianceScore(BaseModel):
    """Compliance score for a single regulation."""

    regulation: str = Field(..., description="Regulation identifier")
    raw_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Raw compliance score 0-100")
    weighted_score: float = Field(default=0.0, description="Weighted contribution to bundle score")
    risk_adjusted_score: float = Field(default=0.0, description="Score after risk adjustment")
    maturity_level: int = Field(default=1, ge=1, le=5, description="Maturity level 1-5")
    risk_flags: List[str] = Field(default_factory=list, description="Active risk flags")
    near_deadline: bool = Field(default=False, description="Is a deadline approaching?")
    weight: float = Field(default=0.0, description="Regulation weight")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class HeatmapCell(BaseModel):
    """A single cell in the compliance heatmap."""

    regulation: str = Field(..., description="Regulation identifier")
    dimension: str = Field(..., description="Compliance dimension")
    score: float = Field(default=0.0, description="Score for this cell")
    status: str = Field(default="GREEN", description="Heatmap colour status")


class Recommendation(BaseModel):
    """A single improvement recommendation."""

    recommendation_id: str = Field(default_factory=_new_uuid, description="Recommendation identifier")
    regulation: str = Field(..., description="Target regulation")
    priority: str = Field(default="MEDIUM", description="Priority level (LOW/MEDIUM/HIGH/CRITICAL)")
    title: str = Field(default="", description="Short recommendation title")
    description: str = Field(default="", description="Detailed recommendation")
    effort_estimate: str = Field(default="", description="Estimated effort (LOW/MEDIUM/HIGH)")
    impact_estimate: str = Field(default="", description="Expected impact (LOW/MEDIUM/HIGH)")


class BenchmarkComparison(BaseModel):
    """Comparison of scores against industry benchmarks."""

    regulation: str = Field(..., description="Regulation identifier")
    company_score: float = Field(default=0.0, description="Company's compliance score")
    benchmark_score: float = Field(default=0.0, description="Industry benchmark score")
    delta: float = Field(default=0.0, description="Difference (company - benchmark)")
    position: str = Field(default="", description="Position relative to benchmark (ABOVE/AT/BELOW)")


class BundleScoringResult(BaseModel):
    """Complete result from the BundleComplianceScoringEngine."""

    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Weighted overall score 0-100")
    per_regulation: List[ComplianceScore] = Field(
        default_factory=list, description="Per-regulation compliance scores"
    )
    maturity_profile: List[MaturityAssessment] = Field(
        default_factory=list, description="Maturity assessments per regulation"
    )
    heatmap: List[HeatmapCell] = Field(
        default_factory=list, description="Heatmap data for dashboard visualisation"
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    benchmarks: List[BenchmarkComparison] = Field(
        default_factory=list, description="Benchmark comparisons"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time in ms")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BundleComplianceScoringEngine:
    """
    Weighted compliance scoring engine spanning all 4 EU regulations.

    Scoring is entirely deterministic:
      - raw_score from item counts
      - risk adjustment from risk flag count and multiplier
      - deadline boost from calendar proximity
      - maturity from threshold comparison

    Attributes:
        config: Engine configuration.

    Example:
        >>> config = ScoringConfig(industry="manufacturing")
        >>> engine = BundleComplianceScoringEngine(config)
        >>> inputs = [RegulationInput(regulation="CSRD", items_assessed=100, items_compliant=80)]
        >>> result = engine.calculate_scores(inputs)
        >>> assert result.overall_score > 0
    """

    def __init__(self, config: Optional[ScoringConfig] = None) -> None:
        """Initialize the BundleComplianceScoringEngine.

        Args:
            config: Engine configuration. Uses defaults when *None*.
        """
        self.config = config or ScoringConfig()
        # Apply industry-specific weights if available
        if self.config.industry in DEFAULT_WEIGHTS:
            self.config.regulation_weights = dict(DEFAULT_WEIGHTS[self.config.industry])
        logger.info("BundleComplianceScoringEngine v%s initialised (industry=%s)", _MODULE_VERSION, self.config.industry)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_scores(self, inputs: List[RegulationInput]) -> BundleScoringResult:
        """Calculate compliance scores for all regulations.

        Args:
            inputs: Per-regulation input data.

        Returns:
            BundleScoringResult with overall score, per-regulation scores,
            maturity profile, heatmap, recommendations, and benchmarks.
        """
        start = _utcnow()

        per_regulation: List[ComplianceScore] = []
        maturity_profile: List[MaturityAssessment] = []

        for inp in inputs:
            raw = self._compute_raw_score(inp)
            risk_adj = self.calculate_risk_adjusted_score(raw, inp.risk_flags)
            deadline_adj = self._apply_deadline_boost(risk_adj, inp.near_deadline)
            weight = self.config.regulation_weights.get(inp.regulation, 0.0)
            maturity = self.assess_maturity(inp)

            score = ComplianceScore(
                regulation=inp.regulation,
                raw_score=round(raw, 2),
                weighted_score=round(deadline_adj * weight, 4),
                risk_adjusted_score=round(risk_adj, 2),
                maturity_level=maturity.level,
                risk_flags=inp.risk_flags,
                near_deadline=inp.near_deadline,
                weight=weight,
            )
            score.provenance_hash = _compute_hash(score)
            per_regulation.append(score)
            maturity_profile.append(maturity)

        overall = self._compute_overall(per_regulation)
        heatmap = self.generate_heatmap_data(inputs)
        recommendations = self.get_improvement_recommendations(inputs, per_regulation)
        benchmarks = self.compare_to_benchmark(per_regulation)

        elapsed_ms = (_utcnow() - start).total_seconds() * 1000

        result = BundleScoringResult(
            overall_score=round(overall, 2),
            per_regulation=per_regulation,
            maturity_profile=maturity_profile,
            heatmap=heatmap,
            recommendations=recommendations,
            benchmarks=benchmarks,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        logger.info("Scoring complete: overall=%.2f across %d regulations", overall, len(inputs))
        return result

    def apply_weights(
        self, scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply regulation weights to raw scores.

        Args:
            scores: Dict mapping regulation name to raw score.

        Returns:
            Dict mapping regulation name to weighted score.
        """
        weighted: Dict[str, float] = {}
        for reg, raw in scores.items():
            w = self.config.regulation_weights.get(reg, 0.0)
            weighted[reg] = round(raw * w, 4)
        return weighted

    def assess_maturity(self, inp: RegulationInput) -> MaturityAssessment:
        """Assess the maturity level for a single regulation.

        Maturity is derived from compliance percentage + evidence count:
          Level 1: score < 20 or evidence < 5
          Level 2: score < 40 or evidence < 15
          Level 3: score < 60 or evidence < 30
          Level 4: score < 80 or evidence < 50
          Level 5: score >= 80 and evidence >= 50

        Args:
            inp: Regulation input data.

        Returns:
            MaturityAssessment for the regulation.
        """
        raw = self._compute_raw_score(inp)
        evidence = inp.evidence_count
        gaps = len(inp.gaps)

        level = self._determine_maturity_level(raw, evidence)

        reg_defs = MATURITY_DEFINITIONS.get(inp.regulation, {})
        level_def = reg_defs.get(level, {"name": "Unknown", "description": ""})

        next_reqs = self._next_level_requirements(inp.regulation, level, raw, evidence, gaps)

        assessment = MaturityAssessment(
            regulation=inp.regulation,
            level=level,
            level_name=level_def["name"],
            description=level_def["description"],
            evidence_count=evidence,
            gaps_remaining=gaps,
            next_level_requirements=next_reqs,
        )
        assessment.provenance_hash = _compute_hash(assessment)
        return assessment

    def calculate_risk_adjusted_score(
        self, raw_score: float, risk_flags: List[str]
    ) -> float:
        """Apply risk penalties to a raw score.

        Each risk flag incurs a 3% penalty, multiplied by risk_multiplier.
        The score cannot go below 0.

        Args:
            raw_score: Unadjusted compliance score 0-100.
            risk_flags: List of risk flag descriptions.

        Returns:
            Risk-adjusted score.
        """
        if not risk_flags:
            return raw_score

        penalty_per_flag = 3.0 * self.config.risk_multiplier
        total_penalty = len(risk_flags) * penalty_per_flag
        adjusted = raw_score - total_penalty
        return _clamp(adjusted, 0.0, 100.0)

    def generate_heatmap_data(
        self, inputs: List[RegulationInput]
    ) -> List[HeatmapCell]:
        """Generate heatmap data for dashboard visualisation.

        Creates a matrix of regulation x dimension with colour-coded status.

        Args:
            inputs: Per-regulation input data.

        Returns:
            List of HeatmapCell objects.
        """
        dimensions = [
            "Compliance Level",
            "Data Quality",
            "Evidence Coverage",
            "Gap Closure",
            "Risk Exposure",
        ]

        cells: List[HeatmapCell] = []
        for inp in inputs:
            raw = self._compute_raw_score(inp)

            # Compliance Level
            cells.append(HeatmapCell(
                regulation=inp.regulation,
                dimension="Compliance Level",
                score=round(raw, 2),
                status=self._score_to_status(raw),
            ))

            # Data Quality
            cells.append(HeatmapCell(
                regulation=inp.regulation,
                dimension="Data Quality",
                score=round(inp.data_quality_score, 2),
                status=self._score_to_status(inp.data_quality_score),
            ))

            # Evidence Coverage
            evidence_score = min(inp.evidence_count / 50.0 * 100.0, 100.0)
            cells.append(HeatmapCell(
                regulation=inp.regulation,
                dimension="Evidence Coverage",
                score=round(evidence_score, 2),
                status=self._score_to_status(evidence_score),
            ))

            # Gap Closure
            total_gaps = len(inp.gaps)
            gap_closure = _clamp(100.0 - (total_gaps * 5.0), 0.0, 100.0)
            cells.append(HeatmapCell(
                regulation=inp.regulation,
                dimension="Gap Closure",
                score=round(gap_closure, 2),
                status=self._score_to_status(gap_closure),
            ))

            # Risk Exposure (inverted: fewer flags = higher score)
            risk_score = _clamp(100.0 - (len(inp.risk_flags) * 10.0), 0.0, 100.0)
            cells.append(HeatmapCell(
                regulation=inp.regulation,
                dimension="Risk Exposure",
                score=round(risk_score, 2),
                status=self._score_to_status(risk_score),
            ))

        return cells

    def get_improvement_recommendations(
        self,
        inputs: List[RegulationInput],
        scores: List[ComplianceScore],
    ) -> List[Recommendation]:
        """Generate improvement recommendations based on gaps and scores.

        Args:
            inputs: Per-regulation input data.
            scores: Computed compliance scores.

        Returns:
            List of prioritised Recommendation objects.
        """
        recommendations: List[Recommendation] = []
        score_map = {s.regulation: s for s in scores}

        for inp in inputs:
            cs = score_map.get(inp.regulation)
            if cs is None:
                continue

            # Low overall score
            if cs.raw_score < self.config.risk_flags_threshold:
                recommendations.append(Recommendation(
                    regulation=inp.regulation,
                    priority="HIGH" if cs.raw_score < 40 else "MEDIUM",
                    title=f"Improve {inp.regulation} compliance score",
                    description=(
                        f"Current score is {cs.raw_score:.1f}%, which is below the "
                        f"threshold of {self.config.risk_flags_threshold:.0f}%. "
                        f"Address {inp.items_non_compliant} non-compliant items."
                    ),
                    effort_estimate="HIGH" if inp.items_non_compliant > 20 else "MEDIUM",
                    impact_estimate="HIGH",
                ))

            # Pending items
            if inp.items_pending > 0:
                recommendations.append(Recommendation(
                    regulation=inp.regulation,
                    priority="MEDIUM",
                    title=f"Complete pending {inp.regulation} assessments",
                    description=f"There are {inp.items_pending} items still pending assessment.",
                    effort_estimate="LOW" if inp.items_pending < 10 else "MEDIUM",
                    impact_estimate="MEDIUM",
                ))

            # Insufficient evidence
            if inp.evidence_count < 20:
                recommendations.append(Recommendation(
                    regulation=inp.regulation,
                    priority="MEDIUM",
                    title=f"Strengthen {inp.regulation} evidence base",
                    description=(
                        f"Only {inp.evidence_count} evidence artefacts collected. "
                        f"Aim for at least 50 to reach maturity level 5."
                    ),
                    effort_estimate="MEDIUM",
                    impact_estimate="MEDIUM",
                ))

            # Data quality
            if inp.data_quality_score < 70.0:
                recommendations.append(Recommendation(
                    regulation=inp.regulation,
                    priority="HIGH" if inp.data_quality_score < 50.0 else "MEDIUM",
                    title=f"Improve {inp.regulation} data quality",
                    description=(
                        f"Data quality score is {inp.data_quality_score:.1f}%. "
                        f"Investigate data gaps and implement validation controls."
                    ),
                    effort_estimate="MEDIUM",
                    impact_estimate="HIGH",
                ))

            # Near deadline with low score
            if inp.near_deadline and cs.raw_score < 75.0:
                recommendations.append(Recommendation(
                    regulation=inp.regulation,
                    priority="CRITICAL",
                    title=f"Urgent: {inp.regulation} deadline approaching",
                    description=(
                        f"A compliance deadline is approaching and the score is only "
                        f"{cs.raw_score:.1f}%. Accelerate remediation efforts."
                    ),
                    effort_estimate="HIGH",
                    impact_estimate="CRITICAL",
                ))

            # Specific gaps
            for gap in inp.gaps[:3]:
                recommendations.append(Recommendation(
                    regulation=inp.regulation,
                    priority="MEDIUM",
                    title=f"Close gap: {gap}",
                    description=f"Address the identified gap in {inp.regulation}: {gap}.",
                    effort_estimate="MEDIUM",
                    impact_estimate="MEDIUM",
                ))

        # Sort by priority
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 9))

        return recommendations

    def compare_to_benchmark(
        self, scores: List[ComplianceScore], industry: Optional[str] = None
    ) -> List[BenchmarkComparison]:
        """Compare scores against industry benchmarks.

        Args:
            scores: Computed compliance scores.
            industry: Industry sector. Falls back to config.industry.

        Returns:
            List of BenchmarkComparison objects.
        """
        industry = industry or self.config.industry
        benchmarks = INDUSTRY_BENCHMARKS.get(industry, INDUSTRY_BENCHMARKS["default"])

        comparisons: List[BenchmarkComparison] = []
        for cs in scores:
            bench = benchmarks.get(cs.regulation, 50.0)
            delta = round(cs.raw_score - bench, 2)
            if delta > 5.0:
                position = "ABOVE"
            elif delta < -5.0:
                position = "BELOW"
            else:
                position = "AT"

            comparisons.append(BenchmarkComparison(
                regulation=cs.regulation,
                company_score=cs.raw_score,
                benchmark_score=bench,
                delta=delta,
                position=position,
            ))
        return comparisons

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_raw_score(inp: RegulationInput) -> float:
        """Compute raw compliance score from item counts."""
        if inp.items_assessed == 0:
            return 0.0
        return _clamp(
            (inp.items_compliant / inp.items_assessed) * 100.0,
            0.0, 100.0,
        )

    def _apply_deadline_boost(self, score: float, near_deadline: bool) -> float:
        """Apply deadline-proximity score boost."""
        if not near_deadline:
            return score
        boosted = score * (1.0 + self.config.near_deadline_boost)
        return _clamp(boosted, 0.0, 100.0)

    def _compute_overall(self, per_regulation: List[ComplianceScore]) -> float:
        """Compute the overall weighted bundle score."""
        total_weight = sum(cs.weight for cs in per_regulation)
        if total_weight == 0.0:
            return 0.0
        weighted_sum = sum(cs.weighted_score for cs in per_regulation)
        return _clamp(weighted_sum / total_weight, 0.0, 100.0)

    @staticmethod
    def _determine_maturity_level(raw_score: float, evidence_count: int) -> int:
        """Determine maturity level from score and evidence count."""
        if raw_score >= 80.0 and evidence_count >= 50:
            return 5
        elif raw_score >= 60.0 and evidence_count >= 30:
            return 4
        elif raw_score >= 40.0 and evidence_count >= 15:
            return 3
        elif raw_score >= 20.0 and evidence_count >= 5:
            return 2
        return 1

    @staticmethod
    def _next_level_requirements(
        regulation: str, current_level: int, score: float,
        evidence: int, gaps: int
    ) -> List[str]:
        """Generate requirements to reach the next maturity level."""
        if current_level >= 5:
            return ["Already at highest maturity level. Focus on continuous improvement."]

        reqs: List[str] = []
        target_level = current_level + 1
        thresholds = {2: (20, 5), 3: (40, 15), 4: (60, 30), 5: (80, 50)}
        target_score, target_evidence = thresholds.get(target_level, (100, 100))

        if score < target_score:
            reqs.append(f"Increase compliance score from {score:.1f}% to at least {target_score}%.")

        if evidence < target_evidence:
            reqs.append(f"Collect {target_evidence - evidence} more evidence artefacts (current: {evidence}).")

        if gaps > 0:
            reqs.append(f"Close {min(gaps, 5)} of the {gaps} remaining gap(s).")

        if not reqs:
            reqs.append(f"Meet all criteria for {regulation} maturity level {target_level}.")

        return reqs

    @staticmethod
    def _score_to_status(score: float) -> str:
        """Map a score to a heatmap colour status."""
        if score >= 80.0:
            return HeatmapStatus.GREEN.value
        elif score >= 60.0:
            return HeatmapStatus.YELLOW.value
        elif score >= 40.0:
            return HeatmapStatus.ORANGE.value
        return HeatmapStatus.RED.value
