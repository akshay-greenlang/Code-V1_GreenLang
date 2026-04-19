# -*- coding: utf-8 -*-
"""
ConsolidatedMetricsEngine - PACK-009 EU Climate Compliance Bundle Engine 5

Aggregates KPIs from all 4 constituent regulations (CSRD, CBAM, EU Taxonomy,
EUDR) into a unified metrics dashboard. Computes a weighted bundle compliance
score, tracks trend data across periods, and produces executive summaries with
per-regulation breakdowns.

Capabilities:
    1. Aggregate metrics across 4 EU regulations
    2. Compute weighted bundle compliance score
    3. Calculate data completeness per regulation
    4. Analyze multi-period trends
    5. Produce per-regulation breakdowns
    6. Generate executive summaries
    7. Compare metrics across reporting periods

Zero-Hallucination:
    - All scores computed via deterministic weighted-average formulae
    - Trend analysis uses simple delta / percentage-change arithmetic
    - No LLM involvement in any numeric path
    - SHA-256 provenance hash on all result objects

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def _pct_change(current: float, previous: float) -> float:
    """Calculate percentage change between two values."""
    if previous == 0.0:
        return 0.0 if current == 0.0 else 100.0
    return round(((current - previous) / abs(previous)) * 100.0, 2)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RegulationType(str, Enum):
    """Supported EU regulations in the bundle."""
    CSRD = "CSRD"
    CBAM = "CBAM"
    EU_TAXONOMY = "EU_TAXONOMY"
    EUDR = "EUDR"

class TrendDirection(str, Enum):
    """Direction of a metric trend."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"

class SummaryRating(str, Enum):
    """High-level compliance rating for executive summaries."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ADEQUATE = "ADEQUATE"
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"
    CRITICAL = "CRITICAL"

# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

REGULATION_METRIC_DEFINITIONS: Dict[str, List[Dict[str, str]]] = {
    "CSRD": [
        {"key": "esrs_disclosure_completion", "label": "ESRS Disclosure Completion", "unit": "%"},
        {"key": "double_materiality_coverage", "label": "Double Materiality Coverage", "unit": "%"},
        {"key": "scope1_emissions_reported", "label": "Scope 1 Emissions Reported", "unit": "tCO2e"},
        {"key": "scope2_emissions_reported", "label": "Scope 2 Emissions Reported", "unit": "tCO2e"},
        {"key": "scope3_emissions_reported", "label": "Scope 3 Emissions Reported", "unit": "tCO2e"},
        {"key": "governance_disclosures", "label": "Governance Disclosures", "unit": "count"},
        {"key": "strategy_disclosures", "label": "Strategy Disclosures", "unit": "count"},
        {"key": "irm_disclosures", "label": "Impact/Risk/Opportunity Disclosures", "unit": "count"},
        {"key": "metrics_targets_count", "label": "Metrics & Targets Defined", "unit": "count"},
        {"key": "transition_plan_completeness", "label": "Transition Plan Completeness", "unit": "%"},
        {"key": "value_chain_coverage", "label": "Value Chain Data Coverage", "unit": "%"},
        {"key": "data_quality_score", "label": "Data Quality Score", "unit": "score"},
        {"key": "assurance_readiness", "label": "Assurance Readiness Level", "unit": "%"},
        {"key": "taxonomy_alignment_reported", "label": "Taxonomy Alignment Reported", "unit": "bool"},
        {"key": "digital_tagging_progress", "label": "XBRL/Digital Tagging Progress", "unit": "%"},
    ],
    "CBAM": [
        {"key": "declarations_submitted", "label": "Quarterly Declarations Submitted", "unit": "count"},
        {"key": "goods_categories_covered", "label": "Goods Categories Covered", "unit": "count"},
        {"key": "embedded_emissions_total", "label": "Total Embedded Emissions", "unit": "tCO2e"},
        {"key": "supplier_data_coverage", "label": "Supplier-Specific Data Coverage", "unit": "%"},
        {"key": "default_value_reliance", "label": "Default Value Reliance", "unit": "%"},
        {"key": "certificate_sufficiency", "label": "Certificate Sufficiency", "unit": "%"},
        {"key": "carbon_price_paid_abroad", "label": "Carbon Price Paid Abroad", "unit": "EUR"},
        {"key": "financial_exposure", "label": "Financial Exposure", "unit": "EUR"},
        {"key": "verification_status", "label": "Verification Status", "unit": "count"},
        {"key": "precursor_chain_depth", "label": "Precursor Chain Depth", "unit": "levels"},
        {"key": "cn_code_accuracy", "label": "CN Code Accuracy", "unit": "%"},
        {"key": "reporting_timeliness", "label": "Reporting Timeliness", "unit": "days"},
        {"key": "cost_optimization_savings", "label": "Cost Optimization Savings", "unit": "EUR"},
        {"key": "de_minimis_utilization", "label": "De Minimis Utilization", "unit": "%"},
        {"key": "nca_readiness", "label": "NCA Examination Readiness", "unit": "%"},
    ],
    "EU_TAXONOMY": [
        {"key": "eligible_turnover_pct", "label": "Eligible Turnover %", "unit": "%"},
        {"key": "aligned_turnover_pct", "label": "Aligned Turnover %", "unit": "%"},
        {"key": "eligible_capex_pct", "label": "Eligible CapEx %", "unit": "%"},
        {"key": "aligned_capex_pct", "label": "Aligned CapEx %", "unit": "%"},
        {"key": "eligible_opex_pct", "label": "Eligible OpEx %", "unit": "%"},
        {"key": "aligned_opex_pct", "label": "Aligned OpEx %", "unit": "%"},
        {"key": "activities_assessed", "label": "Economic Activities Assessed", "unit": "count"},
        {"key": "substantial_contribution_met", "label": "Substantial Contribution Criteria Met", "unit": "count"},
        {"key": "dnsh_criteria_assessed", "label": "DNSH Criteria Assessed", "unit": "count"},
        {"key": "minimum_safeguards_status", "label": "Minimum Safeguards Status", "unit": "bool"},
        {"key": "climate_mitigation_aligned", "label": "Climate Mitigation Aligned", "unit": "count"},
        {"key": "climate_adaptation_aligned", "label": "Climate Adaptation Aligned", "unit": "count"},
        {"key": "water_aligned", "label": "Water & Marine Resources Aligned", "unit": "count"},
        {"key": "circular_economy_aligned", "label": "Circular Economy Aligned", "unit": "count"},
        {"key": "biodiversity_aligned", "label": "Biodiversity Aligned", "unit": "count"},
    ],
    "EUDR": [
        {"key": "commodities_covered", "label": "Commodities Covered", "unit": "count"},
        {"key": "supply_chains_mapped", "label": "Supply Chains Mapped", "unit": "count"},
        {"key": "geolocation_coverage", "label": "Geolocation Data Coverage", "unit": "%"},
        {"key": "risk_assessments_complete", "label": "Risk Assessments Complete", "unit": "count"},
        {"key": "due_diligence_statements", "label": "Due Diligence Statements Filed", "unit": "count"},
        {"key": "deforestation_free_pct", "label": "Deforestation-Free Verified %", "unit": "%"},
        {"key": "satellite_monitoring_coverage", "label": "Satellite Monitoring Coverage", "unit": "%"},
        {"key": "supplier_certifications", "label": "Supplier Certifications Verified", "unit": "count"},
        {"key": "traceability_depth", "label": "Traceability Chain Depth", "unit": "levels"},
        {"key": "country_benchmarking_applied", "label": "Country Benchmarking Applied", "unit": "count"},
        {"key": "mitigation_measures_active", "label": "Mitigation Measures Active", "unit": "count"},
        {"key": "information_system_entries", "label": "Information System Entries", "unit": "count"},
        {"key": "legacy_compliance_pct", "label": "Legacy Product Compliance %", "unit": "%"},
        {"key": "audit_trail_completeness", "label": "Audit Trail Completeness", "unit": "%"},
        {"key": "nca_inspection_readiness", "label": "NCA Inspection Readiness", "unit": "%"},
    ],
}

DEFAULT_REGULATION_WEIGHTS: Dict[str, float] = {
    "CSRD": 0.30,
    "CBAM": 0.25,
    "EU_TAXONOMY": 0.25,
    "EUDR": 0.20,
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConsolidatedMetricsConfig(BaseModel):
    """Configuration for the ConsolidatedMetricsEngine."""

    include_trends: bool = Field(default=True, description="Include trend analysis in results")
    trend_periods: int = Field(default=4, ge=2, le=24, description="Number of periods for trend analysis")
    completeness_threshold: float = Field(
        default=80.0, ge=0.0, le=100.0,
        description="Minimum data completeness percentage to consider a regulation reportable",
    )
    regulation_weights: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_REGULATION_WEIGHTS),
        description="Weight per regulation for bundle score (must sum to 1.0)",
    )
    improvement_target_pct: float = Field(
        default=5.0, ge=0.0, le=50.0,
        description="Target improvement percentage per period",
    )

    @field_validator("regulation_weights")
    @classmethod
    def _validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Regulation weights must sum to 1.0, got {total:.4f}")
        return v

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class RegulationMetrics(BaseModel):
    """Aggregated KPIs for a single regulation."""

    regulation: str = Field(..., description="Regulation identifier (CSRD, CBAM, EU_TAXONOMY, EUDR)")
    compliance_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall compliance score 0-100")
    data_completeness: float = Field(default=0.0, ge=0.0, le=100.0, description="Data completeness 0-100")
    items_assessed: int = Field(default=0, ge=0, description="Total items assessed")
    items_compliant: int = Field(default=0, ge=0, description="Items that are compliant")
    items_non_compliant: int = Field(default=0, ge=0, description="Items that are non-compliant")
    items_pending: int = Field(default=0, ge=0, description="Items pending assessment")
    key_metrics: Dict[str, Any] = Field(default_factory=dict, description="Regulation-specific KPI values")
    assessment_date: str = Field(default="", description="ISO-8601 date of assessment")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class TrendDataPoint(BaseModel):
    """A single data point in a trend series."""

    period: str = Field(..., description="Period label (e.g. 2026-Q1)")
    regulation: str = Field(..., description="Regulation identifier")
    metric_name: str = Field(..., description="Metric key name")
    value: float = Field(default=0.0, description="Metric value for this period")
    previous_value: float = Field(default=0.0, description="Value in the previous period")
    change_pct: float = Field(default=0.0, description="Percentage change from previous period")
    direction: str = Field(default="STABLE", description="Trend direction")

class PeriodSnapshot(BaseModel):
    """Metrics snapshot for a single reporting period."""

    period: str = Field(..., description="Period label")
    bundle_score: float = Field(default=0.0, description="Weighted bundle score for this period")
    per_regulation: Dict[str, float] = Field(
        default_factory=dict, description="Per-regulation compliance scores"
    )
    data_completeness: Dict[str, float] = Field(
        default_factory=dict, description="Per-regulation data completeness"
    )
    timestamp: str = Field(default="", description="ISO-8601 timestamp of snapshot")

class ExecutiveSummary(BaseModel):
    """Executive summary of consolidated metrics."""

    summary_id: str = Field(default_factory=_new_uuid, description="Summary identifier")
    bundle_score: float = Field(default=0.0, description="Weighted bundle compliance score 0-100")
    rating: str = Field(default="", description="Overall compliance rating")
    regulations_assessed: int = Field(default=0, description="Number of regulations assessed")
    strongest_regulation: str = Field(default="", description="Best-performing regulation")
    weakest_regulation: str = Field(default="", description="Regulation needing most attention")
    overall_completeness: float = Field(default=0.0, description="Average data completeness across all regulations")
    key_findings: List[str] = Field(default_factory=list, description="Key findings and observations")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    generated_at: str = Field(default="", description="ISO-8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class PeriodComparison(BaseModel):
    """Comparison of metrics between two reporting periods."""

    period_a: str = Field(..., description="First period label")
    period_b: str = Field(..., description="Second period label")
    bundle_score_a: float = Field(default=0.0, description="Bundle score for period A")
    bundle_score_b: float = Field(default=0.0, description="Bundle score for period B")
    bundle_score_change_pct: float = Field(default=0.0, description="Percentage change in bundle score")
    regulation_changes: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Per-regulation score changes"
    )
    improved_regulations: List[str] = Field(default_factory=list, description="Regulations that improved")
    declined_regulations: List[str] = Field(default_factory=list, description="Regulations that declined")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ConsolidatedMetricsResult(BaseModel):
    """Complete result from the ConsolidatedMetricsEngine."""

    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    bundle_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Weighted bundle score 0-100")
    per_regulation: List[RegulationMetrics] = Field(
        default_factory=list, description="Per-regulation metrics"
    )
    trends: List[TrendDataPoint] = Field(
        default_factory=list, description="Trend data points across periods"
    )
    completeness: Dict[str, float] = Field(
        default_factory=dict, description="Per-regulation data completeness"
    )
    executive_summary: Optional[ExecutiveSummary] = Field(
        default=None, description="Executive summary"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash of full result")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ConsolidatedMetricsEngine:
    """
    Aggregates KPIs from all 4 constituent EU regulations into a unified
    metrics view with weighted bundle scoring, trend analysis, and executive
    summaries.

    The engine is entirely deterministic: scores are calculated using weighted
    averages, completeness is derived from field-presence counts, and trends
    use simple arithmetic delta / pct-change.

    Attributes:
        config: Engine configuration.
        _history: Internal list of period snapshots for trend analysis.

    Example:
        >>> config = ConsolidatedMetricsConfig()
        >>> engine = ConsolidatedMetricsEngine(config)
        >>> metrics = [RegulationMetrics(regulation="CSRD", compliance_score=85, ...)]
        >>> result = engine.aggregate_metrics(metrics)
        >>> assert 0 <= result.bundle_score <= 100
    """

    def __init__(self, config: Optional[ConsolidatedMetricsConfig] = None) -> None:
        """Initialize the ConsolidatedMetricsEngine.

        Args:
            config: Engine configuration. Uses defaults when *None*.
        """
        self.config = config or ConsolidatedMetricsConfig()
        self._history: List[PeriodSnapshot] = []
        logger.info("ConsolidatedMetricsEngine v%s initialised", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate_metrics(
        self,
        regulation_metrics: List[RegulationMetrics],
        period: Optional[str] = None,
    ) -> ConsolidatedMetricsResult:
        """Aggregate per-regulation metrics into a consolidated result.

        Args:
            regulation_metrics: List of RegulationMetrics, one per regulation.
            period: Optional period label (e.g. '2026-Q1'). Defaults to current date.

        Returns:
            ConsolidatedMetricsResult with bundle score, breakdowns, trends, and summary.
        """
        start = utcnow()
        if not period:
            now = utcnow()
            quarter = ((now.month - 1) // 3) + 1
            period = f"{now.year}-Q{quarter}"

        # Ensure provenance hashes are populated on inputs
        for rm in regulation_metrics:
            if not rm.provenance_hash:
                rm.provenance_hash = _compute_hash(rm)

        # Compute bundle score
        bundle_score = self.compute_bundle_score(regulation_metrics)

        # Compute completeness
        completeness = self.calculate_completeness(regulation_metrics)

        # Store snapshot for trend analysis
        snapshot = PeriodSnapshot(
            period=period,
            bundle_score=bundle_score,
            per_regulation={rm.regulation: rm.compliance_score for rm in regulation_metrics},
            data_completeness=completeness,
            timestamp=start.isoformat(),
        )
        self._history.append(snapshot)

        # Trends
        trends: List[TrendDataPoint] = []
        if self.config.include_trends and len(self._history) >= 2:
            trends = self.analyze_trends()

        # Executive summary
        summary = self.get_executive_summary(regulation_metrics, bundle_score, completeness)

        elapsed_ms = (utcnow() - start).total_seconds() * 1000

        result = ConsolidatedMetricsResult(
            bundle_score=bundle_score,
            per_regulation=regulation_metrics,
            trends=trends,
            completeness=completeness,
            executive_summary=summary,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        logger.info(
            "Aggregated metrics: bundle_score=%.2f, regulations=%d",
            bundle_score, len(regulation_metrics),
        )
        return result

    def compute_bundle_score(self, regulation_metrics: List[RegulationMetrics]) -> float:
        """Compute a weighted bundle compliance score.

        The formula is:
            bundle_score = sum(weight_i * score_i) for each regulation i

        Regulations not present in *regulation_metrics* contribute 0.

        Args:
            regulation_metrics: Per-regulation metrics.

        Returns:
            Weighted bundle score in [0, 100].
        """
        weights = self.config.regulation_weights
        total_weight = 0.0
        weighted_sum = 0.0

        for rm in regulation_metrics:
            w = weights.get(rm.regulation, 0.0)
            weighted_sum += w * rm.compliance_score
            total_weight += w

        if total_weight == 0.0:
            return 0.0

        # Re-normalise in case not all regulations are present
        score = weighted_sum / total_weight
        return round(min(max(score, 0.0), 100.0), 2)

    def calculate_completeness(
        self, regulation_metrics: List[RegulationMetrics]
    ) -> Dict[str, float]:
        """Calculate data completeness per regulation.

        Completeness is derived from the number of non-empty key_metrics fields
        relative to the expected KPIs defined in REGULATION_METRIC_DEFINITIONS.

        Args:
            regulation_metrics: Per-regulation metrics.

        Returns:
            Dict mapping regulation name to completeness percentage.
        """
        result: Dict[str, float] = {}
        for rm in regulation_metrics:
            expected = REGULATION_METRIC_DEFINITIONS.get(rm.regulation, [])
            if not expected:
                result[rm.regulation] = rm.data_completeness
                continue
            expected_keys = {d["key"] for d in expected}
            present = 0
            for key in expected_keys:
                val = rm.key_metrics.get(key)
                if val is not None and val != "" and val != 0:
                    present += 1
            pct = _safe_div(present, len(expected_keys)) * 100.0
            # Use the higher of computed and provided completeness
            result[rm.regulation] = round(max(pct, rm.data_completeness), 2)
        return result

    def analyze_trends(self) -> List[TrendDataPoint]:
        """Analyze trends across stored period snapshots.

        Returns the most recent *trend_periods* worth of data points, with
        period-over-period change percentages.

        Returns:
            List of TrendDataPoint objects.
        """
        snapshots = self._history[-self.config.trend_periods:]
        if len(snapshots) < 2:
            return []

        trends: List[TrendDataPoint] = []
        for idx in range(1, len(snapshots)):
            prev = snapshots[idx - 1]
            curr = snapshots[idx]

            # Bundle-level trend
            change = _pct_change(curr.bundle_score, prev.bundle_score)
            direction = self._classify_direction(change)
            trends.append(TrendDataPoint(
                period=curr.period,
                regulation="BUNDLE",
                metric_name="bundle_score",
                value=curr.bundle_score,
                previous_value=prev.bundle_score,
                change_pct=change,
                direction=direction,
            ))

            # Per-regulation trends
            all_regs = set(curr.per_regulation.keys()) | set(prev.per_regulation.keys())
            for reg in sorted(all_regs):
                cur_score = curr.per_regulation.get(reg, 0.0)
                prev_score = prev.per_regulation.get(reg, 0.0)
                reg_change = _pct_change(cur_score, prev_score)
                trends.append(TrendDataPoint(
                    period=curr.period,
                    regulation=reg,
                    metric_name="compliance_score",
                    value=cur_score,
                    previous_value=prev_score,
                    change_pct=reg_change,
                    direction=self._classify_direction(reg_change),
                ))

            # Completeness trends
            for reg in sorted(all_regs):
                cur_c = curr.data_completeness.get(reg, 0.0)
                prev_c = prev.data_completeness.get(reg, 0.0)
                c_change = _pct_change(cur_c, prev_c)
                trends.append(TrendDataPoint(
                    period=curr.period,
                    regulation=reg,
                    metric_name="data_completeness",
                    value=cur_c,
                    previous_value=prev_c,
                    change_pct=c_change,
                    direction=self._classify_direction(c_change),
                ))

        return trends

    def get_per_regulation_breakdown(
        self, regulation_metrics: List[RegulationMetrics]
    ) -> Dict[str, Dict[str, Any]]:
        """Get detailed breakdown per regulation.

        Args:
            regulation_metrics: Per-regulation metrics.

        Returns:
            Dict keyed by regulation name containing score, completeness,
            item counts, and KPI definitions with values.
        """
        breakdown: Dict[str, Dict[str, Any]] = {}
        for rm in regulation_metrics:
            kpi_defs = REGULATION_METRIC_DEFINITIONS.get(rm.regulation, [])
            kpi_detail: List[Dict[str, Any]] = []
            for kpi in kpi_defs:
                kpi_detail.append({
                    "key": kpi["key"],
                    "label": kpi["label"],
                    "unit": kpi["unit"],
                    "value": rm.key_metrics.get(kpi["key"]),
                    "has_value": rm.key_metrics.get(kpi["key"]) is not None,
                })
            breakdown[rm.regulation] = {
                "compliance_score": rm.compliance_score,
                "data_completeness": rm.data_completeness,
                "items_assessed": rm.items_assessed,
                "items_compliant": rm.items_compliant,
                "items_non_compliant": rm.items_non_compliant,
                "items_pending": rm.items_pending,
                "kpi_count": len(kpi_defs),
                "kpis": kpi_detail,
                "assessment_date": rm.assessment_date,
                "weight": self.config.regulation_weights.get(rm.regulation, 0.0),
            }
        return breakdown

    def get_executive_summary(
        self,
        regulation_metrics: List[RegulationMetrics],
        bundle_score: float,
        completeness: Dict[str, float],
    ) -> ExecutiveSummary:
        """Generate an executive summary of the consolidated metrics.

        Args:
            regulation_metrics: Per-regulation metrics.
            bundle_score: Pre-computed weighted bundle score.
            completeness: Per-regulation completeness dict.

        Returns:
            ExecutiveSummary with rating, findings, and recommendations.
        """
        if not regulation_metrics:
            return ExecutiveSummary(
                bundle_score=0.0,
                rating=SummaryRating.CRITICAL.value,
                regulations_assessed=0,
                generated_at=utcnow().isoformat(),
            )

        sorted_by_score = sorted(regulation_metrics, key=lambda r: r.compliance_score, reverse=True)
        strongest = sorted_by_score[0].regulation
        weakest = sorted_by_score[-1].regulation

        avg_completeness = _safe_div(
            sum(completeness.values()), len(completeness)
        )

        rating = self._determine_rating(bundle_score)

        findings = self._generate_findings(regulation_metrics, bundle_score, completeness)
        recommendations = self._generate_recommendations(
            regulation_metrics, bundle_score, completeness
        )

        summary = ExecutiveSummary(
            bundle_score=bundle_score,
            rating=rating,
            regulations_assessed=len(regulation_metrics),
            strongest_regulation=strongest,
            weakest_regulation=weakest,
            overall_completeness=round(avg_completeness, 2),
            key_findings=findings,
            recommendations=recommendations,
            generated_at=utcnow().isoformat(),
        )
        summary.provenance_hash = _compute_hash(summary)
        return summary

    def compare_periods(
        self,
        period_a: str,
        period_b: str,
    ) -> PeriodComparison:
        """Compare metrics between two stored periods.

        Args:
            period_a: First period label.
            period_b: Second period label.

        Returns:
            PeriodComparison with per-regulation deltas.

        Raises:
            ValueError: If either period is not found in history.
        """
        snap_a = self._find_snapshot(period_a)
        snap_b = self._find_snapshot(period_b)

        if snap_a is None:
            raise ValueError(f"Period '{period_a}' not found in history")
        if snap_b is None:
            raise ValueError(f"Period '{period_b}' not found in history")

        bundle_change = _pct_change(snap_b.bundle_score, snap_a.bundle_score)

        reg_changes: Dict[str, Dict[str, float]] = {}
        improved: List[str] = []
        declined: List[str] = []
        all_regs = set(snap_a.per_regulation.keys()) | set(snap_b.per_regulation.keys())
        for reg in sorted(all_regs):
            score_a = snap_a.per_regulation.get(reg, 0.0)
            score_b = snap_b.per_regulation.get(reg, 0.0)
            delta = round(score_b - score_a, 2)
            change = _pct_change(score_b, score_a)
            reg_changes[reg] = {
                "score_a": score_a,
                "score_b": score_b,
                "delta": delta,
                "change_pct": change,
            }
            if delta > 0.5:
                improved.append(reg)
            elif delta < -0.5:
                declined.append(reg)

        comparison = PeriodComparison(
            period_a=period_a,
            period_b=period_b,
            bundle_score_a=snap_a.bundle_score,
            bundle_score_b=snap_b.bundle_score,
            bundle_score_change_pct=bundle_change,
            regulation_changes=reg_changes,
            improved_regulations=improved,
            declined_regulations=declined,
        )
        comparison.provenance_hash = _compute_hash(comparison)
        return comparison

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_snapshot(self, period: str) -> Optional[PeriodSnapshot]:
        """Find a snapshot by period label."""
        for snap in self._history:
            if snap.period == period:
                return snap
        return None

    @staticmethod
    def _classify_direction(change_pct: float) -> str:
        """Classify a percentage change into a trend direction."""
        if change_pct > 1.0:
            return TrendDirection.IMPROVING.value
        elif change_pct < -1.0:
            return TrendDirection.DECLINING.value
        return TrendDirection.STABLE.value

    @staticmethod
    def _determine_rating(bundle_score: float) -> str:
        """Map a bundle score to an executive rating."""
        if bundle_score >= 90.0:
            return SummaryRating.EXCELLENT.value
        elif bundle_score >= 75.0:
            return SummaryRating.GOOD.value
        elif bundle_score >= 60.0:
            return SummaryRating.ADEQUATE.value
        elif bundle_score >= 40.0:
            return SummaryRating.NEEDS_IMPROVEMENT.value
        return SummaryRating.CRITICAL.value

    @staticmethod
    def _generate_findings(
        metrics: List[RegulationMetrics],
        bundle_score: float,
        completeness: Dict[str, float],
    ) -> List[str]:
        """Generate key findings from the metrics."""
        findings: List[str] = []

        findings.append(
            f"Bundle compliance score is {bundle_score:.1f}/100 across "
            f"{len(metrics)} EU regulations."
        )

        high_performers = [m for m in metrics if m.compliance_score >= 80.0]
        if high_performers:
            names = ", ".join(m.regulation for m in high_performers)
            findings.append(f"Strong performance in: {names}.")

        low_performers = [m for m in metrics if m.compliance_score < 60.0]
        if low_performers:
            names = ", ".join(m.regulation for m in low_performers)
            findings.append(f"Attention required for: {names}.")

        low_completeness = [
            reg for reg, pct in completeness.items() if pct < 50.0
        ]
        if low_completeness:
            findings.append(
                f"Data completeness below 50% for: {', '.join(low_completeness)}."
            )

        total_assessed = sum(m.items_assessed for m in metrics)
        total_compliant = sum(m.items_compliant for m in metrics)
        total_nc = sum(m.items_non_compliant for m in metrics)
        total_pending = sum(m.items_pending for m in metrics)
        findings.append(
            f"Total items: {total_assessed} assessed, {total_compliant} compliant, "
            f"{total_nc} non-compliant, {total_pending} pending."
        )

        return findings

    @staticmethod
    def _generate_recommendations(
        metrics: List[RegulationMetrics],
        bundle_score: float,
        completeness: Dict[str, float],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations: List[str] = []

        for m in metrics:
            if m.compliance_score < 60.0:
                recommendations.append(
                    f"Prioritise {m.regulation} compliance improvements "
                    f"(current score: {m.compliance_score:.1f})."
                )

        for reg, pct in completeness.items():
            if pct < 50.0:
                recommendations.append(
                    f"Improve data collection for {reg} "
                    f"(current completeness: {pct:.1f}%)."
                )

        for m in metrics:
            if m.items_pending > 0:
                recommendations.append(
                    f"Complete {m.items_pending} pending assessment(s) for {m.regulation}."
                )

        if bundle_score < 75.0:
            recommendations.append(
                "Consider engaging external advisors to accelerate compliance."
            )

        if not recommendations:
            recommendations.append(
                "Maintain current trajectory and focus on continuous improvement."
            )

        return recommendations
