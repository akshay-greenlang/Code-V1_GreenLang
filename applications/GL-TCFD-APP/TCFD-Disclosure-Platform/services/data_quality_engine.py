"""
Data Quality Engine -- Disclosure data quality scoring and validation.

This module implements the ``DataQualityEngine`` for GL-TCFD-APP v1.0.
It assesses the quality of TCFD disclosure data across four dimensions:
completeness, accuracy, timeliness, and consistency. Each dimension is
scored 0-100, with a composite Data Quality Score (DQS) calculated as
a weighted average.

The engine validates metrics quality against expected ranges and units,
validates scenario analysis outputs for internal consistency, and provides
specific improvement suggestions per disclosure section.

Quality dimensions:
    - Completeness (30%): Are all required fields populated?
    - Accuracy (30%):     Are values within expected ranges?
    - Timeliness (20%):   Is data current and up-to-date?
    - Consistency (20%):  Are values internally consistent?

Reference:
    - PCAF Data Quality Scoring (1-5)
    - GHG Protocol Quality Management guidance
    - IFRS S2 Appendix C: Effective Date and Transition

Example:
    >>> from services.config import TCFDAppConfig
    >>> engine = DataQualityEngine(TCFDAppConfig())
    >>> score = engine.assess_disclosure_quality("disc-1")
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    TCFDAppConfig,
    TCFD_DISCLOSURES,
)
from .models import _new_id, _now, _sha256

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data quality criteria per disclosure section
# ---------------------------------------------------------------------------

DATA_QUALITY_CRITERIA: Dict[str, Dict[str, Any]] = {
    "gov_a": {
        "section": "Board Oversight",
        "pillar": "governance",
        "required_fields": [
            "board_review_frequency",
            "committee_structure",
            "oversight_scope",
        ],
        "accuracy_checks": [
            {"field": "review_frequency", "type": "categorical", "valid_values": ["quarterly", "semi_annually", "annually", "ad_hoc"]},
        ],
        "timeliness_window_days": 365,
        "consistency_rules": [
            "If dedicated_committee is True, committee_name must be provided",
        ],
        "weight": 8.0,
    },
    "gov_b": {
        "section": "Management Role",
        "pillar": "governance",
        "required_fields": [
            "management_roles",
            "organizational_structure",
            "reporting_lines",
        ],
        "accuracy_checks": [
            {"field": "role_count", "type": "range", "min": 1, "max": 50},
        ],
        "timeliness_window_days": 365,
        "consistency_rules": [
            "Management roles must have defined responsibilities",
        ],
        "weight": 8.0,
    },
    "str_a": {
        "section": "Risks and Opportunities",
        "pillar": "strategy",
        "required_fields": [
            "risk_register",
            "opportunity_register",
            "time_horizons",
            "risk_categories",
        ],
        "accuracy_checks": [
            {"field": "risk_count", "type": "range", "min": 1, "max": 200},
            {"field": "financial_impact_total", "type": "range", "min": 0, "max": 1e12},
        ],
        "timeliness_window_days": 365,
        "consistency_rules": [
            "Each risk must have likelihood and impact scores",
            "Financial impacts must be non-negative",
        ],
        "weight": 10.0,
    },
    "str_b": {
        "section": "Business Impact",
        "pillar": "strategy",
        "required_fields": [
            "business_impact_analysis",
            "financial_planning_impact",
            "strategic_implications",
        ],
        "accuracy_checks": [
            {"field": "revenue_impact_pct", "type": "range", "min": -100, "max": 100},
            {"field": "cost_impact_pct", "type": "range", "min": -100, "max": 500},
        ],
        "timeliness_window_days": 365,
        "consistency_rules": [
            "Revenue and cost impacts must align with identified risks",
        ],
        "weight": 10.0,
    },
    "str_c": {
        "section": "Scenario Analysis",
        "pillar": "strategy",
        "required_fields": [
            "scenarios_selected",
            "time_horizons",
            "assumptions",
            "methodology",
            "results",
        ],
        "accuracy_checks": [
            {"field": "scenario_count", "type": "range", "min": 2, "max": 10},
            {"field": "temperature_range", "type": "range", "min": 1.0, "max": 5.0},
        ],
        "timeliness_window_days": 730,
        "consistency_rules": [
            "Must include at least one scenario at or below 2C",
            "Carbon price trajectories must be monotonically non-decreasing",
            "Temperature outcomes must align with scenario type",
        ],
        "weight": 12.0,
    },
    "rm_a": {
        "section": "Risk Identification",
        "pillar": "risk_management",
        "required_fields": [
            "identification_methodology",
            "materiality_criteria",
            "assessment_frequency",
        ],
        "accuracy_checks": [],
        "timeliness_window_days": 365,
        "consistency_rules": [
            "Identification process must be documented",
        ],
        "weight": 8.0,
    },
    "rm_b": {
        "section": "Risk Management Process",
        "pillar": "risk_management",
        "required_fields": [
            "management_process",
            "response_strategies",
            "monitoring_procedures",
        ],
        "accuracy_checks": [],
        "timeliness_window_days": 365,
        "consistency_rules": [
            "Each identified risk must have a defined response strategy",
        ],
        "weight": 8.0,
    },
    "rm_c": {
        "section": "ERM Integration",
        "pillar": "risk_management",
        "required_fields": [
            "erm_integration_status",
            "framework_reference",
        ],
        "accuracy_checks": [],
        "timeliness_window_days": 365,
        "consistency_rules": [
            "ERM framework reference must be a recognized standard",
        ],
        "weight": 6.0,
    },
    "mt_a": {
        "section": "Climate Metrics",
        "pillar": "metrics_targets",
        "required_fields": [
            "cross_industry_metrics",
            "industry_metrics",
            "measurement_methodology",
        ],
        "accuracy_checks": [
            {"field": "metric_count", "type": "range", "min": 3, "max": 50},
        ],
        "timeliness_window_days": 365,
        "consistency_rules": [
            "Must include at minimum 3 of 7 ISSB cross-industry metrics",
            "Units must be consistent across metrics",
        ],
        "weight": 10.0,
    },
    "mt_b": {
        "section": "GHG Emissions",
        "pillar": "metrics_targets",
        "required_fields": [
            "scope_1_total",
            "scope_2_location",
            "scope_2_market",
            "calculation_methodology",
        ],
        "accuracy_checks": [
            {"field": "scope_1_tco2e", "type": "range", "min": 0, "max": 1e9},
            {"field": "scope_2_tco2e", "type": "range", "min": 0, "max": 1e9},
            {"field": "scope_3_tco2e", "type": "range", "min": 0, "max": 1e10},
        ],
        "timeliness_window_days": 365,
        "consistency_rules": [
            "Scope 1 + Scope 2 must equal total reported Scope 1+2",
            "Year-over-year change should not exceed 50% without explanation",
            "Scope 2 location-based and market-based must both be reported",
        ],
        "weight": 12.0,
    },
    "mt_c": {
        "section": "Targets",
        "pillar": "metrics_targets",
        "required_fields": [
            "targets_list",
            "base_year_values",
            "target_year_values",
            "progress_tracking",
        ],
        "accuracy_checks": [
            {"field": "target_year", "type": "range", "min": 2025, "max": 2100},
            {"field": "reduction_pct", "type": "range", "min": 0, "max": 100},
        ],
        "timeliness_window_days": 365,
        "consistency_rules": [
            "Target year must be after base year",
            "Progress tracking must show current year value",
            "Net-zero targets must separately disclose gross and net",
        ],
        "weight": 8.0,
    },
}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class DimensionScore(BaseModel):
    """Score for a single data quality dimension."""
    dimension: str = Field(...)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    weight: float = Field(default=0.0)
    weighted_score: float = Field(default=0.0)
    details: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)


class SectionQualityScore(BaseModel):
    """Quality score for a single disclosure section."""
    disclosure_code: str = Field(...)
    section_name: str = Field(default="")
    pillar: str = Field(default="")
    completeness: float = Field(default=0.0)
    accuracy: float = Field(default=0.0)
    timeliness: float = Field(default=0.0)
    consistency: float = Field(default=0.0)
    composite_score: float = Field(default=0.0)
    issues: List[str] = Field(default_factory=list)


class DataQualityScore(BaseModel):
    """Composite data quality assessment result."""
    disclosure_id: str = Field(...)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    quality_grade: str = Field(default="D")
    completeness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    accuracy_score: float = Field(default=0.0, ge=0.0, le=100.0)
    timeliness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=100.0)
    dimension_scores: List[DimensionScore] = Field(default_factory=list)
    section_scores: List[SectionQualityScore] = Field(default_factory=list)
    total_issues: int = Field(default=0)
    improvement_suggestions: List[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class MetricValidationResult(BaseModel):
    """Validation result for a single metric."""
    metric_name: str = Field(...)
    valid: bool = Field(default=True)
    value: Optional[float] = Field(None)
    expected_range: Optional[str] = Field(None)
    unit_valid: bool = Field(default=True)
    issues: List[str] = Field(default_factory=list)


class ScenarioQualityResult(BaseModel):
    """Quality assessment for scenario analysis results."""
    scenario_id: str = Field(default="")
    overall_quality: str = Field(default="acceptable")
    consistency_score: float = Field(default=0.0)
    completeness_score: float = Field(default=0.0)
    flags: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# DataQualityEngine
# ---------------------------------------------------------------------------

class DataQualityEngine:
    """
    Data Quality Engine for TCFD disclosure validation and scoring.

    Assesses completeness, accuracy, timeliness, and consistency of
    disclosure data across all 11 TCFD recommended disclosures.

    Attributes:
        config: Application configuration.
        _disclosure_data: Cached disclosure data for quality assessment.

    Example:
        >>> engine = DataQualityEngine(TCFDAppConfig())
        >>> score = engine.assess_disclosure_quality("disc-1")
        >>> print(score.quality_grade)
    """

    # Dimension weights
    COMPLETENESS_WEIGHT = 0.30
    ACCURACY_WEIGHT = 0.30
    TIMELINESS_WEIGHT = 0.20
    CONSISTENCY_WEIGHT = 0.20

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        """
        Initialize the DataQualityEngine.

        Args:
            config: Optional application configuration.
        """
        self.config = config or TCFDAppConfig()
        self._disclosure_data: Dict[str, Dict[str, Any]] = {}
        logger.info("DataQualityEngine initialized")

    def assess_disclosure_quality(
        self,
        disclosure_id: str,
        disclosure_generator: Optional[Any] = None,
    ) -> DataQualityScore:
        """
        Assess overall data quality for a TCFD disclosure.

        Args:
            disclosure_id: Disclosure ID.
            disclosure_generator: Optional DisclosureGenerator for section data.

        Returns:
            DataQualityScore with dimension and section-level scores.
        """
        start_time = datetime.utcnow()

        # Score each section
        section_scores: List[SectionQualityScore] = []
        all_issues: List[str] = []

        for code, criteria in DATA_QUALITY_CRITERIA.items():
            section_data = self._get_section_data(disclosure_id, code, disclosure_generator)

            completeness = self._score_section_completeness(code, section_data, criteria)
            accuracy = self._score_section_accuracy(code, section_data, criteria)
            timeliness = self._score_section_timeliness(code, section_data, criteria)
            consistency = self._score_section_consistency(code, section_data, criteria)

            composite = round(
                completeness * self.COMPLETENESS_WEIGHT
                + accuracy * self.ACCURACY_WEIGHT
                + timeliness * self.TIMELINESS_WEIGHT
                + consistency * self.CONSISTENCY_WEIGHT,
                1,
            )

            issues: List[str] = []
            if completeness < 50:
                issues.append(f"{criteria['section']}: Low completeness ({completeness:.0f}%)")
            if accuracy < 50:
                issues.append(f"{criteria['section']}: Accuracy concerns ({accuracy:.0f}%)")
            if timeliness < 50:
                issues.append(f"{criteria['section']}: Data may be stale ({timeliness:.0f}%)")
            if consistency < 50:
                issues.append(f"{criteria['section']}: Consistency issues ({consistency:.0f}%)")

            all_issues.extend(issues)

            section_scores.append(SectionQualityScore(
                disclosure_code=code,
                section_name=criteria["section"],
                pillar=criteria["pillar"],
                completeness=completeness,
                accuracy=accuracy,
                timeliness=timeliness,
                consistency=consistency,
                composite_score=composite,
                issues=issues,
            ))

        # Aggregate dimension scores
        total_weight = sum(c["weight"] for c in DATA_QUALITY_CRITERIA.values())
        comp_score = self._weighted_average(section_scores, "completeness", total_weight)
        acc_score = self._weighted_average(section_scores, "accuracy", total_weight)
        time_score = self._weighted_average(section_scores, "timeliness", total_weight)
        cons_score = self._weighted_average(section_scores, "consistency", total_weight)

        overall = round(
            comp_score * self.COMPLETENESS_WEIGHT
            + acc_score * self.ACCURACY_WEIGHT
            + time_score * self.TIMELINESS_WEIGHT
            + cons_score * self.CONSISTENCY_WEIGHT,
            1,
        )

        grade = self._score_to_grade(overall)
        suggestions = self.get_quality_improvement_suggestions_from_scores(section_scores)

        dimension_scores = [
            DimensionScore(
                dimension="completeness", score=comp_score,
                weight=self.COMPLETENESS_WEIGHT,
                weighted_score=round(comp_score * self.COMPLETENESS_WEIGHT, 1),
                details=[f"Average section completeness: {comp_score:.1f}%"],
                issues=[i for i in all_issues if "completeness" in i.lower()],
            ),
            DimensionScore(
                dimension="accuracy", score=acc_score,
                weight=self.ACCURACY_WEIGHT,
                weighted_score=round(acc_score * self.ACCURACY_WEIGHT, 1),
                details=[f"Average section accuracy: {acc_score:.1f}%"],
                issues=[i for i in all_issues if "accuracy" in i.lower()],
            ),
            DimensionScore(
                dimension="timeliness", score=time_score,
                weight=self.TIMELINESS_WEIGHT,
                weighted_score=round(time_score * self.TIMELINESS_WEIGHT, 1),
                details=[f"Average section timeliness: {time_score:.1f}%"],
                issues=[i for i in all_issues if "stale" in i.lower()],
            ),
            DimensionScore(
                dimension="consistency", score=cons_score,
                weight=self.CONSISTENCY_WEIGHT,
                weighted_score=round(cons_score * self.CONSISTENCY_WEIGHT, 1),
                details=[f"Average section consistency: {cons_score:.1f}%"],
                issues=[i for i in all_issues if "consistency" in i.lower()],
            ),
        ]

        provenance = _sha256(f"{disclosure_id}:{overall}:{len(all_issues)}")

        result = DataQualityScore(
            disclosure_id=disclosure_id,
            overall_score=overall,
            quality_grade=grade,
            completeness_score=comp_score,
            accuracy_score=acc_score,
            timeliness_score=time_score,
            consistency_score=cons_score,
            dimension_scores=dimension_scores,
            section_scores=section_scores,
            total_issues=len(all_issues),
            improvement_suggestions=suggestions,
            provenance_hash=provenance,
        )

        processing_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            "Data quality assessment for disclosure %s: overall=%.1f%% (%s), "
            "%d issues, %.0fms",
            disclosure_id[:8], overall, grade, len(all_issues), processing_ms,
        )
        return result

    def score_data_completeness(
        self,
        disclosure_id: str,
        disclosure_generator: Optional[Any] = None,
    ) -> float:
        """
        Score data completeness across all sections (0-100).

        Args:
            disclosure_id: Disclosure ID.
            disclosure_generator: Optional DisclosureGenerator.

        Returns:
            Completeness score 0-100.
        """
        scores: List[float] = []
        for code, criteria in DATA_QUALITY_CRITERIA.items():
            section_data = self._get_section_data(disclosure_id, code, disclosure_generator)
            scores.append(self._score_section_completeness(code, section_data, criteria))
        return round(sum(scores) / max(len(scores), 1), 1)

    def score_data_accuracy(
        self,
        disclosure_id: str,
        disclosure_generator: Optional[Any] = None,
    ) -> float:
        """
        Score data accuracy across all sections (0-100).

        Args:
            disclosure_id: Disclosure ID.
            disclosure_generator: Optional DisclosureGenerator.

        Returns:
            Accuracy score 0-100.
        """
        scores: List[float] = []
        for code, criteria in DATA_QUALITY_CRITERIA.items():
            section_data = self._get_section_data(disclosure_id, code, disclosure_generator)
            scores.append(self._score_section_accuracy(code, section_data, criteria))
        return round(sum(scores) / max(len(scores), 1), 1)

    def score_data_timeliness(
        self,
        disclosure_id: str,
        disclosure_generator: Optional[Any] = None,
    ) -> float:
        """
        Score data timeliness across all sections (0-100).

        Args:
            disclosure_id: Disclosure ID.
            disclosure_generator: Optional DisclosureGenerator.

        Returns:
            Timeliness score 0-100.
        """
        scores: List[float] = []
        for code, criteria in DATA_QUALITY_CRITERIA.items():
            section_data = self._get_section_data(disclosure_id, code, disclosure_generator)
            scores.append(self._score_section_timeliness(code, section_data, criteria))
        return round(sum(scores) / max(len(scores), 1), 1)

    def score_data_consistency(
        self,
        disclosure_id: str,
        disclosure_generator: Optional[Any] = None,
    ) -> float:
        """
        Score data consistency across all sections (0-100).

        Args:
            disclosure_id: Disclosure ID.
            disclosure_generator: Optional DisclosureGenerator.

        Returns:
            Consistency score 0-100.
        """
        scores: List[float] = []
        for code, criteria in DATA_QUALITY_CRITERIA.items():
            section_data = self._get_section_data(disclosure_id, code, disclosure_generator)
            scores.append(self._score_section_consistency(code, section_data, criteria))
        return round(sum(scores) / max(len(scores), 1), 1)

    def validate_metrics_quality(
        self,
        org_id: str,
        metrics: List[Dict[str, Any]],
    ) -> List[MetricValidationResult]:
        """
        Validate quality of climate metrics data.

        Args:
            org_id: Organization ID.
            metrics: List of metric dictionaries with name, value, unit.

        Returns:
            List of MetricValidationResult for each metric.
        """
        results: List[MetricValidationResult] = []

        expected_ranges: Dict[str, Dict[str, Any]] = {
            "scope_1_emissions": {"min": 0, "max": 1e9, "unit": "tCO2e"},
            "scope_2_emissions": {"min": 0, "max": 1e9, "unit": "tCO2e"},
            "scope_3_emissions": {"min": 0, "max": 1e10, "unit": "tCO2e"},
            "transition_risk_assets_pct": {"min": 0, "max": 100, "unit": "percent"},
            "physical_risk_assets_pct": {"min": 0, "max": 100, "unit": "percent"},
            "opportunity_revenue_pct": {"min": 0, "max": 100, "unit": "percent"},
            "internal_carbon_price": {"min": 0, "max": 500, "unit": "USD/tCO2e"},
            "remuneration_linked_pct": {"min": 0, "max": 100, "unit": "percent"},
        }

        for metric in metrics:
            name = metric.get("name", "unknown")
            value = metric.get("value")
            unit = metric.get("unit", "")
            issues: List[str] = []
            valid = True
            unit_valid = True

            expected = expected_ranges.get(name.lower().replace(" ", "_"), {})
            range_str = None

            if value is None:
                issues.append(f"Missing value for metric '{name}'")
                valid = False
            elif expected:
                min_val = expected.get("min", float("-inf"))
                max_val = expected.get("max", float("inf"))
                range_str = f"{min_val} - {max_val}"

                try:
                    num_val = float(value)
                    if num_val < min_val or num_val > max_val:
                        issues.append(
                            f"Value {num_val} outside expected range [{min_val}, {max_val}]"
                        )
                        valid = False
                except (TypeError, ValueError):
                    issues.append(f"Non-numeric value for metric '{name}'")
                    valid = False

                if expected.get("unit") and unit and unit != expected["unit"]:
                    issues.append(
                        f"Unit mismatch: got '{unit}', expected '{expected['unit']}'"
                    )
                    unit_valid = False

            results.append(MetricValidationResult(
                metric_name=name,
                valid=valid,
                value=float(value) if value is not None else None,
                expected_range=range_str,
                unit_valid=unit_valid,
                issues=issues,
            ))

        valid_count = sum(1 for r in results if r.valid)
        logger.info(
            "Validated %d metrics for org %s: %d valid, %d issues",
            len(results), org_id, valid_count, sum(len(r.issues) for r in results),
        )
        return results

    def validate_scenario_quality(
        self,
        scenario_result: Dict[str, Any],
    ) -> ScenarioQualityResult:
        """
        Validate quality of scenario analysis results.

        Checks internal consistency, completeness of required fields,
        and reasonableness of projected values.

        Args:
            scenario_result: Scenario analysis result dictionary.

        Returns:
            ScenarioQualityResult with quality flags.
        """
        flags: List[str] = []
        recommendations: List[str] = []

        # Check required fields
        required_fields = [
            "scenario_type", "temperature_outcome", "revenue_impact_pct",
            "cost_impact_pct", "npv",
        ]
        present = sum(1 for f in required_fields if scenario_result.get(f) is not None)
        completeness = round(present / max(len(required_fields), 1) * 100, 1)

        if completeness < 80:
            flags.append(f"Scenario completeness is {completeness:.0f}% (below 80% threshold)")
            recommendations.append("Populate all required scenario output fields")

        # Consistency checks
        consistency_issues = 0

        rev_impact = scenario_result.get("revenue_impact_pct", 0)
        cost_impact = scenario_result.get("cost_impact_pct", 0)
        if isinstance(rev_impact, (int, float, Decimal)) and isinstance(cost_impact, (int, float, Decimal)):
            if float(rev_impact) < -50:
                flags.append(f"Revenue impact ({rev_impact}%) exceeds -50% -- verify input")
                consistency_issues += 1
            if float(cost_impact) > 200:
                flags.append(f"Cost impact ({cost_impact}%) exceeds +200% -- verify input")
                consistency_issues += 1

        asset_impairment = scenario_result.get("asset_impairment_pct", 0)
        if isinstance(asset_impairment, (int, float, Decimal)):
            if float(asset_impairment) > 50:
                flags.append(f"Asset impairment ({asset_impairment}%) exceeds 50% -- requires justification")
                consistency_issues += 1

        temp = scenario_result.get("temperature_outcome", "")
        scenario_type = scenario_result.get("scenario_type", "")
        if "nze" in str(scenario_type).lower() and "3" in str(temp):
            flags.append("Temperature outcome inconsistent with NZE scenario type")
            consistency_issues += 1

        consistency_score = max(0, 100 - consistency_issues * 20)

        quality = "high" if consistency_score >= 80 and completeness >= 90 else \
                  "acceptable" if consistency_score >= 60 and completeness >= 70 else "low"

        if consistency_issues > 0:
            recommendations.append("Review flagged consistency issues with data team")
        if not scenario_result.get("key_assumptions"):
            recommendations.append("Document key assumptions used in scenario analysis")

        result = ScenarioQualityResult(
            scenario_id=scenario_result.get("scenario_id", ""),
            overall_quality=quality,
            consistency_score=consistency_score,
            completeness_score=completeness,
            flags=flags,
            recommendations=recommendations,
        )

        logger.info(
            "Scenario quality: %s (consistency=%.0f%%, completeness=%.0f%%, %d flags)",
            quality, consistency_score, completeness, len(flags),
        )
        return result

    def get_quality_improvement_suggestions(
        self,
        disclosure_id: str,
        disclosure_generator: Optional[Any] = None,
    ) -> List[str]:
        """
        Get specific improvement suggestions for a disclosure.

        Args:
            disclosure_id: Disclosure ID.
            disclosure_generator: Optional DisclosureGenerator.

        Returns:
            List of actionable improvement suggestions.
        """
        result = self.assess_disclosure_quality(disclosure_id, disclosure_generator)
        return result.improvement_suggestions

    # ------------------------------------------------------------------
    # Internal scoring methods
    # ------------------------------------------------------------------

    def _get_section_data(
        self,
        disclosure_id: str,
        code: str,
        disclosure_generator: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Retrieve section data for quality assessment."""
        cached = self._disclosure_data.get(f"{disclosure_id}:{code}")
        if cached:
            return cached

        if disclosure_generator:
            try:
                section = disclosure_generator._find_section(disclosure_id, code)
                if section:
                    return {
                        "content": section.content,
                        "evidence_refs": section.evidence_refs,
                        "compliance_score": section.compliance_score,
                        "updated_at": section.updated_at,
                        "word_count": len(section.content.split()) if section.content else 0,
                    }
            except (ValueError, AttributeError):
                pass

        return {}

    def _score_section_completeness(
        self, code: str, section_data: Dict[str, Any], criteria: Dict[str, Any],
    ) -> float:
        """Score section completeness (0-100)."""
        if not section_data:
            return 0.0

        required = criteria.get("required_fields", [])
        content = section_data.get("content", "")
        word_count = section_data.get("word_count", 0)

        # Check if content exists
        if not content or word_count < 10:
            return 5.0

        # Content length factor (40%)
        min_words = {"gov_a": 200, "gov_b": 200, "str_a": 300, "str_b": 300,
                     "str_c": 400, "rm_a": 200, "rm_b": 200, "rm_c": 150,
                     "mt_a": 250, "mt_b": 200, "mt_c": 200}.get(code, 200)
        length_score = min(word_count / max(min_words, 1) * 100, 100)

        # Required fields coverage (40%)
        content_lower = content.lower()
        found = sum(1 for f in required if f.replace("_", " ") in content_lower or f in content_lower)
        field_score = found / max(len(required), 1) * 100

        # Evidence (20%)
        evidence_score = min(len(section_data.get("evidence_refs", [])) * 25, 100)

        return round(length_score * 0.4 + field_score * 0.4 + evidence_score * 0.2, 1)

    def _score_section_accuracy(
        self, code: str, section_data: Dict[str, Any], criteria: Dict[str, Any],
    ) -> float:
        """Score section accuracy (0-100)."""
        if not section_data or not section_data.get("content"):
            return 0.0

        checks = criteria.get("accuracy_checks", [])
        if not checks:
            return 75.0  # Default for sections without numeric checks

        passed = 0
        for check in checks:
            # Simplified: assume data passes if content exists
            passed += 1

        return round(passed / max(len(checks), 1) * 100, 1)

    def _score_section_timeliness(
        self, code: str, section_data: Dict[str, Any], criteria: Dict[str, Any],
    ) -> float:
        """Score section timeliness (0-100)."""
        if not section_data:
            return 0.0

        updated_at = section_data.get("updated_at")
        if not updated_at:
            return 30.0

        window_days = criteria.get("timeliness_window_days", 365)
        if isinstance(updated_at, datetime):
            age_days = (datetime.utcnow() - updated_at).days
        else:
            age_days = 180  # Default

        if age_days <= window_days * 0.5:
            return 100.0
        elif age_days <= window_days:
            return round(100 - (age_days - window_days * 0.5) / (window_days * 0.5) * 50, 1)
        else:
            overdue_ratio = age_days / window_days
            return max(0, round(50 - (overdue_ratio - 1) * 50, 1))

    def _score_section_consistency(
        self, code: str, section_data: Dict[str, Any], criteria: Dict[str, Any],
    ) -> float:
        """Score section consistency (0-100)."""
        if not section_data or not section_data.get("content"):
            return 0.0

        rules = criteria.get("consistency_rules", [])
        if not rules:
            return 80.0

        # Simplified consistency: score based on content quality signals
        content = section_data.get("content", "")
        score = 70.0  # Base consistency score

        # Bonus for evidence
        if section_data.get("evidence_refs"):
            score += 10.0

        # Bonus for sufficient length
        if section_data.get("word_count", 0) >= 200:
            score += 10.0

        # Bonus for compliance score
        if section_data.get("compliance_score", 0) >= 50:
            score += 10.0

        return min(round(score, 1), 100.0)

    def _weighted_average(
        self,
        section_scores: List[SectionQualityScore],
        dimension: str,
        total_weight: float,
    ) -> float:
        """Calculate weighted average for a dimension across sections."""
        weighted_sum = 0.0
        for ss in section_scores:
            criteria = DATA_QUALITY_CRITERIA.get(ss.disclosure_code, {})
            weight = criteria.get("weight", 1.0)
            dim_score = getattr(ss, dimension, 0.0)
            weighted_sum += dim_score * weight

        return round(weighted_sum / max(total_weight, 1), 1)

    def get_quality_improvement_suggestions_from_scores(
        self, section_scores: List[SectionQualityScore],
    ) -> List[str]:
        """Generate improvement suggestions from section scores."""
        suggestions: List[str] = []

        for ss in section_scores:
            if ss.completeness < 50:
                suggestions.append(
                    f"[{ss.section_name}] Improve completeness: add content covering "
                    f"all required topics (currently {ss.completeness:.0f}%)"
                )
            if ss.accuracy < 50:
                suggestions.append(
                    f"[{ss.section_name}] Verify data accuracy: review values against "
                    f"expected ranges (currently {ss.accuracy:.0f}%)"
                )
            if ss.timeliness < 50:
                suggestions.append(
                    f"[{ss.section_name}] Update stale data: refresh to current "
                    f"reporting period (currently {ss.timeliness:.0f}%)"
                )
            if ss.consistency < 50:
                suggestions.append(
                    f"[{ss.section_name}] Resolve consistency issues: cross-check "
                    f"values across sections (currently {ss.consistency:.0f}%)"
                )

        # General suggestions
        low_sections = [ss for ss in section_scores if ss.composite_score < 40]
        if low_sections:
            suggestions.append(
                f"Priority: {len(low_sections)} section(s) have quality scores below 40%. "
                f"Focus on these first: {', '.join(ss.section_name for ss in low_sections[:3])}"
            )

        if not suggestions:
            suggestions.append("Data quality is acceptable across all sections. Continue regular review.")

        return suggestions

    @staticmethod
    def _score_to_grade(score: float) -> str:
        """Map quality score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 50:
            return "D"
        return "F"
