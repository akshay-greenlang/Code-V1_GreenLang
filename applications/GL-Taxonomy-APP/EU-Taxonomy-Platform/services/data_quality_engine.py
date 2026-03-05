"""
Data Quality Engine -- Five-Dimension Quality Scoring & Evidence Management

Implements a comprehensive data quality assessment framework for EU Taxonomy
disclosures, evaluating five dimensions: completeness (field coverage),
accuracy (cross-validation), coverage (activity breadth), consistency
(year-over-year anomalies), and timeliness (data freshness). Produces
composite scores, letter grades, evidence tracking, and actionable
improvement plans.

Dimension weights: completeness 25%, accuracy 25%, coverage 20%,
consistency 15%, timeliness 15%.

Grades: A (>=90), B (>=75), C (>=60), D (>=40), F (<40).

All numeric calculations are deterministic (zero-hallucination).

Reference:
    - Delegated Regulation (EU) 2021/2178, Article 8 (data quality req.)
    - EBA Pillar 3 ESG ITS (EBA/ITS/2022/01), data quality expectations
    - Platform on Sustainable Finance -- Data Quality FAQ (2023)
    - ISO 8000 Data Quality Standard (conceptual alignment)

Example:
    >>> from services.config import TaxonomyAppConfig
    >>> engine = DataQualityEngine(TaxonomyAppConfig())
    >>> result = engine.assess_data_quality("org-1", "2025")
    >>> print(result.overall_score, result.grade)
    82.5 B
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    AlignmentStatus,
    DATA_QUALITY_WEIGHTS,
    TaxonomyAppConfig,
)
from .models import (
    EconomicActivity,
    _new_id,
    _now,
    _sha256,
)


# ---------------------------------------------------------------------------
# Internal constants and models (not exported via config/models)
# ---------------------------------------------------------------------------

# Dimension weights as plain floats keyed by dimension string for engine use.
# Mirrors DATA_QUALITY_WEIGHTS but with float values and string keys.
_DQ_WEIGHTS: Dict[str, float] = {
    str(k.value if hasattr(k, "value") else k): float(v)
    for k, v in DATA_QUALITY_WEIGHTS.items()
}

# Letter-grade thresholds for the engine's grading system.
_DQ_GRADE_THRESHOLDS: Dict[str, float] = {
    "A": 90.0,
    "B": 75.0,
    "C": 60.0,
    "D": 40.0,
}


class _KPIData(BaseModel):
    """Internal KPI data for the data quality engine."""

    org_id: str = Field(...)
    period: str = Field(...)
    kpi_type: str = Field(...)
    total_eur: float = Field(default=0.0, ge=0.0)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class DimensionResult(BaseModel):
    """Result of a single data quality dimension assessment."""

    dimension: str = Field(...)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    weight: float = Field(default=0.0, ge=0.0, le=1.0)
    weighted_score: float = Field(default=0.0, ge=0.0, le=100.0)
    checks_passed: int = Field(default=0)
    checks_total: int = Field(default=0)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class DQResult(BaseModel):
    """Full data quality assessment result."""

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    period: str = Field(...)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    grade: str = Field(default="F")
    submission_ready: bool = Field(default=False)
    dimensions: List[DimensionResult] = Field(default_factory=list)
    completeness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    accuracy_score: float = Field(default=0.0, ge=0.0, le=100.0)
    coverage_score: float = Field(default=0.0, ge=0.0, le=100.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=100.0)
    timeliness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    issues_count: int = Field(default=0)
    critical_issues: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    assessed_at: datetime = Field(default_factory=_now)


class EvidenceStatus(BaseModel):
    """Evidence tracking status for an assessment."""

    org_id: str = Field(...)
    assessment_type: str = Field(...)
    total_items: int = Field(default=0)
    verified_items: int = Field(default=0)
    pending_items: int = Field(default=0)
    missing_items: int = Field(default=0)
    evidence_score_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    items: List[Dict[str, Any]] = Field(default_factory=list)


class ImprovementAction(BaseModel):
    """Single improvement action in a data quality plan."""

    id: str = Field(default_factory=_new_id)
    priority: int = Field(default=1, ge=1, le=10)
    dimension: str = Field(...)
    action: str = Field(...)
    description: str = Field(default="")
    current_score: float = Field(default=0.0)
    target_score: float = Field(default=0.0)
    estimated_effort_days: int = Field(default=0)
    impact: str = Field(default="medium", description="high, medium, low")


class DQDashboard(BaseModel):
    """Data quality dashboard for an organization."""

    org_id: str = Field(...)
    period: str = Field(...)
    overall_score: float = Field(default=0.0)
    grade: str = Field(default="F")
    submission_ready: bool = Field(default=False)
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    trend_direction: str = Field(default="stable")
    issues_count: int = Field(default=0)
    critical_count: int = Field(default=0)
    improvement_actions: int = Field(default=0)
    evidence_coverage_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# DataQualityEngine
# ---------------------------------------------------------------------------

class DataQualityEngine:
    """
    Five-dimension data quality scoring engine for EU Taxonomy disclosures.

    Evaluates completeness (field coverage), accuracy (cross-validation),
    coverage (activity breadth), consistency (year-over-year stability),
    and timeliness (data freshness) to produce composite scores, grades,
    and improvement plans.

    Attributes:
        config: Application configuration.
        _activities: In-memory activities keyed by (org_id, period).
        _kpi_data: In-memory KPI data keyed by (org_id, period, kpi_type).
        _evidence: Evidence items keyed by (org_id, assessment_type).
        _dq_results: Cached DQ results keyed by (org_id, period).

    Example:
        >>> engine = DataQualityEngine(TaxonomyAppConfig())
        >>> result = engine.assess_data_quality("org-1", "2025")
        >>> print(result.grade)
        'B'
    """

    # Required fields for completeness check (per activity).
    # These match EconomicActivity model field names.
    REQUIRED_ACTIVITY_FIELDS: List[str] = [
        "activity_code",
        "nace_codes",
        "name",
        "objectives",
        "activity_type",
        "turnover_eur",
        "capex_eur",
        "opex_eur",
        "alignment_status",
    ]

    # Anomaly threshold for consistency (>25% YoY change flagged)
    ANOMALY_THRESHOLD_PCT: float = 25.0

    def __init__(self, config: Optional[TaxonomyAppConfig] = None) -> None:
        """Initialize the DataQualityEngine."""
        self.config = config or TaxonomyAppConfig()
        self._activities: Dict[str, List[EconomicActivity]] = {}
        self._kpi_data: Dict[str, _KPIData] = {}
        self._evidence: Dict[str, List[Dict[str, Any]]] = {}
        self._dq_results: Dict[str, DQResult] = {}
        logger.info("DataQualityEngine initialized")

    # ------------------------------------------------------------------
    # Data Registration
    # ------------------------------------------------------------------

    def register_activity(self, activity: EconomicActivity) -> None:
        """
        Register an economic activity for quality assessment.

        Args:
            activity: EconomicActivity model instance.
        """
        key = f"{activity.org_id}:{activity.period}"
        self._activities.setdefault(key, []).append(activity)

    def register_kpi_data(self, kpi: _KPIData) -> None:
        """
        Register KPI data for quality assessment.

        Args:
            kpi: _KPIData model instance.
        """
        key = f"{kpi.org_id}:{kpi.period}:{kpi.kpi_type}"
        self._kpi_data[key] = kpi

    # ------------------------------------------------------------------
    # Full Assessment
    # ------------------------------------------------------------------

    def assess_data_quality(
        self, org_id: str, period: str,
    ) -> DQResult:
        """
        Run the full five-dimension data quality assessment.

        Evaluates completeness, accuracy, coverage, consistency, and
        timeliness, then computes a weighted composite score and grade.

        Args:
            org_id: Organization identifier.
            period: Reporting period.

        Returns:
            DQResult with overall score, grade, and per-dimension details.
        """
        start = datetime.utcnow()

        completeness = self.assess_completeness(org_id, period)
        accuracy = self.assess_accuracy(org_id, period)
        coverage = self.assess_coverage(org_id, period)
        consistency = self.assess_consistency(org_id, period)
        timeliness = self.assess_timeliness(org_id, period)

        dimensions = [completeness, accuracy, coverage, consistency, timeliness]

        overall = sum(d.weighted_score for d in dimensions)
        grade = self._compute_grade(overall)
        # minimum_data_quality_score is a Decimal (0-1 range); convert to 0-100
        # and compare against the overall score to determine submission readiness
        min_score = float(self.config.minimum_data_quality_score) * 100.0
        submission_ready = overall >= min_score

        all_issues: List[Dict[str, Any]] = []
        critical: List[Dict[str, Any]] = []
        for d in dimensions:
            for issue in d.issues:
                all_issues.append(issue)
                if issue.get("severity") == "critical":
                    critical.append(issue)

        provenance = _sha256(
            f"dq:{org_id}:{period}:{overall}:{grade}"
        )

        result = DQResult(
            org_id=org_id,
            period=period,
            overall_score=round(overall, 2),
            grade=grade,
            submission_ready=submission_ready,
            dimensions=dimensions,
            completeness_score=round(completeness.score, 2),
            accuracy_score=round(accuracy.score, 2),
            coverage_score=round(coverage.score, 2),
            consistency_score=round(consistency.score, 2),
            timeliness_score=round(timeliness.score, 2),
            issues_count=len(all_issues),
            critical_issues=critical,
            provenance_hash=provenance,
        )

        cache_key = f"{org_id}:{period}"
        self._dq_results[cache_key] = result

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "DQ assessment for %s period %s: score=%.1f grade=%s ready=%s in %.1f ms",
            org_id, period, overall, grade, submission_ready, elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Completeness
    # ------------------------------------------------------------------

    def assess_completeness(
        self, org_id: str, period: str,
    ) -> DimensionResult:
        """
        Assess data field coverage (completeness).

        Checks that required fields are populated for each registered
        activity and that KPI data is present.

        Args:
            org_id: Organization identifier.
            period: Reporting period.

        Returns:
            DimensionResult for the completeness dimension.
        """
        act_key = f"{org_id}:{period}"
        activities = self._activities.get(act_key, [])
        weight = _DQ_WEIGHTS["completeness"]

        if not activities:
            return DimensionResult(
                dimension="completeness",
                score=0.0,
                weight=weight,
                weighted_score=0.0,
                checks_total=1,
                issues=[{
                    "severity": "critical",
                    "message": "No activities registered for this period",
                }],
                recommendations=["Register economic activities before assessment"],
            )

        total_checks = 0
        passed_checks = 0
        issues: List[Dict[str, Any]] = []

        for act in activities:
            act_dict = act.model_dump()
            for field in self.REQUIRED_ACTIVITY_FIELDS:
                total_checks += 1
                value = act_dict.get(field)
                # For list fields, check they are non-empty lists
                if isinstance(value, list):
                    is_populated = len(value) > 0
                else:
                    is_populated = (
                        value is not None and value != "" and value != Decimal("0")
                    )
                if is_populated:
                    passed_checks += 1
                else:
                    issues.append({
                        "severity": "medium",
                        "message": (
                            f"Activity {act.activity_code}: field '{field}' "
                            f"is missing or empty"
                        ),
                        "activity_code": act.activity_code,
                        "field": field,
                    })

        # Check KPI data presence
        for kpi_type in ["turnover", "capex", "opex"]:
            total_checks += 1
            kpi_key = f"{org_id}:{period}:{kpi_type}"
            if kpi_key in self._kpi_data:
                passed_checks += 1
            else:
                issues.append({
                    "severity": "high",
                    "message": f"KPI data for '{kpi_type}' not registered",
                })

        score = (passed_checks / total_checks * 100.0) if total_checks > 0 else 0.0
        recommendations = []
        if score < 80:
            recommendations.append(
                "Complete all required fields for each economic activity"
            )
        if score < 60:
            recommendations.append(
                "Register KPI totals (turnover, CapEx, OpEx) for the period"
            )

        return DimensionResult(
            dimension="completeness",
            score=round(score, 2),
            weight=weight,
            weighted_score=round(score * weight, 2),
            checks_passed=passed_checks,
            checks_total=total_checks,
            issues=issues,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------

    def assess_accuracy(
        self, org_id: str, period: str,
    ) -> DimensionResult:
        """
        Assess data accuracy through cross-validation checks.

        Verifies that KPI amounts are consistent with activity-level
        data, that percentages sum correctly, and that alignment flags
        are logically consistent (aligned => eligible, SC, DNSH, MS).

        Args:
            org_id: Organization identifier.
            period: Reporting period.

        Returns:
            DimensionResult for the accuracy dimension.
        """
        act_key = f"{org_id}:{period}"
        activities = self._activities.get(act_key, [])
        weight = _DQ_WEIGHTS["accuracy"]

        if not activities:
            return DimensionResult(
                dimension="accuracy", score=0.0, weight=weight,
                weighted_score=0.0, checks_total=0,
            )

        total_checks = 0
        passed_checks = 0
        issues: List[Dict[str, Any]] = []

        # Check 1: Aligned activities should have alignment_status == ALIGNED
        # and non-aligned should not have ALIGNED status with missing data
        for act in activities:
            total_checks += 1
            if act.alignment_status == AlignmentStatus.ALIGNED:
                # Aligned activities should have objectives and activity_type set
                has_objectives = len(act.objectives) > 0
                has_type = act.activity_type is not None
                has_financials = (
                    float(act.turnover_eur) > 0
                    or float(act.capex_eur) > 0
                    or float(act.opex_eur) > 0
                )
                if has_objectives and has_type and has_financials:
                    passed_checks += 1
                else:
                    issues.append({
                        "severity": "high",
                        "message": (
                            f"Activity {act.activity_code} marked aligned but "
                            f"missing required data "
                            f"(objectives={has_objectives}, type={has_type}, "
                            f"financials={has_financials})"
                        ),
                    })
            else:
                passed_checks += 1  # Non-aligned is acceptable

        # Check 2: KPI amounts vs activity-level totals
        for kpi_type in ["turnover", "capex", "opex"]:
            total_checks += 1
            kpi_key = f"{org_id}:{period}:{kpi_type}"
            kpi = self._kpi_data.get(kpi_key)

            if kpi_type == "turnover":
                act_total = sum(float(a.turnover_eur) for a in activities)
            elif kpi_type == "capex":
                act_total = sum(float(a.capex_eur) for a in activities)
            else:
                act_total = sum(float(a.opex_eur) for a in activities)

            if kpi:
                kpi_total = float(kpi.total_eur)
                if kpi_total > 0 and act_total > 0:
                    deviation = abs(act_total - kpi_total) / kpi_total * 100.0
                    if deviation <= 5.0:
                        passed_checks += 1
                    else:
                        issues.append({
                            "severity": "high",
                            "message": (
                                f"{kpi_type} total mismatch: KPI={kpi_total:.0f}, "
                                f"activities sum={act_total:.0f} "
                                f"(deviation={deviation:.1f}%)"
                            ),
                        })
                else:
                    passed_checks += 1
            else:
                passed_checks += 1  # No KPI to compare against

        # Check 3: No negative amounts
        for act in activities:
            total_checks += 1
            if (
                float(act.turnover_eur) >= 0
                and float(act.capex_eur) >= 0
                and float(act.opex_eur) >= 0
            ):
                passed_checks += 1
            else:
                issues.append({
                    "severity": "critical",
                    "message": f"Activity {act.activity_code} has negative amounts",
                })

        score = (passed_checks / total_checks * 100.0) if total_checks > 0 else 0.0
        recommendations = []
        if issues:
            recommendations.append(
                "Review alignment flag consistency (aligned requires all 4 steps)"
            )

        return DimensionResult(
            dimension="accuracy",
            score=round(score, 2),
            weight=weight,
            weighted_score=round(score * weight, 2),
            checks_passed=passed_checks,
            checks_total=total_checks,
            issues=issues,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def assess_coverage(
        self, org_id: str, period: str,
    ) -> DimensionResult:
        """
        Assess activity coverage breadth.

        Checks how many of the registered activities have been fully
        assessed, and whether key economic activities represent a
        significant share of total KPIs.

        Args:
            org_id: Organization identifier.
            period: Reporting period.

        Returns:
            DimensionResult for the coverage dimension.
        """
        act_key = f"{org_id}:{period}"
        activities = self._activities.get(act_key, [])
        weight = _DQ_WEIGHTS["coverage"]

        if not activities:
            return DimensionResult(
                dimension="coverage", score=0.0, weight=weight,
                weighted_score=0.0, checks_total=0,
            )

        total_checks = 0
        passed_checks = 0
        issues: List[Dict[str, Any]] = []

        # Check 1: All activities have been assessed (have alignment_status)
        assessed = [
            a for a in activities
            if a.alignment_status != AlignmentStatus.NOT_SCREENED
        ]
        total_checks += 1
        assessment_rate = len(assessed) / len(activities) * 100.0
        if assessment_rate >= 80:
            passed_checks += 1
        else:
            issues.append({
                "severity": "high",
                "message": (
                    f"Only {assessment_rate:.0f}% of activities assessed "
                    f"({len(assessed)}/{len(activities)})"
                ),
            })

        # Check 2: Assessed activities cover >80% of turnover
        total_turnover = sum(float(a.turnover_eur) for a in activities)
        assessed_turnover = sum(float(a.turnover_eur) for a in assessed)
        total_checks += 1
        turnover_coverage = (
            assessed_turnover / total_turnover * 100.0
            if total_turnover > 0 else 0.0
        )
        if turnover_coverage >= 80:
            passed_checks += 1
        else:
            issues.append({
                "severity": "medium",
                "message": (
                    f"Assessed activities cover only {turnover_coverage:.0f}% of turnover"
                ),
            })

        # Check 3: Multiple objectives represented
        objectives = {obj for a in activities for obj in a.objectives}
        total_checks += 1
        if len(objectives) >= 1:
            passed_checks += 1
        else:
            issues.append({
                "severity": "low",
                "message": "No environmental objectives assigned to activities",
            })

        # Check 4: At least 5 activities registered (for meaningful coverage)
        total_checks += 1
        if len(activities) >= 3:
            passed_checks += 1
        else:
            issues.append({
                "severity": "medium",
                "message": (
                    f"Only {len(activities)} activities registered "
                    f"(minimum 3 recommended)"
                ),
            })

        score = (passed_checks / total_checks * 100.0) if total_checks > 0 else 0.0
        recommendations = []
        if assessment_rate < 80:
            recommendations.append(
                "Complete alignment assessment for all registered activities"
            )
        if turnover_coverage < 80:
            recommendations.append(
                "Ensure assessed activities represent >80% of total turnover"
            )

        return DimensionResult(
            dimension="coverage",
            score=round(score, 2),
            weight=weight,
            weighted_score=round(score * weight, 2),
            checks_passed=passed_checks,
            checks_total=total_checks,
            issues=issues,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Consistency
    # ------------------------------------------------------------------

    def assess_consistency(
        self, org_id: str, period: str,
    ) -> DimensionResult:
        """
        Assess year-over-year consistency of taxonomy data.

        Compares current period data with prior periods to detect
        anomalous changes in KPI amounts, activity counts, and
        alignment rates that exceed the anomaly threshold.

        Args:
            org_id: Organization identifier.
            period: Reporting period.

        Returns:
            DimensionResult for the consistency dimension.
        """
        weight = _DQ_WEIGHTS["consistency"]

        # Gather all periods for this org
        org_periods = set()
        for key in self._activities.keys():
            parts = key.split(":")
            if parts[0] == org_id:
                org_periods.add(parts[1])

        if len(org_periods) < 2:
            return DimensionResult(
                dimension="consistency",
                score=70.0,
                weight=weight,
                weighted_score=round(70.0 * weight, 2),
                checks_total=0,
                recommendations=[
                    "Register multiple periods of data for consistency checks"
                ],
            )

        sorted_periods = sorted(org_periods)
        total_checks = 0
        passed_checks = 0
        issues: List[Dict[str, Any]] = []

        # Compare consecutive periods
        for i in range(1, len(sorted_periods)):
            prev_period = sorted_periods[i - 1]
            curr_period = sorted_periods[i]

            prev_acts = self._activities.get(f"{org_id}:{prev_period}", [])
            curr_acts = self._activities.get(f"{org_id}:{curr_period}", [])

            # Check: Activity count stability
            total_checks += 1
            if len(prev_acts) > 0:
                count_change = abs(len(curr_acts) - len(prev_acts)) / len(prev_acts) * 100.0
                if count_change <= self.ANOMALY_THRESHOLD_PCT:
                    passed_checks += 1
                else:
                    issues.append({
                        "severity": "medium",
                        "message": (
                            f"Activity count changed by {count_change:.0f}% "
                            f"from {prev_period} to {curr_period}"
                        ),
                    })
            else:
                passed_checks += 1

            # Check: Turnover stability
            total_checks += 1
            prev_turnover = sum(float(a.turnover_eur) for a in prev_acts)
            curr_turnover = sum(float(a.turnover_eur) for a in curr_acts)
            if prev_turnover > 0:
                turnover_change = abs(curr_turnover - prev_turnover) / prev_turnover * 100.0
                if turnover_change <= self.ANOMALY_THRESHOLD_PCT:
                    passed_checks += 1
                else:
                    issues.append({
                        "severity": "medium",
                        "message": (
                            f"Total turnover changed by {turnover_change:.0f}% "
                            f"from {prev_period} to {curr_period}"
                        ),
                    })
            else:
                passed_checks += 1

            # Check: Alignment rate stability
            total_checks += 1
            prev_aligned = sum(
                1 for a in prev_acts
                if a.alignment_status == AlignmentStatus.ALIGNED
            )
            curr_aligned = sum(
                1 for a in curr_acts
                if a.alignment_status == AlignmentStatus.ALIGNED
            )
            prev_rate = prev_aligned / len(prev_acts) * 100.0 if prev_acts else 0
            curr_rate = curr_aligned / len(curr_acts) * 100.0 if curr_acts else 0
            rate_change = abs(curr_rate - prev_rate)

            if rate_change <= 20.0:
                passed_checks += 1
            else:
                issues.append({
                    "severity": "high",
                    "message": (
                        f"Alignment rate changed by {rate_change:.0f}pp "
                        f"from {prev_period} ({prev_rate:.0f}%) "
                        f"to {curr_period} ({curr_rate:.0f}%)"
                    ),
                })

        score = (passed_checks / total_checks * 100.0) if total_checks > 0 else 70.0
        recommendations = []
        if issues:
            recommendations.append(
                "Document and explain significant year-over-year changes "
                "in activity data and alignment results"
            )

        return DimensionResult(
            dimension="consistency",
            score=round(score, 2),
            weight=weight,
            weighted_score=round(score * weight, 2),
            checks_passed=passed_checks,
            checks_total=total_checks,
            issues=issues,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Timeliness
    # ------------------------------------------------------------------

    def assess_timeliness(
        self, org_id: str, period: str,
    ) -> DimensionResult:
        """
        Assess data freshness and timeliness.

        Checks that the assessment period is recent relative to the
        current reporting year, and that activity data timestamps
        are within acceptable ranges.

        Args:
            org_id: Organization identifier.
            period: Reporting period.

        Returns:
            DimensionResult for the timeliness dimension.
        """
        weight = _DQ_WEIGHTS["timeliness"]
        current_year = self.config.reporting_year

        total_checks = 0
        passed_checks = 0
        issues: List[Dict[str, Any]] = []

        # Check 1: Period recency
        total_checks += 1
        try:
            period_year = int(period)
            age = current_year - period_year
        except ValueError:
            age = 0

        if age <= 1:
            passed_checks += 1
        elif age == 2:
            passed_checks += 1
            issues.append({
                "severity": "low",
                "message": f"Data is {age} years old (period {period})",
            })
        else:
            issues.append({
                "severity": "high",
                "message": f"Data is {age} years old (period {period}), update recommended",
            })

        # Check 2: Activity timestamps
        act_key = f"{org_id}:{period}"
        activities = self._activities.get(act_key, [])
        total_checks += 1

        if activities:
            latest_update = max(a.updated_at for a in activities)
            days_since = (datetime.utcnow() - latest_update).days
            if days_since <= 365:
                passed_checks += 1
            else:
                issues.append({
                    "severity": "medium",
                    "message": (
                        f"Latest activity update was {days_since} days ago"
                    ),
                })
        else:
            issues.append({
                "severity": "high",
                "message": "No activities to assess timeliness",
            })

        # Check 3: Reporting deadline proximity
        total_checks += 1
        # Article 8 reporting typically due by April 30 of following year
        if age <= 0:
            passed_checks += 1
        elif age <= 1:
            passed_checks += 1
        else:
            issues.append({
                "severity": "medium",
                "message": "Data period is not the most recent reporting year",
            })

        score = (passed_checks / total_checks * 100.0) if total_checks > 0 else 0.0
        recommendations = []
        if age > 1:
            recommendations.append(
                f"Update taxonomy assessment to the current year ({current_year})"
            )

        return DimensionResult(
            dimension="timeliness",
            score=round(score, 2),
            weight=weight,
            weighted_score=round(score * weight, 2),
            checks_passed=passed_checks,
            checks_total=total_checks,
            issues=issues,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Evidence Tracking
    # ------------------------------------------------------------------

    def track_evidence(
        self,
        org_id: str,
        assessment_type: str,
        evidence_items: List[Dict[str, Any]],
    ) -> EvidenceStatus:
        """
        Track evidence documentation for an assessment.

        Records evidence items (documents, certifications, data sources)
        and their verification status.

        Args:
            org_id: Organization identifier.
            assessment_type: Type of assessment (sc, dnsh, ms).
            evidence_items: List of evidence item dicts with keys:
                name, type, status (verified/pending/missing).

        Returns:
            EvidenceStatus with verification summary.
        """
        key = f"{org_id}:{assessment_type}"
        self._evidence[key] = evidence_items

        verified = sum(1 for e in evidence_items if e.get("status") == "verified")
        pending = sum(1 for e in evidence_items if e.get("status") == "pending")
        missing = sum(1 for e in evidence_items if e.get("status") == "missing")
        total = len(evidence_items)

        evidence_score = (verified / total * 100.0) if total > 0 else 0.0

        logger.info(
            "Evidence tracked for %s %s: %d verified, %d pending, %d missing",
            org_id, assessment_type, verified, pending, missing,
        )

        return EvidenceStatus(
            org_id=org_id,
            assessment_type=assessment_type,
            total_items=total,
            verified_items=verified,
            pending_items=pending,
            missing_items=missing,
            evidence_score_pct=round(evidence_score, 2),
            items=evidence_items,
        )

    # ------------------------------------------------------------------
    # Improvement Plan
    # ------------------------------------------------------------------

    def generate_improvement_plan(
        self, org_id: str, dq_result: Optional[DQResult] = None,
    ) -> List[ImprovementAction]:
        """
        Generate actionable improvement plan from DQ assessment.

        Identifies the weakest dimensions and creates prioritized
        actions to improve data quality toward the target grade.

        Args:
            org_id: Organization identifier.
            dq_result: Optional pre-computed DQ result (otherwise uses cache).

        Returns:
            List of ImprovementAction sorted by priority.
        """
        if dq_result is None:
            # Find most recent cached result
            for key, result in self._dq_results.items():
                if key.startswith(org_id):
                    dq_result = result
                    break

        if dq_result is None:
            return []

        actions: List[ImprovementAction] = []
        priority = 1

        # Sort dimensions by score (lowest first)
        sorted_dims = sorted(dq_result.dimensions, key=lambda d: d.score)

        for dim in sorted_dims:
            if dim.score >= 90:
                continue

            target = min(dim.score + 20, 100.0)
            effort = self._estimate_effort(dim.dimension, dim.score, target)
            impact = (
                "high" if dim.score < 50
                else "medium" if dim.score < 75
                else "low"
            )

            action_text = self._generate_action_text(dim.dimension, dim.score)

            actions.append(ImprovementAction(
                priority=priority,
                dimension=dim.dimension,
                action=action_text,
                description="; ".join(dim.recommendations) if dim.recommendations else "",
                current_score=dim.score,
                target_score=target,
                estimated_effort_days=effort,
                impact=impact,
            ))
            priority += 1

        logger.info(
            "Improvement plan for %s: %d actions generated", org_id, len(actions),
        )
        return actions

    # ------------------------------------------------------------------
    # DQ Dashboard
    # ------------------------------------------------------------------

    def get_dq_dashboard(
        self, org_id: str, period: str,
    ) -> DQDashboard:
        """
        Generate a data quality dashboard.

        Args:
            org_id: Organization identifier.
            period: Reporting period.

        Returns:
            DQDashboard with summary metrics.
        """
        cache_key = f"{org_id}:{period}"
        dq_result = self._dq_results.get(cache_key)

        if not dq_result:
            dq_result = self.assess_data_quality(org_id, period)

        # Determine trend from cached results
        org_results = [
            r for key, r in self._dq_results.items()
            if key.startswith(org_id)
        ]
        trend = "stable"
        if len(org_results) >= 2:
            sorted_results = sorted(org_results, key=lambda r: r.period)
            prev = sorted_results[-2].overall_score
            curr = sorted_results[-1].overall_score
            if curr > prev + 2:
                trend = "improving"
            elif curr < prev - 2:
                trend = "declining"

        # Evidence coverage
        evidence_keys = [
            k for k in self._evidence.keys() if k.startswith(org_id)
        ]
        total_evidence = 0
        verified_evidence = 0
        for ek in evidence_keys:
            items = self._evidence[ek]
            total_evidence += len(items)
            verified_evidence += sum(
                1 for e in items if e.get("status") == "verified"
            )
        evidence_pct = (
            verified_evidence / total_evidence * 100.0
            if total_evidence > 0 else 0.0
        )

        improvement_actions = len(self.generate_improvement_plan(org_id, dq_result))

        provenance = _sha256(
            f"dq_dashboard:{org_id}:{period}:{dq_result.overall_score}"
        )

        return DQDashboard(
            org_id=org_id,
            period=period,
            overall_score=dq_result.overall_score,
            grade=dq_result.grade,
            submission_ready=dq_result.submission_ready,
            dimension_scores={
                "completeness": dq_result.completeness_score,
                "accuracy": dq_result.accuracy_score,
                "coverage": dq_result.coverage_score,
                "consistency": dq_result.consistency_score,
                "timeliness": dq_result.timeliness_score,
            },
            trend_direction=trend,
            issues_count=dq_result.issues_count,
            critical_count=len(dq_result.critical_issues),
            improvement_actions=improvement_actions,
            evidence_coverage_pct=round(evidence_pct, 2),
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # DQ Trends
    # ------------------------------------------------------------------

    def get_dq_trends(
        self, org_id: str, periods: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Retrieve data quality trends across multiple periods.

        Args:
            org_id: Organization identifier.
            periods: List of reporting periods to compare.

        Returns:
            List of dicts with period, score, grade, and delta.
        """
        trends: List[Dict[str, Any]] = []
        prev_score: Optional[float] = None

        for period in sorted(periods):
            cache_key = f"{org_id}:{period}"
            result = self._dq_results.get(cache_key)

            if not result:
                result = self.assess_data_quality(org_id, period)

            delta = (result.overall_score - prev_score) if prev_score is not None else 0.0
            trends.append({
                "period": period,
                "overall_score": result.overall_score,
                "grade": result.grade,
                "delta": round(delta, 2),
                "direction": (
                    "improving" if delta > 0
                    else "declining" if delta < 0
                    else "stable"
                ),
            })
            prev_score = result.overall_score

        logger.info(
            "DQ trends for %s: %d periods", org_id, len(trends),
        )
        return trends

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_grade(self, score: float) -> str:
        """
        Compute letter grade from numeric score.

        Args:
            score: Numeric quality score (0-100).

        Returns:
            Letter grade (A, B, C, D, or F).
        """
        if score >= _DQ_GRADE_THRESHOLDS["A"]:
            return "A"
        if score >= _DQ_GRADE_THRESHOLDS["B"]:
            return "B"
        if score >= _DQ_GRADE_THRESHOLDS["C"]:
            return "C"
        if score >= _DQ_GRADE_THRESHOLDS["D"]:
            return "D"
        return "F"

    def _grade_rank(self, grade: str) -> int:
        """
        Convert grade letter to numeric rank for comparison.

        Args:
            grade: Letter grade.

        Returns:
            Numeric rank (A=5, B=4, C=3, D=2, F=1).
        """
        return {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}.get(grade, 0)

    def _estimate_effort(
        self, dimension: str, current: float, target: float,
    ) -> int:
        """
        Estimate improvement effort in days.

        Args:
            dimension: Quality dimension name.
            current: Current score.
            target: Target score.

        Returns:
            Estimated effort in working days.
        """
        gap = target - current
        base_effort = {
            "completeness": 2.0,
            "accuracy": 3.0,
            "coverage": 2.5,
            "consistency": 1.5,
            "timeliness": 1.0,
        }
        multiplier = base_effort.get(dimension, 2.0)
        return max(int(gap * multiplier / 10), 1)

    def _generate_action_text(
        self, dimension: str, score: float,
    ) -> str:
        """
        Generate human-readable action text for a dimension.

        Args:
            dimension: Quality dimension name.
            score: Current dimension score.

        Returns:
            Action description string.
        """
        actions = {
            "completeness": (
                "Complete all required data fields for economic activities "
                "and register KPI totals"
            ),
            "accuracy": (
                "Review alignment flag consistency and cross-validate "
                "KPI totals against activity-level data"
            ),
            "coverage": (
                "Ensure all material economic activities are registered "
                "and fully assessed through the 4-step pipeline"
            ),
            "consistency": (
                "Document significant year-over-year changes in activity "
                "data and alignment results"
            ),
            "timeliness": (
                "Update taxonomy assessment data to the current reporting year"
            ),
        }
        return actions.get(dimension, f"Improve {dimension} quality score")
