"""
Data Quality Engine -- Emissions and Target Data Quality Assessment

Implements data quality scoring for SBTi target validation, covering
emissions data quality tiering (measured/calculated/estimated/proxy/default),
completeness assessment, consistency checks, scope coverage verification,
temporal quality analysis, emission factor quality evaluation, and
composite data quality scoring for SBTi submission readiness.

All numeric calculations are deterministic (zero-hallucination).

Reference:
    - SBTi Criteria and Recommendations v5.1 (2023), Criterion C7-C8
    - GHG Protocol Corporate Standard, Chapter 7 (Data Quality)
    - PCAF Global GHG Accounting & Reporting Standard (DQ 1-5)
    - ISO 14064-1:2018, Clause 8 (Uncertainty Assessment)

Example:
    >>> from services.config import SBTiAppConfig
    >>> engine = DataQualityEngine(SBTiAppConfig())
    >>> score = engine.assess_inventory_quality("org-1")
    >>> print(score.overall_score)
    75.0
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    DATA_QUALITY_SCORES,
    DataQualityTier,
    SBTiAppConfig,
    VERIFICATION_ASSURANCE_SCORES,
    VerificationAssurance,
)
from .models import (
    EmissionsInventory,
    Scope3CategoryEmissions,
    Target,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class DQScorecard(BaseModel):
    """Overall data quality scorecard for an organization."""

    org_id: str = Field(...)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=100.0)
    timeliness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    accuracy_score: float = Field(default=0.0, ge=0.0, le=100.0)
    coverage_score: float = Field(default=0.0, ge=0.0, le=100.0)
    grade: str = Field(
        default="C", description="A, B, C, D, F quality grade",
    )
    sbti_submission_ready: bool = Field(default=False)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class CompletenessResult(BaseModel):
    """Completeness assessment of emissions data."""

    org_id: str = Field(...)
    scope1_complete: bool = Field(default=False)
    scope2_complete: bool = Field(default=False)
    scope3_complete: bool = Field(default=False)
    scope1_coverage_pct: float = Field(default=0.0)
    scope2_coverage_pct: float = Field(default=0.0)
    scope3_categories_reported: int = Field(default=0)
    scope3_categories_total: int = Field(default=15)
    missing_scopes: List[str] = Field(default_factory=list)
    missing_scope3_categories: List[int] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=100.0)


class ConsistencyResult(BaseModel):
    """Consistency check across reporting years."""

    org_id: str = Field(...)
    years_analyzed: int = Field(default=0)
    methodology_consistent: bool = Field(default=True)
    boundary_consistent: bool = Field(default=True)
    year_over_year_checks: List[Dict[str, Any]] = Field(default_factory=list)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=100.0)


class TimelinessResult(BaseModel):
    """Timeliness assessment of emissions data."""

    org_id: str = Field(...)
    latest_data_year: Optional[int] = Field(None)
    current_year: int = Field(default=2025)
    data_age_years: int = Field(default=0)
    is_current: bool = Field(default=False)
    base_year_recency: bool = Field(default=True)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    message: str = Field(default="")


class AccuracyResult(BaseModel):
    """Accuracy assessment based on methodology and verification."""

    org_id: str = Field(...)
    primary_data_pct: float = Field(default=0.0)
    secondary_data_pct: float = Field(default=0.0)
    emission_factor_quality: str = Field(default="default")
    verification_level: str = Field(default="not_verified")
    verification_score: int = Field(default=0, ge=0, le=2)
    tier_distribution: Dict[str, float] = Field(default_factory=dict)
    score: float = Field(default=0.0, ge=0.0, le=100.0)


class CoverageResult(BaseModel):
    """Scope coverage against SBTi thresholds."""

    org_id: str = Field(...)
    scope1_2_coverage_pct: float = Field(default=0.0)
    scope1_2_threshold_pct: float = Field(default=95.0)
    scope1_2_meets_threshold: bool = Field(default=False)
    scope3_coverage_pct: float = Field(default=0.0)
    scope3_near_term_threshold_pct: float = Field(default=67.0)
    scope3_long_term_threshold_pct: float = Field(default=90.0)
    scope3_meets_near_term: bool = Field(default=False)
    scope3_meets_long_term: bool = Field(default=False)
    score: float = Field(default=0.0, ge=0.0, le=100.0)


class ImprovementPlan(BaseModel):
    """Data quality improvement plan."""

    org_id: str = Field(...)
    current_grade: str = Field(default="C")
    target_grade: str = Field(default="B")
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_improvement: float = Field(default=0.0)
    estimated_effort_days: int = Field(default=0)
    generated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# DataQualityEngine
# ---------------------------------------------------------------------------

class DataQualityEngine:
    """
    Emissions and target data quality assessment engine.

    Evaluates data quality across five dimensions: completeness,
    consistency, timeliness, accuracy, and coverage. Produces composite
    scores, grades, and improvement plans to achieve SBTi submission
    readiness.

    Attributes:
        config: Application configuration.
        _inventories: In-memory inventories keyed by org_id (list for multi-year).
        _targets: In-memory targets keyed by org_id.

    Example:
        >>> engine = DataQualityEngine(SBTiAppConfig())
        >>> scorecard = engine.assess_inventory_quality("org-1")
    """

    # Dimension weights for composite score
    WEIGHT_COMPLETENESS: float = 0.25
    WEIGHT_CONSISTENCY: float = 0.15
    WEIGHT_TIMELINESS: float = 0.15
    WEIGHT_ACCURACY: float = 0.25
    WEIGHT_COVERAGE: float = 0.20

    # Grade thresholds
    GRADE_A_THRESHOLD: float = 85.0
    GRADE_B_THRESHOLD: float = 70.0
    GRADE_C_THRESHOLD: float = 55.0
    GRADE_D_THRESHOLD: float = 40.0

    # Anomaly threshold (>30% year-over-year change is flagged)
    ANOMALY_THRESHOLD_PCT: float = 30.0

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """Initialize the DataQualityEngine."""
        self.config = config or SBTiAppConfig()
        self._inventories: Dict[str, List[EmissionsInventory]] = {}
        self._targets: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("DataQualityEngine initialized")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_inventory(self, inventory: EmissionsInventory) -> None:
        """Register an emissions inventory (supports multi-year)."""
        self._inventories.setdefault(inventory.org_id, []).append(inventory)

    def register_target(
        self, org_id: str, target_data: Dict[str, Any],
    ) -> None:
        """Register a target for quality assessment."""
        self._targets.setdefault(org_id, []).append(target_data)

    # ------------------------------------------------------------------
    # Composite Assessment
    # ------------------------------------------------------------------

    def assess_inventory_quality(self, org_id: str) -> DQScorecard:
        """
        Run a comprehensive data quality assessment.

        Evaluates all five quality dimensions and produces a composite
        scorecard with grade and SBTi submission readiness determination.

        Args:
            org_id: Organization identifier.

        Returns:
            DQScorecard with overall and dimensional scores.
        """
        start = datetime.utcnow()

        completeness = self.assess_completeness(org_id)
        consistency = self.assess_consistency(org_id)
        timeliness = self.assess_timeliness(org_id)
        accuracy = self.assess_accuracy(org_id)
        coverage = self.assess_coverage(org_id)

        # Weighted composite score
        overall = (
            completeness.score * self.WEIGHT_COMPLETENESS
            + consistency.score * self.WEIGHT_CONSISTENCY
            + timeliness.score * self.WEIGHT_TIMELINESS
            + accuracy.score * self.WEIGHT_ACCURACY
            + coverage.score * self.WEIGHT_COVERAGE
        )

        # Grade
        if overall >= self.GRADE_A_THRESHOLD:
            grade = "A"
        elif overall >= self.GRADE_B_THRESHOLD:
            grade = "B"
        elif overall >= self.GRADE_C_THRESHOLD:
            grade = "C"
        elif overall >= self.GRADE_D_THRESHOLD:
            grade = "D"
        else:
            grade = "F"

        # SBTi submission readiness requires minimum B grade
        submission_ready = grade in ("A", "B")

        # Collect issues
        issues: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        if completeness.score < 70:
            issues.append({
                "dimension": "completeness",
                "severity": "high",
                "message": f"Completeness score {completeness.score:.0f} below threshold",
            })
            if completeness.missing_scopes:
                recommendations.append(
                    f"Complete missing scope data: {', '.join(completeness.missing_scopes)}"
                )

        if consistency.anomalies:
            issues.append({
                "dimension": "consistency",
                "severity": "medium",
                "message": f"{len(consistency.anomalies)} year-over-year anomalies detected",
            })
            recommendations.append(
                "Review and explain year-over-year emission changes exceeding 30%"
            )

        if not timeliness.is_current:
            issues.append({
                "dimension": "timeliness",
                "severity": "high",
                "message": f"Data is {timeliness.data_age_years} years old",
            })
            recommendations.append(
                "Update emissions inventory to most recent reporting year"
            )

        if accuracy.score < 60:
            issues.append({
                "dimension": "accuracy",
                "severity": "medium",
                "message": f"Accuracy score {accuracy.score:.0f}; increase primary data share",
            })
            recommendations.append(
                "Increase primary (measured/calculated) data coverage above 50%"
            )

        if not coverage.scope1_2_meets_threshold:
            issues.append({
                "dimension": "coverage",
                "severity": "high",
                "message": (
                    f"Scope 1+2 coverage ({coverage.scope1_2_coverage_pct:.0f}%) "
                    f"below 95% threshold"
                ),
            })
            recommendations.append(
                "Increase Scope 1+2 coverage to >= 95% of total S1+S2 emissions"
            )

        provenance = _sha256(
            f"dq_scorecard:{org_id}:{overall}:{grade}"
        )

        scorecard = DQScorecard(
            org_id=org_id,
            overall_score=round(overall, 2),
            completeness_score=round(completeness.score, 2),
            consistency_score=round(consistency.score, 2),
            timeliness_score=round(timeliness.score, 2),
            accuracy_score=round(accuracy.score, 2),
            coverage_score=round(coverage.score, 2),
            grade=grade,
            sbti_submission_ready=submission_ready,
            issues=issues,
            recommendations=recommendations,
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "DQ assessment for org %s: overall=%.1f, grade=%s, "
            "ready=%s in %.1f ms",
            org_id, overall, grade, submission_ready, elapsed_ms,
        )
        return scorecard

    # ------------------------------------------------------------------
    # Completeness
    # ------------------------------------------------------------------

    def assess_completeness(self, org_id: str) -> CompletenessResult:
        """
        Assess data completeness for an organization.

        Checks whether Scope 1, 2, and 3 data exists, and evaluates
        Scope 3 category coverage.

        Args:
            org_id: Organization identifier.

        Returns:
            CompletenessResult with missing data identification.
        """
        inventories = self._inventories.get(org_id, [])
        if not inventories:
            return CompletenessResult(
                org_id=org_id,
                missing_scopes=["scope_1", "scope_2", "scope_3"],
                missing_scope3_categories=list(range(1, 16)),
                score=0.0,
            )

        # Use latest inventory
        latest = max(inventories, key=lambda inv: inv.year)

        s1_complete = float(latest.scope1_tco2e) > 0
        s2_complete = float(latest.scope2_market_tco2e) > 0 or float(latest.scope2_location_tco2e) > 0
        s3_complete = float(latest.scope3_total_tco2e) > 0

        missing_scopes: List[str] = []
        if not s1_complete:
            missing_scopes.append("scope_1")
        if not s2_complete:
            missing_scopes.append("scope_2")
        if not s3_complete:
            missing_scopes.append("scope_3")

        # Scope 3 category coverage
        reported_categories = set()
        for cat in latest.scope3_categories:
            if float(cat.emissions_tco2e) > 0:
                reported_categories.add(cat.category_number)

        missing_categories = [
            c for c in range(1, 16) if c not in reported_categories
        ]

        s1_coverage = float(latest.scope1_coverage_pct)
        s2_coverage = float(latest.scope2_coverage_pct)

        # Score: 40% for S1+S2 presence, 30% for S3 presence,
        # 30% for S3 category breadth
        score = 0.0
        if s1_complete:
            score += 20.0
        if s2_complete:
            score += 20.0
        if s3_complete:
            score += 30.0
        category_pct = len(reported_categories) / 15.0 * 30.0
        score += category_pct

        return CompletenessResult(
            org_id=org_id,
            scope1_complete=s1_complete,
            scope2_complete=s2_complete,
            scope3_complete=s3_complete,
            scope1_coverage_pct=s1_coverage,
            scope2_coverage_pct=s2_coverage,
            scope3_categories_reported=len(reported_categories),
            missing_scopes=missing_scopes,
            missing_scope3_categories=missing_categories,
            score=round(score, 2),
        )

    # ------------------------------------------------------------------
    # Consistency
    # ------------------------------------------------------------------

    def assess_consistency(self, org_id: str) -> ConsistencyResult:
        """
        Assess year-over-year consistency of emissions data.

        Flags anomalies where emissions change by more than 30%
        year-over-year without structural changes.

        Args:
            org_id: Organization identifier.

        Returns:
            ConsistencyResult with anomaly detection.
        """
        inventories = sorted(
            self._inventories.get(org_id, []),
            key=lambda inv: inv.year,
        )

        if len(inventories) < 2:
            return ConsistencyResult(
                org_id=org_id,
                years_analyzed=len(inventories),
                score=50.0,  # Neutral when insufficient data
            )

        yoy_checks: List[Dict[str, Any]] = []
        anomalies: List[Dict[str, Any]] = []

        for i in range(1, len(inventories)):
            prev = inventories[i - 1]
            curr = inventories[i]

            prev_total = float(prev.total_s1_s2_s3_tco2e)
            curr_total = float(curr.total_s1_s2_s3_tco2e)

            if prev_total > 0:
                change_pct = ((curr_total - prev_total) / prev_total) * 100.0
            else:
                change_pct = 0.0

            is_anomaly = abs(change_pct) > self.ANOMALY_THRESHOLD_PCT

            check = {
                "from_year": prev.year,
                "to_year": curr.year,
                "prev_total": round(prev_total, 2),
                "curr_total": round(curr_total, 2),
                "change_pct": round(change_pct, 2),
                "is_anomaly": is_anomaly,
            }
            yoy_checks.append(check)

            if is_anomaly:
                anomalies.append({
                    "years": f"{prev.year}-{curr.year}",
                    "change_pct": round(change_pct, 2),
                    "message": (
                        f"Total emissions changed by {change_pct:.1f}% "
                        f"from {prev.year} to {curr.year}. "
                        f"Exceeds {self.ANOMALY_THRESHOLD_PCT}% threshold."
                    ),
                })

        # Score: 100 if no anomalies, reduced by 20 per anomaly
        score = max(100.0 - len(anomalies) * 20.0, 0.0)

        return ConsistencyResult(
            org_id=org_id,
            years_analyzed=len(inventories),
            methodology_consistent=len(anomalies) == 0,
            boundary_consistent=True,
            year_over_year_checks=yoy_checks,
            anomalies=anomalies,
            score=round(score, 2),
        )

    # ------------------------------------------------------------------
    # Timeliness
    # ------------------------------------------------------------------

    def assess_timeliness(self, org_id: str) -> TimelinessResult:
        """
        Assess timeliness and recency of emissions data.

        SBTi requires base year 2015 or later, and current data
        should be within 1-2 years of the reporting period.

        Args:
            org_id: Organization identifier.

        Returns:
            TimelinessResult with data age assessment.
        """
        inventories = self._inventories.get(org_id, [])
        current_year = self.config.reporting_year

        if not inventories:
            return TimelinessResult(
                org_id=org_id,
                current_year=current_year,
                score=0.0,
                message="No inventory data available.",
            )

        latest = max(inventories, key=lambda inv: inv.year)
        data_age = current_year - latest.year
        is_current = data_age <= 2

        # Base year recency check (>= 2015)
        base_year_inventories = [inv for inv in inventories if inv.is_base_year]
        base_year_recent = True
        if base_year_inventories:
            base_inv = base_year_inventories[0]
            base_year_recent = base_inv.year >= 2015

        # Score: full marks for current year data, decreasing with age
        if data_age == 0:
            score = 100.0
        elif data_age == 1:
            score = 90.0
        elif data_age == 2:
            score = 70.0
        elif data_age == 3:
            score = 50.0
        else:
            score = max(30.0 - (data_age - 3) * 10.0, 0.0)

        if not base_year_recent:
            score *= 0.8

        return TimelinessResult(
            org_id=org_id,
            latest_data_year=latest.year,
            current_year=current_year,
            data_age_years=data_age,
            is_current=is_current,
            base_year_recency=base_year_recent,
            score=round(score, 2),
            message=(
                f"Latest data: {latest.year} (age: {data_age} years). "
                f"{'Current.' if is_current else 'Outdated - update needed.'}"
            ),
        )

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------

    def assess_accuracy(self, org_id: str) -> AccuracyResult:
        """
        Assess accuracy based on data quality tiers and verification.

        Higher scores for measured/calculated data and third-party
        verification. Lower scores for estimated/proxy/default data.

        Args:
            org_id: Organization identifier.

        Returns:
            AccuracyResult with methodology and verification assessment.
        """
        inventories = self._inventories.get(org_id, [])
        if not inventories:
            return AccuracyResult(org_id=org_id, score=0.0)

        latest = max(inventories, key=lambda inv: inv.year)

        # Determine primary vs secondary data share from S3 categories
        tier_counts: Dict[str, int] = {}
        total_cats = 0
        for cat in latest.scope3_categories:
            dq = cat.data_quality
            tier_counts[dq] = tier_counts.get(dq, 0) + 1
            total_cats += 1

        primary_count = tier_counts.get("measured", 0) + tier_counts.get("calculated", 0)
        secondary_count = (
            tier_counts.get("estimated", 0)
            + tier_counts.get("proxy", 0)
            + tier_counts.get("default", 0)
        )

        primary_pct = (primary_count / total_cats * 100.0) if total_cats > 0 else 0.0
        secondary_pct = (secondary_count / total_cats * 100.0) if total_cats > 0 else 0.0

        # Tier distribution as percentages
        tier_distribution: Dict[str, float] = {}
        for tier, count in tier_counts.items():
            tier_distribution[tier] = round(count / total_cats * 100.0, 2) if total_cats > 0 else 0.0

        # Overall data quality from inventory
        ef_quality = latest.data_quality_overall
        verification = latest.verification_status

        # Verification score
        try:
            va = VerificationAssurance(verification)
            ver_score = VERIFICATION_ASSURANCE_SCORES.get(va, 0)
        except ValueError:
            ver_score = 0

        # Accuracy score: 50% primary data share, 30% EF quality, 20% verification
        ef_scores = {
            "measured": 100, "calculated": 80, "estimated": 50,
            "proxy": 30, "default": 10,
        }
        ef_score = ef_scores.get(ef_quality, 30)

        score = (
            primary_pct * 0.50
            + ef_score * 0.30
            + (ver_score / 2.0) * 100.0 * 0.20
        )

        return AccuracyResult(
            org_id=org_id,
            primary_data_pct=round(primary_pct, 2),
            secondary_data_pct=round(secondary_pct, 2),
            emission_factor_quality=ef_quality,
            verification_level=verification,
            verification_score=ver_score,
            tier_distribution=tier_distribution,
            score=round(score, 2),
        )

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def assess_coverage(self, org_id: str) -> CoverageResult:
        """
        Assess scope coverage against SBTi thresholds.

        SBTi requires >= 95% Scope 1+2 coverage, >= 67% Scope 3
        coverage (near-term), and >= 90% Scope 3 coverage (long-term).

        Args:
            org_id: Organization identifier.

        Returns:
            CoverageResult with threshold compliance.
        """
        inventories = self._inventories.get(org_id, [])
        if not inventories:
            return CoverageResult(org_id=org_id, score=0.0)

        latest = max(inventories, key=lambda inv: inv.year)

        # S1+S2 coverage
        s1_cov = float(latest.scope1_coverage_pct)
        s2_cov = float(latest.scope2_coverage_pct)
        s1_s2_cov = (s1_cov + s2_cov) / 2.0
        s1_s2_meets = s1_s2_cov >= 95.0

        # S3 coverage from included categories
        total_s3 = float(latest.scope3_total_tco2e)
        included_s3 = 0.0
        for cat in latest.scope3_categories:
            if cat.included_in_target:
                included_s3 += float(cat.emissions_tco2e)

        s3_cov = (included_s3 / total_s3 * 100.0) if total_s3 > 0 else 0.0
        s3_near_term_meets = s3_cov >= 67.0
        s3_long_term_meets = s3_cov >= 90.0

        # Score: 50% S1+S2, 50% S3
        s1_s2_score = min(s1_s2_cov / 95.0 * 100.0, 100.0)
        s3_score = min(s3_cov / 67.0 * 100.0, 100.0)
        score = s1_s2_score * 0.50 + s3_score * 0.50

        return CoverageResult(
            org_id=org_id,
            scope1_2_coverage_pct=round(s1_s2_cov, 2),
            scope1_2_threshold_pct=95.0,
            scope1_2_meets_threshold=s1_s2_meets,
            scope3_coverage_pct=round(s3_cov, 2),
            scope3_near_term_threshold_pct=67.0,
            scope3_long_term_threshold_pct=90.0,
            scope3_meets_near_term=s3_near_term_meets,
            scope3_meets_long_term=s3_long_term_meets,
            score=round(score, 2),
        )

    # ------------------------------------------------------------------
    # Improvement Plan
    # ------------------------------------------------------------------

    def generate_improvement_plan(self, org_id: str) -> ImprovementPlan:
        """
        Generate a data quality improvement plan.

        Based on the current scorecard, identifies specific actions
        to improve data quality and achieve SBTi submission readiness.

        Args:
            org_id: Organization identifier.

        Returns:
            ImprovementPlan with prioritized actions.
        """
        scorecard = self.assess_inventory_quality(org_id)
        current_grade = scorecard.grade

        target_grade = "B" if current_grade in ("C", "D", "F") else "A"

        actions: List[Dict[str, Any]] = []
        total_effort = 0

        # Generate actions based on dimensional scores
        if scorecard.completeness_score < 80:
            effort = 20
            actions.append({
                "priority": 1,
                "dimension": "completeness",
                "action": "Complete all Scope 1, 2, and 3 inventory data",
                "current_score": scorecard.completeness_score,
                "target_score": 90.0,
                "estimated_days": effort,
            })
            total_effort += effort

        if scorecard.accuracy_score < 70:
            effort = 30
            actions.append({
                "priority": 2,
                "dimension": "accuracy",
                "action": "Increase primary data collection; obtain third-party verification",
                "current_score": scorecard.accuracy_score,
                "target_score": 80.0,
                "estimated_days": effort,
            })
            total_effort += effort

        if scorecard.coverage_score < 80:
            effort = 15
            actions.append({
                "priority": 3,
                "dimension": "coverage",
                "action": "Ensure Scope 1+2 coverage >= 95% and Scope 3 >= 67%",
                "current_score": scorecard.coverage_score,
                "target_score": 90.0,
                "estimated_days": effort,
            })
            total_effort += effort

        if scorecard.timeliness_score < 70:
            effort = 10
            actions.append({
                "priority": 4,
                "dimension": "timeliness",
                "action": "Update inventory to current reporting year",
                "current_score": scorecard.timeliness_score,
                "target_score": 90.0,
                "estimated_days": effort,
            })
            total_effort += effort

        if scorecard.consistency_score < 80:
            effort = 10
            actions.append({
                "priority": 5,
                "dimension": "consistency",
                "action": "Document and explain year-over-year emission changes",
                "current_score": scorecard.consistency_score,
                "target_score": 90.0,
                "estimated_days": effort,
            })
            total_effort += effort

        estimated_improvement = min(
            sum(
                (a["target_score"] - a["current_score"])
                * ({"completeness": 0.25, "consistency": 0.15, "timeliness": 0.15,
                    "accuracy": 0.25, "coverage": 0.20}.get(a["dimension"], 0.2))
                for a in actions
            ),
            100 - scorecard.overall_score,
        )

        return ImprovementPlan(
            org_id=org_id,
            current_grade=current_grade,
            target_grade=target_grade,
            actions=actions,
            estimated_improvement=round(estimated_improvement, 2),
            estimated_effort_days=total_effort,
        )

    # ------------------------------------------------------------------
    # Retrieve Scores
    # ------------------------------------------------------------------

    def get_tier_score(self, tier: str) -> int:
        """
        Get the numeric score for a data quality tier.

        Args:
            tier: Data quality tier name (measured, calculated, etc.).

        Returns:
            Numeric score (1=best, 5=worst).
        """
        try:
            dq = DataQualityTier(tier)
            return DATA_QUALITY_SCORES.get(dq, 5)
        except ValueError:
            return 5
