# -*- coding: utf-8 -*-
"""
SOC 2 Gap Analyzer - SEC-009

Implements the GapAnalyzer class for identifying and prioritizing compliance
gaps from SOC 2 self-assessments. Provides gap analysis, prioritization,
remediation effort estimation, and report generation.

The gap analyzer uses deterministic algorithms for all calculations
(ZERO-HALLUCINATION). LLM assistance is only used for generating
human-readable recommendations in the markdown report.

Example:
    >>> from greenlang.infrastructure.soc2_preparation.self_assessment import GapAnalyzer
    >>> analyzer = GapAnalyzer()
    >>> gaps = analyzer.analyze_gaps(assessment)
    >>> prioritized = analyzer.prioritize_gaps(gaps)
    >>> report = analyzer.generate_gap_report(prioritized)
    >>> print(report)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.soc2_preparation.models import (
    Assessment,
    AssessmentCriteria,
    ScoreLevel,
)
from greenlang.infrastructure.soc2_preparation.self_assessment.criteria import (
    TSC_CRITERIA,
    CriterionDefinition,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gap Data Model
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    """Risk level for compliance gaps."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EffortLevel(str, Enum):
    """Effort level for remediation."""

    TRIVIAL = "trivial"  # < 8 hours
    SMALL = "small"  # 8-24 hours
    MEDIUM = "medium"  # 24-80 hours
    LARGE = "large"  # 80-200 hours
    EPIC = "epic"  # > 200 hours


@dataclass
class Gap:
    """Represents a compliance gap for a SOC 2 criterion.

    Captures the details of a gap between current state and required
    compliance level, including effort estimates and recommendations.

    Attributes:
        criterion_id: The SOC 2 criterion identifier.
        current_score: Current maturity score (0-4).
        required_score: Required score for compliance (typically 4).
        missing_evidence: List of missing evidence items.
        remediation_recommendation: Recommended remediation steps.
        effort_hours: Estimated remediation effort in hours.
        risk_level: Risk level of this gap.
        category: TSC category (security, availability, etc.).
        subcategory: TSC subcategory (access_controls, etc.).
        priority_score: Calculated priority score for sorting.
        owner: Suggested owner/team for remediation.
        target_date: Suggested target completion date.
    """

    criterion_id: str
    current_score: int
    required_score: int = 4
    missing_evidence: List[str] = field(default_factory=list)
    remediation_recommendation: str = ""
    effort_hours: int = 0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    category: str = ""
    subcategory: str = ""
    priority_score: float = 0.0
    owner: str = ""
    target_date: Optional[date] = None

    @property
    def score_gap(self) -> int:
        """Calculate the gap between required and current score."""
        return max(0, self.required_score - self.current_score)

    @property
    def effort_level(self) -> EffortLevel:
        """Categorize effort into levels."""
        if self.effort_hours < 8:
            return EffortLevel.TRIVIAL
        elif self.effort_hours < 24:
            return EffortLevel.SMALL
        elif self.effort_hours < 80:
            return EffortLevel.MEDIUM
        elif self.effort_hours < 200:
            return EffortLevel.LARGE
        else:
            return EffortLevel.EPIC

    @property
    def is_critical_path(self) -> bool:
        """Check if this gap is on the critical path to compliance."""
        return (
            self.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
            and self.score_gap >= 2
        )


# ---------------------------------------------------------------------------
# Effort Estimation Constants
# ---------------------------------------------------------------------------

# Base hours per score level improvement
BASE_EFFORT_PER_LEVEL = {
    0: 40,  # NOT_IMPLEMENTED -> PARTIAL: 40 hours (setup from scratch)
    1: 24,  # PARTIAL -> IMPLEMENTED: 24 hours (complete implementation)
    2: 16,  # IMPLEMENTED -> TESTED: 16 hours (testing)
    3: 8,   # TESTED -> COMPLIANT: 8 hours (verification)
}

# Effort multipliers by risk level
RISK_EFFORT_MULTIPLIER = {
    RiskLevel.CRITICAL: 1.5,
    RiskLevel.HIGH: 1.2,
    RiskLevel.MEDIUM: 1.0,
    RiskLevel.LOW: 0.8,
}

# Effort multipliers by category complexity
CATEGORY_EFFORT_MULTIPLIER = {
    "security": 1.2,  # More complex, more controls
    "availability": 1.1,
    "confidentiality": 1.0,
    "processing_integrity": 1.0,
    "privacy": 1.3,  # Many requirements, often complex
}

# Priority weights for scoring
PRIORITY_WEIGHTS = {
    "risk_level": 0.4,
    "score_gap": 0.3,
    "effort": 0.2,
    "category_importance": 0.1,
}


# ---------------------------------------------------------------------------
# GapAnalyzer Class
# ---------------------------------------------------------------------------


class GapAnalyzer:
    """SOC 2 compliance gap analyzer.

    Analyzes self-assessment results to identify compliance gaps,
    estimate remediation effort, and prioritize remediation activities.

    All calculations use deterministic formulas (ZERO-HALLUCINATION).
    No LLM inference is used for scoring or prioritization.

    Attributes:
        required_score: Target score for compliance (default 4).
        effort_base: Base effort estimates per level.
        risk_multipliers: Effort multipliers by risk level.

    Example:
        >>> analyzer = GapAnalyzer()
        >>> gaps = analyzer.analyze_gaps(assessment)
        >>> prioritized = analyzer.prioritize_gaps(gaps)
        >>> report = analyzer.generate_gap_report(prioritized)
    """

    def __init__(
        self,
        required_score: int = ScoreLevel.COMPLIANT,
        effort_base: Optional[Dict[int, int]] = None,
        risk_multipliers: Optional[Dict[RiskLevel, float]] = None,
    ) -> None:
        """Initialize the gap analyzer.

        Args:
            required_score: Target compliance score (default 4).
            effort_base: Optional custom base effort per level.
            risk_multipliers: Optional custom risk multipliers.
        """
        self.required_score = required_score
        self.effort_base = effort_base or BASE_EFFORT_PER_LEVEL
        self.risk_multipliers = risk_multipliers or RISK_EFFORT_MULTIPLIER

    def analyze_gaps(self, assessment: Assessment) -> List[Gap]:
        """Analyze assessment and identify all gaps.

        Examines each criterion in the assessment and creates Gap objects
        for any criterion below the required score.

        Args:
            assessment: The assessment to analyze.

        Returns:
            List of Gap objects for all non-compliant criteria.

        Example:
            >>> gaps = analyzer.analyze_gaps(assessment)
            >>> print(f"Found {len(gaps)} gaps")
        """
        gaps: List[Gap] = []

        for criterion in assessment.criteria:
            if criterion.score < self.required_score:
                gap = self._create_gap(criterion)
                gaps.append(gap)

        logger.info(
            "Gap analysis complete: found %d gaps from %d criteria",
            len(gaps),
            len(assessment.criteria),
        )

        return gaps

    def _create_gap(self, criterion: AssessmentCriteria) -> Gap:
        """Create a Gap object from an assessment criterion.

        Args:
            criterion: The criterion with a gap.

        Returns:
            Populated Gap object.
        """
        criterion_def = TSC_CRITERIA.get(criterion.criterion_id, {})

        # Extract criterion metadata
        category = criterion_def.get("category", "security")
        subcategory = criterion_def.get("subcategory", "")
        risk_level_str = criterion_def.get("risk_level", "medium")
        risk_level = RiskLevel(risk_level_str)

        # Identify missing evidence
        missing_evidence = self._identify_missing_evidence(
            criterion=criterion,
            criterion_def=criterion_def,
        )

        # Generate remediation recommendation
        recommendation = self._generate_recommendation(
            criterion_def=criterion_def,
            current_score=criterion.score,
        )

        # Estimate effort
        effort_hours = self.estimate_remediation_effort(
            current_score=criterion.score,
            required_score=self.required_score,
            risk_level=risk_level,
            category=category,
        )

        return Gap(
            criterion_id=criterion.criterion_id,
            current_score=criterion.score,
            required_score=self.required_score,
            missing_evidence=missing_evidence,
            remediation_recommendation=recommendation,
            effort_hours=effort_hours,
            risk_level=risk_level,
            category=category,
            subcategory=subcategory,
            priority_score=0.0,  # Calculated in prioritize_gaps
            owner=self._suggest_owner(category, subcategory),
            target_date=None,  # Set based on priority
        )

    def _identify_missing_evidence(
        self,
        criterion: AssessmentCriteria,
        criterion_def: Dict[str, Any],
    ) -> List[str]:
        """Identify missing evidence for a criterion.

        Args:
            criterion: The assessment criterion.
            criterion_def: The criterion definition from TSC_CRITERIA.

        Returns:
            List of missing evidence descriptions.
        """
        required_evidence = criterion_def.get("evidence_requirements", [])

        # For now, return all required evidence if score is low
        # In a full implementation, this would compare against collected evidence
        if criterion.score <= ScoreLevel.PARTIAL:
            return required_evidence
        elif criterion.score == ScoreLevel.IMPLEMENTED:
            return [e for e in required_evidence if "test" in e.lower() or "report" in e.lower()]
        elif criterion.score == ScoreLevel.TESTED:
            return [e for e in required_evidence if "approval" in e.lower() or "review" in e.lower()]

        return []

    def _generate_recommendation(
        self,
        criterion_def: Dict[str, Any],
        current_score: int,
    ) -> str:
        """Generate remediation recommendation.

        Args:
            criterion_def: The criterion definition.
            current_score: Current maturity score.

        Returns:
            Remediation recommendation string.
        """
        recommendations = []
        common_controls = criterion_def.get("common_controls", [])
        control_points = criterion_def.get("control_points", [])

        if current_score == ScoreLevel.NOT_IMPLEMENTED:
            recommendations.append(
                "Begin control implementation by establishing foundational policies "
                "and procedures."
            )
            if common_controls:
                recommendations.append(
                    f"Implement common controls: {', '.join(common_controls[:3])}."
                )

        elif current_score == ScoreLevel.PARTIAL:
            recommendations.append(
                "Complete control implementation and ensure comprehensive documentation."
            )
            if control_points:
                recommendations.append(
                    f"Address control points: {', '.join(control_points[:2])}."
                )

        elif current_score == ScoreLevel.IMPLEMENTED:
            recommendations.append(
                "Conduct formal control testing to validate operating effectiveness. "
                "Document test procedures and maintain test results."
            )

        elif current_score == ScoreLevel.TESTED:
            recommendations.append(
                "Obtain management review and sign-off on control effectiveness. "
                "Ensure evidence packages are complete and audit-ready."
            )

        return " ".join(recommendations)

    def _suggest_owner(self, category: str, subcategory: str) -> str:
        """Suggest owner/team for remediation.

        Args:
            category: TSC category.
            subcategory: TSC subcategory.

        Returns:
            Suggested owner string.
        """
        owner_mapping = {
            ("security", "access_controls"): "Security Engineering",
            ("security", "system_operations"): "Platform/SRE Team",
            ("security", "change_management"): "Engineering Leadership",
            ("security", "risk_assessment"): "Security/Compliance Team",
            ("security", "monitoring"): "Security Operations",
            ("security", "control_environment"): "Executive/HR",
            ("security", "communication"): "Compliance Team",
            ("security", "control_activities"): "GRC Team",
            ("security", "risk_mitigation"): "Business Continuity",
            ("availability", "capacity_management"): "Platform/SRE Team",
            ("availability", "system_recovery"): "Platform/SRE Team",
            ("confidentiality", "data_classification"): "Data Governance",
            ("confidentiality", "data_disposal"): "Security Engineering",
            ("processing_integrity", "data_quality"): "Data Engineering",
            ("processing_integrity", "processing_controls"): "Engineering",
            ("privacy", "notice"): "Legal/Privacy Team",
            ("privacy", "choice_consent"): "Product/Privacy Team",
            ("privacy", "collection"): "Privacy Team",
            ("privacy", "use_retention"): "Privacy Team",
            ("privacy", "access"): "Privacy Engineering",
            ("privacy", "disclosure"): "Legal/Privacy Team",
            ("privacy", "security_for_privacy"): "Security Engineering",
            ("privacy", "quality"): "Privacy Team",
        }

        return owner_mapping.get((category, subcategory), "Compliance Team")

    def estimate_remediation_effort(
        self,
        current_score: int,
        required_score: int = ScoreLevel.COMPLIANT,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        category: str = "security",
    ) -> int:
        """Estimate remediation effort in hours.

        ZERO-HALLUCINATION: Uses deterministic formula only.

        Formula:
            base_effort = sum(base_hours for each level to improve)
            multiplied_effort = base_effort * risk_multiplier * category_multiplier

        Args:
            current_score: Current maturity score (0-4).
            required_score: Target score (default 4).
            risk_level: Risk level of the criterion.
            category: TSC category for complexity multiplier.

        Returns:
            Estimated hours for remediation.

        Example:
            >>> effort = analyzer.estimate_remediation_effort(
            ...     current_score=1,
            ...     risk_level=RiskLevel.HIGH
            ... )
            >>> print(f"Estimated: {effort} hours")
        """
        if current_score >= required_score:
            return 0

        # Calculate base effort for each level improvement needed
        base_effort = 0
        for level in range(current_score, required_score):
            base_effort += self.effort_base.get(level, 20)

        # Apply multipliers
        risk_mult = self.risk_multipliers.get(risk_level, 1.0)
        category_mult = CATEGORY_EFFORT_MULTIPLIER.get(category, 1.0)

        total_effort = base_effort * risk_mult * category_mult

        return int(round(total_effort))

    def prioritize_gaps(self, gaps: List[Gap]) -> List[Gap]:
        """Prioritize gaps for remediation.

        Calculates priority score for each gap and returns sorted list.
        Priority is based on risk level, score gap, effort, and category.

        Priority Score Formula:
            priority = (risk_weight * risk_score) +
                      (gap_weight * normalized_gap) +
                      (effort_weight * inverse_effort) +
                      (category_weight * category_score)

        Higher priority scores indicate more urgent gaps.

        Args:
            gaps: List of gaps to prioritize.

        Returns:
            Sorted list of gaps (highest priority first).

        Example:
            >>> prioritized = analyzer.prioritize_gaps(gaps)
            >>> top_gap = prioritized[0]
            >>> print(f"Top priority: {top_gap.criterion_id}")
        """
        if not gaps:
            return []

        # Calculate priority scores
        risk_scores = {
            RiskLevel.CRITICAL: 1.0,
            RiskLevel.HIGH: 0.75,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.LOW: 0.25,
        }

        category_scores = {
            "security": 1.0,
            "privacy": 0.9,
            "availability": 0.8,
            "confidentiality": 0.7,
            "processing_integrity": 0.6,
        }

        max_effort = max(g.effort_hours for g in gaps) if gaps else 1
        max_gap = 4  # Maximum possible score gap

        for gap in gaps:
            # Risk component (higher risk = higher priority)
            risk_component = risk_scores.get(gap.risk_level, 0.5)

            # Gap component (larger gap = higher priority)
            gap_component = gap.score_gap / max_gap

            # Effort component (inverted - smaller effort = higher priority for quick wins)
            effort_component = 1 - (gap.effort_hours / max_effort) if max_effort > 0 else 0.5

            # Category component
            category_component = category_scores.get(gap.category, 0.5)

            # Weighted priority score
            gap.priority_score = (
                PRIORITY_WEIGHTS["risk_level"] * risk_component +
                PRIORITY_WEIGHTS["score_gap"] * gap_component +
                PRIORITY_WEIGHTS["effort"] * effort_component +
                PRIORITY_WEIGHTS["category_importance"] * category_component
            )

        # Sort by priority (highest first)
        sorted_gaps = sorted(gaps, key=lambda g: g.priority_score, reverse=True)

        logger.info(
            "Prioritized %d gaps. Top priority: %s (score: %.3f)",
            len(sorted_gaps),
            sorted_gaps[0].criterion_id if sorted_gaps else "N/A",
            sorted_gaps[0].priority_score if sorted_gaps else 0,
        )

        return sorted_gaps

    def generate_gap_report(
        self,
        gaps: List[Gap],
        assessment_name: str = "SOC 2 Assessment",
        include_recommendations: bool = True,
    ) -> str:
        """Generate markdown gap analysis report.

        Creates a comprehensive markdown report suitable for sharing
        with stakeholders and tracking remediation progress.

        Args:
            gaps: List of gaps (preferably prioritized).
            assessment_name: Name to include in report header.
            include_recommendations: Whether to include recommendations.

        Returns:
            Markdown-formatted gap report string.

        Example:
            >>> report = analyzer.generate_gap_report(gaps)
            >>> with open("gap_report.md", "w") as f:
            ...     f.write(report)
        """
        lines = [
            f"# SOC 2 Gap Analysis Report",
            f"",
            f"**Assessment:** {assessment_name}",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Total Gaps:** {len(gaps)}",
            f"",
        ]

        if not gaps:
            lines.append("No compliance gaps identified. All criteria meet required standards.")
            return "\n".join(lines)

        # Summary statistics
        total_effort = sum(g.effort_hours for g in gaps)
        critical_count = sum(1 for g in gaps if g.risk_level == RiskLevel.CRITICAL)
        high_count = sum(1 for g in gaps if g.risk_level == RiskLevel.HIGH)

        lines.extend([
            "## Executive Summary",
            "",
            f"- **Critical Gaps:** {critical_count}",
            f"- **High-Risk Gaps:** {high_count}",
            f"- **Total Estimated Effort:** {total_effort} hours ({total_effort // 40} weeks)",
            "",
        ])

        # Gap distribution by category
        category_gaps = {}
        for gap in gaps:
            category_gaps.setdefault(gap.category, []).append(gap)

        lines.extend([
            "## Gaps by Category",
            "",
            "| Category | Count | Critical | High | Effort (hrs) |",
            "|----------|-------|----------|------|--------------|",
        ])

        for category, cat_gaps in sorted(category_gaps.items()):
            cat_critical = sum(1 for g in cat_gaps if g.risk_level == RiskLevel.CRITICAL)
            cat_high = sum(1 for g in cat_gaps if g.risk_level == RiskLevel.HIGH)
            cat_effort = sum(g.effort_hours for g in cat_gaps)
            lines.append(
                f"| {category.title()} | {len(cat_gaps)} | {cat_critical} | {cat_high} | {cat_effort} |"
            )

        lines.append("")

        # Detailed gaps (top 10 for brevity)
        lines.extend([
            "## Priority Gaps (Top 10)",
            "",
        ])

        for i, gap in enumerate(gaps[:10], 1):
            lines.extend([
                f"### {i}. {gap.criterion_id}",
                "",
                f"- **Risk Level:** {gap.risk_level.value.upper()}",
                f"- **Current Score:** {gap.current_score}/4",
                f"- **Score Gap:** {gap.score_gap}",
                f"- **Category:** {gap.category.title()} > {gap.subcategory.replace('_', ' ').title()}",
                f"- **Effort Estimate:** {gap.effort_hours} hours ({gap.effort_level.value})",
                f"- **Suggested Owner:** {gap.owner}",
                "",
            ])

            if gap.missing_evidence:
                lines.append("**Missing Evidence:**")
                for evidence in gap.missing_evidence[:5]:
                    lines.append(f"- {evidence}")
                lines.append("")

            if include_recommendations and gap.remediation_recommendation:
                lines.append(f"**Recommendation:** {gap.remediation_recommendation}")
                lines.append("")

        # Remediation timeline suggestion
        lines.extend([
            "## Suggested Remediation Timeline",
            "",
            "| Phase | Focus | Criteria | Weeks |",
            "|-------|-------|----------|-------|",
        ])

        # Group by effort level for phasing
        critical_gaps = [g for g in gaps if g.risk_level == RiskLevel.CRITICAL]
        high_gaps = [g for g in gaps if g.risk_level == RiskLevel.HIGH]
        other_gaps = [g for g in gaps if g.risk_level not in (RiskLevel.CRITICAL, RiskLevel.HIGH)]

        if critical_gaps:
            critical_effort = sum(g.effort_hours for g in critical_gaps)
            lines.append(
                f"| Phase 1 | Critical Gaps | {len(critical_gaps)} | {max(1, critical_effort // 40)} |"
            )

        if high_gaps:
            high_effort = sum(g.effort_hours for g in high_gaps)
            lines.append(
                f"| Phase 2 | High-Risk Gaps | {len(high_gaps)} | {max(1, high_effort // 40)} |"
            )

        if other_gaps:
            other_effort = sum(g.effort_hours for g in other_gaps)
            lines.append(
                f"| Phase 3 | Remaining Gaps | {len(other_gaps)} | {max(1, other_effort // 40)} |"
            )

        lines.extend([
            "",
            "---",
            "*Report generated by GreenLang SOC 2 Preparation Platform*",
        ])

        return "\n".join(lines)

    def get_remediation_roadmap(
        self,
        gaps: List[Gap],
        start_date: date,
        weekly_capacity_hours: int = 40,
    ) -> List[Dict[str, Any]]:
        """Generate remediation roadmap with target dates.

        Creates a week-by-week remediation plan based on effort estimates
        and available capacity.

        Args:
            gaps: Prioritized list of gaps.
            start_date: Remediation start date.
            weekly_capacity_hours: Available hours per week.

        Returns:
            List of roadmap items with dates and assignments.
        """
        from datetime import timedelta

        roadmap = []
        current_date = start_date
        accumulated_hours = 0

        for gap in gaps:
            # Calculate target date based on effort
            weeks_needed = (accumulated_hours + gap.effort_hours) / weekly_capacity_hours
            target_date = current_date + timedelta(weeks=int(weeks_needed))

            roadmap.append({
                "criterion_id": gap.criterion_id,
                "risk_level": gap.risk_level.value,
                "effort_hours": gap.effort_hours,
                "owner": gap.owner,
                "start_week": int(accumulated_hours / weekly_capacity_hours) + 1,
                "target_date": target_date.isoformat(),
                "priority_score": round(gap.priority_score, 3),
            })

            accumulated_hours += gap.effort_hours

        return roadmap


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def analyze_assessment_gaps(assessment: Assessment) -> List[Gap]:
    """Analyze and prioritize gaps from an assessment.

    Convenience function that creates analyzer, analyzes, and prioritizes.

    Args:
        assessment: The assessment to analyze.

    Returns:
        Prioritized list of gaps.
    """
    analyzer = GapAnalyzer()
    gaps = analyzer.analyze_gaps(assessment)
    return analyzer.prioritize_gaps(gaps)


def generate_report(assessment: Assessment) -> str:
    """Generate gap report from assessment.

    Convenience function for quick report generation.

    Args:
        assessment: The assessment to report on.

    Returns:
        Markdown gap report.
    """
    analyzer = GapAnalyzer()
    gaps = analyzer.analyze_gaps(assessment)
    prioritized = analyzer.prioritize_gaps(gaps)
    return analyzer.generate_gap_report(prioritized, assessment.name)


__all__ = [
    "GapAnalyzer",
    "Gap",
    "RiskLevel",
    "EffortLevel",
    "analyze_assessment_gaps",
    "generate_report",
]
