"""
CDP Gap Analysis Engine -- Gap Identification and Recommendations

This module identifies gaps between current CDP questionnaire responses and
scoring criteria, categorizes gaps by level (disclosure/awareness/management/
leadership), prioritizes by score impact, provides actionable recommendations,
estimates effort, and predicts score uplift per gap.

Key capabilities:
  - Compare responses against scoring criteria for each question
  - Gap categorization: disclosure/awareness/management/leadership
  - Priority ranking by score impact
  - Actionable recommendations per gap with examples
  - Effort estimation (low/medium/high)
  - Score uplift prediction per gap closed
  - Gap tracking over time

Example:
    >>> engine = GapAnalysisEngine(config, q_engine, r_manager, scorer)
    >>> analysis = engine.analyze("questionnaire-123")
    >>> print(f"Found {analysis.total_gaps} gaps, potential uplift: {analysis.total_potential_uplift}%")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import (
    CDPAppConfig,
    CDPModule,
    EffortLevel,
    GapLevel,
    GapSeverity,
    SCORING_CATEGORY_WEIGHTS,
    ScoringLevel,
)
from .models import (
    GapAnalysis,
    GapItem,
    GapRecommendation,
    Question,
    Response,
    _new_id,
    _now,
)
from .questionnaire_engine import QuestionnaireEngine
from .response_manager import ResponseManager
from .scoring_simulator import ScoringSimulator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recommendation Templates per scoring category
# ---------------------------------------------------------------------------

CATEGORY_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "SC01": {
        "name": "Governance",
        "disclosure": GapRecommendation(
            title="Disclose board oversight of climate issues",
            description="Describe whether the board has oversight of climate-related issues, including the position(s) responsible.",
            example_response="The Board's Sustainability Committee, chaired by the Lead Independent Director, has formal oversight of climate-related issues and meets quarterly to review climate strategy and risks.",
        ),
        "management": GapRecommendation(
            title="Document management accountability and incentives",
            description="Provide details of management-level climate responsibilities and performance-linked incentives.",
            example_response="Climate KPIs representing 15% of executive variable compensation include Scope 1+2 reduction targets, renewable energy procurement milestones, and transition plan implementation progress.",
        ),
        "leadership": GapRecommendation(
            title="Demonstrate board competency and regular training",
            description="Show that board members have climate competency through regular training and skills assessments.",
            example_response="Board members complete annual climate literacy training covering TCFD, transition risk scenarios, and emerging regulations. A skills matrix is maintained and reviewed during annual board effectiveness evaluation.",
        ),
    },
    "SC07": {
        "name": "Targets",
        "disclosure": GapRecommendation(
            title="Disclose emission reduction targets",
            description="Report at least one emission reduction target covering Scope 1 and 2 emissions.",
        ),
        "management": GapRecommendation(
            title="Set science-based targets",
            description="Commit to or set Science Based Targets covering Scope 1, 2, and 3 emissions aligned with 1.5C.",
        ),
        "leadership": GapRecommendation(
            title="Validate SBTi targets and demonstrate progress",
            description="Have SBTi-validated near-term and long-term targets with documented annual progress toward each target.",
        ),
    },
    "SC09": {
        "name": "Scope 1 & 2 emissions",
        "disclosure": GapRecommendation(
            title="Report Scope 1 and 2 emissions",
            description="Disclose gross global Scope 1 and Scope 2 (location and market-based) emissions in metric tonnes CO2e.",
        ),
        "management": GapRecommendation(
            title="Break down emissions and describe methodology",
            description="Provide emissions breakdowns by GHG type, country, and business division. Describe calculation methodologies used.",
        ),
        "leadership": GapRecommendation(
            title="Obtain third-party verification of 100% Scope 1+2",
            description="Have 100% of Scope 1 and Scope 2 emissions verified by an accredited third party at limited or reasonable assurance.",
        ),
    },
    "SC10": {
        "name": "Scope 3 emissions",
        "disclosure": GapRecommendation(
            title="Screen and report Scope 3 categories",
            description="Screen all 15 Scope 3 categories and report emissions for all relevant categories.",
        ),
        "management": GapRecommendation(
            title="Use detailed methodologies for material categories",
            description="Apply supplier-specific or hybrid methodologies for the most material Scope 3 categories.",
        ),
        "leadership": GapRecommendation(
            title="Verify Scope 3 emissions",
            description="Obtain third-party verification of at least 70% of one or more Scope 3 categories.",
        ),
    },
    "SC15": {
        "name": "Transition plan",
        "disclosure": GapRecommendation(
            title="Describe transition plan existence",
            description="Indicate whether your organization has a climate transition plan.",
        ),
        "management": GapRecommendation(
            title="Develop a comprehensive transition plan",
            description="Create a 1.5C-aligned transition plan with milestones, decarbonization levers, and investment commitments.",
        ),
        "leadership": GapRecommendation(
            title="Publicly publish and independently review transition plan",
            description="Publish the transition plan publicly, obtain board approval, and have it independently reviewed or assured.",
        ),
    },
}


# Default recommendations for categories without specific templates
DEFAULT_RECOMMENDATIONS: Dict[str, GapRecommendation] = {
    "disclosure": GapRecommendation(
        title="Provide basic disclosure",
        description="Answer the question with relevant information about your organization's approach.",
    ),
    "awareness": GapRecommendation(
        title="Demonstrate awareness and assessment",
        description="Show that your organization is aware of the issue and has assessed its relevance.",
    ),
    "management": GapRecommendation(
        title="Document management actions",
        description="Describe specific management actions, policies, and processes in place.",
    ),
    "leadership": GapRecommendation(
        title="Demonstrate best practice",
        description="Show industry-leading practices, quantitative evidence, and continuous improvement.",
    ),
}


# Effort estimation heuristics
EFFORT_BY_QUESTION_TYPE: Dict[str, EffortLevel] = {
    "yes_no": EffortLevel.LOW,
    "single_select": EffortLevel.LOW,
    "multi_select": EffortLevel.LOW,
    "numeric": EffortLevel.LOW,
    "percentage": EffortLevel.LOW,
    "text": EffortLevel.MEDIUM,
    "table": EffortLevel.HIGH,
    "currency": EffortLevel.MEDIUM,
    "date": EffortLevel.LOW,
    "file_upload": EffortLevel.HIGH,
}


class GapAnalysisEngine:
    """
    CDP Gap Analysis Engine -- identifies gaps and recommends improvements.

    Analyzes responses against scoring criteria to find gaps at each
    scoring level, prioritize by impact, and provide actionable
    recommendations with effort estimation.

    Attributes:
        config: Application configuration.
        questionnaire_engine: Reference to questionnaire engine.
        response_manager: Reference to response manager.
        scoring_simulator: Reference to scoring simulator.
        _analysis_history: Cache of past analyses.

    Example:
        >>> engine = GapAnalysisEngine(config, q_engine, r_manager, scorer)
        >>> result = engine.analyze("q-123", target_level=ScoringLevel.A)
    """

    def __init__(
        self,
        config: CDPAppConfig,
        questionnaire_engine: QuestionnaireEngine,
        response_manager: ResponseManager,
        scoring_simulator: ScoringSimulator,
    ) -> None:
        """Initialize the Gap Analysis Engine."""
        self.config = config
        self.questionnaire_engine = questionnaire_engine
        self.response_manager = response_manager
        self.scoring_simulator = scoring_simulator
        self._analysis_history: Dict[str, List[GapAnalysis]] = {}
        logger.info("GapAnalysisEngine initialized")

    # ------------------------------------------------------------------
    # Main Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        questionnaire_id: str,
        target_level: ScoringLevel = ScoringLevel.A,
        modules: Optional[List[str]] = None,
    ) -> GapAnalysis:
        """
        Run gap analysis on a questionnaire.

        Identifies all gaps between current response quality and the
        target scoring level, categorizes them, and provides recommendations.

        Args:
            questionnaire_id: Questionnaire to analyze.
            target_level: Target scoring level for gap identification.
            modules: Specific modules to analyze; all if None.

        Returns:
            Complete GapAnalysis with prioritized gaps.
        """
        start_time = datetime.utcnow()

        # Get current score
        current_score = self.scoring_simulator.calculate_score(questionnaire_id)

        # Get all questions and responses
        questionnaire = self.questionnaire_engine.get_questionnaire(questionnaire_id)
        org_id = questionnaire.org_id if questionnaire else ""
        year = questionnaire.year if questionnaire else 2026

        all_questions = self.questionnaire_engine.get_all_questions()
        responses = self.response_manager.get_all_responses(questionnaire_id)
        response_map = {r.question_id: r for r in responses}

        # Identify gaps
        gaps = []
        for question in all_questions:
            if modules and question.module_code.value not in modules:
                continue

            response = response_map.get(question.question_number)
            question_gaps = self._identify_question_gaps(
                question, response, target_level, current_score.overall_score_pct,
            )
            gaps.extend(question_gaps)

        # Sort by score uplift (highest impact first)
        gaps.sort(key=lambda g: g.score_uplift, reverse=True)

        # Assign priorities based on combined score impact and effort
        gaps = self._assign_priorities(gaps)

        # Calculate totals
        total_uplift = sum(g.score_uplift for g in gaps)
        projected_score = min(current_score.overall_score_pct + total_uplift, 100.0)

        # Group by module and category
        gaps_by_module: Dict[str, int] = {}
        gaps_by_category: Dict[str, int] = {}
        gaps_by_severity: Dict[str, int] = {}

        for gap in gaps:
            mod = gap.module_code.value
            gaps_by_module[mod] = gaps_by_module.get(mod, 0) + 1

            cat = gap.scoring_category
            if cat:
                gaps_by_category[cat] = gaps_by_category.get(cat, 0) + 1

            sev = gap.severity.value
            gaps_by_severity[sev] = gaps_by_severity.get(sev, 0) + 1

        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000

        analysis = GapAnalysis(
            questionnaire_id=questionnaire_id,
            org_id=org_id,
            year=year,
            gaps=gaps,
            total_potential_uplift=round(total_uplift, 2),
            current_score=round(current_score.overall_score_pct, 2),
            projected_score=round(projected_score, 2),
            gaps_by_module=gaps_by_module,
            gaps_by_category=gaps_by_category,
            gaps_by_severity=gaps_by_severity,
        )

        # Cache
        if questionnaire_id not in self._analysis_history:
            self._analysis_history[questionnaire_id] = []
        self._analysis_history[questionnaire_id].append(analysis)

        logger.info(
            "Gap analysis for %s: %d gaps found, potential uplift: %.1f%% (%.1f ms)",
            questionnaire_id, analysis.total_gaps, total_uplift, elapsed,
        )
        return analysis

    # ------------------------------------------------------------------
    # Gap Detail Queries
    # ------------------------------------------------------------------

    def get_latest_analysis(
        self,
        questionnaire_id: str,
    ) -> Optional[GapAnalysis]:
        """Get the most recent gap analysis for a questionnaire."""
        history = self._analysis_history.get(questionnaire_id, [])
        return history[-1] if history else None

    def get_gaps_by_module(
        self,
        questionnaire_id: str,
        module_code: str,
    ) -> List[GapItem]:
        """Get gaps for a specific module."""
        analysis = self.get_latest_analysis(questionnaire_id)
        if not analysis:
            return []
        return [g for g in analysis.gaps if g.module_code.value == module_code]

    def get_gaps_by_severity(
        self,
        questionnaire_id: str,
        severity: GapSeverity,
    ) -> List[GapItem]:
        """Get gaps of a specific severity."""
        analysis = self.get_latest_analysis(questionnaire_id)
        if not analysis:
            return []
        return [g for g in analysis.gaps if g.severity == severity]

    def get_gaps_by_category(
        self,
        questionnaire_id: str,
        category_id: str,
    ) -> List[GapItem]:
        """Get gaps for a specific scoring category."""
        analysis = self.get_latest_analysis(questionnaire_id)
        if not analysis:
            return []
        return [g for g in analysis.gaps if g.scoring_category == category_id]

    def get_top_impact_gaps(
        self,
        questionnaire_id: str,
        limit: int = 10,
    ) -> List[GapItem]:
        """Get the top N gaps ranked by score impact."""
        analysis = self.get_latest_analysis(questionnaire_id)
        if not analysis:
            return []
        sorted_gaps = sorted(analysis.gaps, key=lambda g: g.score_uplift, reverse=True)
        return sorted_gaps[:limit]

    def get_quick_wins(
        self,
        questionnaire_id: str,
        limit: int = 10,
    ) -> List[GapItem]:
        """Get gaps that are low effort but high impact (quick wins)."""
        analysis = self.get_latest_analysis(questionnaire_id)
        if not analysis:
            return []

        quick_wins = [
            g for g in analysis.gaps
            if g.effort == EffortLevel.LOW and g.score_uplift > 0
        ]
        quick_wins.sort(key=lambda g: g.score_uplift, reverse=True)
        return quick_wins[:limit]

    # ------------------------------------------------------------------
    # Gap Resolution Tracking
    # ------------------------------------------------------------------

    def mark_gap_resolved(
        self,
        questionnaire_id: str,
        gap_id: str,
    ) -> bool:
        """Mark a gap as resolved."""
        analysis = self.get_latest_analysis(questionnaire_id)
        if not analysis:
            return False

        for gap in analysis.gaps:
            if gap.id == gap_id:
                gap.resolved = True
                gap.resolved_at = _now()
                logger.info("Gap %s marked as resolved", gap_id)
                return True
        return False

    def get_resolution_progress(
        self,
        questionnaire_id: str,
    ) -> Dict[str, Any]:
        """Get gap resolution progress statistics."""
        analysis = self.get_latest_analysis(questionnaire_id)
        if not analysis:
            return {"total": 0, "resolved": 0, "remaining": 0, "progress_pct": 0.0}

        resolved = sum(1 for g in analysis.gaps if g.resolved)
        total = analysis.total_gaps
        remaining = total - resolved

        return {
            "total": total,
            "resolved": resolved,
            "remaining": remaining,
            "progress_pct": round(resolved / total * 100, 1) if total > 0 else 0.0,
            "resolved_uplift": sum(g.score_uplift for g in analysis.gaps if g.resolved),
            "remaining_uplift": sum(g.score_uplift for g in analysis.gaps if not g.resolved),
        }

    # ------------------------------------------------------------------
    # Comparison Over Time
    # ------------------------------------------------------------------

    def compare_analyses(
        self,
        questionnaire_id: str,
    ) -> Dict[str, Any]:
        """Compare the latest two gap analyses to show progress."""
        history = self._analysis_history.get(questionnaire_id, [])
        if len(history) < 2:
            return {"comparison_available": False}

        current = history[-1]
        previous = history[-2]

        return {
            "comparison_available": True,
            "current_gaps": current.total_gaps,
            "previous_gaps": previous.total_gaps,
            "gaps_delta": current.total_gaps - previous.total_gaps,
            "current_uplift": current.total_potential_uplift,
            "previous_uplift": previous.total_potential_uplift,
            "uplift_delta": current.total_potential_uplift - previous.total_potential_uplift,
            "current_score": current.current_score,
            "previous_score": previous.current_score,
            "score_improvement": current.current_score - previous.current_score,
        }

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _identify_question_gaps(
        self,
        question: Question,
        response: Optional[Response],
        target_level: ScoringLevel,
        current_overall_score: float,
    ) -> List[GapItem]:
        """Identify gaps for a single question."""
        gaps = []

        # Determine current score level for this question
        current_q_score = 0.0
        if response:
            current_q_score = response.total_score

        # Determine target score for this question based on target level
        target_points = self._get_target_points(question, target_level)

        if current_q_score >= target_points:
            return gaps  # No gap

        # Determine gap level
        gap_level = self._determine_gap_level(current_q_score, question)

        # Calculate score uplift
        uplift = self._estimate_score_uplift(
            question, current_q_score, target_points, current_overall_score,
        )

        # Get recommendations
        recommendations = self._get_recommendations(
            question, gap_level,
        )

        # Estimate effort
        effort = self._estimate_effort(question, gap_level, response)

        # Determine severity
        severity = self._determine_severity(uplift, gap_level)

        # Build description
        description = self._build_gap_description(
            question, response, gap_level,
        )

        gap = GapItem(
            question_id=question.question_number,
            question_number=question.question_number,
            module_code=question.module_code,
            scoring_category=question.scoring_categories[0] if question.scoring_categories else "",
            gap_level=gap_level,
            severity=severity,
            current_score=round(current_q_score, 2),
            target_score=round(target_points, 2),
            score_uplift=round(uplift, 4),
            effort=effort,
            description=description,
            recommendations=recommendations,
        )
        gaps.append(gap)

        return gaps

    def _get_target_points(
        self,
        question: Question,
        target_level: ScoringLevel,
    ) -> float:
        """Get target points for a question based on target scoring level."""
        if target_level in (ScoringLevel.A, ScoringLevel.A_MINUS):
            return question.leadership_points
        elif target_level in (ScoringLevel.B, ScoringLevel.B_MINUS):
            return question.management_points
        elif target_level in (ScoringLevel.C, ScoringLevel.C_MINUS):
            return question.awareness_points
        else:
            return question.disclosure_points

    def _determine_gap_level(
        self,
        current_score: float,
        question: Question,
    ) -> GapLevel:
        """Determine which scoring level the gap falls at."""
        if current_score < question.disclosure_points:
            return GapLevel.DISCLOSURE
        elif current_score < question.awareness_points:
            return GapLevel.AWARENESS
        elif current_score < question.management_points:
            return GapLevel.MANAGEMENT
        else:
            return GapLevel.LEADERSHIP

    def _estimate_score_uplift(
        self,
        question: Question,
        current_score: float,
        target_score: float,
        current_overall: float,
    ) -> float:
        """
        Estimate the overall score uplift from closing this gap.

        Uses the question's scoring weight and category weight to
        estimate impact on overall score.
        """
        score_improvement = target_score - current_score
        if score_improvement <= 0:
            return 0.0

        # Get category weight
        total_weight = 0.0
        for cat_id in question.scoring_categories:
            cat_info = SCORING_CATEGORY_WEIGHTS.get(cat_id, {})
            total_weight += cat_info.get("weight_management", 0.0)

        if total_weight <= 0:
            return 0.0

        # Normalize: single question impact relative to category
        cat_questions = 0
        for cat_id in question.scoring_categories:
            cat_qs = self.questionnaire_engine.get_scoring_category_questions(cat_id)
            cat_questions = max(cat_questions, len(cat_qs))

        if cat_questions <= 0:
            return 0.0

        per_question_impact = total_weight / cat_questions
        uplift = (score_improvement / question.max_points) * per_question_impact

        return max(0.0, uplift)

    def _determine_severity(
        self,
        uplift: float,
        gap_level: GapLevel,
    ) -> GapSeverity:
        """Determine gap severity based on uplift and level."""
        if gap_level == GapLevel.DISCLOSURE:
            return GapSeverity.CRITICAL if uplift > 0.5 else GapSeverity.HIGH
        elif gap_level == GapLevel.AWARENESS:
            return GapSeverity.HIGH if uplift > 0.3 else GapSeverity.MEDIUM
        elif gap_level == GapLevel.MANAGEMENT:
            return GapSeverity.MEDIUM if uplift > 0.2 else GapSeverity.LOW
        else:
            return GapSeverity.LOW

    def _estimate_effort(
        self,
        question: Question,
        gap_level: GapLevel,
        response: Optional[Response],
    ) -> EffortLevel:
        """Estimate effort to close a gap."""
        base_effort = EFFORT_BY_QUESTION_TYPE.get(
            question.question_type.value, EffortLevel.MEDIUM,
        )

        # If no response at all, effort is higher
        if not response or not response.content:
            if base_effort == EffortLevel.LOW:
                return EffortLevel.MEDIUM
            return EffortLevel.HIGH

        # Leadership gaps are always high effort
        if gap_level == GapLevel.LEADERSHIP:
            return EffortLevel.HIGH

        return base_effort

    def _get_recommendations(
        self,
        question: Question,
        gap_level: GapLevel,
    ) -> List[GapRecommendation]:
        """Get recommendations for closing a gap."""
        recommendations = []

        for cat_id in question.scoring_categories:
            cat_recs = CATEGORY_RECOMMENDATIONS.get(cat_id)
            if cat_recs:
                level_key = gap_level.value
                rec = cat_recs.get(level_key)
                if rec:
                    recommendations.append(rec)

        # Add default if no specific recommendations found
        if not recommendations:
            level_key = gap_level.value
            default_rec = DEFAULT_RECOMMENDATIONS.get(level_key)
            if default_rec:
                recommendations.append(default_rec)

        return recommendations

    def _build_gap_description(
        self,
        question: Question,
        response: Optional[Response],
        gap_level: GapLevel,
    ) -> str:
        """Build a human-readable gap description."""
        if not response or not response.content:
            return (
                f"Question {question.question_number} has no response. "
                f"Provide at least a {gap_level.value}-level answer to earn points "
                f"in the {', '.join(question.scoring_categories)} scoring categories."
            )

        return (
            f"Question {question.question_number} has a response but does not meet "
            f"{gap_level.value}-level criteria. Enhance the response to address "
            f"the specific requirements for {gap_level.value}-level scoring."
        )

    def _assign_priorities(self, gaps: List[GapItem]) -> List[GapItem]:
        """Assign severity priorities considering both uplift and effort."""
        for gap in gaps:
            # Quick wins get bumped up
            if gap.effort == EffortLevel.LOW and gap.score_uplift > 0.3:
                if gap.severity == GapSeverity.LOW:
                    gap.severity = GapSeverity.MEDIUM
            # High effort, low impact get lowered
            elif gap.effort == EffortLevel.HIGH and gap.score_uplift < 0.1:
                gap.severity = GapSeverity.LOW

        return gaps
