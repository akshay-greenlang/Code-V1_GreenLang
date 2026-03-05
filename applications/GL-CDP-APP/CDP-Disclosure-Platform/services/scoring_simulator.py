"""
CDP Scoring Simulator -- CDP Climate Change Scoring Engine

This module implements the complete CDP scoring algorithm with 17 scoring
categories, dual weightings (management and leadership), 4 scoring levels per
question (disclosure/awareness/management/leadership), overall score calculation
with band determination (D- through A), what-if analysis, A-level eligibility
checking, and score confidence intervals.

CDP Scoring Methodology:
  1. Each question earns points at 4 levels: disclosure, awareness, management, leadership.
  2. Points are aggregated into 17 scoring categories.
  3. Categories are weighted differently for management (B) and leadership (A) bands.
  4. Overall score = weighted average of category percentages.
  5. Score is mapped to a level (D- through A) based on percentage thresholds.
  6. A-level requires passing 5 mandatory requirements (hard gates).

Example:
    >>> simulator = ScoringSimulator(config, questionnaire_engine, response_manager)
    >>> result = simulator.calculate_score("questionnaire-123")
    >>> print(f"Score: {result.overall_score_pct}% ({result.overall_level.value})")
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    CDPAppConfig,
    SCORING_CATEGORY_WEIGHTS,
    SCORING_LEVEL_BANDS,
    SCORING_LEVEL_THRESHOLDS,
    ScoringBand,
    ScoringCategory,
    ScoringLevel,
)
from .models import (
    ARequirementStatus,
    CategoryScore,
    Response,
    ScoringResult,
    _new_id,
    _now,
)
from .questionnaire_engine import QuestionnaireEngine
from .response_manager import ResponseManager

logger = logging.getLogger(__name__)


class ScoringSimulator:
    """
    CDP Scoring Simulator -- calculates and predicts CDP scores.

    Implements the full CDP scoring methodology with 17 categories,
    dual weight sets, A-level requirements, what-if analysis, and
    confidence interval calculation.

    Attributes:
        config: Application configuration.
        questionnaire_engine: Reference to questionnaire engine.
        response_manager: Reference to response manager.
        _scoring_history: Cache of scoring results.

    Example:
        >>> sim = ScoringSimulator(config, q_engine, r_manager)
        >>> result = sim.calculate_score("q-123")
        >>> assert result.overall_level in ScoringLevel
    """

    def __init__(
        self,
        config: CDPAppConfig,
        questionnaire_engine: QuestionnaireEngine,
        response_manager: ResponseManager,
    ) -> None:
        """Initialize the Scoring Simulator."""
        self.config = config
        self.questionnaire_engine = questionnaire_engine
        self.response_manager = response_manager
        self._scoring_history: Dict[str, List[ScoringResult]] = {}
        self._a_requirement_data: Dict[str, Dict[str, bool]] = {}
        logger.info("ScoringSimulator initialized with 17 scoring categories")

    # ------------------------------------------------------------------
    # Main Score Calculation
    # ------------------------------------------------------------------

    def calculate_score(
        self,
        questionnaire_id: str,
        org_id: Optional[str] = None,
        year: Optional[int] = None,
        weight_mode: Optional[str] = None,
    ) -> ScoringResult:
        """
        Calculate the full CDP score for a questionnaire.

        Pipeline:
          1. Retrieve all responses for the questionnaire
          2. Score each response at 4 levels
          3. Aggregate into 17 scoring categories
          4. Apply category weights
          5. Calculate overall percentage
          6. Determine scoring level (D- through A)
          7. Check A-level requirements
          8. Calculate confidence interval

        Args:
            questionnaire_id: Questionnaire ID to score.
            org_id: Organization ID.
            year: Reporting year.
            weight_mode: "management" or "leadership" weight set.

        Returns:
            Complete ScoringResult with all breakdowns.
        """
        start_time = datetime.utcnow()
        mode = weight_mode or self.config.scoring_weight_mode

        # Step 1: Get all responses
        responses = self.response_manager.get_all_responses(questionnaire_id)
        questionnaire = self.questionnaire_engine.get_questionnaire(questionnaire_id)

        q_org_id = org_id or (questionnaire.org_id if questionnaire else "")
        q_year = year or (questionnaire.year if questionnaire else 2026)
        total_questions = questionnaire.total_questions if questionnaire else 0

        # Step 2+3: Score responses and aggregate by category
        category_scores = self._calculate_category_scores(
            questionnaire_id, responses, mode,
        )

        # Step 4+5: Calculate weighted overall score
        overall_pct = self._calculate_weighted_overall(category_scores, mode)

        # Step 6: Determine level
        overall_level = self._pct_to_level(overall_pct)
        overall_band = SCORING_LEVEL_BANDS.get(overall_level, ScoringBand.DISCLOSURE)

        # Step 7: Check A-level requirements
        a_requirements = self._check_a_requirements(questionnaire_id, q_org_id)
        a_eligible = all(r.met for r in a_requirements)

        # If score qualifies for A but requirements not met, cap at A-
        if overall_pct >= 80.0 and not a_eligible:
            overall_level = ScoringLevel.A_MINUS
            overall_band = ScoringBand.LEADERSHIP

        # Step 8: Confidence interval
        completion_pct = 0.0
        answered = len([r for r in responses if r.content])
        if total_questions > 0:
            completion_pct = answered / total_questions * 100

        confidence = self._calculate_confidence(completion_pct)
        lower_bound, upper_bound = self._calculate_confidence_interval(
            overall_pct, confidence,
        )

        # Previous year comparison
        prev_score = self._get_previous_year_score(q_org_id, q_year)
        score_delta = overall_pct - prev_score if prev_score is not None else None

        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = ScoringResult(
            questionnaire_id=questionnaire_id,
            org_id=q_org_id,
            year=q_year,
            overall_score_pct=round(overall_pct, 2),
            overall_level=overall_level,
            overall_band=overall_band,
            category_scores=category_scores,
            a_requirements=a_requirements,
            a_eligible=a_eligible,
            completion_pct=round(completion_pct, 1),
            total_questions=total_questions,
            answered_questions=answered,
            score_confidence=round(confidence, 3),
            score_lower_bound=round(lower_bound, 2),
            score_upper_bound=round(upper_bound, 2),
            previous_year_score=prev_score,
            score_delta=round(score_delta, 2) if score_delta is not None else None,
        )

        # Cache result
        if questionnaire_id not in self._scoring_history:
            self._scoring_history[questionnaire_id] = []
        self._scoring_history[questionnaire_id].append(result)

        logger.info(
            "Calculated score for %s: %.1f%% (%s) in %.1f ms",
            questionnaire_id, overall_pct, overall_level.value, elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # What-If Analysis
    # ------------------------------------------------------------------

    def simulate_changes(
        self,
        questionnaire_id: str,
        changes: Dict[str, Dict[str, Any]],
        weight_mode: Optional[str] = None,
    ) -> ScoringResult:
        """
        Simulate score changes from improving specific responses.

        Creates a virtual copy of responses with the proposed changes
        applied, then recalculates the score.

        Args:
            questionnaire_id: Base questionnaire ID.
            changes: Dict of question_id -> {content, score_level, etc.}.
            weight_mode: Scoring weight mode.

        Returns:
            Simulated ScoringResult with simulated=True.
        """
        mode = weight_mode or self.config.scoring_weight_mode

        # Get current responses
        current_responses = self.response_manager.get_all_responses(questionnaire_id)
        response_map = {r.question_id: r for r in current_responses}

        # Apply simulated changes
        simulated_responses = []
        for resp in current_responses:
            if resp.question_id in changes:
                sim_resp = resp.model_copy(deep=True)
                change = changes[resp.question_id]
                if "content" in change:
                    sim_resp.content = change["content"]
                if "score_level" in change:
                    level = change["score_level"]
                    self._apply_score_level(sim_resp, level)
                simulated_responses.append(sim_resp)
            else:
                simulated_responses.append(resp)

        # Handle new questions in changes that dont have responses yet
        for q_id, change in changes.items():
            if q_id not in response_map:
                question = self.questionnaire_engine.get_question(q_id)
                if question:
                    sim_resp = Response(
                        questionnaire_id=questionnaire_id,
                        question_id=q_id,
                        question_number=question.question_number,
                        module_code=question.module_code,
                        content=change.get("content", "Simulated response"),
                    )
                    level = change.get("score_level", "management")
                    self._apply_score_level(sim_resp, level)
                    simulated_responses.append(sim_resp)

        # Recalculate
        category_scores = self._calculate_category_scores_from_responses(
            questionnaire_id, simulated_responses, mode,
        )
        overall_pct = self._calculate_weighted_overall(category_scores, mode)
        overall_level = self._pct_to_level(overall_pct)

        questionnaire = self.questionnaire_engine.get_questionnaire(questionnaire_id)
        q_org_id = questionnaire.org_id if questionnaire else ""
        q_year = questionnaire.year if questionnaire else 2026

        a_requirements = self._check_a_requirements(questionnaire_id, q_org_id)
        a_eligible = all(r.met for r in a_requirements)

        if overall_pct >= 80.0 and not a_eligible:
            overall_level = ScoringLevel.A_MINUS

        result = ScoringResult(
            questionnaire_id=questionnaire_id,
            org_id=q_org_id,
            year=q_year,
            overall_score_pct=round(overall_pct, 2),
            overall_level=overall_level,
            overall_band=SCORING_LEVEL_BANDS.get(overall_level, ScoringBand.DISCLOSURE),
            category_scores=category_scores,
            a_requirements=a_requirements,
            a_eligible=a_eligible,
            simulated=True,
        )

        logger.info(
            "Simulated score for %s with %d changes: %.1f%% (%s)",
            questionnaire_id, len(changes), overall_pct, overall_level.value,
        )
        return result

    # ------------------------------------------------------------------
    # Score Breakdown
    # ------------------------------------------------------------------

    def get_category_breakdown(
        self,
        questionnaire_id: str,
    ) -> List[CategoryScore]:
        """Get the latest category-level score breakdown."""
        history = self._scoring_history.get(questionnaire_id, [])
        if history:
            return history[-1].category_scores
        result = self.calculate_score(questionnaire_id)
        return result.category_scores

    def get_score_trajectory(
        self,
        questionnaire_id: str,
    ) -> Dict[str, Any]:
        """
        Predict score trajectory based on current completion rate.

        Projects what the final score might be at different completion
        levels, assuming similar quality responses.
        """
        result = self.calculate_score(questionnaire_id)
        current_completion = result.completion_pct / 100.0
        current_score = result.overall_score_pct

        if current_completion <= 0:
            return {
                "current_score": 0.0,
                "current_completion": 0.0,
                "projections": [],
            }

        # Score per completion unit
        score_per_completion = current_score / max(current_completion, 0.01)

        projections = []
        for target_completion in [0.25, 0.50, 0.75, 0.90, 1.0]:
            if target_completion <= current_completion:
                projected = current_score
            else:
                # Diminishing returns model: score grows sub-linearly
                growth_factor = math.log(1 + target_completion) / math.log(1 + current_completion)
                projected = min(current_score * growth_factor, 100.0)

            projected_level = self._pct_to_level(projected)
            projections.append({
                "completion_pct": round(target_completion * 100, 1),
                "projected_score": round(projected, 2),
                "projected_level": projected_level.value,
                "projected_band": SCORING_LEVEL_BANDS.get(projected_level, ScoringBand.DISCLOSURE).value,
            })

        return {
            "current_score": round(current_score, 2),
            "current_completion": round(current_completion * 100, 1),
            "projections": projections,
        }

    # ------------------------------------------------------------------
    # A-Level Eligibility
    # ------------------------------------------------------------------

    def check_a_eligibility(
        self,
        questionnaire_id: str,
        org_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check all 5 A-level requirements.

        A-level requires:
          1. Public 1.5C transition plan
          2. Complete emissions inventory
          3. 100% Scope 1+2 verified
          4. >= 70% of one Scope 3 category verified
          5. SBTi-validated target (>= 4.2% annual reduction)

        Returns detailed requirement status.
        """
        a_reqs = self._check_a_requirements(
            questionnaire_id, org_id or "",
        )
        met_count = sum(1 for r in a_reqs if r.met)

        return {
            "a_eligible": met_count == 5,
            "requirements_met": met_count,
            "requirements_total": 5,
            "requirements": [r.model_dump() for r in a_reqs],
            "blocking_requirements": [
                r.model_dump() for r in a_reqs if not r.met
            ],
        }

    def set_a_requirement_data(
        self,
        org_id: str,
        requirement_data: Dict[str, bool],
    ) -> None:
        """
        Set A-level requirement fulfillment data for an organization.

        Args:
            org_id: Organization ID.
            requirement_data: Dict with requirement keys and boolean values.
                Keys: transition_plan_public, emissions_complete,
                      scope12_verified, scope3_verified, sbti_target.
        """
        self._a_requirement_data[org_id] = requirement_data
        logger.info(
            "Updated A-level requirement data for org %s: %s",
            org_id, requirement_data,
        )

    # ------------------------------------------------------------------
    # Score Comparison
    # ------------------------------------------------------------------

    def compare_scores(
        self,
        questionnaire_id_a: str,
        questionnaire_id_b: str,
    ) -> Dict[str, Any]:
        """
        Compare scores between two questionnaires (e.g., current vs. previous year).

        Returns overall and category-level deltas.
        """
        score_a = self.calculate_score(questionnaire_id_a)
        score_b = self.calculate_score(questionnaire_id_b)

        category_deltas = {}
        cat_map_a = {cs.category_id: cs for cs in score_a.category_scores}
        cat_map_b = {cs.category_id: cs for cs in score_b.category_scores}

        for cat_id in cat_map_a:
            score_a_val = cat_map_a[cat_id].score_pct
            score_b_val = cat_map_b.get(cat_id, CategoryScore(category_id=cat_id)).score_pct
            category_deltas[cat_id] = {
                "name": cat_map_a[cat_id].category_name,
                "score_a": round(score_a_val, 2),
                "score_b": round(score_b_val, 2),
                "delta": round(score_a_val - score_b_val, 2),
            }

        return {
            "overall_a": round(score_a.overall_score_pct, 2),
            "overall_b": round(score_b.overall_score_pct, 2),
            "overall_delta": round(score_a.overall_score_pct - score_b.overall_score_pct, 2),
            "level_a": score_a.overall_level.value,
            "level_b": score_b.overall_level.value,
            "category_deltas": category_deltas,
        }

    # ------------------------------------------------------------------
    # Internal Calculation Methods
    # ------------------------------------------------------------------

    def _calculate_category_scores(
        self,
        questionnaire_id: str,
        responses: List[Response],
        weight_mode: str,
    ) -> List[CategoryScore]:
        """Calculate scores for all 17 categories from responses."""
        return self._calculate_category_scores_from_responses(
            questionnaire_id, responses, weight_mode,
        )

    def _calculate_category_scores_from_responses(
        self,
        questionnaire_id: str,
        responses: List[Response],
        weight_mode: str,
    ) -> List[CategoryScore]:
        """Calculate category scores from a list of responses."""
        # Build response map
        response_by_question = {r.question_id: r for r in responses}

        category_scores = []
        for cat_id, cat_info in SCORING_CATEGORY_WEIGHTS.items():
            # Check sector applicability
            is_applicable = True
            if cat_info.get("sector_specific", False):
                is_applicable = False  # Default off; caller must enable

            # Get all questions for this category
            cat_questions = self.questionnaire_engine.get_scoring_category_questions(
                cat_id,
            )

            raw_score = 0.0
            max_possible = 0.0
            answered_count = 0

            for question in cat_questions:
                max_possible += question.max_points

                response = response_by_question.get(question.question_number)
                if response and response.content:
                    answered_count += 1
                    q_score = self._score_response(response, question)
                    raw_score += q_score

            # Category percentage
            score_pct = 0.0
            if max_possible > 0:
                score_pct = (raw_score / max_possible) * 100.0

            # Category level
            cat_level = self._pct_to_level(score_pct)
            cat_band = SCORING_LEVEL_BANDS.get(cat_level, ScoringBand.DISCLOSURE)

            # Weights
            w_mgmt = cat_info.get("weight_management", 0.0)
            w_lead = cat_info.get("weight_leadership", 0.0)

            weighted_mgmt = score_pct * w_mgmt / 100.0
            weighted_lead = score_pct * w_lead / 100.0

            category_scores.append(CategoryScore(
                category_id=cat_id,
                category_name=cat_info.get("name", ""),
                raw_score=round(raw_score, 2),
                max_possible=round(max_possible, 2),
                score_pct=round(score_pct, 2),
                weight_management=w_mgmt,
                weight_leadership=w_lead,
                weighted_score_mgmt=round(weighted_mgmt, 4),
                weighted_score_lead=round(weighted_lead, 4),
                level=cat_level,
                band=cat_band,
                question_count=len(cat_questions),
                answered_count=answered_count,
                applicable=is_applicable,
            ))

        return category_scores

    def _score_response(self, response: Response, question: Any) -> float:
        """
        Score a single response against its question's criteria.

        Scoring logic:
          - If response has pre-calculated scores, use those.
          - Otherwise, heuristically assign score based on content quality.
          - Text responses: length and keyword presence.
          - Numeric: whether value is provided.
          - Table: number of rows filled.
          - Yes/No: full points for positive answer.
          - Select: points based on option score_points.

        Returns score between 0 and question.max_points (deterministic).
        """
        # Use pre-calculated scores if available
        if response.total_score > 0:
            return min(response.total_score, question.max_points)

        score = 0.0
        q_type = question.question_type

        if q_type.value == "yes_no":
            content_lower = response.content.lower().strip()
            if content_lower in ("yes", "true", "1"):
                score = question.management_points
            elif content_lower in ("no", "false", "0"):
                score = question.disclosure_points

        elif q_type.value == "numeric":
            if response.numeric_value is not None or response.content.strip():
                score = question.awareness_points

        elif q_type.value == "percentage":
            if response.content.strip():
                score = question.awareness_points

        elif q_type.value == "table":
            if response.table_data:
                row_count = len(response.table_data)
                if row_count >= 3:
                    score = question.management_points
                elif row_count >= 1:
                    score = question.awareness_points
            elif response.content.strip():
                score = question.disclosure_points

        elif q_type.value in ("single_select", "multi_select"):
            if response.selected_options:
                # Sum score_points from selected options
                option_map = {opt.value: opt.score_points for opt in question.options}
                option_score = sum(
                    option_map.get(opt, 0) for opt in response.selected_options
                )
                if option_score > 0:
                    score = min(option_score, question.leadership_points)
                else:
                    score = question.disclosure_points
            elif response.content.strip():
                score = question.disclosure_points

        elif q_type.value == "text":
            content_len = len(response.content.strip())
            if content_len >= 500:
                score = question.leadership_points
            elif content_len >= 200:
                score = question.management_points
            elif content_len >= 50:
                score = question.awareness_points
            elif content_len > 0:
                score = question.disclosure_points

        elif q_type.value == "currency":
            if response.content.strip() or response.numeric_value is not None:
                score = question.awareness_points

        elif q_type.value == "file_upload":
            if response.evidence:
                score = question.management_points
            elif response.content.strip():
                score = question.disclosure_points

        else:
            # Default: any content gets disclosure points
            if response.content.strip():
                score = question.disclosure_points

        # Bonus for evidence attachments on any question type
        if response.evidence and score < question.leadership_points:
            score = min(score + 0.5, question.leadership_points)

        return min(score, question.max_points)

    def _calculate_weighted_overall(
        self,
        category_scores: List[CategoryScore],
        weight_mode: str,
    ) -> float:
        """
        Calculate weighted overall score from category scores.

        Uses either management or leadership weights depending on mode.
        Non-applicable categories are excluded and weights are normalized.
        """
        total_weighted = 0.0
        total_weight = 0.0

        for cs in category_scores:
            if not cs.applicable:
                continue

            if weight_mode == "leadership":
                weight = cs.weight_leadership
                weighted_val = cs.weighted_score_lead
            else:
                weight = cs.weight_management
                weighted_val = cs.weighted_score_mgmt

            total_weighted += weighted_val
            total_weight += weight

        if total_weight > 0:
            # Normalize to account for excluded categories
            return (total_weighted / total_weight) * 100.0

        return 0.0

    def _pct_to_level(self, pct: float) -> ScoringLevel:
        """Convert a percentage score to a CDP scoring level."""
        for level, (min_score, max_score) in SCORING_LEVEL_THRESHOLDS.items():
            if min_score <= pct <= max_score:
                return level
        if pct >= 80.0:
            return ScoringLevel.A
        return ScoringLevel.D_MINUS

    def _check_a_requirements(
        self,
        questionnaire_id: str,
        org_id: str,
    ) -> List[ARequirementStatus]:
        """Check all 5 A-level requirements."""
        req_data = self._a_requirement_data.get(org_id, {})

        requirements = [
            ARequirementStatus(
                requirement_id="AREQ01",
                name="Public transition plan",
                description="Publicly available 1.5C-aligned transition plan",
                met=req_data.get("transition_plan_public", False),
                details="Transition plan must be publicly available",
            ),
            ARequirementStatus(
                requirement_id="AREQ02",
                name="Complete emissions inventory",
                description="Complete emissions inventory with no material exclusions",
                met=req_data.get("emissions_complete", False),
                details="All material emission sources must be included",
            ),
            ARequirementStatus(
                requirement_id="AREQ03",
                name="Scope 1+2 verification",
                description="Third-party verification of 100% Scope 1 and Scope 2 emissions",
                met=req_data.get("scope12_verified", False),
                details="100% of Scope 1 and 2 must be third-party verified",
            ),
            ARequirementStatus(
                requirement_id="AREQ04",
                name="Scope 3 verification",
                description="Third-party verification of >= 70% of at least one Scope 3 category",
                met=req_data.get("scope3_verified", False),
                details="At least one Scope 3 category must have >= 70% verification",
            ),
            ARequirementStatus(
                requirement_id="AREQ05",
                name="Science-based target",
                description="SBTi-validated or 1.5C-aligned target (>= 4.2% annual absolute reduction)",
                met=req_data.get("sbti_target", False),
                details="Target must achieve >= 4.2% annual absolute reduction",
            ),
        ]

        return requirements

    def _calculate_confidence(self, completion_pct: float) -> float:
        """
        Calculate score confidence based on questionnaire completion.

        Uses a sigmoid curve: confidence increases rapidly between 30-70%
        completion, reaching 0.95 at full completion.
        """
        completion_ratio = completion_pct / 100.0
        min_completion = self.config.score_confidence_min_completion

        if completion_ratio < min_completion:
            return 0.0

        # Sigmoid-like confidence curve
        x = (completion_ratio - min_completion) / (1.0 - min_completion)
        confidence = 1.0 / (1.0 + math.exp(-8 * (x - 0.5)))
        return min(confidence * 0.95, 0.95)

    def _calculate_confidence_interval(
        self,
        score: float,
        confidence: float,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval around the predicted score.

        Lower confidence = wider interval.
        """
        if confidence <= 0:
            return (0.0, 100.0)

        # Width inversely proportional to confidence
        half_width = (1.0 - confidence) * 30.0  # Max +/- 30 points at 0 confidence
        lower = max(0.0, score - half_width)
        upper = min(100.0, score + half_width)
        return (lower, upper)

    def _apply_score_level(self, response: Response, level: str) -> None:
        """Apply a scoring level to a response for simulation."""
        question = self.questionnaire_engine.get_question(response.question_id)
        if not question:
            return

        if level == "leadership":
            response.total_score = question.leadership_points
        elif level == "management":
            response.total_score = question.management_points
        elif level == "awareness":
            response.total_score = question.awareness_points
        elif level == "disclosure":
            response.total_score = question.disclosure_points
        else:
            response.total_score = 0.0

    def _get_previous_year_score(
        self,
        org_id: str,
        current_year: int,
    ) -> Optional[float]:
        """Get the previous year's overall score for comparison."""
        prev_year = current_year - 1
        for q_id, results in self._scoring_history.items():
            for result in results:
                if result.org_id == org_id and result.year == prev_year:
                    return result.overall_score_pct
        return None
