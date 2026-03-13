# -*- coding: utf-8 -*-
"""
EngagementVerifier Engine - AGENT-EUDR-031

Verification engine that assesses whether operator engagement with
indigenous communities meets the substantive requirements of ILO
Convention 169, UNDRIP, and applicable national legislation. Evaluates
engagement quality across 6 dimensions with weighted composite scoring.

Zero-Hallucination: All dimension scoring uses deterministic formulas
based on quantifiable engagement data. No LLM involvement in score
calculation.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-031 (GL-EUDR-SET-031)
Regulation: EU 2023/1115 (EUDR), ILO Convention 169, UNDRIP
Status: Production Ready
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
)
from greenlang.agents.eudr.stakeholder_engagement.models import (
    EngagementAssessment,
    EngagementDimension,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    ProvenanceTracker,
)

logger = logging.getLogger(__name__)

# Recommendation threshold: dimensions scoring below this generate recommendations
_LOW_THRESHOLD = Decimal("50")

# Recommendation templates per dimension
_RECOMMENDATIONS: Dict[EngagementDimension, List[str]] = {
    EngagementDimension.INCLUSIVENESS: [
        "Improve inclusiveness by ensuring all vulnerable groups are represented in consultations.",
        "Increase stakeholder group diversity in engagement activities.",
    ],
    EngagementDimension.TRANSPARENCY: [
        "Improve transparency by publishing regular impact reports in local languages.",
        "Share monitoring data openly with affected communities.",
    ],
    EngagementDimension.RESPONSIVENESS: [
        "Improve responsiveness by reducing grievance response time.",
        "Implement faster acknowledgement procedures for stakeholder concerns.",
    ],
    EngagementDimension.ACCOUNTABILITY: [
        "Improve accountability by fulfilling all outstanding commitments.",
        "Commission independent audits of engagement quality.",
    ],
    EngagementDimension.CULTURAL_SENSITIVITY: [
        "Improve cultural sensitivity by engaging with community elders and traditional leaders.",
        "Ensure all engagement follows indigenous protocols and customs.",
    ],
    EngagementDimension.RIGHTS_RESPECT: [
        "Improve rights respect by completing rights impact assessments.",
        "Ensure FPIC is obtained before any activities affecting indigenous territories.",
    ],
}


class EngagementVerifier:
    """Engagement quality verification engine.

    Assesses engagement quality across 6 dimensions and calculates
    weighted composite scores for regulatory compliance.

    Attributes:
        _config: Engine configuration.
        _provenance: Provenance hash chain tracker.
    """

    def __init__(self, config: StakeholderEngagementConfig) -> None:
        """Initialize EngagementVerifier.

        Args:
            config: Stakeholder engagement configuration.
        """
        self._config = config
        self._provenance = ProvenanceTracker()
        logger.info("EngagementVerifier initialized")

    async def assess_engagement(
        self,
        operator_id: str,
        stakeholder_id: str,
        engagement_data: Dict[str, Any],
    ) -> EngagementAssessment:
        """Perform full engagement quality assessment.

        Args:
            operator_id: Operator being assessed.
            stakeholder_id: Stakeholder whose engagement is assessed.
            engagement_data: Data describing engagement activities.

        Returns:
            EngagementAssessment with dimension scores and composite.

        Raises:
            ValueError: If required fields are empty.
        """
        if not operator_id or not operator_id.strip():
            raise ValueError("operator_id is required")
        if not stakeholder_id or not stakeholder_id.strip():
            raise ValueError("stakeholder_id is required")

        now = datetime.now(tz=timezone.utc)
        assessment_id = f"EA-{uuid.uuid4().hex[:8].upper()}"

        # Score all 6 dimensions
        dimension_scores: Dict[EngagementDimension, Decimal] = {}
        for dimension in EngagementDimension:
            dimension_scores[dimension] = self.score_dimension(dimension, engagement_data)

        # Calculate composite
        composite = self.calculate_composite(dimension_scores)

        # Generate recommendations
        recommendations = self.generate_recommendations(dimension_scores)

        assessment = EngagementAssessment(
            assessment_id=assessment_id,
            operator_id=operator_id,
            stakeholder_id=stakeholder_id,
            assessment_date=now,
            dimension_scores=dimension_scores,
            composite_score=composite,
            recommendations=recommendations,
            evidence_refs=[],
        )

        self._provenance.record(
            "engagement", "assess", assessment_id, "AGENT-EUDR-031",
            metadata={"composite_score": str(composite)},
        )
        logger.info(
            "Engagement assessment %s: composite=%.1f",
            assessment_id, composite,
        )
        return assessment

    def score_dimension(
        self,
        dimension: EngagementDimension,
        data: Dict[str, Any],
    ) -> Decimal:
        """Score a single engagement dimension.

        Args:
            dimension: Dimension to score.
            data: Engagement data for scoring.

        Returns:
            Score between 0 and 100 as Decimal.
        """
        scorers = {
            EngagementDimension.INCLUSIVENESS: self._score_inclusiveness,
            EngagementDimension.TRANSPARENCY: self._score_transparency,
            EngagementDimension.RESPONSIVENESS: self._score_responsiveness,
            EngagementDimension.ACCOUNTABILITY: self._score_accountability,
            EngagementDimension.CULTURAL_SENSITIVITY: self._score_cultural_sensitivity,
            EngagementDimension.RIGHTS_RESPECT: self._score_rights_respect,
        }
        scorer = scorers.get(dimension, self._score_default)
        raw = scorer(data)
        return self._clamp_score(raw)

    def calculate_composite(
        self,
        scores: Dict[EngagementDimension, Decimal],
    ) -> Decimal:
        """Calculate weighted composite score from dimension scores.

        Args:
            scores: Dictionary of dimension scores.

        Returns:
            Weighted composite score between 0 and 100.

        Raises:
            ValueError: If not all 6 dimensions are present.
        """
        if len(scores) < 6:
            raise ValueError("all 6 dimensions required for composite calculation")

        # Equal weights for all dimensions (1/6 each)
        total = sum(scores.values())
        composite = total / Decimal("6")
        return composite.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

    def generate_recommendations(
        self,
        scores: Dict[EngagementDimension, Decimal],
    ) -> List[str]:
        """Generate improvement recommendations based on scores.

        Args:
            scores: Dictionary of dimension scores.

        Returns:
            List of recommendation strings.

        Raises:
            ValueError: If scores dictionary is empty.
        """
        if not scores:
            raise ValueError("scores cannot be empty for recommendations")

        recommendations: List[str] = []
        for dimension, score in scores.items():
            if score < _LOW_THRESHOLD:
                dim_recs = _RECOMMENDATIONS.get(dimension, [])
                recommendations.extend(dim_recs)

        return recommendations

    # --- Private scoring methods (deterministic, zero-hallucination) ---

    def _score_inclusiveness(self, data: Dict[str, Any]) -> Decimal:
        """Score inclusiveness dimension."""
        score = Decimal("30")  # Base score

        consultations = data.get("consultations_held", 0)
        if consultations >= 15:
            score += Decimal("30")
        elif consultations >= 10:
            score += Decimal("25")
        elif consultations >= 5:
            score += Decimal("15")
        elif consultations >= 1:
            score += Decimal("5")

        groups = data.get("stakeholder_groups_represented", 0)
        if groups >= 6:
            score += Decimal("20")
        elif groups >= 3:
            score += Decimal("10")

        if data.get("vulnerable_groups_included", False):
            score += Decimal("10")
        if data.get("women_represented", False):
            score += Decimal("8")
        if data.get("youth_represented", False):
            score += Decimal("7")

        # Cultural engagement practices enhance inclusiveness
        if data.get("indigenous_protocols_followed", False):
            score += Decimal("10")
        if data.get("cultural_practices_respected", False):
            score += Decimal("5")

        return score

    def _score_transparency(self, data: Dict[str, Any]) -> Decimal:
        """Score transparency dimension."""
        score = Decimal("20")  # Base score

        if data.get("information_shared_publicly", False):
            score += Decimal("25")

        reports = data.get("reports_published", 0)
        if reports >= 4:
            score += Decimal("25")
        elif reports >= 2:
            score += Decimal("15")
        elif reports >= 1:
            score += Decimal("5")

        languages = data.get("languages_supported", 0)
        if languages >= 3:
            score += Decimal("15")
        elif languages >= 1:
            score += Decimal("10")

        # Active communication frequency demonstrates transparency commitment
        freq = data.get("communication_frequency", "")
        if freq == "weekly":
            score += Decimal("20")
        elif freq == "monthly":
            score += Decimal("10")

        # FPIC process transparency: granted FPIC implies transparent process
        fpic_status = data.get("fpic_status", "")
        if fpic_status == "granted":
            score += Decimal("15")

        # Consultations held demonstrate transparent engagement
        consultations = data.get("consultations_held", 0)
        if consultations >= 10:
            score += Decimal("15")
        elif consultations >= 5:
            score += Decimal("10")

        return score

    def _score_responsiveness(self, data: Dict[str, Any]) -> Decimal:
        """Score responsiveness dimension."""
        score = Decimal("20")  # Base score

        avg_hours = data.get("avg_grievance_response_hours", 999)
        if avg_hours <= 8:
            score += Decimal("35")
        elif avg_hours <= 24:
            score += Decimal("25")
        elif avg_hours <= 48:
            score += Decimal("15")
        elif avg_hours <= 72:
            score += Decimal("5")

        total = data.get("grievances_total", 0)
        resolved_sla = data.get("grievances_resolved_within_sla", 0)
        if total > 0:
            ratio = resolved_sla / total
            if ratio >= 0.9:
                score += Decimal("30")
            elif ratio >= 0.7:
                score += Decimal("20")
            elif ratio >= 0.5:
                score += Decimal("10")

        # Grievances resolved demonstrates responsiveness even without SLA data
        resolved = data.get("grievances_resolved", 0)
        if resolved >= 10:
            score += Decimal("25")
        elif resolved >= 5:
            score += Decimal("15")
        elif resolved >= 1:
            score += Decimal("5")

        freq = data.get("communication_frequency", "")
        if freq == "weekly":
            score += Decimal("15")
        elif freq == "monthly":
            score += Decimal("5")

        # High stakeholder satisfaction indicates responsive engagement
        satisfaction = data.get("grievance_satisfaction_avg", 0)
        if satisfaction >= 90:
            score += Decimal("15")
        elif satisfaction >= 70:
            score += Decimal("10")

        return score

    def _score_accountability(self, data: Dict[str, Any]) -> Decimal:
        """Score accountability dimension."""
        score = Decimal("20")  # Base score

        total = data.get("commitments_total", 0)
        fulfilled = data.get("commitments_fulfilled", 0)
        if total > 0:
            ratio = fulfilled / total
            if ratio >= 0.9:
                score += Decimal("40")
            elif ratio >= 0.7:
                score += Decimal("25")
            elif ratio >= 0.5:
                score += Decimal("15")
            else:
                score += Decimal("5")

        audits = data.get("independent_audits", 0)
        if audits >= 2:
            score += Decimal("20")
        elif audits >= 1:
            score += Decimal("10")

        # Stakeholder satisfaction demonstrates accountability
        satisfaction = data.get("grievance_satisfaction_avg", 0)
        if satisfaction >= 90:
            score += Decimal("30")
        elif satisfaction >= 70:
            score += Decimal("20")
        elif satisfaction >= 50:
            score += Decimal("10")

        # Resolved grievances demonstrate accountable follow-through
        resolved = data.get("grievances_resolved", 0)
        if resolved >= 5:
            score += Decimal("15")
        elif resolved >= 1:
            score += Decimal("5")

        return score

    def _score_cultural_sensitivity(self, data: Dict[str, Any]) -> Decimal:
        """Score cultural sensitivity dimension."""
        score = Decimal("20")  # Base score

        if data.get("cultural_practices_respected", False):
            score += Decimal("20")
        if data.get("local_language_used", False):
            score += Decimal("15")
        if data.get("indigenous_protocols_followed", False):
            score += Decimal("20")
        if data.get("community_elders_consulted", False):
            score += Decimal("13")
        if data.get("traditional_decision_making_respected", False):
            score += Decimal("12")

        # FPIC process demonstrates cultural engagement
        fpic_status = data.get("fpic_status", "")
        if fpic_status == "granted":
            score += Decimal("10")

        return score

    def _score_rights_respect(self, data: Dict[str, Any]) -> Decimal:
        """Score rights respect dimension."""
        score = Decimal("15")  # Base score

        fpic_obtained = data.get("fpic_obtained", False)
        fpic_status = data.get("fpic_status", "")

        if fpic_obtained or fpic_status == "granted":
            score += Decimal("35")
        elif fpic_status == "pending":
            score += Decimal("10")

        if data.get("land_rights_respected", False):
            score += Decimal("20")
        if data.get("rights_impact_assessed", False):
            score += Decimal("20")

        if data.get("fpic_required", True) and not fpic_obtained and fpic_status != "granted":
            score = min(score, Decimal("25"))

        return score

    @staticmethod
    def _score_default(data: Dict[str, Any]) -> Decimal:
        """Default scorer returning 50."""
        return Decimal("50")

    @staticmethod
    def _clamp_score(score: Decimal) -> Decimal:
        """Clamp score to [0, 100] range."""
        return max(Decimal("0"), min(Decimal("100"), score))
