# -*- coding: utf-8 -*-
"""
Prioritization Engine - AGENT-EUDR-035: Improvement Plan Creator

Prioritizes improvement actions using Eisenhower matrix classification
and multi-factor risk-based scoring. Combines urgency, importance,
compliance impact, resource efficiency, stakeholder impact, and time
sensitivity into a deterministic composite priority score.

Zero-Hallucination:
    - All scores are Decimal arithmetic with configurable weights
    - Eisenhower quadrant assignment is deterministic threshold-based
    - No LLM involvement in priority scoring

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (GL-EUDR-IPC-035)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import ImprovementPlanCreatorConfig, get_config
from .models import (
    AGENT_ID,
    ActionStatus,
    ComplianceGap,
    EisenhowerQuadrant,
    GapSeverity,
    ImprovementAction,
    RootCause,
)
from .provenance import ProvenanceTracker
from . import metrics as m

logger = logging.getLogger(__name__)

# Urgency scoring by gap severity
_SEVERITY_URGENCY: Dict[GapSeverity, Decimal] = {
    GapSeverity.CRITICAL: Decimal("95"),
    GapSeverity.HIGH: Decimal("75"),
    GapSeverity.MEDIUM: Decimal("50"),
    GapSeverity.LOW: Decimal("30"),
    GapSeverity.INFORMATIONAL: Decimal("15"),
}

# Importance scoring by gap severity
_SEVERITY_IMPORTANCE: Dict[GapSeverity, Decimal] = {
    GapSeverity.CRITICAL: Decimal("95"),
    GapSeverity.HIGH: Decimal("80"),
    GapSeverity.MEDIUM: Decimal("55"),
    GapSeverity.LOW: Decimal("35"),
    GapSeverity.INFORMATIONAL: Decimal("20"),
}

# Eisenhower quadrant thresholds
_URGENCY_THRESHOLD = Decimal("50")
_IMPORTANCE_THRESHOLD = Decimal("50")


class PrioritizationEngine:
    """Prioritizes improvement actions using Eisenhower + risk-based scoring.

    Applies a multi-factor scoring model that combines Eisenhower matrix
    (urgency vs importance) with weighted risk, compliance, resource,
    stakeholder, and time sensitivity factors for deterministic,
    reproducible priority rankings.

    Example:
        >>> engine = PrioritizationEngine()
        >>> ranked = await engine.prioritize_actions(actions, gaps)
        >>> assert ranked[0].priority_score >= ranked[-1].priority_score
    """

    def __init__(self, config: Optional[ImprovementPlanCreatorConfig] = None) -> None:
        """Initialize PrioritizationEngine.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        logger.info("PrioritizationEngine initialized")

    async def prioritize_actions(
        self,
        actions: List[ImprovementAction],
        gaps: List[ComplianceGap],
        root_causes: Optional[List[RootCause]] = None,
    ) -> List[ImprovementAction]:
        """Score and rank actions by priority.

        Args:
            actions: Actions to prioritize.
            gaps: Associated compliance gaps.
            root_causes: Optional root causes for enhanced scoring.

        Returns:
            Actions sorted by priority_score descending.
        """
        start = time.monotonic()

        # Build gap lookup
        gap_map: Dict[str, ComplianceGap] = {g.gap_id: g for g in gaps}

        # Build root cause lookup
        rc_map: Dict[str, List[RootCause]] = {}
        if root_causes:
            for rc in root_causes:
                rc_map.setdefault(rc.gap_id, []).append(rc)

        for action in actions:
            gap = gap_map.get(action.gap_id)
            self._score_action(action, gap, rc_map.get(action.gap_id, []))
            m.record_action_prioritized(action.eisenhower_quadrant.value)

        # Sort by priority_score descending
        actions.sort(key=lambda a: a.priority_score, reverse=True)

        # Provenance
        provenance_data = {
            "actions_prioritized": len(actions),
            "gaps_referenced": len(gap_map),
        }
        self._provenance.compute_hash(provenance_data)

        self._provenance.record(
            "prioritization", "rank", f"batch-{uuid.uuid4().hex[:8]}", AGENT_ID,
            metadata={"count": len(actions)},
        )

        elapsed = time.monotonic() - start
        m.observe_prioritization_duration(elapsed)

        logger.info(
            "Prioritized %d actions in %.1fms",
            len(actions), elapsed * 1000,
        )
        return actions

    def _score_action(
        self,
        action: ImprovementAction,
        gap: Optional[ComplianceGap],
        root_causes: List[RootCause],
    ) -> None:
        """Score a single action and assign Eisenhower quadrant.

        Args:
            action: Action to score (modified in place).
            gap: Associated compliance gap.
            root_causes: Associated root causes.
        """
        severity = gap.severity if gap else GapSeverity.MEDIUM

        # Urgency score
        urgency = _SEVERITY_URGENCY.get(severity, Decimal("50"))
        # Boost urgency if deadline is approaching
        if action.time_bound_deadline:
            days_remaining = (
                action.time_bound_deadline - datetime.now(timezone.utc)
            ).days
            if days_remaining <= 7:
                urgency = min(urgency + Decimal("20"), Decimal("100"))
            elif days_remaining <= 14:
                urgency = min(urgency + Decimal("10"), Decimal("100"))

        # Importance score
        importance = _SEVERITY_IMPORTANCE.get(severity, Decimal("55"))
        # Boost importance if root cause is systemic
        if any(rc.systemic for rc in root_causes):
            importance = min(importance + Decimal("15"), Decimal("100"))

        # Assign Eisenhower quadrant
        quadrant = self._classify_eisenhower(urgency, importance)

        # Multi-factor composite score
        risk_factor = (gap.severity_score * Decimal("100")) if gap else Decimal("50")
        compliance_factor = importance  # Compliance impact correlates with importance
        resource_factor = self._resource_efficiency_score(action)
        stakeholder_factor = Decimal("50")  # Default; enhanced by stakeholder engine
        time_factor = urgency  # Time sensitivity correlates with urgency

        composite = (
            self.config.risk_score_weight * risk_factor
            + self.config.compliance_impact_weight * compliance_factor
            + self.config.resource_efficiency_weight * resource_factor
            + self.config.stakeholder_impact_weight * stakeholder_factor
            + self.config.time_sensitivity_weight * time_factor
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        composite = min(max(composite, Decimal("0")), Decimal("100"))

        action.urgency_score = urgency
        action.importance_score = importance
        action.eisenhower_quadrant = quadrant
        action.priority_score = composite

    def _classify_eisenhower(
        self, urgency: Decimal, importance: Decimal
    ) -> EisenhowerQuadrant:
        """Classify action into Eisenhower matrix quadrant.

        Args:
            urgency: Urgency score (0-100).
            importance: Importance score (0-100).

        Returns:
            EisenhowerQuadrant classification.
        """
        urgent = urgency >= _URGENCY_THRESHOLD
        important = importance >= _IMPORTANCE_THRESHOLD

        if urgent and important:
            return EisenhowerQuadrant.DO_FIRST
        elif not urgent and important:
            return EisenhowerQuadrant.SCHEDULE
        elif urgent and not important:
            return EisenhowerQuadrant.DELEGATE
        else:
            return EisenhowerQuadrant.ELIMINATE

    def _resource_efficiency_score(self, action: ImprovementAction) -> Decimal:
        """Calculate resource efficiency score for an action.

        Lower estimated effort and cost = higher resource efficiency.

        Args:
            action: Action to evaluate.

        Returns:
            Resource efficiency score (0-100).
        """
        # Normalize effort (0-200 hours range)
        effort_norm = min(float(action.estimated_effort_hours) / 200.0, 1.0)
        # Invert: lower effort = higher score
        effort_score = Decimal(str(1.0 - effort_norm)) * Decimal("100")

        # Normalize cost (0-50000 range)
        cost_norm = min(float(action.estimated_cost) / 50000.0, 1.0)
        cost_score = Decimal(str(1.0 - cost_norm)) * Decimal("100")

        # Average of effort and cost efficiency
        result = ((effort_score + cost_score) / Decimal("2")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return min(max(result, Decimal("0")), Decimal("100"))

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "PrioritizationEngine",
            "status": "healthy",
        }
