# -*- coding: utf-8 -*-
"""
Risk Scoring Engine - AGENT-EUDR-032

Predictive risk analytics across operator, supplier, commodity, and region
scopes with multi-factor weighted scoring, historical trend analysis, and
confidence-rated predictions.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-032 (GL-EUDR-GMM-032)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import GrievanceMechanismManagerConfig, get_config
from .models import (
    AGENT_ID,
    SEVERITY_SCORES,
    RiskLevel,
    RiskScope,
    RiskScoreRecord,
    ScoreFactor,
    TrendDirection,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class RiskScoringEngine:
    """Predictive grievance risk scoring engine.

    Example:
        >>> engine = RiskScoringEngine()
        >>> score = await engine.compute_risk_score(
        ...     operator_id="OP-001", scope="operator",
        ...     scope_identifier="OP-001",
        ...     grievances=[{"severity": "high", "status": "resolved"}],
        ... )
        >>> assert 0 <= score.risk_score <= 100
    """

    def __init__(
        self, config: Optional[GrievanceMechanismManagerConfig] = None,
    ) -> None:
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._scores: Dict[str, RiskScoreRecord] = {}
        logger.info("RiskScoringEngine initialized")

    async def compute_risk_score(
        self,
        operator_id: str,
        scope: str,
        scope_identifier: str,
        grievances: List[Dict[str, Any]],
    ) -> RiskScoreRecord:
        """Compute risk score for an entity."""
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        risk_score_id = str(uuid.uuid4())

        try:
            risk_scope = RiskScope(scope)
        except ValueError:
            risk_scope = RiskScope.OPERATOR

        total = len(grievances)
        weights = self.config.get_risk_weights()

        # Factor 1: Frequency
        freq_normalized = min(Decimal("100"), Decimal(str(total)) * Decimal("10"))
        freq_factor = ScoreFactor(
            factor_name="frequency",
            weight=weights["frequency"],
            raw_value=Decimal(str(total)),
            weighted_value=freq_normalized * weights["frequency"],
        )

        # Factor 2: Average severity
        severity_sum = sum(
            SEVERITY_SCORES.get(g.get("severity", "medium"), 50)
            for g in grievances
        )
        avg_severity = Decimal(str(severity_sum / max(total, 1)))
        severity_factor = ScoreFactor(
            factor_name="severity",
            weight=weights["severity"],
            raw_value=avg_severity,
            weighted_value=avg_severity * weights["severity"],
        )

        # Factor 3: Resolution time trend
        resolved = [g for g in grievances if g.get("status") == "resolved"]
        unresolved = [g for g in grievances if g.get("status") != "resolved"]
        if total == 0:
            resolution_score = Decimal("0")  # no grievances = no risk
        elif len(resolved) > len(unresolved):
            resolution_score = Decimal("30")  # good
        elif len(unresolved) > len(resolved):
            resolution_score = Decimal("80")  # bad
        else:
            resolution_score = Decimal("50")  # neutral default

        resolution_factor = ScoreFactor(
            factor_name="resolution",
            weight=weights["resolution"],
            raw_value=resolution_score,
            weighted_value=resolution_score * weights["resolution"],
        )

        # Factor 4: Escalation rate
        escalated = sum(1 for g in grievances if g.get("status") in ("appealed", "escalated"))
        escalation_pct = Decimal(str(round(escalated / max(total, 1) * 100, 2)))
        escalation_factor = ScoreFactor(
            factor_name="escalation",
            weight=weights["escalation"],
            raw_value=escalation_pct,
            weighted_value=escalation_pct * weights["escalation"],
        )

        # Factor 5: Unresolved count
        unresolved_norm = min(Decimal("100"), Decimal(str(len(unresolved))) * Decimal("15"))
        unresolved_factor = ScoreFactor(
            factor_name="unresolved",
            weight=weights["unresolved"],
            raw_value=Decimal(str(len(unresolved))),
            weighted_value=unresolved_norm * weights["unresolved"],
        )

        # Composite score
        factors = [freq_factor, severity_factor, resolution_factor, escalation_factor, unresolved_factor]
        composite = sum(f.weighted_value for f in factors)
        composite = min(Decimal("100"), max(Decimal("0"), composite))

        risk_level_str = self.config.get_risk_level(composite)
        try:
            risk_level = RiskLevel(risk_level_str)
        except ValueError:
            risk_level = RiskLevel.LOW

        # Determine trend
        trend = TrendDirection.STABLE
        if len(unresolved) > total * 0.6:
            trend = TrendDirection.WORSENING
        elif len(resolved) > total * 0.8:
            trend = TrendDirection.IMPROVING

        # Confidence based on data volume
        confidence = min(Decimal("95"), Decimal(str(total)) * Decimal("5") + Decimal("20"))

        record = RiskScoreRecord(
            risk_score_id=risk_score_id,
            operator_id=operator_id,
            scope=risk_scope,
            scope_identifier=scope_identifier,
            risk_score=composite.quantize(Decimal("0.01")),
            risk_level=risk_level,
            grievance_frequency=total,
            average_severity=avg_severity.quantize(Decimal("0.01")),
            resolution_time_trend=trend,
            unresolved_count=len(unresolved),
            escalation_rate=escalation_pct.quantize(Decimal("0.01")),
            prediction_confidence=confidence.quantize(Decimal("0.01")),
            score_factors=factors,
            created_at=now,
        )

        record.provenance_hash = self._provenance.compute_hash({
            "risk_score_id": risk_score_id,
            "scope": risk_scope.value,
            "score": str(composite),
            "created_at": now.isoformat(),
        })

        self._scores[risk_score_id] = record

        self._provenance.record(
            entity_type="risk_score",
            action="score",
            entity_id=risk_score_id,
            actor=AGENT_ID,
            metadata={
                "scope": risk_scope.value,
                "score": str(composite),
                "level": risk_level.value,
            },
        )

        elapsed = time.monotonic() - start_time
        logger.info(
            "Risk score %s: scope=%s, score=%s, level=%s (%.3fs)",
            risk_score_id, risk_scope.value, composite, risk_level.value, elapsed,
        )

        return record

    async def get_risk_score(self, risk_score_id: str) -> Optional[RiskScoreRecord]:
        """Retrieve a risk score by ID."""
        return self._scores.get(risk_score_id)

    async def list_risk_scores(
        self,
        operator_id: Optional[str] = None,
        scope: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> List[RiskScoreRecord]:
        """List risk scores with optional filters."""
        results = list(self._scores.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        if scope:
            results = [r for r in results if r.scope.value == scope]
        if risk_level:
            results = [r for r in results if r.risk_level.value == risk_level]
        return results

    async def health_check(self) -> Dict[str, Any]:
        high_count = sum(
            1 for s in self._scores.values()
            if s.risk_level.value in ("high", "critical")
        )
        return {
            "engine": "RiskScoringEngine",
            "status": "healthy",
            "score_count": len(self._scores),
            "high_risk_count": high_count,
        }
