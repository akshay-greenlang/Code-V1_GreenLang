# -*- coding: utf-8 -*-
"""
AGENT-EUDR-028: Risk Assessment Engine - Risk Classification Engine

Classifies composite risk scores into 5-tier risk levels (NEGLIGIBLE, LOW,
STANDARD, HIGH, CRITICAL) with hysteresis to prevent oscillation at
threshold boundaries. Integrates Article 10 criteria results for
escalation and de-escalation decisions, and evaluates eligibility for
simplified due diligence under Article 13.

The hysteresis buffer (default 3 points) prevents risk level changes
when scores fluctuate near thresholds. For example, if a score drops
from 62 (HIGH) to 59 (below the 60 STANDARD/HIGH boundary), it remains
HIGH because 59 > (60 - 3 = 57). The score must drop to 57 or below to
trigger a de-escalation to STANDARD.

Production infrastructure includes:
    - 5-tier threshold-based classification with Decimal precision
    - Hysteresis buffer for oscillation prevention
    - Article 10 criteria-driven escalation (3+ CONCERN -> escalate)
    - Article 10 criteria-driven de-escalation (all PASS -> eligible)
    - Simplified due diligence eligibility check (Article 13)
    - SHA-256 provenance hash on classification decisions
    - Prometheus metrics integration

Zero-Hallucination Guarantees:
    - All classifications use deterministic threshold comparisons
    - Hysteresis applied via simple arithmetic (threshold +/- buffer)
    - No LLM involvement in classification or escalation decisions
    - All provenance hashes computed from canonical JSON

Regulatory References:
    - EUDR Article 10(2): Risk assessment outcome categories
    - EUDR Article 13: Simplified due diligence for low-risk sourcing
    - EUDR Article 29: Country benchmark integration
    - EUDR Article 31: 5-year retention of classification decisions

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-028 (Engine 5: Risk Classification Engine)
Agent ID: GL-EUDR-RAE-028
Status: Production Ready
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
    get_config,
)
from greenlang.agents.eudr.risk_assessment_engine.models import (
    Article10CriteriaResult,
    CompositeRiskScore,
    CountryBenchmark,
    CountryBenchmarkLevel,
    CriterionResult,
    RiskLevel,
    SimplifiedDDEligibility,
    RiskDimension,
)
from greenlang.agents.eudr.risk_assessment_engine.provenance import ProvenanceTracker
from greenlang.agents.eudr.risk_assessment_engine.metrics import (
    record_risk_classification,
)

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------

_CONCERN_ESCALATION_THRESHOLD = 3  # 3+ CONCERN criteria -> escalate
_SCORE_PRECISION = Decimal("0.01")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash of data.

    Args:
        data: Any JSON-serializable object.

    Returns:
        64-character lowercase hex SHA-256 hash string.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Risk level ordering for comparison
# ---------------------------------------------------------------------------

_RISK_LEVEL_ORDER: Dict[RiskLevel, int] = {
    RiskLevel.NEGLIGIBLE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.STANDARD: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}


def _risk_level_above(a: RiskLevel, b: RiskLevel) -> bool:
    """Check if risk level a is strictly above b.

    Args:
        a: First risk level.
        b: Second risk level.

    Returns:
        True if a > b in severity ordering.
    """
    return _RISK_LEVEL_ORDER.get(a, 0) > _RISK_LEVEL_ORDER.get(b, 0)


def _risk_level_below(a: RiskLevel, b: RiskLevel) -> bool:
    """Check if risk level a is strictly below b.

    Args:
        a: First risk level.
        b: Second risk level.

    Returns:
        True if a < b in severity ordering.
    """
    return _RISK_LEVEL_ORDER.get(a, 0) < _RISK_LEVEL_ORDER.get(b, 0)


def _next_level_up(level: RiskLevel) -> RiskLevel:
    """Return the next higher risk level.

    Args:
        level: Current risk level.

    Returns:
        Next higher RiskLevel, or CRITICAL if already at max.
    """
    order = _RISK_LEVEL_ORDER.get(level, 0)
    for rl, idx in _RISK_LEVEL_ORDER.items():
        if idx == order + 1:
            return rl
    return RiskLevel.CRITICAL


def _next_level_down(level: RiskLevel) -> RiskLevel:
    """Return the next lower risk level.

    Args:
        level: Current risk level.

    Returns:
        Next lower RiskLevel, or NEGLIGIBLE if already at min.
    """
    order = _RISK_LEVEL_ORDER.get(level, 0)
    for rl, idx in _RISK_LEVEL_ORDER.items():
        if idx == order - 1:
            return rl
    return RiskLevel.NEGLIGIBLE


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class RiskClassificationEngine:
    """Engine for classifying composite risk scores into 5-tier risk levels.

    Applies threshold-based classification with hysteresis to prevent
    oscillation at boundaries. Integrates Article 10 criteria results
    to escalate or de-escalate risk levels beyond pure score-based
    classification. Evaluates simplified due diligence eligibility.

    Args:
        config: Agent configuration (uses singleton if None).

    Example:
        >>> engine = RiskClassificationEngine()
        >>> level = engine.classify_risk(Decimal("45"))
        >>> assert level == RiskLevel.STANDARD
        >>> level_with_hysteresis = engine.classify_risk(
        ...     Decimal("59"), previous_level=RiskLevel.HIGH
        ... )
        >>> assert level_with_hysteresis == RiskLevel.HIGH  # hysteresis
    """

    def __init__(self, config: Optional[RiskAssessmentEngineConfig] = None) -> None:
        """Initialize RiskClassificationEngine.

        Args:
            config: Agent configuration (uses singleton if None).
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._classification_count: int = 0
        self._hysteresis_applied_count: int = 0
        self._escalation_count: int = 0
        self._deescalation_count: int = 0

        # Load thresholds from config
        thresholds = self._config.risk_thresholds
        self._negligible_max = Decimal(str(thresholds.get("negligible", 15)))
        self._low_max = Decimal(str(thresholds.get("low", 30)))
        self._standard_max = Decimal(str(thresholds.get("standard", 60)))
        self._high_max = Decimal(str(thresholds.get("high", 80)))
        self._hysteresis = Decimal(str(self._config.hysteresis_buffer))

        logger.info(
            "RiskClassificationEngine initialized "
            "(thresholds: negligible<=%s, low<=%s, standard<=%s, "
            "high<=%s, hysteresis=%s)",
            self._negligible_max,
            self._low_max,
            self._standard_max,
            self._high_max,
            self._hysteresis,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_risk(
        self,
        score: Decimal,
        previous_level: Optional[RiskLevel] = None,
    ) -> RiskLevel:
        """Classify a composite risk score into a 5-tier risk level.

        Thresholds:
            0-15:   NEGLIGIBLE
            16-30:  LOW
            31-60:  STANDARD
            61-80:  HIGH
            81-100: CRITICAL

        When previous_level is provided, hysteresis is applied: the score
        must cross the threshold by at least the hysteresis buffer to
        trigger a level change.

        Args:
            score: Composite risk score (0-100).
            previous_level: Previous risk level for hysteresis.

        Returns:
            Classified RiskLevel.
        """
        self._classification_count += 1

        # Base classification without hysteresis
        base_level = self._classify_raw(score)

        if previous_level is None:
            record_risk_classification(base_level.value)
            return base_level

        # Apply hysteresis
        classified = self._apply_hysteresis(score, base_level, previous_level)

        if classified != base_level:
            self._hysteresis_applied_count += 1
            logger.debug(
                "Hysteresis applied: score=%s would be %s, kept at %s "
                "(previous=%s, buffer=%s)",
                score,
                base_level.value,
                classified.value,
                previous_level.value,
                self._hysteresis,
            )

        record_risk_classification(classified.value)
        return classified

    def classify_with_article10(
        self,
        composite: CompositeRiskScore,
        article10: Article10CriteriaResult,
    ) -> RiskLevel:
        """Classify risk with Article 10 criteria-based adjustments.

        If the composite classification is LOW but Article 10 has >= 3
        CONCERN criteria, the level is escalated to STANDARD.

        If the composite classification is STANDARD but Article 10 has
        all PASS criteria, the level may be de-escalated to LOW.

        Args:
            composite: Composite risk score.
            article10: Article 10 criteria evaluation result.

        Returns:
            Adjusted RiskLevel accounting for criteria.
        """
        base_level = composite.risk_level

        # Escalation: LOW + many concerns -> STANDARD
        if base_level == RiskLevel.LOW:
            if article10.concern_count >= _CONCERN_ESCALATION_THRESHOLD:
                self._escalation_count += 1
                escalated = RiskLevel.STANDARD
                logger.info(
                    "Risk escalated %s -> %s: %d CONCERN criteria (threshold=%d)",
                    base_level.value,
                    escalated.value,
                    article10.concern_count,
                    _CONCERN_ESCALATION_THRESHOLD,
                )
                return escalated

        # Escalation: NEGLIGIBLE + any concerns -> LOW
        if base_level == RiskLevel.NEGLIGIBLE:
            if article10.concern_count >= 1:
                self._escalation_count += 1
                escalated = RiskLevel.LOW
                logger.info(
                    "Risk escalated %s -> %s: %d CONCERN criteria detected",
                    base_level.value,
                    escalated.value,
                    article10.concern_count,
                )
                return escalated

        # Escalation: any FAIL criteria -> at least STANDARD
        if article10.fail_count > 0 and _risk_level_below(
            base_level, RiskLevel.STANDARD
        ):
            self._escalation_count += 1
            escalated = RiskLevel.STANDARD
            logger.info(
                "Risk escalated %s -> %s: %d FAIL criteria detected",
                base_level.value,
                escalated.value,
                article10.fail_count,
            )
            return escalated

        # De-escalation: STANDARD + all PASS -> LOW (only if score < 40)
        if base_level == RiskLevel.STANDARD:
            all_pass = (
                article10.pass_count == article10.total_evaluated
                and article10.total_evaluated > 0
            )
            if all_pass and composite.overall_score < Decimal("40"):
                self._deescalation_count += 1
                deescalated = RiskLevel.LOW
                logger.info(
                    "Risk de-escalated %s -> %s: all %d criteria PASS, "
                    "score=%s < 40",
                    base_level.value,
                    deescalated.value,
                    article10.pass_count,
                    composite.overall_score,
                )
                return deescalated

        return base_level

    def check_simplified_dd_eligibility(
        self,
        composite: CompositeRiskScore,
        benchmarks: List[CountryBenchmark],
    ) -> SimplifiedDDEligibility:
        """Check eligibility for simplified due diligence (Article 13).

        Eligible if all three conditions are met:
            1. All sourcing countries are classified as LOW risk
            2. Composite risk score is below 30
            3. No deforestation alerts (deforestation dimension score < 20)

        Args:
            composite: Composite risk score.
            benchmarks: Country benchmarks for sourcing countries.

        Returns:
            SimplifiedDDEligibility with eligibility status and reasons.
        """
        simplified_config = self._config.simplified_dd
        max_score = Decimal(str(simplified_config.get("max_score", 30)))
        require_all_low = simplified_config.get("require_all_low", True)

        reasons: List[str] = []
        eligible = True

        # Condition 1: All countries LOW
        if require_all_low:
            non_low = [
                b for b in benchmarks
                if b.level != CountryBenchmarkLevel.LOW
            ]
            if non_low:
                eligible = False
                non_low_codes = [b.country_code for b in non_low]
                reasons.append(
                    f"Non-LOW countries present: {', '.join(non_low_codes)}"
                )
            else:
                reasons.append("All sourcing countries are LOW risk")

        # Condition 2: Composite score below threshold
        if composite.overall_score > max_score:
            eligible = False
            reasons.append(
                f"Composite score {composite.overall_score} exceeds "
                f"max threshold {max_score}"
            )
        else:
            reasons.append(
                f"Composite score {composite.overall_score} within "
                f"threshold {max_score}"
            )

        # Condition 3: No deforestation alerts
        deforestation_scores = [
            ds for ds in composite.dimension_scores
            if ds.dimension == RiskDimension.DEFORESTATION
        ]
        if deforestation_scores:
            deforestation_score = deforestation_scores[0].raw_score
            if deforestation_score > Decimal("20"):
                eligible = False
                reasons.append(
                    f"Deforestation risk score {deforestation_score} > 20"
                )
            else:
                reasons.append(
                    f"Deforestation risk score {deforestation_score} acceptable"
                )
        else:
            reasons.append("No deforestation dimension data (acceptable)")

        provenance_hash = _compute_hash({
            "eligible": eligible,
            "composite_score": str(composite.overall_score),
            "benchmark_count": len(benchmarks),
            "reasons": reasons,
        })

        logger.info(
            "Simplified DD eligibility: %s (score=%s, countries=%d)",
            eligible,
            composite.overall_score,
            len(benchmarks),
        )

        return SimplifiedDDEligibility(
            eligible=eligible,
            reasons=reasons,
            composite_score=composite.overall_score,
            all_countries_low=all(
                b.level == CountryBenchmarkLevel.LOW for b in benchmarks
            ) if benchmarks else False,
            evaluated_at=_utcnow(),
            provenance_hash=provenance_hash,
        )

    def get_classification_stats(self) -> Dict[str, Any]:
        """Return risk classification engine statistics.

        Returns:
            Dict with total_classifications, hysteresis_applied,
            escalations, deescalations, and thresholds keys.
        """
        return {
            "total_classifications": self._classification_count,
            "hysteresis_applied": self._hysteresis_applied_count,
            "escalations": self._escalation_count,
            "deescalations": self._deescalation_count,
            "thresholds": {
                "negligible": float(self._negligible_max),
                "low": float(self._low_max),
                "standard": float(self._standard_max),
                "high": float(self._high_max),
            },
            "hysteresis_buffer": float(self._hysteresis),
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _classify_raw(self, score: Decimal) -> RiskLevel:
        """Classify score into risk level without hysteresis.

        Args:
            score: Composite risk score (0-100).

        Returns:
            RiskLevel based on pure threshold comparison.
        """
        if score <= self._negligible_max:
            return RiskLevel.NEGLIGIBLE
        elif score <= self._low_max:
            return RiskLevel.LOW
        elif score <= self._standard_max:
            return RiskLevel.STANDARD
        elif score <= self._high_max:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _apply_hysteresis(
        self,
        score: Decimal,
        new_level: RiskLevel,
        previous_level: RiskLevel,
    ) -> RiskLevel:
        """Apply hysteresis buffer to prevent oscillation.

        When the score moves across a threshold but remains within the
        hysteresis buffer zone, the previous level is maintained. This
        prevents rapid oscillation for scores near boundaries.

        Args:
            score: Current composite risk score.
            new_level: Level from raw classification.
            previous_level: Previous classification level.

        Returns:
            Final RiskLevel after hysteresis application.
        """
        if new_level == previous_level:
            return new_level

        # Get the boundary between previous and new levels
        boundaries = self._get_level_boundaries(previous_level)
        lower_bound, upper_bound = boundaries

        # Downgrade: score dropped below lower boundary
        if _risk_level_below(new_level, previous_level):
            hysteresis_boundary = lower_bound - self._hysteresis
            if score > hysteresis_boundary:
                return previous_level  # Keep previous (within buffer)
            return new_level

        # Upgrade: score rose above upper boundary
        if _risk_level_above(new_level, previous_level):
            hysteresis_boundary = upper_bound + self._hysteresis
            if score < hysteresis_boundary:
                return previous_level  # Keep previous (within buffer)
            return new_level

        return new_level

    def _get_level_boundaries(
        self,
        level: RiskLevel,
    ) -> tuple:
        """Get the score boundaries for a risk level.

        Args:
            level: Risk level.

        Returns:
            Tuple of (lower_bound, upper_bound) Decimal values.
        """
        boundaries = {
            RiskLevel.NEGLIGIBLE: (Decimal("0"), self._negligible_max),
            RiskLevel.LOW: (self._negligible_max + Decimal("1"), self._low_max),
            RiskLevel.STANDARD: (self._low_max + Decimal("1"), self._standard_max),
            RiskLevel.HIGH: (self._standard_max + Decimal("1"), self._high_max),
            RiskLevel.CRITICAL: (self._high_max + Decimal("1"), Decimal("100")),
        }
        return boundaries.get(level, (Decimal("0"), Decimal("100")))
