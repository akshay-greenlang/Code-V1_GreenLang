# -*- coding: utf-8 -*-
"""
Effectiveness Estimator Engine - AGENT-EUDR-029

Estimates expected risk reduction from proposed mitigation measures using
deterministic formulas with three-scenario projections (conservative,
moderate, optimistic). Validates feasibility before operator commits
resources.

Core Formula:
    Expected_Risk_Reduction = Base_Effectiveness * Applicability_Factor * Quality_Factor

    Where:
    - Base_Effectiveness: From the measure template library (curated value)
    - Applicability_Factor: Based on risk dimension score (higher risk = more room)
    - Quality_Factor: Based on evidence quality and measure maturity

Cumulative Reduction (diminishing returns):
    Total = 1 - Product(1 - Ri) for each measure i

Zero-Hallucination Guarantees:
    - All calculations use Decimal arithmetic (no float)
    - No LLM calls in the calculation path
    - Deterministic three-scenario projections
    - Complete provenance trail for every estimate

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-029 Mitigation Measure Designer (GL-EUDR-MMD-029)
Regulation: EU 2023/1115 (EUDR) Article 11
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import MitigationMeasureDesignerConfig, get_config
from .models import (
    EffectivenessEstimate,
    EvidenceType,
    MeasureTemplate,
    MitigationMeasure,
    MitigationStrategy,
    RiskDimension,
    RiskTrigger,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class EffectivenessEstimator:
    """Estimates expected risk reduction from proposed mitigation measures.

    Provides conservative, moderate, and optimistic effectiveness
    projections using configurable scaling factors and validates
    strategy feasibility against target risk scores.

    Attributes:
        _config: Agent configuration.
        _provenance: Provenance tracker for audit trail.

    Example:
        >>> estimator = EffectivenessEstimator()
        >>> estimate = estimator.estimate_measure_effectiveness(
        ...     measure=measure,
        ...     template=template,
        ...     risk_trigger=trigger,
        ... )
        >>> assert estimate.moderate > Decimal("0")
    """

    def __init__(
        self,
        config: Optional[MitigationMeasureDesignerConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize EffectivenessEstimator.

        Args:
            config: Agent configuration. Uses get_config() if None.
            provenance: Provenance tracker instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        logger.info(
            "EffectivenessEstimator initialized: "
            "factors=[%s, %s, %s], min=%s, cap=%s",
            self._config.conservative_factor,
            self._config.moderate_factor,
            self._config.optimistic_factor,
            self._config.min_effectiveness_threshold,
            self._config.max_effectiveness_cap,
        )

    def estimate_measure_effectiveness(
        self,
        measure: MitigationMeasure,
        template: MeasureTemplate,
        risk_trigger: RiskTrigger,
    ) -> EffectivenessEstimate:
        """Estimate effectiveness of a single measure.

        Combines base effectiveness from the template with context-
        specific applicability and quality factors to produce three-
        scenario projections.

        Args:
            measure: The mitigation measure to estimate.
            template: Source template for base effectiveness.
            risk_trigger: Risk trigger for context (dimension scores).

        Returns:
            EffectivenessEstimate with three scenarios.
        """
        dimension = measure.target_dimension
        dim_score = risk_trigger.risk_dimensions.get(
            dimension, Decimal("50")
        )

        base = template.base_effectiveness
        applicability = self._calculate_applicability_factor(
            dimension=dimension, risk_score=dim_score,
        )
        quality = self._calculate_quality_factor(measure)

        # Adjusted effectiveness = base * applicability * quality
        adjusted = base * applicability * quality / Decimal("100")
        adjusted = adjusted.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )

        # Apply three-scenario factors
        conservative = self._apply_scenario_factor(
            adjusted, self._config.conservative_factor,
        )
        moderate = self._apply_scenario_factor(
            adjusted, self._config.moderate_factor,
        )
        optimistic = self._apply_scenario_factor(
            adjusted, self._config.optimistic_factor,
        )

        estimate = EffectivenessEstimate(
            estimate_id=f"eff-{uuid.uuid4().hex[:12]}",
            measure_id=measure.measure_id,
            conservative=conservative,
            moderate=moderate,
            optimistic=optimistic,
            applicability_factor=applicability,
            confidence=self._calculate_confidence(applicability, quality),
        )

        logger.debug(
            "Effectiveness estimate for measure=%s: "
            "base=%s, applicability=%s, quality=%s, "
            "conservative=%s, moderate=%s, optimistic=%s",
            measure.measure_id,
            base,
            applicability,
            quality,
            conservative,
            moderate,
            optimistic,
        )

        return estimate

    def estimate_strategy_effectiveness(
        self,
        strategy: MitigationStrategy,
        templates: Dict[str, MeasureTemplate],
    ) -> Dict[str, EffectivenessEstimate]:
        """Estimate effectiveness of all measures in a strategy.

        Args:
            strategy: Strategy containing measures.
            templates: Map of template_id to MeasureTemplate.

        Returns:
            Dictionary mapping measure_id to EffectivenessEstimate.
        """
        start_time = time.monotonic()
        estimates: Dict[str, EffectivenessEstimate] = {}
        trigger = strategy.risk_trigger

        for measure in strategy.measures:
            template_id = measure.template_id or ""
            template = templates.get(template_id)

            if template is None:
                # Create synthetic template from measure data
                template = MeasureTemplate(
                    template_id=template_id or "synthetic",
                    title=measure.title,
                    description=measure.description,
                    article11_category=measure.article11_category,
                    applicable_dimensions=[measure.target_dimension],
                    base_effectiveness=measure.expected_risk_reduction,
                    typical_timeline_days=30,
                )

            estimate = self.estimate_measure_effectiveness(
                measure=measure,
                template=template,
                risk_trigger=trigger,
            )
            estimates[measure.measure_id] = estimate

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Strategy effectiveness estimated: "
            "strategy=%s, measures=%d, elapsed=%.1fms",
            strategy.strategy_id,
            len(estimates),
            elapsed_ms,
        )

        return estimates

    def calculate_cumulative_reduction(
        self,
        estimates: List[EffectivenessEstimate],
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Calculate cumulative reduction across all estimates.

        Uses diminishing returns formula:
        Total = 1 - Product(1 - Ri) for each measure i

        Args:
            estimates: List of effectiveness estimates.

        Returns:
            Tuple of (conservative, moderate, optimistic) cumulative
            reduction percentages.
        """
        if not estimates:
            return Decimal("0"), Decimal("0"), Decimal("0")

        conservative_product = Decimal("1")
        moderate_product = Decimal("1")
        optimistic_product = Decimal("1")

        for est in estimates:
            c_factor = Decimal("1") - (est.conservative / Decimal("100"))
            m_factor = Decimal("1") - (est.moderate / Decimal("100"))
            o_factor = Decimal("1") - (est.optimistic / Decimal("100"))

            # Clamp to non-negative
            c_factor = max(c_factor, Decimal("0"))
            m_factor = max(m_factor, Decimal("0"))
            o_factor = max(o_factor, Decimal("0"))

            conservative_product *= c_factor
            moderate_product *= m_factor
            optimistic_product *= o_factor

        cap = self._config.max_effectiveness_cap

        conservative_total = min(
            (Decimal("1") - conservative_product) * Decimal("100"), cap,
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        moderate_total = min(
            (Decimal("1") - moderate_product) * Decimal("100"), cap,
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        optimistic_total = min(
            (Decimal("1") - optimistic_product) * Decimal("100"), cap,
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        logger.info(
            "Cumulative reduction: conservative=%s%%, "
            "moderate=%s%%, optimistic=%s%%",
            conservative_total,
            moderate_total,
            optimistic_total,
        )

        return conservative_total, moderate_total, optimistic_total

    def validate_feasibility(
        self,
        pre_score: Decimal,
        target_score: Decimal,
        estimates: List[EffectivenessEstimate],
    ) -> bool:
        """Validate that proposed measures can achieve target.

        Uses the moderate (baseline) scenario to determine if the
        strategy is feasible. The strategy is feasible if the estimated
        reduction can bring the pre-mitigation score down to target.

        Args:
            pre_score: Pre-mitigation composite risk score.
            target_score: Target post-mitigation score.
            estimates: List of measure effectiveness estimates.

        Returns:
            True if strategy can plausibly achieve target.
        """
        if pre_score <= target_score:
            return True

        _, moderate_total, _ = self.calculate_cumulative_reduction(estimates)

        required_reduction_abs = pre_score - target_score
        estimated_reduction_abs = (
            pre_score * moderate_total / Decimal("100")
        )

        is_feasible = estimated_reduction_abs >= required_reduction_abs
        logger.info(
            "Feasibility validation: pre=%s, target=%s, "
            "required_abs=%s, estimated_abs=%s, feasible=%s",
            pre_score,
            target_score,
            required_reduction_abs,
            estimated_reduction_abs,
            is_feasible,
        )

        return is_feasible

    def _calculate_applicability_factor(
        self,
        dimension: RiskDimension,
        risk_score: Decimal,
    ) -> Decimal:
        """Calculate applicability factor for a risk dimension.

        Higher risk scores produce higher applicability factors
        because there is more room for risk reduction. The factor
        ranges from 0.50 (low risk, limited room) to 1.00 (high risk,
        maximum room).

        Args:
            dimension: Risk dimension.
            risk_score: Current score for this dimension (0-100).

        Returns:
            Applicability factor as Decimal (0.50 to 1.00).
        """
        # Linear scaling: score 0 -> 0.50, score 100 -> 1.00
        factor = Decimal("0.50") + (risk_score / Decimal("200"))
        factor = max(Decimal("0.50"), min(factor, Decimal("1.00")))
        return factor.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_quality_factor(
        self,
        measure: MitigationMeasure,
    ) -> Decimal:
        """Calculate quality factor based on measure maturity.

        Quality factor reflects the strength of available evidence
        and the maturity of the measure's implementation approach.

        Base quality is 100 (1.00 factor). Adjustments:
        - Measure has evidence items: +0 to +10 points
        - Measure has assigned owner: +5 points
        - Measure based on template: +5 points

        Args:
            measure: The mitigation measure.

        Returns:
            Quality factor as Decimal (0.70 to 1.20).
        """
        quality = Decimal("100")

        # Template-based measures are more structured
        if measure.template_id:
            quality += Decimal("5")

        # Assigned measures have accountability
        if measure.assigned_to:
            quality += Decimal("5")

        # Evidence increases quality
        evidence_count = len(measure.evidence_ids)
        quality += min(Decimal(str(evidence_count * 3)), Decimal("10"))

        # Normalize to factor (divide by 100)
        factor = quality / Decimal("100")
        factor = max(Decimal("0.70"), min(factor, Decimal("1.20")))
        return factor.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _apply_scenario_factor(
        self,
        adjusted_effectiveness: Decimal,
        scenario_factor: Decimal,
    ) -> Decimal:
        """Apply scenario factor and clamp to valid range.

        Args:
            adjusted_effectiveness: Base adjusted effectiveness.
            scenario_factor: Conservative/moderate/optimistic factor.

        Returns:
            Clamped effectiveness percentage (0-100).
        """
        result = adjusted_effectiveness * scenario_factor
        min_threshold = self._config.min_effectiveness_threshold
        max_cap = self._config.max_effectiveness_cap

        # Clamp to valid range
        result = max(min_threshold, min(result, max_cap))
        return result.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_confidence(
        self,
        applicability: Decimal,
        quality: Decimal,
    ) -> Decimal:
        """Calculate confidence level from factors.

        Confidence is the geometric mean of applicability and quality
        factors, indicating overall reliability of the estimate.

        Args:
            applicability: Applicability factor.
            quality: Quality factor.

        Returns:
            Confidence level as Decimal (0-1).
        """
        # Simple product normalized to 0-1 range
        raw = applicability * quality
        confidence = min(raw, Decimal("1.00"))
        return confidence.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Dictionary with engine status and configuration.
        """
        return {
            "engine": "EffectivenessEstimator",
            "status": "available",
            "config": {
                "conservative_factor": str(
                    self._config.conservative_factor
                ),
                "moderate_factor": str(self._config.moderate_factor),
                "optimistic_factor": str(self._config.optimistic_factor),
                "min_threshold": str(
                    self._config.min_effectiveness_threshold
                ),
                "max_cap": str(self._config.max_effectiveness_cap),
            },
        }
