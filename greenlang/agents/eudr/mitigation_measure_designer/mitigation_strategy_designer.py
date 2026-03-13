# -*- coding: utf-8 -*-
"""
Mitigation Strategy Designer Engine - AGENT-EUDR-029

Core engine that analyzes risk assessment decomposition from EUDR-028
and designs targeted mitigation plans. Decomposes composite risk scores
into dimensional contributors, ranks by impact, selects optimal
mitigation measures from the template library, estimates cumulative
effectiveness, and assembles feasible strategies.

Zero-Hallucination Guarantees:
    - All numeric calculations use Decimal arithmetic
    - No LLM calls in the calculation path
    - Dimension-to-measure mapping via deterministic decision tree
    - Cumulative reduction uses diminishing-returns formula
    - Complete provenance trail for every strategy design

Algorithm:
    1. Decompose risk into dimensions (identify those above threshold)
    2. Rank dimensions by score * weight (highest impact first)
    3. For each elevated dimension, select best-fit templates
    4. Estimate cumulative effectiveness with diminishing returns
    5. Verify strategy can plausibly reduce risk to target
    6. Assign priorities based on risk impact and urgency
    7. Calculate estimated timeline considering parallel execution

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-029 Mitigation Measure Designer (GL-EUDR-MMD-029)
Regulation: EU 2023/1115 (EUDR) Article 11; ISO 31000:2018
Status: Production Ready
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import MitigationMeasureDesignerConfig, get_config
from .models import (
    Article11Category,
    EUDRCommodity,
    MeasurePriority,
    MeasureStatus,
    MeasureTemplate,
    MitigationMeasure,
    MitigationStrategy,
    RiskDimension,
    RiskLevel,
    RiskTrigger,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimension-to-measure mapping decision tree
# ---------------------------------------------------------------------------

_DIMENSION_MEASURE_PRIORITIES: Dict[RiskDimension, List[str]] = {
    RiskDimension.COUNTRY: [
        "country_legal_compliance",
        "governance_audit",
        "alternative_sourcing",
    ],
    RiskDimension.SUPPLIER: [
        "supplier_verification",
        "site_visit_audit",
        "supplier_compliance_program",
        "certificate_verification",
    ],
    RiskDimension.COMMODITY: [
        "traceability_enhancement",
        "commodity_certification",
        "supply_chain_segregation",
    ],
    RiskDimension.DEFORESTATION: [
        "satellite_monitoring",
        "ground_verification",
        "deforestation_free_certification",
    ],
    RiskDimension.CORRUPTION: [
        "third_party_audit",
        "enhanced_documentation",
        "anti_corruption_due_diligence",
    ],
    RiskDimension.SUPPLY_CHAIN_COMPLEXITY: [
        "supply_chain_simplification",
        "tier_mapping",
        "intermediary_verification",
    ],
    RiskDimension.MIXING_RISK: [
        "segregation_requirements",
        "batch_identity_preservation",
        "mass_balance_audit",
    ],
    RiskDimension.CIRCUMVENTION_RISK: [
        "route_verification",
        "independent_origin_checks",
        "additional_supplier_declarations",
    ],
}

# Threshold multipliers for priority assignment
_PRIORITY_THRESHOLDS: Dict[MeasurePriority, Decimal] = {
    MeasurePriority.CRITICAL: Decimal("80"),
    MeasurePriority.HIGH: Decimal("60"),
    MeasurePriority.MEDIUM: Decimal("40"),
    MeasurePriority.LOW: Decimal("0"),
}


class MitigationStrategyDesigner:
    """Designs targeted mitigation strategies based on risk decomposition.

    Analyzes which risk dimensions drive elevated risk from EUDR-028
    and selects optimal mitigation measures from the template library.
    Supports deterministic, auditable strategy design with SHA-256
    provenance hashing.

    Attributes:
        _config: Agent configuration instance.
        _provenance: Provenance tracker for audit trail.

    Example:
        >>> designer = MitigationStrategyDesigner()
        >>> strategy = await designer.design_strategy(risk_trigger, templates)
        >>> assert strategy.is_feasible
        >>> assert len(strategy.measures) > 0
    """

    def __init__(
        self,
        config: Optional[MitigationMeasureDesignerConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize MitigationStrategyDesigner.

        Args:
            config: Agent configuration. Uses get_config() if None.
            provenance: Provenance tracker instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        logger.info(
            "MitigationStrategyDesigner initialized: "
            "target_score=%s, max_measures=%d",
            self._config.mitigation_target_score,
            self._config.max_measures_per_strategy,
        )

    async def design_strategy(
        self,
        risk_trigger: RiskTrigger,
        templates: List[MeasureTemplate],
    ) -> MitigationStrategy:
        """Design a mitigation strategy for a risk trigger.

        Algorithm:
        1. Decompose risk into dimensions (which are above threshold?)
        2. Rank dimensions by score * weight (highest impact first)
        3. For each elevated dimension, select best-fit templates
        4. Estimate cumulative effectiveness
        5. Verify strategy can plausibly reduce risk to target
        6. Return assembled strategy

        Args:
            risk_trigger: Risk trigger from EUDR-028.
            templates: Available measure templates from the library.

        Returns:
            MitigationStrategy with selected measures and feasibility.

        Raises:
            ValueError: If risk_trigger has no dimension scores.
        """
        start_time = time.monotonic()
        logger.info(
            "Designing strategy for trigger=%s, operator=%s, "
            "commodity=%s, composite_score=%s",
            risk_trigger.assessment_id,
            risk_trigger.operator_id,
            risk_trigger.commodity.value,
            risk_trigger.composite_score,
        )

        # Step 1: Identify elevated dimensions
        elevated = self._identify_elevated_dimensions(risk_trigger)
        logger.info(
            "Identified %d elevated dimensions: %s",
            len(elevated),
            [(d.value, str(s)) for d, s in elevated],
        )

        if not elevated:
            logger.info(
                "No elevated dimensions found. "
                "Composite score %s may be below threshold.",
                risk_trigger.composite_score,
            )

        # Step 2: Select measures for each elevated dimension
        all_measures: List[MitigationMeasure] = []
        for dimension, score in elevated:
            dim_measures = self._select_measures_for_dimension(
                dimension=dimension,
                score=score,
                commodity=risk_trigger.commodity,
                templates=templates,
            )
            all_measures.extend(dim_measures)

            if len(all_measures) >= self._config.max_measures_per_strategy:
                all_measures = all_measures[
                    : self._config.max_measures_per_strategy
                ]
                break

        # Step 3: Assign priorities
        all_measures = self._assign_priorities(all_measures, risk_trigger)

        # Step 4: Estimate cumulative effectiveness
        total_reduction = self._estimate_cumulative_reduction(all_measures)
        logger.info(
            "Cumulative estimated reduction: %s%%",
            total_reduction,
        )

        # Step 5: Calculate timeline
        timeline_days = self._calculate_timeline(all_measures)

        # Step 6: Build strategy
        strategy_id = f"stg-{uuid.uuid4().hex[:12]}"
        target_score = self._config.mitigation_target_score

        is_feasible = self._validate_strategy_feasibility_score(
            pre_score=risk_trigger.composite_score,
            target_score=target_score,
            estimated_reduction=total_reduction,
        )

        # Step 7: Compute provenance hash
        provenance_data = {
            "strategy_id": strategy_id,
            "trigger_assessment_id": risk_trigger.assessment_id,
            "operator_id": risk_trigger.operator_id,
            "commodity": risk_trigger.commodity.value,
            "composite_score": str(risk_trigger.composite_score),
            "measure_count": len(all_measures),
            "measure_ids": [m.measure_id for m in all_measures],
            "total_reduction": str(total_reduction),
            "is_feasible": is_feasible,
        }
        provenance_hash = self._provenance.compute_hash(provenance_data)

        strategy = MitigationStrategy(
            strategy_id=strategy_id,
            workflow_id=f"wfl-{uuid.uuid4().hex[:12]}",
            risk_trigger=risk_trigger,
            measures=all_measures,
            pre_mitigation_score=risk_trigger.composite_score,
            target_score=target_score,
            status="strategy_designed",
            provenance_hash=provenance_hash,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Strategy designed: id=%s, measures=%d, "
            "reduction=%s%%, feasible=%s, elapsed=%.1fms",
            strategy_id,
            len(all_measures),
            total_reduction,
            is_feasible,
            elapsed_ms,
        )

        # Record provenance entry
        self._provenance.create_entry(
            step="design_strategy",
            source="eudr_028_risk_trigger",
            input_hash=self._provenance.compute_hash(
                {"assessment_id": risk_trigger.assessment_id}
            ),
            output_hash=provenance_hash,
        )

        return strategy

    def _identify_elevated_dimensions(
        self, risk_trigger: RiskTrigger,
    ) -> List[Tuple[RiskDimension, Decimal]]:
        """Identify which risk dimensions exceed the threshold.

        A dimension is elevated if its score exceeds the configured
        low_max threshold (default 30), indicating non-negligible risk
        that requires mitigation action.

        Args:
            risk_trigger: Risk trigger with dimensional scores.

        Returns:
            List of (dimension, score) tuples sorted by weighted impact
            descending.
        """
        from .models import DEFAULT_RISK_WEIGHTS

        threshold = self._config.low_max
        elevated: List[Tuple[RiskDimension, Decimal]] = []

        for dimension, score in risk_trigger.risk_dimensions.items():
            if score > threshold:
                elevated.append((dimension, score))

        # Sort by weighted impact (score * weight) descending
        elevated.sort(
            key=lambda item: item[1] * DEFAULT_RISK_WEIGHTS.get(
                item[0], Decimal("0.10")
            ),
            reverse=True,
        )

        return elevated

    def _select_measures_for_dimension(
        self,
        dimension: RiskDimension,
        score: Decimal,
        commodity: EUDRCommodity,
        templates: List[MeasureTemplate],
    ) -> List[MitigationMeasure]:
        """Select optimal measures for a specific risk dimension.

        Filters templates by dimension and commodity applicability,
        then selects the top matches based on base_effectiveness.

        Args:
            dimension: The elevated risk dimension.
            score: Risk score for this dimension.
            commodity: EUDR commodity for filtering.
            templates: Available measure templates.

        Returns:
            List of MitigationMeasure instances created from templates.
        """
        # Filter templates for this dimension and commodity
        applicable = self._filter_templates(
            templates=templates,
            dimension=dimension,
            commodity=commodity,
        )

        if not applicable:
            logger.warning(
                "No templates found for dimension=%s, commodity=%s. "
                "Creating generic measure.",
                dimension.value,
                commodity.value,
            )
            return [self._create_generic_measure(dimension, score)]

        # Sort by base_effectiveness descending
        applicable.sort(
            key=lambda t: t.base_effectiveness, reverse=True
        )

        # Select top 2-3 templates depending on score severity
        max_per_dim = 3 if score >= Decimal("60") else 2
        selected = applicable[:max_per_dim]

        measures: List[MitigationMeasure] = []
        for template in selected:
            measure = self._create_measure_from_template(
                template=template,
                dimension=dimension,
                score=score,
            )
            measures.append(measure)

        return measures

    def _filter_templates(
        self,
        templates: List[MeasureTemplate],
        dimension: RiskDimension,
        commodity: EUDRCommodity,
    ) -> List[MeasureTemplate]:
        """Filter templates by dimension and commodity.

        Args:
            templates: All available templates.
            dimension: Target risk dimension.
            commodity: Target commodity.

        Returns:
            Filtered list of applicable templates.
        """
        result: List[MeasureTemplate] = []
        for t in templates:
            # Check dimension match
            if dimension not in t.applicable_dimensions:
                continue
            # Check commodity match (empty list means all commodities)
            if t.applicable_commodities:
                if commodity not in t.applicable_commodities:
                    continue
            result.append(t)
        return result

    def _create_measure_from_template(
        self,
        template: MeasureTemplate,
        dimension: RiskDimension,
        score: Decimal,
    ) -> MitigationMeasure:
        """Create a MitigationMeasure from a MeasureTemplate.

        Args:
            template: Source template.
            dimension: Target dimension.
            score: Dimension risk score.

        Returns:
            MitigationMeasure instance with template properties.
        """
        measure_id = f"msr-{uuid.uuid4().hex[:12]}"

        return MitigationMeasure(
            measure_id=measure_id,
            strategy_id="",  # Set when added to strategy
            template_id=template.template_id,
            title=template.title,
            description=template.description,
            article11_category=template.article11_category,
            target_dimension=dimension,
            status=MeasureStatus.PROPOSED,
            priority=MeasurePriority.MEDIUM,
            expected_risk_reduction=template.base_effectiveness,
        )

    def _create_generic_measure(
        self,
        dimension: RiskDimension,
        score: Decimal,
    ) -> MitigationMeasure:
        """Create a generic measure when no templates match.

        Args:
            dimension: Target dimension.
            score: Dimension risk score.

        Returns:
            Generic MitigationMeasure for the dimension.
        """
        titles = {
            RiskDimension.COUNTRY: "Country-Level Enhanced Due Diligence",
            RiskDimension.SUPPLIER: "Supplier Verification Program",
            RiskDimension.COMMODITY: "Commodity Traceability Enhancement",
            RiskDimension.DEFORESTATION: "Deforestation Monitoring Intensification",
            RiskDimension.CORRUPTION: "Anti-Corruption Due Diligence",
            RiskDimension.SUPPLY_CHAIN_COMPLEXITY: "Supply Chain Simplification",
            RiskDimension.MIXING_RISK: "Product Segregation Enforcement",
            RiskDimension.CIRCUMVENTION_RISK: "Origin Verification Protocol",
        }

        return MitigationMeasure(
            measure_id=f"msr-{uuid.uuid4().hex[:12]}",
            strategy_id="",
            title=titles.get(dimension, f"Mitigation for {dimension.value}"),
            description=(
                f"Generic mitigation measure for {dimension.value} "
                f"risk dimension with score {score}."
            ),
            article11_category=Article11Category.OTHER_MEASURES,
            target_dimension=dimension,
            status=MeasureStatus.PROPOSED,
            priority=MeasurePriority.MEDIUM,
            expected_risk_reduction=Decimal("15"),
        )

    def _estimate_cumulative_reduction(
        self, measures: List[MitigationMeasure],
    ) -> Decimal:
        """Estimate total risk reduction from all measures.

        Uses diminishing returns formula:
        Total = 1 - Product(1 - Ri) for each measure i
        where Ri is the measure's expected_risk_reduction / 100.

        This prevents unrealistic cumulative claims (e.g., three
        30% measures do not give 90% reduction).

        Args:
            measures: List of measures with expected_risk_reduction.

        Returns:
            Cumulative risk reduction percentage (0-100).
        """
        if not measures:
            return Decimal("0")

        product = Decimal("1")
        for m in measures:
            ri = m.expected_risk_reduction / Decimal("100")
            ri = min(ri, Decimal("1"))  # Cap at 100%
            product *= (Decimal("1") - ri)

        total = (Decimal("1") - product) * Decimal("100")

        # Cap at maximum effectiveness
        cap = self._config.max_effectiveness_cap
        total = min(total, cap)

        return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _validate_strategy_feasibility_score(
        self,
        pre_score: Decimal,
        target_score: Decimal,
        estimated_reduction: Decimal,
    ) -> bool:
        """Verify strategy can plausibly achieve target risk reduction.

        The strategy is feasible if the estimated reduction can bring
        the pre-mitigation score down to the target score.

        Args:
            pre_score: Pre-mitigation composite score.
            target_score: Target post-mitigation score.
            estimated_reduction: Estimated cumulative reduction percentage.

        Returns:
            True if strategy can plausibly achieve target.
        """
        required_reduction_abs = pre_score - target_score
        if required_reduction_abs <= Decimal("0"):
            return True  # Already at or below target

        estimated_reduction_abs = (
            pre_score * estimated_reduction / Decimal("100")
        )

        is_feasible = estimated_reduction_abs >= required_reduction_abs
        logger.debug(
            "Feasibility check: required=%s, estimated=%s, feasible=%s",
            required_reduction_abs,
            estimated_reduction_abs,
            is_feasible,
        )
        return is_feasible

    def _assign_priorities(
        self,
        measures: List[MitigationMeasure],
        risk_trigger: RiskTrigger,
    ) -> List[MitigationMeasure]:
        """Assign priorities based on risk impact and urgency.

        Priority assignment rules:
        - Dimension score >= 80 -> CRITICAL
        - Dimension score >= 60 -> HIGH
        - Dimension score >= 40 -> MEDIUM
        - Otherwise -> LOW

        Args:
            measures: Measures to prioritize.
            risk_trigger: Original risk trigger for context.

        Returns:
            Measures with updated priority assignments.
        """
        result: List[MitigationMeasure] = []
        dim_scores = risk_trigger.risk_dimensions

        for measure in measures:
            dim = measure.target_dimension
            score = dim_scores.get(dim, Decimal("0"))

            priority = MeasurePriority.LOW
            for p, threshold in sorted(
                _PRIORITY_THRESHOLDS.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                if score >= threshold:
                    priority = p
                    break

            # Create updated measure with new priority
            updated = MitigationMeasure(
                measure_id=measure.measure_id,
                strategy_id=measure.strategy_id,
                template_id=measure.template_id,
                title=measure.title,
                description=measure.description,
                article11_category=measure.article11_category,
                target_dimension=measure.target_dimension,
                status=measure.status,
                priority=priority,
                expected_risk_reduction=measure.expected_risk_reduction,
            )
            result.append(updated)

        # Sort by priority (CRITICAL first)
        priority_order = {
            MeasurePriority.CRITICAL: 0,
            MeasurePriority.HIGH: 1,
            MeasurePriority.MEDIUM: 2,
            MeasurePriority.LOW: 3,
        }
        result.sort(key=lambda m: priority_order.get(m.priority, 4))

        return result

    def _calculate_timeline(
        self, measures: List[MitigationMeasure],
    ) -> int:
        """Calculate estimated timeline considering parallel execution.

        Assumes measures within the same priority level can execute
        in parallel, so the timeline is the maximum across each
        priority group.

        Args:
            measures: List of measures.

        Returns:
            Estimated total timeline in days.
        """
        if not measures:
            return 0

        # Group by priority and take max within each group
        priority_groups: Dict[MeasurePriority, int] = {}
        default_days = self._config.default_deadline_days

        for m in measures:
            current_max = priority_groups.get(m.priority, 0)
            # Use default deadline as estimate
            priority_groups[m.priority] = max(current_max, default_days)

        # Sum across priority groups (sequential execution between groups)
        total_days = sum(priority_groups.values())
        return total_days

    async def redesign_strategy(
        self,
        original_strategy: MitigationStrategy,
        templates: List[MeasureTemplate],
        excluded_template_ids: Optional[List[str]] = None,
    ) -> MitigationStrategy:
        """Redesign a strategy after insufficient verification.

        Creates a new strategy excluding previously ineffective
        measures and selecting alternative templates.

        Args:
            original_strategy: The strategy that was insufficient.
            templates: Available templates.
            excluded_template_ids: Templates to exclude from selection.

        Returns:
            New MitigationStrategy with alternative measures.
        """
        excluded = set(excluded_template_ids or [])
        # Add templates from original strategy to exclusion list
        for m in original_strategy.measures:
            if m.template_id:
                excluded.add(m.template_id)

        # Filter out excluded templates
        available = [t for t in templates if t.template_id not in excluded]

        if not available:
            logger.warning(
                "No alternative templates available after exclusion. "
                "Using original template set."
            )
            available = templates

        return await self.design_strategy(
            risk_trigger=original_strategy.risk_trigger,
            templates=available,
        )

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Dictionary with engine status and configuration details.
        """
        return {
            "engine": "MitigationStrategyDesigner",
            "status": "available",
            "config": {
                "target_score": str(self._config.mitigation_target_score),
                "max_measures": self._config.max_measures_per_strategy,
                "max_effectiveness_cap": str(
                    self._config.max_effectiveness_cap
                ),
            },
        }
