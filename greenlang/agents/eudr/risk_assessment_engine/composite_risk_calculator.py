# -*- coding: utf-8 -*-
"""
AGENT-EUDR-028: Risk Assessment Engine - Composite Risk Calculator

Core computation engine for EUDR due diligence risk assessments. Computes
weighted composite risk scores from multi-dimensional risk factor inputs
using the confidence-weighted formula:

    Composite_Risk = SUM(W_i * S_i * C_i) / SUM(W_i * C_i)

where W_i is the dimension weight, S_i is the raw score (0-100), and C_i
is the confidence coefficient (0-1). Country benchmarking multipliers per
Article 29 are applied post-aggregation. All arithmetic uses Python Decimal
with ROUND_HALF_UP to guarantee deterministic, audit-ready results.

Production infrastructure includes:
    - 8-dimension weighted composite scoring with Decimal precision
    - Confidence-weighted aggregation for multi-source inputs per dimension
    - Country benchmark multiplier application (Article 29: 0.70/1.00/1.50)
    - Override recalculation with provenance chain continuity
    - SHA-256 provenance hash on every CompositeRiskScore
    - Prometheus metrics integration for calculation monitoring

Zero-Hallucination Guarantees:
    - All scores computed via deterministic weighted Decimal arithmetic
    - No LLM involvement in scoring, weighting, or aggregation
    - Country benchmark multipliers from EC-published lookup tables only
    - All provenance hashes computed from canonical JSON

Regulatory References:
    - EUDR Article 10(2): Risk assessment criteria and weighting
    - EUDR Article 29: Country benchmarking system
    - EUDR Article 31: 5-year record retention for risk scores
    - EUDR Article 13: Simplified due diligence for low-risk countries

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-028 (Engine 1: Composite Risk Calculator)
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
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
    get_config,
)
from greenlang.agents.eudr.risk_assessment_engine.models import (
    CompositeRiskScore,
    CountryBenchmark,
    CountryBenchmarkLevel,
    DimensionScore,
    RiskDimension,
    RiskFactorInput,
    RiskLevel,
    RiskOverride,
    DEFAULT_WEIGHTS,
    COUNTRY_BENCHMARK_MULTIPLIERS,
    RISK_THRESHOLDS,
)
from greenlang.agents.eudr.risk_assessment_engine.provenance import ProvenanceTracker
from greenlang.schemas import utcnow
from greenlang.agents.eudr.risk_assessment_engine.metrics import (
    record_composite_calculation,
    observe_calculation_duration,
)

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Precision constants
# ---------------------------------------------------------------------------

_SCORE_PRECISION = Decimal("0.01")
_WEIGHT_PRECISION = Decimal("0.0001")
_CONFIDENCE_PRECISION = Decimal("0.0001")
_SCORE_MIN = Decimal("0")
_SCORE_MAX = Decimal("100")
_CONFIDENCE_MIN = Decimal("0")
_CONFIDENCE_MAX = Decimal("1")
_WEIGHT_SUM_TOLERANCE = Decimal("0.01")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash of data.

    Args:
        data: Any JSON-serializable object.

    Returns:
        64-character lowercase hex SHA-256 hash string.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def _clamp(value: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    """Clamp a Decimal value between lo and hi inclusive.

    Args:
        value: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped Decimal value.
    """
    return max(lo, min(value, hi))

# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class CompositeRiskCalculator:
    """Engine for computing weighted composite risk scores.

    Aggregates multi-dimensional risk factor inputs into a single composite
    score using confidence-weighted arithmetic. Supports country benchmark
    multipliers (Article 29) and override-based recalculation with full
    provenance chain continuity.

    The core formula is:

        Composite = SUM(W_i * S_i * C_i) / SUM(W_i * C_i)

    where W_i is the dimension weight, S_i is the raw score (0-100), and
    C_i is the confidence coefficient (0.0-1.0). When multiple inputs exist
    for the same dimension, they are averaged weighted by confidence before
    the dimension-level aggregation.

    Args:
        config: Agent configuration (uses singleton if None).

    Example:
        >>> calculator = CompositeRiskCalculator()
        >>> inputs = [
        ...     RiskFactorInput(
        ...         source_agent="EUDR-016",
        ...         dimension=RiskDimension.COUNTRY,
        ...         raw_score=Decimal("45"),
        ...         confidence=Decimal("0.90"),
        ...     ),
        ... ]
        >>> result = calculator.calculate_composite_score(inputs)
        >>> assert result.risk_level in RiskLevel
    """

    def __init__(self, config: Optional[RiskAssessmentEngineConfig] = None) -> None:
        """Initialize CompositeRiskCalculator.

        Args:
            config: Agent configuration (uses singleton if None).
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._calculation_count: int = 0
        self._total_dimensions_scored: int = 0
        self._override_count: int = 0
        logger.info(
            "CompositeRiskCalculator initialized "
            "(dimensions=%d, weights_valid=%s)",
            len(self._config.dimension_weights),
            self._validate_weights(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_composite_score(
        self,
        factor_inputs: List[RiskFactorInput],
        country_benchmarks: Optional[List[CountryBenchmark]] = None,
    ) -> CompositeRiskScore:
        """Calculate composite risk score from multi-dimensional factor inputs.

        Groups inputs by dimension, computes a confidence-weighted score per
        dimension, aggregates to a single composite using configured weights,
        and optionally applies country benchmark multipliers.

        Args:
            factor_inputs: Risk factor inputs from upstream agents.
            country_benchmarks: Optional country benchmarks for Article 29
                multiplier application.

        Returns:
            CompositeRiskScore with full dimension breakdown and provenance.

        Raises:
            ValueError: If factor_inputs is empty.
        """
        start_time = time.monotonic()

        if not factor_inputs:
            raise ValueError("factor_inputs must contain at least one input")

        # Step 1: Group inputs by dimension
        dimension_groups: Dict[RiskDimension, List[RiskFactorInput]] = {}
        for inp in factor_inputs:
            dimension_groups.setdefault(inp.dimension, []).append(inp)

        # Step 2: Compute per-dimension scores
        dimension_scores: List[DimensionScore] = []
        for dimension, inputs in dimension_groups.items():
            dim_score = self._compute_dimension_score(inputs, dimension)
            dimension_scores.append(dim_score)

        # Step 3: Compute weighted composite
        numerator = Decimal("0")
        denominator = Decimal("0")
        total_weight = Decimal("0")

        for ds in dimension_scores:
            numerator += ds.weighted_score * ds.confidence
            denominator += ds.weight * ds.confidence
            total_weight += ds.weight

        if denominator == Decimal("0"):
            overall_score = Decimal("0")
        else:
            overall_score = (numerator / denominator).quantize(
                _SCORE_PRECISION, rounding=ROUND_HALF_UP
            )

        overall_score = _clamp(overall_score, _SCORE_MIN, _SCORE_MAX)

        # Step 4: Compute effective confidence
        if total_weight == Decimal("0"):
            effective_confidence = Decimal("0")
        else:
            confidence_sum = sum(
                ds.confidence * ds.weight for ds in dimension_scores
            )
            effective_confidence = (confidence_sum / total_weight).quantize(
                _CONFIDENCE_PRECISION, rounding=ROUND_HALF_UP
            )
        effective_confidence = _clamp(
            effective_confidence, _CONFIDENCE_MIN, _CONFIDENCE_MAX
        )

        # Step 5: Apply country benchmark multiplier
        benchmark_applied = False
        benchmark_multiplier = Decimal("1.00")
        if country_benchmarks:
            overall_score, benchmark_applied, benchmark_multiplier = (
                self._apply_country_benchmark(overall_score, country_benchmarks)
            )

        # Step 6: Classify risk level from thresholds
        risk_level = self._classify_score(overall_score)

        # Step 7: Provenance hash
        provenance_data = {
            "factor_count": len(factor_inputs),
            "dimensions": len(dimension_scores),
            "overall_score": str(overall_score),
            "risk_level": risk_level.value,
            "benchmark_applied": benchmark_applied,
            "benchmark_multiplier": str(benchmark_multiplier),
            "total_weight": str(total_weight),
        }
        provenance_hash = _compute_hash(provenance_data)

        # Step 8: Build result
        result = CompositeRiskScore(
            overall_score=overall_score,
            risk_level=risk_level,
            dimension_scores=dimension_scores,
            total_weight=total_weight.quantize(
                _WEIGHT_PRECISION, rounding=ROUND_HALF_UP
            ),
            effective_confidence=effective_confidence,
            country_benchmark_applied=benchmark_applied,
            benchmark_multiplier=benchmark_multiplier,
            provenance_hash=provenance_hash,
        )

        # Step 9: Provenance chain entry
        self._provenance.create_entry(
            step="composite_risk_calculation",
            source="risk_factor_inputs",
            input_hash=_compute_hash({
                "inputs": [
                    {
                        "dimension": inp.dimension.value,
                        "raw_score": str(inp.raw_score),
                        "confidence": str(inp.confidence),
                        "source_agent": inp.source_agent,
                    }
                    for inp in factor_inputs
                ]
            }),
            output_hash=provenance_hash,
        )

        # Step 10: Metrics
        self._calculation_count += 1
        self._total_dimensions_scored += len(dimension_scores)
        elapsed = time.monotonic() - start_time
        record_composite_calculation(risk_level.value)
        observe_calculation_duration(elapsed)

        logger.info(
            "Composite risk calculated: score=%s, level=%s, "
            "dimensions=%d, benchmark=%s (%.0fms)",
            overall_score,
            risk_level.value,
            len(dimension_scores),
            benchmark_applied,
            elapsed * 1000,
        )
        return result

    def recalculate_with_override(
        self,
        existing: CompositeRiskScore,
        override: RiskOverride,
    ) -> CompositeRiskScore:
        """Recalculate composite score with a manual override applied.

        Replaces the targeted dimension score with the override value and
        recomputes the composite. The override is recorded in the provenance
        chain for audit trail integrity.

        Args:
            existing: Current composite risk score to override.
            override: Override specification with target dimension and value.

        Returns:
            New CompositeRiskScore reflecting the override.
        """
        start_time = time.monotonic()

        # Clone dimension scores and apply override
        updated_scores: List[DimensionScore] = []
        for ds in existing.dimension_scores:
            if ds.dimension == override.dimension:
                new_raw = _clamp(override.override_score, _SCORE_MIN, _SCORE_MAX)
                new_weight = self._get_dimension_weight(ds.dimension)
                new_weighted = (new_weight * new_raw).quantize(
                    _SCORE_PRECISION, rounding=ROUND_HALF_UP
                )
                updated_scores.append(DimensionScore(
                    dimension=ds.dimension,
                    weighted_score=new_weighted,
                    raw_score=new_raw,
                    weight=new_weight,
                    confidence=ds.confidence,
                    source_agent=f"OVERRIDE:{override.overridden_by}",
                    explanation=(
                        f"Manual override ({override.reason.value}): "
                        f"{override.justification}"
                    ),
                ))
            else:
                updated_scores.append(ds)

        # Recompute composite from updated dimension scores
        numerator = Decimal("0")
        denominator = Decimal("0")
        total_weight = Decimal("0")

        for ds in updated_scores:
            numerator += ds.weighted_score * ds.confidence
            denominator += ds.weight * ds.confidence
            total_weight += ds.weight

        if denominator == Decimal("0"):
            overall_score = Decimal("0")
        else:
            overall_score = (numerator / denominator).quantize(
                _SCORE_PRECISION, rounding=ROUND_HALF_UP
            )
        overall_score = _clamp(overall_score, _SCORE_MIN, _SCORE_MAX)

        # Apply same benchmark if previously applied
        benchmark_applied = existing.country_benchmark_applied
        benchmark_multiplier = existing.benchmark_multiplier
        if benchmark_applied and benchmark_multiplier != Decimal("1.00"):
            overall_score = (overall_score * benchmark_multiplier).quantize(
                _SCORE_PRECISION, rounding=ROUND_HALF_UP
            )
            overall_score = _clamp(overall_score, _SCORE_MIN, _SCORE_MAX)

        # Effective confidence
        if total_weight == Decimal("0"):
            effective_confidence = Decimal("0")
        else:
            confidence_sum = sum(
                ds.confidence * ds.weight for ds in updated_scores
            )
            effective_confidence = (confidence_sum / total_weight).quantize(
                _CONFIDENCE_PRECISION, rounding=ROUND_HALF_UP
            )

        risk_level = self._classify_score(overall_score)

        provenance_hash = _compute_hash({
            "override_dimension": override.dimension.value,
            "override_score": str(override.override_score),
            "override_reason": override.reason.value,
            "previous_hash": existing.provenance_hash,
            "new_score": str(overall_score),
        })

        self._override_count += 1
        elapsed = time.monotonic() - start_time

        logger.info(
            "Composite recalculated with override on %s: "
            "score=%s->%s, level=%s (%.0fms)",
            override.dimension.value,
            existing.overall_score,
            overall_score,
            risk_level.value,
            elapsed * 1000,
        )

        return CompositeRiskScore(
            overall_score=overall_score,
            risk_level=risk_level,
            dimension_scores=updated_scores,
            total_weight=total_weight.quantize(
                _WEIGHT_PRECISION, rounding=ROUND_HALF_UP
            ),
            effective_confidence=effective_confidence,
            country_benchmark_applied=benchmark_applied,
            benchmark_multiplier=benchmark_multiplier,
            provenance_hash=provenance_hash,
        )

    def get_calculation_stats(self) -> Dict[str, Any]:
        """Return composite risk calculation statistics.

        Returns:
            Dict with total_calculations, total_dimensions_scored,
            average_dimensions_per_calc, override_count, and
            weights_valid keys.
        """
        avg_dims = (
            self._total_dimensions_scored / self._calculation_count
            if self._calculation_count > 0
            else 0
        )
        return {
            "total_calculations": self._calculation_count,
            "total_dimensions_scored": self._total_dimensions_scored,
            "average_dimensions_per_calc": round(avg_dims, 2),
            "override_count": self._override_count,
            "weights_valid": self._validate_weights(),
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_dimension_score(
        self,
        inputs: List[RiskFactorInput],
        dimension: RiskDimension,
    ) -> DimensionScore:
        """Compute aggregated score for a single risk dimension.

        When multiple inputs exist for the same dimension (e.g., multiple
        supplier risk scores), they are averaged weighted by confidence:

            dim_score = SUM(score_i * conf_i) / SUM(conf_i)

        Args:
            inputs: Risk factor inputs for this dimension.
            dimension: The risk dimension being scored.

        Returns:
            DimensionScore with weighted score and explanation.
        """
        if not inputs:
            weight = self._get_dimension_weight(dimension)
            return DimensionScore(
                dimension=dimension,
                weighted_score=Decimal("0"),
                raw_score=Decimal("0"),
                weight=weight,
                confidence=Decimal("0"),
                source_agent="none",
                explanation=f"No inputs for {dimension.value}",
            )

        # Confidence-weighted average of raw scores
        score_numerator = Decimal("0")
        confidence_sum = Decimal("0")
        source_agents: List[str] = []

        for inp in inputs:
            clamped_score = _clamp(inp.raw_score, _SCORE_MIN, _SCORE_MAX)
            clamped_conf = _clamp(inp.confidence, _CONFIDENCE_MIN, _CONFIDENCE_MAX)
            score_numerator += clamped_score * clamped_conf
            confidence_sum += clamped_conf
            if inp.source_agent not in source_agents:
                source_agents.append(inp.source_agent)

        if confidence_sum == Decimal("0"):
            raw_score = Decimal("0")
            avg_confidence = Decimal("0")
        else:
            raw_score = (score_numerator / confidence_sum).quantize(
                _SCORE_PRECISION, rounding=ROUND_HALF_UP
            )
            avg_confidence = (confidence_sum / Decimal(str(len(inputs)))).quantize(
                _CONFIDENCE_PRECISION, rounding=ROUND_HALF_UP
            )

        raw_score = _clamp(raw_score, _SCORE_MIN, _SCORE_MAX)
        avg_confidence = _clamp(avg_confidence, _CONFIDENCE_MIN, _CONFIDENCE_MAX)

        weight = self._get_dimension_weight(dimension)
        weighted_score = (weight * raw_score).quantize(
            _SCORE_PRECISION, rounding=ROUND_HALF_UP
        )

        explanation = (
            f"{dimension.value}: raw={raw_score}, weight={weight}, "
            f"confidence={avg_confidence}, sources={len(inputs)} "
            f"({', '.join(source_agents)})"
        )

        return DimensionScore(
            dimension=dimension,
            weighted_score=weighted_score,
            raw_score=raw_score,
            weight=weight,
            confidence=avg_confidence,
            source_agent=", ".join(source_agents),
            explanation=explanation,
        )

    def _apply_country_benchmark(
        self,
        score: Decimal,
        benchmarks: List[CountryBenchmark],
    ) -> Tuple[Decimal, bool, Decimal]:
        """Apply country benchmark multiplier to composite score.

        Takes the highest-risk country benchmark from the list and applies
        its multiplier. The score is capped at 100 after application.

        Args:
            score: Pre-benchmark composite score (0-100).
            benchmarks: Country benchmarks to evaluate.

        Returns:
            Tuple of (adjusted_score, was_applied, multiplier_used).
        """
        if not benchmarks:
            return score, False, Decimal("1.00")

        # Determine highest-risk benchmark
        level_priority = {
            CountryBenchmarkLevel.HIGH: 3,
            CountryBenchmarkLevel.STANDARD: 2,
            CountryBenchmarkLevel.LOW: 1,
        }
        highest_benchmark = max(
            benchmarks, key=lambda b: level_priority.get(b.level, 0)
        )

        multiplier = COUNTRY_BENCHMARK_MULTIPLIERS.get(
            highest_benchmark.level, Decimal("1.00")
        )

        adjusted = (score * multiplier).quantize(
            _SCORE_PRECISION, rounding=ROUND_HALF_UP
        )
        adjusted = _clamp(adjusted, _SCORE_MIN, _SCORE_MAX)

        if multiplier != Decimal("1.00"):
            logger.info(
                "Country benchmark applied: %s (%s) multiplier=%s, "
                "score %s -> %s",
                highest_benchmark.country_code,
                highest_benchmark.level.value,
                multiplier,
                score,
                adjusted,
            )

        return adjusted, True, multiplier

    def _classify_score(self, score: Decimal) -> RiskLevel:
        """Classify a composite score into a risk level using thresholds.

        Thresholds (from config/defaults):
            0-15:   NEGLIGIBLE
            16-30:  LOW
            31-60:  STANDARD
            61-80:  HIGH
            81-100: CRITICAL

        Args:
            score: Composite risk score (0-100).

        Returns:
            RiskLevel classification.
        """
        negligible = Decimal(str(
            self._config.risk_thresholds.get("negligible", 15)
        ))
        low = Decimal(str(self._config.risk_thresholds.get("low", 30)))
        standard = Decimal(str(self._config.risk_thresholds.get("standard", 60)))
        high = Decimal(str(self._config.risk_thresholds.get("high", 80)))

        if score <= negligible:
            return RiskLevel.NEGLIGIBLE
        elif score <= low:
            return RiskLevel.LOW
        elif score <= standard:
            return RiskLevel.STANDARD
        elif score <= high:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _get_dimension_weight(self, dimension: RiskDimension) -> Decimal:
        """Get the configured weight for a risk dimension.

        Falls back to DEFAULT_WEIGHTS if not found in config.

        Args:
            dimension: Risk dimension to look up.

        Returns:
            Weight as Decimal (0.0-1.0).
        """
        config_weights = self._config.dimension_weights
        weight = config_weights.get(
            dimension.value,
            DEFAULT_WEIGHTS.get(dimension.value, Decimal("0.05")),
        )
        return Decimal(str(weight))

    def _validate_weights(self) -> bool:
        """Verify that configured dimension weights sum to approximately 1.0.

        Returns:
            True if weights sum is within tolerance of 1.0.
        """
        weights = self._config.dimension_weights
        if not weights:
            weights = DEFAULT_WEIGHTS
        total = sum(Decimal(str(w)) for w in weights.values())
        return abs(total - Decimal("1.0")) <= _WEIGHT_SUM_TOLERANCE
