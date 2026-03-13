# -*- coding: utf-8 -*-
"""
AGENT-EUDR-028: Risk Assessment Engine - Article 10 Criteria Evaluator

Systematically evaluates all 10 risk assessment criteria defined in
EUDR Article 10(2) against collected risk factor data. Each criterion
is evaluated independently and classified as PASS, CONCERN, FAIL, or
NOT_EVALUATED. The aggregate result informs risk classification
escalation or de-escalation decisions.

The 10 Article 10(2) criteria cover:
    1. Prevalence of deforestation/forest degradation
    2. Supply chain complexity
    3. Risk of mixing with unknown-origin products
    4. Risk of circumvention of the regulation
    5. Country governance and rule of law
    6. Supplier compliance history
    7. Commodity-specific risk profile
    8. Certification and verification coverage
    9. Deforestation alerts in sourcing regions
   10. Legal framework adequacy in production countries

Production infrastructure includes:
    - Independent per-criterion evaluation with deterministic thresholds
    - Aggregated criteria result with summary statistics
    - Integration with country benchmarks for governance/legal criteria
    - Integration with composite scores for threshold-based evaluation
    - SHA-256 provenance hash on every Article10CriteriaResult
    - Prometheus metrics integration

Zero-Hallucination Guarantees:
    - All criterion thresholds are hardcoded Decimal constants
    - Evaluations use deterministic numeric comparisons only
    - No LLM involvement in criterion assessment or classification
    - All provenance hashes computed from canonical JSON

Regulatory References:
    - EUDR Article 10(2)(a-j): 10 specific risk assessment criteria
    - EUDR Article 10(1): Obligation to assess risk
    - EUDR Article 29: Country benchmarking for criteria (a), (e), (j)
    - EUDR Article 13: Simplified DD implications from criteria results

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-028 (Engine 4: Article 10 Criteria Evaluator)
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
    Article10Criterion,
    Article10CriteriaResult,
    Article10CriterionEvaluation,
    CompositeRiskScore,
    CountryBenchmark,
    CountryBenchmarkLevel,
    CriterionResult,
    RiskDimension,
    RiskFactorInput,
)
from greenlang.agents.eudr.risk_assessment_engine.provenance import ProvenanceTracker
from greenlang.agents.eudr.risk_assessment_engine.metrics import (
    record_criteria_evaluation,
)

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Evaluation thresholds (deterministic, no LLM)
# ---------------------------------------------------------------------------

_DEFORESTATION_CONCERN_THRESHOLD = Decimal("50")
_COMPLEXITY_CONCERN_THRESHOLD = Decimal("60")
_MIXING_CONCERN_THRESHOLD = Decimal("50")
_CIRCUMVENTION_CONCERN_THRESHOLD = Decimal("50")
_GOVERNANCE_CONCERN_THRESHOLD = Decimal("55")
_SUPPLIER_CONCERN_THRESHOLD = Decimal("55")
_COMMODITY_CONCERN_THRESHOLD = Decimal("60")
_CERTIFICATION_CONCERN_THRESHOLD = Decimal("60")
_ALERT_CONCERN_THRESHOLD = Decimal("50")
_LEGAL_CONCERN_THRESHOLD = Decimal("55")

_FAIL_MULTIPLIER = Decimal("1.4")  # FAIL threshold = CONCERN * 1.4


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


def _get_dimension_score(
    inputs: List[RiskFactorInput],
    dimension: RiskDimension,
) -> Optional[Decimal]:
    """Extract average score for a dimension from factor inputs.

    Args:
        inputs: Risk factor inputs.
        dimension: Dimension to filter.

    Returns:
        Average score or None if no matching inputs.
    """
    matching = [i for i in inputs if i.dimension == dimension]
    if not matching:
        return None
    total = sum(i.raw_score for i in matching)
    return (total / Decimal(str(len(matching)))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def _has_high_risk_country(benchmarks: List[CountryBenchmark]) -> bool:
    """Check if any benchmark is HIGH risk.

    Args:
        benchmarks: Country benchmarks.

    Returns:
        True if at least one country is HIGH risk.
    """
    return any(b.level == CountryBenchmarkLevel.HIGH for b in benchmarks)


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class Article10CriteriaEvaluator:
    """Engine for evaluating EUDR Article 10(2) risk assessment criteria.

    Evaluates each of the 10 criteria independently using deterministic
    threshold comparisons against risk factor inputs and country
    benchmarks. Results are classified as PASS, CONCERN, FAIL, or
    NOT_EVALUATED per criterion, with an aggregate summary.

    Args:
        config: Agent configuration (uses singleton if None).

    Example:
        >>> evaluator = Article10CriteriaEvaluator()
        >>> result = evaluator.evaluate_all_criteria(
        ...     factor_inputs=inputs,
        ...     country_benchmarks=benchmarks,
        ...     composite_score=composite,
        ... )
        >>> assert result.total_evaluated == 10
    """

    def __init__(self, config: Optional[RiskAssessmentEngineConfig] = None) -> None:
        """Initialize Article10CriteriaEvaluator.

        Args:
            config: Agent configuration (uses singleton if None).
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._evaluation_count: int = 0
        self._concern_count: int = 0
        self._fail_count: int = 0
        logger.info("Article10CriteriaEvaluator initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_all_criteria(
        self,
        factor_inputs: List[RiskFactorInput],
        country_benchmarks: List[CountryBenchmark],
        composite_score: CompositeRiskScore,
    ) -> Article10CriteriaResult:
        """Evaluate all 10 Article 10(2) criteria.

        Runs each criterion evaluator and aggregates results into a
        single Article10CriteriaResult with summary statistics.

        Args:
            factor_inputs: Risk factor inputs from aggregation.
            country_benchmarks: Country benchmark data.
            composite_score: Pre-computed composite risk score.

        Returns:
            Article10CriteriaResult with all evaluations and summary.
        """
        start_time = time.monotonic()

        evaluations: List[Article10CriterionEvaluation] = [
            self._evaluate_prevalence_of_deforestation(
                factor_inputs, country_benchmarks
            ),
            self._evaluate_supply_chain_complexity(factor_inputs),
            self._evaluate_mixing_risk(factor_inputs),
            self._evaluate_circumvention_risk(factor_inputs),
            self._evaluate_country_governance(factor_inputs, country_benchmarks),
            self._evaluate_supplier_compliance(factor_inputs),
            self._evaluate_commodity_risk_profile(factor_inputs),
            self._evaluate_certification_coverage(factor_inputs),
            self._evaluate_deforestation_alerts(factor_inputs),
            self._evaluate_legal_framework(factor_inputs, country_benchmarks),
        ]

        # Summary counts
        pass_count = sum(
            1 for e in evaluations if e.result == CriterionResult.PASS
        )
        concern_count = sum(
            1 for e in evaluations if e.result == CriterionResult.CONCERN
        )
        fail_count = sum(
            1 for e in evaluations if e.result == CriterionResult.FAIL
        )
        not_evaluated = sum(
            1 for e in evaluations if e.result == CriterionResult.NOT_EVALUATED
        )

        # Provenance
        provenance_hash = _compute_hash({
            "total": len(evaluations),
            "pass": pass_count,
            "concern": concern_count,
            "fail": fail_count,
            "not_evaluated": not_evaluated,
            "composite_score": str(composite_score.overall_score),
        })

        result = Article10CriteriaResult(
            evaluations=evaluations,
            total_evaluated=len(evaluations) - not_evaluated,
            pass_count=pass_count,
            concern_count=concern_count,
            fail_count=fail_count,
            not_evaluated_count=not_evaluated,
            evaluated_at=_utcnow(),
            provenance_hash=provenance_hash,
        )

        # Provenance chain
        self._provenance.create_entry(
            step="article10_criteria_evaluation",
            source="risk_factor_inputs",
            input_hash=_compute_hash({
                "factor_count": len(factor_inputs),
                "benchmark_count": len(country_benchmarks),
                "composite_score": str(composite_score.overall_score),
            }),
            output_hash=provenance_hash,
        )

        # Stats
        self._evaluation_count += 1
        self._concern_count += concern_count
        self._fail_count += fail_count
        elapsed = time.monotonic() - start_time
        record_criteria_evaluation(
            pass_count=pass_count,
            concern_count=concern_count,
            fail_count=fail_count,
        )

        logger.info(
            "Article 10 criteria evaluated: %d pass, %d concern, %d fail, "
            "%d not_evaluated (%.0fms)",
            pass_count,
            concern_count,
            fail_count,
            not_evaluated,
            elapsed * 1000,
        )
        return result

    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Return Article 10 criteria evaluation statistics.

        Returns:
            Dict with total_evaluations, total_concerns, total_fails,
            and average_concerns_per_eval keys.
        """
        avg_concerns = (
            self._concern_count / self._evaluation_count
            if self._evaluation_count > 0
            else 0
        )
        return {
            "total_evaluations": self._evaluation_count,
            "total_concerns": self._concern_count,
            "total_fails": self._fail_count,
            "average_concerns_per_eval": round(avg_concerns, 2),
        }

    # ------------------------------------------------------------------
    # Individual criterion evaluators
    # ------------------------------------------------------------------

    def _evaluate_prevalence_of_deforestation(
        self,
        inputs: List[RiskFactorInput],
        benchmarks: List[CountryBenchmark],
    ) -> Article10CriterionEvaluation:
        """Evaluate criterion (a): Prevalence of deforestation.

        CONCERN if deforestation score > 50 or any HIGH-risk country.
        FAIL if deforestation score > 70 AND HIGH-risk country present.

        Args:
            inputs: Risk factor inputs.
            benchmarks: Country benchmarks.

        Returns:
            Article10CriterionEvaluation for this criterion.
        """
        deforestation_score = _get_dimension_score(
            inputs, RiskDimension.DEFORESTATION
        )

        if deforestation_score is None:
            return Article10CriterionEvaluation(
                criterion=Article10Criterion.PREVALENCE_OF_DEFORESTATION,
                result=CriterionResult.NOT_EVALUATED,
                score=Decimal("0"),
                threshold=_DEFORESTATION_CONCERN_THRESHOLD,
                explanation="No deforestation risk data available",
                evidence=[],
            )

        has_high = _has_high_risk_country(benchmarks)
        fail_threshold = (_DEFORESTATION_CONCERN_THRESHOLD * _FAIL_MULTIPLIER).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if deforestation_score > fail_threshold and has_high:
            result = CriterionResult.FAIL
            explanation = (
                f"Deforestation score {deforestation_score} exceeds fail "
                f"threshold {fail_threshold} with HIGH-risk country present"
            )
        elif deforestation_score > _DEFORESTATION_CONCERN_THRESHOLD or has_high:
            result = CriterionResult.CONCERN
            explanation = (
                f"Deforestation score {deforestation_score} exceeds concern "
                f"threshold {_DEFORESTATION_CONCERN_THRESHOLD} "
                f"or HIGH-risk country present (has_high={has_high})"
            )
        else:
            result = CriterionResult.PASS
            explanation = (
                f"Deforestation score {deforestation_score} within acceptable "
                f"range (threshold={_DEFORESTATION_CONCERN_THRESHOLD})"
            )

        return Article10CriterionEvaluation(
            criterion=Article10Criterion.PREVALENCE_OF_DEFORESTATION,
            result=result,
            score=deforestation_score,
            threshold=_DEFORESTATION_CONCERN_THRESHOLD,
            explanation=explanation,
            evidence=[f"deforestation_score={deforestation_score}"],
        )

    def _evaluate_supply_chain_complexity(
        self,
        inputs: List[RiskFactorInput],
    ) -> Article10CriterionEvaluation:
        """Evaluate criterion (b): Supply chain complexity.

        CONCERN if complexity score > 60.

        Args:
            inputs: Risk factor inputs.

        Returns:
            Article10CriterionEvaluation for this criterion.
        """
        score = _get_dimension_score(
            inputs, RiskDimension.SUPPLY_CHAIN_COMPLEXITY
        )

        if score is None:
            return Article10CriterionEvaluation(
                criterion=Article10Criterion.SUPPLY_CHAIN_COMPLEXITY,
                result=CriterionResult.NOT_EVALUATED,
                score=Decimal("0"),
                threshold=_COMPLEXITY_CONCERN_THRESHOLD,
                explanation="No supply chain complexity data available",
                evidence=[],
            )

        fail_threshold = (_COMPLEXITY_CONCERN_THRESHOLD * _FAIL_MULTIPLIER).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if score > fail_threshold:
            result = CriterionResult.FAIL
            explanation = (
                f"Complexity score {score} exceeds fail threshold {fail_threshold}"
            )
        elif score > _COMPLEXITY_CONCERN_THRESHOLD:
            result = CriterionResult.CONCERN
            explanation = (
                f"Complexity score {score} exceeds concern threshold "
                f"{_COMPLEXITY_CONCERN_THRESHOLD}"
            )
        else:
            result = CriterionResult.PASS
            explanation = (
                f"Complexity score {score} within acceptable range "
                f"(threshold={_COMPLEXITY_CONCERN_THRESHOLD})"
            )

        return Article10CriterionEvaluation(
            criterion=Article10Criterion.SUPPLY_CHAIN_COMPLEXITY,
            result=result,
            score=score,
            threshold=_COMPLEXITY_CONCERN_THRESHOLD,
            explanation=explanation,
            evidence=[f"complexity_score={score}"],
        )

    def _evaluate_mixing_risk(
        self,
        inputs: List[RiskFactorInput],
    ) -> Article10CriterionEvaluation:
        """Evaluate criterion (c): Risk of mixing with unknown-origin products.

        CONCERN if mixing score > 50.

        Args:
            inputs: Risk factor inputs.

        Returns:
            Article10CriterionEvaluation for this criterion.
        """
        score = _get_dimension_score(inputs, RiskDimension.MIXING_RISK)

        if score is None:
            return Article10CriterionEvaluation(
                criterion=Article10Criterion.MIXING_RISK,
                result=CriterionResult.NOT_EVALUATED,
                score=Decimal("0"),
                threshold=_MIXING_CONCERN_THRESHOLD,
                explanation="No mixing risk data available",
                evidence=[],
            )

        fail_threshold = (_MIXING_CONCERN_THRESHOLD * _FAIL_MULTIPLIER).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if score > fail_threshold:
            result = CriterionResult.FAIL
            explanation = f"Mixing risk {score} exceeds fail threshold {fail_threshold}"
        elif score > _MIXING_CONCERN_THRESHOLD:
            result = CriterionResult.CONCERN
            explanation = (
                f"Mixing risk {score} exceeds concern threshold "
                f"{_MIXING_CONCERN_THRESHOLD}"
            )
        else:
            result = CriterionResult.PASS
            explanation = (
                f"Mixing risk {score} within acceptable range "
                f"(threshold={_MIXING_CONCERN_THRESHOLD})"
            )

        return Article10CriterionEvaluation(
            criterion=Article10Criterion.MIXING_RISK,
            result=result,
            score=score,
            threshold=_MIXING_CONCERN_THRESHOLD,
            explanation=explanation,
            evidence=[f"mixing_risk_score={score}"],
        )

    def _evaluate_circumvention_risk(
        self,
        inputs: List[RiskFactorInput],
    ) -> Article10CriterionEvaluation:
        """Evaluate criterion (d): Risk of circumvention.

        CONCERN if circumvention score > 50.

        Args:
            inputs: Risk factor inputs.

        Returns:
            Article10CriterionEvaluation for this criterion.
        """
        score = _get_dimension_score(inputs, RiskDimension.CIRCUMVENTION_RISK)

        if score is None:
            return Article10CriterionEvaluation(
                criterion=Article10Criterion.CIRCUMVENTION_RISK,
                result=CriterionResult.NOT_EVALUATED,
                score=Decimal("0"),
                threshold=_CIRCUMVENTION_CONCERN_THRESHOLD,
                explanation="No circumvention risk data available",
                evidence=[],
            )

        fail_threshold = (
            _CIRCUMVENTION_CONCERN_THRESHOLD * _FAIL_MULTIPLIER
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        if score > fail_threshold:
            result = CriterionResult.FAIL
            explanation = (
                f"Circumvention risk {score} exceeds fail threshold "
                f"{fail_threshold}"
            )
        elif score > _CIRCUMVENTION_CONCERN_THRESHOLD:
            result = CriterionResult.CONCERN
            explanation = (
                f"Circumvention risk {score} exceeds concern threshold "
                f"{_CIRCUMVENTION_CONCERN_THRESHOLD}"
            )
        else:
            result = CriterionResult.PASS
            explanation = (
                f"Circumvention risk {score} within acceptable range "
                f"(threshold={_CIRCUMVENTION_CONCERN_THRESHOLD})"
            )

        return Article10CriterionEvaluation(
            criterion=Article10Criterion.CIRCUMVENTION_RISK,
            result=result,
            score=score,
            threshold=_CIRCUMVENTION_CONCERN_THRESHOLD,
            explanation=explanation,
            evidence=[f"circumvention_score={score}"],
        )

    def _evaluate_country_governance(
        self,
        inputs: List[RiskFactorInput],
        benchmarks: List[CountryBenchmark],
    ) -> Article10CriterionEvaluation:
        """Evaluate criterion (e): Country governance and rule of law.

        CONCERN if corruption score > 55 or any HIGH-risk country.

        Args:
            inputs: Risk factor inputs.
            benchmarks: Country benchmarks.

        Returns:
            Article10CriterionEvaluation for this criterion.
        """
        corruption_score = _get_dimension_score(
            inputs, RiskDimension.CORRUPTION
        )

        if corruption_score is None:
            return Article10CriterionEvaluation(
                criterion=Article10Criterion.COUNTRY_GOVERNANCE,
                result=CriterionResult.NOT_EVALUATED,
                score=Decimal("0"),
                threshold=_GOVERNANCE_CONCERN_THRESHOLD,
                explanation="No governance/corruption data available",
                evidence=[],
            )

        has_high = _has_high_risk_country(benchmarks)
        fail_threshold = (_GOVERNANCE_CONCERN_THRESHOLD * _FAIL_MULTIPLIER).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if corruption_score > fail_threshold and has_high:
            result = CriterionResult.FAIL
            explanation = (
                f"Governance score {corruption_score} exceeds fail threshold "
                f"{fail_threshold} with HIGH-risk country"
            )
        elif corruption_score > _GOVERNANCE_CONCERN_THRESHOLD or has_high:
            result = CriterionResult.CONCERN
            explanation = (
                f"Governance score {corruption_score} exceeds concern threshold "
                f"{_GOVERNANCE_CONCERN_THRESHOLD} or HIGH-risk country present"
            )
        else:
            result = CriterionResult.PASS
            explanation = (
                f"Governance score {corruption_score} within acceptable range "
                f"(threshold={_GOVERNANCE_CONCERN_THRESHOLD})"
            )

        return Article10CriterionEvaluation(
            criterion=Article10Criterion.COUNTRY_GOVERNANCE,
            result=result,
            score=corruption_score,
            threshold=_GOVERNANCE_CONCERN_THRESHOLD,
            explanation=explanation,
            evidence=[
                f"corruption_score={corruption_score}",
                f"has_high_risk_country={has_high}",
            ],
        )

    def _evaluate_supplier_compliance(
        self,
        inputs: List[RiskFactorInput],
    ) -> Article10CriterionEvaluation:
        """Evaluate criterion (f): Supplier compliance history.

        CONCERN if supplier risk score > 55.

        Args:
            inputs: Risk factor inputs.

        Returns:
            Article10CriterionEvaluation for this criterion.
        """
        score = _get_dimension_score(inputs, RiskDimension.SUPPLIER)

        if score is None:
            return Article10CriterionEvaluation(
                criterion=Article10Criterion.SUPPLIER_COMPLIANCE,
                result=CriterionResult.NOT_EVALUATED,
                score=Decimal("0"),
                threshold=_SUPPLIER_CONCERN_THRESHOLD,
                explanation="No supplier compliance data available",
                evidence=[],
            )

        fail_threshold = (_SUPPLIER_CONCERN_THRESHOLD * _FAIL_MULTIPLIER).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if score > fail_threshold:
            result = CriterionResult.FAIL
            explanation = (
                f"Supplier risk {score} exceeds fail threshold {fail_threshold}"
            )
        elif score > _SUPPLIER_CONCERN_THRESHOLD:
            result = CriterionResult.CONCERN
            explanation = (
                f"Supplier risk {score} exceeds concern threshold "
                f"{_SUPPLIER_CONCERN_THRESHOLD}"
            )
        else:
            result = CriterionResult.PASS
            explanation = (
                f"Supplier risk {score} within acceptable range "
                f"(threshold={_SUPPLIER_CONCERN_THRESHOLD})"
            )

        return Article10CriterionEvaluation(
            criterion=Article10Criterion.SUPPLIER_COMPLIANCE,
            result=result,
            score=score,
            threshold=_SUPPLIER_CONCERN_THRESHOLD,
            explanation=explanation,
            evidence=[f"supplier_score={score}"],
        )

    def _evaluate_commodity_risk_profile(
        self,
        inputs: List[RiskFactorInput],
    ) -> Article10CriterionEvaluation:
        """Evaluate criterion (g): Commodity-specific risk profile.

        CONCERN if commodity risk score > 60.

        Args:
            inputs: Risk factor inputs.

        Returns:
            Article10CriterionEvaluation for this criterion.
        """
        score = _get_dimension_score(inputs, RiskDimension.COMMODITY)

        if score is None:
            return Article10CriterionEvaluation(
                criterion=Article10Criterion.COMMODITY_RISK_PROFILE,
                result=CriterionResult.NOT_EVALUATED,
                score=Decimal("0"),
                threshold=_COMMODITY_CONCERN_THRESHOLD,
                explanation="No commodity risk data available",
                evidence=[],
            )

        fail_threshold = (_COMMODITY_CONCERN_THRESHOLD * _FAIL_MULTIPLIER).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if score > fail_threshold:
            result = CriterionResult.FAIL
            explanation = (
                f"Commodity risk {score} exceeds fail threshold {fail_threshold}"
            )
        elif score > _COMMODITY_CONCERN_THRESHOLD:
            result = CriterionResult.CONCERN
            explanation = (
                f"Commodity risk {score} exceeds concern threshold "
                f"{_COMMODITY_CONCERN_THRESHOLD}"
            )
        else:
            result = CriterionResult.PASS
            explanation = (
                f"Commodity risk {score} within acceptable range "
                f"(threshold={_COMMODITY_CONCERN_THRESHOLD})"
            )

        return Article10CriterionEvaluation(
            criterion=Article10Criterion.COMMODITY_RISK_PROFILE,
            result=result,
            score=score,
            threshold=_COMMODITY_CONCERN_THRESHOLD,
            explanation=explanation,
            evidence=[f"commodity_score={score}"],
        )

    def _evaluate_certification_coverage(
        self,
        inputs: List[RiskFactorInput],
    ) -> Article10CriterionEvaluation:
        """Evaluate criterion (h): Certification and verification coverage.

        Uses an inverse heuristic: lower supplier risk implies better
        certification coverage. CONCERN if inferred coverage score > 60.

        Args:
            inputs: Risk factor inputs.

        Returns:
            Article10CriterionEvaluation for this criterion.
        """
        # Certification coverage is inversely derived from supplier risk
        supplier_score = _get_dimension_score(inputs, RiskDimension.SUPPLIER)

        if supplier_score is None:
            return Article10CriterionEvaluation(
                criterion=Article10Criterion.CERTIFICATION_COVERAGE,
                result=CriterionResult.NOT_EVALUATED,
                score=Decimal("0"),
                threshold=_CERTIFICATION_CONCERN_THRESHOLD,
                explanation="No certification coverage data available",
                evidence=[],
            )

        # Higher supplier risk -> higher certification concern
        score = supplier_score

        fail_threshold = (
            _CERTIFICATION_CONCERN_THRESHOLD * _FAIL_MULTIPLIER
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        if score > fail_threshold:
            result = CriterionResult.FAIL
            explanation = (
                f"Certification gap score {score} exceeds fail threshold "
                f"{fail_threshold}"
            )
        elif score > _CERTIFICATION_CONCERN_THRESHOLD:
            result = CriterionResult.CONCERN
            explanation = (
                f"Certification gap score {score} exceeds concern threshold "
                f"{_CERTIFICATION_CONCERN_THRESHOLD}"
            )
        else:
            result = CriterionResult.PASS
            explanation = (
                f"Certification coverage adequate: gap score {score} "
                f"(threshold={_CERTIFICATION_CONCERN_THRESHOLD})"
            )

        return Article10CriterionEvaluation(
            criterion=Article10Criterion.CERTIFICATION_COVERAGE,
            result=result,
            score=score,
            threshold=_CERTIFICATION_CONCERN_THRESHOLD,
            explanation=explanation,
            evidence=[f"certification_gap_score={score}"],
        )

    def _evaluate_deforestation_alerts(
        self,
        inputs: List[RiskFactorInput],
    ) -> Article10CriterionEvaluation:
        """Evaluate criterion (i): Deforestation alerts in sourcing regions.

        CONCERN if deforestation score > 50 (reuses DEFORESTATION dimension
        with alert-specific threshold).

        Args:
            inputs: Risk factor inputs.

        Returns:
            Article10CriterionEvaluation for this criterion.
        """
        score = _get_dimension_score(inputs, RiskDimension.DEFORESTATION)

        if score is None:
            return Article10CriterionEvaluation(
                criterion=Article10Criterion.DEFORESTATION_ALERTS,
                result=CriterionResult.NOT_EVALUATED,
                score=Decimal("0"),
                threshold=_ALERT_CONCERN_THRESHOLD,
                explanation="No deforestation alert data available",
                evidence=[],
            )

        fail_threshold = (_ALERT_CONCERN_THRESHOLD * _FAIL_MULTIPLIER).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if score > fail_threshold:
            result = CriterionResult.FAIL
            explanation = (
                f"Deforestation alert score {score} exceeds fail threshold "
                f"{fail_threshold}"
            )
        elif score > _ALERT_CONCERN_THRESHOLD:
            result = CriterionResult.CONCERN
            explanation = (
                f"Deforestation alert score {score} exceeds concern threshold "
                f"{_ALERT_CONCERN_THRESHOLD}"
            )
        else:
            result = CriterionResult.PASS
            explanation = (
                f"Deforestation alert score {score} within acceptable range "
                f"(threshold={_ALERT_CONCERN_THRESHOLD})"
            )

        return Article10CriterionEvaluation(
            criterion=Article10Criterion.DEFORESTATION_ALERTS,
            result=result,
            score=score,
            threshold=_ALERT_CONCERN_THRESHOLD,
            explanation=explanation,
            evidence=[f"alert_score={score}"],
        )

    def _evaluate_legal_framework(
        self,
        inputs: List[RiskFactorInput],
        benchmarks: List[CountryBenchmark],
    ) -> Article10CriterionEvaluation:
        """Evaluate criterion (j): Legal framework adequacy.

        CONCERN if country risk > 55 or any HIGH-risk country. Uses country
        risk dimension as proxy for legal framework adequacy.

        Args:
            inputs: Risk factor inputs.
            benchmarks: Country benchmarks.

        Returns:
            Article10CriterionEvaluation for this criterion.
        """
        country_score = _get_dimension_score(inputs, RiskDimension.COUNTRY)

        if country_score is None:
            return Article10CriterionEvaluation(
                criterion=Article10Criterion.LEGAL_FRAMEWORK,
                result=CriterionResult.NOT_EVALUATED,
                score=Decimal("0"),
                threshold=_LEGAL_CONCERN_THRESHOLD,
                explanation="No legal framework data available",
                evidence=[],
            )

        has_high = _has_high_risk_country(benchmarks)
        fail_threshold = (_LEGAL_CONCERN_THRESHOLD * _FAIL_MULTIPLIER).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if country_score > fail_threshold and has_high:
            result = CriterionResult.FAIL
            explanation = (
                f"Legal framework score {country_score} exceeds fail threshold "
                f"{fail_threshold} with HIGH-risk country"
            )
        elif country_score > _LEGAL_CONCERN_THRESHOLD or has_high:
            result = CriterionResult.CONCERN
            explanation = (
                f"Legal framework score {country_score} exceeds concern "
                f"threshold {_LEGAL_CONCERN_THRESHOLD} "
                f"or HIGH-risk country present"
            )
        else:
            result = CriterionResult.PASS
            explanation = (
                f"Legal framework score {country_score} within acceptable "
                f"range (threshold={_LEGAL_CONCERN_THRESHOLD})"
            )

        return Article10CriterionEvaluation(
            criterion=Article10Criterion.LEGAL_FRAMEWORK,
            result=result,
            score=country_score,
            threshold=_LEGAL_CONCERN_THRESHOLD,
            explanation=explanation,
            evidence=[
                f"country_score={country_score}",
                f"has_high_risk_country={has_high}",
            ],
        )
