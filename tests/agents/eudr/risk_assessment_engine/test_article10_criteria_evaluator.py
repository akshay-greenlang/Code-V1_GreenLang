# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-028 Article10CriteriaEvaluator.

Tests evaluation of all 10 Article 10(2) criteria with deterministic
thresholds, summary statistics, and edge cases for PASS/CONCERN/FAIL.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
)
from greenlang.agents.eudr.risk_assessment_engine.models import (
    Article10Criterion,
    CompositeRiskScore,
    CountryBenchmark,
    CountryBenchmarkLevel,
    CriterionResult,
    RiskDimension,
    RiskFactorInput,
    RiskLevel,
    SourceAgent,
)


def _make_evaluator():
    """Instantiate Article10CriteriaEvaluator with mocked dependencies."""
    from greenlang.agents.eudr.risk_assessment_engine.article10_criteria_evaluator import (
        Article10CriteriaEvaluator,
    )
    cfg = MagicMock(spec=RiskAssessmentEngineConfig)
    with patch(
        "greenlang.agents.eudr.risk_assessment_engine.article10_criteria_evaluator.record_criteria_evaluation"
    ):
        return Article10CriteriaEvaluator(config=cfg)


def _make_input(dimension: RiskDimension, score: Decimal) -> RiskFactorInput:
    """Create a minimal RiskFactorInput."""
    return RiskFactorInput(
        source_agent=SourceAgent.EUDR_016_COUNTRY,
        dimension=dimension,
        raw_score=score,
        confidence=Decimal("0.90"),
    )


def _make_composite(score: Decimal) -> CompositeRiskScore:
    """Create a minimal CompositeRiskScore."""
    return CompositeRiskScore(
        overall_score=score,
        risk_level=RiskLevel.STANDARD,
        provenance_hash="a" * 64,
    )


def _make_low_benchmark() -> CountryBenchmark:
    return CountryBenchmark(
        country_code="DE",
        benchmark_level=CountryBenchmarkLevel.LOW,
    )


def _make_high_benchmark() -> CountryBenchmark:
    return CountryBenchmark(
        country_code="BR",
        benchmark_level=CountryBenchmarkLevel.HIGH,
    )


class TestEvaluateAllCriteria:
    """Test evaluate_all_criteria returns 10 evaluations."""

    def test_evaluate_all_criteria_returns_10_evaluations(self):
        """All 10 criteria should be evaluated."""
        evaluator = _make_evaluator()
        inputs = [
            _make_input(RiskDimension.DEFORESTATION, Decimal("30")),
            _make_input(RiskDimension.SUPPLY_CHAIN_COMPLEXITY, Decimal("25")),
            _make_input(RiskDimension.MIXING_RISK, Decimal("20")),
            _make_input(RiskDimension.CIRCUMVENTION_RISK, Decimal("15")),
            _make_input(RiskDimension.CORRUPTION, Decimal("30")),
            _make_input(RiskDimension.SUPPLIER, Decimal("25")),
            _make_input(RiskDimension.COMMODITY, Decimal("35")),
            _make_input(RiskDimension.COUNTRY, Decimal("20")),
        ]
        benchmarks = [_make_low_benchmark()]
        composite = _make_composite(Decimal("30"))

        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.article10_criteria_evaluator.record_criteria_evaluation"
        ):
            result = evaluator.evaluate_all_criteria(
                factor_inputs=inputs,
                country_benchmarks=benchmarks,
                composite_score=composite,
            )

        assert len(result.evaluations) == 10
        assert result.total_evaluated > 0


class TestPrevalenceOfDeforestation:
    """Test criterion (a): prevalence of deforestation."""

    def test_evaluate_prevalence_of_deforestation_pass(self):
        """Low deforestation score with LOW benchmark -> PASS."""
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.DEFORESTATION, Decimal("30"))]
        benchmarks = [_make_low_benchmark()]
        ev = evaluator._evaluate_prevalence_of_deforestation(inputs, benchmarks)
        assert ev.result == CriterionResult.PASS

    def test_evaluate_prevalence_of_deforestation_concern(self):
        """High deforestation score -> CONCERN."""
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.DEFORESTATION, Decimal("55"))]
        benchmarks = [_make_low_benchmark()]
        ev = evaluator._evaluate_prevalence_of_deforestation(inputs, benchmarks)
        assert ev.result == CriterionResult.CONCERN

    def test_evaluate_prevalence_of_deforestation_concern_high_country(self):
        """Any HIGH-risk country -> CONCERN even with low score."""
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.DEFORESTATION, Decimal("30"))]
        benchmarks = [_make_high_benchmark()]
        ev = evaluator._evaluate_prevalence_of_deforestation(inputs, benchmarks)
        assert ev.result == CriterionResult.CONCERN


class TestSupplyChainComplexity:
    """Test criterion (b): supply chain complexity."""

    def test_evaluate_supply_chain_complexity_pass(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.SUPPLY_CHAIN_COMPLEXITY, Decimal("40"))]
        ev = evaluator._evaluate_supply_chain_complexity(inputs)
        assert ev.result == CriterionResult.PASS

    def test_evaluate_supply_chain_complexity_concern(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.SUPPLY_CHAIN_COMPLEXITY, Decimal("65"))]
        ev = evaluator._evaluate_supply_chain_complexity(inputs)
        assert ev.result == CriterionResult.CONCERN


class TestMixingRisk:
    """Test criterion (c): mixing risk."""

    def test_evaluate_mixing_risk_pass(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.MIXING_RISK, Decimal("30"))]
        ev = evaluator._evaluate_mixing_risk(inputs)
        assert ev.result == CriterionResult.PASS

    def test_evaluate_mixing_risk_concern(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.MIXING_RISK, Decimal("55"))]
        ev = evaluator._evaluate_mixing_risk(inputs)
        assert ev.result == CriterionResult.CONCERN


class TestCircumventionRisk:
    """Test criterion (d): circumvention risk."""

    def test_evaluate_circumvention_risk_pass(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.CIRCUMVENTION_RISK, Decimal("30"))]
        ev = evaluator._evaluate_circumvention_risk(inputs)
        assert ev.result == CriterionResult.PASS

    def test_evaluate_circumvention_risk_concern(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.CIRCUMVENTION_RISK, Decimal("55"))]
        ev = evaluator._evaluate_circumvention_risk(inputs)
        assert ev.result == CriterionResult.CONCERN


class TestCountryGovernance:
    """Test criterion (e): country governance."""

    def test_evaluate_country_governance_pass(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.CORRUPTION, Decimal("30"))]
        benchmarks = [_make_low_benchmark()]
        ev = evaluator._evaluate_country_governance(inputs, benchmarks)
        assert ev.result == CriterionResult.PASS

    def test_evaluate_country_governance_concern(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.CORRUPTION, Decimal("60"))]
        benchmarks = [_make_low_benchmark()]
        ev = evaluator._evaluate_country_governance(inputs, benchmarks)
        assert ev.result == CriterionResult.CONCERN


class TestSupplierCompliance:
    """Test criterion (f): supplier compliance."""

    def test_evaluate_supplier_compliance_pass(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.SUPPLIER, Decimal("30"))]
        ev = evaluator._evaluate_supplier_compliance(inputs)
        assert ev.result == CriterionResult.PASS

    def test_evaluate_supplier_compliance_concern(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.SUPPLIER, Decimal("60"))]
        ev = evaluator._evaluate_supplier_compliance(inputs)
        assert ev.result == CriterionResult.CONCERN


class TestCommodityRiskProfile:
    """Test criterion (g): commodity risk profile."""

    def test_evaluate_commodity_risk_profile_pass(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.COMMODITY, Decimal("40"))]
        ev = evaluator._evaluate_commodity_risk_profile(inputs)
        assert ev.result == CriterionResult.PASS

    def test_evaluate_commodity_risk_profile_concern(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.COMMODITY, Decimal("65"))]
        ev = evaluator._evaluate_commodity_risk_profile(inputs)
        assert ev.result == CriterionResult.CONCERN


class TestCertificationCoverage:
    """Test criterion (h): certification coverage."""

    def test_evaluate_certification_coverage_pass(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.SUPPLIER, Decimal("30"))]
        ev = evaluator._evaluate_certification_coverage(inputs)
        assert ev.result == CriterionResult.PASS

    def test_evaluate_certification_coverage_concern(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.SUPPLIER, Decimal("65"))]
        ev = evaluator._evaluate_certification_coverage(inputs)
        assert ev.result == CriterionResult.CONCERN


class TestDeforestationAlerts:
    """Test criterion (i): deforestation alerts."""

    def test_evaluate_deforestation_alerts_pass(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.DEFORESTATION, Decimal("30"))]
        ev = evaluator._evaluate_deforestation_alerts(inputs)
        assert ev.result == CriterionResult.PASS

    def test_evaluate_deforestation_alerts_concern(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.DEFORESTATION, Decimal("55"))]
        ev = evaluator._evaluate_deforestation_alerts(inputs)
        assert ev.result == CriterionResult.CONCERN


class TestLegalFramework:
    """Test criterion (j): legal framework."""

    def test_evaluate_legal_framework_pass(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.COUNTRY, Decimal("30"))]
        benchmarks = [_make_low_benchmark()]
        ev = evaluator._evaluate_legal_framework(inputs, benchmarks)
        assert ev.result == CriterionResult.PASS

    def test_evaluate_legal_framework_concern_high_country(self):
        evaluator = _make_evaluator()
        inputs = [_make_input(RiskDimension.COUNTRY, Decimal("30"))]
        benchmarks = [_make_high_benchmark()]
        ev = evaluator._evaluate_legal_framework(inputs, benchmarks)
        assert ev.result == CriterionResult.CONCERN


class TestAllPassScenario:
    """Test scenario where all criteria pass."""

    def test_all_criteria_pass_scenario(self):
        """Low scores across all dimensions should result in all PASS."""
        evaluator = _make_evaluator()
        inputs = [
            _make_input(RiskDimension.DEFORESTATION, Decimal("20")),
            _make_input(RiskDimension.SUPPLY_CHAIN_COMPLEXITY, Decimal("20")),
            _make_input(RiskDimension.MIXING_RISK, Decimal("20")),
            _make_input(RiskDimension.CIRCUMVENTION_RISK, Decimal("20")),
            _make_input(RiskDimension.CORRUPTION, Decimal("20")),
            _make_input(RiskDimension.SUPPLIER, Decimal("20")),
            _make_input(RiskDimension.COMMODITY, Decimal("20")),
            _make_input(RiskDimension.COUNTRY, Decimal("20")),
        ]
        benchmarks = [_make_low_benchmark()]
        composite = _make_composite(Decimal("20"))

        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.article10_criteria_evaluator.record_criteria_evaluation"
        ):
            result = evaluator.evaluate_all_criteria(inputs, benchmarks, composite)

        pass_count = sum(
            1 for e in result.evaluations if e.result == CriterionResult.PASS
        )
        assert pass_count == 10


class TestMultipleConcernsScenario:
    """Test scenario with multiple CONCERN results."""

    def test_multiple_concerns_scenario(self):
        """High scores across many dimensions should produce multiple CONCERNs."""
        evaluator = _make_evaluator()
        inputs = [
            _make_input(RiskDimension.DEFORESTATION, Decimal("65")),
            _make_input(RiskDimension.SUPPLY_CHAIN_COMPLEXITY, Decimal("70")),
            _make_input(RiskDimension.MIXING_RISK, Decimal("55")),
            _make_input(RiskDimension.CIRCUMVENTION_RISK, Decimal("55")),
            _make_input(RiskDimension.CORRUPTION, Decimal("60")),
            _make_input(RiskDimension.SUPPLIER, Decimal("60")),
            _make_input(RiskDimension.COMMODITY, Decimal("65")),
            _make_input(RiskDimension.COUNTRY, Decimal("60")),
        ]
        benchmarks = [_make_high_benchmark()]
        composite = _make_composite(Decimal("60"))

        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.article10_criteria_evaluator.record_criteria_evaluation"
        ):
            result = evaluator.evaluate_all_criteria(inputs, benchmarks, composite)

        concern_count = sum(
            1 for e in result.evaluations
            if e.result in (CriterionResult.CONCERN, CriterionResult.FAIL)
        )
        assert concern_count >= 5


class TestEvaluationStats:
    """Test evaluation statistics."""

    def test_evaluation_stats(self):
        evaluator = _make_evaluator()
        inputs = [
            _make_input(RiskDimension.DEFORESTATION, Decimal("30")),
            _make_input(RiskDimension.COUNTRY, Decimal("20")),
            _make_input(RiskDimension.SUPPLIER, Decimal("20")),
            _make_input(RiskDimension.COMMODITY, Decimal("20")),
            _make_input(RiskDimension.CORRUPTION, Decimal("20")),
            _make_input(RiskDimension.SUPPLY_CHAIN_COMPLEXITY, Decimal("20")),
            _make_input(RiskDimension.MIXING_RISK, Decimal("20")),
            _make_input(RiskDimension.CIRCUMVENTION_RISK, Decimal("20")),
        ]
        benchmarks = [_make_low_benchmark()]
        composite = _make_composite(Decimal("25"))

        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.article10_criteria_evaluator.record_criteria_evaluation"
        ):
            evaluator.evaluate_all_criteria(inputs, benchmarks, composite)

        stats = evaluator.get_evaluation_stats()
        assert stats["total_evaluations"] >= 1
        assert "total_concerns" in stats
        assert "total_fails" in stats
