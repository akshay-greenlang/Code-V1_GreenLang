# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-028 RiskReportGenerator.

Tests report generation, recommendation generation, DDS readiness
validation, report validation, and report statistics.
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
)
from greenlang.agents.eudr.risk_assessment_engine.models import (
    Article10CriteriaResult,
    Article10CriterionEvaluation,
    Article10Criterion,
    CompositeRiskScore,
    CountryBenchmark,
    CountryBenchmarkLevel,
    CriterionResult,
    DimensionScore,
    EUDRCommodity,
    RiskAssessmentOperation,
    RiskAssessmentStatus,
    RiskDimension,
    RiskLevel,
    RiskOverride,
    OverrideReason,
    SimplifiedDDEligibility,
    SourceAgent,
)


def _make_generator():
    """Instantiate RiskReportGenerator with mocked dependencies."""
    from greenlang.agents.eudr.risk_assessment_engine.risk_report_generator import (
        RiskReportGenerator,
    )
    cfg = MagicMock(spec=RiskAssessmentEngineConfig)
    cfg.report_format = "json"
    cfg.include_factor_decomposition = True
    cfg.include_trend_data = True
    with patch(
        "greenlang.agents.eudr.risk_assessment_engine.risk_report_generator.record_report_generation"
    ):
        return RiskReportGenerator(config=cfg)


def _make_operation() -> RiskAssessmentOperation:
    return RiskAssessmentOperation(
        operation_id=f"OP-{uuid.uuid4().hex[:8]}",
        operator_id="OPERATOR-001",
        commodity=EUDRCommodity.COCOA,
        status=RiskAssessmentStatus.COMPLETED,
    )


def _make_composite(
    score: Decimal = Decimal("45"),
    level: RiskLevel = RiskLevel.STANDARD,
    num_dims: int = 5,
) -> CompositeRiskScore:
    dims = []
    for i in range(num_dims):
        dim = list(RiskDimension)[i % len(RiskDimension)]
        dims.append(DimensionScore(
            dimension=dim,
            weighted_score=Decimal("5.00"),
            raw_score=Decimal("50"),
            weight=Decimal("0.10"),
            confidence=Decimal("0.90"),
            source_agent=SourceAgent.EUDR_016_COUNTRY,
        ))
    return CompositeRiskScore(
        overall_score=score,
        risk_level=level,
        dimension_scores=dims,
        provenance_hash="a" * 64,
    )


def _make_article10(
    num_evaluations: int = 10,
    concern_count: int = 0,
) -> Article10CriteriaResult:
    evals = []
    criteria_list = list(Article10Criterion)
    for i in range(num_evaluations):
        criterion = criteria_list[i % len(criteria_list)]
        result = CriterionResult.CONCERN if i < concern_count else CriterionResult.PASS
        evals.append(Article10CriterionEvaluation(
            criterion=criterion,
            result=result,
            score=Decimal("30"),
        ))
    r = MagicMock(spec=Article10CriteriaResult)
    r.evaluations = evals
    r.total_evaluated = num_evaluations
    r.pass_count = num_evaluations - concern_count
    r.concern_count = concern_count
    r.fail_count = 0
    r.provenance_hash = "b" * 64
    return r


def _make_benchmarks() -> list:
    return [
        CountryBenchmark(
            country_code="GH",
            benchmark_level=CountryBenchmarkLevel.HIGH,
        ),
    ]


def _make_simplified_dd(eligible: bool = False) -> SimplifiedDDEligibility:
    sdd = MagicMock(spec=SimplifiedDDEligibility)
    sdd.eligible = eligible
    sdd.provenance_hash = "c" * 64
    return sdd


class TestGenerateReport:
    """Test report generation."""

    def test_generate_report_basic(self):
        """Basic report generation should produce a valid report."""
        gen = _make_generator()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_report_generator.record_report_generation"
        ):
            report = gen.generate_report(
                operation=_make_operation(),
                composite=_make_composite(),
                article10=_make_article10(),
                benchmarks=_make_benchmarks(),
                simplified_dd=_make_simplified_dd(),
            )

        assert report is not None
        assert report.report_id.startswith("RAR-")
        assert report.risk_level == RiskLevel.STANDARD
        assert len(report.recommendations) > 0
        assert report.provenance_hash != ""
        assert len(report.provenance_hash) == 64

    def test_generate_report_with_overrides(self):
        """Report with overrides should include override note in recommendations."""
        gen = _make_generator()
        overrides = [
            RiskOverride(
                override_id="OVR-001",
                assessment_id="ASM-001",
                original_score=Decimal("70"),
                overridden_score=Decimal("50"),
                original_level=RiskLevel.HIGH,
                overridden_level=RiskLevel.STANDARD,
                reason=OverrideReason.EXPERT_JUDGMENT,
                justification="Test override",
            ),
        ]
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_report_generator.record_report_generation"
        ):
            report = gen.generate_report(
                operation=_make_operation(),
                composite=_make_composite(),
                article10=_make_article10(),
                benchmarks=_make_benchmarks(),
                simplified_dd=_make_simplified_dd(),
                overrides=overrides,
            )

        override_recs = [r for r in report.recommendations if "override" in r.lower()]
        assert len(override_recs) >= 1


class TestGenerateRecommendations:
    """Test recommendation generation by risk level."""

    def test_generate_recommendations_negligible(self):
        gen = _make_generator()
        article10 = _make_article10(concern_count=0)
        recs = gen._generate_recommendations(RiskLevel.NEGLIGIBLE, article10)
        assert len(recs) >= 1
        assert any("monitoring" in r.lower() or "maintain" in r.lower() for r in recs)

    def test_generate_recommendations_critical(self):
        gen = _make_generator()
        article10 = _make_article10(concern_count=5)
        recs = gen._generate_recommendations(RiskLevel.CRITICAL, article10)
        assert len(recs) >= 5
        assert any("suspend" in r.lower() or "audit" in r.lower() for r in recs)


class TestDDSReadiness:
    """Test DDS readiness checks."""

    def test_check_dds_readiness_complete(self):
        """Report with all required sections -> DDS ready."""
        gen = _make_generator()
        composite = _make_composite(num_dims=5)
        article10 = _make_article10(num_evaluations=10)
        benchmarks = _make_benchmarks()
        result = gen._check_dds_readiness_internal(composite, article10, benchmarks)
        assert result is True

    def test_check_dds_readiness_incomplete_dims(self):
        """Fewer than 3 dimensions -> not DDS ready."""
        gen = _make_generator()
        composite = _make_composite(num_dims=2)
        article10 = _make_article10(num_evaluations=10)
        benchmarks = _make_benchmarks()
        result = gen._check_dds_readiness_internal(composite, article10, benchmarks)
        assert result is False

    def test_check_dds_readiness_no_benchmarks(self):
        """No benchmarks -> not DDS ready."""
        gen = _make_generator()
        composite = _make_composite(num_dims=5)
        article10 = _make_article10(num_evaluations=10)
        result = gen._check_dds_readiness_internal(composite, article10, [])
        assert result is False


class TestValidateReport:
    """Test report validation."""

    def test_validate_report_valid(self):
        """A fully populated report should pass all validation checks."""
        gen = _make_generator()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_report_generator.record_report_generation"
        ):
            report = gen.generate_report(
                operation=_make_operation(),
                composite=_make_composite(),
                article10=_make_article10(),
                benchmarks=_make_benchmarks(),
                simplified_dd=_make_simplified_dd(),
            )
        result = gen.validate_report(report)
        assert result["is_valid"] is True
        assert result["checks_passed"] == result["total_checks"]

    def test_validate_report_missing_composite(self):
        """Report without composite score should fail validation."""
        gen = _make_generator()
        report = MagicMock()
        report.composite_score = None
        report.article10_result = None
        report.country_benchmarks = []
        report.simplified_dd_eligibility = None
        report.risk_level = None
        report.provenance_hash = ""
        report.recommendations = []
        report.report_id = ""
        report.operation = None
        report.generated_at = None
        result = gen.validate_report(report)
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0


class TestReportStats:
    """Test report generator statistics."""

    def test_report_stats(self):
        gen = _make_generator()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_report_generator.record_report_generation"
        ):
            gen.generate_report(
                operation=_make_operation(),
                composite=_make_composite(),
                article10=_make_article10(),
                benchmarks=_make_benchmarks(),
                simplified_dd=_make_simplified_dd(),
            )

        stats = gen.get_report_stats()
        assert stats["total_reports"] >= 1
        assert "dds_ready_count" in stats
        assert "dds_ready_pct" in stats
        assert "total_recommendations" in stats
