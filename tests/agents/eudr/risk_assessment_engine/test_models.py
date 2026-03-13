# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-028 models.py

Tests all 10 enums, 15 Pydantic models, and module-level constants.
Validates field defaults, Decimal precision, computed values, and enum membership.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.risk_assessment_engine.models import (
    Article10Criterion,
    Article10CriteriaResult,
    Article10CriterionEvaluation,
    CompositeRiskScore,
    CountryBenchmark,
    CountryBenchmarkLevel,
    CriterionResult,
    DEFAULT_WEIGHTS,
    DimensionScore,
    EUDRCommodity,
    HealthResponse,
    OverrideReason,
    COUNTRY_BENCHMARK_MULTIPLIERS,
    RISK_THRESHOLDS,
    RiskAssessmentOperation,
    RiskAssessmentStatus,
    RiskDimension,
    RiskFactorInput,
    RiskLevel,
    RiskOverride,
    RiskTrendPoint,
    RiskTrendAnalysis,
    SimplifiedDDEligibility,
    SourceAgent,
    SUPPORTED_COMMODITIES,
    TrendDirection,
    VERSION,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    """Verify all 10 enum classes have expected members."""

    def test_risk_dimension_values(self):
        assert len(RiskDimension) == 8
        expected = {
            "country", "commodity", "supplier", "deforestation",
            "corruption", "supply_chain_complexity", "mixing_risk",
            "circumvention_risk",
        }
        assert {d.value for d in RiskDimension} == expected

    def test_risk_level_values(self):
        assert len(RiskLevel) == 5
        expected = {"negligible", "low", "standard", "high", "critical"}
        assert {l.value for l in RiskLevel} == expected

    def test_risk_assessment_status_values(self):
        assert len(RiskAssessmentStatus) == 7
        expected = {
            "initiated", "aggregating", "evaluating", "classifying",
            "completed", "failed", "cancelled",
        }
        assert {s.value for s in RiskAssessmentStatus} == expected

    def test_country_benchmark_level_values(self):
        assert len(CountryBenchmarkLevel) == 3
        expected = {"low", "standard", "high"}
        assert {l.value for l in CountryBenchmarkLevel} == expected

    def test_article10_criterion_values(self):
        assert len(Article10Criterion) == 10
        expected = {
            "prevalence_of_deforestation", "supply_chain_complexity",
            "mixing_risk", "circumvention_risk", "country_governance",
            "supplier_compliance", "commodity_risk_profile",
            "certification_coverage", "deforestation_alerts",
            "legal_framework",
        }
        assert {c.value for c in Article10Criterion} == expected

    def test_criterion_result_values(self):
        assert len(CriterionResult) == 4
        expected = {"pass", "concern", "fail", "not_evaluated"}
        assert {r.value for r in CriterionResult} == expected

    def test_trend_direction_values(self):
        assert len(TrendDirection) == 4
        expected = {"improving", "stable", "degrading", "insufficient_data"}
        assert {d.value for d in TrendDirection} == expected

    def test_override_reason_values(self):
        assert len(OverrideReason) == 5
        expected = {
            "expert_judgment", "new_evidence", "regulatory_change",
            "data_correction", "mitigating_factors",
        }
        assert {r.value for r in OverrideReason} == expected

    def test_eudr_commodity_values(self):
        assert len(EUDRCommodity) == 7
        expected = {
            "cattle", "cocoa", "coffee", "oil_palm", "rubber",
            "soya", "wood",
        }
        assert {c.value for c in EUDRCommodity} == expected

    def test_source_agent_values(self):
        assert len(SourceAgent) == 6
        expected = {
            "eudr_016_country_risk", "eudr_017_supplier_risk",
            "eudr_018_commodity_risk", "eudr_019_corruption_index",
            "eudr_020_deforestation_alert", "eudr_028_derived",
        }
        assert {a.value for a in SourceAgent} == expected


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestRiskFactorInput:
    """Verify RiskFactorInput model behaviour."""

    def test_risk_factor_input_defaults(self):
        fi = RiskFactorInput(
            source_agent=SourceAgent.EUDR_016_COUNTRY,
            dimension=RiskDimension.COUNTRY,
            raw_score=Decimal("50"),
            confidence=Decimal("0.90"),
        )
        assert fi.raw_score == Decimal("50")
        assert fi.confidence == Decimal("0.90")
        assert fi.metadata == {}
        assert fi.provenance_hash == ""
        assert fi.timestamp is not None

    def test_decimal_precision_in_models(self):
        fi = RiskFactorInput(
            source_agent=SourceAgent.EUDR_016_COUNTRY,
            dimension=RiskDimension.COUNTRY,
            raw_score=Decimal("99.99"),
            confidence=Decimal("0.999"),
        )
        assert fi.raw_score == Decimal("99.99")
        assert fi.confidence == Decimal("0.999")


class TestDimensionScore:
    """Verify DimensionScore model."""

    def test_dimension_score_creation(self):
        ds = DimensionScore(
            dimension=RiskDimension.COUNTRY,
            weighted_score=Decimal("9.00"),
            raw_score=Decimal("45.00"),
            weight=Decimal("0.20"),
            confidence=Decimal("0.90"),
            source_agent=SourceAgent.EUDR_016_COUNTRY,
        )
        assert ds.dimension == RiskDimension.COUNTRY
        assert ds.weighted_score == Decimal("9.00")
        assert ds.explanation == ""


class TestCompositeRiskScore:
    """Verify CompositeRiskScore model."""

    def test_composite_risk_score_defaults(self):
        crs = CompositeRiskScore(
            overall_score=Decimal("50"),
            risk_level=RiskLevel.STANDARD,
        )
        assert crs.overall_score == Decimal("50")
        assert crs.risk_level == RiskLevel.STANDARD
        assert crs.dimension_scores == []
        assert crs.total_weight == Decimal("1.00")
        assert crs.effective_confidence == Decimal("0")
        assert crs.country_benchmark_applied is False
        assert crs.benchmark_multiplier is None
        assert crs.provenance_hash == ""


class TestArticle10CriterionEvaluation:
    """Verify Article10CriterionEvaluation model."""

    def test_article10_criterion_evaluation(self):
        ev = Article10CriterionEvaluation(
            criterion=Article10Criterion.PREVALENCE_OF_DEFORESTATION,
            result=CriterionResult.PASS,
            score=Decimal("30"),
        )
        assert ev.criterion == Article10Criterion.PREVALENCE_OF_DEFORESTATION
        assert ev.result == CriterionResult.PASS
        assert ev.evidence_summary == ""
        assert ev.data_sources == []


class TestArticle10CriteriaResult:
    """Verify Article10CriteriaResult model."""

    def test_article10_criteria_result_defaults(self):
        result = Article10CriteriaResult()
        assert result.evaluations == []
        assert result.overall_concern_count == 0
        assert result.criteria_evaluated == 0
        assert result.criteria_passed == 0
        assert result.criteria_with_concerns == 0


class TestCountryBenchmark:
    """Verify CountryBenchmark model."""

    def test_country_benchmark_creation(self):
        cb = CountryBenchmark(
            country_code="DE",
            benchmark_level=CountryBenchmarkLevel.LOW,
        )
        assert cb.country_code == "DE"
        assert cb.benchmark_level == CountryBenchmarkLevel.LOW
        assert cb.source == ""
        assert cb.governance_score == Decimal("0")
        assert cb.deforestation_rate == Decimal("0")
        assert cb.confidence == Decimal("0")


class TestSimplifiedDDEligibility:
    """Verify SimplifiedDDEligibility model."""

    def test_simplified_dd_eligibility(self):
        sdd = SimplifiedDDEligibility()
        assert sdd.is_eligible is False
        assert sdd.reasons == []
        assert sdd.country_benchmarks == []
        assert sdd.composite_score is None
        assert sdd.all_countries_low is False


class TestRiskTrendPoint:
    """Verify RiskTrendPoint model."""

    def test_risk_trend_point(self):
        now = datetime.now(timezone.utc)
        pt = RiskTrendPoint(
            assessment_date=now,
            composite_score=Decimal("45"),
            risk_level=RiskLevel.STANDARD,
        )
        assert pt.composite_score == Decimal("45")
        assert pt.key_changes == []


class TestRiskTrendAnalysis:
    """Verify RiskTrendAnalysis model."""

    def test_risk_trend_analysis(self):
        rta = RiskTrendAnalysis(
            operator_id="OP-001",
            commodity=EUDRCommodity.COCOA,
        )
        assert rta.direction == TrendDirection.INSUFFICIENT_DATA
        assert rta.average_score == Decimal("0")
        assert rta.score_change_30d is None
        assert rta.score_change_90d is None
        assert rta.score_change_365d is None
        assert rta.trend_points == []


class TestRiskOverride:
    """Verify RiskOverride model."""

    def test_risk_override_creation(self):
        ro = RiskOverride(
            override_id="OVR-001",
            assessment_id="ASM-001",
            original_score=Decimal("75"),
            overridden_score=Decimal("55"),
            original_level=RiskLevel.HIGH,
            overridden_level=RiskLevel.STANDARD,
            reason=OverrideReason.EXPERT_JUDGMENT,
            justification="Mitigating evidence found",
        )
        assert ro.override_id == "OVR-001"
        assert ro.original_score == Decimal("75")
        assert ro.overridden_score == Decimal("55")
        assert ro.overridden_by == ""
        assert ro.approved_by is None
        assert ro.valid_until is None


class TestRiskAssessmentOperation:
    """Verify RiskAssessmentOperation model."""

    def test_risk_assessment_operation_defaults(self):
        op = RiskAssessmentOperation(
            operation_id="OP-001",
            operator_id="OPERATOR-001",
            commodity=EUDRCommodity.COCOA,
        )
        assert op.status == RiskAssessmentStatus.INITIATED
        assert op.risk_factor_inputs == []
        assert op.composite_score is None
        assert op.risk_level is None
        assert op.article10_result is None
        assert op.report_id is None
        assert op.completed_at is None
        assert op.duration_ms is None
        assert op.provenance_hash == ""
        assert op.workflow_id is None


class TestHealthResponse:
    """Verify HealthResponse model."""

    def test_health_response(self):
        hr = HealthResponse()
        assert hr.status == "healthy"
        assert hr.version == VERSION
        assert hr.agent_id == "AGENT-EUDR-028"
        assert hr.active_assessments == 0
        assert hr.upstream_agents_status == {}
        assert hr.database_connected is False
        assert hr.cache_connected is False


class TestRiskAssessmentReport:
    """Verify RiskAssessmentReport model (minimal)."""

    def test_risk_assessment_report(self, sample_composite_score):
        from greenlang.agents.eudr.risk_assessment_engine.models import (
            RiskAssessmentReport,
        )
        rr = RiskAssessmentReport(
            report_id="RPT-001",
            assessment_id="ASM-001",
            operator_id="OP-001",
            commodity=EUDRCommodity.COFFEE,
            composite_score=sample_composite_score,
            risk_level=RiskLevel.STANDARD,
        )
        assert rr.report_id == "RPT-001"
        assert rr.dds_ready is False
        assert rr.recommendations == []
        assert rr.provenance_hash == ""
        assert rr.overrides == []


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants."""

    def test_default_weights_constant(self):
        assert len(DEFAULT_WEIGHTS) == 8
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == Decimal("1.00"), f"Weights sum to {total}"

    def test_risk_thresholds_constant(self):
        assert len(RISK_THRESHOLDS) == 5
        assert RISK_THRESHOLDS[RiskLevel.NEGLIGIBLE] == Decimal("15")
        assert RISK_THRESHOLDS[RiskLevel.LOW] == Decimal("30")
        assert RISK_THRESHOLDS[RiskLevel.STANDARD] == Decimal("60")
        assert RISK_THRESHOLDS[RiskLevel.HIGH] == Decimal("80")
        assert RISK_THRESHOLDS[RiskLevel.CRITICAL] == Decimal("100")

    def test_country_benchmark_multipliers_constant(self):
        assert COUNTRY_BENCHMARK_MULTIPLIERS[CountryBenchmarkLevel.LOW] == Decimal("0.70")
        assert COUNTRY_BENCHMARK_MULTIPLIERS[CountryBenchmarkLevel.STANDARD] == Decimal("1.00")
        assert COUNTRY_BENCHMARK_MULTIPLIERS[CountryBenchmarkLevel.HIGH] == Decimal("1.50")

    def test_supported_commodities_constant(self):
        assert len(SUPPORTED_COMMODITIES) == 7
        for c in EUDRCommodity:
            assert c.value in SUPPORTED_COMMODITIES
