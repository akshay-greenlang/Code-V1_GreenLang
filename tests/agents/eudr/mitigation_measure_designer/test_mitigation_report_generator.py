# -*- coding: utf-8 -*-
"""
Unit tests for MitigationReportGenerator - AGENT-EUDR-029

Tests report generation, DDS formatting, verification inclusion,
recommendations, provenance hash, regulatory section, measures
summary, and cumulative reduction calculation.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.mitigation_measure_designer.config import (
    MitigationMeasureDesignerConfig,
)
from greenlang.agents.eudr.mitigation_measure_designer.mitigation_report_generator import (
    MitigationReportGenerator,
    _ARTICLE_REFERENCES,
    _RISK_LEVEL_LABELS,
    _VERIFICATION_LABELS,
)
from greenlang.agents.eudr.mitigation_measure_designer.models import (
    AGENT_ID,
    AGENT_VERSION,
    Article11Category,
    MeasurePriority,
    MeasureStatus,
    MeasureSummary,
    MitigationMeasure,
    MitigationReport,
    MitigationStrategy,
    RiskDimension,
    RiskLevel,
    VerificationReport,
    VerificationResult,
    WorkflowStatus,
)
from greenlang.agents.eudr.mitigation_measure_designer.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return MitigationMeasureDesignerConfig()


@pytest.fixture
def generator(config):
    return MitigationReportGenerator(
        config=config, provenance=ProvenanceTracker(),
    )


class TestGenerateReport:
    """Test generate_report method."""

    def test_returns_mitigation_report(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        report = generator.generate_report(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        assert isinstance(report, MitigationReport)
        assert report.report_id.startswith("rpt-")

    def test_report_has_strategy_id(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        report = generator.generate_report(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        assert report.strategy_id == sample_strategy.strategy_id

    def test_report_has_operator_id(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        report = generator.generate_report(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        assert report.operator_id == sample_risk_trigger.operator_id

    def test_report_has_pre_score(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        report = generator.generate_report(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        assert report.pre_score == sample_risk_trigger.composite_score

    def test_report_provenance_hash_64_chars(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        report = generator.generate_report(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        assert len(report.provenance_hash) == 64

    def test_report_measures_summary_populated(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        report = generator.generate_report(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        assert len(report.measures_summary) == len(
            sample_strategy.measures
        )
        for ms in report.measures_summary:
            assert isinstance(ms, MeasureSummary)

    def test_report_without_verification(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        report = generator.generate_report(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
            verification=None,
        )
        assert report.verification_result is None

    def test_report_with_verification(
        self,
        generator,
        sample_strategy,
        sample_risk_trigger,
        sample_verification_report,
    ):
        report = generator.generate_report(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
            verification=sample_verification_report,
        )
        assert report.verification_result == VerificationResult.SUFFICIENT


class TestFormatForDDS:
    """Test format_for_dds DDS-structured report."""

    def test_dds_has_required_sections(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        dds = generator.format_for_dds(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        assert "report_metadata" in dds
        assert "risk_trigger" in dds
        assert "strategy" in dds
        assert "measures" in dds
        assert "verification" in dds
        assert "regulatory_compliance" in dds
        assert "recommendations" in dds

    def test_dds_metadata_has_agent_info(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        dds = generator.format_for_dds(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        meta = dds["report_metadata"]
        assert meta["agent_id"] == AGENT_ID
        assert meta["agent_version"] == AGENT_VERSION
        assert "provenance_hash" in meta
        assert len(meta["provenance_hash"]) == 64

    def test_dds_includes_provenance_chain_when_configured(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        # Default config has include_provenance=True
        dds = generator.format_for_dds(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        assert "provenance_chain" in dds
        prov = dds["provenance_chain"]
        assert prov["algorithm"] == "sha256"
        assert prov["chain_valid"] is True

    def test_dds_risk_trigger_section(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        dds = generator.format_for_dds(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        rt = dds["risk_trigger"]
        assert rt["assessment_id"] == sample_risk_trigger.assessment_id
        assert rt["operator_id"] == sample_risk_trigger.operator_id
        assert rt["commodity"] == sample_risk_trigger.commodity.value
        assert "dimension_breakdown" in rt

    def test_dds_strategy_section(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        dds = generator.format_for_dds(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        strat = dds["strategy"]
        assert strat["strategy_id"] == sample_strategy.strategy_id
        assert "total_measures" in strat
        assert "cumulative_expected_reduction" in strat


class TestVerificationSection:
    """Test verification section in DDS format."""

    def test_verification_pending_when_none(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        dds = generator.format_for_dds(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
            verification=None,
        )
        assert dds["verification"]["status"] == "pending"

    def test_verification_completed_when_provided(
        self,
        generator,
        sample_strategy,
        sample_risk_trigger,
        sample_verification_report,
    ):
        dds = generator.format_for_dds(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
            verification=sample_verification_report,
        )
        ver = dds["verification"]
        assert ver["status"] == "completed"
        assert ver["result"] == VerificationResult.SUFFICIENT.value
        assert ver["target_achieved"] is True


class TestRegulatorySection:
    """Test regulatory compliance section."""

    def test_article_references_complete(self):
        assert Article11Category.ADDITIONAL_INFO in _ARTICLE_REFERENCES
        assert Article11Category.INDEPENDENT_AUDIT in _ARTICLE_REFERENCES
        assert Article11Category.OTHER_MEASURES in _ARTICLE_REFERENCES

    def test_regulatory_section_has_primary_articles(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        dds = generator.format_for_dds(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        reg = dds["regulatory_compliance"]
        assert reg["regulation"] == "EU 2023/1115 (EUDR)"
        assert len(reg["primary_articles"]) == 4

    def test_regulatory_section_enforcement_dates(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        dds = generator.format_for_dds(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        reg = dds["regulatory_compliance"]
        assert reg["enforcement_dates"]["large_operators"] == "2025-12-30"
        assert reg["enforcement_dates"]["sme_operators"] == "2026-06-30"


class TestRecommendations:
    """Test _generate_recommendations."""

    def test_recommendations_without_verification(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        dds = generator.format_for_dds(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
            verification=None,
        )
        recs = dds["recommendations"]
        assert len(recs) >= 1
        # Should recommend verification
        verification_rec = [
            r for r in recs if r["category"] == "verification"
        ]
        assert len(verification_rec) >= 1

    def test_recommendations_with_sufficient(
        self,
        generator,
        sample_strategy,
        sample_risk_trigger,
        sample_verification_report,
    ):
        dds = generator.format_for_dds(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
            verification=sample_verification_report,
        )
        recs = dds["recommendations"]
        monitoring_recs = [
            r for r in recs if r["category"] == "monitoring"
        ]
        assert len(monitoring_recs) >= 1

    def test_recommendations_with_insufficient(
        self, generator, sample_strategy, sample_risk_trigger,
    ):
        insuf_verification = VerificationReport(
            verification_id="ver-insuf",
            strategy_id=sample_strategy.strategy_id,
            pre_score=Decimal("72"),
            post_score=Decimal("75"),
            risk_reduction=Decimal("-3"),
            result=VerificationResult.INSUFFICIENT,
            verified_by="AGENT-EUDR-029",
        )
        dds = generator.format_for_dds(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
            verification=insuf_verification,
        )
        recs = dds["recommendations"]
        critical_recs = [
            r for r in recs if r.get("priority") == "critical"
        ]
        assert len(critical_recs) >= 1


class TestCumulativeReduction:
    """Test _calculate_cumulative_reduction."""

    def test_empty_measures_returns_zero(self, generator):
        result = generator._calculate_cumulative_reduction([])
        assert result == Decimal("0")

    def test_single_measure_reduction(self, generator, sample_measure):
        result = generator._calculate_cumulative_reduction(
            [sample_measure],
        )
        assert result == Decimal("25.00")

    def test_diminishing_returns(self, generator):
        measures = [
            MitigationMeasure(
                measure_id=f"m{i}",
                strategy_id="s1",
                title=f"M{i}",
                article11_category=Article11Category.OTHER_MEASURES,
                target_dimension=RiskDimension.COUNTRY,
                expected_risk_reduction=Decimal("30"),
            )
            for i in range(3)
        ]
        result = generator._calculate_cumulative_reduction(measures)
        # 1 - (0.7^3) = 1 - 0.343 = 0.657 -> 65.70%
        assert result == Decimal("65.70")

    def test_capped_at_max_effectiveness(self, generator):
        measures = [
            MitigationMeasure(
                measure_id=f"m{i}",
                strategy_id="s1",
                title=f"M{i}",
                article11_category=Article11Category.OTHER_MEASURES,
                target_dimension=RiskDimension.COUNTRY,
                expected_risk_reduction=Decimal("50"),
            )
            for i in range(5)
        ]
        result = generator._calculate_cumulative_reduction(measures)
        assert result <= generator._config.max_effectiveness_cap


class TestDeterminePostScore:
    """Test _determine_post_score."""

    def test_uses_verification_post_score_when_available(
        self,
        generator,
        sample_strategy,
        sample_verification_report,
    ):
        score = generator._determine_post_score(
            strategy=sample_strategy,
            verification=sample_verification_report,
        )
        assert score == sample_verification_report.post_score

    def test_uses_strategy_post_score_when_no_verification(
        self, generator, sample_strategy,
    ):
        sample_strategy.post_mitigation_score = Decimal("40.00")
        score = generator._determine_post_score(
            strategy=sample_strategy,
            verification=None,
        )
        assert score == Decimal("40.00")

    def test_estimates_when_neither_available(
        self, generator, sample_strategy,
    ):
        sample_strategy.post_mitigation_score = None
        score = generator._determine_post_score(
            strategy=sample_strategy,
            verification=None,
        )
        # Should estimate based on cumulative reduction
        assert score >= Decimal("0")
        assert score < sample_strategy.pre_mitigation_score


class TestMappingConstants:
    """Test module-level mapping constants."""

    def test_risk_level_labels_complete(self):
        for level in RiskLevel:
            assert level in _RISK_LEVEL_LABELS

    def test_verification_labels_complete(self):
        for result in VerificationResult:
            assert result in _VERIFICATION_LABELS

    def test_article_references_have_required_keys(self):
        for cat, ref in _ARTICLE_REFERENCES.items():
            assert "primary" in ref
            assert "description" in ref
            assert "record_keeping" in ref
            assert "dds_requirement" in ref
