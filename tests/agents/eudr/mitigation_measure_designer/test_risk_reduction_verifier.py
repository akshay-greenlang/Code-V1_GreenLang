# -*- coding: utf-8 -*-
"""
Unit tests for RiskReductionVerifier - AGENT-EUDR-029

Tests verification of risk reduction after mitigation measures are
implemented: _classify_result, _calculate_reduction,
_calculate_reduction_percentage, verify_risk_reduction (async),
recommend_additional_measures, and _generate_recommendations.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from greenlang.agents.eudr.mitigation_measure_designer.config import (
    MitigationMeasureDesignerConfig,
)
from greenlang.agents.eudr.mitigation_measure_designer.models import (
    Article11Category,
    MeasureTemplate,
    MitigationStrategy,
    RiskDimension,
    RiskTrigger,
    VerificationReport,
    VerificationResult,
    WorkflowStatus,
)
from greenlang.agents.eudr.mitigation_measure_designer.provenance import (
    ProvenanceTracker,
)
from greenlang.agents.eudr.mitigation_measure_designer.risk_reduction_verifier import (
    RiskReductionVerifier,
)


@pytest.fixture
def config():
    return MitigationMeasureDesignerConfig()


@pytest.fixture
def verifier(config):
    return RiskReductionVerifier(config=config, provenance=ProvenanceTracker())


class TestClassifyResult:
    """Test _classify_result logic."""

    def test_sufficient_when_post_le_target(self, verifier):
        result = verifier._classify_result(
            pre_score=Decimal("72"),
            post_score=Decimal("25"),
            target=Decimal("30"),
        )
        assert result == VerificationResult.SUFFICIENT

    def test_sufficient_when_post_equals_target(self, verifier):
        result = verifier._classify_result(
            pre_score=Decimal("72"),
            post_score=Decimal("30"),
            target=Decimal("30"),
        )
        assert result == VerificationResult.SUFFICIENT

    def test_partial_when_post_lt_pre_but_gt_target(self, verifier):
        result = verifier._classify_result(
            pre_score=Decimal("72"),
            post_score=Decimal("50"),
            target=Decimal("30"),
        )
        assert result == VerificationResult.PARTIAL

    def test_insufficient_when_post_ge_pre(self, verifier):
        result = verifier._classify_result(
            pre_score=Decimal("72"),
            post_score=Decimal("72"),
            target=Decimal("30"),
        )
        assert result == VerificationResult.INSUFFICIENT

    def test_insufficient_when_post_gt_pre(self, verifier):
        result = verifier._classify_result(
            pre_score=Decimal("72"),
            post_score=Decimal("80"),
            target=Decimal("30"),
        )
        assert result == VerificationResult.INSUFFICIENT


class TestCalculateReduction:
    """Test _calculate_reduction and _calculate_reduction_percentage."""

    def test_positive_reduction(self, verifier):
        result = verifier._calculate_reduction(Decimal("72"), Decimal("35"))
        assert result == Decimal("37.00")

    def test_zero_reduction(self, verifier):
        result = verifier._calculate_reduction(Decimal("72"), Decimal("72"))
        assert result == Decimal("0.00")

    def test_negative_reduction(self, verifier):
        result = verifier._calculate_reduction(Decimal("50"), Decimal("60"))
        assert result == Decimal("-10.00")

    def test_percentage_reduction(self, verifier):
        result = verifier._calculate_reduction_percentage(
            Decimal("100"), Decimal("75"),
        )
        assert result == Decimal("25.00")

    def test_percentage_reduction_zero_pre(self, verifier):
        result = verifier._calculate_reduction_percentage(
            Decimal("0"), Decimal("10"),
        )
        assert result == Decimal("0")

    def test_percentage_reduction_full(self, verifier):
        result = verifier._calculate_reduction_percentage(
            Decimal("80"), Decimal("0"),
        )
        assert result == Decimal("100.00")


class TestVerifyRiskReduction:
    """Test async verify_risk_reduction method."""

    @pytest.mark.asyncio
    async def test_verify_returns_verification_report(
        self, verifier, sample_risk_trigger, sample_strategy,
    ):
        report = await verifier.verify_risk_reduction(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        assert isinstance(report, VerificationReport)
        assert report.verification_id.startswith("ver-")
        assert report.strategy_id == sample_strategy.strategy_id

    @pytest.mark.asyncio
    async def test_verify_pre_score_from_trigger(
        self, verifier, sample_risk_trigger, sample_strategy,
    ):
        report = await verifier.verify_risk_reduction(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        assert report.pre_score == sample_risk_trigger.composite_score

    @pytest.mark.asyncio
    async def test_verify_post_score_is_simulated(
        self, verifier, sample_risk_trigger, sample_strategy,
    ):
        # Simulated _query_current_risk returns Decimal("35.00")
        report = await verifier.verify_risk_reduction(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        assert report.post_score == Decimal("35.00")

    @pytest.mark.asyncio
    async def test_verify_provenance_hash_present(
        self, verifier, sample_risk_trigger, sample_strategy,
    ):
        report = await verifier.verify_risk_reduction(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        assert report.provenance_hash is not None
        assert len(report.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_verify_result_classification(
        self, verifier, sample_risk_trigger, sample_strategy,
    ):
        """With pre=72 and simulated post=35.00, target=30 -> PARTIAL."""
        report = await verifier.verify_risk_reduction(
            strategy=sample_strategy,
            risk_trigger=sample_risk_trigger,
        )
        # post_score=35 > target=30, but 35 < pre=72 -> PARTIAL
        assert report.result == VerificationResult.PARTIAL


class TestRecommendAdditionalMeasures:
    """Test recommend_additional_measures."""

    def test_no_recommendations_when_sufficient(self, verifier, sample_verification_report, multiple_templates):
        # sample_verification_report has result=SUFFICIENT
        result = verifier.recommend_additional_measures(
            verification=sample_verification_report,
            templates=multiple_templates,
        )
        assert result == []

    def test_recommendations_when_partial(self, verifier, multiple_templates):
        partial_report = VerificationReport(
            verification_id="ver-partial",
            strategy_id="stg-001",
            pre_score=Decimal("72"),
            post_score=Decimal("50"),
            risk_reduction=Decimal("22"),
            result=VerificationResult.PARTIAL,
            verified_by="AGENT-EUDR-029",
        )
        result = verifier.recommend_additional_measures(
            verification=partial_report,
            templates=multiple_templates,
        )
        assert len(result) > 0
        # Should be sorted by base_effectiveness descending
        for i in range(len(result) - 1):
            assert result[i].base_effectiveness >= result[i + 1].base_effectiveness

    def test_max_five_recommendations(self, verifier):
        insufficient_report = VerificationReport(
            verification_id="ver-insuf",
            strategy_id="stg-001",
            pre_score=Decimal("72"),
            post_score=Decimal("72"),
            risk_reduction=Decimal("0"),
            result=VerificationResult.INSUFFICIENT,
            verified_by="AGENT-EUDR-029",
        )
        templates = [
            MeasureTemplate(
                template_id=f"T-{i:03d}",
                title=f"Template {i}",
                article11_category=Article11Category.OTHER_MEASURES,
                applicable_dimensions=[RiskDimension.COUNTRY],
                base_effectiveness=Decimal(str(10 + i)),
            )
            for i in range(10)
        ]
        result = verifier.recommend_additional_measures(
            verification=insufficient_report,
            templates=templates,
        )
        assert len(result) == 5


class TestGenerateRecommendations:
    """Test _generate_recommendations."""

    def test_sufficient_recommendations(self, verifier):
        recs = verifier._generate_recommendations(
            result=VerificationResult.SUFFICIENT,
            pre_score=Decimal("72"),
            post_score=Decimal("25"),
            target_score=Decimal("30"),
        )
        assert len(recs) == 2
        assert "acceptable level" in recs[0].lower()

    def test_partial_recommendations(self, verifier):
        recs = verifier._generate_recommendations(
            result=VerificationResult.PARTIAL,
            pre_score=Decimal("72"),
            post_score=Decimal("50"),
            target_score=Decimal("30"),
        )
        assert len(recs) == 3

    def test_insufficient_recommendations(self, verifier):
        recs = verifier._generate_recommendations(
            result=VerificationResult.INSUFFICIENT,
            pre_score=Decimal("72"),
            post_score=Decimal("75"),
            target_score=Decimal("30"),
        )
        assert len(recs) == 3
        assert "immediate" in recs[0].lower()
