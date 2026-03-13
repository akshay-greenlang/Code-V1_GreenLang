# -*- coding: utf-8 -*-
"""
Unit tests for RiskDataIntegrator - AGENT-EUDR-037

Tests risk integration, overall risk computation, score aggregation,
score-to-level conversion, mitigations, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.due_diligence_statement_creator.config import DDSCreatorConfig
from greenlang.agents.eudr.due_diligence_statement_creator.risk_data_integrator import RiskDataIntegrator
from greenlang.agents.eudr.due_diligence_statement_creator.models import RiskLevel, RiskReference


@pytest.fixture
def config():
    return DDSCreatorConfig()


@pytest.fixture
def integrator(config):
    return RiskDataIntegrator(config=config)


class TestIntegrateRisk:
    @pytest.mark.asyncio
    async def test_returns_risk_reference(self, integrator):
        ref = await integrator.integrate_risk(
            risk_id="R-001", source_agent="EUDR-016",
            risk_category="country", risk_score=45.0)
        assert isinstance(ref, RiskReference)

    @pytest.mark.asyncio
    async def test_risk_id_set(self, integrator):
        ref = await integrator.integrate_risk(
            risk_id="R-TEST", source_agent="EUDR-016", risk_category="country")
        assert ref.risk_id == "R-TEST"

    @pytest.mark.asyncio
    async def test_source_agent_set(self, integrator):
        ref = await integrator.integrate_risk(
            risk_id="R-001", source_agent="EUDR-017", risk_category="supplier")
        assert ref.source_agent == "EUDR-017"

    @pytest.mark.asyncio
    async def test_risk_level_parsed(self, integrator):
        ref = await integrator.integrate_risk(
            risk_id="R-001", source_agent="EUDR-016",
            risk_category="country", risk_level="high")
        assert ref.risk_level == RiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_invalid_risk_level_defaults_standard(self, integrator):
        ref = await integrator.integrate_risk(
            risk_id="R-001", source_agent="EUDR-016",
            risk_category="country", risk_level="invalid_xyz")
        assert ref.risk_level == RiskLevel.STANDARD

    @pytest.mark.asyncio
    async def test_risk_score_rounded(self, integrator):
        ref = await integrator.integrate_risk(
            risk_id="R-001", source_agent="EUDR-016",
            risk_category="country", risk_score=45.678)
        assert ref.risk_score == Decimal("45.68")

    @pytest.mark.asyncio
    async def test_factors_set(self, integrator):
        ref = await integrator.integrate_risk(
            risk_id="R-001", source_agent="EUDR-016",
            risk_category="country", factors=["deforestation", "governance"])
        assert len(ref.factors) == 2

    @pytest.mark.asyncio
    async def test_mitigation_measures_set(self, integrator):
        ref = await integrator.integrate_risk(
            risk_id="R-001", source_agent="EUDR-016",
            risk_category="country",
            mitigation_measures=["enhanced_monitoring"])
        assert len(ref.mitigation_measures) == 1

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, integrator):
        ref = await integrator.integrate_risk(
            risk_id="R-001", source_agent="EUDR-016", risk_category="country")
        assert len(ref.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_integration_count_increments(self, integrator):
        await integrator.integrate_risk(
            risk_id="R-001", source_agent="EUDR-016", risk_category="country")
        await integrator.integrate_risk(
            risk_id="R-002", source_agent="EUDR-017", risk_category="supplier")
        health = await integrator.health_check()
        assert health["integrations_completed"] == 2

    @pytest.mark.asyncio
    async def test_score_clamped_to_100(self, integrator):
        ref = await integrator.integrate_risk(
            risk_id="R-001", source_agent="EUDR-016",
            risk_category="country", risk_score=150.0)
        assert ref.risk_score <= Decimal("100")


class TestComputeOverallRisk:
    @pytest.mark.asyncio
    async def test_empty_returns_standard(self, integrator):
        level = await integrator.compute_overall_risk([])
        assert level == RiskLevel.STANDARD

    @pytest.mark.asyncio
    async def test_single_reference(self, integrator):
        ref = await integrator.integrate_risk(
            risk_id="R-001", source_agent="EUDR-016",
            risk_category="country", risk_level="high")
        level = await integrator.compute_overall_risk([ref])
        assert level == RiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_max_level_wins(self, integrator, multiple_risk_references):
        level = await integrator.compute_overall_risk(multiple_risk_references)
        assert level == RiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_critical_overrides_all(self, integrator):
        refs = [
            RiskReference(risk_id="R1", source_agent="E016", risk_category="c",
                          risk_level=RiskLevel.LOW, risk_score=Decimal("10")),
            RiskReference(risk_id="R2", source_agent="E017", risk_category="c",
                          risk_level=RiskLevel.CRITICAL, risk_score=Decimal("90")),
        ]
        level = await integrator.compute_overall_risk(refs)
        assert level == RiskLevel.CRITICAL


class TestAggregateScores:
    @pytest.mark.asyncio
    async def test_empty_returns_zero(self, integrator):
        avg = await integrator.aggregate_scores([])
        assert avg == Decimal("0")

    @pytest.mark.asyncio
    async def test_single_reference_returns_score(self, integrator):
        ref = RiskReference(risk_id="R1", source_agent="E016",
                            risk_category="c", risk_score=Decimal("50.00"))
        avg = await integrator.aggregate_scores([ref])
        assert avg == Decimal("50.00")

    @pytest.mark.asyncio
    async def test_multiple_references_average(self, integrator):
        refs = [
            RiskReference(risk_id="R1", source_agent="E016",
                          risk_category="c", risk_score=Decimal("30.00")),
            RiskReference(risk_id="R2", source_agent="E017",
                          risk_category="c", risk_score=Decimal("70.00")),
        ]
        avg = await integrator.aggregate_scores(refs)
        assert avg == Decimal("50.00")


class TestScoreToRiskLevel:
    @pytest.mark.asyncio
    async def test_low_score(self, integrator):
        level = await integrator.score_to_risk_level(Decimal("15"))
        assert level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_standard_score(self, integrator):
        level = await integrator.score_to_risk_level(Decimal("45"))
        assert level == RiskLevel.STANDARD

    @pytest.mark.asyncio
    async def test_high_score(self, integrator):
        level = await integrator.score_to_risk_level(Decimal("70"))
        assert level == RiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_critical_score(self, integrator):
        level = await integrator.score_to_risk_level(Decimal("85"))
        assert level == RiskLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_zero_score(self, integrator):
        level = await integrator.score_to_risk_level(Decimal("0"))
        assert level == RiskLevel.LOW


class TestGetRequiredMitigations:
    @pytest.mark.asyncio
    async def test_low_risk_no_mitigations(self, integrator):
        mitigations = await integrator.get_required_mitigations(RiskLevel.LOW)
        assert mitigations == []

    @pytest.mark.asyncio
    async def test_standard_risk_has_mitigations(self, integrator):
        mitigations = await integrator.get_required_mitigations(RiskLevel.STANDARD)
        assert len(mitigations) >= 1

    @pytest.mark.asyncio
    async def test_critical_risk_most_mitigations(self, integrator):
        mitigations = await integrator.get_required_mitigations(RiskLevel.CRITICAL)
        assert len(mitigations) >= 5


class TestRiskHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, integrator):
        health = await integrator.health_check()
        assert health["engine"] == "RiskDataIntegrator"
        assert health["status"] == "healthy"
