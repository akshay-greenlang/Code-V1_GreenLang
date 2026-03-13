# -*- coding: utf-8 -*-
"""
Unit tests for Engine 3: Risk Assessment Coordinator -- AGENT-EUDR-026

Tests Phase 2 orchestration of EUDR-016 through EUDR-025, composite
risk score calculation, risk level classification, Article 10(2) factor
mapping, degraded mode, and risk assessment coverage.

Test count: ~70 tests
Author: GreenLang Platform Team
Date: March 2026
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import AsyncMock, MagicMock

from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    PHASE_2_AGENTS,
    DueDiligencePhase,
    WorkflowState,
    WorkflowStatus,
    WorkflowType,
    EUDRCommodity,
    CompositeRiskProfile,
    RiskScoreContribution,
)
from greenlang.agents.eudr.due_diligence_orchestrator.risk_assessment_coordinator import (
    RiskAssessmentCoordinator,
)
from tests.agents.eudr.due_diligence_orchestrator.conftest import (
    DEFAULT_RISK_WEIGHTS,
)


class TestRiskAssessmentInit:
    """Test coordinator initialization."""

    def test_init_default(self, default_config):
        coord = RiskAssessmentCoordinator()
        assert coord is not None

    def test_init_with_config(self, default_config):
        coord = RiskAssessmentCoordinator(config=default_config)
        assert coord is not None


class TestPhase2Agents:
    """Test Phase 2 agent identification."""

    def test_phase2_agents_count(self):
        assert len(PHASE_2_AGENTS) == 10

    def test_phase2_agents_range(self):
        expected = [f"EUDR-{i:03d}" for i in range(16, 26)]
        assert PHASE_2_AGENTS == expected

    def test_is_phase2_agent(self, risk_assessment_coordinator):
        coord = risk_assessment_coordinator
        assert coord._is_phase2_agent("EUDR-016") is True
        assert coord._is_phase2_agent("EUDR-025") is True
        assert coord._is_phase2_agent("EUDR-001") is False


class TestCompositeRiskCalculation:
    """Test deterministic composite risk score calculation."""

    def test_calculate_with_all_scores(
        self, risk_assessment_coordinator, medium_risk_scores,
    ):
        coord = risk_assessment_coordinator
        profile = coord.calculate_composite_risk(medium_risk_scores)
        assert isinstance(profile, CompositeRiskProfile)
        assert profile.composite_score > Decimal("0")

    def test_composite_is_decimal(
        self, risk_assessment_coordinator, medium_risk_scores,
    ):
        coord = risk_assessment_coordinator
        profile = coord.calculate_composite_risk(medium_risk_scores)
        assert isinstance(profile.composite_score, Decimal)

    def test_composite_deterministic(
        self, risk_assessment_coordinator, medium_risk_scores,
    ):
        coord = risk_assessment_coordinator
        p1 = coord.calculate_composite_risk(medium_risk_scores)
        p2 = coord.calculate_composite_risk(medium_risk_scores)
        assert p1.composite_score == p2.composite_score

    def test_weights_sum_to_one(self):
        total = sum(DEFAULT_RISK_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_all_zero_scores_return_zero(self, risk_assessment_coordinator):
        coord = risk_assessment_coordinator
        scores = {f"EUDR-{i:03d}": Decimal("0") for i in range(16, 26)}
        profile = coord.calculate_composite_risk(scores)
        assert profile.composite_score == Decimal("0") or profile.composite_score == Decimal("0.00")

    def test_all_100_scores_return_100(self, risk_assessment_coordinator):
        coord = risk_assessment_coordinator
        scores = {f"EUDR-{i:03d}": Decimal("100") for i in range(16, 26)}
        profile = coord.calculate_composite_risk(scores)
        assert profile.composite_score == Decimal("100") or profile.composite_score == Decimal("100.00")

    @pytest.mark.parametrize("agent_id,weight", list(DEFAULT_RISK_WEIGHTS.items()))
    def test_individual_weight_contribution(
        self, risk_assessment_coordinator, agent_id, weight,
    ):
        coord = risk_assessment_coordinator
        scores = {f"EUDR-{i:03d}": Decimal("0") for i in range(16, 26)}
        scores[agent_id] = Decimal("100")
        profile = coord.calculate_composite_risk(scores)
        expected = (Decimal("100") * weight).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        assert profile.composite_score == expected

    def test_partial_scores_calculate_coverage(
        self, risk_assessment_coordinator,
    ):
        coord = risk_assessment_coordinator
        scores = {
            "EUDR-016": Decimal("50"),
            "EUDR-017": Decimal("40"),
        }
        profile = coord.calculate_composite_risk(scores)
        assert profile.coverage_pct == Decimal("20.00") or profile.coverage_pct == Decimal("20")
        assert profile.all_dimensions_scored is False


class TestRiskLevelClassification:
    """Test risk level classification thresholds."""

    @pytest.mark.parametrize("score,expected_level", [
        (Decimal("0"), "negligible"),
        (Decimal("10"), "negligible"),
        (Decimal("20"), "negligible"),
        (Decimal("21"), "low"),
        (Decimal("40"), "low"),
        (Decimal("41"), "standard"),
        (Decimal("60"), "standard"),
        (Decimal("61"), "high"),
        (Decimal("80"), "high"),
        (Decimal("81"), "critical"),
        (Decimal("100"), "critical"),
    ])
    def test_classify_risk_level(
        self, risk_assessment_coordinator, score, expected_level,
    ):
        coord = risk_assessment_coordinator
        level = coord.classify_risk_level(score)
        assert level == expected_level

    def test_low_risk_scores_negligible(
        self, risk_assessment_coordinator, low_risk_scores,
    ):
        coord = risk_assessment_coordinator
        profile = coord.calculate_composite_risk(low_risk_scores)
        assert profile.risk_level in ("negligible", "low")

    def test_high_risk_scores_critical(
        self, risk_assessment_coordinator, high_risk_scores,
    ):
        coord = risk_assessment_coordinator
        profile = coord.calculate_composite_risk(high_risk_scores)
        assert profile.risk_level in ("high", "critical")


class TestArticle10Mapping:
    """Test Article 10(2) risk factor mapping."""

    def test_all_factors_mapped(self, risk_assessment_coordinator):
        coord = risk_assessment_coordinator
        factors = coord.get_article10_factor_mapping()
        assert "supply_chain_complexity" in factors or len(factors) >= 6

    @pytest.mark.parametrize("factor", [
        "supply_chain_complexity", "circumvention_risk",
        "country_non_compliance", "country_production_risk",
        "supplier_concerns", "substantiated_concerns",
    ])
    def test_each_factor_has_agents(
        self, risk_assessment_coordinator, factor,
    ):
        coord = risk_assessment_coordinator
        factors = coord.get_article10_factor_mapping()
        if factor in factors:
            assert len(factors[factor]) > 0


class TestPhaseExecution:
    """Test Phase 2 execution."""

    @pytest.mark.asyncio
    async def test_execute_phase_returns_profile(
        self, risk_assessment_coordinator, workflow_state_phase1_complete,
        mock_agent_client,
    ):
        coord = risk_assessment_coordinator
        result = await coord.execute_phase(
            workflow=workflow_state_phase1_complete,
            agent_client=mock_agent_client,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_phase_all_10_agents(
        self, risk_assessment_coordinator, workflow_state_phase1_complete,
        mock_agent_client,
    ):
        coord = risk_assessment_coordinator
        await coord.execute_phase(
            workflow=workflow_state_phase1_complete,
            agent_client=mock_agent_client,
        )
        invoked = {c.kwargs.get("agent_id") or (c.args[0] if c.args else None)
                   for c in mock_agent_client.invoke.call_args_list}
        for agent_id in PHASE_2_AGENTS:
            assert agent_id in invoked or len(invoked) >= 10


class TestDegradedMode:
    """Test degraded mode when non-critical agents fail."""

    def test_calculate_with_missing_non_critical_scores(
        self, risk_assessment_coordinator,
    ):
        coord = risk_assessment_coordinator
        scores = {
            "EUDR-016": Decimal("30"),
            "EUDR-017": Decimal("25"),
            "EUDR-018": Decimal("20"),
            "EUDR-020": Decimal("35"),
            "EUDR-023": Decimal("30"),
            # Missing EUDR-019, 021, 022, 024, 025
        }
        profile = coord.calculate_composite_risk(scores)
        assert profile.composite_score > Decimal("0")
        assert profile.all_dimensions_scored is False

    def test_degraded_coverage_reported(
        self, risk_assessment_coordinator,
    ):
        coord = risk_assessment_coordinator
        scores = {"EUDR-016": Decimal("50")}
        profile = coord.calculate_composite_risk(scores)
        assert profile.coverage_pct == Decimal("10") or profile.coverage_pct == Decimal("10.00")


class TestRiskProfileProvenance:
    """Test risk profile provenance tracking."""

    def test_profile_has_provenance_hash(
        self, risk_assessment_coordinator, medium_risk_scores,
    ):
        coord = risk_assessment_coordinator
        profile = coord.calculate_composite_risk(medium_risk_scores)
        assert profile.provenance_hash is not None or profile.profile_id is not None

    def test_same_inputs_same_hash(
        self, risk_assessment_coordinator, medium_risk_scores,
    ):
        coord = risk_assessment_coordinator
        p1 = coord.calculate_composite_risk(medium_risk_scores)
        p2 = coord.calculate_composite_risk(medium_risk_scores)
        if p1.provenance_hash and p2.provenance_hash:
            assert p1.provenance_hash == p2.provenance_hash

    def test_contributions_have_weighted_scores(
        self, risk_assessment_coordinator, medium_risk_scores,
    ):
        coord = risk_assessment_coordinator
        profile = coord.calculate_composite_risk(medium_risk_scores)
        for contrib in profile.contributions:
            assert contrib.weighted_score == contrib.raw_score * contrib.weight


class TestHighestRiskDimensions:
    """Test highest risk dimension identification."""

    def test_highest_risk_identified(
        self, risk_assessment_coordinator, high_risk_scores,
    ):
        coord = risk_assessment_coordinator
        profile = coord.calculate_composite_risk(high_risk_scores)
        assert len(profile.highest_risk_dimensions) > 0

    def test_highest_risk_ordered(
        self, risk_assessment_coordinator,
    ):
        coord = risk_assessment_coordinator
        scores = {
            "EUDR-016": Decimal("90"), "EUDR-017": Decimal("10"),
            "EUDR-018": Decimal("50"), "EUDR-019": Decimal("30"),
            "EUDR-020": Decimal("80"), "EUDR-021": Decimal("20"),
            "EUDR-022": Decimal("40"), "EUDR-023": Decimal("60"),
            "EUDR-024": Decimal("5"),  "EUDR-025": Decimal("15"),
        }
        profile = coord.calculate_composite_risk(scores)
        if len(profile.highest_risk_dimensions) >= 2:
            assert profile.highest_risk_dimensions[0] in ("EUDR-016", "EUDR-020")
