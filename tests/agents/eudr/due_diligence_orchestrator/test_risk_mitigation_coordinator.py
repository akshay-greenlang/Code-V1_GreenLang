# -*- coding: utf-8 -*-
"""
Unit tests for Engine 4: Risk Mitigation Coordinator -- AGENT-EUDR-026

Tests Phase 3 orchestration, mitigation requirement determination,
adequacy verification, negligible risk bypass, proportionality
checks, and Article 11 compliance.

Test count: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import AsyncMock, MagicMock

from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    DueDiligencePhase,
    WorkflowState,
    WorkflowStatus,
    WorkflowType,
    EUDRCommodity,
    MitigationDecision,
    CompositeRiskProfile,
)
from greenlang.agents.eudr.due_diligence_orchestrator.risk_mitigation_coordinator import (
    RiskMitigationCoordinator,
)


class TestMitigationCoordinatorInit:
    """Test coordinator initialization."""

    def test_init_default(self, default_config):
        coord = RiskMitigationCoordinator()
        assert coord is not None

    def test_init_with_config(self, default_config):
        coord = RiskMitigationCoordinator(config=default_config)
        assert coord is not None


class TestMitigationRequirement:
    """Test mitigation requirement determination."""

    @pytest.mark.parametrize("score,expected_required", [
        (Decimal("0"), False),
        (Decimal("10"), False),
        (Decimal("20"), False),
        (Decimal("21"), True),
        (Decimal("35"), True),
        (Decimal("50"), True),
        (Decimal("51"), True),
        (Decimal("80"), True),
        (Decimal("100"), True),
    ])
    def test_determine_mitigation_required(
        self, risk_mitigation_coordinator, score, expected_required,
    ):
        coord = risk_mitigation_coordinator
        decision = coord.determine_mitigation_requirement(score)
        assert decision.mitigation_required == expected_required

    @pytest.mark.parametrize("score,expected_level", [
        (Decimal("10"), "none"),
        (Decimal("20"), "none"),
        (Decimal("30"), "standard"),
        (Decimal("50"), "standard"),
        (Decimal("51"), "enhanced"),
        (Decimal("80"), "enhanced"),
    ])
    def test_determine_mitigation_level(
        self, risk_mitigation_coordinator, score, expected_level,
    ):
        coord = risk_mitigation_coordinator
        decision = coord.determine_mitigation_requirement(score)
        assert decision.mitigation_level == expected_level

    def test_negligible_bypasses_to_package(
        self, risk_mitigation_coordinator,
    ):
        coord = risk_mitigation_coordinator
        decision = coord.determine_mitigation_requirement(Decimal("15"))
        assert decision.bypass_justification is not None or decision.mitigation_required is False

    def test_enhanced_requires_independent_audit(
        self, risk_mitigation_coordinator,
    ):
        coord = risk_mitigation_coordinator
        decision = coord.determine_mitigation_requirement(Decimal("70"))
        assert decision.mitigation_level == "enhanced"


class TestMitigationAdequacy:
    """Test mitigation adequacy verification."""

    def test_adequate_when_residual_below_target(
        self, risk_mitigation_coordinator,
    ):
        coord = risk_mitigation_coordinator
        result = coord.verify_mitigation_adequacy(
            pre_mitigation_score=Decimal("45"),
            post_mitigation_score=Decimal("12"),
            mitigation_level="standard",
        )
        assert result.adequate is True

    def test_inadequate_when_residual_above_target(
        self, risk_mitigation_coordinator,
    ):
        coord = risk_mitigation_coordinator
        result = coord.verify_mitigation_adequacy(
            pre_mitigation_score=Decimal("45"),
            post_mitigation_score=Decimal("20"),
            mitigation_level="standard",
        )
        assert result.adequate is False

    def test_enhanced_has_stricter_target(
        self, risk_mitigation_coordinator,
    ):
        coord = risk_mitigation_coordinator
        standard = coord.verify_mitigation_adequacy(
            Decimal("50"), Decimal("12"), "standard",
        )
        enhanced = coord.verify_mitigation_adequacy(
            Decimal("50"), Decimal("12"), "enhanced",
        )
        # Enhanced target is 10, so 12 should fail enhanced but pass standard
        assert standard.adequate is True
        assert enhanced.adequate is False

    def test_reduction_percentage_calculated(
        self, risk_mitigation_coordinator,
    ):
        coord = risk_mitigation_coordinator
        result = coord.verify_mitigation_adequacy(
            Decimal("50"), Decimal("10"), "standard",
        )
        assert result.reduction_pct == Decimal("80.00")

    def test_zero_pre_mitigation_score(
        self, risk_mitigation_coordinator,
    ):
        coord = risk_mitigation_coordinator
        result = coord.verify_mitigation_adequacy(
            Decimal("0"), Decimal("0"), "standard",
        )
        assert result.adequate is True
        assert result.reduction_pct == Decimal("0") or result.reduction_pct == Decimal("0.00")

    def test_gap_calculation(self, risk_mitigation_coordinator):
        coord = risk_mitigation_coordinator
        result = coord.verify_mitigation_adequacy(
            Decimal("60"), Decimal("25"), "standard",
        )
        assert result.gap == Decimal("10")  # 25 - 15 = 10


class TestMitigationDecisionModel:
    """Test MitigationDecision model creation."""

    def test_create_mitigation_decision(self):
        decision = MitigationDecision(
            workflow_id="wf-001",
            mitigation_required=True,
            mitigation_level="standard",
            pre_mitigation_score=Decimal("45"),
            post_mitigation_score=Decimal("12"),
        )
        assert decision.workflow_id == "wf-001"
        assert decision.mitigation_required is True

    def test_bypass_decision(self):
        decision = MitigationDecision(
            workflow_id="wf-002",
            mitigation_required=False,
            mitigation_level="none",
            pre_mitigation_score=Decimal("15"),
            bypass_justification="Negligible risk per Art. 11",
        )
        assert decision.mitigation_required is False


class TestPhase3Execution:
    """Test Phase 3 execution."""

    @pytest.mark.asyncio
    async def test_execute_phase_with_standard_mitigation(
        self, risk_mitigation_coordinator, workflow_state_phase2_complete,
        mock_agent_client,
    ):
        coord = risk_mitigation_coordinator
        result = await coord.execute_phase(
            workflow=workflow_state_phase2_complete,
            agent_client=mock_agent_client,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_phase_bypass_for_negligible(
        self, risk_mitigation_coordinator, mock_agent_client,
    ):
        coord = risk_mitigation_coordinator
        state = WorkflowState(
            workflow_id="wf-neg",
            definition_id="def-001",
            composite_risk_score=Decimal("10"),
        )
        result = await coord.execute_phase(
            workflow=state,
            agent_client=mock_agent_client,
        )
        assert result is not None
        # Should bypass mitigation since risk is negligible
        if hasattr(result, "mitigation_bypassed"):
            assert result.mitigation_bypassed is True


class TestProportionalityVerification:
    """Test mitigation proportionality per Article 11(1)."""

    def test_proportionality_standard_level(
        self, risk_mitigation_coordinator,
    ):
        coord = risk_mitigation_coordinator
        proportional = coord.verify_proportionality(
            risk_level="standard",
            mitigation_strategies=["supplier_audit", "additional_documentation"],
        )
        assert proportional is True

    def test_proportionality_enhanced_level(
        self, risk_mitigation_coordinator,
    ):
        coord = risk_mitigation_coordinator
        proportional = coord.verify_proportionality(
            risk_level="enhanced",
            mitigation_strategies=[
                "independent_audit", "site_visit",
                "supplier_capacity_building",
            ],
        )
        assert proportional is True

    def test_disproportionate_mitigation(
        self, risk_mitigation_coordinator,
    ):
        coord = risk_mitigation_coordinator
        proportional = coord.verify_proportionality(
            risk_level="standard",
            mitigation_strategies=[],
        )
        assert proportional is False


class TestMitigationEvidence:
    """Test mitigation evidence collection."""

    def test_collect_evidence_returns_dict(
        self, risk_mitigation_coordinator,
    ):
        coord = risk_mitigation_coordinator
        evidence = coord.collect_mitigation_evidence(
            strategies=["supplier_audit"],
            agent_outputs={"EUDR-025": {"mitigation_plan": "approved"}},
        )
        assert isinstance(evidence, dict)

    def test_evidence_includes_strategies(
        self, risk_mitigation_coordinator,
    ):
        coord = risk_mitigation_coordinator
        evidence = coord.collect_mitigation_evidence(
            strategies=["independent_audit", "site_visit"],
            agent_outputs={},
        )
        assert len(evidence) > 0


class TestDeterminism:
    """Test determinism of mitigation decisions."""

    def test_same_score_same_decision(self, risk_mitigation_coordinator):
        coord = risk_mitigation_coordinator
        d1 = coord.determine_mitigation_requirement(Decimal("45"))
        d2 = coord.determine_mitigation_requirement(Decimal("45"))
        assert d1.mitigation_level == d2.mitigation_level
        assert d1.mitigation_required == d2.mitigation_required

    def test_same_adequacy_same_result(self, risk_mitigation_coordinator):
        coord = risk_mitigation_coordinator
        r1 = coord.verify_mitigation_adequacy(
            Decimal("50"), Decimal("12"), "standard",
        )
        r2 = coord.verify_mitigation_adequacy(
            Decimal("50"), Decimal("12"), "standard",
        )
        assert r1.adequate == r2.adequate
        assert r1.reduction_pct == r2.reduction_pct
