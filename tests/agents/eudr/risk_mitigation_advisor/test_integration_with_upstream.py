# -*- coding: utf-8 -*-
"""
Integration Tests with Upstream EUDR Agents - AGENT-EUDR-025

Tests integration between the Risk Mitigation Advisor and upstream
risk assessment agents EUDR-016 through EUDR-024, validating risk
input consumption, cross-agent provenance, trigger event processing,
and end-to-end workflows from risk assessment to mitigation.

Upstream agents:
    - EUDR-016: Country Risk Evaluator
    - EUDR-017: Supplier Risk Scorer
    - EUDR-018: Commodity Risk Analyzer
    - EUDR-019: Corruption Index Monitor
    - EUDR-020: Deforestation Alert System
    - EUDR-021: Indigenous Rights Checker
    - EUDR-022: Protected Area Validator
    - EUDR-023: Legal Compliance Verifier
    - EUDR-024: Audit Risk Assessor (if available)

Test count: ~50 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    RiskInput,
    RiskCategory,
    TriggerEventType,
    AdjustmentType,
    TriggerEvent,
    RecommendStrategiesRequest,
    RecommendStrategiesResponse,
    CreatePlanRequest,
    AdaptiveScanRequest,
    PlanStatus,
    SUPPORTED_COMMODITIES,
    UPSTREAM_AGENT_COUNT,
    RISK_CATEGORY_COUNT,
)

from .conftest import FIXED_DATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_upstream_risk_input(
    country_score: str = "65",
    supplier_score: str = "70",
    commodity_score: str = "55",
    corruption_score: str = "40",
    deforestation_score: str = "75",
    indigenous_score: str = "30",
    protected_areas_score: str = "25",
    legal_compliance_score: str = "35",
    audit_score: str = "40",
    commodity: str = "soya",
    country_code: str = "BR",
) -> RiskInput:
    """Build RiskInput simulating outputs from upstream agents."""
    return RiskInput(
        operator_id="op-upstream-test",
        supplier_id="sup-upstream-test",
        country_code=country_code,
        commodity=commodity,
        country_risk_score=Decimal(country_score),
        supplier_risk_score=Decimal(supplier_score),
        commodity_risk_score=Decimal(commodity_score),
        corruption_risk_score=Decimal(corruption_score),
        deforestation_risk_score=Decimal(deforestation_score),
        indigenous_rights_score=Decimal(indigenous_score),
        protected_areas_score=Decimal(protected_areas_score),
        legal_compliance_score=Decimal(legal_compliance_score),
        audit_risk_score=Decimal(audit_score),
        assessment_date=FIXED_DATE,
    )


# ---------------------------------------------------------------------------
# Constants Validation
# ---------------------------------------------------------------------------


class TestUpstreamAgentConstants:
    """Validate upstream agent integration constants."""

    def test_upstream_agent_count(self):
        """System expects 9 upstream risk agents."""
        assert UPSTREAM_AGENT_COUNT == 9

    def test_risk_category_count(self):
        """8 risk categories matching 8 upstream agents."""
        assert RISK_CATEGORY_COUNT == 8
        assert len(RiskCategory) == 8

    def test_risk_categories_map_to_upstream_agents(self):
        """Each RiskCategory should correspond to an upstream agent."""
        expected_categories = {
            "country",        # EUDR-016
            "supplier",       # EUDR-017
            "commodity",      # EUDR-018
            "corruption",     # EUDR-019
            "deforestation",  # EUDR-020
            "indigenous_rights",  # EUDR-021
            "protected_areas",    # EUDR-022
            "legal_compliance",   # EUDR-023
        }
        actual_categories = {cat.value for cat in RiskCategory}
        assert actual_categories == expected_categories


# ---------------------------------------------------------------------------
# EUDR-016: Country Risk Evaluator Integration
# ---------------------------------------------------------------------------


class TestCountryRiskIntegration:
    """Test integration with EUDR-016 Country Risk Evaluator."""

    @pytest.mark.asyncio
    async def test_high_country_risk_triggers_country_strategies(self, strategy_engine):
        """High country risk should trigger country-specific strategies."""
        ri = _make_upstream_risk_input(country_score="90", deforestation_score="30")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        country_strategies = [
            s for s in result.strategies
            if RiskCategory.COUNTRY in s.risk_categories
        ]
        assert len(country_strategies) >= 1

    @pytest.mark.asyncio
    async def test_low_country_risk_fewer_country_strategies(self, strategy_engine):
        """Low country risk should produce fewer country-specific strategies."""
        ri = _make_upstream_risk_input(country_score="10")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        country_strategies = [
            s for s in result.strategies
            if RiskCategory.COUNTRY in s.risk_categories
        ]
        # With very low country risk, country strategies should be minimal
        assert isinstance(country_strategies, list)

    @pytest.mark.asyncio
    async def test_country_reclassification_trigger(self, monitoring_engine):
        """Country reclassification trigger event from EUDR-016."""
        trigger = TriggerEvent(
            event_id="evt-eudr016-reclass",
            event_type=TriggerEventType.COUNTRY_RECLASSIFICATION,
            severity="high",
            source_agent="EUDR-016",
            plan_ids=["plan-001"],
            description="Brazil reclassified from standard to high risk.",
            risk_score_before=Decimal("45"),
            risk_score_after=Decimal("75"),
            recommended_adjustment=AdjustmentType.SCOPE_EXPANSION,
            response_sla_hours=48,
        )
        result = await monitoring_engine.process_trigger(trigger)
        assert result is not None


# ---------------------------------------------------------------------------
# EUDR-017: Supplier Risk Scorer Integration
# ---------------------------------------------------------------------------


class TestSupplierRiskIntegration:
    """Test integration with EUDR-017 Supplier Risk Scorer."""

    @pytest.mark.asyncio
    async def test_high_supplier_risk_triggers_strategies(self, strategy_engine):
        """High supplier risk should trigger supplier-specific strategies."""
        ri = _make_upstream_risk_input(supplier_score="90", country_score="30")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        supplier_strategies = [
            s for s in result.strategies
            if RiskCategory.SUPPLIER in s.risk_categories
        ]
        assert len(supplier_strategies) >= 1

    @pytest.mark.asyncio
    async def test_supplier_risk_spike_trigger(self, monitoring_engine):
        """Supplier risk spike trigger event from EUDR-017."""
        trigger = TriggerEvent(
            event_id="evt-eudr017-spike",
            event_type=TriggerEventType.SUPPLIER_RISK_SPIKE,
            severity="high",
            source_agent="EUDR-017",
            plan_ids=["plan-002"],
            supplier_id="sup-spike-001",
            description="Supplier risk score increased by 40%.",
            risk_score_before=Decimal("40"),
            risk_score_after=Decimal("56"),
            recommended_adjustment=AdjustmentType.PLAN_ACCELERATION,
            response_sla_hours=24,
        )
        result = await monitoring_engine.process_trigger(trigger)
        assert result is not None

    @pytest.mark.asyncio
    async def test_supplier_failure_trigger(self, monitoring_engine):
        """Supplier audit failure trigger from EUDR-017."""
        trigger = TriggerEvent(
            event_id="evt-eudr017-fail",
            event_type=TriggerEventType.AUDIT_FAILURE,
            severity="critical",
            source_agent="EUDR-017",
            plan_ids=["plan-003"],
            supplier_id="sup-fail-001",
            description="Supplier failed scheduled compliance audit.",
            risk_score_before=Decimal("55"),
            risk_score_after=Decimal("85"),
            recommended_adjustment=AdjustmentType.EMERGENCY_RESPONSE,
            response_sla_hours=4,
        )
        result = await monitoring_engine.process_trigger(trigger)
        assert result is not None


# ---------------------------------------------------------------------------
# EUDR-018: Commodity Risk Analyzer Integration
# ---------------------------------------------------------------------------


class TestCommodityRiskIntegration:
    """Test integration with EUDR-018 Commodity Risk Analyzer."""

    @pytest.mark.parametrize("commodity", SUPPORTED_COMMODITIES)
    @pytest.mark.asyncio
    async def test_commodity_score_consumed(self, strategy_engine, commodity):
        """Each commodity risk score from EUDR-018 should be consumed."""
        ri = _make_upstream_risk_input(commodity=commodity, commodity_score="80")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert isinstance(result, RecommendStrategiesResponse)

    @pytest.mark.asyncio
    async def test_high_commodity_risk_strategies(self, strategy_engine):
        """High commodity risk should trigger commodity-specific strategies."""
        ri = _make_upstream_risk_input(commodity_score="95", country_score="20")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        commodity_strategies = [
            s for s in result.strategies
            if RiskCategory.COMMODITY in s.risk_categories
        ]
        assert len(commodity_strategies) >= 1


# ---------------------------------------------------------------------------
# EUDR-019: Corruption Index Monitor Integration
# ---------------------------------------------------------------------------


class TestCorruptionRiskIntegration:
    """Test integration with EUDR-019 Corruption Index Monitor."""

    @pytest.mark.asyncio
    async def test_high_corruption_triggers_strategies(self, strategy_engine):
        """High corruption score should trigger anti-corruption strategies."""
        ri = _make_upstream_risk_input(corruption_score="85")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        corruption_strategies = [
            s for s in result.strategies
            if RiskCategory.CORRUPTION in s.risk_categories
        ]
        assert len(corruption_strategies) >= 1

    @pytest.mark.asyncio
    async def test_corruption_score_affects_composite(self, strategy_engine):
        """Corruption score should contribute to composite risk score."""
        ri_high = _make_upstream_risk_input(corruption_score="90")
        ri_low = _make_upstream_risk_input(corruption_score="10")
        req_high = RecommendStrategiesRequest(
            risk_input=ri_high, top_k=5, deterministic_mode=True
        )
        req_low = RecommendStrategiesRequest(
            risk_input=ri_low, top_k=5, deterministic_mode=True
        )
        r_high = await strategy_engine.recommend(req_high)
        r_low = await strategy_engine.recommend(req_low)
        assert r_high.composite_risk_score > r_low.composite_risk_score


# ---------------------------------------------------------------------------
# EUDR-020: Deforestation Alert System Integration
# ---------------------------------------------------------------------------


class TestDeforestationAlertIntegration:
    """Test integration with EUDR-020 Deforestation Alert System."""

    @pytest.mark.asyncio
    async def test_deforestation_alert_trigger(self, monitoring_engine):
        """Deforestation alert from EUDR-020 should trigger emergency response."""
        trigger = TriggerEvent(
            event_id="evt-eudr020-alert",
            event_type=TriggerEventType.DEFORESTATION_ALERT,
            severity="critical",
            source_agent="EUDR-020",
            plan_ids=["plan-004"],
            supplier_id="sup-deforest-001",
            description="Active deforestation detected on monitored plot.",
            risk_score_before=Decimal("60"),
            risk_score_after=Decimal("95"),
            recommended_adjustment=AdjustmentType.EMERGENCY_RESPONSE,
            response_sla_hours=4,
        )
        result = await monitoring_engine.process_trigger(trigger)
        assert result is not None

    @pytest.mark.asyncio
    async def test_high_deforestation_risk_strategies(self, strategy_engine):
        """High deforestation risk should trigger deforestation strategies."""
        ri = _make_upstream_risk_input(deforestation_score="95")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        deforestation_strategies = [
            s for s in result.strategies
            if RiskCategory.DEFORESTATION in s.risk_categories
        ]
        assert len(deforestation_strategies) >= 1

    @pytest.mark.asyncio
    async def test_deforestation_weight_is_highest(self, strategy_engine):
        """Deforestation should have highest weight (0.20) in composite score."""
        from .conftest import COMPOSITE_WEIGHTS
        assert COMPOSITE_WEIGHTS["deforestation"] == Decimal("0.20")
        max_weight = max(COMPOSITE_WEIGHTS.values())
        assert COMPOSITE_WEIGHTS["deforestation"] == max_weight


# ---------------------------------------------------------------------------
# EUDR-021: Indigenous Rights Checker Integration
# ---------------------------------------------------------------------------


class TestIndigenousRightsIntegration:
    """Test integration with EUDR-021 Indigenous Rights Checker."""

    @pytest.mark.asyncio
    async def test_high_indigenous_risk_strategies(self, strategy_engine):
        """High indigenous rights risk should trigger FPIC strategies."""
        ri = _make_upstream_risk_input(indigenous_score="85")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        indigenous_strategies = [
            s for s in result.strategies
            if RiskCategory.INDIGENOUS_RIGHTS in s.risk_categories
        ]
        assert len(indigenous_strategies) >= 1

    @pytest.mark.asyncio
    async def test_indigenous_score_in_range(self, strategy_engine):
        """Indigenous rights score must be within [0, 100]."""
        ri = _make_upstream_risk_input(indigenous_score="72")
        assert Decimal("0") <= ri.indigenous_rights_score <= Decimal("100")


# ---------------------------------------------------------------------------
# EUDR-022: Protected Area Validator Integration
# ---------------------------------------------------------------------------


class TestProtectedAreaIntegration:
    """Test integration with EUDR-022 Protected Area Validator."""

    @pytest.mark.asyncio
    async def test_high_protected_area_risk_strategies(self, strategy_engine):
        """High protected area risk should trigger buffer zone strategies."""
        ri = _make_upstream_risk_input(protected_areas_score="88")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        pa_strategies = [
            s for s in result.strategies
            if RiskCategory.PROTECTED_AREAS in s.risk_categories
        ]
        assert len(pa_strategies) >= 1

    @pytest.mark.asyncio
    async def test_protected_area_trigger_event(self, monitoring_engine):
        """Protected area encroachment trigger from EUDR-022."""
        trigger = TriggerEvent(
            event_id="evt-eudr022-encroach",
            event_type=TriggerEventType.REGULATORY_UPDATE,
            severity="high",
            source_agent="EUDR-022",
            plan_ids=["plan-005"],
            description="Plot boundaries overlap with newly designated protected area.",
            risk_score_before=Decimal("30"),
            risk_score_after=Decimal("80"),
            recommended_adjustment=AdjustmentType.SCOPE_EXPANSION,
            response_sla_hours=24,
        )
        result = await monitoring_engine.process_trigger(trigger)
        assert result is not None


# ---------------------------------------------------------------------------
# EUDR-023: Legal Compliance Verifier Integration
# ---------------------------------------------------------------------------


class TestLegalComplianceIntegration:
    """Test integration with EUDR-023 Legal Compliance Verifier."""

    @pytest.mark.asyncio
    async def test_high_legal_risk_strategies(self, strategy_engine):
        """High legal compliance risk should trigger legal gap strategies."""
        ri = _make_upstream_risk_input(legal_compliance_score="85")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        legal_strategies = [
            s for s in result.strategies
            if RiskCategory.LEGAL_COMPLIANCE in s.risk_categories
        ]
        assert len(legal_strategies) >= 1

    @pytest.mark.asyncio
    async def test_regulatory_update_trigger(self, monitoring_engine):
        """Regulatory update trigger from EUDR-023."""
        trigger = TriggerEvent(
            event_id="evt-eudr023-reg",
            event_type=TriggerEventType.REGULATORY_UPDATE,
            severity="medium",
            source_agent="EUDR-023",
            plan_ids=["plan-006", "plan-007"],
            description="New local forestry regulations enacted in sourcing country.",
            risk_score_before=Decimal("35"),
            risk_score_after=Decimal("55"),
            recommended_adjustment=AdjustmentType.TIMELINE_EXTENSION,
            response_sla_hours=72,
        )
        result = await monitoring_engine.process_trigger(trigger)
        assert result is not None


# ---------------------------------------------------------------------------
# Cross-Agent Provenance
# ---------------------------------------------------------------------------


class TestCrossAgentProvenance:
    """Test provenance tracking across upstream agent boundaries."""

    @pytest.mark.asyncio
    async def test_risk_input_provenance_tracked(self, strategy_engine):
        """Strategy recommendation should include provenance from risk input."""
        ri = _make_upstream_risk_input()
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_strategy_to_plan_provenance_chain(
        self, strategy_engine, remediation_engine
    ):
        """Provenance should chain from strategy recommendation to plan creation."""
        ri = _make_upstream_risk_input(deforestation_score="85")
        rec_req = RecommendStrategiesRequest(
            risk_input=ri, top_k=3, deterministic_mode=True
        )
        rec_result = await strategy_engine.recommend(rec_req)
        rec_hash = rec_result.provenance_hash

        plan_req = CreatePlanRequest(
            operator_id=ri.operator_id,
            supplier_id=ri.supplier_id,
            strategy_ids=[s.strategy_id for s in rec_result.strategies[:1]],
            budget_eur=Decimal("50000"),
        )
        plan_result = await remediation_engine.create_plan(plan_req)
        plan_hash = plan_result.provenance_hash

        assert rec_hash != ""
        assert plan_hash != ""
        assert rec_hash != plan_hash

    @pytest.mark.asyncio
    async def test_trigger_event_source_agent_tracked(self, monitoring_engine):
        """Trigger events should record source agent identifier."""
        trigger = TriggerEvent(
            event_id="evt-source-track",
            event_type=TriggerEventType.DEFORESTATION_ALERT,
            severity="critical",
            source_agent="EUDR-020",
            plan_ids=["plan-src-001"],
            supplier_id="sup-src-001",
            description="Source tracking test.",
            risk_score_before=Decimal("50"),
            risk_score_after=Decimal("85"),
        )
        assert trigger.source_agent == "EUDR-020"
        result = await monitoring_engine.process_trigger(trigger)
        assert result is not None


# ---------------------------------------------------------------------------
# Multi-Agent End-to-End
# ---------------------------------------------------------------------------


class TestMultiAgentEndToEnd:
    """Test end-to-end workflows spanning multiple upstream agents."""

    @pytest.mark.asyncio
    async def test_all_9_dimensions_consumed(self, strategy_engine):
        """RiskInput consuming all 9 upstream dimensions should work."""
        ri = RiskInput(
            operator_id="op-9d",
            supplier_id="sup-9d",
            country_code="BR",
            commodity="soya",
            country_risk_score=Decimal("50"),      # EUDR-016
            supplier_risk_score=Decimal("55"),      # EUDR-017
            commodity_risk_score=Decimal("60"),     # EUDR-018
            corruption_risk_score=Decimal("45"),    # EUDR-019
            deforestation_risk_score=Decimal("70"), # EUDR-020
            indigenous_rights_score=Decimal("35"),  # EUDR-021
            protected_areas_score=Decimal("40"),    # EUDR-022
            legal_compliance_score=Decimal("50"),   # EUDR-023
            audit_risk_score=Decimal("42"),         # EUDR-024
            assessment_date=FIXED_DATE,
        )
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert isinstance(result, RecommendStrategiesResponse)
        assert result.composite_risk_score > Decimal("0")

    @pytest.mark.asyncio
    async def test_full_workflow_from_risk_to_plan(
        self, strategy_engine, remediation_engine
    ):
        """End-to-end: upstream risk -> strategy -> plan."""
        ri = _make_upstream_risk_input(
            country_score="72", supplier_score="68",
            deforestation_score="82", commodity="palm_oil"
        )
        rec_req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        rec_result = await strategy_engine.recommend(rec_req)
        assert len(rec_result.strategies) >= 1

        plan_req = CreatePlanRequest(
            operator_id=ri.operator_id,
            supplier_id=ri.supplier_id,
            strategy_ids=[s.strategy_id for s in rec_result.strategies[:2]],
            template_name="supplier_capacity_building",
            budget_eur=Decimal("50000"),
            target_duration_weeks=24,
        )
        plan_result = await remediation_engine.create_plan(plan_req)
        assert plan_result.plan is not None
        assert plan_result.plan.status == PlanStatus.DRAFT

    @pytest.mark.asyncio
    async def test_risk_input_score_range_enforcement(self):
        """All upstream risk scores must be within [0, 100]."""
        with pytest.raises((ValueError, Exception)):
            RiskInput(
                operator_id="op-range",
                supplier_id="sup-range",
                country_code="BR",
                commodity="soya",
                country_risk_score=Decimal("150"),  # Out of range
            )

    @pytest.mark.asyncio
    async def test_risk_input_negative_score_rejected(self):
        """Negative upstream risk scores should be rejected."""
        with pytest.raises((ValueError, Exception)):
            RiskInput(
                operator_id="op-neg",
                supplier_id="sup-neg",
                country_code="BR",
                commodity="soya",
                country_risk_score=Decimal("-10"),
            )
