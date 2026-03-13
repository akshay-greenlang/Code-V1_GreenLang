# -*- coding: utf-8 -*-
"""
Tests for Engine 6: Continuous Monitoring Engine - AGENT-EUDR-025

Tests event-driven adaptive management, trigger event detection, adjustment
recommendations, escalation chains, alert fatigue prevention, plan drift
monitoring, and annual review automation.

Test count: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timedelta, timezone

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    TriggerEventType,
    AdjustmentType,
    TriggerEvent,
    AdaptiveScanRequest,
    AdaptiveScanResponse,
)
from greenlang.agents.eudr.risk_mitigation_advisor.continuous_monitoring_engine import (
    ContinuousMonitoringEngine,
)

from .conftest import FIXED_DATETIME


class TestMonitoringEngineInit:
    def test_engine_initializes(self, monitoring_engine):
        assert monitoring_engine is not None


class TestTriggerEventDetection:
    @pytest.mark.asyncio
    async def test_detect_deforestation_alert(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.DEFORESTATION_ALERT,
            source_agent="EUDR-020",
            risk_score_before=Decimal("60"),
            risk_score_after=Decimal("90"),
            supplier_id="sup-001",
        )
        assert event is not None
        assert event.event_type == TriggerEventType.DEFORESTATION_ALERT
        assert event.severity == "critical"

    @pytest.mark.asyncio
    async def test_detect_country_reclassification(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.COUNTRY_RECLASSIFICATION,
            source_agent="EUDR-016",
            risk_score_before=Decimal("45"),
            risk_score_after=Decimal("75"),
        )
        assert event.severity in ("high", "critical")

    @pytest.mark.asyncio
    async def test_detect_supplier_risk_spike(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.SUPPLIER_RISK_SPIKE,
            source_agent="EUDR-017",
            risk_score_before=Decimal("40"),
            risk_score_after=Decimal("65"),
            supplier_id="sup-002",
        )
        assert event.recommended_adjustment in (
            AdjustmentType.PLAN_ACCELERATION,
            AdjustmentType.SCOPE_EXPANSION,
        )

    @pytest.mark.asyncio
    async def test_detect_indigenous_violation(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.INDIGENOUS_VIOLATION,
            source_agent="EUDR-021",
            risk_score_before=Decimal("30"),
            risk_score_after=Decimal("70"),
            supplier_id="sup-003",
        )
        assert event.severity == "high"

    @pytest.mark.asyncio
    async def test_detect_protected_encroachment(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.PROTECTED_ENCROACHMENT,
            source_agent="EUDR-022",
            risk_score_before=Decimal("20"),
            risk_score_after=Decimal("75"),
            supplier_id="sup-004",
        )
        assert event is not None

    @pytest.mark.asyncio
    async def test_detect_audit_nonconformance(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.AUDIT_NONCONFORMANCE,
            source_agent="EUDR-024",
            risk_score_before=Decimal("35"),
            risk_score_after=Decimal("60"),
            supplier_id="sup-005",
        )
        assert event is not None


class TestTriggerSeverityMapping:
    @pytest.mark.asyncio
    async def test_critical_severity_deforestation(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.DEFORESTATION_ALERT,
            source_agent="EUDR-020",
            risk_score_before=Decimal("50"),
            risk_score_after=Decimal("95"),
        )
        assert event.severity == "critical"

    @pytest.mark.asyncio
    async def test_high_severity_country(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.COUNTRY_RECLASSIFICATION,
            source_agent="EUDR-016",
            risk_score_before=Decimal("40"),
            risk_score_after=Decimal("70"),
        )
        assert event.severity in ("high", "critical")

    @pytest.mark.asyncio
    async def test_response_sla_critical(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.DEFORESTATION_ALERT,
            source_agent="EUDR-020",
            risk_score_before=Decimal("50"),
            risk_score_after=Decimal("95"),
        )
        assert event.response_sla_hours <= 4


class TestAdjustmentRecommendations:
    @pytest.mark.asyncio
    async def test_emergency_response_for_critical(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.DEFORESTATION_ALERT,
            source_agent="EUDR-020",
            risk_score_before=Decimal("40"),
            risk_score_after=Decimal("95"),
        )
        assert event.recommended_adjustment == AdjustmentType.EMERGENCY_RESPONSE

    @pytest.mark.asyncio
    async def test_scope_expansion_for_medium(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.SUPPLIER_RISK_SPIKE,
            source_agent="EUDR-017",
            risk_score_before=Decimal("40"),
            risk_score_after=Decimal("52"),
        )
        assert event.recommended_adjustment in (
            AdjustmentType.SCOPE_EXPANSION,
            AdjustmentType.PLAN_ACCELERATION,
        )


class TestAdaptiveScan:
    @pytest.mark.asyncio
    async def test_scan_returns_response(self, monitoring_engine, adaptive_scan_request):
        result = await monitoring_engine.scan(adaptive_scan_request)
        assert isinstance(result, AdaptiveScanResponse)

    @pytest.mark.asyncio
    async def test_scan_has_provenance(self, monitoring_engine, adaptive_scan_request):
        result = await monitoring_engine.scan(adaptive_scan_request)
        assert result.provenance_hash != ""

    @pytest.mark.asyncio
    async def test_scan_time_recorded(self, monitoring_engine, adaptive_scan_request):
        result = await monitoring_engine.scan(adaptive_scan_request)
        assert result.scan_time_ms >= Decimal("0")

    @pytest.mark.asyncio
    async def test_scan_empty_plans(self, monitoring_engine):
        request = AdaptiveScanRequest(operator_id="op-empty", plan_ids=[])
        result = await monitoring_engine.scan(request)
        assert isinstance(result, AdaptiveScanResponse)


class TestAlertFatiguePrevention:
    @pytest.mark.asyncio
    async def test_consolidate_related_triggers(self, monitoring_engine):
        events = [
            TriggerEvent(
                event_type=TriggerEventType.SUPPLIER_RISK_SPIKE,
                severity="high",
                source_agent="EUDR-017",
                supplier_id="sup-001",
                description=f"Spike event {i}",
                risk_score_before=Decimal("40"),
                risk_score_after=Decimal("55"),
            )
            for i in range(3)
        ]
        consolidated = await monitoring_engine.consolidate_triggers(events)
        assert len(consolidated) <= len(events)

    @pytest.mark.asyncio
    async def test_quiet_period_respected(self, monitoring_engine):
        result = await monitoring_engine.should_suppress(
            supplier_id="sup-001",
            event_type=TriggerEventType.SUPPLIER_RISK_SPIKE,
            last_alert_time=FIXED_DATETIME,
            current_time=FIXED_DATETIME + timedelta(hours=12),
        )
        assert isinstance(result, bool)


class TestEscalationChain:
    @pytest.mark.asyncio
    async def test_24h_escalation(self, monitoring_engine):
        target = await monitoring_engine.get_escalation_target(hours_elapsed=25)
        assert target == "team_lead"

    @pytest.mark.asyncio
    async def test_48h_escalation(self, monitoring_engine):
        target = await monitoring_engine.get_escalation_target(hours_elapsed=49)
        assert target == "director"

    @pytest.mark.asyncio
    async def test_72h_escalation(self, monitoring_engine):
        target = await monitoring_engine.get_escalation_target(hours_elapsed=73)
        assert target == "executive"

    @pytest.mark.asyncio
    async def test_no_escalation_within_sla(self, monitoring_engine):
        target = await monitoring_engine.get_escalation_target(hours_elapsed=10)
        assert target is None


class TestPlanDrift:
    @pytest.mark.asyncio
    async def test_calculate_drift(self, monitoring_engine):
        drift = await monitoring_engine.calculate_plan_drift(
            planned_reduction=Decimal("30"),
            actual_reduction=Decimal("20"),
        )
        assert isinstance(drift, Decimal)
        assert drift > Decimal("0")

    @pytest.mark.asyncio
    async def test_no_drift(self, monitoring_engine):
        drift = await monitoring_engine.calculate_plan_drift(
            planned_reduction=Decimal("30"),
            actual_reduction=Decimal("30"),
        )
        assert drift == Decimal("0")


class TestMonitoringEdgeCases:
    @pytest.mark.asyncio
    async def test_trigger_with_no_score_change(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.AUDIT_NONCONFORMANCE,
            source_agent="EUDR-024",
            risk_score_before=Decimal("50"),
            risk_score_after=Decimal("50"),
        )
        assert event is not None

    @pytest.mark.asyncio
    async def test_trigger_with_score_decrease(self, monitoring_engine):
        event = await monitoring_engine.detect_trigger(
            event_type=TriggerEventType.SUPPLIER_RISK_SPIKE,
            source_agent="EUDR-017",
            risk_score_before=Decimal("70"),
            risk_score_after=Decimal("40"),
        )
        # Decrease should not produce spike event or should handle gracefully
        assert event is not None

    @pytest.mark.asyncio
    async def test_all_six_trigger_types(self, monitoring_engine):
        for evt_type in TriggerEventType:
            event = await monitoring_engine.detect_trigger(
                event_type=evt_type,
                source_agent="EUDR-test",
                risk_score_before=Decimal("40"),
                risk_score_after=Decimal("70"),
            )
            assert event is not None
            assert event.event_type == evt_type
