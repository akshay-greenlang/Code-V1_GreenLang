# -*- coding: utf-8 -*-
"""
Unit tests for models.py - AGENT-EUDR-029

Tests all enumerations, model creation, defaults, Decimal fields,
constants, serialization, and optional fields.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.mitigation_measure_designer.models import (
    AGENT_ID,
    AGENT_VERSION,
    DEFAULT_RISK_WEIGHTS,
    RISK_THRESHOLDS,
    SUPPORTED_COMMODITIES,
    Article11Category,
    EffectivenessEstimate,
    EffectivenessLevel,
    EUDRCommodity,
    EvidenceType,
    HealthStatus,
    ImplementationMilestone,
    MeasureEvidence,
    MeasurePriority,
    MeasureStatus,
    MeasureSummary,
    MeasureTemplate,
    MitigationMeasure,
    MitigationReport,
    MitigationStrategy,
    RiskDimension,
    RiskLevel,
    RiskTrigger,
    VerificationReport,
    VerificationResult,
    WorkflowState,
    WorkflowStatus,
)


class TestEnums:
    """Test all enum definitions and membership."""

    def test_eudr_commodity_values(self):
        assert EUDRCommodity.CATTLE == "cattle"
        assert EUDRCommodity.COCOA == "cocoa"
        assert EUDRCommodity.COFFEE == "coffee"
        assert EUDRCommodity.OIL_PALM == "oil_palm"
        assert EUDRCommodity.RUBBER == "rubber"
        assert EUDRCommodity.SOYA == "soya"
        assert EUDRCommodity.WOOD == "wood"
        assert len(EUDRCommodity) == 7

    def test_risk_level_values(self):
        assert RiskLevel.NEGLIGIBLE == "negligible"
        assert RiskLevel.LOW == "low"
        assert RiskLevel.STANDARD == "standard"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"
        assert len(RiskLevel) == 5

    def test_article11_category_values(self):
        assert Article11Category.ADDITIONAL_INFO == "additional_info"
        assert Article11Category.INDEPENDENT_AUDIT == "independent_audit"
        assert Article11Category.OTHER_MEASURES == "other_measures"
        assert len(Article11Category) == 3

    def test_measure_status_values(self):
        expected = {"proposed", "approved", "in_progress", "completed",
                    "verified", "closed", "cancelled"}
        actual = {s.value for s in MeasureStatus}
        assert actual == expected

    def test_workflow_status_values(self):
        expected = {"initiated", "strategy_designed", "measures_approved",
                    "implementing", "verifying", "closed", "escalated", "failed"}
        actual = {s.value for s in WorkflowStatus}
        assert actual == expected

    def test_measure_priority_values(self):
        expected = {"critical", "high", "medium", "low"}
        actual = {p.value for p in MeasurePriority}
        assert actual == expected

    def test_effectiveness_level_values(self):
        expected = {"high_impact", "medium_impact", "low_impact", "minimal_impact"}
        actual = {e.value for e in EffectivenessLevel}
        assert actual == expected

    def test_verification_result_values(self):
        expected = {"sufficient", "partial", "insufficient"}
        actual = {v.value for v in VerificationResult}
        assert actual == expected

    def test_risk_dimension_values(self):
        expected = {"country", "commodity", "supplier", "deforestation",
                    "corruption", "supply_chain_complexity", "mixing_risk",
                    "circumvention_risk"}
        actual = {d.value for d in RiskDimension}
        assert actual == expected
        assert len(RiskDimension) == 8

    def test_evidence_type_values(self):
        expected = {"document", "certificate", "audit_report", "satellite_image",
                    "site_visit_report", "supplier_declaration", "other"}
        actual = {e.value for e in EvidenceType}
        assert actual == expected


class TestConstants:
    """Test module-level constants."""

    def test_agent_id(self):
        assert AGENT_ID == "GL-EUDR-MMD-029"

    def test_agent_version(self):
        assert AGENT_VERSION == "1.0.0"

    def test_default_risk_weights_sum_to_one(self):
        total = sum(DEFAULT_RISK_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_default_risk_weights_has_all_dimensions(self):
        for dim in RiskDimension:
            assert dim in DEFAULT_RISK_WEIGHTS

    def test_risk_thresholds_ascending(self):
        levels = [RiskLevel.NEGLIGIBLE, RiskLevel.LOW, RiskLevel.STANDARD,
                  RiskLevel.HIGH, RiskLevel.CRITICAL]
        for i in range(len(levels) - 1):
            assert RISK_THRESHOLDS[levels[i]] < RISK_THRESHOLDS[levels[i + 1]]

    def test_supported_commodities(self):
        assert len(SUPPORTED_COMMODITIES) == 7
        assert "coffee" in SUPPORTED_COMMODITIES
        assert "wood" in SUPPORTED_COMMODITIES


class TestRiskTriggerModel:
    """Test RiskTrigger model creation and defaults."""

    def test_create_valid_risk_trigger(self, sample_risk_trigger):
        assert sample_risk_trigger.assessment_id == "assess-001"
        assert sample_risk_trigger.operator_id == "operator-001"
        assert sample_risk_trigger.commodity == EUDRCommodity.COFFEE
        assert sample_risk_trigger.composite_score == Decimal("72")
        assert sample_risk_trigger.risk_level == RiskLevel.HIGH

    def test_risk_trigger_has_dimensions(self, sample_risk_trigger):
        assert len(sample_risk_trigger.risk_dimensions) == 5
        assert RiskDimension.SUPPLIER in sample_risk_trigger.risk_dimensions

    def test_risk_trigger_triggered_at_is_datetime(self, sample_risk_trigger):
        assert isinstance(sample_risk_trigger.triggered_at, datetime)

    def test_risk_trigger_default_dimensions_empty(self):
        trigger = RiskTrigger(
            assessment_id="a",
            operator_id="o",
            commodity=EUDRCommodity.SOYA,
            composite_score=Decimal("50"),
            risk_level=RiskLevel.STANDARD,
        )
        assert trigger.risk_dimensions == {}


class TestMeasureTemplateModel:
    """Test MeasureTemplate model."""

    def test_create_valid_template(self, sample_template):
        assert sample_template.template_id == "MMD-TPL-TEST-001"
        assert sample_template.base_effectiveness == Decimal("25")

    def test_template_defaults(self):
        t = MeasureTemplate(
            template_id="t1",
            title="Minimal Template",
            article11_category=Article11Category.OTHER_MEASURES,
        )
        assert t.description == ""
        assert t.applicable_dimensions == []
        assert t.applicable_commodities == []
        assert t.base_effectiveness == Decimal("0")
        assert t.typical_timeline_days == 30
        assert t.evidence_requirements == []
        assert t.regulatory_reference == ""


class TestMitigationMeasureModel:
    """Test MitigationMeasure model."""

    def test_create_valid_measure(self, sample_measure):
        assert sample_measure.measure_id == "msr-test-001"
        assert sample_measure.status == MeasureStatus.PROPOSED
        assert sample_measure.priority == MeasurePriority.HIGH

    def test_measure_defaults(self):
        m = MitigationMeasure(
            measure_id="m1",
            strategy_id="s1",
            title="Test",
            article11_category=Article11Category.OTHER_MEASURES,
            target_dimension=RiskDimension.COUNTRY,
        )
        assert m.template_id is None
        assert m.status == MeasureStatus.PROPOSED
        assert m.priority == MeasurePriority.MEDIUM
        assert m.assigned_to is None
        assert m.deadline is None
        assert m.started_at is None
        assert m.completed_at is None
        assert m.evidence_ids == []
        assert m.expected_risk_reduction == Decimal("0")
        assert m.actual_risk_reduction is None


class TestMitigationStrategyModel:
    """Test MitigationStrategy model."""

    def test_create_valid_strategy(self, sample_strategy):
        assert sample_strategy.strategy_id == "stg-test-001"
        assert sample_strategy.pre_mitigation_score == Decimal("72")
        assert sample_strategy.target_score == Decimal("30")
        assert len(sample_strategy.measures) == 1

    def test_strategy_defaults(self, sample_risk_trigger):
        s = MitigationStrategy(
            strategy_id="s1",
            workflow_id="w1",
            risk_trigger=sample_risk_trigger,
            pre_mitigation_score=Decimal("72"),
            target_score=Decimal("30"),
        )
        assert s.measures == []
        assert s.post_mitigation_score is None
        assert s.status == WorkflowStatus.STRATEGY_DESIGNED
        assert s.designed_by == AGENT_ID
        assert s.provenance_hash == ""


class TestEffectivenessEstimateModel:
    """Test EffectivenessEstimate model."""

    def test_create_effectiveness_estimate(self):
        est = EffectivenessEstimate(
            estimate_id="eff-001",
            measure_id="msr-001",
            conservative=Decimal("14"),
            moderate=Decimal("20"),
            optimistic=Decimal("26"),
        )
        assert est.applicability_factor == Decimal("1.00")
        assert est.confidence == Decimal("0.50")
        assert est.provenance_hash == ""


class TestVerificationReportModel:
    """Test VerificationReport model."""

    def test_create_verification_report(self, sample_verification_report):
        assert sample_verification_report.verification_id == "ver-test-001"
        assert sample_verification_report.pre_score == Decimal("72")
        assert sample_verification_report.post_score == Decimal("25")
        assert sample_verification_report.risk_reduction == Decimal("47")
        assert sample_verification_report.result == VerificationResult.SUFFICIENT


class TestWorkflowStateModel:
    """Test WorkflowState model."""

    def test_create_workflow_state(self, sample_workflow):
        assert sample_workflow.workflow_id == "wfl-test-001"
        assert sample_workflow.status == WorkflowStatus.INITIATED
        assert sample_workflow.strategy_id is None
        assert sample_workflow.closed_at is None
        assert sample_workflow.escalated_at is None


class TestHealthStatusModel:
    """Test HealthStatus model."""

    def test_health_status_defaults(self):
        h = HealthStatus()
        assert h.agent_id == AGENT_ID
        assert h.status == "healthy"
        assert h.version == AGENT_VERSION
        assert h.database is False
        assert h.redis is False


class TestMitigationReportModel:
    """Test MitigationReport model."""

    def test_create_report(self):
        rpt = MitigationReport(
            report_id="rpt-001",
            strategy_id="stg-001",
            operator_id="op-001",
            commodity=EUDRCommodity.WOOD,
            pre_score=Decimal("75"),
            post_score=Decimal("25"),
        )
        assert rpt.measures_summary == []
        assert rpt.verification_result is None
        assert rpt.provenance_hash == ""


class TestMeasureSummaryModel:
    """Test MeasureSummary model."""

    def test_create_measure_summary(self):
        ms = MeasureSummary(
            measure_id="m1",
            title="Test",
            article11_category=Article11Category.ADDITIONAL_INFO,
            target_dimension=RiskDimension.COUNTRY,
            status=MeasureStatus.COMPLETED,
            priority=MeasurePriority.HIGH,
            expected_risk_reduction=Decimal("20"),
            actual_risk_reduction=Decimal("18"),
        )
        assert ms.measure_id == "m1"
        assert ms.actual_risk_reduction == Decimal("18")
