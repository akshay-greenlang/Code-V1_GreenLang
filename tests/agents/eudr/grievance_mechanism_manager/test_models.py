# -*- coding: utf-8 -*-
"""
Unit tests for Grievance Mechanism Manager models - AGENT-EUDR-032

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.grievance_mechanism_manager.models import (
    AGENT_ID,
    AGENT_VERSION,
    MEDIATION_STAGES_ORDERED,
    SEVERITY_SCORES,
    AnalysisMethod,
    AuditAction,
    AuditEntry,
    CausalChainStep,
    CollectiveDemand,
    CollectiveGrievanceRecord,
    CollectiveStatus,
    GrievanceAnalyticsRecord,
    HealthStatus,
    ImplementationStatus,
    MediationRecord,
    MediationSession,
    MediationStage,
    MediatorType,
    NegotiationStatus,
    PatternType,
    RegulatoryReport,
    RegulatoryReportType,
    RemediationAction,
    RemediationRecord,
    RemediationType,
    ReportSection,
    RiskLevel,
    RiskScope,
    RiskScoreRecord,
    ScoreFactor,
    SettlementStatus,
    TrendDirection,
)


class TestConstants:
    def test_agent_id(self):
        assert AGENT_ID == "GL-EUDR-GMM-032"

    def test_agent_version(self):
        assert AGENT_VERSION == "1.0.0"

    def test_mediation_stages_ordered(self):
        assert len(MEDIATION_STAGES_ORDERED) == 7
        assert MEDIATION_STAGES_ORDERED[0] == MediationStage.INITIATED
        assert MEDIATION_STAGES_ORDERED[-1] == MediationStage.CLOSED

    def test_severity_scores(self):
        assert SEVERITY_SCORES["critical"] == 100
        assert SEVERITY_SCORES["low"] == 25


class TestEnums:
    def test_pattern_type_values(self):
        assert len(PatternType) == 5
        assert PatternType.RECURRING.value == "recurring"

    def test_trend_direction_values(self):
        assert len(TrendDirection) == 3

    def test_analysis_method_values(self):
        assert len(AnalysisMethod) == 4
        assert AnalysisMethod.FIVE_WHYS.value == "five_whys"

    def test_mediation_stage_values(self):
        assert len(MediationStage) == 7
        assert MediationStage.INITIATED.value == "initiated"

    def test_mediator_type_values(self):
        assert len(MediatorType) == 4

    def test_settlement_status_values(self):
        assert len(SettlementStatus) == 4

    def test_remediation_type_values(self):
        assert len(RemediationType) == 5
        assert RemediationType.COMPENSATION.value == "compensation"

    def test_implementation_status_values(self):
        assert len(ImplementationStatus) == 5

    def test_risk_scope_values(self):
        assert len(RiskScope) == 4

    def test_risk_level_values(self):
        assert len(RiskLevel) == 5

    def test_collective_status_values(self):
        assert len(CollectiveStatus) == 6

    def test_negotiation_status_values(self):
        assert len(NegotiationStatus) == 5

    def test_regulatory_report_type_values(self):
        assert len(RegulatoryReportType) == 4

    def test_audit_action_values(self):
        assert len(AuditAction) == 11


class TestGrievanceAnalyticsRecord:
    def test_create(self, sample_analytics_record):
        assert sample_analytics_record.analytics_id == "ana-001"
        assert sample_analytics_record.pattern_type == PatternType.RECURRING

    def test_defaults(self):
        record = GrievanceAnalyticsRecord(analytics_id="a1", operator_id="op1")
        assert record.pattern_type == PatternType.ISOLATED
        assert record.trend_direction == TrendDirection.STABLE
        assert record.affected_stakeholder_count == 0

    def test_severity_distribution(self, sample_analytics_record):
        assert sample_analytics_record.severity_distribution["high"] == 3


class TestMediationRecord:
    def test_create(self):
        record = MediationRecord(
            mediation_id="med-001",
            grievance_id="g-001",
            operator_id="OP-001",
        )
        assert record.mediation_stage == MediationStage.INITIATED
        assert record.settlement_status == SettlementStatus.PENDING
        assert record.session_count == 0

    def test_session_tracking(self):
        session = MediationSession(
            session_number=1, duration_minutes=120,
            summary="Initial dialogue", attendees=["A", "B"],
        )
        assert session.session_number == 1
        assert session.duration_minutes == 120


class TestRemediationRecord:
    def test_create(self):
        record = RemediationRecord(
            remediation_id="rem-001",
            grievance_id="g-001",
            operator_id="OP-001",
        )
        assert record.implementation_status == ImplementationStatus.PLANNED
        assert record.completion_percentage == Decimal("0")
        assert record.cost_incurred == Decimal("0")

    def test_action_model(self):
        action = RemediationAction(
            action="Install filters", status="pending",
            responsible_party="Engineering",
        )
        assert action.action == "Install filters"


class TestRiskScoreRecord:
    def test_create(self):
        record = RiskScoreRecord(
            risk_score_id="risk-001",
            operator_id="OP-001",
            scope_identifier="OP-001",
            risk_score=Decimal("65"),
            risk_level=RiskLevel.HIGH,
        )
        assert record.risk_score == Decimal("65")
        assert record.risk_level == RiskLevel.HIGH

    def test_score_factor(self):
        factor = ScoreFactor(
            factor_name="frequency",
            weight=Decimal("0.30"),
            raw_value=Decimal("5"),
            weighted_value=Decimal("15"),
        )
        assert factor.factor_name == "frequency"


class TestCollectiveGrievanceRecord:
    def test_create(self):
        record = CollectiveGrievanceRecord(
            collective_id="cg-001",
            operator_id="OP-001",
            title="Community Water Rights",
        )
        assert record.collective_status == CollectiveStatus.FORMING
        assert record.negotiation_status == NegotiationStatus.NOT_STARTED

    def test_demand_model(self):
        demand = CollectiveDemand(
            demand="Clean water", priority="critical", negotiable=False,
        )
        assert demand.demand == "Clean water"
        assert demand.negotiable is False


class TestRegulatoryReport:
    def test_create(self):
        report = RegulatoryReport(
            report_id="rpt-001",
            operator_id="OP-001",
            total_grievances=50,
            resolved_count=40,
        )
        assert report.report_type == RegulatoryReportType.ANNUAL_SUMMARY
        assert report.total_grievances == 50

    def test_section_model(self):
        section = ReportSection(
            title="Summary",
            content={"total": 50},
            regulatory_reference="EUDR Art. 16",
        )
        assert section.title == "Summary"


class TestAuditEntry:
    def test_create(self):
        entry = AuditEntry(
            entry_id="aud-001",
            entity_type="mediation",
            entity_id="med-001",
            actor="system",
        )
        assert entry.action == AuditAction.CREATE


class TestHealthStatus:
    def test_defaults(self):
        health = HealthStatus()
        assert health.agent_id == AGENT_ID
        assert health.status == "healthy"
