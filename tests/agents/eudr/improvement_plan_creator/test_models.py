# -*- coding: utf-8 -*-
"""
Unit tests for models.py - AGENT-EUDR-035

Tests all enumerations, model creation, defaults, Decimal fields,
constants, serialization, and optional fields.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.improvement_plan_creator.models import (
    AGENT_ID,
    AGENT_VERSION,
    SUPPORTED_COMMODITIES,
    ActionStatus,
    ActionType,
    AggregatedFindings,
    ComplianceGap,
    EisenhowerQuadrant,
    EUDRCommodity,
    Finding,
    FindingSource,
    FishboneAnalysis,
    FishboneCategory,
    GapSeverity,
    HealthStatus,
    ImprovementAction,
    ImprovementPlan,
    NotificationChannel,
    NotificationRecord,
    PlanStatus,
    PlanSummary,
    PlanReport,
    ProgressMilestone,
    ProgressSnapshot,
    RACIRole,
    RiskLevel,
    RootCause,
    StakeholderAssignment,
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
        expected = {"negligible", "low", "standard", "high", "critical"}
        actual = {s.value for s in RiskLevel}
        assert actual == expected
        assert len(RiskLevel) == 5

    def test_gap_severity_values(self):
        expected = {"critical", "high", "medium", "low", "informational"}
        actual = {s.value for s in GapSeverity}
        assert actual == expected
        assert len(GapSeverity) == 5

    def test_action_status_values(self):
        expected = {
            "draft", "proposed", "approved", "in_progress", "on_hold",
            "completed", "verified", "closed", "cancelled",
        }
        actual = {s.value for s in ActionStatus}
        assert actual == expected
        assert len(ActionStatus) == 9

    def test_action_type_values(self):
        expected = {
            "corrective", "preventive", "monitoring_enhancement",
            "documentation_update", "training", "process_change",
            "supplier_engagement", "technology_upgrade",
            "audit_enhancement", "policy_update",
        }
        actual = {t.value for t in ActionType}
        assert actual == expected
        assert len(ActionType) == 10

    def test_plan_status_values(self):
        expected = {
            "draft", "under_review", "approved", "active",
            "completed", "archived", "cancelled",
        }
        actual = {s.value for s in PlanStatus}
        assert actual == expected
        assert len(PlanStatus) == 7

    def test_eisenhower_quadrant_values(self):
        expected = {"do_first", "schedule", "delegate", "eliminate"}
        actual = {q.value for q in EisenhowerQuadrant}
        assert actual == expected
        assert len(EisenhowerQuadrant) == 4

    def test_raci_role_values(self):
        expected = {"responsible", "accountable", "consulted", "informed"}
        actual = {r.value for r in RACIRole}
        assert actual == expected
        assert len(RACIRole) == 4

    def test_finding_source_values(self):
        expected = {
            "risk_assessment", "country_risk", "supplier_risk",
            "commodity_risk", "deforestation_alert", "legal_compliance",
            "document_authentication", "satellite_monitoring",
            "mitigation_measure", "audit_manager", "manual",
        }
        actual = {s.value for s in FindingSource}
        assert actual == expected
        assert len(FindingSource) == 11

    def test_fishbone_category_values(self):
        expected = {
            "people", "process", "technology", "data",
            "policy", "environment", "suppliers", "management",
        }
        actual = {c.value for c in FishboneCategory}
        assert actual == expected
        assert len(FishboneCategory) == 8

    def test_notification_channel_values(self):
        expected = {"email", "slack", "webhook", "in_app", "sms"}
        actual = {c.value for c in NotificationChannel}
        assert actual == expected
        assert len(NotificationChannel) == 5


class TestConstants:
    """Test module-level constants."""

    def test_agent_id(self):
        assert AGENT_ID == "GL-EUDR-IPC-035"

    def test_agent_version(self):
        assert AGENT_VERSION == "1.0.0"

    def test_supported_commodities(self):
        assert len(SUPPORTED_COMMODITIES) == 7
        assert "coffee" in SUPPORTED_COMMODITIES
        assert "wood" in SUPPORTED_COMMODITIES



class TestFindingModel:
    """Test Finding model creation and defaults."""

    def test_create_valid_finding(self, sample_finding):
        assert sample_finding.finding_id == "fnd-test-001"
        assert sample_finding.operator_id == "operator-001"
        assert sample_finding.source == FindingSource.RISK_ASSESSMENT
        assert sample_finding.severity == GapSeverity.HIGH

    def test_finding_has_commodity(self, sample_finding):
        assert sample_finding.commodity == EUDRCommodity.COCOA

    def test_finding_has_eudr_article_ref(self, sample_finding):
        assert "Article 9" in sample_finding.eudr_article_ref

    def test_finding_has_metadata(self, sample_finding):
        """Test that findings can include metadata dictionary."""
        assert isinstance(sample_finding.metadata, dict)

    def test_finding_has_risk_score(self, sample_finding):
        assert sample_finding.risk_score == Decimal("85.00")


class TestComplianceGapModel:
    """Test ComplianceGap model creation and defaults."""

    def test_create_valid_gap(self, sample_gap):
        assert sample_gap.gap_id == "gap-test-001"
        assert sample_gap.plan_id == "plan-001"
        assert sample_gap.severity == GapSeverity.HIGH

    def test_gap_has_current_and_required_state(self, sample_gap):
        assert sample_gap.current_state != ""
        assert sample_gap.required_state != ""

    def test_gap_has_severity_score(self, sample_gap):
        assert sample_gap.severity_score == Decimal("0.725")
        assert sample_gap.severity_score >= Decimal("0")
        assert sample_gap.severity_score <= Decimal("1")

    def test_gap_has_finding_ids(self, sample_gap):
        assert "fnd-test-001" in sample_gap.finding_ids

    def test_gap_has_eudr_article_ref(self, sample_gap):
        assert sample_gap.eudr_article_ref == "Article 9(1)(d)"

    def test_gap_has_provenance_hash(self, sample_gap):
        assert len(sample_gap.provenance_hash) == 64


class TestImprovementActionModel:
    """Test ImprovementAction model creation and defaults."""

    def test_create_valid_action(self, sample_action):
        assert sample_action.action_id == "act-test-001"
        assert sample_action.plan_id == "plan-001"
        assert sample_action.gap_id == "gap-test-001"
        assert sample_action.action_type == ActionType.CORRECTIVE
        assert sample_action.status == ActionStatus.PROPOSED

    def test_action_has_priority_score(self, sample_action):
        assert sample_action.priority_score == Decimal("85.00")

    def test_action_has_eisenhower_quadrant(self, sample_action):
        assert sample_action.eisenhower_quadrant == EisenhowerQuadrant.DO_FIRST

    def test_action_has_deadline(self, sample_action):
        assert sample_action.time_bound_deadline is not None

    def test_action_has_cost_estimate(self, sample_action):
        assert sample_action.estimated_cost == Decimal("25000.00")

    def test_action_has_effort_estimate(self, sample_action):
        assert sample_action.estimated_effort_hours == Decimal("320")

    def test_action_has_provenance_hash(self, sample_action):
        assert len(sample_action.provenance_hash) == 64


class TestRootCauseModel:
    """Test RootCause model creation and defaults."""

    def test_create_valid_root_cause(self, sample_root_cause):
        assert sample_root_cause.root_cause_id == "rc-test-001"
        assert sample_root_cause.gap_id == "gap-test-001"
        assert sample_root_cause.category == FishboneCategory.PROCESS

    def test_root_cause_has_contributing_factors(self, sample_root_cause):
        assert len(sample_root_cause.contributing_factors) == 3

    def test_root_cause_has_analysis_chain(self, sample_root_cause):
        assert len(sample_root_cause.analysis_chain) == 4

    def test_root_cause_has_depth(self, sample_root_cause):
        assert sample_root_cause.depth == 4

    def test_root_cause_has_confidence(self, sample_root_cause):
        assert sample_root_cause.confidence == Decimal("0.85")


class TestProgressSnapshotModel:
    """Test ProgressSnapshot model creation."""

    def test_create_valid_snapshot(self, sample_progress_snapshot):
        assert sample_progress_snapshot.snapshot_id == "snap-001"
        assert sample_progress_snapshot.plan_id == "plan-001"
        assert sample_progress_snapshot.overall_progress == Decimal("35.00")

    def test_snapshot_has_action_counts(self, sample_progress_snapshot):
        assert sample_progress_snapshot.actions_total == 4
        assert sample_progress_snapshot.actions_completed == 1
        assert sample_progress_snapshot.actions_in_progress == 2


class TestStakeholderAssignmentModel:
    """Test StakeholderAssignment model creation."""

    def test_create_valid_assignment(self, sample_assignment):
        assert sample_assignment.assignment_id == "asgn-test-001"
        assert sample_assignment.action_id == "act-test-001"
        assert sample_assignment.role == RACIRole.RESPONSIBLE

    def test_assignment_has_stakeholder_info(self, sample_assignment):
        assert sample_assignment.stakeholder_name == "Maria Sustainability"
        assert sample_assignment.stakeholder_email == "maria.sustainability@company.com"


class TestImprovementPlanModel:
    """Test ImprovementPlan model creation."""

    def test_create_valid_plan(self, sample_plan):
        assert sample_plan.plan_id == "plan-001"
        assert sample_plan.operator_id == "operator-001"
        assert sample_plan.status == PlanStatus.DRAFT

    def test_plan_has_commodity(self, sample_plan):
        assert sample_plan.commodity == EUDRCommodity.COCOA

    def test_plan_has_risk_level(self, sample_plan):
        assert sample_plan.risk_level == RiskLevel.HIGH

    def test_plan_has_counts(self, sample_plan):
        assert sample_plan.total_gaps == 3
        assert sample_plan.total_actions == 4

    def test_plan_has_target_completion(self, sample_plan):
        assert sample_plan.target_completion is not None

    def test_active_plan_status(self, active_plan):
        assert active_plan.status == PlanStatus.ACTIVE
        assert active_plan.approved_at is not None


class TestHealthStatusModel:
    """Test HealthStatus model."""

    def test_health_status_defaults(self):
        hs = HealthStatus()
        assert hs.agent_id == AGENT_ID
        assert hs.status == "healthy"
        assert hs.version == AGENT_VERSION

    def test_health_status_engines(self):
        hs = HealthStatus(engines={
            "finding_aggregator": "ok",
            "gap_analyzer": "ok",
            "action_generator": "ok",
        })
        assert len(hs.engines) == 3
