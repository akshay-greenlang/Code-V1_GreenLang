# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-035 Improvement Plan Creator tests.

Provides reusable test fixtures for config, models, findings, gaps,
actions, root causes, priorities, progress tracking, stakeholder
coordination, and provenance tracking across all test modules.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone, date
from decimal import Decimal
from typing import Dict, List, Optional

from greenlang.agents.eudr.improvement_plan_creator.config import (
    ImprovementPlanCreatorConfig,
    reset_config,
)
from greenlang.agents.eudr.improvement_plan_creator.models import (
    AGENT_ID,
    AGENT_VERSION,
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
from greenlang.agents.eudr.improvement_plan_creator.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


# ---------------------------------------------------------------------------
# Auto-reset config singleton after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before/after each test."""
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> ImprovementPlanCreatorConfig:
    """Create a default ImprovementPlanCreatorConfig instance."""
    return ImprovementPlanCreatorConfig()


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Finding fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_finding() -> Finding:
    """Create a sample finding."""
    return Finding(
        finding_id="fnd-test-001",
        operator_id="operator-001",
        source=FindingSource.RISK_ASSESSMENT,
        severity=GapSeverity.HIGH,
        title="Incomplete GPS coordinates for supplier plots",
        description=(
            "Geolocation data missing for 12 of 45 supplier production "
            "plots in Ivory Coast region."
        ),
        commodity=EUDRCommodity.COCOA,
        eudr_article_ref="Article 9(1)(d)",
        detected_at=datetime.now(tz=timezone.utc) - timedelta(days=14),
        risk_score=Decimal("85.00"),
    )


@pytest.fixture
def closed_finding() -> Finding:
    """Create a finding."""
    now = datetime.now(tz=timezone.utc)
    return Finding(
        finding_id="fnd-closed-001",
        operator_id="operator-001",
        source=FindingSource.SUPPLIER_RISK,
        severity=GapSeverity.MEDIUM,
        title="Risk score not updated for Q3 2025",
        description="Quarterly risk assessment was not refreshed.",
        commodity=EUDRCommodity.COFFEE,
        eudr_article_ref="Article 10(2)",
        detected_at=now - timedelta(days=60),
        risk_score=Decimal("45.00"),
    )


@pytest.fixture
def multiple_findings() -> List[Finding]:
    """Create multiple findings across categories and severities."""
    now = datetime.now(tz=timezone.utc)
    return [
        Finding(
            finding_id="fnd-multi-001",
            operator_id="operator-001",
            source=FindingSource.LEGAL_COMPLIANCE,
            severity=GapSeverity.CRITICAL,
            title="No DDS submitted for 5 shipments",
            description="Due diligence statements missing for palm oil imports.",
            commodity=EUDRCommodity.OIL_PALM,
            eudr_article_ref="Article 4(1)",
            detected_at=now - timedelta(days=7),
            risk_score=Decimal("95.00"),
        ),
        Finding(
            finding_id="fnd-multi-002",
            operator_id="operator-001",
            source=FindingSource.DOCUMENT_AUTHENTICATION,
            severity=GapSeverity.HIGH,
            title="Supplier certifications expired",
            description="3 suppliers have expired sustainability certifications.",
            commodity=EUDRCommodity.COFFEE,
            eudr_article_ref="Article 9(1)(g)",
            detected_at=now - timedelta(days=21),
            risk_score=Decimal("75.00"),
        ),
        Finding(
            finding_id="fnd-multi-003",
            operator_id="operator-001",
            source=FindingSource.MANUAL,
            severity=GapSeverity.LOW,
            title="Manual data entry for shipment records",
            description="Shipment data still entered manually instead of automated.",
            commodity=EUDRCommodity.SOYA,
            eudr_article_ref="Article 12",
            detected_at=now - timedelta(days=45),
            risk_score=Decimal("25.00"),
        ),
        Finding(
            finding_id="fnd-multi-004",
            operator_id="operator-001",
            source=FindingSource.COUNTRY_RISK,
            severity=GapSeverity.MEDIUM,
            title="New country risk classification not applied",
            description="Indonesia reclassified to high-risk; not reflected in system.",
            commodity=EUDRCommodity.RUBBER,
            eudr_article_ref="Article 29(3)",
            detected_at=now - timedelta(days=10),
            risk_score=Decimal("55.00"),
        ),
    ]


@pytest.fixture
def sample_aggregated_findings(sample_finding) -> AggregatedFindings:
    """Create a sample aggregated findings object."""
    return AggregatedFindings(
        aggregation_id="agg-001",
        operator_id="operator-001",
        findings=[sample_finding],
        total_findings=1,
        critical_count=0,
        high_count=1,
        medium_count=0,
        low_count=0,
        source_agents=["risk_assessment"],
        duplicates_removed=0,
        provenance_hash="z" * 64,
    )


@pytest.fixture
def multiple_aggregated_findings(multiple_findings) -> AggregatedFindings:
    """Create aggregated findings with multiple findings."""
    return AggregatedFindings(
        aggregation_id="agg-multi-001",
        operator_id="operator-001",
        findings=multiple_findings,
        total_findings=len(multiple_findings),
        critical_count=1,
        high_count=1,
        medium_count=1,
        low_count=1,
        source_agents=["legal_compliance", "document_authentication", "manual", "country_risk"],
        duplicates_removed=0,
        provenance_hash="z" * 64,
    )


# ---------------------------------------------------------------------------
# Gap fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_gap() -> ComplianceGap:
    """Create a sample compliance gap."""
    return ComplianceGap(
        gap_id="gap-test-001",
        plan_id="plan-001",
        finding_ids=["fnd-test-001"],
        severity=GapSeverity.HIGH,
        title="Geolocation verification gap",
        description=(
            "Current geolocation verification process does not meet EUDR "
            "Article 9(1)(d) requirements for plot-level precision."
        ),
        current_state="Manual GPS coordinate collection with 26% missing data",
        required_state="Automated polygon-based geolocation with <1% missing data",
        severity_score=Decimal("0.725"),
        eudr_article_ref="Article 9(1)(d)",
        commodity=EUDRCommodity.COCOA,
        risk_dimension="geolocation",
        provenance_hash="g" * 64,
    )


@pytest.fixture
def multiple_gaps() -> List[ComplianceGap]:
    """Create multiple compliance gaps."""
    now = datetime.now(tz=timezone.utc)
    return [
        ComplianceGap(
            gap_id="gap-multi-001",
            plan_id="plan-001",
            finding_ids=["fnd-multi-001"],
            severity=GapSeverity.CRITICAL,
            title="Missing due diligence statements",
            description="DDS not submitted for palm oil shipments.",
            current_state="No DDS for 5 shipments",
            required_state="100% DDS coverage",
            severity_score=Decimal("0.95"),
            eudr_article_ref="Article 4(1)",
            commodity=EUDRCommodity.OIL_PALM,
            risk_dimension="due_diligence",
            provenance_hash="h" * 64,
        ),
        ComplianceGap(
            gap_id="gap-multi-002",
            plan_id="plan-001",
            finding_ids=["fnd-multi-002"],
            severity=GapSeverity.HIGH,
            title="Certification renewal process gap",
            description="No automated tracking of certification expiry dates.",
            current_state="Manual tracking in spreadsheets",
            required_state="Automated certification monitoring with 90-day alerts",
            severity_score=Decimal("0.65"),
            eudr_article_ref="Article 9(1)(g)",
            commodity=EUDRCommodity.COFFEE,
            risk_dimension="supply_chain",
            provenance_hash="i" * 64,
        ),
        ComplianceGap(
            gap_id="gap-multi-003",
            plan_id="plan-001",
            finding_ids=["fnd-multi-003"],
            severity=GapSeverity.MEDIUM,
            title="Data entry automation gap",
            description="Shipment data relies on manual input.",
            current_state="100% manual data entry",
            required_state="Automated ERP integration",
            severity_score=Decimal("0.45"),
            eudr_article_ref="Article 12",
            commodity=EUDRCommodity.SOYA,
            risk_dimension="data_quality",
            provenance_hash="j" * 64,
        ),
    ]


# ---------------------------------------------------------------------------
# Action fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_action() -> ImprovementAction:
    """Create a sample corrective action item."""
    now = datetime.now(tz=timezone.utc)
    return ImprovementAction(
        action_id="act-test-001",
        plan_id="plan-001",
        gap_id="gap-test-001",
        action_type=ActionType.CORRECTIVE,
        title="Implement polygon-based geolocation collection",
        description=(
            "Deploy mobile application with polygon drawing capabilities "
            "for field-level geolocation data capture."
        ),
        assigned_to="geo-team@company.com",
        status=ActionStatus.PROPOSED,
        time_bound_deadline=now + timedelta(days=60),
        estimated_effort_hours=Decimal("320"),
        estimated_cost=Decimal("25000.00"),
        priority_score=Decimal("85.00"),
        eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
        urgency_score=Decimal("90.00"),
        importance_score=Decimal("85.00"),
        provenance_hash="k" * 64,
    )


@pytest.fixture
def multiple_actions() -> List[ImprovementAction]:
    """Create multiple actions with different types and statuses."""
    now = datetime.now(tz=timezone.utc)
    return [
        ImprovementAction(
            action_id="act-multi-001",
            plan_id="plan-001",
            gap_id="gap-multi-001",
            action_type=ActionType.CORRECTIVE,
            title="Submit missing DDS for palm oil shipments",
            description="Retroactively create and submit 5 DDS.",
            assigned_to="compliance@company.com",
            status=ActionStatus.IN_PROGRESS,
            time_bound_deadline=now + timedelta(days=14),
            estimated_effort_hours=Decimal("80"),
            estimated_cost=Decimal("5000.00"),
            priority_score=Decimal("95.00"),
            eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
            urgency_score=Decimal("98.00"),
            importance_score=Decimal("95.00"),
            started_at=now - timedelta(days=3),
            provenance_hash="l" * 64,
        ),
        ImprovementAction(
            action_id="act-multi-002",
            plan_id="plan-001",
            gap_id="gap-multi-002",
            action_type=ActionType.PREVENTIVE,
            title="Implement certification monitoring system",
            description="Deploy automated tracking of supplier certification expiry.",
            assigned_to="supply-chain@company.com",
            status=ActionStatus.APPROVED,
            time_bound_deadline=now + timedelta(days=75),
            estimated_effort_hours=Decimal("240"),
            estimated_cost=Decimal("15000.00"),
            priority_score=Decimal("75.00"),
            eisenhower_quadrant=EisenhowerQuadrant.SCHEDULE,
            urgency_score=Decimal("70.00"),
            importance_score=Decimal("80.00"),
            provenance_hash="m" * 64,
        ),
        ImprovementAction(
            action_id="act-multi-003",
            plan_id="plan-001",
            gap_id="gap-multi-003",
            action_type=ActionType.TECHNOLOGY_UPGRADE,
            title="Integrate ERP for automated shipment data",
            description="Build API connector to ERP for real-time shipment data.",
            assigned_to="it-team@company.com",
            status=ActionStatus.PROPOSED,
            time_bound_deadline=now + timedelta(days=120),
            estimated_effort_hours=Decimal("480"),
            estimated_cost=Decimal("50000.00"),
            priority_score=Decimal("55.00"),
            eisenhower_quadrant=EisenhowerQuadrant.SCHEDULE,
            urgency_score=Decimal("50.00"),
            importance_score=Decimal("60.00"),
            provenance_hash="n" * 64,
        ),
        ImprovementAction(
            action_id="act-multi-004",
            plan_id="plan-001",
            gap_id="gap-multi-001",
            action_type=ActionType.CORRECTIVE,
            title="Update country risk classification tables",
            description="Apply latest EU commission country risk assessments.",
            assigned_to="risk-team@company.com",
            status=ActionStatus.COMPLETED,
            time_bound_deadline=now - timedelta(days=3),
            estimated_effort_hours=Decimal("40"),
            estimated_cost=Decimal("2000.00"),
            actual_effort_hours=Decimal("38"),
            actual_cost=Decimal("1900.00"),
            priority_score=Decimal("80.00"),
            eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
            urgency_score=Decimal("85.00"),
            importance_score=Decimal("75.00"),
            started_at=now - timedelta(days=10),
            completed_at=now - timedelta(days=2),
            provenance_hash="o" * 64,
        ),
    ]


# ---------------------------------------------------------------------------
# Root Cause fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_root_cause() -> RootCause:
    """Create a sample root cause analysis record."""
    return RootCause(
        root_cause_id="rc-test-001",
        gap_id="gap-test-001",
        category=FishboneCategory.PROCESS,
        description=(
            "No documented SOP for field-level geolocation capture. "
            "Suppliers use inconsistent methods leading to data gaps."
        ),
        contributing_factors=[
            "No training for field agents",
            "No standardized mobile application",
            "No quality checks on GPS data at submission",
        ],
        analysis_chain=[
            "GPS data is incomplete",
            "Field agents lack standard procedures",
            "No documented SOP exists",
            "Process standardization was not prioritized",
        ],
        depth=4,
        confidence=Decimal("0.85"),
        systemic=True,
        provenance_hash="p" * 64,
    )


@pytest.fixture
def multiple_root_causes() -> List[RootCause]:
    """Create multiple root causes across categories."""
    return [
        RootCause(
            root_cause_id="rc-multi-001",
            gap_id="gap-multi-001",
            category=FishboneCategory.PEOPLE,
            description="Only 1 compliance officer managing 120+ shipments.",
            contributing_factors=[
                "Understaffing in compliance department",
                "High volume of palm oil shipments",
            ],
            analysis_chain=[
                "DDS submissions are delayed",
                "Compliance officer is overwhelmed",
                "Insufficient staffing in compliance team",
            ],
            depth=3,
            confidence=Decimal("0.90"),
            systemic=True,
            provenance_hash="q" * 64,
        ),
        RootCause(
            root_cause_id="rc-multi-002",
            gap_id="gap-multi-002",
            category=FishboneCategory.TECHNOLOGY,
            description="Spreadsheet-based tracking fails to send expiry reminders.",
            contributing_factors=[
                "Legacy spreadsheet-based system",
                "No integration with supplier portals",
                "No automated alerting",
            ],
            analysis_chain=[
                "Certifications expire without notice",
                "No automated reminder system",
                "Spreadsheet-based manual tracking",
                "Technology upgrade not prioritized",
            ],
            depth=4,
            confidence=Decimal("0.75"),
            systemic=False,
            provenance_hash="r" * 64,
        ),
        RootCause(
            root_cause_id="rc-multi-003",
            gap_id="gap-multi-003",
            category=FishboneCategory.DATA,
            description="No automated data flow from logistics systems.",
            contributing_factors=[
                "ERP system lacks API capabilities",
                "Budget constraints for integration project",
            ],
            analysis_chain=[
                "Manual data entry errors occur",
                "No automated data transfer",
                "ERP integration not implemented",
            ],
            depth=3,
            confidence=Decimal("0.60"),
            systemic=False,
            provenance_hash="s" * 64,
        ),
    ]


# ---------------------------------------------------------------------------
# Progress fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_progress_milestone() -> ProgressMilestone:
    """Create a sample progress milestone."""
    now = datetime.now(tz=timezone.utc)
    return ProgressMilestone(
        milestone_id="ms-001",
        action_id="act-test-001",
        title="Requirements gathering complete",
        description="Complete requirements gathering phase",
        due_date=now + timedelta(days=14),
        status=ActionStatus.COMPLETED,
        weight=Decimal("0.25"),
    )


@pytest.fixture
def sample_progress_snapshot() -> ProgressSnapshot:
    """Create a sample progress snapshot."""
    now = datetime.now(tz=timezone.utc)
    return ProgressSnapshot(
        snapshot_id="snap-001",
        plan_id="plan-001",
        overall_progress=Decimal("35.00"),
        actions_total=4,
        actions_completed=1,
        actions_in_progress=2,
        actions_overdue=0,
        actions_on_hold=0,
        gaps_closed=0,
        gaps_total=3,
        avg_effectiveness_score=Decimal("75.00"),
        on_track=True,
        risk_trend="stable",
        captured_at=now,
        provenance_hash="u" * 64,
    )


# ---------------------------------------------------------------------------
# Improvement Plan fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_plan() -> ImprovementPlan:
    """Create a sample improvement plan in DRAFT status."""
    now = datetime.now(tz=timezone.utc)
    return ImprovementPlan(
        plan_id="plan-001",
        operator_id="operator-001",
        title="EUDR Compliance Improvement Plan Q1 2026",
        description=(
            "Comprehensive improvement plan addressing audit findings "
            "and compliance gaps identified during Q4 2025 review."
        ),
        status=PlanStatus.DRAFT,
        commodity=EUDRCommodity.COCOA,
        risk_level=RiskLevel.HIGH,
        total_gaps=3,
        total_actions=4,
        estimated_total_cost=Decimal("92000.00"),
        estimated_completion_days=120,
        target_completion=now + timedelta(days=120),
        created_at=now,
        provenance_hash="w" * 64,
    )


@pytest.fixture
def active_plan() -> ImprovementPlan:
    """Create an active improvement plan in ACTIVE status."""
    now = datetime.now(tz=timezone.utc)
    return ImprovementPlan(
        plan_id="plan-active-001",
        operator_id="operator-001",
        title="EUDR Compliance Improvement Plan - Active",
        description="Currently executing improvement plan.",
        status=PlanStatus.ACTIVE,
        commodity=EUDRCommodity.COFFEE,
        risk_level=RiskLevel.STANDARD,
        total_gaps=2,
        total_actions=5,
        estimated_total_cost=Decimal("50000.00"),
        estimated_completion_days=60,
        target_completion=now + timedelta(days=60),
        created_at=now - timedelta(days=30),
        approved_at=now - timedelta(days=28),
        provenance_hash="x" * 64,
    )


# ---------------------------------------------------------------------------
# Stakeholder Assignment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_assignment() -> StakeholderAssignment:
    """Create a sample stakeholder assignment."""
    return StakeholderAssignment(
        assignment_id="asgn-test-001",
        action_id="act-test-001",
        stakeholder_id="stk-test-001",
        stakeholder_name="Maria Sustainability",
        stakeholder_email="maria.sustainability@company.com",
        role=RACIRole.RESPONSIBLE,
        department="Sustainability & Compliance",
        notification_channel=NotificationChannel.EMAIL,
        assigned_at=datetime.now(tz=timezone.utc),
    )


@pytest.fixture
def multiple_stakeholder_assignments() -> List[StakeholderAssignment]:
    """Create multiple stakeholder assignments."""
    now = datetime.now(tz=timezone.utc)
    return [
        StakeholderAssignment(
            assignment_id="asgn-multi-001",
            action_id="act-multi-001",
            stakeholder_id="stk-001",
            stakeholder_name="Compliance Manager",
            stakeholder_email="compliance.mgr@company.com",
            role=RACIRole.ACCOUNTABLE,
            department="Compliance",
            notification_channel=NotificationChannel.EMAIL,
            assigned_at=now,
            notified_at=now,
            acknowledged_at=now + timedelta(hours=2),
        ),
        StakeholderAssignment(
            assignment_id="asgn-multi-002",
            action_id="act-multi-002",
            stakeholder_id="stk-002",
            stakeholder_name="Supply Chain Director",
            stakeholder_email="sc.director@company.com",
            role=RACIRole.RESPONSIBLE,
            department="Supply Chain",
            notification_channel=NotificationChannel.SLACK,
            assigned_at=now,
        ),
        StakeholderAssignment(
            assignment_id="asgn-multi-003",
            action_id="act-multi-003",
            stakeholder_id="stk-003",
            stakeholder_name="IT Solutions Architect",
            stakeholder_email="it.architect@company.com",
            role=RACIRole.CONSULTED,
            department="Information Technology",
            notification_channel=NotificationChannel.EMAIL,
            assigned_at=now,
        ),
    ]
