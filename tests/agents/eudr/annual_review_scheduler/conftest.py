# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-034 Annual Review Scheduler tests.

Provides reusable test fixtures for config, models, review cycles,
deadlines, checklists, entities, calendar entries, notifications,
and provenance tracking across all test modules.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone, date
from decimal import Decimal
from typing import Dict, List, Optional

from greenlang.agents.eudr.annual_review_scheduler.config import (
    AnnualReviewSchedulerConfig,
    reset_config,
)
from greenlang.agents.eudr.annual_review_scheduler.models import (
    AGENT_ID,
    AGENT_VERSION,
    CalendarEntry,
    CalendarEntryType,
    ChecklistItem,
    ChecklistItemStatus,
    ChecklistTemplate,
    CommodityScope,
    ComparisonDimension,
    ComparisonMetric,
    ComparisonResult,
    DeadlineAlert,
    DeadlineAlertLevel,
    DeadlineStatus,
    DeadlineTrack,
    EntityCoordination,
    EntityDependency,
    EntityRole,
    EntityStatus,
    EUDRCommodity,
    HealthStatus,
    NotificationChannel,
    NotificationPriority,
    NotificationRecord,
    NotificationStatus,
    NotificationTemplate,
    ReviewCycle,
    ReviewCycleStatus,
    ReviewPhase,
    ReviewPhaseConfig,
    ReviewType,
    YearComparison,
    YearComparisonStatus,
    YearMetricSnapshot,
)
from greenlang.agents.eudr.annual_review_scheduler.provenance import (
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
def sample_config() -> AnnualReviewSchedulerConfig:
    """Create a default AnnualReviewSchedulerConfig instance."""
    return AnnualReviewSchedulerConfig()


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Review Cycle fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_phase_configs() -> List[ReviewPhaseConfig]:
    """Create sample review phase configurations."""
    return [
        ReviewPhaseConfig(
            phase=ReviewPhase.PREPARATION,
            duration_days=14,
            required_checklist_items=3,
            auto_advance=False,
        ),
        ReviewPhaseConfig(
            phase=ReviewPhase.DATA_COLLECTION,
            duration_days=30,
            required_checklist_items=5,
            auto_advance=False,
        ),
        ReviewPhaseConfig(
            phase=ReviewPhase.ANALYSIS,
            duration_days=21,
            required_checklist_items=4,
            auto_advance=False,
        ),
        ReviewPhaseConfig(
            phase=ReviewPhase.REVIEW_MEETING,
            duration_days=7,
            required_checklist_items=2,
            auto_advance=False,
        ),
        ReviewPhaseConfig(
            phase=ReviewPhase.REMEDIATION,
            duration_days=30,
            required_checklist_items=3,
            auto_advance=False,
        ),
        ReviewPhaseConfig(
            phase=ReviewPhase.SIGN_OFF,
            duration_days=7,
            required_checklist_items=2,
            auto_advance=True,
        ),
    ]


@pytest.fixture
def sample_review_cycle(sample_phase_configs) -> ReviewCycle:
    """Create a sample ReviewCycle in SCHEDULED status."""
    now = datetime.now(tz=timezone.utc)
    return ReviewCycle(
        cycle_id="cyc-test-001",
        operator_id="operator-001",
        review_year=2026,
        review_type=ReviewType.ANNUAL,
        commodity_scope=[
            CommodityScope(
                commodity=EUDRCommodity.COFFEE,
                supplier_count=15,
                shipment_count=120,
            ),
        ],
        status=ReviewCycleStatus.SCHEDULED,
        current_phase=ReviewPhase.PREPARATION,
        phase_configs=sample_phase_configs,
        scheduled_start=now + timedelta(days=7),
        scheduled_end=now + timedelta(days=120),
        created_by="scheduler-agent",
        provenance_hash="a" * 64,
    )


@pytest.fixture
def active_review_cycle(sample_phase_configs) -> ReviewCycle:
    """Create a ReviewCycle in IN_PROGRESS status."""
    now = datetime.now(tz=timezone.utc)
    return ReviewCycle(
        cycle_id="cyc-active-001",
        operator_id="operator-001",
        review_year=2026,
        review_type=ReviewType.ANNUAL,
        commodity_scope=[
            CommodityScope(
                commodity=EUDRCommodity.COFFEE,
                supplier_count=15,
                shipment_count=120,
            ),
            CommodityScope(
                commodity=EUDRCommodity.COCOA,
                supplier_count=8,
                shipment_count=45,
            ),
        ],
        status=ReviewCycleStatus.IN_PROGRESS,
        current_phase=ReviewPhase.DATA_COLLECTION,
        phase_configs=sample_phase_configs,
        scheduled_start=now - timedelta(days=14),
        scheduled_end=now + timedelta(days=106),
        actual_start=now - timedelta(days=14),
        created_by="scheduler-agent",
        provenance_hash="b" * 64,
    )


# ---------------------------------------------------------------------------
# Deadline fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_deadline() -> DeadlineTrack:
    """Create a sample deadline in ON_TRACK status."""
    now = datetime.now(tz=timezone.utc)
    return DeadlineTrack(
        deadline_id="dln-test-001",
        cycle_id="cyc-test-001",
        phase=ReviewPhase.DATA_COLLECTION,
        description="Complete supplier data collection",
        due_date=now + timedelta(days=30),
        status=DeadlineStatus.ON_TRACK,
        assigned_entity_id="entity-001",
        warning_days_before=7,
        critical_days_before=3,
        provenance_hash="c" * 64,
    )


@pytest.fixture
def overdue_deadline() -> DeadlineTrack:
    """Create an overdue deadline."""
    now = datetime.now(tz=timezone.utc)
    return DeadlineTrack(
        deadline_id="dln-overdue-001",
        cycle_id="cyc-test-001",
        phase=ReviewPhase.ANALYSIS,
        description="Complete risk analysis report",
        due_date=now - timedelta(days=5),
        status=DeadlineStatus.OVERDUE,
        assigned_entity_id="entity-002",
        warning_days_before=7,
        critical_days_before=3,
        provenance_hash="d" * 64,
    )


@pytest.fixture
def sample_deadline_alert() -> DeadlineAlert:
    """Create a sample deadline alert."""
    return DeadlineAlert(
        alert_id="alt-test-001",
        deadline_id="dln-test-001",
        cycle_id="cyc-test-001",
        alert_level=DeadlineAlertLevel.WARNING,
        message="Deadline approaching: 7 days remaining for data collection",
        days_remaining=7,
        acknowledged=False,
    )


# ---------------------------------------------------------------------------
# Checklist fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_checklist_item() -> ChecklistItem:
    """Create a sample checklist item in PENDING status."""
    return ChecklistItem(
        item_id="chk-test-001",
        cycle_id="cyc-test-001",
        phase=ReviewPhase.DATA_COLLECTION,
        title="Collect supplier emission data",
        description="Gather all Scope 1/2/3 emission data from tier-1 suppliers",
        status=ChecklistItemStatus.PENDING,
        assigned_to="analyst@greenlang.com",
        priority=1,
        required=True,
        evidence_required=True,
    )


@pytest.fixture
def sample_checklist_template() -> ChecklistTemplate:
    """Create a sample checklist template."""
    return ChecklistTemplate(
        template_id="tpl-test-001",
        name="Annual EUDR Review - Data Collection",
        phase=ReviewPhase.DATA_COLLECTION,
        commodity=EUDRCommodity.COFFEE,
        items=[
            ChecklistItem(
                item_id="chk-tpl-001",
                cycle_id="",
                phase=ReviewPhase.DATA_COLLECTION,
                title="Verify supplier geolocation data",
                description="Confirm GPS coordinates for all production plots",
                status=ChecklistItemStatus.PENDING,
                priority=1,
                required=True,
                evidence_required=True,
            ),
            ChecklistItem(
                item_id="chk-tpl-002",
                cycle_id="",
                phase=ReviewPhase.DATA_COLLECTION,
                title="Review deforestation monitoring results",
                description="Analyze satellite data for deforestation alerts",
                status=ChecklistItemStatus.PENDING,
                priority=2,
                required=True,
                evidence_required=True,
            ),
            ChecklistItem(
                item_id="chk-tpl-003",
                cycle_id="",
                phase=ReviewPhase.DATA_COLLECTION,
                title="Collect supplier certifications",
                description="Gather updated sustainability certifications",
                status=ChecklistItemStatus.PENDING,
                priority=3,
                required=False,
                evidence_required=False,
            ),
        ],
        version="1.0.0",
        regulatory_reference="EUDR Art. 4(2)",
    )


@pytest.fixture
def completed_checklist_items() -> List[ChecklistItem]:
    """Create a list of completed checklist items."""
    return [
        ChecklistItem(
            item_id=f"chk-done-{i:03d}",
            cycle_id="cyc-test-001",
            phase=ReviewPhase.DATA_COLLECTION,
            title=f"Task {i}",
            description=f"Completed task {i}",
            status=ChecklistItemStatus.COMPLETED,
            assigned_to="analyst@greenlang.com",
            priority=i,
            required=True,
            completed_at=datetime.now(tz=timezone.utc),
            completed_by="analyst@greenlang.com",
        )
        for i in range(1, 6)
    ]


# ---------------------------------------------------------------------------
# Entity Coordination fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_entity() -> EntityCoordination:
    """Create a sample entity coordination record."""
    return EntityCoordination(
        entity_id="entity-001",
        cycle_id="cyc-test-001",
        name="Sustainability Manager",
        role=EntityRole.REVIEWER,
        email="sustainability@company.com",
        status=EntityStatus.ACTIVE,
        assigned_phases=[ReviewPhase.DATA_COLLECTION, ReviewPhase.ANALYSIS],
        dependencies=[],
    )


@pytest.fixture
def sample_entity_dependency() -> EntityDependency:
    """Create a sample entity dependency."""
    return EntityDependency(
        dependency_id="dep-test-001",
        source_entity_id="entity-002",
        target_entity_id="entity-001",
        dependency_type="data_handoff",
        phase=ReviewPhase.DATA_COLLECTION,
        description="Analysis team requires completed data from collection team",
        resolved=False,
    )


@pytest.fixture
def multiple_entities() -> List[EntityCoordination]:
    """Create multiple entity coordination records."""
    return [
        EntityCoordination(
            entity_id="entity-lead-001",
            cycle_id="cyc-test-001",
            name="Review Lead",
            role=EntityRole.LEAD,
            email="lead@company.com",
            status=EntityStatus.ACTIVE,
            assigned_phases=[
                ReviewPhase.PREPARATION,
                ReviewPhase.REVIEW_MEETING,
                ReviewPhase.SIGN_OFF,
            ],
            dependencies=[],
        ),
        EntityCoordination(
            entity_id="entity-analyst-001",
            cycle_id="cyc-test-001",
            name="Data Analyst",
            role=EntityRole.ANALYST,
            email="analyst@company.com",
            status=EntityStatus.ACTIVE,
            assigned_phases=[
                ReviewPhase.DATA_COLLECTION,
                ReviewPhase.ANALYSIS,
            ],
            dependencies=[],
        ),
        EntityCoordination(
            entity_id="entity-approver-001",
            cycle_id="cyc-test-001",
            name="Compliance Officer",
            role=EntityRole.APPROVER,
            email="compliance@company.com",
            status=EntityStatus.ACTIVE,
            assigned_phases=[
                ReviewPhase.REVIEW_MEETING,
                ReviewPhase.SIGN_OFF,
            ],
            dependencies=[],
        ),
        EntityCoordination(
            entity_id="entity-external-001",
            cycle_id="cyc-test-001",
            name="Third-Party Auditor",
            role=EntityRole.EXTERNAL_AUDITOR,
            email="auditor@external.com",
            status=EntityStatus.INVITED,
            assigned_phases=[ReviewPhase.REVIEW_MEETING],
            dependencies=[],
        ),
    ]


# ---------------------------------------------------------------------------
# Year Comparison fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_year_metric_2025() -> YearMetricSnapshot:
    """Create a metric snapshot for year 2025."""
    return YearMetricSnapshot(
        snapshot_id="snap-2025-001",
        operator_id="operator-001",
        year=2025,
        commodity=EUDRCommodity.COFFEE,
        total_suppliers=12,
        compliant_suppliers=10,
        compliance_rate=Decimal("83.33"),
        average_risk_score=Decimal("35.20"),
        total_shipments=95,
        deforestation_free_rate=Decimal("97.50"),
        dds_submitted=90,
        dds_approved=85,
        audit_findings=3,
        remediation_actions=2,
        provenance_hash="e" * 64,
    )


@pytest.fixture
def sample_year_metric_2026() -> YearMetricSnapshot:
    """Create a metric snapshot for year 2026."""
    return YearMetricSnapshot(
        snapshot_id="snap-2026-001",
        operator_id="operator-001",
        year=2026,
        commodity=EUDRCommodity.COFFEE,
        total_suppliers=15,
        compliant_suppliers=14,
        compliance_rate=Decimal("93.33"),
        average_risk_score=Decimal("28.50"),
        total_shipments=120,
        deforestation_free_rate=Decimal("99.10"),
        dds_submitted=118,
        dds_approved=115,
        audit_findings=1,
        remediation_actions=1,
        provenance_hash="f" * 64,
    )


@pytest.fixture
def sample_year_comparison(
    sample_year_metric_2025,
    sample_year_metric_2026,
) -> YearComparison:
    """Create a sample year-over-year comparison."""
    return YearComparison(
        comparison_id="cmp-test-001",
        operator_id="operator-001",
        commodity=EUDRCommodity.COFFEE,
        base_year=2025,
        compare_year=2026,
        base_snapshot=sample_year_metric_2025,
        compare_snapshot=sample_year_metric_2026,
        status=YearComparisonStatus.COMPLETED,
        metrics=[],
        overall_trend="improving",
        provenance_hash="g" * 64,
    )


# ---------------------------------------------------------------------------
# Calendar fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_calendar_entry() -> CalendarEntry:
    """Create a sample calendar entry."""
    now = datetime.now(tz=timezone.utc)
    return CalendarEntry(
        entry_id="cal-test-001",
        cycle_id="cyc-test-001",
        entry_type=CalendarEntryType.PHASE_START,
        title="Data Collection Phase Start",
        description="Begin annual data collection for EUDR review",
        start_time=now + timedelta(days=14),
        end_time=now + timedelta(days=14, hours=1),
        phase=ReviewPhase.DATA_COLLECTION,
        attendees=["analyst@company.com", "lead@company.com"],
        location="Virtual",
        recurring=False,
    )


@pytest.fixture
def multiple_calendar_entries() -> List[CalendarEntry]:
    """Create multiple calendar entries for a full cycle."""
    now = datetime.now(tz=timezone.utc)
    return [
        CalendarEntry(
            entry_id="cal-prep-001",
            cycle_id="cyc-test-001",
            entry_type=CalendarEntryType.PHASE_START,
            title="Preparation Phase Start",
            start_time=now,
            end_time=now + timedelta(hours=1),
            phase=ReviewPhase.PREPARATION,
            attendees=["lead@company.com"],
        ),
        CalendarEntry(
            entry_id="cal-dc-001",
            cycle_id="cyc-test-001",
            entry_type=CalendarEntryType.PHASE_START,
            title="Data Collection Start",
            start_time=now + timedelta(days=14),
            end_time=now + timedelta(days=14, hours=1),
            phase=ReviewPhase.DATA_COLLECTION,
            attendees=["analyst@company.com"],
        ),
        CalendarEntry(
            entry_id="cal-deadline-001",
            cycle_id="cyc-test-001",
            entry_type=CalendarEntryType.DEADLINE,
            title="Data Collection Deadline",
            start_time=now + timedelta(days=44),
            end_time=now + timedelta(days=44, hours=1),
            phase=ReviewPhase.DATA_COLLECTION,
            attendees=["analyst@company.com", "lead@company.com"],
        ),
        CalendarEntry(
            entry_id="cal-meeting-001",
            cycle_id="cyc-test-001",
            entry_type=CalendarEntryType.REVIEW_MEETING,
            title="Annual Review Meeting",
            start_time=now + timedelta(days=65),
            end_time=now + timedelta(days=65, hours=2),
            phase=ReviewPhase.REVIEW_MEETING,
            attendees=[
                "lead@company.com",
                "analyst@company.com",
                "compliance@company.com",
            ],
            location="Conference Room A",
        ),
        CalendarEntry(
            entry_id="cal-signoff-001",
            cycle_id="cyc-test-001",
            entry_type=CalendarEntryType.SIGN_OFF,
            title="Final Sign-Off Deadline",
            start_time=now + timedelta(days=109),
            end_time=now + timedelta(days=109, hours=1),
            phase=ReviewPhase.SIGN_OFF,
            attendees=["compliance@company.com"],
        ),
    ]


# ---------------------------------------------------------------------------
# Notification fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_notification() -> NotificationRecord:
    """Create a sample notification record."""
    return NotificationRecord(
        notification_id="ntf-test-001",
        cycle_id="cyc-test-001",
        channel=NotificationChannel.EMAIL,
        priority=NotificationPriority.NORMAL,
        recipient="analyst@company.com",
        subject="EUDR Annual Review - Data Collection Phase Starting",
        body="The data collection phase for EUDR annual review cycle cyc-test-001 begins in 7 days.",
        status=NotificationStatus.PENDING,
        template_id="tpl-ntf-phase-start",
    )


@pytest.fixture
def sample_notification_template() -> NotificationTemplate:
    """Create a sample notification template."""
    return NotificationTemplate(
        template_id="tpl-ntf-phase-start",
        name="Phase Start Notification",
        channel=NotificationChannel.EMAIL,
        subject_template="EUDR Annual Review - {phase_name} Phase Starting",
        body_template=(
            "Dear {recipient_name},\n\n"
            "The {phase_name} phase for EUDR annual review cycle {cycle_id} "
            "begins on {start_date}.\n\n"
            "Please ensure all prerequisites are completed.\n\n"
            "Best regards,\nGreenLang Annual Review Scheduler"
        ),
        priority=NotificationPriority.NORMAL,
        trigger_event="phase_start",
        days_before=7,
    )


@pytest.fixture
def multiple_notifications() -> List[NotificationRecord]:
    """Create multiple notifications of different types."""
    return [
        NotificationRecord(
            notification_id="ntf-email-001",
            cycle_id="cyc-test-001",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="analyst@company.com",
            subject="Phase Starting",
            body="Data collection phase starting.",
            status=NotificationStatus.SENT,
            sent_at=datetime.now(tz=timezone.utc) - timedelta(hours=2),
        ),
        NotificationRecord(
            notification_id="ntf-webhook-001",
            cycle_id="cyc-test-001",
            channel=NotificationChannel.WEBHOOK,
            priority=NotificationPriority.HIGH,
            recipient="https://hooks.company.com/eudr",
            subject="Deadline Warning",
            body='{"event":"deadline_warning","cycle_id":"cyc-test-001"}',
            status=NotificationStatus.SENT,
            sent_at=datetime.now(tz=timezone.utc) - timedelta(hours=1),
        ),
        NotificationRecord(
            notification_id="ntf-slack-001",
            cycle_id="cyc-test-001",
            channel=NotificationChannel.SLACK,
            priority=NotificationPriority.URGENT,
            recipient="#eudr-compliance",
            subject="Deadline Overdue",
            body="URGENT: Data collection deadline is overdue by 2 days.",
            status=NotificationStatus.PENDING,
        ),
        NotificationRecord(
            notification_id="ntf-sms-001",
            cycle_id="cyc-test-001",
            channel=NotificationChannel.SMS,
            priority=NotificationPriority.CRITICAL,
            recipient="+1234567890",
            subject="Critical Deadline",
            body="CRITICAL: Sign-off deadline missed. Escalation required.",
            status=NotificationStatus.FAILED,
            failure_reason="SMS gateway unavailable",
            retry_count=3,
        ),
    ]
