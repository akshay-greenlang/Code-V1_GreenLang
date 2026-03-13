# -*- coding: utf-8 -*-
"""
Unit tests for models.py - AGENT-EUDR-034

Tests all enumerations, model creation, defaults, Decimal fields,
constants, serialization, and optional fields.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.annual_review_scheduler.models import (
    AGENT_ID,
    AGENT_VERSION,
    SUPPORTED_COMMODITIES,
    REVIEW_PHASES_ORDER,
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

    def test_review_type_values(self):
        expected = {"annual", "semi_annual", "ad_hoc", "triggered"}
        actual = {t.value for t in ReviewType}
        assert actual == expected

    def test_review_phase_values(self):
        expected = {
            "preparation", "data_collection", "analysis",
            "review_meeting", "remediation", "sign_off",
        }
        actual = {p.value for p in ReviewPhase}
        assert actual == expected
        assert len(ReviewPhase) == 6

    def test_review_cycle_status_values(self):
        expected = {
            "draft", "scheduled", "in_progress", "paused",
            "completed", "cancelled", "overdue",
        }
        actual = {s.value for s in ReviewCycleStatus}
        assert actual == expected

    def test_deadline_status_values(self):
        expected = {"on_track", "at_risk", "overdue", "completed", "waived"}
        actual = {s.value for s in DeadlineStatus}
        assert actual == expected

    def test_deadline_alert_level_values(self):
        expected = {"info", "warning", "critical", "escalation"}
        actual = {l.value for l in DeadlineAlertLevel}
        assert actual == expected

    def test_checklist_item_status_values(self):
        expected = {"pending", "in_progress", "completed", "skipped", "blocked"}
        actual = {s.value for s in ChecklistItemStatus}
        assert actual == expected

    def test_entity_role_values(self):
        expected = {
            "lead", "reviewer", "analyst", "approver",
            "contributor", "observer", "external_auditor",
        }
        actual = {r.value for r in EntityRole}
        assert actual == expected

    def test_entity_status_values(self):
        expected = {"active", "invited", "declined", "inactive", "removed"}
        actual = {s.value for s in EntityStatus}
        assert actual == expected

    def test_calendar_entry_type_values(self):
        expected = {
            "phase_start", "phase_end", "deadline", "review_meeting",
            "milestone", "reminder", "sign_off",
        }
        actual = {t.value for t in CalendarEntryType}
        assert actual == expected

    def test_notification_channel_values(self):
        expected = {"email", "webhook", "slack", "sms", "in_app"}
        actual = {c.value for c in NotificationChannel}
        assert actual == expected

    def test_notification_priority_values(self):
        expected = {"low", "normal", "high", "urgent", "critical"}
        actual = {p.value for p in NotificationPriority}
        assert actual == expected

    def test_notification_status_values(self):
        expected = {"pending", "sent", "delivered", "failed", "cancelled"}
        actual = {s.value for s in NotificationStatus}
        assert actual == expected

    def test_year_comparison_status_values(self):
        expected = {"pending", "in_progress", "completed", "failed"}
        actual = {s.value for s in YearComparisonStatus}
        assert actual == expected

    def test_comparison_dimension_values(self):
        expected = {
            "compliance_rate", "risk_score", "supplier_count",
            "deforestation_rate", "audit_findings", "dds_approval_rate",
        }
        actual = {d.value for d in ComparisonDimension}
        assert actual == expected


class TestConstants:
    """Test module-level constants."""

    def test_agent_id(self):
        assert AGENT_ID == "GL-EUDR-ARS-034"

    def test_agent_version(self):
        assert AGENT_VERSION == "1.0.0"

    def test_supported_commodities(self):
        assert len(SUPPORTED_COMMODITIES) == 7
        assert "coffee" in SUPPORTED_COMMODITIES
        assert "wood" in SUPPORTED_COMMODITIES

    def test_review_phases_order(self):
        assert len(REVIEW_PHASES_ORDER) == 6
        assert REVIEW_PHASES_ORDER[0] == ReviewPhase.PREPARATION
        assert REVIEW_PHASES_ORDER[-1] == ReviewPhase.SIGN_OFF


class TestReviewCycleModel:
    """Test ReviewCycle model creation and defaults."""

    def test_create_valid_review_cycle(self, sample_review_cycle):
        assert sample_review_cycle.cycle_id == "cyc-test-001"
        assert sample_review_cycle.operator_id == "operator-001"
        assert sample_review_cycle.review_year == 2026
        assert sample_review_cycle.review_type == ReviewType.ANNUAL
        assert sample_review_cycle.status == ReviewCycleStatus.SCHEDULED

    def test_review_cycle_has_commodity_scope(self, sample_review_cycle):
        assert len(sample_review_cycle.commodity_scope) == 1
        assert sample_review_cycle.commodity_scope[0].commodity == EUDRCommodity.COFFEE

    def test_review_cycle_has_phase_configs(self, sample_review_cycle):
        assert len(sample_review_cycle.phase_configs) == 6

    def test_review_cycle_current_phase(self, sample_review_cycle):
        assert sample_review_cycle.current_phase == ReviewPhase.PREPARATION

    def test_review_cycle_provenance_hash_present(self, sample_review_cycle):
        assert len(sample_review_cycle.provenance_hash) == 64

    def test_review_cycle_defaults(self):
        now = datetime.now(tz=timezone.utc)
        cycle = ReviewCycle(
            cycle_id="cyc-minimal",
            operator_id="op-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            scheduled_start=now,
            scheduled_end=now,
        )
        assert cycle.status == ReviewCycleStatus.DRAFT
        assert cycle.current_phase == ReviewPhase.PREPARATION
        assert cycle.commodity_scope == []
        assert cycle.phase_configs == []
        assert cycle.actual_start is None
        assert cycle.actual_end is None
        assert cycle.created_by == ""
        assert cycle.provenance_hash == ""

    def test_review_cycle_created_at_is_datetime(self, sample_review_cycle):
        assert isinstance(sample_review_cycle.created_at, datetime)


class TestCommodityScopeModel:
    """Test CommodityScope model."""

    def test_create_commodity_scope(self):
        scope = CommodityScope(
            commodity=EUDRCommodity.COCOA,
            supplier_count=10,
            shipment_count=50,
        )
        assert scope.commodity == EUDRCommodity.COCOA
        assert scope.supplier_count == 10
        assert scope.shipment_count == 50

    def test_commodity_scope_defaults(self):
        scope = CommodityScope(commodity=EUDRCommodity.SOYA)
        assert scope.supplier_count == 0
        assert scope.shipment_count == 0


class TestReviewPhaseConfigModel:
    """Test ReviewPhaseConfig model."""

    def test_create_phase_config(self, sample_phase_configs):
        prep = sample_phase_configs[0]
        assert prep.phase == ReviewPhase.PREPARATION
        assert prep.duration_days == 14
        assert prep.required_checklist_items == 3
        assert prep.auto_advance is False

    def test_phase_config_defaults(self):
        config = ReviewPhaseConfig(
            phase=ReviewPhase.ANALYSIS,
        )
        assert config.duration_days == 30
        assert config.required_checklist_items == 0
        assert config.auto_advance is False


class TestDeadlineTrackModel:
    """Test DeadlineTrack model."""

    def test_create_valid_deadline(self, sample_deadline):
        assert sample_deadline.deadline_id == "dln-test-001"
        assert sample_deadline.cycle_id == "cyc-test-001"
        assert sample_deadline.phase == ReviewPhase.DATA_COLLECTION
        assert sample_deadline.status == DeadlineStatus.ON_TRACK

    def test_overdue_deadline(self, overdue_deadline):
        assert overdue_deadline.status == DeadlineStatus.OVERDUE
        assert overdue_deadline.due_date < datetime.now(tz=timezone.utc)

    def test_deadline_defaults(self):
        from datetime import datetime, timezone
        d = DeadlineTrack(
            deadline_id="d1",
            cycle_id="c1",
            phase=ReviewPhase.PREPARATION,
            description="Test",
            due_date=datetime.now(tz=timezone.utc),
        )
        assert d.status == DeadlineStatus.ON_TRACK
        assert d.assigned_entity_id is None
        assert d.warning_days_before == 7
        assert d.critical_days_before == 3
        assert d.completed_at is None
        assert d.provenance_hash == ""


class TestDeadlineAlertModel:
    """Test DeadlineAlert model."""

    def test_create_deadline_alert(self, sample_deadline_alert):
        assert sample_deadline_alert.alert_id == "alt-test-001"
        assert sample_deadline_alert.alert_level == DeadlineAlertLevel.WARNING
        assert sample_deadline_alert.days_remaining == 7
        assert sample_deadline_alert.acknowledged is False

    def test_deadline_alert_defaults(self):
        alert = DeadlineAlert(
            alert_id="a1",
            deadline_id="d1",
            cycle_id="c1",
            alert_level=DeadlineAlertLevel.INFO,
            message="Test alert",
        )
        assert alert.days_remaining == 0
        assert alert.acknowledged is False


class TestChecklistItemModel:
    """Test ChecklistItem model."""

    def test_create_checklist_item(self, sample_checklist_item):
        assert sample_checklist_item.item_id == "chk-test-001"
        assert sample_checklist_item.status == ChecklistItemStatus.PENDING
        assert sample_checklist_item.required is True
        assert sample_checklist_item.evidence_required is True

    def test_checklist_item_defaults(self):
        item = ChecklistItem(
            item_id="i1",
            cycle_id="c1",
            phase=ReviewPhase.PREPARATION,
            title="Test item",
        )
        assert item.description == ""
        assert item.status == ChecklistItemStatus.PENDING
        assert item.assigned_to is None
        assert item.priority == 0
        assert item.required is False
        assert item.evidence_required is False
        assert item.completed_at is None
        assert item.completed_by is None


class TestChecklistTemplateModel:
    """Test ChecklistTemplate model."""

    def test_create_checklist_template(self, sample_checklist_template):
        assert sample_checklist_template.template_id == "tpl-test-001"
        assert len(sample_checklist_template.items) == 3
        assert sample_checklist_template.phase == ReviewPhase.DATA_COLLECTION

    def test_checklist_template_defaults(self):
        tpl = ChecklistTemplate(
            template_id="tpl1",
            name="Test Template",
            phase=ReviewPhase.PREPARATION,
        )
        assert tpl.commodity is None
        assert tpl.items == []
        assert tpl.version == "1.0.0"
        assert tpl.regulatory_reference == ""


class TestEntityCoordinationModel:
    """Test EntityCoordination model."""

    def test_create_entity(self, sample_entity):
        assert sample_entity.entity_id == "entity-001"
        assert sample_entity.role == EntityRole.REVIEWER
        assert sample_entity.status == EntityStatus.ACTIVE
        assert len(sample_entity.assigned_phases) == 2

    def test_entity_defaults(self):
        entity = EntityCoordination(
            entity_id="e1",
            cycle_id="c1",
            name="Test Entity",
            role=EntityRole.CONTRIBUTOR,
            email="test@example.com",
        )
        assert entity.status == EntityStatus.ACTIVE
        assert entity.assigned_phases == []
        assert entity.dependencies == []


class TestEntityDependencyModel:
    """Test EntityDependency model."""

    def test_create_dependency(self, sample_entity_dependency):
        assert sample_entity_dependency.dependency_id == "dep-test-001"
        assert sample_entity_dependency.source_entity_id == "entity-002"
        assert sample_entity_dependency.target_entity_id == "entity-001"
        assert sample_entity_dependency.resolved is False

    def test_dependency_defaults(self):
        dep = EntityDependency(
            dependency_id="dep1",
            source_entity_id="src",
            target_entity_id="tgt",
            dependency_type="data_handoff",
            phase=ReviewPhase.DATA_COLLECTION,
        )
        assert dep.description == ""
        assert dep.resolved is False


class TestYearMetricSnapshotModel:
    """Test YearMetricSnapshot model."""

    def test_create_2025_snapshot(self, sample_year_metric_2025):
        assert sample_year_metric_2025.year == 2025
        assert sample_year_metric_2025.compliance_rate == Decimal("83.33")
        assert sample_year_metric_2025.total_suppliers == 12

    def test_create_2026_snapshot(self, sample_year_metric_2026):
        assert sample_year_metric_2026.year == 2026
        assert sample_year_metric_2026.compliance_rate == Decimal("93.33")
        assert sample_year_metric_2026.total_suppliers == 15

    def test_snapshot_defaults(self):
        snap = YearMetricSnapshot(
            snapshot_id="s1",
            operator_id="op1",
            year=2026,
            commodity=EUDRCommodity.WOOD,
        )
        assert snap.total_suppliers == 0
        assert snap.compliant_suppliers == 0
        assert snap.compliance_rate == Decimal("0")
        assert snap.average_risk_score == Decimal("0")
        assert snap.total_shipments == 0
        assert snap.deforestation_free_rate == Decimal("0")
        assert snap.dds_submitted == 0
        assert snap.dds_approved == 0
        assert snap.audit_findings == 0
        assert snap.remediation_actions == 0
        assert snap.provenance_hash == ""


class TestYearComparisonModel:
    """Test YearComparison model."""

    def test_create_year_comparison(self, sample_year_comparison):
        assert sample_year_comparison.comparison_id == "cmp-test-001"
        assert sample_year_comparison.base_year == 2025
        assert sample_year_comparison.compare_year == 2026
        assert sample_year_comparison.overall_trend == "improving"
        assert sample_year_comparison.status == YearComparisonStatus.COMPLETED

    def test_comparison_defaults(self):
        snap = YearMetricSnapshot(
            snapshot_id="s1", operator_id="op1", year=2025,
            commodity=EUDRCommodity.COFFEE,
        )
        cmp = YearComparison(
            comparison_id="c1",
            operator_id="op1",
            commodity=EUDRCommodity.COFFEE,
            base_year=2025,
            compare_year=2026,
            base_snapshot=snap,
            compare_snapshot=snap,
        )
        assert cmp.status == YearComparisonStatus.PENDING
        assert cmp.metrics == []
        assert cmp.overall_trend == ""
        assert cmp.provenance_hash == ""


class TestCalendarEntryModel:
    """Test CalendarEntry model."""

    def test_create_calendar_entry(self, sample_calendar_entry):
        assert sample_calendar_entry.entry_id == "cal-test-001"
        assert sample_calendar_entry.entry_type == CalendarEntryType.PHASE_START
        assert len(sample_calendar_entry.attendees) == 2

    def test_calendar_entry_defaults(self):
        now = datetime.now(tz=timezone.utc)
        entry = CalendarEntry(
            entry_id="cal1",
            cycle_id="c1",
            entry_type=CalendarEntryType.REMINDER,
            title="Test Reminder",
            start_time=now,
        )
        assert entry.description == ""
        assert entry.end_time is None
        assert entry.phase is None
        assert entry.attendees == []
        assert entry.location == ""
        assert entry.recurring is False


class TestNotificationRecordModel:
    """Test NotificationRecord model."""

    def test_create_notification(self, sample_notification):
        assert sample_notification.notification_id == "ntf-test-001"
        assert sample_notification.channel == NotificationChannel.EMAIL
        assert sample_notification.status == NotificationStatus.PENDING

    def test_notification_defaults(self):
        ntf = NotificationRecord(
            notification_id="n1",
            cycle_id="c1",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.LOW,
            recipient="test@example.com",
            subject="Test",
            body="Body",
        )
        assert ntf.status == NotificationStatus.PENDING
        assert ntf.template_id is None
        assert ntf.sent_at is None
        assert ntf.failure_reason is None
        assert ntf.retry_count == 0


class TestNotificationTemplateModel:
    """Test NotificationTemplate model."""

    def test_create_notification_template(self, sample_notification_template):
        assert sample_notification_template.template_id == "tpl-ntf-phase-start"
        assert sample_notification_template.trigger_event == "phase_start"
        assert sample_notification_template.days_before == 7

    def test_template_defaults(self):
        tpl = NotificationTemplate(
            template_id="t1",
            name="Test",
            channel=NotificationChannel.SLACK,
            subject_template="Subject",
            body_template="Body",
        )
        assert tpl.priority == NotificationPriority.NORMAL
        assert tpl.trigger_event == ""
        assert tpl.days_before == 0


class TestHealthStatusModel:
    """Test HealthStatus model."""

    def test_health_status_defaults(self):
        h = HealthStatus()
        assert h.agent_id == AGENT_ID
        assert h.status == "healthy"
        assert h.version == AGENT_VERSION
        assert h.database is False
        assert h.redis is False

    def test_health_status_custom(self):
        h = HealthStatus(
            status="degraded",
            database=True,
            redis=True,
        )
        assert h.status == "degraded"
        assert h.database is True
        assert h.redis is True
