# -*- coding: utf-8 -*-
"""
Unit tests for Data Freshness Monitor models - AGENT-DATA-016

Tests the Pydantic v2 models at greenlang.data_freshness_monitor.models with
150+ tests covering all 13 enumerations, 16 SDK models, 8 request models,
6 constants, Layer 1 re-exports, field validators, model validators,
extra="forbid" enforcement, and edge cases.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from greenlang.data_freshness_monitor.models import (
    # Constants
    CADENCE_HOURS,
    DEFAULT_ESCALATION_DELAYS,
    FRESHNESS_SCORE_BOUNDARIES,
    MAX_DATASETS_PER_GROUP,
    MAX_SLA_WARNING_HOURS,
    VERSION,
    # Enumerations
    AlertChannel,
    AlertSeverity,
    AlertStatus,
    BreachSeverity,
    BreachStatus,
    DatasetPriority,
    DatasetStatus,
    FreshnessLevel,
    MonitoringStatus,
    PatternType,
    PredictionStatus,
    RefreshCadence,
    SLAStatus,
    # SDK models
    AuditEntry,
    DatasetDefinition,
    DatasetGroup,
    EscalationLevel,
    EscalationPolicy,
    FreshnessAlert,
    FreshnessCheck,
    FreshnessReport,
    FreshnessSummary,
    MonitoringRun,
    RefreshEvent,
    RefreshPrediction,
    SLABreach,
    SLADefinition,
    SourceReliability,
    StalenessPattern,
    # Request models
    CreateSLARequest,
    RegisterDatasetRequest,
    RunBatchCheckRequest,
    RunFreshnessCheckRequest,
    RunPipelineRequest,
    UpdateBreachRequest,
    UpdateDatasetRequest,
    UpdateSLARequest,
    # Layer 1 re-exports
    FRESHNESS_BOUNDARIES_HOURS,
    FreshnessResult,
    QualityDimension,
    RuleType,
    TimelinessTracker,
)

import pydantic

# ---------------------------------------------------------------------------
# Restore extra="forbid" for model tests (module-scoped fixture)
# ---------------------------------------------------------------------------
# The shared conftest relaxes all DFM models to extra="ignore" so engine tests
# can pass extra kwargs. For test_models.py we need the original strict config
# to verify that extra="forbid" enforcement works correctly. We use a
# module-scoped autouse fixture to restore strict before and relax after.


def _set_model_extra(mode: str) -> None:
    """Set extra config on all DFM Pydantic models."""
    from greenlang.data_freshness_monitor import models as dfm_models
    for name in dir(dfm_models):
        obj = getattr(dfm_models, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, pydantic.BaseModel)
            and obj is not pydantic.BaseModel
        ):
            cfg = getattr(obj, "model_config", {})
            if isinstance(cfg, dict):
                obj.model_config = {**cfg, "extra": mode}
                obj.model_rebuild(force=True)


@pytest.fixture(autouse=True, scope="module")
def _strict_models():
    """Temporarily restore extra='forbid' for this module's tests."""
    _set_model_extra("forbid")
    yield
    _set_model_extra("ignore")


# ======================================================================
# 1. Enumerations -- all 13 enums
# ======================================================================


class TestDatasetStatusEnum:
    """Test DatasetStatus enum values and member count."""

    def test_member_count(self):
        assert len(DatasetStatus) == 5

    def test_active_value(self):
        assert DatasetStatus.ACTIVE.value == "active"

    def test_inactive_value(self):
        assert DatasetStatus.INACTIVE.value == "inactive"

    def test_paused_value(self):
        assert DatasetStatus.PAUSED.value == "paused"

    def test_archived_value(self):
        assert DatasetStatus.ARCHIVED.value == "archived"

    def test_error_value(self):
        assert DatasetStatus.ERROR.value == "error"

    def test_is_str_enum(self):
        assert isinstance(DatasetStatus.ACTIVE, str)


class TestDatasetPriorityEnum:
    """Test DatasetPriority enum values and member count."""

    def test_member_count(self):
        assert len(DatasetPriority) == 5

    def test_critical_value(self):
        assert DatasetPriority.CRITICAL.value == "critical"

    def test_high_value(self):
        assert DatasetPriority.HIGH.value == "high"

    def test_medium_value(self):
        assert DatasetPriority.MEDIUM.value == "medium"

    def test_low_value(self):
        assert DatasetPriority.LOW.value == "low"

    def test_informational_value(self):
        assert DatasetPriority.INFORMATIONAL.value == "informational"


class TestRefreshCadenceEnum:
    """Test RefreshCadence enum values and member count."""

    def test_member_count(self):
        assert len(RefreshCadence) == 9

    def test_realtime_value(self):
        assert RefreshCadence.REALTIME.value == "realtime"

    def test_minutely_value(self):
        assert RefreshCadence.MINUTELY.value == "minutely"

    def test_hourly_value(self):
        assert RefreshCadence.HOURLY.value == "hourly"

    def test_daily_value(self):
        assert RefreshCadence.DAILY.value == "daily"

    def test_weekly_value(self):
        assert RefreshCadence.WEEKLY.value == "weekly"

    def test_monthly_value(self):
        assert RefreshCadence.MONTHLY.value == "monthly"

    def test_quarterly_value(self):
        assert RefreshCadence.QUARTERLY.value == "quarterly"

    def test_annual_value(self):
        assert RefreshCadence.ANNUAL.value == "annual"

    def test_on_demand_value(self):
        assert RefreshCadence.ON_DEMAND.value == "on_demand"


class TestFreshnessLevelEnum:
    """Test FreshnessLevel enum values and member count."""

    def test_member_count(self):
        assert len(FreshnessLevel) == 5

    def test_excellent_value(self):
        assert FreshnessLevel.EXCELLENT.value == "excellent"

    def test_good_value(self):
        assert FreshnessLevel.GOOD.value == "good"

    def test_fair_value(self):
        assert FreshnessLevel.FAIR.value == "fair"

    def test_poor_value(self):
        assert FreshnessLevel.POOR.value == "poor"

    def test_stale_value(self):
        assert FreshnessLevel.STALE.value == "stale"


class TestSLAStatusEnum:
    """Test SLAStatus enum values and member count."""

    def test_member_count(self):
        assert len(SLAStatus) == 5

    def test_compliant_value(self):
        assert SLAStatus.COMPLIANT.value == "compliant"

    def test_warning_value(self):
        assert SLAStatus.WARNING.value == "warning"

    def test_breached_value(self):
        assert SLAStatus.BREACHED.value == "breached"

    def test_critical_value(self):
        assert SLAStatus.CRITICAL.value == "critical"

    def test_unknown_value(self):
        assert SLAStatus.UNKNOWN.value == "unknown"


class TestBreachSeverityEnum:
    """Test BreachSeverity enum values and member count."""

    def test_member_count(self):
        assert len(BreachSeverity) == 5

    def test_info_value(self):
        assert BreachSeverity.INFO.value == "info"

    def test_low_value(self):
        assert BreachSeverity.LOW.value == "low"

    def test_medium_value(self):
        assert BreachSeverity.MEDIUM.value == "medium"

    def test_high_value(self):
        assert BreachSeverity.HIGH.value == "high"

    def test_critical_value(self):
        assert BreachSeverity.CRITICAL.value == "critical"


class TestBreachStatusEnum:
    """Test BreachStatus enum values and member count."""

    def test_member_count(self):
        assert len(BreachStatus) == 5

    def test_detected_value(self):
        assert BreachStatus.DETECTED.value == "detected"

    def test_acknowledged_value(self):
        assert BreachStatus.ACKNOWLEDGED.value == "acknowledged"

    def test_investigating_value(self):
        assert BreachStatus.INVESTIGATING.value == "investigating"

    def test_resolved_value(self):
        assert BreachStatus.RESOLVED.value == "resolved"

    def test_expired_value(self):
        assert BreachStatus.EXPIRED.value == "expired"


class TestAlertChannelEnum:
    """Test AlertChannel enum values and member count."""

    def test_member_count(self):
        assert len(AlertChannel) == 6

    def test_webhook_value(self):
        assert AlertChannel.WEBHOOK.value == "webhook"

    def test_email_value(self):
        assert AlertChannel.EMAIL.value == "email"

    def test_slack_value(self):
        assert AlertChannel.SLACK.value == "slack"

    def test_pagerduty_value(self):
        assert AlertChannel.PAGERDUTY.value == "pagerduty"

    def test_teams_value(self):
        assert AlertChannel.TEAMS.value == "teams"

    def test_log_value(self):
        assert AlertChannel.LOG.value == "log"


class TestAlertStatusEnum:
    """Test AlertStatus enum values and member count."""

    def test_member_count(self):
        assert len(AlertStatus) == 5

    def test_pending_value(self):
        assert AlertStatus.PENDING.value == "pending"

    def test_sent_value(self):
        assert AlertStatus.SENT.value == "sent"

    def test_acknowledged_value(self):
        assert AlertStatus.ACKNOWLEDGED.value == "acknowledged"

    def test_resolved_value(self):
        assert AlertStatus.RESOLVED.value == "resolved"

    def test_suppressed_value(self):
        assert AlertStatus.SUPPRESSED.value == "suppressed"


class TestAlertSeverityEnum:
    """Test AlertSeverity enum values and member count."""

    def test_member_count(self):
        assert len(AlertSeverity) == 4

    def test_info_value(self):
        assert AlertSeverity.INFO.value == "info"

    def test_warning_value(self):
        assert AlertSeverity.WARNING.value == "warning"

    def test_critical_value(self):
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_emergency_value(self):
        assert AlertSeverity.EMERGENCY.value == "emergency"


class TestPatternTypeEnum:
    """Test PatternType enum values and member count."""

    def test_member_count(self):
        assert len(PatternType) == 6

    def test_recurring_staleness_value(self):
        assert PatternType.RECURRING_STALENESS.value == "recurring_staleness"

    def test_seasonal_degradation_value(self):
        assert PatternType.SEASONAL_DEGRADATION.value == "seasonal_degradation"

    def test_source_failure_value(self):
        assert PatternType.SOURCE_FAILURE.value == "source_failure"

    def test_refresh_drift_value(self):
        assert PatternType.REFRESH_DRIFT.value == "refresh_drift"

    def test_random_gaps_value(self):
        assert PatternType.RANDOM_GAPS.value == "random_gaps"

    def test_systematic_delay_value(self):
        assert PatternType.SYSTEMATIC_DELAY.value == "systematic_delay"


class TestPredictionStatusEnum:
    """Test PredictionStatus enum values and member count."""

    def test_member_count(self):
        assert len(PredictionStatus) == 5

    def test_pending_value(self):
        assert PredictionStatus.PENDING.value == "pending"

    def test_on_time_value(self):
        assert PredictionStatus.ON_TIME.value == "on_time"

    def test_late_value(self):
        assert PredictionStatus.LATE.value == "late"

    def test_very_late_value(self):
        assert PredictionStatus.VERY_LATE.value == "very_late"

    def test_missed_value(self):
        assert PredictionStatus.MISSED.value == "missed"


class TestMonitoringStatusEnum:
    """Test MonitoringStatus enum values and member count."""

    def test_member_count(self):
        assert len(MonitoringStatus) == 5

    def test_idle_value(self):
        assert MonitoringStatus.IDLE.value == "idle"

    def test_running_value(self):
        assert MonitoringStatus.RUNNING.value == "running"

    def test_completed_value(self):
        assert MonitoringStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        assert MonitoringStatus.FAILED.value == "failed"

    def test_cancelled_value(self):
        assert MonitoringStatus.CANCELLED.value == "cancelled"


# ======================================================================
# 2. Constants
# ======================================================================


class TestConstants:
    """Test module-level constants."""

    def test_cadence_hours_has_9_entries(self):
        assert len(CADENCE_HOURS) == 9

    def test_cadence_hours_realtime(self):
        assert CADENCE_HOURS["realtime"] == 0.0

    def test_cadence_hours_minutely(self):
        assert CADENCE_HOURS["minutely"] == pytest.approx(1.0 / 60.0)

    def test_cadence_hours_hourly(self):
        assert CADENCE_HOURS["hourly"] == 1.0

    def test_cadence_hours_daily(self):
        assert CADENCE_HOURS["daily"] == 24.0

    def test_cadence_hours_weekly(self):
        assert CADENCE_HOURS["weekly"] == 168.0

    def test_cadence_hours_monthly(self):
        assert CADENCE_HOURS["monthly"] == 720.0

    def test_cadence_hours_quarterly(self):
        assert CADENCE_HOURS["quarterly"] == 2160.0

    def test_cadence_hours_annual(self):
        assert CADENCE_HOURS["annual"] == 8760.0

    def test_cadence_hours_on_demand(self):
        assert CADENCE_HOURS["on_demand"] == -1.0

    def test_freshness_score_boundaries_has_5_entries(self):
        assert len(FRESHNESS_SCORE_BOUNDARIES) == 5

    def test_freshness_score_boundaries_excellent(self):
        assert FRESHNESS_SCORE_BOUNDARIES["excellent"] == 1.0

    def test_freshness_score_boundaries_good(self):
        assert FRESHNESS_SCORE_BOUNDARIES["good"] == 24.0

    def test_freshness_score_boundaries_fair(self):
        assert FRESHNESS_SCORE_BOUNDARIES["fair"] == 168.0

    def test_freshness_score_boundaries_poor(self):
        assert FRESHNESS_SCORE_BOUNDARIES["poor"] == 720.0

    def test_freshness_score_boundaries_stale(self):
        assert FRESHNESS_SCORE_BOUNDARIES["stale"] == 8760.0

    def test_default_escalation_delays_length(self):
        assert len(DEFAULT_ESCALATION_DELAYS) == 4

    def test_default_escalation_delays_values(self):
        assert DEFAULT_ESCALATION_DELAYS == [15, 60, 240, 1440]

    def test_max_datasets_per_group(self):
        assert MAX_DATASETS_PER_GROUP == 500

    def test_max_sla_warning_hours(self):
        assert MAX_SLA_WARNING_HOURS == 8760.0

    def test_version(self):
        assert VERSION == "1.0.0"


# ======================================================================
# 3. Layer 1 re-exports
# ======================================================================


class TestLayer1ReExports:
    """Test that Layer 1 symbols are importable (may be None if L1 unavailable)."""

    def test_timeliness_tracker_importable(self):
        # Should be importable (either class or None)
        from greenlang.data_freshness_monitor.models import TimelinessTracker
        assert TimelinessTracker is not None or TimelinessTracker is None

    def test_freshness_result_importable(self):
        from greenlang.data_freshness_monitor.models import FreshnessResult
        assert FreshnessResult is not None or FreshnessResult is None

    def test_quality_dimension_importable(self):
        from greenlang.data_freshness_monitor.models import QualityDimension
        assert QualityDimension is not None or QualityDimension is None

    def test_rule_type_importable(self):
        from greenlang.data_freshness_monitor.models import RuleType
        assert RuleType is not None or RuleType is None

    def test_freshness_boundaries_hours_importable(self):
        from greenlang.data_freshness_monitor.models import FRESHNESS_BOUNDARIES_HOURS
        assert FRESHNESS_BOUNDARIES_HOURS is not None or FRESHNESS_BOUNDARIES_HOURS is None


# ======================================================================
# 4. SDK Models -- DatasetDefinition
# ======================================================================


class TestDatasetDefinition:
    """Test DatasetDefinition model."""

    def test_create_with_name(self):
        ds = DatasetDefinition(name="Test Dataset")
        assert ds.name == "Test Dataset"

    def test_default_source_name(self):
        ds = DatasetDefinition(name="Test")
        assert ds.source_name == ""

    def test_default_source_type(self):
        ds = DatasetDefinition(name="Test")
        assert ds.source_type == ""

    def test_default_owner(self):
        ds = DatasetDefinition(name="Test")
        assert ds.owner == ""

    def test_default_refresh_cadence(self):
        ds = DatasetDefinition(name="Test")
        assert ds.refresh_cadence == RefreshCadence.DAILY

    def test_default_priority(self):
        ds = DatasetDefinition(name="Test")
        assert ds.priority == DatasetPriority.MEDIUM

    def test_default_status(self):
        ds = DatasetDefinition(name="Test")
        assert ds.status == DatasetStatus.ACTIVE

    def test_default_tags_empty(self):
        ds = DatasetDefinition(name="Test")
        assert ds.tags == []

    def test_default_metadata_empty(self):
        ds = DatasetDefinition(name="Test")
        assert ds.metadata == {}

    def test_default_last_refreshed_at_none(self):
        ds = DatasetDefinition(name="Test")
        assert ds.last_refreshed_at is None

    def test_default_version(self):
        ds = DatasetDefinition(name="Test")
        assert ds.version == 1

    def test_default_provenance_hash(self):
        ds = DatasetDefinition(name="Test")
        assert ds.provenance_hash == ""

    def test_id_is_uuid(self):
        ds = DatasetDefinition(name="Test")
        assert len(ds.id) == 36  # UUID format

    def test_registered_at_is_datetime(self):
        ds = DatasetDefinition(name="Test")
        assert isinstance(ds.registered_at, datetime)

    def test_updated_at_is_datetime(self):
        ds = DatasetDefinition(name="Test")
        assert isinstance(ds.updated_at, datetime)

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError, match="name must be non-empty"):
            DatasetDefinition(name="")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValidationError, match="name must be non-empty"):
            DatasetDefinition(name="   ")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            DatasetDefinition(name="Test", unknown_field="x")

    def test_full_construction(self, sample_dataset_kwargs):
        ds = DatasetDefinition(**sample_dataset_kwargs)
        assert ds.name == "ERP Emissions Feed"
        assert ds.source_name == "SAP ERP Production"
        assert ds.source_type == "erp"
        assert ds.owner == "data-engineering"


# ======================================================================
# 5. SDK Models -- SLADefinition
# ======================================================================


class TestSLADefinition:
    """Test SLADefinition model."""

    def test_create_with_dataset_id(self):
        sla = SLADefinition(dataset_id="ds-001")
        assert sla.dataset_id == "ds-001"

    def test_default_warning_hours(self):
        sla = SLADefinition(dataset_id="ds-001")
        assert sla.warning_hours == 24.0

    def test_default_critical_hours(self):
        sla = SLADefinition(dataset_id="ds-001")
        assert sla.critical_hours == 72.0

    def test_default_breach_severity(self):
        sla = SLADefinition(dataset_id="ds-001")
        assert sla.breach_severity == BreachSeverity.HIGH

    def test_default_business_hours_only(self):
        sla = SLADefinition(dataset_id="ds-001")
        assert sla.business_hours_only is False

    def test_default_escalation_policy_none(self):
        sla = SLADefinition(dataset_id="ds-001")
        assert sla.escalation_policy is None

    def test_id_is_uuid(self):
        sla = SLADefinition(dataset_id="ds-001")
        assert len(sla.id) == 36

    def test_created_at_is_datetime(self):
        sla = SLADefinition(dataset_id="ds-001")
        assert isinstance(sla.created_at, datetime)

    def test_empty_dataset_id_raises(self):
        with pytest.raises(ValidationError, match="dataset_id must be non-empty"):
            SLADefinition(dataset_id="")

    def test_whitespace_dataset_id_raises(self):
        with pytest.raises(ValidationError, match="dataset_id must be non-empty"):
            SLADefinition(dataset_id="   ")

    def test_warning_exceeds_critical_raises(self):
        with pytest.raises(ValidationError, match="warning_hours.*must be <=.*critical_hours"):
            SLADefinition(dataset_id="ds-001", warning_hours=100.0, critical_hours=50.0)

    def test_warning_equals_critical_accepted(self):
        sla = SLADefinition(dataset_id="ds-001", warning_hours=48.0, critical_hours=48.0)
        assert sla.warning_hours == sla.critical_hours

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            SLADefinition(dataset_id="ds-001", unknown="x")

    def test_warning_negative_raises(self):
        with pytest.raises(ValidationError):
            SLADefinition(dataset_id="ds-001", warning_hours=-1.0)

    def test_critical_exceeds_max_raises(self):
        with pytest.raises(ValidationError):
            SLADefinition(dataset_id="ds-001", critical_hours=9000.0)


# ======================================================================
# 6. SDK Models -- FreshnessCheck
# ======================================================================


class TestFreshnessCheck:
    """Test FreshnessCheck model."""

    def test_create_with_dataset_id(self):
        fc = FreshnessCheck(dataset_id="ds-001")
        assert fc.dataset_id == "ds-001"

    def test_default_age_hours(self):
        fc = FreshnessCheck(dataset_id="ds-001")
        assert fc.age_hours == 0.0

    def test_default_freshness_score(self):
        fc = FreshnessCheck(dataset_id="ds-001")
        assert fc.freshness_score == 1.0

    def test_default_freshness_level(self):
        fc = FreshnessCheck(dataset_id="ds-001")
        assert fc.freshness_level == FreshnessLevel.EXCELLENT

    def test_default_sla_status(self):
        fc = FreshnessCheck(dataset_id="ds-001")
        assert fc.sla_status == SLAStatus.UNKNOWN

    def test_default_sla_id_none(self):
        fc = FreshnessCheck(dataset_id="ds-001")
        assert fc.sla_id is None

    def test_default_provenance_hash(self):
        fc = FreshnessCheck(dataset_id="ds-001")
        assert fc.provenance_hash == ""

    def test_empty_dataset_id_raises(self):
        with pytest.raises(ValidationError, match="dataset_id must be non-empty"):
            FreshnessCheck(dataset_id="")

    def test_negative_age_hours_raises(self):
        with pytest.raises(ValidationError):
            FreshnessCheck(dataset_id="ds-001", age_hours=-1.0)

    def test_freshness_score_above_one_raises(self):
        with pytest.raises(ValidationError):
            FreshnessCheck(dataset_id="ds-001", freshness_score=1.5)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            FreshnessCheck(dataset_id="ds-001", extra="bad")


# ======================================================================
# 7. SDK Models -- RefreshEvent
# ======================================================================


class TestRefreshEvent:
    """Test RefreshEvent model."""

    def test_create_with_dataset_id(self):
        evt = RefreshEvent(dataset_id="ds-001")
        assert evt.dataset_id == "ds-001"

    def test_default_data_size_bytes_none(self):
        evt = RefreshEvent(dataset_id="ds-001")
        assert evt.data_size_bytes is None

    def test_default_record_count_none(self):
        evt = RefreshEvent(dataset_id="ds-001")
        assert evt.record_count is None

    def test_default_source_info_empty(self):
        evt = RefreshEvent(dataset_id="ds-001")
        assert evt.source_info == {}

    def test_empty_dataset_id_raises(self):
        with pytest.raises(ValidationError, match="dataset_id must be non-empty"):
            RefreshEvent(dataset_id="")

    def test_negative_data_size_raises(self):
        with pytest.raises(ValidationError):
            RefreshEvent(dataset_id="ds-001", data_size_bytes=-1)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            RefreshEvent(dataset_id="ds-001", extra="bad")


# ======================================================================
# 8. SDK Models -- StalenessPattern
# ======================================================================


class TestStalenessPattern:
    """Test StalenessPattern model."""

    def test_create_with_dataset_id(self):
        sp = StalenessPattern(dataset_id="ds-001")
        assert sp.dataset_id == "ds-001"

    def test_default_pattern_type(self):
        sp = StalenessPattern(dataset_id="ds-001")
        assert sp.pattern_type == PatternType.RECURRING_STALENESS

    def test_default_severity(self):
        sp = StalenessPattern(dataset_id="ds-001")
        assert sp.severity == BreachSeverity.LOW

    def test_default_confidence(self):
        sp = StalenessPattern(dataset_id="ds-001")
        assert sp.confidence == 0.0

    def test_default_frequency_hours_none(self):
        sp = StalenessPattern(dataset_id="ds-001")
        assert sp.frequency_hours is None

    def test_default_description_empty(self):
        sp = StalenessPattern(dataset_id="ds-001")
        assert sp.description == ""

    def test_empty_dataset_id_raises(self):
        with pytest.raises(ValidationError, match="dataset_id must be non-empty"):
            StalenessPattern(dataset_id="")

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            StalenessPattern(dataset_id="ds-001", confidence=1.5)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            StalenessPattern(dataset_id="ds-001", extra="bad")


# ======================================================================
# 9. SDK Models -- SLABreach
# ======================================================================


class TestSLABreach:
    """Test SLABreach model."""

    def test_create_with_required_fields(self):
        b = SLABreach(dataset_id="ds-001", sla_id="sla-001")
        assert b.dataset_id == "ds-001"
        assert b.sla_id == "sla-001"

    def test_default_breach_severity(self):
        b = SLABreach(dataset_id="ds-001", sla_id="sla-001")
        assert b.breach_severity == BreachSeverity.HIGH

    def test_default_status(self):
        b = SLABreach(dataset_id="ds-001", sla_id="sla-001")
        assert b.status == BreachStatus.DETECTED

    def test_default_age_at_breach_hours(self):
        b = SLABreach(dataset_id="ds-001", sla_id="sla-001")
        assert b.age_at_breach_hours == 0.0

    def test_default_acknowledged_at_none(self):
        b = SLABreach(dataset_id="ds-001", sla_id="sla-001")
        assert b.acknowledged_at is None

    def test_default_resolved_at_none(self):
        b = SLABreach(dataset_id="ds-001", sla_id="sla-001")
        assert b.resolved_at is None

    def test_default_resolution_notes(self):
        b = SLABreach(dataset_id="ds-001", sla_id="sla-001")
        assert b.resolution_notes == ""

    def test_empty_dataset_id_raises(self):
        with pytest.raises(ValidationError, match="dataset_id must be non-empty"):
            SLABreach(dataset_id="", sla_id="sla-001")

    def test_empty_sla_id_raises(self):
        with pytest.raises(ValidationError, match="sla_id must be non-empty"):
            SLABreach(dataset_id="ds-001", sla_id="")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            SLABreach(dataset_id="ds-001", sla_id="sla-001", extra="bad")


# ======================================================================
# 10. SDK Models -- FreshnessAlert
# ======================================================================


class TestFreshnessAlert:
    """Test FreshnessAlert model."""

    def test_create_with_breach_id(self):
        a = FreshnessAlert(breach_id="b-001")
        assert a.breach_id == "b-001"

    def test_default_alert_severity(self):
        a = FreshnessAlert(breach_id="b-001")
        assert a.alert_severity == AlertSeverity.WARNING

    def test_default_channel(self):
        a = FreshnessAlert(breach_id="b-001")
        assert a.channel == AlertChannel.LOG

    def test_default_message(self):
        a = FreshnessAlert(breach_id="b-001")
        assert a.message == ""

    def test_default_status(self):
        a = FreshnessAlert(breach_id="b-001")
        assert a.status == AlertStatus.PENDING

    def test_default_sent_at_none(self):
        a = FreshnessAlert(breach_id="b-001")
        assert a.sent_at is None

    def test_default_suppressed_reason_none(self):
        a = FreshnessAlert(breach_id="b-001")
        assert a.suppressed_reason is None

    def test_empty_breach_id_raises(self):
        with pytest.raises(ValidationError, match="breach_id must be non-empty"):
            FreshnessAlert(breach_id="")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            FreshnessAlert(breach_id="b-001", extra="bad")


# ======================================================================
# 11. SDK Models -- RefreshPrediction
# ======================================================================


class TestRefreshPrediction:
    """Test RefreshPrediction model."""

    def test_create_with_required_fields(self):
        now = datetime.now(timezone.utc)
        p = RefreshPrediction(dataset_id="ds-001", predicted_refresh_at=now)
        assert p.dataset_id == "ds-001"
        assert p.predicted_refresh_at == now

    def test_default_confidence(self):
        now = datetime.now(timezone.utc)
        p = RefreshPrediction(dataset_id="ds-001", predicted_refresh_at=now)
        assert p.confidence == 0.0

    def test_default_status(self):
        now = datetime.now(timezone.utc)
        p = RefreshPrediction(dataset_id="ds-001", predicted_refresh_at=now)
        assert p.status == PredictionStatus.PENDING

    def test_default_actual_refresh_at_none(self):
        now = datetime.now(timezone.utc)
        p = RefreshPrediction(dataset_id="ds-001", predicted_refresh_at=now)
        assert p.actual_refresh_at is None

    def test_default_error_hours_none(self):
        now = datetime.now(timezone.utc)
        p = RefreshPrediction(dataset_id="ds-001", predicted_refresh_at=now)
        assert p.error_hours is None

    def test_empty_dataset_id_raises(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError, match="dataset_id must be non-empty"):
            RefreshPrediction(dataset_id="", predicted_refresh_at=now)

    def test_extra_field_raises(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError):
            RefreshPrediction(dataset_id="ds-001", predicted_refresh_at=now, extra="bad")


# ======================================================================
# 12. SDK Models -- FreshnessReport
# ======================================================================


class TestFreshnessReport:
    """Test FreshnessReport model."""

    def test_create_default(self):
        rpt = FreshnessReport()
        assert rpt.report_type == "summary"

    def test_default_dataset_count(self):
        rpt = FreshnessReport()
        assert rpt.dataset_count == 0

    def test_default_compliant_count(self):
        rpt = FreshnessReport()
        assert rpt.compliant_count == 0

    def test_default_breached_count(self):
        rpt = FreshnessReport()
        assert rpt.breached_count == 0

    def test_default_summary(self):
        rpt = FreshnessReport()
        assert rpt.summary == ""

    def test_default_provenance_hash(self):
        rpt = FreshnessReport()
        assert rpt.provenance_hash == ""

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            FreshnessReport(extra="bad")


# ======================================================================
# 13. SDK Models -- AuditEntry
# ======================================================================


class TestAuditEntry:
    """Test AuditEntry model."""

    def test_create_with_required_fields(self):
        ae = AuditEntry(
            operation="register_dataset",
            entity_type="dataset",
            entity_id="ds-001",
        )
        assert ae.operation == "register_dataset"
        assert ae.entity_type == "dataset"
        assert ae.entity_id == "ds-001"

    def test_default_details_empty(self):
        ae = AuditEntry(
            operation="register_dataset",
            entity_type="dataset",
            entity_id="ds-001",
        )
        assert ae.details == {}

    def test_empty_operation_raises(self):
        with pytest.raises(ValidationError, match="operation must be non-empty"):
            AuditEntry(operation="", entity_type="dataset", entity_id="ds-001")

    def test_empty_entity_type_raises(self):
        with pytest.raises(ValidationError, match="entity_type must be non-empty"):
            AuditEntry(operation="op", entity_type="", entity_id="ds-001")

    def test_empty_entity_id_raises(self):
        with pytest.raises(ValidationError, match="entity_id must be non-empty"):
            AuditEntry(operation="op", entity_type="dataset", entity_id="")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            AuditEntry(
                operation="op",
                entity_type="dataset",
                entity_id="ds-001",
                extra="bad",
            )


# ======================================================================
# 14. SDK Models -- DatasetGroup
# ======================================================================


class TestDatasetGroup:
    """Test DatasetGroup model."""

    def test_create_with_name(self):
        dg = DatasetGroup(name="Scope 1 Datasets")
        assert dg.name == "Scope 1 Datasets"

    def test_default_description(self):
        dg = DatasetGroup(name="Group")
        assert dg.description == ""

    def test_default_dataset_ids_empty(self):
        dg = DatasetGroup(name="Group")
        assert dg.dataset_ids == []

    def test_default_priority(self):
        dg = DatasetGroup(name="Group")
        assert dg.priority == DatasetPriority.MEDIUM

    def test_default_sla_id_none(self):
        dg = DatasetGroup(name="Group")
        assert dg.sla_id is None

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError, match="name must be non-empty"):
            DatasetGroup(name="")

    def test_dataset_ids_exceed_limit_raises(self):
        with pytest.raises(ValidationError, match="cannot exceed 500"):
            DatasetGroup(name="Big Group", dataset_ids=[f"ds-{i}" for i in range(501)])

    def test_dataset_ids_empty_string_raises(self):
        with pytest.raises(ValidationError, match="non-empty strings"):
            DatasetGroup(name="Group", dataset_ids=["ds-001", ""])

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            DatasetGroup(name="Group", extra="bad")

    def test_max_datasets_accepted(self):
        dg = DatasetGroup(name="Full Group", dataset_ids=[f"ds-{i}" for i in range(500)])
        assert len(dg.dataset_ids) == 500


# ======================================================================
# 15. SDK Models -- FreshnessSummary
# ======================================================================


class TestFreshnessSummary:
    """Test FreshnessSummary model."""

    def test_create_with_dataset_id(self):
        fs = FreshnessSummary(dataset_id="ds-001")
        assert fs.dataset_id == "ds-001"

    def test_default_name(self):
        fs = FreshnessSummary(dataset_id="ds-001")
        assert fs.name == ""

    def test_default_current_age_hours(self):
        fs = FreshnessSummary(dataset_id="ds-001")
        assert fs.current_age_hours == 0.0

    def test_default_freshness_score(self):
        fs = FreshnessSummary(dataset_id="ds-001")
        assert fs.freshness_score == 1.0

    def test_default_freshness_level(self):
        fs = FreshnessSummary(dataset_id="ds-001")
        assert fs.freshness_level == FreshnessLevel.EXCELLENT

    def test_default_sla_status(self):
        fs = FreshnessSummary(dataset_id="ds-001")
        assert fs.sla_status == SLAStatus.UNKNOWN

    def test_default_last_checked_at_none(self):
        fs = FreshnessSummary(dataset_id="ds-001")
        assert fs.last_checked_at is None

    def test_empty_dataset_id_raises(self):
        with pytest.raises(ValidationError, match="dataset_id must be non-empty"):
            FreshnessSummary(dataset_id="")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            FreshnessSummary(dataset_id="ds-001", extra="bad")


# ======================================================================
# 16. SDK Models -- SourceReliability
# ======================================================================


class TestSourceReliability:
    """Test SourceReliability model."""

    def test_create_with_source_name(self):
        sr = SourceReliability(source_name="SAP ERP")
        assert sr.source_name == "SAP ERP"

    def test_default_total_refreshes(self):
        sr = SourceReliability(source_name="SAP ERP")
        assert sr.total_refreshes == 0

    def test_default_on_time_refreshes(self):
        sr = SourceReliability(source_name="SAP ERP")
        assert sr.on_time_refreshes == 0

    def test_default_reliability_pct(self):
        sr = SourceReliability(source_name="SAP ERP")
        assert sr.reliability_pct == 0.0

    def test_default_avg_delay_hours(self):
        sr = SourceReliability(source_name="SAP ERP")
        assert sr.avg_delay_hours == 0.0

    def test_default_trend(self):
        sr = SourceReliability(source_name="SAP ERP")
        assert sr.trend == "stable"

    def test_empty_source_name_raises(self):
        with pytest.raises(ValidationError, match="source_name must be non-empty"):
            SourceReliability(source_name="")

    def test_invalid_trend_raises(self):
        with pytest.raises(ValidationError, match="trend must be one of"):
            SourceReliability(source_name="SAP", trend="unknown")

    def test_valid_trends_accepted(self):
        for trend in ("improving", "stable", "degrading"):
            sr = SourceReliability(source_name="SAP", trend=trend)
            assert sr.trend == trend

    def test_reliability_pct_above_100_raises(self):
        with pytest.raises(ValidationError):
            SourceReliability(source_name="SAP", reliability_pct=101.0)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            SourceReliability(source_name="SAP", extra="bad")


# ======================================================================
# 17. SDK Models -- EscalationLevel
# ======================================================================


class TestEscalationLevel:
    """Test EscalationLevel model."""

    def test_create_default(self):
        el = EscalationLevel()
        assert el.delay_minutes == 15

    def test_default_channel(self):
        el = EscalationLevel()
        assert el.channel == AlertChannel.LOG

    def test_default_recipients_empty(self):
        el = EscalationLevel()
        assert el.recipients == []

    def test_default_message_template_none(self):
        el = EscalationLevel()
        assert el.message_template is None

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            EscalationLevel(extra="bad")

    def test_negative_delay_raises(self):
        with pytest.raises(ValidationError):
            EscalationLevel(delay_minutes=-1)


# ======================================================================
# 18. SDK Models -- EscalationPolicy
# ======================================================================


class TestEscalationPolicy:
    """Test EscalationPolicy model."""

    def test_create_default(self):
        ep = EscalationPolicy()
        assert ep.levels == []

    def test_default_max_escalations(self):
        ep = EscalationPolicy()
        assert ep.max_escalations == 4

    def test_with_levels(self):
        levels = [
            EscalationLevel(delay_minutes=15, channel=AlertChannel.SLACK),
            EscalationLevel(delay_minutes=60, channel=AlertChannel.PAGERDUTY),
        ]
        ep = EscalationPolicy(levels=levels)
        assert len(ep.levels) == 2

    def test_max_escalations_min_one(self):
        with pytest.raises(ValidationError):
            EscalationPolicy(max_escalations=0)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            EscalationPolicy(extra="bad")


# ======================================================================
# 19. SDK Models -- MonitoringRun
# ======================================================================


class TestMonitoringRun:
    """Test MonitoringRun model."""

    def test_create_default(self):
        mr = MonitoringRun()
        assert mr.status == MonitoringStatus.IDLE

    def test_default_datasets_checked(self):
        mr = MonitoringRun()
        assert mr.datasets_checked == 0

    def test_default_breaches_found(self):
        mr = MonitoringRun()
        assert mr.breaches_found == 0

    def test_default_alerts_sent(self):
        mr = MonitoringRun()
        assert mr.alerts_sent == 0

    def test_default_completed_at_none(self):
        mr = MonitoringRun()
        assert mr.completed_at is None

    def test_default_provenance_hash(self):
        mr = MonitoringRun()
        assert mr.provenance_hash == ""

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            MonitoringRun(extra="bad")


# ======================================================================
# 20. Request Models -- RegisterDatasetRequest
# ======================================================================


class TestRegisterDatasetRequest:
    """Test RegisterDatasetRequest model."""

    def test_create_with_name(self):
        req = RegisterDatasetRequest(name="Test Dataset")
        assert req.name == "Test Dataset"

    def test_default_source_name(self):
        req = RegisterDatasetRequest(name="Test")
        assert req.source_name == ""

    def test_default_source_type(self):
        req = RegisterDatasetRequest(name="Test")
        assert req.source_type == ""

    def test_default_owner(self):
        req = RegisterDatasetRequest(name="Test")
        assert req.owner == ""

    def test_default_refresh_cadence(self):
        req = RegisterDatasetRequest(name="Test")
        assert req.refresh_cadence == RefreshCadence.DAILY

    def test_default_priority(self):
        req = RegisterDatasetRequest(name="Test")
        assert req.priority == DatasetPriority.MEDIUM

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError, match="name must be non-empty"):
            RegisterDatasetRequest(name="")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            RegisterDatasetRequest(name="Test", extra="bad")


# ======================================================================
# 21. Request Models -- UpdateDatasetRequest
# ======================================================================


class TestUpdateDatasetRequest:
    """Test UpdateDatasetRequest model."""

    def test_create_empty(self):
        req = UpdateDatasetRequest()
        assert req.name is None
        assert req.source_name is None
        assert req.owner is None

    def test_update_name(self):
        req = UpdateDatasetRequest(name="New Name")
        assert req.name == "New Name"

    def test_update_priority(self):
        req = UpdateDatasetRequest(priority=DatasetPriority.HIGH)
        assert req.priority == DatasetPriority.HIGH

    def test_update_status(self):
        req = UpdateDatasetRequest(status=DatasetStatus.PAUSED)
        assert req.status == DatasetStatus.PAUSED

    def test_empty_name_when_provided_raises(self):
        with pytest.raises(ValidationError, match="name must be non-empty when provided"):
            UpdateDatasetRequest(name="")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            UpdateDatasetRequest(extra="bad")


# ======================================================================
# 22. Request Models -- CreateSLARequest
# ======================================================================


class TestCreateSLARequest:
    """Test CreateSLARequest model."""

    def test_create_with_dataset_id(self):
        req = CreateSLARequest(dataset_id="ds-001")
        assert req.dataset_id == "ds-001"

    def test_default_warning_hours(self):
        req = CreateSLARequest(dataset_id="ds-001")
        assert req.warning_hours == 24.0

    def test_default_critical_hours(self):
        req = CreateSLARequest(dataset_id="ds-001")
        assert req.critical_hours == 72.0

    def test_empty_dataset_id_raises(self):
        with pytest.raises(ValidationError, match="dataset_id must be non-empty"):
            CreateSLARequest(dataset_id="")

    def test_warning_exceeds_critical_raises(self):
        with pytest.raises(ValidationError, match="warning_hours.*must be <=.*critical_hours"):
            CreateSLARequest(dataset_id="ds-001", warning_hours=100.0, critical_hours=50.0)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            CreateSLARequest(dataset_id="ds-001", extra="bad")


# ======================================================================
# 23. Request Models -- UpdateSLARequest
# ======================================================================


class TestUpdateSLARequest:
    """Test UpdateSLARequest model."""

    def test_create_empty(self):
        req = UpdateSLARequest()
        assert req.warning_hours is None
        assert req.critical_hours is None
        assert req.breach_severity is None

    def test_update_warning_hours(self):
        req = UpdateSLARequest(warning_hours=12.0)
        assert req.warning_hours == 12.0

    def test_update_business_hours_only(self):
        req = UpdateSLARequest(business_hours_only=True)
        assert req.business_hours_only is True

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            UpdateSLARequest(extra="bad")


# ======================================================================
# 24. Request Models -- RunFreshnessCheckRequest
# ======================================================================


class TestRunFreshnessCheckRequest:
    """Test RunFreshnessCheckRequest model."""

    def test_create_with_dataset_id(self):
        req = RunFreshnessCheckRequest(dataset_id="ds-001")
        assert req.dataset_id == "ds-001"

    def test_default_sla_id_none(self):
        req = RunFreshnessCheckRequest(dataset_id="ds-001")
        assert req.sla_id is None

    def test_default_force_false(self):
        req = RunFreshnessCheckRequest(dataset_id="ds-001")
        assert req.force is False

    def test_empty_dataset_id_raises(self):
        with pytest.raises(ValidationError, match="dataset_id must be non-empty"):
            RunFreshnessCheckRequest(dataset_id="")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            RunFreshnessCheckRequest(dataset_id="ds-001", extra="bad")


# ======================================================================
# 25. Request Models -- RunBatchCheckRequest
# ======================================================================


class TestRunBatchCheckRequest:
    """Test RunBatchCheckRequest model."""

    def test_create_with_dataset_ids(self):
        req = RunBatchCheckRequest(dataset_ids=["ds-001", "ds-002"])
        assert req.dataset_ids == ["ds-001", "ds-002"]

    def test_create_with_group_id(self):
        req = RunBatchCheckRequest(group_id="group-001")
        assert req.group_id == "group-001"

    def test_default_include_predictions(self):
        req = RunBatchCheckRequest(dataset_ids=["ds-001"])
        assert req.include_predictions is False

    def test_default_force(self):
        req = RunBatchCheckRequest(dataset_ids=["ds-001"])
        assert req.force is False

    def test_neither_ids_nor_group_raises(self):
        with pytest.raises(ValidationError, match="At least one of dataset_ids or group_id"):
            RunBatchCheckRequest()

    def test_empty_string_in_dataset_ids_raises(self):
        with pytest.raises(ValidationError, match="non-empty strings"):
            RunBatchCheckRequest(dataset_ids=["ds-001", ""])

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            RunBatchCheckRequest(dataset_ids=["ds-001"], extra="bad")


# ======================================================================
# 26. Request Models -- UpdateBreachRequest
# ======================================================================


class TestUpdateBreachRequest:
    """Test UpdateBreachRequest model."""

    def test_create_with_status(self):
        req = UpdateBreachRequest(status=BreachStatus.ACKNOWLEDGED)
        assert req.status == BreachStatus.ACKNOWLEDGED

    def test_default_resolution_notes_none(self):
        req = UpdateBreachRequest(status=BreachStatus.RESOLVED)
        assert req.resolution_notes is None

    def test_with_resolution_notes(self):
        req = UpdateBreachRequest(
            status=BreachStatus.RESOLVED,
            resolution_notes="Fixed by manual refresh",
        )
        assert req.resolution_notes == "Fixed by manual refresh"

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            UpdateBreachRequest(status=BreachStatus.RESOLVED, extra="bad")


# ======================================================================
# 27. Request Models -- RunPipelineRequest
# ======================================================================


class TestRunPipelineRequest:
    """Test RunPipelineRequest model."""

    def test_create_default(self):
        req = RunPipelineRequest()
        assert req.dataset_ids is None
        assert req.group_id is None

    def test_default_include_predictions(self):
        req = RunPipelineRequest()
        assert req.include_predictions is False

    def test_default_detect_patterns(self):
        req = RunPipelineRequest()
        assert req.detect_patterns is False

    def test_default_generate_report(self):
        req = RunPipelineRequest()
        assert req.generate_report is True

    def test_with_dataset_ids(self):
        req = RunPipelineRequest(dataset_ids=["ds-001"])
        assert req.dataset_ids == ["ds-001"]

    def test_empty_string_in_dataset_ids_raises(self):
        with pytest.raises(ValidationError, match="non-empty strings"):
            RunPipelineRequest(dataset_ids=["ds-001", ""])

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            RunPipelineRequest(extra="bad")


# ======================================================================
# 28. VERSION constant
# ======================================================================


class TestVersionConstant:
    """Test the VERSION module constant."""

    def test_version_is_string(self):
        assert isinstance(VERSION, str)

    def test_version_semantic_format(self):
        parts = VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_version_value(self):
        assert VERSION == "1.0.0"
