# -*- coding: utf-8 -*-
"""
Unit Tests for Observability Agent Data Models (AGENT-FOUND-010)

Tests all Pydantic models, enumerations, validators, and request/response
wrappers defined in greenlang.observability_agent.models.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from greenlang.observability_agent.models import (
    # Enums
    AlertSeverity,
    AlertStatus,
    HealthStatus,
    LogLevel,
    MetricType,
    ProbeType,
    SLOType,
    TraceStatus,
    # SDK models
    AlertInstance,
    DashboardConfig,
    HealthProbeResult,
    LogRecord,
    MetricRecording,
    ObservabilityStatistics,
    SLODefinition,
    SLOStatus,
    TraceRecord,
    # Request models
    CreateAlertRuleRequest,
    CreateSLORequest,
    CreateSpanRequest,
    HealthCheckRequest,
    IngestLogRequest,
    RecordMetricRequest,
)


# ==========================================================================
# Enum Tests
# ==========================================================================

class TestEnumValues:
    """Verify every enum member has the expected string value."""

    def test_metric_type_values(self):
        assert MetricType.COUNTER == "counter"
        assert MetricType.GAUGE == "gauge"
        assert MetricType.HISTOGRAM == "histogram"
        assert MetricType.SUMMARY == "summary"

    def test_metric_type_member_count(self):
        assert len(MetricType) == 4

    def test_alert_severity_values(self):
        assert AlertSeverity.INFO == "info"
        assert AlertSeverity.WARNING == "warning"
        assert AlertSeverity.ERROR == "error"
        assert AlertSeverity.CRITICAL == "critical"

    def test_alert_severity_member_count(self):
        assert len(AlertSeverity) == 4

    def test_alert_status_values(self):
        assert AlertStatus.FIRING == "firing"
        assert AlertStatus.RESOLVED == "resolved"
        assert AlertStatus.PENDING == "pending"

    def test_alert_status_member_count(self):
        assert len(AlertStatus) == 3

    def test_health_status_values(self):
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"

    def test_health_status_member_count(self):
        assert len(HealthStatus) == 3

    def test_trace_status_values(self):
        assert TraceStatus.UNSET == "unset"
        assert TraceStatus.OK == "ok"
        assert TraceStatus.ERROR == "error"

    def test_trace_status_member_count(self):
        assert len(TraceStatus) == 3

    def test_log_level_values(self):
        assert LogLevel.DEBUG == "debug"
        assert LogLevel.INFO == "info"
        assert LogLevel.WARNING == "warning"
        assert LogLevel.ERROR == "error"
        assert LogLevel.CRITICAL == "critical"

    def test_log_level_member_count(self):
        assert len(LogLevel) == 5

    def test_slo_type_values(self):
        assert SLOType.AVAILABILITY == "availability"
        assert SLOType.LATENCY == "latency"
        assert SLOType.THROUGHPUT == "throughput"
        assert SLOType.ERROR_RATE == "error_rate"
        assert SLOType.SATURATION == "saturation"

    def test_slo_type_member_count(self):
        assert len(SLOType) == 5

    def test_probe_type_values(self):
        assert ProbeType.LIVENESS == "liveness"
        assert ProbeType.READINESS == "readiness"
        assert ProbeType.STARTUP == "startup"

    def test_probe_type_member_count(self):
        assert len(ProbeType) == 3


# ==========================================================================
# MetricRecording Tests
# ==========================================================================

class TestMetricRecording:
    """Tests for the MetricRecording Pydantic model."""

    def test_creation_with_required_fields(self):
        rec = MetricRecording(metric_name="http_requests_total", value=42.0)
        assert rec.metric_name == "http_requests_total"
        assert rec.value == 42.0

    def test_defaults(self):
        rec = MetricRecording(metric_name="test_metric", value=1.0)
        assert rec.recording_id  # non-empty UUID
        assert rec.labels == {}
        assert rec.tenant_id == "default"
        assert rec.provenance_hash == ""
        assert isinstance(rec.timestamp, datetime)

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            MetricRecording(metric_name="", value=1.0)

    def test_whitespace_name_raises(self):
        with pytest.raises(ValidationError):
            MetricRecording(metric_name="   ", value=1.0)

    def test_labels_stored(self):
        rec = MetricRecording(
            metric_name="cpu_usage", value=0.85, labels={"host": "web1"},
        )
        assert rec.labels == {"host": "web1"}

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            MetricRecording(metric_name="m", value=1.0, unknown_field="x")


# ==========================================================================
# TraceRecord Tests
# ==========================================================================

class TestTraceRecord:
    """Tests for the TraceRecord Pydantic model."""

    def test_creation(self):
        rec = TraceRecord(
            trace_id="abc123",
            span_id="def456",
            operation_name="db_query",
        )
        assert rec.trace_id == "abc123"
        assert rec.span_id == "def456"
        assert rec.operation_name == "db_query"

    def test_defaults(self):
        rec = TraceRecord(
            trace_id="t1", span_id="s1", operation_name="op",
        )
        assert rec.parent_span_id is None
        assert rec.service_name == ""
        assert rec.status == TraceStatus.UNSET
        assert rec.end_time is None
        assert rec.duration_ms == 0.0
        assert rec.attributes == {}
        assert rec.events == []
        assert rec.tenant_id == "default"
        assert rec.provenance_hash == ""

    def test_empty_trace_id_raises(self):
        with pytest.raises(ValidationError):
            TraceRecord(trace_id="", span_id="s1", operation_name="op")

    def test_empty_span_id_raises(self):
        with pytest.raises(ValidationError):
            TraceRecord(trace_id="t1", span_id="", operation_name="op")

    def test_empty_operation_name_raises(self):
        with pytest.raises(ValidationError):
            TraceRecord(trace_id="t1", span_id="s1", operation_name="")

    def test_duration_field(self):
        rec = TraceRecord(
            trace_id="t1", span_id="s1", operation_name="op", duration_ms=42.5,
        )
        assert rec.duration_ms == pytest.approx(42.5)

    def test_events_list(self):
        rec = TraceRecord(
            trace_id="t1", span_id="s1", operation_name="op",
            events=[{"name": "event1", "ts": "2026-01-01T00:00:00Z"}],
        )
        assert len(rec.events) == 1


# ==========================================================================
# LogRecord Tests
# ==========================================================================

class TestLogRecord:
    """Tests for the LogRecord Pydantic model."""

    def test_creation(self):
        rec = LogRecord(message="Hello world")
        assert rec.message == "Hello world"

    def test_defaults(self):
        rec = LogRecord(message="msg")
        assert rec.level == LogLevel.INFO
        assert rec.correlation_id is None
        assert rec.trace_id is None
        assert rec.span_id is None
        assert rec.agent_id is None
        assert rec.tenant_id == "default"
        assert rec.attributes == {}

    def test_levels(self):
        for lvl in LogLevel:
            rec = LogRecord(message="test", level=lvl)
            assert rec.level == lvl

    def test_empty_message_raises(self):
        with pytest.raises(ValidationError):
            LogRecord(message="")

    def test_whitespace_message_raises(self):
        with pytest.raises(ValidationError):
            LogRecord(message="   ")

    def test_all_fields_populated(self):
        rec = LogRecord(
            message="detail",
            level=LogLevel.ERROR,
            correlation_id="corr-1",
            trace_id="t-1",
            span_id="s-1",
            agent_id="agent-1",
            tenant_id="tenant-a",
            attributes={"key": "value"},
        )
        assert rec.correlation_id == "corr-1"
        assert rec.attributes["key"] == "value"


# ==========================================================================
# AlertInstance Tests
# ==========================================================================

class TestAlertInstance:
    """Tests for the AlertInstance Pydantic model."""

    def test_creation(self):
        ai = AlertInstance(rule_name="high_cpu")
        assert ai.rule_name == "high_cpu"

    def test_defaults(self):
        ai = AlertInstance(rule_name="rule1")
        assert ai.status == AlertStatus.FIRING
        assert ai.severity == AlertSeverity.WARNING
        assert ai.metric_name == ""
        assert ai.metric_value == 0.0
        assert ai.threshold == 0.0
        assert ai.labels == {}
        assert ai.annotations == {}
        assert ai.resolved_at is None

    def test_status_values(self):
        for status in AlertStatus:
            ai = AlertInstance(rule_name="r", status=status)
            assert ai.status == status

    def test_empty_rule_name_raises(self):
        with pytest.raises(ValidationError):
            AlertInstance(rule_name="")


# ==========================================================================
# HealthProbeResult Tests
# ==========================================================================

class TestHealthProbeResult:
    """Tests for the HealthProbeResult Pydantic model."""

    def test_creation(self):
        hpr = HealthProbeResult(service_name="api-gateway")
        assert hpr.service_name == "api-gateway"

    def test_defaults(self):
        hpr = HealthProbeResult(service_name="svc")
        assert hpr.probe_type == ProbeType.LIVENESS
        assert hpr.status == HealthStatus.HEALTHY
        assert hpr.message is None
        assert hpr.details == {}
        assert hpr.duration_ms == 0.0

    def test_probe_types(self):
        for pt in ProbeType:
            hpr = HealthProbeResult(service_name="svc", probe_type=pt)
            assert hpr.probe_type == pt

    def test_empty_service_name_raises(self):
        with pytest.raises(ValidationError):
            HealthProbeResult(service_name="")


# ==========================================================================
# DashboardConfig Tests
# ==========================================================================

class TestDashboardConfig:
    """Tests for the DashboardConfig Pydantic model."""

    def test_creation(self):
        dc = DashboardConfig(name="Platform Overview")
        assert dc.name == "Platform Overview"

    def test_defaults(self):
        dc = DashboardConfig(name="dash")
        assert dc.description == ""
        assert dc.panels == []
        assert dc.time_range == "1h"
        assert dc.refresh_interval == "30s"
        assert dc.tenant_id == "default"

    def test_panels_list(self):
        panels = [{"type": "graph", "title": "CPU"}]
        dc = DashboardConfig(name="d", panels=panels)
        assert len(dc.panels) == 1

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            DashboardConfig(name="")


# ==========================================================================
# SLODefinition Tests
# ==========================================================================

class TestSLODefinition:
    """Tests for the SLODefinition Pydantic model."""

    def test_creation(self):
        slo = SLODefinition(name="API Availability")
        assert slo.name == "API Availability"

    def test_defaults(self):
        slo = SLODefinition(name="slo1")
        assert slo.slo_type == SLOType.AVAILABILITY
        assert slo.target == pytest.approx(0.999)
        assert slo.window_days == 30
        assert slo.burn_rate_thresholds == {
            "fast_burn": 14.4,
            "medium_burn": 6.0,
            "slow_burn": 1.0,
        }

    def test_target_validation_below_zero(self):
        with pytest.raises(ValidationError):
            SLODefinition(name="slo", target=-0.1)

    def test_target_validation_above_one(self):
        with pytest.raises(ValidationError):
            SLODefinition(name="slo", target=1.1)

    def test_target_boundary_zero(self):
        slo = SLODefinition(name="slo", target=0.0)
        assert slo.target == 0.0

    def test_target_boundary_one(self):
        slo = SLODefinition(name="slo", target=1.0)
        assert slo.target == 1.0

    def test_window_days_minimum(self):
        with pytest.raises(ValidationError):
            SLODefinition(name="slo", window_days=0)

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            SLODefinition(name="")


# ==========================================================================
# SLOStatus Tests
# ==========================================================================

class TestSLOStatus:
    """Tests for the SLOStatus Pydantic model."""

    def test_creation(self):
        st = SLOStatus(slo_id="slo-001")
        assert st.slo_id == "slo-001"

    def test_defaults(self):
        st = SLOStatus(slo_id="x")
        assert st.current_value == 1.0
        assert st.target == pytest.approx(0.999)
        assert st.compliance_ratio == 1.0
        assert st.error_budget_total == 0.0
        assert st.error_budget_consumed == 0.0
        assert st.error_budget_remaining == 0.0
        assert st.burn_rate_1h == 0.0
        assert st.burn_rate_6h == 0.0
        assert st.burn_rate_24h == 0.0
        assert st.is_burning is False
        assert st.window_start is None
        assert st.window_end is None

    def test_empty_slo_id_raises(self):
        with pytest.raises(ValidationError):
            SLOStatus(slo_id="")

    def test_burn_rate_fields(self):
        st = SLOStatus(
            slo_id="slo-1", burn_rate_1h=2.5, burn_rate_6h=1.2, burn_rate_24h=0.8,
        )
        assert st.burn_rate_1h == pytest.approx(2.5)
        assert st.burn_rate_6h == pytest.approx(1.2)
        assert st.burn_rate_24h == pytest.approx(0.8)


# ==========================================================================
# ObservabilityStatistics Tests
# ==========================================================================

class TestObservabilityStatistics:
    """Tests for the ObservabilityStatistics Pydantic model."""

    def test_creation_all_defaults(self):
        stats = ObservabilityStatistics()
        assert stats.total_metrics == 0
        assert stats.total_spans == 0
        assert stats.total_logs == 0
        assert stats.total_alerts_fired == 0
        assert stats.active_alerts == 0
        assert stats.active_spans == 0
        assert stats.slo_count == 0
        assert stats.health_checks_total == 0
        assert stats.uptime_seconds == 0.0

    def test_creation_with_values(self):
        stats = ObservabilityStatistics(
            total_metrics=100, total_spans=50, total_logs=200,
        )
        assert stats.total_metrics == 100
        assert stats.total_spans == 50
        assert stats.total_logs == 200


# ==========================================================================
# Request Model Tests
# ==========================================================================

class TestRecordMetricRequest:
    """Tests for RecordMetricRequest."""

    def test_creation(self):
        req = RecordMetricRequest(metric_name="cpu_usage", value=0.85)
        assert req.metric_name == "cpu_usage"
        assert req.value == 0.85

    def test_defaults(self):
        req = RecordMetricRequest(metric_name="m", value=1.0)
        assert req.labels == {}
        assert req.tenant_id == "default"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            RecordMetricRequest(metric_name="", value=1.0)


class TestCreateSpanRequest:
    """Tests for CreateSpanRequest."""

    def test_creation(self):
        req = CreateSpanRequest(operation_name="handle_request")
        assert req.operation_name == "handle_request"

    def test_defaults(self):
        req = CreateSpanRequest(operation_name="op")
        assert req.service_name == ""
        assert req.parent_span_id is None
        assert req.attributes == {}
        assert req.tenant_id == "default"

    def test_empty_operation_name_raises(self):
        with pytest.raises(ValidationError):
            CreateSpanRequest(operation_name="")


class TestIngestLogRequest:
    """Tests for IngestLogRequest."""

    def test_creation(self):
        req = IngestLogRequest(message="Log message")
        assert req.message == "Log message"

    def test_defaults(self):
        req = IngestLogRequest(message="msg")
        assert req.level == LogLevel.INFO
        assert req.agent_id is None
        assert req.trace_id is None

    def test_empty_message_raises(self):
        with pytest.raises(ValidationError):
            IngestLogRequest(message="")


class TestCreateAlertRuleRequest:
    """Tests for CreateAlertRuleRequest."""

    def test_creation(self):
        req = CreateAlertRuleRequest(
            name="high_cpu", metric_name="cpu_usage",
            condition="gt", threshold=0.9,
        )
        assert req.name == "high_cpu"
        assert req.condition == "gt"

    def test_invalid_condition_raises(self):
        with pytest.raises(ValidationError):
            CreateAlertRuleRequest(
                name="r", metric_name="m", condition="invalid", threshold=1.0,
            )

    @pytest.mark.parametrize("cond", ["gt", "lt", "eq", "gte", "lte", "ne"])
    def test_valid_conditions(self, cond):
        req = CreateAlertRuleRequest(
            name="r", metric_name="m", condition=cond, threshold=1.0,
        )
        assert req.condition == cond

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            CreateAlertRuleRequest(
                name="", metric_name="m", condition="gt", threshold=1.0,
            )


class TestCreateSLORequest:
    """Tests for CreateSLORequest."""

    def test_creation(self):
        req = CreateSLORequest(name="Availability SLO")
        assert req.name == "Availability SLO"

    def test_target_range_enforcement(self):
        with pytest.raises(ValidationError):
            CreateSLORequest(name="slo", target=-0.1)
        with pytest.raises(ValidationError):
            CreateSLORequest(name="slo", target=1.5)

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            CreateSLORequest(name="")


class TestHealthCheckRequest:
    """Tests for HealthCheckRequest."""

    def test_creation(self):
        req = HealthCheckRequest()
        assert req.probe_type == ProbeType.LIVENESS
        assert req.service_name == ""
        assert req.tenant_id == "default"

    def test_probe_type_options(self):
        for pt in ProbeType:
            req = HealthCheckRequest(probe_type=pt)
            assert req.probe_type == pt
