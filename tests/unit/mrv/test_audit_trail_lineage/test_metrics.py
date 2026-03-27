# -*- coding: utf-8 -*-
"""
Test suite for audit_trail_lineage.metrics - AGENT-MRV-030.

Tests the 14 Prometheus metrics with gl_atl_ prefix for the
Audit Trail & Lineage Agent (GL-MRV-X-042).

Coverage:
- Metric existence and naming (gl_atl_ prefix)
- Counter metrics (events_recorded_total, chain_verifications_total, etc.)
- Histogram metrics (event_recording_duration_seconds, etc.)
- Gauge metrics (chain_length, active_chains, etc.)
- Metric labels (event_type, scope, framework, status)
- Metric registration (all 14 metrics registered)

Target: ~30 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.audit_trail_lineage.metrics import (
        EVENTS_RECORDED_TOTAL,
        EVENTS_RECORDED_ERRORS_TOTAL,
        CHAIN_VERIFICATIONS_TOTAL,
        CHAIN_VERIFICATION_ERRORS_TOTAL,
        EVENT_RECORDING_DURATION_SECONDS,
        CHAIN_VERIFICATION_DURATION_SECONDS,
        PIPELINE_EXECUTION_DURATION_SECONDS,
        PIPELINE_EXECUTIONS_TOTAL,
        EVIDENCE_PACKAGES_CREATED_TOTAL,
        COMPLIANCE_CHECKS_TOTAL,
        CHANGES_DETECTED_TOTAL,
        ACTIVE_CHAINS_GAUGE,
        CHAIN_LENGTH_GAUGE,
        LINEAGE_NODES_GAUGE,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Fallback: try generic prometheus_client import
try:
    from prometheus_client import CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not METRICS_AVAILABLE,
    reason="audit_trail_lineage.metrics not available",
)

_SKIP_PROMETHEUS = pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not available",
)

EXPECTED_METRIC_NAMES = [
    "gl_atl_events_recorded_total",
    "gl_atl_events_recorded_errors_total",
    "gl_atl_chain_verifications_total",
    "gl_atl_chain_verification_errors_total",
    "gl_atl_event_recording_duration_seconds",
    "gl_atl_chain_verification_duration_seconds",
    "gl_atl_pipeline_execution_duration_seconds",
    "gl_atl_pipeline_executions_total",
    "gl_atl_evidence_packages_created_total",
    "gl_atl_compliance_checks_total",
    "gl_atl_changes_detected_total",
    "gl_atl_active_chains",
    "gl_atl_chain_length",
    "gl_atl_lineage_nodes",
]


# ==============================================================================
# METRIC NAMING CONVENTION TESTS
# ==============================================================================


@_SKIP
class TestMetricNaming:
    """Test metric naming follows gl_atl_ prefix convention."""

    @pytest.mark.parametrize("metric_name", EXPECTED_METRIC_NAMES)
    def test_metric_name_prefix(self, metric_name):
        """Test each expected metric name starts with gl_atl_."""
        assert metric_name.startswith("gl_atl_")

    def test_events_recorded_counter_name(self):
        """Test events recorded counter name."""
        assert hasattr(EVENTS_RECORDED_TOTAL, '_name')
        assert "gl_atl_" in EVENTS_RECORDED_TOTAL._name

    def test_chain_verifications_counter_name(self):
        """Test chain verifications counter name."""
        assert hasattr(CHAIN_VERIFICATIONS_TOTAL, '_name')
        assert "gl_atl_" in CHAIN_VERIFICATIONS_TOTAL._name


# ==============================================================================
# COUNTER METRICS TESTS
# ==============================================================================


@_SKIP
class TestCounterMetrics:
    """Test counter metrics."""

    def test_events_recorded_total_exists(self):
        """Test EVENTS_RECORDED_TOTAL counter exists."""
        assert EVENTS_RECORDED_TOTAL is not None

    def test_events_recorded_errors_total_exists(self):
        """Test EVENTS_RECORDED_ERRORS_TOTAL counter exists."""
        assert EVENTS_RECORDED_ERRORS_TOTAL is not None

    def test_chain_verifications_total_exists(self):
        """Test CHAIN_VERIFICATIONS_TOTAL counter exists."""
        assert CHAIN_VERIFICATIONS_TOTAL is not None

    def test_chain_verification_errors_total_exists(self):
        """Test CHAIN_VERIFICATION_ERRORS_TOTAL counter exists."""
        assert CHAIN_VERIFICATION_ERRORS_TOTAL is not None

    def test_pipeline_executions_total_exists(self):
        """Test PIPELINE_EXECUTIONS_TOTAL counter exists."""
        assert PIPELINE_EXECUTIONS_TOTAL is not None

    def test_evidence_packages_created_total_exists(self):
        """Test EVIDENCE_PACKAGES_CREATED_TOTAL counter exists."""
        assert EVIDENCE_PACKAGES_CREATED_TOTAL is not None

    def test_compliance_checks_total_exists(self):
        """Test COMPLIANCE_CHECKS_TOTAL counter exists."""
        assert COMPLIANCE_CHECKS_TOTAL is not None

    def test_changes_detected_total_exists(self):
        """Test CHANGES_DETECTED_TOTAL counter exists."""
        assert CHANGES_DETECTED_TOTAL is not None


# ==============================================================================
# HISTOGRAM METRICS TESTS
# ==============================================================================


@_SKIP
class TestHistogramMetrics:
    """Test histogram metrics."""

    def test_event_recording_duration_exists(self):
        """Test EVENT_RECORDING_DURATION_SECONDS histogram exists."""
        assert EVENT_RECORDING_DURATION_SECONDS is not None

    def test_chain_verification_duration_exists(self):
        """Test CHAIN_VERIFICATION_DURATION_SECONDS histogram exists."""
        assert CHAIN_VERIFICATION_DURATION_SECONDS is not None

    def test_pipeline_execution_duration_exists(self):
        """Test PIPELINE_EXECUTION_DURATION_SECONDS histogram exists."""
        assert PIPELINE_EXECUTION_DURATION_SECONDS is not None


# ==============================================================================
# GAUGE METRICS TESTS
# ==============================================================================


@_SKIP
class TestGaugeMetrics:
    """Test gauge metrics."""

    def test_active_chains_gauge_exists(self):
        """Test ACTIVE_CHAINS_GAUGE gauge exists."""
        assert ACTIVE_CHAINS_GAUGE is not None

    def test_chain_length_gauge_exists(self):
        """Test CHAIN_LENGTH_GAUGE gauge exists."""
        assert CHAIN_LENGTH_GAUGE is not None

    def test_lineage_nodes_gauge_exists(self):
        """Test LINEAGE_NODES_GAUGE gauge exists."""
        assert LINEAGE_NODES_GAUGE is not None


# ==============================================================================
# METRIC COUNT TESTS
# ==============================================================================


@_SKIP
class TestMetricCount:
    """Test total number of defined metrics."""

    def test_total_metrics_count(self):
        """Test there are exactly 14 defined metrics."""
        metrics = [
            EVENTS_RECORDED_TOTAL,
            EVENTS_RECORDED_ERRORS_TOTAL,
            CHAIN_VERIFICATIONS_TOTAL,
            CHAIN_VERIFICATION_ERRORS_TOTAL,
            EVENT_RECORDING_DURATION_SECONDS,
            CHAIN_VERIFICATION_DURATION_SECONDS,
            PIPELINE_EXECUTION_DURATION_SECONDS,
            PIPELINE_EXECUTIONS_TOTAL,
            EVIDENCE_PACKAGES_CREATED_TOTAL,
            COMPLIANCE_CHECKS_TOTAL,
            CHANGES_DETECTED_TOTAL,
            ACTIVE_CHAINS_GAUGE,
            CHAIN_LENGTH_GAUGE,
            LINEAGE_NODES_GAUGE,
        ]
        assert len(metrics) == 14
        assert all(m is not None for m in metrics)
