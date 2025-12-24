# -*- coding: utf-8 -*-
"""
Unit tests for GL-008 TRAPCATCHER Prometheus Metrics Exporter.

Tests the PrometheusMetricsExporter class and related utilities
for observability compliance.

Author: GL-BackendDeveloper
Date: December 2025
"""

import pytest
import time
from unittest.mock import Mock, patch
from prometheus_client import CollectorRegistry

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from monitoring.metrics_exporter import (
    PrometheusMetricsExporter,
    DiagnosisMetrics,
    AccuracyWindow,
    TrapConditionLabel,
    TrapTypeLabel,
    ModalityLabel,
    SeverityLabel,
    get_metrics_exporter,
    create_metrics_endpoint,
    create_metrics_middleware,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def registry():
    """Create isolated registry for testing."""
    return CollectorRegistry()


@pytest.fixture
def exporter(registry):
    """Create exporter with isolated registry."""
    exp = PrometheusMetricsExporter(registry=registry)
    exp.initialize("1.0.0", "test")
    return exp


@pytest.fixture
def sample_diagnosis_metrics():
    """Create sample diagnosis metrics."""
    return DiagnosisMetrics(
        trap_id="ST-001",
        trap_type="thermodynamic",
        condition="leaking",
        severity="moderate",
        confidence=0.87,
        energy_loss_kw=8.2,
        co2_kg=4500.0,
        diagnosis_duration_seconds=0.05,
        is_failure=True,
        failure_mode="leak_detected",
        facility="plant_a",
        modality="multimodal",
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestMetricsExporterInitialization:
    """Test exporter initialization."""

    def test_create_exporter(self, registry):
        """Test basic exporter creation."""
        exporter = PrometheusMetricsExporter(registry=registry)
        assert exporter is not None
        assert exporter._initialized is False

    def test_initialize_exporter(self, registry):
        """Test exporter initialization with metadata."""
        exporter = PrometheusMetricsExporter(registry=registry)
        exporter.initialize("1.0.0", "production", "instance-1")

        assert exporter._initialized is True
        assert exporter.version == "1.0.0"
        assert exporter.environment == "production"

    def test_get_global_exporter(self):
        """Test global exporter singleton."""
        exporter1 = get_metrics_exporter()
        exporter2 = get_metrics_exporter()

        assert exporter1 is exporter2


# =============================================================================
# DIAGNOSIS METRICS TESTS
# =============================================================================

class TestDiagnosisMetrics:
    """Test diagnosis metrics recording."""

    def test_measure_diagnosis_context_manager(self, exporter):
        """Test diagnosis timing context manager."""
        with exporter.measure_diagnosis("thermodynamic", "acoustic"):
            time.sleep(0.01)  # Simulate work

        # Metrics should be recorded (no assertion on values due to timing)
        metrics = exporter.get_metrics()
        assert b"trapcatcher_diagnosis_latency_seconds" in metrics

    def test_measure_diagnosis_records_error(self, exporter):
        """Test diagnosis timing on error."""
        with pytest.raises(ValueError):
            with exporter.measure_diagnosis("thermodynamic", "acoustic"):
                raise ValueError("Test error")

        metrics = exporter.get_metrics()
        assert b"trapcatcher_diagnosis_latency_seconds" in metrics

    def test_record_diagnosis(self, exporter, sample_diagnosis_metrics):
        """Test recording complete diagnosis metrics."""
        exporter.record_diagnosis(sample_diagnosis_metrics)

        metrics = exporter.get_metrics()

        # Check counters were incremented
        assert b"trapcatcher_traps_analyzed_total" in metrics
        assert b"trapcatcher_energy_loss_kwh_total" in metrics
        assert b"trapcatcher_co2_emissions_kg_total" in metrics
        assert b"trapcatcher_failed_traps_total" in metrics

    def test_record_healthy_diagnosis(self, exporter):
        """Test recording healthy trap diagnosis."""
        metrics = DiagnosisMetrics(
            trap_id="ST-002",
            trap_type="thermostatic",
            condition="operating_normal",
            severity="none",
            confidence=0.95,
            energy_loss_kw=0.0,
            co2_kg=0.0,
            diagnosis_duration_seconds=0.03,
            is_failure=False,
            facility="plant_a",
            modality="thermal",
        )

        exporter.record_diagnosis(metrics)
        output = exporter.get_metrics()

        assert b"trapcatcher_traps_analyzed_total" in output


# =============================================================================
# ACCURACY TRACKING TESTS
# =============================================================================

class TestAccuracyTracking:
    """Test accuracy rate tracking."""

    def test_accuracy_window_initialization(self):
        """Test AccuracyWindow initial state."""
        window = AccuracyWindow()

        assert window.correct_predictions == 0
        assert window.total_predictions == 0
        assert window.accuracy == 1.0  # Default when no predictions

    def test_accuracy_window_add_prediction(self):
        """Test adding predictions to window."""
        window = AccuracyWindow()

        window.add_prediction(is_correct=True)
        window.add_prediction(is_correct=True)
        window.add_prediction(is_correct=False)

        assert window.total_predictions == 3
        assert window.correct_predictions == 2
        assert window.accuracy == pytest.approx(0.6667, rel=0.01)

    def test_record_accuracy(self, exporter):
        """Test recording accuracy to gauge."""
        exporter.record_accuracy("thermodynamic", "acoustic", is_correct=True)
        exporter.record_accuracy("thermodynamic", "acoustic", is_correct=True)
        exporter.record_accuracy("thermodynamic", "acoustic", is_correct=False)

        metrics = exporter.get_metrics()
        assert b"trapcatcher_accuracy_rate" in metrics


# =============================================================================
# FLEET METRICS TESTS
# =============================================================================

class TestFleetMetrics:
    """Test fleet-level metrics."""

    def test_update_fleet_health(self, exporter):
        """Test updating fleet health metrics."""
        exporter.update_fleet_health(
            facility="plant_a",
            health_score=85.5,
            total_energy_loss_kw=125.0,
            total_co2_kg=45000.0,
        )

        metrics = exporter.get_metrics()
        assert b"trapcatcher_fleet_health_score" in metrics
        assert b"trapcatcher_energy_loss_kw" in metrics


# =============================================================================
# API METRICS TESTS (RED PATTERN)
# =============================================================================

class TestAPIMetrics:
    """Test API request metrics (RED pattern)."""

    def test_measure_api_request(self, exporter):
        """Test API request timing context manager."""
        with exporter.measure_api_request("POST", "/diagnose"):
            time.sleep(0.001)

        metrics = exporter.get_metrics()
        assert b"trapcatcher_api_requests_total" in metrics
        assert b"trapcatcher_api_latency_seconds" in metrics

    def test_record_api_request_manual(self, exporter):
        """Test manual API request recording."""
        exporter.record_api_request(
            method="GET",
            endpoint="/health",
            status_code=200,
            duration_seconds=0.005,
        )

        metrics = exporter.get_metrics()
        assert b"trapcatcher_api_requests_total" in metrics


# =============================================================================
# ERROR METRICS TESTS
# =============================================================================

class TestErrorMetrics:
    """Test error tracking metrics."""

    def test_record_error(self, exporter):
        """Test error counter increment."""
        exporter.record_error(
            error_type="validation",
            component="bounds_validator",
            severity="medium",
        )

        metrics = exporter.get_metrics()
        assert b"trapcatcher_errors_total" in metrics


# =============================================================================
# CONNECTION METRICS TESTS (USE PATTERN)
# =============================================================================

class TestConnectionMetrics:
    """Test connection/utilization metrics (USE pattern)."""

    def test_set_active_connections(self, exporter):
        """Test setting active connection count."""
        exporter.set_active_connections(
            connection_type="sensor",
            protocol="opcua",
            count=5,
        )

        metrics = exporter.get_metrics()
        assert b"trapcatcher_active_connections" in metrics

    def test_update_sensor_freshness(self, exporter):
        """Test sensor data freshness gauge."""
        exporter.update_sensor_freshness(
            sensor_type="acoustic",
            trap_id="ST-001",
            age_seconds=2.5,
        )

        metrics = exporter.get_metrics()
        assert b"trapcatcher_sensor_data_age_seconds" in metrics


# =============================================================================
# COMPONENT HEALTH TESTS
# =============================================================================

class TestComponentHealth:
    """Test component health metrics."""

    def test_set_component_health_healthy(self, exporter):
        """Test setting component as healthy."""
        exporter.set_component_health(
            component="classifier",
            instance="main",
            is_healthy=True,
        )

        metrics = exporter.get_metrics()
        assert b"trapcatcher_component_health" in metrics

    def test_set_component_health_unhealthy(self, exporter):
        """Test setting component as unhealthy."""
        exporter.set_component_health(
            component="database",
            instance="primary",
            is_healthy=False,
        )

        metrics = exporter.get_metrics()
        assert b"trapcatcher_component_health" in metrics


# =============================================================================
# VALIDATION METRICS TESTS
# =============================================================================

class TestValidationMetrics:
    """Test validation timing metrics."""

    def test_measure_validation(self, exporter):
        """Test validation timing context manager."""
        with exporter.measure_validation("bounds"):
            time.sleep(0.001)

        metrics = exporter.get_metrics()
        assert b"trapcatcher_validation_duration_seconds" in metrics


# =============================================================================
# OUTPUT TESTS
# =============================================================================

class TestMetricsOutput:
    """Test metrics output generation."""

    def test_get_metrics_returns_bytes(self, exporter):
        """Test get_metrics returns bytes."""
        metrics = exporter.get_metrics()

        assert isinstance(metrics, bytes)
        assert len(metrics) > 0

    def test_get_content_type(self, exporter):
        """Test content type is correct."""
        content_type = exporter.get_content_type()

        assert "text/plain" in content_type or "openmetrics" in content_type.lower()

    def test_get_metrics_summary(self, exporter):
        """Test human-readable metrics summary."""
        summary = exporter.get_metrics_summary()

        assert "initialized" in summary
        assert "version" in summary
        assert "environment" in summary
        assert summary["initialized"] is True


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestEnums:
    """Test metric label enums."""

    def test_trap_condition_labels(self):
        """Test TrapConditionLabel enum values."""
        assert TrapConditionLabel.NORMAL == "normal"
        assert TrapConditionLabel.LEAKING == "leaking"
        assert TrapConditionLabel.FAILED_OPEN == "failed_open"

    def test_trap_type_labels(self):
        """Test TrapTypeLabel enum values."""
        assert TrapTypeLabel.THERMODYNAMIC == "thermodynamic"
        assert TrapTypeLabel.THERMOSTATIC == "thermostatic"

    def test_modality_labels(self):
        """Test ModalityLabel enum values."""
        assert ModalityLabel.ACOUSTIC == "acoustic"
        assert ModalityLabel.MULTIMODAL == "multimodal"

    def test_severity_labels(self):
        """Test SeverityLabel enum values."""
        assert SeverityLabel.CRITICAL == "critical"
        assert SeverityLabel.LOW == "low"


# =============================================================================
# DATA CLASS TESTS
# =============================================================================

class TestDataClasses:
    """Test metric data classes."""

    def test_diagnosis_metrics_creation(self):
        """Test DiagnosisMetrics creation."""
        metrics = DiagnosisMetrics(
            trap_id="ST-001",
            trap_type="thermodynamic",
            condition="leaking",
            severity="moderate",
            confidence=0.87,
            energy_loss_kw=8.2,
            co2_kg=4500.0,
            diagnosis_duration_seconds=0.05,
            is_failure=True,
        )

        assert metrics.trap_id == "ST-001"
        assert metrics.confidence == 0.87
        assert metrics.is_failure is True

    def test_diagnosis_metrics_defaults(self):
        """Test DiagnosisMetrics default values."""
        metrics = DiagnosisMetrics(
            trap_id="ST-002",
            trap_type="thermostatic",
            condition="normal",
            severity="none",
            confidence=0.95,
            energy_loss_kw=0.0,
            co2_kg=0.0,
            diagnosis_duration_seconds=0.03,
            is_failure=False,
        )

        assert metrics.failure_mode is None
        assert metrics.facility == "default"
        assert metrics.modality == "multimodal"


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

class TestThreadSafety:
    """Test thread safety of metrics recording."""

    def test_concurrent_diagnosis_recording(self, exporter, sample_diagnosis_metrics):
        """Test concurrent access to record_diagnosis."""
        import threading

        def record():
            for _ in range(100):
                exporter.record_diagnosis(sample_diagnosis_metrics)

        threads = [threading.Thread(target=record) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not raise any errors
        metrics = exporter.get_metrics()
        assert b"trapcatcher_traps_analyzed_total" in metrics
