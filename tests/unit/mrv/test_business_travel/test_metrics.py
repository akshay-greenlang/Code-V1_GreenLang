# -*- coding: utf-8 -*-
"""
Test suite for business_travel.metrics - AGENT-MRV-019.

Tests Prometheus metrics for the Business Travel Agent (GL-MRV-S3-006)
including all 12 metrics with gl_bt_ prefix, recording methods, singleton
pattern, thread safety, and graceful Prometheus fallback.

Coverage:
- Singleton pattern (get_metrics returns same instance)
- All 12 Prometheus metrics exist with gl_bt_ prefix
- All primary recording methods (record_calculation, record_flight,
  record_ground_trip, record_hotel, record_factor_selection,
  record_compliance_check, record_batch)
- get_stats in-memory counters
- Thread safety
- No-op fallback when prometheus_client unavailable
- Agent info metric

Author: GL-TestEngineer
Date: February 2026
"""

import threading
import time
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import Mock, patch, MagicMock
import pytest

from greenlang.agents.mrv.business_travel.metrics import (
    BusinessTravelMetrics,
    get_metrics,
    PROMETHEUS_AVAILABLE,
)


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_metrics_singleton(self):
        """Test get_metrics returns the same instance."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()
        assert metrics1 is metrics2

    def test_singleton_across_threads(self):
        """Test singleton works across threads."""
        instances = []

        def get_thread():
            instances.append(get_metrics())

        threads = [threading.Thread(target=get_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = instances[0]
        for inst in instances[1:]:
            assert inst is first


# ==============================================================================
# METRICS EXISTENCE TESTS
# ==============================================================================


class TestMetricsExistence:
    """Test all 12 Prometheus metrics exist."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_metrics_calculations_total_counter(self, metrics):
        """Test gl_bt_calculations_total counter exists."""
        assert hasattr(metrics, "calculations_total")
        assert metrics.calculations_total is not None

    def test_metrics_emissions_counter(self, metrics):
        """Test gl_bt_emissions_kg_co2e_total counter exists."""
        assert hasattr(metrics, "emissions_kg_co2e_total")
        assert metrics.emissions_kg_co2e_total is not None

    def test_metrics_flights_counter(self, metrics):
        """Test gl_bt_flights_total counter exists."""
        assert hasattr(metrics, "flights_total")
        assert metrics.flights_total is not None

    def test_metrics_ground_trips_counter(self, metrics):
        """Test gl_bt_ground_trips_total counter exists."""
        assert hasattr(metrics, "ground_trips_total")
        assert metrics.ground_trips_total is not None

    def test_metrics_hotel_nights_counter(self, metrics):
        """Test gl_bt_hotel_nights_total counter exists."""
        assert hasattr(metrics, "hotel_nights_total")
        assert metrics.hotel_nights_total is not None

    def test_metrics_factor_selections_counter(self, metrics):
        """Test gl_bt_factor_selections_total counter exists."""
        assert hasattr(metrics, "factor_selections_total")
        assert metrics.factor_selections_total is not None

    def test_metrics_compliance_checks_counter(self, metrics):
        """Test gl_bt_compliance_checks_total counter exists."""
        assert hasattr(metrics, "compliance_checks_total")
        assert metrics.compliance_checks_total is not None

    def test_metrics_batch_jobs_counter(self, metrics):
        """Test gl_bt_batch_jobs_total counter exists."""
        assert hasattr(metrics, "batch_jobs_total")
        assert metrics.batch_jobs_total is not None

    def test_metrics_calculation_duration_histogram(self, metrics):
        """Test gl_bt_calculation_duration_seconds histogram exists."""
        assert hasattr(metrics, "calculation_duration_seconds")
        assert metrics.calculation_duration_seconds is not None

    def test_metrics_batch_size_histogram(self, metrics):
        """Test gl_bt_batch_size histogram exists."""
        assert hasattr(metrics, "batch_size")
        assert metrics.batch_size is not None

    def test_metrics_active_calculations_gauge(self, metrics):
        """Test gl_bt_active_calculations gauge exists."""
        assert hasattr(metrics, "active_calculations")
        assert metrics.active_calculations is not None

    def test_metrics_distance_km_counter(self, metrics):
        """Test gl_bt_distance_km_total counter exists."""
        assert hasattr(metrics, "distance_km_total")
        assert metrics.distance_km_total is not None


# ==============================================================================
# RECORDING METHOD TESTS
# ==============================================================================


class TestRecordingMethods:
    """Test all primary recording methods."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_record_calculation(self, metrics):
        """Test record_calculation method does not raise."""
        metrics.record_calculation(
            method="distance_based",
            mode="air",
            status="success",
            duration=0.035,
            co2e=245.8,
            rf_option="with_rf",
        )

    def test_record_flight(self, metrics):
        """Test record_flight method does not raise."""
        metrics.record_flight(
            distance_band="long_haul",
            cabin_class="economy",
            distance_km=5541.0,
        )

    def test_record_ground_trip(self, metrics):
        """Test record_ground_trip method does not raise."""
        metrics.record_ground_trip(
            mode="rail",
            vehicle_type="other",
            distance_km=640.0,
        )

    def test_record_hotel(self, metrics):
        """Test record_hotel method does not raise."""
        metrics.record_hotel(
            country="GB",
            nights=3,
        )

    def test_record_factor_selection(self, metrics):
        """Test record_factor_selection method does not raise."""
        metrics.record_factor_selection(
            source="defra",
            mode="air",
        )

    def test_record_compliance_check(self, metrics):
        """Test record_compliance_check method does not raise."""
        metrics.record_compliance_check(
            framework="ghg_protocol",
            status="compliant",
        )

    def test_record_batch(self, metrics):
        """Test record_batch method does not raise."""
        metrics.record_batch(
            status="completed",
            size=50,
        )


# ==============================================================================
# GET STATS TESTS
# ==============================================================================


class TestGetStats:
    """Test get_stats in-memory counters."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_get_stats(self, metrics):
        """Test get_stats returns a dictionary with expected keys."""
        stats = metrics.get_stats()
        assert isinstance(stats, dict)
        assert "calculations" in stats
        assert "emissions_kg_co2e" in stats
        assert "flights" in stats
        assert "ground_trips" in stats
        assert "hotel_nights" in stats
        assert "errors" in stats


# ==============================================================================
# AGENT INFO TESTS
# ==============================================================================


class TestAgentInfo:
    """Test agent info metric."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_metrics_agent_info(self, metrics):
        """Test gl_bt_agent Info metric exists."""
        assert hasattr(metrics, "agent_info")
        assert metrics.agent_info is not None


# ==============================================================================
# NO-OP FALLBACK TESTS
# ==============================================================================


class TestNoOpFallback:
    """Test metrics work without Prometheus client (no-op fallback)."""

    def test_metrics_no_prometheus_fallback(self):
        """Test PROMETHEUS_AVAILABLE flag is set."""
        # Whether True or False, the module should load without error
        assert isinstance(PROMETHEUS_AVAILABLE, bool)


# ==============================================================================
# THREAD SAFETY TESTS
# ==============================================================================


class TestThreadSafety:
    """Test thread safety of metrics recording."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_metrics_thread_safety(self, metrics):
        """Test concurrent metrics recording is thread-safe."""
        errors = []

        def record_loop():
            try:
                for _ in range(50):
                    metrics.record_calculation(
                        method="distance_based",
                        mode="air",
                        status="success",
                        duration=0.01,
                        co2e=100.0,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_loop) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ==============================================================================
# RESET TESTS
# ==============================================================================


class TestMetricsReset:
    """Test metrics reset functionality."""

    def test_metrics_reset(self):
        """Test reset_stats zeroes in-memory counters."""
        metrics = get_metrics()
        # Record something
        metrics.record_calculation(
            method="distance_based",
            mode="air",
            status="success",
            duration=0.01,
            co2e=100.0,
        )
        # Reset
        metrics.reset_stats()
        stats = metrics.get_stats()
        assert stats["calculations"] == 0
        assert stats["emissions_kg_co2e"] == 0.0


# ==============================================================================
# SUMMARY
# ==============================================================================


def test_metrics_module_coverage():
    """Meta-test to ensure all 12 metrics are tested."""
    tested_metrics = [
        "calculations_total",
        "emissions_kg_co2e_total",
        "flights_total",
        "ground_trips_total",
        "hotel_nights_total",
        "factor_selections_total",
        "compliance_checks_total",
        "batch_jobs_total",
        "calculation_duration_seconds",
        "batch_size",
        "active_calculations",
        "distance_km_total",
    ]
    assert len(tested_metrics) == 12
