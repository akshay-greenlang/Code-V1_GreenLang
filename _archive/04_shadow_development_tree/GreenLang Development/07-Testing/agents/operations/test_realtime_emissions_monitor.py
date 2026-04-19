# -*- coding: utf-8 -*-
"""
Tests for GL-OPS-X-001: Real-time Emissions Monitor

Tests cover:
    - Emissions reading recording
    - Data aggregation by facility and gas type
    - Trend analysis
    - Threshold management and breach detection
    - Provenance tracking

Author: GreenLang Team
"""

import pytest
from datetime import datetime, timedelta

from greenlang.agents.base import AgentConfig, AgentResult
from greenlang.agents.operations.realtime_emissions_monitor import (
    RealtimeEmissionsMonitor,
    EmissionsMonitorInput,
    EmissionsMonitorOutput,
    EmissionsReading,
    AggregatedEmissions,
    EmissionsThreshold,
    AggregationPeriod,
    MonitoringStatus,
    EmissionsSource,
    GasType,
)
from greenlang.utilities.determinism import DeterministicClock


class TestRealtimeEmissionsMonitorInitialization:
    """Tests for agent initialization."""

    def test_agent_creation_default_config(self):
        """Test creating agent with default configuration."""
        agent = RealtimeEmissionsMonitor()

        assert agent.AGENT_ID == "GL-OPS-X-001"
        assert agent.AGENT_NAME == "Real-time Emissions Monitor"
        assert agent.VERSION == "1.0.0"

    def test_agent_creation_custom_config(self):
        """Test creating agent with custom configuration."""
        config = AgentConfig(
            name="Custom Emissions Monitor",
            description="Custom config test",
            version="2.0.0",
        )
        agent = RealtimeEmissionsMonitor(config)

        assert agent.config.name == "Custom Emissions Monitor"


class TestEmissionsRecording:
    """Tests for emissions recording functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return RealtimeEmissionsMonitor()

    def test_record_emissions_reading(self, agent):
        """Test recording an emissions reading."""
        result = agent.run({
            "operation": "record_reading",
            "reading": {
                "facility_id": "FAC-001",
                "source": "natural_gas",
                "gas_type": "co2",
                "value_kg": 150.5,
                "timestamp": DeterministicClock.now().isoformat(),
            }
        })

        assert result.success
        assert result.data["data"]["recorded"]
        assert result.data["data"]["facility_id"] == "FAC-001"

    def test_record_multiple_readings(self, agent):
        """Test recording multiple emissions readings."""
        for i in range(5):
            result = agent.run({
                "operation": "record_reading",
                "reading": {
                    "facility_id": f"FAC-{i:03d}",
                    "source": "electricity",
                    "gas_type": "co2",
                    "value_kg": 100.0 + i * 10,
                }
            })
            assert result.success

        stats = agent.run({"operation": "get_statistics"})
        assert stats.data["data"]["total_readings"] == 5

    def test_record_reading_missing_facility_fails(self, agent):
        """Test that recording without facility_id fails."""
        result = agent.run({
            "operation": "record_reading",
            "reading": {
                "source": "natural_gas",
                "gas_type": "co2",
                "value_kg": 100.0,
            }
        })

        assert not result.success or "error" in result.data.get("data", {})


class TestEmissionsAggregation:
    """Tests for emissions aggregation functionality."""

    @pytest.fixture
    def agent_with_readings(self):
        """Create agent with pre-recorded emissions."""
        agent = RealtimeEmissionsMonitor()

        # Record readings for multiple facilities
        for i in range(10):
            agent.run({
                "operation": "record_reading",
                "reading": {
                    "facility_id": f"FAC-{i % 3 + 1:03d}",
                    "source": "electricity",
                    "gas_type": "co2",
                    "value_kg": 100.0 + i * 5,
                }
            })

        return agent

    def test_aggregate_by_facility(self, agent_with_readings):
        """Test aggregation by facility."""
        result = agent_with_readings.run({
            "operation": "aggregate",
            "facility_id": "FAC-001",
            "aggregation_period": "hourly",
        })

        assert result.success
        assert "aggregated_data" in result.data["data"] or "total" in str(result.data)

    def test_aggregate_all_facilities(self, agent_with_readings):
        """Test aggregation across all facilities."""
        result = agent_with_readings.run({
            "operation": "aggregate",
            "aggregation_period": "daily",
        })

        assert result.success

    def test_get_current_emissions(self, agent_with_readings):
        """Test getting current emissions status."""
        result = agent_with_readings.run({
            "operation": "get_current",
            "facility_id": "FAC-001",
        })

        assert result.success


class TestTrendAnalysis:
    """Tests for trend analysis functionality."""

    @pytest.fixture
    def agent_with_history(self):
        """Create agent with historical emissions data."""
        agent = RealtimeEmissionsMonitor()

        # Create trending data (increasing emissions)
        for i in range(20):
            agent.run({
                "operation": "record_reading",
                "reading": {
                    "facility_id": "FAC-001",
                    "source": "natural_gas",
                    "gas_type": "co2",
                    "value_kg": 100.0 + i * 5,  # Increasing trend
                }
            })

        return agent

    def test_get_trends(self, agent_with_history):
        """Test getting emission trends."""
        result = agent_with_history.run({
            "operation": "get_trends",
            "facility_id": "FAC-001",
        })

        assert result.success
        assert "trends" in result.data["data"] or "error" not in result.data["data"]


class TestThresholdManagement:
    """Tests for threshold management."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return RealtimeEmissionsMonitor()

    def test_add_threshold(self, agent):
        """Test adding a monitoring threshold."""
        result = agent.run({
            "operation": "add_threshold",
            "threshold": {
                "facility_id": "FAC-001",
                "gas_type": "co2",
                "max_hourly_kg": 500.0,
                "max_daily_kg": 10000.0,
            }
        })

        assert result.success
        assert result.data["data"].get("added") or "threshold_id" in result.data["data"]

    def test_check_threshold_no_breach(self, agent):
        """Test threshold check when within limits."""
        # Add threshold
        agent.run({
            "operation": "add_threshold",
            "threshold": {
                "facility_id": "FAC-001",
                "gas_type": "co2",
                "max_hourly_kg": 500.0,
            }
        })

        # Record emissions below threshold
        agent.run({
            "operation": "record_reading",
            "reading": {
                "facility_id": "FAC-001",
                "source": "electricity",
                "gas_type": "co2",
                "value_kg": 100.0,
            }
        })

        # Check thresholds
        result = agent.run({
            "operation": "check_thresholds",
            "facility_id": "FAC-001",
        })

        assert result.success


class TestProvenanceTracking:
    """Tests for provenance and audit trail."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return RealtimeEmissionsMonitor()

    def test_output_contains_provenance_hash(self, agent):
        """Test that all operations include provenance hash."""
        result = agent.run({
            "operation": "record_reading",
            "reading": {
                "facility_id": "FAC-001",
                "source": "natural_gas",
                "gas_type": "co2",
                "value_kg": 150.0,
            }
        })

        assert result.success
        assert "provenance_hash" in result.data
        assert len(result.data["provenance_hash"]) == 16

    def test_processing_time_tracked(self, agent):
        """Test that processing time is tracked."""
        result = agent.run({
            "operation": "get_statistics",
        })

        assert result.success
        assert "processing_time_ms" in result.data
        assert result.data["processing_time_ms"] >= 0


class TestInputValidation:
    """Tests for input validation."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return RealtimeEmissionsMonitor()

    def test_invalid_operation_fails(self, agent):
        """Test that invalid operation fails."""
        result = agent.run({"operation": "invalid_operation"})

        assert not result.success


class TestIntegration:
    """Integration tests."""

    def test_full_monitoring_workflow(self):
        """Test a complete monitoring workflow."""
        agent = RealtimeEmissionsMonitor()

        # 1. Add threshold
        agent.run({
            "operation": "add_threshold",
            "threshold": {
                "facility_id": "FAC-001",
                "gas_type": "co2",
                "max_hourly_kg": 1000.0,
            }
        })

        # 2. Record multiple readings
        for i in range(5):
            agent.run({
                "operation": "record_reading",
                "reading": {
                    "facility_id": "FAC-001",
                    "source": "natural_gas",
                    "gas_type": "co2",
                    "value_kg": 100.0 + i * 20,
                }
            })

        # 3. Check thresholds
        check_result = agent.run({
            "operation": "check_thresholds",
            "facility_id": "FAC-001",
        })
        assert check_result.success

        # 4. Get statistics
        stats_result = agent.run({"operation": "get_statistics"})
        assert stats_result.success
        assert stats_result.data["data"]["total_readings"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
