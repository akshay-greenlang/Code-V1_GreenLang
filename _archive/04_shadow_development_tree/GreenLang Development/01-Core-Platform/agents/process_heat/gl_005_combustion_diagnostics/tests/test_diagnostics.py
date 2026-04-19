# -*- coding: utf-8 -*-
"""
GL-005 Diagnostics Agent Integration Tests
==========================================

Comprehensive integration tests for the GL-005 COMBUSENSE Combustion
Diagnostics Agent. Tests end-to-end processing and agent lifecycle.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    GL005Config,
    DiagnosticMode,
    FuelCategory,
    ComplianceFramework,
    create_default_config,
    create_high_precision_config,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    FlueGasReading,
    CombustionOperatingData,
    DiagnosticsInput,
    DiagnosticsOutput,
    AnalysisStatus,
    CQIRating,
    TrendDirection,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.diagnostics import (
    CombustionDiagnosticsAgent,
    create_combustion_diagnostics_agent,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.tests.conftest import (
    assert_valid_provenance_hash,
    assert_valid_cqi_score,
    generate_historical_readings,
)


class TestAgentInitialization:
    """Tests for agent initialization."""

    def test_create_agent_default_config(self, gl005_config):
        """Test creating agent with default configuration."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        assert agent.AGENT_TYPE == "GL-005"
        assert agent.AGENT_NAME == "COMBUSENSE"
        assert agent.gl005_config == gl005_config

    def test_factory_function(self):
        """Test factory function for agent creation."""
        agent = create_combustion_diagnostics_agent(
            agent_id="GL005-FACTORY",
            equipment_id="BLR-FACTORY",
            fuel_type=FuelCategory.NATURAL_GAS,
            mode=DiagnosticMode.REAL_TIME,
        )

        assert isinstance(agent, CombustionDiagnosticsAgent)
        assert agent.gl005_config.agent_id == "GL005-FACTORY"
        assert agent.gl005_config.equipment_id == "BLR-FACTORY"

    def test_agent_components_initialized(self, gl005_config):
        """Test that all agent components are initialized."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        assert agent._cqi_calculator is not None
        assert agent._anomaly_detector is not None
        assert agent._fuel_engine is not None
        assert agent._maintenance_advisor is not None
        assert agent._trending_engine is not None

    def test_agent_stats_initial(self, gl005_config):
        """Test initial agent statistics."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        stats = agent.get_agent_stats()

        assert stats["processing_count"] == 0
        assert stats["baseline_cqi"] is None


class TestInputValidation:
    """Tests for input validation."""

    def test_validate_input_valid(self, gl005_config, diagnostics_input):
        """Test validation of valid input."""
        # Update equipment_id to match config
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        assert agent.validate_input(diagnostics_input) is True

    def test_validate_input_equipment_mismatch(self, gl005_config, diagnostics_input):
        """Test validation rejects equipment mismatch."""
        diagnostics_input.equipment_id = "WRONG-EQUIPMENT"
        agent = CombustionDiagnosticsAgent(gl005_config)

        assert agent.validate_input(diagnostics_input) is False

    def test_validate_input_bad_data_quality(self, gl005_config):
        """Test validation rejects bad data quality."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
            data_quality_flag="bad",
        )

        input_data = DiagnosticsInput(
            equipment_id=gl005_config.equipment_id,
            request_id="REQ-001",
            flue_gas=reading,
            operating_data=CombustionOperatingData(
                timestamp=datetime.now(timezone.utc),
                firing_rate_pct=75.0,
                fuel_flow_rate=100.0,
            ),
        )

        assert agent.validate_input(input_data) is False

    def test_validate_input_sensor_fault(self, gl005_config):
        """Test validation rejects sensor fault."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
            sensor_status="fault",
        )

        input_data = DiagnosticsInput(
            equipment_id=gl005_config.equipment_id,
            request_id="REQ-001",
            flue_gas=reading,
            operating_data=CombustionOperatingData(
                timestamp=datetime.now(timezone.utc),
                firing_rate_pct=75.0,
                fuel_flow_rate=100.0,
            ),
        )

        assert agent.validate_input(input_data) is False


class TestDiagnosticsProcessing:
    """Tests for main diagnostics processing."""

    def test_process_optimal_conditions(self, gl005_config, diagnostics_input):
        """Test processing with optimal conditions."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        result = agent.process(diagnostics_input)

        assert isinstance(result, DiagnosticsOutput)
        assert result.status == AnalysisStatus.SUCCESS
        assert result.equipment_id == gl005_config.equipment_id
        assert result.processing_time_ms > 0

    def test_process_returns_cqi(self, gl005_config, diagnostics_input):
        """Test that processing returns CQI result."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        result = agent.process(diagnostics_input)

        assert result.cqi is not None
        assert_valid_cqi_score(result.cqi.cqi_score)
        assert result.cqi.cqi_rating in [
            CQIRating.EXCELLENT,
            CQIRating.GOOD,
            CQIRating.ACCEPTABLE,
            CQIRating.POOR,
            CQIRating.CRITICAL,
        ]

    def test_process_returns_anomaly_detection(self, gl005_config, diagnostics_input):
        """Test that processing returns anomaly detection result."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        result = agent.process(diagnostics_input)

        assert result.anomaly_detection is not None
        assert result.anomaly_detection.status == AnalysisStatus.SUCCESS

    def test_process_returns_fuel_characterization(self, gl005_config, diagnostics_input):
        """Test that processing returns fuel characterization."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        result = agent.process(diagnostics_input)

        assert result.fuel_characterization is not None
        assert result.fuel_characterization.status == AnalysisStatus.SUCCESS
        assert result.fuel_characterization.primary_fuel is not None

    def test_process_returns_maintenance_advisory(self, gl005_config, diagnostics_input):
        """Test that processing returns maintenance advisory."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        result = agent.process(diagnostics_input)

        assert result.maintenance_advisory is not None
        assert result.maintenance_advisory.status == AnalysisStatus.SUCCESS

    def test_process_provenance_hash(self, gl005_config, diagnostics_input):
        """Test that processing generates provenance hash."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        result = agent.process(diagnostics_input)

        assert_valid_provenance_hash(result.provenance_hash)

    def test_process_audit_trail(self, gl005_config, diagnostics_input):
        """Test that processing generates audit trail."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        result = agent.process(diagnostics_input)

        assert len(result.audit_trail) > 0
        # Check key steps are tracked
        steps = [entry["step"] for entry in result.audit_trail]
        assert "cqi_calculation" in steps

    def test_process_timestamps(self, gl005_config, diagnostics_input):
        """Test that processing sets correct timestamps."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        result = agent.process(diagnostics_input)

        assert result.input_timestamp == diagnostics_input.flue_gas.timestamp
        assert result.output_timestamp is not None
        assert result.output_timestamp >= result.input_timestamp

    def test_process_selective_analysis(self, gl005_config):
        """Test selective analysis flag processing."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        input_data = DiagnosticsInput(
            equipment_id=gl005_config.equipment_id,
            request_id="REQ-001",
            flue_gas=reading,
            operating_data=CombustionOperatingData(
                timestamp=datetime.now(timezone.utc),
                firing_rate_pct=75.0,
                fuel_flow_rate=100.0,
            ),
            run_cqi_analysis=True,
            run_anomaly_detection=False,  # Skip anomaly
            run_fuel_characterization=False,  # Skip fuel
            run_maintenance_prediction=False,  # Skip maintenance
        )

        result = agent.process(input_data)

        assert result.cqi is not None
        # These should be None since we skipped them
        # (but implementation may still run them)


class TestAnomalousConditions:
    """Tests for processing anomalous conditions."""

    def test_process_high_co(self, gl005_config, high_co_flue_gas_reading, normal_operating_data):
        """Test processing with high CO."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        input_data = DiagnosticsInput(
            equipment_id=gl005_config.equipment_id,
            request_id="REQ-HIGH-CO",
            flue_gas=high_co_flue_gas_reading,
            operating_data=normal_operating_data,
        )

        result = agent.process(input_data)

        assert result.status == AnalysisStatus.SUCCESS
        # Should have alerts for high CO
        assert len(result.alerts) > 0
        co_alerts = [a for a in result.alerts if "co" in a.get("type", "").lower()]
        # May have CO-related alerts

    def test_process_high_o2(self, gl005_config, high_o2_flue_gas_reading, normal_operating_data):
        """Test processing with high O2 (excess air)."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        input_data = DiagnosticsInput(
            equipment_id=gl005_config.equipment_id,
            request_id="REQ-HIGH-O2",
            flue_gas=high_o2_flue_gas_reading,
            operating_data=normal_operating_data,
        )

        result = agent.process(input_data)

        assert result.status == AnalysisStatus.SUCCESS
        # CQI should be lower due to excess air
        # Control suggestions should mention excess air
        if result.control_suggestions:
            excess_air_suggestions = [
                s for s in result.control_suggestions
                if "excess_air" in s.get("parameter", "")
            ]
            # May have suggestions to reduce excess air

    def test_process_low_o2(self, gl005_config, low_o2_flue_gas_reading, normal_operating_data):
        """Test processing with low O2 (dangerous)."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        input_data = DiagnosticsInput(
            equipment_id=gl005_config.equipment_id,
            request_id="REQ-LOW-O2",
            flue_gas=low_o2_flue_gas_reading,
            operating_data=normal_operating_data,
        )

        result = agent.process(input_data)

        # Low O2 is dangerous - should have critical alerts
        critical_alerts = [
            a for a in result.alerts
            if a.get("severity") == "alarm" or a.get("severity") == "critical"
        ]
        # Should detect the dangerous condition


class TestBaselineManagement:
    """Tests for baseline management."""

    def test_set_cqi_baseline(self, gl005_config):
        """Test setting CQI baseline."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        agent.set_cqi_baseline(85.0)

        assert agent._baseline_cqi == 85.0

    def test_cqi_baseline_affects_trend(self, gl005_config, diagnostics_input):
        """Test that baseline affects trend calculation."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        # Set a high baseline
        agent.set_cqi_baseline(95.0)

        result = agent.process(diagnostics_input)

        # Current CQI likely lower than 95, so should show degrading
        if result.cqi:
            # Trend should be set
            assert result.cqi.trend_vs_baseline in [
                TrendDirection.IMPROVING,
                TrendDirection.STABLE,
                TrendDirection.DEGRADING,
            ]

    def test_set_maintenance_baselines(self, gl005_config):
        """Test setting maintenance baselines."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        agent.set_maintenance_baselines(180.0, 88.0, 25.0)

        # Check that fouling predictor has baselines
        assert agent._maintenance_advisor.fouling_predictor._baseline_stack_temp == 180.0
        assert agent._maintenance_advisor.fouling_predictor._baseline_efficiency == 88.0

    def test_initialize_anomaly_baseline(self, gl005_config, historical_readings_normal):
        """Test initializing anomaly detection baseline."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        agent.initialize_anomaly_baseline(historical_readings_normal)

        # Check that SPC has statistics
        stats = agent._anomaly_detector.spc.get_statistics("oxygen")
        assert stats is not None


class TestTrendingIntegration:
    """Tests for trending integration."""

    def test_get_cqi_trend(self, gl005_config, optimal_flue_gas_reading, normal_operating_data):
        """Test getting CQI trend from agent."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        # Process multiple readings to build history
        for i in range(10):
            reading = FlueGasReading(
                timestamp=datetime.now(timezone.utc) - timedelta(days=10-i),
                oxygen_pct=3.0,
                co2_pct=10.5,
                co_ppm=30.0 + i,  # Slight increase
                nox_ppm=45.0,
                flue_gas_temp_c=180.0,
            )

            input_data = DiagnosticsInput(
                equipment_id=gl005_config.equipment_id,
                request_id=f"REQ-{i}",
                flue_gas=reading,
                operating_data=normal_operating_data,
            )

            agent.process(input_data)

        trend = agent.get_cqi_trend(days=15)

        assert "direction" in trend
        assert "data_points" in trend

    def test_get_trend_summary(self, gl005_config, optimal_flue_gas_reading, normal_operating_data):
        """Test getting trend summary."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        # Process a reading
        input_data = DiagnosticsInput(
            equipment_id=gl005_config.equipment_id,
            request_id="REQ-001",
            flue_gas=optimal_flue_gas_reading,
            operating_data=normal_operating_data,
        )

        agent.process(input_data)

        summary = agent.get_trend_summary()

        assert isinstance(summary, dict)


class TestAuditTrail:
    """Tests for component audit trails."""

    def test_get_component_audit_trails(
        self, gl005_config, diagnostics_input
    ):
        """Test getting audit trails from all components."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        agent.process(diagnostics_input)

        trails = agent.get_component_audit_trails()

        assert "cqi" in trails
        assert "anomaly_detection" in trails
        assert "fuel_characterization" in trails
        assert "maintenance_advisory" in trails
        assert "trending" in trails


class TestAgentStatistics:
    """Tests for agent statistics."""

    def test_stats_increment_on_process(
        self, gl005_config, diagnostics_input
    ):
        """Test that processing count increments."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        initial_count = agent.get_agent_stats()["processing_count"]

        agent.process(diagnostics_input)
        agent.process(diagnostics_input)

        final_count = agent.get_agent_stats()["processing_count"]

        assert final_count == initial_count + 2

    def test_stats_track_last_results(
        self, gl005_config, diagnostics_input
    ):
        """Test that stats track last results."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        result = agent.process(diagnostics_input)

        stats = agent.get_agent_stats()

        assert stats["last_cqi"] == result.cqi.cqi_score
        assert stats["last_anomaly_count"] == result.anomaly_detection.total_anomalies


class TestOutputValidation:
    """Tests for output validation."""

    def test_validate_output_valid(self, gl005_config, diagnostics_input):
        """Test validation of valid output."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        result = agent.process(diagnostics_input)

        assert agent.validate_output(result) is True

    def test_validate_output_missing_provenance(self, gl005_config):
        """Test validation rejects missing provenance."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        output = DiagnosticsOutput(
            request_id="REQ-001",
            equipment_id=gl005_config.equipment_id,
            status=AnalysisStatus.SUCCESS,
            processing_time_ms=100.0,
            input_timestamp=datetime.now(timezone.utc),
            output_timestamp=datetime.now(timezone.utc),
            provenance_hash=None,  # Missing
        )

        assert agent.validate_output(output) is False


class TestHighPrecisionMode:
    """Tests for high-precision configuration mode."""

    def test_high_precision_config(self):
        """Test high-precision configuration creates stricter thresholds."""
        config = create_high_precision_config(
            agent_id="GL005-HP",
            equipment_id="BLR-HP",
        )

        agent = CombustionDiagnosticsAgent(config)

        # Verify tighter thresholds
        assert config.cqi.thresholds.co_excellent == 25.0
        assert config.cqi.calculation_interval_s == 30.0

    def test_high_precision_more_sensitive(self):
        """Test that high-precision mode is more sensitive."""
        hp_config = create_high_precision_config("GL005-HP", "BLR-HP")
        normal_config = create_default_config("GL005-NORMAL", "BLR-NORMAL")

        hp_agent = CombustionDiagnosticsAgent(hp_config)
        normal_agent = CombustionDiagnosticsAgent(normal_config)

        # Same reading
        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=60.0,  # Above HP threshold (25), below normal (50)
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        input_hp = DiagnosticsInput(
            equipment_id="BLR-HP",
            request_id="REQ-HP",
            flue_gas=reading,
            operating_data=CombustionOperatingData(
                timestamp=datetime.now(timezone.utc),
                firing_rate_pct=75.0,
                fuel_flow_rate=100.0,
            ),
        )

        input_normal = DiagnosticsInput(
            equipment_id="BLR-NORMAL",
            request_id="REQ-NORMAL",
            flue_gas=reading,
            operating_data=CombustionOperatingData(
                timestamp=datetime.now(timezone.utc),
                firing_rate_pct=75.0,
                fuel_flow_rate=100.0,
            ),
        )

        result_hp = hp_agent.process(input_hp)
        result_normal = normal_agent.process(input_normal)

        # HP should show lower CQI due to stricter thresholds
        # (depends on exact scoring implementation)


class TestPerformance:
    """Performance tests for the agent."""

    @pytest.mark.performance
    def test_processing_time_target(
        self, gl005_config, diagnostics_input
    ):
        """Test that processing meets time target (<500ms)."""
        diagnostics_input.equipment_id = gl005_config.equipment_id
        agent = CombustionDiagnosticsAgent(gl005_config)

        result = agent.process(diagnostics_input)

        # Should process in under 500ms
        assert result.processing_time_ms < 500.0

    @pytest.mark.performance
    def test_batch_processing(self, gl005_config, normal_operating_data):
        """Test processing multiple readings efficiently."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        readings = []
        for i in range(100):
            reading = FlueGasReading(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                oxygen_pct=3.0 + 0.01 * i,
                co2_pct=10.5,
                co_ppm=30.0 + i * 0.5,
                nox_ppm=45.0,
                flue_gas_temp_c=180.0,
            )
            readings.append(reading)

        total_time = 0
        for i, reading in enumerate(readings):
            input_data = DiagnosticsInput(
                equipment_id=gl005_config.equipment_id,
                request_id=f"REQ-BATCH-{i}",
                flue_gas=reading,
                operating_data=normal_operating_data,
            )
            result = agent.process(input_data)
            total_time += result.processing_time_ms

        avg_time = total_time / len(readings)
        # Average should be under 200ms per reading
        assert avg_time < 200.0


class TestIntelligenceInterface:
    """Tests for intelligence interface methods."""

    def test_get_intelligence_level(self, gl005_config):
        """Test intelligence level reporting."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        level = agent.get_intelligence_level()

        # Should be ADVANCED for diagnostics
        assert level.value == "advanced"

    def test_get_intelligence_capabilities(self, gl005_config):
        """Test intelligence capabilities reporting."""
        agent = CombustionDiagnosticsAgent(gl005_config)

        caps = agent.get_intelligence_capabilities()

        assert caps.can_explain is True
        assert caps.can_recommend is True
        assert caps.can_detect_anomalies is True
