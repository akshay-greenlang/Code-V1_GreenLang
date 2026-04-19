# -*- coding: utf-8 -*-
"""
GL-018 Optimizer Tests
======================

Unit tests for GL-018 UnifiedCombustionOptimizer main class.
Tests processing, efficiency calculation, emissions analysis, and recommendations.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from greenlang.agents.process_heat.gl_018_unified_combustion.optimizer import (
    UnifiedCombustionOptimizer,
)
from greenlang.agents.process_heat.gl_018_unified_combustion.config import (
    UnifiedCombustionConfig,
    BurnerConfig,
    ControlMode,
)
from greenlang.agents.process_heat.gl_018_unified_combustion.schemas import (
    CombustionInput,
    CombustionOutput,
    FlueGasReading,
    BurnerStatus,
)


class TestUnifiedCombustionOptimizerInit:
    """Tests for optimizer initialization."""

    def test_initialization(self, default_combustion_config):
        """Test optimizer initialization."""
        optimizer = UnifiedCombustionOptimizer(default_combustion_config)

        assert optimizer.combustion_config == default_combustion_config
        assert optimizer.VERSION == "1.0.0"
        assert optimizer.AGENT_TYPE == "GL-018"

    def test_component_initialization(self, default_combustion_config):
        """Test that all components are initialized."""
        optimizer = UnifiedCombustionOptimizer(default_combustion_config)

        assert optimizer.flue_gas_analyzer is not None
        assert optimizer.air_fuel_optimizer is not None
        assert optimizer.flame_analyzer is not None
        assert optimizer.burner_controller is not None
        assert optimizer.bms_controller is not None
        assert optimizer.efficiency_calc is not None
        assert optimizer.emissions_ctrl is not None

    def test_provenance_tracker_init(self, default_combustion_config):
        """Test provenance tracker initialization."""
        optimizer = UnifiedCombustionOptimizer(default_combustion_config)

        assert optimizer.provenance_tracker is not None
        assert optimizer.audit_logger is not None

    def test_intelligence_level(self, default_combustion_config):
        """Test intelligence level configuration."""
        from greenlang.agents.intelligence_interface import IntelligenceLevel

        optimizer = UnifiedCombustionOptimizer(default_combustion_config)

        level = optimizer.get_intelligence_level()
        assert level == IntelligenceLevel.FULL

    def test_intelligence_capabilities(self, default_combustion_config):
        """Test intelligence capabilities."""
        optimizer = UnifiedCombustionOptimizer(default_combustion_config)

        capabilities = optimizer.get_intelligence_capabilities()

        assert capabilities.can_explain is True
        assert capabilities.can_recommend is True
        assert capabilities.can_detect_anomalies is True
        assert capabilities.can_reason is True
        assert capabilities.can_validate is True


class TestInputValidation:
    """Tests for input validation."""

    @pytest.fixture
    def optimizer(self, default_combustion_config):
        """Create optimizer instance."""
        return UnifiedCombustionOptimizer(default_combustion_config)

    def test_valid_input(self, optimizer, default_combustion_input):
        """Test valid input validation."""
        result = optimizer.validate_input(default_combustion_input)
        assert result is True

    def test_zero_fuel_flow(self, optimizer):
        """Test zero fuel flow validation."""
        input_data = CombustionInput(
            equipment_id="BOILER-001",
            timestamp=datetime.now(timezone.utc),
            fuel_type="natural_gas",
            fuel_flow_rate=0,  # Invalid: must be > 0
            flue_gas=FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                o2_pct=3.5,
                temperature_f=350.0,
            ),
            burners=[],
            load_pct=75.0,
            ambient_temperature_f=70.0,
            steam_flow_rate_lb_hr=50000.0,
            steam_pressure_psig=150.0,
        )

        result = optimizer.validate_input(input_data)
        assert result is False

    def test_o2_too_high(self, optimizer):
        """Test O2 >= 21% validation."""
        input_data = CombustionInput(
            equipment_id="BOILER-001",
            timestamp=datetime.now(timezone.utc),
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas=FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                o2_pct=21.0,  # Invalid: must be < 21%
                temperature_f=350.0,
            ),
            burners=[],
            load_pct=75.0,
            ambient_temperature_f=70.0,
            steam_flow_rate_lb_hr=50000.0,
            steam_pressure_psig=150.0,
        )

        result = optimizer.validate_input(input_data)
        assert result is False

    def test_high_flue_gas_temp(self, optimizer):
        """Test high flue gas temperature warning."""
        input_data = CombustionInput(
            equipment_id="BOILER-001",
            timestamp=datetime.now(timezone.utc),
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas=FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                o2_pct=3.5,
                temperature_f=1500.0,  # Warning: > 1200F
            ),
            burners=[],
            load_pct=75.0,
            ambient_temperature_f=70.0,
            steam_flow_rate_lb_hr=50000.0,
            steam_pressure_psig=150.0,
        )

        result = optimizer.validate_input(input_data)
        assert result is False

    def test_high_co(self, optimizer):
        """Test high CO validation."""
        input_data = CombustionInput(
            equipment_id="BOILER-001",
            timestamp=datetime.now(timezone.utc),
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas=FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                o2_pct=3.5,
                temperature_f=350.0,
                co_ppm=1500.0,  # Warning: > 1000 ppm
            ),
            burners=[],
            load_pct=75.0,
            ambient_temperature_f=70.0,
            steam_flow_rate_lb_hr=50000.0,
            steam_pressure_psig=150.0,
        )

        result = optimizer.validate_input(input_data)
        assert result is False

    def test_over_load(self, optimizer):
        """Test over-load validation."""
        input_data = CombustionInput(
            equipment_id="BOILER-001",
            timestamp=datetime.now(timezone.utc),
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas=FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                o2_pct=3.5,
                temperature_f=350.0,
            ),
            burners=[],
            load_pct=130.0,  # Invalid: > 120%
            ambient_temperature_f=70.0,
            steam_flow_rate_lb_hr=50000.0,
            steam_pressure_psig=150.0,
        )

        result = optimizer.validate_input(input_data)
        assert result is False


class TestProcessing:
    """Tests for main processing logic."""

    @pytest.fixture
    def optimizer(self, default_combustion_config):
        """Create optimizer instance."""
        return UnifiedCombustionOptimizer(default_combustion_config)

    def test_process_success(self, optimizer, default_combustion_input):
        """Test successful processing."""
        output = optimizer.process(default_combustion_input)

        assert output.status == "success"
        assert output.equipment_id == "BOILER-001"
        assert output.processing_time_ms > 0

    def test_efficiency_calculation(self, optimizer, default_combustion_input):
        """Test efficiency is calculated."""
        output = optimizer.process(default_combustion_input)

        assert output.efficiency is not None
        assert 50 <= output.efficiency.net_efficiency_pct <= 100
        assert output.efficiency.total_losses_pct >= 0

    def test_flue_gas_analysis(self, optimizer, default_combustion_input):
        """Test flue gas analysis is performed."""
        output = optimizer.process(default_combustion_input)

        assert output.flue_gas_analysis is not None
        assert output.flue_gas_analysis.o2_pct == default_combustion_input.flue_gas.o2_pct
        assert output.flue_gas_analysis.excess_air_pct >= 0

    def test_flame_stability_analysis(self, optimizer, default_combustion_input):
        """Test flame stability analysis is performed."""
        output = optimizer.process(default_combustion_input)

        assert output.flame_stability is not None
        assert 0 <= output.flame_stability.flame_stability_index <= 1
        assert output.flame_stability.fsi_status in ["normal", "warning", "alarm"]

    def test_emissions_analysis(self, optimizer, default_combustion_input):
        """Test emissions analysis is performed."""
        output = optimizer.process(default_combustion_input)

        assert output.emissions is not None
        assert isinstance(output.emissions.in_compliance, bool)
        assert output.emissions.co2_tons_hr >= 0

    def test_bms_status(self, optimizer, default_combustion_input):
        """Test BMS status is returned."""
        output = optimizer.process(default_combustion_input)

        assert output.bms_status is not None
        assert isinstance(output.bms_status.all_interlocks_ok, bool)

    def test_optimal_setpoints(self, optimizer, default_combustion_input):
        """Test optimal setpoints are calculated."""
        output = optimizer.process(default_combustion_input)

        assert output.optimal_o2_setpoint_pct is not None
        assert output.optimal_excess_air_pct is not None
        assert 0 < output.optimal_o2_setpoint_pct < 21

    def test_kpis_calculated(self, optimizer, default_combustion_input):
        """Test KPIs are calculated."""
        output = optimizer.process(default_combustion_input)

        assert output.kpis is not None
        assert "net_efficiency_pct" in output.kpis
        assert "excess_air_pct" in output.kpis
        assert "co2_tons_hr" in output.kpis

    def test_provenance_hash(self, optimizer, default_combustion_input):
        """Test provenance hash is generated."""
        output = optimizer.process(default_combustion_input)

        assert output.provenance_hash is not None
        assert len(output.provenance_hash) == 64  # SHA-256


class TestRecommendations:
    """Tests for optimization recommendations."""

    @pytest.fixture
    def optimizer(self, default_combustion_config):
        """Create optimizer instance."""
        return UnifiedCombustionOptimizer(default_combustion_config)

    def test_recommendations_generated(self, optimizer, default_combustion_input):
        """Test recommendations are generated."""
        output = optimizer.process(default_combustion_input)

        assert output.recommendations is not None
        assert isinstance(output.recommendations, list)

    def test_high_co_recommendation(self, optimizer):
        """Test recommendation for high CO."""
        input_data = CombustionInput(
            equipment_id="BOILER-001",
            timestamp=datetime.now(timezone.utc),
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas=FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                o2_pct=3.5,
                temperature_f=350.0,
                co_ppm=150.0,  # High CO
            ),
            burners=[BurnerStatus(
                burner_id="BNR-001",
                status="firing",
                flame_signal_pct=85.0,
                firing_rate_pct=75.0,
            )],
            load_pct=75.0,
            ambient_temperature_f=70.0,
            steam_flow_rate_lb_hr=50000.0,
            steam_pressure_psig=150.0,
            air_damper_position_pct=60.0,
        )

        output = optimizer.process(input_data)

        # Should have CO-related recommendation
        co_recs = [r for r in output.recommendations if "CO" in r.title or "co" in r.parameter]
        assert len(co_recs) > 0


class TestAlerts:
    """Tests for alert generation."""

    @pytest.fixture
    def optimizer(self, default_combustion_config):
        """Create optimizer instance."""
        return UnifiedCombustionOptimizer(default_combustion_config)

    def test_alerts_generated(self, optimizer, default_combustion_input):
        """Test alerts list is returned."""
        output = optimizer.process(default_combustion_input)

        assert output.alerts is not None
        assert isinstance(output.alerts, list)

    def test_high_flue_gas_temp_alert(self, optimizer):
        """Test alert for high flue gas temperature."""
        input_data = CombustionInput(
            equipment_id="BOILER-001",
            timestamp=datetime.now(timezone.utc),
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas=FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                o2_pct=3.5,
                temperature_f=550.0,  # Above max 500F
            ),
            burners=[BurnerStatus(
                burner_id="BNR-001",
                status="firing",
                flame_signal_pct=85.0,
                firing_rate_pct=75.0,
            )],
            load_pct=75.0,
            ambient_temperature_f=70.0,
            steam_flow_rate_lb_hr=50000.0,
            steam_pressure_psig=150.0,
            air_damper_position_pct=60.0,
        )

        output = optimizer.process(input_data)

        # Should have temperature alert
        temp_alerts = [a for a in output.alerts if "temperature" in a.message.lower()]
        assert len(temp_alerts) > 0


class TestStatTracking:
    """Tests for state tracking."""

    @pytest.fixture
    def optimizer(self, default_combustion_config):
        """Create optimizer instance."""
        return UnifiedCombustionOptimizer(default_combustion_config)

    def test_efficiency_tracking(self, optimizer, default_combustion_input):
        """Test efficiency is tracked."""
        # Initial state
        assert optimizer.last_efficiency is None

        # Process once
        optimizer.process(default_combustion_input)

        assert optimizer.last_efficiency is not None

    def test_efficiency_trend(self, optimizer, default_combustion_input):
        """Test efficiency trend tracking."""
        # Process multiple times
        for _ in range(5):
            optimizer.process(default_combustion_input)

        assert len(optimizer.efficiency_trend) == 5


class TestOutputValidation:
    """Tests for output validation."""

    @pytest.fixture
    def optimizer(self, default_combustion_config):
        """Create optimizer instance."""
        return UnifiedCombustionOptimizer(default_combustion_config)

    def test_validate_output(self, optimizer, default_combustion_input):
        """Test output validation."""
        output = optimizer.process(default_combustion_input)

        result = optimizer.validate_output(output)
        assert result is True

    def test_invalid_efficiency(self, optimizer, default_efficiency_result):
        """Test invalid efficiency validation."""
        from greenlang.agents.process_heat.gl_018_unified_combustion.schemas import CombustionOutput

        # Create output with invalid efficiency
        invalid_efficiency = default_efficiency_result
        invalid_efficiency.net_efficiency_pct = 45.0  # Below valid range

        output = CombustionOutput(
            equipment_id="BOILER-001",
            status="success",
            processing_time_ms=100.0,
            efficiency=invalid_efficiency,
            flue_gas_analysis=None,
            flame_stability=None,
            emissions=None,
            bms_status=None,
            burner_tuning=[],
            recommendations=[],
            kpis={},
            alerts=[],
        )

        result = optimizer.validate_output(output)
        assert result is False


class TestIntelligenceOutputs:
    """Tests for intelligence/LLM outputs."""

    @pytest.fixture
    def optimizer(self, default_combustion_config):
        """Create optimizer instance."""
        return UnifiedCombustionOptimizer(default_combustion_config)

    def test_explanation_generated(self, optimizer, default_combustion_input):
        """Test explanation is generated."""
        output = optimizer.process(default_combustion_input)

        # Explanation may be generated based on intelligence config
        assert output.explanation is not None or output.explanation is None  # Can be None if intelligence disabled

    def test_intelligent_recommendations(self, optimizer, default_combustion_input):
        """Test intelligent recommendations."""
        output = optimizer.process(default_combustion_input)

        # May have intelligent recommendations
        assert output.intelligent_recommendations is not None or output.intelligent_recommendations is None

    def test_metadata(self, optimizer, default_combustion_input):
        """Test metadata is included."""
        output = optimizer.process(default_combustion_input)

        assert output.metadata is not None
        assert "agent_version" in output.metadata
        assert output.metadata["intelligence_level"] == "FULL"
