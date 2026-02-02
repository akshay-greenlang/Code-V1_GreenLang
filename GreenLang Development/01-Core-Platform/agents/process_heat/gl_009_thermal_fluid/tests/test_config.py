"""
Unit tests for GL-009 THERMALIQ Agent Configuration

Tests configuration schemas, factory functions, and default values.
"""

import pytest
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_009_thermal_fluid.config import (
    # Configuration classes
    TemperatureLimits,
    FlowLimits,
    PressureLimits,
    SafetyConfig,
    DegradationThresholds,
    HeaterConfig,
    ExpansionTankConfig,
    PumpConfig,
    PipingConfig,
    ExergyConfig,
    ThermalFluidConfig,
    # Factory functions
    create_default_config,
    create_high_temperature_config,
    create_food_grade_config,
    _get_temperature_limits_for_fluid,
)
from greenlang.agents.process_heat.gl_009_thermal_fluid.schemas import (
    ThermalFluidType,
    HeaterType,
)


# =============================================================================
# TEMPERATURE LIMITS TESTS
# =============================================================================

class TestTemperatureLimits:
    """Tests for TemperatureLimits configuration."""

    def test_default_values(self):
        """Test default temperature limit values."""
        limits = TemperatureLimits()

        assert limits.max_film_temp_f == 700.0
        assert limits.max_bulk_temp_f == 650.0
        assert limits.high_bulk_temp_alarm_f == 620.0
        assert limits.high_bulk_temp_trip_f == 640.0
        assert limits.low_bulk_temp_alarm_f == 200.0
        assert limits.min_flash_point_margin_f == 50.0
        assert limits.min_auto_ignition_margin_f == 100.0

    def test_custom_values(self):
        """Test custom temperature limit values."""
        limits = TemperatureLimits(
            max_film_temp_f=750.0,
            max_bulk_temp_f=700.0,
            high_bulk_temp_alarm_f=680.0,
            high_bulk_temp_trip_f=690.0,
        )

        assert limits.max_film_temp_f == 750.0
        assert limits.max_bulk_temp_f == 700.0

    def test_temperature_must_be_positive(self):
        """Test temperatures must be positive."""
        with pytest.raises(ValidationError):
            TemperatureLimits(max_film_temp_f=0)

        with pytest.raises(ValidationError):
            TemperatureLimits(max_bulk_temp_f=-10)

    def test_low_temp_alarm_can_be_zero(self):
        """Test low temp alarm can be zero."""
        limits = TemperatureLimits(low_bulk_temp_alarm_f=0)
        assert limits.low_bulk_temp_alarm_f == 0


class TestFlowLimits:
    """Tests for FlowLimits configuration."""

    def test_default_values(self):
        """Test default flow limit values."""
        limits = FlowLimits()

        assert limits.min_flow_pct == 25.0
        assert limits.low_flow_alarm_pct == 30.0
        assert limits.low_flow_trip_pct == 25.0
        assert limits.max_velocity_ft_s == 12.0

    def test_percentage_range(self):
        """Test percentage fields are within range."""
        limits = FlowLimits(min_flow_pct=0)
        assert limits.min_flow_pct == 0

        limits = FlowLimits(min_flow_pct=100)
        assert limits.min_flow_pct == 100

        with pytest.raises(ValidationError):
            FlowLimits(min_flow_pct=-1)

        with pytest.raises(ValidationError):
            FlowLimits(min_flow_pct=101)

    def test_velocity_must_be_positive(self):
        """Test max velocity must be positive."""
        with pytest.raises(ValidationError):
            FlowLimits(max_velocity_ft_s=0)


class TestPressureLimits:
    """Tests for PressureLimits configuration."""

    def test_default_values(self):
        """Test default pressure limit values."""
        limits = PressureLimits()

        assert limits.max_system_pressure_psig == 150.0
        assert limits.min_pump_suction_pressure_psig == 5.0
        assert limits.min_npsh_margin_ft == 3.0

    def test_pressure_must_be_positive(self):
        """Test max pressure must be positive."""
        with pytest.raises(ValidationError):
            PressureLimits(max_system_pressure_psig=0)

    def test_suction_pressure_vacuum_allowed(self):
        """Test suction pressure can be vacuum (negative)."""
        limits = PressureLimits(min_pump_suction_pressure_psig=-10)
        assert limits.min_pump_suction_pressure_psig == -10

        # But not below absolute vacuum
        with pytest.raises(ValidationError):
            PressureLimits(min_pump_suction_pressure_psig=-15)


# =============================================================================
# SAFETY CONFIG TESTS
# =============================================================================

class TestSafetyConfig:
    """Tests for SafetyConfig configuration."""

    def test_default_values(self):
        """Test default safety config values."""
        config = SafetyConfig()

        assert config.sil_level == 2
        assert config.emergency_shutdown_enabled == True
        assert config.watchdog_timeout_ms == 5000

    def test_sil_level_range(self):
        """Test SIL level range validation."""
        for sil in [0, 1, 2, 3, 4]:
            config = SafetyConfig(sil_level=sil)
            assert config.sil_level == sil

        with pytest.raises(ValidationError):
            SafetyConfig(sil_level=-1)

        with pytest.raises(ValidationError):
            SafetyConfig(sil_level=5)

    def test_watchdog_timeout_range(self):
        """Test watchdog timeout range."""
        config = SafetyConfig(watchdog_timeout_ms=100)
        assert config.watchdog_timeout_ms == 100

        config = SafetyConfig(watchdog_timeout_ms=60000)
        assert config.watchdog_timeout_ms == 60000

        with pytest.raises(ValidationError):
            SafetyConfig(watchdog_timeout_ms=99)

        with pytest.raises(ValidationError):
            SafetyConfig(watchdog_timeout_ms=60001)

    def test_sub_configs_default(self):
        """Test sub-configurations have defaults."""
        config = SafetyConfig()

        assert isinstance(config.temperature_limits, TemperatureLimits)
        assert isinstance(config.flow_limits, FlowLimits)
        assert isinstance(config.pressure_limits, PressureLimits)


# =============================================================================
# DEGRADATION THRESHOLDS TESTS
# =============================================================================

class TestDegradationThresholds:
    """Tests for DegradationThresholds configuration."""

    def test_default_values(self):
        """Test default degradation threshold values."""
        thresholds = DegradationThresholds()

        assert thresholds.viscosity_warning_pct == 10.0
        assert thresholds.viscosity_critical_pct == 25.0
        assert thresholds.flash_point_warning_drop_f == 30.0
        assert thresholds.flash_point_critical_drop_f == 50.0
        assert thresholds.acid_number_warning == 0.2
        assert thresholds.acid_number_critical == 0.5
        assert thresholds.sampling_interval_months == 6

    def test_thresholds_must_be_positive(self):
        """Test thresholds must be positive."""
        with pytest.raises(ValidationError):
            DegradationThresholds(viscosity_warning_pct=0)

        with pytest.raises(ValidationError):
            DegradationThresholds(flash_point_warning_drop_f=0)

    def test_acid_number_can_be_zero(self):
        """Test acid number can be zero."""
        thresholds = DegradationThresholds(acid_number_warning=0)
        assert thresholds.acid_number_warning == 0

    def test_sampling_interval_range(self):
        """Test sampling interval range."""
        thresholds = DegradationThresholds(sampling_interval_months=1)
        assert thresholds.sampling_interval_months == 1

        thresholds = DegradationThresholds(sampling_interval_months=24)
        assert thresholds.sampling_interval_months == 24

        with pytest.raises(ValidationError):
            DegradationThresholds(sampling_interval_months=0)

        with pytest.raises(ValidationError):
            DegradationThresholds(sampling_interval_months=25)


# =============================================================================
# HEATER CONFIG TESTS
# =============================================================================

class TestHeaterConfig:
    """Tests for HeaterConfig configuration."""

    def test_default_values(self):
        """Test default heater config values."""
        config = HeaterConfig()

        assert config.heater_id == "HT-001"
        assert config.heater_type == HeaterType.FIRED_HEATER
        assert config.design_duty_btu_hr == 10_000_000.0
        assert config.design_flow_gpm == 500.0
        assert config.design_delta_t_f == 50.0

    def test_custom_heater_type(self):
        """Test custom heater types."""
        config = HeaterConfig(heater_type=HeaterType.ELECTRIC)
        assert config.heater_type == HeaterType.ELECTRIC

        config = HeaterConfig(heater_type=HeaterType.WASTE_HEAT)
        assert config.heater_type == HeaterType.WASTE_HEAT

    def test_positive_values_required(self):
        """Test positive values required."""
        with pytest.raises(ValidationError):
            HeaterConfig(design_duty_btu_hr=0)

        with pytest.raises(ValidationError):
            HeaterConfig(design_flow_gpm=-100)

    def test_tube_dimensions(self):
        """Test tube dimension defaults."""
        config = HeaterConfig()

        assert config.coil_tube_od_in == 3.5
        assert config.coil_tube_id_in == 3.068
        assert config.coil_length_ft == 1000.0


# =============================================================================
# EXPANSION TANK CONFIG TESTS
# =============================================================================

class TestExpansionTankConfig:
    """Tests for ExpansionTankConfig configuration."""

    def test_default_values(self):
        """Test default expansion tank config values."""
        config = ExpansionTankConfig()

        assert config.tank_id == "ET-001"
        assert config.volume_gallons == 1000.0
        assert config.design_pressure_psig == 15.0
        assert config.design_temperature_f == 300.0
        assert config.inert_gas_blanket == True

    def test_nitrogen_blanket_settings(self):
        """Test nitrogen blanket settings."""
        config = ExpansionTankConfig()

        assert config.inert_gas_blanket == True
        assert config.blanket_pressure_psig == 2.0

        config = ExpansionTankConfig(inert_gas_blanket=False)
        assert config.inert_gas_blanket == False

    def test_level_limits(self):
        """Test level limit defaults."""
        config = ExpansionTankConfig()

        assert config.min_level_pct == 10.0
        assert config.max_level_pct == 90.0
        assert config.cold_level_target_pct == 25.0

    def test_level_percentage_range(self):
        """Test level percentage range validation."""
        with pytest.raises(ValidationError):
            ExpansionTankConfig(min_level_pct=-1)

        with pytest.raises(ValidationError):
            ExpansionTankConfig(max_level_pct=101)


# =============================================================================
# PUMP CONFIG TESTS
# =============================================================================

class TestPumpConfig:
    """Tests for PumpConfig configuration."""

    def test_default_values(self):
        """Test default pump config values."""
        config = PumpConfig()

        assert config.pump_id == "P-001"
        assert config.design_flow_gpm == 500.0
        assert config.design_head_ft == 150.0
        assert config.npsh_required_ft == 10.0
        assert config.efficiency_pct == 75.0
        assert config.motor_hp == 50.0

    def test_efficiency_range(self):
        """Test efficiency percentage range."""
        config = PumpConfig(efficiency_pct=50.0)
        assert config.efficiency_pct == 50.0

        config = PumpConfig(efficiency_pct=100.0)
        assert config.efficiency_pct == 100.0

        with pytest.raises(ValidationError):
            PumpConfig(efficiency_pct=0)

        with pytest.raises(ValidationError):
            PumpConfig(efficiency_pct=101)

    def test_npsh_can_be_zero(self):
        """Test NPSH required can be zero."""
        config = PumpConfig(npsh_required_ft=0)
        assert config.npsh_required_ft == 0


# =============================================================================
# PIPING CONFIG TESTS
# =============================================================================

class TestPipingConfig:
    """Tests for PipingConfig configuration."""

    def test_default_values(self):
        """Test default piping config values."""
        config = PipingConfig()

        assert config.pipe_schedule == "40"
        assert config.main_header_size_in == 6.0
        assert config.branch_line_size_in == 3.0
        assert config.total_pipe_length_ft == 500.0
        assert config.insulation_thickness_in == 2.0

    def test_positive_dimensions_required(self):
        """Test positive dimensions required."""
        with pytest.raises(ValidationError):
            PipingConfig(main_header_size_in=0)

        with pytest.raises(ValidationError):
            PipingConfig(total_pipe_length_ft=-100)

    def test_insulation_can_be_zero(self):
        """Test insulation can be zero (uninsulated)."""
        config = PipingConfig(insulation_thickness_in=0)
        assert config.insulation_thickness_in == 0


# =============================================================================
# EXERGY CONFIG TESTS
# =============================================================================

class TestExergyConfig:
    """Tests for ExergyConfig configuration."""

    def test_default_values(self):
        """Test default exergy config values."""
        config = ExergyConfig()

        assert config.enabled == True
        assert config.reference_temperature_f == 77.0
        assert config.reference_pressure_psia == 14.696
        assert config.include_chemical_exergy == False

    def test_can_disable_exergy(self):
        """Test exergy analysis can be disabled."""
        config = ExergyConfig(enabled=False)
        assert config.enabled == False


# =============================================================================
# MAIN CONFIG TESTS
# =============================================================================

class TestThermalFluidConfig:
    """Tests for ThermalFluidConfig main configuration."""

    def test_requires_system_id(self):
        """Test system_id is required."""
        with pytest.raises(ValidationError):
            ThermalFluidConfig()

    def test_minimal_config(self):
        """Test minimal configuration."""
        config = ThermalFluidConfig(system_id="TF-001")

        assert config.system_id == "TF-001"
        assert config.fluid_type == ThermalFluidType.THERMINOL_66
        assert config.system_volume_gallons == 5000.0

    def test_agent_id_auto_generated(self):
        """Test agent ID is auto-generated."""
        config = ThermalFluidConfig(system_id="TF-001")

        assert config.agent_id is not None
        assert config.agent_id.startswith("GL-009-")

    def test_all_sub_configs_default(self):
        """Test all sub-configurations have defaults."""
        config = ThermalFluidConfig(system_id="TF-001")

        assert isinstance(config.safety, SafetyConfig)
        assert isinstance(config.degradation, DegradationThresholds)
        assert isinstance(config.heater, HeaterConfig)
        assert isinstance(config.expansion_tank, ExpansionTankConfig)
        assert isinstance(config.pump, PumpConfig)
        assert isinstance(config.piping, PipingConfig)
        assert isinstance(config.exergy, ExergyConfig)

    def test_custom_fluid_type(self):
        """Test custom fluid type."""
        config = ThermalFluidConfig(
            system_id="TF-001",
            fluid_type=ThermalFluidType.DOWTHERM_A,
        )

        assert config.fluid_type == ThermalFluidType.DOWTHERM_A

    def test_cost_parameters(self):
        """Test cost parameters have defaults."""
        config = ThermalFluidConfig(system_id="TF-001")

        assert config.fuel_cost_usd_mmbtu == 8.0
        assert config.electricity_cost_usd_kwh == 0.10
        assert config.fluid_cost_usd_gallon == 15.0

    def test_feature_flags(self):
        """Test feature flags have defaults."""
        config = ThermalFluidConfig(system_id="TF-001")

        assert config.enable_ml_predictions == False
        assert config.enable_trending == True
        assert config.audit_enabled == True
        assert config.provenance_tracking == True

    def test_enum_serialization(self):
        """Test enum values serialize correctly."""
        config = ThermalFluidConfig(
            system_id="TF-001",
            fluid_type=ThermalFluidType.THERMINOL_66,
        )

        data = config.dict()
        assert data["fluid_type"] == "therminol_66"


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestCreateDefaultConfig:
    """Tests for create_default_config factory function."""

    def test_basic_creation(self):
        """Test basic config creation."""
        config = create_default_config(system_id="TF-001")

        assert config.system_id == "TF-001"
        assert config.fluid_type == ThermalFluidType.THERMINOL_66
        assert config.design_temperature_f == 600.0
        assert config.design_flow_gpm == 500.0

    def test_custom_fluid_type(self):
        """Test custom fluid type."""
        config = create_default_config(
            system_id="TF-001",
            fluid_type=ThermalFluidType.DOWTHERM_A,
        )

        assert config.fluid_type == ThermalFluidType.DOWTHERM_A

    def test_custom_design_conditions(self):
        """Test custom design conditions."""
        config = create_default_config(
            system_id="TF-001",
            design_temperature_f=700.0,
            design_flow_gpm=1000.0,
            system_volume_gallons=10000.0,
        )

        assert config.design_temperature_f == 700.0
        assert config.design_flow_gpm == 1000.0
        assert config.system_volume_gallons == 10000.0

    @pytest.mark.parametrize("fluid_type,expected_max_bulk", [
        (ThermalFluidType.THERMINOL_66, 650.0),
        (ThermalFluidType.THERMINOL_VP1, 750.0),
        (ThermalFluidType.DOWTHERM_A, 750.0),
        (ThermalFluidType.THERMINOL_55, 500.0),
    ])
    def test_fluid_specific_temperature_limits(self, fluid_type, expected_max_bulk):
        """Test fluid-specific temperature limits are set."""
        config = create_default_config(
            system_id="TF-001",
            fluid_type=fluid_type,
        )

        assert config.safety.temperature_limits.max_bulk_temp_f == expected_max_bulk

    def test_heater_config_matches_design(self):
        """Test heater config matches design flow."""
        config = create_default_config(
            system_id="TF-001",
            design_flow_gpm=800.0,
        )

        assert config.heater.design_flow_gpm == 800.0
        assert config.pump.design_flow_gpm == 800.0


class TestCreateHighTemperatureConfig:
    """Tests for create_high_temperature_config factory function."""

    def test_high_temp_config(self):
        """Test high temperature configuration."""
        config = create_high_temperature_config(
            system_id="TF-001",
            design_temperature_f=700.0,
        )

        assert config.design_temperature_f == 700.0
        # Should use Dowtherm A for high temp
        assert config.fluid_type == ThermalFluidType.DOWTHERM_A

    def test_medium_temp_uses_therminol(self):
        """Test medium temperature uses Therminol 66."""
        config = create_high_temperature_config(
            system_id="TF-001",
            design_temperature_f=600.0,
        )

        assert config.fluid_type == ThermalFluidType.THERMINOL_66

    def test_exceeds_max_temp_raises_error(self):
        """Test temperature exceeding max raises error."""
        with pytest.raises(ValueError) as exc_info:
            create_high_temperature_config(
                system_id="TF-001",
                design_temperature_f=800.0,
            )

        assert "exceeds maximum" in str(exc_info.value)

    def test_increased_safety_margins(self):
        """Test increased safety margins for high temp."""
        config = create_high_temperature_config(
            system_id="TF-001",
            design_temperature_f=700.0,
        )

        assert config.safety.temperature_limits.min_flash_point_margin_f == 75.0
        assert config.safety.temperature_limits.min_auto_ignition_margin_f == 125.0


class TestCreateFoodGradeConfig:
    """Tests for create_food_grade_config factory function."""

    def test_food_grade_config(self):
        """Test food grade configuration."""
        config = create_food_grade_config(
            system_id="TF-001",
            design_temperature_f=450.0,
        )

        assert config.fluid_type == ThermalFluidType.PARATHERM_NF

    def test_enhanced_monitoring(self):
        """Test enhanced monitoring for food grade."""
        config = create_food_grade_config(system_id="TF-001")

        # Shorter sampling interval
        assert config.degradation.sampling_interval_months == 3
        # Lower moisture limits
        assert config.degradation.moisture_warning_ppm == 200.0
        assert config.degradation.moisture_critical_ppm == 500.0


class TestGetTemperatureLimitsForFluid:
    """Tests for _get_temperature_limits_for_fluid helper."""

    @pytest.mark.parametrize("fluid_type,expected_max_film,expected_max_bulk", [
        (ThermalFluidType.THERMINOL_66, 705.0, 650.0),
        (ThermalFluidType.THERMINOL_VP1, 750.0, 750.0),
        (ThermalFluidType.THERMINOL_55, 550.0, 500.0),
        (ThermalFluidType.DOWTHERM_A, 750.0, 750.0),
        (ThermalFluidType.DOWTHERM_G, 700.0, 650.0),
        (ThermalFluidType.DOWTHERM_Q, 650.0, 600.0),
        (ThermalFluidType.MARLOTHERM_SH, 660.0, 610.0),
        (ThermalFluidType.SYLTHERM_800, 780.0, 750.0),
    ])
    def test_fluid_specific_limits(self, fluid_type, expected_max_film, expected_max_bulk):
        """Test fluid-specific temperature limits."""
        limits = _get_temperature_limits_for_fluid(fluid_type)

        assert limits.max_film_temp_f == expected_max_film
        assert limits.max_bulk_temp_f == expected_max_bulk

    def test_unknown_fluid_returns_defaults(self):
        """Test unknown fluid returns default limits."""
        # CUSTOM fluid type should return defaults
        limits = _get_temperature_limits_for_fluid(ThermalFluidType.CUSTOM)

        assert limits.max_film_temp_f == 700.0  # Default
        assert limits.max_bulk_temp_f == 650.0  # Default


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestConfigIntegration:
    """Integration tests for configuration."""

    def test_full_config_hierarchy(self):
        """Test full configuration hierarchy."""
        config = ThermalFluidConfig(
            system_id="TF-001",
            safety=SafetyConfig(
                sil_level=3,
                temperature_limits=TemperatureLimits(
                    max_film_temp_f=680.0,
                    max_bulk_temp_f=630.0,
                ),
            ),
        )

        assert config.safety.sil_level == 3
        assert config.safety.temperature_limits.max_film_temp_f == 680.0

    def test_config_serialization(self):
        """Test configuration serializes correctly."""
        config = create_default_config(system_id="TF-001")

        data = config.dict()

        assert data["system_id"] == "TF-001"
        assert "safety" in data
        assert "temperature_limits" in data["safety"]
        assert data["safety"]["temperature_limits"]["max_film_temp_f"] == 705.0

    def test_config_json_serialization(self):
        """Test configuration JSON serialization."""
        config = create_default_config(system_id="TF-001")

        json_str = config.json()

        assert "TF-001" in json_str
        assert "therminol_66" in json_str

    def test_config_copy_with_changes(self):
        """Test copying config with changes."""
        config1 = create_default_config(system_id="TF-001")
        config2 = config1.copy(update={"system_id": "TF-002"})

        assert config1.system_id == "TF-001"
        assert config2.system_id == "TF-002"
        # Other values should be same
        assert config1.fluid_type == config2.fluid_type
