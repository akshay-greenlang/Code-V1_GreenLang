# -*- coding: utf-8 -*-
"""
GL-018 Configuration Tests
==========================

Unit tests for GL-018 UnifiedCombustionOptimizer configuration module.
Tests configuration schemas, validation, and defaults per NFPA 85, ASME PTC 4.1, API 560.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_018_unified_combustion.config import (
    UnifiedCombustionConfig,
    BurnerConfig,
    AirFuelConfig,
    FlueGasConfig,
    FlameStabilityConfig,
    EmissionsConfig,
    BMSConfig,
    SootBlowingConfig,
    BlowdownConfig,
    EfficiencyConfig,
    FuelType,
    EquipmentType,
    BurnerType,
    ControlMode,
    EmissionControlTechnology,
    BMSSequence,
)


class TestBurnerConfig:
    """Tests for burner configuration."""

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            BurnerConfig()

    def test_valid_config(self, default_burner_config):
        """Test valid burner configuration."""
        config = default_burner_config
        assert config.burner_id == "BNR-001"
        assert config.capacity_mmbtu_hr == 50.0
        assert config.turndown_ratio == 4.0

    def test_burner_count_bounds(self):
        """Test burner count bounds."""
        with pytest.raises(ValidationError):
            BurnerConfig(
                burner_id="BNR-001",
                capacity_mmbtu_hr=50.0,
                burner_count=0,  # Must be >= 1
            )

        with pytest.raises(ValidationError):
            BurnerConfig(
                burner_id="BNR-001",
                capacity_mmbtu_hr=50.0,
                burner_count=25,  # Must be <= 20
            )

    def test_turndown_ratio_bounds(self):
        """Test turndown ratio bounds."""
        with pytest.raises(ValidationError):
            BurnerConfig(
                burner_id="BNR-001",
                capacity_mmbtu_hr=50.0,
                turndown_ratio=1.5,  # Must be >= 2.0
            )

    def test_capacity_bounds(self):
        """Test capacity bounds."""
        with pytest.raises(ValidationError):
            BurnerConfig(
                burner_id="BNR-001",
                capacity_mmbtu_hr=0,  # Must be > 0
            )

        with pytest.raises(ValidationError):
            BurnerConfig(
                burner_id="BNR-001",
                capacity_mmbtu_hr=1500,  # Must be <= 1000
            )


class TestAirFuelConfig:
    """Tests for air-fuel ratio configuration."""

    def test_default_config(self, default_air_fuel_config):
        """Test default air-fuel configuration."""
        config = default_air_fuel_config
        assert config.target_o2_pct == 3.0
        assert config.cross_limiting_enabled is True

    def test_o2_bounds(self):
        """Test O2 percentage bounds."""
        with pytest.raises(ValidationError):
            AirFuelConfig(target_o2_pct=0.5)  # Must be >= 1.0

        with pytest.raises(ValidationError):
            AirFuelConfig(target_o2_pct=12.0)  # Must be <= 10.0

    def test_o2_ordering(self):
        """Test O2 min/max ordering."""
        # min_o2 should be less than max_o2
        config = AirFuelConfig(
            min_o2_pct=1.5,
            max_o2_pct=6.0,
            target_o2_pct=3.0,
        )
        assert config.min_o2_pct < config.max_o2_pct

    def test_excess_air_bounds(self):
        """Test excess air bounds."""
        with pytest.raises(ValidationError):
            AirFuelConfig(target_excess_air_pct=2.0)  # Must be >= 5.0

    def test_o2_trim_bounds(self):
        """Test O2 trim bias bounds."""
        with pytest.raises(ValidationError):
            AirFuelConfig(o2_trim_bias_max_pct=20.0)  # Must be <= 15.0


class TestFlueGasConfig:
    """Tests for flue gas configuration."""

    def test_default_config(self, default_flue_gas_config):
        """Test default flue gas configuration."""
        config = default_flue_gas_config
        assert config.max_flue_temp_f == 500.0
        assert config.co_alarm_ppm == 100.0

    def test_temperature_bounds(self):
        """Test temperature bounds."""
        with pytest.raises(ValidationError):
            FlueGasConfig(max_flue_temp_f=200.0)  # Must be >= 300

        with pytest.raises(ValidationError):
            FlueGasConfig(max_flue_temp_f=1500.0)  # Must be <= 1200

    def test_co_alarm_bounds(self):
        """Test CO alarm bounds."""
        with pytest.raises(ValidationError):
            FlueGasConfig(co_alarm_ppm=30.0)  # Must be >= 50

    def test_co_trip_bounds(self):
        """Test CO trip bounds."""
        with pytest.raises(ValidationError):
            FlueGasConfig(co_trip_ppm=50.0)  # Must be >= 100


class TestFlameStabilityConfig:
    """Tests for flame stability configuration."""

    def test_default_config(self, default_flame_stability_config):
        """Test default flame stability configuration."""
        config = default_flame_stability_config
        assert config.fsi_optimal_min == 0.85
        assert config.flame_failure_response_s == 4.0

    def test_fsi_threshold_ordering(self, default_flame_stability_config):
        """Test FSI threshold ordering."""
        config = default_flame_stability_config
        assert config.fsi_alarm_threshold < config.fsi_warning_threshold < config.fsi_optimal_min

    def test_flame_signal_bounds(self):
        """Test flame signal bounds."""
        with pytest.raises(ValidationError):
            FlameStabilityConfig(flame_signal_min_pct=5.0)  # Must be >= 10.0

    def test_flame_failure_response_bounds(self):
        """Test flame failure response time bounds."""
        with pytest.raises(ValidationError):
            FlameStabilityConfig(flame_failure_response_s=0.5)  # Must be >= 1.0


class TestEmissionsConfig:
    """Tests for emissions configuration."""

    def test_default_config(self, default_emissions_config):
        """Test default emissions configuration."""
        config = default_emissions_config
        assert config.nox_control == EmissionControlTechnology.LOW_NOX_BURNER
        assert config.nox_permit_limit_lb_mmbtu == 0.05

    def test_fgr_rate_bounds(self):
        """Test FGR rate bounds."""
        with pytest.raises(ValidationError):
            EmissionsConfig(fgr_rate_pct=3.0)  # Must be >= 5.0

        with pytest.raises(ValidationError):
            EmissionsConfig(fgr_rate_pct=35.0)  # Must be <= 30.0

    def test_scr_temperature_bounds(self):
        """Test SCR temperature bounds."""
        with pytest.raises(ValidationError):
            EmissionsConfig(scr_inlet_temp_min_f=300.0)  # Must be >= 400

    def test_permit_limit_bounds(self):
        """Test permit limit bounds."""
        with pytest.raises(ValidationError):
            EmissionsConfig(nox_permit_limit_lb_mmbtu=0.005)  # Must be >= 0.01


class TestBMSConfig:
    """Tests for BMS configuration per NFPA 85."""

    def test_default_config(self, default_bms_config):
        """Test default BMS configuration."""
        config = default_bms_config
        assert config.sil_level == 2
        assert config.pre_purge_time_s == 60.0

    def test_sil_level_bounds(self):
        """Test SIL level bounds."""
        with pytest.raises(ValidationError):
            BMSConfig(sil_level=0)  # Must be >= 1

        with pytest.raises(ValidationError):
            BMSConfig(sil_level=4)  # Must be <= 3

    def test_pre_purge_time_nfpa85(self):
        """Test pre-purge time per NFPA 85."""
        # NFPA 85 requires minimum 30 seconds
        with pytest.raises(ValidationError):
            BMSConfig(pre_purge_time_s=20.0)  # Must be >= 30.0

    def test_purge_air_flow_nfpa85(self):
        """Test purge air flow per NFPA 85."""
        # NFPA 85 requires minimum 25% of full load air
        with pytest.raises(ValidationError):
            BMSConfig(purge_air_flow_pct=20.0)  # Must be >= 25.0

    def test_purge_volume_changes(self):
        """Test purge volume changes per NFPA 85."""
        # NFPA 85 requires minimum 4 volume changes
        with pytest.raises(ValidationError):
            BMSConfig(purge_volume_changes=3)  # Must be >= 4


class TestSootBlowingConfig:
    """Tests for soot blowing configuration."""

    def test_default_config(self):
        """Test default soot blowing configuration."""
        config = SootBlowingConfig()
        assert config.enabled is True
        assert config.interval_hours == 8.0

    def test_interval_bounds(self):
        """Test blowing interval bounds."""
        with pytest.raises(ValidationError):
            SootBlowingConfig(interval_hours=0.5)  # Must be >= 1.0


class TestBlowdownConfig:
    """Tests for blowdown configuration."""

    def test_default_config(self):
        """Test default blowdown configuration."""
        config = BlowdownConfig()
        assert config.enabled is True
        assert config.control_type == "tds_based"

    def test_tds_setpoint_bounds(self):
        """Test TDS setpoint bounds."""
        with pytest.raises(ValidationError):
            BlowdownConfig(tds_setpoint_ppm=200.0)  # Must be >= 500


class TestEfficiencyConfig:
    """Tests for efficiency configuration per ASME PTC 4.1."""

    def test_default_config(self, default_efficiency_config):
        """Test default efficiency configuration."""
        config = default_efficiency_config
        assert config.calculation_method == "losses"
        assert config.design_efficiency_pct == 82.0

    def test_efficiency_bounds(self):
        """Test efficiency bounds."""
        with pytest.raises(ValidationError):
            EfficiencyConfig(design_efficiency_pct=40.0)  # Must be >= 50

        with pytest.raises(ValidationError):
            EfficiencyConfig(design_efficiency_pct=105.0)  # Must be <= 100

    def test_uncertainty_bounds(self):
        """Test measurement uncertainty bounds."""
        with pytest.raises(ValidationError):
            EfficiencyConfig(measurement_uncertainty_pct=0.3)  # Must be >= 0.5


class TestUnifiedCombustionConfig:
    """Tests for complete unified combustion configuration."""

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            UnifiedCombustionConfig()

    def test_minimal_config(self, default_burner_config):
        """Test minimal valid configuration."""
        config = UnifiedCombustionConfig(
            equipment_id="BOILER-001",
            burner=default_burner_config,
        )
        assert config.equipment_id == "BOILER-001"
        assert config.fuel_type == FuelType.NATURAL_GAS

    def test_full_config(self, default_combustion_config):
        """Test complete configuration."""
        config = default_combustion_config
        assert config.equipment_id == "BOILER-001"
        assert config.name == "Main Process Boiler"
        assert config.equipment_type == EquipmentType.BOILER_WATERTUBE

    def test_default_name_generation(self, default_burner_config):
        """Test default name generation."""
        config = UnifiedCombustionConfig(
            equipment_id="B-001",
            burner=default_burner_config,
        )
        assert "B-001" in config.name

    def test_load_bounds(self, default_burner_config):
        """Test load percentage bounds."""
        with pytest.raises(ValidationError):
            UnifiedCombustionConfig(
                equipment_id="B-001",
                burner=default_burner_config,
                min_load_pct=-10.0,  # Must be >= 0
            )

    def test_capacity_bounds(self, default_burner_config):
        """Test capacity bounds."""
        with pytest.raises(ValidationError):
            UnifiedCombustionConfig(
                equipment_id="B-001",
                burner=default_burner_config,
                design_capacity_mmbtu_hr=0,  # Must be > 0
            )

    def test_sub_configs_defaults(self, default_burner_config):
        """Test sub-configuration defaults."""
        config = UnifiedCombustionConfig(
            equipment_id="B-001",
            burner=default_burner_config,
        )
        assert config.air_fuel is not None
        assert config.flue_gas is not None
        assert config.flame_stability is not None
        assert config.emissions is not None
        assert config.bms is not None
        assert config.soot_blowing is not None
        assert config.blowdown is not None
        assert config.efficiency is not None


class TestEnums:
    """Tests for configuration enums."""

    def test_fuel_type_values(self):
        """Test fuel type enum values."""
        assert FuelType.NATURAL_GAS.value == "natural_gas"
        assert FuelType.NO2_FUEL_OIL.value == "no2_fuel_oil"
        assert FuelType.HYDROGEN.value == "hydrogen"

    def test_equipment_type_values(self):
        """Test equipment type enum values."""
        assert EquipmentType.BOILER_WATERTUBE.value == "boiler_watertube"
        assert EquipmentType.BOILER_FIRETUBE.value == "boiler_firetube"
        assert EquipmentType.FURNACE_PROCESS.value == "furnace_process"

    def test_burner_type_values(self):
        """Test burner type enum values."""
        assert BurnerType.LOW_NOX.value == "low_nox"
        assert BurnerType.ULTRA_LOW_NOX.value == "ultra_low_nox"
        assert BurnerType.STAGED_AIR.value == "staged_air"

    def test_control_mode_values(self):
        """Test control mode enum values."""
        assert ControlMode.MANUAL.value == "manual"
        assert ControlMode.AUTOMATIC.value == "automatic"
        assert ControlMode.OPTIMIZING.value == "optimizing"
        assert ControlMode.CROSS_LIMITING.value == "cross_limiting"

    def test_emission_control_values(self):
        """Test emission control technology enum values."""
        assert EmissionControlTechnology.LOW_NOX_BURNER.value == "low_nox_burner"
        assert EmissionControlTechnology.FLUE_GAS_RECIRCULATION.value == "fgr"
        assert EmissionControlTechnology.SELECTIVE_CATALYTIC_REDUCTION.value == "scr"

    def test_bms_sequence_values(self):
        """Test BMS sequence enum values."""
        assert BMSSequence.IDLE.value == "idle"
        assert BMSSequence.PRE_PURGE.value == "pre_purge"
        assert BMSSequence.RUNNING.value == "running"
        assert BMSSequence.LOCKOUT.value == "lockout"
