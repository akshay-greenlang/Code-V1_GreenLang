"""
Unit tests for GL-020 ECONOPULSE Configuration Module

Tests all configuration schemas with validation, defaults, and edge cases.
Target coverage: 85%+

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units
"""

import pytest
from pydantic import ValidationError

from ..config import (
    EconomizerType,
    EconomizerArrangement,
    TubeMaterial,
    FuelType,
    SootBlowerType,
    GasSideFoulingConfig,
    WaterSideFoulingConfig,
    SootBlowerConfig,
    AcidDewPointConfig,
    EffectivenessConfig,
    SteamingConfig,
    EconomizerDesignConfig,
    PerformanceBaselineConfig,
    EconomizerOptimizationConfig,
)


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestEconomizerType:
    """Test EconomizerType enum."""

    def test_all_values_exist(self):
        """Test all economizer types are defined."""
        assert EconomizerType.BARE_TUBE.value == "bare_tube"
        assert EconomizerType.FINNED_TUBE.value == "finned_tube"
        assert EconomizerType.EXTENDED_SURFACE.value == "extended_surface"
        assert EconomizerType.CAST_IRON.value == "cast_iron"
        assert EconomizerType.CONDENSING.value == "condensing"
        assert EconomizerType.NON_CONDENSING.value == "non_condensing"

    def test_enum_count(self):
        """Test correct number of economizer types."""
        assert len(EconomizerType) == 6


class TestEconomizerArrangement:
    """Test EconomizerArrangement enum."""

    def test_all_values_exist(self):
        """Test all flow arrangements are defined."""
        assert EconomizerArrangement.COUNTERFLOW.value == "counterflow"
        assert EconomizerArrangement.PARALLEL_FLOW.value == "parallel_flow"
        assert EconomizerArrangement.CROSSFLOW.value == "crossflow"
        assert EconomizerArrangement.CROSSFLOW_MIXED.value == "crossflow_mixed"


class TestTubeMaterial:
    """Test TubeMaterial enum."""

    def test_all_materials_exist(self):
        """Test all tube materials are defined."""
        assert TubeMaterial.CARBON_STEEL.value == "carbon_steel"
        assert TubeMaterial.LOW_ALLOY_STEEL.value == "low_alloy_steel"
        assert TubeMaterial.STAINLESS_304.value == "stainless_304"
        assert TubeMaterial.STAINLESS_316.value == "stainless_316"
        assert TubeMaterial.CORTEN.value == "corten"
        assert TubeMaterial.CAST_IRON.value == "cast_iron"


class TestFuelType:
    """Test FuelType enum."""

    def test_all_fuel_types_exist(self):
        """Test all fuel types are defined."""
        assert FuelType.NATURAL_GAS.value == "natural_gas"
        assert FuelType.NO2_FUEL_OIL.value == "no2_fuel_oil"
        assert FuelType.NO6_FUEL_OIL.value == "no6_fuel_oil"
        assert FuelType.COAL_BITUMINOUS.value == "coal_bituminous"
        assert FuelType.COAL_SUB_BITUMINOUS.value == "coal_sub_bituminous"
        assert FuelType.BIOMASS.value == "biomass"
        assert FuelType.REFINERY_GAS.value == "refinery_gas"


class TestSootBlowerType:
    """Test SootBlowerType enum."""

    def test_all_blower_types_exist(self):
        """Test all soot blower types are defined."""
        assert SootBlowerType.ROTARY.value == "rotary"
        assert SootBlowerType.RETRACTABLE.value == "retractable"
        assert SootBlowerType.STATIONARY.value == "stationary"
        assert SootBlowerType.ACOUSTIC.value == "acoustic"
        assert SootBlowerType.STEAM.value == "steam"
        assert SootBlowerType.AIR.value == "air"


# =============================================================================
# GAS-SIDE FOULING CONFIG TESTS
# =============================================================================

class TestGasSideFoulingConfig:
    """Test GasSideFoulingConfig schema."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GasSideFoulingConfig()

        assert config.design_gas_dp_in_wc == 2.0
        assert config.design_gas_velocity_fps == 50.0
        assert config.design_heat_transfer_coeff == 10.0
        assert config.dp_warning_ratio == 1.3
        assert config.dp_alarm_ratio == 1.5
        assert config.dp_cleaning_trigger_ratio == 1.7
        assert config.u_degradation_warning_pct == 10.0
        assert config.u_degradation_alarm_pct == 20.0
        assert config.trend_analysis_hours == 168
        assert config.trend_threshold_pct_per_day == 0.5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GasSideFoulingConfig(
            design_gas_dp_in_wc=3.0,
            dp_warning_ratio=1.4,
            dp_alarm_ratio=1.6,
        )

        assert config.design_gas_dp_in_wc == 3.0
        assert config.dp_warning_ratio == 1.4
        assert config.dp_alarm_ratio == 1.6

    def test_invalid_dp_warning_ratio_too_low(self):
        """Test validation rejects warning ratio below minimum."""
        with pytest.raises(ValidationError):
            GasSideFoulingConfig(dp_warning_ratio=1.0)

    def test_invalid_dp_warning_ratio_too_high(self):
        """Test validation rejects warning ratio above maximum."""
        with pytest.raises(ValidationError):
            GasSideFoulingConfig(dp_warning_ratio=2.5)

    def test_invalid_design_dp_zero(self):
        """Test validation rejects zero design DP."""
        with pytest.raises(ValidationError):
            GasSideFoulingConfig(design_gas_dp_in_wc=0)

    def test_invalid_design_dp_negative(self):
        """Test validation rejects negative design DP."""
        with pytest.raises(ValidationError):
            GasSideFoulingConfig(design_gas_dp_in_wc=-1.0)

    def test_trend_analysis_hours_bounds(self):
        """Test trend analysis hours within bounds."""
        # Minimum
        config = GasSideFoulingConfig(trend_analysis_hours=24)
        assert config.trend_analysis_hours == 24

        # Maximum
        config = GasSideFoulingConfig(trend_analysis_hours=720)
        assert config.trend_analysis_hours == 720

    def test_trend_analysis_hours_invalid(self):
        """Test invalid trend analysis hours."""
        with pytest.raises(ValidationError):
            GasSideFoulingConfig(trend_analysis_hours=10)


# =============================================================================
# WATER-SIDE FOULING CONFIG TESTS
# =============================================================================

class TestWaterSideFoulingConfig:
    """Test WaterSideFoulingConfig schema."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WaterSideFoulingConfig()

        assert config.design_water_dp_psi == 5.0
        assert config.design_water_velocity_fps == 6.0
        assert config.design_fouling_factor == 0.001
        assert config.dp_warning_ratio == 1.2
        assert config.dp_alarm_ratio == 1.4
        assert config.max_hardness_ppm == 0.5
        assert config.max_silica_ppm == 0.02
        assert config.max_iron_ppm == 0.01
        assert config.max_copper_ppm == 0.005
        assert config.target_ph == 9.2
        assert config.ph_tolerance == 0.3

    def test_chemistry_limits_validation(self):
        """Test water chemistry limit validation."""
        # Valid hardness
        config = WaterSideFoulingConfig(max_hardness_ppm=2.0)
        assert config.max_hardness_ppm == 2.0

        # Invalid hardness (too high)
        with pytest.raises(ValidationError):
            WaterSideFoulingConfig(max_hardness_ppm=10.0)

    def test_ph_bounds(self):
        """Test pH bounds validation."""
        # Valid pH range
        config = WaterSideFoulingConfig(target_ph=9.5)
        assert config.target_ph == 9.5

        # pH too low
        with pytest.raises(ValidationError):
            WaterSideFoulingConfig(target_ph=8.0)

        # pH too high
        with pytest.raises(ValidationError):
            WaterSideFoulingConfig(target_ph=11.0)

    def test_inspection_interval(self):
        """Test inspection interval validation."""
        config = WaterSideFoulingConfig(inspection_interval_months=12)
        assert config.inspection_interval_months == 12

        # Too frequent
        with pytest.raises(ValidationError):
            WaterSideFoulingConfig(inspection_interval_months=3)


# =============================================================================
# SOOT BLOWER CONFIG TESTS
# =============================================================================

class TestSootBlowerConfig:
    """Test SootBlowerConfig schema."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SootBlowerConfig()

        assert config.num_soot_blowers == 4
        assert config.blower_type == SootBlowerType.ROTARY
        assert config.steam_pressure_psig == 200.0
        assert config.steam_flow_per_blower_lb == 500.0
        assert config.blowing_duration_s == 90
        assert config.fixed_schedule_enabled is False
        assert config.fixed_interval_hours == 8.0
        assert config.min_interval_hours == 2.0
        assert config.max_interval_hours == 12.0

    def test_scheduling_parameters(self):
        """Test scheduling parameter validation."""
        config = SootBlowerConfig(
            min_interval_hours=3.0,
            max_interval_hours=24.0,
            fixed_interval_hours=12.0,
        )

        assert config.min_interval_hours == 3.0
        assert config.max_interval_hours == 24.0
        assert config.fixed_interval_hours == 12.0

    def test_trigger_thresholds(self):
        """Test trigger threshold validation."""
        config = SootBlowerConfig(
            dp_trigger_ratio=1.3,
            u_degradation_trigger_pct=7.0,
            exit_temp_rise_trigger_f=25.0,
        )

        assert config.dp_trigger_ratio == 1.3
        assert config.u_degradation_trigger_pct == 7.0
        assert config.exit_temp_rise_trigger_f == 25.0

    def test_invalid_num_blowers(self):
        """Test invalid number of soot blowers."""
        with pytest.raises(ValidationError):
            SootBlowerConfig(num_soot_blowers=-1)

        with pytest.raises(ValidationError):
            SootBlowerConfig(num_soot_blowers=25)

    def test_enum_value_serialization(self):
        """Test that enum values serialize correctly."""
        config = SootBlowerConfig(blower_type=SootBlowerType.RETRACTABLE)
        assert config.blower_type == SootBlowerType.RETRACTABLE


# =============================================================================
# ACID DEW POINT CONFIG TESTS
# =============================================================================

class TestAcidDewPointConfig:
    """Test AcidDewPointConfig schema."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AcidDewPointConfig()

        assert config.fuel_type == FuelType.NATURAL_GAS
        assert config.fuel_sulfur_pct == 0.001
        assert config.so3_conversion_pct == 2.0
        assert config.flue_gas_moisture_pct == 10.0
        assert config.acid_dew_point_margin_f == 30.0
        assert config.min_metal_temp_f == 270.0
        assert config.corrosion_probe_enabled is False

    def test_fuel_type_configuration(self):
        """Test different fuel type configurations."""
        # Coal configuration
        config = AcidDewPointConfig(
            fuel_type=FuelType.COAL_BITUMINOUS,
            fuel_sulfur_pct=2.5,
            so3_conversion_pct=3.0,
        )

        assert config.fuel_type == FuelType.COAL_BITUMINOUS
        assert config.fuel_sulfur_pct == 2.5
        assert config.so3_conversion_pct == 3.0

    def test_safety_margin_bounds(self):
        """Test safety margin bounds."""
        # Valid range
        config = AcidDewPointConfig(acid_dew_point_margin_f=50.0)
        assert config.acid_dew_point_margin_f == 50.0

        # Too low
        with pytest.raises(ValidationError):
            AcidDewPointConfig(acid_dew_point_margin_f=5.0)

        # Too high
        with pytest.raises(ValidationError):
            AcidDewPointConfig(acid_dew_point_margin_f=100.0)

    def test_corrosion_monitoring(self):
        """Test corrosion monitoring configuration."""
        config = AcidDewPointConfig(
            corrosion_probe_enabled=True,
            corrosion_rate_warning_mpy=8.0,
            corrosion_rate_alarm_mpy=15.0,
        )

        assert config.corrosion_probe_enabled is True
        assert config.corrosion_rate_warning_mpy == 8.0
        assert config.corrosion_rate_alarm_mpy == 15.0


# =============================================================================
# EFFECTIVENESS CONFIG TESTS
# =============================================================================

class TestEffectivenessConfig:
    """Test EffectivenessConfig schema."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EffectivenessConfig()

        assert config.design_effectiveness == 0.80
        assert config.design_ntu == 2.0
        assert config.effectiveness_warning_pct == 90.0
        assert config.effectiveness_alarm_pct == 80.0
        assert config.calculation_method == "ntu_epsilon"
        assert config.include_radiation_correction is True

    def test_effectiveness_bounds(self):
        """Test effectiveness bounds validation."""
        # Valid effectiveness
        config = EffectivenessConfig(design_effectiveness=0.85)
        assert config.design_effectiveness == 0.85

        # Too low
        with pytest.raises(ValidationError):
            EffectivenessConfig(design_effectiveness=0.4)

        # Too high
        with pytest.raises(ValidationError):
            EffectivenessConfig(design_effectiveness=1.0)

    def test_threshold_configuration(self):
        """Test alarm/warning threshold configuration."""
        config = EffectivenessConfig(
            effectiveness_warning_pct=85.0,
            effectiveness_alarm_pct=75.0,
        )

        assert config.effectiveness_warning_pct == 85.0
        assert config.effectiveness_alarm_pct == 75.0

    def test_reference_flow_rates(self):
        """Test reference flow rate configuration."""
        config = EffectivenessConfig(
            reference_gas_flow_lb_hr=150000.0,
            reference_water_flow_lb_hr=120000.0,
        )

        assert config.reference_gas_flow_lb_hr == 150000.0
        assert config.reference_water_flow_lb_hr == 120000.0


# =============================================================================
# STEAMING CONFIG TESTS
# =============================================================================

class TestSteamingConfig:
    """Test SteamingConfig schema."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SteamingConfig()

        assert config.design_approach_temp_f == 30.0
        assert config.design_subcooling_f == 20.0
        assert config.design_outlet_pressure_psig == 500.0
        assert config.approach_warning_f == 15.0
        assert config.approach_alarm_f == 10.0
        assert config.approach_trip_f == 5.0
        assert config.steaming_detection_enabled is True

    def test_approach_thresholds(self):
        """Test approach temperature thresholds."""
        config = SteamingConfig(
            approach_warning_f=20.0,
            approach_alarm_f=12.0,
            approach_trip_f=6.0,
        )

        assert config.approach_warning_f == 20.0
        assert config.approach_alarm_f == 12.0
        assert config.approach_trip_f == 6.0

    def test_recirculation_config(self):
        """Test recirculation configuration."""
        config = SteamingConfig(
            recirculation_enabled=True,
            recirculation_trigger_approach_f=15.0,
        )

        assert config.recirculation_enabled is True
        assert config.recirculation_trigger_approach_f == 15.0

    def test_low_load_limits(self):
        """Test low load limit configuration."""
        config = SteamingConfig(
            steaming_risk_load_pct=25.0,
            min_water_flow_pct=20.0,
        )

        assert config.steaming_risk_load_pct == 25.0
        assert config.min_water_flow_pct == 20.0


# =============================================================================
# ECONOMIZER DESIGN CONFIG TESTS
# =============================================================================

class TestEconomizerDesignConfig:
    """Test EconomizerDesignConfig schema."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EconomizerDesignConfig()

        assert config.economizer_type == EconomizerType.FINNED_TUBE
        assert config.arrangement == EconomizerArrangement.COUNTERFLOW
        assert config.tube_material == TubeMaterial.CARBON_STEEL
        assert config.total_surface_area_ft2 == 5000.0
        assert config.num_tubes == 200
        assert config.num_passes == 2

    def test_tube_specifications(self):
        """Test tube specification configuration."""
        config = EconomizerDesignConfig(
            tube_od_in=2.5,
            tube_wall_thickness_in=0.15,
            tube_length_ft=20.0,
            num_tubes=300,
        )

        assert config.tube_od_in == 2.5
        assert config.tube_wall_thickness_in == 0.15
        assert config.tube_length_ft == 20.0
        assert config.num_tubes == 300

    def test_fin_specifications(self):
        """Test fin specification configuration."""
        config = EconomizerDesignConfig(
            fin_height_in=1.0,
            fin_pitch_per_in=6.0,
            fin_thickness_in=0.06,
        )

        assert config.fin_height_in == 1.0
        assert config.fin_pitch_per_in == 6.0
        assert config.fin_thickness_in == 0.06

    def test_extended_surface_ratio(self):
        """Test extended surface ratio validation."""
        config = EconomizerDesignConfig(extended_surface_ratio=8.0)
        assert config.extended_surface_ratio == 8.0

        # Too low
        with pytest.raises(ValidationError):
            EconomizerDesignConfig(extended_surface_ratio=0.5)


# =============================================================================
# PERFORMANCE BASELINE CONFIG TESTS
# =============================================================================

class TestPerformanceBaselineConfig:
    """Test PerformanceBaselineConfig schema."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PerformanceBaselineConfig()

        assert config.design_duty_btu_hr == 20_000_000.0
        assert config.design_gas_inlet_temp_f == 600.0
        assert config.design_gas_outlet_temp_f == 350.0
        assert config.design_water_inlet_temp_f == 250.0
        assert config.design_water_outlet_temp_f == 350.0
        assert config.design_gas_flow_lb_hr == 100000.0
        assert config.design_water_flow_lb_hr == 80000.0

    def test_temperature_configuration(self):
        """Test temperature configuration."""
        config = PerformanceBaselineConfig(
            design_gas_inlet_temp_f=700.0,
            design_gas_outlet_temp_f=400.0,
            design_water_inlet_temp_f=280.0,
            design_water_outlet_temp_f=380.0,
        )

        assert config.design_gas_inlet_temp_f == 700.0
        assert config.design_gas_outlet_temp_f == 400.0
        assert config.design_water_inlet_temp_f == 280.0
        assert config.design_water_outlet_temp_f == 380.0

    def test_ua_values(self):
        """Test UA value configuration."""
        config = PerformanceBaselineConfig(
            design_ua_btu_hr_f=120000.0,
            clean_ua_btu_hr_f=140000.0,
        )

        assert config.design_ua_btu_hr_f == 120000.0
        assert config.clean_ua_btu_hr_f == 140000.0


# =============================================================================
# COMPLETE OPTIMIZATION CONFIG TESTS
# =============================================================================

class TestEconomizerOptimizationConfig:
    """Test EconomizerOptimizationConfig schema."""

    def test_minimal_config(self):
        """Test minimal required configuration."""
        config = EconomizerOptimizationConfig(economizer_id="ECO-001")

        assert config.economizer_id == "ECO-001"
        assert config.name == "Economizer ECO-001"  # Auto-generated
        assert config.boiler_id == ""
        assert config.optimization_enabled is True

    def test_full_config(self):
        """Test full configuration with all parameters."""
        config = EconomizerOptimizationConfig(
            economizer_id="ECO-001",
            name="Primary Economizer",
            boiler_id="BLR-001",
            design=EconomizerDesignConfig(
                economizer_type=EconomizerType.FINNED_TUBE,
                num_tubes=250,
            ),
            baseline=PerformanceBaselineConfig(
                design_duty_btu_hr=25_000_000.0,
            ),
            gas_side=GasSideFoulingConfig(
                dp_warning_ratio=1.25,
            ),
            optimization_enabled=True,
            sil_level=2,
        )

        assert config.economizer_id == "ECO-001"
        assert config.name == "Primary Economizer"
        assert config.boiler_id == "BLR-001"
        assert config.design.num_tubes == 250
        assert config.baseline.design_duty_btu_hr == 25_000_000.0
        assert config.gas_side.dp_warning_ratio == 1.25
        assert config.sil_level == 2

    def test_safety_settings(self):
        """Test safety settings configuration."""
        config = EconomizerOptimizationConfig(
            economizer_id="ECO-001",
            sil_level=3,
            high_water_temp_trip_f=500.0,
            low_water_flow_trip_pct=25.0,
            high_gas_temp_alarm_f=750.0,
        )

        assert config.sil_level == 3
        assert config.high_water_temp_trip_f == 500.0
        assert config.low_water_flow_trip_pct == 25.0
        assert config.high_gas_temp_alarm_f == 750.0

    def test_data_collection_settings(self):
        """Test data collection configuration."""
        config = EconomizerOptimizationConfig(
            economizer_id="ECO-001",
            historian_tag_prefix="PLANT.ECO1",
            data_collection_interval_s=5,
        )

        assert config.historian_tag_prefix == "PLANT.ECO1"
        assert config.data_collection_interval_s == 5

    def test_invalid_sil_level(self):
        """Test invalid SIL level validation."""
        with pytest.raises(ValidationError):
            EconomizerOptimizationConfig(
                economizer_id="ECO-001",
                sil_level=0,
            )

        with pytest.raises(ValidationError):
            EconomizerOptimizationConfig(
                economizer_id="ECO-001",
                sil_level=4,
            )

    def test_name_auto_generation(self):
        """Test automatic name generation from ID."""
        config = EconomizerOptimizationConfig(economizer_id="PRIMARY")
        assert config.name == "Economizer PRIMARY"

        # Explicit name should not be overwritten
        config = EconomizerOptimizationConfig(
            economizer_id="PRIMARY",
            name="My Custom Name"
        )
        assert config.name == "My Custom Name"

    def test_nested_config_modification(self):
        """Test nested configuration modification."""
        config = EconomizerOptimizationConfig(economizer_id="ECO-001")

        # Verify nested configs are accessible
        assert config.gas_side.dp_warning_ratio == 1.3
        assert config.water_side.max_hardness_ppm == 0.5
        assert config.soot_blower.num_soot_blowers == 4
        assert config.acid_dew_point.fuel_sulfur_pct == 0.001
        assert config.effectiveness.design_effectiveness == 0.80
        assert config.steaming.design_approach_temp_f == 30.0


# =============================================================================
# PARAMETERIZED TESTS
# =============================================================================

class TestConfigParameterized:
    """Parameterized tests for configuration validation."""

    @pytest.mark.parametrize("dp_ratio,expected_valid", [
        (1.1, True),
        (1.5, True),
        (2.0, True),
        (1.0, False),  # Below minimum
        (2.5, False),  # Above maximum
        (0.5, False),  # Way below
    ])
    def test_dp_warning_ratio_validation(self, dp_ratio, expected_valid):
        """Test DP warning ratio validation with various values."""
        if expected_valid:
            config = GasSideFoulingConfig(dp_warning_ratio=dp_ratio)
            assert config.dp_warning_ratio == dp_ratio
        else:
            with pytest.raises(ValidationError):
                GasSideFoulingConfig(dp_warning_ratio=dp_ratio)

    @pytest.mark.parametrize("fuel_type", list(FuelType))
    def test_all_fuel_types_valid(self, fuel_type):
        """Test all fuel types are valid in configuration."""
        config = AcidDewPointConfig(fuel_type=fuel_type)
        assert config.fuel_type == fuel_type

    @pytest.mark.parametrize("sulfur_pct,expected_valid", [
        (0.0, True),
        (0.001, True),
        (2.5, True),
        (5.0, True),
        (-0.1, False),  # Negative
        (5.1, False),   # Above maximum
    ])
    def test_fuel_sulfur_validation(self, sulfur_pct, expected_valid):
        """Test fuel sulfur percentage validation."""
        if expected_valid:
            config = AcidDewPointConfig(fuel_sulfur_pct=sulfur_pct)
            assert config.fuel_sulfur_pct == sulfur_pct
        else:
            with pytest.raises(ValidationError):
                AcidDewPointConfig(fuel_sulfur_pct=sulfur_pct)
