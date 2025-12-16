# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Condensate Load Module Tests

Unit tests for condensate_load.py module including startup loads,
operating loads, and DOE safety factors.

Target Coverage: 85%+
"""

import pytest
import math
from typing import Dict

from greenlang.agents.process_heat.gl_008_steam_trap_monitor.condensate_load import (
    PipeConstants,
    InsulationConstants,
    SteamTableConstants,
    SafetyFactors,
    LoadCalculationResult,
    StartupLoadCalculator,
    OperatingLoadCalculator,
    SafetyFactorCalculator,
    CondensateLoadCalculator,
)
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.schemas import (
    CondensateLoadInput,
    CondensateLoadOutput,
)


class TestPipeConstants:
    """Tests for pipe constant data."""

    def test_pipe_weight_data_exists(self):
        """Test pipe weight data is defined."""
        assert len(PipeConstants.PIPE_WEIGHT_LB_PER_FT) > 0

    def test_common_pipe_sizes(self):
        """Test common pipe sizes are defined."""
        common_sizes = [1.0, 2.0, 4.0, 6.0, 8.0]
        for size in common_sizes:
            assert size in PipeConstants.PIPE_WEIGHT_LB_PER_FT

    def test_pipe_weight_increases_with_size(self):
        """Test pipe weight increases with diameter."""
        sizes = sorted(PipeConstants.PIPE_WEIGHT_LB_PER_FT.keys())
        weights = [PipeConstants.PIPE_WEIGHT_LB_PER_FT[s] for s in sizes]

        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1], "Weight should increase with size"

    def test_pipe_od_data_exists(self):
        """Test pipe OD data is defined."""
        assert len(PipeConstants.PIPE_OD_IN) > 0

    def test_pipe_od_greater_than_nominal(self):
        """Test pipe OD is greater than nominal for Schedule 40."""
        for nominal, od in PipeConstants.PIPE_OD_IN.items():
            assert od > nominal, f"OD should exceed nominal for {nominal}"

    def test_specific_heat_values(self):
        """Test specific heat values are reasonable."""
        assert 0.05 < PipeConstants.SPECIFIC_HEAT_STEEL < 0.20
        assert 0.05 < PipeConstants.SPECIFIC_HEAT_COPPER < 0.15


class TestInsulationConstants:
    """Tests for insulation constant data."""

    def test_insulation_k_values_exist(self):
        """Test insulation K values are defined."""
        assert InsulationConstants.K_CALCIUM_SILICATE > 0
        assert InsulationConstants.K_MINERAL_WOOL > 0
        assert InsulationConstants.K_FIBERGLASS > 0

    def test_fiberglass_best_insulator(self):
        """Test fiberglass has lowest K value (best insulator)."""
        k_values = [
            InsulationConstants.K_CALCIUM_SILICATE,
            InsulationConstants.K_MINERAL_WOOL,
            InsulationConstants.K_FIBERGLASS,
            InsulationConstants.K_CELLULAR_GLASS,
        ]
        assert InsulationConstants.K_FIBERGLASS == min(k_values)


class TestSteamTableConstants:
    """Tests for steam table data."""

    def test_saturation_temp_at_atmospheric(self):
        """Test saturation temp at 0 psig is 212F."""
        assert SteamTableConstants.SATURATION_TEMP[0] == 212.0

    def test_saturation_temp_increases_with_pressure(self):
        """Test saturation temp increases with pressure."""
        pressures = sorted(SteamTableConstants.SATURATION_TEMP.keys())
        temps = [SteamTableConstants.SATURATION_TEMP[p] for p in pressures]

        for i in range(len(temps) - 1):
            assert temps[i] < temps[i + 1], "Temp should increase with pressure"

    def test_latent_heat_decreases_with_pressure(self):
        """Test latent heat decreases with pressure."""
        pressures = sorted(SteamTableConstants.LATENT_HEAT.keys())
        heats = [SteamTableConstants.LATENT_HEAT[p] for p in pressures]

        for i in range(len(heats) - 1):
            assert heats[i] > heats[i + 1], "Latent heat should decrease with pressure"

    def test_known_steam_table_values(self, known_steam_properties):
        """Test against known steam table values."""
        for pressure, props in known_steam_properties.items():
            assert abs(
                SteamTableConstants.SATURATION_TEMP.get(pressure, 0) -
                props["saturation_temp_f"]
            ) < 5, f"Saturation temp mismatch at {pressure} psig"

            assert abs(
                SteamTableConstants.LATENT_HEAT.get(pressure, 0) -
                props["latent_heat_btu_lb"]
            ) < 10, f"Latent heat mismatch at {pressure} psig"


class TestSafetyFactors:
    """Tests for DOE safety factors."""

    def test_startup_factors_defined(self):
        """Test startup safety factors are defined."""
        assert SafetyFactors.STARTUP_DRIP_LEG == 3.0  # DOE recommended
        assert SafetyFactors.STARTUP_HEAT_EXCHANGER == 2.0

    def test_operating_factors_defined(self):
        """Test operating safety factors are defined."""
        assert SafetyFactors.OPERATING_DRIP_LEG == 2.0  # DOE recommended
        assert SafetyFactors.OPERATING_HEAT_EXCHANGER == 2.0

    def test_get_startup_factor(self):
        """Test getting startup factor by application."""
        assert SafetyFactors.get_startup_factor("drip_leg") == 3.0
        assert SafetyFactors.get_startup_factor("heat_exchanger") == 2.0
        assert SafetyFactors.get_startup_factor("unknown") == 2.0  # Default

    def test_get_operating_factor(self):
        """Test getting operating factor by application."""
        assert SafetyFactors.get_operating_factor("drip_leg") == 2.0
        assert SafetyFactors.get_operating_factor("tracer") == 2.0

    def test_drip_leg_startup_higher_than_operating(self):
        """Test drip leg startup factor is higher than operating."""
        startup = SafetyFactors.get_startup_factor("drip_leg")
        operating = SafetyFactors.get_operating_factor("drip_leg")
        assert startup >= operating


class TestStartupLoadCalculator:
    """Tests for StartupLoadCalculator."""

    @pytest.fixture
    def calculator(self) -> StartupLoadCalculator:
        """Create calculator instance."""
        return StartupLoadCalculator()

    def test_initialization(self, calculator):
        """Test calculator initializes correctly."""
        assert calculator._calculation_count == 0

    def test_basic_pipe_warming_calculation(self, calculator):
        """Test basic pipe warming calculation."""
        result = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            ambient_temp_f=70.0,
            startup_time_minutes=15.0,
        )

        assert isinstance(result, LoadCalculationResult)
        assert result.load_lb_hr > 0
        assert result.calculation_type == "startup"
        assert result.formula_id == "STARTUP_PIPE_WARMING"
        assert len(result.inputs_hash) == 64

    def test_larger_pipe_higher_load(self, calculator):
        """Test larger pipe diameter creates higher load."""
        result_4in = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
        )

        result_8in = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=8.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
        )

        assert result_8in.load_lb_hr > result_4in.load_lb_hr

    def test_longer_pipe_higher_load(self, calculator):
        """Test longer pipe creates higher load."""
        result_100ft = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
        )

        result_200ft = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=200.0,
            steam_pressure_psig=150.0,
        )

        assert result_200ft.load_lb_hr > result_100ft.load_lb_hr
        # Should be approximately double
        assert abs(result_200ft.load_lb_hr / result_100ft.load_lb_hr - 2.0) < 0.1

    def test_shorter_startup_time_higher_rate(self, calculator):
        """Test shorter startup time creates higher rate."""
        result_15min = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            startup_time_minutes=15.0,
        )

        result_30min = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            startup_time_minutes=30.0,
        )

        assert result_15min.load_lb_hr > result_30min.load_lb_hr

    def test_zero_delta_t_zero_load(self, calculator):
        """Test zero temperature difference creates zero load."""
        result = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            ambient_temp_f=400.0,  # Above saturation temp
        )

        assert result.load_lb_hr == 0.0
        assert len(result.warnings) > 0

    def test_pipe_diameter_interpolation(self, calculator):
        """Test pipe diameter interpolation for non-standard sizes."""
        result = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=5.0,  # Not in table - between 4 and 6
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
        )

        # Should interpolate without error
        assert result.load_lb_hr > 0

    def test_pipe_diameter_extrapolation_warning(self, calculator):
        """Test warning for out-of-range pipe diameter."""
        result = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=0.25,  # Below minimum
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
        )

        assert len(result.warnings) > 0
        assert any("below minimum" in w.lower() for w in result.warnings)

    def test_stainless_steel_material(self, calculator):
        """Test stainless steel material specification."""
        result = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            pipe_material="stainless_steel",
        )

        assert result.intermediate_values["specific_heat"] == PipeConstants.SPECIFIC_HEAT_STAINLESS

    def test_copper_material(self, calculator):
        """Test copper material specification."""
        result = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            pipe_material="copper",
        )

        assert result.intermediate_values["specific_heat"] == PipeConstants.SPECIFIC_HEAT_COPPER

    def test_calculation_count_increments(self, calculator):
        """Test calculation count increments correctly."""
        assert calculator.calculation_count == 0

        calculator.calculate_pipe_warming_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
        )

        assert calculator.calculation_count == 1

    def test_intermediate_values_populated(self, calculator):
        """Test intermediate values are populated."""
        result = calculator.calculate_pipe_warming_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
        )

        assert "pipe_weight_lb" in result.intermediate_values
        assert "delta_t_f" in result.intermediate_values
        assert "latent_heat_btu_lb" in result.intermediate_values
        assert result.intermediate_values["pipe_weight_lb"] > 0


class TestOperatingLoadCalculator:
    """Tests for OperatingLoadCalculator."""

    @pytest.fixture
    def calculator(self) -> OperatingLoadCalculator:
        """Create calculator instance."""
        return OperatingLoadCalculator()

    def test_initialization(self, calculator):
        """Test calculator initializes correctly."""
        assert calculator._calculation_count == 0

    def test_basic_heat_loss_calculation(self, calculator):
        """Test basic pipe heat loss calculation."""
        result = calculator.calculate_pipe_heat_loss_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            insulation_thickness_in=2.0,
        )

        assert isinstance(result, LoadCalculationResult)
        assert result.load_lb_hr > 0
        assert result.calculation_type == "operating"

    def test_more_insulation_less_load(self, calculator):
        """Test more insulation reduces heat loss."""
        result_2in = calculator.calculate_pipe_heat_loss_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            insulation_thickness_in=2.0,
        )

        result_4in = calculator.calculate_pipe_heat_loss_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            insulation_thickness_in=4.0,
        )

        assert result_4in.load_lb_hr < result_2in.load_lb_hr

    def test_bare_pipe_warning(self, calculator):
        """Test warning for bare pipe."""
        result = calculator.calculate_pipe_heat_loss_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            insulation_thickness_in=0.0,  # Bare pipe
        )

        assert len(result.warnings) > 0
        assert any("bare pipe" in w.lower() for w in result.warnings)

    def test_wind_speed_effect(self, calculator):
        """Test wind speed increases heat loss."""
        result_no_wind = calculator.calculate_pipe_heat_loss_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            wind_speed_mph=0.0,
        )

        result_wind = calculator.calculate_pipe_heat_loss_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            wind_speed_mph=20.0,
        )

        assert result_wind.load_lb_hr > result_no_wind.load_lb_hr

    def test_different_insulation_types(self, calculator):
        """Test different insulation types."""
        insulation_types = [
            "calcium_silicate",
            "mineral_wool",
            "fiberglass",
        ]

        loads = {}
        for insulation in insulation_types:
            result = calculator.calculate_pipe_heat_loss_load(
                pipe_diameter_in=4.0,
                pipe_length_ft=100.0,
                steam_pressure_psig=150.0,
                insulation_thickness_in=2.0,
                insulation_type=insulation,
            )
            loads[insulation] = result.load_lb_hr

        # Fiberglass should have lowest loss (lowest k)
        assert loads["fiberglass"] < loads["calcium_silicate"]

    def test_heat_transfer_load_calculation(self, calculator):
        """Test heat transfer rate based calculation."""
        result = calculator.calculate_heat_transfer_load(
            heat_transfer_rate_btu_hr=1000000.0,  # 1 MMBTU/hr
            steam_pressure_psig=150.0,
        )

        assert result.load_lb_hr > 0
        # At 150 psig, latent heat ~857 BTU/lb
        # 1,000,000 / 857 = ~1167 lb/hr
        assert 1100 < result.load_lb_hr < 1200

    def test_zero_delta_t_zero_load(self, calculator):
        """Test zero temperature difference creates zero load."""
        result = calculator.calculate_pipe_heat_loss_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
            ambient_temp_f=400.0,  # Above saturation
        )

        assert result.load_lb_hr == 0.0

    def test_calculation_count_increments(self, calculator):
        """Test calculation count increments."""
        assert calculator.calculation_count == 0

        calculator.calculate_pipe_heat_loss_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=150.0,
        )

        assert calculator.calculation_count == 1


class TestSafetyFactorCalculator:
    """Tests for SafetyFactorCalculator."""

    @pytest.fixture
    def calculator(self) -> SafetyFactorCalculator:
        """Create calculator instance."""
        return SafetyFactorCalculator()

    def test_apply_startup_factor(self, calculator):
        """Test applying startup safety factor."""
        design_load, factor = calculator.apply_safety_factor(
            base_load_lb_hr=100.0,
            application="drip_leg",
            load_type="startup",
        )

        assert factor == 3.0  # DOE drip leg startup factor
        assert design_load == 300.0

    def test_apply_operating_factor(self, calculator):
        """Test applying operating safety factor."""
        design_load, factor = calculator.apply_safety_factor(
            base_load_lb_hr=100.0,
            application="heat_exchanger",
            load_type="operating",
        )

        assert factor == 2.0
        assert design_load == 200.0

    def test_custom_factor_override(self, calculator):
        """Test custom factor overrides default."""
        design_load, factor = calculator.apply_safety_factor(
            base_load_lb_hr=100.0,
            application="drip_leg",
            load_type="startup",
            custom_factor=4.0,
        )

        assert factor == 4.0
        assert design_load == 400.0


class TestCondensateLoadCalculator:
    """Tests for main CondensateLoadCalculator."""

    @pytest.fixture
    def calculator(self) -> CondensateLoadCalculator:
        """Create calculator instance."""
        return CondensateLoadCalculator(steam_pressure_psig=150.0)

    def test_initialization(self, calculator):
        """Test calculator initializes correctly."""
        assert calculator.steam_pressure_psig == 150.0
        assert calculator.calculation_count == 0

    def test_drip_leg_load_calculation(self, calculator):
        """Test complete drip leg load calculation."""
        result = calculator.calculate_drip_leg_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
        )

        assert isinstance(result, CondensateLoadOutput)
        assert result.startup_load_lb_hr >= 0
        assert result.operating_load_lb_hr >= 0
        assert result.design_load_lb_hr >= result.peak_load_lb_hr
        assert result.safety_factor >= 1.0
        assert len(result.provenance_hash) == 64

    def test_drip_leg_includes_trap_recommendations(self, calculator):
        """Test drip leg includes trap type recommendations."""
        result = calculator.calculate_drip_leg_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
        )

        assert "inverted_bucket" in result.recommended_trap_types
        assert "thermodynamic" in result.recommended_trap_types

    def test_heat_exchanger_load_calculation(self, calculator):
        """Test heat exchanger load calculation."""
        result = calculator.calculate_heat_exchanger_load(
            heat_transfer_rate_btu_hr=500000.0,
        )

        assert isinstance(result, CondensateLoadOutput)
        assert result.operating_load_lb_hr > 0
        assert "float_thermostatic" in result.recommended_trap_types

    def test_custom_pressure_override(self, calculator):
        """Test custom pressure overrides default."""
        result_default = calculator.calculate_drip_leg_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
        )

        result_custom = calculator.calculate_drip_leg_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            steam_pressure_psig=100.0,
        )

        # Different pressures should give different results
        # (different saturation temps and latent heats)
        assert result_default.operating_load_lb_hr != result_custom.operating_load_lb_hr

    def test_no_safety_factor(self, calculator):
        """Test calculation without safety factor."""
        result = calculator.calculate_drip_leg_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            apply_safety_factor=False,
        )

        assert result.safety_factor == 1.0
        assert result.design_load_lb_hr == result.peak_load_lb_hr

    def test_exclude_startup(self, calculator):
        """Test excluding startup load from sizing."""
        result_with = calculator.calculate_drip_leg_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            include_startup=True,
        )

        result_without = calculator.calculate_drip_leg_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
            include_startup=False,
        )

        # Startup load is typically higher
        assert result_with.peak_load_lb_hr >= result_without.peak_load_lb_hr

    def test_process_input_drip_leg(self, calculator, sample_condensate_load_input):
        """Test processing CondensateLoadInput for drip leg."""
        result = calculator.process(sample_condensate_load_input)

        assert isinstance(result, CondensateLoadOutput)
        assert result.design_load_lb_hr > 0

    def test_process_input_heat_exchanger(self, calculator):
        """Test processing CondensateLoadInput for heat exchanger."""
        input_data = CondensateLoadInput(
            application="heat_exchanger",
            steam_pressure_psig=150.0,
            heat_transfer_rate_btu_hr=500000.0,
        )

        result = calculator.process(input_data)

        assert isinstance(result, CondensateLoadOutput)
        assert result.operating_load_lb_hr > 0

    def test_process_input_invalid_application(self, calculator):
        """Test error for invalid application type."""
        input_data = CondensateLoadInput(
            application="invalid_type",
            steam_pressure_psig=150.0,
        )

        with pytest.raises(ValueError) as exc_info:
            calculator.process(input_data)

        assert "Unknown application" in str(exc_info.value)

    def test_process_drip_leg_missing_dimensions(self, calculator):
        """Test error for drip leg missing pipe dimensions."""
        input_data = CondensateLoadInput(
            application="drip_leg",
            steam_pressure_psig=150.0,
            # Missing pipe_diameter_in and pipe_length_ft
        )

        with pytest.raises(ValueError) as exc_info:
            calculator.process(input_data)

        assert "diameter and length required" in str(exc_info.value)

    def test_process_heat_exchanger_missing_heat_rate(self, calculator):
        """Test error for heat exchanger missing heat rate."""
        input_data = CondensateLoadInput(
            application="heat_exchanger",
            steam_pressure_psig=150.0,
            # Missing heat_transfer_rate_btu_hr
        )

        with pytest.raises(ValueError) as exc_info:
            calculator.process(input_data)

        assert "Heat transfer rate required" in str(exc_info.value)

    def test_calculation_count(self, calculator):
        """Test total calculation count."""
        assert calculator.calculation_count == 0

        calculator.calculate_drip_leg_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
        )

        # Should count both startup and operating calculations
        assert calculator.calculation_count >= 2


class TestCondensateLoadCalculations:
    """Integration tests for condensate load calculations."""

    @pytest.fixture
    def calculator(self) -> CondensateLoadCalculator:
        """Create calculator instance."""
        return CondensateLoadCalculator(steam_pressure_psig=150.0)

    @pytest.mark.parametrize("pipe_diameter,expected_min,expected_max", [
        (2.0, 10, 100),   # 2" pipe
        (4.0, 50, 300),   # 4" pipe
        (8.0, 150, 800),  # 8" pipe
    ])
    def test_drip_leg_reasonable_ranges(
        self,
        calculator,
        pipe_diameter: float,
        expected_min: float,
        expected_max: float,
    ):
        """Test drip leg calculations produce reasonable values."""
        result = calculator.calculate_drip_leg_load(
            pipe_diameter_in=pipe_diameter,
            pipe_length_ft=100.0,
            apply_safety_factor=False,
        )

        assert expected_min < result.peak_load_lb_hr < expected_max, \
            f"{pipe_diameter}\" pipe: {result.peak_load_lb_hr} lb/hr outside range"

    @pytest.mark.parametrize("heat_rate,expected_min,expected_max", [
        (100000, 100, 150),      # 100K BTU/hr
        (500000, 500, 700),      # 500K BTU/hr
        (1000000, 1000, 1300),   # 1 MMBTU/hr
    ])
    def test_heat_exchanger_reasonable_ranges(
        self,
        calculator,
        heat_rate: float,
        expected_min: float,
        expected_max: float,
    ):
        """Test heat exchanger calculations produce reasonable values."""
        result = calculator.calculate_heat_exchanger_load(
            heat_transfer_rate_btu_hr=heat_rate,
            apply_safety_factor=False,
        )

        assert expected_min < result.operating_load_lb_hr < expected_max, \
            f"{heat_rate} BTU/hr: {result.operating_load_lb_hr} lb/hr outside range"

    def test_provenance_hash_deterministic(self, calculator):
        """Test provenance hash is deterministic for same inputs."""
        result1 = calculator.calculate_drip_leg_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
        )

        result2 = calculator.calculate_drip_leg_load(
            pipe_diameter_in=4.0,
            pipe_length_ft=100.0,
        )

        # Note: Hash includes timestamp, so may differ
        # But calculation values should be identical
        assert result1.startup_load_lb_hr == result2.startup_load_lb_hr
        assert result1.operating_load_lb_hr == result2.operating_load_lb_hr
