"""
Unit tests for GL-020 ECONOPULSE Heat Transfer Effectiveness Calculator

Tests the epsilon-NTU method for heat exchanger effectiveness calculation.
Target coverage: 85%+

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units
    - Incropera & DeWitt, Fundamentals of Heat and Mass Transfer

Zero-Hallucination: Tests validate against known analytical solutions.
"""

import pytest
import math

from ..effectiveness import (
    EffectivenessCalculator,
    EffectivenessInput,
    create_effectiveness_calculator,
    CP_FLUE_GAS,
    CP_WATER,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def calculator():
    """Create default effectiveness calculator."""
    return EffectivenessCalculator()


@pytest.fixture
def calculator_custom_cp():
    """Create calculator with custom specific heats."""
    return EffectivenessCalculator(cp_gas=0.28, cp_water=1.0)


@pytest.fixture
def balanced_flow_input():
    """Input with balanced flow (Cr = 1)."""
    return EffectivenessInput(
        gas_inlet_temp_f=600.0,
        gas_outlet_temp_f=350.0,
        water_inlet_temp_f=250.0,
        water_outlet_temp_f=315.0,
        gas_flow_lb_hr=100000.0,
        water_flow_lb_hr=26000.0,  # Adjusted for Cr ~ 1
        design_effectiveness=0.80,
        design_ua_btu_hr_f=100000.0,
        clean_ua_btu_hr_f=120000.0,
        design_ntu=2.0,
        flow_arrangement="counterflow",
    )


@pytest.fixture
def standard_input():
    """Standard economizer input."""
    return EffectivenessInput(
        gas_inlet_temp_f=600.0,
        gas_outlet_temp_f=350.0,
        water_inlet_temp_f=250.0,
        water_outlet_temp_f=350.0,
        gas_flow_lb_hr=100000.0,
        water_flow_lb_hr=80000.0,
        design_effectiveness=0.80,
        design_ua_btu_hr_f=100000.0,
        clean_ua_btu_hr_f=120000.0,
        design_ntu=2.0,
        flow_arrangement="counterflow",
    )


@pytest.fixture
def degraded_input():
    """Input representing degraded performance."""
    return EffectivenessInput(
        gas_inlet_temp_f=600.0,
        gas_outlet_temp_f=400.0,  # Higher outlet temp (less heat transfer)
        water_inlet_temp_f=250.0,
        water_outlet_temp_f=310.0,  # Lower outlet temp
        gas_flow_lb_hr=100000.0,
        water_flow_lb_hr=80000.0,
        design_effectiveness=0.80,
        design_ua_btu_hr_f=100000.0,
        clean_ua_btu_hr_f=120000.0,
        design_ntu=2.0,
        flow_arrangement="counterflow",
    )


# =============================================================================
# CALCULATOR INITIALIZATION TESTS
# =============================================================================

class TestEffectivenessCalculatorInit:
    """Test EffectivenessCalculator initialization."""

    def test_default_initialization(self, calculator):
        """Test default calculator initialization."""
        assert calculator.cp_gas == CP_FLUE_GAS
        assert calculator.cp_water == CP_WATER

    def test_custom_specific_heats(self, calculator_custom_cp):
        """Test calculator with custom specific heats."""
        assert calculator_custom_cp.cp_gas == 0.28
        assert calculator_custom_cp.cp_water == 1.0

    def test_factory_function(self):
        """Test factory function creates calculator."""
        calc = create_effectiveness_calculator(cp_gas=0.27, cp_water=1.0)
        assert isinstance(calc, EffectivenessCalculator)
        assert calc.cp_gas == 0.27

    def test_default_constants(self):
        """Test default constant values are reasonable."""
        assert 0.2 < CP_FLUE_GAS < 0.3
        assert CP_WATER == 1.0


# =============================================================================
# CAPACITY RATE CALCULATION TESTS
# =============================================================================

class TestCapacityRates:
    """Test heat capacity rate calculations."""

    def test_capacity_rates_calculation(self, calculator):
        """Test capacity rate calculations."""
        c_gas, c_water, c_min, c_max = calculator.calculate_capacity_rates(
            gas_flow_lb_hr=100000.0,
            water_flow_lb_hr=80000.0,
        )

        expected_c_gas = 100000.0 * CP_FLUE_GAS  # 26000 BTU/hr-F
        expected_c_water = 80000.0 * CP_WATER    # 80000 BTU/hr-F

        assert c_gas == pytest.approx(expected_c_gas, rel=1e-6)
        assert c_water == pytest.approx(expected_c_water, rel=1e-6)
        assert c_min == expected_c_gas  # Gas side has smaller capacity
        assert c_max == expected_c_water

    def test_gas_side_minimum(self, calculator):
        """Test when gas side has minimum capacity."""
        c_gas, c_water, c_min, c_max = calculator.calculate_capacity_rates(
            gas_flow_lb_hr=100000.0,
            water_flow_lb_hr=50000.0,
        )

        assert c_min == c_gas
        assert c_max == c_water

    def test_water_side_minimum(self, calculator):
        """Test when water side has minimum capacity."""
        c_gas, c_water, c_min, c_max = calculator.calculate_capacity_rates(
            gas_flow_lb_hr=500000.0,  # Very high gas flow
            water_flow_lb_hr=10000.0,
        )

        assert c_min == c_water
        assert c_max == c_gas

    def test_balanced_flow(self, calculator):
        """Test balanced flow condition (Cr = 1)."""
        # Adjust flows to get Cr = 1
        water_flow = 100000.0 * CP_FLUE_GAS / CP_WATER  # 26000 lb/hr

        c_gas, c_water, c_min, c_max = calculator.calculate_capacity_rates(
            gas_flow_lb_hr=100000.0,
            water_flow_lb_hr=water_flow,
        )

        assert c_gas == pytest.approx(c_water, rel=1e-6)


# =============================================================================
# LMTD CALCULATION TESTS
# =============================================================================

class TestLMTDCalculation:
    """Test Log Mean Temperature Difference calculations."""

    def test_counterflow_lmtd(self, calculator):
        """Test LMTD for counterflow arrangement."""
        lmtd = calculator.calculate_lmtd(
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=250.0,
            water_outlet_temp_f=350.0,
            flow_arrangement="counterflow",
        )

        # Counterflow: delta_T1 = 600-350 = 250, delta_T2 = 350-250 = 100
        delta_t1 = 600.0 - 350.0  # 250
        delta_t2 = 350.0 - 250.0  # 100
        expected_lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

        assert lmtd == pytest.approx(expected_lmtd, rel=0.01)

    def test_parallel_flow_lmtd(self, calculator):
        """Test LMTD for parallel flow arrangement."""
        lmtd = calculator.calculate_lmtd(
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=250.0,
            water_outlet_temp_f=350.0,
            flow_arrangement="parallel",
        )

        # Parallel: delta_T1 = 600-250 = 350, delta_T2 = 350-350 = 0
        # But this is an edge case where outlet temps are equal
        # LMTD should handle this gracefully

        assert lmtd > 0  # Must be positive

    def test_equal_delta_t(self, calculator):
        """Test LMTD when temperature differences are equal."""
        lmtd = calculator.calculate_lmtd(
            gas_inlet_temp_f=500.0,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=200.0,
            water_outlet_temp_f=350.0,
            flow_arrangement="counterflow",
        )

        # delta_T1 = 500-350 = 150, delta_T2 = 350-200 = 150
        # When equal, LMTD = delta_T
        assert lmtd == pytest.approx(150.0, rel=0.01)

    def test_negative_delta_t_handling(self, calculator):
        """Test LMTD handles negative temperature differences."""
        # Unrealistic but tests error handling
        lmtd = calculator.calculate_lmtd(
            gas_inlet_temp_f=300.0,
            gas_outlet_temp_f=350.0,  # Gas heats up (invalid)
            water_inlet_temp_f=250.0,
            water_outlet_temp_f=400.0,
            flow_arrangement="counterflow",
        )

        # Should return minimum value to avoid errors
        assert lmtd > 0

    def test_lmtd_always_positive(self, calculator):
        """Test LMTD is always positive."""
        test_cases = [
            (600, 350, 250, 350),
            (500, 400, 200, 300),
            (700, 300, 150, 400),
        ]

        for gin, gout, win, wout in test_cases:
            lmtd = calculator.calculate_lmtd(gin, gout, win, wout)
            assert lmtd > 0


# =============================================================================
# EFFECTIVENESS CALCULATION TESTS
# =============================================================================

class TestActualEffectiveness:
    """Test actual effectiveness calculation from temperatures."""

    def test_effectiveness_from_temperatures(self, calculator):
        """Test effectiveness calculation from measured temperatures."""
        _, _, c_min, c_max = calculator.calculate_capacity_rates(100000.0, 80000.0)

        effectiveness = calculator.calculate_actual_effectiveness(
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=250.0,
            water_outlet_temp_f=350.0,
            c_min=c_min,
            c_max=c_max,
        )

        # Effectiveness should be between 0 and 1
        assert 0.0 <= effectiveness <= 1.0

    def test_maximum_effectiveness(self, calculator):
        """Test effectiveness approaches 1 for very high heat transfer."""
        _, _, c_min, c_max = calculator.calculate_capacity_rates(100000.0, 80000.0)

        effectiveness = calculator.calculate_actual_effectiveness(
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=260.0,  # Very close to water inlet
            water_inlet_temp_f=250.0,
            water_outlet_temp_f=500.0,
            c_min=c_min,
            c_max=c_max,
        )

        assert effectiveness > 0.9

    def test_minimum_effectiveness(self, calculator):
        """Test low effectiveness when little heat transfer."""
        _, _, c_min, c_max = calculator.calculate_capacity_rates(100000.0, 80000.0)

        effectiveness = calculator.calculate_actual_effectiveness(
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=580.0,  # Very little cooling
            water_inlet_temp_f=250.0,
            water_outlet_temp_f=260.0,
            c_min=c_min,
            c_max=c_max,
        )

        assert effectiveness < 0.2

    def test_effectiveness_clamped_to_valid_range(self, calculator):
        """Test effectiveness is clamped to 0-1 range."""
        _, _, c_min, c_max = calculator.calculate_capacity_rates(100000.0, 80000.0)

        effectiveness = calculator.calculate_actual_effectiveness(
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=250.0,
            water_outlet_temp_f=350.0,
            c_min=c_min,
            c_max=c_max,
        )

        assert 0.0 <= effectiveness <= 1.0


# =============================================================================
# EFFECTIVENESS-NTU RELATIONSHIP TESTS
# =============================================================================

class TestEffectivenessNTU:
    """Test effectiveness-NTU relationships."""

    def test_counterflow_effectiveness(self, calculator):
        """Test counterflow effectiveness formula."""
        # For counterflow: epsilon = (1 - exp(-NTU(1-Cr))) / (1 - Cr*exp(-NTU(1-Cr)))
        effectiveness = calculator.effectiveness_counterflow(ntu=2.0, c_r=0.5)

        assert 0.0 < effectiveness < 1.0
        # At NTU=2, Cr=0.5, effectiveness should be around 0.77
        assert effectiveness == pytest.approx(0.77, rel=0.05)

    def test_parallel_flow_effectiveness(self, calculator):
        """Test parallel flow effectiveness formula."""
        # For parallel: epsilon = (1 - exp(-NTU(1+Cr))) / (1 + Cr)
        effectiveness = calculator.effectiveness_parallel(ntu=2.0, c_r=0.5)

        assert 0.0 < effectiveness < 1.0
        # Parallel flow is always less effective than counterflow
        eff_counter = calculator.effectiveness_counterflow(ntu=2.0, c_r=0.5)
        assert effectiveness < eff_counter

    def test_balanced_flow_effectiveness(self, calculator):
        """Test effectiveness at balanced flow (Cr = 1)."""
        # For Cr=1: epsilon = NTU / (1 + NTU)
        effectiveness = calculator.effectiveness_counterflow(ntu=2.0, c_r=1.0)

        expected = 2.0 / (1 + 2.0)  # = 0.667
        assert effectiveness == pytest.approx(expected, rel=0.01)

    def test_effectiveness_increases_with_ntu(self, calculator):
        """Test effectiveness increases with NTU."""
        eff_low = calculator.effectiveness_counterflow(ntu=1.0, c_r=0.5)
        eff_high = calculator.effectiveness_counterflow(ntu=4.0, c_r=0.5)

        assert eff_high > eff_low

    def test_effectiveness_approaches_one(self, calculator):
        """Test effectiveness approaches 1 at very high NTU."""
        effectiveness = calculator.effectiveness_counterflow(ntu=10.0, c_r=0.5)

        assert effectiveness > 0.95

    def test_zero_ntu_effectiveness(self, calculator):
        """Test zero NTU gives zero effectiveness."""
        effectiveness = calculator.effectiveness_counterflow(ntu=0.0, c_r=0.5)

        assert effectiveness == 0.0


# =============================================================================
# NTU FROM EFFECTIVENESS TESTS
# =============================================================================

class TestNTUFromEffectiveness:
    """Test NTU calculation from effectiveness (inverse relationship)."""

    def test_ntu_from_effectiveness_counterflow(self, calculator):
        """Test NTU calculation for counterflow."""
        ntu = calculator.calculate_ntu_from_effectiveness(
            effectiveness=0.75,
            c_r=0.5,
            flow_arrangement="counterflow",
        )

        # Verify by recalculating effectiveness
        eff_check = calculator.effectiveness_counterflow(ntu, 0.5)
        assert eff_check == pytest.approx(0.75, rel=0.01)

    def test_ntu_from_effectiveness_balanced(self, calculator):
        """Test NTU calculation for balanced flow."""
        ntu = calculator.calculate_ntu_from_effectiveness(
            effectiveness=0.667,
            c_r=1.0,
            flow_arrangement="counterflow",
        )

        # For Cr=1: NTU = epsilon / (1 - epsilon)
        expected_ntu = 0.667 / (1 - 0.667)
        assert ntu == pytest.approx(expected_ntu, rel=0.01)

    def test_ntu_increases_with_effectiveness(self, calculator):
        """Test NTU increases with effectiveness."""
        ntu_low = calculator.calculate_ntu_from_effectiveness(0.5, 0.5)
        ntu_high = calculator.calculate_ntu_from_effectiveness(0.9, 0.5)

        assert ntu_high > ntu_low

    def test_zero_effectiveness_gives_zero_ntu(self, calculator):
        """Test zero effectiveness gives zero NTU."""
        ntu = calculator.calculate_ntu_from_effectiveness(0.0, 0.5)

        assert ntu == 0.0

    def test_unit_effectiveness_gives_infinite_ntu(self, calculator):
        """Test effectiveness of 1 gives infinite NTU."""
        ntu = calculator.calculate_ntu_from_effectiveness(1.0, 0.5)

        assert ntu == float('inf')


# =============================================================================
# UA CALCULATION TESTS
# =============================================================================

class TestUACalculation:
    """Test UA value calculation."""

    def test_ua_from_ntu(self, calculator):
        """Test UA calculation from NTU."""
        c_min = 26000.0  # BTU/hr-F
        ntu = 2.0

        ua = calculator.calculate_ua_from_ntu(ntu, c_min)

        expected_ua = ntu * c_min  # 52000 BTU/hr-F
        assert ua == pytest.approx(expected_ua, rel=1e-6)

    def test_ua_proportional_to_ntu(self, calculator):
        """Test UA is proportional to NTU."""
        c_min = 26000.0

        ua_low = calculator.calculate_ua_from_ntu(1.0, c_min)
        ua_high = calculator.calculate_ua_from_ntu(3.0, c_min)

        assert ua_high / ua_low == pytest.approx(3.0, rel=1e-6)

    def test_ua_proportional_to_c_min(self, calculator):
        """Test UA is proportional to C_min."""
        ntu = 2.0

        ua_low = calculator.calculate_ua_from_ntu(ntu, 20000.0)
        ua_high = calculator.calculate_ua_from_ntu(ntu, 40000.0)

        assert ua_high / ua_low == pytest.approx(2.0, rel=1e-6)


# =============================================================================
# COMPLETE CALCULATION TESTS
# =============================================================================

class TestCompleteCalculation:
    """Test complete effectiveness analysis."""

    def test_standard_calculation(self, calculator, standard_input):
        """Test complete calculation with standard input."""
        result = calculator.calculate(standard_input)

        # Verify all required fields
        assert "current_effectiveness" in result
        assert "design_effectiveness" in result
        assert "effectiveness_ratio" in result
        assert "current_ntu" in result
        assert "current_ua_btu_hr_f" in result
        assert "actual_duty_btu_hr" in result
        assert "lmtd_f" in result
        assert "provenance_hash" in result

        # Verify values are reasonable
        assert 0.0 < result["current_effectiveness"] < 1.0
        assert result["effectiveness_ratio"] > 0.0
        assert result["actual_duty_btu_hr"] > 0
        assert result["lmtd_f"] > 0

    def test_degraded_performance(self, calculator, degraded_input):
        """Test calculation shows degraded performance."""
        result = calculator.calculate(degraded_input)

        # Degraded performance should show lower effectiveness
        assert result["effectiveness_ratio"] < 1.0
        assert result["performance_status"] in ["degraded", "critical"]
        assert result["ua_degradation_pct"] > 0

    def test_performance_status_determination(self, calculator, standard_input):
        """Test performance status is determined correctly."""
        result = calculator.calculate(standard_input)

        # Status should be one of the valid values
        assert result["performance_status"] in ["normal", "degraded", "critical"]

    def test_duty_calculation(self, calculator, standard_input):
        """Test heat duty calculation."""
        result = calculator.calculate(standard_input)

        # Manual duty calculation
        c_gas = standard_input.gas_flow_lb_hr * calculator.cp_gas
        expected_duty = c_gas * (standard_input.gas_inlet_temp_f -
                                  standard_input.gas_outlet_temp_f)

        assert result["actual_duty_btu_hr"] == pytest.approx(expected_duty, rel=0.01)

    def test_temperature_differentials(self, calculator, standard_input):
        """Test temperature differential calculations."""
        result = calculator.calculate(standard_input)

        expected_gas_drop = standard_input.gas_inlet_temp_f - standard_input.gas_outlet_temp_f
        expected_water_rise = standard_input.water_outlet_temp_f - standard_input.water_inlet_temp_f

        assert result["gas_temp_drop_f"] == pytest.approx(expected_gas_drop, rel=0.01)
        assert result["water_temp_rise_f"] == pytest.approx(expected_water_rise, rel=0.01)

    def test_capacity_ratio_calculation(self, calculator, standard_input):
        """Test capacity ratio calculation."""
        result = calculator.calculate(standard_input)

        c_gas = standard_input.gas_flow_lb_hr * calculator.cp_gas
        c_water = standard_input.water_flow_lb_hr * calculator.cp_water
        expected_cr = min(c_gas, c_water) / max(c_gas, c_water)

        assert result["capacity_ratio"] == pytest.approx(expected_cr, rel=0.01)

    def test_provenance_hash_deterministic(self, calculator, standard_input):
        """Test provenance hash is deterministic."""
        result1 = calculator.calculate(standard_input)
        result2 = calculator.calculate(standard_input)

        assert result1["provenance_hash"] == result2["provenance_hash"]

    def test_calculation_method_recorded(self, calculator, standard_input):
        """Test calculation method is recorded."""
        result = calculator.calculate(standard_input)

        assert result["calculation_method"] == "NTU_EPSILON"
        assert "ASME" in result["formula_reference"]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_low_flow(self, calculator):
        """Test with very low flow rates."""
        input_data = EffectivenessInput(
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=250.0,
            water_outlet_temp_f=350.0,
            gas_flow_lb_hr=10000.0,  # Very low
            water_flow_lb_hr=8000.0,
            design_effectiveness=0.80,
            design_ua_btu_hr_f=100000.0,
            clean_ua_btu_hr_f=120000.0,
            design_ntu=2.0,
            flow_arrangement="counterflow",
        )

        result = calculator.calculate(input_data)

        # Should still produce valid results
        assert 0.0 < result["current_effectiveness"] < 1.0
        assert result["actual_duty_btu_hr"] > 0

    def test_very_high_flow(self, calculator):
        """Test with very high flow rates."""
        input_data = EffectivenessInput(
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=450.0,  # Less heat transfer at high flow
            water_inlet_temp_f=250.0,
            water_outlet_temp_f=280.0,
            gas_flow_lb_hr=500000.0,  # Very high
            water_flow_lb_hr=400000.0,
            design_effectiveness=0.80,
            design_ua_btu_hr_f=100000.0,
            clean_ua_btu_hr_f=120000.0,
            design_ntu=2.0,
            flow_arrangement="counterflow",
        )

        result = calculator.calculate(input_data)

        # High flow means lower effectiveness
        assert result["current_effectiveness"] < 0.5

    def test_small_temperature_difference(self, calculator):
        """Test with small temperature differences."""
        input_data = EffectivenessInput(
            gas_inlet_temp_f=350.0,
            gas_outlet_temp_f=340.0,  # Only 10F drop
            water_inlet_temp_f=300.0,
            water_outlet_temp_f=305.0,
            gas_flow_lb_hr=100000.0,
            water_flow_lb_hr=80000.0,
            design_effectiveness=0.80,
            design_ua_btu_hr_f=100000.0,
            clean_ua_btu_hr_f=120000.0,
            design_ntu=2.0,
            flow_arrangement="counterflow",
        )

        result = calculator.calculate(input_data)

        # Small temp difference means very low effectiveness
        assert result["current_effectiveness"] < 0.3


# =============================================================================
# FLOW ARRANGEMENT TESTS
# =============================================================================

class TestFlowArrangements:
    """Test different flow arrangements."""

    def test_counterflow_vs_parallel(self, calculator):
        """Test counterflow is more effective than parallel."""
        input_counter = EffectivenessInput(
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=250.0,
            water_outlet_temp_f=350.0,
            gas_flow_lb_hr=100000.0,
            water_flow_lb_hr=80000.0,
            design_effectiveness=0.80,
            design_ua_btu_hr_f=100000.0,
            clean_ua_btu_hr_f=120000.0,
            design_ntu=2.0,
            flow_arrangement="counterflow",
        )

        input_parallel = EffectivenessInput(
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=250.0,
            water_outlet_temp_f=350.0,
            gas_flow_lb_hr=100000.0,
            water_flow_lb_hr=80000.0,
            design_effectiveness=0.80,
            design_ua_btu_hr_f=100000.0,
            clean_ua_btu_hr_f=120000.0,
            design_ntu=2.0,
            flow_arrangement="parallel",
        )

        result_counter = calculator.calculate(input_counter)
        result_parallel = calculator.calculate(input_parallel)

        # LMTD should be higher for counterflow
        assert result_counter["lmtd_f"] > result_parallel["lmtd_f"]


# =============================================================================
# PARAMETERIZED TESTS
# =============================================================================

class TestParameterized:
    """Parameterized tests for various conditions."""

    @pytest.mark.parametrize("ntu,cr,expected_eff_range", [
        (0.5, 0.5, (0.35, 0.45)),
        (1.0, 0.5, (0.50, 0.60)),
        (2.0, 0.5, (0.70, 0.80)),
        (4.0, 0.5, (0.85, 0.95)),
        (2.0, 0.0, (0.85, 0.90)),  # C_r = 0
        (2.0, 1.0, (0.65, 0.70)),  # Balanced flow
    ])
    def test_effectiveness_ntu_relationship(self, calculator, ntu, cr, expected_eff_range):
        """Test effectiveness-NTU relationship at various conditions."""
        effectiveness = calculator.effectiveness_counterflow(ntu, cr)

        assert expected_eff_range[0] <= effectiveness <= expected_eff_range[1]

    @pytest.mark.parametrize("gas_inlet,gas_outlet,water_inlet,water_outlet", [
        (600, 350, 250, 350),
        (700, 400, 280, 380),
        (500, 300, 200, 280),
        (800, 450, 300, 420),
    ])
    def test_various_temperature_profiles(
        self, calculator, gas_inlet, gas_outlet, water_inlet, water_outlet
    ):
        """Test calculation with various temperature profiles."""
        input_data = EffectivenessInput(
            gas_inlet_temp_f=gas_inlet,
            gas_outlet_temp_f=gas_outlet,
            water_inlet_temp_f=water_inlet,
            water_outlet_temp_f=water_outlet,
            gas_flow_lb_hr=100000.0,
            water_flow_lb_hr=80000.0,
            design_effectiveness=0.80,
            design_ua_btu_hr_f=100000.0,
            clean_ua_btu_hr_f=120000.0,
            design_ntu=2.0,
            flow_arrangement="counterflow",
        )

        result = calculator.calculate(input_data)

        # All should produce valid results
        assert 0.0 < result["current_effectiveness"] < 1.0
        assert result["lmtd_f"] > 0
        assert result["actual_duty_btu_hr"] > 0

    @pytest.mark.parametrize("gas_flow,water_flow", [
        (50000, 40000),
        (100000, 80000),
        (150000, 120000),
        (200000, 160000),
    ])
    def test_various_flow_rates(self, calculator, gas_flow, water_flow):
        """Test calculation with various flow rates."""
        input_data = EffectivenessInput(
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=250.0,
            water_outlet_temp_f=350.0,
            gas_flow_lb_hr=gas_flow,
            water_flow_lb_hr=water_flow,
            design_effectiveness=0.80,
            design_ua_btu_hr_f=100000.0,
            clean_ua_btu_hr_f=120000.0,
            design_ntu=2.0,
            flow_arrangement="counterflow",
        )

        result = calculator.calculate(input_data)

        # All should produce valid results
        assert result["current_effectiveness"] > 0
        assert result["c_min_btu_hr_f"] > 0
        assert result["c_max_btu_hr_f"] > 0
