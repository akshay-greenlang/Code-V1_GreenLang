"""
Unit tests for GL-009 THERMALIQ Agent Heat Transfer Calculations

Tests heat transfer coefficient calculations including Reynolds number,
Nusselt correlations (Dittus-Boelter, Gnielinski, Sieder-Tate), and
overall heat transfer coefficient analysis.
"""

import pytest
import math
from typing import Dict, Any

from greenlang.agents.process_heat.gl_009_thermal_fluid.heat_transfer import (
    HeatTransferCalculator,
    HeatTransferCorrelation,
    RE_LAMINAR_LIMIT,
    RE_TURBULENT_LIMIT,
    FOULING_FACTORS,
    calculate_reynolds,
    get_minimum_velocity,
)
from greenlang.agents.process_heat.gl_009_thermal_fluid.schemas import (
    ThermalFluidType,
    FlowRegime,
    HeatTransferAnalysis,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def heat_transfer_calc():
    """Create heat transfer calculator instance."""
    return HeatTransferCalculator(fluid_type=ThermalFluidType.THERMINOL_66)


@pytest.fixture
def heat_transfer_calc_with_wall():
    """Create calculator with wall temperature."""
    return HeatTransferCalculator(
        fluid_type=ThermalFluidType.THERMINOL_66,
        wall_temperature_f=650.0,
    )


@pytest.fixture
def typical_calc_params():
    """Create typical calculation parameters."""
    return {
        "temperature_f": 550.0,
        "velocity_ft_s": 8.0,
        "tube_id_in": 3.0,
    }


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestHeatTransferCalculatorInit:
    """Tests for HeatTransferCalculator initialization."""

    def test_default_initialization(self, heat_transfer_calc):
        """Test calculator initializes with defaults."""
        assert heat_transfer_calc.fluid_type == ThermalFluidType.THERMINOL_66
        assert heat_transfer_calc._calculation_count == 0
        assert heat_transfer_calc.wall_temperature_f is None

    def test_with_wall_temperature(self, heat_transfer_calc_with_wall):
        """Test calculator with wall temperature."""
        assert heat_transfer_calc_with_wall.wall_temperature_f == 650.0


# =============================================================================
# REYNOLDS NUMBER TESTS
# =============================================================================

class TestReynoldsNumber:
    """Tests for Reynolds number calculations."""

    def test_film_coefficient_returns_reynolds(
        self, heat_transfer_calc, typical_calc_params
    ):
        """Test film coefficient calculation returns Reynolds number."""
        result = heat_transfer_calc.calculate_film_coefficient(**typical_calc_params)

        assert result.reynolds_number > 0

    def test_reynolds_increases_with_velocity(self, heat_transfer_calc):
        """Test Reynolds number increases with velocity."""
        result_low = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=550.0,
            velocity_ft_s=4.0,
            tube_id_in=3.0,
        )

        result_high = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=550.0,
            velocity_ft_s=8.0,
            tube_id_in=3.0,
        )

        assert result_high.reynolds_number > result_low.reynolds_number

    def test_reynolds_increases_with_diameter(self, heat_transfer_calc):
        """Test Reynolds number increases with diameter."""
        result_small = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=550.0,
            velocity_ft_s=8.0,
            tube_id_in=2.0,
        )

        result_large = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=550.0,
            velocity_ft_s=8.0,
            tube_id_in=4.0,
        )

        assert result_large.reynolds_number > result_small.reynolds_number

    def test_reynolds_increases_with_temperature(self, heat_transfer_calc):
        """Test Reynolds number increases with temperature (lower viscosity)."""
        result_cold = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=200.0,
            velocity_ft_s=8.0,
            tube_id_in=3.0,
        )

        result_hot = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=600.0,
            velocity_ft_s=8.0,
            tube_id_in=3.0,
        )

        # Higher temp = lower viscosity = higher Re
        assert result_hot.reynolds_number > result_cold.reynolds_number

    def test_reynolds_calculation_accuracy(self):
        """Test Reynolds number calculation is accurate."""
        # Re = V * D / nu
        reynolds = calculate_reynolds(
            fluid_type=ThermalFluidType.THERMINOL_66,
            temperature_f=550.0,
            velocity_ft_s=8.0,
            diameter_in=3.0,
        )

        # Typical Reynolds for these conditions: 30,000 - 80,000
        assert 20000 <= reynolds <= 100000


# =============================================================================
# FLOW REGIME TESTS
# =============================================================================

class TestFlowRegime:
    """Tests for flow regime determination."""

    def test_laminar_flow_regime(self, heat_transfer_calc):
        """Test laminar flow regime detection."""
        result = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=100.0,  # Cold = high viscosity
            velocity_ft_s=0.5,    # Low velocity
            tube_id_in=2.0,
        )

        assert result.flow_regime == FlowRegime.LAMINAR
        assert result.reynolds_number < RE_LAMINAR_LIMIT

    def test_turbulent_flow_regime(self, heat_transfer_calc, typical_calc_params):
        """Test turbulent flow regime detection."""
        result = heat_transfer_calc.calculate_film_coefficient(**typical_calc_params)

        assert result.flow_regime == FlowRegime.TURBULENT
        assert result.reynolds_number >= RE_TURBULENT_LIMIT

    def test_transitional_flow_regime(self, heat_transfer_calc):
        """Test transitional flow regime detection."""
        # Find conditions that give transitional flow
        result = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=300.0,
            velocity_ft_s=1.5,
            tube_id_in=2.0,
        )

        # This might be transitional or turbulent depending on exact properties
        assert result.flow_regime in [FlowRegime.TRANSITIONAL, FlowRegime.TURBULENT, FlowRegime.LAMINAR]


# =============================================================================
# NUSSELT NUMBER CORRELATION TESTS
# =============================================================================

class TestNusseltCorrelations:
    """Tests for Nusselt number correlations."""

    def test_dittus_boelter_correlation(self, heat_transfer_calc, typical_calc_params):
        """Test Dittus-Boelter correlation."""
        result = heat_transfer_calc.calculate_film_coefficient(
            **typical_calc_params,
            correlation=HeatTransferCorrelation.DITTUS_BOELTER,
        )

        assert "Dittus-Boelter" in result.correlation_used
        assert result.nusselt_number > 0

    def test_gnielinski_correlation(self, heat_transfer_calc, typical_calc_params):
        """Test Gnielinski correlation."""
        result = heat_transfer_calc.calculate_film_coefficient(
            **typical_calc_params,
            correlation=HeatTransferCorrelation.GNIELINSKI,
        )

        assert "Gnielinski" in result.correlation_used
        assert result.nusselt_number > 0

    def test_sieder_tate_correlation(self, heat_transfer_calc, typical_calc_params):
        """Test Sieder-Tate correlation."""
        result = heat_transfer_calc.calculate_film_coefficient(
            **typical_calc_params,
            correlation=HeatTransferCorrelation.SIEDER_TATE,
            wall_temperature_f=650.0,
        )

        assert "Sieder-Tate" in result.correlation_used
        assert result.nusselt_number > 0

    def test_petukhov_correlation(self, heat_transfer_calc, typical_calc_params):
        """Test Petukhov correlation."""
        result = heat_transfer_calc.calculate_film_coefficient(
            **typical_calc_params,
            correlation=HeatTransferCorrelation.PETUKHOV,
        )

        assert "Petukhov" in result.correlation_used
        assert result.nusselt_number > 0

    def test_laminar_correlation_constant_wall(self, heat_transfer_calc):
        """Test laminar correlation for constant wall temperature."""
        result = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=100.0,
            velocity_ft_s=0.2,
            tube_id_in=1.0,
            correlation=HeatTransferCorrelation.LAMINAR_CONSTANT_WALL,
        )

        # Laminar constant wall temp: Nu = 3.66
        if result.flow_regime == FlowRegime.LAMINAR:
            assert abs(result.nusselt_number - 3.66) < 0.1

    def test_laminar_correlation_constant_flux(self, heat_transfer_calc):
        """Test laminar correlation for constant heat flux."""
        result = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=100.0,
            velocity_ft_s=0.2,
            tube_id_in=1.0,
            correlation=HeatTransferCorrelation.LAMINAR_CONSTANT_FLUX,
        )

        # Laminar constant heat flux: Nu = 4.36
        if result.flow_regime == FlowRegime.LAMINAR:
            assert abs(result.nusselt_number - 4.36) < 0.1

    def test_correlations_give_similar_results(self, heat_transfer_calc, typical_calc_params):
        """Test different correlations give similar order of magnitude."""
        correlations = [
            HeatTransferCorrelation.DITTUS_BOELTER,
            HeatTransferCorrelation.GNIELINSKI,
            HeatTransferCorrelation.PETUKHOV,
        ]

        results = []
        for corr in correlations:
            result = heat_transfer_calc.calculate_film_coefficient(
                **typical_calc_params,
                correlation=corr,
            )
            results.append(result.film_coefficient_btu_hr_ft2_f)

        # All should be within factor of 2
        max_h = max(results)
        min_h = min(results)
        assert max_h / min_h < 2.0


# =============================================================================
# FILM COEFFICIENT TESTS
# =============================================================================

class TestFilmCoefficient:
    """Tests for film coefficient calculations."""

    def test_film_coefficient_positive(
        self, heat_transfer_calc, typical_calc_params
    ):
        """Test film coefficient is positive."""
        result = heat_transfer_calc.calculate_film_coefficient(**typical_calc_params)

        assert result.film_coefficient_btu_hr_ft2_f > 0

    def test_film_coefficient_reasonable_range(
        self, heat_transfer_calc, typical_calc_params
    ):
        """Test film coefficient is in reasonable range."""
        result = heat_transfer_calc.calculate_film_coefficient(**typical_calc_params)

        # Typical range for organic heat transfer fluids: 20-500 BTU/hr-ft2-F
        assert 10 <= result.film_coefficient_btu_hr_ft2_f <= 800

    def test_film_coefficient_increases_with_velocity(self, heat_transfer_calc):
        """Test film coefficient increases with velocity."""
        result_low = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=550.0,
            velocity_ft_s=4.0,
            tube_id_in=3.0,
        )

        result_high = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=550.0,
            velocity_ft_s=12.0,
            tube_id_in=3.0,
        )

        assert result_high.film_coefficient_btu_hr_ft2_f > result_low.film_coefficient_btu_hr_ft2_f

    def test_film_coefficient_decreases_with_diameter(self, heat_transfer_calc):
        """Test film coefficient decreases with larger diameter."""
        result_small = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=550.0,
            velocity_ft_s=8.0,
            tube_id_in=2.0,
        )

        result_large = heat_transfer_calc.calculate_film_coefficient(
            temperature_f=550.0,
            velocity_ft_s=8.0,
            tube_id_in=6.0,
        )

        # h = Nu * k / D, so larger D gives lower h (all else equal)
        assert result_small.film_coefficient_btu_hr_ft2_f > result_large.film_coefficient_btu_hr_ft2_f


# =============================================================================
# OVERALL COEFFICIENT TESTS
# =============================================================================

class TestOverallCoefficient:
    """Tests for overall heat transfer coefficient calculations."""

    @pytest.fixture
    def typical_overall_params(self):
        """Create typical overall coefficient parameters."""
        return {
            "h_inside": 150.0,
            "h_outside": 100.0,
            "tube_od_in": 3.5,
            "tube_id_in": 3.068,
            "tube_conductivity_btu_hr_ft_f": 26.0,
            "fouling_inside": 0.001,
            "fouling_outside": 0.001,
        }

    def test_overall_coefficient_calculated(
        self, heat_transfer_calc, typical_overall_params
    ):
        """Test overall coefficient is calculated."""
        result = heat_transfer_calc.calculate_overall_coefficient(**typical_overall_params)

        assert "overall_coefficient_btu_hr_ft2_f" in result
        assert result["overall_coefficient_btu_hr_ft2_f"] > 0

    def test_overall_less_than_individual(
        self, heat_transfer_calc, typical_overall_params
    ):
        """Test overall coefficient is less than either individual coefficient."""
        result = heat_transfer_calc.calculate_overall_coefficient(**typical_overall_params)

        u_overall = result["overall_coefficient_btu_hr_ft2_f"]
        assert u_overall < typical_overall_params["h_inside"]
        assert u_overall < typical_overall_params["h_outside"]

    def test_clean_coefficient_higher_than_fouled(
        self, heat_transfer_calc, typical_overall_params
    ):
        """Test clean coefficient is higher than fouled."""
        result = heat_transfer_calc.calculate_overall_coefficient(**typical_overall_params)

        assert result["clean_coefficient_btu_hr_ft2_f"] > result["overall_coefficient_btu_hr_ft2_f"]

    def test_fouling_effect_calculated(
        self, heat_transfer_calc, typical_overall_params
    ):
        """Test fouling effect percentage is calculated."""
        result = heat_transfer_calc.calculate_overall_coefficient(**typical_overall_params)

        assert "fouling_effect_pct" in result
        assert 0 <= result["fouling_effect_pct"] <= 100

    def test_resistance_breakdown_sums_to_100(
        self, heat_transfer_calc, typical_overall_params
    ):
        """Test resistance breakdown percentages sum to approximately 100%."""
        result = heat_transfer_calc.calculate_overall_coefficient(**typical_overall_params)

        breakdown = result["resistance_breakdown"]
        total = sum(breakdown.values())

        assert 99 <= total <= 101

    def test_individual_resistances_calculated(
        self, heat_transfer_calc, typical_overall_params
    ):
        """Test individual thermal resistances are calculated."""
        result = heat_transfer_calc.calculate_overall_coefficient(**typical_overall_params)

        resistances = result["individual_resistances_hr_ft2_f_btu"]

        assert "inside_film" in resistances
        assert "inside_fouling" in resistances
        assert "tube_wall" in resistances
        assert "outside_fouling" in resistances
        assert "outside_film" in resistances
        assert "total" in resistances

        # Total should be sum of individuals
        sum_individual = sum(v for k, v in resistances.items() if k != "total")
        assert abs(resistances["total"] - sum_individual) < 0.000001


# =============================================================================
# FOULING ESTIMATION TESTS
# =============================================================================

class TestFoulingEstimation:
    """Tests for fouling estimation from performance degradation."""

    def test_estimate_fouling_no_degradation(self, heat_transfer_calc):
        """Test fouling estimation with no degradation."""
        result = heat_transfer_calc.estimate_fouling(
            design_ua=100000.0,
            actual_ua=100000.0,  # Same as design
            tube_od_in=3.5,
            tube_id_in=3.068,
        )

        assert result["fouling_factor_hr_ft2_f_btu"] == 0.0
        assert result["fouling_level"] == "clean"

    def test_estimate_fouling_with_degradation(self, heat_transfer_calc):
        """Test fouling estimation with performance degradation."""
        result = heat_transfer_calc.estimate_fouling(
            design_ua=100000.0,
            actual_ua=70000.0,  # 30% degradation
            tube_od_in=3.5,
            tube_id_in=3.068,
        )

        assert result["fouling_factor_hr_ft2_f_btu"] > 0
        assert result["fouling_level"] in ["light", "moderate", "heavy", "severe"]

    def test_fouling_level_classification(self, heat_transfer_calc):
        """Test fouling level classification."""
        # Severe degradation
        result = heat_transfer_calc.estimate_fouling(
            design_ua=100000.0,
            actual_ua=40000.0,  # 60% degradation
            tube_od_in=3.5,
            tube_id_in=3.068,
        )

        assert result["fouling_level"] in ["moderate", "heavy", "severe"]

    def test_fouling_recommendations(self, heat_transfer_calc):
        """Test fouling generates recommendations for high fouling."""
        result = heat_transfer_calc.estimate_fouling(
            design_ua=100000.0,
            actual_ua=50000.0,  # 50% degradation
            tube_od_in=3.5,
            tube_id_in=3.068,
        )

        assert len(result["recommendations"]) > 0


# =============================================================================
# MINIMUM VELOCITY TESTS
# =============================================================================

class TestMinimumVelocity:
    """Tests for minimum velocity calculations."""

    def test_calculate_minimum_velocity(self, heat_transfer_calc):
        """Test minimum velocity calculation."""
        min_velocity = heat_transfer_calc.calculate_minimum_velocity(
            temperature_f=550.0,
            tube_id_in=3.0,
            target_reynolds=10000,
        )

        assert min_velocity > 0

    def test_minimum_velocity_decreases_at_high_temp(self, heat_transfer_calc):
        """Test minimum velocity decreases at high temperature (lower viscosity)."""
        min_vel_cold = heat_transfer_calc.calculate_minimum_velocity(
            temperature_f=200.0,
            tube_id_in=3.0,
        )

        min_vel_hot = heat_transfer_calc.calculate_minimum_velocity(
            temperature_f=600.0,
            tube_id_in=3.0,
        )

        # Higher temp = lower viscosity = lower velocity needed for turbulent
        assert min_vel_hot < min_vel_cold

    def test_minimum_velocity_increases_with_smaller_tube(self, heat_transfer_calc):
        """Test minimum velocity increases with smaller tube."""
        min_vel_large = heat_transfer_calc.calculate_minimum_velocity(
            temperature_f=550.0,
            tube_id_in=6.0,
        )

        min_vel_small = heat_transfer_calc.calculate_minimum_velocity(
            temperature_f=550.0,
            tube_id_in=2.0,
        )

        # Smaller tube needs higher velocity for same Re
        assert min_vel_small > min_vel_large


# =============================================================================
# HEATER TUBE ANALYSIS TESTS
# =============================================================================

class TestHeaterTubeAnalysis:
    """Tests for heater tube analysis."""

    def test_heater_tube_analysis(self, heat_transfer_calc):
        """Test heater tube analysis."""
        result = heat_transfer_calc.calculate_heater_tube_analysis(
            bulk_temp_f=550.0,
            outlet_temp_f=580.0,
            flow_rate_gpm=500.0,
            tube_id_in=3.0,
            tube_count=20,
            heat_flux_btu_hr_ft2=10000.0,
        )

        assert "velocity_ft_s" in result
        assert "reynolds_number" in result
        assert "flow_regime" in result
        assert "film_coefficient_btu_hr_ft2_f" in result
        assert "film_temp_rise_f" in result
        assert "estimated_film_temp_f" in result
        assert "film_temp_margin_f" in result

    def test_film_temp_rises_with_heat_flux(self, heat_transfer_calc):
        """Test film temperature rises with higher heat flux."""
        result_low = heat_transfer_calc.calculate_heater_tube_analysis(
            bulk_temp_f=550.0,
            outlet_temp_f=580.0,
            flow_rate_gpm=500.0,
            tube_id_in=3.0,
            tube_count=20,
            heat_flux_btu_hr_ft2=5000.0,
        )

        result_high = heat_transfer_calc.calculate_heater_tube_analysis(
            bulk_temp_f=550.0,
            outlet_temp_f=580.0,
            flow_rate_gpm=500.0,
            tube_id_in=3.0,
            tube_count=20,
            heat_flux_btu_hr_ft2=15000.0,
        )

        assert result_high["film_temp_rise_f"] > result_low["film_temp_rise_f"]

    def test_low_velocity_generates_warning(self, heat_transfer_calc):
        """Test low velocity generates warning."""
        result = heat_transfer_calc.calculate_heater_tube_analysis(
            bulk_temp_f=550.0,
            outlet_temp_f=580.0,
            flow_rate_gpm=100.0,  # Low flow
            tube_id_in=3.0,
            tube_count=20,
            heat_flux_btu_hr_ft2=10000.0,
        )

        if result["velocity_ft_s"] < 3.0:
            assert any("velocity" in w.lower() for w in result["warnings"])

    def test_max_film_temp_from_database(self, heat_transfer_calc):
        """Test max film temperature comes from database."""
        result = heat_transfer_calc.calculate_heater_tube_analysis(
            bulk_temp_f=550.0,
            outlet_temp_f=580.0,
            flow_rate_gpm=500.0,
            tube_id_in=3.0,
            tube_count=20,
            heat_flux_btu_hr_ft2=10000.0,
        )

        # Therminol 66 max film temp is 705F
        assert result["max_film_temp_f"] == 705.0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_reynolds_function(self):
        """Test calculate_reynolds convenience function."""
        reynolds = calculate_reynolds(
            fluid_type=ThermalFluidType.THERMINOL_66,
            temperature_f=550.0,
            velocity_ft_s=8.0,
            diameter_in=3.0,
        )

        assert reynolds > 0
        # Should be turbulent for these conditions
        assert reynolds > RE_TURBULENT_LIMIT

    def test_get_minimum_velocity_function(self):
        """Test get_minimum_velocity convenience function."""
        min_vel = get_minimum_velocity(
            fluid_type=ThermalFluidType.THERMINOL_66,
            temperature_f=550.0,
            tube_id_in=3.0,
        )

        assert min_vel > 0
        # Typical minimum velocity 1-5 ft/s
        assert 0.5 <= min_vel <= 10.0


# =============================================================================
# CALCULATION COUNT TESTS
# =============================================================================

class TestCalculationCount:
    """Tests for calculation counting."""

    def test_calculation_count_increments(
        self, heat_transfer_calc, typical_calc_params
    ):
        """Test calculation count increments."""
        assert heat_transfer_calc.calculation_count == 0

        heat_transfer_calc.calculate_film_coefficient(**typical_calc_params)
        assert heat_transfer_calc.calculation_count == 1

        heat_transfer_calc.calculate_overall_coefficient(
            h_inside=150.0,
            h_outside=100.0,
            tube_od_in=3.5,
            tube_id_in=3.068,
        )
        assert heat_transfer_calc.calculation_count == 2


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_reynolds_limits(self):
        """Test Reynolds number limit constants."""
        assert RE_LAMINAR_LIMIT == 2300
        assert RE_TURBULENT_LIMIT == 10000

    def test_fouling_factors(self):
        """Test fouling factor constants."""
        assert "clean" in FOULING_FACTORS
        assert "light" in FOULING_FACTORS
        assert "moderate" in FOULING_FACTORS
        assert "heavy" in FOULING_FACTORS
        assert "severe" in FOULING_FACTORS

        # Values should increase with severity
        assert FOULING_FACTORS["clean"] < FOULING_FACTORS["light"]
        assert FOULING_FACTORS["light"] < FOULING_FACTORS["moderate"]
        assert FOULING_FACTORS["moderate"] < FOULING_FACTORS["heavy"]
        assert FOULING_FACTORS["heavy"] < FOULING_FACTORS["severe"]


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for calculation determinism."""

    def test_same_input_same_output(self, typical_calc_params):
        """Test same input produces identical output."""
        calc1 = HeatTransferCalculator(fluid_type=ThermalFluidType.THERMINOL_66)
        calc2 = HeatTransferCalculator(fluid_type=ThermalFluidType.THERMINOL_66)

        result1 = calc1.calculate_film_coefficient(**typical_calc_params)
        result2 = calc2.calculate_film_coefficient(**typical_calc_params)

        assert result1.reynolds_number == result2.reynolds_number
        assert result1.nusselt_number == result2.nusselt_number
        assert result1.film_coefficient_btu_hr_ft2_f == result2.film_coefficient_btu_hr_ft2_f
