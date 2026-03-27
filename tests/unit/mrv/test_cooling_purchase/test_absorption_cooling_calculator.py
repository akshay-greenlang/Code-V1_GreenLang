"""
test_absorption_cooling_calculator.py - Tests for AbsorptionCoolingCalculatorEngine

Tests absorption chiller emission calculations for AGENT-MRV-012 (Cooling Purchase Agent).
Validates single/double/triple effect, ammonia, heat input, parasitic electricity, and
zero-emission heat sources.

Test Coverage:
- Singleton pattern
- Single effect calculation (COP 0.70)
- Double effect calculation (COP 1.20)
- Triple effect calculation (COP 1.60)
- Ammonia absorption (COP 0.55)
- Heat input calculation (GJ, kWh)
- Parasitic electricity calculation
- Default parasitic ratios per type
- Thermal emissions calculation
- Parasitic emissions calculation
- Heat source emission factors (11 sources)
- Zero-emission sources (waste heat, solar, geothermal)
- Waste heat path (only parasitic emissions)
- CHP heat emission factor resolution
- Hybrid plant aggregation
- Emission decomposition (gas breakdown)
- Provenance tracking
"""

import pytest
from decimal import Decimal
from typing import Dict, Any

try:
    from greenlang.agents.mrv.cooling_purchase.absorption_cooling_calculator import (
        AbsorptionCoolingCalculatorEngine,
    )
except ImportError:
    pytest.skip("cooling_purchase not available", allow_module_level=True)


@pytest.fixture
def calculator():
    """Fresh calculator engine instance for each test."""
    calc = AbsorptionCoolingCalculatorEngine()
    calc.reset()
    return calc


class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_singleton_instance(self, calculator):
        """Test singleton returns same instance."""
        calc1 = AbsorptionCoolingCalculatorEngine()
        calc2 = AbsorptionCoolingCalculatorEngine()
        assert calc1 is calc2

    def test_reset_clears_state(self, calculator):
        """Test reset clears internal state."""
        calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        calculator.reset()

        result = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )
        assert result is not None


class TestSingleEffectCalculation:
    """Test calculate_single_effect() - COP 0.70."""

    def test_calculate_single_effect_basic(self, calculator):
        """Test single effect calculation: heat_input = cooling/0.70."""
        result = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        # Heat input = 1000 / 0.70 = 1428.57 kWh = 5.143 GJ
        # Thermal emissions = 5.143 * 70.1 = 360.5 kgCO2e
        # Parasitic = 1000 * 0.04 = 40 kWh, 40 * 0.5 = 20 kgCO2e
        # Total = 360.5 + 20 = 380.5 kgCO2e

        assert "total_emissions_kg_co2e" in result
        assert result["total_emissions_kg_co2e"] > Decimal("300.0")

    def test_calculate_single_effect_cop_070(self, calculator):
        """Test single effect uses COP 0.70."""
        result = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("700.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        # Heat input = 700 / 0.70 = 1000 kWh
        expected_heat_kwh = Decimal("700.0") / Decimal("0.70")
        # Convert to GJ: 1000 kWh * 0.0036 = 3.6 GJ
        expected_heat_gj = expected_heat_kwh * Decimal("0.0036")

        assert "heat_input_gj" in result
        assert abs(result["heat_input_gj"] - expected_heat_gj) < Decimal("0.1")

    def test_calculate_single_effect_thermal_emissions(self, calculator):
        """Test thermal emissions = heat_input_gj * heat_source_ef."""
        result = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
            parasitic_ratio=Decimal("0.0"),  # No parasitic for this test
        )

        # Heat input = 1000 / 0.70 = 1428.57 kWh = 5.143 GJ
        # Thermal = 5.143 * 70.1 = 360.5 kgCO2e
        heat_gj = (Decimal("1000.0") / Decimal("0.70")) * Decimal("0.0036")
        expected_thermal = heat_gj * Decimal("70.1")

        assert "thermal_emissions_kg_co2e" in result
        assert abs(result["thermal_emissions_kg_co2e"] - expected_thermal) < Decimal("5.0")


class TestDoubleEffectCalculation:
    """Test calculate_double_effect() - COP 1.20."""

    def test_calculate_double_effect_basic(self, calculator):
        """Test double effect calculation: heat_input = cooling/1.20."""
        result = calculator.calculate_double_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert "total_emissions_kg_co2e" in result
        assert result["total_emissions_kg_co2e"] > Decimal("0.0")

    def test_calculate_double_effect_cop_120(self, calculator):
        """Test double effect uses COP 1.20."""
        result = calculator.calculate_double_effect(
            cooling_output_kwh_th=Decimal("1200.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        # Heat input = 1200 / 1.20 = 1000 kWh = 3.6 GJ
        expected_heat_gj = (Decimal("1200.0") / Decimal("1.20")) * Decimal("0.0036")

        assert abs(result["heat_input_gj"] - expected_heat_gj) < Decimal("0.1")

    def test_calculate_double_effect_lower_emissions_than_single(self, calculator):
        """Test double effect produces lower emissions than single effect."""
        single = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
            parasitic_ratio=Decimal("0.05"),
        )

        double = calculator.calculate_double_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
            parasitic_ratio=Decimal("0.05"),
        )

        # Double effect is more efficient (higher COP), so lower emissions
        assert double["total_emissions_kg_co2e"] < single["total_emissions_kg_co2e"]


class TestTripleEffectCalculation:
    """Test calculate_triple_effect() - COP 1.60."""

    def test_calculate_triple_effect_basic(self, calculator):
        """Test triple effect calculation: heat_input = cooling/1.60."""
        result = calculator.calculate_triple_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert "total_emissions_kg_co2e" in result
        assert result["total_emissions_kg_co2e"] > Decimal("0.0")

    def test_calculate_triple_effect_cop_160(self, calculator):
        """Test triple effect uses COP 1.60."""
        result = calculator.calculate_triple_effect(
            cooling_output_kwh_th=Decimal("1600.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        # Heat input = 1600 / 1.60 = 1000 kWh = 3.6 GJ
        expected_heat_gj = (Decimal("1600.0") / Decimal("1.60")) * Decimal("0.0036")

        assert abs(result["heat_input_gj"] - expected_heat_gj) < Decimal("0.1")

    def test_calculate_triple_effect_lowest_emissions(self, calculator):
        """Test triple effect produces lowest emissions (highest efficiency)."""
        single = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        double = calculator.calculate_double_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        triple = calculator.calculate_triple_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        assert triple["total_emissions_kg_co2e"] < double["total_emissions_kg_co2e"]
        assert double["total_emissions_kg_co2e"] < single["total_emissions_kg_co2e"]


class TestAmmoniaAbsorption:
    """Test calculate_ammonia() - COP 0.55."""

    def test_calculate_ammonia_basic(self, calculator):
        """Test ammonia absorption calculation: heat_input = cooling/0.55."""
        result = calculator.calculate_ammonia(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert "total_emissions_kg_co2e" in result
        assert result["total_emissions_kg_co2e"] > Decimal("0.0")

    def test_calculate_ammonia_cop_055(self, calculator):
        """Test ammonia uses COP 0.55."""
        result = calculator.calculate_ammonia(
            cooling_output_kwh_th=Decimal("550.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        # Heat input = 550 / 0.55 = 1000 kWh = 3.6 GJ
        expected_heat_gj = (Decimal("550.0") / Decimal("0.55")) * Decimal("0.0036")

        assert abs(result["heat_input_gj"] - expected_heat_gj) < Decimal("0.1")

    def test_calculate_ammonia_highest_emissions(self, calculator):
        """Test ammonia produces highest emissions (lowest COP)."""
        ammonia = calculator.calculate_ammonia(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        single = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        # Ammonia (COP 0.55) < Single effect (COP 0.70)
        assert ammonia["total_emissions_kg_co2e"] > single["total_emissions_kg_co2e"]


class TestHeatInputCalculation:
    """Test heat input calculations."""

    def test_calculate_heat_input_gj(self, calculator):
        """Test calculate_heat_input_gj()."""
        heat_gj = calculator.calculate_heat_input_gj(
            cooling_output_kwh_th=Decimal("1000.0"), cop=Decimal("0.70")
        )

        # Heat = 1000 / 0.70 = 1428.57 kWh
        # GJ = 1428.57 * 0.0036 = 5.143 GJ
        expected = (Decimal("1000.0") / Decimal("0.70")) * Decimal("0.0036")

        assert abs(heat_gj - expected) < Decimal("0.1")

    def test_calculate_heat_input_kwh(self, calculator):
        """Test calculate_heat_input_kwh()."""
        heat_kwh = calculator.calculate_heat_input_kwh(
            cooling_output_kwh_th=Decimal("1000.0"), cop=Decimal("0.70")
        )

        # Heat = 1000 / 0.70 = 1428.57 kWh
        expected = Decimal("1000.0") / Decimal("0.70")

        assert abs(heat_kwh - expected) < Decimal("0.1")

    def test_calculate_heat_input_gj_kwh_conversion(self, calculator):
        """Test GJ and kWh heat input conversion."""
        heat_gj = calculator.calculate_heat_input_gj(
            cooling_output_kwh_th=Decimal("1000.0"), cop=Decimal("1.0")
        )

        heat_kwh = calculator.calculate_heat_input_kwh(
            cooling_output_kwh_th=Decimal("1000.0"), cop=Decimal("1.0")
        )

        # 1 kWh = 0.0036 GJ
        assert abs(heat_gj - heat_kwh * Decimal("0.0036")) < Decimal("0.001")


class TestParasiticElectricity:
    """Test parasitic electricity calculation."""

    def test_calculate_parasitic_electricity(self, calculator):
        """Test calculate_parasitic_electricity()."""
        parasitic = calculator.calculate_parasitic_electricity(
            cooling_output_kwh_th=Decimal("1000.0"), parasitic_ratio=Decimal("0.05")
        )

        # Parasitic = 1000 * 0.05 = 50 kWh
        expected = Decimal("1000.0") * Decimal("0.05")

        assert abs(parasitic - expected) < Decimal("0.1")

    def test_calculate_parasitic_electricity_zero_ratio(self, calculator):
        """Test parasitic electricity with zero ratio."""
        parasitic = calculator.calculate_parasitic_electricity(
            cooling_output_kwh_th=Decimal("1000.0"), parasitic_ratio=Decimal("0.0")
        )

        assert parasitic == Decimal("0.0")

    def test_get_default_parasitic_ratio_single_effect(self, calculator):
        """Test get_default_parasitic_ratio for single_effect (0.04)."""
        ratio = calculator.get_default_parasitic_ratio("single_effect")
        assert ratio == Decimal("0.04")

    def test_get_default_parasitic_ratio_double_effect(self, calculator):
        """Test get_default_parasitic_ratio for double_effect (0.05)."""
        ratio = calculator.get_default_parasitic_ratio("double_effect")
        assert ratio == Decimal("0.05")

    def test_get_default_parasitic_ratio_triple_effect(self, calculator):
        """Test get_default_parasitic_ratio for triple_effect (0.06)."""
        ratio = calculator.get_default_parasitic_ratio("triple_effect")
        assert ratio == Decimal("0.06")

    def test_get_default_parasitic_ratio_ammonia(self, calculator):
        """Test get_default_parasitic_ratio for ammonia (0.08)."""
        ratio = calculator.get_default_parasitic_ratio("ammonia")
        assert ratio == Decimal("0.08")


class TestThermalEmissionsCalculation:
    """Test thermal emissions calculation."""

    def test_calculate_thermal_emissions(self, calculator):
        """Test calculate_thermal_emissions()."""
        thermal = calculator.calculate_thermal_emissions(
            heat_input_gj=Decimal("5.0"), heat_source_ef_kg_co2e_per_gj=Decimal("70.1")
        )

        # Thermal = 5.0 * 70.1 = 350.5 kgCO2e
        expected = Decimal("5.0") * Decimal("70.1")

        assert abs(thermal - expected) < Decimal("0.1")

    def test_calculate_thermal_emissions_zero_ef(self, calculator):
        """Test thermal emissions with zero emission factor (waste heat)."""
        thermal = calculator.calculate_thermal_emissions(
            heat_input_gj=Decimal("5.0"), heat_source_ef_kg_co2e_per_gj=Decimal("0.0")
        )

        assert thermal == Decimal("0.0")


class TestParasiticEmissionsCalculation:
    """Test parasitic emissions calculation."""

    def test_calculate_parasitic_emissions(self, calculator):
        """Test calculate_parasitic_emissions()."""
        parasitic = calculator.calculate_parasitic_emissions(
            parasitic_electricity_kwh=Decimal("50.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        # Parasitic = 50 * 0.5 = 25 kgCO2e
        expected = Decimal("50.0") * Decimal("0.5")

        assert abs(parasitic - expected) < Decimal("0.1")

    def test_calculate_parasitic_emissions_zero_grid_ef(self, calculator):
        """Test parasitic emissions with zero grid emission factor."""
        parasitic = calculator.calculate_parasitic_emissions(
            parasitic_electricity_kwh=Decimal("50.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.0"),
        )

        assert parasitic == Decimal("0.0")


class TestHeatSourceEmissionFactors:
    """Test heat source emission factors for 11 sources."""

    def test_get_heat_source_ef_natural_gas(self, calculator):
        """Test get_heat_source_ef for natural_gas."""
        ef = calculator.get_heat_source_ef("natural_gas")
        assert ef == Decimal("70.1")  # Typical natural gas EF

    def test_get_heat_source_ef_waste_heat(self, calculator):
        """Test get_heat_source_ef for waste_heat."""
        ef = calculator.get_heat_source_ef("waste_heat")
        assert ef == Decimal("0.0")

    def test_get_heat_source_ef_chp(self, calculator):
        """Test get_heat_source_ef for chp."""
        ef = calculator.get_heat_source_ef("chp")
        assert isinstance(ef, Decimal)
        assert ef > Decimal("0.0")

    def test_get_heat_source_ef_steam(self, calculator):
        """Test get_heat_source_ef for steam."""
        ef = calculator.get_heat_source_ef("steam")
        assert isinstance(ef, Decimal)

    def test_get_heat_source_ef_hot_water(self, calculator):
        """Test get_heat_source_ef for hot_water."""
        ef = calculator.get_heat_source_ef("hot_water")
        assert isinstance(ef, Decimal)

    def test_get_heat_source_ef_biomass(self, calculator):
        """Test get_heat_source_ef for biomass."""
        ef = calculator.get_heat_source_ef("biomass")
        assert isinstance(ef, Decimal)

    def test_get_heat_source_ef_solar_thermal(self, calculator):
        """Test get_heat_source_ef for solar_thermal."""
        ef = calculator.get_heat_source_ef("solar_thermal")
        assert ef == Decimal("0.0")

    def test_get_heat_source_ef_geothermal(self, calculator):
        """Test get_heat_source_ef for geothermal."""
        ef = calculator.get_heat_source_ef("geothermal")
        # Geothermal can be zero or very low
        assert ef <= Decimal("10.0")

    def test_get_heat_source_ef_district_heat(self, calculator):
        """Test get_heat_source_ef for district_heat."""
        ef = calculator.get_heat_source_ef("district_heat")
        assert isinstance(ef, Decimal)

    def test_get_heat_source_ef_oil(self, calculator):
        """Test get_heat_source_ef for oil."""
        ef = calculator.get_heat_source_ef("oil")
        assert ef > Decimal("50.0")  # Oil has high EF

    def test_get_heat_source_ef_electric_resistance(self, calculator):
        """Test get_heat_source_ef for electric_resistance."""
        ef = calculator.get_heat_source_ef("electric_resistance")
        assert isinstance(ef, Decimal)


class TestZeroEmissionSources:
    """Test zero-emission heat sources."""

    def test_is_zero_emission_source_waste_heat(self, calculator):
        """Test is_zero_emission_source for waste_heat."""
        assert calculator.is_zero_emission_source("waste_heat") is True

    def test_is_zero_emission_source_solar(self, calculator):
        """Test is_zero_emission_source for solar_thermal."""
        assert calculator.is_zero_emission_source("solar_thermal") is True

    def test_is_zero_emission_source_geothermal(self, calculator):
        """Test is_zero_emission_source for geothermal."""
        assert calculator.is_zero_emission_source("geothermal") is True

    def test_is_zero_emission_source_natural_gas_false(self, calculator):
        """Test is_zero_emission_source returns False for natural_gas."""
        assert calculator.is_zero_emission_source("natural_gas") is False

    def test_is_zero_emission_source_biomass_false(self, calculator):
        """Test is_zero_emission_source returns False for biomass."""
        assert calculator.is_zero_emission_source("biomass") is False


class TestWasteHeatPath:
    """Test waste heat path (only parasitic emissions, thermal=0)."""

    def test_waste_heat_zero_thermal_emissions(self, calculator):
        """Test waste heat produces zero thermal emissions."""
        result = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("0.0"),  # Waste heat
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert result["thermal_emissions_kg_co2e"] == Decimal("0.0")

    def test_waste_heat_only_parasitic_emissions(self, calculator):
        """Test waste heat only produces parasitic emissions."""
        result = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("0.0"),  # Waste heat
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
            parasitic_ratio=Decimal("0.04"),
        )

        # Parasitic = 1000 * 0.04 = 40 kWh, 40 * 0.5 = 20 kgCO2e
        expected_parasitic = Decimal("1000.0") * Decimal("0.04") * Decimal("0.5")

        assert result["thermal_emissions_kg_co2e"] == Decimal("0.0")
        assert abs(result["parasitic_emissions_kg_co2e"] - expected_parasitic) < Decimal("0.1")
        assert result["total_emissions_kg_co2e"] == result["parasitic_emissions_kg_co2e"]

    def test_waste_heat_much_lower_emissions(self, calculator):
        """Test waste heat produces much lower emissions than natural gas."""
        waste_heat = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("0.0"),  # Waste heat
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        natural_gas = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),  # Natural gas
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert waste_heat["total_emissions_kg_co2e"] < Decimal("50.0")
        assert natural_gas["total_emissions_kg_co2e"] > Decimal("300.0")


class TestCHPHeatEmissionFactor:
    """Test CHP heat emission factor resolution."""

    def test_resolve_chp_heat_ef_default(self, calculator):
        """Test resolve_chp_heat_ef() returns default 70.0."""
        ef = calculator.resolve_chp_heat_ef()
        assert ef == Decimal("70.0")  # Default CHP allocation

    def test_resolve_chp_heat_ef_custom(self, calculator):
        """Test resolve_chp_heat_ef() with custom value."""
        ef = calculator.resolve_chp_heat_ef(custom_ef=Decimal("50.0"))
        assert ef == Decimal("50.0")

    def test_calculate_with_chp_heat(self, calculator):
        """Test calculation with CHP heat source."""
        result = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source="chp",
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert "total_emissions_kg_co2e" in result
        assert result["total_emissions_kg_co2e"] > Decimal("0.0")


class TestHybridPlantAggregation:
    """Test calculate_hybrid_plant() aggregation."""

    def test_calculate_hybrid_plant_basic(self, calculator):
        """Test hybrid plant aggregation."""
        chillers = [
            {
                "cooling_output_kwh_th": Decimal("1000.0"),
                "technology": "single_effect",
                "heat_source_ef_kg_co2e_per_gj": Decimal("70.1"),
            },
            {
                "cooling_output_kwh_th": Decimal("500.0"),
                "technology": "double_effect",
                "heat_source_ef_kg_co2e_per_gj": Decimal("70.1"),
            },
        ]

        result = calculator.calculate_hybrid_plant(
            chillers, grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5")
        )

        assert "total_emissions_kg_co2e" in result
        assert "total_cooling_kwh_th" in result
        assert result["total_cooling_kwh_th"] == Decimal("1500.0")

    def test_calculate_hybrid_plant_aggregates_emissions(self, calculator):
        """Test hybrid plant aggregates emissions correctly."""
        chillers = [
            {
                "cooling_output_kwh_th": Decimal("1000.0"),
                "technology": "single_effect",
                "heat_source_ef_kg_co2e_per_gj": Decimal("0.0"),  # Waste heat
            },
            {
                "cooling_output_kwh_th": Decimal("1000.0"),
                "technology": "single_effect",
                "heat_source_ef_kg_co2e_per_gj": Decimal("70.1"),  # Natural gas
            },
        ]

        result = calculator.calculate_hybrid_plant(
            chillers, grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5")
        )

        # Should be sum of both chillers
        assert result["total_emissions_kg_co2e"] > Decimal("300.0")

    def test_calculate_hybrid_plant_mixed_technologies(self, calculator):
        """Test hybrid plant with mixed technologies."""
        chillers = [
            {
                "cooling_output_kwh_th": Decimal("1000.0"),
                "technology": "single_effect",
                "heat_source_ef_kg_co2e_per_gj": Decimal("70.1"),
            },
            {
                "cooling_output_kwh_th": Decimal("1000.0"),
                "technology": "triple_effect",
                "heat_source_ef_kg_co2e_per_gj": Decimal("70.1"),
            },
        ]

        result = calculator.calculate_hybrid_plant(
            chillers, grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5")
        )

        # Triple effect should have lower emissions, so total should be between
        # 2x single effect and 2x triple effect
        single_x2_est = Decimal("700.0")
        triple_x2_est = Decimal("400.0")

        assert triple_x2_est < result["total_emissions_kg_co2e"] < single_x2_est


class TestEmissionDecomposition:
    """Test decompose_emissions() produces gas breakdown."""

    def test_decompose_emissions_returns_dict(self, calculator):
        """Test decompose_emissions returns dict with gas breakdown."""
        decomposed = calculator.decompose_emissions(
            total_emissions_kg_co2e=Decimal("100.0"), heat_source="natural_gas"
        )

        assert isinstance(decomposed, dict)
        assert "co2_kg" in decomposed
        assert "ch4_kg" in decomposed
        assert "n2o_kg" in decomposed

    def test_decompose_emissions_natural_gas_breakdown(self, calculator):
        """Test natural gas emission decomposition."""
        decomposed = calculator.decompose_emissions(
            total_emissions_kg_co2e=Decimal("100.0"), heat_source="natural_gas"
        )

        # Natural gas: ~95% CO2, ~4% CH4 (in CO2e), ~1% N2O (in CO2e)
        assert decomposed["co2_kg"] > Decimal("85.0")
        assert decomposed["ch4_kg"] < Decimal("1.0")  # Mass, not CO2e
        assert decomposed["n2o_kg"] < Decimal("0.1")  # Mass, not CO2e

    def test_decompose_emissions_sum_equals_total(self, calculator):
        """Test sum of gases equals total emissions (in CO2e)."""
        total = Decimal("100.0")
        decomposed = calculator.decompose_emissions(
            total_emissions_kg_co2e=total, heat_source="natural_gas"
        )

        # Convert to CO2e and sum
        # CH4 GWP = 25, N2O GWP = 298
        co2e_sum = (
            decomposed["co2_kg"]
            + decomposed["ch4_kg"] * Decimal("25")
            + decomposed["n2o_kg"] * Decimal("298")
        )

        assert abs(co2e_sum - total) < Decimal("1.0")


class TestProvenanceTracking:
    """Test provenance hash tracking."""

    def test_result_has_provenance_hash(self, calculator):
        """Test result has provenance_hash field."""
        result = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        assert "provenance_hash" in result
        assert isinstance(result["provenance_hash"], str)
        assert len(result["provenance_hash"]) == 64  # SHA-256 hex

    def test_provenance_hash_is_64_char_hex(self, calculator):
        """Test provenance_hash is 64-character hex string."""
        result = calculator.calculate_double_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        hash_value = result["provenance_hash"]
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value.lower())

    def test_result_has_trace_steps(self, calculator):
        """Test result has trace_steps field."""
        result = calculator.calculate_triple_effect(
            cooling_output_kwh_th=Decimal("1000.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        assert "trace_steps" in result
        assert isinstance(result["trace_steps"], list)
        assert len(result["trace_steps"]) > 0


class TestZeroEmissions:
    """Test zero cooling output returns zero emissions."""

    def test_zero_cooling_output_zero_emissions(self, calculator):
        """Test zero cooling output returns zero emissions."""
        result = calculator.calculate_single_effect(
            cooling_output_kwh_th=Decimal("0.0"),
            heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
        )

        assert result["total_emissions_kg_co2e"] == Decimal("0.0")
        assert result["heat_input_gj"] == Decimal("0.0")


class TestNegativeValueValidation:
    """Test negative values raise errors."""

    def test_negative_cooling_raises_error(self, calculator):
        """Test negative cooling output raises error."""
        with pytest.raises(ValueError, match="negative|positive"):
            calculator.calculate_single_effect(
                cooling_output_kwh_th=Decimal("-1000.0"),
                heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
            )

    def test_negative_heat_ef_raises_error(self, calculator):
        """Test negative heat emission factor raises error."""
        with pytest.raises(ValueError, match="negative"):
            calculator.calculate_single_effect(
                cooling_output_kwh_th=Decimal("1000.0"),
                heat_source_ef_kg_co2e_per_gj=Decimal("-70.1"),
            )

    def test_negative_grid_ef_raises_error(self, calculator):
        """Test negative grid emission factor raises error."""
        with pytest.raises(ValueError, match="negative"):
            calculator.calculate_single_effect(
                cooling_output_kwh_th=Decimal("1000.0"),
                heat_source_ef_kg_co2e_per_gj=Decimal("70.1"),
                grid_emission_factor_kg_co2e_per_kwh=Decimal("-0.5"),
            )


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_heat_source_raises_error(self, calculator):
        """Test invalid heat source raises error."""
        with pytest.raises((KeyError, ValueError)):
            calculator.get_heat_source_ef("invalid_source")

    def test_invalid_technology_raises_error(self, calculator):
        """Test invalid technology raises error."""
        with pytest.raises((KeyError, ValueError)):
            calculator.get_default_parasitic_ratio("invalid_technology")
