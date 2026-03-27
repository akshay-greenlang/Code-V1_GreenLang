"""
test_electric_chiller_calculator.py - Tests for ElectricChillerCalculatorEngine

Tests electric chiller emission calculations for AGENT-MRV-012 (Cooling Purchase Agent).
Validates full load, IPLV, part load, auxiliary energy, and multi-chiller calculations.

Test Coverage:
- Singleton pattern
- Full load calculation (E = cooling/COP * grid_ef)
- IPLV weighted calculation
- Custom part load calculation
- IPLV calculation with AHRI weights
- Part load COP estimation
- Auxiliary energy calculation
- Total emissions with auxiliary
- Grid emission decomposition (CO2/CH4/N2O)
- COP adjustments for condenser type
- COP conversions (EER, kW/ton)
- Multi-chiller aggregation
- Weighted COP calculation
- Provenance tracking
"""

import pytest
from decimal import Decimal
from typing import Dict, Any

try:
    from greenlang.agents.mrv.cooling_purchase.electric_chiller_calculator import (
        ElectricChillerCalculatorEngine,
    )
except ImportError:
    pytest.skip("cooling_purchase not available", allow_module_level=True)


@pytest.fixture
def calculator():
    """Fresh calculator engine instance for each test."""
    calc = ElectricChillerCalculatorEngine()
    calc.reset()
    return calc


class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_singleton_instance(self, calculator):
        """Test singleton returns same instance."""
        calc1 = ElectricChillerCalculatorEngine()
        calc2 = ElectricChillerCalculatorEngine()
        assert calc1 is calc2

    def test_reset_clears_state(self, calculator):
        """Test reset clears internal state."""
        # Perform calculation
        calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        calculator.reset()
        # After reset, should be able to calculate again
        result = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )
        assert result is not None


class TestFullLoadCalculation:
    """Test calculate_full_load() - E = cooling/COP * grid_ef."""

    def test_calculate_full_load_basic(self, calculator):
        """Test basic full load calculation."""
        result = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.1"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        # Energy = 1000 / 6.1 = 163.93 kWh
        # Emissions = 163.93 * 0.5 = 81.97 kgCO2e
        expected_energy = Decimal("1000.0") / Decimal("6.1")
        expected_emissions = expected_energy * Decimal("0.5")

        assert abs(result["energy_consumption_kwh"] - expected_energy) < Decimal("0.1")
        assert abs(result["total_emissions_kg_co2e"] - expected_emissions) < Decimal("0.1")

    def test_calculate_full_load_returns_dict(self, calculator):
        """Test calculate_full_load returns dict with required fields."""
        result = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("500.0"),
            cop=Decimal("5.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.4"),
        )

        assert isinstance(result, dict)
        assert "total_emissions_kg_co2e" in result
        assert "energy_consumption_kwh" in result

    def test_calculate_full_load_high_cop(self, calculator):
        """Test full load with high COP (more efficient)."""
        result = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("8.0"),  # High efficiency
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        # Energy = 1000 / 8.0 = 125 kWh
        # Emissions = 125 * 0.5 = 62.5 kgCO2e
        expected_emissions = (Decimal("1000.0") / Decimal("8.0")) * Decimal("0.5")
        assert abs(result["total_emissions_kg_co2e"] - expected_emissions) < Decimal("0.1")

    def test_calculate_full_load_low_cop(self, calculator):
        """Test full load with low COP (less efficient)."""
        result = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("3.0"),  # Low efficiency
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        # Energy = 1000 / 3.0 = 333.33 kWh
        # Emissions = 333.33 * 0.5 = 166.67 kgCO2e
        expected_emissions = (Decimal("1000.0") / Decimal("3.0")) * Decimal("0.5")
        assert abs(result["total_emissions_kg_co2e"] - expected_emissions) < Decimal("0.5")

    def test_calculate_full_load_zero_grid_ef(self, calculator):
        """Test full load with zero grid emission factor (renewable grid)."""
        result = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.0"),
        )

        assert result["total_emissions_kg_co2e"] == Decimal("0.0")


class TestIPLVWeightedCalculation:
    """Test calculate_iplv_weighted() - E = cooling/IPLV * grid_ef."""

    def test_calculate_iplv_weighted_basic(self, calculator):
        """Test IPLV weighted calculation."""
        result = calculator.calculate_iplv_weighted(
            cooling_output_kwh_th=Decimal("1000.0"),
            iplv=Decimal("7.2"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        # Energy = 1000 / 7.2 = 138.89 kWh
        # Emissions = 138.89 * 0.5 = 69.44 kgCO2e
        expected_energy = Decimal("1000.0") / Decimal("7.2")
        expected_emissions = expected_energy * Decimal("0.5")

        assert abs(result["energy_consumption_kwh"] - expected_energy) < Decimal("0.1")
        assert abs(result["total_emissions_kg_co2e"] - expected_emissions) < Decimal("0.1")

    def test_calculate_iplv_weighted_lower_emissions(self, calculator):
        """Test IPLV method produces lower emissions than full load."""
        full_load = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        iplv_weighted = calculator.calculate_iplv_weighted(
            cooling_output_kwh_th=Decimal("1000.0"),
            iplv=Decimal("7.2"),  # IPLV > COP
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert iplv_weighted["total_emissions_kg_co2e"] < full_load["total_emissions_kg_co2e"]


class TestCustomPartLoadCalculation:
    """Test calculate_custom_part_load() with 4 custom COPs."""

    def test_calculate_custom_part_load_basic(self, calculator):
        """Test custom part load calculation."""
        result = calculator.calculate_custom_part_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop_100=Decimal("6.0"),
            cop_75=Decimal("7.0"),
            cop_50=Decimal("8.0"),
            cop_25=Decimal("9.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert "total_emissions_kg_co2e" in result
        assert result["total_emissions_kg_co2e"] > Decimal("0.0")

    def test_calculate_custom_part_load_uses_iplv_formula(self, calculator):
        """Test custom part load uses IPLV weighting formula."""
        cops = {
            "cop_100": Decimal("6.0"),
            "cop_75": Decimal("7.0"),
            "cop_50": Decimal("8.0"),
            "cop_25": Decimal("9.0"),
        }

        result = calculator.calculate_custom_part_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
            **cops,
        )

        # Calculate expected IPLV
        iplv = (
            Decimal("0.01") * cops["cop_100"]
            + Decimal("0.42") * cops["cop_75"]
            + Decimal("0.45") * cops["cop_50"]
            + Decimal("0.12") * cops["cop_25"]
        )

        expected_energy = Decimal("1000.0") / iplv
        expected_emissions = expected_energy * Decimal("0.5")

        assert abs(result["total_emissions_kg_co2e"] - expected_emissions) < Decimal("1.0")


class TestIPLVCalculation:
    """Test calculate_iplv() with AHRI weights."""

    def test_calculate_iplv_ahri_weights(self, calculator):
        """Test calculate_iplv with AHRI 550/590 weights."""
        iplv = calculator.calculate_iplv(
            cop_100=Decimal("6.0"),
            cop_75=Decimal("7.0"),
            cop_50=Decimal("8.0"),
            cop_25=Decimal("9.0"),
        )

        # IPLV = 0.01*6.0 + 0.42*7.0 + 0.45*8.0 + 0.12*9.0
        expected = (
            Decimal("0.01") * Decimal("6.0")
            + Decimal("0.42") * Decimal("7.0")
            + Decimal("0.45") * Decimal("8.0")
            + Decimal("0.12") * Decimal("9.0")
        )

        assert abs(iplv - expected) < Decimal("0.01")

    def test_calculate_iplv_returns_decimal(self, calculator):
        """Test calculate_iplv returns Decimal."""
        iplv = calculator.calculate_iplv(
            cop_100=Decimal("5.0"),
            cop_75=Decimal("6.0"),
            cop_50=Decimal("7.0"),
            cop_25=Decimal("8.0"),
        )

        assert isinstance(iplv, Decimal)

    def test_calculate_iplv_weighted_toward_mid_load(self, calculator):
        """Test IPLV heavily weighted toward 50% and 75% load."""
        # 42% at 75% load, 45% at 50% load = 87% total
        iplv = calculator.calculate_iplv(
            cop_100=Decimal("5.0"),
            cop_75=Decimal("10.0"),  # High COP at 75%
            cop_50=Decimal("10.0"),  # High COP at 50%
            cop_25=Decimal("5.0"),
        )

        # Should be close to 10.0 due to weighting
        assert iplv > Decimal("8.0")


class TestPartLoadCOPEstimation:
    """Test estimate_part_load_cops() from full-load COP."""

    def test_estimate_part_load_cops_returns_dict(self, calculator):
        """Test estimate_part_load_cops returns dict with 4 COPs."""
        cops = calculator.estimate_part_load_cops(full_load_cop=Decimal("6.0"))

        assert isinstance(cops, dict)
        assert "cop_100" in cops
        assert "cop_75" in cops
        assert "cop_50" in cops
        assert "cop_25" in cops

    def test_estimate_part_load_cops_cop_100_matches_input(self, calculator):
        """Test cop_100 matches input full_load_cop."""
        full_load_cop = Decimal("6.0")
        cops = calculator.estimate_part_load_cops(full_load_cop=full_load_cop)

        assert cops["cop_100"] == full_load_cop

    def test_estimate_part_load_cops_increases_at_part_load(self, calculator):
        """Test COPs increase at part load (typical chiller behavior)."""
        cops = calculator.estimate_part_load_cops(full_load_cop=Decimal("6.0"))

        # Typically COP increases at part load
        assert cops["cop_75"] >= cops["cop_100"]
        assert cops["cop_50"] >= cops["cop_75"]


class TestAuxiliaryEnergy:
    """Test auxiliary energy calculation."""

    def test_calculate_auxiliary_energy_basic(self, calculator):
        """Test auxiliary energy = cooling * auxiliary_pct."""
        aux_energy = calculator.calculate_auxiliary_energy(
            cooling_output_kwh_th=Decimal("1000.0"),
            auxiliary_percent=Decimal("5.0"),
        )

        # Auxiliary = 1000 * 0.05 = 50 kWh
        expected = Decimal("1000.0") * Decimal("0.05") / Decimal("100")
        assert abs(aux_energy - expected) < Decimal("0.1")

    def test_calculate_auxiliary_energy_zero_percent(self, calculator):
        """Test auxiliary energy with 0% auxiliary load."""
        aux_energy = calculator.calculate_auxiliary_energy(
            cooling_output_kwh_th=Decimal("1000.0"),
            auxiliary_percent=Decimal("0.0"),
        )

        assert aux_energy == Decimal("0.0")

    def test_calculate_auxiliary_energy_typical_range(self, calculator):
        """Test auxiliary energy with typical 3-7% range."""
        cooling = Decimal("1000.0")

        aux_3pct = calculator.calculate_auxiliary_energy(cooling, Decimal("3.0"))
        aux_7pct = calculator.calculate_auxiliary_energy(cooling, Decimal("7.0"))

        assert Decimal("20") < aux_3pct < Decimal("40")
        assert Decimal("60") < aux_7pct < Decimal("80")


class TestTotalEmissionsWithAuxiliary:
    """Test total emissions including auxiliary energy."""

    def test_total_emissions_with_auxiliary(self, calculator):
        """Test total = (cooling/COP + auxiliary) * grid_ef."""
        result = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
            auxiliary_percent=Decimal("5.0"),
        )

        # Chiller energy = 1000 / 6.0 = 166.67 kWh
        # Auxiliary = 1000 * 0.05 = 50 kWh
        # Total energy = 166.67 + 50 = 216.67 kWh
        # Emissions = 216.67 * 0.5 = 108.33 kgCO2e

        chiller_energy = Decimal("1000.0") / Decimal("6.0")
        aux_energy = Decimal("1000.0") * Decimal("0.05") / Decimal("100")
        total_energy = chiller_energy + aux_energy
        expected_emissions = total_energy * Decimal("0.5")

        assert abs(result["total_emissions_kg_co2e"] - expected_emissions) < Decimal("1.0")

    def test_auxiliary_increases_emissions(self, calculator):
        """Test auxiliary energy increases total emissions."""
        without_aux = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
            auxiliary_percent=Decimal("0.0"),
        )

        with_aux = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
            auxiliary_percent=Decimal("5.0"),
        )

        assert with_aux["total_emissions_kg_co2e"] > without_aux["total_emissions_kg_co2e"]


class TestGridEmissionDecomposition:
    """Test decompose_grid_emissions() produces CO2/CH4/N2O."""

    def test_decompose_grid_emissions_returns_dict(self, calculator):
        """Test decompose_grid_emissions returns dict with gas breakdown."""
        decomposed = calculator.decompose_grid_emissions(
            total_emissions_kg_co2e=Decimal("100.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert isinstance(decomposed, dict)
        assert "co2_kg" in decomposed
        assert "ch4_kg" in decomposed
        assert "n2o_kg" in decomposed

    def test_decompose_grid_emissions_co2_dominant(self, calculator):
        """Test CO2 is dominant greenhouse gas in grid emissions."""
        decomposed = calculator.decompose_grid_emissions(
            total_emissions_kg_co2e=Decimal("100.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        # CO2 should be ~95-98% of total emissions
        assert decomposed["co2_kg"] > Decimal("90.0")
        assert decomposed["ch4_kg"] < Decimal("5.0")
        assert decomposed["n2o_kg"] < Decimal("5.0")

    def test_decompose_grid_emissions_sum_equals_total(self, calculator):
        """Test sum of gases equals total emissions (in CO2e)."""
        total = Decimal("100.0")
        decomposed = calculator.decompose_grid_emissions(
            total_emissions_kg_co2e=total,
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        # Convert CH4 and N2O to CO2e and sum
        # CH4 GWP = 25 (AR4), N2O GWP = 298 (AR4)
        co2e_sum = (
            decomposed["co2_kg"]
            + decomposed["ch4_kg"] * Decimal("25")
            + decomposed["n2o_kg"] * Decimal("298")
        )

        assert abs(co2e_sum - total) < Decimal("1.0")


class TestCOPAdjustments:
    """Test COP adjustments for condenser type."""

    def test_adjust_cop_for_condenser_air_cooled(self, calculator):
        """Test adjust_cop_for_condenser for air-cooled."""
        base_cop = Decimal("6.0")
        adjusted = calculator.adjust_cop_for_condenser(
            cop=base_cop, condenser_type="air_cooled"
        )

        # Air-cooled typically 10-20% lower than water-cooled
        assert adjusted <= base_cop

    def test_adjust_cop_for_condenser_water_cooled(self, calculator):
        """Test adjust_cop_for_condenser for water-cooled."""
        base_cop = Decimal("6.0")
        adjusted = calculator.adjust_cop_for_condenser(
            cop=base_cop, condenser_type="water_cooled"
        )

        # Water-cooled is baseline (no adjustment or slight improvement)
        assert adjusted >= base_cop * Decimal("0.95")

    def test_adjust_cop_for_condenser_evaporative(self, calculator):
        """Test adjust_cop_for_condenser for evaporative."""
        base_cop = Decimal("6.0")
        adjusted = calculator.adjust_cop_for_condenser(
            cop=base_cop, condenser_type="evaporative"
        )

        # Evaporative typically better than air-cooled
        assert adjusted > Decimal("0.0")


class TestCOPConversions:
    """Test COP conversion methods."""

    def test_convert_to_cop_from_eer(self, calculator):
        """Test convert_to_cop from EER."""
        eer = Decimal("20.0")
        cop = calculator.convert_to_cop(eer, from_unit="EER")

        # COP = EER / 3.412
        expected = eer / Decimal("3.412")
        assert abs(cop - expected) < Decimal("0.1")

    def test_convert_to_cop_from_kw_ton(self, calculator):
        """Test convert_to_cop from kW/ton."""
        kw_ton = Decimal("0.6")
        cop = calculator.convert_to_cop(kw_ton, from_unit="kW/ton")

        # COP = 3.517 / kW_ton
        expected = Decimal("3.517") / kw_ton
        assert abs(cop - expected) < Decimal("0.1")

    def test_convert_to_cop_from_cop_returns_same(self, calculator):
        """Test convert_to_cop with COP input returns same value."""
        cop_input = Decimal("6.0")
        cop_output = calculator.convert_to_cop(cop_input, from_unit="COP")

        assert cop_output == cop_input


class TestMultiChillerAggregation:
    """Test calculate_multi_chiller() aggregation."""

    def test_calculate_multi_chiller_basic(self, calculator):
        """Test multi-chiller aggregation."""
        chillers = [
            {
                "cooling_output_kwh_th": Decimal("1000.0"),
                "cop": Decimal("6.0"),
                "grid_emission_factor_kg_co2e_per_kwh": Decimal("0.5"),
            },
            {
                "cooling_output_kwh_th": Decimal("500.0"),
                "cop": Decimal("7.0"),
                "grid_emission_factor_kg_co2e_per_kwh": Decimal("0.5"),
            },
        ]

        result = calculator.calculate_multi_chiller(chillers)

        assert "total_emissions_kg_co2e" in result
        assert "total_cooling_kwh_th" in result
        assert result["total_cooling_kwh_th"] == Decimal("1500.0")

    def test_calculate_multi_chiller_aggregates_emissions(self, calculator):
        """Test multi-chiller aggregates emissions correctly."""
        chillers = [
            {
                "cooling_output_kwh_th": Decimal("1000.0"),
                "cop": Decimal("6.0"),
                "grid_emission_factor_kg_co2e_per_kwh": Decimal("0.5"),
            },
            {
                "cooling_output_kwh_th": Decimal("1000.0"),
                "cop": Decimal("6.0"),
                "grid_emission_factor_kg_co2e_per_kwh": Decimal("0.5"),
            },
        ]

        result = calculator.calculate_multi_chiller(chillers)

        # Each chiller: 1000/6.0 = 166.67 kWh, 166.67*0.5 = 83.33 kgCO2e
        # Total: 166.67 kgCO2e
        expected_emissions = Decimal("2") * (Decimal("1000.0") / Decimal("6.0")) * Decimal("0.5")
        assert abs(result["total_emissions_kg_co2e"] - expected_emissions) < Decimal("1.0")


class TestWeightedCOPCalculation:
    """Test calculate_weighted_cop()."""

    def test_calculate_weighted_cop_basic(self, calculator):
        """Test weighted COP calculation."""
        chillers = [
            {"cooling_output_kwh_th": Decimal("1000.0"), "cop": Decimal("6.0")},
            {"cooling_output_kwh_th": Decimal("500.0"), "cop": Decimal("8.0")},
        ]

        weighted_cop = calculator.calculate_weighted_cop(chillers)

        # Weighted COP = total_cooling / total_energy
        # Chiller 1: 1000/6.0 = 166.67 kWh
        # Chiller 2: 500/8.0 = 62.5 kWh
        # Total cooling = 1500, Total energy = 229.17
        # Weighted COP = 1500 / 229.17 = 6.54

        total_cooling = Decimal("1500.0")
        total_energy = Decimal("1000.0") / Decimal("6.0") + Decimal("500.0") / Decimal("8.0")
        expected_cop = total_cooling / total_energy

        assert abs(weighted_cop - expected_cop) < Decimal("0.1")

    def test_calculate_weighted_cop_single_chiller(self, calculator):
        """Test weighted COP with single chiller equals chiller COP."""
        chillers = [{"cooling_output_kwh_th": Decimal("1000.0"), "cop": Decimal("6.0")}]

        weighted_cop = calculator.calculate_weighted_cop(chillers)
        assert abs(weighted_cop - Decimal("6.0")) < Decimal("0.01")


class TestCalculateElectricChiller:
    """Test calculate_electric_chiller() dispatcher."""

    def test_calculate_electric_chiller_full_load_method(self, calculator):
        """Test calculate_electric_chiller dispatches to full_load method."""
        result = calculator.calculate_electric_chiller(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
            method="full_load",
        )

        assert "total_emissions_kg_co2e" in result
        assert result["total_emissions_kg_co2e"] > Decimal("0.0")

    def test_calculate_electric_chiller_iplv_method(self, calculator):
        """Test calculate_electric_chiller dispatches to IPLV method."""
        result = calculator.calculate_electric_chiller(
            cooling_output_kwh_th=Decimal("1000.0"),
            iplv=Decimal("7.2"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
            method="iplv_weighted",
        )

        assert "total_emissions_kg_co2e" in result

    def test_calculate_electric_chiller_custom_part_load_method(self, calculator):
        """Test calculate_electric_chiller dispatches to custom part load."""
        result = calculator.calculate_electric_chiller(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop_100=Decimal("6.0"),
            cop_75=Decimal("7.0"),
            cop_50=Decimal("8.0"),
            cop_25=Decimal("9.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
            method="custom_part_load",
        )

        assert "total_emissions_kg_co2e" in result


class TestZeroEmissions:
    """Test zero cooling output returns zero emissions."""

    def test_zero_cooling_output_zero_emissions(self, calculator):
        """Test zero cooling output returns zero emissions."""
        result = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("0.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert result["total_emissions_kg_co2e"] == Decimal("0.0")
        assert result["energy_consumption_kwh"] == Decimal("0.0")


class TestNegativeValueValidation:
    """Test negative values raise errors."""

    def test_negative_cooling_raises_error(self, calculator):
        """Test negative cooling output raises error."""
        with pytest.raises(ValueError, match="negative|positive"):
            calculator.calculate_full_load(
                cooling_output_kwh_th=Decimal("-1000.0"),
                cop=Decimal("6.0"),
                grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
            )

    def test_negative_cop_raises_error(self, calculator):
        """Test negative COP raises error."""
        with pytest.raises(ValueError, match="negative|positive|greater than"):
            calculator.calculate_full_load(
                cooling_output_kwh_th=Decimal("1000.0"),
                cop=Decimal("-6.0"),
                grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
            )

    def test_zero_cop_raises_error(self, calculator):
        """Test zero COP raises error (division by zero)."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            calculator.calculate_full_load(
                cooling_output_kwh_th=Decimal("1000.0"),
                cop=Decimal("0.0"),
                grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
            )

    def test_negative_grid_ef_raises_error(self, calculator):
        """Test negative grid emission factor raises error."""
        with pytest.raises(ValueError, match="negative"):
            calculator.calculate_full_load(
                cooling_output_kwh_th=Decimal("1000.0"),
                cop=Decimal("6.0"),
                grid_emission_factor_kg_co2e_per_kwh=Decimal("-0.5"),
            )


class TestProvenanceTracking:
    """Test provenance hash tracking."""

    def test_result_has_provenance_hash(self, calculator):
        """Test result has provenance_hash field."""
        result = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert "provenance_hash" in result
        assert isinstance(result["provenance_hash"], str)
        assert len(result["provenance_hash"]) == 64  # SHA-256 hex

    def test_provenance_hash_is_64_char_hex(self, calculator):
        """Test provenance_hash is 64-character hex string."""
        result = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        hash_value = result["provenance_hash"]
        assert len(hash_value) == 64
        # Check all characters are hex
        assert all(c in "0123456789abcdef" for c in hash_value.lower())

    def test_result_has_trace_steps(self, calculator):
        """Test result has trace_steps field."""
        result = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert "trace_steps" in result
        assert isinstance(result["trace_steps"], list)
        assert len(result["trace_steps"]) > 0

    def test_trace_steps_non_empty(self, calculator):
        """Test trace_steps list is non-empty."""
        result = calculator.calculate_full_load(
            cooling_output_kwh_th=Decimal("1000.0"),
            cop=Decimal("6.0"),
            grid_emission_factor_kg_co2e_per_kwh=Decimal("0.5"),
        )

        assert len(result["trace_steps"]) > 0
        # Each step should be a dict or string describing the calculation step
        for step in result["trace_steps"]:
            assert isinstance(step, (dict, str))
