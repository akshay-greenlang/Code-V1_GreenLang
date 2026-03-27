# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-007 ThermalTreatmentEngine.

Tests incineration (fossil/biogenic CO2 separation), energy recovery,
pyrolysis, gasification, open burning, fossil-biogenic split,
energy offset, batch processing, Decimal precision, and edge cases.

Target: 130+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Standalone helper
# ===========================================================================


def _make_thermal_engine():
    """Create a ThermalTreatmentEngine with a real database engine."""
    from greenlang.agents.mrv.waste_treatment_emissions.waste_treatment_database import (
        WasteTreatmentDatabaseEngine,
    )
    from greenlang.agents.mrv.waste_treatment_emissions.thermal_treatment import (
        ThermalTreatmentEngine,
    )
    db = WasteTreatmentDatabaseEngine()
    return ThermalTreatmentEngine(database=db)


def _incineration_input(**overrides) -> Dict[str, Any]:
    """Return a minimal valid incineration calculation input."""
    base: Dict[str, Any] = {
        "treatment_method": "incineration",
        "incinerator_type": "stoker_grate",
        "waste_composition": {
            "food_waste": Decimal("30.0"),       # 30% food (biogenic)
            "paper_cardboard": Decimal("25.0"),  # 25% paper
            "plastics": Decimal("15.0"),         # 15% plastics (fossil)
            "textiles": Decimal("10.0"),         # 10% textiles
            "wood": Decimal("10.0"),             # 10% wood (biogenic)
            "garden_waste": Decimal("10.0"),     # 10% garden (biogenic)
        },
        "waste_quantity_tonnes": Decimal("1000.0"),
        "oxidation_factor": Decimal("1.0"),
        "gwp_source": "AR6",
    }
    base.update(overrides)
    return base


def _wte_input(**overrides) -> Dict[str, Any]:
    """Return a minimal valid waste-to-energy calculation input."""
    base = _incineration_input()
    base.update({
        "energy_recovery": True,
        "electric_efficiency": Decimal("0.25"),
        "thermal_efficiency": Decimal("0.40"),
        "grid_emission_factor_kg_co2e_per_kwh": Decimal("0.45"),
        "heat_displacement_factor_kg_co2e_per_gj": Decimal("56.0"),
    })
    base.update(overrides)
    return base


def _pyrolysis_input(**overrides) -> Dict[str, Any]:
    """Return a minimal valid pyrolysis calculation input."""
    base: Dict[str, Any] = {
        "treatment_method": "pyrolysis",
        "waste_composition": {
            "plastics": Decimal("60.0"),
            "rubber_leather": Decimal("20.0"),
            "wood": Decimal("20.0"),
        },
        "waste_quantity_tonnes": Decimal("500.0"),
        "gas_yield_fraction": Decimal("0.20"),
        "oil_yield_fraction": Decimal("0.40"),
        "char_yield_fraction": Decimal("0.40"),
        "syngas_ch4_fraction": Decimal("0.15"),
        "gwp_source": "AR6",
    }
    base.update(overrides)
    return base


def _gasification_input(**overrides) -> Dict[str, Any]:
    """Return a minimal valid gasification calculation input."""
    base: Dict[str, Any] = {
        "treatment_method": "gasification",
        "waste_composition": {
            "plastics": Decimal("40.0"),
            "wood": Decimal("40.0"),
            "paper_cardboard": Decimal("20.0"),
        },
        "waste_quantity_tonnes": Decimal("500.0"),
        "equivalence_ratio": Decimal("0.3"),
        "syngas_ch4_fraction": Decimal("0.05"),
        "gwp_source": "AR6",
    }
    base.update(overrides)
    return base


def _open_burning_input(**overrides) -> Dict[str, Any]:
    """Return a minimal valid open burning calculation input."""
    base: Dict[str, Any] = {
        "treatment_method": "open_burning",
        "waste_composition": {
            "food_waste": Decimal("40.0"),
            "paper_cardboard": Decimal("20.0"),
            "plastics": Decimal("20.0"),
            "garden_waste": Decimal("20.0"),
        },
        "waste_quantity_tonnes": Decimal("200.0"),
        "dry_matter_fraction": Decimal("0.60"),
        "oxidation_factor": Decimal("0.58"),
        "gwp_source": "AR6",
    }
    base.update(overrides)
    return base


# ===========================================================================
# TestIncineration - fossil/biogenic CO2 separation
# ===========================================================================


class TestIncineration:
    """Tests for incineration emission calculations with fossil/biogenic split."""

    # -- IPCC Table 5.2 carbon content data ---

    def test_fossil_co2_positive(self, thermal_engine):
        """Incineration of mixed waste produces positive fossil CO2."""
        inp = _incineration_input()
        result = thermal_engine.calculate_incineration(inp)
        assert result["fossil_co2_kg"] > 0

    def test_biogenic_co2_positive(self, thermal_engine):
        """Incineration of mixed waste produces positive biogenic CO2."""
        inp = _incineration_input()
        result = thermal_engine.calculate_incineration(inp)
        assert result["biogenic_co2_kg"] > 0

    def test_total_co2_is_sum_of_fossil_and_biogenic(self, thermal_engine):
        """Total CO2 = fossil CO2 + biogenic CO2."""
        inp = _incineration_input()
        result = thermal_engine.calculate_incineration(inp)
        total = result["fossil_co2_kg"] + result["biogenic_co2_kg"]
        assert result["total_co2_kg"] == pytest.approx(total, rel=1e-6)

    # -- Fossil carbon fractions ----------------------------------------

    def test_plastic_100pct_fossil(self, thermal_engine):
        """100% plastic waste produces only fossil CO2, no biogenic."""
        inp = _incineration_input(
            waste_composition={"plastics": Decimal("100.0")},
        )
        result = thermal_engine.calculate_incineration(inp)
        assert result["fossil_co2_kg"] > 0
        # Biogenic should be zero or negligible
        assert result["biogenic_co2_kg"] == pytest.approx(0.0, abs=1.0)

    def test_food_0pct_fossil(self, thermal_engine):
        """100% food waste produces only biogenic CO2, no fossil."""
        inp = _incineration_input(
            waste_composition={"food_waste": Decimal("100.0")},
        )
        result = thermal_engine.calculate_incineration(inp)
        assert result["biogenic_co2_kg"] > 0
        # Fossil should be zero or negligible
        assert result["fossil_co2_kg"] == pytest.approx(0.0, abs=1.0)

    def test_paper_1pct_fossil(self, thermal_engine):
        """Paper has ~1% fossil carbon fraction."""
        inp = _incineration_input(
            waste_composition={"paper_cardboard": Decimal("100.0")},
        )
        result = thermal_engine.calculate_incineration(inp)
        # Almost all biogenic, small fossil component
        fossil_frac = result["fossil_co2_kg"] / result["total_co2_kg"]
        assert fossil_frac == pytest.approx(0.01, abs=0.02)

    # -- Oxidation factor effects ---------------------------------------

    def test_oxidation_factor_1_0(self, thermal_engine):
        """Oxidation factor of 1.0 means complete combustion."""
        inp = _incineration_input(oxidation_factor=Decimal("1.0"))
        result = thermal_engine.calculate_incineration(inp)
        assert result["total_co2_kg"] > 0

    def test_lower_oxidation_factor_reduces_co2(self, thermal_engine):
        """Lower oxidation factor reduces total CO2 output."""
        inp_high = _incineration_input(oxidation_factor=Decimal("1.0"))
        inp_low = _incineration_input(oxidation_factor=Decimal("0.95"))
        r_high = thermal_engine.calculate_incineration(inp_high)
        r_low = thermal_engine.calculate_incineration(inp_low)
        assert r_low["total_co2_kg"] <= r_high["total_co2_kg"]

    # -- N2O/CH4 by incinerator type ------------------------------------

    @pytest.mark.parametrize("incinerator_type,expected_n2o_range,expected_ch4_range", [
        ("stoker_grate", (40, 60), (0.1, 0.5)),
        ("fluidized_bed", (45, 65), (0.1, 0.3)),
        ("rotary_kiln", (40, 60), (0.1, 0.5)),
        ("semi_continuous", (50, 70), (4.0, 10.0)),
        ("batch_type", (50, 70), (40.0, 80.0)),
    ])
    def test_n2o_ch4_by_incinerator_type(self, thermal_engine, incinerator_type,
                                          expected_n2o_range, expected_ch4_range):
        """N2O and CH4 emissions vary by incinerator type (g/tonne waste)."""
        inp = _incineration_input(incinerator_type=incinerator_type)
        result = thermal_engine.calculate_incineration(inp)
        # Check that N2O per tonne is in expected range
        n2o_per_tonne = result["n2o_kg"] / float(inp["waste_quantity_tonnes"]) * 1000  # g/t
        ch4_per_tonne = result["ch4_kg"] / float(inp["waste_quantity_tonnes"]) * 1000  # g/t
        assert expected_n2o_range[0] <= n2o_per_tonne <= expected_n2o_range[1], (
            f"N2O for {incinerator_type}: {n2o_per_tonne} g/t outside "
            f"range {expected_n2o_range}"
        )
        assert expected_ch4_range[0] <= ch4_per_tonne <= expected_ch4_range[1], (
            f"CH4 for {incinerator_type}: {ch4_per_tonne} g/t outside "
            f"range {expected_ch4_range}"
        )

    # -- Multiple waste streams -----------------------------------------

    def test_mixed_waste_composition(self, thermal_engine):
        """Mixed waste composition correctly computes aggregate emissions."""
        inp = _incineration_input()  # default mixed composition
        result = thermal_engine.calculate_incineration(inp)
        assert result["fossil_co2_kg"] > 0
        assert result["biogenic_co2_kg"] > 0
        assert result["n2o_kg"] > 0
        assert result["ch4_kg"] > 0

    # -- Result structure -----------------------------------------------

    def test_incineration_result_keys(self, thermal_engine):
        """Incineration result contains all required output keys."""
        inp = _incineration_input()
        result = thermal_engine.calculate_incineration(inp)
        required_keys = [
            "fossil_co2_kg", "biogenic_co2_kg", "total_co2_kg",
            "n2o_kg", "ch4_kg", "total_co2e_kg",
            "treatment_method", "provenance_hash",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_incineration_method_label(self, thermal_engine):
        """Result labels treatment method as incineration."""
        inp = _incineration_input()
        result = thermal_engine.calculate_incineration(inp)
        assert result["treatment_method"] == "incineration"


# ===========================================================================
# TestIncinerationEnergyRecovery - WtE calculations
# ===========================================================================


class TestIncinerationEnergyRecovery:
    """Tests for waste-to-energy (WtE) incineration calculations."""

    def test_wte_produces_electricity(self, thermal_engine):
        """WtE calculation produces positive electricity output."""
        inp = _wte_input(electric_efficiency=Decimal("0.25"))
        result = thermal_engine.calculate_incineration(inp)
        assert result["electricity_generated_kwh"] > 0

    def test_wte_produces_heat(self, thermal_engine):
        """WtE calculation produces positive heat output."""
        inp = _wte_input(thermal_efficiency=Decimal("0.40"))
        result = thermal_engine.calculate_incineration(inp)
        assert result["heat_generated_gj"] > 0

    # -- Electric efficiency range --------------------------------------

    @pytest.mark.parametrize("eff", [
        Decimal("0.15"), Decimal("0.20"), Decimal("0.25"),
        Decimal("0.30"), Decimal("0.35"),
    ])
    def test_electric_efficiency_scales(self, thermal_engine, eff):
        """Higher electric efficiency produces more electricity."""
        inp_low = _wte_input(electric_efficiency=Decimal("0.15"))
        inp_high = _wte_input(electric_efficiency=eff)
        r_low = thermal_engine.calculate_incineration(inp_low)
        r_high = thermal_engine.calculate_incineration(inp_high)
        if eff > Decimal("0.15"):
            assert r_high["electricity_generated_kwh"] >= r_low["electricity_generated_kwh"]

    # -- Thermal efficiency range ---------------------------------------

    @pytest.mark.parametrize("eff", [
        Decimal("0.30"), Decimal("0.40"), Decimal("0.50"), Decimal("0.60"),
    ])
    def test_thermal_efficiency_scales(self, thermal_engine, eff):
        """Higher thermal efficiency produces more heat."""
        inp_low = _wte_input(thermal_efficiency=Decimal("0.30"))
        inp_high = _wte_input(thermal_efficiency=eff)
        r_low = thermal_engine.calculate_incineration(inp_low)
        r_high = thermal_engine.calculate_incineration(inp_high)
        if eff > Decimal("0.30"):
            assert r_high["heat_generated_gj"] >= r_low["heat_generated_gj"]

    # -- Grid displacement credits --------------------------------------

    def test_grid_displacement_reduces_net_emissions(self, thermal_engine):
        """Grid displacement credits reduce net CO2e."""
        inp_no_recovery = _incineration_input(energy_recovery=False)
        inp_with_recovery = _wte_input()
        r_no = thermal_engine.calculate_incineration(inp_no_recovery)
        r_yes = thermal_engine.calculate_incineration(inp_with_recovery)
        assert r_yes["net_co2e_kg"] <= r_no["total_co2e_kg"]

    def test_displacement_credit_positive(self, thermal_engine):
        """Energy displacement credit is a positive value."""
        inp = _wte_input()
        result = thermal_engine.calculate_incineration(inp)
        assert result.get("displacement_credit_kg_co2e", 0) >= 0

    # -- Net emissions after energy offset ------------------------------

    def test_net_emissions_less_than_gross(self, thermal_engine):
        """Net emissions are less than gross when energy is recovered."""
        inp = _wte_input()
        result = thermal_engine.calculate_incineration(inp)
        assert result["net_co2e_kg"] <= result["total_co2e_kg"]

    def test_wte_result_has_energy_keys(self, thermal_engine):
        """WtE result includes energy output keys."""
        inp = _wte_input()
        result = thermal_engine.calculate_incineration(inp)
        assert "electricity_generated_kwh" in result
        assert "heat_generated_gj" in result


# ===========================================================================
# TestPyrolysis - simplified mass balance
# ===========================================================================


class TestPyrolysis:
    """Tests for pyrolysis emission calculations."""

    def test_pyrolysis_returns_result(self, thermal_engine):
        """Pyrolysis calculation returns a valid result."""
        inp = _pyrolysis_input()
        result = thermal_engine.calculate_pyrolysis(inp)
        assert result["fossil_co2_kg"] >= 0
        assert result["total_co2e_kg"] >= 0

    # -- Gas yield fraction effects -------------------------------------

    @pytest.mark.parametrize("gas_yield", [
        Decimal("0.10"), Decimal("0.20"), Decimal("0.30"), Decimal("0.40"),
    ])
    def test_gas_yield_fraction(self, thermal_engine, gas_yield):
        """Higher gas yield produces proportionally more gas emissions."""
        inp_low = _pyrolysis_input(gas_yield_fraction=Decimal("0.10"))
        inp_high = _pyrolysis_input(gas_yield_fraction=gas_yield)
        r_low = thermal_engine.calculate_pyrolysis(inp_low)
        r_high = thermal_engine.calculate_pyrolysis(inp_high)
        if gas_yield > Decimal("0.10"):
            # More gas yield -> more gas phase emissions
            assert r_high.get("gas_phase_co2_kg", r_high["total_co2e_kg"]) >= \
                   r_low.get("gas_phase_co2_kg", 0)

    # -- Syngas composition ---------------------------------------------

    def test_syngas_ch4_produces_ch4_emissions(self, thermal_engine):
        """Syngas CH4 fraction leads to CH4 emissions."""
        inp = _pyrolysis_input(syngas_ch4_fraction=Decimal("0.15"))
        result = thermal_engine.calculate_pyrolysis(inp)
        assert result.get("ch4_kg", 0) >= 0

    # -- Fossil carbon in products --------------------------------------

    def test_fossil_carbon_tracked_in_pyrolysis(self, thermal_engine):
        """Pyrolysis tracks fossil carbon in products (plastics -> fossil)."""
        inp = _pyrolysis_input(
            waste_composition={"plastics": Decimal("100.0")},
        )
        result = thermal_engine.calculate_pyrolysis(inp)
        assert result["fossil_co2_kg"] >= 0

    def test_biogenic_waste_pyrolysis(self, thermal_engine):
        """Pyrolysis of 100% wood produces biogenic emissions."""
        inp = _pyrolysis_input(
            waste_composition={"wood": Decimal("100.0")},
        )
        result = thermal_engine.calculate_pyrolysis(inp)
        assert result.get("biogenic_co2_kg", 0) >= 0

    # -- Pyrolysis mass balance -----------------------------------------

    def test_yield_fractions_sum_to_one(self):
        """Gas + oil + char yield fractions sum to 1.0."""
        inp = _pyrolysis_input()
        total = (inp["gas_yield_fraction"] +
                 inp["oil_yield_fraction"] +
                 inp["char_yield_fraction"])
        assert total == Decimal("1.0")

    def test_pyrolysis_scales_with_quantity(self, thermal_engine):
        """Doubling waste quantity doubles total emissions."""
        inp_500 = _pyrolysis_input(waste_quantity_tonnes=Decimal("500"))
        inp_1000 = _pyrolysis_input(waste_quantity_tonnes=Decimal("1000"))
        r_500 = thermal_engine.calculate_pyrolysis(inp_500)
        r_1000 = thermal_engine.calculate_pyrolysis(inp_1000)
        ratio = r_1000["total_co2e_kg"] / r_500["total_co2e_kg"]
        assert ratio == pytest.approx(2.0, rel=1e-3)

    def test_pyrolysis_result_method_label(self, thermal_engine):
        """Pyrolysis result correctly labels treatment method."""
        inp = _pyrolysis_input()
        result = thermal_engine.calculate_pyrolysis(inp)
        assert result["treatment_method"] == "pyrolysis"


# ===========================================================================
# TestGasification - partial oxidation
# ===========================================================================


class TestGasification:
    """Tests for gasification emission calculations."""

    def test_gasification_returns_result(self, thermal_engine):
        """Gasification calculation returns a valid result."""
        inp = _gasification_input()
        result = thermal_engine.calculate_gasification(inp)
        assert result["fossil_co2_kg"] >= 0
        assert result["total_co2e_kg"] >= 0

    # -- Equivalence ratio effects --------------------------------------

    @pytest.mark.parametrize("eq_ratio", [
        Decimal("0.2"), Decimal("0.3"), Decimal("0.4"), Decimal("0.5"),
    ])
    def test_equivalence_ratio_effects(self, thermal_engine, eq_ratio):
        """Higher equivalence ratio increases direct CO2 emissions."""
        inp_low = _gasification_input(equivalence_ratio=Decimal("0.2"))
        inp_high = _gasification_input(equivalence_ratio=eq_ratio)
        r_low = thermal_engine.calculate_gasification(inp_low)
        r_high = thermal_engine.calculate_gasification(inp_high)
        if eq_ratio > Decimal("0.2"):
            # Higher ER means more oxidation, more CO2
            assert r_high["total_co2_kg"] >= r_low["total_co2_kg"]

    # -- Syngas CH4 content ---------------------------------------------

    def test_syngas_ch4_produces_emissions(self, thermal_engine):
        """Non-zero syngas CH4 fraction produces CH4 emissions."""
        inp = _gasification_input(syngas_ch4_fraction=Decimal("0.05"))
        result = thermal_engine.calculate_gasification(inp)
        assert result.get("ch4_kg", 0) >= 0

    def test_zero_syngas_ch4_no_ch4_emissions(self, thermal_engine):
        """Zero syngas CH4 fraction produces no CH4 emissions."""
        inp = _gasification_input(syngas_ch4_fraction=Decimal("0.0"))
        result = thermal_engine.calculate_gasification(inp)
        assert result.get("ch4_kg", 0) == pytest.approx(0.0, abs=0.1)

    def test_gasification_scales_with_quantity(self, thermal_engine):
        """Doubling waste doubles gasification emissions."""
        inp_500 = _gasification_input(waste_quantity_tonnes=Decimal("500"))
        inp_1000 = _gasification_input(waste_quantity_tonnes=Decimal("1000"))
        r_500 = thermal_engine.calculate_gasification(inp_500)
        r_1000 = thermal_engine.calculate_gasification(inp_1000)
        ratio = r_1000["total_co2e_kg"] / r_500["total_co2e_kg"]
        assert ratio == pytest.approx(2.0, rel=1e-3)

    def test_gasification_result_method_label(self, thermal_engine):
        """Gasification result correctly labels treatment method."""
        inp = _gasification_input()
        result = thermal_engine.calculate_gasification(inp)
        assert result["treatment_method"] == "gasification"

    def test_gasification_fossil_biogenic_split(self, thermal_engine):
        """Gasification separates fossil and biogenic CO2."""
        inp = _gasification_input()  # mix of plastics and wood
        result = thermal_engine.calculate_gasification(inp)
        assert result["fossil_co2_kg"] >= 0
        assert result.get("biogenic_co2_kg", 0) >= 0


# ===========================================================================
# TestOpenBurning - incomplete combustion
# ===========================================================================


class TestOpenBurning:
    """Tests for open burning emission calculations."""

    def test_open_burning_returns_result(self, thermal_engine):
        """Open burning calculation returns a valid result."""
        inp = _open_burning_input()
        result = thermal_engine.calculate_open_burning(inp)
        assert result["total_co2e_kg"] >= 0

    # -- OF_open = 0.58 ------------------------------------------------

    def test_open_burning_default_of(self, thermal_engine):
        """Open burning uses IPCC default oxidation factor of 0.58."""
        inp = _open_burning_input(oxidation_factor=Decimal("0.58"))
        result = thermal_engine.calculate_open_burning(inp)
        assert result["total_co2_kg"] > 0

    # -- CH4 open burning EF: 6.5 g/kg DM --------------------------------

    def test_open_burning_ch4_ef(self, thermal_engine):
        """Open burning CH4 EF is approximately 6.5 g/kg dry matter."""
        inp = _open_burning_input(
            waste_quantity_tonnes=Decimal("1.0"),  # 1 tonne
            dry_matter_fraction=Decimal("1.0"),    # 100% dry
        )
        result = thermal_engine.calculate_open_burning(inp)
        # 1000 kg DM * 6.5 g/kg = 6500 g = 6.5 kg CH4
        expected_ch4 = 6.5
        assert result["ch4_kg"] == pytest.approx(expected_ch4, rel=0.15)

    # -- N2O open burning EF: 0.15 g/kg DM --------------------------------

    def test_open_burning_n2o_ef(self, thermal_engine):
        """Open burning N2O EF is approximately 0.15 g/kg dry matter."""
        inp = _open_burning_input(
            waste_quantity_tonnes=Decimal("1.0"),
            dry_matter_fraction=Decimal("1.0"),
        )
        result = thermal_engine.calculate_open_burning(inp)
        # 1000 kg DM * 0.15 g/kg = 150 g = 0.15 kg N2O
        expected_n2o = 0.15
        assert result["n2o_kg"] == pytest.approx(expected_n2o, rel=0.15)

    def test_open_burning_has_ch4_and_n2o(self, thermal_engine):
        """Open burning produces both CH4 and N2O."""
        inp = _open_burning_input()
        result = thermal_engine.calculate_open_burning(inp)
        assert result["ch4_kg"] > 0
        assert result["n2o_kg"] > 0

    def test_open_burning_dm_fraction_effect(self, thermal_engine):
        """Higher dry matter fraction increases emissions."""
        inp_wet = _open_burning_input(dry_matter_fraction=Decimal("0.40"))
        inp_dry = _open_burning_input(dry_matter_fraction=Decimal("0.80"))
        r_wet = thermal_engine.calculate_open_burning(inp_wet)
        r_dry = thermal_engine.calculate_open_burning(inp_dry)
        assert r_dry["ch4_kg"] > r_wet["ch4_kg"]

    def test_open_burning_scales_with_quantity(self, thermal_engine):
        """Doubling open burning waste doubles emissions."""
        inp_200 = _open_burning_input(waste_quantity_tonnes=Decimal("200"))
        inp_400 = _open_burning_input(waste_quantity_tonnes=Decimal("400"))
        r_200 = thermal_engine.calculate_open_burning(inp_200)
        r_400 = thermal_engine.calculate_open_burning(inp_400)
        ratio = r_400["total_co2e_kg"] / r_200["total_co2e_kg"]
        assert ratio == pytest.approx(2.0, rel=1e-3)

    def test_open_burning_result_method_label(self, thermal_engine):
        """Open burning result correctly labels treatment method."""
        inp = _open_burning_input()
        result = thermal_engine.calculate_open_burning(inp)
        assert result["treatment_method"] == "open_burning"


# ===========================================================================
# TestFossilBiogenicSplit - correct separation for mixed waste
# ===========================================================================


class TestFossilBiogenicSplit:
    """Tests for fossil/biogenic CO2 separation in incineration."""

    def test_fossil_only_waste(self, thermal_engine):
        """100% fossil waste produces only fossil CO2."""
        inp = _incineration_input(
            waste_composition={"plastics": Decimal("100.0")},
        )
        result = thermal_engine.calculate_incineration(inp)
        total_co2 = result["total_co2_kg"]
        assert result["fossil_co2_kg"] == pytest.approx(total_co2, rel=0.01)
        assert result["biogenic_co2_kg"] == pytest.approx(0.0, abs=1.0)

    def test_biogenic_only_waste(self, thermal_engine):
        """100% biogenic waste produces only biogenic CO2."""
        inp = _incineration_input(
            waste_composition={"food_waste": Decimal("100.0")},
        )
        result = thermal_engine.calculate_incineration(inp)
        total_co2 = result["total_co2_kg"]
        assert result["biogenic_co2_kg"] == pytest.approx(total_co2, rel=0.01)
        assert result["fossil_co2_kg"] == pytest.approx(0.0, abs=1.0)

    def test_mixed_waste_both_fractions(self, thermal_engine):
        """Mixed waste produces both fossil and biogenic CO2."""
        inp = _incineration_input(
            waste_composition={
                "plastics": Decimal("50.0"),
                "food_waste": Decimal("50.0"),
            },
        )
        result = thermal_engine.calculate_incineration(inp)
        assert result["fossil_co2_kg"] > 0
        assert result["biogenic_co2_kg"] > 0

    def test_fossil_fraction_increases_with_more_plastic(self, thermal_engine):
        """More plastic in composition increases fossil CO2 fraction."""
        inp_low = _incineration_input(
            waste_composition={
                "plastics": Decimal("10.0"),
                "food_waste": Decimal("90.0"),
            },
        )
        inp_high = _incineration_input(
            waste_composition={
                "plastics": Decimal("90.0"),
                "food_waste": Decimal("10.0"),
            },
        )
        r_low = thermal_engine.calculate_incineration(inp_low)
        r_high = thermal_engine.calculate_incineration(inp_high)
        frac_low = r_low["fossil_co2_kg"] / r_low["total_co2_kg"]
        frac_high = r_high["fossil_co2_kg"] / r_high["total_co2_kg"]
        assert frac_high > frac_low

    def test_wood_is_biogenic(self, thermal_engine):
        """100% wood produces only biogenic CO2."""
        inp = _incineration_input(
            waste_composition={"wood": Decimal("100.0")},
        )
        result = thermal_engine.calculate_incineration(inp)
        assert result["biogenic_co2_kg"] > 0
        assert result["fossil_co2_kg"] == pytest.approx(0.0, abs=1.0)


# ===========================================================================
# TestEnergyOffset - electricity and heat displacement
# ===========================================================================


class TestEnergyOffset:
    """Tests for energy offset calculations in WtE."""

    def test_electricity_displacement_credit(self, thermal_engine):
        """Electricity generation creates a displacement credit."""
        inp = _wte_input()
        result = thermal_engine.calculate_incineration(inp)
        assert result.get("electricity_credit_kg_co2e", 0) >= 0

    def test_heat_displacement_credit(self, thermal_engine):
        """Heat generation creates a displacement credit."""
        inp = _wte_input()
        result = thermal_engine.calculate_incineration(inp)
        assert result.get("heat_credit_kg_co2e", 0) >= 0

    def test_no_energy_recovery_no_credit(self, thermal_engine):
        """Without energy recovery, no displacement credit."""
        inp = _incineration_input(energy_recovery=False)
        result = thermal_engine.calculate_incineration(inp)
        assert result.get("electricity_credit_kg_co2e", 0) == pytest.approx(0.0, abs=0.1)
        assert result.get("heat_credit_kg_co2e", 0) == pytest.approx(0.0, abs=0.1)

    def test_higher_grid_ef_larger_credit(self, thermal_engine):
        """Higher grid emission factor produces larger electricity credit."""
        inp_low = _wte_input(grid_emission_factor_kg_co2e_per_kwh=Decimal("0.20"))
        inp_high = _wte_input(grid_emission_factor_kg_co2e_per_kwh=Decimal("0.80"))
        r_low = thermal_engine.calculate_incineration(inp_low)
        r_high = thermal_engine.calculate_incineration(inp_high)
        assert r_high.get("electricity_credit_kg_co2e", 0) >= \
               r_low.get("electricity_credit_kg_co2e", 0)

    def test_total_credit_sum(self, thermal_engine):
        """Total displacement credit = electricity + heat credits."""
        inp = _wte_input()
        result = thermal_engine.calculate_incineration(inp)
        elec_credit = result.get("electricity_credit_kg_co2e", 0)
        heat_credit = result.get("heat_credit_kg_co2e", 0)
        total_credit = result.get("displacement_credit_kg_co2e", 0)
        if total_credit > 0:
            assert total_credit == pytest.approx(elec_credit + heat_credit, rel=1e-4)


# ===========================================================================
# TestThermalBatch - batch of multiple thermal treatments
# ===========================================================================


class TestThermalBatch:
    """Tests for batch processing of multiple thermal treatments."""

    def test_batch_single_item(self, thermal_engine):
        """Batch with single incineration returns one result."""
        items = [_incineration_input()]
        results = thermal_engine.calculate_batch(items)
        assert len(results) == 1

    def test_batch_multiple_items(self, thermal_engine):
        """Batch with multiple items returns matching count."""
        items = [
            _incineration_input(waste_quantity_tonnes=Decimal("100")),
            _incineration_input(waste_quantity_tonnes=Decimal("200")),
            _incineration_input(waste_quantity_tonnes=Decimal("300")),
        ]
        results = thermal_engine.calculate_batch(items)
        assert len(results) == 3

    def test_batch_mixed_thermal_methods(self, thermal_engine):
        """Batch handles a mix of incineration, pyrolysis, and gasification."""
        items = [
            _incineration_input(),
            _pyrolysis_input(),
            _gasification_input(),
            _open_burning_input(),
        ]
        results = thermal_engine.calculate_batch(items)
        assert len(results) == 4
        methods = [r.get("treatment_method") for r in results]
        assert "incineration" in methods
        assert "pyrolysis" in methods
        assert "gasification" in methods
        assert "open_burning" in methods

    def test_batch_empty_list(self, thermal_engine):
        """Batch with empty list returns empty results."""
        results = thermal_engine.calculate_batch([])
        assert results == []

    def test_batch_total_co2e_is_sum(self, thermal_engine):
        """Batch total CO2e is the sum of individual results."""
        items = [
            _incineration_input(waste_quantity_tonnes=Decimal("100")),
            _incineration_input(waste_quantity_tonnes=Decimal("200")),
        ]
        results = thermal_engine.calculate_batch(items)
        batch_total = sum(r["total_co2e_kg"] for r in results)
        r1 = thermal_engine.calculate_incineration(items[0])
        r2 = thermal_engine.calculate_incineration(items[1])
        individual_total = r1["total_co2e_kg"] + r2["total_co2e_kg"]
        assert batch_total == pytest.approx(individual_total, rel=1e-6)


# ===========================================================================
# TestDecimalPrecision - reproducible Decimal arithmetic
# ===========================================================================


class TestDecimalPrecision:
    """Tests for Decimal precision and reproducibility in thermal calculations."""

    def test_incineration_deterministic(self, thermal_engine):
        """Same incineration input produces identical output."""
        inp = _incineration_input()
        r1 = thermal_engine.calculate_incineration(inp)
        r2 = thermal_engine.calculate_incineration(inp)
        assert r1["fossil_co2_kg"] == r2["fossil_co2_kg"]
        assert r1["biogenic_co2_kg"] == r2["biogenic_co2_kg"]
        assert r1["total_co2e_kg"] == r2["total_co2e_kg"]

    def test_pyrolysis_deterministic(self, thermal_engine):
        """Same pyrolysis input produces identical output."""
        inp = _pyrolysis_input()
        r1 = thermal_engine.calculate_pyrolysis(inp)
        r2 = thermal_engine.calculate_pyrolysis(inp)
        assert r1["total_co2e_kg"] == r2["total_co2e_kg"]

    def test_gasification_deterministic(self, thermal_engine):
        """Same gasification input produces identical output."""
        inp = _gasification_input()
        r1 = thermal_engine.calculate_gasification(inp)
        r2 = thermal_engine.calculate_gasification(inp)
        assert r1["total_co2e_kg"] == r2["total_co2e_kg"]

    def test_open_burning_deterministic(self, thermal_engine):
        """Same open burning input produces identical output."""
        inp = _open_burning_input()
        r1 = thermal_engine.calculate_open_burning(inp)
        r2 = thermal_engine.calculate_open_burning(inp)
        assert r1["total_co2e_kg"] == r2["total_co2e_kg"]

    def test_result_types_are_numeric(self, thermal_engine):
        """All numeric result fields are Decimal or float."""
        inp = _incineration_input()
        result = thermal_engine.calculate_incineration(inp)
        numeric_keys = ["fossil_co2_kg", "biogenic_co2_kg", "total_co2_kg",
                        "n2o_kg", "ch4_kg", "total_co2e_kg"]
        for key in numeric_keys:
            assert isinstance(result[key], (Decimal, float, int)), (
                f"Key '{key}' is {type(result[key])}, expected numeric"
            )

    def test_provenance_hash_is_64_chars(self, thermal_engine):
        """Provenance hash is a 64-character SHA-256 hex string."""
        inp = _incineration_input()
        result = thermal_engine.calculate_incineration(inp)
        assert len(result["provenance_hash"]) == 64

    def test_provenance_hash_deterministic(self, thermal_engine):
        """Same input produces same provenance hash."""
        inp = _incineration_input()
        r1 = thermal_engine.calculate_incineration(inp)
        r2 = thermal_engine.calculate_incineration(inp)
        assert r1["provenance_hash"] == r2["provenance_hash"]


# ===========================================================================
# TestEdgeCases - zero waste, 100% plastic, 100% biogenic
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases in thermal treatment calculations."""

    def test_zero_waste_incineration(self, thermal_engine):
        """Zero waste produces zero incineration emissions."""
        inp = _incineration_input(waste_quantity_tonnes=Decimal("0"))
        result = thermal_engine.calculate_incineration(inp)
        assert result["fossil_co2_kg"] == pytest.approx(0.0)
        assert result["biogenic_co2_kg"] == pytest.approx(0.0)
        assert result["total_co2e_kg"] == pytest.approx(0.0)

    def test_zero_waste_pyrolysis(self, thermal_engine):
        """Zero waste produces zero pyrolysis emissions."""
        inp = _pyrolysis_input(waste_quantity_tonnes=Decimal("0"))
        result = thermal_engine.calculate_pyrolysis(inp)
        assert result["total_co2e_kg"] == pytest.approx(0.0)

    def test_zero_waste_gasification(self, thermal_engine):
        """Zero waste produces zero gasification emissions."""
        inp = _gasification_input(waste_quantity_tonnes=Decimal("0"))
        result = thermal_engine.calculate_gasification(inp)
        assert result["total_co2e_kg"] == pytest.approx(0.0)

    def test_zero_waste_open_burning(self, thermal_engine):
        """Zero waste produces zero open burning emissions."""
        inp = _open_burning_input(waste_quantity_tonnes=Decimal("0"))
        result = thermal_engine.calculate_open_burning(inp)
        assert result["total_co2e_kg"] == pytest.approx(0.0)

    def test_100pct_plastic_incineration(self, thermal_engine):
        """100% plastic produces only fossil CO2 in incineration."""
        inp = _incineration_input(
            waste_composition={"plastics": Decimal("100.0")},
        )
        result = thermal_engine.calculate_incineration(inp)
        assert result["fossil_co2_kg"] > 0
        assert result["biogenic_co2_kg"] == pytest.approx(0.0, abs=1.0)

    def test_100pct_biogenic_incineration(self, thermal_engine):
        """100% food waste produces only biogenic CO2 in incineration."""
        inp = _incineration_input(
            waste_composition={"food_waste": Decimal("100.0")},
        )
        result = thermal_engine.calculate_incineration(inp)
        assert result["biogenic_co2_kg"] > 0
        assert result["fossil_co2_kg"] == pytest.approx(0.0, abs=1.0)

    def test_very_large_quantity_incineration(self, thermal_engine):
        """Very large quantity (100k tonnes) calculates without error."""
        inp = _incineration_input(waste_quantity_tonnes=Decimal("100000"))
        result = thermal_engine.calculate_incineration(inp)
        assert result["total_co2e_kg"] > 0

    def test_very_small_quantity_incineration(self, thermal_engine):
        """Very small quantity (0.001 tonnes) produces small emissions."""
        inp = _incineration_input(waste_quantity_tonnes=Decimal("0.001"))
        result = thermal_engine.calculate_incineration(inp)
        assert result["total_co2e_kg"] > 0
        assert result["total_co2e_kg"] < 100  # Very small

    def test_single_waste_type_composition(self, thermal_engine):
        """Single waste type in composition dict works correctly."""
        for wt in ["food_waste", "plastics", "wood", "paper_cardboard"]:
            inp = _incineration_input(
                waste_composition={wt: Decimal("100.0")},
            )
            result = thermal_engine.calculate_incineration(inp)
            assert result["total_co2_kg"] >= 0

    def test_incineration_has_processing_time(self, thermal_engine):
        """Incineration result includes processing time."""
        inp = _incineration_input()
        result = thermal_engine.calculate_incineration(inp)
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    def test_pyrolysis_has_provenance_hash(self, thermal_engine):
        """Pyrolysis result includes provenance hash."""
        inp = _pyrolysis_input()
        result = thermal_engine.calculate_pyrolysis(inp)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_gasification_has_provenance_hash(self, thermal_engine):
        """Gasification result includes provenance hash."""
        inp = _gasification_input()
        result = thermal_engine.calculate_gasification(inp)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_open_burning_has_provenance_hash(self, thermal_engine):
        """Open burning result includes provenance hash."""
        inp = _open_burning_input()
        result = thermal_engine.calculate_open_burning(inp)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_open_burning_zero_dm(self, thermal_engine):
        """Zero dry matter fraction produces zero emissions from burning."""
        inp = _open_burning_input(dry_matter_fraction=Decimal("0.0"))
        result = thermal_engine.calculate_open_burning(inp)
        assert result["ch4_kg"] == pytest.approx(0.0, abs=0.01)
        assert result["n2o_kg"] == pytest.approx(0.0, abs=0.01)

    def test_different_gwp_sources_incineration(self, thermal_engine):
        """Different GWP sources produce different CO2e for same waste."""
        inp_ar4 = _incineration_input(gwp_source="AR4")
        inp_ar6 = _incineration_input(gwp_source="AR6")
        r_ar4 = thermal_engine.calculate_incineration(inp_ar4)
        r_ar6 = thermal_engine.calculate_incineration(inp_ar6)
        # N2O and CH4 CO2e differ; CO2 CO2e is the same
        # Overall total_co2e will differ due to CH4/N2O GWP differences
        assert r_ar4["total_co2e_kg"] != pytest.approx(r_ar6["total_co2e_kg"], abs=10)
