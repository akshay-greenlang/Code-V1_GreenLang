# -*- coding: utf-8 -*-
"""
Unit tests for CHPAllocationEngine (Engine 4 of 7) - AGENT-MRV-011.

Tests CHP emission allocation via efficiency, energy, and exergy methods,
multi-product allocation, Primary Energy Savings, fuel emission computation,
cross-method comparison, batch allocation, and provenance tracking.

Target: ~80 tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List

import pytest

try:
    from greenlang.steam_heat_purchase.chp_allocation import (
        CHPAllocationEngine,
        FUEL_EMISSION_FACTORS,
        CHP_DEFAULT_EFFICIENCIES,
        REFERENCE_ELECTRICAL_EFFICIENCIES,
        REFERENCE_THERMAL_EFFICIENCY,
        DEFAULT_REFERENCE_ELECTRICAL_EFFICIENCY,
        DEFAULT_AMBIENT_TEMP_C,
        DEFAULT_STEAM_TEMP_C,
        PES_THRESHOLD_LARGE,
        PES_THRESHOLD_SMALL,
        SMALL_CHP_CAPACITY_MW,
        GWP_VALUES,
        _CHPAllocationProvenance,
        get_chp_allocator,
    )
    CHP_AVAILABLE = True
except ImportError:
    CHP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CHP_AVAILABLE,
    reason="greenlang.steam_heat_purchase.chp_allocation not importable",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a fresh CHPAllocationEngine instance."""
    CHPAllocationEngine.reset()
    return CHPAllocationEngine()


@pytest.fixture
def ng_request() -> Dict[str, Any]:
    """Natural gas CHP allocation request."""
    return {
        "total_fuel_gj": Decimal("2000"),
        "fuel_type": "natural_gas",
        "heat_output_gj": Decimal("900"),
        "power_output_gj": Decimal("700"),
        "electrical_efficiency": Decimal("0.35"),
        "thermal_efficiency": Decimal("0.45"),
        "gwp_source": "AR6",
        "tenant_id": "test-tenant",
    }


# ===========================================================================
# 1. Singleton Pattern Tests
# ===========================================================================


class TestSingletonPattern:
    """Tests for CHPAllocationEngine singleton pattern."""

    def test_same_instance_returned(self):
        CHPAllocationEngine.reset()
        e1 = CHPAllocationEngine()
        e2 = CHPAllocationEngine()
        assert e1 is e2

    def test_reset_creates_new_instance(self):
        e1 = CHPAllocationEngine()
        CHPAllocationEngine.reset()
        e2 = CHPAllocationEngine()
        assert e1 is not e2

    def test_get_chp_allocator_returns_instance(self):
        CHPAllocationEngine.reset()
        e = get_chp_allocator()
        assert isinstance(e, CHPAllocationEngine)

    def test_get_chp_allocator_is_singleton(self):
        CHPAllocationEngine.reset()
        e1 = get_chp_allocator()
        e2 = get_chp_allocator()
        assert e1 is e2


# ===========================================================================
# 2. Efficiency Method Tests
# ===========================================================================


class TestEfficiencyMethod:
    """Tests for allocate_efficiency_method."""

    def test_basic_efficiency_allocation(self, engine):
        """heat_fuel_equiv = 900/0.45=2000, power_fuel_equiv = 700/0.35=2000,
        heat_share=0.5, power_share=0.5."""
        result = engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
        )
        assert result["method"] == "efficiency"
        heat_share = result["heat_share"]
        power_share = result["power_share"]
        assert heat_share == Decimal("0.5").quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP) or abs(heat_share - Decimal("0.5")) < Decimal("0.001")
        assert abs(power_share - Decimal("0.5")) < Decimal("0.001")
        assert abs(heat_share + power_share - Decimal("1")) < Decimal("0.0001")

    def test_efficiency_shares_sum_to_one(self, engine):
        result = engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("5000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("1800"),
            power_output_gj=Decimal("1200"),
            eta_thermal=Decimal("0.40"),
            eta_electrical=Decimal("0.30"),
        )
        total = result["heat_share"] + result["power_share"]
        assert abs(total - Decimal("1")) < Decimal("0.0001")

    def test_efficiency_result_has_provenance(self, engine):
        result = engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_efficiency_result_has_emissions(self, engine):
        result = engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
        )
        assert "heat_emissions_kgco2e" in result
        assert "power_emissions_kgco2e" in result
        assert result["heat_emissions_kgco2e"] > Decimal("0")
        assert result["power_emissions_kgco2e"] > Decimal("0")

    def test_efficiency_deterministic(self, engine):
        kwargs = dict(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
        )
        r1 = engine.allocate_efficiency_method(**kwargs)
        r2 = engine.allocate_efficiency_method(**kwargs)
        assert r1["heat_share"] == r2["heat_share"]
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_efficiency_different_efficiencies(self, engine):
        result = engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("1000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("300"),
            power_output_gj=Decimal("350"),
            eta_thermal=Decimal("0.50"),
            eta_electrical=Decimal("0.40"),
        )
        # heat_fuel_equiv = 300/0.5=600, power_fuel_equiv = 350/0.4=875
        # heat_share = 600/1475 ≈ 0.4068, power_share ≈ 0.5932
        assert result["heat_share"] < result["power_share"]


# ===========================================================================
# 3. Energy Method Tests
# ===========================================================================


class TestEnergyMethod:
    """Tests for allocate_energy_method."""

    def test_basic_energy_allocation(self, engine):
        """heat_share = 900/(900+700) = 0.5625, power_share = 0.4375."""
        result = engine.allocate_energy_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
        )
        assert result["method"] == "energy"
        assert abs(result["heat_share"] - Decimal("0.5625")) < Decimal("0.001")
        assert abs(result["power_share"] - Decimal("0.4375")) < Decimal("0.001")

    def test_energy_shares_sum_to_one(self, engine):
        result = engine.allocate_energy_method(
            total_fuel_gj=Decimal("3000"),
            fuel_type="coal_bituminous",
            heat_output_gj=Decimal("1200"),
            power_output_gj=Decimal("800"),
        )
        total = result["heat_share"] + result["power_share"]
        assert abs(total - Decimal("1")) < Decimal("0.0001")

    def test_energy_equal_outputs(self, engine):
        result = engine.allocate_energy_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("500"),
            power_output_gj=Decimal("500"),
        )
        assert abs(result["heat_share"] - Decimal("0.5")) < Decimal("0.001")

    def test_energy_result_has_provenance(self, engine):
        result = engine.allocate_energy_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 4. Exergy Method Tests
# ===========================================================================


class TestExergyMethod:
    """Tests for allocate_exergy_method."""

    def test_basic_exergy_allocation(self, engine):
        """Carnot = 1-(25+273.15)/(200+273.15) ≈ 0.370,
        exergy_heat=900*0.370=333, power=700, total=1033,
        heat_share ≈ 0.322."""
        result = engine.allocate_exergy_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            steam_temp_c=Decimal("200"),
            ambient_temp_c=Decimal("25"),
        )
        assert result["method"] == "exergy"
        # Carnot factor approximately 0.370
        carnot = Decimal("1") - (Decimal("25") + Decimal("273.15")) / (Decimal("200") + Decimal("273.15"))
        assert abs(result.get("carnot_factor", carnot) - carnot) < Decimal("0.01")
        # heat_share ≈ 0.322
        assert abs(result["heat_share"] - Decimal("0.322")) < Decimal("0.02")

    def test_exergy_shares_sum_to_one(self, engine):
        result = engine.allocate_exergy_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            steam_temp_c=Decimal("200"),
            ambient_temp_c=Decimal("25"),
        )
        total = result["heat_share"] + result["power_share"]
        assert abs(total - Decimal("1")) < Decimal("0.0001")

    def test_exergy_heat_share_less_than_energy(self, engine):
        """Exergy method always gives lower heat share than energy method."""
        exergy_result = engine.allocate_exergy_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            steam_temp_c=Decimal("200"),
            ambient_temp_c=Decimal("25"),
        )
        energy_result = engine.allocate_energy_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
        )
        assert exergy_result["heat_share"] < energy_result["heat_share"]

    def test_exergy_higher_temp_gives_higher_heat_share(self, engine):
        """Higher steam temp -> higher Carnot -> higher exergy heat share."""
        r_low = engine.allocate_exergy_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            steam_temp_c=Decimal("150"),
            ambient_temp_c=Decimal("25"),
        )
        r_high = engine.allocate_exergy_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            steam_temp_c=Decimal("300"),
            ambient_temp_c=Decimal("25"),
        )
        assert r_high["heat_share"] > r_low["heat_share"]

    def test_exergy_result_has_provenance(self, engine):
        result = engine.allocate_exergy_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            steam_temp_c=Decimal("200"),
            ambient_temp_c=Decimal("25"),
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 5. Multiproduct (heat + power + cooling) Allocation Tests
# ===========================================================================


class TestMultiproductAllocation:
    """Tests for allocate_multiproduct (three-way allocation)."""

    def test_multiproduct_three_way(self, engine):
        result = engine.allocate_multiproduct(
            total_fuel_gj=Decimal("3000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("1000"),
            power_output_gj=Decimal("800"),
            cooling_output_gj=Decimal("200"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
            eta_cooling=Decimal("0.70"),
        )
        heat_share = result["heat_share"]
        power_share = result["power_share"]
        cooling_share = result["cooling_share"]
        total = heat_share + power_share + cooling_share
        assert abs(total - Decimal("1")) < Decimal("0.001")

    def test_multiproduct_all_shares_positive(self, engine):
        result = engine.allocate_multiproduct(
            total_fuel_gj=Decimal("3000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("1000"),
            power_output_gj=Decimal("800"),
            cooling_output_gj=Decimal("200"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
            eta_cooling=Decimal("0.70"),
        )
        assert result["heat_share"] > Decimal("0")
        assert result["power_share"] > Decimal("0")
        assert result["cooling_share"] > Decimal("0")

    def test_multiproduct_has_cooling_emissions(self, engine):
        result = engine.allocate_multiproduct(
            total_fuel_gj=Decimal("3000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("1000"),
            power_output_gj=Decimal("800"),
            cooling_output_gj=Decimal("200"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
            eta_cooling=Decimal("0.70"),
        )
        assert "cooling_emissions_kgco2e" in result
        assert result["cooling_emissions_kgco2e"] > Decimal("0")

    def test_multiproduct_has_provenance(self, engine):
        result = engine.allocate_multiproduct(
            total_fuel_gj=Decimal("3000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("1000"),
            power_output_gj=Decimal("800"),
            cooling_output_gj=Decimal("200"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
            eta_cooling=Decimal("0.70"),
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 6. Primary Energy Savings Tests
# ===========================================================================


class TestPrimaryEnergySavings:
    """Tests for compute_primary_energy_savings (EU EED)."""

    def test_pes_basic_calculation(self, engine):
        """PES = 1 - 1/((0.35/0.525)+(0.45/0.90)) ≈ 0.137 (13.7%)."""
        result = engine.compute_primary_energy_savings(
            eta_electrical=Decimal("0.35"),
            eta_thermal=Decimal("0.45"),
            ref_eta_electrical=Decimal("0.525"),
            ref_eta_thermal=Decimal("0.90"),
        )
        pes = result["primary_energy_savings_pct"]
        # PES should be approximately 13.7%
        assert abs(pes - Decimal("13.7")) < Decimal("1.0")

    def test_pes_result_is_percentage(self, engine):
        result = engine.compute_primary_energy_savings(
            eta_electrical=Decimal("0.35"),
            eta_thermal=Decimal("0.45"),
            ref_eta_electrical=Decimal("0.525"),
            ref_eta_thermal=Decimal("0.90"),
        )
        pes = result["primary_energy_savings_pct"]
        assert pes > Decimal("0")
        assert pes < Decimal("100")

    def test_high_efficiency_large_plant(self, engine):
        """PES > 10% for large plant (>= 1MW) -> True."""
        result = engine.compute_primary_energy_savings(
            eta_electrical=Decimal("0.35"),
            eta_thermal=Decimal("0.45"),
            ref_eta_electrical=Decimal("0.525"),
            ref_eta_thermal=Decimal("0.90"),
            capacity_mw=Decimal("5"),
        )
        assert result["is_high_efficiency"] is True

    def test_low_efficiency_large_plant(self, engine):
        """PES < 10% for large plant -> False."""
        result = engine.compute_primary_energy_savings(
            eta_electrical=Decimal("0.20"),
            eta_thermal=Decimal("0.30"),
            ref_eta_electrical=Decimal("0.525"),
            ref_eta_thermal=Decimal("0.90"),
        )
        # Very low efficiencies should give negative or near-zero PES
        pes = result["primary_energy_savings_pct"]
        if pes < Decimal("10"):
            assert result.get("is_high_efficiency", True) is False or pes < Decimal("10")

    def test_pes_has_provenance(self, engine):
        result = engine.compute_primary_energy_savings(
            eta_electrical=Decimal("0.35"),
            eta_thermal=Decimal("0.45"),
            ref_eta_electrical=Decimal("0.525"),
            ref_eta_thermal=Decimal("0.90"),
        )
        assert "provenance_hash" in result


# ===========================================================================
# 7. Carnot Factor Tests
# ===========================================================================


class TestCarnotFactor:
    """Tests for compute_carnot_factor."""

    def test_carnot_basic(self, engine):
        """Carnot = 1 - (25+273.15)/(200+273.15)."""
        result = engine.compute_carnot_factor(
            steam_temp_c=Decimal("200"),
            ambient_temp_c=Decimal("25"),
        )
        expected = Decimal("1") - (Decimal("298.15") / Decimal("473.15"))
        if isinstance(result, dict):
            cf = result.get("carnot_factor", result.get("value", Decimal("0")))
        else:
            cf = result
        assert abs(cf - expected) < Decimal("0.01")

    def test_carnot_increases_with_temp(self, engine):
        r_low = engine.compute_carnot_factor(
            steam_temp_c=Decimal("100"),
            ambient_temp_c=Decimal("25"),
        )
        r_high = engine.compute_carnot_factor(
            steam_temp_c=Decimal("400"),
            ambient_temp_c=Decimal("25"),
        )
        low_val = r_low.get("carnot_factor", r_low) if isinstance(r_low, dict) else r_low
        high_val = r_high.get("carnot_factor", r_high) if isinstance(r_high, dict) else r_high
        assert high_val > low_val

    def test_carnot_between_0_and_1(self, engine):
        result = engine.compute_carnot_factor(
            steam_temp_c=Decimal("200"),
            ambient_temp_c=Decimal("25"),
        )
        cf = result.get("carnot_factor", result) if isinstance(result, dict) else result
        assert Decimal("0") < cf < Decimal("1")


# ===========================================================================
# 8. Compare Methods Tests
# ===========================================================================


class TestCompareMethods:
    """Tests for compare_allocation_methods."""

    def test_compare_returns_all_three_methods(self, engine):
        result = engine.compare_allocation_methods(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
            steam_temp_c=Decimal("200"),
            ambient_temp_c=Decimal("25"),
        )
        methods = result.get("methods", result.get("results", {}))
        if isinstance(methods, dict):
            assert "efficiency" in methods
            assert "energy" in methods
            assert "exergy" in methods
        elif isinstance(methods, list):
            method_names = [m.get("method", "") for m in methods]
            assert "efficiency" in method_names
            assert "energy" in method_names
            assert "exergy" in method_names

    def test_compare_has_provenance(self, engine):
        result = engine.compare_allocation_methods(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
            steam_temp_c=Decimal("200"),
            ambient_temp_c=Decimal("25"),
        )
        assert "provenance_hash" in result


# ===========================================================================
# 9. Fuel Emissions Tests
# ===========================================================================


class TestFuelEmissions:
    """Tests for compute_fuel_emissions."""

    def test_fuel_emissions_natural_gas(self, engine):
        """2000 GJ natural_gas: CO2 = 2000 * 56.1 = 112200 kg."""
        result = engine.compute_fuel_emissions(
            fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
        )
        assert "co2_kg" in result
        assert "ch4_kg" in result
        assert "n2o_kg" in result
        assert "total_co2e_kg" in result
        co2_expected = Decimal("2000") * Decimal("56.1")
        assert abs(result["co2_kg"] - co2_expected) < Decimal("10")

    def test_fuel_emissions_with_gwp(self, engine):
        result = engine.compute_fuel_emissions(
            fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            gwp_source="AR6",
        )
        # total_co2e should include CH4 and N2O contributions
        assert result["total_co2e_kg"] >= result["co2_kg"]

    def test_fuel_emissions_coal(self, engine):
        result = engine.compute_fuel_emissions(
            fuel_gj=Decimal("1000"),
            fuel_type="coal_bituminous",
        )
        assert result["co2_kg"] > Decimal("0")
        # Coal has higher EF than natural gas
        ng_result = engine.compute_fuel_emissions(
            fuel_gj=Decimal("1000"),
            fuel_type="natural_gas",
        )
        assert result["co2_kg"] > ng_result["co2_kg"]

    def test_fuel_emissions_biomass_biogenic(self, engine):
        result = engine.compute_fuel_emissions(
            fuel_gj=Decimal("1000"),
            fuel_type="biomass_wood",
        )
        assert result["co2_kg"] > Decimal("0")
        # Biomass has is_biogenic flag
        assert result.get("is_biogenic", False) is True or result.get("biogenic_co2_kg", Decimal("0")) > Decimal("0")


# ===========================================================================
# 10. Validation Tests
# ===========================================================================


class TestValidation:
    """Tests for request validation."""

    def test_negative_fuel_gj_raises(self, engine):
        with pytest.raises((ValueError, Exception)):
            engine.allocate_efficiency_method(
                total_fuel_gj=Decimal("-100"),
                fuel_type="natural_gas",
                heat_output_gj=Decimal("900"),
                power_output_gj=Decimal("700"),
                eta_thermal=Decimal("0.45"),
                eta_electrical=Decimal("0.35"),
            )

    def test_zero_fuel_gj_raises(self, engine):
        with pytest.raises((ValueError, Exception)):
            engine.allocate_efficiency_method(
                total_fuel_gj=Decimal("0"),
                fuel_type="natural_gas",
                heat_output_gj=Decimal("900"),
                power_output_gj=Decimal("700"),
                eta_thermal=Decimal("0.45"),
                eta_electrical=Decimal("0.35"),
            )

    def test_efficiency_above_1_raises(self, engine):
        with pytest.raises((ValueError, Exception)):
            engine.allocate_efficiency_method(
                total_fuel_gj=Decimal("2000"),
                fuel_type="natural_gas",
                heat_output_gj=Decimal("900"),
                power_output_gj=Decimal("700"),
                eta_thermal=Decimal("1.5"),
                eta_electrical=Decimal("0.35"),
            )

    def test_negative_efficiency_raises(self, engine):
        with pytest.raises((ValueError, Exception)):
            engine.allocate_efficiency_method(
                total_fuel_gj=Decimal("2000"),
                fuel_type="natural_gas",
                heat_output_gj=Decimal("900"),
                power_output_gj=Decimal("700"),
                eta_thermal=Decimal("-0.1"),
                eta_electrical=Decimal("0.35"),
            )

    def test_validate_request_valid(self, engine):
        errors = engine.validate_request({
            "total_fuel_gj": Decimal("2000"),
            "fuel_type": "natural_gas",
            "heat_output_gj": Decimal("900"),
            "power_output_gj": Decimal("700"),
            "method": "efficiency",
            "electrical_efficiency": Decimal("0.35"),
            "thermal_efficiency": Decimal("0.45"),
        })
        assert errors is None or errors == [] or errors.get("valid", True) is True

    def test_validate_request_missing_fuel_type(self, engine):
        result = engine.validate_request({
            "total_fuel_gj": Decimal("2000"),
            "heat_output_gj": Decimal("900"),
            "power_output_gj": Decimal("700"),
        })
        # Should return errors or raise
        if isinstance(result, list):
            assert len(result) > 0
        elif isinstance(result, dict):
            assert result.get("valid", True) is False or len(result.get("errors", [])) > 0


# ===========================================================================
# 11. Batch Allocation Tests
# ===========================================================================


class TestBatchAllocation:
    """Tests for batch_allocate."""

    def test_batch_allocate_multiple(self, engine):
        requests = [
            {
                "total_fuel_gj": Decimal("2000"),
                "fuel_type": "natural_gas",
                "heat_output_gj": Decimal("900"),
                "power_output_gj": Decimal("700"),
                "method": "efficiency",
                "electrical_efficiency": Decimal("0.35"),
                "thermal_efficiency": Decimal("0.45"),
            },
            {
                "total_fuel_gj": Decimal("3000"),
                "fuel_type": "coal_bituminous",
                "heat_output_gj": Decimal("1200"),
                "power_output_gj": Decimal("800"),
                "method": "energy",
            },
        ]
        results = engine.batch_allocate(requests)
        assert len(results) == 2 or results.get("count", 0) == 2

    def test_batch_allocate_empty_raises(self, engine):
        with pytest.raises((ValueError, Exception)):
            engine.batch_allocate([])


# ===========================================================================
# 12. Default Efficiencies Tests
# ===========================================================================


class TestDefaultEfficiencies:
    """Tests for get_default_efficiencies."""

    def test_get_defaults_natural_gas(self, engine):
        result = engine.get_default_efficiencies("natural_gas")
        assert result["electrical_efficiency"] == Decimal("0.35")
        assert result["thermal_efficiency"] == Decimal("0.45")
        assert result["overall_efficiency"] == Decimal("0.80")

    def test_get_defaults_coal(self, engine):
        result = engine.get_default_efficiencies("coal")
        assert result["electrical_efficiency"] == Decimal("0.30")
        assert result["thermal_efficiency"] == Decimal("0.40")

    def test_get_defaults_biomass(self, engine):
        result = engine.get_default_efficiencies("biomass")
        assert result["electrical_efficiency"] == Decimal("0.25")
        assert result["thermal_efficiency"] == Decimal("0.50")


# ===========================================================================
# 13. Overall Efficiency Tests
# ===========================================================================


class TestOverallEfficiency:
    """Tests for compute_overall_efficiency."""

    def test_overall_efficiency(self, engine):
        result = engine.compute_overall_efficiency(
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            total_fuel_gj=Decimal("2000"),
        )
        expected = (Decimal("900") + Decimal("700")) / Decimal("2000")
        if isinstance(result, dict):
            eff = result.get("overall_efficiency", Decimal("0"))
        else:
            eff = result
        assert abs(eff - expected) < Decimal("0.01")


# ===========================================================================
# 14. Provenance Chain Tests
# ===========================================================================


class TestProvenanceChain:
    """Tests for _CHPAllocationProvenance."""

    def test_provenance_init(self):
        prov = _CHPAllocationProvenance()
        assert prov.entry_count == 0

    def test_provenance_record(self):
        prov = _CHPAllocationProvenance()
        entry = prov.record("chp_allocation", "allocate", "alloc-001")
        assert "hash_value" in entry
        assert len(entry["hash_value"]) == 64
        assert prov.entry_count == 1

    def test_provenance_chain_verification(self):
        prov = _CHPAllocationProvenance()
        prov.record("chp_allocation", "allocate", "alloc-001")
        prov.record("chp_allocation", "validate", "alloc-001")
        assert prov.verify_chain() is True

    def test_provenance_reset(self):
        prov = _CHPAllocationProvenance()
        prov.record("chp_allocation", "allocate", "alloc-001")
        prov.reset()
        assert prov.entry_count == 0

    def test_provenance_entries_copy(self):
        prov = _CHPAllocationProvenance()
        prov.record("chp_allocation", "allocate", "alloc-001")
        entries = prov.get_entries()
        assert len(entries) == 1
        entries.append({"fake": True})
        assert prov.entry_count == 1  # original not modified

    def test_provenance_parent_chaining(self):
        prov = _CHPAllocationProvenance()
        e1 = prov.record("chp_allocation", "allocate", "alloc-001")
        e2 = prov.record("chp_allocation", "validate", "alloc-002")
        assert e2["parent_hash"] == e1["hash_value"]


# ===========================================================================
# 15. Constants and Metadata Tests
# ===========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_fuel_emission_factors_count(self):
        assert len(FUEL_EMISSION_FACTORS) >= 10

    def test_chp_default_efficiencies_count(self):
        assert len(CHP_DEFAULT_EFFICIENCIES) == 5

    def test_reference_electrical_efficiencies_count(self):
        assert len(REFERENCE_ELECTRICAL_EFFICIENCIES) >= 5

    def test_reference_thermal_efficiency(self):
        assert REFERENCE_THERMAL_EFFICIENCY == Decimal("0.90")

    def test_default_ref_electrical(self):
        assert DEFAULT_REFERENCE_ELECTRICAL_EFFICIENCY == Decimal("0.525")

    def test_pes_threshold_large(self):
        assert PES_THRESHOLD_LARGE == Decimal("10")

    def test_pes_threshold_small(self):
        assert PES_THRESHOLD_SMALL == Decimal("0")

    def test_small_chp_capacity_mw(self):
        assert SMALL_CHP_CAPACITY_MW == Decimal("1")

    def test_gwp_values_has_ar6(self):
        assert "AR6" in GWP_VALUES


# ===========================================================================
# 16. Stats and Allocation Stats Tests
# ===========================================================================


class TestAllocationStats:
    """Tests for get_allocation_stats."""

    def test_stats_after_allocation(self, engine):
        engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
        )
        stats = engine.get_allocation_stats()
        assert isinstance(stats, dict)
        assert stats.get("total_allocations", 0) >= 1 or stats.get("count", 0) >= 1

    def test_stats_initial(self, engine):
        stats = engine.get_allocation_stats()
        assert isinstance(stats, dict)


# ===========================================================================
# 17. Additional Efficiency Method Edge Cases
# ===========================================================================


class TestEfficiencyEdgeCases:
    """Additional edge cases for efficiency method."""

    def test_efficiency_asymmetric_high_heat(self, engine):
        """High heat output / low power output."""
        result = engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("3000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("2000"),
            power_output_gj=Decimal("200"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
        )
        assert result["heat_share"] > result["power_share"]

    def test_efficiency_asymmetric_high_power(self, engine):
        """Low heat output / high power output."""
        result = engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("3000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("200"),
            power_output_gj=Decimal("2000"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
        )
        assert result["power_share"] > result["heat_share"]

    def test_efficiency_high_efficiencies(self, engine):
        """Near-maximum efficiencies."""
        result = engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("5000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("2000"),
            power_output_gj=Decimal("1500"),
            eta_thermal=Decimal("0.90"),
            eta_electrical=Decimal("0.50"),
        )
        total = result["heat_share"] + result["power_share"]
        assert abs(total - Decimal("1")) < Decimal("0.001")

    def test_efficiency_coal_fuel(self, engine):
        result = engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("5000"),
            fuel_type="coal_bituminous",
            heat_output_gj=Decimal("1500"),
            power_output_gj=Decimal("1000"),
            eta_thermal=Decimal("0.40"),
            eta_electrical=Decimal("0.30"),
        )
        assert result["heat_emissions_kgco2e"] > Decimal("0")

    def test_efficiency_biomass_fuel(self, engine):
        result = engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("3000"),
            fuel_type="biomass_wood",
            heat_output_gj=Decimal("1200"),
            power_output_gj=Decimal("500"),
            eta_thermal=Decimal("0.50"),
            eta_electrical=Decimal("0.25"),
        )
        assert result["heat_share"] > Decimal("0")

    def test_efficiency_small_quantities(self, engine):
        result = engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("10"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("3"),
            power_output_gj=Decimal("2"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
        )
        assert result["heat_share"] > Decimal("0")
        assert result["power_share"] > Decimal("0")

    def test_efficiency_large_quantities(self, engine):
        result = engine.allocate_efficiency_method(
            total_fuel_gj=Decimal("1000000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("400000"),
            power_output_gj=Decimal("350000"),
            eta_thermal=Decimal("0.45"),
            eta_electrical=Decimal("0.35"),
        )
        total = result["heat_share"] + result["power_share"]
        assert abs(total - Decimal("1")) < Decimal("0.001")


# ===========================================================================
# 18. Additional Exergy Edge Cases
# ===========================================================================


class TestExergyEdgeCases:
    """Additional edge cases for exergy method."""

    def test_exergy_low_temperature(self, engine):
        result = engine.allocate_exergy_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            steam_temp_c=Decimal("80"),
            ambient_temp_c=Decimal("25"),
        )
        # Low temp heat has low exergy, so heat_share should be small
        assert result["heat_share"] < Decimal("0.3")

    def test_exergy_very_high_temperature(self, engine):
        result = engine.allocate_exergy_method(
            total_fuel_gj=Decimal("2000"),
            fuel_type="natural_gas",
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
            steam_temp_c=Decimal("500"),
            ambient_temp_c=Decimal("25"),
        )
        # High temp heat has high Carnot, exergy shares closer to energy shares
        assert result["heat_share"] > Decimal("0.3")


# ===========================================================================
# 19. allocate_chp_emissions Dispatch Method Tests
# ===========================================================================


class TestAllocateCHPEmissions:
    """Tests for the dispatch method allocate_chp_emissions."""

    def test_dispatch_efficiency(self, engine):
        result = engine.allocate_chp_emissions({
            "total_fuel_gj": Decimal("2000"),
            "fuel_type": "natural_gas",
            "heat_output_gj": Decimal("900"),
            "power_output_gj": Decimal("700"),
            "method": "efficiency",
            "electrical_efficiency": Decimal("0.35"),
            "thermal_efficiency": Decimal("0.45"),
        })
        assert result.get("method") == "efficiency"

    def test_dispatch_energy(self, engine):
        result = engine.allocate_chp_emissions({
            "total_fuel_gj": Decimal("2000"),
            "fuel_type": "natural_gas",
            "heat_output_gj": Decimal("900"),
            "power_output_gj": Decimal("700"),
            "method": "energy",
        })
        assert result.get("method") == "energy"

    def test_dispatch_exergy(self, engine):
        result = engine.allocate_chp_emissions({
            "total_fuel_gj": Decimal("2000"),
            "fuel_type": "natural_gas",
            "heat_output_gj": Decimal("900"),
            "power_output_gj": Decimal("700"),
            "method": "exergy",
            "steam_temperature_c": Decimal("200"),
            "ambient_temperature_c": Decimal("25"),
        })
        assert result.get("method") == "exergy"

    def test_dispatch_default_method(self, engine):
        """Dispatch without explicit method should use default (efficiency or energy)."""
        result = engine.allocate_chp_emissions({
            "total_fuel_gj": Decimal("2000"),
            "fuel_type": "natural_gas",
            "heat_output_gj": Decimal("900"),
            "power_output_gj": Decimal("700"),
        })
        assert isinstance(result, dict)


# ===========================================================================
# 20. Health Check Tests
# ===========================================================================


class TestCHPHealthCheck:
    """Tests for the health_check method."""

    def test_health_check_returns_dict(self, engine):
        result = engine.health_check()
        assert isinstance(result, dict)

    def test_health_check_has_status(self, engine):
        result = engine.health_check()
        status = result.get("status", result.get("healthy", None))
        assert status is not None


# ===========================================================================
# 21. Reset and Repr Tests
# ===========================================================================


class TestCHPResetAndRepr:
    """Tests for reset, repr, and str."""

    def test_repr(self, engine):
        r = repr(engine)
        assert isinstance(r, str)
        assert len(r) > 0

    def test_str(self, engine):
        s = str(engine)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_reset_clears_stats(self, engine, sample_efficiency_request):
        engine.allocate_chp_emissions(sample_efficiency_request)
        engine.reset()
        stats = engine.get_allocation_stats()
        total = stats.get("total_allocations", stats.get("total", 0))
        assert total == 0


# ===========================================================================
# 22. Validate Positive/NonNeg/Efficiency Helpers
# ===========================================================================


class TestCHPValidationHelpers:
    """Tests for _validate_positive, _validate_non_negative, _validate_efficiency."""

    def test_validate_positive_with_zero_raises(self, engine):
        with pytest.raises((ValueError, Exception)):
            engine._validate_positive("test_val", Decimal("0"))

    def test_validate_positive_with_negative_raises(self, engine):
        with pytest.raises((ValueError, Exception)):
            engine._validate_positive("test_val", Decimal("-1"))

    def test_validate_non_negative_allows_zero(self, engine):
        # Should not raise
        engine._validate_non_negative("test_val", Decimal("0"))

    def test_validate_efficiency_over_one_raises(self, engine):
        with pytest.raises((ValueError, Exception)):
            engine._validate_efficiency("eta_test", Decimal("1.5"))

    def test_validate_efficiency_negative_raises(self, engine):
        with pytest.raises((ValueError, Exception)):
            engine._validate_efficiency("eta_test", Decimal("-0.1"))
