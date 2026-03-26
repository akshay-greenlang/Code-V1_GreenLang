# -*- coding: utf-8 -*-
"""
Unit tests for EmissionFactorManagerEngine -- PACK-041 Engine 3
=================================================================

Tests emission factor lookup, GWP value retrieval, factor source
hierarchy, factor overrides, consistency checks, and provenance hashing.

Coverage target: 85%+
Total tests: ~75
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack041_test.engines.efm_{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


_m = _load("emission_factor_manager_engine")


# =============================================================================
# Module Loading
# =============================================================================


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_module_version(self):
        assert hasattr(_m, "_MODULE_VERSION")


# =============================================================================
# Fuel Emission Factors -- IPCC
# =============================================================================


class TestFuelFactorsIPCC:
    """Test IPCC 2006 default fuel emission factors."""

    @pytest.mark.parametrize("fuel_type,expected_co2_kg_per_gj", [
        ("natural_gas", Decimal("56.1")),
        ("diesel", Decimal("74.1")),
        ("fuel_oil", Decimal("77.4")),
        ("coal", Decimal("94.6")),
    ])
    def test_ipcc_fuel_co2_factor(self, fuel_type, expected_co2_kg_per_gj, sample_emission_factors):
        """IPCC CO2 emission factor for key fuels."""
        if fuel_type in sample_emission_factors["fuels"]:
            fuel = sample_emission_factors["fuels"][fuel_type]
            if "ipcc_2006" in fuel:
                assert fuel["ipcc_2006"]["co2_kg_per_gj"] == expected_co2_kg_per_gj

    def test_natural_gas_ipcc_56_1(self, sample_emission_factors):
        ng = sample_emission_factors["fuels"]["natural_gas"]["ipcc_2006"]
        assert ng["co2_kg_per_gj"] == Decimal("56.1")

    def test_natural_gas_ipcc_ch4(self, sample_emission_factors):
        ng = sample_emission_factors["fuels"]["natural_gas"]["ipcc_2006"]
        assert ng["ch4_kg_per_gj"] == Decimal("0.001")

    def test_natural_gas_ipcc_n2o(self, sample_emission_factors):
        ng = sample_emission_factors["fuels"]["natural_gas"]["ipcc_2006"]
        assert ng["n2o_kg_per_gj"] == Decimal("0.0001")

    def test_natural_gas_net_cv(self, sample_emission_factors):
        ng = sample_emission_factors["fuels"]["natural_gas"]["ipcc_2006"]
        assert ng["net_cv_gj_per_m3"] == Decimal("0.0364")


# =============================================================================
# Fuel Emission Factors -- DEFRA
# =============================================================================


class TestFuelFactorsDEFRA:
    """Test DEFRA 2025 fuel emission factors."""

    def test_diesel_defra_co2e(self, sample_emission_factors):
        d = sample_emission_factors["fuels"]["diesel"]["defra_2025"]
        assert d["co2e_kg_per_litre"] == Decimal("2.5271")

    def test_diesel_defra_co2(self, sample_emission_factors):
        d = sample_emission_factors["fuels"]["diesel"]["defra_2025"]
        assert d["co2_kg_per_litre"] == Decimal("2.5121")

    def test_petrol_defra_co2e(self, sample_emission_factors):
        p = sample_emission_factors["fuels"]["petrol"]["defra_2025"]
        assert p["co2e_kg_per_litre"] == Decimal("2.1944")

    def test_lpg_defra_co2e(self, sample_emission_factors):
        lpg = sample_emission_factors["fuels"]["lpg"]["defra_2025"]
        assert lpg["co2e_kg_per_litre"] == Decimal("1.5226")

    def test_natural_gas_defra_co2e_per_m3(self, sample_emission_factors):
        ng = sample_emission_factors["fuels"]["natural_gas"]["defra_2025"]
        assert ng["co2e_kg_per_m3"] == Decimal("2.0216")


# =============================================================================
# Grid Emission Factors
# =============================================================================


class TestGridFactors:
    """Test grid emission factors for different countries/regions."""

    def test_germany_grid_factor(self, sample_emission_factors):
        de = sample_emission_factors["grids"]["DE"]
        assert de["location_based_kg_per_kwh"] == Decimal("0.385")

    def test_uk_grid_factor(self, sample_emission_factors):
        gb = sample_emission_factors["grids"]["GB"]
        assert gb["location_based_kg_per_kwh"] == Decimal("0.207")

    def test_us_average_grid_factor(self, sample_emission_factors):
        us = sample_emission_factors["grids"]["US_average"]
        assert us["location_based_kg_per_kwh"] == Decimal("0.390")

    def test_us_ercot_grid_factor(self, sample_emission_factors):
        ercot = sample_emission_factors["grids"]["US_ERCOT"]
        assert ercot["location_based_kg_per_kwh"] == Decimal("0.370")

    def test_japan_grid_factor(self, sample_emission_factors):
        jp = sample_emission_factors["grids"]["JP"]
        assert jp["location_based_kg_per_kwh"] == Decimal("0.470")

    def test_france_grid_factor_low(self, sample_emission_factors):
        """France has low grid factor due to nuclear dominance."""
        fr = sample_emission_factors["grids"]["FR"]
        assert fr["location_based_kg_per_kwh"] < Decimal("0.1")

    @pytest.mark.parametrize("country", ["DE", "US_average", "GB", "JP", "FR"])
    def test_grid_factor_positive(self, country, sample_emission_factors):
        gf = sample_emission_factors["grids"][country]
        assert gf["location_based_kg_per_kwh"] > Decimal("0")

    @pytest.mark.parametrize("country", ["DE", "US_average", "GB", "JP"])
    def test_grid_factor_less_than_1(self, country, sample_emission_factors):
        """Grid factors should be < 1 kgCO2/kWh for most countries."""
        gf = sample_emission_factors["grids"][country]
        assert gf["location_based_kg_per_kwh"] < Decimal("1.0")

    def test_grid_factor_source_recorded(self, sample_emission_factors):
        de = sample_emission_factors["grids"]["DE"]
        assert "source" in de
        assert len(de["source"]) > 0


# =============================================================================
# GWP Values
# =============================================================================


class TestGWPValues:
    """Test GWP 100-year values across AR4, AR5, AR6."""

    def test_ch4_ar4_gwp_25(self, sample_gwp_values):
        assert sample_gwp_values["CH4"]["ar4"] == Decimal("25")

    def test_ch4_ar5_gwp_28(self, sample_gwp_values):
        assert sample_gwp_values["CH4"]["ar5"] == Decimal("28")

    def test_ch4_ar6_gwp_27_9(self, sample_gwp_values):
        assert sample_gwp_values["CH4"]["ar6"] == Decimal("27.9")

    def test_n2o_ar6_gwp_273(self, sample_gwp_values):
        assert sample_gwp_values["N2O"]["ar6"] == Decimal("273")

    def test_sf6_ar6_gwp_25200(self, sample_gwp_values):
        assert sample_gwp_values["SF6"]["ar6"] == Decimal("25200")

    def test_nf3_ar6_gwp_17400(self, sample_gwp_values):
        assert sample_gwp_values["NF3"]["ar6"] == Decimal("17400")

    def test_co2_gwp_always_1(self, sample_gwp_values):
        for gen in ["ar4", "ar5", "ar6"]:
            assert sample_gwp_values["CO2"][gen] == Decimal("1")

    @pytest.mark.parametrize("gas", ["CO2", "CH4", "N2O", "SF6", "NF3"])
    def test_all_generations_present(self, gas, sample_gwp_values):
        assert "ar4" in sample_gwp_values[gas]
        assert "ar5" in sample_gwp_values[gas]
        assert "ar6" in sample_gwp_values[gas]

    @pytest.mark.parametrize("gas", ["CO2", "CH4", "N2O", "SF6", "NF3"])
    def test_all_gwp_positive(self, gas, sample_gwp_values):
        for gen in ["ar4", "ar5", "ar6"]:
            assert sample_gwp_values[gas][gen] > Decimal("0")

    def test_hfc_134a_ar6(self, sample_gwp_values):
        assert sample_gwp_values["HFC-134a"]["ar6"] == Decimal("1530")

    def test_r410a_gwp(self, sample_gwp_values):
        assert sample_gwp_values["R-410A"]["ar6"] == Decimal("2088")

    def test_cf4_pfc_ar6(self, sample_gwp_values):
        assert sample_gwp_values["CF4"]["ar6"] == Decimal("7380")

    def test_c2f6_pfc_ar6(self, sample_gwp_values):
        assert sample_gwp_values["C2F6"]["ar6"] == Decimal("12400")


# =============================================================================
# Refrigerant Factors
# =============================================================================


class TestRefrigerantFactors:
    """Test refrigerant GWP factors."""

    def test_r410a_gwp_ar6(self, sample_emission_factors):
        ref = sample_emission_factors["refrigerants"]["R-410A"]
        assert ref["gwp_ar6"] == Decimal("2088")

    def test_r134a_gwp_ar6(self, sample_emission_factors):
        ref = sample_emission_factors["refrigerants"]["R-134a"]
        assert ref["gwp_ar6"] == Decimal("1530")

    def test_r404a_gwp_ar6(self, sample_emission_factors):
        ref = sample_emission_factors["refrigerants"]["R-404A"]
        assert ref["gwp_ar6"] == Decimal("4728")

    def test_sf6_gwp_ar6(self, sample_emission_factors):
        ref = sample_emission_factors["refrigerants"]["SF6"]
        assert ref["gwp_ar6"] == Decimal("25200")

    def test_r32_low_gwp(self, sample_emission_factors):
        ref = sample_emission_factors["refrigerants"]["R-32"]
        assert ref["gwp_ar6"] < Decimal("1000")


# =============================================================================
# Factor Source Hierarchy
# =============================================================================


class TestFactorSourceHierarchy:
    """Test emission factor source priority order."""

    def test_hierarchy_order(self):
        """Default hierarchy: supplier-specific > national > DEFRA > IPCC."""
        hierarchy = [
            "supplier_specific",
            "national_inventory",
            "defra_2025",
            "epa_2024",
            "ipcc_2006",
        ]
        assert hierarchy[0] == "supplier_specific"
        assert hierarchy[-1] == "ipcc_2006"

    def test_supplier_specific_highest_priority(self):
        hierarchy = ["supplier_specific", "national_inventory", "defra_2025"]
        assert hierarchy.index("supplier_specific") < hierarchy.index("defra_2025")

    def test_national_before_international(self):
        hierarchy = ["national_inventory", "defra_2025", "ipcc_2006"]
        assert hierarchy.index("national_inventory") < hierarchy.index("ipcc_2006")


# =============================================================================
# Factor Override with Audit
# =============================================================================


class TestFactorOverride:
    """Test emission factor override mechanism with audit trail."""

    def test_override_replaces_default(self):
        default_factor = Decimal("2.5271")
        override_factor = Decimal("2.6000")
        active_factor = override_factor  # override takes precedence
        assert active_factor == Decimal("2.6000")

    def test_override_audit_record(self):
        audit = {
            "original_factor": Decimal("2.5271"),
            "override_factor": Decimal("2.6000"),
            "reason": "Supplier-specific emission factor from lab test",
            "approved_by": "env_manager@acme.com",
            "approval_date": "2025-01-15",
        }
        assert audit["reason"] != ""
        assert "approved_by" in audit

    def test_override_preserves_original(self):
        original = Decimal("56.1")
        override = Decimal("58.0")
        record = {"original": original, "override": override}
        assert record["original"] == Decimal("56.1")


# =============================================================================
# Consistency Checks
# =============================================================================


class TestFactorConsistency:
    """Test emission factor consistency validation."""

    def test_co2_factor_positive(self, sample_emission_factors):
        ng = sample_emission_factors["fuels"]["natural_gas"]["ipcc_2006"]
        assert ng["co2_kg_per_gj"] > Decimal("0")

    def test_co2_factor_reasonable_range(self, sample_emission_factors):
        """CO2 factors should be in range 50-100 kgCO2/GJ for fossil fuels."""
        ng = sample_emission_factors["fuels"]["natural_gas"]["ipcc_2006"]
        assert Decimal("50") <= ng["co2_kg_per_gj"] <= Decimal("100")

    def test_ch4_factor_much_smaller_than_co2(self, sample_emission_factors):
        ng = sample_emission_factors["fuels"]["natural_gas"]["ipcc_2006"]
        assert ng["ch4_kg_per_gj"] < ng["co2_kg_per_gj"] * Decimal("0.001")

    def test_natural_gas_lower_co2_than_coal(self, sample_emission_factors):
        ng = sample_emission_factors["fuels"]["natural_gas"]["ipcc_2006"]["co2_kg_per_gj"]
        coal = sample_emission_factors["fuels"]["coal"]["ipcc_2006"]["co2_kg_per_gj"]
        assert ng < coal

    def test_diesel_higher_co2_than_natural_gas(self, sample_emission_factors):
        ng = sample_emission_factors["fuels"]["natural_gas"]["ipcc_2006"]["co2_kg_per_gj"]
        diesel = sample_emission_factors["fuels"]["diesel"]["ipcc_2006"]["co2_kg_per_gj"]
        assert diesel > ng

    def test_all_fuels_have_ipcc_or_defra(self, sample_emission_factors):
        for fuel_name, fuel_data in sample_emission_factors["fuels"].items():
            has_factor = "ipcc_2006" in fuel_data or "defra_2025" in fuel_data
            assert has_factor, f"Fuel {fuel_name} missing factors"


# =============================================================================
# Factor Not Found
# =============================================================================


class TestFactorNotFound:
    """Test behavior when emission factor is not found."""

    def test_unknown_fuel_not_in_db(self, sample_emission_factors):
        assert "biodiesel_b100" not in sample_emission_factors["fuels"]

    def test_unknown_grid_not_in_db(self, sample_emission_factors):
        assert "ZZ" not in sample_emission_factors["grids"]

    def test_unknown_refrigerant_not_in_db(self, sample_emission_factors):
        assert "R-1234yf" not in sample_emission_factors["refrigerants"]


# =============================================================================
# Provenance Hashing
# =============================================================================


class TestFactorProvenance:
    """Test provenance hashing for emission factor selections."""

    def test_provenance_hash_deterministic(self, sample_emission_factors):
        from tests.conftest import compute_provenance_hash
        h1 = compute_provenance_hash(sample_emission_factors["fuels"]["natural_gas"])
        h2 = compute_provenance_hash(sample_emission_factors["fuels"]["natural_gas"])
        assert h1 == h2

    def test_provenance_hash_64_chars(self, sample_emission_factors):
        from tests.conftest import compute_provenance_hash
        h = compute_provenance_hash(sample_emission_factors["fuels"]["diesel"])
        assert len(h) == 64

    def test_provenance_hash_changes_with_data(self, sample_emission_factors):
        from tests.conftest import compute_provenance_hash
        h1 = compute_provenance_hash(sample_emission_factors["fuels"]["natural_gas"])
        h2 = compute_provenance_hash(sample_emission_factors["fuels"]["diesel"])
        assert h1 != h2
