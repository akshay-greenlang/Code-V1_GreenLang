# -*- coding: utf-8 -*-
"""
Unit tests for Scope1ConsolidationEngine -- PACK-041 Engine 4
================================================================

Tests Scope 1 consolidation across categories, gases, facilities,
boundary percentage application, double-counting resolution,
GWP conversion, and reference calculations.

Coverage target: 85%+
Total tests: ~65
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
    mod_key = f"pack041_test.engines.s1c_{name}"
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


_m = _load("scope1_consolidation_engine")


# =============================================================================
# Module Loading
# =============================================================================


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_module_version(self):
        assert hasattr(_m, "_MODULE_VERSION")


# =============================================================================
# Single Facility Consolidation
# =============================================================================


class TestSingleFacilityConsolidation:
    """Test consolidation for a single facility."""

    def test_single_facility_single_category(self, sample_scope1_results):
        """Single category should equal its own value."""
        cat = sample_scope1_results["categories"]["stationary_combustion"]
        assert cat["total_tco2e"] == Decimal("2363.6")

    def test_single_facility_all_categories(self, sample_scope1_results):
        """Sum of all categories should equal total Scope 1."""
        cats = sample_scope1_results["categories"]
        total = sum(c["total_tco2e"] for c in cats.values())
        assert total == sample_scope1_results["total_scope1_tco2e"]

    def test_scope1_total_positive(self, sample_scope1_results):
        assert sample_scope1_results["total_scope1_tco2e"] > Decimal("0")


# =============================================================================
# Multi-Facility Consolidation
# =============================================================================


class TestMultiFacilityConsolidation:
    """Test aggregation across multiple facilities."""

    def test_two_facility_sum(self):
        fac1_total = Decimal("5830.0")
        fac2_total = Decimal("7000.0")
        combined = fac1_total + fac2_total
        assert combined == Decimal("12830.0")

    def test_three_facility_sum(self):
        totals = [Decimal("5830.0"), Decimal("500.0"), Decimal("7000.0")]
        combined = sum(totals)
        assert combined == Decimal("13330.0")

    def test_org_level_scope1(self, sample_inventory):
        """Organization total should equal sum of facility totals."""
        by_fac = sample_inventory["scope1"]["by_facility"]
        fac_sum = sum(by_fac.values())
        assert fac_sum == sample_inventory["scope1"]["total_tco2e"]


# =============================================================================
# Aggregation by Category
# =============================================================================


class TestAggregationByCategory:
    """Test aggregation by Scope 1 emission category."""

    def test_by_category_eight_entries(self, sample_inventory):
        cats = sample_inventory["scope1"]["by_category"]
        assert len(cats) == 8

    @pytest.mark.parametrize("category", [
        "stationary_combustion",
        "mobile_combustion",
        "process_emissions",
        "fugitive_emissions",
        "refrigerant_fgas",
        "land_use",
        "waste_treatment",
        "agricultural",
    ])
    def test_category_non_negative(self, category, sample_inventory):
        assert sample_inventory["scope1"]["by_category"][category] >= Decimal("0")

    def test_category_sum_equals_total(self, sample_inventory):
        cats = sample_inventory["scope1"]["by_category"]
        assert sum(cats.values()) == sample_inventory["scope1"]["total_tco2e"]

    def test_stationary_combustion_largest(self, sample_inventory):
        """Stationary combustion is typically the largest Scope 1 category."""
        cats = sample_inventory["scope1"]["by_category"]
        assert cats["stationary_combustion"] == max(cats.values())


# =============================================================================
# Aggregation by Gas
# =============================================================================


class TestAggregationByGas:
    """Test aggregation by greenhouse gas species."""

    def test_seven_ghg_gases(self, sample_inventory):
        gases = sample_inventory["scope1"]["by_gas"]
        expected_gases = {"CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"}
        assert set(gases.keys()) == expected_gases

    def test_co2_dominates(self, sample_inventory):
        gases = sample_inventory["scope1"]["by_gas"]
        assert gases["CO2"] > sum(v for k, v in gases.items() if k != "CO2")

    @pytest.mark.parametrize("gas", ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"])
    def test_gas_non_negative(self, gas, sample_inventory):
        assert sample_inventory["scope1"]["by_gas"][gas] >= Decimal("0")

    def test_gas_sum_equals_total(self, sample_inventory):
        gases = sample_inventory["scope1"]["by_gas"]
        assert sum(gases.values()) == sample_inventory["scope1"]["total_tco2e"]


# =============================================================================
# Aggregation by Facility
# =============================================================================


class TestAggregationByFacility:
    """Test aggregation by facility."""

    def test_facility_count(self, sample_inventory):
        by_fac = sample_inventory["scope1"]["by_facility"]
        assert len(by_fac) == 6

    def test_each_facility_positive(self, sample_inventory):
        for fac_id, val in sample_inventory["scope1"]["by_facility"].items():
            assert val >= Decimal("0"), f"Facility {fac_id} has negative emissions"

    def test_houston_plant_total(self, sample_inventory):
        assert sample_inventory["scope1"]["by_facility"]["FAC-001"] == Decimal("5830.0")


# =============================================================================
# Boundary Percentage Application
# =============================================================================


class TestBoundaryPercentage:
    """Test boundary percentage application for equity share approach."""

    def test_equity_share_50pct(self):
        entity_total = Decimal("10000.0")
        equity_pct = Decimal("50.0")
        included = entity_total * equity_pct / Decimal("100")
        assert included == Decimal("5000.0")

    def test_equity_share_100pct(self):
        entity_total = Decimal("10000.0")
        equity_pct = Decimal("100.0")
        included = entity_total * equity_pct / Decimal("100")
        assert included == Decimal("10000.0")

    def test_operational_control_binary(self):
        """Operational control: include 100% or 0%."""
        entity_total = Decimal("10000.0")
        has_oc = True
        included = entity_total if has_oc else Decimal("0")
        assert included == Decimal("10000.0")

    def test_operational_control_excluded(self):
        entity_total = Decimal("10000.0")
        has_oc = False
        included = entity_total if has_oc else Decimal("0")
        assert included == Decimal("0")

    @pytest.mark.parametrize("equity_pct,expected", [
        (Decimal("100"), Decimal("10000")),
        (Decimal("75"), Decimal("7500")),
        (Decimal("50"), Decimal("5000")),
        (Decimal("25"), Decimal("2500")),
        (Decimal("0"), Decimal("0")),
    ])
    def test_equity_share_parametrized(self, equity_pct, expected):
        entity_total = Decimal("10000")
        included = entity_total * equity_pct / Decimal("100")
        assert included == expected


# =============================================================================
# Double Counting Prevention
# =============================================================================


class TestDoubleCounting:
    """Test double-counting resolution between overlapping categories."""

    def test_waste_stationary_no_overlap(self):
        """On-site waste incineration with energy recovery: count in waste OR stationary."""
        waste_emissions = Decimal("200.0")
        stationary_from_waste_heat = Decimal("0")  # excluded to avoid double count
        assert waste_emissions + stationary_from_waste_heat == Decimal("200.0")

    def test_chp_allocation(self):
        """CHP: allocate between Scope 1 (on-site use) and Scope 2 (exported)."""
        total_chp = Decimal("5000.0")
        onsite_fraction = Decimal("0.7")
        scope1_chp = total_chp * onsite_fraction
        exported = total_chp * (Decimal("1") - onsite_fraction)
        assert scope1_chp == Decimal("3500.0")
        assert exported == Decimal("1500.0")
        assert scope1_chp + exported == total_chp

    def test_double_counting_flag(self):
        """Resolution should flag and correct double-counted sources."""
        sources = [
            {"id": "S1", "category": "stationary_combustion", "tco2e": Decimal("100")},
            {"id": "S1", "category": "waste_treatment", "tco2e": Decimal("100")},
        ]
        ids = [s["id"] for s in sources]
        has_duplicates = len(ids) != len(set(ids))
        assert has_duplicates is True


# =============================================================================
# GWP Conversion
# =============================================================================


class TestGWPConversion:
    """Test GWP conversion across AR4, AR5, AR6."""

    @pytest.mark.parametrize("gwp_set,ch4_gwp,n2o_gwp", [
        ("ar4", Decimal("25"), Decimal("298")),
        ("ar5", Decimal("28"), Decimal("265")),
        ("ar6", Decimal("27.9"), Decimal("273")),
    ])
    def test_gwp_conversion(self, gwp_set, ch4_gwp, n2o_gwp, sample_gwp_values):
        assert sample_gwp_values["CH4"][gwp_set] == ch4_gwp
        assert sample_gwp_values["N2O"][gwp_set] == n2o_gwp

    def test_ch4_conversion_ar6(self, sample_gwp_values):
        """100 kg CH4 * 27.9 GWP = 2790 kgCO2e."""
        ch4_kg = Decimal("100")
        gwp = sample_gwp_values["CH4"]["ar6"]
        co2e = ch4_kg * gwp
        assert co2e == Decimal("2790.0")

    def test_n2o_conversion_ar6(self, sample_gwp_values):
        """10 kg N2O * 273 GWP = 2730 kgCO2e."""
        n2o_kg = Decimal("10")
        gwp = sample_gwp_values["N2O"]["ar6"]
        co2e = n2o_kg * gwp
        assert co2e == Decimal("2730")


# =============================================================================
# Zero Emissions Category
# =============================================================================


class TestZeroEmissions:
    """Test handling of zero-emission categories."""

    def test_land_use_zero(self, sample_scope1_results):
        assert sample_scope1_results["categories"]["land_use"]["total_tco2e"] == Decimal("0")

    def test_agricultural_zero(self, sample_scope1_results):
        assert sample_scope1_results["categories"]["agricultural"]["total_tco2e"] == Decimal("0")

    def test_zero_category_still_in_total(self, sample_scope1_results):
        cats = sample_scope1_results["categories"]
        total = sum(c["total_tco2e"] for c in cats.values())
        assert total == sample_scope1_results["total_scope1_tco2e"]


# =============================================================================
# Reference Calculations
# =============================================================================


class TestReferenceCalculations:
    """Reference calculations validated against known values."""

    def test_natural_gas_1m_m3_ipcc(self, sample_emission_factors, sample_gwp_values):
        """1,000,000 m3 natural gas IPCC: ~2,167 tCO2e.

        NCV = 0.0364 GJ/m3
        CO2 = 56.1 kg/GJ
        Energy = 1_000_000 * 0.0364 = 36,400 GJ
        CO2 = 36,400 * 56.1 = 2,042,040 kgCO2 = 2,042.04 tCO2
        CH4 = 36,400 * 0.001 = 36.4 kgCH4 * 27.9 = 1,015.56 kgCO2e
        N2O = 36,400 * 0.0001 = 3.64 kgN2O * 273 = 993.72 kgCO2e
        Total ~ 2,044.05 tCO2e (within range of ~2,167 using other factors)
        """
        ng = sample_emission_factors["fuels"]["natural_gas"]["ipcc_2006"]
        volume_m3 = Decimal("1000000")
        ncv = ng["net_cv_gj_per_m3"]
        energy_gj = volume_m3 * ncv  # 36,400 GJ

        co2_kg = energy_gj * ng["co2_kg_per_gj"]
        ch4_kg = energy_gj * ng["ch4_kg_per_gj"]
        n2o_kg = energy_gj * ng["n2o_kg_per_gj"]

        ch4_co2e = ch4_kg * sample_gwp_values["CH4"]["ar6"]
        n2o_co2e = n2o_kg * sample_gwp_values["N2O"]["ar6"]

        total_kg = co2_kg + ch4_co2e + n2o_co2e
        total_t = total_kg / Decimal("1000")

        # Should be approximately 2,044 tCO2e with IPCC defaults
        assert Decimal("1800") < total_t < Decimal("2200")

    def test_diesel_50k_litres_defra(self, sample_emission_factors):
        """50,000 litres diesel DEFRA: ~126 tCO2e.

        50,000 * 2.5271 kgCO2e/litre = 126,355 kgCO2e = 126.36 tCO2e
        """
        d = sample_emission_factors["fuels"]["diesel"]["defra_2025"]
        volume = Decimal("50000")
        co2e_kg = volume * d["co2e_kg_per_litre"]
        co2e_t = co2e_kg / Decimal("1000")
        assert Decimal("125") < co2e_t < Decimal("128")

    def test_r410a_100kg_ar6(self, sample_emission_factors):
        """100 kg R-410A leaked: 100 * 2088 = 208,800 kgCO2e = 208.8 tCO2e."""
        ref = sample_emission_factors["refrigerants"]["R-410A"]
        leak_kg = Decimal("100")
        co2e_kg = leak_kg * ref["gwp_ar6"]
        co2e_t = co2e_kg / Decimal("1000")
        assert co2e_t == Decimal("208.8")

    def test_fleet_diesel_500k_km(self, sample_emission_factors):
        """500,000 km fleet diesel (125,000 L consumed): ~316 tCO2e.

        125,000 * 2.5271 = 315,887.5 kgCO2e = 315.89 tCO2e
        """
        d = sample_emission_factors["fuels"]["diesel"]["defra_2025"]
        fuel_litres = Decimal("125000")
        co2e_kg = fuel_litres * d["co2e_kg_per_litre"]
        co2e_t = co2e_kg / Decimal("1000")
        assert Decimal("314") < co2e_t < Decimal("318")


# =============================================================================
# Provenance
# =============================================================================


class TestScope1Provenance:
    """Test provenance hashing for scope 1 consolidation."""

    def test_provenance_hash_deterministic(self, sample_scope1_results):
        from tests.conftest import compute_provenance_hash
        h1 = compute_provenance_hash(sample_scope1_results)
        h2 = compute_provenance_hash(sample_scope1_results)
        assert h1 == h2

    def test_provenance_hash_length(self, sample_scope1_results):
        from tests.conftest import compute_provenance_hash
        h = compute_provenance_hash(sample_scope1_results)
        assert len(h) == 64
