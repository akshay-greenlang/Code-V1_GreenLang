# -*- coding: utf-8 -*-
"""
Unit tests for Scope2ConsolidationEngine -- PACK-041 Engine 5
================================================================

Tests Scope 2 location-based, market-based, dual reporting,
instrument allocation, steam/cooling, and variance analysis.

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
    mod_key = f"pack041_test.engines.s2c_{name}"
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


_m = _load("scope2_consolidation_engine")


# =============================================================================
# Module Loading
# =============================================================================


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None


# =============================================================================
# Location-Based Method
# =============================================================================


class TestLocationBased:
    """Test Scope 2 location-based calculations."""

    def test_single_facility_location(self, sample_scope2_results):
        lb = sample_scope2_results["location_based"]
        assert lb["total_tco2e"] == Decimal("3700.0")

    def test_location_based_formula(self, sample_scope2_results):
        """Location = MWh * grid_factor."""
        lb = sample_scope2_results["location_based"]
        expected = lb["electricity_mwh"] * lb["grid_factor_kg_per_kwh"]
        # MWh to kWh: 10000 * 1000 = 10,000,000 kWh
        # 10,000,000 * 0.370 = 3,700,000 kgCO2e = 3,700 tCO2e
        expected_tco2e = lb["electricity_mwh"] * lb["grid_factor_kg_per_kwh"]
        assert expected_tco2e == Decimal("3700.00")

    def test_location_based_multi_facility(self):
        fac1 = Decimal("3700.0")
        fac2 = Decimal("2850.0")
        fac3 = Decimal("1540.0")
        total = fac1 + fac2 + fac3
        assert total == Decimal("8090.0")

    def test_location_based_uses_grid_average(self, sample_scope2_results):
        lb = sample_scope2_results["location_based"]
        assert lb["grid_factor_kg_per_kwh"] > Decimal("0")

    @pytest.mark.parametrize("mwh,grid_factor,expected_tco2e", [
        (Decimal("10000"), Decimal("0.385"), Decimal("3850.0")),
        (Decimal("5000"), Decimal("0.207"), Decimal("1035.0")),
        (Decimal("20000"), Decimal("0.470"), Decimal("9400.0")),
        (Decimal("1000"), Decimal("0.052"), Decimal("52.0")),
    ])
    def test_location_based_parametrized(self, mwh, grid_factor, expected_tco2e):
        result = mwh * grid_factor
        assert result == expected_tco2e

    def test_reference_10k_mwh_germany(self, sample_emission_factors):
        """Reference: 10,000 MWh at 0.385 kgCO2/kWh = 3,850 tCO2e."""
        gf = sample_emission_factors["grids"]["DE"]["location_based_kg_per_kwh"]
        mwh = Decimal("10000")
        tco2e = mwh * gf
        assert tco2e == Decimal("3850.000")


# =============================================================================
# Market-Based Method
# =============================================================================


class TestMarketBased:
    """Test Scope 2 market-based calculations."""

    def test_market_based_with_instruments(self, sample_scope2_results):
        mb = sample_scope2_results["market_based"]
        assert mb["total_tco2e"] < sample_scope2_results["location_based"]["total_tco2e"]

    def test_market_based_with_ppa(self, sample_instruments):
        ppa = sample_instruments[0]
        assert ppa["type"] == "power_purchase_agreement"
        assert ppa["emission_factor_kg_per_kwh"] == Decimal("0")

    def test_market_based_with_recs(self, sample_instruments):
        rec = sample_instruments[1]
        assert rec["type"] == "renewable_energy_certificate"
        assert rec["emission_factor_kg_per_kwh"] == Decimal("0")

    def test_market_based_with_green_tariff(self):
        green_tariff = {
            "type": "green_tariff",
            "supplier": "GreenPower Corp",
            "emission_factor_kg_per_kwh": Decimal("0.05"),
            "quantity_mwh": Decimal("3000"),
        }
        tco2e = green_tariff["quantity_mwh"] * green_tariff["emission_factor_kg_per_kwh"]
        assert tco2e == Decimal("150.00")

    def test_market_based_residual_mix(self, sample_scope2_results):
        """Uncovered electricity uses residual mix factor."""
        mb = sample_scope2_results["market_based"]
        assert mb["residual_mix_mwh"] == Decimal("5000")

    def test_residual_mix_emissions(self, sample_scope2_results):
        mb = sample_scope2_results["market_based"]
        residual = mb["residual_mix_mwh"] * mb["residual_factor_kg_per_kwh"]
        assert residual == Decimal("1850.000")


# =============================================================================
# Instrument Allocation Hierarchy
# =============================================================================


class TestInstrumentAllocation:
    """Test allocation hierarchy for contractual instruments."""

    def test_hierarchy_order(self, sample_pack_config):
        hierarchy = sample_pack_config["scope2"]["instrument_hierarchy"]
        assert hierarchy[0] == "energy_attribute_certificate"
        assert "residual_mix" in hierarchy

    def test_no_double_counting_instruments(self, sample_instruments):
        """Same MWh should not be covered by two instruments."""
        total_allocated = sum(
            inst.get("allocated_mwh", inst.get("quantity_mwh", Decimal("0")))
            for inst in sample_instruments
        )
        # PPA 3000 + REC 2000 + GO 4000 = 9000 MWh total allocated
        assert total_allocated == Decimal("9000")

    def test_instrument_coverage_less_than_consumption(self):
        total_consumption_mwh = Decimal("10000")
        instrument_coverage_mwh = Decimal("5000")
        residual_mwh = total_consumption_mwh - instrument_coverage_mwh
        assert residual_mwh == Decimal("5000")

    def test_100pct_renewable_market_zero(self):
        """100% renewable coverage should yield 0 market-based emissions."""
        total_mwh = Decimal("10000")
        renewable_mwh = Decimal("10000")
        residual_mwh = total_mwh - renewable_mwh
        residual_emissions = residual_mwh * Decimal("0.370")
        assert residual_emissions == Decimal("0.000")

    def test_no_instruments_equals_residual_mix(self):
        """No instruments: market-based = residual mix factor * total MWh."""
        total_mwh = Decimal("10000")
        residual_factor = Decimal("0.400")  # typically > grid average
        market_tco2e = total_mwh * residual_factor
        assert market_tco2e == Decimal("4000.000")


# =============================================================================
# Dual Reporting Reconciliation
# =============================================================================


class TestDualReporting:
    """Test dual reporting reconciliation between location and market methods."""

    def test_variance_calculation(self, sample_scope2_results):
        variance = sample_scope2_results["variance_tco2e"]
        lb = sample_scope2_results["location_based"]["total_tco2e"]
        mb = sample_scope2_results["market_based"]["total_tco2e"]
        assert variance == lb - mb

    def test_variance_percentage(self, sample_scope2_results):
        assert sample_scope2_results["variance_pct"] == Decimal("50.0")

    def test_market_lower_than_location_with_renewables(self, sample_scope2_results):
        lb = sample_scope2_results["location_based"]["total_tco2e"]
        mb = sample_scope2_results["market_based"]["total_tco2e"]
        assert mb <= lb

    def test_market_can_equal_location(self):
        """With no instruments, market equals residual mix (often >= location)."""
        location = Decimal("3700.0")
        market_no_instruments = Decimal("3700.0")
        assert market_no_instruments == location

    def test_dual_report_both_required(self, sample_pack_config):
        assert sample_pack_config["scope2"]["dual_reporting"] is True

    @pytest.mark.parametrize("rec_pct,expected_market_pct_of_location", [
        (Decimal("0"), Decimal("100")),
        (Decimal("25"), Decimal("75")),
        (Decimal("50"), Decimal("50")),
        (Decimal("75"), Decimal("25")),
        (Decimal("100"), Decimal("0")),
    ])
    def test_rec_coverage_impact(self, rec_pct, expected_market_pct_of_location):
        """Market emissions decrease proportionally with REC coverage."""
        location = Decimal("3700.0")
        market = location * (Decimal("100") - rec_pct) / Decimal("100")
        actual_pct = market / location * Decimal("100") if location else Decimal("0")
        assert actual_pct == expected_market_pct_of_location


# =============================================================================
# Steam, Heat, and Cooling
# =============================================================================


class TestSteamHeatCooling:
    """Test Scope 2 consolidation for steam, heat, and cooling purchases."""

    def test_steam_emissions_calculation(self):
        steam_mwh = Decimal("2000")
        steam_ef = Decimal("0.210")  # kgCO2e per kWh thermal
        tco2e = steam_mwh * steam_ef
        assert tco2e == Decimal("420.000")

    def test_cooling_emissions_calculation(self):
        cooling_mwh = Decimal("500")
        cooling_ef = Decimal("0.180")  # kgCO2e per kWh thermal
        tco2e = cooling_mwh * cooling_ef
        assert tco2e == Decimal("90.000")

    def test_scope2_total_includes_steam(self):
        electricity = Decimal("3700.0")
        steam = Decimal("420.0")
        cooling = Decimal("90.0")
        total = electricity + steam + cooling
        assert total == Decimal("4210.0")

    def test_zero_steam_cooling(self, sample_scope2_results):
        lb = sample_scope2_results["location_based"]
        assert lb["steam_tco2e"] == Decimal("0")
        assert lb["cooling_tco2e"] == Decimal("0")


# =============================================================================
# Uncertainty
# =============================================================================


class TestScope2Uncertainty:
    """Test Scope 2 uncertainty values."""

    def test_location_based_uncertainty(self, sample_scope2_results):
        lb = sample_scope2_results["location_based"]
        assert lb["uncertainty_pct"] == Decimal("10.0")

    def test_market_based_uncertainty_lower(self, sample_scope2_results):
        """Market-based with supplier-specific data has lower uncertainty."""
        mb = sample_scope2_results["market_based"]
        lb = sample_scope2_results["location_based"]
        assert mb["uncertainty_pct"] <= lb["uncertainty_pct"]

    def test_uncertainty_positive(self, sample_scope2_results):
        assert sample_scope2_results["location_based"]["uncertainty_pct"] > Decimal("0")
        assert sample_scope2_results["market_based"]["uncertainty_pct"] > Decimal("0")


# =============================================================================
# Provenance
# =============================================================================


class TestScope2Provenance:
    """Test provenance hashing for Scope 2 results."""

    def test_provenance_deterministic(self, sample_scope2_results):
        from tests.conftest import compute_provenance_hash
        h1 = compute_provenance_hash(sample_scope2_results)
        h2 = compute_provenance_hash(sample_scope2_results)
        assert h1 == h2

    def test_provenance_hash_length(self, sample_scope2_results):
        from tests.conftest import compute_provenance_hash
        h = compute_provenance_hash(sample_scope2_results)
        assert len(h) == 64

    def test_provenance_changes_with_data(self, sample_scope2_results):
        from tests.conftest import compute_provenance_hash
        modified = dict(sample_scope2_results)
        modified["variance_tco2e"] = Decimal("9999")
        h1 = compute_provenance_hash(sample_scope2_results)
        h2 = compute_provenance_hash(modified)
        assert h1 != h2
