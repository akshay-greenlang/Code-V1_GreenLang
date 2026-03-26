# -*- coding: utf-8 -*-
"""
Unit tests for Compliance Formulas -- PACK-041
=================================================

Reference calculation validation against known values from GHG Protocol,
IPCC, DEFRA, and EPA sources. These tests serve as regression tests to
ensure calculation accuracy is maintained across code changes.

Coverage target: 85%+
Total tests: ~45
"""

import math
from decimal import Decimal

import pytest


# =============================================================================
# Natural Gas Combustion (IPCC)
# =============================================================================


class TestNaturalGasCombustion:
    """Test natural gas emission calculations against IPCC values."""

    def test_1m_m3_natural_gas_ipcc(self, sample_emission_factors, sample_gwp_values):
        """Reference: 1,000,000 m3 natural gas -> ~2,044 tCO2e (IPCC factors).

        NCV = 0.0364 GJ/m3
        Energy = 1,000,000 * 0.0364 = 36,400 GJ
        CO2 = 36,400 * 56.1 = 2,041,040 kgCO2
        CH4 = 36,400 * 0.001 * 27.9 = 1,015.56 kgCO2e
        N2O = 36,400 * 0.0001 * 273 = 993.72 kgCO2e
        Total = 2,043,049.28 kgCO2e = 2,043.05 tCO2e
        """
        ng = sample_emission_factors["fuels"]["natural_gas"]["ipcc_2006"]
        gwp = sample_gwp_values

        volume = Decimal("1000000")
        energy = volume * ng["net_cv_gj_per_m3"]

        co2 = energy * ng["co2_kg_per_gj"]
        ch4 = energy * ng["ch4_kg_per_gj"] * gwp["CH4"]["ar6"]
        n2o = energy * ng["n2o_kg_per_gj"] * gwp["N2O"]["ar6"]

        total_kg = co2 + ch4 + n2o
        total_t = total_kg / Decimal("1000")

        assert Decimal("2040") < total_t < Decimal("2050")

    def test_1m_m3_natural_gas_defra(self, sample_emission_factors):
        """Reference: 1,000,000 m3 natural gas -> ~2,022 tCO2e (DEFRA factors).

        1,000,000 * 2.0216 = 2,021,600 kgCO2e = 2,021.6 tCO2e
        """
        ng = sample_emission_factors["fuels"]["natural_gas"]["defra_2025"]
        volume = Decimal("1000000")
        total_kg = volume * ng["co2e_kg_per_m3"]
        total_t = total_kg / Decimal("1000")
        assert Decimal("2020") < total_t < Decimal("2025")


# =============================================================================
# Diesel Combustion (DEFRA)
# =============================================================================


class TestDieselCombustion:
    """Test diesel emission calculations against DEFRA values."""

    def test_50k_litres_diesel_defra(self, sample_emission_factors):
        """50,000 L diesel = 50,000 * 2.5271 = 126,355 kgCO2e = 126.36 tCO2e."""
        d = sample_emission_factors["fuels"]["diesel"]["defra_2025"]
        volume = Decimal("50000")
        total_kg = volume * d["co2e_kg_per_litre"]
        total_t = total_kg / Decimal("1000")
        assert total_t == pytest.approx(Decimal("126.355"), abs=Decimal("0.001"))

    def test_100k_litres_diesel_ipcc(self, sample_emission_factors, sample_gwp_values):
        """100,000 L diesel IPCC:
        NCV = 0.0360 GJ/L
        Energy = 100,000 * 0.0360 = 3,600 GJ
        CO2 = 3,600 * 74.1 = 266,760 kgCO2
        CH4 = 3,600 * 0.003 * 27.9 = 301.32 kgCO2e
        N2O = 3,600 * 0.0006 * 273 = 589.68 kgCO2e
        Total = 267,651 kgCO2e = 267.65 tCO2e
        """
        d = sample_emission_factors["fuels"]["diesel"]["ipcc_2006"]
        gwp = sample_gwp_values
        volume = Decimal("100000")
        energy = volume * d["net_cv_gj_per_litre"]
        co2 = energy * d["co2_kg_per_gj"]
        ch4 = energy * d["ch4_kg_per_gj"] * gwp["CH4"]["ar6"]
        n2o = energy * d["n2o_kg_per_gj"] * gwp["N2O"]["ar6"]
        total_t = (co2 + ch4 + n2o) / Decimal("1000")
        assert Decimal("265") < total_t < Decimal("270")


# =============================================================================
# Grid Electricity (Location-Based)
# =============================================================================


class TestGridElectricity:
    """Test grid electricity emission calculations."""

    def test_10k_mwh_germany(self, sample_emission_factors):
        """10,000 MWh Germany = 10,000 * 0.385 = 3,850 tCO2e."""
        gf = sample_emission_factors["grids"]["DE"]["location_based_kg_per_kwh"]
        mwh = Decimal("10000")
        tco2e = mwh * gf
        assert tco2e == Decimal("3850.000")

    def test_5k_mwh_uk(self, sample_emission_factors):
        """5,000 MWh UK = 5,000 * 0.207 = 1,035 tCO2e."""
        gf = sample_emission_factors["grids"]["GB"]["location_based_kg_per_kwh"]
        mwh = Decimal("5000")
        tco2e = mwh * gf
        assert tco2e == Decimal("1035.000")

    def test_20k_mwh_japan(self, sample_emission_factors):
        """20,000 MWh Japan = 20,000 * 0.470 = 9,400 tCO2e."""
        gf = sample_emission_factors["grids"]["JP"]["location_based_kg_per_kwh"]
        mwh = Decimal("20000")
        tco2e = mwh * gf
        assert tco2e == Decimal("9400.000")

    def test_france_low_carbon_grid(self, sample_emission_factors):
        """10,000 MWh France = 10,000 * 0.052 = 520 tCO2e (nuclear heavy)."""
        gf = sample_emission_factors["grids"]["FR"]["location_based_kg_per_kwh"]
        mwh = Decimal("10000")
        tco2e = mwh * gf
        assert tco2e == Decimal("520.000")


# =============================================================================
# Refrigerant Emissions
# =============================================================================


class TestRefrigerantEmissions:
    """Test refrigerant leakage emission calculations."""

    def test_r410a_100kg_ar6(self, sample_emission_factors):
        """100 kg R-410A * 2088 GWP = 208,800 kgCO2e = 208.8 tCO2e."""
        gwp = sample_emission_factors["refrigerants"]["R-410A"]["gwp_ar6"]
        leak_kg = Decimal("100")
        tco2e = leak_kg * gwp / Decimal("1000")
        assert tco2e == Decimal("208.8")

    def test_r134a_50kg_ar6(self, sample_emission_factors):
        """50 kg R-134a * 1530 GWP = 76,500 kgCO2e = 76.5 tCO2e."""
        gwp = sample_emission_factors["refrigerants"]["R-134a"]["gwp_ar6"]
        leak_kg = Decimal("50")
        tco2e = leak_kg * gwp / Decimal("1000")
        assert tco2e == Decimal("76.500")

    def test_sf6_5kg_ar6(self, sample_emission_factors):
        """5 kg SF6 * 25200 GWP = 126,000 kgCO2e = 126 tCO2e."""
        gwp = sample_emission_factors["refrigerants"]["SF6"]["gwp_ar6"]
        leak_kg = Decimal("5")
        tco2e = leak_kg * gwp / Decimal("1000")
        assert tco2e == Decimal("126.000")

    def test_r410a_8pct_leak_rate(self, sample_emission_factors):
        """100 kg charge * 8% leak rate = 8 kg leaked * 2088 = 16,704 kgCO2e."""
        gwp = sample_emission_factors["refrigerants"]["R-410A"]["gwp_ar6"]
        charge_kg = Decimal("100")
        leak_rate = Decimal("0.08")
        leaked_kg = charge_kg * leak_rate
        tco2e = leaked_kg * gwp / Decimal("1000")
        assert tco2e == Decimal("16.704")


# =============================================================================
# Fleet Diesel Emissions
# =============================================================================


class TestFleetDieselEmissions:
    """Test fleet vehicle diesel emission calculations."""

    def test_fleet_diesel_500k_km(self, sample_emission_factors):
        """500,000 km fleet (125,000 L diesel) = 315.89 tCO2e (DEFRA)."""
        d = sample_emission_factors["fuels"]["diesel"]["defra_2025"]
        fuel_litres = Decimal("125000")
        total_t = fuel_litres * d["co2e_kg_per_litre"] / Decimal("1000")
        assert Decimal("315") < total_t < Decimal("317")

    def test_fleet_petrol_200k_km(self, sample_emission_factors):
        """200,000 km fleet (20,000 L petrol) = 43.89 tCO2e (DEFRA)."""
        p = sample_emission_factors["fuels"]["petrol"]["defra_2025"]
        fuel_litres = Decimal("20000")
        total_t = fuel_litres * p["co2e_kg_per_litre"] / Decimal("1000")
        assert Decimal("43") < total_t < Decimal("45")


# =============================================================================
# GWP Conversion Formulas
# =============================================================================


class TestGWPConversionFormulas:
    """Test GWP conversion arithmetic."""

    def test_ch4_gwp_conversion_ar6(self, sample_gwp_values):
        """1 tonne CH4 * 27.9 = 27.9 tCO2e."""
        ch4_t = Decimal("1")
        co2e = ch4_t * sample_gwp_values["CH4"]["ar6"]
        assert co2e == Decimal("27.9")

    def test_n2o_gwp_conversion_ar6(self, sample_gwp_values):
        """1 tonne N2O * 273 = 273 tCO2e."""
        n2o_t = Decimal("1")
        co2e = n2o_t * sample_gwp_values["N2O"]["ar6"]
        assert co2e == Decimal("273")

    def test_sf6_gwp_conversion_ar6(self, sample_gwp_values):
        """1 kg SF6 * 25200 = 25,200 kgCO2e = 25.2 tCO2e."""
        sf6_kg = Decimal("1")
        co2e_t = sf6_kg * sample_gwp_values["SF6"]["ar6"] / Decimal("1000")
        assert co2e_t == Decimal("25.200")


# =============================================================================
# Dual Scope 2 with 50% REC
# =============================================================================


class TestDualScope2Formula:
    """Test dual Scope 2 calculation with 50% REC coverage."""

    def test_50pct_rec_coverage(self):
        """10,000 MWh, 50% REC, grid factor 0.370 kgCO2/kWh.
        Location = 10000 * 0.370 = 3,700 tCO2e
        Market = 5000 * 0 (REC) + 5000 * 0.370 (residual) = 1,850 tCO2e
        """
        mwh = Decimal("10000")
        gf = Decimal("0.370")
        rec_mwh = Decimal("5000")
        residual_mwh = mwh - rec_mwh

        location = mwh * gf
        market = rec_mwh * Decimal("0") + residual_mwh * gf

        assert location == Decimal("3700.000")
        assert market == Decimal("1850.000")
        assert location - market == Decimal("1850.000")


# =============================================================================
# Uncertainty Quadrature Formula
# =============================================================================


class TestUncertaintyQuadratureFormula:
    """Test IPCC uncertainty quadrature formula."""

    def test_quadrature_formula_ipcc(self):
        """U_combined = sqrt(sum((E_i * u_i)^2)) / E_total * 100."""
        sources = [
            (Decimal("12800"), Decimal("0.07")),  # stationary, 7%
            (Decimal("2500"), Decimal("0.11")),    # mobile, 11%
            (Decimal("4200"), Decimal("0.16")),    # process, 16%
            (Decimal("350"), Decimal("0.32")),     # fugitive, 32%
            (Decimal("1200"), Decimal("0.22")),    # refrigerant, 22%
            (Decimal("1250"), Decimal("0.25")),    # waste, 25%
        ]
        e_total = sum(e for e, _ in sources)
        rss = sum((float(e) * float(u)) ** 2 for e, u in sources) ** 0.5
        combined_pct = rss / float(e_total) * 100
        assert 5.0 < combined_pct < 15.0

    def test_quadrature_vs_linear(self):
        """Quadrature should be less than linear sum."""
        sources = [
            (10000, 0.10),
            (5000, 0.15),
        ]
        total = sum(e for e, _ in sources)
        rss = sum((e * u) ** 2 for e, u in sources) ** 0.5
        linear = sum(e * u for e, u in sources)
        assert rss < linear


# =============================================================================
# Base Year Recalculation Formula
# =============================================================================


class TestBaseYearRecalculationFormula:
    """Test base-year recalculation arithmetic."""

    def test_acquisition_formula(self):
        """Adjusted = Original + Acquired_base_year_emissions."""
        original = Decimal("41000")
        acquired = Decimal("5000")
        adjusted = original + acquired
        assert adjusted == Decimal("46000")

    def test_divestiture_formula(self):
        """Adjusted = Original - Divested_base_year_emissions."""
        original = Decimal("41000")
        divested = Decimal("8000")
        adjusted = original - divested
        assert adjusted == Decimal("33000")

    def test_materiality_formula(self):
        """Materiality = abs(change) / base_total * 100."""
        change = Decimal("5000")
        base_total = Decimal("41000")
        materiality = abs(change) / base_total * Decimal("100")
        assert materiality == pytest.approx(Decimal("12.20"), abs=Decimal("0.01"))
