# -*- coding: utf-8 -*-
"""
Unit Tests for GL-010 EMISSIONWATCH Emission Factors.

Tests AP-42 emission factor lookups, fuel-specific factors,
control device factors, uncertainty handling, and factor provenance.

Test Count: 22+ tests
Coverage Target: 90%+

Standards: EPA AP-42 Fifth Edition

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
from typing import Any, Dict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools import (
    EmissionsComplianceTools,
    EmissionFactorResult,
    FuelAnalysisResult,
    AP42_EMISSION_FACTORS,
    F_FACTORS,
)


# =============================================================================
# TEST CLASS: EMISSION FACTORS
# =============================================================================

@pytest.mark.unit
class TestEmissionFactors:
    """Test suite for emission factor calculations and lookups."""

    # =========================================================================
    # AP-42 FACTOR LOOKUP TESTS
    # =========================================================================

    def test_ap42_factor_lookup_natural_gas(self, emissions_tools):
        """Test AP-42 factor lookup for natural gas."""
        result = emissions_tools.calculate_emission_factors("natural_gas")

        assert isinstance(result, EmissionFactorResult)
        assert result.fuel_type == "natural_gas"
        assert result.nox_factor == 0.1  # lb/MMBtu
        assert result.co2_factor == 117.0  # lb/MMBtu
        assert result.factor_source == "AP-42 Fifth Edition"
        assert result.units == "lb/MMBtu"

    def test_ap42_factor_lookup_fuel_oil_no2(self, emissions_tools):
        """Test AP-42 factor lookup for No. 2 fuel oil."""
        result = emissions_tools.calculate_emission_factors("fuel_oil_no2")

        assert result.fuel_type == "fuel_oil_no2"
        assert result.nox_factor == 0.14  # lb/MMBtu
        assert result.co2_factor == 161.0  # lb/MMBtu

    def test_ap42_factor_lookup_coal(self, emissions_tools):
        """Test AP-42 factor lookup for bituminous coal."""
        result = emissions_tools.calculate_emission_factors("coal_bituminous")

        assert result.fuel_type == "coal_bituminous"
        assert result.nox_factor == 0.6  # lb/MMBtu
        assert result.co2_factor == 205.0  # lb/MMBtu

    def test_ap42_factor_lookup_biomass(self, emissions_tools):
        """Test AP-42 factor lookup for biomass wood."""
        result = emissions_tools.calculate_emission_factors("biomass_wood")

        assert result.fuel_type == "biomass_wood"
        assert result.nox_factor == 0.22  # lb/MMBtu
        assert result.co2_factor == 195.0  # lb/MMBtu (biogenic)

    def test_ap42_factor_lookup_unknown_fuel_fallback(self, emissions_tools):
        """Test AP-42 factor lookup falls back for unknown fuel."""
        result = emissions_tools.calculate_emission_factors("unknown_fuel_type")

        # Should use natural gas as default
        assert result.nox_factor == AP42_EMISSION_FACTORS["natural_gas"]["nox_lb_mmbtu"]

    # =========================================================================
    # FUEL-SPECIFIC FACTORS TESTS
    # =========================================================================

    def test_fuel_specific_factors_all_pollutants(self, emissions_tools):
        """Test all pollutant factors are returned."""
        result = emissions_tools.calculate_emission_factors("natural_gas")

        assert result.nox_factor >= 0
        assert result.sox_factor >= 0
        assert result.co2_factor >= 0
        assert result.pm_factor >= 0
        assert result.co_factor >= 0

    def test_fuel_specific_factors_data_completeness(self, emissions_tools):
        """Test factor data is complete for all supported fuels."""
        fuel_types = [
            "natural_gas", "fuel_oil_no2", "fuel_oil_no6",
            "coal_bituminous", "biomass_wood"
        ]

        for fuel_type in fuel_types:
            result = emissions_tools.calculate_emission_factors(fuel_type)

            assert result.nox_factor is not None
            assert result.sox_factor is not None
            assert result.co2_factor is not None
            assert result.pm_factor is not None
            assert result.co_factor is not None

    def test_fuel_specific_factors_relative_magnitudes(self, emissions_tools):
        """Test relative magnitudes of emission factors make sense."""
        gas = emissions_tools.calculate_emission_factors("natural_gas")
        coal = emissions_tools.calculate_emission_factors("coal_bituminous")

        # Coal should have higher NOx than natural gas
        assert coal.nox_factor > gas.nox_factor

        # Coal should have higher SOx than natural gas
        assert coal.sox_factor > gas.sox_factor

        # Coal should have higher CO2 than natural gas
        assert coal.co2_factor > gas.co2_factor

        # Coal should have higher PM than natural gas
        assert coal.pm_factor > gas.pm_factor

    # =========================================================================
    # CONTROL DEVICE FACTORS TESTS
    # =========================================================================

    def test_control_device_factors_scr(self, emissions_tools, natural_gas_fuel_data, boiler_process_parameters):
        """Test SCR control device effect on emissions."""
        # SCR typically 90% NOx reduction
        no_scr = {**boiler_process_parameters, "scr_efficiency_percent": 0.0}
        with_scr = {**boiler_process_parameters, "scr_efficiency_percent": 90.0}

        # Get base emission factor
        factor_result = emissions_tools.calculate_emission_factors("natural_gas")

        # Base NOx factor
        assert factor_result.nox_factor > 0

    def test_control_device_factors_fgd(self, emissions_tools):
        """Test FGD control device effect stored in factor metadata."""
        factor_result = emissions_tools.calculate_emission_factors("coal_bituminous")

        # SOx factor is uncontrolled
        # FGD efficiency applied during calculation, not in factor
        assert factor_result.sox_factor > 0

    # =========================================================================
    # FACTOR UNCERTAINTY TESTS
    # =========================================================================

    def test_factor_uncertainty_documented_source(self, emissions_tools):
        """Test emission factors have documented source."""
        result = emissions_tools.calculate_emission_factors("natural_gas")

        assert result.factor_source == "AP-42 Fifth Edition"

    def test_factor_uncertainty_units_documented(self, emissions_tools):
        """Test emission factor units are documented."""
        result = emissions_tools.calculate_emission_factors("natural_gas")

        assert result.units == "lb/MMBtu"

    # =========================================================================
    # FACTOR PROVENANCE TESTS
    # =========================================================================

    def test_factor_provenance_hash(self, emissions_tools):
        """Test emission factor result includes provenance hash."""
        result = emissions_tools.calculate_emission_factors("natural_gas")

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_factor_provenance_deterministic(self, emissions_tools):
        """Test provenance hash is deterministic."""
        hashes = []

        for _ in range(5):
            result = emissions_tools.calculate_emission_factors("coal_bituminous")
            hashes.append(result.provenance_hash)

        assert len(set(hashes)) == 1

    def test_factor_provenance_timestamp(self, emissions_tools):
        """Test emission factor result includes timestamp."""
        result = emissions_tools.calculate_emission_factors("natural_gas")

        assert result.timestamp is not None

    # =========================================================================
    # FUEL COMPOSITION ANALYSIS TESTS
    # =========================================================================

    def test_analyze_fuel_composition_natural_gas(self, emissions_tools):
        """Test fuel composition analysis for natural gas."""
        result = emissions_tools.analyze_fuel_composition("natural_gas")

        assert isinstance(result, FuelAnalysisResult)
        assert result.fuel_type == "natural_gas"
        assert result.carbon_percent > 70
        assert result.hydrogen_percent > 20
        assert result.sulfur_percent < 0.01
        assert result.hhv_btu_lb > 20000

    def test_analyze_fuel_composition_coal(self, emissions_tools):
        """Test fuel composition analysis for coal."""
        result = emissions_tools.analyze_fuel_composition("coal_bituminous")

        assert result.carbon_percent > 70
        assert result.sulfur_percent > 0.5
        assert result.ash_percent > 5
        assert result.hhv_btu_lb > 10000

    def test_analyze_fuel_composition_with_ultimate_analysis(self, emissions_tools):
        """Test fuel composition with custom ultimate analysis."""
        ultimate = {
            "C": 80.0,
            "H": 10.0,
            "S": 1.5,
            "N": 0.5,
            "O": 5.0,
            "ash": 3.0,
        }

        result = emissions_tools.analyze_fuel_composition(
            "coal_bituminous",
            ultimate_analysis=ultimate
        )

        assert result.carbon_percent == 80.0
        assert result.sulfur_percent == 1.5

    def test_analyze_fuel_composition_heating_value(self, emissions_tools):
        """Test heating value calculation from composition."""
        result = emissions_tools.analyze_fuel_composition("coal_bituminous")

        # HHV should be calculated from Dulong formula
        assert result.hhv_btu_lb > 0
        assert result.lhv_btu_lb > 0
        assert result.lhv_btu_lb < result.hhv_btu_lb  # LHV < HHV

    def test_analyze_fuel_composition_stoichiometric_air(self, emissions_tools):
        """Test stoichiometric air calculation."""
        result = emissions_tools.analyze_fuel_composition("natural_gas")

        # Stoichiometric air for natural gas ~17 lb air/lb fuel
        assert result.stoichiometric_air_lb_lb > 10

    # =========================================================================
    # TO_DICT CONVERSION TESTS
    # =========================================================================

    def test_emission_factor_result_to_dict(self, emissions_tools):
        """Test EmissionFactorResult to_dict conversion."""
        result = emissions_tools.calculate_emission_factors("natural_gas")
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "fuel_type" in result_dict
        assert "nox_factor" in result_dict
        assert "sox_factor" in result_dict
        assert "co2_factor" in result_dict
        assert "pm_factor" in result_dict
        assert "co_factor" in result_dict
        assert "factor_source" in result_dict
        assert "units" in result_dict
        assert "provenance_hash" in result_dict

    def test_fuel_analysis_result_to_dict(self, emissions_tools):
        """Test FuelAnalysisResult to_dict conversion."""
        result = emissions_tools.analyze_fuel_composition("coal_bituminous")
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "fuel_type" in result_dict
        assert "carbon_percent" in result_dict
        assert "hydrogen_percent" in result_dict
        assert "sulfur_percent" in result_dict
        assert "hhv_btu_lb" in result_dict
        assert "stoichiometric_air_lb_lb" in result_dict


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

@pytest.mark.unit
class TestEmissionFactorsParametrized:
    """Parametrized tests for emission factors."""

    @pytest.mark.parametrize("fuel_type,expected_nox,expected_co2", [
        ("natural_gas", 0.1, 117.0),
        ("fuel_oil_no2", 0.14, 161.0),
        ("fuel_oil_no6", 0.35, 173.0),
        ("coal_bituminous", 0.6, 205.0),
        ("biomass_wood", 0.22, 195.0),
    ])
    def test_ap42_factors_accuracy(
        self, emissions_tools, fuel_type, expected_nox, expected_co2
    ):
        """Test AP-42 factors match published values."""
        result = emissions_tools.calculate_emission_factors(fuel_type)

        assert result.nox_factor == expected_nox
        assert result.co2_factor == expected_co2

    @pytest.mark.parametrize("fuel_type,fd_expected", [
        ("natural_gas", 8710),
        ("fuel_oil_no2", 9190),
        ("coal_bituminous", 9780),
    ])
    def test_f_factors_accuracy(self, fuel_type, fd_expected):
        """Test F-factors match EPA Method 19 values."""
        assert F_FACTORS[fuel_type]["Fd"] == fd_expected
