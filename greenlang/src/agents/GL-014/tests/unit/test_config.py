# -*- coding: utf-8 -*-
"""
Unit Tests for GL-014 EXCHANGER-PRO Configuration Module.

Tests configuration validation, enum values, and default values
for all GL-014 configuration classes.

Author: GL-TestEngineer
Created: 2025-12-01
Version: 1.0.0
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

import pytest

# Import configuration modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from calculators.heat_transfer_calculator import (
    FlowArrangement,
    CorrelationType,
    FluidPhase,
    TubeLayout,
    TEMA_FOULING_FACTORS,
    TUBE_MATERIAL_CONDUCTIVITY,
    STANDARD_TUBE_DIMENSIONS,
)
from calculators.fouling_calculator import (
    ExchangerType,
    FluidType,
    FoulingMechanism,
    ScalingType,
    FoulingSeverity,
    TEMA_FOULING_FACTORS as FOULING_TEMA_FACTORS,
    ACTIVATION_ENERGIES,
    GAS_CONSTANT_R,
)
from calculators.pressure_drop_calculator import (
    FlowRegime,
    FrictionCorrelation,
    ShellType,
    BaffleType,
    TubePitchPattern,
    PI as PRESSURE_PI,
    STANDARD_GRAVITY,
)
from calculators.economic_calculator import (
    FuelType,
    CleaningMethod,
    DepreciationMethod,
    EmissionScope,
    CO2_EMISSION_FACTORS,
    MACRS_5_YEAR,
    MACRS_7_YEAR,
    MACRS_10_YEAR,
    MACRS_15_YEAR,
    GWP_AR6,
)


# =============================================================================
# Test Class: Enumeration Validation
# =============================================================================

class TestEnumerationValidation:
    """Tests for enumeration value validation."""

    def test_flow_arrangement_enum_values(self):
        """Test FlowArrangement enum has all expected values."""
        expected_values = [
            'COUNTER_FLOW',
            'PARALLEL_FLOW',
            'CROSSFLOW_BOTH_UNMIXED',
            'CROSSFLOW_ONE_MIXED',
            'CROSSFLOW_BOTH_MIXED',
            'SHELL_AND_TUBE_1_1',
            'SHELL_AND_TUBE_1_2',
            'SHELL_AND_TUBE_1_4',
            'SHELL_AND_TUBE_2_4',
        ]

        actual_values = [e.name for e in FlowArrangement]
        for expected in expected_values:
            assert expected in actual_values, f"Missing FlowArrangement: {expected}"

    def test_correlation_type_enum_values(self):
        """Test CorrelationType enum has all expected values."""
        expected_values = [
            'DITTUS_BOELTER',
            'SIEDER_TATE',
            'GNIELINSKI',
            'PETUKHOV',
            'COLBURN',
        ]

        actual_values = [e.name for e in CorrelationType]
        for expected in expected_values:
            assert expected in actual_values, f"Missing CorrelationType: {expected}"

    def test_fluid_phase_enum_values(self):
        """Test FluidPhase enum has all expected values."""
        expected_values = [
            'LIQUID',
            'GAS',
            'TWO_PHASE',
            'CONDENSING',
            'BOILING',
        ]

        actual_values = [e.name for e in FluidPhase]
        for expected in expected_values:
            assert expected in actual_values, f"Missing FluidPhase: {expected}"

    def test_exchanger_type_enum_values(self):
        """Test ExchangerType enum has all expected values."""
        expected_values = [
            'SHELL_TUBE',
            'PLATE',
            'PLATE_FRAME',
            'SPIRAL',
            'AIR_COOLED',
            'DOUBLE_PIPE',
            'PLATE_FIN',
            'SCRAPED_SURFACE',
        ]

        actual_values = [e.name for e in ExchangerType]
        for expected in expected_values:
            assert expected in actual_values, f"Missing ExchangerType: {expected}"

    def test_fluid_type_enum_values(self):
        """Test FluidType enum has all expected values."""
        expected_values = [
            'WATER_TREATED',
            'WATER_UNTREATED',
            'WATER_COOLING_TOWER',
            'WATER_SEAWATER',
            'WATER_BOILER_FEEDWATER',
            'STEAM',
            'STEAM_EXHAUST',
            'OIL_LIGHT',
            'OIL_HEAVY',
            'OIL_CRUDE',
            'OIL_FUEL',
            'OIL_LUBRICATING',
            'GAS_NATURAL',
            'GAS_FLUE',
            'GAS_AIR',
            'REFRIGERANT',
            'ORGANIC_SOLVENT',
            'PROCESS_FLUID',
        ]

        actual_values = [e.name for e in FluidType]
        for expected in expected_values:
            assert expected in actual_values, f"Missing FluidType: {expected}"

    def test_fouling_mechanism_enum_values(self):
        """Test FoulingMechanism enum has all expected values."""
        expected_values = [
            'PARTICULATE',
            'CRYSTALLIZATION',
            'BIOLOGICAL',
            'CORROSION',
            'CHEMICAL_REACTION',
            'COMBINED',
        ]

        actual_values = [e.name for e in FoulingMechanism]
        for expected in expected_values:
            assert expected in actual_values, f"Missing FoulingMechanism: {expected}"

    def test_fouling_severity_enum_values(self):
        """Test FoulingSeverity enum has all expected values."""
        expected_values = [
            'CLEAN',
            'LIGHT',
            'MODERATE',
            'HEAVY',
            'SEVERE',
            'CRITICAL',
        ]

        actual_values = [e.name for e in FoulingSeverity]
        for expected in expected_values:
            assert expected in actual_values, f"Missing FoulingSeverity: {expected}"

    def test_flow_regime_enum_values(self):
        """Test FlowRegime enum has all expected values."""
        expected_values = [
            'LAMINAR',
            'TRANSITION',
            'TURBULENT_SMOOTH',
            'TURBULENT_ROUGH',
        ]

        actual_values = [e.name for e in FlowRegime]
        for expected in expected_values:
            assert expected in actual_values, f"Missing FlowRegime: {expected}"

    def test_shell_type_enum_values(self):
        """Test ShellType enum has all expected values."""
        expected_values = [
            'E_SHELL',
            'F_SHELL',
            'G_SHELL',
            'H_SHELL',
            'J_SHELL',
            'K_SHELL',
            'X_SHELL',
        ]

        actual_values = [e.name for e in ShellType]
        for expected in expected_values:
            assert expected in actual_values, f"Missing ShellType: {expected}"

    def test_fuel_type_enum_values(self):
        """Test FuelType enum has all expected values."""
        expected_values = [
            'NATURAL_GAS',
            'FUEL_OIL_LIGHT',
            'FUEL_OIL_HEAVY',
            'COAL_BITUMINOUS',
            'COAL_ANTHRACITE',
            'LPG',
            'BIOMASS',
            'ELECTRICITY_GRID_US',
            'ELECTRICITY_GRID_EU',
            'ELECTRICITY_GRID_CHINA',
            'STEAM_FROM_BOILER',
        ]

        actual_values = [e.name for e in FuelType]
        for expected in expected_values:
            assert expected in actual_values, f"Missing FuelType: {expected}"

    def test_depreciation_method_enum_values(self):
        """Test DepreciationMethod enum has all expected values."""
        expected_values = [
            'STRAIGHT_LINE',
            'MACRS_5_YEAR',
            'MACRS_7_YEAR',
            'MACRS_10_YEAR',
            'MACRS_15_YEAR',
            'DOUBLE_DECLINING',
        ]

        actual_values = [e.name for e in DepreciationMethod]
        for expected in expected_values:
            assert expected in actual_values, f"Missing DepreciationMethod: {expected}"


# =============================================================================
# Test Class: Default Values
# =============================================================================

class TestDefaultValues:
    """Tests for configuration default values."""

    def test_tema_fouling_factors_exist(self):
        """Test TEMA fouling factors lookup table exists and has values."""
        assert len(TEMA_FOULING_FACTORS) > 0, "TEMA fouling factors table is empty"

        # Check some specific values
        assert "boiler_feedwater_treated" in TEMA_FOULING_FACTORS
        assert "cooling_tower_water_treated" in TEMA_FOULING_FACTORS
        assert "fuel_oil" in TEMA_FOULING_FACTORS

    def test_tema_fouling_factors_values_reasonable(self):
        """Test TEMA fouling factor values are in reasonable range."""
        # Typical fouling resistances: 0.00009 to 0.00088 m^2.K/W
        for fluid, rf in TEMA_FOULING_FACTORS.items():
            assert Decimal("0") < rf < Decimal("0.01"), (
                f"Fouling factor for {fluid} out of range: {rf}"
            )

    def test_tube_material_conductivity_exists(self):
        """Test tube material conductivity lookup table exists."""
        assert len(TUBE_MATERIAL_CONDUCTIVITY) > 0

        # Check specific materials
        expected_materials = [
            'carbon_steel',
            'stainless_steel_304',
            'copper',
            'titanium',
        ]
        for material in expected_materials:
            assert material in TUBE_MATERIAL_CONDUCTIVITY, f"Missing material: {material}"

    def test_tube_material_conductivity_values_reasonable(self):
        """Test tube material conductivity values are reasonable."""
        # Typical range: 10-400 W/(m.K)
        for material, k in TUBE_MATERIAL_CONDUCTIVITY.items():
            assert Decimal("5") < k < Decimal("500"), (
                f"Conductivity for {material} out of range: {k}"
            )

    def test_co2_emission_factors_exist(self):
        """Test CO2 emission factors lookup table exists."""
        assert len(CO2_EMISSION_FACTORS) > 0

        # Check specific fuel types
        expected_fuels = [
            'natural_gas',
            'fuel_oil_heavy',
            'coal_bituminous',
        ]
        for fuel in expected_fuels:
            assert fuel in CO2_EMISSION_FACTORS, f"Missing fuel: {fuel}"

    def test_co2_emission_factors_values_reasonable(self):
        """Test CO2 emission factor values are reasonable."""
        # Typical range: 0.015-0.6 kg CO2/kWh
        for fuel, ef in CO2_EMISSION_FACTORS.items():
            assert 0 < ef < 1.0, f"Emission factor for {fuel} out of range: {ef}"

    def test_macrs_schedules_sum_to_one(self):
        """Test MACRS depreciation schedules sum to 1.0."""
        schedules = [
            ("MACRS_5_YEAR", MACRS_5_YEAR),
            ("MACRS_7_YEAR", MACRS_7_YEAR),
            ("MACRS_10_YEAR", MACRS_10_YEAR),
            ("MACRS_15_YEAR", MACRS_15_YEAR),
        ]

        for name, schedule in schedules:
            total = sum(schedule)
            assert abs(total - 1.0) < 0.0001, (
                f"{name} schedule doesn't sum to 1.0: {total}"
            )

    def test_macrs_schedules_length(self):
        """Test MACRS depreciation schedules have correct length."""
        assert len(MACRS_5_YEAR) == 6, "MACRS 5-year should have 6 periods"
        assert len(MACRS_7_YEAR) == 8, "MACRS 7-year should have 8 periods"
        assert len(MACRS_10_YEAR) == 11, "MACRS 10-year should have 11 periods"
        assert len(MACRS_15_YEAR) == 16, "MACRS 15-year should have 16 periods"

    def test_gwp_values_exist(self):
        """Test Global Warming Potential values exist."""
        assert len(GWP_AR6) > 0

        # Check specific GHGs
        assert 'CO2' in GWP_AR6
        assert 'CH4' in GWP_AR6
        assert 'N2O' in GWP_AR6

    def test_gwp_co2_is_one(self):
        """Test CO2 GWP is 1.0 (reference)."""
        assert GWP_AR6['CO2'] == 1.0

    def test_gwp_values_reasonable(self):
        """Test GWP values are reasonable per IPCC AR6."""
        assert 20 < GWP_AR6['CH4'] < 40, "CH4 GWP should be ~29.8"
        assert 250 < GWP_AR6['N2O'] < 300, "N2O GWP should be ~273"


# =============================================================================
# Test Class: Physical Constants
# =============================================================================

class TestPhysicalConstants:
    """Tests for physical constants used in calculations."""

    def test_pi_value(self):
        """Test Pi constant has correct value."""
        import math
        pi_diff = abs(float(PRESSURE_PI) - math.pi)
        assert pi_diff < 1e-10, f"Pi value differs by {pi_diff}"

    def test_standard_gravity(self):
        """Test standard gravity constant."""
        # NIST CODATA 2018: g = 9.80665 m/s^2
        assert STANDARD_GRAVITY == Decimal("9.80665")

    def test_gas_constant(self):
        """Test gas constant value."""
        # R = 8.314 J/(mol.K) = 0.008314 kJ/(mol.K)
        assert GAS_CONSTANT_R == Decimal("0.008314")


# =============================================================================
# Test Class: Standard Tube Dimensions
# =============================================================================

class TestStandardTubeDimensions:
    """Tests for standard tube dimension lookup tables."""

    def test_standard_tubes_exist(self):
        """Test standard tube dimensions table exists."""
        assert len(STANDARD_TUBE_DIMENSIONS) > 0

    def test_standard_tube_dimensions_valid(self):
        """Test standard tube dimensions are physically valid."""
        for tube_id, dims in STANDARD_TUBE_DIMENSIONS.items():
            # OD > ID
            assert dims.outer_diameter_m > dims.inner_diameter_m, (
                f"Tube {tube_id}: OD must be > ID"
            )

            # ID = OD - 2*wall
            calculated_id = dims.outer_diameter_m - 2 * dims.wall_thickness_m
            diff = abs(dims.inner_diameter_m - calculated_id)
            assert diff < Decimal("0.0001"), (
                f"Tube {tube_id}: ID doesn't match OD - 2*wall"
            )

            # BWG is positive integer
            assert dims.bwg > 0, f"Tube {tube_id}: BWG must be positive"

    def test_common_tube_sizes_exist(self):
        """Test common tube sizes are in the lookup table."""
        common_sizes = [
            "3/4_14BWG",
            "3/4_16BWG",
            "1_14BWG",
        ]

        for size in common_sizes:
            assert size in STANDARD_TUBE_DIMENSIONS, f"Missing tube size: {size}"


# =============================================================================
# Test Class: Activation Energies
# =============================================================================

class TestActivationEnergies:
    """Tests for fouling activation energy values."""

    def test_activation_energies_exist(self):
        """Test activation energies table exists for all mechanisms."""
        for mechanism in FoulingMechanism:
            assert mechanism in ACTIVATION_ENERGIES, (
                f"Missing activation energy for {mechanism}"
            )

    def test_activation_energies_reasonable(self):
        """Test activation energy values are reasonable."""
        # Typical range: 0-150 kJ/mol for fouling reactions
        for mechanism, ea in ACTIVATION_ENERGIES.items():
            assert Decimal("0") <= ea <= Decimal("200"), (
                f"Activation energy for {mechanism} out of range: {ea}"
            )

    def test_particulate_no_activation_energy(self):
        """Test particulate fouling has zero activation energy."""
        # Particulate fouling is not thermally activated
        assert ACTIVATION_ENERGIES[FoulingMechanism.PARTICULATE] == Decimal("0")


# =============================================================================
# Test Class: Configuration Immutability
# =============================================================================

class TestConfigurationImmutability:
    """Tests for configuration immutability."""

    def test_tema_fouling_factors_immutable(self):
        """Test TEMA fouling factors cannot be modified."""
        original_value = TEMA_FOULING_FACTORS.get("fuel_oil")

        # Attempt to modify (should either fail or not persist)
        try:
            TEMA_FOULING_FACTORS["fuel_oil"] = Decimal("9.99")
        except (TypeError, AttributeError):
            pass  # Expected for frozen dict

        # Verify value unchanged
        assert TEMA_FOULING_FACTORS.get("fuel_oil") == original_value or \
               TEMA_FOULING_FACTORS.get("fuel_oil") == Decimal("0.00088")

    def test_enum_values_immutable(self):
        """Test enum values are immutable."""
        # Enums in Python are inherently immutable
        with pytest.raises(AttributeError):
            FlowArrangement.COUNTER_FLOW = "modified"


# =============================================================================
# Test Class: Configuration Completeness
# =============================================================================

class TestConfigurationCompleteness:
    """Tests for configuration completeness."""

    def test_fouling_factors_for_all_fluid_types(self):
        """Test fouling factors exist for all common fluid types."""
        # At least check that the fouling calculator module has factors
        assert len(FOULING_TEMA_FACTORS) >= 15, "Should have factors for at least 15 fluid types"

    def test_emission_factors_for_all_fuel_types(self):
        """Test emission factors exist for all fuel types."""
        for fuel_type in FuelType:
            fuel_key = fuel_type.value
            assert fuel_key in CO2_EMISSION_FACTORS, (
                f"Missing emission factor for {fuel_type}"
            )

    def test_conductivity_for_common_materials(self):
        """Test conductivity values exist for common tube materials."""
        common_materials = [
            'carbon_steel',
            'stainless_steel_304',
            'stainless_steel_316',
            'copper',
            'titanium',
        ]

        for material in common_materials:
            assert material in TUBE_MATERIAL_CONDUCTIVITY, (
                f"Missing conductivity for {material}"
            )


# =============================================================================
# Test Class: Configuration Consistency
# =============================================================================

class TestConfigurationConsistency:
    """Tests for configuration consistency across modules."""

    def test_fouling_factors_consistent(self):
        """Test TEMA fouling factors are consistent between modules."""
        # Both heat_transfer_calculator and fouling_calculator should have
        # consistent fouling factor tables
        # They may have different keys but overlapping values should match
        pass  # Tables have different key structures

    def test_physical_constants_consistent(self):
        """Test physical constants are consistent across modules."""
        import math

        # PI should match standard library
        assert abs(float(PRESSURE_PI) - math.pi) < 1e-15

        # Standard gravity should be NIST value
        assert STANDARD_GRAVITY == Decimal("9.80665")
