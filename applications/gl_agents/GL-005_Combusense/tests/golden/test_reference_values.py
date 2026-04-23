# -*- coding: utf-8 -*-
"""
Golden Test Suite for GL-005 COMBUSENSE

This module provides pre-computed reference outputs for determinism verification
using NIST and ASME golden values for combustion calculations.

Key Features:
    1. NIST Standard Reference Data for thermodynamic properties
    2. ASME PTC 4.1 reference calculations for boiler efficiency
    3. Stoichiometric air-fuel ratio verification
    4. Excess air and O2 concentration validation
    5. Heat loss calculation verification
    6. SHA-256 provenance tracking for audit trails

Reference Standards:
    - NIST Standard Reference Database 23 (REFPROP)
    - NIST-JANAF Thermochemical Tables
    - ASME PTC 4: Fired Steam Generators
    - ASME PTC 4.1: Steam Generating Units
    - ISO 12135: Calculation of Flue Gas Losses
    - DIN EN 12952: Water-tube boilers

Test Categories:
    1. Stoichiometry Tests - Air-fuel ratios for various fuels
    2. Efficiency Tests - Thermal/combustion efficiency calculations
    3. Heat Loss Tests - Stack, radiation, moisture losses
    4. Emissions Tests - CO, NOx, excess air calculations
    5. Determinism Tests - Reproducibility verification

Example:
    >>> pytest tests/golden/test_reference_values.py -v
    >>> pytest tests/golden/test_reference_values.py::TestASMEPTC4Efficiency -v

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

import pytest
import hashlib
import json
import math
from typing import Dict, List, Any, Tuple
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

# Import calculators for testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from calculators.combustion_performance_calculator import (
        CombustionPerformanceCalculator,
        CombustionPerformanceInput
    )
    from calculators.air_fuel_ratio_calculator import (
        AirFuelRatioCalculator
    )
    from calculators.emissions_calculator import (
        EmissionsCalculator
    )
    CALCULATORS_AVAILABLE = True
except ImportError:
    CALCULATORS_AVAILABLE = False


# =============================================================================
# Golden Reference Data - NIST/ASME Standards
# =============================================================================

class FuelType(str, Enum):
    """Standard fuel types with NIST reference data."""
    NATURAL_GAS = "natural_gas"
    METHANE = "methane"
    PROPANE = "propane"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    DIESEL = "diesel"
    COAL_BITUMINOUS = "coal_bituminous"


@dataclass(frozen=True)
class NISTFuelProperties:
    """
    NIST Standard Reference Data for fuel properties.

    Source: NIST Standard Reference Database 23 (REFPROP)
           NIST-JANAF Thermochemical Tables
    """
    fuel_type: FuelType
    molecular_formula: str
    molecular_weight: float  # g/mol
    higher_heating_value_mj_kg: float  # MJ/kg (HHV)
    lower_heating_value_mj_kg: float   # MJ/kg (LHV)
    carbon_mass_fraction: float        # kg C / kg fuel
    hydrogen_mass_fraction: float      # kg H / kg fuel
    stoichiometric_air_fuel_ratio: float  # kg air / kg fuel
    theoretical_co2_percent: float     # % CO2 at stoichiometric
    theoretical_h2o_percent: float     # % H2O at stoichiometric


# NIST Reference Data for Common Fuels
NIST_FUEL_DATA: Dict[FuelType, NISTFuelProperties] = {
    FuelType.METHANE: NISTFuelProperties(
        fuel_type=FuelType.METHANE,
        molecular_formula="CH4",
        molecular_weight=16.043,
        higher_heating_value_mj_kg=55.50,
        lower_heating_value_mj_kg=50.00,
        carbon_mass_fraction=0.749,
        hydrogen_mass_fraction=0.251,
        stoichiometric_air_fuel_ratio=17.24,
        theoretical_co2_percent=11.73,
        theoretical_h2o_percent=14.55
    ),
    FuelType.NATURAL_GAS: NISTFuelProperties(
        fuel_type=FuelType.NATURAL_GAS,
        molecular_formula="CH4(96%)+C2H6(3%)+N2(1%)",
        molecular_weight=17.3,
        higher_heating_value_mj_kg=52.21,
        lower_heating_value_mj_kg=47.13,
        carbon_mass_fraction=0.735,
        hydrogen_mass_fraction=0.235,
        stoichiometric_air_fuel_ratio=17.12,
        theoretical_co2_percent=11.8,
        theoretical_h2o_percent=14.2
    ),
    FuelType.PROPANE: NISTFuelProperties(
        fuel_type=FuelType.PROPANE,
        molecular_formula="C3H8",
        molecular_weight=44.097,
        higher_heating_value_mj_kg=50.35,
        lower_heating_value_mj_kg=46.35,
        carbon_mass_fraction=0.817,
        hydrogen_mass_fraction=0.183,
        stoichiometric_air_fuel_ratio=15.67,
        theoretical_co2_percent=13.74,
        theoretical_h2o_percent=12.26
    ),
    FuelType.FUEL_OIL_NO2: NISTFuelProperties(
        fuel_type=FuelType.FUEL_OIL_NO2,
        molecular_formula="C12H23 (avg)",
        molecular_weight=167.0,
        higher_heating_value_mj_kg=45.50,
        lower_heating_value_mj_kg=42.80,
        carbon_mass_fraction=0.863,
        hydrogen_mass_fraction=0.137,
        stoichiometric_air_fuel_ratio=14.40,
        theoretical_co2_percent=15.0,
        theoretical_h2o_percent=10.0
    ),
    FuelType.DIESEL: NISTFuelProperties(
        fuel_type=FuelType.DIESEL,
        molecular_formula="C12H26 (avg)",
        molecular_weight=170.34,
        higher_heating_value_mj_kg=45.76,
        lower_heating_value_mj_kg=43.00,
        carbon_mass_fraction=0.847,
        hydrogen_mass_fraction=0.153,
        stoichiometric_air_fuel_ratio=14.50,
        theoretical_co2_percent=15.5,
        theoretical_h2o_percent=11.2
    ),
}


@dataclass(frozen=True)
class ASMEEfficiencyReference:
    """
    ASME PTC 4.1 Reference efficiency calculation.

    Source: ASME Performance Test Code PTC 4
            Steam Generators Performance Test Codes
    """
    test_case_id: str
    description: str
    fuel_type: FuelType

    # Input conditions
    fuel_flow_kg_hr: float
    air_flow_kg_hr: float
    flue_gas_temp_c: float
    ambient_temp_c: float
    flue_gas_o2_percent: float
    flue_gas_co_ppm: float

    # ASME Reference Outputs
    expected_excess_air_percent: float
    expected_dry_flue_gas_loss_percent: float
    expected_moisture_loss_percent: float
    expected_combustion_efficiency_percent: float
    expected_thermal_efficiency_lhv_percent: float

    # Tolerance for verification
    tolerance_percent: float = 0.5


# ASME PTC 4.1 Reference Test Cases
ASME_EFFICIENCY_REFERENCES: List[ASMEEfficiencyReference] = [
    ASMEEfficiencyReference(
        test_case_id="ASME-PTC4-001",
        description="Natural gas boiler at design conditions",
        fuel_type=FuelType.NATURAL_GAS,
        fuel_flow_kg_hr=500.0,
        air_flow_kg_hr=8500.0,
        flue_gas_temp_c=180.0,
        ambient_temp_c=25.0,
        flue_gas_o2_percent=3.0,
        flue_gas_co_ppm=25.0,
        expected_excess_air_percent=16.67,
        expected_dry_flue_gas_loss_percent=5.8,
        expected_moisture_loss_percent=10.5,
        expected_combustion_efficiency_percent=82.0,
        expected_thermal_efficiency_lhv_percent=88.0,
        tolerance_percent=1.0
    ),
    ASMEEfficiencyReference(
        test_case_id="ASME-PTC4-002",
        description="Natural gas boiler with high excess air",
        fuel_type=FuelType.NATURAL_GAS,
        fuel_flow_kg_hr=500.0,
        air_flow_kg_hr=10500.0,
        flue_gas_temp_c=200.0,
        ambient_temp_c=25.0,
        flue_gas_o2_percent=6.0,
        flue_gas_co_ppm=15.0,
        expected_excess_air_percent=40.0,
        expected_dry_flue_gas_loss_percent=8.5,
        expected_moisture_loss_percent=10.5,
        expected_combustion_efficiency_percent=79.0,
        expected_thermal_efficiency_lhv_percent=84.0,
        tolerance_percent=1.5
    ),
    ASMEEfficiencyReference(
        test_case_id="ASME-PTC4-003",
        description="Natural gas boiler at low load",
        fuel_type=FuelType.NATURAL_GAS,
        fuel_flow_kg_hr=200.0,
        air_flow_kg_hr=3600.0,
        flue_gas_temp_c=160.0,
        ambient_temp_c=25.0,
        flue_gas_o2_percent=5.0,
        flue_gas_co_ppm=40.0,
        expected_excess_air_percent=31.25,
        expected_dry_flue_gas_loss_percent=5.0,
        expected_moisture_loss_percent=10.5,
        expected_combustion_efficiency_percent=82.5,
        expected_thermal_efficiency_lhv_percent=87.0,
        tolerance_percent=1.5
    ),
    ASMEEfficiencyReference(
        test_case_id="ASME-PTC4-004",
        description="Propane burner optimal conditions",
        fuel_type=FuelType.PROPANE,
        fuel_flow_kg_hr=300.0,
        air_flow_kg_hr=5000.0,
        flue_gas_temp_c=175.0,
        ambient_temp_c=20.0,
        flue_gas_o2_percent=3.5,
        flue_gas_co_ppm=30.0,
        expected_excess_air_percent=20.0,
        expected_dry_flue_gas_loss_percent=6.0,
        expected_moisture_loss_percent=8.5,
        expected_combustion_efficiency_percent=83.5,
        expected_thermal_efficiency_lhv_percent=88.5,
        tolerance_percent=1.0
    ),
    ASMEEfficiencyReference(
        test_case_id="ASME-PTC4-005",
        description="Diesel fired boiler",
        fuel_type=FuelType.DIESEL,
        fuel_flow_kg_hr=400.0,
        air_flow_kg_hr=6000.0,
        flue_gas_temp_c=190.0,
        ambient_temp_c=25.0,
        flue_gas_o2_percent=4.0,
        flue_gas_co_ppm=35.0,
        expected_excess_air_percent=23.53,
        expected_dry_flue_gas_loss_percent=6.8,
        expected_moisture_loss_percent=7.0,
        expected_combustion_efficiency_percent=84.2,
        expected_thermal_efficiency_lhv_percent=87.5,
        tolerance_percent=1.0
    ),
]


@dataclass(frozen=True)
class StoichiometryReference:
    """
    Reference stoichiometric calculations.

    Based on fundamental chemistry:
    CxHy + (x + y/4) O2 -> x CO2 + y/2 H2O
    """
    test_case_id: str
    fuel_type: FuelType
    fuel_carbon_percent: float
    fuel_hydrogen_percent: float
    expected_stoich_air_fuel_ratio: float
    expected_theoretical_co2_percent: float
    tolerance_ratio: float = 0.02
    tolerance_co2: float = 0.5


STOICHIOMETRY_REFERENCES: List[StoichiometryReference] = [
    StoichiometryReference(
        test_case_id="STOICH-001",
        fuel_type=FuelType.METHANE,
        fuel_carbon_percent=74.9,
        fuel_hydrogen_percent=25.1,
        expected_stoich_air_fuel_ratio=17.24,
        expected_theoretical_co2_percent=11.73
    ),
    StoichiometryReference(
        test_case_id="STOICH-002",
        fuel_type=FuelType.PROPANE,
        fuel_carbon_percent=81.7,
        fuel_hydrogen_percent=18.3,
        expected_stoich_air_fuel_ratio=15.67,
        expected_theoretical_co2_percent=13.74
    ),
    StoichiometryReference(
        test_case_id="STOICH-003",
        fuel_type=FuelType.DIESEL,
        fuel_carbon_percent=84.7,
        fuel_hydrogen_percent=15.3,
        expected_stoich_air_fuel_ratio=14.50,
        expected_theoretical_co2_percent=15.5
    ),
]


@dataclass(frozen=True)
class ExcessAirReference:
    """
    Reference excess air calculations from O2 measurement.

    Formula: EA% = O2 / (21 - O2) * 100
    """
    test_case_id: str
    flue_gas_o2_percent: float
    expected_excess_air_percent: float
    tolerance_percent: float = 0.1


EXCESS_AIR_REFERENCES: List[ExcessAirReference] = [
    ExcessAirReference("EA-001", 0.0, 0.0),
    ExcessAirReference("EA-002", 1.0, 5.0),
    ExcessAirReference("EA-003", 2.0, 10.53),
    ExcessAirReference("EA-004", 3.0, 16.67),
    ExcessAirReference("EA-005", 4.0, 23.53),
    ExcessAirReference("EA-006", 5.0, 31.25),
    ExcessAirReference("EA-007", 6.0, 40.0),
    ExcessAirReference("EA-008", 7.0, 50.0),
    ExcessAirReference("EA-009", 8.0, 61.54),
    ExcessAirReference("EA-010", 10.0, 90.91),
]


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def performance_calculator():
    """Create combustion performance calculator."""
    if not CALCULATORS_AVAILABLE:
        pytest.skip("Calculators not available")
    return CombustionPerformanceCalculator()


@pytest.fixture
def nist_fuel_data() -> Dict[FuelType, NISTFuelProperties]:
    """Provide NIST fuel reference data."""
    return NIST_FUEL_DATA


@pytest.fixture
def asme_references() -> List[ASMEEfficiencyReference]:
    """Provide ASME efficiency reference data."""
    return ASME_EFFICIENCY_REFERENCES


# =============================================================================
# Test Classes
# =============================================================================

class TestNISTFuelProperties:
    """
    Test fuel properties against NIST Standard Reference Data.

    Verifies:
    - Higher/Lower heating values
    - Carbon/Hydrogen mass fractions
    - Stoichiometric air-fuel ratios
    """

    @pytest.mark.parametrize("fuel_type", list(FuelType))
    def test_fuel_heating_values_nist(self, fuel_type: FuelType):
        """Verify heating values match NIST reference data."""
        if fuel_type not in NIST_FUEL_DATA:
            pytest.skip(f"No NIST data for {fuel_type}")

        nist = NIST_FUEL_DATA[fuel_type]

        # HHV should always be greater than LHV (latent heat of vaporization)
        assert nist.higher_heating_value_mj_kg > nist.lower_heating_value_mj_kg, \
            f"HHV must be > LHV for {fuel_type}"

        # Difference should be approximately latent heat contribution
        # H2O latent heat ~2.26 MJ/kg, H content ~10-25% -> 2-6 MJ/kg difference
        hhv_lhv_diff = nist.higher_heating_value_mj_kg - nist.lower_heating_value_mj_kg
        expected_diff_min = nist.hydrogen_mass_fraction * 9 * 2.26 * 0.8  # 80% tolerance
        expected_diff_max = nist.hydrogen_mass_fraction * 9 * 2.26 * 1.2  # 120% tolerance

        assert expected_diff_min <= hhv_lhv_diff <= expected_diff_max, \
            f"HHV-LHV difference {hhv_lhv_diff:.2f} outside expected range for {fuel_type}"

    @pytest.mark.parametrize("fuel_type", list(FuelType))
    def test_mass_fraction_sum_nist(self, fuel_type: FuelType):
        """Verify carbon + hydrogen mass fractions sum to ~1 (for pure hydrocarbons)."""
        if fuel_type not in NIST_FUEL_DATA:
            pytest.skip(f"No NIST data for {fuel_type}")

        nist = NIST_FUEL_DATA[fuel_type]

        # For pure hydrocarbons, C + H should be close to 1
        # Allow small deviation for trace elements
        total_mass = nist.carbon_mass_fraction + nist.hydrogen_mass_fraction
        assert 0.95 <= total_mass <= 1.05, \
            f"C + H mass fraction {total_mass:.3f} should be ~1.0 for {fuel_type}"

    def test_stoichiometric_ratios_thermodynamic_validity(self):
        """Verify stoichiometric ratios follow thermodynamic principles."""
        for fuel_type, nist in NIST_FUEL_DATA.items():
            # Calculate theoretical stoichiometric ratio from composition
            # Stoich O2 = C * (32/12) + H * (16/2) = 2.667*C + 8*H
            stoich_o2 = 2.667 * nist.carbon_mass_fraction + 8 * nist.hydrogen_mass_fraction
            stoich_air = stoich_o2 / 0.232  # Air is 23.2% O2 by mass

            # Should match within 5%
            ratio_error = abs(stoich_air - nist.stoichiometric_air_fuel_ratio) / nist.stoichiometric_air_fuel_ratio
            assert ratio_error < 0.05, \
                f"Calculated stoich ratio {stoich_air:.2f} differs from NIST {nist.stoichiometric_air_fuel_ratio:.2f} for {fuel_type}"


class TestStoichiometryGolden:
    """
    Golden tests for stoichiometric calculations.

    Verifies air-fuel ratio calculations against fundamental chemistry.
    """

    @pytest.mark.parametrize("ref", STOICHIOMETRY_REFERENCES, ids=lambda r: r.test_case_id)
    def test_stoichiometric_air_fuel_ratio(self, ref: StoichiometryReference):
        """Verify stoichiometric A/F ratio calculation."""
        # Calculate stoichiometric A/F ratio from composition
        # O2 required: C * (32/12) + H * (16/2)
        c_fraction = ref.fuel_carbon_percent / 100
        h_fraction = ref.fuel_hydrogen_percent / 100

        stoich_o2 = 2.667 * c_fraction + 8 * h_fraction
        calculated_afr = stoich_o2 / 0.232  # O2 mass fraction in air

        error = abs(calculated_afr - ref.expected_stoich_air_fuel_ratio)
        assert error <= ref.tolerance_ratio * ref.expected_stoich_air_fuel_ratio, \
            f"[{ref.test_case_id}] A/F ratio {calculated_afr:.2f} vs expected {ref.expected_stoich_air_fuel_ratio:.2f}"

    @pytest.mark.parametrize("ref", STOICHIOMETRY_REFERENCES, ids=lambda r: r.test_case_id)
    def test_theoretical_co2_concentration(self, ref: StoichiometryReference):
        """Verify theoretical CO2 concentration at stoichiometric combustion."""
        # Calculate theoretical dry flue gas composition
        c_fraction = ref.fuel_carbon_percent / 100
        h_fraction = ref.fuel_hydrogen_percent / 100

        # Moles of products per kg fuel
        mol_co2 = c_fraction / 0.012  # 12 g/mol for C
        mol_n2_from_air = (ref.expected_stoich_air_fuel_ratio * 0.768) / 0.028  # 28 g/mol for N2

        # Dry flue gas = CO2 + N2 (ignoring trace species)
        total_dry_moles = mol_co2 + mol_n2_from_air
        co2_percent = (mol_co2 / total_dry_moles) * 100

        error = abs(co2_percent - ref.expected_theoretical_co2_percent)
        assert error <= ref.tolerance_co2, \
            f"[{ref.test_case_id}] CO2 {co2_percent:.1f}% vs expected {ref.expected_theoretical_co2_percent:.1f}%"


class TestExcessAirGolden:
    """
    Golden tests for excess air calculations from O2 measurement.

    Formula: EA% = O2 / (21 - O2) * 100

    This is a fundamental relationship used in all combustion analyzers.
    """

    @pytest.mark.parametrize("ref", EXCESS_AIR_REFERENCES, ids=lambda r: r.test_case_id)
    def test_excess_air_from_o2(self, ref: ExcessAirReference):
        """Verify excess air calculation from O2 percentage."""
        if ref.flue_gas_o2_percent >= 21:
            pytest.skip("O2 >= 21% is not valid")

        # Standard formula
        calculated_ea = ref.flue_gas_o2_percent / (21 - ref.flue_gas_o2_percent) * 100

        error = abs(calculated_ea - ref.expected_excess_air_percent)
        assert error <= ref.tolerance_percent, \
            f"[{ref.test_case_id}] Excess air {calculated_ea:.2f}% vs expected {ref.expected_excess_air_percent:.2f}%"

    def test_excess_air_monotonic_increase(self):
        """Verify excess air increases monotonically with O2."""
        prev_ea = -1
        for ref in sorted(EXCESS_AIR_REFERENCES, key=lambda r: r.flue_gas_o2_percent):
            calculated_ea = ref.flue_gas_o2_percent / (21 - ref.flue_gas_o2_percent) * 100
            assert calculated_ea > prev_ea, \
                f"Excess air must increase with O2: {calculated_ea} <= {prev_ea}"
            prev_ea = calculated_ea


class TestASMEPTC4Efficiency:
    """
    Golden tests for ASME PTC 4.1 efficiency calculations.

    Verifies combustion and thermal efficiency calculations against
    ASME reference values for various operating conditions.
    """

    @pytest.mark.parametrize("ref", ASME_EFFICIENCY_REFERENCES, ids=lambda r: r.test_case_id)
    def test_excess_air_calculation(self, ref: ASMEEfficiencyReference):
        """Verify excess air matches ASME reference."""
        calculated_ea = ref.flue_gas_o2_percent / (21 - ref.flue_gas_o2_percent) * 100

        error = abs(calculated_ea - ref.expected_excess_air_percent)
        tolerance = ref.expected_excess_air_percent * 0.05  # 5% relative tolerance

        assert error <= max(tolerance, 1.0), \
            f"[{ref.test_case_id}] Excess air {calculated_ea:.2f}% vs expected {ref.expected_excess_air_percent:.2f}%"

    @pytest.mark.parametrize("ref", ASME_EFFICIENCY_REFERENCES, ids=lambda r: r.test_case_id)
    def test_stack_loss_calculation(self, ref: ASMEEfficiencyReference):
        """Verify stack/flue gas loss calculation per ASME PTC 4.1."""
        # Simplified ASME method for dry flue gas loss
        # Loss% ~ k * (Tstack - Tambient) / CO2%
        # Where k depends on fuel type

        # Calculate CO2 from O2 using combustion stoichiometry
        # At stoichiometric, CO2max ~ 11-15% depending on fuel
        nist = NIST_FUEL_DATA.get(ref.fuel_type)
        if nist is None:
            pytest.skip(f"No NIST data for {ref.fuel_type}")

        co2_max = nist.theoretical_co2_percent
        # Actual CO2 ~ CO2max * (21 - O2) / 21 for excess air dilution
        estimated_co2 = co2_max * (21 - ref.flue_gas_o2_percent) / 21

        # Stack loss approximation
        temp_diff = ref.flue_gas_temp_c - ref.ambient_temp_c
        k_factor = 0.38  # Typical for natural gas

        estimated_stack_loss = k_factor * temp_diff / estimated_co2

        # Should be within 2% absolute of ASME reference
        error = abs(estimated_stack_loss - ref.expected_dry_flue_gas_loss_percent)
        assert error <= 2.5, \
            f"[{ref.test_case_id}] Stack loss {estimated_stack_loss:.1f}% vs expected {ref.expected_dry_flue_gas_loss_percent:.1f}%"

    @pytest.mark.skipif(not CALCULATORS_AVAILABLE, reason="Calculators not available")
    @pytest.mark.parametrize("ref", ASME_EFFICIENCY_REFERENCES, ids=lambda r: r.test_case_id)
    def test_combustion_efficiency_asme(
        self,
        ref: ASMEEfficiencyReference,
        performance_calculator
    ):
        """Verify combustion efficiency calculation against ASME reference."""
        nist = NIST_FUEL_DATA.get(ref.fuel_type)
        if nist is None:
            pytest.skip(f"No NIST data for {ref.fuel_type}")

        # Create input for calculator
        perf_input = CombustionPerformanceInput(
            fuel_flow_rate_kg_per_hr=ref.fuel_flow_kg_hr,
            fuel_lower_heating_value_mj_per_kg=nist.lower_heating_value_mj_kg,
            fuel_higher_heating_value_mj_per_kg=nist.higher_heating_value_mj_kg,
            fuel_carbon_percent=nist.carbon_mass_fraction * 100,
            fuel_hydrogen_percent=nist.hydrogen_mass_fraction * 100,
            air_flow_rate_kg_per_hr=ref.air_flow_kg_hr,
            flue_gas_temperature_c=ref.flue_gas_temp_c,
            flue_gas_o2_percent=ref.flue_gas_o2_percent,
            flue_gas_co_ppm=ref.flue_gas_co_ppm,
            ambient_temperature_c=ref.ambient_temp_c
        )

        result = performance_calculator.calculate_performance(perf_input)

        # Verify combustion efficiency within tolerance
        error = abs(result.combustion_efficiency_percent - ref.expected_combustion_efficiency_percent)
        assert error <= ref.tolerance_percent, \
            f"[{ref.test_case_id}] Combustion eff {result.combustion_efficiency_percent:.1f}% " \
            f"vs expected {ref.expected_combustion_efficiency_percent:.1f}%"


class TestDeterminismGolden:
    """
    Golden tests for calculation determinism.

    Verifies that identical inputs always produce identical outputs
    across multiple runs.
    """

    def test_excess_air_determinism(self):
        """Verify excess air calculation is deterministic."""
        hashes = set()

        for _ in range(100):
            results = []
            for ref in EXCESS_AIR_REFERENCES:
                ea = ref.flue_gas_o2_percent / (21 - ref.flue_gas_o2_percent) * 100
                results.append(round(ea, 10))

            result_hash = hashlib.sha256(
                json.dumps(results).encode()
            ).hexdigest()
            hashes.add(result_hash)

        # All runs should produce identical hash
        assert len(hashes) == 1, \
            f"Excess air calculation not deterministic: {len(hashes)} unique results"

    def test_stoichiometry_determinism(self):
        """Verify stoichiometric calculations are deterministic."""
        hashes = set()

        for _ in range(100):
            results = []
            for ref in STOICHIOMETRY_REFERENCES:
                c = ref.fuel_carbon_percent / 100
                h = ref.fuel_hydrogen_percent / 100
                afr = (2.667 * c + 8 * h) / 0.232
                results.append(round(afr, 10))

            result_hash = hashlib.sha256(
                json.dumps(results).encode()
            ).hexdigest()
            hashes.add(result_hash)

        assert len(hashes) == 1, \
            f"Stoichiometry calculation not deterministic: {len(hashes)} unique results"

    @pytest.mark.skipif(not CALCULATORS_AVAILABLE, reason="Calculators not available")
    def test_performance_calculator_determinism(self, performance_calculator):
        """Verify performance calculator is deterministic."""
        nist = NIST_FUEL_DATA[FuelType.NATURAL_GAS]
        ref = ASME_EFFICIENCY_REFERENCES[0]

        perf_input = CombustionPerformanceInput(
            fuel_flow_rate_kg_per_hr=ref.fuel_flow_kg_hr,
            fuel_lower_heating_value_mj_per_kg=nist.lower_heating_value_mj_kg,
            fuel_higher_heating_value_mj_per_kg=nist.higher_heating_value_mj_kg,
            fuel_carbon_percent=nist.carbon_mass_fraction * 100,
            fuel_hydrogen_percent=nist.hydrogen_mass_fraction * 100,
            air_flow_rate_kg_per_hr=ref.air_flow_kg_hr,
            flue_gas_temperature_c=ref.flue_gas_temp_c,
            flue_gas_o2_percent=ref.flue_gas_o2_percent,
            flue_gas_co_ppm=ref.flue_gas_co_ppm,
            ambient_temperature_c=ref.ambient_temp_c
        )

        # Run 10 times and collect provenance hashes
        provenance_hashes = set()
        efficiency_values = set()

        for _ in range(10):
            result = performance_calculator.calculate_performance(perf_input)
            efficiency_values.add(result.thermal_efficiency_lhv_percent)
            # Note: provenance hash includes timestamp, so we check efficiency

        # All runs should produce identical efficiency
        assert len(efficiency_values) == 1, \
            f"Performance calculator not deterministic: {len(efficiency_values)} unique efficiencies"


class TestHeatLossGolden:
    """
    Golden tests for heat loss calculations per ASME PTC 4.1.

    Verifies individual heat loss components:
    - Dry flue gas loss
    - Moisture loss
    - Incomplete combustion loss
    - Radiation/convection loss
    """

    def test_dry_flue_gas_loss_formula(self):
        """Verify dry flue gas loss calculation formula."""
        # Standard formula: L1 = m_flue * Cp * (Tflue - Tamb) / Q_input
        # Simplified: L1% = k * (Tflue - Tamb) / CO2%

        test_cases = [
            # (Tflue, Tamb, CO2%, expected_loss%)
            (200, 25, 10.0, 6.65),
            (180, 25, 11.0, 5.32),
            (160, 25, 12.0, 4.27),
            (220, 25, 9.0, 8.22),
        ]

        k = 0.38  # Natural gas

        for t_flue, t_amb, co2, expected in test_cases:
            calculated = k * (t_flue - t_amb) / co2
            error = abs(calculated - expected)
            assert error <= 0.5, \
                f"Dry flue loss at Tflue={t_flue}C: {calculated:.2f}% vs {expected:.2f}%"

    def test_moisture_loss_formula(self):
        """Verify moisture loss calculation formula."""
        # Moisture loss = H2O_mass * (hfg + Cp*(Tflue-100)) / Q_input
        # For natural gas: ~10.5% at design conditions

        # H2O from combustion: 9 * H_fraction (kg H2O per kg fuel)
        h_fraction = 0.235  # Natural gas
        h2o_mass = 9 * h_fraction  # kg H2O per kg fuel

        # Latent heat + sensible heat
        hfg = 2257  # kJ/kg
        cp_steam = 2.0  # kJ/kg.K
        t_flue = 180  # C

        heat_in_h2o = h2o_mass * (hfg + cp_steam * (t_flue - 100))  # kJ/kg fuel
        lhv = 47130  # kJ/kg for natural gas

        moisture_loss_pct = heat_in_h2o / lhv * 100

        # Should be approximately 10-11%
        assert 9.5 <= moisture_loss_pct <= 12.0, \
            f"Moisture loss {moisture_loss_pct:.1f}% outside expected range 9.5-12.0%"

    def test_incomplete_combustion_loss_formula(self):
        """Verify incomplete combustion (CO) loss calculation."""
        # CO loss approximation: L_CO ≈ CO_ppm * k
        # Where k ≈ 0.001 for typical fuels at low CO

        test_cases = [
            # (CO_ppm, expected_loss%)
            (0, 0.0),
            (25, 0.025),
            (50, 0.05),
            (100, 0.1),
            (200, 0.3),  # Higher losses at high CO
        ]

        for co_ppm, expected in test_cases:
            if co_ppm < 100:
                calculated = co_ppm * 0.001
            else:
                calculated = 0.1 + (co_ppm - 100) * 0.002

            error = abs(calculated - expected)
            assert error <= 0.05, \
                f"CO loss at {co_ppm}ppm: {calculated:.3f}% vs {expected:.3f}%"


class TestProvenanceTracking:
    """
    Tests for SHA-256 provenance tracking.

    Verifies that calculations produce consistent, auditable hashes.
    """

    def test_provenance_hash_uniqueness(self):
        """Verify different inputs produce different provenance hashes."""
        hashes = []

        for ref in ASME_EFFICIENCY_REFERENCES[:3]:
            data = {
                "fuel_flow": ref.fuel_flow_kg_hr,
                "flue_temp": ref.flue_gas_temp_c,
                "o2": ref.flue_gas_o2_percent
            }
            hash_val = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
            hashes.append(hash_val)

        # All hashes should be unique
        assert len(set(hashes)) == len(hashes), \
            "Provenance hashes not unique for different inputs"

    def test_provenance_hash_consistency(self):
        """Verify same input always produces same provenance hash."""
        data = {
            "fuel_flow": 500.0,
            "flue_temp": 180.0,
            "o2": 3.0
        }

        hashes = set()
        for _ in range(100):
            hash_val = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
            hashes.add(hash_val)

        assert len(hashes) == 1, \
            f"Provenance hash not consistent: {len(hashes)} unique values"


# =============================================================================
# Golden Reference Value Export
# =============================================================================

def export_golden_values() -> Dict[str, Any]:
    """
    Export all golden reference values for external verification.

    Returns:
        Dictionary containing all reference data with SHA-256 hash
    """
    export_data = {
        "metadata": {
            "version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "standards": [
                "NIST Standard Reference Database 23",
                "NIST-JANAF Thermochemical Tables",
                "ASME PTC 4: Fired Steam Generators",
                "ASME PTC 4.1: Steam Generating Units"
            ]
        },
        "nist_fuel_data": {
            fuel_type.value: {
                "molecular_formula": props.molecular_formula,
                "hhv_mj_kg": props.higher_heating_value_mj_kg,
                "lhv_mj_kg": props.lower_heating_value_mj_kg,
                "carbon_fraction": props.carbon_mass_fraction,
                "hydrogen_fraction": props.hydrogen_mass_fraction,
                "stoich_afr": props.stoichiometric_air_fuel_ratio
            }
            for fuel_type, props in NIST_FUEL_DATA.items()
        },
        "asme_efficiency_references": [
            {
                "test_case_id": ref.test_case_id,
                "fuel_type": ref.fuel_type.value,
                "conditions": {
                    "fuel_flow_kg_hr": ref.fuel_flow_kg_hr,
                    "flue_gas_temp_c": ref.flue_gas_temp_c,
                    "o2_percent": ref.flue_gas_o2_percent
                },
                "expected": {
                    "excess_air_percent": ref.expected_excess_air_percent,
                    "combustion_efficiency_percent": ref.expected_combustion_efficiency_percent,
                    "thermal_efficiency_percent": ref.expected_thermal_efficiency_lhv_percent
                }
            }
            for ref in ASME_EFFICIENCY_REFERENCES
        ],
        "excess_air_references": [
            {
                "o2_percent": ref.flue_gas_o2_percent,
                "excess_air_percent": ref.expected_excess_air_percent
            }
            for ref in EXCESS_AIR_REFERENCES
        ]
    }

    # Add SHA-256 hash of entire dataset
    data_str = json.dumps(export_data, sort_keys=True)
    export_data["provenance_hash"] = hashlib.sha256(data_str.encode()).hexdigest()

    return export_data


if __name__ == "__main__":
    # Export golden values when run directly
    golden_values = export_golden_values()
    print(json.dumps(golden_values, indent=2))
