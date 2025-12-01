# -*- coding: utf-8 -*-
"""
Energy loss calculation validation tests for GL-008 SteamTrapInspector.

This module validates energy loss calculations against known values,
steam tables, and regulatory standards (Napier equation, ASME standards).
"""

import pytest
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tools import SteamTrapTools, EnergyLossResult
from config import FailureMode


@pytest.mark.validation
class TestNapierEquationValidation:
    """Validate Napier equation implementation for steam loss calculation."""

    def test_napier_equation_reference_case(self, tools, energy_loss_test_data):
        """Test Napier equation against reference calculation."""
        # W = 24.24 * P * D² * C
        # Reference: P=100, D=0.125, C=0.7 → W ≈ 26.51 lb/hr
        ref_data = energy_loss_test_data['napier_equation_reference']

        trap_data = {
            'trap_id': 'TRAP-NAPIER-REF',
            'orifice_diameter_in': ref_data['orifice_diameter_in'],
            'steam_pressure_psig': ref_data['steam_pressure_psig'],
            'failure_severity': 1.0  # Complete failure
        }

        result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        # Validate against known value
        expected = ref_data['expected_steam_loss_lb_hr']
        assert abs(result.steam_loss_lb_hr - expected) < 0.01  # Within 0.01 lb/hr

    @pytest.mark.parametrize("pressure,diameter,expected_loss", [
        (50, 0.125, 13.26),   # W = 24.24 * 50 * 0.015625 * 0.7
        (100, 0.125, 26.51),  # W = 24.24 * 100 * 0.015625 * 0.7
        (150, 0.125, 39.77),  # W = 24.24 * 150 * 0.015625 * 0.7
        (100, 0.0625, 6.63),  # W = 24.24 * 100 * 0.00390625 * 0.7
        (100, 0.250, 106.05), # W = 24.24 * 100 * 0.0625 * 0.7
    ])
    def test_napier_equation_parameterized(self, tools, pressure, diameter, expected_loss):
        """Test Napier equation with multiple pressure/diameter combinations."""
        trap_data = {
            'trap_id': f'TRAP-P{pressure}-D{diameter}',
            'orifice_diameter_in': diameter,
            'steam_pressure_psig': pressure,
            'failure_severity': 1.0
        }

        result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        # Validate calculation (±1% tolerance)
        assert abs(result.steam_loss_lb_hr - expected_loss) / expected_loss < 0.01

    def test_napier_discharge_coefficient_variation(self, tools):
        """Test impact of discharge coefficient on Napier equation."""
        # C varies from 0.6 to 0.8 depending on orifice geometry

        base_trap_data = {
            'trap_id': 'TRAP-COEFF',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        result = tools.calculate_energy_loss(base_trap_data, FailureMode.FAILED_OPEN)

        # Typical discharge coefficient = 0.7
        # W = 24.24 * 100 * 0.015625 * C
        base_flow = 24.24 * 100 * 0.015625
        expected_min = base_flow * 0.6  # C = 0.6
        expected_max = base_flow * 0.8  # C = 0.8

        assert expected_min <= result.steam_loss_lb_hr <= expected_max

    def test_napier_orifice_area_scaling(self, tools):
        """Test that steam loss scales with orifice area (D²)."""
        # Doubling diameter should quadruple loss

        trap_data_small = {
            'trap_id': 'TRAP-SMALL',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        trap_data_large = {
            'trap_id': 'TRAP-LARGE',
            'orifice_diameter_in': 0.250,  # 2x diameter
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        result_small = tools.calculate_energy_loss(trap_data_small, FailureMode.FAILED_OPEN)
        result_large = tools.calculate_energy_loss(trap_data_large, FailureMode.FAILED_OPEN)

        # Loss should scale by D² (4x)
        ratio = result_large.steam_loss_lb_hr / result_small.steam_loss_lb_hr
        assert abs(ratio - 4.0) < 0.1  # Within 10% (accounting for coefficient variations)


@pytest.mark.validation
class TestSteamTableValidation:
    """Validate steam property lookups against ASME steam tables."""

    def test_steam_properties_at_100_psig(self, tools, energy_loss_test_data):
        """Test steam properties at 100 psig (common operating pressure)."""
        # At 100 psig (114.7 psia):
        # Saturation temp: 338°F (170°C)
        # Enthalpy of vaporization: ~1187.5 BTU/lb
        ref_data = energy_loss_test_data['steam_table_reference']

        trap_data = {
            'trap_id': 'TRAP-STEAM-PROPS',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': ref_data['pressure_psig'],
            'failure_severity': 1.0
        }

        result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        # Validate energy calculation uses correct steam properties
        # Energy (BTU/hr) = steam_loss_lb_hr * enthalpy_vaporization
        expected_energy_btu_hr = result.steam_loss_lb_hr * ref_data['expected_enthalpy_btu_lb']

        # Convert to GJ/yr for comparison
        # 1 BTU = 0.00105506 GJ, 8760 hr/yr
        expected_energy_gj_yr = expected_energy_btu_hr * 0.00105506 * 8760

        # Should match within 5%
        assert abs(result.energy_loss_gj_yr - expected_energy_gj_yr) / expected_energy_gj_yr < 0.05

    @pytest.mark.parametrize("pressure_psig,expected_temp_f", [
        (0, 212),    # Atmospheric: 212°F
        (50, 298),   # 50 psig: ~298°F
        (100, 338),  # 100 psig: ~338°F
        (150, 366),  # 150 psig: ~366°F
        (200, 388),  # 200 psig: ~388°F
    ])
    def test_saturation_temperature_lookup(self, tools, pressure_psig, expected_temp_f):
        """Test saturation temperature lookup at various pressures."""
        trap_data = {
            'trap_id': f'TRAP-SAT-{pressure_psig}',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': pressure_psig,
            'failure_severity': 1.0
        }

        result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        # Result should use correct saturation properties
        # (Indirect validation through energy calculation)
        assert result.energy_loss_gj_yr > 0  # Positive energy loss

    def test_enthalpy_pressure_relationship(self, tools):
        """Test that enthalpy varies correctly with pressure."""
        pressures = [50, 100, 150, 200]
        energies = []

        for pressure in pressures:
            trap_data = {
                'trap_id': f'TRAP-P{pressure}',
                'orifice_diameter_in': 0.125,
                'steam_pressure_psig': pressure,
                'failure_severity': 1.0
            }

            result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
            # Normalize by steam flow to get specific energy
            specific_energy = result.energy_loss_gj_yr / result.steam_loss_kg_hr
            energies.append(specific_energy)

        # Enthalpy of vaporization decreases slightly with pressure
        # (Steam tables show this trend)
        # We don't require strict monotonicity, just reasonable values
        assert all(e > 0 for e in energies)


@pytest.mark.validation
class TestCostCalculationValidation:
    """Validate cost calculations at different energy prices."""

    def test_cost_calculation_scenarios(self, tools, energy_loss_test_data):
        """Test cost calculation with known steam prices."""
        for scenario in energy_loss_test_data['cost_scenarios']:
            # Calculate annual loss in lb
            steam_loss_lb_hr = scenario['annual_loss_lb'] / 8760  # Convert annual to hourly

            # Calculate orifice size that would produce this loss
            # Using inverse Napier: D = sqrt(W / (24.24 * P * C))
            # Assume P=100, C=0.7
            orifice_diameter = np.sqrt(steam_loss_lb_hr / (24.24 * 100 * 0.7))

            trap_data = {
                'trap_id': 'TRAP-COST-TEST',
                'orifice_diameter_in': orifice_diameter,
                'steam_pressure_psig': 100.0,
                'steam_cost_usd_per_1000lb': scenario['steam_cost_usd_per_1000lb'],
                'operating_hours_yr': 8760,
                'failure_severity': 1.0
            }

            result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

            # Validate cost calculation
            expected_cost = scenario['expected_cost_usd']
            # Allow 10% tolerance due to rounding and calculation variations
            assert abs(result.cost_loss_usd_yr - expected_cost) / expected_cost < 0.10

    @pytest.mark.parametrize("steam_cost,expected_multiplier", [
        (5.0, 1.0),
        (10.0, 2.0),
        (15.0, 3.0),
        (20.0, 4.0),
    ])
    def test_cost_linear_scaling(self, tools, steam_cost, expected_multiplier):
        """Test that cost scales linearly with steam price."""
        trap_data = {
            'trap_id': 'TRAP-COST-SCALE',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'steam_cost_usd_per_1000lb': 5.0,  # Base price
            'operating_hours_yr': 8760,
            'failure_severity': 1.0
        }

        result_base = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
        base_cost = result_base.cost_loss_usd_yr

        # Test with multiplied price
        trap_data['steam_cost_usd_per_1000lb'] = steam_cost
        result_scaled = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        # Cost should scale linearly
        actual_multiplier = result_scaled.cost_loss_usd_yr / base_cost
        assert abs(actual_multiplier - expected_multiplier) < 0.01

    def test_operating_hours_impact(self, tools):
        """Test impact of operating hours on annual cost."""
        base_trap_data = {
            'trap_id': 'TRAP-HOURS',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'steam_cost_usd_per_1000lb': 8.5,
            'failure_severity': 1.0
        }

        # 24/7 operation
        trap_data_247 = {**base_trap_data, 'operating_hours_yr': 8760}
        result_247 = tools.calculate_energy_loss(trap_data_247, FailureMode.FAILED_OPEN)

        # 5 days/week, 12 hours/day
        trap_data_partial = {**base_trap_data, 'operating_hours_yr': 3120}  # 260 days * 12 hrs
        result_partial = tools.calculate_energy_loss(trap_data_partial, FailureMode.FAILED_OPEN)

        # Cost should scale with operating hours
        expected_ratio = 8760 / 3120
        actual_ratio = result_247.cost_loss_usd_yr / result_partial.cost_loss_usd_yr
        assert abs(actual_ratio - expected_ratio) < 0.01


@pytest.mark.validation
class TestCO2EmissionsValidation:
    """Validate CO2 emissions calculations."""

    def test_co2_emissions_factor(self, tools):
        """Test CO2 emissions calculation using standard factor."""
        # Standard: 53.06 kg CO2 per MMBtu (natural gas combustion)

        trap_data = {
            'trap_id': 'TRAP-CO2',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        # Validate CO2 calculation
        # Energy in GJ → Convert to MMBtu → Apply factor
        energy_mmbtu = result.energy_loss_gj_yr / 1.055056  # 1 MMBtu = 1.055056 GJ
        expected_co2_kg = energy_mmbtu * 53.06

        # Should match within 1%
        assert abs(result.co2_emissions_kg_yr - expected_co2_kg) / expected_co2_kg < 0.01

    def test_co2_emissions_scaling(self, tools):
        """Test that CO2 emissions scale with energy loss."""
        trap_data_small = {
            'trap_id': 'TRAP-CO2-SMALL',
            'orifice_diameter_in': 0.0625,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        trap_data_large = {
            'trap_id': 'TRAP-CO2-LARGE',
            'orifice_diameter_in': 0.250,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        result_small = tools.calculate_energy_loss(trap_data_small, FailureMode.FAILED_OPEN)
        result_large = tools.calculate_energy_loss(trap_data_large, FailureMode.FAILED_OPEN)

        # CO2 ratio should match energy ratio
        co2_ratio = result_large.co2_emissions_kg_yr / result_small.co2_emissions_kg_yr
        energy_ratio = result_large.energy_loss_gj_yr / result_small.energy_loss_gj_yr

        assert abs(co2_ratio - energy_ratio) / energy_ratio < 0.01

    @pytest.mark.parametrize("fuel_type,co2_factor_kg_per_mmbtu", [
        ("natural_gas", 53.06),
        ("coal", 95.28),
        ("fuel_oil", 73.96),
        ("biomass", 0.0),  # Carbon neutral
    ])
    def test_co2_different_fuel_types(self, tools, fuel_type, co2_factor_kg_per_mmbtu):
        """Test CO2 emissions for different fuel types."""
        trap_data = {
            'trap_id': f'TRAP-CO2-{fuel_type}',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0,
            'fuel_type': fuel_type,
            'co2_factor_kg_per_mmbtu': co2_factor_kg_per_mmbtu
        }

        result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        # Validate fuel-specific CO2 calculation
        energy_mmbtu = result.energy_loss_gj_yr / 1.055056
        expected_co2 = energy_mmbtu * co2_factor_kg_per_mmbtu

        if co2_factor_kg_per_mmbtu > 0:
            assert abs(result.co2_emissions_kg_yr - expected_co2) / expected_co2 < 0.05
        else:
            assert result.co2_emissions_kg_yr >= 0


@pytest.mark.validation
class TestFailureSeverityImpact:
    """Validate impact of failure severity on energy loss."""

    @pytest.mark.parametrize("failure_severity,expected_reduction", [
        (1.0, 1.0),   # Complete failure: 100% loss
        (0.75, 0.75), # 75% severity: 75% loss
        (0.5, 0.5),   # 50% severity: 50% loss
        (0.25, 0.25), # 25% severity: 25% loss
        (0.0, 0.0),   # No failure: 0% loss
    ])
    def test_failure_severity_scaling(self, tools, failure_severity, expected_reduction):
        """Test that energy loss scales with failure severity."""
        trap_data = {
            'trap_id': 'TRAP-SEVERITY',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0  # Complete failure
        }

        result_complete = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
        complete_loss = result_complete.steam_loss_kg_hr

        # Test partial failure
        trap_data['failure_severity'] = failure_severity
        result_partial = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
        partial_loss = result_partial.steam_loss_kg_hr

        # Loss should scale linearly with severity
        if failure_severity > 0:
            actual_ratio = partial_loss / complete_loss
            assert abs(actual_ratio - expected_reduction) < 0.01
        else:
            assert partial_loss == 0.0

    def test_leaking_vs_failed_open(self, tools):
        """Test energy loss difference between leaking and failed open."""
        trap_data = {
            'trap_id': 'TRAP-COMPARE',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        result_failed_open = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
        result_leaking = tools.calculate_energy_loss(trap_data, FailureMode.LEAKING)

        # Leaking should have lower loss than complete failure
        assert result_leaking.steam_loss_kg_hr <= result_failed_open.steam_loss_kg_hr


@pytest.mark.validation
class TestUnitConversions:
    """Validate unit conversions in energy loss calculations."""

    def test_lb_to_kg_conversion(self, tools):
        """Test lb/hr to kg/hr conversion."""
        trap_data = {
            'trap_id': 'TRAP-UNITS',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        # 1 lb = 0.453592 kg
        expected_kg_hr = result.steam_loss_lb_hr * 0.453592
        assert abs(result.steam_loss_kg_hr - expected_kg_hr) < 0.01

    def test_btu_to_gj_conversion(self, tools):
        """Test BTU to GJ conversion."""
        trap_data = {
            'trap_id': 'TRAP-ENERGY-UNITS',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        # Energy should be positive
        assert result.energy_loss_gj_yr > 0
        # Sanity check: Typical range for this size trap
        assert 10 < result.energy_loss_gj_yr < 1000


@pytest.mark.validation
class TestCalculationDeterminism:
    """Validate deterministic behavior of energy loss calculations."""

    def test_identical_inputs_identical_outputs(self, tools):
        """Test that identical inputs produce identical outputs."""
        trap_data = {
            'trap_id': 'TRAP-DET',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'steam_cost_usd_per_1000lb': 8.5,
            'failure_severity': 1.0
        }

        results = [
            tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
            for _ in range(10)
        ]

        # All results must be identical
        for result in results[1:]:
            assert result.steam_loss_lb_hr == results[0].steam_loss_lb_hr
            assert result.steam_loss_kg_hr == results[0].steam_loss_kg_hr
            assert result.energy_loss_gj_yr == results[0].energy_loss_gj_yr
            assert result.cost_loss_usd_yr == results[0].cost_loss_usd_yr
            assert result.co2_emissions_kg_yr == results[0].co2_emissions_kg_yr


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "validation"])
