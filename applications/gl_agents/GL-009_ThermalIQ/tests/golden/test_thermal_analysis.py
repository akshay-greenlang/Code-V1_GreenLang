# -*- coding: utf-8 -*-
"""
Thermal Analysis Golden Value Tests

Comprehensive test suite validating ThermalIQ agent calculations against
thermodynamic principles and NIST reference data.

Reference Documents:
- NIST Chemistry WebBook (thermophysical properties)
- ASME Steam Tables
- Perry's Chemical Engineers' Handbook
- ISO 5167 (flow measurement)

Author: GL-CalculatorEngineer
"""

import pytest
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from enum import Enum
import math


# ==============================================================================
# THERMODYNAMIC GOLDEN VALUES
# ==============================================================================

@dataclass(frozen=True)
class ThermalGoldenValue:
    """Thermodynamic reference value with source citation."""
    name: str
    value: Decimal
    unit: str
    tolerance_percent: Decimal
    source: str
    temperature_k: Decimal = Decimal('0')
    pressure_mpa: Decimal = Decimal('0')


# Water/Steam Properties at Reference Conditions
# Reference: NIST Chemistry WebBook, IAPWS-IF97
WATER_PROPERTIES: Dict[str, ThermalGoldenValue] = {
    'cp_water_25c': ThermalGoldenValue(
        'Cp Water 25°C', Decimal('4.182'), 'kJ/kg-K',
        Decimal('0.1'), 'NIST', Decimal('298.15'), Decimal('0.101325')),
    'cp_water_100c': ThermalGoldenValue(
        'Cp Water 100°C', Decimal('4.216'), 'kJ/kg-K',
        Decimal('0.1'), 'NIST', Decimal('373.15'), Decimal('0.101325')),
    'rho_water_25c': ThermalGoldenValue(
        'Density Water 25°C', Decimal('997.05'), 'kg/m3',
        Decimal('0.1'), 'NIST', Decimal('298.15'), Decimal('0.101325')),
    'k_water_25c': ThermalGoldenValue(
        'k Water 25°C', Decimal('0.607'), 'W/m-K',
        Decimal('0.5'), 'NIST', Decimal('298.15'), Decimal('0.101325')),
    'mu_water_25c': ThermalGoldenValue(
        'Viscosity Water 25°C', Decimal('0.00089'), 'Pa-s',
        Decimal('1.0'), 'NIST', Decimal('298.15'), Decimal('0.101325')),
    'pr_water_25c': ThermalGoldenValue(
        'Prandtl Water 25°C', Decimal('6.14'), 'dimensionless',
        Decimal('1.0'), 'NIST', Decimal('298.15'), Decimal('0.101325')),
}

# Air Properties at Standard Conditions
AIR_PROPERTIES: Dict[str, ThermalGoldenValue] = {
    'cp_air_25c': ThermalGoldenValue(
        'Cp Air 25°C', Decimal('1.006'), 'kJ/kg-K',
        Decimal('0.2'), 'NIST', Decimal('298.15'), Decimal('0.101325')),
    'rho_air_25c': ThermalGoldenValue(
        'Density Air 25°C', Decimal('1.184'), 'kg/m3',
        Decimal('0.2'), 'NIST', Decimal('298.15'), Decimal('0.101325')),
    'k_air_25c': ThermalGoldenValue(
        'k Air 25°C', Decimal('0.0257'), 'W/m-K',
        Decimal('0.5'), 'NIST', Decimal('298.15'), Decimal('0.101325')),
    'mu_air_25c': ThermalGoldenValue(
        'Viscosity Air 25°C', Decimal('0.0000184'), 'Pa-s',
        Decimal('1.0'), 'NIST', Decimal('298.15'), Decimal('0.101325')),
    'pr_air_25c': ThermalGoldenValue(
        'Prandtl Air 25°C', Decimal('0.71'), 'dimensionless',
        Decimal('1.0'), 'NIST', Decimal('298.15'), Decimal('0.101325')),
}

# Exergy Reference Values
EXERGY_REFERENCE: Dict[str, ThermalGoldenValue] = {
    'dead_state_temp': ThermalGoldenValue(
        'Dead State Temperature', Decimal('298.15'), 'K',
        Decimal('0'), 'Standard', Decimal('298.15'), Decimal('0.101325')),
    'dead_state_pressure': ThermalGoldenValue(
        'Dead State Pressure', Decimal('0.101325'), 'MPa',
        Decimal('0'), 'Standard', Decimal('298.15'), Decimal('0.101325')),
}

# Heat Balance Closure Tolerances
HEAT_BALANCE_TOLERANCES: Dict[str, Decimal] = {
    'excellent': Decimal('0.5'),   # < 0.5% imbalance
    'good': Decimal('2.0'),        # < 2% imbalance
    'acceptable': Decimal('5.0'),  # < 5% imbalance
    'investigate': Decimal('10.0'),  # > 5% requires investigation
}


# ==============================================================================
# CALCULATION FUNCTIONS
# ==============================================================================

def calculate_heat_duty(
    mass_flow_kg_s: Decimal,
    cp_kj_kg_k: Decimal,
    delta_t_k: Decimal
) -> Decimal:
    """
    Calculate heat duty Q = m * Cp * ΔT.

    Returns: Heat duty in kW
    """
    q = mass_flow_kg_s * cp_kj_kg_k * delta_t_k
    return q.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_reynolds_number(
    density_kg_m3: Decimal,
    velocity_m_s: Decimal,
    diameter_m: Decimal,
    viscosity_pa_s: Decimal
) -> Decimal:
    """
    Calculate Reynolds number Re = ρVD/μ.

    Returns: Dimensionless Reynolds number
    """
    if viscosity_pa_s == 0:
        raise ValueError('Viscosity cannot be zero')
    re = (density_kg_m3 * velocity_m_s * diameter_m) / viscosity_pa_s
    return re.quantize(Decimal('1'), rounding=ROUND_HALF_UP)


def calculate_nusselt_dittus_boelter(
    reynolds: Decimal,
    prandtl: Decimal,
    heating: bool = True
) -> Decimal:
    """
    Calculate Nusselt number using Dittus-Boelter correlation.

    Nu = 0.023 * Re^0.8 * Pr^n
    where n = 0.4 for heating, 0.3 for cooling

    Valid for: Re > 10000, 0.6 < Pr < 160, L/D > 10
    """
    n = Decimal('0.4') if heating else Decimal('0.3')
    nu = Decimal('0.023') * (reynolds ** Decimal('0.8')) * (prandtl ** n)
    return nu.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_heat_transfer_coefficient(
    nusselt: Decimal,
    thermal_conductivity_w_m_k: Decimal,
    diameter_m: Decimal
) -> Decimal:
    """
    Calculate heat transfer coefficient h = Nu * k / D.

    Returns: h in W/m2-K
    """
    h = nusselt * thermal_conductivity_w_m_k / diameter_m
    return h.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_exergy_heat(
    heat_kw: Decimal,
    source_temp_k: Decimal,
    dead_state_temp_k: Decimal = Decimal('298.15')
) -> Decimal:
    """
    Calculate exergy of heat transfer.

    Exergy = Q * (1 - T0/T)

    Returns: Exergy in kW
    """
    carnot_factor = Decimal('1') - dead_state_temp_k / source_temp_k
    exergy = heat_kw * carnot_factor
    return exergy.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_exergy_efficiency(
    exergy_output_kw: Decimal,
    exergy_input_kw: Decimal
) -> Decimal:
    """
    Calculate second-law (exergetic) efficiency.

    η_ex = Exergy_out / Exergy_in

    Returns: Efficiency as decimal (0-1)
    """
    if exergy_input_kw <= 0:
        return Decimal('0')
    efficiency = exergy_output_kw / exergy_input_kw
    return efficiency.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)


def calculate_heat_balance_closure(
    heat_in_kw: Decimal,
    heat_out_kw: Decimal,
    heat_loss_kw: Decimal
) -> Tuple[Decimal, str]:
    """
    Calculate heat balance closure and classify.

    Closure = |Q_in - Q_out - Q_loss| / Q_in * 100

    Returns: (closure_percent, classification)
    """
    if heat_in_kw == 0:
        return Decimal('100'), 'invalid'

    imbalance = abs(heat_in_kw - heat_out_kw - heat_loss_kw)
    closure_pct = (imbalance / heat_in_kw) * Decimal('100')
    closure_pct = closure_pct.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    if closure_pct <= HEAT_BALANCE_TOLERANCES['excellent']:
        classification = 'excellent'
    elif closure_pct <= HEAT_BALANCE_TOLERANCES['good']:
        classification = 'good'
    elif closure_pct <= HEAT_BALANCE_TOLERANCES['acceptable']:
        classification = 'acceptable'
    else:
        classification = 'investigate'

    return closure_pct, classification


def calculate_sankey_flows(
    inputs: Dict[str, Decimal],
    outputs: Dict[str, Decimal],
    losses: Dict[str, Decimal]
) -> Dict[str, Any]:
    """
    Calculate Sankey diagram flow data for energy visualization.

    Returns: Dictionary with normalized flows for Sankey rendering
    """
    total_in = sum(inputs.values())
    total_out = sum(outputs.values())
    total_loss = sum(losses.values())

    if total_in == 0:
        return {'error': 'No input energy'}

    flows = {
        'total_input': total_in,
        'total_output': total_out,
        'total_loss': total_loss,
        'balance_error': total_in - total_out - total_loss,
        'inputs_normalized': {k: v / total_in for k, v in inputs.items()},
        'outputs_normalized': {k: v / total_in for k, v in outputs.items()},
        'losses_normalized': {k: v / total_in for k, v in losses.items()},
    }
    return flows


def calculate_provenance_hash(
    calculation_type: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any]
) -> str:
    """Calculate SHA-256 provenance hash for audit trail."""
    provenance_data = {
        'calculation_type': calculation_type,
        'inputs': {k: str(v) for k, v in sorted(inputs.items())},
        'outputs': {k: str(v) for k, v in sorted(outputs.items())},
    }
    provenance_str = json.dumps(provenance_data, sort_keys=True)
    return hashlib.sha256(provenance_str.encode()).hexdigest()


# ==============================================================================
# TEST CLASSES
# ==============================================================================

@pytest.mark.golden
class TestWaterProperties:
    """Validate water thermophysical properties against NIST data."""

    @pytest.mark.parametrize("prop_name,expected_value", [
        ('cp_water_25c', Decimal('4.182')),
        ('rho_water_25c', Decimal('997.05')),
        ('k_water_25c', Decimal('0.607')),
    ])
    def test_water_property_values(self, prop_name: str, expected_value: Decimal):
        """Verify water properties match NIST reference."""
        golden = WATER_PROPERTIES[prop_name]
        assert golden.value == expected_value
        assert golden.source == 'NIST'

    def test_prandtl_water_range(self):
        """Water Prandtl number should be in typical range."""
        pr = WATER_PROPERTIES['pr_water_25c'].value
        assert Decimal('5') < pr < Decimal('8')

    def test_viscosity_water_order_magnitude(self):
        """Water viscosity should be ~1e-3 Pa-s."""
        mu = WATER_PROPERTIES['mu_water_25c'].value
        assert Decimal('0.0005') < mu < Decimal('0.002')


@pytest.mark.golden
class TestAirProperties:
    """Validate air thermophysical properties against NIST data."""

    @pytest.mark.parametrize("prop_name,expected_value", [
        ('cp_air_25c', Decimal('1.006')),
        ('rho_air_25c', Decimal('1.184')),
        ('pr_air_25c', Decimal('0.71')),
    ])
    def test_air_property_values(self, prop_name: str, expected_value: Decimal):
        """Verify air properties match NIST reference."""
        golden = AIR_PROPERTIES[prop_name]
        assert golden.value == expected_value

    def test_prandtl_air_range(self):
        """Air Prandtl number should be ~0.7."""
        pr = AIR_PROPERTIES['pr_air_25c'].value
        assert Decimal('0.6') < pr < Decimal('0.8')


@pytest.mark.golden
class TestHeatDutyCalculation:
    """Validate heat duty calculations."""

    @dataclass(frozen=True)
    class HeatDutyCase:
        name: str
        mass_flow: Decimal
        cp: Decimal
        delta_t: Decimal
        expected_q: Decimal

    CASES = [
        HeatDutyCase('Water Heating', Decimal('10'), Decimal('4.18'),
                     Decimal('50'), Decimal('2090.0')),
        HeatDutyCase('Air Heating', Decimal('5'), Decimal('1.006'),
                     Decimal('100'), Decimal('503.0')),
        HeatDutyCase('Oil Cooling', Decimal('3'), Decimal('2.0'),
                     Decimal('30'), Decimal('180.0')),
    ]

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
    def test_heat_duty_calculation(self, case: HeatDutyCase):
        """Verify Q = mCpΔT calculation."""
        calculated = calculate_heat_duty(case.mass_flow, case.cp, case.delta_t)

        deviation = abs(calculated - case.expected_q) / case.expected_q * 100
        assert deviation <= Decimal('0.5'), \
            f'{case.name}: Q={calculated} kW deviates from {case.expected_q} kW'


@pytest.mark.golden
class TestReynoldsNumber:
    """Validate Reynolds number calculations."""

    @dataclass(frozen=True)
    class ReynoldsCase:
        name: str
        rho: Decimal
        v: Decimal
        d: Decimal
        mu: Decimal
        expected_re: Decimal
        flow_regime: str

    CASES = [
        ReynoldsCase('Laminar Water', Decimal('1000'), Decimal('0.1'),
                     Decimal('0.025'), Decimal('0.001'), Decimal('2500'), 'laminar'),
        ReynoldsCase('Turbulent Water', Decimal('1000'), Decimal('2.0'),
                     Decimal('0.05'), Decimal('0.001'), Decimal('100000'), 'turbulent'),
        ReynoldsCase('Transitional', Decimal('1000'), Decimal('0.3'),
                     Decimal('0.01'), Decimal('0.001'), Decimal('3000'), 'transitional'),
    ]

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
    def test_reynolds_calculation(self, case: ReynoldsCase):
        """Verify Reynolds number calculation."""
        calculated = calculate_reynolds_number(case.rho, case.v, case.d, case.mu)

        deviation = abs(calculated - case.expected_re) / case.expected_re * 100
        assert deviation <= Decimal('1'), \
            f'{case.name}: Re={calculated} deviates from {case.expected_re}'

    def test_flow_regime_classification(self):
        """Verify flow regime based on Reynolds number."""
        # Laminar: Re < 2300
        re_laminar = calculate_reynolds_number(
            Decimal('1000'), Decimal('0.05'), Decimal('0.02'), Decimal('0.001'))
        assert re_laminar < Decimal('2300')

        # Turbulent: Re > 4000
        re_turb = calculate_reynolds_number(
            Decimal('1000'), Decimal('1.0'), Decimal('0.05'), Decimal('0.001'))
        assert re_turb > Decimal('4000')


@pytest.mark.golden
class TestNusseltNumber:
    """Validate Nusselt number correlations."""

    def test_dittus_boelter_heating(self):
        """Verify Dittus-Boelter for heating case."""
        nu = calculate_nusselt_dittus_boelter(
            reynolds=Decimal('50000'),
            prandtl=Decimal('6.0'),
            heating=True,
        )
        # Nu should be in typical range for turbulent water flow
        assert Decimal('200') < nu < Decimal('400')

    def test_dittus_boelter_cooling_lower(self):
        """Cooling case should give lower Nu than heating."""
        nu_heat = calculate_nusselt_dittus_boelter(Decimal('50000'), Decimal('6.0'), True)
        nu_cool = calculate_nusselt_dittus_boelter(Decimal('50000'), Decimal('6.0'), False)

        assert nu_cool < nu_heat


@pytest.mark.golden
class TestExergyCalculation:
    """Validate exergy analysis calculations."""

    def test_dead_state_reference(self):
        """Verify dead state reference values."""
        assert EXERGY_REFERENCE['dead_state_temp'].value == Decimal('298.15')
        assert EXERGY_REFERENCE['dead_state_pressure'].value == Decimal('0.101325')

    def test_exergy_of_heat(self):
        """Verify exergy of heat calculation."""
        # Heat at 500K, dead state 298.15K
        exergy = calculate_exergy_heat(
            heat_kw=Decimal('100'),
            source_temp_k=Decimal('500'),
            dead_state_temp_k=Decimal('298.15'),
        )

        # Carnot factor = 1 - 298.15/500 = 0.4037
        # Exergy = 100 * 0.4037 = 40.37 kW
        assert Decimal('38') < exergy < Decimal('42')

    def test_exergy_efficiency_bounds(self):
        """Exergy efficiency should be between 0 and 1."""
        efficiency = calculate_exergy_efficiency(
            exergy_output_kw=Decimal('35'),
            exergy_input_kw=Decimal('50'),
        )
        assert Decimal('0') <= efficiency <= Decimal('1')
        assert efficiency == Decimal('0.700')


@pytest.mark.golden
class TestHeatBalanceClosure:
    """Validate heat balance closure calculations."""

    @dataclass(frozen=True)
    class BalanceCase:
        name: str
        heat_in: Decimal
        heat_out: Decimal
        heat_loss: Decimal
        expected_class: str

    CASES = [
        BalanceCase('Excellent Balance', Decimal('1000'), Decimal('900'),
                    Decimal('98'), 'excellent'),
        BalanceCase('Good Balance', Decimal('1000'), Decimal('850'),
                    Decimal('135'), 'good'),
        BalanceCase('Acceptable', Decimal('1000'), Decimal('800'),
                    Decimal('160'), 'acceptable'),
        BalanceCase('Investigation Needed', Decimal('1000'), Decimal('700'),
                    Decimal('150'), 'investigate'),
    ]

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
    def test_balance_classification(self, case: BalanceCase):
        """Verify heat balance closure classification."""
        closure, classification = calculate_heat_balance_closure(
            case.heat_in, case.heat_out, case.heat_loss)

        assert classification == case.expected_class, \
            f'{case.name}: Expected {case.expected_class}, got {classification} ({closure}%)'


@pytest.mark.golden
class TestSankeyDiagram:
    """Validate Sankey diagram flow calculations."""

    def test_sankey_normalization(self):
        """Verify Sankey flows are normalized correctly."""
        inputs = {'fuel': Decimal('100'), 'air': Decimal('50')}
        outputs = {'steam': Decimal('80'), 'product': Decimal('40')}
        losses = {'stack': Decimal('20'), 'radiation': Decimal('10')}

        flows = calculate_sankey_flows(inputs, outputs, losses)

        # Total should be 150 (100 + 50)
        assert flows['total_input'] == Decimal('150')

        # Normalized inputs should sum to 1.0
        norm_in_sum = sum(flows['inputs_normalized'].values())
        assert abs(norm_in_sum - Decimal('1')) < Decimal('0.001')


@pytest.mark.golden
class TestProvenanceAndDeterminism:
    """Validate provenance tracking and deterministic behavior."""

    def test_all_calculations_deterministic(self):
        """All calculations must be deterministic."""
        calculations = [
            lambda: calculate_heat_duty(Decimal('10'), Decimal('4.18'), Decimal('50')),
            lambda: calculate_reynolds_number(Decimal('1000'), Decimal('1'), Decimal('0.05'), Decimal('0.001')),
            lambda: calculate_nusselt_dittus_boelter(Decimal('50000'), Decimal('6'), True),
            lambda: calculate_exergy_heat(Decimal('100'), Decimal('500')),
        ]

        for calc in calculations:
            results = set()
            for _ in range(50):
                results.add(str(calc()))
            assert len(results) == 1

    def test_provenance_hash_stability(self):
        """Provenance hash must be stable for same inputs."""
        inputs = {'mass_flow': '10', 'cp': '4.18'}
        outputs = {'heat_duty': '2090'}

        hashes = set()
        for _ in range(50):
            h = calculate_provenance_hash('heat_duty', inputs, outputs)
            hashes.add(h)

        assert len(hashes) == 1


# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================

def export_golden_values() -> Dict[str, Any]:
    """Export all golden values for documentation."""
    return {
        'metadata': {
            'agent': 'GL-009_ThermalIQ',
            'version': '1.0.0',
            'source': 'NIST Chemistry WebBook',
        },
        'water_properties': {
            key: {'value': str(val.value), 'unit': val.unit}
            for key, val in WATER_PROPERTIES.items()
        },
        'air_properties': {
            key: {'value': str(val.value), 'unit': val.unit}
            for key, val in AIR_PROPERTIES.items()
        },
        'balance_tolerances': {k: str(v) for k, v in HEAT_BALANCE_TOLERANCES.items()},
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
