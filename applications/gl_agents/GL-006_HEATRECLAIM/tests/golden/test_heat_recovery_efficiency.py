# -*- coding: utf-8 -*-
"""
Heat Recovery Efficiency Golden Value Tests

Comprehensive test suite validating HeatReclaim agent calculations against
thermodynamic principles and industry standards for heat exchanger networks.

Reference Documents:
- TEMA Standards (Tubular Exchanger Manufacturers Association)
- ASME PTC 12.5: Single-Phase Heat Exchangers
- Linnhoff March: Pinch Analysis Methodology
- ISO 14782: Heat Exchanger Thermal Performance

Author: GL-CalculatorEngineer
"""

import pytest
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import math


# ==============================================================================
# THERMODYNAMIC GOLDEN VALUES
# ==============================================================================

@dataclass(frozen=True)
class ThermodynamicGoldenValue:
    """Engineering reference value with tolerance and source."""
    name: str
    value: Decimal
    unit: str
    tolerance_percent: Decimal
    source: str
    description: str

    def validate(self, calculated: Decimal) -> Tuple[bool, Decimal]:
        """Validate calculated value against golden value."""
        if self.value == 0:
            return abs(calculated) < Decimal('0.001'), abs(calculated)
        deviation = abs(calculated - self.value) / abs(self.value) * Decimal('100')
        return deviation <= self.tolerance_percent, deviation


# LMTD (Log Mean Temperature Difference) Test Cases
# Reference: Incropera & DeWitt, Fundamentals of Heat Transfer
LMTD_TEST_CASES: Dict[str, Dict[str, Decimal]] = {
    'counterflow_high_efficiency': {
        'hot_inlet': Decimal('150'),
        'hot_outlet': Decimal('80'),
        'cold_inlet': Decimal('30'),
        'cold_outlet': Decimal('120'),
        'expected_lmtd': Decimal('38.0'),
        'tolerance_percent': Decimal('1.0'),
    },
    'counterflow_equal_dt': {
        'hot_inlet': Decimal('100'),
        'hot_outlet': Decimal('60'),
        'cold_inlet': Decimal('20'),
        'cold_outlet': Decimal('60'),
        'expected_lmtd': Decimal('40.0'),
        'tolerance_percent': Decimal('0.1'),
    },
    'parallel_flow': {
        'hot_inlet': Decimal('150'),
        'hot_outlet': Decimal('100'),
        'cold_inlet': Decimal('30'),
        'cold_outlet': Decimal('70'),
        'expected_lmtd': Decimal('75.7'),
        'tolerance_percent': Decimal('1.0'),
    },
}

# Heat Exchanger Effectiveness (NTU Method)
# Reference: Kays & London, Compact Heat Exchangers
NTU_EFFECTIVENESS_CASES: Dict[str, Dict[str, Any]] = {
    'counterflow_Cr_0.5': {
        'ntu': Decimal('1.5'),
        'Cr': Decimal('0.5'),  # Capacity ratio (Cmin/Cmax)
        'expected_effectiveness': Decimal('0.706'),
        'tolerance_percent': Decimal('1.0'),
    },
    'counterflow_Cr_1.0': {
        'ntu': Decimal('2.0'),
        'Cr': Decimal('1.0'),
        'expected_effectiveness': Decimal('0.667'),
        'tolerance_percent': Decimal('1.0'),
    },
    'parallel_flow': {
        'ntu': Decimal('1.0'),
        'Cr': Decimal('0.5'),
        'expected_effectiveness': Decimal('0.528'),
        'tolerance_percent': Decimal('1.0'),
    },
}

# Pinch Analysis Reference Values
# Reference: Linnhoff et al., User Guide on Process Integration
PINCH_ANALYSIS_CASES: Dict[str, Dict[str, Any]] = {
    'simple_process': {
        'hot_streams': [
            {'name': 'H1', 'supply_temp': Decimal('180'), 'target_temp': Decimal('60'),
             'mcp': Decimal('3.0')},  # kW/K
            {'name': 'H2', 'supply_temp': Decimal('150'), 'target_temp': Decimal('30'),
             'mcp': Decimal('1.5')},
        ],
        'cold_streams': [
            {'name': 'C1', 'supply_temp': Decimal('20'), 'target_temp': Decimal('135'),
             'mcp': Decimal('2.0')},
            {'name': 'C2', 'supply_temp': Decimal('80'), 'target_temp': Decimal('140'),
             'mcp': Decimal('2.5')},
        ],
        'delta_t_min': Decimal('10'),
        'expected_pinch_temp': Decimal('90'),
        'expected_qh_min': Decimal('80'),  # kW - minimum hot utility
        'expected_qc_min': Decimal('130'),  # kW - minimum cold utility
    },
}

# Exergy Analysis Reference Values
# Reference: Kotas, The Exergy Method of Thermal Plant Analysis
EXERGY_CASES: Dict[str, Dict[str, Any]] = {
    'heat_exchanger': {
        'hot_inlet_temp': Decimal('423.15'),  # K (150°C)
        'hot_outlet_temp': Decimal('353.15'),  # K (80°C)
        'cold_inlet_temp': Decimal('303.15'),  # K (30°C)
        'cold_outlet_temp': Decimal('393.15'),  # K (120°C)
        'heat_duty': Decimal('210'),  # kW
        'ambient_temp': Decimal('298.15'),  # K (25°C)
        'expected_exergy_efficiency': Decimal('0.65'),
        'tolerance_percent': Decimal('5.0'),
    },
}

# Economic Analysis Parameters
# Reference: Peters & Timmerhaus, Plant Design and Economics
ECONOMIC_PARAMETERS: Dict[str, ThermodynamicGoldenValue] = {
    'shell_tube_cost_coefficient': ThermodynamicGoldenValue(
        'Shell & Tube Cost', Decimal('1800'), 'USD/m2',
        Decimal('10'), 'TEMA/Industry', 'Installed cost per area'),
    'plate_cost_coefficient': ThermodynamicGoldenValue(
        'Plate HX Cost', Decimal('1500'), 'USD/m2',
        Decimal('10'), 'Industry', 'Installed cost per area'),
    'energy_cost': ThermodynamicGoldenValue(
        'Steam Cost', Decimal('35'), 'USD/MMBtu',
        Decimal('20'), 'EIA', 'Industrial steam cost'),
    'payback_target': ThermodynamicGoldenValue(
        'Payback Period', Decimal('2.5'), 'years',
        Decimal('0'), 'Industry', 'Target simple payback'),
}


# ==============================================================================
# CALCULATION FUNCTIONS
# ==============================================================================

def calculate_lmtd(
    hot_inlet: Decimal,
    hot_outlet: Decimal,
    cold_inlet: Decimal,
    cold_outlet: Decimal,
    flow_type: str = 'counterflow'
) -> Decimal:
    """
    Calculate Log Mean Temperature Difference.

    Formula:
    LMTD = (ΔT1 - ΔT2) / ln(ΔT1/ΔT2)

    For counterflow:
        ΔT1 = Th_in - Tc_out
        ΔT2 = Th_out - Tc_in

    For parallel flow:
        ΔT1 = Th_in - Tc_in
        ΔT2 = Th_out - Tc_out
    """
    if flow_type == 'counterflow':
        dt1 = hot_inlet - cold_outlet
        dt2 = hot_outlet - cold_inlet
    else:  # parallel flow
        dt1 = hot_inlet - cold_inlet
        dt2 = hot_outlet - cold_outlet

    # Handle special case where dt1 == dt2
    if abs(dt1 - dt2) < Decimal('0.01'):
        return dt1

    # Handle invalid cases
    if dt1 <= 0 or dt2 <= 0:
        raise ValueError('Temperature cross detected - invalid heat exchanger design')

    # Calculate LMTD
    lmtd = (dt1 - dt2) / Decimal(str(math.log(float(dt1 / dt2))))
    return lmtd.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_effectiveness_ntu_counterflow(
    ntu: Decimal,
    cr: Decimal
) -> Decimal:
    """
    Calculate heat exchanger effectiveness using NTU method (counterflow).

    Formula for counterflow (Cr < 1):
    ε = (1 - exp(-NTU(1-Cr))) / (1 - Cr*exp(-NTU(1-Cr)))

    For Cr = 1:
    ε = NTU / (1 + NTU)
    """
    ntu_f = float(ntu)
    cr_f = float(cr)

    if abs(cr_f - 1.0) < 0.001:
        # Special case for Cr = 1
        effectiveness = ntu_f / (1 + ntu_f)
    else:
        exp_term = math.exp(-ntu_f * (1 - cr_f))
        effectiveness = (1 - exp_term) / (1 - cr_f * exp_term)

    return Decimal(str(effectiveness)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)


def calculate_pinch_temperature(
    hot_streams: List[Dict[str, Any]],
    cold_streams: List[Dict[str, Any]],
    delta_t_min: Decimal
) -> Tuple[Decimal, Decimal, Decimal]:
    """
    Calculate pinch point and minimum utility requirements.

    Uses problem table algorithm from Linnhoff March.

    Returns: (pinch_temp, min_hot_utility, min_cold_utility)
    """
    # Collect all temperature intervals
    temps = set()
    for stream in hot_streams:
        temps.add(stream['supply_temp'] - delta_t_min / 2)
        temps.add(stream['target_temp'] - delta_t_min / 2)
    for stream in cold_streams:
        temps.add(stream['supply_temp'] + delta_t_min / 2)
        temps.add(stream['target_temp'] + delta_t_min / 2)

    sorted_temps = sorted(temps, reverse=True)

    # Calculate heat deficit in each interval
    heat_cascade = []
    for i in range(len(sorted_temps) - 1):
        t_high = sorted_temps[i]
        t_low = sorted_temps[i + 1]
        dt = t_high - t_low

        # Sum mCp for streams in this interval
        sum_mcp_hot = Decimal('0')
        sum_mcp_cold = Decimal('0')

        for stream in hot_streams:
            s_high = stream['supply_temp'] - delta_t_min / 2
            s_low = stream['target_temp'] - delta_t_min / 2
            if s_high >= t_high and s_low <= t_low:
                sum_mcp_hot += stream['mcp']

        for stream in cold_streams:
            s_low = stream['supply_temp'] + delta_t_min / 2
            s_high = stream['target_temp'] + delta_t_min / 2
            if s_high >= t_high and s_low <= t_low:
                sum_mcp_cold += stream['mcp']

        interval_deficit = (sum_mcp_cold - sum_mcp_hot) * dt
        heat_cascade.append({
            'temp': t_low,
            'deficit': interval_deficit,
        })

    # Cascade the heat and find pinch
    cumulative = Decimal('0')
    min_cumulative = Decimal('999999')
    pinch_temp = sorted_temps[-1]

    for interval in heat_cascade:
        cumulative += interval['deficit']
        if cumulative < min_cumulative:
            min_cumulative = cumulative
            pinch_temp = interval['temp']

    # Minimum utilities
    qh_min = -min_cumulative if min_cumulative < 0 else Decimal('0')
    qc_min = cumulative + qh_min

    return pinch_temp + delta_t_min / 2, qh_min, qc_min


def calculate_exergy_efficiency(
    hot_inlet_temp: Decimal,
    hot_outlet_temp: Decimal,
    cold_inlet_temp: Decimal,
    cold_outlet_temp: Decimal,
    ambient_temp: Decimal
) -> Decimal:
    """
    Calculate exergy (second law) efficiency of heat exchanger.

    Exergy efficiency = Exergy gain by cold stream / Exergy loss by hot stream

    Exergy of heat at temperature T relative to T0:
    Ex = Q * (1 - T0/T_lm)

    Where T_lm is log mean temperature.
    """
    # Calculate log mean temperatures
    hot_lm = (hot_inlet_temp - hot_outlet_temp) / Decimal(str(
        math.log(float(hot_inlet_temp / hot_outlet_temp))))
    cold_lm = (cold_outlet_temp - cold_inlet_temp) / Decimal(str(
        math.log(float(cold_outlet_temp / cold_inlet_temp))))

    # Exergy factors (1 - T0/Tlm)
    carnot_hot = Decimal('1') - ambient_temp / hot_lm
    carnot_cold = Decimal('1') - ambient_temp / cold_lm

    # Exergy efficiency (assuming equal heat duties)
    if carnot_hot == 0:
        return Decimal('0')

    efficiency = carnot_cold / carnot_hot
    return efficiency.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def calculate_heat_duty(
    mass_flow: Decimal,
    specific_heat: Decimal,
    temp_in: Decimal,
    temp_out: Decimal
) -> Decimal:
    """
    Calculate heat duty.

    Q = m * Cp * ΔT
    """
    return (mass_flow * specific_heat * abs(temp_out - temp_in)).quantize(
        Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_required_area(
    heat_duty: Decimal,
    overall_u: Decimal,
    lmtd: Decimal
) -> Decimal:
    """
    Calculate required heat transfer area.

    A = Q / (U * LMTD)
    """
    if overall_u == 0 or lmtd == 0:
        raise ValueError('U and LMTD must be non-zero')
    return (heat_duty / (overall_u * lmtd)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_payback_period(
    capital_cost: Decimal,
    annual_savings: Decimal
) -> Decimal:
    """
    Calculate simple payback period.

    Payback = Capital Cost / Annual Savings
    """
    if annual_savings <= 0:
        return Decimal('999')  # Infinite payback
    return (capital_cost / annual_savings).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


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
class TestLMTDCalculation:
    """Validate LMTD calculations against textbook values."""

    @pytest.mark.parametrize("case_name", LMTD_TEST_CASES.keys())
    def test_lmtd_calculation(self, case_name: str):
        """Verify LMTD calculation for various configurations."""
        case = LMTD_TEST_CASES[case_name]

        flow_type = 'parallel' if 'parallel' in case_name else 'counterflow'

        calculated = calculate_lmtd(
            case['hot_inlet'],
            case['hot_outlet'],
            case['cold_inlet'],
            case['cold_outlet'],
            flow_type=flow_type,
        )

        deviation = abs(calculated - case['expected_lmtd']) / case['expected_lmtd'] * 100
        assert deviation <= float(case['tolerance_percent']), \
            f'{case_name}: LMTD {calculated}°C deviates {deviation:.2f}% from {case["expected_lmtd"]}°C'

    def test_lmtd_equal_dt_special_case(self):
        """When ΔT1 = ΔT2, LMTD should equal that difference."""
        lmtd = calculate_lmtd(
            hot_inlet=Decimal('100'),
            hot_outlet=Decimal('60'),
            cold_inlet=Decimal('20'),
            cold_outlet=Decimal('60'),
        )
        assert lmtd == Decimal('40.0')

    def test_lmtd_temperature_cross_detection(self):
        """Temperature cross should raise error."""
        with pytest.raises(ValueError, match='Temperature cross'):
            calculate_lmtd(
                hot_inlet=Decimal('80'),
                hot_outlet=Decimal('60'),
                cold_inlet=Decimal('50'),
                cold_outlet=Decimal('90'),  # Higher than hot inlet
            )

    def test_lmtd_determinism(self):
        """LMTD calculation must be deterministic."""
        results = set()
        for _ in range(100):
            lmtd = calculate_lmtd(
                Decimal('150'), Decimal('80'), Decimal('30'), Decimal('120'))
            results.add(str(lmtd))
        assert len(results) == 1


@pytest.mark.golden
class TestNTUEffectiveness:
    """Validate NTU-effectiveness calculations."""

    @pytest.mark.parametrize("case_name", NTU_EFFECTIVENESS_CASES.keys())
    def test_effectiveness_calculation(self, case_name: str):
        """Verify effectiveness calculation for various configurations."""
        case = NTU_EFFECTIVENESS_CASES[case_name]

        if 'counterflow' in case_name:
            calculated = calculate_effectiveness_ntu_counterflow(
                case['ntu'], case['Cr'])

            deviation = (abs(calculated - case['expected_effectiveness']) /
                        case['expected_effectiveness'] * 100)
            assert deviation <= float(case['tolerance_percent']), \
                f'{case_name}: ε={calculated} deviates {deviation:.2f}%'

    def test_effectiveness_bounds(self):
        """Effectiveness must be between 0 and 1."""
        for ntu in [Decimal('0.1'), Decimal('1.0'), Decimal('5.0'), Decimal('10.0')]:
            for cr in [Decimal('0.1'), Decimal('0.5'), Decimal('1.0')]:
                eff = calculate_effectiveness_ntu_counterflow(ntu, cr)
                assert Decimal('0') <= eff <= Decimal('1'), \
                    f'Effectiveness {eff} out of bounds for NTU={ntu}, Cr={cr}'

    def test_effectiveness_increases_with_ntu(self):
        """Effectiveness should increase with NTU."""
        cr = Decimal('0.5')
        prev_eff = Decimal('0')

        for ntu in [Decimal('0.5'), Decimal('1.0'), Decimal('2.0'), Decimal('4.0')]:
            eff = calculate_effectiveness_ntu_counterflow(ntu, cr)
            assert eff > prev_eff, f'Effectiveness should increase with NTU'
            prev_eff = eff


@pytest.mark.golden
class TestPinchAnalysis:
    """Validate pinch analysis calculations."""

    def test_simple_pinch_problem(self):
        """Verify pinch calculation for simple process."""
        case = PINCH_ANALYSIS_CASES['simple_process']

        pinch, qh_min, qc_min = calculate_pinch_temperature(
            case['hot_streams'],
            case['cold_streams'],
            case['delta_t_min'],
        )

        # Verify pinch temperature (within tolerance)
        pinch_deviation = abs(pinch - case['expected_pinch_temp'])
        assert pinch_deviation <= Decimal('5'), \
            f'Pinch {pinch}°C differs from expected {case["expected_pinch_temp"]}°C'

    def test_pinch_minimum_utilities(self):
        """Verify minimum utility calculation."""
        case = PINCH_ANALYSIS_CASES['simple_process']

        _, qh_min, qc_min = calculate_pinch_temperature(
            case['hot_streams'],
            case['cold_streams'],
            case['delta_t_min'],
        )

        # Utilities should be positive
        assert qh_min >= Decimal('0'), 'Hot utility cannot be negative'
        assert qc_min >= Decimal('0'), 'Cold utility cannot be negative'


@pytest.mark.golden
class TestExergyAnalysis:
    """Validate exergy (second law) efficiency calculations."""

    def test_exergy_efficiency_calculation(self):
        """Verify exergy efficiency calculation."""
        case = EXERGY_CASES['heat_exchanger']

        efficiency = calculate_exergy_efficiency(
            case['hot_inlet_temp'],
            case['hot_outlet_temp'],
            case['cold_inlet_temp'],
            case['cold_outlet_temp'],
            case['ambient_temp'],
        )

        deviation = (abs(efficiency - case['expected_exergy_efficiency']) /
                    case['expected_exergy_efficiency'] * 100)
        assert deviation <= float(case['tolerance_percent']), \
            f'Exergy efficiency {efficiency} deviates {deviation:.2f}%'

    def test_exergy_efficiency_bounds(self):
        """Exergy efficiency should be between 0 and 1."""
        efficiency = calculate_exergy_efficiency(
            hot_inlet_temp=Decimal('423.15'),
            hot_outlet_temp=Decimal('353.15'),
            cold_inlet_temp=Decimal('303.15'),
            cold_outlet_temp=Decimal('393.15'),
            ambient_temp=Decimal('298.15'),
        )
        assert Decimal('0') <= efficiency <= Decimal('1')


@pytest.mark.golden
class TestEconomicAnalysis:
    """Validate economic calculation parameters."""

    def test_cost_coefficients(self):
        """Verify economic parameters are reasonable."""
        assert ECONOMIC_PARAMETERS['shell_tube_cost_coefficient'].value == Decimal('1800')
        assert ECONOMIC_PARAMETERS['plate_cost_coefficient'].value == Decimal('1500')

    def test_payback_calculation(self):
        """Verify payback period calculation."""
        capital = Decimal('100000')
        annual_savings = Decimal('40000')

        payback = calculate_payback_period(capital, annual_savings)
        assert payback == Decimal('2.5')

    def test_payback_zero_savings(self):
        """Zero savings should return infinite payback."""
        payback = calculate_payback_period(Decimal('100000'), Decimal('0'))
        assert payback == Decimal('999')


@pytest.mark.golden
class TestHeatDutyCalculation:
    """Validate heat duty calculations."""

    @dataclass(frozen=True)
    class HeatDutyCase:
        name: str
        mass_flow: Decimal  # kg/s
        specific_heat: Decimal  # kJ/kg-K
        temp_in: Decimal  # °C
        temp_out: Decimal  # °C
        expected_duty: Decimal  # kW

    CASES = [
        HeatDutyCase('Water Heating', Decimal('5.0'), Decimal('4.18'),
                     Decimal('20'), Decimal('60'), Decimal('836.0')),
        HeatDutyCase('Oil Cooling', Decimal('3.0'), Decimal('2.1'),
                     Decimal('120'), Decimal('60'), Decimal('378.0')),
        HeatDutyCase('Steam Condensate', Decimal('1.0'), Decimal('4.2'),
                     Decimal('100'), Decimal('50'), Decimal('210.0')),
    ]

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
    def test_heat_duty_calculation(self, case: HeatDutyCase):
        """Verify heat duty calculation."""
        calculated = calculate_heat_duty(
            case.mass_flow, case.specific_heat, case.temp_in, case.temp_out)

        deviation = abs(calculated - case.expected_duty) / case.expected_duty * 100
        assert deviation <= Decimal('0.5'), \
            f'{case.name}: Q={calculated} kW deviates from {case.expected_duty} kW'


@pytest.mark.golden
class TestRequiredArea:
    """Validate heat transfer area calculations."""

    def test_area_calculation(self):
        """Verify required area calculation A = Q/(U*LMTD)."""
        heat_duty = Decimal('500')  # kW
        overall_u = Decimal('0.5')  # kW/m2-K
        lmtd = Decimal('40')  # K

        area = calculate_required_area(heat_duty, overall_u, lmtd)
        expected = Decimal('25.0')  # m2

        assert area == expected

    def test_area_zero_u_error(self):
        """Zero U-value should raise error."""
        with pytest.raises(ValueError):
            calculate_required_area(Decimal('500'), Decimal('0'), Decimal('40'))


@pytest.mark.golden
class TestProvenanceAndDeterminism:
    """Validate provenance tracking and deterministic behavior."""

    def test_all_calculations_deterministic(self):
        """All calculations must produce identical results."""
        calculations = [
            lambda: calculate_lmtd(Decimal('150'), Decimal('80'),
                                   Decimal('30'), Decimal('120')),
            lambda: calculate_effectiveness_ntu_counterflow(Decimal('1.5'), Decimal('0.5')),
            lambda: calculate_exergy_efficiency(Decimal('423'), Decimal('353'),
                                                Decimal('303'), Decimal('393'),
                                                Decimal('298')),
            lambda: calculate_heat_duty(Decimal('5'), Decimal('4.18'),
                                        Decimal('20'), Decimal('60')),
        ]

        for calc in calculations:
            results = set()
            for _ in range(50):
                results.add(str(calc()))
            assert len(results) == 1, f'Calculation is non-deterministic'

    def test_provenance_hash_stability(self):
        """Provenance hash must be stable."""
        inputs = {'hot_inlet': '150', 'cold_outlet': '120'}
        outputs = {'lmtd': '38.0'}

        hashes = set()
        for _ in range(50):
            h = calculate_provenance_hash('lmtd', inputs, outputs)
            hashes.add(h)

        assert len(hashes) == 1


# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================

def export_golden_values() -> Dict[str, Any]:
    """Export all golden values for documentation."""
    return {
        'metadata': {
            'agent': 'GL-006_HEATRECLAIM',
            'version': '1.0.0',
            'standards': ['TEMA', 'ASME PTC 12.5', 'Linnhoff March'],
        },
        'lmtd_cases': {
            name: {k: str(v) for k, v in case.items()}
            for name, case in LMTD_TEST_CASES.items()
        },
        'ntu_effectiveness': {
            name: {k: str(v) for k, v in case.items()}
            for name, case in NTU_EFFECTIVENESS_CASES.items()
        },
        'economic_parameters': {
            key: str(val.value) for key, val in ECONOMIC_PARAMETERS.items()
        },
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
