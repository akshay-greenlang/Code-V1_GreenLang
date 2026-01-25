# -*- coding: utf-8 -*-
"""
Steam Trap Validation Golden Value Tests

Comprehensive test suite validating TrapCatcher agent calculations against
industry standards for steam trap performance and energy loss assessment.

Reference Documents:
- DOE Steam Tip Sheets (Office of Energy Efficiency)
- Spirax Sarco Steam Engineering Guides
- Armstrong International Steam Conservation Guidelines
- ASME B16.34 Valves

Author: GL-CalculatorEngineer
"""

import pytest
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum


# ==============================================================================
# STEAM TRAP GOLDEN VALUE REFERENCE DATA
# ==============================================================================

@dataclass(frozen=True)
class SteamTrapGoldenValue:
    """Industry reference value for steam trap calculations."""
    name: str
    value: Decimal
    unit: str
    tolerance_percent: Decimal
    source: str
    description: str

    def validate(self, calculated: Decimal) -> Tuple[bool, Decimal]:
        """Validate calculated value."""
        if self.value == 0:
            return abs(calculated) < Decimal('0.001'), abs(calculated)
        deviation = abs(calculated - self.value) / abs(self.value) * Decimal('100')
        return deviation <= self.tolerance_percent, deviation


class TrapType(Enum):
    """Steam trap types."""
    THERMOSTATIC = 'thermostatic'
    THERMODYNAMIC = 'thermodynamic'
    FLOAT_THERMOSTATIC = 'float_thermostatic'
    INVERTED_BUCKET = 'inverted_bucket'
    BIMETALLIC = 'bimetallic'


class TrapCondition(Enum):
    """Steam trap condition states."""
    GOOD = 'good'  # Operating correctly
    BLOWING_THROUGH = 'blowing_through'  # Passing live steam
    COLD = 'cold'  # Blocked or failed closed
    LEAKING = 'leaking'  # Partial steam loss
    UNKNOWN = 'unknown'  # Cannot determine


# Steam latent heat at various pressures (Btu/lb)
# Reference: ASME Steam Tables
STEAM_LATENT_HEAT: Dict[int, Decimal] = {
    15: Decimal('945'),   # psig
    50: Decimal('912'),
    100: Decimal('880'),
    150: Decimal('857'),
    200: Decimal('837'),
    250: Decimal('820'),
    300: Decimal('802'),
}

# Steam specific volume at saturation (ft3/lb)
STEAM_SPECIFIC_VOLUME: Dict[int, Decimal] = {
    15: Decimal('13.75'),   # psig
    50: Decimal('6.65'),
    100: Decimal('3.88'),
    150: Decimal('2.75'),
    200: Decimal('2.13'),
    250: Decimal('1.74'),
    300: Decimal('1.47'),
}

# Trap failure rates by type (failures per year)
# Reference: DOE Industrial Steam Assessment Manual
TRAP_FAILURE_RATES: Dict[str, SteamTrapGoldenValue] = {
    'thermodynamic': SteamTrapGoldenValue(
        'Thermodynamic Failure Rate', Decimal('0.15'), 'per year',
        Decimal('20'), 'DOE', 'Annual failure probability'),
    'float_thermostatic': SteamTrapGoldenValue(
        'F&T Failure Rate', Decimal('0.10'), 'per year',
        Decimal('20'), 'DOE', 'Annual failure probability'),
    'inverted_bucket': SteamTrapGoldenValue(
        'Inverted Bucket Failure Rate', Decimal('0.05'), 'per year',
        Decimal('20'), 'DOE', 'Annual failure probability'),
    'thermostatic': SteamTrapGoldenValue(
        'Thermostatic Failure Rate', Decimal('0.12'), 'per year',
        Decimal('20'), 'DOE', 'Annual failure probability'),
}

# Orifice coefficient for steam loss calculation
ORIFICE_COEFFICIENT = Decimal('0.7')  # Typical for worn traps

# Steam cost parameters
STEAM_COST_PER_KLBS: Dict[str, SteamTrapGoldenValue] = {
    'natural_gas': SteamTrapGoldenValue(
        'NG Steam Cost', Decimal('8.50'), 'USD/klbs',
        Decimal('30'), 'EIA', 'Steam generation cost with natural gas'),
    'fuel_oil': SteamTrapGoldenValue(
        'Oil Steam Cost', Decimal('12.00'), 'USD/klbs',
        Decimal('30'), 'EIA', 'Steam generation cost with fuel oil'),
}


# ==============================================================================
# CALCULATION FUNCTIONS
# ==============================================================================

def calculate_steam_loss_orifice(
    orifice_diameter_inches: Decimal,
    steam_pressure_psig: Decimal,
    orifice_coefficient: Decimal = ORIFICE_COEFFICIENT
) -> Decimal:
    """
    Calculate steam loss through orifice (lb/hr).

    Formula from DOE Steam Tip Sheet #1:
    Steam Loss = C × d² × P / 1000

    Where:
    - C = orifice coefficient × 24.24
    - d = orifice diameter (inches)
    - P = steam pressure (psig + 14.7)

    Returns: Steam loss in lb/hr
    """
    c = orifice_coefficient * Decimal('24.24')
    p_abs = steam_pressure_psig + Decimal('14.7')
    d_squared = orifice_diameter_inches ** 2

    steam_loss = c * d_squared * p_abs
    return steam_loss.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_energy_loss(
    steam_loss_lb_hr: Decimal,
    steam_pressure_psig: Decimal,
    operating_hours: Decimal = Decimal('8760')
) -> Decimal:
    """
    Calculate annual energy loss from steam trap failure (MMBtu/yr).

    Energy Loss = Steam Loss × Latent Heat × Hours / 1,000,000

    Returns: Annual energy loss in MMBtu/yr
    """
    # Get latent heat (interpolate if needed)
    pressure = int(steam_pressure_psig)
    pressures = sorted(STEAM_LATENT_HEAT.keys())

    if pressure <= pressures[0]:
        latent_heat = STEAM_LATENT_HEAT[pressures[0]]
    elif pressure >= pressures[-1]:
        latent_heat = STEAM_LATENT_HEAT[pressures[-1]]
    else:
        # Linear interpolation
        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                h1, h2 = STEAM_LATENT_HEAT[p1], STEAM_LATENT_HEAT[p2]
                latent_heat = h1 + (h2 - h1) * (pressure - p1) / (p2 - p1)
                break
        else:
            latent_heat = Decimal('880')  # Default

    energy_loss = steam_loss_lb_hr * latent_heat * operating_hours / Decimal('1000000')
    return energy_loss.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_annual_cost_savings(
    energy_loss_mmbtu: Decimal,
    steam_cost_per_mmbtu: Decimal
) -> Decimal:
    """
    Calculate annual cost savings from repairing failed trap.

    Savings = Energy Loss × Steam Cost

    Returns: Annual savings in USD
    """
    savings = energy_loss_mmbtu * steam_cost_per_mmbtu
    return savings.quantize(Decimal('1'), rounding=ROUND_HALF_UP)


def calculate_trap_discharge_capacity(
    trap_cv: Decimal,
    inlet_pressure_psig: Decimal,
    back_pressure_psig: Decimal
) -> Decimal:
    """
    Calculate steam trap condensate discharge capacity (lb/hr).

    Formula: Capacity = Cv × 500 × sqrt(dP)

    Where dP = inlet pressure - back pressure

    Returns: Condensate capacity in lb/hr
    """
    dp = inlet_pressure_psig - back_pressure_psig
    if dp <= 0:
        return Decimal('0')

    capacity = trap_cv * Decimal('500') * Decimal(str(dp ** Decimal('0.5')))
    return capacity.quantize(Decimal('1'), rounding=ROUND_HALF_UP)


def diagnose_trap_condition(
    upstream_temp_f: Decimal,
    downstream_temp_f: Decimal,
    saturation_temp_f: Decimal,
    subcool_threshold_f: Decimal = Decimal('30')
) -> TrapCondition:
    """
    Diagnose trap condition from temperature measurements.

    Diagnosis rules:
    - BLOWING_THROUGH: Downstream temp near saturation
    - COLD: Both temps well below saturation
    - LEAKING: Downstream slightly below saturation
    - GOOD: Appropriate subcooling observed

    Returns: TrapCondition diagnosis
    """
    upstream_delta = saturation_temp_f - upstream_temp_f
    downstream_delta = saturation_temp_f - downstream_temp_f

    if downstream_delta < Decimal('5'):
        return TrapCondition.BLOWING_THROUGH
    elif upstream_delta > Decimal('50') and downstream_delta > Decimal('50'):
        return TrapCondition.COLD
    elif downstream_delta < Decimal('20'):
        return TrapCondition.LEAKING
    elif downstream_delta >= subcool_threshold_f:
        return TrapCondition.GOOD
    else:
        return TrapCondition.UNKNOWN


def calculate_survey_roi(
    num_traps: int,
    estimated_failure_rate: Decimal,
    avg_pressure_psig: Decimal,
    avg_orifice_dia: Decimal,
    steam_cost_per_mmbtu: Decimal,
    survey_cost: Decimal,
    repair_cost_per_trap: Decimal
) -> Tuple[Decimal, Decimal, Decimal]:
    """
    Calculate ROI for steam trap survey program.

    Returns: (annual_savings, simple_payback_months, roi_percent)
    """
    # Estimate failed traps
    failed_traps = Decimal(num_traps) * estimated_failure_rate

    # Steam loss per failed trap
    steam_loss = calculate_steam_loss_orifice(avg_orifice_dia, avg_pressure_psig)
    energy_loss = calculate_energy_loss(steam_loss, avg_pressure_psig)

    # Total savings
    total_energy_loss = energy_loss * failed_traps
    annual_savings = total_energy_loss * steam_cost_per_mmbtu

    # Total cost (survey + repairs)
    total_cost = survey_cost + (failed_traps * repair_cost_per_trap)

    # Simple payback
    if annual_savings > 0:
        payback_months = (total_cost / annual_savings) * Decimal('12')
    else:
        payback_months = Decimal('999')

    # ROI
    if total_cost > 0:
        roi = ((annual_savings - total_cost) / total_cost) * Decimal('100')
    else:
        roi = Decimal('0')

    return (
        annual_savings.quantize(Decimal('1'), rounding=ROUND_HALF_UP),
        payback_months.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
        roi.quantize(Decimal('1'), rounding=ROUND_HALF_UP),
    )


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
class TestSteamProperties:
    """Validate steam property reference values."""

    @pytest.mark.parametrize("pressure,expected_latent", [
        (15, Decimal('945')),
        (100, Decimal('880')),
        (200, Decimal('837')),
    ])
    def test_latent_heat_values(self, pressure: int, expected_latent: Decimal):
        """Verify latent heat values from steam tables."""
        assert STEAM_LATENT_HEAT[pressure] == expected_latent

    @pytest.mark.parametrize("pressure,expected_volume", [
        (15, Decimal('13.75')),
        (100, Decimal('3.88')),
        (200, Decimal('2.13')),
    ])
    def test_specific_volume_values(self, pressure: int, expected_volume: Decimal):
        """Verify specific volume values from steam tables."""
        assert STEAM_SPECIFIC_VOLUME[pressure] == expected_volume


@pytest.mark.golden
class TestSteamLossCalculation:
    """Validate steam loss through orifice calculations."""

    @dataclass(frozen=True)
    class SteamLossTestCase:
        name: str
        orifice_dia: Decimal
        pressure_psig: Decimal
        expected_loss: Decimal  # lb/hr
        tolerance_percent: Decimal

    TEST_CASES = [
        SteamLossTestCase('1/8" at 100 psig', Decimal('0.125'),
                          Decimal('100'), Decimal('48'), Decimal('10')),
        SteamLossTestCase('1/4" at 100 psig', Decimal('0.25'),
                          Decimal('100'), Decimal('193'), Decimal('10')),
        SteamLossTestCase('1/4" at 150 psig', Decimal('0.25'),
                          Decimal('150'), Decimal('277'), Decimal('10')),
    ]

    @pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
    def test_steam_loss_calculation(self, case: SteamLossTestCase):
        """Verify steam loss calculation per DOE methodology."""
        calculated = calculate_steam_loss_orifice(
            case.orifice_dia, case.pressure_psig)

        deviation = abs(calculated - case.expected_loss) / case.expected_loss * 100
        assert deviation <= float(case.tolerance_percent), \
            f'{case.name}: {calculated} lb/hr deviates {deviation:.1f}%'

    def test_steam_loss_increases_with_pressure(self):
        """Steam loss should increase with pressure."""
        loss_low = calculate_steam_loss_orifice(Decimal('0.25'), Decimal('50'))
        loss_high = calculate_steam_loss_orifice(Decimal('0.25'), Decimal('150'))
        assert loss_high > loss_low

    def test_steam_loss_increases_with_orifice(self):
        """Steam loss should increase with orifice size."""
        loss_small = calculate_steam_loss_orifice(Decimal('0.125'), Decimal('100'))
        loss_large = calculate_steam_loss_orifice(Decimal('0.25'), Decimal('100'))
        assert loss_large > loss_small


@pytest.mark.golden
class TestEnergyLossCalculation:
    """Validate annual energy loss calculations."""

    def test_energy_loss_calculation(self):
        """Verify energy loss calculation."""
        steam_loss = Decimal('100')  # lb/hr
        pressure = Decimal('100')  # psig

        energy = calculate_energy_loss(steam_loss, pressure)

        # 100 lb/hr × 880 Btu/lb × 8760 hr/yr / 1,000,000
        expected = Decimal('770.9')
        deviation = abs(energy - expected) / expected * 100
        assert deviation <= Decimal('5')

    def test_energy_loss_proportional_to_steam(self):
        """Energy loss should be proportional to steam loss."""
        e1 = calculate_energy_loss(Decimal('100'), Decimal('100'))
        e2 = calculate_energy_loss(Decimal('200'), Decimal('100'))

        # Should be approximately 2x
        ratio = e2 / e1
        assert Decimal('1.9') <= ratio <= Decimal('2.1')


@pytest.mark.golden
class TestTrapDiagnosis:
    """Validate steam trap diagnosis logic."""

    @dataclass(frozen=True)
    class DiagnosisTestCase:
        name: str
        upstream_temp: Decimal
        downstream_temp: Decimal
        saturation_temp: Decimal
        expected_condition: TrapCondition

    TEST_CASES = [
        DiagnosisTestCase('Blowing Through', Decimal('350'), Decimal('348'),
                          Decimal('350'), TrapCondition.BLOWING_THROUGH),
        DiagnosisTestCase('Good Trap', Decimal('340'), Decimal('280'),
                          Decimal('350'), TrapCondition.GOOD),
        DiagnosisTestCase('Cold/Blocked', Decimal('200'), Decimal('180'),
                          Decimal('350'), TrapCondition.COLD),
        DiagnosisTestCase('Leaking', Decimal('340'), Decimal('335'),
                          Decimal('350'), TrapCondition.LEAKING),
    ]

    @pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
    def test_trap_diagnosis(self, case: DiagnosisTestCase):
        """Verify trap condition diagnosis."""
        result = diagnose_trap_condition(
            case.upstream_temp, case.downstream_temp, case.saturation_temp)
        assert result == case.expected_condition, \
            f'{case.name}: Expected {case.expected_condition}, got {result}'


@pytest.mark.golden
class TestTrapFailureRates:
    """Validate trap failure rate reference values."""

    @pytest.mark.parametrize("trap_type,max_rate", [
        ('thermodynamic', Decimal('0.20')),
        ('float_thermostatic', Decimal('0.15')),
        ('inverted_bucket', Decimal('0.10')),
    ])
    def test_failure_rates_reasonable(self, trap_type: str, max_rate: Decimal):
        """Failure rates should be within reasonable bounds."""
        rate = TRAP_FAILURE_RATES[trap_type].value
        assert rate <= max_rate, \
            f'{trap_type} failure rate {rate} exceeds expected maximum'

    def test_inverted_bucket_most_reliable(self):
        """Inverted bucket should have lowest failure rate."""
        ib_rate = TRAP_FAILURE_RATES['inverted_bucket'].value
        for trap_type, golden in TRAP_FAILURE_RATES.items():
            if trap_type != 'inverted_bucket':
                assert golden.value >= ib_rate


@pytest.mark.golden
class TestDischargeCapacity:
    """Validate condensate discharge capacity calculations."""

    def test_capacity_calculation(self):
        """Verify discharge capacity calculation."""
        capacity = calculate_trap_discharge_capacity(
            trap_cv=Decimal('1.0'),
            inlet_pressure_psig=Decimal('100'),
            back_pressure_psig=Decimal('0'),
        )

        # Cv × 500 × sqrt(100) = 1.0 × 500 × 10 = 5000 lb/hr
        expected = Decimal('5000')
        deviation = abs(capacity - expected) / expected * 100
        assert deviation <= Decimal('1')

    def test_no_flow_at_equal_pressure(self):
        """No flow when inlet equals back pressure."""
        capacity = calculate_trap_discharge_capacity(
            trap_cv=Decimal('1.0'),
            inlet_pressure_psig=Decimal('100'),
            back_pressure_psig=Decimal('100'),
        )
        assert capacity == Decimal('0')


@pytest.mark.golden
class TestSurveyROI:
    """Validate survey ROI calculations."""

    def test_roi_calculation(self):
        """Verify ROI calculation for trap survey."""
        savings, payback, roi = calculate_survey_roi(
            num_traps=100,
            estimated_failure_rate=Decimal('0.15'),
            avg_pressure_psig=Decimal('100'),
            avg_orifice_dia=Decimal('0.1875'),
            steam_cost_per_mmbtu=Decimal('10'),
            survey_cost=Decimal('5000'),
            repair_cost_per_trap=Decimal('200'),
        )

        # Verify savings are positive
        assert savings > Decimal('0'), 'Annual savings should be positive'

        # Verify payback is reasonable (< 24 months typically)
        assert payback < Decimal('24'), \
            f'Payback {payback} months seems too long'


@pytest.mark.golden
class TestProvenanceAndDeterminism:
    """Validate provenance tracking and deterministic behavior."""

    def test_all_calculations_deterministic(self):
        """All calculations must be deterministic."""
        calculations = [
            lambda: calculate_steam_loss_orifice(Decimal('0.25'), Decimal('100')),
            lambda: calculate_energy_loss(Decimal('100'), Decimal('100')),
            lambda: calculate_trap_discharge_capacity(Decimal('1'), Decimal('100'), Decimal('0')),
        ]

        for calc in calculations:
            results = set()
            for _ in range(50):
                results.add(str(calc()))
            assert len(results) == 1

    def test_provenance_hash_stability(self):
        """Provenance hash must be stable."""
        inputs = {'orifice': '0.25', 'pressure': '100'}
        outputs = {'steam_loss': '193'}

        hashes = set()
        for _ in range(50):
            h = calculate_provenance_hash('steam_loss', inputs, outputs)
            hashes.add(h)

        assert len(hashes) == 1


# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================

def export_golden_values() -> Dict[str, Any]:
    """Export all golden values for documentation."""
    return {
        'metadata': {
            'agent': 'GL-008_Trapcatcher',
            'version': '1.0.0',
            'sources': ['DOE', 'Spirax Sarco', 'ASME Steam Tables'],
        },
        'latent_heat': {str(k): str(v) for k, v in STEAM_LATENT_HEAT.items()},
        'failure_rates': {
            k: str(v.value) for k, v in TRAP_FAILURE_RATES.items()
        },
        'steam_costs': {
            k: str(v.value) for k, v in STEAM_COST_PER_KLBS.items()
        },
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
