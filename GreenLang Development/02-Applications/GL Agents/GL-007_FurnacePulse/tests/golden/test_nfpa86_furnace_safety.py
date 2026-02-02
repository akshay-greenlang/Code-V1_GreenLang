# -*- coding: utf-8 -*-
"""
NFPA 86 Furnace Safety Golden Value Tests

Comprehensive test suite validating FurnacePulse agent calculations against
NFPA 86 (Standard for Ovens and Furnaces) safety requirements.

Reference Documents:
- NFPA 86-2023: Standard for Ovens and Furnaces
- FM Global Data Sheet 6-9: Industrial Ovens and Dryers
- API 560: Fired Heaters for Petroleum Refineries
- ASME PTC 4.3: Air Heaters

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
# NFPA 86 GOLDEN VALUE REFERENCE DATA
# ==============================================================================

@dataclass(frozen=True)
class NFPA86GoldenValue:
    """NFPA 86 regulatory reference value with citation."""
    name: str
    value: Decimal
    unit: str
    tolerance_percent: Decimal
    nfpa_section: str
    description: str

    def validate(self, calculated: Decimal) -> Tuple[bool, Decimal]:
        """Validate calculated value against golden value."""
        if self.value == 0:
            return abs(calculated) < Decimal('0.001'), abs(calculated)
        deviation = abs(calculated - self.value) / abs(self.value) * Decimal('100')
        return deviation <= self.tolerance_percent, deviation


# NFPA 86 Safety Limits
NFPA_86_SAFETY_LIMITS: Dict[str, NFPA86GoldenValue] = {
    'lel_limit': NFPA86GoldenValue(
        'LEL Limit', Decimal('25'), 'percent of LEL',
        Decimal('0'), '8.3.1', 'Maximum allowable flammable vapor concentration'),
    'purge_time_min': NFPA86GoldenValue(
        'Minimum Purge Time', Decimal('4'), 'volume changes',
        Decimal('0'), '8.5.2.1', 'Minimum furnace volume changes during purge'),
    'purge_airflow_min': NFPA86GoldenValue(
        'Purge Airflow', Decimal('25'), 'percent',
        Decimal('0'), '8.5.2.2', 'Minimum purge airflow rate'),
    'pilot_trial_max': NFPA86GoldenValue(
        'Pilot Trial Time', Decimal('15'), 'seconds',
        Decimal('0'), '8.6.3', 'Maximum pilot trial for ignition'),
    'main_burner_trial_max': NFPA86GoldenValue(
        'Main Burner Trial', Decimal('15'), 'seconds',
        Decimal('0'), '8.6.4', 'Maximum main burner trial'),
    'flame_failure_response': NFPA86GoldenValue(
        'Flame Failure Response', Decimal('4'), 'seconds',
        Decimal('0'), '8.8.3', 'Maximum flame failure response time'),
    'high_limit_setpoint_max': NFPA86GoldenValue(
        'High Limit Setpoint', Decimal('50'), 'degrees above operating',
        Decimal('0'), '7.6.2', 'Maximum high limit above operating temp'),
    'relief_area_ratio': NFPA86GoldenValue(
        'Relief Area', Decimal('0.05'), 'ft2/ft3',
        Decimal('0'), '7.5.4', 'Explosion relief area ratio'),
}

# Furnace Class Definitions per NFPA 86
class FurnaceClass(Enum):
    """Furnace classifications per NFPA 86 Chapter 4."""
    CLASS_A = 'class_a'  # Flammable solids, liquids, or vapors
    CLASS_B = 'class_b'  # Non-flammable materials with flammable atmospheres
    CLASS_C = 'class_c'  # Combustible materials in non-flammable atmospheres
    CLASS_D = 'class_d'  # Non-combustible materials

# LEL Values for Common Substances (vol% in air)
LEL_VALUES: Dict[str, Decimal] = {
    'methane': Decimal('5.0'),
    'propane': Decimal('2.1'),
    'natural_gas': Decimal('4.4'),
    'hydrogen': Decimal('4.0'),
    'acetone': Decimal('2.5'),
    'toluene': Decimal('1.1'),
    'methanol': Decimal('6.0'),
    'ethanol': Decimal('3.3'),
    'gasoline': Decimal('1.4'),
}

# Draft Requirements per API 560
DRAFT_REQUIREMENTS: Dict[str, NFPA86GoldenValue] = {
    'natural_draft_min': NFPA86GoldenValue(
        'Natural Draft Minimum', Decimal('0.1'), 'inWC',
        Decimal('10'), 'API 560', 'Minimum natural draft'),
    'forced_draft_typical': NFPA86GoldenValue(
        'Forced Draft Typical', Decimal('2.0'), 'inWC',
        Decimal('20'), 'API 560', 'Typical forced draft pressure'),
    'induced_draft_typical': NFPA86GoldenValue(
        'Induced Draft Typical', Decimal('-0.5'), 'inWC',
        Decimal('20'), 'API 560', 'Typical induced draft pressure'),
}

# Temperature Uniformity Requirements
TEMP_UNIFORMITY: Dict[str, NFPA86GoldenValue] = {
    'annealing_furnace': NFPA86GoldenValue(
        'Annealing Uniformity', Decimal('10'), 'degrees F',
        Decimal('0'), 'AMS 2750', 'Class 1 uniformity requirement'),
    'heat_treating': NFPA86GoldenValue(
        'Heat Treating Uniformity', Decimal('15'), 'degrees F',
        Decimal('0'), 'AMS 2750', 'Class 2 uniformity requirement'),
    'drying_oven': NFPA86GoldenValue(
        'Drying Oven Uniformity', Decimal('25'), 'degrees F',
        Decimal('0'), 'Industry', 'Typical drying oven uniformity'),
}

# Hotspot Detection Thresholds
HOTSPOT_THRESHOLDS: Dict[str, NFPA86GoldenValue] = {
    'warning': NFPA86GoldenValue(
        'Hotspot Warning', Decimal('50'), 'degrees F above average',
        Decimal('0'), 'Best Practice', 'Warning threshold'),
    'alarm': NFPA86GoldenValue(
        'Hotspot Alarm', Decimal('100'), 'degrees F above average',
        Decimal('0'), 'Best Practice', 'Alarm threshold'),
    'trip': NFPA86GoldenValue(
        'Hotspot Trip', Decimal('150'), 'degrees F above average',
        Decimal('0'), 'NFPA 86', 'Trip threshold'),
}


# ==============================================================================
# CALCULATION FUNCTIONS
# ==============================================================================

def calculate_lel_percentage(
    vapor_concentration_ppm: Decimal,
    lel_vol_percent: Decimal
) -> Decimal:
    """
    Calculate vapor concentration as percentage of LEL.

    Formula: %LEL = (Concentration ppm / (LEL% × 10000)) × 100

    Args:
        vapor_concentration_ppm: Measured concentration in ppm
        lel_vol_percent: LEL of substance in vol%

    Returns: Concentration as percentage of LEL
    """
    lel_ppm = lel_vol_percent * Decimal('10000')  # Convert vol% to ppm
    pct_lel = (vapor_concentration_ppm / lel_ppm) * Decimal('100')
    return pct_lel.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_purge_time(
    furnace_volume_cuft: Decimal,
    purge_airflow_cfm: Decimal,
    required_volume_changes: Decimal = Decimal('4')
) -> Decimal:
    """
    Calculate required purge time per NFPA 86.

    Formula: Time = (Volume × Required Changes) / Airflow

    Returns: Purge time in minutes
    """
    total_volume = furnace_volume_cuft * required_volume_changes
    purge_time = total_volume / purge_airflow_cfm
    return purge_time.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_relief_area(
    furnace_volume_cuft: Decimal,
    relief_ratio: Decimal = Decimal('0.05')
) -> Decimal:
    """
    Calculate required explosion relief area per NFPA 86.

    Formula: Relief Area = Volume × Relief Ratio

    Args:
        furnace_volume_cuft: Furnace volume in cubic feet
        relief_ratio: Relief area ratio (ft2/ft3), default 0.05

    Returns: Required relief area in square feet
    """
    relief_area = furnace_volume_cuft * relief_ratio
    return relief_area.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_stack_draft(
    stack_height_ft: Decimal,
    avg_flue_temp_f: Decimal,
    ambient_temp_f: Decimal
) -> Decimal:
    """
    Calculate natural stack draft.

    Formula: Draft = 0.0034 × H × (1/Ta - 1/Tf)

    Where:
    - H = stack height (ft)
    - Ta = ambient temp (°R = °F + 460)
    - Tf = flue gas temp (°R)

    Returns: Draft in inches of water column (inWC)
    """
    ta_r = ambient_temp_f + Decimal('460')
    tf_r = avg_flue_temp_f + Decimal('460')

    draft = Decimal('0.0034') * stack_height_ft * (
        Decimal('1') / ta_r - Decimal('1') / tf_r
    ) * Decimal('1000')  # Scale factor

    return draft.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def calculate_thermal_efficiency(
    heat_absorbed: Decimal,
    fuel_input: Decimal
) -> Decimal:
    """
    Calculate furnace thermal efficiency.

    Efficiency = (Heat Absorbed / Fuel Input) × 100

    Returns: Efficiency as percentage
    """
    if fuel_input == 0:
        return Decimal('0')
    efficiency = (heat_absorbed / fuel_input) * Decimal('100')
    return efficiency.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_hotspot_severity(
    local_temp: Decimal,
    average_temp: Decimal,
    max_allowable_deviation: Decimal
) -> Tuple[Decimal, str]:
    """
    Calculate hotspot severity and classify.

    Returns: (deviation, severity_level)
    """
    deviation = local_temp - average_temp

    if deviation < HOTSPOT_THRESHOLDS['warning'].value:
        severity = 'normal'
    elif deviation < HOTSPOT_THRESHOLDS['alarm'].value:
        severity = 'warning'
    elif deviation < HOTSPOT_THRESHOLDS['trip'].value:
        severity = 'alarm'
    else:
        severity = 'trip'

    return deviation, severity


def classify_furnace(
    has_flammable_materials: bool,
    has_flammable_atmosphere: bool,
    has_combustible_materials: bool
) -> FurnaceClass:
    """
    Classify furnace per NFPA 86 Chapter 4.

    Class A: Flammable solids, liquids, or vapors present
    Class B: Non-flammable materials with flammable atmospheres
    Class C: Combustible materials in non-flammable atmospheres
    Class D: Non-combustible materials
    """
    if has_flammable_materials or has_flammable_atmosphere:
        return FurnaceClass.CLASS_A
    elif has_flammable_atmosphere:
        return FurnaceClass.CLASS_B
    elif has_combustible_materials:
        return FurnaceClass.CLASS_C
    else:
        return FurnaceClass.CLASS_D


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
class TestNFPA86SafetyLimits:
    """Validate NFPA 86 safety limit values."""

    def test_lel_limit(self):
        """Verify 25% LEL limit per 8.3.1."""
        golden = NFPA_86_SAFETY_LIMITS['lel_limit']
        assert golden.value == Decimal('25')
        assert golden.unit == 'percent of LEL'
        assert golden.nfpa_section == '8.3.1'

    def test_purge_volume_changes(self):
        """Verify 4 volume changes requirement per 8.5.2.1."""
        golden = NFPA_86_SAFETY_LIMITS['purge_time_min']
        assert golden.value == Decimal('4')
        assert golden.unit == 'volume changes'

    def test_pilot_trial_time(self):
        """Verify 15 second pilot trial maximum per 8.6.3."""
        golden = NFPA_86_SAFETY_LIMITS['pilot_trial_max']
        assert golden.value == Decimal('15')
        assert golden.unit == 'seconds'

    def test_flame_failure_response(self):
        """Verify 4 second flame failure response per 8.8.3."""
        golden = NFPA_86_SAFETY_LIMITS['flame_failure_response']
        assert golden.value == Decimal('4')

    def test_relief_area_ratio(self):
        """Verify explosion relief area ratio per 7.5.4."""
        golden = NFPA_86_SAFETY_LIMITS['relief_area_ratio']
        assert golden.value == Decimal('0.05')
        assert golden.unit == 'ft2/ft3'


@pytest.mark.golden
class TestLELCalculation:
    """Validate LEL percentage calculations."""

    @dataclass(frozen=True)
    class LELTestCase:
        name: str
        vapor_ppm: Decimal
        substance: str
        expected_pct_lel: Decimal
        is_safe: bool

    TEST_CASES = [
        LELTestCase('Safe Natural Gas', Decimal('1000'), 'natural_gas',
                    Decimal('2.3'), True),
        LELTestCase('Warning Propane', Decimal('4000'), 'propane',
                    Decimal('19.0'), True),
        LELTestCase('Alarm Methane', Decimal('11000'), 'methane',
                    Decimal('22.0'), True),
        LELTestCase('Dangerous Hydrogen', Decimal('15000'), 'hydrogen',
                    Decimal('37.5'), False),
    ]

    @pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
    def test_lel_calculation(self, case: LELTestCase):
        """Verify LEL percentage calculation."""
        lel = LEL_VALUES[case.substance]
        calculated = calculate_lel_percentage(case.vapor_ppm, lel)

        deviation = abs(calculated - case.expected_pct_lel)
        assert deviation <= Decimal('0.5'), \
            f'{case.name}: {calculated}% LEL differs from {case.expected_pct_lel}%'

        is_safe = calculated <= Decimal('25')
        assert is_safe == case.is_safe

    def test_lel_values_documented(self):
        """Verify LEL values for common substances."""
        assert LEL_VALUES['methane'] == Decimal('5.0')
        assert LEL_VALUES['propane'] == Decimal('2.1')
        assert LEL_VALUES['hydrogen'] == Decimal('4.0')


@pytest.mark.golden
class TestPurgeCalculation:
    """Validate purge time calculations."""

    @dataclass(frozen=True)
    class PurgeTestCase:
        name: str
        volume_cuft: Decimal
        airflow_cfm: Decimal
        expected_time_min: Decimal

    TEST_CASES = [
        PurgeTestCase('Small Oven', Decimal('100'), Decimal('200'), Decimal('2.0')),
        PurgeTestCase('Medium Furnace', Decimal('500'), Decimal('500'), Decimal('4.0')),
        PurgeTestCase('Large Furnace', Decimal('2000'), Decimal('1000'), Decimal('8.0')),
    ]

    @pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
    def test_purge_time_calculation(self, case: PurgeTestCase):
        """Verify purge time calculation."""
        calculated = calculate_purge_time(case.volume_cuft, case.airflow_cfm)

        deviation = abs(calculated - case.expected_time_min)
        assert deviation <= Decimal('0.1'), \
            f'{case.name}: {calculated} min differs from {case.expected_time_min} min'

    def test_minimum_4_volume_changes(self):
        """Verify minimum 4 volume changes used in calculation."""
        volume = Decimal('100')
        airflow = Decimal('100')

        time = calculate_purge_time(volume, airflow)

        # 4 × 100 / 100 = 4 minutes
        assert time == Decimal('4.0')


@pytest.mark.golden
class TestReliefArea:
    """Validate explosion relief area calculations."""

    @dataclass(frozen=True)
    class ReliefTestCase:
        name: str
        volume_cuft: Decimal
        expected_area_sqft: Decimal

    TEST_CASES = [
        ReliefTestCase('Small Oven', Decimal('100'), Decimal('5.0')),
        ReliefTestCase('Medium Furnace', Decimal('500'), Decimal('25.0')),
        ReliefTestCase('Large Furnace', Decimal('1000'), Decimal('50.0')),
    ]

    @pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
    def test_relief_area_calculation(self, case: ReliefTestCase):
        """Verify relief area calculation."""
        calculated = calculate_relief_area(case.volume_cuft)

        assert calculated == case.expected_area_sqft, \
            f'{case.name}: {calculated} ft2 differs from {case.expected_area_sqft} ft2'


@pytest.mark.golden
class TestDraftCalculation:
    """Validate natural draft calculations."""

    def test_natural_draft_calculation(self):
        """Verify stack draft calculation."""
        draft = calculate_stack_draft(
            stack_height_ft=Decimal('50'),
            avg_flue_temp_f=Decimal('400'),
            ambient_temp_f=Decimal('70'),
        )

        # Draft should be positive (negative pressure in furnace)
        assert draft > Decimal('0'), 'Natural draft should create negative pressure'

    def test_draft_increases_with_height(self):
        """Draft should increase with stack height."""
        draft_low = calculate_stack_draft(Decimal('30'), Decimal('400'), Decimal('70'))
        draft_high = calculate_stack_draft(Decimal('60'), Decimal('400'), Decimal('70'))

        assert draft_high > draft_low

    def test_draft_increases_with_temp_diff(self):
        """Draft should increase with temperature difference."""
        draft_low = calculate_stack_draft(Decimal('50'), Decimal('300'), Decimal('70'))
        draft_high = calculate_stack_draft(Decimal('50'), Decimal('500'), Decimal('70'))

        assert draft_high > draft_low


@pytest.mark.golden
class TestThermalEfficiency:
    """Validate thermal efficiency calculations."""

    @dataclass(frozen=True)
    class EfficiencyTestCase:
        name: str
        heat_absorbed: Decimal
        fuel_input: Decimal
        expected_efficiency: Decimal

    TEST_CASES = [
        EfficiencyTestCase('High Efficiency', Decimal('85'), Decimal('100'), Decimal('85.0')),
        EfficiencyTestCase('Moderate Efficiency', Decimal('75'), Decimal('100'), Decimal('75.0')),
        EfficiencyTestCase('Low Efficiency', Decimal('60'), Decimal('100'), Decimal('60.0')),
    ]

    @pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
    def test_efficiency_calculation(self, case: EfficiencyTestCase):
        """Verify thermal efficiency calculation."""
        calculated = calculate_thermal_efficiency(
            case.heat_absorbed, case.fuel_input)

        assert calculated == case.expected_efficiency


@pytest.mark.golden
class TestHotspotDetection:
    """Validate hotspot detection and classification."""

    @dataclass(frozen=True)
    class HotspotTestCase:
        name: str
        local_temp: Decimal
        avg_temp: Decimal
        expected_severity: str

    TEST_CASES = [
        HotspotTestCase('Normal', Decimal('700'), Decimal('680'), 'normal'),
        HotspotTestCase('Warning', Decimal('750'), Decimal('680'), 'warning'),
        HotspotTestCase('Alarm', Decimal('800'), Decimal('680'), 'alarm'),
        HotspotTestCase('Trip', Decimal('850'), Decimal('680'), 'trip'),
    ]

    @pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
    def test_hotspot_classification(self, case: HotspotTestCase):
        """Verify hotspot severity classification."""
        _, severity = calculate_hotspot_severity(
            case.local_temp, case.avg_temp, Decimal('150'))

        assert severity == case.expected_severity, \
            f'{case.name}: Expected {case.expected_severity}, got {severity}'

    def test_hotspot_threshold_values(self):
        """Verify hotspot threshold values."""
        assert HOTSPOT_THRESHOLDS['warning'].value == Decimal('50')
        assert HOTSPOT_THRESHOLDS['alarm'].value == Decimal('100')
        assert HOTSPOT_THRESHOLDS['trip'].value == Decimal('150')


@pytest.mark.golden
class TestFurnaceClassification:
    """Validate furnace classification per NFPA 86 Chapter 4."""

    def test_class_a_flammable(self):
        """Class A for flammable materials."""
        result = classify_furnace(
            has_flammable_materials=True,
            has_flammable_atmosphere=False,
            has_combustible_materials=False,
        )
        assert result == FurnaceClass.CLASS_A

    def test_class_c_combustible(self):
        """Class C for combustible materials in non-flammable atmosphere."""
        result = classify_furnace(
            has_flammable_materials=False,
            has_flammable_atmosphere=False,
            has_combustible_materials=True,
        )
        assert result == FurnaceClass.CLASS_C

    def test_class_d_non_combustible(self):
        """Class D for non-combustible materials."""
        result = classify_furnace(
            has_flammable_materials=False,
            has_flammable_atmosphere=False,
            has_combustible_materials=False,
        )
        assert result == FurnaceClass.CLASS_D


@pytest.mark.golden
class TestTemperatureUniformity:
    """Validate temperature uniformity requirements."""

    def test_annealing_uniformity(self):
        """Verify Class 1 uniformity for annealing."""
        golden = TEMP_UNIFORMITY['annealing_furnace']
        assert golden.value == Decimal('10')
        assert golden.unit == 'degrees F'

    def test_heat_treating_uniformity(self):
        """Verify Class 2 uniformity for heat treating."""
        golden = TEMP_UNIFORMITY['heat_treating']
        assert golden.value == Decimal('15')


@pytest.mark.golden
class TestProvenanceAndDeterminism:
    """Validate provenance tracking and deterministic behavior."""

    def test_all_calculations_deterministic(self):
        """All calculations must be deterministic."""
        calculations = [
            lambda: calculate_lel_percentage(Decimal('5000'), Decimal('5.0')),
            lambda: calculate_purge_time(Decimal('500'), Decimal('500')),
            lambda: calculate_relief_area(Decimal('1000')),
            lambda: calculate_stack_draft(Decimal('50'), Decimal('400'), Decimal('70')),
            lambda: calculate_thermal_efficiency(Decimal('80'), Decimal('100')),
        ]

        for calc in calculations:
            results = set()
            for _ in range(50):
                results.add(str(calc()))
            assert len(results) == 1

    def test_provenance_hash_stability(self):
        """Provenance hash must be stable."""
        inputs = {'volume': '1000', 'airflow': '500'}
        outputs = {'purge_time': '8.0'}

        hashes = set()
        for _ in range(50):
            h = calculate_provenance_hash('purge', inputs, outputs)
            hashes.add(h)

        assert len(hashes) == 1


# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================

def export_golden_values() -> Dict[str, Any]:
    """Export all golden values for documentation."""
    return {
        'metadata': {
            'agent': 'GL-007_FurnacePulse',
            'version': '1.0.0',
            'standard': 'NFPA 86-2023',
        },
        'safety_limits': {
            key: {
                'value': str(val.value),
                'unit': val.unit,
                'section': val.nfpa_section,
            } for key, val in NFPA_86_SAFETY_LIMITS.items()
        },
        'lel_values': {k: str(v) for k, v in LEL_VALUES.items()},
        'hotspot_thresholds': {
            key: str(val.value) for key, val in HOTSPOT_THRESHOLDS.items()
        },
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
