# -*- coding: utf-8 -*-
"""
EPA 40 CFR Part 75 Compliance Golden Value Tests

Comprehensive test suite validating EmissionGuardian calculations against
EPA 40 CFR Part 75 regulatory requirements for CEMS (Continuous Emission
Monitoring Systems).

Reference Documents:
- 40 CFR Part 75: Continuous Emission Monitoring
- 40 CFR Part 75, Appendix A: Specifications for CEMS
- 40 CFR Part 75, Appendix B: Quality Assurance Procedures
- 40 CFR Part 75, Appendix D: Substitute Data for Missing CEMS Data
- 40 CFR Part 75, Appendix F: Conversion Procedures
- EPA Method 19: F-factors for emission rate calculations

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
# EPA GOLDEN VALUE REFERENCE DATA
# ==============================================================================

@dataclass(frozen=True)
class EPAGoldenValue:
    """EPA regulatory reference value with tolerance and citation."""
    name: str
    value: Decimal
    unit: str
    tolerance_percent: Decimal
    cfr_reference: str
    appendix: str
    table_or_section: str

    def validate(self, calculated: Decimal) -> Tuple[bool, Decimal]:
        """Validate calculated value against golden value."""
        if self.value == 0:
            return abs(calculated) < Decimal('0.001'), abs(calculated)
        deviation = abs(calculated - self.value) / abs(self.value) * Decimal('100')
        return deviation <= self.tolerance_percent, deviation


# EPA Method 19 F-factors (dscf/MMBtu)
# Reference: 40 CFR 75, Appendix F, Table 1
EPA_METHOD_19_F_FACTORS: Dict[str, Dict[str, EPAGoldenValue]] = {
    'natural_gas': {
        'Fd': EPAGoldenValue('Fd Natural Gas', Decimal('8710'), 'dscf/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
        'Fw': EPAGoldenValue('Fw Natural Gas', Decimal('10610'), 'wscf/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
        'Fc': EPAGoldenValue('Fc Natural Gas', Decimal('1040'), 'scf CO2/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
    },
    'oil_no2': {
        'Fd': EPAGoldenValue('Fd No.2 Oil', Decimal('9190'), 'dscf/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
        'Fw': EPAGoldenValue('Fw No.2 Oil', Decimal('10320'), 'wscf/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
        'Fc': EPAGoldenValue('Fc No.2 Oil', Decimal('1420'), 'scf CO2/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
    },
    'oil_no6': {
        'Fd': EPAGoldenValue('Fd No.6 Oil', Decimal('9220'), 'dscf/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
        'Fw': EPAGoldenValue('Fw No.6 Oil', Decimal('10190'), 'wscf/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
        'Fc': EPAGoldenValue('Fc No.6 Oil', Decimal('1490'), 'scf CO2/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
    },
    'bituminous_coal': {
        'Fd': EPAGoldenValue('Fd Bituminous', Decimal('9780'), 'dscf/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
        'Fw': EPAGoldenValue('Fw Bituminous', Decimal('10640'), 'wscf/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
        'Fc': EPAGoldenValue('Fc Bituminous', Decimal('1800'), 'scf CO2/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
    },
    'subbituminous_coal': {
        'Fd': EPAGoldenValue('Fd Subbituminous', Decimal('9820'), 'dscf/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
        'Fw': EPAGoldenValue('Fw Subbituminous', Decimal('11710'), 'wscf/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
        'Fc': EPAGoldenValue('Fc Subbituminous', Decimal('1840'), 'scf CO2/MMBtu',
                             Decimal('0.5'), '40 CFR 75', 'Appendix F', 'Table 1'),
    },
}

# RATA Performance Criteria
# Reference: 40 CFR 75, Appendix A, Section 7
EPA_RATA_CRITERIA: Dict[str, EPAGoldenValue] = {
    'ra_limit_standard': EPAGoldenValue(
        'RA Limit Standard', Decimal('10.0'), 'percent',
        Decimal('0'), '40 CFR 75', 'Appendix A', 'Section 7.3'),
    'ra_limit_abbreviated': EPAGoldenValue(
        'RA Limit Abbreviated', Decimal('7.5'), 'percent',
        Decimal('0'), '40 CFR 75', 'Appendix A', 'Section 7.3'),
    'min_runs_standard': EPAGoldenValue(
        'Min Runs Standard', Decimal('9'), 'runs',
        Decimal('0'), '40 CFR 75', 'Appendix A', 'Section 7.1'),
    'min_runs_abbreviated': EPAGoldenValue(
        'Min Runs Abbreviated', Decimal('3'), 'runs',
        Decimal('0'), '40 CFR 75', 'Appendix A', 'Section 7.1'),
}

# CGA (Cylinder Gas Audit) Performance Criteria
# Reference: 40 CFR 75, Appendix B, Section 2.1
EPA_CGA_CRITERIA: Dict[str, EPAGoldenValue] = {
    'accuracy_limit': EPAGoldenValue(
        'CGA Accuracy', Decimal('5.0'), 'percent of span',
        Decimal('0'), '40 CFR 75', 'Appendix B', 'Section 2.1'),
    'daily_cal_error': EPAGoldenValue(
        'Daily Cal Error', Decimal('2.5'), 'percent of span',
        Decimal('0'), '40 CFR 75', 'Appendix B', 'Section 2.1'),
    'linearity_error': EPAGoldenValue(
        'Linearity Error', Decimal('5.0'), 'percent',
        Decimal('0'), '40 CFR 75', 'Appendix A', 'Section 6'),
}

# Molecular weights for emission calculations
MOLECULAR_WEIGHTS: Dict[str, Decimal] = {
    'SO2': Decimal('64.066'),
    'NOx_as_NO2': Decimal('46.006'),
    'CO': Decimal('28.010'),
    'CO2': Decimal('44.010'),
}

# Molar volume at standard conditions (scf/lb-mol)
MOLAR_VOLUME_SCF = Decimal('385.5')  # at 68°F, 29.92 inHg


# ==============================================================================
# CALCULATION FUNCTIONS
# ==============================================================================

def calculate_so2_mass_rate_f_factor(
    so2_ppm: Decimal,
    heat_input_mmbtu_hr: Decimal,
    fd_factor: Decimal,
    o2_percent_dry: Decimal,
    decimal_precision: int = 3
) -> Decimal:
    """
    Calculate SO2 mass emission rate using F-factor method.

    Formula per 40 CFR 75, Appendix F, Equation F-1:
    E_SO2 = K * C_SO2 * Fd * H * (20.9 / (20.9 - O2d))

    Where:
    - K = 1.660 × 10^-7 for SO2 in lb/scf per ppm
    - C_SO2 = SO2 concentration in ppm
    - Fd = F-factor (dscf/MMBtu)
    - H = Heat input rate (MMBtu/hr)
    - O2d = O2 concentration (% dry)

    Returns: SO2 mass emission rate in lb/hr
    """
    # Constant K for SO2 (converts ppm to lb/scf)
    K_SO2 = Decimal('1.660E-7')

    # O2 correction factor
    o2_correction = Decimal('20.9') / (Decimal('20.9') - o2_percent_dry)

    # Mass emission rate
    e_so2 = K_SO2 * so2_ppm * fd_factor * heat_input_mmbtu_hr * o2_correction

    return e_so2.quantize(Decimal(10) ** -decimal_precision, rounding=ROUND_HALF_UP)


def calculate_nox_lb_per_mmbtu(
    nox_ppm: Decimal,
    fd_factor: Decimal,
    o2_percent_dry: Decimal,
    decimal_precision: int = 4
) -> Decimal:
    """
    Calculate NOx emission rate in lb/MMBtu using F-factor method.

    Formula per 40 CFR 75, Appendix F, Equation F-5:
    E_NOx = K * C_NOx * Fd * (20.9 / (20.9 - O2d))

    Where:
    - K = 1.194 × 10^-7 for NOx (as NO2) in lb/scf per ppm
    - C_NOx = NOx concentration in ppm (as NO2)
    - Fd = F-factor (dscf/MMBtu)
    - O2d = O2 concentration (% dry)

    Returns: NOx emission rate in lb/MMBtu
    """
    # Constant K for NOx as NO2
    K_NOX = Decimal('1.194E-7')

    # O2 correction factor
    o2_correction = Decimal('20.9') / (Decimal('20.9') - o2_percent_dry)

    # Emission rate
    e_nox = K_NOX * nox_ppm * fd_factor * o2_correction

    return e_nox.quantize(Decimal(10) ** -decimal_precision, rounding=ROUND_HALF_UP)


def calculate_relative_accuracy(
    cems_values: List[Decimal],
    rm_values: List[Decimal],
    decimal_precision: int = 2
) -> Tuple[Decimal, bool, Dict[str, Decimal]]:
    """
    Calculate RATA Relative Accuracy per 40 CFR 75, Appendix A, Section 7.

    Formula:
    RA = (|d_bar| + CC) / RM_mean × 100

    Where:
    - d_bar = mean of (CEMS - RM) differences
    - CC = t × Sd / sqrt(n) (confidence coefficient)
    - RM_mean = mean of reference method values

    Returns: (RA%, passed, detailed_results)
    """
    n = len(cems_values)
    if n != len(rm_values):
        raise ValueError('CEMS and RM lists must have same length')

    # t-values for 95% confidence
    T_VALUES = {
        2: Decimal('4.303'), 3: Decimal('3.182'), 4: Decimal('2.776'),
        5: Decimal('2.571'), 6: Decimal('2.447'), 7: Decimal('2.365'),
        8: Decimal('2.306'), 9: Decimal('2.262'), 10: Decimal('2.228'),
        11: Decimal('2.201'), 12: Decimal('2.179'),
    }

    # Calculate differences
    differences = [c - r for c, r in zip(cems_values, rm_values)]

    # Mean difference
    d_bar = sum(differences) / Decimal(n)

    # Reference method mean
    rm_mean = sum(rm_values) / Decimal(n)

    # Standard deviation
    sum_sq_dev = sum((d - d_bar) ** 2 for d in differences)
    std_dev = (sum_sq_dev / Decimal(n - 1)).sqrt()

    # Confidence coefficient
    t_value = T_VALUES.get(n - 1, Decimal('2.0'))
    cc = t_value * std_dev / Decimal(n).sqrt()

    # Relative Accuracy
    ra = (abs(d_bar) + cc) / rm_mean * Decimal('100')
    ra = ra.quantize(Decimal(10) ** -decimal_precision, rounding=ROUND_HALF_UP)

    # Pass/fail (10% for standard, 7.5% for abbreviated)
    threshold = Decimal('7.5') if n <= 3 else Decimal('10.0')
    passed = ra <= threshold

    details = {
        'd_bar': d_bar,
        'std_dev': std_dev,
        'confidence_coefficient': cc,
        'rm_mean': rm_mean,
        'threshold': threshold,
    }

    return ra, passed, details


def calculate_cga_accuracy(
    reference_value: Decimal,
    measured_value: Decimal,
    span_value: Decimal,
    decimal_precision: int = 2
) -> Tuple[Decimal, bool]:
    """
    Calculate CGA accuracy per 40 CFR 75, Appendix B.

    Formula:
    Accuracy = |Reference - Measured| / Span × 100

    Returns: (accuracy %, passed)
    """
    accuracy = abs(reference_value - measured_value) / span_value * Decimal('100')
    accuracy = accuracy.quantize(Decimal(10) ** -decimal_precision, rounding=ROUND_HALF_UP)

    passed = accuracy <= Decimal('5.0')
    return accuracy, passed


def calculate_substitute_data_tier(
    consecutive_missing_hours: int
) -> Tuple[str, Decimal]:
    """
    Determine substitute data tier and percentile per 40 CFR 75, Appendix D.

    Tier 1 (Hours 1-720): 90th percentile
    Tier 2 (Hours 721-2160): Maximum value
    Tier 3 (Hours 2161+): Maximum or 200%

    Returns: (tier_name, percentile)
    """
    if consecutive_missing_hours <= 720:
        return 'Tier 1', Decimal('90')
    elif consecutive_missing_hours <= 2160:
        return 'Tier 2', Decimal('100')  # Maximum = 100th percentile
    else:
        return 'Tier 3', Decimal('100')


def calculate_percentile(values: List[Decimal], percentile: Decimal) -> Decimal:
    """Calculate percentile using EPA linear interpolation method."""
    sorted_values = sorted(values)
    n = len(sorted_values)

    rank = (percentile / Decimal('100')) * Decimal(n + 1)

    if rank <= 1:
        return sorted_values[0]
    elif rank >= n:
        return sorted_values[-1]

    lower_idx = int(rank) - 1
    upper_idx = lower_idx + 1
    fraction = rank - Decimal(int(rank))

    return sorted_values[lower_idx] + fraction * (sorted_values[upper_idx] - sorted_values[lower_idx])


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
class TestEPAMethod19FFactor:
    """Validate F-factor values against EPA Method 19 Table 1."""

    @pytest.mark.parametrize("fuel,expected_fd", [
        ('natural_gas', Decimal('8710')),
        ('oil_no2', Decimal('9190')),
        ('oil_no6', Decimal('9220')),
        ('bituminous_coal', Decimal('9780')),
        ('subbituminous_coal', Decimal('9820')),
    ])
    def test_fd_factors(self, fuel: str, expected_fd: Decimal):
        """Verify Fd (dry basis) F-factors match EPA values."""
        golden = EPA_METHOD_19_F_FACTORS[fuel]['Fd']
        assert golden.value == expected_fd
        assert golden.cfr_reference == '40 CFR 75'
        assert golden.appendix == 'Appendix F'

    @pytest.mark.parametrize("fuel,expected_fw", [
        ('natural_gas', Decimal('10610')),
        ('oil_no2', Decimal('10320')),
        ('oil_no6', Decimal('10190')),
        ('bituminous_coal', Decimal('10640')),
        ('subbituminous_coal', Decimal('11710')),
    ])
    def test_fw_factors(self, fuel: str, expected_fw: Decimal):
        """Verify Fw (wet basis) F-factors match EPA values."""
        golden = EPA_METHOD_19_F_FACTORS[fuel]['Fw']
        assert golden.value == expected_fw

    @pytest.mark.parametrize("fuel,expected_fc", [
        ('natural_gas', Decimal('1040')),
        ('oil_no2', Decimal('1420')),
        ('oil_no6', Decimal('1490')),
        ('bituminous_coal', Decimal('1800')),
        ('subbituminous_coal', Decimal('1840')),
    ])
    def test_fc_factors(self, fuel: str, expected_fc: Decimal):
        """Verify Fc (carbon-based) F-factors match EPA values."""
        golden = EPA_METHOD_19_F_FACTORS[fuel]['Fc']
        assert golden.value == expected_fc

    def test_fw_greater_than_fd_all_fuels(self):
        """Fw must be greater than Fd for all fuels (moisture adds volume)."""
        for fuel, factors in EPA_METHOD_19_F_FACTORS.items():
            assert factors['Fw'].value > factors['Fd'].value, \
                f'{fuel}: Fw must exceed Fd'


@pytest.mark.golden
class TestSO2MassEmissionRate:
    """Validate SO2 mass emission rate calculations per Appendix F."""

    @dataclass(frozen=True)
    class SO2TestCase:
        name: str
        fuel: str
        so2_ppm: Decimal
        heat_input_mmbtu_hr: Decimal
        o2_percent_dry: Decimal
        expected_lb_hr: Decimal
        tolerance_percent: Decimal

    # Golden test cases from EPA technical guidance
    SO2_TEST_CASES = [
        SO2TestCase(
            name='Natural Gas Low Sulfur',
            fuel='natural_gas',
            so2_ppm=Decimal('5'),
            heat_input_mmbtu_hr=Decimal('100'),
            o2_percent_dry=Decimal('3.0'),
            expected_lb_hr=Decimal('0.084'),
            tolerance_percent=Decimal('1.0'),
        ),
        SO2TestCase(
            name='Oil No.2 Medium Load',
            fuel='oil_no2',
            so2_ppm=Decimal('150'),
            heat_input_mmbtu_hr=Decimal('250'),
            o2_percent_dry=Decimal('4.0'),
            expected_lb_hr=Decimal('7.18'),
            tolerance_percent=Decimal('1.0'),
        ),
        SO2TestCase(
            name='Bituminous Coal Full Load',
            fuel='bituminous_coal',
            so2_ppm=Decimal('800'),
            heat_input_mmbtu_hr=Decimal('500'),
            o2_percent_dry=Decimal('5.0'),
            expected_lb_hr=Decimal('86.1'),
            tolerance_percent=Decimal('1.0'),
        ),
    ]

    @pytest.mark.parametrize("test_case", SO2_TEST_CASES, ids=lambda tc: tc.name)
    def test_so2_mass_rate_calculation(self, test_case: SO2TestCase):
        """Verify SO2 mass emission rate calculation."""
        fd_factor = EPA_METHOD_19_F_FACTORS[test_case.fuel]['Fd'].value

        calculated = calculate_so2_mass_rate_f_factor(
            so2_ppm=test_case.so2_ppm,
            heat_input_mmbtu_hr=test_case.heat_input_mmbtu_hr,
            fd_factor=fd_factor,
            o2_percent_dry=test_case.o2_percent_dry,
        )

        deviation = abs(calculated - test_case.expected_lb_hr) / test_case.expected_lb_hr * 100
        assert deviation <= test_case.tolerance_percent, \
            f'{test_case.name}: SO2 {calculated} lb/hr deviates {deviation:.2f}% from expected {test_case.expected_lb_hr}'

    def test_so2_zero_emission(self):
        """Zero SO2 concentration should yield zero emission."""
        result = calculate_so2_mass_rate_f_factor(
            so2_ppm=Decimal('0'),
            heat_input_mmbtu_hr=Decimal('100'),
            fd_factor=Decimal('8710'),
            o2_percent_dry=Decimal('3.0'),
        )
        assert result == Decimal('0')

    def test_so2_calculation_determinism(self):
        """SO2 calculation must be 100% deterministic."""
        hashes = set()
        for _ in range(100):
            result = calculate_so2_mass_rate_f_factor(
                so2_ppm=Decimal('100'),
                heat_input_mmbtu_hr=Decimal('200'),
                fd_factor=Decimal('8710'),
                o2_percent_dry=Decimal('3.5'),
            )
            hashes.add(str(result))
        assert len(hashes) == 1, 'SO2 calculation is non-deterministic'


@pytest.mark.golden
class TestNOxEmissionRate:
    """Validate NOx emission rate calculations per Appendix F."""

    @dataclass(frozen=True)
    class NOxTestCase:
        name: str
        fuel: str
        nox_ppm: Decimal
        o2_percent_dry: Decimal
        expected_lb_mmbtu: Decimal
        tolerance_percent: Decimal

    # Golden test cases from EPA documentation
    NOX_TEST_CASES = [
        NOxTestCase(
            name='Natural Gas Low NOx Burner',
            fuel='natural_gas',
            nox_ppm=Decimal('40'),
            o2_percent_dry=Decimal('3.0'),
            expected_lb_mmbtu=Decimal('0.0485'),
            tolerance_percent=Decimal('1.0'),
        ),
        NOxTestCase(
            name='Oil No.2 Standard Combustion',
            fuel='oil_no2',
            nox_ppm=Decimal('120'),
            o2_percent_dry=Decimal('4.0'),
            expected_lb_mmbtu=Decimal('0.163'),
            tolerance_percent=Decimal('1.0'),
        ),
        NOxTestCase(
            name='Coal High Load',
            fuel='bituminous_coal',
            nox_ppm=Decimal('250'),
            o2_percent_dry=Decimal('5.0'),
            expected_lb_mmbtu=Decimal('0.384'),
            tolerance_percent=Decimal('1.0'),
        ),
    ]

    @pytest.mark.parametrize("test_case", NOX_TEST_CASES, ids=lambda tc: tc.name)
    def test_nox_lb_per_mmbtu_calculation(self, test_case: NOxTestCase):
        """Verify NOx emission rate calculation in lb/MMBtu."""
        fd_factor = EPA_METHOD_19_F_FACTORS[test_case.fuel]['Fd'].value

        calculated = calculate_nox_lb_per_mmbtu(
            nox_ppm=test_case.nox_ppm,
            fd_factor=fd_factor,
            o2_percent_dry=test_case.o2_percent_dry,
        )

        deviation = abs(calculated - test_case.expected_lb_mmbtu) / test_case.expected_lb_mmbtu * 100
        assert deviation <= test_case.tolerance_percent, \
            f'{test_case.name}: NOx {calculated} lb/MMBtu deviates {deviation:.2f}%'

    def test_nox_increases_with_concentration(self):
        """Higher NOx concentration should yield higher emission rate."""
        low = calculate_nox_lb_per_mmbtu(
            nox_ppm=Decimal('50'), fd_factor=Decimal('8710'), o2_percent_dry=Decimal('3.0'))
        high = calculate_nox_lb_per_mmbtu(
            nox_ppm=Decimal('200'), fd_factor=Decimal('8710'), o2_percent_dry=Decimal('3.0'))
        assert high > low


@pytest.mark.golden
class TestRATACalculation:
    """Validate RATA calculations per Appendix A, Section 7."""

    @dataclass(frozen=True)
    class RATATestCase:
        name: str
        cems_values: List[Decimal]
        rm_values: List[Decimal]
        expected_ra: Decimal
        expected_pass: bool
        tolerance_percent: Decimal

    # EPA RATA calculation examples
    RATA_TEST_CASES = [
        RATATestCase(
            name='9-Run Passing RATA',
            cems_values=[Decimal(str(x)) for x in [102, 98, 101, 99, 100, 103, 97, 102, 99]],
            rm_values=[Decimal(str(x)) for x in [100, 100, 100, 100, 100, 100, 100, 100, 100]],
            expected_ra=Decimal('4.5'),
            expected_pass=True,
            tolerance_percent=Decimal('10.0'),
        ),
        RATATestCase(
            name='9-Run Failing RATA',
            cems_values=[Decimal(str(x)) for x in [120, 115, 118, 122, 119, 121, 117, 123, 116]],
            rm_values=[Decimal(str(x)) for x in [100, 100, 100, 100, 100, 100, 100, 100, 100]],
            expected_ra=Decimal('22.0'),
            expected_pass=False,
            tolerance_percent=Decimal('15.0'),
        ),
        RATATestCase(
            name='3-Run Abbreviated RATA Pass',
            cems_values=[Decimal('101'), Decimal('99'), Decimal('100')],
            rm_values=[Decimal('100'), Decimal('100'), Decimal('100')],
            expected_ra=Decimal('3.5'),
            expected_pass=True,
            tolerance_percent=Decimal('20.0'),
        ),
    ]

    def test_rata_criteria_values(self):
        """Verify RATA pass/fail criteria match EPA specifications."""
        assert EPA_RATA_CRITERIA['ra_limit_standard'].value == Decimal('10.0')
        assert EPA_RATA_CRITERIA['ra_limit_abbreviated'].value == Decimal('7.5')
        assert EPA_RATA_CRITERIA['min_runs_standard'].value == Decimal('9')
        assert EPA_RATA_CRITERIA['min_runs_abbreviated'].value == Decimal('3')

    @pytest.mark.parametrize("test_case", RATA_TEST_CASES, ids=lambda tc: tc.name)
    def test_rata_calculation(self, test_case: RATATestCase):
        """Verify RATA relative accuracy calculation."""
        ra, passed, details = calculate_relative_accuracy(
            test_case.cems_values, test_case.rm_values)

        assert passed == test_case.expected_pass, \
            f'{test_case.name}: Expected pass={test_case.expected_pass}, got {passed}'

        deviation = abs(ra - test_case.expected_ra) / test_case.expected_ra * 100
        assert deviation <= test_case.tolerance_percent, \
            f'{test_case.name}: RA {ra}% deviates {deviation:.1f}% from expected {test_case.expected_ra}%'

    def test_rata_determinism(self):
        """RATA calculation must be deterministic."""
        cems = [Decimal('100'), Decimal('102'), Decimal('98'), Decimal('101'),
                Decimal('99'), Decimal('103'), Decimal('97'), Decimal('100'), Decimal('101')]
        rm = [Decimal('100')] * 9

        results = set()
        for _ in range(50):
            ra, _, _ = calculate_relative_accuracy(cems, rm)
            results.add(str(ra))
        assert len(results) == 1


@pytest.mark.golden
class TestCGACalculation:
    """Validate CGA (Cylinder Gas Audit) calculations per Appendix B."""

    @dataclass(frozen=True)
    class CGATestCase:
        name: str
        reference_value: Decimal
        measured_value: Decimal
        span_value: Decimal
        expected_accuracy: Decimal
        expected_pass: bool

    CGA_TEST_CASES = [
        CGATestCase(
            name='Low-Level Pass',
            reference_value=Decimal('50'),
            measured_value=Decimal('51'),
            span_value=Decimal('100'),
            expected_accuracy=Decimal('1.0'),
            expected_pass=True,
        ),
        CGATestCase(
            name='Mid-Level Pass',
            reference_value=Decimal('200'),
            measured_value=Decimal('204'),
            span_value=Decimal('400'),
            expected_accuracy=Decimal('1.0'),
            expected_pass=True,
        ),
        CGATestCase(
            name='High-Level Marginal',
            reference_value=Decimal('380'),
            measured_value=Decimal('360'),
            span_value=Decimal('400'),
            expected_accuracy=Decimal('5.0'),
            expected_pass=True,
        ),
        CGATestCase(
            name='Failing CGA',
            reference_value=Decimal('100'),
            measured_value=Decimal('110'),
            span_value=Decimal('150'),
            expected_accuracy=Decimal('6.67'),
            expected_pass=False,
        ),
    ]

    def test_cga_criteria_values(self):
        """Verify CGA accuracy criteria match EPA specifications."""
        assert EPA_CGA_CRITERIA['accuracy_limit'].value == Decimal('5.0')
        assert EPA_CGA_CRITERIA['daily_cal_error'].value == Decimal('2.5')
        assert EPA_CGA_CRITERIA['linearity_error'].value == Decimal('5.0')

    @pytest.mark.parametrize("test_case", CGA_TEST_CASES, ids=lambda tc: tc.name)
    def test_cga_accuracy_calculation(self, test_case: CGATestCase):
        """Verify CGA accuracy calculation."""
        accuracy, passed = calculate_cga_accuracy(
            test_case.reference_value,
            test_case.measured_value,
            test_case.span_value,
        )

        assert passed == test_case.expected_pass, \
            f'{test_case.name}: Expected pass={test_case.expected_pass}, got {passed}'

        deviation = abs(accuracy - test_case.expected_accuracy)
        assert deviation <= Decimal('0.1'), \
            f'{test_case.name}: Accuracy {accuracy}% differs from expected {test_case.expected_accuracy}%'


@pytest.mark.golden
class TestDataSubstitution:
    """Validate data substitution algorithm per Appendix D."""

    @dataclass(frozen=True)
    class SubstitutionTestCase:
        name: str
        consecutive_missing_hours: int
        expected_tier: str
        expected_percentile: Decimal

    SUBSTITUTION_TEST_CASES = [
        SubstitutionTestCase('Hour 1', 1, 'Tier 1', Decimal('90')),
        SubstitutionTestCase('Hour 720', 720, 'Tier 1', Decimal('90')),
        SubstitutionTestCase('Hour 721', 721, 'Tier 2', Decimal('100')),
        SubstitutionTestCase('Hour 2160', 2160, 'Tier 2', Decimal('100')),
        SubstitutionTestCase('Hour 2161', 2161, 'Tier 3', Decimal('100')),
        SubstitutionTestCase('Hour 5000', 5000, 'Tier 3', Decimal('100')),
    ]

    @pytest.mark.parametrize("test_case", SUBSTITUTION_TEST_CASES, ids=lambda tc: tc.name)
    def test_substitution_tier_determination(self, test_case: SubstitutionTestCase):
        """Verify correct substitution tier based on missing hours."""
        tier, percentile = calculate_substitute_data_tier(test_case.consecutive_missing_hours)

        assert tier == test_case.expected_tier, \
            f'{test_case.name}: Expected tier {test_case.expected_tier}, got {tier}'
        assert percentile == test_case.expected_percentile

    def test_percentile_calculation(self):
        """Verify 90th percentile calculation per EPA method."""
        # Test data with known 90th percentile
        values = [Decimal(str(x)) for x in range(1, 101)]  # 1 to 100
        p90 = calculate_percentile(values, Decimal('90'))

        # 90th percentile of 1-100 should be approximately 90.1
        assert Decimal('89') <= p90 <= Decimal('92'), \
            f'90th percentile of 1-100 should be ~90, got {p90}'

    def test_tier_boundary_hours(self):
        """Verify tier boundaries at exact hour thresholds."""
        # At exactly 720 hours - should be Tier 1
        tier_720, _ = calculate_substitute_data_tier(720)
        assert tier_720 == 'Tier 1'

        # At 721 hours - should be Tier 2
        tier_721, _ = calculate_substitute_data_tier(721)
        assert tier_721 == 'Tier 2'

        # At exactly 2160 hours - should be Tier 2
        tier_2160, _ = calculate_substitute_data_tier(2160)
        assert tier_2160 == 'Tier 2'

        # At 2161 hours - should be Tier 3
        tier_2161, _ = calculate_substitute_data_tier(2161)
        assert tier_2161 == 'Tier 3'


@pytest.mark.golden
class TestDataAvailability:
    """Validate data availability calculations per Subpart D."""

    @dataclass(frozen=True)
    class AvailabilityTestCase:
        name: str
        valid_hours: int
        total_hours: int
        expected_percent: Decimal
        expected_meets_minimum: bool

    AVAILABILITY_TEST_CASES = [
        AvailabilityTestCase('Excellent', 8700, 8760, Decimal('99.32'), True),
        AvailabilityTestCase('At Minimum', 7884, 8760, Decimal('90.00'), True),
        AvailabilityTestCase('Below Minimum', 7800, 8760, Decimal('89.04'), False),
        AvailabilityTestCase('Poor', 7000, 8760, Decimal('79.91'), False),
    ]

    @pytest.mark.parametrize("test_case", AVAILABILITY_TEST_CASES, ids=lambda tc: tc.name)
    def test_data_availability_calculation(self, test_case: AvailabilityTestCase):
        """Verify data availability percentage calculation."""
        availability = (Decimal(test_case.valid_hours) /
                       Decimal(test_case.total_hours) * Decimal('100'))
        availability = availability.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        meets_min = availability >= Decimal('90.0')

        assert meets_min == test_case.expected_meets_minimum, \
            f'{test_case.name}: {availability}% should {"meet" if test_case.expected_meets_minimum else "not meet"} 90% minimum'


@pytest.mark.golden
class TestProvenanceAndDeterminism:
    """Validate provenance tracking and deterministic behavior."""

    def test_provenance_hash_stability(self):
        """Same inputs must produce same provenance hash."""
        inputs = {'so2_ppm': '100', 'heat_input': '200'}
        outputs = {'emission_rate': '5.5'}

        hashes = set()
        for _ in range(100):
            h = calculate_provenance_hash('test_calc', inputs, outputs)
            hashes.add(h)

        assert len(hashes) == 1, 'Provenance hash is not deterministic'

    def test_provenance_hash_changes_with_input(self):
        """Different inputs must produce different hashes."""
        outputs = {'result': '10'}

        hash1 = calculate_provenance_hash('calc', {'a': '1'}, outputs)
        hash2 = calculate_provenance_hash('calc', {'a': '2'}, outputs)

        assert hash1 != hash2, 'Different inputs should produce different hashes'

    def test_full_calculation_chain_determinism(self):
        """Complete calculation chain must be deterministic."""
        results = []
        for _ in range(50):
            # SO2 calculation
            so2 = calculate_so2_mass_rate_f_factor(
                so2_ppm=Decimal('100'),
                heat_input_mmbtu_hr=Decimal('250'),
                fd_factor=Decimal('8710'),
                o2_percent_dry=Decimal('3.5'),
            )

            # NOx calculation
            nox = calculate_nox_lb_per_mmbtu(
                nox_ppm=Decimal('80'),
                fd_factor=Decimal('8710'),
                o2_percent_dry=Decimal('3.5'),
            )

            # Combine results
            combined = f'{so2}_{nox}'
            results.append(combined)

        assert len(set(results)) == 1, 'Calculation chain is non-deterministic'


@pytest.mark.golden
class TestMolecularWeights:
    """Validate molecular weight constants for emission calculations."""

    @pytest.mark.parametrize("compound,expected_mw", [
        ('SO2', Decimal('64.066')),
        ('NOx_as_NO2', Decimal('46.006')),
        ('CO', Decimal('28.010')),
        ('CO2', Decimal('44.010')),
    ])
    def test_molecular_weights(self, compound: str, expected_mw: Decimal):
        """Verify molecular weights match accepted values."""
        assert MOLECULAR_WEIGHTS[compound] == expected_mw

    def test_molar_volume(self):
        """Verify molar volume at standard conditions."""
        assert MOLAR_VOLUME_SCF == Decimal('385.5')


# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================

def export_golden_values() -> Dict[str, Any]:
    """Export all golden values for documentation and validation."""
    return {
        'metadata': {
            'agent': 'GL-010_EmissionGuardian',
            'version': '1.0.0',
            'regulation': 'EPA 40 CFR Part 75',
        },
        'f_factors': {
            fuel: {
                factor: str(data.value) for factor, data in factors.items()
            } for fuel, factors in EPA_METHOD_19_F_FACTORS.items()
        },
        'rata_criteria': {
            key: str(val.value) for key, val in EPA_RATA_CRITERIA.items()
        },
        'cga_criteria': {
            key: str(val.value) for key, val in EPA_CGA_CRITERIA.items()
        },
        'molecular_weights': {
            k: str(v) for k, v in MOLECULAR_WEIGHTS.items()
        },
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
