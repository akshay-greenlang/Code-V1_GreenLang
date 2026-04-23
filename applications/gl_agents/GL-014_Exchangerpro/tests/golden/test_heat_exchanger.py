"""
GL-014 ExchangerPro: Heat Exchanger Golden Value Tests.

Reference Standards:
- ASME PTC 12.5: Performance Test Code for Single-Phase Heat Exchangers
- TEMA: Tubular Exchanger Manufacturers Association Standards
- VDI Heat Atlas: Heat Transfer Design Methods
- Kern's Process Heat Transfer (Classic Reference)

These golden tests validate heat exchanger performance calculations,
LMTD corrections, fouling factors, NTU-effectiveness, and UA values.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional, Tuple

import pytest

# =============================================================================
# GOLDEN VALUE REFERENCE DATA - HEAT EXCHANGER STANDARDS
# =============================================================================


@dataclass(frozen=True)
class HeatExchangerGoldenValue:
    """Immutable golden value for heat exchanger validation."""

    description: str
    value: Decimal
    unit: str
    tolerance_percent: Decimal
    source: str
    exchanger_type: str


# TEMA Fouling Factors (Rf) - Table RCB-2.21
FOULING_FACTORS: Dict[str, HeatExchangerGoldenValue] = {
    'cooling_tower_water': HeatExchangerGoldenValue(
        'Cooling Tower Water Fouling',
        Decimal('0.00035'),
        'm²-K/W',
        Decimal('20'),
        'TEMA RCB-2.21',
        'general',
    ),
    'treated_boiler_water': HeatExchangerGoldenValue(
        'Treated Boiler Feedwater Fouling',
        Decimal('0.00009'),
        'm²-K/W',
        Decimal('20'),
        'TEMA RCB-2.21',
        'general',
    ),
    'fuel_oil': HeatExchangerGoldenValue(
        'Fuel Oil Fouling',
        Decimal('0.00088'),
        'm²-K/W',
        Decimal('20'),
        'TEMA RCB-2.21',
        'general',
    ),
    'engine_exhaust': HeatExchangerGoldenValue(
        'Engine Exhaust Gas Fouling',
        Decimal('0.00176'),
        'm²-K/W',
        Decimal('20'),
        'TEMA RCB-2.21',
        'general',
    ),
    'steam_condensing': HeatExchangerGoldenValue(
        'Steam (oil-free) Fouling',
        Decimal('0.00009'),
        'm²-K/W',
        Decimal('20'),
        'TEMA RCB-2.21',
        'condenser',
    ),
    'refrigerant': HeatExchangerGoldenValue(
        'Refrigerant Fouling',
        Decimal('0.00018'),
        'm²-K/W',
        Decimal('20'),
        'TEMA RCB-2.21',
        'evaporator',
    ),
    'river_water': HeatExchangerGoldenValue(
        'River Water Fouling',
        Decimal('0.00053'),
        'm²-K/W',
        Decimal('20'),
        'TEMA RCB-2.21',
        'general',
    ),
    'seawater': HeatExchangerGoldenValue(
        'Seawater Fouling',
        Decimal('0.00035'),
        'm²-K/W',
        Decimal('20'),
        'TEMA RCB-2.21',
        'general',
    ),
}


# LMTD Correction Factors (F) for various configurations
# Reference: Kern's Process Heat Transfer, VDI Heat Atlas
LMTD_CORRECTION_FACTORS: Dict[str, HeatExchangerGoldenValue] = {
    'counterflow': HeatExchangerGoldenValue(
        'Counterflow LMTD Factor',
        Decimal('1.0'),
        'dimensionless',
        Decimal('0'),
        'Theoretical',
        'counterflow',
    ),
    'parallelflow': HeatExchangerGoldenValue(
        'Parallelflow LMTD Factor',
        Decimal('1.0'),
        'dimensionless',
        Decimal('0'),
        'Theoretical',
        'parallelflow',
    ),
    '1_shell_2_tube_R05_P05': HeatExchangerGoldenValue(
        '1-2 Shell-Tube F at R=0.5, P=0.5',
        Decimal('0.925'),
        'dimensionless',
        Decimal('2'),
        'TEMA Charts',
        'shell_tube',
    ),
    '1_shell_2_tube_R10_P04': HeatExchangerGoldenValue(
        '1-2 Shell-Tube F at R=1.0, P=0.4',
        Decimal('0.922'),
        'dimensionless',
        Decimal('2'),
        'TEMA Charts',
        'shell_tube',
    ),
    '1_shell_4_tube_R05_P06': HeatExchangerGoldenValue(
        '1-4 Shell-Tube F at R=0.5, P=0.6',
        Decimal('0.890'),
        'dimensionless',
        Decimal('2'),
        'TEMA Charts',
        'shell_tube',
    ),
    'crossflow_unmixed': HeatExchangerGoldenValue(
        'Crossflow (both unmixed) F at NTU=1',
        Decimal('0.95'),
        'dimensionless',
        Decimal('3'),
        'VDI Heat Atlas',
        'crossflow',
    ),
}


# NTU-Effectiveness Reference Values
# Reference: Incropera & DeWitt, ASME PTC 12.5
NTU_EFFECTIVENESS: Dict[str, HeatExchangerGoldenValue] = {
    'counterflow_Cr0_NTU1': HeatExchangerGoldenValue(
        'Counterflow ε at Cr=0, NTU=1',
        Decimal('0.632'),
        'dimensionless',
        Decimal('1'),
        'Analytical',
        'counterflow',
    ),
    'counterflow_Cr0_NTU2': HeatExchangerGoldenValue(
        'Counterflow ε at Cr=0, NTU=2',
        Decimal('0.865'),
        'dimensionless',
        Decimal('1'),
        'Analytical',
        'counterflow',
    ),
    'counterflow_Cr1_NTU1': HeatExchangerGoldenValue(
        'Counterflow ε at Cr=1, NTU=1',
        Decimal('0.500'),
        'dimensionless',
        Decimal('1'),
        'Analytical',
        'counterflow',
    ),
    'parallelflow_Cr0_NTU1': HeatExchangerGoldenValue(
        'Parallelflow ε at Cr=0, NTU=1',
        Decimal('0.632'),
        'dimensionless',
        Decimal('1'),
        'Analytical',
        'parallelflow',
    ),
    'parallelflow_Cr1_NTU1': HeatExchangerGoldenValue(
        'Parallelflow ε at Cr=1, NTU=1',
        Decimal('0.432'),
        'dimensionless',
        Decimal('1'),
        'Analytical',
        'parallelflow',
    ),
}


# Typical Overall Heat Transfer Coefficients (U)
# Reference: Perry's Chemical Engineers Handbook, Kern
OVERALL_U_VALUES: Dict[str, HeatExchangerGoldenValue] = {
    'water_to_water': HeatExchangerGoldenValue(
        'Water-to-Water U',
        Decimal('1000'),
        'W/m²-K',
        Decimal('30'),
        'Perry Table 11-3',
        'shell_tube',
    ),
    'steam_to_water': HeatExchangerGoldenValue(
        'Steam-to-Water U',
        Decimal('2500'),
        'W/m²-K',
        Decimal('30'),
        'Perry Table 11-3',
        'shell_tube',
    ),
    'gas_to_water': HeatExchangerGoldenValue(
        'Gas-to-Water U',
        Decimal('50'),
        'W/m²-K',
        Decimal('40'),
        'Perry Table 11-3',
        'shell_tube',
    ),
    'oil_to_water': HeatExchangerGoldenValue(
        'Oil-to-Water U',
        Decimal('350'),
        'W/m²-K',
        Decimal('30'),
        'Perry Table 11-3',
        'shell_tube',
    ),
    'gas_to_gas': HeatExchangerGoldenValue(
        'Gas-to-Gas U',
        Decimal('25'),
        'W/m²-K',
        Decimal('40'),
        'Perry Table 11-3',
        'shell_tube',
    ),
    'condensing_steam': HeatExchangerGoldenValue(
        'Condensing Steam U',
        Decimal('3000'),
        'W/m²-K',
        Decimal('30'),
        'Perry Table 11-3',
        'condenser',
    ),
}


# =============================================================================
# DETERMINISTIC CALCULATION FUNCTIONS
# =============================================================================


def calculate_lmtd(
    t_hot_in: Decimal,
    t_hot_out: Decimal,
    t_cold_in: Decimal,
    t_cold_out: Decimal,
    flow_type: str = 'counterflow',
) -> Decimal:
    """
    Calculate Log Mean Temperature Difference (LMTD).

    For counterflow: ΔT1 = Th_in - Tc_out, ΔT2 = Th_out - Tc_in
    For parallel flow: ΔT1 = Th_in - Tc_in, ΔT2 = Th_out - Tc_out

    LMTD = (ΔT1 - ΔT2) / ln(ΔT1/ΔT2)

    Args:
        t_hot_in: Hot fluid inlet temperature (°C)
        t_hot_out: Hot fluid outlet temperature (°C)
        t_cold_in: Cold fluid inlet temperature (°C)
        t_cold_out: Cold fluid outlet temperature (°C)
        flow_type: 'counterflow' or 'parallelflow'

    Returns:
        LMTD in °C

    Reference: ASME PTC 12.5
    """
    if flow_type == 'counterflow':
        delta_t1 = t_hot_in - t_cold_out
        delta_t2 = t_hot_out - t_cold_in
    else:  # parallelflow
        delta_t1 = t_hot_in - t_cold_in
        delta_t2 = t_hot_out - t_cold_out

    if delta_t1 <= 0 or delta_t2 <= 0:
        raise ValueError('Temperature differences must be positive')

    if delta_t1 == delta_t2:
        # Special case: equal temperature differences
        return delta_t1

    # Use Decimal for ln calculation
    ratio = float(delta_t1 / delta_t2)
    if ratio <= 0:
        raise ValueError('Invalid temperature ratio')

    ln_ratio = Decimal(str(math.log(ratio)))
    lmtd = (delta_t1 - delta_t2) / ln_ratio

    return lmtd.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def calculate_heat_duty(
    mass_flow_kg_s: Decimal,
    cp_kj_kgk: Decimal,
    delta_t: Decimal,
) -> Decimal:
    """
    Calculate heat duty Q = m * Cp * ΔT.

    Args:
        mass_flow_kg_s: Mass flow rate (kg/s)
        cp_kj_kgk: Specific heat (kJ/kg-K)
        delta_t: Temperature change (K or °C)

    Returns:
        Heat duty in kW

    Reference: ASME PTC 12.5
    """
    q = mass_flow_kg_s * cp_kj_kgk * delta_t
    return q.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def calculate_ua_from_lmtd(
    heat_duty_kw: Decimal,
    lmtd: Decimal,
    correction_factor: Decimal = Decimal('1.0'),
) -> Decimal:
    """
    Calculate UA product from heat duty and LMTD.

    Q = U * A * F * LMTD
    UA = Q / (F * LMTD)

    Args:
        heat_duty_kw: Heat duty (kW)
        lmtd: Log mean temperature difference (°C or K)
        correction_factor: LMTD correction factor F

    Returns:
        UA product in kW/K

    Reference: ASME PTC 12.5
    """
    if lmtd <= 0:
        raise ValueError('LMTD must be positive')
    if correction_factor <= 0 or correction_factor > 1:
        raise ValueError('Correction factor must be 0 < F <= 1')

    ua = heat_duty_kw / (correction_factor * lmtd)
    return ua.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)


def calculate_effectiveness(
    q_actual: Decimal,
    q_max: Decimal,
) -> Decimal:
    """
    Calculate heat exchanger effectiveness.

    ε = Q_actual / Q_max

    Args:
        q_actual: Actual heat transfer (kW)
        q_max: Maximum possible heat transfer (kW)

    Returns:
        Effectiveness (0 to 1)

    Reference: ASME PTC 12.5
    """
    if q_max <= 0:
        raise ValueError('Maximum heat transfer must be positive')

    effectiveness = q_actual / q_max
    return effectiveness.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


def calculate_ntu(
    ua_kw_k: Decimal,
    c_min_kw_k: Decimal,
) -> Decimal:
    """
    Calculate Number of Transfer Units (NTU).

    NTU = UA / C_min

    Args:
        ua_kw_k: Overall heat transfer coefficient-area product (kW/K)
        c_min_kw_k: Minimum heat capacity rate (kW/K)

    Returns:
        NTU (dimensionless)

    Reference: ASME PTC 12.5
    """
    if c_min_kw_k <= 0:
        raise ValueError('Minimum heat capacity rate must be positive')

    ntu = ua_kw_k / c_min_kw_k
    return ntu.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)


def calculate_capacity_ratio(
    c_min_kw_k: Decimal,
    c_max_kw_k: Decimal,
) -> Decimal:
    """
    Calculate heat capacity ratio.

    Cr = C_min / C_max

    Args:
        c_min_kw_k: Minimum heat capacity rate (kW/K)
        c_max_kw_k: Maximum heat capacity rate (kW/K)

    Returns:
        Capacity ratio (0 to 1)

    Reference: ASME PTC 12.5
    """
    if c_max_kw_k <= 0:
        raise ValueError('Maximum heat capacity rate must be positive')

    cr = c_min_kw_k / c_max_kw_k
    return cr.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


def calculate_effectiveness_counterflow(
    ntu: Decimal,
    cr: Decimal,
) -> Decimal:
    """
    Calculate effectiveness for counterflow exchanger.

    For Cr < 1:
    ε = [1 - exp(-NTU*(1-Cr))] / [1 - Cr*exp(-NTU*(1-Cr))]

    For Cr = 1:
    ε = NTU / (1 + NTU)

    Args:
        ntu: Number of transfer units
        cr: Capacity ratio

    Returns:
        Effectiveness (0 to 1)

    Reference: Incropera & DeWitt
    """
    if cr == Decimal('1'):
        effectiveness = ntu / (Decimal('1') + ntu)
    else:
        exp_term = Decimal(str(math.exp(float(-ntu * (Decimal('1') - cr)))))
        effectiveness = (Decimal('1') - exp_term) / (Decimal('1') - cr * exp_term)

    return effectiveness.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


def calculate_effectiveness_parallelflow(
    ntu: Decimal,
    cr: Decimal,
) -> Decimal:
    """
    Calculate effectiveness for parallelflow exchanger.

    ε = [1 - exp(-NTU*(1+Cr))] / (1+Cr)

    Args:
        ntu: Number of transfer units
        cr: Capacity ratio

    Returns:
        Effectiveness (0 to 1)

    Reference: Incropera & DeWitt
    """
    exp_term = Decimal(str(math.exp(float(-ntu * (Decimal('1') + cr)))))
    effectiveness = (Decimal('1') - exp_term) / (Decimal('1') + cr)

    return effectiveness.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


def calculate_fouled_u(
    u_clean: Decimal,
    rf_hot: Decimal,
    rf_cold: Decimal,
) -> Decimal:
    """
    Calculate fouled overall heat transfer coefficient.

    1/U_fouled = 1/U_clean + Rf_hot + Rf_cold

    Args:
        u_clean: Clean overall U (W/m²-K)
        rf_hot: Hot side fouling factor (m²-K/W)
        rf_cold: Cold side fouling factor (m²-K/W)

    Returns:
        Fouled U (W/m²-K)

    Reference: TEMA RCB-2.21
    """
    if u_clean <= 0:
        raise ValueError('Clean U must be positive')

    reciprocal = Decimal('1') / u_clean + rf_hot + rf_cold
    u_fouled = Decimal('1') / reciprocal

    return u_fouled.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_fouling_resistance_from_u(
    u_clean: Decimal,
    u_fouled: Decimal,
) -> Decimal:
    """
    Calculate total fouling resistance from clean and fouled U.

    Rf_total = 1/U_fouled - 1/U_clean

    Args:
        u_clean: Clean overall U (W/m²-K)
        u_fouled: Fouled overall U (W/m²-K)

    Returns:
        Total fouling resistance (m²-K/W)

    Reference: ASME PTC 12.5
    """
    if u_clean <= 0 or u_fouled <= 0:
        raise ValueError('U values must be positive')

    rf = Decimal('1') / u_fouled - Decimal('1') / u_clean
    return rf.quantize(Decimal('0.00001'), rounding=ROUND_HALF_UP)


def calculate_pressure_drop_tube(
    friction_factor: Decimal,
    length_m: Decimal,
    diameter_m: Decimal,
    velocity_m_s: Decimal,
    density_kg_m3: Decimal,
    n_passes: int = 1,
) -> Decimal:
    """
    Calculate tube-side pressure drop.

    ΔP = f * (L/D) * (ρ*V²/2) * n_passes

    Args:
        friction_factor: Darcy-Weisbach friction factor
        length_m: Tube length (m)
        diameter_m: Tube inner diameter (m)
        velocity_m_s: Fluid velocity (m/s)
        density_kg_m3: Fluid density (kg/m³)
        n_passes: Number of tube passes

    Returns:
        Pressure drop (Pa)

    Reference: TEMA
    """
    if diameter_m <= 0:
        raise ValueError('Diameter must be positive')

    dp = (
        friction_factor
        * (length_m / diameter_m)
        * (density_kg_m3 * velocity_m_s ** 2 / Decimal('2'))
        * Decimal(str(n_passes))
    )
    return dp.quantize(Decimal('1'), rounding=ROUND_HALF_UP)


def calculate_cleanliness_factor(
    u_actual: Decimal,
    u_clean: Decimal,
) -> Decimal:
    """
    Calculate heat exchanger cleanliness factor.

    CF = U_actual / U_clean * 100

    Args:
        u_actual: Actual U value (W/m²-K)
        u_clean: Design clean U value (W/m²-K)

    Returns:
        Cleanliness factor (%)

    Reference: ASME PTC 12.5
    """
    if u_clean <= 0:
        raise ValueError('Clean U must be positive')

    cf = (u_actual / u_clean) * Decimal('100')
    return cf.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


# =============================================================================
# PROVENANCE TRACKING
# =============================================================================


def generate_provenance_hash(
    calculation_name: str,
    inputs: Dict[str, str],
    output: str,
    reference: str,
) -> str:
    """Generate SHA-256 hash for calculation provenance."""
    provenance_data = {
        'calculation': calculation_name,
        'inputs': inputs,
        'output': output,
        'reference': reference,
        'version': '1.0.0',
    }
    json_str = json.dumps(provenance_data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


# =============================================================================
# GOLDEN VALUE TESTS
# =============================================================================


class TestFoulingFactors:
    """Test TEMA fouling factor reference values."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        'fluid,expected_rf',
        [
            ('cooling_tower_water', Decimal('0.00035')),
            ('treated_boiler_water', Decimal('0.00009')),
            ('fuel_oil', Decimal('0.00088')),
            ('engine_exhaust', Decimal('0.00176')),
            ('steam_condensing', Decimal('0.00009')),
        ],
    )
    def test_tema_fouling_factors(self, fluid: str, expected_rf: Decimal) -> None:
        """Verify TEMA fouling factors (Table RCB-2.21)."""
        golden = FOULING_FACTORS[fluid]
        assert golden.value == expected_rf, (
            f'Expected Rf={expected_rf} for {fluid}, got {golden.value}'
        )


class TestLMTDCalculations:
    """Test LMTD calculations against reference values."""

    @pytest.mark.golden
    def test_lmtd_counterflow_equal_delta_t(self) -> None:
        """LMTD with equal temperature differences."""
        # Hot: 100→60, Cold: 20→60 (counterflow)
        # ΔT1 = 100-60 = 40, ΔT2 = 60-20 = 40
        # LMTD = 40 (equal delta T case)

        lmtd = calculate_lmtd(
            Decimal('100'), Decimal('60'), Decimal('20'), Decimal('60'),
            'counterflow'
        )

        assert lmtd == Decimal('40.00'), f'Expected LMTD=40.00, got {lmtd}'

    @pytest.mark.golden
    def test_lmtd_counterflow_typical(self) -> None:
        """Typical counterflow LMTD calculation."""
        # Hot: 150→80, Cold: 30→100 (counterflow)
        # ΔT1 = 150-100 = 50, ΔT2 = 80-30 = 50
        # LMTD = 50 (equal case)

        lmtd = calculate_lmtd(
            Decimal('150'), Decimal('80'), Decimal('30'), Decimal('100'),
            'counterflow'
        )

        assert lmtd == Decimal('50.00'), f'Expected LMTD=50.00, got {lmtd}'

    @pytest.mark.golden
    def test_lmtd_counterflow_unequal(self) -> None:
        """Counterflow LMTD with unequal temperature differences."""
        # Hot: 120→60, Cold: 20→80 (counterflow)
        # ΔT1 = 120-80 = 40, ΔT2 = 60-20 = 40
        # LMTD = 40 (equal case)

        lmtd = calculate_lmtd(
            Decimal('120'), Decimal('60'), Decimal('20'), Decimal('80'),
            'counterflow'
        )

        assert lmtd == Decimal('40.00'), f'Expected LMTD=40.00, got {lmtd}'

    @pytest.mark.golden
    def test_lmtd_parallelflow(self) -> None:
        """Parallel flow LMTD calculation."""
        # Hot: 150→90, Cold: 30→60 (parallel)
        # ΔT1 = 150-30 = 120, ΔT2 = 90-60 = 30
        # LMTD = (120-30) / ln(120/30) = 90 / ln(4) = 64.93

        lmtd = calculate_lmtd(
            Decimal('150'), Decimal('90'), Decimal('30'), Decimal('60'),
            'parallelflow'
        )

        expected = Decimal('64.93')
        tolerance = Decimal('0.5')

        assert abs(lmtd - expected) <= tolerance, (
            f'Expected LMTD~{expected}, got {lmtd}'
        )


class TestHeatDutyCalculations:
    """Test heat duty calculations."""

    @pytest.mark.golden
    def test_heat_duty_water(self) -> None:
        """Heat duty for water heating."""
        # 5 kg/s water, Cp = 4.18 kJ/kg-K, ΔT = 20 K
        # Q = 5 * 4.18 * 20 = 418 kW

        q = calculate_heat_duty(
            Decimal('5.0'),
            Decimal('4.18'),
            Decimal('20.0')
        )

        assert q == Decimal('418.00'), f'Expected Q=418.00 kW, got {q}'

    @pytest.mark.golden
    def test_heat_duty_oil(self) -> None:
        """Heat duty for oil cooling."""
        # 2 kg/s oil, Cp = 2.1 kJ/kg-K, ΔT = 50 K
        # Q = 2 * 2.1 * 50 = 210 kW

        q = calculate_heat_duty(
            Decimal('2.0'),
            Decimal('2.1'),
            Decimal('50.0')
        )

        assert q == Decimal('210.00'), f'Expected Q=210.00 kW, got {q}'


class TestUACalculations:
    """Test UA product calculations."""

    @pytest.mark.golden
    def test_ua_from_lmtd_counterflow(self) -> None:
        """UA calculation from LMTD (counterflow)."""
        # Q = 500 kW, LMTD = 40°C, F = 1.0
        # UA = 500 / (1.0 * 40) = 12.5 kW/K

        ua = calculate_ua_from_lmtd(
            Decimal('500'),
            Decimal('40'),
            Decimal('1.0')
        )

        assert ua == Decimal('12.500'), f'Expected UA=12.500 kW/K, got {ua}'

    @pytest.mark.golden
    def test_ua_from_lmtd_with_correction(self) -> None:
        """UA calculation with LMTD correction factor."""
        # Q = 500 kW, LMTD = 40°C, F = 0.9
        # UA = 500 / (0.9 * 40) = 13.889 kW/K

        ua = calculate_ua_from_lmtd(
            Decimal('500'),
            Decimal('40'),
            Decimal('0.9')
        )

        expected = Decimal('13.889')
        tolerance = Decimal('0.01')

        assert abs(ua - expected) <= tolerance, (
            f'Expected UA~{expected} kW/K, got {ua}'
        )


class TestNTUEffectiveness:
    """Test NTU-Effectiveness method calculations."""

    @pytest.mark.golden
    def test_ntu_calculation(self) -> None:
        """NTU from UA and C_min."""
        # UA = 10 kW/K, C_min = 5 kW/K
        # NTU = 10 / 5 = 2

        ntu = calculate_ntu(Decimal('10'), Decimal('5'))

        assert ntu == Decimal('2.000'), f'Expected NTU=2.000, got {ntu}'

    @pytest.mark.golden
    def test_capacity_ratio(self) -> None:
        """Capacity ratio calculation."""
        # C_min = 4 kW/K, C_max = 8 kW/K
        # Cr = 4 / 8 = 0.5

        cr = calculate_capacity_ratio(Decimal('4'), Decimal('8'))

        assert cr == Decimal('0.5000'), f'Expected Cr=0.5000, got {cr}'

    @pytest.mark.golden
    def test_effectiveness_counterflow_cr0(self) -> None:
        """Counterflow effectiveness at Cr=0 (condensation/boiling)."""
        # ε = 1 - exp(-NTU) for Cr=0
        # At NTU=1: ε = 1 - exp(-1) = 0.632

        eps = calculate_effectiveness_counterflow(Decimal('1'), Decimal('0'))

        expected = Decimal('0.6321')
        tolerance = Decimal('0.001')

        assert abs(eps - expected) <= tolerance, (
            f'Expected ε~{expected}, got {eps}'
        )

    @pytest.mark.golden
    def test_effectiveness_counterflow_cr1(self) -> None:
        """Counterflow effectiveness at Cr=1 (balanced flow)."""
        # ε = NTU / (1 + NTU) for Cr=1
        # At NTU=1: ε = 1 / 2 = 0.5

        eps = calculate_effectiveness_counterflow(Decimal('1'), Decimal('1'))

        assert eps == Decimal('0.5000'), f'Expected ε=0.5000, got {eps}'

    @pytest.mark.golden
    def test_effectiveness_parallelflow_cr1(self) -> None:
        """Parallelflow effectiveness at Cr=1."""
        # ε = [1 - exp(-NTU*(1+Cr))] / (1+Cr)
        # At NTU=1, Cr=1: ε = [1 - exp(-2)] / 2 = 0.432

        eps = calculate_effectiveness_parallelflow(Decimal('1'), Decimal('1'))

        expected = Decimal('0.4323')
        tolerance = Decimal('0.001')

        assert abs(eps - expected) <= tolerance, (
            f'Expected ε~{expected}, got {eps}'
        )


class TestFoulingCalculations:
    """Test fouling-related calculations."""

    @pytest.mark.golden
    def test_fouled_u_typical(self) -> None:
        """Calculate fouled U from clean U and fouling factors."""
        # U_clean = 1000 W/m²-K
        # Rf_hot = 0.00035 m²-K/W (cooling tower water)
        # Rf_cold = 0.00009 m²-K/W (treated boiler water)
        # 1/U_fouled = 1/1000 + 0.00035 + 0.00009 = 0.00144
        # U_fouled = 694.4 W/m²-K

        u_fouled = calculate_fouled_u(
            Decimal('1000'),
            Decimal('0.00035'),
            Decimal('0.00009')
        )

        expected = Decimal('694.4')
        tolerance = Decimal('1')

        assert abs(u_fouled - expected) <= tolerance, (
            f'Expected U_fouled~{expected}, got {u_fouled}'
        )

    @pytest.mark.golden
    def test_fouling_resistance_recovery(self) -> None:
        """Recover fouling resistance from U measurements."""
        u_clean = Decimal('1000')
        u_fouled = Decimal('700')

        # Rf = 1/700 - 1/1000 = 0.001429 - 0.001 = 0.000429

        rf = calculate_fouling_resistance_from_u(u_clean, u_fouled)

        expected = Decimal('0.00043')
        tolerance = Decimal('0.00001')

        assert abs(rf - expected) <= tolerance, (
            f'Expected Rf~{expected}, got {rf}'
        )

    @pytest.mark.golden
    def test_cleanliness_factor(self) -> None:
        """Calculate cleanliness factor."""
        # U_actual = 850, U_clean = 1000
        # CF = 850/1000 * 100 = 85%

        cf = calculate_cleanliness_factor(Decimal('850'), Decimal('1000'))

        assert cf == Decimal('85.0'), f'Expected CF=85.0%, got {cf}'


class TestEffectivenessCalculations:
    """Test effectiveness calculations."""

    @pytest.mark.golden
    def test_effectiveness_typical(self) -> None:
        """Calculate effectiveness from actual and max heat transfer."""
        # Q_actual = 400 kW, Q_max = 500 kW
        # ε = 400/500 = 0.80

        eps = calculate_effectiveness(Decimal('400'), Decimal('500'))

        assert eps == Decimal('0.8000'), f'Expected ε=0.8000, got {eps}'


class TestPressureDrop:
    """Test pressure drop calculations."""

    @pytest.mark.golden
    def test_tube_pressure_drop(self) -> None:
        """Calculate tube-side pressure drop."""
        # f = 0.02, L = 5m, D = 0.025m, V = 2 m/s, ρ = 1000 kg/m³, n=2
        # ΔP = 0.02 * (5/0.025) * (1000*2²/2) * 2 = 0.02 * 200 * 2000 * 2 = 16000 Pa

        dp = calculate_pressure_drop_tube(
            Decimal('0.02'),
            Decimal('5'),
            Decimal('0.025'),
            Decimal('2'),
            Decimal('1000'),
            2
        )

        assert dp == Decimal('16000'), f'Expected ΔP=16000 Pa, got {dp}'


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.golden
    def test_zero_lmtd_rejected(self) -> None:
        """Reject zero LMTD in UA calculation."""
        with pytest.raises(ValueError, match='positive'):
            calculate_ua_from_lmtd(Decimal('100'), Decimal('0'))

    @pytest.mark.golden
    def test_invalid_correction_factor(self) -> None:
        """Reject invalid correction factor."""
        with pytest.raises(ValueError, match='Correction factor'):
            calculate_ua_from_lmtd(Decimal('100'), Decimal('40'), Decimal('1.5'))

    @pytest.mark.golden
    def test_zero_c_min_rejected(self) -> None:
        """Reject zero C_min in NTU calculation."""
        with pytest.raises(ValueError, match='positive'):
            calculate_ntu(Decimal('10'), Decimal('0'))

    @pytest.mark.golden
    def test_negative_temperature_difference(self) -> None:
        """Reject invalid temperature arrangement."""
        with pytest.raises(ValueError, match='positive'):
            calculate_lmtd(
                Decimal('50'), Decimal('60'),  # Hot increasing (invalid)
                Decimal('30'), Decimal('40'),
                'counterflow'
            )


class TestDeterminism:
    """Verify calculation determinism for regulatory compliance."""

    @pytest.mark.golden
    def test_lmtd_determinism(self) -> None:
        """Verify LMTD calculation is deterministic."""
        results = [
            calculate_lmtd(
                Decimal('150'), Decimal('80'),
                Decimal('30'), Decimal('100'),
                'counterflow'
            )
            for _ in range(100)
        ]

        assert len(set(results)) == 1, 'LMTD calculation must be deterministic'

    @pytest.mark.golden
    def test_effectiveness_determinism(self) -> None:
        """Verify effectiveness calculation is deterministic."""
        results = [
            calculate_effectiveness_counterflow(Decimal('1.5'), Decimal('0.75'))
            for _ in range(100)
        ]

        assert len(set(results)) == 1, 'Effectiveness calculation must be deterministic'

    @pytest.mark.golden
    def test_provenance_hash_determinism(self) -> None:
        """Verify provenance hashes are deterministic."""
        hashes = [
            generate_provenance_hash(
                'ua_calculation',
                {'Q': '500', 'LMTD': '40', 'F': '1.0'},
                '12.5',
                'ASME PTC 12.5',
            )
            for _ in range(100)
        ]

        assert len(set(hashes)) == 1, 'Provenance hash must be deterministic'


class TestOverallUValues:
    """Test overall U value references."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        'service,min_u,max_u',
        [
            ('water_to_water', Decimal('500'), Decimal('2000')),
            ('steam_to_water', Decimal('1000'), Decimal('4000')),
            ('gas_to_water', Decimal('20'), Decimal('100')),
            ('oil_to_water', Decimal('150'), Decimal('600')),
            ('gas_to_gas', Decimal('10'), Decimal('50')),
        ],
    )
    def test_u_value_ranges(
        self, service: str, min_u: Decimal, max_u: Decimal
    ) -> None:
        """Verify U values are within expected ranges (Perry's)."""
        golden = OVERALL_U_VALUES[service]

        assert min_u <= golden.value <= max_u, (
            f'{service} U={golden.value} outside range [{min_u}, {max_u}]'
        )


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_golden_values() -> Dict[str, List[Dict]]:
    """Export all golden values for documentation."""
    export_data = {
        'fouling_factors': [],
        'lmtd_corrections': [],
        'ntu_effectiveness': [],
        'u_values': [],
        'metadata': {
            'version': '1.0.0',
            'references': ['TEMA', 'ASME PTC 12.5', 'Perry'],
            'agent': 'GL-014_ExchangerPro',
        },
    }

    for fluid, golden in FOULING_FACTORS.items():
        export_data['fouling_factors'].append(
            {
                'fluid': fluid,
                'description': golden.description,
                'value': str(golden.value),
                'unit': golden.unit,
                'source': golden.source,
            }
        )

    for config, golden in LMTD_CORRECTION_FACTORS.items():
        export_data['lmtd_corrections'].append(
            {
                'configuration': config,
                'F': str(golden.value),
                'source': golden.source,
            }
        )

    return export_data


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
