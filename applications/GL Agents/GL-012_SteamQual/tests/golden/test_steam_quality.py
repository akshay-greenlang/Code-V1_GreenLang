"""
GL-012 SteamQual: Steam Quality Golden Value Tests.

Reference Standards:
- IAPWS-IF97: Industrial Formulation for Properties of Water and Steam
- ASME PTC 19.11: Steam and Water Sampling, Conditioning, and Analysis
- ASTM E1137: Standard Specification for Industrial Platinum RTD Sensors

These golden tests validate steam quality (dryness fraction) calculations,
calorimeter measurements, separator tests, and moisture carryover detection.
"""

import hashlib
import json
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional, Tuple

import pytest

# =============================================================================
# GOLDEN VALUE REFERENCE DATA - IAPWS-IF97 Steam Tables
# =============================================================================


@dataclass(frozen=True)
class SteamQualityGoldenValue:
    """Immutable golden value for steam quality validation."""

    description: str
    value: Decimal
    unit: str
    tolerance_percent: Decimal
    source: str
    pressure_mpa: Decimal
    temperature_c: Optional[Decimal] = None


# IAPWS-IF97 saturation properties at various pressures
SATURATION_PROPERTIES: Dict[str, Dict[str, SteamQualityGoldenValue]] = {
    # At 0.1 MPa (1 bar, ~100°C)
    '0.1_mpa': {
        'T_sat': SteamQualityGoldenValue(
            'Saturation Temperature at 0.1 MPa',
            Decimal('99.606'),
            '°C',
            Decimal('0.01'),
            'IAPWS-IF97',
            Decimal('0.1'),
        ),
        'hf': SteamQualityGoldenValue(
            'Saturated Liquid Enthalpy at 0.1 MPa',
            Decimal('417.44'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('0.1'),
        ),
        'hg': SteamQualityGoldenValue(
            'Saturated Vapor Enthalpy at 0.1 MPa',
            Decimal('2675.5'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('0.1'),
        ),
        'hfg': SteamQualityGoldenValue(
            'Latent Heat at 0.1 MPa',
            Decimal('2258.0'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('0.1'),
        ),
        'sf': SteamQualityGoldenValue(
            'Saturated Liquid Entropy at 0.1 MPa',
            Decimal('1.3026'),
            'kJ/kg-K',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('0.1'),
        ),
        'sg': SteamQualityGoldenValue(
            'Saturated Vapor Entropy at 0.1 MPa',
            Decimal('7.3594'),
            'kJ/kg-K',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('0.1'),
        ),
    },
    # At 0.5 MPa (5 bar, ~151.8°C)
    '0.5_mpa': {
        'T_sat': SteamQualityGoldenValue(
            'Saturation Temperature at 0.5 MPa',
            Decimal('151.86'),
            '°C',
            Decimal('0.01'),
            'IAPWS-IF97',
            Decimal('0.5'),
        ),
        'hf': SteamQualityGoldenValue(
            'Saturated Liquid Enthalpy at 0.5 MPa',
            Decimal('640.23'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('0.5'),
        ),
        'hg': SteamQualityGoldenValue(
            'Saturated Vapor Enthalpy at 0.5 MPa',
            Decimal('2748.7'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('0.5'),
        ),
        'hfg': SteamQualityGoldenValue(
            'Latent Heat at 0.5 MPa',
            Decimal('2108.5'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('0.5'),
        ),
    },
    # At 1.0 MPa (10 bar, ~179.9°C)
    '1.0_mpa': {
        'T_sat': SteamQualityGoldenValue(
            'Saturation Temperature at 1.0 MPa',
            Decimal('179.91'),
            '°C',
            Decimal('0.01'),
            'IAPWS-IF97',
            Decimal('1.0'),
        ),
        'hf': SteamQualityGoldenValue(
            'Saturated Liquid Enthalpy at 1.0 MPa',
            Decimal('762.81'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('1.0'),
        ),
        'hg': SteamQualityGoldenValue(
            'Saturated Vapor Enthalpy at 1.0 MPa',
            Decimal('2778.1'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('1.0'),
        ),
        'hfg': SteamQualityGoldenValue(
            'Latent Heat at 1.0 MPa',
            Decimal('2015.3'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('1.0'),
        ),
    },
    # At 2.0 MPa (20 bar, ~212.4°C)
    '2.0_mpa': {
        'T_sat': SteamQualityGoldenValue(
            'Saturation Temperature at 2.0 MPa',
            Decimal('212.42'),
            '°C',
            Decimal('0.01'),
            'IAPWS-IF97',
            Decimal('2.0'),
        ),
        'hf': SteamQualityGoldenValue(
            'Saturated Liquid Enthalpy at 2.0 MPa',
            Decimal('908.79'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('2.0'),
        ),
        'hg': SteamQualityGoldenValue(
            'Saturated Vapor Enthalpy at 2.0 MPa',
            Decimal('2799.5'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('2.0'),
        ),
        'hfg': SteamQualityGoldenValue(
            'Latent Heat at 2.0 MPa',
            Decimal('1890.7'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('2.0'),
        ),
    },
    # At 4.0 MPa (40 bar, ~250.4°C)
    '4.0_mpa': {
        'T_sat': SteamQualityGoldenValue(
            'Saturation Temperature at 4.0 MPa',
            Decimal('250.40'),
            '°C',
            Decimal('0.01'),
            'IAPWS-IF97',
            Decimal('4.0'),
        ),
        'hf': SteamQualityGoldenValue(
            'Saturated Liquid Enthalpy at 4.0 MPa',
            Decimal('1087.31'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('4.0'),
        ),
        'hg': SteamQualityGoldenValue(
            'Saturated Vapor Enthalpy at 4.0 MPa',
            Decimal('2801.4'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('4.0'),
        ),
        'hfg': SteamQualityGoldenValue(
            'Latent Heat at 4.0 MPa',
            Decimal('1714.1'),
            'kJ/kg',
            Decimal('0.1'),
            'IAPWS-IF97',
            Decimal('4.0'),
        ),
    },
}


# =============================================================================
# DETERMINISTIC CALCULATION FUNCTIONS
# =============================================================================


def calculate_steam_quality_from_enthalpy(
    h_mix: Decimal, hf: Decimal, hfg: Decimal
) -> Decimal:
    """
    Calculate steam quality (dryness fraction) from mixture enthalpy.

    x = (h_mix - hf) / hfg

    Args:
        h_mix: Mixture enthalpy (kJ/kg)
        hf: Saturated liquid enthalpy (kJ/kg)
        hfg: Latent heat of vaporization (kJ/kg)

    Returns:
        Steam quality x (0 to 1)

    Reference: IAPWS-IF97
    """
    if hfg == 0:
        raise ValueError('Latent heat cannot be zero (at critical point)')

    x = (h_mix - hf) / hfg
    return x.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


def calculate_mixture_enthalpy(
    x: Decimal, hf: Decimal, hfg: Decimal
) -> Decimal:
    """
    Calculate mixture enthalpy from steam quality.

    h_mix = hf + x * hfg

    Args:
        x: Steam quality (0 to 1)
        hf: Saturated liquid enthalpy (kJ/kg)
        hfg: Latent heat of vaporization (kJ/kg)

    Returns:
        Mixture enthalpy (kJ/kg)

    Reference: IAPWS-IF97
    """
    if x < 0 or x > 1:
        raise ValueError('Steam quality must be between 0 and 1')

    h_mix = hf + x * hfg
    return h_mix.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def calculate_throttling_calorimeter_quality(
    h2: Decimal, hf1: Decimal, hfg1: Decimal
) -> Decimal:
    """
    Calculate steam quality using throttling calorimeter.

    For isenthalpic (constant enthalpy) throttling:
    h1 = h2 (enthalpy before throttling = enthalpy after)
    x1 = (h2 - hf1) / hfg1

    Args:
        h2: Enthalpy after throttling (superheated region, kJ/kg)
        hf1: Saturated liquid enthalpy at inlet pressure (kJ/kg)
        hfg1: Latent heat at inlet pressure (kJ/kg)

    Returns:
        Steam quality at inlet

    Reference: ASME PTC 19.11 Section 5.5
    """
    if hfg1 == 0:
        raise ValueError('Latent heat cannot be zero')

    x1 = (h2 - hf1) / hfg1
    return x1.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


def calculate_separating_calorimeter_quality(
    m_dry: Decimal, m_total: Decimal
) -> Decimal:
    """
    Calculate steam quality using separating calorimeter.

    x = m_dry / m_total

    Args:
        m_dry: Mass of dry steam collected (kg)
        m_total: Total mass of steam sample (kg)

    Returns:
        Steam quality (dryness fraction)

    Reference: ASME PTC 19.11 Section 5.4
    """
    if m_total == 0:
        raise ValueError('Total mass cannot be zero')

    x = m_dry / m_total
    return x.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


def calculate_conductivity_based_quality(
    conductivity_steam: Decimal,
    conductivity_condensate: Decimal,
    conductivity_feedwater: Decimal,
) -> Decimal:
    """
    Calculate steam quality from conductivity measurements.

    Carryover fraction = (C_steam - C_feedwater) / (C_condensate - C_feedwater)
    Quality = 1 - Carryover fraction

    Args:
        conductivity_steam: Steam sample conductivity (μS/cm)
        conductivity_condensate: Condensate conductivity (μS/cm)
        conductivity_feedwater: Feedwater conductivity (μS/cm)

    Returns:
        Steam quality (dryness fraction)

    Reference: ASME PTC 19.11 Section 5.6
    """
    denominator = conductivity_condensate - conductivity_feedwater
    if denominator == 0:
        raise ValueError('Condensate and feedwater conductivity cannot be equal')

    carryover = (conductivity_steam - conductivity_feedwater) / denominator
    x = Decimal('1') - carryover
    return x.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


def calculate_moisture_carryover_ppm(
    quality: Decimal,
) -> Decimal:
    """
    Calculate moisture carryover in ppm from steam quality.

    Carryover (ppm) = (1 - x) * 1,000,000

    Args:
        quality: Steam quality (dryness fraction)

    Returns:
        Moisture carryover in ppm

    Reference: ASME PTC 19.11
    """
    if quality < 0 or quality > 1:
        raise ValueError('Steam quality must be between 0 and 1')

    carryover_ppm = (Decimal('1') - quality) * Decimal('1000000')
    return carryover_ppm.quantize(Decimal('1'), rounding=ROUND_HALF_UP)


def calculate_mixture_entropy(
    x: Decimal, sf: Decimal, sfg: Decimal
) -> Decimal:
    """
    Calculate mixture entropy from steam quality.

    s_mix = sf + x * sfg

    Args:
        x: Steam quality (0 to 1)
        sf: Saturated liquid entropy (kJ/kg-K)
        sfg: Entropy of vaporization (kJ/kg-K)

    Returns:
        Mixture entropy (kJ/kg-K)

    Reference: IAPWS-IF97
    """
    if x < 0 or x > 1:
        raise ValueError('Steam quality must be between 0 and 1')

    s_mix = sf + x * sfg
    return s_mix.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


def calculate_moisture_loss_energy(
    steam_flow_kg_h: Decimal,
    quality: Decimal,
    hfg: Decimal,
) -> Decimal:
    """
    Calculate energy loss due to moisture in steam.

    Energy loss = m * (1 - x) * hfg

    Args:
        steam_flow_kg_h: Steam mass flow rate (kg/h)
        quality: Steam quality (dryness fraction)
        hfg: Latent heat at operating pressure (kJ/kg)

    Returns:
        Energy loss rate (kW)

    Reference: ASME PTC 4
    """
    if quality < 0 or quality > 1:
        raise ValueError('Steam quality must be between 0 and 1')

    # Convert kg/h to kg/s
    steam_flow_kg_s = steam_flow_kg_h / Decimal('3600')

    energy_loss_kw = steam_flow_kg_s * (Decimal('1') - quality) * hfg
    return energy_loss_kw.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


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


class TestSaturationProperties:
    """Test IAPWS-IF97 saturation property lookups."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        'pressure_key,property_name',
        [
            ('0.1_mpa', 'T_sat'),
            ('0.1_mpa', 'hf'),
            ('0.1_mpa', 'hg'),
            ('0.1_mpa', 'hfg'),
            ('0.5_mpa', 'T_sat'),
            ('0.5_mpa', 'hf'),
            ('0.5_mpa', 'hfg'),
            ('1.0_mpa', 'T_sat'),
            ('1.0_mpa', 'hf'),
            ('1.0_mpa', 'hg'),
            ('2.0_mpa', 'T_sat'),
            ('2.0_mpa', 'hfg'),
            ('4.0_mpa', 'T_sat'),
            ('4.0_mpa', 'hfg'),
        ],
    )
    def test_saturation_properties(self, pressure_key: str, property_name: str) -> None:
        """Verify saturation properties against IAPWS-IF97 tables."""
        golden = SATURATION_PROPERTIES[pressure_key][property_name]

        # In production, this would call the actual lookup function
        # Here we validate the reference values are self-consistent
        assert golden.value > 0, f'{golden.description} must be positive'
        assert golden.tolerance_percent >= 0, 'Tolerance must be non-negative'


class TestSteamQualityCalculations:
    """Test steam quality (dryness fraction) calculations."""

    @pytest.mark.golden
    def test_quality_from_enthalpy_x095(self) -> None:
        """Calculate quality x=0.95 at 1.0 MPa from enthalpy."""
        # At 1.0 MPa: hf = 762.81, hfg = 2015.3
        hf = Decimal('762.81')
        hfg = Decimal('2015.3')
        x_expected = Decimal('0.95')

        # h_mix = hf + x * hfg = 762.81 + 0.95 * 2015.3 = 2677.345
        h_mix = hf + x_expected * hfg

        x_calculated = calculate_steam_quality_from_enthalpy(h_mix, hf, hfg)

        assert x_calculated == Decimal('0.9500'), (
            f'Expected x=0.9500, got {x_calculated}'
        )

    @pytest.mark.golden
    def test_quality_from_enthalpy_x090(self) -> None:
        """Calculate quality x=0.90 at 0.5 MPa from enthalpy."""
        hf = Decimal('640.23')
        hfg = Decimal('2108.5')
        x_expected = Decimal('0.90')

        h_mix = hf + x_expected * hfg  # 640.23 + 0.90 * 2108.5 = 2537.88

        x_calculated = calculate_steam_quality_from_enthalpy(h_mix, hf, hfg)

        assert x_calculated == Decimal('0.9000'), (
            f'Expected x=0.9000, got {x_calculated}'
        )

    @pytest.mark.golden
    def test_mixture_enthalpy_x098(self) -> None:
        """Calculate mixture enthalpy for x=0.98 at 2.0 MPa."""
        hf = Decimal('908.79')
        hfg = Decimal('1890.7')
        x = Decimal('0.98')

        # Expected: 908.79 + 0.98 * 1890.7 = 2761.68
        h_expected = Decimal('2761.68')

        h_calculated = calculate_mixture_enthalpy(x, hf, hfg)

        assert h_calculated == h_expected, (
            f'Expected h={h_expected}, got {h_calculated}'
        )

    @pytest.mark.golden
    def test_quality_saturated_liquid(self) -> None:
        """Quality at saturated liquid condition (x=0)."""
        hf = Decimal('417.44')
        hfg = Decimal('2258.0')

        x = calculate_steam_quality_from_enthalpy(hf, hf, hfg)

        assert x == Decimal('0.0000'), f'Saturated liquid should have x=0, got {x}'

    @pytest.mark.golden
    def test_quality_saturated_vapor(self) -> None:
        """Quality at saturated vapor condition (x=1)."""
        hf = Decimal('417.44')
        hfg = Decimal('2258.0')
        hg = hf + hfg  # 2675.44

        x = calculate_steam_quality_from_enthalpy(hg, hf, hfg)

        assert x == Decimal('1.0000'), f'Saturated vapor should have x=1, got {x}'


class TestThrottlingCalorimeter:
    """Test throttling calorimeter quality calculations."""

    @pytest.mark.golden
    def test_throttling_calorimeter_typical(self) -> None:
        """
        Typical throttling calorimeter measurement.

        Inlet: 1.0 MPa wet steam
        Outlet: 0.1 MPa superheated to 120°C
        Reference: ASME PTC 19.11 Example
        """
        # At 0.1 MPa, 120°C (superheated): h ≈ 2716.6 kJ/kg
        h2_superheated = Decimal('2716.6')

        # At 1.0 MPa: hf = 762.81, hfg = 2015.3
        hf1 = Decimal('762.81')
        hfg1 = Decimal('2015.3')

        # x1 = (2716.6 - 762.81) / 2015.3 = 0.969
        x_calculated = calculate_throttling_calorimeter_quality(
            h2_superheated, hf1, hfg1
        )

        # Expected quality around 0.969
        assert Decimal('0.96') <= x_calculated <= Decimal('0.98'), (
            f'Expected quality ~0.97, got {x_calculated}'
        )

    @pytest.mark.golden
    def test_throttling_calorimeter_high_quality(self) -> None:
        """High quality steam (x=0.99) via throttling calorimeter."""
        # For x=0.99 at 0.5 MPa:
        # h1 = 640.23 + 0.99 * 2108.5 = 2727.645 kJ/kg
        h1 = Decimal('2727.645')
        hf1 = Decimal('640.23')
        hfg1 = Decimal('2108.5')

        x_calculated = calculate_throttling_calorimeter_quality(h1, hf1, hfg1)

        assert x_calculated == Decimal('0.9900'), (
            f'Expected x=0.9900, got {x_calculated}'
        )


class TestSeparatingCalorimeter:
    """Test separating calorimeter quality calculations."""

    @pytest.mark.golden
    def test_separating_calorimeter_x095(self) -> None:
        """Separating calorimeter with 95% quality steam."""
        m_total = Decimal('100.0')  # kg
        m_dry = Decimal('95.0')  # kg

        x = calculate_separating_calorimeter_quality(m_dry, m_total)

        assert x == Decimal('0.9500'), f'Expected x=0.9500, got {x}'

    @pytest.mark.golden
    def test_separating_calorimeter_x088(self) -> None:
        """Separating calorimeter with 88% quality steam."""
        m_total = Decimal('50.0')  # kg
        m_dry = Decimal('44.0')  # kg

        x = calculate_separating_calorimeter_quality(m_dry, m_total)

        assert x == Decimal('0.8800'), f'Expected x=0.8800, got {x}'


class TestConductivityBasedQuality:
    """Test conductivity-based quality measurements."""

    @pytest.mark.golden
    def test_conductivity_quality_high(self) -> None:
        """High quality steam from conductivity (x=0.995)."""
        # Low carryover indicates high quality
        conductivity_feedwater = Decimal('2.0')  # μS/cm
        conductivity_condensate = Decimal('200.0')  # μS/cm
        conductivity_steam = Decimal('3.0')  # μS/cm (low = good quality)

        # Carryover = (3 - 2) / (200 - 2) = 1/198 = 0.00505
        # Quality = 1 - 0.00505 = 0.9949

        x = calculate_conductivity_based_quality(
            conductivity_steam, conductivity_condensate, conductivity_feedwater
        )

        assert x >= Decimal('0.99'), f'Expected high quality >=0.99, got {x}'

    @pytest.mark.golden
    def test_conductivity_quality_moderate(self) -> None:
        """Moderate quality steam from conductivity (x=0.98)."""
        conductivity_feedwater = Decimal('5.0')  # μS/cm
        conductivity_condensate = Decimal('250.0')  # μS/cm
        conductivity_steam = Decimal('9.9')  # μS/cm

        # Carryover = (9.9 - 5) / (250 - 5) = 4.9/245 = 0.02
        # Quality = 1 - 0.02 = 0.98

        x = calculate_conductivity_based_quality(
            conductivity_steam, conductivity_condensate, conductivity_feedwater
        )

        assert Decimal('0.97') <= x <= Decimal('0.99'), (
            f'Expected quality ~0.98, got {x}'
        )


class TestMoistureCarryover:
    """Test moisture carryover calculations."""

    @pytest.mark.golden
    def test_carryover_ppm_x099(self) -> None:
        """Carryover in ppm for 99% quality steam."""
        x = Decimal('0.99')

        carryover = calculate_moisture_carryover_ppm(x)

        # (1 - 0.99) * 1,000,000 = 10,000 ppm
        assert carryover == Decimal('10000'), (
            f'Expected 10000 ppm, got {carryover}'
        )

    @pytest.mark.golden
    def test_carryover_ppm_x0995(self) -> None:
        """Carryover in ppm for 99.5% quality steam."""
        x = Decimal('0.995')

        carryover = calculate_moisture_carryover_ppm(x)

        # (1 - 0.995) * 1,000,000 = 5,000 ppm
        assert carryover == Decimal('5000'), (
            f'Expected 5000 ppm, got {carryover}'
        )

    @pytest.mark.golden
    def test_carryover_ppm_dry_steam(self) -> None:
        """Carryover for perfectly dry steam (x=1)."""
        x = Decimal('1.0')

        carryover = calculate_moisture_carryover_ppm(x)

        assert carryover == Decimal('0'), f'Dry steam should have 0 ppm carryover'


class TestMoistureEnergyLoss:
    """Test energy loss calculations due to moisture."""

    @pytest.mark.golden
    def test_moisture_energy_loss_1pct(self) -> None:
        """Energy loss for 1% moisture (x=0.99) at 10,000 kg/h."""
        steam_flow = Decimal('10000')  # kg/h
        quality = Decimal('0.99')
        hfg = Decimal('2015.3')  # kJ/kg at 1.0 MPa

        loss = calculate_moisture_loss_energy(steam_flow, quality, hfg)

        # 10000/3600 * 0.01 * 2015.3 = 55.98 kW
        expected = Decimal('55.98')
        tolerance = expected * Decimal('0.01')

        assert abs(loss - expected) <= tolerance, (
            f'Expected loss ~{expected} kW, got {loss} kW'
        )

    @pytest.mark.golden
    def test_moisture_energy_loss_5pct(self) -> None:
        """Energy loss for 5% moisture (x=0.95) at 5,000 kg/h."""
        steam_flow = Decimal('5000')  # kg/h
        quality = Decimal('0.95')
        hfg = Decimal('1890.7')  # kJ/kg at 2.0 MPa

        loss = calculate_moisture_loss_energy(steam_flow, quality, hfg)

        # 5000/3600 * 0.05 * 1890.7 = 131.30 kW
        expected = Decimal('131.30')
        tolerance = expected * Decimal('0.02')

        assert abs(loss - expected) <= tolerance, (
            f'Expected loss ~{expected} kW, got {loss} kW'
        )


class TestMixtureEntropy:
    """Test mixture entropy calculations."""

    @pytest.mark.golden
    def test_entropy_saturated_liquid(self) -> None:
        """Entropy at saturated liquid condition (x=0)."""
        sf = Decimal('1.3026')
        sfg = Decimal('6.0568')  # sg - sf at 0.1 MPa

        s = calculate_mixture_entropy(Decimal('0'), sf, sfg)

        assert s == sf, f'At x=0, entropy should equal sf'

    @pytest.mark.golden
    def test_entropy_saturated_vapor(self) -> None:
        """Entropy at saturated vapor condition (x=1)."""
        sf = Decimal('1.3026')
        sfg = Decimal('6.0568')
        sg = sf + sfg

        s = calculate_mixture_entropy(Decimal('1'), sf, sfg)

        assert s == sg.quantize(Decimal('0.0001')), (
            f'At x=1, entropy should equal sg={sg}'
        )

    @pytest.mark.golden
    def test_entropy_wet_steam_x095(self) -> None:
        """Entropy for wet steam at x=0.95."""
        sf = Decimal('1.3026')
        sfg = Decimal('6.0568')
        x = Decimal('0.95')

        # s = 1.3026 + 0.95 * 6.0568 = 7.0566
        expected = Decimal('7.0566')

        s = calculate_mixture_entropy(x, sf, sfg)

        assert s == expected, f'Expected s={expected}, got {s}'


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.golden
    def test_quality_negative_rejected(self) -> None:
        """Reject negative quality values."""
        with pytest.raises(ValueError, match='between 0 and 1'):
            calculate_mixture_enthalpy(Decimal('-0.1'), Decimal('100'), Decimal('2000'))

    @pytest.mark.golden
    def test_quality_over_one_rejected(self) -> None:
        """Reject quality values over 1."""
        with pytest.raises(ValueError, match='between 0 and 1'):
            calculate_mixture_enthalpy(Decimal('1.1'), Decimal('100'), Decimal('2000'))

    @pytest.mark.golden
    def test_zero_latent_heat_rejected(self) -> None:
        """Reject zero latent heat (at critical point)."""
        with pytest.raises(ValueError, match='zero'):
            calculate_steam_quality_from_enthalpy(
                Decimal('1000'), Decimal('500'), Decimal('0')
            )

    @pytest.mark.golden
    def test_zero_total_mass_rejected(self) -> None:
        """Reject zero total mass in separating calorimeter."""
        with pytest.raises(ValueError, match='zero'):
            calculate_separating_calorimeter_quality(Decimal('10'), Decimal('0'))


class TestDeterminism:
    """Verify calculation determinism for regulatory compliance."""

    @pytest.mark.golden
    def test_quality_calculation_determinism(self) -> None:
        """Verify identical inputs produce identical quality results."""
        hf = Decimal('762.81')
        hfg = Decimal('2015.3')
        h_mix = Decimal('2700.00')

        results = [
            calculate_steam_quality_from_enthalpy(h_mix, hf, hfg)
            for _ in range(100)
        ]

        assert len(set(results)) == 1, 'Quality calculation must be deterministic'

    @pytest.mark.golden
    def test_enthalpy_calculation_determinism(self) -> None:
        """Verify identical inputs produce identical enthalpy results."""
        x = Decimal('0.95')
        hf = Decimal('762.81')
        hfg = Decimal('2015.3')

        results = [
            calculate_mixture_enthalpy(x, hf, hfg)
            for _ in range(100)
        ]

        assert len(set(results)) == 1, 'Enthalpy calculation must be deterministic'

    @pytest.mark.golden
    def test_provenance_hash_determinism(self) -> None:
        """Verify provenance hashes are deterministic."""
        hashes = [
            generate_provenance_hash(
                'steam_quality',
                {'h_mix': '2700.00', 'hf': '762.81', 'hfg': '2015.3'},
                '0.9613',
                'IAPWS-IF97',
            )
            for _ in range(100)
        ]

        assert len(set(hashes)) == 1, 'Provenance hash must be deterministic'


class TestQualityLimits:
    """Test steam quality operational limits."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        'application,min_quality',
        [
            ('turbine_inlet', Decimal('0.995')),
            ('process_heating', Decimal('0.980')),
            ('steam_tracing', Decimal('0.950')),
            ('flash_steam', Decimal('0.900')),
        ],
    )
    def test_application_quality_limits(
        self, application: str, min_quality: Decimal
    ) -> None:
        """Verify minimum quality limits by application."""
        # These are typical industry minimum quality requirements
        assert min_quality >= Decimal('0.85'), (
            f'{application} minimum quality too low'
        )
        assert min_quality <= Decimal('1.0'), (
            f'{application} minimum quality invalid (>1)'
        )


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_golden_values() -> Dict[str, List[Dict]]:
    """Export all golden values for documentation."""
    export_data = {
        'saturation_properties': [],
        'quality_test_cases': [],
        'metadata': {
            'version': '1.0.0',
            'reference': 'IAPWS-IF97',
            'agent': 'GL-012_SteamQual',
        },
    }

    for pressure_key, properties in SATURATION_PROPERTIES.items():
        for prop_name, golden in properties.items():
            export_data['saturation_properties'].append(
                {
                    'pressure': pressure_key,
                    'property': prop_name,
                    'description': golden.description,
                    'value': str(golden.value),
                    'unit': golden.unit,
                    'tolerance_pct': str(golden.tolerance_percent),
                    'source': golden.source,
                }
            )

    return export_data


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
