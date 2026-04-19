"""
GL-016 Waterguard: Water Treatment Chemistry Golden Value Tests.

Reference Standards:
- ASME: Consensus on Operating Practices for Boiler Feedwater/Steam Purity
- ASTM D1066: Standard Methods for Sampling Steam
- ASTM D1293: Standard Test Methods for pH of Water
- Langelier Saturation Index (LSI): AWWA Standard
- Ryznar Stability Index (RSI): AWWA Technical References
- CTI: Cooling Technology Institute Guidelines

These golden tests validate water chemistry calculations, corrosion indices,
cycles of concentration, blowdown optimization, and treatment dosing.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional, Tuple

import pytest

# =============================================================================
# GOLDEN VALUE REFERENCE DATA - WATER TREATMENT STANDARDS
# =============================================================================


@dataclass(frozen=True)
class WaterChemistryGoldenValue:
    """Immutable golden value for water chemistry validation."""

    description: str
    value: Decimal
    unit: str
    tolerance_percent: Decimal
    source: str
    application: str


# ASME Boiler Water Guidelines - Industrial Watertube Boilers
# Reference: ASME Consensus on Operating Practices
BOILER_WATER_LIMITS: Dict[str, Dict[str, WaterChemistryGoldenValue]] = {
    # Low pressure: 0-300 psig (0-2.07 MPa)
    '0_300_psig': {
        'total_dissolved_solids': WaterChemistryGoldenValue(
            'TDS Limit 0-300 psig',
            Decimal('3500'),
            'ppm',
            Decimal('5'),
            'ASME',
            'boiler_water',
        ),
        'total_alkalinity': WaterChemistryGoldenValue(
            'Total Alkalinity 0-300 psig',
            Decimal('700'),
            'ppm as CaCO3',
            Decimal('10'),
            'ASME',
            'boiler_water',
        ),
        'silica': WaterChemistryGoldenValue(
            'Silica Limit 0-300 psig',
            Decimal('150'),
            'ppm as SiO2',
            Decimal('10'),
            'ASME',
            'boiler_water',
        ),
    },
    # Medium pressure: 301-450 psig (2.08-3.10 MPa)
    '301_450_psig': {
        'total_dissolved_solids': WaterChemistryGoldenValue(
            'TDS Limit 301-450 psig',
            Decimal('3000'),
            'ppm',
            Decimal('5'),
            'ASME',
            'boiler_water',
        ),
        'total_alkalinity': WaterChemistryGoldenValue(
            'Total Alkalinity 301-450 psig',
            Decimal('600'),
            'ppm as CaCO3',
            Decimal('10'),
            'ASME',
            'boiler_water',
        ),
        'silica': WaterChemistryGoldenValue(
            'Silica Limit 301-450 psig',
            Decimal('90'),
            'ppm as SiO2',
            Decimal('10'),
            'ASME',
            'boiler_water',
        ),
    },
    # Medium-high pressure: 451-600 psig (3.11-4.14 MPa)
    '451_600_psig': {
        'total_dissolved_solids': WaterChemistryGoldenValue(
            'TDS Limit 451-600 psig',
            Decimal('2500'),
            'ppm',
            Decimal('5'),
            'ASME',
            'boiler_water',
        ),
        'total_alkalinity': WaterChemistryGoldenValue(
            'Total Alkalinity 451-600 psig',
            Decimal('500'),
            'ppm as CaCO3',
            Decimal('10'),
            'ASME',
            'boiler_water',
        ),
        'silica': WaterChemistryGoldenValue(
            'Silica Limit 451-600 psig',
            Decimal('40'),
            'ppm as SiO2',
            Decimal('10'),
            'ASME',
            'boiler_water',
        ),
    },
    # High pressure: 601-750 psig (4.15-5.17 MPa)
    '601_750_psig': {
        'total_dissolved_solids': WaterChemistryGoldenValue(
            'TDS Limit 601-750 psig',
            Decimal('2000'),
            'ppm',
            Decimal('5'),
            'ASME',
            'boiler_water',
        ),
        'silica': WaterChemistryGoldenValue(
            'Silica Limit 601-750 psig',
            Decimal('30'),
            'ppm as SiO2',
            Decimal('10'),
            'ASME',
            'boiler_water',
        ),
    },
    # Very high pressure: 751-900 psig (5.18-6.21 MPa)
    '751_900_psig': {
        'total_dissolved_solids': WaterChemistryGoldenValue(
            'TDS Limit 751-900 psig',
            Decimal('1500'),
            'ppm',
            Decimal('5'),
            'ASME',
            'boiler_water',
        ),
        'silica': WaterChemistryGoldenValue(
            'Silica Limit 751-900 psig',
            Decimal('20'),
            'ppm as SiO2',
            Decimal('10'),
            'ASME',
            'boiler_water',
        ),
    },
}


# Feedwater Quality Guidelines
FEEDWATER_LIMITS: Dict[str, WaterChemistryGoldenValue] = {
    'dissolved_oxygen': WaterChemistryGoldenValue(
        'Dissolved Oxygen Limit',
        Decimal('0.007'),
        'ppm',
        Decimal('20'),
        'ASME',
        'feedwater',
    ),
    'total_iron': WaterChemistryGoldenValue(
        'Total Iron Limit',
        Decimal('0.1'),
        'ppm as Fe',
        Decimal('20'),
        'ASME',
        'feedwater',
    ),
    'total_copper': WaterChemistryGoldenValue(
        'Total Copper Limit',
        Decimal('0.05'),
        'ppm as Cu',
        Decimal('20'),
        'ASME',
        'feedwater',
    ),
    'total_hardness': WaterChemistryGoldenValue(
        'Total Hardness Limit',
        Decimal('0.3'),
        'ppm as CaCO3',
        Decimal('20'),
        'ASME',
        'feedwater',
    ),
    'ph_min': WaterChemistryGoldenValue(
        'pH Minimum',
        Decimal('8.3'),
        'pH units',
        Decimal('1'),
        'ASME',
        'feedwater',
    ),
    'ph_max': WaterChemistryGoldenValue(
        'pH Maximum',
        Decimal('10.0'),
        'pH units',
        Decimal('1'),
        'ASME',
        'feedwater',
    ),
}


# Cooling Water Guidelines
# Reference: CTI, ASHRAE
COOLING_WATER_LIMITS: Dict[str, WaterChemistryGoldenValue] = {
    'ph_min': WaterChemistryGoldenValue(
        'Cooling Water pH Min',
        Decimal('7.0'),
        'pH units',
        Decimal('2'),
        'CTI',
        'cooling_water',
    ),
    'ph_max': WaterChemistryGoldenValue(
        'Cooling Water pH Max',
        Decimal('9.0'),
        'pH units',
        Decimal('2'),
        'CTI',
        'cooling_water',
    ),
    'conductivity_max': WaterChemistryGoldenValue(
        'Conductivity Max',
        Decimal('3000'),
        'μS/cm',
        Decimal('10'),
        'CTI',
        'cooling_water',
    ),
    'chloride_max': WaterChemistryGoldenValue(
        'Chloride Max (carbon steel)',
        Decimal('250'),
        'ppm as Cl',
        Decimal('10'),
        'CTI',
        'cooling_water',
    ),
    'lsi_min': WaterChemistryGoldenValue(
        'LSI Minimum (scaling tendency)',
        Decimal('-0.5'),
        'dimensionless',
        Decimal('20'),
        'AWWA',
        'cooling_water',
    ),
    'lsi_max': WaterChemistryGoldenValue(
        'LSI Maximum (scaling tendency)',
        Decimal('0.5'),
        'dimensionless',
        Decimal('20'),
        'AWWA',
        'cooling_water',
    ),
}


# Equilibrium Constants for LSI Calculation at Various Temperatures
# Reference: Standard Methods for Water and Wastewater
PH_SATURATION_CONSTANTS: Dict[str, WaterChemistryGoldenValue] = {
    'pKs_25c': WaterChemistryGoldenValue(
        'pKs at 25°C',
        Decimal('8.34'),
        'dimensionless',
        Decimal('1'),
        'Standard Methods',
        'calculation',
    ),
    'pKs_40c': WaterChemistryGoldenValue(
        'pKs at 40°C',
        Decimal('8.14'),
        'dimensionless',
        Decimal('1'),
        'Standard Methods',
        'calculation',
    ),
    'pKs_60c': WaterChemistryGoldenValue(
        'pKs at 60°C',
        Decimal('7.89'),
        'dimensionless',
        Decimal('1'),
        'Standard Methods',
        'calculation',
    ),
}


# =============================================================================
# DETERMINISTIC CALCULATION FUNCTIONS
# =============================================================================


def calculate_langelier_saturation_index(
    ph: Decimal,
    temperature_c: Decimal,
    calcium_hardness_ppm: Decimal,
    total_alkalinity_ppm: Decimal,
    tds_ppm: Decimal,
) -> Decimal:
    """
    Calculate Langelier Saturation Index (LSI).

    LSI = pH - pHs
    where pHs = pK2 - pKs + pCa + pAlk

    Interpretation:
    LSI > 0: Scaling tendency (CaCO3 supersaturated)
    LSI = 0: Balanced (saturated)
    LSI < 0: Corrosive tendency (CaCO3 undersaturated)

    Args:
        ph: Measured pH
        temperature_c: Temperature (°C)
        calcium_hardness_ppm: Calcium hardness as CaCO3 (ppm)
        total_alkalinity_ppm: Total alkalinity as CaCO3 (ppm)
        tds_ppm: Total dissolved solids (ppm)

    Returns:
        Langelier Saturation Index

    Reference: AWWA Standard, Langelier (1936)
    """
    if calcium_hardness_ppm <= 0 or total_alkalinity_ppm <= 0:
        raise ValueError('Hardness and alkalinity must be positive')

    # Calculate ionic strength factor
    # μ = 2.5e-5 * TDS (approximate)
    ionic_strength = Decimal('2.5e-5') * tds_ppm

    # Temperature factor (A)
    t_k = float(temperature_c + Decimal('273.15'))
    a = Decimal(str((math.log10(t_k) - 1.0) * 0.01706 + 2.1605))

    # TDS factor (B)
    b = Decimal(str(9.70 + 2.5 * float(ionic_strength) ** 0.5))

    # pCa = -log10([Ca] in mol/L)
    # [Ca] (mol/L) = (ppm CaCO3) / (100000 mg/g) * (1 mol/40 g)
    ca_mol = calcium_hardness_ppm / Decimal('100000') / Decimal('40') * Decimal('1000')
    p_ca = Decimal(str(-math.log10(float(ca_mol))))

    # pAlk = -log10([Alk] in eq/L)
    alk_eq = total_alkalinity_ppm / Decimal('50000')  # 50 g/eq for CaCO3
    p_alk = Decimal(str(-math.log10(float(alk_eq))))

    # Calculate pHs (saturation pH)
    phs = a + b - p_ca - p_alk

    # LSI = pH - pHs
    lsi = ph - phs

    return lsi.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def calculate_ryznar_stability_index(
    ph: Decimal,
    phs: Decimal,
) -> Decimal:
    """
    Calculate Ryznar Stability Index (RSI).

    RSI = 2*pHs - pH

    Interpretation:
    RSI < 6.0: Heavy scale formation
    RSI 6.0-6.5: Light scale
    RSI 6.5-7.0: Balanced
    RSI 7.0-7.5: Light corrosion
    RSI 7.5-9.0: Moderate corrosion
    RSI > 9.0: Heavy corrosion

    Args:
        ph: Measured pH
        phs: Saturation pH (calculated)

    Returns:
        Ryznar Stability Index

    Reference: Ryznar (1944)
    """
    rsi = Decimal('2') * phs - ph
    return rsi.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def calculate_cycles_of_concentration(
    makeup_conductivity: Decimal,
    blowdown_conductivity: Decimal,
) -> Decimal:
    """
    Calculate cycles of concentration from conductivity.

    CoC = Conductivity_blowdown / Conductivity_makeup

    Args:
        makeup_conductivity: Makeup water conductivity (μS/cm)
        blowdown_conductivity: Blowdown water conductivity (μS/cm)

    Returns:
        Cycles of concentration

    Reference: CTI Guidelines
    """
    if makeup_conductivity <= 0:
        raise ValueError('Makeup conductivity must be positive')

    coc = blowdown_conductivity / makeup_conductivity
    return coc.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_blowdown_rate(
    evaporation_rate_kg_h: Decimal,
    cycles_of_concentration: Decimal,
) -> Decimal:
    """
    Calculate required blowdown rate.

    BD = E / (CoC - 1)

    where:
    BD = Blowdown rate (kg/h)
    E = Evaporation rate (kg/h)
    CoC = Cycles of concentration

    Args:
        evaporation_rate_kg_h: Evaporation rate (kg/h)
        cycles_of_concentration: Target cycles

    Returns:
        Blowdown rate (kg/h)

    Reference: ASHRAE
    """
    if cycles_of_concentration <= 1:
        raise ValueError('Cycles of concentration must be > 1')

    bd = evaporation_rate_kg_h / (cycles_of_concentration - Decimal('1'))
    return bd.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_makeup_rate(
    evaporation_rate_kg_h: Decimal,
    blowdown_rate_kg_h: Decimal,
    drift_rate_kg_h: Decimal = Decimal('0'),
) -> Decimal:
    """
    Calculate required makeup water rate.

    Makeup = Evaporation + Blowdown + Drift

    Args:
        evaporation_rate_kg_h: Evaporation losses (kg/h)
        blowdown_rate_kg_h: Blowdown rate (kg/h)
        drift_rate_kg_h: Drift losses (kg/h)

    Returns:
        Makeup rate (kg/h)

    Reference: CTI
    """
    makeup = evaporation_rate_kg_h + blowdown_rate_kg_h + drift_rate_kg_h
    return makeup.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_boiler_blowdown_percentage(
    feedwater_tds_ppm: Decimal,
    max_boiler_water_tds_ppm: Decimal,
) -> Decimal:
    """
    Calculate boiler blowdown as percentage of feedwater.

    BD% = (TDS_feedwater / TDS_boiler_max) * 100

    Args:
        feedwater_tds_ppm: Feedwater TDS (ppm)
        max_boiler_water_tds_ppm: Maximum allowed boiler water TDS (ppm)

    Returns:
        Blowdown percentage

    Reference: ASME
    """
    if max_boiler_water_tds_ppm <= 0:
        raise ValueError('Maximum TDS must be positive')

    bd_pct = (feedwater_tds_ppm / max_boiler_water_tds_ppm) * Decimal('100')
    return bd_pct.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_chemical_dosage(
    water_flow_m3_h: Decimal,
    target_concentration_ppm: Decimal,
    chemical_concentration_pct: Decimal,
) -> Decimal:
    """
    Calculate chemical dosing rate.

    Dose (L/h) = Flow * Target_ppm / (Chemical_% * 10000)

    Args:
        water_flow_m3_h: Water flow rate (m³/h)
        target_concentration_ppm: Target chemical concentration (ppm)
        chemical_concentration_pct: Chemical product concentration (%)

    Returns:
        Dosing rate (L/h)

    Reference: General water treatment practice
    """
    if chemical_concentration_pct <= 0:
        raise ValueError('Chemical concentration must be positive')

    # Convert m³/h to L/h (1 m³ = 1000 L)
    flow_l_h = water_flow_m3_h * Decimal('1000')

    dose = (flow_l_h * target_concentration_ppm) / (
        chemical_concentration_pct * Decimal('10000')
    )
    return dose.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def calculate_conductivity_from_tds(
    tds_ppm: Decimal,
    conversion_factor: Decimal = Decimal('0.65'),
) -> Decimal:
    """
    Estimate conductivity from TDS.

    Conductivity (μS/cm) ≈ TDS (ppm) / conversion_factor

    Typical factor: 0.55-0.75 depending on ionic composition

    Args:
        tds_ppm: Total dissolved solids (ppm)
        conversion_factor: TDS to conductivity conversion factor

    Returns:
        Estimated conductivity (μS/cm)

    Reference: Standard Methods
    """
    if conversion_factor <= 0:
        raise ValueError('Conversion factor must be positive')

    conductivity = tds_ppm / conversion_factor
    return conductivity.quantize(Decimal('1'), rounding=ROUND_HALF_UP)


def calculate_hardness_removal_efficiency(
    inlet_hardness_ppm: Decimal,
    outlet_hardness_ppm: Decimal,
) -> Decimal:
    """
    Calculate softener/demin efficiency.

    Efficiency = (Inlet - Outlet) / Inlet * 100

    Args:
        inlet_hardness_ppm: Inlet hardness (ppm as CaCO3)
        outlet_hardness_ppm: Outlet hardness (ppm as CaCO3)

    Returns:
        Removal efficiency (%)

    Reference: Water treatment practice
    """
    if inlet_hardness_ppm <= 0:
        raise ValueError('Inlet hardness must be positive')

    efficiency = (
        (inlet_hardness_ppm - outlet_hardness_ppm) / inlet_hardness_ppm
    ) * Decimal('100')
    return efficiency.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_silica_solubility(
    temperature_c: Decimal,
    ph: Decimal,
) -> Decimal:
    """
    Estimate silica solubility at temperature and pH.

    Approximate formula based on amorphous silica:
    Solubility (ppm) ≈ 10^(2.9 - 900/T_K + 0.05*pH)

    Args:
        temperature_c: Temperature (°C)
        ph: pH

    Returns:
        Estimated silica solubility (ppm as SiO2)

    Reference: Marshall & Warakomski (1980)
    """
    t_k = float(temperature_c + Decimal('273.15'))

    log_solubility = 2.9 - 900.0 / t_k + 0.05 * float(ph)
    solubility = Decimal(str(10 ** log_solubility))

    return solubility.quantize(Decimal('1'), rounding=ROUND_HALF_UP)


def calculate_oxygen_scavenger_dosage(
    dissolved_oxygen_ppm: Decimal,
    scavenger_ratio: Decimal = Decimal('10'),
    safety_factor: Decimal = Decimal('1.5'),
) -> Decimal:
    """
    Calculate oxygen scavenger (sulfite) dosage.

    Stoichiometric: 7.88 ppm Na2SO3 per 1 ppm O2
    Typical ratio: 8-10:1 with safety factor

    Args:
        dissolved_oxygen_ppm: Dissolved oxygen (ppm)
        scavenger_ratio: Scavenger to O2 ratio
        safety_factor: Additional safety margin

    Returns:
        Scavenger dosage (ppm as Na2SO3)

    Reference: Nalco Water Handbook
    """
    dosage = dissolved_oxygen_ppm * scavenger_ratio * safety_factor
    return dosage.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_phosphate_dosage(
    target_po4_ppm: Decimal,
    water_volume_m3: Decimal,
    product_concentration_pct: Decimal,
) -> Decimal:
    """
    Calculate phosphate treatment dosage.

    Dose (kg) = Volume (m³) * Target (ppm) / (Product% * 10)

    Args:
        target_po4_ppm: Target PO4 concentration (ppm)
        water_volume_m3: Water volume (m³)
        product_concentration_pct: Product concentration (% as P2O5 or PO4)

    Returns:
        Product dosage (kg)

    Reference: Boiler water treatment practice
    """
    if product_concentration_pct <= 0:
        raise ValueError('Product concentration must be positive')

    dosage = (
        water_volume_m3 * target_po4_ppm
    ) / (product_concentration_pct * Decimal('10'))

    return dosage.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


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


class TestBoilerWaterLimits:
    """Test ASME boiler water limit values."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        'pressure_range,limit_type,expected',
        [
            ('0_300_psig', 'total_dissolved_solids', Decimal('3500')),
            ('0_300_psig', 'silica', Decimal('150')),
            ('301_450_psig', 'total_dissolved_solids', Decimal('3000')),
            ('451_600_psig', 'total_dissolved_solids', Decimal('2500')),
            ('601_750_psig', 'silica', Decimal('30')),
            ('751_900_psig', 'total_dissolved_solids', Decimal('1500')),
        ],
    )
    def test_boiler_water_limits(
        self, pressure_range: str, limit_type: str, expected: Decimal
    ) -> None:
        """Verify ASME boiler water limits."""
        golden = BOILER_WATER_LIMITS[pressure_range][limit_type]
        assert golden.value == expected, (
            f'Expected {expected} for {limit_type} at {pressure_range}'
        )

    @pytest.mark.golden
    def test_tds_limits_decrease_with_pressure(self) -> None:
        """TDS limits decrease as pressure increases."""
        tds_0_300 = BOILER_WATER_LIMITS['0_300_psig']['total_dissolved_solids'].value
        tds_301_450 = BOILER_WATER_LIMITS['301_450_psig']['total_dissolved_solids'].value
        tds_451_600 = BOILER_WATER_LIMITS['451_600_psig']['total_dissolved_solids'].value
        tds_601_750 = BOILER_WATER_LIMITS['601_750_psig']['total_dissolved_solids'].value
        tds_751_900 = BOILER_WATER_LIMITS['751_900_psig']['total_dissolved_solids'].value

        assert tds_0_300 > tds_301_450 > tds_451_600 > tds_601_750 > tds_751_900, (
            'TDS limits must decrease with increasing pressure'
        )


class TestFeedwaterLimits:
    """Test feedwater quality limits."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        'parameter,expected',
        [
            ('dissolved_oxygen', Decimal('0.007')),
            ('total_iron', Decimal('0.1')),
            ('total_hardness', Decimal('0.3')),
        ],
    )
    def test_feedwater_limits(self, parameter: str, expected: Decimal) -> None:
        """Verify feedwater quality limits."""
        golden = FEEDWATER_LIMITS[parameter]
        assert golden.value == expected, f'Expected {expected} for {parameter}'


class TestLangelierIndex:
    """Test Langelier Saturation Index calculations."""

    @pytest.mark.golden
    def test_lsi_balanced_water(self) -> None:
        """LSI for balanced cooling water near 0."""
        # Typical balanced cooling water
        lsi = calculate_langelier_saturation_index(
            ph=Decimal('8.2'),
            temperature_c=Decimal('35'),
            calcium_hardness_ppm=Decimal('200'),
            total_alkalinity_ppm=Decimal('150'),
            tds_ppm=Decimal('500'),
        )

        # Should be near 0 (-0.5 to +0.5)
        assert Decimal('-0.5') <= lsi <= Decimal('0.5'), (
            f'Balanced water LSI should be near 0, got {lsi}'
        )

    @pytest.mark.golden
    def test_lsi_scaling_water(self) -> None:
        """LSI for scaling-prone water (high hardness, high pH)."""
        lsi = calculate_langelier_saturation_index(
            ph=Decimal('9.0'),
            temperature_c=Decimal('40'),
            calcium_hardness_ppm=Decimal('400'),
            total_alkalinity_ppm=Decimal('300'),
            tds_ppm=Decimal('800'),
        )

        # Should be positive (scaling tendency)
        assert lsi > Decimal('0'), f'High hardness water should have positive LSI'

    @pytest.mark.golden
    def test_lsi_corrosive_water(self) -> None:
        """LSI for corrosive water (low hardness, low pH)."""
        lsi = calculate_langelier_saturation_index(
            ph=Decimal('6.5'),
            temperature_c=Decimal('25'),
            calcium_hardness_ppm=Decimal('50'),
            total_alkalinity_ppm=Decimal('40'),
            tds_ppm=Decimal('200'),
        )

        # Should be negative (corrosive tendency)
        assert lsi < Decimal('0'), f'Low hardness water should have negative LSI'


class TestCyclesOfConcentration:
    """Test cycles of concentration calculations."""

    @pytest.mark.golden
    def test_coc_from_conductivity(self) -> None:
        """Calculate CoC from conductivity measurements."""
        # Makeup: 500 μS/cm, Blowdown: 2500 μS/cm
        # CoC = 2500 / 500 = 5.0

        coc = calculate_cycles_of_concentration(
            Decimal('500'),
            Decimal('2500')
        )

        assert coc == Decimal('5.0'), f'Expected CoC=5.0, got {coc}'

    @pytest.mark.golden
    def test_typical_coc_range(self) -> None:
        """Verify typical CoC is in practical range (3-6)."""
        coc = calculate_cycles_of_concentration(
            Decimal('400'),
            Decimal('1600')
        )

        # 1600/400 = 4.0
        assert Decimal('3') <= coc <= Decimal('6'), (
            f'Typical CoC should be 3-6, got {coc}'
        )


class TestBlowdownCalculations:
    """Test blowdown rate calculations."""

    @pytest.mark.golden
    def test_cooling_tower_blowdown(self) -> None:
        """Calculate cooling tower blowdown rate."""
        # Evaporation = 1000 kg/h, CoC = 5
        # BD = 1000 / (5-1) = 250 kg/h

        bd = calculate_blowdown_rate(Decimal('1000'), Decimal('5'))

        assert bd == Decimal('250.0'), f'Expected BD=250.0 kg/h, got {bd}'

    @pytest.mark.golden
    def test_makeup_rate(self) -> None:
        """Calculate total makeup rate."""
        # Evap = 1000, BD = 250, Drift = 10
        # Makeup = 1000 + 250 + 10 = 1260 kg/h

        makeup = calculate_makeup_rate(
            Decimal('1000'),
            Decimal('250'),
            Decimal('10')
        )

        assert makeup == Decimal('1260.0'), f'Expected makeup=1260.0 kg/h, got {makeup}'

    @pytest.mark.golden
    def test_boiler_blowdown_percentage(self) -> None:
        """Calculate boiler blowdown percentage."""
        # Feedwater TDS = 50 ppm, Max boiler TDS = 2500 ppm
        # BD% = 50/2500 * 100 = 2.0%

        bd_pct = calculate_boiler_blowdown_percentage(
            Decimal('50'),
            Decimal('2500')
        )

        assert bd_pct == Decimal('2.0'), f'Expected BD=2.0%, got {bd_pct}'


class TestChemicalDosing:
    """Test chemical dosing calculations."""

    @pytest.mark.golden
    def test_chemical_dosage(self) -> None:
        """Calculate chemical product dosing rate."""
        # Flow = 100 m³/h, Target = 10 ppm, Product = 25%
        # Dose = 100*1000*10 / (25*10000) = 4 L/h

        dose = calculate_chemical_dosage(
            Decimal('100'),
            Decimal('10'),
            Decimal('25')
        )

        assert dose == Decimal('4.00'), f'Expected dose=4.00 L/h, got {dose}'

    @pytest.mark.golden
    def test_oxygen_scavenger_dosage(self) -> None:
        """Calculate oxygen scavenger (sulfite) dosage."""
        # DO = 0.1 ppm, ratio = 10, SF = 1.5
        # Dose = 0.1 * 10 * 1.5 = 1.5 ppm

        dose = calculate_oxygen_scavenger_dosage(
            Decimal('0.1'),
            Decimal('10'),
            Decimal('1.5')
        )

        assert dose == Decimal('1.5'), f'Expected dose=1.5 ppm, got {dose}'

    @pytest.mark.golden
    def test_phosphate_dosage(self) -> None:
        """Calculate phosphate treatment dosage."""
        # Target = 10 ppm, Volume = 50 m³, Product = 50%
        # Dose = 50 * 10 / (50 * 10) = 1.0 kg

        dose = calculate_phosphate_dosage(
            Decimal('10'),
            Decimal('50'),
            Decimal('50')
        )

        assert dose == Decimal('1.00'), f'Expected dose=1.00 kg, got {dose}'


class TestConductivityTDS:
    """Test conductivity/TDS conversions."""

    @pytest.mark.golden
    def test_conductivity_from_tds(self) -> None:
        """Estimate conductivity from TDS."""
        # TDS = 500 ppm, factor = 0.65
        # Conductivity = 500 / 0.65 = 769 μS/cm

        conductivity = calculate_conductivity_from_tds(
            Decimal('500'),
            Decimal('0.65')
        )

        expected = Decimal('769')
        tolerance = Decimal('5')

        assert abs(conductivity - expected) <= tolerance, (
            f'Expected ~{expected} μS/cm, got {conductivity}'
        )


class TestSoftenerEfficiency:
    """Test water softener efficiency calculations."""

    @pytest.mark.golden
    def test_hardness_removal_efficiency(self) -> None:
        """Calculate softener removal efficiency."""
        # Inlet = 200 ppm, Outlet = 2 ppm
        # Efficiency = (200-2)/200 * 100 = 99%

        efficiency = calculate_hardness_removal_efficiency(
            Decimal('200'),
            Decimal('2')
        )

        assert efficiency == Decimal('99.0'), f'Expected 99.0%, got {efficiency}'

    @pytest.mark.golden
    def test_perfect_softener(self) -> None:
        """Perfect softener (0 outlet hardness)."""
        efficiency = calculate_hardness_removal_efficiency(
            Decimal('150'),
            Decimal('0')
        )

        assert efficiency == Decimal('100.0'), 'Perfect removal = 100%'


class TestSilicaSolubility:
    """Test silica solubility estimates."""

    @pytest.mark.golden
    def test_silica_solubility_25c(self) -> None:
        """Silica solubility at 25°C, pH 7."""
        solubility = calculate_silica_solubility(Decimal('25'), Decimal('7'))

        # Should be approximately 100-150 ppm
        assert Decimal('80') < solubility < Decimal('180'), (
            f'25°C solubility {solubility} out of expected range'
        )

    @pytest.mark.golden
    def test_silica_solubility_increases_with_temp(self) -> None:
        """Silica solubility increases with temperature."""
        sol_25 = calculate_silica_solubility(Decimal('25'), Decimal('7'))
        sol_50 = calculate_silica_solubility(Decimal('50'), Decimal('7'))
        sol_100 = calculate_silica_solubility(Decimal('100'), Decimal('7'))

        assert sol_25 < sol_50 < sol_100, (
            'Silica solubility must increase with temperature'
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.golden
    def test_zero_hardness_rejected(self) -> None:
        """Reject zero calcium hardness in LSI."""
        with pytest.raises(ValueError, match='positive'):
            calculate_langelier_saturation_index(
                Decimal('8.0'), Decimal('25'),
                Decimal('0'), Decimal('100'), Decimal('500')
            )

    @pytest.mark.golden
    def test_coc_less_than_one_rejected(self) -> None:
        """Reject CoC <= 1 in blowdown calculation."""
        with pytest.raises(ValueError, match='> 1'):
            calculate_blowdown_rate(Decimal('1000'), Decimal('1'))

    @pytest.mark.golden
    def test_zero_inlet_hardness_rejected(self) -> None:
        """Reject zero inlet hardness in efficiency calc."""
        with pytest.raises(ValueError, match='positive'):
            calculate_hardness_removal_efficiency(Decimal('0'), Decimal('0'))


class TestDeterminism:
    """Verify calculation determinism for regulatory compliance."""

    @pytest.mark.golden
    def test_lsi_determinism(self) -> None:
        """Verify LSI calculation is deterministic."""
        results = [
            calculate_langelier_saturation_index(
                Decimal('8.0'), Decimal('30'),
                Decimal('200'), Decimal('150'), Decimal('500')
            )
            for _ in range(100)
        ]

        assert len(set(results)) == 1, 'LSI calculation must be deterministic'

    @pytest.mark.golden
    def test_blowdown_determinism(self) -> None:
        """Verify blowdown calculation is deterministic."""
        results = [
            calculate_blowdown_rate(Decimal('1000'), Decimal('5'))
            for _ in range(100)
        ]

        assert len(set(results)) == 1, 'Blowdown calculation must be deterministic'

    @pytest.mark.golden
    def test_provenance_hash_determinism(self) -> None:
        """Verify provenance hashes are deterministic."""
        hashes = [
            generate_provenance_hash(
                'lsi_calculation',
                {'pH': '8.0', 'temp': '30', 'Ca': '200'},
                '-0.15',
                'AWWA',
            )
            for _ in range(100)
        ]

        assert len(set(hashes)) == 1, 'Provenance hash must be deterministic'


class TestCoolingWaterLimits:
    """Test cooling water limit values."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        'parameter,min_val,max_val',
        [
            ('ph_min', Decimal('6.5'), Decimal('7.5')),
            ('ph_max', Decimal('8.5'), Decimal('9.5')),
            ('conductivity_max', Decimal('2500'), Decimal('5000')),
        ],
    )
    def test_cooling_water_limits(
        self, parameter: str, min_val: Decimal, max_val: Decimal
    ) -> None:
        """Verify cooling water limits are in expected ranges."""
        golden = COOLING_WATER_LIMITS[parameter]

        assert min_val <= golden.value <= max_val, (
            f'{parameter}={golden.value} outside [{min_val}, {max_val}]'
        )


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_golden_values() -> Dict[str, List[Dict]]:
    """Export all golden values for documentation."""
    export_data = {
        'boiler_water_limits': [],
        'feedwater_limits': [],
        'cooling_water_limits': [],
        'metadata': {
            'version': '1.0.0',
            'references': ['ASME', 'CTI', 'AWWA', 'Standard Methods'],
            'agent': 'GL-016_Waterguard',
        },
    }

    for pressure_range, limits in BOILER_WATER_LIMITS.items():
        for param, golden in limits.items():
            export_data['boiler_water_limits'].append(
                {
                    'pressure_range': pressure_range,
                    'parameter': param,
                    'value': str(golden.value),
                    'unit': golden.unit,
                    'source': golden.source,
                }
            )

    for param, golden in FEEDWATER_LIMITS.items():
        export_data['feedwater_limits'].append(
            {
                'parameter': param,
                'value': str(golden.value),
                'unit': golden.unit,
                'source': golden.source,
            }
        )

    for param, golden in COOLING_WATER_LIMITS.items():
        export_data['cooling_water_limits'].append(
            {
                'parameter': param,
                'value': str(golden.value),
                'unit': golden.unit,
                'source': golden.source,
            }
        )

    return export_data


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
