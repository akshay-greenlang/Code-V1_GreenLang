# -*- coding: utf-8 -*-
"""
Fuel Properties Golden Value Tests

Comprehensive test suite validating FuelCraft agent calculations against
EPA, EIA, and industry reference data for fuel properties and blending.

Reference Documents:
- EPA 40 CFR Part 98, Table C-1 (Default Emission Factors)
- EIA (Energy Information Administration) Fuel Data
- ASTM D3588 (Natural Gas Heating Value)
- API Technical Data Book

Author: GL-CalculatorEngineer
"""

import pytest
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from enum import Enum


# ==============================================================================
# FUEL PROPERTIES GOLDEN VALUES
# ==============================================================================

@dataclass(frozen=True)
class FuelGoldenValue:
    """Fuel property reference value with regulatory citation."""
    name: str
    value: Decimal
    unit: str
    tolerance_percent: Decimal
    source: str
    regulation: str = ''


class FuelType(Enum):
    """Standard fuel classifications."""
    NATURAL_GAS = 'natural_gas'
    FUEL_OIL_2 = 'fuel_oil_2'
    FUEL_OIL_6 = 'fuel_oil_6'
    COAL_BITUMINOUS = 'coal_bituminous'
    COAL_SUBBITUMINOUS = 'coal_subbituminous'
    PROPANE = 'propane'
    BIOMASS_WOOD = 'biomass_wood'


# Heating Values (HHV) - EPA 40 CFR Part 98 Table C-1
HEATING_VALUES_HHV: Dict[str, FuelGoldenValue] = {
    'natural_gas': FuelGoldenValue(
        'Natural Gas HHV', Decimal('1028'), 'Btu/scf',
        Decimal('2'), 'EPA 40 CFR 98', 'Table C-1'),
    'fuel_oil_2': FuelGoldenValue(
        'Fuel Oil #2 HHV', Decimal('138000'), 'Btu/gal',
        Decimal('2'), 'EPA 40 CFR 98', 'Table C-1'),
    'fuel_oil_6': FuelGoldenValue(
        'Fuel Oil #6 HHV', Decimal('150000'), 'Btu/gal',
        Decimal('2'), 'EPA 40 CFR 98', 'Table C-1'),
    'coal_bituminous': FuelGoldenValue(
        'Bituminous Coal HHV', Decimal('24930'), 'Btu/lb',
        Decimal('5'), 'EPA 40 CFR 98', 'Table C-1'),
    'coal_subbituminous': FuelGoldenValue(
        'Subbituminous Coal HHV', Decimal('17250'), 'Btu/lb',
        Decimal('5'), 'EPA 40 CFR 98', 'Table C-1'),
    'propane': FuelGoldenValue(
        'Propane HHV', Decimal('91500'), 'Btu/gal',
        Decimal('2'), 'EPA 40 CFR 98', 'Table C-1'),
    'biomass_wood': FuelGoldenValue(
        'Wood Biomass HHV', Decimal('8000'), 'Btu/lb',
        Decimal('10'), 'EPA 40 CFR 98', 'Table C-1'),
}

# CO2 Emission Factors (kg CO2/MMBtu)
CO2_EMISSION_FACTORS: Dict[str, FuelGoldenValue] = {
    'natural_gas': FuelGoldenValue(
        'NG CO2 Factor', Decimal('53.06'), 'kg/MMBtu',
        Decimal('1'), 'EPA 40 CFR 98', 'Table C-1'),
    'fuel_oil_2': FuelGoldenValue(
        'Oil #2 CO2 Factor', Decimal('73.96'), 'kg/MMBtu',
        Decimal('1'), 'EPA 40 CFR 98', 'Table C-1'),
    'fuel_oil_6': FuelGoldenValue(
        'Oil #6 CO2 Factor', Decimal('75.10'), 'kg/MMBtu',
        Decimal('1'), 'EPA 40 CFR 98', 'Table C-1'),
    'coal_bituminous': FuelGoldenValue(
        'Bit Coal CO2 Factor', Decimal('93.28'), 'kg/MMBtu',
        Decimal('2'), 'EPA 40 CFR 98', 'Table C-1'),
    'coal_subbituminous': FuelGoldenValue(
        'Subbit Coal CO2 Factor', Decimal('97.17'), 'kg/MMBtu',
        Decimal('2'), 'EPA 40 CFR 98', 'Table C-1'),
    'propane': FuelGoldenValue(
        'Propane CO2 Factor', Decimal('62.87'), 'kg/MMBtu',
        Decimal('1'), 'EPA 40 CFR 98', 'Table C-1'),
    'biomass_wood': FuelGoldenValue(
        'Wood CO2 Factor', Decimal('93.80'), 'kg/MMBtu',
        Decimal('5'), 'EPA 40 CFR 98', 'Table C-1'),
}

# Fuel Density Values
FUEL_DENSITIES: Dict[str, FuelGoldenValue] = {
    'fuel_oil_2': FuelGoldenValue(
        'Fuel Oil #2 Density', Decimal('7.1'), 'lb/gal',
        Decimal('2'), 'API', ''),
    'fuel_oil_6': FuelGoldenValue(
        'Fuel Oil #6 Density', Decimal('7.88'), 'lb/gal',
        Decimal('2'), 'API', ''),
    'propane': FuelGoldenValue(
        'Propane Density', Decimal('4.2'), 'lb/gal',
        Decimal('2'), 'API', ''),
}

# Natural Gas Composition (typical pipeline quality)
NATURAL_GAS_COMPOSITION: Dict[str, Decimal] = {
    'methane': Decimal('95.0'),      # mol%
    'ethane': Decimal('2.5'),
    'propane': Decimal('0.5'),
    'nitrogen': Decimal('1.5'),
    'co2': Decimal('0.5'),
}


# ==============================================================================
# CALCULATION FUNCTIONS
# ==============================================================================

def calculate_hhv_to_lhv(
    hhv_btu_per_unit: Decimal,
    hydrogen_weight_percent: Decimal,
    moisture_weight_percent: Decimal = Decimal('0')
) -> Decimal:
    """
    Convert Higher Heating Value to Lower Heating Value.

    LHV = HHV - (1040 × (9 × H/100 + M/100))

    Where:
    - 1040 Btu/lb is latent heat of water
    - H = hydrogen content (wt%)
    - M = moisture content (wt%)

    Returns: LHV in same units as HHV
    """
    water_formed = Decimal('9') * hydrogen_weight_percent / Decimal('100')
    water_total = water_formed + moisture_weight_percent / Decimal('100')
    latent_heat_loss = Decimal('1040') * water_total

    lhv = hhv_btu_per_unit - latent_heat_loss
    return lhv.quantize(Decimal('1'), rounding=ROUND_HALF_UP)


def calculate_blend_heating_value(
    fuel_fractions: Dict[str, Decimal],
    heating_values: Dict[str, Decimal]
) -> Decimal:
    """
    Calculate weighted average heating value for fuel blend.

    HHV_blend = Σ (fraction_i × HHV_i)

    Returns: Blended heating value
    """
    if abs(sum(fuel_fractions.values()) - Decimal('1')) > Decimal('0.001'):
        raise ValueError('Fuel fractions must sum to 1.0')

    blend_hhv = Decimal('0')
    for fuel, fraction in fuel_fractions.items():
        if fuel in heating_values:
            blend_hhv += fraction * heating_values[fuel]

    return blend_hhv.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_co2_emissions(
    heat_input_mmbtu: Decimal,
    fuel_type: str,
    custom_ef: Decimal = None
) -> Decimal:
    """
    Calculate CO2 emissions from fuel combustion.

    CO2 (kg) = Heat Input (MMBtu) × Emission Factor (kg/MMBtu)

    Returns: CO2 emissions in kg
    """
    if custom_ef is not None:
        ef = custom_ef
    elif fuel_type in CO2_EMISSION_FACTORS:
        ef = CO2_EMISSION_FACTORS[fuel_type].value
    else:
        raise ValueError(f'Unknown fuel type: {fuel_type}')

    co2 = heat_input_mmbtu * ef
    return co2.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_fuel_cost(
    heat_required_mmbtu: Decimal,
    fuel_price_per_unit: Decimal,
    heating_value_btu_per_unit: Decimal,
    boiler_efficiency: Decimal = Decimal('0.80')
) -> Decimal:
    """
    Calculate fuel cost for required heat output.

    Fuel Required = Heat Required / (HHV × Efficiency)
    Cost = Fuel Required × Price

    Returns: Fuel cost in currency units
    """
    if heating_value_btu_per_unit == 0 or boiler_efficiency == 0:
        raise ValueError('Heating value and efficiency must be non-zero')

    heat_required_btu = heat_required_mmbtu * Decimal('1000000')
    fuel_required = heat_required_btu / (heating_value_btu_per_unit * boiler_efficiency)
    cost = fuel_required * fuel_price_per_unit

    return cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def calculate_carbon_intensity(
    fuel_type: str,
    heating_value_btu_per_unit: Decimal
) -> Decimal:
    """
    Calculate carbon intensity (kg CO2/MMBtu).

    This is the emission factor for the fuel.

    Returns: Carbon intensity in kg CO2/MMBtu
    """
    if fuel_type in CO2_EMISSION_FACTORS:
        return CO2_EMISSION_FACTORS[fuel_type].value
    return Decimal('0')


def calculate_natural_gas_hhv_from_composition(
    methane_mol_pct: Decimal,
    ethane_mol_pct: Decimal,
    propane_mol_pct: Decimal,
    nitrogen_mol_pct: Decimal,
    co2_mol_pct: Decimal
) -> Decimal:
    """
    Calculate natural gas HHV from composition (ASTM D3588).

    HHV = Σ (component_fraction × component_HHV)

    Component HHVs (Btu/scf):
    - Methane: 1010
    - Ethane: 1769
    - Propane: 2516
    - N2: 0
    - CO2: 0

    Returns: HHV in Btu/scf
    """
    component_hhv = {
        'methane': Decimal('1010'),
        'ethane': Decimal('1769'),
        'propane': Decimal('2516'),
        'nitrogen': Decimal('0'),
        'co2': Decimal('0'),
    }

    total = methane_mol_pct + ethane_mol_pct + propane_mol_pct + \
            nitrogen_mol_pct + co2_mol_pct

    if abs(total - Decimal('100')) > Decimal('0.1'):
        raise ValueError('Composition must sum to 100%')

    hhv = (methane_mol_pct / 100 * component_hhv['methane'] +
           ethane_mol_pct / 100 * component_hhv['ethane'] +
           propane_mol_pct / 100 * component_hhv['propane'])

    return hhv.quantize(Decimal('1'), rounding=ROUND_HALF_UP)


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
class TestHeatingValues:
    """Validate heating values against EPA 40 CFR Part 98."""

    @pytest.mark.parametrize("fuel,expected_hhv", [
        ('natural_gas', Decimal('1028')),
        ('fuel_oil_2', Decimal('138000')),
        ('fuel_oil_6', Decimal('150000')),
        ('coal_bituminous', Decimal('24930')),
        ('propane', Decimal('91500')),
    ])
    def test_hhv_values(self, fuel: str, expected_hhv: Decimal):
        """Verify HHV matches EPA Table C-1."""
        golden = HEATING_VALUES_HHV[fuel]
        assert golden.value == expected_hhv
        assert golden.regulation == 'Table C-1'

    def test_coal_hhv_range(self):
        """Coal HHV should be in typical range."""
        bit_hhv = HEATING_VALUES_HHV['coal_bituminous'].value
        subbit_hhv = HEATING_VALUES_HHV['coal_subbituminous'].value

        assert bit_hhv > subbit_hhv  # Bituminous has higher heating value
        assert Decimal('15000') < subbit_hhv < Decimal('20000')
        assert Decimal('20000') < bit_hhv < Decimal('30000')

    def test_liquid_fuel_ranking(self):
        """Fuel oil #6 should have higher HHV per gallon than #2."""
        oil2 = HEATING_VALUES_HHV['fuel_oil_2'].value
        oil6 = HEATING_VALUES_HHV['fuel_oil_6'].value
        assert oil6 > oil2


@pytest.mark.golden
class TestCO2EmissionFactors:
    """Validate CO2 emission factors against EPA 40 CFR Part 98."""

    @pytest.mark.parametrize("fuel,expected_ef", [
        ('natural_gas', Decimal('53.06')),
        ('fuel_oil_2', Decimal('73.96')),
        ('coal_bituminous', Decimal('93.28')),
    ])
    def test_emission_factor_values(self, fuel: str, expected_ef: Decimal):
        """Verify emission factors match EPA Table C-1."""
        golden = CO2_EMISSION_FACTORS[fuel]
        assert golden.value == expected_ef
        assert golden.source == 'EPA 40 CFR 98'

    def test_natural_gas_lowest_carbon(self):
        """Natural gas should have lowest carbon intensity."""
        ng_ef = CO2_EMISSION_FACTORS['natural_gas'].value
        for fuel, golden in CO2_EMISSION_FACTORS.items():
            if fuel != 'natural_gas':
                assert golden.value > ng_ef, \
                    f'{fuel} should have higher CO2 factor than natural gas'

    def test_coal_highest_carbon(self):
        """Coal should have highest carbon intensity."""
        coal_ef = CO2_EMISSION_FACTORS['coal_bituminous'].value
        ng_ef = CO2_EMISSION_FACTORS['natural_gas'].value
        oil_ef = CO2_EMISSION_FACTORS['fuel_oil_2'].value

        assert coal_ef > ng_ef
        assert coal_ef > oil_ef


@pytest.mark.golden
class TestHHVToLHVConversion:
    """Validate HHV to LHV conversion."""

    def test_natural_gas_hhv_lhv(self):
        """Natural gas LHV should be ~10% lower than HHV."""
        # Natural gas ~23% hydrogen by weight
        hhv = Decimal('1028')
        lhv = calculate_hhv_to_lhv(hhv, hydrogen_weight_percent=Decimal('23'))

        ratio = lhv / hhv
        assert Decimal('0.88') < ratio < Decimal('0.92')

    def test_lhv_always_less_than_hhv(self):
        """LHV must always be less than HHV."""
        for h_pct in [Decimal('5'), Decimal('10'), Decimal('15'), Decimal('20')]:
            hhv = Decimal('10000')
            lhv = calculate_hhv_to_lhv(hhv, h_pct)
            assert lhv < hhv


@pytest.mark.golden
class TestFuelBlending:
    """Validate fuel blending calculations."""

    def test_blend_heating_value(self):
        """Verify weighted average HHV calculation."""
        fractions = {
            'natural_gas': Decimal('0.7'),
            'propane': Decimal('0.3'),
        }
        heating_values = {
            'natural_gas': Decimal('1028'),  # Btu/scf
            'propane': Decimal('2500'),  # Btu/scf equivalent
        }

        blend_hhv = calculate_blend_heating_value(fractions, heating_values)

        # 0.7 × 1028 + 0.3 × 2500 = 719.6 + 750 = 1469.6
        expected = Decimal('1469.6')
        deviation = abs(blend_hhv - expected) / expected * 100
        assert deviation <= Decimal('0.1')

    def test_blend_fractions_must_sum_to_one(self):
        """Fuel fractions must sum to 1.0."""
        fractions = {
            'natural_gas': Decimal('0.5'),
            'propane': Decimal('0.3'),
        }
        heating_values = {'natural_gas': Decimal('1028'), 'propane': Decimal('2500')}

        with pytest.raises(ValueError, match='sum to 1.0'):
            calculate_blend_heating_value(fractions, heating_values)


@pytest.mark.golden
class TestCO2Calculation:
    """Validate CO2 emission calculations."""

    @dataclass(frozen=True)
    class CO2TestCase:
        name: str
        heat_input: Decimal
        fuel_type: str
        expected_co2: Decimal

    CASES = [
        CO2TestCase('NG 100 MMBtu', Decimal('100'), 'natural_gas', Decimal('5306.0')),
        CO2TestCase('Oil 50 MMBtu', Decimal('50'), 'fuel_oil_2', Decimal('3698.0')),
        CO2TestCase('Coal 200 MMBtu', Decimal('200'), 'coal_bituminous', Decimal('18656.0')),
    ]

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
    def test_co2_emission_calculation(self, case: CO2TestCase):
        """Verify CO2 emission calculation."""
        calculated = calculate_co2_emissions(case.heat_input, case.fuel_type)

        deviation = abs(calculated - case.expected_co2) / case.expected_co2 * 100
        assert deviation <= Decimal('0.5'), \
            f'{case.name}: {calculated} kg differs from {case.expected_co2} kg'


@pytest.mark.golden
class TestNaturalGasComposition:
    """Validate natural gas HHV from composition."""

    def test_pipeline_quality_hhv(self):
        """Verify HHV calculation from typical pipeline composition."""
        hhv = calculate_natural_gas_hhv_from_composition(
            methane_mol_pct=Decimal('95.0'),
            ethane_mol_pct=Decimal('2.5'),
            propane_mol_pct=Decimal('0.5'),
            nitrogen_mol_pct=Decimal('1.5'),
            co2_mol_pct=Decimal('0.5'),
        )

        # Should be close to 1028 Btu/scf
        assert Decimal('1000') < hhv < Decimal('1060'), \
            f'Pipeline gas HHV {hhv} outside typical range'

    def test_pure_methane_hhv(self):
        """Pure methane should have HHV of 1010 Btu/scf."""
        hhv = calculate_natural_gas_hhv_from_composition(
            methane_mol_pct=Decimal('100'),
            ethane_mol_pct=Decimal('0'),
            propane_mol_pct=Decimal('0'),
            nitrogen_mol_pct=Decimal('0'),
            co2_mol_pct=Decimal('0'),
        )
        assert hhv == Decimal('1010')

    def test_composition_must_sum_to_100(self):
        """Composition must sum to 100%."""
        with pytest.raises(ValueError, match='sum to 100'):
            calculate_natural_gas_hhv_from_composition(
                methane_mol_pct=Decimal('50'),
                ethane_mol_pct=Decimal('10'),
                propane_mol_pct=Decimal('5'),
                nitrogen_mol_pct=Decimal('5'),
                co2_mol_pct=Decimal('5'),
            )


@pytest.mark.golden
class TestFuelCost:
    """Validate fuel cost calculations."""

    def test_fuel_cost_calculation(self):
        """Verify fuel cost calculation."""
        cost = calculate_fuel_cost(
            heat_required_mmbtu=Decimal('100'),
            fuel_price_per_unit=Decimal('3.00'),  # $/MCF
            heating_value_btu_per_unit=Decimal('1028000'),  # Btu/MCF
            boiler_efficiency=Decimal('0.80'),
        )

        # Heat needed = 100 MMBtu = 100,000,000 Btu
        # Fuel needed = 100,000,000 / (1,028,000 × 0.80) = 121.55 MCF
        # Cost = 121.55 × 3.00 = $364.65
        expected = Decimal('364.65')
        deviation = abs(cost - expected) / expected * 100
        assert deviation <= Decimal('1')


@pytest.mark.golden
class TestProvenanceAndDeterminism:
    """Validate provenance tracking and deterministic behavior."""

    def test_all_calculations_deterministic(self):
        """All calculations must be deterministic."""
        calculations = [
            lambda: calculate_hhv_to_lhv(Decimal('1028'), Decimal('23')),
            lambda: calculate_co2_emissions(Decimal('100'), 'natural_gas'),
            lambda: calculate_natural_gas_hhv_from_composition(
                Decimal('95'), Decimal('2.5'), Decimal('0.5'),
                Decimal('1.5'), Decimal('0.5')),
        ]

        for calc in calculations:
            results = set()
            for _ in range(50):
                results.add(str(calc()))
            assert len(results) == 1

    def test_provenance_hash_stability(self):
        """Provenance hash must be stable."""
        inputs = {'fuel_type': 'natural_gas', 'heat_input': '100'}
        outputs = {'co2_kg': '5306'}

        hashes = set()
        for _ in range(50):
            h = calculate_provenance_hash('co2_emissions', inputs, outputs)
            hashes.add(h)

        assert len(hashes) == 1


# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================

def export_golden_values() -> Dict[str, Any]:
    """Export all golden values for documentation."""
    return {
        'metadata': {
            'agent': 'GL-011_FuelCraft',
            'version': '1.0.0',
            'source': 'EPA 40 CFR Part 98, Table C-1',
        },
        'heating_values': {
            k: {'value': str(v.value), 'unit': v.unit}
            for k, v in HEATING_VALUES_HHV.items()
        },
        'co2_factors': {
            k: {'value': str(v.value), 'unit': v.unit}
            for k, v in CO2_EMISSION_FACTORS.items()
        },
        'natural_gas_composition': {
            k: str(v) for k, v in NATURAL_GAS_COMPOSITION.items()
        },
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
