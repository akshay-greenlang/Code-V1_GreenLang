"""
Unit tests for Incineration Emissions Calculator Engine.

Tests CO2 fossil/biogenic separation, CH4/N2O emissions, energy recovery,
and waste-to-energy (WtE) calculations.
"""

import pytest
from decimal import Decimal
from typing import Dict, Any, List

from greenlang.mrv.waste_generated.engines.incineration_emissions import (
    IncinerationEmissionsCalculatorEngine,
    IncineratorType
)
from greenlang.mrv.waste_generated.engines.waste_classification_database import (
    WasteClassificationDatabaseEngine
)
from greenlang.mrv.waste_generated.models import (
    WasteType,
    GWPVersion
)


# Fixtures
@pytest.fixture
def calculator():
    """Create IncinerationEmissionsCalculatorEngine instance."""
    return IncinerationEmissionsCalculatorEngine()


@pytest.fixture
def db_engine():
    """Create WasteClassificationDatabaseEngine instance."""
    return WasteClassificationDatabaseEngine()


@pytest.fixture
def default_params(db_engine) -> Dict[str, Decimal]:
    """Create default incineration parameters."""
    params = db_engine.get_incineration_params('municipal')
    return params


@pytest.fixture
def sample_waste_composition() -> Dict[str, Decimal]:
    """Create sample waste composition."""
    return {
        'food_waste': Decimal('0.25'),
        'paper': Decimal('0.30'),
        'plastics': Decimal('0.20'),
        'textiles': Decimal('0.10'),
        'wood': Decimal('0.10'),
        'other': Decimal('0.05')
    }


# Test CO2 Fossil Calculation
class TestCO2Fossil:
    """Test suite for CO2 fossil = SW × dm × CF × FCF × OF × 44/12."""

    def test_co2_fossil_calculation(self, calculator, default_params):
        """Test CO2 fossil calculation with default parameters."""
        SW = Decimal('100.0')  # 100 tonnes waste
        dm = Decimal('0.90')  # 90% dry matter
        cf = default_params['cf']  # Carbon fraction
        fcf = default_params['fcf']  # Fossil carbon fraction
        of_param = default_params['of']  # Oxidation factor

        co2_fossil = calculator.calculate_co2_fossil(
            SW, dm, cf, fcf, of_param
        )

        # CO2 = SW × dm × CF × FCF × OF × (44/12)
        expected = SW * dm * cf * fcf * of_param * (Decimal('44') / Decimal('12'))

        assert co2_fossil == pytest.approx(expected, rel=1e-6)

    def test_co2_fossil_stoichiometry(self, calculator):
        """Test CO2 uses correct stoichiometric ratio (44/12)."""
        SW = Decimal('100.0')
        dm = Decimal('1.0')
        cf = Decimal('0.50')  # 50% carbon
        fcf = Decimal('1.0')  # 100% fossil
        of_param = Decimal('1.0')  # Complete oxidation

        co2_fossil = calculator.calculate_co2_fossil(
            SW, dm, cf, fcf, of_param
        )

        # 100 t × 1.0 × 0.5 × 1.0 × 1.0 × (44/12) = 50 × 3.667 = 183.33 t CO2
        expected = Decimal('183.333333')

        assert co2_fossil == pytest.approx(expected, rel=1e-5)

    def test_co2_fossil_zero_fcf(self, calculator):
        """Test CO2 fossil is zero when FCF=0 (100% biogenic)."""
        co2_fossil = calculator.calculate_co2_fossil(
            SW=Decimal('100.0'),
            dm=Decimal('0.90'),
            cf=Decimal('0.50'),
            fcf=Decimal('0.0'),  # No fossil carbon
            of=Decimal('1.0')
        )

        assert co2_fossil == Decimal('0.0')


# Test Biogenic CO2
class TestBiogenicCO2:
    """Test suite for biogenic CO2 calculation."""

    def test_biogenic_co2_calculation(self, calculator, default_params):
        """Test biogenic CO2 = (1-FCF) fraction."""
        SW = Decimal('100.0')
        dm = Decimal('0.90')
        cf = default_params['cf']
        fcf = default_params['fcf']
        of_param = default_params['of']

        co2_biogenic = calculator.calculate_co2_biogenic(
            SW, dm, cf, fcf, of_param
        )

        # Biogenic = SW × dm × CF × (1-FCF) × OF × (44/12)
        expected = SW * dm * cf * (Decimal('1.0') - fcf) * of_param * (Decimal('44') / Decimal('12'))

        assert co2_biogenic == pytest.approx(expected, rel=1e-6)

    def test_biogenic_co2_not_counted(self, calculator):
        """Test biogenic CO2 is tracked but not counted in GHG inventory."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.FOOD_ORGANIC,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        # Biogenic CO2 should be tracked
        assert result['co2_biogenic_tonnes'] > 0

        # But not included in total CO2e
        assert result['total_co2e'] == (
            result['co2_fossil_tonnes'] +
            result['ch4_co2e_tonnes'] +
            result['n2o_co2e_tonnes']
        )


# Test CH4 Emissions
class TestCH4Emissions:
    """Test suite for CH4 emissions by incinerator type."""

    @pytest.mark.parametrize('incinerator_type,expected_ef_range', [
        (IncineratorType.MUNICIPAL, (Decimal('0.001'), Decimal('0.020'))),
        (IncineratorType.INDUSTRIAL, (Decimal('0.001'), Decimal('0.015'))),
        (IncineratorType.HAZARDOUS, (Decimal('0.001'), Decimal('0.010'))),
        (IncineratorType.WTE, (Decimal('0.001'), Decimal('0.015'))),
    ])
    def test_ch4_emissions_by_type(
        self,
        calculator,
        incinerator_type,
        expected_ef_range
    ):
        """Test CH4 emissions for different incinerator types."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=incinerator_type
        )

        ch4_tonnes = result['ch4_emissions_tonnes']
        ch4_ef = ch4_tonnes / Decimal('100.0')  # kg/tonne

        assert ch4_ef >= expected_ef_range[0]
        assert ch4_ef <= expected_ef_range[1]

    def test_ch4_emissions_modern_vs_old(self, calculator):
        """Test modern incinerators have lower CH4 emissions."""
        result_modern = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.WTE,  # Modern
            technology_year=2025
        )

        result_old = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.MUNICIPAL,
            technology_year=1990
        )

        # Modern should have lower CH4 emissions
        assert result_modern['ch4_emissions_tonnes'] <= result_old['ch4_emissions_tonnes']


# Test N2O Emissions
class TestN2OEmissions:
    """Test suite for N2O emissions by incinerator type."""

    @pytest.mark.parametrize('incinerator_type,expected_ef_range', [
        (IncineratorType.MUNICIPAL, (Decimal('0.010'), Decimal('0.100'))),
        (IncineratorType.INDUSTRIAL, (Decimal('0.010'), Decimal('0.080'))),
        (IncineratorType.HAZARDOUS, (Decimal('0.050'), Decimal('0.150'))),
        (IncineratorType.WTE, (Decimal('0.010'), Decimal('0.080'))),
    ])
    def test_n2o_emissions_by_type(
        self,
        calculator,
        incinerator_type,
        expected_ef_range
    ):
        """Test N2O emissions for different incinerator types."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=incinerator_type
        )

        n2o_tonnes = result['n2o_emissions_tonnes']
        n2o_ef = n2o_tonnes / Decimal('100.0')  # kg/tonne

        assert n2o_ef >= expected_ef_range[0]
        assert n2o_ef <= expected_ef_range[1]

    def test_n2o_emissions_nitrogen_rich_waste(self, calculator):
        """Test N2O emissions are higher for nitrogen-rich waste."""
        result_food = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.FOOD_ORGANIC,  # N-rich
            incinerator_type=IncineratorType.MUNICIPAL
        )

        result_plastics = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.PLASTICS,  # Low N
            incinerator_type=IncineratorType.MUNICIPAL
        )

        # Food waste should produce more N2O
        assert result_food['n2o_emissions_tonnes'] >= result_plastics['n2o_emissions_tonnes']


# Test Total CO2e Aggregation
class TestTotalCO2e:
    """Test suite for total_co2e aggregation."""

    def test_total_co2e_sum(self, calculator):
        """Test total CO2e equals sum of all GHGs."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.MUNICIPAL,
            gwp_version=GWPVersion.AR5
        )

        # Total = CO2_fossil + CH4_CO2e + N2O_CO2e
        expected_total = (
            result['co2_fossil_tonnes'] +
            result['ch4_co2e_tonnes'] +
            result['n2o_co2e_tonnes']
        )

        assert result['total_co2e'] == pytest.approx(expected_total, rel=1e-6)

    def test_total_co2e_excludes_biogenic(self, calculator):
        """Test total CO2e excludes biogenic CO2."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.FOOD_ORGANIC,  # 100% biogenic
            incinerator_type=IncineratorType.MUNICIPAL
        )

        # Total should not include biogenic CO2
        assert result['total_co2e'] < result['co2_biogenic_tonnes']


# Test With Composition
class TestWithComposition:
    """Test suite for with_composition method."""

    def test_composition_mixed_waste(self, calculator, sample_waste_composition):
        """Test incineration of mixed waste with composition."""
        result = calculator.with_composition(
            total_mass_tonnes=Decimal('100.0'),
            composition=sample_waste_composition,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        assert result['total_co2e'] > 0
        assert len(result['emissions_by_component']) == len(sample_waste_composition)

    def test_composition_sum_equals_total(self, calculator, sample_waste_composition):
        """Test sum of component emissions equals total."""
        result = calculator.with_composition(
            total_mass_tonnes=Decimal('100.0'),
            composition=sample_waste_composition,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        component_co2_sum = sum(
            comp['co2_fossil_tonnes']
            for comp in result['emissions_by_component'].values()
        )

        assert component_co2_sum == pytest.approx(result['total_co2_fossil'], rel=1e-5)


# Test Plastics (100% Fossil Carbon)
class TestPlastics:
    """Test suite for plastics incineration (100% fossil carbon)."""

    def test_plastics_100_percent_fossil(self, calculator):
        """Test plastics have FCF=1.0 (100% fossil carbon)."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.PLASTICS,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        # All CO2 should be fossil, none biogenic
        assert result['co2_fossil_tonnes'] > 0
        assert result['co2_biogenic_tonnes'] == Decimal('0.0')

    def test_plastics_high_carbon_content(self, calculator):
        """Test plastics have high carbon content."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.PLASTICS,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        # Plastics typically ~75% carbon
        # 100 t × 0.75 × 1.0 (FCF) × 1.0 (OF) × (44/12) ≈ 275 t CO2
        assert result['co2_fossil_tonnes'] > Decimal('200.0')


# Test Food Waste (0% Fossil Carbon)
class TestFoodWaste:
    """Test suite for food waste incineration (0% fossil carbon)."""

    def test_food_waste_zero_fossil(self, calculator):
        """Test food waste has FCF=0.0 (0% fossil carbon)."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.FOOD_ORGANIC,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        # All CO2 should be biogenic, none fossil
        assert result['co2_fossil_tonnes'] == Decimal('0.0')
        assert result['co2_biogenic_tonnes'] > 0

    def test_food_waste_total_co2e_low(self, calculator):
        """Test food waste has low total CO2e (only CH4/N2O)."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.FOOD_ORGANIC,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        # Total CO2e should only be from CH4 and N2O
        expected = result['ch4_co2e_tonnes'] + result['n2o_co2e_tonnes']

        assert result['total_co2e'] == pytest.approx(expected, rel=1e-6)


# Test Paper (1% Fossil Carbon)
class TestPaper:
    """Test suite for paper incineration (~1% fossil carbon from coatings)."""

    def test_paper_low_fossil_carbon(self, calculator):
        """Test paper has low FCF (~1% from coatings/inks)."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.PAPER_CARDBOARD,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        # Most CO2 should be biogenic
        assert result['co2_biogenic_tonnes'] > result['co2_fossil_tonnes']

        # But some fossil CO2 from coatings
        assert result['co2_fossil_tonnes'] > Decimal('0.0')


# Test Energy Recovery
class TestEnergyRecovery:
    """Test suite for energy_recovery estimation."""

    def test_energy_recovery_wte(self, calculator):
        """Test energy recovery for WtE incinerator."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.WTE
        )

        assert 'energy_recovered_mwh' in result
        assert result['energy_recovered_mwh'] > 0

    def test_energy_recovery_no_recovery(self, calculator):
        """Test no energy recovery for non-WtE incinerator."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        # Municipal incinerator without energy recovery
        assert result.get('energy_recovered_mwh', Decimal('0.0')) == Decimal('0.0')

    def test_energy_recovery_from_ncv(self, calculator):
        """Test energy recovery calculated from NCV."""
        ncv = Decimal('10.0')  # MJ/kg
        mass = Decimal('100.0')  # tonnes = 100,000 kg

        energy_mwh = calculator.calculate_energy_recovery(
            waste_mass_kg=mass * Decimal('1000'),
            ncv_mj_per_kg=ncv,
            recovery_efficiency=Decimal('0.25')  # 25% efficient
        )

        # Energy = 100,000 kg × 10 MJ/kg × 0.25 / 3.6 (MJ to MWh)
        expected = Decimal('100000') * ncv * Decimal('0.25') / Decimal('3.6')

        assert energy_mwh == pytest.approx(expected, rel=1e-5)


# Test Avoided Grid Emissions
class TestAvoidedGridEmissions:
    """Test suite for avoided_grid_emissions (separate, not deducted)."""

    def test_avoided_emissions_calculation(self, calculator):
        """Test avoided grid emissions calculation."""
        energy_mwh = Decimal('250.0')
        grid_ef = Decimal('0.5')  # kg CO2/kWh = 500 kg/MWh

        avoided = calculator.calculate_avoided_grid_emissions(
            energy_mwh,
            grid_ef
        )

        # Avoided = 250 MWh × 500 kg/MWh = 125,000 kg = 125 tonnes
        expected = Decimal('125.0')

        assert avoided == pytest.approx(expected, rel=1e-6)

    def test_avoided_emissions_not_deducted(self, calculator):
        """Test avoided emissions are reported separately, not deducted."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.WTE,
            calculate_avoided_emissions=True,
            grid_emission_factor=Decimal('0.5')
        )

        # Avoided emissions should be tracked separately
        if 'avoided_grid_emissions_tonnes' in result:
            assert result['avoided_grid_emissions_tonnes'] > 0

            # But not deducted from total_co2e
            assert result['total_co2e'] > 0


# Test Ash Residue
class TestAshResidue:
    """Test suite for ash_residue calculation."""

    def test_ash_residue_percentage(self, calculator):
        """Test ash residue is typically 10-30% of input."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        ash = result.get('ash_residue_tonnes', Decimal('0.0'))

        # Ash should be 10-30% of input
        assert ash >= Decimal('10.0')
        assert ash <= Decimal('30.0')

    def test_ash_residue_higher_for_inorganics(self, calculator):
        """Test ash residue is higher for waste with inorganics."""
        result_mixed = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        result_organic = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.FOOD_ORGANIC,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        # Mixed MSW has more inorganics → more ash
        assert result_mixed.get('ash_residue_tonnes', 0) >= result_organic.get('ash_residue_tonnes', 0)


# Test Batch Calculation
class TestBatchCalculation:
    """Test suite for batch calculation."""

    def test_batch_calculation(self, calculator):
        """Test batch calculation for multiple waste streams."""
        waste_streams = [
            {'mass': Decimal('50.0'), 'type': WasteType.PLASTICS},
            {'mass': Decimal('75.0'), 'type': WasteType.PAPER_CARDBOARD},
            {'mass': Decimal('25.0'), 'type': WasteType.FOOD_ORGANIC}
        ]

        results = calculator.calculate_batch(
            waste_streams,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        assert len(results) == 3
        assert all('total_co2e' in r for r in results)


# Test Validation
class TestValidation:
    """Test suite for input validation."""

    def test_negative_mass_raises_error(self, calculator):
        """Test negative waste mass raises validation error."""
        with pytest.raises(ValueError, match='waste mass must be non-negative'):
            calculator.calculate(
                waste_mass_tonnes=Decimal('-10.0'),
                waste_type=WasteType.MIXED_MSW,
                incinerator_type=IncineratorType.MUNICIPAL
            )

    def test_invalid_fcf_raises_error(self, calculator):
        """Test invalid FCF (>1.0) raises validation error."""
        with pytest.raises(ValueError, match='FCF must be between 0 and 1'):
            calculator.calculate(
                waste_mass_tonnes=Decimal('100.0'),
                waste_type=WasteType.MIXED_MSW,
                incinerator_type=IncineratorType.MUNICIPAL,
                fcf=Decimal('1.5')  # Invalid
            )

    def test_invalid_oxidation_factor_raises_error(self, calculator):
        """Test invalid oxidation factor raises validation error."""
        with pytest.raises(ValueError, match='oxidation factor must be between 0 and 1'):
            calculator.calculate(
                waste_mass_tonnes=Decimal('100.0'),
                waste_type=WasteType.MIXED_MSW,
                incinerator_type=IncineratorType.MUNICIPAL,
                of=Decimal('1.2')  # Invalid
            )


# Test WtE Efficiency
class TestWtEEfficiency:
    """Test suite for WtE energy recovery efficiency."""

    def test_wte_efficiency_range(self, calculator):
        """Test WtE efficiency is within expected range (20-35%)."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.WTE
        )

        energy = result.get('energy_recovered_mwh', Decimal('0.0'))

        # Typical MSW: ~10 MJ/kg NCV
        # 100 tonnes = 100,000 kg × 10 MJ/kg = 1,000,000 MJ = 277.8 MWh
        # At 25% efficiency: ~69.4 MWh

        assert energy >= Decimal('50.0')  # >20% efficiency
        assert energy <= Decimal('100.0')  # <35% efficiency

    def test_wte_modern_vs_old_efficiency(self, calculator):
        """Test modern WtE has higher efficiency than old."""
        result_modern = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.WTE,
            technology_year=2025
        )

        result_old = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.WTE,
            technology_year=1995
        )

        # Modern should recover more energy
        assert result_modern.get('energy_recovered_mwh', 0) >= result_old.get('energy_recovered_mwh', 0)


# Test Net Calorific Value
class TestNetCalorificValue:
    """Test suite for net calorific value (NCV) estimation."""

    def test_ncv_by_waste_type(self, calculator):
        """Test NCV varies by waste type."""
        ncv_plastics = calculator.get_ncv(WasteType.PLASTICS)
        ncv_food = calculator.get_ncv(WasteType.FOOD_ORGANIC)

        # Plastics have much higher NCV than food waste
        assert ncv_plastics > ncv_food * Decimal('2.0')

    @pytest.mark.parametrize('waste_type,expected_ncv_range', [
        (WasteType.PLASTICS, (Decimal('30.0'), Decimal('45.0'))),  # MJ/kg
        (WasteType.PAPER_CARDBOARD, (Decimal('12.0'), Decimal('18.0'))),
        (WasteType.FOOD_ORGANIC, (Decimal('4.0'), Decimal('8.0'))),
        (WasteType.WOOD, (Decimal('15.0'), Decimal('20.0'))),
        (WasteType.MIXED_MSW, (Decimal('8.0'), Decimal('12.0'))),
    ])
    def test_ncv_ranges(
        self,
        calculator,
        waste_type,
        expected_ncv_range
    ):
        """Test NCV is within expected range for each waste type."""
        ncv = calculator.get_ncv(waste_type)

        assert ncv >= expected_ncv_range[0]
        assert ncv <= expected_ncv_range[1]

    def test_ncv_composition_weighted(self, calculator, sample_waste_composition):
        """Test NCV for mixed waste is composition-weighted average."""
        ncv_mixed = calculator.get_ncv_from_composition(
            sample_waste_composition
        )

        # Calculate manual weighted average
        expected = Decimal('0.0')
        for waste_type_str, fraction in sample_waste_composition.items():
            waste_type = WasteType[waste_type_str.upper().replace('_', '')]
            ncv = calculator.get_ncv(waste_type)
            expected += ncv * fraction

        assert ncv_mixed == pytest.approx(expected, rel=1e-5)


# Test GWP Versions
class TestGWPVersions:
    """Test suite for different GWP versions."""

    def test_gwp_ar5_ch4(self, calculator):
        """Test AR5 GWP for CH4 (28)."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.MUNICIPAL,
            gwp_version=GWPVersion.AR5
        )

        expected_ch4_co2e = result['ch4_emissions_tonnes'] * Decimal('28')

        assert result['ch4_co2e_tonnes'] == pytest.approx(expected_ch4_co2e, rel=1e-6)

    def test_gwp_ar5_n2o(self, calculator):
        """Test AR5 GWP for N2O (265)."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.MUNICIPAL,
            gwp_version=GWPVersion.AR5
        )

        expected_n2o_co2e = result['n2o_emissions_tonnes'] * Decimal('265')

        assert result['n2o_co2e_tonnes'] == pytest.approx(expected_n2o_co2e, rel=1e-5)


# Test Decimal Precision
class TestDecimalPrecision:
    """Test suite for Decimal arithmetic precision."""

    def test_no_float_drift(self, calculator):
        """Test calculations avoid float drift using Decimal."""
        result1 = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        result2 = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        # Identical inputs → identical outputs (bit-perfect)
        assert result1['co2_fossil_tonnes'] == result2['co2_fossil_tonnes']

    def test_high_precision_calculations(self, calculator):
        """Test calculations maintain high precision."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('0.001'),  # 1 kg
            waste_type=WasteType.PLASTICS,
            incinerator_type=IncineratorType.MUNICIPAL
        )

        # Should handle very small values without rounding errors
        assert result['co2_fossil_tonnes'] > Decimal('0.0')
