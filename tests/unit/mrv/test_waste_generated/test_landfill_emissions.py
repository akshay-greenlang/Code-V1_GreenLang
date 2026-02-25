"""
Unit tests for Landfill Emissions Calculator Engine.

Tests IPCC First Order Decay (FOD) model, multi-year decay, gas capture,
oxidation, and lifetime emissions calculations.
"""

import pytest
from decimal import Decimal
from typing import Dict, Any, List
from datetime import datetime

from greenlang.mrv.waste_generated.engines.landfill_emissions import (
    LandfillEmissionsCalculatorEngine,
    FODModel,
    GasCaptureTimeline
)
from greenlang.mrv.waste_generated.engines.waste_classification_database import (
    WasteClassificationDatabaseEngine,
    ClimateZone,
    LandfillType
)
from greenlang.mrv.waste_generated.models import (
    WasteType,
    GWPVersion
)


# Fixtures
@pytest.fixture
def calculator():
    """Create LandfillEmissionsCalculatorEngine instance."""
    return LandfillEmissionsCalculatorEngine()


@pytest.fixture
def db_engine():
    """Create WasteClassificationDatabaseEngine instance."""
    return WasteClassificationDatabaseEngine()


@pytest.fixture
def default_params(db_engine) -> Dict[str, Decimal]:
    """Create default calculation parameters."""
    return {
        'doc': db_engine.get_doc(WasteType.MIXED_MSW),
        'docf': Decimal('0.50'),
        'mcf': db_engine.get_mcf(LandfillType.MANAGED_ANAEROBIC),
        'k': db_engine.get_decay_rate(ClimateZone.TEMPERATE_DRY, WasteType.MIXED_MSW),
        'f': Decimal('0.50'),
        'ox': Decimal('0.10')
    }


@pytest.fixture
def sample_waste_composition() -> Dict[str, Decimal]:
    """Create sample waste composition."""
    return {
        'food_waste': Decimal('0.30'),
        'paper': Decimal('0.25'),
        'wood': Decimal('0.15'),
        'textiles': Decimal('0.10'),
        'plastics': Decimal('0.10'),
        'other': Decimal('0.10')
    }


# Test FOD Model Step 1: DDOCm Calculation
class TestFODModelStep1:
    """Test suite for FOD Step 1: DDOCm = W × DOC × DOCf × MCF."""

    def test_ddocm_calculation(self, calculator, default_params):
        """Test DDOCm calculation with default parameters."""
        W = Decimal('100.0')  # 100 tonnes waste
        doc = default_params['doc']
        docf = default_params['docf']
        mcf = default_params['mcf']

        ddocm = calculator.calculate_ddocm(W, doc, docf, mcf)

        expected = W * doc * docf * mcf
        assert ddocm == expected

    def test_ddocm_zero_waste(self, calculator, default_params):
        """Test DDOCm is zero when waste mass is zero."""
        W = Decimal('0.0')
        ddocm = calculator.calculate_ddocm(
            W,
            default_params['doc'],
            default_params['docf'],
            default_params['mcf']
        )

        assert ddocm == Decimal('0.0')

    def test_ddocm_high_doc(self, calculator):
        """Test DDOCm with high DOC waste (food waste)."""
        W = Decimal('100.0')
        doc = Decimal('0.15')  # Food waste DOC
        docf = Decimal('0.50')
        mcf = Decimal('1.0')

        ddocm = calculator.calculate_ddocm(W, doc, docf, mcf)

        expected = Decimal('100.0') * Decimal('0.15') * Decimal('0.50') * Decimal('1.0')
        assert ddocm == expected


# Test FOD Model Step 2: DDOCm Accumulated
class TestFODModelStep2:
    """Test suite for FOD Step 2: DDOCm accumulated with decay."""

    def test_ddocm_accumulated_single_year(self, calculator, default_params):
        """Test DDOCm accumulated for single year."""
        waste_deposited = [Decimal('100.0')]  # 100 tonnes in year 1

        ddocm_accumulated = calculator.calculate_ddocm_accumulated(
            waste_deposited,
            default_params['doc'],
            default_params['docf'],
            default_params['mcf'],
            default_params['k'],
            years=1
        )

        # For single year, accumulated = initial DDOCm
        expected_ddocm = (
            Decimal('100.0') *
            default_params['doc'] *
            default_params['docf'] *
            default_params['mcf']
        )

        assert ddocm_accumulated[0] == pytest.approx(expected_ddocm, rel=1e-6)

    def test_ddocm_accumulated_multi_year(self, calculator, default_params):
        """Test DDOCm accumulated over multiple years."""
        waste_deposited = [Decimal('100.0')] * 10  # 100 tonnes/year for 10 years

        ddocm_accumulated = calculator.calculate_ddocm_accumulated(
            waste_deposited,
            default_params['doc'],
            default_params['docf'],
            default_params['mcf'],
            default_params['k'],
            years=10
        )

        # Accumulated should increase each year
        assert len(ddocm_accumulated) == 10
        for i in range(1, 10):
            assert ddocm_accumulated[i] >= ddocm_accumulated[i-1]


# Test FOD Model Step 3: DDOCm Decomposed
class TestFODModelStep3:
    """Test suite for FOD Step 3: DDOCm decomposed."""

    def test_ddocm_decomposed_single_year(self, calculator, default_params):
        """Test DDOCm decomposed for single year."""
        ddocm_accumulated = [Decimal('9.0')]  # From Step 2

        ddocm_decomposed = calculator.calculate_ddocm_decomposed(
            ddocm_accumulated,
            default_params['k']
        )

        # Decomposed = accumulated × (1 - e^(-k))
        expected = ddocm_accumulated[0] * (Decimal('1.0') - ((-default_params['k']).exp()))

        assert ddocm_decomposed[0] == pytest.approx(expected, rel=1e-6)

    def test_ddocm_decomposed_approaches_accumulated(self, calculator):
        """Test decomposed approaches accumulated over long time."""
        ddocm_accumulated = [Decimal('10.0')] * 100
        k = Decimal('0.09')

        ddocm_decomposed = calculator.calculate_ddocm_decomposed(
            ddocm_accumulated,
            k
        )

        # After many years, decomposed should approach accumulated
        # (1 - e^(-k*t)) → 1 as t → ∞
        assert ddocm_decomposed[-1] >= ddocm_accumulated[-1] * Decimal('0.99')


# Test FOD Model Step 4: CH4 Generated
class TestFODModelStep4:
    """Test suite for FOD Step 4: CH4 generated = DDOCm_decomposed × F × 16/12."""

    def test_ch4_generated(self, calculator, default_params):
        """Test CH4 generation from decomposed DOC."""
        ddocm_decomposed = [Decimal('8.167')]  # From Step 3
        f = default_params['f']

        ch4_generated = calculator.calculate_ch4_generated(
            ddocm_decomposed,
            f
        )

        # CH4 = DDOCm_decomposed × F × (16/12)
        expected = ddocm_decomposed[0] * f * (Decimal('16') / Decimal('12'))

        assert ch4_generated[0] == pytest.approx(expected, rel=1e-6)

    def test_ch4_generated_stoichiometry(self, calculator):
        """Test CH4 generation uses correct stoichiometric ratio."""
        ddocm_decomposed = [Decimal('12.0')]  # 12 tonnes carbon
        f = Decimal('0.50')

        ch4_generated = calculator.calculate_ch4_generated(
            ddocm_decomposed,
            f
        )

        # 12 tonnes C × 0.5 × (16/12) = 8 tonnes CH4
        expected = Decimal('8.0')

        assert ch4_generated[0] == pytest.approx(expected, rel=1e-6)


# Test FOD Model Step 5: CH4 Emitted
class TestFODModelStep5:
    """Test suite for FOD Step 5: CH4 emitted = (generated - recovered) × (1-OX)."""

    def test_ch4_emitted_no_recovery(self, calculator, default_params):
        """Test CH4 emissions with no gas recovery."""
        ch4_generated = [Decimal('10.0')]
        ch4_recovered = [Decimal('0.0')]
        ox = default_params['ox']

        ch4_emitted = calculator.calculate_ch4_emitted(
            ch4_generated,
            ch4_recovered,
            ox
        )

        expected = (ch4_generated[0] - ch4_recovered[0]) * (Decimal('1.0') - ox)

        assert ch4_emitted[0] == pytest.approx(expected, rel=1e-6)

    def test_ch4_emitted_with_recovery(self, calculator):
        """Test CH4 emissions with gas recovery."""
        ch4_generated = [Decimal('10.0')]
        ch4_recovered = [Decimal('7.5')]  # 75% capture
        ox = Decimal('0.10')

        ch4_emitted = calculator.calculate_ch4_emitted(
            ch4_generated,
            ch4_recovered,
            ox
        )

        # (10 - 7.5) × (1 - 0.10) = 2.5 × 0.90 = 2.25
        expected = Decimal('2.25')

        assert ch4_emitted[0] == pytest.approx(expected, rel=1e-6)

    def test_ch4_emitted_full_recovery(self, calculator):
        """Test CH4 emissions with 100% gas recovery."""
        ch4_generated = [Decimal('10.0')]
        ch4_recovered = [Decimal('10.0')]
        ox = Decimal('0.10')

        ch4_emitted = calculator.calculate_ch4_emitted(
            ch4_generated,
            ch4_recovered,
            ox
        )

        assert ch4_emitted[0] == Decimal('0.0')


# Test Full Calculation
class TestCalculate:
    """Test suite for complete calculate method."""

    def test_calculate_default_params(self, calculator, default_params):
        """Test calculate with default parameters."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            doc=default_params['doc'],
            docf=default_params['docf'],
            mcf=default_params['mcf'],
            k=default_params['k'],
            f=default_params['f'],
            ox=default_params['ox'],
            gas_capture_efficiency=Decimal('0.0')
        )

        assert result['ch4_emissions_tonnes'] > 0
        assert result['co2e_emissions_tonnes'] > 0
        assert result['ddocm'] > 0

    def test_calculate_with_gas_capture(self, calculator, default_params):
        """Test calculate with gas capture system."""
        result_no_capture = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            gas_capture_efficiency=Decimal('0.0'),
            **default_params
        )

        result_with_capture = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            gas_capture_efficiency=Decimal('0.75'),
            **default_params
        )

        # Emissions should be lower with gas capture
        assert result_with_capture['ch4_emissions_tonnes'] < result_no_capture['ch4_emissions_tonnes']

    def test_calculate_without_cover(self, calculator, default_params):
        """Test calculate without soil cover (OX=0)."""
        params_no_cover = default_params.copy()
        params_no_cover['ox'] = Decimal('0.0')

        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            **params_no_cover
        )

        # Higher emissions without oxidation
        assert result['ch4_emissions_tonnes'] > 0


# Test Multi-Year Decay
class TestMultiYearDecay:
    """Test suite for multi_year_decay method."""

    def test_multi_year_decay_10_years(self, calculator, default_params):
        """Test multi-year decay for 10 years."""
        result = calculator.multi_year_decay(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            years=10,
            **default_params
        )

        assert len(result['yearly_emissions']) == 10
        assert result['cumulative_emissions'][-1] > 0

    def test_multi_year_decay_50_years(self, calculator, default_params):
        """Test multi-year decay for 50 years."""
        result = calculator.multi_year_decay(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            years=50,
            **default_params
        )

        assert len(result['yearly_emissions']) == 50

        # Emissions should decrease over time (exponential decay)
        assert result['yearly_emissions'][-1] < result['yearly_emissions'][0]

    def test_multi_year_decay_100_years(self, calculator, default_params):
        """Test multi-year decay for 100 years (lifetime)."""
        result = calculator.multi_year_decay(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            years=100,
            **default_params
        )

        # After 100 years, most organic carbon should have decomposed
        cumulative = result['cumulative_emissions'][-1]
        assert cumulative > 0


# Test First Year vs Lifetime Emissions
class TestFirstYearVsLifetime:
    """Test suite comparing first year vs lifetime emissions."""

    def test_first_year_emissions(self, calculator, default_params):
        """Test first-year emissions only."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            lifetime_emissions=False,
            **default_params
        )

        assert result['calculation_year'] == 1
        assert result['lifetime_emissions'] is False

    def test_lifetime_emissions(self, calculator, default_params):
        """Test lifetime (100-year) emissions."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            lifetime_emissions=True,
            years=100,
            **default_params
        )

        assert result['lifetime_emissions'] is True
        assert result['total_lifetime_emissions'] > result['ch4_emissions_tonnes']

    def test_lifetime_greater_than_first_year(self, calculator, default_params):
        """Test lifetime emissions are greater than first year."""
        first_year = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            lifetime_emissions=False,
            **default_params
        )

        lifetime = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            lifetime_emissions=True,
            years=100,
            **default_params
        )

        assert lifetime['total_lifetime_emissions'] > first_year['ch4_emissions_tonnes']


# Test Mixed Waste Composition
class TestMixedWaste:
    """Test suite for mixed_waste calculation."""

    def test_mixed_waste_composition(self, calculator, sample_waste_composition, default_params):
        """Test mixed waste with composition breakdown."""
        result = calculator.mixed_waste(
            total_mass_tonnes=Decimal('100.0'),
            composition=sample_waste_composition,
            **default_params
        )

        assert result['total_ch4_emissions'] > 0
        assert len(result['emissions_by_component']) == len(sample_waste_composition)

    def test_mixed_waste_sum_equals_total(self, calculator, sample_waste_composition, default_params):
        """Test sum of component emissions equals total."""
        result = calculator.mixed_waste(
            total_mass_tonnes=Decimal('100.0'),
            composition=sample_waste_composition,
            **default_params
        )

        component_sum = sum(
            comp['ch4_emissions']
            for comp in result['emissions_by_component'].values()
        )

        assert component_sum == pytest.approx(result['total_ch4_emissions'], rel=1e-6)


# Test Half-Life Calculation
class TestHalfLife:
    """Test suite for half_life calculation."""

    def test_half_life_from_k(self, calculator):
        """Test half-life calculation from decay constant k."""
        k = Decimal('0.09')

        half_life = calculator.calculate_half_life(k)

        # t_half = ln(2) / k
        expected = Decimal('0.693147') / k

        assert half_life == pytest.approx(expected, rel=1e-4)

    def test_decay_constant_from_half_life(self, calculator):
        """Test decay constant calculation from half-life."""
        half_life = Decimal('7.7')  # years

        k = calculator.calculate_k_from_half_life(half_life)

        # k = ln(2) / t_half
        expected = Decimal('0.693147') / half_life

        assert k == pytest.approx(expected, rel=1e-4)

    def test_half_life_roundtrip(self, calculator):
        """Test half-life conversion roundtrip."""
        k_original = Decimal('0.09')

        half_life = calculator.calculate_half_life(k_original)
        k_recovered = calculator.calculate_k_from_half_life(half_life)

        assert k_recovered == pytest.approx(k_original, rel=1e-6)


# Test Gas Capture Timeline
class TestGasCaptureTimeline:
    """Test suite for gas_capture_timeline method."""

    def test_gas_capture_ramp_up(self, calculator):
        """Test gas capture system ramps up over time."""
        timeline = calculator.gas_capture_timeline(
            system_type='modern',
            start_year=1,
            ramp_up_years=5
        )

        # Year 1 should have lower efficiency than Year 5
        assert timeline.get_efficiency(1) < timeline.get_efficiency(5)

    def test_gas_capture_phased(self, calculator):
        """Test phased gas capture implementation."""
        timeline = calculator.gas_capture_timeline(
            system_type='modern',
            start_year=3,  # Starts in year 3
            ramp_up_years=5
        )

        # Years 1-2 should have zero capture
        assert timeline.get_efficiency(1) == Decimal('0.0')
        assert timeline.get_efficiency(2) == Decimal('0.0')

        # Year 3+ should have capture
        assert timeline.get_efficiency(3) > Decimal('0.0')

    def test_gas_capture_full_efficiency(self, calculator):
        """Test gas capture reaches full efficiency."""
        timeline = calculator.gas_capture_timeline(
            system_type='modern',
            start_year=1,
            ramp_up_years=5
        )

        # After ramp-up, should reach target efficiency
        final_efficiency = timeline.get_efficiency(10)
        assert final_efficiency >= Decimal('0.75')


# Test Cumulative Emissions
class TestCumulativeEmissions:
    """Test suite for cumulative_emissions calculation."""

    def test_cumulative_monotonically_increasing(self, calculator, default_params):
        """Test cumulative emissions are monotonically increasing."""
        result = calculator.multi_year_decay(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            years=20,
            **default_params
        )

        cumulative = result['cumulative_emissions']

        for i in range(1, len(cumulative)):
            assert cumulative[i] >= cumulative[i-1]

    def test_cumulative_equals_sum(self, calculator, default_params):
        """Test cumulative equals sum of yearly emissions."""
        result = calculator.multi_year_decay(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            years=10,
            **default_params
        )

        yearly_sum = sum(result['yearly_emissions'])
        cumulative_final = result['cumulative_emissions'][-1]

        assert cumulative_final == pytest.approx(yearly_sum, rel=1e-6)


# Test Batch Calculation
class TestBatchCalculation:
    """Test suite for batch calculation."""

    def test_batch_calculation(self, calculator, default_params):
        """Test batch calculation for multiple waste streams."""
        waste_streams = [
            {'mass': Decimal('50.0'), 'type': WasteType.FOOD_ORGANIC},
            {'mass': Decimal('75.0'), 'type': WasteType.PAPER_CARDBOARD},
            {'mass': Decimal('25.0'), 'type': WasteType.WOOD}
        ]

        results = calculator.calculate_batch(
            waste_streams,
            **default_params
        )

        assert len(results) == 3
        assert all('ch4_emissions_tonnes' in r for r in results)


# Test GWP Versions
class TestGWPVersions:
    """Test suite for different GWP versions."""

    def test_gwp_ar4(self, calculator, default_params):
        """Test calculation with AR4 GWP (CH4 = 25)."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            gwp_version=GWPVersion.AR4,
            **default_params
        )

        # AR4: CH4 GWP = 25
        expected_co2e = result['ch4_emissions_tonnes'] * Decimal('25')
        assert result['co2e_emissions_tonnes'] == pytest.approx(expected_co2e, rel=1e-6)

    def test_gwp_ar5(self, calculator, default_params):
        """Test calculation with AR5 GWP (CH4 = 28)."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            gwp_version=GWPVersion.AR5,
            **default_params
        )

        # AR5: CH4 GWP = 28
        expected_co2e = result['ch4_emissions_tonnes'] * Decimal('28')
        assert result['co2e_emissions_tonnes'] == pytest.approx(expected_co2e, rel=1e-6)

    def test_gwp_ar6(self, calculator, default_params):
        """Test calculation with AR6 GWP (CH4 = 27.9)."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            gwp_version=GWPVersion.AR6,
            **default_params
        )

        # AR6: CH4 GWP = 27.9
        expected_co2e = result['ch4_emissions_tonnes'] * Decimal('27.9')
        assert result['co2e_emissions_tonnes'] == pytest.approx(expected_co2e, rel=1e-5)


# Test Zero Mass Input
class TestZeroMass:
    """Test suite for zero mass edge cases."""

    def test_zero_mass_input(self, calculator, default_params):
        """Test calculation with zero waste mass."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('0.0'),
            waste_type=WasteType.MIXED_MSW,
            **default_params
        )

        assert result['ch4_emissions_tonnes'] == Decimal('0.0')
        assert result['co2e_emissions_tonnes'] == Decimal('0.0')


# Test Validation Errors
class TestValidation:
    """Test suite for input validation."""

    def test_negative_mass_raises_error(self, calculator, default_params):
        """Test negative waste mass raises validation error."""
        with pytest.raises(ValueError, match='waste mass must be non-negative'):
            calculator.calculate(
                waste_mass_tonnes=Decimal('-10.0'),
                waste_type=WasteType.MIXED_MSW,
                **default_params
            )

    def test_invalid_doc_raises_error(self, calculator, default_params):
        """Test invalid DOC (>1.0) raises validation error."""
        params = default_params.copy()
        params['doc'] = Decimal('1.5')  # Invalid: >1.0

        with pytest.raises(ValueError, match='DOC must be between 0 and 1'):
            calculator.calculate(
                waste_mass_tonnes=Decimal('100.0'),
                waste_type=WasteType.MIXED_MSW,
                **params
            )

    def test_invalid_k_raises_error(self, calculator, default_params):
        """Test negative decay constant raises validation error."""
        params = default_params.copy()
        params['k'] = Decimal('-0.09')

        with pytest.raises(ValueError, match='decay constant k must be positive'):
            calculator.calculate(
                waste_mass_tonnes=Decimal('100.0'),
                waste_type=WasteType.MIXED_MSW,
                **params
            )


# Test Decimal Precision
class TestDecimalPrecision:
    """Test suite for Decimal arithmetic precision."""

    def test_no_float_drift(self, calculator, default_params):
        """Test calculations avoid float drift using Decimal."""
        result1 = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            **default_params
        )

        result2 = calculator.calculate(
            waste_mass_tonnes=Decimal('100.0'),
            waste_type=WasteType.MIXED_MSW,
            **default_params
        )

        # Identical inputs → identical outputs (bit-perfect)
        assert result1['ch4_emissions_tonnes'] == result2['ch4_emissions_tonnes']

    def test_high_precision_calculations(self, calculator, default_params):
        """Test calculations maintain high precision."""
        result = calculator.calculate(
            waste_mass_tonnes=Decimal('0.001'),  # 1 kg
            waste_type=WasteType.MIXED_MSW,
            **default_params
        )

        # Should handle very small values without rounding errors
        assert result['ch4_emissions_tonnes'] > Decimal('0.0')
