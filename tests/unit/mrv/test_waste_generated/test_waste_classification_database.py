"""
Unit tests for Waste Classification Database Engine.

Tests waste classification, EWC codes, emission factors, treatment compatibility,
and data quality scoring for waste generated in operations.
"""

import pytest
from decimal import Decimal
from typing import Dict, Any, Optional, List

from greenlang.mrv.waste_generated.engines.waste_classification_database import (
    WasteClassificationDatabaseEngine,
    WasteCategory,
    EWCChapter,
    ClimateZone,
    LandfillType
)
from greenlang.mrv.waste_generated.models import (
    WasteType,
    TreatmentMethod
)


# Fixtures
@pytest.fixture
def db_engine():
    """Create WasteClassificationDatabaseEngine instance."""
    return WasteClassificationDatabaseEngine()


# Test Waste Classification
class TestWasteClassification:
    """Test suite for waste classification methods."""

    def test_classify_waste_by_ewc_code(self, db_engine):
        """Test waste classification using EWC code."""
        result = db_engine.classify_waste(
            ewc_code='20 03 01',
            composition=None,
            description=None
        )

        assert result['waste_category'] == WasteCategory.MUNICIPAL_SOLID_WASTE
        assert result['ewc_chapter'] == EWCChapter.MUNICIPAL_WASTES
        assert result['is_hazardous'] is False

    def test_classify_waste_by_composition(self, db_engine):
        """Test waste classification using composition."""
        composition = {
            'food_waste': Decimal('0.40'),
            'paper': Decimal('0.30'),
            'plastics': Decimal('0.20'),
            'other': Decimal('0.10')
        }

        result = db_engine.classify_waste(
            ewc_code=None,
            composition=composition,
            description=None
        )

        assert result['waste_category'] == WasteCategory.MUNICIPAL_SOLID_WASTE
        assert 'dominant_material' in result

    def test_classify_waste_by_description(self, db_engine):
        """Test waste classification using text description."""
        result = db_engine.classify_waste(
            ewc_code=None,
            composition=None,
            description='Mixed municipal waste from office operations'
        )

        assert result['waste_category'] is not None
        assert result['confidence_score'] <= 1.0

    def test_classify_hazardous_waste(self, db_engine):
        """Test classification of hazardous waste."""
        result = db_engine.classify_waste(
            ewc_code='13 02 05*',  # Asterisk denotes hazardous
            composition=None,
            description=None
        )

        assert result['is_hazardous'] is True

    def test_classify_industrial_waste(self, db_engine):
        """Test classification of industrial waste."""
        result = db_engine.classify_waste(
            ewc_code='10 01 01',  # Slag from coal power plants
            composition=None,
            description=None
        )

        assert result['ewc_chapter'] == EWCChapter.WASTES_FROM_THERMAL_PROCESSES


# Test Waste Category Retrieval
class TestGetWasteCategory:
    """Test suite for get_waste_category method."""

    @pytest.mark.parametrize('ewc_code,expected_category', [
        ('20 03 01', WasteCategory.MUNICIPAL_SOLID_WASTE),
        ('15 01 01', WasteCategory.PACKAGING),
        ('16 01 03', WasteCategory.END_OF_LIFE_VEHICLES),
        ('17 01 01', WasteCategory.CONSTRUCTION_DEMOLITION),
        ('19 12 12', WasteCategory.MECHANICAL_TREATMENT),
        ('08 01 11', WasteCategory.PAINT_VARNISH),
    ])
    def test_get_category_for_ewc_chapters(
        self,
        db_engine,
        ewc_code,
        expected_category
    ):
        """Test get_waste_category for all major EWC chapters."""
        category = db_engine.get_waste_category(ewc_code)

        assert category == expected_category


# Test Compatible Treatments
class TestGetCompatibleTreatments:
    """Test suite for get_compatible_treatments method."""

    def test_compatible_treatments_municipal_waste(self, db_engine):
        """Test compatible treatments for municipal solid waste."""
        treatments = db_engine.get_compatible_treatments(
            WasteType.MIXED_MSW
        )

        assert TreatmentMethod.LANDFILL in treatments
        assert TreatmentMethod.INCINERATION in treatments
        assert TreatmentMethod.COMPOSTING in treatments

    def test_compatible_treatments_food_waste(self, db_engine):
        """Test compatible treatments for food waste."""
        treatments = db_engine.get_compatible_treatments(
            WasteType.FOOD_ORGANIC
        )

        assert TreatmentMethod.COMPOSTING in treatments
        assert TreatmentMethod.ANAEROBIC_DIGESTION in treatments
        assert TreatmentMethod.LANDFILL in treatments

    def test_compatible_treatments_hazardous_waste(self, db_engine):
        """Test compatible treatments for hazardous waste."""
        treatments = db_engine.get_compatible_treatments(
            WasteType.HAZARDOUS
        )

        assert TreatmentMethod.INCINERATION in treatments
        # Hazardous waste should not allow standard composting
        assert TreatmentMethod.COMPOSTING not in treatments

    def test_compatible_treatments_construction_waste(self, db_engine):
        """Test compatible treatments for construction waste."""
        treatments = db_engine.get_compatible_treatments(
            WasteType.CONSTRUCTION_DEMOLITION
        )

        assert TreatmentMethod.RECYCLING in treatments
        assert TreatmentMethod.LANDFILL in treatments


# Test Hazardous Classification
class TestIsHazardous:
    """Test suite for is_hazardous method."""

    @pytest.mark.parametrize('ewc_code,expected_hazardous', [
        ('20 03 01', False),  # Mixed municipal waste
        ('13 02 05*', True),  # Mineral-based engine oils
        ('15 01 10*', True),  # Packaging containing hazardous residues
        ('17 01 01', False),  # Concrete
        ('16 06 01*', True),  # Lead batteries
        ('20 01 21*', True),  # Fluorescent tubes
    ])
    def test_is_hazardous(self, db_engine, ewc_code, expected_hazardous):
        """Test is_hazardous for hazardous and non-hazardous codes."""
        is_hazardous = db_engine.is_hazardous(ewc_code)

        assert is_hazardous == expected_hazardous


# Test Emission Factor Retrieval
class TestGetEmissionFactor:
    """Test suite for get_emission_factor method."""

    def test_get_emission_factor_epa_warm(self, db_engine):
        """Test emission factor retrieval from EPA WARM."""
        ef = db_engine.get_emission_factor(
            waste_type=WasteType.FOOD_ORGANIC,
            treatment_method=TreatmentMethod.LANDFILL,
            source='EPA_WARM'
        )

        assert ef is not None
        assert ef > 0

    def test_get_emission_factor_defra(self, db_engine):
        """Test emission factor retrieval from DEFRA."""
        ef = db_engine.get_emission_factor(
            waste_type=WasteType.MIXED_MSW,
            treatment_method=TreatmentMethod.LANDFILL,
            source='DEFRA'
        )

        assert ef is not None
        assert isinstance(ef, Decimal)

    def test_get_emission_factor_ipcc(self, db_engine):
        """Test emission factor retrieval from IPCC."""
        ef = db_engine.get_emission_factor(
            waste_type=WasteType.PAPER_CARDBOARD,
            treatment_method=TreatmentMethod.LANDFILL,
            source='IPCC'
        )

        assert ef is not None

    def test_get_emission_factor_invalid_combo(self, db_engine):
        """Test emission factor for invalid waste/treatment combo."""
        ef = db_engine.get_emission_factor(
            waste_type=WasteType.METAL,
            treatment_method=TreatmentMethod.COMPOSTING,  # Invalid
            source='EPA_WARM'
        )

        assert ef is None or ef == 0


# Test EPA WARM Factors
class TestGetEPAWARMFactor:
    """Test suite for get_epa_warm_factor method."""

    @pytest.mark.parametrize('material,treatment,expected_gt_zero', [
        ('food_waste', 'landfill', True),
        ('mixed_msw', 'landfill', True),
        ('paper', 'recycling', True),
        ('plastics', 'incineration', True),
        ('glass', 'recycling', True),
    ])
    def test_get_epa_warm_factor_known_materials(
        self,
        db_engine,
        material,
        treatment,
        expected_gt_zero
    ):
        """Test get_epa_warm_factor for known materials."""
        factor = db_engine.get_epa_warm_factor(material, treatment)

        if expected_gt_zero:
            assert factor > 0
        else:
            assert factor >= 0


# Test DEFRA Factors
class TestGetDEFRAFactor:
    """Test suite for get_defra_factor method."""

    @pytest.mark.parametrize('waste_type,treatment,expected_not_none', [
        (WasteType.MIXED_MSW, TreatmentMethod.LANDFILL, True),
        (WasteType.FOOD_ORGANIC, TreatmentMethod.COMPOSTING, True),
        (WasteType.PAPER_CARDBOARD, TreatmentMethod.RECYCLING, True),
        (WasteType.PLASTICS, TreatmentMethod.INCINERATION, True),
    ])
    def test_get_defra_factor_known_types(
        self,
        db_engine,
        waste_type,
        treatment,
        expected_not_none
    ):
        """Test get_defra_factor for known waste types."""
        factor = db_engine.get_defra_factor(waste_type, treatment)

        if expected_not_none:
            assert factor is not None
        else:
            assert factor is None or factor == 0


# Test EPA WARM to Metric Conversion
class TestConvertWARMToMetric:
    """Test suite for convert_warm_to_metric method."""

    def test_convert_short_tons_to_tonnes(self, db_engine):
        """Test conversion from US short tons to metric tonnes."""
        mtco2e_per_short_ton = Decimal('1.0')
        mtco2e_per_tonne = db_engine.convert_warm_to_metric(mtco2e_per_short_ton)

        # 1 short ton = 0.907185 metric tonnes
        expected = mtco2e_per_short_ton / Decimal('0.907185')

        assert mtco2e_per_tonne == pytest.approx(expected, rel=1e-5)

    def test_convert_zero(self, db_engine):
        """Test conversion of zero value."""
        result = db_engine.convert_warm_to_metric(Decimal('0.0'))

        assert result == Decimal('0.0')


# Test Best Available Factor
class TestGetBestAvailableFactor:
    """Test suite for get_best_available_factor fallback hierarchy."""

    def test_fallback_hierarchy_epa_first(self, db_engine):
        """Test fallback hierarchy prefers EPA WARM."""
        factor = db_engine.get_best_available_factor(
            waste_type=WasteType.FOOD_ORGANIC,
            treatment_method=TreatmentMethod.LANDFILL
        )

        # Should retrieve EPA WARM factor first
        assert factor is not None
        assert factor['source'] == 'EPA_WARM'

    def test_fallback_hierarchy_defra_second(self, db_engine):
        """Test fallback hierarchy uses DEFRA if EPA unavailable."""
        # Mock EPA unavailable
        factor = db_engine.get_best_available_factor(
            waste_type=WasteType.TEXTILES,  # Not in EPA
            treatment_method=TreatmentMethod.LANDFILL
        )

        # Should fall back to DEFRA or IPCC
        assert factor is not None
        assert factor['source'] in ['DEFRA', 'IPCC']

    def test_fallback_hierarchy_ipcc_last(self, db_engine):
        """Test fallback hierarchy uses IPCC as last resort."""
        factor = db_engine.get_best_available_factor(
            waste_type=WasteType.OTHER,
            treatment_method=TreatmentMethod.LANDFILL
        )

        # Should fall back to IPCC default
        assert factor is not None
        assert factor['source'] == 'IPCC'


# Test DOC (Degradable Organic Carbon)
class TestGetDOC:
    """Test suite for get_doc method."""

    @pytest.mark.parametrize('waste_type,expected_doc', [
        (WasteType.FOOD_ORGANIC, Decimal('0.15')),
        (WasteType.PAPER_CARDBOARD, Decimal('0.40')),
        (WasteType.WOOD, Decimal('0.43')),
        (WasteType.TEXTILES, Decimal('0.24')),
        (WasteType.MIXED_MSW, Decimal('0.18')),  # Weighted average
    ])
    def test_get_doc_for_waste_types(
        self,
        db_engine,
        waste_type,
        expected_doc
    ):
        """Test get_doc for all waste types."""
        doc = db_engine.get_doc(waste_type)

        assert doc == pytest.approx(expected_doc, rel=0.01)


# Test MCF (Methane Correction Factor)
class TestGetMCF:
    """Test suite for get_mcf method."""

    @pytest.mark.parametrize('landfill_type,expected_mcf', [
        (LandfillType.MANAGED_ANAEROBIC, Decimal('1.0')),
        (LandfillType.MANAGED_SEMI_AEROBIC, Decimal('0.5')),
        (LandfillType.UNMANAGED_DEEP, Decimal('0.8')),
        (LandfillType.UNMANAGED_SHALLOW, Decimal('0.4')),
        (LandfillType.UNCATEGORIZED, Decimal('0.6')),
    ])
    def test_get_mcf_for_landfill_types(
        self,
        db_engine,
        landfill_type,
        expected_mcf
    ):
        """Test get_mcf for all landfill types."""
        mcf = db_engine.get_mcf(landfill_type)

        assert mcf == expected_mcf


# Test Decay Rate (k)
class TestGetDecayRate:
    """Test suite for get_decay_rate method."""

    @pytest.mark.parametrize('climate_zone,waste_type,expected_k_range', [
        (ClimateZone.TROPICAL_WET, WasteType.FOOD_ORGANIC, (0.15, 0.25)),
        (ClimateZone.TROPICAL_DRY, WasteType.PAPER_CARDBOARD, (0.05, 0.08)),
        (ClimateZone.TEMPERATE_WET, WasteType.WOOD, (0.02, 0.04)),
        (ClimateZone.TEMPERATE_DRY, WasteType.MIXED_MSW, (0.03, 0.06)),
        (ClimateZone.BOREAL_WET, WasteType.FOOD_ORGANIC, (0.05, 0.10)),
    ])
    def test_get_decay_rate_climate_waste_combinations(
        self,
        db_engine,
        climate_zone,
        waste_type,
        expected_k_range
    ):
        """Test get_decay_rate for climate zone × waste type combinations."""
        k = db_engine.get_decay_rate(climate_zone, waste_type)

        assert k >= expected_k_range[0]
        assert k <= expected_k_range[1]


# Test Gas Capture Efficiency
class TestGetGasCaptureEfficiency:
    """Test suite for get_gas_capture_efficiency method."""

    def test_gas_capture_efficiency_modern_system(self, db_engine):
        """Test gas capture efficiency for modern LFG system."""
        efficiency = db_engine.get_gas_capture_efficiency(
            system_type='modern',
            year=2025
        )

        assert efficiency >= Decimal('0.75')
        assert efficiency <= Decimal('0.95')

    def test_gas_capture_efficiency_basic_system(self, db_engine):
        """Test gas capture efficiency for basic LFG system."""
        efficiency = db_engine.get_gas_capture_efficiency(
            system_type='basic',
            year=2025
        )

        assert efficiency >= Decimal('0.30')
        assert efficiency <= Decimal('0.60')

    def test_gas_capture_efficiency_no_system(self, db_engine):
        """Test gas capture efficiency with no capture system."""
        efficiency = db_engine.get_gas_capture_efficiency(
            system_type='none',
            year=2025
        )

        assert efficiency == Decimal('0.0')

    def test_gas_capture_efficiency_year_ramp_up(self, db_engine):
        """Test gas capture efficiency ramps up over time."""
        efficiency_year1 = db_engine.get_gas_capture_efficiency(
            system_type='modern',
            year=1
        )
        efficiency_year5 = db_engine.get_gas_capture_efficiency(
            system_type='modern',
            year=5
        )

        # Later years should have higher efficiency
        assert efficiency_year5 >= efficiency_year1


# Test Incineration Parameters
class TestGetIncinerationParams:
    """Test suite for get_incineration_params method."""

    @pytest.mark.parametrize('incinerator_type', [
        'municipal',
        'industrial',
        'hazardous',
        'wte'  # Waste-to-Energy
    ])
    def test_get_incineration_params_by_type(
        self,
        db_engine,
        incinerator_type
    ):
        """Test get_incineration_params for different incinerator types."""
        params = db_engine.get_incineration_params(incinerator_type)

        assert 'cf' in params  # Carbon fraction
        assert 'fcf' in params  # Fossil carbon fraction
        assert 'of' in params  # Oxidation factor
        assert 'ch4_ef' in params  # CH4 emission factor
        assert 'n2o_ef' in params  # N2O emission factor

    def test_incineration_params_wte_efficiency(self, db_engine):
        """Test WtE incinerator has energy recovery efficiency."""
        params = db_engine.get_incineration_params('wte')

        assert 'energy_recovery_efficiency' in params
        assert params['energy_recovery_efficiency'] > 0


# Test Composting Emission Factors
class TestGetCompostingEF:
    """Test suite for get_composting_ef method."""

    @pytest.mark.parametrize('composting_method', [
        'windrow',
        'static_pile',
        'in_vessel',
        'vermicomposting'
    ])
    def test_get_composting_ef_by_method(
        self,
        db_engine,
        composting_method
    ):
        """Test get_composting_ef for different composting methods."""
        ef = db_engine.get_composting_ef(composting_method)

        assert 'ch4_ef' in ef
        assert 'n2o_ef' in ef
        assert ef['ch4_ef'] >= 0
        assert ef['n2o_ef'] >= 0


# Test Anaerobic Digestion Leakage Rate
class TestGetADLeakageRate:
    """Test suite for get_ad_leakage_rate method."""

    @pytest.mark.parametrize('ad_type,expected_range', [
        ('covered_lagoon', (0.10, 0.30)),
        ('complete_mix', (0.01, 0.05)),
        ('plug_flow', (0.01, 0.05)),
    ])
    def test_get_ad_leakage_rate_by_type(
        self,
        db_engine,
        ad_type,
        expected_range
    ):
        """Test get_ad_leakage_rate for different AD types."""
        leakage = db_engine.get_ad_leakage_rate(ad_type)

        assert leakage >= expected_range[0]
        assert leakage <= expected_range[1]


# Test Wastewater MCF
class TestGetWastewaterMCF:
    """Test suite for get_wastewater_mcf method."""

    @pytest.mark.parametrize('treatment_type,expected_mcf', [
        ('aerobic', Decimal('0.0')),
        ('anaerobic_lagoon', Decimal('0.8')),
        ('septic_tank', Decimal('0.5')),
    ])
    def test_get_wastewater_mcf(
        self,
        db_engine,
        treatment_type,
        expected_mcf
    ):
        """Test get_wastewater_mcf for treatment types."""
        mcf = db_engine.get_wastewater_mcf(treatment_type)

        assert mcf == expected_mcf


# Test Wastewater BO (Biochemical Oxygen Demand)
class TestGetWastewaterBO:
    """Test suite for get_wastewater_bo method."""

    def test_get_wastewater_bo(self, db_engine):
        """Test get_wastewater_bo for wastewater characteristics."""
        bo = db_engine.get_wastewater_bo(
            industry_type='food_processing'
        )

        assert bo > 0
        assert isinstance(bo, Decimal)


# Test MSW Composition
class TestGetMSWComposition:
    """Test suite for get_msw_composition method."""

    def test_get_msw_composition_default(self, db_engine):
        """Test get_msw_composition with default region."""
        composition = db_engine.get_msw_composition(region='US')

        assert 'food_waste' in composition
        assert 'paper' in composition
        assert 'plastics' in composition

        # Sum should be ~1.0 (allowing for rounding)
        total = sum(composition.values())
        assert total == pytest.approx(Decimal('1.0'), rel=0.01)

    def test_get_msw_composition_regional_variation(self, db_engine):
        """Test MSW composition varies by region."""
        us_composition = db_engine.get_msw_composition(region='US')
        eu_composition = db_engine.get_msw_composition(region='EU')

        # Compositions should differ
        assert us_composition != eu_composition


# Test Data Quality Score
class TestGetDQIScore:
    """Test suite for get_dqi_score method."""

    def test_get_dqi_score_high_quality(self, db_engine):
        """Test DQI score for high-quality data."""
        dqi = db_engine.get_dqi_score(
            data_source='measured',
            temporal_representativeness=5,
            geographical_correlation=5,
            technological_correlation=5
        )

        assert dqi >= Decimal('4.0')

    def test_get_dqi_score_low_quality(self, db_engine):
        """Test DQI score for low-quality data."""
        dqi = db_engine.get_dqi_score(
            data_source='estimated',
            temporal_representativeness=1,
            geographical_correlation=1,
            technological_correlation=1
        )

        assert dqi <= Decimal('2.0')

    def test_get_dqi_score_range(self, db_engine):
        """Test DQI score is within valid range."""
        dqi = db_engine.get_dqi_score(
            data_source='default',
            temporal_representativeness=3,
            geographical_correlation=3,
            technological_correlation=3
        )

        assert dqi >= Decimal('1.0')
        assert dqi <= Decimal('5.0')


# Test Singleton Pattern
class TestSingletonPattern:
    """Test suite for WasteClassificationDatabaseEngine singleton."""

    def test_singleton_instance(self):
        """Test engine uses singleton pattern."""
        engine1 = WasteClassificationDatabaseEngine()
        engine2 = WasteClassificationDatabaseEngine()

        assert engine1 is engine2

    def test_singleton_state_shared(self):
        """Test singleton state is shared across instances."""
        engine1 = WasteClassificationDatabaseEngine()
        # Assume engine has a cache
        result1 = engine1.get_doc(WasteType.FOOD_ORGANIC)

        engine2 = WasteClassificationDatabaseEngine()
        result2 = engine2.get_doc(WasteType.FOOD_ORGANIC)

        assert result1 == result2
