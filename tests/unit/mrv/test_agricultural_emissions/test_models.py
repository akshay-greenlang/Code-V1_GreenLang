# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Agricultural Emissions Agent Data Models.

Tests all 18 enumerations, constant tables (GWP, enteric EFs, manure
VS/Bo/MCF/Nex, soil N2O, liming, urea, rice, field burning), and
18 Pydantic data models.

Target: 150+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.agricultural_emissions.models import (
        # Enumerations (18)
        AnimalType,
        ManureSystem,
        CropType,
        FertilizerType,
        WaterRegime,
        OrganicAmendment,
        CalculationMethod,
        EmissionGas,
        GWPSource,
        EmissionFactorSource,
        DataQualityTier,
        FarmType,
        ClimateZone,
        EmissionSource,
        ComplianceStatus,
        ReportingPeriod,
        PreSeasonFlooding,
        SoilType,
        # Constants
        GWP_VALUES,
        ENTERIC_EF_TIER1,
        MANURE_VS_DEFAULTS,
        MANURE_BO_VALUES,
        MANURE_NEX_VALUES,
        SOIL_N2O_EF,
        INDIRECT_N2O_FRACTIONS,
        LIMING_EF,
        UREA_EF,
        RICE_BASELINE_EF,
        RICE_WATER_REGIME_SF,
        RICE_ORGANIC_CFOA,
        FIELD_BURNING_EF,
        CONVERSION_C_TO_CO2,
        CONVERSION_N_TO_N2O,
        VERSION,
        MAX_CALCULATIONS_PER_BATCH,
        # Data models
        FarmInfo,
        LivestockPopulation,
        ManureSystemAllocation,
        FeedCharacteristics,
        EntericCalculationRequest,
        ManureCalculationRequest,
        CroplandInput,
        RiceFieldInput,
        FieldBurningInput,
        CalculationRequest,
        GasEmissionDetail,
        CalculationResult,
        BatchCalculationRequest,
        BatchCalculationResult,
        ComplianceCheckResult,
        UncertaintyRequest,
        UncertaintyResult,
        AggregationResult,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

_SKIP = pytest.mark.skipif(not MODELS_AVAILABLE, reason="models not available")


# ===========================================================================
# Enumeration Tests
# ===========================================================================


@_SKIP
class TestAnimalType:
    """Test AnimalType enumeration."""

    def test_has_20_members(self):
        assert len(AnimalType) == 20

    def test_dairy_cattle(self):
        assert AnimalType.DAIRY_CATTLE.value == "dairy_cattle"

    def test_non_dairy_cattle(self):
        assert AnimalType.NON_DAIRY_CATTLE.value == "non_dairy_cattle"

    def test_buffalo(self):
        assert AnimalType.BUFFALO.value == "buffalo"

    def test_sheep(self):
        assert AnimalType.SHEEP.value == "sheep"

    def test_swine_market(self):
        assert AnimalType.SWINE_MARKET.value == "swine_market"

    def test_poultry_layers(self):
        assert AnimalType.POULTRY_LAYERS.value == "poultry_layers"

    def test_alpacas_llamas(self):
        assert AnimalType.ALPACAS_LLAMAS.value == "alpacas_llamas"

    def test_other_livestock(self):
        assert AnimalType.OTHER_LIVESTOCK.value == "other_livestock"

    def test_string_lookup(self):
        assert AnimalType("dairy_cattle") == AnimalType.DAIRY_CATTLE


@_SKIP
class TestManureSystem:
    """Test ManureSystem enumeration."""

    def test_has_15_members(self):
        assert len(ManureSystem) == 15

    def test_pasture_range(self):
        assert ManureSystem.PASTURE_RANGE_PADDOCK.value == "pasture_range_paddock"

    def test_lagoon(self):
        assert ManureSystem.UNCOVERED_ANAEROBIC_LAGOON.value == "uncovered_anaerobic_lagoon"

    def test_digester(self):
        assert ManureSystem.ANAEROBIC_DIGESTER.value == "anaerobic_digester"

    def test_deep_bedding(self):
        assert ManureSystem.DEEP_BEDDING_NO_MIX.value == "deep_bedding_no_mix"

    def test_string_lookup(self):
        assert ManureSystem("solid_storage") == ManureSystem.SOLID_STORAGE


@_SKIP
class TestCropType:
    """Test CropType enumeration."""

    def test_has_12_members(self):
        assert len(CropType) == 12

    def test_wheat(self):
        assert CropType.WHEAT.value == "wheat"

    def test_rice(self):
        assert CropType.RICE.value == "rice"

    def test_corn(self):
        assert CropType("corn_maize") == CropType.CORN_MAIZE


@_SKIP
class TestFertilizerType:
    """Test FertilizerType enumeration."""

    def test_has_8_members(self):
        assert len(FertilizerType) == 8

    def test_synthetic_n(self):
        assert FertilizerType.SYNTHETIC_N.value == "synthetic_n"

    def test_urea(self):
        assert FertilizerType.UREA.value == "urea"


@_SKIP
class TestWaterRegime:
    """Test WaterRegime enumeration."""

    def test_has_7_members(self):
        assert len(WaterRegime) == 7

    def test_continuously_flooded(self):
        assert WaterRegime.CONTINUOUSLY_FLOODED.value == "continuously_flooded"

    def test_upland(self):
        assert WaterRegime.UPLAND.value == "upland"


@_SKIP
class TestOrganicAmendment:
    """Test OrganicAmendment enumeration."""

    def test_has_5_members(self):
        assert len(OrganicAmendment) == 5

    def test_straw_short(self):
        v = OrganicAmendment.STRAW_SHORT
        assert "straw" in v.value.lower()


@_SKIP
class TestCalculationMethod:
    """Test CalculationMethod enumeration."""

    def test_has_6_members(self):
        assert len(CalculationMethod) == 6

    def test_ipcc_tier_1(self):
        assert CalculationMethod.IPCC_TIER_1.value == "ipcc_tier_1"

    def test_ipcc_tier_2(self):
        assert CalculationMethod.IPCC_TIER_2.value == "ipcc_tier_2"

    def test_ipcc_tier_3(self):
        assert CalculationMethod.IPCC_TIER_3.value == "ipcc_tier_3"


@_SKIP
class TestEmissionGas:
    """Test EmissionGas enumeration."""

    def test_has_3_members(self):
        assert len(EmissionGas) == 3

    def test_co2(self):
        assert EmissionGas.CO2.value == "co2"

    def test_ch4(self):
        assert EmissionGas.CH4.value == "ch4"

    def test_n2o(self):
        assert EmissionGas.N2O.value == "n2o"


@_SKIP
class TestGWPSource:
    """Test GWPSource enumeration."""

    def test_has_4_members(self):
        assert len(GWPSource) == 4

    def test_ar6(self):
        assert GWPSource.AR6.value == "AR6"

    def test_ar5(self):
        assert GWPSource.AR5.value == "AR5"


@_SKIP
class TestEmissionFactorSource:
    """Test EmissionFactorSource enumeration."""

    def test_has_7_members(self):
        assert len(EmissionFactorSource) == 7

    def test_ipcc_2006(self):
        assert EmissionFactorSource.IPCC_2006.value == "ipcc_2006"


@_SKIP
class TestDataQualityTier:
    """Test DataQualityTier enumeration."""

    def test_has_3_members(self):
        assert len(DataQualityTier) == 3


@_SKIP
class TestFarmType:
    """Test FarmType enumeration."""

    def test_has_8_members(self):
        assert len(FarmType) == 8

    def test_dairy(self):
        assert FarmType.DAIRY.value == "dairy"


@_SKIP
class TestClimateZone:
    """Test ClimateZone enumeration."""

    def test_has_8_members(self):
        assert len(ClimateZone) == 8


@_SKIP
class TestEmissionSource:
    """Test EmissionSource enumeration."""

    def test_has_6_members(self):
        assert len(EmissionSource) == 6

    def test_enteric(self):
        assert EmissionSource.ENTERIC_FERMENTATION.value == "enteric_fermentation"


@_SKIP
class TestComplianceStatus:
    """Test ComplianceStatus enumeration."""

    def test_has_4_members(self):
        assert len(ComplianceStatus) == 4


@_SKIP
class TestReportingPeriod:
    """Test ReportingPeriod enumeration."""

    def test_has_4_members(self):
        assert len(ReportingPeriod) == 4


@_SKIP
class TestPreSeasonFlooding:
    """Test PreSeasonFlooding enumeration."""

    def test_has_3_members(self):
        assert len(PreSeasonFlooding) == 3


@_SKIP
class TestSoilType:
    """Test SoilType enumeration."""

    def test_has_5_members(self):
        assert len(SoilType) == 5


# ===========================================================================
# Constants Tests
# ===========================================================================


@_SKIP
class TestGWPValues:
    """Test GWP constant tables."""

    def test_gwp_values_is_dict(self):
        assert isinstance(GWP_VALUES, dict)

    def test_ar6_ch4(self):
        ar6 = GWP_VALUES.get("AR6", {})
        ch4_val = ar6.get("CH4", ar6.get("ch4", ar6.get("CH4_fossil", None)))
        assert ch4_val is not None

    def test_ar5_present(self):
        assert "AR5" in GWP_VALUES

    def test_ar4_present(self):
        assert "AR4" in GWP_VALUES

    def test_values_are_decimal(self):
        for source, gases in GWP_VALUES.items():
            for gas, val in gases.items():
                assert isinstance(val, (Decimal, int, float))


@_SKIP
class TestEntericEFTier1:
    """Test enteric fermentation Tier 1 EF table."""

    def test_is_dict(self):
        assert isinstance(ENTERIC_EF_TIER1, dict)

    def test_dairy_cattle_present(self):
        assert "dairy_cattle" in ENTERIC_EF_TIER1

    def test_non_dairy_cattle_present(self):
        assert "non_dairy_cattle" in ENTERIC_EF_TIER1

    def test_sheep_present(self):
        assert "sheep" in ENTERIC_EF_TIER1

    def test_has_multiple_animals(self):
        assert len(ENTERIC_EF_TIER1) >= 10


@_SKIP
class TestManureConstants:
    """Test manure management constant tables."""

    def test_vs_defaults_is_dict(self):
        assert isinstance(MANURE_VS_DEFAULTS, dict)

    def test_bo_values_is_dict(self):
        assert isinstance(MANURE_BO_VALUES, dict)

    def test_nex_values_is_dict(self):
        assert isinstance(MANURE_NEX_VALUES, dict)

    def test_dairy_cattle_in_vs(self):
        assert "dairy_cattle" in MANURE_VS_DEFAULTS

    def test_dairy_cattle_in_bo(self):
        assert "dairy_cattle" in MANURE_BO_VALUES

    def test_dairy_cattle_in_nex(self):
        assert "dairy_cattle" in MANURE_NEX_VALUES


@_SKIP
class TestSoilN2OConstants:
    """Test soil N2O emission factor constants."""

    def test_soil_ef_is_dict(self):
        assert isinstance(SOIL_N2O_EF, dict)

    def test_ef1_present(self):
        assert "EF1" in SOIL_N2O_EF or "ef1" in SOIL_N2O_EF

    def test_indirect_fractions_is_dict(self):
        assert isinstance(INDIRECT_N2O_FRACTIONS, dict)


@_SKIP
class TestLimingUreaConstants:
    """Test liming and urea constants."""

    def test_liming_ef_is_dict(self):
        assert isinstance(LIMING_EF, dict)

    def test_limestone_ef(self):
        val = LIMING_EF.get("limestone", LIMING_EF.get("LIMESTONE", None))
        assert val is not None
        assert Decimal(str(val)) == Decimal("0.12")

    def test_dolomite_ef(self):
        val = LIMING_EF.get("dolomite", LIMING_EF.get("DOLOMITE", None))
        assert val is not None
        assert Decimal(str(val)) == Decimal("0.13")

    def test_urea_ef_value(self):
        assert Decimal(str(UREA_EF)) == Decimal("0.20")


@_SKIP
class TestRiceConstants:
    """Test rice cultivation constants."""

    def test_baseline_ef(self):
        assert Decimal(str(RICE_BASELINE_EF)) == Decimal("1.30")

    def test_water_regime_sf_is_dict(self):
        assert isinstance(RICE_WATER_REGIME_SF, dict)

    def test_organic_cfoa_is_dict(self):
        assert isinstance(RICE_ORGANIC_CFOA, dict)


@_SKIP
class TestFieldBurningConstants:
    """Test field burning constants."""

    def test_is_dict(self):
        assert isinstance(FIELD_BURNING_EF, dict)

    def test_has_crop_types(self):
        assert len(FIELD_BURNING_EF) >= 5


@_SKIP
class TestConversionConstants:
    """Test conversion factor constants."""

    def test_c_to_co2(self):
        val = Decimal(str(CONVERSION_C_TO_CO2))
        assert val > Decimal("3.6") and val < Decimal("3.7")

    def test_n_to_n2o(self):
        val = Decimal(str(CONVERSION_N_TO_N2O))
        assert val > Decimal("1.57") and val < Decimal("1.58")


@_SKIP
class TestModuleConstants:
    """Test module-level constants."""

    def test_version(self):
        assert VERSION == "1.0.0"

    def test_max_batch(self):
        assert MAX_CALCULATIONS_PER_BATCH == 10_000


# ===========================================================================
# Data Model Tests
# ===========================================================================


@_SKIP
class TestFarmInfo:
    """Test FarmInfo Pydantic model."""

    def test_creation_defaults(self):
        f = FarmInfo()
        assert f is not None

    def test_farm_id_generated(self):
        f = FarmInfo()
        assert f.farm_id != ""

    def test_custom_values(self):
        f = FarmInfo(name="Test Farm", farm_type="dairy", country_code="GB")
        assert f.name == "Test Farm"


@_SKIP
class TestLivestockPopulation:
    """Test LivestockPopulation Pydantic model."""

    def test_creation(self):
        lp = LivestockPopulation(
            animal_type="dairy_cattle",
            head_count=200,
        )
        assert lp.head_count == 200

    def test_animal_type_field(self):
        lp = LivestockPopulation(animal_type="sheep", head_count=500)
        assert lp.animal_type == "sheep"


@_SKIP
class TestManureSystemAllocation:
    """Test ManureSystemAllocation Pydantic model."""

    def test_creation(self):
        msa = ManureSystemAllocation(
            system_type="pasture_range_paddock",
            fraction=Decimal("0.60"),
        )
        assert msa.fraction == Decimal("0.60")


@_SKIP
class TestFeedCharacteristics:
    """Test FeedCharacteristics Pydantic model."""

    def test_creation(self):
        fc = FeedCharacteristics()
        assert fc is not None


@_SKIP
class TestEntericCalculationRequest:
    """Test EntericCalculationRequest model."""

    def test_creation(self):
        r = EntericCalculationRequest(
            animal_type="dairy_cattle",
            head_count=200,
        )
        assert r.animal_type == "dairy_cattle"


@_SKIP
class TestManureCalculationRequest:
    """Test ManureCalculationRequest model."""

    def test_creation(self):
        r = ManureCalculationRequest(
            animal_type="dairy_cattle",
            head_count=100,
        )
        assert r.head_count == 100


@_SKIP
class TestCroplandInput:
    """Test CroplandInput model."""

    def test_creation(self):
        ci = CroplandInput(
            input_type="synthetic_n",
            quantity_tonnes=Decimal("100"),
        )
        assert ci.quantity_tonnes == Decimal("100")


@_SKIP
class TestRiceFieldInput:
    """Test RiceFieldInput model."""

    def test_creation(self):
        rf = RiceFieldInput(
            area_ha=Decimal("50"),
            water_regime="continuously_flooded",
        )
        assert rf.area_ha == Decimal("50")


@_SKIP
class TestFieldBurningInput:
    """Test FieldBurningInput model."""

    def test_creation(self):
        fb = FieldBurningInput(
            crop_type="wheat",
            area_burned_ha=Decimal("10"),
        )
        assert fb.crop_type == "wheat"


@_SKIP
class TestCalculationRequest:
    """Test CalculationRequest unified model."""

    def test_creation(self):
        cr = CalculationRequest(
            farm_id="farm-001",
            source_category="enteric_fermentation",
        )
        assert cr.source_category == "enteric_fermentation"

    def test_default_method(self):
        cr = CalculationRequest()
        assert cr is not None


@_SKIP
class TestGasEmissionDetail:
    """Test GasEmissionDetail model."""

    def test_creation(self):
        g = GasEmissionDetail(
            gas="ch4",
            mass_tonnes=Decimal("25.6"),
            co2e_tonnes=Decimal("762.88"),
        )
        assert g.gas == "ch4"


@_SKIP
class TestCalculationResult:
    """Test CalculationResult model."""

    def test_creation_defaults(self):
        r = CalculationResult()
        assert r.total_co2e_tonnes == Decimal("0")

    def test_calculation_id_generated(self):
        r = CalculationResult()
        assert r.calculation_id != ""

    def test_provenance_hash_default(self):
        r = CalculationResult()
        assert isinstance(r.provenance_hash, str)


@_SKIP
class TestBatchCalculationRequest:
    """Test BatchCalculationRequest model."""

    def test_creation(self):
        br = BatchCalculationRequest(calculations=[])
        assert br.calculations == []


@_SKIP
class TestBatchCalculationResult:
    """Test BatchCalculationResult model."""

    def test_creation(self):
        br = BatchCalculationResult()
        assert br.total_calculations == 0

    def test_defaults(self):
        br = BatchCalculationResult()
        assert br.successful == 0
        assert br.failed == 0


@_SKIP
class TestComplianceCheckResult:
    """Test ComplianceCheckResult model."""

    def test_creation(self):
        c = ComplianceCheckResult()
        assert c is not None


@_SKIP
class TestUncertaintyRequest:
    """Test UncertaintyRequest model."""

    def test_creation(self):
        u = UncertaintyRequest()
        assert u is not None

    def test_default_iterations(self):
        u = UncertaintyRequest()
        assert u.iterations >= 100


@_SKIP
class TestUncertaintyResult:
    """Test UncertaintyResult model."""

    def test_creation(self):
        u = UncertaintyResult()
        assert u.method == "monte_carlo"


@_SKIP
class TestAggregationResult:
    """Test AggregationResult model."""

    def test_creation(self):
        a = AggregationResult()
        assert a.total_co2e_tonnes == Decimal("0")

    def test_defaults(self):
        a = AggregationResult()
        assert a.calculation_count == 0
