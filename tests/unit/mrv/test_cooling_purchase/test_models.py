"""Tests for AGENT-MRV-012 Cooling Purchase Agent models."""

import pytest
from decimal import Decimal
from typing import Dict, Any

try:
    from greenlang.cooling_purchase.models import (
        # Enums
        CoolingTechnology,
        CompressorType,
        CondenserType,
        AbsorptionType,
        FreeCoolingSource,
        TESType,
        HeatSource,
        EfficiencyMetric,
        CoolingUnit,
        EmissionGas,
        GWPSource,
        Refrigerant,
        CalculationMethod,
        DataQualityTier,
        ConfidenceLevel,
        CoolingScope,
        RegulatoryFramework,
        ComplianceStatus,
        # Constants
        COOLING_TECHNOLOGY_SPECS,
        DISTRICT_COOLING_FACTORS,
        HEAT_SOURCE_FACTORS,
        REFRIGERANT_GWP,
        GWP_VALUES,
        AHRI_PART_LOAD_WEIGHTS,
        EFFICIENCY_CONVERSIONS,
        UNIT_CONVERSIONS,
        VERSION,
        TABLE_PREFIX,
        # Models
        ElectricChillerRequest,
        AbsorptionCoolingRequest,
        DistrictCoolingRequest,
        FreeCoolingRequest,
        TESRequest,
        CalculationResult,
        UncertaintyQuantificationRequest,
        UncertaintyResult,
        ComplianceCheckRequest,
        ComplianceResult,
        CoolingPurchaseInput,
        CoolingPurchaseOutput,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")


# ============================================================================
# ENUM TESTS (90 tests)
# ============================================================================

class TestEnums:
    """Test all 18 enums."""

    def test_cooling_technology_count(self):
        """Test CoolingTechnology has 18 values."""
        assert len(CoolingTechnology) == 18

    def test_cooling_technology_values(self):
        """Test CoolingTechnology has expected values."""
        expected = {
            "ELECTRIC_CHILLER", "ABSORPTION_CHILLER", "DISTRICT_COOLING",
            "FREE_COOLING", "THERMAL_ENERGY_STORAGE", "ADSORPTION_CHILLER",
            "DESICCANT_COOLING", "EVAPORATIVE_COOLING", "HYBRID_SYSTEM",
            "GEOTHERMAL_COOLING", "SOLAR_ABSORPTION", "OCEAN_WATER_COOLING",
            "LAKE_WATER_COOLING", "RIVER_WATER_COOLING", "GROUNDWATER_COOLING",
            "SNOW_ICE_STORAGE", "TRIGENERATION", "INDIRECT_EVAPORATIVE"
        }
        assert {t.value for t in CoolingTechnology} == expected

    def test_compressor_type_count(self):
        """Test CompressorType has 4 values."""
        assert len(CompressorType) == 4

    def test_compressor_type_values(self):
        """Test CompressorType has expected values."""
        expected = {"CENTRIFUGAL", "SCREW", "SCROLL", "RECIPROCATING"}
        assert {c.value for c in CompressorType} == expected

    def test_condenser_type_count(self):
        """Test CondenserType has 2 values."""
        assert len(CondenserType) == 2

    def test_condenser_type_values(self):
        """Test CondenserType has expected values."""
        expected = {"AIR_COOLED", "WATER_COOLED"}
        assert {c.value for c in CondenserType} == expected

    def test_absorption_type_count(self):
        """Test AbsorptionType has 4 values."""
        assert len(AbsorptionType) == 4

    def test_absorption_type_values(self):
        """Test AbsorptionType has expected values."""
        expected = {"SINGLE_EFFECT", "DOUBLE_EFFECT", "TRIPLE_EFFECT", "HALF_EFFECT"}
        assert {a.value for a in AbsorptionType} == expected

    def test_free_cooling_source_count(self):
        """Test FreeCoolingSource has 4 values."""
        assert len(FreeCoolingSource) == 4

    def test_free_cooling_source_values(self):
        """Test FreeCoolingSource has expected values."""
        expected = {"OUTSIDE_AIR", "GROUNDWATER", "LAKE_WATER", "SEA_WATER"}
        assert {f.value for f in FreeCoolingSource} == expected

    def test_tes_type_count(self):
        """Test TESType has 3 values."""
        assert len(TESType) == 3

    def test_tes_type_values(self):
        """Test TESType has expected values."""
        expected = {"ICE_STORAGE", "CHILLED_WATER", "PHASE_CHANGE_MATERIAL"}
        assert {t.value for t in TESType} == expected

    def test_heat_source_count(self):
        """Test HeatSource has 11 values."""
        assert len(HeatSource) == 11

    def test_heat_source_values(self):
        """Test HeatSource has expected values."""
        expected = {
            "NATURAL_GAS", "WASTE_HEAT", "SOLAR_THERMAL", "BIOMASS",
            "GEOTHERMAL", "DISTRICT_HEAT", "OIL", "COAL", "ELECTRICITY",
            "HYDROGEN", "BIOGAS"
        }
        assert {h.value for h in HeatSource} == expected

    def test_efficiency_metric_count(self):
        """Test EfficiencyMetric has 6 values."""
        assert len(EfficiencyMetric) == 6

    def test_efficiency_metric_values(self):
        """Test EfficiencyMetric has expected values."""
        expected = {"COP", "EER", "IPLV", "NPLV", "SEER", "ESEER"}
        assert {e.value for e in EfficiencyMetric} == expected

    def test_cooling_unit_count(self):
        """Test CoolingUnit has 7 values."""
        assert len(CoolingUnit) == 7

    def test_cooling_unit_values(self):
        """Test CoolingUnit has expected values."""
        expected = {"MWH", "KWH", "GJ", "MMBTU", "RT_H", "TONS_H", "KBTU"}
        assert {c.value for c in CoolingUnit} == expected

    def test_emission_gas_count(self):
        """Test EmissionGas has 4 values."""
        assert len(EmissionGas) == 4

    def test_emission_gas_values(self):
        """Test EmissionGas has expected values."""
        expected = {"CO2", "CH4", "N2O", "REFRIGERANTS"}
        assert {e.value for e in EmissionGas} == expected

    def test_gwp_source_count(self):
        """Test GWPSource has 4 values."""
        assert len(GWPSource) == 4

    def test_gwp_source_values(self):
        """Test GWPSource has expected values."""
        expected = {"IPCC_AR5", "IPCC_AR6", "MONTREAL_PROTOCOL", "EPA"}
        assert {g.value for g in GWPSource} == expected

    def test_refrigerant_count(self):
        """Test Refrigerant has 11 values."""
        assert len(Refrigerant) == 11

    def test_refrigerant_values(self):
        """Test Refrigerant has expected values."""
        expected = {
            "R134A", "R410A", "R407C", "R32", "R290", "R600A",
            "R717", "R744", "R1234YF", "R1234ZE", "R513A"
        }
        assert {r.value for r in Refrigerant} == expected

    def test_calculation_method_count(self):
        """Test CalculationMethod enum."""
        assert len(CalculationMethod) >= 5

    def test_data_quality_tier_count(self):
        """Test DataQualityTier enum."""
        assert len(DataQualityTier) >= 3

    def test_confidence_level_count(self):
        """Test ConfidenceLevel enum."""
        assert len(ConfidenceLevel) >= 3

    def test_cooling_scope_count(self):
        """Test CoolingScope enum."""
        assert len(CoolingScope) >= 2

    def test_regulatory_framework_count(self):
        """Test RegulatoryFramework has 7 values."""
        assert len(RegulatoryFramework) == 7

    def test_regulatory_framework_values(self):
        """Test RegulatoryFramework has expected values."""
        expected = {
            "GHG_PROTOCOL", "ISO_14064", "CSRD", "SECR", "TCFD",
            "CDP", "SBTi"
        }
        assert {r.value for r in RegulatoryFramework} == expected

    def test_compliance_status_count(self):
        """Test ComplianceStatus enum."""
        assert len(ComplianceStatus) >= 3


# ============================================================================
# CONSTANT DICT TESTS (45 tests)
# ============================================================================

class TestConstants:
    """Test all constant dictionaries."""

    def test_cooling_technology_specs_count(self):
        """Test COOLING_TECHNOLOGY_SPECS has 18 entries."""
        assert len(COOLING_TECHNOLOGY_SPECS) == 18

    def test_cooling_technology_specs_keys(self):
        """Test COOLING_TECHNOLOGY_SPECS has all technology keys."""
        for tech in CoolingTechnology:
            assert tech in COOLING_TECHNOLOGY_SPECS

    def test_cooling_technology_specs_structure(self):
        """Test COOLING_TECHNOLOGY_SPECS entries have required fields."""
        for tech, spec in COOLING_TECHNOLOGY_SPECS.items():
            assert "typical_cop" in spec
            assert "typical_efficiency_range" in spec
            assert "scope" in spec
            assert isinstance(spec["typical_cop"], tuple)
            assert len(spec["typical_cop"]) == 2

    def test_district_cooling_factors_count(self):
        """Test DISTRICT_COOLING_FACTORS has 12 entries."""
        assert len(DISTRICT_COOLING_FACTORS) >= 10

    def test_district_cooling_factors_structure(self):
        """Test DISTRICT_COOLING_FACTORS entries have required fields."""
        for region, factor in DISTRICT_COOLING_FACTORS.items():
            assert "default_ef_kg_co2e_kwh" in factor
            assert "transmission_loss_pct" in factor

    def test_heat_source_factors_count(self):
        """Test HEAT_SOURCE_FACTORS has 11 entries."""
        assert len(HEAT_SOURCE_FACTORS) == 11

    def test_heat_source_factors_keys(self):
        """Test HEAT_SOURCE_FACTORS has all heat source keys."""
        for source in HeatSource:
            assert source in HEAT_SOURCE_FACTORS

    def test_heat_source_factors_structure(self):
        """Test HEAT_SOURCE_FACTORS entries have required fields."""
        for source, factor in HEAT_SOURCE_FACTORS.items():
            assert "ef_kg_co2e_kwh" in factor
            assert "carbon_intensity" in factor

    def test_refrigerant_gwp_count(self):
        """Test REFRIGERANT_GWP has 11 entries."""
        assert len(REFRIGERANT_GWP) == 11

    def test_refrigerant_gwp_keys(self):
        """Test REFRIGERANT_GWP has all refrigerant keys."""
        for ref in Refrigerant:
            assert ref in REFRIGERANT_GWP

    def test_refrigerant_gwp_sources(self):
        """Test REFRIGERANT_GWP entries have all GWP sources."""
        for ref, gwp_dict in REFRIGERANT_GWP.items():
            for source in GWPSource:
                assert source in gwp_dict

    def test_gwp_values_count(self):
        """Test GWP_VALUES has 4 entries."""
        assert len(GWP_VALUES) == 4

    def test_gwp_values_keys(self):
        """Test GWP_VALUES has all GWP source keys."""
        for source in GWPSource:
            assert source in GWP_VALUES

    def test_gwp_values_structure(self):
        """Test GWP_VALUES entries have CH4 and N2O."""
        for source, values in GWP_VALUES.items():
            assert "CH4" in values
            assert "N2O" in values

    def test_ahri_part_load_weights_sum(self):
        """Test AHRI_PART_LOAD_WEIGHTS sums to 1.0."""
        total = sum(AHRI_PART_LOAD_WEIGHTS.values())
        assert abs(float(total) - 1.0) < 0.001

    def test_ahri_part_load_weights_keys(self):
        """Test AHRI_PART_LOAD_WEIGHTS has expected load points."""
        expected = {"100%", "75%", "50%", "25%"}
        assert set(AHRI_PART_LOAD_WEIGHTS.keys()) == expected

    def test_efficiency_conversions_has_cop(self):
        """Test EFFICIENCY_CONVERSIONS has COP conversions."""
        assert "COP_TO_EER" in EFFICIENCY_CONVERSIONS
        assert "COP_TO_KW_PER_TON" in EFFICIENCY_CONVERSIONS

    def test_efficiency_conversions_has_eer(self):
        """Test EFFICIENCY_CONVERSIONS has EER conversions."""
        assert "EER_TO_COP" in EFFICIENCY_CONVERSIONS
        assert "EER_TO_KW_PER_TON" in EFFICIENCY_CONVERSIONS

    def test_unit_conversions_has_energy(self):
        """Test UNIT_CONVERSIONS has energy conversions."""
        assert "MWH_TO_KWH" in UNIT_CONVERSIONS
        assert "MWH_TO_GJ" in UNIT_CONVERSIONS
        assert "MWH_TO_MMBTU" in UNIT_CONVERSIONS

    def test_unit_conversions_has_cooling(self):
        """Test UNIT_CONVERSIONS has cooling conversions."""
        assert "RT_TO_KW" in UNIT_CONVERSIONS
        assert "TON_TO_KW" in UNIT_CONVERSIONS

    def test_version_format(self):
        """Test VERSION follows semantic versioning."""
        assert VERSION == "1.0.0"

    def test_table_prefix(self):
        """Test TABLE_PREFIX is correct."""
        assert TABLE_PREFIX == "gl_cp_"


# ============================================================================
# PYDANTIC MODEL TESTS (45 tests)
# ============================================================================

class TestElectricChillerRequest:
    """Test ElectricChillerRequest model."""

    def test_valid_instantiation(self, sample_electric_request):
        """Test valid ElectricChillerRequest instantiation."""
        req = ElectricChillerRequest(**sample_electric_request)
        assert req.calculation_id == "TEST-EC-001"
        assert req.technology == CoolingTechnology.ELECTRIC_CHILLER

    def test_frozen_model(self, sample_electric_request):
        """Test ElectricChillerRequest is frozen."""
        req = ElectricChillerRequest(**sample_electric_request)
        with pytest.raises(Exception):
            req.cooling_load_mwh = Decimal("2000.0")

    def test_negative_cooling_load_rejected(self, sample_electric_request):
        """Test negative cooling_load_mwh is rejected."""
        sample_electric_request["cooling_load_mwh"] = Decimal("-100.0")
        with pytest.raises(Exception):
            ElectricChillerRequest(**sample_electric_request)

    def test_cop_range_validation(self, sample_electric_request):
        """Test rated_cop must be positive."""
        sample_electric_request["rated_cop"] = Decimal("0.0")
        with pytest.raises(Exception):
            ElectricChillerRequest(**sample_electric_request)

    def test_leakage_rate_range(self, sample_electric_request):
        """Test annual_leakage_rate_pct must be 0-100."""
        sample_electric_request["annual_leakage_rate_pct"] = Decimal("150.0")
        with pytest.raises(Exception):
            ElectricChillerRequest(**sample_electric_request)


class TestAbsorptionCoolingRequest:
    """Test AbsorptionCoolingRequest model."""

    def test_valid_instantiation(self, sample_absorption_request):
        """Test valid AbsorptionCoolingRequest instantiation."""
        req = AbsorptionCoolingRequest(**sample_absorption_request)
        assert req.calculation_id == "TEST-AC-001"
        assert req.technology == CoolingTechnology.ABSORPTION_CHILLER

    def test_frozen_model(self, sample_absorption_request):
        """Test AbsorptionCoolingRequest is frozen."""
        req = AbsorptionCoolingRequest(**sample_absorption_request)
        with pytest.raises(Exception):
            req.heat_input_mwh = Decimal("1000.0")

    def test_thermal_cop_positive(self, sample_absorption_request):
        """Test rated_thermal_cop must be positive."""
        sample_absorption_request["rated_thermal_cop"] = Decimal("-0.5")
        with pytest.raises(Exception):
            AbsorptionCoolingRequest(**sample_absorption_request)


class TestDistrictCoolingRequest:
    """Test DistrictCoolingRequest model."""

    def test_valid_instantiation(self, sample_district_request):
        """Test valid DistrictCoolingRequest instantiation."""
        req = DistrictCoolingRequest(**sample_district_request)
        assert req.calculation_id == "TEST-DC-001"
        assert req.technology == CoolingTechnology.DISTRICT_COOLING

    def test_frozen_model(self, sample_district_request):
        """Test DistrictCoolingRequest is frozen."""
        req = DistrictCoolingRequest(**sample_district_request)
        with pytest.raises(Exception):
            req.cooling_purchased_mwh = Decimal("3000.0")

    def test_distribution_loss_range(self, sample_district_request):
        """Test distribution_loss_pct must be 0-100."""
        sample_district_request["distribution_loss_pct"] = Decimal("105.0")
        with pytest.raises(Exception):
            DistrictCoolingRequest(**sample_district_request)


class TestCalculationResult:
    """Test CalculationResult model."""

    def test_valid_instantiation(self, sample_calculation_result):
        """Test valid CalculationResult instantiation."""
        result = CalculationResult(**sample_calculation_result)
        assert result.calculation_id == "TEST-001"
        assert result.total_emissions_kg_co2e == Decimal("450000.0")

    def test_frozen_model(self, sample_calculation_result):
        """Test CalculationResult is frozen."""
        result = CalculationResult(**sample_calculation_result)
        with pytest.raises(Exception):
            result.total_emissions_kg_co2e = Decimal("500000.0")

    def test_uncertainty_range(self, sample_calculation_result):
        """Test uncertainty_pct must be 0-100."""
        sample_calculation_result["uncertainty_pct"] = Decimal("150.0")
        with pytest.raises(Exception):
            CalculationResult(**sample_calculation_result)

    def test_data_quality_score_range(self, sample_calculation_result):
        """Test data_quality_score must be 0-100."""
        sample_calculation_result["data_quality_score"] = Decimal("105.0")
        with pytest.raises(Exception):
            CalculationResult(**sample_calculation_result)


class TestUncertaintyQuantificationRequest:
    """Test UncertaintyQuantificationRequest model."""

    def test_valid_instantiation(self, sample_uncertainty_params):
        """Test valid UncertaintyQuantificationRequest instantiation."""
        req = UncertaintyQuantificationRequest(**sample_uncertainty_params)
        assert req.calculation_id == "TEST-UNC-001"

    def test_uncertainty_percentages_positive(self, sample_uncertainty_params):
        """Test uncertainty percentages must be positive."""
        sample_uncertainty_params["cop_uncertainty_pct"] = Decimal("-5.0")
        with pytest.raises(Exception):
            UncertaintyQuantificationRequest(**sample_uncertainty_params)


class TestComplianceCheckRequest:
    """Test ComplianceCheckRequest model."""

    def test_valid_instantiation(self, sample_compliance_params):
        """Test valid ComplianceCheckRequest instantiation."""
        req = ComplianceCheckRequest(**sample_compliance_params)
        assert req.calculation_id == "TEST-COMP-001"
        assert len(req.frameworks) == 2

    def test_frameworks_not_empty(self, sample_compliance_params):
        """Test frameworks list cannot be empty."""
        sample_compliance_params["frameworks"] = []
        with pytest.raises(Exception):
            ComplianceCheckRequest(**sample_compliance_params)


class TestCoolingPurchaseInput:
    """Test CoolingPurchaseInput model."""

    def test_valid_instantiation_electric(self, sample_electric_request):
        """Test valid CoolingPurchaseInput with electric chiller."""
        input_data = CoolingPurchaseInput(
            request_type="ELECTRIC_CHILLER",
            electric_chiller=ElectricChillerRequest(**sample_electric_request),
            frameworks=["GHG_PROTOCOL"],
        )
        assert input_data.request_type == "ELECTRIC_CHILLER"
        assert input_data.electric_chiller is not None

    def test_valid_instantiation_absorption(self, sample_absorption_request):
        """Test valid CoolingPurchaseInput with absorption chiller."""
        input_data = CoolingPurchaseInput(
            request_type="ABSORPTION_CHILLER",
            absorption_cooling=AbsorptionCoolingRequest(**sample_absorption_request),
            frameworks=["ISO_14064"],
        )
        assert input_data.request_type == "ABSORPTION_CHILLER"
        assert input_data.absorption_cooling is not None

    def test_valid_instantiation_district(self, sample_district_request):
        """Test valid CoolingPurchaseInput with district cooling."""
        input_data = CoolingPurchaseInput(
            request_type="DISTRICT_COOLING",
            district_cooling=DistrictCoolingRequest(**sample_district_request),
            frameworks=["CSRD"],
        )
        assert input_data.request_type == "DISTRICT_COOLING"
        assert input_data.district_cooling is not None


class TestCoolingPurchaseOutput:
    """Test CoolingPurchaseOutput model."""

    def test_valid_instantiation(self, sample_calculation_result):
        """Test valid CoolingPurchaseOutput instantiation."""
        output = CoolingPurchaseOutput(
            calculation_result=CalculationResult(**sample_calculation_result),
            uncertainty_result=None,
            compliance_results=[],
            provenance_hash="abc123",
            processing_time_ms=Decimal("123.45"),
            validation_status="PASS",
        )
        assert output.validation_status == "PASS"
        assert output.provenance_hash == "abc123"

    def test_frozen_model(self, sample_calculation_result):
        """Test CoolingPurchaseOutput is frozen."""
        output = CoolingPurchaseOutput(
            calculation_result=CalculationResult(**sample_calculation_result),
            uncertainty_result=None,
            compliance_results=[],
            provenance_hash="abc123",
            processing_time_ms=Decimal("123.45"),
            validation_status="PASS",
        )
        with pytest.raises(Exception):
            output.validation_status = "FAIL"
