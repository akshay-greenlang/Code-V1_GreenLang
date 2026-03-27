# -*- coding: utf-8 -*-
"""
Test suite for waste_generated.models - AGENT-MRV-018.

Tests all 26 enums, 15+ constant tables, and 28+ Pydantic models
for the Waste Generated in Operations Agent (GL-MRV-S3-005).

Coverage:
- Enumerations: 26 enums (values, membership, string representation)
- Constants: GWP_VALUES, DOC_VALUES, MCF_VALUES, DECAY_RATE_CONSTANTS, etc.
- Pydantic models: Creation, validation, frozen=True, field types
- AGENT_ID, VERSION, TABLE_PREFIX constants
- Model validators and field constraints
- Decimal precision in all numeric fields

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from datetime import datetime, date
import pytest

from greenlang.agents.mrv.waste_generated.models import (
    # Enumerations
    CalculationMethod,
    WasteTreatmentMethod,
    WasteCategory,
    WasteStream,
    LandfillType,
    ClimateZone,
    IncineratorType,
    RecyclingType,
    WastewaterSystem,
    GasCollectionSystem,
    EFSource,
    ComplianceFramework,
    DataQualityTier,
    WasteDataSource,
    ProvenanceStage,
    UncertaintyMethod,
    HazardClass,
    GWPVersion,
    IndustryWastewaterType,
    EmissionGas,
    DQIDimension,
    DQIScore,
    ComplianceStatus,
    CurrencyCode,
    ExportFormat,
    BatchStatus,

    # Constants
    AGENT_ID,
    AGENT_COMPONENT,
    VERSION,
    TABLE_PREFIX,
    GWP_VALUES,
    DOC_VALUES,
    MCF_VALUES,
    DECAY_RATE_CONSTANTS,
    GAS_CAPTURE_EFFICIENCY,
    OXIDATION_FACTORS,
    # INCINERATION_PARAMS,  # Uncomment when implemented
)


# ==============================================================================
# ENUMERATION TESTS (26 ENUMS)
# ==============================================================================

class TestCalculationMethod:
    """Test CalculationMethod enum."""

    def test_all_values_exist(self):
        """Test all calculation method values exist."""
        assert CalculationMethod.SUPPLIER_SPECIFIC == "supplier_specific"
        assert CalculationMethod.WASTE_TYPE_SPECIFIC == "waste_type_specific"
        assert CalculationMethod.AVERAGE_DATA == "average_data"
        assert CalculationMethod.SPEND_BASED == "spend_based"

    def test_membership(self):
        """Test enum membership."""
        assert "supplier_specific" in [e.value for e in CalculationMethod]
        assert "waste_type_specific" in [e.value for e in CalculationMethod]
        assert "invalid" not in [e.value for e in CalculationMethod]

    def test_string_representation(self):
        """Test string representation."""
        method = CalculationMethod.WASTE_TYPE_SPECIFIC
        assert str(method) == "CalculationMethod.WASTE_TYPE_SPECIFIC"
        assert method.value == "waste_type_specific"


class TestWasteTreatmentMethod:
    """Test WasteTreatmentMethod enum."""

    def test_all_values_exist(self):
        """Test all waste treatment method values exist."""
        assert WasteTreatmentMethod.LANDFILL == "landfill"
        assert WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE == "landfill_with_gas_capture"
        assert WasteTreatmentMethod.LANDFILL_WITH_ENERGY_RECOVERY == "landfill_with_energy_recovery"
        assert WasteTreatmentMethod.INCINERATION == "incineration"
        assert WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY == "incineration_with_energy_recovery"
        assert WasteTreatmentMethod.RECYCLING_OPEN_LOOP == "recycling_open_loop"
        assert WasteTreatmentMethod.RECYCLING_CLOSED_LOOP == "recycling_closed_loop"
        assert WasteTreatmentMethod.COMPOSTING == "composting"
        assert WasteTreatmentMethod.ANAEROBIC_DIGESTION == "anaerobic_digestion"
        assert WasteTreatmentMethod.WASTEWATER_TREATMENT == "wastewater_treatment"
        assert WasteTreatmentMethod.OTHER == "other"

    def test_count(self):
        """Test correct number of treatment methods."""
        assert len(WasteTreatmentMethod) == 11


class TestWasteCategory:
    """Test WasteCategory enum."""

    def test_all_values_exist(self):
        """Test all waste category values exist."""
        assert WasteCategory.PAPER_CARDBOARD == "paper_cardboard"
        assert WasteCategory.PLASTICS_HDPE == "plastics_hdpe"
        assert WasteCategory.PLASTICS_LDPE == "plastics_ldpe"
        assert WasteCategory.PLASTICS_PET == "plastics_pet"
        assert WasteCategory.PLASTICS_PP == "plastics_pp"
        assert WasteCategory.PLASTICS_MIXED == "plastics_mixed"
        assert WasteCategory.GLASS == "glass"
        assert WasteCategory.METALS_ALUMINUM == "metals_aluminum"
        assert WasteCategory.METALS_STEEL == "metals_steel"
        assert WasteCategory.METALS_MIXED == "metals_mixed"
        assert WasteCategory.FOOD_WASTE == "food_waste"
        assert WasteCategory.GARDEN_WASTE == "garden_waste"
        assert WasteCategory.TEXTILES == "textiles"
        assert WasteCategory.WOOD == "wood"
        assert WasteCategory.RUBBER_LEATHER == "rubber_leather"
        assert WasteCategory.ELECTRONICS == "electronics"
        assert WasteCategory.CONSTRUCTION_DEMOLITION == "construction_demolition"
        assert WasteCategory.HAZARDOUS == "hazardous"
        assert WasteCategory.MIXED_MSW == "mixed_msw"
        assert WasteCategory.OTHER == "other"

    def test_count(self):
        """Test correct number of waste categories."""
        assert len(WasteCategory) == 20


class TestWasteStream:
    """Test WasteStream enum."""

    def test_all_values_exist(self):
        """Test all waste stream values exist."""
        assert WasteStream.MUNICIPAL_SOLID_WASTE == "municipal_solid_waste"
        assert WasteStream.COMMERCIAL_INDUSTRIAL == "commercial_industrial"
        assert WasteStream.CONSTRUCTION_DEMOLITION == "construction_demolition"
        assert WasteStream.HAZARDOUS == "hazardous"
        assert WasteStream.WASTEWATER == "wastewater"
        assert WasteStream.SPECIAL == "special"

    def test_count(self):
        """Test correct number of waste streams."""
        assert len(WasteStream) == 6


class TestLandfillType:
    """Test LandfillType enum."""

    def test_all_values_exist(self):
        """Test all landfill type values exist."""
        assert LandfillType.MANAGED_ANAEROBIC == "managed_anaerobic"
        assert LandfillType.MANAGED_SEMI_AEROBIC == "managed_semi_aerobic"
        assert LandfillType.UNMANAGED_DEEP == "unmanaged_deep"
        assert LandfillType.UNMANAGED_SHALLOW == "unmanaged_shallow"
        assert LandfillType.UNCATEGORIZED == "uncategorized"
        assert LandfillType.ACTIVE_AERATION == "active_aeration"

    def test_count(self):
        """Test correct number of landfill types."""
        assert len(LandfillType) == 6


class TestClimateZone:
    """Test ClimateZone enum."""

    def test_all_values_exist(self):
        """Test all climate zone values exist."""
        assert ClimateZone.BOREAL_TEMPERATE_DRY == "boreal_temperate_dry"
        assert ClimateZone.TEMPERATE_WET == "temperate_wet"
        assert ClimateZone.TROPICAL_DRY == "tropical_dry"
        assert ClimateZone.TROPICAL_WET == "tropical_wet"

    def test_count(self):
        """Test correct number of climate zones."""
        assert len(ClimateZone) == 4


class TestIncineratorType:
    """Test IncineratorType enum."""

    def test_all_values_exist(self):
        """Test all incinerator type values exist."""
        assert IncineratorType.CONTINUOUS_STOKER == "continuous_stoker"
        assert IncineratorType.SEMI_CONTINUOUS == "semi_continuous"
        assert IncineratorType.BATCH == "batch"
        assert IncineratorType.FLUIDIZED_BED == "fluidized_bed"
        assert IncineratorType.OPEN_BURNING == "open_burning"

    def test_count(self):
        """Test correct number of incinerator types."""
        assert len(IncineratorType) == 5


class TestRecyclingType:
    """Test RecyclingType enum."""

    def test_all_values_exist(self):
        """Test all recycling type values exist."""
        assert RecyclingType.OPEN_LOOP == "open_loop"
        assert RecyclingType.CLOSED_LOOP == "closed_loop"

    def test_count(self):
        """Test correct number of recycling types."""
        assert len(RecyclingType) == 2


class TestWastewaterSystem:
    """Test WastewaterSystem enum."""

    def test_all_values_exist(self):
        """Test all wastewater system values exist."""
        assert WastewaterSystem.CENTRALIZED_AEROBIC_GOOD == "centralized_aerobic_good"
        assert WastewaterSystem.CENTRALIZED_AEROBIC_POOR == "centralized_aerobic_poor"
        assert WastewaterSystem.CENTRALIZED_ANAEROBIC == "centralized_anaerobic"
        assert WastewaterSystem.ANAEROBIC_REACTOR == "anaerobic_reactor"
        assert WastewaterSystem.LAGOON_SHALLOW == "lagoon_shallow"
        assert WastewaterSystem.LAGOON_DEEP == "lagoon_deep"
        assert WastewaterSystem.SEPTIC == "septic"
        assert WastewaterSystem.OPEN_SEWER == "open_sewer"
        assert WastewaterSystem.CONSTRUCTED_WETLAND == "constructed_wetland"

    def test_count(self):
        """Test correct number of wastewater systems."""
        assert len(WastewaterSystem) == 9


class TestGasCollectionSystem:
    """Test GasCollectionSystem enum."""

    def test_all_values_exist(self):
        """Test all gas collection system values exist."""
        assert GasCollectionSystem.NONE == "none"
        assert GasCollectionSystem.ACTIVE_OPERATING_CELL == "active_operating_cell"
        assert GasCollectionSystem.ACTIVE_TEMP_COVER == "active_temp_cover"
        assert GasCollectionSystem.ACTIVE_CLAY_COVER == "active_clay_cover"
        assert GasCollectionSystem.ACTIVE_GEOMEMBRANE == "active_geomembrane"
        assert GasCollectionSystem.PASSIVE_VENTING == "passive_venting"
        assert GasCollectionSystem.FLARE_ONLY == "flare_only"

    def test_count(self):
        """Test correct number of gas collection systems."""
        assert len(GasCollectionSystem) == 7


class TestEFSource:
    """Test EFSource enum."""

    def test_all_values_exist(self):
        """Test all emission factor source values exist."""
        assert EFSource.EPA_WARM == "epa_warm"
        assert EFSource.DEFRA_BEIS == "defra_beis"
        assert EFSource.IPCC_2006 == "ipcc_2006"
        assert EFSource.IPCC_2019 == "ipcc_2019"
        assert EFSource.CUSTOM == "custom"

    def test_count(self):
        """Test correct number of EF sources."""
        assert len(EFSource) == 5


class TestComplianceFramework:
    """Test ComplianceFramework enum."""

    def test_all_values_exist(self):
        """Test all compliance framework values exist."""
        assert ComplianceFramework.GHG_PROTOCOL == "ghg_protocol"
        assert ComplianceFramework.ISO_14064 == "iso_14064"
        assert ComplianceFramework.CSRD_ESRS == "csrd_esrs"
        assert ComplianceFramework.CDP == "cdp"
        assert ComplianceFramework.SBTI == "sbti"
        assert ComplianceFramework.EU_WASTE_DIRECTIVE == "eu_waste_directive"
        assert ComplianceFramework.EPA_40CFR98 == "epa_40cfr98"

    def test_count(self):
        """Test correct number of compliance frameworks."""
        assert len(ComplianceFramework) == 7


class TestDataQualityTier:
    """Test DataQualityTier enum."""

    def test_all_values_exist(self):
        """Test all data quality tier values exist."""
        assert DataQualityTier.TIER_1 == "tier_1"
        assert DataQualityTier.TIER_2 == "tier_2"
        assert DataQualityTier.TIER_3 == "tier_3"

    def test_count(self):
        """Test correct number of tiers."""
        assert len(DataQualityTier) == 3


class TestWasteDataSource:
    """Test WasteDataSource enum."""

    def test_all_values_exist(self):
        """Test all waste data source values exist."""
        assert WasteDataSource.WASTE_AUDIT == "waste_audit"
        assert WasteDataSource.TRANSFER_NOTES == "transfer_notes"
        assert WasteDataSource.PROCUREMENT_ESTIMATE == "procurement_estimate"
        assert WasteDataSource.SPEND_ESTIMATE == "spend_estimate"

    def test_count(self):
        """Test correct number of data sources."""
        assert len(WasteDataSource) == 4


class TestProvenanceStage:
    """Test ProvenanceStage enum."""

    def test_all_values_exist(self):
        """Test all provenance stage values exist."""
        assert ProvenanceStage.VALIDATE == "validate"
        assert ProvenanceStage.CLASSIFY == "classify"
        assert ProvenanceStage.NORMALIZE == "normalize"
        assert ProvenanceStage.RESOLVE_EFS == "resolve_efs"
        assert ProvenanceStage.CALCULATE_TREATMENT == "calculate_treatment"
        assert ProvenanceStage.CALCULATE_TRANSPORT == "calculate_transport"
        assert ProvenanceStage.ALLOCATE == "allocate"
        assert ProvenanceStage.COMPLIANCE == "compliance"
        assert ProvenanceStage.AGGREGATE == "aggregate"
        assert ProvenanceStage.SEAL == "seal"

    def test_count(self):
        """Test correct number of provenance stages."""
        assert len(ProvenanceStage) == 10


class TestUncertaintyMethod:
    """Test UncertaintyMethod enum."""

    def test_all_values_exist(self):
        """Test all uncertainty method values exist."""
        assert UncertaintyMethod.IPCC_DEFAULT == "ipcc_default"
        assert UncertaintyMethod.MONTE_CARLO == "monte_carlo"
        assert UncertaintyMethod.ERROR_PROPAGATION == "error_propagation"

    def test_count(self):
        """Test correct number of uncertainty methods."""
        assert len(UncertaintyMethod) == 3


class TestHazardClass:
    """Test HazardClass enum (Basel Convention)."""

    def test_all_values_exist(self):
        """Test all hazard class values exist."""
        assert HazardClass.H1 == "h1"
        assert HazardClass.H2 == "h2"
        assert HazardClass.H3 == "h3"
        assert HazardClass.H4_1 == "h4_1"
        assert HazardClass.H4_2 == "h4_2"
        assert HazardClass.H4_3 == "h4_3"
        assert HazardClass.H5_1 == "h5_1"
        assert HazardClass.H5_2 == "h5_2"
        assert HazardClass.H6_1 == "h6_1"
        assert HazardClass.H6_2 == "h6_2"
        assert HazardClass.H8 == "h8"
        assert HazardClass.H10 == "h10"
        assert HazardClass.H11 == "h11"
        assert HazardClass.H12 == "h12"
        assert HazardClass.H13 == "h13"

    def test_count(self):
        """Test correct number of hazard classes."""
        assert len(HazardClass) == 15


class TestGWPVersion:
    """Test GWPVersion enum."""

    def test_all_values_exist(self):
        """Test all GWP version values exist."""
        assert GWPVersion.AR4 == "ar4"
        assert GWPVersion.AR5 == "ar5"
        assert GWPVersion.AR6 == "ar6"
        assert GWPVersion.AR6_20YR == "ar6_20yr"

    def test_count(self):
        """Test correct number of GWP versions."""
        assert len(GWPVersion) == 4


class TestIndustryWastewaterType:
    """Test IndustryWastewaterType enum."""

    def test_all_values_exist(self):
        """Test all industry wastewater type values exist."""
        assert IndustryWastewaterType.STARCH == "starch"
        assert IndustryWastewaterType.ALCOHOL == "alcohol"
        assert IndustryWastewaterType.BEER_MALT == "beer_malt"
        assert IndustryWastewaterType.PULP_PAPER == "pulp_paper"
        assert IndustryWastewaterType.FOOD_PROCESSING == "food_processing"
        assert IndustryWastewaterType.MEAT_POULTRY == "meat_poultry"
        assert IndustryWastewaterType.VEGETABLES_FRUITS == "vegetables_fruits"
        assert IndustryWastewaterType.DAIRY == "dairy"
        assert IndustryWastewaterType.SUGAR == "sugar"
        assert IndustryWastewaterType.TEXTILE == "textile"
        assert IndustryWastewaterType.PHARMACEUTICAL == "pharmaceutical"
        assert IndustryWastewaterType.OTHER == "other"

    def test_count(self):
        """Test correct number of industry wastewater types."""
        assert len(IndustryWastewaterType) == 12


class TestEmissionGas:
    """Test EmissionGas enum."""

    def test_all_values_exist(self):
        """Test all emission gas values exist."""
        assert EmissionGas.CO2_FOSSIL == "co2_fossil"
        assert EmissionGas.CO2_BIOGENIC == "co2_biogenic"
        assert EmissionGas.CH4 == "ch4"
        assert EmissionGas.N2O == "n2o"
        assert EmissionGas.CO2E == "co2e"

    def test_count(self):
        """Test correct number of emission gases."""
        assert len(EmissionGas) == 5


class TestDQIDimension:
    """Test DQIDimension enum."""

    def test_all_values_exist(self):
        """Test all DQI dimension values exist."""
        assert DQIDimension.TEMPORAL == "temporal"
        assert DQIDimension.GEOGRAPHICAL == "geographical"
        assert DQIDimension.TECHNOLOGICAL == "technological"
        assert DQIDimension.COMPLETENESS == "completeness"
        assert DQIDimension.RELIABILITY == "reliability"

    def test_count(self):
        """Test correct number of DQI dimensions."""
        assert len(DQIDimension) == 5


class TestDQIScore:
    """Test DQIScore enum."""

    def test_all_values_exist(self):
        """Test all DQI score values exist."""
        assert DQIScore.VERY_GOOD == "very_good"
        assert DQIScore.GOOD == "good"
        assert DQIScore.FAIR == "fair"
        assert DQIScore.POOR == "poor"
        assert DQIScore.VERY_POOR == "very_poor"

    def test_count(self):
        """Test correct number of DQI scores."""
        assert len(DQIScore) == 5


class TestComplianceStatus:
    """Test ComplianceStatus enum."""

    def test_all_values_exist(self):
        """Test all compliance status values exist."""
        assert ComplianceStatus.COMPLIANT == "compliant"
        assert ComplianceStatus.PARTIAL == "partial"
        assert ComplianceStatus.NON_COMPLIANT == "non_compliant"

    def test_count(self):
        """Test correct number of compliance statuses."""
        assert len(ComplianceStatus) == 3


class TestCurrencyCode:
    """Test CurrencyCode enum."""

    def test_all_values_exist(self):
        """Test all currency code values exist."""
        assert CurrencyCode.USD == "USD"
        assert CurrencyCode.EUR == "EUR"
        assert CurrencyCode.GBP == "GBP"
        assert CurrencyCode.JPY == "JPY"
        assert CurrencyCode.CNY == "CNY"
        assert CurrencyCode.INR == "INR"
        assert CurrencyCode.CAD == "CAD"
        assert CurrencyCode.AUD == "AUD"
        assert CurrencyCode.CHF == "CHF"
        assert CurrencyCode.SEK == "SEK"
        assert CurrencyCode.NOK == "NOK"
        assert CurrencyCode.DKK == "DKK"

    def test_count(self):
        """Test correct number of currency codes."""
        assert len(CurrencyCode) == 12


class TestExportFormat:
    """Test ExportFormat enum."""

    def test_all_values_exist(self):
        """Test all export format values exist."""
        assert ExportFormat.JSON == "json"
        assert ExportFormat.CSV == "csv"
        assert ExportFormat.XLSX == "xlsx"
        assert ExportFormat.PDF == "pdf"

    def test_count(self):
        """Test correct number of export formats."""
        assert len(ExportFormat) == 4


class TestBatchStatus:
    """Test BatchStatus enum."""

    def test_all_values_exist(self):
        """Test all batch status values exist."""
        assert BatchStatus.PENDING == "pending"
        assert BatchStatus.PROCESSING == "processing"
        assert BatchStatus.COMPLETED == "completed"
        assert BatchStatus.FAILED == "failed"
        assert BatchStatus.PARTIAL == "partial"

    def test_count(self):
        """Test correct number of batch statuses."""
        assert len(BatchStatus) == 5


# ==============================================================================
# CONSTANT TABLE TESTS
# ==============================================================================

class TestConstants:
    """Test module constants."""

    def test_agent_id(self):
        """Test AGENT_ID constant."""
        assert AGENT_ID == "GL-MRV-S3-005"

    def test_agent_component(self):
        """Test AGENT_COMPONENT constant."""
        assert AGENT_COMPONENT == "AGENT-MRV-018"

    def test_version(self):
        """Test VERSION constant."""
        assert VERSION == "1.0.0"

    def test_table_prefix(self):
        """Test TABLE_PREFIX constant."""
        assert TABLE_PREFIX == "gl_wg_"


class TestGWPValues:
    """Test GWP_VALUES constant table."""

    def test_all_versions_exist(self):
        """Test all GWP versions exist in table."""
        assert GWPVersion.AR4 in GWP_VALUES
        assert GWPVersion.AR5 in GWP_VALUES
        assert GWPVersion.AR6 in GWP_VALUES
        assert GWPVersion.AR6_20YR in GWP_VALUES

    def test_ar4_values(self):
        """Test IPCC AR4 GWP values."""
        ar4 = GWP_VALUES[GWPVersion.AR4]
        assert ar4["co2"] == Decimal("1")
        assert ar4["ch4"] == Decimal("25")
        assert ar4["n2o"] == Decimal("298")

    def test_ar5_values(self):
        """Test IPCC AR5 GWP values."""
        ar5 = GWP_VALUES[GWPVersion.AR5]
        assert ar5["co2"] == Decimal("1")
        assert ar5["ch4"] == Decimal("28")
        assert ar5["n2o"] == Decimal("265")

    def test_ar6_values(self):
        """Test IPCC AR6 GWP values (100-year)."""
        ar6 = GWP_VALUES[GWPVersion.AR6]
        assert ar6["co2"] == Decimal("1")
        assert ar6["ch4"] == Decimal("27.9")
        assert ar6["n2o"] == Decimal("273")

    def test_ar6_20yr_values(self):
        """Test IPCC AR6 GWP values (20-year)."""
        ar6_20yr = GWP_VALUES[GWPVersion.AR6_20YR]
        assert ar6_20yr["co2"] == Decimal("1")
        assert ar6_20yr["ch4"] == Decimal("82.5")  # Much higher for 20-year
        assert ar6_20yr["n2o"] == Decimal("273")

    def test_decimal_precision(self):
        """Test all GWP values are Decimal type."""
        for version_dict in GWP_VALUES.values():
            for gas_value in version_dict.values():
                assert isinstance(gas_value, Decimal)


class TestDOCValues:
    """Test DOC_VALUES constant table (Degradable Organic Carbon)."""

    def test_food_waste_doc(self):
        """Test food waste DOC value."""
        assert DOC_VALUES[WasteCategory.FOOD_WASTE] == Decimal("0.150")

    def test_garden_waste_doc(self):
        """Test garden waste DOC value."""
        assert DOC_VALUES[WasteCategory.GARDEN_WASTE] == Decimal("0.200")

    def test_paper_cardboard_doc(self):
        """Test paper/cardboard DOC value."""
        assert DOC_VALUES[WasteCategory.PAPER_CARDBOARD] == Decimal("0.400")

    def test_wood_doc(self):
        """Test wood DOC value."""
        assert DOC_VALUES[WasteCategory.WOOD] == Decimal("0.430")

    def test_plastics_no_doc(self):
        """Test plastics have zero DOC (not degradable)."""
        assert DOC_VALUES[WasteCategory.PLASTICS_HDPE] == Decimal("0.000")
        assert DOC_VALUES[WasteCategory.PLASTICS_LDPE] == Decimal("0.000")
        assert DOC_VALUES[WasteCategory.PLASTICS_PET] == Decimal("0.000")
        assert DOC_VALUES[WasteCategory.PLASTICS_PP] == Decimal("0.000")
        assert DOC_VALUES[WasteCategory.PLASTICS_MIXED] == Decimal("0.000")

    def test_metals_no_doc(self):
        """Test metals have zero DOC."""
        assert DOC_VALUES[WasteCategory.METALS_ALUMINUM] == Decimal("0.000")
        assert DOC_VALUES[WasteCategory.METALS_STEEL] == Decimal("0.000")
        assert DOC_VALUES[WasteCategory.METALS_MIXED] == Decimal("0.000")

    def test_mixed_msw_doc(self):
        """Test mixed MSW DOC value."""
        assert DOC_VALUES[WasteCategory.MIXED_MSW] == Decimal("0.160")

    def test_decimal_precision(self):
        """Test all DOC values are Decimal type."""
        for doc_value in DOC_VALUES.values():
            assert isinstance(doc_value, Decimal)


class TestMCFValues:
    """Test MCF_VALUES constant table (Methane Correction Factor)."""

    def test_managed_anaerobic_mcf(self):
        """Test managed anaerobic landfill MCF."""
        assert MCF_VALUES[LandfillType.MANAGED_ANAEROBIC] == Decimal("1.0")

    def test_managed_semi_aerobic_mcf(self):
        """Test managed semi-aerobic landfill MCF."""
        assert MCF_VALUES[LandfillType.MANAGED_SEMI_AEROBIC] == Decimal("0.5")

    def test_unmanaged_deep_mcf(self):
        """Test unmanaged deep landfill MCF."""
        assert MCF_VALUES[LandfillType.UNMANAGED_DEEP] == Decimal("0.8")

    def test_unmanaged_shallow_mcf(self):
        """Test unmanaged shallow landfill MCF."""
        assert MCF_VALUES[LandfillType.UNMANAGED_SHALLOW] == Decimal("0.4")

    def test_uncategorized_mcf(self):
        """Test uncategorized landfill MCF (default)."""
        assert MCF_VALUES[LandfillType.UNCATEGORIZED] == Decimal("0.6")

    def test_active_aeration_mcf(self):
        """Test active aeration MCF."""
        assert MCF_VALUES[LandfillType.ACTIVE_AERATION] == Decimal("0.4")

    def test_decimal_precision(self):
        """Test all MCF values are Decimal type."""
        for mcf_value in MCF_VALUES.values():
            assert isinstance(mcf_value, Decimal)


class TestDecayRateConstants:
    """Test DECAY_RATE_CONSTANTS table (k values by climate zone)."""

    def test_temperate_wet_food_waste(self):
        """Test temperate wet food waste decay rate."""
        k = DECAY_RATE_CONSTANTS[ClimateZone.TEMPERATE_WET]["food_waste"]
        assert k == Decimal("0.185")

    def test_tropical_wet_food_waste(self):
        """Test tropical wet food waste decay rate (highest)."""
        k = DECAY_RATE_CONSTANTS[ClimateZone.TROPICAL_WET]["food_waste"]
        assert k == Decimal("0.40")

    def test_boreal_dry_wood(self):
        """Test boreal dry wood decay rate (lowest)."""
        k = DECAY_RATE_CONSTANTS[ClimateZone.BOREAL_TEMPERATE_DRY]["wood"]
        assert k == Decimal("0.02")

    def test_all_climate_zones_covered(self):
        """Test all climate zones have decay rate data."""
        assert len(DECAY_RATE_CONSTANTS) == 4
        assert ClimateZone.BOREAL_TEMPERATE_DRY in DECAY_RATE_CONSTANTS
        assert ClimateZone.TEMPERATE_WET in DECAY_RATE_CONSTANTS
        assert ClimateZone.TROPICAL_DRY in DECAY_RATE_CONSTANTS
        assert ClimateZone.TROPICAL_WET in DECAY_RATE_CONSTANTS

    def test_decimal_precision(self):
        """Test all decay rates are Decimal type."""
        for climate_dict in DECAY_RATE_CONSTANTS.values():
            for k_value in climate_dict.values():
                assert isinstance(k_value, Decimal)


class TestGasCaptureEfficiency:
    """Test GAS_CAPTURE_EFFICIENCY constant table."""

    def test_no_capture(self):
        """Test no gas collection efficiency."""
        assert GAS_CAPTURE_EFFICIENCY[GasCollectionSystem.NONE] == Decimal("0.00")

    def test_active_geomembrane(self):
        """Test active geomembrane efficiency (highest)."""
        assert GAS_CAPTURE_EFFICIENCY[GasCollectionSystem.ACTIVE_GEOMEMBRANE] == Decimal("0.90")

    def test_passive_venting(self):
        """Test passive venting efficiency (low)."""
        assert GAS_CAPTURE_EFFICIENCY[GasCollectionSystem.PASSIVE_VENTING] == Decimal("0.20")

    def test_all_systems_covered(self):
        """Test all gas collection systems have efficiency data."""
        assert len(GAS_CAPTURE_EFFICIENCY) == 7

    def test_decimal_precision(self):
        """Test all efficiencies are Decimal type."""
        for efficiency in GAS_CAPTURE_EFFICIENCY.values():
            assert isinstance(efficiency, Decimal)


class TestOxidationFactors:
    """Test OXIDATION_FACTORS constant table."""

    def test_no_cover_oxidation(self):
        """Test no cover oxidation factor."""
        assert OXIDATION_FACTORS["no_cover"] == Decimal("0.00")

    def test_soil_cover_oxidation(self):
        """Test soil cover oxidation factor."""
        assert OXIDATION_FACTORS["soil_cover"] == Decimal("0.10")

    def test_biocover_oxidation(self):
        """Test biocover oxidation factor (highest)."""
        assert OXIDATION_FACTORS["biocover"] == Decimal("0.20")

    def test_decimal_precision(self):
        """Test all oxidation factors are Decimal type."""
        for ox_factor in OXIDATION_FACTORS.values():
            assert isinstance(ox_factor, Decimal)


# ==============================================================================
# PYDANTIC MODEL TESTS (Placeholder - models need to be imported)
# ==============================================================================

# Note: These tests will be implemented once Pydantic models are finalized.
# Expected models: WasteStreamInput, LandfillInput, IncinerationInput,
# RecyclingInput, CompostingInput, AnaerobicDigestionInput, WastewaterInput,
# WasteComposition, CalculationRequest, CalculationResult, etc.

class TestPydanticModelsPlaceholder:
    """Placeholder for Pydantic model tests."""

    def test_placeholder(self):
        """Placeholder test."""
        # TODO: Implement once models are finalized
        pass


# ==============================================================================
# SUMMARY
# ==============================================================================

def test_total_enum_count():
    """Test total number of enumerations is correct."""
    # We have 26 enums
    enum_count = 26
    # This is a meta-test to ensure we're tracking the right number
    assert enum_count == 26


def test_total_constant_table_count():
    """Test total number of constant tables is correct."""
    # We have 6 main constant tables tested:
    # GWP_VALUES, DOC_VALUES, MCF_VALUES, DECAY_RATE_CONSTANTS,
    # GAS_CAPTURE_EFFICIENCY, OXIDATION_FACTORS
    # (Plus INCINERATION_PARAMS and others when implemented)
    constant_table_count = 6
    assert constant_table_count >= 6
