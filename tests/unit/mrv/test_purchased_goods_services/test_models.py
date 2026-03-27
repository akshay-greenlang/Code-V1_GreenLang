"""
Test suite for AGENT-MRV-014 models.

Tests all enums, constant tables, and Pydantic models for the Purchased Goods & Services Agent.
"""

import pytest
from decimal import Decimal
from datetime import date, datetime
from pydantic import ValidationError

from greenlang.agents.mrv.purchased_goods_services.models import (
    # Enums
    CalculationMethod,
    EEIODatabase,
    PhysicalEFSource,
    SupplierDataSource,
    AllocationMethod,
    MaterialCategory,
    CurrencyCode,
    DQIDimension,
    ComplianceFramework,
    ProcurementType,
    CoverageLevel,
    ExportFormat,
    UncertaintyDistribution,
    DataCollectionStatus,
    VerificationStatus,
    BoundaryScope,
    EmissionStage,
    IndustryClassification,
    RecycledContentMethod,
    BiogenicAccountingMethod,
    # Constant tables
    EEIO_EMISSION_FACTORS,
    PHYSICAL_EMISSION_FACTORS,
    CURRENCY_EXCHANGE_RATES,
    INDUSTRY_MARGIN_PERCENTAGES,
    # Models
    ProcurementItem,
    SpendRecord,
    PhysicalRecord,
    SupplierRecord,
    SpendBasedResult,
    AverageDataResult,
    SupplierSpecificResult,
    HybridResult,
    BatchRequest,
    DQIAssessment,
    UncertaintyAnalysis,
    ComplianceDisclosure,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestEnums:
    """Test all enum definitions."""

    def test_calculation_method_enum(self):
        """Test CalculationMethod enum has all expected values."""
        assert CalculationMethod.SPEND_BASED.value == "spend_based"
        assert CalculationMethod.AVERAGE_DATA.value == "average_data"
        assert CalculationMethod.SUPPLIER_SPECIFIC.value == "supplier_specific"
        assert CalculationMethod.HYBRID.value == "hybrid"
        assert len(CalculationMethod) == 4

    def test_eeio_database_enum(self):
        """Test EEIODatabase enum has all expected values."""
        assert EEIODatabase.EXIOBASE.value == "EXIOBASE"
        assert EEIODatabase.USEEIO.value == "USEEIO"
        assert EEIODatabase.WIOD.value == "WIOD"
        assert EEIODatabase.DEFRA.value == "DEFRA"
        assert EEIODatabase.EORA.value == "EORA"
        assert len(EEIODatabase) == 5

    def test_physical_ef_source_enum(self):
        """Test PhysicalEFSource enum has all expected values."""
        assert PhysicalEFSource.IPCC_2006.value == "IPCC_2006"
        assert PhysicalEFSource.IPCC_2019.value == "IPCC_2019"
        assert PhysicalEFSource.ECOINVENT.value == "ECOINVENT"
        assert PhysicalEFSource.GHG_PROTOCOL.value == "GHG_PROTOCOL"
        assert len(PhysicalEFSource) >= 4

    def test_supplier_data_source_enum(self):
        """Test SupplierDataSource enum has all expected values."""
        assert SupplierDataSource.EPD.value == "EPD"
        assert SupplierDataSource.SUPPLIER_SURVEY.value == "SUPPLIER_SURVEY"
        assert SupplierDataSource.THIRD_PARTY_AUDIT.value == "THIRD_PARTY_AUDIT"
        assert SupplierDataSource.CDP.value == "CDP"
        assert SupplierDataSource.LCA_STUDY.value == "LCA_STUDY"
        assert len(SupplierDataSource) >= 5

    def test_allocation_method_enum(self):
        """Test AllocationMethod enum has all expected values."""
        assert AllocationMethod.MASS.value == "mass"
        assert AllocationMethod.ECONOMIC.value == "economic"
        assert AllocationMethod.ENERGY.value == "energy"
        assert AllocationMethod.PHYSICAL_CAUSALITY.value == "physical_causality"
        assert len(AllocationMethod) >= 4

    def test_material_category_enum(self):
        """Test MaterialCategory enum has all expected values."""
        assert MaterialCategory.METALS_STEEL.value == "metals_steel"
        assert MaterialCategory.METALS_ALUMINUM.value == "metals_aluminum"
        assert MaterialCategory.CEMENT_LIME.value == "cement_lime"
        assert MaterialCategory.PLASTICS.value == "plastics"
        assert MaterialCategory.PAPER_CARDBOARD.value == "paper_cardboard"
        assert MaterialCategory.ELECTRONICS.value == "electronics"
        assert MaterialCategory.CHEMICALS_ORGANIC.value == "chemicals_organic"
        assert MaterialCategory.GLASS.value == "glass"
        assert MaterialCategory.TEXTILES.value == "textiles"
        assert MaterialCategory.BUSINESS_SERVICES.value == "business_services"
        assert MaterialCategory.LOGISTICS.value == "logistics"
        assert len(MaterialCategory) >= 11

    def test_currency_code_enum(self):
        """Test CurrencyCode enum has all expected values."""
        assert CurrencyCode.USD.value == "USD"
        assert CurrencyCode.EUR.value == "EUR"
        assert CurrencyCode.GBP.value == "GBP"
        assert CurrencyCode.JPY.value == "JPY"
        assert CurrencyCode.CNY.value == "CNY"
        assert len(CurrencyCode) >= 5

    def test_dqi_dimension_enum(self):
        """Test DQIDimension enum has all expected values."""
        assert DQIDimension.TECHNOLOGICAL.value == "technological_representativeness"
        assert DQIDimension.TEMPORAL.value == "temporal_representativeness"
        assert DQIDimension.GEOGRAPHICAL.value == "geographical_representativeness"
        assert DQIDimension.COMPLETENESS.value == "completeness"
        assert DQIDimension.RELIABILITY.value == "reliability"
        assert len(DQIDimension) == 5

    def test_compliance_framework_enum(self):
        """Test ComplianceFramework enum has all expected values."""
        assert ComplianceFramework.GHG_PROTOCOL.value == "GHG_PROTOCOL"
        assert ComplianceFramework.ISO_14064_1.value == "ISO_14064_1"
        assert ComplianceFramework.CSRD.value == "CSRD"
        assert ComplianceFramework.CDP.value == "CDP"
        assert ComplianceFramework.SBTI.value == "SBTI"
        assert len(ComplianceFramework) >= 5

    def test_procurement_type_enum(self):
        """Test ProcurementType enum has all expected values."""
        assert ProcurementType.GOODS.value == "goods"
        assert ProcurementType.SERVICES.value == "services"
        assert ProcurementType.CAPITAL_GOODS.value == "capital_goods"
        assert len(ProcurementType) == 3

    def test_coverage_level_enum(self):
        """Test CoverageLevel enum has all expected values."""
        assert CoverageLevel.LOW.value == "low"
        assert CoverageLevel.MEDIUM.value == "medium"
        assert CoverageLevel.HIGH.value == "high"
        assert len(CoverageLevel) == 3

    def test_export_format_enum(self):
        """Test ExportFormat enum has all expected values."""
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.EXCEL.value == "excel"
        assert ExportFormat.XML.value == "xml"
        assert len(ExportFormat) >= 4

    def test_uncertainty_distribution_enum(self):
        """Test UncertaintyDistribution enum has all expected values."""
        assert UncertaintyDistribution.NORMAL.value == "normal"
        assert UncertaintyDistribution.LOGNORMAL.value == "lognormal"
        assert UncertaintyDistribution.UNIFORM.value == "uniform"
        assert UncertaintyDistribution.TRIANGULAR.value == "triangular"
        assert len(UncertaintyDistribution) >= 4

    def test_data_collection_status_enum(self):
        """Test DataCollectionStatus enum has all expected values."""
        assert DataCollectionStatus.PENDING.value == "pending"
        assert DataCollectionStatus.IN_PROGRESS.value == "in_progress"
        assert DataCollectionStatus.COMPLETED.value == "completed"
        assert DataCollectionStatus.FAILED.value == "failed"
        assert len(DataCollectionStatus) >= 4

    def test_verification_status_enum(self):
        """Test VerificationStatus enum has all expected values."""
        assert VerificationStatus.UNVERIFIED.value == "unverified"
        assert VerificationStatus.SELF_DECLARED.value == "self_declared"
        assert VerificationStatus.THIRD_PARTY_VERIFIED.value == "third_party_verified"
        assert len(VerificationStatus) >= 3

    def test_boundary_scope_enum(self):
        """Test BoundaryScope enum has all expected values."""
        assert BoundaryScope.CRADLE_TO_GATE.value == "cradle_to_gate"
        assert BoundaryScope.CRADLE_TO_GRAVE.value == "cradle_to_grave"
        assert BoundaryScope.GATE_TO_GATE.value == "gate_to_gate"
        assert len(BoundaryScope) >= 3

    def test_emission_stage_enum(self):
        """Test EmissionStage enum has all expected values."""
        assert EmissionStage.RAW_MATERIAL.value == "raw_material"
        assert EmissionStage.PROCESSING.value == "processing"
        assert EmissionStage.TRANSPORT.value == "transport"
        assert EmissionStage.USE_PHASE.value == "use_phase"
        assert EmissionStage.END_OF_LIFE.value == "end_of_life"
        assert len(EmissionStage) >= 5

    def test_industry_classification_enum(self):
        """Test IndustryClassification enum has all expected values."""
        assert IndustryClassification.ISIC.value == "ISIC"
        assert IndustryClassification.NAICS.value == "NAICS"
        assert IndustryClassification.NACE.value == "NACE"
        assert len(IndustryClassification) >= 3

    def test_recycled_content_method_enum(self):
        """Test RecycledContentMethod enum has all expected values."""
        assert RecycledContentMethod.CUT_OFF.value == "cut_off"
        assert RecycledContentMethod.AVOIDED_BURDEN.value == "avoided_burden"
        assert RecycledContentMethod.SUBSTITUTION.value == "substitution"
        assert len(RecycledContentMethod) >= 3

    def test_biogenic_accounting_method_enum(self):
        """Test BiogenicAccountingMethod enum has all expected values."""
        assert BiogenicAccountingMethod.INCLUDE_ALL.value == "include_all"
        assert BiogenicAccountingMethod.EXCLUDE_ALL.value == "exclude_all"
        assert BiogenicAccountingMethod.NET_ZERO.value == "net_zero"
        assert len(BiogenicAccountingMethod) >= 3


# ============================================================================
# CONSTANT TABLE TESTS
# ============================================================================

class TestConstantTables:
    """Test constant data tables."""

    def test_eeio_emission_factors_size(self):
        """Test EEIO emission factors table has expected size."""
        assert len(EEIO_EMISSION_FACTORS) == 71
        assert all(isinstance(k, str) for k in EEIO_EMISSION_FACTORS.keys())
        assert all(isinstance(v, Decimal) for v in EEIO_EMISSION_FACTORS.values())

    def test_eeio_emission_factors_sample_values(self):
        """Test sample EEIO emission factor values."""
        # Should have steel sector
        assert any("steel" in k.lower() for k in EEIO_EMISSION_FACTORS.keys())
        # All factors should be positive
        assert all(v > 0 for v in EEIO_EMISSION_FACTORS.values())

    def test_physical_emission_factors_size(self):
        """Test physical emission factors table has expected size."""
        assert len(PHYSICAL_EMISSION_FACTORS) == 45
        assert all(isinstance(k, str) for k in PHYSICAL_EMISSION_FACTORS.keys())
        assert all(isinstance(v, Decimal) for v in PHYSICAL_EMISSION_FACTORS.values())

    def test_physical_emission_factors_sample_values(self):
        """Test sample physical emission factor values."""
        # Should have cement
        assert any("cement" in k.lower() for k in PHYSICAL_EMISSION_FACTORS.keys())
        # All factors should be positive
        assert all(v > 0 for v in PHYSICAL_EMISSION_FACTORS.values())

    def test_currency_exchange_rates_size(self):
        """Test currency exchange rates table has expected size."""
        assert len(CURRENCY_EXCHANGE_RATES) == 20
        assert all(isinstance(k, str) for k in CURRENCY_EXCHANGE_RATES.keys())
        assert all(isinstance(v, Decimal) for v in CURRENCY_EXCHANGE_RATES.values())

    def test_currency_exchange_rates_has_usd(self):
        """Test USD is the base currency with rate 1.0."""
        assert "USD" in CURRENCY_EXCHANGE_RATES
        assert CURRENCY_EXCHANGE_RATES["USD"] == Decimal("1.0")

    def test_industry_margin_percentages_size(self):
        """Test industry margin percentages table has expected size."""
        assert len(INDUSTRY_MARGIN_PERCENTAGES) == 24
        assert all(isinstance(k, str) for k in INDUSTRY_MARGIN_PERCENTAGES.keys())
        assert all(isinstance(v, Decimal) for v in INDUSTRY_MARGIN_PERCENTAGES.values())

    def test_industry_margin_percentages_range(self):
        """Test industry margin percentages are in valid range."""
        # All margins should be between 0% and 100%
        assert all(Decimal("0.0") <= v <= Decimal("100.0") for v in INDUSTRY_MARGIN_PERCENTAGES.values())


# ============================================================================
# MODEL TESTS
# ============================================================================

class TestProcurementItem:
    """Test ProcurementItem model."""

    def test_procurement_item_creation(self, sample_procurement_item):
        """Test creating a valid ProcurementItem."""
        assert sample_procurement_item.item_id == "ITEM-001"
        assert sample_procurement_item.item_name == "Hot-rolled steel beams"
        assert sample_procurement_item.procurement_type == ProcurementType.GOODS
        assert sample_procurement_item.category == MaterialCategory.METALS_STEEL
        assert sample_procurement_item.quantity == Decimal("15000.00")
        assert sample_procurement_item.total_spend == Decimal("37500.00")

    def test_procurement_item_period_validation(self):
        """Test period_end must be >= period_start."""
        with pytest.raises(ValidationError) as exc_info:
            ProcurementItem(
                item_id="ITEM-BAD",
                item_name="Test item",
                procurement_type=ProcurementType.GOODS,
                category=MaterialCategory.METALS_STEEL,
                supplier_id="SUP-001",
                supplier_name="TestCo",
                quantity=Decimal("1000.00"),
                unit="kg",
                unit_cost=Decimal("1.00"),
                total_spend=Decimal("1000.00"),
                currency=CurrencyCode.USD,
                period_start=date(2024, 12, 31),
                period_end=date(2024, 1, 1),  # Before start
                country_origin="USA",
                calculation_method=CalculationMethod.SPEND_BASED,
            )
        assert "period_end must be >= period_start" in str(exc_info.value)

    def test_procurement_item_immutability(self, sample_procurement_item):
        """Test ProcurementItem is frozen."""
        with pytest.raises(ValidationError):
            sample_procurement_item.quantity = Decimal("20000.00")


class TestSpendRecord:
    """Test SpendRecord model."""

    def test_spend_record_creation(self):
        """Test creating a valid SpendRecord."""
        record = SpendRecord(
            item_id="ITEM-001",
            spend_amount=Decimal("50000.00"),
            currency=CurrencyCode.USD,
            eeio_database=EEIODatabase.EXIOBASE,
            eeio_sector="Iron and steel",
            emission_factor=Decimal("0.833"),
            emission_factor_unit="kgCO2e/USD",
        )
        assert record.spend_amount == Decimal("50000.00")
        assert record.currency == CurrencyCode.USD
        assert record.eeio_database == EEIODatabase.EXIOBASE


class TestPhysicalRecord:
    """Test PhysicalRecord model."""

    def test_physical_record_creation(self):
        """Test creating a valid PhysicalRecord."""
        record = PhysicalRecord(
            item_id="ITEM-002",
            quantity=Decimal("25000.00"),
            unit="kg",
            ef_source=PhysicalEFSource.IPCC_2006,
            emission_factor=Decimal("0.583"),
            emission_factor_unit="kgCO2e/kg",
        )
        assert record.quantity == Decimal("25000.00")
        assert record.unit == "kg"
        assert record.ef_source == PhysicalEFSource.IPCC_2006


class TestSupplierRecord:
    """Test SupplierRecord model."""

    def test_supplier_record_creation(self, sample_supplier_record):
        """Test creating a valid SupplierRecord."""
        assert sample_supplier_record.supplier_id == "SUP-CONCRETE-789"
        assert sample_supplier_record.data_source == SupplierDataSource.EPD
        assert sample_supplier_record.emission_factor == Decimal("0.245")
        assert sample_supplier_record.epd_number == "EPD-CONC-2024-001"

    def test_supplier_record_coverage_validation(self):
        """Test coverage_percentage must be 0-100."""
        with pytest.raises(ValidationError):
            SupplierRecord(
                supplier_id="SUP-001",
                supplier_name="TestCo",
                data_source=SupplierDataSource.EPD,
                emission_factor=Decimal("0.5"),
                emission_factor_unit="kgCO2e/kg",
                validity_start=date(2024, 1, 1),
                validity_end=date(2025, 12, 31),
                coverage_percentage=Decimal("150.0"),  # Invalid
            )


class TestDQIAssessment:
    """Test DQIAssessment model."""

    def test_dqi_assessment_creation(self):
        """Test creating a valid DQIAssessment."""
        dqi = DQIAssessment(
            technological_representativeness=4,
            temporal_representativeness=5,
            geographical_representativeness=3,
            completeness=4,
            reliability=5,
            overall_score=Decimal("4.2"),
        )
        assert dqi.technological_representativeness == 4
        assert dqi.overall_score == Decimal("4.2")

    def test_dqi_assessment_score_range(self):
        """Test DQI scores must be 1-5."""
        with pytest.raises(ValidationError):
            DQIAssessment(
                technological_representativeness=6,  # Invalid
                temporal_representativeness=5,
                geographical_representativeness=3,
                completeness=4,
                reliability=5,
                overall_score=Decimal("4.0"),
            )


class TestSpendBasedResult:
    """Test SpendBasedResult model."""

    def test_spend_based_result_creation(self, sample_spend_based_result):
        """Test creating a valid SpendBasedResult."""
        assert sample_spend_based_result.item_id == "ITEM-001"
        assert sample_spend_based_result.method == CalculationMethod.SPEND_BASED
        assert sample_spend_based_result.total_emissions == Decimal("12500.00")
        assert sample_spend_based_result.eeio_database == EEIODatabase.EXIOBASE

    def test_spend_based_result_immutability(self, sample_spend_based_result):
        """Test SpendBasedResult is frozen."""
        with pytest.raises(ValidationError):
            sample_spend_based_result.total_emissions = Decimal("15000.00")


class TestAverageDataResult:
    """Test AverageDataResult model."""

    def test_average_data_result_creation(self, sample_average_data_result):
        """Test creating a valid AverageDataResult."""
        assert sample_average_data_result.item_id == "ITEM-002"
        assert sample_average_data_result.method == CalculationMethod.AVERAGE_DATA
        assert sample_average_data_result.ef_source == PhysicalEFSource.IPCC_2006
        assert sample_average_data_result.quantity == Decimal("15000.00")


class TestSupplierSpecificResult:
    """Test SupplierSpecificResult model."""

    def test_supplier_specific_result_creation(self, sample_supplier_specific_result):
        """Test creating a valid SupplierSpecificResult."""
        assert sample_supplier_specific_result.item_id == "ITEM-003"
        assert sample_supplier_specific_result.method == CalculationMethod.SUPPLIER_SPECIFIC
        assert sample_supplier_specific_result.supplier_id == "SUP-CONCRETE-789"
        assert sample_supplier_specific_result.data_source == SupplierDataSource.EPD
        assert sample_supplier_specific_result.epd_number == "EPD-CONC-2024-001"


class TestHybridResult:
    """Test HybridResult model."""

    def test_hybrid_result_creation(self, sample_hybrid_result):
        """Test creating a valid HybridResult."""
        assert sample_hybrid_result.method == CalculationMethod.HYBRID
        assert sample_hybrid_result.supplier_specific_coverage == Decimal("65.0")
        assert sample_hybrid_result.average_data_coverage == Decimal("35.0")
        assert sample_hybrid_result.coverage_level == CoverageLevel.MEDIUM

    def test_hybrid_result_coverage_sum(self):
        """Test supplier and average coverage should sum to ~100%."""
        result = HybridResult(
            item_id="ITEM-HYBRID-TEST",
            method=CalculationMethod.HYBRID,
            total_emissions=Decimal("10000.00"),
            emissions_unit="kgCO2e",
            supplier_specific_emissions=Decimal("6000.00"),
            supplier_specific_coverage=Decimal("60.0"),
            average_data_emissions=Decimal("4000.00"),
            average_data_coverage=Decimal("40.0"),
            coverage_level=CoverageLevel.MEDIUM,
            supplier_records_used=3,
            average_data_records_used=2,
            dqi_assessment=DQIAssessment(
                technological_representativeness=4,
                temporal_representativeness=4,
                geographical_representativeness=4,
                completeness=4,
                reliability=4,
                overall_score=Decimal("4.0"),
            ),
            calculation_timestamp=datetime.now(),
            provenance_hash="test123",
        )
        total_coverage = result.supplier_specific_coverage + result.average_data_coverage
        assert total_coverage == Decimal("100.0")


class TestBatchRequest:
    """Test BatchRequest model."""

    def test_batch_request_creation(self, sample_batch_request):
        """Test creating a valid BatchRequest."""
        assert sample_batch_request.request_id == "BATCH-2024-Q1"
        assert len(sample_batch_request.item_ids) == 5
        assert len(sample_batch_request.calculation_methods) == 3
        assert sample_batch_request.export_format == ExportFormat.JSON

    def test_batch_request_period_validation(self):
        """Test reporting_period_end must be >= reporting_period_start."""
        with pytest.raises(ValidationError):
            BatchRequest(
                request_id="BATCH-BAD",
                tenant_id="tenant-test",
                organization_id="org-test",
                reporting_period_start=date(2024, 12, 31),
                reporting_period_end=date(2024, 1, 1),  # Before start
                item_ids=["ITEM-001"],
                calculation_methods=[CalculationMethod.SPEND_BASED],
                compliance_frameworks=[ComplianceFramework.GHG_PROTOCOL],
                export_format=ExportFormat.JSON,
            )


class TestUncertaintyAnalysis:
    """Test UncertaintyAnalysis model."""

    def test_uncertainty_analysis_creation(self):
        """Test creating a valid UncertaintyAnalysis."""
        uncertainty = UncertaintyAnalysis(
            distribution=UncertaintyDistribution.LOGNORMAL,
            mean_value=Decimal("10000.00"),
            std_deviation=Decimal("1500.00"),
            confidence_interval_95_lower=Decimal("7500.00"),
            confidence_interval_95_upper=Decimal("12500.00"),
            monte_carlo_iterations=10000,
        )
        assert uncertainty.distribution == UncertaintyDistribution.LOGNORMAL
        assert uncertainty.monte_carlo_iterations == 10000


class TestComplianceDisclosure:
    """Test ComplianceDisclosure model."""

    def test_compliance_disclosure_creation(self, sample_disclosures):
        """Test creating valid ComplianceDisclosure."""
        ghg_disclosure = sample_disclosures[0]
        assert ghg_disclosure.framework == ComplianceFramework.GHG_PROTOCOL
        assert ghg_disclosure.total_emissions == Decimal("125000.00")
        assert ghg_disclosure.coverage_completeness == Decimal("92.0")

    def test_compliance_disclosure_percentage_validation(self):
        """Test coverage percentages must sum to 100%."""
        # This should be valid (sums to 100%)
        disclosure = ComplianceDisclosure(
            framework=ComplianceFramework.GHG_PROTOCOL,
            disclosure_id="TEST-001",
            total_emissions=Decimal("100000.00"),
            emissions_unit="tCO2e",
            spend_based_percentage=Decimal("30.0"),
            average_data_percentage=Decimal("40.0"),
            supplier_specific_percentage=Decimal("30.0"),
            data_quality_rating="High",
            coverage_completeness=Decimal("95.0"),
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
        )
        total = (
            disclosure.spend_based_percentage
            + disclosure.average_data_percentage
            + disclosure.supplier_specific_percentage
        )
        assert total == Decimal("100.0")
