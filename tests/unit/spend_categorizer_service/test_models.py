# -*- coding: utf-8 -*-
"""
Unit tests for Spend Data Categorizer Models (AGENT-DATA-009)

Tests all 12 new enumerations, 15 SDK data models, 7 request models,
Layer 1 re-exports, the _utcnow helper, and __all__ export completeness.

Target: 150+ tests for comprehensive model coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from greenlang.spend_categorizer.models import (
    # New enumerations (12)
    TaxonomySystem,
    IngestionSource,
    RecordStatus,
    ClassificationConfidence,
    EmissionFactorSource,
    EmissionFactorUnit,
    CurrencyCode,
    AnalyticsTimeframe,
    ReportFormat,
    RuleMatchType,
    RulePriority,
    HotspotType,
    # SDK data models (15)
    SpendRecord,
    NormalizedSpendRecord,
    TaxonomyCode,
    TaxonomyClassification,
    Scope3Assignment,
    EmissionFactor,
    EmissionCalculationResult,
    CategoryRule,
    SpendAggregate,
    HotspotResult,
    TrendDataPoint,
    CategorizationReport,
    IngestionBatch,
    SpendCategorizerStatistics,
    VendorProfile,
    # Request models (7)
    IngestSpendRequest,
    ClassifyRequest,
    MapScope3Request,
    CalculateEmissionsRequest,
    CreateRuleRequest,
    AnalyticsRequest,
    GenerateReportRequest,
    # Layer 1 re-exports
    Scope3Category,
    DataSourceType,
    CalculationApproach,
    L1SpendRecord,
    L1PurchaseOrder,
    BOMItem,
    CategoryMappingResult,
    NAICS_TO_CATEGORY,
    SPEND_KEYWORDS_TO_CATEGORY,
    ERPSystem,
    ERPSpendCategory,
    TransactionType,
    SPEND_TO_SCOPE3_MAPPING,
    DEFAULT_EMISSION_FACTORS,
    CalculationMethod,
    ProcurementItem,
    EmissionCalculation,
)
from greenlang.spend_categorizer.models import _utcnow
from greenlang.spend_categorizer import models as models_module


# ============================================================================
# Enumeration tests (12 enums)
# ============================================================================


class TestTaxonomySystem:
    """Test TaxonomySystem enum members and values."""

    def test_member_count(self):
        assert len(TaxonomySystem) == 7

    def test_unspsc_value(self):
        assert TaxonomySystem.UNSPSC.value == "unspsc"

    def test_naics_value(self):
        assert TaxonomySystem.NAICS.value == "naics"

    def test_eclass_value(self):
        assert TaxonomySystem.ECLASS.value == "eclass"

    def test_isic_value(self):
        assert TaxonomySystem.ISIC.value == "isic"

    def test_sic_value(self):
        assert TaxonomySystem.SIC.value == "sic"

    def test_cpv_value(self):
        assert TaxonomySystem.CPV.value == "cpv"

    def test_hs_cn_value(self):
        assert TaxonomySystem.HS_CN.value == "hs_cn"

    def test_is_str_enum(self):
        assert isinstance(TaxonomySystem.UNSPSC, str)


class TestIngestionSource:
    """Test IngestionSource enum members and values."""

    def test_member_count(self):
        assert len(IngestionSource) == 6

    def test_erp_extract_value(self):
        assert IngestionSource.ERP_EXTRACT.value == "erp_extract"

    def test_csv_file_value(self):
        assert IngestionSource.CSV_FILE.value == "csv_file"

    def test_excel_file_value(self):
        assert IngestionSource.EXCEL_FILE.value == "excel_file"

    def test_api_feed_value(self):
        assert IngestionSource.API_FEED.value == "api_feed"

    def test_manual_entry_value(self):
        assert IngestionSource.MANUAL_ENTRY.value == "manual_entry"

    def test_procurement_platform_value(self):
        assert IngestionSource.PROCUREMENT_PLATFORM.value == "procurement_platform"


class TestRecordStatus:
    """Test RecordStatus enum members and values."""

    def test_member_count(self):
        assert len(RecordStatus) == 7

    def test_raw_value(self):
        assert RecordStatus.RAW.value == "raw"

    def test_normalized_value(self):
        assert RecordStatus.NORMALIZED.value == "normalized"

    def test_classified_value(self):
        assert RecordStatus.CLASSIFIED.value == "classified"

    def test_mapped_value(self):
        assert RecordStatus.MAPPED.value == "mapped"

    def test_calculated_value(self):
        assert RecordStatus.CALCULATED.value == "calculated"

    def test_validated_value(self):
        assert RecordStatus.VALIDATED.value == "validated"

    def test_exported_value(self):
        assert RecordStatus.EXPORTED.value == "exported"


class TestClassificationConfidence:
    """Test ClassificationConfidence enum members and values."""

    def test_member_count(self):
        assert len(ClassificationConfidence) == 4

    def test_high_value(self):
        assert ClassificationConfidence.HIGH.value == "high"

    def test_medium_value(self):
        assert ClassificationConfidence.MEDIUM.value == "medium"

    def test_low_value(self):
        assert ClassificationConfidence.LOW.value == "low"

    def test_unclassified_value(self):
        assert ClassificationConfidence.UNCLASSIFIED.value == "unclassified"


class TestEmissionFactorSource:
    """Test EmissionFactorSource enum members and values."""

    def test_member_count(self):
        assert len(EmissionFactorSource) == 6

    def test_epa_eeio_value(self):
        assert EmissionFactorSource.EPA_EEIO.value == "epa_eeio"

    def test_exiobase_value(self):
        assert EmissionFactorSource.EXIOBASE.value == "exiobase"

    def test_defra_value(self):
        assert EmissionFactorSource.DEFRA.value == "defra"

    def test_ecoinvent_value(self):
        assert EmissionFactorSource.ECOINVENT.value == "ecoinvent"

    def test_custom_value(self):
        assert EmissionFactorSource.CUSTOM.value == "custom"

    def test_supplier_specific_value(self):
        assert EmissionFactorSource.SUPPLIER_SPECIFIC.value == "supplier_specific"


class TestEmissionFactorUnit:
    """Test EmissionFactorUnit enum members and values."""

    def test_member_count(self):
        assert len(EmissionFactorUnit) == 5

    def test_per_usd_value(self):
        assert EmissionFactorUnit.KG_CO2E_PER_USD.value == "kgCO2e/USD"

    def test_per_eur_value(self):
        assert EmissionFactorUnit.KG_CO2E_PER_EUR.value == "kgCO2e/EUR"

    def test_per_kg_value(self):
        assert EmissionFactorUnit.KG_CO2E_PER_KG.value == "kgCO2e/kg"

    def test_per_unit_value(self):
        assert EmissionFactorUnit.KG_CO2E_PER_UNIT.value == "kgCO2e/unit"

    def test_per_kwh_value(self):
        assert EmissionFactorUnit.KG_CO2E_PER_KWH.value == "kgCO2e/kWh"


class TestCurrencyCode:
    """Test CurrencyCode enum members and values."""

    def test_member_count(self):
        assert len(CurrencyCode) == 12

    def test_usd_value(self):
        assert CurrencyCode.USD.value == "USD"

    def test_eur_value(self):
        assert CurrencyCode.EUR.value == "EUR"

    def test_gbp_value(self):
        assert CurrencyCode.GBP.value == "GBP"

    def test_jpy_value(self):
        assert CurrencyCode.JPY.value == "JPY"

    def test_cny_value(self):
        assert CurrencyCode.CNY.value == "CNY"

    def test_all_12_currencies(self):
        expected = {"USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "CHF", "INR", "BRL", "KRW", "MXN"}
        actual = {c.value for c in CurrencyCode}
        assert actual == expected


class TestAnalyticsTimeframe:
    """Test AnalyticsTimeframe enum members and values."""

    def test_member_count(self):
        assert len(AnalyticsTimeframe) == 4

    def test_monthly_value(self):
        assert AnalyticsTimeframe.MONTHLY.value == "monthly"

    def test_quarterly_value(self):
        assert AnalyticsTimeframe.QUARTERLY.value == "quarterly"

    def test_annual_value(self):
        assert AnalyticsTimeframe.ANNUAL.value == "annual"

    def test_custom_value(self):
        assert AnalyticsTimeframe.CUSTOM.value == "custom"


class TestReportFormat:
    """Test ReportFormat enum members and values."""

    def test_member_count(self):
        assert len(ReportFormat) == 4

    def test_json_value(self):
        assert ReportFormat.JSON.value == "json"

    def test_csv_value(self):
        assert ReportFormat.CSV.value == "csv"

    def test_markdown_value(self):
        assert ReportFormat.MARKDOWN.value == "markdown"

    def test_html_value(self):
        assert ReportFormat.HTML.value == "html"


class TestRuleMatchType:
    """Test RuleMatchType enum members and values."""

    def test_member_count(self):
        assert len(RuleMatchType) == 6

    def test_exact_value(self):
        assert RuleMatchType.EXACT.value == "exact"

    def test_contains_value(self):
        assert RuleMatchType.CONTAINS.value == "contains"

    def test_regex_value(self):
        assert RuleMatchType.REGEX.value == "regex"

    def test_fuzzy_value(self):
        assert RuleMatchType.FUZZY.value == "fuzzy"

    def test_starts_with_value(self):
        assert RuleMatchType.STARTS_WITH.value == "starts_with"

    def test_ends_with_value(self):
        assert RuleMatchType.ENDS_WITH.value == "ends_with"


class TestRulePriority:
    """Test RulePriority enum members and values."""

    def test_member_count(self):
        assert len(RulePriority) == 5

    def test_critical_value(self):
        assert RulePriority.CRITICAL.value == "critical"

    def test_high_value(self):
        assert RulePriority.HIGH.value == "high"

    def test_medium_value(self):
        assert RulePriority.MEDIUM.value == "medium"

    def test_low_value(self):
        assert RulePriority.LOW.value == "low"

    def test_default_value(self):
        assert RulePriority.DEFAULT.value == "default"


class TestHotspotType:
    """Test HotspotType enum members and values."""

    def test_member_count(self):
        assert len(HotspotType) == 4

    def test_top_spend_value(self):
        assert HotspotType.TOP_SPEND.value == "top_spend"

    def test_top_emissions_value(self):
        assert HotspotType.TOP_EMISSIONS.value == "top_emissions"

    def test_top_intensity_value(self):
        assert HotspotType.TOP_INTENSITY.value == "top_intensity"

    def test_rising_trend_value(self):
        assert HotspotType.RISING_TREND.value == "rising_trend"


# ============================================================================
# _utcnow helper tests
# ============================================================================


class TestUtcnowHelper:
    """Test the _utcnow helper function."""

    def test_returns_datetime(self):
        result = _utcnow()
        assert isinstance(result, datetime)

    def test_has_utc_timezone(self):
        result = _utcnow()
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc

    def test_microseconds_zeroed(self):
        result = _utcnow()
        assert result.microsecond == 0


# ============================================================================
# SDK Data Models tests (15 models)
# ============================================================================


class TestSpendRecord:
    """Test SpendRecord model."""

    def _make_valid(self, **overrides):
        defaults = {
            "vendor_id": "V-001",
            "vendor_name": "Test Vendor",
            "transaction_date": date(2025, 6, 1),
            "amount": 1000.0,
        }
        defaults.update(overrides)
        return SpendRecord(**defaults)

    def test_valid_creation(self):
        rec = self._make_valid()
        assert rec.vendor_id == "V-001"
        assert rec.amount == 1000.0

    def test_record_id_is_uuid(self):
        rec = self._make_valid()
        uuid.UUID(rec.record_id)  # Should not raise

    def test_default_currency_is_usd(self):
        rec = self._make_valid()
        assert rec.currency == CurrencyCode.USD

    def test_default_status_is_raw(self):
        rec = self._make_valid()
        assert rec.status == RecordStatus.RAW

    def test_default_ingestion_source(self):
        rec = self._make_valid()
        assert rec.ingestion_source == IngestionSource.MANUAL_ENTRY

    def test_default_tenant_id(self):
        rec = self._make_valid()
        assert rec.tenant_id == "default"

    def test_created_at_is_utc(self):
        rec = self._make_valid()
        assert rec.created_at.tzinfo == timezone.utc

    def test_empty_vendor_id_raises(self):
        with pytest.raises(ValidationError, match="vendor_id"):
            self._make_valid(vendor_id="")

    def test_whitespace_vendor_id_raises(self):
        with pytest.raises(ValidationError, match="vendor_id"):
            self._make_valid(vendor_id="   ")

    def test_empty_vendor_name_raises(self):
        with pytest.raises(ValidationError, match="vendor_name"):
            self._make_valid(vendor_name="")

    def test_negative_amount_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(amount=-100)

    def test_zero_amount_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(amount=0)

    def test_model_dump_round_trip(self):
        rec = self._make_valid()
        data = rec.model_dump()
        rec2 = SpendRecord(**data)
        assert rec2.vendor_id == rec.vendor_id
        assert rec2.amount == rec.amount

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError, match="extra"):
            self._make_valid(unknown_field="not allowed")


class TestNormalizedSpendRecord:
    """Test NormalizedSpendRecord model."""

    def _make_valid(self, **overrides):
        defaults = {
            "record_id": "rec-001",
            "original_record_id": "orig-001",
            "normalized_vendor_name": "Test Vendor Normalized",
            "amount_usd": 1000.0,
        }
        defaults.update(overrides)
        return NormalizedSpendRecord(**defaults)

    def test_valid_creation(self):
        rec = self._make_valid()
        assert rec.record_id == "rec-001"

    def test_default_status_is_normalized(self):
        rec = self._make_valid()
        assert rec.status == RecordStatus.NORMALIZED

    def test_default_exchange_rate(self):
        rec = self._make_valid()
        assert rec.exchange_rate == 1.0

    def test_default_is_duplicate_false(self):
        rec = self._make_valid()
        assert rec.is_duplicate is False

    def test_empty_record_id_raises(self):
        with pytest.raises(ValidationError, match="record_id"):
            self._make_valid(record_id="")

    def test_empty_original_record_id_raises(self):
        with pytest.raises(ValidationError, match="original_record_id"):
            self._make_valid(original_record_id="")

    def test_empty_normalized_vendor_name_raises(self):
        with pytest.raises(ValidationError, match="normalized_vendor_name"):
            self._make_valid(normalized_vendor_name="")

    def test_dedup_similarity_bounds(self):
        rec = self._make_valid(dedup_similarity=0.95)
        assert rec.dedup_similarity == 0.95

    def test_dedup_similarity_above_1_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(dedup_similarity=1.5)

    def test_negative_amount_usd_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(amount_usd=-1.0)


class TestTaxonomyCode:
    """Test TaxonomyCode model."""

    def _make_valid(self, **overrides):
        defaults = {
            "code": "43211500",
            "system": TaxonomySystem.UNSPSC,
            "level": 4,
        }
        defaults.update(overrides)
        return TaxonomyCode(**defaults)

    def test_valid_creation(self):
        tc = self._make_valid()
        assert tc.code == "43211500"

    def test_default_description_empty(self):
        tc = self._make_valid()
        assert tc.description == ""

    def test_default_parent_code_none(self):
        tc = self._make_valid()
        assert tc.parent_code is None

    def test_empty_code_raises(self):
        with pytest.raises(ValidationError, match="code"):
            self._make_valid(code="")

    def test_level_below_1_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(level=0)

    def test_level_above_8_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(level=9)

    def test_level_boundary_1(self):
        tc = self._make_valid(level=1)
        assert tc.level == 1

    def test_level_boundary_8(self):
        tc = self._make_valid(level=8)
        assert tc.level == 8


class TestTaxonomyClassification:
    """Test TaxonomyClassification model."""

    def _make_valid(self, **overrides):
        primary = TaxonomyCode(code="43211500", system=TaxonomySystem.UNSPSC, level=4)
        defaults = {
            "record_id": "rec-001",
            "primary_code": primary,
            "confidence": 0.85,
            "confidence_level": ClassificationConfidence.HIGH,
        }
        defaults.update(overrides)
        return TaxonomyClassification(**defaults)

    def test_valid_creation(self):
        tc = self._make_valid()
        assert tc.record_id == "rec-001"

    def test_classification_id_is_uuid(self):
        tc = self._make_valid()
        uuid.UUID(tc.classification_id)

    def test_default_method_rule_based(self):
        tc = self._make_valid()
        assert tc.classification_method == "rule_based"

    def test_default_secondary_codes_empty(self):
        tc = self._make_valid()
        assert tc.secondary_codes == []

    def test_empty_record_id_raises(self):
        with pytest.raises(ValidationError, match="record_id"):
            self._make_valid(record_id="")

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(confidence=1.5)

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(confidence=-0.1)


class TestScope3Assignment:
    """Test Scope3Assignment model."""

    def _make_valid(self, **overrides):
        defaults = {
            "record_id": "rec-001",
            "scope3_category": Scope3Category.CAT1_PURCHASED_GOODS,
            "category_number": 1,
            "category_name": "Purchased Goods and Services",
            "confidence": 0.9,
            "confidence_level": ClassificationConfidence.HIGH,
        }
        defaults.update(overrides)
        return Scope3Assignment(**defaults)

    def test_valid_creation(self):
        sa = self._make_valid()
        assert sa.category_number == 1

    def test_assignment_id_is_uuid(self):
        sa = self._make_valid()
        uuid.UUID(sa.assignment_id)

    def test_default_mapping_rule(self):
        sa = self._make_valid()
        assert sa.mapping_rule == "default"

    def test_default_recommended_approach(self):
        sa = self._make_valid()
        assert sa.recommended_approach == CalculationApproach.SPEND_BASED

    def test_empty_record_id_raises(self):
        with pytest.raises(ValidationError, match="record_id"):
            self._make_valid(record_id="")

    def test_empty_category_name_raises(self):
        with pytest.raises(ValidationError, match="category_name"):
            self._make_valid(category_name="")

    def test_category_number_below_1_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(category_number=0)

    def test_category_number_above_15_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(category_number=16)


class TestEmissionFactor:
    """Test EmissionFactor model."""

    def _make_valid(self, **overrides):
        defaults = {
            "taxonomy_code": "424120",
            "source": EmissionFactorSource.EPA_EEIO,
            "value": 0.30,
        }
        defaults.update(overrides)
        return EmissionFactor(**defaults)

    def test_valid_creation(self):
        ef = self._make_valid()
        assert ef.value == 0.30

    def test_factor_id_is_uuid(self):
        ef = self._make_valid()
        uuid.UUID(ef.factor_id)

    def test_default_taxonomy_system(self):
        ef = self._make_valid()
        assert ef.taxonomy_system == TaxonomySystem.NAICS

    def test_default_unit(self):
        ef = self._make_valid()
        assert ef.unit == EmissionFactorUnit.KG_CO2E_PER_USD

    def test_default_region(self):
        ef = self._make_valid()
        assert ef.region == "global"

    def test_default_year(self):
        ef = self._make_valid()
        assert ef.year == 2024

    def test_default_data_quality_score(self):
        ef = self._make_valid()
        assert ef.data_quality_score == 0.5

    def test_empty_taxonomy_code_raises(self):
        with pytest.raises(ValidationError, match="taxonomy_code"):
            self._make_valid(taxonomy_code="")

    def test_negative_value_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(value=-0.1)

    def test_year_below_1990_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(year=1989)

    def test_year_above_2100_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(year=2101)


class TestEmissionCalculationResult:
    """Test EmissionCalculationResult model."""

    def _make_valid(self, **overrides):
        defaults = {
            "record_id": "rec-001",
            "vendor_id": "V-001",
            "amount_usd": 10000.0,
            "emission_factor_value": 0.30,
            "emissions_kgco2e": 3000.0,
            "emissions_tco2e": 3.0,
        }
        defaults.update(overrides)
        return EmissionCalculationResult(**defaults)

    def test_valid_creation(self):
        ecr = self._make_valid()
        assert ecr.emissions_kgco2e == 3000.0

    def test_result_id_is_uuid(self):
        ecr = self._make_valid()
        uuid.UUID(ecr.result_id)

    def test_default_emission_factor_source(self):
        ecr = self._make_valid()
        assert ecr.emission_factor_source == EmissionFactorSource.EPA_EEIO

    def test_default_calculation_method(self):
        ecr = self._make_valid()
        assert ecr.calculation_method == CalculationMethod.SPEND_BASED

    def test_empty_record_id_raises(self):
        with pytest.raises(ValidationError, match="record_id"):
            self._make_valid(record_id="")

    def test_empty_vendor_id_raises(self):
        with pytest.raises(ValidationError, match="vendor_id"):
            self._make_valid(vendor_id="")

    def test_negative_emissions_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(emissions_kgco2e=-1.0)


class TestCategoryRule:
    """Test CategoryRule model."""

    def _make_valid(self, **overrides):
        defaults = {
            "name": "Test Rule",
            "match_type": RuleMatchType.CONTAINS,
            "pattern": "office",
        }
        defaults.update(overrides)
        return CategoryRule(**defaults)

    def test_valid_creation(self):
        rule = self._make_valid()
        assert rule.name == "Test Rule"

    def test_rule_id_is_uuid(self):
        rule = self._make_valid()
        uuid.UUID(rule.rule_id)

    def test_default_priority(self):
        rule = self._make_valid()
        assert rule.priority == RulePriority.MEDIUM

    def test_default_match_field(self):
        rule = self._make_valid()
        assert rule.match_field == "description"

    def test_default_is_active(self):
        rule = self._make_valid()
        assert rule.is_active is True

    def test_default_confidence_boost(self):
        rule = self._make_valid()
        assert rule.confidence_boost == 0.0

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError, match="name"):
            self._make_valid(name="")

    def test_empty_pattern_raises(self):
        with pytest.raises(ValidationError, match="pattern"):
            self._make_valid(pattern="")

    def test_confidence_boost_above_1_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(confidence_boost=1.5)

    def test_confidence_boost_below_neg1_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(confidence_boost=-1.5)


class TestSpendAggregate:
    """Test SpendAggregate model."""

    def test_valid_creation(self):
        sa = SpendAggregate(category="cat_1")
        assert sa.category == "cat_1"

    def test_defaults(self):
        sa = SpendAggregate(category="cat_1")
        assert sa.total_spend_usd == 0.0
        assert sa.total_emissions_kgco2e == 0.0
        assert sa.record_count == 0
        assert sa.vendor_count == 0
        assert sa.percentage_of_total_spend == 0.0

    def test_empty_category_raises(self):
        with pytest.raises(ValidationError, match="category"):
            SpendAggregate(category="")

    def test_percentage_above_100_raises(self):
        with pytest.raises(ValidationError):
            SpendAggregate(category="c1", percentage_of_total_spend=101.0)


class TestHotspotResult:
    """Test HotspotResult model."""

    def _make_valid(self, **overrides):
        defaults = {
            "hotspot_type": HotspotType.TOP_SPEND,
            "category": "chemicals",
            "rank": 1,
        }
        defaults.update(overrides)
        return HotspotResult(**defaults)

    def test_valid_creation(self):
        hr = self._make_valid()
        assert hr.rank == 1

    def test_hotspot_id_is_uuid(self):
        hr = self._make_valid()
        uuid.UUID(hr.hotspot_id)

    def test_empty_category_raises(self):
        with pytest.raises(ValidationError, match="category"):
            self._make_valid(category="")

    def test_rank_below_1_raises(self):
        with pytest.raises(ValidationError):
            self._make_valid(rank=0)


class TestTrendDataPoint:
    """Test TrendDataPoint model."""

    def _make_valid(self, **overrides):
        defaults = {
            "period": "2024-Q1",
            "period_start": date(2024, 1, 1),
            "period_end": date(2024, 3, 31),
        }
        defaults.update(overrides)
        return TrendDataPoint(**defaults)

    def test_valid_creation(self):
        tdp = self._make_valid()
        assert tdp.period == "2024-Q1"

    def test_default_direction(self):
        tdp = self._make_valid()
        assert tdp.direction == "flat"

    def test_empty_period_raises(self):
        with pytest.raises(ValidationError, match="period"):
            self._make_valid(period="")

    def test_invalid_direction_raises(self):
        with pytest.raises(ValidationError, match="direction"):
            self._make_valid(direction="sideways")

    def test_valid_direction_up(self):
        tdp = self._make_valid(direction="up")
        assert tdp.direction == "up"

    def test_valid_direction_down(self):
        tdp = self._make_valid(direction="down")
        assert tdp.direction == "down"


class TestCategorizationReport:
    """Test CategorizationReport model."""

    def _make_valid(self, **overrides):
        defaults = {
            "title": "Q1 2025 Spend Report",
            "period_start": date(2025, 1, 1),
            "period_end": date(2025, 3, 31),
        }
        defaults.update(overrides)
        return CategorizationReport(**defaults)

    def test_valid_creation(self):
        rpt = self._make_valid()
        assert rpt.title == "Q1 2025 Spend Report"

    def test_report_id_is_uuid(self):
        rpt = self._make_valid()
        uuid.UUID(rpt.report_id)

    def test_default_format(self):
        rpt = self._make_valid()
        assert rpt.report_format == ReportFormat.JSON

    def test_default_generated_by(self):
        rpt = self._make_valid()
        assert rpt.generated_by == "system"

    def test_default_lists_empty(self):
        rpt = self._make_valid()
        assert rpt.aggregates == []
        assert rpt.hotspots == []
        assert rpt.trends == []

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError, match="title"):
            self._make_valid(title="")


class TestIngestionBatch:
    """Test IngestionBatch model."""

    def test_valid_creation(self):
        batch = IngestionBatch(source=IngestionSource.CSV_FILE)
        assert batch.source == IngestionSource.CSV_FILE

    def test_batch_id_is_uuid(self):
        batch = IngestionBatch(source=IngestionSource.CSV_FILE)
        uuid.UUID(batch.batch_id)

    def test_default_status(self):
        batch = IngestionBatch(source=IngestionSource.CSV_FILE)
        assert batch.status == RecordStatus.RAW

    def test_default_errors_empty(self):
        batch = IngestionBatch(source=IngestionSource.CSV_FILE)
        assert batch.errors == []

    def test_default_records_zero(self):
        batch = IngestionBatch(source=IngestionSource.CSV_FILE)
        assert batch.record_count == 0
        assert batch.records_accepted == 0
        assert batch.records_rejected == 0


class TestSpendCategorizerStatistics:
    """Test SpendCategorizerStatistics model."""

    def test_all_defaults_zero(self):
        stats = SpendCategorizerStatistics()
        assert stats.total_records_ingested == 0
        assert stats.total_records_classified == 0
        assert stats.total_records_mapped == 0
        assert stats.total_records_calculated == 0
        assert stats.total_spend_usd == 0.0
        assert stats.total_emissions_kgco2e == 0.0
        assert stats.avg_classification_confidence == 0.0
        assert stats.classification_coverage_pct == 0.0
        assert stats.scope3_coverage_pct == 0.0
        assert stats.emissions_coverage_pct == 0.0
        assert stats.total_vendors == 0
        assert stats.total_rules == 0
        assert stats.total_batches == 0
        assert stats.avg_batch_duration_seconds == 0.0

    def test_coverage_pct_above_100_raises(self):
        with pytest.raises(ValidationError):
            SpendCategorizerStatistics(classification_coverage_pct=101.0)

    def test_negative_ingested_raises(self):
        with pytest.raises(ValidationError):
            SpendCategorizerStatistics(total_records_ingested=-1)


class TestVendorProfile:
    """Test VendorProfile model."""

    def _make_valid(self, **overrides):
        defaults = {
            "vendor_id": "V-001",
            "normalized_name": "Test Vendor Corp",
        }
        defaults.update(overrides)
        return VendorProfile(**defaults)

    def test_valid_creation(self):
        vp = self._make_valid()
        assert vp.vendor_id == "V-001"

    def test_default_aliases_empty(self):
        vp = self._make_valid()
        assert vp.aliases == []

    def test_default_is_strategic(self):
        vp = self._make_valid()
        assert vp.is_strategic is False

    def test_default_totals_zero(self):
        vp = self._make_valid()
        assert vp.total_spend_usd == 0.0
        assert vp.total_emissions_kgco2e == 0.0
        assert vp.record_count == 0

    def test_empty_vendor_id_raises(self):
        with pytest.raises(ValidationError, match="vendor_id"):
            self._make_valid(vendor_id="")

    def test_empty_normalized_name_raises(self):
        with pytest.raises(ValidationError, match="normalized_name"):
            self._make_valid(normalized_name="")

    def test_model_dump_round_trip(self):
        vp = self._make_valid(aliases=["Test Corp", "TV Corp"])
        data = vp.model_dump()
        vp2 = VendorProfile(**data)
        assert vp2.aliases == vp.aliases


# ============================================================================
# Request Model tests (7 request models)
# ============================================================================


class TestIngestSpendRequest:
    """Test IngestSpendRequest model."""

    def _make_record(self):
        return SpendRecord(
            vendor_id="V-001",
            vendor_name="Test",
            transaction_date=date(2025, 1, 1),
            amount=100.0,
        )

    def test_valid_creation(self):
        req = IngestSpendRequest(
            source=IngestionSource.CSV_FILE,
            records=[self._make_record()],
        )
        assert req.source == IngestionSource.CSV_FILE

    def test_empty_records_raises(self):
        with pytest.raises(ValidationError):
            IngestSpendRequest(source=IngestionSource.CSV_FILE, records=[])

    def test_default_enable_deduplication(self):
        req = IngestSpendRequest(
            source=IngestionSource.CSV_FILE,
            records=[self._make_record()],
        )
        assert req.enable_deduplication is True

    def test_default_enable_normalization(self):
        req = IngestSpendRequest(
            source=IngestionSource.CSV_FILE,
            records=[self._make_record()],
        )
        assert req.enable_normalization is True


class TestClassifyRequest:
    """Test ClassifyRequest model."""

    def test_valid_creation(self):
        req = ClassifyRequest()
        assert req.taxonomy_system == TaxonomySystem.UNSPSC

    def test_default_min_confidence(self):
        req = ClassifyRequest()
        assert req.min_confidence == 0.3

    def test_default_use_rules(self):
        req = ClassifyRequest()
        assert req.use_rules is True

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValidationError):
            ClassifyRequest(min_confidence=1.5)


class TestMapScope3Request:
    """Test MapScope3Request model."""

    def test_valid_creation(self):
        req = MapScope3Request()
        assert req.mapping_strategy == "rule_based"

    def test_default_use_naics_lookup(self):
        req = MapScope3Request()
        assert req.use_naics_lookup is True

    def test_default_use_keyword_matching(self):
        req = MapScope3Request()
        assert req.use_keyword_matching is True


class TestCalculateEmissionsRequest:
    """Test CalculateEmissionsRequest model."""

    def test_valid_creation(self):
        req = CalculateEmissionsRequest()
        assert req.calculation_method == CalculationMethod.SPEND_BASED

    def test_default_factor_sources(self):
        req = CalculateEmissionsRequest()
        assert len(req.emission_factor_sources) == 4
        assert req.emission_factor_sources[0] == EmissionFactorSource.SUPPLIER_SPECIFIC

    def test_default_include_data_quality(self):
        req = CalculateEmissionsRequest()
        assert req.include_data_quality is True


class TestCreateRuleRequest:
    """Test CreateRuleRequest model."""

    def test_valid_creation(self):
        req = CreateRuleRequest(
            name="Test Rule",
            match_type=RuleMatchType.CONTAINS,
            pattern="office",
        )
        assert req.name == "Test Rule"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError, match="name"):
            CreateRuleRequest(
                name="",
                match_type=RuleMatchType.CONTAINS,
                pattern="office",
            )

    def test_empty_pattern_raises(self):
        with pytest.raises(ValidationError, match="pattern"):
            CreateRuleRequest(
                name="Rule",
                match_type=RuleMatchType.CONTAINS,
                pattern="",
            )

    def test_default_priority(self):
        req = CreateRuleRequest(
            name="Rule",
            match_type=RuleMatchType.EXACT,
            pattern="test",
        )
        assert req.priority == RulePriority.MEDIUM


class TestAnalyticsRequest:
    """Test AnalyticsRequest model."""

    def test_valid_creation(self):
        req = AnalyticsRequest(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )
        assert req.timeframe == AnalyticsTimeframe.QUARTERLY

    def test_default_top_n(self):
        req = AnalyticsRequest(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )
        assert req.top_n == 10

    def test_top_n_above_100_raises(self):
        with pytest.raises(ValidationError):
            AnalyticsRequest(
                start_date=date(2025, 1, 1),
                end_date=date(2025, 12, 31),
                top_n=101,
            )

    def test_default_hotspot_types(self):
        req = AnalyticsRequest(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )
        assert len(req.hotspot_types) == 3


class TestGenerateReportRequest:
    """Test GenerateReportRequest model."""

    def test_valid_creation(self):
        req = GenerateReportRequest(
            title="Q1 Report",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 31),
        )
        assert req.title == "Q1 Report"

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError, match="title"):
            GenerateReportRequest(
                title="",
                start_date=date(2025, 1, 1),
                end_date=date(2025, 3, 31),
            )

    def test_default_format(self):
        req = GenerateReportRequest(
            title="Report",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 31),
        )
        assert req.report_format == ReportFormat.JSON


# ============================================================================
# Layer 1 re-export tests
# ============================================================================


class TestLayer1ReExports:
    """Verify Layer 1 models and enums are importable from models module."""

    def test_scope3_category_importable(self):
        assert Scope3Category is not None

    def test_data_source_type_importable(self):
        assert DataSourceType is not None

    def test_calculation_approach_importable(self):
        assert CalculationApproach is not None

    def test_l1_spend_record_importable(self):
        assert L1SpendRecord is not None

    def test_l1_purchase_order_importable(self):
        assert L1PurchaseOrder is not None

    def test_bom_item_importable(self):
        assert BOMItem is not None

    def test_category_mapping_result_importable(self):
        assert CategoryMappingResult is not None

    def test_naics_to_category_is_dict(self):
        assert isinstance(NAICS_TO_CATEGORY, dict)

    def test_spend_keywords_to_category_is_dict(self):
        assert isinstance(SPEND_KEYWORDS_TO_CATEGORY, dict)

    def test_erp_system_importable(self):
        assert ERPSystem is not None

    def test_erp_spend_category_importable(self):
        assert ERPSpendCategory is not None

    def test_transaction_type_importable(self):
        assert TransactionType is not None

    def test_spend_to_scope3_mapping_is_dict(self):
        assert isinstance(SPEND_TO_SCOPE3_MAPPING, dict)

    def test_default_emission_factors_is_dict(self):
        assert isinstance(DEFAULT_EMISSION_FACTORS, dict)

    def test_calculation_method_importable(self):
        assert CalculationMethod is not None

    def test_procurement_item_importable(self):
        assert ProcurementItem is not None

    def test_emission_calculation_importable(self):
        assert EmissionCalculation is not None


# ============================================================================
# __all__ export completeness tests
# ============================================================================


class TestModuleExports:
    """Verify __all__ exports are complete and accessible."""

    def test_all_is_list(self):
        assert isinstance(models_module.__all__, list)

    def test_all_count(self):
        # 3 L1 enums + 4 L1 models + 2 L1 constants
        # + 3 ERP enums + 2 ERP constants + 3 procurement models
        # + 12 new enums + 15 SDK models + 7 request models = 51 total
        assert len(models_module.__all__) == 51, (
            f"Expected 51 exports, got {len(models_module.__all__)}"
        )

    def test_all_entries_are_accessible(self):
        for name in models_module.__all__:
            assert hasattr(models_module, name), (
                f"__all__ entry '{name}' is not accessible on models module"
            )

    def test_new_enums_in_all(self):
        new_enums = [
            "TaxonomySystem", "IngestionSource", "RecordStatus",
            "ClassificationConfidence", "EmissionFactorSource",
            "EmissionFactorUnit", "CurrencyCode", "AnalyticsTimeframe",
            "ReportFormat", "RuleMatchType", "RulePriority", "HotspotType",
        ]
        for name in new_enums:
            assert name in models_module.__all__, f"{name} missing from __all__"

    def test_sdk_models_in_all(self):
        sdk_models = [
            "SpendRecord", "NormalizedSpendRecord", "TaxonomyCode",
            "TaxonomyClassification", "Scope3Assignment", "EmissionFactor",
            "EmissionCalculationResult", "CategoryRule", "SpendAggregate",
            "HotspotResult", "TrendDataPoint", "CategorizationReport",
            "IngestionBatch", "SpendCategorizerStatistics", "VendorProfile",
        ]
        for name in sdk_models:
            assert name in models_module.__all__, f"{name} missing from __all__"

    def test_request_models_in_all(self):
        request_models = [
            "IngestSpendRequest", "ClassifyRequest", "MapScope3Request",
            "CalculateEmissionsRequest", "CreateRuleRequest",
            "AnalyticsRequest", "GenerateReportRequest",
        ]
        for name in request_models:
            assert name in models_module.__all__, f"{name} missing from __all__"
