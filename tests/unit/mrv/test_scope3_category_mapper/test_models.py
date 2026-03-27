# -*- coding: utf-8 -*-
"""
Test suite for scope3_category_mapper.models - AGENT-MRV-029.

Tests all enumerations, constants, metadata, and Pydantic models for the
Scope 3 Category Mapper Agent (GL-MRV-X-040).

Coverage:
- Enumerations: 8 enums (values, membership, count, string representation)
- Constants: AGENT_ID, AGENT_COMPONENT, VERSION, TABLE_PREFIX,
  SCOPE3_CATEGORY_NAMES, SCOPE3_CATEGORY_NUMBERS, ALL_SCOPE3_CATEGORIES,
  quantization constants
- Pydantic models: CategoryCompletenessEntry, CompletenessReport,
  ComplianceFinding, DetailedComplianceAssessment, BenchmarkComparison
- Model validation: frozen immutability, field constraints, ranges
- Edge cases: boundary values, invalid inputs, optional fields

Total: ~80 tests

Author: GL-TestEngineer
Date: March 2026
"""

import re
from decimal import Decimal
from datetime import datetime

import pytest
from pydantic import ValidationError as PydanticValidationError

from greenlang.agents.mrv.scope3_category_mapper.models import (
    # Enumerations
    Scope3Category,
    CompanyType,
    CategoryRelevance,
    ComplianceFramework,
    ComplianceStatus,
    ComplianceSeverity,
    ScreeningResult,
    DataQualityTier,
    # Constants
    AGENT_ID,
    AGENT_COMPONENT,
    VERSION,
    TABLE_PREFIX,
    SCOPE3_CATEGORY_NAMES,
    SCOPE3_CATEGORY_NUMBERS,
    ALL_SCOPE3_CATEGORIES,
    _QUANT_2DP,
    _QUANT_4DP,
    _QUANT_8DP,
    # Pydantic models
    CategoryCompletenessEntry,
    CompletenessReport,
    ComplianceFinding,
    DetailedComplianceAssessment,
    BenchmarkComparison,
)


# ==============================================================================
# AGENT METADATA CONSTANTS
# ==============================================================================


class TestAgentMetadata:
    """Test agent metadata constants defined in models.py."""

    def test_agent_id_is_gl_mrv_x_040(self):
        """AGENT_ID must be GL-MRV-X-040."""
        assert AGENT_ID == "GL-MRV-X-040"

    def test_agent_component_is_agent_mrv_029(self):
        """AGENT_COMPONENT must be AGENT-MRV-029."""
        assert AGENT_COMPONENT == "AGENT-MRV-029"

    def test_version_is_semver(self):
        """VERSION must follow SemVer (major.minor.patch)."""
        assert re.match(r"^\d+\.\d+\.\d+$", VERSION) is not None

    def test_version_is_1_0_0(self):
        """VERSION must be 1.0.0 for initial release."""
        assert VERSION == "1.0.0"

    def test_table_prefix_is_gl_scm(self):
        """TABLE_PREFIX must be gl_scm_ for all database tables."""
        assert TABLE_PREFIX == "gl_scm_"

    def test_table_prefix_ends_with_underscore(self):
        """TABLE_PREFIX must end with underscore separator."""
        assert TABLE_PREFIX.endswith("_")

    def test_quant_2dp(self):
        """_QUANT_2DP must be Decimal('0.01') for 2-decimal precision."""
        assert _QUANT_2DP == Decimal("0.01")

    def test_quant_4dp(self):
        """_QUANT_4DP must be Decimal('0.0001') for 4-decimal precision."""
        assert _QUANT_4DP == Decimal("0.0001")

    def test_quant_8dp(self):
        """_QUANT_8DP must be Decimal('0.00000001') for 8-decimal precision."""
        assert _QUANT_8DP == Decimal("0.00000001")


# ==============================================================================
# SCOPE3CATEGORY ENUM
# ==============================================================================


class TestScope3CategoryEnum:
    """Test Scope3Category enumeration -- 15 GHG Protocol Scope 3 categories."""

    def test_scope3_category_has_15_members(self):
        """Scope3Category must have exactly 15 members (one per category)."""
        assert len(Scope3Category) == 15

    def test_scope3_category_values_contain_number(self):
        """Every Scope3Category value must contain a number prefix."""
        for member in Scope3Category:
            # Values like "1_purchased_goods_services", "2_capital_goods", etc.
            assert member.value[0].isdigit(), (
                f"{member.name} has value '{member.value}' without number prefix"
            )

    @pytest.mark.parametrize("member,expected_value", [
        (Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES, "1_purchased_goods_services"),
        (Scope3Category.CAT_2_CAPITAL_GOODS, "2_capital_goods"),
        (Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES, "3_fuel_energy_activities"),
        (Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION, "4_upstream_transportation"),
        (Scope3Category.CAT_5_WASTE_GENERATED, "5_waste_generated"),
        (Scope3Category.CAT_6_BUSINESS_TRAVEL, "6_business_travel"),
        (Scope3Category.CAT_7_EMPLOYEE_COMMUTING, "7_employee_commuting"),
        (Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS, "8_upstream_leased_assets"),
        (Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION, "9_downstream_transportation"),
        (Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS, "10_processing_sold_products"),
        (Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS, "11_use_of_sold_products"),
        (Scope3Category.CAT_12_END_OF_LIFE_TREATMENT, "12_end_of_life_treatment"),
        (Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS, "13_downstream_leased_assets"),
        (Scope3Category.CAT_14_FRANCHISES, "14_franchises"),
        (Scope3Category.CAT_15_INVESTMENTS, "15_investments"),
    ])
    def test_scope3_category_specific_value(self, member, expected_value):
        """Each Scope3Category member must have the correct string value."""
        assert member.value == expected_value

    def test_scope3_category_is_str_enum(self):
        """Scope3Category must be a string enum (str, Enum)."""
        assert issubclass(Scope3Category, str)

    def test_scope3_category_string_comparison(self):
        """Scope3Category values must be comparable as strings."""
        assert Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES == "1_purchased_goods_services"

    def test_scope3_category_from_value(self):
        """Scope3Category must be constructable from string value."""
        cat = Scope3Category("6_business_travel")
        assert cat == Scope3Category.CAT_6_BUSINESS_TRAVEL

    def test_scope3_category_invalid_value_raises(self):
        """Invalid value must raise ValueError."""
        with pytest.raises(ValueError):
            Scope3Category("cat_16_nonexistent")


# ==============================================================================
# COMPANY TYPE ENUM
# ==============================================================================


class TestCompanyTypeEnum:
    """Test CompanyType enumeration -- 8 company types."""

    def test_company_type_has_8_members(self):
        """CompanyType must have exactly 8 members."""
        assert len(CompanyType) == 8

    @pytest.mark.parametrize("member,expected_value", [
        (CompanyType.MANUFACTURER, "manufacturer"),
        (CompanyType.SERVICES, "services"),
        (CompanyType.FINANCIAL, "financial"),
        (CompanyType.RETAILER, "retailer"),
        (CompanyType.ENERGY, "energy"),
        (CompanyType.MINING, "mining"),
        (CompanyType.AGRICULTURE, "agriculture"),
        (CompanyType.TRANSPORT, "transport"),
    ])
    def test_company_type_values(self, member, expected_value):
        """Each CompanyType member must have the correct value."""
        assert member.value == expected_value

    def test_company_type_is_str_enum(self):
        """CompanyType must be a string enum."""
        assert issubclass(CompanyType, str)

    def test_company_type_from_value(self):
        """CompanyType must be constructable from string value."""
        ct = CompanyType("financial")
        assert ct == CompanyType.FINANCIAL


# ==============================================================================
# CATEGORY RELEVANCE ENUM
# ==============================================================================


class TestCategoryRelevanceEnum:
    """Test CategoryRelevance enumeration -- 4 relevance levels."""

    def test_category_relevance_has_4_members(self):
        """CategoryRelevance must have exactly 4 members."""
        assert len(CategoryRelevance) == 4

    @pytest.mark.parametrize("member,expected_value", [
        (CategoryRelevance.MATERIAL, "material"),
        (CategoryRelevance.RELEVANT, "relevant"),
        (CategoryRelevance.NOT_RELEVANT, "not_relevant"),
        (CategoryRelevance.UNKNOWN, "unknown"),
    ])
    def test_category_relevance_values(self, member, expected_value):
        """Each CategoryRelevance member must have the correct value."""
        assert member.value == expected_value


# ==============================================================================
# COMPLIANCE FRAMEWORK ENUM
# ==============================================================================


class TestComplianceFrameworkEnum:
    """Test ComplianceFramework enumeration -- 8 regulatory frameworks."""

    def test_compliance_framework_has_8_members(self):
        """ComplianceFramework must have exactly 8 members."""
        assert len(ComplianceFramework) == 8

    @pytest.mark.parametrize("member,expected_value", [
        (ComplianceFramework.GHG_PROTOCOL, "ghg_protocol"),
        (ComplianceFramework.ISO_14064, "iso_14064"),
        (ComplianceFramework.CSRD_ESRS, "csrd_esrs"),
        (ComplianceFramework.CDP, "cdp"),
        (ComplianceFramework.SBTI, "sbti"),
        (ComplianceFramework.SB_253, "sb_253"),
        (ComplianceFramework.SEC_CLIMATE, "sec_climate"),
        (ComplianceFramework.EU_TAXONOMY, "eu_taxonomy"),
    ])
    def test_compliance_framework_values(self, member, expected_value):
        """Each ComplianceFramework member must have the correct value."""
        assert member.value == expected_value

    def test_compliance_framework_is_str_enum(self):
        """ComplianceFramework must be a string enum."""
        assert issubclass(ComplianceFramework, str)


# ==============================================================================
# COMPLIANCE STATUS ENUM
# ==============================================================================


class TestComplianceStatusEnum:
    """Test ComplianceStatus enumeration -- 4 statuses."""

    def test_compliance_status_has_4_members(self):
        """ComplianceStatus must have exactly 4 members."""
        assert len(ComplianceStatus) == 4

    @pytest.mark.parametrize("member,expected_value", [
        (ComplianceStatus.PASS, "PASS"),
        (ComplianceStatus.FAIL, "FAIL"),
        (ComplianceStatus.WARNING, "WARNING"),
        (ComplianceStatus.NOT_APPLICABLE, "NOT_APPLICABLE"),
    ])
    def test_compliance_status_values(self, member, expected_value):
        """Each ComplianceStatus member must have the correct value."""
        assert member.value == expected_value

    def test_compliance_status_uppercase(self):
        """All ComplianceStatus values must be UPPERCASE."""
        for member in ComplianceStatus:
            assert member.value == member.value.upper()


# ==============================================================================
# COMPLIANCE SEVERITY ENUM
# ==============================================================================


class TestComplianceSeverityEnum:
    """Test ComplianceSeverity enumeration -- 5 severity levels."""

    def test_compliance_severity_has_5_members(self):
        """ComplianceSeverity must have exactly 5 members."""
        assert len(ComplianceSeverity) == 5

    @pytest.mark.parametrize("member,expected_value", [
        (ComplianceSeverity.CRITICAL, "CRITICAL"),
        (ComplianceSeverity.HIGH, "HIGH"),
        (ComplianceSeverity.MEDIUM, "MEDIUM"),
        (ComplianceSeverity.LOW, "LOW"),
        (ComplianceSeverity.INFO, "INFO"),
    ])
    def test_compliance_severity_values(self, member, expected_value):
        """Each ComplianceSeverity member must have the correct value."""
        assert member.value == expected_value

    def test_compliance_severity_uppercase(self):
        """All ComplianceSeverity values must be UPPERCASE."""
        for member in ComplianceSeverity:
            assert member.value == member.value.upper()


# ==============================================================================
# SCREENING RESULT ENUM
# ==============================================================================


class TestScreeningResultEnum:
    """Test ScreeningResult enumeration -- 3 screening outcomes."""

    def test_screening_result_has_3_members(self):
        """ScreeningResult must have exactly 3 members."""
        assert len(ScreeningResult) == 3

    @pytest.mark.parametrize("member,expected_value", [
        (ScreeningResult.COMPLETE, "complete"),
        (ScreeningResult.PARTIAL, "partial"),
        (ScreeningResult.MISSING, "missing"),
    ])
    def test_screening_result_values(self, member, expected_value):
        """Each ScreeningResult member must have the correct value."""
        assert member.value == expected_value


# ==============================================================================
# DATA QUALITY TIER ENUM
# ==============================================================================


class TestDataQualityTierEnum:
    """Test DataQualityTier enumeration -- 5 quality tiers."""

    def test_data_quality_tier_has_5_members(self):
        """DataQualityTier must have exactly 5 members."""
        assert len(DataQualityTier) == 5

    @pytest.mark.parametrize("member,expected_value", [
        (DataQualityTier.TIER_1, "tier_1"),
        (DataQualityTier.TIER_2, "tier_2"),
        (DataQualityTier.TIER_3, "tier_3"),
        (DataQualityTier.TIER_4, "tier_4"),
        (DataQualityTier.TIER_5, "tier_5"),
    ])
    def test_data_quality_tier_values(self, member, expected_value):
        """Each DataQualityTier member must have the correct value."""
        assert member.value == expected_value

    def test_data_quality_tier_from_value(self):
        """DataQualityTier must be constructable from string value."""
        tier = DataQualityTier("tier_1")
        assert tier == DataQualityTier.TIER_1

    def test_data_quality_tier_invalid_value_raises(self):
        """Invalid tier value must raise ValueError."""
        with pytest.raises(ValueError):
            DataQualityTier("tier_6")


# ==============================================================================
# SCOPE3 CATEGORY NAME AND NUMBER LOOKUP TABLES
# ==============================================================================


class TestScope3CategoryLookupTables:
    """Test SCOPE3_CATEGORY_NAMES, SCOPE3_CATEGORY_NUMBERS, ALL_SCOPE3_CATEGORIES."""

    def test_scope3_category_names_has_15_entries(self):
        """SCOPE3_CATEGORY_NAMES must have entries for all 15 categories."""
        assert len(SCOPE3_CATEGORY_NAMES) == 15

    def test_scope3_category_names_all_keys_are_enum(self):
        """All keys in SCOPE3_CATEGORY_NAMES must be Scope3Category members."""
        for key in SCOPE3_CATEGORY_NAMES:
            assert isinstance(key, Scope3Category)

    def test_scope3_category_names_all_values_are_str(self):
        """All values in SCOPE3_CATEGORY_NAMES must be non-empty strings."""
        for name in SCOPE3_CATEGORY_NAMES.values():
            assert isinstance(name, str)
            assert len(name) > 0

    @pytest.mark.parametrize("category,expected_name", [
        (Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES, "Purchased Goods and Services"),
        (Scope3Category.CAT_6_BUSINESS_TRAVEL, "Business Travel"),
        (Scope3Category.CAT_15_INVESTMENTS, "Investments"),
    ])
    def test_scope3_category_names_specific(self, category, expected_name):
        """Specific category names must match GHG Protocol terminology."""
        assert SCOPE3_CATEGORY_NAMES[category] == expected_name

    def test_scope3_category_numbers_has_15_entries(self):
        """SCOPE3_CATEGORY_NUMBERS must have entries for all 15 categories."""
        assert len(SCOPE3_CATEGORY_NUMBERS) == 15

    def test_scope3_category_numbers_range(self):
        """Category numbers must be 1 through 15."""
        numbers = sorted(SCOPE3_CATEGORY_NUMBERS.values())
        assert numbers == list(range(1, 16))

    @pytest.mark.parametrize("category,expected_number", [
        (Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES, 1),
        (Scope3Category.CAT_5_WASTE_GENERATED, 5),
        (Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS, 10),
        (Scope3Category.CAT_15_INVESTMENTS, 15),
    ])
    def test_scope3_category_numbers_specific(self, category, expected_number):
        """Specific category numbers must match GHG Protocol numbering."""
        assert SCOPE3_CATEGORY_NUMBERS[category] == expected_number

    def test_all_scope3_categories_has_15_entries(self):
        """ALL_SCOPE3_CATEGORIES must contain all 15 categories."""
        assert len(ALL_SCOPE3_CATEGORIES) == 15

    def test_all_scope3_categories_is_list(self):
        """ALL_SCOPE3_CATEGORIES must be a list."""
        assert isinstance(ALL_SCOPE3_CATEGORIES, list)

    def test_all_scope3_categories_matches_enum(self):
        """ALL_SCOPE3_CATEGORIES must match list(Scope3Category)."""
        assert ALL_SCOPE3_CATEGORIES == list(Scope3Category)

    def test_names_and_numbers_keys_match(self):
        """SCOPE3_CATEGORY_NAMES and SCOPE3_CATEGORY_NUMBERS must have same keys."""
        assert set(SCOPE3_CATEGORY_NAMES.keys()) == set(SCOPE3_CATEGORY_NUMBERS.keys())


# ==============================================================================
# CategoryCompletenessEntry MODEL
# ==============================================================================


class TestCategoryCompletenessEntryModel:
    """Test CategoryCompletenessEntry Pydantic model."""

    def _make_entry(self, **overrides) -> CategoryCompletenessEntry:
        """Helper to create a valid CategoryCompletenessEntry."""
        defaults = {
            "category": Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
            "relevance": CategoryRelevance.MATERIAL,
            "data_available": True,
            "data_quality_tier": DataQualityTier.TIER_3,
            "estimated_materiality_pct": Decimal("60.00"),
            "screening_result": ScreeningResult.COMPLETE,
            "recommended_action": None,
        }
        defaults.update(overrides)
        return CategoryCompletenessEntry(**defaults)

    def test_category_completeness_entry_valid(self):
        """Valid CategoryCompletenessEntry must construct without error."""
        entry = self._make_entry()
        assert entry.category == Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES
        assert entry.data_available is True

    def test_category_completeness_entry_frozen(self):
        """CategoryCompletenessEntry must be frozen (immutable)."""
        entry = self._make_entry()
        with pytest.raises(Exception):
            entry.data_available = False  # type: ignore[misc]

    def test_data_available_true(self):
        """data_available True must be accepted."""
        entry = self._make_entry(data_available=True)
        assert entry.data_available is True

    def test_data_available_false(self):
        """data_available False must be accepted."""
        entry = self._make_entry(data_available=False)
        assert entry.data_available is False

    def test_estimated_materiality_pct_zero(self):
        """Materiality percentage 0 must be accepted."""
        entry = self._make_entry(estimated_materiality_pct=Decimal("0.00"))
        assert entry.estimated_materiality_pct == Decimal("0.00")

    def test_estimated_materiality_pct_100(self):
        """Materiality percentage 100 must be accepted."""
        entry = self._make_entry(estimated_materiality_pct=Decimal("100.00"))
        assert entry.estimated_materiality_pct == Decimal("100.00")

    def test_data_quality_tier_optional(self):
        """data_quality_tier must be optional (None accepted)."""
        entry = self._make_entry(data_quality_tier=None)
        assert entry.data_quality_tier is None

    def test_recommended_action_optional(self):
        """recommended_action must be optional (None accepted)."""
        entry = self._make_entry(recommended_action=None)
        assert entry.recommended_action is None

    def test_recommended_action_with_value(self):
        """recommended_action must accept a string value."""
        entry = self._make_entry(recommended_action="Collect data for this category")
        assert entry.recommended_action == "Collect data for this category"

    def test_screening_result_complete(self):
        """screening_result COMPLETE must be accepted."""
        entry = self._make_entry(screening_result=ScreeningResult.COMPLETE)
        assert entry.screening_result == ScreeningResult.COMPLETE

    def test_screening_result_missing(self):
        """screening_result MISSING must be accepted."""
        entry = self._make_entry(screening_result=ScreeningResult.MISSING)
        assert entry.screening_result == ScreeningResult.MISSING

    def test_screening_result_partial(self):
        """screening_result PARTIAL must be accepted."""
        entry = self._make_entry(screening_result=ScreeningResult.PARTIAL)
        assert entry.screening_result == ScreeningResult.PARTIAL


# ==============================================================================
# CompletenessReport MODEL
# ==============================================================================


class TestCompletenessReportModel:
    """Test CompletenessReport Pydantic model."""

    def _make_report(self, **overrides) -> CompletenessReport:
        """Helper to create a valid CompletenessReport."""
        defaults = {
            "company_type": CompanyType.MANUFACTURER,
            "entries": [],
            "overall_score": Decimal("62.50"),
            "categories_reported": 8,
            "categories_material": 7,
            "gaps": [],
            "provenance_hash": "a" * 64,
        }
        defaults.update(overrides)
        return CompletenessReport(**defaults)

    def test_completeness_report_valid(self):
        """Valid CompletenessReport must construct without error."""
        report = self._make_report()
        assert report.company_type == CompanyType.MANUFACTURER

    def test_completeness_report_frozen(self):
        """CompletenessReport must be frozen (immutable)."""
        report = self._make_report()
        with pytest.raises(Exception):
            report.overall_score = Decimal("99.00")  # type: ignore[misc]

    def test_overall_score_range_zero(self):
        """Overall score 0 must be accepted."""
        report = self._make_report(overall_score=Decimal("0.00"))
        assert report.overall_score == Decimal("0.00")

    def test_overall_score_range_100(self):
        """Overall score 100 must be accepted."""
        report = self._make_report(overall_score=Decimal("100.00"))
        assert report.overall_score == Decimal("100.00")

    def test_overall_score_negative_rejected(self):
        """Negative overall score must be rejected."""
        with pytest.raises(PydanticValidationError):
            self._make_report(overall_score=Decimal("-1.00"))

    def test_overall_score_above_100_rejected(self):
        """Overall score above 100 must be rejected."""
        with pytest.raises(PydanticValidationError):
            self._make_report(overall_score=Decimal("101.00"))

    def test_categories_reported_range(self):
        """categories_reported must be in range 0-15."""
        report = self._make_report(categories_reported=0)
        assert report.categories_reported == 0
        report = self._make_report(categories_reported=15)
        assert report.categories_reported == 15

    def test_categories_reported_negative_rejected(self):
        """Negative categories_reported must be rejected."""
        with pytest.raises(PydanticValidationError):
            self._make_report(categories_reported=-1)

    def test_categories_reported_above_15_rejected(self):
        """categories_reported above 15 must be rejected."""
        with pytest.raises(PydanticValidationError):
            self._make_report(categories_reported=16)

    def test_entries_default_empty_list(self):
        """Default entries must be an empty list."""
        report = self._make_report()
        assert report.entries == []

    def test_gaps_default_empty_list(self):
        """Default gaps must be an empty list."""
        report = self._make_report()
        assert report.gaps == []

    def test_provenance_hash_present(self):
        """Provenance hash must be present in report."""
        report = self._make_report(provenance_hash="b" * 64)
        assert len(report.provenance_hash) == 64


# ==============================================================================
# ComplianceFinding MODEL
# ==============================================================================


class TestComplianceFindingModel:
    """Test ComplianceFinding Pydantic model."""

    def _make_finding(self, **overrides) -> ComplianceFinding:
        """Helper to create a valid ComplianceFinding."""
        defaults = {
            "rule_code": "GHG-SCR-001",
            "description": "All material categories must be reported",
            "severity": ComplianceSeverity.HIGH,
            "framework": "ghg_protocol",
            "status": ComplianceStatus.FAIL,
            "recommendation": "Collect data for missing material categories",
            "regulation_reference": "GHG Protocol Scope 3 Standard, Chapter 7",
            "details": {"missing_categories": [2, 10, 12]},
        }
        defaults.update(overrides)
        return ComplianceFinding(**defaults)

    def test_compliance_finding_valid(self):
        """Valid ComplianceFinding must construct without error."""
        finding = self._make_finding()
        assert finding.rule_code == "GHG-SCR-001"
        assert finding.severity == ComplianceSeverity.HIGH

    def test_compliance_finding_frozen(self):
        """ComplianceFinding must be frozen (immutable)."""
        finding = self._make_finding()
        with pytest.raises(Exception):
            finding.status = ComplianceStatus.PASS  # type: ignore[misc]

    def test_compliance_finding_default_status_fail(self):
        """Default status must be FAIL."""
        finding = ComplianceFinding(
            rule_code="TEST-001",
            description="Test finding",
            severity=ComplianceSeverity.LOW,
            framework="ghg_protocol",
        )
        assert finding.status == ComplianceStatus.FAIL

    def test_compliance_finding_recommendation_optional(self):
        """recommendation must be optional (None accepted)."""
        finding = self._make_finding(recommendation=None)
        assert finding.recommendation is None

    def test_compliance_finding_regulation_reference_optional(self):
        """regulation_reference must be optional (None accepted)."""
        finding = self._make_finding(regulation_reference=None)
        assert finding.regulation_reference is None

    def test_compliance_finding_details_optional(self):
        """details must be optional (None accepted)."""
        finding = self._make_finding(details=None)
        assert finding.details is None

    def test_compliance_finding_details_with_dict(self):
        """details must accept a dictionary."""
        finding = self._make_finding(details={"key": "value", "count": 42})
        assert finding.details == {"key": "value", "count": 42}


# ==============================================================================
# DetailedComplianceAssessment MODEL
# ==============================================================================


class TestDetailedComplianceAssessmentModel:
    """Test DetailedComplianceAssessment Pydantic model."""

    def _make_assessment(self, **overrides) -> DetailedComplianceAssessment:
        """Helper to create a valid DetailedComplianceAssessment."""
        defaults = {
            "framework": ComplianceFramework.GHG_PROTOCOL,
            "framework_description": "GHG Protocol Corporate Value Chain (Scope 3) Standard",
            "status": ComplianceStatus.PASS,
            "score": Decimal("85.00"),
            "findings": [],
            "recommendations": [],
            "passed_checks": 17,
            "failed_checks": 3,
            "warning_checks": 2,
            "total_checks": 22,
            "provenance_hash": "b" * 64,
            "assessed_at": "2025-03-15T12:00:00+00:00",
            "processing_time_ms": 25.0,
        }
        defaults.update(overrides)
        return DetailedComplianceAssessment(**defaults)

    def test_detailed_compliance_assessment_valid(self):
        """Valid DetailedComplianceAssessment must construct without error."""
        assessment = self._make_assessment()
        assert assessment.framework == ComplianceFramework.GHG_PROTOCOL
        assert assessment.status == ComplianceStatus.PASS
        assert assessment.score == Decimal("85.00")

    def test_detailed_compliance_assessment_frozen(self):
        """DetailedComplianceAssessment must be frozen (immutable)."""
        assessment = self._make_assessment()
        with pytest.raises(Exception):
            assessment.score = Decimal("99.00")  # type: ignore[misc]

    def test_compliance_assessment_score_range_zero(self):
        """Compliance score 0 must be accepted."""
        assessment = self._make_assessment(score=Decimal("0.00"))
        assert assessment.score == Decimal("0.00")

    def test_compliance_assessment_score_range_100(self):
        """Compliance score 100 must be accepted."""
        assessment = self._make_assessment(score=Decimal("100.00"))
        assert assessment.score == Decimal("100.00")

    def test_compliance_assessment_score_negative_rejected(self):
        """Negative compliance score must be rejected."""
        with pytest.raises(PydanticValidationError):
            self._make_assessment(score=Decimal("-1.00"))

    def test_compliance_assessment_score_above_100_rejected(self):
        """Compliance score above 100 must be rejected."""
        with pytest.raises(PydanticValidationError):
            self._make_assessment(score=Decimal("101.00"))

    def test_compliance_assessment_checks_defaults(self):
        """Check counters must default to 0."""
        assessment = DetailedComplianceAssessment(
            framework=ComplianceFramework.CDP,
            status=ComplianceStatus.PASS,
            score=Decimal("90.00"),
        )
        assert assessment.passed_checks == 0
        assert assessment.failed_checks == 0
        assert assessment.warning_checks == 0
        assert assessment.total_checks == 0

    def test_compliance_assessment_findings_list(self):
        """findings must accept a list of ComplianceFinding."""
        finding = ComplianceFinding(
            rule_code="CDP-001",
            description="Test",
            severity=ComplianceSeverity.LOW,
            framework="cdp",
        )
        assessment = self._make_assessment(findings=[finding])
        assert len(assessment.findings) == 1
        assert assessment.findings[0].rule_code == "CDP-001"


# ==============================================================================
# BenchmarkComparison MODEL
# ==============================================================================


class TestBenchmarkComparisonModel:
    """Test BenchmarkComparison Pydantic model."""

    def _make_comparison(self, **overrides) -> BenchmarkComparison:
        """Helper to create a valid BenchmarkComparison."""
        defaults = {
            "category": Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
            "benchmark_pct": Decimal("60.00"),
            "actual_pct": Decimal("55.00"),
            "deviation_pct": Decimal("-5.00"),
            "within_tolerance": True,
            "flag": None,
        }
        defaults.update(overrides)
        return BenchmarkComparison(**defaults)

    def test_benchmark_comparison_valid(self):
        """Valid BenchmarkComparison must construct without error."""
        comparison = self._make_comparison()
        assert comparison.category == Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES
        assert comparison.within_tolerance is True

    def test_benchmark_comparison_frozen(self):
        """BenchmarkComparison must be frozen (immutable)."""
        comparison = self._make_comparison()
        with pytest.raises(Exception):
            comparison.within_tolerance = False  # type: ignore[misc]

    def test_benchmark_comparison_positive_deviation(self):
        """Positive deviation (above benchmark) must be accepted."""
        comparison = self._make_comparison(
            actual_pct=Decimal("75.00"),
            deviation_pct=Decimal("15.00"),
            within_tolerance=False,
            flag="Category 1 is 15.00% ABOVE benchmark.",
        )
        assert comparison.deviation_pct == Decimal("15.00")
        assert comparison.within_tolerance is False
        assert comparison.flag is not None

    def test_benchmark_comparison_negative_deviation(self):
        """Negative deviation (below benchmark) must be accepted."""
        comparison = self._make_comparison(
            actual_pct=Decimal("40.00"),
            deviation_pct=Decimal("-20.00"),
            within_tolerance=False,
            flag="Category 1 is 20.00% BELOW benchmark.",
        )
        assert comparison.deviation_pct == Decimal("-20.00")
        assert comparison.within_tolerance is False

    def test_benchmark_comparison_flag_optional(self):
        """flag must be optional (None accepted when within tolerance)."""
        comparison = self._make_comparison(flag=None)
        assert comparison.flag is None

    def test_benchmark_comparison_benchmark_pct_zero(self):
        """Benchmark percentage 0 must be accepted."""
        comparison = self._make_comparison(benchmark_pct=Decimal("0.00"))
        assert comparison.benchmark_pct == Decimal("0.00")

    def test_benchmark_comparison_actual_pct_zero(self):
        """Actual percentage 0 must be accepted."""
        comparison = self._make_comparison(actual_pct=Decimal("0.00"))
        assert comparison.actual_pct == Decimal("0.00")

    def test_benchmark_comparison_benchmark_pct_100(self):
        """Benchmark percentage 100 must be accepted."""
        comparison = self._make_comparison(benchmark_pct=Decimal("100.00"))
        assert comparison.benchmark_pct == Decimal("100.00")

    def test_benchmark_comparison_negative_benchmark_rejected(self):
        """Negative benchmark_pct must be rejected."""
        with pytest.raises(PydanticValidationError):
            self._make_comparison(benchmark_pct=Decimal("-1.00"))
