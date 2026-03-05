# -*- coding: utf-8 -*-
"""
Unit tests for ISO 14064-1:2018 Platform Configuration.

Tests all enumerations, GWP tables, constants, and the ISO14064AppConfig
class with 20+ individual test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    ActionCategory,
    ActionStatus,
    ConsolidationApproach,
    DataQualityTier,
    FindingSeverity,
    FindingStatus,
    GHGGas,
    GWPSource,
    GWP_AR5,
    GWP_AR6,
    GWP_TABLES,
    InventoryStatus,
    ISOCategory,
    ISO14064AppConfig,
    ISO_CATEGORY_NAMES,
    MANDATORY_REPORTING_ELEMENTS,
    MRV_AGENT_TO_ISO_CATEGORY,
    PermanenceLevel,
    QuantificationMethod,
    RemovalType,
    ReportFormat,
    ReportingPeriod,
    SECTOR_BENCHMARKS,
    SignificanceLevel,
    UNCERTAINTY_CV_BY_TIER,
    VerificationLevel,
    VerificationStage,
)


class TestGHGGasEnum:
    """Test seven GHGs per ISO 14064-1:2018."""

    def test_seven_ghg_gases_defined(self):
        assert len(GHGGas) == 7

    def test_gas_values(self):
        expected = {"CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"}
        actual = {g.value for g in GHGGas}
        assert actual == expected


class TestISOCategoryEnum:
    """Test six ISO categories."""

    def test_six_categories_defined(self):
        assert len(ISOCategory) == 6

    def test_category_1_is_direct(self):
        assert ISOCategory.CATEGORY_1_DIRECT.value == "category_1_direct"

    def test_category_display_names_complete(self):
        for cat in ISOCategory:
            assert cat in ISO_CATEGORY_NAMES, f"Missing display name for {cat.value}"


class TestConsolidationApproach:
    """Test three consolidation approaches per Clause 5.1."""

    def test_three_approaches(self):
        assert len(ConsolidationApproach) == 3

    def test_operational_control(self):
        assert ConsolidationApproach.OPERATIONAL_CONTROL.value == "operational_control"

    def test_equity_share(self):
        assert ConsolidationApproach.EQUITY_SHARE.value == "equity_share"


class TestGWPTables:
    """Test GWP tables for AR5 and AR6."""

    def test_ar5_co2_equals_1(self):
        assert GWP_AR5[GHGGas.CO2] == 1

    def test_ar5_ch4_equals_28(self):
        assert GWP_AR5[GHGGas.CH4] == 28

    def test_ar5_n2o_equals_265(self):
        assert GWP_AR5[GHGGas.N2O] == 265

    def test_ar5_sf6_equals_23500(self):
        assert GWP_AR5[GHGGas.SF6] == 23500

    def test_ar6_ch4_equals_27_9(self):
        assert GWP_AR6[GHGGas.CH4] == 27.9

    def test_ar6_n2o_equals_273(self):
        assert GWP_AR6[GHGGas.N2O] == 273

    def test_ar6_sf6_equals_25200(self):
        assert GWP_AR6[GHGGas.SF6] == 25200

    def test_gwp_tables_has_ar5_and_ar6(self):
        assert GWPSource.AR5 in GWP_TABLES
        assert GWPSource.AR6 in GWP_TABLES

    def test_all_seven_gases_in_ar5(self):
        for gas in GHGGas:
            assert gas in GWP_AR5, f"Missing {gas.value} in AR5 table"

    def test_all_seven_gases_in_ar6(self):
        for gas in GHGGas:
            assert gas in GWP_AR6, f"Missing {gas.value} in AR6 table"


class TestUncertaintyCVByTier:
    """Test uncertainty CV percentages by data quality tier."""

    def test_tier_1_cv_is_50(self):
        assert UNCERTAINTY_CV_BY_TIER[DataQualityTier.TIER_1] == Decimal("50.0")

    def test_tier_4_cv_is_2(self):
        assert UNCERTAINTY_CV_BY_TIER[DataQualityTier.TIER_4] == Decimal("2.0")

    def test_cv_decreases_with_tier(self):
        tiers = [DataQualityTier.TIER_1, DataQualityTier.TIER_2,
                 DataQualityTier.TIER_3, DataQualityTier.TIER_4]
        cvs = [UNCERTAINTY_CV_BY_TIER[t] for t in tiers]
        for i in range(len(cvs) - 1):
            assert cvs[i] > cvs[i + 1], "CV should decrease with higher tier"


class TestMRVAgentMapping:
    """Test MRV agent to ISO category mapping."""

    def test_28_agents_mapped(self):
        assert len(MRV_AGENT_TO_ISO_CATEGORY) == 28

    def test_stationary_combustion_maps_to_cat1(self):
        assert MRV_AGENT_TO_ISO_CATEGORY["stationary_combustion"] == ISOCategory.CATEGORY_1_DIRECT

    def test_scope2_location_maps_to_cat2(self):
        assert MRV_AGENT_TO_ISO_CATEGORY["scope2_location"] == ISOCategory.CATEGORY_2_ENERGY

    def test_business_travel_maps_to_cat3(self):
        assert MRV_AGENT_TO_ISO_CATEGORY["business_travel"] == ISOCategory.CATEGORY_3_TRANSPORT

    def test_investments_maps_to_cat6(self):
        assert MRV_AGENT_TO_ISO_CATEGORY["investments"] == ISOCategory.CATEGORY_6_OTHER


class TestMandatoryReportingElements:
    """Test the 14 mandatory reporting elements."""

    def test_14_elements(self):
        assert len(MANDATORY_REPORTING_ELEMENTS) == 14

    def test_elements_start_with_mre(self):
        for element in MANDATORY_REPORTING_ELEMENTS:
            assert element.startswith("MRE-")

    def test_elements_numbered_01_to_14(self):
        ids = [int(e.split("-")[1]) for e in MANDATORY_REPORTING_ELEMENTS]
        assert ids == list(range(1, 15))


class TestSectorBenchmarks:
    """Test sector benchmark data."""

    def test_12_sectors_defined(self):
        assert len(SECTOR_BENCHMARKS) == 12

    def test_energy_sector_has_revenue_intensity(self):
        assert "revenue_intensity" in SECTOR_BENCHMARKS["energy"]
        assert SECTOR_BENCHMARKS["energy"]["revenue_intensity"] == Decimal("850.0")


class TestISO14064AppConfig:
    """Test the main configuration class."""

    def test_default_app_name(self):
        config = ISO14064AppConfig()
        assert config.app_name == "GL-ISO14064-APP"

    def test_default_version(self):
        config = ISO14064AppConfig()
        assert config.version == "1.0.0"

    def test_default_consolidation_approach(self):
        config = ISO14064AppConfig()
        assert config.default_consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL

    def test_default_gwp_source(self):
        config = ISO14064AppConfig()
        assert config.default_gwp_source == GWPSource.AR5

    def test_default_significance_threshold(self):
        config = ISO14064AppConfig()
        assert config.significance_threshold_percent == Decimal("1.0")

    def test_default_recalculation_threshold(self):
        config = ISO14064AppConfig()
        assert config.recalculation_threshold_percent == Decimal("5.0")

    def test_default_monte_carlo_iterations(self):
        config = ISO14064AppConfig()
        assert config.monte_carlo_iterations == 10000

    def test_default_confidence_levels(self):
        config = ISO14064AppConfig()
        assert config.confidence_levels == [90, 95, 99]

    def test_default_report_format(self):
        config = ISO14064AppConfig()
        assert config.default_report_format == ReportFormat.JSON


class TestEnumCompleteness:
    """Test that all required enums are present."""

    def test_permanence_levels(self):
        assert len(PermanenceLevel) == 5

    def test_removal_types(self):
        assert len(RemovalType) == 8

    def test_data_quality_tiers(self):
        assert len(DataQualityTier) == 4

    def test_quantification_methods(self):
        assert len(QuantificationMethod) == 3

    def test_verification_stages(self):
        assert len(VerificationStage) == 5

    def test_finding_severities(self):
        assert len(FindingSeverity) == 4

    def test_report_formats(self):
        assert len(ReportFormat) == 4
