# -*- coding: utf-8 -*-
"""
Tests for BaseYearReportingEngine (Engine 10).

Covers report generation, framework mapping, export, disclosure completeness.
Target: ~40 tests.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.base_year_reporting_engine import (
    BaseYearReportingEngine,
    ReportConfig,
    InventoryData,
    BaseYearReport,
    MultiFrameworkReport,
    ReportingFramework,
    OutputFormat,
    ReportSection,
    ReportContent,
    FrameworkRequirement,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def reporting_engine():
    return BaseYearReportingEngine()


@pytest.fixture
def sample_inventory_data():
    return InventoryData(
        organization_name="Test Corp",
        organization_id="ORG-001",
        base_year=2022,
        reporting_year=2026,
        base_year_scope1_tco2e=Decimal("5000"),
        base_year_scope2_location_tco2e=Decimal("3000"),
        base_year_scope2_market_tco2e=Decimal("2800"),
        base_year_scope3_tco2e=Decimal("15000"),
        base_year_total_tco2e=Decimal("25800"),
    )


@pytest.fixture
def report_config():
    return ReportConfig(
        output_format=OutputFormat.JSON,
        include_provenance=True,
    )


@pytest.fixture
def minimal_inventory():
    return InventoryData(
        organization_name="Minimal Corp",
        base_year=2022,
        reporting_year=2026,
    )


# ============================================================================
# Engine Init
# ============================================================================

class TestBaseYearReportingEngineInit:
    def test_engine_creation(self, reporting_engine):
        assert reporting_engine is not None

    def test_engine_is_instance(self, reporting_engine):
        assert isinstance(reporting_engine, BaseYearReportingEngine)


# ============================================================================
# Generate Report (single framework)
# ============================================================================

class TestGenerateReport:
    def test_generate_ghg_protocol_report(self, reporting_engine, sample_inventory_data, report_config):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.GHG_PROTOCOL, report_config
        )
        assert isinstance(report, BaseYearReport)

    def test_generate_cdp_report(self, reporting_engine, sample_inventory_data, report_config):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.CDP, report_config
        )
        assert isinstance(report, BaseYearReport)

    def test_generate_sbti_report(self, reporting_engine, sample_inventory_data, report_config):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.SBTI, report_config
        )
        assert isinstance(report, BaseYearReport)

    def test_generate_sec_report(self, reporting_engine, sample_inventory_data, report_config):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.SEC, report_config
        )
        assert isinstance(report, BaseYearReport)

    def test_generate_esrs_report(self, reporting_engine, sample_inventory_data, report_config):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.ESRS_E1, report_config
        )
        assert isinstance(report, BaseYearReport)

    def test_report_has_provenance_hash(self, reporting_engine, sample_inventory_data, report_config):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.GHG_PROTOCOL, report_config
        )
        assert report.provenance_hash != ""
        assert len(report.provenance_hash) == 64

    def test_report_has_sections(self, reporting_engine, sample_inventory_data, report_config):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.GHG_PROTOCOL, report_config
        )
        assert len(report.sections) >= 1

    def test_report_has_framework(self, reporting_engine, sample_inventory_data, report_config):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.GHG_PROTOCOL, report_config
        )
        assert report.framework == ReportingFramework.GHG_PROTOCOL

    def test_report_has_organization(self, reporting_engine, sample_inventory_data, report_config):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.GHG_PROTOCOL, report_config
        )
        assert report.organization == "Test Corp"

    def test_report_has_report_id(self, reporting_engine, sample_inventory_data, report_config):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.GHG_PROTOCOL, report_config
        )
        assert report.report_id != ""

    def test_report_default_config(self, reporting_engine, sample_inventory_data):
        """Generate report with default config (no explicit config)."""
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.GHG_PROTOCOL
        )
        assert isinstance(report, BaseYearReport)


# ============================================================================
# Multi-Framework Report
# ============================================================================

class TestMultiFrameworkReport:
    def test_generate_multi_framework(self, reporting_engine, sample_inventory_data):
        report = reporting_engine.generate_multi_framework_report(
            sample_inventory_data,
            [ReportingFramework.GHG_PROTOCOL, ReportingFramework.CDP, ReportingFramework.SBTI],
        )
        assert isinstance(report, MultiFrameworkReport)

    def test_multi_framework_has_reports(self, reporting_engine, sample_inventory_data):
        report = reporting_engine.generate_multi_framework_report(
            sample_inventory_data,
            [ReportingFramework.GHG_PROTOCOL, ReportingFramework.CDP],
        )
        assert len(report.reports) >= 1

    def test_multi_framework_provenance(self, reporting_engine, sample_inventory_data):
        report = reporting_engine.generate_multi_framework_report(
            sample_inventory_data,
            [ReportingFramework.GHG_PROTOCOL],
        )
        assert report.provenance_hash != ""

    def test_multi_framework_with_config(self, reporting_engine, sample_inventory_data, report_config):
        report = reporting_engine.generate_multi_framework_report(
            sample_inventory_data,
            [ReportingFramework.GHG_PROTOCOL, ReportingFramework.CDP],
            report_config,
        )
        assert isinstance(report, MultiFrameworkReport)

    def test_multi_framework_single(self, reporting_engine, sample_inventory_data):
        report = reporting_engine.generate_multi_framework_report(
            sample_inventory_data,
            [ReportingFramework.GHG_PROTOCOL],
        )
        assert len(report.reports) >= 1


# ============================================================================
# Framework-Specific Format Methods
# ============================================================================

class TestFrameworkSpecificReports:
    def test_format_ghg_protocol(self, reporting_engine, sample_inventory_data):
        report = reporting_engine.format_ghg_protocol_report(sample_inventory_data)
        assert report is not None
        assert isinstance(report, str)

    def test_format_cdp_c5(self, reporting_engine, sample_inventory_data):
        report = reporting_engine.format_cdp_c5(sample_inventory_data)
        assert report is not None
        assert isinstance(report, str)

    def test_format_esrs_e1_6(self, reporting_engine, sample_inventory_data):
        report = reporting_engine.format_esrs_e1_6(sample_inventory_data)
        assert report is not None
        assert isinstance(report, str)

    def test_format_sbti(self, reporting_engine, sample_inventory_data):
        report = reporting_engine.format_sbti_report(sample_inventory_data)
        assert report is not None
        assert isinstance(report, str)

    def test_format_sec(self, reporting_engine, sample_inventory_data):
        report = reporting_engine.format_sec_report(sample_inventory_data)
        assert report is not None
        assert isinstance(report, str)


# ============================================================================
# Export Report
# ============================================================================

class TestExportReport:
    def test_export_json(self, reporting_engine, sample_inventory_data, report_config):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.GHG_PROTOCOL, report_config
        )
        exported = reporting_engine.export_report(report, OutputFormat.JSON)
        assert exported is not None
        assert isinstance(exported, str)
        assert len(exported) > 0

    def test_export_markdown(self, reporting_engine, sample_inventory_data):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.GHG_PROTOCOL
        )
        exported = reporting_engine.export_report(report, OutputFormat.MARKDOWN)
        assert exported is not None
        assert isinstance(exported, str)

    def test_export_default_format(self, reporting_engine, sample_inventory_data):
        report = reporting_engine.generate_report(
            sample_inventory_data, ReportingFramework.GHG_PROTOCOL
        )
        exported = reporting_engine.export_report(report)
        assert exported is not None
        assert isinstance(exported, str)


# ============================================================================
# Check Disclosure Completeness
# ============================================================================

class TestCheckDisclosureCompleteness:
    def test_check_ghg_protocol_completeness(self, reporting_engine, sample_inventory_data):
        score, gaps = reporting_engine.check_disclosure_completeness(
            sample_inventory_data, ReportingFramework.GHG_PROTOCOL
        )
        assert isinstance(score, Decimal)
        assert isinstance(gaps, list)

    def test_check_cdp_completeness(self, reporting_engine, sample_inventory_data):
        score, gaps = reporting_engine.check_disclosure_completeness(
            sample_inventory_data, ReportingFramework.CDP
        )
        assert isinstance(score, Decimal)

    def test_check_sbti_completeness(self, reporting_engine, sample_inventory_data):
        score, gaps = reporting_engine.check_disclosure_completeness(
            sample_inventory_data, ReportingFramework.SBTI
        )
        assert isinstance(score, Decimal)

    def test_check_minimal_completeness(self, reporting_engine, minimal_inventory):
        score, gaps = reporting_engine.check_disclosure_completeness(
            minimal_inventory, ReportingFramework.GHG_PROTOCOL
        )
        assert isinstance(score, Decimal)
        # Minimal data should have more gaps
        assert isinstance(gaps, list)


# ============================================================================
# InventoryData Model
# ============================================================================

class TestInventoryDataModel:
    def test_create_full_inventory(self, sample_inventory_data):
        assert sample_inventory_data.organization_name == "Test Corp"
        assert sample_inventory_data.base_year == 2022
        assert sample_inventory_data.reporting_year == 2026
        assert sample_inventory_data.base_year_scope1_tco2e == Decimal("5000")

    def test_create_minimal_inventory(self, minimal_inventory):
        assert minimal_inventory.organization_name == "Minimal Corp"
        assert minimal_inventory.base_year_scope1_tco2e == Decimal("0")

    def test_inventory_has_defaults(self, minimal_inventory):
        assert minimal_inventory.base_year_scope2_location_tco2e == Decimal("0")
        assert minimal_inventory.base_year_scope2_market_tco2e == Decimal("0")
        assert minimal_inventory.base_year_scope3_tco2e == Decimal("0")
        assert minimal_inventory.base_year_total_tco2e == Decimal("0")

    def test_inventory_current_year_defaults(self, minimal_inventory):
        assert minimal_inventory.current_year_scope1_tco2e == Decimal("0")
        assert minimal_inventory.current_year_scope2_location_tco2e == Decimal("0")


# ============================================================================
# ReportConfig Model
# ============================================================================

class TestReportConfigModel:
    def test_create_config(self, report_config):
        assert report_config.output_format == OutputFormat.JSON
        assert report_config.include_provenance is True

    def test_config_defaults(self):
        config = ReportConfig()
        assert config.output_format == OutputFormat.MARKDOWN
        assert config.include_provenance is True
        assert config.decimal_places == 3
        assert config.include_recommendations is True

    def test_config_custom_decimal_places(self):
        config = ReportConfig(decimal_places=6)
        assert config.decimal_places == 6


# ============================================================================
# Enums
# ============================================================================

class TestEnums:
    def test_reporting_frameworks(self):
        assert ReportingFramework.GHG_PROTOCOL is not None
        assert ReportingFramework.CDP is not None
        assert ReportingFramework.SBTI is not None
        assert ReportingFramework.SEC is not None
        assert ReportingFramework.ESRS_E1 is not None
        assert ReportingFramework.ISO_14064 is not None
        assert ReportingFramework.TCFD is not None
        assert ReportingFramework.SB_253 is not None
        assert len(ReportingFramework) == 8

    def test_output_formats(self):
        assert OutputFormat.JSON is not None
        assert OutputFormat.MARKDOWN is not None
        assert OutputFormat.HTML is not None
        assert OutputFormat.CSV is not None
        assert len(OutputFormat) == 4
