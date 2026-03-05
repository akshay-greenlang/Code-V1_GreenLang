# -*- coding: utf-8 -*-
"""
Unit tests for ReportGenerator -- ISO 14064-1:2018 Clause 8.

Tests report generation, mandatory element compliance (14 MREs),
section content generation, multi-format export (JSON, CSV, Excel,
PDF), report history tracking, and summary with 25+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    DataQualityTier,
    ISOCategory,
    ReportFormat,
)
from services.models import (
    CategoryResult,
    ISOInventory,
    Organization,
    _new_id,
    _now,
)
from services.report_generator import (
    REPORT_SECTIONS,
    SECTION_TITLES,
    SECTION_TO_MRE,
    ReportGenerator,
)


# ---------------------------------------------------------------------------
# Fixtures specific to report tests
# ---------------------------------------------------------------------------

@pytest.fixture
def org_store():
    """Populated organization store."""
    org = Organization(
        name="Acme Corp",
        industry="manufacturing",
        country="US",
        description="Test company",
        contact_person="Jane Doe",
    )
    return {org.id: org}


@pytest.fixture
def inventory_store(org_store):
    """Populated inventory store with one inventory."""
    org_id = list(org_store.keys())[0]
    inv = ISOInventory(
        org_id=org_id,
        year=2025,
    )
    return {inv.id: inv}


@pytest.fixture
def category_store(inventory_store, sample_category_results):
    """Category store populated with sample results."""
    inv_id = list(inventory_store.keys())[0]
    return {inv_id: sample_category_results}


@pytest.fixture
def report_gen(default_config, inventory_store, org_store, category_store):
    """ReportGenerator wired to populated stores."""
    return ReportGenerator(
        config=default_config,
        inventory_store=inventory_store,
        org_store=org_store,
        category_store=category_store,
    )


@pytest.fixture
def inv_id(inventory_store):
    """Convenience: single inventory ID from the store."""
    return list(inventory_store.keys())[0]


# ===========================================================================
# Tests
# ===========================================================================


class TestReportSectionConstants:
    """Test report section definitions."""

    def test_report_sections_count(self):
        assert len(REPORT_SECTIONS) == 23

    def test_section_titles_count(self):
        assert len(SECTION_TITLES) == 23

    def test_section_to_mre_count(self):
        # 14 mandatory elements are mapped
        assert len(SECTION_TO_MRE) == 14

    def test_all_section_keys_have_titles(self):
        for key in REPORT_SECTIONS:
            assert key in SECTION_TITLES, f"Missing title for {key}"

    def test_mre_ids_range(self):
        for mre_id in SECTION_TO_MRE.values():
            assert mre_id.startswith("MRE-")
            num = int(mre_id.split("-")[1])
            assert 1 <= num <= 14


class TestGenerateReport:
    """Test main report generation method."""

    def test_generate_all_sections(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id)
        assert len(report.sections) == 23

    def test_report_has_inventory_id(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id)
        assert report.inventory_id == inv_id

    def test_report_has_provenance_hash(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id)
        assert report.provenance_hash is not None
        assert len(report.provenance_hash) == 64

    def test_report_format_default_json(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id)
        assert report.format == ReportFormat.JSON

    def test_report_has_mandatory_elements(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id)
        assert len(report.mandatory_elements) == 14

    def test_mandatory_compliance_pct_is_decimal(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id)
        assert isinstance(report.mandatory_compliance_pct, Decimal)

    def test_generate_specific_sections(self, report_gen, inv_id):
        report = report_gen.generate_report(
            inv_id, sections=["reporting_organization", "total_emissions"],
        )
        assert len(report.sections) == 2
        keys = [s.key for s in report.sections]
        assert "reporting_organization" in keys
        assert "total_emissions" in keys

    def test_generate_report_pdf_format(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id, ReportFormat.PDF)
        assert report.format == ReportFormat.PDF

    def test_nonexistent_inventory_raises(self, report_gen):
        with pytest.raises(ValueError, match="not found"):
            report_gen.generate_report("bad-id")

    def test_unknown_section_skipped(self, report_gen, inv_id):
        report = report_gen.generate_report(
            inv_id, sections=["total_emissions", "totally_fake_section"],
        )
        # Only the valid section should be present
        assert len(report.sections) == 1


class TestCheckCompliance:
    """Test mandatory element compliance check."""

    def test_compliance_dict_structure(self, report_gen, inv_id):
        result = report_gen.check_compliance(inv_id)
        assert "total_mandatory" in result
        assert result["total_mandatory"] == 14
        assert "present" in result
        assert "missing_count" in result
        assert "missing_elements" in result
        assert "compliant" in result
        assert "compliance_pct" in result

    def test_compliance_with_populated_categories(self, report_gen, inv_id):
        result = report_gen.check_compliance(inv_id)
        # With category data provided, several MREs should be present
        assert result["present"] > 0

    def test_compliance_missing_elements_sorted(self, report_gen, inv_id):
        result = report_gen.check_compliance(inv_id)
        missing = result["missing_elements"]
        assert missing == sorted(missing)


class TestSectionContent:
    """Test individual section content generation."""

    def test_reporting_org_section(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id, sections=["reporting_organization"])
        content = report.sections[0].content
        assert "name" in content
        assert content["name"] == "Acme Corp"

    def test_responsible_person_section(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id, sections=["responsible_person"])
        content = report.sections[0].content
        assert "responsible_person" in content
        assert content["responsible_person"] == "Jane Doe"

    def test_reporting_period_section(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id, sections=["reporting_period"])
        content = report.sections[0].content
        assert content["reporting_year"] == 2025

    def test_total_emissions_section(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id, sections=["total_emissions"])
        content = report.sections[0].content
        # Total = 5000+3000+1500+2000+800+200 = 12500
        assert Decimal(content["gross_emissions_tco2e"]) == Decimal("12500")

    def test_gas_breakdown_section(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id, sections=["gas_breakdown"])
        content = report.sections[0].content
        assert "by_gas" in content
        assert "CO2" in content["by_gas"]

    def test_methodology_section(self, report_gen, inv_id):
        report = report_gen.generate_report(inv_id, sections=["quantification_methodology"])
        content = report.sections[0].content
        assert "ISO 14064-1:2018" in content["standard"]


class TestExportJson:
    """Test JSON export."""

    def test_export_json_has_metadata(self, report_gen, inv_id):
        export = report_gen.export_json(inv_id)
        assert "metadata" in export
        assert export["metadata"]["standard"] == "ISO 14064-1:2018"
        assert export["metadata"]["format"] == "json"

    def test_export_json_has_sections(self, report_gen, inv_id):
        export = report_gen.export_json(inv_id)
        assert "sections" in export
        assert len(export["sections"]) > 0

    def test_export_json_has_provenance(self, report_gen, inv_id):
        export = report_gen.export_json(inv_id)
        assert len(export["metadata"]["provenance_hash"]) == 64


class TestExportCsv:
    """Test CSV export."""

    def test_csv_is_string(self, report_gen, inv_id):
        csv_str = report_gen.export_csv(inv_id)
        assert isinstance(csv_str, str)

    def test_csv_has_header(self, report_gen, inv_id):
        csv_str = report_gen.export_csv(inv_id)
        first_line = csv_str.split("\n")[0]
        assert "ISO_Category" in first_line
        assert "Total_tCO2e" in first_line

    def test_csv_has_six_category_rows(self, report_gen, inv_id):
        csv_str = report_gen.export_csv(inv_id)
        lines = [l for l in csv_str.strip().split("\n") if l]
        # 1 header + 6 category rows
        assert len(lines) == 7


class TestExportExcel:
    """Test Excel structure export."""

    def test_excel_has_filename(self, report_gen, inv_id):
        result = report_gen.export_excel(inv_id)
        assert result["filename"].endswith(".xlsx")

    def test_excel_has_four_sheets(self, report_gen, inv_id):
        result = report_gen.export_excel(inv_id)
        assert len(result["sheets"]) == 4
        assert "Summary" in result["sheets"]
        assert "Category_Detail" in result["sheets"]
        assert "Gas_Breakdown" in result["sheets"]
        assert "Compliance_Check" in result["sheets"]

    def test_excel_summary_has_rows(self, report_gen, inv_id):
        result = report_gen.export_excel(inv_id)
        summary = result["sheets"]["Summary"]
        assert len(summary["rows"]) > 0


class TestExportPdf:
    """Test PDF structure export."""

    def test_pdf_has_filename(self, report_gen, inv_id):
        result = report_gen.export_pdf(inv_id)
        assert result["filename"].endswith(".pdf")

    def test_pdf_has_title(self, report_gen, inv_id):
        result = report_gen.export_pdf(inv_id)
        assert "ISO 14064-1:2018" in result["title"]

    def test_pdf_has_provenance(self, report_gen, inv_id):
        result = report_gen.export_pdf(inv_id)
        assert len(result["footer"]["provenance_hash"]) == 64


class TestReportHistory:
    """Test report history tracking."""

    def test_history_empty_initially(self, report_gen, inv_id):
        assert len(report_gen.get_report_history(inv_id)) == 0

    def test_history_accumulates(self, report_gen, inv_id):
        report_gen.generate_report(inv_id)
        report_gen.generate_report(inv_id, ReportFormat.PDF)
        history = report_gen.get_report_history(inv_id)
        assert len(history) == 2

    def test_history_for_unknown_inventory(self, report_gen):
        assert report_gen.get_report_history("unknown") == []
