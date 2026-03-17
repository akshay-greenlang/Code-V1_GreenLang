# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - DMA Report Engine Tests
========================================================================

Unit tests for DMAReportEngine (Engine 8) covering report assembly,
executive summary generation, methodology section generation, report
comparison, completeness validation, report export, section access,
ordering, statistics, and provenance hashing.

_compute_hash excludes "generated_at" (NOT "calculated_at"),
"processing_time_ms", and "provenance_hash".

Target: 40+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the dma_report engine module."""
    return _load_engine("dma_report")


@pytest.fixture
def engine(mod):
    """Create a fresh DMAReportEngine instance."""
    return mod.DMAReportEngine()


@pytest.fixture
def sample_report_input(mod):
    """Create a sample ReportAssemblyInput for testing.

    Uses the actual model fields:
        company_name, reporting_period, methodology, impact_results,
        financial_results, matrix_data, stakeholder_summary, iro_entries,
        esrs_mapping, gap_analysis, material_topic_ids, non_material_topic_ids.
    """
    return mod.ReportAssemblyInput(
        company_name="NordTech Industries GmbH",
        reporting_period="FY2025",
        methodology=mod.DMAMethodology(
            scoring_approach="ABSOLUTE_CUTOFF",
            stakeholder_methods=["surveys", "interviews", "workshops"],
            data_sources=["ERP system", "supplier questionnaires", "public databases"],
            assessment_date="2025-06-15",
            assessor_name="Sustainability Team",
            review_date="2025-07-01",
            reviewer_name="External Auditor",
        ),
        impact_results={
            "E1": {"score": 4.5, "passes": True},
            "S1": {"score": 3.8, "passes": True},
        },
        financial_results={
            "E1": {"score": 4.2, "passes": True},
            "S1": {"score": 2.5, "passes": False},
        },
        matrix_data={
            "total_matters": 10,
            "material_count": 6,
            "double_material_count": 3,
            "impact_only_count": 2,
            "financial_only_count": 1,
            "not_material_count": 4,
            "impact_threshold": 3.0,
            "financial_threshold": 3.0,
        },
        stakeholder_summary=mod.StakeholderSummary(
            stakeholder_groups=["Employees", "Investors", "Customers",
                                "Suppliers", "Regulators", "NGOs",
                                "Local Communities", "Industry Peers"],
            total_consulted=120,
            engagement_methods=["surveys", "interviews", "workshops"],
            key_concerns=["climate change", "supply chain transparency"],
            period="Q1-Q2 2025",
        ),
        iro_entries=[
            mod.IROEntry(
                iro_type="impact",
                esrs_topic="E1",
                matter_name="GHG emissions reduction",
                description="Reduction of Scope 1 and 2 GHG emissions",
                time_horizon="medium_term",
                severity=Decimal("4.5"),
                is_material=True,
            ),
            mod.IROEntry(
                iro_type="risk",
                esrs_topic="E1",
                matter_name="Carbon pricing risk",
                description="Financial exposure from carbon pricing mechanisms",
                time_horizon="short_term",
                likelihood=Decimal("3.8"),
                is_material=True,
            ),
            mod.IROEntry(
                iro_type="opportunity",
                esrs_topic="E1",
                matter_name="Clean technology opportunity",
                description="Revenue growth from green products",
                time_horizon="long_term",
                likelihood=Decimal("3.2"),
                is_material=False,
            ),
        ],
        esrs_mapping={
            "total_disclosures": 45,
            "total_data_points": 180,
            "mapped_topics": ["E1", "S1"],
        },
        gap_analysis={
            "total_disclosures": 45,
            "fully_covered": 32,
            "partially_covered": 8,
            "not_covered": 5,
            "total_estimated_effort_hours": 320,
        },
        material_topic_ids=["E1", "S1", "S2", "G1", "E4", "E5"],
        non_material_topic_ids=["E2", "E3", "S3", "S4"],
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestDMAReportEnums:
    """Tests for DMA report enums."""

    def test_report_format_count(self, mod):
        """ReportFormat has 4 values."""
        assert len(mod.ReportFormat) == 4
        names = {m.name for m in mod.ReportFormat}
        expected = {"JSON", "HTML", "XBRL", "PDF"}
        assert names == expected

    def test_section_type_count(self, mod):
        """SectionType has 10 values."""
        assert len(mod.SectionType) == 10


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestDMAReportConstants:
    """Tests for DMA report constants."""

    def test_report_sections_order(self, mod):
        """REPORT_SECTIONS_ORDER has 10 sections."""
        assert len(mod.REPORT_SECTIONS_ORDER) == 10

    def test_section_templates_count(self, mod):
        """SECTION_TEMPLATES has at least 5 templates."""
        assert len(mod.SECTION_TEMPLATES) >= 5


# ===========================================================================
# Pydantic Model Tests
# ===========================================================================


class TestDMAReportModels:
    """Tests for DMA report Pydantic models."""

    def test_dma_methodology_model(self, mod):
        """DMAMethodology model can be created with correct fields."""
        methodology = mod.DMAMethodology(
            scoring_approach="ABSOLUTE_CUTOFF",
            stakeholder_methods=["surveys", "interviews"],
            data_sources=["ERP", "public databases"],
            assessment_date="2025-06-15",
            assessor_name="Sustainability Team",
            review_date="2025-07-01",
            reviewer_name="External Auditor",
        )
        assert methodology.scoring_approach == "ABSOLUTE_CUTOFF"
        assert len(methodology.stakeholder_methods) == 2
        assert methodology.assessor_name == "Sustainability Team"

    def test_dma_section_model(self, mod):
        """DMASection model can be created with correct fields."""
        section = mod.DMASection(
            title="Executive Summary",
            content_type=mod.SectionType.EXECUTIVE_SUMMARY,
            narrative="Test content",
            order=1,
        )
        assert section.title == "Executive Summary"
        assert section.content_type == mod.SectionType.EXECUTIVE_SUMMARY

    def test_report_assembly_input_model(self, sample_report_input):
        """ReportAssemblyInput creates successfully."""
        assert sample_report_input.company_name == "NordTech Industries GmbH"
        assert sample_report_input.reporting_period == "FY2025"


# ===========================================================================
# Assemble Report Tests
# ===========================================================================


class TestAssembleReport:
    """Tests for assemble_report method."""

    def test_assemble_report_basic(self, engine, sample_report_input):
        """assemble_report returns a DMAReport."""
        report = engine.assemble_report(sample_report_input)
        assert report is not None
        assert hasattr(report, "sections")
        assert hasattr(report, "provenance_hash")

    def test_assemble_report_has_10_sections(self, engine, sample_report_input):
        """Assembled report has 10 sections."""
        report = engine.assemble_report(sample_report_input)
        assert len(report.sections) == 10

    def test_assemble_report_provenance_hash(self, engine, sample_report_input):
        """Report has a 64-character provenance hash."""
        report = engine.assemble_report(sample_report_input)
        assert len(report.provenance_hash) == 64
        int(report.provenance_hash, 16)

    def test_assemble_report_company_name(self, engine, sample_report_input):
        """Report preserves company name."""
        report = engine.assemble_report(sample_report_input)
        assert report.company_name == "NordTech Industries GmbH"

    def test_assemble_report_reporting_period(self, engine, sample_report_input):
        """Report preserves reporting period."""
        report = engine.assemble_report(sample_report_input)
        assert report.reporting_period == "FY2025"


# ===========================================================================
# Executive Summary Tests
# ===========================================================================


class TestExecutiveSummary:
    """Tests for generate_executive_summary method."""

    def test_executive_summary_generated(self, engine, sample_report_input):
        """Executive summary is generated as a string."""
        summary = engine.generate_executive_summary(sample_report_input)
        assert isinstance(summary, str)
        assert len(summary) > 100

    def test_executive_summary_contains_company_name(self, engine, sample_report_input):
        """Executive summary includes company name."""
        summary = engine.generate_executive_summary(sample_report_input)
        assert "NordTech" in summary

    def test_executive_summary_mentions_material_topics(self, engine, sample_report_input):
        """Executive summary references material topics."""
        summary = engine.generate_executive_summary(sample_report_input)
        has_topics = (
            "material" in summary.lower()
            or "Climate" in summary
            or "topic" in summary.lower()
        )
        assert has_topics


# ===========================================================================
# Methodology Section Tests
# ===========================================================================


class TestMethodologySection:
    """Tests for generate_methodology_section method."""

    def test_methodology_section_generated(self, engine, sample_report_input):
        """Methodology section is generated as a DMASection."""
        section = engine.generate_methodology_section(sample_report_input.methodology)
        assert section is not None
        assert hasattr(section, "narrative")
        assert isinstance(section.narrative, str)
        assert len(section.narrative) > 50

    def test_methodology_section_references_scoring(self, engine, sample_report_input):
        """Methodology section references the scoring approach."""
        section = engine.generate_methodology_section(sample_report_input.methodology)
        has_scoring = (
            "ABSOLUTE_CUTOFF" in section.narrative
            or "scoring" in section.narrative.lower()
            or "Scoring" in section.narrative
        )
        assert has_scoring


# ===========================================================================
# Compare Reports Tests
# ===========================================================================


class TestCompareReports:
    """Tests for compare_reports method."""

    def test_compare_identical_reports(self, engine, sample_report_input):
        """Comparing identical reports shows no changes."""
        r1 = engine.assemble_report(sample_report_input)
        r2 = engine.assemble_report(sample_report_input)
        delta = engine.compare_reports(r1, r2)
        assert delta is not None
        assert len(delta.new_material) == 0
        assert len(delta.no_longer_material) == 0

    def test_compare_different_reports(self, engine, mod):
        """Different reports show changes."""
        input1 = mod.ReportAssemblyInput(
            company_name="Corp A",
            reporting_period="FY2024",
            methodology=mod.DMAMethodology(
                scoring_approach="ABSOLUTE_CUTOFF",
                stakeholder_methods=["surveys"],
                data_sources=["ERP"],
                assessment_date="2024-06-15",
                assessor_name="Team A",
            ),
            matrix_data={
                "total_matters": 1, "material_count": 1,
                "double_material_count": 1,
                "impact_only_count": 0, "financial_only_count": 0,
                "not_material_count": 0, "impact_threshold": 3.0,
                "financial_threshold": 3.0,
            },
            esrs_mapping={"total_disclosures": 10, "total_data_points": 40},
            gap_analysis={
                "total_disclosures": 10, "fully_covered": 8,
                "partially_covered": 1, "not_covered": 1,
                "total_estimated_effort_hours": 40,
            },
            material_topic_ids=["E1"],
            non_material_topic_ids=["S1"],
        )
        input2 = mod.ReportAssemblyInput(
            company_name="Corp A",
            reporting_period="FY2025",
            methodology=mod.DMAMethodology(
                scoring_approach="ABSOLUTE_CUTOFF",
                stakeholder_methods=["surveys"],
                data_sources=["ERP"],
                assessment_date="2025-06-15",
                assessor_name="Team A",
            ),
            matrix_data={
                "total_matters": 1, "material_count": 1,
                "double_material_count": 1,
                "impact_only_count": 0, "financial_only_count": 0,
                "not_material_count": 0, "impact_threshold": 3.0,
                "financial_threshold": 3.0,
            },
            esrs_mapping={"total_disclosures": 10, "total_data_points": 40},
            gap_analysis={
                "total_disclosures": 10, "fully_covered": 8,
                "partially_covered": 1, "not_covered": 1,
                "total_estimated_effort_hours": 40,
            },
            material_topic_ids=["S1"],
            non_material_topic_ids=["E1"],
        )
        r1 = engine.assemble_report(input1)
        r2 = engine.assemble_report(input2)
        delta = engine.compare_reports(r2, r1)
        # E1 was material in r1 but not in r2, S1 is new
        assert len(delta.new_material) >= 1 or len(delta.no_longer_material) >= 1


# ===========================================================================
# Validate Completeness Tests
# ===========================================================================


class TestValidateCompleteness:
    """Tests for validate_completeness method."""

    def test_validate_complete_report(self, engine, sample_report_input):
        """Complete report has few warnings (methodology is fully populated)."""
        report = engine.assemble_report(sample_report_input)
        warnings = engine.validate_completeness(report)
        assert isinstance(warnings, list)
        # A well-assembled report should have few or no warnings
        assert len(warnings) <= 3

    def test_validate_incomplete_report(self, engine, mod):
        """Report missing sections produces warnings."""
        report = mod.DMAReport(
            company_name="Incomplete Corp",
            reporting_period="FY2025",
            sections=[],  # No sections
        )
        warnings = engine.validate_completeness(report)
        assert len(warnings) >= 1


# ===========================================================================
# Export Report Tests
# ===========================================================================


class TestExportReport:
    """Tests for export_report method."""

    def test_export_json(self, engine, sample_report_input, mod):
        """Export report as JSON returns bytes."""
        report = engine.assemble_report(sample_report_input)
        exported = engine.export_report(report, mod.ReportFormat.JSON)
        assert isinstance(exported, bytes)

    def test_export_html(self, engine, sample_report_input, mod):
        """Export report as HTML returns bytes."""
        report = engine.assemble_report(sample_report_input)
        exported = engine.export_report(report, mod.ReportFormat.HTML)
        assert isinstance(exported, bytes)
        assert b"<html>" in exported

    def test_export_xbrl(self, engine, sample_report_input, mod):
        """Export report as XBRL returns bytes."""
        report = engine.assemble_report(sample_report_input)
        exported = engine.export_report(report, mod.ReportFormat.XBRL)
        assert isinstance(exported, bytes)

    def test_export_pdf_placeholder(self, engine, sample_report_input, mod):
        """Export report as PDF (placeholder) returns bytes."""
        report = engine.assemble_report(sample_report_input)
        exported = engine.export_report(report, mod.ReportFormat.PDF)
        assert exported is not None
        assert isinstance(exported, bytes)


# ===========================================================================
# Section Access Tests
# ===========================================================================


class TestSectionAccess:
    """Tests for section access methods."""

    def test_get_section_by_type(self, engine, sample_report_input, mod):
        """get_section_by_type returns the correct section."""
        report = engine.assemble_report(sample_report_input)
        section = engine.get_section_by_type(
            report, mod.SectionType.EXECUTIVE_SUMMARY,
        )
        assert section is not None
        assert section.content_type == mod.SectionType.EXECUTIVE_SUMMARY

    def test_get_sections_ordered(self, engine, sample_report_input):
        """get_sections_ordered returns sections in standard order."""
        report = engine.assemble_report(sample_report_input)
        ordered = engine.get_sections_ordered(report)
        assert isinstance(ordered, list)
        assert len(ordered) == 10


# ===========================================================================
# Report Statistics Tests
# ===========================================================================


class TestReportStatistics:
    """Tests for get_report_statistics method."""

    def test_report_statistics_returned(self, engine, sample_report_input):
        """get_report_statistics returns a dictionary."""
        report = engine.assemble_report(sample_report_input)
        stats = engine.get_report_statistics(report)
        assert isinstance(stats, dict)

    def test_report_statistics_section_count(self, engine, sample_report_input):
        """Statistics include section count."""
        report = engine.assemble_report(sample_report_input)
        stats = engine.get_report_statistics(report)
        has_sections = (
            "section_count" in stats
            or "sections" in str(stats).lower()
            or "total_sections" in stats
        )
        assert has_sections


# ===========================================================================
# Provenance Hash Tests
# ===========================================================================


class TestDMAReportProvenanceHash:
    """Tests for DMA report provenance hashing."""

    def test_hash_is_64_chars(self, engine, sample_report_input):
        """Provenance hash is a 64-character SHA-256 hex string."""
        report = engine.assemble_report(sample_report_input)
        assert len(report.provenance_hash) == 64
        int(report.provenance_hash, 16)

    def test_hash_is_valid_hex(self, engine, sample_report_input):
        """Provenance hash is a valid hexadecimal SHA-256 string."""
        report = engine.assemble_report(sample_report_input)
        # Verify it is a valid hex string (no ValueError on conversion)
        hash_int = int(report.provenance_hash, 16)
        assert hash_int > 0

    def test_hash_changes_with_different_input(self, engine, mod):
        """Different inputs produce different provenance hashes."""
        input1 = mod.ReportAssemblyInput(
            company_name="Corp Alpha",
            reporting_period="FY2025",
            matrix_data={"total_matters": 5},
            material_topic_ids=["E1"],
        )
        input2 = mod.ReportAssemblyInput(
            company_name="Corp Beta",
            reporting_period="FY2025",
            matrix_data={"total_matters": 10},
            material_topic_ids=["E1", "S1"],
        )
        r1 = engine.assemble_report(input1)
        r2 = engine.assemble_report(input2)
        # Different company names and data should produce different hashes
        assert r1.provenance_hash != r2.provenance_hash


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestDMAReportEdgeCases:
    """Edge case tests for DMA report engine."""

    def test_report_with_no_material_topics(self, engine, mod):
        """Report can be assembled with no material topics."""
        input_data = mod.ReportAssemblyInput(
            company_name="Empty Corp",
            reporting_period="FY2025",
            methodology=mod.DMAMethodology(
                scoring_approach="ABSOLUTE_CUTOFF",
                stakeholder_methods=["surveys"],
                data_sources=["ERP"],
                assessment_date="2025-06-15",
                assessor_name="Team",
            ),
            matrix_data={
                "total_matters": 5, "material_count": 0,
                "double_material_count": 0,
                "impact_only_count": 0, "financial_only_count": 0,
                "not_material_count": 5, "impact_threshold": 3.0,
                "financial_threshold": 3.0,
            },
            esrs_mapping={"total_disclosures": 10, "total_data_points": 0},
            gap_analysis={
                "total_disclosures": 10, "fully_covered": 0,
                "partially_covered": 0, "not_covered": 10,
                "total_estimated_effort_hours": 500,
            },
            material_topic_ids=[],
            non_material_topic_ids=["E1", "E2", "E3", "S1", "G1"],
        )
        report = engine.assemble_report(input_data)
        assert report is not None
        assert len(report.sections) == 10
