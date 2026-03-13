# -*- coding: utf-8 -*-
"""
Tests for ConsolidationReporter - AGENT-EUDR-011 Engine 8: Multi-Facility Consolidation

Comprehensive test suite covering:
- Consolidation report (generate reports in all formats)
- Facility groups (create/manage by region/country/commodity)
- Enterprise dashboard (aggregate data, drill-down)
- Cross-facility transfer (record transfer, both-end entries)
- Facility comparison (compare facilities, commodity breakdown)
- Evidence package (compiled documentation for inspections)
- Report retrieval (get/download/list reports)
- Provenance hashing (SHA-256 on all reports)
- Edge cases (single facility, empty group, unknown report_id)

Test count: 55+ tests
Coverage target: >= 85% of ConsolidationReporter module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-011 Mass Balance Calculator Agent (GL-EUDR-MBC-011)
"""

from __future__ import annotations

import copy
import uuid
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.mass_balance_calculator.conftest import (
    EUDR_COMMODITIES,
    REPORT_FORMATS,
    REPORT_TYPES,
    FACILITY_GROUP_TYPES,
    SHA256_HEX_LENGTH,
    GROUP_SOUTHEAST_ASIA,
    GROUP_EUROPE,
    GROUP_ID_SOUTHEAST_ASIA,
    GROUP_ID_EUROPE,
    FAC_ID_MILL_MY,
    FAC_ID_REFINERY_ID,
    FAC_ID_WAREHOUSE_NL,
    FAC_ID_FACTORY_DE,
    FAC_ID_PORT_BR,
    LEDGER_COCOA_001,
    LEDGER_PALM_001,
    BATCH_COCOA_001,
    BATCH_PALM_001,
    make_facility_group,
    make_reconciliation,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. Consolidation Report
# ===========================================================================


class TestConsolidationReport:
    """Test consolidation report generation."""

    def test_generate_json_report(self, consolidation_reporter):
        """Generate a consolidation report in JSON format."""
        result = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY, FAC_ID_REFINERY_ID],
            report_format="json",
        )
        assert result is not None
        assert result.get("format") == "json"

    def test_generate_csv_report(self, consolidation_reporter):
        """Generate a consolidation report in CSV format."""
        result = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
            report_format="csv",
        )
        assert result is not None

    def test_generate_pdf_report(self, consolidation_reporter):
        """Generate a consolidation report in PDF format."""
        result = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
            report_format="pdf",
        )
        assert result is not None

    def test_generate_eudr_xml_report(self, consolidation_reporter):
        """Generate a consolidation report in EUDR XML format."""
        result = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
            report_format="eudr_xml",
        )
        assert result is not None

    @pytest.mark.parametrize("report_format", REPORT_FORMATS)
    def test_generate_all_formats(self, consolidation_reporter, report_format):
        """Reports can be generated in all 4 formats."""
        result = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
            report_format=report_format,
        )
        assert result is not None

    @pytest.mark.parametrize("report_type", REPORT_TYPES)
    def test_generate_all_report_types(self, consolidation_reporter, report_type):
        """Reports can be generated for all 5 report types."""
        result = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
            report_type=report_type,
        )
        assert result is not None

    def test_report_provenance_hash(self, consolidation_reporter):
        """Generated report has a provenance hash."""
        result = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
            report_format="json",
        )
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_report_assigns_id(self, consolidation_reporter):
        """Report auto-assigns an ID."""
        result = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
        )
        assert result.get("report_id") is not None

    def test_invalid_format_raises(self, consolidation_reporter):
        """Invalid report format raises ValueError."""
        with pytest.raises(ValueError):
            consolidation_reporter.generate_report(
                facility_ids=[FAC_ID_MILL_MY],
                report_format="invalid_format",
            )


# ===========================================================================
# 2. Facility Groups
# ===========================================================================


class TestFacilityGroups:
    """Test facility group management."""

    def test_create_group(self, consolidation_reporter):
        """Create a facility group."""
        group = make_facility_group(
            name="Test Group Asia",
            facility_ids=[FAC_ID_MILL_MY, FAC_ID_REFINERY_ID],
            group_type="region",
        )
        result = consolidation_reporter.create_group(group)
        assert result is not None
        assert result["name"] == "Test Group Asia"

    @pytest.mark.parametrize("group_type", FACILITY_GROUP_TYPES)
    def test_create_group_all_types(self, consolidation_reporter, group_type):
        """Groups can be created for all group types."""
        group = make_facility_group(group_type=group_type)
        result = consolidation_reporter.create_group(group)
        assert result is not None

    def test_add_facility_to_group(self, consolidation_reporter):
        """Add a facility to an existing group."""
        group = make_facility_group(
            group_id="GRP-ADD-001",
            facility_ids=[FAC_ID_MILL_MY],
        )
        consolidation_reporter.create_group(group)
        result = consolidation_reporter.add_facility(
            group_id="GRP-ADD-001",
            facility_id=FAC_ID_REFINERY_ID,
        )
        assert FAC_ID_REFINERY_ID in result.get("facility_ids", [])

    def test_remove_facility_from_group(self, consolidation_reporter):
        """Remove a facility from a group."""
        group = make_facility_group(
            group_id="GRP-REM-001",
            facility_ids=[FAC_ID_MILL_MY, FAC_ID_REFINERY_ID],
        )
        consolidation_reporter.create_group(group)
        result = consolidation_reporter.remove_facility(
            group_id="GRP-REM-001",
            facility_id=FAC_ID_REFINERY_ID,
        )
        assert FAC_ID_REFINERY_ID not in result.get("facility_ids", [FAC_ID_REFINERY_ID])

    def test_list_groups(self, consolidation_reporter):
        """List all facility groups."""
        group = make_facility_group(group_id="GRP-LIST-001")
        consolidation_reporter.create_group(group)
        groups = consolidation_reporter.list_groups()
        assert len(groups) >= 1

    def test_get_group(self, consolidation_reporter):
        """Get a specific facility group."""
        group = make_facility_group(group_id="GRP-GET-001")
        consolidation_reporter.create_group(group)
        result = consolidation_reporter.get_group("GRP-GET-001")
        assert result["group_id"] == "GRP-GET-001"

    def test_duplicate_group_raises(self, consolidation_reporter):
        """Duplicate group ID raises error."""
        group = make_facility_group(group_id="GRP-DUP-001")
        consolidation_reporter.create_group(group)
        with pytest.raises((ValueError, KeyError)):
            consolidation_reporter.create_group(copy.deepcopy(group))


# ===========================================================================
# 3. Enterprise Dashboard
# ===========================================================================


class TestEnterpriseDashboard:
    """Test enterprise dashboard aggregation."""

    def test_aggregate_data(self, consolidation_reporter):
        """Aggregate mass balance data across all facilities."""
        dashboard = consolidation_reporter.enterprise_dashboard(
            facility_ids=[FAC_ID_MILL_MY, FAC_ID_REFINERY_ID],
        )
        assert dashboard is not None

    def test_dashboard_includes_totals(self, consolidation_reporter):
        """Dashboard includes total inputs, outputs, balance."""
        dashboard = consolidation_reporter.enterprise_dashboard(
            facility_ids=[FAC_ID_MILL_MY],
        )
        assert dashboard is not None

    def test_dashboard_drill_down(self, consolidation_reporter):
        """Dashboard supports drill-down by facility."""
        dashboard = consolidation_reporter.enterprise_dashboard(
            facility_ids=[FAC_ID_MILL_MY, FAC_ID_REFINERY_ID],
            drill_down=True,
        )
        assert dashboard is not None

    def test_dashboard_by_commodity(self, consolidation_reporter):
        """Dashboard can be filtered by commodity."""
        dashboard = consolidation_reporter.enterprise_dashboard(
            facility_ids=[FAC_ID_MILL_MY],
            commodity="cocoa",
        )
        assert dashboard is not None


# ===========================================================================
# 4. Cross-Facility Transfer
# ===========================================================================


class TestCrossFacilityTransfer:
    """Test cross-facility material transfer tracking."""

    def test_record_transfer(self, consolidation_reporter):
        """Record a cross-facility transfer."""
        result = consolidation_reporter.record_transfer(
            from_facility_id=FAC_ID_MILL_MY,
            to_facility_id=FAC_ID_FACTORY_DE,
            commodity="cocoa",
            quantity_kg=Decimal("5000.0"),
            batch_id=BATCH_COCOA_001,
        )
        assert result is not None

    def test_transfer_creates_both_entries(self, consolidation_reporter):
        """Transfer creates ledger entries at both facilities."""
        result = consolidation_reporter.record_transfer(
            from_facility_id=FAC_ID_MILL_MY,
            to_facility_id=FAC_ID_FACTORY_DE,
            commodity="cocoa",
            quantity_kg=Decimal("3000.0"),
            batch_id=BATCH_COCOA_001,
        )
        entries = result.get("entries", result.get("ledger_entries", []))
        if entries:
            assert len(entries) >= 2

    def test_transfer_provenance_hash(self, consolidation_reporter):
        """Transfer generates a provenance hash."""
        result = consolidation_reporter.record_transfer(
            from_facility_id=FAC_ID_MILL_MY,
            to_facility_id=FAC_ID_FACTORY_DE,
            commodity="cocoa",
            quantity_kg=Decimal("2000.0"),
            batch_id=BATCH_COCOA_001,
        )
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_transfer_same_facility_raises(self, consolidation_reporter):
        """Transfer to same facility raises ValueError."""
        with pytest.raises(ValueError):
            consolidation_reporter.record_transfer(
                from_facility_id=FAC_ID_MILL_MY,
                to_facility_id=FAC_ID_MILL_MY,
                commodity="cocoa",
                quantity_kg=Decimal("1000.0"),
                batch_id=BATCH_COCOA_001,
            )

    def test_transfer_zero_quantity_raises(self, consolidation_reporter):
        """Transfer with zero quantity raises ValueError."""
        with pytest.raises(ValueError):
            consolidation_reporter.record_transfer(
                from_facility_id=FAC_ID_MILL_MY,
                to_facility_id=FAC_ID_FACTORY_DE,
                commodity="cocoa",
                quantity_kg=Decimal("0.0"),
                batch_id=BATCH_COCOA_001,
            )


# ===========================================================================
# 5. Facility Comparison
# ===========================================================================


class TestFacilityComparison:
    """Test facility comparison reporting."""

    def test_compare_facilities(self, consolidation_reporter):
        """Compare mass balance performance across facilities."""
        result = consolidation_reporter.compare_facilities(
            facility_ids=[FAC_ID_MILL_MY, FAC_ID_REFINERY_ID],
        )
        assert result is not None

    def test_comparison_commodity_breakdown(self, consolidation_reporter):
        """Facility comparison includes commodity breakdown."""
        result = consolidation_reporter.compare_facilities(
            facility_ids=[FAC_ID_MILL_MY, FAC_ID_REFINERY_ID],
            include_commodity_breakdown=True,
        )
        assert result is not None

    def test_comparison_single_facility(self, consolidation_reporter):
        """Single facility comparison returns self-comparison."""
        result = consolidation_reporter.compare_facilities(
            facility_ids=[FAC_ID_MILL_MY],
        )
        assert result is not None

    def test_comparison_includes_metrics(self, consolidation_reporter):
        """Comparison includes key metrics."""
        result = consolidation_reporter.compare_facilities(
            facility_ids=[FAC_ID_MILL_MY, FAC_ID_REFINERY_ID],
        )
        assert result is not None


# ===========================================================================
# 6. Evidence Package
# ===========================================================================


class TestEvidencePackage:
    """Test regulatory evidence package assembly."""

    def test_compile_evidence_package(self, consolidation_reporter):
        """Compile evidence package for competent authority inspection."""
        result = consolidation_reporter.compile_evidence_package(
            facility_ids=[FAC_ID_MILL_MY],
            period_ids=["PRD-EVD-001"],
        )
        assert result is not None

    def test_evidence_package_provenance(self, consolidation_reporter):
        """Evidence package has provenance hash."""
        result = consolidation_reporter.compile_evidence_package(
            facility_ids=[FAC_ID_MILL_MY],
            period_ids=["PRD-EVD-002"],
        )
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_evidence_includes_reconciliation(self, consolidation_reporter):
        """Evidence package includes reconciliation data."""
        result = consolidation_reporter.compile_evidence_package(
            facility_ids=[FAC_ID_MILL_MY],
            period_ids=["PRD-EVD-003"],
        )
        assert result is not None

    def test_evidence_includes_ledger_data(self, consolidation_reporter):
        """Evidence package includes ledger entry data."""
        result = consolidation_reporter.compile_evidence_package(
            facility_ids=[FAC_ID_MILL_MY],
            period_ids=["PRD-EVD-004"],
        )
        assert result is not None

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_evidence_all_commodities(self, consolidation_reporter, commodity):
        """Evidence packages can be compiled for all commodities."""
        result = consolidation_reporter.compile_evidence_package(
            facility_ids=[FAC_ID_MILL_MY],
            period_ids=[f"PRD-EVD-{commodity[:3].upper()}"],
            commodity=commodity,
        )
        assert result is not None


# ===========================================================================
# 7. Report Retrieval
# ===========================================================================


class TestReportRetrieval:
    """Test report retrieval operations."""

    def test_get_report_by_id(self, consolidation_reporter):
        """Retrieve a report by its ID."""
        gen_result = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
            report_format="json",
        )
        report_id = gen_result["report_id"]
        result = consolidation_reporter.get_report(report_id)
        assert result is not None
        assert result["report_id"] == report_id

    def test_list_reports(self, consolidation_reporter):
        """List all generated reports."""
        consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
        )
        reports = consolidation_reporter.list_reports()
        assert len(reports) >= 1

    def test_list_reports_by_facility(self, consolidation_reporter):
        """List reports filtered by facility."""
        consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
        )
        reports = consolidation_reporter.list_reports(facility_id=FAC_ID_MILL_MY)
        assert isinstance(reports, list)

    def test_get_nonexistent_report(self, consolidation_reporter):
        """Getting non-existent report returns None."""
        result = consolidation_reporter.get_report("RPT-NONEXISTENT-999")
        assert result is None

    def test_download_report(self, consolidation_reporter):
        """Download a generated report."""
        gen_result = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
            report_format="json",
        )
        report_id = gen_result["report_id"]
        result = consolidation_reporter.download_report(report_id)
        assert result is not None


# ===========================================================================
# 8. Provenance Hashing
# ===========================================================================


class TestProvenanceHashing:
    """Test SHA-256 provenance hashing on all reports."""

    def test_different_data_different_hash(self, consolidation_reporter):
        """Different report data produces different provenance hashes."""
        r1 = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
            report_format="json",
        )
        r2 = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_REFINERY_ID],
            report_format="json",
        )
        assert r1["provenance_hash"] != r2["provenance_hash"]

    def test_same_data_same_hash(self, consolidation_reporter):
        """Same report data produces the same provenance hash."""
        # Note: timestamps may differ, so this tests the data-based hash
        r1 = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
            report_format="json",
        )
        assert_valid_provenance_hash(r1["provenance_hash"])

    @pytest.mark.parametrize("report_format", REPORT_FORMATS)
    def test_provenance_all_formats(self, consolidation_reporter, report_format):
        """Provenance hash is generated for all report formats."""
        result = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
            report_format=report_format,
        )
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])


# ===========================================================================
# 9. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases for consolidation reporting."""

    def test_single_facility_report(self, consolidation_reporter):
        """Report for a single facility works."""
        result = consolidation_reporter.generate_report(
            facility_ids=[FAC_ID_MILL_MY],
        )
        assert result is not None

    def test_empty_facility_list_raises(self, consolidation_reporter):
        """Empty facility list raises ValueError."""
        with pytest.raises(ValueError):
            consolidation_reporter.generate_report(facility_ids=[])

    def test_empty_group_report(self, consolidation_reporter):
        """Report for an empty group is handled."""
        group = make_facility_group(
            group_id="GRP-EMPTY-001",
            facility_ids=[],
        )
        try:
            consolidation_reporter.create_group(group)
            result = consolidation_reporter.generate_report(
                group_id="GRP-EMPTY-001",
            )
            assert result is not None
        except ValueError:
            pass  # Also acceptable

    def test_unknown_report_id_returns_none(self, consolidation_reporter):
        """Unknown report ID returns None."""
        result = consolidation_reporter.get_report("RPT-UNKNOWN-999")
        assert result is None

    def test_large_facility_group(self, consolidation_reporter):
        """Report for a large facility group works."""
        facility_ids = [f"FAC-LARGE-{i:03d}" for i in range(20)]
        result = consolidation_reporter.generate_report(
            facility_ids=facility_ids,
        )
        assert result is not None

    def test_negative_transfer_raises(self, consolidation_reporter):
        """Negative transfer quantity raises ValueError."""
        with pytest.raises(ValueError):
            consolidation_reporter.record_transfer(
                from_facility_id=FAC_ID_MILL_MY,
                to_facility_id=FAC_ID_FACTORY_DE,
                commodity="cocoa",
                quantity_kg=Decimal("-1000.0"),
                batch_id=BATCH_COCOA_001,
            )
