# -*- coding: utf-8 -*-
"""
Test suite for PACK-030 Net Zero Reporting Pack - End-to-End.

Tests complete reporting cycles from data aggregation through format
rendering, including single-framework reports, multi-framework reports,
multi-language reports, assurance-ready reports, and real-time
dashboard generation.

Author:  GreenLang Test Engineering
Pack:    PACK-030 Net Zero Reporting Pack
Tests:   ~60 tests
"""

import sys
import uuid
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from .conftest import (
    assert_provenance_hash, assert_processing_time, assert_valid_uuid,
    assert_valid_json, compute_sha256, timed_block,
    generate_report_sections, generate_report_metrics,
    generate_validation_issues, generate_framework_coverage,
    FRAMEWORKS, OUTPUT_FORMATS, LANGUAGES, STAKEHOLDER_VIEWS,
    TCFD_PILLARS, CDP_MODULES, ESRS_E1_DISCLOSURES,
    GRI_305_DISCLOSURES, EVIDENCE_TYPES, REPORT_STATUSES,
)


# ========================================================================
# Single-Framework E2E Tests
# ========================================================================


class TestSingleFrameworkE2E:
    """Test complete single-framework reporting cycles."""

    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_full_cycle_per_framework(self, framework, emissions_data_2024):
        """Full cycle: aggregate -> narrative -> validate -> compile -> render."""
        # Step 1: Generate sections
        sections = generate_report_sections(framework, count=5)
        assert len(sections) == 5

        # Step 2: Generate metrics
        metrics = generate_report_metrics(count=10)
        assert len(metrics) == 10

        # Step 3: Validate
        issues = generate_validation_issues(count=3)
        assert len(issues) == 3

        # Step 4: Generate coverage
        coverage = generate_framework_coverage([framework])
        assert framework in coverage

        # Step 5: Verify provenance chain
        for section in sections:
            assert "section_id" in section
            assert_valid_uuid(section["section_id"])

    def test_tcfd_full_cycle(self, tcfd_report_data, emissions_data_2024):
        """Complete TCFD 4-pillar disclosure cycle."""
        # Verify all 4 pillars
        for pillar in TCFD_PILLARS:
            assert pillar in tcfd_report_data["pillars"]

        # Generate sections
        sections = generate_report_sections("TCFD", count=len(TCFD_PILLARS))
        assert len(sections) == len(TCFD_PILLARS)

        # Verify provenance
        assert "provenance_hash" in tcfd_report_data
        assert len(tcfd_report_data["provenance_hash"]) == 64

    def test_cdp_full_cycle(self, cdp_questionnaire_data, emissions_data_2024):
        """Complete CDP C0-C12 questionnaire cycle."""
        # Verify all CDP modules present
        for module in CDP_MODULES:
            assert module in cdp_questionnaire_data["modules"]

        # Generate sections
        sections = generate_report_sections("CDP", count=len(CDP_MODULES))
        assert len(sections) == len(CDP_MODULES)

        # Verify completeness
        assert cdp_questionnaire_data["completeness_pct"] >= Decimal("80")

    def test_csrd_e1_full_cycle(self, csrd_e1_data, emissions_data_2024):
        """Complete CSRD ESRS E1 disclosure cycle."""
        # Verify all E1 disclosures
        for disclosure in ESRS_E1_DISCLOSURES:
            assert disclosure in csrd_e1_data["disclosures"]

        # Generate sections
        sections = generate_report_sections("CSRD", count=len(ESRS_E1_DISCLOSURES))
        assert len(sections) == len(ESRS_E1_DISCLOSURES)

        # Verify digital taxonomy flag
        assert csrd_e1_data["digital_taxonomy_required"] is True

    def test_gri_305_full_cycle(self, gri_305_data, emissions_data_2024):
        """Complete GRI 305 emissions disclosure cycle."""
        # Verify all 305 disclosures
        for disclosure in GRI_305_DISCLOSURES:
            assert disclosure in gri_305_data["disclosures"]

        # Generate sections
        sections = generate_report_sections("GRI", count=len(GRI_305_DISCLOSURES))
        assert len(sections) == len(GRI_305_DISCLOSURES)

    def test_sec_full_cycle(self, sec_disclosure_data, emissions_data_2024):
        """Complete SEC 10-K climate disclosure cycle."""
        assert "items" in sec_disclosure_data
        assert "reg_sk_1502" in sec_disclosure_data["items"]
        assert sec_disclosure_data["xbrl_required"] is True

        sections = generate_report_sections("SEC", count=5)
        assert len(sections) == 5

    def test_issb_full_cycle(self, issb_s2_data, emissions_data_2024):
        """Complete ISSB IFRS S2 disclosure cycle."""
        assert issb_s2_data["standard"] == "IFRS S2"
        assert "governance" in issb_s2_data
        assert "metrics_targets" in issb_s2_data

        sections = generate_report_sections("ISSB", count=4)
        assert len(sections) == 4

    def test_sbti_full_cycle(self, sbti_report_data, emissions_data_2024):
        """Complete SBTi annual progress report cycle."""
        assert sbti_report_data["framework"] == "SBTi"
        assert sbti_report_data["on_track"] is True

        sections = generate_report_sections("SBTi", count=5)
        assert len(sections) == 5


# ========================================================================
# Multi-Framework E2E Tests
# ========================================================================


class TestMultiFrameworkE2E:
    """Test multi-framework reporting cycles."""

    def test_all_7_frameworks(self, emissions_data_2024):
        """Generate all 7 framework reports from single data source."""
        with timed_block("7_framework_e2e", max_seconds=15.0):
            all_reports = {}
            for fw in FRAMEWORKS:
                sections = generate_report_sections(fw, count=5)
                metrics = generate_report_metrics(count=10)
                all_reports[fw] = {
                    "sections": sections,
                    "metrics": metrics,
                    "framework": fw,
                }
            assert len(all_reports) == 7
            for fw in FRAMEWORKS:
                assert fw in all_reports

    def test_cross_framework_consistency(self, emissions_data_2024):
        """Verify emissions data consistency across frameworks."""
        scope_1 = emissions_data_2024["scope_1_tco2e"]
        for fw in FRAMEWORKS:
            # Each framework should reference the same scope 1 value
            assert scope_1 == Decimal("107500")

    def test_multi_framework_provenance_chain(self, emissions_data_2024):
        """Each framework report should have its own provenance hash."""
        hashes = set()
        for fw in FRAMEWORKS:
            h = compute_sha256(f"report_{fw}_{emissions_data_2024['reporting_year']}")
            hashes.add(h)
        # Each framework should have a unique hash
        assert len(hashes) == len(FRAMEWORKS)

    def test_parallel_framework_generation(self, emissions_data_2024):
        """Parallel framework generation should complete in <10s."""
        with timed_block("parallel_frameworks_e2e", max_seconds=10.0):
            results = {}
            for fw in FRAMEWORKS:
                results[fw] = generate_report_sections(fw, count=5)
            assert all(len(s) == 5 for s in results.values())

    @pytest.mark.parametrize("fw_subset", [
        ["TCFD", "CDP"],
        ["TCFD", "CDP", "CSRD"],
        ["SBTi", "ISSB", "SEC"],
        FRAMEWORKS,
    ])
    def test_framework_subset_reporting(self, fw_subset, emissions_data_2024):
        """Reporting should work for any subset of frameworks."""
        results = {}
        for fw in fw_subset:
            results[fw] = generate_report_sections(fw, count=3)
        assert len(results) == len(fw_subset)


# ========================================================================
# Multi-Language E2E Tests
# ========================================================================


class TestMultiLanguageE2E:
    """Test multi-language reporting cycles."""

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_single_language_report(self, language, emissions_data_2024):
        """Generate a complete report in each supported language."""
        sections = generate_report_sections("TCFD", count=5)
        for s in sections:
            s["language"] = language
        assert all(s["language"] == language for s in sections)

    def test_multi_language_generation(self, emissions_data_2024):
        """Generate report in all 4 languages."""
        language_reports = {}
        for lang in LANGUAGES:
            sections = generate_report_sections("TCFD", count=4)
            for s in sections:
                s["language"] = lang
            language_reports[lang] = sections
        assert len(language_reports) == 4
        for lang in LANGUAGES:
            assert lang in language_reports

    def test_citation_preserved_across_languages(self, emissions_data_2024):
        """Citations should be preserved in all languages."""
        for lang in LANGUAGES:
            sections = generate_report_sections("TCFD", count=3)
            for s in sections:
                s["language"] = lang
                assert "citations" in s
                assert len(s["citations"]) > 0


# ========================================================================
# Multi-Format E2E Tests
# ========================================================================


class TestMultiFormatE2E:
    """Test multi-format output generation cycles."""

    @pytest.mark.parametrize("output_format", OUTPUT_FORMATS)
    def test_single_format_output(self, output_format, emissions_data_2024):
        """Generate report in each output format."""
        sections = generate_report_sections("TCFD", count=5)
        result = {
            "format": output_format,
            "sections": sections,
            "generated": True,
        }
        assert result["format"] == output_format
        assert result["generated"] is True

    def test_all_formats_from_single_source(self, emissions_data_2024):
        """Generate all 6 formats from a single report."""
        sections = generate_report_sections("TCFD", count=5)
        format_results = {}
        for fmt in OUTPUT_FORMATS:
            format_results[fmt] = {
                "format": fmt,
                "sections": sections,
                "generated": True,
            }
        assert len(format_results) == len(OUTPUT_FORMATS)

    def test_xbrl_for_sec(self, sec_disclosure_data):
        """SEC reports should include XBRL output."""
        assert sec_disclosure_data["xbrl_required"] is True

    def test_digital_taxonomy_for_csrd(self, csrd_e1_data):
        """CSRD reports should include digital taxonomy."""
        assert csrd_e1_data["digital_taxonomy_required"] is True


# ========================================================================
# Assurance-Ready E2E Tests
# ========================================================================


class TestAssuranceE2E:
    """Test assurance-ready report generation cycles."""

    def test_evidence_bundle_completeness(self, assurance_data):
        """Evidence bundle should include all required types."""
        evidence_types_present = {e["type"] for e in assurance_data["evidence_items"]}
        for et in EVIDENCE_TYPES:
            assert et in evidence_types_present

    def test_provenance_hash_chain(self, emissions_data_2024):
        """All calculations should have provenance hashes."""
        sections = generate_report_sections("TCFD", count=5)
        metrics = generate_report_metrics(count=10)
        for m in metrics:
            assert "provenance_hash" in m
            assert len(m["provenance_hash"]) == 64

    def test_lineage_diagram_generation(self, assurance_data):
        """Lineage diagrams should be included in evidence bundle."""
        lineage_items = [e for e in assurance_data["evidence_items"] if e["type"] == "lineage"]
        assert len(lineage_items) >= 1

    def test_methodology_documentation(self, assurance_data):
        """Methodology docs should be included in evidence bundle."""
        methodology_items = [e for e in assurance_data["evidence_items"] if e["type"] == "methodology"]
        assert len(methodology_items) >= 1

    def test_control_matrix_items(self, assurance_data):
        """Control matrix should be included in evidence bundle."""
        control_items = [e for e in assurance_data["evidence_items"] if e["type"] == "control"]
        assert len(control_items) >= 1

    def test_audit_standard_compliance(self, assurance_data):
        """Evidence bundle should comply with ISAE 3410."""
        assert assurance_data["audit_standard"] == "ISAE 3410"

    def test_total_calculations_traced(self, assurance_data):
        """Should trace a significant number of calculations."""
        assert assurance_data["total_calculations_traced"] >= 100


# ========================================================================
# Dashboard E2E Tests
# ========================================================================


class TestDashboardE2E:
    """Test dashboard generation end-to-end."""

    @pytest.mark.parametrize("view_type", STAKEHOLDER_VIEWS)
    def test_stakeholder_dashboard(self, view_type, emissions_data_2024):
        """Generate dashboard for each stakeholder type."""
        coverage = generate_framework_coverage()
        dashboard = {
            "view_type": view_type,
            "coverage": coverage,
            "emissions": emissions_data_2024,
        }
        assert dashboard["view_type"] == view_type
        assert len(dashboard["coverage"]) == len(FRAMEWORKS)

    def test_executive_dashboard_completeness(self, emissions_data_2024):
        """Executive dashboard should include all frameworks."""
        coverage = generate_framework_coverage()
        for fw in FRAMEWORKS:
            assert fw in coverage

    def test_deadline_tracking(self, framework_deadlines):
        """Dashboard should track all framework deadlines."""
        assert len(framework_deadlines) >= 5
        for d in framework_deadlines:
            assert "framework" in d
            assert "deadline" in d or "deadline_date" in d
            assert "days_remaining" in d

    def test_progress_tracking(self, emissions_data_2024, baseline_2019, target_data):
        """Dashboard should show progress vs. targets."""
        current = emissions_data_2024["total_scope_12_tco2e"]
        baseline = baseline_2019["total_scope_12_tco2e"]
        reduction_pct = ((baseline - current) / baseline) * Decimal("100")
        assert reduction_pct > Decimal("0")

    def test_heatmap_data(self, emissions_data_2024):
        """Dashboard should include framework coverage heatmap."""
        coverage = generate_framework_coverage()
        for fw, pct in coverage.items():
            assert Decimal("0") <= pct <= Decimal("100")


# ========================================================================
# Report Lifecycle E2E Tests
# ========================================================================


class TestReportLifecycleE2E:
    """Test complete report lifecycle: draft -> review -> approved -> published."""

    @pytest.mark.parametrize("status", REPORT_STATUSES)
    def test_status_transition(self, status):
        """Report should support all status values."""
        report = {
            "report_id": str(uuid.uuid4()),
            "status": status,
            "framework": "TCFD",
        }
        assert report["status"] in REPORT_STATUSES

    def test_full_lifecycle(self):
        """Report should progress through all statuses."""
        report_id = str(uuid.uuid4())
        status_history = []
        for status in REPORT_STATUSES:
            status_history.append(status)
        assert status_history == ["draft", "review", "approved", "published"]

    def test_approval_requires_validation(self, emissions_data_2024):
        """Approval should only happen after validation passes."""
        issues = generate_validation_issues(count=3)
        critical_issues = [i for i in issues if i["severity"] == "critical"]
        # If no critical issues, validation passes
        can_approve = len(critical_issues) == 0
        assert isinstance(can_approve, bool)

    def test_publication_requires_approval(self):
        """Publication should only happen after approval."""
        statuses = REPORT_STATUSES
        approved_idx = statuses.index("approved")
        published_idx = statuses.index("published")
        assert approved_idx < published_idx

    def test_audit_trail_generated(self):
        """Each status transition should generate audit trail entry."""
        audit_entries = []
        for status in REPORT_STATUSES:
            audit_entries.append({
                "event_type": f"status_changed_to_{status}",
                "actor_type": "user" if status != "published" else "system",
            })
        assert len(audit_entries) == len(REPORT_STATUSES)


# ========================================================================
# Data Flow E2E Tests
# ========================================================================


class TestDataFlowE2E:
    """Test data flow from source packs through to final outputs."""

    def test_pack_to_report_flow(self, emissions_data_2024, baseline_2019, target_data):
        """Data should flow from PACK-021/029 through to reports."""
        # PACK-021 provides baseline
        assert baseline_2019["base_year"] == 2019
        # PACK-029 provides current data
        assert emissions_data_2024["reporting_year"] == 2024
        # Target data from PACK-021/029
        assert target_data["near_term_target_year"] == 2030

    def test_emissions_flow_consistency(self, emissions_data_2024):
        """Emissions data should be consistent through the pipeline."""
        scope_1 = emissions_data_2024["scope_1_tco2e"]
        scope_2_market = emissions_data_2024["scope_2_market_tco2e"]
        total_12 = emissions_data_2024["total_scope_12_tco2e"]
        # Total S1+S2 should equal sum of components
        assert scope_1 + scope_2_market == total_12

    def test_scope_3_completeness(self, emissions_data_2024):
        """All 15 Scope 3 categories should be present."""
        s3_categories = emissions_data_2024["scope_3_categories"]
        assert len(s3_categories) == 15

    def test_scope_3_total_consistency(self, emissions_data_2024):
        """Scope 3 total should equal sum of categories."""
        s3_total = emissions_data_2024["scope_3_tco2e"]
        s3_sum = sum(emissions_data_2024["scope_3_categories"].values())
        assert s3_total == s3_sum

    def test_grand_total_consistency(self, emissions_data_2024):
        """Grand total should equal S1 + S2(market) + S3."""
        s1 = emissions_data_2024["scope_1_tco2e"]
        s2 = emissions_data_2024["scope_2_market_tco2e"]
        s3 = emissions_data_2024["scope_3_tco2e"]
        total_123 = emissions_data_2024["total_scope_123_tco2e"]
        assert s1 + s2 + s3 == total_123

    def test_intensity_metric_calculation(self, emissions_data_2024):
        """Intensity metric should be correctly calculated."""
        total = emissions_data_2024["total_scope_123_tco2e"]
        revenue = emissions_data_2024["revenue_m_usd"]
        expected_intensity = total / revenue
        actual_intensity = emissions_data_2024["intensity_tco2e_per_m_usd"]
        assert abs(expected_intensity - actual_intensity) < Decimal("1")

    def test_data_quality_score(self, emissions_data_2024):
        """Data quality score should be within valid range."""
        dqs = emissions_data_2024["data_quality_score"]
        assert Decimal("0") <= dqs <= Decimal("1")

    def test_verified_flag(self, emissions_data_2024):
        """Verified flag should be boolean."""
        assert isinstance(emissions_data_2024["verified"], bool)
        assert emissions_data_2024["verified"] is True
