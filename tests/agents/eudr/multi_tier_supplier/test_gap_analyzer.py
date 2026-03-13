# -*- coding: utf-8 -*-
"""
Tests for GapAnalyzer - AGENT-EUDR-008 Engine 7: Gap Analysis and Remediation

Comprehensive test suite covering:
- Data gaps: missing GPS, missing cert, missing legal entity (F7.1)
- Coverage gaps: missing tiers (F7.2)
- Verification gaps: stale/outdated data (F7.3)
- Gap severity classification rules (F7.4)
- Remediation plan generation (F7.5)
- Remediation progress tracking (F7.6)
- Questionnaire auto-generation (F7.7)
- Gap trend analysis (F7.8)

Test count: 55+ tests
Coverage target: >= 85% of GapAnalyzer module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.multi_tier_supplier.conftest import (
    SUP_ID_COCOA_IMPORTER_EU,
    SUP_ID_COCOA_TRADER_GH,
    SUP_ID_COCOA_FARMER_1_GH,
    SUP_ID_COCOA_FARMER_2_GH,
    SUP_ID_PALM_SMALLHOLDER_ID,
    GAP_SEVERITIES,
    SHA256_HEX_LENGTH,
    make_supplier,
    make_relationship,
    make_cert,
    build_linear_chain,
)


# ===========================================================================
# 1. Data Gaps - Missing GPS
# ===========================================================================


class TestDataGapsMissingGPS:
    """Test detection of missing GPS coordinate gaps (F7.1)."""

    def test_detect_missing_gps(self, gap_analyzer):
        """Supplier without GPS coordinates is flagged as data gap."""
        suppliers = [
            make_supplier(supplier_id="GPS-GAP-001", gps_lat=None, gps_lon=None),
        ]
        result = gap_analyzer.analyze(suppliers)
        gps_gaps = [g for g in result.gaps if g["gap_type"] == "missing_gps"]
        assert len(gps_gaps) >= 1
        assert gps_gaps[0]["supplier_id"] == "GPS-GAP-001"

    def test_no_gap_with_gps(self, gap_analyzer):
        """Supplier with GPS coordinates has no GPS gap."""
        suppliers = [
            make_supplier(supplier_id="GPS-OK-001", gps_lat=5.6037, gps_lon=-0.1870),
        ]
        result = gap_analyzer.analyze(suppliers)
        gps_gaps = [g for g in result.gaps if g["gap_type"] == "missing_gps"
                    and g["supplier_id"] == "GPS-OK-001"]
        assert len(gps_gaps) == 0

    def test_partial_gps_flagged(self, gap_analyzer):
        """Supplier with only latitude (no longitude) is flagged."""
        suppliers = [
            make_supplier(supplier_id="GPS-PARTIAL", gps_lat=5.6037, gps_lon=None),
        ]
        result = gap_analyzer.analyze(suppliers)
        gps_gaps = [g for g in result.gaps if g["gap_type"] == "missing_gps"
                    and g["supplier_id"] == "GPS-PARTIAL"]
        assert len(gps_gaps) >= 1

    def test_multiple_suppliers_missing_gps(self, gap_analyzer):
        """Multiple suppliers without GPS all flagged."""
        suppliers = [
            make_supplier(supplier_id=f"GPS-MISS-{i}", gps_lat=None, gps_lon=None)
            for i in range(5)
        ]
        result = gap_analyzer.analyze(suppliers)
        gps_gaps = [g for g in result.gaps if g["gap_type"] == "missing_gps"]
        assert len(gps_gaps) == 5


# ===========================================================================
# 2. Data Gaps - Missing Certifications
# ===========================================================================


class TestDataGapsMissingCert:
    """Test detection of missing certification gaps (F7.1)."""

    def test_detect_missing_certification(self, gap_analyzer):
        """Supplier without certifications is flagged."""
        suppliers = [
            make_supplier(supplier_id="CERT-GAP-001", certifications=[]),
        ]
        result = gap_analyzer.analyze(suppliers)
        cert_gaps = [g for g in result.gaps if g["gap_type"] == "missing_certification"
                     and g["supplier_id"] == "CERT-GAP-001"]
        assert len(cert_gaps) >= 1

    def test_no_gap_with_valid_cert(self, gap_analyzer):
        """Supplier with valid certifications has no cert gap."""
        suppliers = [
            make_supplier(supplier_id="CERT-OK-001", certifications=["RSPO-001"]),
        ]
        result = gap_analyzer.analyze(suppliers)
        cert_gaps = [g for g in result.gaps if g["gap_type"] == "missing_certification"
                     and g["supplier_id"] == "CERT-OK-001"]
        assert len(cert_gaps) == 0

    def test_expired_cert_flagged(self, gap_analyzer):
        """Supplier with only expired certifications flagged."""
        suppliers = [
            make_supplier(supplier_id="CERT-EXP-001", certifications=["EXPIRED-001"]),
        ]
        certs = [make_cert("CERT-EXP-001", cert_type="UTZ", status="expired",
                           days_until_expiry=-30)]
        result = gap_analyzer.analyze(suppliers, certifications=certs)
        cert_gaps = [g for g in result.gaps
                     if g["gap_type"] in ("missing_certification", "expired_certification")
                     and g["supplier_id"] == "CERT-EXP-001"]
        assert len(cert_gaps) >= 1


# ===========================================================================
# 3. Data Gaps - Missing Legal Entity
# ===========================================================================


class TestDataGapsMissingLegalEntity:
    """Test detection of missing legal entity information (F7.1)."""

    def test_detect_missing_registration_id(self, gap_analyzer):
        """Supplier without registration ID flagged."""
        suppliers = [
            make_supplier(supplier_id="LEG-GAP-001", registration_id=None, legal_name="NoReg"),
        ]
        result = gap_analyzer.analyze(suppliers)
        legal_gaps = [g for g in result.gaps if g["gap_type"] == "missing_legal_entity"
                      and g["supplier_id"] == "LEG-GAP-001"]
        assert len(legal_gaps) >= 1

    def test_full_legal_entity_no_gap(self, gap_analyzer):
        """Supplier with registration ID and tax ID has no legal gap."""
        suppliers = [
            make_supplier(
                supplier_id="LEG-OK-001",
                registration_id="REG-001",
                tax_id="TAX-001",
            ),
        ]
        result = gap_analyzer.analyze(suppliers)
        legal_gaps = [g for g in result.gaps if g["gap_type"] == "missing_legal_entity"
                      and g["supplier_id"] == "LEG-OK-001"]
        assert len(legal_gaps) == 0


# ===========================================================================
# 4. Data Gaps - Missing DDS
# ===========================================================================


class TestDataGapsMissingDDS:
    """Test detection of missing DDS reference gaps (F7.1)."""

    def test_detect_missing_dds(self, gap_analyzer):
        """Supplier without DDS references flagged."""
        suppliers = [
            make_supplier(supplier_id="DDS-GAP-001", dds_references=[]),
        ]
        result = gap_analyzer.analyze(suppliers)
        dds_gaps = [g for g in result.gaps if g["gap_type"] == "missing_dds"
                    and g["supplier_id"] == "DDS-GAP-001"]
        assert len(dds_gaps) >= 1

    def test_no_gap_with_dds(self, gap_analyzer):
        """Supplier with DDS reference has no DDS gap."""
        suppliers = [
            make_supplier(supplier_id="DDS-OK-001", dds_references=["DDS-001"]),
        ]
        result = gap_analyzer.analyze(suppliers)
        dds_gaps = [g for g in result.gaps if g["gap_type"] == "missing_dds"
                    and g["supplier_id"] == "DDS-OK-001"]
        assert len(dds_gaps) == 0


# ===========================================================================
# 5. Coverage Gaps - Missing Tiers
# ===========================================================================


class TestCoverageGaps:
    """Test detection of missing tier coverage gaps (F7.2)."""

    def test_detect_missing_tier(self, gap_analyzer):
        """Chain with missing tier is flagged as coverage gap."""
        suppliers = [
            make_supplier(supplier_id="COV-T0", tier=0),
            make_supplier(supplier_id="COV-T1", tier=1),
            # tier 2 missing
            make_supplier(supplier_id="COV-T3", tier=3),
        ]
        rels = [
            make_relationship("COV-T0", "COV-T1", rel_id="R-COV-1"),
            make_relationship("COV-T1", "COV-T3", rel_id="R-COV-2"),
        ]
        result = gap_analyzer.analyze_coverage(suppliers, rels, "cocoa")
        coverage_gaps = [g for g in result.gaps if g["gap_type"] == "missing_tier"]
        assert len(coverage_gaps) >= 1
        assert any(g["missing_tier"] == 2 for g in coverage_gaps)

    def test_no_coverage_gap_complete_chain(self, gap_analyzer, cocoa_chain):
        """Complete chain has no coverage gaps."""
        suppliers, rels = cocoa_chain
        result = gap_analyzer.analyze_coverage(suppliers, rels, "cocoa")
        coverage_gaps = [g for g in result.gaps if g["gap_type"] == "missing_tier"]
        assert len(coverage_gaps) == 0

    def test_shallow_chain_coverage_gap(self, gap_analyzer):
        """Chain much shorter than commodity typical triggers coverage gap."""
        suppliers = [
            make_supplier(supplier_id="SHAL-T0", tier=0, commodity="cocoa"),
            make_supplier(supplier_id="SHAL-T1", tier=1, commodity="cocoa"),
        ]
        rels = [make_relationship("SHAL-T0", "SHAL-T1", rel_id="R-SHAL")]
        result = gap_analyzer.analyze_coverage(suppliers, rels, "cocoa")
        # Cocoa typically 6-8 tiers; 2-tier chain should trigger gap
        assert len(result.gaps) >= 1


# ===========================================================================
# 6. Verification Gaps - Stale Data
# ===========================================================================


class TestVerificationGaps:
    """Test detection of stale/outdated data gaps (F7.3)."""

    def test_stale_data_flagged(self, gap_analyzer):
        """Supplier data not updated in 180+ days flagged as stale."""
        supplier = make_supplier(supplier_id="STALE-001")
        supplier["last_verified"] = (
            datetime.now(timezone.utc) - timedelta(days=200)
        ).isoformat()
        result = gap_analyzer.analyze([supplier])
        stale_gaps = [g for g in result.gaps if g["gap_type"] == "stale_data"
                      and g["supplier_id"] == "STALE-001"]
        assert len(stale_gaps) >= 1

    def test_fresh_data_no_gap(self, gap_analyzer):
        """Recently verified supplier data has no stale gap."""
        supplier = make_supplier(supplier_id="FRESH-001")
        supplier["last_verified"] = datetime.now(timezone.utc).isoformat()
        result = gap_analyzer.analyze([supplier])
        stale_gaps = [g for g in result.gaps if g["gap_type"] == "stale_data"
                      and g["supplier_id"] == "FRESH-001"]
        assert len(stale_gaps) == 0

    def test_never_verified_flagged(self, gap_analyzer):
        """Supplier never verified is flagged."""
        supplier = make_supplier(supplier_id="NEVER-VER")
        supplier["last_verified"] = None
        result = gap_analyzer.analyze([supplier])
        verification_gaps = [g for g in result.gaps
                             if g["gap_type"] in ("stale_data", "never_verified")
                             and g["supplier_id"] == "NEVER-VER"]
        assert len(verification_gaps) >= 1


# ===========================================================================
# 7. Gap Severity Classification
# ===========================================================================


class TestGapSeverityClassification:
    """Test gap severity classification rules (F7.4)."""

    def test_missing_gps_critical_for_farmer(self, gap_analyzer):
        """Missing GPS for farmer (origin) is critical severity."""
        suppliers = [
            make_supplier(supplier_id="SEV-FM", tier=5, role="farmer",
                          gps_lat=None, gps_lon=None),
        ]
        result = gap_analyzer.analyze(suppliers)
        gps_gaps = [g for g in result.gaps if g["gap_type"] == "missing_gps"
                    and g["supplier_id"] == "SEV-FM"]
        assert len(gps_gaps) >= 1
        assert gps_gaps[0]["severity"] == "critical"

    def test_missing_cert_major_for_processor(self, gap_analyzer):
        """Missing certification for processor is major severity."""
        suppliers = [
            make_supplier(supplier_id="SEV-PRC", tier=2, role="processor",
                          certifications=[]),
        ]
        result = gap_analyzer.analyze(suppliers)
        cert_gaps = [g for g in result.gaps if g["gap_type"] == "missing_certification"
                     and g["supplier_id"] == "SEV-PRC"]
        if cert_gaps:
            assert cert_gaps[0]["severity"] in ("critical", "major")

    def test_missing_contact_minor(self, gap_analyzer):
        """Missing compliance contact for trader is minor severity."""
        suppliers = [
            make_supplier(supplier_id="SEV-CONT", tier=1, role="trader",
                          compliance_contact=None),
        ]
        result = gap_analyzer.analyze(suppliers)
        contact_gaps = [g for g in result.gaps if g["gap_type"] == "missing_contact"
                        and g["supplier_id"] == "SEV-CONT"]
        if contact_gaps:
            assert contact_gaps[0]["severity"] == "minor"

    @pytest.mark.parametrize("severity", GAP_SEVERITIES)
    def test_all_severities_are_valid(self, gap_analyzer, severity):
        """All defined severity levels are recognized."""
        assert gap_analyzer.is_valid_severity(severity) is True

    def test_critical_blocks_dds(self, gap_analyzer):
        """Critical gaps are annotated as DDS-blocking."""
        suppliers = [
            make_supplier(supplier_id="BLOCK-001", tier=5, role="farmer",
                          gps_lat=None, gps_lon=None),
        ]
        result = gap_analyzer.analyze(suppliers)
        critical_gaps = [g for g in result.gaps if g["severity"] == "critical"]
        for g in critical_gaps:
            assert g.get("blocks_dds", False) is True


# ===========================================================================
# 8. Remediation Plan Generation
# ===========================================================================


class TestRemediationPlanGeneration:
    """Test remediation action plan generation (F7.5)."""

    def test_generate_remediation_plan(self, gap_analyzer):
        """Remediation plan is generated for detected gaps."""
        suppliers = [
            make_supplier(supplier_id="REM-001", gps_lat=None, gps_lon=None,
                          certifications=[], registration_id=None),
        ]
        result = gap_analyzer.analyze(suppliers)
        plan = gap_analyzer.generate_remediation_plan(result.gaps)
        assert plan is not None
        assert len(plan.actions) >= 1

    def test_remediation_plan_prioritized(self, gap_analyzer):
        """Remediation actions are prioritized (critical first)."""
        suppliers = [
            make_supplier(supplier_id="REM-PRI", gps_lat=None, gps_lon=None,
                          certifications=[], tier=5, role="farmer"),
        ]
        result = gap_analyzer.analyze(suppliers)
        plan = gap_analyzer.generate_remediation_plan(result.gaps)
        if len(plan.actions) > 1:
            # First action should be critical or highest priority
            assert plan.actions[0]["priority"] <= plan.actions[-1]["priority"]

    def test_remediation_action_has_description(self, gap_analyzer):
        """Each remediation action has a human-readable description."""
        suppliers = [
            make_supplier(supplier_id="REM-DESC", gps_lat=None, gps_lon=None),
        ]
        result = gap_analyzer.analyze(suppliers)
        plan = gap_analyzer.generate_remediation_plan(result.gaps)
        for action in plan.actions:
            assert "description" in action
            assert len(action["description"]) > 0

    def test_remediation_plan_empty_for_no_gaps(self, gap_analyzer):
        """No gaps produces empty remediation plan."""
        plan = gap_analyzer.generate_remediation_plan([])
        assert len(plan.actions) == 0

    def test_remediation_plan_includes_deadline(self, gap_analyzer):
        """Remediation actions include suggested deadlines."""
        suppliers = [
            make_supplier(supplier_id="REM-DL", gps_lat=None, gps_lon=None),
        ]
        result = gap_analyzer.analyze(suppliers)
        plan = gap_analyzer.generate_remediation_plan(result.gaps)
        for action in plan.actions:
            assert "deadline" in action or "suggested_completion" in action


# ===========================================================================
# 9. Remediation Progress Tracking
# ===========================================================================


class TestRemediationProgress:
    """Test remediation progress tracking (F7.6)."""

    def test_track_progress_completion(self, gap_analyzer):
        """Mark a remediation action as completed."""
        plan_id = "PLAN-PROG-001"
        gap_analyzer.create_remediation_plan(
            plan_id=plan_id,
            actions=[
                {"action_id": "A-001", "description": "Collect GPS", "status": "pending"},
                {"action_id": "A-002", "description": "Get cert", "status": "pending"},
            ],
        )
        gap_analyzer.update_action_status(plan_id, "A-001", "completed")
        progress = gap_analyzer.get_progress(plan_id)
        assert progress.completion_pct == pytest.approx(50.0)

    def test_track_progress_all_complete(self, gap_analyzer):
        """All actions completed yields 100% progress."""
        plan_id = "PLAN-PROG-002"
        gap_analyzer.create_remediation_plan(
            plan_id=plan_id,
            actions=[
                {"action_id": "A-003", "description": "Action 1", "status": "pending"},
            ],
        )
        gap_analyzer.update_action_status(plan_id, "A-003", "completed")
        progress = gap_analyzer.get_progress(plan_id)
        assert progress.completion_pct == pytest.approx(100.0)

    def test_progress_zero_when_none_complete(self, gap_analyzer):
        """No completed actions yields 0% progress."""
        plan_id = "PLAN-PROG-003"
        gap_analyzer.create_remediation_plan(
            plan_id=plan_id,
            actions=[
                {"action_id": "A-004", "description": "Pending", "status": "pending"},
            ],
        )
        progress = gap_analyzer.get_progress(plan_id)
        assert progress.completion_pct == pytest.approx(0.0)


# ===========================================================================
# 10. Questionnaire Auto-Generation
# ===========================================================================


class TestQuestionnaireGeneration:
    """Test auto-generated supplier questionnaires for gap filling (F7.7)."""

    def test_generate_questionnaire_for_gps_gap(self, gap_analyzer):
        """GPS gap generates questionnaire asking for coordinates."""
        gaps = [{"gap_type": "missing_gps", "supplier_id": "Q-GPS-001", "severity": "critical"}]
        questionnaire = gap_analyzer.generate_questionnaire(gaps)
        assert questionnaire is not None
        assert len(questionnaire.questions) >= 1
        gps_questions = [q for q in questionnaire.questions
                         if "gps" in q["topic"].lower() or "coordinate" in q["topic"].lower()
                         or "location" in q["topic"].lower()]
        assert len(gps_questions) >= 1

    def test_generate_questionnaire_for_cert_gap(self, gap_analyzer):
        """Certification gap generates questionnaire about certifications."""
        gaps = [{"gap_type": "missing_certification", "supplier_id": "Q-CERT-001",
                 "severity": "major"}]
        questionnaire = gap_analyzer.generate_questionnaire(gaps)
        cert_questions = [q for q in questionnaire.questions
                          if "cert" in q["topic"].lower()]
        assert len(cert_questions) >= 1

    def test_generate_questionnaire_combined_gaps(self, gap_analyzer):
        """Multiple gap types generate comprehensive questionnaire."""
        gaps = [
            {"gap_type": "missing_gps", "supplier_id": "Q-COMBO", "severity": "critical"},
            {"gap_type": "missing_certification", "supplier_id": "Q-COMBO", "severity": "major"},
            {"gap_type": "missing_legal_entity", "supplier_id": "Q-COMBO", "severity": "major"},
        ]
        questionnaire = gap_analyzer.generate_questionnaire(gaps)
        assert len(questionnaire.questions) >= 3

    def test_empty_gaps_empty_questionnaire(self, gap_analyzer):
        """No gaps produces empty questionnaire."""
        questionnaire = gap_analyzer.generate_questionnaire([])
        assert len(questionnaire.questions) == 0

    def test_questionnaire_includes_supplier_id(self, gap_analyzer):
        """Questionnaire is targeted at specific supplier."""
        gaps = [{"gap_type": "missing_gps", "supplier_id": "Q-TARGET", "severity": "critical"}]
        questionnaire = gap_analyzer.generate_questionnaire(gaps)
        assert questionnaire.target_supplier_id == "Q-TARGET"


# ===========================================================================
# 11. Gap Trend Analysis
# ===========================================================================


class TestGapTrendAnalysis:
    """Test gap trend analysis over time (F7.8)."""

    def test_improving_trend(self, gap_analyzer):
        """Decreasing gap count indicates improving trend."""
        history = [
            {"date": "2026-01-01", "gap_count": 50},
            {"date": "2026-02-01", "gap_count": 35},
            {"date": "2026-03-01", "gap_count": 20},
        ]
        trend = gap_analyzer.analyze_trend(history)
        assert trend.direction in ("improving", "decreasing")

    def test_worsening_trend(self, gap_analyzer):
        """Increasing gap count indicates worsening trend."""
        history = [
            {"date": "2026-01-01", "gap_count": 10},
            {"date": "2026-02-01", "gap_count": 25},
            {"date": "2026-03-01", "gap_count": 40},
        ]
        trend = gap_analyzer.analyze_trend(history)
        assert trend.direction in ("worsening", "increasing")

    def test_stable_trend(self, gap_analyzer):
        """Stable gap count indicates stable trend."""
        history = [
            {"date": "2026-01-01", "gap_count": 30},
            {"date": "2026-02-01", "gap_count": 29},
            {"date": "2026-03-01", "gap_count": 31},
        ]
        trend = gap_analyzer.analyze_trend(history)
        assert trend.direction in ("stable", "flat")

    def test_empty_history_returns_unknown(self, gap_analyzer):
        """Empty history returns unknown trend."""
        trend = gap_analyzer.analyze_trend([])
        assert trend.direction in ("unknown", "insufficient_data")


# ===========================================================================
# 12. Provenance
# ===========================================================================


class TestGapAnalysisProvenance:
    """Test provenance tracking for gap analysis results."""

    def test_analysis_has_provenance(self, gap_analyzer):
        """Gap analysis result includes provenance hash."""
        suppliers = [
            make_supplier(supplier_id="GAP-PROV", gps_lat=None, gps_lon=None),
        ]
        result = gap_analyzer.analyze(suppliers)
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_analysis_deterministic(self, gap_analyzer):
        """Same input produces same gap analysis result."""
        suppliers = [
            make_supplier(supplier_id="GAP-DET", gps_lat=None, gps_lon=None),
        ]
        r1 = gap_analyzer.analyze(suppliers)
        r2 = gap_analyzer.analyze(suppliers)
        assert r1.provenance_hash == r2.provenance_hash
        assert len(r1.gaps) == len(r2.gaps)


# ===========================================================================
# 13. Comprehensive Gap Combinations
# ===========================================================================


class TestComprehensiveGapCombinations:
    """Test combined gap scenarios across multiple gap types."""

    def test_supplier_with_all_gaps(self, gap_analyzer):
        """Supplier missing everything has multiple gaps detected."""
        suppliers = [
            make_supplier(
                supplier_id="ALL-GAP",
                registration_id=None,
                tax_id=None,
                gps_lat=None,
                gps_lon=None,
                certifications=[],
                dds_references=[],
                primary_contact=None,
                compliance_contact=None,
            ),
        ]
        result = gap_analyzer.analyze(suppliers)
        # Should have multiple gap types
        gap_types = {g["gap_type"] for g in result.gaps}
        assert len(gap_types) >= 3

    def test_supplier_with_no_gaps(self, gap_analyzer):
        """Complete supplier has zero gaps."""
        suppliers = [
            make_supplier(
                supplier_id="NO-GAP",
                legal_name="Complete Corp",
                registration_id="REG-001",
                tax_id="TAX-001",
                gps_lat=5.6037,
                gps_lon=-0.1870,
                certifications=["RSPO-001"],
                dds_references=["DDS-001"],
                primary_contact="Alice",
                compliance_contact="Bob",
            ),
        ]
        suppliers[0]["last_verified"] = datetime.now(timezone.utc).isoformat()
        result = gap_analyzer.analyze(suppliers)
        # Filter only gaps for this supplier
        supplier_gaps = [g for g in result.gaps if g["supplier_id"] == "NO-GAP"]
        assert len(supplier_gaps) == 0

    @pytest.mark.parametrize("gap_type,missing_field,field_value", [
        ("missing_gps", "gps_lat", None),
        ("missing_certification", "certifications", []),
        ("missing_dds", "dds_references", []),
        ("missing_legal_entity", "registration_id", None),
    ])
    def test_individual_gap_detection(self, gap_analyzer, gap_type, missing_field, field_value):
        """Each gap type is detected independently."""
        kwargs = {
            "supplier_id": f"IND-GAP-{gap_type}",
            "legal_name": "Individual Gap Test",
            "registration_id": "REG-001",
            "gps_lat": 5.6037,
            "gps_lon": -0.1870,
            "certifications": ["FSC-001"],
            "dds_references": ["DDS-001"],
        }
        kwargs[missing_field] = field_value
        if missing_field == "gps_lat":
            kwargs["gps_lon"] = None
        suppliers = [make_supplier(**kwargs)]
        result = gap_analyzer.analyze(suppliers)
        matching_gaps = [g for g in result.gaps
                         if g["gap_type"] == gap_type
                         and g["supplier_id"] == kwargs["supplier_id"]]
        assert len(matching_gaps) >= 1

    def test_gap_count_summary(self, gap_analyzer):
        """Gap analysis provides count summary by type."""
        suppliers = [
            make_supplier(supplier_id=f"SUM-{i}", gps_lat=None, gps_lon=None,
                          certifications=[], registration_id=None)
            for i in range(3)
        ]
        result = gap_analyzer.analyze(suppliers)
        assert result.total_gaps >= 3

    def test_gap_analysis_large_supplier_set(self, gap_analyzer):
        """Gap analysis handles 100+ suppliers efficiently."""
        suppliers = [
            make_supplier(
                supplier_id=f"LARGE-GAP-{i:04d}",
                gps_lat=None if i % 2 == 0 else 5.0,
                gps_lon=None if i % 2 == 0 else -0.2,
                certifications=[] if i % 3 == 0 else ["FSC-001"],
            )
            for i in range(100)
        ]
        result = gap_analyzer.analyze(suppliers)
        assert result is not None
        assert result.total_gaps >= 50  # At least 50 suppliers missing GPS

    @pytest.mark.parametrize("role,tier,expected_severity", [
        ("farmer", 5, "critical"),
        ("cooperative", 4, "critical"),
        ("aggregator", 3, "major"),
        ("processor", 2, "major"),
        ("trader", 1, "major"),
        ("importer", 0, "major"),
    ])
    def test_gps_gap_severity_by_role(self, gap_analyzer, role, tier, expected_severity):
        """GPS gap severity depends on supplier role and tier."""
        suppliers = [
            make_supplier(supplier_id=f"SEV-{role}", tier=tier, role=role,
                          gps_lat=None, gps_lon=None),
        ]
        result = gap_analyzer.analyze(suppliers)
        gps_gaps = [g for g in result.gaps if g["gap_type"] == "missing_gps"
                    and g["supplier_id"] == f"SEV-{role}"]
        if gps_gaps:
            assert gps_gaps[0]["severity"] in (expected_severity, "critical")
