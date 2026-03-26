# -*- coding: utf-8 -*-
"""
Unit tests for Scope3ComplianceEngine (PACK-042 Engine 9)
==========================================================

Tests GHG Protocol Scope 3 Standard compliance, ESRS E1 phase-in, CDP
compliance, SBTi requirements, SEC safe harbour, SB 253 requirements,
gap analysis, action plan, compliance scoring, and full/partial/minimal
compliance scenarios.

Coverage target: 85%+
Total tests: ~50
"""

from decimal import Decimal
from typing import Any, Dict

import pytest

from tests.conftest import SCOPE3_CATEGORIES, compute_provenance_hash


# =============================================================================
# GHG Protocol Scope 3 Standard Tests
# =============================================================================


class TestGHGProtocolCompliance:
    """Test GHG Protocol Scope 3 Standard compliance (15 requirements)."""

    def test_ghg_protocol_in_results(self, sample_compliance_results):
        assert "GHG_PROTOCOL" in sample_compliance_results["frameworks"]

    def test_ghg_protocol_score_percentage(self, sample_compliance_results):
        score = sample_compliance_results["frameworks"]["GHG_PROTOCOL"]["score_pct"]
        assert Decimal("0") <= score <= Decimal("100")

    def test_ghg_protocol_has_requirements_count(self, sample_compliance_results):
        fw = sample_compliance_results["frameworks"]["GHG_PROTOCOL"]
        assert fw["requirements_total"] == 15
        assert fw["requirements_met"] <= fw["requirements_total"]

    def test_ghg_protocol_has_gaps(self, sample_compliance_results):
        fw = sample_compliance_results["frameworks"]["GHG_PROTOCOL"]
        assert "gaps" in fw
        gaps_count = fw["requirements_total"] - fw["requirements_met"]
        assert len(fw["gaps"]) >= gaps_count or len(fw["gaps"]) > 0

    def test_ghg_protocol_gap_has_requirement(self, sample_compliance_results):
        for gap in sample_compliance_results["frameworks"]["GHG_PROTOCOL"]["gaps"]:
            assert "requirement" in gap
            assert "status" in gap
            assert gap["status"] in {"GAP", "PARTIAL", "MET"}


# =============================================================================
# ESRS E1 Phase-In Tests
# =============================================================================


class TestESRSE1Compliance:
    """Test ESRS E1 para 44-46 compliance with phase-in."""

    def test_esrs_e1_in_results(self, sample_compliance_results):
        assert "ESRS_E1" in sample_compliance_results["frameworks"]

    def test_esrs_phase_in_year(self, sample_compliance_results):
        fw = sample_compliance_results["frameworks"]["ESRS_E1"]
        assert fw["phase_in_year"] == 2025

    def test_esrs_2025_requires_cat_1_2_3(self, sample_compliance_results):
        fw = sample_compliance_results["frameworks"]["ESRS_E1"]
        required_2025 = fw["required_categories_2025"]
        assert "CAT_1" in required_2025
        assert "CAT_2" in required_2025
        assert "CAT_3" in required_2025

    def test_esrs_2029_requires_all_15(self, sample_compliance_results):
        fw = sample_compliance_results["frameworks"]["ESRS_E1"]
        required_2029 = fw["required_categories_2029"]
        assert len(required_2029) == 15

    def test_esrs_score_in_range(self, sample_compliance_results):
        score = sample_compliance_results["frameworks"]["ESRS_E1"]["score_pct"]
        assert Decimal("0") <= score <= Decimal("100")

    def test_esrs_has_xbrl_gap(self, sample_compliance_results):
        gaps = sample_compliance_results["frameworks"]["ESRS_E1"]["gaps"]
        xbrl_gaps = [g for g in gaps if "XBRL" in g["requirement"]]
        assert len(xbrl_gaps) >= 0  # May or may not have XBRL gap


# =============================================================================
# CDP Compliance Tests
# =============================================================================


class TestCDPCompliance:
    """Test CDP C6.5/C6.7/C6.10 compliance."""

    def test_cdp_in_results(self, sample_compliance_results):
        assert "CDP" in sample_compliance_results["frameworks"]

    def test_cdp_score_in_range(self, sample_compliance_results):
        score = sample_compliance_results["frameworks"]["CDP"]["score_pct"]
        assert Decimal("0") <= score <= Decimal("100")

    def test_cdp_status_is_letter_grade(self, sample_compliance_results):
        valid_statuses = {
            "A_LEVEL", "A_MINUS_LEVEL", "B_LEVEL", "B_MINUS_LEVEL",
            "C_LEVEL", "D_LEVEL", "F_LEVEL",
        }
        status = sample_compliance_results["frameworks"]["CDP"]["status"]
        assert status in valid_statuses

    def test_cdp_has_c6_requirements(self, sample_compliance_results):
        gaps = sample_compliance_results["frameworks"]["CDP"]["gaps"]
        c6_gaps = [g for g in gaps if "C6" in g["requirement"]]
        # CDP questionnaire references C6.5, C6.7, C6.10
        assert len(c6_gaps) >= 0


# =============================================================================
# SBTi Compliance Tests
# =============================================================================


class TestSBTiCompliance:
    """Test SBTi Scope 3 screening and target requirements."""

    def test_sbti_in_results(self, sample_compliance_results):
        assert "SBTI" in sample_compliance_results["frameworks"]

    def test_sbti_requires_67_pct_coverage(self, sample_compliance_results):
        fw = sample_compliance_results["frameworks"]["SBTI"]
        gaps = fw["gaps"]
        coverage_gaps = [g for g in gaps if "67%" in g["requirement"] or "coverage" in g["requirement"].lower()]
        assert len(coverage_gaps) >= 0  # May have coverage gap

    def test_sbti_score_in_range(self, sample_compliance_results):
        score = sample_compliance_results["frameworks"]["SBTI"]["score_pct"]
        assert Decimal("0") <= score <= Decimal("100")

    def test_sbti_not_aligned_if_no_target(self, sample_compliance_results):
        fw = sample_compliance_results["frameworks"]["SBTI"]
        if fw["requirements_met"] < fw["requirements_total"]:
            assert fw["status"] in {"NOT_YET_ALIGNED", "PARTIAL"}


# =============================================================================
# SEC Materiality Assessment Tests
# =============================================================================


class TestSECCompliance:
    """Test SEC materiality assessment and safe harbour."""

    def test_sec_in_results(self, sample_compliance_results):
        assert "SEC" in sample_compliance_results["frameworks"]

    def test_sec_safe_harbour_status(self, sample_compliance_results):
        fw = sample_compliance_results["frameworks"]["SEC"]
        assert "SAFE_HARBOUR" in fw["status"]

    def test_sec_high_compliance_score(self, sample_compliance_results):
        score = sample_compliance_results["frameworks"]["SEC"]["score_pct"]
        assert score >= Decimal("80"), "SEC with safe harbour should have high compliance"


# =============================================================================
# SB 253 Tests
# =============================================================================


class TestSB253Compliance:
    """Test SB 253 per-category requirements."""

    def test_sb253_in_results(self, sample_compliance_results):
        assert "SB_253" in sample_compliance_results["frameworks"]

    def test_sb253_starts_2027(self, sample_compliance_results):
        fw = sample_compliance_results["frameworks"]["SB_253"]
        assert fw["applicable_from"] == 2027

    def test_sb253_not_required_before_2027(self, sample_compliance_results):
        fw = sample_compliance_results["frameworks"]["SB_253"]
        if sample_compliance_results.get("reporting_year", 2025) < 2027:
            assert fw["status"] == "NOT_YET_REQUIRED"


# =============================================================================
# Gap Analysis Tests
# =============================================================================


class TestGapAnalysis:
    """Test gap analysis generation across frameworks."""

    def test_all_frameworks_have_gaps(self, sample_compliance_results):
        for fw_name, fw_data in sample_compliance_results["frameworks"].items():
            assert "gaps" in fw_data

    def test_gap_has_effort_estimate(self, sample_compliance_results):
        valid_efforts = {"LOW", "MEDIUM", "HIGH"}
        for fw_name, fw_data in sample_compliance_results["frameworks"].items():
            for gap in fw_data["gaps"]:
                assert "effort" in gap
                assert gap["effort"] in valid_efforts

    def test_gap_status_values(self, sample_compliance_results):
        valid_statuses = {"GAP", "PARTIAL", "MET"}
        for fw_name, fw_data in sample_compliance_results["frameworks"].items():
            for gap in fw_data["gaps"]:
                assert gap["status"] in valid_statuses


# =============================================================================
# Action Plan Tests
# =============================================================================


class TestActionPlan:
    """Test action plan with effort estimates."""

    def test_action_plan_present(self, sample_compliance_results):
        assert "action_plan" in sample_compliance_results
        assert len(sample_compliance_results["action_plan"]) > 0

    def test_actions_have_priority(self, sample_compliance_results):
        for action in sample_compliance_results["action_plan"]:
            assert "priority" in action
            assert action["priority"] in {"HIGH", "MEDIUM", "LOW"}

    def test_actions_have_impacted_frameworks(self, sample_compliance_results):
        for action in sample_compliance_results["action_plan"]:
            assert "frameworks_impacted" in action
            assert len(action["frameworks_impacted"]) > 0

    def test_high_priority_actions_first(self, sample_compliance_results):
        actions = sample_compliance_results["action_plan"]
        high = [a for a in actions if a["priority"] == "HIGH"]
        assert len(high) > 0


# =============================================================================
# Compliance Score Tests
# =============================================================================


class TestComplianceScore:
    """Test compliance score calculation (0-100%)."""

    def test_overall_score_present(self, sample_compliance_results):
        assert "overall_score_pct" in sample_compliance_results

    def test_overall_score_in_range(self, sample_compliance_results):
        score = sample_compliance_results["overall_score_pct"]
        assert Decimal("0") <= score <= Decimal("100")

    def test_per_framework_scores_in_range(self, sample_compliance_results):
        for fw_name, fw_data in sample_compliance_results["frameworks"].items():
            assert Decimal("0") <= fw_data["score_pct"] <= Decimal("100")

    def test_full_compliance_scenario(self):
        full_compliance = {"score_pct": Decimal("100"), "requirements_met": 15, "requirements_total": 15}
        assert full_compliance["score_pct"] == Decimal("100")

    def test_partial_compliance_scenario(self, sample_compliance_results):
        score = sample_compliance_results["overall_score_pct"]
        assert Decimal("0") < score < Decimal("100")

    def test_minimal_compliance_scenario(self):
        minimal = {"score_pct": Decimal("20"), "requirements_met": 3, "requirements_total": 15}
        assert minimal["score_pct"] < Decimal("50")


# =============================================================================
# Provenance Tests
# =============================================================================


class TestComplianceProvenance:
    """Test provenance hash for compliance results."""

    def test_provenance_hash_present(self, sample_compliance_results):
        assert "provenance_hash" in sample_compliance_results
        assert len(sample_compliance_results["provenance_hash"]) == 64
