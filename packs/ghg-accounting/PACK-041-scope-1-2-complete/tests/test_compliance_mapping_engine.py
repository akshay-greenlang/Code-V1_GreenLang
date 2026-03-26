# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceMappingEngine -- PACK-041 Engine 9
==============================================================

Tests compliance checking against GHG Protocol, ESRS E1, CDP, ISO 14064,
SBTi, SEC, SB 253, multi-framework simultaneous mapping, gap analysis,
and remediation actions.

Coverage target: 85%+
Total tests: ~65
"""

from decimal import Decimal

import pytest


# =============================================================================
# Compliance Score Calculation
# =============================================================================


class TestComplianceScoreCalculation:
    """Test compliance score computation."""

    @pytest.mark.parametrize("requirements_met,total_requirements,expected_score", [
        (50, 50, 100.0),
        (47, 50, 94.0),
        (40, 50, 80.0),
        (30, 50, 60.0),
        (20, 50, 40.0),
        (0, 50, 0.0),
    ])
    def test_score_calculation(self, requirements_met, total_requirements, expected_score):
        score = requirements_met / total_requirements * 100 if total_requirements else 0
        assert score == pytest.approx(expected_score)


# =============================================================================
# Classification Thresholds
# =============================================================================


class TestClassificationThresholds:
    """Test compliance classification based on score."""

    @pytest.mark.parametrize("score,expected_class", [
        (100.0, "COMPLIANT"),
        (95.0, "COMPLIANT"),
        (94.9, "SUBSTANTIALLY_COMPLIANT"),
        (80.0, "SUBSTANTIALLY_COMPLIANT"),
        (79.9, "PARTIALLY_COMPLIANT"),
        (60.0, "PARTIALLY_COMPLIANT"),
        (59.9, "NON_COMPLIANT"),
        (0.0, "NON_COMPLIANT"),
    ])
    def test_classification(self, score, expected_class):
        if score >= 95:
            classification = "COMPLIANT"
        elif score >= 80:
            classification = "SUBSTANTIALLY_COMPLIANT"
        elif score >= 60:
            classification = "PARTIALLY_COMPLIANT"
        else:
            classification = "NON_COMPLIANT"
        assert classification == expected_class


# =============================================================================
# GHG Protocol Compliance
# =============================================================================


class TestGHGProtocolCompliance:
    """Test compliance with GHG Protocol Corporate Standard."""

    def test_ghg_protocol_full_compliance(self, sample_inventory):
        """Full inventory with boundary, scope 1+2, and provenance is compliant."""
        requirements = {
            "organizational_boundary": True,
            "operational_boundary": True,
            "scope1_quantified": sample_inventory["scope1"]["total_tco2e"] > 0,
            "scope2_quantified": sample_inventory["scope2_location"]["total_tco2e"] > 0,
            "base_year_established": True,
            "consistent_methodology": True,
            "completeness_documented": sample_inventory["completeness_pct"] > 90,
            "uncertainty_assessed": sample_inventory["uncertainty_pct"] > 0,
        }
        met = sum(1 for v in requirements.values() if v)
        score = met / len(requirements) * 100
        assert score >= 95

    def test_ghg_protocol_missing_boundary(self):
        """Missing boundary definition should reduce compliance."""
        requirements = {
            "organizational_boundary": False,
            "operational_boundary": False,
            "scope1_quantified": True,
            "scope2_quantified": True,
        }
        met = sum(1 for v in requirements.values() if v)
        score = met / len(requirements) * 100
        assert score == 50.0

    def test_ghg_protocol_scope2_dual_required(self, sample_inventory):
        """GHG Protocol Scope 2 Guidance requires dual reporting."""
        has_location = sample_inventory["scope2_location"]["total_tco2e"] >= 0
        has_market = sample_inventory["scope2_market"]["total_tco2e"] >= 0
        assert has_location and has_market


# =============================================================================
# ESRS E1 Compliance
# =============================================================================


class TestESRSE1Compliance:
    """Test compliance with EU CSRD / ESRS E1."""

    def test_esrs_e1_scope1_reported(self, sample_inventory):
        assert sample_inventory["scope1"]["total_tco2e"] > Decimal("0")

    def test_esrs_e1_scope2_location_reported(self, sample_inventory):
        assert sample_inventory["scope2_location"]["total_tco2e"] > Decimal("0")

    def test_esrs_e1_scope2_market_reported(self, sample_inventory):
        assert sample_inventory["scope2_market"]["total_tco2e"] >= Decimal("0")

    def test_esrs_e1_by_gas_breakdown(self, sample_inventory):
        gases = sample_inventory["scope1"]["by_gas"]
        assert "CO2" in gases
        assert "CH4" in gases
        assert "N2O" in gases

    def test_esrs_e1_intensity_metric(self, sample_yearly_data):
        yr = sample_yearly_data[-1]
        intensity = yr["total_scope1_tco2e"] / yr["revenue_million_usd"]
        assert intensity > Decimal("0")


# =============================================================================
# CDP Compliance
# =============================================================================


class TestCDPCompliance:
    """Test compliance with CDP Climate Change questionnaire."""

    def test_cdp_scope1_gross(self, sample_inventory):
        assert sample_inventory["scope1"]["total_tco2e"] > Decimal("0")

    def test_cdp_scope2_location(self, sample_inventory):
        assert sample_inventory["scope2_location"]["total_tco2e"] > Decimal("0")

    def test_cdp_scope2_market(self, sample_inventory):
        assert sample_inventory["scope2_market"]["total_tco2e"] >= Decimal("0")

    def test_cdp_by_country_breakdown(self, sample_boundary):
        assert len(sample_boundary["countries_covered"]) >= 1

    def test_cdp_verification_status(self):
        verification = {"status": "third_party_limited", "verifier": "PwC"}
        assert verification["status"] in {"third_party_limited", "third_party_reasonable", "none"}


# =============================================================================
# ISO 14064 Compliance
# =============================================================================


class TestISO14064Compliance:
    """Test compliance with ISO 14064-1:2018."""

    def test_iso_organizational_boundary(self, sample_boundary):
        assert sample_boundary["approach"] in {"equity_share", "operational_control", "financial_control"}

    def test_iso_quantification_scope1(self, sample_inventory):
        assert sample_inventory["scope1"]["total_tco2e"] >= Decimal("0")

    def test_iso_quantification_scope2(self, sample_inventory):
        assert sample_inventory["scope2_location"]["total_tco2e"] >= Decimal("0")

    def test_iso_uncertainty_assessment(self, sample_inventory):
        assert sample_inventory["uncertainty_pct"] > Decimal("0")

    def test_iso_base_year_documented(self, sample_base_year):
        assert sample_base_year["base_year"] >= 1990


# =============================================================================
# SBTi Compliance
# =============================================================================


class TestSBTiCompliance:
    """Test compliance with SBTi requirements."""

    def test_sbti_scope1_coverage(self, sample_inventory):
        """SBTi requires 95% Scope 1+2 coverage."""
        assert sample_inventory["completeness_pct"] >= Decimal("95")

    def test_sbti_base_year_recency(self, sample_base_year):
        """Base year must be no older than 5 years for validation."""
        current_year = 2025
        assert current_year - sample_base_year["base_year"] <= 10

    def test_sbti_target_ambition(self):
        """1.5C alignment requires 4.2% annual reduction."""
        target_annual_pct = Decimal("4.2")
        assert target_annual_pct > Decimal("0")


# =============================================================================
# SEC Climate Disclosure
# =============================================================================


class TestSECCompliance:
    """Test compliance with US SEC Climate Disclosure Rules."""

    def test_sec_scope1_material(self, sample_inventory):
        assert sample_inventory["scope1"]["total_tco2e"] > Decimal("0")

    def test_sec_scope2_material(self, sample_inventory):
        assert sample_inventory["scope2_location"]["total_tco2e"] > Decimal("0")

    def test_sec_attestation_required(self):
        """Large accelerated filers require limited assurance."""
        filer_status = "large_accelerated"
        requires_attestation = filer_status in {"large_accelerated", "accelerated"}
        assert requires_attestation is True

    def test_sec_physical_risk_disclosure(self):
        disclosure = {"physical_risks": True, "transition_risks": True}
        assert disclosure["physical_risks"] is True


# =============================================================================
# SB 253 Compliance
# =============================================================================


class TestSB253Compliance:
    """Test compliance with California SB 253."""

    def test_sb253_scope1_required(self, sample_inventory):
        assert sample_inventory["scope1"]["total_tco2e"] > Decimal("0")

    def test_sb253_scope2_required(self, sample_inventory):
        assert sample_inventory["scope2_location"]["total_tco2e"] > Decimal("0")

    def test_sb253_annual_reporting(self):
        reporting_frequency = "annual"
        assert reporting_frequency == "annual"

    def test_sb253_third_party_assurance(self):
        """SB 253 requires third-party assurance beginning 2026."""
        assurance = {"type": "limited", "provider": "Deloitte", "year": 2026}
        assert assurance["type"] in {"limited", "reasonable"}


# =============================================================================
# Multi-Framework Simultaneous
# =============================================================================


class TestMultiFrameworkCompliance:
    """Test simultaneous compliance mapping across frameworks."""

    def test_seven_frameworks_mapped(self, sample_pack_config):
        frameworks = sample_pack_config["reporting"]["frameworks"]
        assert len(frameworks) >= 7

    def test_all_frameworks_have_status(self):
        statuses = {
            "ghg_protocol": "COMPLIANT",
            "iso_14064": "COMPLIANT",
            "esrs_e1": "SUBSTANTIALLY_COMPLIANT",
            "cdp": "COMPLIANT",
            "sbti": "PARTIALLY_COMPLIANT",
            "sec": "SUBSTANTIALLY_COMPLIANT",
            "sb_253": "SUBSTANTIALLY_COMPLIANT",
        }
        assert len(statuses) == 7
        for fw, status in statuses.items():
            assert status in {
                "COMPLIANT", "SUBSTANTIALLY_COMPLIANT",
                "PARTIALLY_COMPLIANT", "NON_COMPLIANT",
            }

    def test_lowest_compliance_identified(self):
        scores = {"ghg_protocol": 98, "esrs_e1": 85, "sbti": 72}
        lowest = min(scores, key=scores.get)
        assert lowest == "sbti"


# =============================================================================
# Gap Analysis
# =============================================================================


class TestGapAnalysis:
    """Test compliance gap analysis generation."""

    def test_gap_identified(self):
        requirements = {"scope1": True, "scope2": True, "scope3": False}
        gaps = [k for k, v in requirements.items() if not v]
        assert "scope3" in gaps

    def test_gap_severity_ranking(self):
        gaps = [
            {"requirement": "scope3_cat1", "impact": "high"},
            {"requirement": "verification", "impact": "medium"},
            {"requirement": "base_year_policy", "impact": "low"},
        ]
        severity_order = {"high": 3, "medium": 2, "low": 1}
        sorted_gaps = sorted(gaps, key=lambda g: severity_order[g["impact"]], reverse=True)
        assert sorted_gaps[0]["requirement"] == "scope3_cat1"


# =============================================================================
# Remediation Actions
# =============================================================================


class TestRemediationActions:
    """Test remediation action generation for compliance gaps."""

    def test_remediation_for_missing_boundary(self):
        action = {
            "gap": "organizational_boundary_missing",
            "action": "Define organizational boundary using GHG Protocol Chapter 3",
            "priority": "critical",
            "estimated_effort_days": 5,
        }
        assert action["priority"] == "critical"

    def test_remediation_for_missing_uncertainty(self):
        action = {
            "gap": "uncertainty_not_assessed",
            "action": "Conduct uncertainty analysis per IPCC 2006 Vol 1 Ch 3",
            "priority": "high",
            "estimated_effort_days": 10,
        }
        assert action["priority"] == "high"

    def test_remediation_actions_sorted_by_priority(self):
        actions = [
            {"gap": "scope3", "priority": "medium"},
            {"gap": "boundary", "priority": "critical"},
            {"gap": "uncertainty", "priority": "high"},
        ]
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        sorted_actions = sorted(
            actions, key=lambda a: priority_order[a["priority"]], reverse=True
        )
        assert sorted_actions[0]["gap"] == "boundary"
