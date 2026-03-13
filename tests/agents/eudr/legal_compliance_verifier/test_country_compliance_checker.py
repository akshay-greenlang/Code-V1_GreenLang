# -*- coding: utf-8 -*-
"""
Tests for CountryComplianceCheckerEngine - AGENT-EUDR-023 Engine 5

Comprehensive test suite covering:
- Per-country rule sets for all 27 EUDR-relevant countries
- 8-category compliance checking per EUDR Article 2(40)
- Gap analysis generation with remediation recommendations
- Evidence sufficiency scoring
- Compliance determination (COMPLIANT/PARTIALLY_COMPLIANT/NON_COMPLIANT)
- Score aggregation across categories
- High-risk vs standard country handling
- Missing category handling
- Boundary value testing for compliance thresholds

Test count: 75+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 (Engine 5 - Country Compliance Checker)
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from tests.agents.eudr.legal_compliance_verifier.conftest import (
    compute_test_hash,
    compute_compliance_score,
    determine_compliance,
    SHA256_HEX_LENGTH,
    LEGISLATION_CATEGORIES,
    EUDR_COMMODITIES,
    EUDR_COUNTRIES_27,
    HIGH_RISK_COUNTRIES,
    LOW_RISK_COUNTRIES,
    COMPLIANCE_DETERMINATIONS,
    COMPLIANCE_BOUNDARIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_country_compliance(
    country_code: str,
    category_scores: Dict[str, Decimal],
    country_rules: Optional[Dict] = None,
    compliant_threshold: int = 80,
    partial_threshold: int = 50,
) -> Dict[str, Any]:
    """Check compliance for a country across all 8 categories."""
    result = {
        "country_code": country_code,
        "category_results": {},
        "overall_score": Decimal("0"),
        "determination": "NON_COMPLIANT",
        "compliant_categories": 0,
        "partial_categories": 0,
        "non_compliant_categories": 0,
        "missing_categories": [],
        "provenance_hash": None,
    }

    for category in LEGISLATION_CATEGORIES:
        if category in category_scores:
            score = category_scores[category]
            status = determine_compliance(score, compliant_threshold, partial_threshold)
            result["category_results"][category] = {
                "score": score,
                "status": status,
            }
            if status == "COMPLIANT":
                result["compliant_categories"] += 1
            elif status == "PARTIALLY_COMPLIANT":
                result["partial_categories"] += 1
            else:
                result["non_compliant_categories"] += 1
        else:
            result["missing_categories"].append(category)
            result["category_results"][category] = {
                "score": Decimal("0"),
                "status": "NON_COMPLIANT",
            }
            result["non_compliant_categories"] += 1

    result["overall_score"] = compute_compliance_score(
        {c: r["score"] for c, r in result["category_results"].items()}
    )
    result["determination"] = determine_compliance(
        result["overall_score"], compliant_threshold, partial_threshold,
    )
    result["provenance_hash"] = compute_test_hash({
        "country": country_code,
        "score": str(result["overall_score"]),
    })
    return result


def _generate_gap_analysis(
    compliance_result: Dict,
    country_rules: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Generate gap analysis from compliance check results."""
    gaps = []
    for category, cat_result in compliance_result["category_results"].items():
        if cat_result["status"] != "COMPLIANT":
            gap = {
                "category": category,
                "current_score": cat_result["score"],
                "target_score": Decimal("80"),
                "gap_size": Decimal("80") - cat_result["score"],
                "severity": "high" if cat_result["score"] < Decimal("50") else "moderate",
                "remediation": f"Improve {category} compliance to meet threshold",
            }
            gaps.append(gap)

    return {
        "supplier_id": compliance_result.get("supplier_id"),
        "country_code": compliance_result["country_code"],
        "gaps": gaps,
        "gap_count": len(gaps),
        "highest_priority_gap": min(gaps, key=lambda g: g["current_score"]) if gaps else None,
    }


def _calculate_evidence_sufficiency(
    documents_provided: int,
    documents_required: int,
    documents_verified: int,
    certifications_valid: int,
    audit_passed: bool,
) -> Decimal:
    """Calculate evidence sufficiency score (0-100)."""
    if documents_required == 0:
        return Decimal("100")

    presence_score = (documents_provided / documents_required) * 40
    verification_score = (documents_verified / max(documents_provided, 1)) * 30
    cert_score = min(certifications_valid * 10, 20)
    audit_score = 10 if audit_passed else 0

    total = presence_score + verification_score + cert_score + audit_score
    return Decimal(str(round(min(total, 100), 2)))


# ===========================================================================
# 1. Per-Country Rule Sets (27 tests)
# ===========================================================================


class TestPerCountryRuleSets:
    """Test country-specific compliance rule sets for all 27 countries."""

    @pytest.mark.parametrize("country_code", EUDR_COUNTRIES_27)
    def test_country_has_rule_set(self, country_code, sample_country_rules):
        """Test each of the 27 EUDR countries has a rule set."""
        assert country_code in sample_country_rules

    @pytest.mark.parametrize("country_code", EUDR_COUNTRIES_27[:5])
    def test_country_rules_cover_8_categories(self, country_code, sample_country_rules):
        """Test each country rule set covers all 8 legislation categories."""
        rules = sample_country_rules[country_code]["rules"]
        for category in LEGISLATION_CATEGORIES:
            assert category in rules, f"Country {country_code} missing rules for {category}"

    def test_high_risk_countries_require_primary_evidence(self, sample_country_rules):
        """Test high-risk countries require primary evidence level."""
        for country in HIGH_RISK_COUNTRIES:
            if country in sample_country_rules:
                rules = sample_country_rules[country]["rules"]
                for category in LEGISLATION_CATEGORIES:
                    assert rules[category]["evidence_level"] == "primary"

    def test_standard_countries_accept_secondary_evidence(self, sample_country_rules):
        """Test non-high-risk countries accept secondary evidence."""
        for country in EUDR_COUNTRIES_27:
            if country not in HIGH_RISK_COUNTRIES and country in sample_country_rules:
                rules = sample_country_rules[country]["rules"]
                for category in LEGISLATION_CATEGORIES:
                    assert rules[category]["evidence_level"] == "secondary"

    def test_country_rules_include_required_documents(self, sample_country_rules):
        """Test each rule specifies required documents."""
        for country in EUDR_COUNTRIES_27[:3]:
            rules = sample_country_rules[country]["rules"]
            for category in LEGISLATION_CATEGORIES:
                assert "required_documents" in rules[category]

    def test_country_overall_risk_level(self, sample_country_rules):
        """Test countries are assigned correct overall risk level."""
        for country in HIGH_RISK_COUNTRIES:
            if country in sample_country_rules:
                assert sample_country_rules[country]["overall_risk_level"] == "high"


# ===========================================================================
# 2. Eight-Category Compliance Checking (15 tests)
# ===========================================================================


class TestEightCategoryCompliance:
    """Test compliance checking across all 8 EUDR Article 2(40) categories."""

    def test_all_categories_compliant(self):
        """Test full compliance across all 8 categories."""
        scores = {cat: Decimal("90") for cat in LEGISLATION_CATEGORIES}
        result = _check_country_compliance("BR", scores)
        assert result["determination"] == "COMPLIANT"
        assert result["compliant_categories"] == 8
        assert result["non_compliant_categories"] == 0

    def test_all_categories_non_compliant(self):
        """Test full non-compliance across all 8 categories."""
        scores = {cat: Decimal("20") for cat in LEGISLATION_CATEGORIES}
        result = _check_country_compliance("BR", scores)
        assert result["determination"] == "NON_COMPLIANT"
        assert result["non_compliant_categories"] == 8

    def test_mixed_compliance(self):
        """Test mixed compliance (some pass, some fail)."""
        scores = {
            "land_use_rights": Decimal("85"),
            "environmental_protection": Decimal("90"),
            "forest_related_rules": Decimal("70"),
            "third_party_rights": Decimal("45"),
            "labour_rights": Decimal("60"),
            "tax_and_royalty": Decimal("95"),
            "trade_and_customs": Decimal("55"),
            "anti_corruption": Decimal("30"),
        }
        result = _check_country_compliance("BR", scores)
        assert result["compliant_categories"] >= 2
        assert result["non_compliant_categories"] >= 1

    @pytest.mark.parametrize("category", LEGISLATION_CATEGORIES)
    def test_single_category_compliant(self, category):
        """Test compliance when only one category passes."""
        scores = {cat: Decimal("20") for cat in LEGISLATION_CATEGORIES}
        scores[category] = Decimal("90")
        result = _check_country_compliance("BR", scores)
        assert result["category_results"][category]["status"] == "COMPLIANT"

    @pytest.mark.parametrize("category", LEGISLATION_CATEGORIES)
    def test_single_category_non_compliant(self, category):
        """Test non-compliance when only one category fails."""
        scores = {cat: Decimal("90") for cat in LEGISLATION_CATEGORIES}
        scores[category] = Decimal("20")
        result = _check_country_compliance("BR", scores)
        assert result["category_results"][category]["status"] == "NON_COMPLIANT"

    def test_missing_category_treated_as_non_compliant(self):
        """Test missing categories are treated as non-compliant."""
        scores = {cat: Decimal("90") for cat in LEGISLATION_CATEGORIES[:6]}
        # Missing last 2 categories
        result = _check_country_compliance("BR", scores)
        assert len(result["missing_categories"]) == 2
        assert result["non_compliant_categories"] >= 2


# ===========================================================================
# 3. Gap Analysis (12 tests)
# ===========================================================================


class TestGapAnalysis:
    """Test gap analysis generation."""

    def test_gap_analysis_no_gaps(self):
        """Test gap analysis when all categories are compliant."""
        scores = {cat: Decimal("90") for cat in LEGISLATION_CATEGORIES}
        compliance = _check_country_compliance("DK", scores)
        analysis = _generate_gap_analysis(compliance)
        assert analysis["gap_count"] == 0
        assert len(analysis["gaps"]) == 0

    def test_gap_analysis_all_gaps(self):
        """Test gap analysis when all categories have gaps."""
        scores = {cat: Decimal("40") for cat in LEGISLATION_CATEGORIES}
        compliance = _check_country_compliance("CD", scores)
        analysis = _generate_gap_analysis(compliance)
        assert analysis["gap_count"] == 8

    def test_gap_analysis_includes_remediation(self):
        """Test gap analysis includes remediation recommendations."""
        scores = {cat: Decimal("40") for cat in LEGISLATION_CATEGORIES}
        compliance = _check_country_compliance("BR", scores)
        analysis = _generate_gap_analysis(compliance)
        for gap in analysis["gaps"]:
            assert "remediation" in gap
            assert len(gap["remediation"]) > 0

    def test_gap_analysis_severity_classification(self):
        """Test gap severity is classified correctly."""
        scores = {
            "land_use_rights": Decimal("30"),       # NON_COMPLIANT -> high
            "environmental_protection": Decimal("60"),  # PARTIALLY -> moderate
            "forest_related_rules": Decimal("90"),   # COMPLIANT -> no gap
        }
        for cat in LEGISLATION_CATEGORIES:
            if cat not in scores:
                scores[cat] = Decimal("90")

        compliance = _check_country_compliance("BR", scores)
        analysis = _generate_gap_analysis(compliance)
        gap_severities = {g["category"]: g["severity"] for g in analysis["gaps"]}
        assert gap_severities.get("land_use_rights") == "high"
        assert gap_severities.get("environmental_protection") == "moderate"

    def test_gap_analysis_gap_size_calculation(self):
        """Test gap size is calculated correctly (target - current)."""
        scores = {"land_use_rights": Decimal("60")}
        for cat in LEGISLATION_CATEGORIES:
            if cat not in scores:
                scores[cat] = Decimal("90")
        compliance = _check_country_compliance("BR", scores)
        analysis = _generate_gap_analysis(compliance)
        lu_gap = next(g for g in analysis["gaps"] if g["category"] == "land_use_rights")
        assert lu_gap["gap_size"] == Decimal("20")

    def test_gap_analysis_highest_priority(self):
        """Test highest priority gap is the one with lowest score."""
        scores = {
            "land_use_rights": Decimal("70"),
            "anti_corruption": Decimal("20"),
            "labour_rights": Decimal("50"),
        }
        for cat in LEGISLATION_CATEGORIES:
            if cat not in scores:
                scores[cat] = Decimal("90")
        compliance = _check_country_compliance("BR", scores)
        analysis = _generate_gap_analysis(compliance)
        assert analysis["highest_priority_gap"]["category"] == "anti_corruption"

    def test_gap_analysis_fixture(self, sample_gap_analysis):
        """Test sample gap analysis fixture has expected structure."""
        assert sample_gap_analysis["determination"] == "PARTIALLY_COMPLIANT"
        assert len(sample_gap_analysis["gaps"]) >= 2
        assert sample_gap_analysis["overall_score"] == Decimal("62")

    def test_gap_analysis_country_code(self):
        """Test gap analysis includes correct country code."""
        scores = {cat: Decimal("40") for cat in LEGISLATION_CATEGORIES}
        compliance = _check_country_compliance("ID", scores)
        analysis = _generate_gap_analysis(compliance)
        assert analysis["country_code"] == "ID"

    def test_gap_analysis_empty_scores(self):
        """Test gap analysis with no scores (all categories missing)."""
        compliance = _check_country_compliance("BR", {})
        analysis = _generate_gap_analysis(compliance)
        assert analysis["gap_count"] == 8

    def test_gap_analysis_partial_scores(self):
        """Test gap analysis with partial category scores."""
        scores = {
            "land_use_rights": Decimal("85"),
            "environmental_protection": Decimal("55"),
        }
        compliance = _check_country_compliance("BR", scores)
        analysis = _generate_gap_analysis(compliance)
        # land_use_rights is COMPLIANT, no gap
        gap_categories = {g["category"] for g in analysis["gaps"]}
        assert "land_use_rights" not in gap_categories

    def test_gap_analysis_provenance(self, sample_gap_analysis):
        """Test gap analysis includes provenance hash."""
        assert "provenance_hash" in sample_gap_analysis
        assert len(sample_gap_analysis["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_gap_analysis_evidence_sufficiency(self, sample_gap_analysis):
        """Test gap analysis includes evidence sufficiency score."""
        assert "evidence_sufficiency" in sample_gap_analysis
        assert Decimal("0") <= sample_gap_analysis["evidence_sufficiency"] <= Decimal("100")


# ===========================================================================
# 4. Evidence Sufficiency Scoring (10 tests)
# ===========================================================================


class TestEvidenceSufficiency:
    """Test evidence sufficiency score calculation."""

    def test_perfect_evidence(self):
        """Test perfect evidence sufficiency score."""
        score = _calculate_evidence_sufficiency(
            documents_provided=10, documents_required=10,
            documents_verified=10, certifications_valid=2,
            audit_passed=True,
        )
        assert score == Decimal("100")

    def test_zero_evidence(self):
        """Test zero evidence sufficiency."""
        score = _calculate_evidence_sufficiency(
            documents_provided=0, documents_required=10,
            documents_verified=0, certifications_valid=0,
            audit_passed=False,
        )
        assert score == Decimal("0")

    def test_partial_evidence(self):
        """Test partial evidence provides proportional score."""
        score = _calculate_evidence_sufficiency(
            documents_provided=5, documents_required=10,
            documents_verified=3, certifications_valid=1,
            audit_passed=False,
        )
        assert Decimal("0") < score < Decimal("100")

    def test_no_documents_required(self):
        """Test evidence score is 100 when no documents required."""
        score = _calculate_evidence_sufficiency(
            documents_provided=0, documents_required=0,
            documents_verified=0, certifications_valid=0,
            audit_passed=False,
        )
        assert score == Decimal("100")

    def test_audit_passed_adds_bonus(self):
        """Test passing audit adds bonus to evidence score."""
        score_no_audit = _calculate_evidence_sufficiency(
            documents_provided=5, documents_required=10,
            documents_verified=5, certifications_valid=0,
            audit_passed=False,
        )
        score_audit = _calculate_evidence_sufficiency(
            documents_provided=5, documents_required=10,
            documents_verified=5, certifications_valid=0,
            audit_passed=True,
        )
        assert score_audit > score_no_audit

    def test_certifications_add_bonus(self):
        """Test valid certifications add bonus to evidence score."""
        score_no_cert = _calculate_evidence_sufficiency(
            documents_provided=5, documents_required=10,
            documents_verified=5, certifications_valid=0,
            audit_passed=False,
        )
        score_cert = _calculate_evidence_sufficiency(
            documents_provided=5, documents_required=10,
            documents_verified=5, certifications_valid=2,
            audit_passed=False,
        )
        assert score_cert > score_no_cert

    def test_certification_bonus_capped(self):
        """Test certification bonus is capped at maximum."""
        score_2 = _calculate_evidence_sufficiency(
            documents_provided=10, documents_required=10,
            documents_verified=10, certifications_valid=2,
            audit_passed=True,
        )
        score_10 = _calculate_evidence_sufficiency(
            documents_provided=10, documents_required=10,
            documents_verified=10, certifications_valid=10,
            audit_passed=True,
        )
        # Both should hit the cap (max 20 from certs)
        assert score_2 == score_10

    def test_score_capped_at_100(self):
        """Test evidence sufficiency never exceeds 100."""
        score = _calculate_evidence_sufficiency(
            documents_provided=20, documents_required=10,
            documents_verified=20, certifications_valid=5,
            audit_passed=True,
        )
        assert score <= Decimal("100")

    def test_verified_documents_matter(self):
        """Test more verified documents increase score."""
        score_low = _calculate_evidence_sufficiency(
            documents_provided=10, documents_required=10,
            documents_verified=2, certifications_valid=0,
            audit_passed=False,
        )
        score_high = _calculate_evidence_sufficiency(
            documents_provided=10, documents_required=10,
            documents_verified=10, certifications_valid=0,
            audit_passed=False,
        )
        assert score_high > score_low

    def test_evidence_score_type(self):
        """Test evidence sufficiency returns Decimal type."""
        score = _calculate_evidence_sufficiency(5, 10, 3, 1, False)
        assert isinstance(score, Decimal)


# ===========================================================================
# 5. Compliance Determination (10 tests)
# ===========================================================================


class TestComplianceDetermination:
    """Test compliance determination from overall scores."""

    @pytest.mark.parametrize("score,expected", [
        (Decimal("0"), "NON_COMPLIANT"),
        (Decimal("25"), "NON_COMPLIANT"),
        (Decimal("49"), "NON_COMPLIANT"),
        (Decimal("50"), "PARTIALLY_COMPLIANT"),
        (Decimal("65"), "PARTIALLY_COMPLIANT"),
        (Decimal("79"), "PARTIALLY_COMPLIANT"),
        (Decimal("80"), "COMPLIANT"),
        (Decimal("90"), "COMPLIANT"),
        (Decimal("100"), "COMPLIANT"),
    ])
    def test_determination_thresholds(self, score, expected):
        """Test compliance determination at threshold boundaries."""
        assert determine_compliance(score) == expected

    def test_determination_custom_thresholds(self):
        """Test compliance determination with custom thresholds."""
        assert determine_compliance(Decimal("85"), compliant=90, partial=40) == "PARTIALLY_COMPLIANT"
        assert determine_compliance(Decimal("95"), compliant=90, partial=40) == "COMPLIANT"
        assert determine_compliance(Decimal("30"), compliant=90, partial=40) == "NON_COMPLIANT"

    def test_all_determinations_valid(self):
        """Test all possible determinations are in valid set."""
        for score in range(0, 101, 10):
            result = determine_compliance(Decimal(str(score)))
            assert result in COMPLIANCE_DETERMINATIONS

    def test_provenance_hash_in_result(self):
        """Test compliance check result includes provenance hash."""
        scores = {cat: Decimal("75") for cat in LEGISLATION_CATEGORIES}
        result = _check_country_compliance("BR", scores)
        assert result["provenance_hash"] is not None
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH
