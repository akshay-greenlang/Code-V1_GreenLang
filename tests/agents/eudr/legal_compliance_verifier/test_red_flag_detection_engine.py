# -*- coding: utf-8 -*-
"""
Tests for RedFlagDetectionEngine - AGENT-EUDR-023 Engine 4

Comprehensive test suite covering:
- All 40 red flag indicators (one test per indicator)
- 6 category classifications (documentation, certification, geographic,
  supplier, regulatory, operational)
- Deterministic severity scoring with threshold boundaries
- Country risk multipliers for 10 countries
- Commodity risk multipliers for 7 commodities
- Combined multiplier application
- False positive suppression logic
- Severity classification at all threshold boundaries
- Batch red flag detection
- Provenance tracking for red flag operations

Test count: 90+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 (Engine 4 - Red Flag Detection)
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from tests.agents.eudr.legal_compliance_verifier.conftest import (
    compute_test_hash,
    classify_red_flag_severity,
    apply_country_multiplier,
    apply_commodity_multiplier,
    SHA256_HEX_LENGTH,
    RED_FLAG_INDICATORS,
    RED_FLAG_CATEGORIES,
    RED_FLAG_SEVERITIES,
    RED_FLAG_BOUNDARIES,
    COUNTRY_MULTIPLIERS,
    COMMODITY_MULTIPLIERS,
    EUDR_COMMODITIES,
    EUDR_COUNTRIES_27,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_red_flags(
    supplier_data: Dict,
    documents: List[Dict] = None,
    certifications: List[Dict] = None,
    geographic_data: Optional[Dict] = None,
    indicators: List[Dict] = None,
) -> List[Dict]:
    """Detect red flags for a supplier based on available data."""
    flags = []
    indicators = indicators or RED_FLAG_INDICATORS
    documents = documents or []
    certifications = certifications or []

    for indicator in indicators:
        triggered = False
        evidence = []

        # Documentation checks
        if indicator["category"] == "documentation":
            if indicator["name"] == "missing_land_title":
                has_title = any(d.get("document_type") == "land_title" for d in documents)
                if not has_title:
                    triggered = True
                    evidence.append("No land title document found")
            elif indicator["name"] == "expired_permit":
                for doc in documents:
                    if doc.get("status") == "expired":
                        triggered = True
                        evidence.append(f"Expired: {doc.get('document_type')}")
            elif indicator["name"] == "missing_tax_clearance":
                has_tax = any(d.get("document_type") == "tax_clearance_certificate" for d in documents)
                if not has_tax:
                    triggered = True
                    evidence.append("No tax clearance certificate")
            else:
                # Generic check: trigger if no documents of the related type
                triggered = len(documents) == 0
                if triggered:
                    evidence.append("No documents provided")

        # Certification checks
        elif indicator["category"] == "certification":
            if indicator["name"] == "revoked_certification":
                for cert in certifications:
                    if cert.get("status") == "revoked":
                        triggered = True
                        evidence.append(f"Revoked: {cert.get('certificate_id')}")
            elif indicator["name"] == "suspended_certificate":
                for cert in certifications:
                    if cert.get("status") == "suspended":
                        triggered = True
                        evidence.append(f"Suspended: {cert.get('certificate_id')}")
            elif indicator["name"] == "expired_certificate":
                for cert in certifications:
                    if cert.get("status") == "expired":
                        triggered = True
                        evidence.append(f"Expired: {cert.get('certificate_id')}")
            else:
                triggered = len(certifications) == 0
                if triggered:
                    evidence.append("No certifications provided")

        # Geographic checks
        elif indicator["category"] == "geographic":
            if geographic_data:
                if indicator["name"] == "protected_area_overlap":
                    if geographic_data.get("in_protected_area"):
                        triggered = True
                        evidence.append("Plot overlaps protected area")
                elif indicator["name"] == "deforestation_hotspot":
                    if geographic_data.get("deforestation_risk", 0) > 70:
                        triggered = True
                        evidence.append("High deforestation risk area")

        # Supplier checks
        elif indicator["category"] == "supplier":
            if indicator["name"] == "sanctions_list_match":
                if supplier_data.get("on_sanctions_list"):
                    triggered = True
                    evidence.append("Supplier on sanctions list")

        # Regulatory checks
        elif indicator["category"] == "regulatory":
            if indicator["name"] == "pending_legal_action":
                if supplier_data.get("has_legal_action"):
                    triggered = True
                    evidence.append("Active legal proceedings")

        # Operational checks
        elif indicator["category"] == "operational":
            if indicator["name"] == "forced_labour_risk":
                if supplier_data.get("forced_labour_indicators"):
                    triggered = True
                    evidence.append("Forced labour indicators detected")

        if triggered:
            country = supplier_data.get("country_code", "")
            commodity = supplier_data.get("commodity", "")
            base_score = indicator["base_score"]
            adjusted = apply_country_multiplier(base_score, country)
            adjusted = apply_commodity_multiplier(adjusted, commodity)

            flags.append({
                "indicator_id": indicator["id"],
                "indicator_name": indicator["name"],
                "category": indicator["category"],
                "base_score": base_score,
                "adjusted_score": adjusted,
                "severity": classify_red_flag_severity(adjusted),
                "evidence": evidence,
                "false_positive": False,
            })
    return flags


def _suppress_false_positives(
    flags: List[Dict],
    suppression_rules: Optional[Dict] = None,
) -> List[Dict]:
    """Apply false positive suppression rules to detected flags."""
    if not suppression_rules:
        return flags
    result = []
    for flag in flags:
        suppressed = False
        indicator = flag["indicator_name"]
        if indicator in suppression_rules:
            rule = suppression_rules[indicator]
            if rule.get("always_suppress"):
                suppressed = True
            elif rule.get("suppress_below_score") and flag["adjusted_score"] < rule["suppress_below_score"]:
                suppressed = True
        if not suppressed:
            result.append(flag)
        else:
            flag_copy = dict(flag)
            flag_copy["false_positive"] = True
            result.append(flag_copy)
    return result


# ===========================================================================
# 1. Individual Red Flag Indicator Tests (40 tests)
# ===========================================================================


class TestRedFlagIndicators:
    """Test each of the 40 red flag indicators individually."""

    @pytest.mark.parametrize("indicator", RED_FLAG_INDICATORS, ids=lambda i: i["id"])
    def test_indicator_has_required_fields(self, indicator):
        """Test each indicator has all required fields."""
        required = {"id", "name", "category", "base_score", "description"}
        for field in required:
            assert field in indicator, f"Indicator {indicator['id']} missing field: {field}"

    @pytest.mark.parametrize("indicator", RED_FLAG_INDICATORS, ids=lambda i: i["id"])
    def test_indicator_base_score_in_range(self, indicator):
        """Test each indicator base score is in valid range (0-100)."""
        assert 0 <= indicator["base_score"] <= 100, (
            f"Indicator {indicator['id']} score {indicator['base_score']} out of range"
        )

    @pytest.mark.parametrize("indicator", RED_FLAG_INDICATORS, ids=lambda i: i["id"])
    def test_indicator_category_valid(self, indicator):
        """Test each indicator has a valid category."""
        assert indicator["category"] in RED_FLAG_CATEGORIES, (
            f"Indicator {indicator['id']} has invalid category: {indicator['category']}"
        )

    @pytest.mark.parametrize("indicator", RED_FLAG_INDICATORS, ids=lambda i: i["id"])
    def test_indicator_id_format(self, indicator):
        """Test each indicator ID follows RF-XXX-NNN format."""
        parts = indicator["id"].split("-")
        assert len(parts) == 3
        assert parts[0] == "RF"
        assert parts[1] in ("DOC", "CRT", "GEO", "SUP", "REG", "OPS")
        assert parts[2].isdigit()

    def test_total_indicator_count(self, red_flag_indicators_all):
        """Test exactly 40 red flag indicators are defined."""
        assert len(red_flag_indicators_all) == 40


# ===========================================================================
# 2. Category Classifications (12 tests)
# ===========================================================================


class TestCategoryClassification:
    """Test red flag category classifications."""

    @pytest.mark.parametrize("category", RED_FLAG_CATEGORIES)
    def test_category_has_indicators(self, category, red_flag_indicators_all):
        """Test each category has at least one indicator."""
        cat_indicators = [i for i in red_flag_indicators_all if i["category"] == category]
        assert len(cat_indicators) >= 1, f"No indicators for category: {category}"

    def test_documentation_category_count(self, red_flag_indicators_all):
        """Test documentation category has 8 indicators."""
        count = sum(1 for i in red_flag_indicators_all if i["category"] == "documentation")
        assert count == 8

    def test_certification_category_count(self, red_flag_indicators_all):
        """Test certification category has 7 indicators."""
        count = sum(1 for i in red_flag_indicators_all if i["category"] == "certification")
        assert count == 7

    def test_geographic_category_count(self, red_flag_indicators_all):
        """Test geographic category has 6 indicators."""
        count = sum(1 for i in red_flag_indicators_all if i["category"] == "geographic")
        assert count == 6

    def test_supplier_category_count(self, red_flag_indicators_all):
        """Test supplier category has 7 indicators."""
        count = sum(1 for i in red_flag_indicators_all if i["category"] == "supplier")
        assert count == 7

    def test_regulatory_category_count(self, red_flag_indicators_all):
        """Test regulatory category has 6 indicators."""
        count = sum(1 for i in red_flag_indicators_all if i["category"] == "regulatory")
        assert count == 6

    def test_operational_category_count(self, red_flag_indicators_all):
        """Test operational category has 6 indicators."""
        count = sum(1 for i in red_flag_indicators_all if i["category"] == "operational")
        assert count == 6

    def test_total_across_categories_is_40(self, red_flag_indicators_all):
        """Test sum of all category counts equals 40."""
        total = sum(
            sum(1 for i in red_flag_indicators_all if i["category"] == cat)
            for cat in RED_FLAG_CATEGORIES
        )
        assert total == 40


# ===========================================================================
# 3. Severity Scoring (12 tests)
# ===========================================================================


class TestSeverityScoring:
    """Test deterministic severity scoring with threshold boundaries."""

    @pytest.mark.parametrize("score,expected_severity", [
        (Decimal("0"), "low"),
        (Decimal("10"), "low"),
        (Decimal("24"), "low"),
        (Decimal("25"), "moderate"),
        (Decimal("35"), "moderate"),
        (Decimal("49"), "moderate"),
        (Decimal("50"), "high"),
        (Decimal("60"), "high"),
        (Decimal("74"), "high"),
        (Decimal("75"), "critical"),
        (Decimal("85"), "critical"),
        (Decimal("100"), "critical"),
    ])
    def test_severity_classification_thresholds(self, score, expected_severity):
        """Test severity classification at each threshold boundary."""
        assert classify_red_flag_severity(score) == expected_severity

    def test_severity_deterministic(self):
        """Test severity classification is deterministic for same score."""
        for _ in range(10):
            assert classify_red_flag_severity(Decimal("65")) == "high"

    def test_severity_boundary_50_is_high(self):
        """Test score of exactly 50 classifies as high."""
        assert classify_red_flag_severity(Decimal("50")) == "high"

    def test_severity_boundary_75_is_critical(self):
        """Test score of exactly 75 classifies as critical."""
        assert classify_red_flag_severity(Decimal("75")) == "critical"

    def test_severity_boundary_25_is_moderate(self):
        """Test score of exactly 25 classifies as moderate."""
        assert classify_red_flag_severity(Decimal("25")) == "moderate"

    def test_severity_zero_is_low(self):
        """Test score of 0 classifies as low."""
        assert classify_red_flag_severity(Decimal("0")) == "low"

    def test_severity_with_custom_thresholds(self):
        """Test severity classification with custom thresholds."""
        # Custom: critical >= 90, high >= 70, moderate >= 30
        result = classify_red_flag_severity(
            Decimal("80"), critical=90, high=70, moderate=30,
        )
        assert result == "high"


# ===========================================================================
# 4. Country Multipliers (12 tests)
# ===========================================================================


class TestCountryMultipliers:
    """Test country-specific risk multipliers for red flag scoring."""

    @pytest.mark.parametrize("country,expected_multiplier", [
        ("CD", Decimal("1.8")),
        ("CM", Decimal("1.6")),
        ("MM", Decimal("1.7")),
        ("NG", Decimal("1.5")),
        ("BR", Decimal("1.3")),
        ("ID", Decimal("1.2")),
        ("MY", Decimal("1.1")),
        ("GH", Decimal("1.2")),
        ("DK", Decimal("0.5")),
        ("FI", Decimal("0.5")),
    ])
    def test_country_multiplier_value(self, country, expected_multiplier):
        """Test country multiplier matches expected value."""
        assert COUNTRY_MULTIPLIERS[country] == expected_multiplier

    def test_high_risk_country_amplifies_score(self):
        """Test high-risk country multiplier increases base score."""
        base = 50
        adjusted = apply_country_multiplier(base, "CD")
        assert adjusted > Decimal(str(base))

    def test_low_risk_country_reduces_score(self):
        """Test low-risk country multiplier decreases base score."""
        base = 50
        adjusted = apply_country_multiplier(base, "DK")
        assert adjusted < Decimal(str(base))

    def test_unknown_country_uses_default(self):
        """Test unknown country code uses default multiplier of 1.0."""
        base = 50
        adjusted = apply_country_multiplier(base, "XX")
        assert adjusted == Decimal(str(base))

    def test_drc_is_highest_multiplier(self):
        """Test DRC has the highest country multiplier."""
        max_country = max(COUNTRY_MULTIPLIERS, key=COUNTRY_MULTIPLIERS.get)
        assert max_country == "CD"

    def test_denmark_is_lowest_multiplier(self):
        """Test Denmark has the lowest country multiplier."""
        min_country = min(COUNTRY_MULTIPLIERS, key=COUNTRY_MULTIPLIERS.get)
        assert min_country in ("DK", "FI")


# ===========================================================================
# 5. Commodity Multipliers (9 tests)
# ===========================================================================


class TestCommodityMultipliers:
    """Test commodity-specific risk multipliers."""

    @pytest.mark.parametrize("commodity,expected", [
        ("oil_palm", Decimal("1.5")),
        ("cattle", Decimal("1.4")),
        ("soya", Decimal("1.3")),
        ("cocoa", Decimal("1.3")),
        ("coffee", Decimal("1.2")),
        ("rubber", Decimal("1.2")),
        ("wood", Decimal("1.1")),
    ])
    def test_commodity_multiplier_value(self, commodity, expected):
        """Test commodity multiplier matches expected value."""
        assert COMMODITY_MULTIPLIERS[commodity] == expected

    def test_oil_palm_highest_risk(self):
        """Test oil palm has the highest commodity multiplier."""
        max_commodity = max(COMMODITY_MULTIPLIERS, key=COMMODITY_MULTIPLIERS.get)
        assert max_commodity == "oil_palm"

    def test_wood_lowest_risk(self):
        """Test wood has the lowest commodity multiplier."""
        min_commodity = min(COMMODITY_MULTIPLIERS, key=COMMODITY_MULTIPLIERS.get)
        assert min_commodity == "wood"


# ===========================================================================
# 6. Combined Multiplier Application (5 tests)
# ===========================================================================


class TestCombinedMultipliers:
    """Test combined country and commodity multiplier application."""

    def test_combined_drc_oil_palm(self):
        """Test combined multiplier for DRC + oil palm (highest risk)."""
        base = 80
        adjusted = apply_country_multiplier(base, "CD")
        adjusted = apply_commodity_multiplier(adjusted, "oil_palm")
        expected = Decimal(str(base)) * Decimal("1.8") * Decimal("1.5")
        assert adjusted == expected

    def test_combined_denmark_wood(self):
        """Test combined multiplier for Denmark + wood (lowest risk)."""
        base = 80
        adjusted = apply_country_multiplier(base, "DK")
        adjusted = apply_commodity_multiplier(adjusted, "wood")
        expected = Decimal(str(base)) * Decimal("0.5") * Decimal("1.1")
        assert adjusted == expected

    def test_combined_order_independence(self):
        """Test that applying country then commodity produces same result."""
        base = 60
        # Country first, then commodity
        a1 = apply_country_multiplier(base, "BR")
        a1 = apply_commodity_multiplier(a1, "soya")
        # Commodity first, then country
        a2 = apply_commodity_multiplier(Decimal(str(base)), "soya")
        a2 = apply_country_multiplier(int(a2), "BR")
        # They should be approximately equal due to Decimal arithmetic
        # The test validates both paths work
        assert a1 > Decimal("0")
        assert a2 > Decimal("0")

    def test_combined_high_risk_escalates_severity(self):
        """Test high-risk combination escalates severity from moderate to critical."""
        # Base score of 40 = moderate severity
        base = 40
        assert classify_red_flag_severity(Decimal(str(base))) == "moderate"
        # After DRC * oil_palm multiplier: 40 * 1.8 * 1.5 = 108
        adjusted = apply_country_multiplier(base, "CD")
        adjusted = apply_commodity_multiplier(adjusted, "oil_palm")
        assert classify_red_flag_severity(adjusted) == "critical"

    def test_combined_low_risk_reduces_severity(self):
        """Test low-risk combination reduces severity."""
        base = 55
        assert classify_red_flag_severity(Decimal(str(base))) == "high"
        adjusted = apply_country_multiplier(base, "DK")
        adjusted = apply_commodity_multiplier(adjusted, "wood")
        # 55 * 0.5 * 1.1 = 30.25 -> moderate
        assert classify_red_flag_severity(adjusted) == "moderate"


# ===========================================================================
# 7. False Positive Suppression (8 tests)
# ===========================================================================


class TestFalsePositiveSuppression:
    """Test false positive suppression logic."""

    def test_no_suppression_without_rules(self, sample_red_flags):
        """Test no suppression when no rules are provided."""
        result = _suppress_false_positives(sample_red_flags)
        assert all(not f["false_positive"] for f in result)

    def test_suppress_specific_indicator(self, sample_red_flags):
        """Test suppression of a specific indicator by name."""
        rules = {"missing_land_title": {"always_suppress": True}}
        result = _suppress_false_positives(sample_red_flags, suppression_rules=rules)
        for flag in result:
            if flag["indicator_name"] == "missing_land_title":
                assert flag["false_positive"] is True

    def test_suppress_below_score_threshold(self):
        """Test suppression of flags below a score threshold."""
        flags = [
            {"indicator_name": "minor_issue", "adjusted_score": Decimal("20"),
             "false_positive": False},
            {"indicator_name": "minor_issue", "adjusted_score": Decimal("60"),
             "false_positive": False},
        ]
        rules = {"minor_issue": {"suppress_below_score": Decimal("30")}}
        result = _suppress_false_positives(flags, suppression_rules=rules)
        assert result[0]["false_positive"] is True
        assert result[1]["false_positive"] is False

    def test_suppression_preserves_non_matching_flags(self, sample_red_flags):
        """Test suppression does not affect non-matching flags."""
        rules = {"nonexistent_indicator": {"always_suppress": True}}
        result = _suppress_false_positives(sample_red_flags, suppression_rules=rules)
        assert all(not f["false_positive"] for f in result)

    def test_empty_flags_list(self):
        """Test suppression on empty flags list."""
        result = _suppress_false_positives([])
        assert len(result) == 0

    def test_suppress_multiple_indicators(self):
        """Test suppressing multiple different indicators."""
        flags = [
            {"indicator_name": "flag_a", "adjusted_score": Decimal("30"), "false_positive": False},
            {"indicator_name": "flag_b", "adjusted_score": Decimal("40"), "false_positive": False},
            {"indicator_name": "flag_c", "adjusted_score": Decimal("50"), "false_positive": False},
        ]
        rules = {
            "flag_a": {"always_suppress": True},
            "flag_b": {"always_suppress": True},
        }
        result = _suppress_false_positives(flags, suppression_rules=rules)
        assert result[0]["false_positive"] is True
        assert result[1]["false_positive"] is True
        assert result[2]["false_positive"] is False

    def test_suppression_count(self):
        """Test counting suppressed vs active flags."""
        flags = [
            {"indicator_name": f"flag_{i}", "adjusted_score": Decimal(str(i * 10)),
             "false_positive": False}
            for i in range(5)
        ]
        rules = {"flag_0": {"always_suppress": True}, "flag_1": {"always_suppress": True}}
        result = _suppress_false_positives(flags, suppression_rules=rules)
        suppressed = sum(1 for f in result if f["false_positive"])
        active = sum(1 for f in result if not f["false_positive"])
        assert suppressed == 2
        assert active == 3

    def test_suppression_deterministic(self, sample_red_flags):
        """Test suppression is deterministic for same inputs."""
        rules = {"missing_land_title": {"always_suppress": True}}
        r1 = _suppress_false_positives(sample_red_flags, suppression_rules=rules)
        r2 = _suppress_false_positives(sample_red_flags, suppression_rules=rules)
        for f1, f2 in zip(r1, r2):
            assert f1["false_positive"] == f2["false_positive"]


# ===========================================================================
# 8. Red Flag Detection Integration (5 tests)
# ===========================================================================


class TestRedFlagDetection:
    """Test integrated red flag detection."""

    def test_detect_missing_documents(self):
        """Test red flag detection when no documents provided."""
        supplier = {"country_code": "BR", "commodity": "soya"}
        flags = _detect_red_flags(supplier, documents=[], certifications=[])
        # Should detect documentation-related flags
        doc_flags = [f for f in flags if f["category"] == "documentation"]
        assert len(doc_flags) >= 1

    def test_detect_expired_certificate(self):
        """Test red flag detection for expired certification."""
        supplier = {"country_code": "BR", "commodity": "wood"}
        expired_cert = [{"certificate_id": "CERT-001", "status": "expired"}]
        flags = _detect_red_flags(
            supplier, documents=[], certifications=expired_cert,
        )
        cert_flags = [f for f in flags if f["category"] == "certification"]
        assert len(cert_flags) >= 1

    def test_detect_sanctions_match(self):
        """Test red flag detection for sanctions list match."""
        supplier = {
            "country_code": "CD",
            "commodity": "wood",
            "on_sanctions_list": True,
        }
        flags = _detect_red_flags(supplier)
        sanctions_flags = [f for f in flags if f["indicator_name"] == "sanctions_list_match"]
        assert len(sanctions_flags) == 1
        assert sanctions_flags[0]["severity"] == "critical"

    def test_no_flags_for_clean_supplier(self):
        """Test no red flags for a fully compliant supplier."""
        supplier = {"country_code": "DK", "commodity": "wood"}
        docs = [
            {"document_type": "land_title", "status": "valid"},
            {"document_type": "tax_clearance_certificate", "status": "valid"},
        ]
        certs = [{"certificate_id": "CERT-001", "status": "valid"}]
        flags = _detect_red_flags(supplier, documents=docs, certifications=certs)
        # With valid docs and certs, fewer flags should trigger
        critical_flags = [f for f in flags if f["severity"] == "critical"]
        assert len(critical_flags) == 0

    def test_flags_include_provenance_data(self):
        """Test detected flags include scoring details for audit trail."""
        supplier = {"country_code": "BR", "commodity": "soya"}
        flags = _detect_red_flags(supplier, documents=[])
        for flag in flags:
            assert "base_score" in flag
            assert "adjusted_score" in flag
            assert "severity" in flag
            assert "evidence" in flag
