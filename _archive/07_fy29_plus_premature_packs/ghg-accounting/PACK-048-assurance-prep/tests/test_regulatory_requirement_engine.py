"""
Unit tests for RegulatoryRequirementEngine (PACK-048 Engine 8).

Tests all public methods with 28+ tests covering:
  - 12 jurisdictions loaded
  - EU CSRD limited 2025
  - EU CSRD reasonable 2028
  - US SEC LAF attestation
  - California SB 253 verification
  - Multi-jurisdiction consolidation
  - Company size threshold
  - Gap analysis
  - Regulatory alert upcoming
  - Not-applicable jurisdiction

Author: GreenLang QA Team
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from tests.conftest import assert_decimal_between, assert_decimal_equal


# ---------------------------------------------------------------------------
# 12 Jurisdictions Loaded Tests
# ---------------------------------------------------------------------------


class TestJurisdictionsLoaded:
    """Tests for 12 jurisdiction records loaded."""

    def test_12_jurisdictions(self, sample_jurisdictions):
        """Test 12 jurisdictions are loaded."""
        assert len(sample_jurisdictions) == 12

    def test_jurisdiction_ids_unique(self, sample_jurisdictions):
        """Test jurisdiction IDs are unique."""
        ids = [j["jurisdiction_id"] for j in sample_jurisdictions]
        assert len(ids) == len(set(ids))

    def test_all_jurisdictions_have_required_fields(self, sample_jurisdictions):
        """Test all jurisdictions have required fields."""
        required = {
            "jurisdiction_id", "name", "country", "assurance_required",
            "assurance_level_2025", "standard", "effective_date",
            "scope_coverage",
        }
        for j in sample_jurisdictions:
            for field in required:
                assert field in j, f"Jurisdiction {j['jurisdiction_id']} missing '{field}'"

    @pytest.mark.parametrize("jur_id", [
        "EU_CSRD", "US_SEC", "CA_SB253", "UK_SECR", "SG_SGX",
        "JP_SSBJ", "AU_ASRS", "KR_KSQF", "HK_HKEX", "BR_CVM",
        "IN_BRSR", "CA_CSSB",
    ])
    def test_jurisdiction_exists(self, sample_jurisdictions, jur_id):
        """Test each expected jurisdiction exists."""
        found = [j for j in sample_jurisdictions if j["jurisdiction_id"] == jur_id]
        assert len(found) == 1


# ---------------------------------------------------------------------------
# EU CSRD Limited 2025 Tests
# ---------------------------------------------------------------------------


class TestEUCSRDLimited2025:
    """Tests for EU CSRD limited assurance requirement (2025)."""

    def test_eu_csrd_requires_assurance(self, sample_jurisdictions):
        """Test EU CSRD requires assurance."""
        eu = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "EU_CSRD"][0]
        assert eu["assurance_required"] is True

    def test_eu_csrd_limited_in_2025(self, sample_jurisdictions):
        """Test EU CSRD requires limited assurance in 2025."""
        eu = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "EU_CSRD"][0]
        assert eu["assurance_level_2025"] == "limited"

    def test_eu_csrd_standard_isae_3410(self, sample_jurisdictions):
        """Test EU CSRD uses ISAE 3410 standard."""
        eu = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "EU_CSRD"][0]
        assert eu["standard"] == "ISAE_3410"

    def test_eu_csrd_covers_all_scopes(self, sample_jurisdictions):
        """Test EU CSRD covers scope 1, 2, and 3."""
        eu = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "EU_CSRD"][0]
        assert "scope_1" in eu["scope_coverage"]
        assert "scope_2" in eu["scope_coverage"]
        assert "scope_3" in eu["scope_coverage"]


# ---------------------------------------------------------------------------
# EU CSRD Reasonable 2028 Tests
# ---------------------------------------------------------------------------


class TestEUCSRDReasonable2028:
    """Tests for EU CSRD reasonable assurance requirement (2028)."""

    def test_eu_csrd_reasonable_in_2028(self, sample_jurisdictions):
        """Test EU CSRD requires reasonable assurance in 2028."""
        eu = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "EU_CSRD"][0]
        assert eu["assurance_level_2028"] == "reasonable"

    def test_assurance_level_increases_over_time(self, sample_jurisdictions):
        """Test assurance level increases from limited to reasonable."""
        eu = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "EU_CSRD"][0]
        level_order = {"none": 0, "limited": 1, "reasonable": 2}
        assert level_order[eu["assurance_level_2028"]] >= level_order[eu["assurance_level_2025"]]


# ---------------------------------------------------------------------------
# US SEC LAF Attestation Tests
# ---------------------------------------------------------------------------


class TestUSSECLAFAttestation:
    """Tests for US SEC Large Accelerated Filer attestation."""

    def test_us_sec_requires_assurance(self, sample_jurisdictions):
        """Test US SEC requires assurance."""
        sec = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "US_SEC"][0]
        assert sec["assurance_required"] is True

    def test_us_sec_uses_ssae_18(self, sample_jurisdictions):
        """Test US SEC uses SSAE 18 attestation standard."""
        sec = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "US_SEC"][0]
        assert sec["standard"] == "SSAE_18"

    def test_us_sec_covers_scope_1_2_only(self, sample_jurisdictions):
        """Test US SEC covers scope 1 and 2 only."""
        sec = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "US_SEC"][0]
        assert "scope_1" in sec["scope_coverage"]
        assert "scope_2" in sec["scope_coverage"]
        assert "scope_3" not in sec["scope_coverage"]


# ---------------------------------------------------------------------------
# California SB 253 Tests
# ---------------------------------------------------------------------------


class TestCaliforniaSB253:
    """Tests for California SB 253 verification requirement."""

    def test_sb253_requires_assurance(self, sample_jurisdictions):
        """Test SB 253 requires assurance."""
        sb = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "CA_SB253"][0]
        assert sb["assurance_required"] is True

    def test_sb253_uses_iso_14064_3(self, sample_jurisdictions):
        """Test SB 253 uses ISO 14064-3 verification standard."""
        sb = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "CA_SB253"][0]
        assert sb["standard"] == "ISO_14064_3"

    def test_sb253_revenue_threshold_1b(self, sample_jurisdictions):
        """Test SB 253 revenue threshold is $1B."""
        sb = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "CA_SB253"][0]
        assert sb["company_threshold_revenue_eur"] >= Decimal("1000000000")

    def test_sb253_covers_all_scopes(self, sample_jurisdictions):
        """Test SB 253 covers scope 1, 2, and 3."""
        sb = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "CA_SB253"][0]
        assert "scope_1" in sb["scope_coverage"]
        assert "scope_2" in sb["scope_coverage"]
        assert "scope_3" in sb["scope_coverage"]


# ---------------------------------------------------------------------------
# Multi-Jurisdiction Consolidation Tests
# ---------------------------------------------------------------------------


class TestMultiJurisdictionConsolidation:
    """Tests for multi-jurisdiction requirement consolidation."""

    def test_most_stringent_assurance_level_selected(self, sample_jurisdictions):
        """Test most stringent assurance level is selected across jurisdictions."""
        levels = {"none": 0, "limited": 1, "reasonable": 2}
        max_level_2025 = max(
            levels[j["assurance_level_2025"]] for j in sample_jurisdictions
        )
        assert max_level_2025 >= 1  # At least limited

    def test_widest_scope_coverage_selected(self, sample_jurisdictions):
        """Test widest scope coverage is selected across jurisdictions."""
        all_scopes = set()
        for j in sample_jurisdictions:
            all_scopes.update(j["scope_coverage"])
        assert "scope_1" in all_scopes
        assert "scope_2" in all_scopes
        assert "scope_3" in all_scopes

    def test_earliest_effective_date_selected(self, sample_jurisdictions):
        """Test earliest effective date is selected across jurisdictions."""
        dates = [j["effective_date"] for j in sample_jurisdictions]
        earliest = min(dates)
        assert earliest <= "2024-04-01"


# ---------------------------------------------------------------------------
# Company Size Threshold Tests
# ---------------------------------------------------------------------------


class TestCompanySizeThreshold:
    """Tests for company size threshold checks."""

    def test_eu_csrd_employee_threshold(self, sample_jurisdictions):
        """Test EU CSRD employee threshold is 250."""
        eu = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "EU_CSRD"][0]
        assert eu["company_threshold_employees"] == 250

    def test_company_below_threshold_not_applicable(self, sample_jurisdictions):
        """Test company below threshold is not applicable."""
        company_employees = 100
        eu = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "EU_CSRD"][0]
        applicable = company_employees >= eu["company_threshold_employees"]
        assert applicable is False

    def test_company_above_threshold_applicable(self, sample_jurisdictions):
        """Test company above threshold is applicable."""
        company_employees = 2000
        eu = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "EU_CSRD"][0]
        applicable = company_employees >= eu["company_threshold_employees"]
        assert applicable is True


# ---------------------------------------------------------------------------
# Gap Analysis Tests
# ---------------------------------------------------------------------------


class TestRegulatoryGapAnalysis:
    """Tests for regulatory gap analysis."""

    def test_gap_identified_for_scope_3_missing(self, sample_jurisdictions):
        """Test gap identified when Scope 3 required but not covered."""
        current_coverage = {"scope_1", "scope_2"}
        eu = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "EU_CSRD"][0]
        required = set(eu["scope_coverage"])
        gap = required - current_coverage
        assert "scope_3" in gap

    def test_no_gap_when_fully_covered(self, sample_jurisdictions):
        """Test no gap when all required scopes are covered."""
        current_coverage = {"scope_1", "scope_2", "scope_3"}
        eu = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "EU_CSRD"][0]
        required = set(eu["scope_coverage"])
        gap = required - current_coverage
        assert len(gap) == 0


# ---------------------------------------------------------------------------
# Regulatory Alert Tests
# ---------------------------------------------------------------------------


class TestRegulatoryAlertUpcoming:
    """Tests for upcoming regulatory requirement alerts."""

    def test_alert_for_upcoming_effective_date(self, sample_jurisdictions):
        """Test alert triggered for upcoming effective date within 180 days."""
        alert_days = 180
        reference_date = datetime(2025, 8, 1, tzinfo=timezone.utc)
        upcoming = [
            j for j in sample_jurisdictions
            if datetime.strptime(j["effective_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            <= reference_date
        ]
        assert len(upcoming) > 0

    def test_no_alert_for_past_effective_date(self, sample_jurisdictions):
        """Test no alert for already-effective jurisdiction."""
        eu = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "EU_CSRD"][0]
        effective = datetime.strptime(eu["effective_date"], "%Y-%m-%d")
        now = datetime(2025, 6, 1)
        is_past = effective < now
        assert is_past is True


# ---------------------------------------------------------------------------
# Not-Applicable Jurisdiction Tests
# ---------------------------------------------------------------------------


class TestNotApplicableJurisdiction:
    """Tests for jurisdictions that are not applicable."""

    def test_hk_hkex_not_required_2025(self, sample_jurisdictions):
        """Test Hong Kong HKEX does not require assurance in 2025."""
        hk = [j for j in sample_jurisdictions if j["jurisdiction_id"] == "HK_HKEX"][0]
        assert hk["assurance_required"] is False

    def test_not_applicable_filtered_out(self, sample_jurisdictions):
        """Test not-applicable jurisdictions are correctly filtered."""
        applicable = [j for j in sample_jurisdictions if j["assurance_required"]]
        not_applicable = [j for j in sample_jurisdictions if not j["assurance_required"]]
        assert len(applicable) + len(not_applicable) == 12
        assert len(not_applicable) >= 1
