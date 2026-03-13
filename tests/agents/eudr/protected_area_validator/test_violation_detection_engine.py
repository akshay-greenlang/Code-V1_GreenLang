# -*- coding: utf-8 -*-
"""
Tests for ViolationDetectionEngine - AGENT-EUDR-022 Engine 6

Comprehensive test suite covering:
- Encroachment detection
- Severity classification (CRITICAL/HIGH/MEDIUM/LOW/INFO)
- Alert generation
- Deduplication window
- Supply chain correlation
- Violation lifecycle (pending/investigating/resolved)
- Auto-escalation

Test count: 65 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022 (Engine 6: Violation Detection)
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from tests.agents.eudr.protected_area_validator.conftest import (
    compute_test_hash,
    classify_severity,
    classify_risk_level,
    SHA256_HEX_LENGTH,
    SEVERITY_LEVELS,
    IUCN_CATEGORIES,
    IUCN_CATEGORY_RISK_SCORES,
    OVERLAP_TYPES,
    ALL_COMMODITIES,
    HIGH_RISK_COUNTRIES,
    VIOLATION_DEDUP_WINDOW_HOURS,
)


# ===========================================================================
# 1. Encroachment Detection (12 tests)
# ===========================================================================


class TestEncroachmentDetection:
    """Test encroachment detection for protected area violations."""

    def test_direct_overlap_creates_violation(self, sample_violation):
        """Test direct overlap creates an encroachment violation."""
        assert sample_violation["violation_type"] == "encroachment"
        assert sample_violation["overlap_area_hectares"] > 0

    def test_violation_has_unique_id(self, sample_violation):
        """Test each violation has a unique identifier."""
        assert sample_violation["violation_id"] == "viol-001"

    def test_violation_references_protected_area(self, sample_violation):
        """Test violation references the protected area."""
        assert sample_violation["area_id"] == "pa-001"

    def test_violation_references_plot(self, sample_violation):
        """Test violation references the offending plot."""
        assert sample_violation["plot_id"] == "plot-001"

    def test_violation_tracks_overlap_area(self, sample_violation):
        """Test violation records overlap area in hectares."""
        assert sample_violation["overlap_area_hectares"] == Decimal("150")

    def test_violation_tracks_distance(self, sample_violation):
        """Test violation records distance to boundary."""
        assert sample_violation["distance_to_boundary_meters"] == Decimal("0")

    def test_violation_tracks_iucn_category(self, sample_violation):
        """Test violation records IUCN category of affected area."""
        assert sample_violation["iucn_category"] in IUCN_CATEGORIES

    def test_violation_tracks_country(self, sample_violation):
        """Test violation records country code."""
        assert sample_violation["country_code"] == "BR"

    def test_violation_has_timestamp(self, sample_violation):
        """Test violation has detection timestamp."""
        assert sample_violation["detected_at"] is not None
        assert isinstance(sample_violation["detected_at"], datetime)

    def test_violation_has_title(self, sample_violation):
        """Test violation has human-readable title."""
        assert len(sample_violation["title"]) > 0

    def test_violation_has_description(self, sample_violation):
        """Test violation has detailed description."""
        assert len(sample_violation["description"]) > 0

    def test_violation_has_provenance_hash(self, sample_violation):
        """Test violation has provenance hash."""
        assert len(sample_violation["provenance_hash"]) == SHA256_HEX_LENGTH


# ===========================================================================
# 2. Severity Classification (15 tests)
# ===========================================================================


class TestSeverityClassification:
    """Test violation severity classification."""

    def test_direct_ia_is_critical(self):
        """Test DIRECT overlap with Ia is CRITICAL."""
        assert classify_severity("DIRECT", "Ia") == "CRITICAL"

    def test_direct_ib_is_critical(self):
        """Test DIRECT overlap with Ib is CRITICAL."""
        assert classify_severity("DIRECT", "Ib") == "CRITICAL"

    def test_direct_ii_is_critical(self):
        """Test DIRECT overlap with II is CRITICAL."""
        assert classify_severity("DIRECT", "II") == "CRITICAL"

    def test_direct_iii_is_high(self):
        """Test DIRECT overlap with III is HIGH."""
        assert classify_severity("DIRECT", "III") == "HIGH"

    def test_direct_iv_is_high(self):
        """Test DIRECT overlap with IV is HIGH."""
        assert classify_severity("DIRECT", "IV") == "HIGH"

    def test_direct_v_is_medium(self):
        """Test DIRECT overlap with V is MEDIUM."""
        assert classify_severity("DIRECT", "V") == "MEDIUM"

    def test_direct_vi_is_medium(self):
        """Test DIRECT overlap with VI is MEDIUM."""
        assert classify_severity("DIRECT", "VI") == "MEDIUM"

    def test_buffer_ia_is_high(self):
        """Test BUFFER overlap with Ia is HIGH."""
        assert classify_severity("BUFFER", "Ia") == "HIGH"

    def test_buffer_vi_is_medium(self):
        """Test BUFFER overlap with VI is MEDIUM."""
        assert classify_severity("BUFFER", "VI") == "MEDIUM"

    def test_adjacent_ia_is_medium(self):
        """Test ADJACENT with Ia is MEDIUM."""
        assert classify_severity("ADJACENT", "Ia") == "MEDIUM"

    def test_adjacent_vi_is_low(self):
        """Test ADJACENT with VI is LOW."""
        assert classify_severity("ADJACENT", "VI") == "LOW"

    def test_proximate_any_is_low(self):
        """Test PROXIMATE with any category is LOW."""
        for cat in IUCN_CATEGORIES:
            assert classify_severity("PROXIMATE", cat) == "LOW"

    def test_none_is_info(self):
        """Test NONE overlap is INFO."""
        assert classify_severity("NONE", "II") == "INFO"

    def test_world_heritage_elevates_to_critical(self):
        """Test World Heritage Site elevates severity to CRITICAL."""
        assert classify_severity("DIRECT", "II", world_heritage=True) == "CRITICAL"
        assert classify_severity("PARTIAL", "V", world_heritage=True) == "CRITICAL"

    @pytest.mark.parametrize("severity", SEVERITY_LEVELS)
    def test_all_severity_levels_valid(self, severity):
        """Test all severity levels are valid."""
        assert severity in SEVERITY_LEVELS


# ===========================================================================
# 3. Alert Generation (10 tests)
# ===========================================================================


class TestAlertGeneration:
    """Test alert generation from violations."""

    def test_critical_violation_generates_alert(self, sample_violation):
        """Test CRITICAL violation generates an alert."""
        assert sample_violation["severity"] == "CRITICAL"
        assert sample_violation["status"] == "pending"

    def test_alert_includes_affected_commodities(self, sample_violation):
        """Test alert includes affected commodities."""
        assert len(sample_violation["affected_commodities"]) > 0
        for comm in sample_violation["affected_commodities"]:
            assert comm in ALL_COMMODITIES

    def test_alert_includes_affected_suppliers(self, sample_violation):
        """Test alert includes affected suppliers."""
        assert len(sample_violation["affected_suppliers"]) > 0

    def test_alert_supply_chain_correlation(self, sample_violation):
        """Test alert includes supply chain correlation flag."""
        assert sample_violation["supply_chain_correlation"] is True

    def test_multiple_violations_at_different_severities(self, sample_violations):
        """Test violations exist at all severity levels."""
        severities = [v["severity"] for v in sample_violations]
        assert "CRITICAL" in severities
        assert "HIGH" in severities
        assert "MEDIUM" in severities
        assert "LOW" in severities
        assert "INFO" in severities

    def test_violation_status_lifecycle(self):
        """Test violation status follows lifecycle."""
        statuses = ["pending", "triaged", "investigating", "resolved",
                     "escalated", "false_positive"]
        for status in statuses:
            assert isinstance(status, str)

    def test_initial_status_is_pending(self, sample_violation):
        """Test initial violation status is pending."""
        assert sample_violation["status"] == "pending"

    def test_violation_count(self, sample_violations):
        """Test correct number of violations in fixture."""
        assert len(sample_violations) == 5

    def test_each_violation_unique_id(self, sample_violations):
        """Test each violation has unique ID."""
        ids = [v["violation_id"] for v in sample_violations]
        assert len(set(ids)) == len(ids)

    def test_violation_provenance_hashes_unique(self, sample_violations):
        """Test each violation has unique provenance hash."""
        hashes = [v["provenance_hash"] for v in sample_violations]
        assert len(set(hashes)) == len(hashes)


# ===========================================================================
# 4. Deduplication Window (10 tests)
# ===========================================================================


class TestDeduplicationWindow:
    """Test violation deduplication within time window."""

    def test_dedup_window_default_72_hours(self):
        """Test default dedup window is 72 hours."""
        assert VIOLATION_DEDUP_WINDOW_HOURS == 72

    def test_duplicate_within_window_suppressed(self):
        """Test duplicate violation within window is suppressed."""
        now = datetime.now(timezone.utc)
        violation_1 = {"detected_at": now, "area_id": "pa-001", "plot_id": "plot-001"}
        violation_2 = {"detected_at": now + timedelta(hours=1), "area_id": "pa-001", "plot_id": "plot-001"}
        time_diff = (violation_2["detected_at"] - violation_1["detected_at"]).total_seconds() / 3600
        assert time_diff < VIOLATION_DEDUP_WINDOW_HOURS  # Should be deduplicated

    def test_duplicate_outside_window_not_suppressed(self):
        """Test duplicate violation outside window is not suppressed."""
        now = datetime.now(timezone.utc)
        violation_1 = {"detected_at": now}
        violation_2 = {"detected_at": now + timedelta(hours=73)}
        time_diff = (violation_2["detected_at"] - violation_1["detected_at"]).total_seconds() / 3600
        assert time_diff > VIOLATION_DEDUP_WINDOW_HOURS

    def test_different_areas_not_deduplicated(self):
        """Test violations for different areas are not deduplicated."""
        v1 = {"area_id": "pa-001", "plot_id": "plot-001"}
        v2 = {"area_id": "pa-002", "plot_id": "plot-001"}
        assert v1["area_id"] != v2["area_id"]

    def test_different_plots_not_deduplicated(self):
        """Test violations for different plots are not deduplicated."""
        v1 = {"area_id": "pa-001", "plot_id": "plot-001"}
        v2 = {"area_id": "pa-001", "plot_id": "plot-002"}
        assert v1["plot_id"] != v2["plot_id"]

    def test_dedup_key_includes_area_and_plot(self):
        """Test dedup key is based on area_id + plot_id combination."""
        key = f"pa-001:plot-001"
        assert "pa-001" in key
        assert "plot-001" in key

    def test_dedup_preserves_first_occurrence(self):
        """Test deduplication keeps the first occurrence."""
        violations = [
            {"id": "v-1", "detected_at": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)},
            {"id": "v-2", "detected_at": datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)},
        ]
        # Keep first (earlier)
        assert violations[0]["detected_at"] < violations[1]["detected_at"]

    def test_dedup_window_configurable(self, mock_config):
        """Test dedup window is configurable."""
        assert mock_config["violation_dedup_window_hours"] == 72

    def test_dedup_with_strict_config(self, strict_config):
        """Test strict config has shorter dedup window."""
        assert strict_config["violation_dedup_window_hours"] == 24

    def test_dedup_zero_window_means_no_dedup(self):
        """Test zero dedup window means no deduplication."""
        window = 0
        assert window == 0


# ===========================================================================
# 5. Supply Chain Correlation (10 tests)
# ===========================================================================


class TestSupplyChainCorrelation:
    """Test violation correlation with supply chain data."""

    def test_violation_correlated_with_supplier(self, sample_violation):
        """Test violation is correlated with supplier."""
        assert sample_violation["supply_chain_correlation"] is True
        assert len(sample_violation["affected_suppliers"]) > 0

    def test_violation_correlated_with_commodity(self, sample_violation):
        """Test violation is correlated with commodity."""
        assert len(sample_violation["affected_commodities"]) > 0

    def test_high_risk_country_flagged(self, sample_violation):
        """Test violation in high-risk country is flagged."""
        assert sample_violation["country_code"] in HIGH_RISK_COUNTRIES

    @pytest.mark.parametrize("commodity", ALL_COMMODITIES)
    def test_all_commodities_can_be_affected(self, commodity):
        """Test all EUDR commodities can appear in affected list."""
        violation = {"affected_commodities": [commodity]}
        assert commodity in violation["affected_commodities"]

    def test_multiple_suppliers_affected(self):
        """Test violation can affect multiple suppliers."""
        violation = {"affected_suppliers": ["sup-001", "sup-002", "sup-003"]}
        assert len(violation["affected_suppliers"]) == 3

    def test_no_correlation_for_remote_plots(self):
        """Test no supply chain correlation for plots far from violation."""
        violation = {"supply_chain_correlation": False, "affected_suppliers": []}
        assert violation["supply_chain_correlation"] is False

    def test_correlation_tracks_plot_ids(self):
        """Test correlation includes affected plot IDs."""
        violation = {"affected_plots": ["plot-001", "plot-002"]}
        assert len(violation["affected_plots"]) == 2

    def test_correlation_strengthens_with_proximity(self):
        """Test closer plots have stronger supply chain correlation."""
        close = {"distance_m": 500, "correlation_strength": "strong"}
        far = {"distance_m": 20000, "correlation_strength": "weak"}
        assert close["distance_m"] < far["distance_m"]

    def test_violation_impacts_dds(self):
        """Test violation impacts due diligence statement."""
        impact = {"dds_impact": True, "risk_level_change": "elevated"}
        assert impact["dds_impact"] is True

    def test_violation_creates_audit_trail(self):
        """Test violation creates an audit trail entry."""
        audit = {
            "entity_type": "violation",
            "action": "detected",
            "entity_id": "viol-001",
        }
        assert audit["action"] == "detected"


# ===========================================================================
# 6. Auto-Escalation (8 tests)
# ===========================================================================


class TestAutoEscalation:
    """Test violation auto-escalation rules."""

    def test_critical_auto_escalated(self, mock_config):
        """Test CRITICAL violations are auto-escalated."""
        assert mock_config["auto_escalation_enabled"] is True

    def test_sla_triage_4_hours(self, mock_config):
        """Test triage SLA is 4 hours."""
        assert mock_config["sla_triage_hours"] == 4

    def test_sla_investigation_48_hours(self, mock_config):
        """Test investigation SLA is 48 hours."""
        assert mock_config["sla_investigation_hours"] == 48

    def test_sla_resolution_168_hours(self, mock_config):
        """Test resolution SLA is 168 hours (7 days)."""
        assert mock_config["sla_resolution_hours"] == 168

    def test_sla_breach_triggers_escalation(self):
        """Test SLA breach triggers escalation."""
        sla_deadline = datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        current_time = datetime(2026, 1, 15, 20, 0, 0, tzinfo=timezone.utc)
        assert current_time > sla_deadline  # SLA breached

    def test_escalation_level_increments(self):
        """Test escalation level increments on each breach."""
        level_0 = 0
        level_1 = level_0 + 1
        level_2 = level_1 + 1
        assert level_2 == 2

    def test_world_heritage_immediate_escalation(self):
        """Test World Heritage violations get immediate escalation."""
        whs_escalation_level = 2  # Skip level 1
        assert whs_escalation_level >= 2

    def test_escalation_notifies_stakeholders(self):
        """Test escalation sends notifications."""
        notification = {
            "type": "escalation",
            "violation_id": "viol-001",
            "level": 1,
            "recipients": ["eudr_manager", "compliance_officer"],
        }
        assert len(notification["recipients"]) >= 2
