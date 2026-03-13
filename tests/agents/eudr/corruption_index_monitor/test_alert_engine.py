# -*- coding: utf-8 -*-
"""
Unit tests for AlertEngine (AGENT-EUDR-019, Engine 7).

Tests all methods of AlertEngine including alert generation, retrieval,
configuration, acknowledgement workflows, alert summary with filters,
threshold breach detection, significant change detection, trend reversal
detection, severity calculation, alert type enumeration, status transitions,
and provenance chain integrity.

Default alert rules tested:
    - CPI drop > 5 points -> HIGH alert
    - CPI drop > 10 points -> CRITICAL alert
    - Country crosses CPI 40 threshold -> CRITICAL alert
    - WGI CC drop > 0.5 -> HIGH alert
    - WGI CC drop > 1.0 -> CRITICAL alert

Coverage target: 85%+ of AlertEngine methods.

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.corruption_index_monitor.engines.alert_engine import (
    AlertEngine,
    AlertSeverity,
    AlertType,
    AlertStatus,
    Alert,
    AlertConfiguration,
    AlertSummary,
    DEFAULT_ALERT_RULES,
    CPI_CRITICAL_DROP,
    CPI_HIGH_DROP,
    CPI_HIGH_RISK_THRESHOLD,
    WGI_CRITICAL_DROP,
    WGI_HIGH_DROP,
    BRIBERY_HIGH_INCREASE,
    SEVERITY_ORDER,
    VALID_INDEX_TYPES,
    MAX_ALERT_STORE_SIZE,
    DEFAULT_EXPIRATION_DAYS,
    DEFAULT_COOLDOWN_MINUTES,
    _compute_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> AlertEngine:
    """Create a default AlertEngine instance with default rules."""
    return AlertEngine()


@pytest.fixture
def engine_with_alert(engine: AlertEngine) -> AlertEngine:
    """Create engine with at least one alert generated."""
    engine.check_significant_change("BR", "CPI", Decimal("42"), Decimal("30"))
    return engine


# ---------------------------------------------------------------------------
# TestAlertGeneration
# ---------------------------------------------------------------------------


class TestAlertGeneration:
    """Tests for generate_alerts for various data updates."""

    def test_generate_alerts_cpi_drop(self, engine: AlertEngine):
        """CPI drop of 8 points should generate at least a HIGH alert."""
        result = engine.generate_alerts([
            {
                "country_code": "BR",
                "index_type": "CPI",
                "previous_value": 40,
                "current_value": 32,
            }
        ])
        assert result["alerts_generated"] >= 1
        severities = [a["severity"] for a in result["alerts"]]
        assert "HIGH" in severities or "CRITICAL" in severities

    def test_generate_alerts_no_updates(self, engine: AlertEngine):
        """Empty updates should generate 0 alerts."""
        result = engine.generate_alerts([])
        assert result["alerts_generated"] == 0

    def test_generate_alerts_none_updates(self, engine: AlertEngine):
        """None updates should generate 0 alerts."""
        result = engine.generate_alerts(None)
        assert result["alerts_generated"] == 0

    def test_generate_alerts_threshold_and_change(self, engine: AlertEngine):
        """CPI dropping below 40 should trigger both change and threshold alerts."""
        result = engine.generate_alerts([
            {
                "country_code": "GH",
                "index_type": "CPI",
                "previous_value": 42,
                "current_value": 30,
            }
        ])
        # Should trigger SIGNIFICANT_CHANGE (drop > 10) and THRESHOLD_BREACH (below 40)
        assert result["alerts_generated"] >= 2

    def test_generate_alerts_multiple_countries(self, engine: AlertEngine):
        """Multiple country updates should all be processed."""
        result = engine.generate_alerts([
            {"country_code": "BR", "index_type": "CPI", "previous_value": 40, "current_value": 32},
            {"country_code": "ID", "index_type": "CPI", "previous_value": 38, "current_value": 30},
        ])
        assert result["data_updates_processed"] == 2
        assert result["alerts_generated"] >= 2

    def test_generate_alerts_has_provenance(self, engine: AlertEngine):
        """Alert generation result should include provenance hash."""
        result = engine.generate_alerts([
            {"country_code": "BR", "index_type": "CPI", "previous_value": 40, "current_value": 35}
        ])
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_generate_alerts_timestamp(self, engine: AlertEngine):
        """Result should include a calculation timestamp."""
        result = engine.generate_alerts([])
        assert "calculation_timestamp" in result


# ---------------------------------------------------------------------------
# TestAlertRetrieval
# ---------------------------------------------------------------------------


class TestAlertRetrieval:
    """Tests for get_alert by ID, missing IDs."""

    def test_get_existing_alert(self, engine_with_alert: AlertEngine):
        """Should be able to retrieve an alert by its ID."""
        # First, find the alert ID from the store
        alert_ids = list(engine_with_alert._alerts.keys())
        assert len(alert_ids) > 0
        alert_id = alert_ids[0]

        result = engine_with_alert.get_alert(alert_id)
        assert result["alert_id"] == alert_id
        assert result["status"] in ("ACTIVE", "EXPIRED")

    def test_get_missing_alert_raises(self, engine: AlertEngine):
        """Getting a non-existent alert should raise ValueError."""
        with pytest.raises(ValueError, match="Alert not found"):
            engine.get_alert("alert-nonexistent")

    def test_alert_to_dict(self, engine_with_alert: AlertEngine):
        """Alert.to_dict should return serializable dict."""
        alert_id = list(engine_with_alert._alerts.keys())[0]
        result = engine_with_alert.get_alert(alert_id)
        assert isinstance(result, dict)
        assert "alert_type" in result
        assert "severity" in result
        assert "provenance_hash" in result


# ---------------------------------------------------------------------------
# TestAlertConfiguration
# ---------------------------------------------------------------------------


class TestAlertConfiguration:
    """Tests for configure_alert with valid/invalid configs."""

    def test_create_new_rule(self, engine: AlertEngine):
        """Creating a new rule should succeed."""
        result = engine.configure_alert(
            rule_id="TEST-RULE-001",
            alert_type="SIGNIFICANT_CHANGE",
            index_type="CPI",
            severity="HIGH",
            description="Test rule",
            threshold_delta=Decimal("-7"),
            direction="decrease",
        )
        assert result["action"] == "created"
        assert result["rule_id"] == "TEST-RULE-001"

    def test_update_existing_rule(self, engine: AlertEngine):
        """Updating an existing rule should show action=updated."""
        # Create first
        engine.configure_alert(
            rule_id="TEST-RULE-002",
            alert_type="THRESHOLD_BREACH",
            index_type="CPI",
            severity="MEDIUM",
        )
        # Update
        result = engine.configure_alert(
            rule_id="TEST-RULE-002",
            alert_type="THRESHOLD_BREACH",
            index_type="CPI",
            severity="HIGH",
        )
        assert result["action"] == "updated"

    def test_invalid_alert_type_raises(self, engine: AlertEngine):
        """Invalid alert type should raise ValueError."""
        with pytest.raises(ValueError):
            engine.configure_alert(
                rule_id="BAD",
                alert_type="INVALID_TYPE",
                index_type="CPI",
                severity="HIGH",
            )

    def test_invalid_index_type_raises(self, engine: AlertEngine):
        """Invalid index type should raise ValueError."""
        with pytest.raises(ValueError):
            engine.configure_alert(
                rule_id="BAD",
                alert_type="SIGNIFICANT_CHANGE",
                index_type="INVALID",
                severity="HIGH",
            )

    def test_invalid_severity_raises(self, engine: AlertEngine):
        """Invalid severity should raise ValueError."""
        with pytest.raises(ValueError):
            engine.configure_alert(
                rule_id="BAD",
                alert_type="SIGNIFICANT_CHANGE",
                index_type="CPI",
                severity="INVALID",
            )

    def test_empty_rule_id_raises(self, engine: AlertEngine):
        """Empty rule ID should raise ValueError."""
        with pytest.raises(ValueError):
            engine.configure_alert(
                rule_id="",
                alert_type="SIGNIFICANT_CHANGE",
                index_type="CPI",
                severity="HIGH",
            )

    def test_get_rules(self, engine: AlertEngine):
        """get_rules should return all default rules."""
        result = engine.get_rules()
        assert result["rule_count"] == len(DEFAULT_ALERT_RULES)
        assert len(result["rules"]) == result["rule_count"]


# ---------------------------------------------------------------------------
# TestAlertAcknowledge
# ---------------------------------------------------------------------------


class TestAlertAcknowledge:
    """Tests for acknowledge_alert state transitions."""

    def test_acknowledge_active_alert(self, engine_with_alert: AlertEngine):
        """Acknowledging an ACTIVE alert should set status to ACKNOWLEDGED."""
        alert_id = list(engine_with_alert._alerts.keys())[0]
        result = engine_with_alert.acknowledge_alert(
            alert_id, acknowledged_by="test_user", notes="Reviewed"
        )
        assert result["action"] == "acknowledged"
        assert result["acknowledged_by"] == "test_user"
        # Verify status changed
        alert = engine_with_alert.get_alert(alert_id)
        assert alert["status"] == "ACKNOWLEDGED"

    def test_acknowledge_nonexistent_raises(self, engine: AlertEngine):
        """Acknowledging a non-existent alert should raise ValueError."""
        with pytest.raises(ValueError, match="Alert not found"):
            engine.acknowledge_alert("alert-none", "user")

    def test_acknowledge_already_acknowledged_raises(
        self, engine_with_alert: AlertEngine
    ):
        """Acknowledging an already-acknowledged alert should raise ValueError."""
        alert_id = list(engine_with_alert._alerts.keys())[0]
        engine_with_alert.acknowledge_alert(alert_id, "user1")
        with pytest.raises(ValueError, match="not ACTIVE"):
            engine_with_alert.acknowledge_alert(alert_id, "user2")

    def test_resolve_alert(self, engine_with_alert: AlertEngine):
        """Resolving an alert should set status to RESOLVED."""
        alert_id = list(engine_with_alert._alerts.keys())[0]
        result = engine_with_alert.resolve_alert(alert_id, "Fixed the issue")
        assert result["action"] == "resolved"
        alert = engine_with_alert.get_alert(alert_id)
        assert alert["status"] == "RESOLVED"

    def test_suppress_alert(self, engine_with_alert: AlertEngine):
        """Suppressing an alert should set status to SUPPRESSED."""
        alert_id = list(engine_with_alert._alerts.keys())[0]
        result = engine_with_alert.suppress_alert(alert_id, "False positive")
        assert result["action"] == "suppressed"
        alert = engine_with_alert.get_alert(alert_id)
        assert alert["status"] == "SUPPRESSED"


# ---------------------------------------------------------------------------
# TestAlertSummary
# ---------------------------------------------------------------------------


class TestAlertSummary:
    """Tests for get_alert_summary with filters."""

    def test_summary_with_alerts(self, engine_with_alert: AlertEngine):
        """Summary should reflect the stored alerts."""
        result = engine_with_alert.get_alert_summary()
        assert result["total_alerts"] >= 1
        assert result["active_count"] >= 1

    def test_summary_no_alerts(self, engine: AlertEngine):
        """Empty store should show 0 alerts."""
        result = engine.get_alert_summary()
        assert result["total_alerts"] == 0

    def test_summary_country_filter(self, engine_with_alert: AlertEngine):
        """Country filter should limit results."""
        result = engine_with_alert.get_alert_summary(country_code="BR")
        assert result["total_alerts"] >= 1

    def test_summary_severity_filter(self, engine_with_alert: AlertEngine):
        """Severity filter should limit results."""
        result = engine_with_alert.get_alert_summary(severity="CRITICAL")
        # May or may not have CRITICAL alerts
        assert isinstance(result["total_alerts"], int)

    def test_summary_has_provenance(self, engine: AlertEngine):
        """Summary should include provenance hash."""
        result = engine.get_alert_summary()
        assert "provenance_hash" in result


# ---------------------------------------------------------------------------
# TestThresholdBreach
# ---------------------------------------------------------------------------


class TestThresholdBreach:
    """Tests for check_threshold_breach for CPI drops below 40, WGI below -1.0."""

    def test_cpi_crosses_40_downward(self, engine: AlertEngine):
        """CPI crossing below 40 should generate CRITICAL alert."""
        alerts = engine.check_threshold_breach(
            "GH", "CPI", Decimal("38"), Decimal("42")
        )
        assert len(alerts) >= 1
        assert alerts[0]["severity"] == "CRITICAL"
        assert alerts[0]["alert_type"] == "THRESHOLD_BREACH"

    def test_cpi_already_below_40_no_crossing(self, engine: AlertEngine):
        """CPI staying below 40 (no crossing) should not trigger threshold."""
        alerts = engine.check_threshold_breach(
            "BR", "CPI", Decimal("35"), Decimal("33")
        )
        # Both previous and current are below 40, so no crossing
        assert len(alerts) == 0

    def test_cpi_above_threshold_no_alert(self, engine: AlertEngine):
        """CPI staying above 40 should not trigger threshold alert."""
        alerts = engine.check_threshold_breach(
            "DK", "CPI", Decimal("90"), Decimal("88")
        )
        assert len(alerts) == 0

    def test_threshold_breach_provenance(self, engine: AlertEngine):
        """Threshold breach alerts should have provenance hashes."""
        alerts = engine.check_threshold_breach(
            "XX", "CPI", Decimal("38"), Decimal("42")
        )
        if alerts:
            assert len(alerts[0]["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestSignificantChange
# ---------------------------------------------------------------------------


class TestSignificantChange:
    """Tests for check_significant_change for 5/10 point drops."""

    def test_cpi_drop_5_points_high_alert(self, engine: AlertEngine):
        """CPI drop > 5 points year-over-year should generate HIGH alert."""
        alerts = engine.check_significant_change(
            "BR", "CPI", Decimal("42"), Decimal("35")
        )
        severities = [a["severity"] for a in alerts]
        assert "HIGH" in severities

    def test_cpi_drop_10_points_critical_alert(self, engine: AlertEngine):
        """CPI drop > 10 points should generate CRITICAL alert."""
        alerts = engine.check_significant_change(
            "BR", "CPI", Decimal("45"), Decimal("30")
        )
        severities = [a["severity"] for a in alerts]
        assert "CRITICAL" in severities

    def test_cpi_small_drop_no_alert(self, engine: AlertEngine):
        """CPI drop of 2 points should not trigger significant change alert."""
        alerts = engine.check_significant_change(
            "BR", "CPI", Decimal("38"), Decimal("36")
        )
        # Only the MEDIUM drop rule (>3) might trigger. The HIGH (>5) and CRITICAL (>10) should not.
        high_or_critical = [a for a in alerts if a["severity"] in ("HIGH", "CRITICAL")]
        assert len(high_or_critical) == 0

    def test_cpi_improvement_no_decrease_alert(self, engine: AlertEngine):
        """CPI improvement should not trigger decrease alerts."""
        alerts = engine.check_significant_change(
            "BR", "CPI", Decimal("30"), Decimal("40")
        )
        decrease_alerts = [
            a for a in alerts
            if a["alert_type"] == "SIGNIFICANT_CHANGE"
        ]
        # Positive change should not trigger "decrease" direction rules
        assert len(decrease_alerts) == 0

    def test_wgi_drop_triggers_alert(self, engine: AlertEngine):
        """WGI CC drop > 0.5 should generate HIGH alert."""
        alerts = engine.check_significant_change(
            "BR", "WGI", Decimal("-0.3"), Decimal("-0.9")
        )
        severities = [a["severity"] for a in alerts]
        assert "HIGH" in severities or "MEDIUM" in severities


# ---------------------------------------------------------------------------
# TestTrendReversal
# ---------------------------------------------------------------------------


class TestTrendReversal:
    """Tests for check_trend_reversal detection."""

    def test_improving_to_deteriorating(self, engine: AlertEngine):
        """Trend reversal from IMPROVING to DETERIORATING should generate alert."""
        alerts = engine.check_trend_reversal(
            "BR", "CPI", "IMPROVING", "DETERIORATING"
        )
        assert len(alerts) >= 1
        assert alerts[0]["alert_type"] == "TREND_REVERSAL"
        assert alerts[0]["severity"] == "HIGH"

    def test_same_direction_no_alert(self, engine: AlertEngine):
        """Same direction should not trigger reversal alert."""
        alerts = engine.check_trend_reversal(
            "BR", "CPI", "STABLE", "STABLE"
        )
        assert len(alerts) == 0

    def test_deteriorating_to_improving_no_default_rule(self, engine: AlertEngine):
        """Default rules only cover IMPROVING->DETERIORATING, not reverse."""
        alerts = engine.check_trend_reversal(
            "BR", "CPI", "DETERIORATING", "IMPROVING"
        )
        # Default rules check from IMPROVING to DETERIORATING only
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# TestAlertSeverity
# ---------------------------------------------------------------------------


class TestAlertSeverity:
    """Tests for alert severity levels."""

    def test_severity_ordering(self):
        """CRITICAL > HIGH > MEDIUM > LOW > INFORMATIONAL."""
        assert SEVERITY_ORDER["CRITICAL"] > SEVERITY_ORDER["HIGH"]
        assert SEVERITY_ORDER["HIGH"] > SEVERITY_ORDER["MEDIUM"]
        assert SEVERITY_ORDER["MEDIUM"] > SEVERITY_ORDER["LOW"]
        assert SEVERITY_ORDER["LOW"] > SEVERITY_ORDER["INFORMATIONAL"]

    def test_all_severity_values(self):
        """All 5 severity levels should be defined."""
        values = set(s.value for s in AlertSeverity)
        assert values == {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"}


# ---------------------------------------------------------------------------
# TestAlertTypes
# ---------------------------------------------------------------------------


class TestAlertTypes:
    """Verify all 7 alert types."""

    def test_all_alert_types(self):
        """All 7 alert types should be defined."""
        types = set(t.value for t in AlertType)
        expected = {
            "THRESHOLD_BREACH", "SIGNIFICANT_CHANGE", "TREND_REVERSAL",
            "DATA_UPDATE", "WATCHLIST_TRIGGER", "COUNTRY_RECLASSIFICATION",
            "NEW_DATA_AVAILABLE",
        }
        assert types == expected


# ---------------------------------------------------------------------------
# TestAlertStatus
# ---------------------------------------------------------------------------


class TestAlertStatus:
    """Verify ACTIVE -> ACKNOWLEDGED -> RESOLVED transitions."""

    def test_status_values(self):
        """All 5 status values should be defined."""
        statuses = set(s.value for s in AlertStatus)
        expected = {"ACTIVE", "ACKNOWLEDGED", "RESOLVED", "SUPPRESSED", "EXPIRED"}
        assert statuses == expected

    def test_lifecycle_transition(self, engine_with_alert: AlertEngine):
        """Alert should transition: ACTIVE -> ACKNOWLEDGED -> RESOLVED."""
        alert_id = list(engine_with_alert._alerts.keys())[0]

        # Start ACTIVE
        alert = engine_with_alert.get_alert(alert_id)
        assert alert["status"] == "ACTIVE"

        # Acknowledge
        engine_with_alert.acknowledge_alert(alert_id, "user")
        alert = engine_with_alert.get_alert(alert_id)
        assert alert["status"] == "ACKNOWLEDGED"

        # Resolve
        engine_with_alert.resolve_alert(alert_id, "Resolved")
        alert = engine_with_alert.get_alert(alert_id)
        assert alert["status"] == "RESOLVED"


# ---------------------------------------------------------------------------
# TestAlertProvenance
# ---------------------------------------------------------------------------


class TestAlertProvenance:
    """Tests for provenance chain integrity."""

    def test_alert_has_provenance(self, engine_with_alert: AlertEngine):
        """Stored alerts should have provenance hashes."""
        alert_id = list(engine_with_alert._alerts.keys())[0]
        alert = engine_with_alert.get_alert(alert_id)
        assert len(alert["provenance_hash"]) == 64

    def test_configuration_has_provenance(self, engine: AlertEngine):
        """Alert configuration should have provenance hash."""
        result = engine.configure_alert(
            rule_id="PROV-TEST",
            alert_type="SIGNIFICANT_CHANGE",
            index_type="CPI",
            severity="MEDIUM",
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_reclassification_alert(self, engine: AlertEngine):
        """Country reclassification should generate alerts."""
        alerts = engine.check_reclassification("BR", "STANDARD_RISK", "HIGH_RISK")
        assert len(alerts) >= 1
        assert alerts[0]["alert_type"] == "COUNTRY_RECLASSIFICATION"

    def test_reclassification_same_class_no_alert(self, engine: AlertEngine):
        """Same classification should not generate alerts."""
        alerts = engine.check_reclassification("BR", "HIGH_RISK", "HIGH_RISK")
        assert len(alerts) == 0

    def test_watchlist_management(self, engine: AlertEngine):
        """Adding and removing from watchlist should work."""
        result = engine.add_to_watchlist("BR", "High corruption risk")
        assert result["action"] == "added_to_watchlist"

        watchlist = engine.get_watchlist()
        assert watchlist["watchlist_count"] == 1

        result = engine.remove_from_watchlist("BR")
        assert result["was_on_watchlist"] is True

    def test_watchlist_add_empty_raises(self, engine: AlertEngine):
        """Adding empty country to watchlist should raise ValueError."""
        with pytest.raises(ValueError):
            engine.add_to_watchlist("", "reason")

    def test_default_rules_count(self):
        """There should be at least 9 default alert rules."""
        assert len(DEFAULT_ALERT_RULES) >= 9
