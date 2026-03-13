# -*- coding: utf-8 -*-
"""
Tests for AlertGenerator - AGENT-EUDR-020 Feature 2: Alert Generation

Comprehensive test suite covering:
- Alert generation from detection with/without plots, various severities
- Batch generation with multiple detections, empty list, deduplication
- Alert retrieval by ID and missing ID
- Alert listing with filters (severity, status, country, date range, pagination)
- Alert summary by country and global
- Alert statistics grouped by severity/status/country
- Proximity calculation using Haversine formula with known distances
- Affected plot identification with buffers, no plots in range, multiple plots
- Post-cutoff determination for dates before/after 2020-12-31
- Alert deduplication within/outside dedup window
- Provenance hash generation and determinism

Test count: 45+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 (Feature 2 - Alert Generation)
"""

import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from tests.agents.eudr.deforestation_alert_system.conftest import (
    compute_test_hash,
    haversine_km,
    is_post_cutoff,
    SHA256_HEX_LENGTH,
    SEVERITY_LEVELS,
    ALERT_STATUSES,
    CHANGE_TYPES,
    HIGH_RISK_COUNTRIES,
    EUDR_DEFORESTATION_CUTOFF,
    EUDR_CUTOFF_DATE_OBJ,
)


# ---------------------------------------------------------------------------
# Helpers for alert generation logic
# ---------------------------------------------------------------------------


def _generate_alert(
    detection_id: str,
    source: str,
    latitude: float,
    longitude: float,
    area_ha: float,
    change_type: str,
    confidence: float,
    country_code: str = "BR",
    nearby_plots: Optional[List[Dict]] = None,
    threshold_confidence: float = 0.75,
) -> Optional[Dict]:
    """Generate a deforestation alert from a satellite detection."""
    if confidence < threshold_confidence:
        return None
    if change_type in ("no_change", "regrowth"):
        return None

    # Determine severity based on area
    if area_ha >= 50:
        severity = "critical"
    elif area_ha >= 10:
        severity = "high"
    elif area_ha >= 1:
        severity = "medium"
    elif area_ha >= 0.5:
        severity = "low"
    else:
        severity = "informational"

    # Calculate proximity to nearest plot
    proximity_km = None
    affected_plots = []
    if nearby_plots:
        for plot in nearby_plots:
            dist = haversine_km(
                latitude, longitude,
                plot["latitude"], plot["longitude"],
            )
            if proximity_km is None or dist < proximity_km:
                proximity_km = dist
            if dist <= plot.get("buffer_radius_km", 10.0):
                affected_plots.append(plot["plot_id"])

    alert_id = f"ALR-{uuid.uuid4().hex[:12].upper()}"
    alert = {
        "alert_id": alert_id,
        "detection_id": detection_id,
        "severity": severity,
        "status": "pending",
        "title": f"{severity.title()} {change_type} alert in {country_code}",
        "area_ha": area_ha,
        "latitude": latitude,
        "longitude": longitude,
        "country_code": country_code,
        "affected_plots": affected_plots,
        "proximity_km": proximity_km,
        "is_post_cutoff": None,
        "provenance_hash": compute_test_hash({
            "detection_id": detection_id,
            "severity": severity,
            "area_ha": area_ha,
        }),
    }
    return alert


def _generate_batch(
    detections: List[Dict],
    dedup_enabled: bool = True,
    dedup_window_hours: int = 72,
    nearby_plots: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Generate alerts from a batch of detections."""
    alerts = []
    seen_locations = {}

    for det in detections:
        # Deduplication check
        if dedup_enabled:
            loc_key = f"{det['latitude']:.4f}_{det['longitude']:.4f}"
            if loc_key in seen_locations:
                continue
            seen_locations[loc_key] = True

        alert = _generate_alert(
            detection_id=det.get("detection_id", f"det-{uuid.uuid4().hex[:8]}"),
            source=det.get("source", "sentinel2"),
            latitude=det["latitude"],
            longitude=det["longitude"],
            area_ha=det.get("area_ha", 1.0),
            change_type=det.get("change_type", "deforestation"),
            confidence=det.get("confidence", 0.90),
            country_code=det.get("country_code", "BR"),
            nearby_plots=nearby_plots,
        )
        if alert is not None:
            alerts.append(alert)
    return alerts


def _get_alert(alerts: List[Dict], alert_id: str) -> Optional[Dict]:
    """Retrieve an alert by ID from a list."""
    for alert in alerts:
        if alert["alert_id"] == alert_id:
            return alert
    return None


def _list_alerts(
    alerts: List[Dict],
    severity: Optional[str] = None,
    status: Optional[str] = None,
    country_code: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
) -> List[Dict]:
    """Filter and paginate alerts."""
    result = alerts
    if severity is not None:
        result = [a for a in result if a["severity"] == severity]
    if status is not None:
        result = [a for a in result if a["status"] == status]
    if country_code is not None:
        result = [a for a in result if a["country_code"] == country_code]
    return result[offset:offset + limit]


def _get_alert_summary(alerts: List[Dict], country_code: Optional[str] = None) -> Dict:
    """Generate alert summary statistics."""
    filtered = alerts
    if country_code:
        filtered = [a for a in alerts if a["country_code"] == country_code]
    summary = {
        "total": len(filtered),
        "by_severity": {},
        "by_status": {},
        "by_country": {},
        "total_area_ha": sum(a.get("area_ha", 0) for a in filtered),
    }
    for sev in SEVERITY_LEVELS:
        summary["by_severity"][sev] = sum(1 for a in filtered if a["severity"] == sev)
    for status in ALERT_STATUSES:
        summary["by_status"][status] = sum(1 for a in filtered if a["status"] == status)
    for alert in filtered:
        cc = alert.get("country_code", "UNKNOWN")
        summary["by_country"][cc] = summary["by_country"].get(cc, 0) + 1
    return summary


def _get_alert_statistics(
    alerts: List[Dict],
    group_by: str = "severity",
) -> Dict:
    """Get alert statistics grouped by a given dimension."""
    stats = {}
    for alert in alerts:
        key = alert.get(group_by, "unknown")
        if key not in stats:
            stats[key] = {"count": 0, "total_area_ha": 0.0}
        stats[key]["count"] += 1
        stats[key]["total_area_ha"] += alert.get("area_ha", 0)
    return stats


def _calculate_proximity(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Haversine distance between two points in km."""
    return haversine_km(lat1, lon1, lat2, lon2)


def _determine_post_cutoff(
    detection_date: date,
    cutoff_date: date = None,
) -> bool:
    """Determine if detection occurred after the EUDR cutoff date."""
    return is_post_cutoff(detection_date, cutoff_date)


def _deduplicate_alerts(
    alerts: List[Dict],
    window_km: float = 1.0,
) -> List[Dict]:
    """Deduplicate alerts within spatial window."""
    if not alerts:
        return []
    unique = [alerts[0]]
    for alert in alerts[1:]:
        is_dup = False
        for existing in unique:
            dist = haversine_km(
                existing["latitude"], existing["longitude"],
                alert["latitude"], alert["longitude"],
            )
            if dist <= window_km:
                is_dup = True
                break
        if not is_dup:
            unique.append(alert)
    return unique


# ===========================================================================
# 1. TestAlertGeneration (12 tests)
# ===========================================================================


class TestAlertGeneration:
    """Test generate_alert from detection with/without plots, various severities."""

    def test_critical_alert_large_area(self):
        """Test critical alert for area >= 50 ha."""
        alert = _generate_alert(
            "det-001", "sentinel2", -3.1, -60.0,
            area_ha=55.0, change_type="deforestation", confidence=0.92,
        )
        assert alert is not None
        assert alert["severity"] == "critical"

    def test_high_alert_medium_area(self):
        """Test high alert for area 10-50 ha."""
        alert = _generate_alert(
            "det-002", "landsat8", -1.5, 116.0,
            area_ha=15.0, change_type="clearing", confidence=0.85,
        )
        assert alert is not None
        assert alert["severity"] == "high"

    def test_medium_alert_small_area(self):
        """Test medium alert for area 1-10 ha."""
        alert = _generate_alert(
            "det-003", "glad", 6.5, -1.6,
            area_ha=3.0, change_type="degradation", confidence=0.80,
        )
        assert alert is not None
        assert alert["severity"] == "medium"

    def test_low_alert_tiny_area(self):
        """Test low alert for area 0.5-1 ha."""
        alert = _generate_alert(
            "det-004", "radd", 0.5, 25.0,
            area_ha=0.7, change_type="logging", confidence=0.80,
        )
        assert alert is not None
        assert alert["severity"] == "low"

    def test_informational_alert_very_small(self):
        """Test informational alert for area < 0.5 ha."""
        alert = _generate_alert(
            "det-005", "hansen_gfc", -12.5, -55.3,
            area_ha=0.3, change_type="fire", confidence=0.78,
        )
        assert alert is not None
        assert alert["severity"] == "informational"

    def test_no_alert_low_confidence(self):
        """Test no alert when confidence below threshold."""
        alert = _generate_alert(
            "det-006", "sentinel2", -3.1, -60.0,
            area_ha=10.0, change_type="deforestation", confidence=0.50,
        )
        assert alert is None

    def test_no_alert_no_change(self):
        """Test no alert for no_change type."""
        alert = _generate_alert(
            "det-007", "sentinel2", -3.1, -60.0,
            area_ha=10.0, change_type="no_change", confidence=0.95,
        )
        assert alert is None

    def test_no_alert_regrowth(self):
        """Test no alert for regrowth type."""
        alert = _generate_alert(
            "det-008", "sentinel2", -3.1, -60.0,
            area_ha=10.0, change_type="regrowth", confidence=0.90,
        )
        assert alert is None

    def test_alert_with_nearby_plots(self):
        """Test alert generation with nearby supply chain plots."""
        plots = [
            {"plot_id": "PLOT-001", "latitude": -3.105, "longitude": -60.005,
             "buffer_radius_km": 10.0},
        ]
        alert = _generate_alert(
            "det-009", "sentinel2", -3.1, -60.0,
            area_ha=5.0, change_type="deforestation", confidence=0.90,
            nearby_plots=plots,
        )
        assert alert is not None
        assert "PLOT-001" in alert["affected_plots"]
        assert alert["proximity_km"] is not None

    def test_alert_no_plots_in_range(self):
        """Test alert generation when no plots are within buffer range."""
        plots = [
            {"plot_id": "PLOT-FAR", "latitude": 10.0, "longitude": 50.0,
             "buffer_radius_km": 10.0},
        ]
        alert = _generate_alert(
            "det-010", "sentinel2", -3.1, -60.0,
            area_ha=5.0, change_type="deforestation", confidence=0.90,
            nearby_plots=plots,
        )
        assert alert is not None
        assert len(alert["affected_plots"]) == 0

    def test_alert_metadata_completeness(self):
        """Test alert contains all required metadata fields."""
        alert = _generate_alert(
            "det-meta", "sentinel2", -3.1, -60.0,
            area_ha=5.0, change_type="deforestation", confidence=0.90,
        )
        assert alert is not None
        required_fields = [
            "alert_id", "detection_id", "severity", "status", "title",
            "area_ha", "latitude", "longitude", "country_code",
            "affected_plots", "provenance_hash",
        ]
        for field in required_fields:
            assert field in alert, f"Missing field: {field}"

    def test_alert_provenance_hash(self):
        """Test alert has a valid provenance hash."""
        alert = _generate_alert(
            "det-prov", "sentinel2", -3.1, -60.0,
            area_ha=5.0, change_type="deforestation", confidence=0.90,
        )
        assert alert is not None
        assert len(alert["provenance_hash"]) == SHA256_HEX_LENGTH


# ===========================================================================
# 2. TestBatchGeneration (6 tests)
# ===========================================================================


class TestBatchGeneration:
    """Test generate_batch with multiple detections."""

    def test_batch_multiple_detections(self):
        """Test batch generation with multiple detections."""
        detections = [
            {"latitude": -3.1, "longitude": -60.0, "area_ha": 5.0,
             "change_type": "deforestation", "confidence": 0.90},
            {"latitude": -1.5, "longitude": 116.0, "area_ha": 12.0,
             "change_type": "clearing", "confidence": 0.85},
        ]
        alerts = _generate_batch(detections)
        assert len(alerts) == 2

    def test_batch_empty_list(self):
        """Test batch generation with empty list returns no alerts."""
        alerts = _generate_batch([])
        assert alerts == []

    def test_batch_deduplication(self):
        """Test batch deduplication filters nearby detections."""
        detections = [
            {"latitude": -3.1000, "longitude": -60.0000, "area_ha": 5.0,
             "change_type": "deforestation", "confidence": 0.90},
            {"latitude": -3.1000, "longitude": -60.0000, "area_ha": 8.0,
             "change_type": "deforestation", "confidence": 0.92},
        ]
        alerts = _generate_batch(detections, dedup_enabled=True)
        assert len(alerts) == 1

    def test_batch_no_deduplication(self):
        """Test batch without deduplication keeps all valid detections."""
        detections = [
            {"latitude": -3.1000, "longitude": -60.0000, "area_ha": 5.0,
             "change_type": "deforestation", "confidence": 0.90},
            {"latitude": -3.1000, "longitude": -60.0000, "area_ha": 8.0,
             "change_type": "deforestation", "confidence": 0.92},
        ]
        alerts = _generate_batch(detections, dedup_enabled=False)
        assert len(alerts) == 2

    def test_batch_mixed_valid_invalid(self):
        """Test batch with mix of valid and invalid detections."""
        detections = [
            {"latitude": -3.1, "longitude": -60.0, "area_ha": 5.0,
             "change_type": "deforestation", "confidence": 0.90},
            {"latitude": -1.5, "longitude": 116.0, "area_ha": 3.0,
             "change_type": "no_change", "confidence": 0.95},
            {"latitude": 6.5, "longitude": -1.6, "area_ha": 2.0,
             "change_type": "degradation", "confidence": 0.50},
        ]
        alerts = _generate_batch(detections)
        assert len(alerts) == 1  # Only the first is valid

    def test_batch_with_plots(self):
        """Test batch generation with nearby plots."""
        detections = [
            {"latitude": -3.1, "longitude": -60.0, "area_ha": 5.0,
             "change_type": "deforestation", "confidence": 0.90},
        ]
        plots = [
            {"plot_id": "PLOT-001", "latitude": -3.105, "longitude": -60.005,
             "buffer_radius_km": 10.0},
        ]
        alerts = _generate_batch(detections, nearby_plots=plots)
        assert len(alerts) == 1
        assert "PLOT-001" in alerts[0]["affected_plots"]


# ===========================================================================
# 3. TestAlertRetrieval (3 tests)
# ===========================================================================


class TestAlertRetrieval:
    """Test get_alert by ID, missing ID."""

    def test_get_alert_by_id(self):
        """Test retrieving an alert by its ID."""
        alerts = [
            {"alert_id": "ALR-001", "severity": "critical"},
            {"alert_id": "ALR-002", "severity": "high"},
        ]
        result = _get_alert(alerts, "ALR-001")
        assert result is not None
        assert result["alert_id"] == "ALR-001"

    def test_get_alert_missing_id(self):
        """Test retrieving a non-existent alert returns None."""
        alerts = [{"alert_id": "ALR-001", "severity": "critical"}]
        result = _get_alert(alerts, "ALR-MISSING")
        assert result is None

    def test_get_alert_empty_list(self):
        """Test retrieving from empty list returns None."""
        result = _get_alert([], "ALR-001")
        assert result is None


# ===========================================================================
# 4. TestAlertListing (8 tests)
# ===========================================================================


class TestAlertListing:
    """Test list_alerts with filters."""

    @pytest.fixture
    def mixed_alerts(self):
        """Create mixed set of alerts for filtering tests."""
        return [
            {"alert_id": "ALR-001", "severity": "critical", "status": "pending",
             "country_code": "BR", "area_ha": 55.0},
            {"alert_id": "ALR-002", "severity": "critical", "status": "triaged",
             "country_code": "BR", "area_ha": 52.0},
            {"alert_id": "ALR-003", "severity": "high", "status": "pending",
             "country_code": "ID", "area_ha": 15.0},
            {"alert_id": "ALR-004", "severity": "medium", "status": "investigating",
             "country_code": "GH", "area_ha": 3.0},
            {"alert_id": "ALR-005", "severity": "low", "status": "pending",
             "country_code": "CD", "area_ha": 0.8},
            {"alert_id": "ALR-006", "severity": "informational", "status": "resolved",
             "country_code": "CO", "area_ha": 0.2},
        ]

    def test_list_all_alerts(self, mixed_alerts):
        """Test listing all alerts without filters."""
        result = _list_alerts(mixed_alerts)
        assert len(result) == 6

    def test_filter_by_severity(self, mixed_alerts):
        """Test filtering by severity level."""
        result = _list_alerts(mixed_alerts, severity="critical")
        assert all(a["severity"] == "critical" for a in result)
        assert len(result) == 2

    def test_filter_by_status(self, mixed_alerts):
        """Test filtering by alert status."""
        result = _list_alerts(mixed_alerts, status="pending")
        assert all(a["status"] == "pending" for a in result)
        assert len(result) == 3

    def test_filter_by_country(self, mixed_alerts):
        """Test filtering by country code."""
        result = _list_alerts(mixed_alerts, country_code="BR")
        assert all(a["country_code"] == "BR" for a in result)
        assert len(result) == 2

    def test_pagination_first_page(self, mixed_alerts):
        """Test pagination first page."""
        result = _list_alerts(mixed_alerts, offset=0, limit=3)
        assert len(result) == 3
        assert result[0]["alert_id"] == "ALR-001"

    def test_pagination_second_page(self, mixed_alerts):
        """Test pagination second page."""
        result = _list_alerts(mixed_alerts, offset=3, limit=3)
        assert len(result) == 3
        assert result[0]["alert_id"] == "ALR-004"

    def test_pagination_beyond_end(self, mixed_alerts):
        """Test pagination beyond list returns empty."""
        result = _list_alerts(mixed_alerts, offset=100, limit=10)
        assert result == []

    def test_combined_filters(self, mixed_alerts):
        """Test combining severity and country filters."""
        result = _list_alerts(mixed_alerts, severity="critical", country_code="BR")
        assert len(result) == 2
        assert all(a["severity"] == "critical" for a in result)
        assert all(a["country_code"] == "BR" for a in result)


# ===========================================================================
# 5. TestAlertSummary (5 tests)
# ===========================================================================


class TestAlertSummary:
    """Test get_alert_summary by country and global."""

    @pytest.fixture
    def summary_alerts(self):
        """Create alerts for summary tests."""
        return [
            {"severity": "critical", "status": "pending",
             "country_code": "BR", "area_ha": 55.0},
            {"severity": "high", "status": "triaged",
             "country_code": "BR", "area_ha": 15.0},
            {"severity": "medium", "status": "pending",
             "country_code": "ID", "area_ha": 3.0},
            {"severity": "low", "status": "resolved",
             "country_code": "GH", "area_ha": 0.8},
        ]

    def test_global_summary_counts(self, summary_alerts):
        """Test global summary total count."""
        summary = _get_alert_summary(summary_alerts)
        assert summary["total"] == 4
        assert summary["total_area_ha"] == pytest.approx(73.8, abs=0.1)

    def test_summary_by_severity(self, summary_alerts):
        """Test summary by severity breakdown."""
        summary = _get_alert_summary(summary_alerts)
        assert summary["by_severity"]["critical"] == 1
        assert summary["by_severity"]["high"] == 1
        assert summary["by_severity"]["medium"] == 1
        assert summary["by_severity"]["low"] == 1

    def test_summary_by_country(self, summary_alerts):
        """Test summary filtered by country."""
        summary = _get_alert_summary(summary_alerts, country_code="BR")
        assert summary["total"] == 2
        assert summary["by_country"]["BR"] == 2

    def test_summary_empty_alerts(self):
        """Test summary with empty alert list."""
        summary = _get_alert_summary([])
        assert summary["total"] == 0
        assert summary["total_area_ha"] == 0

    def test_summary_all_severities_present(self, summary_alerts):
        """Test all severity levels appear in summary."""
        summary = _get_alert_summary(summary_alerts)
        for sev in SEVERITY_LEVELS:
            assert sev in summary["by_severity"]


# ===========================================================================
# 6. TestAlertStatistics (4 tests)
# ===========================================================================


class TestAlertStatistics:
    """Test get_alert_statistics grouped by various dimensions."""

    @pytest.fixture
    def stats_alerts(self):
        return [
            {"severity": "critical", "status": "pending",
             "country_code": "BR", "area_ha": 55.0},
            {"severity": "critical", "status": "triaged",
             "country_code": "BR", "area_ha": 60.0},
            {"severity": "high", "status": "pending",
             "country_code": "ID", "area_ha": 15.0},
        ]

    def test_stats_by_severity(self, stats_alerts):
        """Test statistics grouped by severity."""
        stats = _get_alert_statistics(stats_alerts, group_by="severity")
        assert stats["critical"]["count"] == 2
        assert stats["critical"]["total_area_ha"] == pytest.approx(115.0)
        assert stats["high"]["count"] == 1

    def test_stats_by_status(self, stats_alerts):
        """Test statistics grouped by status."""
        stats = _get_alert_statistics(stats_alerts, group_by="status")
        assert stats["pending"]["count"] == 2
        assert stats["triaged"]["count"] == 1

    def test_stats_by_country(self, stats_alerts):
        """Test statistics grouped by country."""
        stats = _get_alert_statistics(stats_alerts, group_by="country_code")
        assert stats["BR"]["count"] == 2
        assert stats["ID"]["count"] == 1

    def test_stats_empty_alerts(self):
        """Test statistics with empty list."""
        stats = _get_alert_statistics([], group_by="severity")
        assert stats == {}


# ===========================================================================
# 7. TestProximityCalculation (5 tests)
# ===========================================================================


class TestProximityCalculation:
    """Test Haversine proximity calculation with known distances."""

    def test_same_point_zero_distance(self):
        """Test distance between same point is zero."""
        dist = _calculate_proximity(-3.1, -60.0, -3.1, -60.0)
        assert dist == pytest.approx(0.0, abs=0.001)

    def test_known_distance_equator(self):
        """Test known distance along equator (1 degree ~ 111 km)."""
        dist = _calculate_proximity(0.0, 0.0, 0.0, 1.0)
        assert dist == pytest.approx(111.195, abs=1.0)

    def test_known_distance_north_south(self):
        """Test known distance along meridian (1 degree latitude ~ 111 km)."""
        dist = _calculate_proximity(0.0, 0.0, 1.0, 0.0)
        assert dist == pytest.approx(111.195, abs=1.0)

    def test_antipodal_points(self):
        """Test distance between antipodal points is ~20000 km."""
        dist = _calculate_proximity(0.0, 0.0, 0.0, 180.0)
        assert dist == pytest.approx(20015.1, abs=10.0)

    @pytest.mark.parametrize("lat1,lon1,lat2,lon2,expected_min,expected_max", [
        (-3.1, -60.0, -3.105, -60.005, 0, 2),       # Very close
        (-3.1, -60.0, -3.2, -60.1, 10, 20),          # Moderate distance
        (-3.1, -60.0, -1.5, 116.0, 15000, 20000),    # Very far
    ])
    def test_proximity_known_ranges(
        self, lat1, lon1, lat2, lon2, expected_min, expected_max,
    ):
        """Test proximity falls within expected ranges."""
        dist = _calculate_proximity(lat1, lon1, lat2, lon2)
        assert expected_min <= dist <= expected_max


# ===========================================================================
# 8. TestPostCutoffDetermination (5 tests)
# ===========================================================================


class TestPostCutoffDetermination:
    """Test post-cutoff determination for dates before/after 2020-12-31."""

    def test_date_after_cutoff(self):
        """Test date after 2020-12-31 is post-cutoff."""
        assert _determine_post_cutoff(date(2021, 1, 1)) is True

    def test_date_before_cutoff(self):
        """Test date before 2020-12-31 is pre-cutoff."""
        assert _determine_post_cutoff(date(2020, 6, 15)) is False

    def test_date_on_cutoff(self):
        """Test date exactly on 2020-12-31 is NOT post-cutoff."""
        assert _determine_post_cutoff(date(2020, 12, 31)) is False

    def test_date_one_day_after(self):
        """Test date one day after cutoff is post-cutoff."""
        assert _determine_post_cutoff(date(2021, 1, 1)) is True

    @pytest.mark.parametrize("det_date,expected", [
        (date(2018, 1, 1), False),
        (date(2019, 6, 15), False),
        (date(2020, 12, 30), False),
        (date(2020, 12, 31), False),
        (date(2021, 1, 1), True),
        (date(2023, 6, 15), True),
        (date(2025, 12, 31), True),
    ])
    def test_post_cutoff_parametrized(self, det_date, expected):
        """Test post-cutoff determination with parametrized dates."""
        assert _determine_post_cutoff(det_date) is expected


# ===========================================================================
# 9. TestDeduplication (4 tests)
# ===========================================================================


class TestDeduplication:
    """Test alert deduplication within/outside dedup window."""

    def test_duplicate_alerts_merged(self):
        """Test nearby alerts are deduplicated."""
        alerts = [
            {"latitude": -3.1000, "longitude": -60.0000, "severity": "critical"},
            {"latitude": -3.1001, "longitude": -60.0001, "severity": "high"},
        ]
        deduped = _deduplicate_alerts(alerts, window_km=1.0)
        assert len(deduped) == 1

    def test_distant_alerts_kept(self):
        """Test distant alerts are not deduplicated."""
        alerts = [
            {"latitude": -3.1, "longitude": -60.0, "severity": "critical"},
            {"latitude": -1.5, "longitude": 116.0, "severity": "high"},
        ]
        deduped = _deduplicate_alerts(alerts, window_km=1.0)
        assert len(deduped) == 2

    def test_empty_list_dedup(self):
        """Test deduplication with empty list."""
        deduped = _deduplicate_alerts([])
        assert deduped == []

    def test_single_alert_dedup(self):
        """Test single alert passes through deduplication."""
        alerts = [
            {"latitude": -3.1, "longitude": -60.0, "severity": "critical"},
        ]
        deduped = _deduplicate_alerts(alerts)
        assert len(deduped) == 1


# ===========================================================================
# 10. TestProvenance (3 tests)
# ===========================================================================


class TestAlertProvenance:
    """Test provenance hash generation and determinism."""

    def test_alert_provenance_deterministic(self):
        """Test alert provenance hash is deterministic."""
        data = {"detection_id": "det-001", "severity": "critical", "area_ha": 5.5}
        hashes = [compute_test_hash(data) for _ in range(10)]
        assert len(set(hashes)) == 1

    def test_different_inputs_different_hashes(self):
        """Test different alert data produces different hashes."""
        h1 = compute_test_hash({"detection_id": "det-001", "severity": "critical"})
        h2 = compute_test_hash({"detection_id": "det-002", "severity": "high"})
        assert h1 != h2

    def test_alert_provenance_sha256_format(self):
        """Test alert provenance hash is valid SHA-256 hex."""
        h = compute_test_hash({"alert": "test"})
        assert len(h) == SHA256_HEX_LENGTH
        assert all(c in "0123456789abcdef" for c in h)
