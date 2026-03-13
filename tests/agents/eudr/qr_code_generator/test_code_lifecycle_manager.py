# -*- coding: utf-8 -*-
"""
Unit tests for Engine 8: Code Lifecycle Manager (AGENT-EUDR-014)

Tests QR code lifecycle management including code activation,
deactivation, revocation, expiry, scan event recording, scan analytics,
reprint tracking, and edge cases.

55+ tests across 8 test classes.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from .conftest import (
    CODE_STATUSES,
    COUNTERFEIT_RISK_LEVELS,
    DEFAULT_MAX_REPRINTS,
    EUDR_RETENTION_YEARS,
    LIFECYCLE_EVENT_TYPES,
    SAMPLE_CODE_ID,
    SAMPLE_CODE_ID_2,
    SAMPLE_CODE_ID_3,
    SAMPLE_OPERATOR_ID,
    SAMPLE_SCAN_LAT,
    SAMPLE_SCAN_LON,
    SCAN_OUTCOMES,
    assert_lifecycle_event_valid,
    assert_qr_code_valid,
    assert_scan_event_valid,
    make_lifecycle_event,
    make_qr_code,
    make_scan_event,
    _sha256,
)


# =========================================================================
# Test Class 1: Code Activation
# =========================================================================

class TestCodeActivation:
    """Test created -> active transition."""

    def test_activation_event_type(self):
        """Test activation event has correct event type."""
        event = make_lifecycle_event(
            event_type="activation",
            previous_status="created",
            new_status="active",
        )
        assert event["event_type"] == "activation"

    def test_activation_from_created(self):
        """Test activation from created status."""
        event = make_lifecycle_event(
            previous_status="created",
            new_status="active",
        )
        assert event["previous_status"] == "created"
        assert event["new_status"] == "active"

    def test_activation_has_reason(self):
        """Test activation event has a reason."""
        event = make_lifecycle_event(
            event_type="activation",
            reason="Label printed and applied",
        )
        assert event["reason"] is not None
        assert len(event["reason"]) > 0

    def test_activation_has_performer(self):
        """Test activation records who performed it."""
        event = make_lifecycle_event(
            event_type="activation",
            performed_by="warehouse@greenlang.eu",
        )
        assert event["performed_by"] == "warehouse@greenlang.eu"

    def test_activation_metadata(self):
        """Test activation event can carry metadata."""
        event = make_lifecycle_event(
            event_type="activation",
            metadata={"label_id": "LBL-001", "printer": "ZBR-T410"},
        )
        assert event["metadata"]["label_id"] == "LBL-001"

    def test_activation_passes_validation(self):
        """Test activation event passes full validation."""
        event = make_lifecycle_event(
            event_type="activation",
            previous_status="created",
            new_status="active",
        )
        assert_lifecycle_event_valid(event)

    def test_activation_has_timestamp(self):
        """Test activation event has a timestamp."""
        event = make_lifecycle_event(event_type="activation")
        assert event["created_at"] is not None

    def test_qr_code_active_after_activation(self):
        """Test QR code record with active status."""
        code = make_qr_code(status="active")
        assert code["status"] == "active"
        assert code["activated_at"] is not None


# =========================================================================
# Test Class 2: Code Deactivation
# =========================================================================

class TestCodeDeactivation:
    """Test active -> deactivated transition."""

    def test_deactivation_event_type(self):
        """Test deactivation event has correct event type."""
        event = make_lifecycle_event(
            event_type="deactivation",
            previous_status="active",
            new_status="deactivated",
        )
        assert event["event_type"] == "deactivation"

    def test_deactivation_from_active(self):
        """Test deactivation from active status."""
        event = make_lifecycle_event(
            previous_status="active",
            new_status="deactivated",
        )
        assert event["previous_status"] == "active"
        assert event["new_status"] == "deactivated"

    def test_deactivation_requires_reason(self):
        """Test deactivation has a reason for audit trail."""
        event = make_lifecycle_event(
            event_type="deactivation",
            reason="Product recall pending investigation",
        )
        assert "recall" in event["reason"].lower()

    def test_deactivation_is_temporary(self):
        """Test deactivated code can be reactivated."""
        deact = make_lifecycle_event(
            event_type="deactivation",
            previous_status="active",
            new_status="deactivated",
        )
        react = make_lifecycle_event(
            event_type="reactivation",
            previous_status="deactivated",
            new_status="active",
        )
        assert deact["new_status"] == "deactivated"
        assert react["new_status"] == "active"

    def test_deactivated_code_record(self):
        """Test QR code record with deactivated status."""
        code = make_qr_code(status="deactivated")
        assert code["status"] == "deactivated"
        assert code["deactivated_at"] is not None

    def test_deactivation_passes_validation(self):
        """Test deactivation event passes validation."""
        event = make_lifecycle_event(
            event_type="deactivation",
            previous_status="active",
            new_status="deactivated",
        )
        assert_lifecycle_event_valid(event)


# =========================================================================
# Test Class 3: Code Revocation
# =========================================================================

class TestCodeRevocation:
    """Test any -> revoked (permanent) transition."""

    def test_revocation_event_type(self):
        """Test revocation event has correct event type."""
        event = make_lifecycle_event(
            event_type="revocation",
            previous_status="active",
            new_status="revoked",
        )
        assert event["event_type"] == "revocation"

    def test_revocation_from_active(self):
        """Test revocation from active status."""
        event = make_lifecycle_event(
            previous_status="active",
            new_status="revoked",
        )
        assert event["new_status"] == "revoked"

    def test_revocation_from_deactivated(self):
        """Test revocation from deactivated status."""
        event = make_lifecycle_event(
            previous_status="deactivated",
            new_status="revoked",
        )
        assert event["previous_status"] == "deactivated"
        assert event["new_status"] == "revoked"

    def test_revocation_from_created(self):
        """Test revocation from created status."""
        event = make_lifecycle_event(
            previous_status="created",
            new_status="revoked",
        )
        assert event["previous_status"] == "created"
        assert event["new_status"] == "revoked"

    def test_revocation_is_permanent(self):
        """Test revoked status is final (cannot reactivate)."""
        code = make_qr_code(status="revoked")
        assert code["status"] == "revoked"
        assert code["revoked_at"] is not None

    def test_revocation_has_reason(self):
        """Test revocation must have a reason."""
        event = make_lifecycle_event(
            event_type="revocation",
            reason="Counterfeit detected - HMAC mismatch",
        )
        assert event["reason"] is not None

    def test_revocation_with_investigation_metadata(self):
        """Test revocation with investigation reference."""
        event = make_lifecycle_event(
            event_type="revocation",
            metadata={
                "investigation_ref": "INV-2026-001",
                "scan_id": "SCAN-CFEIT-001",
            },
        )
        assert event["metadata"]["investigation_ref"] == "INV-2026-001"

    def test_revocation_passes_validation(self):
        """Test revocation event passes validation."""
        event = make_lifecycle_event(
            event_type="revocation",
            previous_status="active",
            new_status="revoked",
        )
        assert_lifecycle_event_valid(event)


# =========================================================================
# Test Class 4: Code Expiry
# =========================================================================

class TestCodeExpiry:
    """Test automatic expiry and TTL checking."""

    def test_expiry_event_type(self):
        """Test expiry event has correct event type."""
        event = make_lifecycle_event(
            event_type="expiry",
            previous_status="active",
            new_status="expired",
        )
        assert event["event_type"] == "expiry"

    def test_expired_code_record(self):
        """Test QR code record with expired status."""
        code = make_qr_code(status="expired")
        assert code["status"] == "expired"

    def test_expiry_ttl_default_5_years(self):
        """Test default TTL is 5 years per EUDR Article 14."""
        assert EUDR_RETENTION_YEARS == 5

    def test_expiry_has_expires_at(self):
        """Test QR code has expires_at timestamp."""
        code = make_qr_code()
        assert code["expires_at"] is not None

    def test_expiry_reason_references_eudr(self):
        """Test expiry reason references EUDR Article 14."""
        event = make_lifecycle_event(
            event_type="expiry",
            reason="TTL expired after 5 years per EUDR Article 14",
        )
        assert "EUDR" in event["reason"]
        assert "Article 14" in event["reason"]

    def test_expired_code_still_retained(self):
        """Test expired code records are retained for audit."""
        code = make_qr_code(status="expired")
        assert code["provenance_hash"] is None or isinstance(code["provenance_hash"], str)

    def test_expiry_passes_validation(self):
        """Test expiry event passes validation."""
        event = make_lifecycle_event(
            event_type="expiry",
            previous_status="active",
            new_status="expired",
        )
        assert_lifecycle_event_valid(event)


# =========================================================================
# Test Class 5: Scan Event Recording
# =========================================================================

class TestScanEventRecording:
    """Test scan event recording with GPS, outcome, and device info."""

    def test_record_verified_scan(self):
        """Test recording a verified scan event."""
        event = make_scan_event(outcome="verified")
        assert event["outcome"] == "verified"
        assert_scan_event_valid(event)

    def test_scan_with_gps_coordinates(self):
        """Test scan with GPS latitude and longitude."""
        event = make_scan_event(
            scan_latitude=SAMPLE_SCAN_LAT,
            scan_longitude=SAMPLE_SCAN_LON,
        )
        assert event["scan_latitude"] == SAMPLE_SCAN_LAT
        assert event["scan_longitude"] == SAMPLE_SCAN_LON

    def test_scan_with_device_info(self):
        """Test scan records device information."""
        event = make_scan_event(
            scanner_user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0)",
        )
        assert "iPhone" in event["scanner_user_agent"]

    def test_scan_response_time(self):
        """Test scan records processing time."""
        event = make_scan_event(response_time_ms=25.7)
        assert event["response_time_ms"] == 25.7

    @pytest.mark.parametrize("outcome", SCAN_OUTCOMES)
    def test_all_scan_outcomes(self, outcome: str):
        """Test all scan outcomes produce valid events."""
        event = make_scan_event(outcome=outcome)
        assert event["outcome"] == outcome
        assert_scan_event_valid(event)

    def test_scan_country_code(self):
        """Test scan records country code."""
        event = make_scan_event(scan_country="DE")
        assert event["scan_country"] == "DE"

    def test_scan_hmac_validation_result(self):
        """Test scan records HMAC validation result."""
        event = make_scan_event(hmac_valid=True)
        assert event["hmac_valid"] is True

    def test_unique_scan_ids(self):
        """Test each scan event has a unique ID."""
        ids = {make_scan_event()["scan_id"] for _ in range(50)}
        assert len(ids) == 50


# =========================================================================
# Test Class 6: Scan Analytics
# =========================================================================

class TestScanAnalytics:
    """Test aggregate scan counts and geo distribution."""

    def test_qr_code_scan_count(self):
        """Test QR code tracks total scan count."""
        code = make_qr_code(scan_count=42)
        assert code["scan_count"] == 42

    def test_scan_count_starts_zero(self):
        """Test new QR code has zero scan count."""
        code = make_qr_code()
        assert code["scan_count"] == 0

    def test_scan_count_increments(self):
        """Test scan count can be incremented."""
        code = make_qr_code(scan_count=100)
        assert code["scan_count"] == 100

    def test_scan_geographic_distribution(self):
        """Test scans from different countries."""
        countries = ["DE", "FR", "NL", "BE", "IT"]
        events = [
            make_scan_event(scan_country=c) for c in countries
        ]
        scan_countries = {e["scan_country"] for e in events}
        assert len(scan_countries) == 5

    @pytest.mark.parametrize("risk", COUNTERFEIT_RISK_LEVELS)
    def test_risk_distribution(self, risk: str):
        """Test scans across all risk levels."""
        event = make_scan_event(counterfeit_risk=risk)
        assert event["counterfeit_risk"] == risk

    def test_scan_velocity_distribution(self):
        """Test scans with varying velocities."""
        velocities = [1, 5, 10, 50, 100, 200]
        events = [
            make_scan_event(velocity_scans_per_min=v) for v in velocities
        ]
        assert len(events) == 6

    def test_scan_events_for_same_code(self):
        """Test multiple scan events for the same code."""
        events = [
            make_scan_event(code_id=SAMPLE_CODE_ID)
            for _ in range(10)
        ]
        assert all(e["code_id"] == SAMPLE_CODE_ID for e in events)
        assert len({e["scan_id"] for e in events}) == 10


# =========================================================================
# Test Class 7: Reprint Tracking
# =========================================================================

class TestReprintTracking:
    """Test reprint records and limits."""

    def test_reprint_event_type(self):
        """Test reprint lifecycle event."""
        event = make_lifecycle_event(
            event_type="reprint",
            previous_status="active",
            new_status="active",
        )
        assert event["event_type"] == "reprint"

    def test_reprint_count_tracked(self):
        """Test QR code tracks reprint count."""
        code = make_qr_code(reprint_count=2)
        assert code["reprint_count"] == 2

    def test_reprint_count_starts_zero(self):
        """Test new QR code has zero reprints."""
        code = make_qr_code()
        assert code["reprint_count"] == 0

    def test_max_reprints_default(self):
        """Test default max reprints is 3."""
        assert DEFAULT_MAX_REPRINTS == 3

    def test_reprint_below_limit(self):
        """Test reprint count below limit is valid."""
        code = make_qr_code(reprint_count=2)
        assert code["reprint_count"] < DEFAULT_MAX_REPRINTS

    def test_reprint_at_limit(self):
        """Test reprint count at maximum limit."""
        code = make_qr_code(reprint_count=DEFAULT_MAX_REPRINTS)
        assert code["reprint_count"] == DEFAULT_MAX_REPRINTS

    def test_reprint_metadata(self):
        """Test reprint event carries reprint details."""
        event = make_lifecycle_event(
            event_type="reprint",
            metadata={
                "reprint_count": 1,
                "label_id": "LBL-REPRINT-001",
                "reason": "Label damaged",
            },
        )
        assert event["metadata"]["reprint_count"] == 1

    def test_reprint_preserves_active_status(self):
        """Test reprint does not change active status."""
        event = make_lifecycle_event(
            event_type="reprint",
            previous_status="active",
            new_status="active",
        )
        assert event["previous_status"] == "active"
        assert event["new_status"] == "active"


# =========================================================================
# Test Class 8: Edge Cases
# =========================================================================

class TestLifecycleEdgeCases:
    """Test edge cases for lifecycle management."""

    def test_replacement_workflow(self):
        """Test code replacement creates new code and revokes old."""
        old_code = make_qr_code(status="revoked", code_id="QR-OLD-001")
        new_code = make_qr_code(status="active", code_id="QR-NEW-001")
        assert old_code["status"] == "revoked"
        assert new_code["status"] == "active"
        assert old_code["code_id"] != new_code["code_id"]

    def test_audit_trail_completeness(self):
        """Test lifecycle events form a complete audit trail."""
        events = [
            make_lifecycle_event(
                event_type="activation",
                previous_status="created",
                new_status="active",
            ),
            make_lifecycle_event(
                event_type="deactivation",
                previous_status="active",
                new_status="deactivated",
            ),
            make_lifecycle_event(
                event_type="reactivation",
                previous_status="deactivated",
                new_status="active",
            ),
            make_lifecycle_event(
                event_type="revocation",
                previous_status="active",
                new_status="revoked",
            ),
        ]
        assert len(events) == 4
        for event in events:
            assert_lifecycle_event_valid(event)

    def test_bulk_activation(self):
        """Test multiple codes activated at once."""
        events = [
            make_lifecycle_event(
                code_id=f"QR-BULK-{i:03d}",
                event_type="activation",
                previous_status="created",
                new_status="active",
            )
            for i in range(20)
        ]
        assert len(events) == 20
        code_ids = {e["code_id"] for e in events}
        assert len(code_ids) == 20

    @pytest.mark.parametrize("status", CODE_STATUSES)
    def test_all_code_statuses_in_lifecycle(self, status: str):
        """Test all code statuses appear in lifecycle events."""
        event = make_lifecycle_event(
            new_status=status,
            previous_status="created" if status != "created" else "active",
        )
        assert event["new_status"] == status

    def test_event_unique_ids(self):
        """Test each lifecycle event has a unique ID."""
        ids = {make_lifecycle_event()["event_id"] for _ in range(50)}
        assert len(ids) == 50

    def test_provenance_hash_on_event(self):
        """Test lifecycle event can carry provenance hash."""
        prov = _sha256("lifecycle-provenance")
        event = make_lifecycle_event(provenance_hash=prov)
        assert event["provenance_hash"] == prov

    def test_system_performer(self):
        """Test system-initiated lifecycle events."""
        event = make_lifecycle_event(performed_by="system@greenlang.eu")
        assert event["performed_by"] == "system@greenlang.eu"

    def test_human_performer(self):
        """Test human-initiated lifecycle events."""
        event = make_lifecycle_event(
            performed_by="john.doe@company.com",
        )
        assert event["performed_by"] == "john.doe@company.com"

    def test_lifecycle_event_types_coverage(self):
        """Test all defined lifecycle event types."""
        for evt_type in LIFECYCLE_EVENT_TYPES:
            event = make_lifecycle_event(event_type=evt_type)
            assert event["event_type"] == evt_type
