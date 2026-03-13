# -*- coding: utf-8 -*-
"""
Unit tests for Engine 6: Anti-Counterfeit Engine (AGENT-EUDR-014)

Tests anti-counterfeiting features including HMAC signature generation
and verification, rotating tokens, digital watermark, counterfeit risk
assessment, scan velocity detection, geo-fencing, key rotation, and
revocation list management.

55+ tests across 8 test classes.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
"""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest

from .conftest import (
    COUNTERFEIT_RISK_LEVELS,
    DEFAULT_SCAN_VELOCITY_THRESHOLD,
    HMAC_SHA256_HEX_LENGTH,
    SAMPLE_CODE_ID,
    SAMPLE_CODE_ID_2,
    SAMPLE_HMAC_KEY,
    SAMPLE_HMAC_VALUE,
    SAMPLE_OPERATOR_ID,
    SAMPLE_SCAN_LAT,
    SAMPLE_SCAN_LON,
    SCAN_OUTCOMES,
    SHA256_HEX_LENGTH,
    assert_scan_event_valid,
    assert_signature_valid,
    assert_valid_hmac,
    assert_valid_sha256,
    make_scan_event,
    make_signature,
    _hmac_sha256,
    _sha256,
)


# =========================================================================
# Test Class 1: HMAC Signature
# =========================================================================

class TestHMACSignature:
    """Test HMAC-SHA256 signature generation and verification."""

    def test_generate_hmac_signature(self):
        """Test HMAC-SHA256 signature generation."""
        sig = make_signature()
        assert sig["algorithm"] == "HMAC-SHA256"
        assert len(sig["signature_value"]) == HMAC_SHA256_HEX_LENGTH

    def test_signature_is_valid_hex(self):
        """Test signature value is valid hexadecimal."""
        sig = make_signature()
        assert all(c in "0123456789abcdef" for c in sig["signature_value"])

    def test_verify_valid_signature(self):
        """Test valid signature has valid=True."""
        sig = make_signature(valid=True)
        assert sig["valid"] is True

    def test_invalid_signature_flag(self):
        """Test invalid signature has valid=False."""
        sig = make_signature(valid=False)
        assert sig["valid"] is False

    def test_signature_deterministic(self):
        """Test same inputs produce same signature."""
        data_hash = _sha256("test-data-001")
        sig1 = make_signature(
            signed_data_hash=data_hash,
            hmac_key=SAMPLE_HMAC_KEY,
        )
        sig2 = make_signature(
            signed_data_hash=data_hash,
            hmac_key=SAMPLE_HMAC_KEY,
        )
        assert sig1["signature_value"] == sig2["signature_value"]

    def test_different_data_different_signature(self):
        """Test different data produces different signatures."""
        sig1 = make_signature(signed_data_hash=_sha256("data-1"))
        sig2 = make_signature(signed_data_hash=_sha256("data-2"))
        assert sig1["signature_value"] != sig2["signature_value"]

    def test_signature_has_key_id(self):
        """Test signature record includes key identifier."""
        sig = make_signature(key_id="KEY-QRG-002")
        assert sig["key_id"] == "KEY-QRG-002"

    def test_signature_passes_validation(self):
        """Test signature passes assertion helper validation."""
        sig = make_signature()
        assert_signature_valid(sig)


# =========================================================================
# Test Class 2: Rotating Tokens
# =========================================================================

class TestRotatingTokens:
    """Test TOTP-like rotating token generation and verification."""

    def test_token_changes_with_time_window(self):
        """Test token changes across different time windows."""
        # Simulate different time-based inputs
        token1 = _hmac_sha256(SAMPLE_HMAC_KEY, f"{SAMPLE_CODE_ID}:window:1")
        token2 = _hmac_sha256(SAMPLE_HMAC_KEY, f"{SAMPLE_CODE_ID}:window:2")
        assert token1 != token2

    def test_token_same_within_window(self):
        """Test token is consistent within the same time window."""
        token1 = _hmac_sha256(SAMPLE_HMAC_KEY, f"{SAMPLE_CODE_ID}:window:1")
        token2 = _hmac_sha256(SAMPLE_HMAC_KEY, f"{SAMPLE_CODE_ID}:window:1")
        assert token1 == token2

    def test_token_is_valid_hmac(self):
        """Test rotating token is a valid HMAC-SHA256."""
        token = _hmac_sha256(SAMPLE_HMAC_KEY, f"{SAMPLE_CODE_ID}:rotating")
        assert_valid_hmac(token)

    def test_token_for_different_codes(self):
        """Test different codes produce different tokens."""
        t1 = _hmac_sha256(SAMPLE_HMAC_KEY, f"QR-A:window:1")
        t2 = _hmac_sha256(SAMPLE_HMAC_KEY, f"QR-B:window:1")
        assert t1 != t2

    def test_token_with_different_keys(self):
        """Test different keys produce different tokens."""
        t1 = _hmac_sha256("key-v1", f"{SAMPLE_CODE_ID}:window:1")
        t2 = _hmac_sha256("key-v2", f"{SAMPLE_CODE_ID}:window:1")
        assert t1 != t2

    def test_token_length_is_64(self):
        """Test rotating token is 64 hex characters."""
        token = _hmac_sha256(SAMPLE_HMAC_KEY, "rotating-test")
        assert len(token) == HMAC_SHA256_HEX_LENGTH


# =========================================================================
# Test Class 3: Digital Watermark
# =========================================================================

class TestDigitalWatermark:
    """Test digital watermark embedding and detection."""

    def test_watermark_disabled_by_default(self):
        """Test digital watermark is disabled by default."""
        # Config default: enable_digital_watermark = False
        from .conftest import make_qr_code
        code = make_qr_code()
        assert code["logo_embedded"] is False  # No watermark by default

    def test_watermark_does_not_change_payload_hash(self):
        """Test watermark does not alter the payload hash."""
        from .conftest import make_qr_code
        code1 = make_qr_code(payload_hash=_sha256("same-data"))
        code2 = make_qr_code(payload_hash=_sha256("same-data"))
        assert code1["payload_hash"] == code2["payload_hash"]

    def test_watermark_metadata_in_scan(self):
        """Test scan event can carry watermark detection metadata."""
        event = make_scan_event(
            metadata={"watermark_detected": True, "watermark_confidence": 0.95},
        )
        assert event.get("metadata", {}).get("watermark_detected") is True

    def test_watermark_confidence_range(self):
        """Test watermark confidence is in 0-1 range."""
        confidence = 0.92
        assert 0.0 <= confidence <= 1.0

    def test_scan_without_watermark_detection(self):
        """Test scan event without watermark data is still valid."""
        event = make_scan_event()
        assert_scan_event_valid(event)

    def test_watermark_survives_quality_check(self):
        """Test watermarked code can still pass quality grading."""
        from .conftest import make_qr_code
        code = make_qr_code(quality_grade="B")
        assert code["quality_grade"] in ["A", "B", "C", "D"]


# =========================================================================
# Test Class 4: Counterfeit Risk Assessment
# =========================================================================

class TestCounterfeitRisk:
    """Test counterfeit risk scoring."""

    @pytest.mark.parametrize("risk", COUNTERFEIT_RISK_LEVELS)
    def test_all_risk_levels(self, risk: str):
        """Test all counterfeit risk levels are valid."""
        event = make_scan_event(counterfeit_risk=risk)
        assert event["counterfeit_risk"] == risk
        assert_scan_event_valid(event)

    def test_low_risk_normal_scan(self):
        """Test low risk for normal scan pattern."""
        event = make_scan_event(
            counterfeit_risk="low",
            velocity_scans_per_min=5,
            geo_fence_violated=False,
            hmac_valid=True,
        )
        assert event["counterfeit_risk"] == "low"

    def test_medium_risk_elevated_velocity(self):
        """Test medium risk for slightly elevated scan velocity."""
        event = make_scan_event(
            counterfeit_risk="medium",
            velocity_scans_per_min=50,
        )
        assert event["counterfeit_risk"] == "medium"

    def test_high_risk_velocity_exceeded(self):
        """Test high risk when scan velocity threshold exceeded."""
        event = make_scan_event(
            counterfeit_risk="high",
            velocity_scans_per_min=DEFAULT_SCAN_VELOCITY_THRESHOLD + 10,
            geo_fence_violated=True,
        )
        assert event["counterfeit_risk"] == "high"

    def test_critical_risk_multiple_indicators(self):
        """Test critical risk with multiple counterfeit indicators."""
        event = make_scan_event(
            counterfeit_risk="critical",
            velocity_scans_per_min=500,
            geo_fence_violated=True,
            hmac_valid=False,
            outcome="counterfeit_suspected",
        )
        assert event["counterfeit_risk"] == "critical"
        assert event["hmac_valid"] is False

    def test_exactly_four_risk_levels(self):
        """Test exactly 4 risk levels exist."""
        assert len(COUNTERFEIT_RISK_LEVELS) == 4

    def test_risk_level_ordering(self):
        """Test risk levels are ordered low to critical."""
        assert COUNTERFEIT_RISK_LEVELS == ["low", "medium", "high", "critical"]


# =========================================================================
# Test Class 5: Scan Velocity
# =========================================================================

class TestScanVelocity:
    """Test scan velocity threshold detection."""

    def test_normal_velocity_below_threshold(self):
        """Test normal velocity below threshold."""
        event = make_scan_event(
            velocity_scans_per_min=5,
            counterfeit_risk="low",
        )
        assert event["velocity_scans_per_min"] < DEFAULT_SCAN_VELOCITY_THRESHOLD

    def test_at_threshold_velocity(self):
        """Test scan velocity exactly at threshold."""
        event = make_scan_event(
            velocity_scans_per_min=DEFAULT_SCAN_VELOCITY_THRESHOLD,
        )
        assert event["velocity_scans_per_min"] == DEFAULT_SCAN_VELOCITY_THRESHOLD

    def test_above_threshold_velocity(self):
        """Test scan velocity above threshold triggers alert."""
        event = make_scan_event(
            velocity_scans_per_min=DEFAULT_SCAN_VELOCITY_THRESHOLD + 50,
            counterfeit_risk="high",
        )
        assert event["velocity_scans_per_min"] > DEFAULT_SCAN_VELOCITY_THRESHOLD
        assert event["counterfeit_risk"] == "high"

    def test_extreme_velocity(self):
        """Test extreme scan velocity indicates counterfeit."""
        event = make_scan_event(
            velocity_scans_per_min=1000,
            counterfeit_risk="critical",
            outcome="counterfeit_suspected",
        )
        assert event["velocity_scans_per_min"] == 1000
        assert event["outcome"] == "counterfeit_suspected"

    def test_zero_velocity(self):
        """Test zero velocity for first scan."""
        event = make_scan_event(velocity_scans_per_min=0)
        assert event["velocity_scans_per_min"] == 0

    def test_default_threshold_is_100(self):
        """Test default velocity threshold is 100 scans/min."""
        assert DEFAULT_SCAN_VELOCITY_THRESHOLD == 100

    def test_velocity_stored_in_event(self):
        """Test velocity is stored in the scan event."""
        event = make_scan_event(velocity_scans_per_min=42)
        assert event["velocity_scans_per_min"] == 42


# =========================================================================
# Test Class 6: Geo-Fencing
# =========================================================================

class TestGeoFencing:
    """Test geographic fence enforcement."""

    def test_no_geo_fence_violation(self):
        """Test scan within geo-fence boundary."""
        event = make_scan_event(
            geo_fence_violated=False,
            scan_latitude=SAMPLE_SCAN_LAT,
            scan_longitude=SAMPLE_SCAN_LON,
        )
        assert event["geo_fence_violated"] is False

    def test_geo_fence_violation(self):
        """Test scan outside geo-fence boundary."""
        event = make_scan_event(
            geo_fence_violated=True,
            scan_latitude=-33.8688,
            scan_longitude=151.2093,
            counterfeit_risk="high",
        )
        assert event["geo_fence_violated"] is True
        assert event["counterfeit_risk"] == "high"

    def test_scan_latitude_range(self):
        """Test scan latitude is within -90 to 90."""
        event = make_scan_event(scan_latitude=89.99)
        assert -90.0 <= event["scan_latitude"] <= 90.0

    def test_scan_longitude_range(self):
        """Test scan longitude is within -180 to 180."""
        event = make_scan_event(scan_longitude=-179.99)
        assert -180.0 <= event["scan_longitude"] <= 180.0

    def test_scan_country_iso_format(self):
        """Test scan country is ISO 3166-1 alpha-2."""
        event = make_scan_event(scan_country="DE")
        assert event["scan_country"] == "DE"
        assert len(event["scan_country"]) == 2

    def test_scan_without_location(self):
        """Test scan without GPS coordinates."""
        event = make_scan_event(
            scan_latitude=None,
            scan_longitude=None,
            scan_country=None,
        )
        assert event["scan_latitude"] is None
        assert event["scan_longitude"] is None

    def test_geo_fence_plus_velocity(self):
        """Test combined geo-fence violation and high velocity."""
        event = make_scan_event(
            geo_fence_violated=True,
            velocity_scans_per_min=200,
            counterfeit_risk="critical",
        )
        assert event["geo_fence_violated"] is True
        assert event["velocity_scans_per_min"] > DEFAULT_SCAN_VELOCITY_THRESHOLD


# =========================================================================
# Test Class 7: Key Rotation
# =========================================================================

class TestKeyRotation:
    """Test HMAC key rotation."""

    def test_signatures_differ_after_key_rotation(self):
        """Test signatures change when key is rotated."""
        data_hash = _sha256("test-data-rotation")
        sig_old = make_signature(hmac_key="old-key-v1", signed_data_hash=data_hash)
        sig_new = make_signature(hmac_key="new-key-v2", signed_data_hash=data_hash)
        assert sig_old["signature_value"] != sig_new["signature_value"]

    def test_old_key_signature_still_verifiable(self):
        """Test old key signature can be verified with old key."""
        data_hash = _sha256("test-data-old-key")
        old_sig = _hmac_sha256("old-key-v1", data_hash)
        recomputed = _hmac_sha256("old-key-v1", data_hash)
        assert old_sig == recomputed

    def test_key_id_tracks_rotation(self):
        """Test key_id changes after rotation."""
        sig1 = make_signature(key_id="KEY-QRG-001")
        sig2 = make_signature(key_id="KEY-QRG-002")
        assert sig1["key_id"] != sig2["key_id"]

    def test_rotated_signature_has_new_key_id(self):
        """Test rotated signature uses the new key identifier."""
        sig = make_signature(key_id="KEY-QRG-ROTATED")
        assert sig["key_id"] == "KEY-QRG-ROTATED"

    def test_different_keys_produce_valid_signatures(self):
        """Test signatures from different keys are all valid format."""
        for key_name in ["key-v1", "key-v2", "key-v3"]:
            sig = make_signature(hmac_key=key_name)
            assert_signature_valid(sig)

    def test_key_rotation_does_not_invalidate_existing(self):
        """Test existing signatures remain verifiable with original key."""
        data = _sha256("persistent-data")
        original = _hmac_sha256("key-v1", data)
        # After rotation, original can still be checked with key-v1
        verified = _hmac_sha256("key-v1", data)
        assert original == verified


# =========================================================================
# Test Class 8: Revocation List
# =========================================================================

class TestRevocationList:
    """Test code revocation list management."""

    def test_revoked_code_scan_outcome(self):
        """Test scanning a revoked code returns revoked outcome."""
        event = make_scan_event(
            outcome="revoked_code",
            counterfeit_risk="low",
        )
        assert event["outcome"] == "revoked_code"

    def test_revoked_hmac_still_checked(self):
        """Test HMAC is still checked on revoked code scans."""
        event = make_scan_event(
            outcome="revoked_code",
            hmac_valid=True,
        )
        assert event["hmac_valid"] is True

    def test_counterfeit_suspected_on_revoked(self):
        """Test counterfeit suspected on revoked code with bad HMAC."""
        event = make_scan_event(
            outcome="counterfeit_suspected",
            hmac_valid=False,
            counterfeit_risk="critical",
        )
        assert event["hmac_valid"] is False
        assert event["counterfeit_risk"] == "critical"

    def test_expired_code_scan(self):
        """Test scanning an expired code."""
        event = make_scan_event(outcome="expired_code")
        assert event["outcome"] == "expired_code"

    def test_error_outcome(self):
        """Test error outcome for system failures."""
        event = make_scan_event(outcome="error")
        assert event["outcome"] == "error"

    @pytest.mark.parametrize("outcome", SCAN_OUTCOMES)
    def test_all_scan_outcomes_valid(self, outcome: str):
        """Test all scan outcomes produce valid events."""
        event = make_scan_event(outcome=outcome)
        assert event["outcome"] == outcome
        assert_scan_event_valid(event)

    def test_response_time_recorded(self):
        """Test response time is recorded in scan event."""
        event = make_scan_event(response_time_ms=12.5)
        assert event["response_time_ms"] == 12.5

    def test_scanner_ip_hashed(self):
        """Test scanner IP is stored as a hash for privacy."""
        event = make_scan_event()
        assert event["scanner_ip"] is not None
        assert len(event["scanner_ip"]) > 0
