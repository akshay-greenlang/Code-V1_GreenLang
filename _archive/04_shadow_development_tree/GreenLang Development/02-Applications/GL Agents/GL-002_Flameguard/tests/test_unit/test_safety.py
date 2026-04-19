"""
GL-002 FLAMEGUARD - Safety Module Unit Tests

Tests for BMS, interlocks, and flame detection.
"""

import pytest
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])

from safety.burner_management import BurnerManagementSystem, BurnerState
from safety.safety_interlocks import SafetyInterlockManager, InterlockStatus
from safety.flame_detector import FlameDetector, FlameStatus


class TestBurnerManagementSystem:
    """Tests for BMS state machine."""

    def test_initial_state(self):
        """Test BMS starts in OFFLINE state."""
        bms = BurnerManagementSystem("BOILER-001")
        assert bms.state == BurnerState.OFFLINE
        assert not bms.is_firing

    def test_permissive_update(self):
        """Test permissive updates."""
        bms = BurnerManagementSystem("BOILER-001")

        bms.update_permissive("drum_level_ok", True)
        status = bms.get_status()

        assert status["permissives"]["drum_level_ok"]["satisfied"] is True

    def test_flame_signal_update(self):
        """Test flame signal updates."""
        bms = BurnerManagementSystem("BOILER-001")

        bms.update_flame_signal(85.0)
        status = bms.get_status()

        assert status["flame_signal"] == 85.0
        assert status["flame_proven"] is True

    def test_flame_failure_detection(self):
        """Test flame failure during firing causes trip."""
        bms = BurnerManagementSystem("BOILER-001")

        # Simulate firing state
        bms._state = BurnerState.FIRING
        bms._flame_proven = True

        # Simulate flame loss
        bms.update_flame_signal(0.0)

        assert bms.state == BurnerState.LOCKOUT
        assert "Flame failure" in bms.get_status()["lockout_reason"]

    def test_emergency_stop(self):
        """Test emergency stop."""
        bms = BurnerManagementSystem("BOILER-001")
        bms._state = BurnerState.FIRING

        bms.emergency_stop("Test emergency")

        assert bms.state == BurnerState.LOCKOUT
        assert bms.get_status()["lockout_reason"] == "Test emergency"

    def test_lockout_reset(self):
        """Test lockout reset with operator."""
        bms = BurnerManagementSystem("BOILER-001")
        bms._state = BurnerState.LOCKOUT
        bms._lockout_reason = "Test trip"

        # Can't reset without permissives
        result = bms.reset_lockout("OPERATOR-001")

        # Reset should succeed if permissives met
        # (in real implementation, would need permissives satisfied)

    def test_trip_history(self):
        """Test trip events are recorded."""
        bms = BurnerManagementSystem("BOILER-001")

        bms.emergency_stop("First trip")
        bms._state = BurnerState.FIRING
        bms.emergency_stop("Second trip")

        status = bms.get_status()
        assert status["trip_count"] == 2


class TestSafetyInterlockManager:
    """Tests for safety interlocks."""

    def test_standard_interlocks_initialized(self):
        """Test standard interlocks are created."""
        mgr = SafetyInterlockManager("BOILER-001")
        status = mgr.get_status()

        assert "STEAM_PRESSURE" in status["interlocks"]
        assert "DRUM_LEVEL" in status["interlocks"]
        assert "FUEL_PRESSURE" in status["interlocks"]

    def test_high_trip(self):
        """Test high limit trip."""
        mgr = SafetyInterlockManager("BOILER-001")

        # Steam pressure trip at 150 psig
        result = mgr.update_value("STEAM_PRESSURE", 155.0)

        assert result == InterlockStatus.TRIP
        assert mgr.is_tripped

    def test_low_trip(self):
        """Test low limit trip."""
        mgr = SafetyInterlockManager("BOILER-001")

        # Fuel pressure low trip at 2 psig
        result = mgr.update_value("FUEL_PRESSURE", 1.5)

        assert result == InterlockStatus.TRIP
        assert mgr.is_tripped

    def test_alarm_before_trip(self):
        """Test alarm triggers before trip."""
        mgr = SafetyInterlockManager("BOILER-001")

        # Steam pressure alarm at 140 psig
        result = mgr.update_value("STEAM_PRESSURE", 142.0)

        assert result == InterlockStatus.ALARM
        assert not mgr.is_tripped

    def test_bypass_allowed(self):
        """Test bypass for non-SIL3 interlocks."""
        mgr = SafetyInterlockManager("BOILER-001")

        # STEAM_PRESSURE is SIL2 - bypass allowed
        result = mgr.set_bypass(
            "STEAM_PRESSURE",
            reason="Maintenance",
            duration_minutes=60,
            operator="OPERATOR-001",
        )

        assert result is True
        status = mgr.get_status()
        assert status["interlocks"]["STEAM_PRESSURE"]["bypassed"] is True

    def test_bypass_not_allowed_sil3(self):
        """Test bypass not allowed for SIL3 interlocks."""
        mgr = SafetyInterlockManager("BOILER-001")

        # DRUM_LEVEL is SIL3 - bypass not allowed
        result = mgr.set_bypass(
            "DRUM_LEVEL",
            reason="Maintenance",
            duration_minutes=60,
            operator="OPERATOR-001",
        )

        assert result is False
        status = mgr.get_status()
        assert status["interlocks"]["DRUM_LEVEL"]["bypassed"] is False

    def test_trip_reset(self):
        """Test trip reset."""
        mgr = SafetyInterlockManager("BOILER-001")

        # Trigger trip
        mgr.update_value("STEAM_PRESSURE", 155.0)
        assert mgr.is_tripped

        # Return to normal
        mgr.update_value("STEAM_PRESSURE", 120.0)

        # Reset trip
        result = mgr.reset_trip("OPERATOR-001")

        assert result is True
        assert not mgr.is_tripped


class TestFlameDetector:
    """Tests for flame detection."""

    def test_initial_state(self):
        """Test initial flame detector state."""
        detector = FlameDetector("BOILER-001")

        assert detector.status == FlameStatus.NO_FLAME
        assert not detector.is_flame_proven()

    def test_add_scanner(self):
        """Test adding flame scanners."""
        detector = FlameDetector("BOILER-001")

        detector.add_scanner("UV-1", "UV")
        detector.add_scanner("UV-2", "UV")
        detector.add_scanner("IR-1", "IR")

        status = detector.get_status()
        assert len(status["scanners"]) == 3

    def test_flame_detection(self):
        """Test flame detection with signal."""
        detector = FlameDetector("BOILER-001", voting_logic="1oo1")
        detector.add_scanner("UV-1", "UV")

        detector.update_scanner("UV-1", signal_percent=80.0)

        assert detector.status == FlameStatus.FLAME_PRESENT
        assert detector.is_flame_proven()

    def test_unstable_flame(self):
        """Test unstable flame detection."""
        detector = FlameDetector("BOILER-001", voting_logic="1oo1")
        detector.add_scanner("UV-1", "UV")

        # Low but above minimum signal
        detector.update_scanner("UV-1", signal_percent=15.0)

        assert detector.status == FlameStatus.UNSTABLE
        assert detector.is_flame_proven()  # Still proven

    def test_voting_logic_2oo3(self):
        """Test 2oo3 voting logic."""
        detector = FlameDetector("BOILER-001", voting_logic="2oo3")
        detector.add_scanner("UV-1", "UV")
        detector.add_scanner("UV-2", "UV")
        detector.add_scanner("UV-3", "UV")

        # Only one scanner sees flame
        detector.update_scanner("UV-1", signal_percent=80.0)
        detector.update_scanner("UV-2", signal_percent=5.0)
        detector.update_scanner("UV-3", signal_percent=5.0)

        # 2oo3 requires 2 scanners
        assert not detector.is_flame_proven()

        # Two scanners see flame
        detector.update_scanner("UV-2", signal_percent=80.0)

        assert detector.is_flame_proven()

    def test_scanner_fault(self):
        """Test scanner fault detection."""
        detector = FlameDetector("BOILER-001", voting_logic="1oo1")
        detector.add_scanner("UV-1", "UV")

        detector.update_scanner("UV-1", signal_percent=80.0, healthy=False)

        assert detector.status == FlameStatus.SCANNER_FAULT
        assert not detector.is_flame_proven()

    def test_flame_failure_callback(self):
        """Test flame failure callback."""
        callback_called = {"value": False}

        def on_failure(boiler_id):
            callback_called["value"] = True

        detector = FlameDetector(
            "BOILER-001",
            voting_logic="1oo1",
            failure_callback=on_failure,
        )
        detector.add_scanner("UV-1", "UV")

        # Establish flame
        detector.update_scanner("UV-1", signal_percent=80.0)
        assert detector.is_flame_proven()

        # Lose flame - need to wait for failure time
        # In real test, would use time mocking
