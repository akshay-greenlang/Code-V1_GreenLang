"""
GL-002 FLAMEGUARD - Comprehensive Safety Interlock Tests

Tests for safety interlocks, SIL level enforcement, trip logic,
flame detection, and burner management system.
Targets 70%+ coverage with edge cases and state machine testing.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock, call
import time
import sys

sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])

from safety.safety_interlocks import (
    SafetyInterlockManager,
    SafetyInterlock,
    InterlockStatus,
)
from safety.flame_detector import (
    FlameDetector,
    FlameScanner,
    FlameStatus,
)


# =============================================================================
# SAFETY INTERLOCK MANAGER TESTS
# =============================================================================


class TestSafetyInterlockManagerInit:
    """Test SafetyInterlockManager initialization."""

    def test_basic_initialization(self):
        """Test manager initializes correctly."""
        manager = SafetyInterlockManager(boiler_id="BOILER-001")

        assert manager.boiler_id == "BOILER-001"
        assert not manager.is_tripped

    def test_standard_interlocks_created(self):
        """Test standard interlocks are initialized."""
        manager = SafetyInterlockManager(boiler_id="BOILER-001")
        status = manager.get_status()

        assert "STEAM_PRESSURE" in status["interlocks"]
        assert "DRUM_LEVEL" in status["interlocks"]
        assert "FUEL_PRESSURE" in status["interlocks"]
        assert "COMBUSTION_AIR" in status["interlocks"]
        assert "FLUE_GAS_TEMP" in status["interlocks"]

    def test_callbacks_stored(self):
        """Test callbacks are stored."""
        trip_cb = Mock()
        alarm_cb = Mock()

        manager = SafetyInterlockManager(
            boiler_id="BOILER-001",
            trip_callback=trip_cb,
            alarm_callback=alarm_cb,
        )

        assert manager._trip_callback == trip_cb
        assert manager._alarm_callback == alarm_cb

    def test_sil_levels_assigned(self):
        """Test SIL levels are correctly assigned."""
        manager = SafetyInterlockManager(boiler_id="BOILER-001")
        status = manager.get_status()

        # DRUM_LEVEL should be SIL3
        drum = manager._interlocks["DRUM_LEVEL"]
        assert drum.sil_level == 3

        # STEAM_PRESSURE should be SIL2
        steam = manager._interlocks["STEAM_PRESSURE"]
        assert steam.sil_level == 2


class TestInterlockValueUpdates:
    """Test interlock value updates and status changes."""

    @pytest.fixture
    def manager(self):
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_normal_value_returns_normal(self, manager):
        """Test normal values return NORMAL status."""
        status = manager.update_value("STEAM_PRESSURE", 120.0)
        assert status == InterlockStatus.NORMAL
        assert not manager.is_tripped

    def test_unknown_tag_returns_normal(self, manager):
        """Test unknown tag returns NORMAL."""
        status = manager.update_value("UNKNOWN_TAG", 100.0)
        assert status == InterlockStatus.NORMAL

    def test_value_stored(self, manager):
        """Test value is stored in interlock."""
        manager.update_value("STEAM_PRESSURE", 125.5)

        interlock = manager._interlocks["STEAM_PRESSURE"]
        assert interlock.current_value == 125.5


class TestHighLimitTrips:
    """Test high limit trip logic."""

    @pytest.fixture
    def manager(self):
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_high_trip_triggers(self, manager):
        """Test high trip triggers at setpoint."""
        # STEAM_PRESSURE trips at 150 psig
        status = manager.update_value("STEAM_PRESSURE", 155.0)

        assert status == InterlockStatus.TRIP
        assert manager.is_tripped

    def test_high_trip_at_exact_limit(self, manager):
        """Test trip at exact limit value."""
        status = manager.update_value("STEAM_PRESSURE", 150.0)
        assert status == InterlockStatus.TRIP

    def test_high_trip_just_below_limit(self, manager):
        """Test no trip just below limit."""
        status = manager.update_value("STEAM_PRESSURE", 149.9)
        assert status != InterlockStatus.TRIP

    def test_trip_causes_recorded(self, manager):
        """Test trip causes are recorded."""
        manager.update_value("STEAM_PRESSURE", 155.0)

        status = manager.get_status()
        assert len(status["trip_causes"]) > 0
        assert "STEAM_PRESSURE" in status["trip_causes"][0]

    @pytest.mark.parametrize("tag,trip_value", [
        ("STEAM_PRESSURE", 155.0),
        ("DRUM_LEVEL", 10.0),
        ("FUEL_PRESSURE", 30.0),
        ("FLUE_GAS_TEMP", 750.0),
    ])
    def test_multiple_high_trips(self, manager, tag, trip_value):
        """Test various high trip conditions."""
        status = manager.update_value(tag, trip_value)
        assert status == InterlockStatus.TRIP


class TestLowLimitTrips:
    """Test low limit trip logic."""

    @pytest.fixture
    def manager(self):
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_low_trip_triggers(self, manager):
        """Test low trip triggers at setpoint."""
        # FUEL_PRESSURE trips at 2 psig
        status = manager.update_value("FUEL_PRESSURE", 1.5)

        assert status == InterlockStatus.TRIP
        assert manager.is_tripped

    def test_low_trip_at_exact_limit(self, manager):
        """Test trip at exact low limit."""
        status = manager.update_value("FUEL_PRESSURE", 2.0)
        assert status == InterlockStatus.TRIP

    def test_low_trip_just_above_limit(self, manager):
        """Test no trip just above limit."""
        status = manager.update_value("FUEL_PRESSURE", 2.1)
        assert status != InterlockStatus.TRIP

    @pytest.mark.parametrize("tag,trip_value", [
        ("DRUM_LEVEL", -5.0),
        ("FUEL_PRESSURE", 1.0),
        ("COMBUSTION_AIR", 0.3),
    ])
    def test_multiple_low_trips(self, manager, tag, trip_value):
        """Test various low trip conditions."""
        status = manager.update_value(tag, trip_value)
        assert status == InterlockStatus.TRIP


class TestAlarmLogic:
    """Test alarm (pre-trip) logic."""

    @pytest.fixture
    def manager(self):
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_high_alarm_triggers(self, manager):
        """Test high alarm triggers before trip."""
        # STEAM_PRESSURE alarms at 140, trips at 150
        status = manager.update_value("STEAM_PRESSURE", 142.0)

        assert status == InterlockStatus.ALARM
        assert not manager.is_tripped

    def test_low_alarm_triggers(self, manager):
        """Test low alarm triggers before trip."""
        # FUEL_PRESSURE alarms at 5, trips at 2
        status = manager.update_value("FUEL_PRESSURE", 4.0)

        assert status == InterlockStatus.ALARM
        assert not manager.is_tripped

    def test_alarm_callback_invoked(self):
        """Test alarm callback is invoked."""
        alarm_cb = Mock()
        manager = SafetyInterlockManager(
            boiler_id="BOILER-001",
            alarm_callback=alarm_cb,
        )

        manager.update_value("STEAM_PRESSURE", 142.0)

        alarm_cb.assert_called_once()

    def test_alarm_callback_not_repeated(self):
        """Test alarm callback not repeated if already in alarm."""
        alarm_cb = Mock()
        manager = SafetyInterlockManager(
            boiler_id="BOILER-001",
            alarm_callback=alarm_cb,
        )

        manager.update_value("STEAM_PRESSURE", 142.0)
        manager.update_value("STEAM_PRESSURE", 143.0)

        # Should only be called once
        assert alarm_cb.call_count == 1

    def test_alarm_to_normal_transition(self, manager):
        """Test return to normal from alarm."""
        manager.update_value("STEAM_PRESSURE", 142.0)
        assert manager._interlocks["STEAM_PRESSURE"].status == InterlockStatus.ALARM

        status = manager.update_value("STEAM_PRESSURE", 120.0)
        assert status == InterlockStatus.NORMAL


class TestTripCallbacks:
    """Test trip callback behavior."""

    def test_trip_callback_invoked(self):
        """Test trip callback is invoked on trip."""
        trip_cb = Mock()
        manager = SafetyInterlockManager(
            boiler_id="BOILER-001",
            trip_callback=trip_cb,
        )

        manager.update_value("STEAM_PRESSURE", 155.0)

        trip_cb.assert_called_once()

    def test_trip_callback_receives_tag_and_message(self):
        """Test trip callback receives correct arguments."""
        trip_cb = Mock()
        manager = SafetyInterlockManager(
            boiler_id="BOILER-001",
            trip_callback=trip_cb,
        )

        manager.update_value("STEAM_PRESSURE", 155.0)

        args = trip_cb.call_args[0]
        assert args[0] == "STEAM_PRESSURE"
        assert "High trip" in args[1]


class TestBypassManagement:
    """Test interlock bypass functionality."""

    @pytest.fixture
    def manager(self):
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_bypass_allowed_sil2(self, manager):
        """Test bypass allowed for SIL2 interlock."""
        result = manager.set_bypass(
            tag="STEAM_PRESSURE",
            reason="Maintenance",
            duration_minutes=60,
            operator="OPERATOR-001",
        )

        assert result is True
        interlock = manager._interlocks["STEAM_PRESSURE"]
        assert interlock.bypassed is True
        assert interlock.status == InterlockStatus.BYPASSED

    def test_bypass_not_allowed_sil3(self, manager):
        """Test bypass not allowed for SIL3 interlock."""
        result = manager.set_bypass(
            tag="DRUM_LEVEL",
            reason="Maintenance",
            duration_minutes=60,
            operator="OPERATOR-001",
        )

        assert result is False
        interlock = manager._interlocks["DRUM_LEVEL"]
        assert interlock.bypassed is False

    def test_bypass_stores_reason(self, manager):
        """Test bypass reason is stored."""
        manager.set_bypass(
            tag="STEAM_PRESSURE",
            reason="Calibration",
            duration_minutes=30,
            operator="TECH-002",
        )

        interlock = manager._interlocks["STEAM_PRESSURE"]
        assert "Calibration" in interlock.bypass_reason
        assert "TECH-002" in interlock.bypass_reason

    def test_bypass_unknown_tag(self, manager):
        """Test bypass on unknown tag returns False."""
        result = manager.set_bypass(
            tag="UNKNOWN_TAG",
            reason="Test",
            duration_minutes=60,
            operator="OPERATOR-001",
        )

        assert result is False

    def test_bypassed_interlock_no_trip(self, manager):
        """Test bypassed interlock does not trip."""
        manager.set_bypass(
            tag="STEAM_PRESSURE",
            reason="Maintenance",
            duration_minutes=60,
            operator="OPERATOR-001",
        )

        status = manager.update_value("STEAM_PRESSURE", 200.0)

        assert status == InterlockStatus.BYPASSED
        assert not manager.is_tripped

    def test_clear_bypass(self, manager):
        """Test clearing individual bypass."""
        manager.set_bypass(
            tag="STEAM_PRESSURE",
            reason="Maintenance",
            duration_minutes=60,
            operator="OPERATOR-001",
        )

        manager._clear_bypass("STEAM_PRESSURE")

        interlock = manager._interlocks["STEAM_PRESSURE"]
        assert interlock.bypassed is False
        assert interlock.status == InterlockStatus.NORMAL

    def test_clear_all_bypasses(self, manager):
        """Test clearing all bypasses."""
        manager.set_bypass("STEAM_PRESSURE", "Test", 60, "OP1")
        manager.set_bypass("FUEL_PRESSURE", "Test", 60, "OP1")

        manager.clear_all_bypasses()

        assert manager._interlocks["STEAM_PRESSURE"].bypassed is False
        assert manager._interlocks["FUEL_PRESSURE"].bypassed is False

    def test_bypass_count_in_status(self, manager):
        """Test bypassed count in status."""
        manager.set_bypass("STEAM_PRESSURE", "Test", 60, "OP1")
        manager.set_bypass("FUEL_PRESSURE", "Test", 60, "OP1")

        status = manager.get_status()
        assert status["bypassed_count"] == 2


class TestTripReset:
    """Test trip reset functionality."""

    @pytest.fixture
    def manager(self):
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_reset_when_not_tripped(self, manager):
        """Test reset returns True when not tripped."""
        result = manager.reset_trip("OPERATOR-001")
        assert result is True

    def test_reset_requires_normal_conditions(self, manager):
        """Test reset fails if conditions still abnormal."""
        manager.update_value("STEAM_PRESSURE", 155.0)
        assert manager.is_tripped

        # Still in trip condition
        result = manager.reset_trip("OPERATOR-001")
        assert result is False
        assert manager.is_tripped

    def test_reset_succeeds_after_normal(self, manager):
        """Test reset succeeds after conditions normalize."""
        manager.update_value("STEAM_PRESSURE", 155.0)
        assert manager.is_tripped

        # Return to normal
        manager.update_value("STEAM_PRESSURE", 120.0)

        result = manager.reset_trip("OPERATOR-001")
        assert result is True
        assert not manager.is_tripped

    def test_reset_clears_trip_causes(self, manager):
        """Test reset clears trip causes list."""
        manager.update_value("STEAM_PRESSURE", 155.0)
        manager.update_value("STEAM_PRESSURE", 120.0)
        manager.reset_trip("OPERATOR-001")

        status = manager.get_status()
        assert len(status["trip_causes"]) == 0


class TestStatusReporting:
    """Test status reporting functionality."""

    @pytest.fixture
    def manager(self):
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_status_contains_required_fields(self, manager):
        """Test status contains all required fields."""
        status = manager.get_status()

        assert "boiler_id" in status
        assert "tripped" in status
        assert "trip_causes" in status
        assert "interlocks" in status
        assert "bypassed_count" in status
        assert "alarm_count" in status

    def test_interlock_status_details(self, manager):
        """Test interlock details in status."""
        manager.update_value("STEAM_PRESSURE", 130.0)

        status = manager.get_status()
        steam = status["interlocks"]["STEAM_PRESSURE"]

        assert "value" in steam
        assert "status" in steam
        assert "unit" in steam
        assert "bypassed" in steam
        assert steam["value"] == 130.0

    def test_alarm_count(self, manager):
        """Test alarm count in status."""
        manager.update_value("STEAM_PRESSURE", 142.0)  # Alarm
        manager.update_value("FUEL_PRESSURE", 4.0)  # Alarm

        status = manager.get_status()
        assert status["alarm_count"] == 2


# =============================================================================
# FLAME DETECTOR TESTS
# =============================================================================


class TestFlameDetectorInit:
    """Test FlameDetector initialization."""

    def test_basic_initialization(self):
        """Test detector initializes correctly."""
        detector = FlameDetector(boiler_id="BOILER-001")

        assert detector.boiler_id == "BOILER-001"
        assert detector.status == FlameStatus.NO_FLAME
        assert not detector.is_flame_proven()

    def test_voting_logic_stored(self):
        """Test voting logic is stored."""
        detector = FlameDetector(boiler_id="BOILER-001", voting_logic="2oo3")
        assert detector.voting_logic == "2oo3"

    def test_failure_callback_stored(self):
        """Test failure callback is stored."""
        callback = Mock()
        detector = FlameDetector(
            boiler_id="BOILER-001",
            failure_callback=callback,
        )
        assert detector._failure_callback == callback

    def test_constants_set(self):
        """Test class constants are set."""
        assert FlameDetector.FLAME_FAILURE_TIME_S == 4.0
        assert FlameDetector.MIN_SIGNAL_PERCENT == 10.0
        assert FlameDetector.UNSTABLE_THRESHOLD == 20.0


class TestFlameScanner:
    """Test FlameScanner operations."""

    @pytest.fixture
    def detector(self):
        return FlameDetector(boiler_id="BOILER-001")

    def test_add_scanner(self, detector):
        """Test adding a scanner."""
        detector.add_scanner("UV-1", "UV")

        status = detector.get_status()
        assert "UV-1" in status["scanners"]
        assert status["scanners"]["UV-1"]["type"] == "UV"

    def test_add_multiple_scanners(self, detector):
        """Test adding multiple scanners."""
        detector.add_scanner("UV-1", "UV")
        detector.add_scanner("UV-2", "UV")
        detector.add_scanner("IR-1", "IR")

        status = detector.get_status()
        assert len(status["scanners"]) == 3

    def test_update_scanner(self, detector):
        """Test updating scanner signal."""
        detector.add_scanner("UV-1", "UV")
        detector.update_scanner("UV-1", signal_percent=75.0)

        status = detector.get_status()
        assert status["scanners"]["UV-1"]["signal"] == 75.0

    def test_update_unknown_scanner(self, detector):
        """Test updating unknown scanner does nothing."""
        detector.update_scanner("UNKNOWN", signal_percent=75.0)
        # Should not raise exception

    def test_scanner_healthy_status(self, detector):
        """Test scanner healthy status update."""
        detector.add_scanner("UV-1", "UV")
        detector.update_scanner("UV-1", signal_percent=75.0, healthy=False)

        status = detector.get_status()
        assert status["scanners"]["UV-1"]["healthy"] is False


class TestFlameDetection:
    """Test flame detection logic."""

    @pytest.fixture
    def detector(self):
        det = FlameDetector(boiler_id="BOILER-001", voting_logic="1oo1")
        det.add_scanner("UV-1", "UV")
        return det

    def test_flame_detected_good_signal(self, detector):
        """Test flame detected with good signal."""
        detector.update_scanner("UV-1", signal_percent=80.0)

        assert detector.status == FlameStatus.FLAME_PRESENT
        assert detector.is_flame_proven()

    def test_no_flame_low_signal(self, detector):
        """Test no flame with low signal."""
        detector.update_scanner("UV-1", signal_percent=5.0)

        assert detector.status == FlameStatus.NO_FLAME
        assert not detector.is_flame_proven()

    def test_unstable_flame(self, detector):
        """Test unstable flame detection."""
        detector.update_scanner("UV-1", signal_percent=15.0)  # Between 10 and 20

        assert detector.status == FlameStatus.UNSTABLE
        assert detector.is_flame_proven()  # Still proven

    def test_flame_at_minimum_threshold(self, detector):
        """Test flame at minimum threshold."""
        detector.update_scanner("UV-1", signal_percent=10.0)

        assert detector.is_flame_proven()

    def test_combined_signal_percent(self, detector):
        """Test combined signal percentage calculation."""
        detector.add_scanner("UV-2", "UV")
        detector.update_scanner("UV-1", signal_percent=80.0)
        detector.update_scanner("UV-2", signal_percent=60.0)

        assert detector.signal_percent == 70.0  # Average


class TestVotingLogic:
    """Test flame detection voting logic."""

    def test_1oo1_single_scanner(self):
        """Test 1oo1 with single scanner."""
        detector = FlameDetector(boiler_id="BOILER-001", voting_logic="1oo1")
        detector.add_scanner("UV-1", "UV")

        detector.update_scanner("UV-1", signal_percent=80.0)
        assert detector.is_flame_proven()

        detector.update_scanner("UV-1", signal_percent=5.0)
        assert not detector.is_flame_proven()

    def test_1oo2_any_scanner(self):
        """Test 1oo2 requires any scanner."""
        detector = FlameDetector(boiler_id="BOILER-001", voting_logic="1oo2")
        detector.add_scanner("UV-1", "UV")
        detector.add_scanner("UV-2", "UV")

        # One sees flame
        detector.update_scanner("UV-1", signal_percent=80.0)
        detector.update_scanner("UV-2", signal_percent=5.0)

        assert detector.is_flame_proven()

    def test_2oo2_both_scanners(self):
        """Test 2oo2 requires both scanners."""
        detector = FlameDetector(boiler_id="BOILER-001", voting_logic="2oo2")
        detector.add_scanner("UV-1", "UV")
        detector.add_scanner("UV-2", "UV")

        # Only one sees flame
        detector.update_scanner("UV-1", signal_percent=80.0)
        detector.update_scanner("UV-2", signal_percent=5.0)
        assert not detector.is_flame_proven()

        # Both see flame
        detector.update_scanner("UV-2", signal_percent=80.0)
        assert detector.is_flame_proven()

    def test_2oo3_majority_voting(self):
        """Test 2oo3 requires majority."""
        detector = FlameDetector(boiler_id="BOILER-001", voting_logic="2oo3")
        detector.add_scanner("UV-1", "UV")
        detector.add_scanner("UV-2", "UV")
        detector.add_scanner("UV-3", "UV")

        # One scanner sees flame
        detector.update_scanner("UV-1", signal_percent=80.0)
        detector.update_scanner("UV-2", signal_percent=5.0)
        detector.update_scanner("UV-3", signal_percent=5.0)
        assert not detector.is_flame_proven()

        # Two scanners see flame
        detector.update_scanner("UV-2", signal_percent=80.0)
        assert detector.is_flame_proven()

    def test_2oo3_with_two_scanners(self):
        """Test 2oo3 with only two scanners (degraded)."""
        detector = FlameDetector(boiler_id="BOILER-001", voting_logic="2oo3")
        detector.add_scanner("UV-1", "UV")
        detector.add_scanner("UV-2", "UV")

        # Falls back to 2oo2 behavior
        detector.update_scanner("UV-1", signal_percent=80.0)
        detector.update_scanner("UV-2", signal_percent=5.0)
        assert not detector.is_flame_proven()

        detector.update_scanner("UV-2", signal_percent=80.0)
        assert detector.is_flame_proven()


class TestScannerFaults:
    """Test scanner fault handling."""

    @pytest.fixture
    def detector(self):
        det = FlameDetector(boiler_id="BOILER-001", voting_logic="2oo3")
        det.add_scanner("UV-1", "UV")
        det.add_scanner("UV-2", "UV")
        det.add_scanner("UV-3", "UV")
        return det

    def test_scanner_fault_excludes_from_voting(self, detector):
        """Test faulty scanner excluded from voting."""
        detector.update_scanner("UV-1", signal_percent=80.0, healthy=True)
        detector.update_scanner("UV-2", signal_percent=80.0, healthy=True)
        detector.update_scanner("UV-3", signal_percent=80.0, healthy=False)  # Fault

        # Should still prove flame (2 healthy scanners)
        assert detector.is_flame_proven()

    def test_all_scanners_faulty(self, detector):
        """Test all scanners faulty."""
        detector.update_scanner("UV-1", signal_percent=80.0, healthy=False)
        detector.update_scanner("UV-2", signal_percent=80.0, healthy=False)
        detector.update_scanner("UV-3", signal_percent=80.0, healthy=False)

        assert detector.status == FlameStatus.SCANNER_FAULT
        assert not detector.is_flame_proven()

    def test_no_scanners_fault_status(self):
        """Test no scanners returns fault."""
        detector = FlameDetector(boiler_id="BOILER-001")
        # No scanners added

        detector._evaluate_flame_status()

        assert detector.status == FlameStatus.NO_FLAME


class TestFlameFailure:
    """Test flame failure detection."""

    @pytest.fixture
    def detector(self):
        det = FlameDetector(boiler_id="BOILER-001", voting_logic="1oo1")
        det.add_scanner("UV-1", "UV")
        return det

    def test_flame_loss_starts_timer(self, detector):
        """Test flame loss starts failure timer."""
        detector.update_scanner("UV-1", signal_percent=80.0)
        assert detector.is_flame_proven()

        detector.update_scanner("UV-1", signal_percent=5.0)
        assert detector._flame_loss_time is not None

    def test_failure_callback_after_timeout(self):
        """Test failure callback invoked after timeout."""
        callback = Mock()
        detector = FlameDetector(
            boiler_id="BOILER-001",
            voting_logic="1oo1",
            failure_callback=callback,
        )
        detector.add_scanner("UV-1", "UV")

        # Establish flame
        detector.update_scanner("UV-1", signal_percent=80.0)

        # Lose flame and simulate timeout
        detector._flame_loss_time = datetime.now(timezone.utc) - timedelta(seconds=5)
        detector.update_scanner("UV-1", signal_percent=5.0)

        callback.assert_called_once_with("BOILER-001")

    def test_flame_events_recorded(self):
        """Test flame events are recorded."""
        detector = FlameDetector(boiler_id="BOILER-001", voting_logic="1oo1")
        detector.add_scanner("UV-1", "UV")

        detector.update_scanner("UV-1", signal_percent=80.0)
        detector._flame_loss_time = datetime.now(timezone.utc) - timedelta(seconds=5)
        detector.update_scanner("UV-1", signal_percent=5.0)

        status = detector.get_status()
        assert status["event_count"] >= 1


class TestFlameDetectorStatus:
    """Test flame detector status reporting."""

    @pytest.fixture
    def detector(self):
        det = FlameDetector(boiler_id="BOILER-001", voting_logic="2oo3")
        det.add_scanner("UV-1", "UV")
        det.add_scanner("UV-2", "UV")
        return det

    def test_status_contains_required_fields(self, detector):
        """Test status contains all fields."""
        status = detector.get_status()

        assert "boiler_id" in status
        assert "status" in status
        assert "flame_proven" in status
        assert "signal_percent" in status
        assert "voting_logic" in status
        assert "scanners" in status
        assert "event_count" in status

    def test_status_values(self, detector):
        """Test status values are correct."""
        detector.update_scanner("UV-1", signal_percent=75.0)
        detector.update_scanner("UV-2", signal_percent=65.0)

        status = detector.get_status()

        assert status["boiler_id"] == "BOILER-001"
        assert status["voting_logic"] == "2oo3"
        assert status["signal_percent"] == 70.0


class TestFlameStatusEnum:
    """Test FlameStatus enum values."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert FlameStatus.NO_FLAME.value == "no_flame"
        assert FlameStatus.FLAME_PRESENT.value == "flame_present"
        assert FlameStatus.UNSTABLE.value == "unstable"
        assert FlameStatus.SCANNER_FAULT.value == "scanner_fault"


class TestInterlockStatusEnum:
    """Test InterlockStatus enum values."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert InterlockStatus.NORMAL.value == "normal"
        assert InterlockStatus.ALARM.value == "alarm"
        assert InterlockStatus.TRIP.value == "trip"
        assert InterlockStatus.BYPASSED.value == "bypassed"


# =============================================================================
# SAFETY INTERLOCK DATACLASS TESTS
# =============================================================================


class TestSafetyInterlockDataclass:
    """Test SafetyInterlock dataclass."""

    def test_default_values(self):
        """Test default values are set."""
        interlock = SafetyInterlock(
            tag="TEST_TAG",
            description="Test interlock",
        )

        assert interlock.trip_high is None
        assert interlock.trip_low is None
        assert interlock.alarm_high is None
        assert interlock.alarm_low is None
        assert interlock.unit == ""
        assert interlock.sil_level == 2
        assert interlock.current_value == 0.0
        assert interlock.status == InterlockStatus.NORMAL
        assert interlock.bypassed is False
        assert interlock.bypass_reason is None
        assert interlock.bypass_expiry is None

    def test_custom_values(self):
        """Test custom values are set."""
        interlock = SafetyInterlock(
            tag="PRESSURE",
            description="Pressure interlock",
            trip_high=150.0,
            trip_low=10.0,
            alarm_high=140.0,
            alarm_low=20.0,
            unit="psig",
            sil_level=3,
        )

        assert interlock.tag == "PRESSURE"
        assert interlock.trip_high == 150.0
        assert interlock.trip_low == 10.0
        assert interlock.sil_level == 3


# =============================================================================
# EDGE CASES AND STATE MACHINE TESTS
# =============================================================================


class TestInterlockStateMachine:
    """Test interlock state machine transitions."""

    @pytest.fixture
    def manager(self):
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_normal_to_alarm_to_trip(self, manager):
        """Test progression from normal to alarm to trip."""
        status = manager.update_value("STEAM_PRESSURE", 120.0)
        assert status == InterlockStatus.NORMAL

        status = manager.update_value("STEAM_PRESSURE", 142.0)
        assert status == InterlockStatus.ALARM

        status = manager.update_value("STEAM_PRESSURE", 155.0)
        assert status == InterlockStatus.TRIP

    def test_trip_to_normal_requires_reset(self, manager):
        """Test trip to normal requires reset."""
        manager.update_value("STEAM_PRESSURE", 155.0)
        assert manager.is_tripped

        manager.update_value("STEAM_PRESSURE", 120.0)
        # Still tripped until reset
        assert manager.is_tripped

        manager.reset_trip("OPERATOR")
        assert not manager.is_tripped

    def test_bypass_overrides_all_states(self, manager):
        """Test bypass overrides alarm and trip states."""
        manager.set_bypass("STEAM_PRESSURE", "Test", 60, "OP")

        # Values that would normally trip
        status = manager.update_value("STEAM_PRESSURE", 200.0)
        assert status == InterlockStatus.BYPASSED
        assert not manager.is_tripped

    def test_multiple_trips_recorded(self, manager):
        """Test multiple trip causes recorded."""
        manager.update_value("STEAM_PRESSURE", 155.0)
        manager.update_value("FUEL_PRESSURE", 1.0)

        status = manager.get_status()
        assert len(status["trip_causes"]) >= 2


class TestConcurrentUpdates:
    """Test concurrent value updates."""

    @pytest.fixture
    def manager(self):
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_rapid_updates(self, manager):
        """Test rapid value updates."""
        for i in range(100):
            value = 100.0 + (i % 60)  # Oscillate 100-159
            manager.update_value("STEAM_PRESSURE", value)

        # Should handle without error
        status = manager.get_status()
        assert "STEAM_PRESSURE" in status["interlocks"]

    def test_update_all_interlocks(self, manager):
        """Test updating all interlocks."""
        manager.update_value("STEAM_PRESSURE", 130.0)
        manager.update_value("DRUM_LEVEL", 0.5)
        manager.update_value("FUEL_PRESSURE", 15.0)
        manager.update_value("COMBUSTION_AIR", 2.0)
        manager.update_value("FLUE_GAS_TEMP", 450.0)

        status = manager.get_status()
        assert status["alarm_count"] == 0
        assert not status["tripped"]
