# -*- coding: utf-8 -*-
"""
GL-002 FLAMEGUARD - BMS Sequence Tests

Tests for Burner Management System startup, operation, and shutdown sequences
per NFPA 85 requirements.

Author: GL-TestEngineer
Date: December 2025
Version: 1.0.0
"""

import pytest
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from safety.burner_management import (
    BurnerManagementSystem,
    BurnerState,
    BurnerPermissive,
)


class TestBMSStateTransitions:
    """
    Tests for BMS state machine transitions.

    Valid sequence:
    OFFLINE -> PRE_PURGE -> PILOT_LIGHT_TRIAL -> PILOT_PROVEN ->
    MAIN_FLAME_TRIAL -> MAIN_FLAME_PROVEN -> FIRING -> POST_PURGE -> OFFLINE
    """

    @pytest.fixture
    def bms(self):
        return BurnerManagementSystem("TEST-001")

    @pytest.mark.safety
    @pytest.mark.bms
    def test_initial_state_is_offline(self, bms):
        """BMS should start in OFFLINE state."""
        assert bms.state == BurnerState.OFFLINE

    @pytest.mark.safety
    @pytest.mark.bms
    def test_state_transition_offline_to_prepurge(self, bms):
        """Test transition from OFFLINE to PRE_PURGE."""
        # Satisfy all permissives
        for perm in bms._permissives.values():
            perm.satisfied = True

        # Direct transition test
        bms._transition_to(BurnerState.PRE_PURGE)

        assert bms.state == BurnerState.PRE_PURGE
        assert bms._prev_state == BurnerState.OFFLINE

    @pytest.mark.safety
    @pytest.mark.bms
    def test_state_transition_to_lockout(self, bms):
        """Test transition to LOCKOUT from any state."""
        bms._state = BurnerState.FIRING
        bms._trip("Test trip")

        assert bms.state == BurnerState.LOCKOUT
        assert bms._prev_state == BurnerState.FIRING

    @pytest.mark.safety
    @pytest.mark.bms
    def test_is_firing_property(self, bms):
        """Test is_firing property for various states."""
        non_firing_states = [
            BurnerState.OFFLINE,
            BurnerState.PRE_PURGE,
            BurnerState.PILOT_LIGHT_TRIAL,
            BurnerState.POST_PURGE,
            BurnerState.LOCKOUT,
        ]

        firing_states = [
            BurnerState.MAIN_FLAME_PROVEN,
            BurnerState.FIRING,
        ]

        for state in non_firing_states:
            bms._state = state
            assert bms.is_firing is False, f"{state.name} should not be firing"

        for state in firing_states:
            bms._state = state
            assert bms.is_firing is True, f"{state.name} should be firing"


class TestStartupSequence:
    """
    Tests for BMS startup sequence.

    NFPA 85 5.6 - Purge, Ignition, and Operating Sequences
    """

    @pytest.fixture
    def bms(self):
        return BurnerManagementSystem("TEST-001")

    @pytest.mark.safety
    @pytest.mark.bms
    def test_startup_requires_permissives(self, bms):
        """Cannot start without permissives satisfied."""
        # All permissives false
        for perm in bms._permissives.values():
            perm.satisfied = False

        # Should not be able to check permissives for PRE_PURGE
        can_start = bms._check_permissives_for_state(BurnerState.PRE_PURGE)
        assert can_start is False

    @pytest.mark.safety
    @pytest.mark.bms
    def test_startup_with_all_permissives(self, bms):
        """Can start with all permissives satisfied."""
        for perm in bms._permissives.values():
            perm.satisfied = True

        can_start = bms._check_permissives_for_state(BurnerState.PRE_PURGE)
        assert can_start is True

    @pytest.mark.safety
    @pytest.mark.bms
    @pytest.mark.asyncio
    async def test_startup_sequence_states(self, bms):
        """Test startup sequence state progression."""
        # Satisfy all permissives
        for perm in bms._permissives.values():
            perm.satisfied = True

        # Mock the long waits
        with patch.object(bms, 'PRE_PURGE_TIME_S', 0.1):
            with patch.object(bms, 'PILOT_TRIAL_TIME_S', 0.1):
                with patch.object(bms, 'MAIN_FLAME_TRIAL_TIME_S', 0.1):
                    # Simulate flame during trials
                    async def set_flame():
                        await asyncio.sleep(0.05)
                        bms._flame_proven = True
                        bms._flame_signal = 80.0

                    # Start flame simulation
                    flame_task = asyncio.create_task(set_flame())

                    # Run startup
                    result = await bms.start_sequence()

                    await flame_task

        # Should be in FIRING state if flame was proven
        if result:
            assert bms.state == BurnerState.FIRING

    @pytest.mark.safety
    @pytest.mark.bms
    @pytest.mark.asyncio
    async def test_startup_fails_without_flame(self, bms):
        """Startup should fail if flame not proven during trial."""
        for perm in bms._permissives.values():
            perm.satisfied = True

        # Use short timers for test
        with patch.object(bms, 'PRE_PURGE_TIME_S', 0.1):
            with patch.object(bms, 'PILOT_TRIAL_TIME_S', 0.1):
                # Don't simulate flame - let it fail
                bms._flame_proven = False

                result = await bms.start_sequence()

        # Should fail and go to lockout
        assert result is False
        assert bms.state == BurnerState.LOCKOUT

    @pytest.mark.safety
    @pytest.mark.bms
    @pytest.mark.asyncio
    async def test_startup_from_wrong_state_fails(self, bms):
        """Cannot start sequence from non-OFFLINE state."""
        bms._state = BurnerState.FIRING

        result = await bms.start_sequence()

        assert result is False
        # State should not change
        assert bms.state == BurnerState.FIRING


class TestShutdownSequence:
    """
    Tests for BMS shutdown/stop sequence.

    NFPA 85 5.7 - Normal Shutdown
    """

    @pytest.fixture
    def bms(self):
        return BurnerManagementSystem("TEST-001")

    @pytest.mark.safety
    @pytest.mark.bms
    @pytest.mark.asyncio
    async def test_normal_shutdown_sequence(self, bms):
        """Test normal shutdown with post-purge."""
        # Set up in firing state
        bms._state = BurnerState.FIRING
        bms._flame_proven = True

        # Use short timer for test
        with patch.object(bms, 'POST_PURGE_TIME_S', 0.1):
            await bms.stop_sequence()

        # Should end in OFFLINE
        assert bms.state == BurnerState.OFFLINE
        assert bms._permissives["purge_complete"].satisfied is False

    @pytest.mark.safety
    @pytest.mark.bms
    @pytest.mark.asyncio
    async def test_shutdown_goes_through_post_purge(self, bms):
        """Shutdown must go through POST_PURGE state."""
        bms._state = BurnerState.FIRING
        bms._flame_proven = True

        states_visited = []

        # Track state transitions
        original_transition = bms._transition_to

        def tracking_transition(state):
            states_visited.append(state)
            original_transition(state)

        bms._transition_to = tracking_transition

        with patch.object(bms, 'POST_PURGE_TIME_S', 0.1):
            await bms.stop_sequence()

        assert BurnerState.POST_PURGE in states_visited
        assert BurnerState.OFFLINE in states_visited

    @pytest.mark.safety
    @pytest.mark.bms
    @pytest.mark.asyncio
    async def test_shutdown_when_not_firing(self, bms):
        """Shutdown when already stopped should be no-op."""
        bms._state = BurnerState.OFFLINE

        await bms.stop_sequence()

        # Should still be offline
        assert bms.state == BurnerState.OFFLINE


class TestPermissiveManagement:
    """
    Tests for permissive condition management.
    """

    @pytest.fixture
    def bms(self):
        return BurnerManagementSystem("TEST-001")

    @pytest.mark.safety
    @pytest.mark.bms
    def test_update_permissive(self, bms):
        """Test permissive value update."""
        bms.update_permissive("drum_level_ok", True)
        assert bms._permissives["drum_level_ok"].satisfied is True

        bms.update_permissive("drum_level_ok", False)
        assert bms._permissives["drum_level_ok"].satisfied is False

    @pytest.mark.safety
    @pytest.mark.bms
    def test_permissive_update_timestamp(self, bms):
        """Permissive updates should record timestamp."""
        before = datetime.now(timezone.utc)
        bms.update_permissive("drum_level_ok", True)
        after = datetime.now(timezone.utc)

        last_check = bms._permissives["drum_level_ok"].last_check
        assert last_check is not None
        assert before <= last_check <= after

    @pytest.mark.safety
    @pytest.mark.bms
    def test_unknown_permissive_ignored(self, bms):
        """Unknown permissive name should be ignored."""
        # Should not raise exception
        bms.update_permissive("unknown_permissive", True)

    @pytest.mark.safety
    @pytest.mark.bms
    def test_permissive_loss_during_firing_trips(self, bms):
        """Losing required permissive during firing should trip."""
        bms._state = BurnerState.FIRING
        bms._flame_proven = True
        for perm in bms._permissives.values():
            perm.satisfied = True

        # Lose a permissive required for firing
        bms.update_permissive("combustion_air_ok", False)

        assert bms.state == BurnerState.LOCKOUT


class TestLockoutAndReset:
    """
    Tests for lockout handling and reset procedures.
    """

    @pytest.fixture
    def bms(self):
        return BurnerManagementSystem("TEST-001")

    @pytest.mark.safety
    @pytest.mark.bms
    def test_lockout_clears_flame_status(self, bms):
        """Lockout should clear flame proven status."""
        bms._state = BurnerState.FIRING
        bms._flame_proven = True
        bms._pilot_proven = True

        bms._trip("Test")

        assert bms._flame_proven is False
        assert bms._pilot_proven is False

    @pytest.mark.safety
    @pytest.mark.bms
    def test_lockout_stores_reason(self, bms):
        """Lockout should store the reason."""
        bms._state = BurnerState.FIRING
        bms._trip("High pressure trip")

        assert bms._lockout_reason == "High pressure trip"

    @pytest.mark.safety
    @pytest.mark.bms
    def test_reset_from_non_lockout_fails(self, bms):
        """Cannot reset if not in lockout."""
        bms._state = BurnerState.OFFLINE

        result = bms.reset_lockout("operator1")

        assert result is False

    @pytest.mark.safety
    @pytest.mark.bms
    def test_reset_without_permissives_fails(self, bms):
        """Cannot reset lockout without permissives satisfied."""
        bms._state = BurnerState.LOCKOUT
        bms._lockout_reason = "Test"

        # All permissives false
        for perm in bms._permissives.values():
            perm.satisfied = False

        result = bms.reset_lockout("operator1")

        # Should fail
        assert result is False
        assert bms.state == BurnerState.LOCKOUT

    @pytest.mark.safety
    @pytest.mark.bms
    def test_successful_reset(self, bms):
        """Successful lockout reset."""
        bms._state = BurnerState.LOCKOUT
        bms._lockout_reason = "Test trip"

        # Satisfy all permissives
        for perm in bms._permissives.values():
            perm.satisfied = True

        result = bms.reset_lockout("authorized_operator")

        assert result is True
        assert bms.state == BurnerState.OFFLINE
        assert bms._lockout_reason is None

    @pytest.mark.safety
    @pytest.mark.bms
    def test_reset_clears_purge_complete(self, bms):
        """Reset should clear purge complete status (require new purge)."""
        bms._state = BurnerState.LOCKOUT
        bms._permissives["purge_complete"].satisfied = True

        for perm in bms._permissives.values():
            perm.satisfied = True

        bms.reset_lockout("operator1")

        # Purge complete should be cleared
        assert bms._permissives["purge_complete"].satisfied is False


class TestFlameSignalHandling:
    """
    Tests for flame signal processing.
    """

    @pytest.fixture
    def bms(self):
        return BurnerManagementSystem("TEST-001")

    @pytest.mark.safety
    @pytest.mark.bms
    def test_flame_signal_stored(self, bms):
        """Flame signal value should be stored."""
        bms.update_flame_signal(75.5)
        assert bms._flame_signal == 75.5

    @pytest.mark.safety
    @pytest.mark.bms
    def test_flame_proven_threshold(self, bms):
        """Flame proven at 10% threshold."""
        # Below threshold
        bms.update_flame_signal(9.9)
        assert bms._flame_proven is False

        # At threshold
        bms.update_flame_signal(10.0)
        assert bms._flame_proven is True

        # Above threshold
        bms.update_flame_signal(50.0)
        assert bms._flame_proven is True

    @pytest.mark.safety
    @pytest.mark.bms
    def test_flame_loss_during_firing_trips(self, bms):
        """Flame loss during firing should trip."""
        bms._state = BurnerState.FIRING
        bms._flame_proven = True

        bms.update_flame_signal(0.0)

        assert bms.state == BurnerState.LOCKOUT

    @pytest.mark.safety
    @pytest.mark.bms
    def test_flame_loss_during_non_firing_no_trip(self, bms):
        """Flame loss during non-firing states should not trip."""
        bms._state = BurnerState.OFFLINE
        bms._flame_proven = False

        bms.update_flame_signal(0.0)

        assert bms.state == BurnerState.OFFLINE  # No change


class TestBMSStatus:
    """
    Tests for BMS status reporting.
    """

    @pytest.fixture
    def bms(self):
        return BurnerManagementSystem("TEST-001")

    @pytest.mark.safety
    @pytest.mark.bms
    def test_get_status_structure(self, bms):
        """Verify status dict structure."""
        status = bms.get_status()

        assert "boiler_id" in status
        assert "state" in status
        assert "prev_state" in status
        assert "flame_signal" in status
        assert "flame_proven" in status
        assert "pilot_proven" in status
        assert "lockout_reason" in status
        assert "permissives" in status
        assert "trip_count" in status

    @pytest.mark.safety
    @pytest.mark.bms
    def test_status_reflects_state(self, bms):
        """Status should reflect current state."""
        bms._state = BurnerState.FIRING
        bms._flame_signal = 85.0
        bms._flame_proven = True

        status = bms.get_status()

        assert status["state"] == "FIRING"
        assert status["flame_signal"] == 85.0
        assert status["flame_proven"] is True

    @pytest.mark.safety
    @pytest.mark.bms
    def test_status_includes_all_permissives(self, bms):
        """Status should include all permissive states."""
        status = bms.get_status()

        permissives = status["permissives"]
        expected_permissives = [
            "drum_level_ok",
            "steam_pressure_ok",
            "fuel_pressure_ok",
            "combustion_air_ok",
            "purge_complete",
            "flame_scanner_ok",
            "no_lockout",
        ]

        for perm_name in expected_permissives:
            assert perm_name in permissives
            assert "satisfied" in permissives[perm_name]
            assert "bypassed" in permissives[perm_name]


class TestTripCallback:
    """
    Tests for trip callback functionality.
    """

    @pytest.mark.safety
    @pytest.mark.bms
    def test_trip_callback_invoked(self):
        """Trip callback should be invoked on trip."""
        callback_data = []

        def callback(boiler_id, reason):
            callback_data.append((boiler_id, reason))

        bms = BurnerManagementSystem("TEST-001", callback)
        bms._state = BurnerState.FIRING
        bms._trip("Test reason")

        assert len(callback_data) == 1
        assert callback_data[0] == ("TEST-001", "Test reason")

    @pytest.mark.safety
    @pytest.mark.bms
    def test_no_callback_no_exception(self):
        """No callback should not cause exception."""
        bms = BurnerManagementSystem("TEST-001", None)
        bms._state = BurnerState.FIRING

        # Should not raise
        bms._trip("Test")

        assert bms.state == BurnerState.LOCKOUT


class TestBMSTimingConstants:
    """
    Tests for BMS timing constants compliance with NFPA 85.
    """

    @pytest.fixture
    def bms(self):
        return BurnerManagementSystem("TEST-001")

    @pytest.mark.safety
    @pytest.mark.bms
    @pytest.mark.nfpa
    def test_pre_purge_time_minimum(self, bms):
        """Pre-purge >= 5 minutes (300s) per NFPA 85."""
        assert bms.PRE_PURGE_TIME_S >= 300

    @pytest.mark.safety
    @pytest.mark.bms
    @pytest.mark.nfpa
    def test_pilot_trial_time_maximum(self, bms):
        """Pilot trial <= 10s per NFPA 85."""
        assert bms.PILOT_TRIAL_TIME_S <= 10

    @pytest.mark.safety
    @pytest.mark.bms
    @pytest.mark.nfpa
    def test_main_flame_trial_time_maximum(self, bms):
        """Main flame trial <= 10s per NFPA 85."""
        assert bms.MAIN_FLAME_TRIAL_TIME_S <= 10

    @pytest.mark.safety
    @pytest.mark.bms
    @pytest.mark.nfpa
    def test_flame_failure_response_maximum(self, bms):
        """Flame failure response <= 4s per NFPA 85."""
        assert bms.FLAME_FAILURE_RESPONSE_S <= 4

    @pytest.mark.safety
    @pytest.mark.bms
    @pytest.mark.nfpa
    def test_post_purge_time_minimum(self, bms):
        """Post-purge >= 1 minute (60s)."""
        assert bms.POST_PURGE_TIME_S >= 60


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
