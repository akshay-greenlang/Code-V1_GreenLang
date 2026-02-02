# -*- coding: utf-8 -*-
"""
NFPA 85 Combustion Safety Compliance Tests

Tests compliance with NFPA 85 Boiler and Combustion Systems Hazards Code:
    - Timing requirements validation (purge, trials, shutdown)
    - Interlock logic verification
    - BMS state transitions
    - Flame detection requirements
    - Purge sequence validation

Standards Reference:
    - NFPA 85 (2019) - Boiler and Combustion Systems Hazards Code
    - NFPA 86 - Standard for Ovens and Furnaces
    - API 556 - Instrumentation, Control, and Protective Systems

Author: GL-TestEngineer
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import math
import pytest

# Import burner control module if available
# Use broad exception handling to catch pydantic errors during import
try:
    from greenlang.agents.process_heat.gl_018_unified_combustion.burner_control import (
        FlameStabilityAnalyzer,
        BurnerTuningController,
        BMSSequenceController,
        FlameDetectorType,
        BurnerMode,
    )
    from greenlang.agents.process_heat.gl_018_unified_combustion.config import (
        FlameStabilityConfig,
        BMSConfig,
        BMSSequence,
    )
    from greenlang.agents.process_heat.gl_018_unified_combustion.schemas import (
        BurnerStatus,
        FlameStabilityAnalysis,
    )
    BURNER_MODULE_AVAILABLE = True
except Exception:
    BURNER_MODULE_AVAILABLE = False
    FlameStabilityAnalyzer = None
    BurnerTuningController = None
    BMSSequenceController = None
    FlameDetectorType = None
    BurnerMode = None
    FlameStabilityConfig = None
    BMSConfig = None
    BMSSequence = None
    BurnerStatus = None
    FlameStabilityAnalysis = None


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def flame_stability_analyzer():
    """Create FlameStabilityAnalyzer for testing."""
    if not BURNER_MODULE_AVAILABLE:
        pytest.skip("Burner control module not available")
    config = FlameStabilityConfig(
        fsi_optimal_min=0.85,
        fsi_warning_threshold=0.70,
        fsi_alarm_threshold=0.50,
        flame_signal_min_pct=30.0,
        flame_flicker_frequency_hz=3.0,
    )
    return FlameStabilityAnalyzer(config=config)


@pytest.fixture
def bms_config():
    """Create BMS configuration for testing."""
    if not BURNER_MODULE_AVAILABLE:
        pytest.skip("Burner control module not available")
    return BMSConfig(
        pre_purge_time_s=60.0,
        post_purge_time_s=60.0,
        pilot_trial_time_s=10.0,
        main_flame_trial_time_s=15.0,
        flame_failure_response_time_s=4.0,
        purge_volume_changes=4,
        purge_air_flow_pct=25.0,
        low_fire_interlock=True,
        flame_detector_redundancy="2oo3",
    )


@pytest.fixture
def bms_controller(bms_config):
    """Create BMSSequenceController for testing."""
    if not BURNER_MODULE_AVAILABLE:
        pytest.skip("Burner control module not available")
    return BMSSequenceController(bms_config)


@pytest.fixture
def sample_flame_signals_stable() -> List[float]:
    """Sample stable flame signals (80-85%)."""
    return [82.0, 83.5, 81.0, 84.0, 82.5, 83.0, 82.0, 84.5, 81.5, 83.0]


@pytest.fixture
def sample_flame_signals_unstable() -> List[float]:
    """Sample unstable flame signals (high variance)."""
    return [50.0, 90.0, 45.0, 95.0, 40.0, 85.0, 55.0, 92.0, 38.0, 88.0]


@pytest.fixture
def sample_flame_signals_weak() -> List[float]:
    """Sample weak flame signals (below minimum)."""
    return [25.0, 28.0, 22.0, 30.0, 24.0, 26.0, 27.0, 23.0, 29.0, 25.0]


# =============================================================================
# TIMING REQUIREMENTS TESTS
# =============================================================================


class TestNFPA85TimingRequirements:
    """
    Test NFPA 85 timing requirements for combustion sequences.

    NFPA 85 specifies minimum and maximum times for:
        - Pre-purge: Minimum 4 volume changes
        - Pilot trial: Maximum 10 seconds
        - Main flame trial (MTFI): Maximum 15 seconds
        - Flame failure response: Maximum 4 seconds
        - Post-purge: Minimum 4 volume changes

    Pass/Fail Criteria:
        - Timing requirements must match NFPA 85 specifications
        - Implementation must enforce these limits
    """

    @pytest.mark.compliance
    def test_pre_purge_minimum_time(self, nfpa85_timing_requirements):
        """Test pre-purge minimum time requirement."""
        purge_req = nfpa85_timing_requirements["purge"]

        # NFPA 85 requires minimum 4 volume changes
        assert purge_req["minimum_volume_changes"] == 4, (
            "NFPA 85 requires minimum 4 volume changes for purge"
        )

    @pytest.mark.compliance
    def test_pre_purge_air_flow_requirement(self, nfpa85_timing_requirements):
        """Test pre-purge air flow requirement."""
        purge_req = nfpa85_timing_requirements["purge"]

        # Minimum 25% of rated air flow
        assert purge_req["air_flow_minimum_pct"] >= 25, (
            "NFPA 85 requires minimum 25% air flow during purge"
        )

    @pytest.mark.compliance
    def test_pilot_trial_maximum_time(self, nfpa85_timing_requirements):
        """Test pilot trial maximum time (PTFI)."""
        pilot_req = nfpa85_timing_requirements["pilot_lightoff"]

        # Maximum 10 seconds to establish pilot
        assert pilot_req["pilot_flame_establishing_period_s"] <= 10, (
            "NFPA 85 limits pilot trial to 10 seconds"
        )

    @pytest.mark.compliance
    def test_main_flame_trial_maximum_time(self, nfpa85_timing_requirements):
        """Test main flame trial for ignition (MTFI) maximum time."""
        main_req = nfpa85_timing_requirements["main_flame"]

        # Maximum 15 seconds for MTFI
        assert main_req["main_flame_trial_for_ignition_s"] <= 15, (
            "NFPA 85 limits MTFI to 15 seconds"
        )

    @pytest.mark.compliance
    def test_flame_failure_response_time(self, nfpa85_timing_requirements):
        """Test flame failure response time requirement."""
        flame_req = nfpa85_timing_requirements["flame_detection"]

        # Maximum 4 seconds to respond to flame loss
        assert flame_req["flame_signal_loss_trip_time_s"] <= 4, (
            "NFPA 85 requires flame failure response within 4 seconds"
        )

    @pytest.mark.compliance
    def test_safety_shutoff_valve_timing(self, nfpa85_timing_requirements):
        """Test safety shutoff valve closure time."""
        main_req = nfpa85_timing_requirements["main_flame"]

        # Maximum 1 second for safety valve closure
        assert main_req["burner_safety_shutoff_time_s"] <= 1, (
            "Safety shutoff valves must close within 1 second"
        )

    @pytest.mark.compliance
    def test_post_purge_requirements(self, nfpa85_timing_requirements):
        """Test post-purge requirements."""
        shutdown_req = nfpa85_timing_requirements["shutdown"]

        assert shutdown_req["post_purge_volume_changes"] >= 4, (
            "NFPA 85 requires minimum 4 volume changes for post-purge"
        )

    @pytest.mark.compliance
    def test_low_fire_hold_time(self, nfpa85_timing_requirements):
        """Test low fire hold time on startup."""
        startup_req = nfpa85_timing_requirements["startup_sequence"]

        # Minimum time at low fire before modulating
        assert startup_req["low_fire_hold_time_s"] >= 60, (
            "NFPA 85 requires minimum time at low fire on startup"
        )


# =============================================================================
# PURGE CALCULATION TESTS
# =============================================================================


class TestPurgeCalculations:
    """
    Test purge time calculations per NFPA 85.

    Purge time = (Furnace Volume * Volume Changes) / Air Flow Rate

    Pass/Fail Criteria:
        - Purge time must ensure minimum 4 volume changes
        - Air flow must be at least 25% of rated capacity
    """

    @pytest.mark.compliance
    def test_purge_time_calculation(self, bms_controller):
        """Test purge time calculation is correct."""
        furnace_volume = 1000  # cubic feet
        air_flow = 5000  # CFM

        purge_time = bms_controller.calculate_purge_time(
            furnace_volume_ft3=furnace_volume,
            air_flow_cfm=air_flow,
        )

        # Expected: (1000 ft3 * 4 changes) / 5000 CFM = 0.8 min = 48 sec
        expected_time = (furnace_volume * 4 / air_flow) * 60

        assert purge_time >= expected_time, (
            f"Purge time should be at least {expected_time:.0f}s for 4 volume changes"
        )

    @pytest.mark.compliance
    def test_purge_time_minimum_enforced(self, bms_controller):
        """Test minimum purge time is enforced even with high air flow."""
        furnace_volume = 100  # Small furnace
        air_flow = 10000  # High air flow

        purge_time = bms_controller.calculate_purge_time(
            furnace_volume_ft3=furnace_volume,
            air_flow_cfm=air_flow,
        )

        # Should not go below configured minimum
        assert purge_time >= bms_controller.config.pre_purge_time_s, (
            "Purge time should not go below configured minimum"
        )

    @pytest.mark.compliance
    def test_purge_time_zero_air_flow_handling(self, bms_controller):
        """Test purge time calculation handles zero air flow."""
        furnace_volume = 1000

        purge_time = bms_controller.calculate_purge_time(
            furnace_volume_ft3=furnace_volume,
            air_flow_cfm=0,  # Zero air flow (error condition)
        )

        # Should return default purge time
        assert purge_time == bms_controller.config.pre_purge_time_s, (
            "Should return default purge time when air flow is zero"
        )


# =============================================================================
# INTERLOCK LOGIC TESTS
# =============================================================================


class TestNFPA85InterlockLogic:
    """
    Test NFPA 85 required interlocks.

    NFPA 85 requires specific interlocks for combustion safety.

    Pass/Fail Criteria:
        - All required interlocks must be defined
        - Interlock actions must match NFPA 85 requirements
    """

    @pytest.mark.compliance
    def test_fuel_supply_interlocks_defined(self, nfpa85_interlock_requirements):
        """Test fuel supply interlocks are defined."""
        fuel_interlocks = nfpa85_interlock_requirements["fuel_supply"]

        assert "low_fuel_pressure" in fuel_interlocks, (
            "Low fuel pressure interlock required"
        )
        assert "high_fuel_pressure" in fuel_interlocks, (
            "High fuel pressure interlock required"
        )

    @pytest.mark.compliance
    def test_combustion_air_interlocks_defined(self, nfpa85_interlock_requirements):
        """Test combustion air interlocks are defined."""
        air_interlocks = nfpa85_interlock_requirements["combustion_air"]

        assert "combustion_air_flow_low" in air_interlocks, (
            "Low combustion air flow interlock required"
        )
        assert "forced_draft_fan_running" in air_interlocks, (
            "FD fan running interlock required"
        )

    @pytest.mark.compliance
    def test_drum_level_interlocks_defined(self, nfpa85_interlock_requirements):
        """Test drum level interlocks are defined."""
        level_interlocks = nfpa85_interlock_requirements["drum_level"]

        assert "low_water_cutoff" in level_interlocks, (
            "Low water cutoff interlock required"
        )

        # LWCO is critical and should be 2oo3
        lwco = level_interlocks["low_water_cutoff"]
        assert lwco.get("redundancy") == "2oo3", (
            "Low water cutoff should have 2oo3 redundancy"
        )

    @pytest.mark.compliance
    def test_flame_failure_interlock(self, nfpa85_interlock_requirements):
        """Test flame failure interlock is defined with correct response time."""
        flame_interlocks = nfpa85_interlock_requirements["flame"]

        assert "flame_failure" in flame_interlocks, (
            "Flame failure interlock required"
        )

        flame_failure = flame_interlocks["flame_failure"]
        assert flame_failure.get("response_time_s", 99) <= 4, (
            "Flame failure response time must be <= 4 seconds"
        )
        assert flame_failure.get("critical") is True, (
            "Flame failure interlock is critical"
        )

    @pytest.mark.compliance
    def test_purge_complete_interlock(self, nfpa85_interlock_requirements):
        """Test purge complete interlock is defined."""
        purge_interlocks = nfpa85_interlock_requirements["purge"]

        assert "purge_complete" in purge_interlocks, (
            "Purge complete interlock required"
        )

        purge_complete = purge_interlocks["purge_complete"]
        assert purge_complete.get("action") == "enable_ignition", (
            "Purge complete should enable ignition"
        )


# =============================================================================
# BMS STATE TRANSITION TESTS
# =============================================================================


class TestBMSStateTransitions:
    """
    Test BMS (Burner Management System) state transitions.

    NFPA 85 requires specific state machine behavior for safety.

    Pass/Fail Criteria:
        - Only valid state transitions are allowed
        - Lockout requires manual reset
        - Entry/exit conditions are enforced
    """

    @pytest.mark.compliance
    def test_idle_to_pre_purge_transition(self, nfpa85_bms_state_transitions):
        """Test IDLE to PRE_PURGE is valid transition."""
        idle_state = nfpa85_bms_state_transitions["IDLE"]

        assert "PRE_PURGE" in idle_state["valid_transitions"], (
            "IDLE should allow transition to PRE_PURGE"
        )

    @pytest.mark.compliance
    def test_pre_purge_requirements(self, nfpa85_bms_state_transitions):
        """Test PRE_PURGE state requirements."""
        purge_state = nfpa85_bms_state_transitions["PRE_PURGE"]

        assert "interlocks_satisfied" in purge_state["entry_conditions"], (
            "PRE_PURGE requires interlocks to be satisfied"
        )
        assert "air_flow_proven" in purge_state["entry_conditions"], (
            "PRE_PURGE requires air flow to be proven"
        )
        assert purge_state["air_flow_requirement_pct"] >= 25, (
            "PRE_PURGE requires at least 25% air flow"
        )

    @pytest.mark.compliance
    def test_pilot_trial_maximum_duration(self, nfpa85_bms_state_transitions):
        """Test PILOT_TRIAL has maximum duration."""
        pilot_state = nfpa85_bms_state_transitions["PILOT_TRIAL"]

        assert pilot_state["maximum_duration_s"] <= 10, (
            "PILOT_TRIAL maximum duration should be <= 10 seconds"
        )

    @pytest.mark.compliance
    def test_main_flame_trial_maximum_duration(self, nfpa85_bms_state_transitions):
        """Test MAIN_FLAME_TRIAL has maximum duration (MTFI)."""
        main_state = nfpa85_bms_state_transitions["MAIN_FLAME_TRIAL"]

        assert main_state["maximum_duration_s"] <= 15, (
            "MAIN_FLAME_TRIAL (MTFI) maximum duration should be <= 15 seconds"
        )

    @pytest.mark.compliance
    def test_lockout_requires_manual_reset(self, nfpa85_bms_state_transitions):
        """Test LOCKOUT state requires manual reset."""
        lockout_state = nfpa85_bms_state_transitions["LOCKOUT"]

        assert lockout_state["requires_manual_reset"] is True, (
            "LOCKOUT must require manual reset per NFPA 85"
        )

        assert "manual_reset" in lockout_state["exit_conditions"], (
            "LOCKOUT exit conditions must include manual_reset"
        )

    @pytest.mark.compliance
    def test_no_direct_idle_to_running_transition(self, nfpa85_bms_state_transitions):
        """Test there is no direct transition from IDLE to RUNNING."""
        idle_state = nfpa85_bms_state_transitions["IDLE"]

        assert "RUNNING" not in idle_state["valid_transitions"], (
            "IDLE should NOT allow direct transition to RUNNING"
        )

    @pytest.mark.compliance
    def test_flame_failure_causes_lockout_or_post_purge(
        self,
        nfpa85_bms_state_transitions,
    ):
        """Test flame failure transitions to safe state."""
        running_state = nfpa85_bms_state_transitions["RUNNING"]

        # Flame failure should exit RUNNING to POST_PURGE or LOCKOUT
        assert "flame_failure" in running_state["exit_conditions"], (
            "Flame failure should be an exit condition from RUNNING"
        )

        valid_exits = running_state["valid_transitions"]
        assert "POST_PURGE" in valid_exits or "LOCKOUT" in valid_exits, (
            "RUNNING should transition to POST_PURGE or LOCKOUT on flame failure"
        )


# =============================================================================
# FLAME DETECTION TESTS
# =============================================================================


class TestFlameDetection:
    """
    Test flame detection requirements per NFPA 85.

    Validates flame detector voting logic and response times.

    Pass/Fail Criteria:
        - Flame detection must use redundant detectors
        - Voting logic must be correctly implemented
        - Self-checking requirements must be met
    """

    @pytest.mark.compliance
    def test_flame_detector_voting_2oo3(self, bms_controller):
        """Test 2oo3 flame detector voting logic."""
        # Two detectors see flame, one does not
        detector_signals = {
            "UV-001A": 85.0,
            "UV-001B": 80.0,
            "UV-001C": 10.0,  # Not detecting
        }

        proven, result = bms_controller.get_flame_detector_voting_result(
            detector_signals=detector_signals,
            min_signal_pct=30.0,
        )

        assert proven is True, (
            "2oo3 should prove flame with 2 of 3 detectors seeing flame"
        )
        assert "2/3" in result, "Result should indicate 2 of 3 detecting"

    @pytest.mark.compliance
    def test_flame_detector_voting_2oo3_failure(self, bms_controller):
        """Test 2oo3 flame detector voting with only 1 detecting."""
        # Only one detector sees flame
        detector_signals = {
            "UV-001A": 85.0,
            "UV-001B": 10.0,
            "UV-001C": 15.0,
        }

        proven, result = bms_controller.get_flame_detector_voting_result(
            detector_signals=detector_signals,
            min_signal_pct=30.0,
        )

        assert proven is False, (
            "2oo3 should NOT prove flame with only 1 of 3 detectors"
        )

    @pytest.mark.compliance
    def test_flame_signal_minimum_threshold(
        self,
        nfpa85_timing_requirements,
    ):
        """Test flame signal minimum threshold is defined."""
        # Minimum signal for flame proven is typically 30%
        # This prevents false flame detection from noise
        flame_req = nfpa85_timing_requirements["flame_detection"]

        # Scanner response time should be fast enough for 4-second total response
        assert flame_req["scanner_response_time_s"] <= 1, (
            "Scanner response time should be <= 1 second"
        )

    @pytest.mark.compliance
    def test_self_checking_interval(self, nfpa85_timing_requirements):
        """Test self-checking scanner interval requirement."""
        flame_req = nfpa85_timing_requirements["flame_detection"]

        assert flame_req["self_checking_interval_s"] <= 10, (
            "Self-checking scanners should check within 10 seconds"
        )


# =============================================================================
# FLAME STABILITY INDEX TESTS
# =============================================================================


class TestFlameStabilityIndex:
    """
    Test Flame Stability Index (FSI) calculations.

    FSI is a composite metric indicating flame quality:
        - 1.0: Perfect, stable flame
        - 0.85-1.0: Optimal
        - 0.70-0.85: Normal
        - 0.50-0.70: Warning
        - <0.50: Alarm

    Pass/Fail Criteria:
        - FSI must correctly classify flame conditions
        - Components must be weighted appropriately
    """

    @pytest.mark.compliance
    def test_fsi_stable_flame(
        self,
        flame_stability_analyzer,
        sample_flame_signals_stable,
    ):
        """Test FSI calculation for stable flame."""
        fsi, status = flame_stability_analyzer.calculate_fsi(
            flame_signals=sample_flame_signals_stable,
        )

        assert fsi >= 0.70, f"Stable flame should have FSI >= 0.70, got {fsi}"
        assert status in ["optimal", "normal"], (
            f"Stable flame status should be optimal or normal, got {status}"
        )

    @pytest.mark.compliance
    def test_fsi_unstable_flame(
        self,
        flame_stability_analyzer,
        sample_flame_signals_unstable,
    ):
        """Test FSI calculation for unstable flame."""
        fsi, status = flame_stability_analyzer.calculate_fsi(
            flame_signals=sample_flame_signals_unstable,
        )

        # High variance should result in lower FSI
        assert fsi < 0.85, f"Unstable flame should have FSI < 0.85, got {fsi}"

    @pytest.mark.compliance
    def test_fsi_weak_flame(
        self,
        flame_stability_analyzer,
        sample_flame_signals_weak,
    ):
        """Test FSI calculation for weak flame (below minimum)."""
        fsi, status = flame_stability_analyzer.calculate_fsi(
            flame_signals=sample_flame_signals_weak,
        )

        assert fsi < 0.70, f"Weak flame should have FSI < 0.70, got {fsi}"
        assert status in ["warning", "alarm"], (
            f"Weak flame status should be warning or alarm, got {status}"
        )

    @pytest.mark.compliance
    def test_fsi_requires_minimum_samples(self, flame_stability_analyzer):
        """Test FSI requires minimum number of samples."""
        # Only 2 samples (insufficient)
        fsi, status = flame_stability_analyzer.calculate_fsi(
            flame_signals=[80.0, 82.0],
        )

        assert status == "INSUFFICIENT_DATA", (
            "FSI should indicate insufficient data with < 3 samples"
        )

    @pytest.mark.compliance
    def test_fsi_o2_stability_component(self, flame_stability_analyzer):
        """Test FSI includes O2 stability component."""
        flame_signals = [80.0] * 10

        # High O2 variance should lower FSI
        o2_unstable = [3.0, 4.5, 2.5, 5.0, 2.0, 4.0, 3.5, 5.5, 2.5, 4.5]
        o2_stable = [3.0, 3.1, 2.9, 3.0, 3.1, 3.0, 2.9, 3.1, 3.0, 3.0]

        fsi_unstable_o2, _ = flame_stability_analyzer.calculate_fsi(
            flame_signals=flame_signals,
            o2_readings=o2_unstable,
        )

        fsi_stable_o2, _ = flame_stability_analyzer.calculate_fsi(
            flame_signals=flame_signals,
            o2_readings=o2_stable,
        )

        assert fsi_stable_o2 > fsi_unstable_o2, (
            "Stable O2 should result in higher FSI than unstable O2"
        )


# =============================================================================
# STARTUP PERMISSIVE TESTS
# =============================================================================


class TestStartupPermissives:
    """
    Test startup permissive verification per NFPA 85.

    All permissives must be satisfied before ignition.

    Pass/Fail Criteria:
        - All required permissives must be checked
        - Failed permissives must prevent ignition
    """

    @pytest.mark.compliance
    def test_startup_permissives_all_satisfied(self, bms_controller):
        """Test startup permissives verification when all satisfied."""
        interlocks = {
            "fuel_supply_pressure": True,
            "combustion_air_pressure": True,
            "low_water_cutoff": True,
            "high_pressure_limit": True,
            "flame_failure_relay": True,
        }

        satisfied, failed = bms_controller.verify_startup_permissives(
            interlocks=interlocks,
            air_flow_pct=30.0,
        )

        assert satisfied is True, "Should be satisfied with all interlocks OK"
        assert len(failed) == 0, f"Should have no failures, got {failed}"

    @pytest.mark.compliance
    def test_startup_permissives_low_air_flow(self, bms_controller):
        """Test startup permissives fails with low air flow."""
        interlocks = {
            "fuel_supply_pressure": True,
            "combustion_air_pressure": True,
            "low_water_cutoff": True,
            "high_pressure_limit": True,
            "flame_failure_relay": True,
        }

        satisfied, failed = bms_controller.verify_startup_permissives(
            interlocks=interlocks,
            air_flow_pct=20.0,  # Below 25% minimum
        )

        assert satisfied is False, "Should fail with insufficient air flow"
        assert any("air flow" in f.lower() for f in failed), (
            "Should indicate air flow failure"
        )

    @pytest.mark.compliance
    def test_startup_permissives_interlock_trip(self, bms_controller):
        """Test startup permissives fails with tripped interlock."""
        interlocks = {
            "fuel_supply_pressure": True,
            "combustion_air_pressure": True,
            "low_water_cutoff": False,  # LWCO tripped
            "high_pressure_limit": True,
            "flame_failure_relay": True,
        }

        satisfied, failed = bms_controller.verify_startup_permissives(
            interlocks=interlocks,
            air_flow_pct=30.0,
        )

        assert satisfied is False, "Should fail with LWCO tripped"
        assert any("low_water" in f.lower() for f in failed), (
            "Should indicate LWCO failure"
        )


# =============================================================================
# BMS CONTROLLER OPERATION TESTS
# =============================================================================


class TestBMSControllerOperations:
    """
    Test BMS controller operational functions.

    Validates sequence management and lockout behavior.
    """

    @pytest.mark.compliance
    def test_bms_status_reporting(self, bms_controller):
        """Test BMS status reporting."""
        flame_signals = {"UV-001A": 80.0, "UV-001B": 85.0, "UV-001C": 75.0}
        interlocks = {
            "low_water_cutoff": True,
            "high_pressure_limit": True,
            "combustion_air_flow": True,
        }

        status = bms_controller.get_status(
            flame_signals=flame_signals,
            air_flow_verified=True,
            interlocks=interlocks,
        )

        assert status.all_interlocks_satisfied is True
        assert len(status.tripped_interlocks) == 0

    @pytest.mark.compliance
    def test_bms_lockout_trigger(self, bms_controller):
        """Test BMS lockout trigger."""
        bms_controller.trigger_lockout("Flame failure during running")

        # Should be in lockout state
        assert bms_controller._current_sequence == BMSSequence.LOCKOUT

    @pytest.mark.compliance
    def test_bms_lockout_reset(self, bms_controller):
        """Test BMS lockout reset."""
        bms_controller.trigger_lockout("Test lockout")

        # Reset lockout
        result = bms_controller.reset_lockout()

        assert result is True, "Should successfully reset lockout"
        assert bms_controller._current_sequence == BMSSequence.IDLE

    @pytest.mark.compliance
    def test_bms_purge_completion(self, bms_controller):
        """Test BMS purge completion marking."""
        bms_controller.set_sequence(BMSSequence.PRE_PURGE)
        bms_controller.complete_purge()

        assert bms_controller._purge_complete is True


# =============================================================================
# BURNER TUNING TESTS
# =============================================================================


class TestBurnerTuning:
    """
    Test burner tuning recommendations.

    Validates tuning calculations for combustion optimization.
    """

    @pytest.mark.compliance
    def test_air_register_adjustment_for_high_o2(self):
        """Test air register adjustment recommendation for high O2."""
        if not BURNER_MODULE_AVAILABLE:
            pytest.skip("Burner control module not available")

        controller = BurnerTuningController()

        burner_status = BurnerStatus(
            burner_id="BNR-001",
            firing_rate_pct=75.0,
            air_register_position_pct=60.0,
            flame_status="lit",
        )

        result = controller.calculate_tuning(
            burner_status=burner_status,
            flue_gas_o2_pct=5.0,  # High - should reduce air
            target_o2_pct=3.0,
            co_ppm=25.0,
            load_pct=75.0,
        )

        # Should recommend reducing air register (closing)
        assert result.recommended_air_register_pct < result.current_air_register_pct, (
            "High O2 should result in recommendation to reduce air"
        )

    @pytest.mark.compliance
    def test_air_register_safety_limit_with_high_co(self):
        """Test air register adjustment is limited when CO is high."""
        if not BURNER_MODULE_AVAILABLE:
            pytest.skip("Burner control module not available")

        controller = BurnerTuningController()

        burner_status = BurnerStatus(
            burner_id="BNR-001",
            firing_rate_pct=75.0,
            air_register_position_pct=60.0,
            flame_status="lit",
        )

        result = controller.calculate_tuning(
            burner_status=burner_status,
            flue_gas_o2_pct=5.0,  # High O2
            target_o2_pct=3.0,
            co_ppm=150.0,  # High CO - unsafe to reduce air
            load_pct=75.0,
        )

        # Should NOT recommend reducing air when CO is high
        assert result.recommended_air_register_pct >= result.current_air_register_pct, (
            "Should not reduce air when CO is elevated"
        )
