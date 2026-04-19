# -*- coding: utf-8 -*-
"""
IEC 61511 Functional Safety Compliance Tests

Tests compliance with IEC 61511 Safety Instrumented Systems requirements:
    - SIL (Safety Integrity Level) verification calculations
    - Proof test interval validation
    - PFD (Probability of Failure on Demand) calculations
    - Voting architecture validation (1oo1, 1oo2, 2oo2, 2oo3)
    - Response time requirements

Standards Reference:
    - IEC 61511-1:2016 Functional Safety - Safety Instrumented Systems
    - IEC 61508-6:2010 Functional Safety - Guidelines for PFD Calculation
    - IEC 61511-2:2016 Guidelines for Application
    - OSHA 29 CFR 1910.119 Process Safety Management

Author: GL-TestEngineer
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import math
import pytest

# Import SIS integration module if available
try:
    from greenlang.agents.process_heat.gl_001_thermal_command.sis_integration import (
        SISManager,
        SISInterlock,
        VotingEngine,
        VotingType,
        VotingResult,
        SensorConfig,
        SensorReading,
        SensorType,
        SensorStatus,
        SafeStateAction,
        InterlockStatus,
        create_high_temperature_interlock,
        create_high_pressure_interlock,
        create_low_level_interlock,
    )
    SIS_MODULE_AVAILABLE = True
except ImportError:
    SIS_MODULE_AVAILABLE = False


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sis_manager():
    """Create SISManager instance for testing."""
    if not SIS_MODULE_AVAILABLE:
        pytest.skip("SIS module not available")
    return SISManager(sil_level=2, fail_safe_on_fault=True, max_bypass_hours=8.0)


@pytest.fixture
def voting_engine():
    """Create VotingEngine instance for testing."""
    if not SIS_MODULE_AVAILABLE:
        pytest.skip("SIS module not available")
    return VotingEngine(fail_safe_on_fault=True)


@pytest.fixture
def sample_temperature_interlock():
    """Create sample high temperature interlock for testing."""
    if not SIS_MODULE_AVAILABLE:
        pytest.skip("SIS module not available")
    return create_high_temperature_interlock(
        name="HIGH_TEMP_SHUTDOWN",
        tag_prefix="TT-101",
        setpoint_f=500.0,
        response_time_ms=250,
    )


@pytest.fixture
def sample_pressure_interlock():
    """Create sample high pressure interlock for testing."""
    if not SIS_MODULE_AVAILABLE:
        pytest.skip("SIS module not available")
    return create_high_pressure_interlock(
        name="HIGH_PRESSURE_SHUTDOWN",
        tag_prefix="PT-201",
        setpoint_psig=200.0,
        response_time_ms=200,
    )


@pytest.fixture
def sample_sensor_readings_below_setpoint():
    """Create sensor readings below trip setpoint."""
    if not SIS_MODULE_AVAILABLE:
        pytest.skip("SIS module not available")
    return [
        SensorReading(sensor_id="TT-101A", channel="A", value=450.0, status=SensorStatus.NORMAL),
        SensorReading(sensor_id="TT-101B", channel="B", value=455.0, status=SensorStatus.NORMAL),
        SensorReading(sensor_id="TT-101C", channel="C", value=448.0, status=SensorStatus.NORMAL),
    ]


@pytest.fixture
def sample_sensor_readings_above_setpoint():
    """Create sensor readings above trip setpoint."""
    if not SIS_MODULE_AVAILABLE:
        pytest.skip("SIS module not available")
    return [
        SensorReading(sensor_id="TT-101A", channel="A", value=510.0, status=SensorStatus.NORMAL),
        SensorReading(sensor_id="TT-101B", channel="B", value=505.0, status=SensorStatus.NORMAL),
        SensorReading(sensor_id="TT-101C", channel="C", value=520.0, status=SensorStatus.NORMAL),
    ]


# =============================================================================
# SIL VERIFICATION TESTS
# =============================================================================


class TestSILVerification:
    """
    Test SIL (Safety Integrity Level) verification calculations.

    SIL levels are defined by PFD (Probability of Failure on Demand):
        - SIL 1: 10^-2 to 10^-1 PFD (RRF 10-100)
        - SIL 2: 10^-3 to 10^-2 PFD (RRF 100-1000)
        - SIL 3: 10^-4 to 10^-3 PFD (RRF 1000-10000)
        - SIL 4: 10^-5 to 10^-4 PFD (RRF 10000-100000)

    Pass/Fail Criteria:
        - PFD calculations must be within IEC 61508-6 tolerance
        - SIL assignment must match calculated PFD
    """

    @pytest.mark.compliance
    @pytest.mark.parametrize("sil,pfd_min,pfd_max", [
        (1, 1e-2, 1e-1),
        (2, 1e-3, 1e-2),
        (3, 1e-4, 1e-3),
        (4, 1e-5, 1e-4),
    ])
    def test_sil_pfd_ranges(
        self,
        iec61511_sil_targets,
        sil: int,
        pfd_min: float,
        pfd_max: float,
    ):
        """Test SIL PFD ranges match IEC 61511 requirements."""
        sil_target = iec61511_sil_targets[sil]

        assert sil_target["pfd_avg_min"] == pfd_min, (
            f"SIL {sil} PFD min should be {pfd_min}"
        )
        assert sil_target["pfd_avg_max"] == pfd_max, (
            f"SIL {sil} PFD max should be {pfd_max}"
        )

    @pytest.mark.compliance
    @pytest.mark.parametrize("sil,rrf_min,rrf_max", [
        (1, 10, 100),
        (2, 100, 1000),
        (3, 1000, 10000),
        (4, 10000, 100000),
    ])
    def test_sil_rrf_ranges(
        self,
        iec61511_sil_targets,
        sil: int,
        rrf_min: int,
        rrf_max: int,
    ):
        """Test SIL Risk Reduction Factor ranges."""
        sil_target = iec61511_sil_targets[sil]

        assert sil_target["rrf_min"] == rrf_min, f"SIL {sil} RRF min should be {rrf_min}"
        assert sil_target["rrf_max"] == rrf_max, f"SIL {sil} RRF max should be {rrf_max}"

    @pytest.mark.compliance
    def test_pfd_to_rrf_relationship(self, iec61511_sil_targets):
        """Test PFD and RRF are correctly related (RRF = 1/PFD)."""
        for sil in range(1, 5):
            target = iec61511_sil_targets[sil]

            # RRF_min should equal 1/PFD_max
            expected_rrf_min = int(1 / target["pfd_avg_max"])
            assert target["rrf_min"] == expected_rrf_min, (
                f"SIL {sil} RRF min should be 1/PFD_max"
            )

            # RRF_max should equal 1/PFD_min
            expected_rrf_max = int(1 / target["pfd_avg_min"])
            assert target["rrf_max"] == expected_rrf_max, (
                f"SIL {sil} RRF max should be 1/PFD_min"
            )


# =============================================================================
# PFD CALCULATION TESTS
# =============================================================================


class TestPFDCalculations:
    """
    Test PFD (Probability of Failure on Demand) calculations per IEC 61508-6.

    Validates PFD formulas for different voting architectures.

    Pass/Fail Criteria:
        - PFD calculations must match IEC 61508-6 formulas
        - Results must be within 5% of analytical values
    """

    @pytest.mark.compliance
    def test_pfd_1oo1_calculation(self, pfd_calculator):
        """
        Test PFD calculation for 1oo1 architecture.

        Formula: PFD = lambda_DU * TI / 2
        """
        lambda_du = 1e-6  # 1E-6 per hour
        proof_test_hours = 8760  # 1 year

        pfd = pfd_calculator.calculate_1oo1(
            lambda_du=lambda_du,
            proof_test_interval_hours=proof_test_hours,
        )

        # Expected: 1E-6 * 8760 / 2 = 0.00438
        expected_pfd = lambda_du * proof_test_hours / 2

        assert abs(pfd - expected_pfd) / expected_pfd < 0.01, (
            f"1oo1 PFD calculation: {pfd} vs expected {expected_pfd}"
        )

    @pytest.mark.compliance
    def test_pfd_1oo2_calculation(self, pfd_calculator):
        """
        Test PFD calculation for 1oo2 architecture.

        Formula: PFD = ((1-beta) * lambda_DU * TI)^2 / 3 + beta * lambda_DU * TI / 2
        """
        lambda_du = 1e-6
        proof_test_hours = 8760
        beta = 0.1  # 10% common cause factor

        pfd = pfd_calculator.calculate_1oo2(
            lambda_du=lambda_du,
            proof_test_interval_hours=proof_test_hours,
            beta_factor=beta,
        )

        # PFD for 1oo2 should be much lower than 1oo1
        pfd_1oo1 = pfd_calculator.calculate_1oo1(lambda_du, proof_test_hours)

        assert pfd < pfd_1oo1, "1oo2 PFD should be lower than 1oo1"

        # Expected analytical value (simplified)
        independent_term = ((1 - beta) * lambda_du * proof_test_hours) ** 2 / 3
        common_cause_term = beta * lambda_du * proof_test_hours / 2
        expected_pfd = independent_term + common_cause_term

        assert abs(pfd - expected_pfd) / expected_pfd < 0.05, (
            f"1oo2 PFD calculation: {pfd} vs expected {expected_pfd}"
        )

    @pytest.mark.compliance
    def test_pfd_2oo2_calculation(self, pfd_calculator):
        """
        Test PFD calculation for 2oo2 architecture.

        Formula: PFD = lambda_DU * TI (approximately 2x of 1oo1)
        """
        lambda_du = 1e-6
        proof_test_hours = 8760

        pfd = pfd_calculator.calculate_2oo2(
            lambda_du=lambda_du,
            proof_test_interval_hours=proof_test_hours,
        )

        # Expected: lambda_DU * TI
        expected_pfd = lambda_du * proof_test_hours

        assert abs(pfd - expected_pfd) / expected_pfd < 0.01, (
            f"2oo2 PFD calculation: {pfd} vs expected {expected_pfd}"
        )

    @pytest.mark.compliance
    def test_pfd_2oo3_calculation(self, pfd_calculator):
        """
        Test PFD calculation for 2oo3 (TMR) architecture.

        Formula: PFD = 3 * ((1-beta) * lambda_DU * TI)^2 + beta * lambda_DU * TI / 2
        """
        lambda_du = 1e-6
        proof_test_hours = 8760
        beta = 0.1

        pfd = pfd_calculator.calculate_2oo3(
            lambda_du=lambda_du,
            proof_test_interval_hours=proof_test_hours,
            beta_factor=beta,
        )

        # 2oo3 should have similar PFD to 1oo2 but slightly higher
        pfd_1oo2 = pfd_calculator.calculate_1oo2(lambda_du, proof_test_hours, beta_factor=beta)

        # Both should achieve SIL 2 or better
        assert pfd < 1e-2, "2oo3 should achieve at least SIL 2"

    @pytest.mark.compliance
    @pytest.mark.parametrize("architecture,expected_sil", [
        ("1oo1", 1),   # Limited to SIL 1 due to single point failure
        ("1oo2", 3),   # Can achieve SIL 3 with typical parameters
        ("2oo2", 1),   # Limited to SIL 1 due to series failure mode
        ("2oo3", 3),   # Can achieve SIL 3 with typical parameters
    ])
    def test_achievable_sil_by_architecture(
        self,
        pfd_calculator,
        iec61511_voting_architectures,
        architecture: str,
        expected_sil: int,
    ):
        """Test maximum achievable SIL by voting architecture."""
        arch_spec = iec61511_voting_architectures[architecture]

        assert arch_spec["max_sil"] == expected_sil, (
            f"{architecture} max SIL should be {expected_sil}"
        )

    @pytest.mark.compliance
    def test_pfd_full_calculation_meets_sil_target(self, pfd_calculator):
        """Test full PFD calculation meets target SIL."""
        # Typical SIL 2 SIS parameters
        result = pfd_calculator.calculate_full(
            architecture="2oo3",
            lambda_du=2e-6,  # Typical transmitter
            proof_test_interval_hours=8760,  # Annual testing
            target_sil=2,
            beta_factor=0.1,
        )

        assert result.meets_target, (
            f"2oo3 with typical parameters should achieve SIL 2, "
            f"got SIL {result.sil_achieved} with PFD {result.pfd_avg}"
        )
        assert result.sil_achieved >= 2, f"Should achieve at least SIL 2"


# =============================================================================
# PROOF TEST INTERVAL VALIDATION TESTS
# =============================================================================


class TestProofTestIntervalValidation:
    """
    Test proof test interval validation per IEC 61511.

    Proof testing reveals dangerous undetected faults.
    Shorter intervals = lower PFD but higher maintenance cost.

    Pass/Fail Criteria:
        - Intervals must be appropriate for target SIL
        - PFD must be recalculated when interval changes
    """

    @pytest.mark.compliance
    @pytest.mark.parametrize("interval_hours,expected_effect", [
        (2190, "lower_pfd"),    # Quarterly - lower PFD
        (4380, "moderate_pfd"), # Semi-annual - moderate PFD
        (8760, "baseline_pfd"), # Annual - baseline
        (17520, "higher_pfd"),  # 2-year - higher PFD
    ])
    def test_proof_test_interval_effect_on_pfd(
        self,
        pfd_calculator,
        interval_hours: float,
        expected_effect: str,
    ):
        """Test effect of proof test interval on PFD."""
        lambda_du = 2e-6

        pfd = pfd_calculator.calculate_1oo1(
            lambda_du=lambda_du,
            proof_test_interval_hours=interval_hours,
        )

        # PFD is proportional to proof test interval
        baseline_pfd = pfd_calculator.calculate_1oo1(lambda_du, 8760)

        if expected_effect == "lower_pfd":
            assert pfd < baseline_pfd, "Shorter interval should reduce PFD"
        elif expected_effect == "higher_pfd":
            assert pfd > baseline_pfd, "Longer interval should increase PFD"

    @pytest.mark.compliance
    def test_proof_test_interval_for_sil2(
        self,
        pfd_calculator,
        iec61511_proof_test_intervals,
    ):
        """Test typical proof test interval for SIL 2 system."""
        typical_interval = iec61511_proof_test_intervals["typical_sis"]

        assert typical_interval == 8760, (
            "Typical SIL 2 proof test interval is annual (8760 hours)"
        )

        # Verify this interval achieves SIL 2 with 2oo3 architecture
        result = pfd_calculator.calculate_full(
            architecture="2oo3",
            lambda_du=2e-6,
            proof_test_interval_hours=typical_interval,
            target_sil=2,
        )

        assert result.meets_target, (
            f"Annual proof testing with 2oo3 should achieve SIL 2"
        )

    @pytest.mark.compliance
    def test_high_demand_sis_requires_shorter_interval(
        self,
        iec61511_proof_test_intervals,
    ):
        """Test high-demand SIS requires shorter proof test interval."""
        high_demand_interval = iec61511_proof_test_intervals["high_demand_sis"]
        typical_interval = iec61511_proof_test_intervals["typical_sis"]

        assert high_demand_interval < typical_interval, (
            "High-demand SIS should have shorter proof test interval"
        )
        assert high_demand_interval == 2190, (
            "High-demand SIS typically requires quarterly testing"
        )


# =============================================================================
# VOTING ARCHITECTURE TESTS
# =============================================================================


class TestVotingArchitectures:
    """
    Test voting architecture implementations per IEC 61511.

    Validates voting logic for safety systems.

    Pass/Fail Criteria:
        - Trip decisions must match voting requirements
        - Degraded mode must be correctly implemented
        - Fault handling must be fail-safe
    """

    @pytest.mark.compliance
    def test_2oo3_voting_trip_condition(
        self,
        voting_engine,
        sample_sensor_readings_above_setpoint,
    ):
        """Test 2oo3 voting trips when 2 of 3 channels exceed setpoint."""
        # All 3 channels above setpoint
        result = voting_engine.evaluate(
            voting_type=VotingType.TWO_OO_THREE,
            readings=sample_sensor_readings_above_setpoint,
            setpoint=500.0,
            trip_direction="high",
        )

        assert result.trip_decision is True, (
            "Should trip when all 3 channels exceed setpoint"
        )
        assert result.channels_voting_trip >= 2, (
            f"At least 2 channels should vote trip, got {result.channels_voting_trip}"
        )

    @pytest.mark.compliance
    def test_2oo3_voting_no_trip_condition(
        self,
        voting_engine,
        sample_sensor_readings_below_setpoint,
    ):
        """Test 2oo3 voting does not trip when below setpoint."""
        result = voting_engine.evaluate(
            voting_type=VotingType.TWO_OO_THREE,
            readings=sample_sensor_readings_below_setpoint,
            setpoint=500.0,
            trip_direction="high",
        )

        assert result.trip_decision is False, (
            "Should not trip when all channels below setpoint"
        )
        assert result.channels_voting_trip < 2, (
            f"Less than 2 channels should vote trip"
        )

    @pytest.mark.compliance
    def test_2oo3_voting_one_channel_above(self, voting_engine):
        """Test 2oo3 voting with only 1 channel above setpoint."""
        if not SIS_MODULE_AVAILABLE:
            pytest.skip("SIS module not available")

        readings = [
            SensorReading(sensor_id="TT-101A", channel="A", value=510.0, status=SensorStatus.NORMAL),
            SensorReading(sensor_id="TT-101B", channel="B", value=490.0, status=SensorStatus.NORMAL),
            SensorReading(sensor_id="TT-101C", channel="C", value=485.0, status=SensorStatus.NORMAL),
        ]

        result = voting_engine.evaluate(
            voting_type=VotingType.TWO_OO_THREE,
            readings=readings,
            setpoint=500.0,
            trip_direction="high",
        )

        assert result.trip_decision is False, (
            "Should NOT trip with only 1 of 3 channels above setpoint"
        )
        assert result.channels_voting_trip == 1, (
            f"Only 1 channel should vote trip, got {result.channels_voting_trip}"
        )

    @pytest.mark.compliance
    def test_2oo3_degraded_to_1oo2(self, voting_engine):
        """Test 2oo3 degrades to 1oo2 when one channel faults."""
        if not SIS_MODULE_AVAILABLE:
            pytest.skip("SIS module not available")

        readings = [
            SensorReading(sensor_id="TT-101A", channel="A", value=510.0, status=SensorStatus.NORMAL),
            SensorReading(sensor_id="TT-101B", channel="B", value=490.0, status=SensorStatus.NORMAL),
            SensorReading(sensor_id="TT-101C", channel="C", value=0.0, status=SensorStatus.FAULT),
        ]

        result = voting_engine.evaluate(
            voting_type=VotingType.TWO_OO_THREE,
            readings=readings,
            setpoint=500.0,
            trip_direction="high",
        )

        # With fail-safe, faulted channel votes for trip
        # So we have: 1 above + 1 fault = 2 trip votes
        assert result.degraded_mode is True, "Should be in degraded mode"

    @pytest.mark.compliance
    def test_1oo2_voting_either_channel_trips(self, voting_engine):
        """Test 1oo2 voting trips on either channel."""
        if not SIS_MODULE_AVAILABLE:
            pytest.skip("SIS module not available")

        # Only first channel above setpoint
        readings = [
            SensorReading(sensor_id="PT-201A", channel="A", value=210.0, status=SensorStatus.NORMAL),
            SensorReading(sensor_id="PT-201B", channel="B", value=180.0, status=SensorStatus.NORMAL),
        ]

        result = voting_engine.evaluate(
            voting_type=VotingType.ONE_OO_TWO,
            readings=readings,
            setpoint=200.0,
            trip_direction="high",
        )

        assert result.trip_decision is True, (
            "1oo2 should trip when any channel exceeds setpoint"
        )

    @pytest.mark.compliance
    def test_2oo2_voting_requires_both_channels(self, voting_engine):
        """Test 2oo2 voting requires both channels to agree."""
        if not SIS_MODULE_AVAILABLE:
            pytest.skip("SIS module not available")

        # Only first channel above setpoint
        readings = [
            SensorReading(sensor_id="PT-201A", channel="A", value=210.0, status=SensorStatus.NORMAL),
            SensorReading(sensor_id="PT-201B", channel="B", value=180.0, status=SensorStatus.NORMAL),
        ]

        result = voting_engine.evaluate(
            voting_type=VotingType.TWO_OO_TWO,
            readings=readings,
            setpoint=200.0,
            trip_direction="high",
        )

        assert result.trip_decision is False, (
            "2oo2 should NOT trip when only one channel exceeds setpoint"
        )

    @pytest.mark.compliance
    def test_voting_architecture_specifications(
        self,
        iec61511_voting_architectures,
    ):
        """Test voting architecture specifications are correctly defined."""
        expected_configs = {
            "1oo1": (1, 1),
            "1oo2": (1, 2),
            "2oo2": (2, 2),
            "2oo3": (2, 3),
            "2oo4": (2, 4),
        }

        for arch, (required, total) in expected_configs.items():
            spec = iec61511_voting_architectures[arch]
            assert spec["trips_on"] == required, (
                f"{arch} should require {required} channels to trip"
            )
            assert spec["channels"] == total, (
                f"{arch} should have {total} total channels"
            )


# =============================================================================
# FAIL-SAFE BEHAVIOR TESTS
# =============================================================================


class TestFailSafeBehavior:
    """
    Test fail-safe behavior per IEC 61511.

    Safety systems must fail to the safe state.

    Pass/Fail Criteria:
        - Faulted sensors must vote for trip (fail-safe)
        - Unknown states must be handled safely
    """

    @pytest.mark.compliance
    def test_faulted_sensor_votes_trip(self, voting_engine):
        """Test that faulted sensors vote for trip (fail-safe)."""
        if not SIS_MODULE_AVAILABLE:
            pytest.skip("SIS module not available")

        # All channels faulted
        readings = [
            SensorReading(sensor_id="TT-101A", channel="A", value=0.0, status=SensorStatus.FAULT),
            SensorReading(sensor_id="TT-101B", channel="B", value=0.0, status=SensorStatus.FAULT),
            SensorReading(sensor_id="TT-101C", channel="C", value=0.0, status=SensorStatus.FAULT),
        ]

        result = voting_engine.evaluate(
            voting_type=VotingType.TWO_OO_THREE,
            readings=readings,
            setpoint=500.0,
            trip_direction="high",
        )

        assert result.trip_decision is True, (
            "All faulted channels should result in trip (fail-safe)"
        )
        assert result.channels_faulted == 3, "Should report 3 faulted channels"

    @pytest.mark.compliance
    def test_unknown_state_handled_safely(self, voting_engine):
        """Test that unknown sensor states are handled fail-safe."""
        if not SIS_MODULE_AVAILABLE:
            pytest.skip("SIS module not available")

        readings = [
            SensorReading(sensor_id="TT-101A", channel="A", value=450.0, status=SensorStatus.NORMAL),
            SensorReading(sensor_id="TT-101B", channel="B", value=450.0, status=SensorStatus.UNKNOWN),
            SensorReading(sensor_id="TT-101C", channel="C", value=450.0, status=SensorStatus.UNKNOWN),
        ]

        result = voting_engine.evaluate(
            voting_type=VotingType.TWO_OO_THREE,
            readings=readings,
            setpoint=500.0,
            trip_direction="high",
        )

        # With fail-safe enabled, unknown channels vote for trip
        assert result.trip_decision is True, (
            "Unknown states should vote for trip (fail-safe)"
        )

    @pytest.mark.compliance
    def test_bypassed_channel_excluded_from_voting(self, voting_engine):
        """Test that bypassed channels are excluded from voting."""
        if not SIS_MODULE_AVAILABLE:
            pytest.skip("SIS module not available")

        readings = [
            SensorReading(sensor_id="TT-101A", channel="A", value=510.0, status=SensorStatus.NORMAL),
            SensorReading(sensor_id="TT-101B", channel="B", value=490.0, status=SensorStatus.NORMAL),
            SensorReading(sensor_id="TT-101C", channel="C", value=520.0, status=SensorStatus.BYPASSED),
        ]

        result = voting_engine.evaluate(
            voting_type=VotingType.TWO_OO_THREE,
            readings=readings,
            setpoint=500.0,
            trip_direction="high",
        )

        assert result.channels_bypassed == 1, "Should report 1 bypassed channel"
        assert result.degraded_mode is True, "Should be in degraded mode"


# =============================================================================
# RESPONSE TIME TESTS
# =============================================================================


class TestResponseTimeRequirements:
    """
    Test response time requirements per IEC 61511.

    SIL 2 systems typically require response time < 500ms.

    Pass/Fail Criteria:
        - Response time must be within specification
        - Total safety function response time verified
    """

    @pytest.mark.compliance
    def test_sil2_response_time_limit(self, sample_temperature_interlock):
        """Test SIL 2 interlock has response time < 500ms."""
        assert sample_temperature_interlock.response_time_ms <= 500, (
            f"SIL 2 response time should be <= 500ms, "
            f"got {sample_temperature_interlock.response_time_ms}ms"
        )

    @pytest.mark.compliance
    def test_interlock_response_time_validation(self, sis_manager, sample_temperature_interlock):
        """Test that SIS manager validates response time on registration."""
        # This should succeed (response time is 250ms < 500ms)
        result = sis_manager.register_interlock(sample_temperature_interlock)
        assert result is True, "Should accept interlock with valid response time"

    @pytest.mark.compliance
    def test_interlock_rejects_slow_response_time(self, sis_manager):
        """Test that SIS manager rejects interlocks with slow response time."""
        if not SIS_MODULE_AVAILABLE:
            pytest.skip("SIS module not available")

        # Create interlock with response time > 500ms
        with pytest.raises(ValueError, match="response time"):
            slow_interlock = SISInterlock(
                name="SLOW_INTERLOCK",
                voting_logic=VotingType.TWO_OO_THREE,
                sensors=[
                    SensorConfig(
                        sensor_id="TT-999A",
                        channel="A",
                        sensor_type=SensorType.TEMPERATURE,
                        tag_name="TT-999A",
                        engineering_units="degF",
                        range_low=0.0,
                        range_high=1000.0,
                    ),
                    SensorConfig(
                        sensor_id="TT-999B",
                        channel="B",
                        sensor_type=SensorType.TEMPERATURE,
                        tag_name="TT-999B",
                        engineering_units="degF",
                        range_low=0.0,
                        range_high=1000.0,
                    ),
                    SensorConfig(
                        sensor_id="TT-999C",
                        channel="C",
                        sensor_type=SensorType.TEMPERATURE,
                        tag_name="TT-999C",
                        engineering_units="degF",
                        range_low=0.0,
                        range_high=1000.0,
                    ),
                ],
                trip_setpoint=500.0,
                safe_state=SafeStateAction.TRIP_BURNER,
                response_time_ms=600,  # Too slow for SIL 2
                sil_level=2,
            )


# =============================================================================
# TYPICAL FAILURE RATE TESTS
# =============================================================================


class TestTypicalFailureRates:
    """
    Test typical failure rates for SIS components.

    Validates failure rate data used in PFD calculations.
    """

    @pytest.mark.compliance
    def test_transmitter_failure_rates(self, iec61511_typical_failure_rates):
        """Test transmitter failure rates are in expected range."""
        pressure = iec61511_typical_failure_rates["pressure_transmitter"]
        temperature = iec61511_typical_failure_rates["temperature_transmitter"]

        # Lambda_DU should be in range 1E-7 to 1E-5 per hour for transmitters
        assert 1e-7 < pressure["lambda_du"] < 1e-5, (
            "Pressure transmitter lambda_DU out of expected range"
        )
        assert 1e-7 < temperature["lambda_du"] < 1e-5, (
            "Temperature transmitter lambda_DU out of expected range"
        )

    @pytest.mark.compliance
    def test_final_element_failure_rates(self, iec61511_typical_failure_rates):
        """Test final element failure rates are in expected range."""
        valve = iec61511_typical_failure_rates["shutdown_valve"]
        solenoid = iec61511_typical_failure_rates["solenoid_valve"]

        # Final elements typically have higher failure rates
        assert valve["lambda_du"] > iec61511_typical_failure_rates["pressure_transmitter"]["lambda_du"], (
            "Valves typically have higher failure rates than transmitters"
        )

    @pytest.mark.compliance
    def test_safe_failure_fraction(self, iec61511_typical_failure_rates):
        """Test Safe Failure Fraction (SFF) values are valid."""
        for component, rates in iec61511_typical_failure_rates.items():
            sff = rates.get("sff", 0)
            assert 0 <= sff <= 1, f"SFF for {component} should be between 0 and 1"

            # Safety PLCs should have high SFF
            if "plc" in component.lower():
                assert sff >= 0.90, f"Safety PLC SFF should be >= 90%"


# =============================================================================
# SIS MANAGER OPERATION TESTS
# =============================================================================


class TestSISManagerOperations:
    """
    Test SIS Manager operational functions.

    Validates interlock management, bypass control, and audit logging.
    """

    @pytest.mark.compliance
    def test_interlock_registration(self, sis_manager, sample_temperature_interlock):
        """Test interlock registration."""
        result = sis_manager.register_interlock(sample_temperature_interlock)

        assert result is True, "Should successfully register interlock"

        # Verify interlock is registered
        registered = sis_manager.get_interlock(sample_temperature_interlock.interlock_id)
        assert registered is not None, "Should retrieve registered interlock"
        assert registered.name == sample_temperature_interlock.name

    @pytest.mark.compliance
    def test_interlock_evaluation(
        self,
        sis_manager,
        sample_temperature_interlock,
        sample_sensor_readings_above_setpoint,
    ):
        """Test interlock evaluation."""
        sis_manager.register_interlock(sample_temperature_interlock)

        result = sis_manager.evaluate_interlock(
            interlock_id=sample_temperature_interlock.interlock_id,
            readings=sample_sensor_readings_above_setpoint,
        )

        assert isinstance(result, VotingResult), "Should return VotingResult"
        assert result.trip_decision is True, "Should trip with readings above setpoint"

    @pytest.mark.compliance
    def test_bypass_management(self, sis_manager, sample_temperature_interlock):
        """Test interlock bypass management."""
        sis_manager.register_interlock(sample_temperature_interlock)

        # Request bypass
        result = sis_manager.request_bypass(
            interlock_id=sample_temperature_interlock.interlock_id,
            reason=BypassReason.MAINTENANCE if SIS_MODULE_AVAILABLE else "maintenance",
            authorized_by="Test Engineer",
            duration_hours=4.0,
        )

        assert result is True, "Should successfully request bypass"

        # Check bypass is active
        interlock = sis_manager.get_interlock(sample_temperature_interlock.interlock_id)
        assert interlock.bypass_active is True, "Bypass should be active"
        assert interlock.status == InterlockStatus.BYPASSED, "Status should be BYPASSED"

    @pytest.mark.compliance
    def test_max_bypass_duration_enforced(self, sis_manager, sample_temperature_interlock):
        """Test maximum bypass duration is enforced."""
        sis_manager.register_interlock(sample_temperature_interlock)

        # Request bypass longer than max (8 hours)
        result = sis_manager.request_bypass(
            interlock_id=sample_temperature_interlock.interlock_id,
            reason=BypassReason.MAINTENANCE if SIS_MODULE_AVAILABLE else "maintenance",
            authorized_by="Test Engineer",
            duration_hours=24.0,  # Request 24 hours
        )

        assert result is True, "Should accept request but limit duration"

        # Bypass expiry should be limited to max_bypass_hours
        interlock = sis_manager.get_interlock(sample_temperature_interlock.interlock_id)
        if interlock.bypass_expiry:
            duration = (interlock.bypass_expiry - datetime.now(timezone.utc)).total_seconds() / 3600
            assert duration <= sis_manager.max_bypass_hours + 0.1, (
                f"Bypass duration {duration}h should not exceed {sis_manager.max_bypass_hours}h"
            )

    @pytest.mark.compliance
    def test_audit_log_generation(self, sis_manager, sample_temperature_interlock):
        """Test audit log entries are generated for SIS operations."""
        sis_manager.register_interlock(sample_temperature_interlock)

        # Get audit log
        audit_log = sis_manager.get_audit_log()

        assert len(audit_log) > 0, "Should have audit log entries"

        # Check for registration event
        registration_events = [
            e for e in audit_log if e.get("event_type") == "INTERLOCK_REGISTERED"
        ]
        assert len(registration_events) > 0, (
            "Should have INTERLOCK_REGISTERED event"
        )

        # Check provenance hash
        for event in audit_log:
            assert "provenance_hash" in event, "Events should have provenance hash"
            assert len(event["provenance_hash"]) >= 16, "Hash should be at least 16 chars"


# Add BypassReason import for the test
if SIS_MODULE_AVAILABLE:
    from greenlang.agents.process_heat.gl_001_thermal_command.sis_integration import BypassReason
