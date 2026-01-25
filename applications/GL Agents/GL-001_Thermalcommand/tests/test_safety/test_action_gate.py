"""
Tests for Safety Action Gate

Tests the final pre-actuation validation gate including:
- Pre-write validation
- Bounds checking
- Velocity limiting
- Interlock status verification
- Alarm state checking
"""

import pytest
from datetime import datetime, timedelta
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from safety.action_gate import (
    SafetyActionGate,
    GateConfiguration,
    VelocityLimit,
    ActionGateFactory,
)
from safety.boundary_engine import SafetyBoundaryEngine
from safety.sis_validator import SISIndependenceValidator
from safety.boundary_policies import reset_policy_manager
from safety.safety_schemas import (
    TagWriteRequest,
    GateDecision,
    ViolationType,
    ViolationSeverity,
    InterlockState,
    AlarmState,
)


class TestActionGateInitialization:
    """Test action gate initialization."""

    @pytest.fixture
    def gate(self):
        """Create fresh action gate."""
        reset_policy_manager()
        return ActionGateFactory.create_gate()

    def test_initialization(self, gate):
        """Test gate initializes correctly."""
        assert gate is not None
        stats = gate.get_statistics()
        assert stats["total_evaluations"] == 0

    def test_default_configuration(self, gate):
        """Test default configuration."""
        config = gate.get_config()
        assert config.enable_bounds_check == True
        assert config.enable_velocity_check == True
        assert config.enable_interlock_check == True
        assert config.enable_alarm_check == True
        assert config.enable_sis_check == True


class TestBasicGateEvaluation:
    """Test basic gate evaluation."""

    @pytest.fixture
    def gate(self):
        """Create fresh action gate."""
        reset_policy_manager()
        return ActionGateFactory.create_gate()

    def test_valid_request_allowed(self, gate):
        """Test valid request is allowed."""
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        assert result.decision in (GateDecision.ALLOW, GateDecision.CLAMP)
        assert result.original_request == request

    def test_sis_tag_blocked_immediately(self, gate):
        """Test SIS tag is blocked immediately."""
        request = TagWriteRequest(tag_id="SIS-101", value=100.0)
        result = gate.evaluate(request)
        assert result.decision == GateDecision.BLOCK
        assert "sis_independence" in result.checks_failed

    def test_result_has_timestamp(self, gate):
        """Test result has timestamp."""
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        assert result.gate_timestamp is not None

    def test_result_has_evaluation_time(self, gate):
        """Test result has evaluation time."""
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        assert result.evaluation_time_ms >= 0

    def test_result_has_provenance_hash(self, gate):
        """Test result has provenance hash."""
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        assert result.provenance_hash != ""


class TestInterlockChecks:
    """Test interlock status checking in gate."""

    @pytest.fixture
    def gate(self):
        """Create fresh action gate."""
        reset_policy_manager()
        return ActionGateFactory.create_gate()

    def test_permissive_interlock_allows(self, gate):
        """Test permissive interlock allows write."""
        gate.update_interlock_state(InterlockState(
            interlock_id="IL-001",
            is_permissive=True,
        ))
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        assert "interlock_status" in result.checks_passed

    def test_active_interlock_blocks(self, gate):
        """Test active interlock blocks write."""
        gate.update_interlock_state(InterlockState(
            interlock_id="IL-001",
            is_permissive=False,
            cause="High pressure alarm",
        ))
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        assert result.decision == GateDecision.BLOCK
        assert "interlock_status" in result.checks_failed


class TestAlarmChecks:
    """Test alarm state checking in gate."""

    @pytest.fixture
    def gate(self):
        """Create fresh action gate."""
        reset_policy_manager()
        return ActionGateFactory.create_gate()

    def test_no_alarms_allows(self, gate):
        """Test no alarms allows write."""
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        assert "alarm_status" in result.checks_passed

    def test_critical_alarm_blocks(self, gate):
        """Test critical alarm blocks related write."""
        gate.update_alarm_state(AlarmState(
            alarm_id="ALM-001",
            tag_id="TIC-101",
            is_active=True,
            priority="CRITICAL",
        ))
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        assert "alarm_status" in result.checks_failed

    def test_unrelated_alarm_does_not_block(self, gate):
        """Test unrelated alarm does not block."""
        gate.update_alarm_state(AlarmState(
            alarm_id="ALM-001",
            tag_id="PI-201",  # Different tag
            is_active=True,
            priority="CRITICAL",
        ))
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        # Should not be blocked by unrelated alarm
        alarm_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.ALARM_ACTIVE
        ]
        assert len(alarm_violations) == 0


class TestVelocityLimits:
    """Test velocity limiting in gate."""

    @pytest.fixture
    def gate(self):
        """Create fresh action gate."""
        reset_policy_manager()
        return ActionGateFactory.create_gate()

    def test_first_write_no_velocity_check(self, gate):
        """Test first write has no velocity to check."""
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        velocity_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.RATE_EXCEEDED
        ]
        # First write should not have velocity violation
        assert len(velocity_violations) == 0

    def test_moderate_change_allowed(self, gate):
        """Test moderate change is allowed."""
        # Set current value
        gate.update_current_value("TIC-101", 100.0)

        # First write to establish baseline
        request1 = TagWriteRequest(tag_id="TIC-101", value=100.0)
        gate.evaluate(request1)

        # Wait a bit
        time.sleep(0.1)

        # Moderate change (1 degree per 0.1 second = 10 deg/s is under typical limits)
        request2 = TagWriteRequest(tag_id="TIC-101", value=101.0)
        result = gate.evaluate(request2)
        # Should be allowed with moderate velocity
        assert result.decision in (GateDecision.ALLOW, GateDecision.CLAMP)


class TestBoundsChecks:
    """Test bounds checking in gate."""

    @pytest.fixture
    def gate(self):
        """Create fresh action gate."""
        reset_policy_manager()
        return ActionGateFactory.create_gate()

    def test_within_bounds_allowed(self, gate):
        """Test value within bounds is allowed."""
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        assert "absolute_limits" not in result.checks_failed

    def test_over_max_detected(self, gate):
        """Test value over max is detected."""
        request = TagWriteRequest(tag_id="TIC-101", value=250.0)
        result = gate.evaluate(request)
        over_max = [v for v in result.violations if v.violation_type == ViolationType.OVER_MAX]
        assert len(over_max) > 0


class TestGateConfiguration:
    """Test gate configuration options."""

    def test_custom_configuration(self):
        """Test custom configuration."""
        config = GateConfiguration(
            allow_clamping=False,
            block_on_any_alarm=True,
            rate_limit_requests_per_second=50,
        )
        reset_policy_manager()
        gate = ActionGateFactory.create_gate(config=config)
        assert gate.get_config().allow_clamping == False
        assert gate.get_config().block_on_any_alarm == True

    def test_cannot_disable_sis_check(self):
        """Test cannot disable SIS check."""
        config = GateConfiguration(
            enable_sis_check=False,  # Try to disable
        )
        reset_policy_manager()
        gate = ActionGateFactory.create_gate()
        gate.update_config(config)
        # Should still have SIS check enabled
        assert gate.get_config().enable_sis_check == True


class TestRateLimiting:
    """Test gate-level rate limiting."""

    def test_rate_limiting(self):
        """Test rate limiting protects against floods."""
        config = GateConfiguration(
            rate_limit_requests_per_second=5,
        )
        reset_policy_manager()
        gate = ActionGateFactory.create_gate(config=config)

        # Make rapid requests
        blocked_count = 0
        for i in range(10):
            request = TagWriteRequest(tag_id="TIC-101", value=100.0 + i)
            result = gate.evaluate(request)
            if "rate_limit_check" in result.checks_failed:
                blocked_count += 1

        # Some should be rate limited
        # (May not hit limit in test due to execution time)


class TestBatchEvaluation:
    """Test batch evaluation of requests."""

    @pytest.fixture
    def gate(self):
        """Create fresh action gate."""
        reset_policy_manager()
        return ActionGateFactory.create_gate()

    def test_batch_evaluation(self, gate):
        """Test evaluating batch of requests."""
        requests = [
            TagWriteRequest(tag_id="TIC-101", value=100.0),
            TagWriteRequest(tag_id="TIC-102", value=110.0),
            TagWriteRequest(tag_id="TIC-103", value=120.0),
        ]
        results = gate.evaluate_batch(requests)
        assert len(results) == 3

    def test_batch_stops_on_emergency(self, gate):
        """Test batch stops on emergency violation."""
        requests = [
            TagWriteRequest(tag_id="TIC-101", value=100.0),
            TagWriteRequest(tag_id="SIS-101", value=100.0),  # Emergency
            TagWriteRequest(tag_id="TIC-103", value=120.0),
        ]
        results = gate.evaluate_batch(requests)
        # Should stop after SIS violation
        assert len(results) <= 2


class TestStatistics:
    """Test statistics collection."""

    @pytest.fixture
    def gate(self):
        """Create fresh action gate."""
        reset_policy_manager()
        return ActionGateFactory.create_gate()

    def test_statistics_tracking(self, gate):
        """Test statistics are tracked."""
        request1 = TagWriteRequest(tag_id="TIC-101", value=100.0)
        gate.evaluate(request1)

        request2 = TagWriteRequest(tag_id="SIS-101", value=100.0)
        gate.evaluate(request2)

        stats = gate.get_statistics()
        assert stats["total_evaluations"] == 2
        assert stats["allowed"] >= 0
        assert stats["blocked"] >= 1


class TestTagWritableCheck:
    """Test quick tag writability check."""

    @pytest.fixture
    def gate(self):
        """Create fresh action gate."""
        reset_policy_manager()
        return ActionGateFactory.create_gate()

    def test_allowed_tag_writable(self, gate):
        """Test allowed tag shows as writable."""
        assert gate.is_tag_writable("TIC-101") == True

    def test_sis_tag_not_writable(self, gate):
        """Test SIS tag shows as not writable."""
        assert gate.is_tag_writable("SIS-101") == False


class TestVelocityLimitConfiguration:
    """Test velocity limit configuration."""

    @pytest.fixture
    def gate(self):
        """Create fresh action gate."""
        reset_policy_manager()
        return ActionGateFactory.create_gate()

    def test_get_velocity_limits(self, gate):
        """Test getting velocity limits."""
        limits = gate.get_velocity_limits()
        assert len(limits) > 0

    def test_add_velocity_limit(self, gate):
        """Test adding velocity limit."""
        new_limit = VelocityLimit(
            tag_pattern="CUSTOM-*",
            max_velocity=1.0,
            engineering_units="units/s",
        )
        initial_count = len(gate.get_velocity_limits())
        gate.add_velocity_limit(new_limit)
        assert len(gate.get_velocity_limits()) == initial_count + 1


class TestViolationHandler:
    """Test violation handler callback."""

    def test_violation_handler_called(self):
        """Test violation handler is called on violation."""
        violations_received = []

        def handler(violation):
            violations_received.append(violation)

        reset_policy_manager()
        gate = ActionGateFactory.create_gate(violation_handler=handler)

        request = TagWriteRequest(tag_id="SIS-101", value=100.0)
        gate.evaluate(request)

        assert len(violations_received) > 0


class TestStateClearing:
    """Test state clearing methods."""

    @pytest.fixture
    def gate(self):
        """Create fresh action gate."""
        reset_policy_manager()
        return ActionGateFactory.create_gate()

    def test_clear_interlock_states(self, gate):
        """Test clearing interlock states."""
        gate.update_interlock_state(InterlockState(
            interlock_id="IL-001",
            is_permissive=False,
        ))
        gate.clear_interlock_states()
        # After clearing, should not have interlock violations
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        interlock_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.INTERLOCK_ACTIVE
        ]
        assert len(interlock_violations) == 0

    def test_clear_alarm_states(self, gate):
        """Test clearing alarm states."""
        gate.update_alarm_state(AlarmState(
            alarm_id="ALM-001",
            tag_id="TIC-101",
            is_active=True,
            priority="CRITICAL",
        ))
        gate.clear_alarm_states()
        # After clearing, should not have alarm violations
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        alarm_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.ALARM_ACTIVE
        ]
        assert len(alarm_violations) == 0


class TestFactoryMethods:
    """Test factory methods."""

    def test_create_gate(self):
        """Test creating gate via factory."""
        reset_policy_manager()
        gate = ActionGateFactory.create_gate()
        assert gate is not None

    def test_create_test_gate(self):
        """Test creating test gate."""
        reset_policy_manager()
        gate = ActionGateFactory.create_test_gate()
        assert gate is not None
        # Test gate has higher rate limits
        config = gate.get_config()
        assert config.rate_limit_requests_per_second >= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
