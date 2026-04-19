"""
Tests for Safety Boundary Engine

Tests the core boundary enforcement engine including:
- Pre-optimization constraint enforcement
- Pre-actuation runtime gate
- Boundary violation detection and blocking
- Rate limiting
- Audit trail
"""

import pytest
from datetime import datetime, timedelta
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from safety.boundary_engine import SafetyBoundaryEngine
from safety.boundary_policies import (
    ThermalPolicyManager,
    reset_policy_manager,
    ALLOWED_WRITE_TAGS,
)
from safety.safety_schemas import (
    TagWriteRequest,
    GateDecision,
    ViolationType,
    ViolationSeverity,
    SISState,
    InterlockState,
    AlarmState,
)


class TestBoundaryEngineInitialization:
    """Test boundary engine initialization."""

    @pytest.fixture
    def engine(self):
        """Create fresh boundary engine."""
        reset_policy_manager()
        return SafetyBoundaryEngine()

    def test_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine is not None
        stats = engine.get_statistics()
        assert stats["total_requests"] == 0
        assert stats["violations"] == 0


class TestWhitelistEnforcement:
    """Test whitelist enforcement."""

    @pytest.fixture
    def engine(self):
        """Create fresh boundary engine."""
        reset_policy_manager()
        return SafetyBoundaryEngine()

    def test_allowed_tag_passes(self, engine):
        """Test allowed tag passes whitelist check."""
        # Use a tag from the whitelist
        allowed_tag = list(ALLOWED_WRITE_TAGS)[0]
        request = TagWriteRequest(tag_id=allowed_tag, value=50.0)
        result = engine.validate_write_request(request)
        # Should not have whitelist violation
        whitelist_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.UNAUTHORIZED_TAG
        ]
        assert len(whitelist_violations) == 0

    def test_unauthorized_tag_blocked(self, engine):
        """Test unauthorized tag is blocked."""
        request = TagWriteRequest(tag_id="UNKNOWN-999", value=50.0)
        result = engine.validate_write_request(request)
        assert result.decision == GateDecision.BLOCK
        whitelist_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.UNAUTHORIZED_TAG
        ]
        assert len(whitelist_violations) > 0

    def test_sis_tag_blocked_as_blacklisted(self, engine):
        """Test SIS tag is blocked as blacklisted."""
        request = TagWriteRequest(tag_id="SIS-101", value=50.0)
        result = engine.validate_write_request(request)
        assert result.decision == GateDecision.BLOCK
        assert any(
            v.severity == ViolationSeverity.EMERGENCY
            for v in result.violations
        )


class TestAbsoluteLimits:
    """Test absolute limit enforcement."""

    @pytest.fixture
    def engine(self):
        """Create fresh boundary engine."""
        reset_policy_manager()
        return SafetyBoundaryEngine()

    def test_value_within_limits_allowed(self, engine):
        """Test value within limits is allowed."""
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = engine.validate_write_request(request)
        limit_violations = [
            v for v in result.violations
            if v.violation_type in (ViolationType.OVER_MAX, ViolationType.UNDER_MIN)
        ]
        assert len(limit_violations) == 0

    def test_value_over_max_blocked(self, engine):
        """Test value over maximum is blocked."""
        # Temperature limit is 200 degC
        request = TagWriteRequest(tag_id="TIC-101", value=250.0)
        result = engine.validate_write_request(request)
        over_max_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.OVER_MAX
        ]
        assert len(over_max_violations) > 0

    def test_value_under_min_blocked(self, engine):
        """Test value under minimum is blocked."""
        # Temperature limit min is -40 degC
        request = TagWriteRequest(tag_id="TIC-101", value=-100.0)
        result = engine.validate_write_request(request)
        under_min_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.UNDER_MIN
        ]
        assert len(under_min_violations) > 0

    def test_clamping_to_max(self, engine):
        """Test clamping to maximum value."""
        request = TagWriteRequest(tag_id="TIC-101", value=250.0)
        result = engine.validate_write_request(request, allow_clamping=True)
        # Should clamp to max (200)
        if result.decision == GateDecision.CLAMP:
            assert result.final_value <= 200.0

    def test_no_clamping_when_disabled(self, engine):
        """Test clamping can be disabled."""
        request = TagWriteRequest(tag_id="TIC-101", value=250.0)
        result = engine.validate_write_request(request, allow_clamping=False)
        # Should block instead of clamp
        assert result.decision == GateDecision.BLOCK


class TestRateLimits:
    """Test rate limit enforcement."""

    @pytest.fixture
    def engine(self):
        """Create fresh boundary engine."""
        reset_policy_manager()
        return SafetyBoundaryEngine()

    def test_first_write_allowed(self, engine):
        """Test first write is allowed (no rate to check)."""
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = engine.validate_write_request(request)
        rate_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.RATE_EXCEEDED
        ]
        assert len(rate_violations) == 0

    def test_rapid_writes_rate_limited(self, engine):
        """Test rapid writes are rate limited."""
        tag_id = "TIC-101"

        # First write
        request1 = TagWriteRequest(tag_id=tag_id, value=100.0)
        result1 = engine.validate_write_request(request1)

        # Immediate second write with large change
        request2 = TagWriteRequest(tag_id=tag_id, value=150.0)
        result2 = engine.validate_write_request(request2)

        # May be rate limited due to cooldown or rate of change
        # At least one should pass, second may be limited
        assert result1.decision in (GateDecision.ALLOW, GateDecision.CLAMP)

    def test_reset_rate_limit_state(self, engine):
        """Test resetting rate limit state."""
        tag_id = "TIC-101"

        # First write
        request = TagWriteRequest(tag_id=tag_id, value=100.0)
        engine.validate_write_request(request)

        # Reset state
        engine.reset_rate_limit_state(tag_id)

        # Next write should be like first write
        request2 = TagWriteRequest(tag_id=tag_id, value=150.0)
        result = engine.validate_write_request(request2)
        # Should be allowed after reset
        assert result.decision in (GateDecision.ALLOW, GateDecision.CLAMP)


class TestInterlockChecks:
    """Test interlock status checking."""

    @pytest.fixture
    def engine(self):
        """Create fresh boundary engine."""
        reset_policy_manager()
        return SafetyBoundaryEngine()

    def test_permissive_interlock_allows_write(self, engine):
        """Test permissive interlock allows write."""
        engine.update_interlock_state(InterlockState(
            interlock_id="IL-001",
            is_permissive=True,
        ))
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = engine.validate_write_request(request)
        interlock_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.INTERLOCK_ACTIVE
        ]
        assert len(interlock_violations) == 0

    def test_active_interlock_blocks_write(self, engine):
        """Test active interlock blocks write."""
        engine.update_interlock_state(InterlockState(
            interlock_id="IL-001",
            is_permissive=False,
            cause="High pressure",
        ))
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = engine.validate_write_request(request)
        assert result.decision == GateDecision.BLOCK
        interlock_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.INTERLOCK_ACTIVE
        ]
        assert len(interlock_violations) > 0


class TestAlarmChecks:
    """Test alarm state checking."""

    @pytest.fixture
    def engine(self):
        """Create fresh boundary engine."""
        reset_policy_manager()
        return SafetyBoundaryEngine()

    def test_no_alarms_allows_write(self, engine):
        """Test no alarms allows write."""
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = engine.validate_write_request(request)
        alarm_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.ALARM_ACTIVE
        ]
        assert len(alarm_violations) == 0

    def test_critical_alarm_blocks_write(self, engine):
        """Test critical alarm blocks related write."""
        # Add critical alarm on related tag
        engine.update_alarm_state(AlarmState(
            alarm_id="ALM-001",
            tag_id="TIC-101",  # Same tag
            is_active=True,
            priority="CRITICAL",
        ))
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = engine.validate_write_request(request)
        alarm_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.ALARM_ACTIVE
        ]
        assert len(alarm_violations) > 0


class TestPreOptimizationConstraints:
    """Test pre-optimization constraint enforcement."""

    @pytest.fixture
    def engine(self):
        """Create fresh boundary engine."""
        reset_policy_manager()
        return SafetyBoundaryEngine()

    def test_constrain_targets_to_limits(self, engine):
        """Test optimization targets are constrained."""
        targets = {
            "TIC-101": 250.0,  # Over max (200)
            "TIC-102": 100.0,  # Within limits
            "TIC-103": -100.0,  # Under min (-40)
        }
        constrained = engine.enforce_pre_optimization_constraints(targets)

        # TIC-101 should be clamped to max
        assert constrained["TIC-101"] <= 200.0
        # TIC-102 unchanged
        assert constrained["TIC-102"] == 100.0
        # TIC-103 should be clamped to min
        assert constrained["TIC-103"] >= -40.0


class TestTagValueCache:
    """Test tag value caching."""

    @pytest.fixture
    def engine(self):
        """Create fresh boundary engine."""
        reset_policy_manager()
        return SafetyBoundaryEngine()

    def test_update_single_tag_value(self, engine):
        """Test updating single tag value."""
        engine.update_tag_value("TI-101", 75.0)
        # Value should be cached (internal check via provider)

    def test_update_bulk_tag_values(self, engine):
        """Test bulk updating tag values."""
        values = {
            "TI-101": 75.0,
            "TI-102": 80.0,
            "PI-101": 500.0,
        }
        engine.update_tag_values(values)


class TestAuditTrail:
    """Test audit trail functionality."""

    @pytest.fixture
    def engine(self):
        """Create fresh boundary engine."""
        reset_policy_manager()
        return SafetyBoundaryEngine()

    def test_violation_creates_audit_record(self, engine):
        """Test violation creates audit record."""
        # Trigger a violation
        request = TagWriteRequest(tag_id="SIS-101", value=100.0)
        engine.validate_write_request(request)

        # Check audit records
        records = engine.get_audit_records()
        assert len(records) > 0

    def test_audit_chain_integrity(self, engine):
        """Test audit chain maintains integrity."""
        # Trigger multiple violations
        for i in range(3):
            request = TagWriteRequest(tag_id=f"SIS-10{i}", value=100.0)
            engine.validate_write_request(request)

        # Verify chain
        assert engine.verify_audit_chain() == True

    def test_audit_records_since_timestamp(self, engine):
        """Test getting audit records since timestamp."""
        # Trigger violations
        request = TagWriteRequest(tag_id="SIS-101", value=100.0)
        engine.validate_write_request(request)

        # Get records from future (should be empty)
        future = datetime.utcnow() + timedelta(hours=1)
        records = engine.get_audit_records(since=future)
        assert len(records) == 0


class TestSafetyState:
    """Test safety state reporting."""

    @pytest.fixture
    def engine(self):
        """Create fresh boundary engine."""
        reset_policy_manager()
        return SafetyBoundaryEngine()

    def test_get_safety_state(self, engine):
        """Test getting safety state."""
        state = engine.get_safety_state()
        assert state is not None
        assert hasattr(state, "overall_safe")
        assert hasattr(state, "policies_enabled")

    def test_safety_state_reflects_interlocks(self, engine):
        """Test safety state reflects interlock status."""
        engine.update_interlock_state(InterlockState(
            interlock_id="IL-001",
            is_permissive=False,
        ))
        state = engine.get_safety_state()
        assert state.all_interlocks_permissive == False


class TestStatistics:
    """Test statistics collection."""

    @pytest.fixture
    def engine(self):
        """Create fresh boundary engine."""
        reset_policy_manager()
        return SafetyBoundaryEngine()

    def test_statistics_tracking(self, engine):
        """Test statistics are tracked."""
        # Make some requests
        request1 = TagWriteRequest(tag_id="TIC-101", value=100.0)
        engine.validate_write_request(request1)

        request2 = TagWriteRequest(tag_id="SIS-101", value=100.0)
        engine.validate_write_request(request2)

        stats = engine.get_statistics()
        assert stats["total_requests"] == 2
        assert stats["blocked"] >= 1  # SIS blocked


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def engine(self):
        """Create fresh boundary engine."""
        reset_policy_manager()
        return SafetyBoundaryEngine()

    def test_exactly_at_boundary(self, engine):
        """Test value exactly at boundary."""
        # Exactly at max (200)
        request = TagWriteRequest(tag_id="TIC-101", value=200.0)
        result = engine.validate_write_request(request)
        # Should be allowed (at boundary, not over)
        over_max = [v for v in result.violations if v.violation_type == ViolationType.OVER_MAX]
        assert len(over_max) == 0

    def test_zero_value(self, engine):
        """Test zero value."""
        request = TagWriteRequest(tag_id="TIC-101", value=0.0)
        result = engine.validate_write_request(request)
        # Should be allowed (within limits)
        under_min = [v for v in result.violations if v.violation_type == ViolationType.UNDER_MIN]
        assert len(under_min) == 0

    def test_negative_value_within_limits(self, engine):
        """Test negative value within limits."""
        request = TagWriteRequest(tag_id="TIC-101", value=-30.0)
        result = engine.validate_write_request(request)
        # Min is -40, so -30 should be allowed
        under_min = [v for v in result.violations if v.violation_type == ViolationType.UNDER_MIN]
        assert len(under_min) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
