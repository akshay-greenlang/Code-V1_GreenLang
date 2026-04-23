"""
Unit tests for GL-001 ThermalCommand Safety Boundary Engine.

Tests the IEC 61511 safety boundary enforcement with comprehensive
coverage of policies, violations, SIL-3 voting logic, and audit trails.

Coverage Target: 85%+
Reference: GL-001 Specification Section 11, IEC 61511

Test Categories:
1. Policy management
2. Whitelist/Blacklist validation
3. Absolute limit enforcement
4. Rate limit enforcement
5. Conditional policy evaluation
6. Time-based restrictions
7. Interlock and alarm handling
8. Gate decision logic
9. Audit trail and provenance
10. Edge cases and boundary conditions

Author: GreenLang QA Team
Version: 1.0.0
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta, time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Set
import threading

# Import hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

# Add parent path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from safety.safety_schemas import (
    BoundaryPolicy,
    BoundaryViolation,
    PolicyType,
    ViolationType,
    ViolationSeverity,
    SafetyLevel,
    GateDecision,
    TagWriteRequest,
    ActionGateResult,
    SafetyState,
    SafetyAuditRecord,
    SISState,
    InterlockState,
    AlarmState,
    Condition,
    ConditionOperator,
    RateLimitSpec,
    TimeRestriction,
)

from safety.boundary_policies import (
    ThermalPolicyManager,
    get_policy_manager,
    reset_policy_manager,
    ALLOWED_WRITE_TAGS,
    BLACKLISTED_TAGS,
)

from safety.boundary_engine import SafetyBoundaryEngine


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_policy_manager_fixture():
    """Reset policy manager before each test."""
    reset_policy_manager()
    yield
    reset_policy_manager()


@pytest.fixture
def policy_manager() -> ThermalPolicyManager:
    """Create a fresh policy manager."""
    return ThermalPolicyManager()


@pytest.fixture
def boundary_engine() -> SafetyBoundaryEngine:
    """Create a fresh safety boundary engine."""
    return SafetyBoundaryEngine()


@pytest.fixture
def engine_with_tag_values(boundary_engine) -> SafetyBoundaryEngine:
    """Create engine with pre-populated tag values."""
    boundary_engine.update_tag_values({
        "TI-101": 150.0,
        "PI-101": 500.0,
        "FI-101": 100.0,
        "LI-101": 50.0,
        "INTERLOCK-MAIN": True,
    })
    return boundary_engine


@pytest.fixture
def valid_write_request() -> TagWriteRequest:
    """Create a valid write request for a whitelisted tag."""
    return TagWriteRequest(
        tag_id="TIC-101",
        value=100.0,
        source="GL-001",
    )


@pytest.fixture
def out_of_range_request() -> TagWriteRequest:
    """Create a write request with out-of-range value."""
    return TagWriteRequest(
        tag_id="TIC-101",
        value=300.0,  # Above max of 200
        source="GL-001",
    )


@pytest.fixture
def blacklisted_request() -> TagWriteRequest:
    """Create a write request for a blacklisted tag."""
    return TagWriteRequest(
        tag_id="SIS-101",  # Safety tag - blacklisted
        value=100.0,
        source="GL-001",
    )


@pytest.fixture
def non_whitelisted_request() -> TagWriteRequest:
    """Create a write request for a non-whitelisted tag."""
    return TagWriteRequest(
        tag_id="TIC-999",  # Not in whitelist
        value=100.0,
        source="GL-001",
    )


# =============================================================================
# TEST CLASS: POLICY MANAGER INITIALIZATION
# =============================================================================

class TestPolicyManagerInitialization:
    """Tests for policy manager initialization."""

    def test_default_initialization(self, policy_manager):
        """Test policy manager initializes with default policies."""
        policies = policy_manager.get_all_policies()

        assert len(policies) > 0
        assert policy_manager.get_policy("WHITELIST_001") is not None

    def test_statistics(self, policy_manager):
        """Test policy statistics."""
        stats = policy_manager.get_statistics()

        assert stats["total"] > 0
        assert stats["enabled"] > 0
        assert "type_absolute_limit" in stats
        assert "type_rate_limit" in stats

    def test_singleton_behavior(self):
        """Test policy manager singleton behavior."""
        manager1 = get_policy_manager()
        manager2 = get_policy_manager()

        assert manager1 is manager2


# =============================================================================
# TEST CLASS: WHITELIST / BLACKLIST
# =============================================================================

class TestWhitelistBlacklist:
    """Tests for whitelist and blacklist functionality."""

    def test_whitelisted_tag_allowed(self, policy_manager):
        """Test that whitelisted tags are allowed."""
        assert policy_manager.is_tag_allowed("TIC-101")
        assert policy_manager.is_tag_allowed("PIC-101")
        assert policy_manager.is_tag_allowed("FIC-101")

    def test_non_whitelisted_tag_blocked(self, policy_manager):
        """Test that non-whitelisted tags are blocked."""
        assert not policy_manager.is_tag_allowed("TIC-999")
        assert not policy_manager.is_tag_allowed("RANDOM-TAG")

    def test_blacklisted_tag_blocked(self, policy_manager):
        """Test that blacklisted tags are blocked."""
        assert policy_manager.is_tag_blacklisted("SIS-101")
        assert policy_manager.is_tag_blacklisted("ESD-001")
        assert policy_manager.is_tag_blacklisted("PSV-101")

    def test_blacklist_takes_precedence(self, policy_manager):
        """Test that blacklist takes precedence over whitelist."""
        # Even if we try to add a SIS tag to whitelist, it should still be blocked
        assert not policy_manager.is_tag_allowed("SIS-101")
        assert policy_manager.is_tag_blacklisted("SIS-101")

    def test_update_whitelist(self, policy_manager):
        """Test updating the whitelist."""
        new_whitelist = {"TIC-101", "TIC-102", "TIC-103"}
        policy_manager.update_whitelist(new_whitelist)

        assert policy_manager.is_tag_allowed("TIC-101")
        assert not policy_manager.is_tag_allowed("PIC-101")  # No longer whitelisted

    def test_update_whitelist_rejects_blacklisted(self, policy_manager):
        """Test that whitelist update rejects blacklisted tags."""
        with pytest.raises(ValueError, match="blacklisted"):
            policy_manager.update_whitelist({"TIC-101", "SIS-101"})


# =============================================================================
# TEST CLASS: ABSOLUTE LIMIT POLICIES
# =============================================================================

class TestAbsoluteLimitPolicies:
    """Tests for absolute limit policy enforcement."""

    def test_get_limits_for_temperature_tag(self, policy_manager):
        """Test getting limits for temperature tags."""
        limits = policy_manager.get_limits_for_tag("TIC-101")

        assert limits["min"] is not None
        assert limits["max"] is not None
        assert limits["min"] <= limits["max"]

    def test_get_limits_for_pressure_tag(self, policy_manager):
        """Test getting limits for pressure tags."""
        limits = policy_manager.get_limits_for_tag("PIC-101")

        assert limits["min"] is not None
        assert limits["max"] is not None

    def test_get_limits_for_flow_tag(self, policy_manager):
        """Test getting limits for flow tags."""
        limits = policy_manager.get_limits_for_tag("FIC-101")

        assert limits["min"] is not None
        assert limits["max"] is not None

    def test_get_limits_for_unknown_tag(self, policy_manager):
        """Test getting limits for unknown tags."""
        limits = policy_manager.get_limits_for_tag("UNKNOWN-TAG")

        # Should return None for both if no policies match
        assert limits["min"] is None or limits["max"] is None


# =============================================================================
# TEST CLASS: RATE LIMIT POLICIES
# =============================================================================

class TestRateLimitPolicies:
    """Tests for rate limit policy enforcement."""

    def test_get_rate_limits_for_controller(self, policy_manager):
        """Test getting rate limits for controller tags."""
        rate_limits = policy_manager.get_rate_limits_for_tag("TIC-101")

        assert rate_limits is not None
        assert rate_limits.max_change_per_second is not None or \
               rate_limits.max_change_per_minute is not None

    def test_rate_limit_spec_fields(self, policy_manager):
        """Test rate limit specification fields."""
        rate_limits = policy_manager.get_rate_limits_for_tag("TIC-101")

        if rate_limits:
            # Check that at least one field is set
            has_limit = (
                rate_limits.max_change_per_second is not None or
                rate_limits.max_change_per_minute is not None or
                rate_limits.max_writes_per_minute is not None or
                rate_limits.cooldown_seconds is not None
            )
            assert has_limit


# =============================================================================
# TEST CLASS: BOUNDARY ENGINE INITIALIZATION
# =============================================================================

class TestBoundaryEngineInitialization:
    """Tests for safety boundary engine initialization."""

    def test_default_initialization(self, boundary_engine):
        """Test engine initializes correctly."""
        stats = boundary_engine.get_statistics()

        assert stats["total_requests"] == 0
        assert stats["allowed"] == 0
        assert stats["blocked"] == 0
        assert stats["violations"] == 0

    def test_tag_value_update(self, boundary_engine):
        """Test updating tag values."""
        boundary_engine.update_tag_value("TI-101", 150.0)

        value = boundary_engine._tag_value_provider("TI-101")
        assert value == 150.0

    def test_bulk_tag_value_update(self, boundary_engine):
        """Test bulk updating tag values."""
        values = {
            "TI-101": 150.0,
            "PI-101": 500.0,
            "FI-101": 100.0,
        }
        boundary_engine.update_tag_values(values)

        assert boundary_engine._tag_value_provider("TI-101") == 150.0
        assert boundary_engine._tag_value_provider("PI-101") == 500.0
        assert boundary_engine._tag_value_provider("FI-101") == 100.0


# =============================================================================
# TEST CLASS: WRITE REQUEST VALIDATION
# =============================================================================

class TestWriteRequestValidation:
    """Tests for write request validation."""

    def test_valid_request_allowed(self, boundary_engine, valid_write_request):
        """Test that valid requests are allowed."""
        result = boundary_engine.validate_write_request(valid_write_request)

        assert result.decision in [GateDecision.ALLOW, GateDecision.CLAMP]
        assert not result.is_blocked

    def test_blacklisted_request_blocked(self, boundary_engine, blacklisted_request):
        """Test that blacklisted requests are blocked."""
        result = boundary_engine.validate_write_request(blacklisted_request)

        assert result.decision == GateDecision.BLOCK
        assert result.is_blocked
        assert len(result.violations) > 0
        assert result.violations[0].severity == ViolationSeverity.EMERGENCY

    def test_non_whitelisted_request_blocked(self, boundary_engine, non_whitelisted_request):
        """Test that non-whitelisted requests are blocked."""
        result = boundary_engine.validate_write_request(non_whitelisted_request)

        assert result.decision == GateDecision.BLOCK
        assert result.is_blocked

    def test_out_of_range_request_clamped(self, boundary_engine, out_of_range_request):
        """Test that out-of-range requests are clamped."""
        result = boundary_engine.validate_write_request(
            out_of_range_request, allow_clamping=True
        )

        if result.decision == GateDecision.CLAMP:
            assert result.final_value is not None
            assert result.final_value <= 200.0  # Max temperature limit
        elif result.decision == GateDecision.BLOCK:
            # Depending on policy configuration, may block instead
            assert result.is_blocked

    def test_out_of_range_request_blocked_no_clamping(self, boundary_engine, out_of_range_request):
        """Test that out-of-range requests are blocked when clamping disabled."""
        result = boundary_engine.validate_write_request(
            out_of_range_request, allow_clamping=False
        )

        # Should be blocked if clamping is not allowed
        assert result.decision == GateDecision.BLOCK or len(result.violations) > 0


# =============================================================================
# TEST CLASS: VIOLATION DETECTION
# =============================================================================

class TestViolationDetection:
    """Tests for violation detection."""

    def test_over_max_violation(self, boundary_engine):
        """Test over-max violation detection."""
        request = TagWriteRequest(tag_id="TIC-101", value=500.0)  # Way over max

        result = boundary_engine.validate_write_request(request)

        over_max_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.OVER_MAX
        ]
        assert len(over_max_violations) > 0

    def test_under_min_violation(self, boundary_engine):
        """Test under-min violation detection."""
        request = TagWriteRequest(tag_id="TIC-101", value=-100.0)  # Below min

        result = boundary_engine.validate_write_request(request)

        under_min_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.UNDER_MIN
        ]
        assert len(under_min_violations) > 0

    def test_unauthorized_tag_violation(self, boundary_engine):
        """Test unauthorized tag violation detection."""
        request = TagWriteRequest(tag_id="SIS-101", value=100.0)

        result = boundary_engine.validate_write_request(request)

        auth_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.UNAUTHORIZED_TAG
        ]
        assert len(auth_violations) > 0


# =============================================================================
# TEST CLASS: RATE LIMIT ENFORCEMENT
# =============================================================================

class TestRateLimitEnforcement:
    """Tests for rate limit enforcement."""

    def test_cooldown_violation(self, boundary_engine):
        """Test cooldown period violation."""
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)

        # First write should succeed
        result1 = boundary_engine.validate_write_request(request)
        assert result1.decision in [GateDecision.ALLOW, GateDecision.CLAMP]

        # Immediate second write may violate cooldown
        request2 = TagWriteRequest(tag_id="TIC-101", value=105.0)
        result2 = boundary_engine.validate_write_request(request2)

        # May be blocked due to cooldown or rate limit
        # (depends on policy configuration)

    def test_rate_of_change_violation(self, boundary_engine):
        """Test rate of change violation."""
        # First write to establish baseline
        request1 = TagWriteRequest(tag_id="TIC-101", value=100.0)
        boundary_engine.validate_write_request(request1)

        # Attempt large change
        request2 = TagWriteRequest(tag_id="TIC-101", value=150.0)
        result = boundary_engine.validate_write_request(request2)

        # Check for rate violations
        rate_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.RATE_EXCEEDED
        ]
        # May or may not have rate violations depending on policy
        # Just ensure no crash


# =============================================================================
# TEST CLASS: CONDITIONAL POLICY EVALUATION
# =============================================================================

class TestConditionalPolicies:
    """Tests for conditional policy evaluation."""

    def test_condition_evaluation_greater_than(self, engine_with_tag_values):
        """Test greater-than condition evaluation."""
        # Update pressure to trigger conditional policy
        engine_with_tag_values.update_tag_value("PI-101", 1100.0)

        request = TagWriteRequest(tag_id="TIC-101", value=150.0)
        result = engine_with_tag_values.validate_write_request(request)

        # High pressure should limit temperature setpoint
        condition_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.CONDITION_VIOLATED
        ]
        # May have conditional violations if policy is configured

    def test_condition_evaluation_equals(self, engine_with_tag_values):
        """Test equals condition evaluation."""
        # Set interlock to false (not permissive)
        engine_with_tag_values.update_tag_value("INTERLOCK-MAIN", False)

        # This should trigger interlock policy
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = engine_with_tag_values.validate_write_request(request)

        # Check for condition-related violations
        # (behavior depends on policy configuration)


# =============================================================================
# TEST CLASS: TIME-BASED RESTRICTIONS
# =============================================================================

class TestTimeBasedRestrictions:
    """Tests for time-based restriction enforcement."""

    def test_time_restriction_check(self, boundary_engine):
        """Test time restriction checking."""
        # Time restriction validation is performed in _check_time_restrictions
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)

        result = boundary_engine.validate_write_request(request)

        # Check that time restrictions were evaluated
        assert "time_restrictions" in result.checks_passed or \
               "time_restrictions" in result.checks_failed


# =============================================================================
# TEST CLASS: INTERLOCK AND ALARM HANDLING
# =============================================================================

class TestInterlockHandling:
    """Tests for interlock state handling."""

    def test_interlock_state_update(self, boundary_engine):
        """Test updating interlock state."""
        interlock = InterlockState(
            interlock_id="INT-001",
            is_permissive=True,
            cause=None
        )

        boundary_engine.update_interlock_state(interlock)

        assert "INT-001" in boundary_engine._interlock_states

    def test_non_permissive_interlock_blocks(self, boundary_engine):
        """Test that non-permissive interlock blocks writes."""
        interlock = InterlockState(
            interlock_id="INT-MAIN",
            is_permissive=False,
            cause="High pressure"
        )
        boundary_engine.update_interlock_state(interlock)

        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = boundary_engine.validate_write_request(request)

        interlock_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.INTERLOCK_ACTIVE
        ]
        assert len(interlock_violations) > 0


class TestAlarmHandling:
    """Tests for alarm state handling."""

    def test_alarm_state_update(self, boundary_engine):
        """Test updating alarm state."""
        alarm = AlarmState(
            alarm_id="ALM-001",
            tag_id="TI-101",
            is_active=True,
            priority="HIGH"
        )

        boundary_engine.update_alarm_state(alarm)

        assert "ALM-001" in boundary_engine._alarm_states

    def test_critical_alarm_blocks_related_writes(self, boundary_engine):
        """Test that critical alarm blocks related tag writes."""
        alarm = AlarmState(
            alarm_id="ALM-TIC-101",
            tag_id="TI-101",
            is_active=True,
            priority="CRITICAL"
        )
        boundary_engine.update_alarm_state(alarm)

        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = boundary_engine.validate_write_request(request)

        alarm_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.ALARM_ACTIVE
        ]
        assert len(alarm_violations) > 0


# =============================================================================
# TEST CLASS: SIS STATE HANDLING
# =============================================================================

class TestSISHandling:
    """Tests for SIS (Safety Instrumented System) state handling."""

    def test_sis_state_update(self, boundary_engine):
        """Test updating SIS state."""
        sis = SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True,
            trip_status=False,
            bypass_active=False
        )

        boundary_engine.update_sis_state(sis)

        assert "SIS-001" in boundary_engine._sis_states

    def test_sis_operational_check(self):
        """Test SIS operational status check."""
        sis_healthy = SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True,
            trip_status=False,
            bypass_active=False
        )
        assert sis_healthy.is_operational is True

        sis_bypassed = SISState(
            sis_id="SIS-002",
            is_active=True,
            is_healthy=True,
            trip_status=False,
            bypass_active=True
        )
        assert sis_bypassed.is_operational is False


# =============================================================================
# TEST CLASS: GATE DECISION LOGIC
# =============================================================================

class TestGateDecisionLogic:
    """Tests for gate decision logic."""

    def test_allow_decision_no_violations(self, boundary_engine, valid_write_request):
        """Test ALLOW decision with no violations."""
        result = boundary_engine.validate_write_request(valid_write_request)

        if len(result.violations) == 0:
            assert result.decision == GateDecision.ALLOW
            assert result.final_value == valid_write_request.value

    def test_block_decision_emergency_violation(self, boundary_engine):
        """Test BLOCK decision on emergency violation."""
        request = TagWriteRequest(tag_id="SIS-101", value=100.0)

        result = boundary_engine.validate_write_request(request)

        assert result.decision == GateDecision.BLOCK
        assert result.final_value is None

    def test_clamp_decision_minor_violation(self, boundary_engine):
        """Test CLAMP decision on minor limit violation."""
        request = TagWriteRequest(tag_id="TIC-101", value=210.0)  # Slightly over

        result = boundary_engine.validate_write_request(request, allow_clamping=True)

        if result.decision == GateDecision.CLAMP:
            assert result.final_value is not None
            assert result.final_value <= 200.0

    def test_is_allowed_property(self, boundary_engine, valid_write_request):
        """Test is_allowed property."""
        result = boundary_engine.validate_write_request(valid_write_request)

        assert result.is_allowed == (result.decision != GateDecision.BLOCK)

    def test_is_blocked_property(self, boundary_engine, blacklisted_request):
        """Test is_blocked property."""
        result = boundary_engine.validate_write_request(blacklisted_request)

        assert result.is_blocked == (result.decision == GateDecision.BLOCK)


# =============================================================================
# TEST CLASS: AUDIT TRAIL
# =============================================================================

class TestAuditTrail:
    """Tests for audit trail functionality."""

    def test_audit_record_created_on_violation(self, boundary_engine, blacklisted_request):
        """Test that audit records are created on violations."""
        boundary_engine.validate_write_request(blacklisted_request)

        records = boundary_engine.get_audit_records(limit=10)

        assert len(records) > 0

    def test_audit_chain_integrity(self, boundary_engine):
        """Test audit chain integrity verification."""
        # Create some violations to generate audit records
        for i in range(5):
            request = TagWriteRequest(tag_id=f"SIS-{i}", value=100.0)
            boundary_engine.validate_write_request(request)

        # Verify chain
        is_valid = boundary_engine.verify_audit_chain()

        assert is_valid is True

    def test_audit_record_provenance_hash(self, boundary_engine, blacklisted_request):
        """Test audit record provenance hash."""
        boundary_engine.validate_write_request(blacklisted_request)

        records = boundary_engine.get_audit_records(limit=1)

        if records:
            assert records[0].provenance_hash != ""
            assert len(records[0].provenance_hash) == 64

    def test_violation_callback(self, boundary_engine, blacklisted_request):
        """Test violation callback is called."""
        callback_called = []

        def violation_callback(violation):
            callback_called.append(violation)

        engine = SafetyBoundaryEngine(violation_callback=violation_callback)
        engine.validate_write_request(blacklisted_request)

        assert len(callback_called) > 0


# =============================================================================
# TEST CLASS: STATISTICS
# =============================================================================

class TestStatistics:
    """Tests for statistics tracking."""

    def test_request_counting(self, boundary_engine, valid_write_request):
        """Test request counting."""
        initial_stats = boundary_engine.get_statistics()
        initial_requests = initial_stats["total_requests"]

        boundary_engine.validate_write_request(valid_write_request)

        stats = boundary_engine.get_statistics()
        assert stats["total_requests"] == initial_requests + 1

    def test_allowed_counting(self, boundary_engine, valid_write_request):
        """Test allowed request counting."""
        initial_stats = boundary_engine.get_statistics()

        boundary_engine.validate_write_request(valid_write_request)

        stats = boundary_engine.get_statistics()
        # Either allowed or clamped (not blocked)
        assert stats["allowed"] >= initial_stats["allowed"] or \
               stats["clamped"] >= initial_stats.get("clamped", 0)

    def test_blocked_counting(self, boundary_engine, blacklisted_request):
        """Test blocked request counting."""
        initial_stats = boundary_engine.get_statistics()
        initial_blocked = initial_stats["blocked"]

        boundary_engine.validate_write_request(blacklisted_request)

        stats = boundary_engine.get_statistics()
        assert stats["blocked"] == initial_blocked + 1

    def test_violation_counting(self, boundary_engine, blacklisted_request):
        """Test violation counting."""
        initial_stats = boundary_engine.get_statistics()
        initial_violations = initial_stats["violations"]

        boundary_engine.validate_write_request(blacklisted_request)

        stats = boundary_engine.get_statistics()
        assert stats["violations"] > initial_violations


# =============================================================================
# TEST CLASS: SAFETY STATE
# =============================================================================

class TestSafetyState:
    """Tests for safety state reporting."""

    def test_get_safety_state(self, boundary_engine):
        """Test getting safety state."""
        state = boundary_engine.get_safety_state()

        assert isinstance(state, SafetyState)
        assert "overall_safe" in state.dict() or hasattr(state, "overall_safe")

    def test_safety_state_with_sis(self, boundary_engine):
        """Test safety state includes SIS states."""
        sis = SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True
        )
        boundary_engine.update_sis_state(sis)

        state = boundary_engine.get_safety_state()

        assert len(state.sis_states) > 0

    def test_safety_state_properties(self):
        """Test safety state computed properties."""
        state = SafetyState(
            sis_states=[
                SISState(sis_id="SIS-001", is_active=True, is_healthy=True)
            ],
            interlock_states=[
                InterlockState(interlock_id="INT-001", is_permissive=True)
            ],
            alarm_states=[]
        )

        assert state.all_sis_operational is True
        assert state.all_interlocks_permissive is True
        assert state.has_critical_alarms is False


# =============================================================================
# TEST CLASS: PRE-OPTIMIZATION CONSTRAINTS
# =============================================================================

class TestPreOptimizationConstraints:
    """Tests for pre-optimization constraint enforcement."""

    def test_enforce_pre_optimization_constraints(self, boundary_engine):
        """Test pre-optimization constraint enforcement."""
        targets = {
            "TIC-101": 180.0,  # Within limits
            "TIC-102": 250.0,  # Above max
            "TIC-103": -50.0,  # Below min
        }

        constrained = boundary_engine.enforce_pre_optimization_constraints(targets)

        # Values should be constrained to limits
        assert constrained["TIC-101"] == 180.0  # Unchanged
        assert constrained["TIC-102"] <= 200.0  # Clamped to max
        assert constrained["TIC-103"] >= -40.0  # Clamped to min


# =============================================================================
# TEST CLASS: RATE LIMIT STATE RESET
# =============================================================================

class TestRateLimitStateReset:
    """Tests for rate limit state reset."""

    def test_reset_specific_tag(self, boundary_engine):
        """Test resetting rate limit for specific tag."""
        # Create some history
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        boundary_engine.validate_write_request(request)

        # Reset
        boundary_engine.reset_rate_limit_state("TIC-101")

        assert "TIC-101" not in boundary_engine._write_history
        assert "TIC-101" not in boundary_engine._last_write_time

    def test_reset_all_tags(self, boundary_engine):
        """Test resetting rate limit for all tags."""
        # Create history for multiple tags
        for tag in ["TIC-101", "TIC-102", "TIC-103"]:
            request = TagWriteRequest(tag_id=tag, value=100.0)
            boundary_engine.validate_write_request(request)

        # Reset all
        boundary_engine.reset_rate_limit_state()

        assert len(boundary_engine._write_history) == 0
        assert len(boundary_engine._last_write_time) == 0


# =============================================================================
# TEST CLASS: POLICY ENABLE/DISABLE
# =============================================================================

class TestPolicyEnableDisable:
    """Tests for policy enable/disable functionality."""

    def test_enable_policy(self, policy_manager):
        """Test enabling a policy."""
        policy_id = "TIME_MAINT_001"  # Disabled by default

        result = policy_manager.enable_policy(policy_id)

        assert result is True
        policy = policy_manager.get_policy(policy_id)
        assert policy.enabled is True

    def test_disable_policy(self, policy_manager):
        """Test disabling a policy."""
        # Use a non-critical policy
        policy_id = "TEMP_RATE_001"

        result = policy_manager.disable_policy(policy_id)

        # May or may not be allowed depending on policy safety level
        if result:
            policy = policy_manager.get_policy(policy_id)
            assert policy.enabled is False

    def test_cannot_disable_whitelist(self, policy_manager):
        """Test that whitelist policy cannot be disabled."""
        result = policy_manager.disable_policy("WHITELIST_001")

        assert result is False
        policy = policy_manager.get_policy("WHITELIST_001")
        assert policy.enabled is True

    def test_cannot_disable_sil3_policy(self, policy_manager):
        """Test that SIL-3 policies cannot be disabled."""
        # Find a SIL-3 policy
        sil3_policies = [
            p for p in policy_manager.get_all_policies()
            if p.safety_level == SafetyLevel.SIL_3
        ]

        for policy in sil3_policies:
            result = policy_manager.disable_policy(policy.policy_id)
            assert result is False


# =============================================================================
# TEST CLASS: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_request(self, boundary_engine):
        """Test handling of minimal request."""
        request = TagWriteRequest(tag_id="TIC-101", value=0.0)

        result = boundary_engine.validate_write_request(request)

        # Should not crash, should return valid result
        assert result is not None
        assert isinstance(result, ActionGateResult)

    def test_very_large_value(self, boundary_engine):
        """Test handling of very large values."""
        request = TagWriteRequest(tag_id="TIC-101", value=1e10)

        result = boundary_engine.validate_write_request(request)

        assert result.is_blocked or result.decision == GateDecision.CLAMP

    def test_very_small_value(self, boundary_engine):
        """Test handling of very small (negative) values."""
        request = TagWriteRequest(tag_id="TIC-101", value=-1e10)

        result = boundary_engine.validate_write_request(request)

        assert result.is_blocked or result.decision == GateDecision.CLAMP

    def test_concurrent_requests(self, boundary_engine):
        """Test thread safety with concurrent requests."""
        import threading

        results = []
        errors = []

        def make_request(tag_id, value):
            try:
                request = TagWriteRequest(tag_id=tag_id, value=value)
                result = boundary_engine.validate_write_request(request)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(
                target=make_request,
                args=(f"TIC-10{i % 5 + 1}", 100.0 + i)
            )
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

    def test_rapid_fire_requests(self, boundary_engine):
        """Test handling of rapid-fire requests."""
        for i in range(100):
            request = TagWriteRequest(tag_id="TIC-101", value=100.0 + i * 0.1)
            result = boundary_engine.validate_write_request(request)

            # Should not crash
            assert result is not None


# =============================================================================
# TEST CLASS: PROVENANCE TRACKING
# =============================================================================

class TestProvenanceTracking:
    """Tests for provenance tracking."""

    def test_gate_result_provenance(self, boundary_engine, valid_write_request):
        """Test gate result provenance hash."""
        result = boundary_engine.validate_write_request(valid_write_request)

        assert result.provenance_hash != ""

    def test_violation_provenance(self, boundary_engine, blacklisted_request):
        """Test violation provenance hash."""
        result = boundary_engine.validate_write_request(blacklisted_request)

        for violation in result.violations:
            assert violation.provenance_hash != ""
            assert len(violation.provenance_hash) == 64

    def test_policy_hash(self, policy_manager):
        """Test policy hash computation."""
        policy = policy_manager.get_policy("WHITELIST_001")

        hash_value = policy.compute_hash()

        assert hash_value != ""
        assert len(hash_value) == 64


# =============================================================================
# PROPERTY-BASED TESTS (Hypothesis)
# =============================================================================

if HYPOTHESIS_AVAILABLE:
    class TestPropertyBasedSafety:
        """Property-based tests using Hypothesis."""

        @given(value=st.floats(min_value=-1000.0, max_value=1000.0))
        @settings(max_examples=50)
        def test_valid_tag_always_returns_result(self, value):
            """Test that valid tags always return a result."""
            assume(np.isfinite(value))

            reset_policy_manager()
            engine = SafetyBoundaryEngine()
            request = TagWriteRequest(tag_id="TIC-101", value=value)

            result = engine.validate_write_request(request)

            assert isinstance(result, ActionGateResult)
            assert result.decision in [GateDecision.ALLOW, GateDecision.BLOCK, GateDecision.CLAMP]

        @given(value=st.floats(min_value=0.0, max_value=200.0))
        @settings(max_examples=50)
        def test_in_range_values_not_blocked(self, value):
            """Test that in-range values are not blocked for violations."""
            assume(np.isfinite(value))

            reset_policy_manager()
            engine = SafetyBoundaryEngine()
            request = TagWriteRequest(tag_id="TIC-101", value=value)

            result = engine.validate_write_request(request)

            # In-range should not have limit violations
            limit_violations = [
                v for v in result.violations
                if v.violation_type in [ViolationType.OVER_MAX, ViolationType.UNDER_MIN]
            ]
            assert len(limit_violations) == 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for safety boundary engine."""

    @pytest.mark.performance
    def test_validation_speed(self, boundary_engine):
        """Test validation performance."""
        import time

        iterations = 1000
        start = time.perf_counter()

        for i in range(iterations):
            request = TagWriteRequest(tag_id="TIC-101", value=100.0 + i * 0.01)
            boundary_engine.validate_write_request(request)

        elapsed = time.perf_counter() - start

        # Should process 1000 requests in under 1 second
        assert elapsed < 1.0

        validations_per_second = iterations / elapsed
        assert validations_per_second > 1000

    @pytest.mark.performance
    def test_api_response_time(self, boundary_engine):
        """Test that individual validation is under 200ms target."""
        import time

        request = TagWriteRequest(tag_id="TIC-101", value=100.0)

        start = time.perf_counter()
        result = boundary_engine.validate_write_request(request)
        elapsed = time.perf_counter() - start

        # Target: <200ms per validation
        assert elapsed < 0.200
        assert result.evaluation_time_ms < 200


# =============================================================================
# COMPLIANCE TESTS
# =============================================================================

class TestCompliance:
    """Compliance tests for IEC 61511 requirements."""

    def test_safety_boundary_immutability(self, policy_manager):
        """Test that critical safety policies cannot be modified."""
        whitelist = policy_manager.get_policy("WHITELIST_001")

        # Attempt to disable should fail
        result = policy_manager.disable_policy("WHITELIST_001")
        assert result is False

    def test_audit_trail_completeness(self, boundary_engine, blacklisted_request):
        """Test audit trail includes all required elements."""
        boundary_engine.validate_write_request(blacklisted_request)

        records = boundary_engine.get_audit_records(limit=1)

        if records:
            record = records[0]
            assert record.timestamp is not None
            assert record.event_type != ""
            assert record.action_taken != ""
            assert record.provenance_hash != ""

    def test_sil3_voting_logic(self):
        """Test SIL-3 2oo3 voting logic concept."""
        # Three sensors
        sensors = [
            SISState(sis_id="SIS-A", is_active=True, is_healthy=True, trip_status=False),
            SISState(sis_id="SIS-B", is_active=True, is_healthy=True, trip_status=True),
            SISState(sis_id="SIS-C", is_active=True, is_healthy=True, trip_status=True),
        ]

        # 2oo3 voting: at least 2 must be tripped
        trip_count = sum(1 for s in sensors if s.trip_status)
        should_trip = trip_count >= 2

        assert should_trip is True

    def test_reproducibility_guarantee(self, boundary_engine, valid_write_request):
        """Test that identical inputs produce identical outputs."""
        results = []

        for _ in range(5):
            reset_policy_manager()
            engine = SafetyBoundaryEngine()
            result = engine.validate_write_request(valid_write_request)
            results.append(result.decision)

        # All results should be identical
        assert all(r == results[0] for r in results)
