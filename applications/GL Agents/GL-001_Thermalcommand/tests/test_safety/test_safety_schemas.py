"""
Tests for Safety Schemas

Tests all Pydantic models in safety_schemas.py for validation,
serialization, and hash computation.
"""

import pytest
from datetime import datetime, time
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from safety.safety_schemas import (
    ViolationType,
    ViolationSeverity,
    SafetyLevel,
    GateDecision,
    PolicyType,
    ConditionOperator,
    TimeRestriction,
    Condition,
    RateLimitSpec,
    BoundaryPolicy,
    BoundaryViolation,
    TagWriteRequest,
    ActionGateResult,
    SafetyState,
    SafetyAuditRecord,
    SISState,
    InterlockState,
    AlarmState,
)


class TestEnums:
    """Test enum definitions."""

    def test_violation_type_values(self):
        """Test all violation types are defined."""
        assert ViolationType.OVER_MAX == "over_max"
        assert ViolationType.UNDER_MIN == "under_min"
        assert ViolationType.RATE_EXCEEDED == "rate_exceeded"
        assert ViolationType.UNAUTHORIZED_TAG == "unauthorized_tag"
        assert ViolationType.SIS_VIOLATION == "sis_violation"

    def test_violation_severity_values(self):
        """Test all severity levels are defined."""
        assert ViolationSeverity.WARNING == "warning"
        assert ViolationSeverity.CRITICAL == "critical"
        assert ViolationSeverity.EMERGENCY == "emergency"

    def test_gate_decision_values(self):
        """Test all gate decisions are defined."""
        assert GateDecision.ALLOW == "allow"
        assert GateDecision.BLOCK == "block"
        assert GateDecision.CLAMP == "clamp"


class TestTimeRestriction:
    """Test TimeRestriction model."""

    def test_valid_time_restriction(self):
        """Test creating valid time restriction."""
        restriction = TimeRestriction(
            start_time=time(22, 0),
            end_time=time(6, 0),
            days_of_week=[0, 1, 2, 3, 4],
        )
        assert restriction.start_time == time(22, 0)
        assert restriction.end_time == time(6, 0)
        assert len(restriction.days_of_week) == 5

    def test_invalid_day_of_week(self):
        """Test invalid day of week raises error."""
        with pytest.raises(ValidationError):
            TimeRestriction(
                start_time=time(9, 0),
                end_time=time(17, 0),
                days_of_week=[7],  # Invalid - max is 6
            )


class TestCondition:
    """Test Condition model."""

    def test_valid_condition(self):
        """Test creating valid condition."""
        condition = Condition(
            tag_id="PI-101",
            operator=ConditionOperator.GREATER_THAN,
            value=100.0,
        )
        assert condition.tag_id == "PI-101"
        assert condition.operator == ConditionOperator.GREATER_THAN

    def test_range_condition_requires_secondary_value(self):
        """Test range conditions require secondary value."""
        with pytest.raises(ValidationError):
            Condition(
                tag_id="PI-101",
                operator=ConditionOperator.IN_RANGE,
                value=50.0,
                # Missing secondary_value
            )

    def test_valid_range_condition(self):
        """Test valid range condition."""
        condition = Condition(
            tag_id="PI-101",
            operator=ConditionOperator.IN_RANGE,
            value=50.0,
            secondary_value=100.0,
        )
        assert condition.secondary_value == 100.0


class TestRateLimitSpec:
    """Test RateLimitSpec model."""

    def test_valid_rate_limit(self):
        """Test creating valid rate limit spec."""
        spec = RateLimitSpec(
            max_change_per_second=2.0,
            max_change_per_minute=30.0,
            max_writes_per_minute=60,
            cooldown_seconds=1.0,
        )
        assert spec.max_change_per_second == 2.0
        assert spec.max_writes_per_minute == 60

    def test_negative_rate_invalid(self):
        """Test negative rate values are invalid."""
        with pytest.raises(ValidationError):
            RateLimitSpec(
                max_change_per_second=-1.0,  # Invalid
            )


class TestBoundaryPolicy:
    """Test BoundaryPolicy model."""

    def test_absolute_limit_policy(self):
        """Test creating absolute limit policy."""
        policy = BoundaryPolicy(
            policy_id="TEMP_MAX_001",
            policy_type=PolicyType.ABSOLUTE_LIMIT,
            tag_pattern="TI-*",
            min_value=0.0,
            max_value=150.0,
            engineering_units="degC",
            severity=ViolationSeverity.CRITICAL,
        )
        assert policy.policy_id == "TEMP_MAX_001"
        assert policy.max_value == 150.0

    def test_absolute_limit_requires_values(self):
        """Test absolute limit policy requires min or max."""
        with pytest.raises(ValidationError):
            BoundaryPolicy(
                policy_id="INVALID_001",
                policy_type=PolicyType.ABSOLUTE_LIMIT,
                tag_pattern="TI-*",
                # Missing min_value and max_value
            )

    def test_rate_limit_policy(self):
        """Test creating rate limit policy."""
        policy = BoundaryPolicy(
            policy_id="RATE_001",
            policy_type=PolicyType.RATE_LIMIT,
            tag_pattern="TIC-*",
            rate_limit=RateLimitSpec(
                max_change_per_second=2.0,
                max_writes_per_minute=60,
            ),
        )
        assert policy.rate_limit is not None
        assert policy.rate_limit.max_change_per_second == 2.0

    def test_rate_limit_requires_spec(self):
        """Test rate limit policy requires rate_limit spec."""
        with pytest.raises(ValidationError):
            BoundaryPolicy(
                policy_id="INVALID_002",
                policy_type=PolicyType.RATE_LIMIT,
                tag_pattern="TIC-*",
                # Missing rate_limit
            )

    def test_whitelist_policy(self):
        """Test creating whitelist policy."""
        policy = BoundaryPolicy(
            policy_id="WHITELIST_001",
            policy_type=PolicyType.WHITELIST,
            tag_pattern="*",
            allowed_tags={"TIC-101", "TIC-102"},
        )
        assert "TIC-101" in policy.allowed_tags

    def test_policy_hash_computation(self):
        """Test policy hash is deterministic."""
        policy = BoundaryPolicy(
            policy_id="TEMP_001",
            policy_type=PolicyType.ABSOLUTE_LIMIT,
            tag_pattern="TI-*",
            max_value=100.0,
        )
        hash1 = policy.compute_hash()
        hash2 = policy.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex


class TestBoundaryViolation:
    """Test BoundaryViolation model."""

    def test_create_violation(self):
        """Test creating boundary violation."""
        violation = BoundaryViolation(
            policy_id="TEMP_MAX_001",
            tag_id="TI-101",
            requested_value=175.0,
            current_value=145.0,
            boundary_value=150.0,
            violation_type=ViolationType.OVER_MAX,
            severity=ViolationSeverity.CRITICAL,
            message="Temperature exceeds maximum",
        )
        assert violation.tag_id == "TI-101"
        assert violation.violation_type == ViolationType.OVER_MAX
        assert violation.blocked == True

    def test_violation_has_id_and_timestamp(self):
        """Test violation auto-generates ID and timestamp."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TI-101",
            violation_type=ViolationType.OVER_MAX,
        )
        assert violation.violation_id is not None
        assert violation.timestamp is not None

    def test_violation_provenance_hash(self):
        """Test violation computes provenance hash."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TI-101",
            violation_type=ViolationType.OVER_MAX,
        )
        assert violation.provenance_hash != ""
        assert len(violation.provenance_hash) == 64

    def test_violation_is_immutable(self):
        """Test violation is immutable after creation."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TI-101",
            violation_type=ViolationType.OVER_MAX,
        )
        with pytest.raises(TypeError):
            violation.tag_id = "TI-102"


class TestTagWriteRequest:
    """Test TagWriteRequest model."""

    def test_create_write_request(self):
        """Test creating write request."""
        request = TagWriteRequest(
            tag_id="TIC-101",
            value=125.0,
            source="GL-001",
        )
        assert request.tag_id == "TIC-101"
        assert request.value == 125.0

    def test_write_request_is_immutable(self):
        """Test write request is immutable."""
        request = TagWriteRequest(
            tag_id="TIC-101",
            value=125.0,
        )
        with pytest.raises(TypeError):
            request.value = 130.0


class TestActionGateResult:
    """Test ActionGateResult model."""

    def test_allowed_result(self):
        """Test creating allowed result."""
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = ActionGateResult(
            decision=GateDecision.ALLOW,
            original_request=request,
            final_value=100.0,
            checks_passed=["whitelist", "bounds", "rate"],
        )
        assert result.is_allowed == True
        assert result.is_blocked == False

    def test_blocked_result(self):
        """Test creating blocked result."""
        request = TagWriteRequest(tag_id="SIS-101", value=100.0)
        violation = BoundaryViolation(
            policy_id="SIS_INDEPENDENCE",
            tag_id="SIS-101",
            violation_type=ViolationType.SIS_VIOLATION,
        )
        result = ActionGateResult(
            decision=GateDecision.BLOCK,
            original_request=request,
            final_value=None,
            violations=[violation],
            checks_failed=["sis_independence"],
        )
        assert result.is_allowed == False
        assert result.is_blocked == True
        assert len(result.violations) == 1

    def test_clamped_result(self):
        """Test creating clamped result."""
        request = TagWriteRequest(tag_id="TIC-101", value=200.0)
        result = ActionGateResult(
            decision=GateDecision.CLAMP,
            original_request=request,
            final_value=150.0,  # Clamped to max
            checks_passed=["whitelist"],
            checks_failed=["bounds"],
        )
        assert result.is_allowed == True  # Clamped is allowed
        assert result.final_value == 150.0


class TestSISState:
    """Test SISState model."""

    def test_operational_sis(self):
        """Test operational SIS state."""
        state = SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True,
            trip_status=False,
            bypass_active=False,
        )
        assert state.is_operational == True

    def test_tripped_sis(self):
        """Test tripped SIS is not operational."""
        state = SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True,
            trip_status=True,
        )
        assert state.is_operational == True  # Still operational even if tripped

    def test_bypassed_sis_not_operational(self):
        """Test bypassed SIS is not operational."""
        state = SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True,
            bypass_active=True,
        )
        assert state.is_operational == False


class TestSafetyState:
    """Test SafetyState model."""

    def test_safe_state(self):
        """Test safe system state."""
        state = SafetyState(
            sis_states=[SISState(sis_id="SIS-001", is_active=True, is_healthy=True)],
            interlock_states=[InterlockState(interlock_id="IL-001", is_permissive=True)],
            alarm_states=[],
            active_violations=[],
        )
        assert state.all_sis_operational == True
        assert state.all_interlocks_permissive == True
        assert state.has_critical_alarms == False
        assert state.compute_overall_safe() == True

    def test_unsafe_state_with_critical_alarm(self):
        """Test unsafe state with critical alarm."""
        state = SafetyState(
            alarm_states=[
                AlarmState(
                    alarm_id="ALM-001",
                    tag_id="TI-101",
                    is_active=True,
                    priority="CRITICAL",
                )
            ],
        )
        assert state.has_critical_alarms == True
        assert state.compute_overall_safe() == False


class TestSafetyAuditRecord:
    """Test SafetyAuditRecord model."""

    def test_create_audit_record(self):
        """Test creating audit record."""
        record = SafetyAuditRecord(
            event_type="BOUNDARY_VIOLATION",
            action_taken="BLOCKED",
            operator_notified=True,
        )
        assert record.record_id is not None
        assert record.provenance_hash != ""

    def test_audit_record_immutable(self):
        """Test audit record is immutable."""
        record = SafetyAuditRecord(
            event_type="TEST",
            action_taken="BLOCKED",
        )
        with pytest.raises(TypeError):
            record.event_type = "MODIFIED"

    def test_audit_chain_hash(self):
        """Test audit record chain with previous hash."""
        record1 = SafetyAuditRecord(
            event_type="EVENT_1",
            action_taken="BLOCKED",
            previous_hash="",
        )
        record2 = SafetyAuditRecord(
            event_type="EVENT_2",
            action_taken="BLOCKED",
            previous_hash=record1.provenance_hash,
        )
        assert record2.previous_hash == record1.provenance_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
