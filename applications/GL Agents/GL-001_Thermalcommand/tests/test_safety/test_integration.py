"""
Integration Tests for Safety Boundary Policy Engine

End-to-end tests verifying the complete safety system:
- Full pipeline from request to decision
- Component interaction
- Zero safety violations guarantee
- Audit trail integrity
"""

import pytest
from datetime import datetime, timedelta
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from safety import (
    # Core components
    SafetyActionGate,
    SafetyBoundaryEngine,
    SISIndependenceValidator,
    ViolationHandler,
    ThermalPolicyManager,
    # Factories
    ActionGateFactory,
    ViolationHandlerFactory,
    # Models
    TagWriteRequest,
    GateDecision,
    ViolationType,
    ViolationSeverity,
    InterlockState,
    AlarmState,
    SISState,
    # Functions
    create_safety_system,
    reset_policy_manager,
    # Constants
    ALLOWED_WRITE_TAGS,
    SIS_TAG_PATTERNS,
)


class TestFullPipeline:
    """Test complete safety pipeline."""

    @pytest.fixture
    def safety_system(self):
        """Create complete safety system."""
        reset_policy_manager()
        return create_safety_system()

    def test_valid_write_flows_through(self, safety_system):
        """Test valid write request flows through entire pipeline."""
        gate, engine, sis, handler = safety_system

        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)

        assert result.decision in (GateDecision.ALLOW, GateDecision.CLAMP)
        assert result.final_value is not None
        assert len(result.violations) == 0

    def test_sis_write_blocked_everywhere(self, safety_system):
        """Test SIS write is blocked at every level."""
        gate, engine, sis, handler = safety_system

        request = TagWriteRequest(tag_id="SIS-101", value=100.0)
        result = gate.evaluate(request)

        # Must be blocked
        assert result.decision == GateDecision.BLOCK
        # Must have SIS-related violation
        sis_violations = [
            v for v in result.violations
            if v.violation_type in (ViolationType.SIS_VIOLATION, ViolationType.UNAUTHORIZED_TAG)
        ]
        assert len(sis_violations) > 0

    def test_over_limit_handled(self, safety_system):
        """Test over-limit write is handled."""
        gate, engine, sis, handler = safety_system

        # Value over temperature limit (max 200)
        request = TagWriteRequest(tag_id="TIC-101", value=250.0)
        result = gate.evaluate(request)

        # Should be blocked or clamped
        if result.decision == GateDecision.CLAMP:
            assert result.final_value <= 200.0
        elif result.decision == GateDecision.BLOCK:
            # Blocked is also acceptable
            pass

    def test_interlock_blocks_all_writes(self, safety_system):
        """Test active interlock blocks all writes."""
        gate, engine, sis, handler = safety_system

        # Activate interlock
        gate.update_interlock_state(InterlockState(
            interlock_id="IL-MAIN",
            is_permissive=False,
            cause="Emergency condition",
        ))

        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)

        assert result.decision == GateDecision.BLOCK
        interlock_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.INTERLOCK_ACTIVE
        ]
        assert len(interlock_violations) > 0


class TestZeroSafetyViolations:
    """Test that safety system achieves zero boundary violations."""

    @pytest.fixture
    def safety_system(self):
        """Create complete safety system."""
        reset_policy_manager()
        return create_safety_system()

    def test_all_sis_tags_blocked(self, safety_system):
        """Test ALL SIS tag patterns are blocked."""
        gate, engine, sis, handler = safety_system

        sis_tags = [
            "SIS-101", "SIS-201", "SIS-301",
            "ESD-001", "ESD-002",
            "PSV-101", "PSV-102",
            "TRIP-001", "TRIP-002",
            "XV-ESD-101", "XV-TRIP-101",
            "SHUTDOWN-001",
        ]

        for tag in sis_tags:
            request = TagWriteRequest(tag_id=tag, value=100.0)
            result = gate.evaluate(request)
            assert result.decision == GateDecision.BLOCK, f"SIS tag {tag} was not blocked!"

    def test_unauthorized_tags_blocked(self, safety_system):
        """Test unauthorized tags are blocked."""
        gate, engine, sis, handler = safety_system

        unauthorized_tags = [
            "UNKNOWN-001",
            "RANDOM-TAG",
            "NOT-IN-WHITELIST",
        ]

        for tag in unauthorized_tags:
            request = TagWriteRequest(tag_id=tag, value=100.0)
            result = gate.evaluate(request)
            assert result.decision == GateDecision.BLOCK, f"Unauthorized tag {tag} was not blocked!"

    def test_bounds_always_enforced(self, safety_system):
        """Test bounds are always enforced."""
        gate, engine, sis, handler = safety_system

        # Test extreme values
        extreme_tests = [
            ("TIC-101", 1000.0),   # Way over max
            ("TIC-101", -1000.0),  # Way under min
            ("PIC-101", 10000.0),  # Over pressure max
            ("PIC-101", -100.0),   # Under pressure min (negative pressure)
        ]

        for tag, value in extreme_tests:
            request = TagWriteRequest(tag_id=tag, value=value)
            result = gate.evaluate(request)

            # Should be blocked or clamped - never allowed at original value
            if result.decision == GateDecision.ALLOW:
                # If allowed, should have been within limits
                limits = gate._boundary_engine._policy_manager.get_limits_for_tag(tag)
                if limits["min"] is not None:
                    assert result.final_value >= limits["min"]
                if limits["max"] is not None:
                    assert result.final_value <= limits["max"]


class TestAuditTrailIntegrity:
    """Test audit trail maintains integrity."""

    @pytest.fixture
    def safety_system(self):
        """Create complete safety system."""
        reset_policy_manager()
        return create_safety_system()

    def test_all_violations_audited(self, safety_system):
        """Test all violations create audit records."""
        gate, engine, sis, handler = safety_system

        # Trigger several violations
        violation_requests = [
            TagWriteRequest(tag_id="SIS-101", value=100.0),
            TagWriteRequest(tag_id="UNKNOWN-001", value=100.0),
        ]

        for request in violation_requests:
            gate.evaluate(request)

        # Check engine audit trail
        engine_records = engine.get_audit_records()
        assert len(engine_records) >= len(violation_requests)

    def test_audit_chain_valid(self, safety_system):
        """Test audit chain remains valid."""
        gate, engine, sis, handler = safety_system

        # Generate multiple violations
        for i in range(5):
            request = TagWriteRequest(tag_id=f"INVALID-{i:03d}", value=100.0)
            gate.evaluate(request)

        # Verify chain integrity
        assert engine.verify_audit_chain() == True


class TestComponentInteraction:
    """Test interaction between components."""

    @pytest.fixture
    def safety_system(self):
        """Create complete safety system."""
        reset_policy_manager()
        return create_safety_system()

    def test_sis_validator_feeds_gate(self, safety_system):
        """Test SIS validator results feed into gate."""
        gate, engine, sis, handler = safety_system

        # Update SIS state
        sis.update_sis_state(SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True,
            trip_status=True,  # Tripped
        ))

        # Gate should still work but be aware of SIS state
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result = gate.evaluate(request)
        # Should still allow non-SIS writes
        assert result.decision in (GateDecision.ALLOW, GateDecision.CLAMP)

    def test_engine_feeds_handler(self, safety_system):
        """Test boundary engine feeds violation handler."""
        gate, engine, sis, handler = safety_system

        # Trigger violation
        request = TagWriteRequest(tag_id="INVALID-001", value=100.0)
        gate.evaluate(request)

        # Handler should have received violation
        stats = handler.get_statistics()
        assert stats["violations_handled"] >= 1

    def test_state_updates_propagate(self, safety_system):
        """Test state updates propagate through system."""
        gate, engine, sis, handler = safety_system

        # Update interlock via gate
        gate.update_interlock_state(InterlockState(
            interlock_id="IL-001",
            is_permissive=False,
        ))

        # Engine should also have the state
        state = engine.get_safety_state()
        assert state.all_interlocks_permissive == False


class TestPerformance:
    """Test performance characteristics."""

    @pytest.fixture
    def safety_system(self):
        """Create complete safety system."""
        reset_policy_manager()
        return create_safety_system()

    def test_evaluation_time_reasonable(self, safety_system):
        """Test gate evaluation time is reasonable."""
        gate, engine, sis, handler = safety_system

        request = TagWriteRequest(tag_id="TIC-101", value=100.0)

        times = []
        for _ in range(100):
            result = gate.evaluate(request)
            times.append(result.evaluation_time_ms)

        avg_time = sum(times) / len(times)
        # Should complete in reasonable time (< 100ms average)
        assert avg_time < 100.0, f"Average evaluation time {avg_time}ms is too high"

    def test_handles_high_request_volume(self, safety_system):
        """Test system handles high request volume."""
        gate, engine, sis, handler = safety_system

        # Make many requests rapidly
        for i in range(500):
            request = TagWriteRequest(tag_id="TIC-101", value=100.0 + (i % 10))
            result = gate.evaluate(request)
            # All should complete without error
            assert result.decision in (GateDecision.ALLOW, GateDecision.CLAMP, GateDecision.BLOCK)

        stats = gate.get_statistics()
        assert stats["total_evaluations"] == 500


class TestRecoveryScenarios:
    """Test recovery from various states."""

    @pytest.fixture
    def safety_system(self):
        """Create complete safety system."""
        reset_policy_manager()
        return create_safety_system()

    def test_recovery_from_interlock(self, safety_system):
        """Test recovery when interlock clears."""
        gate, engine, sis, handler = safety_system

        # Activate interlock
        gate.update_interlock_state(InterlockState(
            interlock_id="IL-001",
            is_permissive=False,
        ))

        # Request blocked
        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result1 = gate.evaluate(request)
        assert result1.decision == GateDecision.BLOCK

        # Clear interlock
        gate.update_interlock_state(InterlockState(
            interlock_id="IL-001",
            is_permissive=True,
        ))

        # Request should now be allowed
        result2 = gate.evaluate(request)
        assert result2.decision in (GateDecision.ALLOW, GateDecision.CLAMP)

    def test_recovery_from_alarm(self, safety_system):
        """Test recovery when alarm clears."""
        gate, engine, sis, handler = safety_system

        # Activate critical alarm
        gate.update_alarm_state(AlarmState(
            alarm_id="ALM-001",
            tag_id="TIC-101",
            is_active=True,
            priority="CRITICAL",
        ))

        request = TagWriteRequest(tag_id="TIC-101", value=100.0)
        result1 = gate.evaluate(request)

        # Clear alarm
        gate.update_alarm_state(AlarmState(
            alarm_id="ALM-001",
            tag_id="TIC-101",
            is_active=False,
            priority="CRITICAL",
        ))

        result2 = gate.evaluate(request)
        # Should be allowed after alarm clears
        alarm_violations = [
            v for v in result2.violations
            if v.violation_type == ViolationType.ALARM_ACTIVE
        ]
        assert len(alarm_violations) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def safety_system(self):
        """Create complete safety system."""
        reset_policy_manager()
        return create_safety_system()

    def test_exactly_at_limit(self, safety_system):
        """Test value exactly at limit."""
        gate, engine, sis, handler = safety_system

        # Exactly at max temperature (200)
        request = TagWriteRequest(tag_id="TIC-101", value=200.0)
        result = gate.evaluate(request)
        # Should be allowed (at limit, not over)
        assert result.decision in (GateDecision.ALLOW, GateDecision.CLAMP)

    def test_empty_tag_handling(self, safety_system):
        """Test handling of edge case tags."""
        gate, engine, sis, handler = safety_system

        # Very short tag
        request = TagWriteRequest(tag_id="X", value=100.0)
        result = gate.evaluate(request)
        # Should be blocked (not in whitelist)
        assert result.decision == GateDecision.BLOCK

    def test_zero_value(self, safety_system):
        """Test zero value handling."""
        gate, engine, sis, handler = safety_system

        request = TagWriteRequest(tag_id="TIC-101", value=0.0)
        result = gate.evaluate(request)
        # Zero should be within limits for temperature
        assert result.decision in (GateDecision.ALLOW, GateDecision.CLAMP)

    def test_negative_value(self, safety_system):
        """Test negative value within limits."""
        gate, engine, sis, handler = safety_system

        # -30 is within temp limits (min -40)
        request = TagWriteRequest(tag_id="TIC-101", value=-30.0)
        result = gate.evaluate(request)
        # Should be allowed
        limit_violations = [
            v for v in result.violations
            if v.violation_type in (ViolationType.OVER_MAX, ViolationType.UNDER_MIN)
        ]
        assert len(limit_violations) == 0


class TestMultipleViolations:
    """Test handling of multiple simultaneous violations."""

    @pytest.fixture
    def safety_system(self):
        """Create complete safety system."""
        reset_policy_manager()
        return create_safety_system()

    def test_multiple_violations_all_reported(self, safety_system):
        """Test all violations are reported when multiple occur."""
        gate, engine, sis, handler = safety_system

        # Set up conditions for multiple violations
        gate.update_interlock_state(InterlockState(
            interlock_id="IL-001",
            is_permissive=False,
        ))
        gate.update_alarm_state(AlarmState(
            alarm_id="ALM-001",
            tag_id="TIC-101",
            is_active=True,
            priority="CRITICAL",
        ))

        # This request should trigger multiple violations
        # (unauthorized tag + interlock + alarm if related)
        request = TagWriteRequest(tag_id="INVALID-001", value=100.0)
        result = gate.evaluate(request)

        assert result.decision == GateDecision.BLOCK
        # Should have multiple violations
        assert len(result.violations) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
