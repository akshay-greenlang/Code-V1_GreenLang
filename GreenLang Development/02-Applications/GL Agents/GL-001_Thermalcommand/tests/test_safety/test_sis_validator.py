"""
Tests for SIS Independence Validator

Tests to ensure GL-001 NEVER interferes with SIS operations:
- GL-001 NEVER writes to SIS tags
- GL-001 NEVER disables SIS
- GL-001 NEVER bypasses SIS
- SIS state is read-only
"""

import pytest
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from safety.sis_validator import (
    SISIndependenceValidator,
    SISValidationResult,
    SISViolationType,
    SISMonitor,
    SIS_TAG_PATTERNS,
    SIS_CONFIG_PATTERNS,
)
from safety.safety_schemas import (
    SISState,
    ViolationType,
    ViolationSeverity,
)


class TestSISTagPatterns:
    """Test SIS tag pattern definitions."""

    def test_sis_patterns_defined(self):
        """Test SIS patterns are defined."""
        assert len(SIS_TAG_PATTERNS) > 0
        assert "SIS-*" in SIS_TAG_PATTERNS
        assert "ESD-*" in SIS_TAG_PATTERNS

    def test_config_patterns_defined(self):
        """Test SIS config patterns are defined."""
        assert len(SIS_CONFIG_PATTERNS) > 0
        assert "SIS-CFG-*" in SIS_CONFIG_PATTERNS
        assert "SIS-BYPASS-*" in SIS_CONFIG_PATTERNS


class TestSISIndependenceValidator:
    """Test SIS Independence Validator."""

    @pytest.fixture
    def validator(self):
        """Create fresh validator."""
        return SISIndependenceValidator()

    # =========================================================================
    # CRITICAL: SIS Write Prevention Tests
    # =========================================================================

    def test_cannot_write_to_sis_tag(self, validator):
        """CRITICAL: Test GL-001 cannot write to SIS tags."""
        result = validator.validate_tag_access("SIS-101", "write")
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.WRITE_ATTEMPT

    def test_cannot_write_to_esd_tag(self, validator):
        """CRITICAL: Test GL-001 cannot write to ESD tags."""
        result = validator.validate_tag_access("ESD-001", "write")
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.WRITE_ATTEMPT

    def test_cannot_write_to_psv_tag(self, validator):
        """CRITICAL: Test GL-001 cannot write to PSV tags."""
        result = validator.validate_tag_access("PSV-101", "write")
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.WRITE_ATTEMPT

    def test_cannot_write_to_trip_tag(self, validator):
        """CRITICAL: Test GL-001 cannot write to TRIP tags."""
        result = validator.validate_tag_access("TRIP-001", "write")
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.WRITE_ATTEMPT

    def test_cannot_write_to_sis_valve(self, validator):
        """CRITICAL: Test GL-001 cannot write to SIS valves."""
        result = validator.validate_tag_access("XV-SIS-101", "write")
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.WRITE_ATTEMPT

    # =========================================================================
    # SIS Config Prevention Tests
    # =========================================================================

    def test_cannot_access_sis_config(self, validator):
        """CRITICAL: Test GL-001 cannot access SIS configuration."""
        result = validator.validate_tag_access("SIS-CFG-001", "read")
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.CONFIG_CHANGE_ATTEMPT

    def test_cannot_access_sis_bypass_config(self, validator):
        """CRITICAL: Test GL-001 cannot access SIS bypass config."""
        result = validator.validate_tag_access("SIS-BYPASS-001", "write")
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.CONFIG_CHANGE_ATTEMPT

    def test_cannot_access_sis_setpoint(self, validator):
        """CRITICAL: Test GL-001 cannot access SIS setpoints."""
        result = validator.validate_tag_access("SIS-SETPOINT-001", "read")
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.CONFIG_CHANGE_ATTEMPT

    # =========================================================================
    # SIS Read Access Tests (Allowed)
    # =========================================================================

    def test_can_read_sis_status(self, validator):
        """Test GL-001 CAN read SIS status (permissive)."""
        result = validator.validate_tag_access("SIS-101", "read")
        assert result.is_valid == True

    def test_can_read_esd_status(self, validator):
        """Test GL-001 CAN read ESD status."""
        result = validator.validate_tag_access("ESD-001", "read")
        assert result.is_valid == True

    # =========================================================================
    # Non-SIS Tag Tests
    # =========================================================================

    def test_non_sis_tag_allowed(self, validator):
        """Test non-SIS tags are allowed."""
        result = validator.validate_tag_access("TIC-101", "write")
        assert result.is_valid == True

    def test_temperature_tag_allowed(self, validator):
        """Test temperature tags are allowed."""
        result = validator.validate_tag_access("TI-101", "write")
        assert result.is_valid == True


class TestSISTagClassification:
    """Test SIS tag classification methods."""

    @pytest.fixture
    def validator(self):
        """Create fresh validator."""
        return SISIndependenceValidator()

    def test_is_sis_tag(self, validator):
        """Test SIS tag identification."""
        assert validator.is_sis_tag("SIS-101") == True
        assert validator.is_sis_tag("ESD-001") == True
        assert validator.is_sis_tag("PSV-101") == True
        assert validator.is_sis_tag("TRIP-001") == True
        assert validator.is_sis_tag("TIC-101") == False

    def test_is_sis_config_tag(self, validator):
        """Test SIS config tag identification."""
        assert validator.is_sis_config_tag("SIS-CFG-001") == True
        assert validator.is_sis_config_tag("SIS-BYPASS-001") == True
        assert validator.is_sis_config_tag("SIS-101") == False


class TestSISActionValidation:
    """Test action validation for SIS independence."""

    @pytest.fixture
    def validator(self):
        """Create fresh validator."""
        return SISIndependenceValidator()

    def test_cannot_disable_sis(self, validator):
        """CRITICAL: Test GL-001 cannot disable SIS."""
        result = validator.validate_action(
            action_type="disable",
            target="SIS-001",
        )
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.DISABLE_ATTEMPT

    def test_cannot_stop_sis(self, validator):
        """CRITICAL: Test GL-001 cannot stop SIS."""
        result = validator.validate_action(
            action_type="stop",
            target="SIS-001",
        )
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.DISABLE_ATTEMPT

    def test_cannot_bypass_sis(self, validator):
        """CRITICAL: Test GL-001 cannot bypass SIS."""
        result = validator.validate_action(
            action_type="bypass",
            target="SIS-001",
        )
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.BYPASS_ATTEMPT

    def test_cannot_inhibit_sis(self, validator):
        """CRITICAL: Test GL-001 cannot inhibit SIS."""
        result = validator.validate_action(
            action_type="inhibit",
            target="SIS-001",
        )
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.BYPASS_ATTEMPT

    def test_cannot_override_trip(self, validator):
        """CRITICAL: Test GL-001 cannot override SIS trip."""
        result = validator.validate_action(
            action_type="override",
            target="SIS-101",
        )
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.TRIP_OVERRIDE_ATTEMPT

    def test_cannot_clear_trip(self, validator):
        """CRITICAL: Test GL-001 cannot clear SIS trip."""
        result = validator.validate_action(
            action_type="clear_trip",
            target="SIS-101",
        )
        assert result.is_valid == False
        assert result.violation_type == SISViolationType.TRIP_OVERRIDE_ATTEMPT


class TestBatchValidation:
    """Test batch action validation."""

    @pytest.fixture
    def validator(self):
        """Create fresh validator."""
        return SISIndependenceValidator()

    def test_verify_multiple_actions(self, validator):
        """Test verifying multiple actions."""
        actions = [
            {"action_type": "write", "target": "TIC-101"},
            {"action_type": "write", "target": "TIC-102"},
        ]
        all_valid, results = validator.verify_sis_independence(actions)
        assert all_valid == True
        assert len(results) == 2

    def test_batch_fails_with_sis_action(self, validator):
        """Test batch fails if any SIS action included."""
        actions = [
            {"action_type": "write", "target": "TIC-101"},
            {"action_type": "write", "target": "SIS-101"},  # Invalid
        ]
        all_valid, results = validator.verify_sis_independence(actions)
        assert all_valid == False


class TestSISStateMonitoring:
    """Test SIS state monitoring (read-only)."""

    @pytest.fixture
    def validator(self):
        """Create fresh validator."""
        return SISIndependenceValidator()

    def test_update_sis_state(self, validator):
        """Test updating SIS state cache."""
        state = SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True,
        )
        validator.update_sis_state(state)
        cached = validator.get_sis_state("SIS-001")
        assert cached is not None
        assert cached.is_active == True

    def test_get_all_sis_states(self, validator):
        """Test getting all SIS states."""
        validator.update_sis_state(SISState(sis_id="SIS-001", is_active=True, is_healthy=True))
        validator.update_sis_state(SISState(sis_id="SIS-002", is_active=True, is_healthy=True))
        states = validator.get_all_sis_states()
        assert len(states) == 2

    def test_is_any_sis_tripped(self, validator):
        """Test checking if any SIS is tripped."""
        validator.update_sis_state(SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True,
            trip_status=False,
        ))
        assert validator.is_any_sis_tripped() == False

        validator.update_sis_state(SISState(
            sis_id="SIS-002",
            is_active=True,
            is_healthy=True,
            trip_status=True,
        ))
        assert validator.is_any_sis_tripped() == True

    def test_is_any_sis_bypassed(self, validator):
        """Test checking if any SIS is bypassed."""
        validator.update_sis_state(SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True,
            bypass_active=False,
        ))
        assert validator.is_any_sis_bypassed() == False

        validator.update_sis_state(SISState(
            sis_id="SIS-002",
            is_active=True,
            is_healthy=True,
            bypass_active=True,
        ))
        assert validator.is_any_sis_bypassed() == True

    def test_are_all_sis_healthy(self, validator):
        """Test checking if all SIS are healthy."""
        validator.update_sis_state(SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True,
        ))
        validator.update_sis_state(SISState(
            sis_id="SIS-002",
            is_active=True,
            is_healthy=True,
        ))
        assert validator.are_all_sis_healthy() == True

        validator.update_sis_state(SISState(
            sis_id="SIS-002",
            is_active=True,
            is_healthy=False,  # Unhealthy
        ))
        assert validator.are_all_sis_healthy() == False


class TestSISHeartbeat:
    """Test SIS heartbeat monitoring."""

    @pytest.fixture
    def validator(self):
        """Create fresh validator."""
        return SISIndependenceValidator()

    def test_fresh_heartbeat(self, validator):
        """Test fresh heartbeat detection."""
        validator.update_sis_state(SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True,
            last_heartbeat=datetime.utcnow(),
        ))
        results = validator.check_sis_heartbeats(max_age_seconds=60.0)
        assert results["SIS-001"] == True

    def test_stale_heartbeat(self, validator):
        """Test stale heartbeat detection."""
        validator.update_sis_state(SISState(
            sis_id="SIS-001",
            is_active=True,
            is_healthy=True,
            last_heartbeat=datetime.utcnow() - timedelta(minutes=5),
        ))
        results = validator.check_sis_heartbeats(max_age_seconds=60.0)
        assert results["SIS-001"] == False


class TestOptimizationTargetValidation:
    """Test optimization target validation."""

    @pytest.fixture
    def validator(self):
        """Create fresh validator."""
        return SISIndependenceValidator()

    def test_valid_targets_unchanged(self, validator):
        """Test valid targets are unchanged."""
        targets = {
            "TIC-101": 100.0,
            "TIC-102": 110.0,
        }
        valid, removed = validator.validate_optimization_targets(targets)
        assert "TIC-101" in valid
        assert "TIC-102" in valid
        assert len(removed) == 0

    def test_sis_targets_removed(self, validator):
        """Test SIS targets are removed."""
        targets = {
            "TIC-101": 100.0,
            "SIS-101": 50.0,  # Should be removed
        }
        valid, removed = validator.validate_optimization_targets(targets)
        assert "TIC-101" in valid
        assert "SIS-101" not in valid
        assert "SIS-101" in removed


class TestAuditTrail:
    """Test SIS validator audit trail."""

    @pytest.fixture
    def validator(self):
        """Create fresh validator."""
        return SISIndependenceValidator()

    def test_violation_creates_audit_record(self, validator):
        """Test SIS violation creates audit record."""
        # Trigger violation
        validator.validate_tag_access("SIS-101", "write")

        # Check audit records
        records = validator.get_audit_records()
        assert len(records) > 0

    def test_statistics_tracking(self, validator):
        """Test statistics are tracked."""
        # Trigger some validations
        validator.validate_tag_access("TIC-101", "write")  # Pass
        validator.validate_tag_access("SIS-101", "write")  # Fail

        stats = validator.get_statistics()
        assert stats["validations_total"] == 2
        assert stats["validations_passed"] == 1
        assert stats["validations_failed"] == 1


class TestSISMonitorClass:
    """Test SISMonitor class."""

    @pytest.fixture
    def validator(self):
        """Create fresh validator."""
        return SISIndependenceValidator()

    def test_monitor_initialization(self, validator):
        """Test monitor initializes correctly."""
        monitor = SISMonitor(validator)
        assert monitor is not None

    def test_register_callback(self, validator):
        """Test registering callback."""
        monitor = SISMonitor(validator)
        callbacks_received = []

        def callback(event_type, state):
            callbacks_received.append((event_type, state))

        monitor.register_callback(callback)
        # Callbacks registered (would be called when monitoring)


class TestSISViolationCallback:
    """Test violation callback functionality."""

    def test_violation_callback_invoked(self):
        """Test violation callback is invoked."""
        violations_received = []

        def callback(violation):
            violations_received.append(violation)

        validator = SISIndependenceValidator(violation_callback=callback)
        validator.validate_tag_access("SIS-101", "write")

        assert len(violations_received) == 1
        assert violations_received[0].violation_type == ViolationType.SIS_VIOLATION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
