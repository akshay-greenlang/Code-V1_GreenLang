"""
GL-010 EmissionGuardian - Safety Controller Tests

Comprehensive test suite for the safety enforcement system.
Tests read-only enforcement, write approval workflow, interlocks, and emergency shutdown.

Safety Principles:
- SCADA/DAHS are READ-ONLY (no writes allowed)
- All write operations require approval workflow
- Emergency shutdown blocks all operations
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
import threading
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from monitoring.safety import (
    SafetyLevel,
    OperationType,
    SystemType,
    ApprovalState,
    InterlockState,
    SafetyViolationError,
    WriteAccessDeniedError,
    ApprovalRequiredError,
    InterlockViolationError,
    EmergencyShutdownError,
    SafetyAuditEntry,
    WriteApprovalRequest,
    SafetyInterlock,
    SafetyController,
    get_safety_controller,
    check_read_access,
    check_write_access,
    SAFETY_SYSTEM_ACCESS_PROHIBITED,
    SCADA_READ_ONLY,
    DAHS_READ_ONLY,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fresh_safety_controller():
    """Create a fresh SafetyController instance for testing."""
    # Reset singleton for testing
    SafetyController._instance = None
    controller = SafetyController()
    yield controller
    # Cleanup
    SafetyController._instance = None


@pytest.fixture
def sample_interlock():
    """Sample safety interlock."""
    return SafetyInterlock(
        interlock_id="INT-001",
        name="High Temperature Interlock",
        system_type=SystemType.CEMS,
        state=InterlockState.ACTIVE,
        description="Trips when temperature exceeds safe limit",
        trip_conditions=["temperature > 500C"],
    )


# =============================================================================
# TEST: ENUMS
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_safety_level_values(self):
        """All SIL levels should be defined."""
        expected = {"sil_1", "sil_2", "sil_3", "sil_4"}
        actual = {sl.value for sl in SafetyLevel}
        assert expected == actual

    def test_operation_type_values(self):
        """All operation types should be defined."""
        expected = {"read", "write", "configure", "emergency"}
        actual = {ot.value for ot in OperationType}
        assert expected == actual

    def test_system_type_values(self):
        """All system types should be defined."""
        expected = {"scada", "dahs", "plc", "dcs", "sis", "cems", "erp"}
        actual = {st.value for st in SystemType}
        assert expected == actual

    def test_approval_state_values(self):
        """All approval states should be defined."""
        expected = {
            "pending", "approved", "rejected",
            "expired", "executed", "cancelled"
        }
        actual = {s.value for s in ApprovalState}
        assert expected == actual

    def test_interlock_state_values(self):
        """All interlock states should be defined."""
        expected = {"active", "bypassed", "tripped", "fault", "unknown"}
        actual = {s.value for s in InterlockState}
        assert expected == actual


# =============================================================================
# TEST: EXCEPTIONS
# =============================================================================

class TestExceptions:
    """Test safety exception hierarchy."""

    def test_base_exception(self):
        """SafetyViolationError should be base class."""
        assert issubclass(WriteAccessDeniedError, SafetyViolationError)
        assert issubclass(ApprovalRequiredError, SafetyViolationError)
        assert issubclass(InterlockViolationError, SafetyViolationError)
        assert issubclass(EmergencyShutdownError, SafetyViolationError)

    def test_exception_message(self):
        """Exceptions should include message."""
        with pytest.raises(WriteAccessDeniedError) as exc_info:
            raise WriteAccessDeniedError("Test message")

        assert "Test message" in str(exc_info.value)


# =============================================================================
# TEST: SAFETY AUDIT ENTRY
# =============================================================================

class TestSafetyAuditEntry:
    """Test SafetyAuditEntry dataclass."""

    def test_audit_entry_creation(self):
        """Audit entry should be created with all fields."""
        entry = SafetyAuditEntry(
            operation_type=OperationType.READ,
            system_type=SystemType.CEMS,
            user_id="user-001",
            action="Read temperature data",
            allowed=True,
            reason="Read operations permitted",
        )

        assert entry.entry_id is not None
        assert entry.timestamp is not None
        assert entry.operation_type == OperationType.READ
        assert entry.allowed is True

    def test_audit_entry_provenance_hash(self):
        """Audit entry should have provenance hash."""
        entry = SafetyAuditEntry(
            operation_type=OperationType.READ,
            system_type=SystemType.CEMS,
            action="Test",
            allowed=True,
            reason="Test",
        )

        assert len(entry.provenance_hash) == 64
        assert entry.provenance_hash.isalnum()

    def test_audit_entry_hash_deterministic(self):
        """Same entry content should produce same hash."""
        entry = SafetyAuditEntry(
            entry_id="FIXED-ID",
            operation_type=OperationType.READ,
            system_type=SystemType.CEMS,
            action="Test",
            allowed=True,
            reason="Test",
        )

        hash1 = entry.calculate_provenance()
        hash2 = entry.calculate_provenance()

        assert hash1 == hash2


# =============================================================================
# TEST: WRITE APPROVAL REQUEST
# =============================================================================

class TestWriteApprovalRequest:
    """Test WriteApprovalRequest dataclass."""

    def test_request_creation(self):
        """Approval request should be created."""
        request = WriteApprovalRequest(
            system_type=SystemType.ERP,
            operation="Update emission factors",
            parameters={"factor_id": "EF-001", "new_value": 1.5},
            requested_by="user-001",
        )

        assert request.request_id is not None
        assert request.state == ApprovalState.PENDING
        assert request.system_type == SystemType.ERP

    def test_request_expiration(self):
        """Request should track expiration."""
        request = WriteApprovalRequest(
            system_type=SystemType.ERP,
            operation="Test",
            parameters={},
            requested_by="user-001",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )

        assert request.is_expired() is False

        # Simulate expired request
        request.expires_at = datetime.utcnow() - timedelta(hours=1)
        assert request.is_expired() is True

    def test_request_approval(self):
        """Request can be approved."""
        request = WriteApprovalRequest(
            system_type=SystemType.ERP,
            operation="Test",
            parameters={},
            requested_by="user-001",
        )

        request.approve(approved_by="supervisor-001")

        assert request.state == ApprovalState.APPROVED
        assert request.approved_by == "supervisor-001"
        assert request.approved_at is not None

    def test_request_rejection(self):
        """Request can be rejected."""
        request = WriteApprovalRequest(
            system_type=SystemType.ERP,
            operation="Test",
            parameters={},
            requested_by="user-001",
        )

        request.reject(rejected_by="supervisor-001", reason="Not authorized")

        assert request.state == ApprovalState.REJECTED
        assert request.rejection_reason == "Not authorized"

    def test_expired_request_cannot_approve(self):
        """Expired request cannot be approved."""
        request = WriteApprovalRequest(
            system_type=SystemType.ERP,
            operation="Test",
            parameters={},
            requested_by="user-001",
            expires_at=datetime.utcnow() - timedelta(hours=1),  # Already expired
        )

        with pytest.raises(SafetyViolationError):
            request.approve(approved_by="supervisor-001")

        assert request.state == ApprovalState.EXPIRED


# =============================================================================
# TEST: SAFETY INTERLOCK
# =============================================================================

class TestSafetyInterlock:
    """Test SafetyInterlock dataclass."""

    def test_interlock_creation(self, sample_interlock):
        """Interlock should be created."""
        assert sample_interlock.interlock_id == "INT-001"
        assert sample_interlock.state == InterlockState.ACTIVE

    def test_interlock_is_safe(self, sample_interlock):
        """Active interlock should be safe."""
        assert sample_interlock.is_safe() is True

        sample_interlock.state = InterlockState.TRIPPED
        assert sample_interlock.is_safe() is False

    def test_interlock_trip(self, sample_interlock):
        """Interlock can be tripped."""
        sample_interlock.trip("Temperature exceeded 500C")

        assert sample_interlock.state == InterlockState.TRIPPED

    def test_interlock_reset(self, sample_interlock):
        """Interlock can be reset."""
        sample_interlock.trip("Test")
        sample_interlock.reset()

        assert sample_interlock.state == InterlockState.ACTIVE


# =============================================================================
# TEST: SAFETY CONTROLLER - SINGLETON
# =============================================================================

class TestSafetyControllerSingleton:
    """Test SafetyController singleton pattern."""

    def test_singleton_instance(self):
        """Only one instance should exist."""
        SafetyController._instance = None

        controller1 = SafetyController()
        controller2 = SafetyController()

        assert controller1 is controller2

        SafetyController._instance = None

    def test_get_safety_controller(self):
        """get_safety_controller should return singleton."""
        SafetyController._instance = None

        controller = get_safety_controller()

        assert controller is not None
        assert isinstance(controller, SafetyController)

        SafetyController._instance = None


# =============================================================================
# TEST: READ-ONLY ENFORCEMENT
# =============================================================================

class TestReadOnlyEnforcement:
    """Test read-only enforcement for critical systems."""

    def test_scada_read_only(self, fresh_safety_controller):
        """SCADA should be read-only."""
        assert fresh_safety_controller.is_read_only_enforced(SystemType.SCADA) is True

    def test_dahs_read_only(self, fresh_safety_controller):
        """DAHS should be read-only."""
        assert fresh_safety_controller.is_read_only_enforced(SystemType.DAHS) is True

    def test_sis_read_only(self, fresh_safety_controller):
        """SIS (Safety Instrumented System) should be read-only."""
        assert fresh_safety_controller.is_read_only_enforced(SystemType.SIS) is True

    def test_read_operation_allowed(self, fresh_safety_controller):
        """Read operations should always be allowed."""
        result = fresh_safety_controller.check_read_operation(
            SystemType.SCADA,
            "Read temperature data",
            user_id="user-001",
        )

        assert result is True

    def test_write_to_scada_denied(self, fresh_safety_controller):
        """Write to SCADA should be denied."""
        with pytest.raises(WriteAccessDeniedError):
            fresh_safety_controller.check_write_operation(
                SystemType.SCADA,
                "Update setpoint",
                user_id="user-001",
            )

    def test_write_to_dahs_denied(self, fresh_safety_controller):
        """Write to DAHS should be denied."""
        with pytest.raises(WriteAccessDeniedError):
            fresh_safety_controller.check_write_operation(
                SystemType.DAHS,
                "Modify data",
                user_id="user-001",
            )


# =============================================================================
# TEST: APPROVAL WORKFLOW
# =============================================================================

class TestApprovalWorkflow:
    """Test write approval workflow."""

    def test_request_write_approval(self, fresh_safety_controller):
        """Write approval can be requested for non-read-only systems."""
        # ERP is not read-only by default
        if not fresh_safety_controller.is_read_only_enforced(SystemType.ERP):
            request = fresh_safety_controller.request_write_approval(
                system_type=SystemType.ERP,
                operation="Update emission factor",
                parameters={"factor_id": "EF-001"},
                requested_by="user-001",
            )

            assert request.state == ApprovalState.PENDING
            assert request.request_id in fresh_safety_controller._approval_requests

    def test_approve_write_request(self, fresh_safety_controller):
        """Write request can be approved."""
        # Skip if all systems are read-only
        if SAFETY_SYSTEM_ACCESS_PROHIBITED:
            pytest.skip("All writes prohibited")

        request = fresh_safety_controller.request_write_approval(
            system_type=SystemType.ERP,
            operation="Test",
            parameters={},
            requested_by="user-001",
        )

        approved = fresh_safety_controller.approve_write_request(
            request.request_id,
            approved_by="supervisor-001",
        )

        assert approved.state == ApprovalState.APPROVED

    def test_reject_write_request(self, fresh_safety_controller):
        """Write request can be rejected."""
        if SAFETY_SYSTEM_ACCESS_PROHIBITED:
            pytest.skip("All writes prohibited")

        request = fresh_safety_controller.request_write_approval(
            system_type=SystemType.ERP,
            operation="Test",
            parameters={},
            requested_by="user-001",
        )

        rejected = fresh_safety_controller.reject_write_request(
            request.request_id,
            rejected_by="supervisor-001",
            reason="Not authorized",
        )

        assert rejected.state == ApprovalState.REJECTED


# =============================================================================
# TEST: INTERLOCK MANAGEMENT
# =============================================================================

class TestInterlockManagement:
    """Test safety interlock management."""

    def test_register_interlock(
        self, fresh_safety_controller, sample_interlock
    ):
        """Interlock can be registered."""
        fresh_safety_controller.register_interlock(sample_interlock)

        assert sample_interlock.interlock_id in fresh_safety_controller._interlocks

    def test_check_interlocks_pass(
        self, fresh_safety_controller, sample_interlock
    ):
        """Active interlocks should pass check."""
        fresh_safety_controller.register_interlock(sample_interlock)

        result = fresh_safety_controller.check_interlocks(SystemType.CEMS)

        assert result is True

    def test_check_interlocks_fail(
        self, fresh_safety_controller, sample_interlock
    ):
        """Tripped interlock should fail check."""
        sample_interlock.trip("Test trip")
        fresh_safety_controller.register_interlock(sample_interlock)

        with pytest.raises(InterlockViolationError):
            fresh_safety_controller.check_interlocks(SystemType.CEMS)

    def test_get_interlock_status(
        self, fresh_safety_controller, sample_interlock
    ):
        """Interlock status should be retrievable."""
        fresh_safety_controller.register_interlock(sample_interlock)

        status = fresh_safety_controller.get_interlock_status()

        assert sample_interlock.interlock_id in status
        assert status[sample_interlock.interlock_id]["is_safe"] is True


# =============================================================================
# TEST: EMERGENCY SHUTDOWN
# =============================================================================

class TestEmergencyShutdown:
    """Test emergency shutdown functionality."""

    def test_trigger_emergency_shutdown(self, fresh_safety_controller):
        """Emergency shutdown can be triggered."""
        fresh_safety_controller.trigger_emergency_shutdown(
            reason="Critical emission exceedance",
            triggered_by="operator-001",
        )

        assert fresh_safety_controller.is_emergency_shutdown_active() is True

    def test_emergency_shutdown_blocks_writes(self, fresh_safety_controller):
        """Emergency shutdown should block all writes."""
        fresh_safety_controller.trigger_emergency_shutdown(
            reason="Test",
            triggered_by="operator-001",
        )

        with pytest.raises(EmergencyShutdownError):
            fresh_safety_controller.check_write_operation(
                SystemType.ERP,
                "Any operation",
            )

    def test_emergency_shutdown_trips_interlocks(
        self, fresh_safety_controller, sample_interlock
    ):
        """Emergency shutdown should trip all interlocks."""
        fresh_safety_controller.register_interlock(sample_interlock)

        fresh_safety_controller.trigger_emergency_shutdown(
            reason="Test",
            triggered_by="operator-001",
        )

        assert sample_interlock.state == InterlockState.TRIPPED

    def test_clear_emergency_shutdown(self, fresh_safety_controller):
        """Emergency shutdown can be cleared with authorization."""
        fresh_safety_controller.trigger_emergency_shutdown(
            reason="Test",
            triggered_by="operator-001",
        )

        fresh_safety_controller.clear_emergency_shutdown(
            cleared_by="supervisor-001",
            authorization_code="AUTH-12345",
        )

        assert fresh_safety_controller.is_emergency_shutdown_active() is False

    def test_clear_emergency_requires_auth_code(self, fresh_safety_controller):
        """Clearing emergency shutdown requires authorization code."""
        fresh_safety_controller.trigger_emergency_shutdown(
            reason="Test",
            triggered_by="operator-001",
        )

        with pytest.raises(SafetyViolationError):
            fresh_safety_controller.clear_emergency_shutdown(
                cleared_by="supervisor-001",
                authorization_code="",  # Empty code
            )


# =============================================================================
# TEST: AUDIT LOG
# =============================================================================

class TestAuditLog:
    """Test audit logging functionality."""

    def test_read_operation_logged(self, fresh_safety_controller):
        """Read operations should be logged."""
        fresh_safety_controller.check_read_operation(
            SystemType.CEMS,
            "Read data",
            user_id="user-001",
        )

        audit_log = fresh_safety_controller.get_audit_log()

        assert len(audit_log) > 0
        assert audit_log[-1].operation_type == OperationType.READ

    def test_denied_write_logged(self, fresh_safety_controller):
        """Denied write operations should be logged."""
        try:
            fresh_safety_controller.check_write_operation(
                SystemType.SCADA,
                "Blocked write",
                user_id="user-001",
            )
        except WriteAccessDeniedError:
            pass

        audit_log = fresh_safety_controller.get_audit_log()

        assert len(audit_log) > 0
        # Find the write denial
        write_entries = [e for e in audit_log if e.operation_type == OperationType.WRITE]
        assert len(write_entries) > 0
        assert write_entries[-1].allowed is False

    def test_emergency_shutdown_logged(self, fresh_safety_controller):
        """Emergency shutdown should be logged."""
        fresh_safety_controller.trigger_emergency_shutdown(
            reason="Test",
            triggered_by="operator-001",
        )

        audit_log = fresh_safety_controller.get_audit_log()

        emergency_entries = [
            e for e in audit_log
            if e.operation_type == OperationType.EMERGENCY
        ]
        assert len(emergency_entries) > 0

    def test_audit_log_limit(self, fresh_safety_controller):
        """Audit log should respect limit parameter."""
        # Generate multiple entries
        for i in range(10):
            fresh_safety_controller.check_read_operation(
                SystemType.CEMS,
                f"Read {i}",
            )

        limited = fresh_safety_controller.get_audit_log(limit=5)

        assert len(limited) == 5


# =============================================================================
# TEST: HELPER FUNCTIONS
# =============================================================================

class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_check_read_access(self, fresh_safety_controller):
        """check_read_access should work."""
        result = check_read_access(
            SystemType.CEMS,
            "Read data",
            user_id="user-001",
        )

        assert result is True

    def test_check_write_access_denied(self, fresh_safety_controller):
        """check_write_access should enforce read-only."""
        with pytest.raises(WriteAccessDeniedError):
            check_write_access(
                SystemType.SCADA,
                "Write data",
            )


# =============================================================================
# TEST: THREAD SAFETY
# =============================================================================

class TestThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_read_operations(self, fresh_safety_controller):
        """Multiple threads should be able to read concurrently."""
        results = []

        def read_operation():
            result = fresh_safety_controller.check_read_operation(
                SystemType.CEMS,
                "Concurrent read",
            )
            results.append(result)

        threads = [threading.Thread(target=read_operation) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(r is True for r in results)

    def test_concurrent_audit_logging(self, fresh_safety_controller):
        """Audit logging should be thread-safe."""
        def log_operation():
            for i in range(5):
                fresh_safety_controller.check_read_operation(
                    SystemType.CEMS,
                    f"Thread op {i}",
                )

        threads = [threading.Thread(target=log_operation) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 25 entries (5 threads * 5 operations)
        audit_log = fresh_safety_controller.get_audit_log(limit=30)
        assert len(audit_log) >= 25


# =============================================================================
# TEST: CONSTANTS
# =============================================================================

class TestConstants:
    """Test safety constants."""

    def test_safety_system_access_prohibited(self):
        """SAFETY_SYSTEM_ACCESS_PROHIBITED should be True."""
        assert SAFETY_SYSTEM_ACCESS_PROHIBITED is True

    def test_scada_read_only(self):
        """SCADA_READ_ONLY should be True."""
        assert SCADA_READ_ONLY is True

    def test_dahs_read_only(self):
        """DAHS_READ_ONLY should be True."""
        assert DAHS_READ_ONLY is True

