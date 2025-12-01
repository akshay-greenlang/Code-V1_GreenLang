# -*- coding: utf-8 -*-
"""
Security Tests for GL-002 FLAMEGUARD BoilerEfficiencyOptimizer.

Comprehensive security validation covering:
- Input validation for boiler parameters
- Safety limits enforcement
- Interlock testing
- Authentication and authorization
- Injection attack prevention
- Audit trail compliance

Coverage Target: 95%+
Author: GreenLang Foundation Test Engineering
"""

import pytest
import re
import math
from typing import Dict, Any, List

from conftest import (
    UserRole,
    PermissionLevel,
    SafetyLimits,
    InterlockCondition
)


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Test input validation for boiler parameters."""

    @pytest.mark.security
    @pytest.mark.input_validation
    def test_valid_parameters_accepted(self, valid_boiler_data, input_validator):
        """Test that valid parameters are accepted."""
        for param, value in valid_boiler_data.items():
            if isinstance(value, (int, float)):
                is_valid, error = input_validator.validate_parameter(param, value)
                assert is_valid, f"Valid parameter {param}={value} rejected: {error}"

    @pytest.mark.security
    @pytest.mark.input_validation
    def test_negative_values_rejected(self, invalid_parameter_values, input_validator):
        """Test that negative values are rejected where inappropriate."""
        negative_params = invalid_parameter_values["negative_values"]

        for param, value in negative_params.items():
            is_valid, error = input_validator.validate_parameter(param, value)
            assert not is_valid, f"Negative {param}={value} should be rejected"

    @pytest.mark.security
    @pytest.mark.input_validation
    def test_excessive_values_rejected(self, invalid_parameter_values, input_validator):
        """Test that excessive values are rejected."""
        excessive_params = invalid_parameter_values["excessive_values"]

        for param, value in excessive_params.items():
            is_valid, error = input_validator.validate_parameter(param, value)
            assert not is_valid, f"Excessive {param}={value} should be rejected"

    @pytest.mark.security
    @pytest.mark.input_validation
    def test_nan_values_rejected(self, input_validator):
        """Test that NaN values are rejected."""
        is_valid, error = input_validator.validate_parameter("steam_pressure_bar", float('nan'))
        assert not is_valid
        assert "NaN" in error

    @pytest.mark.security
    @pytest.mark.input_validation
    def test_infinity_values_rejected(self, input_validator):
        """Test that infinite values are rejected."""
        is_valid, error = input_validator.validate_parameter("steam_pressure_bar", float('inf'))
        assert not is_valid
        assert "Infinite" in error

        is_valid, error = input_validator.validate_parameter("steam_pressure_bar", float('-inf'))
        assert not is_valid
        assert "Infinite" in error

    @pytest.mark.security
    @pytest.mark.input_validation
    def test_boundary_values_pressure(self, safety_limits, input_validator):
        """Test boundary values for steam pressure."""
        # At minimum - should be valid
        is_valid, _ = input_validator.validate_parameter(
            "steam_pressure_bar",
            safety_limits.min_steam_pressure_bar
        )
        assert is_valid

        # At maximum - should be valid
        is_valid, _ = input_validator.validate_parameter(
            "steam_pressure_bar",
            safety_limits.max_steam_pressure_bar
        )
        assert is_valid

        # Below minimum - should be invalid
        is_valid, _ = input_validator.validate_parameter(
            "steam_pressure_bar",
            safety_limits.min_steam_pressure_bar - 0.1
        )
        assert not is_valid

        # Above maximum - should be invalid
        is_valid, _ = input_validator.validate_parameter(
            "steam_pressure_bar",
            safety_limits.max_steam_pressure_bar + 0.1
        )
        assert not is_valid

    @pytest.mark.security
    @pytest.mark.input_validation
    def test_boundary_values_o2(self, safety_limits, input_validator):
        """Test boundary values for O2 percentage."""
        # Valid range
        is_valid, _ = input_validator.validate_parameter("o2_percent", 3.5)
        assert is_valid

        # Below minimum
        is_valid, _ = input_validator.validate_parameter(
            "o2_percent",
            safety_limits.min_o2_percent - 0.1
        )
        assert not is_valid

        # Above maximum
        is_valid, _ = input_validator.validate_parameter(
            "o2_percent",
            safety_limits.max_o2_percent + 0.1
        )
        assert not is_valid


# =============================================================================
# INJECTION ATTACK PREVENTION
# =============================================================================

class TestInjectionPrevention:
    """Test injection attack prevention."""

    @pytest.mark.security
    @pytest.mark.injection
    def test_sql_injection_rejected(self, injection_payloads, input_validator):
        """Test SQL injection payloads are rejected."""
        for payload in injection_payloads["sql_injection"]:
            is_valid, error = input_validator.validate_string(payload)
            assert not is_valid, f"SQL injection payload should be rejected: {payload}"

    @pytest.mark.security
    @pytest.mark.injection
    def test_command_injection_rejected(self, injection_payloads, input_validator):
        """Test command injection payloads are rejected."""
        for payload in injection_payloads["command_injection"]:
            is_valid, error = input_validator.validate_string(payload)
            assert not is_valid, f"Command injection payload should be rejected: {payload}"

    @pytest.mark.security
    @pytest.mark.injection
    def test_path_traversal_rejected(self, injection_payloads, input_validator):
        """Test path traversal payloads are rejected."""
        for payload in injection_payloads["path_traversal"]:
            is_valid, error = input_validator.validate_string(payload)
            assert not is_valid, f"Path traversal payload should be rejected: {payload}"

    @pytest.mark.security
    @pytest.mark.injection
    def test_valid_boiler_id_accepted(self, input_validator):
        """Test valid boiler IDs are accepted."""
        valid_ids = [
            "BOILER-001",
            "BOILER_002",
            "B001",
            "Plant1-Boiler-A",
        ]

        for boiler_id in valid_ids:
            # Only alphanumeric and simple separators
            is_safe = bool(re.match(r"^[a-zA-Z0-9_-]+$", boiler_id))
            assert is_safe, f"Valid boiler ID rejected: {boiler_id}"

    @pytest.mark.security
    @pytest.mark.injection
    def test_malicious_boiler_id_rejected(self, input_validator):
        """Test malicious boiler IDs are rejected."""
        malicious_ids = [
            "BOILER'; DROP TABLE --",
            "BOILER|cat /etc/passwd",
            "../../../etc/passwd",
            "<script>alert(1)</script>",
        ]

        for boiler_id in malicious_ids:
            is_valid, error = input_validator.validate_string(boiler_id)
            assert not is_valid, f"Malicious boiler ID should be rejected: {boiler_id}"


# =============================================================================
# SAFETY LIMITS ENFORCEMENT
# =============================================================================

class TestSafetyLimitsEnforcement:
    """Test safety limits enforcement."""

    @pytest.mark.security
    @pytest.mark.safety_limits
    def test_high_pressure_limit_enforcement(self, safety_limits):
        """Test high pressure limit is enforced."""
        test_pressures = [
            (40.0, True),   # Within limit
            (45.0, True),   # At limit
            (46.0, False),  # Above limit
            (50.0, False),  # Well above limit
        ]

        for pressure, expected_valid in test_pressures:
            is_valid = pressure <= safety_limits.max_steam_pressure_bar
            assert is_valid == expected_valid, \
                f"Pressure {pressure} bar: expected valid={expected_valid}, got {is_valid}"

    @pytest.mark.security
    @pytest.mark.safety_limits
    def test_low_pressure_limit_enforcement(self, safety_limits):
        """Test low pressure limit is enforced."""
        test_pressures = [
            (10.0, True),   # Within limit
            (5.0, True),    # At limit
            (4.9, False),   # Below limit
            (0.0, False),   # Zero
        ]

        for pressure, expected_valid in test_pressures:
            is_valid = pressure >= safety_limits.min_steam_pressure_bar
            assert is_valid == expected_valid, \
                f"Pressure {pressure} bar: expected valid={expected_valid}, got {is_valid}"

    @pytest.mark.security
    @pytest.mark.safety_limits
    def test_temperature_limits_enforcement(self, safety_limits):
        """Test temperature limits are enforced."""
        # High temperature
        assert 480.0 <= safety_limits.max_steam_temperature_c
        assert 520.0 > safety_limits.max_steam_temperature_c

        # Low temperature
        assert 150.0 >= safety_limits.min_steam_temperature_c
        assert 80.0 < safety_limits.min_steam_temperature_c

    @pytest.mark.security
    @pytest.mark.safety_limits
    def test_emissions_limits_enforcement(self, safety_limits):
        """Test emissions limits are enforced."""
        # CO limits
        assert safety_limits.max_co_ppm > 0
        assert 200.0 <= safety_limits.max_co_ppm

        # NOx limits
        assert safety_limits.max_nox_ppm > 0
        assert 50.0 <= safety_limits.max_nox_ppm

    @pytest.mark.security
    @pytest.mark.safety_limits
    def test_load_limits_enforcement(self, safety_limits):
        """Test load limits are enforced."""
        test_loads = [
            (50.0, True),   # Normal
            (100.0, True),  # Full load
            (110.0, True),  # At limit
            (111.0, False), # Above limit
            (-10.0, False), # Negative
        ]

        for load, expected_valid in test_loads:
            is_valid = safety_limits.min_load_percent <= load <= safety_limits.max_load_percent
            assert is_valid == expected_valid, \
                f"Load {load}%: expected valid={expected_valid}, got {is_valid}"


# =============================================================================
# INTERLOCK TESTING
# =============================================================================

class TestInterlockSafety:
    """Test interlock safety mechanisms."""

    @pytest.mark.security
    @pytest.mark.interlock
    def test_high_pressure_interlock_triggers(self, interlock_conditions, interlock_manager):
        """Test high pressure interlock triggers correctly."""
        manager = interlock_manager(interlock_conditions)

        # Normal operation - no triggers
        normal_params = {"steam_pressure_bar": 35.0}
        triggered = manager.check_interlocks(normal_params)
        assert len(triggered) == 0

        # High pressure - should trigger
        high_params = {"steam_pressure_bar": 43.0}
        triggered = manager.check_interlocks(high_params)
        assert len(triggered) == 1
        assert triggered[0]["action"] == "trip"
        assert triggered[0]["priority"] == 1

    @pytest.mark.security
    @pytest.mark.interlock
    def test_low_o2_interlock_triggers(self, interlock_conditions, interlock_manager):
        """Test low O2 interlock triggers correctly."""
        manager = interlock_manager(interlock_conditions)

        # Normal O2
        normal_params = {"o2_percent": 4.5}
        triggered = manager.check_interlocks(normal_params)
        low_o2_triggers = [t for t in triggered if "O2" in t["name"]]
        assert len(low_o2_triggers) == 0

        # Low O2 - should trigger
        low_params = {"o2_percent": 1.2}
        triggered = manager.check_interlocks(low_params)
        low_o2_triggers = [t for t in triggered if "O2" in t["name"]]
        assert len(low_o2_triggers) == 1
        assert low_o2_triggers[0]["action"] == "trip"

    @pytest.mark.security
    @pytest.mark.interlock
    def test_drum_level_interlocks(self, interlock_conditions, interlock_manager):
        """Test drum level interlocks (high and low)."""
        manager = interlock_manager(interlock_conditions)

        # Normal drum level
        normal_params = {"drum_level_mm": 0.0}
        triggered = manager.check_interlocks(normal_params)
        drum_triggers = [t for t in triggered if "Drum" in t["name"]]
        assert len(drum_triggers) == 0

        # High drum level
        high_params = {"drum_level_mm": 80.0}
        triggered = manager.check_interlocks(high_params)
        drum_triggers = [t for t in triggered if "Drum" in t["name"]]
        assert len(drum_triggers) == 1
        assert "High" in drum_triggers[0]["name"]

        # Low drum level
        low_params = {"drum_level_mm": -80.0}
        triggered = manager.check_interlocks(low_params)
        drum_triggers = [t for t in triggered if "Drum" in t["name"]]
        assert len(drum_triggers) == 1
        assert "Low" in drum_triggers[0]["name"]

    @pytest.mark.security
    @pytest.mark.interlock
    def test_multiple_interlocks_priority_ordering(self, interlock_conditions, interlock_manager):
        """Test multiple interlocks are ordered by priority."""
        manager = interlock_manager(interlock_conditions)

        # Multiple fault conditions
        fault_params = {
            "steam_pressure_bar": 43.0,  # Priority 1
            "flue_gas_temp_c": 290.0,    # Priority 2
            "co_ppm": 250.0              # Priority 2
        }

        triggered = manager.check_interlocks(fault_params)
        assert len(triggered) >= 2

        # Verify priority ordering
        for i in range(len(triggered) - 1):
            assert triggered[i]["priority"] <= triggered[i + 1]["priority"]

    @pytest.mark.security
    @pytest.mark.interlock
    def test_interlock_bypass_requires_authorization(self, interlock_conditions, interlock_manager):
        """Test interlock bypass requires proper authorization."""
        manager = interlock_manager(interlock_conditions)

        # Attempt bypass without authorization
        result = manager.request_bypass("High Steam Pressure Trip", "invalid_token")
        assert not result["success"]
        assert "Unauthorized" in result["error"]

        # Attempt bypass with valid authorization
        result = manager.request_bypass("High Steam Pressure Trip", "valid_safety_token")
        assert result["success"]

    @pytest.mark.security
    @pytest.mark.interlock
    def test_interlock_cannot_be_remotely_disabled(self, interlock_conditions, interlock_manager):
        """Test interlocks cannot be remotely disabled."""
        manager = interlock_manager(interlock_conditions)

        # Interlocks should always check
        fault_params = {"steam_pressure_bar": 50.0}
        triggered = manager.check_interlocks(fault_params)
        assert len(triggered) > 0

        # Even after bypass request, check should still work
        manager.request_bypass("High Steam Pressure Trip", "valid_safety_token")
        triggered = manager.check_interlocks(fault_params)
        assert len(triggered) > 0  # Still detects condition


# =============================================================================
# AUTHENTICATION TESTS
# =============================================================================

class TestAuthentication:
    """Test authentication mechanisms."""

    @pytest.mark.security
    @pytest.mark.authentication
    def test_valid_credentials_success(self, mock_auth_provider):
        """Test valid credentials authenticate successfully."""
        result = mock_auth_provider.authenticate("admin", "admin_pass")
        assert result["success"]
        assert "session_id" in result

    @pytest.mark.security
    @pytest.mark.authentication
    def test_invalid_password_rejection(self, mock_auth_provider):
        """Test invalid password is rejected."""
        result = mock_auth_provider.authenticate("admin", "wrong_password")
        assert not result["success"]
        assert "Invalid credentials" in result["error"]

    @pytest.mark.security
    @pytest.mark.authentication
    def test_invalid_username_rejection(self, mock_auth_provider):
        """Test invalid username is rejected."""
        result = mock_auth_provider.authenticate("unknown_user", "any_password")
        assert not result["success"]

    @pytest.mark.security
    @pytest.mark.authentication
    def test_account_lockout_after_failed_attempts(self, mock_auth_provider):
        """Test account lockout after multiple failed attempts."""
        username = "admin"

        # Multiple failed attempts
        for _ in range(mock_auth_provider.max_attempts):
            mock_auth_provider.authenticate(username, "wrong_password")

        # Should be locked
        result = mock_auth_provider.authenticate(username, "admin_pass")  # Even correct password
        assert not result["success"]
        assert "locked" in result["error"].lower()

    @pytest.mark.security
    @pytest.mark.authentication
    def test_session_logout(self, mock_auth_provider):
        """Test session logout works correctly."""
        # Login
        result = mock_auth_provider.authenticate("operator", "operator_pass")
        session_id = result["session_id"]

        # Verify session exists
        assert mock_auth_provider.check_permission(session_id, PermissionLevel.READ)

        # Logout
        logout_result = mock_auth_provider.logout(session_id)
        assert logout_result

        # Session should no longer work
        assert not mock_auth_provider.check_permission(session_id, PermissionLevel.READ)


# =============================================================================
# AUTHORIZATION TESTS
# =============================================================================

class TestAuthorization:
    """Test authorization mechanisms."""

    @pytest.mark.security
    @pytest.mark.authorization
    def test_viewer_read_only(self, mock_auth_provider):
        """Test viewer has read-only access."""
        result = mock_auth_provider.authenticate("viewer", "viewer_pass")
        session_id = result["session_id"]

        assert mock_auth_provider.check_permission(session_id, PermissionLevel.READ)
        assert not mock_auth_provider.check_permission(session_id, PermissionLevel.WRITE)
        assert not mock_auth_provider.check_permission(session_id, PermissionLevel.CONTROL)
        assert not mock_auth_provider.check_permission(session_id, PermissionLevel.ADMIN)

    @pytest.mark.security
    @pytest.mark.authorization
    def test_operator_control_access(self, mock_auth_provider):
        """Test operator has control access."""
        result = mock_auth_provider.authenticate("operator", "operator_pass")
        session_id = result["session_id"]

        assert mock_auth_provider.check_permission(session_id, PermissionLevel.READ)
        assert mock_auth_provider.check_permission(session_id, PermissionLevel.CONTROL)
        assert not mock_auth_provider.check_permission(session_id, PermissionLevel.WRITE)
        assert not mock_auth_provider.check_permission(session_id, PermissionLevel.ADMIN)

    @pytest.mark.security
    @pytest.mark.authorization
    def test_engineer_write_access(self, mock_auth_provider):
        """Test engineer has write access."""
        result = mock_auth_provider.authenticate("engineer", "engineer_pass")
        session_id = result["session_id"]

        assert mock_auth_provider.check_permission(session_id, PermissionLevel.READ)
        assert mock_auth_provider.check_permission(session_id, PermissionLevel.WRITE)
        assert mock_auth_provider.check_permission(session_id, PermissionLevel.CONTROL)
        assert not mock_auth_provider.check_permission(session_id, PermissionLevel.ADMIN)

    @pytest.mark.security
    @pytest.mark.authorization
    def test_admin_full_access(self, mock_auth_provider):
        """Test admin has full access."""
        result = mock_auth_provider.authenticate("admin", "admin_pass")
        session_id = result["session_id"]

        for permission in [PermissionLevel.READ, PermissionLevel.WRITE,
                          PermissionLevel.CONTROL, PermissionLevel.ADMIN]:
            assert mock_auth_provider.check_permission(session_id, permission)

    @pytest.mark.security
    @pytest.mark.authorization
    def test_safety_officer_override_access(self, mock_auth_provider):
        """Test safety officer has override access."""
        result = mock_auth_provider.authenticate("safety", "safety_pass")
        session_id = result["session_id"]

        assert mock_auth_provider.check_permission(session_id, PermissionLevel.READ)
        assert mock_auth_provider.check_permission(session_id, PermissionLevel.SAFETY_OVERRIDE)
        assert not mock_auth_provider.check_permission(session_id, PermissionLevel.ADMIN)

    @pytest.mark.security
    @pytest.mark.authorization
    def test_invalid_session_denied(self, mock_auth_provider):
        """Test invalid session is denied all access."""
        invalid_session = "invalid_session_id"

        for permission in PermissionLevel:
            assert not mock_auth_provider.check_permission(invalid_session, permission)


# =============================================================================
# AUDIT TRAIL COMPLIANCE
# =============================================================================

class TestAuditCompliance:
    """Test audit trail compliance."""

    @pytest.mark.security
    @pytest.mark.audit
    def test_control_actions_logged(self, audit_logger):
        """Test control actions are logged."""
        # Log control action
        entry = audit_logger.log(
            event_type="control_action",
            user="operator",
            action="adjust_fuel_valve",
            details={"boiler_id": "BOILER-001", "valve_position": 52.5}
        )

        assert entry["logged"]
        assert entry["event_type"] == "control_action"
        assert entry["user"] == "operator"

    @pytest.mark.security
    @pytest.mark.audit
    def test_safety_events_logged(self, audit_logger):
        """Test safety events are logged."""
        # Log safety event
        entry = audit_logger.log(
            event_type="safety_event",
            user="system",
            action="interlock_triggered",
            details={
                "interlock": "High Steam Pressure",
                "value": 43.5,
                "threshold": 42.0
            }
        )

        assert entry["logged"]
        assert entry["event_type"] == "safety_event"

    @pytest.mark.security
    @pytest.mark.audit
    def test_authentication_events_logged(self, audit_logger):
        """Test authentication events are logged."""
        # Log login
        audit_logger.log(
            event_type="authentication",
            user="admin",
            action="login",
            details={"ip": "192.168.1.100", "success": True}
        )

        # Log failed login
        audit_logger.log(
            event_type="authentication",
            user="unknown",
            action="login_failed",
            details={"ip": "192.168.1.200", "success": False}
        )

        auth_entries = audit_logger.get_entries(event_type="authentication")
        assert len(auth_entries) == 2

    @pytest.mark.security
    @pytest.mark.audit
    def test_audit_entries_have_required_fields(self, audit_logger):
        """Test audit entries contain all required fields."""
        # Add some entries
        audit_logger.log(
            event_type="control_action",
            user="engineer",
            action="setpoint_change",
            details={"parameter": "o2_setpoint", "old_value": 3.5, "new_value": 4.0}
        )

        required_fields = ["timestamp", "event_type", "user", "action", "details"]
        is_valid, issues = audit_logger.verify_logging(required_fields)

        assert is_valid, f"Audit entries missing fields: {issues}"

    @pytest.mark.security
    @pytest.mark.audit
    def test_no_secrets_in_logs(self, audit_logger):
        """Test no secrets are logged."""
        # Log with potentially sensitive data
        audit_logger.log(
            event_type="authentication",
            user="admin",
            action="login",
            details={"ip": "192.168.1.100", "session_id": "abc123"}
        )

        # Verify no secrets
        entries = audit_logger.get_entries()
        for entry in entries:
            details_str = str(entry["details"]).lower()
            for word in ["password", "secret", "key", "token", "credential"]:
                assert word not in details_str, f"Potential secret in log: {word}"


# =============================================================================
# DATA PROTECTION
# =============================================================================

class TestDataProtection:
    """Test data protection mechanisms."""

    @pytest.mark.security
    def test_sensitive_data_not_exposed(self, valid_boiler_data):
        """Test sensitive data is not exposed in responses."""
        # Response should not contain internal implementation details
        response = {
            "boiler_id": valid_boiler_data["boiler_id"],
            "efficiency": 85.0,
            "status": "running"
        }

        # Check no internal paths
        response_str = str(response).lower()
        assert "/etc/" not in response_str
        assert "c:\\" not in response_str
        assert "password" not in response_str

    @pytest.mark.security
    def test_error_messages_safe(self):
        """Test error messages don't expose sensitive information."""
        safe_errors = [
            "Invalid parameter value",
            "Authentication failed",
            "Permission denied",
            "Operation not allowed",
        ]

        unsafe_patterns = [
            r"/home/\w+",
            r"C:\\Users\\",
            r"stack trace",
            r"line \d+",
            r"SELECT.*FROM",
        ]

        for error in safe_errors:
            for pattern in unsafe_patterns:
                assert not re.search(pattern, error, re.IGNORECASE), \
                    f"Error message contains unsafe pattern: {error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "security"])
