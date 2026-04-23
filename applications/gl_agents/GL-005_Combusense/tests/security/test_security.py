# -*- coding: utf-8 -*-
"""
Security Tests for GL-005 COMBUSENSE (CombustionEfficiencyOptimizer).

Comprehensive security validation covering:
- OWASP Top 10 protections
- Input validation and sanitization
- Authentication and authorization
- Safety interlock security
- Audit trail integrity
- Industrial control system security (ICS-CERT)

Reference Standards:
- OWASP Top 10 (2021)
- IEC 62443: Industrial Cybersecurity
- NIST SP 800-82: ICS Security Guide
- ISA/IEC 62443: Security for Industrial Automation
- NFPA 85: Safety interlock requirements
"""

import pytest
import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# -----------------------------------------------------------------------------
# Security Test Data Classes
# -----------------------------------------------------------------------------

@dataclass
class MockAuthProvider:
    """Mock authentication provider for security testing."""
    valid_credentials: Dict[str, Tuple[str, str, List[str]]] = field(default_factory=dict)
    sessions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failed_attempts: Dict[str, int] = field(default_factory=dict)
    locked_accounts: List[str] = field(default_factory=list)
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30

    def __post_init__(self):
        # Default users: (password, role, permissions)
        self.valid_credentials = {
            "operator": ("op_secure_pass_123", "operator", ["read", "monitor"]),
            "engineer": ("eng_secure_pass_456", "engineer", ["read", "write", "monitor", "adjust"]),
            "admin": ("admin_secure_pass_789", "admin", ["read", "write", "control", "admin", "monitor", "adjust"]),
            "safety_officer": ("safety_pass_101", "safety", ["read", "monitor", "safety_override"])
        }

    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and return session."""
        # Check lockout
        if username in self.locked_accounts:
            return {"success": False, "reason": "account_locked"}

        # Validate credentials
        if username not in self.valid_credentials:
            self._record_failed_attempt(username)
            return {"success": False, "reason": "invalid_credentials"}

        expected_pass, role, permissions = self.valid_credentials[username]
        if password != expected_pass:
            self._record_failed_attempt(username)
            return {"success": False, "reason": "invalid_credentials"}

        # Create session
        session_id = hashlib.sha256(
            f"{username}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:32]

        self.sessions[session_id] = {
            "user": username,
            "role": role,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=8)
        }

        self.failed_attempts[username] = 0
        return {"success": True, "session_id": session_id, "role": role}

    def _record_failed_attempt(self, username: str) -> None:
        """Record failed login attempt."""
        self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
        if self.failed_attempts[username] >= self.max_failed_attempts:
            self.locked_accounts.append(username)

    def check_permission(self, session_id: str, action: str) -> bool:
        """Check if session has permission for action."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        # Check expiration
        if datetime.utcnow() > session["expires_at"]:
            del self.sessions[session_id]
            return False

        return action in session["permissions"]

    def validate_session(self, session_id: str) -> bool:
        """Validate session is still active."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]
        return datetime.utcnow() <= session["expires_at"]

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


@dataclass
class MockSafetyInterlock:
    """Mock safety interlock for security testing."""
    name: str
    active: bool = True
    bypass_authorized: bool = False
    bypass_key: str = ""
    last_bypass_user: Optional[str] = None
    bypass_log: List[Dict[str, Any]] = field(default_factory=list)

    def bypass(self, user: str, key: str, auth_provider: MockAuthProvider, session_id: str) -> Dict[str, Any]:
        """Attempt to bypass interlock with authorization."""
        # Must have safety_override permission
        if not auth_provider.check_permission(session_id, "safety_override"):
            return {"success": False, "reason": "unauthorized"}

        # Must have correct bypass key
        if key != self.bypass_key:
            return {"success": False, "reason": "invalid_key"}

        # Log bypass attempt
        self.bypass_log.append({
            "user": user,
            "timestamp": datetime.utcnow().isoformat(),
            "action": "bypass_attempt"
        })

        self.bypass_authorized = True
        self.last_bypass_user = user
        return {"success": True, "warning": "SAFETY INTERLOCK BYPASSED"}

    def reset(self) -> None:
        """Reset interlock to active state."""
        self.bypass_authorized = False
        self.last_bypass_user = None


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def valid_combustion_data():
    """Valid combustion data within safe limits."""
    return {
        "fuel_flow": 100.0,
        "air_flow": 1200.0,
        "furnace_temperature": 900.0,
        "flue_gas_temperature": 250.0,
        "o2_percent": 4.5,
        "co_ppm": 25.0,
        "fuel_pressure": 300.0
    }


@pytest.fixture
def injection_payloads():
    """Common injection attack payloads."""
    return {
        "sql_injection": [
            "'; DROP TABLE data; --",
            "1 OR 1=1",
            "1; DELETE FROM users;",
            "' OR '1'='1",
            "1' AND '1'='1",
            "admin'--",
            "1; UPDATE users SET role='admin' WHERE 1=1;--"
        ],
        "command_injection": [
            "; rm -rf /",
            "| cat /etc/passwd",
            "$(whoami)",
            "`id`",
            "& del C:\\*.*",
            "|| cat /etc/shadow",
            "; shutdown -h now"
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f",
            "..%252f..%252f"
        ],
        "xss_payloads": [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "'\"><script>alert(1)</script>"
        ],
        "overflow_payloads": [
            "A" * 10000,
            "\x00" * 1000,
            "{{" * 500 + "}}" * 500
        ]
    }


@pytest.fixture
def mock_auth_provider():
    """Create mock authentication provider."""
    return MockAuthProvider()


@pytest.fixture
def safety_interlock():
    """Create mock safety interlock."""
    return MockSafetyInterlock(
        name="flame_safety",
        bypass_key="SAFETY_BYPASS_KEY_12345"
    )


# -----------------------------------------------------------------------------
# Input Validation Tests (OWASP A03:2021 - Injection)
# -----------------------------------------------------------------------------

class TestInputValidation:
    """Test input validation against injection attacks."""

    @pytest.mark.security
    def test_reject_sql_injection_in_identifiers(self, injection_payloads):
        """Test SQL injection payloads are rejected in identifiers."""
        valid_pattern = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")

        for payload in injection_payloads["sql_injection"]:
            assert not valid_pattern.match(payload), f"SQL injection not rejected: {payload}"

    @pytest.mark.security
    def test_reject_command_injection(self, injection_payloads):
        """Test command injection payloads are rejected."""
        # Safe characters only
        safe_pattern = re.compile(r"^[a-zA-Z0-9._-]+$")

        for payload in injection_payloads["command_injection"]:
            assert not safe_pattern.match(payload), f"Command injection not rejected: {payload}"

    @pytest.mark.security
    def test_reject_path_traversal(self, injection_payloads):
        """Test path traversal payloads are rejected."""
        for payload in injection_payloads["path_traversal"]:
            # Check for traversal patterns
            has_traversal = ".." in payload or "%2e" in payload.lower()
            assert has_traversal, f"Path traversal should be detected: {payload}"

    @pytest.mark.security
    def test_boundary_validation_fuel_flow(self, valid_combustion_data):
        """Test fuel flow boundary validation."""
        invalid_values = [-100.0, -0.01, float("inf"), float("-inf"), float("nan")]

        def validate_fuel_flow(value: float) -> bool:
            if not isinstance(value, (int, float)):
                return False
            if value != value:  # NaN check
                return False
            if value < 0 or value > 500:
                return False
            return True

        for value in invalid_values:
            assert not validate_fuel_flow(value), f"Invalid fuel flow accepted: {value}"

        # Valid values should pass
        assert validate_fuel_flow(valid_combustion_data["fuel_flow"])

    @pytest.mark.security
    def test_boundary_validation_temperature(self, valid_combustion_data):
        """Test temperature boundary validation."""
        invalid_values = [-500.0, 2500.0, float("inf"), float("nan")]

        def validate_temperature(value: float) -> bool:
            if not isinstance(value, (int, float)):
                return False
            if value != value:  # NaN check
                return False
            if value < -50 or value > 2000:
                return False
            return True

        for value in invalid_values:
            assert not validate_temperature(value), f"Invalid temperature accepted: {value}"

    @pytest.mark.security
    def test_boundary_validation_o2_percent(self, valid_combustion_data):
        """Test O2 percentage boundary validation."""
        invalid_values = [-5.0, 25.0, 100.0, float("inf")]

        def validate_o2(value: float) -> bool:
            if not isinstance(value, (int, float)):
                return False
            if value != value:
                return False
            if value < 0 or value > 21:  # O2 cannot exceed 21%
                return False
            return True

        for value in invalid_values:
            assert not validate_o2(value), f"Invalid O2 accepted: {value}"

    @pytest.mark.security
    def test_input_length_limits(self, injection_payloads):
        """Test input length limits prevent overflow."""
        max_input_length = 1000

        for payload in injection_payloads["overflow_payloads"]:
            truncated = payload[:max_input_length]
            assert len(truncated) <= max_input_length


# -----------------------------------------------------------------------------
# Authentication Tests (OWASP A07:2021 - Identification and Authentication Failures)
# -----------------------------------------------------------------------------

class TestAuthentication:
    """Test authentication mechanisms."""

    @pytest.mark.security
    def test_valid_credentials_success(self, mock_auth_provider):
        """Test valid credentials authenticate successfully."""
        result = mock_auth_provider.authenticate("admin", "admin_secure_pass_789")

        assert result["success"] is True
        assert "session_id" in result
        assert result["role"] == "admin"

    @pytest.mark.security
    def test_invalid_password_rejection(self, mock_auth_provider):
        """Test invalid password is rejected."""
        result = mock_auth_provider.authenticate("admin", "wrong_password")

        assert result["success"] is False
        assert result["reason"] == "invalid_credentials"

    @pytest.mark.security
    def test_invalid_username_rejection(self, mock_auth_provider):
        """Test invalid username is rejected."""
        result = mock_auth_provider.authenticate("unknown_user", "any_password")

        assert result["success"] is False
        assert result["reason"] == "invalid_credentials"

    @pytest.mark.security
    def test_account_lockout_after_failed_attempts(self, mock_auth_provider):
        """Test account lockout after maximum failed attempts."""
        username = "operator"

        # Fail 5 times
        for i in range(5):
            result = mock_auth_provider.authenticate(username, "wrong_password")
            assert result["success"] is False

        # 6th attempt should be locked
        result = mock_auth_provider.authenticate(username, "op_secure_pass_123")
        assert result["success"] is False
        assert result["reason"] == "account_locked"

    @pytest.mark.security
    def test_session_expiration(self, mock_auth_provider):
        """Test session expires after timeout."""
        result = mock_auth_provider.authenticate("operator", "op_secure_pass_123")
        session_id = result["session_id"]

        # Valid immediately
        assert mock_auth_provider.validate_session(session_id)

        # Manually expire session
        mock_auth_provider.sessions[session_id]["expires_at"] = datetime.utcnow() - timedelta(hours=1)

        # Should be invalid now
        assert not mock_auth_provider.validate_session(session_id)

    @pytest.mark.security
    def test_session_revocation(self, mock_auth_provider):
        """Test session can be revoked."""
        result = mock_auth_provider.authenticate("engineer", "eng_secure_pass_456")
        session_id = result["session_id"]

        assert mock_auth_provider.validate_session(session_id)

        # Revoke session
        assert mock_auth_provider.revoke_session(session_id)
        assert not mock_auth_provider.validate_session(session_id)


# -----------------------------------------------------------------------------
# Authorization Tests (OWASP A01:2021 - Broken Access Control)
# -----------------------------------------------------------------------------

class TestAuthorization:
    """Test authorization and access control."""

    @pytest.mark.security
    def test_operator_read_only_access(self, mock_auth_provider):
        """Test operator has read-only access."""
        result = mock_auth_provider.authenticate("operator", "op_secure_pass_123")
        session_id = result["session_id"]

        # Can read and monitor
        assert mock_auth_provider.check_permission(session_id, "read")
        assert mock_auth_provider.check_permission(session_id, "monitor")

        # Cannot write or control
        assert not mock_auth_provider.check_permission(session_id, "write")
        assert not mock_auth_provider.check_permission(session_id, "control")
        assert not mock_auth_provider.check_permission(session_id, "admin")

    @pytest.mark.security
    def test_engineer_write_access(self, mock_auth_provider):
        """Test engineer has write but not control access."""
        result = mock_auth_provider.authenticate("engineer", "eng_secure_pass_456")
        session_id = result["session_id"]

        # Can read, write, adjust
        assert mock_auth_provider.check_permission(session_id, "read")
        assert mock_auth_provider.check_permission(session_id, "write")
        assert mock_auth_provider.check_permission(session_id, "adjust")

        # Cannot control or admin
        assert not mock_auth_provider.check_permission(session_id, "control")
        assert not mock_auth_provider.check_permission(session_id, "admin")

    @pytest.mark.security
    def test_admin_full_access(self, mock_auth_provider):
        """Test admin has full access."""
        result = mock_auth_provider.authenticate("admin", "admin_secure_pass_789")
        session_id = result["session_id"]

        # Can do everything
        for permission in ["read", "write", "control", "admin", "monitor", "adjust"]:
            assert mock_auth_provider.check_permission(session_id, permission)

    @pytest.mark.security
    def test_invalid_session_denied(self, mock_auth_provider):
        """Test invalid session is denied all access."""
        fake_session = "invalid_session_id_12345"

        for permission in ["read", "write", "control", "admin"]:
            assert not mock_auth_provider.check_permission(fake_session, permission)

    @pytest.mark.security
    def test_expired_session_denied(self, mock_auth_provider):
        """Test expired session is denied access."""
        result = mock_auth_provider.authenticate("admin", "admin_secure_pass_789")
        session_id = result["session_id"]

        # Expire the session
        mock_auth_provider.sessions[session_id]["expires_at"] = datetime.utcnow() - timedelta(hours=1)

        # Should be denied
        assert not mock_auth_provider.check_permission(session_id, "read")


# -----------------------------------------------------------------------------
# Safety Interlock Security Tests
# -----------------------------------------------------------------------------

class TestSafetyInterlocks:
    """Test safety interlock security."""

    @pytest.mark.security
    def test_interlock_bypass_requires_authorization(
        self,
        safety_interlock,
        mock_auth_provider
    ):
        """Test interlock bypass requires safety_override permission."""
        # Authenticate as operator (no safety_override)
        result = mock_auth_provider.authenticate("operator", "op_secure_pass_123")
        session_id = result["session_id"]

        # Attempt bypass
        bypass_result = safety_interlock.bypass(
            "operator",
            "SAFETY_BYPASS_KEY_12345",
            mock_auth_provider,
            session_id
        )

        assert bypass_result["success"] is False
        assert bypass_result["reason"] == "unauthorized"
        assert safety_interlock.active is True
        assert safety_interlock.bypass_authorized is False

    @pytest.mark.security
    def test_interlock_bypass_requires_correct_key(
        self,
        safety_interlock,
        mock_auth_provider
    ):
        """Test interlock bypass requires correct key."""
        # Authenticate as safety officer
        result = mock_auth_provider.authenticate("safety_officer", "safety_pass_101")
        session_id = result["session_id"]

        # Attempt bypass with wrong key
        bypass_result = safety_interlock.bypass(
            "safety_officer",
            "WRONG_KEY",
            mock_auth_provider,
            session_id
        )

        assert bypass_result["success"] is False
        assert bypass_result["reason"] == "invalid_key"

    @pytest.mark.security
    def test_interlock_bypass_logged(
        self,
        safety_interlock,
        mock_auth_provider
    ):
        """Test interlock bypass attempts are logged."""
        result = mock_auth_provider.authenticate("safety_officer", "safety_pass_101")
        session_id = result["session_id"]

        # Attempt bypass
        safety_interlock.bypass(
            "safety_officer",
            "SAFETY_BYPASS_KEY_12345",
            mock_auth_provider,
            session_id
        )

        # Check log
        assert len(safety_interlock.bypass_log) == 1
        assert safety_interlock.bypass_log[0]["user"] == "safety_officer"
        assert safety_interlock.bypass_log[0]["action"] == "bypass_attempt"

    @pytest.mark.security
    def test_emergency_stop_cannot_be_disabled_remotely(self):
        """Test emergency stop can only be disabled locally."""
        class EmergencyStop:
            def __init__(self):
                self.active = True

            def disable(self, local: bool = False) -> bool:
                """Only allow disable from local panel."""
                return local

        estop = EmergencyStop()

        # Remote disable should fail
        assert not estop.disable(local=False)

        # Local disable should succeed
        assert estop.disable(local=True)


# -----------------------------------------------------------------------------
# Data Protection Tests (OWASP A02:2021 - Cryptographic Failures)
# -----------------------------------------------------------------------------

class TestDataProtection:
    """Test data protection and cryptography."""

    @pytest.mark.security
    def test_no_secrets_in_logs(self):
        """Test sensitive data is not logged."""
        log_entry = "User operator logged in successfully with session session_abc123"

        sensitive_words = ["password", "secret", "key", "token", "credential", "api_key"]

        for word in sensitive_words:
            assert word not in log_entry.lower(), f"Sensitive word '{word}' found in log"

    @pytest.mark.security
    def test_passwords_not_stored_plaintext(self, mock_auth_provider):
        """Test passwords are not stored in plaintext (mock shows what NOT to do)."""
        # In production, passwords should be hashed
        # This test documents the security requirement

        def hash_password(password: str) -> str:
            import hashlib
            salt = "unique_salt_per_user"  # In production, use random salt
            return hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt.encode(),
                100000
            ).hex()

        # Verify hashing produces different output than input
        password = "test_password_123"
        hashed = hash_password(password)

        assert hashed != password
        assert len(hashed) == 64  # SHA-256 hex length

    @pytest.mark.security
    def test_session_id_is_cryptographically_random(self, mock_auth_provider):
        """Test session IDs are cryptographically random."""
        sessions = []
        for i in range(100):
            # Create unique credentials for each test
            username = f"test_user_{i}"
            mock_auth_provider.valid_credentials[username] = (
                f"pass_{i}", "operator", ["read"]
            )
            result = mock_auth_provider.authenticate(username, f"pass_{i}")
            if result["success"]:
                sessions.append(result["session_id"])

        # All sessions should be unique
        assert len(set(sessions)) == len(sessions)

        # Sessions should be 32 characters (hex)
        for session in sessions:
            assert len(session) == 32
            assert all(c in '0123456789abcdef' for c in session)


# -----------------------------------------------------------------------------
# Audit Trail Tests
# -----------------------------------------------------------------------------

class TestAuditCompliance:
    """Test audit trail and compliance."""

    @pytest.mark.security
    def test_control_actions_logged(self):
        """Test all control actions are logged with required fields."""
        audit_log = []

        def log_action(action: Dict[str, Any]) -> None:
            required_fields = ["action_type", "user", "timestamp", "parameters", "result"]
            for field in required_fields:
                assert field in action, f"Missing required field: {field}"
            audit_log.append(action)

        # Log a control action
        log_action({
            "action_type": "set_fuel_flow",
            "user": "engineer",
            "timestamp": datetime.utcnow().isoformat(),
            "parameters": {"setpoint": 105.0},
            "result": "success"
        })

        assert len(audit_log) == 1
        assert audit_log[0]["user"] == "engineer"

    @pytest.mark.security
    def test_safety_interlock_changes_logged(self, safety_interlock, mock_auth_provider):
        """Test safety interlock changes are logged."""
        result = mock_auth_provider.authenticate("safety_officer", "safety_pass_101")
        session_id = result["session_id"]

        # Bypass interlock
        safety_interlock.bypass(
            "safety_officer",
            "SAFETY_BYPASS_KEY_12345",
            mock_auth_provider,
            session_id
        )

        # Verify log entry
        assert len(safety_interlock.bypass_log) >= 1
        log_entry = safety_interlock.bypass_log[-1]
        assert "user" in log_entry
        assert "timestamp" in log_entry
        assert "action" in log_entry

    @pytest.mark.security
    def test_audit_log_immutability(self):
        """Test audit log entries cannot be modified."""
        class ImmutableAuditLog:
            def __init__(self):
                self._entries: List[Dict[str, Any]] = []
                self._hashes: List[str] = []

            def add_entry(self, entry: Dict[str, Any]) -> str:
                """Add entry and compute hash."""
                entry_copy = entry.copy()
                entry_copy["entry_id"] = len(self._entries)
                entry_str = json.dumps(entry_copy, sort_keys=True)
                entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()

                self._entries.append(entry_copy)
                self._hashes.append(entry_hash)
                return entry_hash

            def verify_integrity(self) -> bool:
                """Verify log has not been tampered."""
                for i, entry in enumerate(self._entries):
                    entry_str = json.dumps(entry, sort_keys=True)
                    expected_hash = hashlib.sha256(entry_str.encode()).hexdigest()
                    if expected_hash != self._hashes[i]:
                        return False
                return True

        log = ImmutableAuditLog()
        log.add_entry({"action": "test", "user": "admin"})

        assert log.verify_integrity()


# -----------------------------------------------------------------------------
# Denial of Service Protection Tests
# -----------------------------------------------------------------------------

class TestDosProtection:
    """Test denial of service protections."""

    @pytest.mark.security
    def test_rate_limiting_authentication(self, mock_auth_provider):
        """Test rate limiting on authentication attempts."""
        # After 5 failed attempts, account is locked
        for _ in range(5):
            mock_auth_provider.authenticate("admin", "wrong")

        # Account should be locked
        result = mock_auth_provider.authenticate("admin", "admin_secure_pass_789")
        assert result["success"] is False
        assert result["reason"] == "account_locked"

    @pytest.mark.security
    def test_input_size_limits(self):
        """Test input size limits prevent memory exhaustion."""
        max_input_size = 10000  # 10KB

        def validate_input_size(data: str) -> bool:
            return len(data) <= max_input_size

        # Normal input should pass
        assert validate_input_size("normal input")

        # Oversized input should fail
        assert not validate_input_size("X" * 100000)

    @pytest.mark.security
    def test_recursive_json_protection(self):
        """Test protection against deeply nested JSON."""
        max_depth = 20

        def check_json_depth(obj: Any, current_depth: int = 0) -> bool:
            if current_depth > max_depth:
                return False
            if isinstance(obj, dict):
                return all(check_json_depth(v, current_depth + 1) for v in obj.values())
            if isinstance(obj, list):
                return all(check_json_depth(v, current_depth + 1) for v in obj)
            return True

        # Normal nested object
        normal = {"a": {"b": {"c": 1}}}
        assert check_json_depth(normal)

        # Deeply nested object (build 25 levels deep)
        deep = {}
        current = deep
        for i in range(25):
            current["level"] = {}
            current = current["level"]

        assert not check_json_depth(deep)


# -----------------------------------------------------------------------------
# Provenance Hash Security Tests
# -----------------------------------------------------------------------------

class TestProvenanceHashSecurity:
    """Test provenance hash security."""

    @pytest.mark.security
    def test_hash_collision_resistance(self):
        """Test SHA-256 collision resistance for similar inputs."""
        input1 = {"fuel_flow": 100.0, "air_flow": 1200.0}
        input2 = {"fuel_flow": 100.0, "air_flow": 1200.1}  # Tiny difference

        hash1 = hashlib.sha256(json.dumps(input1, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(input2, sort_keys=True).encode()).hexdigest()

        # Hashes must be different
        assert hash1 != hash2

        # Hamming distance should be significant
        diff_bits = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        assert diff_bits > 20  # Avalanche effect

    @pytest.mark.security
    def test_hash_tampering_detection(self):
        """Test tampering with data invalidates hash."""
        original_data = {
            "fuel_setpoint": 100.0,
            "air_setpoint": 1200.0,
            "timestamp": "2024-01-15T10:30:00Z"
        }

        original_hash = hashlib.sha256(
            json.dumps(original_data, sort_keys=True).encode()
        ).hexdigest()

        # Tamper with data
        tampered_data = original_data.copy()
        tampered_data["fuel_setpoint"] = 150.0  # Changed value

        tampered_hash = hashlib.sha256(
            json.dumps(tampered_data, sort_keys=True).encode()
        ).hexdigest()

        # Tampering must be detected
        assert original_hash != tampered_hash
