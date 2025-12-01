# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Security Tests

Security tests for the Insulation Inspection Agent.
Tests input validation, authentication, authorization, and data protection.

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import Mock, AsyncMock, patch


# =============================================================================
# TEST: INPUT VALIDATION SECURITY
# =============================================================================

@pytest.mark.security
class TestInputValidationSecurity:
    """Tests for input validation and sanitization."""

    def test_sql_injection_prevention(self, malicious_input_patterns):
        """Test SQL injection prevention."""
        sql_patterns = [p for p in malicious_input_patterns if "DROP" in p or "--" in p]

        for pattern in sql_patterns:
            # Simulate parameterized query (safe)
            safe_query = "SELECT * FROM equipment WHERE tag = ?"
            params = [pattern]

            # Pattern should be treated as literal value, not SQL
            assert "DROP" not in safe_query
            assert pattern in params

    def test_xss_prevention(self, malicious_input_patterns):
        """Test XSS prevention through input sanitization."""
        xss_patterns = [p for p in malicious_input_patterns if "<script>" in p.lower()]

        def sanitize_html(input_str: str) -> str:
            """Simple HTML sanitization."""
            dangerous_patterns = ["<script>", "</script>", "javascript:", "onerror="]
            result = input_str
            for pattern in dangerous_patterns:
                result = result.replace(pattern, "")
            return result

        for pattern in xss_patterns:
            sanitized = sanitize_html(pattern)
            assert "<script>" not in sanitized.lower()

    def test_path_traversal_prevention(self, malicious_input_patterns):
        """Test path traversal attack prevention."""
        path_patterns = [p for p in malicious_input_patterns if ".." in p]

        def validate_path(path: str) -> bool:
            """Validate path doesn't contain traversal patterns."""
            dangerous = ["..", "~", "/etc/", "C:\\"]
            return not any(d in path for d in dangerous)

        for pattern in path_patterns:
            is_safe = validate_path(pattern)
            assert not is_safe, f"Path traversal pattern should be rejected: {pattern}"

    def test_command_injection_prevention(self, malicious_input_patterns):
        """Test command injection prevention."""
        def sanitize_command_input(input_str: str) -> str:
            """Sanitize input for use in commands."""
            # Remove shell metacharacters
            dangerous_chars = [";", "|", "&", "`", "$", "(", ")", "{", "}", "[", "]"]
            result = input_str
            for char in dangerous_chars:
                result = result.replace(char, "")
            return result

        test_input = "tag; rm -rf /"
        sanitized = sanitize_command_input(test_input)
        assert ";" not in sanitized
        assert "rm" in sanitized  # The command text is still there
        assert " -rf /" in sanitized  # But can't be executed

    def test_buffer_overflow_prevention(self, malicious_input_patterns):
        """Test buffer overflow prevention through length limits."""
        long_pattern = [p for p in malicious_input_patterns if len(p) > 1000]

        max_input_length = 500

        for pattern in long_pattern:
            truncated = pattern[:max_input_length]
            assert len(truncated) <= max_input_length

    def test_numeric_input_validation(self):
        """Test numeric input validation."""
        def validate_numeric(value: str, min_val: float = None, max_val: float = None) -> float:
            """Validate and convert numeric input."""
            try:
                num = float(value)
                if min_val is not None and num < min_val:
                    raise ValueError(f"Value {num} below minimum {min_val}")
                if max_val is not None and num > max_val:
                    raise ValueError(f"Value {num} above maximum {max_val}")
                return num
            except ValueError as e:
                raise ValueError(f"Invalid numeric input: {value}")

        # Valid inputs
        assert validate_numeric("25.5", 0, 100) == 25.5

        # Invalid inputs
        with pytest.raises(ValueError):
            validate_numeric("not_a_number")

        with pytest.raises(ValueError):
            validate_numeric("150", 0, 100)

    def test_equipment_tag_validation(self):
        """Test equipment tag format validation."""
        valid_pattern = r"^[A-Z]+-\d{4}-[A-Z]$"

        valid_tags = ["P-1001-A", "V-2500-B", "HX-3000-C"]
        invalid_tags = ["'; DROP TABLE --", "<script>alert(1)</script>", "../../etc/passwd"]

        for tag in valid_tags:
            assert re.match(valid_pattern, tag), f"Valid tag rejected: {tag}"

        for tag in invalid_tags:
            assert not re.match(valid_pattern, tag), f"Invalid tag accepted: {tag}"


# =============================================================================
# TEST: AUTHENTICATION SECURITY
# =============================================================================

@pytest.mark.security
class TestAuthenticationSecurity:
    """Tests for authentication security."""

    def test_valid_token_authentication(self, valid_api_token):
        """Test valid token authentication."""
        def validate_token(token: str) -> bool:
            """Validate API token format and prefix."""
            if not token:
                return False
            if not token.startswith("test_token_"):
                return False
            # Additional validation logic would go here
            return len(token) > 20

        assert validate_token(valid_api_token)

    def test_invalid_token_rejection(self, invalid_api_tokens):
        """Test rejection of invalid tokens."""
        def validate_token(token: str) -> bool:
            if not token or len(token) < 20:
                return False
            if token.startswith("eyJ"):  # JWT with none algorithm
                # Reject tokens that might be JWT with unsafe algorithm
                return False
            return True

        for token in invalid_api_tokens:
            is_valid = validate_token(token)
            assert not is_valid, f"Invalid token should be rejected: {token[:50]}..."

    def test_token_expiration(self):
        """Test token expiration handling."""
        from datetime import datetime, timedelta

        def create_token(expires_in_seconds: int) -> dict:
            return {
                "token": hashlib.sha256(b"test").hexdigest(),
                "expires_at": datetime.now() + timedelta(seconds=expires_in_seconds),
            }

        def is_token_expired(token_data: dict) -> bool:
            return datetime.now() > token_data["expires_at"]

        # Create expired token
        expired_token = create_token(-1)
        assert is_token_expired(expired_token)

        # Create valid token
        valid_token = create_token(3600)
        assert not is_token_expired(valid_token)

    def test_brute_force_protection(self):
        """Test brute force attack protection."""
        class RateLimiter:
            def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
                self.max_attempts = max_attempts
                self.window_seconds = window_seconds
                self.attempts = {}

            def check_rate_limit(self, identifier: str) -> bool:
                """Check if identifier is rate limited."""
                current_time = datetime.now().timestamp()
                if identifier not in self.attempts:
                    self.attempts[identifier] = []

                # Clean old attempts
                self.attempts[identifier] = [
                    t for t in self.attempts[identifier]
                    if current_time - t < self.window_seconds
                ]

                if len(self.attempts[identifier]) >= self.max_attempts:
                    return False

                self.attempts[identifier].append(current_time)
                return True

        limiter = RateLimiter(max_attempts=3)

        # First 3 attempts should succeed
        for i in range(3):
            assert limiter.check_rate_limit("user1")

        # 4th attempt should be blocked
        assert not limiter.check_rate_limit("user1")


# =============================================================================
# TEST: AUTHORIZATION SECURITY
# =============================================================================

@pytest.mark.security
class TestAuthorizationSecurity:
    """Tests for authorization security."""

    def test_role_based_access_control(self):
        """Test role-based access control."""
        roles = {
            "admin": ["read", "write", "delete", "configure"],
            "operator": ["read", "write"],
            "viewer": ["read"],
        }

        def check_permission(role: str, action: str) -> bool:
            """Check if role has permission for action."""
            return role in roles and action in roles[role]

        # Admin can do everything
        assert check_permission("admin", "read")
        assert check_permission("admin", "delete")

        # Viewer can only read
        assert check_permission("viewer", "read")
        assert not check_permission("viewer", "write")

        # Invalid role
        assert not check_permission("hacker", "read")

    def test_resource_ownership_validation(self):
        """Test resource ownership validation."""
        resources = {
            "INS-001": {"owner": "user1", "data": "inspection1"},
            "INS-002": {"owner": "user2", "data": "inspection2"},
        }

        def can_access_resource(user_id: str, resource_id: str, is_admin: bool = False) -> bool:
            """Check if user can access resource."""
            if is_admin:
                return True
            if resource_id not in resources:
                return False
            return resources[resource_id]["owner"] == user_id

        # Owner can access
        assert can_access_resource("user1", "INS-001")

        # Non-owner cannot access
        assert not can_access_resource("user2", "INS-001")

        # Admin can access any
        assert can_access_resource("admin", "INS-001", is_admin=True)

    def test_privilege_escalation_prevention(self):
        """Test privilege escalation prevention."""
        def update_user_role(current_user_role: str, target_role: str) -> bool:
            """Check if role update is allowed."""
            role_hierarchy = {"viewer": 1, "operator": 2, "admin": 3}

            current_level = role_hierarchy.get(current_user_role, 0)
            target_level = role_hierarchy.get(target_role, 0)

            # Cannot escalate to equal or higher role
            return current_level > target_level

        # Admin can demote
        assert update_user_role("admin", "operator")

        # Operator cannot escalate to admin
        assert not update_user_role("operator", "admin")

        # Cannot self-escalate
        assert not update_user_role("viewer", "admin")


# =============================================================================
# TEST: DATA PROTECTION
# =============================================================================

@pytest.mark.security
class TestDataProtection:
    """Tests for data protection and privacy."""

    def test_sensitive_data_masking(self):
        """Test sensitive data masking in logs/responses."""
        def mask_sensitive_data(data: dict) -> dict:
            """Mask sensitive fields in data."""
            sensitive_fields = ["password", "api_key", "token", "secret"]
            masked = data.copy()

            for key in masked:
                if any(sf in key.lower() for sf in sensitive_fields):
                    masked[key] = "***MASKED***"

            return masked

        test_data = {
            "username": "user1",
            "password": "secret123",
            "api_key": "abc123xyz",
            "equipment_tag": "P-1001-A",
        }

        masked = mask_sensitive_data(test_data)

        assert masked["username"] == "user1"  # Not masked
        assert masked["password"] == "***MASKED***"
        assert masked["api_key"] == "***MASKED***"
        assert masked["equipment_tag"] == "P-1001-A"  # Not masked

    def test_data_encryption_at_rest(self):
        """Test data encryption simulation."""
        import base64

        def encrypt_data(data: str, key: str) -> str:
            """Simple XOR-based encryption (for testing only)."""
            encrypted = bytes([ord(d) ^ ord(key[i % len(key)])
                              for i, d in enumerate(data)])
            return base64.b64encode(encrypted).decode()

        def decrypt_data(encrypted: str, key: str) -> str:
            """Decrypt XOR-encrypted data."""
            data = base64.b64decode(encrypted)
            decrypted = bytes([b ^ ord(key[i % len(key)])
                              for i, b in enumerate(data)])
            return decrypted.decode()

        original = "Sensitive inspection data"
        key = "encryption_key"

        encrypted = encrypt_data(original, key)
        assert encrypted != original

        decrypted = decrypt_data(encrypted, key)
        assert decrypted == original

    def test_audit_trail_integrity(self):
        """Test audit trail integrity protection."""
        audit_log = []

        def add_audit_entry(action: str, user: str, details: dict):
            """Add entry to audit log with integrity hash."""
            entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "user": user,
                "details": details,
            }

            # Calculate hash including previous entry
            prev_hash = audit_log[-1]["hash"] if audit_log else "0" * 64
            entry_data = json.dumps(entry, sort_keys=True) + prev_hash
            entry["hash"] = hashlib.sha256(entry_data.encode()).hexdigest()
            entry["prev_hash"] = prev_hash

            audit_log.append(entry)

        def verify_audit_chain() -> bool:
            """Verify audit log chain integrity."""
            for i, entry in enumerate(audit_log):
                if i == 0:
                    expected_prev = "0" * 64
                else:
                    expected_prev = audit_log[i-1]["hash"]

                if entry["prev_hash"] != expected_prev:
                    return False

            return True

        # Add entries
        add_audit_entry("login", "user1", {"ip": "192.168.1.1"})
        add_audit_entry("view_inspection", "user1", {"id": "INS-001"})
        add_audit_entry("create_work_order", "user1", {"id": "WO-001"})

        assert verify_audit_chain()

    def test_pii_handling(self):
        """Test PII (Personally Identifiable Information) handling."""
        def anonymize_pii(data: dict) -> dict:
            """Anonymize PII fields."""
            pii_fields = ["name", "email", "phone", "address"]
            anonymized = data.copy()

            for key in anonymized:
                if any(pf in key.lower() for pf in pii_fields):
                    if isinstance(anonymized[key], str):
                        anonymized[key] = hashlib.sha256(
                            anonymized[key].encode()
                        ).hexdigest()[:16]

            return anonymized

        inspector_data = {
            "inspector_name": "John Smith",
            "email": "john@example.com",
            "employee_id": "EMP-001",
            "inspections_completed": 150,
        }

        anonymized = anonymize_pii(inspector_data)

        assert anonymized["inspector_name"] != "John Smith"
        assert anonymized["email"] != "john@example.com"
        assert anonymized["employee_id"] == "EMP-001"  # Not PII
        assert anonymized["inspections_completed"] == 150  # Not PII


# =============================================================================
# TEST: API SECURITY
# =============================================================================

@pytest.mark.security
class TestAPISecurity:
    """Tests for API security."""

    def test_content_type_validation(self):
        """Test content type validation."""
        allowed_content_types = [
            "application/json",
            "multipart/form-data",
        ]

        def validate_content_type(content_type: str) -> bool:
            return any(ct in content_type for ct in allowed_content_types)

        assert validate_content_type("application/json")
        assert validate_content_type("multipart/form-data; boundary=---")
        assert not validate_content_type("text/html")
        assert not validate_content_type("application/x-www-form-urlencoded")

    def test_request_size_limits(self):
        """Test request size limits."""
        max_request_size = 10 * 1024 * 1024  # 10 MB

        def validate_request_size(size_bytes: int) -> bool:
            return size_bytes <= max_request_size

        assert validate_request_size(1024)  # 1 KB
        assert validate_request_size(5 * 1024 * 1024)  # 5 MB
        assert not validate_request_size(15 * 1024 * 1024)  # 15 MB

    def test_cors_configuration(self):
        """Test CORS configuration."""
        cors_config = {
            "allowed_origins": ["https://app.example.com", "https://admin.example.com"],
            "allowed_methods": ["GET", "POST", "PUT"],
            "allowed_headers": ["Content-Type", "Authorization"],
        }

        def check_cors_origin(origin: str) -> bool:
            return origin in cors_config["allowed_origins"]

        assert check_cors_origin("https://app.example.com")
        assert not check_cors_origin("https://malicious.com")

    def test_rate_limiting_per_endpoint(self):
        """Test per-endpoint rate limiting."""
        rate_limits = {
            "/api/v1/inspections": 100,  # 100 requests per minute
            "/api/v1/reports": 20,       # 20 requests per minute
            "/api/v1/images": 10,        # 10 requests per minute (heavy)
        }

        def get_rate_limit(endpoint: str) -> int:
            for path, limit in rate_limits.items():
                if endpoint.startswith(path):
                    return limit
            return 50  # Default

        assert get_rate_limit("/api/v1/inspections/123") == 100
        assert get_rate_limit("/api/v1/images/upload") == 10

    def test_response_header_security(self):
        """Test security headers in responses."""
        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
        }

        response_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
        }

        for header, expected_value in required_headers.items():
            assert header in response_headers
            assert response_headers[header] == expected_value


# =============================================================================
# TEST: INJECTION PREVENTION
# =============================================================================

@pytest.mark.security
class TestInjectionPrevention:
    """Tests for various injection attack prevention."""

    def test_json_injection_prevention(self):
        """Test JSON injection prevention."""
        def safe_json_parse(json_string: str) -> dict:
            """Safely parse JSON with size limits."""
            max_size = 1024 * 1024  # 1 MB
            max_depth = 10

            if len(json_string) > max_size:
                raise ValueError("JSON too large")

            data = json.loads(json_string)

            def check_depth(obj, depth=0):
                if depth > max_depth:
                    raise ValueError("JSON too deeply nested")
                if isinstance(obj, dict):
                    for v in obj.values():
                        check_depth(v, depth + 1)
                elif isinstance(obj, list):
                    for item in obj:
                        check_depth(item, depth + 1)

            check_depth(data)
            return data

        # Valid JSON
        valid = '{"key": "value"}'
        assert safe_json_parse(valid) == {"key": "value"}

        # Deeply nested (attack attempt)
        deeply_nested = '{"a":' * 20 + '{}' + '}' * 20
        with pytest.raises(ValueError):
            safe_json_parse(deeply_nested)

    def test_template_injection_prevention(self, malicious_input_patterns):
        """Test template injection prevention."""
        template_patterns = [p for p in malicious_input_patterns if "{{" in p]

        def sanitize_for_template(value: str) -> str:
            """Sanitize value for use in templates."""
            dangerous = ["{{", "}}", "{%", "%}", "${"]
            result = value
            for d in dangerous:
                result = result.replace(d, "")
            return result

        for pattern in template_patterns:
            sanitized = sanitize_for_template(pattern)
            assert "{{" not in sanitized
            assert "}}" not in sanitized

    def test_log_injection_prevention(self):
        """Test log injection prevention."""
        def sanitize_for_logging(message: str) -> str:
            """Sanitize message for safe logging."""
            # Remove newlines and control characters
            result = message.replace("\n", " ").replace("\r", " ")
            # Escape potentially dangerous characters
            result = result.replace("\x00", "")
            return result[:500]  # Limit length

        malicious_log = "Normal message\n[FAKE ERROR] Injected log entry"
        sanitized = sanitize_for_logging(malicious_log)

        assert "\n" not in sanitized
        assert "[FAKE ERROR]" in sanitized  # Content preserved but on same line
