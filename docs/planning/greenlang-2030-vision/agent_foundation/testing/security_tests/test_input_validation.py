# -*- coding: utf-8 -*-
"""
Comprehensive tests for input validation framework.

Tests cover:
- SQL injection prevention
- Command injection prevention
- Path traversal prevention
- XSS prevention
- SSRF prevention
- Field name whitelisting
- Data type validation
- Pydantic model validation
"""

import pytest
from pathlib import Path
from typing import Dict, Any

from security.input_validation import (
    InputValidator,
    TenantIdModel,
    UserIdModel,
    EmailModel,
    SafeQueryInput,
    SafePathInput,
    SafeUrlInput,
    SafeCommandInput,
    PaginationInput,
    FilterInput,
)


class TestInputValidator:
    """Test InputValidator static methods."""

    # ===== ALPHANUMERIC VALIDATION =====

    def test_alphanumeric_valid(self):
        """Test valid alphanumeric strings."""
        valid_inputs = [
            "test123",
            "user-id",
            "tenant_123",
            "ABC-123_xyz",
            "a",
            "1",
        ]

        for input_str in valid_inputs:
            result = InputValidator.validate_alphanumeric(input_str, "field")
            assert result == input_str

    def test_alphanumeric_invalid(self):
        """Test invalid alphanumeric strings."""
        invalid_inputs = [
            "test@123",  # @ not allowed
            "user.id",   # . not allowed
            "test 123",  # space not allowed
            "test/123",  # / not allowed
            "test;123",  # ; not allowed
            "test'123",  # ' not allowed
            "test\"123", # " not allowed
            "test<123",  # < not allowed
            "test>123",  # > not allowed
        ]

        for input_str in invalid_inputs:
            with pytest.raises(ValueError, match="alphanumeric"):
                InputValidator.validate_alphanumeric(input_str, "field")

    def test_alphanumeric_length_constraints(self):
        """Test length validation."""
        # Too short
        with pytest.raises(ValueError, match="length"):
            InputValidator.validate_alphanumeric("", "field", min_length=1)

        # Too long
        with pytest.raises(ValueError, match="length"):
            InputValidator.validate_alphanumeric("a" * 300, "field", max_length=255)

        # Just right
        result = InputValidator.validate_alphanumeric("a" * 255, "field", max_length=255)
        assert len(result) == 255

    def test_alphanumeric_type_error(self):
        """Test non-string input."""
        with pytest.raises(ValueError, match="must be string"):
            InputValidator.validate_alphanumeric(123, "field")

    # ===== UUID VALIDATION =====

    def test_uuid_valid(self):
        """Test valid UUID formats."""
        valid_uuids = [
            "123e4567-e89b-12d3-a456-426614174000",
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        ]

        for uuid_str in valid_uuids:
            result = InputValidator.validate_uuid(uuid_str, "user_id")
            assert result == uuid_str.lower()

    def test_uuid_invalid(self):
        """Test invalid UUID formats."""
        invalid_uuids = [
            "not-a-uuid",
            "123e4567-e89b-12d3-a456",  # Too short
            "123e4567-e89b-12d3-a456-426614174000-extra",  # Too long
            "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",  # Invalid characters
            "123e4567e89b12d3a456426614174000",  # Missing dashes
        ]

        for uuid_str in invalid_uuids:
            with pytest.raises(ValueError, match="UUID"):
                InputValidator.validate_uuid(uuid_str, "user_id")

    # ===== EMAIL VALIDATION =====

    def test_email_valid(self):
        """Test valid email addresses."""
        valid_emails = [
            "user@example.com",
            "test.user@example.com",
            "user+tag@example.co.uk",
            "user123@test-domain.com",
        ]

        for email in valid_emails:
            result = InputValidator.validate_email(email)
            assert result == email.lower()

    def test_email_invalid(self):
        """Test invalid email addresses."""
        invalid_emails = [
            "not-an-email",
            "@example.com",  # Missing local part
            "user@",  # Missing domain
            "user@.com",  # Invalid domain
            "user@example",  # Missing TLD
            "user space@example.com",  # Space not allowed
        ]

        for email in invalid_emails:
            with pytest.raises(ValueError, match="email"):
                InputValidator.validate_email(email)

    def test_email_too_long(self):
        """Test email length limit."""
        long_email = "a" * 300 + "@example.com"
        with pytest.raises(ValueError, match="too long"):
            InputValidator.validate_email(long_email)

    # ===== SQL INJECTION PREVENTION =====

    def test_sql_injection_detected(self):
        """Test SQL injection pattern detection."""
        sql_injections = [
            "test' OR '1'='1",
            "admin'--",
            "1; DROP TABLE users",
            "' UNION SELECT * FROM passwords--",
            "1' AND '1'='1",
            "'; DELETE FROM agents; --",
            "' OR 1=1 --",
            "admin' /*",
            "test'; EXEC xp_cmdshell('dir')",
        ]

        for injection in sql_injections:
            with pytest.raises(ValueError, match="SQL"):
                InputValidator.validate_no_sql_injection(injection, "field")

    def test_sql_injection_safe_strings(self):
        """Test safe strings that should pass."""
        safe_strings = [
            "normal-value",
            "test_123",
            "user input without sql",
            "O'Brien",  # Common name with apostrophe - will fail (expected)
        ]

        # Most should pass except O'Brien
        for safe_str in safe_strings[:3]:
            result = InputValidator.validate_no_sql_injection(safe_str, "field")
            assert result == safe_str

        # O'Brien will fail due to apostrophe
        with pytest.raises(ValueError):
            InputValidator.validate_no_sql_injection("O'Brien", "field")

    # ===== COMMAND INJECTION PREVENTION =====

    def test_command_injection_detected(self):
        """Test command injection pattern detection."""
        command_injections = [
            "test; rm -rf /",
            "test && echo hacked",
            "test | cat /etc/passwd",
            "test `whoami`",
            "test $(ls -la)",
            "test & background",
            "test > /tmp/output",
            "test < /etc/passwd",
            "test{1..10}",
            "test[0-9]",
        ]

        for injection in command_injections:
            with pytest.raises(ValueError, match="shell characters"):
                InputValidator.validate_no_command_injection(injection, "field")

    def test_command_injection_safe_strings(self):
        """Test safe command strings."""
        safe_strings = [
            "normal-value",
            "test_123",
            "safe-command",
            "value",
        ]

        for safe_str in safe_strings:
            result = InputValidator.validate_no_command_injection(safe_str, "field")
            assert result == safe_str

    # ===== XSS PREVENTION =====

    def test_xss_detected(self):
        """Test XSS pattern detection."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(1)'>",
            "javascript:alert(1)",
            "<object data='javascript:alert(1)'>",
            "<embed src='javascript:alert(1)'>",
        ]

        for payload in xss_payloads:
            with pytest.raises(ValueError, match="HTML|JavaScript"):
                InputValidator.validate_no_xss(payload, "field")

    def test_html_sanitization(self):
        """Test HTML sanitization."""
        dangerous_html = "<script>alert('xss')</script>"
        safe_html = InputValidator.sanitize_html(dangerous_html)
        assert "&lt;script&gt;" in safe_html
        assert "<script>" not in safe_html

    # ===== PATH TRAVERSAL PREVENTION =====

    def test_path_traversal_detected(self):
        """Test path traversal attack detection."""
        traversal_paths = [
            "../../etc/passwd",
            "../../../root/.ssh/id_rsa",
            "/etc/passwd",
            "/var/log/auth.log",
            "C:\\Windows\\System32",
            "\\\\network\\share",
            "/proc/self/environ",
            "/root/.bashrc",
        ]

        for path in traversal_paths:
            with pytest.raises(ValueError, match="traversal"):
                InputValidator.validate_path(path, allow_relative=False)

    def test_path_valid_absolute(self):
        """Test valid absolute paths."""
        # Use current file path as valid example
        current_file = Path(__file__).resolve()
        result = InputValidator.validate_path(
            str(current_file),
            must_exist=True,
            allow_relative=False
        )
        assert result == current_file

    def test_path_extension_validation(self):
        """Test file extension validation."""
        yaml_path = "C:\\data\\config.yaml"

        # Should pass with correct extension
        result = InputValidator.validate_path(
            yaml_path,
            must_exist=False,
            allowed_extensions=['.yaml', '.yml']
        )
        assert str(result) == yaml_path

        # Should fail with wrong extension
        json_path = "C:\\data\\config.json"
        with pytest.raises(ValueError, match="extension"):
            InputValidator.validate_path(
                json_path,
                must_exist=False,
                allowed_extensions=['.yaml', '.yml']
            )

    # ===== SSRF PREVENTION =====

    def test_ssrf_private_ip_blocked(self):
        """Test private IP addresses are blocked."""
        private_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "127.0.0.1",
        ]

        for ip in private_ips:
            with pytest.raises(ValueError, match="Private IP|Loopback"):
                InputValidator.validate_ip_address(ip, allow_private=False)

    def test_ssrf_private_ip_allowed(self):
        """Test private IPs can be allowed if configured."""
        private_ip = "192.168.1.1"
        result = InputValidator.validate_ip_address(private_ip, allow_private=True)
        assert result == private_ip

    def test_ssrf_public_ip_allowed(self):
        """Test public IP addresses are allowed."""
        public_ips = [
            "8.8.8.8",
            "1.1.1.1",
            "208.67.222.222",
        ]

        for ip in public_ips:
            result = InputValidator.validate_ip_address(ip, allow_private=False)
            assert result == ip

    def test_url_scheme_validation(self):
        """Test URL scheme whitelisting."""
        # HTTPS should pass
        https_url = "https://api.example.com/data"
        result = InputValidator.validate_url(https_url, allowed_schemes=['https'])
        assert result == https_url

        # HTTP should fail if not in whitelist
        http_url = "http://api.example.com/data"
        with pytest.raises(ValueError, match="scheme"):
            InputValidator.validate_url(http_url, allowed_schemes=['https'])

        # Dangerous schemes should fail
        dangerous_urls = [
            "file:///etc/passwd",
            "ftp://internal.server/data",
            "gopher://vulnerable.server",
        ]

        for url in dangerous_urls:
            with pytest.raises(ValueError):
                InputValidator.validate_url(url, allowed_schemes=['https'])

    # ===== FIELD NAME WHITELISTING =====

    def test_field_name_whitelist_pass(self):
        """Test whitelisted field names."""
        whitelisted_fields = [
            "tenant_id",
            "user_id",
            "agent_id",
            "name",
            "email",
            "status",
        ]

        for field in whitelisted_fields:
            result = InputValidator.validate_field_name(field)
            assert result == field

    def test_field_name_whitelist_fail(self):
        """Test non-whitelisted field names."""
        non_whitelisted_fields = [
            "password",
            "secret_key",
            "internal_data",
            "admin_flag",
        ]

        for field in non_whitelisted_fields:
            with pytest.raises(ValueError, match="whitelist"):
                InputValidator.validate_field_name(field)

    def test_field_name_sql_injection_attempt(self):
        """Test SQL injection in field names blocked by whitelist."""
        injection_attempts = [
            "id; DROP TABLE users--",
            "name' OR '1'='1",
        ]

        for injection in injection_attempts:
            with pytest.raises(ValueError, match="whitelist"):
                InputValidator.validate_field_name(injection)

    # ===== OPERATOR VALIDATION =====

    def test_operator_whitelist(self):
        """Test SQL operator whitelisting."""
        valid_operators = ['=', '!=', '>', '<', '>=', '<=', 'IN', 'LIKE']

        for op in valid_operators:
            result = InputValidator.validate_operator(op)
            assert result == op.upper()

    def test_operator_invalid(self):
        """Test invalid operators rejected."""
        invalid_operators = [
            'OR',
            'AND',
            'UNION',
            '--',
            ';',
        ]

        for op in invalid_operators:
            with pytest.raises(ValueError, match="Operator"):
                InputValidator.validate_operator(op)

    # ===== INTEGER VALIDATION =====

    def test_integer_validation(self):
        """Test integer validation with ranges."""
        # Valid
        result = InputValidator.validate_integer(42, "field", min_value=0, max_value=100)
        assert result == 42

        # Too small
        with pytest.raises(ValueError, match=">="):
            InputValidator.validate_integer(-1, "field", min_value=0)

        # Too large
        with pytest.raises(ValueError, match="<="):
            InputValidator.validate_integer(101, "field", max_value=100)

        # Not an integer
        with pytest.raises(ValueError, match="integer"):
            InputValidator.validate_integer("not-int", "field")

    # ===== JSON VALIDATION =====

    def test_json_validation(self):
        """Test JSON validation."""
        # Valid JSON string
        json_str = '{"key": "value", "number": 42}'
        result = InputValidator.validate_json(json_str)
        assert result == {"key": "value", "number": 42}

        # Valid dict
        json_dict = {"key": "value"}
        result = InputValidator.validate_json(json_dict)
        assert result == json_dict

        # Invalid JSON
        with pytest.raises(ValueError, match="JSON"):
            InputValidator.validate_json("{invalid json}")

        # Too large
        large_json = '{"key": "' + ('a' * 2_000_000) + '"}'
        with pytest.raises(ValueError, match="too large"):
            InputValidator.validate_json(large_json)


class TestPydanticModels:
    """Test Pydantic validation models."""

    def test_tenant_id_model_valid(self):
        """Test valid tenant ID."""
        model = TenantIdModel(tenant_id="tenant-123")
        assert model.tenant_id == "tenant-123"

    def test_tenant_id_model_invalid(self):
        """Test invalid tenant ID."""
        with pytest.raises(ValueError):
            TenantIdModel(tenant_id="tenant@123")  # @ not allowed

    def test_user_id_model_valid(self):
        """Test valid user ID (UUID)."""
        model = UserIdModel(user_id="123e4567-e89b-12d3-a456-426614174000")
        assert "123e4567-e89b" in model.user_id

    def test_user_id_model_invalid(self):
        """Test invalid user ID."""
        with pytest.raises(ValueError):
            UserIdModel(user_id="not-a-uuid")

    def test_email_model_valid(self):
        """Test valid email model."""
        model = EmailModel(email="user@example.com")
        assert model.email == "user@example.com"

    def test_email_model_invalid(self):
        """Test invalid email model."""
        with pytest.raises(ValueError):
            EmailModel(email="not-an-email")

    def test_safe_query_input_valid(self):
        """Test safe query input."""
        query = SafeQueryInput(
            field="tenant_id",
            value="tenant-123",
            operator="="
        )
        assert query.field == "tenant_id"
        assert query.value == "tenant-123"
        assert query.operator == "="

    def test_safe_query_input_invalid_field(self):
        """Test safe query with invalid field."""
        with pytest.raises(ValueError, match="whitelist"):
            SafeQueryInput(
                field="invalid_field",
                value="value",
                operator="="
            )

    def test_safe_query_input_invalid_operator(self):
        """Test safe query with invalid operator."""
        with pytest.raises(ValueError, match="Operator"):
            SafeQueryInput(
                field="tenant_id",
                value="value",
                operator="UNION"
            )

    def test_safe_query_input_sql_injection_in_value(self):
        """Test SQL injection in value detected."""
        with pytest.raises(ValueError, match="SQL"):
            SafeQueryInput(
                field="tenant_id",
                value="test' OR '1'='1",
                operator="="
            )

    def test_pagination_input_valid(self):
        """Test pagination input."""
        pagination = PaginationInput(
            limit=50,
            offset=0,
            sort_by="created_at",
            sort_direction="DESC"
        )
        assert pagination.limit == 50
        assert pagination.sort_by == "created_at"

    def test_pagination_input_invalid_limit(self):
        """Test pagination with invalid limit."""
        with pytest.raises(ValueError):
            PaginationInput(limit=2000)  # Exceeds max

    def test_pagination_input_invalid_sort_field(self):
        """Test pagination with invalid sort field."""
        with pytest.raises(ValueError, match="whitelist"):
            PaginationInput(sort_by="invalid_field")

    def test_filter_input_valid(self):
        """Test filter input."""
        filter_input = FilterInput(
            filters=[
                SafeQueryInput(field="tenant_id", value="tenant-123", operator="="),
                SafeQueryInput(field="status", value="active", operator="="),
            ],
            pagination=PaginationInput(limit=100)
        )
        assert len(filter_input.filters) == 2

    def test_filter_input_too_many_filters(self):
        """Test filter with too many conditions."""
        filters = [
            SafeQueryInput(field="tenant_id", value=f"tenant-{i}", operator="=")
            for i in range(60)  # Exceeds max of 50
        ]

        with pytest.raises(ValueError, match="Too many filters"):
            FilterInput(filters=filters)


class TestCommandValidation:
    """Test command execution validation."""

    def test_command_whitelist(self):
        """Test command whitelisting."""
        allowed_commands = ["kubectl", "docker", "helm"]

        for cmd in allowed_commands:
            result = InputValidator.validate_command(cmd, allowed_commands)
            assert result == cmd

    def test_command_not_in_whitelist(self):
        """Test non-whitelisted command."""
        with pytest.raises(ValueError, match="not allowed"):
            InputValidator.validate_command("rm", ["kubectl", "docker"])

    def test_safe_command_input_valid(self):
        """Test safe command input model."""
        cmd = SafeCommandInput(
            command="kubectl",
            args=["get", "pods"],
            allowed_commands=["kubectl", "docker"]
        )
        assert cmd.command == "kubectl"
        assert cmd.args == ["get", "pods"]

    def test_safe_command_input_invalid_command(self):
        """Test safe command with invalid command."""
        with pytest.raises(ValueError, match="not allowed"):
            SafeCommandInput(
                command="rm",
                args=["-rf", "/"],
                allowed_commands=["kubectl"]
            )

    def test_safe_command_input_injection_in_args(self):
        """Test command injection in arguments."""
        with pytest.raises(ValueError, match="shell characters"):
            SafeCommandInput(
                command="kubectl",
                args=["get", "pods; rm -rf /"],
                allowed_commands=["kubectl"]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
