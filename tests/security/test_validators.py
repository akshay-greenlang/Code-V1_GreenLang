"""
Tests for Security Validators
"""

import json
from pathlib import Path

import pytest

from greenlang.security.validators import (
    CommandInjectionValidator,
    PathTraversalValidator,
    SQLInjectionValidator,
    URLValidator,
    ValidationError,
    XSSValidator,
    validate_api_key,
    validate_email,
    validate_json_data,
    validate_username,
)


class TestSQLInjectionValidator:
    """Test SQL injection prevention."""

    def test_validate_clean_input(self):
        """Test validation of clean input."""
        result = SQLInjectionValidator.validate("John Doe", allow_quotes=False)
        assert result == "John Doe"

    def test_reject_sql_keywords(self):
        """Test rejection of SQL keywords."""
        with pytest.raises(ValidationError, match="SQL keyword"):
            SQLInjectionValidator.validate("SELECT * FROM users")

    def test_reject_sql_comments(self):
        """Test rejection of SQL comments."""
        with pytest.raises(ValidationError, match="comment"):
            SQLInjectionValidator.validate("admin' --")

    def test_reject_quotes_when_not_allowed(self):
        """Test rejection of quotes."""
        with pytest.raises(ValidationError, match="Quotes not allowed"):
            SQLInjectionValidator.validate("admin'")

    def test_allow_quotes_when_permitted(self):
        """Test allowing quotes when permitted."""
        result = SQLInjectionValidator.validate("O'Brien", allow_quotes=True)
        assert result == "O'Brien"

    def test_escape_string(self):
        """Test string escaping."""
        result = SQLInjectionValidator.escape_string("O'Brien")
        assert result == "O''Brien"


class TestXSSValidator:
    """Test XSS prevention."""

    def test_validate_clean_html(self):
        """Test validation of clean HTML."""
        result = XSSValidator.validate_html("Hello World", strict=True)
        assert result == "Hello World"

    def test_reject_script_tags(self):
        """Test rejection of script tags."""
        with pytest.raises(ValidationError, match="script"):
            XSSValidator.validate_html("<script>alert('xss')</script>")

    def test_reject_event_handlers(self):
        """Test rejection of event handlers."""
        with pytest.raises(ValidationError, match="onclick"):
            XSSValidator.validate_html('<div onclick="alert()">Click</div>')

    def test_reject_javascript_protocol(self):
        """Test rejection of javascript: protocol."""
        with pytest.raises(ValidationError, match="javascript:"):
            XSSValidator.validate_html('<a href="javascript:alert()">Link</a>')

    def test_sanitize_html(self):
        """Test HTML sanitization."""
        result = XSSValidator.sanitize_html("<script>alert()</script>")
        assert "&lt;script&gt;" in result
        assert "<script>" not in result

    def test_sanitize_json(self):
        """Test JSON sanitization."""
        data = {"key": "<script>alert()</script>"}
        result = XSSValidator.sanitize_json(data)
        assert "\\u003c" in result  # < escaped


class TestPathTraversalValidator:
    """Test path traversal prevention."""

    def test_validate_safe_path(self, tmp_path):
        """Test validation of safe path."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        result = PathTraversalValidator.validate_path(
            test_file, base_dir=tmp_path, must_exist=True
        )
        assert result == test_file.resolve()

    def test_reject_parent_directory_traversal(self, tmp_path):
        """Test rejection of .. traversal."""
        with pytest.raises(ValidationError, match="traversal"):
            PathTraversalValidator.validate_path("../../../etc/passwd", base_dir=tmp_path)

    def test_reject_path_outside_base_dir(self, tmp_path):
        """Test rejection of path outside base directory."""
        other_dir = Path("/tmp/other")
        with pytest.raises(ValidationError, match="must be within"):
            PathTraversalValidator.validate_path(other_dir, base_dir=tmp_path)

    def test_reject_nonexistent_path_when_required(self, tmp_path):
        """Test rejection of non-existent path."""
        with pytest.raises(ValidationError, match="does not exist"):
            PathTraversalValidator.validate_path(
                tmp_path / "nonexistent.txt", base_dir=tmp_path, must_exist=True
            )

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        result = PathTraversalValidator.sanitize_filename("test/../../file.txt")
        assert ".." not in result
        assert "/" not in result
        assert result == "test___file.txt"

    def test_sanitize_filename_length_limit(self):
        """Test filename length limiting."""
        long_name = "a" * 300 + ".txt"
        result = PathTraversalValidator.sanitize_filename(long_name, max_length=255)
        assert len(result) <= 255
        assert result.endswith(".txt")


class TestCommandInjectionValidator:
    """Test command injection prevention."""

    def test_validate_safe_argument(self):
        """Test validation of safe argument."""
        result = CommandInjectionValidator.validate_command_arg("test.txt")
        assert result == "test.txt"

    def test_reject_shell_metacharacters(self):
        """Test rejection of shell metacharacters."""
        dangerous_args = [
            "test; rm -rf /",
            "test && echo pwned",
            "test | cat /etc/passwd",
            "test `whoami`",
            "test $(whoami)",
        ]

        for arg in dangerous_args:
            with pytest.raises(ValidationError, match="metacharacter"):
                CommandInjectionValidator.validate_command_arg(arg)

    def test_escape_shell_arg(self):
        """Test shell argument escaping."""
        result = CommandInjectionValidator.escape_shell_arg("file with spaces.txt")
        assert "'" in result or '"' in result  # Should be quoted

    def test_validate_command_list(self):
        """Test command list validation."""
        cmd = ["python", "script.py", "--input", "test.txt"]
        result = CommandInjectionValidator.validate_command_list(cmd)
        assert result == cmd

    def test_reject_invalid_command_list(self):
        """Test rejection of invalid command list."""
        with pytest.raises(ValidationError):
            CommandInjectionValidator.validate_command_list(["test", 123])  # Non-string


class TestURLValidator:
    """Test URL validation."""

    def test_validate_safe_url(self):
        """Test validation of safe URL."""
        result = URLValidator.validate_url("https://example.com/api")
        assert result == "https://example.com/api"

    def test_reject_invalid_scheme(self):
        """Test rejection of invalid URL scheme."""
        with pytest.raises(ValidationError, match="scheme not allowed"):
            URLValidator.validate_url("ftp://example.com")

    def test_reject_localhost(self):
        """Test rejection of localhost (SSRF prevention)."""
        with pytest.raises(ValidationError, match="blocked"):
            URLValidator.validate_url("http://localhost:8000")

    def test_reject_private_ips(self):
        """Test rejection of private IP addresses."""
        with pytest.raises(ValidationError, match="private IP"):
            URLValidator.validate_url("http://192.168.1.1")

    def test_reject_aws_metadata(self):
        """Test rejection of AWS metadata endpoint."""
        with pytest.raises(ValidationError, match="blocked"):
            URLValidator.validate_url("http://169.254.169.254/latest/meta-data/")

    def test_allow_private_ips_when_permitted(self):
        """Test allowing private IPs when permitted."""
        result = URLValidator.validate_url("http://192.168.1.1", allow_private_ips=True)
        assert result == "http://192.168.1.1"


class TestConvenienceFunctions:
    """Test convenience validation functions."""

    def test_validate_api_key_valid(self):
        """Test valid API key."""
        result = validate_api_key("gl_1234567890abcdef")
        assert result == "gl_1234567890abcdef"

    def test_validate_api_key_too_short(self):
        """Test API key too short."""
        with pytest.raises(ValidationError, match="too short"):
            validate_api_key("short")

    def test_validate_api_key_invalid_chars(self):
        """Test API key with invalid characters."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_api_key("gl_123!@#$%^&*()")

    def test_validate_email_valid(self):
        """Test valid email."""
        result = validate_email("user@example.com")
        assert result == "user@example.com"

    def test_validate_email_invalid(self):
        """Test invalid email."""
        with pytest.raises(ValidationError, match="Invalid email"):
            validate_email("not-an-email")

    def test_validate_username_valid(self):
        """Test valid username."""
        result = validate_username("john_doe")
        assert result == "john_doe"

    def test_validate_username_too_short(self):
        """Test username too short."""
        with pytest.raises(ValidationError, match="too short"):
            validate_username("ab")

    def test_validate_username_invalid_chars(self):
        """Test username with invalid characters."""
        with pytest.raises(ValidationError, match="can only contain"):
            validate_username("user@domain")

    def test_validate_json_data_valid(self):
        """Test valid JSON data."""
        json_str = '{"key": "value"}'
        result = validate_json_data(json_str)
        assert result == {"key": "value"}

    def test_validate_json_data_invalid(self):
        """Test invalid JSON data."""
        with pytest.raises(ValidationError, match="Invalid JSON"):
            validate_json_data("{invalid json}")

    def test_validate_json_data_too_large(self):
        """Test JSON data too large."""
        large_json = '{"key": "' + ("x" * 2000000) + '"}'
        with pytest.raises(ValidationError, match="too large"):
            validate_json_data(large_json, max_size=1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
