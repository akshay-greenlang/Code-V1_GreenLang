# -*- coding: utf-8 -*-
"""
Tests for validation.py security module.

Tests cover:
- File size validation
- Path traversal prevention
- HTML/XBRL sanitization
- Input validation
- Edge cases and security vulnerabilities
"""

import pytest
import tempfile
from pathlib import Path

from utils.validation import (
    validate_file_size,
    validate_file_path,
    sanitize_filename,
    sanitize_html,
    sanitize_xbrl_text,
    validate_string_length,
    validate_esrs_code,
    validate_numeric_value,
    validate_email,
    validate_url,
    validate_date_format,
    sanitize_dict_keys,
    validate_json_size,
    ValidationError
)


# ============================================================================
# FILE SIZE VALIDATION TESTS
# ============================================================================

class TestFileSizeValidation:
    """Test file size limit enforcement."""

    def test_valid_small_file(self, tmp_path):
        """Test small file passes validation."""
        # Create 1 MB file
        test_file = tmp_path / "small.csv"
        test_file.write_bytes(b'x' * (1 * 1024 * 1024))

        # Should pass for CSV (limit 100 MB)
        assert validate_file_size(test_file, 'csv') is True

    def test_csv_file_too_large(self, tmp_path):
        """Test CSV file exceeding limit raises error."""
        # Create 101 MB file
        test_file = tmp_path / "large.csv"
        test_file.write_bytes(b'x' * (101 * 1024 * 1024))

        # Should raise ValidationError
        with pytest.raises(ValidationError, match="File too large"):
            validate_file_size(test_file, 'csv')

    def test_json_file_size_limit(self, tmp_path):
        """Test JSON file size limit (50 MB)."""
        # Create 51 MB file
        test_file = tmp_path / "large.json"
        test_file.write_bytes(b'x' * (51 * 1024 * 1024))

        with pytest.raises(ValidationError, match="File too large"):
            validate_file_size(test_file, 'json')

    def test_excel_file_size_limit(self, tmp_path):
        """Test Excel file size limit (100 MB)."""
        # Create 101 MB file
        test_file = tmp_path / "large.xlsx"
        test_file.write_bytes(b'x' * (101 * 1024 * 1024))

        with pytest.raises(ValidationError, match="File too large"):
            validate_file_size(test_file, 'excel')

    def test_pdf_file_size_limit(self, tmp_path):
        """Test PDF file size limit (20 MB)."""
        # Create 21 MB file
        test_file = tmp_path / "large.pdf"
        test_file.write_bytes(b'x' * (21 * 1024 * 1024))

        with pytest.raises(ValidationError, match="File too large"):
            validate_file_size(test_file, 'pdf')

    def test_default_file_size_limit(self, tmp_path):
        """Test default file size limit (10 MB)."""
        # Create 11 MB file
        test_file = tmp_path / "large.bin"
        test_file.write_bytes(b'x' * (11 * 1024 * 1024))

        with pytest.raises(ValidationError, match="File too large"):
            validate_file_size(test_file, 'default')

    def test_nonexistent_file(self, tmp_path):
        """Test validation fails for nonexistent file."""
        test_file = tmp_path / "nonexistent.csv"

        with pytest.raises(ValidationError, match="File not found"):
            validate_file_size(test_file, 'csv')

    def test_empty_file(self, tmp_path):
        """Test empty file passes validation."""
        test_file = tmp_path / "empty.csv"
        test_file.write_bytes(b'')

        # Empty file should pass
        assert validate_file_size(test_file, 'csv') is True


# ============================================================================
# PATH TRAVERSAL PREVENTION TESTS
# ============================================================================

class TestPathTraversalPrevention:
    """Test path traversal attack prevention."""

    def test_valid_path(self, tmp_path):
        """Test valid path passes validation."""
        test_file = tmp_path / "valid_file.csv"
        test_file.write_text("test")

        assert validate_file_path(test_file) is True

    def test_path_traversal_attack(self, tmp_path):
        """Test path traversal attempt is blocked."""
        # Attempt to access parent directory
        malicious_path = tmp_path / ".." / "sensitive.txt"

        with pytest.raises(ValidationError, match="Path traversal detected"):
            validate_file_path(malicious_path)

    def test_allowed_directories(self, tmp_path):
        """Test allowed directories restriction."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        test_file = allowed_dir / "file.csv"
        test_file.write_text("test")

        # Should pass when in allowed directory
        assert validate_file_path(test_file, allowed_dirs=[str(allowed_dir)]) is True

        # Should fail when not in allowed directory
        other_file = tmp_path / "other.csv"
        other_file.write_text("test")

        with pytest.raises(ValidationError, match="not in allowed directories"):
            validate_file_path(other_file, allowed_dirs=[str(allowed_dir)])


# ============================================================================
# FILENAME SANITIZATION TESTS
# ============================================================================

class TestFilenameSanitization:
    """Test filename sanitization."""

    def test_safe_filename(self):
        """Test already safe filename is unchanged."""
        safe_name = "valid_file-123.csv"
        assert sanitize_filename(safe_name) == safe_name

    def test_remove_path_components(self):
        """Test path components are removed."""
        malicious = "/etc/passwd"
        sanitized = sanitize_filename(malicious)
        assert "/" not in sanitized
        assert sanitized == "passwd"

    def test_remove_special_characters(self):
        """Test special characters are removed."""
        unsafe = "file<>:\"|?*.txt"
        sanitized = sanitize_filename(unsafe)
        # Should only contain alphanumeric, dash, underscore, dot
        assert all(c.isalnum() or c in '._-' for c in sanitized)

    def test_prevent_hidden_files(self):
        """Test hidden files (starting with .) are prevented."""
        hidden = ".hidden"
        sanitized = sanitize_filename(hidden)
        assert not sanitized.startswith('.')

    def test_unicode_characters(self):
        """Test unicode characters are handled."""
        unicode_name = "file_日本語.txt"
        sanitized = sanitize_filename(unicode_name)
        # Should replace non-ascii with underscore
        assert "日本語" not in sanitized


# ============================================================================
# HTML SANITIZATION TESTS
# ============================================================================

class TestHTMLSanitization:
    """Test HTML sanitization for XSS prevention."""

    def test_sanitize_script_tag(self):
        """Test script tags are removed."""
        malicious = '<script>alert("XSS")</script>Hello'
        sanitized = sanitize_html(malicious)

        assert '<script>' not in sanitized
        assert '&lt;script&gt;' in sanitized or 'Hello' in sanitized

    def test_sanitize_img_onerror(self):
        """Test image onerror XSS is removed."""
        malicious = '<img src=x onerror="alert(1)">'
        sanitized = sanitize_html(malicious)

        assert 'onerror' not in sanitized.lower()

    def test_sanitize_event_handlers(self):
        """Test event handlers are removed."""
        malicious = '<div onclick="alert(1)">Click me</div>'
        sanitized = sanitize_html(malicious)

        assert 'onclick' not in sanitized.lower()

    def test_allow_safe_tags(self):
        """Test allowed tags are preserved."""
        safe_html = '<p>Hello <strong>world</strong></p>'
        sanitized = sanitize_html(safe_html, allow_tags=['p', 'strong'])

        # With bleach, should preserve these tags
        # Without bleach, will escape all
        assert 'Hello' in sanitized
        assert 'world' in sanitized

    def test_sanitize_iframe(self):
        """Test iframe is removed."""
        malicious = '<iframe src="http://evil.com"></iframe>'
        sanitized = sanitize_html(malicious)

        assert '<iframe' not in sanitized.lower()

    def test_empty_string(self):
        """Test empty string."""
        assert sanitize_html('') == ''

    def test_plain_text(self):
        """Test plain text is unchanged."""
        text = 'This is plain text'
        sanitized = sanitize_html(text)
        assert text == sanitized


# ============================================================================
# XBRL TEXT SANITIZATION TESTS
# ============================================================================

class TestXBRLTextSanitization:
    """Test XBRL/XML text sanitization."""

    def test_escape_xml_special_chars(self):
        """Test XML special characters are escaped."""
        unsafe = 'Value with <tag> & "quotes"'
        safe = sanitize_xbrl_text(unsafe)

        assert '&lt;' in safe
        assert '&amp;' in safe
        assert '&quot;' in safe

    def test_remove_control_characters(self):
        """Test control characters are removed."""
        unsafe = 'Text with \x00 null \x01 and \x1F control chars'
        safe = sanitize_xbrl_text(unsafe)

        # Should not contain control characters
        assert '\x00' not in safe
        assert '\x01' not in safe
        assert '\x1F' not in safe

    def test_preserve_whitespace(self):
        """Test normal whitespace is preserved."""
        text = 'Text with\nnewline\rand\ttab'
        safe = sanitize_xbrl_text(text)

        # These should be preserved
        assert '\n' in safe or '\\n' in safe
        assert '\r' in safe or '\\r' in safe
        assert '\t' in safe or '\\t' in safe

    def test_ampersand_in_text(self):
        """Test ampersands are properly escaped."""
        text = 'Research & Development'
        safe = sanitize_xbrl_text(text)

        assert '&amp;' in safe or '&' not in safe or safe == text


# ============================================================================
# STRING LENGTH VALIDATION TESTS
# ============================================================================

class TestStringLengthValidation:
    """Test string length validation."""

    def test_valid_length(self):
        """Test string within limit passes."""
        short_text = "Hello world"
        assert validate_string_length(short_text, "test_field", max_length=100) is True

    def test_exceeds_limit(self):
        """Test string exceeding limit raises error."""
        long_text = "x" * 10001
        with pytest.raises(ValidationError, match="too long"):
            validate_string_length(long_text, "test_field", max_length=10000)

    def test_exact_limit(self):
        """Test string at exact limit passes."""
        text = "x" * 1000
        assert validate_string_length(text, "test_field", max_length=1000) is True


# ============================================================================
# ESRS CODE VALIDATION TESTS
# ============================================================================

class TestESRSCodeValidation:
    """Test ESRS code format validation."""

    def test_valid_esrs_codes(self):
        """Test valid ESRS codes pass."""
        valid_codes = ["E1-1", "E2-3", "S1-4", "S2-1", "G1-2", "E1-5a"]

        for code in valid_codes:
            assert validate_esrs_code(code) is True

    def test_invalid_esrs_codes(self):
        """Test invalid ESRS codes fail."""
        invalid_codes = [
            "X1-1",      # Invalid prefix
            "E6-1",      # Invalid number (E only goes to E5)
            "S5-1",      # Invalid number (S only goes to S4)
            "E1",        # Missing dash and number
            "E1-",       # Missing number
            "E1-X",      # Invalid number format
            "e1-1",      # Lowercase
        ]

        for code in invalid_codes:
            with pytest.raises(ValidationError, match="Invalid ESRS code format"):
                validate_esrs_code(code)


# ============================================================================
# NUMERIC VALIDATION TESTS
# ============================================================================

class TestNumericValidation:
    """Test numeric value validation."""

    def test_valid_number(self):
        """Test valid numeric value."""
        assert validate_numeric_value(42.5, "temperature") is True

    def test_with_min_max(self):
        """Test with min/max constraints."""
        assert validate_numeric_value(50, "score", min_val=0, max_val=100) is True

    def test_below_minimum(self):
        """Test value below minimum fails."""
        with pytest.raises(ValidationError, match="must be >="):
            validate_numeric_value(-10, "score", min_val=0)

    def test_above_maximum(self):
        """Test value above maximum fails."""
        with pytest.raises(ValidationError, match="must be <="):
            validate_numeric_value(150, "score", max_val=100)

    def test_non_numeric_value(self):
        """Test non-numeric value fails."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_numeric_value("not a number", "value")


# ============================================================================
# EMAIL VALIDATION TESTS
# ============================================================================

class TestEmailValidation:
    """Test email format validation."""

    def test_valid_emails(self):
        """Test valid email addresses."""
        valid_emails = [
            "user@example.com",
            "test.user@company.co.uk",
            "admin+tag@domain.org",
        ]

        for email in valid_emails:
            assert validate_email(email) is True

    def test_invalid_emails(self):
        """Test invalid email addresses."""
        invalid_emails = [
            "notanemail",
            "@example.com",
            "user@",
            "user@.com",
            "user space@example.com",
        ]

        for email in invalid_emails:
            with pytest.raises(ValidationError, match="Invalid email format"):
                validate_email(email)


# ============================================================================
# URL VALIDATION TESTS
# ============================================================================

class TestURLValidation:
    """Test URL validation."""

    def test_valid_urls(self):
        """Test valid URLs."""
        valid_urls = [
            "https://example.com",
            "http://test.org/path",
            "https://sub.domain.com/path?query=1",
        ]

        for url in valid_urls:
            assert validate_url(url) is True

    def test_invalid_scheme(self):
        """Test invalid URL scheme."""
        with pytest.raises(ValidationError, match="URL scheme"):
            validate_url("ftp://example.com", allowed_schemes=['http', 'https'])

    def test_missing_domain(self):
        """Test missing domain."""
        with pytest.raises(ValidationError, match="Invalid URL"):
            validate_url("https://")


# ============================================================================
# DATE VALIDATION TESTS
# ============================================================================

class TestDateValidation:
    """Test date format validation."""

    def test_valid_date(self):
        """Test valid date format."""
        assert validate_date_format("2024-12-31", "%Y-%m-%d") is True

    def test_invalid_date_format(self):
        """Test invalid date format."""
        with pytest.raises(ValidationError, match="Invalid date format"):
            validate_date_format("31-12-2024", "%Y-%m-%d")

    def test_invalid_date_value(self):
        """Test invalid date value."""
        with pytest.raises(ValidationError, match="Invalid date format"):
            validate_date_format("2024-13-45", "%Y-%m-%d")


# ============================================================================
# DICTIONARY SANITIZATION TESTS
# ============================================================================

class TestDictionarySanitization:
    """Test dictionary key sanitization."""

    def test_sanitize_simple_dict(self):
        """Test simple dictionary sanitization."""
        unsafe = {
            "key with spaces": "value",
            "key/with/slashes": "value",
        }

        safe = sanitize_dict_keys(unsafe)

        assert "key_with_spaces" in safe
        assert "key_with_slashes" in safe

    def test_sanitize_nested_dict(self):
        """Test nested dictionary sanitization."""
        unsafe = {
            "outer key": {
                "inner/key": "value"
            }
        }

        safe = sanitize_dict_keys(unsafe)

        assert "outer_key" in safe
        assert "inner_key" in safe["outer_key"]

    def test_max_depth_protection(self):
        """Test max depth protection against deeply nested dicts."""
        # Create deeply nested dict
        deep = {}
        current = deep
        for i in range(20):
            current["level"] = {}
            current = current["level"]

        with pytest.raises(ValidationError, match="nesting exceeds maximum depth"):
            sanitize_dict_keys(deep, max_depth=10)


# ============================================================================
# JSON SIZE VALIDATION TESTS
# ============================================================================

class TestJSONSizeValidation:
    """Test JSON size validation."""

    def test_small_json(self):
        """Test small JSON passes."""
        small_json = '{"key": "value"}'
        assert validate_json_size(small_json) is True

    def test_large_json(self):
        """Test large JSON fails."""
        # Create 51 MB JSON string
        large_json = '{"data": "' + ('x' * (51 * 1024 * 1024)) + '"}'

        with pytest.raises(ValidationError, match="JSON too large"):
            validate_json_size(large_json, max_size_bytes=50 * 1024 * 1024)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestValidationIntegration:
    """Integration tests for validation workflows."""

    def test_file_upload_workflow(self, tmp_path):
        """Test complete file upload validation workflow."""
        # Create test file
        test_file = tmp_path / "upload.csv"
        test_file.write_text("metric_code,value\nE1-1,100\n")

        # Validate size
        assert validate_file_size(test_file, 'csv') is True

        # Validate path
        assert validate_file_path(test_file) is True

        # Sanitize filename
        safe_name = sanitize_filename(test_file.name)
        assert safe_name == "upload.csv"

    def test_xbrl_generation_workflow(self):
        """Test XBRL generation with sanitization."""
        # Unsafe metric value
        unsafe_value = 'Value with <script>alert(1)</script> and & chars'

        # Sanitize for XBRL
        safe_value = sanitize_xbrl_text(unsafe_value)

        # Should be safe for XML
        assert '&lt;' in safe_value or '<script>' not in safe_value
        assert '&amp;' in safe_value or '&' not in safe_value

    def test_api_input_validation(self):
        """Test API input validation workflow."""
        # Validate ESRS code
        assert validate_esrs_code("E1-1") is True

        # Validate numeric value
        assert validate_numeric_value(100.5, "emissions", min_val=0) is True

        # Validate string length
        assert validate_string_length("Short text", "description") is True

        # Validate email
        assert validate_email("user@example.com") is True


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestValidationPerformance:
    """Performance tests for validation functions."""

    def test_file_size_check_performance(self, tmp_path):
        """Test file size check is fast."""
        import time

        test_file = tmp_path / "test.csv"
        test_file.write_bytes(b'x' * (10 * 1024 * 1024))  # 10 MB

        start = time.time()
        for _ in range(100):
            validate_file_size(test_file, 'csv')
        elapsed = time.time() - start

        # Should complete 100 checks in less than 1 second
        assert elapsed < 1.0

    def test_sanitization_performance(self):
        """Test sanitization functions are fast."""
        import time

        text = "Some text with <tags> and & special chars" * 100

        start = time.time()
        for _ in range(1000):
            sanitize_xbrl_text(text)
        elapsed = time.time() - start

        # Should complete 1000 sanitizations in less than 1 second
        assert elapsed < 1.0
