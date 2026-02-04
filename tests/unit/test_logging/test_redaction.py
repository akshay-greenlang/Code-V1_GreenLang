# -*- coding: utf-8 -*-
"""
Unit Tests for Sensitive Data Redaction - INFRA-009

Tests the RedactionProcessor and SensitiveDataPatterns classes that strip PII,
credentials, and other sensitive data from structured log events before they
leave the process boundary.  This is CRITICAL for GHG-protocol compliance and
SOC-2 / GDPR obligations.

Module under test: greenlang.infrastructure.logging.redaction
"""

import re
from typing import Any, Dict

import pytest

from greenlang.infrastructure.logging.redaction import (
    RedactionProcessor,
    SensitiveDataPatterns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def processor() -> RedactionProcessor:
    """Create a default RedactionProcessor instance (no IP redaction)."""
    return RedactionProcessor()


@pytest.fixture
def processor_with_ip_redaction() -> RedactionProcessor:
    """Create a RedactionProcessor that also redacts IPv4 addresses."""
    return RedactionProcessor(redact_ips=True)


@pytest.fixture
def processor_with_custom_pattern() -> RedactionProcessor:
    """Create a RedactionProcessor with a custom regex pattern added."""
    return RedactionProcessor(patterns=[r"GREENLANG-\d{6}"])


def _make_event_dict(event: str = "test_event", **extra) -> Dict[str, Any]:
    """Build a minimal structlog event_dict for testing."""
    ed: Dict[str, Any] = {"event": event}
    ed.update(extra)
    return ed


# ---------------------------------------------------------------------------
# Email Redaction
# ---------------------------------------------------------------------------


class TestEmailRedaction:
    """Tests for email address redaction."""

    def test_redact_email_address(self, processor):
        """A plain email is replaced with [REDACTED_EMAIL]."""
        ed = _make_event_dict(message="Contact user@example.com for info")
        result = processor(None, "info", ed)
        assert "user@example.com" not in result["message"]
        assert "[REDACTED_EMAIL]" in result["message"]

    def test_redact_multiple_emails(self, processor):
        """Multiple emails in one string are all redacted."""
        text = "From alice@corp.io to bob@corp.io cc carol@corp.io"
        ed = _make_event_dict(message=text)
        result = processor(None, "info", ed)
        output = result["message"]
        assert "alice@corp.io" not in output
        assert "bob@corp.io" not in output
        assert "carol@corp.io" not in output
        assert output.count("[REDACTED_EMAIL]") >= 3


# ---------------------------------------------------------------------------
# API Key / Secret / Token Redaction
# ---------------------------------------------------------------------------


class TestApiKeyRedaction:
    """Tests for API key, secret, password, and token redaction."""

    def test_redact_api_key(self, processor):
        """api_key=<value> is redacted."""
        ed = _make_event_dict(message="api_key=abc123def456ghi789")
        result = processor(None, "info", ed)
        output = result["message"]
        assert "abc123def456ghi789" not in output
        assert "REDACTED" in output

    @pytest.mark.parametrize(
        "text",
        [
            "apikey:s3cr3tV4lu3X9z8w7abc",
            "API-KEY=s3cr3tV4lu3X9z8w7abc",
            "token:s3cr3tV4lu3X9z8w7abc",
            "secret=s3cr3tV4lu3X9z8w7abc",
            "password=s3cr3tV4lu3X9z8w7abc",
        ],
        ids=["apikey_colon", "API-KEY_equals", "token_colon", "secret_equals", "password_equals"],
    )
    def test_redact_api_key_variations(self, processor, text):
        """Various credential prefixes are detected and redacted."""
        ed = _make_event_dict(message=text)
        result = processor(None, "info", ed)
        output = result["message"]
        assert "s3cr3tV4lu3X9z8w7abc" not in output
        assert "REDACTED" in output


# ---------------------------------------------------------------------------
# AWS Credential Redaction
# ---------------------------------------------------------------------------


class TestAwsCredentialRedaction:
    """Tests for AWS access key and secret key redaction."""

    def test_redact_aws_access_key(self, processor):
        """An AWS access key ID (AKIA...) is redacted."""
        ed = _make_event_dict(message="key=AKIAIOSFODNN7EXAMPLE")
        result = processor(None, "info", ed)
        output = result["message"]
        assert "AKIAIOSFODNN7EXAMPLE" not in output
        assert "[REDACTED_AWS_KEY]" in output

    def test_redact_aws_secret_key(self, processor):
        """An aws_secret_access_key value is redacted."""
        ed = _make_event_dict(
            message="aws_secret_access_key=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY"
        )
        result = processor(None, "info", ed)
        output = result["message"]
        assert "wJalrXUtnFEMI" not in output
        assert "REDACTED" in output


# ---------------------------------------------------------------------------
# Credit Card Redaction
# ---------------------------------------------------------------------------


class TestCreditCardRedaction:
    """Tests for credit card number redaction."""

    def test_redact_credit_card_visa(self, processor):
        """Visa card numbers (16 digits starting with 4) are redacted."""
        ed = _make_event_dict(message="card: 4111111111111111")
        result = processor(None, "info", ed)
        output = result["message"]
        assert "4111111111111111" not in output
        assert "[REDACTED_CC]" in output

    def test_redact_credit_card_mastercard(self, processor):
        """MasterCard numbers starting with 55 are redacted."""
        ed = _make_event_dict(message="card=5500000000000004")
        result = processor(None, "info", ed)
        assert "5500000000000004" not in result["message"]
        assert "[REDACTED_CC]" in result["message"]

    def test_redact_credit_card_amex(self, processor):
        """American Express (15-digit, starts with 37/34) is redacted."""
        ed = _make_event_dict(message="amex=378282246310005")
        result = processor(None, "info", ed)
        assert "378282246310005" not in result["message"]
        assert "[REDACTED_CC]" in result["message"]


# ---------------------------------------------------------------------------
# JWT Redaction
# ---------------------------------------------------------------------------


class TestJwtRedaction:
    """Tests for JWT token redaction."""

    def test_redact_jwt_token(self, processor):
        """A JWT (three base64url segments separated by dots) is redacted."""
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        ed = _make_event_dict(message=f"token={jwt}")
        result = processor(None, "info", ed)
        output = result["message"]
        assert "eyJhbGciOiJIUzI1NiIs" not in output
        assert "[REDACTED_JWT]" in output


# ---------------------------------------------------------------------------
# IP Address Redaction
# ---------------------------------------------------------------------------


class TestIpAddressRedaction:
    """Tests for optional IPv4 address redaction."""

    def test_redact_ipv4_when_enabled(self, processor_with_ip_redaction):
        """When redact_ips=True, IPv4 addresses are redacted."""
        ed = _make_event_dict(message="client_ip=192.168.1.1 connected")
        result = processor_with_ip_redaction(None, "info", ed)
        output = result["message"]
        assert "192.168.1.1" not in output
        assert "[REDACTED_IP]" in output

    def test_no_redact_ipv4_by_default(self, processor):
        """By default IPv4 addresses are NOT redacted."""
        ed = _make_event_dict(message="client_ip=192.168.1.1 connected")
        result = processor(None, "info", ed)
        output = result["message"]
        assert "192.168.1.1" in output


# ---------------------------------------------------------------------------
# Nested / Complex Data Structures
# ---------------------------------------------------------------------------


class TestNestedStructureRedaction:
    """Tests for redaction inside nested dicts and lists."""

    def test_redact_nested_dict(self, processor):
        """Sensitive data inside nested dicts is redacted."""
        ed = _make_event_dict(
            user={"email": "deep@nested.com", "name": "Test User"}
        )
        result = processor(None, "info", ed)
        assert "deep@nested.com" not in str(result["user"])
        assert "[REDACTED_EMAIL]" in str(result["user"])
        # Non-sensitive field should survive
        assert result["user"]["name"] == "Test User"

    def test_redact_list_values(self, processor):
        """Sensitive data inside list values is redacted."""
        ed = _make_event_dict(
            emails=["alpha@test.com", "beta@test.com", "not-an-email"]
        )
        result = processor(None, "info", ed)
        assert "alpha@test.com" not in str(result["emails"])
        assert "beta@test.com" not in str(result["emails"])
        # Non-sensitive entry should survive
        assert "not-an-email" in result["emails"]

    def test_redact_preserves_non_string(self, processor):
        """Non-string values (int, float, bool, None) pass through unchanged."""
        ed = _make_event_dict(
            count=42,
            ratio=3.14,
            active=True,
            nothing=None,
        )
        result = processor(None, "info", ed)
        assert result["count"] == 42
        assert result["ratio"] == 3.14
        assert result["active"] is True
        assert result["nothing"] is None


# ---------------------------------------------------------------------------
# Mixed / Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case and integration-level redaction tests."""

    def test_redact_mixed_sensitive_data(self, processor):
        """A string containing both an email AND an API key is fully redacted."""
        ed = _make_event_dict(
            message="User user@corp.com with api_key=SUPERSECRET12345678 logged in"
        )
        result = processor(None, "info", ed)
        output = result["message"]
        assert "user@corp.com" not in output
        assert "SUPERSECRET12345678" not in output

    def test_redact_custom_patterns(self, processor_with_custom_pattern):
        """Custom regex patterns from config are applied alongside built-in patterns."""
        ed = _make_event_dict(message="Internal ref GREENLANG-123456 created")
        result = processor_with_custom_pattern(None, "info", ed)
        output = result["message"]
        assert "GREENLANG-123456" not in output
        assert "[REDACTED_CUSTOM]" in output

    def test_redact_empty_string(self, processor):
        """An empty string value is returned unchanged."""
        ed = _make_event_dict(message="")
        result = processor(None, "info", ed)
        assert result["message"] == ""

    def test_redact_no_sensitive_data(self, processor):
        """A normal log message without sensitive data passes through intact."""
        ed = _make_event_dict(message="System started successfully in 42ms")
        result = processor(None, "info", ed)
        assert result["message"] == "System started successfully in 42ms"
        assert "REDACTED" not in result["message"]


# ---------------------------------------------------------------------------
# Structlog Processor Integration
# ---------------------------------------------------------------------------


class TestStructlogIntegration:
    """Tests verifying RedactionProcessor works in a structlog processor chain."""

    def test_redact_event_dict_integration(self, processor):
        """Full event_dict with multiple sensitive fields is properly redacted."""
        ed = _make_event_dict(
            event="user_login",
            user_email="admin@greenlang.io",
            ip_address="10.0.0.1",
            request_id="req-00001",
        )
        result = processor(None, "info", ed)
        # email should be redacted
        assert "admin@greenlang.io" not in str(result)
        assert "[REDACTED_EMAIL]" in result["user_email"]
        # Non-sensitive field should survive
        assert result["request_id"] == "req-00001"
        # IP should NOT be redacted by default
        assert result["ip_address"] == "10.0.0.1"

    def test_redaction_processor_as_structlog_processor(self, processor):
        """RedactionProcessor has the correct structlog processor signature:
        (logger, method_name, event_dict) -> event_dict
        """
        assert callable(processor)

        ed = _make_event_dict(message="benign log line")
        result = processor(None, "info", ed)
        # Must return a dict (the event_dict)
        assert isinstance(result, dict)

    def test_redaction_processor_returns_new_dict(self, processor):
        """RedactionProcessor returns a new dict (does not mutate the original)."""
        ed = _make_event_dict(message="email is user@corp.com")
        original_msg = ed["message"]
        result = processor(None, "info", ed)
        assert isinstance(result, dict)
        # The _redact_dict method creates a new dict
        assert result is not ed
        # Result should have the redacted version
        assert "user@corp.com" not in result["message"]
        assert "[REDACTED_EMAIL]" in result["message"]


# ---------------------------------------------------------------------------
# SensitiveDataPatterns class
# ---------------------------------------------------------------------------


class TestSensitiveDataPatterns:
    """Tests for the SensitiveDataPatterns configuration class."""

    def test_email_pattern_exists(self):
        """Default patterns include an EMAIL regex string attribute."""
        assert hasattr(SensitiveDataPatterns, "EMAIL")
        assert isinstance(SensitiveDataPatterns.EMAIL, str)
        # Verify it can compile
        compiled = re.compile(SensitiveDataPatterns.EMAIL)
        assert compiled.search("user@example.com") is not None

    def test_api_key_pattern_exists(self):
        """Default patterns include an API_KEY regex string attribute."""
        assert hasattr(SensitiveDataPatterns, "API_KEY")
        compiled = re.compile(SensitiveDataPatterns.API_KEY)
        assert compiled.search("api_key=abcdefghijklmnop") is not None

    def test_aws_access_key_pattern_exists(self):
        """Default patterns include an AWS_ACCESS_KEY regex string attribute."""
        assert hasattr(SensitiveDataPatterns, "AWS_ACCESS_KEY")
        compiled = re.compile(SensitiveDataPatterns.AWS_ACCESS_KEY)
        assert compiled.search("AKIAIOSFODNN7EXAMPLE") is not None

    def test_credit_card_pattern_exists(self):
        """Default patterns include a CREDIT_CARD regex string attribute."""
        assert hasattr(SensitiveDataPatterns, "CREDIT_CARD")
        compiled = re.compile(SensitiveDataPatterns.CREDIT_CARD)
        assert compiled.search("4111111111111111") is not None

    def test_jwt_pattern_exists(self):
        """Default patterns include a JWT_TOKEN regex string attribute."""
        assert hasattr(SensitiveDataPatterns, "JWT_TOKEN")
        compiled = re.compile(SensitiveDataPatterns.JWT_TOKEN)
        jwt = (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        assert compiled.search(jwt) is not None

    def test_ipv4_pattern_exists(self):
        """Default patterns include an IPV4 regex string attribute."""
        assert hasattr(SensitiveDataPatterns, "IPV4")
        compiled = re.compile(SensitiveDataPatterns.IPV4)
        assert compiled.search("192.168.1.1") is not None

    def test_all_patterns_are_valid_regex(self):
        """Every string attribute on SensitiveDataPatterns compiles as valid regex."""
        for attr_name in dir(SensitiveDataPatterns):
            if attr_name.startswith("_"):
                continue
            value = getattr(SensitiveDataPatterns, attr_name)
            if isinstance(value, str):
                # Should compile without error
                re.compile(value)
