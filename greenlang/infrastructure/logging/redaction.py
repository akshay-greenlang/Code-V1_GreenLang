"""
Sensitive Data Redaction - INFRA-009

Provides automatic redaction of PII, secrets, and other sensitive data from
structured log entries before they leave the application boundary. The
:class:`RedactionProcessor` is designed as a structlog processor that plugs
into the processor chain configured in :mod:`greenlang.infrastructure.logging.setup`.

Supported patterns include email addresses, API keys, AWS credentials,
credit card numbers, JWT tokens, and IPv4 addresses. Additional custom
patterns can be supplied via :class:`LoggingConfig.redaction_patterns`.

Classes:
    - SensitiveDataPatterns: Regex pattern constants for known sensitive data.
    - RedactionProcessor: structlog processor that redacts matching values.

Example:
    >>> from greenlang.infrastructure.logging.redaction import RedactionProcessor
    >>> processor = RedactionProcessor()
    >>> event = {"event": "User logged in", "email": "user@example.com"}
    >>> redacted = processor(None, None, event)
    >>> redacted["email"]
    'user@[REDACTED_EMAIL]'
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern Constants
# ---------------------------------------------------------------------------


class SensitiveDataPatterns:
    """Regex patterns for identifying sensitive data in log entries.

    Each class attribute is a raw regex string that matches a specific
    category of sensitive information. These are compiled once during
    :class:`RedactionProcessor` initialization for performance.

    Attributes:
        EMAIL: Matches standard email address format.
        API_KEY: Matches key=value pairs where the key suggests a secret
            (api_key, token, secret, password, authorization).
        AWS_ACCESS_KEY: Matches AWS access key IDs (AKIA/ABIA/ACCA/ASIA prefix).
        AWS_SECRET_KEY: Matches AWS secret access key assignments.
        CREDIT_CARD: Matches Visa, Mastercard, Amex, and Discover card numbers.
        JWT_TOKEN: Matches JWT tokens (three base64url segments).
        IPV4: Matches IPv4 addresses in dotted-decimal notation.
    """

    EMAIL: str = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

    API_KEY: str = (
        r"(?i)(api[_-]?key|apikey|token|secret|password|authorization)"
        r"\s*[=:]\s*[\"']?([A-Za-z0-9/+=._-]{16,})[\"']?"
    )

    AWS_ACCESS_KEY: str = r"(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}"

    AWS_SECRET_KEY: str = (
        r"(?i)(aws_secret_access_key|secret_key)"
        r"\s*[=:]\s*[\"']?([A-Za-z0-9/+=]{40})[\"']?"
    )

    CREDIT_CARD: str = (
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}"
        r"|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
    )

    JWT_TOKEN: str = r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"

    IPV4: str = (
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )


# ---------------------------------------------------------------------------
# Default pattern -> replacement mapping
# ---------------------------------------------------------------------------

_DEFAULT_PATTERNS: list[tuple[str, str]] = [
    (SensitiveDataPatterns.JWT_TOKEN, "[REDACTED_JWT]"),
    (SensitiveDataPatterns.AWS_ACCESS_KEY, "[REDACTED_AWS_KEY]"),
    (SensitiveDataPatterns.AWS_SECRET_KEY, "[REDACTED_SECRET]"),
    (SensitiveDataPatterns.API_KEY, "[REDACTED_KEY]"),
    (SensitiveDataPatterns.CREDIT_CARD, "[REDACTED_CC]"),
    (SensitiveDataPatterns.EMAIL, "[REDACTED_EMAIL]"),
]

# IP redaction is opt-in because IPs appear in many non-sensitive contexts
_IP_PATTERN: tuple[str, str] = (
    SensitiveDataPatterns.IPV4,
    "[REDACTED_IP]",
)


# ---------------------------------------------------------------------------
# Redaction Processor
# ---------------------------------------------------------------------------


class RedactionProcessor:
    """structlog processor that redacts sensitive data from log event dicts.

    The processor walks every string value in the event dictionary
    (recursively into nested dicts and lists) and applies compiled regex
    patterns, replacing matches with safe placeholder tokens.

    Pattern application order matters: more specific patterns (JWT, AWS keys)
    are applied before generic ones (email, API key) to avoid partial matches.

    Args:
        patterns: Additional regex patterns as a list of ``(pattern, replacement)``
            tuples. These are appended after the defaults.
        redact_ips: If True, also redact IPv4 addresses. Disabled by default
            because IPs appear in many operational log entries.

    Example:
        >>> processor = RedactionProcessor(redact_ips=True)
        >>> event = {"event": "Connection from 192.168.1.1", "email": "a@b.com"}
        >>> result = processor(None, None, event)
        >>> "192.168.1.1" not in result["event"]
        True
        >>> result["email"]
        '[REDACTED_EMAIL]'
    """

    def __init__(
        self,
        patterns: Optional[list[str]] = None,
        redact_ips: bool = False,
    ) -> None:
        """Initialize the redaction processor.

        Args:
            patterns: Additional regex patterns to redact. Each pattern
                will use ``[REDACTED_CUSTOM]`` as replacement text.
            redact_ips: Whether to redact IPv4 addresses.
        """
        self._compiled: list[tuple[re.Pattern[str], str]] = []

        # Compile default patterns
        for pattern_str, replacement in _DEFAULT_PATTERNS:
            self._compiled.append((re.compile(pattern_str), replacement))

        # Optionally add IP redaction
        if redact_ips:
            ip_pattern, ip_replacement = _IP_PATTERN
            self._compiled.append((re.compile(ip_pattern), ip_replacement))

        # Add custom patterns from config
        if patterns:
            for custom_pattern in patterns:
                try:
                    compiled = re.compile(custom_pattern)
                    self._compiled.append((compiled, "[REDACTED_CUSTOM]"))
                except re.error as exc:
                    logger.warning(
                        "Invalid custom redaction pattern '%s': %s",
                        custom_pattern,
                        exc,
                    )

        logger.debug(
            "RedactionProcessor initialized with %d patterns (ips=%s)",
            len(self._compiled),
            redact_ips,
        )

    def __call__(
        self,
        logger_: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply redaction patterns to every string value in the event dict.

        This is the structlog processor interface. It is called for every
        log entry that passes through the processor chain.

        Args:
            logger_: The wrapped logger object (unused).
            method_name: The name of the log method called (unused).
            event_dict: The structured log event dictionary.

        Returns:
            The event dict with sensitive values replaced by redaction tokens.
        """
        return self._redact_dict(event_dict)

    # -- Private helpers -----------------------------------------------------

    def _redact_value(self, value: Any) -> Any:
        """Recursively redact sensitive data from a value.

        Handles strings, dicts, and lists. Other types are returned unchanged.

        Args:
            value: The value to inspect and potentially redact.

        Returns:
            The redacted value (same type as input where possible).
        """
        if isinstance(value, str):
            return self._redact_string(value)
        if isinstance(value, dict):
            return self._redact_dict(value)
        if isinstance(value, (list, tuple)):
            return self._redact_sequence(value)
        return value

    def _redact_string(self, text: str) -> str:
        """Apply all compiled patterns to a string value.

        Args:
            text: The string to redact.

        Returns:
            The string with all matching patterns replaced.
        """
        result = text
        for compiled_re, replacement in self._compiled:
            result = compiled_re.sub(replacement, result)
        return result

    def _redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact all string values in a dictionary, recursing into nested structures.

        Args:
            data: The dictionary to redact.

        Returns:
            A new dictionary with redacted values.
        """
        redacted: dict[str, Any] = {}
        for key, value in data.items():
            redacted[key] = self._redact_value(value)
        return redacted

    def _redact_sequence(self, seq: list[Any] | tuple[Any, ...]) -> list[Any]:
        """Redact all elements in a list or tuple.

        Args:
            seq: The sequence to redact.

        Returns:
            A new list with redacted elements.
        """
        return [self._redact_value(item) for item in seq]
