# -*- coding: utf-8 -*-
"""
Input sanitization and validation for the Factors API.

Provides defensive input handling to prevent SQL injection, XSS,
and other injection attacks on API inputs.

Rules:
  - Max query length: 500 characters
  - Allowed characters: alphanumeric, spaces, hyphens, underscores,
    periods, colons
  - Factor IDs follow format: EF:{source}:{fuel}:{geo}:{year}:v{version}
  - Edition IDs follow format: YYYY.MM.N or builtin-v*
  - Geography codes follow ISO 3166-1 alpha-2 or ISO 3166-2

Example:
    >>> sanitize_search_query("diesel; DROP TABLE factors;--")
    'diesel DROP TABLE factors'
    >>> validate_factor_id("EF:EPA:diesel:US:2024:v1")
    True
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

MAX_QUERY_LENGTH = 500

# Allowed characters in search queries: alphanumeric, spaces, hyphens,
# underscores, periods, colons, forward slashes, plus signs, parentheses
_ALLOWED_QUERY_CHARS = re.compile(r"[^a-zA-Z0-9\s\-_.:/()+,]")

# Factor ID format: EF:{source}:{fuel}:{geo}:{year}:v{version}
# Each segment allows alphanumeric, hyphens, underscores
_FACTOR_ID_PATTERN = re.compile(
    r"^EF:"
    r"[A-Za-z0-9_\-]+"  # source
    r":"
    r"[A-Za-z0-9_\-]+"  # fuel
    r":"
    r"[A-Za-z0-9_\-]+"  # geography
    r":"
    r"\d{4}"  # year
    r":v"
    r"[0-9]+(?:\.[0-9]+)*"  # version
    r"$"
)

# Edition ID format: YYYY.MM.N or builtin-v*
_EDITION_ID_PATTERN = re.compile(
    r"^(?:"
    r"\d{4}\.\d{1,2}\.\d+"  # YYYY.MM.N
    r"|"
    r"builtin-v[0-9]+(?:\.[0-9]+)*"  # builtin-v1.0.0
    r"|"
    r"[a-z][a-z0-9\-]*"  # simple slug like "test-edition"
    r")$"
)

# ISO 3166-1 alpha-2 (country) or ISO 3166-2 (subdivision) pattern
_GEOGRAPHY_PATTERN = re.compile(
    r"^[A-Z]{2}(?:-[A-Z0-9]{1,3})?$"
)

# SQL injection patterns to strip
_SQL_INJECTION_PATTERNS = [
    re.compile(r";\s*(?:DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC)\b", re.IGNORECASE),
    re.compile(r"--\s*$"),
    re.compile(r"/\*.*?\*/", re.DOTALL),
    re.compile(r"'\s*(?:OR|AND)\s+'", re.IGNORECASE),
    re.compile(r"'\s*=\s*'", re.IGNORECASE),
    re.compile(r"UNION\s+(?:ALL\s+)?SELECT", re.IGNORECASE),
    re.compile(r"(?:SLEEP|BENCHMARK|WAITFOR)\s*\(", re.IGNORECASE),
]

# XSS patterns to strip
_XSS_PATTERNS = [
    re.compile(r"<\s*script\b[^>]*>.*?</\s*script\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<\s*(?:img|iframe|object|embed|svg|link)\b[^>]*>", re.IGNORECASE),
    re.compile(r"(?:javascript|vbscript|data)\s*:", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),
    re.compile(r"<[^>]+>"),  # strip all HTML tags
]


def sanitize_search_query(q: str) -> str:
    """Sanitize a search query string for safe use.

    Strips SQL injection patterns, XSS payloads, and characters outside
    the allowed set. Enforces maximum length of 500 characters.

    Args:
        q: Raw search query string.

    Returns:
        Sanitized query string.
    """
    if not q:
        return ""

    result = q

    # Step 1: Strip XSS patterns
    for pattern in _XSS_PATTERNS:
        result = pattern.sub("", result)

    # Step 2: Strip SQL injection patterns
    for pattern in _SQL_INJECTION_PATTERNS:
        result = pattern.sub("", result)

    # Step 3: Remove disallowed characters
    result = _ALLOWED_QUERY_CHARS.sub("", result)

    # Step 4: Collapse multiple spaces
    result = re.sub(r"\s+", " ", result).strip()

    # Step 5: Enforce max length
    if len(result) > MAX_QUERY_LENGTH:
        result = result[:MAX_QUERY_LENGTH]
        logger.warning("Search query truncated to %d chars", MAX_QUERY_LENGTH)

    return result


def validate_factor_id(factor_id: str) -> bool:
    """Validate that a factor ID matches the expected format.

    Expected format: EF:{source}:{fuel}:{geo}:{year}:v{version}

    Each segment must be alphanumeric with optional hyphens/underscores.
    Year must be 4 digits. Version must be numeric (e.g., v1, v1.2).

    Args:
        factor_id: Factor ID string to validate.

    Returns:
        True if the format is valid.
    """
    if not factor_id:
        return False

    if len(factor_id) > 200:
        return False

    return bool(_FACTOR_ID_PATTERN.match(factor_id))


def validate_edition_id(edition_id: str) -> bool:
    """Validate that an edition ID matches the expected format.

    Expected formats:
      - YYYY.MM.N (e.g., 2026.04.1)
      - builtin-v* (e.g., builtin-v1.0.0)
      - Simple lowercase slug (e.g., test-edition)

    Args:
        edition_id: Edition ID string to validate.

    Returns:
        True if the format is valid.
    """
    if not edition_id:
        return False

    if len(edition_id) > 100:
        return False

    return bool(_EDITION_ID_PATTERN.match(edition_id))


def sanitize_geography(geo: str) -> str:
    """Validate and normalize an ISO 3166 geography code.

    Accepts ISO 3166-1 alpha-2 (e.g., US, GB) and ISO 3166-2
    subdivision codes (e.g., US-CA, GB-ENG).

    Normalizes to uppercase. Returns empty string if invalid.

    Args:
        geo: Geography code string.

    Returns:
        Normalized geography code, or empty string if invalid.
    """
    if not geo:
        return ""

    normalized = geo.strip().upper()

    if len(normalized) > 10:
        logger.warning("Geography code too long: %r", geo)
        return ""

    if not _GEOGRAPHY_PATTERN.match(normalized):
        logger.warning("Invalid geography code: %r", geo)
        return ""

    return normalized


def validate_pagination(
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    max_limit: int = 500,
) -> tuple:
    """Validate and clamp pagination parameters.

    Args:
        offset: Requested offset (clamped to >= 0).
        limit: Requested page size (clamped to 1..max_limit).
        max_limit: Maximum allowed limit.

    Returns:
        Tuple of (safe_offset, safe_limit).
    """
    safe_offset = max(0, offset or 0)
    safe_limit = max(1, min(limit or 25, max_limit))
    return safe_offset, safe_limit
