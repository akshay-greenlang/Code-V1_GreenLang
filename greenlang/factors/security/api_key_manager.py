# -*- coding: utf-8 -*-
"""
API key management for the Factors API.

Handles generation, hashing, validation, and rotation of API keys
used for authenticating Factors API requests.

Key format: gl_{tier}_{32_random_chars}

Supported tiers: community, pro, enterprise, partner, test, internal

Example:
    >>> key = generate_api_key("pro")
    >>> print(key)
    gl_pro_...
    >>> validate_api_key_format(key)
    True
    >>> key_hash = hash_api_key(key)
    >>> len(key_hash)
    64
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

VALID_TIERS = frozenset({
    "community",
    "pro",
    "enterprise",
    "partner",
    "test",
    "internal",
})

_KEY_PREFIX = "gl_"

# API key format pattern: gl_{tier}_{32_alphanumeric_chars}
_KEY_PATTERN = re.compile(
    r"^gl_"
    r"(?:community|pro|enterprise|partner|test|internal)"
    r"_"
    r"[A-Za-z0-9_\-]{24,64}"
    r"$"
)

# Minimum key entropy (characters of random part)
_MIN_RANDOM_LENGTH = 24
_DEFAULT_RANDOM_LENGTH = 32


def generate_api_key(tier: str) -> str:
    """Generate a secure API key for the specified tier.

    Format: gl_{tier}_{32_random_chars}

    Uses cryptographically secure random generation via the secrets
    module to produce URL-safe random characters.

    Args:
        tier: Access tier - one of community, pro, enterprise, partner,
            test, or internal.

    Returns:
        Generated API key string.

    Raises:
        ValueError: If tier is not a valid tier name.
    """
    tier_lower = tier.lower().strip()
    if tier_lower not in VALID_TIERS:
        raise ValueError(
            "Invalid tier %r. Must be one of: %s"
            % (tier, ", ".join(sorted(VALID_TIERS)))
        )

    random_part = secrets.token_urlsafe(_DEFAULT_RANDOM_LENGTH)[:_DEFAULT_RANDOM_LENGTH]
    key = "%s%s_%s" % (_KEY_PREFIX, tier_lower, random_part)

    logger.info("Generated API key for tier=%s prefix=%s", tier_lower, key[:12])
    return key


def hash_api_key(key: str) -> str:
    """Compute SHA-256 hash of an API key for secure storage.

    API keys should never be stored in plaintext. This function
    produces a deterministic hash suitable for database storage
    and lookup.

    Args:
        key: API key string to hash.

    Returns:
        64-character lowercase hex SHA-256 hash.

    Raises:
        ValueError: If key is empty.
    """
    if not key:
        raise ValueError("API key must not be empty")

    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def validate_api_key_format(key: str) -> bool:
    """Validate that an API key matches the expected format.

    Checks:
      - Starts with 'gl_'
      - Contains a valid tier segment
      - Has a random suffix of at least 24 characters
      - Total length is reasonable

    Args:
        key: API key string to validate.

    Returns:
        True if the key format is valid.
    """
    if not key:
        return False

    if len(key) > 200:
        return False

    return bool(_KEY_PATTERN.match(key))


def extract_tier_from_key(key: str) -> Optional[str]:
    """Extract the tier segment from an API key.

    Args:
        key: API key string.

    Returns:
        Tier string (e.g., "pro") or None if format is invalid.
    """
    if not validate_api_key_format(key):
        return None

    # Format: gl_{tier}_{random}
    parts = key.split("_", 2)
    if len(parts) >= 2:
        return parts[1]
    return None


def rotate_api_key(old_key_hash: str) -> str:
    """Generate a new API key to replace one identified by its hash.

    This function does not invalidate the old key (that must be done
    at the storage layer). It generates a fresh key with the same
    tier as would be implied by the caller's context.

    Since we only have the hash (not the original key), the new key
    is generated with 'internal' tier by default. The caller should
    specify the correct tier via generate_api_key() if they know it.

    Args:
        old_key_hash: SHA-256 hash of the key being rotated.

    Returns:
        Newly generated API key string.

    Raises:
        ValueError: If old_key_hash is not a valid SHA-256 hex string.
    """
    if not old_key_hash or len(old_key_hash) != 64:
        raise ValueError("old_key_hash must be a 64-character SHA-256 hex string")

    try:
        int(old_key_hash, 16)
    except ValueError:
        raise ValueError("old_key_hash must be a valid hexadecimal string")

    new_key = generate_api_key("internal")
    new_hash = hash_api_key(new_key)

    logger.info(
        "API key rotated: old_hash_prefix=%s new_hash_prefix=%s",
        old_key_hash[:8], new_hash[:8],
    )
    return new_key


def generate_partner_api_key(partner_id: str) -> str:
    """Generate an API key specifically for a partner.

    Format: gl_partner_{partner_id}_{32_random_chars}

    This is a convenience wrapper that embeds the partner_id in the key
    for easier identification in logs.

    Args:
        partner_id: Unique partner identifier.

    Returns:
        Generated partner API key string.

    Raises:
        ValueError: If partner_id is empty.
    """
    if not partner_id or not partner_id.strip():
        raise ValueError("partner_id must not be empty")

    random_part = secrets.token_urlsafe(_DEFAULT_RANDOM_LENGTH)[:_DEFAULT_RANDOM_LENGTH]
    key = "gl_partner_%s_%s" % (partner_id.strip(), random_part)

    logger.info("Generated partner API key: partner=%s", partner_id)
    return key
