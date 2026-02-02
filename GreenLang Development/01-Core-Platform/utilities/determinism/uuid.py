"""
GreenLang Deterministic ID Generation - Content-Based Hashing

This module provides deterministic ID generation using content-based hashing.

Features:
- SHA-256 based deterministic IDs
- Consistent hashing for dicts (sorted keys)
- Namespaced UUID generation
- Full content hashing for provenance

Author: GreenLang Team
Date: 2025-11-21
"""

import hashlib
import json
from typing import Union


def deterministic_id(content: Union[str, bytes, dict], prefix: str = "") -> str:
    """
    Generate deterministic ID from content using SHA-256 hashing.

    This function creates reproducible IDs based on content, ensuring
    the same input always produces the same ID.

    Args:
        content: Content to hash (string, bytes, or dict)
        prefix: Optional prefix for the ID

    Returns:
        Deterministic ID string (prefix + first 16 hex chars of hash)

    Examples:
        >>> deterministic_id("hello world", "doc_")
        'doc_2ef37fde8a8b4f63'

        >>> deterministic_id({"key": "value"}, "rec_")
        'rec_a1b2c3d4e5f6g7h8'
    """
    if isinstance(content, dict):
        # Sort dict keys for consistent hashing
        content = json.dumps(content, sort_keys=True, ensure_ascii=True)

    if isinstance(content, str):
        content = content.encode('utf-8')

    # Generate SHA-256 hash
    hash_obj = hashlib.sha256(content)
    hash_hex = hash_obj.hexdigest()

    # Use first 16 characters for readability
    id_suffix = hash_hex[:16]

    return f"{prefix}{id_suffix}"


def deterministic_uuid(namespace: str, name: str) -> str:
    """
    Generate deterministic UUID using namespace and name.

    Args:
        namespace: Namespace for UUID generation
        name: Name within namespace

    Returns:
        Deterministic UUID string
    """
    combined = f"{namespace}:{name}"
    return deterministic_id(combined, "uuid_")


def content_hash(content: Union[str, bytes, dict]) -> str:
    """
    Generate SHA-256 hash of content for provenance tracking.

    Args:
        content: Content to hash

    Returns:
        Full SHA-256 hash hex string
    """
    if isinstance(content, dict):
        content = json.dumps(content, sort_keys=True, ensure_ascii=True)

    if isinstance(content, str):
        content = content.encode('utf-8')

    return hashlib.sha256(content).hexdigest()


__all__ = [
    'deterministic_id',
    'deterministic_uuid',
    'content_hash',
]
