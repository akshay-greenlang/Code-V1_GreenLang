# -*- coding: utf-8 -*-
"""
Climate Ledger - Hashing Utilities
====================================

Re-exports cryptographic hashing primitives from
``greenlang.utilities.provenance.hashing`` and adds a thin
``content_address`` convenience function for the v3 Climate Ledger
product surface.

Exported symbols:

- ``hash_file`` -- file-level SHA-256 integrity hashing
- ``hash_data`` -- in-memory data hashing (str, bytes, dict)
- ``MerkleTree`` -- hierarchical hash tree with proof generation
- ``content_address`` -- convenience wrapper returning a content address

Example::

    >>> from greenlang.climate_ledger.hashing import hash_data, content_address
    >>> digest = hash_data({"scope": 1, "value": 42.5})
    >>> addr = content_address(b"raw payload bytes")

Author: GreenLang Platform Team
Date: April 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
from typing import Union

# Re-exports from the canonical hashing module
from greenlang.utilities.provenance.hashing import (
    MerkleTree,
    hash_data,
    hash_file,
)

logger = logging.getLogger(__name__)

__all__ = [
    "hash_file",
    "hash_data",
    "MerkleTree",
    "content_address",
]


def content_address(
    data: Union[bytes, str],
    algorithm: str = "sha256",
) -> str:
    """Return a content-addressable identifier for raw data.

    This is a thin convenience wrapper around ``hashlib`` that accepts
    either ``bytes`` or ``str`` and returns the hex digest.  It is
    intentionally simpler than ``hash_data`` (which handles dicts and
    canonical JSON) -- use this when you already have a raw payload.

    Args:
        data: The raw payload to hash.  Strings are encoded as UTF-8.
        algorithm: Hash algorithm name accepted by ``hashlib``
            (default ``"sha256"``).

    Returns:
        Hex-encoded digest string.

    Raises:
        ValueError: If *algorithm* is not available in ``hashlib``.

    Example::

        >>> content_address(b"hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
        >>> content_address("hello world")  # same result
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(
            "Unsupported hash algorithm %r; available: %s"
            % (algorithm, sorted(hashlib.algorithms_available))
        )

    if isinstance(data, str):
        data = data.encode("utf-8")

    digest = hashlib.new(algorithm, data).hexdigest()
    logger.debug("content_address (%s): %s", algorithm, digest[:16])
    return digest
