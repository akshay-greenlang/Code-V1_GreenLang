"""
GreenLang Serialization Module

Provides canonical JSON serialization for consistent hashing and data integrity.
Implements RFC 8785 (JSON Canonicalization Scheme - JCS) for deterministic output.
"""

from .canonical import (
    # Encoder
    CanonicalJSONEncoder,

    # Core functions
    canonical_hash,
    canonical_dumps,
    canonical_loads,

    # Comparison utilities
    canonical_equals,
    diff_canonical,

    # Type handlers
    register_type_handler,

    # Exceptions
    CanonicalSerializationError,
)

__all__ = [
    "CanonicalJSONEncoder",
    "canonical_hash",
    "canonical_dumps",
    "canonical_loads",
    "canonical_equals",
    "diff_canonical",
    "register_type_handler",
    "CanonicalSerializationError",
]

__version__ = "1.0.0"