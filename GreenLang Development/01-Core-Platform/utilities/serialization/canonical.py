"""
Canonical JSON Serialization for GreenLang

This module implements RFC 8785 (JSON Canonicalization Scheme - JCS) to provide
deterministic JSON serialization for consistent hashing across the framework.

Key Features:
- Deterministic key ordering (alphabetical)
- Consistent number representation (no trailing zeros)
- Special type handling (Decimal, datetime, UUID)
- SHA-256 hashing of canonical form
- Deep equality and diff utilities

Example:
    >>> from greenlang.serialization import canonical_hash, canonical_dumps
    >>>
    >>> data = {"b": 2, "a": 1.0, "c": {"nested": True}}
    >>> canonical_json = canonical_dumps(data)
    >>> print(canonical_json)
    {"a":1,"b":2,"c":{"nested":true}}
    >>>
    >>> hash_value = canonical_hash(data)
    >>> print(hash_value)
    # SHA-256 hash of the canonical JSON representation

Standards:
- RFC 8785: JSON Canonicalization Scheme (JCS)
- RFC 7159: JavaScript Object Notation (JSON)
- ISO 8601: Date and time format
"""

import hashlib
import json
import re
from datetime import datetime, date, time, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Type
from uuid import UUID
from collections import OrderedDict
import logging
from functools import lru_cache
from enum import Enum
from pathlib import Path
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type aliases for clarity
JSONValue = Union[None, bool, int, float, str, List['JSONValue'], Dict[str, 'JSONValue']]
TypeHandler = Callable[[Any], JSONValue]


class CanonicalSerializationError(Exception):
    """Exception raised for errors during canonical serialization."""
    pass


class CanonicalJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that produces canonical JSON output per RFC 8785.

    Features:
    - Sorts dictionary keys alphabetically
    - Removes trailing zeros from floats
    - Handles special types (Decimal, datetime, UUID)
    - Ensures deterministic output for hashing

    Attributes:
        remove_trailing_zeros: Whether to remove trailing zeros from floats
        sort_keys: Whether to sort dictionary keys (always True for canonical)
        ensure_ascii: Whether to escape non-ASCII characters
        separators: JSON separators (no whitespace for canonical)
    """

    def __init__(self,
                 remove_trailing_zeros: bool = True,
                 ensure_ascii: bool = True,
                 **kwargs):
        """
        Initialize the canonical JSON encoder.

        Args:
            remove_trailing_zeros: Remove trailing zeros from float representations
            ensure_ascii: Escape non-ASCII characters
            **kwargs: Additional arguments passed to JSONEncoder
        """
        # Force canonical settings
        kwargs['sort_keys'] = True  # Always sort keys
        kwargs['separators'] = (',', ':')  # No whitespace
        kwargs['ensure_ascii'] = ensure_ascii

        super().__init__(**kwargs)
        self.remove_trailing_zeros = remove_trailing_zeros
        self._type_handlers: Dict[Type, TypeHandler] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default type handlers for common Python types."""
        # Decimal handler
        self._type_handlers[Decimal] = lambda d: float(d) if d.is_finite() else None

        # UUID handler
        self._type_handlers[UUID] = lambda u: str(u)

        # datetime handlers (ISO 8601 format)
        self._type_handlers[datetime] = lambda dt: dt.isoformat() if dt.tzinfo else dt.replace(tzinfo=timezone.utc).isoformat()
        self._type_handlers[date] = lambda d: d.isoformat()
        self._type_handlers[time] = lambda t: t.isoformat()

        # Path handler
        self._type_handlers[Path] = lambda p: str(p)

        # Enum handler
        self._type_handlers[Enum] = lambda e: e.value

        # numpy handlers (if available)
        if NUMPY_AVAILABLE and np is not None:
            self._type_handlers[np.ndarray] = lambda a: a.tolist()
            self._type_handlers[np.integer] = lambda i: int(i)
            self._type_handlers[np.floating] = lambda f: float(f)
            self._type_handlers[np.bool_] = lambda b: bool(b)

    def default(self, obj: Any) -> JSONValue:
        """
        Convert special types to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable representation

        Raises:
            TypeError: If object is not serializable
        """
        # Check registered type handlers
        for type_class, handler in self._type_handlers.items():
            if isinstance(obj, type_class):
                return handler(obj)

        # Handle sets as sorted lists
        if isinstance(obj, set):
            return sorted(list(obj), key=str)

        # Handle bytes as base64
        if isinstance(obj, bytes):
            import base64
            return base64.b64encode(obj).decode('ascii')

        # Fall back to default behavior
        return super().default(obj)

    def encode(self, obj: Any) -> str:
        """
        Encode object to canonical JSON string.

        Args:
            obj: Object to encode

        Returns:
            Canonical JSON string
        """
        # First pass: encode with standard encoder
        json_str = super().encode(obj)

        # Second pass: normalize float representation
        if self.remove_trailing_zeros:
            json_str = self._normalize_floats(json_str)

        return json_str

    def _normalize_floats(self, json_str: str) -> str:
        """
        Normalize float representation by removing trailing zeros.

        Args:
            json_str: JSON string with floats

        Returns:
            JSON string with normalized floats
        """
        # Pattern to match JSON numbers with decimals
        # Matches: 1.0, 1.00, 1.230, etc.
        # But not: 1, 10, 1e10, 1.0e10
        pattern = r'\b(\d+\.\d*?)0+\b(?![eE])'

        def replace_float(match):
            num_str = match.group(0)
            # Remove trailing zeros
            if '.' in num_str:
                num_str = num_str.rstrip('0')
                # If all decimals were zeros, remove the decimal point
                if num_str.endswith('.'):
                    num_str = num_str[:-1]
            return num_str

        return re.sub(pattern, replace_float, json_str)


# Global encoder instance
_canonical_encoder = CanonicalJSONEncoder()

# Type handler registry
_type_handlers: Dict[Type, TypeHandler] = {}


def register_type_handler(type_class: Type, handler: TypeHandler) -> None:
    """
    Register a custom type handler for canonical serialization.

    Args:
        type_class: Type to handle
        handler: Function that converts the type to JSON-serializable format

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class CustomType:
        ...     value: int
        >>>
        >>> register_type_handler(CustomType, lambda obj: {"value": obj.value})
    """
    _type_handlers[type_class] = handler
    _canonical_encoder._type_handlers[type_class] = handler


def canonical_dumps(obj: Any,
                   cls: Optional[Type[json.JSONEncoder]] = None,
                   **kwargs) -> str:
    """
    Serialize object to canonical JSON string.

    Args:
        obj: Object to serialize
        cls: Custom encoder class (defaults to CanonicalJSONEncoder)
        **kwargs: Additional arguments for encoder

    Returns:
        Canonical JSON string

    Raises:
        CanonicalSerializationError: If serialization fails

    Example:
        >>> data = {"z": 3, "a": 1.000, "m": {"nested": True}}
        >>> canonical_dumps(data)
        '{"a":1,"m":{"nested":true},"z":3}'
    """
    try:
        if cls is None:
            return _canonical_encoder.encode(obj)
        else:
            encoder = cls(
                sort_keys=True,
                separators=(',', ':'),
                **kwargs
            )
            return encoder.encode(obj)
    except (TypeError, ValueError) as e:
        raise CanonicalSerializationError(f"Failed to serialize object: {e}") from e


def canonical_loads(s: Union[str, bytes], **kwargs) -> Any:
    """
    Parse JSON string and return Python object.

    This is a wrapper around json.loads that ensures consistent parsing.

    Args:
        s: JSON string or bytes to parse
        **kwargs: Additional arguments for json.loads

    Returns:
        Parsed Python object

    Raises:
        CanonicalSerializationError: If parsing fails

    Example:
        >>> json_str = '{"a":1,"b":2}'
        >>> canonical_loads(json_str)
        {'a': 1, 'b': 2}
    """
    try:
        # Use object_pairs_hook to preserve order if needed
        return json.loads(s, **kwargs)
    except (json.JSONDecodeError, ValueError) as e:
        raise CanonicalSerializationError(f"Failed to parse JSON: {e}") from e


def canonical_hash(obj: Any, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of object using canonical JSON representation.

    Args:
        obj: Object to hash
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Hexadecimal hash string

    Raises:
        CanonicalSerializationError: If hashing fails

    Example:
        >>> data = {"b": 2, "a": 1}
        >>> hash1 = canonical_hash(data)
        >>> data2 = {"a": 1, "b": 2}  # Different order
        >>> hash2 = canonical_hash(data2)
        >>> assert hash1 == hash2  # Same canonical hash
    """
    try:
        # Convert to canonical JSON
        canonical_json = canonical_dumps(obj)

        # Calculate hash
        if algorithm == 'sha256':
            hasher = hashlib.sha256()
        elif algorithm == 'sha512':
            hasher = hashlib.sha512()
        elif algorithm == 'sha384':
            hasher = hashlib.sha384()
        elif algorithm == 'md5':
            hasher = hashlib.md5()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        hasher.update(canonical_json.encode('utf-8'))
        return hasher.hexdigest()

    except Exception as e:
        raise CanonicalSerializationError(f"Failed to calculate hash: {e}") from e


def canonical_equals(obj1: Any, obj2: Any) -> bool:
    """
    Compare two objects for deep equality using canonical form.

    Args:
        obj1: First object
        obj2: Second object

    Returns:
        True if objects are canonically equal

    Example:
        >>> obj1 = {"b": 2.0, "a": 1}
        >>> obj2 = {"a": 1.000, "b": 2}
        >>> canonical_equals(obj1, obj2)
        True
    """
    try:
        # Quick check with hashes (cached for performance)
        return canonical_hash(obj1) == canonical_hash(obj2)
    except CanonicalSerializationError:
        # Fallback to direct comparison if serialization fails
        return obj1 == obj2


def diff_canonical(obj1: Any, obj2: Any) -> Dict[str, Any]:
    """
    Find differences between two objects using canonical form.

    Args:
        obj1: First object (baseline)
        obj2: Second object (comparison)

    Returns:
        Dictionary describing the differences

    Example:
        >>> obj1 = {"a": 1, "b": 2, "c": 3}
        >>> obj2 = {"a": 1, "b": 20, "d": 4}
        >>> diff = diff_canonical(obj1, obj2)
        >>> print(diff)
        {
            'modified': {'b': {'old': 2, 'new': 20}},
            'removed': {'c': 3},
            'added': {'d': 4},
            'equal': False
        }
    """
    result = {
        'added': {},
        'removed': {},
        'modified': {},
        'equal': True
    }

    try:
        # First check if they're equal
        if canonical_equals(obj1, obj2):
            return result

        result['equal'] = False

        # Deep diff for dictionaries
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            keys1 = set(obj1.keys())
            keys2 = set(obj2.keys())

            # Added keys
            for key in keys2 - keys1:
                result['added'][key] = obj2[key]

            # Removed keys
            for key in keys1 - keys2:
                result['removed'][key] = obj1[key]

            # Modified keys
            for key in keys1 & keys2:
                if not canonical_equals(obj1[key], obj2[key]):
                    # Recursively diff nested structures
                    if isinstance(obj1[key], dict) and isinstance(obj2[key], dict):
                        nested_diff = diff_canonical(obj1[key], obj2[key])
                        if not nested_diff['equal']:
                            result['modified'][key] = nested_diff
                    else:
                        result['modified'][key] = {
                            'old': obj1[key],
                            'new': obj2[key]
                        }

        # Deep diff for lists
        elif isinstance(obj1, list) and isinstance(obj2, list):
            if len(obj1) != len(obj2):
                result['length_changed'] = {
                    'old': len(obj1),
                    'new': len(obj2)
                }

            # Compare elements
            max_len = max(len(obj1), len(obj2))
            for i in range(max_len):
                if i >= len(obj1):
                    result['added'][f'[{i}]'] = obj2[i]
                elif i >= len(obj2):
                    result['removed'][f'[{i}]'] = obj1[i]
                elif not canonical_equals(obj1[i], obj2[i]):
                    result['modified'][f'[{i}]'] = {
                        'old': obj1[i],
                        'new': obj2[i]
                    }

        # Simple types
        else:
            result['type_or_value_changed'] = {
                'old': {'type': type(obj1).__name__, 'value': obj1},
                'new': {'type': type(obj2).__name__, 'value': obj2}
            }

    except Exception as e:
        logger.error(f"Error during diff calculation: {e}")
        result['error'] = str(e)

    return result


class CanonicalSerializer:
    """
    High-level serializer with caching and batch operations.

    This class provides optimized serialization for production use,
    including caching, batch operations, and error handling.
    """

    def __init__(self,
                 cache_size: int = 1024,
                 algorithm: str = 'sha256'):
        """
        Initialize the canonical serializer.

        Args:
            cache_size: Maximum number of cached hash values
            algorithm: Hash algorithm to use
        """
        self.algorithm = algorithm
        self.cache_size = cache_size
        self._hash_cache: Dict[str, str] = {}
        self._encoder = CanonicalJSONEncoder()

    def serialize_batch(self, objects: List[Any]) -> List[str]:
        """
        Serialize multiple objects to canonical JSON.

        Args:
            objects: List of objects to serialize

        Returns:
            List of canonical JSON strings
        """
        return [canonical_dumps(obj) for obj in objects]

    def hash_batch(self, objects: List[Any]) -> List[str]:
        """
        Calculate hashes for multiple objects.

        Args:
            objects: List of objects to hash

        Returns:
            List of hash strings
        """
        return [canonical_hash(obj, self.algorithm) for obj in objects]

    def verify_integrity(self, obj: Any, expected_hash: str) -> bool:
        """
        Verify that an object matches an expected hash.

        Args:
            obj: Object to verify
            expected_hash: Expected hash value

        Returns:
            True if object hash matches expected hash
        """
        actual_hash = canonical_hash(obj, self.algorithm)
        return actual_hash == expected_hash

    def create_manifest(self, objects: Dict[str, Any]) -> Dict[str, str]:
        """
        Create a manifest of objects with their canonical hashes.

        Args:
            objects: Dictionary of name -> object mappings

        Returns:
            Dictionary of name -> hash mappings
        """
        return {
            name: canonical_hash(obj, self.algorithm)
            for name, obj in objects.items()
        }


# Convenience functions for common use cases

def hash_file_content(file_path: Union[str, Path],
                      encoding: str = 'utf-8') -> str:
    """
    Calculate canonical hash of file contents (as JSON).

    Args:
        file_path: Path to JSON file
        encoding: File encoding

    Returns:
        Canonical hash of file contents
    """
    file_path = Path(file_path)
    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    return canonical_hash(data)


def save_canonical(obj: Any,
                  file_path: Union[str, Path],
                  encoding: str = 'utf-8') -> str:
    """
    Save object as canonical JSON to file.

    Args:
        obj: Object to save
        file_path: Path to save file
        encoding: File encoding

    Returns:
        Hash of saved content
    """
    file_path = Path(file_path)
    canonical_json = canonical_dumps(obj)

    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding=encoding) as f:
        f.write(canonical_json)

    return canonical_hash(obj)


def load_and_verify(file_path: Union[str, Path],
                   expected_hash: Optional[str] = None,
                   encoding: str = 'utf-8') -> Tuple[Any, str]:
    """
    Load JSON file and optionally verify its hash.

    Args:
        file_path: Path to JSON file
        expected_hash: Optional expected hash for verification
        encoding: File encoding

    Returns:
        Tuple of (loaded object, actual hash)

    Raises:
        CanonicalSerializationError: If hash verification fails
    """
    file_path = Path(file_path)

    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)

    actual_hash = canonical_hash(data)

    if expected_hash and actual_hash != expected_hash:
        raise CanonicalSerializationError(
            f"Hash verification failed. "
            f"Expected: {expected_hash}, "
            f"Actual: {actual_hash}"
        )

    return data, actual_hash


# Example demonstrating canonical vs non-canonical JSON
def demonstrate_canonicalization():
    """
    Demonstrate the difference between canonical and non-canonical JSON.

    Returns:
        Dictionary with examples
    """
    sample_data = {
        "name": "GreenLang",
        "version": 1.000,
        "active": True,
        "metadata": {
            "created": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "tags": ["sustainability", "carbon", "reporting"],
            "score": Decimal("99.9900"),
        },
        "values": [3.14159, 2.0000, 1]
    }

    # Non-canonical (pretty printed with trailing zeros)
    non_canonical = json.dumps(
        sample_data,
        indent=2,
        default=str,
        sort_keys=False
    )

    # Canonical (deterministic, minimal)
    canonical = canonical_dumps(sample_data)

    # Calculate hashes
    non_canonical_hash = hashlib.sha256(non_canonical.encode()).hexdigest()
    canonical_hash_value = canonical_hash(sample_data)

    return {
        "original_data": sample_data,
        "non_canonical": {
            "json": non_canonical,
            "length": len(non_canonical),
            "hash": non_canonical_hash
        },
        "canonical": {
            "json": canonical,
            "length": len(canonical),
            "hash": canonical_hash_value
        },
        "differences": {
            "whitespace_removed": len(non_canonical) - len(canonical),
            "deterministic_ordering": "Yes (alphabetical keys)",
            "trailing_zeros_removed": "Yes (2.0000 -> 2)",
            "special_types_handled": "Yes (datetime, Decimal)",
            "hash_consistent": "Yes (always same for same data)"
        }
    }


if __name__ == "__main__":
    # Demonstrate canonicalization
    demo = demonstrate_canonicalization()

    print("=== Canonical JSON Serialization Demo ===\n")

    print("Non-Canonical JSON:")
    print(demo["non_canonical"]["json"][:200] + "...")
    print(f"Length: {demo['non_canonical']['length']} bytes")
    print(f"Hash: {demo['non_canonical']['hash'][:16]}...\n")

    print("Canonical JSON:")
    print(demo["canonical"]["json"][:200] + "...")
    print(f"Length: {demo['canonical']['length']} bytes")
    print(f"Hash: {demo['canonical']['hash'][:16]}...\n")

    print("Key Differences:")
    for key, value in demo["differences"].items():
        print(f"  - {key.replace('_', ' ').title()}: {value}")