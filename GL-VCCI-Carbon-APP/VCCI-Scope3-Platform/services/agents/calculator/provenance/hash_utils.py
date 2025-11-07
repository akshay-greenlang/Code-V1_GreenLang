"""
Hash utilities for provenance tracking.
"""

import hashlib
import json
from typing import Any, Dict


def hash_data(data: Any) -> str:
    """
    Generate SHA256 hash of data.

    Args:
        data: Data to hash (will be JSON serialized)

    Returns:
        SHA256 hash as hex string
    """
    if isinstance(data, dict):
        # Sort keys for consistent hashing
        json_str = json.dumps(data, sort_keys=True, default=str)
    else:
        json_str = json.dumps(data, default=str)

    return hashlib.sha256(json_str.encode()).hexdigest()


def hash_factor_info(value: float, source: str) -> str:
    """
    Generate hash for emission factor info.

    Args:
        value: Factor value
        source: Factor source

    Returns:
        SHA256 hash
    """
    data = {
        "value": value,
        "source": source
    }
    return hash_data(data)


__all__ = ["hash_data", "hash_factor_info"]
