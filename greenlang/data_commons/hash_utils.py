# -*- coding: utf-8 -*-
"""
Deterministic Hashing Utilities for GreenLang Data Agents

Eliminates duplicate ``_compute_hash()`` and ``_deterministic_id()`` functions
that are copy-pasted across 30+ modules in the data-layer agent codebase.

The following identical function appears in at least these files:
  - ``data_gateway/schema_translator.py``    _compute_hash()
  - ``data_gateway/response_aggregator.py``  _compute_hash()
  - ``duplicate_detector/setup.py``          _compute_hash()
  - ``gis_connector/geocoder.py``            _compute_hash()
  - ``gis_connector/layer_manager.py``       _compute_hash()
  - ``gis_connector/land_cover.py``          _compute_hash()
  - ``gis_connector/spatial_analyzer.py``    _compute_hash()
  - ``excel_normalizer/setup.py``            _compute_hash()
  - ``pdf_extractor/setup.py``               _compute_hash()
  - ``data_quality_profiler/setup.py``       _compute_hash()
  - ``spend_categorizer/setup.py``           _compute_hash()
  - ``erp_connector/connection_manager.py``  _deterministic_id()
  - ``erp_connector/provenance.py``          build_hash()
  - ``excel_normalizer/provenance.py``       build_hash()
  - ``pdf_extractor/provenance.py``          build_hash()
  - (and many more)

This module provides three deterministic, pure functions:

1.  **compute_hash(data)** -- SHA-256 of arbitrary data (Pydantic model,
    dict, list, or fallback ``str()``).
2.  **deterministic_id(prefix, *parts)** -- short hex ID from a prefix +
    concatenated parts.
3.  **file_hash(file_path)** -- streaming SHA-256 of file contents.

All functions are stateless, side-effect-free, and safe for concurrent use.

Zero-Hallucination Guarantees:
    - All hashes use deterministic SHA-256.
    - JSON serialisation uses ``sort_keys=True`` and ``default=str`` for
      reproducibility across invocations.
    - No randomness or non-deterministic inputs.

Example:
    >>> from greenlang.data_commons.hash_utils import compute_hash, deterministic_id
    >>> compute_hash({"foo": 1, "bar": 2})
    'e3...'   # 64-char hex SHA-256
    >>> deterministic_id("conn", "sap", "us-east-1", "prod")
    'conn-a1b2c3d4e5f6'

Author: GreenLang Platform Team
Date: April 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)

# Default buffer size for streaming file hashing (64 KiB).
_FILE_HASH_BUFFER_SIZE: int = 65_536


# ---------------------------------------------------------------------------
# compute_hash -- canonical SHA-256 of arbitrary data
# ---------------------------------------------------------------------------


def compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Handles Pydantic v2 models (via ``model_dump``), plain dicts, lists, and
    falls back to ``str(data)`` for all other types.  JSON serialisation uses
    ``sort_keys=True`` and ``default=str`` so that the output is identical
    regardless of key insertion order or non-JSON-native types (e.g.
    ``datetime``, ``Decimal``, ``UUID``).

    Args:
        data: Data to hash.  Accepted types include:
            - Pydantic ``BaseModel`` / ``GreenLangBase`` instances
            - ``dict`` or ``list``
            - Any object with a meaningful ``__str__``

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.

    Example:
        >>> compute_hash({"b": 2, "a": 1})
        '...'
        >>> compute_hash({"a": 1, "b": 2})  # same hash (sorted keys)
        '...'
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, (dict, list)):
        serializable = data
    else:
        serializable = str(data)

    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# deterministic_id -- short hex identifier from a prefix + parts
# ---------------------------------------------------------------------------


def deterministic_id(prefix: str, *parts: str, length: int = 12) -> str:
    """Generate a deterministic short ID from a prefix and concatenated parts.

    Hashes the joined parts with SHA-256 and takes the first *length*
    hexadecimal characters, then prepends the prefix with a hyphen separator.

    This replaces the ``_deterministic_id()`` helper found in modules like
    ``erp_connector/connection_manager.py``.

    Args:
        prefix: Human-readable prefix (e.g. ``"conn"``, ``"sync"``, ``"job"``).
        *parts: One or more string components that together uniquely identify
            the entity (e.g. host, port, system type).
        length: Number of hex characters to take from the digest.
            Defaults to 12 (48 bits of entropy).  Must be between 4 and 64.

    Returns:
        A string of the form ``"{prefix}-{hex_chars}"``.

    Raises:
        ValueError: If no parts are provided or length is out of range.

    Example:
        >>> deterministic_id("conn", "sap", "us-east-1", "443")
        'conn-...'
        >>> deterministic_id("job", "batch-2024-01-15", length=8)
        'job-...'
    """
    if not parts:
        raise ValueError("deterministic_id requires at least one part")
    if not 4 <= length <= 64:
        raise ValueError(f"length must be between 4 and 64, got {length}")

    seed = "|".join(parts)
    hex_digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"{prefix}-{hex_digest[:length]}"


# ---------------------------------------------------------------------------
# file_hash -- streaming SHA-256 of file contents
# ---------------------------------------------------------------------------


def file_hash(
    file_path: Union[str, Path],
    *,
    buffer_size: int = _FILE_HASH_BUFFER_SIZE,
) -> str:
    """Compute SHA-256 hash of a file's contents using streaming reads.

    Reads the file in fixed-size chunks so that arbitrarily large files can
    be hashed without loading them entirely into memory.

    Args:
        file_path: Filesystem path to the file (string or ``pathlib.Path``).
        buffer_size: Read buffer size in bytes.  Defaults to 64 KiB.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
        IsADirectoryError: If the path points to a directory.

    Example:
        >>> file_hash("/data/invoice.pdf")
        '5d41...'
    """
    path = Path(file_path)
    hasher = hashlib.sha256()

    with path.open("rb") as fh:
        while True:
            chunk = fh.read(buffer_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# build_hash -- alias matching the ProvenanceTracker.build_hash signature
# ---------------------------------------------------------------------------


def build_hash(data: Any) -> str:
    """Build a SHA-256 hash for arbitrary data (dict, list, or other).

    This is a convenience alias that matches the signature of
    ``ProvenanceTracker.build_hash()`` for callers that need a standalone
    function without a tracker instance.  Internally delegates to
    :func:`compute_hash` after normalising the input to a JSON-safe form.

    Unlike :func:`compute_hash`, this function always serialises via
    ``json.dumps(data, sort_keys=True, default=str)`` without special
    Pydantic ``model_dump()`` handling, matching the exact behaviour of
    the ``ProvenanceTracker.build_hash()`` method.

    Args:
        data: Data to hash (dict, list, or other).

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "compute_hash",
    "deterministic_id",
    "file_hash",
    "build_hash",
]
