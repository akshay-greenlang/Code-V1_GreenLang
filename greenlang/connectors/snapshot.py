# -*- coding: utf-8 -*-
"""
Connector Snapshot Manager - Deterministic Data Caching
=======================================================

Reuses existing determinism infrastructure from:
- greenlang/intelligence/rag/hashing.py (canonical hashing)
- greenlang/intelligence/determinism.py (cache backends)

Provides byte-exact, reproducible snapshots for connector data.

Key Features:
- Canonical JSON serialization (sorted keys, UTF-8, no whitespace)
- SHA-256 content addressing
- Decimal→string for numeric stability
- UTC datetime normalization
- Multiple backends (JSON for dev/test, SQLite for production)

Design:
- write_canonical_snapshot() → bytes (deterministic)
- read_canonical_snapshot() → (payload, provenance)
- Snapshot ID = SHA-256(canonical_bytes)
- File naming: {connector_id}_{query_hash[:8]}_{snapshot_id[:8]}.snap.json
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type
from decimal import Decimal
from datetime import datetime, timezone
from pydantic import BaseModel

# Reuse existing hashing infrastructure!
from greenlang.intelligence.rag.hashing import (
    sha256_str,
    sha256_bytes,
    canonicalize_text
)

from greenlang.connectors.errors import (
    ConnectorError,
    ConnectorSnapshotNotFound,
    ConnectorSnapshotCorrupt
)


def _canonicalize(obj: Any) -> Any:
    """
    Canonicalize object for deterministic serialization

    Handles:
    - Decimal → string (for precision)
    - Datetime → ISO 8601 UTC
    - Dict → sorted keys
    - List → preserved order
    - BaseModel → dict

    Args:
        obj: Object to canonicalize

    Returns:
        Canonicalized object (JSON-serializable)
    """
    # Decimal to string (stable representation)
    if isinstance(obj, Decimal):
        # Normalize: remove trailing zeros, use 'f' format
        normalized = obj.normalize()
        return format(normalized, 'f')

    # Datetime to ISO 8601 UTC
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            # Assume UTC if naive
            obj = obj.replace(tzinfo=timezone.utc)
        # Convert to UTC and format
        utc_dt = obj.astimezone(timezone.utc)
        # Use 'Z' suffix for UTC (not +00:00)
        return utc_dt.isoformat().replace("+00:00", "Z")

    # Pydantic model to dict
    if isinstance(obj, BaseModel):
        return _canonicalize(obj.model_dump())

    # Dict: sort keys recursively
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys())}

    # List: preserve order, canonicalize elements
    if isinstance(obj, list):
        return [_canonicalize(x) for x in obj]

    # Primitive types: return as-is
    return obj


def write_canonical_snapshot(
    connector_id: str,
    connector_version: str,
    payload: BaseModel,
    provenance: BaseModel
) -> bytes:
    """
    Write canonical snapshot bytes

    Produces deterministic, byte-exact output:
    1. Canonicalize payload and provenance
    2. Build snapshot body with sorted keys
    3. Compute SHA-256 snapshot ID
    4. Update provenance with snapshot ID
    5. Re-serialize with snapshot ID (still deterministic)

    Args:
        connector_id: Connector identifier
        connector_version: Connector version
        payload: Data payload (Pydantic model)
        provenance: Provenance metadata (Pydantic model)

    Returns:
        Canonical snapshot bytes (UTF-8 JSON)
    """
    # Build snapshot body
    body = {
        "connector_id": connector_id,
        "connector_version": connector_version,
        "payload": _canonicalize(payload),
        "provenance": _canonicalize(provenance.model_dump(exclude={"snapshot_id"})),
        "schema": {
            "payload_model": payload.__class__.__name__,
            "provenance_model": provenance.__class__.__name__,
        }
    }

    # Compute snapshot ID (without snapshot_id in provenance)
    raw = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    ).encode("utf-8")

    snapshot_id = sha256_bytes(raw)

    # Add snapshot ID to provenance
    body["provenance"]["snapshot_id"] = snapshot_id

    # Re-serialize with snapshot ID (deterministic)
    final_raw = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    ).encode("utf-8")

    return final_raw


def read_canonical_snapshot(
    raw: bytes,
    payload_type: Optional[Type[BaseModel]] = None
) -> Dict[str, Any]:
    """
    Read canonical snapshot

    Args:
        raw: Snapshot bytes
        payload_type: Expected payload type (for validation)

    Returns:
        Dict with 'payload', 'provenance', 'connector_id', etc.

    Raises:
        ConnectorSnapshotCorrupt: If snapshot is invalid
    """
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ConnectorSnapshotCorrupt(
            f"Failed to decode snapshot: {e}",
            connector="unknown",
            original_error=e
        )

    # Validate structure
    required_keys = ["connector_id", "connector_version", "payload", "provenance", "schema"]
    for key in required_keys:
        if key not in data:
            raise ConnectorSnapshotCorrupt(
                f"Missing required field: {key}",
                connector=data.get("connector_id", "unknown"),
                context={"available_keys": list(data.keys())}
            )

    # Return raw data (connector will reconstruct Pydantic models)
    return data


def save_snapshot(
    snapshot_bytes: bytes,
    connector_id: str,
    query_hash: str,
    output_dir: Path = Path(".greenlang/snapshots")
) -> Path:
    """
    Save snapshot to file

    Args:
        snapshot_bytes: Canonical snapshot bytes
        connector_id: Connector ID
        query_hash: Query hash (for filename)
        output_dir: Output directory

    Returns:
        Path to saved snapshot file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute snapshot ID
    snapshot_id = sha256_bytes(snapshot_bytes)

    # Generate filename
    safe_connector_id = connector_id.replace("/", "_")
    filename = f"{safe_connector_id}_{query_hash[:8]}_{snapshot_id[:8]}.snap.json"
    filepath = output_dir / filename

    # Atomic write (temp + rename)
    temp_path = filepath.with_suffix(".tmp")
    temp_path.write_bytes(snapshot_bytes)
    temp_path.rename(filepath)

    return filepath


def load_snapshot(snapshot_path: Path) -> bytes:
    """
    Load snapshot from file

    Args:
        snapshot_path: Path to snapshot file

    Returns:
        Snapshot bytes

    Raises:
        ConnectorSnapshotNotFound: If file doesn't exist
        ConnectorSnapshotCorrupt: If file is corrupted
    """
    if not snapshot_path.exists():
        raise ConnectorSnapshotNotFound(
            f"Snapshot file not found",
            connector="unknown",
            snapshot_path=str(snapshot_path)
        )

    try:
        return snapshot_path.read_bytes()
    except Exception as e:
        raise ConnectorSnapshotCorrupt(
            f"Failed to read snapshot: {e}",
            connector="unknown",
            original_error=e,
            context={"path": str(snapshot_path)}
        )


def verify_snapshot_integrity(snapshot_bytes: bytes) -> bool:
    """
    Verify snapshot integrity

    Checks:
    1. Valid JSON
    2. Required fields present
    3. Snapshot ID matches computed hash

    Args:
        snapshot_bytes: Snapshot bytes

    Returns:
        True if valid, False otherwise
    """
    try:
        data = read_canonical_snapshot(snapshot_bytes)

        # Recompute snapshot ID (exclude snapshot_id from provenance)
        body = {
            "connector_id": data["connector_id"],
            "connector_version": data["connector_version"],
            "payload": data["payload"],
            "provenance": {k: v for k, v in data["provenance"].items() if k != "snapshot_id"},
            "schema": data["schema"]
        }

        raw = json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True
        ).encode("utf-8")

        computed_id = sha256_bytes(raw)
        stored_id = data["provenance"].get("snapshot_id")

        return computed_id == stored_id

    except Exception:
        return False


# Export canonical query hash function for connectors
def compute_query_hash(query: BaseModel) -> str:
    """
    Compute canonical hash of query

    Uses same canonicalization as snapshots for consistency.

    Args:
        query: Query object (Pydantic model)

    Returns:
        SHA-256 hash (hex)
    """
    canonical = _canonicalize(query)
    json_str = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    )
    return sha256_str(json_str)


def compute_schema_hash(model_class: Type[BaseModel]) -> str:
    """
    Compute hash of Pydantic model schema

    Used for provenance to track schema versions.

    Args:
        model_class: Pydantic model class

    Returns:
        SHA-256 hash of schema (hex)
    """
    schema = model_class.schema()
    schema_str = json.dumps(
        schema,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    )
    return sha256_str(schema_str)
