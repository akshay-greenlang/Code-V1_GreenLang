# -*- coding: utf-8 -*-
"""
Conversion Provenance Tracker - AGENT-FOUND-003: Unit & Reference Normalizer

Provides SHA-256 based audit trail tracking for all conversion and
entity resolution operations. Maintains an in-memory operation log
with chain hashing for tamper evidence.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems

Example:
    >>> from greenlang.normalizer.provenance import ConversionProvenanceTracker
    >>> tracker = ConversionProvenanceTracker()
    >>> prov = tracker.record_conversion(
    ...     {"value": "100", "from": "kWh"},
    ...     {"converted": "0.1"},
    ...     {"factor": "0.001"},
    ... )
    >>> print(prov.chain_hash)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-003 Unit & Reference Normalizer
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from greenlang.normalizer.models import ConversionProvenance

logger = logging.getLogger(__name__)


class ConversionProvenanceTracker:
    """Tracks provenance for conversion and resolution operations.

    Maintains an ordered log of operations with SHA-256 hashes
    that chain together to provide tamper-evident audit trails.

    Attributes:
        _operations: Ordered list of provenance records.
        _last_chain_hash: Most recent chain hash for linking.

    Example:
        >>> tracker = ConversionProvenanceTracker()
        >>> prov = tracker.record_conversion(input_d, output_d, factors)
        >>> trail = tracker.get_audit_trail(prov.operation_id)
    """

    # Initial chain hash (genesis)
    _GENESIS_HASH = hashlib.sha256(b"greenlang-normalizer-genesis").hexdigest()

    def __init__(self) -> None:
        """Initialize ConversionProvenanceTracker."""
        self._operations: List[ConversionProvenance] = []
        self._last_chain_hash: str = self._GENESIS_HASH
        logger.info("ConversionProvenanceTracker initialized")

    def record_conversion(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        factors: Dict[str, Any],
    ) -> ConversionProvenance:
        """Record a conversion operation in the provenance log.

        Args:
            input_data: Input parameters of the conversion.
            output_data: Output result of the conversion.
            factors: Conversion factors or details applied.

        Returns:
            ConversionProvenance record with chain hash.
        """
        operation_id = f"conv-{uuid.uuid4().hex[:12]}"
        input_hash = self._hash_dict(input_data)
        output_hash = self._hash_dict(output_data)
        chain_hash = self._build_next_chain_hash(input_hash, output_hash)

        prov = ConversionProvenance(
            operation_id=operation_id,
            timestamp=datetime.utcnow(),
            input_hash=input_hash,
            output_hash=output_hash,
            chain_hash=chain_hash,
            factors_used=factors,
        )

        self._operations.append(prov)
        self._last_chain_hash = chain_hash

        logger.debug("Recorded conversion provenance: %s", operation_id)
        return prov

    def record_resolution(
        self,
        input_name: str,
        resolved_entity: Dict[str, Any],
    ) -> ConversionProvenance:
        """Record an entity resolution operation in the provenance log.

        Args:
            input_name: Original entity name.
            resolved_entity: Resolved entity details.

        Returns:
            ConversionProvenance record with chain hash.
        """
        operation_id = f"res-{uuid.uuid4().hex[:12]}"
        input_hash = self._hash_dict({"input_name": input_name})
        output_hash = self._hash_dict(resolved_entity)
        chain_hash = self._build_next_chain_hash(input_hash, output_hash)

        prov = ConversionProvenance(
            operation_id=operation_id,
            timestamp=datetime.utcnow(),
            input_hash=input_hash,
            output_hash=output_hash,
            chain_hash=chain_hash,
            factors_used=resolved_entity,
        )

        self._operations.append(prov)
        self._last_chain_hash = chain_hash

        logger.debug("Recorded resolution provenance: %s", operation_id)
        return prov

    def build_chain_hash(self, operations: List[Dict[str, Any]]) -> str:
        """Build a cumulative SHA-256 chain hash over a list of operations.

        Each hash incorporates the previous hash to form a chain.

        Args:
            operations: List of operation dictionaries.

        Returns:
            Final chain hash hex string.
        """
        current = self._GENESIS_HASH
        for op in operations:
            op_hash = self._hash_dict(op)
            combined = f"{current}:{op_hash}"
            current = hashlib.sha256(combined.encode()).hexdigest()
        return current

    def get_audit_trail(
        self, operation_id: Optional[str] = None,
    ) -> List[ConversionProvenance]:
        """Get the audit trail, optionally filtered by operation ID.

        Args:
            operation_id: Optional filter for a specific operation.

        Returns:
            List of ConversionProvenance records.
        """
        if operation_id is None:
            return list(self._operations)
        return [op for op in self._operations if op.operation_id == operation_id]

    def export_json(
        self, operations: Optional[List[ConversionProvenance]] = None,
    ) -> str:
        """Export provenance records as JSON string.

        Args:
            operations: Records to export. Exports all if None.

        Returns:
            JSON string of provenance records.
        """
        ops = operations if operations is not None else self._operations
        records = [op.model_dump(mode="json") for op in ops]
        return json.dumps(records, indent=2, default=str)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_next_chain_hash(
        self, input_hash: str, output_hash: str,
    ) -> str:
        """Build the next chain hash linking to the previous.

        Args:
            input_hash: Hash of the input data.
            output_hash: Hash of the output data.

        Returns:
            New chain hash incorporating the previous.
        """
        combined = f"{self._last_chain_hash}:{input_hash}:{output_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()

    @staticmethod
    def _hash_dict(data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary.

        Args:
            data: Dictionary to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


__all__ = [
    "ConversionProvenanceTracker",
]
