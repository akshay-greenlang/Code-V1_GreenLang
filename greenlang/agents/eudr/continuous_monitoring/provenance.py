# -*- coding: utf-8 -*-
"""
Provenance Tracker - AGENT-EUDR-033: Continuous Monitoring Agent

SHA-256 provenance hash chain for deterministic, auditable data lineage
tracking across all continuous monitoring operations. Every supply chain
scan, deforestation correlation, compliance audit, change detection,
risk score monitor, freshness validation, and regulatory tracking event
is hash-chained for integrity verification per EUDR Article 31
record-keeping requirements.

Zero-Hallucination:
    - All hashes are computed from deterministic JSON serialization
    - No LLM involvement in provenance computation
    - Hash chain verifiable by any third party

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-033 (GL-EUDR-CM-033)
Status: Production Ready
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


class ProvenanceTracker:
    """SHA-256 provenance hash chain tracker.

    Computes deterministic hashes from canonical JSON representations
    and builds verifiable hash chains across continuous monitoring
    processing steps.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> h1 = tracker.compute_hash({"key": "value"})
        >>> assert len(h1) == 64
        >>> entry = tracker.create_entry(
        ...     "supply_chain_scan", "eudr_033", GENESIS_HASH, h1
        ... )
        >>> assert entry["output_hash"] == h1
    """

    def __init__(self, algorithm: str = "sha256") -> None:
        """Initialize provenance tracker.

        Args:
            algorithm: Hash algorithm (only sha256 supported).

        Raises:
            ValueError: If an unsupported algorithm is specified.
        """
        if algorithm != "sha256":
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'sha256'.")
        self._algorithm = algorithm
        self._chain: List[Dict[str, Any]] = []
        logger.debug("ProvenanceTracker initialized with %s", algorithm)

    @staticmethod
    def compute_hash(data: Dict[str, Any]) -> str:
        """Compute deterministic SHA-256 hash of data.

        Args:
            data: Dictionary to hash.

        Returns:
            64-character lowercase hex SHA-256 hash string.
        """
        canonical = json.dumps(
            data, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_hash_bytes(data: bytes) -> str:
        """Compute SHA-256 hash of raw bytes.

        Args:
            data: Bytes to hash.

        Returns:
            64-character lowercase hex SHA-256 hash string.
        """
        return hashlib.sha256(data).hexdigest()

    def create_entry(
        self,
        step: str,
        source: str,
        input_hash: str,
        output_hash: str,
        actor: str = "AGENT-EUDR-033",
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Create a provenance chain entry.

        Args:
            step: Processing step name.
            source: Data source identifier.
            input_hash: Hash of input data.
            output_hash: Hash of output data.
            actor: Agent or system performing the step.
            timestamp: Optional timestamp (defaults to UTC now).

        Returns:
            Provenance entry dictionary.
        """
        ts = timestamp or datetime.now(timezone.utc).replace(microsecond=0)
        entry = {
            "step": step,
            "source": source,
            "timestamp": ts.isoformat(),
            "actor": actor,
            "input_hash": input_hash,
            "output_hash": output_hash,
        }
        self._chain.append(entry)
        return entry

    def record(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        actor: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Record a provenance event with entity context.

        Args:
            entity_type: Type of entity (scan, audit, detection, etc.).
            action: Action performed.
            entity_id: Entity identifier.
            actor: Actor performing the action.
            metadata: Additional metadata to include.
            timestamp: Optional timestamp.

        Returns:
            Provenance entry dictionary.
        """
        ts = timestamp or datetime.now(timezone.utc).replace(microsecond=0)
        data = {
            "entity_type": entity_type,
            "action": action,
            "entity_id": entity_id,
            "actor": actor,
            "timestamp": ts.isoformat(),
            **(metadata or {}),
        }
        output_hash = self.compute_hash(data)
        prev_hash = self._chain[-1]["output_hash"] if self._chain else GENESIS_HASH

        entry = {
            "step": f"{entity_type}:{action}",
            "source": entity_id,
            "timestamp": ts.isoformat(),
            "actor": actor,
            "input_hash": prev_hash,
            "output_hash": output_hash,
            "metadata": metadata or {},
        }
        self._chain.append(entry)
        return entry

    def verify_chain(self, entries: List[Dict[str, Any]]) -> bool:
        """Verify integrity of a provenance hash chain.

        Args:
            entries: Ordered list of provenance entries.

        Returns:
            True if the chain is valid, False otherwise.
        """
        if not entries:
            return True

        for i in range(1, len(entries)):
            prev_output = entries[i - 1].get("output_hash", "")
            curr_input = entries[i].get("input_hash", "")
            if prev_output and curr_input and prev_output != curr_input:
                logger.warning(
                    "Provenance chain broken at step %d: "
                    "expected input_hash=%s, got=%s",
                    i, prev_output, curr_input,
                )
                return False

        return True

    def build_chain(
        self,
        steps: List[Dict[str, Any]],
        genesis_hash: str = GENESIS_HASH,
    ) -> List[Dict[str, Any]]:
        """Build a complete provenance chain from step definitions.

        Args:
            steps: List of step definitions with step/source/data keys.
            genesis_hash: Starting hash for the chain.

        Returns:
            List of provenance entries forming a valid chain.
        """
        chain: List[Dict[str, Any]] = []
        prev_hash = genesis_hash

        for step_def in steps:
            data = step_def.get("data", {})
            output_hash = self.compute_hash(data) if data else prev_hash
            entry = self.create_entry(
                step=step_def.get("step", "unknown"),
                source=step_def.get("source", "unknown"),
                input_hash=prev_hash,
                output_hash=output_hash,
            )
            chain.append(entry)
            prev_hash = output_hash

        return chain

    def get_chain(self) -> List[Dict[str, Any]]:
        """Return the current accumulated chain."""
        return list(self._chain)

    def reset(self) -> None:
        """Clear the accumulated chain."""
        self._chain.clear()
