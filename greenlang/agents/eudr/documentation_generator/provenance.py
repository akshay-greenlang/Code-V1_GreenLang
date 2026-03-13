# -*- coding: utf-8 -*-
"""
Provenance Tracker - AGENT-EUDR-030: Documentation Generator

SHA-256 provenance hash chain for deterministic, auditable data lineage
tracking across all documentation generation, Article 9 assembly,
compliance package compilation, and DDS submission operations. Every
document generation, validation, versioning, and submission step is
hash-chained for integrity verification per EUDR Article 31
record-keeping requirements.

Zero-Hallucination:
    - All hashes are computed from deterministic JSON serialization
    - No LLM involvement in provenance computation
    - Hash chain verifiable by any third party

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-030 (GL-EUDR-DGN-030)
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
    and builds verifiable hash chains across documentation processing
    steps.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> h1 = tracker.compute_hash({"key": "value"})
        >>> assert len(h1) == 64
        >>> entry = tracker.create_entry("generate_dds", "eudr_029", GENESIS_HASH, h1)
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

        Serializes data to canonical JSON (sorted keys, compact separators,
        str default) and computes SHA-256 digest.

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
        actor: str = "AGENT-EUDR-030",
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Create a provenance chain entry.

        Args:
            step: Processing step name (e.g., "generate_dds",
                  "assemble_article9", "build_package", "submit_dds").
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

    def verify_chain(self, entries: List[Dict[str, Any]]) -> bool:
        """Verify integrity of a provenance hash chain.

        Each entry's input_hash should match the previous entry's
        output_hash (except the first entry which should reference
        the genesis hash or be standalone).

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
                    i,
                    prev_output,
                    curr_input,
                )
                return False

        return True

    def build_chain(
        self,
        steps: List[Dict[str, Any]],
        genesis_hash: str = GENESIS_HASH,
    ) -> List[Dict[str, Any]]:
        """Build a complete provenance chain from step definitions.

        Each step dict should contain: step, source, data.
        The chain is built by computing hashes sequentially.

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
        """Return the current accumulated chain.

        Returns:
            List of provenance entries accumulated so far.
        """
        return list(self._chain)

    def reset(self) -> None:
        """Clear the accumulated chain."""
        self._chain.clear()
