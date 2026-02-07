# -*- coding: utf-8 -*-
"""
Provenance Tracker - AGENT-FOUND-001: GreenLang DAG Orchestrator

Execution provenance tracking with SHA-256 hash chains for regulatory
audit. Records per-node provenance and builds a chain hash linking all
node executions in topological order.

Key Features:
- Per-node provenance records (input_hash, output_hash, timing)
- Parent hash linking (predecessor provenance hashes)
- Chain hash: SHA-256 linking all provenances in execution order
- JSON export for audit compliance
- Chain integrity verification

Uses DeterministicClock for all timestamps.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from greenlang.orchestrator.models import NodeProvenance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional deterministic clock
# ---------------------------------------------------------------------------

try:
    from greenlang.utilities.determinism.clock import DeterministicClock
except ImportError:
    DeterministicClock = None  # type: ignore[assignment, misc]


# ===================================================================
# ProvenanceTracker
# ===================================================================


class ProvenanceTracker:
    """Tracks execution provenance for DAG runs.

    Maintains per-execution provenance records and provides chain
    hash computation, export, and verification.

    Attributes:
        _traces: Mapping from execution_id to list of NodeProvenance.
    """

    def __init__(self) -> None:
        """Initialize empty provenance tracker."""
        self._traces: Dict[str, List[NodeProvenance]] = {}
        logger.debug("ProvenanceTracker initialized")

    def record_node(
        self,
        execution_id: str,
        node_id: str,
        input_data: Any,
        output_data: Any,
        duration_ms: float,
        attempt_count: int,
        parent_node_ids: Optional[List[str]] = None,
    ) -> NodeProvenance:
        """Record provenance for a single node execution.

        Args:
            execution_id: Parent execution identifier.
            node_id: Node that was executed.
            input_data: Input data (will be hashed).
            output_data: Output data (will be hashed).
            duration_ms: Execution duration in milliseconds.
            attempt_count: Number of attempts.
            parent_node_ids: IDs of predecessor nodes for parent hash
                linking.

        Returns:
            NodeProvenance record with calculated hashes.
        """
        # Compute input hash
        input_hash = _compute_hash(input_data)

        # Compute output hash
        output_hash = _compute_hash(output_data)

        # Resolve parent hashes
        parent_hashes: List[str] = []
        if parent_node_ids:
            exec_provenances = self._traces.get(execution_id, [])
            prov_map = {p.node_id: p for p in exec_provenances}
            for pid in sorted(parent_node_ids):
                parent_prov = prov_map.get(pid)
                if parent_prov and parent_prov.chain_hash:
                    parent_hashes.append(parent_prov.chain_hash)

        # Create provenance record
        prov = NodeProvenance(
            node_id=node_id,
            input_hash=input_hash,
            output_hash=output_hash,
            duration_ms=duration_ms,
            attempt_count=attempt_count,
            parent_hashes=parent_hashes,
        )
        prov.chain_hash = prov.calculate_chain_hash()

        # Store
        if execution_id not in self._traces:
            self._traces[execution_id] = []
        self._traces[execution_id].append(prov)

        logger.debug(
            "Recorded provenance: execution=%s node=%s chain=%s",
            execution_id, node_id, prov.chain_hash[:16],
        )
        return prov

    def build_chain_hash(self, execution_id: str) -> str:
        """Build a single chain hash over all node provenances.

        Combines all node chain hashes in order to produce a single
        SHA-256 hash representing the complete execution provenance.

        Args:
            execution_id: Execution identifier.

        Returns:
            SHA-256 hex string of the chain hash.
        """
        provenances = self._traces.get(execution_id, [])
        if not provenances:
            return hashlib.sha256(b"empty").hexdigest()

        combined = ":".join(p.chain_hash for p in provenances)
        chain_hash = hashlib.sha256(combined.encode()).hexdigest()

        logger.debug(
            "Built chain hash for execution=%s: %s (%d nodes)",
            execution_id, chain_hash[:16], len(provenances),
        )
        return chain_hash

    def get_trace(self, execution_id: str) -> List[NodeProvenance]:
        """Get all provenance records for an execution.

        Args:
            execution_id: Execution identifier.

        Returns:
            List of NodeProvenance records in recording order.
        """
        return list(self._traces.get(execution_id, []))

    def export_json(self, execution_id: str) -> str:
        """Export full provenance as JSON for regulatory audit.

        Args:
            execution_id: Execution identifier.

        Returns:
            JSON string with all provenance records and chain hash.
        """
        provenances = self.get_trace(execution_id)
        chain_hash = self.build_chain_hash(execution_id)

        export_data = {
            "execution_id": execution_id,
            "chain_hash": chain_hash,
            "node_count": len(provenances),
            "provenances": [p.to_dict() for p in provenances],
        }
        return json.dumps(export_data, indent=2, default=str)

    def verify_chain(self, execution_id: str) -> bool:
        """Verify the integrity of the provenance chain.

        Recalculates each node's chain hash and verifies it matches
        the stored value.

        Args:
            execution_id: Execution identifier.

        Returns:
            True if all chain hashes are valid.
        """
        provenances = self._traces.get(execution_id, [])
        for prov in provenances:
            expected = prov.calculate_chain_hash()
            if expected != prov.chain_hash:
                logger.error(
                    "Provenance chain verification failed: "
                    "execution=%s node=%s expected=%s actual=%s",
                    execution_id, prov.node_id,
                    expected, prov.chain_hash,
                )
                return False
        return True

    def clear(self, execution_id: Optional[str] = None) -> None:
        """Clear provenance records.

        Args:
            execution_id: Clear specific execution, or all if None.
        """
        if execution_id:
            self._traces.pop(execution_id, None)
        else:
            self._traces.clear()


# ===================================================================
# Hash utility
# ===================================================================


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or other).

    Returns:
        SHA-256 hex digest string.
    """
    if isinstance(data, (dict, list)):
        content = json.dumps(data, sort_keys=True, default=str)
    elif isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    else:
        content = str(data)
    return hashlib.sha256(content.encode()).hexdigest()


__all__ = [
    "ProvenanceTracker",
]
