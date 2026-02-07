# -*- coding: utf-8 -*-
"""
Deterministic Scheduling - AGENT-FOUND-001: GreenLang DAG Orchestrator

Deterministic scheduling utilities for reproducible DAG execution:
- Sorted node ordering within parallel levels
- Deterministic execution ID generation
- Execution replay verification

Integrates with:
- greenlang.utilities.determinism.clock.DeterministicClock
- greenlang.utilities.determinism.uuid.deterministic_uuid

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, List, Optional

from greenlang.orchestrator.models import ExecutionTrace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional determinism imports
# ---------------------------------------------------------------------------

try:
    from greenlang.utilities.determinism.clock import DeterministicClock
    _CLOCK_AVAILABLE = True
except ImportError:
    DeterministicClock = None  # type: ignore[assignment, misc]
    _CLOCK_AVAILABLE = False

try:
    from greenlang.utilities.determinism.uuid import (
        deterministic_uuid as _det_uuid,
        content_hash as _content_hash,
    )
    _UUID_AVAILABLE = True
except ImportError:
    _det_uuid = None  # type: ignore[assignment]
    _content_hash = None  # type: ignore[assignment]
    _UUID_AVAILABLE = False


# ===================================================================
# DeterministicScheduler
# ===================================================================


class DeterministicScheduler:
    """Provides deterministic scheduling utilities for DAG execution.

    All methods are stateless and produce identical outputs for identical
    inputs, ensuring execution replay produces the same ordering.
    """

    @staticmethod
    def sort_nodes(nodes: List[str]) -> List[str]:
        """Sort nodes alphabetically for deterministic parallel scheduling.

        This ensures that within a given parallel execution level, nodes
        are always processed in the same order regardless of runtime
        conditions.

        Args:
            nodes: List of node IDs to sort.

        Returns:
            Sorted list of node IDs.
        """
        return sorted(nodes)

    @staticmethod
    def generate_execution_id(
        dag_id: str,
        input_hash: str,
        seed: Optional[str] = None,
    ) -> str:
        """Generate a deterministic execution ID.

        Given the same dag_id, input_hash, and seed, this always
        produces the same execution ID.

        Args:
            dag_id: DAG identifier.
            input_hash: SHA-256 hash of the input data.
            seed: Optional additional seed for uniqueness (e.g.,
                timestamp string).

        Returns:
            Deterministic execution ID string.
        """
        if _UUID_AVAILABLE and _det_uuid is not None:
            namespace = f"orchestrator:{dag_id}"
            name = f"{input_hash}:{seed or 'default'}"
            return _det_uuid(namespace, name)

        # Fallback: manual SHA-256 based ID
        combined = f"{dag_id}:{input_hash}:{seed or 'default'}"
        hash_hex = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return f"exec_{hash_hex}"

    @staticmethod
    def compute_input_hash(input_data: Dict) -> str:
        """Compute deterministic hash of input data.

        Args:
            input_data: Input dictionary to hash.

        Returns:
            SHA-256 hex digest string.
        """
        if _UUID_AVAILABLE and _content_hash is not None:
            return _content_hash(input_data)

        # Fallback
        content = json.dumps(input_data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def verify_replay(
        trace1: ExecutionTrace,
        trace2: ExecutionTrace,
    ) -> bool:
        """Compare two execution traces for identical ordering.

        Verifies that both executions processed nodes in the same
        topological order and produced the same level groupings.

        Args:
            trace1: First execution trace.
            trace2: Second execution trace.

        Returns:
            True if both traces have identical execution ordering.
        """
        # Compare topology levels
        if trace1.topology_levels != trace2.topology_levels:
            logger.warning(
                "Replay mismatch: topology_levels differ "
                "(%d levels vs %d levels)",
                len(trace1.topology_levels),
                len(trace2.topology_levels),
            )
            return False

        # Compare node execution order within each level
        for i, (level1, level2) in enumerate(
            zip(trace1.topology_levels, trace2.topology_levels)
        ):
            if sorted(level1) != sorted(level2):
                logger.warning(
                    "Replay mismatch at level %d: %s vs %s",
                    i, level1, level2,
                )
                return False

        # Compare node statuses
        for node_id in trace1.node_traces:
            if node_id not in trace2.node_traces:
                logger.warning(
                    "Replay mismatch: node '%s' missing from trace2",
                    node_id,
                )
                return False
            t1 = trace1.node_traces[node_id]
            t2 = trace2.node_traces[node_id]
            if t1.status != t2.status:
                logger.warning(
                    "Replay mismatch: node '%s' status %s vs %s",
                    node_id, t1.status, t2.status,
                )
                return False

        logger.debug(
            "Replay verification passed for executions %s and %s",
            trace1.execution_id, trace2.execution_id,
        )
        return True


__all__ = [
    "DeterministicScheduler",
]
