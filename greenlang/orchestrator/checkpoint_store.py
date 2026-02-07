# -*- coding: utf-8 -*-
"""
DAG Checkpoint Store - AGENT-FOUND-001: GreenLang DAG Orchestrator

DAG-aware checkpoint storage for node-level checkpoint/resume.
Provides:
- Abstract base class for pluggable backends
- In-memory implementation (development/testing)
- File-based implementation (staging/production)
- Completed-node tracking for resume
- SHA-256 integrity verification
- Automatic cleanup with retention policy

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

from greenlang.orchestrator.models import DAGCheckpoint, NodeStatus

logger = logging.getLogger(__name__)


# ===================================================================
# Abstract base class
# ===================================================================


class DAGCheckpointStore(ABC):
    """Abstract base class for DAG checkpoint storage backends."""

    @abstractmethod
    def save(self, checkpoint: DAGCheckpoint) -> bool:
        """Save a checkpoint.

        Args:
            checkpoint: DAGCheckpoint to persist.

        Returns:
            True if save succeeded.
        """

    @abstractmethod
    def load(
        self, execution_id: str, node_id: str,
    ) -> Optional[DAGCheckpoint]:
        """Load a specific checkpoint.

        Args:
            execution_id: Execution identifier.
            node_id: Node identifier.

        Returns:
            DAGCheckpoint or None if not found.
        """

    @abstractmethod
    def list_checkpoints(
        self, execution_id: str,
    ) -> List[DAGCheckpoint]:
        """List all checkpoints for an execution.

        Args:
            execution_id: Execution identifier.

        Returns:
            List of checkpoints ordered by creation time.
        """

    @abstractmethod
    def delete(self, execution_id: str) -> int:
        """Delete all checkpoints for an execution.

        Args:
            execution_id: Execution identifier.

        Returns:
            Number of checkpoints deleted.
        """

    @abstractmethod
    def exists(self, execution_id: str, node_id: str) -> bool:
        """Check if a checkpoint exists.

        Args:
            execution_id: Execution identifier.
            node_id: Node identifier.

        Returns:
            True if checkpoint exists.
        """

    def get_completed_nodes(self, execution_id: str) -> Set[str]:
        """Get set of completed node IDs for resume.

        Args:
            execution_id: Execution identifier.

        Returns:
            Set of node IDs with COMPLETED status.
        """
        checkpoints = self.list_checkpoints(execution_id)
        return {
            cp.node_id
            for cp in checkpoints
            if cp.status == NodeStatus.COMPLETED
        }

    def verify_integrity(self, execution_id: str) -> bool:
        """Verify SHA-256 hash integrity of all checkpoints.

        Args:
            execution_id: Execution identifier.

        Returns:
            True if all checkpoints pass integrity check.
        """
        checkpoints = self.list_checkpoints(execution_id)
        for cp in checkpoints:
            if cp.output_hash:
                expected = hashlib.sha256(
                    json.dumps(
                        cp.outputs, sort_keys=True, default=str,
                    ).encode()
                ).hexdigest()
                if expected != cp.output_hash:
                    logger.error(
                        "Checkpoint integrity failure: execution=%s "
                        "node=%s expected=%s actual=%s",
                        execution_id, cp.node_id,
                        expected, cp.output_hash,
                    )
                    return False
        return True

    def cleanup(self, retention_days: int) -> int:
        """Automatic cleanup of old checkpoints.

        Default implementation is a no-op. Override in subclasses.

        Args:
            retention_days: Days to retain checkpoints.

        Returns:
            Number of checkpoints removed.
        """
        return 0


# ===================================================================
# In-memory implementation
# ===================================================================


class MemoryDAGCheckpointStore(DAGCheckpointStore):
    """In-memory checkpoint store for development and testing."""

    def __init__(self) -> None:
        """Initialize empty in-memory store."""
        # execution_id -> {node_id -> DAGCheckpoint}
        self._store: Dict[str, Dict[str, DAGCheckpoint]] = {}
        logger.info("Initialized MemoryDAGCheckpointStore")

    def save(self, checkpoint: DAGCheckpoint) -> bool:
        """Save checkpoint to memory."""
        try:
            exec_id = checkpoint.execution_id
            if exec_id not in self._store:
                self._store[exec_id] = {}
            self._store[exec_id][checkpoint.node_id] = checkpoint
            logger.debug(
                "Saved checkpoint: execution=%s node=%s",
                exec_id, checkpoint.node_id,
            )
            return True
        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)
            return False

    def load(
        self, execution_id: str, node_id: str,
    ) -> Optional[DAGCheckpoint]:
        """Load checkpoint from memory."""
        exec_store = self._store.get(execution_id, {})
        return exec_store.get(node_id)

    def list_checkpoints(self, execution_id: str) -> List[DAGCheckpoint]:
        """List all checkpoints for an execution."""
        exec_store = self._store.get(execution_id, {})
        checkpoints = list(exec_store.values())
        return sorted(checkpoints, key=lambda c: c.node_id)

    def delete(self, execution_id: str) -> int:
        """Delete all checkpoints for an execution."""
        exec_store = self._store.pop(execution_id, {})
        count = len(exec_store)
        if count > 0:
            logger.info(
                "Deleted %d checkpoints for execution=%s",
                count, execution_id,
            )
        return count

    def exists(self, execution_id: str, node_id: str) -> bool:
        """Check if checkpoint exists in memory."""
        return node_id in self._store.get(execution_id, {})

    def cleanup(self, retention_days: int) -> int:
        """Clean up old checkpoints from memory."""
        cutoff = datetime.now() - timedelta(days=retention_days)
        removed = 0
        empty_executions: List[str] = []

        for exec_id, nodes in self._store.items():
            to_remove: List[str] = []
            for nid, cp in nodes.items():
                if cp.created_at and cp.created_at < cutoff:
                    to_remove.append(nid)
            for nid in to_remove:
                del nodes[nid]
                removed += 1
            if not nodes:
                empty_executions.append(exec_id)

        for exec_id in empty_executions:
            del self._store[exec_id]

        if removed > 0:
            logger.info(
                "Cleaned up %d checkpoints older than %d days",
                removed, retention_days,
            )
        return removed


# ===================================================================
# File-based implementation
# ===================================================================


class FileDAGCheckpointStore(DAGCheckpointStore):
    """File-based checkpoint store using JSON files."""

    def __init__(self, base_dir: str) -> None:
        """Initialize file-based store.

        Args:
            base_dir: Directory for checkpoint files.
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Initialized FileDAGCheckpointStore at %s", self._base_dir,
        )

    def _get_exec_dir(self, execution_id: str) -> Path:
        """Get directory for an execution's checkpoints."""
        return self._base_dir / execution_id

    def _get_checkpoint_path(
        self, execution_id: str, node_id: str,
    ) -> Path:
        """Get file path for a specific checkpoint."""
        return self._get_exec_dir(execution_id) / f"{node_id}.json"

    def save(self, checkpoint: DAGCheckpoint) -> bool:
        """Save checkpoint to JSON file."""
        try:
            exec_dir = self._get_exec_dir(checkpoint.execution_id)
            exec_dir.mkdir(parents=True, exist_ok=True)

            filepath = self._get_checkpoint_path(
                checkpoint.execution_id, checkpoint.node_id,
            )
            data = checkpoint.to_dict()

            # Write atomically via temp file
            temp_path = filepath.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            temp_path.replace(filepath)

            logger.debug(
                "Saved checkpoint to %s", filepath,
            )
            return True
        except Exception as e:
            logger.error("Failed to save checkpoint to file: %s", e)
            return False

    def load(
        self, execution_id: str, node_id: str,
    ) -> Optional[DAGCheckpoint]:
        """Load checkpoint from JSON file."""
        try:
            filepath = self._get_checkpoint_path(execution_id, node_id)
            if not filepath.exists():
                return None
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DAGCheckpoint.from_dict(data)
        except Exception as e:
            logger.error(
                "Failed to load checkpoint file %s/%s: %s",
                execution_id, node_id, e,
            )
            return None

    def list_checkpoints(self, execution_id: str) -> List[DAGCheckpoint]:
        """List all checkpoint files for an execution."""
        exec_dir = self._get_exec_dir(execution_id)
        if not exec_dir.exists():
            return []

        checkpoints: List[DAGCheckpoint] = []
        for filepath in sorted(exec_dir.glob("*.json")):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                checkpoints.append(DAGCheckpoint.from_dict(data))
            except Exception as e:
                logger.warning(
                    "Failed to load checkpoint %s: %s", filepath, e,
                )
        return checkpoints

    def delete(self, execution_id: str) -> int:
        """Delete all checkpoint files for an execution."""
        import shutil
        exec_dir = self._get_exec_dir(execution_id)
        if not exec_dir.exists():
            return 0

        count = sum(1 for _ in exec_dir.glob("*.json"))
        shutil.rmtree(exec_dir, ignore_errors=True)
        logger.info(
            "Deleted %d checkpoints for execution=%s",
            count, execution_id,
        )
        return count

    def exists(self, execution_id: str, node_id: str) -> bool:
        """Check if checkpoint file exists."""
        return self._get_checkpoint_path(execution_id, node_id).exists()

    def cleanup(self, retention_days: int) -> int:
        """Clean up old checkpoint directories."""
        cutoff = datetime.now() - timedelta(days=retention_days)
        removed = 0

        if not self._base_dir.exists():
            return 0

        import shutil
        for exec_dir in self._base_dir.iterdir():
            if not exec_dir.is_dir():
                continue
            try:
                mtime = datetime.fromtimestamp(exec_dir.stat().st_mtime)
                if mtime < cutoff:
                    count = sum(1 for _ in exec_dir.glob("*.json"))
                    shutil.rmtree(exec_dir, ignore_errors=True)
                    removed += count
                    logger.info(
                        "Cleaned up %d checkpoints from %s",
                        count, exec_dir,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to clean up %s: %s", exec_dir, e,
                )

        return removed


# ===================================================================
# Factory function
# ===================================================================


def create_checkpoint_store(
    strategy: str,
    checkpoint_dir: str = "",
    db_connection_string: str = "",
) -> DAGCheckpointStore:
    """Create a checkpoint store based on strategy name.

    Args:
        strategy: One of "memory", "file", "database".
        checkpoint_dir: Directory for file-based storage.
        db_connection_string: Connection string for database storage.

    Returns:
        DAGCheckpointStore implementation.

    Raises:
        ValueError: If strategy is unknown.
    """
    if strategy == "memory":
        return MemoryDAGCheckpointStore()
    elif strategy == "file":
        if not checkpoint_dir:
            checkpoint_dir = "/tmp/greenlang/orchestrator/checkpoints"
        return FileDAGCheckpointStore(checkpoint_dir)
    elif strategy == "database":
        logger.warning(
            "Database checkpoint store not yet implemented; "
            "falling back to file-based store"
        )
        if not checkpoint_dir:
            checkpoint_dir = "/tmp/greenlang/orchestrator/checkpoints"
        return FileDAGCheckpointStore(checkpoint_dir)
    else:
        raise ValueError(f"Unknown checkpoint strategy: {strategy}")


__all__ = [
    "DAGCheckpointStore",
    "MemoryDAGCheckpointStore",
    "FileDAGCheckpointStore",
    "create_checkpoint_store",
]
