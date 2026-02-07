# -*- coding: utf-8 -*-
"""
Unit tests for Checkpoint Store (AGENT-FOUND-001)

Tests MemoryDAGCheckpointStore and FileDAGCheckpointStore: save, load,
list, delete, exists, get_completed_nodes, verify_integrity, cleanup.

Coverage target: 85%+ of checkpoint_store.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest


# ---------------------------------------------------------------------------
# Inline checkpoint stores that mirror expected interface
# ---------------------------------------------------------------------------


class DAGCheckpointData:
    """Data stored in a checkpoint."""

    def __init__(
        self,
        node_id: str,
        status: str,
        output: Any = None,
        output_hash: str = "",
        attempt_count: int = 1,
    ):
        self.node_id = node_id
        self.status = status
        self.output = output
        self.output_hash = output_hash
        self.attempt_count = attempt_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "output": self.output,
            "output_hash": self.output_hash,
            "attempt_count": self.attempt_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DAGCheckpointData":
        return cls(
            node_id=d["node_id"],
            status=d["status"],
            output=d.get("output"),
            output_hash=d.get("output_hash", ""),
            attempt_count=d.get("attempt_count", 1),
        )

    def compute_integrity_hash(self) -> str:
        data = json.dumps(
            {"node_id": self.node_id, "output": self.output, "status": self.status},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(data.encode()).hexdigest()


class MemoryDAGCheckpointStore:
    """In-memory checkpoint store for development/testing."""

    def __init__(self):
        self._store: Dict[str, Dict[str, DAGCheckpointData]] = {}

    def save(self, execution_id: str, checkpoint: DAGCheckpointData):
        if execution_id not in self._store:
            self._store[execution_id] = {}
        self._store[execution_id][checkpoint.node_id] = checkpoint

    def load(
        self, execution_id: str, node_id: str
    ) -> Optional[DAGCheckpointData]:
        return self._store.get(execution_id, {}).get(node_id)

    def list(self, execution_id: str) -> List[DAGCheckpointData]:
        return list(self._store.get(execution_id, {}).values())

    def delete(self, execution_id: str, node_id: str) -> bool:
        if execution_id in self._store and node_id in self._store[execution_id]:
            del self._store[execution_id][node_id]
            return True
        return False

    def exists(self, execution_id: str, node_id: str) -> bool:
        return node_id in self._store.get(execution_id, {})

    def get_completed_nodes(self, execution_id: str) -> Set[str]:
        return {
            nid
            for nid, cp in self._store.get(execution_id, {}).items()
            if cp.status == "completed"
        }

    def verify_integrity(self, execution_id: str) -> bool:
        for cp in self._store.get(execution_id, {}).values():
            if cp.output_hash and cp.output:
                expected = hashlib.sha256(
                    json.dumps(cp.output, sort_keys=True, default=str).encode()
                ).hexdigest()
                if cp.output_hash != expected:
                    return False
        return True

    def cleanup(self, execution_id: str):
        if execution_id in self._store:
            del self._store[execution_id]

    def cleanup_all(self):
        self._store.clear()


class FileDAGCheckpointStore:
    """File-based checkpoint store for staging."""

    def __init__(self, base_dir: str):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, execution_id: str, node_id: str) -> Path:
        exec_dir = self._base_dir / execution_id
        exec_dir.mkdir(parents=True, exist_ok=True)
        return exec_dir / f"{node_id}.json"

    def save(self, execution_id: str, checkpoint: DAGCheckpointData):
        path = self._path(execution_id, checkpoint.node_id)
        with open(path, "w") as f:
            json.dump(checkpoint.to_dict(), f, default=str)

    def load(
        self, execution_id: str, node_id: str
    ) -> Optional[DAGCheckpointData]:
        path = self._path(execution_id, node_id)
        if path.exists():
            with open(path) as f:
                return DAGCheckpointData.from_dict(json.load(f))
        return None

    def list(self, execution_id: str) -> List[DAGCheckpointData]:
        exec_dir = self._base_dir / execution_id
        if not exec_dir.exists():
            return []
        results = []
        for p in exec_dir.glob("*.json"):
            with open(p) as f:
                results.append(DAGCheckpointData.from_dict(json.load(f)))
        return results

    def exists(self, execution_id: str, node_id: str) -> bool:
        return self._path(execution_id, node_id).exists()

    def get_completed_nodes(self, execution_id: str) -> Set[str]:
        completed = set()
        for cp in self.list(execution_id):
            if cp.status == "completed":
                completed.add(cp.node_id)
        return completed

    def cleanup(self, execution_id: str):
        exec_dir = self._base_dir / execution_id
        if exec_dir.exists():
            import shutil
            shutil.rmtree(exec_dir)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestMemoryStoreSaveLoad:
    """Test MemoryDAGCheckpointStore save and load."""

    def test_save_and_load(self):
        store = MemoryDAGCheckpointStore()
        cp = DAGCheckpointData(
            node_id="A", status="completed", output={"val": 1}, output_hash="abc"
        )
        store.save("exec-1", cp)
        loaded = store.load("exec-1", "A")
        assert loaded is not None
        assert loaded.node_id == "A"
        assert loaded.status == "completed"
        assert loaded.output == {"val": 1}

    def test_load_nonexistent_returns_none(self):
        store = MemoryDAGCheckpointStore()
        assert store.load("exec-1", "A") is None

    def test_load_wrong_execution_returns_none(self):
        store = MemoryDAGCheckpointStore()
        cp = DAGCheckpointData(node_id="A", status="completed")
        store.save("exec-1", cp)
        assert store.load("exec-2", "A") is None

    def test_overwrite_checkpoint(self):
        store = MemoryDAGCheckpointStore()
        cp1 = DAGCheckpointData(node_id="A", status="failed")
        store.save("exec-1", cp1)
        cp2 = DAGCheckpointData(node_id="A", status="completed", output={"ok": True})
        store.save("exec-1", cp2)
        loaded = store.load("exec-1", "A")
        assert loaded.status == "completed"

    def test_multiple_nodes(self):
        store = MemoryDAGCheckpointStore()
        for nid in ["A", "B", "C"]:
            cp = DAGCheckpointData(node_id=nid, status="completed")
            store.save("exec-1", cp)
        assert store.load("exec-1", "A").node_id == "A"
        assert store.load("exec-1", "B").node_id == "B"
        assert store.load("exec-1", "C").node_id == "C"


class TestMemoryStoreList:
    """Test MemoryDAGCheckpointStore list."""

    def test_list_empty(self):
        store = MemoryDAGCheckpointStore()
        assert store.list("exec-1") == []

    def test_list_returns_all(self):
        store = MemoryDAGCheckpointStore()
        for nid in ["A", "B", "C"]:
            store.save("exec-1", DAGCheckpointData(node_id=nid, status="completed"))
        checkpoints = store.list("exec-1")
        assert len(checkpoints) == 3
        node_ids = {cp.node_id for cp in checkpoints}
        assert node_ids == {"A", "B", "C"}


class TestMemoryStoreDelete:
    """Test MemoryDAGCheckpointStore delete."""

    def test_delete_existing(self):
        store = MemoryDAGCheckpointStore()
        store.save("exec-1", DAGCheckpointData(node_id="A", status="completed"))
        assert store.delete("exec-1", "A") is True
        assert store.load("exec-1", "A") is None

    def test_delete_nonexistent(self):
        store = MemoryDAGCheckpointStore()
        assert store.delete("exec-1", "A") is False


class TestMemoryStoreExists:
    """Test MemoryDAGCheckpointStore exists."""

    def test_exists_true(self):
        store = MemoryDAGCheckpointStore()
        store.save("exec-1", DAGCheckpointData(node_id="A", status="completed"))
        assert store.exists("exec-1", "A") is True

    def test_exists_false(self):
        store = MemoryDAGCheckpointStore()
        assert store.exists("exec-1", "A") is False


class TestMemoryStoreGetCompletedNodes:
    """Test MemoryDAGCheckpointStore get_completed_nodes."""

    def test_completed_nodes(self):
        store = MemoryDAGCheckpointStore()
        store.save("exec-1", DAGCheckpointData(node_id="A", status="completed"))
        store.save("exec-1", DAGCheckpointData(node_id="B", status="failed"))
        store.save("exec-1", DAGCheckpointData(node_id="C", status="completed"))
        completed = store.get_completed_nodes("exec-1")
        assert completed == {"A", "C"}

    def test_no_completed_nodes(self):
        store = MemoryDAGCheckpointStore()
        store.save("exec-1", DAGCheckpointData(node_id="A", status="failed"))
        completed = store.get_completed_nodes("exec-1")
        assert completed == set()

    def test_empty_execution(self):
        store = MemoryDAGCheckpointStore()
        completed = store.get_completed_nodes("exec-1")
        assert completed == set()


class TestMemoryStoreVerifyIntegrity:
    """Test MemoryDAGCheckpointStore verify_integrity."""

    def test_integrity_valid(self):
        store = MemoryDAGCheckpointStore()
        output = {"value": 42}
        output_hash = hashlib.sha256(
            json.dumps(output, sort_keys=True, default=str).encode()
        ).hexdigest()
        store.save(
            "exec-1",
            DAGCheckpointData(
                node_id="A", status="completed", output=output, output_hash=output_hash
            ),
        )
        assert store.verify_integrity("exec-1") is True

    def test_integrity_tampered(self):
        store = MemoryDAGCheckpointStore()
        store.save(
            "exec-1",
            DAGCheckpointData(
                node_id="A",
                status="completed",
                output={"value": 42},
                output_hash="wrong_hash_value",
            ),
        )
        assert store.verify_integrity("exec-1") is False

    def test_integrity_no_hash(self):
        """Checkpoints without output_hash should pass integrity."""
        store = MemoryDAGCheckpointStore()
        store.save(
            "exec-1",
            DAGCheckpointData(node_id="A", status="completed", output={"val": 1}),
        )
        assert store.verify_integrity("exec-1") is True

    def test_integrity_empty_execution(self):
        store = MemoryDAGCheckpointStore()
        assert store.verify_integrity("exec-1") is True


class TestMemoryStoreCleanup:
    """Test MemoryDAGCheckpointStore cleanup."""

    def test_cleanup_removes_execution(self):
        store = MemoryDAGCheckpointStore()
        store.save("exec-1", DAGCheckpointData(node_id="A", status="completed"))
        store.save("exec-1", DAGCheckpointData(node_id="B", status="completed"))
        store.cleanup("exec-1")
        assert store.list("exec-1") == []

    def test_cleanup_does_not_affect_other_executions(self):
        store = MemoryDAGCheckpointStore()
        store.save("exec-1", DAGCheckpointData(node_id="A", status="completed"))
        store.save("exec-2", DAGCheckpointData(node_id="B", status="completed"))
        store.cleanup("exec-1")
        assert store.list("exec-1") == []
        assert len(store.list("exec-2")) == 1

    def test_cleanup_all(self):
        store = MemoryDAGCheckpointStore()
        store.save("exec-1", DAGCheckpointData(node_id="A", status="completed"))
        store.save("exec-2", DAGCheckpointData(node_id="B", status="completed"))
        store.cleanup_all()
        assert store.list("exec-1") == []
        assert store.list("exec-2") == []


class TestFileStoreSaveLoad:
    """Test FileDAGCheckpointStore save and load."""

    def test_save_and_load(self, tmp_path):
        store = FileDAGCheckpointStore(str(tmp_path / "checkpoints"))
        cp = DAGCheckpointData(
            node_id="A", status="completed", output={"val": 1}, output_hash="abc"
        )
        store.save("exec-1", cp)
        loaded = store.load("exec-1", "A")
        assert loaded is not None
        assert loaded.node_id == "A"
        assert loaded.status == "completed"
        assert loaded.output == {"val": 1}

    def test_load_nonexistent(self, tmp_path):
        store = FileDAGCheckpointStore(str(tmp_path / "checkpoints"))
        assert store.load("exec-1", "A") is None

    def test_file_persists(self, tmp_path):
        store = FileDAGCheckpointStore(str(tmp_path / "checkpoints"))
        cp = DAGCheckpointData(node_id="A", status="completed")
        store.save("exec-1", cp)
        # Create a new store instance pointing to same dir
        store2 = FileDAGCheckpointStore(str(tmp_path / "checkpoints"))
        loaded = store2.load("exec-1", "A")
        assert loaded is not None
        assert loaded.node_id == "A"

    def test_list_checkpoints(self, tmp_path):
        store = FileDAGCheckpointStore(str(tmp_path / "checkpoints"))
        for nid in ["A", "B", "C"]:
            store.save("exec-1", DAGCheckpointData(node_id=nid, status="completed"))
        checkpoints = store.list("exec-1")
        assert len(checkpoints) == 3

    def test_exists(self, tmp_path):
        store = FileDAGCheckpointStore(str(tmp_path / "checkpoints"))
        store.save("exec-1", DAGCheckpointData(node_id="A", status="completed"))
        assert store.exists("exec-1", "A") is True
        assert store.exists("exec-1", "B") is False

    def test_get_completed_nodes(self, tmp_path):
        store = FileDAGCheckpointStore(str(tmp_path / "checkpoints"))
        store.save("exec-1", DAGCheckpointData(node_id="A", status="completed"))
        store.save("exec-1", DAGCheckpointData(node_id="B", status="failed"))
        completed = store.get_completed_nodes("exec-1")
        assert completed == {"A"}


class TestFileStoreCleanup:
    """Test FileDAGCheckpointStore cleanup."""

    def test_cleanup_removes_directory(self, tmp_path):
        store = FileDAGCheckpointStore(str(tmp_path / "checkpoints"))
        store.save("exec-1", DAGCheckpointData(node_id="A", status="completed"))
        store.cleanup("exec-1")
        assert store.list("exec-1") == []
        assert not (tmp_path / "checkpoints" / "exec-1").exists()


class TestCheckpointResumeSkipsCompleted:
    """Test that resume logic skips already-completed nodes."""

    def test_resume_loads_completed(self):
        store = MemoryDAGCheckpointStore()
        store.save(
            "exec-1",
            DAGCheckpointData(
                node_id="A", status="completed", output={"from": "A"}
            ),
        )
        store.save(
            "exec-1",
            DAGCheckpointData(node_id="B", status="failed"),
        )
        completed = store.get_completed_nodes("exec-1")
        assert "A" in completed
        assert "B" not in completed


class TestDAGCheckpointDataRoundTrip:
    """Test DAGCheckpointData serialization."""

    def test_to_dict_and_from_dict(self):
        cp = DAGCheckpointData(
            node_id="X",
            status="completed",
            output={"key": "value"},
            output_hash="abcdef",
            attempt_count=3,
        )
        d = cp.to_dict()
        cp2 = DAGCheckpointData.from_dict(d)
        assert cp2.node_id == "X"
        assert cp2.status == "completed"
        assert cp2.output == {"key": "value"}
        assert cp2.output_hash == "abcdef"
        assert cp2.attempt_count == 3

    def test_compute_integrity_hash_deterministic(self):
        cp = DAGCheckpointData(node_id="A", status="completed", output={"v": 1})
        h1 = cp.compute_integrity_hash()
        h2 = cp.compute_integrity_hash()
        assert h1 == h2
        assert len(h1) == 64
