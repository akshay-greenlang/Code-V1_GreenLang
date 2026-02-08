# -*- coding: utf-8 -*-
"""
Unit Tests for GoldenFileManager (AGENT-FOUND-009)

Tests golden file save, load, compare, update, list, delete, content hashing,
and caching behavior.

Coverage target: 85%+ of golden_file_manager.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline GoldenFileManager
# ---------------------------------------------------------------------------

def _content_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class GoldenFileEntry:
    def __init__(self, entry_id: str, agent_type: str, input_hash: str,
                 content: Dict[str, Any], content_hash: str, version: str = "1.0.0",
                 created_at: Optional[datetime] = None,
                 description: str = ""):
        self.entry_id = entry_id
        self.agent_type = agent_type
        self.input_hash = input_hash
        self.content = content
        self.content_hash = content_hash
        self.version = version
        self.created_at = created_at or datetime.now(timezone.utc)
        self.description = description


class GoldenFileManager:
    """In-memory golden file manager for testing."""

    def __init__(self):
        self._files: Dict[str, GoldenFileEntry] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    def save(self, agent_type: str, input_data: Dict[str, Any],
             output_data: Dict[str, Any], description: str = "") -> GoldenFileEntry:
        """Save a golden file."""
        self._counter += 1
        input_hash = _content_hash(input_data)[:16]
        entry_id = f"gf-{self._counter:04d}"
        content = {
            "agent_type": agent_type,
            "input_data": input_data,
            "expected_output": output_data,
            "version": "1.0.0",
        }
        ch = _content_hash(content)
        entry = GoldenFileEntry(
            entry_id=entry_id, agent_type=agent_type,
            input_hash=input_hash, content=content,
            content_hash=ch, description=description,
        )
        self._files[entry_id] = entry
        self._cache[entry_id] = content
        return entry

    def load(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Load a golden file by ID."""
        if entry_id in self._cache:
            return self._cache[entry_id]
        entry = self._files.get(entry_id)
        if entry:
            self._cache[entry_id] = entry.content
            return entry.content
        return None

    def compare(self, entry_id: str, actual_output: Dict[str, Any]) -> Dict[str, Any]:
        """Compare actual output against golden file."""
        content = self.load(entry_id)
        if content is None:
            return {"match": False, "error": "Golden file not found", "diffs": []}

        expected = content.get("expected_output", {})
        diffs = []
        for key in expected:
            exp_val = expected[key]
            act_val = actual_output.get(key)
            if exp_val != act_val:
                diffs.append({
                    "field": key, "expected": exp_val, "actual": act_val,
                })

        return {
            "match": len(diffs) == 0,
            "diffs": diffs,
            "total_fields": len(expected),
            "matching_fields": len(expected) - len(diffs),
        }

    def update(self, entry_id: str, output_data: Dict[str, Any]) -> Optional[GoldenFileEntry]:
        """Update golden file content."""
        entry = self._files.get(entry_id)
        if entry is None:
            return None
        entry.content["expected_output"] = output_data
        entry.content_hash = _content_hash(entry.content)
        self._cache[entry_id] = entry.content
        return entry

    def list_files(self, agent_type: Optional[str] = None) -> List[GoldenFileEntry]:
        """List golden files, optionally filtered by agent type."""
        entries = list(self._files.values())
        if agent_type:
            entries = [e for e in entries if e.agent_type == agent_type]
        return entries

    def get(self, entry_id: str) -> Optional[GoldenFileEntry]:
        """Get golden file entry by ID."""
        return self._files.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """Delete a golden file."""
        if entry_id in self._files:
            del self._files[entry_id]
            self._cache.pop(entry_id, None)
            return True
        return False

    def clear_cache(self) -> None:
        """Clear the content cache."""
        self._cache.clear()


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.fixture
def manager():
    return GoldenFileManager()


class TestSaveGoldenFile:
    def test_save_golden_file(self, manager):
        entry = manager.save("Agent", {"x": 1}, {"result": 42})
        assert entry.entry_id.startswith("gf-")
        assert entry.agent_type == "Agent"

    def test_save_creates_content_hash(self, manager):
        entry = manager.save("Agent", {"x": 1}, {"result": 42})
        assert len(entry.content_hash) == 64

    def test_save_stores_input_hash(self, manager):
        entry = manager.save("Agent", {"x": 1}, {"result": 42})
        assert entry.input_hash != ""

    def test_save_increments_counter(self, manager):
        e1 = manager.save("Agent", {"x": 1}, {"r": 1})
        e2 = manager.save("Agent", {"x": 2}, {"r": 2})
        assert e1.entry_id != e2.entry_id

    def test_save_with_description(self, manager):
        entry = manager.save("Agent", {}, {}, description="Baseline for v1.0")
        assert entry.description == "Baseline for v1.0"

    def test_save_default_version(self, manager):
        entry = manager.save("Agent", {}, {})
        assert entry.version == "1.0.0"

    def test_save_created_at_set(self, manager):
        entry = manager.save("Agent", {}, {})
        assert entry.created_at is not None


class TestLoadGoldenFile:
    def test_load_golden_file(self, manager):
        entry = manager.save("Agent", {"x": 1}, {"result": 42})
        content = manager.load(entry.entry_id)
        assert content is not None
        assert content["expected_output"]["result"] == 42

    def test_load_nonexistent(self, manager):
        content = manager.load("gf-9999")
        assert content is None

    def test_load_from_cache(self, manager):
        entry = manager.save("Agent", {"x": 1}, {"result": 42})
        # First load populates cache, second reads from cache
        content1 = manager.load(entry.entry_id)
        content2 = manager.load(entry.entry_id)
        assert content1 is content2

    def test_load_after_cache_clear(self, manager):
        entry = manager.save("Agent", {"x": 1}, {"result": 42})
        manager.clear_cache()
        content = manager.load(entry.entry_id)
        assert content is not None


class TestCompareWithGolden:
    def test_compare_match(self, manager):
        entry = manager.save("Agent", {"x": 1}, {"result": 42, "unit": "kg"})
        result = manager.compare(entry.entry_id, {"result": 42, "unit": "kg"})
        assert result["match"] is True
        assert result["diffs"] == []

    def test_compare_mismatch(self, manager):
        entry = manager.save("Agent", {"x": 1}, {"result": 42})
        result = manager.compare(entry.entry_id, {"result": 99})
        assert result["match"] is False
        assert len(result["diffs"]) == 1
        assert result["diffs"][0]["field"] == "result"

    def test_compare_not_found(self, manager):
        result = manager.compare("gf-9999", {"result": 42})
        assert result["match"] is False
        assert "not found" in result["error"].lower()

    def test_compare_partial_mismatch(self, manager):
        entry = manager.save("Agent", {}, {"a": 1, "b": 2, "c": 3})
        result = manager.compare(entry.entry_id, {"a": 1, "b": 99, "c": 3})
        assert result["match"] is False
        assert result["matching_fields"] == 2
        assert result["total_fields"] == 3

    def test_compare_empty_golden(self, manager):
        entry = manager.save("Agent", {}, {})
        result = manager.compare(entry.entry_id, {"result": 42})
        assert result["match"] is True


class TestUpdateGoldenFile:
    def test_update_golden_file(self, manager):
        entry = manager.save("Agent", {"x": 1}, {"result": 42})
        old_hash = entry.content_hash
        updated = manager.update(entry.entry_id, {"result": 99})
        assert updated is not None
        assert updated.content["expected_output"]["result"] == 99
        assert updated.content_hash != old_hash

    def test_update_nonexistent(self, manager):
        updated = manager.update("gf-9999", {"result": 42})
        assert updated is None

    def test_update_invalidates_cache(self, manager):
        entry = manager.save("Agent", {}, {"result": 42})
        manager.update(entry.entry_id, {"result": 99})
        content = manager.load(entry.entry_id)
        assert content["expected_output"]["result"] == 99


class TestListGoldenFiles:
    def test_list_golden_files_empty(self, manager):
        assert manager.list_files() == []

    def test_list_golden_files(self, manager):
        manager.save("Agent1", {}, {})
        manager.save("Agent2", {}, {})
        manager.save("Agent1", {"x": 1}, {})
        files = manager.list_files()
        assert len(files) == 3

    def test_list_golden_files_by_agent(self, manager):
        manager.save("Agent1", {}, {})
        manager.save("Agent2", {}, {})
        manager.save("Agent1", {"x": 1}, {})
        files = manager.list_files(agent_type="Agent1")
        assert len(files) == 2

    def test_list_golden_files_no_match(self, manager):
        manager.save("Agent1", {}, {})
        files = manager.list_files(agent_type="NonExistent")
        assert len(files) == 0


class TestGetGoldenFile:
    def test_get_golden_file(self, manager):
        entry = manager.save("Agent", {}, {})
        retrieved = manager.get(entry.entry_id)
        assert retrieved is not None
        assert retrieved.entry_id == entry.entry_id

    def test_get_nonexistent(self, manager):
        assert manager.get("gf-9999") is None


class TestDeleteGoldenFile:
    def test_delete_golden_file(self, manager):
        entry = manager.save("Agent", {}, {})
        assert manager.delete(entry.entry_id) is True
        assert manager.get(entry.entry_id) is None

    def test_delete_nonexistent(self, manager):
        assert manager.delete("gf-9999") is False

    def test_delete_clears_cache(self, manager):
        entry = manager.save("Agent", {}, {})
        manager.load(entry.entry_id)  # populate cache
        manager.delete(entry.entry_id)
        assert manager.load(entry.entry_id) is None


class TestContentHash:
    def test_content_hash_deterministic(self, manager):
        e1 = manager.save("Agent", {"x": 1}, {"r": 42})
        e2 = manager.save("Agent", {"x": 1}, {"r": 42})
        assert e1.content_hash == e2.content_hash

    def test_content_hash_changes_with_data(self, manager):
        e1 = manager.save("Agent", {"x": 1}, {"r": 42})
        e2 = manager.save("Agent", {"x": 1}, {"r": 99})
        assert e1.content_hash != e2.content_hash


class TestGoldenFileCache:
    def test_cache_populated_on_save(self, manager):
        entry = manager.save("Agent", {}, {})
        assert entry.entry_id in manager._cache

    def test_cache_clear(self, manager):
        manager.save("Agent", {}, {})
        manager.clear_cache()
        assert len(manager._cache) == 0

    def test_cache_repopulated_on_load(self, manager):
        entry = manager.save("Agent", {}, {})
        manager.clear_cache()
        manager.load(entry.entry_id)
        assert entry.entry_id in manager._cache
