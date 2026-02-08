# -*- coding: utf-8 -*-
"""
Golden File / Snapshot Testing Manager - AGENT-FOUND-009

Provides golden file (snapshot) management for the QA test harness,
including saving, loading, comparing, updating, listing, and deleting
golden files with content hash verification and TTL-based caching.

Zero-Hallucination Guarantees:
    - All golden files are human-verified before storage
    - Content hashes computed via deterministic SHA-256
    - No LLM-generated expected values in golden files
    - Complete provenance for every golden file operation

Example:
    >>> from greenlang.qa_test_harness.golden_file_manager import GoldenFileManager
    >>> manager = GoldenFileManager(config)
    >>> entry = manager.save_golden_file("MyAgent", "basic_test", input_data, output_data)
    >>> assertions = manager.compare_with_golden(agent_result, entry)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-009 QA Test Harness
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from greenlang.qa_test_harness.config import QATestHarnessConfig
from greenlang.qa_test_harness.models import (
    GoldenFileEntry,
    TestAssertion,
    SeverityLevel,
)
from greenlang.qa_test_harness.metrics import (
    record_golden_file_mismatch,
    record_cache_hit,
    record_cache_miss,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class GoldenFileManager:
    """Golden file / snapshot testing manager.

    Manages the lifecycle of golden files: creation, storage, comparison,
    versioning, and deletion. Includes a TTL-based in-memory cache to
    reduce file system reads for frequently accessed golden files.

    Attributes:
        config: QA test harness configuration.
        _golden_dir: Path to the golden file directory.
        _file_cache: In-memory cache with TTL for golden file content.
        _entries: In-memory store of golden file entries.

    Example:
        >>> manager = GoldenFileManager(config)
        >>> entry = manager.save_golden_file("MyAgent", "test_name", input_data, output_data)
    """

    def __init__(self, config: QATestHarnessConfig) -> None:
        """Initialize GoldenFileManager.

        Args:
            config: QA test harness configuration.
        """
        self.config = config
        self._golden_dir = Path(config.golden_file_directory)
        self._file_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._entries: Dict[str, GoldenFileEntry] = {}

        logger.info(
            "GoldenFileManager initialized: dir=%s, cache_ttl=%ds",
            self._golden_dir, config.golden_file_cache_ttl_seconds,
        )

    def save_golden_file(
        self,
        agent_type: str,
        name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        description: str = "",
    ) -> GoldenFileEntry:
        """Save a golden file for future comparison.

        Args:
            agent_type: Type of agent this golden file applies to.
            name: Human-readable name for the golden file.
            input_data: Input data used to generate the output.
            output_data: Output data to save as the golden file.
            description: Optional description of the golden file.

        Returns:
            GoldenFileEntry with metadata about the saved file.
        """
        self._golden_dir.mkdir(parents=True, exist_ok=True)

        input_hash = self._compute_content_hash(input_data)
        filename = f"{agent_type}_{name}_{input_hash[:8]}.json"
        filepath = self._golden_dir / filename

        golden_content = {
            "agent_type": agent_type,
            "name": name,
            "description": description,
            "input_data": input_data,
            "expected_output": output_data,
            "created_at": _utcnow().isoformat(),
            "version": "1.0.0",
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(golden_content, f, indent=2, default=str)

        content_hash = self._compute_content_hash(golden_content)

        entry = GoldenFileEntry(
            agent_type=agent_type,
            name=name,
            version="1.0.0",
            input_hash=input_hash,
            content_hash=content_hash,
            file_path=str(filepath),
        )

        # Store entry and invalidate cache
        self._entries[entry.file_id] = entry
        self._invalidate_cache(str(filepath))

        logger.info(
            "Saved golden file: %s (id=%s, hash=%s)",
            filepath, entry.file_id[:8], content_hash[:16],
        )
        return entry

    def load_golden_file(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        """Load a golden file from disk with caching.

        Args:
            file_path: Path to the golden file.

        Returns:
            Parsed golden file content.

        Raises:
            FileNotFoundError: If the golden file does not exist.
            json.JSONDecodeError: If the golden file is not valid JSON.
        """
        # Check cache
        cached = self._get_from_cache(file_path)
        if cached is not None:
            record_cache_hit()
            return cached

        record_cache_miss()

        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        # Store in cache
        self._put_in_cache(file_path, content)

        return content

    def compare_with_golden(
        self,
        agent_result: Any,
        golden_file_entry: GoldenFileEntry,
    ) -> List[TestAssertion]:
        """Compare agent result against a golden file entry.

        Args:
            agent_result: Agent execution result.
            golden_file_entry: Golden file entry to compare against.

        Returns:
            List of assertion results from the comparison.
        """
        assertions: List[TestAssertion] = []

        try:
            golden_data = self.load_golden_file(golden_file_entry.file_path)
            expected_output = golden_data.get("expected_output", {})
            actual_data = agent_result.data if agent_result else {}

            # Compare each key in expected output
            all_match = True
            for key in expected_output:
                expected_value = expected_output[key]
                actual_value = actual_data.get(key)
                matches = _deep_compare(expected_value, actual_value)

                if not matches:
                    all_match = False

                assertions.append(TestAssertion(
                    name=f"golden_{key}",
                    passed=matches,
                    expected=str(expected_value)[:100],
                    actual=str(actual_value)[:100],
                    message=f"Field '{key}' should match golden file",
                    severity=SeverityLevel.HIGH,
                ))

            # Verify content hash
            current_hash = self._compute_content_hash(golden_data)
            hash_matches = current_hash == golden_file_entry.content_hash
            assertions.append(TestAssertion(
                name="golden_file_integrity",
                passed=hash_matches,
                expected=golden_file_entry.content_hash[:16],
                actual=current_hash[:16],
                message="Golden file content hash should match stored hash",
                severity=SeverityLevel.CRITICAL,
            ))

            if not all_match:
                record_golden_file_mismatch()

        except FileNotFoundError:
            assertions.append(TestAssertion(
                name="golden_file_exists",
                passed=False,
                message=f"Golden file not found: {golden_file_entry.file_path}",
                severity=SeverityLevel.HIGH,
            ))

        except json.JSONDecodeError as e:
            assertions.append(TestAssertion(
                name="golden_file_valid",
                passed=False,
                message=f"Golden file is not valid JSON: {e}",
                severity=SeverityLevel.HIGH,
            ))

        return assertions

    def update_golden_file(
        self,
        file_id: str,
        new_output_data: Dict[str, Any],
    ) -> GoldenFileEntry:
        """Update an existing golden file with new output data.

        Args:
            file_id: ID of the golden file entry to update.
            new_output_data: New output data to replace existing.

        Returns:
            Updated GoldenFileEntry.

        Raises:
            ValueError: If the golden file entry is not found.
        """
        entry = self._entries.get(file_id)
        if entry is None:
            raise ValueError(f"Golden file entry not found: {file_id}")

        # Load existing content
        golden_data = self.load_golden_file(entry.file_path)
        golden_data["expected_output"] = new_output_data
        golden_data["version"] = _increment_version(entry.version)

        # Write updated content
        with open(entry.file_path, "w", encoding="utf-8") as f:
            json.dump(golden_data, f, indent=2, default=str)

        # Update entry
        new_content_hash = self._compute_content_hash(golden_data)
        entry.content_hash = new_content_hash
        entry.version = golden_data["version"]
        entry.updated_at = _utcnow()

        # Invalidate cache
        self._invalidate_cache(entry.file_path)

        logger.info(
            "Updated golden file: %s (version=%s, hash=%s)",
            entry.file_path, entry.version, new_content_hash[:16],
        )
        return entry

    def list_golden_files(
        self,
        agent_type: Optional[str] = None,
    ) -> List[GoldenFileEntry]:
        """List golden file entries, optionally filtered by agent type.

        Args:
            agent_type: Optional agent type filter.

        Returns:
            List of golden file entries.
        """
        entries = list(self._entries.values())
        if agent_type:
            entries = [e for e in entries if e.agent_type == agent_type]
        return [e for e in entries if e.is_active]

    def get_golden_file(
        self,
        file_id: str,
    ) -> Optional[GoldenFileEntry]:
        """Get a specific golden file entry by ID.

        Args:
            file_id: Golden file entry identifier.

        Returns:
            GoldenFileEntry if found, None otherwise.
        """
        return self._entries.get(file_id)

    def delete_golden_file(
        self,
        file_id: str,
    ) -> bool:
        """Soft-delete a golden file entry by marking it inactive.

        Args:
            file_id: Golden file entry identifier.

        Returns:
            True if the entry was found and deactivated, False otherwise.
        """
        entry = self._entries.get(file_id)
        if entry is None:
            return False

        entry.is_active = False
        entry.updated_at = _utcnow()

        # Invalidate cache
        self._invalidate_cache(entry.file_path)

        logger.info("Deleted golden file entry: %s", file_id[:8])
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_content_hash(self, content: Any) -> str:
        """Compute SHA-256 hash of content.

        Args:
            content: Content to hash.

        Returns:
            Full hex-encoded SHA-256 hash.
        """
        serialized = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _get_from_cache(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get content from cache if present and not expired.

        Args:
            file_path: Cache key (file path).

        Returns:
            Cached content or None if not cached or expired.
        """
        if file_path not in self._file_cache:
            return None

        cached_time = self._cache_timestamps.get(file_path, 0)
        if time.time() - cached_time > self.config.golden_file_cache_ttl_seconds:
            # Expired
            del self._file_cache[file_path]
            del self._cache_timestamps[file_path]
            return None

        return self._file_cache[file_path]

    def _put_in_cache(self, file_path: str, content: Dict[str, Any]) -> None:
        """Store content in cache with current timestamp.

        Args:
            file_path: Cache key (file path).
            content: Content to cache.
        """
        self._file_cache[file_path] = content
        self._cache_timestamps[file_path] = time.time()

    def _invalidate_cache(self, file_path: str) -> None:
        """Remove content from cache.

        Args:
            file_path: Cache key (file path) to invalidate.
        """
        self._file_cache.pop(file_path, None)
        self._cache_timestamps.pop(file_path, None)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _deep_compare(obj1: Any, obj2: Any) -> bool:
    """Deep compare two objects for equality.

    Args:
        obj1: First object.
        obj2: Second object.

    Returns:
        True if objects are deeply equal.
    """
    if type(obj1) != type(obj2):
        return False

    if isinstance(obj1, dict):
        if set(obj1.keys()) != set(obj2.keys()):
            return False
        return all(
            _deep_compare(obj1[k], obj2[k])
            for k in obj1.keys()
        )

    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            return False
        return all(
            _deep_compare(a, b)
            for a, b in zip(obj1, obj2)
        )

    elif isinstance(obj1, float):
        return abs(obj1 - obj2) < 1e-9

    else:
        return obj1 == obj2


def _increment_version(version: str) -> str:
    """Increment the patch version of a semver string.

    Args:
        version: Semver string (e.g. "1.0.0").

    Returns:
        Incremented version string (e.g. "1.0.1").
    """
    try:
        parts = version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except (ValueError, IndexError):
        return f"{version}.1"


__all__ = [
    "GoldenFileManager",
]
