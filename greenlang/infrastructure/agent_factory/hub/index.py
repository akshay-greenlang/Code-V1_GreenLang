# -*- coding: utf-8 -*-
"""
Local Index - Local agent package index with file locking.

Maintains a thread-safe, file-backed index of installed agent packages at
.greenlang/agent_index.json. Supports add, remove, get, and list operations
with atomic file writes and cross-platform advisory file locking.

Example:
    >>> index = LocalIndex()
    >>> index.add(IndexEntry(agent_key="my-agent", version="1.0.0", installed_path="/path"))
    >>> entry = index.get("my-agent")
    >>> all_entries = index.list_all()

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-platform file locking
# ---------------------------------------------------------------------------

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    import msvcrt

    def _lock_shared(fh: Any) -> None:
        """Acquire a shared (read) lock on Windows."""
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)

    def _lock_exclusive(fh: Any) -> None:
        """Acquire an exclusive (write) lock on Windows."""
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock(fh: Any) -> None:
        """Release a file lock on Windows."""
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:
    import fcntl

    def _lock_shared(fh: Any) -> None:
        """Acquire a shared (read) lock on Unix."""
        fcntl.flock(fh.fileno(), fcntl.LOCK_SH)

    def _lock_exclusive(fh: Any) -> None:
        """Acquire an exclusive (write) lock on Unix."""
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

    def _unlock(fh: Any) -> None:
        """Release a file lock on Unix."""
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INDEX_PATH = ".greenlang/agent_index.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IndexEntry:
    """An entry in the local agent package index.

    Attributes:
        agent_key: Agent identifier.
        version: Installed version string.
        installed_path: Filesystem path where the agent is installed.
        installed_at: UTC ISO-8601 timestamp of installation.
        checksum: SHA-256 hex digest of the installed package.
    """

    agent_key: str
    version: str
    installed_path: str
    installed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    checksum: str = ""

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> IndexEntry:
        """Deserialize from dictionary."""
        return cls(
            agent_key=data["agent_key"],
            version=data["version"],
            installed_path=data["installed_path"],
            installed_at=data.get("installed_at", ""),
            checksum=data.get("checksum", ""),
        )


# ---------------------------------------------------------------------------
# Local Index
# ---------------------------------------------------------------------------


class LocalIndex:
    """Thread-safe, file-backed local agent package index.

    Stores installed agent metadata in a JSON file with cross-platform
    advisory file locking and an in-process threading lock for safe
    concurrent access.

    Attributes:
        index_path: Path to the JSON index file.
    """

    def __init__(self, index_path: Optional[str | Path] = None) -> None:
        """Initialize the local index.

        Args:
            index_path: Path to the index file. Defaults to ~/.greenlang/agent_index.json.
        """
        if index_path is None:
            self.index_path = Path.home() / DEFAULT_INDEX_PATH
        else:
            self.index_path = Path(index_path).resolve()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def add(self, entry: IndexEntry) -> None:
        """Add or update an agent entry in the index.

        Args:
            entry: Index entry to add.
        """
        with self._lock:
            data = self._read()
            data[entry.agent_key] = entry.to_dict()
            self._write(data)
        logger.info(
            "Added to index: %s v%s at %s",
            entry.agent_key,
            entry.version,
            entry.installed_path,
        )

    def remove(self, agent_key: str) -> bool:
        """Remove an agent from the index.

        Args:
            agent_key: Agent to remove.

        Returns:
            True if the agent was found and removed.
        """
        with self._lock:
            data = self._read()
            if agent_key not in data:
                return False
            del data[agent_key]
            self._write(data)
        logger.info("Removed from index: %s", agent_key)
        return True

    def get(self, agent_key: str) -> Optional[IndexEntry]:
        """Get an agent entry from the index.

        Args:
            agent_key: Agent identifier.

        Returns:
            IndexEntry if found, None otherwise.
        """
        with self._lock:
            data = self._read()
        entry_data = data.get(agent_key)
        if entry_data is None:
            return None
        return IndexEntry.from_dict(entry_data)

    def list_all(self) -> List[IndexEntry]:
        """List all entries in the index.

        Returns:
            List of all IndexEntry objects, sorted by agent_key.
        """
        with self._lock:
            data = self._read()
        entries = [IndexEntry.from_dict(v) for v in data.values()]
        return sorted(entries, key=lambda e: e.agent_key)

    def contains(self, agent_key: str) -> bool:
        """Check if an agent is in the index.

        Args:
            agent_key: Agent identifier.

        Returns:
            True if the agent is indexed.
        """
        with self._lock:
            data = self._read()
        return agent_key in data

    def count(self) -> int:
        """Return the number of agents in the index."""
        with self._lock:
            data = self._read()
        return len(data)

    def clear(self) -> int:
        """Remove all entries from the index.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            data = self._read()
            count = len(data)
            self._write({})
        logger.info("Cleared index (%d entries removed)", count)
        return count

    # ------------------------------------------------------------------
    # File I/O with locking
    # ------------------------------------------------------------------

    def _read(self) -> Dict[str, Dict]:
        """Read the index file with a shared lock.

        Returns:
            Deserialized index data.
        """
        if not self.index_path.exists():
            return {}

        try:
            with open(self.index_path, "r", encoding="utf-8") as fh:
                try:
                    _lock_shared(fh)
                    content = fh.read()
                    if not content.strip():
                        return {}
                    return json.loads(content)
                finally:
                    _unlock(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read index file: %s", exc)
            return {}

    def _write(self, data: Dict[str, Dict]) -> None:
        """Write the index file atomically with an exclusive lock.

        Uses a temporary file + rename for atomic writes to prevent
        corruption on crashes.
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.index_path.parent),
            prefix=".agent_index_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                _lock_exclusive(fh)
                json.dump(data, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
                _unlock(fh)

            # Atomic rename (os.replace is atomic on both Windows and Unix)
            os.replace(tmp_path, str(self.index_path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
