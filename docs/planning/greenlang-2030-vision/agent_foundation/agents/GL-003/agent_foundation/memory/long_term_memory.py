# -*- coding: utf-8 -*-
"""
Long-term memory for GreenLang agents.

Provides persistent storage for agent learning, pattern recognition,
and historical data with support for categorization and retrieval.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import asyncio
import hashlib

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Persistent long-term storage for agent memory.

    Stores memories to disk with categorization, indexing, and
    retrieval capabilities for agent learning and pattern recognition.

    Attributes:
        storage_path: Path to storage directory

    Example:
        >>> memory = LongTermMemory(Path("./agent_memory"))
        >>> await memory.store("analysis_123", data, category="analyses")
        >>> result = await memory.retrieve("analysis_123")
    """

    def __init__(self, storage_path: Path):
        """
        Initialize long-term memory.

        Args:
            storage_path: Directory for persistent storage
        """
        self.storage_path = storage_path
        self._index: Dict[str, Dict[str, Any]] = {}
        self._categories: Dict[str, List[str]] = {}

        # Create storage directory if needed
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing index
        self._load_index()

        logger.info(f"LongTermMemory initialized: {storage_path}")

    def _load_index(self) -> None:
        """Load memory index from disk."""
        index_path = self.storage_path / "index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    self._index = data.get('index', {})
                    self._categories = data.get('categories', {})
                logger.info(f"Loaded memory index: {len(self._index)} entries")
            except Exception as e:
                logger.error(f"Failed to load memory index: {e}")
                self._index = {}
                self._categories = {}

    def _save_index(self) -> None:
        """Save memory index to disk."""
        index_path = self.storage_path / "index.json"
        try:
            with open(index_path, 'w') as f:
                json.dump({
                    'index': self._index,
                    'categories': self._categories
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory index: {e}")

    async def store(
        self,
        key: str,
        value: Any,
        category: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory entry.

        Args:
            key: Unique key for the entry
            value: Data to store
            category: Category for organization
            metadata: Optional metadata

        Returns:
            Entry key
        """
        timestamp = datetime.now(timezone.utc)

        # Create entry
        entry = {
            'key': key,
            'value': value,
            'category': category,
            'metadata': metadata or {},
            'timestamp': timestamp.isoformat(),
            'hash': self._compute_hash(value)
        }

        # Save to file
        file_path = self._get_file_path(key)
        await asyncio.to_thread(self._write_file, file_path, entry)

        # Update index
        self._index[key] = {
            'category': category,
            'timestamp': timestamp.isoformat(),
            'file': str(file_path.relative_to(self.storage_path)),
            'hash': entry['hash']
        }

        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        if key not in self._categories[category]:
            self._categories[category].append(key)

        # Save index
        self._save_index()

        logger.debug(f"Stored memory: {key} in category {category}")
        return key

    async def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a memory entry by key.

        Args:
            key: Entry key

        Returns:
            Stored value or None if not found
        """
        if key not in self._index:
            return None

        file_path = self._get_file_path(key)
        if not file_path.exists():
            return None

        try:
            entry = await asyncio.to_thread(self._read_file, file_path)
            return entry.get('value')
        except Exception as e:
            logger.error(f"Failed to retrieve memory {key}: {e}")
            return None

    async def retrieve_by_category(
        self,
        category: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories by category.

        Args:
            category: Category to retrieve
            limit: Maximum entries to return

        Returns:
            List of memory entries
        """
        if category not in self._categories:
            return []

        keys = self._categories[category]
        if limit:
            keys = keys[-limit:]

        results = []
        for key in keys:
            value = await self.retrieve(key)
            if value is not None:
                results.append({
                    'key': key,
                    'value': value,
                    'metadata': self._index.get(key, {})
                })

        return results

    async def search(
        self,
        query: Dict[str, Any],
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search memories by metadata.

        Args:
            query: Metadata query
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of matching entries
        """
        results = []

        keys = self._categories.get(category, list(self._index.keys())) if category else list(self._index.keys())

        for key in keys:
            if len(results) >= limit:
                break

            index_entry = self._index.get(key, {})

            # Simple metadata matching
            match = True
            for q_key, q_value in query.items():
                if q_key == 'category' and index_entry.get('category') != q_value:
                    match = False
                    break

            if match:
                value = await self.retrieve(key)
                if value is not None:
                    results.append({
                        'key': key,
                        'value': value,
                        'metadata': index_entry
                    })

        return results

    async def delete(self, key: str) -> bool:
        """
        Delete a memory entry.

        Args:
            key: Entry key

        Returns:
            True if deleted, False if not found
        """
        if key not in self._index:
            return False

        # Remove file
        file_path = self._get_file_path(key)
        if file_path.exists():
            await asyncio.to_thread(file_path.unlink)

        # Update index
        category = self._index[key].get('category', 'default')
        del self._index[key]

        if category in self._categories and key in self._categories[category]:
            self._categories[category].remove(key)

        self._save_index()
        logger.debug(f"Deleted memory: {key}")
        return True

    def _get_file_path(self, key: str) -> Path:
        """Get file path for a key."""
        # Use hash to create subdirectory structure
        key_hash = hashlib.md5(key.encode()).hexdigest()
        subdir = self.storage_path / key_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key_hash}.json"

    def _write_file(self, path: Path, data: Dict[str, Any]) -> None:
        """Write data to file."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _read_file(self, path: Path) -> Dict[str, Any]:
        """Read data from file."""
        with open(path, 'r') as f:
            return json.load(f)

    def _compute_hash(self, value: Any) -> str:
        """Compute SHA-256 hash of value."""
        value_str = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(value_str.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_entries': len(self._index),
            'categories': {
                cat: len(keys) for cat, keys in self._categories.items()
            },
            'storage_path': str(self.storage_path)
        }
