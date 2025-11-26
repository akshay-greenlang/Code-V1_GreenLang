# -*- coding: utf-8 -*-
"""
Short-term memory for GreenLang agents.

Provides in-memory storage for recent events, context, and working
data with automatic expiration and capacity management.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    In-memory short-term storage for agent context and recent events.

    Provides a bounded, thread-safe memory store with automatic
    expiration and FIFO eviction when capacity is reached.

    Attributes:
        capacity: Maximum number of entries
        ttl_seconds: Time-to-live for entries (None for no expiration)

    Example:
        >>> memory = ShortTermMemory(capacity=1000, ttl_seconds=3600)
        >>> memory.store({'event': 'leak_detected', 'location': 'valve-12'})
        >>> recent = memory.retrieve(limit=10)
    """

    def __init__(
        self,
        capacity: int = 1000,
        ttl_seconds: Optional[int] = None
    ):
        """
        Initialize short-term memory.

        Args:
            capacity: Maximum number of entries
            ttl_seconds: Time-to-live in seconds (None for no expiration)
        """
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._memory: deque = deque(maxlen=capacity)
        self._timestamps: deque = deque(maxlen=capacity)
        self._lock = threading.RLock()
        self._index: Dict[str, int] = {}

        logger.info(f"ShortTermMemory initialized: capacity={capacity}, ttl={ttl_seconds}s")

    def store(
        self,
        entry: Dict[str, Any],
        key: Optional[str] = None
    ) -> str:
        """
        Store an entry in memory.

        Args:
            entry: Data to store
            key: Optional key for retrieval

        Returns:
            Entry key
        """
        with self._lock:
            timestamp = datetime.now(timezone.utc)
            entry_key = key or f"entry_{len(self._memory)}_{timestamp.timestamp()}"

            # Add entry
            self._memory.append({
                'key': entry_key,
                'data': entry,
                'timestamp': timestamp
            })
            self._timestamps.append(timestamp)

            # Update index
            self._index[entry_key] = len(self._memory) - 1

            logger.debug(f"Stored entry: {entry_key}")
            return entry_key

    def retrieve(
        self,
        limit: Optional[int] = None,
        since: Optional[datetime] = None,
        filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve entries from memory.

        Args:
            limit: Maximum entries to return
            since: Only entries after this timestamp
            filter_fn: Optional filter function

        Returns:
            List of matching entries
        """
        with self._lock:
            self._cleanup_expired()

            results = []
            for entry in reversed(self._memory):
                # Apply timestamp filter
                if since and entry['timestamp'] < since:
                    continue

                # Apply custom filter
                if filter_fn and not filter_fn(entry['data']):
                    continue

                results.append(entry['data'])

                # Apply limit
                if limit and len(results) >= limit:
                    break

            return results

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific entry by key.

        Args:
            key: Entry key

        Returns:
            Entry data or None if not found
        """
        with self._lock:
            self._cleanup_expired()

            for entry in self._memory:
                if entry['key'] == key:
                    return entry['data']

            return None

    def search(
        self,
        query: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search memory for matching entries.

        Args:
            query: Key-value pairs to match
            limit: Maximum results

        Returns:
            List of matching entries
        """
        def match_fn(entry: Dict[str, Any]) -> bool:
            for key, value in query.items():
                if key not in entry or entry[key] != value:
                    return False
            return True

        return self.retrieve(limit=limit, filter_fn=match_fn)

    def clear(self) -> None:
        """Clear all entries from memory."""
        with self._lock:
            self._memory.clear()
            self._timestamps.clear()
            self._index.clear()
            logger.info("ShortTermMemory cleared")

    def size(self) -> int:
        """Get current number of entries."""
        with self._lock:
            return len(self._memory)

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        if not self.ttl_seconds:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.ttl_seconds)

        # Remove expired from front (oldest)
        while self._memory and self._timestamps[0] < cutoff:
            entry = self._memory.popleft()
            self._timestamps.popleft()
            if entry['key'] in self._index:
                del self._index[entry['key']]

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            return {
                'capacity': self.capacity,
                'size': len(self._memory),
                'utilization': len(self._memory) / self.capacity if self.capacity > 0 else 0,
                'ttl_seconds': self.ttl_seconds,
                'oldest_entry': self._timestamps[0].isoformat() if self._timestamps else None,
                'newest_entry': self._timestamps[-1].isoformat() if self._timestamps else None
            }
