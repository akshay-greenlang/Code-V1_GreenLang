# -*- coding: utf-8 -*-
"""
GreenLang L3 Disk Cache Implementation

Persistent disk-based cache for large artifacts using LRU eviction.
Optimized for storing model weights, datasets, and large computation results.

Features:
- Persistent storage in ~/.greenlang/cache/
- LRU eviction with size limits (10GB default)
- TTL support with background cleanup
- Atomic write operations
- Corruption detection with checksums
- Compression for space efficiency

Author: GreenLang Infrastructure Team (TEAM 2)
Date: 2025-11-08
Version: 5.0.0
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gzip
import pickle

from .architecture import CacheLayer

logger = logging.getLogger(__name__)


@dataclass
class DiskCacheEntry:
    """
    Metadata for disk cache entry.

    Attributes:
        key: Cache key
        file_path: Path to cached file
        size_bytes: Size of cached data
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        ttl_seconds: Time-to-live
        checksum: SHA256 checksum for integrity
        compressed: Whether data is compressed
    """
    key: str
    file_path: str
    size_bytes: int
    created_at: float
    accessed_at: float
    ttl_seconds: int
    checksum: str
    compressed: bool = False

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds <= 0:
            return False
        return time.time() > (self.created_at + self.ttl_seconds)


class L3DiskCache:
    """
    High-performance persistent disk cache.

    Uses SQLite for metadata and filesystem for data storage.
    Implements LRU eviction with size limits and corruption detection.

    Example:
        >>> cache = L3DiskCache(
        ...     cache_dir="~/.greenlang/cache",
        ...     max_size_gb=10
        ... )
        >>> await cache.start()
        >>> await cache.set("model_weights", large_tensor, ttl=86400)
        >>> weights = await cache.get("model_weights")
        >>> await cache.stop()
    """

    def __init__(
        self,
        cache_dir: str = "~/.greenlang/cache",
        max_size_gb: int = 10,
        default_ttl_seconds: int = 86400,  # 24 hours
        compression_enabled: bool = True,
        compression_threshold_bytes: int = 10 * 1024,  # 10KB
        checkpoint_interval_seconds: int = 300,  # 5 minutes
        cleanup_interval_seconds: int = 3600,  # 1 hour
        corruption_check: bool = True
    ):
        """
        Initialize L3 disk cache.

        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
            default_ttl_seconds: Default TTL
            compression_enabled: Enable gzip compression
            compression_threshold_bytes: Compress items larger than this
            checkpoint_interval_seconds: Checkpoint frequency
            cleanup_interval_seconds: Cleanup frequency
            corruption_check: Enable checksum verification
        """
        self._cache_dir = Path(os.path.expanduser(cache_dir))
        self._max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self._default_ttl = default_ttl_seconds
        self._compression_enabled = compression_enabled
        self._compression_threshold = compression_threshold_bytes
        self._checkpoint_interval = checkpoint_interval_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._corruption_check = corruption_check

        # Paths
        self._data_dir = self._cache_dir / "data"
        self._db_path = self._cache_dir / "cache.db"

        # Database connection
        self._db: Optional[sqlite3.Connection] = None

        # Current size tracking
        self._current_size_bytes = 0

        # Background tasks
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0
        self._corruption_errors = 0

        logger.info(
            f"Initialized L3 disk cache: "
            f"dir={cache_dir}, max_size={max_size_gb}GB, "
            f"compression={compression_enabled}"
        )

    async def start(self) -> None:
        """Initialize cache directory and database."""
        # Create directories
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        await self._init_db()

        # Calculate current size
        await self._calculate_size()

        # Start background tasks
        self._running = True
        self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(
            f"L3 disk cache started: "
            f"{self._get_entry_count()} entries, "
            f"{self._current_size_bytes / (1024*1024):.2f}MB used"
        )

    async def stop(self) -> None:
        """Stop cache and background tasks."""
        self._running = False

        # Cancel background tasks
        for task in [self._checkpoint_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close database
        if self._db:
            self._db.close()

        logger.info("L3 disk cache stopped")

    async def _init_db(self) -> None:
        """Initialize SQLite database for metadata."""
        self._db = sqlite3.connect(str(self._db_path))
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                ttl_seconds INTEGER NOT NULL,
                checksum TEXT NOT NULL,
                compressed INTEGER NOT NULL
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_accessed_at
            ON cache_entries(accessed_at)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON cache_entries(created_at)
        """)
        self._db.commit()

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from disk cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired/corrupted

        Example:
            >>> data = await cache.get("large_dataset")
        """
        try:
            # Get metadata
            cursor = self._db.execute(
                "SELECT * FROM cache_entries WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()

            if row is None:
                self._misses += 1
                return None

            entry = self._row_to_entry(row)

            # Check expiration
            if entry.is_expired():
                await self._evict_entry(entry, reason="ttl_expired")
                self._misses += 1
                return None

            # Read file
            file_path = Path(entry.file_path)
            if not file_path.exists():
                logger.warning(f"Cache file missing: {file_path}")
                await self._evict_entry(entry, reason="file_missing")
                self._misses += 1
                return None

            # Read data
            with open(file_path, 'rb') as f:
                data = f.read()

            # Verify checksum if enabled
            if self._corruption_check:
                checksum = hashlib.sha256(data).hexdigest()
                if checksum != entry.checksum:
                    logger.error(f"Checksum mismatch for key: {key}")
                    self._corruption_errors += 1
                    await self._evict_entry(entry, reason="corruption")
                    self._misses += 1
                    return None

            # Decompress if needed
            if entry.compressed:
                data = gzip.decompress(data)

            # Deserialize
            value = pickle.loads(data)

            # Update access time
            self._db.execute(
                "UPDATE cache_entries SET accessed_at = ? WHERE key = ?",
                (time.time(), key)
            )
            self._db.commit()

            self._hits += 1
            return value

        except Exception as e:
            logger.error(f"Error getting key {key}: {e}", exc_info=True)
            self._misses += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in disk cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds

        Returns:
            True if successful

        Example:
            >>> await cache.set("large_model", model_weights, ttl=7*86400)
        """
        try:
            ttl_seconds = ttl if ttl is not None else self._default_ttl

            # Serialize
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            # Compress if needed
            compressed = False
            if (
                self._compression_enabled
                and len(data) >= self._compression_threshold
            ):
                compressed_data = gzip.compress(data, compresslevel=6)
                # Use compression only if it saves space
                if len(compressed_data) < len(data):
                    data = compressed_data
                    compressed = True

            size_bytes = len(data)

            # Check if value is too large
            if size_bytes > self._max_size_bytes:
                logger.warning(
                    f"Value too large for cache: {size_bytes} bytes "
                    f"(max: {self._max_size_bytes})"
                )
                return False

            # Evict entries if needed
            while (
                self._current_size_bytes + size_bytes > self._max_size_bytes
                and self._get_entry_count() > 0
            ):
                await self._evict_lru()

            # Generate file path
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
            file_path = self._data_dir / f"{key_hash}.cache"

            # Write atomically (write to temp, then rename)
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())  # Ensure written to disk

            # Rename to final location
            temp_path.replace(file_path)

            # Calculate checksum
            checksum = hashlib.sha256(data).hexdigest()

            # Update database
            now = time.time()
            self._db.execute("""
                INSERT OR REPLACE INTO cache_entries
                (key, file_path, size_bytes, created_at, accessed_at,
                 ttl_seconds, checksum, compressed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key,
                str(file_path),
                size_bytes,
                now,
                now,
                ttl_seconds,
                checksum,
                1 if compressed else 0
            ))
            self._db.commit()

            self._current_size_bytes += size_bytes
            self._sets += 1

            logger.debug(
                f"Cached key: {key} "
                f"({size_bytes / 1024:.2f}KB, compressed={compressed})"
            )
            return True

        except Exception as e:
            logger.error(f"Error setting key {key}: {e}", exc_info=True)
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        try:
            cursor = self._db.execute(
                "SELECT * FROM cache_entries WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()

            if row is None:
                return False

            entry = self._row_to_entry(row)
            await self._evict_entry(entry, reason="manual_delete")
            return True

        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is valid."""
        cursor = self._db.execute(
            "SELECT created_at, ttl_seconds FROM cache_entries WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()

        if row is None:
            return False

        created_at, ttl_seconds = row
        if ttl_seconds > 0 and time.time() > (created_at + ttl_seconds):
            return False

        return True

    async def clear(self) -> None:
        """Clear all cache entries."""
        # Delete all files
        for file_path in self._data_dir.glob("*.cache"):
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")

        # Clear database
        self._db.execute("DELETE FROM cache_entries")
        self._db.commit()

        self._current_size_bytes = 0
        logger.info("L3 cache cleared")

    async def _evict_entry(self, entry: DiskCacheEntry, reason: str) -> None:
        """
        Evict a cache entry.

        Args:
            entry: Entry to evict
            reason: Reason for eviction
        """
        # Delete file
        file_path = Path(entry.file_path)
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")

        # Remove from database
        self._db.execute("DELETE FROM cache_entries WHERE key = ?", (entry.key,))
        self._db.commit()

        self._current_size_bytes -= entry.size_bytes
        self._evictions += 1

        logger.debug(f"Evicted entry: {entry.key} (reason: {reason})")

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        cursor = self._db.execute("""
            SELECT * FROM cache_entries
            ORDER BY accessed_at ASC
            LIMIT 1
        """)
        row = cursor.fetchone()

        if row:
            entry = self._row_to_entry(row)
            await self._evict_entry(entry, reason="lru_eviction")

    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        cursor = self._db.execute("""
            SELECT * FROM cache_entries
            WHERE ttl_seconds > 0
            AND (created_at + ttl_seconds) < ?
        """, (now,))

        expired_entries = [self._row_to_entry(row) for row in cursor.fetchall()]

        for entry in expired_entries:
            await self._evict_entry(entry, reason="ttl_expired")

        if expired_entries:
            logger.info(f"Cleaned up {len(expired_entries)} expired entries")

    async def _calculate_size(self) -> None:
        """Calculate current cache size."""
        cursor = self._db.execute("SELECT SUM(size_bytes) FROM cache_entries")
        result = cursor.fetchone()[0]
        self._current_size_bytes = result if result is not None else 0

    def _get_entry_count(self) -> int:
        """Get number of cache entries."""
        cursor = self._db.execute("SELECT COUNT(*) FROM cache_entries")
        return cursor.fetchone()[0]

    def _row_to_entry(self, row: Tuple) -> DiskCacheEntry:
        """Convert database row to DiskCacheEntry."""
        return DiskCacheEntry(
            key=row[0],
            file_path=row[1],
            size_bytes=row[2],
            created_at=row[3],
            accessed_at=row[4],
            ttl_seconds=row[5],
            checksum=row[6],
            compressed=bool(row[7])
        )

    async def _checkpoint_loop(self) -> None:
        """Background task for periodic checkpoints."""
        while self._running:
            try:
                await asyncio.sleep(self._checkpoint_interval)
                # SQLite auto-commits, but we can optimize with PRAGMA
                self._db.execute("PRAGMA optimize")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in checkpoint loop: {e}")

    async def _cleanup_loop(self) -> None:
        """Background task for cleanup."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "entry_count": self._get_entry_count(),
            "size_bytes": self._current_size_bytes,
            "size_mb": self._current_size_bytes / (1024 * 1024),
            "size_gb": self._current_size_bytes / (1024 * 1024 * 1024),
            "max_size_gb": self._max_size_bytes / (1024 * 1024 * 1024),
            "utilization": self._current_size_bytes / self._max_size_bytes
            if self._max_size_bytes > 0 else 0,
            "hits": self._hits,
            "misses": self._misses,
            "sets": self._sets,
            "evictions": self._evictions,
            "corruption_errors": self._corruption_errors,
            "hit_rate": self._hits / (self._hits + self._misses)
            if (self._hits + self._misses) > 0 else 0,
        }

    async def vacuum(self) -> None:
        """
        Optimize database and reclaim disk space.

        Should be called periodically or when cache is idle.
        """
        try:
            logger.info("Starting cache vacuum...")
            self._db.execute("VACUUM")
            self._db.commit()
            logger.info("Cache vacuum completed")
        except Exception as e:
            logger.error(f"Error during vacuum: {e}")

    async def get_top_keys(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N most accessed keys.

        Args:
            n: Number of keys to return

        Returns:
            List of key info dicts
        """
        cursor = self._db.execute("""
            SELECT key, size_bytes, accessed_at
            FROM cache_entries
            ORDER BY accessed_at DESC
            LIMIT ?
        """, (n,))

        return [
            {
                "key": row[0],
                "size_mb": row[1] / (1024 * 1024),
                "accessed_at": row[2]
            }
            for row in cursor.fetchall()
        ]
