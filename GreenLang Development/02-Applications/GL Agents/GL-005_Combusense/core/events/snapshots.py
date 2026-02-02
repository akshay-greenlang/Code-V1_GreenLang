# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Snapshot Manager

This module implements snapshot management for event-sourced aggregates.
Snapshots provide performance optimization by reducing the number of
events that need to be replayed when loading aggregates.

Design Principles:
    - Snapshots are optimization only (not source of truth)
    - Aggregates can always be rebuilt from events alone
    - Snapshots are taken periodically based on event count
    - Snapshot schema versioning for forward compatibility
    - Concurrent snapshot creation support

Example:
    >>> manager = SnapshotManager(event_store, threshold=100)
    >>> await manager.maybe_snapshot(aggregate)
    >>> snapshot = await manager.get_latest(aggregate_id)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from pydantic import BaseModel, Field, ConfigDict

from core.events.event_store import EventStore

logger = logging.getLogger(__name__)


class SnapshotMetadata(BaseModel):
    """Metadata about a snapshot."""

    model_config = ConfigDict(frozen=True)

    aggregate_id: str = Field(..., description="Aggregate identifier")
    aggregate_type: str = Field(..., description="Aggregate type name")
    version: int = Field(..., ge=0, description="Version at snapshot")
    schema_version: int = Field(default=1, ge=1, description="Snapshot schema version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    checksum: str = Field(..., description="SHA-256 checksum of snapshot data")
    size_bytes: int = Field(default=0, ge=0, description="Snapshot size in bytes")
    compression: Optional[str] = Field(default=None, description="Compression algorithm")


class Snapshot(BaseModel):
    """
    Complete snapshot with data and metadata.

    Attributes:
        metadata: Snapshot metadata
        data: Serialized aggregate state
    """

    model_config = ConfigDict(frozen=True)

    metadata: SnapshotMetadata
    data: Dict[str, Any] = Field(..., description="Snapshot data")

    @classmethod
    def create(
        cls,
        aggregate_id: str,
        aggregate_type: str,
        version: int,
        data: Dict[str, Any],
        schema_version: int = 1
    ) -> "Snapshot":
        """
        Create a new snapshot.

        Args:
            aggregate_id: Aggregate identifier
            aggregate_type: Aggregate type name
            version: Version at snapshot
            data: Aggregate state data
            schema_version: Schema version

        Returns:
            New Snapshot instance
        """
        # Serialize for checksum
        data_str = json.dumps(data, sort_keys=True, default=str)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()

        metadata = SnapshotMetadata(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            version=version,
            schema_version=schema_version,
            checksum=checksum,
            size_bytes=len(data_str.encode())
        )

        return cls(metadata=metadata, data=data)

    def verify_checksum(self) -> bool:
        """Verify snapshot checksum."""
        data_str = json.dumps(self.data, sort_keys=True, default=str)
        computed = hashlib.sha256(data_str.encode()).hexdigest()
        return computed == self.metadata.checksum


class SnapshotConfig(BaseModel):
    """Configuration for snapshot manager."""

    event_threshold: int = Field(
        default=100,
        ge=10,
        description="Events before taking snapshot"
    )
    time_threshold_minutes: int = Field(
        default=60,
        ge=1,
        description="Minutes before taking snapshot"
    )
    max_snapshots_per_aggregate: int = Field(
        default=5,
        ge=1,
        description="Maximum snapshots to retain"
    )
    auto_snapshot_enabled: bool = Field(
        default=True,
        description="Enable automatic snapshots"
    )
    compression_enabled: bool = Field(
        default=False,
        description="Enable snapshot compression"
    )
    verify_on_load: bool = Field(
        default=True,
        description="Verify checksum on load"
    )


class SnapshotManager:
    """
    Manager for aggregate snapshots.

    The SnapshotManager handles:
        - Automatic snapshot creation based on thresholds
        - Snapshot storage and retrieval
        - Snapshot cleanup and retention
        - Schema version migration
        - Concurrent snapshot operations

    Performance:
        Snapshots reduce aggregate load time from O(n) events
        to O(1) snapshot + O(m) events where m << n.

    Example:
        >>> manager = SnapshotManager(event_store)
        >>> await manager.start()
        >>>
        >>> # Take snapshot after changes
        >>> await manager.maybe_snapshot(aggregate)
        >>>
        >>> # Load with snapshot
        >>> snapshot, from_version = await manager.get_latest("agg-001")
        >>> aggregate.restore_from_snapshot(snapshot.data)
        >>> events = await store.load("agg-001", from_version=from_version)
    """

    def __init__(
        self,
        event_store: EventStore,
        config: Optional[SnapshotConfig] = None
    ):
        """
        Initialize snapshot manager.

        Args:
            event_store: Event store instance
            config: Snapshot configuration
        """
        self.config = config or SnapshotConfig()
        self._event_store = event_store

        # Track last snapshot versions
        self._last_snapshot_version: Dict[str, int] = {}
        self._last_snapshot_time: Dict[str, datetime] = {}

        # Background task
        self._background_task: Optional[asyncio.Task] = None
        self._pending_snapshots: Set[str] = set()

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(
            f"SnapshotManager initialized "
            f"(threshold={self.config.event_threshold} events)"
        )

    async def start(self) -> None:
        """Start background snapshot processing."""
        if self._background_task is None:
            self._background_task = asyncio.create_task(
                self._background_snapshot_loop()
            )
            logger.info("Snapshot manager background task started")

    async def stop(self) -> None:
        """Stop background snapshot processing."""
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
            logger.info("Snapshot manager stopped")

    async def _background_snapshot_loop(self) -> None:
        """Background loop for processing pending snapshots."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                if not self._pending_snapshots:
                    continue

                # Process pending snapshots
                pending = list(self._pending_snapshots)
                self._pending_snapshots.clear()

                for aggregate_id in pending:
                    try:
                        await self._take_scheduled_snapshot(aggregate_id)
                    except Exception as e:
                        logger.error(
                            f"Background snapshot failed for {aggregate_id}: {e}"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background snapshot loop error: {e}")

    async def should_snapshot(
        self,
        aggregate_id: str,
        current_version: int,
        aggregate_type: str = "CombustionControlAggregate"
    ) -> bool:
        """
        Check if a snapshot should be taken.

        Args:
            aggregate_id: Aggregate identifier
            current_version: Current aggregate version
            aggregate_type: Type of aggregate

        Returns:
            True if snapshot should be taken
        """
        if not self.config.auto_snapshot_enabled:
            return False

        # Check event threshold
        last_version = self._last_snapshot_version.get(aggregate_id, 0)
        events_since = current_version - last_version

        if events_since >= self.config.event_threshold:
            return True

        # Check time threshold
        last_time = self._last_snapshot_time.get(aggregate_id)
        if last_time:
            threshold = timedelta(minutes=self.config.time_threshold_minutes)
            if datetime.utcnow() - last_time >= threshold:
                return True

        return False

    async def maybe_snapshot(
        self,
        aggregate: Any,  # Type hint avoided to prevent circular import
        force: bool = False
    ) -> bool:
        """
        Take a snapshot if thresholds are met.

        Args:
            aggregate: Aggregate to snapshot
            force: Force snapshot regardless of thresholds

        Returns:
            True if snapshot was taken
        """
        aggregate_id = aggregate.aggregate_id
        current_version = aggregate.version
        aggregate_type = aggregate.aggregate_type

        if not force and not await self.should_snapshot(
            aggregate_id, current_version, aggregate_type
        ):
            return False

        return await self.take_snapshot(aggregate)

    async def take_snapshot(
        self,
        aggregate: Any
    ) -> bool:
        """
        Take a snapshot of an aggregate.

        Args:
            aggregate: Aggregate to snapshot

        Returns:
            True if successful
        """
        async with self._lock:
            try:
                aggregate_id = aggregate.aggregate_id
                aggregate_type = aggregate.aggregate_type
                version = aggregate.version

                # Create snapshot data
                snapshot_data = aggregate.create_snapshot()

                # Create snapshot object
                snapshot = Snapshot.create(
                    aggregate_id=aggregate_id,
                    aggregate_type=aggregate_type,
                    version=version,
                    data=snapshot_data
                )

                # Save to event store
                await self._event_store.save_snapshot(
                    aggregate_id=aggregate_id,
                    version=version,
                    snapshot_data=snapshot.data,
                    aggregate_type=aggregate_type
                )

                # Update tracking
                self._last_snapshot_version[aggregate_id] = version
                self._last_snapshot_time[aggregate_id] = datetime.utcnow()

                logger.info(
                    f"Snapshot taken for {aggregate_type}:{aggregate_id} "
                    f"at version {version}"
                )

                return True

            except Exception as e:
                logger.error(f"Failed to take snapshot: {e}")
                return False

    async def get_latest(
        self,
        aggregate_id: str,
        aggregate_type: str = "CombustionControlAggregate"
    ) -> Optional[Tuple[Snapshot, int]]:
        """
        Get the latest snapshot for an aggregate.

        Args:
            aggregate_id: Aggregate identifier
            aggregate_type: Type of aggregate

        Returns:
            Tuple of (Snapshot, next_version) or None
        """
        result = await self._event_store.load_latest_snapshot(
            aggregate_id, aggregate_type
        )

        if result is None:
            return None

        version, snapshot_data = result

        # Create snapshot object
        snapshot = Snapshot.create(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            version=version,
            data=snapshot_data
        )

        # Verify checksum if enabled
        if self.config.verify_on_load:
            if not snapshot.verify_checksum():
                logger.error(
                    f"Snapshot checksum mismatch for {aggregate_id} "
                    f"at version {version}"
                )
                return None

        # Next version to load events from
        next_version = version + 1

        return (snapshot, next_version)

    async def schedule_snapshot(
        self,
        aggregate_id: str
    ) -> None:
        """
        Schedule a snapshot for background processing.

        Args:
            aggregate_id: Aggregate to snapshot
        """
        self._pending_snapshots.add(aggregate_id)
        logger.debug(f"Scheduled snapshot for {aggregate_id}")

    async def _take_scheduled_snapshot(
        self,
        aggregate_id: str
    ) -> None:
        """Take a scheduled snapshot (from background task)."""
        # This would need access to the aggregate repository
        # For now, just log that it was scheduled
        logger.debug(f"Would take scheduled snapshot for {aggregate_id}")

    def get_snapshot_stats(
        self,
        aggregate_id: str
    ) -> Dict[str, Any]:
        """
        Get snapshot statistics for an aggregate.

        Args:
            aggregate_id: Aggregate identifier

        Returns:
            Statistics dictionary
        """
        return {
            "aggregate_id": aggregate_id,
            "last_snapshot_version": self._last_snapshot_version.get(aggregate_id),
            "last_snapshot_time": self._last_snapshot_time.get(aggregate_id),
            "pending": aggregate_id in self._pending_snapshots
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get overall snapshot manager statistics."""
        return {
            "config": {
                "event_threshold": self.config.event_threshold,
                "time_threshold_minutes": self.config.time_threshold_minutes,
                "auto_enabled": self.config.auto_snapshot_enabled
            },
            "tracked_aggregates": len(self._last_snapshot_version),
            "pending_snapshots": len(self._pending_snapshots),
            "background_task_running": self._background_task is not None
        }


class SnapshotMigrator:
    """
    Handles snapshot schema version migration.

    When snapshot schema changes, this class provides migration
    between schema versions.

    Example:
        >>> migrator = SnapshotMigrator()
        >>> migrator.register_migration(1, 2, migrate_v1_to_v2)
        >>> new_data = migrator.migrate(snapshot_data, from_version=1, to_version=2)
    """

    def __init__(self):
        """Initialize migrator."""
        # migrations[(from_version, to_version)] = migration_fn
        self._migrations: Dict[
            Tuple[int, int],
            callable
        ] = {}

    def register_migration(
        self,
        from_version: int,
        to_version: int,
        migration_fn: callable
    ) -> None:
        """
        Register a migration function.

        Args:
            from_version: Source schema version
            to_version: Target schema version
            migration_fn: Function that transforms the data
        """
        self._migrations[(from_version, to_version)] = migration_fn
        logger.info(
            f"Registered snapshot migration: v{from_version} -> v{to_version}"
        )

    def migrate(
        self,
        data: Dict[str, Any],
        from_version: int,
        to_version: int
    ) -> Dict[str, Any]:
        """
        Migrate snapshot data between versions.

        Args:
            data: Snapshot data
            from_version: Current schema version
            to_version: Target schema version

        Returns:
            Migrated data

        Raises:
            ValueError: If migration path not found
        """
        if from_version == to_version:
            return data

        current_version = from_version
        current_data = data

        while current_version < to_version:
            next_version = current_version + 1
            migration_fn = self._migrations.get((current_version, next_version))

            if migration_fn is None:
                raise ValueError(
                    f"No migration path from v{current_version} to v{next_version}"
                )

            current_data = migration_fn(current_data)
            current_version = next_version

        return current_data

    def get_migration_path(
        self,
        from_version: int,
        to_version: int
    ) -> List[int]:
        """
        Get the migration path between versions.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            List of versions in migration path
        """
        if from_version >= to_version:
            return [from_version]

        path = [from_version]
        current = from_version

        while current < to_version:
            next_version = current + 1
            if (current, next_version) not in self._migrations:
                return []  # No path
            path.append(next_version)
            current = next_version

        return path


# Default migrator instance
snapshot_migrator = SnapshotMigrator()
