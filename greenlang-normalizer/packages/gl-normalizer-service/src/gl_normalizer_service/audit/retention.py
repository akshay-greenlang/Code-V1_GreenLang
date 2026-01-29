"""
Retention Policy Enforcement for GL-FOUND-X-003 Audit Events.

This module provides retention policy enforcement for audit data,
ensuring compliance with regulatory requirements (default 7 years)
while optimizing storage costs by purging expired data.

Key Features:
    - Configurable retention periods per storage tier
    - Automatic purging of expired data
    - Batch processing for large datasets
    - Audit logging of purge operations
    - Dry-run mode for validation

Retention Tiers:
    1. Hot (30 days): Kafka topics, outbox table
    2. Warm (1 year): Recent Parquet files
    3. Cold (6 years): Archived Parquet files
    4. Purge (>7 years): Data deleted

Example:
    >>> from gl_normalizer_service.audit.retention import (
    ...     RetentionEnforcer,
    ...     apply_retention,
    ... )
    >>> from gl_normalizer_service.audit.models import RetentionPolicy
    >>> policy = RetentionPolicy(total_retention_years=7)
    >>> enforcer = RetentionEnforcer(policy, storage)
    >>> deleted_count = await enforcer.enforce()

NFR Compliance:
    - NFR-036: 7-year retention for regulatory compliance (SOX, GDPR, etc.)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, Tuple

from gl_normalizer_service.audit.models import (
    ColdStoragePartition,
    OutboxConfig,
    RetentionPolicy,
)

logger = logging.getLogger(__name__)


class StorageBackend(Protocol):
    """Protocol for storage backends that support retention."""

    async def list_partitions(
        self,
        org_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[str]:
        """List partition paths."""
        ...

    async def delete_path(self, path: str) -> bool:
        """Delete a path."""
        ...


class DatabaseBackend(Protocol):
    """Protocol for database backends that support retention."""

    async def execute(self, query: str, *args: Any) -> Any:
        """Execute a query."""
        ...

    async def fetch(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        """Fetch rows."""
        ...


class RetentionEnforcer:
    """
    Enforces retention policies on audit data.

    This class manages the lifecycle of audit data across storage tiers,
    automatically purging data that exceeds the configured retention period.

    Retention Strategy:
        - Hot tier (Kafka/DB): Data kept for quick access and replay
        - Warm tier: Recent archives for frequent queries
        - Cold tier: Long-term archives for compliance
        - Purge: Data older than retention period is deleted

    Attributes:
        policy: Retention policy configuration.
        storage: Cold storage backend.
        database: Database backend for outbox cleanup.

    Example:
        >>> policy = RetentionPolicy(
        ...     hot_retention_days=30,
        ...     warm_retention_days=365,
        ...     cold_retention_years=6,
        ...     total_retention_years=7,
        ... )
        >>> enforcer = RetentionEnforcer(policy, storage, database)
        >>> stats = await enforcer.enforce()
        >>> print(f"Deleted {stats['deleted_count']} records")
    """

    def __init__(
        self,
        policy: RetentionPolicy,
        storage: Optional[StorageBackend] = None,
        database: Optional[DatabaseBackend] = None,
    ):
        """
        Initialize the retention enforcer.

        Args:
            policy: Retention policy configuration.
            storage: Cold storage backend for Parquet cleanup.
            database: Database backend for outbox cleanup.
        """
        self.policy = policy
        self.storage = storage
        self.database = database

        logger.info(
            "RetentionEnforcer initialized (total_retention=%d years, "
            "hot=%d days, warm=%d days, cold=%d years)",
            policy.total_retention_years,
            policy.hot_retention_days,
            policy.warm_retention_days,
            policy.cold_retention_years,
        )

    async def enforce(
        self,
        org_ids: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Enforce retention policy by purging expired data.

        Processes all configured storage tiers and deletes data older
        than the retention period.

        Args:
            org_ids: Optional list of org_ids to process (None = all).
            dry_run: If True, log what would be deleted without deleting.

        Returns:
            Statistics dictionary with counts of deleted items.

        Example:
            >>> stats = await enforcer.enforce(dry_run=True)
            >>> print(f"Would delete {stats['cold_partitions_count']} partitions")
        """
        logger.info(
            "Starting retention enforcement (dry_run=%s, orgs=%s)",
            dry_run,
            org_ids or "all",
        )

        start_time = datetime.utcnow()
        stats = {
            "dry_run": dry_run,
            "start_time": start_time.isoformat(),
            "hot_records_deleted": 0,
            "cold_partitions_deleted": 0,
            "cold_bytes_freed": 0,
            "errors": [],
        }

        # Enforce on outbox table (hot tier)
        if self.database and self.policy.purge_enabled:
            hot_stats = await self._enforce_hot_tier(org_ids, dry_run)
            stats["hot_records_deleted"] = hot_stats["deleted_count"]
            stats["errors"].extend(hot_stats.get("errors", []))

        # Enforce on cold storage
        if self.storage and self.policy.purge_enabled:
            cold_stats = await self._enforce_cold_tier(org_ids, dry_run)
            stats["cold_partitions_deleted"] = cold_stats["deleted_count"]
            stats["cold_bytes_freed"] = cold_stats.get("bytes_freed", 0)
            stats["errors"].extend(cold_stats.get("errors", []))

        end_time = datetime.utcnow()
        stats["end_time"] = end_time.isoformat()
        stats["duration_seconds"] = (end_time - start_time).total_seconds()

        logger.info(
            "Retention enforcement complete: %d hot records, %d cold partitions "
            "deleted in %.2f seconds",
            stats["hot_records_deleted"],
            stats["cold_partitions_deleted"],
            stats["duration_seconds"],
        )

        return stats

    async def _enforce_hot_tier(
        self,
        org_ids: Optional[List[str]],
        dry_run: bool,
    ) -> Dict[str, Any]:
        """
        Enforce retention on hot tier (outbox table).

        Deletes published records older than hot_retention_days.

        Args:
            org_ids: Optional org_id filter.
            dry_run: If True, count without deleting.

        Returns:
            Statistics dictionary.
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.policy.hot_retention_days)
        deleted_count = 0
        errors = []

        logger.info(
            "Enforcing hot tier retention (cutoff=%s, dry_run=%s)",
            cutoff_date.isoformat(),
            dry_run,
        )

        try:
            if dry_run:
                # Count records that would be deleted
                query = """
                    SELECT COUNT(*) as count
                    FROM audit_outbox
                    WHERE status IN ('published', 'archived')
                      AND published_at < $1
                """
                if org_ids:
                    query += " AND org_id = ANY($2)"
                    rows = await self.database.fetch(query, cutoff_date, org_ids)
                else:
                    rows = await self.database.fetch(query, cutoff_date)

                deleted_count = rows[0]["count"] if rows else 0
                logger.info(
                    "Dry run: would delete %d hot tier records",
                    deleted_count,
                )
            else:
                # Delete in batches
                total_deleted = 0
                while True:
                    query = """
                        DELETE FROM audit_outbox
                        WHERE id IN (
                            SELECT id FROM audit_outbox
                            WHERE status IN ('published', 'archived')
                              AND published_at < $1
                    """
                    if org_ids:
                        query += " AND org_id = ANY($2)"
                        query += f" LIMIT {self.policy.purge_batch_size})"
                        result = await self.database.execute(query, cutoff_date, org_ids)
                    else:
                        query += f" LIMIT {self.policy.purge_batch_size})"
                        result = await self.database.execute(query, cutoff_date)

                    # Parse result to get deleted count
                    batch_deleted = _parse_delete_result(result)
                    total_deleted += batch_deleted

                    if batch_deleted < self.policy.purge_batch_size:
                        break

                    # Brief pause between batches
                    await asyncio.sleep(0.1)

                deleted_count = total_deleted
                logger.info("Deleted %d hot tier records", deleted_count)

        except Exception as e:
            error_msg = f"Hot tier retention failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

        return {
            "deleted_count": deleted_count,
            "errors": errors,
        }

    async def _enforce_cold_tier(
        self,
        org_ids: Optional[List[str]],
        dry_run: bool,
    ) -> Dict[str, Any]:
        """
        Enforce retention on cold tier (Parquet partitions).

        Deletes partitions older than total_retention_years.

        Args:
            org_ids: Optional org_id filter.
            dry_run: If True, count without deleting.

        Returns:
            Statistics dictionary.
        """
        cutoff_date = self.policy.get_purge_cutoff().strftime("%Y-%m-%d")
        deleted_count = 0
        bytes_freed = 0
        errors = []

        logger.info(
            "Enforcing cold tier retention (cutoff=%s, dry_run=%s)",
            cutoff_date,
            dry_run,
        )

        try:
            # Get all partitions to check
            # If no org_ids specified, we need to discover them
            orgs_to_process = org_ids or await self._discover_org_ids()

            for org_id in orgs_to_process:
                try:
                    # List partitions older than cutoff
                    partitions = await self.storage.list_partitions(
                        org_id=org_id,
                        end_date=cutoff_date,
                    )

                    for partition_path in partitions:
                        if dry_run:
                            deleted_count += 1
                            logger.debug(
                                "Dry run: would delete partition %s",
                                partition_path,
                            )
                        else:
                            success = await self.storage.delete_path(partition_path)
                            if success:
                                deleted_count += 1
                                logger.debug("Deleted partition %s", partition_path)

                except Exception as e:
                    error_msg = f"Cold tier cleanup failed for org {org_id}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            logger.info(
                "Cold tier retention complete: %d partitions %s",
                deleted_count,
                "would be deleted" if dry_run else "deleted",
            )

        except Exception as e:
            error_msg = f"Cold tier retention failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

        return {
            "deleted_count": deleted_count,
            "bytes_freed": bytes_freed,
            "errors": errors,
        }

    async def _discover_org_ids(self) -> List[str]:
        """
        Discover all org_ids from the outbox table.

        Returns:
            List of unique org_ids.
        """
        if not self.database:
            return []

        try:
            rows = await self.database.fetch(
                "SELECT DISTINCT org_id FROM audit_outbox"
            )
            return [row["org_id"] for row in rows]
        except Exception as e:
            logger.error("Failed to discover org_ids: %s", str(e))
            return []

    async def get_retention_report(
        self,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Generate a retention report for an organization.

        Shows data distribution across tiers and estimates for cleanup.

        Args:
            org_id: Organization ID.

        Returns:
            Retention report dictionary.

        Example:
            >>> report = await enforcer.get_retention_report("org-acme")
            >>> print(f"Hot tier: {report['hot_tier']['count']} records")
            >>> print(f"Cold tier: {report['cold_tier']['partition_count']} partitions")
        """
        report = {
            "org_id": org_id,
            "generated_at": datetime.utcnow().isoformat(),
            "policy": {
                "hot_retention_days": self.policy.hot_retention_days,
                "warm_retention_days": self.policy.warm_retention_days,
                "cold_retention_years": self.policy.cold_retention_years,
                "total_retention_years": self.policy.total_retention_years,
            },
            "hot_tier": {
                "count": 0,
                "oldest_date": None,
                "expiring_soon": 0,
            },
            "cold_tier": {
                "partition_count": 0,
                "oldest_partition": None,
                "expiring_soon": 0,
            },
        }

        # Hot tier stats
        if self.database:
            try:
                # Count and date range
                rows = await self.database.fetch(
                    """
                    SELECT
                        COUNT(*) as count,
                        MIN(created_at) as oldest,
                        SUM(CASE WHEN created_at < $2 THEN 1 ELSE 0 END) as expiring
                    FROM audit_outbox
                    WHERE org_id = $1
                      AND status IN ('published', 'archived')
                    """,
                    org_id,
                    datetime.utcnow() - timedelta(days=self.policy.hot_retention_days - 7),
                )
                if rows:
                    report["hot_tier"]["count"] = rows[0]["count"] or 0
                    report["hot_tier"]["oldest_date"] = (
                        rows[0]["oldest"].isoformat() if rows[0]["oldest"] else None
                    )
                    report["hot_tier"]["expiring_soon"] = rows[0]["expiring"] or 0
            except Exception as e:
                logger.error("Failed to get hot tier stats: %s", str(e))

        # Cold tier stats
        if self.storage:
            try:
                partitions = await self.storage.list_partitions(org_id)
                report["cold_tier"]["partition_count"] = len(partitions)

                if partitions:
                    # Extract dates from paths and find oldest
                    dates = []
                    cutoff = self.policy.get_purge_cutoff().strftime("%Y-%m-%d")
                    for path in partitions:
                        date_str = _extract_date_from_path(path)
                        if date_str:
                            dates.append(date_str)
                            if date_str <= cutoff:
                                report["cold_tier"]["expiring_soon"] += 1

                    if dates:
                        report["cold_tier"]["oldest_partition"] = min(dates)

            except Exception as e:
                logger.error("Failed to get cold tier stats: %s", str(e))

        return report


async def apply_retention(
    storage: StorageBackend,
    policy: RetentionPolicy,
    org_ids: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Apply retention policy to cold storage.

    Convenience function for one-off retention enforcement without
    creating an enforcer instance.

    Args:
        storage: Cold storage backend.
        policy: Retention policy to apply.
        org_ids: Optional org_id filter.
        dry_run: If True, log without deleting.

    Returns:
        Statistics dictionary.

    Example:
        >>> from gl_normalizer_service.audit.retention import apply_retention
        >>> from gl_normalizer_service.audit.models import RetentionPolicy
        >>> policy = RetentionPolicy(total_retention_years=7)
        >>> stats = await apply_retention(storage, policy)
        >>> print(f"Deleted {stats['cold_partitions_deleted']} partitions")
    """
    enforcer = RetentionEnforcer(policy, storage=storage)
    return await enforcer.enforce(org_ids=org_ids, dry_run=dry_run)


def _parse_delete_result(result: Any) -> int:
    """
    Parse delete result to get affected row count.

    Args:
        result: Result from database execute.

    Returns:
        Number of rows deleted.
    """
    if result is None:
        return 0
    if isinstance(result, int):
        return result
    if isinstance(result, str):
        # Parse "DELETE n" format from asyncpg
        parts = result.split()
        if len(parts) >= 2 and parts[0] == "DELETE":
            try:
                return int(parts[1])
            except ValueError:
                pass
    return 0


def _extract_date_from_path(path: str) -> Optional[str]:
    """
    Extract date from a partition path.

    Expects format: .../org_id/YYYY/MM/DD/events.parquet

    Args:
        path: Partition path.

    Returns:
        Date string (YYYY-MM-DD) or None.
    """
    try:
        parts = path.rstrip("/").split("/")
        # Look for year/month/day pattern
        for i, part in enumerate(parts):
            if len(part) == 4 and part.isdigit() and 1900 < int(part) < 2100:
                if i + 2 < len(parts):
                    year = part
                    month = parts[i + 1]
                    day = parts[i + 2]
                    if len(month) == 2 and len(day) == 2:
                        return f"{year}-{month}-{day}"
    except (IndexError, ValueError):
        pass
    return None


class RetentionScheduler:
    """
    Scheduler for periodic retention enforcement.

    Runs retention enforcement on a configurable schedule.

    Attributes:
        enforcer: Retention enforcer instance.
        interval_hours: Hours between enforcement runs.
        _running: Whether the scheduler is running.

    Example:
        >>> scheduler = RetentionScheduler(enforcer, interval_hours=24)
        >>> await scheduler.start()
        >>> # Runs daily
        >>> await scheduler.stop()
    """

    def __init__(
        self,
        enforcer: RetentionEnforcer,
        interval_hours: int = 24,
    ):
        """
        Initialize the retention scheduler.

        Args:
            enforcer: Retention enforcer instance.
            interval_hours: Hours between enforcement runs.
        """
        self.enforcer = enforcer
        self.interval_hours = interval_hours
        self._running = False
        self._task: Optional[asyncio.Task] = None

        logger.info(
            "RetentionScheduler initialized (interval=%d hours)",
            interval_hours,
        )

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            raise RuntimeError("Scheduler is already running")

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Retention scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Retention scheduler stopped")

    async def _run_loop(self) -> None:
        """Internal scheduler loop."""
        while self._running:
            try:
                logger.info("Running scheduled retention enforcement")
                stats = await self.enforcer.enforce()
                logger.info(
                    "Scheduled retention complete: %d hot, %d cold deleted",
                    stats["hot_records_deleted"],
                    stats["cold_partitions_deleted"],
                )
            except Exception as e:
                logger.exception("Scheduled retention failed: %s", str(e))

            # Wait for next run
            await asyncio.sleep(self.interval_hours * 3600)
