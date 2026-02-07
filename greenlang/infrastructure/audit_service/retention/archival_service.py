# -*- coding: utf-8 -*-
"""
Archival Service - Centralized Audit Logging Service (SEC-005)

Manages archival operations for audit data to S3 (Parquet) and Glacier.
Provides efficient batch archival, restore operations, and integrity
verification for long-term compliance storage.

Features:
    - Parquet file generation with columnar compression
    - S3 multipart uploads for large archives
    - Glacier Deep Archive for 7-year retention
    - Restore from archive to temporary PostgreSQL tables
    - SHA-256 integrity checksums for all archives
    - Partition-based organization (year/month)

Storage Layout:
    s3://greenlang-audit-archive/
        cold/
            2026/
                01/
                    audit_2026_01_01_part_001.parquet
                    audit_2026_01_01_part_001.parquet.sha256
        glacier/
            2025/
                audit_2025_01.parquet.gz
                audit_2025_01.parquet.gz.sha256

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import gzip
import hashlib
import io
import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class ArchivalStatus(str, Enum):
    """Status of an archival operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RESTORED = "restored"
    GLACIER_PENDING = "glacier_pending"
    GLACIER_COMPLETED = "glacier_completed"


# S3 bucket configuration
DEFAULT_ARCHIVE_BUCKET = "greenlang-audit-archive"
COLD_PREFIX = "cold"
GLACIER_PREFIX = "glacier"

# File size limits
MAX_PARQUET_FILE_SIZE_MB = 256
MULTIPART_THRESHOLD_MB = 100
GLACIER_TRANSITION_AGE_DAYS = 365

# Parquet compression
PARQUET_COMPRESSION = "snappy"
PARQUET_ROW_GROUP_SIZE = 100_000


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ArchiveMetadata:
    """Metadata for an archived partition.

    Attributes:
        archive_id: Unique identifier for the archive.
        partition_key: Partition identifier (e.g., "2026-01").
        s3_bucket: S3 bucket name.
        s3_key: S3 object key.
        storage_class: S3 storage class (STANDARD, GLACIER, etc.).
        file_size_bytes: Size of the archive file.
        row_count: Number of audit events in the archive.
        checksum_sha256: SHA-256 hash of the archive file.
        archived_at: When the archive was created.
        start_timestamp: Earliest event timestamp in archive.
        end_timestamp: Latest event timestamp in archive.
        event_types: List of event types included.
        tenant_ids: List of tenant IDs included.
    """

    archive_id: str
    partition_key: str
    s3_bucket: str
    s3_key: str
    storage_class: str
    file_size_bytes: int
    row_count: int
    checksum_sha256: str
    archived_at: datetime
    start_timestamp: datetime
    end_timestamp: datetime
    event_types: List[str] = field(default_factory=list)
    tenant_ids: List[str] = field(default_factory=list)


@dataclass
class RestoreRequest:
    """Request to restore archived data.

    Attributes:
        archive_id: ID of the archive to restore.
        restore_id: Unique ID for this restore operation.
        requested_at: When the restore was requested.
        expires_at: When the restored data should be deleted.
        status: Current status of the restore.
        temporary_table: Name of temporary table for restored data.
    """

    archive_id: str
    restore_id: str
    requested_at: datetime
    expires_at: datetime
    status: ArchivalStatus
    temporary_table: Optional[str] = None


# ---------------------------------------------------------------------------
# ArchivalService
# ---------------------------------------------------------------------------


class ArchivalService:
    """Manages archival operations for audit data.

    Handles:
    - Exporting audit events to Parquet files
    - Uploading to S3 with appropriate storage class
    - Migrating from S3 Standard to Glacier
    - Restoring archived data for compliance queries

    Thread-safe: Uses async operations for all I/O.

    Example:
        >>> service = ArchivalService(db_pool, s3_client)
        >>>
        >>> # Archive a date range to S3
        >>> archive = await service.archive_to_s3(
        ...     start_date=datetime(2026, 1, 1),
        ...     end_date=datetime(2026, 1, 31),
        ... )
        >>> print(f"Archived {archive.row_count} events")
        >>>
        >>> # Restore from archive
        >>> temp_table = await service.restore_from_archive("2026-01")
        >>> print(f"Restored to {temp_table}")
    """

    def __init__(
        self,
        db_pool: Any = None,
        s3_client: Any = None,
        bucket_name: str = DEFAULT_ARCHIVE_BUCKET,
    ) -> None:
        """Initialize archival service.

        Args:
            db_pool: Async database connection pool.
            s3_client: Boto3 S3 client (sync or async).
            bucket_name: S3 bucket for archives.
        """
        self._db_pool = db_pool
        self._s3_client = s3_client
        self._bucket = bucket_name

        logger.info(
            "ArchivalService initialized: bucket=%s",
            bucket_name,
        )

    async def archive_to_s3(
        self,
        start_date: datetime,
        end_date: datetime,
        tenant_id: Optional[str] = None,
        event_types: Optional[List[str]] = None,
    ) -> ArchiveMetadata:
        """Archive audit events to S3 as Parquet files.

        Exports audit events from PostgreSQL to Parquet format and
        uploads to S3. Events are organized by year/month partitions.

        Args:
            start_date: Start of the date range (inclusive).
            end_date: End of the date range (exclusive).
            tenant_id: Optional tenant filter.
            event_types: Optional event type filter.

        Returns:
            Metadata about the created archive.

        Raises:
            RuntimeError: If database or S3 client not configured.
            ValueError: If date range is invalid.
        """
        if self._db_pool is None or self._s3_client is None:
            raise RuntimeError("Database pool and S3 client required")

        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")

        archive_id = str(uuid.uuid4())
        partition_key = start_date.strftime("%Y-%m")

        logger.info(
            "Starting S3 archival: archive_id=%s partition=%s range=%s to %s",
            archive_id,
            partition_key,
            start_date.isoformat(),
            end_date.isoformat(),
        )

        # Build query
        query, params = self._build_export_query(
            start_date, end_date, tenant_id, event_types
        )

        # Export to Parquet
        parquet_path, row_count, actual_start, actual_end = (
            await self._export_to_parquet(query, params)
        )

        if row_count == 0:
            logger.info(
                "No events to archive for partition %s",
                partition_key,
            )
            # Clean up temp file
            Path(parquet_path).unlink(missing_ok=True)
            raise ValueError(f"No events found for date range")

        # Calculate checksum
        checksum = self._calculate_checksum(parquet_path)

        # Upload to S3
        s3_key = self._generate_s3_key(
            partition_key,
            start_date,
            archive_id,
        )

        file_size = Path(parquet_path).stat().st_size

        await self._upload_to_s3(
            parquet_path,
            s3_key,
            metadata={
                "archive_id": archive_id,
                "partition_key": partition_key,
                "row_count": str(row_count),
                "checksum_sha256": checksum,
                "start_timestamp": actual_start.isoformat(),
                "end_timestamp": actual_end.isoformat(),
            },
        )

        # Upload checksum file
        checksum_key = f"{s3_key}.sha256"
        await self._upload_checksum(checksum, checksum_key)

        # Clean up temp file
        Path(parquet_path).unlink(missing_ok=True)

        # Mark events as archived in database
        await self._mark_events_archived(
            start_date, end_date, tenant_id, archive_id
        )

        # Record archive metadata
        metadata = ArchiveMetadata(
            archive_id=archive_id,
            partition_key=partition_key,
            s3_bucket=self._bucket,
            s3_key=s3_key,
            storage_class="STANDARD",
            file_size_bytes=file_size,
            row_count=row_count,
            checksum_sha256=checksum,
            archived_at=datetime.now(timezone.utc),
            start_timestamp=actual_start,
            end_timestamp=actual_end,
            event_types=event_types or [],
            tenant_ids=[tenant_id] if tenant_id else [],
        )

        await self._save_archive_metadata(metadata)

        logger.info(
            "S3 archival completed: archive_id=%s events=%d size=%d bytes",
            archive_id,
            row_count,
            file_size,
        )

        return metadata

    async def migrate_to_glacier(
        self,
        partition_key: str,
    ) -> bool:
        """Migrate a partition from S3 Standard to Glacier Deep Archive.

        Changes the storage class of archived Parquet files to
        GLACIER_DEEP_ARCHIVE for long-term retention at reduced cost.

        Args:
            partition_key: Partition to migrate (e.g., "2025-01").

        Returns:
            True if migration was successful.
        """
        if self._s3_client is None:
            raise RuntimeError("S3 client required")

        logger.info(
            "Starting Glacier migration for partition %s",
            partition_key,
        )

        # List objects in the partition
        prefix = f"{COLD_PREFIX}/{partition_key.replace('-', '/')}/"

        try:
            response = self._s3_client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=prefix,
            )

            objects = response.get("Contents", [])
            if not objects:
                logger.warning(
                    "No objects found for partition %s",
                    partition_key,
                )
                return False

            # Change storage class for each object
            for obj in objects:
                key = obj["Key"]

                # Skip if already in Glacier
                head = self._s3_client.head_object(
                    Bucket=self._bucket,
                    Key=key,
                )
                if head.get("StorageClass") in (
                    "GLACIER",
                    "GLACIER_DEEP_ARCHIVE",
                ):
                    logger.debug(
                        "Object %s already in Glacier, skipping",
                        key,
                    )
                    continue

                # Copy to new storage class
                copy_source = {"Bucket": self._bucket, "Key": key}

                self._s3_client.copy_object(
                    Bucket=self._bucket,
                    Key=key,
                    CopySource=copy_source,
                    StorageClass="DEEP_ARCHIVE",
                    MetadataDirective="COPY",
                )

                logger.debug(
                    "Migrated %s to Glacier Deep Archive",
                    key,
                )

            # Update metadata in database
            await self._update_archive_storage_class(
                partition_key,
                "DEEP_ARCHIVE",
            )

            logger.info(
                "Glacier migration completed for partition %s: %d objects",
                partition_key,
                len(objects),
            )

            return True

        except Exception as exc:
            logger.error(
                "Glacier migration failed for partition %s: %s",
                partition_key,
                str(exc),
            )
            raise

    async def restore_from_archive(
        self,
        partition_key: str,
        expires_in_days: int = 7,
    ) -> str:
        """Restore archived data to a temporary PostgreSQL table.

        For data in Glacier, initiates a restore request and returns
        the restore ID. For Standard storage, immediately creates a
        temporary table.

        Args:
            partition_key: Partition to restore (e.g., "2025-01").
            expires_in_days: How long to keep restored data.

        Returns:
            Temporary table name containing restored data.

        Raises:
            RuntimeError: If database or S3 client not configured.
            ValueError: If partition not found.
        """
        if self._db_pool is None or self._s3_client is None:
            raise RuntimeError("Database pool and S3 client required")

        logger.info(
            "Starting restore from archive: partition=%s expires_in=%d days",
            partition_key,
            expires_in_days,
        )

        # Get archive metadata
        metadata = await self._get_archive_metadata(partition_key)
        if metadata is None:
            raise ValueError(f"Archive not found for partition {partition_key}")

        # Check if in Glacier
        if metadata.storage_class in ("GLACIER", "DEEP_ARCHIVE"):
            # Initiate Glacier restore
            restore_id = await self._initiate_glacier_restore(
                metadata,
                expires_in_days,
            )
            raise ValueError(
                f"Archive is in Glacier. Restore initiated, ID: {restore_id}. "
                f"Check back in 12-48 hours."
            )

        # Download from S3
        local_path = await self._download_from_s3(metadata.s3_key)

        try:
            # Verify checksum
            actual_checksum = self._calculate_checksum(local_path)
            if actual_checksum != metadata.checksum_sha256:
                raise ValueError(
                    f"Checksum mismatch: expected {metadata.checksum_sha256}, "
                    f"got {actual_checksum}"
                )

            # Create temporary table
            temp_table = f"audit_restore_{partition_key.replace('-', '_')}_{uuid.uuid4().hex[:8]}"

            await self._create_restore_table(temp_table)

            # Load Parquet data into table
            row_count = await self._load_parquet_to_table(local_path, temp_table)

            logger.info(
                "Restore completed: partition=%s table=%s rows=%d",
                partition_key,
                temp_table,
                row_count,
            )

            # Schedule table cleanup
            await self._schedule_table_cleanup(temp_table, expires_in_days)

            return temp_table

        finally:
            # Clean up temp file
            Path(local_path).unlink(missing_ok=True)

    # -------------------------------------------------------------------------
    # Internal helper methods
    # -------------------------------------------------------------------------

    def _build_export_query(
        self,
        start_date: datetime,
        end_date: datetime,
        tenant_id: Optional[str],
        event_types: Optional[List[str]],
    ) -> tuple:
        """Build the SQL query for exporting audit events."""
        conditions = ["timestamp >= %s", "timestamp < %s"]
        params = [start_date, end_date]

        if tenant_id:
            conditions.append("tenant_id = %s")
            params.append(tenant_id)

        if event_types:
            placeholders = ", ".join(["%s"] * len(event_types))
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(event_types)

        # Exclude events under legal hold
        conditions.append("(legal_hold IS NULL OR legal_hold = false)")

        # Exclude already archived events
        conditions.append("archival_status IS NULL")

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                event_id,
                tenant_id,
                user_id,
                event_type,
                action,
                resource_type,
                resource_id,
                result,
                severity,
                timestamp,
                client_ip,
                user_agent,
                request_id,
                session_id,
                metadata,
                integrity_hash,
                created_at
            FROM audit.audit_log
            WHERE {where_clause}
            ORDER BY timestamp ASC
        """

        return query, params

    async def _export_to_parquet(
        self,
        query: str,
        params: list,
    ) -> tuple:
        """Export query results to a Parquet file."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise RuntimeError(
                "pyarrow required for Parquet export: pip install pyarrow"
            )

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".parquet",
            delete=False,
        )
        temp_path = temp_file.name
        temp_file.close()

        rows = []
        min_ts = None
        max_ts = None

        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)

                async for row in cur:
                    rows.append(row)

                    # Track timestamp range
                    ts = row[9]  # timestamp column
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
                    if max_ts is None or ts > max_ts:
                        max_ts = ts

        if not rows:
            return temp_path, 0, datetime.now(timezone.utc), datetime.now(timezone.utc)

        # Convert to PyArrow table
        columns = [
            "event_id", "tenant_id", "user_id", "event_type", "action",
            "resource_type", "resource_id", "result", "severity", "timestamp",
            "client_ip", "user_agent", "request_id", "session_id",
            "metadata", "integrity_hash", "created_at"
        ]

        # Build column arrays
        data = {col: [] for col in columns}
        for row in rows:
            for i, col in enumerate(columns):
                val = row[i]
                # Convert datetime to string for Parquet
                if isinstance(val, datetime):
                    val = val.isoformat()
                # Convert dict to string
                elif isinstance(val, dict):
                    import json
                    val = json.dumps(val)
                data[col].append(val)

        table = pa.table(data)

        # Write Parquet file
        pq.write_table(
            table,
            temp_path,
            compression=PARQUET_COMPRESSION,
            row_group_size=PARQUET_ROW_GROUP_SIZE,
        )

        return temp_path, len(rows), min_ts, max_ts

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _generate_s3_key(
        self,
        partition_key: str,
        date: datetime,
        archive_id: str,
    ) -> str:
        """Generate S3 key for an archive file."""
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")

        return (
            f"{COLD_PREFIX}/{year}/{month}/"
            f"audit_{year}_{month}_{day}_{archive_id[:8]}.parquet"
        )

    async def _upload_to_s3(
        self,
        local_path: str,
        s3_key: str,
        metadata: Dict[str, str],
    ) -> None:
        """Upload a file to S3."""
        file_size = Path(local_path).stat().st_size

        extra_args = {
            "Metadata": metadata,
            "ContentType": "application/octet-stream",
        }

        # Use multipart upload for large files
        if file_size > MULTIPART_THRESHOLD_MB * 1024 * 1024:
            config = {"multipart_threshold": MULTIPART_THRESHOLD_MB * 1024 * 1024}
            self._s3_client.upload_file(
                local_path,
                self._bucket,
                s3_key,
                ExtraArgs=extra_args,
                Config=config,
            )
        else:
            self._s3_client.upload_file(
                local_path,
                self._bucket,
                s3_key,
                ExtraArgs=extra_args,
            )

        logger.debug(
            "Uploaded to S3: s3://%s/%s (%d bytes)",
            self._bucket,
            s3_key,
            file_size,
        )

    async def _upload_checksum(
        self,
        checksum: str,
        s3_key: str,
    ) -> None:
        """Upload a checksum file to S3."""
        body = checksum.encode("utf-8")

        self._s3_client.put_object(
            Bucket=self._bucket,
            Key=s3_key,
            Body=body,
            ContentType="text/plain",
        )

    async def _mark_events_archived(
        self,
        start_date: datetime,
        end_date: datetime,
        tenant_id: Optional[str],
        archive_id: str,
    ) -> None:
        """Mark events as archived in the database."""
        conditions = ["timestamp >= %s", "timestamp < %s"]
        params = [start_date, end_date]

        if tenant_id:
            conditions.append("tenant_id = %s")
            params.append(tenant_id)

        where_clause = " AND ".join(conditions)

        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE audit.audit_log
                    SET
                        archival_status = 'archived',
                        archival_id = %s,
                        archival_completed_at = %s
                    WHERE {where_clause}
                      AND archival_status IS NULL
                    """,
                    [archive_id, datetime.now(timezone.utc), *params],
                )
                await conn.commit()

    async def _save_archive_metadata(
        self,
        metadata: ArchiveMetadata,
    ) -> None:
        """Save archive metadata to the database."""
        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO audit.archive_metadata (
                        archive_id,
                        partition_key,
                        s3_bucket,
                        s3_key,
                        storage_class,
                        file_size_bytes,
                        row_count,
                        checksum_sha256,
                        archived_at,
                        start_timestamp,
                        end_timestamp,
                        event_types,
                        tenant_ids
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    [
                        metadata.archive_id,
                        metadata.partition_key,
                        metadata.s3_bucket,
                        metadata.s3_key,
                        metadata.storage_class,
                        metadata.file_size_bytes,
                        metadata.row_count,
                        metadata.checksum_sha256,
                        metadata.archived_at,
                        metadata.start_timestamp,
                        metadata.end_timestamp,
                        metadata.event_types,
                        metadata.tenant_ids,
                    ],
                )
                await conn.commit()

    async def _get_archive_metadata(
        self,
        partition_key: str,
    ) -> Optional[ArchiveMetadata]:
        """Get archive metadata for a partition."""
        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        archive_id, partition_key, s3_bucket, s3_key,
                        storage_class, file_size_bytes, row_count,
                        checksum_sha256, archived_at, start_timestamp,
                        end_timestamp, event_types, tenant_ids
                    FROM audit.archive_metadata
                    WHERE partition_key = %s
                    ORDER BY archived_at DESC
                    LIMIT 1
                    """,
                    [partition_key],
                )
                row = await cur.fetchone()

                if row is None:
                    return None

                return ArchiveMetadata(
                    archive_id=row[0],
                    partition_key=row[1],
                    s3_bucket=row[2],
                    s3_key=row[3],
                    storage_class=row[4],
                    file_size_bytes=row[5],
                    row_count=row[6],
                    checksum_sha256=row[7],
                    archived_at=row[8],
                    start_timestamp=row[9],
                    end_timestamp=row[10],
                    event_types=row[11] or [],
                    tenant_ids=row[12] or [],
                )

    async def _update_archive_storage_class(
        self,
        partition_key: str,
        storage_class: str,
    ) -> None:
        """Update storage class in archive metadata."""
        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE audit.archive_metadata
                    SET storage_class = %s
                    WHERE partition_key = %s
                    """,
                    [storage_class, partition_key],
                )
                await conn.commit()

    async def _initiate_glacier_restore(
        self,
        metadata: ArchiveMetadata,
        expires_in_days: int,
    ) -> str:
        """Initiate restore from Glacier."""
        restore_id = str(uuid.uuid4())

        self._s3_client.restore_object(
            Bucket=metadata.s3_bucket,
            Key=metadata.s3_key,
            RestoreRequest={
                "Days": expires_in_days,
                "GlacierJobParameters": {
                    "Tier": "Standard",  # 3-5 hours for Standard tier
                },
            },
        )

        # Record restore request
        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO audit.restore_requests (
                        restore_id,
                        archive_id,
                        requested_at,
                        expires_at,
                        status
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    [
                        restore_id,
                        metadata.archive_id,
                        datetime.now(timezone.utc),
                        datetime.now(timezone.utc) + timedelta(days=expires_in_days),
                        ArchivalStatus.GLACIER_PENDING.value,
                    ],
                )
                await conn.commit()

        logger.info(
            "Glacier restore initiated: restore_id=%s archive_id=%s",
            restore_id,
            metadata.archive_id,
        )

        return restore_id

    async def _download_from_s3(self, s3_key: str) -> str:
        """Download a file from S3."""
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".parquet",
            delete=False,
        )
        temp_path = temp_file.name
        temp_file.close()

        self._s3_client.download_file(
            self._bucket,
            s3_key,
            temp_path,
        )

        return temp_path

    async def _create_restore_table(self, table_name: str) -> None:
        """Create a temporary table for restored data."""
        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS audit.{table_name} (
                        event_id UUID PRIMARY KEY,
                        tenant_id UUID,
                        user_id UUID,
                        event_type VARCHAR(100),
                        action VARCHAR(50),
                        resource_type VARCHAR(100),
                        resource_id VARCHAR(255),
                        result VARCHAR(20),
                        severity VARCHAR(20),
                        timestamp TIMESTAMPTZ,
                        client_ip INET,
                        user_agent TEXT,
                        request_id VARCHAR(64),
                        session_id VARCHAR(64),
                        metadata JSONB,
                        integrity_hash VARCHAR(64),
                        created_at TIMESTAMPTZ
                    )
                    """
                )
                await conn.commit()

    async def _load_parquet_to_table(
        self,
        parquet_path: str,
        table_name: str,
    ) -> int:
        """Load Parquet data into PostgreSQL table."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise RuntimeError("pyarrow required")

        table = pq.read_table(parquet_path)
        rows = table.to_pydict()

        row_count = len(rows["event_id"])

        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                # Insert in batches
                batch_size = 1000
                for i in range(0, row_count, batch_size):
                    batch_values = []
                    for j in range(i, min(i + batch_size, row_count)):
                        batch_values.append((
                            rows["event_id"][j],
                            rows["tenant_id"][j],
                            rows["user_id"][j],
                            rows["event_type"][j],
                            rows["action"][j],
                            rows["resource_type"][j],
                            rows["resource_id"][j],
                            rows["result"][j],
                            rows["severity"][j],
                            rows["timestamp"][j],
                            rows["client_ip"][j],
                            rows["user_agent"][j],
                            rows["request_id"][j],
                            rows["session_id"][j],
                            rows["metadata"][j],
                            rows["integrity_hash"][j],
                            rows["created_at"][j],
                        ))

                    await cur.executemany(
                        f"""
                        INSERT INTO audit.{table_name} (
                            event_id, tenant_id, user_id, event_type, action,
                            resource_type, resource_id, result, severity,
                            timestamp, client_ip, user_agent, request_id,
                            session_id, metadata, integrity_hash, created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        batch_values,
                    )

                await conn.commit()

        return row_count

    async def _schedule_table_cleanup(
        self,
        table_name: str,
        expires_in_days: int,
    ) -> None:
        """Schedule cleanup of temporary restore table."""
        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO audit.restore_table_cleanup (
                        table_name,
                        created_at,
                        expires_at
                    ) VALUES (%s, %s, %s)
                    """,
                    [
                        table_name,
                        datetime.now(timezone.utc),
                        datetime.now(timezone.utc) + timedelta(days=expires_in_days),
                    ],
                )
                await conn.commit()


# Import timedelta for use in restore methods
from datetime import timedelta


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "ArchivalService",
    "ArchivalStatus",
    "ArchiveMetadata",
    "RestoreRequest",
    "DEFAULT_ARCHIVE_BUCKET",
    "COLD_PREFIX",
    "GLACIER_PREFIX",
]
