"""
Cold Storage for GL-FOUND-X-003 Audit Events.

This module provides cold storage implementations for archiving audit events
to S3 or GCS as Parquet files. Events are partitioned by date for efficient
querying and retention management.

Key Features:
    - Abstract base class for storage backends
    - S3 and GCS implementations
    - Daily partitioning with configurable paths
    - Parquet format for efficient columnar storage
    - Checksum verification for integrity
    - Compression with Snappy (default) or Gzip

Storage Layout:
    s3://bucket/prefix/org_id/YYYY/MM/DD/events.parquet
    gs://bucket/prefix/org_id/YYYY/MM/DD/events.parquet

Example:
    >>> from gl_normalizer_service.audit.storage import S3AuditStorage
    >>> from gl_normalizer_service.audit.models import OutboxConfig
    >>> config = OutboxConfig(
    ...     db_url="postgresql://localhost/normalizer",
    ...     s3_bucket="audit-bucket",
    ...     s3_prefix="audit/normalizer",
    ... )
    >>> storage = S3AuditStorage(config)
    >>> partition = await storage.write_partition(records, org_id, date)

NFR Compliance:
    - NFR-036: 7-year retention with date-partitioned storage
    - NFR-039: Integrity verification with checksums
"""

import asyncio
import hashlib
import io
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from gl_normalizer_service.audit.models import (
    ColdStoragePartition,
    OutboxConfig,
    OutboxRecord,
)

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """
    Exception raised when storage operations fail.

    Attributes:
        path: Storage path that failed.
        operation: Operation that failed (read, write, delete).
        message: Error message.
        original_error: Original exception that caused the failure.
    """

    def __init__(
        self,
        path: str,
        operation: str,
        message: str,
        original_error: Optional[Exception] = None,
    ):
        self.path = path
        self.operation = operation
        self.original_error = original_error
        super().__init__(f"Storage {operation} failed for {path}: {message}")


class AuditColdStorage(ABC):
    """
    Abstract base class for cold storage backends.

    Defines the interface for writing audit events to cold storage
    (S3, GCS, or other object stores) as Parquet files.

    Subclasses must implement:
        - write_bytes: Write raw bytes to storage
        - read_bytes: Read raw bytes from storage
        - delete_path: Delete a path from storage
        - list_paths: List paths matching a prefix

    Attributes:
        config: Storage configuration.

    Example:
        >>> class MyStorage(AuditColdStorage):
        ...     async def write_bytes(self, path, data): ...
        ...     async def read_bytes(self, path): ...
        ...     async def delete_path(self, path): ...
        ...     async def list_paths(self, prefix): ...
    """

    def __init__(self, config: OutboxConfig):
        """
        Initialize the cold storage.

        Args:
            config: Storage configuration.
        """
        self.config = config

    @abstractmethod
    async def write_bytes(self, path: str, data: bytes) -> int:
        """
        Write raw bytes to storage.

        Args:
            path: Full storage path.
            data: Bytes to write.

        Returns:
            Number of bytes written.

        Raises:
            StorageError: If write fails.
        """
        pass

    @abstractmethod
    async def read_bytes(self, path: str) -> bytes:
        """
        Read raw bytes from storage.

        Args:
            path: Full storage path.

        Returns:
            Bytes read from storage.

        Raises:
            StorageError: If read fails.
        """
        pass

    @abstractmethod
    async def delete_path(self, path: str) -> bool:
        """
        Delete a path from storage.

        Args:
            path: Full storage path.

        Returns:
            True if deleted, False if not found.

        Raises:
            StorageError: If delete fails.
        """
        pass

    @abstractmethod
    async def list_paths(
        self,
        prefix: str,
        max_results: int = 1000,
    ) -> List[str]:
        """
        List paths matching a prefix.

        Args:
            prefix: Path prefix to match.
            max_results: Maximum number of results.

        Returns:
            List of matching paths.

        Raises:
            StorageError: If listing fails.
        """
        pass

    @abstractmethod
    def get_bucket(self) -> str:
        """Get the storage bucket name."""
        pass

    @abstractmethod
    def get_prefix(self) -> str:
        """Get the storage prefix."""
        pass

    def build_partition_path(
        self,
        org_id: str,
        partition_date: str,
    ) -> str:
        """
        Build the full storage path for a partition.

        Args:
            org_id: Organization ID.
            partition_date: Date string (YYYY-MM-DD).

        Returns:
            Full storage path.

        Example:
            >>> path = storage.build_partition_path("org-acme", "2026-01-30")
            >>> assert path == "s3://bucket/prefix/org-acme/2026/01/30/events.parquet"
        """
        year, month, day = partition_date.split("-")
        prefix = self.get_prefix().rstrip("/")
        bucket = self.get_bucket()

        return f"{bucket}/{prefix}/{org_id}/{year}/{month}/{day}/events.parquet"

    async def write_partition(
        self,
        records: List[OutboxRecord],
        org_id: str,
        partition_date: str,
        compression: str = "snappy",
    ) -> ColdStoragePartition:
        """
        Write audit records as a Parquet partition.

        Converts records to a Parquet file and writes to cold storage.
        Returns metadata about the created partition.

        Args:
            records: List of OutboxRecord instances to archive.
            org_id: Organization ID for partitioning.
            partition_date: Date for the partition (YYYY-MM-DD).
            compression: Compression codec (snappy, gzip, none).

        Returns:
            ColdStoragePartition metadata.

        Raises:
            StorageError: If write fails.
            ValueError: If records is empty.

        Example:
            >>> partition = await storage.write_partition(
            ...     records=records,
            ...     org_id="org-acme",
            ...     partition_date="2026-01-30",
            ... )
            >>> print(f"Wrote {partition.record_count} records to {partition.path}")
        """
        if not records:
            raise ValueError("Cannot write empty partition")

        path = self.build_partition_path(org_id, partition_date)
        logger.info(
            "Writing partition with %d records to %s",
            len(records),
            path,
        )

        try:
            # Convert records to Parquet bytes
            parquet_bytes = self._records_to_parquet(records, compression)

            # Compute checksum
            checksum = f"sha256:{hashlib.sha256(parquet_bytes).hexdigest()}"

            # Write to storage
            bytes_written = await self.write_bytes(path, parquet_bytes)

            # Create partition metadata
            partition = ColdStoragePartition(
                date=partition_date,
                path=path,
                record_count=len(records),
                checksum=checksum,
                size_bytes=bytes_written,
                org_id=org_id,
                compression=compression,
            )

            logger.info(
                "Wrote partition %s: %d records, %d bytes, checksum=%s",
                path,
                len(records),
                bytes_written,
                checksum[:30] + "...",
            )

            return partition

        except Exception as e:
            logger.error("Failed to write partition %s: %s", path, str(e))
            raise StorageError(
                path=path,
                operation="write",
                message=str(e),
                original_error=e,
            )

    async def read_partition(
        self,
        partition: ColdStoragePartition,
        verify_checksum: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Read audit records from a Parquet partition.

        Args:
            partition: Partition metadata.
            verify_checksum: Whether to verify checksum.

        Returns:
            List of audit event dictionaries.

        Raises:
            StorageError: If read fails or checksum mismatch.

        Example:
            >>> records = await storage.read_partition(partition)
            >>> for record in records:
            ...     print(record["event_id"])
        """
        logger.info("Reading partition from %s", partition.path)

        try:
            # Read bytes from storage
            parquet_bytes = await self.read_bytes(partition.path)

            # Verify checksum if requested
            if verify_checksum:
                computed_checksum = f"sha256:{hashlib.sha256(parquet_bytes).hexdigest()}"
                if computed_checksum != partition.checksum:
                    raise StorageError(
                        path=partition.path,
                        operation="read",
                        message=(
                            f"Checksum mismatch: expected {partition.checksum}, "
                            f"got {computed_checksum}"
                        ),
                    )

            # Convert Parquet to records
            records = self._parquet_to_records(parquet_bytes)

            logger.info(
                "Read %d records from partition %s",
                len(records),
                partition.path,
            )

            return records

        except StorageError:
            raise
        except Exception as e:
            logger.error("Failed to read partition %s: %s", partition.path, str(e))
            raise StorageError(
                path=partition.path,
                operation="read",
                message=str(e),
                original_error=e,
            )

    async def delete_partition(self, partition: ColdStoragePartition) -> bool:
        """
        Delete a partition from cold storage.

        Args:
            partition: Partition to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            StorageError: If delete fails.
        """
        logger.info("Deleting partition %s", partition.path)
        return await self.delete_path(partition.path)

    async def list_partitions(
        self,
        org_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[str]:
        """
        List partition paths for an organization.

        Args:
            org_id: Organization ID.
            start_date: Optional start date filter (YYYY-MM-DD).
            end_date: Optional end date filter (YYYY-MM-DD).

        Returns:
            List of partition paths.
        """
        prefix = f"{self.get_prefix().rstrip('/')}/{org_id}/"
        paths = await self.list_paths(prefix)

        if start_date or end_date:
            filtered = []
            for path in paths:
                # Extract date from path
                try:
                    parts = path.split("/")
                    # Find year/month/day in path
                    for i, part in enumerate(parts):
                        if len(part) == 4 and part.isdigit():  # Year
                            if i + 2 < len(parts):
                                year = part
                                month = parts[i + 1]
                                day = parts[i + 2]
                                path_date = f"{year}-{month}-{day}"
                                if start_date and path_date < start_date:
                                    continue
                                if end_date and path_date > end_date:
                                    continue
                                filtered.append(path)
                                break
                except (IndexError, ValueError):
                    continue
            return filtered

        return paths

    def partition_by_date(
        self,
        events: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Partition events by date.

        Groups events by their event_ts date for daily partitioning.

        Args:
            events: List of audit events.

        Returns:
            Dictionary mapping date strings to event lists.

        Example:
            >>> partitions = storage.partition_by_date(events)
            >>> for date_str, date_events in partitions.items():
            ...     print(f"{date_str}: {len(date_events)} events")
        """
        partitions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for event in events:
            # Extract date from event_ts
            event_ts = event.get("event_ts")
            if isinstance(event_ts, str):
                # Parse ISO format
                date_str = event_ts[:10]  # YYYY-MM-DD
            elif isinstance(event_ts, datetime):
                date_str = event_ts.strftime("%Y-%m-%d")
            elif isinstance(event_ts, date):
                date_str = event_ts.strftime("%Y-%m-%d")
            else:
                # Default to today
                date_str = datetime.utcnow().strftime("%Y-%m-%d")

            partitions[date_str].append(event)

        logger.debug(
            "Partitioned %d events into %d daily partitions",
            len(events),
            len(partitions),
        )

        return dict(partitions)

    def _records_to_parquet(
        self,
        records: List[OutboxRecord],
        compression: str = "snappy",
    ) -> bytes:
        """
        Convert outbox records to Parquet bytes.

        Args:
            records: List of OutboxRecord instances.
            compression: Compression codec.

        Returns:
            Parquet file as bytes.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Convert records to dictionaries
            data = []
            for record in records:
                row = {
                    "id": record.id,
                    "event_id": record.event_id,
                    "event_type": record.event_type,
                    "org_id": record.org_id,
                    "payload": json.dumps(record.payload),
                    "status": str(record.status),
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat(),
                    "published_at": (
                        record.published_at.isoformat()
                        if record.published_at
                        else None
                    ),
                    "retries": record.retries,
                }
                data.append(row)

            # Create PyArrow table
            table = pa.Table.from_pylist(data)

            # Write to bytes buffer
            buffer = io.BytesIO()
            pq.write_table(
                table,
                buffer,
                compression=compression,
            )
            buffer.seek(0)
            return buffer.read()

        except ImportError:
            # Fallback to JSON if PyArrow not available
            logger.warning(
                "PyArrow not installed, falling back to JSON format"
            )
            data = [
                {
                    "id": r.id,
                    "event_id": r.event_id,
                    "event_type": r.event_type,
                    "org_id": r.org_id,
                    "payload": r.payload,
                    "status": str(r.status),
                    "created_at": r.created_at.isoformat(),
                    "updated_at": r.updated_at.isoformat(),
                    "published_at": (
                        r.published_at.isoformat() if r.published_at else None
                    ),
                    "retries": r.retries,
                }
                for r in records
            ]
            return json.dumps(data).encode("utf-8")

    def _parquet_to_records(self, parquet_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Convert Parquet bytes to record dictionaries.

        Args:
            parquet_bytes: Parquet file bytes.

        Returns:
            List of record dictionaries.
        """
        try:
            import pyarrow.parquet as pq

            buffer = io.BytesIO(parquet_bytes)
            table = pq.read_table(buffer)
            records = table.to_pylist()

            # Parse JSON payload back to dict
            for record in records:
                if isinstance(record.get("payload"), str):
                    record["payload"] = json.loads(record["payload"])

            return records

        except ImportError:
            # Fallback to JSON
            return json.loads(parquet_bytes.decode("utf-8"))


class S3AuditStorage(AuditColdStorage):
    """
    S3 implementation of cold storage for audit events.

    Stores Parquet files in S3 with date-based partitioning.

    Attributes:
        config: Storage configuration.
        _client: Boto3 S3 client.

    Example:
        >>> config = OutboxConfig(
        ...     db_url="postgresql://localhost/normalizer",
        ...     s3_bucket="audit-bucket",
        ...     s3_prefix="audit/normalizer",
        ...     s3_region="us-east-1",
        ... )
        >>> storage = S3AuditStorage(config)
        >>> partition = await storage.write_partition(records, org_id, date)
    """

    def __init__(self, config: OutboxConfig):
        """
        Initialize S3 storage.

        Args:
            config: Storage configuration with S3 settings.

        Raises:
            ValueError: If S3 bucket is not configured.
        """
        super().__init__(config)

        if not config.s3_bucket:
            raise ValueError("S3 bucket is required for S3AuditStorage")

        self._client: Optional[Any] = None
        self._lock = asyncio.Lock()

        logger.info(
            "S3AuditStorage initialized (bucket=%s, prefix=%s, region=%s)",
            config.s3_bucket,
            config.s3_prefix,
            config.s3_region,
        )

    async def _get_client(self) -> Any:
        """Get or create S3 client."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    try:
                        import aioboto3
                        session = aioboto3.Session()
                        self._client = await session.client(
                            "s3",
                            region_name=self.config.s3_region,
                        ).__aenter__()
                    except ImportError:
                        logger.warning(
                            "aioboto3 not installed, using mock S3 client"
                        )
                        self._client = MockS3Client()
        return self._client

    def get_bucket(self) -> str:
        """Get S3 bucket name."""
        return f"s3://{self.config.s3_bucket}"

    def get_prefix(self) -> str:
        """Get S3 key prefix."""
        return self.config.s3_prefix

    def _path_to_key(self, path: str) -> str:
        """Convert full path to S3 key."""
        # Remove s3://bucket/ prefix
        prefix = f"s3://{self.config.s3_bucket}/"
        if path.startswith(prefix):
            return path[len(prefix):]
        return path

    async def write_bytes(self, path: str, data: bytes) -> int:
        """Write bytes to S3."""
        client = await self._get_client()
        key = self._path_to_key(path)

        try:
            await client.put_object(
                Bucket=self.config.s3_bucket,
                Key=key,
                Body=data,
            )
            logger.debug("Wrote %d bytes to s3://%s/%s", len(data), self.config.s3_bucket, key)
            return len(data)

        except Exception as e:
            raise StorageError(
                path=path,
                operation="write",
                message=str(e),
                original_error=e,
            )

    async def read_bytes(self, path: str) -> bytes:
        """Read bytes from S3."""
        client = await self._get_client()
        key = self._path_to_key(path)

        try:
            response = await client.get_object(
                Bucket=self.config.s3_bucket,
                Key=key,
            )
            body = response["Body"]
            if hasattr(body, "read"):
                if asyncio.iscoroutinefunction(body.read):
                    data = await body.read()
                else:
                    data = body.read()
            else:
                data = body
            logger.debug("Read %d bytes from s3://%s/%s", len(data), self.config.s3_bucket, key)
            return data

        except Exception as e:
            raise StorageError(
                path=path,
                operation="read",
                message=str(e),
                original_error=e,
            )

    async def delete_path(self, path: str) -> bool:
        """Delete object from S3."""
        client = await self._get_client()
        key = self._path_to_key(path)

        try:
            await client.delete_object(
                Bucket=self.config.s3_bucket,
                Key=key,
            )
            logger.debug("Deleted s3://%s/%s", self.config.s3_bucket, key)
            return True

        except Exception as e:
            if "NoSuchKey" in str(e):
                return False
            raise StorageError(
                path=path,
                operation="delete",
                message=str(e),
                original_error=e,
            )

    async def list_paths(
        self,
        prefix: str,
        max_results: int = 1000,
    ) -> List[str]:
        """List objects in S3 with prefix."""
        client = await self._get_client()

        # Normalize prefix
        if prefix.startswith(f"s3://{self.config.s3_bucket}/"):
            prefix = prefix[len(f"s3://{self.config.s3_bucket}/"):]

        try:
            paths = []
            paginator = client.get_paginator("list_objects_v2")

            async for page in paginator.paginate(
                Bucket=self.config.s3_bucket,
                Prefix=prefix,
                MaxKeys=max_results,
            ):
                for obj in page.get("Contents", []):
                    paths.append(f"s3://{self.config.s3_bucket}/{obj['Key']}")
                    if len(paths) >= max_results:
                        return paths

            return paths

        except Exception as e:
            raise StorageError(
                path=prefix,
                operation="list",
                message=str(e),
                original_error=e,
            )


class GCSAuditStorage(AuditColdStorage):
    """
    GCS implementation of cold storage for audit events.

    Stores Parquet files in Google Cloud Storage with date-based partitioning.

    Attributes:
        config: Storage configuration.
        _client: GCS client.

    Example:
        >>> config = OutboxConfig(
        ...     db_url="postgresql://localhost/normalizer",
        ...     gcs_bucket="audit-bucket",
        ...     gcs_prefix="audit/normalizer",
        ... )
        >>> storage = GCSAuditStorage(config)
        >>> partition = await storage.write_partition(records, org_id, date)
    """

    def __init__(self, config: OutboxConfig):
        """
        Initialize GCS storage.

        Args:
            config: Storage configuration with GCS settings.

        Raises:
            ValueError: If GCS bucket is not configured.
        """
        super().__init__(config)

        if not config.gcs_bucket:
            raise ValueError("GCS bucket is required for GCSAuditStorage")

        self._client: Optional[Any] = None
        self._bucket: Optional[Any] = None
        self._lock = asyncio.Lock()

        logger.info(
            "GCSAuditStorage initialized (bucket=%s, prefix=%s)",
            config.gcs_bucket,
            config.gcs_prefix,
        )

    async def _get_bucket(self) -> Any:
        """Get or create GCS bucket reference."""
        if self._bucket is None:
            async with self._lock:
                if self._bucket is None:
                    try:
                        from google.cloud import storage
                        self._client = storage.Client()
                        self._bucket = self._client.bucket(self.config.gcs_bucket)
                    except ImportError:
                        logger.warning(
                            "google-cloud-storage not installed, using mock GCS client"
                        )
                        self._bucket = MockGCSBucket(self.config.gcs_bucket)
        return self._bucket

    def get_bucket(self) -> str:
        """Get GCS bucket name."""
        return f"gs://{self.config.gcs_bucket}"

    def get_prefix(self) -> str:
        """Get GCS key prefix."""
        return self.config.gcs_prefix

    def _path_to_key(self, path: str) -> str:
        """Convert full path to GCS blob name."""
        prefix = f"gs://{self.config.gcs_bucket}/"
        if path.startswith(prefix):
            return path[len(prefix):]
        return path

    async def write_bytes(self, path: str, data: bytes) -> int:
        """Write bytes to GCS."""
        bucket = await self._get_bucket()
        key = self._path_to_key(path)

        try:
            blob = bucket.blob(key)
            # Run sync operation in thread pool
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: blob.upload_from_string(data),
            )
            logger.debug("Wrote %d bytes to gs://%s/%s", len(data), self.config.gcs_bucket, key)
            return len(data)

        except Exception as e:
            raise StorageError(
                path=path,
                operation="write",
                message=str(e),
                original_error=e,
            )

    async def read_bytes(self, path: str) -> bytes:
        """Read bytes from GCS."""
        bucket = await self._get_bucket()
        key = self._path_to_key(path)

        try:
            blob = bucket.blob(key)
            # Run sync operation in thread pool
            data = await asyncio.get_event_loop().run_in_executor(
                None,
                blob.download_as_bytes,
            )
            logger.debug("Read %d bytes from gs://%s/%s", len(data), self.config.gcs_bucket, key)
            return data

        except Exception as e:
            raise StorageError(
                path=path,
                operation="read",
                message=str(e),
                original_error=e,
            )

    async def delete_path(self, path: str) -> bool:
        """Delete blob from GCS."""
        bucket = await self._get_bucket()
        key = self._path_to_key(path)

        try:
            blob = bucket.blob(key)
            await asyncio.get_event_loop().run_in_executor(
                None,
                blob.delete,
            )
            logger.debug("Deleted gs://%s/%s", self.config.gcs_bucket, key)
            return True

        except Exception as e:
            if "NotFound" in str(e):
                return False
            raise StorageError(
                path=path,
                operation="delete",
                message=str(e),
                original_error=e,
            )

    async def list_paths(
        self,
        prefix: str,
        max_results: int = 1000,
    ) -> List[str]:
        """List blobs in GCS with prefix."""
        bucket = await self._get_bucket()

        # Normalize prefix
        if prefix.startswith(f"gs://{self.config.gcs_bucket}/"):
            prefix = prefix[len(f"gs://{self.config.gcs_bucket}/"):]

        try:
            blobs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(bucket.list_blobs(prefix=prefix, max_results=max_results)),
            )
            return [f"gs://{self.config.gcs_bucket}/{blob.name}" for blob in blobs]

        except Exception as e:
            raise StorageError(
                path=prefix,
                operation="list",
                message=str(e),
                original_error=e,
            )


class MockS3Client:
    """Mock S3 client for testing."""

    def __init__(self):
        self._objects: Dict[str, bytes] = {}

    async def put_object(self, Bucket: str, Key: str, Body: bytes) -> Dict:
        self._objects[f"{Bucket}/{Key}"] = Body
        return {"ETag": "mock-etag"}

    async def get_object(self, Bucket: str, Key: str) -> Dict:
        full_key = f"{Bucket}/{Key}"
        if full_key not in self._objects:
            raise Exception("NoSuchKey")
        return {"Body": self._objects[full_key]}

    async def delete_object(self, Bucket: str, Key: str) -> Dict:
        full_key = f"{Bucket}/{Key}"
        if full_key in self._objects:
            del self._objects[full_key]
        return {}

    def get_paginator(self, operation: str):
        return MockS3Paginator(self)


class MockS3Paginator:
    """Mock S3 paginator for testing."""

    def __init__(self, client: MockS3Client):
        self._client = client

    async def paginate(self, Bucket: str, Prefix: str, MaxKeys: int = 1000):
        contents = []
        for key in self._client._objects:
            if key.startswith(f"{Bucket}/{Prefix}"):
                obj_key = key[len(f"{Bucket}/"):]
                contents.append({"Key": obj_key})
        yield {"Contents": contents[:MaxKeys]}


class MockGCSBucket:
    """Mock GCS bucket for testing."""

    def __init__(self, name: str):
        self.name = name
        self._blobs: Dict[str, bytes] = {}

    def blob(self, name: str) -> "MockGCSBlob":
        return MockGCSBlob(self, name)

    def list_blobs(self, prefix: str = "", max_results: int = 1000) -> List["MockGCSBlob"]:
        blobs = []
        for name in self._blobs:
            if name.startswith(prefix):
                blobs.append(MockGCSBlob(self, name))
        return blobs[:max_results]


class MockGCSBlob:
    """Mock GCS blob for testing."""

    def __init__(self, bucket: MockGCSBucket, name: str):
        self._bucket = bucket
        self.name = name

    def upload_from_string(self, data: bytes) -> None:
        self._bucket._blobs[self.name] = data

    def download_as_bytes(self) -> bytes:
        if self.name not in self._bucket._blobs:
            raise Exception("NotFound")
        return self._bucket._blobs[self.name]

    def delete(self) -> None:
        if self.name in self._bucket._blobs:
            del self._bucket._blobs[self.name]
