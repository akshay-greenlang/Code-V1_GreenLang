"""
Audit persistence models for GL-FOUND-X-003 Unit & Reference Normalizer.

This module defines the Pydantic models for audit persistence, including
outbox records, cold storage partitions, and configuration models.

Key Models:
    - OutboxRecord: Record in the transactional outbox table
    - ColdStoragePartition: Metadata for Parquet partitions in cold storage
    - OutboxConfig: Configuration for outbox and publisher
    - RetentionPolicy: Retention policy configuration

Example:
    >>> from gl_normalizer_service.audit.models import OutboxRecord, OutboxStatus
    >>> record = OutboxRecord(
    ...     id="outbox-123",
    ...     event_id="norm-evt-abc",
    ...     event_type="normalization",
    ...     org_id="org-acme",
    ...     payload={"event_id": "norm-evt-abc", ...},
    ...     status=OutboxStatus.PENDING,
    ... )
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class OutboxStatus(str, Enum):
    """
    Status of an outbox record.

    Attributes:
        PENDING: Record is waiting to be published.
        PROCESSING: Record is currently being published.
        PUBLISHED: Record was successfully published to Kafka.
        FAILED: Record failed to publish after max retries.
        ARCHIVED: Record has been archived to cold storage.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    PUBLISHED = "published"
    FAILED = "failed"
    ARCHIVED = "archived"


class OutboxRecord(BaseModel):
    """
    Record in the transactional outbox table.

    This model represents a single audit event that needs to be published
    to Kafka. The outbox pattern ensures at-least-once delivery by
    persisting events to a local database before publishing.

    Attributes:
        id: Unique identifier for this outbox record.
        event_id: ID of the audit event (from NormalizationEvent).
        event_type: Type of event (e.g., "normalization", "batch").
        org_id: Organization ID for Kafka partitioning.
        payload: Complete audit event as dictionary.
        status: Current status of the record.
        created_at: Timestamp when the record was created.
        updated_at: Timestamp when the record was last updated.
        published_at: Timestamp when the record was published (if any).
        retries: Number of publish attempts.
        last_error: Last error message (if any).
        kafka_offset: Kafka offset after successful publish.
        kafka_partition: Kafka partition after successful publish.

    Example:
        >>> record = OutboxRecord(
        ...     id="outbox-001",
        ...     event_id="norm-evt-abc123",
        ...     event_type="normalization",
        ...     org_id="org-acme",
        ...     payload={"event_id": "norm-evt-abc123", "status": "success"},
        ...     status=OutboxStatus.PENDING,
        ... )
    """

    id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for this outbox record",
    )
    event_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="ID of the audit event",
    )
    event_type: str = Field(
        default="normalization",
        min_length=1,
        max_length=50,
        description="Type of event",
    )
    org_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Organization ID for Kafka partitioning",
    )
    payload: Dict[str, Any] = Field(
        ...,
        description="Complete audit event as dictionary",
    )
    status: OutboxStatus = Field(
        default=OutboxStatus.PENDING,
        description="Current status of the record",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the record was last updated",
    )
    published_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the record was published",
    )
    retries: int = Field(
        default=0,
        ge=0,
        description="Number of publish attempts",
    )
    last_error: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Last error message",
    )
    kafka_offset: Optional[int] = Field(
        default=None,
        ge=0,
        description="Kafka offset after successful publish",
    )
    kafka_partition: Optional[int] = Field(
        default=None,
        ge=0,
        description="Kafka partition after successful publish",
    )

    model_config = {"use_enum_values": True}


class ColdStoragePartition(BaseModel):
    """
    Metadata for a Parquet partition in cold storage.

    Each partition contains audit events for a single day, stored as
    a Parquet file in S3 or GCS. The checksum enables integrity verification.

    Attributes:
        date: Date of the partition (YYYY-MM-DD).
        path: Full path to the Parquet file in storage.
        record_count: Number of audit records in the partition.
        checksum: SHA-256 checksum of the Parquet file.
        size_bytes: Size of the Parquet file in bytes.
        org_id: Organization ID (for multi-tenant partitioning).
        created_at: Timestamp when the partition was created.
        schema_version: Version of the audit schema used.
        compression: Compression codec used (e.g., "snappy", "gzip").

    Example:
        >>> partition = ColdStoragePartition(
        ...     date="2026-01-30",
        ...     path="s3://audit-bucket/org-acme/2026/01/30/events.parquet",
        ...     record_count=1000,
        ...     checksum="sha256:abc123...",
        ...     size_bytes=1024000,
        ...     org_id="org-acme",
        ... )
    """

    date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date of the partition (YYYY-MM-DD)",
    )
    path: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Full path to the Parquet file in storage",
    )
    record_count: int = Field(
        ...,
        ge=0,
        description="Number of audit records in the partition",
    )
    checksum: str = Field(
        ...,
        pattern=r"^sha256:[a-f0-9]{64}$",
        description="SHA-256 checksum of the Parquet file",
    )
    size_bytes: int = Field(
        ...,
        ge=0,
        description="Size of the Parquet file in bytes",
    )
    org_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Organization ID",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the partition was created",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Version of the audit schema used",
    )
    compression: str = Field(
        default="snappy",
        description="Compression codec used",
    )


class RetentionPolicy(BaseModel):
    """
    Retention policy configuration for audit data.

    Defines how long audit data should be retained in different tiers
    before being purged. Default is 7 years for regulatory compliance.

    Attributes:
        hot_retention_days: Days to keep in hot storage (Kafka/DB).
        warm_retention_days: Days to keep in warm storage (recent Parquet).
        cold_retention_years: Years to keep in cold storage (archived).
        total_retention_years: Total retention period in years.
        purge_enabled: Whether automatic purging is enabled.
        purge_batch_size: Number of records to purge per batch.

    Example:
        >>> policy = RetentionPolicy(
        ...     hot_retention_days=30,
        ...     warm_retention_days=365,
        ...     cold_retention_years=6,
        ...     total_retention_years=7,
        ... )
    """

    hot_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days to keep in hot storage",
    )
    warm_retention_days: int = Field(
        default=365,
        ge=1,
        le=3650,
        description="Days to keep in warm storage",
    )
    cold_retention_years: int = Field(
        default=6,
        ge=1,
        le=100,
        description="Years to keep in cold storage",
    )
    total_retention_years: int = Field(
        default=7,
        ge=1,
        le=100,
        description="Total retention period in years",
    )
    purge_enabled: bool = Field(
        default=True,
        description="Whether automatic purging is enabled",
    )
    purge_batch_size: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Number of records to purge per batch",
    )

    @field_validator("total_retention_years")
    @classmethod
    def validate_total_retention(cls, v: int, info) -> int:
        """Validate total retention is at least sum of tiers."""
        data = info.data
        hot_days = data.get("hot_retention_days", 30)
        warm_days = data.get("warm_retention_days", 365)
        cold_years = data.get("cold_retention_years", 6)

        min_years = (hot_days + warm_days) / 365 + cold_years
        if v < min_years:
            raise ValueError(
                f"total_retention_years ({v}) must be at least "
                f"{min_years:.1f} years to cover all tiers"
            )
        return v

    def get_purge_cutoff(self) -> datetime:
        """
        Calculate the cutoff date for data purging.

        Returns:
            Datetime before which data should be purged.
        """
        return datetime.utcnow() - timedelta(days=self.total_retention_years * 365)


class OutboxConfig(BaseModel):
    """
    Configuration for the audit outbox and publisher.

    Attributes:
        db_url: PostgreSQL connection URL for outbox table.
        kafka_bootstrap_servers: Comma-separated Kafka bootstrap servers.
        kafka_topic: Kafka topic for audit events.
        kafka_acks: Kafka acknowledgment level (all, 1, 0).
        kafka_retries: Number of Kafka publish retries.
        kafka_compression: Kafka compression type.
        s3_bucket: S3 bucket for cold storage.
        s3_prefix: S3 key prefix for audit data.
        gcs_bucket: GCS bucket for cold storage (alternative to S3).
        gcs_prefix: GCS key prefix for audit data.
        outbox_poll_interval_seconds: Interval for outbox polling.
        outbox_batch_size: Number of records to process per batch.
        outbox_max_retries: Maximum retries before marking as failed.
        retention_policy: Retention policy configuration.

    Example:
        >>> config = OutboxConfig(
        ...     db_url="postgresql://localhost/normalizer",
        ...     kafka_bootstrap_servers="localhost:9092",
        ... )
    """

    # Database configuration
    db_url: str = Field(
        ...,
        min_length=1,
        description="PostgreSQL connection URL for outbox table",
    )

    # Kafka configuration
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        min_length=1,
        description="Comma-separated Kafka bootstrap servers",
    )
    kafka_topic: str = Field(
        default="gl.normalizer.audit.events",
        min_length=1,
        max_length=255,
        description="Kafka topic for audit events",
    )
    kafka_acks: str = Field(
        default="all",
        description="Kafka acknowledgment level",
    )
    kafka_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of Kafka publish retries",
    )
    kafka_compression: str = Field(
        default="gzip",
        description="Kafka compression type",
    )
    kafka_max_request_size: int = Field(
        default=10485760,  # 10 MB
        ge=1024,
        description="Maximum Kafka request size in bytes",
    )

    # S3 configuration
    s3_bucket: Optional[str] = Field(
        default=None,
        description="S3 bucket for cold storage",
    )
    s3_prefix: str = Field(
        default="audit/normalizer",
        description="S3 key prefix for audit data",
    )
    s3_region: str = Field(
        default="us-east-1",
        description="AWS region for S3",
    )

    # GCS configuration
    gcs_bucket: Optional[str] = Field(
        default=None,
        description="GCS bucket for cold storage",
    )
    gcs_prefix: str = Field(
        default="audit/normalizer",
        description="GCS key prefix for audit data",
    )

    # Outbox configuration
    outbox_poll_interval_seconds: int = Field(
        default=5,
        ge=1,
        le=300,
        description="Interval for outbox polling",
    )
    outbox_batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of records to process per batch",
    )
    outbox_max_retries: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum retries before marking as failed",
    )
    outbox_lock_timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Timeout for outbox record processing lock",
    )

    # Retention configuration
    retention_policy: RetentionPolicy = Field(
        default_factory=RetentionPolicy,
        description="Retention policy configuration",
    )

    @field_validator("kafka_acks")
    @classmethod
    def validate_kafka_acks(cls, v: str) -> str:
        """Validate Kafka acks is a valid value."""
        allowed = {"all", "1", "0"}
        if v not in allowed:
            raise ValueError(f"kafka_acks must be one of {allowed}")
        return v

    @field_validator("kafka_compression")
    @classmethod
    def validate_kafka_compression(cls, v: str) -> str:
        """Validate Kafka compression is a valid value."""
        allowed = {"none", "gzip", "snappy", "lz4", "zstd"}
        if v not in allowed:
            raise ValueError(f"kafka_compression must be one of {allowed}")
        return v


class ArchivalBatch(BaseModel):
    """
    Batch of records to be archived to cold storage.

    Attributes:
        org_id: Organization ID for the batch.
        date: Date of the batch (YYYY-MM-DD).
        records: List of outbox records to archive.
        start_time: Timestamp when archival started.

    Example:
        >>> batch = ArchivalBatch(
        ...     org_id="org-acme",
        ...     date="2026-01-30",
        ...     records=[...],
        ... )
    """

    org_id: str = Field(
        ...,
        description="Organization ID for the batch",
    )
    date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date of the batch",
    )
    records: List[OutboxRecord] = Field(
        default_factory=list,
        description="List of outbox records to archive",
    )
    start_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when archival started",
    )
