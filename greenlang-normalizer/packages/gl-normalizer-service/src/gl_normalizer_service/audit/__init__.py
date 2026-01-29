"""
Audit Persistence Module for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides durable audit persistence with the Outbox Pattern for
guaranteed delivery to Kafka, cold storage archival to S3/GCS, and hash
chaining for tamper-evident audit trails.

Key Components:
    - AuditOutbox: Durable outbox pattern for at-least-once delivery
    - AuditKafkaPublisher: Kafka publisher with org_id partitioning
    - AuditColdStorage: Abstract cold storage with S3/GCS implementations
    - Hash chaining utilities for tamper evidence
    - Retention policy enforcement for compliance

Architecture:
    1. Events are written to the outbox table (PostgreSQL) atomically
    2. Background worker processes outbox entries and publishes to Kafka
    3. Periodic archival moves events to cold storage (Parquet on S3/GCS)
    4. Retention policies purge data older than 7 years (configurable)

Example:
    >>> from gl_normalizer_service.audit import (
    ...     AuditOutbox,
    ...     AuditKafkaPublisher,
    ...     S3AuditStorage,
    ...     OutboxConfig,
    ... )
    >>> config = OutboxConfig(
    ...     db_url="postgresql://localhost/normalizer",
    ...     kafka_bootstrap_servers="localhost:9092",
    ... )
    >>> outbox = AuditOutbox(config)
    >>> publisher = AuditKafkaPublisher(config)
    >>> await outbox.write_to_outbox(event)
    >>> await outbox.process_outbox(publisher)

NFR Compliance:
    - NFR-035: Tamper-evident audit hashing with hash chaining
    - NFR-036: 7-year retention for regulatory compliance
    - NFR-037: At-least-once delivery guarantee via Outbox Pattern
"""

from gl_normalizer_service.audit.models import (
    ColdStoragePartition,
    OutboxConfig,
    OutboxRecord,
    OutboxStatus,
    RetentionPolicy,
)
from gl_normalizer_service.audit.outbox import (
    AuditOutbox,
    OutboxProcessingError,
)
from gl_normalizer_service.audit.publisher import (
    AuditKafkaPublisher,
    KafkaPublishError,
)
from gl_normalizer_service.audit.storage import (
    AuditColdStorage,
    GCSAuditStorage,
    S3AuditStorage,
    StorageError,
)
from gl_normalizer_service.audit.chain import (
    ChainIntegrityError,
    compute_chain_hash,
    verify_chain,
)
from gl_normalizer_service.audit.retention import (
    RetentionEnforcer,
    apply_retention,
)

__all__ = [
    # Models
    "OutboxRecord",
    "OutboxStatus",
    "OutboxConfig",
    "ColdStoragePartition",
    "RetentionPolicy",
    # Outbox
    "AuditOutbox",
    "OutboxProcessingError",
    # Publisher
    "AuditKafkaPublisher",
    "KafkaPublishError",
    # Storage
    "AuditColdStorage",
    "S3AuditStorage",
    "GCSAuditStorage",
    "StorageError",
    # Chain
    "compute_chain_hash",
    "verify_chain",
    "ChainIntegrityError",
    # Retention
    "RetentionEnforcer",
    "apply_retention",
]

__version__ = "1.0.0"
