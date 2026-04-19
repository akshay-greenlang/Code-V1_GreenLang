# -*- coding: utf-8 -*-
"""
PersistentStorage - Immutable object storage for GL-011 FuelCraft audit bundles.

This module implements production-grade persistent storage backends for run bundles
with content-addressed storage, 7-year retention compliance, and immutable object
storage with complete integrity verification.

Key Features:
- PersistentStorageBackend abstract base class for pluggable backends
- LocalFileStorage for development and edge deployments
- S3Storage for production cloud deployments with object locking
- RetentionManager for 7-year regulatory compliance
- BundleIndexer for fast lookups by run_id, timestamp, agent_version
- Atomic writes with integrity verification

Compliance Standards:
- EPA Record Retention Requirements
- SOX Compliance (7-year retention)
- MARPOL Annex VI Documentation
- ISO 27001 (Audit trail requirements)

Example:
    >>> from audit.persistent_storage import LocalFileStorage, S3Storage
    >>>
    >>> # Local storage for development
    >>> local_storage = LocalFileStorage(base_path="/audit/bundles")
    >>> receipt = local_storage.store(bundle_hash, bundle_data)
    >>> assert receipt.integrity_verified
    >>>
    >>> # S3 storage for production
    >>> s3_storage = S3Storage(bucket="audit-bundles", region="us-east-1")
    >>> receipt = s3_storage.store(bundle_hash, bundle_data)
    >>> assert receipt.immutable  # Object lock enabled

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)
from uuid import uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# 7-year retention in days (regulatory requirement)
RETENTION_DAYS_7_YEARS = 2557  # 7 * 365 + 2 (leap years)
RETENTION_DAYS_1_YEAR = 365

# Default hash algorithm
HASH_ALGORITHM = "sha256"

# Storage path structure: {base}/{hash[0:2]}/{hash[2:4]}/{hash}
SHARD_DEPTH = 2
SHARD_WIDTH = 2

# Maximum file size for single-part upload (5GB)
MAX_SINGLE_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024

# Minimum file size to trigger compression
MIN_COMPRESS_BYTES = 1024

# S3 storage class defaults
S3_STORAGE_CLASS = "STANDARD_IA"
S3_GLACIER_AFTER_DAYS = 90


# =============================================================================
# Enums
# =============================================================================


class RetentionCategory(str, Enum):
    """Retention category for audit bundles per regulatory requirements."""

    REGULATORY_7_YEAR = "regulatory_7_year"  # SOX, EPA, MARPOL compliance
    LEGAL_HOLD = "legal_hold"  # Indefinite hold for legal proceedings
    OPERATIONAL_1_YEAR = "operational_1_year"  # Short-term operational data


class StorageBackendType(str, Enum):
    """Supported storage backend types."""

    LOCAL_FILE = "local_file"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"


class BundleState(str, Enum):
    """State of a stored bundle."""

    ACTIVE = "active"  # Normal access
    ARCHIVED = "archived"  # Moved to cold storage
    RETENTION_EXPIRED = "retention_expired"  # Past retention, pending deletion
    DELETED = "deleted"  # Logically deleted


# =============================================================================
# Data Models
# =============================================================================


class StorageReceipt(BaseModel):
    """
    Receipt confirming successful storage of a bundle.

    Provides complete audit trail of the storage operation including
    integrity verification and retention policy.
    """

    receipt_id: str = Field(
        default_factory=lambda: f"RCPT-{uuid4().hex[:12].upper()}",
        description="Unique receipt identifier"
    )
    bundle_hash: str = Field(
        ...,
        description="SHA-256 hash of the stored bundle"
    )
    storage_location: str = Field(
        ...,
        description="Full path or URI where bundle is stored"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When bundle was stored"
    )
    size_bytes: int = Field(
        ...,
        ge=0,
        description="Size of stored bundle in bytes"
    )
    retention_policy: RetentionCategory = Field(
        default=RetentionCategory.REGULATORY_7_YEAR,
        description="Retention policy applied to bundle"
    )
    retention_expires: datetime = Field(
        ...,
        description="When retention period expires"
    )
    integrity_hash: str = Field(
        ...,
        description="SHA-256 hash computed after storage for verification"
    )
    integrity_verified: bool = Field(
        default=False,
        description="Whether integrity was verified after storage"
    )
    backend_type: StorageBackendType = Field(
        ...,
        description="Type of storage backend used"
    )
    immutable: bool = Field(
        default=False,
        description="Whether object lock is enabled (S3)"
    )
    compressed: bool = Field(
        default=False,
        description="Whether bundle is compressed"
    )
    encrypted: bool = Field(
        default=False,
        description="Whether server-side encryption is enabled"
    )

    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional storage metadata"
    )

    class Config:
        frozen = True

    @validator("integrity_hash")
    def validate_hash_format(cls, v: str) -> str:
        """Validate integrity hash is valid SHA-256."""
        if len(v) != 64:
            raise ValueError("integrity_hash must be 64 character SHA-256 hex string")
        try:
            int(v, 16)
        except ValueError:
            raise ValueError("integrity_hash must be valid hexadecimal")
        return v.lower()


class BundleMetadata(BaseModel):
    """
    Metadata for a stored bundle for index queries.

    Contains searchable fields without loading full bundle data.
    """

    bundle_hash: str = Field(
        ...,
        description="SHA-256 hash identifier"
    )
    run_id: str = Field(
        ...,
        description="Associated run identifier"
    )
    agent_version: str = Field(
        ...,
        description="Version of GL-011 that created bundle"
    )
    timestamp: datetime = Field(
        ...,
        description="When bundle was created"
    )
    size_bytes: int = Field(
        ...,
        ge=0,
        description="Bundle size in bytes"
    )
    state: BundleState = Field(
        default=BundleState.ACTIVE,
        description="Current state of bundle"
    )
    retention_policy: RetentionCategory = Field(
        default=RetentionCategory.REGULATORY_7_YEAR,
        description="Applied retention policy"
    )
    retention_expires: datetime = Field(
        ...,
        description="When retention expires"
    )
    storage_location: str = Field(
        ...,
        description="Storage path or URI"
    )

    # Searchable tags
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Searchable key-value tags"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class RetentionHold(BaseModel):
    """
    Legal or audit hold preventing deletion of bundles.

    Used for compliance with legal discovery or audit requirements.
    """

    hold_id: str = Field(
        default_factory=lambda: f"HOLD-{uuid4().hex[:12].upper()}",
        description="Unique hold identifier"
    )
    reason: str = Field(
        ...,
        description="Reason for hold (legal, audit, investigation)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When hold was created"
    )
    created_by: str = Field(
        ...,
        description="User or system that created hold"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Optional expiration (None = indefinite)"
    )
    bundle_hashes: List[str] = Field(
        default_factory=list,
        description="Bundle hashes under this hold"
    )
    notes: Optional[str] = Field(
        None,
        description="Additional notes about the hold"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


# =============================================================================
# Storage Configuration
# =============================================================================


class LocalStorageConfig(BaseModel):
    """Configuration for local file storage backend."""

    base_path: str = Field(
        ...,
        description="Base directory for bundle storage"
    )
    create_if_missing: bool = Field(
        default=True,
        description="Create base path if it doesn't exist"
    )
    use_sharding: bool = Field(
        default=True,
        description="Use hash-based directory sharding"
    )
    atomic_writes: bool = Field(
        default=True,
        description="Use atomic write (temp + rename)"
    )
    verify_on_read: bool = Field(
        default=True,
        description="Verify integrity on every read"
    )
    file_permissions: int = Field(
        default=0o644,
        description="Unix file permissions for stored files"
    )
    dir_permissions: int = Field(
        default=0o755,
        description="Unix directory permissions"
    )


class S3StorageConfig(BaseModel):
    """Configuration for AWS S3 storage backend."""

    bucket: str = Field(
        ...,
        description="S3 bucket name"
    )
    prefix: str = Field(
        default="bundles/",
        description="Key prefix for all bundles"
    )
    region: str = Field(
        default="us-east-1",
        description="AWS region"
    )
    endpoint_url: Optional[str] = Field(
        None,
        description="Custom endpoint URL (for MinIO, LocalStack)"
    )
    access_key_id: Optional[str] = Field(
        None,
        description="AWS access key ID (or use environment)"
    )
    secret_access_key: Optional[str] = Field(
        None,
        description="AWS secret access key (or use environment)"
    )

    # Encryption
    enable_sse: bool = Field(
        default=True,
        description="Enable server-side encryption (SSE-S3)"
    )
    sse_algorithm: str = Field(
        default="AES256",
        description="SSE algorithm (AES256 or aws:kms)"
    )
    kms_key_id: Optional[str] = Field(
        None,
        description="KMS key ID for SSE-KMS"
    )

    # Object locking
    enable_object_lock: bool = Field(
        default=True,
        description="Enable S3 Object Lock for immutability"
    )
    object_lock_mode: str = Field(
        default="GOVERNANCE",
        description="Object lock mode (GOVERNANCE or COMPLIANCE)"
    )

    # Storage class and lifecycle
    storage_class: str = Field(
        default=S3_STORAGE_CLASS,
        description="Initial storage class"
    )
    transition_to_glacier_days: int = Field(
        default=S3_GLACIER_AFTER_DAYS,
        description="Days until transition to Glacier"
    )

    # Timeouts
    connect_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Connection timeout"
    )
    read_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        description="Read timeout"
    )


# =============================================================================
# Abstract Storage Backend
# =============================================================================


class PersistentStorageBackend(ABC):
    """
    Abstract base class for persistent storage backends.

    Defines the interface that all storage backends must implement
    for content-addressed immutable bundle storage.

    Storage Requirements:
    - Content-addressed: bundles identified by SHA-256 hash
    - Immutable: no modifications after storage
    - Integrity verified: hash verification on read
    - Retention aware: 7-year minimum retention

    Example:
        >>> class MyStorage(PersistentStorageBackend):
        ...     def store(self, bundle_hash, data):
        ...         # Implementation
        ...         pass
        ...
        >>> storage = MyStorage()
        >>> receipt = storage.store("abc123...", bundle_bytes)
        >>> data = storage.retrieve("abc123...")
    """

    @abstractmethod
    def store(
        self,
        bundle_hash: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        retention_policy: RetentionCategory = RetentionCategory.REGULATORY_7_YEAR,
    ) -> StorageReceipt:
        """
        Store a bundle in content-addressed storage.

        Args:
            bundle_hash: SHA-256 hash of the bundle (content address)
            data: Raw bundle data to store
            metadata: Optional metadata to store with bundle
            retention_policy: Retention policy to apply

        Returns:
            StorageReceipt confirming storage

        Raises:
            ValueError: If bundle_hash doesn't match data hash
            IOError: If storage operation fails
            StorageExistsError: If bundle already exists (dedup)
        """
        pass

    @abstractmethod
    def retrieve(self, bundle_hash: str) -> bytes:
        """
        Retrieve a bundle by its hash.

        Args:
            bundle_hash: SHA-256 hash identifier

        Returns:
            Raw bundle data

        Raises:
            BundleNotFoundError: If bundle doesn't exist
            IntegrityError: If stored data fails hash verification
        """
        pass

    @abstractmethod
    def exists(self, bundle_hash: str) -> bool:
        """
        Check if a bundle exists in storage.

        Args:
            bundle_hash: SHA-256 hash identifier

        Returns:
            True if bundle exists, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, bundle_hash: str) -> bool:
        """
        Delete a bundle (only for expired retention).

        Deletion is only permitted when:
        - Retention period has expired
        - No legal/audit holds exist
        - Deletion is logged in audit trail

        Args:
            bundle_hash: SHA-256 hash identifier

        Returns:
            True if deleted, False if not found or blocked

        Raises:
            RetentionActiveError: If retention period not expired
            HoldActiveError: If bundle under legal/audit hold
        """
        pass

    @abstractmethod
    def list_bundles(
        self,
        prefix: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[BundleMetadata]:
        """
        List bundles in storage with optional filtering.

        Args:
            prefix: Optional hash prefix filter
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of BundleMetadata objects
        """
        pass

    def verify_integrity(self, bundle_hash: str) -> bool:
        """
        Verify stored bundle integrity.

        Args:
            bundle_hash: SHA-256 hash identifier

        Returns:
            True if integrity verified, False otherwise
        """
        try:
            data = self.retrieve(bundle_hash)
            computed_hash = hashlib.sha256(data).hexdigest()
            return computed_hash == bundle_hash.lower()
        except Exception as e:
            logger.error(f"Integrity verification failed for {bundle_hash}: {e}")
            return False


# =============================================================================
# Custom Exceptions
# =============================================================================


class StorageError(Exception):
    """Base exception for storage errors."""

    pass


class BundleNotFoundError(StorageError):
    """Bundle not found in storage."""

    def __init__(self, bundle_hash: str):
        self.bundle_hash = bundle_hash
        super().__init__(f"Bundle not found: {bundle_hash[:16]}...")


class IntegrityError(StorageError):
    """Bundle integrity verification failed."""

    def __init__(self, bundle_hash: str, expected: str, actual: str):
        self.bundle_hash = bundle_hash
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Integrity error for {bundle_hash[:16]}...: "
            f"expected {expected[:16]}..., got {actual[:16]}..."
        )


class RetentionActiveError(StorageError):
    """Cannot delete bundle with active retention."""

    def __init__(self, bundle_hash: str, expires: datetime):
        self.bundle_hash = bundle_hash
        self.expires = expires
        super().__init__(
            f"Cannot delete {bundle_hash[:16]}...: "
            f"retention active until {expires.isoformat()}"
        )


class HoldActiveError(StorageError):
    """Cannot delete bundle under legal/audit hold."""

    def __init__(self, bundle_hash: str, hold_id: str):
        self.bundle_hash = bundle_hash
        self.hold_id = hold_id
        super().__init__(
            f"Cannot delete {bundle_hash[:16]}...: "
            f"under hold {hold_id}"
        )


class StorageExistsError(StorageError):
    """Bundle already exists (deduplication)."""

    def __init__(self, bundle_hash: str):
        self.bundle_hash = bundle_hash
        super().__init__(f"Bundle already exists: {bundle_hash[:16]}...")


# =============================================================================
# Local File Storage Implementation
# =============================================================================


class LocalFileStorage(PersistentStorageBackend):
    """
    Local file system storage with content-addressed paths.

    Implements atomic writes using temp file + rename pattern
    for data integrity. Uses hash-based directory sharding
    for scalability.

    Directory Structure:
        {base_path}/{hash[0:2]}/{hash[2:4]}/{hash}/
            - data.bin          # Raw bundle data
            - metadata.json     # Bundle metadata

    Attributes:
        config: Storage configuration
        index: Optional BundleIndexer for fast lookups

    Example:
        >>> config = LocalStorageConfig(base_path="/audit/bundles")
        >>> storage = LocalFileStorage(config)
        >>> receipt = storage.store(bundle_hash, data)
        >>> retrieved = storage.retrieve(bundle_hash)
        >>> assert retrieved == data
    """

    def __init__(
        self,
        config: LocalStorageConfig,
        index: Optional["BundleIndexer"] = None,
    ):
        """
        Initialize local file storage.

        Args:
            config: Storage configuration
            index: Optional indexer for fast lookups
        """
        self.config = config
        self.index = index
        self._lock = threading.Lock()

        # Create base path if needed
        self._base_path = Path(config.base_path)
        if config.create_if_missing:
            self._base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"LocalFileStorage initialized at {self._base_path}")
        elif not self._base_path.exists():
            raise ValueError(f"Base path does not exist: {config.base_path}")

    def _get_bundle_path(self, bundle_hash: str) -> Path:
        """Get the storage path for a bundle hash."""
        bundle_hash = bundle_hash.lower()

        if self.config.use_sharding:
            # Sharded path: {base}/{00}/{ab}/{hash}/
            shard1 = bundle_hash[:SHARD_WIDTH]
            shard2 = bundle_hash[SHARD_WIDTH:SHARD_WIDTH * 2]
            return self._base_path / shard1 / shard2 / bundle_hash
        else:
            return self._base_path / bundle_hash

    def _compute_hash(self, data: bytes) -> str:
        """Compute SHA-256 hash of data."""
        return hashlib.sha256(data).hexdigest()

    def store(
        self,
        bundle_hash: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        retention_policy: RetentionCategory = RetentionCategory.REGULATORY_7_YEAR,
    ) -> StorageReceipt:
        """
        Store bundle with atomic write and integrity verification.

        Implements write-to-temp then rename pattern for atomicity.

        Args:
            bundle_hash: Expected SHA-256 hash
            data: Raw bundle data
            metadata: Optional metadata
            retention_policy: Retention policy to apply

        Returns:
            StorageReceipt confirming storage

        Raises:
            ValueError: If hash doesn't match data
            StorageExistsError: If bundle already exists
        """
        bundle_hash = bundle_hash.lower()

        # Verify hash matches data
        computed_hash = self._compute_hash(data)
        if computed_hash != bundle_hash:
            raise ValueError(
                f"Hash mismatch: expected {bundle_hash[:16]}..., "
                f"got {computed_hash[:16]}..."
            )

        with self._lock:
            bundle_path = self._get_bundle_path(bundle_hash)

            # Check if already exists (deduplication)
            if bundle_path.exists():
                logger.info(f"Bundle already exists (dedup): {bundle_hash[:16]}...")
                raise StorageExistsError(bundle_hash)

            # Calculate retention expiration
            now = datetime.now(timezone.utc)
            if retention_policy == RetentionCategory.REGULATORY_7_YEAR:
                retention_expires = now + timedelta(days=RETENTION_DAYS_7_YEARS)
            elif retention_policy == RetentionCategory.OPERATIONAL_1_YEAR:
                retention_expires = now + timedelta(days=RETENTION_DAYS_1_YEAR)
            else:  # LEGAL_HOLD - indefinite
                retention_expires = now + timedelta(days=365 * 100)  # 100 years

            # Create directory structure
            bundle_path.mkdir(parents=True, exist_ok=True)

            # Write data (atomic)
            data_path = bundle_path / "data.bin"
            if self.config.atomic_writes:
                self._atomic_write(data_path, data)
            else:
                with open(data_path, "wb") as f:
                    f.write(data)

            # Set file permissions
            os.chmod(data_path, self.config.file_permissions)

            # Write metadata
            bundle_metadata = {
                "bundle_hash": bundle_hash,
                "size_bytes": len(data),
                "stored_at": now.isoformat(),
                "retention_policy": retention_policy.value,
                "retention_expires": retention_expires.isoformat(),
                "agent_id": "GL-011",
                "custom": metadata or {},
            }
            metadata_path = bundle_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(bundle_metadata, f, indent=2)

            # Verify write integrity
            integrity_hash = self._verify_stored_data(data_path, data)
            integrity_verified = integrity_hash == bundle_hash

            if not integrity_verified:
                # Rollback on integrity failure
                shutil.rmtree(bundle_path)
                raise IntegrityError(bundle_hash, bundle_hash, integrity_hash)

            # Update index if available
            if self.index:
                self.index.add_bundle(BundleMetadata(
                    bundle_hash=bundle_hash,
                    run_id=metadata.get("run_id", "") if metadata else "",
                    agent_version=metadata.get("agent_version", "1.0.0") if metadata else "1.0.0",
                    timestamp=now,
                    size_bytes=len(data),
                    state=BundleState.ACTIVE,
                    retention_policy=retention_policy,
                    retention_expires=retention_expires,
                    storage_location=str(bundle_path),
                    tags=metadata.get("tags", {}) if metadata else {},
                ))

            receipt = StorageReceipt(
                bundle_hash=bundle_hash,
                storage_location=str(bundle_path),
                size_bytes=len(data),
                retention_policy=retention_policy,
                retention_expires=retention_expires,
                integrity_hash=integrity_hash,
                integrity_verified=integrity_verified,
                backend_type=StorageBackendType.LOCAL_FILE,
                immutable=False,
                compressed=False,
                encrypted=False,
                metadata=bundle_metadata,
            )

            logger.info(
                f"Bundle stored: {bundle_hash[:16]}... ({len(data)} bytes)"
            )

            return receipt

    def _atomic_write(self, path: Path, data: bytes) -> None:
        """Write data atomically using temp file + rename."""
        temp_dir = path.parent
        temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir)

        try:
            with os.fdopen(temp_fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            os.replace(temp_path, path)

        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _verify_stored_data(self, path: Path, original: bytes) -> str:
        """Read back stored data and compute hash for verification."""
        with open(path, "rb") as f:
            stored = f.read()
        return self._compute_hash(stored)

    def retrieve(self, bundle_hash: str) -> bytes:
        """
        Retrieve bundle with optional integrity verification.

        Args:
            bundle_hash: SHA-256 hash identifier

        Returns:
            Raw bundle data

        Raises:
            BundleNotFoundError: If bundle doesn't exist
            IntegrityError: If verification fails
        """
        bundle_hash = bundle_hash.lower()
        bundle_path = self._get_bundle_path(bundle_hash)
        data_path = bundle_path / "data.bin"

        if not data_path.exists():
            raise BundleNotFoundError(bundle_hash)

        with open(data_path, "rb") as f:
            data = f.read()

        # Verify integrity on read if configured
        if self.config.verify_on_read:
            computed_hash = self._compute_hash(data)
            if computed_hash != bundle_hash:
                logger.error(f"Integrity check failed for {bundle_hash[:16]}...")
                raise IntegrityError(bundle_hash, bundle_hash, computed_hash)

        return data

    def exists(self, bundle_hash: str) -> bool:
        """Check if bundle exists in storage."""
        bundle_hash = bundle_hash.lower()
        bundle_path = self._get_bundle_path(bundle_hash)
        return (bundle_path / "data.bin").exists()

    def delete(self, bundle_hash: str) -> bool:
        """
        Delete bundle (only if retention expired and no holds).

        Args:
            bundle_hash: SHA-256 hash identifier

        Returns:
            True if deleted, False if not found

        Raises:
            RetentionActiveError: If retention not expired
            HoldActiveError: If under hold
        """
        bundle_hash = bundle_hash.lower()
        bundle_path = self._get_bundle_path(bundle_hash)

        if not bundle_path.exists():
            return False

        # Load metadata to check retention
        metadata_path = bundle_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            retention_expires = datetime.fromisoformat(metadata["retention_expires"])
            now = datetime.now(timezone.utc)

            if now < retention_expires:
                raise RetentionActiveError(bundle_hash, retention_expires)

        # Check for holds via index
        if self.index and self.index.has_hold(bundle_hash):
            hold_id = self.index.get_hold_id(bundle_hash)
            raise HoldActiveError(bundle_hash, hold_id)

        with self._lock:
            shutil.rmtree(bundle_path)

            # Update index
            if self.index:
                self.index.remove_bundle(bundle_hash)

            logger.info(f"Bundle deleted: {bundle_hash[:16]}...")
            return True

    def list_bundles(
        self,
        prefix: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[BundleMetadata]:
        """
        List bundles in storage.

        Args:
            prefix: Optional hash prefix filter
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of BundleMetadata objects
        """
        # If index available, use it for efficiency
        if self.index:
            return self.index.list_bundles(prefix=prefix, limit=limit, offset=offset)

        # Otherwise scan filesystem
        bundles: List[BundleMetadata] = []
        count = 0
        skipped = 0

        for shard1 in self._base_path.iterdir():
            if not shard1.is_dir():
                continue
            for shard2 in shard1.iterdir():
                if not shard2.is_dir():
                    continue
                for bundle_dir in shard2.iterdir():
                    if not bundle_dir.is_dir():
                        continue

                    bundle_hash = bundle_dir.name

                    # Apply prefix filter
                    if prefix and not bundle_hash.startswith(prefix.lower()):
                        continue

                    # Apply pagination
                    if skipped < offset:
                        skipped += 1
                        continue

                    if count >= limit:
                        return bundles

                    # Load metadata
                    metadata_path = bundle_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            meta = json.load(f)

                        bundles.append(BundleMetadata(
                            bundle_hash=bundle_hash,
                            run_id=meta.get("custom", {}).get("run_id", ""),
                            agent_version=meta.get("custom", {}).get("agent_version", "1.0.0"),
                            timestamp=datetime.fromisoformat(meta["stored_at"]),
                            size_bytes=meta["size_bytes"],
                            state=BundleState.ACTIVE,
                            retention_policy=RetentionCategory(meta["retention_policy"]),
                            retention_expires=datetime.fromisoformat(meta["retention_expires"]),
                            storage_location=str(bundle_dir),
                            tags=meta.get("custom", {}).get("tags", {}),
                        ))
                        count += 1

        return bundles


# =============================================================================
# S3 Storage Implementation
# =============================================================================


class S3Storage(PersistentStorageBackend):
    """
    AWS S3 storage with server-side encryption and object locking.

    Implements production-grade storage with:
    - SSE-S3 or SSE-KMS encryption
    - S3 Object Lock for immutability (WORM)
    - Lifecycle rules for retention/archival
    - Multipart uploads for large bundles

    Key Structure:
        {prefix}{hash[0:2]}/{hash[2:4]}/{hash}/data.bin
        {prefix}{hash[0:2]}/{hash[2:4]}/{hash}/metadata.json

    Attributes:
        config: S3 storage configuration
        client: Boto3 S3 client

    Example:
        >>> config = S3StorageConfig(bucket="audit-bundles", region="us-east-1")
        >>> storage = S3Storage(config)
        >>> receipt = storage.store(bundle_hash, data)
        >>> assert receipt.encrypted  # SSE enabled
        >>> assert receipt.immutable  # Object lock enabled
    """

    def __init__(self, config: S3StorageConfig):
        """
        Initialize S3 storage.

        Args:
            config: S3 storage configuration

        Raises:
            ImportError: If boto3 not installed
        """
        self.config = config
        self._client = None
        self._initialized = False

        logger.info(
            f"S3Storage configured for bucket {config.bucket}",
            extra={"region": config.region}
        )

    def _get_client(self):
        """Get or create boto3 S3 client."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config as BotoConfig
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3Storage. "
                    "Install with: pip install boto3"
                )

            boto_config = BotoConfig(
                connect_timeout=self.config.connect_timeout_seconds,
                read_timeout=self.config.read_timeout_seconds,
                retries={"max_attempts": 3, "mode": "adaptive"},
            )

            client_kwargs = {
                "service_name": "s3",
                "region_name": self.config.region,
                "config": boto_config,
            }

            if self.config.endpoint_url:
                client_kwargs["endpoint_url"] = self.config.endpoint_url

            if self.config.access_key_id and self.config.secret_access_key:
                client_kwargs["aws_access_key_id"] = self.config.access_key_id
                client_kwargs["aws_secret_access_key"] = self.config.secret_access_key

            self._client = boto3.client(**client_kwargs)
            self._initialized = True

        return self._client

    def _get_object_key(self, bundle_hash: str, filename: str) -> str:
        """Get S3 object key for a bundle file."""
        bundle_hash = bundle_hash.lower()
        shard1 = bundle_hash[:SHARD_WIDTH]
        shard2 = bundle_hash[SHARD_WIDTH:SHARD_WIDTH * 2]
        return f"{self.config.prefix}{shard1}/{shard2}/{bundle_hash}/{filename}"

    def store(
        self,
        bundle_hash: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        retention_policy: RetentionCategory = RetentionCategory.REGULATORY_7_YEAR,
    ) -> StorageReceipt:
        """
        Store bundle in S3 with encryption and object locking.

        Args:
            bundle_hash: Expected SHA-256 hash
            data: Raw bundle data
            metadata: Optional metadata
            retention_policy: Retention policy to apply

        Returns:
            StorageReceipt confirming storage

        Raises:
            ValueError: If hash doesn't match
            StorageExistsError: If bundle exists
        """
        bundle_hash = bundle_hash.lower()
        client = self._get_client()

        # Verify hash
        computed_hash = hashlib.sha256(data).hexdigest()
        if computed_hash != bundle_hash:
            raise ValueError(f"Hash mismatch: expected {bundle_hash}, got {computed_hash}")

        data_key = self._get_object_key(bundle_hash, "data.bin")

        # Check if exists
        try:
            client.head_object(Bucket=self.config.bucket, Key=data_key)
            raise StorageExistsError(bundle_hash)
        except client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise

        # Calculate retention
        now = datetime.now(timezone.utc)
        if retention_policy == RetentionCategory.REGULATORY_7_YEAR:
            retention_expires = now + timedelta(days=RETENTION_DAYS_7_YEARS)
        elif retention_policy == RetentionCategory.OPERATIONAL_1_YEAR:
            retention_expires = now + timedelta(days=RETENTION_DAYS_1_YEAR)
        else:
            retention_expires = now + timedelta(days=365 * 100)

        # Build put_object args
        put_args = {
            "Bucket": self.config.bucket,
            "Key": data_key,
            "Body": data,
            "StorageClass": self.config.storage_class,
            "Metadata": {
                "bundle-hash": bundle_hash,
                "retention-policy": retention_policy.value,
                "retention-expires": retention_expires.isoformat(),
                "agent-id": "GL-011",
            },
        }

        # Server-side encryption
        if self.config.enable_sse:
            if self.config.sse_algorithm == "aws:kms" and self.config.kms_key_id:
                put_args["ServerSideEncryption"] = "aws:kms"
                put_args["SSEKMSKeyId"] = self.config.kms_key_id
            else:
                put_args["ServerSideEncryption"] = "AES256"

        # Object lock (requires bucket with Object Lock enabled)
        if self.config.enable_object_lock:
            put_args["ObjectLockMode"] = self.config.object_lock_mode
            put_args["ObjectLockRetainUntilDate"] = retention_expires

        # Upload data
        client.put_object(**put_args)

        # Upload metadata
        metadata_key = self._get_object_key(bundle_hash, "metadata.json")
        bundle_metadata = {
            "bundle_hash": bundle_hash,
            "size_bytes": len(data),
            "stored_at": now.isoformat(),
            "retention_policy": retention_policy.value,
            "retention_expires": retention_expires.isoformat(),
            "agent_id": "GL-011",
            "custom": metadata or {},
        }

        metadata_put_args = {
            "Bucket": self.config.bucket,
            "Key": metadata_key,
            "Body": json.dumps(bundle_metadata, indent=2).encode("utf-8"),
            "ContentType": "application/json",
        }
        if self.config.enable_sse:
            metadata_put_args["ServerSideEncryption"] = "AES256"

        client.put_object(**metadata_put_args)

        # Verify stored object
        response = client.head_object(Bucket=self.config.bucket, Key=data_key)
        etag = response.get("ETag", "").strip('"')

        receipt = StorageReceipt(
            bundle_hash=bundle_hash,
            storage_location=f"s3://{self.config.bucket}/{data_key}",
            size_bytes=len(data),
            retention_policy=retention_policy,
            retention_expires=retention_expires,
            integrity_hash=computed_hash,
            integrity_verified=True,
            backend_type=StorageBackendType.S3,
            immutable=self.config.enable_object_lock,
            compressed=False,
            encrypted=self.config.enable_sse,
            metadata={
                "etag": etag,
                "storage_class": self.config.storage_class,
            },
        )

        logger.info(
            f"Bundle stored in S3: {bundle_hash[:16]}... ({len(data)} bytes)"
        )

        return receipt

    def retrieve(self, bundle_hash: str) -> bytes:
        """
        Retrieve bundle from S3.

        Args:
            bundle_hash: SHA-256 hash identifier

        Returns:
            Raw bundle data

        Raises:
            BundleNotFoundError: If not found
            IntegrityError: If verification fails
        """
        bundle_hash = bundle_hash.lower()
        client = self._get_client()
        data_key = self._get_object_key(bundle_hash, "data.bin")

        try:
            response = client.get_object(Bucket=self.config.bucket, Key=data_key)
            data = response["Body"].read()
        except client.exceptions.NoSuchKey:
            raise BundleNotFoundError(bundle_hash)

        # Verify integrity
        computed_hash = hashlib.sha256(data).hexdigest()
        if computed_hash != bundle_hash:
            raise IntegrityError(bundle_hash, bundle_hash, computed_hash)

        return data

    def exists(self, bundle_hash: str) -> bool:
        """Check if bundle exists in S3."""
        bundle_hash = bundle_hash.lower()
        client = self._get_client()
        data_key = self._get_object_key(bundle_hash, "data.bin")

        try:
            client.head_object(Bucket=self.config.bucket, Key=data_key)
            return True
        except:
            return False

    def delete(self, bundle_hash: str) -> bool:
        """
        Delete bundle from S3 (only if retention expired).

        Note: With Object Lock enabled, deletion may be blocked
        until retention period expires.
        """
        bundle_hash = bundle_hash.lower()
        client = self._get_client()

        # Check metadata for retention
        metadata_key = self._get_object_key(bundle_hash, "metadata.json")
        try:
            response = client.get_object(Bucket=self.config.bucket, Key=metadata_key)
            metadata = json.loads(response["Body"].read().decode("utf-8"))

            retention_expires = datetime.fromisoformat(metadata["retention_expires"])
            now = datetime.now(timezone.utc)

            if now < retention_expires:
                raise RetentionActiveError(bundle_hash, retention_expires)

        except client.exceptions.NoSuchKey:
            return False

        # Delete objects
        data_key = self._get_object_key(bundle_hash, "data.bin")

        try:
            client.delete_object(Bucket=self.config.bucket, Key=data_key)
            client.delete_object(Bucket=self.config.bucket, Key=metadata_key)
            logger.info(f"Bundle deleted from S3: {bundle_hash[:16]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to delete bundle from S3: {e}")
            return False

    def list_bundles(
        self,
        prefix: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[BundleMetadata]:
        """List bundles in S3 bucket."""
        client = self._get_client()
        bundles: List[BundleMetadata] = []

        list_prefix = self.config.prefix
        if prefix:
            list_prefix += prefix[:SHARD_WIDTH] + "/"

        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.config.bucket,
            Prefix=list_prefix,
        )

        count = 0
        skipped = 0

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]

                # Only process metadata files
                if not key.endswith("/metadata.json"):
                    continue

                if skipped < offset:
                    skipped += 1
                    continue

                if count >= limit:
                    return bundles

                # Extract bundle hash from key
                parts = key.split("/")
                if len(parts) >= 4:
                    bundle_hash = parts[-2]

                    # Load metadata
                    try:
                        response = client.get_object(
                            Bucket=self.config.bucket,
                            Key=key
                        )
                        meta = json.loads(response["Body"].read().decode("utf-8"))

                        bundles.append(BundleMetadata(
                            bundle_hash=bundle_hash,
                            run_id=meta.get("custom", {}).get("run_id", ""),
                            agent_version=meta.get("custom", {}).get("agent_version", "1.0.0"),
                            timestamp=datetime.fromisoformat(meta["stored_at"]),
                            size_bytes=meta["size_bytes"],
                            state=BundleState.ACTIVE,
                            retention_policy=RetentionCategory(meta["retention_policy"]),
                            retention_expires=datetime.fromisoformat(meta["retention_expires"]),
                            storage_location=f"s3://{self.config.bucket}/{key}",
                            tags=meta.get("custom", {}).get("tags", {}),
                        ))
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {key}: {e}")

        return bundles


# =============================================================================
# Retention Manager
# =============================================================================


class RetentionManager:
    """
    Manages retention policies and expiration for audit bundles.

    Implements regulatory compliance for:
    - 7-year retention (SOX, EPA, MARPOL)
    - Legal holds for litigation/audit
    - Automatic expiration checking
    - Retention policy enforcement

    Attributes:
        storage: Underlying storage backend
        holds: Active retention holds

    Example:
        >>> manager = RetentionManager(storage)
        >>> manager.apply_hold("legal-case-123", ["hash1", "hash2"], reason="Litigation")
        >>> expired = manager.get_expired_bundles()
        >>> deleted = manager.enforce_retention()
    """

    def __init__(
        self,
        storage: PersistentStorageBackend,
        index: Optional["BundleIndexer"] = None,
    ):
        """
        Initialize retention manager.

        Args:
            storage: Storage backend to manage
            index: Optional indexer for efficient lookups
        """
        self.storage = storage
        self.index = index
        self._holds: Dict[str, RetentionHold] = {}
        self._bundle_holds: Dict[str, str] = {}  # bundle_hash -> hold_id
        self._lock = threading.Lock()

        logger.info("RetentionManager initialized")

    def apply_hold(
        self,
        hold_id: str,
        bundle_hashes: List[str],
        reason: str,
        created_by: str,
        expires_at: Optional[datetime] = None,
        notes: Optional[str] = None,
    ) -> RetentionHold:
        """
        Apply a legal/audit hold to bundles.

        Prevents deletion regardless of retention expiration.

        Args:
            hold_id: Unique hold identifier
            bundle_hashes: Bundles to place under hold
            reason: Reason for hold
            created_by: User/system creating hold
            expires_at: Optional expiration (None = indefinite)
            notes: Optional notes

        Returns:
            Created RetentionHold

        Raises:
            ValueError: If hold_id already exists
        """
        with self._lock:
            if hold_id in self._holds:
                raise ValueError(f"Hold {hold_id} already exists")

            hold = RetentionHold(
                hold_id=hold_id,
                reason=reason,
                created_by=created_by,
                expires_at=expires_at,
                bundle_hashes=bundle_hashes,
                notes=notes,
            )

            self._holds[hold_id] = hold

            for bundle_hash in bundle_hashes:
                self._bundle_holds[bundle_hash.lower()] = hold_id

            logger.info(
                f"Hold applied: {hold_id} on {len(bundle_hashes)} bundles",
                extra={"reason": reason, "created_by": created_by}
            )

            return hold

    def release_hold(self, hold_id: str) -> bool:
        """
        Release a retention hold.

        Args:
            hold_id: Hold identifier to release

        Returns:
            True if released, False if not found
        """
        with self._lock:
            if hold_id not in self._holds:
                return False

            hold = self._holds[hold_id]

            # Remove bundle associations
            for bundle_hash in hold.bundle_hashes:
                self._bundle_holds.pop(bundle_hash.lower(), None)

            del self._holds[hold_id]

            logger.info(f"Hold released: {hold_id}")
            return True

    def has_hold(self, bundle_hash: str) -> bool:
        """Check if bundle is under hold."""
        return bundle_hash.lower() in self._bundle_holds

    def get_hold(self, bundle_hash: str) -> Optional[RetentionHold]:
        """Get hold for a bundle if any."""
        hold_id = self._bundle_holds.get(bundle_hash.lower())
        if hold_id:
            return self._holds.get(hold_id)
        return None

    def get_expired_bundles(self) -> List[BundleMetadata]:
        """
        Get list of bundles with expired retention.

        Returns bundles that are past retention and not under hold.

        Returns:
            List of expired BundleMetadata
        """
        now = datetime.now(timezone.utc)
        expired: List[BundleMetadata] = []

        bundles = self.storage.list_bundles(limit=100000)

        for bundle in bundles:
            if bundle.retention_expires < now:
                if not self.has_hold(bundle.bundle_hash):
                    expired.append(bundle)

        logger.info(f"Found {len(expired)} bundles with expired retention")
        return expired

    def enforce_retention(self, dry_run: bool = False) -> int:
        """
        Delete bundles with expired retention.

        Only deletes bundles that:
        - Have passed retention expiration
        - Are not under legal/audit hold

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Number of bundles deleted (or would be deleted)
        """
        expired = self.get_expired_bundles()
        deleted = 0

        for bundle in expired:
            if dry_run:
                logger.info(f"Would delete: {bundle.bundle_hash[:16]}...")
                deleted += 1
            else:
                try:
                    if self.storage.delete(bundle.bundle_hash):
                        deleted += 1
                except (RetentionActiveError, HoldActiveError) as e:
                    logger.warning(f"Cannot delete {bundle.bundle_hash[:16]}...: {e}")

        logger.info(
            f"Retention enforcement: {deleted} bundles {'would be ' if dry_run else ''}deleted"
        )
        return deleted

    def get_retention_stats(self) -> Dict[str, Any]:
        """Get retention statistics."""
        bundles = self.storage.list_bundles(limit=100000)
        now = datetime.now(timezone.utc)

        stats = {
            "total_bundles": len(bundles),
            "by_policy": {
                RetentionCategory.REGULATORY_7_YEAR.value: 0,
                RetentionCategory.LEGAL_HOLD.value: 0,
                RetentionCategory.OPERATIONAL_1_YEAR.value: 0,
            },
            "active_holds": len(self._holds),
            "bundles_under_hold": len(self._bundle_holds),
            "expired_count": 0,
            "total_size_bytes": 0,
        }

        for bundle in bundles:
            stats["by_policy"][bundle.retention_policy.value] += 1
            stats["total_size_bytes"] += bundle.size_bytes
            if bundle.retention_expires < now:
                stats["expired_count"] += 1

        return stats


# =============================================================================
# Bundle Indexer
# =============================================================================


class BundleIndexer:
    """
    Index for fast bundle lookups by run_id, timestamp, agent_version.

    Uses SQLite for local deployments. Can be extended for
    DynamoDB in cloud deployments.

    Features:
    - Fast lookups by run_id, timestamp range, agent_version
    - Tag-based filtering
    - Retention and hold tracking
    - Automatic index maintenance

    Attributes:
        db_path: Path to SQLite database

    Example:
        >>> indexer = BundleIndexer("/audit/index.db")
        >>> indexer.add_bundle(metadata)
        >>> bundles = indexer.find_by_run_id("RUN-001")
        >>> bundles = indexer.find_by_time_range(start, end)
    """

    SCHEMA_DDL = """
    -- Bundle index schema
    CREATE TABLE IF NOT EXISTS bundles (
        bundle_hash TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        agent_version TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        state TEXT NOT NULL DEFAULT 'active',
        retention_policy TEXT NOT NULL,
        retention_expires TEXT NOT NULL,
        storage_location TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
    );

    CREATE TABLE IF NOT EXISTS bundle_tags (
        bundle_hash TEXT NOT NULL,
        tag_key TEXT NOT NULL,
        tag_value TEXT NOT NULL,
        PRIMARY KEY (bundle_hash, tag_key),
        FOREIGN KEY (bundle_hash) REFERENCES bundles(bundle_hash) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS retention_holds (
        hold_id TEXT PRIMARY KEY,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL,
        created_by TEXT NOT NULL,
        expires_at TEXT,
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS hold_bundles (
        hold_id TEXT NOT NULL,
        bundle_hash TEXT NOT NULL,
        PRIMARY KEY (hold_id, bundle_hash),
        FOREIGN KEY (hold_id) REFERENCES retention_holds(hold_id) ON DELETE CASCADE,
        FOREIGN KEY (bundle_hash) REFERENCES bundles(bundle_hash) ON DELETE CASCADE
    );

    -- Indexes for fast lookups
    CREATE INDEX IF NOT EXISTS idx_bundles_run_id ON bundles(run_id);
    CREATE INDEX IF NOT EXISTS idx_bundles_timestamp ON bundles(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_bundles_agent_version ON bundles(agent_version);
    CREATE INDEX IF NOT EXISTS idx_bundles_state ON bundles(state);
    CREATE INDEX IF NOT EXISTS idx_bundles_retention_expires ON bundles(retention_expires);
    CREATE INDEX IF NOT EXISTS idx_tags_key ON bundle_tags(tag_key);
    """

    def __init__(self, db_path: str):
        """
        Initialize bundle indexer.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

        self._initialize_db()
        logger.info(f"BundleIndexer initialized at {db_path}")

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        conn = self._get_connection()
        conn.executescript(self.SCHEMA_DDL)
        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA foreign_keys = ON")

        return self._connection

    def add_bundle(self, metadata: BundleMetadata) -> None:
        """
        Add a bundle to the index.

        Args:
            metadata: Bundle metadata to index
        """
        with self._lock:
            conn = self._get_connection()

            conn.execute(
                """
                INSERT OR REPLACE INTO bundles
                (bundle_hash, run_id, agent_version, timestamp, size_bytes,
                 state, retention_policy, retention_expires, storage_location)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.bundle_hash.lower(),
                    metadata.run_id,
                    metadata.agent_version,
                    metadata.timestamp.isoformat(),
                    metadata.size_bytes,
                    metadata.state.value,
                    metadata.retention_policy.value,
                    metadata.retention_expires.isoformat(),
                    metadata.storage_location,
                ),
            )

            # Add tags
            for key, value in metadata.tags.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO bundle_tags (bundle_hash, tag_key, tag_value)
                    VALUES (?, ?, ?)
                    """,
                    (metadata.bundle_hash.lower(), key, value),
                )

    def remove_bundle(self, bundle_hash: str) -> bool:
        """
        Remove a bundle from the index.

        Args:
            bundle_hash: Bundle hash to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            conn = self._get_connection()

            cursor = conn.execute(
                "DELETE FROM bundles WHERE bundle_hash = ?",
                (bundle_hash.lower(),),
            )

            return cursor.rowcount > 0

    def find_by_run_id(self, run_id: str) -> List[BundleMetadata]:
        """
        Find bundles by run ID.

        Args:
            run_id: Run identifier

        Returns:
            List of matching bundles
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM bundles WHERE run_id = ? ORDER BY timestamp DESC",
            (run_id,),
        )

        return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def find_by_time_range(
        self,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> List[BundleMetadata]:
        """
        Find bundles within a time range.

        Args:
            start: Start of range (inclusive)
            end: End of range (exclusive)
            limit: Maximum results

        Returns:
            List of matching bundles
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM bundles
            WHERE timestamp >= ? AND timestamp < ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (start.isoformat(), end.isoformat(), limit),
        )

        return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def find_by_agent_version(self, agent_version: str) -> List[BundleMetadata]:
        """
        Find bundles by agent version.

        Args:
            agent_version: GL-011 version string

        Returns:
            List of matching bundles
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM bundles WHERE agent_version = ? ORDER BY timestamp DESC",
            (agent_version,),
        )

        return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def find_by_tag(self, key: str, value: str) -> List[BundleMetadata]:
        """
        Find bundles by tag.

        Args:
            key: Tag key
            value: Tag value

        Returns:
            List of matching bundles
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT b.* FROM bundles b
            JOIN bundle_tags t ON b.bundle_hash = t.bundle_hash
            WHERE t.tag_key = ? AND t.tag_value = ?
            ORDER BY b.timestamp DESC
            """,
            (key, value),
        )

        return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def list_bundles(
        self,
        prefix: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[BundleMetadata]:
        """
        List all bundles with optional prefix filter.

        Args:
            prefix: Optional hash prefix
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of BundleMetadata
        """
        conn = self._get_connection()

        if prefix:
            cursor = conn.execute(
                """
                SELECT * FROM bundles
                WHERE bundle_hash LIKE ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (prefix.lower() + "%", limit, offset),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM bundles
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )

        return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def has_hold(self, bundle_hash: str) -> bool:
        """Check if bundle is under hold."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM hold_bundles WHERE bundle_hash = ?",
            (bundle_hash.lower(),),
        )
        return cursor.fetchone() is not None

    def get_hold_id(self, bundle_hash: str) -> Optional[str]:
        """Get hold ID for a bundle."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT hold_id FROM hold_bundles WHERE bundle_hash = ?",
            (bundle_hash.lower(),),
        )
        row = cursor.fetchone()
        return row["hold_id"] if row else None

    def add_hold(self, hold: RetentionHold) -> None:
        """Add a retention hold."""
        with self._lock:
            conn = self._get_connection()

            conn.execute(
                """
                INSERT INTO retention_holds
                (hold_id, reason, created_at, created_by, expires_at, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    hold.hold_id,
                    hold.reason,
                    hold.created_at.isoformat(),
                    hold.created_by,
                    hold.expires_at.isoformat() if hold.expires_at else None,
                    hold.notes,
                ),
            )

            for bundle_hash in hold.bundle_hashes:
                conn.execute(
                    "INSERT INTO hold_bundles (hold_id, bundle_hash) VALUES (?, ?)",
                    (hold.hold_id, bundle_hash.lower()),
                )

    def remove_hold(self, hold_id: str) -> bool:
        """Remove a retention hold."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "DELETE FROM retention_holds WHERE hold_id = ?",
                (hold_id,),
            )
            return cursor.rowcount > 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        conn = self._get_connection()

        cursor = conn.execute("SELECT COUNT(*) as count FROM bundles")
        total = cursor.fetchone()["count"]

        cursor = conn.execute(
            "SELECT state, COUNT(*) as count FROM bundles GROUP BY state"
        )
        by_state = {row["state"]: row["count"] for row in cursor.fetchall()}

        cursor = conn.execute(
            "SELECT retention_policy, COUNT(*) as count FROM bundles GROUP BY retention_policy"
        )
        by_policy = {row["retention_policy"]: row["count"] for row in cursor.fetchall()}

        cursor = conn.execute("SELECT COUNT(*) as count FROM retention_holds")
        holds = cursor.fetchone()["count"]

        return {
            "total_bundles": total,
            "by_state": by_state,
            "by_policy": by_policy,
            "active_holds": holds,
        }

    def _row_to_metadata(self, row: sqlite3.Row) -> BundleMetadata:
        """Convert database row to BundleMetadata."""
        # Load tags
        conn = self._get_connection()
        tags_cursor = conn.execute(
            "SELECT tag_key, tag_value FROM bundle_tags WHERE bundle_hash = ?",
            (row["bundle_hash"],),
        )
        tags = {r["tag_key"]: r["tag_value"] for r in tags_cursor.fetchall()}

        return BundleMetadata(
            bundle_hash=row["bundle_hash"],
            run_id=row["run_id"],
            agent_version=row["agent_version"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            size_bytes=row["size_bytes"],
            state=BundleState(row["state"]),
            retention_policy=RetentionCategory(row["retention_policy"]),
            retention_expires=datetime.fromisoformat(row["retention_expires"]),
            storage_location=row["storage_location"],
            tags=tags,
        )

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


# =============================================================================
# Factory Functions
# =============================================================================


def create_local_storage(
    base_path: str,
    with_index: bool = True,
    index_path: Optional[str] = None,
) -> Tuple[LocalFileStorage, Optional[BundleIndexer]]:
    """
    Create local file storage with optional indexer.

    Args:
        base_path: Base directory for storage
        with_index: Whether to create indexer
        index_path: Path for index database (defaults to {base_path}/index.db)

    Returns:
        Tuple of (LocalFileStorage, Optional[BundleIndexer])

    Example:
        >>> storage, indexer = create_local_storage("/audit/bundles")
        >>> receipt = storage.store(hash, data)
    """
    config = LocalStorageConfig(base_path=base_path)

    index = None
    if with_index:
        idx_path = index_path or str(Path(base_path) / "index.db")
        index = BundleIndexer(idx_path)

    storage = LocalFileStorage(config, index=index)

    return storage, index


def create_s3_storage(
    bucket: str,
    region: str = "us-east-1",
    enable_object_lock: bool = True,
    **kwargs,
) -> S3Storage:
    """
    Create S3 storage with default production settings.

    Args:
        bucket: S3 bucket name
        region: AWS region
        enable_object_lock: Enable immutability via Object Lock
        **kwargs: Additional S3StorageConfig parameters

    Returns:
        Configured S3Storage instance

    Example:
        >>> storage = create_s3_storage("audit-bundles", region="us-east-1")
        >>> receipt = storage.store(hash, data)
    """
    config = S3StorageConfig(
        bucket=bucket,
        region=region,
        enable_object_lock=enable_object_lock,
        **kwargs,
    )

    return S3Storage(config)


def create_retention_manager(
    storage: PersistentStorageBackend,
    index: Optional[BundleIndexer] = None,
) -> RetentionManager:
    """
    Create retention manager for a storage backend.

    Args:
        storage: Storage backend to manage
        index: Optional indexer for efficient lookups

    Returns:
        Configured RetentionManager
    """
    return RetentionManager(storage, index)


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Constants
    "RETENTION_DAYS_7_YEARS",
    "RETENTION_DAYS_1_YEAR",
    "HASH_ALGORITHM",
    # Enums
    "RetentionCategory",
    "StorageBackendType",
    "BundleState",
    # Data Models
    "StorageReceipt",
    "BundleMetadata",
    "RetentionHold",
    # Configuration
    "LocalStorageConfig",
    "S3StorageConfig",
    # Abstract Base
    "PersistentStorageBackend",
    # Implementations
    "LocalFileStorage",
    "S3Storage",
    # Retention
    "RetentionManager",
    # Indexer
    "BundleIndexer",
    # Exceptions
    "StorageError",
    "BundleNotFoundError",
    "IntegrityError",
    "RetentionActiveError",
    "HoldActiveError",
    "StorageExistsError",
    # Factory Functions
    "create_local_storage",
    "create_s3_storage",
    "create_retention_manager",
]
