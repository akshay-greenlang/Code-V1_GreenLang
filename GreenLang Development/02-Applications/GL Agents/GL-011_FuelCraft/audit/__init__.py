# -*- coding: utf-8 -*-
"""
GL-011 FuelCraft Audit Module.

This module provides comprehensive audit and provenance functionality for
fuel blending optimization including:

- Immutable run bundles with content-addressed storage
- Complete provenance tracking (inputs, conversions, models, outputs)
- 7-year retention compliance for regulatory requirements
- Structured audit logging with RBAC integration
- Bundle replay validation for reproducibility

All audit modules implement:
- SHA-256 hashing for integrity verification
- Immutable storage (append-only)
- Complete traceability from inputs to outputs
- Regulatory retention compliance (7+ years)

Reference Standards:
- EPA Record Retention Requirements
- SOX Compliance (7-year retention)
- MARPOL Annex VI Documentation

Example:
    >>> from audit import RunBundleBuilder, ProvenanceTracker, AuditLogger
    >>>
    >>> # Create a run bundle
    >>> builder = RunBundleBuilder(run_id="RUN-001", agent_version="1.0.0")
    >>> builder.add_input_snapshot("inventory", inventory_data)
    >>> builder.add_output("blend_plan", optimization_result)
    >>> bundle = builder.seal()
    >>>
    >>> # Track provenance
    >>> tracker = ProvenanceTracker(run_id="RUN-001")
    >>> snapshot = tracker.capture_input("prices", prices, DataSourceType.ERP, "SAP")
    >>> tracker.record_conversion(100, "gallons", 378.5, "liters", 3.785, ...)
    >>>
    >>> # Audit logging
    >>> logger = AuditLogger()
    >>> logger.log_run_started(run_id="RUN-001", user_id="operator1")
    >>> logger.log_optimization_completed(run_id="RUN-001", objective_value=12500.0)

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from .run_bundle import (
    BundleStatus,
    ComponentType,
    BundleComponent,
    BundleManifest,
    ReplayValidationResult,
    RunBundleBuilder,
    ImmutableStorage,
    BundleReplayValidator,
)

from .provenance import (
    DataSourceType,
    VersionType,
    InputSnapshot,
    ConversionRecord,
    ConversionAuditTrail,
    ModelVersionRecord,
    SolverConfigRecord,
    OutputLineage,
    ProvenanceTracker,
)

from .audit_logger import (
    AuditEventType,
    AuditSeverity,
    RetentionCategory,
    AuditEvent,
    RetentionPolicy,
    RBACConfig,
    AuditExportResult,
    AuditLogger,
)

from .persistent_storage import (
    # Constants
    RETENTION_DAYS_7_YEARS,
    RETENTION_DAYS_1_YEAR,
    HASH_ALGORITHM,
    # Enums
    RetentionCategory as StorageRetentionCategory,
    StorageBackendType,
    BundleState,
    # Data Models
    StorageReceipt,
    BundleMetadata,
    RetentionHold,
    # Configuration
    LocalStorageConfig,
    S3StorageConfig,
    # Abstract Base
    PersistentStorageBackend,
    # Implementations
    LocalFileStorage,
    S3Storage,
    # Retention
    RetentionManager,
    # Indexer
    BundleIndexer,
    # Exceptions
    StorageError,
    BundleNotFoundError,
    IntegrityError,
    RetentionActiveError,
    HoldActiveError,
    StorageExistsError,
    # Factory Functions
    create_local_storage,
    create_s3_storage,
    create_retention_manager,
)

__all__ = [
    # Run Bundle
    "BundleStatus",
    "ComponentType",
    "BundleComponent",
    "BundleManifest",
    "ReplayValidationResult",
    "RunBundleBuilder",
    "ImmutableStorage",
    "BundleReplayValidator",
    # Provenance
    "DataSourceType",
    "VersionType",
    "InputSnapshot",
    "ConversionRecord",
    "ConversionAuditTrail",
    "ModelVersionRecord",
    "SolverConfigRecord",
    "OutputLineage",
    "ProvenanceTracker",
    # Audit Logger
    "AuditEventType",
    "AuditSeverity",
    "RetentionCategory",
    "AuditEvent",
    "RetentionPolicy",
    "RBACConfig",
    "AuditExportResult",
    "AuditLogger",
    # Persistent Storage - Constants
    "RETENTION_DAYS_7_YEARS",
    "RETENTION_DAYS_1_YEAR",
    "HASH_ALGORITHM",
    # Persistent Storage - Enums
    "StorageRetentionCategory",
    "StorageBackendType",
    "BundleState",
    # Persistent Storage - Data Models
    "StorageReceipt",
    "BundleMetadata",
    "RetentionHold",
    # Persistent Storage - Configuration
    "LocalStorageConfig",
    "S3StorageConfig",
    # Persistent Storage - Abstract Base
    "PersistentStorageBackend",
    # Persistent Storage - Implementations
    "LocalFileStorage",
    "S3Storage",
    # Persistent Storage - Retention
    "RetentionManager",
    # Persistent Storage - Indexer
    "BundleIndexer",
    # Persistent Storage - Exceptions
    "StorageError",
    "BundleNotFoundError",
    "IntegrityError",
    "RetentionActiveError",
    "HoldActiveError",
    "StorageExistsError",
    # Persistent Storage - Factory Functions
    "create_local_storage",
    "create_s3_storage",
    "create_retention_manager",
]

__version__ = "1.0.0"
