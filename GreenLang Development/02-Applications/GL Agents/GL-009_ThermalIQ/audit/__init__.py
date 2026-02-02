# -*- coding: utf-8 -*-
"""
GL-009 THERMALIQ - Audit and Compliance Module

Provides immutable audit trails and compliance packaging for thermal
efficiency calculations. Supports EPA/FDA 7-year retention requirements.

Components:
- EvidenceGenerator: Creates sealed evidence packs
- PersistentStorage: SQLite-based audit persistence with chain verification
"""

from .evidence_generator import (
    EvidenceType,
    SealStatus,
    ExportFormat,
    EvidenceRecord,
    EvidencePack,
    ThermalIQEvidenceGenerator,
)

from .persistent_storage import (
    StorageBackend,
    RetentionPolicy,
    IntegrityStatus,
    AuditRecord,
    IntegrityCheckResult,
    SQLiteAuditStorage,
    ThermalIQAuditPersistence,
    create_audit_persistence,
)

__all__ = [
    # Evidence Generator
    "EvidenceType",
    "SealStatus",
    "ExportFormat",
    "EvidenceRecord",
    "EvidencePack",
    "ThermalIQEvidenceGenerator",
    # Persistent Storage
    "StorageBackend",
    "RetentionPolicy",
    "IntegrityStatus",
    "AuditRecord",
    "IntegrityCheckResult",
    "SQLiteAuditStorage",
    "ThermalIQAuditPersistence",
    "create_audit_persistence",
]

__version__ = "1.0.0"
