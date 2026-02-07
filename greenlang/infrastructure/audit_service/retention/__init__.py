# -*- coding: utf-8 -*-
"""
Audit Retention Services - Centralized Audit Logging Service (SEC-005)

Provides retention policy management and archival services for audit data.
Implements tiered storage with automatic lifecycle management:
    - Hot tier (PostgreSQL): 30 days - Active queries
    - Warm tier (PostgreSQL compressed): 90 days - Historical queries
    - Cold tier (S3 Parquet): 365 days - Compliance queries
    - Archive tier (S3 Glacier): 7 years - Long-term compliance

Sub-modules:
    retention_service   - Retention policy management and tier classification
    archival_service    - S3/Glacier archival and restore operations

Security Compliance:
    - SOC 2 CC6.6 (Retention and Disposal)
    - ISO 27001 A.18.1.3 (Protection of Records)
    - PCI DSS 10.7 (Audit Trail Retention)
    - GDPR Article 17 (Right to Erasure - with compliance exemptions)

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

from greenlang.infrastructure.audit_service.retention.retention_service import (
    RetentionPolicyService,
    RetentionTier,
    RETENTION_TIERS,
    COMPLIANCE_RETENTION,
    DataClassification,
)

from greenlang.infrastructure.audit_service.retention.archival_service import (
    ArchivalService,
    ArchivalStatus,
    ArchiveMetadata,
)

__all__ = [
    # Retention Service
    "RetentionPolicyService",
    "RetentionTier",
    "RETENTION_TIERS",
    "COMPLIANCE_RETENTION",
    "DataClassification",
    # Archival Service
    "ArchivalService",
    "ArchivalStatus",
    "ArchiveMetadata",
]
