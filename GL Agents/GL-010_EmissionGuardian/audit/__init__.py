# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Audit and Lineage Module

Production-grade audit and data lineage tracking capabilities including:
- Immutable audit log entries with hash chains
- Complete data lineage tracking from source to output
- Regulatory compliance (EPA 40 CFR Part 75, SOX, SOC 2)
- Retention policy management
- Chain verification and tamper detection

Standards Compliance:
    - EPA 40 CFR Part 75 (3+ years data retention)
    - SOX (Sarbanes-Oxley) audit requirements
    - SOC 2 Type II compliance controls

Example:
    >>> from audit import AuditEntry, DataLineage, AuditAction
    >>> # Create audit entry
    >>> entry = AuditEntry(
    ...     entry_id="AUD-001",
    ...     action=AuditAction.CALCULATION,
    ...     resource_type=ResourceType.EMISSION_RECORD,
    ...     resource_id="EM-2024-001"
    ... )
    >>> # Track data lineage
    >>> lineage = DataLineage(
    ...     lineage_id="LIN-001",
    ...     data_id="DATA-001",
    ...     source_type="cems",
    ...     source_id="CEMS-001"
    ... )

Author: GreenLang GL-010 EmissionsGuardian
Version: 1.0.0
"""

from .schemas import (
    # Enums
    AuditAction,
    ResourceType,
    ActorType,
    TransformationOperation,
    ChainVerificationStatus,
    # Data Models
    TransformationStep,
    DataLineage,
    AuditEntry,
    AuditQueryFilter,
    AuditQuery,
    AuditStatistics,
    AuditReport,
    ChainVerificationResult,
    RetentionPolicy,
    RetentionStatus,
    ArchiveResult,
    RetentionResult,
)

__all__ = [
    # Enums
    "AuditAction",
    "ResourceType",
    "ActorType",
    "TransformationOperation",
    "ChainVerificationStatus",
    # Data Models
    "TransformationStep",
    "DataLineage",
    "AuditEntry",
    "AuditQueryFilter",
    "AuditQuery",
    "AuditStatistics",
    "AuditReport",
    "ChainVerificationResult",
    "RetentionPolicy",
    "RetentionStatus",
    "ArchiveResult",
    "RetentionResult",
]

__version__ = "1.0.0"
__author__ = "GreenLang GL-010 EmissionsGuardian"
