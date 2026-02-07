# -*- coding: utf-8 -*-
"""
PII Auto-Remediation Package - SEC-011 PII Detection/Redaction Enhancements

Automated remediation engine for detected PII. Provides configurable policies
for deletion, anonymization, archival, and notification with full audit trails
and GDPR compliance.

Features:
    - Policy-based remediation (DELETE, ANONYMIZE, ARCHIVE, NOTIFY_ONLY)
    - Multi-source support (PostgreSQL, S3, Redis, Loki, Elasticsearch)
    - GDPR-compliant deletion certificates
    - Configurable grace periods and approval workflows
    - Scheduled background processing
    - Prometheus metrics integration

Public API:
    - PIIRemediationEngine: Main engine for processing remediations
    - PIIRemediationJob: Scheduled job runner
    - RemediationPolicy: Policy configuration model
    - PIIRemediationItem: Item awaiting remediation
    - DeletionCertificate: GDPR deletion proof
    - RemediationAction: Available remediation actions

Example:
    >>> from greenlang.infrastructure.pii_service.remediation import (
    ...     PIIRemediationEngine,
    ...     PIIRemediationJob,
    ...     RemediationConfig,
    ...     RemediationAction,
    ... )
    >>> engine = PIIRemediationEngine(RemediationConfig())
    >>> await engine.initialize()
    >>> item = await engine.schedule_remediation(
    ...     pii_type=PIIType.EMAIL,
    ...     source_type=SourceType.POSTGRESQL,
    ...     source_location="users.email",
    ...     record_identifier="user-123",
    ...     tenant_id="tenant-acme"
    ... )
    >>> job = PIIRemediationJob(engine, interval_minutes=60)
    >>> await job.start()

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
Version: 1.0.0
"""

from __future__ import annotations

import logging

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Policies Module
# ---------------------------------------------------------------------------

from greenlang.infrastructure.pii_service.remediation.policies import (
    # Enums
    RemediationAction,
    RemediationStatus,
    SourceType,
    # Models
    RemediationPolicy,
    PIIRemediationItem,
    DeletionCertificate,
    RemediationResult,
    # Defaults
    DEFAULT_REMEDIATION_POLICIES,
    get_default_policy,
    get_all_default_policies,
)

# ---------------------------------------------------------------------------
# Engine Module
# ---------------------------------------------------------------------------

from greenlang.infrastructure.pii_service.remediation.engine import (
    RemediationConfig,
    PIIRemediationEngine,
    RemediationError,
    SourceConnectionError,
    RemediationExecutionError,
    get_remediation_engine,
    reset_remediation_engine,
)

# ---------------------------------------------------------------------------
# Jobs Module
# ---------------------------------------------------------------------------

from greenlang.infrastructure.pii_service.remediation.jobs import (
    JobConfig,
    JobStatus,
    PIIRemediationJob,
    run_remediation_cron,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Policies - Enums
    "RemediationAction",
    "RemediationStatus",
    "SourceType",
    # Policies - Models
    "RemediationPolicy",
    "PIIRemediationItem",
    "DeletionCertificate",
    "RemediationResult",
    # Policies - Defaults
    "DEFAULT_REMEDIATION_POLICIES",
    "get_default_policy",
    "get_all_default_policies",
    # Engine
    "RemediationConfig",
    "PIIRemediationEngine",
    "RemediationError",
    "SourceConnectionError",
    "RemediationExecutionError",
    "get_remediation_engine",
    "reset_remediation_engine",
    # Jobs
    "JobConfig",
    "JobStatus",
    "PIIRemediationJob",
    "run_remediation_cron",
]

logger.debug("PII remediation package loaded (version %s)", __version__)
