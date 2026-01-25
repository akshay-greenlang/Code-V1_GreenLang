"""
GL-006 HEATRECLAIM - Audit Module

Comprehensive audit logging for regulatory compliance and traceability.
Supports 6 event types with SHA-256 integrity verification.

Features:
- 6 audit event types (design, optimization, safety, config, user, system)
- SHA-256 hash chain for tamper detection
- Calculation event logging with full provenance
- Correlation tracking for distributed tracing
- Multiple storage backends (file, database)

Standards:
- ISO 27001: Information security management
- SOC 2 Type II: Service organization controls
- 21 CFR Part 11: Electronic records and signatures

Example:
    >>> from audit import AuditLogger, CalculationEventLogger
    >>> audit = AuditLogger()
    >>> calc_logger = CalculationEventLogger(audit)
    >>> with calc_logger.track_calculation("pinch_analysis") as tracker:
    ...     tracker.set_inputs({"streams": 10})
    ...     result = perform_analysis()
    ...     tracker.set_outputs(result)
"""

from .audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditOutcome,
    AuditContext,
    AuditRecord,
    CalculationEventLogger,
    CalculationAuditEvent,
    CalculationTracker,
    get_audit_logger,
    get_calculation_logger,
)
from .audit_storage import (
    AuditStorage,
    FileAuditStorage,
    DatabaseAuditStorage,
    AuditQuery,
    AuditQueryResult,
)

__all__ = [
    # Core audit logger
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "AuditOutcome",
    "AuditContext",
    "AuditRecord",
    # Calculation logging
    "CalculationEventLogger",
    "CalculationAuditEvent",
    "CalculationTracker",
    # Storage
    "AuditStorage",
    "FileAuditStorage",
    "DatabaseAuditStorage",
    "AuditQuery",
    "AuditQueryResult",
    # Convenience functions
    "get_audit_logger",
    "get_calculation_logger",
]
