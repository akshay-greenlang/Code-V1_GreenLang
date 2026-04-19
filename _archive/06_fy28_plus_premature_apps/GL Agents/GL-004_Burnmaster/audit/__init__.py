"""
BURNMASTER Audit Module - Complete audit trail and compliance for GL-004.

This module provides comprehensive audit capabilities for the BURNMASTER
combustion optimization system, including:

- AuditLogger: Immutable audit logging with cryptographic integrity
- ProvenanceTracker: Complete data lineage and traceability
- CalculationTrace: Deterministic calculation verification
- EvidencePackager: Cryptographically sealed evidence packages
- ComplianceReporter: Regulatory compliance reporting
- AuditRetentionManager: Data retention and lifecycle management

All audit records are immutable once created, include cryptographic hashes
for integrity verification, and support regulatory compliance requirements.

Example:
    >>> from audit import (
    ...     AuditLogger, AuditLoggerConfig,
    ...     ProvenanceTracker, ProvenanceTrackerConfig,
    ...     ComplianceReporter, ComplianceReporterConfig
    ... )
    >>>
    >>> # Initialize audit logger
    >>> logger_config = AuditLoggerConfig()
    >>> audit_logger = AuditLogger(logger_config)
    >>>
    >>> # Log a recommendation
    >>> record = audit_logger.log_recommendation(recommendation, context)
    >>>
    >>> # Generate compliance report
    >>> reporter_config = ComplianceReporterConfig(
    ...     facility_id="FAC-001",
    ...     facility_name="Plant A"
    ... )
    >>> reporter = ComplianceReporter(reporter_config)
    >>> report = reporter.generate_emissions_report(period)
"""

# =============================================================================
# Audit Logger
# =============================================================================
from .audit_logger import (
    # Main class
    AuditLogger,
    AuditLoggerConfig,

    # Enums
    AuditEventType,
    AuditSeverity,
    SafetyEventType,
    OperatorActionType,

    # Input models
    Recommendation,
    SetpointWrite,
    ModeTransition,
    SafetyEvent,
    OperatorAction,
    AuditContext,
    AuditFilters,

    # Output models
    AuditRecord,
)

# =============================================================================
# Provenance Tracker
# =============================================================================
from .provenance import (
    # Main class
    ProvenanceTracker,
    ProvenanceTrackerConfig,

    # Input models
    CombustionData,
    ConstraintSet,
    RecommendationInput,

    # Output models
    DataSnapshot,
    ModelVersion,
    CodeVersion,
    ConstraintSnapshot,
    ProvenanceLink,
    ValidationResult as ProvenanceValidationResult,
)

# =============================================================================
# Calculation Trace
# =============================================================================
from .calculation_trace import (
    # Main class
    CalculationTrace,
    CalculationTraceConfig,

    # Enums
    CalculationType,
    DeterminismStatus,

    # Models
    OptimizationResult,
    CalcTrace,
    OptTrace,
    InferenceTrace,
    ReplayResult,
    DeterminismCheck,

    # Decorator
    register_calculation,
)

# =============================================================================
# Evidence Packager
# =============================================================================
from .evidence_pack import (
    # Main class
    EvidencePackager,
    EvidencePackagerConfig,

    # Enums
    EventType,
    SealStatus,
    ExportFormat as EvidenceExportFormat,

    # Input models
    AuditableEvent,
    DataSnapshotEvidence,
    ModelVersionEvidence,
    CalculationEvidence,
    RecommendationEvidence,
    OutcomeEvidence,

    # Output models
    EvidencePack,
    SealedPack,
    VerificationResult as EvidenceVerificationResult,
)

# =============================================================================
# Compliance Reporter
# =============================================================================
from .compliance_reporter import (
    # Main class
    ComplianceReporter,
    ComplianceReporterConfig,

    # Enums
    ReportType,
    ReportStatus,
    ExportFormat as ReportExportFormat,
    ComplianceStatus as ReportComplianceStatus,

    # Input models
    DateRange,
    EmissionsDataPoint,
    PermitLimit,
    OptimizationEvent,
    SafetyIncident,

    # Output models
    EmissionsSummary,
    ComplianceReport,
    EmissionsReport,
    PermitReport,
    OptAuditReport,
    SafetyReport,
)

# =============================================================================
# Retention Manager
# =============================================================================
from .retention_manager import (
    # Main class
    AuditRetentionManager,
    RetentionManagerConfig,

    # Enums
    RecordCategory,
    RetentionAction,
    ComplianceStatus as RetentionComplianceStatus,
    ArchiveStatus,

    # Policy models
    RetentionRule,
    RetentionPolicy,

    # Result models
    ArchiveRecord,
    ArchiveResult,
    RestoreResult,
    PurgeResult,
    RetentionComplianceStatus as RetentionComplianceStatusReport,
)


# =============================================================================
# Module version and metadata
# =============================================================================
__version__ = "1.0.0"
__author__ = "GreenLang Team"
__description__ = "BURNMASTER Audit Module for GL-004"

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__description__",

    # Audit Logger
    "AuditLogger",
    "AuditLoggerConfig",
    "AuditEventType",
    "AuditSeverity",
    "SafetyEventType",
    "OperatorActionType",
    "Recommendation",
    "SetpointWrite",
    "ModeTransition",
    "SafetyEvent",
    "OperatorAction",
    "AuditContext",
    "AuditFilters",
    "AuditRecord",

    # Provenance Tracker
    "ProvenanceTracker",
    "ProvenanceTrackerConfig",
    "CombustionData",
    "ConstraintSet",
    "RecommendationInput",
    "DataSnapshot",
    "ModelVersion",
    "CodeVersion",
    "ConstraintSnapshot",
    "ProvenanceLink",
    "ProvenanceValidationResult",

    # Calculation Trace
    "CalculationTrace",
    "CalculationTraceConfig",
    "CalculationType",
    "DeterminismStatus",
    "OptimizationResult",
    "CalcTrace",
    "OptTrace",
    "InferenceTrace",
    "ReplayResult",
    "DeterminismCheck",
    "register_calculation",

    # Evidence Packager
    "EvidencePackager",
    "EvidencePackagerConfig",
    "EventType",
    "SealStatus",
    "EvidenceExportFormat",
    "AuditableEvent",
    "DataSnapshotEvidence",
    "ModelVersionEvidence",
    "CalculationEvidence",
    "RecommendationEvidence",
    "OutcomeEvidence",
    "EvidencePack",
    "SealedPack",
    "EvidenceVerificationResult",

    # Compliance Reporter
    "ComplianceReporter",
    "ComplianceReporterConfig",
    "ReportType",
    "ReportStatus",
    "ReportExportFormat",
    "ReportComplianceStatus",
    "DateRange",
    "EmissionsDataPoint",
    "PermitLimit",
    "OptimizationEvent",
    "SafetyIncident",
    "EmissionsSummary",
    "ComplianceReport",
    "EmissionsReport",
    "PermitReport",
    "OptAuditReport",
    "SafetyReport",

    # Retention Manager
    "AuditRetentionManager",
    "RetentionManagerConfig",
    "RecordCategory",
    "RetentionAction",
    "RetentionComplianceStatus",
    "ArchiveStatus",
    "RetentionRule",
    "RetentionPolicy",
    "ArchiveRecord",
    "ArchiveResult",
    "RestoreResult",
    "PurgeResult",
    "RetentionComplianceStatusReport",
]
