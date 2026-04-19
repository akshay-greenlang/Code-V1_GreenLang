"""
GL-003 UnifiedSteam Audit Module

This module provides comprehensive audit, provenance, and compliance
capabilities for the SteamSystemOptimizer as specified in the GreenLang
framework requirements.

Key Components:
    - AuditLogger: Steam system event logging with hash chaining
    - ProvenanceTracker: SHA-256 provenance tracking for calculations
    - EvidencePackager: M&V evidence packaging for IPMVP compliance
    - CalculationTracer: Step-by-step calculation audit trails
    - ComplianceReporter: GHG Protocol and energy reporting

Features:
    - Deterministic calculations with SHA-256 provenance hashing
    - Full audit trail for regulatory compliance
    - Evidence packaging for M&V (Measurement & Verification)
    - IPMVP-style M&V documentation support
    - GHG Protocol aligned CO2e reporting
    - Immutable audit entries with timestamps
    - User attribution for all actions

Reference Standards:
    - IPMVP (International Performance Measurement and Verification Protocol)
    - GHG Protocol (Corporate Accounting and Reporting Standard)
    - ISO 50001:2018 (Energy Management)
    - EPA 40 CFR 98 (GHG Reporting)
    - ISO 14064 (GHG Inventories)

Example:
    >>> from audit import (
    ...     AuditLogger,
    ...     ProvenanceTracker,
    ...     EvidencePackager,
    ...     CalculationTracer,
    ...     ComplianceReporter,
    ... )
    >>>
    >>> # Initialize components
    >>> logger = AuditLogger(retention_years=7)
    >>> provenance = ProvenanceTracker()
    >>> evidence = EvidencePackager(storage_path="/audit/evidence")
    >>> tracer = CalculationTracer(provenance)
    >>> reporter = ComplianceReporter(logger, provenance)
    >>>
    >>> # Track a calculation
    >>> ctx = tracer.start_trace("steam_balance")
    >>> tracer.record_step(ctx, "input_validation", inputs, outputs, formula)
    >>> trace = tracer.end_trace(ctx)
    >>>
    >>> # Generate M&V evidence
    >>> savings = evidence.create_savings_evidence(baseline, post, factors)
    >>> report = evidence.package_mv_report(savings, methodology_notes)

Author: GreenLang Steam Systems Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__all__ = [
    # Audit Logger
    "AuditLogger",
    "AuditEntry",
    "AuditEventType",
    "AuditStorageBackend",
    "InMemoryAuditStorage",
    "FileAuditStorage",
    "TimeWindow",
    "AuditFilter",
    # Provenance
    "ProvenanceTracker",
    "ProvenanceRecord",
    "VerificationResult",
    "HashAlgorithm",
    "ProvenanceChain",
    "LineageNode",
    # Evidence Pack
    "EvidencePackager",
    "BaselineEvidence",
    "PostEvidence",
    "SavingsEvidence",
    "MVReport",
    "SignedEvidence",
    "MVMethodology",
    "NormalizationFactor",
    # Calculation Trace
    "CalculationTracer",
    "TraceContext",
    "CalculationTrace",
    "TraceStep",
    "ConstraintCheckResult",
    "AuditableTrace",
    "FormulaVersion",
    # Compliance Reporter
    "ComplianceReporter",
    "GHGReport",
    "EnergyReport",
    "VerificationReport",
    "ExportedReport",
    "EmissionFactor",
    "GHGScope",
    "ReportFormat",
]

# Import all public components
from .audit_logger import (
    AuditLogger,
    AuditEntry,
    AuditEventType,
    AuditStorageBackend,
    InMemoryAuditStorage,
    FileAuditStorage,
    TimeWindow,
    AuditFilter,
)

from .provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    VerificationResult,
    HashAlgorithm,
    ProvenanceChain,
    LineageNode,
)

from .evidence_pack import (
    EvidencePackager,
    BaselineEvidence,
    PostEvidence,
    SavingsEvidence,
    MVReport,
    SignedEvidence,
    MVMethodology,
    NormalizationFactor,
)

from .calculation_trace import (
    CalculationTracer,
    TraceContext,
    CalculationTrace,
    TraceStep,
    ConstraintCheckResult,
    AuditableTrace,
    FormulaVersion,
)

from .compliance_reporter import (
    ComplianceReporter,
    GHGReport,
    EnergyReport,
    VerificationReport,
    ExportedReport,
    EmissionFactor,
    GHGScope,
    ReportFormat,
)
