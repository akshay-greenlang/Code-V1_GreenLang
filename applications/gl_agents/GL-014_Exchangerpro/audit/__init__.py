# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Audit Module

Comprehensive audit logging, evidence generation, chain of custody tracking,
and compliance reporting for heat exchanger operations.

This module provides complete traceability from raw sensor data to final
recommendations with immutable audit trails and regulatory compliance
for TEMA, ASME, ISO 50001, and EPA standards.

Features:
    - Immutable audit log entries with SHA-256 hash chains
    - Timestamp with microsecond precision (ISO 8601)
    - Complete provenance tracking for all calculations
    - Chain of custody from OPC-UA to CMMS work orders
    - TEMA compliance verification
    - Energy savings verification (ISO 50001)
    - CO2e accounting audit trail (EPA 40 CFR 98)
    - Evidence pack generation with Merkle tree verification
    - Support for 7-year retention per regulatory requirements

Standards:
    - TEMA (Tubular Exchanger Manufacturers Association)
    - ASME PTC 12.5 (Single Phase Heat Exchangers)
    - ISO 50001:2018 (Energy Management Systems)
    - EPA 40 CFR Part 98 (Mandatory GHG Reporting)
    - ISO 14064 (GHG Accounting and Verification)
    - 21 CFR Part 11 (Electronic Records and Signatures)
    - ISO 27001 (Information Security Management)
    - SOC 2 Type II (Service Organization Controls)

Example Usage:
    >>> from audit import AuditLogger, EvidenceGenerator, ChainOfCustody, ComplianceReporter

    # Audit Logging
    >>> audit = AuditLogger()
    >>> audit.log_calculation(
    ...     calculation_type=ComputationType.LMTD_CALCULATION,
    ...     exchanger_id="HEX-001",
    ...     inputs={"T_hot_in": 120, "T_hot_out": 80},
    ...     outputs={"LMTD": 35.6, "duty_kW": 1500},
    ...     algorithm_version="1.2.0",
    ...     duration_ms=15.5
    ... )

    # Evidence Generation
    >>> generator = EvidenceGenerator()
    >>> record = generator.create_evidence_record(
    ...     evidence_type=EvidenceType.HEAT_TRANSFER_CALCULATION,
    ...     data={"method": "LMTD"},
    ...     inputs=input_data,
    ...     outputs=output_data
    ... )
    >>> pack = generator.create_evidence_pack([record])
    >>> sealed = generator.seal_pack(pack)

    # Chain of Custody
    >>> custody = ChainOfCustody()
    >>> custody.start_chain(
    ...     source_type=DataSource.OPC_UA_TAG,
    ...     source_id="tag://server/hx001/temp_in",
    ...     exchanger_id="HEX-001"
    ... )
    >>> custody.add_ingestion_step(raw_data)
    >>> custody.add_computation_step(ComputationType.LMTD_CALCULATION, inputs, outputs)
    >>> chain = custody.finalize()

    # Compliance Reporting
    >>> reporter = ComplianceReporter()
    >>> report = reporter.generate_tema_compliance_report(
    ...     exchanger_id="HEX-001",
    ...     period_start=datetime(2024, 1, 1),
    ...     period_end=datetime(2024, 12, 31)
    ... )

Zero-Hallucination Principle:
    All audit records include SHA-256 hashes of inputs and outputs,
    ensuring complete traceability and reproducibility of calculations.
    No LLM-generated numeric values are used in compliance calculations.

Module Structure:
    - schemas.py: Core audit data models (AuditRecord, ProvenanceChain, ChangeRecord)
    - audit_logger.py: Immutable audit logging with hash chains
    - evidence_generator.py: Sealed evidence pack generation
    - chain_of_custody.py: Data lineage tracking
    - compliance_reporter.py: Regulatory compliance reports
"""

from .schemas import (
    # Core audit record types
    AuditRecord,
    ProvenanceChain,
    ChangeRecord,
    CustodyStep,

    # Enumerations
    ComputationType,
    AuditEventType,
    DataSource,
    ApprovalStatus,
    ChainVerificationStatus,
    SeverityLevel,
    ActorType,

    # Pydantic models
    AuditRecordModel,
    ChangeRecordModel,
    ProvenanceChainModel,
    AuditStatistics,
    ComplianceStatus,

    # Utility functions
    compute_sha256,
)

from .audit_logger import (
    # Main audit logger
    AuditLogger,
    AuditLogEntry,
    AuditEvent,
    AuditContext,
    AuditOutcome,

    # Storage backends
    AuditStorage,
    FileAuditStorage,

    # Convenience functions
    get_audit_logger,
    create_audit_logger,
)

from .evidence_generator import (
    # Evidence generator
    EvidenceGenerator,
    EvidenceRecord,
    EvidenceType,
    SealStatus,
    ExportFormat,

    # Supporting classes
    EvidencePackMetadata,
    SealedPackEnvelope,
    CalculationMethodology,
    ModelProvenance,
    RecommendationTrace,

    # Standard methodologies
    LMTD_METHODOLOGY,
    NTU_METHODOLOGY,
    FOULING_RESISTANCE_METHODOLOGY,
)

from .chain_of_custody import (
    # Chain of custody manager
    ChainOfCustody,
    CustodyChain,
    DetailedCustodyStep,
    CustodyStepType,
)

from .compliance_reporter import (
    # Compliance reporter
    ComplianceReporter,
    ComplianceReport,
    ComplianceFramework,
    ReportType,
    ReportStatus,

    # TEMA compliance
    TEMACheck,

    # Energy and emissions
    EnergySavingsRecord,
    CO2eAccountingRecord,
)


__all__ = [
    # ==========================================================================
    # SCHEMAS
    # ==========================================================================
    # Core audit record types
    "AuditRecord",
    "ProvenanceChain",
    "ChangeRecord",
    "CustodyStep",

    # Enumerations
    "ComputationType",
    "AuditEventType",
    "DataSource",
    "ApprovalStatus",
    "ChainVerificationStatus",
    "SeverityLevel",
    "ActorType",

    # Pydantic models
    "AuditRecordModel",
    "ChangeRecordModel",
    "ProvenanceChainModel",
    "AuditStatistics",
    "ComplianceStatus",

    # Utility functions
    "compute_sha256",

    # ==========================================================================
    # AUDIT LOGGER
    # ==========================================================================
    "AuditLogger",
    "AuditLogEntry",
    "AuditEvent",
    "AuditContext",
    "AuditOutcome",

    # Storage backends
    "AuditStorage",
    "FileAuditStorage",

    # Convenience functions
    "get_audit_logger",
    "create_audit_logger",

    # ==========================================================================
    # EVIDENCE GENERATOR
    # ==========================================================================
    "EvidenceGenerator",
    "EvidenceRecord",
    "EvidenceType",
    "SealStatus",
    "ExportFormat",

    # Supporting classes
    "EvidencePackMetadata",
    "SealedPackEnvelope",
    "CalculationMethodology",
    "ModelProvenance",
    "RecommendationTrace",

    # Standard methodologies
    "LMTD_METHODOLOGY",
    "NTU_METHODOLOGY",
    "FOULING_RESISTANCE_METHODOLOGY",

    # ==========================================================================
    # CHAIN OF CUSTODY
    # ==========================================================================
    "ChainOfCustody",
    "CustodyChain",
    "DetailedCustodyStep",
    "CustodyStepType",

    # ==========================================================================
    # COMPLIANCE REPORTER
    # ==========================================================================
    "ComplianceReporter",
    "ComplianceReport",
    "ComplianceFramework",
    "ReportType",
    "ReportStatus",

    # TEMA compliance
    "TEMACheck",

    # Energy and emissions
    "EnergySavingsRecord",
    "CO2eAccountingRecord",
]


# Module version
__version__ = "1.0.0"

# Agent information
__agent_id__ = "GL-014"
__agent_name__ = "EXCHANGERPRO"
