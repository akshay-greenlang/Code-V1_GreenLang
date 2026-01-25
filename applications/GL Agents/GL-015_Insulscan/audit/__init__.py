# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Audit Module

Comprehensive audit logging, evidence generation, chain of custody tracking,
and compliance reporting for insulation scanning and thermal assessment operations.

This module provides complete traceability from thermal scan data to final
recommendations with immutable audit trails and regulatory compliance
for ISO 50001, ASHRAE, and EPA standards.

Features:
    - Immutable audit log entries with SHA-256 hash chains
    - Timestamp with microsecond precision (ISO 8601)
    - Complete provenance tracking for all calculations
    - Chain of custody from thermal camera to CMMS work orders
    - ISO 50001 Energy Management compliance
    - ASHRAE 90.1/189.1 insulation standards compliance
    - ASTM C1060 (Thermographic Inspection) compliance
    - ASTM C680 (Heat Loss Calculations) compliance
    - CO2e accounting audit trail (EPA 40 CFR 98)
    - Evidence pack generation with Merkle tree verification
    - Support for 7-year retention per regulatory requirements
    - SIEM integration (Splunk, ELK)

Standards:
    - ISO 50001:2018 (Energy Management Systems)
    - ASHRAE 90.1 (Energy Standard for Buildings)
    - ASHRAE 189.1 (High-Performance Buildings)
    - ASTM C1060 (Thermographic Inspection of Insulation)
    - ASTM C680 (Heat Gain/Loss Calculation)
    - EPA 40 CFR Part 98 (Mandatory GHG Reporting)
    - ISO 14064 (GHG Accounting and Verification)
    - 21 CFR Part 11 (Electronic Records and Signatures)
    - ISO 27001 (Information Security Management)
    - SOC 2 Type II (Service Organization Controls)

Example Usage:
    >>> from audit import (
    ...     InsulationAuditLogger,
    ...     InsulationEvidenceGenerator,
    ...     InsulationChainOfCustody,
    ...     InsulationComplianceReporter,
    ... )

    # Audit Logging
    >>> logger = InsulationAuditLogger()
    >>> logger.log_thermal_scan(
    ...     asset_id="INSUL-001",
    ...     camera_id="CAM-FLIR-T640",
    ...     scan_result="PASS",
    ...     max_temp_c=45.2
    ... )
    >>> logger.log_calculation(
    ...     calculation_type=ComputationType.HEAT_LOSS,
    ...     asset_id="INSUL-001",
    ...     inputs={"surface_temp_c": 45},
    ...     outputs={"heat_loss_w_m2": 150.5},
    ...     algorithm_version="1.0.0"
    ... )

    # Evidence Generation
    >>> generator = InsulationEvidenceGenerator()
    >>> record = generator.create_evidence_record(
    ...     evidence_type=EvidenceType.HEAT_LOSS_CALCULATION,
    ...     data={"method": "ASTM_C680"},
    ...     inputs=input_data,
    ...     outputs=output_data
    ... )
    >>> pack = generator.create_evidence_pack([record])
    >>> sealed = generator.seal_pack(pack)

    # Chain of Custody
    >>> custody = InsulationChainOfCustody()
    >>> custody.start_chain(
    ...     source_type=DataSource.THERMAL_CAMERA,
    ...     source_id="CAM-FLIR-T640",
    ...     asset_id="INSUL-001"
    ... )
    >>> custody.add_scan_step(scan_data, camera_model="FLIR-T640")
    >>> custody.add_analysis_step(ComputationType.HEAT_LOSS, inputs, outputs)
    >>> custody.add_recommendation_step(recommendation)
    >>> chain = custody.finalize()

    # Compliance Reporting
    >>> reporter = InsulationComplianceReporter()
    >>> report = reporter.generate_iso_50001_report(
    ...     asset_ids=["INSUL-001"],
    ...     period_start=datetime(2024, 1, 1),
    ...     period_end=datetime(2024, 12, 31)
    ... )
    >>> reporter.add_ashrae_compliance(report, asset_data)
    >>> gap_analysis = reporter.analyze_compliance_gaps(report)
    >>> exported = reporter.export_report(report, format="pdf")

Zero-Hallucination Principle:
    All audit records include SHA-256 hashes of inputs and outputs,
    ensuring complete traceability and reproducibility of calculations.
    No LLM-generated numeric values are used in compliance calculations.

Module Structure:
    - schemas.py: Core audit data models (AuditEvent, EvidenceRecord, etc.)
    - audit_logger.py: Immutable audit logging with hash chains
    - evidence_generator.py: Sealed evidence pack generation
    - chain_of_custody.py: Data lineage tracking
    - compliance_reporter.py: Regulatory compliance reports
"""

from .schemas import (
    # Core audit schemas
    AuditEvent,
    EvidenceRecord,
    ChainOfCustodyEntry,
    ComplianceRecord,
    ProvenanceChain,
    AuditStatistics,

    # Insulation-specific schemas
    InsulationAssetRecord,
    ThermalScanRecord,

    # Enumerations
    ComputationType,
    AuditEventType,
    DataSource,
    ApprovalStatus,
    ChainVerificationStatus,
    SeverityLevel,
    ActorType,
    CustodyType,
    ComplianceStandard,
    ComplianceRequirementStatus,

    # Utility functions
    compute_sha256,
)

from .audit_logger import (
    # Main audit logger
    InsulationAuditLogger,
    AuditLogEntry,
    InsulationAuditEvent,
    AuditContext,
    AuditLogLevel,
    AuditOutcome,

    # Storage backends
    AuditStorage,
    FileAuditStorage,

    # SIEM integrations
    SIEMIntegration,
    SplunkIntegration,
    ElasticIntegration,

    # Convenience functions
    get_audit_logger,
    create_audit_logger,
)

from .evidence_generator import (
    # Evidence generator
    InsulationEvidenceGenerator,
    InsulationEvidenceRecord,
    EvidenceType,
    SealStatus,
    ExportFormat,

    # Supporting classes
    EvidencePackMetadata,
    SealedPackEnvelope,
    CalculationMethodology,
    ModelProvenance,

    # Standard methodologies
    HEAT_LOSS_METHODOLOGY,
    R_VALUE_METHODOLOGY,
    U_VALUE_METHODOLOGY,
)

from .chain_of_custody import (
    # Chain of custody manager
    InsulationChainOfCustody,
    CustodyChain,
    DetailedCustodyStep,
    CustodyStepType,

    # Blockchain integration
    BlockchainAnchor,
)

from .compliance_reporter import (
    # Compliance reporter
    InsulationComplianceReporter,
    InsulationComplianceReport,
    ComplianceFramework,
    ReportType,
    ReportStatus,
    RemediationPriority,

    # ISO 50001 compliance
    ISO50001Check,

    # ASHRAE compliance
    ASHRAECheck,

    # Energy and emissions
    EnergyPerformanceRecord,
    CO2eEmissionsRecord,

    # Gap analysis
    ComplianceGap,
)


__all__ = [
    # ==========================================================================
    # SCHEMAS
    # ==========================================================================
    # Core audit schemas
    "AuditEvent",
    "EvidenceRecord",
    "ChainOfCustodyEntry",
    "ComplianceRecord",
    "ProvenanceChain",
    "AuditStatistics",

    # Insulation-specific schemas
    "InsulationAssetRecord",
    "ThermalScanRecord",

    # Enumerations
    "ComputationType",
    "AuditEventType",
    "DataSource",
    "ApprovalStatus",
    "ChainVerificationStatus",
    "SeverityLevel",
    "ActorType",
    "CustodyType",
    "ComplianceStandard",
    "ComplianceRequirementStatus",

    # Utility functions
    "compute_sha256",

    # ==========================================================================
    # AUDIT LOGGER
    # ==========================================================================
    "InsulationAuditLogger",
    "AuditLogEntry",
    "InsulationAuditEvent",
    "AuditContext",
    "AuditLogLevel",
    "AuditOutcome",

    # Storage backends
    "AuditStorage",
    "FileAuditStorage",

    # SIEM integrations
    "SIEMIntegration",
    "SplunkIntegration",
    "ElasticIntegration",

    # Convenience functions
    "get_audit_logger",
    "create_audit_logger",

    # ==========================================================================
    # EVIDENCE GENERATOR
    # ==========================================================================
    "InsulationEvidenceGenerator",
    "InsulationEvidenceRecord",
    "EvidenceType",
    "SealStatus",
    "ExportFormat",

    # Supporting classes
    "EvidencePackMetadata",
    "SealedPackEnvelope",
    "CalculationMethodology",
    "ModelProvenance",

    # Standard methodologies
    "HEAT_LOSS_METHODOLOGY",
    "R_VALUE_METHODOLOGY",
    "U_VALUE_METHODOLOGY",

    # ==========================================================================
    # CHAIN OF CUSTODY
    # ==========================================================================
    "InsulationChainOfCustody",
    "CustodyChain",
    "DetailedCustodyStep",
    "CustodyStepType",

    # Blockchain integration
    "BlockchainAnchor",

    # ==========================================================================
    # COMPLIANCE REPORTER
    # ==========================================================================
    "InsulationComplianceReporter",
    "InsulationComplianceReport",
    "ComplianceFramework",
    "ReportType",
    "ReportStatus",
    "RemediationPriority",

    # ISO 50001 compliance
    "ISO50001Check",

    # ASHRAE compliance
    "ASHRAECheck",

    # Energy and emissions
    "EnergyPerformanceRecord",
    "CO2eEmissionsRecord",

    # Gap analysis
    "ComplianceGap",
]


# Module version
__version__ = "1.0.0"

# Agent information
__agent_id__ = "GL-015"
__agent_name__ = "INSULSCAN"
