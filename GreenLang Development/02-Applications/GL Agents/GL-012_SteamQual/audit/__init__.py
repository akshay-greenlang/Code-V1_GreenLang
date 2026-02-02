"""
GL-012 SteamQual Audit Module

This module provides comprehensive audit, provenance, and compliance
capabilities for the SteamQualityController as specified in the GreenLang
framework requirements and the SteamQual Playbook data governance requirements.

Key Components:
    - ProvenanceTracker: SHA-256 provenance tracking for all calculations
    - ComplianceReporter: Steam quality KPI reporting and compliance documentation
    - CalculationTracer: Step-by-step calculation audit trails
    - EvidencePackager: Quality event evidence packaging for post-mortem analysis

Features:
    - Deterministic calculations with SHA-256 provenance hashing
    - Full audit trail for regulatory compliance
    - Steam quality KPI tracking (dryness, superheat, pressure)
    - Quality event timeline documentation
    - Evidence packaging for quality excursion analysis
    - Model version tracking for ML-based estimators
    - Data lineage from sensors to control actions

Reference Standards:
    - ASME PTC 19.11 (Steam and Water Sampling)
    - ASME B31.1 (Power Piping)
    - ISO 50001:2018 (Energy Management)
    - ISO 9001:2015 (Quality Management)
    - GreenLang Zero-Hallucination Standard

Steam Quality Metrics Tracked:
    - Steam dryness fraction (0.0 - 1.0)
    - Superheat temperature (degrees above saturation)
    - Pressure stability (PSI or bar)
    - Enthalpy calculations (BTU/lb or kJ/kg)
    - Control loop performance (IAE, ISE, settling time)

Example:
    >>> from audit import (
    ...     ProvenanceTracker,
    ...     ComplianceReporter,
    ...     CalculationTracer,
    ...     EvidencePackager,
    ... )
    >>>
    >>> # Initialize components
    >>> provenance = ProvenanceTracker()
    >>> reporter = ComplianceReporter(provenance_tracker=provenance)
    >>> tracer = CalculationTracer(provenance)
    >>> evidence = EvidencePackager(storage_path="/audit/evidence")
    >>>
    >>> # Track a steam quality calculation
    >>> ctx = tracer.start_trace("dryness_calculation")
    >>> tracer.record_step(ctx, "sensor_validation", inputs, outputs, formula)
    >>> trace = tracer.end_trace(ctx)
    >>>
    >>> # Generate quality event evidence
    >>> pack = evidence.create_quality_event_evidence(event_data, trace)
    >>> evidence.bundle_evidence_pack(pack, output_path)

Author: GreenLang Steam Quality Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__all__ = [
    # Provenance Tracking
    "ProvenanceTracker",
    "ProvenanceRecord",
    "VerificationResult",
    "HashAlgorithm",
    "ProvenanceChain",
    "LineageNode",
    "ModelVersionRecord",
    "DataLineageGraph",
    # Compliance Reporting
    "ComplianceReporter",
    "QualityKPIReport",
    "ControlPerformanceReport",
    "SteamQualityReport",
    "ExportedReport",
    "ReportFormat",
    "ReportingPeriod",
    "QualityMetric",
    "ControlLoopKPI",
    # Calculation Tracing
    "CalculationTracer",
    "TraceContext",
    "CalculationTrace",
    "TraceStep",
    "ConstraintCheckResult",
    "AuditableTrace",
    "FormulaVersion",
    "TraceStatus",
    "ConstraintType",
    # Evidence Pack
    "EvidencePackager",
    "QualityEventEvidence",
    "ControlActionEvidence",
    "SensorDataEvidence",
    "EvidencePack",
    "TimelineEvent",
    "ContributingFactor",
    "PlotData",
]

# Import all public components
from .provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    VerificationResult,
    HashAlgorithm,
    ProvenanceChain,
    LineageNode,
    ModelVersionRecord,
    DataLineageGraph,
)

from .compliance_reporter import (
    ComplianceReporter,
    QualityKPIReport,
    ControlPerformanceReport,
    SteamQualityReport,
    ExportedReport,
    ReportFormat,
    ReportingPeriod,
    QualityMetric,
    ControlLoopKPI,
)

from .calculation_trace import (
    CalculationTracer,
    TraceContext,
    CalculationTrace,
    TraceStep,
    ConstraintCheckResult,
    AuditableTrace,
    FormulaVersion,
    TraceStatus,
    ConstraintType,
)

from .evidence_pack import (
    EvidencePackager,
    QualityEventEvidence,
    ControlActionEvidence,
    SensorDataEvidence,
    EvidencePack,
    TimelineEvent,
    ContributingFactor,
    PlotData,
)
