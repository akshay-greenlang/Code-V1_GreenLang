# -*- coding: utf-8 -*-
"""
GL-DATA-EUDR-001: GreenLang EUDR Traceability Connector Service SDK
====================================================================

This package provides EU Deforestation Regulation (EUDR) traceability
connectivity, geolocation plot registration, chain of custody tracking,
due diligence statement management, risk assessment, commodity
classification, compliance verification, and EU Information System
submission for the GreenLang framework. It supports:

- Geolocation plot registration with polygon validation (WGS84)
- Chain of custody tracking (identity preserved, segregated, mass balance)
- Due diligence statement (DDS) lifecycle management
- Country-level and commodity-level risk assessment scoring
- Commodity classification with CN/HS code mapping
- Compliance verification against EUDR articles
- EU Information System submission and status tracking
- Batch processing with parallel workers
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_EUDR_TRACEABILITY_ env prefix

Key Components:
    - config: EUDRTraceabilityConfig with GL_EUDR_TRACEABILITY_ env prefix
    - models: Pydantic v2 models for all data structures
    - plot_registry: Geolocation plot registration and validation engine
    - chain_of_custody: Chain of custody transfer tracking engine
    - due_diligence: Due diligence statement lifecycle engine
    - risk_assessment: Country, commodity, and supplier risk scoring engine
    - commodity_classifier: EUDR commodity and CN/HS code classification engine
    - compliance_verifier: EUDR article compliance verification engine
    - eu_system_connector: EU Information System submission engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: EUDRTraceabilityService facade

Example:
    >>> from greenlang.eudr_traceability import EUDRTraceabilityService
    >>> service = EUDRTraceabilityService()
    >>> result = service.register_plot(request)
    >>> print(result.compliance_status)
    compliant

Agent ID: GL-DATA-EUDR-001
Agent Name: EUDR Traceability Connector Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-EUDR-001"
__agent_name__ = "EUDR Traceability Connector Agent"

# SDK availability flag
EUDR_TRACEABILITY_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.eudr_traceability.config import (
    EUDRTraceabilityConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, data models, request models)
# ---------------------------------------------------------------------------
from greenlang.eudr_traceability.models import (
    # Enumerations
    EUDRCommodity,
    RiskLevel,
    ComplianceStatus,
    LandUseType,
    CustodyModel,
    DDSStatus,
    DDSType,
    SubmissionStatus,
    # Core data models
    GeolocationData,
    PlotRecord,
    CustodyTransfer,
    BatchRecord,
    RiskScore,
    CommodityClassification,
    SupplierDeclaration,
    DueDiligenceStatement,
    ComplianceCheckResult,
    EUSubmissionRecord,
    EUDRStatistics,
    # Request models
    RegisterPlotRequest,
    RecordTransferRequest,
    GenerateDDSRequest,
    AssessRiskRequest,
    ClassifyCommodityRequest,
    RegisterDeclarationRequest,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.eudr_traceability.plot_registry import PlotRegistryEngine
from greenlang.eudr_traceability.chain_of_custody import ChainOfCustodyEngine
from greenlang.eudr_traceability.due_diligence import DueDiligenceEngine
from greenlang.eudr_traceability.risk_assessment import RiskAssessmentEngine
from greenlang.eudr_traceability.commodity_classifier import CommodityClassifier
from greenlang.eudr_traceability.compliance_verifier import ComplianceVerifier
from greenlang.eudr_traceability.eu_system_connector import EUSystemConnector
from greenlang.eudr_traceability.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.eudr_traceability.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    eudr_plots_registered_total,
    eudr_custody_transfers_total,
    eudr_dds_generated_total,
    eudr_risk_assessments_total,
    eudr_commodity_classifications_total,
    eudr_compliance_checks_total,
    eudr_eu_submissions_total,
    eudr_processing_duration_seconds,
    eudr_processing_errors_total,
    eudr_supplier_declarations_total,
    eudr_active_plots,
    eudr_pending_submissions,
    # Helper functions
    record_plot_registered,
    record_custody_transfer,
    record_dds_generated,
    record_risk_assessment,
    record_commodity_classification,
    record_compliance_check,
    record_eu_submission,
    record_supplier_declaration,
    record_processing_error,
    record_batch_operation,
    update_active_plots,
    update_pending_submissions,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.eudr_traceability.setup import (
    EUDRTraceabilityService,
    configure_eudr_traceability,
    get_eudr_traceability,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "EUDR_TRACEABILITY_SDK_AVAILABLE",
    # Configuration
    "EUDRTraceabilityConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Enumerations
    "EUDRCommodity",
    "RiskLevel",
    "ComplianceStatus",
    "LandUseType",
    "CustodyModel",
    "DDSStatus",
    "DDSType",
    "SubmissionStatus",
    # Core data models
    "GeolocationData",
    "PlotRecord",
    "CustodyTransfer",
    "BatchRecord",
    "RiskScore",
    "CommodityClassification",
    "SupplierDeclaration",
    "DueDiligenceStatement",
    "ComplianceCheckResult",
    "EUSubmissionRecord",
    "EUDRStatistics",
    # Request models
    "RegisterPlotRequest",
    "RecordTransferRequest",
    "GenerateDDSRequest",
    "AssessRiskRequest",
    "ClassifyCommodityRequest",
    "RegisterDeclarationRequest",
    # Core engines
    "PlotRegistryEngine",
    "ChainOfCustodyEngine",
    "DueDiligenceEngine",
    "RiskAssessmentEngine",
    "CommodityClassifier",
    "ComplianceVerifier",
    "EUSystemConnector",
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "eudr_plots_registered_total",
    "eudr_custody_transfers_total",
    "eudr_dds_generated_total",
    "eudr_risk_assessments_total",
    "eudr_commodity_classifications_total",
    "eudr_compliance_checks_total",
    "eudr_eu_submissions_total",
    "eudr_processing_duration_seconds",
    "eudr_processing_errors_total",
    "eudr_supplier_declarations_total",
    "eudr_active_plots",
    "eudr_pending_submissions",
    # Metric helper functions
    "record_plot_registered",
    "record_custody_transfer",
    "record_dds_generated",
    "record_risk_assessment",
    "record_commodity_classification",
    "record_compliance_check",
    "record_eu_submission",
    "record_supplier_declaration",
    "record_processing_error",
    "record_batch_operation",
    "update_active_plots",
    "update_pending_submissions",
    # Service setup facade
    "EUDRTraceabilityService",
    "configure_eudr_traceability",
    "get_eudr_traceability",
    "get_router",
]
