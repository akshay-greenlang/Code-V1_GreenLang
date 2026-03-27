# -*- coding: utf-8 -*-
"""
GL-EUDR-APP Backend Services - EU Deforestation Regulation Compliance Platform
==============================================================================

This package provides the core backend services for the GL-EUDR-APP
compliance platform. It integrates with three GreenLang agents:

- AGENT-DATA-005 (EUDR Traceability): ``greenlang.agents.data.eudr_traceability``
- AGENT-DATA-007 (Deforestation Satellite): ``greenlang.agents.data.deforestation_satellite``
- AGENT-EUDR-001 (Supply Chain Mapper): ``greenlang.agents.eudr.supply_chain_mapper``

Core Engines:
    - config:                       EUDRAppConfig and enumerations
    - models:                       Pydantic domain models
    - pipeline_orchestrator:        5-stage compliance pipeline
    - supplier_intake_engine:       Supplier CRUD and ERP normalization
    - document_verification_engine: Document upload and EUDR verification
    - dds_reporting_engine:         DDS generation and EU submission
    - risk_aggregator:              Multi-source risk scoring
    - supply_chain:                 Supply chain mapping facade (AGENT-EUDR-001)
    - setup:                        EUDRComplianceService facade

Example:
    >>> from services.setup import EUDRComplianceService
    >>> service = EUDRComplianceService()
    >>> metrics = service.get_dashboard_metrics()

Application: GL-EUDR-APP v1.0 + AGENT-EUDR-001 Integration
Author: GreenLang Platform Team
Date: March 2026
"""

__version__ = "1.1.0"
__app_id__ = "GL-EUDR-APP"
__app_name__ = "EU Deforestation Regulation Compliance Platform"

# ---------------------------------------------------------------------------
# Configuration & Enumerations
# ---------------------------------------------------------------------------
from services.config import (
    EUDRAppConfig,
    EUDRCommodity,
    RiskLevel,
    DDSStatus,
    PipelineStage,
    PipelineStatus,
    DocumentType,
    VerificationStatus,
    ComplianceStatus,
    ProcurementStatus,
    SatelliteAssessmentStatus,
)

# ---------------------------------------------------------------------------
# Core Engines
# ---------------------------------------------------------------------------
from services.pipeline_orchestrator import PipelineOrchestrator
from services.supplier_intake_engine import SupplierIntakeEngine
from services.document_verification_engine import DocumentVerificationEngine
from services.dds_reporting_engine import DDSReportingEngine
from services.risk_aggregator import RiskAggregator

# ---------------------------------------------------------------------------
# Supply Chain Mapping (AGENT-EUDR-001)
# ---------------------------------------------------------------------------
from services.supply_chain import (
    SupplyChainAppService,
    SupplyChainError,
    configure_supply_chain_service,
    get_supply_chain_service,
)

# ---------------------------------------------------------------------------
# Service Facade
# ---------------------------------------------------------------------------
from services.setup import (
    EUDRComplianceService,
    configure_eudr_app,
    startup_eudr_app,
    shutdown_eudr_app,
    get_eudr_service,
)

__all__ = [
    # Version
    "__version__",
    "__app_id__",
    "__app_name__",
    # Configuration
    "EUDRAppConfig",
    # Enumerations
    "EUDRCommodity",
    "RiskLevel",
    "DDSStatus",
    "PipelineStage",
    "PipelineStatus",
    "DocumentType",
    "VerificationStatus",
    "ComplianceStatus",
    "ProcurementStatus",
    "SatelliteAssessmentStatus",
    # Engines
    "PipelineOrchestrator",
    "SupplierIntakeEngine",
    "DocumentVerificationEngine",
    "DDSReportingEngine",
    "RiskAggregator",
    # Supply Chain Mapping (AGENT-EUDR-001)
    "SupplyChainAppService",
    "SupplyChainError",
    "configure_supply_chain_service",
    "get_supply_chain_service",
    # Facade
    "EUDRComplianceService",
    "configure_eudr_app",
    "startup_eudr_app",
    "shutdown_eudr_app",
    "get_eudr_service",
]
