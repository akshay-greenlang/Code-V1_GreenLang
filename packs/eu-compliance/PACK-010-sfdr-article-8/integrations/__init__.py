# -*- coding: utf-8 -*-
"""
PACK-010 SFDR Article 8 Pack - Integration Layer
==================================================

Phase 5 integration layer that provides 10-phase pipeline orchestration,
PACK-008 EU Taxonomy bridge, MRV emissions routing, investment screening,
portfolio data ingestion, EET import/export, regulatory monitoring,
data quality enforcement, 20-category health verification, and 8-step
guided setup for SFDR Article 8 financial products.

Components:
    - SFDRPackOrchestrator:        10-phase SFDR pipeline orchestrator
    - TaxonomyPackBridge:          PACK-008 EU Taxonomy alignment bridge
    - MRVEmissionsBridge:          MRV agent emissions to PAI indicators
    - InvestmentScreenerBridge:    SFDR classification and screening
    - PortfolioDataBridge:         Portfolio holdings, NAV, sector data
    - EETDataBridge:               European ESG Template import/export
    - RegulatoryTrackingBridge:    SFDR regulatory updates monitoring
    - DataQualityBridge:           PAI data quality enforcement
    - SFDRHealthCheck:             20-category system verification
    - SFDRSetupWizard:             8-step guided configuration wizard

Architecture:
    Portfolio Data --> SFDRPackOrchestrator --> PAI + Taxonomy + DNSH
                             |                          |
                             v                          v
    TaxonomyPackBridge --> PACK-008      MRVEmissionsBridge --> MRV Agents
                             |                          |
                             v                          v
    InvestmentScreenerBridge <-- Screening    EETDataBridge --> Distributors
                             |                          |
                             v                          v
    SFDRSetupWizard <-- Config        SFDRHealthCheck --> Readiness

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Version: 1.0.0
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-010"
__pack_name__ = "SFDR Article 8 Pack"

# ---------------------------------------------------------------------------
# Pack Orchestrator
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_010_sfdr_article_8.integrations.pack_orchestrator import (
    SFDRPackOrchestrator,
    SFDROrchestrationConfig,
    SFDRPipelinePhase,
    SFDRExecutionStatus,
    SFDRClassification as OrchestratorSFDRClassification,
    DisclosureType,
    PAICategory,
    PhaseResult as SFDRPhaseResult,
    PipelineResult,
    PipelineStatus,
    QualityGateStatus as SFDRQualityGateStatus,
)

# ---------------------------------------------------------------------------
# Taxonomy Pack Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_010_sfdr_article_8.integrations.taxonomy_pack_bridge import (
    TaxonomyPackBridge,
    TaxonomyBridgeConfig,
    TaxonomyObjective,
    AlignmentMethodology,
    AlignmentStatus,
    AlignmentResult,
    EligibilityResult,
)

# ---------------------------------------------------------------------------
# MRV Emissions Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_010_sfdr_article_8.integrations.mrv_emissions_bridge import (
    MRVEmissionsBridge,
    MRVEmissionsBridgeConfig,
    EmissionScope,
    PAIIndicator,
    InvesteeEmissions,
    PAIResult,
    PortfolioEmissions,
)

# ---------------------------------------------------------------------------
# Investment Screener Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_010_sfdr_article_8.integrations.investment_screener_bridge import (
    InvestmentScreenerBridge,
    InvestmentScreenerBridgeConfig,
    SFDRClassification,
    ScreeningVerdict,
    ExclusionCategory,
    ScreeningResult,
    ClassificationResult,
)

# ---------------------------------------------------------------------------
# Portfolio Data Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_010_sfdr_article_8.integrations.portfolio_data_bridge import (
    PortfolioDataBridge,
    PortfolioDataBridgeConfig,
    SectorClassification,
    DataFormat,
    DataCategory,
    ValidatedHolding,
    ImportResult,
    NAVEntry,
)

# ---------------------------------------------------------------------------
# EET Data Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_010_sfdr_article_8.integrations.eet_data_bridge import (
    EETDataBridge,
    EETDataBridgeConfig,
    EETVersion,
    ExportFormat,
    EETSection,
    EETField,
    EETImportResult,
    EETExportResult,
)

# ---------------------------------------------------------------------------
# Regulatory Tracking Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_010_sfdr_article_8.integrations.regulatory_tracking_bridge import (
    RegulatoryTrackingBridge,
    RegulatoryTrackingConfig,
    RegulatorySource,
    ImpactLevel,
    RegulationStatus,
    RegulatoryEvent,
    UpdateCheckResult,
    DeadlineEntry,
)

# ---------------------------------------------------------------------------
# Data Quality Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_010_sfdr_article_8.integrations.data_quality_bridge import (
    DataQualityBridge,
    DataQualityBridgeConfig,
    QualityLevel,
    CheckCategory,
    DataSourceType,
    QualityCheckResult,
    CoverageReport,
    QualityAssessment,
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_010_sfdr_article_8.integrations.health_check import (
    SFDRHealthCheck,
    HealthCheckConfig,
    HealthStatus,
    CheckArea,
    SFDRCheckCategory,
    CategoryCheckResult,
)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_010_sfdr_article_8.integrations.setup_wizard import (
    SFDRSetupWizard,
    SetupWizardConfig,
    WizardStepId,
    StepStatus as WizardStepStatus,
    ProductType,
    PresetId,
    WizardStepState,
    WizardResult,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # Pack Orchestrator
    "SFDRPackOrchestrator",
    "SFDROrchestrationConfig",
    "SFDRPipelinePhase",
    "SFDRExecutionStatus",
    "OrchestratorSFDRClassification",
    "DisclosureType",
    "PAICategory",
    "SFDRPhaseResult",
    "PipelineResult",
    "PipelineStatus",
    "SFDRQualityGateStatus",
    # Taxonomy Pack Bridge
    "TaxonomyPackBridge",
    "TaxonomyBridgeConfig",
    "TaxonomyObjective",
    "AlignmentMethodology",
    "AlignmentStatus",
    "AlignmentResult",
    "EligibilityResult",
    # MRV Emissions Bridge
    "MRVEmissionsBridge",
    "MRVEmissionsBridgeConfig",
    "EmissionScope",
    "PAIIndicator",
    "InvesteeEmissions",
    "PAIResult",
    "PortfolioEmissions",
    # Investment Screener Bridge
    "InvestmentScreenerBridge",
    "InvestmentScreenerBridgeConfig",
    "SFDRClassification",
    "ScreeningVerdict",
    "ExclusionCategory",
    "ScreeningResult",
    "ClassificationResult",
    # Portfolio Data Bridge
    "PortfolioDataBridge",
    "PortfolioDataBridgeConfig",
    "SectorClassification",
    "DataFormat",
    "DataCategory",
    "ValidatedHolding",
    "ImportResult",
    "NAVEntry",
    # EET Data Bridge
    "EETDataBridge",
    "EETDataBridgeConfig",
    "EETVersion",
    "ExportFormat",
    "EETSection",
    "EETField",
    "EETImportResult",
    "EETExportResult",
    # Regulatory Tracking Bridge
    "RegulatoryTrackingBridge",
    "RegulatoryTrackingConfig",
    "RegulatorySource",
    "ImpactLevel",
    "RegulationStatus",
    "RegulatoryEvent",
    "UpdateCheckResult",
    "DeadlineEntry",
    # Data Quality Bridge
    "DataQualityBridge",
    "DataQualityBridgeConfig",
    "QualityLevel",
    "CheckCategory",
    "DataSourceType",
    "QualityCheckResult",
    "CoverageReport",
    "QualityAssessment",
    # Health Check
    "SFDRHealthCheck",
    "HealthCheckConfig",
    "HealthStatus",
    "CheckArea",
    "SFDRCheckCategory",
    "CategoryCheckResult",
    # Setup Wizard
    "SFDRSetupWizard",
    "SetupWizardConfig",
    "WizardStepId",
    "WizardStepStatus",
    "ProductType",
    "PresetId",
    "WizardStepState",
    "WizardResult",
]
