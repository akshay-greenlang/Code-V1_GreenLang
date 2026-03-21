# -*- coding: utf-8 -*-
"""
PACK-008 EU Taxonomy Alignment Pack - Integration Layer
==========================================================

10 integration bridges for the EU Taxonomy Alignment Pack providing
pipeline orchestration, application bridging, MRV data routing,
CSRD cross-framework mapping, financial data integration, activity
registry connectivity, GAR data feeds, evidence management,
regulatory tracking, data quality monitoring, system health checks,
and guided setup configuration.

Components:
    - TaxonomyPackOrchestrator         - Multi-phase pipeline orchestration
    - TaxonomyAppBridge                - GL-Taxonomy-APP integration bridge
    - MRVTaxonomyBridge                - MRV agent routing for taxonomy data
    - CSRDCrossFrameworkBridge         - CSRD cross-framework alignment
    - FinancialDataBridge              - ERP/financial data integration
    - ActivityRegistryBridge           - EU Taxonomy activity registry lookup
    - GARDataBridge                    - EBA GAR exposure data feeds
    - EvidenceManagementBridge         - Document and evidence management
    - RegulatoryTrackingBridge         - Delegated Act update tracking
    - DataQualityBridge                - Data quality monitoring
    - TaxonomyHealthCheck              - System health monitoring
    - TaxonomySetupWizard              - Guided pack configuration wizard

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-008 EU Taxonomy Alignment Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-008"
__pack_name__ = "EU Taxonomy Alignment Pack"

# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
from .pack_orchestrator import (
    TaxonomyPackOrchestrator,
    TaxonomyOrchestratorConfig,
    OrchestratorResult,
)

# ---------------------------------------------------------------------------
# Application Bridge
# ---------------------------------------------------------------------------
from .taxonomy_app_bridge import TaxonomyAppBridge, TaxonomyAppBridgeConfig

# ---------------------------------------------------------------------------
# MRV Taxonomy Bridge
# ---------------------------------------------------------------------------
from .mrv_taxonomy_bridge import MRVTaxonomyBridge, MRVTaxonomyBridgeConfig

# ---------------------------------------------------------------------------
# CSRD Cross-Framework Bridge
# ---------------------------------------------------------------------------
from .csrd_cross_framework_bridge import CSRDCrossFrameworkBridge, CrossFrameworkConfig

# ---------------------------------------------------------------------------
# Financial Data Bridge
# ---------------------------------------------------------------------------
from .financial_data_bridge import FinancialDataBridge, FinancialDataConfig

# ---------------------------------------------------------------------------
# Activity Registry Bridge
# ---------------------------------------------------------------------------
from .activity_registry_bridge import ActivityRegistryBridge, ActivityRegistryConfig

# ---------------------------------------------------------------------------
# GAR Data Bridge
# ---------------------------------------------------------------------------
from .gar_data_bridge import GARDataBridge, GARDataConfig

# ---------------------------------------------------------------------------
# Evidence Management Bridge
# ---------------------------------------------------------------------------
from .evidence_management_bridge import EvidenceManagementBridge, EvidenceConfig

# ---------------------------------------------------------------------------
# Regulatory Tracking Bridge
# ---------------------------------------------------------------------------
from .regulatory_tracking_bridge import RegulatoryTrackingBridge, RegulatoryTrackingConfig

# ---------------------------------------------------------------------------
# Data Quality Bridge
# ---------------------------------------------------------------------------
from .data_quality_bridge import DataQualityBridge, DataQualityConfig

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
from .health_check import TaxonomyHealthCheck, HealthCheckConfig, HealthCheckResult

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
from .setup_wizard import TaxonomySetupWizard, SetupWizardConfig

__all__ = [
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Pipeline Orchestrator ---
    "TaxonomyPackOrchestrator",
    "TaxonomyOrchestratorConfig",
    "OrchestratorResult",
    # --- Application Bridge ---
    "TaxonomyAppBridge",
    "TaxonomyAppBridgeConfig",
    # --- MRV Taxonomy Bridge ---
    "MRVTaxonomyBridge",
    "MRVTaxonomyBridgeConfig",
    # --- CSRD Cross-Framework Bridge ---
    "CSRDCrossFrameworkBridge",
    "CrossFrameworkConfig",
    # --- Financial Data Bridge ---
    "FinancialDataBridge",
    "FinancialDataConfig",
    # --- Activity Registry Bridge ---
    "ActivityRegistryBridge",
    "ActivityRegistryConfig",
    # --- GAR Data Bridge ---
    "GARDataBridge",
    "GARDataConfig",
    # --- Evidence Management Bridge ---
    "EvidenceManagementBridge",
    "EvidenceConfig",
    # --- Regulatory Tracking Bridge ---
    "RegulatoryTrackingBridge",
    "RegulatoryTrackingConfig",
    # --- Data Quality Bridge ---
    "DataQualityBridge",
    "DataQualityConfig",
    # --- Health Check ---
    "TaxonomyHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    # --- Setup Wizard ---
    "TaxonomySetupWizard",
    "SetupWizardConfig",
]
