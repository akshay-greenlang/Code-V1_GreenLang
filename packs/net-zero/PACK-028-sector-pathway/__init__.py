# -*- coding: utf-8 -*-
"""
PACK-028: Sector Pathway Pack
=============================================================================

Comprehensive GreenLang deployment pack for sector-level decarbonization
pathway analysis aligned with SBTi Sectoral Decarbonization Approach (SDA).
Provides 8 engines covering NACE/GICS/ISIC sector classification with SDA
eligibility determination, sector-specific carbon intensity calculation
(physical/economic/hybrid), SDA/IEA convergence pathway generation for 12+
sectors, gap analysis and trajectory convergence assessment, IEA NZE 2050
technology milestone tracking with S-curve adoption modeling, MACC waterfall
and abatement cost curve analysis, multi-dimensional sector benchmarking
against 5 benchmark sources, and multi-scenario risk-return comparison
across 5 temperature scenarios.

Supports 15+ industrial sectors (Power, Steel, Cement, Aluminum, Chemicals,
Pulp & Paper, Aviation, Shipping, Road Transport, Rail, Buildings,
Agriculture, Food & Beverage, Oil & Gas, Cross-Sector) with IEA/IPCC/TPI/
MPP/ACT pathway databases.

Components:
    Engines (8):
        - SectorClassificationEngine      (NACE/GICS/ISIC sector mapping)
        - IntensityCalculatorEngine        (Sector-specific intensity metrics)
        - PathwayGeneratorEngine           (SDA/IEA convergence pathways)
        - ConvergenceAnalyzerEngine        (Gap analysis & trajectory)
        - TechnologyRoadmapEngine          (IEA milestone tracking)
        - AbatementWaterfallEngine         (MACC waterfall & phasing)
        - SectorBenchmarkEngine            (Peer percentile benchmarking)
        - ScenarioComparisonEngine         (Multi-scenario comparison)

    Workflows (6):
        - SectorPathwayDesignWorkflow           (5 phases)
        - PathwayValidationWorkflow             (4 phases)
        - TechnologyPlanningWorkflow            (5 phases)
        - ProgressMonitoringWorkflow            (4 phases)
        - MultiScenarioAnalysisWorkflow         (5 phases)
        - FullSectorAssessmentWorkflow          (7 phases)

    Templates (8):
        - SectorPathwayReportTemplate
        - IntensityConvergenceReportTemplate
        - TechnologyRoadmapReportTemplate
        - AbatementWaterfallReportTemplate
        - SectorBenchmarkReportTemplate
        - ScenarioComparisonReportTemplate
        - SBTiValidationReportTemplate
        - SectorStrategyReportTemplate

    Integrations (10):
        - SectorPathwayPipelineOrchestrator  (10-phase DAG pipeline)
        - SBTiSDABridge                      (12-sector SDA convergence)
        - IEANZEBridge                       (400+ technology milestones)
        - IPCCAR6Bridge                      (GWP-100, emission factors)
        - PACK021Bridge                      (Baseline/target import)
        - SectorMRVBridge                    (30 MRV agents)
        - SectorDecarbBridge                 (Abatement lever registry)
        - SectorDataBridge                   (20 DATA agents)
        - SectorPathwaySetupWizard           (7-step configuration)
        - SectorPathwayHealthCheck           (20-category verification)

    Presets (6):
        - power_generation         (Power & Utilities)
        - heavy_industry           (Steel, Cement, Chemicals)
        - transport                (Aviation, Shipping, Road, Rail)
        - buildings                (Commercial & Residential)
        - agriculture              (Agriculture & Food)
        - cross_sector             (Multi-sector conglomerates)

Agent Dependencies:
    - 30 AGENT-MRV agents (Scope 1/2/3 emissions quantification)
    - 20 AGENT-DATA agents (data intake and quality management)
    - 10 AGENT-FOUND agents (platform foundation services)

Regulatory Framework:
    Primary:
        - SBTi Sectoral Decarbonization Approach (2015, updated 2024)
        - SBTi Corporate Net-Zero Standard v1.2 (2024)
        - IEA Net Zero by 2050 Roadmap (2023)
    Secondary:
        - IPCC AR6 WG1/WG3 (2021-2022)
        - EU ETS Benchmark Values (2021-2025)
        - NACE Rev.2, GICS, ISIC Rev.4 classifications
        - GHG Protocol Corporate/Scope 3 Standards
        - Paris Agreement (2015)

Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-028"
__pack_name__ = "Sector Pathway Pack"
__author__ = "GreenLang Platform Team"
__category__: str = "net-zero"

# ---------------------------------------------------------------------------
# Engines (8)
# ---------------------------------------------------------------------------
from .engines import (
    SectorClassificationEngine,
    IntensityCalculatorEngine,
    PathwayGeneratorEngine,
    ConvergenceAnalyzerEngine,
    TechnologyRoadmapEngine,
    AbatementWaterfallEngine,
    SectorBenchmarkEngine,
    ScenarioComparisonEngine,
)

# ---------------------------------------------------------------------------
# Workflows (6)
# ---------------------------------------------------------------------------
from .workflows import (
    SectorPathwayDesignWorkflow,
    PathwayValidationWorkflow,
    TechnologyPlanningWorkflow,
    ProgressMonitoringWorkflow,
    MultiScenarioAnalysisWorkflow,
    FullSectorAssessmentWorkflow,
)

# ---------------------------------------------------------------------------
# Templates (8 + Registry)
# ---------------------------------------------------------------------------
from .templates import (
    SectorPathwayReportTemplate,
    IntensityConvergenceReportTemplate,
    TechnologyRoadmapReportTemplate,
    AbatementWaterfallReportTemplate,
    SectorBenchmarkReportTemplate,
    ScenarioComparisonReportTemplate,
    SBTiValidationReportTemplate,
    SectorStrategyReportTemplate,
    TemplateRegistry,
)

# ---------------------------------------------------------------------------
# Integrations (10)
# ---------------------------------------------------------------------------
from .integrations import (
    SectorPathwayPipelineOrchestrator,
    SBTiSDABridge,
    IEANZEBridge,
    IPCCAR6Bridge,
    PACK021Bridge,
    SectorMRVBridge,
    SectorDecarbBridge,
    SectorDataBridge,
    SectorPathwaySetupWizard,
    SectorPathwayHealthCheck,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "__author__",
    # --- Engines (8) ---
    "SectorClassificationEngine",
    "IntensityCalculatorEngine",
    "PathwayGeneratorEngine",
    "ConvergenceAnalyzerEngine",
    "TechnologyRoadmapEngine",
    "AbatementWaterfallEngine",
    "SectorBenchmarkEngine",
    "ScenarioComparisonEngine",
    # --- Workflows (6) ---
    "SectorPathwayDesignWorkflow",
    "PathwayValidationWorkflow",
    "TechnologyPlanningWorkflow",
    "ProgressMonitoringWorkflow",
    "MultiScenarioAnalysisWorkflow",
    "FullSectorAssessmentWorkflow",
    # --- Templates (8 + Registry) ---
    "SectorPathwayReportTemplate",
    "IntensityConvergenceReportTemplate",
    "TechnologyRoadmapReportTemplate",
    "AbatementWaterfallReportTemplate",
    "SectorBenchmarkReportTemplate",
    "ScenarioComparisonReportTemplate",
    "SBTiValidationReportTemplate",
    "SectorStrategyReportTemplate",
    "TemplateRegistry",
    # --- Integrations (10) ---
    "SectorPathwayPipelineOrchestrator",
    "SBTiSDABridge",
    "IEANZEBridge",
    "IPCCAR6Bridge",
    "PACK021Bridge",
    "SectorMRVBridge",
    "SectorDecarbBridge",
    "SectorDataBridge",
    "SectorPathwaySetupWizard",
    "SectorPathwayHealthCheck",
]
