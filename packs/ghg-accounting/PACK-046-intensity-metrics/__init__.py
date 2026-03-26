# -*- coding: utf-8 -*-
"""
PACK-046: GHG Intensity Metrics Pack
======================================

GreenLang deployment pack providing comprehensive GHG emissions intensity
metric management including denominator management, intensity calculation,
LMDI decomposition analysis, sector benchmarking, SBTi SDA target pathway
alignment, trend analysis, scenario modelling, uncertainty quantification,
multi-framework disclosure mapping, and intensity reporting.

This pack complements the calculation packs (PACK-041 Scope 1-2 Complete,
PACK-042 Scope 3 Starter, PACK-043 Scope 3 Complete), the governance pack
(PACK-044 Inventory Management), and the base year pack (PACK-045 Base Year
Management) by providing the dedicated intensity metrics layer that converts
absolute emissions into normalised performance indicators, enables
decomposition of emission drivers, and supports science-based intensity
target pathways across multiple reporting frameworks.

While PACK-041 and PACK-043 each contain basic intensity calculation
capabilities, PACK-046 provides the full intensity metrics lifecycle with
10 specialised engines covering denominator selection and management,
multi-scope intensity computation, Logarithmic Mean Divisia Index (LMDI)
decomposition, sector and peer benchmarking, SBTi Sectoral Decarbonisation
Approach (SDA) pathway tracking, time-series trend analysis, scenario and
sensitivity modelling, Monte Carlo uncertainty propagation, multi-framework
disclosure mapping, and automated intensity reporting.

Core Capabilities:
    1. Denominator Management - Selection, validation, and unit normalisation of activity denominators
    2. Intensity Calculation - Multi-scope, multi-denominator intensity computation with weighted averages
    3. LMDI Decomposition - Additive and multiplicative LMDI-I/II decomposition of emission drivers
    4. Benchmarking - Sector, peer group, and pathway benchmarking (CDP, TPI, GRESB, CRREM)
    5. SBTi SDA Targets - Sectoral Decarbonisation Approach pathway generation and progress tracking
    6. Trend Analysis - Rolling window regression, projection, and significance testing
    7. Scenario Analysis - Efficiency, growth, structural, and combined scenario modelling
    8. Uncertainty Quantification - Monte Carlo propagation and data quality-weighted confidence intervals
    9. Disclosure Mapping - Multi-framework intensity disclosure field mapping and validation
    10. Reporting - Automated intensity report generation across all supported frameworks

Engines (10):
    1. DenominatorManagementEngine - Denominator selection, validation, and unit normalisation
    2. IntensityCalculationEngine - Multi-scope intensity computation with weighted averages
    3. LMDIDecompositionEngine - Logarithmic Mean Divisia Index decomposition analysis
    4. BenchmarkingEngine - Sector and peer group benchmarking against external datasets
    5. SBTiSDATargetEngine - SBTi Sectoral Decarbonisation Approach pathway tracking
    6. TrendAnalysisEngine - Time-series trend detection, regression, and projection
    7. ScenarioAnalysisEngine - What-if scenario modelling and sensitivity analysis
    8. UncertaintyQuantificationEngine - Monte Carlo uncertainty propagation and confidence intervals
    9. DisclosureMappingEngine - Multi-framework intensity disclosure field mapping
    10. IntensityReportingEngine - Automated intensity report and dashboard generation

Workflows (8):
    1. IntensityBaselineWorkflow - Initial intensity baseline establishment
    2. AnnualIntensityCalculationWorkflow - Annual intensity computation and validation
    3. DecompositionAnalysisWorkflow - Full LMDI decomposition pipeline
    4. BenchmarkAssessmentWorkflow - Sector benchmarking and peer comparison
    5. TargetProgressWorkflow - SBTi SDA target progress assessment
    6. ScenarioModellingWorkflow - Multi-scenario intensity projection
    7. DisclosurePreparationWorkflow - Multi-framework disclosure package assembly
    8. FullIntensityPipelineWorkflow - End-to-end intensity metrics orchestration

Regulatory Basis:
    EU CSRD / ESRS E1-6 (Climate change disclosures - intensity metrics)
    SBTi Sectoral Decarbonisation Approach (SDA) v2.0
    CDP Climate Change Questionnaire C6.10 (2026) - Emissions intensities
    US SEC Climate Disclosure Rules (2024) - Intensity metrics
    ISO 14064-1:2018 (Clause 5 - Quantification of emissions per unit)
    TCFD Recommendations - Metrics and Targets (cross-industry intensity)
    GRI 305-4 (2016) - GHG emissions intensity
    IFRS S2 (2023) - Climate-related Disclosures - Intensity metrics

Category: GHG Accounting Packs
Pack Tier: Enterprise
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

__version__: str = "1.0.0"
__pack__: str = "PACK-046"
__pack_name__: str = "Intensity Metrics Pack"
__category__: str = "ghg-accounting"
