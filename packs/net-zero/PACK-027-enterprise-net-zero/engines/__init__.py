# -*- coding: utf-8 -*-
"""
PACK-027 Enterprise Net Zero Pack - Engines Module
====================================================

Deterministic, zero-hallucination calculation engines for enterprise-grade
climate management.  Each engine is designed for multi-entity, multi-scope
GHG accounting with financial-grade precision, SBTi alignment, scenario
modeling, and external assurance readiness.

Every engine produces bit-perfect reproducible results with SHA-256
provenance hashing.  No LLM is used in any scoring, classification,
or calculation path.

Engines:
    1.  EnterpriseBaselineEngine          - Full Scope 1+2+3 GHG inventory
    2.  SBTiTargetEngine                  - SBTi Corporate Standard (42 criteria)
    3.  ScenarioModelingEngine            - Monte Carlo + MACC + climate risk
    4.  CarbonPricingEngine               - Internal pricing, CBAM, carbon P&L
    5.  Scope4AvoidedEmissionsEngine      - WBCSD avoided emissions guidance
    6.  SupplyChainMappingEngine          - Multi-tier supplier mapping
    7.  MultiEntityConsolidationEngine    - GHG Protocol Ch.3 consolidation
    8.  FinancialIntegrationEngine        - Carbon P&L, balance sheet, ESRS
    9.  DataQualityGuardianEngine         - 5-level DQ hierarchy + roadmap
    10. RegulatoryComplianceEngine        - 9-framework compliance checking
    11. AssuranceReadinessEngine           - ISO 14064-3 / ISAE 3410 readiness
    12. RiskAssessmentEngine              - TCFD physical + transition risk

Design Principles:
    - Multi-entity consolidation (financial control, operational control, equity share)
    - All 30 MRV agents referenced (Scope 1: 001-008, Scope 2: 009-013, Scope 3: 014-030)
    - Dual Scope 2 reporting (location-based + market-based)
    - SBTi Corporate Manual V5.3 (C1-C28 near-term + NZ-C1 to NZ-C14 net-zero)
    - Deterministic Monte Carlo with LCG pseudo-random number generator
    - SHA-256 provenance hash on every result
    - Decimal arithmetic with ROUND_HALF_UP throughout
    - Performance: <5 seconds per engine, <60 seconds full suite

Regulatory / Framework Basis:
    GHG Protocol Corporate Standard (2004, revised 2015)
    GHG Protocol Scope 3 Standard (2011)
    GHG Protocol Corporate Value Chain (Scope 3) Accounting Standard
    SBTi Corporate Net-Zero Standard v1.2 (2024)
    SBTi Corporate Manual V5.3 (2024)
    IPCC AR6 WG1 (2021) - GWP-100 values
    DEFRA/BEIS 2024 UK GHG Conversion Factors
    US EPA EEIO v2.0 - Spend-based emission factors
    TCFD Recommendations (2017, updated 2021)
    ISO 14064-1:2018, ISO 14064-3:2019
    ISAE 3410 Assurance Engagements on GHG Statements
    WBCSD Avoided Emissions Guidance (2023)
    ESRS E1 Climate Change (EFRAG, 2023)
    SEC Climate Disclosure Rule (2024)
    CA SB 253 / SB 261 (2023)
    IFRS S2 / ISSB (2023)
    EU CBAM Regulation (EU) 2023/956
    Paris Agreement (2015) - 1.5C temperature target

Zero-Hallucination Declaration:
    Every numerical output is derived from hard-coded emission factors,
    peer-reviewed methodologies, or user-supplied data.  No generative
    model is invoked at any point in the calculation pipeline.

Pack Tier: Enterprise (PACK-027)
Category: Net Zero Packs
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-027"
__pack_name__: str = "Enterprise Net Zero Pack"
__engines_count__: int = 12

# --- Engine 1: Enterprise Baseline --------------------------------------------
_engine_1_symbols: list[str] = []
try:
    from .enterprise_baseline_engine import (  # noqa: F401
        ConsolidationApproach,
        DataQualityLevel,
        Scope3Category,
        CalculationApproach,
        GHGGas,
        MaterialityClassification,
        Scope1Source,
        Scope2Method,
        FuelType,
        FuelEntry,
        ElectricityEntry,
        RefrigerantEntry,
        ProcessEmissionEntry,
        SteamCoolingEntry,
        Scope3CategoryEntry,
        EntityDefinition,
        IntercompanyTransaction,
        EnterpriseBaselineInput,
        Scope1Breakdown,
        Scope2Breakdown,
        Scope3CategoryResult,
        Scope3Breakdown,
        DataQualityMatrix,
        MaterialityAssessment,
        ConsolidationSummary,
        IntensityMetrics,
        ConfidenceInterval,
        EnterpriseBaselineResult,
        EnterpriseBaselineEngine,
    )
    _engine_1_symbols = [
        "ConsolidationApproach", "DataQualityLevel", "Scope3Category",
        "CalculationApproach", "GHGGas", "MaterialityClassification",
        "Scope1Source", "Scope2Method", "FuelType",
        "FuelEntry", "ElectricityEntry", "RefrigerantEntry",
        "ProcessEmissionEntry", "SteamCoolingEntry", "Scope3CategoryEntry",
        "EntityDefinition", "IntercompanyTransaction",
        "EnterpriseBaselineInput",
        "Scope1Breakdown", "Scope2Breakdown",
        "Scope3CategoryResult", "Scope3Breakdown",
        "DataQualityMatrix", "MaterialityAssessment",
        "ConsolidationSummary", "IntensityMetrics", "ConfidenceInterval",
        "EnterpriseBaselineResult", "EnterpriseBaselineEngine",
    ]
    logger.debug("Engine 1 (EnterpriseBaselineEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 1 (EnterpriseBaselineEngine) not available: %s", exc)

# --- Engine 2: SBTi Target ----------------------------------------------------
_engine_2_symbols: list[str] = []
try:
    from .sbti_target_engine import (  # noqa: F401
        TargetPathwayType,
        TargetScope,
        TargetType,
        CriterionStatus,
        SDASector,
        MilestoneStatus,
        BaselineData,
        SBTiTargetInput,
        CriterionValidation,
        TargetDefinition,
        MilestoneEntry,
        FairShareAssessment,
        ProgressAssessment,
        SBTiTargetResult,
        SBTiTargetEngine,
    )
    _engine_2_symbols = [
        "TargetPathwayType", "TargetScope", "TargetType",
        "CriterionStatus", "SDASector", "MilestoneStatus",
        "BaselineData", "SBTiTargetInput",
        "CriterionValidation", "TargetDefinition", "MilestoneEntry",
        "FairShareAssessment", "ProgressAssessment",
        "SBTiTargetResult", "SBTiTargetEngine",
    ]
    logger.debug("Engine 2 (SBTiTargetEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 2 (SBTiTargetEngine) not available: %s", exc)

# --- Engine 3: Scenario Modeling -----------------------------------------------
_engine_3_symbols: list[str] = []
try:
    from .scenario_modeling_engine import (  # noqa: F401
        ScenarioType,
        RiskCategory as ScenarioRiskCategory,
        DistributionType,
        ParameterDistribution,
        MACCAction,
        ScenarioModelingInput,
        AnnualTrajectoryPoint,
        ScenarioTrajectory,
        SensitivityDriver,
        MACCResult,
        ClimateRiskScore,
        ScenarioModelingResult,
        ScenarioModelingEngine,
    )
    _engine_3_symbols = [
        "ScenarioType", "ScenarioRiskCategory", "DistributionType",
        "ParameterDistribution", "MACCAction", "ScenarioModelingInput",
        "AnnualTrajectoryPoint", "ScenarioTrajectory",
        "SensitivityDriver", "MACCResult", "ClimateRiskScore",
        "ScenarioModelingResult", "ScenarioModelingEngine",
    ]
    logger.debug("Engine 3 (ScenarioModelingEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 3 (ScenarioModelingEngine) not available: %s", exc)

# --- Engine 4: Carbon Pricing --------------------------------------------------
_engine_4_symbols: list[str] = []
try:
    from .carbon_pricing_engine import (  # noqa: F401
        CarbonPricingApproach,
        PriceTrajectoryScenario,
        AllocationMethod,
        BusinessUnitEmissions,
        InvestmentProposal,
        CBAMImport,
        CarbonPricingInput,
        BUCarbonAllocation,
        InvestmentAppraisal,
        CBAMExposure,
        CarbonPnL,
        CarbonLiability,
        CarbonPricingResult,
        CarbonPricingEngine,
    )
    _engine_4_symbols = [
        "CarbonPricingApproach", "PriceTrajectoryScenario", "AllocationMethod",
        "BusinessUnitEmissions", "InvestmentProposal", "CBAMImport",
        "CarbonPricingInput",
        "BUCarbonAllocation", "InvestmentAppraisal", "CBAMExposure",
        "CarbonPnL", "CarbonLiability",
        "CarbonPricingResult", "CarbonPricingEngine",
    ]
    logger.debug("Engine 4 (CarbonPricingEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 4 (CarbonPricingEngine) not available: %s", exc)

# --- Engine 5: Scope 4 Avoided Emissions --------------------------------------
_engine_5_symbols: list[str] = []
try:
    from .scope4_avoided_emissions_engine import (  # noqa: F401
        AvoidedEmissionCategory,
        BaselineType,
        AdditionalityLevel,
        ConfidenceLevel,
        ProductAvoidedEmissionEntry,
        Scope4Input,
        ProductAvoidedResult,
        DoubleCounting,
        Scope4Result,
        Scope4AvoidedEmissionsEngine,
    )
    _engine_5_symbols = [
        "AvoidedEmissionCategory", "BaselineType",
        "AdditionalityLevel", "ConfidenceLevel",
        "ProductAvoidedEmissionEntry", "Scope4Input",
        "ProductAvoidedResult", "DoubleCounting",
        "Scope4Result", "Scope4AvoidedEmissionsEngine",
    ]
    logger.debug("Engine 5 (Scope4AvoidedEmissionsEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 5 (Scope4AvoidedEmissionsEngine) not available: %s", exc)

# --- Engine 6: Supply Chain Mapping --------------------------------------------
_engine_6_symbols: list[str] = []
try:
    from .supply_chain_mapping_engine import (  # noqa: F401
        SupplierTier,
        EngagementLevel,
        CDPScore,
        SBTiStatus,
        RiskLevel as SupplyChainRiskLevel,
        SupplierEntry,
        SupplyChainMappingInput,
        SupplierScorecard,
        TierSummary,
        GeographicHotspot,
        EngagementProgramStatus,
        SupplyChainMappingResult,
        SupplyChainMappingEngine,
    )
    _engine_6_symbols = [
        "SupplierTier", "EngagementLevel", "CDPScore",
        "SBTiStatus", "SupplyChainRiskLevel",
        "SupplierEntry", "SupplyChainMappingInput",
        "SupplierScorecard", "TierSummary",
        "GeographicHotspot", "EngagementProgramStatus",
        "SupplyChainMappingResult", "SupplyChainMappingEngine",
    ]
    logger.debug("Engine 6 (SupplyChainMappingEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 6 (SupplyChainMappingEngine) not available: %s", exc)

# --- Engine 7: Multi-Entity Consolidation --------------------------------------
_engine_7_symbols: list[str] = []
try:
    from .multi_entity_consolidation_engine import (  # noqa: F401
        ConsolidationApproach as MEConsolidationApproach,
        EntityType,
        RecalculationTrigger,
        CurrencyCode,
        EntityEmissions,
        IntercompanyEntry,
        BaseYearData,
        RecalculationEvent,
        ConsolidationInput,
        EntityContribution,
        EliminationEntry,
        BaseYearRecalculation,
        ConsolidationResult,
        MultiEntityConsolidationEngine,
    )
    _engine_7_symbols = [
        "MEConsolidationApproach", "EntityType",
        "RecalculationTrigger", "CurrencyCode",
        "EntityEmissions", "IntercompanyEntry",
        "BaseYearData", "RecalculationEvent", "ConsolidationInput",
        "EntityContribution", "EliminationEntry",
        "BaseYearRecalculation", "ConsolidationResult",
        "MultiEntityConsolidationEngine",
    ]
    logger.debug("Engine 7 (MultiEntityConsolidationEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 7 (MultiEntityConsolidationEngine) not available: %s", exc)

# --- Engine 8: Financial Integration -------------------------------------------
_engine_8_symbols: list[str] = []
try:
    from .financial_integration_engine import (  # noqa: F401
        PnLLineItem,
        CarbonAssetType,
        CarbonLiabilityType,
        FinancialData,
        CarbonAsset,
        CarbonLiabilityEntry,
        EmissionsByFunction,
        FinancialIntegrationInput,
        CarbonPnLAllocation,
        CarbonBalanceSheet,
        CarbonIntensityMetrics,
        ESRSE1Disclosure,
        FinancialIntegrationResult,
        FinancialIntegrationEngine,
    )
    _engine_8_symbols = [
        "PnLLineItem", "CarbonAssetType", "CarbonLiabilityType",
        "FinancialData", "CarbonAsset", "CarbonLiabilityEntry",
        "EmissionsByFunction", "FinancialIntegrationInput",
        "CarbonPnLAllocation", "CarbonBalanceSheet",
        "CarbonIntensityMetrics", "ESRSE1Disclosure",
        "FinancialIntegrationResult", "FinancialIntegrationEngine",
    ]
    logger.debug("Engine 8 (FinancialIntegrationEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 8 (FinancialIntegrationEngine) not available: %s", exc)

# --- Engine 9: Data Quality Guardian -------------------------------------------
_engine_9_symbols: list[str] = []
try:
    from .data_quality_guardian_engine import (  # noqa: F401
        IssueType,
        IssueSeverity,
        ImprovementAction,
        CategoryDQEntry,
        DataQualityGuardianInput,
        DQIssue,
        ImprovementPriority,
        DataQualityGuardianResult,
        DataQualityGuardianEngine,
    )
    _engine_9_symbols = [
        "IssueType", "IssueSeverity", "ImprovementAction",
        "CategoryDQEntry", "DataQualityGuardianInput",
        "DQIssue", "ImprovementPriority",
        "DataQualityGuardianResult", "DataQualityGuardianEngine",
    ]
    logger.debug("Engine 9 (DataQualityGuardianEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 9 (DataQualityGuardianEngine) not available: %s", exc)

# --- Engine 10: Regulatory Compliance -----------------------------------------
_engine_10_symbols: list[str] = []
try:
    from .regulatory_compliance_engine import (  # noqa: F401
        RegulatoryFramework,
        ComplianceStatus,
        GapSeverity as RegulatoryGapSeverity,
        FrameworkApplicability,
        ComplianceDataAvailability,
        RegulatoryComplianceInput,
        FrameworkAssessment,
        ComplianceGap,
        CrosswalkEntry,
        RegulatoryComplianceResult,
        RegulatoryComplianceEngine,
    )
    _engine_10_symbols = [
        "RegulatoryFramework", "ComplianceStatus", "RegulatoryGapSeverity",
        "FrameworkApplicability", "ComplianceDataAvailability",
        "RegulatoryComplianceInput",
        "FrameworkAssessment", "ComplianceGap", "CrosswalkEntry",
        "RegulatoryComplianceResult", "RegulatoryComplianceEngine",
    ]
    logger.debug("Engine 10 (RegulatoryComplianceEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 10 (RegulatoryComplianceEngine) not available: %s", exc)

# --- Engine 11: Assurance Readiness --------------------------------------------
_engine_11_symbols: list[str] = []
try:
    from .assurance_readiness_engine import (  # noqa: F401
        AssuranceLevel,
        AssuranceStandard,
        WorkpaperStatus,
        ReadinessDimension,
        WorkpaperInput,
        DimensionScore,
        AssuranceReadinessInput,
        WorkpaperAssessment,
        DimensionAssessment,
        TimelineEstimate,
        AssuranceReadinessResult,
        AssuranceReadinessEngine,
    )
    _engine_11_symbols = [
        "AssuranceLevel", "AssuranceStandard", "WorkpaperStatus",
        "ReadinessDimension",
        "WorkpaperInput", "DimensionScore", "AssuranceReadinessInput",
        "WorkpaperAssessment", "DimensionAssessment", "TimelineEstimate",
        "AssuranceReadinessResult", "AssuranceReadinessEngine",
    ]
    logger.debug("Engine 11 (AssuranceReadinessEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 11 (AssuranceReadinessEngine) not available: %s", exc)

# --- Engine 12: Risk Assessment ------------------------------------------------
_engine_12_symbols: list[str] = []
try:
    from .risk_assessment_engine import (  # noqa: F401
        PhysicalRiskType,
        TransitionRiskType,
        RiskCategory as TcfdRiskCategory,
        RiskSeverity,
        TimeHorizon,
        ClimateScenario,
        AssetEntry,
        TransitionRiskEntry,
        MitigationAction,
        RiskAssessmentInput,
        RiskScore,
        AssetRiskExposure,
        ScenarioRiskOutcome,
        MitigationRecommendation,
        RiskAssessmentResult,
        RiskAssessmentEngine,
    )
    _engine_12_symbols = [
        "PhysicalRiskType", "TransitionRiskType",
        "TcfdRiskCategory", "RiskSeverity", "TimeHorizon", "ClimateScenario",
        "AssetEntry", "TransitionRiskEntry", "MitigationAction",
        "RiskAssessmentInput",
        "RiskScore", "AssetRiskExposure",
        "ScenarioRiskOutcome", "MitigationRecommendation",
        "RiskAssessmentResult", "RiskAssessmentEngine",
    ]
    logger.debug("Engine 12 (RiskAssessmentEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 12 (RiskAssessmentEngine) not available: %s", exc)

# --- Dynamic __all__ -----------------------------------------------------------

_loaded_engines: list[str] = []
if _engine_1_symbols:
    _loaded_engines.append("EnterpriseBaselineEngine")
if _engine_2_symbols:
    _loaded_engines.append("SBTiTargetEngine")
if _engine_3_symbols:
    _loaded_engines.append("ScenarioModelingEngine")
if _engine_4_symbols:
    _loaded_engines.append("CarbonPricingEngine")
if _engine_5_symbols:
    _loaded_engines.append("Scope4AvoidedEmissionsEngine")
if _engine_6_symbols:
    _loaded_engines.append("SupplyChainMappingEngine")
if _engine_7_symbols:
    _loaded_engines.append("MultiEntityConsolidationEngine")
if _engine_8_symbols:
    _loaded_engines.append("FinancialIntegrationEngine")
if _engine_9_symbols:
    _loaded_engines.append("DataQualityGuardianEngine")
if _engine_10_symbols:
    _loaded_engines.append("RegulatoryComplianceEngine")
if _engine_11_symbols:
    _loaded_engines.append("AssuranceReadinessEngine")
if _engine_12_symbols:
    _loaded_engines.append("RiskAssessmentEngine")

__all__: list[str] = (
    _engine_1_symbols
    + _engine_2_symbols
    + _engine_3_symbols
    + _engine_4_symbols
    + _engine_5_symbols
    + _engine_6_symbols
    + _engine_7_symbols
    + _engine_8_symbols
    + _engine_9_symbols
    + _engine_10_symbols
    + _engine_11_symbols
    + _engine_12_symbols
)


def get_loaded_engines() -> list[str]:
    """Return names of successfully loaded engines."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return total number of expected engines."""
    return __engines_count__


def get_loaded_engine_count() -> int:
    """Return number of successfully loaded engines."""
    return len(_loaded_engines)


logger.info(
    "PACK-027 engines: %d / %d loaded",
    get_loaded_engine_count(),
    get_engine_count(),
)
