# -*- coding: utf-8 -*-
"""
PACK-012 CSRD Financial Service Pack - Engines Module
======================================================

Calculation engines for financial institutions implementing CSRD ESG disclosures
under ESRS E1, S1, G1 and sector-specific ESRS standards. This pack covers
financed emissions (PCAF), insurance underwriting, Green Asset Ratio (GAR),
Banking Book Taxonomy Alignment Ratio (BTAR), climate risk scoring (NGFS),
financial sector double materiality, transition plans, and Pillar 3 ESG disclosures.

Engines:
    1. FinancedEmissionsEngine          - PCAF methodology for financed emissions
    2. InsuranceUnderwritingEngine      - Emissions from insurance underwriting
    3. GreenAssetRatioEngine            - GAR calculation per EBA guidelines
    4. BTARCalculatorEngine             - Banking Book Taxonomy Alignment Ratio
    5. ClimateRiskScoringEngine         - Physical & transition risk scoring (NGFS)
    6. FSDoubleMaterialityEngine        - Financial sector double materiality assessment
    7. FSTransitionPlanEngine           - Financial sector transition plans (SBTi-FI, NZBA)
    8. Pillar3ESGEngine                 - CRR3 Pillar 3 ESG disclosure templates
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-012"
__pack_name__: str = "CSRD Financial Service Pack"
__engines_count__: int = 8

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: Financed Emissions
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "FinancedEmissionsEngine",
    "FinancedEmissionsConfig",
    "AssetClassData",
    "HoldingEmissions",
    "AttributionResult",
    "PortfolioEmissionsResult",
    "DataQualityScore",
    "EmissionsByAssetClass",
    "PCAFAssetClass",
    "DataQualityLevel",
]

try:
    from .financed_emissions_engine import (
        AssetClassData,
        AttributionResult,
        DataQualityLevel,
        DataQualityScore,
        EmissionsByAssetClass,
        FinancedEmissionsConfig,
        FinancedEmissionsEngine,
        HoldingEmissions,
        PCAFAssetClass,
        PortfolioEmissionsResult,
    )
    _loaded_engines.append("FinancedEmissionsEngine")
except ImportError as e:
    logger.debug("Engine 1 (FinancedEmissionsEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Insurance Underwriting
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "InsuranceUnderwritingEngine",
    "UnderwritingConfig",
    "PolicyData",
    "UnderwritingEmissionsResult",
    "LineOfBusinessResult",
    "ReinsuranceAdjustment",
    "ClaimsEmissions",
    "InsuranceLine",
]

try:
    from .insurance_underwriting_engine import (
        ClaimsEmissions,
        InsuranceLine,
        InsuranceUnderwritingEngine,
        LineOfBusinessResult,
        PolicyData,
        ReinsuranceAdjustment,
        UnderwritingConfig,
        UnderwritingEmissionsResult,
    )
    _loaded_engines.append("InsuranceUnderwritingEngine")
except ImportError as e:
    logger.debug("Engine 2 (InsuranceUnderwritingEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: Green Asset Ratio
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "GreenAssetRatioEngine",
    "GARConfig",
    "CoveredAssetData",
    "GARResult",
    "GARBreakdown",
    "CounterpartyBreakdown",
    "OffBalanceSheetKPI",
    "FlowGAR",
    "GARScope",
    "CounterpartyType",
    "EnvironmentalObjective",
]

try:
    from .green_asset_ratio_engine import (
        CounterpartyBreakdown,
        CounterpartyType,
        CoveredAssetData,
        EnvironmentalObjective,
        FlowGAR,
        GARBreakdown,
        GARConfig,
        GARResult,
        GARScope,
        GreenAssetRatioEngine,
        OffBalanceSheetKPI,
    )
    _loaded_engines.append("GreenAssetRatioEngine")
except ImportError as e:
    logger.debug("Engine 3 (GreenAssetRatioEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: BTAR Calculator
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "BTARCalculatorEngine",
    "BTARConfig",
    "BankingBookData",
    "BTARResult",
    "EstimationMethodology",
    "SectorProxyResult",
    "DataCoverageReport",
    "BTARvsGARReconciliation",
    "EstimationType",
]

try:
    from .btar_calculator_engine import (
        BTARCalculatorEngine,
        BTARConfig,
        BTARResult,
        BTARvsGARReconciliation,
        BankingBookData,
        DataCoverageReport,
        EstimationMethodology,
        EstimationType,
        SectorProxyResult,
    )
    _loaded_engines.append("BTARCalculatorEngine")
except ImportError as e:
    logger.debug("Engine 4 (BTARCalculatorEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Climate Risk Scoring
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "ClimateRiskScoringEngine",
    "ClimateRiskConfig",
    "ExposureData",
    "ClimateRiskResult",
    "PhysicalRiskScore",
    "TransitionRiskScore",
    "NGFSScenarioResult",
    "StrandedAssetExposure",
    "CreditRiskImpact",
    "NGFSScenario",
    "PhysicalHazard",
    "TransitionChannel",
    "TimeHorizon",
    "RiskLevel",
]

try:
    from .climate_risk_scoring_engine import (
        ClimateRiskConfig,
        ClimateRiskResult,
        ClimateRiskScoringEngine,
        CreditRiskImpact,
        ExposureData,
        NGFSScenario,
        NGFSScenarioResult,
        PhysicalHazard,
        PhysicalRiskScore,
        RiskLevel,
        StrandedAssetExposure,
        TimeHorizon,
        TransitionChannel,
        TransitionRiskScore,
    )
    _loaded_engines.append("ClimateRiskScoringEngine")
except ImportError as e:
    logger.debug("Engine 5 (ClimateRiskScoringEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: FS Double Materiality
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "FSDoubleMaterialityEngine",
    "FSMaterialityConfig",
    "MaterialityTopicData",
    "FSMaterialityResult",
    "IROAssessment",
    "FinancedImpactAssessment",
    "StakeholderInput",
    "MaterialityMatrix",
    "DatapointMapping",
    "MaterialityDimension",
    "IROType",
    "ESRSStandard",
]

try:
    from .fs_double_materiality_engine import (
        DatapointMapping,
        ESRSStandard,
        FSDoubleMaterialityEngine,
        FSMaterialityConfig,
        FSMaterialityResult,
        FinancedImpactAssessment,
        IROAssessment,
        IROType,
        MaterialityDimension,
        MaterialityMatrix,
        MaterialityTopicData,
        StakeholderInput,
    )
    _loaded_engines.append("FSDoubleMaterialityEngine")
except ImportError as e:
    logger.debug("Engine 6 (FSDoubleMaterialityEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: FS Transition Plan
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "FSTransitionPlanEngine",
    "TransitionPlanConfig",
    "SectorTargetData",
    "TransitionPlanResult",
    "SBTiFIAssessment",
    "NZBACommitment",
    "SectorDecarbPath",
    "PhaseOutCommitment",
    "CredibilityScore",
    "SBTiMethod",
    "AllianceType",
]

try:
    from .fs_transition_plan_engine import (
        AllianceType,
        CredibilityScore,
        FSTransitionPlanEngine,
        NZBACommitment,
        PhaseOutCommitment,
        SBTiFIAssessment,
        SBTiMethod,
        SectorDecarbPath,
        SectorTargetData,
        TransitionPlanConfig,
        TransitionPlanResult,
    )
    _loaded_engines.append("FSTransitionPlanEngine")
except ImportError as e:
    logger.debug("Engine 7 (FSTransitionPlanEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: Pillar 3 ESG
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "Pillar3ESGEngine",
    "Pillar3Config",
    "BankingBookExposure",
    "Pillar3Result",
    "TransitionRiskTemplate",
    "PhysicalRiskTemplate",
    "RealEstateTemplate",
    "Top20CarbonExposure",
    "TaxonomyAlignmentTemplate",
    "QualitativeDisclosure",
    "Pillar3TemplateType",
    "EPCLabel",
    "NACESector",
]

try:
    from .pillar3_esg_engine import (
        BankingBookExposure,
        EPCLabel,
        NACESector,
        PhysicalRiskTemplate,
        Pillar3Config,
        Pillar3ESGEngine,
        Pillar3Result,
        Pillar3TemplateType,
        QualitativeDisclosure,
        RealEstateTemplate,
        TaxonomyAlignmentTemplate,
        Top20CarbonExposure,
        TransitionRiskTemplate,
    )
    _loaded_engines.append("Pillar3ESGEngine")
except ImportError as e:
    logger.debug("Engine 8 (Pillar3ESGEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []


# ===================================================================
# Public API - dynamically collected from successfully loaded engines
# ===================================================================

_METADATA_SYMBOLS: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__engines_count__",
]

__all__: list[str] = [
    *_METADATA_SYMBOLS,
    # Engine 1: Financed Emissions
    *_ENGINE_1_SYMBOLS,
    # Engine 2: Insurance Underwriting
    *_ENGINE_2_SYMBOLS,
    # Engine 3: Green Asset Ratio
    *_ENGINE_3_SYMBOLS,
    # Engine 4: BTAR Calculator
    *_ENGINE_4_SYMBOLS,
    # Engine 5: Climate Risk Scoring
    *_ENGINE_5_SYMBOLS,
    # Engine 6: FS Double Materiality
    *_ENGINE_6_SYMBOLS,
    # Engine 7: FS Transition Plan
    *_ENGINE_7_SYMBOLS,
    # Engine 8: Pillar 3 ESG
    *_ENGINE_8_SYMBOLS,
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully."""
    return len(_loaded_engines)


logger.info(
    "PACK-012 CSRD Financial Service engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
