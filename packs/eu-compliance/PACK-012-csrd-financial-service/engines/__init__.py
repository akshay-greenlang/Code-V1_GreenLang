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

from .financed_emissions_engine import (
    FinancedEmissionsEngine,
    FinancedEmissionsConfig,
    AssetClassData,
    HoldingEmissions,
    AttributionResult,
    PortfolioEmissionsResult,
    DataQualityScore,
    EmissionsByAssetClass,
    PCAFAssetClass,
    DataQualityLevel,
)

from .insurance_underwriting_engine import (
    InsuranceUnderwritingEngine,
    UnderwritingConfig,
    PolicyData,
    UnderwritingEmissionsResult,
    LineOfBusinessResult,
    ReinsuranceAdjustment,
    ClaimsEmissions,
    InsuranceLine,
)

from .green_asset_ratio_engine import (
    GreenAssetRatioEngine,
    GARConfig,
    CoveredAssetData,
    GARResult,
    GARBreakdown,
    CounterpartyBreakdown,
    OffBalanceSheetKPI,
    FlowGAR,
    GARScope,
    CounterpartyType,
    EnvironmentalObjective,
)

from .btar_calculator_engine import (
    BTARCalculatorEngine,
    BTARConfig,
    BankingBookData,
    BTARResult,
    EstimationMethodology,
    SectorProxyResult,
    DataCoverageReport,
    BTARvsGARReconciliation,
    EstimationType,
)

from .climate_risk_scoring_engine import (
    ClimateRiskScoringEngine,
    ClimateRiskConfig,
    ExposureData,
    ClimateRiskResult,
    PhysicalRiskScore,
    TransitionRiskScore,
    NGFSScenarioResult,
    StrandedAssetExposure,
    CreditRiskImpact,
    NGFSScenario,
    PhysicalHazard,
    TransitionChannel,
    TimeHorizon,
    RiskLevel,
)

from .fs_double_materiality_engine import (
    FSDoubleMaterialityEngine,
    FSMaterialityConfig,
    MaterialityTopicData,
    FSMaterialityResult,
    IROAssessment,
    FinancedImpactAssessment,
    StakeholderInput,
    MaterialityMatrix,
    DatapointMapping,
    MaterialityDimension,
    IROType,
    ESRSStandard,
)

from .fs_transition_plan_engine import (
    FSTransitionPlanEngine,
    TransitionPlanConfig,
    SectorTargetData,
    TransitionPlanResult,
    SBTiFIAssessment,
    NZBACommitment,
    SectorDecarbPath,
    PhaseOutCommitment,
    CredibilityScore,
    SBTiMethod,
    AllianceType,
)

from .pillar3_esg_engine import (
    Pillar3ESGEngine,
    Pillar3Config,
    BankingBookExposure,
    Pillar3Result,
    TransitionRiskTemplate,
    PhysicalRiskTemplate,
    RealEstateTemplate,
    Top20CarbonExposure,
    TaxonomyAlignmentTemplate,
    QualitativeDisclosure,
    Pillar3TemplateType,
    EPCLabel,
    NACESector,
)

__all__ = [
    # Engine 1: Financed Emissions
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
    # Engine 2: Insurance Underwriting
    "InsuranceUnderwritingEngine",
    "UnderwritingConfig",
    "PolicyData",
    "UnderwritingEmissionsResult",
    "LineOfBusinessResult",
    "ReinsuranceAdjustment",
    "ClaimsEmissions",
    "InsuranceLine",
    # Engine 3: Green Asset Ratio
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
    # Engine 4: BTAR Calculator
    "BTARCalculatorEngine",
    "BTARConfig",
    "BankingBookData",
    "BTARResult",
    "EstimationMethodology",
    "SectorProxyResult",
    "DataCoverageReport",
    "BTARvsGARReconciliation",
    "EstimationType",
    # Engine 5: Climate Risk Scoring
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
    # Engine 6: FS Double Materiality
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
    # Engine 7: FS Transition Plan
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
    # Engine 8: Pillar 3 ESG
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
