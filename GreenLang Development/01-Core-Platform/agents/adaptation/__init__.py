# -*- coding: utf-8 -*-
"""
GreenLang Climate Risk & Adaptation Layer
==========================================

Core agents for climate risk assessment and adaptation planning.

Core Agents (GL-ADAPT-X-001 to GL-ADAPT-X-012):
    GL-ADAPT-X-001: Physical Risk Screening Agent
    GL-ADAPT-X-002: Hazard Mapping Agent
    GL-ADAPT-X-003: Vulnerability Assessment Agent
    GL-ADAPT-X-004: Exposure Analysis Agent
    GL-ADAPT-X-005: Adaptation Options Library Agent
    GL-ADAPT-X-006: Resilience Scoring Agent
    GL-ADAPT-X-007: Climate Scenario Agent
    GL-ADAPT-X-008: Financial Impact Agent
    GL-ADAPT-X-009: Insurance & Risk Transfer Agent
    GL-ADAPT-X-010: Adaptation Investment Prioritizer Agent
    GL-ADAPT-X-011: TCFD Alignment Agent
    GL-ADAPT-X-012: Nature-Based Adaptation Agent

Subpackages:
    - energy: Power infrastructure climate resilience
    - public: Public sector adaptation agents

All agents follow GreenLang's zero-hallucination principles with
deterministic calculations and complete provenance tracking.

Author: GreenLang Team
Version: 1.0.0
"""

# Public sector adaptation agents (existing)
from greenlang.agents.adaptation.public import (
    UrbanHeatActionAgent,
    FloodResponsePlanningAgent,
    CriticalInfrastructureProtectionAgent,
    PublicHealthClimateAgent,
    EmergencyServicesAdaptationAgent,
    CommunityResiliencePlannerAgent,
)

# GL-ADAPT-X-001: Physical Risk Screening Agent
from greenlang.agents.adaptation.physical_risk_screening import (
    PhysicalRiskScreeningAgent,
    HazardType,
    RiskCategory,
    TimeHorizon,
    ClimateScenario,
    AssetType,
    GeoLocation,
    AssetDefinition,
    HazardExposure,
    HazardRiskScore,
    AssetRiskProfile,
    PhysicalRiskScreeningInput,
    PhysicalRiskScreeningOutput,
    RISK_THRESHOLDS,
)

# GL-ADAPT-X-002: Hazard Mapping Agent
from greenlang.agents.adaptation.hazard_mapping import (
    HazardMappingAgent,
    HazardCategory,
    HazardSeverity,
    DataResolution,
    HazardSource,
    BoundingBox,
    GridCell,
    HazardIntensity,
    HazardMapCell,
    HazardLayer,
    HazardMappingInput,
    HazardMappingOutput,
    SEVERITY_THRESHOLDS,
)

# GL-ADAPT-X-003: Vulnerability Assessment Agent
from greenlang.agents.adaptation.vulnerability_assessment import (
    VulnerabilityAssessmentAgent,
    VulnerabilityLevel,
    SensitivityFactor,
    AdaptiveCapacityFactor,
    SectorType,
    SensitivityScore,
    AdaptiveCapacityScore,
    VulnerabilityComponent,
    VulnerabilityResult,
    AssetVulnerabilityInput,
    VulnerabilityAssessmentInput,
    VulnerabilityAssessmentOutput,
    VULNERABILITY_THRESHOLDS,
    SECTOR_SENSITIVITY,
)

# GL-ADAPT-X-004: Exposure Analysis Agent
from greenlang.agents.adaptation.exposure_analysis import (
    ExposureAnalysisAgent,
    ExposureType,
    ExposureLevel,
    ValueAtRiskCategory,
    GeographicExposure,
    ValueChainExposure,
    RevenueExposure,
    WorkforceExposure,
    AssetExposureInput,
    ExposureAnalysisInput,
    ExposureResult,
    ExposureAnalysisOutput,
    EXPOSURE_THRESHOLDS,
)

# GL-ADAPT-X-005: Adaptation Options Library Agent
from greenlang.agents.adaptation.adaptation_options_library import (
    AdaptationOptionsLibraryAgent,
    AdaptationCategory,
    ImplementationScale,
    EffectivenessLevel,
    CostCategory,
    CostEstimate,
    EffectivenessMetrics,
    AdaptationMeasure,
    MeasureMatch,
    LibraryQueryInput,
    LibraryQueryOutput,
)

# GL-ADAPT-X-006: Resilience Scoring Agent
from greenlang.agents.adaptation.resilience_scoring import (
    ResilienceScoringAgent,
    ResilienceCapacity,
    ResilienceLevel,
    ResilienceDimension,
    DimensionScore,
    CapacityScore,
    ResilienceProfile,
    ResilienceInput,
    ResilienceScoringInput,
    ResilienceScoringOutput,
    RESILIENCE_THRESHOLDS,
    SECTOR_BENCHMARKS,
)

# GL-ADAPT-X-007: Climate Scenario Agent
from greenlang.agents.adaptation.climate_scenario import (
    ClimateScenarioAgent,
    ScenarioFamily,
    RCPScenario,
    SSPScenario,
    ProjectionVariable,
    VariableProjection,
    ScenarioProjection,
    LocationProjection,
    ScenarioInput,
    ScenarioOutput,
    TEMPERATURE_PROJECTIONS,
    SEA_LEVEL_PROJECTIONS,
)

# GL-ADAPT-X-008: Financial Impact Agent
from greenlang.agents.adaptation.financial_impact import (
    FinancialImpactAgent,
    ImpactType,
    TimeValue,
    AssetFinancials,
    HazardImpactDetail,
    FinancialImpactResult,
    AdaptationCostBenefit,
    FinancialImpactInput,
    FinancialImpactOutput,
    DAMAGE_FUNCTIONS,
)

# GL-ADAPT-X-009: Insurance & Risk Transfer Agent
from greenlang.agents.adaptation.insurance_transfer import (
    InsuranceTransferAgent,
    TransferType,
    CoverageType,
    CoverageStatus,
    ExistingCoverage,
    CoverageGap,
    TransferOption,
    RiskRetentionAnalysis,
    InsuranceAnalysisInput,
    InsuranceAnalysisOutput,
    PREMIUM_RATE_FACTORS,
)

# GL-ADAPT-X-010: Adaptation Investment Prioritizer Agent
from greenlang.agents.adaptation.adaptation_investment_prioritizer import (
    AdaptationInvestmentPrioritizerAgent,
    PriorityLevel,
    InvestmentCategory,
    TimeFrame,
    InvestmentOption,
    PrioritizedInvestment,
    BudgetAllocation,
    PrioritizationInput,
    PrioritizationOutput,
)

# GL-ADAPT-X-011: TCFD Alignment Agent
from greenlang.agents.adaptation.tcfd_alignment import (
    TCFDAlignmentAgent,
    TCFDPillar,
    AlignmentLevel,
    DisclosureStatus,
    DisclosureItem,
    PillarAssessment,
    ScenarioAnalysisAssessment,
    TCFDAlignmentInput,
    TCFDAlignmentOutput,
    TCFD_DISCLOSURES,
)

# GL-ADAPT-X-012: Nature-Based Adaptation Agent
from greenlang.agents.adaptation.nature_based_adaptation import (
    NatureBasedAdaptationAgent,
    NbSCategory,
    EcosystemType,
    EcosystemService,
    FeasibilityLevel,
    NbSSolution,
    SolutionMatch,
    EcosystemServiceValuation,
    NbSInput,
    NbSOutput,
    ECOSYSTEM_SERVICE_VALUES,
    CARBON_SEQUESTRATION_RATES,
)


# Agent registry for the adaptation layer core agents
ADAPTATION_CORE_AGENTS = {
    "GL-ADAPT-X-001": PhysicalRiskScreeningAgent,
    "GL-ADAPT-X-002": HazardMappingAgent,
    "GL-ADAPT-X-003": VulnerabilityAssessmentAgent,
    "GL-ADAPT-X-004": ExposureAnalysisAgent,
    "GL-ADAPT-X-005": AdaptationOptionsLibraryAgent,
    "GL-ADAPT-X-006": ResilienceScoringAgent,
    "GL-ADAPT-X-007": ClimateScenarioAgent,
    "GL-ADAPT-X-008": FinancialImpactAgent,
    "GL-ADAPT-X-009": InsuranceTransferAgent,
    "GL-ADAPT-X-010": AdaptationInvestmentPrioritizerAgent,
    "GL-ADAPT-X-011": TCFDAlignmentAgent,
    "GL-ADAPT-X-012": NatureBasedAdaptationAgent,
}


__version__ = "1.0.0"

__all__ = [
    # Public Sector Adaptation (existing)
    "UrbanHeatActionAgent",
    "FloodResponsePlanningAgent",
    "CriticalInfrastructureProtectionAgent",
    "PublicHealthClimateAgent",
    "EmergencyServicesAdaptationAgent",
    "CommunityResiliencePlannerAgent",

    # Core Agents
    "PhysicalRiskScreeningAgent",
    "HazardMappingAgent",
    "VulnerabilityAssessmentAgent",
    "ExposureAnalysisAgent",
    "AdaptationOptionsLibraryAgent",
    "ResilienceScoringAgent",
    "ClimateScenarioAgent",
    "FinancialImpactAgent",
    "InsuranceTransferAgent",
    "AdaptationInvestmentPrioritizerAgent",
    "TCFDAlignmentAgent",
    "NatureBasedAdaptationAgent",

    # Registry
    "ADAPTATION_CORE_AGENTS",

    # Physical Risk Screening
    "HazardType",
    "RiskCategory",
    "TimeHorizon",
    "ClimateScenario",
    "AssetType",
    "GeoLocation",
    "AssetDefinition",
    "HazardExposure",
    "HazardRiskScore",
    "AssetRiskProfile",
    "PhysicalRiskScreeningInput",
    "PhysicalRiskScreeningOutput",
    "RISK_THRESHOLDS",

    # Hazard Mapping
    "HazardCategory",
    "HazardSeverity",
    "DataResolution",
    "HazardSource",
    "BoundingBox",
    "GridCell",
    "HazardIntensity",
    "HazardMapCell",
    "HazardLayer",
    "HazardMappingInput",
    "HazardMappingOutput",
    "SEVERITY_THRESHOLDS",

    # Vulnerability Assessment
    "VulnerabilityLevel",
    "SensitivityFactor",
    "AdaptiveCapacityFactor",
    "SectorType",
    "SensitivityScore",
    "AdaptiveCapacityScore",
    "VulnerabilityComponent",
    "VulnerabilityResult",
    "AssetVulnerabilityInput",
    "VulnerabilityAssessmentInput",
    "VulnerabilityAssessmentOutput",
    "VULNERABILITY_THRESHOLDS",
    "SECTOR_SENSITIVITY",

    # Exposure Analysis
    "ExposureType",
    "ExposureLevel",
    "ValueAtRiskCategory",
    "GeographicExposure",
    "ValueChainExposure",
    "RevenueExposure",
    "WorkforceExposure",
    "AssetExposureInput",
    "ExposureAnalysisInput",
    "ExposureResult",
    "ExposureAnalysisOutput",
    "EXPOSURE_THRESHOLDS",

    # Adaptation Options Library
    "AdaptationCategory",
    "ImplementationScale",
    "EffectivenessLevel",
    "CostCategory",
    "CostEstimate",
    "EffectivenessMetrics",
    "AdaptationMeasure",
    "MeasureMatch",
    "LibraryQueryInput",
    "LibraryQueryOutput",

    # Resilience Scoring
    "ResilienceCapacity",
    "ResilienceLevel",
    "ResilienceDimension",
    "DimensionScore",
    "CapacityScore",
    "ResilienceProfile",
    "ResilienceInput",
    "ResilienceScoringInput",
    "ResilienceScoringOutput",
    "RESILIENCE_THRESHOLDS",
    "SECTOR_BENCHMARKS",

    # Climate Scenario
    "ScenarioFamily",
    "RCPScenario",
    "SSPScenario",
    "ProjectionVariable",
    "VariableProjection",
    "ScenarioProjection",
    "LocationProjection",
    "ScenarioInput",
    "ScenarioOutput",
    "TEMPERATURE_PROJECTIONS",
    "SEA_LEVEL_PROJECTIONS",

    # Financial Impact
    "ImpactType",
    "TimeValue",
    "AssetFinancials",
    "HazardImpactDetail",
    "FinancialImpactResult",
    "AdaptationCostBenefit",
    "FinancialImpactInput",
    "FinancialImpactOutput",
    "DAMAGE_FUNCTIONS",

    # Insurance & Risk Transfer
    "TransferType",
    "CoverageType",
    "CoverageStatus",
    "ExistingCoverage",
    "CoverageGap",
    "TransferOption",
    "RiskRetentionAnalysis",
    "InsuranceAnalysisInput",
    "InsuranceAnalysisOutput",
    "PREMIUM_RATE_FACTORS",

    # Investment Prioritization
    "PriorityLevel",
    "InvestmentCategory",
    "TimeFrame",
    "InvestmentOption",
    "PrioritizedInvestment",
    "BudgetAllocation",
    "PrioritizationInput",
    "PrioritizationOutput",

    # TCFD Alignment
    "TCFDPillar",
    "AlignmentLevel",
    "DisclosureStatus",
    "DisclosureItem",
    "PillarAssessment",
    "ScenarioAnalysisAssessment",
    "TCFDAlignmentInput",
    "TCFDAlignmentOutput",
    "TCFD_DISCLOSURES",

    # Nature-Based Adaptation
    "NbSCategory",
    "EcosystemType",
    "EcosystemService",
    "FeasibilityLevel",
    "NbSSolution",
    "SolutionMatch",
    "EcosystemServiceValuation",
    "NbSInput",
    "NbSOutput",
    "ECOSYSTEM_SERVICE_VALUES",
    "CARBON_SEQUESTRATION_RATES",
]
