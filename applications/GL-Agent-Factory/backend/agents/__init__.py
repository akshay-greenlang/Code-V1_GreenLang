"""
GreenLang Agent Factory - Production Agents

This package contains all production-ready agent implementations
for regulatory compliance and emissions calculations.

Available Agents:
- GL-001: Carbon Emissions Calculator
- GL-002: CBAM Compliance Agent
- GL-003: CSRD Reporting Agent
- GL-004: EUDR Compliance Agent
- GL-005: Building Energy Agent
- GL-006: Scope 3 Emissions Agent
- GL-007: EU Taxonomy Agent
- GL-008: Green Claims Directive Agent
- GL-009: Product Carbon Footprint Agent
- GL-010: SBTi Validation Agent
- GL-011: Climate Risk Assessment Agent
- GL-012: Carbon Offset Verification Agent
- GL-013: SB 253 Climate Disclosure Agent

All agents follow GreenLang's zero-hallucination principle:
- Deterministic calculations only
- No LLM in calculation path
- Complete SHA-256 provenance tracking
- Pydantic-validated inputs/outputs
"""

from .gl_001_carbon_emissions import (
    CarbonEmissionsAgent,
    CarbonEmissionsInput,
    CarbonEmissionsOutput,
)
from .gl_002_cbam_compliance import (
    CBAMComplianceAgent,
    CBAMInput,
    CBAMOutput,
)
from .gl_003_csrd_reporting import (
    CSRDReportingAgent,
    CSRDInput,
    CSRDOutput,
)
from .gl_004_eudr_compliance import (
    EUDRComplianceAgent,
    EUDRInput,
    EUDROutput,
)
from .gl_005_building_energy import (
    BuildingEnergyAgent,
    BuildingEnergyInput,
    BuildingEnergyOutput,
)
from .gl_006_scope3_emissions import (
    Scope3EmissionsAgent,
    Scope3Input,
    Scope3Output,
)
from .gl_007_eu_taxonomy import (
    EUTaxonomyAgent,
    TaxonomyInput,
    TaxonomyOutput,
)
from .gl_008_green_claims import (
    GreenClaimsAgent,
    GreenClaimsInput,
    GreenClaimsOutput,
    ClaimType,
    EvidenceItem,
    GreenwashingRedFlag,
)
from .gl_009_product_carbon_footprint import (
    ProductCarbonFootprintAgent,
    PCFInput,
    PCFOutput,
    BOMItem,
    MaterialCategory,
    PCFBoundary,
)
from .gl_010_sbti_validation import (
    SBTiValidationAgent,
    SBTiInput,
    SBTiOutput,
    ScopeEmissions,
    TargetDefinition,
    IntensityMetric,
    Scope3EngagementTarget,
    NeutralizationPlan,
    FLAGTarget,
    CurrentProgress,
    ValidationResult,
    TargetValidation,
    ProgressTracking,
    Recommendation,
    PathwayCalculation,
    TargetTrajectory,
    TargetTrajectoryPoint,
    NetZeroValidation,
    FLAGValidation,
    TargetType,
    AmbitionLevel,
    PathwayType,
    SectorPathway,
    ValidationStatus,
    ScopeType,
    Scope3EngagementType,
    NeutralizationType,
    ProgressStatus,
    SBTiPathwayConstants,
)
from .gl_011_climate_risk import (
    ClimateRiskAgent,
    ClimateRiskInput,
    ClimateRiskOutput,
    GeoLocation,
    Asset,
    RevenueStream,
    CarbonExposure,
    MitigationMeasure,
    PhysicalRiskType,
    TransitionRiskType,
    ClimateScenario,
    TimeHorizon,
    RiskCategory,
    AssetType,
    SectorType,
)
from .gl_012_carbon_offset import (
    CarbonOffsetAgent,
    CarbonOffsetInput,
    CarbonOffsetOutput,
    CarbonRegistry,
    ProjectType,
    VerificationStatus,
    RiskLevel,
    RetirementStatus,
    CorrespondingAdjustmentStatus,
    Article6AuthorizationStatus,
    CreditCategory,
    PriceTier,
    CarbonCredit,
    ProjectDetails,
    ICVCMScoreBreakdown,
    VerificationCheck,
    CreditVerificationResult,
    RiskAssessment,
    RegistryCreditVerification,
    Article6Compliance,
    PortfolioAnalysis,
    PriceBenchmark,
    PortfolioPriceSummary,
    REGISTRY_STANDARDS,
    PROJECT_TYPE_PROFILES,
    PRICE_BENCHMARKS,
    REGISTRY_API_ENDPOINTS,
    REMOVAL_PROJECT_TYPES,
    PACK_SPEC,
)
from .gl_013_sb253_disclosure import (
    SB253DisclosureAgent,
    SB253ReportInput,
    SB253ReportOutput,
    CompanyInfo,
    FacilityInfo,
    Scope1Source,
    Scope2Source,
    Scope3Data,
    Scope3CategoryData,
    Scope1Result,
    Scope2Result,
    Scope3Result,
    AssurancePackage,
    CARBFilingData,
    OrganizationalBoundary,
    FuelType,
    FuelUnit,
    SourceCategory,
    Scope3Category,
    CalculationMethod,
    DataQualityScore,
    GWPSet,
    AssuranceLevel,
    RefrigerantType,
)

__all__ = [
    # GL-001
    "CarbonEmissionsAgent",
    "CarbonEmissionsInput",
    "CarbonEmissionsOutput",
    # GL-002
    "CBAMComplianceAgent",
    "CBAMInput",
    "CBAMOutput",
    # GL-003
    "CSRDReportingAgent",
    "CSRDInput",
    "CSRDOutput",
    # GL-004
    "EUDRComplianceAgent",
    "EUDRInput",
    "EUDROutput",
    # GL-005
    "BuildingEnergyAgent",
    "BuildingEnergyInput",
    "BuildingEnergyOutput",
    # GL-006
    "Scope3EmissionsAgent",
    "Scope3Input",
    "Scope3Output",
    # GL-007
    "EUTaxonomyAgent",
    "TaxonomyInput",
    "TaxonomyOutput",
    # GL-008
    "GreenClaimsAgent",
    "GreenClaimsInput",
    "GreenClaimsOutput",
    "ClaimType",
    "EvidenceItem",
    "GreenwashingRedFlag",
    # GL-009
    "ProductCarbonFootprintAgent",
    "PCFInput",
    "PCFOutput",
    "BOMItem",
    "MaterialCategory",
    "PCFBoundary",
    # GL-010
    "SBTiValidationAgent",
    "SBTiInput",
    "SBTiOutput",
    "ScopeEmissions",
    "TargetDefinition",
    "IntensityMetric",
    "Scope3EngagementTarget",
    "NeutralizationPlan",
    "FLAGTarget",
    "CurrentProgress",
    "ValidationResult",
    "TargetValidation",
    "ProgressTracking",
    "Recommendation",
    "PathwayCalculation",
    "TargetTrajectory",
    "TargetTrajectoryPoint",
    "NetZeroValidation",
    "FLAGValidation",
    "TargetType",
    "AmbitionLevel",
    "PathwayType",
    "SectorPathway",
    "ValidationStatus",
    "ScopeType",
    "Scope3EngagementType",
    "NeutralizationType",
    "ProgressStatus",
    "SBTiPathwayConstants",
    # GL-011
    "ClimateRiskAgent",
    "ClimateRiskInput",
    "ClimateRiskOutput",
    "GeoLocation",
    "Asset",
    "RevenueStream",
    "CarbonExposure",
    "MitigationMeasure",
    "PhysicalRiskType",
    "TransitionRiskType",
    "ClimateScenario",
    "TimeHorizon",
    "RiskCategory",
    "AssetType",
    "SectorType",
    # GL-012
    "CarbonOffsetAgent",
    "CarbonOffsetInput",
    "CarbonOffsetOutput",
    "CarbonRegistry",
    "ProjectType",
    "VerificationStatus",
    "RiskLevel",
    "RetirementStatus",
    "CorrespondingAdjustmentStatus",
    "Article6AuthorizationStatus",
    "CreditCategory",
    "PriceTier",
    "CarbonCredit",
    "ProjectDetails",
    "ICVCMScoreBreakdown",
    "VerificationCheck",
    "CreditVerificationResult",
    "RiskAssessment",
    "RegistryCreditVerification",
    "Article6Compliance",
    "PortfolioAnalysis",
    "PriceBenchmark",
    "PortfolioPriceSummary",
    "REGISTRY_STANDARDS",
    "PROJECT_TYPE_PROFILES",
    "PRICE_BENCHMARKS",
    "REGISTRY_API_ENDPOINTS",
    "REMOVAL_PROJECT_TYPES",
    "PACK_SPEC",
    # GL-013
    "SB253DisclosureAgent",
    "SB253ReportInput",
    "SB253ReportOutput",
    "CompanyInfo",
    "FacilityInfo",
    "Scope1Source",
    "Scope2Source",
    "Scope3Data",
    "Scope3CategoryData",
    "Scope1Result",
    "Scope2Result",
    "Scope3Result",
    "AssurancePackage",
    "CARBFilingData",
    "OrganizationalBoundary",
    "FuelType",
    "FuelUnit",
    "SourceCategory",
    "Scope3Category",
    "CalculationMethod",
    "DataQualityScore",
    "GWPSet",
    "AssuranceLevel",
    "RefrigerantType",
]

# Agent registry for dynamic loading
AGENT_REGISTRY = {
    "emissions/carbon_calculator_v1": CarbonEmissionsAgent,
    "regulatory/cbam_compliance_v1": CBAMComplianceAgent,
    "regulatory/csrd_reporting_v1": CSRDReportingAgent,
    "regulatory/eudr_compliance_v1": EUDRComplianceAgent,
    "buildings/energy_performance_v1": BuildingEnergyAgent,
    "emissions/scope3_v1": Scope3EmissionsAgent,
    "regulatory/eu_taxonomy_v1": EUTaxonomyAgent,
    "regulatory/green_claims_v1": GreenClaimsAgent,
    "products/carbon_footprint_v1": ProductCarbonFootprintAgent,
    "targets/sbti_validation_v1": SBTiValidationAgent,
    "risk/climate_risk_v1": ClimateRiskAgent,
    "offsets/carbon_verification_v1": CarbonOffsetAgent,
    "regulatory/sb253_disclosure_v1": SB253DisclosureAgent,
}


def get_agent(agent_id: str):
    """
    Get agent class by ID.

    Args:
        agent_id: Agent identifier (e.g., "emissions/carbon_calculator_v1")

    Returns:
        Agent class

    Raises:
        ValueError: If agent not found
    """
    if agent_id not in AGENT_REGISTRY:
        raise ValueError(f"Agent not found: {agent_id}. Available: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[agent_id]


# Import the comprehensive Process Heat registry
try:
    from .registry import (
        AgentRegistry,
        AgentInfo,
        get_registry,
        get_agent as get_process_heat_agent,
        list_agents,
        get_statistics,
        AGENT_DEFINITIONS,
    )

    __all__.extend([
        "AgentRegistry",
        "AgentInfo",
        "get_registry",
        "get_process_heat_agent",
        "list_agents",
        "get_statistics",
        "AGENT_DEFINITIONS",
    ])
except ImportError:
    pass  # Registry not yet available


# GL-Agent-Factory: 100 Process Heat Agents Summary
# =================================================
# GL-001 to GL-013: Climate & Compliance Agents (13 agents)
# GL-020 to GL-021: Process Heat Baseline (2 agents)
# GL-022 to GL-030: Steam System Agents (9 agents)
# GL-031 to GL-045: Safety & Optimization Agents (15 agents)
# GL-046 to GL-060: Analytics Agents (15 agents)
# GL-061 to GL-075: Digital Twin & Simulation Agents (15 agents)
# GL-076 to GL-100: Financial & Business Agents (25 agents)
# Total: 100 agents covering the complete Process Heat domain
