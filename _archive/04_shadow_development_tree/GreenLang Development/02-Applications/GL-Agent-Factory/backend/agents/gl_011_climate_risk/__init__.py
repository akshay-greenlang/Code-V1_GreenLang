"""
GL-011: Climate Risk Assessment Agent

TCFD-aligned climate risk assessment for physical and transition risks
using zero-hallucination deterministic calculations.

Features:
- Physical Risk Assessment (Acute and Chronic)
  - Acute: floods, cyclones, wildfires, extreme heat, drought
  - Chronic: sea level rise, temperature increase, precipitation changes
- Transition Risk Assessment
  - Policy: carbon pricing, regulations, mandates
  - Technology: disruption, obsolescence
  - Market: demand shifts, commodity prices
  - Reputation: stakeholder concerns, stigmatization
- Scenario Analysis (IPCC Pathways)
  - RCP 2.6, 4.5, 6.0, 8.5
  - SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5
- Financial Impact Quantification
- TCFD-Aligned Outputs

Example:
    >>> from gl_011_climate_risk import ClimateRiskAgent, ClimateRiskInput
    >>> from gl_011_climate_risk import GeoLocation, Asset, AssetType, ClimateScenario
    >>>
    >>> agent = ClimateRiskAgent()
    >>> result = agent.run(ClimateRiskInput(
    ...     organization_name="Example Corp",
    ...     assets=[Asset(
    ...         name="HQ Building",
    ...         asset_type=AssetType.REAL_ESTATE,
    ...         value_usd=50_000_000
    ...     )],
    ...     location=GeoLocation(
    ...         latitude=25.7617,
    ...         longitude=-80.1918,
    ...         country="US"
    ...     ),
    ...     time_horizon_years=10,
    ...     scenario=ClimateScenario.RCP_4_5
    ... ))
    >>> print(f"Total risk score: {result.total_risk_score}")
    >>> print(f"Risk level: {result.overall_risk_level}")
"""

from .agent import (
    # Main Agent
    ClimateRiskAgent,
    # Input Models
    ClimateRiskInput,
    GeoLocation,
    Asset,
    RevenueStream,
    CarbonExposure,
    MitigationMeasure,
    # Output Models
    ClimateRiskOutput,
    RiskScore,
    PhysicalRiskAssessment,
    TransitionRiskAssessment,
    ScenarioImpact,
    FinancialExposure,
    RiskRegister,
    ResilienceRecommendation,
    # Enums
    PhysicalRiskType,
    TransitionRiskType,
    ClimateScenario,
    TimeHorizon,
    RiskCategory,
    AssetType,
    SectorType,
    # Reference Data
    SCENARIO_PARAMETERS,
    SECTOR_TRANSITION_SENSITIVITY,
    PHYSICAL_RISK_BASELINE,
    IMPACT_MULTIPLIERS,
    # Pack Spec
    PACK_SPEC,
)

__all__ = [
    # Main Agent
    "ClimateRiskAgent",
    # Input Models
    "ClimateRiskInput",
    "GeoLocation",
    "Asset",
    "RevenueStream",
    "CarbonExposure",
    "MitigationMeasure",
    # Output Models
    "ClimateRiskOutput",
    "RiskScore",
    "PhysicalRiskAssessment",
    "TransitionRiskAssessment",
    "ScenarioImpact",
    "FinancialExposure",
    "RiskRegister",
    "ResilienceRecommendation",
    # Enums
    "PhysicalRiskType",
    "TransitionRiskType",
    "ClimateScenario",
    "TimeHorizon",
    "RiskCategory",
    "AssetType",
    "SectorType",
    # Reference Data
    "SCENARIO_PARAMETERS",
    "SECTOR_TRANSITION_SENSITIVITY",
    "PHYSICAL_RISK_BASELINE",
    "IMPACT_MULTIPLIERS",
    # Pack Spec
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "risk/climate_risk_v1"
