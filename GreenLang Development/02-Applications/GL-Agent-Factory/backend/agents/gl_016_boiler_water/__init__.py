"""
GL-016 WATERGUARD - Boiler Water Treatment Agent

This module provides the BoilerWaterTreatmentAgent for ASME/ABMA compliant
boiler water chemistry optimization, blowdown control, and chemical dosing.

The agent implements:
- ASME/ABMA water chemistry compliance checking
- Cycles of concentration optimization
- Blowdown rate optimization
- Chemical dosing recommendations with ML-optimized schedules
- SHAP/LIME explainability for all recommendations
- SHA-256 provenance tracking for audit trails

Agent ID: GL-016
Agent Name: WATERGUARD
Version: 1.0.0
Category: Boiler Systems
Type: Controller
Priority: P1
Market Size: $5B

Standards Reference:
    - ASME Boiler and Pressure Vessel Code Section VII
    - ABMA (American Boiler Manufacturers Association) Guidelines
    - EPRI Water Chemistry Guidelines for Fossil Plants
    - NACE SP0590 Standard Practice

Example:
    >>> from gl_016_boiler_water import BoilerWaterTreatmentAgent
    >>> from gl_016_boiler_water.schemas import (
    ...     WaterTreatmentInput,
    ...     WaterChemistryData,
    ...     FeedwaterQuality,
    ...     OperatingParameters,
    ... )
    >>>
    >>> agent = BoilerWaterTreatmentAgent()
    >>> input_data = WaterTreatmentInput(
    ...     boiler_id="BLR-001",
    ...     boiler_water_chemistry=WaterChemistryData(
    ...         conductivity_us_cm=3000,
    ...         ph=10.5,
    ...         alkalinity_ppm_caco3=300,
    ...         silica_ppm=25,
    ...         total_hardness_ppm=0.1,
    ...         iron_ppm=0.3,
    ...         copper_ppm=0.05,
    ...         dissolved_oxygen_ppb=10,
    ...     ),
    ...     feedwater_quality=FeedwaterQuality(
    ...         conductivity_us_cm=200,
    ...         hardness_ppm=0.5,
    ...         silica_ppm=5,
    ...         dissolved_oxygen_ppb=100,
    ...         ph=8.5,
    ...     ),
    ...     operating_parameters=OperatingParameters(
    ...         operating_pressure_psig=600,
    ...         steam_production_rate_lb_hr=50000,
    ...         feedwater_flow_gpm=100,
    ...     ),
    ... )
    >>> result = agent.run(input_data)
    >>> print(f"Compliance: {result.overall_compliance_status}")
    >>> print(f"Optimal COC: {result.optimal_cycles:.1f}")
    >>> print(f"Annual savings: ${result.total_cost_savings_per_year:,.0f}")
"""

from .agent import BoilerWaterTreatmentAgent
from .schemas import (
    # Input/Output models
    WaterTreatmentInput,
    WaterTreatmentOutput,
    # Sub-models for input
    WaterChemistryData,
    FeedwaterQuality,
    OperatingParameters,
    ChemicalInventory,
    # Sub-models for output
    ChemistryLimitResult,
    BlowdownRecommendation,
    DosingRecommendation,
    ChemistryTrend,
    ExplainabilityReport,
    # Enums
    BoilerPressureClass,
    ChemicalType,
    ComplianceStatus,
    DosingPriority,
    # Configuration
    AgentConfig,
)

# Agent metadata
AGENT_ID = "GL-016"
AGENT_NAME = "WATERGUARD"
VERSION = "1.0.0"
DESCRIPTION = "ASME/ABMA compliant boiler water chemistry optimization"
CATEGORY = "Boiler Systems"
AGENT_TYPE = "Controller"
PRIORITY = "P1"
MARKET_SIZE = "$5B"
STANDARDS = [
    "ASME BPVC Section VII",
    "ABMA Guidelines",
    "EPRI Water Chemistry Guidelines",
    "NACE SP0590",
]

__all__ = [
    # Agent class
    "BoilerWaterTreatmentAgent",
    # Input/Output models
    "WaterTreatmentInput",
    "WaterTreatmentOutput",
    # Sub-models
    "WaterChemistryData",
    "FeedwaterQuality",
    "OperatingParameters",
    "ChemicalInventory",
    "ChemistryLimitResult",
    "BlowdownRecommendation",
    "DosingRecommendation",
    "ChemistryTrend",
    "ExplainabilityReport",
    # Enums
    "BoilerPressureClass",
    "ChemicalType",
    "ComplianceStatus",
    "DosingPriority",
    # Configuration
    "AgentConfig",
    # Metadata
    "AGENT_ID",
    "AGENT_NAME",
    "VERSION",
    "DESCRIPTION",
    "CATEGORY",
    "AGENT_TYPE",
    "PRIORITY",
    "MARKET_SIZE",
    "STANDARDS",
]
