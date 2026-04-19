"""
GL-009: Product Carbon Footprint Agent

This module provides Product Carbon Footprint (PCF) calculation capabilities
following ISO 14067, EU PEF methodology, and PACT Pathfinder standards.

Key Features:
- Cradle-to-gate and cradle-to-grave boundaries
- PEF 16 impact categories
- Circular Footprint Formula (CFF) for end-of-life
- PACT Pathfinder 2.1 data exchange
- Catena-X PCF data model
- EU Battery Regulation passport

Example:
    >>> from agents.gl_009_product_carbon_footprint import (
    ...     ProductCarbonFootprintAgent,
    ...     PCFInput,
    ...     BOMItem,
    ...     MaterialCategory,
    ...     PCFBoundary,
    ... )
    >>> agent = ProductCarbonFootprintAgent()
    >>> result = agent.run(PCFInput(
    ...     product_id="PROD-001",
    ...     bill_of_materials=[
    ...         BOMItem(
    ...             material_id="STEEL-001",
    ...             material_category=MaterialCategory.STEEL_PRIMARY,
    ...             quantity_kg=10.0
    ...         )
    ...     ],
    ...     boundary=PCFBoundary.CRADLE_TO_GATE
    ... ))
    >>> print(f"Total PCF: {result.total_co2e} kgCO2e")
"""

from .agent import (
    # Main agent
    ProductCarbonFootprintAgent,
    # Input models
    PCFInput,
    BOMItem,
    ManufacturingEnergy,
    ProcessEmissions,
    TransportLeg,
    TransportData,
    UsePhaseData,
    EndOfLifeData,
    # Output models
    PCFOutput,
    ImpactCategories,
    LifecycleStageBreakdown,
    PACTPathfinderExport,
    CatenaXPCFExport,
    BatteryPassportExport,
    # Enums
    PCFBoundary,
    TransportMode,
    EndOfLifeTreatment,
    DataQualityLevel,
    MaterialCategory,
    # Factor models
    MaterialEmissionFactor,
    GridEmissionFactor,
    # Pack spec
    PACK_SPEC,
)

__all__ = [
    # Main agent
    "ProductCarbonFootprintAgent",
    # Input models
    "PCFInput",
    "BOMItem",
    "ManufacturingEnergy",
    "ProcessEmissions",
    "TransportLeg",
    "TransportData",
    "UsePhaseData",
    "EndOfLifeData",
    # Output models
    "PCFOutput",
    "ImpactCategories",
    "LifecycleStageBreakdown",
    "PACTPathfinderExport",
    "CatenaXPCFExport",
    "BatteryPassportExport",
    # Enums
    "PCFBoundary",
    "TransportMode",
    "EndOfLifeTreatment",
    "DataQualityLevel",
    "MaterialCategory",
    # Factor models
    "MaterialEmissionFactor",
    "GridEmissionFactor",
    # Pack spec
    "PACK_SPEC",
]

__version__ = "1.0.0"
