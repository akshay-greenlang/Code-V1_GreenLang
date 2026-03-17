"""
PACK-013 CSRD Manufacturing Pack - Workflows module.

Provides eight workflow orchestrators for CSRD manufacturing compliance
reporting with full lifecycle emissions, circular economy, and BAT assessment.

Workflows:
    1. ManufacturingEmissionsWorkflow - Scope 1/2/3 emissions (4 phases)
    2. ProductPCFWorkflow - Product carbon footprint (5 phases)
    3. CircularEconomyWorkflow - Circular economy readiness (4 phases)
    4. BATComplianceWorkflow - BAT/BREF compliance (4 phases)
    5. SupplyChainAssessmentWorkflow - Supply chain Scope 3 (5 phases)
    6. ESRSManufacturingWorkflow - ESRS disclosure generation (4 phases)
    7. DecarbonizationRoadmapWorkflow - Decarbonization planning (5 phases)
    8. RegulatoryComplianceWorkflow - Multi-regulation compliance (3 phases)
"""

from .manufacturing_emissions_workflow import (
    ManufacturingEmissionsInput,
    ManufacturingEmissionsResult,
    ManufacturingEmissionsWorkflow,
)
from .product_pcf_workflow import (
    ProductPCFInput,
    ProductPCFResult,
    ProductPCFWorkflow,
)
from .circular_economy_workflow import (
    CircularEconomyInput,
    CircularEconomyResult,
    CircularEconomyWorkflow,
)
from .bat_compliance_workflow import (
    BATComplianceInput,
    BATComplianceResult,
    BATComplianceWorkflow,
)
from .supply_chain_assessment_workflow import (
    SupplyChainInput,
    SupplyChainResult,
    SupplyChainAssessmentWorkflow,
)
from .esrs_manufacturing_workflow import (
    ESRSManufacturingInput,
    ESRSManufacturingResult,
    ESRSManufacturingWorkflow,
)
from .decarbonization_roadmap_workflow import (
    DecarbonizationInput,
    DecarbonizationResult,
    DecarbonizationRoadmapWorkflow,
)
from .regulatory_compliance_workflow import (
    RegulatoryComplianceInput,
    RegulatoryComplianceResult,
    RegulatoryComplianceWorkflow,
)

__all__ = [
    # Manufacturing Emissions (4-phase)
    "ManufacturingEmissionsWorkflow",
    "ManufacturingEmissionsInput",
    "ManufacturingEmissionsResult",
    # Product PCF (5-phase)
    "ProductPCFWorkflow",
    "ProductPCFInput",
    "ProductPCFResult",
    # Circular Economy (4-phase)
    "CircularEconomyWorkflow",
    "CircularEconomyInput",
    "CircularEconomyResult",
    # BAT Compliance (4-phase)
    "BATComplianceWorkflow",
    "BATComplianceInput",
    "BATComplianceResult",
    # Supply Chain Assessment (5-phase)
    "SupplyChainAssessmentWorkflow",
    "SupplyChainInput",
    "SupplyChainResult",
    # ESRS Manufacturing (4-phase)
    "ESRSManufacturingWorkflow",
    "ESRSManufacturingInput",
    "ESRSManufacturingResult",
    # Decarbonization Roadmap (5-phase)
    "DecarbonizationRoadmapWorkflow",
    "DecarbonizationInput",
    "DecarbonizationResult",
    # Regulatory Compliance (3-phase)
    "RegulatoryComplianceWorkflow",
    "RegulatoryComplianceInput",
    "RegulatoryComplianceResult",
]
