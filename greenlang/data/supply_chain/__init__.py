"""
GreenLang Supply Chain Mapping Module.

This module provides comprehensive multi-tier supply chain mapping infrastructure
for Scope 3 emissions tracking and EUDR (EU Deforestation Regulation) compliance.

Key Components:
- Entity models for suppliers, facilities, products, and relationships
- Entity resolution with fuzzy matching and external identifier integration
- Supply chain graph analysis using NetworkX
- EUDR traceability and chain of custody tracking
- Scope 3 emission allocation (spend-based, activity-based, hybrid)
- Data connectors for SAP Ariba, Coupa, and file imports
- Visualization exports for D3.js and geographic mapping
- Risk assessment for environmental, social, and concentration risks
"""

from greenlang.supply_chain.models.entity import (
    Supplier,
    Facility,
    Product,
    Material,
    SupplierRelationship,
    RelationshipType,
    SupplierTier,
)
from greenlang.supply_chain.resolution.entity_resolver import (
    EntityResolver,
    MatchResult,
    MatchConfidence,
)
from greenlang.supply_chain.graph.supply_chain_graph import (
    SupplyChainGraph,
    MaterialFlow,
    SupplyChainPath,
)
from greenlang.supply_chain.eudr.traceability import (
    EUDRTraceabilityManager,
    PlotRecord,
    ChainOfCustodyRecord,
    DueDiligenceStatement,
)
from greenlang.supply_chain.scope3.emission_allocation import (
    Scope3Allocator,
    AllocationMethod,
    EmissionAllocation,
    Scope3Category,
)
from greenlang.supply_chain.risk.supply_chain_risk import (
    SupplyChainRiskAssessor,
    RiskScore,
    RiskCategory,
    RiskLevel,
)

__version__ = "1.0.0"
__all__ = [
    # Entity models
    "Supplier",
    "Facility",
    "Product",
    "Material",
    "SupplierRelationship",
    "RelationshipType",
    "SupplierTier",
    # Entity resolution
    "EntityResolver",
    "MatchResult",
    "MatchConfidence",
    # Supply chain graph
    "SupplyChainGraph",
    "MaterialFlow",
    "SupplyChainPath",
    # EUDR traceability
    "EUDRTraceabilityManager",
    "PlotRecord",
    "ChainOfCustodyRecord",
    "DueDiligenceStatement",
    # Scope 3 allocation
    "Scope3Allocator",
    "AllocationMethod",
    "EmissionAllocation",
    "Scope3Category",
    # Risk assessment
    "SupplyChainRiskAssessor",
    "RiskScore",
    "RiskCategory",
    "RiskLevel",
]
