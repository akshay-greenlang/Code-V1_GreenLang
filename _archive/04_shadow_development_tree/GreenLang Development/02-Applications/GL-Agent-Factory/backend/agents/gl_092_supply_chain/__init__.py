"""GL-092: Supply Chain Integrator Agent (SUPPLY-CHAIN-LINK)"""

from .agent import (
    SupplyChainIntegratorAgent,
    SupplyChainInput,
    MaterialInventory,
    SupplierPerformance,
    DemandForecast,
    LogisticsRoute,
    SupplyChainOutput,
    InventoryOptimization,
    SupplierRecommendation,
    SupplyChainRisk,
    LogisticsOptimization,
    ProvenanceRecord,
    SupplierTier,
    InventoryStatus,
    RiskType,
    OptimizationType,
    PACK_SPEC,
)

__all__ = [
    "SupplyChainIntegratorAgent",
    "SupplyChainInput",
    "MaterialInventory",
    "SupplierPerformance",
    "DemandForecast",
    "LogisticsRoute",
    "SupplyChainOutput",
    "InventoryOptimization",
    "SupplierRecommendation",
    "SupplyChainRisk",
    "LogisticsOptimization",
    "ProvenanceRecord",
    "SupplierTier",
    "InventoryStatus",
    "RiskType",
    "OptimizationType",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-092"
__agent_name__ = "SUPPLY-CHAIN-LINK"
