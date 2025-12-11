"""GL-082: Hydrogen Production Heat Agent (H2-PRODUCTION-HEAT)"""

from .agent import (
    HydrogenProductionHeatAgent,
    HydrogenProductionInput,
    HydrogenProductionOutput,
    PowerSource,
    FeedstockInfo,
    HeatIntegration,
    ProductionPerformance,
    HeatRecovery,
    EconomicAnalysis,
    CarbonFootprint,
    ProvenanceRecord,
    ProductionMethod,
    HydrogenColor,
    PurityGrade,
    PACK_SPEC,
)

__all__ = [
    "HydrogenProductionHeatAgent",
    "HydrogenProductionInput",
    "HydrogenProductionOutput",
    "PowerSource",
    "FeedstockInfo",
    "HeatIntegration",
    "ProductionPerformance",
    "HeatRecovery",
    "EconomicAnalysis",
    "CarbonFootprint",
    "ProvenanceRecord",
    "ProductionMethod",
    "HydrogenColor",
    "PurityGrade",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-082"
__agent_name__ = "H2-PRODUCTION-HEAT"
