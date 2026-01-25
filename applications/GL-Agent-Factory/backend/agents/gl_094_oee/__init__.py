"""GL-094: OEE Maximizer Agent (OEE-MAXIMIZER)"""

from .agent import (
    OEEMaximizerAgent,
    OEEInput,
    EquipmentRuntime,
    ProductionOutput,
    OEEOutput,
    OEEMetrics,
    LossAnalysis,
    ImprovementOpportunity,
    ProvenanceRecord,
    LossCategory,
    OEEClass,
    PACK_SPEC,
)

__all__ = [
    "OEEMaximizerAgent",
    "OEEInput",
    "EquipmentRuntime",
    "ProductionOutput",
    "OEEOutput",
    "OEEMetrics",
    "LossAnalysis",
    "ImprovementOpportunity",
    "ProvenanceRecord",
    "LossCategory",
    "OEEClass",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-094"
__agent_name__ = "OEE-MAXIMIZER"
