"""GL-083: CCS Integration Optimizer (CCS-INTEGRATOR)"""

from .agent import (
    CCSIntegrationAgent,
    CCSIntegrationInput,
    CCSIntegrationOutput,
    FacilityInfo,
    CaptureSystem,
    StorageOptions,
    CapturePerformance,
    TransportAnalysis,
    StorageAnalysis,
    EconomicAnalysis,
    ProvenanceRecord,
    CaptureTechnology,
    StorageType,
    TransportMode,
    PACK_SPEC,
)

__all__ = [
    "CCSIntegrationAgent",
    "CCSIntegrationInput",
    "CCSIntegrationOutput",
    "FacilityInfo",
    "CaptureSystem",
    "StorageOptions",
    "CapturePerformance",
    "TransportAnalysis",
    "StorageAnalysis",
    "EconomicAnalysis",
    "ProvenanceRecord",
    "CaptureTechnology",
    "StorageType",
    "TransportMode",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-083"
__agent_name__ = "CCS-INTEGRATOR"
