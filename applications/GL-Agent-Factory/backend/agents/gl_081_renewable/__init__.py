"""GL-081: Renewable Integration Agent (RENEWABLE-INTEGRATOR)"""

from .agent import (
    RenewableIntegrationAgent,
    RenewableIntegrationInput,
    RenewableIntegrationOutput,
    FacilityInfo,
    ThermalDemand,
    RenewableOption,
    ConventionalSystem,
    RenewableSystemAnalysis,
    IntegrationRecommendation,
    GridInteractionAnalysis,
    StorageAnalysis,
    ProvenanceRecord,
    RenewableType,
    IntegrationStrategy,
    StorageType,
    GridInteractionMode,
    PACK_SPEC,
)

__all__ = [
    "RenewableIntegrationAgent",
    "RenewableIntegrationInput",
    "RenewableIntegrationOutput",
    "FacilityInfo",
    "ThermalDemand",
    "RenewableOption",
    "ConventionalSystem",
    "RenewableSystemAnalysis",
    "IntegrationRecommendation",
    "GridInteractionAnalysis",
    "StorageAnalysis",
    "ProvenanceRecord",
    "RenewableType",
    "IntegrationStrategy",
    "StorageType",
    "GridInteractionMode",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-081"
__agent_name__ = "RENEWABLE-INTEGRATOR"
