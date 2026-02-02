"""GL-093: Product Quality Integrator Agent (QUALITY-INTEGRATOR)"""

from .agent import (
    ProductQualityIntegratorAgent,
    QualityInput,
    ProductionBatch,
    DefectRecord,
    QualitySpecification,
    MeasurementData,
    QualityOutput,
    QualityMetrics,
    ProcessCapabilityAnalysis,
    DefectAnalysis,
    QualityRecommendation,
    ProvenanceRecord,
    DefectSeverity,
    ProcessCapability,
    QualityCostCategory,
    PACK_SPEC,
)

__all__ = [
    "ProductQualityIntegratorAgent",
    "QualityInput",
    "ProductionBatch",
    "DefectRecord",
    "QualitySpecification",
    "MeasurementData",
    "QualityOutput",
    "QualityMetrics",
    "ProcessCapabilityAnalysis",
    "DefectAnalysis",
    "QualityRecommendation",
    "ProvenanceRecord",
    "DefectSeverity",
    "ProcessCapability",
    "QualityCostCategory",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-093"
__agent_name__ = "QUALITY-INTEGRATOR"
