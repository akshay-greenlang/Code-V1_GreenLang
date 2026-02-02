"""
GL-056: Curing Oven Controller Agent (CURE-CTRL)

This package provides the CuringOvenAgent for optimizing curing oven operations
in industrial coating, composites, and powder coating applications.

Key Features:
- Temperature profile optimization per zone
- Cure cycle validation and monitoring
- Energy consumption tracking and optimization
- Complete SHA-256 provenance tracking

Standards Compliance:
- ASTM D4541: Pull-Off Adhesion Testing
- ISO 11507: Paints and Varnishes - Exposure to Artificial Weathering
- NFPA 86: Standard for Ovens and Furnaces

Example Usage:
    >>> from backend.agents.gl_056_curing_oven import (
    ...     CuringOvenAgent,
    ...     CuringOvenInput,
    ... )
    >>> agent = CuringOvenAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Cure Quality Score: {result.cure_quality_score}")
"""

from .agent import (
    CuringOvenAgent,
    CuringOvenInput,
    CuringOvenOutput,
    ZoneData,
    ConveyorData,
    ProductData,
    ZoneAnalysis,
    CureQualityAssessment,
    EnergyAnalysis,
    Recommendation,
    Warning,
    ProductType,
    CureStatus,
    ZoneStatus,
    PACK_SPEC,
)

__all__ = [
    "CuringOvenAgent",
    "CuringOvenInput",
    "CuringOvenOutput",
    "ZoneData",
    "ConveyorData",
    "ProductData",
    "ZoneAnalysis",
    "CureQualityAssessment",
    "EnergyAnalysis",
    "Recommendation",
    "Warning",
    "ProductType",
    "CureStatus",
    "ZoneStatus",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-056"
__agent_name__ = "CURE-CTRL"
