# -*- coding: utf-8 -*-
"""
HotspotAnalysisAgent Package
GL-VCCI Scope 3 Platform

Emissions hotspot analysis and scenario modeling agent.

Features:
- Pareto analysis (80/20 rule)
- Multi-dimensional segmentation
- Scenario modeling framework
- ROI analysis and abatement curves
- Automated hotspot detection
- Actionable insight generation

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

from .agent import HotspotAnalysisAgent
from .models import (
    EmissionRecord,
    ParetoItem,
    ParetoAnalysis,
    Segment,
    SegmentationAnalysis,
    BaseScenario,
    SupplierSwitchScenario,
    ModalShiftScenario,
    ProductSubstitutionScenario,
    ScenarioResult,
    Initiative,
    ROIAnalysis,
    AbatementCurvePoint,
    AbatementCurve,
    Hotspot,
    HotspotReport,
    Insight,
    InsightReport,
)
from .config import (
    AnalysisDimension,
    ScenarioType,
    InsightPriority,
    InsightType,
    HotspotCriteria,
    ParetoConfig,
    ROIConfig,
    SegmentationConfig,
    HotspotAnalysisConfig,
    DEFAULT_CONFIG,
    DIMENSION_FIELD_MAP,
    REQUIRED_EMISSION_FIELDS,
    OPTIONAL_EMISSION_FIELDS,
)
from .exceptions import (
    HotspotAnalysisError,
    InsufficientDataError,
    InvalidDimensionError,
    ScenarioConfigError,
    ROICalculationError,
    AbatementCurveError,
    ParetoAnalysisError,
    SegmentationError,
    HotspotDetectionError,
    InsightGenerationError,
    DataValidationError,
)

__version__ = "1.0.0"
__all__ = [
    "HotspotAnalysisAgent",
    # Models
    "EmissionRecord",
    "ParetoItem",
    "ParetoAnalysis",
    "Segment",
    "SegmentationAnalysis",
    "BaseScenario",
    "SupplierSwitchScenario",
    "ModalShiftScenario",
    "ProductSubstitutionScenario",
    "ScenarioResult",
    "Initiative",
    "ROIAnalysis",
    "AbatementCurvePoint",
    "AbatementCurve",
    "Hotspot",
    "HotspotReport",
    "Insight",
    "InsightReport",
    # Config
    "AnalysisDimension",
    "ScenarioType",
    "InsightPriority",
    "InsightType",
    "HotspotCriteria",
    "ParetoConfig",
    "ROIConfig",
    "SegmentationConfig",
    "HotspotAnalysisConfig",
    "DEFAULT_CONFIG",
    "DIMENSION_FIELD_MAP",
    "REQUIRED_EMISSION_FIELDS",
    "OPTIONAL_EMISSION_FIELDS",
    # Exceptions
    "HotspotAnalysisError",
    "InsufficientDataError",
    "InvalidDimensionError",
    "ScenarioConfigError",
    "ROICalculationError",
    "AbatementCurveError",
    "ParetoAnalysisError",
    "SegmentationError",
    "HotspotDetectionError",
    "InsightGenerationError",
    "DataValidationError",
]
