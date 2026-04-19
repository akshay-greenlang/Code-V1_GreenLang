# -*- coding: utf-8 -*-
"""
HotspotAnalysisAgent Custom Exceptions
GL-VCCI Scope 3 Platform

Custom exception classes for hotspot analysis operations.

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""


class HotspotAnalysisError(Exception):
    """Base exception for hotspot analysis errors."""
    pass


class InsufficientDataError(HotspotAnalysisError):
    """Raised when insufficient data is provided for analysis."""
    pass


class InvalidDimensionError(HotspotAnalysisError):
    """Raised when invalid dimension is specified for segmentation."""
    pass


class ScenarioConfigError(HotspotAnalysisError):
    """Raised when scenario configuration is invalid."""
    pass


class ROICalculationError(HotspotAnalysisError):
    """Raised when ROI calculation fails."""
    pass


class AbatementCurveError(HotspotAnalysisError):
    """Raised when abatement curve generation fails."""
    pass


class ParetoAnalysisError(HotspotAnalysisError):
    """Raised when Pareto analysis fails."""
    pass


class SegmentationError(HotspotAnalysisError):
    """Raised when segmentation analysis fails."""
    pass


class HotspotDetectionError(HotspotAnalysisError):
    """Raised when hotspot detection fails."""
    pass


class InsightGenerationError(HotspotAnalysisError):
    """Raised when insight generation fails."""
    pass


class DataValidationError(HotspotAnalysisError):
    """Raised when data validation fails."""
    pass


__all__ = [
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
