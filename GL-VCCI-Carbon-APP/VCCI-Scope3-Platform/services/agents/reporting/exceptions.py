"""
Scope3ReportingAgent Exceptions
GL-VCCI Scope 3 Platform

Custom exceptions for reporting operations.

Version: 1.0.0
Phase: 3 (Weeks 16-18)
Date: 2025-10-30
"""


class ReportingError(Exception):
    """Base exception for reporting agent errors."""
    pass


class ValidationError(ReportingError):
    """Raised when data validation fails before report generation."""
    pass


class DataQualityError(ReportingError):
    """Raised when data quality is insufficient for reporting."""
    pass


class MissingDataError(ReportingError):
    """Raised when required data fields are missing."""
    pass


class StandardComplianceError(ReportingError):
    """Raised when report cannot meet standard requirements."""
    pass


class ExportError(ReportingError):
    """Raised when export to format fails."""
    pass


class TemplateError(ReportingError):
    """Raised when template rendering fails."""
    pass


class ChartGenerationError(ReportingError):
    """Raised when chart generation fails."""
    pass


class ProvenanceError(ReportingError):
    """Raised when provenance chain is incomplete or invalid."""
    pass


__all__ = [
    "ReportingError",
    "ValidationError",
    "DataQualityError",
    "MissingDataError",
    "StandardComplianceError",
    "ExportError",
    "TemplateError",
    "ChartGenerationError",
    "ProvenanceError",
]
