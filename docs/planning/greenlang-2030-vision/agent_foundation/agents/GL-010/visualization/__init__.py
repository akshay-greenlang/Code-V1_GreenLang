"""
GL-010 EMISSIONWATCH - Visualization Engine

Comprehensive visualization module for emissions compliance monitoring.
Provides interactive dashboards, trend analysis, violation tracking,
source breakdowns, regulatory heatmaps, and export functionality.

Author: GreenLang Team
Version: 1.0.0

Usage:
    from visualization import (
        ComplianceDashboard,
        EmissionsTrendChart,
        ViolationTimelineChart,
        SourceBreakdownChart,
        RegulatoryHeatmap,
        ReportExporter
    )

    # Create compliance dashboard
    dashboard = ComplianceDashboard(data)
    html = dashboard.to_html()

    # Generate trend chart
    chart = EmissionsTrendChart(config)
    chart.set_data(emissions_data)
    json_output = chart.to_plotly_json()

    # Export reports
    exporter = ReportExporter(config)
    exporter.export("report.pdf", ExportFormat.PDF)
"""

# Version info
__version__ = "1.0.0"
__author__ = "GreenLang Team"
__license__ = "MIT"

# Compliance Dashboard exports
from .compliance_dashboard import (
    # Enums
    ComplianceStatus,
    ViolationType,

    # Data classes
    PollutantStatus,
    Violation,
    ComplianceDashboardData,

    # Utilities
    ColorScheme,

    # Chart classes
    ChartBase,
    StatusMatrixChart,
    GaugeChart,
    TrendChart,
    ViolationSummaryChart,
    ComplianceMarginChart,

    # Main dashboard
    ComplianceDashboard,

    # Sample data
    create_sample_dashboard_data,
)

# Emissions Trends exports
from .emissions_trends import (
    # Enums
    TimeResolution,
    TrendDirection,

    # Data classes
    EmissionDataPoint,
    TrendStatistics,
    TrendConfig,

    # Utilities
    StatisticsCalculator,
    RollingAverageCalculator,
    AnomalyDetector,
    SimpleForecast,

    # Chart classes
    EmissionsTrendChart,
    EmissionsTrendDashboard,

    # Sample data
    create_sample_trend_data,
)

# Violation Timeline exports
from .violation_timeline import (
    # Enums
    ViolationSeverity,
    ViolationStatus,
    ViolationType as ViolationTypeTimeline,

    # Data classes
    RegulatoryResponse,
    ViolationRecord,
    TimelineConfig,

    # Chart class
    ViolationTimelineChart,

    # Sample data
    create_sample_violations,
)

# Source Breakdown exports
from .source_breakdown import (
    # Enums
    SourceType,
    FuelType,

    # Data classes
    EmissionSource,
    ProcessUnit,
    SourceBreakdownConfig,

    # Chart class
    SourceBreakdownChart,

    # Sample data
    create_sample_sources,
)

# Regulatory Heatmap exports
from .regulatory_heatmap import (
    # Enums
    ComplianceLevel,

    # Data classes
    JurisdictionInfo,
    PollutantLimit,
    ComplianceCell,
    HeatmapConfig,

    # Chart class
    RegulatoryHeatmap,

    # Sample data
    create_sample_heatmap_data,
)

# Export functionality
from .export import (
    # Enums
    ExportFormat,

    # Data classes
    ExportConfig,
    TableData,

    # Exporter classes
    ExporterBase,
    PDFExporter,
    ImageExporter,
    ExcelExporter,
    JSONExporter,
    HTMLExporter,
    CEDRIXMLExporter,

    # Main exporter
    ReportExporter,

    # Sample data
    create_sample_report,
)

# Module-level convenience functions
def create_compliance_dashboard(
    data: ComplianceDashboardData,
    color_blind_safe: bool = False
) -> ComplianceDashboard:
    """
    Create a compliance dashboard from data.

    Args:
        data: Dashboard data structure
        color_blind_safe: Use color-blind safe palette

    Returns:
        ComplianceDashboard instance
    """
    return ComplianceDashboard(data, color_blind_safe=color_blind_safe)


def create_trend_chart(
    pollutant: str,
    pollutant_name: str,
    unit: str,
    permit_limit: float,
    **kwargs
) -> EmissionsTrendChart:
    """
    Create an emissions trend chart.

    Args:
        pollutant: Pollutant identifier
        pollutant_name: Human-readable pollutant name
        unit: Measurement unit
        permit_limit: Regulatory permit limit
        **kwargs: Additional TrendConfig parameters

    Returns:
        EmissionsTrendChart instance
    """
    config = TrendConfig(
        pollutant=pollutant,
        pollutant_name=pollutant_name,
        unit=unit,
        permit_limit=permit_limit,
        warning_threshold=permit_limit * 0.9,
        resolution=TimeResolution.HOURLY,
        **kwargs
    )
    return EmissionsTrendChart(config)


def create_violation_timeline(
    violations: list,
    **kwargs
) -> ViolationTimelineChart:
    """
    Create a violation timeline chart.

    Args:
        violations: List of ViolationRecord objects
        **kwargs: Additional TimelineConfig parameters

    Returns:
        ViolationTimelineChart instance
    """
    config = TimelineConfig(**kwargs)
    return ViolationTimelineChart(violations, config)


def create_source_breakdown(
    sources: list,
    **kwargs
) -> SourceBreakdownChart:
    """
    Create a source breakdown chart.

    Args:
        sources: List of EmissionSource objects
        **kwargs: Additional SourceBreakdownConfig parameters

    Returns:
        SourceBreakdownChart instance
    """
    config = SourceBreakdownConfig(**kwargs)
    return SourceBreakdownChart(sources, config)


def create_regulatory_heatmap(
    jurisdictions: list,
    pollutants: list,
    **kwargs
) -> RegulatoryHeatmap:
    """
    Create a regulatory compliance heatmap.

    Args:
        jurisdictions: List of JurisdictionInfo objects
        pollutants: List of pollutant identifiers
        **kwargs: Additional HeatmapConfig parameters

    Returns:
        RegulatoryHeatmap instance
    """
    config = HeatmapConfig(**kwargs)
    return RegulatoryHeatmap(jurisdictions, pollutants, config)


def create_report_exporter(**kwargs) -> ReportExporter:
    """
    Create a report exporter.

    Args:
        **kwargs: ExportConfig parameters

    Returns:
        ReportExporter instance
    """
    config = ExportConfig(**kwargs)
    return ReportExporter(config)


# Define what's available with "from visualization import *"
__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Compliance Dashboard
    "ComplianceStatus",
    "ViolationType",
    "PollutantStatus",
    "Violation",
    "ComplianceDashboardData",
    "ColorScheme",
    "ChartBase",
    "StatusMatrixChart",
    "GaugeChart",
    "TrendChart",
    "ViolationSummaryChart",
    "ComplianceMarginChart",
    "ComplianceDashboard",
    "create_sample_dashboard_data",

    # Emissions Trends
    "TimeResolution",
    "TrendDirection",
    "EmissionDataPoint",
    "TrendStatistics",
    "TrendConfig",
    "StatisticsCalculator",
    "RollingAverageCalculator",
    "AnomalyDetector",
    "SimpleForecast",
    "EmissionsTrendChart",
    "EmissionsTrendDashboard",
    "create_sample_trend_data",

    # Violation Timeline
    "ViolationSeverity",
    "ViolationStatus",
    "ViolationTypeTimeline",
    "RegulatoryResponse",
    "ViolationRecord",
    "TimelineConfig",
    "ViolationTimelineChart",
    "create_sample_violations",

    # Source Breakdown
    "SourceType",
    "FuelType",
    "EmissionSource",
    "ProcessUnit",
    "SourceBreakdownConfig",
    "SourceBreakdownChart",
    "create_sample_sources",

    # Regulatory Heatmap
    "ComplianceLevel",
    "JurisdictionInfo",
    "PollutantLimit",
    "ComplianceCell",
    "HeatmapConfig",
    "RegulatoryHeatmap",
    "create_sample_heatmap_data",

    # Export
    "ExportFormat",
    "ExportConfig",
    "TableData",
    "ExporterBase",
    "PDFExporter",
    "ImageExporter",
    "ExcelExporter",
    "JSONExporter",
    "HTMLExporter",
    "CEDRIXMLExporter",
    "ReportExporter",
    "create_sample_report",

    # Convenience functions
    "create_compliance_dashboard",
    "create_trend_chart",
    "create_violation_timeline",
    "create_source_breakdown",
    "create_regulatory_heatmap",
    "create_report_exporter",
]
