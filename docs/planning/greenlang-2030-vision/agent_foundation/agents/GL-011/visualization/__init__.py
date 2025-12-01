# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Visualization Module.

Comprehensive visualization suite for fuel management analytics including
cost waterfall charts, blend Sankey diagrams, carbon footprint breakdowns,
price trends, procurement dashboards, and universal export functionality.

Author: GreenLang Team
Version: 1.0.0
Standards: WCAG 2.1 Level AA, ISO 12647-2, GHG Protocol

Usage:
    from visualization import (
        # Waterfall Charts
        WaterfallChartEngine,
        FuelSwitchingWaterfall,
        BlendOptimizationWaterfall,
        CarbonCostWaterfall,

        # Sankey Diagrams
        FuelBlendSankeyEngine,
        MultiStageBlendSankey,
        SupplyChainSankey,

        # Carbon Footprint
        CarbonFootprintEngine,
        ScopeBreakdownGenerator,
        RegulatoryComplianceGenerator,

        # Price Trends
        PriceTrendEngine,
        SpotPriceTracker,
        ForwardCurveAnalyzer,

        # Procurement Dashboard
        ProcurementDashboardEngine,

        # Export
        ExportEngine,
        export_to_png,
        export_to_pdf,
        export_to_json,

        # Configuration
        VisualizationConfig,
        ThemeConfig,
        ConfigFactory,
    )
"""

# =============================================================================
# VERSION INFO
# =============================================================================

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__email__ = "support@greenlang.io"
__status__ = "Production"

# =============================================================================
# CONFIGURATION IMPORTS
# =============================================================================

from .config import (
    # Enums
    ThemeMode,
    ColorPalette,
    ChartType,
    ExportFormat,
    FontFamily,
    AnimationSpeed,
    ResponsiveBreakpoint,

    # Color Classes
    FuelTypeColors,
    CostCategoryColors,
    EmissionColors,
    StatusColors,
    GradientScales,

    # Config Classes
    FontConfig,
    MarginConfig,
    GridConfig,
    LegendConfig,
    AnimationConfig,
    HoverConfig,
    ExportConfig,
    AccessibilityConfig,
    CachingConfig,
    ResponsiveConfig,
    ThemeConfig,
    VisualizationConfig,

    # Chart-specific Configs
    WaterfallChartConfig,
    SankeyChartConfig,
    PieChartConfig,
    TimeSeriesConfig,
    DashboardConfig,

    # Factory
    ConfigFactory,

    # Utility Functions
    get_default_config,
    get_fuel_color,
    get_cost_color,
    get_emission_color,
    get_status_color,
    get_gradient_scale,
    hex_to_rgba,
    adjust_color_brightness,
    blend_colors,
    generate_color_palette,
    get_plotly_config,
    get_modebar_buttons,
    create_annotation,
    create_shape,
)

# =============================================================================
# WATERFALL CHART IMPORTS
# =============================================================================

from .fuel_cost_waterfall import (
    # Enums
    CostCategory,
    WaterfallType,
    WaterfallOrientation,
    ValueDisplayMode,
    DrillDownLevel,

    # Data Classes
    CostItem,
    CostBreakdown,
    ComparisonData,
    DrillDownNode,
    WaterfallAnnotation,

    # Options
    WaterfallChartOptions,

    # Engines
    WaterfallChartEngine,
    FuelSwitchingWaterfall,
    BlendOptimizationWaterfall,
    ProcurementOptimizationWaterfall,
    CarbonCostWaterfall,

    # Utilities
    WaterfallDataTransformer,

    # Example Functions
    create_sample_cost_breakdown,
)

# =============================================================================
# SANKEY DIAGRAM IMPORTS
# =============================================================================

from .fuel_blend_sankey import (
    # Enums
    NodeType,
    FlowType,
    ColorSchemeType,
    SankeyOrientation,
    NodeArrangement,
    LinkCurveType,

    # Data Classes
    SankeyNode,
    SankeyLink,
    FuelSource,
    BlendRecipe,
    EnergyOutput,
    BlendFlowData,
    SankeyDiagramData,

    # Options
    SankeyChartOptions,

    # Engines
    FuelBlendSankeyEngine,
    MultiStageBlendSankey,
    SupplyChainSankey,

    # Builders
    BlendFlowDataBuilder,

    # Example Functions
    create_sample_blend_flow,
)

# =============================================================================
# CARBON FOOTPRINT IMPORTS
# =============================================================================

from .carbon_footprint_breakdown import (
    # Enums
    EmissionScope,
    EmissionType,
    ChartMode,
    TimeGranularity,
    ComplianceStatus,
    TargetType,

    # Data Classes
    EmissionSource,
    EmissionTarget,
    RegulatoryLimit,
    EmissionDataPoint,
    EmissionTrendData,
    CarbonFootprintData,

    # Options
    CarbonChartOptions,

    # Engines
    CarbonFootprintEngine,
    ScopeBreakdownGenerator,
    RegulatoryComplianceGenerator,

    # Example Functions
    create_sample_carbon_data,
)

# =============================================================================
# PRICE TRENDS IMPORTS
# =============================================================================

from .price_trends import (
    # Enums
    TimeRange,
    ChartStyle,
    MovingAverageType,
    VolatilityIndicator,
    ForecastMethod,
    PriceUnit,
    MarketEventType,

    # Data Classes
    PriceDataPoint,
    FuelPriceSeries,
    MovingAverageConfig,
    ForecastConfig,
    VolatilityConfig,
    MarketEvent,
    PriceTrendData,

    # Options
    PriceTrendOptions,

    # Engines
    PriceTrendEngine,
    SpotPriceTracker,
    ForwardCurveAnalyzer,

    # Example Functions
    create_sample_price_data,
)

# =============================================================================
# PROCUREMENT DASHBOARD IMPORTS
# =============================================================================

from .procurement_dashboard import (
    # Enums
    KPIType,
    AlertSeverity,
    AlertType,
    TrendDirection,
    DashboardPanelType,

    # Data Classes
    InventoryLevel,
    Contract,
    SupplierMetrics,
    DeliveryRecord,
    ProcurementAlert,
    KPIMetric,
    ProcurementDashboardData,

    # Options
    DashboardPanelConfig,
    ProcurementDashboardOptions,

    # Engine
    ProcurementDashboardEngine,

    # Example Functions
    create_sample_dashboard_data,
)

# =============================================================================
# EXPORT ENGINE IMPORTS
# =============================================================================

from .export_engine import (
    # Enums
    ExportQuality,
    PageSize,
    PageOrientation,
    ImageFormat,
    CompressionLevel,
    WatermarkPosition,
    ScheduleFrequency,

    # Data Classes
    ExportDimensions,
    WatermarkConfig,
    BrandingConfig,
    EmailConfig,
    ScheduleConfig,
    ExportMetadata,
    ExportResult,
    BatchExportJob,
    ReportTemplate,

    # Options
    ExportOptions,

    # Engine
    ExportEngine,

    # Convenience Functions
    export_to_png,
    export_to_pdf,
    export_to_json,
    export_data_to_csv,
    export_data_to_excel,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",

    # Configuration
    "ThemeMode",
    "ColorPalette",
    "ChartType",
    "ExportFormat",
    "FuelTypeColors",
    "CostCategoryColors",
    "EmissionColors",
    "StatusColors",
    "GradientScales",
    "FontConfig",
    "MarginConfig",
    "LegendConfig",
    "AnimationConfig",
    "HoverConfig",
    "ExportConfig",
    "AccessibilityConfig",
    "ThemeConfig",
    "VisualizationConfig",
    "ConfigFactory",
    "get_default_config",
    "get_fuel_color",
    "get_cost_color",
    "get_emission_color",
    "get_status_color",
    "hex_to_rgba",
    "get_plotly_config",

    # Waterfall Charts
    "CostCategory",
    "WaterfallType",
    "CostItem",
    "CostBreakdown",
    "WaterfallChartOptions",
    "WaterfallChartEngine",
    "FuelSwitchingWaterfall",
    "BlendOptimizationWaterfall",
    "CarbonCostWaterfall",
    "WaterfallDataTransformer",

    # Sankey Diagrams
    "NodeType",
    "FlowType",
    "SankeyNode",
    "SankeyLink",
    "FuelSource",
    "BlendRecipe",
    "BlendFlowData",
    "SankeyDiagramData",
    "SankeyChartOptions",
    "FuelBlendSankeyEngine",
    "MultiStageBlendSankey",
    "SupplyChainSankey",
    "BlendFlowDataBuilder",

    # Carbon Footprint
    "EmissionScope",
    "EmissionType",
    "ComplianceStatus",
    "EmissionSource",
    "EmissionTarget",
    "RegulatoryLimit",
    "CarbonFootprintData",
    "CarbonChartOptions",
    "CarbonFootprintEngine",
    "ScopeBreakdownGenerator",
    "RegulatoryComplianceGenerator",

    # Price Trends
    "TimeRange",
    "ChartStyle",
    "MovingAverageType",
    "ForecastMethod",
    "MarketEventType",
    "PriceDataPoint",
    "FuelPriceSeries",
    "MovingAverageConfig",
    "ForecastConfig",
    "MarketEvent",
    "PriceTrendData",
    "PriceTrendOptions",
    "PriceTrendEngine",
    "SpotPriceTracker",
    "ForwardCurveAnalyzer",

    # Procurement Dashboard
    "KPIType",
    "AlertSeverity",
    "AlertType",
    "InventoryLevel",
    "Contract",
    "SupplierMetrics",
    "DeliveryRecord",
    "ProcurementAlert",
    "KPIMetric",
    "ProcurementDashboardData",
    "ProcurementDashboardOptions",
    "ProcurementDashboardEngine",

    # Export Engine
    "ExportQuality",
    "PageSize",
    "PageOrientation",
    "ExportDimensions",
    "WatermarkConfig",
    "BrandingConfig",
    "ExportMetadata",
    "ExportResult",
    "BatchExportJob",
    "ReportTemplate",
    "ExportOptions",
    "ExportEngine",
    "export_to_png",
    "export_to_pdf",
    "export_to_json",
    "export_data_to_csv",
    "export_data_to_excel",
]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _init_module():
    """Initialize visualization module."""
    import logging

    # Configure logging for visualization module
    logger = logging.getLogger("gl011.visualization")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    logger.debug("GL-011 FUELCRAFT Visualization module initialized")


_init_module()
