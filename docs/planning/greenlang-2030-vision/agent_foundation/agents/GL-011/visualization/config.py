# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Visualization Configuration Module.

Centralized configuration for all visualization components including
color schemes, themes, accessibility settings, and export configurations.

Author: GreenLang Team
Version: 1.0.0
Standards: WCAG 2.1 Level AA, ISO 12647-2
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum, auto
import json
import os
from abc import ABC, abstractmethod


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ThemeMode(Enum):
    """Available theme modes for visualizations."""
    LIGHT = "light"
    DARK = "dark"
    HIGH_CONTRAST = "high_contrast"
    PRINT = "print"
    PRESENTATION = "presentation"


class ColorPalette(Enum):
    """Color palette options."""
    DEFAULT = "default"
    CORPORATE = "corporate"
    ACCESSIBLE = "accessible"
    MONOCHROME = "monochrome"
    ENERGY = "energy"
    SUSTAINABILITY = "sustainability"


class ChartType(Enum):
    """Supported chart types."""
    WATERFALL = "waterfall"
    SANKEY = "sankey"
    PIE = "pie"
    BAR = "bar"
    LINE = "line"
    AREA = "area"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    SCATTER = "scatter"
    BUBBLE = "bubble"
    RADAR = "radar"
    FUNNEL = "funnel"
    TABLE = "table"
    KPI = "kpi"


class ExportFormat(Enum):
    """Supported export formats."""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    HTML = "html"
    PPTX = "pptx"


class FontFamily(Enum):
    """Supported font families."""
    SYSTEM = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    ARIAL = "Arial, Helvetica, sans-serif"
    HELVETICA = "Helvetica Neue, Helvetica, Arial, sans-serif"
    ROBOTO = "'Roboto', sans-serif"
    OPEN_SANS = "'Open Sans', sans-serif"
    LATO = "'Lato', sans-serif"
    MONOSPACE = "'Consolas', 'Monaco', 'Courier New', monospace"


class AnimationSpeed(Enum):
    """Animation speed presets."""
    NONE = 0
    FAST = 150
    NORMAL = 300
    SLOW = 600
    VERY_SLOW = 1000


class ResponsiveBreakpoint(Enum):
    """Responsive design breakpoints."""
    MOBILE_SM = 320
    MOBILE = 480
    TABLET = 768
    DESKTOP = 1024
    DESKTOP_LG = 1280
    DESKTOP_XL = 1440
    DESKTOP_XXL = 1920


# =============================================================================
# COLOR DEFINITIONS
# =============================================================================

class FuelTypeColors:
    """Color mappings for different fuel types."""

    NATURAL_GAS = "#4ECDC4"
    COAL = "#34495E"
    OIL = "#8B4513"
    DIESEL = "#A0522D"
    GASOLINE = "#CD853F"
    LPG = "#87CEEB"
    LNG = "#5F9EA0"
    BIOMASS = "#228B22"
    BIOGAS = "#32CD32"
    BIODIESEL = "#6B8E23"
    HYDROGEN = "#00CED1"
    ELECTRICITY = "#9B59B6"
    NUCLEAR = "#FF69B4"
    SOLAR = "#FFD700"
    WIND = "#87CEFA"
    HYDRO = "#1E90FF"
    GEOTHERMAL = "#FF6347"
    WASTE = "#696969"
    PEAT = "#8B7355"
    WOOD = "#DEB887"
    ETHANOL = "#90EE90"
    METHANOL = "#98FB98"
    PROPANE = "#B0E0E6"
    BUTANE = "#ADD8E6"
    KEROSENE = "#F4A460"
    HEAVY_FUEL_OIL = "#4A4A4A"
    MARINE_DIESEL = "#5D5D5D"
    AVIATION_FUEL = "#6B6B6B"

    @classmethod
    def get_color(cls, fuel_type: str) -> str:
        """Get color for a fuel type."""
        fuel_map = {
            "natural_gas": cls.NATURAL_GAS,
            "gas": cls.NATURAL_GAS,
            "coal": cls.COAL,
            "oil": cls.OIL,
            "crude_oil": cls.OIL,
            "diesel": cls.DIESEL,
            "gasoline": cls.GASOLINE,
            "petrol": cls.GASOLINE,
            "lpg": cls.LPG,
            "lng": cls.LNG,
            "biomass": cls.BIOMASS,
            "biogas": cls.BIOGAS,
            "biodiesel": cls.BIODIESEL,
            "hydrogen": cls.HYDROGEN,
            "h2": cls.HYDROGEN,
            "electricity": cls.ELECTRICITY,
            "electric": cls.ELECTRICITY,
            "nuclear": cls.NUCLEAR,
            "solar": cls.SOLAR,
            "wind": cls.WIND,
            "hydro": cls.HYDRO,
            "hydroelectric": cls.HYDRO,
            "geothermal": cls.GEOTHERMAL,
            "waste": cls.WASTE,
            "peat": cls.PEAT,
            "wood": cls.WOOD,
            "wood_pellets": cls.WOOD,
            "ethanol": cls.ETHANOL,
            "methanol": cls.METHANOL,
            "propane": cls.PROPANE,
            "butane": cls.BUTANE,
            "kerosene": cls.KEROSENE,
            "heavy_fuel_oil": cls.HEAVY_FUEL_OIL,
            "hfo": cls.HEAVY_FUEL_OIL,
            "marine_diesel": cls.MARINE_DIESEL,
            "mdo": cls.MARINE_DIESEL,
            "aviation_fuel": cls.AVIATION_FUEL,
            "jet_fuel": cls.AVIATION_FUEL,
        }
        normalized = fuel_type.lower().replace(" ", "_").replace("-", "_")
        return fuel_map.get(normalized, "#888888")

    @classmethod
    def get_all_colors(cls) -> Dict[str, str]:
        """Get all fuel type colors."""
        return {
            "natural_gas": cls.NATURAL_GAS,
            "coal": cls.COAL,
            "oil": cls.OIL,
            "diesel": cls.DIESEL,
            "gasoline": cls.GASOLINE,
            "lpg": cls.LPG,
            "lng": cls.LNG,
            "biomass": cls.BIOMASS,
            "biogas": cls.BIOGAS,
            "biodiesel": cls.BIODIESEL,
            "hydrogen": cls.HYDROGEN,
            "electricity": cls.ELECTRICITY,
            "nuclear": cls.NUCLEAR,
            "solar": cls.SOLAR,
            "wind": cls.WIND,
            "hydro": cls.HYDRO,
            "geothermal": cls.GEOTHERMAL,
            "waste": cls.WASTE,
        }


class CostCategoryColors:
    """Color mappings for cost categories."""

    BASELINE = "#3498DB"
    FUEL_COST = "#E74C3C"
    SWITCHING_SAVINGS = "#2ECC71"
    BLEND_SAVINGS = "#27AE60"
    PROCUREMENT_SAVINGS = "#1ABC9C"
    MARKET_TIMING = "#16A085"
    CARBON_COST = "#E67E22"
    CARBON_SAVINGS = "#F39C12"
    DELIVERY_COST = "#9B59B6"
    STORAGE_COST = "#8E44AD"
    HANDLING_COST = "#34495E"
    TAX = "#95A5A6"
    SUBSIDY = "#2ECC71"
    PENALTY = "#C0392B"
    HEDGING_COST = "#7F8C8D"
    OPTIMIZATION_BENEFIT = "#27AE60"
    TOTAL = "#2C3E50"

    @classmethod
    def get_color(cls, category: str) -> str:
        """Get color for a cost category."""
        category_map = {
            "baseline": cls.BASELINE,
            "baseline_cost": cls.BASELINE,
            "fuel_cost": cls.FUEL_COST,
            "fuel": cls.FUEL_COST,
            "switching_savings": cls.SWITCHING_SAVINGS,
            "fuel_switching": cls.SWITCHING_SAVINGS,
            "blend_savings": cls.BLEND_SAVINGS,
            "blending": cls.BLEND_SAVINGS,
            "blend_optimization": cls.BLEND_SAVINGS,
            "procurement_savings": cls.PROCUREMENT_SAVINGS,
            "procurement": cls.PROCUREMENT_SAVINGS,
            "procurement_optimization": cls.PROCUREMENT_SAVINGS,
            "market_timing": cls.MARKET_TIMING,
            "timing": cls.MARKET_TIMING,
            "carbon_cost": cls.CARBON_COST,
            "carbon": cls.CARBON_COST,
            "carbon_savings": cls.CARBON_SAVINGS,
            "delivery_cost": cls.DELIVERY_COST,
            "delivery": cls.DELIVERY_COST,
            "transport": cls.DELIVERY_COST,
            "storage_cost": cls.STORAGE_COST,
            "storage": cls.STORAGE_COST,
            "handling_cost": cls.HANDLING_COST,
            "handling": cls.HANDLING_COST,
            "tax": cls.TAX,
            "taxes": cls.TAX,
            "subsidy": cls.SUBSIDY,
            "subsidies": cls.SUBSIDY,
            "penalty": cls.PENALTY,
            "penalties": cls.PENALTY,
            "hedging_cost": cls.HEDGING_COST,
            "hedging": cls.HEDGING_COST,
            "optimization_benefit": cls.OPTIMIZATION_BENEFIT,
            "optimization": cls.OPTIMIZATION_BENEFIT,
            "total": cls.TOTAL,
            "net_total": cls.TOTAL,
        }
        normalized = category.lower().replace(" ", "_").replace("-", "_")
        return category_map.get(normalized, "#888888")


class EmissionColors:
    """Color mappings for emission types."""

    CO2 = "#607D8B"
    CO2E = "#546E7A"
    CH4 = "#FF9800"
    N2O = "#9C27B0"
    NOX = "#E74C3C"
    SOX = "#9B59B6"
    PM = "#795548"
    PM25 = "#8D6E63"
    PM10 = "#A1887F"
    VOC = "#FF5722"
    CO = "#4CAF50"
    HG = "#E91E63"
    HCL = "#F06292"
    NH3 = "#00BCD4"
    OZONE = "#03A9F4"

    @classmethod
    def get_color(cls, emission_type: str) -> str:
        """Get color for an emission type."""
        emission_map = {
            "co2": cls.CO2,
            "carbon_dioxide": cls.CO2,
            "co2e": cls.CO2E,
            "co2_equivalent": cls.CO2E,
            "ch4": cls.CH4,
            "methane": cls.CH4,
            "n2o": cls.N2O,
            "nitrous_oxide": cls.N2O,
            "nox": cls.NOX,
            "nitrogen_oxides": cls.NOX,
            "sox": cls.SOX,
            "sulfur_oxides": cls.SOX,
            "so2": cls.SOX,
            "pm": cls.PM,
            "particulate_matter": cls.PM,
            "pm2.5": cls.PM25,
            "pm25": cls.PM25,
            "pm10": cls.PM10,
            "voc": cls.VOC,
            "volatile_organic_compounds": cls.VOC,
            "co": cls.CO,
            "carbon_monoxide": cls.CO,
            "hg": cls.HG,
            "mercury": cls.HG,
            "hcl": cls.HCL,
            "hydrogen_chloride": cls.HCL,
            "nh3": cls.NH3,
            "ammonia": cls.NH3,
            "ozone": cls.OZONE,
            "o3": cls.OZONE,
        }
        normalized = emission_type.lower().replace(" ", "_").replace("-", "_")
        return emission_map.get(normalized, "#888888")


class StatusColors:
    """Color mappings for status indicators."""

    SUCCESS = "#2ECC71"
    WARNING = "#F39C12"
    ERROR = "#E74C3C"
    INFO = "#3498DB"
    NEUTRAL = "#95A5A6"
    PENDING = "#9B59B6"
    ACTIVE = "#27AE60"
    INACTIVE = "#BDC3C7"
    CRITICAL = "#C0392B"
    OPTIMAL = "#1ABC9C"
    SUBOPTIMAL = "#E67E22"

    # Accessible versions (WCAG 2.1 AA compliant)
    SUCCESS_A11Y = "#009E73"
    WARNING_A11Y = "#E69F00"
    ERROR_A11Y = "#D55E00"
    INFO_A11Y = "#0072B2"
    NEUTRAL_A11Y = "#999999"

    @classmethod
    def get_color(cls, status: str, accessible: bool = False) -> str:
        """Get color for a status."""
        if accessible:
            status_map = {
                "success": cls.SUCCESS_A11Y,
                "ok": cls.SUCCESS_A11Y,
                "compliant": cls.SUCCESS_A11Y,
                "warning": cls.WARNING_A11Y,
                "caution": cls.WARNING_A11Y,
                "error": cls.ERROR_A11Y,
                "failure": cls.ERROR_A11Y,
                "violation": cls.ERROR_A11Y,
                "info": cls.INFO_A11Y,
                "information": cls.INFO_A11Y,
                "neutral": cls.NEUTRAL_A11Y,
                "unknown": cls.NEUTRAL_A11Y,
            }
        else:
            status_map = {
                "success": cls.SUCCESS,
                "ok": cls.SUCCESS,
                "compliant": cls.SUCCESS,
                "warning": cls.WARNING,
                "caution": cls.WARNING,
                "error": cls.ERROR,
                "failure": cls.ERROR,
                "violation": cls.ERROR,
                "info": cls.INFO,
                "information": cls.INFO,
                "neutral": cls.NEUTRAL,
                "unknown": cls.NEUTRAL,
                "pending": cls.PENDING,
                "active": cls.ACTIVE,
                "inactive": cls.INACTIVE,
                "critical": cls.CRITICAL,
                "optimal": cls.OPTIMAL,
                "suboptimal": cls.SUBOPTIMAL,
            }
        normalized = status.lower().replace(" ", "_").replace("-", "_")
        return status_map.get(normalized, "#888888")


class GradientScales:
    """Predefined gradient color scales."""

    # Sequential scales
    BLUES = [
        [0.0, "#F7FBFF"],
        [0.25, "#DEEBF7"],
        [0.5, "#9ECAE1"],
        [0.75, "#4292C6"],
        [1.0, "#084594"],
    ]

    GREENS = [
        [0.0, "#F7FCF5"],
        [0.25, "#D9F0D3"],
        [0.5, "#A1D99B"],
        [0.75, "#41AB5D"],
        [1.0, "#006D2C"],
    ]

    REDS = [
        [0.0, "#FFF5F0"],
        [0.25, "#FEE0D2"],
        [0.5, "#FC9272"],
        [0.75, "#DE2D26"],
        [1.0, "#A50F15"],
    ]

    ORANGES = [
        [0.0, "#FFF5EB"],
        [0.25, "#FEE6CE"],
        [0.5, "#FDAE6B"],
        [0.75, "#F16913"],
        [1.0, "#8C2D04"],
    ]

    PURPLES = [
        [0.0, "#FCFBFD"],
        [0.25, "#EFEDF5"],
        [0.5, "#BCBDDC"],
        [0.75, "#807DBA"],
        [1.0, "#54278F"],
    ]

    # Diverging scales
    RED_BLUE = [
        [0.0, "#D73027"],
        [0.25, "#FC8D59"],
        [0.5, "#FFFFBF"],
        [0.75, "#91BFDB"],
        [1.0, "#4575B4"],
    ]

    RED_GREEN = [
        [0.0, "#D73027"],
        [0.25, "#FC8D59"],
        [0.5, "#FFFFBF"],
        [0.75, "#91CF60"],
        [1.0, "#1A9850"],
    ]

    BROWN_TEAL = [
        [0.0, "#8C510A"],
        [0.25, "#D8B365"],
        [0.5, "#F6E8C3"],
        [0.75, "#5AB4AC"],
        [1.0, "#01665E"],
    ]

    # Specialized scales
    EFFICIENCY = [
        [0.0, "#C0392B"],
        [0.25, "#E74C3C"],
        [0.5, "#F39C12"],
        [0.75, "#F1C40F"],
        [1.0, "#2ECC71"],
    ]

    COST = [
        [0.0, "#2ECC71"],
        [0.25, "#F1C40F"],
        [0.5, "#F39C12"],
        [0.75, "#E74C3C"],
        [1.0, "#C0392B"],
    ]

    CARBON_INTENSITY = [
        [0.0, "#1A9850"],
        [0.25, "#91CF60"],
        [0.5, "#D9EF8B"],
        [0.75, "#FEE08B"],
        [1.0, "#D73027"],
    ]

    TEMPERATURE = [
        [0.0, "#313695"],
        [0.25, "#74ADD1"],
        [0.5, "#FFFFBF"],
        [0.75, "#F46D43"],
        [1.0, "#A50026"],
    ]

    @classmethod
    def get_scale(cls, scale_name: str) -> List[List]:
        """Get a gradient scale by name."""
        scale_map = {
            "blues": cls.BLUES,
            "greens": cls.GREENS,
            "reds": cls.REDS,
            "oranges": cls.ORANGES,
            "purples": cls.PURPLES,
            "red_blue": cls.RED_BLUE,
            "red_green": cls.RED_GREEN,
            "brown_teal": cls.BROWN_TEAL,
            "efficiency": cls.EFFICIENCY,
            "cost": cls.COST,
            "carbon_intensity": cls.CARBON_INTENSITY,
            "carbon": cls.CARBON_INTENSITY,
            "temperature": cls.TEMPERATURE,
        }
        return scale_map.get(scale_name.lower(), cls.BLUES)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FontConfig:
    """Font configuration for visualizations."""
    family: str = FontFamily.SYSTEM.value
    size_base: int = 12
    size_title: int = 18
    size_subtitle: int = 14
    size_axis_title: int = 12
    size_axis_label: int = 10
    size_legend: int = 11
    size_annotation: int = 10
    size_tooltip: int = 11
    weight_normal: int = 400
    weight_bold: int = 700
    line_height: float = 1.4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Plotly."""
        return {
            "family": self.family,
            "size": self.size_base,
        }

    def get_title_font(self) -> Dict[str, Any]:
        """Get title font configuration."""
        return {
            "family": self.family,
            "size": self.size_title,
            "weight": self.weight_bold,
        }

    def get_axis_font(self) -> Dict[str, Any]:
        """Get axis font configuration."""
        return {
            "family": self.family,
            "size": self.size_axis_label,
        }


@dataclass
class MarginConfig:
    """Margin configuration for charts."""
    left: int = 60
    right: int = 40
    top: int = 60
    bottom: int = 60
    pad: int = 4
    autoexpand: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Plotly margin format."""
        return {
            "l": self.left,
            "r": self.right,
            "t": self.top,
            "b": self.bottom,
            "pad": self.pad,
            "autoexpand": self.autoexpand,
        }

    @classmethod
    def compact(cls) -> "MarginConfig":
        """Create compact margin configuration."""
        return cls(left=40, right=20, top=40, bottom=40, pad=2)

    @classmethod
    def spacious(cls) -> "MarginConfig":
        """Create spacious margin configuration."""
        return cls(left=80, right=60, top=80, bottom=80, pad=8)

    @classmethod
    def dashboard(cls) -> "MarginConfig":
        """Create dashboard-optimized margin configuration."""
        return cls(left=50, right=30, top=50, bottom=50, pad=4)


@dataclass
class GridConfig:
    """Grid configuration for charts."""
    show_x: bool = True
    show_y: bool = True
    color: str = "#E0E0E0"
    width: float = 1.0
    dash: str = "solid"  # solid, dot, dash, longdash, dashdot
    zero_line_color: str = "#CCCCCC"
    zero_line_width: float = 1.5

    def to_x_axis_dict(self) -> Dict[str, Any]:
        """Convert to x-axis grid configuration."""
        return {
            "showgrid": self.show_x,
            "gridcolor": self.color,
            "gridwidth": self.width,
            "griddash": self.dash,
            "zeroline": True,
            "zerolinecolor": self.zero_line_color,
            "zerolinewidth": self.zero_line_width,
        }

    def to_y_axis_dict(self) -> Dict[str, Any]:
        """Convert to y-axis grid configuration."""
        return {
            "showgrid": self.show_y,
            "gridcolor": self.color,
            "gridwidth": self.width,
            "griddash": self.dash,
            "zeroline": True,
            "zerolinecolor": self.zero_line_color,
            "zerolinewidth": self.zero_line_width,
        }


@dataclass
class LegendConfig:
    """Legend configuration for charts."""
    show: bool = True
    orientation: str = "v"  # v (vertical), h (horizontal)
    x: float = 1.02
    y: float = 1.0
    x_anchor: str = "left"
    y_anchor: str = "top"
    bgcolor: str = "rgba(255,255,255,0.8)"
    border_color: str = "#CCCCCC"
    border_width: int = 1
    font_size: int = 11
    item_click: str = "toggle"  # toggle, toggleothers, false
    item_double_click: str = "toggleothers"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Plotly legend format."""
        return {
            "showlegend": self.show,
            "legend": {
                "orientation": self.orientation,
                "x": self.x,
                "y": self.y,
                "xanchor": self.x_anchor,
                "yanchor": self.y_anchor,
                "bgcolor": self.bgcolor,
                "bordercolor": self.border_color,
                "borderwidth": self.border_width,
                "font": {"size": self.font_size},
                "itemclick": self.item_click,
                "itemdoubleclick": self.item_double_click,
            }
        }

    @classmethod
    def horizontal_bottom(cls) -> "LegendConfig":
        """Create horizontal legend at bottom."""
        return cls(
            orientation="h",
            x=0.5,
            y=-0.15,
            x_anchor="center",
            y_anchor="top",
        )

    @classmethod
    def horizontal_top(cls) -> "LegendConfig":
        """Create horizontal legend at top."""
        return cls(
            orientation="h",
            x=0.5,
            y=1.1,
            x_anchor="center",
            y_anchor="bottom",
        )


@dataclass
class AnimationConfig:
    """Animation configuration for charts."""
    enabled: bool = True
    duration: int = AnimationSpeed.NORMAL.value
    easing: str = "cubic-in-out"
    redraw: bool = False
    frame_duration: int = 500
    transition_duration: int = 300

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Plotly animation format."""
        if not self.enabled:
            return {"transition": {"duration": 0}}
        return {
            "transition": {
                "duration": self.transition_duration,
                "easing": self.easing,
            },
            "frame": {
                "duration": self.frame_duration,
                "redraw": self.redraw,
            }
        }


@dataclass
class HoverConfig:
    """Hover/tooltip configuration for charts."""
    mode: str = "closest"  # closest, x, y, x unified, y unified
    bgcolor: str = "white"
    border_color: str = "#CCCCCC"
    border_width: int = 1
    font_size: int = 12
    font_family: str = FontFamily.SYSTEM.value
    align: str = "auto"  # auto, left, right
    namelength: int = 20

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Plotly hover format."""
        return {
            "hovermode": self.mode,
            "hoverlabel": {
                "bgcolor": self.bgcolor,
                "bordercolor": self.border_color,
                "font": {
                    "size": self.font_size,
                    "family": self.font_family,
                },
                "align": self.align,
                "namelength": self.namelength,
            }
        }


@dataclass
class ExportConfig:
    """Export configuration for charts."""
    default_format: ExportFormat = ExportFormat.PNG
    dpi: int = 300
    width: int = 1200
    height: int = 800
    scale: float = 2.0
    background_color: str = "white"
    include_plotlyjs: bool = True
    include_mathjax: bool = False
    full_html: bool = True
    cdn_url: str = "https://cdn.plot.ly/plotly-latest.min.js"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to export configuration dictionary."""
        return {
            "format": self.default_format.value,
            "width": self.width,
            "height": self.height,
            "scale": self.scale,
        }


@dataclass
class AccessibilityConfig:
    """Accessibility configuration (WCAG 2.1 Level AA)."""
    color_blind_safe: bool = True
    high_contrast: bool = False
    reduced_motion: bool = False
    screen_reader_descriptions: bool = True
    keyboard_navigation: bool = True
    focus_indicators: bool = True
    min_touch_target: int = 44  # 44px minimum for touch targets
    min_contrast_ratio: float = 4.5  # WCAG AA for normal text

    def apply_to_colors(self, colors: Dict[str, str]) -> Dict[str, str]:
        """Apply accessibility adjustments to colors."""
        if self.color_blind_safe:
            # Convert to colorblind-safe palette
            accessible_map = {
                "#E74C3C": "#D55E00",  # Red to orange-red
                "#2ECC71": "#009E73",  # Green to teal
                "#F39C12": "#E69F00",  # Orange to amber
                "#3498DB": "#0072B2",  # Blue to darker blue
                "#9B59B6": "#CC79A7",  # Purple to pink
            }
            return {k: accessible_map.get(v, v) for k, v in colors.items()}
        return colors


@dataclass
class CachingConfig:
    """Caching configuration for visualizations."""
    enabled: bool = True
    ttl_seconds: int = 300  # 5 minutes
    max_size_mb: int = 100
    strategy: str = "lru"  # lru, lfu, fifo
    persist_to_disk: bool = False
    cache_directory: str = ".cache/visualizations"

    def get_cache_key(self, data: Dict[str, Any], chart_type: str) -> str:
        """Generate cache key for visualization data."""
        import hashlib
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(f"{chart_type}:{data_str}".encode()).hexdigest()[:16]


@dataclass
class ResponsiveConfig:
    """Responsive design configuration."""
    enabled: bool = True
    breakpoints: Dict[str, int] = field(default_factory=lambda: {
        "mobile": ResponsiveBreakpoint.MOBILE.value,
        "tablet": ResponsiveBreakpoint.TABLET.value,
        "desktop": ResponsiveBreakpoint.DESKTOP.value,
        "large": ResponsiveBreakpoint.DESKTOP_LG.value,
    })
    scale_fonts: bool = True
    adjust_margins: bool = True
    hide_legend_mobile: bool = True
    simplify_on_mobile: bool = True

    def get_config_for_width(self, width: int) -> Dict[str, Any]:
        """Get responsive configuration for a given width."""
        if width < self.breakpoints["mobile"]:
            return {
                "font_scale": 0.8,
                "margin_scale": 0.6,
                "show_legend": False,
                "simplified": True,
            }
        elif width < self.breakpoints["tablet"]:
            return {
                "font_scale": 0.9,
                "margin_scale": 0.8,
                "show_legend": not self.hide_legend_mobile,
                "simplified": self.simplify_on_mobile,
            }
        elif width < self.breakpoints["desktop"]:
            return {
                "font_scale": 1.0,
                "margin_scale": 1.0,
                "show_legend": True,
                "simplified": False,
            }
        else:
            return {
                "font_scale": 1.1,
                "margin_scale": 1.0,
                "show_legend": True,
                "simplified": False,
            }


# =============================================================================
# THEME CONFIGURATION
# =============================================================================

@dataclass
class ThemeConfig:
    """Complete theme configuration for visualizations."""
    name: str = "default"
    mode: ThemeMode = ThemeMode.LIGHT

    # Background colors
    paper_bgcolor: str = "white"
    plot_bgcolor: str = "#FAFAFA"

    # Text colors
    title_color: str = "#2C3E50"
    text_color: str = "#333333"
    axis_color: str = "#666666"

    # Line colors
    axis_line_color: str = "#CCCCCC"
    grid_line_color: str = "#E0E0E0"

    # Component configs
    font: FontConfig = field(default_factory=FontConfig)
    margin: MarginConfig = field(default_factory=MarginConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    legend: LegendConfig = field(default_factory=LegendConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    hover: HoverConfig = field(default_factory=HoverConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    accessibility: AccessibilityConfig = field(default_factory=AccessibilityConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    responsive: ResponsiveConfig = field(default_factory=ResponsiveConfig)

    def to_layout_dict(self) -> Dict[str, Any]:
        """Convert to Plotly layout dictionary."""
        layout = {
            "paper_bgcolor": self.paper_bgcolor,
            "plot_bgcolor": self.plot_bgcolor,
            "font": {
                "family": self.font.family,
                "size": self.font.size_base,
                "color": self.text_color,
            },
            "title": {
                "font": {
                    "size": self.font.size_title,
                    "color": self.title_color,
                }
            },
            "margin": self.margin.to_dict(),
            "xaxis": {
                **self.grid.to_x_axis_dict(),
                "linecolor": self.axis_line_color,
                "tickfont": {"size": self.font.size_axis_label},
                "title": {"font": {"size": self.font.size_axis_title}},
            },
            "yaxis": {
                **self.grid.to_y_axis_dict(),
                "linecolor": self.axis_line_color,
                "tickfont": {"size": self.font.size_axis_label},
                "title": {"font": {"size": self.font.size_axis_title}},
            },
            **self.legend.to_dict(),
            **self.hover.to_dict(),
        }
        return layout

    @classmethod
    def dark_theme(cls) -> "ThemeConfig":
        """Create dark theme configuration."""
        return cls(
            name="dark",
            mode=ThemeMode.DARK,
            paper_bgcolor="#1E1E1E",
            plot_bgcolor="#2D2D2D",
            title_color="#FFFFFF",
            text_color="#E0E0E0",
            axis_color="#AAAAAA",
            axis_line_color="#444444",
            grid_line_color="#333333",
            grid=GridConfig(color="#333333"),
            hover=HoverConfig(bgcolor="#2D2D2D", border_color="#444444"),
        )

    @classmethod
    def high_contrast_theme(cls) -> "ThemeConfig":
        """Create high contrast theme for accessibility."""
        return cls(
            name="high_contrast",
            mode=ThemeMode.HIGH_CONTRAST,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            title_color="#000000",
            text_color="#000000",
            axis_color="#000000",
            axis_line_color="#000000",
            grid_line_color="#666666",
            grid=GridConfig(color="#666666", width=1.5),
            font=FontConfig(weight_normal=500, weight_bold=800),
            accessibility=AccessibilityConfig(
                color_blind_safe=True,
                high_contrast=True,
                min_contrast_ratio=7.0,
            ),
        )

    @classmethod
    def print_theme(cls) -> "ThemeConfig":
        """Create print-optimized theme."""
        return cls(
            name="print",
            mode=ThemeMode.PRINT,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            title_color="#000000",
            text_color="#000000",
            axis_color="#333333",
            axis_line_color="#666666",
            grid_line_color="#CCCCCC",
            animation=AnimationConfig(enabled=False),
            export=ExportConfig(dpi=300, scale=3.0),
        )

    @classmethod
    def presentation_theme(cls) -> "ThemeConfig":
        """Create presentation-optimized theme."""
        return cls(
            name="presentation",
            mode=ThemeMode.PRESENTATION,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8F9FA",
            font=FontConfig(
                size_base=14,
                size_title=24,
                size_subtitle=18,
                size_axis_title=14,
                size_axis_label=12,
                size_legend=13,
            ),
            margin=MarginConfig.spacious(),
            export=ExportConfig(width=1920, height=1080, dpi=150),
        )


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

@dataclass
class VisualizationConfig:
    """Global visualization configuration container."""

    # Theme settings
    theme: ThemeConfig = field(default_factory=ThemeConfig)

    # Color settings
    fuel_colors: Dict[str, str] = field(
        default_factory=lambda: FuelTypeColors.get_all_colors()
    )

    # Chart defaults
    default_chart_height: int = 400
    default_chart_width: int = 800
    max_data_points: int = 10000
    enable_webgl: bool = True

    # Localization
    locale: str = "en-US"
    currency: str = "USD"
    currency_symbol: str = "$"
    date_format: str = "%Y-%m-%d"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    number_format: str = ",.2f"

    # Units
    energy_unit: str = "MJ"
    power_unit: str = "kW"
    mass_unit: str = "kg"
    volume_unit: str = "L"
    temperature_unit: str = "C"
    emission_unit: str = "kg CO2e"
    currency_unit: str = "USD"

    # Performance
    lazy_loading: bool = True
    virtual_scrolling: bool = True
    chunk_size: int = 1000
    debounce_ms: int = 150

    # Watermarks and branding
    watermark_text: str = "GL-011 FUELCRAFT"
    watermark_opacity: float = 0.1
    show_watermark: bool = False
    logo_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "theme": self.theme.name,
            "default_height": self.default_chart_height,
            "default_width": self.default_chart_width,
            "locale": self.locale,
            "currency": self.currency,
            "energy_unit": self.energy_unit,
            "power_unit": self.power_unit,
        }

    def get_number_format(self, value: float, decimals: int = 2) -> str:
        """Format a number according to locale settings."""
        if self.locale.startswith("en"):
            return f"{value:,.{decimals}f}"
        elif self.locale.startswith("de"):
            formatted = f"{value:,.{decimals}f}"
            return formatted.replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            return f"{value:.{decimals}f}"

    def get_currency_format(self, value: float) -> str:
        """Format a currency value."""
        return f"{self.currency_symbol}{self.get_number_format(value)}"

    def get_percentage_format(self, value: float, decimals: int = 1) -> str:
        """Format a percentage value."""
        return f"{value:.{decimals}f}%"


# =============================================================================
# CHART-SPECIFIC CONFIGURATIONS
# =============================================================================

@dataclass
class WaterfallChartConfig:
    """Configuration specific to waterfall charts."""
    connector_color: str = "#888888"
    connector_width: int = 1
    connector_mode: str = "between"  # between, spanning
    increasing_color: str = "#2ECC71"
    decreasing_color: str = "#E74C3C"
    total_color: str = "#3498DB"
    show_connector: bool = True
    show_labels: bool = True
    label_position: str = "outside"  # inside, outside, auto
    orientation: str = "v"  # v (vertical), h (horizontal)
    bar_width: float = 0.7

    def to_trace_dict(self) -> Dict[str, Any]:
        """Convert to Plotly trace configuration."""
        return {
            "connector": {
                "line": {
                    "color": self.connector_color,
                    "width": self.connector_width,
                },
                "mode": self.connector_mode,
                "visible": self.show_connector,
            },
            "increasing": {"marker": {"color": self.increasing_color}},
            "decreasing": {"marker": {"color": self.decreasing_color}},
            "totals": {"marker": {"color": self.total_color}},
            "orientation": self.orientation,
            "width": self.bar_width,
            "textposition": self.label_position,
        }


@dataclass
class SankeyChartConfig:
    """Configuration specific to Sankey diagrams."""
    node_pad: int = 15
    node_thickness: int = 20
    node_line_color: str = "#333333"
    node_line_width: float = 0.5
    link_opacity: float = 0.5
    orientation: str = "h"  # h (horizontal), v (vertical)
    arrangement: str = "snap"  # snap, perpendicular, freeform, fixed
    value_format: str = ".2f"
    value_suffix: str = " kW"

    def to_trace_dict(self) -> Dict[str, Any]:
        """Convert to Plotly Sankey trace configuration."""
        return {
            "orientation": self.orientation,
            "arrangement": self.arrangement,
            "node": {
                "pad": self.node_pad,
                "thickness": self.node_thickness,
                "line": {
                    "color": self.node_line_color,
                    "width": self.node_line_width,
                },
            },
            "link": {
                "opacity": self.link_opacity,
            },
            "valueformat": self.value_format,
            "valuesuffix": self.value_suffix,
        }


@dataclass
class PieChartConfig:
    """Configuration specific to pie/donut charts."""
    hole: float = 0.0  # 0 for pie, 0.3-0.5 for donut
    pull: float = 0.0  # Slice pull distance
    rotation: int = 0  # Starting angle
    direction: str = "clockwise"  # clockwise, counterclockwise
    text_info: str = "percent+label"  # label, value, percent, or combinations
    text_position: str = "auto"  # inside, outside, auto, none
    show_legend: bool = True
    sort: bool = True

    def to_trace_dict(self) -> Dict[str, Any]:
        """Convert to Plotly pie trace configuration."""
        return {
            "hole": self.hole,
            "pull": self.pull,
            "rotation": self.rotation,
            "direction": self.direction,
            "textinfo": self.text_info,
            "textposition": self.text_position,
            "sort": self.sort,
        }


@dataclass
class TimeSeriesConfig:
    """Configuration specific to time series charts."""
    range_selector: bool = True
    range_slider: bool = True
    spike_mode: str = "across"  # across, toaxis, marker
    spike_color: str = "#999999"
    spike_thickness: int = 1
    moving_average_window: int = 7
    show_trend_line: bool = False
    fill_between: bool = False
    fill_color: str = "rgba(68, 68, 68, 0.3)"

    def get_range_selector_buttons(self) -> List[Dict[str, Any]]:
        """Get range selector button configuration."""
        return [
            {"count": 7, "label": "1W", "step": "day", "stepmode": "backward"},
            {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
            {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
            {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
            {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
            {"label": "All", "step": "all"},
        ]

    def to_layout_dict(self) -> Dict[str, Any]:
        """Convert to Plotly layout configuration."""
        layout = {
            "xaxis": {
                "rangeslider": {"visible": self.range_slider},
                "rangeselector": {
                    "visible": self.range_selector,
                    "buttons": self.get_range_selector_buttons(),
                },
                "showspikes": True,
                "spikemode": self.spike_mode,
                "spikecolor": self.spike_color,
                "spikethickness": self.spike_thickness,
            }
        }
        return layout


@dataclass
class DashboardConfig:
    """Configuration for dashboard layouts."""
    columns: int = 2
    row_height: int = 400
    gap: int = 20
    padding: int = 20
    auto_resize: bool = True
    draggable: bool = False
    resizable: bool = False
    show_panel_borders: bool = True
    panel_border_color: str = "#E0E0E0"
    panel_border_radius: int = 4
    panel_shadow: bool = True
    header_height: int = 50
    footer_height: int = 40

    def get_grid_style(self) -> Dict[str, str]:
        """Get CSS grid style for dashboard layout."""
        return {
            "display": "grid",
            "grid-template-columns": f"repeat({self.columns}, 1fr)",
            "gap": f"{self.gap}px",
            "padding": f"{self.padding}px",
        }

    def get_panel_style(self) -> Dict[str, str]:
        """Get CSS style for dashboard panels."""
        style = {
            "border-radius": f"{self.panel_border_radius}px",
            "background": "white",
            "padding": "15px",
        }
        if self.show_panel_borders:
            style["border"] = f"1px solid {self.panel_border_color}"
        if self.panel_shadow:
            style["box-shadow"] = "0 2px 4px rgba(0, 0, 0, 0.1)"
        return style


# =============================================================================
# CONFIGURATION FACTORY
# =============================================================================

class ConfigFactory:
    """Factory for creating visualization configurations."""

    _instance: Optional["ConfigFactory"] = None
    _config: Optional[VisualizationConfig] = None

    def __new__(cls) -> "ConfigFactory":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_config(cls) -> VisualizationConfig:
        """Get global visualization configuration."""
        if cls._config is None:
            cls._config = VisualizationConfig()
        return cls._config

    @classmethod
    def set_config(cls, config: VisualizationConfig) -> None:
        """Set global visualization configuration."""
        cls._config = config

    @classmethod
    def reset_config(cls) -> None:
        """Reset to default configuration."""
        cls._config = VisualizationConfig()

    @classmethod
    def create_theme(
        cls,
        mode: ThemeMode = ThemeMode.LIGHT,
        color_blind_safe: bool = False,
    ) -> ThemeConfig:
        """Create theme configuration based on mode."""
        if mode == ThemeMode.DARK:
            theme = ThemeConfig.dark_theme()
        elif mode == ThemeMode.HIGH_CONTRAST:
            theme = ThemeConfig.high_contrast_theme()
        elif mode == ThemeMode.PRINT:
            theme = ThemeConfig.print_theme()
        elif mode == ThemeMode.PRESENTATION:
            theme = ThemeConfig.presentation_theme()
        else:
            theme = ThemeConfig()

        if color_blind_safe:
            theme.accessibility.color_blind_safe = True

        return theme

    @classmethod
    def create_waterfall_config(
        cls,
        theme: Optional[ThemeConfig] = None,
    ) -> WaterfallChartConfig:
        """Create waterfall chart configuration."""
        config = WaterfallChartConfig()
        if theme and theme.accessibility.color_blind_safe:
            config.increasing_color = StatusColors.SUCCESS_A11Y
            config.decreasing_color = StatusColors.ERROR_A11Y
        return config

    @classmethod
    def create_sankey_config(
        cls,
        theme: Optional[ThemeConfig] = None,
    ) -> SankeyChartConfig:
        """Create Sankey chart configuration."""
        config = SankeyChartConfig()
        if theme and theme.mode == ThemeMode.DARK:
            config.node_line_color = "#666666"
        return config

    @classmethod
    def create_time_series_config(
        cls,
        enable_range_selector: bool = True,
        enable_range_slider: bool = True,
    ) -> TimeSeriesConfig:
        """Create time series chart configuration."""
        return TimeSeriesConfig(
            range_selector=enable_range_selector,
            range_slider=enable_range_slider,
        )

    @classmethod
    def create_dashboard_config(
        cls,
        columns: int = 2,
        compact: bool = False,
    ) -> DashboardConfig:
        """Create dashboard configuration."""
        config = DashboardConfig(columns=columns)
        if compact:
            config.row_height = 300
            config.gap = 10
            config.padding = 10
        return config

    @classmethod
    def load_from_file(cls, filepath: str) -> VisualizationConfig:
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        config = VisualizationConfig()

        # Apply loaded settings
        if "theme" in data:
            theme_mode = ThemeMode(data["theme"].get("mode", "light"))
            config.theme = cls.create_theme(theme_mode)

        if "locale" in data:
            config.locale = data["locale"]

        if "currency" in data:
            config.currency = data["currency"]

        return config

    @classmethod
    def save_to_file(cls, config: VisualizationConfig, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(config.to_dict(), f, indent=2)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_default_config() -> VisualizationConfig:
    """Get default visualization configuration."""
    return ConfigFactory.get_config()


def get_fuel_color(fuel_type: str) -> str:
    """Get color for a fuel type."""
    return FuelTypeColors.get_color(fuel_type)


def get_cost_color(category: str) -> str:
    """Get color for a cost category."""
    return CostCategoryColors.get_color(category)


def get_emission_color(emission_type: str) -> str:
    """Get color for an emission type."""
    return EmissionColors.get_color(emission_type)


def get_status_color(status: str, accessible: bool = False) -> str:
    """Get color for a status indicator."""
    return StatusColors.get_color(status, accessible)


def get_gradient_scale(scale_name: str) -> List[List]:
    """Get a gradient color scale."""
    return GradientScales.get_scale(scale_name)


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to RGBA string."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def adjust_color_brightness(hex_color: str, factor: float) -> str:
    """Adjust color brightness. Factor > 1 = lighter, < 1 = darker."""
    hex_color = hex_color.lstrip("#")
    r = min(255, int(int(hex_color[0:2], 16) * factor))
    g = min(255, int(int(hex_color[2:4], 16) * factor))
    b = min(255, int(int(hex_color[4:6], 16) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def blend_colors(color1: str, color2: str, ratio: float = 0.5) -> str:
    """Blend two colors. Ratio 0 = color1, 1 = color2."""
    c1 = color1.lstrip("#")
    c2 = color2.lstrip("#")

    r = int(int(c1[0:2], 16) * (1 - ratio) + int(c2[0:2], 16) * ratio)
    g = int(int(c1[2:4], 16) * (1 - ratio) + int(c2[2:4], 16) * ratio)
    b = int(int(c1[4:6], 16) * (1 - ratio) + int(c2[4:6], 16) * ratio)

    return f"#{r:02x}{g:02x}{b:02x}"


def generate_color_palette(
    base_color: str,
    count: int,
    variation: str = "brightness",
) -> List[str]:
    """Generate a color palette based on a base color."""
    colors = []

    if variation == "brightness":
        for i in range(count):
            factor = 0.6 + (i / (count - 1)) * 0.8 if count > 1 else 1.0
            colors.append(adjust_color_brightness(base_color, factor))
    elif variation == "opacity":
        for i in range(count):
            alpha = 0.3 + (i / (count - 1)) * 0.7 if count > 1 else 1.0
            colors.append(hex_to_rgba(base_color, alpha))
    else:
        colors = [base_color] * count

    return colors


# =============================================================================
# PLOTLY CONFIGURATION HELPERS
# =============================================================================

def get_plotly_config(interactive: bool = True) -> Dict[str, Any]:
    """Get Plotly configuration dictionary."""
    config = {
        "responsive": True,
        "displayModeBar": interactive,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
            "autoScale2d",
            "toggleSpikelines",
        ] if interactive else [],
    }

    if not interactive:
        config["staticPlot"] = True

    return config


def get_modebar_buttons() -> List[str]:
    """Get list of modebar buttons to show."""
    return [
        "zoom2d",
        "pan2d",
        "zoomIn2d",
        "zoomOut2d",
        "resetScale2d",
        "toImage",
    ]


def create_annotation(
    text: str,
    x: float,
    y: float,
    x_ref: str = "paper",
    y_ref: str = "paper",
    font_size: int = 12,
    font_color: str = "#333333",
    show_arrow: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Create a Plotly annotation dictionary."""
    annotation = {
        "text": text,
        "x": x,
        "y": y,
        "xref": x_ref,
        "yref": y_ref,
        "font": {"size": font_size, "color": font_color},
        "showarrow": show_arrow,
    }
    annotation.update(kwargs)
    return annotation


def create_shape(
    shape_type: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    line_color: str = "#888888",
    line_width: int = 1,
    line_dash: str = "solid",
    fill_color: Optional[str] = None,
    opacity: float = 1.0,
    x_ref: str = "x",
    y_ref: str = "y",
    **kwargs,
) -> Dict[str, Any]:
    """Create a Plotly shape dictionary."""
    shape = {
        "type": shape_type,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "xref": x_ref,
        "yref": y_ref,
        "line": {
            "color": line_color,
            "width": line_width,
            "dash": line_dash,
        },
        "opacity": opacity,
    }
    if fill_color:
        shape["fillcolor"] = fill_color
    shape.update(kwargs)
    return shape


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example usage of visualization configuration."""

    # Get default configuration
    config = get_default_config()
    print(f"Default theme: {config.theme.name}")
    print(f"Locale: {config.locale}")

    # Create dark theme
    dark_theme = ConfigFactory.create_theme(ThemeMode.DARK)
    print(f"Dark theme background: {dark_theme.paper_bgcolor}")

    # Get fuel colors
    natural_gas_color = get_fuel_color("natural_gas")
    coal_color = get_fuel_color("coal")
    print(f"Natural gas color: {natural_gas_color}")
    print(f"Coal color: {coal_color}")

    # Create waterfall config
    waterfall_config = ConfigFactory.create_waterfall_config()
    print(f"Waterfall increasing color: {waterfall_config.increasing_color}")

    # Create dashboard config
    dashboard_config = ConfigFactory.create_dashboard_config(columns=3)
    print(f"Dashboard columns: {dashboard_config.columns}")

    # Color utilities
    rgba_color = hex_to_rgba("#3498DB", 0.5)
    print(f"RGBA color: {rgba_color}")

    lighter_color = adjust_color_brightness("#3498DB", 1.3)
    print(f"Lighter color: {lighter_color}")

    palette = generate_color_palette("#3498DB", 5)
    print(f"Color palette: {palette}")


if __name__ == "__main__":
    example_usage()
