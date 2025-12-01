# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Fuel Cost Waterfall Visualization Module.

Comprehensive waterfall chart visualization for fuel cost breakdown analysis.
Shows cost contributions from baseline, fuel switching savings, blend optimization,
procurement optimization, and market timing.

Author: GreenLang Team
Version: 1.0.0
Standards: WCAG 2.1 Level AA, ISO 12647-2

Features:
- Interactive waterfall charts with drill-down capability
- Multiple cost category breakdown
- Comparison views (actual vs budget, period vs period)
- Cumulative and incremental views
- Export to PNG/PDF/SVG/JSON
- Responsive design with mobile optimization
- Accessibility compliant color schemes
- Real-time data update support
- Caching for performance optimization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
import hashlib
import math
import logging
from decimal import Decimal, ROUND_HALF_UP

# Local imports
from .config import (
    ThemeConfig,
    ThemeMode,
    VisualizationConfig,
    ConfigFactory,
    WaterfallChartConfig,
    CostCategoryColors,
    FuelTypeColors,
    StatusColors,
    GradientScales,
    FontConfig,
    MarginConfig,
    LegendConfig,
    AnimationConfig,
    HoverConfig,
    ExportConfig,
    AccessibilityConfig,
    ExportFormat,
    ChartType,
    get_default_config,
    get_cost_color,
    hex_to_rgba,
    adjust_color_brightness,
    get_plotly_config,
    create_annotation,
    create_shape,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CostCategory(Enum):
    """Cost categories for waterfall analysis."""
    BASELINE_COST = "baseline_cost"
    FUEL_COST = "fuel_cost"
    FUEL_SWITCHING_SAVINGS = "fuel_switching_savings"
    BLEND_OPTIMIZATION_SAVINGS = "blend_optimization_savings"
    PROCUREMENT_OPTIMIZATION = "procurement_optimization"
    MARKET_TIMING_SAVINGS = "market_timing_savings"
    CARBON_COST = "carbon_cost"
    CARBON_SAVINGS = "carbon_savings"
    DELIVERY_COST = "delivery_cost"
    STORAGE_COST = "storage_cost"
    HANDLING_COST = "handling_cost"
    TAX = "tax"
    SUBSIDY = "subsidy"
    PENALTY = "penalty"
    HEDGING_COST = "hedging_cost"
    OPTIMIZATION_BENEFIT = "optimization_benefit"
    NET_COST = "net_cost"
    TOTAL = "total"

    @property
    def display_name(self) -> str:
        """Get human-readable display name."""
        names = {
            "baseline_cost": "Baseline Cost",
            "fuel_cost": "Fuel Cost",
            "fuel_switching_savings": "Fuel Switching Savings",
            "blend_optimization_savings": "Blend Optimization",
            "procurement_optimization": "Procurement Optimization",
            "market_timing_savings": "Market Timing",
            "carbon_cost": "Carbon Cost",
            "carbon_savings": "Carbon Savings",
            "delivery_cost": "Delivery Cost",
            "storage_cost": "Storage Cost",
            "handling_cost": "Handling Cost",
            "tax": "Taxes",
            "subsidy": "Subsidies",
            "penalty": "Penalties",
            "hedging_cost": "Hedging Cost",
            "optimization_benefit": "Optimization Benefit",
            "net_cost": "Net Cost",
            "total": "Total",
        }
        return names.get(self.value, self.value.replace("_", " ").title())

    @property
    def is_cost(self) -> bool:
        """Check if this category represents a cost (positive value)."""
        costs = {
            "baseline_cost", "fuel_cost", "carbon_cost", "delivery_cost",
            "storage_cost", "handling_cost", "tax", "penalty", "hedging_cost"
        }
        return self.value in costs

    @property
    def is_saving(self) -> bool:
        """Check if this category represents a saving (negative value)."""
        savings = {
            "fuel_switching_savings", "blend_optimization_savings",
            "procurement_optimization", "market_timing_savings",
            "carbon_savings", "subsidy", "optimization_benefit"
        }
        return self.value in savings

    @property
    def is_total(self) -> bool:
        """Check if this is a total category."""
        return self.value in {"net_cost", "total"}


class WaterfallType(Enum):
    """Types of waterfall visualizations."""
    STANDARD = "standard"
    CUMULATIVE = "cumulative"
    COMPARISON = "comparison"
    VARIANCE = "variance"
    BRIDGE = "bridge"
    DETAILED = "detailed"


class WaterfallOrientation(Enum):
    """Waterfall chart orientation."""
    VERTICAL = "v"
    HORIZONTAL = "h"


class ValueDisplayMode(Enum):
    """How to display values on the chart."""
    ABSOLUTE = "absolute"
    PERCENTAGE = "percentage"
    BOTH = "both"
    NONE = "none"


class DrillDownLevel(Enum):
    """Drill-down hierarchy levels."""
    SUMMARY = 0
    CATEGORY = 1
    SUBCATEGORY = 2
    DETAIL = 3


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CostItem:
    """Individual cost item in waterfall."""
    category: CostCategory
    value: float
    label: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None
    subcategories: List["CostItem"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_subtotal: bool = False
    is_total: bool = False
    drill_down_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.label is None:
            self.label = self.category.display_name
        if self.color is None:
            self.color = get_cost_color(self.category.value)

    @property
    def measure(self) -> str:
        """Get Plotly measure type for waterfall."""
        if self.is_total:
            return "total"
        elif self.value >= 0:
            return "relative"
        else:
            return "relative"

    @property
    def formatted_value(self) -> str:
        """Get formatted value string."""
        if self.value >= 0:
            return f"${self.value:,.2f}"
        else:
            return f"-${abs(self.value):,.2f}"

    @property
    def formatted_delta(self) -> str:
        """Get formatted delta string (with +/- prefix)."""
        if self.value >= 0:
            return f"+${self.value:,.2f}"
        else:
            return f"-${abs(self.value):,.2f}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "value": self.value,
            "label": self.label,
            "color": self.color,
            "description": self.description,
            "subcategories": [s.to_dict() for s in self.subcategories],
            "metadata": self.metadata,
            "is_subtotal": self.is_subtotal,
            "is_total": self.is_total,
        }


@dataclass
class CostBreakdown:
    """Complete cost breakdown for waterfall visualization."""
    items: List[CostItem]
    currency: str = "USD"
    currency_symbol: str = "$"
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    baseline_value: float = 0.0
    final_value: float = 0.0
    total_savings: float = 0.0
    total_costs: float = 0.0
    net_change: float = 0.0
    net_change_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self):
        """Calculate derived values."""
        if not self.provenance_hash:
            self._calculate_provenance()
        self._calculate_totals()

    def _calculate_provenance(self):
        """Calculate provenance hash for data lineage."""
        data = {
            "items": [(i.category.value, i.value) for i in self.items],
            "currency": self.currency,
            "period": f"{self.period_start}:{self.period_end}",
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _calculate_totals(self):
        """Calculate total costs and savings."""
        self.total_costs = sum(
            item.value for item in self.items
            if item.value > 0 and not item.is_total
        )
        self.total_savings = sum(
            abs(item.value) for item in self.items
            if item.value < 0 and not item.is_total
        )
        self.net_change = self.total_costs - self.total_savings

        if self.baseline_value != 0:
            self.final_value = self.baseline_value + self.net_change
            self.net_change_percent = (self.net_change / self.baseline_value) * 100

    def get_item_by_category(self, category: CostCategory) -> Optional[CostItem]:
        """Get item by category."""
        for item in self.items:
            if item.category == category:
                return item
        return None

    def add_item(self, item: CostItem) -> None:
        """Add a cost item."""
        self.items.append(item)
        self._calculate_totals()
        self._calculate_provenance()

    def remove_item(self, category: CostCategory) -> bool:
        """Remove a cost item by category."""
        for i, item in enumerate(self.items):
            if item.category == category:
                self.items.pop(i)
                self._calculate_totals()
                self._calculate_provenance()
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "items": [item.to_dict() for item in self.items],
            "currency": self.currency,
            "currency_symbol": self.currency_symbol,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "baseline_value": self.baseline_value,
            "final_value": self.final_value,
            "total_savings": self.total_savings,
            "total_costs": self.total_costs,
            "net_change": self.net_change,
            "net_change_percent": self.net_change_percent,
            "metadata": self.metadata,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
        }


@dataclass
class ComparisonData:
    """Data for comparison waterfall charts."""
    actual: CostBreakdown
    budget: Optional[CostBreakdown] = None
    previous_period: Optional[CostBreakdown] = None
    forecast: Optional[CostBreakdown] = None
    target: Optional[CostBreakdown] = None

    def get_variance(self, category: CostCategory) -> Optional[float]:
        """Get variance between actual and budget for a category."""
        if not self.budget:
            return None
        actual_item = self.actual.get_item_by_category(category)
        budget_item = self.budget.get_item_by_category(category)
        if actual_item and budget_item:
            return actual_item.value - budget_item.value
        return None

    def get_variance_percent(self, category: CostCategory) -> Optional[float]:
        """Get variance percentage between actual and budget."""
        if not self.budget:
            return None
        actual_item = self.actual.get_item_by_category(category)
        budget_item = self.budget.get_item_by_category(category)
        if actual_item and budget_item and budget_item.value != 0:
            return ((actual_item.value - budget_item.value) / budget_item.value) * 100
        return None


@dataclass
class DrillDownNode:
    """Node in drill-down hierarchy."""
    id: str
    label: str
    value: float
    level: DrillDownLevel
    parent_id: Optional[str] = None
    children: List["DrillDownNode"] = field(default_factory=list)
    color: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_children_total(self) -> float:
        """Get total value of children."""
        return sum(child.value for child in self.children)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "value": self.value,
            "level": self.level.value,
            "parent_id": self.parent_id,
            "children": [c.to_dict() for c in self.children],
            "color": self.color,
            "metadata": self.metadata,
        }


@dataclass
class WaterfallAnnotation:
    """Custom annotation for waterfall chart."""
    text: str
    x: Union[int, float, str]
    y: float
    x_anchor: str = "center"
    y_anchor: str = "bottom"
    font_size: int = 10
    font_color: str = "#333333"
    bgcolor: Optional[str] = None
    border_color: Optional[str] = None
    show_arrow: bool = False
    arrow_head: int = 2
    arrow_size: float = 1.0
    arrow_width: float = 1.0
    arrow_color: str = "#333333"

    def to_plotly_dict(self) -> Dict[str, Any]:
        """Convert to Plotly annotation format."""
        annotation = {
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "xanchor": self.x_anchor,
            "yanchor": self.y_anchor,
            "font": {
                "size": self.font_size,
                "color": self.font_color,
            },
            "showarrow": self.show_arrow,
        }
        if self.bgcolor:
            annotation["bgcolor"] = self.bgcolor
        if self.border_color:
            annotation["bordercolor"] = self.border_color
        if self.show_arrow:
            annotation.update({
                "arrowhead": self.arrow_head,
                "arrowsize": self.arrow_size,
                "arrowwidth": self.arrow_width,
                "arrowcolor": self.arrow_color,
            })
        return annotation


# =============================================================================
# CHART CONFIGURATION
# =============================================================================

@dataclass
class WaterfallChartOptions:
    """Configuration options for waterfall chart."""
    # Display options
    title: str = "Fuel Cost Breakdown"
    subtitle: Optional[str] = None
    orientation: WaterfallOrientation = WaterfallOrientation.VERTICAL
    waterfall_type: WaterfallType = WaterfallType.STANDARD
    value_display: ValueDisplayMode = ValueDisplayMode.ABSOLUTE

    # Style options
    bar_width: float = 0.7
    bar_gap: float = 0.3
    connector_visible: bool = True
    connector_color: str = "#888888"
    connector_width: int = 1
    connector_mode: str = "between"

    # Color options
    increasing_color: str = "#E74C3C"
    decreasing_color: str = "#2ECC71"
    total_color: str = "#3498DB"
    subtotal_color: str = "#9B59B6"
    use_category_colors: bool = True
    color_blind_safe: bool = False

    # Label options
    show_labels: bool = True
    label_position: str = "outside"
    label_format: str = ",.0f"
    show_percentage: bool = False
    percentage_format: str = ".1f"

    # Axis options
    y_axis_title: str = "Cost (USD)"
    x_axis_title: str = ""
    show_zero_line: bool = True
    zero_line_color: str = "#CCCCCC"
    zero_line_width: float = 1.5

    # Legend options
    show_legend: bool = True
    legend_position: str = "right"

    # Interaction options
    enable_hover: bool = True
    enable_drill_down: bool = False
    enable_selection: bool = False

    # Size options
    width: Optional[int] = None
    height: Optional[int] = None
    auto_size: bool = True

    # Animation options
    animate: bool = True
    animation_duration: int = 500

    # Additional options
    show_cumulative_line: bool = False
    show_threshold: bool = False
    threshold_value: Optional[float] = None
    threshold_color: str = "#E74C3C"
    show_annotations: bool = True
    custom_annotations: List[WaterfallAnnotation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "orientation": self.orientation.value,
            "waterfall_type": self.waterfall_type.value,
            "bar_width": self.bar_width,
            "show_labels": self.show_labels,
            "enable_drill_down": self.enable_drill_down,
        }


# =============================================================================
# WATERFALL CHART ENGINE
# =============================================================================

class WaterfallChartEngine:
    """
    Engine for generating fuel cost waterfall visualizations.

    Supports multiple waterfall types, drill-down capability, comparison views,
    and comprehensive export options.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        theme: Optional[ThemeConfig] = None,
    ):
        """
        Initialize waterfall chart engine.

        Args:
            config: Global visualization configuration
            theme: Theme configuration for styling
        """
        self.config = config or get_default_config()
        self.theme = theme or self.config.theme
        self._cache: Dict[str, Any] = {}
        self._drill_down_stack: List[DrillDownNode] = []

        # Apply accessibility settings
        if self.theme.accessibility.color_blind_safe:
            self._apply_accessible_colors()

    def _apply_accessible_colors(self) -> None:
        """Apply color-blind safe color palette."""
        self._increasing_color = StatusColors.ERROR_A11Y
        self._decreasing_color = StatusColors.SUCCESS_A11Y
        self._total_color = StatusColors.INFO_A11Y
    else:
        self._increasing_color = CostCategoryColors.FUEL_COST
        self._decreasing_color = CostCategoryColors.SWITCHING_SAVINGS
        self._total_color = CostCategoryColors.BASELINE

    def generate(
        self,
        data: CostBreakdown,
        options: Optional[WaterfallChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate waterfall chart from cost breakdown data.

        Args:
            data: Cost breakdown data
            options: Chart configuration options

        Returns:
            Plotly-compatible chart specification
        """
        options = options or WaterfallChartOptions()

        # Check cache
        cache_key = self._get_cache_key(data, options)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build chart based on type
        if options.waterfall_type == WaterfallType.STANDARD:
            chart = self._build_standard_waterfall(data, options)
        elif options.waterfall_type == WaterfallType.CUMULATIVE:
            chart = self._build_cumulative_waterfall(data, options)
        elif options.waterfall_type == WaterfallType.COMPARISON:
            chart = self._build_comparison_waterfall(data, options)
        elif options.waterfall_type == WaterfallType.VARIANCE:
            chart = self._build_variance_waterfall(data, options)
        elif options.waterfall_type == WaterfallType.BRIDGE:
            chart = self._build_bridge_waterfall(data, options)
        elif options.waterfall_type == WaterfallType.DETAILED:
            chart = self._build_detailed_waterfall(data, options)
        else:
            chart = self._build_standard_waterfall(data, options)

        # Add annotations if enabled
        if options.show_annotations:
            chart = self._add_annotations(chart, data, options)

        # Add cumulative line if enabled
        if options.show_cumulative_line:
            chart = self._add_cumulative_line(chart, data, options)

        # Add threshold line if enabled
        if options.show_threshold and options.threshold_value is not None:
            chart = self._add_threshold_line(chart, options)

        # Cache result
        self._cache[cache_key] = chart

        return chart

    def _get_cache_key(
        self,
        data: CostBreakdown,
        options: WaterfallChartOptions,
    ) -> str:
        """Generate cache key for chart data."""
        key_data = {
            "provenance": data.provenance_hash,
            "type": options.waterfall_type.value,
            "orientation": options.orientation.value,
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _build_standard_waterfall(
        self,
        data: CostBreakdown,
        options: WaterfallChartOptions,
    ) -> Dict[str, Any]:
        """Build standard waterfall chart."""
        x_values = []
        y_values = []
        measures = []
        text_values = []
        colors = []
        hover_texts = []

        # Add baseline if present
        if data.baseline_value != 0:
            x_values.append("Baseline")
            y_values.append(data.baseline_value)
            measures.append("absolute")
            text_values.append(f"{data.currency_symbol}{data.baseline_value:,.0f}")
            colors.append(options.total_color)
            hover_texts.append(
                f"<b>Baseline Cost</b><br>"
                f"Value: {data.currency_symbol}{data.baseline_value:,.2f}"
            )

        # Add cost items
        for item in data.items:
            if item.is_total:
                continue

            x_values.append(item.label)
            y_values.append(item.value)
            measures.append(item.measure)

            # Format text
            if options.value_display == ValueDisplayMode.PERCENTAGE:
                if data.baseline_value != 0:
                    pct = (item.value / data.baseline_value) * 100
                    text_values.append(f"{pct:+.1f}%")
                else:
                    text_values.append("")
            elif options.value_display == ValueDisplayMode.BOTH:
                if data.baseline_value != 0:
                    pct = (item.value / data.baseline_value) * 100
                    text_values.append(
                        f"{data.currency_symbol}{item.value:+,.0f} ({pct:+.1f}%)"
                    )
                else:
                    text_values.append(f"{data.currency_symbol}{item.value:+,.0f}")
            elif options.value_display == ValueDisplayMode.ABSOLUTE:
                text_values.append(f"{data.currency_symbol}{item.value:+,.0f}")
            else:
                text_values.append("")

            # Determine color
            if options.use_category_colors and item.color:
                colors.append(item.color)
            elif item.value >= 0:
                colors.append(options.increasing_color)
            else:
                colors.append(options.decreasing_color)

            # Build hover text
            hover_text = f"<b>{item.label}</b><br>"
            hover_text += f"Value: {data.currency_symbol}{item.value:,.2f}<br>"
            if item.description:
                hover_text += f"{item.description}<br>"
            if item.category.is_saving:
                hover_text += "Type: Savings"
            elif item.category.is_cost:
                hover_text += "Type: Cost"
            hover_texts.append(hover_text)

        # Add total
        total_item = None
        for item in data.items:
            if item.is_total:
                total_item = item
                break

        if total_item:
            x_values.append(total_item.label)
            y_values.append(total_item.value)
            measures.append("total")
            text_values.append(f"{data.currency_symbol}{total_item.value:,.0f}")
            colors.append(options.total_color)
            hover_texts.append(
                f"<b>{total_item.label}</b><br>"
                f"Final Value: {data.currency_symbol}{total_item.value:,.2f}"
            )
        elif data.final_value != 0:
            x_values.append("Total")
            y_values.append(data.final_value)
            measures.append("total")
            text_values.append(f"{data.currency_symbol}{data.final_value:,.0f}")
            colors.append(options.total_color)
            hover_texts.append(
                f"<b>Total</b><br>"
                f"Final Value: {data.currency_symbol}{data.final_value:,.2f}"
            )

        # Build trace
        trace = {
            "type": "waterfall",
            "name": "Cost Breakdown",
            "orientation": options.orientation.value,
            "measure": measures,
            "textposition": options.label_position,
            "hovertemplate": "%{customdata}<extra></extra>",
            "customdata": hover_texts,
            "connector": {
                "line": {
                    "color": options.connector_color,
                    "width": options.connector_width,
                },
                "mode": options.connector_mode,
                "visible": options.connector_visible,
            },
            "increasing": {"marker": {"color": options.increasing_color}},
            "decreasing": {"marker": {"color": options.decreasing_color}},
            "totals": {"marker": {"color": options.total_color}},
        }

        if options.orientation == WaterfallOrientation.VERTICAL:
            trace["x"] = x_values
            trace["y"] = y_values
        else:
            trace["x"] = y_values
            trace["y"] = x_values

        if options.show_labels:
            trace["text"] = text_values

        if options.use_category_colors:
            trace["marker"] = {"color": colors}

        # Build layout
        layout = self._build_layout(data, options)

        return {
            "data": [trace],
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

    def _build_cumulative_waterfall(
        self,
        data: CostBreakdown,
        options: WaterfallChartOptions,
    ) -> Dict[str, Any]:
        """Build cumulative waterfall chart."""
        # Start with standard waterfall
        chart = self._build_standard_waterfall(data, options)

        # Calculate cumulative values
        cumulative = []
        running_total = data.baseline_value

        for item in data.items:
            if not item.is_total:
                running_total += item.value
                cumulative.append(running_total)

        # Add cumulative line trace
        x_values = [item.label for item in data.items if not item.is_total]

        cumulative_trace = {
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Cumulative",
            "x": x_values if options.orientation == WaterfallOrientation.VERTICAL else cumulative,
            "y": cumulative if options.orientation == WaterfallOrientation.VERTICAL else x_values,
            "line": {
                "color": "#333333",
                "width": 2,
                "dash": "dot",
            },
            "marker": {
                "size": 6,
                "color": "#333333",
            },
            "hovertemplate": "Cumulative: %{y:,.2f}<extra></extra>",
        }

        chart["data"].append(cumulative_trace)
        return chart

    def _build_comparison_waterfall(
        self,
        data: CostBreakdown,
        options: WaterfallChartOptions,
    ) -> Dict[str, Any]:
        """Build comparison waterfall chart (actual vs budget)."""
        # This would typically receive ComparisonData
        # For now, build a standard waterfall
        return self._build_standard_waterfall(data, options)

    def _build_variance_waterfall(
        self,
        data: CostBreakdown,
        options: WaterfallChartOptions,
    ) -> Dict[str, Any]:
        """Build variance analysis waterfall chart."""
        x_values = []
        y_values = []
        measures = []
        text_values = []
        colors = []
        hover_texts = []

        # Add budget baseline
        x_values.append("Budget")
        y_values.append(data.baseline_value)
        measures.append("absolute")
        text_values.append(f"{data.currency_symbol}{data.baseline_value:,.0f}")
        colors.append(options.total_color)
        hover_texts.append(f"<b>Budget</b><br>Value: {data.currency_symbol}{data.baseline_value:,.2f}")

        # Calculate variances
        for item in data.items:
            if item.is_total:
                continue

            variance = item.value
            x_values.append(f"{item.label} Variance")
            y_values.append(variance)
            measures.append("relative")

            if variance >= 0:
                text_values.append(f"+{data.currency_symbol}{variance:,.0f}")
                colors.append(options.increasing_color)
                variance_type = "Unfavorable"
            else:
                text_values.append(f"-{data.currency_symbol}{abs(variance):,.0f}")
                colors.append(options.decreasing_color)
                variance_type = "Favorable"

            hover_texts.append(
                f"<b>{item.label} Variance</b><br>"
                f"Amount: {data.currency_symbol}{variance:,.2f}<br>"
                f"Type: {variance_type}"
            )

        # Add actual total
        x_values.append("Actual")
        y_values.append(data.final_value)
        measures.append("total")
        text_values.append(f"{data.currency_symbol}{data.final_value:,.0f}")
        colors.append(options.total_color)
        hover_texts.append(f"<b>Actual</b><br>Value: {data.currency_symbol}{data.final_value:,.2f}")

        trace = {
            "type": "waterfall",
            "name": "Variance Analysis",
            "orientation": options.orientation.value,
            "measure": measures,
            "textposition": options.label_position,
            "hovertemplate": "%{customdata}<extra></extra>",
            "customdata": hover_texts,
            "connector": {
                "line": {
                    "color": options.connector_color,
                    "width": options.connector_width,
                },
                "visible": options.connector_visible,
            },
            "marker": {"color": colors},
        }

        if options.orientation == WaterfallOrientation.VERTICAL:
            trace["x"] = x_values
            trace["y"] = y_values
        else:
            trace["x"] = y_values
            trace["y"] = x_values

        if options.show_labels:
            trace["text"] = text_values

        layout = self._build_layout(data, options)
        layout["title"]["text"] = "Cost Variance Analysis"

        return {
            "data": [trace],
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

    def _build_bridge_waterfall(
        self,
        data: CostBreakdown,
        options: WaterfallChartOptions,
    ) -> Dict[str, Any]:
        """Build bridge waterfall (start-to-end analysis)."""
        # Similar to standard but emphasizes start and end points
        chart = self._build_standard_waterfall(data, options)

        # Add start/end markers
        if len(chart["data"]) > 0:
            trace = chart["data"][0]
            if "marker" not in trace:
                trace["marker"] = {}

            # Highlight first and last bars
            if options.orientation == WaterfallOrientation.VERTICAL:
                n_bars = len(trace["x"])
            else:
                n_bars = len(trace["y"])

            if n_bars > 0:
                trace["marker"]["line"] = {
                    "width": [2 if i in [0, n_bars - 1] else 0 for i in range(n_bars)],
                    "color": "#333333",
                }

        return chart

    def _build_detailed_waterfall(
        self,
        data: CostBreakdown,
        options: WaterfallChartOptions,
    ) -> Dict[str, Any]:
        """Build detailed waterfall with subcategory breakdown."""
        x_values = []
        y_values = []
        measures = []
        text_values = []
        colors = []
        hover_texts = []

        # Add baseline
        if data.baseline_value != 0:
            x_values.append("Baseline")
            y_values.append(data.baseline_value)
            measures.append("absolute")
            text_values.append(f"{data.currency_symbol}{data.baseline_value:,.0f}")
            colors.append(options.total_color)
            hover_texts.append(f"<b>Baseline</b><br>Value: {data.currency_symbol}{data.baseline_value:,.2f}")

        # Add items with subcategories expanded
        for item in data.items:
            if item.is_total:
                continue

            # If item has subcategories, expand them
            if item.subcategories:
                # Add category header as subtotal
                x_values.append(f"[{item.label}]")
                y_values.append(0)  # Placeholder
                measures.append("relative")
                text_values.append("")
                colors.append(options.subtotal_color)
                hover_texts.append(f"<b>{item.label}</b>")

                # Add each subcategory
                for sub in item.subcategories:
                    x_values.append(f"  {sub.label}")
                    y_values.append(sub.value)
                    measures.append("relative")
                    text_values.append(f"{data.currency_symbol}{sub.value:+,.0f}")
                    colors.append(sub.color or (options.increasing_color if sub.value >= 0 else options.decreasing_color))
                    hover_texts.append(
                        f"<b>{sub.label}</b><br>"
                        f"Value: {data.currency_symbol}{sub.value:,.2f}"
                    )
            else:
                # Add regular item
                x_values.append(item.label)
                y_values.append(item.value)
                measures.append(item.measure)
                text_values.append(f"{data.currency_symbol}{item.value:+,.0f}")
                colors.append(item.color or (options.increasing_color if item.value >= 0 else options.decreasing_color))
                hover_texts.append(
                    f"<b>{item.label}</b><br>"
                    f"Value: {data.currency_symbol}{item.value:,.2f}"
                )

        # Add total
        x_values.append("Total")
        y_values.append(data.final_value)
        measures.append("total")
        text_values.append(f"{data.currency_symbol}{data.final_value:,.0f}")
        colors.append(options.total_color)
        hover_texts.append(f"<b>Total</b><br>Value: {data.currency_symbol}{data.final_value:,.2f}")

        trace = {
            "type": "waterfall",
            "name": "Detailed Breakdown",
            "orientation": options.orientation.value,
            "measure": measures,
            "textposition": options.label_position,
            "hovertemplate": "%{customdata}<extra></extra>",
            "customdata": hover_texts,
            "connector": {
                "line": {
                    "color": options.connector_color,
                    "width": options.connector_width,
                },
                "visible": options.connector_visible,
            },
            "marker": {"color": colors},
        }

        if options.orientation == WaterfallOrientation.VERTICAL:
            trace["x"] = x_values
            trace["y"] = y_values
        else:
            trace["x"] = y_values
            trace["y"] = x_values

        if options.show_labels:
            trace["text"] = text_values

        layout = self._build_layout(data, options)
        layout["title"]["text"] = "Detailed Cost Breakdown"

        return {
            "data": [trace],
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

    def _build_layout(
        self,
        data: CostBreakdown,
        options: WaterfallChartOptions,
    ) -> Dict[str, Any]:
        """Build Plotly layout configuration."""
        layout = self.theme.to_layout_dict()

        # Title
        layout["title"] = {
            "text": options.title,
            "font": {
                "size": self.theme.font.size_title,
                "color": self.theme.title_color,
            },
            "x": 0.5,
            "xanchor": "center",
        }

        # Subtitle
        if options.subtitle:
            layout["annotations"] = layout.get("annotations", [])
            layout["annotations"].append({
                "text": options.subtitle,
                "x": 0.5,
                "y": 1.05,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": self.theme.font.size_subtitle,
                    "color": self.theme.axis_color,
                },
            })

        # Axis configuration
        if options.orientation == WaterfallOrientation.VERTICAL:
            layout["xaxis"]["title"] = options.x_axis_title
            layout["xaxis"]["tickangle"] = -45 if len(data.items) > 5 else 0
            layout["yaxis"]["title"] = options.y_axis_title
            layout["yaxis"]["tickprefix"] = data.currency_symbol
            layout["yaxis"]["tickformat"] = ",.0f"
        else:
            layout["yaxis"]["title"] = options.x_axis_title
            layout["xaxis"]["title"] = options.y_axis_title
            layout["xaxis"]["tickprefix"] = data.currency_symbol
            layout["xaxis"]["tickformat"] = ",.0f"

        # Zero line
        if options.show_zero_line:
            if options.orientation == WaterfallOrientation.VERTICAL:
                layout["yaxis"]["zeroline"] = True
                layout["yaxis"]["zerolinecolor"] = options.zero_line_color
                layout["yaxis"]["zerolinewidth"] = options.zero_line_width
            else:
                layout["xaxis"]["zeroline"] = True
                layout["xaxis"]["zerolinecolor"] = options.zero_line_color
                layout["xaxis"]["zerolinewidth"] = options.zero_line_width

        # Legend
        if options.show_legend:
            layout["showlegend"] = True
            if options.legend_position == "right":
                layout["legend"] = {
                    "x": 1.02,
                    "y": 1,
                    "xanchor": "left",
                    "yanchor": "top",
                }
            elif options.legend_position == "bottom":
                layout["legend"] = {
                    "x": 0.5,
                    "y": -0.2,
                    "xanchor": "center",
                    "yanchor": "top",
                    "orientation": "h",
                }
        else:
            layout["showlegend"] = False

        # Size
        if options.width:
            layout["width"] = options.width
        if options.height:
            layout["height"] = options.height
        if options.auto_size:
            layout["autosize"] = True

        # Animation
        if not options.animate:
            layout["transition"] = {"duration": 0}

        return layout

    def _add_annotations(
        self,
        chart: Dict[str, Any],
        data: CostBreakdown,
        options: WaterfallChartOptions,
    ) -> Dict[str, Any]:
        """Add annotations to chart."""
        annotations = chart["layout"].get("annotations", [])

        # Add summary annotation
        net_change_str = f"{data.currency_symbol}{data.net_change:+,.2f}"
        if data.net_change_percent != 0:
            net_change_str += f" ({data.net_change_percent:+.1f}%)"

        summary_annotation = {
            "text": f"Net Change: {net_change_str}",
            "x": 1,
            "y": 1,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "right",
            "yanchor": "top",
            "showarrow": False,
            "font": {
                "size": 12,
                "color": self.theme.text_color,
            },
            "bgcolor": "rgba(255, 255, 255, 0.8)",
            "bordercolor": self.theme.axis_line_color,
            "borderwidth": 1,
            "borderpad": 4,
        }
        annotations.append(summary_annotation)

        # Add custom annotations
        for custom_ann in options.custom_annotations:
            annotations.append(custom_ann.to_plotly_dict())

        chart["layout"]["annotations"] = annotations
        return chart

    def _add_cumulative_line(
        self,
        chart: Dict[str, Any],
        data: CostBreakdown,
        options: WaterfallChartOptions,
    ) -> Dict[str, Any]:
        """Add cumulative line to chart."""
        cumulative = []
        running_total = data.baseline_value

        for item in data.items:
            if not item.is_total:
                running_total += item.value
                cumulative.append(running_total)

        x_values = [item.label for item in data.items if not item.is_total]

        cumulative_trace = {
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Cumulative Total",
            "line": {
                "color": "#333333",
                "width": 2,
                "dash": "dot",
            },
            "marker": {
                "size": 6,
                "color": "#333333",
            },
            "hovertemplate": "Cumulative: %{y:$,.2f}<extra></extra>",
        }

        if options.orientation == WaterfallOrientation.VERTICAL:
            cumulative_trace["x"] = x_values
            cumulative_trace["y"] = cumulative
        else:
            cumulative_trace["x"] = cumulative
            cumulative_trace["y"] = x_values

        chart["data"].append(cumulative_trace)
        return chart

    def _add_threshold_line(
        self,
        chart: Dict[str, Any],
        options: WaterfallChartOptions,
    ) -> Dict[str, Any]:
        """Add threshold line to chart."""
        shapes = chart["layout"].get("shapes", [])

        if options.orientation == WaterfallOrientation.VERTICAL:
            threshold_shape = {
                "type": "line",
                "x0": 0,
                "x1": 1,
                "xref": "paper",
                "y0": options.threshold_value,
                "y1": options.threshold_value,
                "line": {
                    "color": options.threshold_color,
                    "width": 2,
                    "dash": "dash",
                },
            }
        else:
            threshold_shape = {
                "type": "line",
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "x0": options.threshold_value,
                "x1": options.threshold_value,
                "line": {
                    "color": options.threshold_color,
                    "width": 2,
                    "dash": "dash",
                },
            }

        shapes.append(threshold_shape)
        chart["layout"]["shapes"] = shapes

        # Add threshold label
        annotations = chart["layout"].get("annotations", [])
        annotations.append({
            "text": f"Threshold: ${options.threshold_value:,.0f}",
            "x": 1,
            "y": options.threshold_value if options.orientation == WaterfallOrientation.VERTICAL else 1,
            "xref": "paper" if options.orientation == WaterfallOrientation.VERTICAL else "x",
            "yref": "y" if options.orientation == WaterfallOrientation.VERTICAL else "paper",
            "xanchor": "right",
            "yanchor": "bottom",
            "showarrow": False,
            "font": {
                "size": 10,
                "color": options.threshold_color,
            },
        })
        chart["layout"]["annotations"] = annotations

        return chart

    def generate_comparison(
        self,
        comparison_data: ComparisonData,
        options: Optional[WaterfallChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate comparison waterfall chart.

        Args:
            comparison_data: Comparison data with actual and baseline
            options: Chart configuration options

        Returns:
            Plotly-compatible chart specification
        """
        options = options or WaterfallChartOptions()
        options.waterfall_type = WaterfallType.COMPARISON

        # Build variance items
        variance_items = []

        for actual_item in comparison_data.actual.items:
            if actual_item.is_total:
                continue

            variance = comparison_data.get_variance(actual_item.category)
            variance_pct = comparison_data.get_variance_percent(actual_item.category)

            if variance is not None:
                variance_items.append(CostItem(
                    category=actual_item.category,
                    value=variance,
                    label=f"{actual_item.label} Variance",
                    description=f"Variance: {variance_pct:.1f}%" if variance_pct else None,
                ))

        # Create variance breakdown
        variance_breakdown = CostBreakdown(
            items=variance_items,
            currency=comparison_data.actual.currency,
            currency_symbol=comparison_data.actual.currency_symbol,
            baseline_value=comparison_data.budget.final_value if comparison_data.budget else 0,
            final_value=comparison_data.actual.final_value,
        )

        return self.generate(variance_breakdown, options)

    def generate_period_comparison(
        self,
        current_period: CostBreakdown,
        previous_period: CostBreakdown,
        options: Optional[WaterfallChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate period-over-period comparison waterfall.

        Args:
            current_period: Current period cost breakdown
            previous_period: Previous period cost breakdown
            options: Chart configuration options

        Returns:
            Plotly-compatible chart specification
        """
        options = options or WaterfallChartOptions()
        options.title = "Period-over-Period Cost Analysis"

        x_values = []
        y_values = []
        measures = []
        text_values = []
        colors = []
        hover_texts = []

        # Previous period total
        x_values.append("Previous Period")
        y_values.append(previous_period.final_value)
        measures.append("absolute")
        text_values.append(f"${previous_period.final_value:,.0f}")
        colors.append(options.total_color)
        hover_texts.append(f"<b>Previous Period</b><br>Total: ${previous_period.final_value:,.2f}")

        # Calculate changes by category
        for current_item in current_period.items:
            if current_item.is_total:
                continue

            prev_item = previous_period.get_item_by_category(current_item.category)
            prev_value = prev_item.value if prev_item else 0
            change = current_item.value - prev_value

            if abs(change) > 0.01:  # Only show significant changes
                x_values.append(f"{current_item.label} Change")
                y_values.append(change)
                measures.append("relative")
                text_values.append(f"${change:+,.0f}")
                colors.append(options.increasing_color if change > 0 else options.decreasing_color)

                pct_change = (change / prev_value * 100) if prev_value != 0 else 0
                hover_texts.append(
                    f"<b>{current_item.label}</b><br>"
                    f"Previous: ${prev_value:,.2f}<br>"
                    f"Current: ${current_item.value:,.2f}<br>"
                    f"Change: ${change:+,.2f} ({pct_change:+.1f}%)"
                )

        # Current period total
        x_values.append("Current Period")
        y_values.append(current_period.final_value)
        measures.append("total")
        text_values.append(f"${current_period.final_value:,.0f}")
        colors.append(options.total_color)
        hover_texts.append(f"<b>Current Period</b><br>Total: ${current_period.final_value:,.2f}")

        trace = {
            "type": "waterfall",
            "name": "Period Comparison",
            "orientation": options.orientation.value,
            "measure": measures,
            "x": x_values if options.orientation == WaterfallOrientation.VERTICAL else y_values,
            "y": y_values if options.orientation == WaterfallOrientation.VERTICAL else x_values,
            "text": text_values,
            "textposition": options.label_position,
            "hovertemplate": "%{customdata}<extra></extra>",
            "customdata": hover_texts,
            "connector": {
                "line": {
                    "color": options.connector_color,
                    "width": options.connector_width,
                },
                "visible": options.connector_visible,
            },
            "marker": {"color": colors},
        }

        layout = self._build_layout(current_period, options)

        # Add period change annotation
        total_change = current_period.final_value - previous_period.final_value
        pct_change = (total_change / previous_period.final_value * 100) if previous_period.final_value != 0 else 0

        annotations = layout.get("annotations", [])
        annotations.append({
            "text": f"Total Change: ${total_change:+,.2f} ({pct_change:+.1f}%)",
            "x": 0.5,
            "y": -0.15,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {
                "size": 14,
                "color": StatusColors.SUCCESS if total_change < 0 else StatusColors.ERROR,
                "weight": "bold",
            },
        })
        layout["annotations"] = annotations

        return {
            "data": [trace],
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

    def generate_drill_down(
        self,
        data: CostBreakdown,
        category: CostCategory,
        options: Optional[WaterfallChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate drill-down waterfall for a specific category.

        Args:
            data: Cost breakdown data
            category: Category to drill down into
            options: Chart configuration options

        Returns:
            Plotly-compatible chart specification
        """
        options = options or WaterfallChartOptions()

        # Find the category item
        category_item = data.get_item_by_category(category)
        if not category_item:
            raise ValueError(f"Category {category.value} not found in data")

        if not category_item.subcategories:
            raise ValueError(f"Category {category.value} has no subcategories for drill-down")

        # Create drill-down breakdown
        drill_down_items = [
            CostItem(
                category=CostCategory.BASELINE_COST,
                value=category_item.value,
                label=f"{category_item.label} Total",
                is_subtotal=True,
            )
        ] + category_item.subcategories

        drill_down_breakdown = CostBreakdown(
            items=drill_down_items,
            currency=data.currency,
            currency_symbol=data.currency_symbol,
            baseline_value=0,
            final_value=category_item.value,
        )

        options.title = f"{category_item.label} Breakdown"
        return self.generate(drill_down_breakdown, options)

    def to_json(self, chart: Dict[str, Any]) -> str:
        """
        Export chart to JSON string.

        Args:
            chart: Chart specification

        Returns:
            JSON string
        """
        return json.dumps(chart, indent=2, default=str)

    def clear_cache(self) -> None:
        """Clear the chart cache."""
        self._cache.clear()


# =============================================================================
# SPECIALIZED WATERFALL GENERATORS
# =============================================================================

class FuelSwitchingWaterfall(WaterfallChartEngine):
    """Specialized waterfall for fuel switching analysis."""

    def generate_fuel_switching_analysis(
        self,
        current_fuel_costs: Dict[str, float],
        alternative_fuel_costs: Dict[str, float],
        switching_costs: Dict[str, float],
        options: Optional[WaterfallChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate fuel switching analysis waterfall.

        Args:
            current_fuel_costs: Current fuel costs by type
            alternative_fuel_costs: Alternative fuel costs by type
            switching_costs: One-time switching costs
            options: Chart options

        Returns:
            Plotly chart specification
        """
        options = options or WaterfallChartOptions()
        options.title = "Fuel Switching Cost Analysis"

        items = []

        # Current fuel costs
        total_current = sum(current_fuel_costs.values())
        items.append(CostItem(
            category=CostCategory.BASELINE_COST,
            value=total_current,
            label="Current Fuel Costs",
            color=FuelTypeColors.get_color("coal"),
        ))

        # Fuel cost changes
        for fuel, current_cost in current_fuel_costs.items():
            alt_cost = alternative_fuel_costs.get(fuel, 0)
            if alt_cost > 0:
                change = alt_cost - current_cost
                items.append(CostItem(
                    category=CostCategory.FUEL_SWITCHING_SAVINGS,
                    value=change,
                    label=f"{fuel.title()} Switch",
                    color=FuelTypeColors.get_color(fuel),
                    description=f"From ${current_cost:,.0f} to ${alt_cost:,.0f}",
                ))

        # Switching costs
        for cost_type, cost in switching_costs.items():
            items.append(CostItem(
                category=CostCategory.FUEL_COST,
                value=cost,
                label=f"Switching: {cost_type.title()}",
                color=CostCategoryColors.HANDLING_COST,
            ))

        # Calculate total
        total_new = sum(alternative_fuel_costs.values()) + sum(switching_costs.values())
        items.append(CostItem(
            category=CostCategory.TOTAL,
            value=total_new,
            label="New Total Cost",
            is_total=True,
        ))

        breakdown = CostBreakdown(
            items=items,
            baseline_value=total_current,
            final_value=total_new,
        )

        return self.generate(breakdown, options)


class BlendOptimizationWaterfall(WaterfallChartEngine):
    """Specialized waterfall for blend optimization analysis."""

    def generate_blend_optimization_analysis(
        self,
        unoptimized_costs: Dict[str, float],
        optimized_costs: Dict[str, float],
        blend_ratios: Dict[str, float],
        options: Optional[WaterfallChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate blend optimization analysis waterfall.

        Args:
            unoptimized_costs: Costs before optimization
            optimized_costs: Costs after optimization
            blend_ratios: Optimized blend ratios
            options: Chart options

        Returns:
            Plotly chart specification
        """
        options = options or WaterfallChartOptions()
        options.title = "Blend Optimization Savings"

        items = []

        # Unoptimized total
        total_unoptimized = sum(unoptimized_costs.values())
        items.append(CostItem(
            category=CostCategory.BASELINE_COST,
            value=total_unoptimized,
            label="Unoptimized Blend",
        ))

        # Per-fuel savings
        for fuel in unoptimized_costs.keys():
            unopt_cost = unoptimized_costs.get(fuel, 0)
            opt_cost = optimized_costs.get(fuel, 0)
            savings = unopt_cost - opt_cost

            if abs(savings) > 0.01:
                ratio = blend_ratios.get(fuel, 0) * 100
                items.append(CostItem(
                    category=CostCategory.BLEND_OPTIMIZATION_SAVINGS,
                    value=-savings,  # Negative for savings
                    label=f"{fuel.title()} ({ratio:.0f}%)",
                    color=FuelTypeColors.get_color(fuel),
                    description=f"Ratio: {ratio:.1f}%",
                ))

        # Optimized total
        total_optimized = sum(optimized_costs.values())
        items.append(CostItem(
            category=CostCategory.TOTAL,
            value=total_optimized,
            label="Optimized Blend",
            is_total=True,
        ))

        breakdown = CostBreakdown(
            items=items,
            baseline_value=total_unoptimized,
            final_value=total_optimized,
        )

        return self.generate(breakdown, options)


class ProcurementOptimizationWaterfall(WaterfallChartEngine):
    """Specialized waterfall for procurement optimization analysis."""

    def generate_procurement_analysis(
        self,
        base_procurement_cost: float,
        volume_discounts: Dict[str, float],
        contract_savings: Dict[str, float],
        timing_savings: float,
        logistics_optimization: float,
        options: Optional[WaterfallChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate procurement optimization analysis waterfall.

        Args:
            base_procurement_cost: Base procurement cost
            volume_discounts: Volume discount savings by supplier
            contract_savings: Contract negotiation savings
            timing_savings: Market timing savings
            logistics_optimization: Logistics optimization savings
            options: Chart options

        Returns:
            Plotly chart specification
        """
        options = options or WaterfallChartOptions()
        options.title = "Procurement Optimization Analysis"

        items = []

        # Base cost
        items.append(CostItem(
            category=CostCategory.BASELINE_COST,
            value=base_procurement_cost,
            label="Base Procurement Cost",
        ))

        # Volume discounts
        for supplier, discount in volume_discounts.items():
            items.append(CostItem(
                category=CostCategory.PROCUREMENT_OPTIMIZATION,
                value=-discount,
                label=f"Volume: {supplier}",
                color=CostCategoryColors.PROCUREMENT_SAVINGS,
            ))

        # Contract savings
        for contract, saving in contract_savings.items():
            items.append(CostItem(
                category=CostCategory.PROCUREMENT_OPTIMIZATION,
                value=-saving,
                label=f"Contract: {contract}",
                color=CostCategoryColors.BLEND_SAVINGS,
            ))

        # Timing savings
        if timing_savings > 0:
            items.append(CostItem(
                category=CostCategory.MARKET_TIMING_SAVINGS,
                value=-timing_savings,
                label="Market Timing",
                color=CostCategoryColors.MARKET_TIMING,
            ))

        # Logistics optimization
        if logistics_optimization > 0:
            items.append(CostItem(
                category=CostCategory.OPTIMIZATION_BENEFIT,
                value=-logistics_optimization,
                label="Logistics Optimization",
                color=CostCategoryColors.OPTIMIZATION_BENEFIT,
            ))

        # Calculate total
        total_savings = (
            sum(volume_discounts.values()) +
            sum(contract_savings.values()) +
            timing_savings +
            logistics_optimization
        )
        final_cost = base_procurement_cost - total_savings

        items.append(CostItem(
            category=CostCategory.TOTAL,
            value=final_cost,
            label="Optimized Cost",
            is_total=True,
        ))

        breakdown = CostBreakdown(
            items=items,
            baseline_value=base_procurement_cost,
            final_value=final_cost,
            total_savings=total_savings,
        )

        return self.generate(breakdown, options)


class CarbonCostWaterfall(WaterfallChartEngine):
    """Specialized waterfall for carbon cost analysis."""

    def generate_carbon_cost_analysis(
        self,
        emissions_by_fuel: Dict[str, float],
        carbon_price_per_ton: float,
        carbon_credits: float = 0,
        carbon_offsets: float = 0,
        regulatory_allowances: float = 0,
        options: Optional[WaterfallChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate carbon cost analysis waterfall.

        Args:
            emissions_by_fuel: CO2 emissions by fuel type (tons)
            carbon_price_per_ton: Carbon price per ton
            carbon_credits: Carbon credits available
            carbon_offsets: Carbon offsets purchased
            regulatory_allowances: Free regulatory allowances
            options: Chart options

        Returns:
            Plotly chart specification
        """
        options = options or WaterfallChartOptions()
        options.title = "Carbon Cost Breakdown"

        items = []
        total_emissions = sum(emissions_by_fuel.values())

        # Emissions by fuel
        for fuel, emissions in emissions_by_fuel.items():
            cost = emissions * carbon_price_per_ton
            items.append(CostItem(
                category=CostCategory.CARBON_COST,
                value=cost,
                label=f"{fuel.title()} Emissions",
                color=FuelTypeColors.get_color(fuel),
                description=f"{emissions:,.0f} tons CO2",
                metadata={"emissions_tons": emissions},
            ))

        # Credits and offsets (reduce cost)
        if carbon_credits > 0:
            items.append(CostItem(
                category=CostCategory.CARBON_SAVINGS,
                value=-carbon_credits * carbon_price_per_ton,
                label="Carbon Credits",
                color=CostCategoryColors.CARBON_SAVINGS,
            ))

        if carbon_offsets > 0:
            items.append(CostItem(
                category=CostCategory.CARBON_SAVINGS,
                value=-carbon_offsets * carbon_price_per_ton,
                label="Carbon Offsets",
                color=CostCategoryColors.SUBSIDY,
            ))

        if regulatory_allowances > 0:
            items.append(CostItem(
                category=CostCategory.SUBSIDY,
                value=-regulatory_allowances * carbon_price_per_ton,
                label="Regulatory Allowances",
                color=CostCategoryColors.SUBSIDY,
            ))

        # Calculate net carbon cost
        gross_carbon_cost = total_emissions * carbon_price_per_ton
        credits_value = (carbon_credits + carbon_offsets + regulatory_allowances) * carbon_price_per_ton
        net_carbon_cost = gross_carbon_cost - credits_value

        items.append(CostItem(
            category=CostCategory.TOTAL,
            value=net_carbon_cost,
            label="Net Carbon Cost",
            is_total=True,
        ))

        breakdown = CostBreakdown(
            items=items,
            baseline_value=gross_carbon_cost,
            final_value=net_carbon_cost,
            metadata={
                "total_emissions_tons": total_emissions,
                "carbon_price": carbon_price_per_ton,
            },
        )

        return self.generate(breakdown, options)


# =============================================================================
# DATA TRANSFORMATION UTILITIES
# =============================================================================

class WaterfallDataTransformer:
    """Utility class for transforming data into waterfall format."""

    @staticmethod
    def from_dict(
        data: Dict[str, float],
        baseline_key: Optional[str] = None,
        total_key: Optional[str] = None,
        category_mapping: Optional[Dict[str, CostCategory]] = None,
    ) -> CostBreakdown:
        """
        Transform dictionary data into CostBreakdown.

        Args:
            data: Dictionary with category labels and values
            baseline_key: Key to use as baseline (optional)
            total_key: Key to use as total (optional)
            category_mapping: Mapping from keys to CostCategory enum

        Returns:
            CostBreakdown instance
        """
        items = []
        baseline_value = 0
        final_value = 0

        category_mapping = category_mapping or {}

        for key, value in data.items():
            if key == baseline_key:
                baseline_value = value
                continue
            if key == total_key:
                final_value = value
                continue

            category = category_mapping.get(key, CostCategory.FUEL_COST)
            items.append(CostItem(
                category=category,
                value=value,
                label=key.replace("_", " ").title(),
            ))

        if total_key and final_value == 0:
            final_value = baseline_value + sum(item.value for item in items)

        return CostBreakdown(
            items=items,
            baseline_value=baseline_value,
            final_value=final_value,
        )

    @staticmethod
    def from_dataframe(
        df: Any,
        category_column: str,
        value_column: str,
        category_mapping: Optional[Dict[str, CostCategory]] = None,
    ) -> CostBreakdown:
        """
        Transform pandas DataFrame into CostBreakdown.

        Args:
            df: Pandas DataFrame
            category_column: Column name for categories
            value_column: Column name for values
            category_mapping: Mapping from category names to CostCategory enum

        Returns:
            CostBreakdown instance
        """
        items = []
        category_mapping = category_mapping or {}

        for _, row in df.iterrows():
            category_name = row[category_column]
            category = category_mapping.get(
                category_name,
                CostCategory.FUEL_COST
            )
            items.append(CostItem(
                category=category,
                value=float(row[value_column]),
                label=str(category_name).replace("_", " ").title(),
            ))

        return CostBreakdown(items=items)

    @staticmethod
    def aggregate_by_category(
        breakdown: CostBreakdown,
        grouping: Dict[CostCategory, List[CostCategory]],
    ) -> CostBreakdown:
        """
        Aggregate cost items by category groupings.

        Args:
            breakdown: Original cost breakdown
            grouping: Mapping of aggregate categories to component categories

        Returns:
            Aggregated CostBreakdown
        """
        aggregated_items = []

        for aggregate_cat, components in grouping.items():
            component_items = [
                item for item in breakdown.items
                if item.category in components
            ]
            if component_items:
                total_value = sum(item.value for item in component_items)
                aggregated_items.append(CostItem(
                    category=aggregate_cat,
                    value=total_value,
                    label=aggregate_cat.display_name,
                    subcategories=component_items,
                ))

        return CostBreakdown(
            items=aggregated_items,
            currency=breakdown.currency,
            currency_symbol=breakdown.currency_symbol,
            baseline_value=breakdown.baseline_value,
        )


# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATIONS
# =============================================================================

def create_sample_cost_breakdown() -> CostBreakdown:
    """Create a sample cost breakdown for demonstration."""
    items = [
        CostItem(
            category=CostCategory.BASELINE_COST,
            value=1000000,
            label="Baseline Fuel Cost",
        ),
        CostItem(
            category=CostCategory.FUEL_SWITCHING_SAVINGS,
            value=-150000,
            label="Fuel Switching Savings",
            description="Natural gas to biomass blend",
        ),
        CostItem(
            category=CostCategory.BLEND_OPTIMIZATION_SAVINGS,
            value=-75000,
            label="Blend Optimization",
            description="Optimized coal/gas ratio",
        ),
        CostItem(
            category=CostCategory.PROCUREMENT_OPTIMIZATION,
            value=-50000,
            label="Procurement Savings",
            description="Volume discounts and contracts",
        ),
        CostItem(
            category=CostCategory.MARKET_TIMING_SAVINGS,
            value=-25000,
            label="Market Timing",
            description="Strategic purchase timing",
        ),
        CostItem(
            category=CostCategory.CARBON_COST,
            value=100000,
            label="Carbon Costs",
            description="EU ETS compliance",
        ),
        CostItem(
            category=CostCategory.CARBON_SAVINGS,
            value=-30000,
            label="Carbon Credits",
            description="Renewable energy credits",
        ),
        CostItem(
            category=CostCategory.TOTAL,
            value=770000,
            label="Net Fuel Cost",
            is_total=True,
        ),
    ]

    return CostBreakdown(
        items=items,
        currency="USD",
        currency_symbol="$",
        period_start="2024-01-01",
        period_end="2024-12-31",
        baseline_value=1000000,
        final_value=770000,
    )


def example_standard_waterfall():
    """Example: Generate standard waterfall chart."""
    print("Generating standard waterfall chart...")

    # Create sample data
    breakdown = create_sample_cost_breakdown()

    # Create engine and generate chart
    engine = WaterfallChartEngine()
    options = WaterfallChartOptions(
        title="Annual Fuel Cost Analysis",
        subtitle="FY 2024",
        show_cumulative_line=False,
        enable_hover=True,
    )

    chart = engine.generate(breakdown, options)
    print(f"Chart generated with {len(chart['data'])} traces")
    return chart


def example_variance_waterfall():
    """Example: Generate variance analysis waterfall."""
    print("Generating variance analysis waterfall...")

    breakdown = create_sample_cost_breakdown()

    engine = WaterfallChartEngine()
    options = WaterfallChartOptions(
        waterfall_type=WaterfallType.VARIANCE,
        title="Budget vs Actual Variance",
    )

    chart = engine.generate(breakdown, options)
    print(f"Variance chart generated")
    return chart


def example_fuel_switching_analysis():
    """Example: Generate fuel switching analysis."""
    print("Generating fuel switching analysis...")

    current_costs = {
        "coal": 500000,
        "natural_gas": 300000,
        "oil": 200000,
    }

    alternative_costs = {
        "coal": 400000,  # Reduced usage
        "natural_gas": 350000,  # Increased usage
        "biomass": 150000,  # New addition
    }

    switching_costs = {
        "equipment_modification": 50000,
        "training": 10000,
        "permits": 5000,
    }

    engine = FuelSwitchingWaterfall()
    chart = engine.generate_fuel_switching_analysis(
        current_costs,
        alternative_costs,
        switching_costs,
    )

    print(f"Fuel switching chart generated")
    return chart


def example_blend_optimization():
    """Example: Generate blend optimization analysis."""
    print("Generating blend optimization analysis...")

    unoptimized = {
        "coal": 400000,
        "natural_gas": 300000,
        "biomass": 100000,
    }

    optimized = {
        "coal": 350000,
        "natural_gas": 280000,
        "biomass": 120000,
    }

    ratios = {
        "coal": 0.45,
        "natural_gas": 0.35,
        "biomass": 0.20,
    }

    engine = BlendOptimizationWaterfall()
    chart = engine.generate_blend_optimization_analysis(
        unoptimized,
        optimized,
        ratios,
    )

    print(f"Blend optimization chart generated")
    return chart


def example_carbon_cost_analysis():
    """Example: Generate carbon cost analysis."""
    print("Generating carbon cost analysis...")

    emissions = {
        "coal": 50000,  # tons CO2
        "natural_gas": 20000,
        "oil": 15000,
    }

    engine = CarbonCostWaterfall()
    chart = engine.generate_carbon_cost_analysis(
        emissions_by_fuel=emissions,
        carbon_price_per_ton=50,
        carbon_credits=5000,
        carbon_offsets=2000,
        regulatory_allowances=10000,
    )

    print(f"Carbon cost chart generated")
    return chart


def example_period_comparison():
    """Example: Generate period-over-period comparison."""
    print("Generating period comparison...")

    # Current period
    current_items = [
        CostItem(CostCategory.FUEL_COST, 800000, "Fuel Cost"),
        CostItem(CostCategory.CARBON_COST, 100000, "Carbon Cost"),
        CostItem(CostCategory.DELIVERY_COST, 50000, "Delivery"),
        CostItem(CostCategory.TOTAL, 950000, "Total", is_total=True),
    ]
    current = CostBreakdown(items=current_items, baseline_value=0, final_value=950000)

    # Previous period
    previous_items = [
        CostItem(CostCategory.FUEL_COST, 850000, "Fuel Cost"),
        CostItem(CostCategory.CARBON_COST, 80000, "Carbon Cost"),
        CostItem(CostCategory.DELIVERY_COST, 55000, "Delivery"),
        CostItem(CostCategory.TOTAL, 985000, "Total", is_total=True),
    ]
    previous = CostBreakdown(items=previous_items, baseline_value=0, final_value=985000)

    engine = WaterfallChartEngine()
    chart = engine.generate_period_comparison(current, previous)

    print(f"Period comparison chart generated")
    return chart


def run_all_examples():
    """Run all waterfall chart examples."""
    print("=" * 60)
    print("GL-011 FUELCRAFT - Waterfall Visualization Examples")
    print("=" * 60)

    examples = [
        ("Standard Waterfall", example_standard_waterfall),
        ("Variance Analysis", example_variance_waterfall),
        ("Fuel Switching", example_fuel_switching_analysis),
        ("Blend Optimization", example_blend_optimization),
        ("Carbon Cost", example_carbon_cost_analysis),
        ("Period Comparison", example_period_comparison),
    ]

    results = {}
    for name, func in examples:
        print(f"\n--- {name} ---")
        try:
            results[name] = func()
            print(f"SUCCESS: {name}")
        except Exception as e:
            print(f"ERROR: {name} - {e}")
            results[name] = None

    print("\n" + "=" * 60)
    print(f"Completed {len([r for r in results.values() if r])} of {len(examples)} examples")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_all_examples()
