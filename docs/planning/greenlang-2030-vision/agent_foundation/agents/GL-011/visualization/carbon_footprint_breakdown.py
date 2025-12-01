# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Carbon Footprint Breakdown Visualization Module.

Comprehensive visualization for carbon emissions analysis including pie charts,
bar charts, trend analysis, and regulatory compliance overlays.

Author: GreenLang Team
Version: 1.0.0
Standards: WCAG 2.1 Level AA, GHG Protocol, ISO 14064

Features:
- Pie charts for emissions breakdown by fuel type/source
- Bar charts for period comparisons
- Trend analysis with baseline and target overlays
- Scope 1/2/3 emissions breakdown
- Regulatory limit visualization
- Carbon intensity metrics
- Export to PNG/PDF/SVG/JSON
- Responsive design with accessibility compliance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum, auto
from datetime import datetime, timedelta, date
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
    FuelTypeColors,
    EmissionColors,
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
    get_fuel_color,
    get_emission_color,
    get_status_color,
    hex_to_rgba,
    adjust_color_brightness,
    blend_colors,
    get_plotly_config,
    create_annotation,
    create_shape,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class EmissionScope(Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"  # Direct emissions
    SCOPE_2 = "scope_2"  # Indirect from purchased energy
    SCOPE_3 = "scope_3"  # Other indirect emissions
    ALL = "all"


class EmissionType(Enum):
    """Types of greenhouse gas emissions."""
    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    HFCS = "hfcs"
    PFCS = "pfcs"
    SF6 = "sf6"
    NF3 = "nf3"
    CO2E = "co2e"  # CO2 equivalent


class ChartMode(Enum):
    """Chart display modes."""
    PIE = "pie"
    DONUT = "donut"
    BAR = "bar"
    STACKED_BAR = "stacked_bar"
    GROUPED_BAR = "grouped_bar"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    WATERFALL = "waterfall"


class TimeGranularity(Enum):
    """Time granularity for trend analysis."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class ComplianceStatus(Enum):
    """Regulatory compliance status."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    EXEMPT = "exempt"
    PENDING = "pending"
    UNKNOWN = "unknown"

    @property
    def color(self) -> str:
        """Get color for status."""
        return get_status_color(self.value)


class TargetType(Enum):
    """Types of emission targets."""
    ABSOLUTE = "absolute"  # Absolute reduction target
    INTENSITY = "intensity"  # Intensity-based target
    SCIENCE_BASED = "science_based"  # Science-based target
    REGULATORY = "regulatory"  # Regulatory limit
    INTERNAL = "internal"  # Internal company target


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EmissionSource:
    """Single emission source."""
    source_id: str
    source_name: str
    fuel_type: Optional[str] = None
    scope: EmissionScope = EmissionScope.SCOPE_1
    emissions_co2e: float = 0.0  # kg CO2e
    emissions_co2: float = 0.0  # kg CO2
    emissions_ch4: float = 0.0  # kg CH4
    emissions_n2o: float = 0.0  # kg N2O
    energy_consumed: float = 0.0  # MJ
    emission_factor: float = 0.0  # kg CO2e/unit
    activity_data: float = 0.0  # Activity quantity
    activity_unit: str = ""
    uncertainty_percent: float = 5.0
    data_quality_score: float = 100.0
    verification_status: str = "verified"
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def color(self) -> str:
        """Get color for this source."""
        if self.fuel_type:
            return get_fuel_color(self.fuel_type)
        return "#888888"

    @property
    def carbon_intensity(self) -> float:
        """Calculate carbon intensity (kg CO2e/GJ)."""
        if self.energy_consumed > 0:
            return (self.emissions_co2e / self.energy_consumed) * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "fuel_type": self.fuel_type,
            "scope": self.scope.value,
            "emissions_co2e": self.emissions_co2e,
            "emissions_co2": self.emissions_co2,
            "emissions_ch4": self.emissions_ch4,
            "emissions_n2o": self.emissions_n2o,
            "energy_consumed": self.energy_consumed,
            "emission_factor": self.emission_factor,
            "carbon_intensity": self.carbon_intensity,
            "uncertainty_percent": self.uncertainty_percent,
            "data_quality_score": self.data_quality_score,
            "verification_status": self.verification_status,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class EmissionTarget:
    """Emission reduction target."""
    target_id: str
    target_name: str
    target_type: TargetType
    target_value: float  # kg CO2e or kg CO2e/unit for intensity
    baseline_value: float
    baseline_year: int
    target_year: int
    current_value: float
    unit: str = "kg CO2e"
    scope: EmissionScope = EmissionScope.ALL
    description: Optional[str] = None
    source: Optional[str] = None  # e.g., "Paris Agreement", "SBTi"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress_percent(self) -> float:
        """Calculate progress towards target."""
        if self.baseline_value == self.target_value:
            return 100.0 if self.current_value <= self.target_value else 0.0

        total_reduction_needed = self.baseline_value - self.target_value
        reduction_achieved = self.baseline_value - self.current_value

        if total_reduction_needed != 0:
            return (reduction_achieved / total_reduction_needed) * 100
        return 0.0

    @property
    def is_on_track(self) -> bool:
        """Check if on track to meet target."""
        return self.current_value <= self._get_expected_value()

    def _get_expected_value(self) -> float:
        """Get expected value for current year on linear trajectory."""
        current_year = datetime.now().year
        total_years = self.target_year - self.baseline_year
        years_elapsed = current_year - self.baseline_year

        if total_years <= 0:
            return self.target_value

        reduction_rate = (self.baseline_value - self.target_value) / total_years
        return self.baseline_value - (reduction_rate * years_elapsed)

    @property
    def status(self) -> ComplianceStatus:
        """Get compliance status."""
        if self.current_value <= self.target_value:
            return ComplianceStatus.COMPLIANT
        elif self.is_on_track:
            return ComplianceStatus.WARNING
        else:
            return ComplianceStatus.VIOLATION

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_id": self.target_id,
            "target_name": self.target_name,
            "target_type": self.target_type.value,
            "target_value": self.target_value,
            "baseline_value": self.baseline_value,
            "baseline_year": self.baseline_year,
            "target_year": self.target_year,
            "current_value": self.current_value,
            "unit": self.unit,
            "scope": self.scope.value,
            "progress_percent": self.progress_percent,
            "is_on_track": self.is_on_track,
            "status": self.status.value,
            "description": self.description,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class RegulatoryLimit:
    """Regulatory emission limit."""
    limit_id: str
    limit_name: str
    jurisdiction: str
    limit_value: float
    unit: str
    emission_type: EmissionType
    averaging_period: str  # e.g., "annual", "monthly", "hourly"
    applicable_scope: EmissionScope
    effective_date: str
    expiry_date: Optional[str] = None
    penalty_per_unit: float = 0.0
    description: Optional[str] = None
    regulation_reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def check_compliance(self, current_value: float) -> ComplianceStatus:
        """Check compliance against limit."""
        margin_percent = ((self.limit_value - current_value) / self.limit_value) * 100

        if current_value <= self.limit_value * 0.9:
            return ComplianceStatus.COMPLIANT
        elif current_value <= self.limit_value:
            return ComplianceStatus.WARNING
        else:
            return ComplianceStatus.VIOLATION

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "limit_id": self.limit_id,
            "limit_name": self.limit_name,
            "jurisdiction": self.jurisdiction,
            "limit_value": self.limit_value,
            "unit": self.unit,
            "emission_type": self.emission_type.value,
            "averaging_period": self.averaging_period,
            "applicable_scope": self.applicable_scope.value,
            "effective_date": self.effective_date,
            "expiry_date": self.expiry_date,
            "penalty_per_unit": self.penalty_per_unit,
            "description": self.description,
            "regulation_reference": self.regulation_reference,
            "metadata": self.metadata,
        }


@dataclass
class EmissionDataPoint:
    """Single data point in time series."""
    timestamp: str
    value: float
    unit: str = "kg CO2e"
    source_id: Optional[str] = None
    scope: Optional[EmissionScope] = None
    verified: bool = False
    estimated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmissionTrendData:
    """Time series data for emission trends."""
    data_points: List[EmissionDataPoint]
    granularity: TimeGranularity
    start_date: str
    end_date: str
    total_emissions: float = 0.0
    average_emissions: float = 0.0
    min_emissions: float = 0.0
    max_emissions: float = 0.0
    trend_direction: str = "stable"  # increasing, decreasing, stable
    trend_percent: float = 0.0

    def __post_init__(self):
        """Calculate statistics."""
        if self.data_points:
            values = [dp.value for dp in self.data_points]
            self.total_emissions = sum(values)
            self.average_emissions = self.total_emissions / len(values)
            self.min_emissions = min(values)
            self.max_emissions = max(values)
            self._calculate_trend()

    def _calculate_trend(self):
        """Calculate trend direction and percentage."""
        if len(self.data_points) < 2:
            return

        # Compare first quarter to last quarter
        quarter_size = len(self.data_points) // 4 or 1
        first_quarter = [dp.value for dp in self.data_points[:quarter_size]]
        last_quarter = [dp.value for dp in self.data_points[-quarter_size:]]

        first_avg = sum(first_quarter) / len(first_quarter)
        last_avg = sum(last_quarter) / len(last_quarter)

        if first_avg != 0:
            self.trend_percent = ((last_avg - first_avg) / first_avg) * 100

        if self.trend_percent > 5:
            self.trend_direction = "increasing"
        elif self.trend_percent < -5:
            self.trend_direction = "decreasing"
        else:
            self.trend_direction = "stable"


@dataclass
class CarbonFootprintData:
    """Complete carbon footprint data structure."""
    sources: List[EmissionSource]
    targets: List[EmissionTarget] = field(default_factory=list)
    regulatory_limits: List[RegulatoryLimit] = field(default_factory=list)
    trend_data: Optional[EmissionTrendData] = None
    reporting_period: str = ""
    facility_id: Optional[str] = None
    facility_name: Optional[str] = None
    organization: Optional[str] = None
    total_co2e: float = 0.0
    scope_1_total: float = 0.0
    scope_2_total: float = 0.0
    scope_3_total: float = 0.0
    carbon_intensity: float = 0.0
    intensity_unit: str = "kg CO2e/MJ"
    verification_status: str = "unverified"
    provenance_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate totals and provenance."""
        self._calculate_totals()
        self._calculate_provenance()

    def _calculate_totals(self):
        """Calculate total emissions by scope."""
        self.scope_1_total = sum(
            s.emissions_co2e for s in self.sources
            if s.scope == EmissionScope.SCOPE_1
        )
        self.scope_2_total = sum(
            s.emissions_co2e for s in self.sources
            if s.scope == EmissionScope.SCOPE_2
        )
        self.scope_3_total = sum(
            s.emissions_co2e for s in self.sources
            if s.scope == EmissionScope.SCOPE_3
        )
        self.total_co2e = self.scope_1_total + self.scope_2_total + self.scope_3_total

        # Calculate intensity
        total_energy = sum(s.energy_consumed for s in self.sources)
        if total_energy > 0:
            self.carbon_intensity = self.total_co2e / total_energy

    def _calculate_provenance(self):
        """Calculate provenance hash."""
        data = {
            "sources": [(s.source_id, s.emissions_co2e) for s in self.sources],
            "period": self.reporting_period,
            "total": self.total_co2e,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def get_by_scope(self, scope: EmissionScope) -> List[EmissionSource]:
        """Get sources by scope."""
        if scope == EmissionScope.ALL:
            return self.sources
        return [s for s in self.sources if s.scope == scope]

    def get_by_fuel_type(self, fuel_type: str) -> List[EmissionSource]:
        """Get sources by fuel type."""
        return [s for s in self.sources if s.fuel_type == fuel_type]

    def get_compliance_status(self) -> ComplianceStatus:
        """Get overall compliance status."""
        for limit in self.regulatory_limits:
            status = limit.check_compliance(self.total_co2e)
            if status == ComplianceStatus.VIOLATION:
                return ComplianceStatus.VIOLATION
            elif status == ComplianceStatus.WARNING:
                return ComplianceStatus.WARNING
        return ComplianceStatus.COMPLIANT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sources": [s.to_dict() for s in self.sources],
            "targets": [t.to_dict() for t in self.targets],
            "regulatory_limits": [l.to_dict() for l in self.regulatory_limits],
            "reporting_period": self.reporting_period,
            "facility_id": self.facility_id,
            "facility_name": self.facility_name,
            "organization": self.organization,
            "total_co2e": self.total_co2e,
            "scope_1_total": self.scope_1_total,
            "scope_2_total": self.scope_2_total,
            "scope_3_total": self.scope_3_total,
            "carbon_intensity": self.carbon_intensity,
            "intensity_unit": self.intensity_unit,
            "verification_status": self.verification_status,
            "compliance_status": self.get_compliance_status().value,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# =============================================================================
# CHART OPTIONS
# =============================================================================

@dataclass
class CarbonChartOptions:
    """Configuration options for carbon footprint charts."""
    # Display options
    title: str = "Carbon Footprint Breakdown"
    subtitle: Optional[str] = None
    chart_mode: ChartMode = ChartMode.DONUT
    show_by_scope: bool = True
    show_by_fuel: bool = True
    show_by_source: bool = False

    # Value display
    value_unit: str = "kg CO2e"
    show_percentages: bool = True
    show_absolute_values: bool = True
    decimal_places: int = 2
    use_scientific_notation: bool = False
    scale_factor: float = 1.0  # e.g., 0.001 to show in tonnes

    # Color options
    color_scheme: str = "fuel_type"  # fuel_type, scope, carbon_intensity
    color_blind_safe: bool = False
    custom_colors: Optional[Dict[str, str]] = None

    # Pie/Donut specific
    hole_size: float = 0.4  # For donut chart
    pull_distance: float = 0.02
    start_angle: int = 90
    direction: str = "clockwise"
    sort_values: bool = True

    # Bar chart specific
    bar_orientation: str = "v"  # v or h
    bar_gap: float = 0.2
    bar_group_gap: float = 0.3
    show_bar_labels: bool = True

    # Target/Limit overlays
    show_targets: bool = True
    show_regulatory_limits: bool = True
    target_line_color: str = "#2ECC71"
    limit_line_color: str = "#E74C3C"
    show_progress_indicator: bool = True

    # Trend options
    show_trend_line: bool = False
    trend_window: int = 7  # Days for moving average

    # Legend options
    show_legend: bool = True
    legend_position: str = "right"

    # Interaction options
    enable_hover: bool = True
    enable_click: bool = False
    enable_drill_down: bool = False

    # Size options
    width: Optional[int] = None
    height: Optional[int] = None
    auto_size: bool = True

    # Animation options
    animate: bool = True
    animation_duration: int = 500

    # Additional options
    show_total_annotation: bool = True
    show_scope_breakdown: bool = True
    show_uncertainty: bool = False
    show_data_quality: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "chart_mode": self.chart_mode.value,
            "value_unit": self.value_unit,
            "show_percentages": self.show_percentages,
            "show_legend": self.show_legend,
        }


# =============================================================================
# CARBON FOOTPRINT ENGINE
# =============================================================================

class CarbonFootprintEngine:
    """
    Engine for generating carbon footprint visualizations.

    Supports pie charts, bar charts, trend analysis, and regulatory overlays.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        theme: Optional[ThemeConfig] = None,
    ):
        """
        Initialize carbon footprint engine.

        Args:
            config: Global visualization configuration
            theme: Theme configuration for styling
        """
        self.config = config or get_default_config()
        self.theme = theme or self.config.theme
        self._cache: Dict[str, Any] = {}

    def generate(
        self,
        data: CarbonFootprintData,
        options: Optional[CarbonChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate carbon footprint visualization.

        Args:
            data: Carbon footprint data
            options: Chart configuration options

        Returns:
            Plotly-compatible chart specification
        """
        options = options or CarbonChartOptions()

        # Check cache
        cache_key = self._get_cache_key(data, options)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build chart based on mode
        if options.chart_mode == ChartMode.PIE:
            chart = self._build_pie_chart(data, options)
        elif options.chart_mode == ChartMode.DONUT:
            chart = self._build_donut_chart(data, options)
        elif options.chart_mode == ChartMode.BAR:
            chart = self._build_bar_chart(data, options)
        elif options.chart_mode == ChartMode.STACKED_BAR:
            chart = self._build_stacked_bar_chart(data, options)
        elif options.chart_mode == ChartMode.GROUPED_BAR:
            chart = self._build_grouped_bar_chart(data, options)
        elif options.chart_mode == ChartMode.TREEMAP:
            chart = self._build_treemap_chart(data, options)
        elif options.chart_mode == ChartMode.SUNBURST:
            chart = self._build_sunburst_chart(data, options)
        else:
            chart = self._build_donut_chart(data, options)

        # Add target overlays
        if options.show_targets and data.targets:
            chart = self._add_target_annotations(chart, data, options)

        # Add regulatory limit overlays
        if options.show_regulatory_limits and data.regulatory_limits:
            chart = self._add_limit_annotations(chart, data, options)

        # Add total annotation
        if options.show_total_annotation:
            chart = self._add_total_annotation(chart, data, options)

        # Cache result
        self._cache[cache_key] = chart

        return chart

    def _get_cache_key(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> str:
        """Generate cache key."""
        key_data = {
            "provenance": data.provenance_hash,
            "mode": options.chart_mode.value,
            "color_scheme": options.color_scheme,
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _build_pie_chart(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Dict[str, Any]:
        """Build pie chart."""
        return self._build_pie_or_donut(data, options, hole_size=0)

    def _build_donut_chart(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Dict[str, Any]:
        """Build donut chart."""
        return self._build_pie_or_donut(data, options, hole_size=options.hole_size)

    def _build_pie_or_donut(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
        hole_size: float,
    ) -> Dict[str, Any]:
        """Build pie or donut chart."""
        # Prepare data based on grouping
        if options.show_by_scope:
            labels, values, colors, hover_texts = self._prepare_scope_data(data, options)
        elif options.show_by_fuel:
            labels, values, colors, hover_texts = self._prepare_fuel_data(data, options)
        else:
            labels, values, colors, hover_texts = self._prepare_source_data(data, options)

        # Apply scale factor
        scaled_values = [v * options.scale_factor for v in values]

        # Build trace
        trace = {
            "type": "pie",
            "labels": labels,
            "values": scaled_values,
            "hole": hole_size,
            "pull": [options.pull_distance] * len(labels),
            "rotation": options.start_angle,
            "direction": options.direction,
            "sort": options.sort_values,
            "marker": {
                "colors": colors,
                "line": {"color": "white", "width": 2},
            },
            "textinfo": self._get_text_info(options),
            "textposition": "auto",
            "hovertemplate": "%{customdata}<extra></extra>",
            "customdata": hover_texts,
        }

        # Build layout
        layout = self._build_layout(data, options)

        return {
            "data": [trace],
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

    def _build_bar_chart(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Dict[str, Any]:
        """Build bar chart."""
        if options.show_by_scope:
            labels, values, colors, hover_texts = self._prepare_scope_data(data, options)
        elif options.show_by_fuel:
            labels, values, colors, hover_texts = self._prepare_fuel_data(data, options)
        else:
            labels, values, colors, hover_texts = self._prepare_source_data(data, options)

        scaled_values = [v * options.scale_factor for v in values]

        trace = {
            "type": "bar",
            "orientation": options.bar_orientation,
            "marker": {
                "color": colors,
                "line": {"color": "rgba(0,0,0,0.3)", "width": 1},
            },
            "hovertemplate": "%{customdata}<extra></extra>",
            "customdata": hover_texts,
        }

        if options.bar_orientation == "v":
            trace["x"] = labels
            trace["y"] = scaled_values
        else:
            trace["y"] = labels
            trace["x"] = scaled_values

        if options.show_bar_labels:
            trace["text"] = [self._format_value(v, options) for v in scaled_values]
            trace["textposition"] = "outside"

        layout = self._build_layout(data, options)
        layout["bargap"] = options.bar_gap

        if options.bar_orientation == "v":
            layout["xaxis"]["title"] = ""
            layout["yaxis"]["title"] = f"Emissions ({options.value_unit})"
        else:
            layout["yaxis"]["title"] = ""
            layout["xaxis"]["title"] = f"Emissions ({options.value_unit})"

        return {
            "data": [trace],
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

    def _build_stacked_bar_chart(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Dict[str, Any]:
        """Build stacked bar chart by scope."""
        traces = []

        scope_data = {
            EmissionScope.SCOPE_1: [],
            EmissionScope.SCOPE_2: [],
            EmissionScope.SCOPE_3: [],
        }

        # Get unique fuel types
        fuel_types = list(set(s.fuel_type for s in data.sources if s.fuel_type))

        for fuel_type in fuel_types:
            for scope in [EmissionScope.SCOPE_1, EmissionScope.SCOPE_2, EmissionScope.SCOPE_3]:
                sources = [s for s in data.sources if s.fuel_type == fuel_type and s.scope == scope]
                total = sum(s.emissions_co2e for s in sources) * options.scale_factor
                scope_data[scope].append(total)

        scope_colors = {
            EmissionScope.SCOPE_1: "#E74C3C",
            EmissionScope.SCOPE_2: "#F39C12",
            EmissionScope.SCOPE_3: "#3498DB",
        }

        scope_names = {
            EmissionScope.SCOPE_1: "Scope 1 (Direct)",
            EmissionScope.SCOPE_2: "Scope 2 (Indirect - Energy)",
            EmissionScope.SCOPE_3: "Scope 3 (Other Indirect)",
        }

        for scope in [EmissionScope.SCOPE_1, EmissionScope.SCOPE_2, EmissionScope.SCOPE_3]:
            trace = {
                "type": "bar",
                "name": scope_names[scope],
                "x": fuel_types,
                "y": scope_data[scope],
                "marker": {"color": scope_colors[scope]},
            }
            traces.append(trace)

        layout = self._build_layout(data, options)
        layout["barmode"] = "stack"
        layout["xaxis"]["title"] = "Fuel Type"
        layout["yaxis"]["title"] = f"Emissions ({options.value_unit})"

        return {
            "data": traces,
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

    def _build_grouped_bar_chart(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Dict[str, Any]:
        """Build grouped bar chart."""
        chart = self._build_stacked_bar_chart(data, options)
        chart["layout"]["barmode"] = "group"
        chart["layout"]["bargroupgap"] = options.bar_group_gap
        return chart

    def _build_treemap_chart(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Dict[str, Any]:
        """Build treemap chart."""
        ids = ["Total"]
        labels = ["Total Emissions"]
        parents = [""]
        values = [data.total_co2e * options.scale_factor]
        colors = ["#FFFFFF"]
        hover_texts = [f"<b>Total Emissions</b><br>{self._format_value(data.total_co2e, options)}"]

        # Add scopes
        for scope in [EmissionScope.SCOPE_1, EmissionScope.SCOPE_2, EmissionScope.SCOPE_3]:
            scope_sources = data.get_by_scope(scope)
            if scope_sources:
                scope_total = sum(s.emissions_co2e for s in scope_sources) * options.scale_factor
                scope_id = scope.value
                scope_label = scope.value.replace("_", " ").title()

                ids.append(scope_id)
                labels.append(scope_label)
                parents.append("Total")
                values.append(scope_total)
                colors.append(self._get_scope_color(scope))
                hover_texts.append(f"<b>{scope_label}</b><br>{self._format_value(scope_total / options.scale_factor, options)}")

                # Add sources within scope
                for source in scope_sources:
                    ids.append(f"{scope_id}_{source.source_id}")
                    labels.append(source.source_name)
                    parents.append(scope_id)
                    values.append(source.emissions_co2e * options.scale_factor)
                    colors.append(source.color)
                    hover_texts.append(
                        f"<b>{source.source_name}</b><br>"
                        f"Emissions: {self._format_value(source.emissions_co2e, options)}<br>"
                        f"Scope: {scope_label}"
                    )

        trace = {
            "type": "treemap",
            "ids": ids,
            "labels": labels,
            "parents": parents,
            "values": values,
            "marker": {"colors": colors},
            "textinfo": "label+percent parent",
            "hovertemplate": "%{customdata}<extra></extra>",
            "customdata": hover_texts,
            "branchvalues": "total",
        }

        layout = self._build_layout(data, options)

        return {
            "data": [trace],
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

    def _build_sunburst_chart(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Dict[str, Any]:
        """Build sunburst chart."""
        chart = self._build_treemap_chart(data, options)
        chart["data"][0]["type"] = "sunburst"
        return chart

    def _prepare_scope_data(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Tuple[List[str], List[float], List[str], List[str]]:
        """Prepare data grouped by scope."""
        labels = []
        values = []
        colors = []
        hover_texts = []

        scope_totals = {
            EmissionScope.SCOPE_1: data.scope_1_total,
            EmissionScope.SCOPE_2: data.scope_2_total,
            EmissionScope.SCOPE_3: data.scope_3_total,
        }

        scope_names = {
            EmissionScope.SCOPE_1: "Scope 1 (Direct)",
            EmissionScope.SCOPE_2: "Scope 2 (Indirect - Energy)",
            EmissionScope.SCOPE_3: "Scope 3 (Other Indirect)",
        }

        for scope, total in scope_totals.items():
            if total > 0:
                labels.append(scope_names[scope])
                values.append(total)
                colors.append(self._get_scope_color(scope))

                percent = (total / data.total_co2e * 100) if data.total_co2e > 0 else 0
                hover_texts.append(
                    f"<b>{scope_names[scope]}</b><br>"
                    f"Emissions: {self._format_value(total, options)}<br>"
                    f"Percentage: {percent:.1f}%"
                )

        return labels, values, colors, hover_texts

    def _prepare_fuel_data(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Tuple[List[str], List[float], List[str], List[str]]:
        """Prepare data grouped by fuel type."""
        fuel_totals: Dict[str, float] = {}

        for source in data.sources:
            fuel_type = source.fuel_type or "Unknown"
            fuel_totals[fuel_type] = fuel_totals.get(fuel_type, 0) + source.emissions_co2e

        labels = []
        values = []
        colors = []
        hover_texts = []

        for fuel_type, total in sorted(fuel_totals.items(), key=lambda x: x[1], reverse=True):
            labels.append(fuel_type.replace("_", " ").title())
            values.append(total)
            colors.append(get_fuel_color(fuel_type))

            percent = (total / data.total_co2e * 100) if data.total_co2e > 0 else 0
            hover_texts.append(
                f"<b>{fuel_type.replace('_', ' ').title()}</b><br>"
                f"Emissions: {self._format_value(total, options)}<br>"
                f"Percentage: {percent:.1f}%"
            )

        return labels, values, colors, hover_texts

    def _prepare_source_data(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Tuple[List[str], List[float], List[str], List[str]]:
        """Prepare data by individual source."""
        labels = []
        values = []
        colors = []
        hover_texts = []

        sorted_sources = sorted(data.sources, key=lambda s: s.emissions_co2e, reverse=True)

        for source in sorted_sources:
            labels.append(source.source_name)
            values.append(source.emissions_co2e)
            colors.append(source.color)

            percent = (source.emissions_co2e / data.total_co2e * 100) if data.total_co2e > 0 else 0
            hover_texts.append(
                f"<b>{source.source_name}</b><br>"
                f"Emissions: {self._format_value(source.emissions_co2e, options)}<br>"
                f"Percentage: {percent:.1f}%<br>"
                f"Scope: {source.scope.value.replace('_', ' ').title()}"
            )

        return labels, values, colors, hover_texts

    def _get_scope_color(self, scope: EmissionScope) -> str:
        """Get color for scope."""
        scope_colors = {
            EmissionScope.SCOPE_1: "#E74C3C",
            EmissionScope.SCOPE_2: "#F39C12",
            EmissionScope.SCOPE_3: "#3498DB",
            EmissionScope.ALL: "#9B59B6",
        }
        return scope_colors.get(scope, "#888888")

    def _get_text_info(self, options: CarbonChartOptions) -> str:
        """Get text info string for pie chart."""
        parts = []
        if options.show_percentages:
            parts.append("percent")
        if options.show_absolute_values:
            parts.append("value")
        parts.append("label")
        return "+".join(parts) if parts else "none"

    def _format_value(
        self,
        value: float,
        options: CarbonChartOptions,
    ) -> str:
        """Format value for display."""
        if options.use_scientific_notation and abs(value) >= 10000:
            return f"{value:.{options.decimal_places}e} {options.value_unit}"
        else:
            return f"{value:,.{options.decimal_places}f} {options.value_unit}"

    def _build_layout(
        self,
        data: CarbonFootprintData,
        options: CarbonChartOptions,
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
                    "y": -0.15,
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

        return layout

    def _add_target_annotations(
        self,
        chart: Dict[str, Any],
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Dict[str, Any]:
        """Add target annotations to chart."""
        annotations = chart["layout"].get("annotations", [])

        for target in data.targets:
            status = target.status
            status_color = status.color

            annotation_text = (
                f"Target: {target.target_name}<br>"
                f"Value: {self._format_value(target.target_value, options)}<br>"
                f"Progress: {target.progress_percent:.1f}%<br>"
                f"Status: {status.value.title()}"
            )

            annotations.append({
                "text": annotation_text,
                "x": 1,
                "y": 0.9 - len([a for a in annotations if "Target:" in a.get("text", "")]) * 0.15,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "right",
                "showarrow": False,
                "font": {"size": 10, "color": status_color},
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": status_color,
                "borderwidth": 1,
                "borderpad": 4,
            })

        chart["layout"]["annotations"] = annotations
        return chart

    def _add_limit_annotations(
        self,
        chart: Dict[str, Any],
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Dict[str, Any]:
        """Add regulatory limit annotations to chart."""
        annotations = chart["layout"].get("annotations", [])

        for limit in data.regulatory_limits:
            status = limit.check_compliance(data.total_co2e)
            status_color = status.color

            annotation_text = (
                f"Limit: {limit.limit_name}<br>"
                f"Value: {limit.limit_value:,.0f} {limit.unit}<br>"
                f"Status: {status.value.title()}"
            )

            annotations.append({
                "text": annotation_text,
                "x": 0,
                "y": 0.9 - len([a for a in annotations if "Limit:" in a.get("text", "")]) * 0.15,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "showarrow": False,
                "font": {"size": 10, "color": status_color},
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": status_color,
                "borderwidth": 1,
                "borderpad": 4,
            })

        chart["layout"]["annotations"] = annotations
        return chart

    def _add_total_annotation(
        self,
        chart: Dict[str, Any],
        data: CarbonFootprintData,
        options: CarbonChartOptions,
    ) -> Dict[str, Any]:
        """Add total emissions annotation to center of donut chart."""
        if options.chart_mode not in [ChartMode.DONUT]:
            return chart

        annotations = chart["layout"].get("annotations", [])

        total_text = self._format_value(data.total_co2e * options.scale_factor, options)

        annotations.append({
            "text": f"<b>Total</b><br>{total_text}",
            "x": 0.5,
            "y": 0.5,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {
                "size": 16,
                "color": self.theme.title_color,
            },
        })

        chart["layout"]["annotations"] = annotations
        return chart

    def generate_trend_chart(
        self,
        trend_data: EmissionTrendData,
        options: Optional[CarbonChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate emission trend chart.

        Args:
            trend_data: Time series emission data
            options: Chart options

        Returns:
            Plotly chart specification
        """
        options = options or CarbonChartOptions()
        options.title = options.title or "Emission Trends"

        timestamps = [dp.timestamp for dp in trend_data.data_points]
        values = [dp.value * options.scale_factor for dp in trend_data.data_points]

        # Main trend line
        trace = {
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Emissions",
            "x": timestamps,
            "y": values,
            "line": {
                "color": EmissionColors.CO2,
                "width": 2,
            },
            "marker": {"size": 4},
            "hovertemplate": "<b>%{x}</b><br>Emissions: %{y:,.2f}<extra></extra>",
        }

        traces = [trace]

        # Add moving average if enabled
        if options.show_trend_line and len(values) > options.trend_window:
            ma_values = self._calculate_moving_average(values, options.trend_window)
            ma_trace = {
                "type": "scatter",
                "mode": "lines",
                "name": f"{options.trend_window}-Day MA",
                "x": timestamps[options.trend_window - 1:],
                "y": ma_values,
                "line": {
                    "color": "#F39C12",
                    "width": 2,
                    "dash": "dash",
                },
            }
            traces.append(ma_trace)

        layout = self._build_layout(CarbonFootprintData(sources=[]), options)
        layout["xaxis"]["title"] = "Date"
        layout["yaxis"]["title"] = f"Emissions ({options.value_unit})"

        # Add trend annotation
        trend_arrow = "^" if trend_data.trend_direction == "increasing" else (
            "v" if trend_data.trend_direction == "decreasing" else "-"
        )
        trend_color = StatusColors.ERROR if trend_data.trend_direction == "increasing" else (
            StatusColors.SUCCESS if trend_data.trend_direction == "decreasing" else StatusColors.NEUTRAL
        )

        annotations = layout.get("annotations", [])
        annotations.append({
            "text": f"Trend: {trend_arrow} {abs(trend_data.trend_percent):.1f}%",
            "x": 1,
            "y": 1,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "right",
            "yanchor": "top",
            "showarrow": False,
            "font": {"size": 12, "color": trend_color},
        })
        layout["annotations"] = annotations

        return {
            "data": traces,
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

    def _calculate_moving_average(
        self,
        values: List[float],
        window: int,
    ) -> List[float]:
        """Calculate moving average."""
        ma = []
        for i in range(window - 1, len(values)):
            avg = sum(values[i - window + 1:i + 1]) / window
            ma.append(avg)
        return ma

    def generate_comparison_chart(
        self,
        data_sets: Dict[str, CarbonFootprintData],
        options: Optional[CarbonChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate comparison chart across multiple periods/scenarios.

        Args:
            data_sets: Dictionary of label -> CarbonFootprintData
            options: Chart options

        Returns:
            Plotly chart specification
        """
        options = options or CarbonChartOptions()
        options.title = options.title or "Emission Comparison"
        options.chart_mode = ChartMode.GROUPED_BAR

        labels = list(data_sets.keys())
        scope_1_values = [d.scope_1_total * options.scale_factor for d in data_sets.values()]
        scope_2_values = [d.scope_2_total * options.scale_factor for d in data_sets.values()]
        scope_3_values = [d.scope_3_total * options.scale_factor for d in data_sets.values()]

        traces = [
            {
                "type": "bar",
                "name": "Scope 1",
                "x": labels,
                "y": scope_1_values,
                "marker": {"color": self._get_scope_color(EmissionScope.SCOPE_1)},
            },
            {
                "type": "bar",
                "name": "Scope 2",
                "x": labels,
                "y": scope_2_values,
                "marker": {"color": self._get_scope_color(EmissionScope.SCOPE_2)},
            },
            {
                "type": "bar",
                "name": "Scope 3",
                "x": labels,
                "y": scope_3_values,
                "marker": {"color": self._get_scope_color(EmissionScope.SCOPE_3)},
            },
        ]

        layout = self._build_layout(list(data_sets.values())[0], options)
        layout["barmode"] = "group"
        layout["xaxis"]["title"] = ""
        layout["yaxis"]["title"] = f"Emissions ({options.value_unit})"

        return {
            "data": traces,
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

    def generate_intensity_chart(
        self,
        data: CarbonFootprintData,
        options: Optional[CarbonChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate carbon intensity chart.

        Args:
            data: Carbon footprint data
            options: Chart options

        Returns:
            Plotly chart specification
        """
        options = options or CarbonChartOptions()
        options.title = options.title or "Carbon Intensity by Source"

        sources_with_intensity = [s for s in data.sources if s.carbon_intensity > 0]
        sorted_sources = sorted(sources_with_intensity, key=lambda s: s.carbon_intensity, reverse=True)

        labels = [s.source_name for s in sorted_sources]
        intensities = [s.carbon_intensity for s in sorted_sources]
        colors = [s.color for s in sorted_sources]

        trace = {
            "type": "bar",
            "x": labels,
            "y": intensities,
            "marker": {
                "color": colors,
                "line": {"color": "rgba(0,0,0,0.3)", "width": 1},
            },
            "text": [f"{i:.1f}" for i in intensities],
            "textposition": "outside",
        }

        layout = self._build_layout(data, options)
        layout["xaxis"]["title"] = ""
        layout["yaxis"]["title"] = "Carbon Intensity (kg CO2e/GJ)"
        layout["xaxis"]["tickangle"] = -45

        return {
            "data": [trace],
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

    def to_json(self, chart: Dict[str, Any]) -> str:
        """Export chart to JSON string."""
        return json.dumps(chart, indent=2, default=str)

    def clear_cache(self) -> None:
        """Clear the chart cache."""
        self._cache.clear()


# =============================================================================
# SPECIALIZED GENERATORS
# =============================================================================

class ScopeBreakdownGenerator(CarbonFootprintEngine):
    """Specialized generator for scope-based breakdown charts."""

    def generate_scope_comparison(
        self,
        data: CarbonFootprintData,
        options: Optional[CarbonChartOptions] = None,
    ) -> Dict[str, Any]:
        """Generate scope comparison visualization."""
        options = options or CarbonChartOptions()
        options.title = "Emissions by GHG Protocol Scope"
        options.show_by_scope = True
        options.chart_mode = ChartMode.BAR

        return self.generate(data, options)


class RegulatoryComplianceGenerator(CarbonFootprintEngine):
    """Specialized generator for regulatory compliance visualization."""

    def generate_compliance_dashboard(
        self,
        data: CarbonFootprintData,
        options: Optional[CarbonChartOptions] = None,
    ) -> Dict[str, Any]:
        """Generate regulatory compliance dashboard."""
        options = options or CarbonChartOptions()
        options.title = "Regulatory Compliance Status"
        options.show_regulatory_limits = True

        # Create gauge-style compliance indicator
        compliance_status = data.get_compliance_status()

        trace = {
            "type": "indicator",
            "mode": "gauge+number+delta",
            "value": data.total_co2e,
            "title": {"text": "Total Emissions vs Limit"},
            "gauge": {
                "axis": {"range": [0, max(l.limit_value for l in data.regulatory_limits) * 1.2] if data.regulatory_limits else [0, data.total_co2e * 1.5]},
                "bar": {"color": compliance_status.color},
                "steps": [
                    {"range": [0, data.regulatory_limits[0].limit_value * 0.9] if data.regulatory_limits else [0, data.total_co2e * 0.9], "color": StatusColors.SUCCESS},
                    {"range": [data.regulatory_limits[0].limit_value * 0.9, data.regulatory_limits[0].limit_value] if data.regulatory_limits else [data.total_co2e * 0.9, data.total_co2e], "color": StatusColors.WARNING},
                    {"range": [data.regulatory_limits[0].limit_value, data.regulatory_limits[0].limit_value * 1.5] if data.regulatory_limits else [data.total_co2e, data.total_co2e * 1.5], "color": StatusColors.ERROR},
                ] if data.regulatory_limits else [],
                "threshold": {
                    "line": {"color": "#E74C3C", "width": 4},
                    "thickness": 0.75,
                    "value": data.regulatory_limits[0].limit_value if data.regulatory_limits else data.total_co2e,
                },
            },
        }

        layout = self._build_layout(data, options)
        layout["height"] = 300

        return {
            "data": [trace],
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def create_sample_carbon_data() -> CarbonFootprintData:
    """Create sample carbon footprint data for demonstration."""
    sources = [
        EmissionSource(
            source_id="coal_boiler",
            source_name="Coal Boiler",
            fuel_type="coal",
            scope=EmissionScope.SCOPE_1,
            emissions_co2e=50000,
            emissions_co2=49500,
            emissions_ch4=100,
            emissions_n2o=50,
            energy_consumed=500000,
            emission_factor=94.6,
        ),
        EmissionSource(
            source_id="natural_gas_turbine",
            source_name="Natural Gas Turbine",
            fuel_type="natural_gas",
            scope=EmissionScope.SCOPE_1,
            emissions_co2e=25000,
            emissions_co2=24800,
            emissions_ch4=150,
            emissions_n2o=25,
            energy_consumed=450000,
            emission_factor=56.1,
        ),
        EmissionSource(
            source_id="grid_electricity",
            source_name="Grid Electricity",
            fuel_type="electricity",
            scope=EmissionScope.SCOPE_2,
            emissions_co2e=15000,
            energy_consumed=50000,
            emission_factor=300,
        ),
        EmissionSource(
            source_id="biomass_boiler",
            source_name="Biomass Boiler",
            fuel_type="biomass",
            scope=EmissionScope.SCOPE_1,
            emissions_co2e=5000,
            energy_consumed=200000,
            emission_factor=10,
        ),
        EmissionSource(
            source_id="transport_fuel",
            source_name="Transport Fleet",
            fuel_type="diesel",
            scope=EmissionScope.SCOPE_3,
            emissions_co2e=8000,
            energy_consumed=100000,
            emission_factor=74.1,
        ),
    ]

    targets = [
        EmissionTarget(
            target_id="sbti_2030",
            target_name="SBTi 2030 Target",
            target_type=TargetType.SCIENCE_BASED,
            target_value=70000,
            baseline_value=120000,
            baseline_year=2020,
            target_year=2030,
            current_value=103000,
            scope=EmissionScope.ALL,
            source="Science Based Targets initiative",
        ),
    ]

    regulatory_limits = [
        RegulatoryLimit(
            limit_id="eu_ets_2024",
            limit_name="EU ETS Annual Cap",
            jurisdiction="European Union",
            limit_value=110000,
            unit="kg CO2e",
            emission_type=EmissionType.CO2E,
            averaging_period="annual",
            applicable_scope=EmissionScope.SCOPE_1,
            effective_date="2024-01-01",
            penalty_per_unit=100,
            regulation_reference="EU ETS Directive 2003/87/EC",
        ),
    ]

    return CarbonFootprintData(
        sources=sources,
        targets=targets,
        regulatory_limits=regulatory_limits,
        reporting_period="2024",
        facility_id="FACILITY_001",
        facility_name="Main Production Facility",
        organization="GreenLang Demo Corp",
    )


def example_donut_chart():
    """Example: Generate donut chart."""
    print("Generating carbon footprint donut chart...")

    data = create_sample_carbon_data()
    engine = CarbonFootprintEngine()
    options = CarbonChartOptions(
        title="Carbon Footprint by Fuel Type",
        subtitle="Annual Emissions 2024",
        chart_mode=ChartMode.DONUT,
        show_by_fuel=True,
        show_by_scope=False,
    )

    chart = engine.generate(data, options)
    print(f"Donut chart generated with {len(chart['data'][0]['labels'])} categories")
    return chart


def example_scope_breakdown():
    """Example: Generate scope breakdown chart."""
    print("Generating scope breakdown chart...")

    data = create_sample_carbon_data()
    engine = ScopeBreakdownGenerator()

    chart = engine.generate_scope_comparison(data)
    print(f"Scope breakdown chart generated")
    return chart


def example_treemap():
    """Example: Generate treemap chart."""
    print("Generating treemap chart...")

    data = create_sample_carbon_data()
    engine = CarbonFootprintEngine()
    options = CarbonChartOptions(
        title="Carbon Footprint Hierarchy",
        chart_mode=ChartMode.TREEMAP,
    )

    chart = engine.generate(data, options)
    print(f"Treemap chart generated")
    return chart


def example_intensity_chart():
    """Example: Generate carbon intensity chart."""
    print("Generating carbon intensity chart...")

    data = create_sample_carbon_data()
    engine = CarbonFootprintEngine()

    chart = engine.generate_intensity_chart(data)
    print(f"Intensity chart generated")
    return chart


def example_compliance_dashboard():
    """Example: Generate compliance dashboard."""
    print("Generating compliance dashboard...")

    data = create_sample_carbon_data()
    engine = RegulatoryComplianceGenerator()

    chart = engine.generate_compliance_dashboard(data)
    print(f"Compliance dashboard generated")
    return chart


def run_all_examples():
    """Run all carbon footprint visualization examples."""
    print("=" * 60)
    print("GL-011 FUELCRAFT - Carbon Footprint Visualization Examples")
    print("=" * 60)

    examples = [
        ("Donut Chart", example_donut_chart),
        ("Scope Breakdown", example_scope_breakdown),
        ("Treemap", example_treemap),
        ("Intensity Chart", example_intensity_chart),
        ("Compliance Dashboard", example_compliance_dashboard),
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
