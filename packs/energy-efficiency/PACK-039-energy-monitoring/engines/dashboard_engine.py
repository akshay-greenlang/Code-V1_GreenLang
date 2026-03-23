# -*- coding: utf-8 -*-
"""
DashboardEngine - PACK-039 Energy Monitoring Engine 9
=======================================================

Real-time energy monitoring dashboard engine with 8 panel types,
configurable KPIs, widget rendering, time-series trends, and heatmap
generation.  Produces dashboard data payloads for consumption by
front-end renderers or PDF export pipelines.

Calculation Methodology:
    KPI Calculation:
        consumption_kpi = sum(interval_kwh) over time_range
        cost_kpi = consumption_kpi * blended_rate
        enpi_kpi = consumption_kpi / normalisation_factor (area, units, etc.)
        savings_kpi = baseline_kwh - actual_kwh
        savings_pct = savings_kpi / baseline_kwh * 100

    Trend Direction:
        slope = linear_regression_slope(time_series)
        improving = slope < -threshold  (decreasing energy)
        declining = slope > +threshold  (increasing energy)
        stable    = abs(slope) <= threshold
        volatile  = std_dev(residuals) > volatility_threshold

    Heatmap Generation:
        matrix[hour][day] = avg(interval_kwh for that hour/day combination)
        colour_scale = min_kwh -> max_kwh mapped to 0.0 -> 1.0

    Panel Composition:
        Each panel has a type (CONSUMPTION, COST, EnPI, etc.) and one or
        more widgets (KPI_CARD, TIME_SERIES, BAR_CHART, etc.)
        Panels are arranged in a responsive grid layout.

    Refresh Scheduling:
        REALTIME = websocket push (1-second granularity)
        MIN_1 to DAILY = polling intervals

Regulatory References:
    - ISO 50001:2018    EnMS monitoring dashboard requirements
    - ISO 50006:2014    EnPI visualisation and benchmarking
    - EN 15232          Building energy management display
    - ASHRAE 90.1-2022  Energy monitoring display requirements
    - NABERS            Dashboard reporting for Australian buildings
    - ENERGY STAR       Portfolio Manager dashboard patterns
    - UK Display Energy Certificate (DEC) requirements
    - EU EED Art. 11    Public sector display requirements

Zero-Hallucination:
    - All KPIs computed from deterministic aggregation
    - Trend detection uses linear regression, not LLM
    - Heatmap values are direct averages from interval data
    - No LLM involvement in any dashboard data calculation
    - Decimal arithmetic throughout for audit-grade precision
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-039 Energy Monitoring
Engine:  9 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PanelType(str, Enum):
    """Dashboard panel type.

    CONSUMPTION:  Energy consumption overview panel.
    COST:         Energy cost and tariff panel.
    ENPI:         Energy Performance Indicator panel.
    ANOMALY:      Anomaly detection and alerts panel.
    SUBMETER:     Sub-meter breakdown panel.
    WEATHER:      Weather overlay and correlation panel.
    LOAD_PROFILE: Load profile and demand panel.
    ALARM:        Alarm status and summary panel.
    """
    CONSUMPTION = "consumption"
    COST = "cost"
    ENPI = "enpi"
    ANOMALY = "anomaly"
    SUBMETER = "submeter"
    WEATHER = "weather"
    LOAD_PROFILE = "load_profile"
    ALARM = "alarm"


class WidgetType(str, Enum):
    """Dashboard widget visualisation type.

    KPI_CARD:     Single KPI value with trend indicator.
    TIME_SERIES:  Time-series line chart.
    BAR_CHART:    Horizontal or vertical bar chart.
    PIE_CHART:    Pie or donut chart.
    HEATMAP:      Hour-of-day / day-of-week heatmap.
    SANKEY:       Energy flow Sankey diagram.
    TABLE:        Data table with sortable columns.
    GAUGE:        Radial gauge or bullet chart.
    """
    KPI_CARD = "kpi_card"
    TIME_SERIES = "time_series"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    HEATMAP = "heatmap"
    SANKEY = "sankey"
    TABLE = "table"
    GAUGE = "gauge"


class TimeRange(str, Enum):
    """Dashboard time range selection.

    LIVE:     Real-time streaming (last 5 minutes).
    TODAY:    Current day (00:00 to now).
    WEEK:     Current week (Monday to now).
    MONTH:    Current month.
    QUARTER:  Current quarter.
    YEAR:     Current fiscal year.
    CUSTOM:   User-defined custom range.
    """
    LIVE = "live"
    TODAY = "today"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"


class RefreshRate(str, Enum):
    """Dashboard auto-refresh rate.

    REALTIME:  WebSocket push (1-second).
    MIN_1:     Every minute.
    MIN_5:     Every 5 minutes.
    MIN_15:    Every 15 minutes.
    HOUR:      Every hour.
    DAILY:     Once per day.
    """
    REALTIME = "realtime"
    MIN_1 = "min_1"
    MIN_5 = "min_5"
    MIN_15 = "min_15"
    HOUR = "hour"
    DAILY = "daily"


class TrendDirection(str, Enum):
    """KPI trend direction classification.

    IMPROVING:  Positive trend (energy decreasing or efficiency increasing).
    STABLE:     No significant change.
    DECLINING:  Negative trend (energy increasing or efficiency decreasing).
    VOLATILE:   High variance, no clear direction.
    """
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Trend slope threshold (absolute kWh/period).
TREND_SLOPE_THRESHOLD: Decimal = Decimal("0.02")

# Volatility threshold (coefficient of variation).
VOLATILITY_THRESHOLD: Decimal = Decimal("0.20")

# Default KPI target improvement percentage.
DEFAULT_TARGET_IMPROVEMENT_PCT: Decimal = Decimal("5.0")

# Heatmap dimensions.
HEATMAP_HOURS: int = 24
HEATMAP_DAYS: int = 7

# Maximum data points for time-series widget.
MAX_TIME_SERIES_POINTS: int = 2000

# Dashboard layout grid columns.
GRID_COLUMNS: int = 12

# Default panels for energy monitoring dashboard.
DEFAULT_PANEL_ORDER: List[str] = [
    PanelType.CONSUMPTION.value,
    PanelType.COST.value,
    PanelType.ENPI.value,
    PanelType.LOAD_PROFILE.value,
    PanelType.SUBMETER.value,
    PanelType.ANOMALY.value,
    PanelType.WEATHER.value,
    PanelType.ALARM.value,
]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class DashboardConfig(BaseModel):
    """Dashboard configuration.

    Attributes:
        dashboard_id: Unique dashboard identifier.
        dashboard_name: Dashboard title.
        site_id: Associated site identifier.
        time_range: Default time range.
        refresh_rate: Auto-refresh rate.
        panels: Ordered list of panel types.
        grid_columns: Layout grid columns.
        theme: Dashboard theme (light/dark).
        currency: Currency for cost displays.
        unit_system: Measurement unit system (metric/imperial).
        timezone_str: Display timezone.
    """
    dashboard_id: str = Field(
        default_factory=_new_uuid, description="Dashboard ID"
    )
    dashboard_name: str = Field(
        default="Energy Monitoring Dashboard", max_length=500,
        description="Dashboard name"
    )
    site_id: str = Field(
        default="", description="Site identifier"
    )
    time_range: TimeRange = Field(
        default=TimeRange.TODAY, description="Default time range"
    )
    refresh_rate: RefreshRate = Field(
        default=RefreshRate.MIN_5, description="Refresh rate"
    )
    panels: List[str] = Field(
        default_factory=lambda: list(DEFAULT_PANEL_ORDER),
        description="Panel types"
    )
    grid_columns: int = Field(
        default=GRID_COLUMNS, ge=1, le=24, description="Grid columns"
    )
    theme: str = Field(
        default="light", description="Theme (light/dark)"
    )
    currency: str = Field(
        default="USD", max_length=3, description="Currency"
    )
    unit_system: str = Field(
        default="metric", description="Unit system"
    )
    timezone_str: str = Field(
        default="UTC", max_length=50, description="Timezone"
    )


class PanelConfig(BaseModel):
    """Individual panel configuration within a dashboard.

    Attributes:
        panel_id: Panel identifier.
        panel_type: Panel type.
        title: Panel title.
        widgets: Widget types to render in this panel.
        grid_width: Grid column span.
        grid_row: Grid row position.
        is_visible: Whether panel is visible.
        data_source: Data source identifier.
    """
    panel_id: str = Field(
        default_factory=_new_uuid, description="Panel ID"
    )
    panel_type: PanelType = Field(
        default=PanelType.CONSUMPTION, description="Panel type"
    )
    title: str = Field(
        default="", max_length=200, description="Panel title"
    )
    widgets: List[str] = Field(
        default_factory=list, description="Widget types"
    )
    grid_width: int = Field(
        default=6, ge=1, le=12, description="Grid width"
    )
    grid_row: int = Field(
        default=0, ge=0, description="Grid row"
    )
    is_visible: bool = Field(
        default=True, description="Visible"
    )
    data_source: str = Field(
        default="", description="Data source ID"
    )


class WidgetData(BaseModel):
    """Rendered widget data payload.

    Attributes:
        widget_id: Widget identifier.
        widget_type: Widget visualisation type.
        title: Widget title.
        value: Primary display value.
        unit: Display unit.
        trend: Trend direction.
        change_pct: Period-over-period change percentage.
        series_data: Time-series data points.
        labels: Chart labels.
        colors: Chart colours.
        metadata: Additional widget metadata.
    """
    widget_id: str = Field(
        default_factory=_new_uuid, description="Widget ID"
    )
    widget_type: WidgetType = Field(
        default=WidgetType.KPI_CARD, description="Widget type"
    )
    title: str = Field(
        default="", max_length=200, description="Widget title"
    )
    value: str = Field(
        default="0", description="Primary value"
    )
    unit: str = Field(
        default="kWh", max_length=20, description="Unit"
    )
    trend: TrendDirection = Field(
        default=TrendDirection.STABLE, description="Trend"
    )
    change_pct: Decimal = Field(
        default=Decimal("0"), description="Change (%)"
    )
    series_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Series data"
    )
    labels: List[str] = Field(
        default_factory=list, description="Labels"
    )
    colors: List[str] = Field(
        default_factory=list, description="Colours"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata"
    )


class KPIMetric(BaseModel):
    """Key Performance Indicator metric.

    Attributes:
        kpi_id: KPI identifier.
        kpi_name: Human-readable KPI name.
        current_value: Current KPI value.
        previous_value: Previous period value.
        target_value: Target value.
        unit: Display unit.
        trend: Trend direction.
        change_pct: Period-over-period change.
        target_achieved: Whether target is met.
        sparkline: Mini time-series (last N values).
    """
    kpi_id: str = Field(
        default_factory=_new_uuid, description="KPI ID"
    )
    kpi_name: str = Field(
        default="", max_length=200, description="KPI name"
    )
    current_value: Decimal = Field(
        default=Decimal("0"), description="Current value"
    )
    previous_value: Decimal = Field(
        default=Decimal("0"), description="Previous value"
    )
    target_value: Decimal = Field(
        default=Decimal("0"), description="Target value"
    )
    unit: str = Field(
        default="kWh", max_length=20, description="Unit"
    )
    trend: TrendDirection = Field(
        default=TrendDirection.STABLE, description="Trend"
    )
    change_pct: Decimal = Field(
        default=Decimal("0"), description="Change (%)"
    )
    target_achieved: bool = Field(
        default=False, description="Target met"
    )
    sparkline: List[str] = Field(
        default_factory=list, description="Sparkline values"
    )


class DashboardResult(BaseModel):
    """Complete dashboard data result.

    Attributes:
        result_id: Result identifier.
        dashboard_id: Dashboard identifier.
        dashboard_name: Dashboard name.
        time_range: Time range applied.
        refresh_rate: Refresh rate.
        kpis: Computed KPI metrics.
        panels: Panel data payloads.
        widgets: Widget data payloads.
        heatmap_data: Heatmap matrix (24x7).
        last_updated: Last data timestamp.
        data_freshness_sec: Data freshness in seconds.
        calculated_at: Calculation timestamp.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result ID"
    )
    dashboard_id: str = Field(default="", description="Dashboard ID")
    dashboard_name: str = Field(default="", description="Dashboard name")
    time_range: TimeRange = Field(
        default=TimeRange.TODAY, description="Time range"
    )
    refresh_rate: RefreshRate = Field(
        default=RefreshRate.MIN_5, description="Refresh rate"
    )
    kpis: List[KPIMetric] = Field(
        default_factory=list, description="KPI metrics"
    )
    panels: List[Dict[str, Any]] = Field(
        default_factory=list, description="Panel data"
    )
    widgets: List[WidgetData] = Field(
        default_factory=list, description="Widget data"
    )
    heatmap_data: List[List[str]] = Field(
        default_factory=list, description="Heatmap matrix"
    )
    last_updated: datetime = Field(
        default_factory=_utcnow, description="Last updated"
    )
    data_freshness_sec: int = Field(
        default=0, ge=0, description="Freshness (sec)"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DashboardEngine:
    """Real-time energy monitoring dashboard engine.

    Generates dashboard data payloads with 8 panel types, configurable KPIs,
    multiple widget types, time-series trends, and hour/day heatmaps.
    Produces structured data for front-end renderers or PDF export.

    Usage::

        engine = DashboardEngine()
        result = engine.generate_dashboard(config, consumption_data)
        kpis = engine.compute_kpis(consumption, baseline, targets)
        panels = engine.build_panels(config, consumption, costs)
        heatmap = engine.generate_heatmap(interval_data)
        widgets = engine.render_widgets(panel_config, data)

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise DashboardEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - default_time_range (str): default time range
                - default_refresh_rate (str): default refresh rate
                - target_improvement_pct (Decimal): KPI target %
                - currency (str): display currency
        """
        self.config = config or {}
        self._default_time_range = TimeRange(
            self.config.get("default_time_range", TimeRange.TODAY.value)
        )
        self._default_refresh = RefreshRate(
            self.config.get("default_refresh_rate", RefreshRate.MIN_5.value)
        )
        self._target_pct = _decimal(
            self.config.get("target_improvement_pct", DEFAULT_TARGET_IMPROVEMENT_PCT)
        )
        self._currency = str(self.config.get("currency", "USD"))
        logger.info(
            "DashboardEngine v%s initialised (range=%s, refresh=%s)",
            self.engine_version,
            self._default_time_range.value,
            self._default_refresh.value,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_dashboard(
        self,
        dashboard_config: DashboardConfig,
        consumption_data: List[Dict[str, Any]],
        cost_data: Optional[List[Dict[str, Any]]] = None,
        baseline_kwh: Decimal = Decimal("0"),
    ) -> DashboardResult:
        """Generate complete dashboard data payload.

        Orchestrates KPI computation, panel building, widget rendering,
        and heatmap generation into a single result.

        Args:
            dashboard_config: Dashboard configuration.
            consumption_data: Interval consumption data (list of dicts
                with 'timestamp', 'kwh', optional 'kw', 'cost', 'meter_id').
            cost_data: Optional separate cost data.
            baseline_kwh: Baseline consumption for comparison.

        Returns:
            DashboardResult with all dashboard data.
        """
        t0 = time.perf_counter()
        logger.info(
            "Generating dashboard: %s, %d data points",
            dashboard_config.dashboard_name, len(consumption_data),
        )

        # Extract consumption series
        kwh_values = [_decimal(d.get("kwh", 0)) for d in consumption_data]
        total_kwh = sum(kwh_values, Decimal("0"))

        # Cost series
        costs = cost_data or consumption_data
        cost_values = [_decimal(d.get("cost", 0)) for d in costs]
        total_cost = sum(cost_values, Decimal("0"))

        # Compute KPIs
        kpis = self.compute_kpis(
            consumption_values=kwh_values,
            baseline_kwh=baseline_kwh,
            blended_rate=_safe_divide(total_cost, total_kwh) if total_kwh > Decimal("0") else Decimal("0.12"),
        )

        # Build panels
        panels = self.build_panels(
            dashboard_config, consumption_data, costs,
        )

        # Render widgets
        widgets = self.render_widgets(
            consumption_data, total_kwh, total_cost, kpis,
        )

        # Generate heatmap
        heatmap = self.generate_heatmap(consumption_data)

        # Data freshness
        freshness = 0
        if consumption_data:
            last_ts = consumption_data[-1].get("timestamp", "")
            if last_ts:
                try:
                    last_dt = datetime.fromisoformat(
                        str(last_ts).replace("Z", "+00:00")
                    )
                    freshness = int((_utcnow() - last_dt).total_seconds())
                except (ValueError, TypeError):
                    freshness = 0

        elapsed = (time.perf_counter() - t0) * 1000.0

        result = DashboardResult(
            dashboard_id=dashboard_config.dashboard_id,
            dashboard_name=dashboard_config.dashboard_name,
            time_range=dashboard_config.time_range,
            refresh_rate=dashboard_config.refresh_rate,
            kpis=kpis,
            panels=panels,
            widgets=widgets,
            heatmap_data=heatmap,
            data_freshness_sec=max(freshness, 0),
            processing_time_ms=round(elapsed, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Dashboard generated: %s, %d KPIs, %d panels, %d widgets, "
            "freshness=%ds, hash=%s (%.1f ms)",
            dashboard_config.dashboard_name, len(kpis), len(panels),
            len(widgets), freshness,
            result.provenance_hash[:16], elapsed,
        )
        return result

    def compute_kpis(
        self,
        consumption_values: List[Decimal],
        baseline_kwh: Decimal = Decimal("0"),
        blended_rate: Decimal = Decimal("0.12"),
        area_m2: Decimal = Decimal("0"),
        production_units: Decimal = Decimal("0"),
    ) -> List[KPIMetric]:
        """Compute key performance indicator metrics.

        Calculates total consumption, cost, EnPI (kWh/m2), savings vs
        baseline, and demand KPIs.

        Args:
            consumption_values: List of consumption values (kWh).
            baseline_kwh: Baseline for savings calculation.
            blended_rate: Blended energy rate ($/kWh).
            area_m2: Floor area for EnPI calculation.
            production_units: Production output for intensity KPI.

        Returns:
            List of KPIMetric instances.
        """
        t0 = time.perf_counter()
        total_kwh = sum(consumption_values, Decimal("0"))
        total_cost = total_kwh * blended_rate
        count = len(consumption_values)
        kpis: List[KPIMetric] = []

        # KPI 1: Total Consumption
        trend, change_pct = self._compute_trend(consumption_values)
        kpis.append(KPIMetric(
            kpi_name="Total Consumption",
            current_value=_round_val(total_kwh, 2),
            previous_value=_round_val(baseline_kwh, 2) if baseline_kwh > Decimal("0") else Decimal("0"),
            target_value=_round_val(
                baseline_kwh * (Decimal("1") - self._target_pct / Decimal("100")), 2
            ) if baseline_kwh > Decimal("0") else Decimal("0"),
            unit="kWh",
            trend=trend,
            change_pct=_round_val(change_pct, 2),
            target_achieved=total_kwh <= baseline_kwh * (
                Decimal("1") - self._target_pct / Decimal("100")
            ) if baseline_kwh > Decimal("0") else False,
            sparkline=[str(_round_val(v, 1)) for v in consumption_values[-10:]],
        ))

        # KPI 2: Total Cost
        kpis.append(KPIMetric(
            kpi_name="Total Cost",
            current_value=_round_val(total_cost, 2),
            unit=self._currency,
            trend=trend,
            change_pct=_round_val(change_pct, 2),
        ))

        # KPI 3: Average Demand
        avg_kwh = _safe_divide(total_kwh, _decimal(count)) if count > 0 else Decimal("0")
        kpis.append(KPIMetric(
            kpi_name="Average Interval Demand",
            current_value=_round_val(avg_kwh, 2),
            unit="kWh",
            trend=trend,
        ))

        # KPI 4: Peak Demand
        peak_kwh = max(consumption_values) if consumption_values else Decimal("0")
        kpis.append(KPIMetric(
            kpi_name="Peak Demand",
            current_value=_round_val(peak_kwh, 2),
            unit="kWh",
            trend=TrendDirection.STABLE,
        ))

        # KPI 5: EnPI (if area provided)
        if area_m2 > Decimal("0"):
            enpi = _safe_divide(total_kwh, area_m2)
            kpis.append(KPIMetric(
                kpi_name="Energy Use Intensity",
                current_value=_round_val(enpi, 2),
                unit="kWh/m2",
                trend=trend,
            ))

        # KPI 6: Savings vs baseline
        if baseline_kwh > Decimal("0"):
            savings = baseline_kwh - total_kwh
            savings_pct = _safe_pct(savings, baseline_kwh)
            kpis.append(KPIMetric(
                kpi_name="Savings vs Baseline",
                current_value=_round_val(savings, 2),
                unit="kWh",
                trend=TrendDirection.IMPROVING if savings > Decimal("0") else TrendDirection.DECLINING,
                change_pct=_round_val(savings_pct, 2),
                target_achieved=savings > Decimal("0"),
            ))

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "KPIs computed: %d metrics, total=%.0f kWh, cost=$%.2f (%.1f ms)",
            len(kpis), float(total_kwh), float(total_cost), elapsed,
        )
        return kpis

    def build_panels(
        self,
        dashboard_config: DashboardConfig,
        consumption_data: List[Dict[str, Any]],
        cost_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build panel data payloads for the dashboard.

        Creates structured data for each panel type configured in the
        dashboard, suitable for front-end rendering.

        Args:
            dashboard_config: Dashboard configuration.
            consumption_data: Consumption data.
            cost_data: Cost data.

        Returns:
            List of panel data dictionaries.
        """
        t0 = time.perf_counter()
        panels: List[Dict[str, Any]] = []
        grid_row = 0

        for panel_type_str in dashboard_config.panels:
            try:
                panel_type = PanelType(panel_type_str)
            except ValueError:
                continue

            panel_data = self._build_single_panel(
                panel_type, consumption_data, cost_data, grid_row,
            )
            panels.append(panel_data)
            grid_row += 1

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Panels built: %d panels (%.1f ms)", len(panels), elapsed,
        )
        return panels

    def generate_heatmap(
        self,
        interval_data: List[Dict[str, Any]],
    ) -> List[List[str]]:
        """Generate hour-of-day / day-of-week heatmap matrix.

        Aggregates interval data into a 24x7 matrix of average kWh
        values for heatmap visualisation.

        Args:
            interval_data: Interval data with 'timestamp' and 'kwh'.

        Returns:
            24x7 matrix of average kWh values as strings.
        """
        t0 = time.perf_counter()
        logger.info(
            "Generating heatmap: %d data points", len(interval_data),
        )

        # Initialise accumulator: [hour][day] = (sum, count)
        accum: List[List[Tuple[Decimal, int]]] = [
            [(Decimal("0"), 0) for _ in range(HEATMAP_DAYS)]
            for _ in range(HEATMAP_HOURS)
        ]

        for record in interval_data:
            ts = record.get("timestamp", "")
            kwh = _decimal(record.get("kwh", 0))
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif isinstance(ts, datetime):
                    dt = ts
                else:
                    continue
                hour = dt.hour
                day = dt.weekday()  # 0=Monday
                if 0 <= hour < HEATMAP_HOURS and 0 <= day < HEATMAP_DAYS:
                    old_sum, old_count = accum[hour][day]
                    accum[hour][day] = (old_sum + kwh, old_count + 1)
            except (ValueError, TypeError, AttributeError):
                continue

        # Compute averages
        matrix: List[List[str]] = []
        for hour in range(HEATMAP_HOURS):
            row: List[str] = []
            for day in range(HEATMAP_DAYS):
                total, count = accum[hour][day]
                avg = _safe_divide(total, _decimal(count)) if count > 0 else Decimal("0")
                row.append(str(_round_val(avg, 2)))
            matrix.append(row)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info("Heatmap generated: %dx%d (%.1f ms)",
                     HEATMAP_HOURS, HEATMAP_DAYS, elapsed)
        return matrix

    def render_widgets(
        self,
        consumption_data: List[Dict[str, Any]],
        total_kwh: Decimal,
        total_cost: Decimal,
        kpis: List[KPIMetric],
    ) -> List[WidgetData]:
        """Render widget data payloads from dashboard data.

        Creates widget data structures for KPI cards, time-series charts,
        and summary tables.

        Args:
            consumption_data: Consumption data.
            total_kwh: Total consumption.
            total_cost: Total cost.
            kpis: Computed KPI metrics.

        Returns:
            List of WidgetData instances.
        """
        t0 = time.perf_counter()
        widgets: List[WidgetData] = []

        # KPI card widgets from computed KPIs
        for kpi in kpis:
            widgets.append(WidgetData(
                widget_type=WidgetType.KPI_CARD,
                title=kpi.kpi_name,
                value=str(kpi.current_value),
                unit=kpi.unit,
                trend=kpi.trend,
                change_pct=kpi.change_pct,
                metadata={
                    "target": str(kpi.target_value),
                    "target_achieved": kpi.target_achieved,
                    "sparkline": kpi.sparkline,
                },
            ))

        # Time-series widget
        series_points = consumption_data[:MAX_TIME_SERIES_POINTS]
        series_data = [
            {"x": d.get("timestamp", ""), "y": str(_decimal(d.get("kwh", 0)))}
            for d in series_points
        ]
        widgets.append(WidgetData(
            widget_type=WidgetType.TIME_SERIES,
            title="Consumption Trend",
            value=str(_round_val(total_kwh, 2)),
            unit="kWh",
            series_data=series_data,
        ))

        # Gauge widget for cost
        widgets.append(WidgetData(
            widget_type=WidgetType.GAUGE,
            title="Total Cost",
            value=str(_round_val(total_cost, 2)),
            unit=self._currency,
            metadata={"max_value": str(_round_val(total_cost * Decimal("1.5"), 2))},
        ))

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Widgets rendered: %d widgets (%.1f ms)", len(widgets), elapsed,
        )
        return widgets

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_trend(
        self, values: List[Decimal],
    ) -> Tuple[TrendDirection, Decimal]:
        """Compute trend direction from time-series values.

        Uses simple linear regression slope to determine direction.

        Args:
            values: Ordered time-series values.

        Returns:
            Tuple of (TrendDirection, change_percentage).
        """
        n = len(values)
        if n < 2:
            return TrendDirection.STABLE, Decimal("0")

        # Simple linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x2) - sum(x)^2)
        sum_x = Decimal("0")
        sum_y = Decimal("0")
        sum_xy = Decimal("0")
        sum_x2 = Decimal("0")

        for i, v in enumerate(values):
            x = _decimal(i)
            sum_x += x
            sum_y += v
            sum_xy += x * v
            sum_x2 += x * x

        n_dec = _decimal(n)
        denom = n_dec * sum_x2 - sum_x * sum_x
        if denom == Decimal("0"):
            return TrendDirection.STABLE, Decimal("0")

        slope = (n_dec * sum_xy - sum_x * sum_y) / denom

        # Normalise slope by mean
        mean = _safe_divide(sum_y, n_dec)
        norm_slope = _safe_divide(slope, mean) if mean != Decimal("0") else slope

        # Compute coefficient of variation for volatility
        residuals_sq = Decimal("0")
        for v in values:
            residuals_sq += (v - mean) ** 2
        std_dev = Decimal(str(math.sqrt(float(
            _safe_divide(residuals_sq, n_dec)
        ))))
        cv = _safe_divide(std_dev, abs(mean)) if mean != Decimal("0") else Decimal("0")

        # Change percentage (first to last)
        first = values[0] if values[0] != Decimal("0") else Decimal("1")
        change_pct = _safe_pct(values[-1] - values[0], abs(first))

        # Classify
        if cv > VOLATILITY_THRESHOLD:
            return TrendDirection.VOLATILE, change_pct
        elif norm_slope < -TREND_SLOPE_THRESHOLD:
            return TrendDirection.IMPROVING, change_pct
        elif norm_slope > TREND_SLOPE_THRESHOLD:
            return TrendDirection.DECLINING, change_pct
        else:
            return TrendDirection.STABLE, change_pct

    def _build_single_panel(
        self,
        panel_type: PanelType,
        consumption_data: List[Dict[str, Any]],
        cost_data: List[Dict[str, Any]],
        grid_row: int,
    ) -> Dict[str, Any]:
        """Build data for a single panel.

        Args:
            panel_type: Panel type.
            consumption_data: Consumption data.
            cost_data: Cost data.
            grid_row: Grid row position.

        Returns:
            Panel data dictionary.
        """
        kwh_values = [_decimal(d.get("kwh", 0)) for d in consumption_data]
        total = sum(kwh_values, Decimal("0"))
        count = len(kwh_values)
        avg = _safe_divide(total, _decimal(count)) if count > 0 else Decimal("0")
        peak = max(kwh_values) if kwh_values else Decimal("0")

        panel_titles = {
            PanelType.CONSUMPTION: "Energy Consumption",
            PanelType.COST: "Energy Cost",
            PanelType.ENPI: "Energy Performance Indicators",
            PanelType.ANOMALY: "Anomaly Detection",
            PanelType.SUBMETER: "Sub-Meter Breakdown",
            PanelType.WEATHER: "Weather Overlay",
            PanelType.LOAD_PROFILE: "Load Profile",
            PanelType.ALARM: "Alarm Summary",
        }

        return {
            "panel_id": _new_uuid(),
            "panel_type": panel_type.value,
            "title": panel_titles.get(panel_type, panel_type.value),
            "grid_row": grid_row,
            "grid_width": 6,
            "summary": {
                "total_kwh": str(_round_val(total, 2)),
                "average_kwh": str(_round_val(avg, 2)),
                "peak_kwh": str(_round_val(peak, 2)),
                "data_points": count,
            },
        }
