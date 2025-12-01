# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Fuel Price Trends Visualization Module.

Comprehensive time-series visualization for fuel price history and analysis.
Supports multiple fuel types, moving averages, forecasts, and market event annotations.

Author: GreenLang Team
Version: 1.0.0
Standards: WCAG 2.1 Level AA, ISO 12647-2

Features:
- Multi-fuel price comparison charts
- Moving averages (7-day, 30-day, 90-day)
- Forecast overlay with confidence intervals
- Volatility indicators (Bollinger bands, ATR)
- Market event annotations
- Interactive range selection
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
import statistics
from decimal import Decimal, ROUND_HALF_UP

# Local imports
from .config import (
    ThemeConfig,
    ThemeMode,
    VisualizationConfig,
    ConfigFactory,
    FuelTypeColors,
    StatusColors,
    GradientScales,
    FontConfig,
    MarginConfig,
    LegendConfig,
    AnimationConfig,
    HoverConfig,
    ExportConfig,
    TimeSeriesConfig,
    AccessibilityConfig,
    ExportFormat,
    ChartType,
    get_default_config,
    get_fuel_color,
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

class TimeRange(Enum):
    """Predefined time ranges for chart display."""
    ONE_WEEK = "1W"
    ONE_MONTH = "1M"
    THREE_MONTHS = "3M"
    SIX_MONTHS = "6M"
    ONE_YEAR = "1Y"
    TWO_YEARS = "2Y"
    FIVE_YEARS = "5Y"
    ALL = "ALL"
    CUSTOM = "CUSTOM"


class ChartStyle(Enum):
    """Chart display styles."""
    LINE = "line"
    AREA = "area"
    CANDLESTICK = "candlestick"
    OHLC = "ohlc"
    STEP = "step"
    SCATTER = "scatter"


class MovingAverageType(Enum):
    """Moving average calculation types."""
    SIMPLE = "sma"
    EXPONENTIAL = "ema"
    WEIGHTED = "wma"
    HULL = "hma"


class VolatilityIndicator(Enum):
    """Volatility indicator types."""
    BOLLINGER_BANDS = "bollinger"
    ATR = "atr"  # Average True Range
    STANDARD_DEVIATION = "std_dev"
    VARIANCE = "variance"
    HISTORICAL_VOLATILITY = "hist_vol"


class ForecastMethod(Enum):
    """Forecast calculation methods."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    MOVING_AVERAGE = "moving_average"
    ARIMA = "arima"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


class PriceUnit(Enum):
    """Price units for different fuels."""
    USD_PER_MMBTU = "$/MMBtu"
    USD_PER_TONNE = "$/tonne"
    USD_PER_BARREL = "$/bbl"
    USD_PER_GJ = "$/GJ"
    USD_PER_MWH = "$/MWh"
    USD_PER_GAL = "$/gal"
    USD_PER_KG = "$/kg"
    EUR_PER_MWH = "EUR/MWh"
    EUR_PER_TONNE = "EUR/tonne"


class MarketEventType(Enum):
    """Types of market events for annotations."""
    PRICE_SPIKE = "price_spike"
    PRICE_DROP = "price_drop"
    SUPPLY_DISRUPTION = "supply_disruption"
    DEMAND_SURGE = "demand_surge"
    REGULATORY_CHANGE = "regulatory_change"
    GEOPOLITICAL = "geopolitical"
    WEATHER = "weather"
    SEASONAL = "seasonal"
    CONTRACT_EXPIRY = "contract_expiry"
    NEWS = "news"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PriceDataPoint:
    """Single price data point."""
    timestamp: str
    price: float
    volume: Optional[float] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    estimated: bool = False
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ohlc(self) -> Tuple[float, float, float, float]:
        """Get OHLC tuple."""
        return (
            self.open_price or self.price,
            self.high_price or self.price,
            self.low_price or self.price,
            self.close_price or self.price,
        )


@dataclass
class FuelPriceSeries:
    """Time series of fuel prices."""
    fuel_id: str
    fuel_name: str
    fuel_type: str
    data_points: List[PriceDataPoint]
    unit: PriceUnit = PriceUnit.USD_PER_GJ
    currency: str = "USD"
    source: Optional[str] = None
    update_frequency: str = "daily"
    last_updated: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate statistics."""
        self._calculate_statistics()

    def _calculate_statistics(self):
        """Calculate price statistics."""
        if not self.data_points:
            return

        prices = [dp.price for dp in self.data_points]
        self.min_price = min(prices)
        self.max_price = max(prices)
        self.avg_price = statistics.mean(prices)
        self.std_dev = statistics.stdev(prices) if len(prices) > 1 else 0
        self.current_price = self.data_points[-1].price
        self.first_price = self.data_points[0].price
        self.price_change = self.current_price - self.first_price
        self.price_change_percent = (self.price_change / self.first_price * 100) if self.first_price != 0 else 0

    @property
    def color(self) -> str:
        """Get color for this fuel type."""
        return get_fuel_color(self.fuel_type)

    def get_prices(self) -> List[float]:
        """Get list of prices."""
        return [dp.price for dp in self.data_points]

    def get_timestamps(self) -> List[str]:
        """Get list of timestamps."""
        return [dp.timestamp for dp in self.data_points]

    def get_slice(self, start_date: str, end_date: str) -> "FuelPriceSeries":
        """Get a slice of the data within date range."""
        filtered = [
            dp for dp in self.data_points
            if start_date <= dp.timestamp <= end_date
        ]
        return FuelPriceSeries(
            fuel_id=self.fuel_id,
            fuel_name=self.fuel_name,
            fuel_type=self.fuel_type,
            data_points=filtered,
            unit=self.unit,
            currency=self.currency,
            source=self.source,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fuel_id": self.fuel_id,
            "fuel_name": self.fuel_name,
            "fuel_type": self.fuel_type,
            "data_points": [{"timestamp": dp.timestamp, "price": dp.price} for dp in self.data_points],
            "unit": self.unit.value,
            "currency": self.currency,
            "statistics": {
                "min": getattr(self, 'min_price', None),
                "max": getattr(self, 'max_price', None),
                "avg": getattr(self, 'avg_price', None),
                "std_dev": getattr(self, 'std_dev', None),
                "current": getattr(self, 'current_price', None),
                "change": getattr(self, 'price_change', None),
                "change_percent": getattr(self, 'price_change_percent', None),
            },
        }


@dataclass
class MovingAverageConfig:
    """Configuration for moving average calculation."""
    window: int
    ma_type: MovingAverageType = MovingAverageType.SIMPLE
    color: Optional[str] = None
    line_dash: str = "solid"
    line_width: float = 1.5
    visible: bool = True
    label: Optional[str] = None

    def __post_init__(self):
        """Set default label."""
        if self.label is None:
            self.label = f"{self.window}-Day {self.ma_type.value.upper()}"


@dataclass
class ForecastConfig:
    """Configuration for price forecast."""
    method: ForecastMethod = ForecastMethod.LINEAR
    periods: int = 30
    confidence_level: float = 0.95
    show_confidence_interval: bool = True
    confidence_fill_color: str = "rgba(100, 100, 100, 0.2)"
    line_color: str = "#9B59B6"
    line_dash: str = "dash"


@dataclass
class VolatilityConfig:
    """Configuration for volatility indicator."""
    indicator_type: VolatilityIndicator
    window: int = 20
    num_std_dev: float = 2.0  # For Bollinger bands
    fill_color: str = "rgba(150, 150, 150, 0.2)"
    upper_line_color: str = "#E74C3C"
    lower_line_color: str = "#2ECC71"
    visible: bool = True


@dataclass
class MarketEvent:
    """Market event for annotation."""
    event_id: str
    event_type: MarketEventType
    timestamp: str
    title: str
    description: Optional[str] = None
    impact: str = "neutral"  # positive, negative, neutral
    price_at_event: Optional[float] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def color(self) -> str:
        """Get color based on impact."""
        colors = {
            "positive": StatusColors.SUCCESS,
            "negative": StatusColors.ERROR,
            "neutral": StatusColors.INFO,
        }
        return colors.get(self.impact, StatusColors.NEUTRAL)

    @property
    def symbol(self) -> str:
        """Get symbol based on event type."""
        symbols = {
            MarketEventType.PRICE_SPIKE: "triangle-up",
            MarketEventType.PRICE_DROP: "triangle-down",
            MarketEventType.SUPPLY_DISRUPTION: "x",
            MarketEventType.DEMAND_SURGE: "circle",
            MarketEventType.REGULATORY_CHANGE: "square",
            MarketEventType.GEOPOLITICAL: "diamond",
            MarketEventType.WEATHER: "star",
            MarketEventType.SEASONAL: "hexagon",
            MarketEventType.CONTRACT_EXPIRY: "cross",
            MarketEventType.NEWS: "asterisk",
        }
        return symbols.get(self.event_type, "circle")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "title": self.title,
            "description": self.description,
            "impact": self.impact,
            "price_at_event": self.price_at_event,
            "source": self.source,
        }


@dataclass
class PriceTrendData:
    """Complete price trend data structure."""
    fuel_series: List[FuelPriceSeries]
    market_events: List[MarketEvent] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    base_currency: str = "USD"
    provenance_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate provenance and date range."""
        self._calculate_provenance()
        self._calculate_date_range()

    def _calculate_provenance(self):
        """Calculate provenance hash."""
        data = {
            "fuels": [(s.fuel_id, len(s.data_points)) for s in self.fuel_series],
            "events": len(self.market_events),
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _calculate_date_range(self):
        """Calculate overall date range."""
        all_timestamps = []
        for series in self.fuel_series:
            all_timestamps.extend(series.get_timestamps())
        if all_timestamps:
            self.start_date = self.start_date or min(all_timestamps)
            self.end_date = self.end_date or max(all_timestamps)

    def get_series_by_fuel(self, fuel_id: str) -> Optional[FuelPriceSeries]:
        """Get series by fuel ID."""
        for series in self.fuel_series:
            if series.fuel_id == fuel_id:
                return series
        return None

    def get_events_in_range(self, start_date: str, end_date: str) -> List[MarketEvent]:
        """Get events within date range."""
        return [
            e for e in self.market_events
            if start_date <= e.timestamp <= end_date
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fuel_series": [s.to_dict() for s in self.fuel_series],
            "market_events": [e.to_dict() for e in self.market_events],
            "start_date": self.start_date,
            "end_date": self.end_date,
            "base_currency": self.base_currency,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
        }


# =============================================================================
# CHART OPTIONS
# =============================================================================

@dataclass
class PriceTrendOptions:
    """Configuration options for price trend charts."""
    # Display options
    title: str = "Fuel Price Trends"
    subtitle: Optional[str] = None
    chart_style: ChartStyle = ChartStyle.LINE
    show_all_fuels: bool = True
    selected_fuels: Optional[List[str]] = None

    # Time range
    time_range: TimeRange = TimeRange.ONE_YEAR
    custom_start_date: Optional[str] = None
    custom_end_date: Optional[str] = None
    show_range_selector: bool = True
    show_range_slider: bool = True

    # Moving averages
    moving_averages: List[MovingAverageConfig] = field(default_factory=lambda: [
        MovingAverageConfig(window=7, color="#F39C12", line_dash="dot"),
        MovingAverageConfig(window=30, color="#3498DB", line_dash="dash"),
        MovingAverageConfig(window=90, color="#9B59B6", line_dash="dashdot"),
    ])

    # Forecast
    show_forecast: bool = False
    forecast_config: Optional[ForecastConfig] = None

    # Volatility
    show_volatility: bool = False
    volatility_config: Optional[VolatilityConfig] = None

    # Market events
    show_market_events: bool = True
    event_types_filter: Optional[List[MarketEventType]] = None

    # Styling
    line_width: float = 2.0
    marker_size: int = 4
    show_markers: bool = False
    fill_area: bool = False
    fill_opacity: float = 0.2
    color_blind_safe: bool = False

    # Axis options
    y_axis_title: str = "Price"
    x_axis_title: str = "Date"
    show_grid: bool = True
    show_zero_line: bool = False
    y_axis_type: str = "linear"  # linear or log

    # Legend options
    show_legend: bool = True
    legend_position: str = "top"

    # Interaction options
    enable_hover: bool = True
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_crosshair: bool = True

    # Size options
    width: Optional[int] = None
    height: Optional[int] = None
    auto_size: bool = True

    # Animation options
    animate: bool = True
    animation_duration: int = 500

    # Additional options
    show_statistics: bool = True
    show_current_price: bool = True
    show_price_change: bool = True
    normalize_prices: bool = False
    compare_to_baseline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "chart_style": self.chart_style.value,
            "time_range": self.time_range.value,
            "show_forecast": self.show_forecast,
            "show_volatility": self.show_volatility,
            "show_market_events": self.show_market_events,
        }


# =============================================================================
# PRICE TREND ENGINE
# =============================================================================

class PriceTrendEngine:
    """
    Engine for generating fuel price trend visualizations.

    Supports multiple fuel types, moving averages, forecasts, and market events.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        theme: Optional[ThemeConfig] = None,
    ):
        """
        Initialize price trend engine.

        Args:
            config: Global visualization configuration
            theme: Theme configuration for styling
        """
        self.config = config or get_default_config()
        self.theme = theme or self.config.theme
        self._cache: Dict[str, Any] = {}

    def generate(
        self,
        data: PriceTrendData,
        options: Optional[PriceTrendOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate price trend visualization.

        Args:
            data: Price trend data
            options: Chart configuration options

        Returns:
            Plotly-compatible chart specification
        """
        options = options or PriceTrendOptions()

        # Check cache
        cache_key = self._get_cache_key(data, options)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Filter fuel series
        fuel_series = self._filter_fuel_series(data, options)

        # Build traces
        traces = []

        for series in fuel_series:
            # Main price trace
            main_trace = self._build_price_trace(series, options)
            traces.append(main_trace)

            # Moving average traces
            for ma_config in options.moving_averages:
                if ma_config.visible:
                    ma_trace = self._build_moving_average_trace(series, ma_config, options)
                    if ma_trace:
                        traces.append(ma_trace)

        # Add volatility bands
        if options.show_volatility and options.volatility_config:
            for series in fuel_series:
                vol_traces = self._build_volatility_traces(series, options.volatility_config)
                traces.extend(vol_traces)

        # Add forecast
        if options.show_forecast and options.forecast_config:
            for series in fuel_series:
                forecast_traces = self._build_forecast_traces(series, options.forecast_config)
                traces.extend(forecast_traces)

        # Build layout
        layout = self._build_layout(data, options)

        # Add market event annotations
        if options.show_market_events and data.market_events:
            layout = self._add_market_event_annotations(layout, data, options)

        # Add statistics annotation
        if options.show_statistics:
            layout = self._add_statistics_annotation(layout, fuel_series, options)

        chart = {
            "data": traces,
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

        # Cache result
        self._cache[cache_key] = chart

        return chart

    def _get_cache_key(
        self,
        data: PriceTrendData,
        options: PriceTrendOptions,
    ) -> str:
        """Generate cache key."""
        key_data = {
            "provenance": data.provenance_hash,
            "style": options.chart_style.value,
            "range": options.time_range.value,
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _filter_fuel_series(
        self,
        data: PriceTrendData,
        options: PriceTrendOptions,
    ) -> List[FuelPriceSeries]:
        """Filter fuel series based on options."""
        if options.show_all_fuels:
            return data.fuel_series
        if options.selected_fuels:
            return [s for s in data.fuel_series if s.fuel_id in options.selected_fuels]
        return data.fuel_series

    def _build_price_trace(
        self,
        series: FuelPriceSeries,
        options: PriceTrendOptions,
    ) -> Dict[str, Any]:
        """Build main price trace."""
        timestamps = series.get_timestamps()
        prices = series.get_prices()

        # Normalize if requested
        if options.normalize_prices:
            base_price = prices[0] if prices else 1
            prices = [p / base_price * 100 for p in prices]

        trace = {
            "name": series.fuel_name,
            "x": timestamps,
            "y": prices,
            "line": {
                "color": series.color,
                "width": options.line_width,
            },
            "hovertemplate": (
                f"<b>{series.fuel_name}</b><br>"
                f"Date: %{{x}}<br>"
                f"Price: %{{y:.2f}} {series.unit.value}<extra></extra>"
            ),
        }

        # Set trace type based on style
        if options.chart_style == ChartStyle.LINE:
            trace["type"] = "scatter"
            trace["mode"] = "lines+markers" if options.show_markers else "lines"
            if options.show_markers:
                trace["marker"] = {"size": options.marker_size}
        elif options.chart_style == ChartStyle.AREA:
            trace["type"] = "scatter"
            trace["mode"] = "lines"
            trace["fill"] = "tozeroy"
            trace["fillcolor"] = hex_to_rgba(series.color, options.fill_opacity)
        elif options.chart_style == ChartStyle.STEP:
            trace["type"] = "scatter"
            trace["mode"] = "lines"
            trace["line"]["shape"] = "hv"
        elif options.chart_style == ChartStyle.SCATTER:
            trace["type"] = "scatter"
            trace["mode"] = "markers"
            trace["marker"] = {"size": options.marker_size, "color": series.color}
        elif options.chart_style == ChartStyle.CANDLESTICK:
            trace = self._build_candlestick_trace(series, options)
        elif options.chart_style == ChartStyle.OHLC:
            trace = self._build_ohlc_trace(series, options)

        return trace

    def _build_candlestick_trace(
        self,
        series: FuelPriceSeries,
        options: PriceTrendOptions,
    ) -> Dict[str, Any]:
        """Build candlestick trace."""
        timestamps = series.get_timestamps()
        opens = [dp.open_price or dp.price for dp in series.data_points]
        highs = [dp.high_price or dp.price for dp in series.data_points]
        lows = [dp.low_price or dp.price for dp in series.data_points]
        closes = [dp.close_price or dp.price for dp in series.data_points]

        return {
            "type": "candlestick",
            "name": series.fuel_name,
            "x": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "increasing": {"line": {"color": StatusColors.SUCCESS}},
            "decreasing": {"line": {"color": StatusColors.ERROR}},
        }

    def _build_ohlc_trace(
        self,
        series: FuelPriceSeries,
        options: PriceTrendOptions,
    ) -> Dict[str, Any]:
        """Build OHLC trace."""
        trace = self._build_candlestick_trace(series, options)
        trace["type"] = "ohlc"
        return trace

    def _build_moving_average_trace(
        self,
        series: FuelPriceSeries,
        ma_config: MovingAverageConfig,
        options: PriceTrendOptions,
    ) -> Optional[Dict[str, Any]]:
        """Build moving average trace."""
        prices = series.get_prices()
        if len(prices) < ma_config.window:
            return None

        ma_values = self._calculate_moving_average(prices, ma_config)
        timestamps = series.get_timestamps()[ma_config.window - 1:]

        return {
            "type": "scatter",
            "mode": "lines",
            "name": f"{series.fuel_name} {ma_config.label}",
            "x": timestamps,
            "y": ma_values,
            "line": {
                "color": ma_config.color or adjust_color_brightness(series.color, 0.7),
                "width": ma_config.line_width,
                "dash": ma_config.line_dash,
            },
            "hovertemplate": f"<b>{ma_config.label}</b><br>%{{y:.2f}}<extra></extra>",
            "showlegend": True,
        }

    def _calculate_moving_average(
        self,
        prices: List[float],
        config: MovingAverageConfig,
    ) -> List[float]:
        """Calculate moving average based on type."""
        if config.ma_type == MovingAverageType.SIMPLE:
            return self._calculate_sma(prices, config.window)
        elif config.ma_type == MovingAverageType.EXPONENTIAL:
            return self._calculate_ema(prices, config.window)
        elif config.ma_type == MovingAverageType.WEIGHTED:
            return self._calculate_wma(prices, config.window)
        else:
            return self._calculate_sma(prices, config.window)

    def _calculate_sma(self, prices: List[float], window: int) -> List[float]:
        """Calculate Simple Moving Average."""
        sma = []
        for i in range(window - 1, len(prices)):
            avg = sum(prices[i - window + 1:i + 1]) / window
            sma.append(avg)
        return sma

    def _calculate_ema(self, prices: List[float], window: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        multiplier = 2 / (window + 1)
        ema = [sum(prices[:window]) / window]  # Start with SMA

        for price in prices[window:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])

        return ema

    def _calculate_wma(self, prices: List[float], window: int) -> List[float]:
        """Calculate Weighted Moving Average."""
        wma = []
        weights = list(range(1, window + 1))
        weight_sum = sum(weights)

        for i in range(window - 1, len(prices)):
            window_prices = prices[i - window + 1:i + 1]
            weighted_sum = sum(p * w for p, w in zip(window_prices, weights))
            wma.append(weighted_sum / weight_sum)

        return wma

    def _build_volatility_traces(
        self,
        series: FuelPriceSeries,
        config: VolatilityConfig,
    ) -> List[Dict[str, Any]]:
        """Build volatility indicator traces."""
        traces = []
        prices = series.get_prices()
        timestamps = series.get_timestamps()

        if config.indicator_type == VolatilityIndicator.BOLLINGER_BANDS:
            traces = self._build_bollinger_bands(series, config)
        elif config.indicator_type == VolatilityIndicator.STANDARD_DEVIATION:
            traces = self._build_std_dev_bands(series, config)

        return traces

    def _build_bollinger_bands(
        self,
        series: FuelPriceSeries,
        config: VolatilityConfig,
    ) -> List[Dict[str, Any]]:
        """Build Bollinger Bands traces."""
        prices = series.get_prices()
        timestamps = series.get_timestamps()
        window = config.window
        num_std = config.num_std_dev

        if len(prices) < window:
            return []

        # Calculate middle band (SMA)
        middle = self._calculate_sma(prices, window)

        # Calculate standard deviation and bands
        upper = []
        lower = []

        for i in range(window - 1, len(prices)):
            window_prices = prices[i - window + 1:i + 1]
            std = statistics.stdev(window_prices)
            mid = middle[i - window + 1]
            upper.append(mid + num_std * std)
            lower.append(mid - num_std * std)

        adjusted_timestamps = timestamps[window - 1:]

        traces = [
            # Upper band
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"{series.fuel_name} Upper BB",
                "x": adjusted_timestamps,
                "y": upper,
                "line": {"color": config.upper_line_color, "width": 1, "dash": "dot"},
                "showlegend": False,
            },
            # Lower band
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"{series.fuel_name} Lower BB",
                "x": adjusted_timestamps,
                "y": lower,
                "line": {"color": config.lower_line_color, "width": 1, "dash": "dot"},
                "fill": "tonexty",
                "fillcolor": config.fill_color,
                "showlegend": False,
            },
        ]

        return traces

    def _build_std_dev_bands(
        self,
        series: FuelPriceSeries,
        config: VolatilityConfig,
    ) -> List[Dict[str, Any]]:
        """Build standard deviation bands."""
        return self._build_bollinger_bands(series, config)

    def _build_forecast_traces(
        self,
        series: FuelPriceSeries,
        config: ForecastConfig,
    ) -> List[Dict[str, Any]]:
        """Build forecast traces."""
        traces = []
        prices = series.get_prices()
        timestamps = series.get_timestamps()

        if len(prices) < 10:
            return traces

        # Generate forecast
        forecast_values, conf_lower, conf_upper = self._generate_forecast(
            prices, config
        )

        # Generate forecast timestamps
        last_timestamp = datetime.fromisoformat(timestamps[-1].replace('Z', '+00:00'))
        forecast_timestamps = [
            (last_timestamp + timedelta(days=i + 1)).isoformat()
            for i in range(config.periods)
        ]

        # Forecast line
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": f"{series.fuel_name} Forecast",
            "x": [timestamps[-1]] + forecast_timestamps,
            "y": [prices[-1]] + forecast_values,
            "line": {
                "color": config.line_color,
                "width": 2,
                "dash": config.line_dash,
            },
        })

        # Confidence interval
        if config.show_confidence_interval:
            # Upper bound
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "name": f"{series.fuel_name} Conf Upper",
                "x": forecast_timestamps,
                "y": conf_upper,
                "line": {"width": 0},
                "showlegend": False,
            })
            # Lower bound with fill
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "name": f"{series.fuel_name} Conf Lower",
                "x": forecast_timestamps,
                "y": conf_lower,
                "line": {"width": 0},
                "fill": "tonexty",
                "fillcolor": config.confidence_fill_color,
                "showlegend": False,
            })

        return traces

    def _generate_forecast(
        self,
        prices: List[float],
        config: ForecastConfig,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Generate price forecast with confidence intervals."""
        if config.method == ForecastMethod.LINEAR:
            return self._linear_forecast(prices, config.periods, config.confidence_level)
        elif config.method == ForecastMethod.MOVING_AVERAGE:
            return self._ma_forecast(prices, config.periods, config.confidence_level)
        else:
            return self._linear_forecast(prices, config.periods, config.confidence_level)

    def _linear_forecast(
        self,
        prices: List[float],
        periods: int,
        confidence: float,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Generate linear regression forecast."""
        n = len(prices)
        x = list(range(n))
        y = prices

        # Calculate linear regression
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean

        # Generate forecast
        forecast = [slope * (n + i) + intercept for i in range(periods)]

        # Calculate standard error
        residuals = [yi - (slope * xi + intercept) for xi, yi in zip(x, y)]
        std_error = statistics.stdev(residuals) if len(residuals) > 1 else 0

        # Confidence intervals
        z_score = 1.96 if confidence >= 0.95 else 1.645
        conf_upper = [f + z_score * std_error * (1 + i / n) for i, f in enumerate(forecast)]
        conf_lower = [f - z_score * std_error * (1 + i / n) for i, f in enumerate(forecast)]

        return forecast, conf_lower, conf_upper

    def _ma_forecast(
        self,
        prices: List[float],
        periods: int,
        confidence: float,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Generate moving average forecast."""
        window = min(30, len(prices) // 2)
        ma = sum(prices[-window:]) / window

        # Simple constant forecast at MA level
        forecast = [ma] * periods

        # Calculate volatility for confidence intervals
        std = statistics.stdev(prices[-window:]) if len(prices) >= window else 0
        z_score = 1.96 if confidence >= 0.95 else 1.645

        conf_upper = [f + z_score * std * math.sqrt(1 + i / window) for i, f in enumerate(forecast)]
        conf_lower = [f - z_score * std * math.sqrt(1 + i / window) for i, f in enumerate(forecast)]

        return forecast, conf_lower, conf_upper

    def _build_layout(
        self,
        data: PriceTrendData,
        options: PriceTrendOptions,
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

        # X-axis
        layout["xaxis"] = {
            "title": options.x_axis_title,
            "showgrid": options.show_grid,
            "gridcolor": self.theme.grid_line_color,
            "rangeslider": {"visible": options.show_range_slider},
        }

        if options.show_range_selector:
            layout["xaxis"]["rangeselector"] = {
                "buttons": [
                    {"count": 7, "label": "1W", "step": "day", "stepmode": "backward"},
                    {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                    {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                    {"label": "All", "step": "all"},
                ],
            }

        # Y-axis
        layout["yaxis"] = {
            "title": options.y_axis_title,
            "showgrid": options.show_grid,
            "gridcolor": self.theme.grid_line_color,
            "type": options.y_axis_type,
        }

        if options.show_zero_line:
            layout["yaxis"]["zeroline"] = True
            layout["yaxis"]["zerolinecolor"] = self.theme.axis_line_color

        # Legend
        if options.show_legend:
            layout["showlegend"] = True
            if options.legend_position == "top":
                layout["legend"] = {
                    "orientation": "h",
                    "x": 0.5,
                    "y": 1.1,
                    "xanchor": "center",
                    "yanchor": "bottom",
                }
            elif options.legend_position == "bottom":
                layout["legend"] = {
                    "orientation": "h",
                    "x": 0.5,
                    "y": -0.15,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            else:
                layout["legend"] = {
                    "x": 1.02,
                    "y": 1,
                    "xanchor": "left",
                    "yanchor": "top",
                }
        else:
            layout["showlegend"] = False

        # Crosshair/spike lines
        if options.enable_crosshair:
            layout["xaxis"]["showspikes"] = True
            layout["xaxis"]["spikemode"] = "across"
            layout["xaxis"]["spikethickness"] = 1
            layout["yaxis"]["showspikes"] = True
            layout["yaxis"]["spikemode"] = "across"
            layout["yaxis"]["spikethickness"] = 1

        # Size
        if options.width:
            layout["width"] = options.width
        if options.height:
            layout["height"] = options.height
        if options.auto_size:
            layout["autosize"] = True

        # Hover mode
        layout["hovermode"] = "x unified"

        return layout

    def _add_market_event_annotations(
        self,
        layout: Dict[str, Any],
        data: PriceTrendData,
        options: PriceTrendOptions,
    ) -> Dict[str, Any]:
        """Add market event annotations to layout."""
        annotations = layout.get("annotations", [])
        shapes = layout.get("shapes", [])

        events = data.market_events
        if options.event_types_filter:
            events = [e for e in events if e.event_type in options.event_types_filter]

        for event in events:
            # Add vertical line
            shapes.append({
                "type": "line",
                "x0": event.timestamp,
                "x1": event.timestamp,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {
                    "color": event.color,
                    "width": 1,
                    "dash": "dot",
                },
            })

            # Add annotation
            annotations.append({
                "x": event.timestamp,
                "y": 1,
                "yref": "paper",
                "text": event.title,
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 1,
                "arrowwidth": 1,
                "arrowcolor": event.color,
                "ax": 0,
                "ay": -30,
                "font": {"size": 9, "color": event.color},
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": event.color,
                "borderwidth": 1,
            })

        layout["annotations"] = annotations
        layout["shapes"] = shapes

        return layout

    def _add_statistics_annotation(
        self,
        layout: Dict[str, Any],
        fuel_series: List[FuelPriceSeries],
        options: PriceTrendOptions,
    ) -> Dict[str, Any]:
        """Add price statistics annotation."""
        annotations = layout.get("annotations", [])

        if len(fuel_series) == 1:
            series = fuel_series[0]
            stats_text = (
                f"Current: {series.current_price:.2f}<br>"
                f"Min: {series.min_price:.2f}<br>"
                f"Max: {series.max_price:.2f}<br>"
                f"Avg: {series.avg_price:.2f}<br>"
                f"Change: {series.price_change:+.2f} ({series.price_change_percent:+.1f}%)"
            )

            annotations.append({
                "text": stats_text,
                "x": 1,
                "y": 1,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "right",
                "yanchor": "top",
                "showarrow": False,
                "font": {"size": 10},
                "bgcolor": "rgba(255,255,255,0.9)",
                "bordercolor": series.color,
                "borderwidth": 1,
                "borderpad": 4,
            })

        layout["annotations"] = annotations
        return layout

    def generate_comparison_chart(
        self,
        data: PriceTrendData,
        options: Optional[PriceTrendOptions] = None,
    ) -> Dict[str, Any]:
        """Generate multi-fuel comparison chart."""
        options = options or PriceTrendOptions()
        options.title = options.title or "Fuel Price Comparison"
        options.normalize_prices = True

        return self.generate(data, options)

    def generate_volatility_chart(
        self,
        series: FuelPriceSeries,
        options: Optional[PriceTrendOptions] = None,
    ) -> Dict[str, Any]:
        """Generate volatility analysis chart."""
        options = options or PriceTrendOptions()
        options.title = f"{series.fuel_name} Price Volatility"
        options.show_volatility = True
        options.volatility_config = VolatilityConfig(
            indicator_type=VolatilityIndicator.BOLLINGER_BANDS,
            window=20,
            num_std_dev=2.0,
        )

        data = PriceTrendData(fuel_series=[series])
        return self.generate(data, options)

    def generate_forecast_chart(
        self,
        series: FuelPriceSeries,
        forecast_periods: int = 30,
        options: Optional[PriceTrendOptions] = None,
    ) -> Dict[str, Any]:
        """Generate price forecast chart."""
        options = options or PriceTrendOptions()
        options.title = f"{series.fuel_name} Price Forecast"
        options.show_forecast = True
        options.forecast_config = ForecastConfig(
            method=ForecastMethod.LINEAR,
            periods=forecast_periods,
            confidence_level=0.95,
        )

        data = PriceTrendData(fuel_series=[series])
        return self.generate(data, options)

    def to_json(self, chart: Dict[str, Any]) -> str:
        """Export chart to JSON string."""
        return json.dumps(chart, indent=2, default=str)

    def clear_cache(self) -> None:
        """Clear the chart cache."""
        self._cache.clear()


# =============================================================================
# SPECIALIZED GENERATORS
# =============================================================================

class SpotPriceTracker(PriceTrendEngine):
    """Specialized tracker for spot prices."""

    def generate_spot_tracker(
        self,
        data: PriceTrendData,
        options: Optional[PriceTrendOptions] = None,
    ) -> Dict[str, Any]:
        """Generate real-time spot price tracker."""
        options = options or PriceTrendOptions()
        options.title = "Real-Time Spot Prices"
        options.time_range = TimeRange.ONE_WEEK
        options.show_current_price = True
        options.show_markers = True
        options.chart_style = ChartStyle.LINE

        return self.generate(data, options)


class ForwardCurveAnalyzer(PriceTrendEngine):
    """Analyzer for forward price curves."""

    def generate_forward_curve(
        self,
        current_prices: Dict[str, float],
        forward_prices: Dict[str, Dict[str, float]],  # fuel -> {date -> price}
        options: Optional[PriceTrendOptions] = None,
    ) -> Dict[str, Any]:
        """Generate forward curve visualization."""
        options = options or PriceTrendOptions()
        options.title = "Forward Price Curves"

        traces = []

        for fuel_id, prices in forward_prices.items():
            dates = sorted(prices.keys())
            values = [prices[d] for d in dates]

            traces.append({
                "type": "scatter",
                "mode": "lines+markers",
                "name": fuel_id.replace("_", " ").title(),
                "x": dates,
                "y": values,
                "line": {"color": get_fuel_color(fuel_id), "width": 2},
                "marker": {"size": 6},
            })

            # Add current spot price marker
            if fuel_id in current_prices:
                traces.append({
                    "type": "scatter",
                    "mode": "markers",
                    "name": f"{fuel_id} Spot",
                    "x": [dates[0]],
                    "y": [current_prices[fuel_id]],
                    "marker": {
                        "size": 12,
                        "color": get_fuel_color(fuel_id),
                        "symbol": "star",
                    },
                    "showlegend": False,
                })

        layout = self._build_layout(PriceTrendData(fuel_series=[]), options)
        layout["xaxis"]["title"] = "Delivery Date"
        layout["yaxis"]["title"] = "Price"

        return {
            "data": traces,
            "layout": layout,
            "config": get_plotly_config(interactive=True),
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def create_sample_price_data() -> PriceTrendData:
    """Create sample price data for demonstration."""
    import random

    # Generate 365 days of sample data
    base_prices = {
        "natural_gas": 5.0,
        "coal": 80.0,
        "oil": 70.0,
        "biomass": 120.0,
    }

    fuel_series = []

    for fuel_id, base_price in base_prices.items():
        data_points = []
        current_price = base_price

        for i in range(365):
            date = (datetime.now() - timedelta(days=365 - i)).strftime("%Y-%m-%d")
            # Add some randomness and trend
            change = random.gauss(0, base_price * 0.02)
            trend = math.sin(i / 50) * base_price * 0.1
            current_price = max(base_price * 0.5, current_price + change + trend * 0.01)

            data_points.append(PriceDataPoint(
                timestamp=date,
                price=round(current_price, 2),
            ))

        fuel_series.append(FuelPriceSeries(
            fuel_id=fuel_id,
            fuel_name=fuel_id.replace("_", " ").title(),
            fuel_type=fuel_id,
            data_points=data_points,
            unit=PriceUnit.USD_PER_GJ,
        ))

    # Add some market events
    events = [
        MarketEvent(
            event_id="evt_001",
            event_type=MarketEventType.SUPPLY_DISRUPTION,
            timestamp=(datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d"),
            title="Pipeline Maintenance",
            description="Major pipeline maintenance affected supply",
            impact="negative",
        ),
        MarketEvent(
            event_id="evt_002",
            event_type=MarketEventType.REGULATORY_CHANGE,
            timestamp=(datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d"),
            title="Carbon Tax Increase",
            description="New carbon tax regulations implemented",
            impact="negative",
        ),
        MarketEvent(
            event_id="evt_003",
            event_type=MarketEventType.WEATHER,
            timestamp=(datetime.now() - timedelta(days=50)).strftime("%Y-%m-%d"),
            title="Mild Winter",
            description="Unseasonably warm weather reduced demand",
            impact="positive",
        ),
    ]

    return PriceTrendData(
        fuel_series=fuel_series,
        market_events=events,
    )


def example_multi_fuel_trend():
    """Example: Generate multi-fuel trend chart."""
    print("Generating multi-fuel trend chart...")

    data = create_sample_price_data()
    engine = PriceTrendEngine()
    options = PriceTrendOptions(
        title="Fuel Price Trends - Last 12 Months",
        show_market_events=True,
    )

    chart = engine.generate(data, options)
    print(f"Multi-fuel chart generated with {len(chart['data'])} traces")
    return chart


def example_volatility_analysis():
    """Example: Generate volatility analysis chart."""
    print("Generating volatility analysis chart...")

    data = create_sample_price_data()
    series = data.fuel_series[0]  # Natural gas
    engine = PriceTrendEngine()

    chart = engine.generate_volatility_chart(series)
    print(f"Volatility chart generated")
    return chart


def example_price_forecast():
    """Example: Generate price forecast chart."""
    print("Generating price forecast chart...")

    data = create_sample_price_data()
    series = data.fuel_series[0]  # Natural gas
    engine = PriceTrendEngine()

    chart = engine.generate_forecast_chart(series, forecast_periods=60)
    print(f"Forecast chart generated")
    return chart


def example_comparison_chart():
    """Example: Generate normalized comparison chart."""
    print("Generating comparison chart...")

    data = create_sample_price_data()
    engine = PriceTrendEngine()

    chart = engine.generate_comparison_chart(data)
    print(f"Comparison chart generated")
    return chart


def run_all_examples():
    """Run all price trend visualization examples."""
    print("=" * 60)
    print("GL-011 FUELCRAFT - Price Trend Visualization Examples")
    print("=" * 60)

    examples = [
        ("Multi-Fuel Trend", example_multi_fuel_trend),
        ("Volatility Analysis", example_volatility_analysis),
        ("Price Forecast", example_price_forecast),
        ("Comparison Chart", example_comparison_chart),
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
