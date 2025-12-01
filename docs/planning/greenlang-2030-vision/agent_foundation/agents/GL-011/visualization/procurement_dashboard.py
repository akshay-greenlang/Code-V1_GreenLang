# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Procurement Dashboard Visualization Module.

Comprehensive dashboard for fuel procurement KPIs including inventory levels,
contract utilization, supplier performance, and delivery reliability.

Author: GreenLang Team
Version: 1.0.0
Standards: WCAG 2.1 Level AA, ISO 12647-2

Features:
- Multi-panel KPI dashboard
- Inventory level gauges and trends
- Contract utilization tracking
- Supplier performance scorecards
- Delivery reliability metrics
- Alerts for low inventory and contract expiry
- Optimization recommendations
- Executive summary view
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

# Local imports
from .config import (
    ThemeConfig,
    ThemeMode,
    VisualizationConfig,
    ConfigFactory,
    DashboardConfig,
    FuelTypeColors,
    CostCategoryColors,
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
    get_status_color,
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

class KPIType(Enum):
    """Types of KPI metrics."""
    INVENTORY_LEVEL = "inventory_level"
    INVENTORY_DAYS = "inventory_days"
    CONTRACT_UTILIZATION = "contract_utilization"
    SUPPLIER_SCORE = "supplier_score"
    DELIVERY_RELIABILITY = "delivery_reliability"
    COST_VARIANCE = "cost_variance"
    QUALITY_SCORE = "quality_score"
    LEAD_TIME = "lead_time"
    ORDER_FILL_RATE = "order_fill_rate"
    PRICE_INDEX = "price_index"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

    @property
    def color(self) -> str:
        """Get color for severity."""
        colors = {
            "info": StatusColors.INFO,
            "warning": StatusColors.WARNING,
            "critical": StatusColors.ERROR,
            "emergency": "#8B0000",  # Dark red
        }
        return colors.get(self.value, StatusColors.NEUTRAL)


class AlertType(Enum):
    """Types of procurement alerts."""
    LOW_INVENTORY = "low_inventory"
    CONTRACT_EXPIRY = "contract_expiry"
    DELIVERY_DELAY = "delivery_delay"
    PRICE_SPIKE = "price_spike"
    QUALITY_ISSUE = "quality_issue"
    SUPPLIER_ISSUE = "supplier_issue"
    BUDGET_EXCEEDED = "budget_exceeded"
    REORDER_POINT = "reorder_point"


class TrendDirection(Enum):
    """Trend direction indicators."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    VOLATILE = "volatile"


class DashboardPanelType(Enum):
    """Types of dashboard panels."""
    KPI_CARD = "kpi_card"
    GAUGE = "gauge"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    TABLE = "table"
    HEATMAP = "heatmap"
    BULLET = "bullet"
    SPARKLINE = "sparkline"
    ALERT_LIST = "alert_list"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class InventoryLevel:
    """Inventory level for a fuel type."""
    fuel_id: str
    fuel_name: str
    fuel_type: str
    current_quantity: float
    max_capacity: float
    min_threshold: float  # Reorder point
    critical_threshold: float  # Emergency level
    unit: str = "tonnes"
    daily_consumption: float = 0.0
    days_of_supply: float = 0.0
    last_delivery_date: Optional[str] = None
    next_delivery_date: Optional[str] = None
    pending_orders: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived values."""
        if self.daily_consumption > 0:
            self.days_of_supply = self.current_quantity / self.daily_consumption

    @property
    def fill_percent(self) -> float:
        """Get fill percentage."""
        return (self.current_quantity / self.max_capacity * 100) if self.max_capacity > 0 else 0

    @property
    def status(self) -> str:
        """Get inventory status."""
        if self.current_quantity <= self.critical_threshold:
            return "critical"
        elif self.current_quantity <= self.min_threshold:
            return "warning"
        elif self.fill_percent >= 90:
            return "optimal"
        else:
            return "normal"

    @property
    def color(self) -> str:
        """Get color based on status."""
        return get_status_color(self.status)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fuel_id": self.fuel_id,
            "fuel_name": self.fuel_name,
            "fuel_type": self.fuel_type,
            "current_quantity": self.current_quantity,
            "max_capacity": self.max_capacity,
            "fill_percent": self.fill_percent,
            "min_threshold": self.min_threshold,
            "critical_threshold": self.critical_threshold,
            "unit": self.unit,
            "daily_consumption": self.daily_consumption,
            "days_of_supply": self.days_of_supply,
            "status": self.status,
        }


@dataclass
class Contract:
    """Procurement contract details."""
    contract_id: str
    contract_name: str
    supplier_id: str
    supplier_name: str
    fuel_type: str
    start_date: str
    end_date: str
    total_volume: float
    utilized_volume: float
    min_take_volume: float
    max_take_volume: float
    unit_price: float
    currency: str = "USD"
    price_unit: str = "$/tonne"
    status: str = "active"  # active, expiring, expired, draft
    payment_terms: str = ""
    delivery_terms: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def utilization_percent(self) -> float:
        """Calculate contract utilization percentage."""
        return (self.utilized_volume / self.total_volume * 100) if self.total_volume > 0 else 0

    @property
    def remaining_volume(self) -> float:
        """Get remaining contract volume."""
        return self.total_volume - self.utilized_volume

    @property
    def days_until_expiry(self) -> int:
        """Calculate days until contract expiry."""
        end = datetime.fromisoformat(self.end_date.replace('Z', '+00:00'))
        today = datetime.now(end.tzinfo)
        return (end - today).days

    @property
    def is_expiring_soon(self) -> bool:
        """Check if contract is expiring within 30 days."""
        return 0 < self.days_until_expiry <= 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contract_id": self.contract_id,
            "contract_name": self.contract_name,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "fuel_type": self.fuel_type,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_volume": self.total_volume,
            "utilized_volume": self.utilized_volume,
            "remaining_volume": self.remaining_volume,
            "utilization_percent": self.utilization_percent,
            "days_until_expiry": self.days_until_expiry,
            "unit_price": self.unit_price,
            "status": self.status,
        }


@dataclass
class SupplierMetrics:
    """Supplier performance metrics."""
    supplier_id: str
    supplier_name: str
    overall_score: float  # 0-100
    quality_score: float  # 0-100
    delivery_score: float  # 0-100
    price_competitiveness: float  # 0-100
    responsiveness_score: float  # 0-100
    total_orders: int = 0
    on_time_deliveries: int = 0
    late_deliveries: int = 0
    rejected_deliveries: int = 0
    average_lead_time_days: float = 0.0
    total_spend: float = 0.0
    currency: str = "USD"
    fuel_types_supplied: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    contract_count: int = 0
    trend: TrendDirection = TrendDirection.STABLE
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def delivery_reliability(self) -> float:
        """Calculate delivery reliability percentage."""
        if self.total_orders == 0:
            return 100.0
        return (self.on_time_deliveries / self.total_orders) * 100

    @property
    def status(self) -> str:
        """Get supplier status based on score."""
        if self.overall_score >= 90:
            return "excellent"
        elif self.overall_score >= 75:
            return "good"
        elif self.overall_score >= 60:
            return "acceptable"
        elif self.overall_score >= 40:
            return "warning"
        else:
            return "critical"

    @property
    def color(self) -> str:
        """Get color based on status."""
        status_colors = {
            "excellent": StatusColors.SUCCESS,
            "good": "#27AE60",
            "acceptable": StatusColors.WARNING,
            "warning": "#E67E22",
            "critical": StatusColors.ERROR,
        }
        return status_colors.get(self.status, StatusColors.NEUTRAL)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "overall_score": self.overall_score,
            "quality_score": self.quality_score,
            "delivery_score": self.delivery_score,
            "price_competitiveness": self.price_competitiveness,
            "responsiveness_score": self.responsiveness_score,
            "delivery_reliability": self.delivery_reliability,
            "total_orders": self.total_orders,
            "on_time_deliveries": self.on_time_deliveries,
            "average_lead_time_days": self.average_lead_time_days,
            "total_spend": self.total_spend,
            "status": self.status,
            "trend": self.trend.value,
        }


@dataclass
class DeliveryRecord:
    """Delivery record for tracking."""
    delivery_id: str
    supplier_id: str
    supplier_name: str
    fuel_type: str
    quantity: float
    unit: str
    scheduled_date: str
    actual_date: Optional[str] = None
    status: str = "pending"  # pending, in_transit, delivered, delayed, cancelled
    quality_check_passed: bool = True
    quality_score: float = 100.0
    variance_quantity: float = 0.0
    delay_days: int = 0
    cost: float = 0.0
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_on_time(self) -> bool:
        """Check if delivery was on time."""
        return self.delay_days <= 0

    @property
    def color(self) -> str:
        """Get color based on status."""
        status_colors = {
            "delivered": StatusColors.SUCCESS if self.is_on_time else StatusColors.WARNING,
            "pending": StatusColors.INFO,
            "in_transit": StatusColors.INFO,
            "delayed": StatusColors.ERROR,
            "cancelled": StatusColors.NEUTRAL,
        }
        return status_colors.get(self.status, StatusColors.NEUTRAL)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "delivery_id": self.delivery_id,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "fuel_type": self.fuel_type,
            "quantity": self.quantity,
            "unit": self.unit,
            "scheduled_date": self.scheduled_date,
            "actual_date": self.actual_date,
            "status": self.status,
            "is_on_time": self.is_on_time,
            "delay_days": self.delay_days,
            "quality_score": self.quality_score,
        }


@dataclass
class ProcurementAlert:
    """Procurement system alert."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: str
    related_entity_id: Optional[str] = None
    related_entity_type: Optional[str] = None
    is_acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    action_required: bool = True
    recommended_action: Optional[str] = None
    expires_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def color(self) -> str:
        """Get color based on severity."""
        return self.severity.color

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp,
            "is_acknowledged": self.is_acknowledged,
            "action_required": self.action_required,
            "recommended_action": self.recommended_action,
        }


@dataclass
class KPIMetric:
    """Generic KPI metric."""
    kpi_type: KPIType
    name: str
    value: float
    unit: str
    target: Optional[float] = None
    previous_value: Optional[float] = None
    trend: TrendDirection = TrendDirection.STABLE
    trend_percent: float = 0.0
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Get KPI status based on thresholds."""
        if self.target:
            variance = ((self.value - self.target) / self.target) * 100
            if abs(variance) <= 5:
                return "on_target"
            elif variance > 0:
                return "above_target"
            else:
                return "below_target"
        return "normal"

    @property
    def progress_percent(self) -> float:
        """Get progress towards target."""
        if self.target and self.target != 0:
            return (self.value / self.target) * 100
        return 0

    @property
    def color(self) -> str:
        """Get color based on status and KPI type."""
        # For most KPIs, being at/above target is good
        if self.status == "on_target":
            return StatusColors.SUCCESS
        elif self.status == "above_target":
            # Depends on KPI type - for cost, above is bad
            if self.kpi_type == KPIType.COST_VARIANCE:
                return StatusColors.ERROR
            return StatusColors.SUCCESS
        elif self.status == "below_target":
            if self.kpi_type == KPIType.COST_VARIANCE:
                return StatusColors.SUCCESS
            return StatusColors.WARNING
        return StatusColors.NEUTRAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "kpi_type": self.kpi_type.value,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "target": self.target,
            "previous_value": self.previous_value,
            "trend": self.trend.value,
            "trend_percent": self.trend_percent,
            "status": self.status,
            "progress_percent": self.progress_percent,
        }


@dataclass
class ProcurementDashboardData:
    """Complete procurement dashboard data."""
    inventory_levels: List[InventoryLevel]
    contracts: List[Contract]
    suppliers: List[SupplierMetrics]
    deliveries: List[DeliveryRecord]
    alerts: List[ProcurementAlert]
    kpis: List[KPIMetric]
    reporting_period: str = ""
    organization: Optional[str] = None
    facility_id: Optional[str] = None
    total_spend_ytd: float = 0.0
    budget_ytd: float = 0.0
    savings_ytd: float = 0.0
    currency: str = "USD"
    provenance_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived values."""
        self._calculate_provenance()

    def _calculate_provenance(self):
        """Calculate provenance hash."""
        data = {
            "inventory": len(self.inventory_levels),
            "contracts": len(self.contracts),
            "suppliers": len(self.suppliers),
            "alerts": len(self.alerts),
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    @property
    def active_alerts_count(self) -> int:
        """Count of unacknowledged alerts."""
        return sum(1 for a in self.alerts if not a.is_acknowledged)

    @property
    def critical_alerts_count(self) -> int:
        """Count of critical/emergency alerts."""
        return sum(
            1 for a in self.alerts
            if not a.is_acknowledged and a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        )

    @property
    def expiring_contracts_count(self) -> int:
        """Count of contracts expiring within 30 days."""
        return sum(1 for c in self.contracts if c.is_expiring_soon)

    @property
    def low_inventory_count(self) -> int:
        """Count of fuels with low inventory."""
        return sum(1 for i in self.inventory_levels if i.status in ["warning", "critical"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "inventory_levels": [i.to_dict() for i in self.inventory_levels],
            "contracts": [c.to_dict() for c in self.contracts],
            "suppliers": [s.to_dict() for s in self.suppliers],
            "deliveries": [d.to_dict() for d in self.deliveries],
            "alerts": [a.to_dict() for a in self.alerts],
            "kpis": [k.to_dict() for k in self.kpis],
            "reporting_period": self.reporting_period,
            "total_spend_ytd": self.total_spend_ytd,
            "budget_ytd": self.budget_ytd,
            "active_alerts_count": self.active_alerts_count,
            "critical_alerts_count": self.critical_alerts_count,
            "expiring_contracts_count": self.expiring_contracts_count,
            "low_inventory_count": self.low_inventory_count,
        }


# =============================================================================
# CHART OPTIONS
# =============================================================================

@dataclass
class DashboardPanelConfig:
    """Configuration for a single dashboard panel."""
    panel_id: str
    panel_type: DashboardPanelType
    title: str
    position: Tuple[int, int]  # (row, col)
    size: Tuple[int, int] = (1, 1)  # (row_span, col_span)
    data_source: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    refresh_interval: int = 0  # seconds, 0 = no auto-refresh


@dataclass
class ProcurementDashboardOptions:
    """Configuration options for procurement dashboard."""
    # Layout options
    title: str = "Procurement Dashboard"
    subtitle: Optional[str] = None
    layout_columns: int = 3
    layout_rows: int = 4
    panel_gap: int = 20
    panel_padding: int = 15

    # Panel configurations
    panels: List[DashboardPanelConfig] = field(default_factory=list)

    # Display options
    show_inventory_panel: bool = True
    show_contracts_panel: bool = True
    show_suppliers_panel: bool = True
    show_deliveries_panel: bool = True
    show_alerts_panel: bool = True
    show_kpi_cards: bool = True
    show_executive_summary: bool = True

    # Style options
    color_blind_safe: bool = False
    theme_mode: str = "light"

    # Alert options
    show_only_active_alerts: bool = True
    max_alerts_shown: int = 10
    highlight_critical_alerts: bool = True

    # Data options
    days_lookahead: int = 30
    days_lookback: int = 90

    # Size options
    width: Optional[int] = None
    height: Optional[int] = None
    auto_size: bool = True

    # Interaction options
    enable_drill_down: bool = True
    enable_filters: bool = True
    enable_export: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "layout_columns": self.layout_columns,
            "layout_rows": self.layout_rows,
            "show_alerts_panel": self.show_alerts_panel,
            "max_alerts_shown": self.max_alerts_shown,
        }


# =============================================================================
# DASHBOARD ENGINE
# =============================================================================

class ProcurementDashboardEngine:
    """
    Engine for generating procurement dashboard visualizations.

    Creates multi-panel dashboards with inventory, contracts, suppliers,
    deliveries, and alert visualizations.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        theme: Optional[ThemeConfig] = None,
    ):
        """
        Initialize dashboard engine.

        Args:
            config: Global visualization configuration
            theme: Theme configuration for styling
        """
        self.config = config or get_default_config()
        self.theme = theme or self.config.theme
        self._cache: Dict[str, Any] = {}

    def generate(
        self,
        data: ProcurementDashboardData,
        options: Optional[ProcurementDashboardOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete procurement dashboard.

        Args:
            data: Dashboard data
            options: Dashboard configuration options

        Returns:
            Dashboard specification with multiple chart panels
        """
        options = options or ProcurementDashboardOptions()

        # Build individual panels
        panels = {}

        if options.show_kpi_cards:
            panels["kpi_cards"] = self._build_kpi_cards(data, options)

        if options.show_inventory_panel:
            panels["inventory"] = self._build_inventory_panel(data, options)

        if options.show_contracts_panel:
            panels["contracts"] = self._build_contracts_panel(data, options)

        if options.show_suppliers_panel:
            panels["suppliers"] = self._build_suppliers_panel(data, options)

        if options.show_deliveries_panel:
            panels["deliveries"] = self._build_deliveries_panel(data, options)

        if options.show_alerts_panel:
            panels["alerts"] = self._build_alerts_panel(data, options)

        if options.show_executive_summary:
            panels["summary"] = self._build_executive_summary(data, options)

        return {
            "dashboard_type": "procurement",
            "title": options.title,
            "subtitle": options.subtitle,
            "layout": {
                "columns": options.layout_columns,
                "rows": options.layout_rows,
                "gap": options.panel_gap,
            },
            "panels": panels,
            "data_summary": data.to_dict(),
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _build_kpi_cards(
        self,
        data: ProcurementDashboardData,
        options: ProcurementDashboardOptions,
    ) -> List[Dict[str, Any]]:
        """Build KPI card visualizations."""
        cards = []

        # Summary KPIs
        summary_kpis = [
            {
                "title": "Total Spend YTD",
                "value": data.total_spend_ytd,
                "format": "${:,.0f}",
                "trend": "neutral",
                "comparison": f"Budget: ${data.budget_ytd:,.0f}",
            },
            {
                "title": "Active Alerts",
                "value": data.active_alerts_count,
                "format": "{:,.0f}",
                "trend": "up" if data.active_alerts_count > 5 else "stable",
                "color": StatusColors.ERROR if data.critical_alerts_count > 0 else StatusColors.WARNING,
            },
            {
                "title": "Low Inventory Items",
                "value": data.low_inventory_count,
                "format": "{:,.0f}",
                "color": StatusColors.ERROR if data.low_inventory_count > 0 else StatusColors.SUCCESS,
            },
            {
                "title": "Expiring Contracts",
                "value": data.expiring_contracts_count,
                "format": "{:,.0f}",
                "color": StatusColors.WARNING if data.expiring_contracts_count > 0 else StatusColors.SUCCESS,
            },
        ]

        for kpi in summary_kpis:
            cards.append(self._create_kpi_card(kpi))

        # Add custom KPIs from data
        for kpi_metric in data.kpis:
            cards.append(self._create_kpi_card_from_metric(kpi_metric))

        return cards

    def _create_kpi_card(self, kpi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a KPI card specification."""
        return {
            "type": "indicator",
            "mode": "number+delta" if "comparison" in kpi_data else "number",
            "value": kpi_data["value"],
            "title": {"text": kpi_data["title"]},
            "number": {
                "font": {"size": 28},
            },
            "domain": {"x": [0, 1], "y": [0, 1]},
        }

    def _create_kpi_card_from_metric(self, metric: KPIMetric) -> Dict[str, Any]:
        """Create KPI card from KPIMetric."""
        card = {
            "type": "indicator",
            "mode": "number+delta" if metric.previous_value else "number",
            "value": metric.value,
            "title": {"text": metric.name},
            "number": {
                "suffix": f" {metric.unit}",
                "font": {"size": 24, "color": metric.color},
            },
        }

        if metric.previous_value:
            card["delta"] = {
                "reference": metric.previous_value,
                "relative": True,
            }

        if metric.target:
            card["mode"] = "gauge+number+delta"
            card["gauge"] = {
                "axis": {"range": [0, metric.target * 1.5]},
                "bar": {"color": metric.color},
                "threshold": {
                    "line": {"color": StatusColors.ERROR, "width": 4},
                    "thickness": 0.75,
                    "value": metric.target,
                },
            }

        return card

    def _build_inventory_panel(
        self,
        data: ProcurementDashboardData,
        options: ProcurementDashboardOptions,
    ) -> Dict[str, Any]:
        """Build inventory level panel."""
        traces = []

        # Inventory bar chart
        fuel_names = [inv.fuel_name for inv in data.inventory_levels]
        current_levels = [inv.current_quantity for inv in data.inventory_levels]
        capacities = [inv.max_capacity for inv in data.inventory_levels]
        colors = [inv.color for inv in data.inventory_levels]

        # Current inventory bars
        traces.append({
            "type": "bar",
            "name": "Current Level",
            "x": fuel_names,
            "y": current_levels,
            "marker": {"color": colors},
            "text": [f"{inv.fill_percent:.0f}%" for inv in data.inventory_levels],
            "textposition": "outside",
            "hovertemplate": (
                "<b>%{x}</b><br>"
                "Current: %{y:,.0f}<br>"
                "Fill: %{text}<extra></extra>"
            ),
        })

        # Capacity line
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "name": "Max Capacity",
            "x": fuel_names,
            "y": capacities,
            "marker": {
                "symbol": "line-ew",
                "size": 20,
                "color": "#333333",
                "line": {"width": 2},
            },
        })

        # Min threshold line
        min_thresholds = [inv.min_threshold for inv in data.inventory_levels]
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "name": "Reorder Point",
            "x": fuel_names,
            "y": min_thresholds,
            "marker": {
                "symbol": "line-ew",
                "size": 15,
                "color": StatusColors.WARNING,
                "line": {"width": 2},
            },
        })

        layout = {
            "title": {"text": "Inventory Levels by Fuel Type"},
            "barmode": "group",
            "xaxis": {"title": ""},
            "yaxis": {"title": "Quantity (tonnes)"},
            "showlegend": True,
            "legend": {"orientation": "h", "y": -0.2},
        }

        return {
            "data": traces,
            "layout": layout,
            "config": get_plotly_config(),
        }

    def _build_contracts_panel(
        self,
        data: ProcurementDashboardData,
        options: ProcurementDashboardOptions,
    ) -> Dict[str, Any]:
        """Build contract utilization panel."""
        # Contract utilization bars
        contract_names = [c.contract_name[:20] for c in data.contracts]
        utilizations = [c.utilization_percent for c in data.contracts]
        colors = [
            StatusColors.SUCCESS if u >= 80 else (StatusColors.WARNING if u >= 50 else StatusColors.ERROR)
            for u in utilizations
        ]

        trace = {
            "type": "bar",
            "x": contract_names,
            "y": utilizations,
            "marker": {"color": colors},
            "text": [f"{u:.0f}%" for u in utilizations],
            "textposition": "outside",
            "hovertemplate": (
                "<b>%{x}</b><br>"
                "Utilization: %{y:.1f}%<extra></extra>"
            ),
        }

        # Add target line at 100%
        target_line = {
            "type": "scatter",
            "mode": "lines",
            "name": "Target",
            "x": contract_names,
            "y": [100] * len(contract_names),
            "line": {"color": StatusColors.INFO, "dash": "dash", "width": 2},
        }

        layout = {
            "title": {"text": "Contract Utilization"},
            "xaxis": {"title": "", "tickangle": -45},
            "yaxis": {"title": "Utilization (%)", "range": [0, 120]},
            "showlegend": False,
        }

        # Add annotations for expiring contracts
        annotations = []
        for i, contract in enumerate(data.contracts):
            if contract.is_expiring_soon:
                annotations.append({
                    "x": contract.contract_name[:20],
                    "y": contract.utilization_percent + 5,
                    "text": f"Expires in {contract.days_until_expiry}d",
                    "showarrow": False,
                    "font": {"size": 9, "color": StatusColors.ERROR},
                })
        layout["annotations"] = annotations

        return {
            "data": [trace, target_line],
            "layout": layout,
            "config": get_plotly_config(),
        }

    def _build_suppliers_panel(
        self,
        data: ProcurementDashboardData,
        options: ProcurementDashboardOptions,
    ) -> Dict[str, Any]:
        """Build supplier performance panel."""
        # Radar chart for supplier scores
        categories = ["Quality", "Delivery", "Price", "Responsiveness"]

        traces = []
        for supplier in data.suppliers[:5]:  # Top 5 suppliers
            traces.append({
                "type": "scatterpolar",
                "r": [
                    supplier.quality_score,
                    supplier.delivery_score,
                    supplier.price_competitiveness,
                    supplier.responsiveness_score,
                ],
                "theta": categories,
                "fill": "toself",
                "name": supplier.supplier_name,
                "opacity": 0.6,
            })

        layout = {
            "title": {"text": "Supplier Performance Comparison"},
            "polar": {
                "radialaxis": {
                    "visible": True,
                    "range": [0, 100],
                },
            },
            "showlegend": True,
        }

        return {
            "data": traces,
            "layout": layout,
            "config": get_plotly_config(),
        }

    def _build_deliveries_panel(
        self,
        data: ProcurementDashboardData,
        options: ProcurementDashboardOptions,
    ) -> Dict[str, Any]:
        """Build delivery reliability panel."""
        # Calculate delivery statistics
        total_deliveries = len(data.deliveries)
        on_time = sum(1 for d in data.deliveries if d.is_on_time)
        delayed = total_deliveries - on_time

        trace = {
            "type": "pie",
            "labels": ["On Time", "Delayed"],
            "values": [on_time, delayed],
            "marker": {"colors": [StatusColors.SUCCESS, StatusColors.ERROR]},
            "hole": 0.4,
            "textinfo": "percent+label",
        }

        layout = {
            "title": {"text": "Delivery Reliability"},
            "showlegend": True,
            "annotations": [{
                "text": f"{on_time}/{total_deliveries}",
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 20},
                "showarrow": False,
            }],
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": get_plotly_config(),
        }

    def _build_alerts_panel(
        self,
        data: ProcurementDashboardData,
        options: ProcurementDashboardOptions,
    ) -> Dict[str, Any]:
        """Build alerts panel."""
        alerts = data.alerts
        if options.show_only_active_alerts:
            alerts = [a for a in alerts if not a.is_acknowledged]

        alerts = sorted(alerts, key=lambda a: a.severity.color, reverse=True)
        alerts = alerts[:options.max_alerts_shown]

        alert_list = []
        for alert in alerts:
            alert_list.append({
                "id": alert.alert_id,
                "type": alert.alert_type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "color": alert.color,
                "action_required": alert.action_required,
                "recommended_action": alert.recommended_action,
            })

        return {
            "type": "alert_list",
            "title": "Active Alerts",
            "alerts": alert_list,
            "total_count": len(data.alerts),
            "active_count": data.active_alerts_count,
            "critical_count": data.critical_alerts_count,
        }

    def _build_executive_summary(
        self,
        data: ProcurementDashboardData,
        options: ProcurementDashboardOptions,
    ) -> Dict[str, Any]:
        """Build executive summary panel."""
        # Calculate key metrics
        budget_variance = ((data.total_spend_ytd - data.budget_ytd) / data.budget_ytd * 100) if data.budget_ytd else 0
        avg_supplier_score = sum(s.overall_score for s in data.suppliers) / len(data.suppliers) if data.suppliers else 0
        avg_contract_util = sum(c.utilization_percent for c in data.contracts) / len(data.contracts) if data.contracts else 0
        delivery_reliability = (
            sum(1 for d in data.deliveries if d.is_on_time) / len(data.deliveries) * 100
        ) if data.deliveries else 100

        summary = {
            "type": "executive_summary",
            "title": "Executive Summary",
            "period": data.reporting_period,
            "metrics": [
                {
                    "name": "Budget Variance",
                    "value": budget_variance,
                    "unit": "%",
                    "status": "good" if budget_variance <= 0 else "warning",
                    "formatted": f"{budget_variance:+.1f}%",
                },
                {
                    "name": "Avg Supplier Score",
                    "value": avg_supplier_score,
                    "unit": "/100",
                    "status": "good" if avg_supplier_score >= 80 else "warning",
                    "formatted": f"{avg_supplier_score:.0f}/100",
                },
                {
                    "name": "Contract Utilization",
                    "value": avg_contract_util,
                    "unit": "%",
                    "status": "good" if avg_contract_util >= 70 else "warning",
                    "formatted": f"{avg_contract_util:.0f}%",
                },
                {
                    "name": "Delivery Reliability",
                    "value": delivery_reliability,
                    "unit": "%",
                    "status": "good" if delivery_reliability >= 95 else "warning",
                    "formatted": f"{delivery_reliability:.0f}%",
                },
            ],
            "recommendations": self._generate_recommendations(data),
            "highlights": self._generate_highlights(data),
        }

        return summary

    def _generate_recommendations(
        self,
        data: ProcurementDashboardData,
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []

        # Low inventory recommendations
        for inv in data.inventory_levels:
            if inv.status == "critical":
                recommendations.append({
                    "priority": "high",
                    "category": "inventory",
                    "title": f"Critical: Reorder {inv.fuel_name}",
                    "description": f"Inventory at {inv.fill_percent:.0f}% ({inv.days_of_supply:.0f} days supply)",
                    "action": "Place emergency order immediately",
                })
            elif inv.status == "warning":
                recommendations.append({
                    "priority": "medium",
                    "category": "inventory",
                    "title": f"Reorder {inv.fuel_name} soon",
                    "description": f"Inventory at {inv.fill_percent:.0f}% ({inv.days_of_supply:.0f} days supply)",
                    "action": "Schedule delivery within 1 week",
                })

        # Contract recommendations
        for contract in data.contracts:
            if contract.is_expiring_soon:
                recommendations.append({
                    "priority": "high",
                    "category": "contracts",
                    "title": f"Renew contract: {contract.contract_name}",
                    "description": f"Expires in {contract.days_until_expiry} days",
                    "action": "Initiate renewal negotiations",
                })

        # Supplier recommendations
        for supplier in data.suppliers:
            if supplier.overall_score < 60:
                recommendations.append({
                    "priority": "medium",
                    "category": "suppliers",
                    "title": f"Review supplier: {supplier.supplier_name}",
                    "description": f"Performance score: {supplier.overall_score:.0f}/100",
                    "action": "Schedule performance review meeting",
                })

        return recommendations[:10]  # Top 10 recommendations

    def _generate_highlights(
        self,
        data: ProcurementDashboardData,
    ) -> List[Dict[str, Any]]:
        """Generate dashboard highlights."""
        highlights = []

        # Spending highlight
        if data.total_spend_ytd < data.budget_ytd:
            savings = data.budget_ytd - data.total_spend_ytd
            highlights.append({
                "type": "positive",
                "title": "Under Budget",
                "description": f"${savings:,.0f} savings vs budget",
            })
        elif data.total_spend_ytd > data.budget_ytd * 1.1:
            overspend = data.total_spend_ytd - data.budget_ytd
            highlights.append({
                "type": "negative",
                "title": "Over Budget",
                "description": f"${overspend:,.0f} over budget",
            })

        # Top supplier
        if data.suppliers:
            top_supplier = max(data.suppliers, key=lambda s: s.overall_score)
            highlights.append({
                "type": "positive",
                "title": "Top Supplier",
                "description": f"{top_supplier.supplier_name} ({top_supplier.overall_score:.0f}/100)",
            })

        # Alert status
        if data.critical_alerts_count > 0:
            highlights.append({
                "type": "negative",
                "title": "Critical Alerts",
                "description": f"{data.critical_alerts_count} critical alerts require attention",
            })
        elif data.active_alerts_count == 0:
            highlights.append({
                "type": "positive",
                "title": "All Clear",
                "description": "No active alerts",
            })

        return highlights

    def generate_inventory_detail(
        self,
        data: ProcurementDashboardData,
        options: Optional[ProcurementDashboardOptions] = None,
    ) -> Dict[str, Any]:
        """Generate detailed inventory view."""
        traces = []

        for inv in data.inventory_levels:
            # Create gauge for each fuel
            trace = {
                "type": "indicator",
                "mode": "gauge+number+delta",
                "value": inv.fill_percent,
                "title": {"text": inv.fuel_name},
                "gauge": {
                    "axis": {"range": [0, 100]},
                    "bar": {"color": inv.color},
                    "steps": [
                        {"range": [0, 20], "color": "#FFEBEE"},
                        {"range": [20, 50], "color": "#FFF3E0"},
                        {"range": [50, 100], "color": "#E8F5E9"},
                    ],
                    "threshold": {
                        "line": {"color": StatusColors.ERROR, "width": 4},
                        "thickness": 0.75,
                        "value": (inv.min_threshold / inv.max_capacity) * 100,
                    },
                },
                "number": {"suffix": "%"},
                "domain": {"row": 0, "column": data.inventory_levels.index(inv)},
            }
            traces.append(trace)

        layout = {
            "title": {"text": "Inventory Levels Detail"},
            "grid": {"rows": 1, "columns": len(data.inventory_levels), "pattern": "independent"},
        }

        return {
            "data": traces,
            "layout": layout,
            "config": get_plotly_config(),
        }

    def generate_supplier_scorecard(
        self,
        supplier: SupplierMetrics,
        options: Optional[ProcurementDashboardOptions] = None,
    ) -> Dict[str, Any]:
        """Generate detailed supplier scorecard."""
        # Create bullet chart for each metric
        metrics = [
            ("Quality", supplier.quality_score, 90),
            ("Delivery", supplier.delivery_score, 95),
            ("Price", supplier.price_competitiveness, 85),
            ("Responsiveness", supplier.responsiveness_score, 90),
        ]

        traces = []
        for i, (name, value, target) in enumerate(metrics):
            traces.append({
                "type": "indicator",
                "mode": "number+gauge",
                "value": value,
                "title": {"text": name},
                "gauge": {
                    "shape": "bullet",
                    "axis": {"range": [0, 100]},
                    "bar": {"color": StatusColors.SUCCESS if value >= target else StatusColors.WARNING},
                    "threshold": {
                        "line": {"color": "#333333", "width": 2},
                        "thickness": 0.75,
                        "value": target,
                    },
                },
                "domain": {"x": [0, 1], "y": [1 - (i + 1) * 0.25, 1 - i * 0.25]},
            })

        layout = {
            "title": {"text": f"Supplier Scorecard: {supplier.supplier_name}"},
            "height": 400,
        }

        return {
            "data": traces,
            "layout": layout,
            "config": get_plotly_config(),
        }

    def to_json(self, dashboard: Dict[str, Any]) -> str:
        """Export dashboard to JSON string."""
        return json.dumps(dashboard, indent=2, default=str)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def create_sample_dashboard_data() -> ProcurementDashboardData:
    """Create sample dashboard data for demonstration."""
    inventory_levels = [
        InventoryLevel(
            fuel_id="coal",
            fuel_name="Bituminous Coal",
            fuel_type="coal",
            current_quantity=8500,
            max_capacity=15000,
            min_threshold=5000,
            critical_threshold=2500,
            daily_consumption=150,
        ),
        InventoryLevel(
            fuel_id="natural_gas",
            fuel_name="Natural Gas",
            fuel_type="natural_gas",
            current_quantity=4200,
            max_capacity=10000,
            min_threshold=3000,
            critical_threshold=1500,
            daily_consumption=100,
        ),
        InventoryLevel(
            fuel_id="biomass",
            fuel_name="Wood Pellets",
            fuel_type="biomass",
            current_quantity=1200,
            max_capacity=5000,
            min_threshold=1500,
            critical_threshold=750,
            daily_consumption=50,
        ),
    ]

    contracts = [
        Contract(
            contract_id="C001",
            contract_name="Coal Supply Agreement",
            supplier_id="S001",
            supplier_name="CoalCorp",
            fuel_type="coal",
            start_date="2024-01-01",
            end_date="2024-12-31",
            total_volume=50000,
            utilized_volume=35000,
            min_take_volume=40000,
            max_take_volume=60000,
            unit_price=85.0,
        ),
        Contract(
            contract_id="C002",
            contract_name="Gas Framework",
            supplier_id="S002",
            supplier_name="GasWorks",
            fuel_type="natural_gas",
            start_date="2024-03-01",
            end_date="2025-02-28",
            total_volume=30000,
            utilized_volume=18000,
            min_take_volume=25000,
            max_take_volume=35000,
            unit_price=4.5,
        ),
    ]

    suppliers = [
        SupplierMetrics(
            supplier_id="S001",
            supplier_name="CoalCorp",
            overall_score=85,
            quality_score=90,
            delivery_score=80,
            price_competitiveness=85,
            responsiveness_score=85,
            total_orders=50,
            on_time_deliveries=45,
            total_spend=2500000,
        ),
        SupplierMetrics(
            supplier_id="S002",
            supplier_name="GasWorks",
            overall_score=92,
            quality_score=95,
            delivery_score=90,
            price_competitiveness=88,
            responsiveness_score=95,
            total_orders=80,
            on_time_deliveries=78,
            total_spend=1800000,
        ),
    ]

    deliveries = [
        DeliveryRecord(
            delivery_id="D001",
            supplier_id="S001",
            supplier_name="CoalCorp",
            fuel_type="coal",
            quantity=500,
            unit="tonnes",
            scheduled_date="2024-11-01",
            actual_date="2024-11-01",
            status="delivered",
        ),
        DeliveryRecord(
            delivery_id="D002",
            supplier_id="S002",
            supplier_name="GasWorks",
            fuel_type="natural_gas",
            quantity=200,
            unit="MWh",
            scheduled_date="2024-11-05",
            actual_date="2024-11-07",
            status="delivered",
            delay_days=2,
        ),
    ]

    alerts = [
        ProcurementAlert(
            alert_id="A001",
            alert_type=AlertType.LOW_INVENTORY,
            severity=AlertSeverity.WARNING,
            title="Low Biomass Inventory",
            message="Wood pellet inventory below reorder point",
            timestamp=datetime.now().isoformat(),
            recommended_action="Place order for 2000 tonnes",
        ),
        ProcurementAlert(
            alert_id="A002",
            alert_type=AlertType.CONTRACT_EXPIRY,
            severity=AlertSeverity.INFO,
            title="Contract Renewal Due",
            message="Coal Supply Agreement expires in 45 days",
            timestamp=datetime.now().isoformat(),
            recommended_action="Initiate renewal discussions",
        ),
    ]

    kpis = [
        KPIMetric(
            kpi_type=KPIType.COST_VARIANCE,
            name="Cost Variance",
            value=-2.5,
            unit="%",
            target=0,
            trend=TrendDirection.DOWN,
        ),
        KPIMetric(
            kpi_type=KPIType.DELIVERY_RELIABILITY,
            name="On-Time Delivery",
            value=94.5,
            unit="%",
            target=95,
            previous_value=92.0,
            trend=TrendDirection.UP,
        ),
    ]

    return ProcurementDashboardData(
        inventory_levels=inventory_levels,
        contracts=contracts,
        suppliers=suppliers,
        deliveries=deliveries,
        alerts=alerts,
        kpis=kpis,
        reporting_period="2024 YTD",
        total_spend_ytd=4500000,
        budget_ytd=5000000,
    )


def example_full_dashboard():
    """Example: Generate full procurement dashboard."""
    print("Generating full procurement dashboard...")

    data = create_sample_dashboard_data()
    engine = ProcurementDashboardEngine()

    dashboard = engine.generate(data)
    print(f"Dashboard generated with {len(dashboard['panels'])} panels")
    return dashboard


def example_inventory_detail():
    """Example: Generate inventory detail view."""
    print("Generating inventory detail view...")

    data = create_sample_dashboard_data()
    engine = ProcurementDashboardEngine()

    chart = engine.generate_inventory_detail(data)
    print(f"Inventory detail generated")
    return chart


def run_all_examples():
    """Run all procurement dashboard examples."""
    print("=" * 60)
    print("GL-011 FUELCRAFT - Procurement Dashboard Examples")
    print("=" * 60)

    examples = [
        ("Full Dashboard", example_full_dashboard),
        ("Inventory Detail", example_inventory_detail),
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
