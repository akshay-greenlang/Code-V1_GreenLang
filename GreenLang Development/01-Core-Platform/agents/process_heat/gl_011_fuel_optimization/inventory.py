"""
GL-011 FUELCRAFT - Inventory Manager

This module provides fuel inventory management including tank level monitoring,
delivery scheduling, consumption forecasting, and minimum stock alerts.

Features:
    - Tank level monitoring and alerts
    - Consumption rate tracking
    - Delivery scheduling optimization
    - Safety stock management
    - Economic Order Quantity (EOQ) calculations
    - Weather-adjusted demand forecasting

Example:
    >>> from greenlang.agents.process_heat.gl_011_fuel_optimization.inventory import (
    ...     InventoryManager,
    ...     TankStatus,
    ... )
    >>>
    >>> manager = InventoryManager(config)
    >>> status = manager.get_tank_status("TANK-001")
    >>> print(f"Days of supply: {status.days_of_supply:.1f}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math
import uuid

from pydantic import BaseModel, Field, validator

from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    InventoryConfig,
    FuelType,
    AlertLevel,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import (
    InventoryStatus,
    InventoryAlertType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class LevelStatus(Enum):
    """Tank level status."""
    CRITICAL = "critical"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    FULL = "full"


class DeliveryStatus(Enum):
    """Delivery status."""
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    IN_TRANSIT = "in_transit"
    ARRIVED = "arrived"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# Unit conversion factors
UNIT_CONVERSIONS = {
    "gal_to_bbl": 1 / 42,
    "gal_to_scf_ng": 0.128,  # Natural gas
    "gal_to_mmbtu_ng": 0.128 * 1.028 / 1000,
    "lb_to_gal_oil": 1 / 7.1,  # Approximate for fuel oil
    "ton_to_lb": 2000,
}


# =============================================================================
# DATA MODELS
# =============================================================================

class TankConfig(BaseModel):
    """Tank configuration."""

    tank_id: str = Field(..., description="Tank identifier")
    name: str = Field(default="", description="Tank name")
    fuel_type: str = Field(..., description="Fuel type stored")

    # Capacity
    capacity_gal: float = Field(..., gt=0, description="Total capacity (gallons)")
    usable_capacity_pct: float = Field(
        default=95.0,
        ge=80,
        le=100,
        description="Usable capacity percentage"
    )
    heel_volume_gal: float = Field(
        default=0.0,
        ge=0,
        description="Minimum heel volume (gallons)"
    )

    # Thresholds
    reorder_point_pct: float = Field(
        default=30.0,
        ge=10,
        le=50,
        description="Reorder point (%)"
    )
    low_level_pct: float = Field(
        default=25.0,
        ge=10,
        le=40,
        description="Low level alert (%)"
    )
    critical_level_pct: float = Field(
        default=15.0,
        ge=5,
        le=25,
        description="Critical level (%)"
    )
    high_level_pct: float = Field(
        default=95.0,
        ge=85,
        le=100,
        description="High level alert (%)"
    )

    # Location
    location: str = Field(default="", description="Tank location")
    connected_equipment: List[str] = Field(
        default_factory=list,
        description="Connected equipment IDs"
    )

    @property
    def usable_capacity_gal(self) -> float:
        """Get usable capacity in gallons."""
        return self.capacity_gal * (self.usable_capacity_pct / 100) - self.heel_volume_gal


class TankStatus(BaseModel):
    """Current tank status."""

    tank_id: str = Field(..., description="Tank identifier")
    fuel_type: str = Field(..., description="Fuel type")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Status timestamp"
    )

    # Levels
    current_level_gal: float = Field(..., ge=0, description="Current level (gallons)")
    current_level_pct: float = Field(..., ge=0, le=100, description="Current level (%)")
    usable_volume_gal: float = Field(..., ge=0, description="Usable volume (gallons)")

    # Status
    level_status: LevelStatus = Field(..., description="Level status")
    temperature_f: Optional[float] = Field(
        default=None,
        description="Fuel temperature (F)"
    )

    # Consumption
    consumption_rate_gal_hr: float = Field(
        default=0.0,
        ge=0,
        description="Current consumption rate"
    )
    avg_daily_consumption_gal: float = Field(
        default=0.0,
        ge=0,
        description="Average daily consumption"
    )
    days_of_supply: float = Field(
        default=0.0,
        ge=0,
        description="Days of supply remaining"
    )
    hours_to_critical: Optional[float] = Field(
        default=None,
        description="Hours until critical level"
    )

    # Quality
    last_delivery_date: Optional[datetime] = Field(
        default=None,
        description="Last delivery date"
    )
    fuel_age_days: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fuel age in days"
    )
    quality_status: str = Field(default="good", description="Quality status")

    class Config:
        use_enum_values = True


class DeliverySchedule(BaseModel):
    """Scheduled delivery."""

    delivery_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Delivery identifier"
    )
    tank_id: str = Field(..., description="Target tank")
    fuel_type: str = Field(..., description="Fuel type")

    # Schedule
    scheduled_date: datetime = Field(..., description="Scheduled delivery date")
    delivery_window_start: datetime = Field(..., description="Window start")
    delivery_window_end: datetime = Field(..., description="Window end")

    # Quantity
    quantity_gal: float = Field(..., gt=0, description="Delivery quantity (gallons)")
    quantity_mmbtu: Optional[float] = Field(
        default=None,
        description="Quantity in MMBTU"
    )

    # Cost
    estimated_cost_usd: Optional[float] = Field(
        default=None,
        description="Estimated cost"
    )
    price_per_gal: Optional[float] = Field(
        default=None,
        description="Price per gallon"
    )

    # Status
    status: DeliveryStatus = Field(
        default=DeliveryStatus.SCHEDULED,
        description="Delivery status"
    )
    carrier: Optional[str] = Field(default=None, description="Carrier name")
    confirmation_number: Optional[str] = Field(
        default=None,
        description="Confirmation number"
    )

    # Actual delivery
    actual_delivery_time: Optional[datetime] = Field(
        default=None,
        description="Actual delivery time"
    )
    actual_quantity_gal: Optional[float] = Field(
        default=None,
        description="Actual quantity delivered"
    )

    class Config:
        use_enum_values = True


class InventoryAlert(BaseModel):
    """Inventory alert."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Alert identifier"
    )
    tank_id: str = Field(..., description="Tank identifier")
    alert_type: InventoryAlertType = Field(..., description="Alert type")
    level: AlertLevel = Field(..., description="Alert level")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert timestamp"
    )

    # Details
    message: str = Field(..., description="Alert message")
    current_value: float = Field(..., description="Current value")
    threshold_value: float = Field(..., description="Threshold value")

    # Status
    acknowledged: bool = Field(default=False, description="Acknowledged")
    acknowledged_by: Optional[str] = Field(
        default=None,
        description="Acknowledged by"
    )
    resolved: bool = Field(default=False, description="Resolved")
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="Resolution time"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# CONSUMPTION TRACKER
# =============================================================================

@dataclass
class ConsumptionRecord:
    """Record of fuel consumption."""
    timestamp: datetime
    tank_id: str
    consumption_gal: float
    duration_hours: float


class ConsumptionTracker:
    """Tracks fuel consumption for forecasting."""

    def __init__(self, history_days: int = 90) -> None:
        """Initialize consumption tracker."""
        self._records: Dict[str, List[ConsumptionRecord]] = {}
        self._history_days = history_days

    def add_record(
        self,
        tank_id: str,
        consumption_gal: float,
        duration_hours: float = 1.0,
    ) -> None:
        """Add consumption record."""
        if tank_id not in self._records:
            self._records[tank_id] = []

        record = ConsumptionRecord(
            timestamp=datetime.now(timezone.utc),
            tank_id=tank_id,
            consumption_gal=consumption_gal,
            duration_hours=duration_hours,
        )
        self._records[tank_id].append(record)

        # Trim old records
        self._trim_history(tank_id)

    def get_average_daily(self, tank_id: str, days: int = 30) -> float:
        """Get average daily consumption."""
        records = self._get_recent_records(tank_id, days)
        if not records:
            return 0.0

        total_consumption = sum(r.consumption_gal for r in records)
        total_hours = sum(r.duration_hours for r in records)

        if total_hours == 0:
            return 0.0

        return (total_consumption / total_hours) * 24

    def get_current_rate(self, tank_id: str) -> float:
        """Get current consumption rate (gal/hr)."""
        records = self._get_recent_records(tank_id, days=1)
        if not records:
            return 0.0

        total_consumption = sum(r.consumption_gal for r in records)
        total_hours = sum(r.duration_hours for r in records)

        if total_hours == 0:
            return 0.0

        return total_consumption / total_hours

    def forecast_consumption(
        self,
        tank_id: str,
        horizon_days: int = 14,
    ) -> List[Tuple[datetime, float]]:
        """Forecast daily consumption."""
        avg_daily = self.get_average_daily(tank_id)
        forecasts = []

        now = datetime.now(timezone.utc)
        for day in range(1, horizon_days + 1):
            forecast_date = now + timedelta(days=day)
            # Simple forecast: use average with day-of-week adjustment
            dow_factor = self._get_dow_factor(forecast_date.weekday())
            forecast = avg_daily * dow_factor
            forecasts.append((forecast_date, forecast))

        return forecasts

    def _get_recent_records(
        self,
        tank_id: str,
        days: int,
    ) -> List[ConsumptionRecord]:
        """Get records from recent period."""
        if tank_id not in self._records:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [r for r in self._records[tank_id] if r.timestamp >= cutoff]

    def _trim_history(self, tank_id: str) -> None:
        """Trim old history records."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._history_days)
        self._records[tank_id] = [
            r for r in self._records[tank_id]
            if r.timestamp >= cutoff
        ]

    def _get_dow_factor(self, day_of_week: int) -> float:
        """Get day-of-week adjustment factor."""
        # Weekend reduction typical for industrial facilities
        if day_of_week >= 5:  # Saturday, Sunday
            return 0.7
        return 1.0


# =============================================================================
# INVENTORY MANAGER
# =============================================================================

class InventoryManager:
    """
    Fuel inventory manager.

    Manages fuel tank levels, consumption tracking, delivery scheduling,
    and inventory alerts for the FuelOptimizationAgent.

    Features:
        - Tank level monitoring
        - Consumption forecasting
        - Delivery scheduling
        - EOQ calculations
        - Minimum stock alerts
        - Weather-adjusted demand

    Example:
        >>> manager = InventoryManager(config)
        >>> status = manager.get_tank_status("TANK-001")
        >>> if status.level_status == LevelStatus.LOW:
        ...     schedule = manager.schedule_delivery("TANK-001")
    """

    def __init__(self, config: InventoryConfig) -> None:
        """
        Initialize the inventory manager.

        Args:
            config: Inventory configuration
        """
        self.config = config
        self._tanks: Dict[str, TankConfig] = {}
        self._levels: Dict[str, float] = {}  # Current levels in gallons
        self._consumption_tracker = ConsumptionTracker()
        self._scheduled_deliveries: List[DeliverySchedule] = []
        self._active_alerts: List[InventoryAlert] = []
        self._last_delivery_dates: Dict[str, datetime] = {}

        # Load tank configs
        for tank_id, tank_config in config.tanks.items():
            self._tanks[tank_id] = TankConfig(tank_id=tank_id, **tank_config)

        logger.info(
            f"InventoryManager initialized with {len(self._tanks)} tanks"
        )

    def add_tank(self, tank_config: TankConfig) -> None:
        """Add a tank to management."""
        self._tanks[tank_config.tank_id] = tank_config
        logger.info(f"Added tank {tank_config.tank_id}")

    def update_level(
        self,
        tank_id: str,
        level_gal: float,
        temperature_f: Optional[float] = None,
    ) -> TankStatus:
        """
        Update tank level and return status.

        Args:
            tank_id: Tank identifier
            level_gal: Current level in gallons
            temperature_f: Optional fuel temperature

        Returns:
            Updated TankStatus
        """
        if tank_id not in self._tanks:
            raise ValueError(f"Unknown tank: {tank_id}")

        tank = self._tanks[tank_id]
        old_level = self._levels.get(tank_id, level_gal)

        # Update level
        self._levels[tank_id] = level_gal

        # Track consumption if level decreased
        if level_gal < old_level:
            consumption = old_level - level_gal
            self._consumption_tracker.add_record(tank_id, consumption)

        # Get current status
        status = self.get_tank_status(tank_id)

        # Check for alerts
        self._check_alerts(tank_id, status)

        return status

    def get_tank_status(self, tank_id: str) -> TankStatus:
        """
        Get current tank status.

        Args:
            tank_id: Tank identifier

        Returns:
            Current TankStatus
        """
        if tank_id not in self._tanks:
            raise ValueError(f"Unknown tank: {tank_id}")

        tank = self._tanks[tank_id]
        current_level = self._levels.get(tank_id, 0.0)

        # Calculate level percentage
        level_pct = (current_level / tank.capacity_gal) * 100 if tank.capacity_gal > 0 else 0

        # Calculate usable volume
        usable = max(0, current_level - tank.heel_volume_gal)

        # Determine level status
        if level_pct <= tank.critical_level_pct:
            level_status = LevelStatus.CRITICAL
        elif level_pct <= tank.low_level_pct:
            level_status = LevelStatus.LOW
        elif level_pct >= tank.high_level_pct:
            level_status = LevelStatus.HIGH
        elif level_pct >= 98:
            level_status = LevelStatus.FULL
        else:
            level_status = LevelStatus.NORMAL

        # Get consumption data
        consumption_rate = self._consumption_tracker.get_current_rate(tank_id)
        avg_daily = self._consumption_tracker.get_average_daily(tank_id)

        # Calculate days of supply
        days_of_supply = usable / avg_daily if avg_daily > 0 else float("inf")

        # Calculate hours to critical
        critical_level = tank.capacity_gal * (tank.critical_level_pct / 100)
        volume_to_critical = current_level - critical_level
        hours_to_critical = None
        if consumption_rate > 0 and volume_to_critical > 0:
            hours_to_critical = volume_to_critical / consumption_rate

        # Fuel age
        last_delivery = self._last_delivery_dates.get(tank_id)
        fuel_age = None
        if last_delivery:
            fuel_age = (datetime.now(timezone.utc) - last_delivery).days

        return TankStatus(
            tank_id=tank_id,
            fuel_type=tank.fuel_type,
            current_level_gal=round(current_level, 1),
            current_level_pct=round(level_pct, 1),
            usable_volume_gal=round(usable, 1),
            level_status=level_status,
            consumption_rate_gal_hr=round(consumption_rate, 2),
            avg_daily_consumption_gal=round(avg_daily, 1),
            days_of_supply=round(days_of_supply, 1) if days_of_supply != float("inf") else 999,
            hours_to_critical=round(hours_to_critical, 1) if hours_to_critical else None,
            last_delivery_date=last_delivery,
            fuel_age_days=fuel_age,
        )

    def get_all_tank_status(self) -> Dict[str, TankStatus]:
        """Get status for all tanks."""
        return {
            tank_id: self.get_tank_status(tank_id)
            for tank_id in self._tanks
        }

    def schedule_delivery(
        self,
        tank_id: str,
        quantity_gal: Optional[float] = None,
        target_date: Optional[datetime] = None,
    ) -> DeliverySchedule:
        """
        Schedule a fuel delivery.

        Args:
            tank_id: Target tank
            quantity_gal: Quantity to order (uses EOQ if not specified)
            target_date: Target delivery date

        Returns:
            DeliverySchedule with delivery details
        """
        if tank_id not in self._tanks:
            raise ValueError(f"Unknown tank: {tank_id}")

        tank = self._tanks[tank_id]
        status = self.get_tank_status(tank_id)

        # Calculate quantity if not specified
        if quantity_gal is None:
            quantity_gal = self._calculate_order_quantity(tank_id, status)

        # Calculate target date if not specified
        if target_date is None:
            target_date = self._calculate_delivery_date(tank_id, status)

        # Create delivery window
        window_start = target_date.replace(hour=8, minute=0, second=0, microsecond=0)
        window_end = window_start + timedelta(hours=self.config.delivery_window_hours)

        schedule = DeliverySchedule(
            tank_id=tank_id,
            fuel_type=tank.fuel_type,
            scheduled_date=target_date,
            delivery_window_start=window_start,
            delivery_window_end=window_end,
            quantity_gal=round(quantity_gal, 0),
        )

        self._scheduled_deliveries.append(schedule)

        logger.info(
            f"Scheduled delivery for {tank_id}: {quantity_gal:.0f} gal on {target_date.date()}"
        )

        return schedule

    def record_delivery(
        self,
        delivery_id: str,
        actual_quantity_gal: float,
    ) -> None:
        """
        Record a completed delivery.

        Args:
            delivery_id: Delivery identifier
            actual_quantity_gal: Actual quantity delivered
        """
        for delivery in self._scheduled_deliveries:
            if delivery.delivery_id == delivery_id:
                delivery.status = DeliveryStatus.COMPLETED
                delivery.actual_delivery_time = datetime.now(timezone.utc)
                delivery.actual_quantity_gal = actual_quantity_gal

                # Update tank level
                tank_id = delivery.tank_id
                current = self._levels.get(tank_id, 0.0)
                self._levels[tank_id] = current + actual_quantity_gal

                # Update last delivery date
                self._last_delivery_dates[tank_id] = datetime.now(timezone.utc)

                logger.info(
                    f"Recorded delivery {delivery_id}: {actual_quantity_gal:.0f} gal"
                )
                return

        logger.warning(f"Delivery {delivery_id} not found")

    def get_pending_deliveries(
        self,
        tank_id: Optional[str] = None,
    ) -> List[DeliverySchedule]:
        """Get pending deliveries."""
        pending = [
            d for d in self._scheduled_deliveries
            if d.status in [DeliveryStatus.SCHEDULED, DeliveryStatus.CONFIRMED, DeliveryStatus.IN_TRANSIT]
        ]

        if tank_id:
            pending = [d for d in pending if d.tank_id == tank_id]

        return pending

    def get_active_alerts(self) -> List[InventoryAlert]:
        """Get active (unresolved) alerts."""
        return [a for a in self._active_alerts if not a.resolved]

    def acknowledge_alert(self, alert_id: str, operator_id: str) -> None:
        """Acknowledge an alert."""
        for alert in self._active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = operator_id
                logger.info(f"Alert {alert_id} acknowledged by {operator_id}")
                return

    def resolve_alert(self, alert_id: str) -> None:
        """Resolve an alert."""
        for alert in self._active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now(timezone.utc)
                logger.info(f"Alert {alert_id} resolved")
                return

    def calculate_eoq(
        self,
        tank_id: str,
        annual_demand_gal: Optional[float] = None,
        ordering_cost_usd: float = 150.0,
        holding_cost_pct: float = 0.20,
        unit_cost_usd_gal: float = 2.50,
    ) -> float:
        """
        Calculate Economic Order Quantity.

        EOQ = sqrt((2 * D * S) / H)
        where:
            D = Annual demand
            S = Ordering cost per order
            H = Holding cost per unit per year

        Args:
            tank_id: Tank identifier
            annual_demand_gal: Annual demand (calculated if not provided)
            ordering_cost_usd: Cost per order
            holding_cost_pct: Holding cost as % of inventory value
            unit_cost_usd_gal: Unit cost per gallon

        Returns:
            Economic Order Quantity in gallons
        """
        if annual_demand_gal is None:
            # Estimate from historical consumption
            avg_daily = self._consumption_tracker.get_average_daily(tank_id)
            annual_demand_gal = avg_daily * 365

        if annual_demand_gal <= 0:
            return 0.0

        # Holding cost per gallon per year
        holding_cost = unit_cost_usd_gal * holding_cost_pct

        # EOQ formula
        eoq = math.sqrt((2 * annual_demand_gal * ordering_cost_usd) / holding_cost)

        return round(eoq, 0)

    def _calculate_order_quantity(
        self,
        tank_id: str,
        status: TankStatus,
    ) -> float:
        """Calculate order quantity for a tank."""
        tank = self._tanks[tank_id]

        if self.config.economic_order_quantity_enabled:
            # Use EOQ
            eoq = self.calculate_eoq(tank_id)
            if eoq > 0:
                # Don't exceed available tank capacity
                available = tank.capacity_gal * 0.95 - status.current_level_gal
                return min(eoq, available)

        # Default: fill to 90% capacity
        target_level = tank.capacity_gal * 0.90
        return max(0, target_level - status.current_level_gal)

    def _calculate_delivery_date(
        self,
        tank_id: str,
        status: TankStatus,
    ) -> datetime:
        """Calculate optimal delivery date."""
        tank = self._tanks[tank_id]
        now = datetime.now(timezone.utc)

        # Calculate when we'll hit reorder point
        reorder_level = tank.capacity_gal * (tank.reorder_point_pct / 100)
        volume_to_reorder = status.current_level_gal - reorder_level

        if status.consumption_rate_gal_hr > 0:
            hours_to_reorder = volume_to_reorder / status.consumption_rate_gal_hr
        else:
            hours_to_reorder = self.config.lead_time_days * 24

        # Subtract lead time
        target_date = now + timedelta(hours=hours_to_reorder)
        target_date -= timedelta(days=self.config.lead_time_days)

        # Ensure it's a preferred delivery day
        while target_date.isoweekday() not in self.config.preferred_delivery_days:
            target_date -= timedelta(days=1)

        # Don't schedule in the past
        if target_date < now:
            target_date = now + timedelta(days=1)
            while target_date.isoweekday() not in self.config.preferred_delivery_days:
                target_date += timedelta(days=1)

        return target_date

    def _check_alerts(self, tank_id: str, status: TankStatus) -> None:
        """Check for alert conditions."""
        tank = self._tanks[tank_id]

        # Check for existing unresolved alerts for this tank
        existing = {
            a.alert_type for a in self._active_alerts
            if a.tank_id == tank_id and not a.resolved
        }

        # Critical level alert
        if status.level_status == LevelStatus.CRITICAL:
            if InventoryAlertType.CRITICAL_LEVEL not in existing:
                self._create_alert(
                    tank_id,
                    InventoryAlertType.CRITICAL_LEVEL,
                    AlertLevel.CRITICAL,
                    f"Tank {tank_id} at critical level ({status.current_level_pct:.1f}%)",
                    status.current_level_pct,
                    tank.critical_level_pct,
                )

        # Low level alert
        elif status.level_status == LevelStatus.LOW:
            if InventoryAlertType.LOW_LEVEL not in existing:
                self._create_alert(
                    tank_id,
                    InventoryAlertType.LOW_LEVEL,
                    AlertLevel.WARNING,
                    f"Tank {tank_id} at low level ({status.current_level_pct:.1f}%)",
                    status.current_level_pct,
                    tank.low_level_pct,
                )

        # High level alert
        elif status.level_status == LevelStatus.HIGH:
            if InventoryAlertType.HIGH_LEVEL not in existing:
                self._create_alert(
                    tank_id,
                    InventoryAlertType.HIGH_LEVEL,
                    AlertLevel.WARNING,
                    f"Tank {tank_id} at high level ({status.current_level_pct:.1f}%)",
                    status.current_level_pct,
                    tank.high_level_pct,
                )

        # Check for delivery needed
        if status.current_level_pct <= tank.reorder_point_pct:
            pending = self.get_pending_deliveries(tank_id)
            if not pending and InventoryAlertType.DELIVERY_REQUIRED not in existing:
                self._create_alert(
                    tank_id,
                    InventoryAlertType.DELIVERY_REQUIRED,
                    AlertLevel.WARNING,
                    f"Tank {tank_id} below reorder point, delivery required",
                    status.current_level_pct,
                    tank.reorder_point_pct,
                )

    def _create_alert(
        self,
        tank_id: str,
        alert_type: InventoryAlertType,
        level: AlertLevel,
        message: str,
        current_value: float,
        threshold_value: float,
    ) -> None:
        """Create a new alert."""
        alert = InventoryAlert(
            tank_id=tank_id,
            alert_type=alert_type,
            level=level,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
        )
        self._active_alerts.append(alert)
        logger.warning(f"Alert created: {message}")

    @property
    def tank_count(self) -> int:
        """Get number of managed tanks."""
        return len(self._tanks)

    @property
    def alert_count(self) -> int:
        """Get number of active alerts."""
        return len(self.get_active_alerts())
