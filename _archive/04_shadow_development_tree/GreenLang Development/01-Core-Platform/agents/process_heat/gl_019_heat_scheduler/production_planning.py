"""
GL-019 HEATSCHEDULER - Production Planning Module

Production schedule integration for heat load optimization, including
shift scheduling, order prioritization, and ERP system integration.

Key Features:
    - Production order scheduling with heat requirements
    - Shift-based heat load modeling
    - Flexible order rescheduling for cost optimization
    - ERP system integration (SAP, Oracle, etc.)
    - Preheat and cool-down time management
    - Zero-hallucination: Deterministic scheduling algorithms

Author: GreenLang Team
Version: 1.0.0
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import heapq

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
    ProductionOrder,
    ProductionScheduleResult,
    ProductionStatus,
    ScheduleActionItem,
    ScheduleAction,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PRODUCTION SHIFT MODEL
# =============================================================================

class ProductionShift(BaseModel):
    """Production shift definition."""

    shift_id: str = Field(..., description="Shift identifier")
    name: str = Field(default="", description="Shift name")
    start_time_hour: int = Field(..., ge=0, le=23, description="Start hour")
    start_time_minute: int = Field(default=0, ge=0, le=59, description="Start minute")
    duration_hours: float = Field(..., gt=0, le=24, description="Shift duration")
    days_active: List[int] = Field(
        default=[0, 1, 2, 3, 4],
        description="Active days (0=Monday)"
    )
    heat_load_factor: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Heat load multiplier"
    )
    ramp_up_minutes: int = Field(
        default=30,
        ge=0,
        description="Pre-shift heat ramp-up time"
    )
    ramp_down_minutes: int = Field(
        default=15,
        ge=0,
        description="Post-shift heat ramp-down time"
    )


class ShiftScheduleManager:
    """
    Manages production shift schedules.

    Tracks shift patterns and calculates heat load requirements
    based on production schedules.
    """

    def __init__(
        self,
        shifts: Optional[List[ProductionShift]] = None,
        baseline_load_kw: float = 1000.0,
    ) -> None:
        """
        Initialize shift schedule manager.

        Args:
            shifts: List of shift definitions
            baseline_load_kw: Baseline heat load
        """
        self._shifts = shifts or []
        self._baseline_load = baseline_load_kw

        # Index shifts by day
        self._shifts_by_day: Dict[int, List[ProductionShift]] = {}
        for shift in self._shifts:
            for day in shift.days_active:
                if day not in self._shifts_by_day:
                    self._shifts_by_day[day] = []
                self._shifts_by_day[day].append(shift)

        logger.info(
            f"ShiftScheduleManager initialized: {len(self._shifts)} shifts, "
            f"baseline={baseline_load_kw}kW"
        )

    def get_heat_load_at_time(
        self,
        target_time: datetime,
        additional_orders: Optional[List[ProductionOrder]] = None,
    ) -> float:
        """
        Calculate expected heat load at a given time.

        Args:
            target_time: Time to calculate load
            additional_orders: Additional production orders

        Returns:
            Expected heat load (kW)
        """
        load = 0.0
        day_of_week = target_time.weekday()
        hour = target_time.hour
        minute = target_time.minute

        # Check shifts
        shifts_today = self._shifts_by_day.get(day_of_week, [])
        for shift in shifts_today:
            if self._is_time_in_shift(target_time, shift):
                load += self._baseline_load * shift.heat_load_factor
            elif self._is_time_in_ramp_up(target_time, shift):
                # Ramping up - partial load
                load += self._baseline_load * shift.heat_load_factor * 0.5
            elif self._is_time_in_ramp_down(target_time, shift):
                # Ramping down - partial load
                load += self._baseline_load * shift.heat_load_factor * 0.3

        # Add orders
        if additional_orders:
            for order in additional_orders:
                if order.scheduled_start <= target_time <= order.scheduled_end:
                    load += order.heat_load_kw
                elif self._is_in_order_ramp(target_time, order):
                    load += order.heat_load_kw * 0.5

        return load

    def get_load_profile(
        self,
        start_time: datetime,
        horizon_hours: int,
        resolution_minutes: int = 15,
        orders: Optional[List[ProductionOrder]] = None,
    ) -> List[Tuple[datetime, float]]:
        """
        Generate heat load profile over horizon.

        Args:
            start_time: Profile start time
            horizon_hours: Profile horizon (hours)
            resolution_minutes: Time resolution (minutes)
            orders: Production orders to include

        Returns:
            List of (timestamp, load_kw) tuples
        """
        profile: List[Tuple[datetime, float]] = []
        n_steps = int(horizon_hours * 60 / resolution_minutes)

        for i in range(n_steps):
            t = start_time + timedelta(minutes=i * resolution_minutes)
            load = self.get_heat_load_at_time(t, orders)
            profile.append((t, load))

        return profile

    def _is_time_in_shift(
        self,
        target_time: datetime,
        shift: ProductionShift,
    ) -> bool:
        """Check if time is within shift working hours."""
        shift_start = target_time.replace(
            hour=shift.start_time_hour,
            minute=shift.start_time_minute,
            second=0,
            microsecond=0,
        )
        shift_end = shift_start + timedelta(hours=shift.duration_hours)

        return shift_start <= target_time < shift_end

    def _is_time_in_ramp_up(
        self,
        target_time: datetime,
        shift: ProductionShift,
    ) -> bool:
        """Check if time is in pre-shift ramp-up period."""
        shift_start = target_time.replace(
            hour=shift.start_time_hour,
            minute=shift.start_time_minute,
            second=0,
            microsecond=0,
        )
        ramp_start = shift_start - timedelta(minutes=shift.ramp_up_minutes)

        return ramp_start <= target_time < shift_start

    def _is_time_in_ramp_down(
        self,
        target_time: datetime,
        shift: ProductionShift,
    ) -> bool:
        """Check if time is in post-shift ramp-down period."""
        shift_start = target_time.replace(
            hour=shift.start_time_hour,
            minute=shift.start_time_minute,
            second=0,
            microsecond=0,
        )
        shift_end = shift_start + timedelta(hours=shift.duration_hours)
        ramp_end = shift_end + timedelta(minutes=shift.ramp_down_minutes)

        return shift_end <= target_time < ramp_end

    def _is_in_order_ramp(
        self,
        target_time: datetime,
        order: ProductionOrder,
    ) -> bool:
        """Check if time is in order ramp-up period."""
        ramp_start = order.scheduled_start - timedelta(minutes=order.ramp_up_time_minutes)
        return ramp_start <= target_time < order.scheduled_start


# =============================================================================
# ORDER SCHEDULER
# =============================================================================

class ProductionOrderScheduler:
    """
    Schedules production orders to optimize heat load profile.

    Uses priority-based scheduling with optional load balancing
    to minimize peak demand and energy costs.
    """

    def __init__(
        self,
        demand_limit_kw: Optional[float] = None,
        allow_rescheduling: bool = True,
    ) -> None:
        """
        Initialize order scheduler.

        Args:
            demand_limit_kw: Peak demand limit
            allow_rescheduling: Allow flexible order rescheduling
        """
        self._demand_limit = demand_limit_kw
        self._allow_rescheduling = allow_rescheduling

        logger.info(
            f"ProductionOrderScheduler initialized: "
            f"demand_limit={demand_limit_kw}kW, rescheduling={allow_rescheduling}"
        )

    def schedule_orders(
        self,
        orders: List[ProductionOrder],
        shift_manager: ShiftScheduleManager,
        horizon_start: datetime,
        horizon_end: datetime,
    ) -> ProductionScheduleResult:
        """
        Schedule production orders.

        Args:
            orders: Orders to schedule
            shift_manager: Shift schedule manager
            horizon_start: Scheduling horizon start
            horizon_end: Scheduling horizon end

        Returns:
            ProductionScheduleResult with scheduled orders
        """
        start_time = datetime.now(timezone.utc)
        logger.info(f"Scheduling {len(orders)} production orders")

        scheduled_orders: List[ProductionOrder] = []
        rescheduled_orders: List[ProductionOrder] = []

        # Sort orders by priority (highest first) and deadline
        sorted_orders = sorted(
            orders,
            key=lambda o: (-o.priority, o.latest_end or o.scheduled_end)
        )

        # Track load at each time slot
        load_profile: Dict[datetime, float] = {}

        for order in sorted_orders:
            # Check if order fits in original slot
            fits_original = self._order_fits_in_slot(
                order,
                order.scheduled_start,
                order.scheduled_end,
                load_profile,
                shift_manager,
            )

            if fits_original:
                # Keep original schedule
                scheduled_orders.append(order)
                self._add_order_to_profile(order, load_profile)
            elif self._allow_rescheduling and order.is_flexible:
                # Try to reschedule
                new_slot = self._find_best_slot(
                    order,
                    horizon_start,
                    horizon_end,
                    load_profile,
                    shift_manager,
                )

                if new_slot:
                    # Reschedule order
                    rescheduled_order = ProductionOrder(
                        order_id=order.order_id,
                        product_id=order.product_id,
                        product_name=order.product_name,
                        scheduled_start=new_slot[0],
                        scheduled_end=new_slot[1],
                        duration_hours=order.duration_hours,
                        heat_load_kw=order.heat_load_kw,
                        temperature_c=order.temperature_c,
                        ramp_up_time_minutes=order.ramp_up_time_minutes,
                        priority=order.priority,
                        is_flexible=order.is_flexible,
                        earliest_start=order.earliest_start,
                        latest_end=order.latest_end,
                        status=ProductionStatus.SCHEDULED,
                    )
                    scheduled_orders.append(rescheduled_order)
                    rescheduled_orders.append(rescheduled_order)
                    self._add_order_to_profile(rescheduled_order, load_profile)
                else:
                    # Could not reschedule - keep original (may exceed limit)
                    scheduled_orders.append(order)
                    self._add_order_to_profile(order, load_profile)
                    logger.warning(f"Could not reschedule order {order.order_id}")
            else:
                # Non-flexible order - keep original
                scheduled_orders.append(order)
                self._add_order_to_profile(order, load_profile)

        # Calculate metrics
        total_heat_kwh = sum(
            o.heat_load_kw * o.duration_hours for o in scheduled_orders
        )

        # Estimate cost savings from rescheduling
        scheduling_savings = self._estimate_rescheduling_savings(rescheduled_orders)

        result = ProductionScheduleResult(
            timestamp=datetime.now(timezone.utc),
            scheduled_orders=scheduled_orders,
            rescheduled_orders=rescheduled_orders,
            total_orders=len(scheduled_orders),
            orders_on_time=len(scheduled_orders) - len(rescheduled_orders),
            orders_rescheduled=len(rescheduled_orders),
            total_heat_load_kwh=round(total_heat_kwh, 2),
            scheduling_cost_savings_usd=round(scheduling_savings, 2),
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            f"Order scheduling complete: {len(scheduled_orders)} scheduled, "
            f"{len(rescheduled_orders)} rescheduled ({processing_time:.1f}ms)"
        )

        return result

    def _order_fits_in_slot(
        self,
        order: ProductionOrder,
        start: datetime,
        end: datetime,
        current_profile: Dict[datetime, float],
        shift_manager: ShiftScheduleManager,
    ) -> bool:
        """Check if order fits in slot without exceeding limit."""
        if self._demand_limit is None:
            return True

        # Check load at each 15-minute interval
        interval = timedelta(minutes=15)
        t = start
        while t < end:
            base_load = shift_manager.get_heat_load_at_time(t)
            existing_load = current_profile.get(t, 0.0)
            total_load = base_load + existing_load + order.heat_load_kw

            if total_load > self._demand_limit:
                return False
            t += interval

        return True

    def _find_best_slot(
        self,
        order: ProductionOrder,
        horizon_start: datetime,
        horizon_end: datetime,
        current_profile: Dict[datetime, float],
        shift_manager: ShiftScheduleManager,
    ) -> Optional[Tuple[datetime, datetime]]:
        """Find best available slot for order."""
        earliest = order.earliest_start or horizon_start
        latest = order.latest_end or horizon_end

        # Try each 15-minute slot
        interval = timedelta(minutes=15)
        t = earliest
        best_slot = None
        best_cost = float('inf')

        while t + timedelta(hours=order.duration_hours) <= latest:
            end = t + timedelta(hours=order.duration_hours)

            if self._order_fits_in_slot(order, t, end, current_profile, shift_manager):
                # Calculate slot cost (simplified - prefer off-peak)
                cost = self._calculate_slot_cost(t, order.duration_hours)

                if cost < best_cost:
                    best_cost = cost
                    best_slot = (t, end)

            t += interval

        return best_slot

    def _add_order_to_profile(
        self,
        order: ProductionOrder,
        profile: Dict[datetime, float],
    ) -> None:
        """Add order load to profile."""
        interval = timedelta(minutes=15)
        t = order.scheduled_start
        while t < order.scheduled_end:
            profile[t] = profile.get(t, 0.0) + order.heat_load_kw
            t += interval

    def _calculate_slot_cost(
        self,
        start_time: datetime,
        duration_hours: float,
    ) -> float:
        """Calculate relative cost of slot (lower is better)."""
        # Prefer off-peak hours
        cost = 0.0
        peak_hours = {14, 15, 16, 17, 18, 19}

        t = start_time
        while t < start_time + timedelta(hours=duration_hours):
            if t.hour in peak_hours:
                cost += 2.0  # Higher cost for peak
            else:
                cost += 1.0  # Base cost for off-peak
            t += timedelta(minutes=15)

        return cost

    def _estimate_rescheduling_savings(
        self,
        rescheduled_orders: List[ProductionOrder],
    ) -> float:
        """Estimate cost savings from rescheduling."""
        # Simplified: $10 per order moved to off-peak
        savings = 0.0
        for order in rescheduled_orders:
            if order.scheduled_start.hour < 14 or order.scheduled_start.hour >= 20:
                savings += 10.0 * order.heat_load_kw / 1000.0  # Scale by load

        return savings


# =============================================================================
# ERP INTEGRATION
# =============================================================================

class ERPConnector:
    """
    Connector for ERP system integration.

    Supports fetching production orders from SAP, Oracle, and other
    ERP systems via standardized APIs.
    """

    def __init__(
        self,
        erp_type: str = "generic",
        api_endpoint: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> None:
        """
        Initialize ERP connector.

        Args:
            erp_type: ERP system type
            api_endpoint: API endpoint URL
            auth_token: Authentication token
        """
        self._erp_type = erp_type
        self._endpoint = api_endpoint
        self._token = auth_token
        self._connected = False

        logger.info(f"ERPConnector initialized for {erp_type}")

    async def connect(self) -> bool:
        """
        Connect to ERP system.

        Returns:
            True if connection successful
        """
        # Placeholder - would implement actual connection
        logger.info(f"Connecting to {self._erp_type} ERP...")
        self._connected = True
        return True

    async def fetch_production_orders(
        self,
        start_date: datetime,
        end_date: datetime,
        work_center: Optional[str] = None,
    ) -> List[ProductionOrder]:
        """
        Fetch production orders from ERP.

        Args:
            start_date: Start of date range
            end_date: End of date range
            work_center: Optional work center filter

        Returns:
            List of ProductionOrder objects
        """
        if not self._connected:
            logger.warning("ERP not connected")
            return []

        # Placeholder - would implement actual ERP query
        logger.info(
            f"Fetching orders from {start_date} to {end_date}"
        )

        # Return empty list - in production would query ERP
        return []

    async def update_order_schedule(
        self,
        order_id: str,
        new_start: datetime,
        new_end: datetime,
    ) -> bool:
        """
        Update order schedule in ERP.

        Args:
            order_id: Order to update
            new_start: New start time
            new_end: New end time

        Returns:
            True if update successful
        """
        if not self._connected:
            logger.warning("ERP not connected")
            return False

        # Placeholder - would implement actual ERP update
        logger.info(f"Updating order {order_id} in ERP")
        return True

    async def disconnect(self) -> None:
        """Disconnect from ERP system."""
        self._connected = False
        logger.info("Disconnected from ERP")


# =============================================================================
# PRODUCTION PLANNER
# =============================================================================

class ProductionPlanner:
    """
    Main production planning coordinator.

    Integrates shift scheduling, order management, and ERP
    connectivity for comprehensive production heat planning.
    """

    def __init__(
        self,
        shifts: Optional[List[ProductionShift]] = None,
        baseline_load_kw: float = 1000.0,
        demand_limit_kw: Optional[float] = None,
        erp_connector: Optional[ERPConnector] = None,
    ) -> None:
        """
        Initialize production planner.

        Args:
            shifts: Production shifts
            baseline_load_kw: Baseline heat load
            demand_limit_kw: Demand limit
            erp_connector: Optional ERP connector
        """
        self._shift_manager = ShiftScheduleManager(shifts, baseline_load_kw)
        self._order_scheduler = ProductionOrderScheduler(
            demand_limit_kw=demand_limit_kw,
            allow_rescheduling=True,
        )
        self._erp = erp_connector

        # Cache
        self._cached_orders: List[ProductionOrder] = []
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_minutes = 15

        logger.info("ProductionPlanner initialized")

    def get_heat_load_profile(
        self,
        start_time: datetime,
        horizon_hours: int,
        resolution_minutes: int = 15,
    ) -> Dict[datetime, float]:
        """
        Get expected heat load profile.

        Args:
            start_time: Profile start
            horizon_hours: Horizon (hours)
            resolution_minutes: Resolution (minutes)

        Returns:
            Dict mapping timestamp to load (kW)
        """
        profile = self._shift_manager.get_load_profile(
            start_time=start_time,
            horizon_hours=horizon_hours,
            resolution_minutes=resolution_minutes,
            orders=self._cached_orders,
        )

        return {t: load for t, load in profile}

    def schedule_orders(
        self,
        orders: List[ProductionOrder],
        horizon_start: datetime,
        horizon_end: datetime,
    ) -> ProductionScheduleResult:
        """
        Schedule production orders.

        Args:
            orders: Orders to schedule
            horizon_start: Horizon start
            horizon_end: Horizon end

        Returns:
            ProductionScheduleResult
        """
        return self._order_scheduler.schedule_orders(
            orders=orders,
            shift_manager=self._shift_manager,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
        )

    async def sync_with_erp(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> int:
        """
        Synchronize orders from ERP.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Number of orders fetched
        """
        if self._erp is None:
            return 0

        orders = await self._erp.fetch_production_orders(start_date, end_date)
        self._cached_orders = orders
        self._cache_timestamp = datetime.now(timezone.utc)

        logger.info(f"Synced {len(orders)} orders from ERP")
        return len(orders)

    def add_manual_order(self, order: ProductionOrder) -> None:
        """Add a manually created order."""
        self._cached_orders.append(order)

    def get_scheduled_orders(self) -> List[ProductionOrder]:
        """Get all scheduled orders."""
        return self._cached_orders.copy()

    def generate_schedule_actions(
        self,
        orders: List[ProductionOrder],
    ) -> List[ScheduleActionItem]:
        """
        Generate schedule actions for orders.

        Args:
            orders: Scheduled orders

        Returns:
            List of schedule actions
        """
        actions: List[ScheduleActionItem] = []

        for order in orders:
            # Preheat action
            preheat_time = order.scheduled_start - timedelta(
                minutes=order.ramp_up_time_minutes
            )
            actions.append(ScheduleActionItem(
                timestamp=preheat_time,
                action_type=ScheduleAction.RAMP_UP,
                power_setpoint_kw=order.heat_load_kw,
                temperature_setpoint_c=order.temperature_c,
                duration_minutes=order.ramp_up_time_minutes,
                reason=f"Preheat for order {order.order_id}",
                priority=order.priority,
            ))

            # Start action
            actions.append(ScheduleActionItem(
                timestamp=order.scheduled_start,
                action_type=ScheduleAction.START,
                power_setpoint_kw=order.heat_load_kw,
                temperature_setpoint_c=order.temperature_c,
                duration_minutes=int(order.duration_hours * 60),
                reason=f"Start order {order.order_id}",
                priority=order.priority,
            ))

            # Stop action
            actions.append(ScheduleActionItem(
                timestamp=order.scheduled_end,
                action_type=ScheduleAction.RAMP_DOWN,
                power_setpoint_kw=0,
                duration_minutes=15,
                reason=f"Complete order {order.order_id}",
                priority=order.priority,
            ))

        return sorted(actions, key=lambda a: a.timestamp)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ProductionShift",
    "ShiftScheduleManager",
    "ProductionOrderScheduler",
    "ERPConnector",
    "ProductionPlanner",
]
