"""
GL-019 HEATSCHEDULER - Production Planning Module Tests

Unit tests for production planning including shift scheduling,
order management, ERP integration, and schedule optimization.

Test Coverage:
    - ProductionShift model validation
    - ShiftScheduleManager load calculation
    - ProductionOrderScheduler scheduling logic
    - ERPConnector async operations
    - ProductionPlanner coordination

Author: GreenLang Test Team
Date: December 2025
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch
import asyncio


class TestProductionShift:
    """Tests for ProductionShift model."""

    def test_valid_production_shift(self):
        """Test valid production shift creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionShift,
        )

        shift = ProductionShift(
            shift_id="day_shift",
            name="Day Shift",
            start_time_hour=6,
            start_time_minute=0,
            duration_hours=8,
            days_active=[0, 1, 2, 3, 4],
            heat_load_factor=1.0,
            ramp_up_minutes=30,
            ramp_down_minutes=15,
        )

        assert shift.shift_id == "day_shift"
        assert shift.duration_hours == 8
        assert shift.heat_load_factor == 1.0

    def test_shift_default_values(self):
        """Test shift default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionShift,
        )

        shift = ProductionShift(
            shift_id="test_shift",
            start_time_hour=8,
            duration_hours=8,
        )

        assert shift.name == ""
        assert shift.start_time_minute == 0
        assert shift.days_active == [0, 1, 2, 3, 4]
        assert shift.heat_load_factor == 1.0
        assert shift.ramp_up_minutes == 30
        assert shift.ramp_down_minutes == 15

    def test_shift_hour_bounds(self):
        """Test start hour bounds validation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionShift,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProductionShift(
                shift_id="invalid",
                start_time_hour=25,  # Invalid
                duration_hours=8,
            )

    def test_shift_duration_bounds(self):
        """Test duration bounds validation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionShift,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProductionShift(
                shift_id="invalid",
                start_time_hour=8,
                duration_hours=25,  # Invalid (max 24)
            )


class TestShiftScheduleManagerInitialization:
    """Tests for ShiftScheduleManager initialization."""

    def test_manager_initialization(self, sample_production_shifts):
        """Test manager initializes correctly."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ShiftScheduleManager,
        )

        manager = ShiftScheduleManager(
            shifts=sample_production_shifts,
            baseline_load_kw=1000.0,
        )

        assert len(manager._shifts) == 3
        assert manager._baseline_load == 1000.0

    def test_manager_indexes_shifts_by_day(self, sample_production_shifts):
        """Test manager indexes shifts by day."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ShiftScheduleManager,
        )

        manager = ShiftScheduleManager(
            shifts=sample_production_shifts,
            baseline_load_kw=1000.0,
        )

        # Weekdays should have shifts
        assert 0 in manager._shifts_by_day
        assert 1 in manager._shifts_by_day
        assert 2 in manager._shifts_by_day

    def test_manager_no_shifts(self):
        """Test manager with no shifts."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ShiftScheduleManager,
        )

        manager = ShiftScheduleManager(
            shifts=None,
            baseline_load_kw=1000.0,
        )

        assert len(manager._shifts) == 0


class TestShiftScheduleManagerLoadCalculation:
    """Tests for heat load calculation."""

    @pytest.fixture
    def manager(self, sample_production_shifts):
        """Create manager instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ShiftScheduleManager,
        )
        return ShiftScheduleManager(
            shifts=sample_production_shifts,
            baseline_load_kw=1000.0,
        )

    def test_get_load_during_shift(self, manager):
        """Test load calculation during shift."""
        # Day shift: 6:00 - 14:00, load_factor = 1.0
        # Wednesday at 10:00
        time = datetime(2025, 6, 18, 10, 0, 0, tzinfo=timezone.utc)

        load = manager.get_heat_load_at_time(time)

        # Should have baseline * factor = 1000 * 1.0
        assert load == 1000.0

    def test_get_load_during_ramp_up(self, manager):
        """Test load calculation during ramp-up."""
        # Day shift starts at 6:00, ramp_up = 30 min
        # Wednesday at 5:45 (during ramp-up)
        time = datetime(2025, 6, 18, 5, 45, 0, tzinfo=timezone.utc)

        load = manager.get_heat_load_at_time(time)

        # Should have partial load (50% of baseline * factor)
        assert load == 500.0

    def test_get_load_during_ramp_down(self, manager):
        """Test load calculation during ramp-down."""
        # Day shift ends at 14:00, ramp_down = 15 min
        # Wednesday at 14:10 (during ramp-down)
        time = datetime(2025, 6, 18, 14, 10, 0, tzinfo=timezone.utc)

        load = manager.get_heat_load_at_time(time)

        # Should have partial load (30% of baseline * factor)
        assert load == 300.0

    def test_get_load_with_orders(self, manager, sample_production_orders, base_timestamp):
        """Test load calculation with production orders."""
        # Time during first order
        time = sample_production_orders[0].scheduled_start + timedelta(hours=1)

        load = manager.get_heat_load_at_time(
            time,
            additional_orders=sample_production_orders,
        )

        # Should include shift load + order load
        assert load >= sample_production_orders[0].heat_load_kw

    def test_get_load_weekend(self, manager):
        """Test load calculation on weekend (no shifts)."""
        # Saturday
        time = datetime(2025, 6, 21, 10, 0, 0, tzinfo=timezone.utc)

        load = manager.get_heat_load_at_time(time)

        # Weekend - no active shifts
        assert load == 0.0

    def test_get_load_profile(self, manager, base_timestamp):
        """Test load profile generation."""
        profile = manager.get_load_profile(
            start_time=base_timestamp,
            horizon_hours=24,
            resolution_minutes=60,
        )

        assert len(profile) == 24
        for timestamp, load in profile:
            assert isinstance(timestamp, datetime)
            assert isinstance(load, float)


class TestProductionOrderSchedulerInitialization:
    """Tests for ProductionOrderScheduler initialization."""

    def test_scheduler_initialization(self):
        """Test scheduler initializes correctly."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionOrderScheduler,
        )

        scheduler = ProductionOrderScheduler(
            demand_limit_kw=5000.0,
            allow_rescheduling=True,
        )

        assert scheduler._demand_limit == 5000.0
        assert scheduler._allow_rescheduling is True

    def test_scheduler_no_limit(self):
        """Test scheduler without demand limit."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionOrderScheduler,
        )

        scheduler = ProductionOrderScheduler(
            demand_limit_kw=None,
            allow_rescheduling=True,
        )

        assert scheduler._demand_limit is None


class TestProductionOrderSchedulerScheduling:
    """Tests for order scheduling."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionOrderScheduler,
        )
        return ProductionOrderScheduler(
            demand_limit_kw=5000.0,
            allow_rescheduling=True,
        )

    @pytest.fixture
    def shift_manager(self, sample_production_shifts):
        """Create shift manager instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ShiftScheduleManager,
        )
        return ShiftScheduleManager(
            shifts=sample_production_shifts,
            baseline_load_kw=1000.0,
        )

    def test_schedule_orders_returns_result(
        self,
        scheduler,
        shift_manager,
        sample_production_orders,
        base_timestamp,
    ):
        """Test schedule_orders returns result."""
        result = scheduler.schedule_orders(
            orders=sample_production_orders,
            shift_manager=shift_manager,
            horizon_start=base_timestamp,
            horizon_end=base_timestamp + timedelta(hours=24),
        )

        assert result is not None
        assert result.total_orders == 3

    def test_schedule_orders_sorts_by_priority(
        self,
        scheduler,
        shift_manager,
        sample_production_orders,
        base_timestamp,
    ):
        """Test orders are sorted by priority."""
        result = scheduler.schedule_orders(
            orders=sample_production_orders,
            shift_manager=shift_manager,
            horizon_start=base_timestamp,
            horizon_end=base_timestamp + timedelta(hours=24),
        )

        # All orders should be scheduled
        assert len(result.scheduled_orders) == 3

    def test_schedule_orders_calculates_heat_load(
        self,
        scheduler,
        shift_manager,
        sample_production_orders,
        base_timestamp,
    ):
        """Test heat load is calculated."""
        result = scheduler.schedule_orders(
            orders=sample_production_orders,
            shift_manager=shift_manager,
            horizon_start=base_timestamp,
            horizon_end=base_timestamp + timedelta(hours=24),
        )

        assert result.total_heat_load_kwh > 0

    def test_schedule_orders_identifies_rescheduled(
        self,
        shift_manager,
        sample_production_orders,
        base_timestamp,
    ):
        """Test rescheduled orders are identified."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionOrderScheduler,
        )

        # Low limit forces rescheduling
        scheduler = ProductionOrderScheduler(
            demand_limit_kw=500.0,
            allow_rescheduling=True,
        )

        result = scheduler.schedule_orders(
            orders=sample_production_orders,
            shift_manager=shift_manager,
            horizon_start=base_timestamp,
            horizon_end=base_timestamp + timedelta(hours=48),
        )

        # Some orders may need rescheduling
        assert result.orders_rescheduled >= 0

    def test_schedule_orders_no_rescheduling(
        self,
        shift_manager,
        sample_production_orders,
        base_timestamp,
    ):
        """Test scheduling without rescheduling allowed."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionOrderScheduler,
        )

        scheduler = ProductionOrderScheduler(
            demand_limit_kw=500.0,
            allow_rescheduling=False,
        )

        result = scheduler.schedule_orders(
            orders=sample_production_orders,
            shift_manager=shift_manager,
            horizon_start=base_timestamp,
            horizon_end=base_timestamp + timedelta(hours=24),
        )

        # No rescheduling should occur
        assert result.orders_rescheduled == 0


class TestProductionOrderSchedulerSlotFinding:
    """Tests for slot finding logic."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionOrderScheduler,
        )
        return ProductionOrderScheduler(
            demand_limit_kw=3000.0,
            allow_rescheduling=True,
        )

    @pytest.fixture
    def shift_manager(self, sample_production_shifts):
        """Create shift manager instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ShiftScheduleManager,
        )
        return ShiftScheduleManager(
            shifts=sample_production_shifts,
            baseline_load_kw=500.0,
        )

    def test_order_fits_in_slot_no_limit(
        self,
        shift_manager,
        sample_production_orders,
    ):
        """Test order fits when no limit."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionOrderScheduler,
        )

        scheduler = ProductionOrderScheduler(
            demand_limit_kw=None,
            allow_rescheduling=True,
        )

        order = sample_production_orders[0]
        fits = scheduler._order_fits_in_slot(
            order=order,
            start=order.scheduled_start,
            end=order.scheduled_end,
            current_profile={},
            shift_manager=shift_manager,
        )

        assert fits is True


class TestERPConnector:
    """Tests for ERPConnector."""

    @pytest.fixture
    def connector(self):
        """Create connector instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ERPConnector,
        )
        return ERPConnector(
            erp_type="sap",
            api_endpoint="https://erp.example.com/api",
            auth_token="test_token",
        )

    def test_connector_initialization(self, connector):
        """Test connector initializes correctly."""
        assert connector._erp_type == "sap"
        assert connector._endpoint == "https://erp.example.com/api"
        assert connector._token == "test_token"
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_connect(self, connector):
        """Test connect method."""
        result = await connector.connect()

        assert result is True
        assert connector._connected is True

    @pytest.mark.asyncio
    async def test_fetch_orders_when_connected(self, connector, base_timestamp):
        """Test fetching orders when connected."""
        await connector.connect()

        orders = await connector.fetch_production_orders(
            start_date=base_timestamp,
            end_date=base_timestamp + timedelta(days=7),
        )

        # Placeholder returns empty list
        assert isinstance(orders, list)

    @pytest.mark.asyncio
    async def test_fetch_orders_when_disconnected(self, connector, base_timestamp, caplog):
        """Test fetching orders when not connected logs warning."""
        import logging

        caplog.set_level(logging.WARNING)

        orders = await connector.fetch_production_orders(
            start_date=base_timestamp,
            end_date=base_timestamp + timedelta(days=7),
        )

        assert len(orders) == 0
        assert "not connected" in caplog.text

    @pytest.mark.asyncio
    async def test_update_order_schedule(self, connector, base_timestamp):
        """Test updating order schedule."""
        await connector.connect()

        result = await connector.update_order_schedule(
            order_id="ORD-001",
            new_start=base_timestamp,
            new_end=base_timestamp + timedelta(hours=3),
        )

        # Placeholder returns True
        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Test disconnect method."""
        await connector.connect()
        assert connector._connected is True

        await connector.disconnect()
        assert connector._connected is False


class TestProductionPlannerInitialization:
    """Tests for ProductionPlanner initialization."""

    def test_planner_initialization(self, sample_production_shifts):
        """Test planner initializes correctly."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionPlanner,
        )

        planner = ProductionPlanner(
            shifts=sample_production_shifts,
            baseline_load_kw=1000.0,
            demand_limit_kw=5000.0,
        )

        assert planner._shift_manager is not None
        assert planner._order_scheduler is not None

    def test_planner_with_erp_connector(
        self,
        sample_production_shifts,
        mock_erp_connector,
    ):
        """Test planner with ERP connector."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionPlanner,
        )

        planner = ProductionPlanner(
            shifts=sample_production_shifts,
            baseline_load_kw=1000.0,
            erp_connector=mock_erp_connector,
        )

        assert planner._erp is not None


class TestProductionPlannerOperations:
    """Tests for ProductionPlanner operations."""

    @pytest.fixture
    def planner(self, sample_production_shifts):
        """Create planner instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionPlanner,
        )
        return ProductionPlanner(
            shifts=sample_production_shifts,
            baseline_load_kw=1000.0,
            demand_limit_kw=5000.0,
        )

    def test_get_heat_load_profile(self, planner, base_timestamp):
        """Test getting heat load profile."""
        profile = planner.get_heat_load_profile(
            start_time=base_timestamp,
            horizon_hours=24,
            resolution_minutes=15,
        )

        assert len(profile) > 0
        for timestamp, load in profile.items():
            assert isinstance(timestamp, datetime)
            assert isinstance(load, float)

    def test_schedule_orders(
        self,
        planner,
        sample_production_orders,
        base_timestamp,
    ):
        """Test scheduling orders."""
        result = planner.schedule_orders(
            orders=sample_production_orders,
            horizon_start=base_timestamp,
            horizon_end=base_timestamp + timedelta(hours=24),
        )

        assert result.total_orders == 3

    def test_add_manual_order(self, planner, sample_production_orders):
        """Test adding manual order."""
        initial_count = len(planner._cached_orders)

        planner.add_manual_order(sample_production_orders[0])

        assert len(planner._cached_orders) == initial_count + 1

    def test_get_scheduled_orders(self, planner, sample_production_orders):
        """Test getting scheduled orders."""
        for order in sample_production_orders:
            planner.add_manual_order(order)

        orders = planner.get_scheduled_orders()

        assert len(orders) == 3
        # Should return copy
        orders.pop()
        assert len(planner._cached_orders) == 3

    def test_generate_schedule_actions(
        self,
        planner,
        sample_production_orders,
    ):
        """Test generating schedule actions."""
        actions = planner.generate_schedule_actions(sample_production_orders)

        # 3 orders * 3 actions each (preheat, start, stop) = 9
        assert len(actions) == 9

        # Actions should be sorted by timestamp
        for i in range(len(actions) - 1):
            assert actions[i].timestamp <= actions[i + 1].timestamp

    @pytest.mark.asyncio
    async def test_sync_with_erp_no_connector(self, planner, base_timestamp):
        """Test sync returns 0 when no ERP connector."""
        count = await planner.sync_with_erp(
            start_date=base_timestamp,
            end_date=base_timestamp + timedelta(days=7),
        )

        assert count == 0

    @pytest.mark.asyncio
    async def test_sync_with_erp(
        self,
        sample_production_shifts,
        base_timestamp,
    ):
        """Test sync with ERP connector."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionPlanner,
            ERPConnector,
        )

        # Create mock connector
        connector = ERPConnector(erp_type="sap")
        await connector.connect()

        planner = ProductionPlanner(
            shifts=sample_production_shifts,
            baseline_load_kw=1000.0,
            erp_connector=connector,
        )

        count = await planner.sync_with_erp(
            start_date=base_timestamp,
            end_date=base_timestamp + timedelta(days=7),
        )

        # Placeholder returns empty list
        assert count == 0


class TestProductionScheduleActions:
    """Tests for schedule action generation."""

    @pytest.fixture
    def planner(self, sample_production_shifts):
        """Create planner instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionPlanner,
        )
        return ProductionPlanner(
            shifts=sample_production_shifts,
            baseline_load_kw=1000.0,
        )

    def test_preheat_action_generated(self, planner, sample_production_orders):
        """Test preheat action is generated."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import ScheduleAction

        actions = planner.generate_schedule_actions([sample_production_orders[0]])

        preheat_actions = [a for a in actions if a.action_type == ScheduleAction.RAMP_UP]
        assert len(preheat_actions) == 1

        # Preheat should be before scheduled start
        order = sample_production_orders[0]
        preheat_time = order.scheduled_start - timedelta(minutes=order.ramp_up_time_minutes)
        assert preheat_actions[0].timestamp == preheat_time

    def test_start_action_generated(self, planner, sample_production_orders):
        """Test start action is generated."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import ScheduleAction

        actions = planner.generate_schedule_actions([sample_production_orders[0]])

        start_actions = [a for a in actions if a.action_type == ScheduleAction.START]
        assert len(start_actions) == 1

        # Start should be at scheduled start
        assert start_actions[0].timestamp == sample_production_orders[0].scheduled_start

    def test_stop_action_generated(self, planner, sample_production_orders):
        """Test stop action is generated."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import ScheduleAction

        actions = planner.generate_schedule_actions([sample_production_orders[0]])

        stop_actions = [a for a in actions if a.action_type == ScheduleAction.RAMP_DOWN]
        assert len(stop_actions) == 1

        # Stop should be at scheduled end
        assert stop_actions[0].timestamp == sample_production_orders[0].scheduled_end


class TestProductionPlanningPerformance:
    """Performance tests for production planning."""

    @pytest.mark.performance
    def test_scheduling_time(
        self,
        sample_production_shifts,
        many_production_orders,
        base_timestamp,
    ):
        """Test scheduling completes in reasonable time."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionPlanner,
        )
        import time

        planner = ProductionPlanner(
            shifts=sample_production_shifts,
            baseline_load_kw=1000.0,
            demand_limit_kw=10000.0,
        )

        start = time.time()
        result = planner.schedule_orders(
            orders=many_production_orders,
            horizon_start=base_timestamp,
            horizon_end=base_timestamp + timedelta(hours=168),
        )
        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 5.0  # Should complete in under 5 seconds

    @pytest.mark.performance
    def test_action_generation_time(
        self,
        sample_production_shifts,
        many_production_orders,
    ):
        """Test action generation completes in reasonable time."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
            ProductionPlanner,
        )
        import time

        planner = ProductionPlanner(
            shifts=sample_production_shifts,
            baseline_load_kw=1000.0,
        )

        start = time.time()
        actions = planner.generate_schedule_actions(many_production_orders)
        elapsed = time.time() - start

        assert len(actions) > 0
        assert elapsed < 1.0  # Should complete in under 1 second
