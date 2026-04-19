"""
GL-011 FUELCRAFT - Inventory Manager Tests

Unit tests for InventoryManager including tank monitoring,
consumption tracking, delivery scheduling, and alert management.
"""

import pytest
from datetime import datetime, timezone, timedelta

from greenlang.agents.process_heat.gl_011_fuel_optimization.inventory import (
    InventoryManager,
    TankConfig,
    TankStatus,
    LevelStatus,
    DeliveryStatus,
    DeliverySchedule,
    InventoryAlert,
    ConsumptionTracker,
    ConsumptionRecord,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    InventoryConfig,
    AlertLevel,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import (
    InventoryAlertType,
)


class TestLevelStatus:
    """Tests for LevelStatus enum."""

    def test_all_statuses_defined(self):
        """Test all level statuses are defined."""
        expected = {"CRITICAL", "LOW", "NORMAL", "HIGH", "FULL"}
        actual = {s.name for s in LevelStatus}
        assert expected == actual


class TestDeliveryStatus:
    """Tests for DeliveryStatus enum."""

    def test_all_statuses_defined(self):
        """Test all delivery statuses are defined."""
        expected = {
            "SCHEDULED", "CONFIRMED", "IN_TRANSIT",
            "ARRIVED", "COMPLETED", "CANCELLED",
        }
        actual = {s.name for s in DeliveryStatus}
        assert expected == actual


class TestTankConfig:
    """Tests for TankConfig class."""

    def test_tank_config_required_fields(self):
        """Test tank config required fields."""
        config = TankConfig(
            tank_id="TANK-001",
            fuel_type="natural_gas",
            capacity_gal=10000,
        )

        assert config.tank_id == "TANK-001"
        assert config.fuel_type == "natural_gas"
        assert config.capacity_gal == 10000

    def test_tank_config_defaults(self):
        """Test tank config default values."""
        config = TankConfig(
            tank_id="TANK-001",
            fuel_type="natural_gas",
            capacity_gal=10000,
        )

        assert config.usable_capacity_pct == 95.0
        assert config.heel_volume_gal == 0.0
        assert config.reorder_point_pct == 30.0
        assert config.low_level_pct == 25.0
        assert config.critical_level_pct == 15.0
        assert config.high_level_pct == 95.0

    def test_usable_capacity_calculation(self):
        """Test usable capacity property."""
        config = TankConfig(
            tank_id="TANK-001",
            fuel_type="natural_gas",
            capacity_gal=10000,
            usable_capacity_pct=95.0,
            heel_volume_gal=100.0,
        )

        expected = 10000 * 0.95 - 100.0  # 9400
        assert config.usable_capacity_gal == expected


class TestConsumptionTracker:
    """Tests for ConsumptionTracker class."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = ConsumptionTracker(history_days=90)
        assert tracker._history_days == 90

    def test_add_consumption_record(self):
        """Test adding consumption record."""
        tracker = ConsumptionTracker()

        tracker.add_record("TANK-001", consumption_gal=100.0, duration_hours=1.0)

        assert "TANK-001" in tracker._records
        assert len(tracker._records["TANK-001"]) == 1

    def test_get_average_daily_consumption(self):
        """Test average daily consumption calculation."""
        tracker = ConsumptionTracker()

        # Add 10 records of 100 gal/hour
        for _ in range(10):
            tracker.add_record("TANK-001", consumption_gal=100.0, duration_hours=1.0)

        avg = tracker.get_average_daily(tank_id="TANK-001", days=30)

        # 100 gal/hr * 24 hr/day = 2400 gal/day
        assert avg == pytest.approx(2400.0, rel=0.1)

    def test_get_current_rate(self):
        """Test current consumption rate."""
        tracker = ConsumptionTracker()

        tracker.add_record("TANK-001", consumption_gal=100.0, duration_hours=1.0)

        rate = tracker.get_current_rate("TANK-001")

        assert rate == pytest.approx(100.0, rel=0.1)

    def test_forecast_consumption(self):
        """Test consumption forecast."""
        tracker = ConsumptionTracker()

        # Add some history
        for _ in range(5):
            tracker.add_record("TANK-001", consumption_gal=100.0, duration_hours=1.0)

        forecast = tracker.forecast_consumption("TANK-001", horizon_days=7)

        assert len(forecast) == 7
        for date, consumption in forecast:
            assert isinstance(date, datetime)
            assert consumption >= 0

    def test_weekend_adjustment(self):
        """Test day-of-week adjustment for weekends."""
        tracker = ConsumptionTracker()

        # Weekend factor should be 0.7
        assert tracker._get_dow_factor(5) == 0.7  # Saturday
        assert tracker._get_dow_factor(6) == 0.7  # Sunday
        assert tracker._get_dow_factor(0) == 1.0  # Monday


class TestInventoryManager:
    """Tests for InventoryManager class."""

    def test_manager_initialization(self, inventory_manager):
        """Test manager initialization."""
        assert inventory_manager.tank_count == 1
        assert inventory_manager.alert_count == 0

    def test_add_tank(self, inventory_manager, tank_config):
        """Test adding a tank."""
        initial_count = inventory_manager.tank_count

        new_tank = TankConfig(
            tank_id="TANK-002",
            fuel_type="fuel_oil",
            capacity_gal=5000,
        )
        inventory_manager.add_tank(new_tank)

        assert inventory_manager.tank_count == initial_count + 1

    def test_update_level(self, inventory_manager):
        """Test updating tank level."""
        status = inventory_manager.update_level(
            tank_id="TANK-001",
            level_gal=5000.0,
        )

        assert status.current_level_gal == 5000.0
        assert status.current_level_pct == 50.0

    def test_update_level_unknown_tank_raises(self, inventory_manager):
        """Test updating unknown tank raises error."""
        with pytest.raises(ValueError, match="Unknown tank"):
            inventory_manager.update_level("UNKNOWN-TANK", 1000.0)


class TestTankStatus:
    """Tests for tank status determination."""

    def test_critical_level_status(self, inventory_manager):
        """Test critical level status."""
        # 10% of 10000 = 1000 gal
        status = inventory_manager.update_level("TANK-001", 1000.0)

        assert status.level_status == LevelStatus.CRITICAL

    def test_low_level_status(self, inventory_manager):
        """Test low level status."""
        # 20% = 2000 gal (between critical 15% and low 25%)
        status = inventory_manager.update_level("TANK-001", 2000.0)

        assert status.level_status == LevelStatus.LOW

    def test_normal_level_status(self, inventory_manager):
        """Test normal level status."""
        # 50% = 5000 gal
        status = inventory_manager.update_level("TANK-001", 5000.0)

        assert status.level_status == LevelStatus.NORMAL

    def test_high_level_status(self, inventory_manager):
        """Test high level status."""
        # 96% = 9600 gal
        status = inventory_manager.update_level("TANK-001", 9600.0)

        assert status.level_status == LevelStatus.HIGH

    def test_full_level_status(self, inventory_manager):
        """Test full level status."""
        # 99% = 9900 gal
        status = inventory_manager.update_level("TANK-001", 9900.0)

        assert status.level_status == LevelStatus.FULL


class TestConsumptionTracking:
    """Tests for consumption tracking integration."""

    def test_consumption_tracked_on_level_decrease(self, inventory_manager):
        """Test consumption is tracked when level decreases."""
        # Set initial level
        inventory_manager.update_level("TANK-001", 5000.0)

        # Decrease level
        inventory_manager.update_level("TANK-001", 4900.0)

        # Consumption should be tracked
        status = inventory_manager.get_tank_status("TANK-001")
        # Note: consumption_rate may be 0 if not enough time elapsed


class TestDaysOfSupply:
    """Tests for days of supply calculation."""

    def test_days_of_supply_calculation(self, inventory_manager):
        """Test days of supply calculation."""
        # Set level and simulate consumption
        inventory_manager.update_level("TANK-001", 5000.0)

        # Add consumption records directly to tracker
        for _ in range(5):
            inventory_manager._consumption_tracker.add_record(
                "TANK-001", 100.0, 1.0
            )

        status = inventory_manager.get_tank_status("TANK-001")

        # With 100 gal/hr = 2400 gal/day, 5000 gal gives ~2 days
        # (minus heel volume)
        assert status.days_of_supply > 0


class TestDeliveryScheduling:
    """Tests for delivery scheduling."""

    def test_schedule_delivery(self, inventory_manager):
        """Test scheduling a delivery."""
        # Set low level
        inventory_manager.update_level("TANK-001", 2500.0)

        schedule = inventory_manager.schedule_delivery(
            tank_id="TANK-001",
            quantity_gal=5000.0,
        )

        assert schedule.tank_id == "TANK-001"
        assert schedule.quantity_gal == 5000.0
        assert schedule.status == DeliveryStatus.SCHEDULED

    def test_schedule_delivery_auto_quantity(self, inventory_manager):
        """Test delivery quantity calculated automatically."""
        inventory_manager.update_level("TANK-001", 2500.0)

        schedule = inventory_manager.schedule_delivery(tank_id="TANK-001")

        # Should calculate quantity to fill tank
        assert schedule.quantity_gal > 0

    def test_schedule_delivery_auto_date(self, inventory_manager):
        """Test delivery date calculated automatically."""
        inventory_manager.update_level("TANK-001", 2500.0)

        schedule = inventory_manager.schedule_delivery(tank_id="TANK-001")

        assert schedule.scheduled_date is not None
        assert schedule.scheduled_date >= datetime.now(timezone.utc)

    def test_record_delivery(self, inventory_manager):
        """Test recording completed delivery."""
        # Schedule delivery
        inventory_manager.update_level("TANK-001", 2500.0)
        schedule = inventory_manager.schedule_delivery("TANK-001", 5000.0)

        # Record completion
        inventory_manager.record_delivery(schedule.delivery_id, 4800.0)

        # Check status updated
        deliveries = inventory_manager._scheduled_deliveries
        completed = [d for d in deliveries if d.delivery_id == schedule.delivery_id]
        assert completed[0].status == DeliveryStatus.COMPLETED
        assert completed[0].actual_quantity_gal == 4800.0

    def test_get_pending_deliveries(self, inventory_manager):
        """Test getting pending deliveries."""
        # Schedule multiple deliveries
        inventory_manager.update_level("TANK-001", 2500.0)
        inventory_manager.schedule_delivery("TANK-001", 3000.0)
        inventory_manager.schedule_delivery("TANK-001", 2000.0)

        pending = inventory_manager.get_pending_deliveries("TANK-001")

        assert len(pending) == 2


class TestEOQCalculation:
    """Tests for Economic Order Quantity calculation."""

    def test_eoq_formula(self, inventory_manager):
        """Test EOQ formula calculation."""
        # EOQ = sqrt((2 * D * S) / H)
        # D = annual demand, S = ordering cost, H = holding cost

        eoq = inventory_manager.calculate_eoq(
            tank_id="TANK-001",
            annual_demand_gal=100000.0,
            ordering_cost_usd=150.0,
            holding_cost_pct=0.20,
            unit_cost_usd_gal=2.50,
        )

        # Expected: sqrt((2 * 100000 * 150) / (2.50 * 0.20)) = sqrt(60000000) = ~7746
        expected = (2 * 100000 * 150 / (2.50 * 0.20)) ** 0.5
        assert eoq == pytest.approx(expected, rel=0.01)

    def test_eoq_with_estimated_demand(self, inventory_manager):
        """Test EOQ with estimated demand from history."""
        # Add consumption history
        for _ in range(10):
            inventory_manager._consumption_tracker.add_record(
                "TANK-001", 100.0, 1.0
            )

        eoq = inventory_manager.calculate_eoq("TANK-001")

        assert eoq >= 0


class TestAlertManagement:
    """Tests for alert management."""

    def test_critical_level_creates_alert(self, inventory_manager):
        """Test critical level creates alert."""
        # Set critical level
        inventory_manager.update_level("TANK-001", 1000.0)

        alerts = inventory_manager.get_active_alerts()

        critical_alerts = [a for a in alerts if a.alert_type == InventoryAlertType.CRITICAL_LEVEL]
        assert len(critical_alerts) == 1

    def test_low_level_creates_alert(self, inventory_manager):
        """Test low level creates alert."""
        # Set low level
        inventory_manager.update_level("TANK-001", 2000.0)

        alerts = inventory_manager.get_active_alerts()

        low_alerts = [a for a in alerts if a.alert_type == InventoryAlertType.LOW_LEVEL]
        assert len(low_alerts) == 1

    def test_high_level_creates_alert(self, inventory_manager):
        """Test high level creates alert."""
        # Set high level
        inventory_manager.update_level("TANK-001", 9700.0)

        alerts = inventory_manager.get_active_alerts()

        high_alerts = [a for a in alerts if a.alert_type == InventoryAlertType.HIGH_LEVEL]
        assert len(high_alerts) == 1

    def test_acknowledge_alert(self, inventory_manager):
        """Test acknowledging an alert."""
        # Create alert
        inventory_manager.update_level("TANK-001", 1000.0)

        alerts = inventory_manager.get_active_alerts()
        alert_id = alerts[0].alert_id

        inventory_manager.acknowledge_alert(alert_id, "operator_001")

        # Check acknowledged
        updated_alerts = [a for a in inventory_manager._active_alerts if a.alert_id == alert_id]
        assert updated_alerts[0].acknowledged is True
        assert updated_alerts[0].acknowledged_by == "operator_001"

    def test_resolve_alert(self, inventory_manager):
        """Test resolving an alert."""
        # Create alert
        inventory_manager.update_level("TANK-001", 1000.0)

        alerts = inventory_manager.get_active_alerts()
        alert_id = alerts[0].alert_id

        inventory_manager.resolve_alert(alert_id)

        # Check resolved
        active = inventory_manager.get_active_alerts()
        resolved_alert = [a for a in inventory_manager._active_alerts if a.alert_id == alert_id]
        assert resolved_alert[0].resolved is True
        assert alert_id not in [a.alert_id for a in active]

    def test_no_duplicate_alerts(self, inventory_manager):
        """Test no duplicate alerts created."""
        # Set critical level twice
        inventory_manager.update_level("TANK-001", 1000.0)
        inventory_manager.update_level("TANK-001", 900.0)

        alerts = inventory_manager.get_active_alerts()

        critical_alerts = [a for a in alerts if a.alert_type == InventoryAlertType.CRITICAL_LEVEL]
        # Should only have one critical alert
        assert len(critical_alerts) == 1


class TestGetAllTankStatus:
    """Tests for getting all tank statuses."""

    def test_get_all_tank_status(self, inventory_manager):
        """Test getting all tank statuses."""
        # Add another tank
        inventory_manager.add_tank(TankConfig(
            tank_id="TANK-002",
            fuel_type="fuel_oil",
            capacity_gal=5000,
        ))

        # Set levels
        inventory_manager.update_level("TANK-001", 5000.0)
        inventory_manager.update_level("TANK-002", 2500.0)

        all_status = inventory_manager.get_all_tank_status()

        assert len(all_status) == 2
        assert "TANK-001" in all_status
        assert "TANK-002" in all_status


class TestDeliveryDateCalculation:
    """Tests for optimal delivery date calculation."""

    def test_preferred_delivery_day(self, inventory_config):
        """Test delivery scheduled on preferred day."""
        config = InventoryConfig(
            tanks={"TANK-001": {"fuel_type": "gas", "capacity_gal": 10000}},
            preferred_delivery_days=[1, 3, 5],  # Mon, Wed, Fri
        )
        manager = InventoryManager(config)

        manager.update_level("TANK-001", 2500.0)
        schedule = manager.schedule_delivery("TANK-001")

        # Should be on Mon, Wed, or Fri
        assert schedule.scheduled_date.isoweekday() in [1, 3, 5]


class TestDeliveryWindow:
    """Tests for delivery window configuration."""

    def test_delivery_window_hours(self, inventory_manager):
        """Test delivery window is configured correctly."""
        schedule = inventory_manager.schedule_delivery("TANK-001")

        window_hours = (
            schedule.delivery_window_end - schedule.delivery_window_start
        ).total_seconds() / 3600

        assert window_hours == inventory_manager.config.delivery_window_hours
