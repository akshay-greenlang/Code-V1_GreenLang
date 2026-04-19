"""
GL-019 HEATSCHEDULER - Schema Module Tests

Unit tests for Pydantic schema models including load forecasts, thermal storage,
demand charge results, production orders, weather forecasts, and main I/O schemas.

Test Coverage:
    - Schema validation
    - Enum values
    - Default values
    - Field constraints
    - Nested schema validation

Author: GreenLang Test Team
Date: December 2025
"""

import pytest
from datetime import datetime, timedelta, timezone
from pydantic import ValidationError


class TestScheduleStatus:
    """Tests for ScheduleStatus enumeration."""

    def test_schedule_status_values(self):
        """Test all schedule status values exist."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import ScheduleStatus

        assert ScheduleStatus.OPTIMAL.value == "optimal"
        assert ScheduleStatus.FEASIBLE.value == "feasible"
        assert ScheduleStatus.INFEASIBLE.value == "infeasible"
        assert ScheduleStatus.TIMEOUT.value == "timeout"
        assert ScheduleStatus.ERROR.value == "error"


class TestLoadForecastStatus:
    """Tests for LoadForecastStatus enumeration."""

    def test_load_forecast_status_values(self):
        """Test all load forecast status values exist."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import LoadForecastStatus

        assert LoadForecastStatus.SUCCESS.value == "success"
        assert LoadForecastStatus.DEGRADED.value == "degraded"
        assert LoadForecastStatus.FAILED.value == "failed"


class TestStorageMode:
    """Tests for StorageMode enumeration."""

    def test_storage_mode_values(self):
        """Test all storage mode values exist."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import StorageMode

        assert StorageMode.CHARGING.value == "charging"
        assert StorageMode.DISCHARGING.value == "discharging"
        assert StorageMode.IDLE.value == "idle"
        assert StorageMode.STANDBY.value == "standby"


class TestDemandAlertLevel:
    """Tests for DemandAlertLevel enumeration."""

    def test_demand_alert_level_values(self):
        """Test all demand alert level values exist."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import DemandAlertLevel

        assert DemandAlertLevel.INFO.value == "info"
        assert DemandAlertLevel.WARNING.value == "warning"
        assert DemandAlertLevel.CRITICAL.value == "critical"
        assert DemandAlertLevel.EMERGENCY.value == "emergency"


class TestScheduleAction:
    """Tests for ScheduleAction enumeration."""

    def test_schedule_action_values(self):
        """Test all schedule action values exist."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import ScheduleAction

        assert ScheduleAction.START.value == "start"
        assert ScheduleAction.STOP.value == "stop"
        assert ScheduleAction.RAMP_UP.value == "ramp_up"
        assert ScheduleAction.RAMP_DOWN.value == "ramp_down"
        assert ScheduleAction.SETPOINT_CHANGE.value == "setpoint_change"
        assert ScheduleAction.STORAGE_CHARGE.value == "storage_charge"
        assert ScheduleAction.STORAGE_DISCHARGE.value == "storage_discharge"
        assert ScheduleAction.LOAD_SHIFT.value == "load_shift"


class TestProductionStatus:
    """Tests for ProductionStatus enumeration."""

    def test_production_status_values(self):
        """Test all production status values exist."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import ProductionStatus

        assert ProductionStatus.SCHEDULED.value == "scheduled"
        assert ProductionStatus.IN_PROGRESS.value == "in_progress"
        assert ProductionStatus.COMPLETED.value == "completed"
        assert ProductionStatus.DELAYED.value == "delayed"
        assert ProductionStatus.CANCELLED.value == "cancelled"


class TestLoadForecastPoint:
    """Tests for LoadForecastPoint schema."""

    def test_valid_load_forecast_point(self, base_timestamp):
        """Test valid load forecast point creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import LoadForecastPoint

        point = LoadForecastPoint(
            timestamp=base_timestamp,
            load_kw=2500.0,
            lower_bound_kw=2000.0,
            upper_bound_kw=3000.0,
            confidence=0.95,
        )

        assert point.timestamp == base_timestamp
        assert point.load_kw == 2500.0
        assert point.lower_bound_kw == 2000.0
        assert point.upper_bound_kw == 3000.0
        assert point.confidence == 0.95

    def test_load_forecast_point_default_confidence(self, base_timestamp):
        """Test default confidence value."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import LoadForecastPoint

        point = LoadForecastPoint(
            timestamp=base_timestamp,
            load_kw=2500.0,
            lower_bound_kw=2000.0,
            upper_bound_kw=3000.0,
        )

        assert point.confidence == 0.95

    def test_load_forecast_point_negative_load_rejected(self, base_timestamp):
        """Test negative load is rejected."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import LoadForecastPoint

        with pytest.raises(ValidationError):
            LoadForecastPoint(
                timestamp=base_timestamp,
                load_kw=-100.0,
                lower_bound_kw=2000.0,
                upper_bound_kw=3000.0,
            )

    def test_load_forecast_point_confidence_bounds(self, base_timestamp):
        """Test confidence must be between 0 and 1."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import LoadForecastPoint

        with pytest.raises(ValidationError):
            LoadForecastPoint(
                timestamp=base_timestamp,
                load_kw=2500.0,
                lower_bound_kw=2000.0,
                upper_bound_kw=3000.0,
                confidence=1.5,
            )


class TestLoadForecastResult:
    """Tests for LoadForecastResult schema."""

    def test_valid_load_forecast_result(self, sample_load_forecast):
        """Test valid load forecast result."""
        assert sample_load_forecast.forecast_id == "TEST-FC-001"
        assert sample_load_forecast.forecast_horizon_hours == 24
        assert len(sample_load_forecast.forecast_points) == 96

    def test_load_forecast_result_auto_generated_id(self, base_timestamp):
        """Test auto-generated forecast ID."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            LoadForecastResult,
            LoadForecastStatus,
        )

        result = LoadForecastResult(
            status=LoadForecastStatus.SUCCESS,
            forecast_points=[],
            forecast_horizon_hours=24,
        )

        assert result.forecast_id is not None
        assert len(result.forecast_id) == 8

    def test_load_forecast_result_metrics(self, sample_load_forecast):
        """Test forecast result metrics."""
        assert sample_load_forecast.mape_pct == 5.2
        assert sample_load_forecast.rmse_kw == 125.0
        assert sample_load_forecast.mae_kw == 95.0
        assert sample_load_forecast.data_quality_score == 0.95


class TestStorageStatePoint:
    """Tests for StorageStatePoint schema."""

    def test_valid_storage_state_point(self, base_timestamp):
        """Test valid storage state point creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            StorageStatePoint,
            StorageMode,
        )

        point = StorageStatePoint(
            timestamp=base_timestamp,
            state_of_charge_pct=75.0,
            state_of_charge_kwh=3750.0,
            temperature_c=80.0,
            mode=StorageMode.CHARGING,
            power_kw=250.0,
        )

        assert point.state_of_charge_pct == 75.0
        assert point.state_of_charge_kwh == 3750.0
        assert point.mode == StorageMode.CHARGING
        assert point.power_kw == 250.0

    def test_storage_state_point_soc_bounds(self, base_timestamp):
        """Test SOC must be between 0 and 100."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import StorageStatePoint

        with pytest.raises(ValidationError):
            StorageStatePoint(
                timestamp=base_timestamp,
                state_of_charge_pct=150.0,
                state_of_charge_kwh=3750.0,
            )


class TestStorageDispatchSchedule:
    """Tests for StorageDispatchSchedule schema."""

    def test_valid_storage_dispatch_schedule(self, base_timestamp):
        """Test valid storage dispatch schedule creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            StorageDispatchSchedule,
            StorageStatePoint,
            StorageMode,
        )

        points = [
            StorageStatePoint(
                timestamp=base_timestamp + timedelta(hours=i),
                state_of_charge_pct=50 + i * 5,
                state_of_charge_kwh=2500 + i * 250,
                mode=StorageMode.CHARGING if i < 4 else StorageMode.DISCHARGING,
                power_kw=200 if i < 4 else -200,
            )
            for i in range(8)
        ]

        schedule = StorageDispatchSchedule(
            storage_id="TES-001",
            dispatch_points=points,
            total_charge_kwh=800.0,
            total_discharge_kwh=600.0,
            charge_hours=4.0,
            discharge_hours=4.0,
            cycles=0.12,
            energy_arbitrage_usd=45.0,
            demand_savings_usd=120.0,
        )

        assert schedule.storage_id == "TES-001"
        assert len(schedule.dispatch_points) == 8
        assert schedule.total_charge_kwh == 800.0
        assert schedule.energy_arbitrage_usd == 45.0


class TestThermalStorageResult:
    """Tests for ThermalStorageResult schema."""

    def test_valid_thermal_storage_result(self, base_timestamp):
        """Test valid thermal storage result creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            ThermalStorageResult,
            StorageDispatchSchedule,
        )

        result = ThermalStorageResult(
            timestamp=base_timestamp,
            unit_schedules=[],
            total_storage_capacity_kwh=5000.0,
            current_soc_kwh=2500.0,
            total_energy_arbitrage_usd=100.0,
            total_demand_savings_usd=200.0,
            total_savings_usd=300.0,
        )

        assert result.total_storage_capacity_kwh == 5000.0
        assert result.total_savings_usd == 300.0


class TestDemandPeriod:
    """Tests for DemandPeriod schema."""

    def test_valid_demand_period(self, base_timestamp):
        """Test valid demand period creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import DemandPeriod

        period = DemandPeriod(
            period_start=base_timestamp,
            period_end=base_timestamp + timedelta(minutes=15),
            avg_demand_kw=3500.0,
            peak_demand_kw=4000.0,
            is_on_peak=True,
        )

        assert period.avg_demand_kw == 3500.0
        assert period.peak_demand_kw == 4000.0
        assert period.is_on_peak is True


class TestDemandChargeResult:
    """Tests for DemandChargeResult schema."""

    def test_valid_demand_charge_result(self, base_timestamp):
        """Test valid demand charge result creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            DemandChargeResult,
            DemandAlertLevel,
        )

        result = DemandChargeResult(
            timestamp=base_timestamp,
            baseline_peak_kw=5000.0,
            optimized_peak_kw=4200.0,
            peak_reduction_kw=800.0,
            peak_reduction_pct=16.0,
            peak_time_baseline=base_timestamp + timedelta(hours=4),
            baseline_demand_charge_usd=750.0,
            optimized_demand_charge_usd=630.0,
            demand_charge_savings_usd=120.0,
            load_shifted_kwh=500.0,
            load_shift_savings_usd=50.0,
            peak_limit_exceeded=False,
        )

        assert result.baseline_peak_kw == 5000.0
        assert result.optimized_peak_kw == 4200.0
        assert result.peak_reduction_pct == 16.0
        assert result.demand_charge_savings_usd == 120.0

    def test_demand_charge_result_with_alert(self, base_timestamp):
        """Test demand charge result with alert."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            DemandChargeResult,
            DemandAlertLevel,
        )

        result = DemandChargeResult(
            timestamp=base_timestamp,
            baseline_peak_kw=6000.0,
            optimized_peak_kw=5500.0,
            baseline_demand_charge_usd=900.0,
            optimized_demand_charge_usd=825.0,
            peak_limit_exceeded=True,
            alert_level=DemandAlertLevel.WARNING,
            alert_message="Peak demand 10% over limit",
        )

        assert result.peak_limit_exceeded is True
        assert result.alert_level == DemandAlertLevel.WARNING


class TestProductionOrder:
    """Tests for ProductionOrder schema."""

    def test_valid_production_order(self, base_timestamp):
        """Test valid production order creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            ProductionOrder,
            ProductionStatus,
        )

        order = ProductionOrder(
            order_id="ORD-001",
            product_id="PRD-A",
            product_name="Steel Treatment",
            scheduled_start=base_timestamp,
            scheduled_end=base_timestamp + timedelta(hours=3),
            duration_hours=3.0,
            heat_load_kw=800.0,
            temperature_c=850.0,
            ramp_up_time_minutes=30,
            priority=8,
            is_flexible=False,
            status=ProductionStatus.SCHEDULED,
        )

        assert order.order_id == "ORD-001"
        assert order.heat_load_kw == 800.0
        assert order.priority == 8
        assert order.is_flexible is False

    def test_production_order_priority_bounds(self, base_timestamp):
        """Test priority must be between 1 and 10."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import ProductionOrder

        with pytest.raises(ValidationError):
            ProductionOrder(
                order_id="ORD-001",
                scheduled_start=base_timestamp,
                scheduled_end=base_timestamp + timedelta(hours=3),
                duration_hours=3.0,
                heat_load_kw=800.0,
                priority=15,  # Invalid
            )


class TestProductionScheduleResult:
    """Tests for ProductionScheduleResult schema."""

    def test_valid_production_schedule_result(self, base_timestamp, sample_production_orders):
        """Test valid production schedule result creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            ProductionScheduleResult,
        )

        result = ProductionScheduleResult(
            timestamp=base_timestamp,
            scheduled_orders=sample_production_orders,
            rescheduled_orders=[sample_production_orders[1]],
            total_orders=3,
            orders_on_time=2,
            orders_rescheduled=1,
            total_heat_load_kwh=5400.0,
            scheduling_cost_savings_usd=60.0,
        )

        assert result.total_orders == 3
        assert result.orders_rescheduled == 1
        assert result.scheduling_cost_savings_usd == 60.0


class TestWeatherForecastPoint:
    """Tests for WeatherForecastPoint schema."""

    def test_valid_weather_forecast_point(self, base_timestamp):
        """Test valid weather forecast point creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import WeatherForecastPoint

        point = WeatherForecastPoint(
            timestamp=base_timestamp,
            temperature_c=22.5,
            humidity_pct=65.0,
            solar_radiation_w_m2=450.0,
            wind_speed_m_s=3.5,
            cloud_cover_pct=30.0,
            heating_degree_hours=0.0,
            cooling_degree_hours=0.0,
        )

        assert point.temperature_c == 22.5
        assert point.humidity_pct == 65.0
        assert point.solar_radiation_w_m2 == 450.0

    def test_weather_forecast_point_humidity_bounds(self, base_timestamp):
        """Test humidity must be between 0 and 100."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import WeatherForecastPoint

        with pytest.raises(ValidationError):
            WeatherForecastPoint(
                timestamp=base_timestamp,
                temperature_c=22.5,
                humidity_pct=150.0,  # Invalid
            )


class TestWeatherForecastResult:
    """Tests for WeatherForecastResult schema."""

    def test_valid_weather_forecast_result(self, sample_weather_forecast):
        """Test valid weather forecast result."""
        assert sample_weather_forecast.forecast_id == "TEST-WX-001"
        assert sample_weather_forecast.provider == "openweathermap"
        assert sample_weather_forecast.latitude == 37.7749
        assert len(sample_weather_forecast.forecast_points) == 24

    def test_weather_forecast_result_temperature_aggregates(self, sample_weather_forecast):
        """Test temperature aggregates are calculated."""
        assert sample_weather_forecast.avg_temperature_c is not None
        assert sample_weather_forecast.max_temperature_c is not None
        assert sample_weather_forecast.min_temperature_c is not None


class TestScheduleActionItem:
    """Tests for ScheduleActionItem schema."""

    def test_valid_schedule_action_item(self, base_timestamp):
        """Test valid schedule action item creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            ScheduleActionItem,
            ScheduleAction,
        )

        action = ScheduleActionItem(
            timestamp=base_timestamp,
            action_type=ScheduleAction.START,
            equipment_id="FURN-001",
            power_setpoint_kw=500.0,
            temperature_setpoint_c=850.0,
            duration_minutes=60,
            reason="Production order ORD-001",
            expected_savings_usd=25.0,
            priority=8,
            is_mandatory=True,
        )

        assert action.action_type == ScheduleAction.START
        assert action.equipment_id == "FURN-001"
        assert action.power_setpoint_kw == 500.0
        assert action.is_mandatory is True

    def test_schedule_action_auto_generated_id(self, base_timestamp):
        """Test auto-generated action ID."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            ScheduleActionItem,
            ScheduleAction,
        )

        action = ScheduleActionItem(
            timestamp=base_timestamp,
            action_type=ScheduleAction.STORAGE_CHARGE,
            storage_id="TES-001",
            power_setpoint_kw=300.0,
        )

        assert action.action_id is not None
        assert len(action.action_id) == 8


class TestHeatSchedulerInput:
    """Tests for HeatSchedulerInput schema."""

    def test_valid_scheduler_input(self, sample_scheduler_input):
        """Test valid scheduler input."""
        assert sample_scheduler_input.facility_id == "PLANT-001"
        assert sample_scheduler_input.optimization_horizon_hours == 24
        assert sample_scheduler_input.current_load_kw == 2500.0

    def test_scheduler_input_auto_generated_request_id(self, base_timestamp):
        """Test auto-generated request ID."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import HeatSchedulerInput

        input_data = HeatSchedulerInput(
            facility_id="PLANT-001",
            current_load_kw=2500.0,
        )

        assert input_data.request_id is not None
        assert len(input_data.request_id) == 36  # UUID format

    def test_scheduler_input_default_values(self, base_timestamp):
        """Test scheduler input default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import HeatSchedulerInput

        input_data = HeatSchedulerInput(
            facility_id="PLANT-001",
            current_load_kw=2500.0,
        )

        assert input_data.optimization_horizon_hours == 24
        assert input_data.time_step_minutes == 15
        assert input_data.prefer_storage_discharge_during_peak is True
        assert input_data.allow_production_rescheduling is True

    def test_scheduler_input_horizon_bounds(self, base_timestamp):
        """Test optimization horizon bounds."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import HeatSchedulerInput

        with pytest.raises(ValidationError):
            HeatSchedulerInput(
                facility_id="PLANT-001",
                current_load_kw=2500.0,
                optimization_horizon_hours=200,  # Max is 168
            )


class TestHeatSchedulerOutput:
    """Tests for HeatSchedulerOutput schema."""

    def test_valid_scheduler_output(self, base_timestamp, sample_load_forecast):
        """Test valid scheduler output creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerOutput,
            ScheduleStatus,
        )

        output = HeatSchedulerOutput(
            facility_id="PLANT-001",
            request_id="REQ-001",
            timestamp=base_timestamp,
            status=ScheduleStatus.OPTIMAL,
            processing_time_ms=150.5,
            schedule_horizon_hours=24,
            schedule_actions=[],
            load_forecast=sample_load_forecast,
            baseline_cost_usd=1500.0,
            optimized_cost_usd=1275.0,
            total_savings_usd=225.0,
            total_energy_kwh=50000.0,
            peak_demand_kw=4200.0,
            average_load_kw=2100.0,
            load_factor_pct=50.0,
        )

        assert output.status == ScheduleStatus.OPTIMAL
        assert output.processing_time_ms == 150.5
        assert output.total_savings_usd == 225.0

    def test_scheduler_output_with_all_components(
        self,
        base_timestamp,
        sample_load_forecast,
        sample_weather_forecast,
    ):
        """Test scheduler output with all components."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerOutput,
            ScheduleStatus,
            ThermalStorageResult,
            DemandChargeResult,
            ProductionScheduleResult,
        )

        storage_result = ThermalStorageResult(
            total_storage_capacity_kwh=5000.0,
            current_soc_kwh=2500.0,
            total_savings_usd=150.0,
        )

        demand_result = DemandChargeResult(
            baseline_peak_kw=5000.0,
            optimized_peak_kw=4200.0,
            baseline_demand_charge_usd=750.0,
            optimized_demand_charge_usd=630.0,
            demand_charge_savings_usd=120.0,
        )

        production_result = ProductionScheduleResult(
            total_orders=3,
            orders_on_time=3,
        )

        output = HeatSchedulerOutput(
            facility_id="PLANT-001",
            request_id="REQ-001",
            status=ScheduleStatus.OPTIMAL,
            schedule_horizon_hours=24,
            load_forecast=sample_load_forecast,
            storage_result=storage_result,
            demand_result=demand_result,
            production_result=production_result,
            weather_forecast=sample_weather_forecast,
            baseline_cost_usd=1500.0,
            optimized_cost_usd=1200.0,
            total_savings_usd=300.0,
            savings_breakdown={
                "demand_savings": 120.0,
                "storage_arbitrage": 100.0,
                "load_shift_savings": 80.0,
            },
        )

        assert output.storage_result is not None
        assert output.demand_result is not None
        assert output.production_result is not None
        assert output.weather_forecast is not None
        assert len(output.savings_breakdown) == 3

    def test_scheduler_output_kpis(self, base_timestamp, sample_load_forecast):
        """Test scheduler output KPIs."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerOutput,
            ScheduleStatus,
        )

        output = HeatSchedulerOutput(
            facility_id="PLANT-001",
            request_id="REQ-001",
            status=ScheduleStatus.OPTIMAL,
            schedule_horizon_hours=24,
            load_forecast=sample_load_forecast,
            baseline_cost_usd=1500.0,
            optimized_cost_usd=1275.0,
            kpis={
                "peak_demand_kw": 4200.0,
                "peak_reduction_pct": 16.0,
                "load_factor_pct": 50.0,
                "total_energy_kwh": 50400.0,
            },
        )

        assert "peak_demand_kw" in output.kpis
        assert "peak_reduction_pct" in output.kpis
        assert output.kpis["peak_reduction_pct"] == 16.0

    def test_scheduler_output_provenance(self, base_timestamp, sample_load_forecast):
        """Test scheduler output provenance tracking."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerOutput,
            ScheduleStatus,
        )

        output = HeatSchedulerOutput(
            facility_id="PLANT-001",
            request_id="REQ-001",
            status=ScheduleStatus.OPTIMAL,
            schedule_horizon_hours=24,
            load_forecast=sample_load_forecast,
            baseline_cost_usd=1500.0,
            optimized_cost_usd=1275.0,
            provenance_hash="abc123def456",
            input_hash="789xyz",
        )

        assert output.provenance_hash == "abc123def456"
        assert output.input_hash == "789xyz"

    def test_scheduler_output_intelligence_fields(self, base_timestamp, sample_load_forecast):
        """Test scheduler output intelligence fields."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerOutput,
            ScheduleStatus,
        )

        output = HeatSchedulerOutput(
            facility_id="PLANT-001",
            request_id="REQ-001",
            status=ScheduleStatus.OPTIMAL,
            schedule_horizon_hours=24,
            load_forecast=sample_load_forecast,
            baseline_cost_usd=1500.0,
            optimized_cost_usd=1275.0,
            explanation="Schedule optimized for cost reduction during peak hours.",
            intelligent_recommendations=[
                {"action": "shift_load", "savings": 50.0},
                {"action": "increase_storage", "savings": 30.0},
            ],
            anomalies_detected=[
                {"type": "load_spike", "timestamp": base_timestamp.isoformat()},
            ],
            reasoning_output="Based on tariff structure, shifting 200kW to off-peak saves $25/day.",
        )

        assert output.explanation is not None
        assert len(output.intelligent_recommendations) == 2
        assert len(output.anomalies_detected) == 1
