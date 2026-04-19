"""
GL-019 HEATSCHEDULER - Test Fixtures and Configuration

Shared pytest fixtures for the Heat Scheduler test suite.
Provides reusable test data, mock objects, and configuration helpers.

Fixtures:
    - Configuration fixtures (tariffs, equipment, storage)
    - Data fixtures (forecasts, weather, production orders)
    - Mock fixtures (ERP, weather providers)
    - Performance benchmarking fixtures

Author: GreenLang Test Team
Date: December 2025
"""

import pytest
from datetime import datetime, timedelta, timezone, time
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import math


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def sample_tariff_config():
    """Create sample tariff configuration."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
        TariffConfiguration,
        TariffType,
    )
    return TariffConfiguration(
        tariff_id="TOU-001",
        tariff_type=TariffType.TIME_OF_USE,
        utility_name="Pacific Gas & Electric",
        rate_schedule="E-19",
        peak_rate_per_kwh=0.15,
        off_peak_rate_per_kwh=0.06,
        shoulder_rate_per_kwh=0.10,
        peak_hours_start=14,
        peak_hours_end=20,
        weekend_off_peak=True,
        holiday_off_peak=True,
        demand_charge_per_kw=12.50,
        peak_demand_charge_per_kw=5.0,
        ratchet_percentage=80.0,
    )


@pytest.fixture
def sample_demand_tariff_config():
    """Create tariff configuration with demand charges."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
        TariffConfiguration,
        TariffType,
    )
    return TariffConfiguration(
        tariff_id="DEMAND-001",
        tariff_type=TariffType.DEMAND_CHARGE,
        peak_rate_per_kwh=0.12,
        off_peak_rate_per_kwh=0.05,
        peak_hours_start=12,
        peak_hours_end=18,
        demand_charge_per_kw=15.0,
        peak_demand_charge_per_kw=8.0,
        ratchet_percentage=75.0,
    )


@pytest.fixture
def sample_equipment_config():
    """Create sample equipment configuration."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
        EquipmentConfiguration,
        EquipmentType,
        EquipmentStatus,
    )
    return EquipmentConfiguration(
        equipment_id="FURN-001",
        equipment_type=EquipmentType.ELECTRIC_FURNACE,
        equipment_name="Main Heat Treatment Furnace",
        capacity_kw=500.0,
        min_power_kw=50.0,
        max_power_kw=550.0,
        standby_power_kw=10.0,
        efficiency=0.92,
        max_temperature_c=1200.0,
        min_temperature_c=20.0,
        ramp_rate_c_per_minute=15.0,
        cooldown_rate_c_per_minute=8.0,
        min_run_time_minutes=60,
        min_idle_time_minutes=15,
        startup_time_minutes=45,
        shutdown_time_minutes=20,
        status=EquipmentStatus.AVAILABLE,
        available_days=[0, 1, 2, 3, 4],
    )


@pytest.fixture
def sample_storage_config():
    """Create sample thermal storage configuration."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
        ThermalStorageConfiguration,
        StorageType,
    )
    return ThermalStorageConfiguration(
        storage_id="TES-001",
        storage_type=StorageType.HOT_WATER_TANK,
        storage_name="Main Hot Water Tank",
        enabled=True,
        capacity_kwh=5000.0,
        max_charge_rate_kw=500.0,
        max_discharge_rate_kw=500.0,
        round_trip_efficiency=0.92,
        standby_loss_pct_per_hour=0.5,
        min_soc_pct=10.0,
        max_soc_pct=95.0,
        design_temperature_c=85.0,
        min_temperature_c=60.0,
        max_temperature_c=95.0,
        current_soc_pct=50.0,
        current_temperature_c=75.0,
        charge_priority=6,
        discharge_priority=7,
    )


@pytest.fixture
def sample_pcm_storage_config():
    """Create sample PCM storage configuration."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
        ThermalStorageConfiguration,
        StorageType,
    )
    return ThermalStorageConfiguration(
        storage_id="PCM-001",
        storage_type=StorageType.PCM_STORAGE,
        storage_name="PCM Storage Unit",
        enabled=True,
        capacity_kwh=3000.0,
        max_charge_rate_kw=300.0,
        max_discharge_rate_kw=300.0,
        round_trip_efficiency=0.88,
        standby_loss_pct_per_hour=0.3,
        min_soc_pct=5.0,
        max_soc_pct=98.0,
        pcm_melt_temperature_c=58.0,
        pcm_latent_heat_kj_kg=200.0,
        current_soc_pct=40.0,
    )


@pytest.fixture
def sample_load_forecasting_config():
    """Create sample load forecasting configuration."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
        LoadForecastingConfiguration,
        ForecastModel,
    )
    return LoadForecastingConfiguration(
        enabled=True,
        model_type=ForecastModel.ENSEMBLE,
        forecast_horizon_hours=48,
        update_interval_minutes=15,
        granularity_minutes=15,
        lookback_days=30,
        confidence_level=0.90,
        use_weather_features=True,
        use_calendar_features=True,
        use_production_features=True,
        use_lagged_features=True,
        retrain_interval_days=7,
        min_training_samples=1000,
        mape_warning_threshold=10.0,
        mape_critical_threshold=20.0,
    )


@pytest.fixture
def sample_demand_charge_config():
    """Create sample demand charge configuration."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
        DemandChargeConfiguration,
    )
    return DemandChargeConfiguration(
        enabled=True,
        peak_demand_limit_kw=5000.0,
        soft_peak_limit_kw=4500.0,
        absolute_max_demand_kw=6000.0,
        demand_interval_minutes=15,
        rolling_demand_average=True,
        enable_load_shifting=True,
        max_shift_hours=4,
        min_shift_savings_threshold_usd=5.0,
        enable_demand_response=True,
        demand_response_threshold_kw=1000.0,
        max_demand_curtailment_pct=30.0,
        dr_notification_lead_time_minutes=30,
        consider_ratchet=True,
        ratchet_percentage=80.0,
        annual_ratchet_peak_kw=4800.0,
    )


@pytest.fixture
def sample_weather_config():
    """Create sample weather configuration."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
        WeatherConfiguration,
    )
    return WeatherConfiguration(
        enabled=True,
        latitude=37.7749,
        longitude=-122.4194,
        timezone="America/Los_Angeles",
        api_provider="openweathermap",
        api_key_secret_name="weather_api_key",
        update_interval_minutes=30,
        forecast_horizon_hours=72,
        include_temperature=True,
        include_humidity=True,
        include_solar_radiation=True,
        include_wind=True,
        heating_base_temp_c=18.0,
        cooling_base_temp_c=24.0,
    )


@pytest.fixture
def sample_optimization_params():
    """Create sample optimization parameters."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
        OptimizationParameters,
        OptimizationObjective,
    )
    return OptimizationParameters(
        primary_objective=OptimizationObjective.MINIMIZE_COST,
        secondary_objective=OptimizationObjective.MINIMIZE_PEAK_DEMAND,
        cost_weight=0.6,
        demand_weight=0.3,
        efficiency_weight=0.1,
        target_cost_reduction_percent=15.0,
        target_peak_demand_reduction_percent=20.0,
        enable_peak_shaving=True,
        peak_demand_limit_kw=5000.0,
        optimization_time_limit_seconds=30,
        solution_gap_tolerance=0.05,
    )


@pytest.fixture
def sample_scheduler_config(
    sample_tariff_config,
    sample_equipment_config,
    sample_storage_config,
    sample_load_forecasting_config,
    sample_demand_charge_config,
    sample_weather_config,
    sample_optimization_params,
):
    """Create complete scheduler configuration."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
        HeatSchedulerConfig,
    )
    return HeatSchedulerConfig(
        agent_name="Test Heat Scheduler",
        version="1.0.0",
        environment="test",
        tariffs=[sample_tariff_config],
        equipment=[sample_equipment_config],
        thermal_storage=[sample_storage_config],
        load_forecasting=sample_load_forecasting_config,
        demand_charge=sample_demand_charge_config,
        weather=sample_weather_config,
        optimization_parameters=sample_optimization_params,
        optimization_interval_minutes=15,
        auto_apply_schedule=False,
        schedule_lookahead_hours=24,
    )


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def base_timestamp():
    """Create base timestamp for tests."""
    return datetime(2025, 6, 15, 8, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_load_forecast_points(base_timestamp):
    """Create sample load forecast points."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
        LoadForecastPoint,
    )
    points = []
    base_load = 2500.0

    for i in range(96):  # 24 hours * 4 (15-min intervals)
        t = base_timestamp + timedelta(minutes=15 * i)
        hour = t.hour

        # Simulate daily load pattern
        hour_factor = 0.7 + 0.6 * math.sin((hour - 6) * math.pi / 12)
        load = base_load * hour_factor

        # Add some variation for weekends
        if t.weekday() >= 5:
            load *= 0.7

        points.append(LoadForecastPoint(
            timestamp=t,
            load_kw=round(load, 2),
            lower_bound_kw=round(load * 0.85, 2),
            upper_bound_kw=round(load * 1.15, 2),
            confidence=0.90,
        ))

    return points


@pytest.fixture
def sample_load_forecast(base_timestamp, sample_load_forecast_points):
    """Create sample load forecast result."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
        LoadForecastResult,
        LoadForecastStatus,
    )
    loads = [p.load_kw for p in sample_load_forecast_points]
    total_kwh = sum(loads) * 0.25  # 15-min intervals

    return LoadForecastResult(
        forecast_id="TEST-FC-001",
        generated_at=base_timestamp,
        status=LoadForecastStatus.SUCCESS,
        forecast_points=sample_load_forecast_points,
        forecast_horizon_hours=24,
        resolution_minutes=15,
        model_used="ensemble",
        mape_pct=5.2,
        rmse_kw=125.0,
        mae_kw=95.0,
        peak_load_kw=max(loads),
        peak_load_time=sample_load_forecast_points[loads.index(max(loads))].timestamp,
        min_load_kw=min(loads),
        avg_load_kw=sum(loads) / len(loads),
        total_energy_kwh=total_kwh,
        data_quality_score=0.95,
        feature_importance={
            "lag_24h": 0.35,
            "temperature_c": 0.25,
            "hour_of_day": 0.15,
            "production_scheduled": 0.10,
            "day_of_week": 0.08,
            "lag_1h": 0.07,
        },
    )


@pytest.fixture
def sample_weather_forecast_points(base_timestamp):
    """Create sample weather forecast points."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
        WeatherForecastPoint,
    )
    points = []

    for i in range(24):  # Hourly for 24 hours
        t = base_timestamp + timedelta(hours=i)
        hour = t.hour

        # Simulate daily temperature pattern
        temp = 15.0 + 8.0 * math.sin((hour - 6) * math.pi / 12)
        solar = max(0, 600 * math.sin((hour - 6) * math.pi / 12)) if 6 <= hour <= 18 else 0

        points.append(WeatherForecastPoint(
            timestamp=t,
            temperature_c=round(temp, 1),
            humidity_pct=round(60 + 15 * math.cos(hour * math.pi / 12), 1),
            solar_radiation_w_m2=round(solar, 1),
            wind_speed_m_s=3.0,
            cloud_cover_pct=30.0,
            heating_degree_hours=max(0, 18 - temp),
            cooling_degree_hours=max(0, temp - 24),
        ))

    return points


@pytest.fixture
def sample_weather_forecast(base_timestamp, sample_weather_forecast_points):
    """Create sample weather forecast result."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
        WeatherForecastResult,
    )
    temps = [p.temperature_c for p in sample_weather_forecast_points]

    return WeatherForecastResult(
        forecast_id="TEST-WX-001",
        generated_at=base_timestamp,
        provider="openweathermap",
        latitude=37.7749,
        longitude=-122.4194,
        forecast_points=sample_weather_forecast_points,
        forecast_horizon_hours=24,
        avg_temperature_c=sum(temps) / len(temps),
        max_temperature_c=max(temps),
        min_temperature_c=min(temps),
        total_heating_degree_hours=sum(
            p.heating_degree_hours for p in sample_weather_forecast_points
            if p.heating_degree_hours
        ),
        total_cooling_degree_hours=sum(
            p.cooling_degree_hours for p in sample_weather_forecast_points
            if p.cooling_degree_hours
        ),
    )


@pytest.fixture
def sample_production_orders(base_timestamp):
    """Create sample production orders."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
        ProductionOrder,
        ProductionStatus,
    )
    return [
        ProductionOrder(
            order_id="ORD-001",
            product_id="PRD-A",
            product_name="Steel Treatment Batch A",
            scheduled_start=base_timestamp + timedelta(hours=2),
            scheduled_end=base_timestamp + timedelta(hours=5),
            duration_hours=3.0,
            heat_load_kw=800.0,
            temperature_c=850.0,
            ramp_up_time_minutes=30,
            priority=8,
            is_flexible=False,
            status=ProductionStatus.SCHEDULED,
        ),
        ProductionOrder(
            order_id="ORD-002",
            product_id="PRD-B",
            product_name="Aluminum Annealing Batch B",
            scheduled_start=base_timestamp + timedelta(hours=10),
            scheduled_end=base_timestamp + timedelta(hours=14),
            duration_hours=4.0,
            heat_load_kw=600.0,
            temperature_c=550.0,
            ramp_up_time_minutes=20,
            priority=6,
            is_flexible=True,
            earliest_start=base_timestamp + timedelta(hours=8),
            latest_end=base_timestamp + timedelta(hours=20),
            status=ProductionStatus.SCHEDULED,
        ),
        ProductionOrder(
            order_id="ORD-003",
            product_id="PRD-C",
            product_name="Ceramic Firing Batch C",
            scheduled_start=base_timestamp + timedelta(hours=16),
            scheduled_end=base_timestamp + timedelta(hours=20),
            duration_hours=4.0,
            heat_load_kw=400.0,
            temperature_c=1100.0,
            ramp_up_time_minutes=45,
            priority=5,
            is_flexible=True,
            earliest_start=base_timestamp + timedelta(hours=12),
            latest_end=base_timestamp + timedelta(hours=24),
            status=ProductionStatus.SCHEDULED,
        ),
    ]


@pytest.fixture
def sample_historical_data(base_timestamp):
    """Create sample historical load data."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
        HistoricalDataPoint,
    )
    data = []
    base_load = 2500.0

    # Generate 7 days of historical data
    for day in range(7):
        for hour in range(24):
            for quarter in range(4):
                t = base_timestamp - timedelta(days=7-day, hours=24-hour, minutes=15*(3-quarter))

                # Daily pattern
                hour_factor = 0.7 + 0.6 * math.sin((hour - 6) * math.pi / 12)
                load = base_load * hour_factor

                # Weekend reduction
                if t.weekday() >= 5:
                    load *= 0.7

                # Add noise
                load *= (0.95 + 0.1 * ((day * hour * quarter) % 10) / 10)

                data.append(HistoricalDataPoint(
                    timestamp=t,
                    load_kw=round(load, 2),
                    temperature_c=15 + 5 * math.sin((hour - 6) * math.pi / 12),
                    humidity_pct=60.0,
                    is_holiday=False,
                    is_weekend=t.weekday() >= 5,
                    production_level=hour_factor,
                ))

    return data


@pytest.fixture
def sample_scheduler_input(base_timestamp, sample_load_forecast, sample_weather_forecast, sample_production_orders):
    """Create sample scheduler input."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
        HeatSchedulerInput,
    )
    return HeatSchedulerInput(
        facility_id="PLANT-001",
        request_id="REQ-001",
        timestamp=base_timestamp,
        optimization_horizon_hours=24,
        time_step_minutes=15,
        current_load_kw=2500.0,
        current_storage_soc_pct={"TES-001": 50.0},
        current_equipment_status={"FURN-001": "available"},
        load_forecast=sample_load_forecast,
        weather_forecast=sample_weather_forecast,
        production_orders=sample_production_orders,
        max_peak_demand_kw=5000.0,
        prefer_storage_discharge_during_peak=True,
        allow_production_rescheduling=True,
    )


# =============================================================================
# PRODUCTION SHIFT FIXTURES
# =============================================================================

@pytest.fixture
def sample_production_shifts():
    """Create sample production shifts."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
        ProductionShift,
    )
    return [
        ProductionShift(
            shift_id="day_shift",
            name="Day Shift",
            start_time_hour=6,
            start_time_minute=0,
            duration_hours=8,
            days_active=[0, 1, 2, 3, 4],
            heat_load_factor=1.0,
            ramp_up_minutes=30,
            ramp_down_minutes=15,
        ),
        ProductionShift(
            shift_id="evening_shift",
            name="Evening Shift",
            start_time_hour=14,
            start_time_minute=0,
            duration_hours=8,
            days_active=[0, 1, 2, 3, 4],
            heat_load_factor=0.9,
            ramp_up_minutes=15,
            ramp_down_minutes=10,
        ),
        ProductionShift(
            shift_id="night_shift",
            name="Night Shift",
            start_time_hour=22,
            start_time_minute=0,
            duration_hours=8,
            days_active=[0, 1, 2, 3, 4],
            heat_load_factor=0.6,
            ramp_up_minutes=20,
            ramp_down_minutes=10,
        ),
    ]


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_weather_provider():
    """Create mock weather provider."""
    provider = AsyncMock()
    provider.get_current_weather = AsyncMock(return_value=Mock(
        timestamp=datetime.now(timezone.utc),
        temperature_c=20.0,
        humidity_pct=60.0,
        wind_speed_m_s=3.0,
        cloud_cover_pct=30.0,
    ))
    provider.get_forecast = AsyncMock(return_value=[])
    return provider


@pytest.fixture
def mock_erp_connector():
    """Create mock ERP connector."""
    connector = AsyncMock()
    connector.connect = AsyncMock(return_value=True)
    connector.fetch_production_orders = AsyncMock(return_value=[])
    connector.update_order_schedule = AsyncMock(return_value=True)
    connector.disconnect = AsyncMock()
    return connector


# =============================================================================
# PERFORMANCE FIXTURES
# =============================================================================

@pytest.fixture
def large_load_forecast(base_timestamp):
    """Create large load forecast for performance testing."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
        LoadForecastPoint,
        LoadForecastResult,
        LoadForecastStatus,
    )
    points = []
    base_load = 2500.0

    # 168 hours (1 week) at 15-min intervals = 672 points
    for i in range(672):
        t = base_timestamp + timedelta(minutes=15 * i)
        hour = t.hour
        hour_factor = 0.7 + 0.6 * math.sin((hour - 6) * math.pi / 12)
        load = base_load * hour_factor

        if t.weekday() >= 5:
            load *= 0.7

        points.append(LoadForecastPoint(
            timestamp=t,
            load_kw=round(load, 2),
            lower_bound_kw=round(load * 0.85, 2),
            upper_bound_kw=round(load * 1.15, 2),
            confidence=0.90,
        ))

    loads = [p.load_kw for p in points]
    return LoadForecastResult(
        status=LoadForecastStatus.SUCCESS,
        forecast_points=points,
        forecast_horizon_hours=168,
        resolution_minutes=15,
        model_used="ensemble",
        peak_load_kw=max(loads),
        avg_load_kw=sum(loads) / len(loads),
        total_energy_kwh=sum(loads) * 0.25,
    )


@pytest.fixture
def many_production_orders(base_timestamp):
    """Create many production orders for performance testing."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
        ProductionOrder,
        ProductionStatus,
    )
    orders = []

    for i in range(100):
        start = base_timestamp + timedelta(hours=i * 0.5)
        duration = 1 + (i % 5)

        orders.append(ProductionOrder(
            order_id=f"ORD-{i:04d}",
            product_id=f"PRD-{i % 10}",
            product_name=f"Product Batch {i}",
            scheduled_start=start,
            scheduled_end=start + timedelta(hours=duration),
            duration_hours=float(duration),
            heat_load_kw=200 + (i % 10) * 50,
            temperature_c=500 + (i % 10) * 50,
            ramp_up_time_minutes=15 + (i % 4) * 5,
            priority=5 + (i % 5),
            is_flexible=(i % 3 == 0),
            status=ProductionStatus.SCHEDULED,
        ))

    return orders


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_test_config_with_overrides(**overrides):
    """Create test configuration with custom overrides."""
    from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
        HeatSchedulerConfig,
        TariffConfiguration,
        TariffType,
        DemandChargeConfiguration,
        LoadForecastingConfiguration,
        WeatherConfiguration,
    )

    defaults = {
        "agent_name": "Test Scheduler",
        "version": "1.0.0",
        "environment": "test",
        "tariffs": [TariffConfiguration(
            tariff_id="TEST-001",
            tariff_type=TariffType.TIME_OF_USE,
            peak_rate_per_kwh=0.15,
            off_peak_rate_per_kwh=0.06,
        )],
        "demand_charge": DemandChargeConfiguration(peak_demand_limit_kw=5000.0),
        "load_forecasting": LoadForecastingConfiguration(),
        "weather": WeatherConfiguration(),
    }

    defaults.update(overrides)
    return HeatSchedulerConfig(**defaults)


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "compliance: mark test as compliance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
