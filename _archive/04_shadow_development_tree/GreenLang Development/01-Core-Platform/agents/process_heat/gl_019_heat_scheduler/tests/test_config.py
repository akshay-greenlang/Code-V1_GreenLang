"""
GL-019 HEATSCHEDULER - Configuration Module Tests

Unit tests for configuration schemas including tariffs, equipment,
thermal storage, load forecasting, demand charge, and weather configurations.

Test Coverage:
    - Pydantic model validation
    - Field constraints and defaults
    - Model validators
    - Enum values
    - Configuration helpers

Author: GreenLang Test Team
Date: December 2025
"""

import pytest
from datetime import datetime, time, timezone
from pydantic import ValidationError


class TestTariffType:
    """Tests for TariffType enumeration."""

    def test_tariff_type_values(self):
        """Test all tariff type values exist."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import TariffType

        assert TariffType.TIME_OF_USE.value == "time_of_use"
        assert TariffType.DEMAND_CHARGE.value == "demand_charge"
        assert TariffType.REAL_TIME_PRICING.value == "real_time_pricing"
        assert TariffType.TIERED.value == "tiered"
        assert TariffType.FLAT_RATE.value == "flat_rate"
        assert TariffType.CRITICAL_PEAK.value == "critical_peak"

    def test_tariff_type_is_string_enum(self):
        """Test TariffType is a string enum."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import TariffType

        assert isinstance(TariffType.TIME_OF_USE.value, str)


class TestEquipmentType:
    """Tests for EquipmentType enumeration."""

    def test_equipment_type_values(self):
        """Test all equipment type values exist."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import EquipmentType

        assert EquipmentType.ELECTRIC_FURNACE.value == "electric_furnace"
        assert EquipmentType.GAS_FURNACE.value == "gas_furnace"
        assert EquipmentType.INDUCTION_FURNACE.value == "induction_furnace"
        assert EquipmentType.BOILER.value == "boiler"
        assert EquipmentType.HEAT_TREATMENT.value == "heat_treatment"
        assert EquipmentType.OVEN.value == "oven"
        assert EquipmentType.KILN.value == "kiln"
        assert EquipmentType.DRYER.value == "dryer"
        assert EquipmentType.HEAT_PUMP.value == "heat_pump"


class TestEquipmentStatus:
    """Tests for EquipmentStatus enumeration."""

    def test_equipment_status_values(self):
        """Test all equipment status values exist."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import EquipmentStatus

        assert EquipmentStatus.AVAILABLE.value == "available"
        assert EquipmentStatus.IN_USE.value == "in_use"
        assert EquipmentStatus.MAINTENANCE.value == "maintenance"
        assert EquipmentStatus.STANDBY.value == "standby"
        assert EquipmentStatus.FAULT.value == "fault"
        assert EquipmentStatus.OFFLINE.value == "offline"


class TestOptimizationObjective:
    """Tests for OptimizationObjective enumeration."""

    def test_optimization_objective_values(self):
        """Test all optimization objective values exist."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import OptimizationObjective

        assert OptimizationObjective.MINIMIZE_COST.value == "minimize_cost"
        assert OptimizationObjective.MINIMIZE_PEAK_DEMAND.value == "minimize_peak_demand"
        assert OptimizationObjective.MAXIMIZE_EFFICIENCY.value == "maximize_efficiency"
        assert OptimizationObjective.BALANCE_COST_DEMAND.value == "balance_cost_demand"
        assert OptimizationObjective.EARLIEST_COMPLETION.value == "earliest_completion"
        assert OptimizationObjective.MINIMIZE_EMISSIONS.value == "minimize_emissions"


class TestStorageType:
    """Tests for StorageType enumeration."""

    def test_storage_type_values(self):
        """Test all storage type values exist."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import StorageType

        assert StorageType.HOT_WATER_TANK.value == "hot_water_tank"
        assert StorageType.CHILLED_WATER_TANK.value == "chilled_water_tank"
        assert StorageType.PCM_STORAGE.value == "pcm_storage"
        assert StorageType.ICE_STORAGE.value == "ice_storage"
        assert StorageType.MOLTEN_SALT.value == "molten_salt"
        assert StorageType.STEAM_ACCUMULATOR.value == "steam_accumulator"


class TestForecastModel:
    """Tests for ForecastModel enumeration."""

    def test_forecast_model_values(self):
        """Test all forecast model values exist."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import ForecastModel

        assert ForecastModel.ARIMA.value == "arima"
        assert ForecastModel.PROPHET.value == "prophet"
        assert ForecastModel.LSTM.value == "lstm"
        assert ForecastModel.GRADIENT_BOOSTING.value == "gradient_boosting"
        assert ForecastModel.ENSEMBLE.value == "ensemble"
        assert ForecastModel.HYBRID.value == "hybrid"


class TestTariffConfiguration:
    """Tests for TariffConfiguration model."""

    def test_valid_tariff_configuration(self, sample_tariff_config):
        """Test valid tariff configuration creation."""
        assert sample_tariff_config.tariff_id == "TOU-001"
        assert sample_tariff_config.peak_rate_per_kwh == 0.15
        assert sample_tariff_config.off_peak_rate_per_kwh == 0.06
        assert sample_tariff_config.peak_hours_start == 14
        assert sample_tariff_config.peak_hours_end == 20

    def test_tariff_default_values(self):
        """Test tariff configuration default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            TariffConfiguration,
            TariffType,
        )

        tariff = TariffConfiguration(
            tariff_id="DEFAULT-001",
            tariff_type=TariffType.TIME_OF_USE,
        )

        assert tariff.peak_rate_per_kwh == 0.15
        assert tariff.off_peak_rate_per_kwh == 0.06
        assert tariff.peak_hours_start == 14
        assert tariff.peak_hours_end == 20
        assert tariff.weekend_off_peak is True
        assert tariff.holiday_off_peak is True
        assert tariff.demand_charge_per_kw == 0.0
        assert tariff.rtp_enabled is False

    def test_tariff_negative_rate_rejected(self):
        """Test that negative rates are rejected."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            TariffConfiguration,
            TariffType,
        )

        with pytest.raises(ValidationError):
            TariffConfiguration(
                tariff_id="INVALID",
                tariff_type=TariffType.TIME_OF_USE,
                peak_rate_per_kwh=-0.15,
            )

    def test_tariff_invalid_peak_hours(self):
        """Test that invalid peak hours are validated."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            TariffConfiguration,
            TariffType,
        )

        with pytest.raises(ValidationError):
            TariffConfiguration(
                tariff_id="INVALID",
                tariff_type=TariffType.TIME_OF_USE,
                peak_hours_start=25,  # Invalid hour
            )

    def test_tariff_rate_structure_validation_warning(self, caplog):
        """Test rate structure validation logs warning for inverted rates."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            TariffConfiguration,
            TariffType,
        )
        import logging

        caplog.set_level(logging.WARNING)

        # Peak rate lower than off-peak should log warning
        TariffConfiguration(
            tariff_id="INVERTED",
            tariff_type=TariffType.TIME_OF_USE,
            peak_rate_per_kwh=0.05,
            off_peak_rate_per_kwh=0.10,
        )

        assert "lower than off-peak" in caplog.text

    def test_tariff_ratchet_percentage_bounds(self):
        """Test ratchet percentage is bounded 0-100."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            TariffConfiguration,
            TariffType,
        )

        with pytest.raises(ValidationError):
            TariffConfiguration(
                tariff_id="INVALID",
                tariff_type=TariffType.TIME_OF_USE,
                ratchet_percentage=150.0,
            )


class TestEquipmentConfiguration:
    """Tests for EquipmentConfiguration model."""

    def test_valid_equipment_configuration(self, sample_equipment_config):
        """Test valid equipment configuration creation."""
        assert sample_equipment_config.equipment_id == "FURN-001"
        assert sample_equipment_config.capacity_kw == 500.0
        assert sample_equipment_config.efficiency == 0.92
        assert sample_equipment_config.max_temperature_c == 1200.0

    def test_equipment_default_values(self):
        """Test equipment configuration default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            EquipmentConfiguration,
            EquipmentType,
            EquipmentStatus,
        )

        equip = EquipmentConfiguration(
            equipment_id="EQUIP-001",
            equipment_type=EquipmentType.ELECTRIC_FURNACE,
            capacity_kw=100.0,
        )

        assert equip.min_power_kw == 0.0
        assert equip.standby_power_kw == 0.0
        assert equip.efficiency == 0.85
        assert equip.min_run_time_minutes == 30
        assert equip.status == EquipmentStatus.AVAILABLE
        assert equip.available_days == [0, 1, 2, 3, 4, 5, 6]

    def test_equipment_capacity_must_be_positive(self):
        """Test that capacity must be positive."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            EquipmentConfiguration,
            EquipmentType,
        )

        with pytest.raises(ValidationError):
            EquipmentConfiguration(
                equipment_id="INVALID",
                equipment_type=EquipmentType.ELECTRIC_FURNACE,
                capacity_kw=0.0,
            )

    def test_equipment_efficiency_bounds(self):
        """Test efficiency is bounded 0.5-1.0."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            EquipmentConfiguration,
            EquipmentType,
        )

        with pytest.raises(ValidationError):
            EquipmentConfiguration(
                equipment_id="INVALID",
                equipment_type=EquipmentType.ELECTRIC_FURNACE,
                capacity_kw=100.0,
                efficiency=1.5,
            )

    def test_equipment_temperature_range_validation(self):
        """Test temperature range validation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            EquipmentConfiguration,
            EquipmentType,
        )

        with pytest.raises(ValidationError):
            EquipmentConfiguration(
                equipment_id="INVALID",
                equipment_type=EquipmentType.ELECTRIC_FURNACE,
                capacity_kw=100.0,
                min_temperature_c=1000.0,
                max_temperature_c=500.0,  # Min > Max
            )


class TestThermalStorageConfiguration:
    """Tests for ThermalStorageConfiguration model."""

    def test_valid_storage_configuration(self, sample_storage_config):
        """Test valid storage configuration creation."""
        assert sample_storage_config.storage_id == "TES-001"
        assert sample_storage_config.capacity_kwh == 5000.0
        assert sample_storage_config.round_trip_efficiency == 0.92
        assert sample_storage_config.min_soc_pct == 10.0
        assert sample_storage_config.max_soc_pct == 95.0

    def test_storage_default_values(self):
        """Test storage configuration default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            ThermalStorageConfiguration,
            StorageType,
        )

        storage = ThermalStorageConfiguration(
            storage_id="STOR-001",
            storage_type=StorageType.HOT_WATER_TANK,
            capacity_kwh=1000.0,
            max_charge_rate_kw=100.0,
            max_discharge_rate_kw=100.0,
        )

        assert storage.enabled is True
        assert storage.round_trip_efficiency == 0.90
        assert storage.standby_loss_pct_per_hour == 0.5
        assert storage.min_soc_pct == 10.0
        assert storage.max_soc_pct == 95.0
        assert storage.current_soc_pct == 50.0

    def test_storage_soc_validation(self):
        """Test SOC limits validation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            ThermalStorageConfiguration,
            StorageType,
        )

        with pytest.raises(ValidationError):
            ThermalStorageConfiguration(
                storage_id="INVALID",
                storage_type=StorageType.HOT_WATER_TANK,
                capacity_kwh=1000.0,
                max_charge_rate_kw=100.0,
                max_discharge_rate_kw=100.0,
                min_soc_pct=80.0,
                max_soc_pct=50.0,  # Min > Max
            )

    def test_storage_efficiency_bounds(self):
        """Test efficiency is bounded 0.5-1.0."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            ThermalStorageConfiguration,
            StorageType,
        )

        with pytest.raises(ValidationError):
            ThermalStorageConfiguration(
                storage_id="INVALID",
                storage_type=StorageType.HOT_WATER_TANK,
                capacity_kwh=1000.0,
                max_charge_rate_kw=100.0,
                max_discharge_rate_kw=100.0,
                round_trip_efficiency=0.3,  # Below minimum
            )

    def test_pcm_storage_configuration(self, sample_pcm_storage_config):
        """Test PCM storage specific fields."""
        assert sample_pcm_storage_config.pcm_melt_temperature_c == 58.0
        assert sample_pcm_storage_config.pcm_latent_heat_kj_kg == 200.0


class TestLoadForecastingConfiguration:
    """Tests for LoadForecastingConfiguration model."""

    def test_valid_forecasting_configuration(self, sample_load_forecasting_config):
        """Test valid forecasting configuration creation."""
        assert sample_load_forecasting_config.enabled is True
        assert sample_load_forecasting_config.forecast_horizon_hours == 48
        assert sample_load_forecasting_config.confidence_level == 0.90

    def test_forecasting_default_values(self):
        """Test forecasting configuration default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            LoadForecastingConfiguration,
            ForecastModel,
        )

        config = LoadForecastingConfiguration()

        assert config.enabled is True
        assert config.model_type == ForecastModel.ENSEMBLE
        assert config.forecast_horizon_hours == 48
        assert config.granularity_minutes == 15
        assert config.lookback_days == 30
        assert config.use_weather_features is True
        assert config.use_calendar_features is True

    def test_forecasting_horizon_bounds(self):
        """Test forecast horizon bounds."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            LoadForecastingConfiguration,
        )

        with pytest.raises(ValidationError):
            LoadForecastingConfiguration(forecast_horizon_hours=200)  # Max is 168

    def test_forecasting_confidence_bounds(self):
        """Test confidence level bounds."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            LoadForecastingConfiguration,
        )

        with pytest.raises(ValidationError):
            LoadForecastingConfiguration(confidence_level=1.5)  # Max is 0.99


class TestDemandChargeConfiguration:
    """Tests for DemandChargeConfiguration model."""

    def test_valid_demand_charge_configuration(self, sample_demand_charge_config):
        """Test valid demand charge configuration creation."""
        assert sample_demand_charge_config.enabled is True
        assert sample_demand_charge_config.peak_demand_limit_kw == 5000.0
        assert sample_demand_charge_config.enable_load_shifting is True
        assert sample_demand_charge_config.max_shift_hours == 4

    def test_demand_charge_default_values(self):
        """Test demand charge configuration default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            DemandChargeConfiguration,
        )

        config = DemandChargeConfiguration(peak_demand_limit_kw=3000.0)

        assert config.enabled is True
        assert config.demand_interval_minutes == 15
        assert config.rolling_demand_average is True
        assert config.enable_load_shifting is True
        assert config.enable_demand_response is True

    def test_demand_limit_must_be_positive(self):
        """Test peak demand limit must be positive."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            DemandChargeConfiguration,
        )

        with pytest.raises(ValidationError):
            DemandChargeConfiguration(peak_demand_limit_kw=0.0)


class TestWeatherConfiguration:
    """Tests for WeatherConfiguration model."""

    def test_valid_weather_configuration(self, sample_weather_config):
        """Test valid weather configuration creation."""
        assert sample_weather_config.enabled is True
        assert sample_weather_config.latitude == 37.7749
        assert sample_weather_config.longitude == -122.4194
        assert sample_weather_config.api_provider == "openweathermap"

    def test_weather_default_values(self):
        """Test weather configuration default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            WeatherConfiguration,
        )

        config = WeatherConfiguration()

        assert config.enabled is True
        assert config.latitude == 37.7749
        assert config.longitude == -122.4194
        assert config.timezone == "America/Los_Angeles"
        assert config.heating_base_temp_c == 18.0
        assert config.cooling_base_temp_c == 24.0

    def test_weather_latitude_bounds(self):
        """Test latitude bounds -90 to 90."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            WeatherConfiguration,
        )

        with pytest.raises(ValidationError):
            WeatherConfiguration(latitude=100.0)

    def test_weather_longitude_bounds(self):
        """Test longitude bounds -180 to 180."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            WeatherConfiguration,
        )

        with pytest.raises(ValidationError):
            WeatherConfiguration(longitude=200.0)


class TestOptimizationParameters:
    """Tests for OptimizationParameters model."""

    def test_valid_optimization_parameters(self, sample_optimization_params):
        """Test valid optimization parameters creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            OptimizationObjective,
        )

        assert sample_optimization_params.primary_objective == OptimizationObjective.MINIMIZE_COST
        assert sample_optimization_params.cost_weight == 0.6
        assert sample_optimization_params.demand_weight == 0.3
        assert sample_optimization_params.efficiency_weight == 0.1

    def test_optimization_default_values(self):
        """Test optimization parameters default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            OptimizationParameters,
            OptimizationObjective,
        )

        params = OptimizationParameters()

        assert params.primary_objective == OptimizationObjective.MINIMIZE_COST
        assert params.cost_weight == 0.6
        assert params.demand_weight == 0.3
        assert params.efficiency_weight == 0.1
        assert params.enable_peak_shaving is True

    def test_optimization_weights_warning(self, caplog):
        """Test weights not summing to 1.0 logs warning."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            OptimizationParameters,
        )
        import logging

        caplog.set_level(logging.WARNING)

        OptimizationParameters(
            cost_weight=0.5,
            demand_weight=0.5,
            efficiency_weight=0.5,  # Sum = 1.5, not 1.0
        )

        assert "not 1.0" in caplog.text


class TestHeatSchedulerConfig:
    """Tests for HeatSchedulerConfig model."""

    def test_valid_scheduler_config(self, sample_scheduler_config):
        """Test valid scheduler configuration creation."""
        assert sample_scheduler_config.agent_name == "Test Heat Scheduler"
        assert sample_scheduler_config.version == "1.0.0"
        assert len(sample_scheduler_config.tariffs) == 1
        assert len(sample_scheduler_config.equipment) == 1
        assert len(sample_scheduler_config.thermal_storage) == 1

    def test_scheduler_config_default_values(self):
        """Test scheduler configuration default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            HeatSchedulerConfig,
            DemandChargeConfiguration,
        )

        config = HeatSchedulerConfig(
            demand_charge=DemandChargeConfiguration(peak_demand_limit_kw=5000.0)
        )

        assert config.agent_name == "GL-019 HEATSCHEDULER"
        assert config.version == "1.0.0"
        assert config.environment == "production"
        assert config.optimization_interval_minutes == 15
        assert config.auto_apply_schedule is False
        assert config.schedule_lookahead_hours == 24

    def test_scheduler_get_tariff(self, sample_scheduler_config):
        """Test get_tariff helper method."""
        tariff = sample_scheduler_config.get_tariff("TOU-001")
        assert tariff is not None
        assert tariff.tariff_id == "TOU-001"

        # Test non-existent tariff
        assert sample_scheduler_config.get_tariff("NONEXISTENT") is None

    def test_scheduler_get_equipment(self, sample_scheduler_config):
        """Test get_equipment helper method."""
        equip = sample_scheduler_config.get_equipment("FURN-001")
        assert equip is not None
        assert equip.equipment_id == "FURN-001"

        # Test non-existent equipment
        assert sample_scheduler_config.get_equipment("NONEXISTENT") is None

    def test_scheduler_get_storage(self, sample_scheduler_config):
        """Test get_storage helper method."""
        storage = sample_scheduler_config.get_storage("TES-001")
        assert storage is not None
        assert storage.storage_id == "TES-001"

        # Test non-existent storage
        assert sample_scheduler_config.get_storage("NONEXISTENT") is None

    def test_scheduler_get_available_equipment(self, sample_scheduler_config):
        """Test get_available_equipment helper method."""
        available = sample_scheduler_config.get_available_equipment()
        assert len(available) == 1
        assert available[0].equipment_id == "FURN-001"

    def test_scheduler_config_optimization_interval_bounds(self):
        """Test optimization interval bounds."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            HeatSchedulerConfig,
            DemandChargeConfiguration,
        )

        with pytest.raises(ValidationError):
            HeatSchedulerConfig(
                demand_charge=DemandChargeConfiguration(peak_demand_limit_kw=5000.0),
                optimization_interval_minutes=3,  # Below minimum of 5
            )


class TestERPIntegration:
    """Tests for ERPIntegration model."""

    def test_erp_integration_default_values(self):
        """Test ERP integration default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            ERPIntegration,
        )

        erp = ERPIntegration()

        assert erp.enabled is True
        assert erp.polling_interval_seconds == 60
        assert erp.sync_mode == "pull"
        assert erp.auth_type == "api_key"

    def test_erp_integration_polling_bounds(self):
        """Test polling interval bounds."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            ERPIntegration,
        )

        with pytest.raises(ValidationError):
            ERPIntegration(polling_interval_seconds=5)  # Below minimum of 10


class TestControlSystemIntegration:
    """Tests for ControlSystemIntegration model."""

    def test_control_system_default_values(self):
        """Test control system integration default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            ControlSystemIntegration,
        )

        ctrl = ControlSystemIntegration()

        assert ctrl.enabled is True
        assert ctrl.connection_protocol == "opcua"
        assert ctrl.enable_safety_checks is True
        assert ctrl.max_setpoint_change_rate == 10.0
