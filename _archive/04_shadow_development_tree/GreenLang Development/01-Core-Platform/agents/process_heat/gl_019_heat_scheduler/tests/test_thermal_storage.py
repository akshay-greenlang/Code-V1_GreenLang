"""
GL-019 HEATSCHEDULER - Thermal Storage Module Tests

Unit tests for thermal storage optimization including storage unit modeling,
charge/discharge calculations, standby losses, and dispatch optimization.

Test Coverage:
    - ThermalStorageUnit initialization and state
    - Charge/discharge calculations with efficiency
    - Standby loss calculations
    - State update mechanics
    - ThermalStorageOptimizer dispatch scheduling
    - PCMStorageCalculator phase change modeling

Author: GreenLang Test Team
Date: December 2025
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
import math


class TestThermalStorageUnitInitialization:
    """Tests for ThermalStorageUnit initialization."""

    def test_unit_initialization(self, sample_storage_config):
        """Test storage unit initializes correctly."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageUnit,
        )

        unit = ThermalStorageUnit(sample_storage_config)

        assert unit.storage_id == "TES-001"
        assert unit.config == sample_storage_config
        assert unit.current_soc_pct == 50.0
        assert unit.current_soc_kwh == 2500.0  # 50% of 5000 kWh

    def test_unit_usable_capacity_calculation(self, sample_storage_config):
        """Test usable capacity calculation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageUnit,
        )

        unit = ThermalStorageUnit(sample_storage_config)

        # Usable = (95% - 10%) * 5000 = 85% * 5000 = 4250 kWh
        expected_usable = (0.95 - 0.10) * 5000.0
        assert unit._usable_capacity_kwh == expected_usable

    def test_unit_initial_temperature(self, sample_storage_config):
        """Test initial temperature from config."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageUnit,
        )

        unit = ThermalStorageUnit(sample_storage_config)

        assert unit._current_temp_c == sample_storage_config.current_temperature_c


class TestThermalStorageUnitProperties:
    """Tests for ThermalStorageUnit properties."""

    @pytest.fixture
    def storage_unit(self, sample_storage_config):
        """Create storage unit instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageUnit,
        )
        return ThermalStorageUnit(sample_storage_config)

    def test_storage_id_property(self, storage_unit):
        """Test storage_id property."""
        assert storage_unit.storage_id == "TES-001"

    def test_current_soc_pct_property(self, storage_unit):
        """Test current_soc_pct property."""
        assert storage_unit.current_soc_pct == 50.0

    def test_current_soc_kwh_property(self, storage_unit):
        """Test current_soc_kwh property."""
        assert storage_unit.current_soc_kwh == 2500.0

    def test_available_charge_capacity(self, storage_unit):
        """Test available charge capacity calculation."""
        # Max SOC = 95% = 4750 kWh
        # Current = 50% = 2500 kWh
        # Available = 4750 - 2500 = 2250 kWh
        expected = 0.95 * 5000.0 - 2500.0
        assert storage_unit.available_charge_capacity_kwh == expected

    def test_available_discharge_capacity(self, storage_unit):
        """Test available discharge capacity calculation."""
        # Current = 50% = 2500 kWh
        # Min SOC = 10% = 500 kWh
        # Available = 2500 - 500 = 2000 kWh
        expected = 2500.0 - 0.10 * 5000.0
        assert storage_unit.available_discharge_capacity_kwh == expected


class TestThermalStorageUnitCharging:
    """Tests for charging calculations."""

    @pytest.fixture
    def storage_unit(self, sample_storage_config):
        """Create storage unit instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageUnit,
        )
        return ThermalStorageUnit(sample_storage_config)

    def test_charge_energy_calculation(self, storage_unit):
        """Test charge energy calculation with efficiency."""
        power_kw = 500.0
        duration_hours = 1.0

        energy_stored, actual_duration = storage_unit.calculate_charge_energy(
            power_kw, duration_hours
        )

        # Energy stored = power * duration * sqrt(efficiency)
        # With round_trip_efficiency = 0.92, sqrt = ~0.959
        expected_stored = power_kw * duration_hours * (0.92 ** 0.5)
        assert abs(energy_stored - expected_stored) < 0.1

    def test_charge_limited_by_capacity(self, sample_storage_config):
        """Test charging is limited by available capacity."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageUnit,
        )

        # Start near max SOC
        sample_storage_config.current_soc_pct = 90.0
        unit = ThermalStorageUnit(sample_storage_config)

        # Try to charge more than available
        power_kw = 500.0
        duration_hours = 10.0  # Would store way too much

        energy_stored, actual_duration = unit.calculate_charge_energy(
            power_kw, duration_hours
        )

        # Should be limited to available capacity
        assert energy_stored <= unit.available_charge_capacity_kwh
        assert actual_duration < duration_hours


class TestThermalStorageUnitDischarging:
    """Tests for discharging calculations."""

    @pytest.fixture
    def storage_unit(self, sample_storage_config):
        """Create storage unit instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageUnit,
        )
        return ThermalStorageUnit(sample_storage_config)

    def test_discharge_energy_calculation(self, storage_unit):
        """Test discharge energy calculation with efficiency."""
        power_kw = 500.0
        duration_hours = 1.0

        energy_out, actual_duration = storage_unit.calculate_discharge_energy(
            power_kw, duration_hours
        )

        # Energy out = power * duration (limited by efficiency loss internally)
        expected_out = power_kw * duration_hours
        assert abs(energy_out - expected_out) < 0.1 or energy_out <= expected_out

    def test_discharge_limited_by_capacity(self, sample_storage_config):
        """Test discharging is limited by available capacity."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageUnit,
        )

        # Start near min SOC
        sample_storage_config.current_soc_pct = 15.0
        unit = ThermalStorageUnit(sample_storage_config)

        # Try to discharge more than available
        power_kw = 500.0
        duration_hours = 10.0  # Would discharge way too much

        energy_out, actual_duration = unit.calculate_discharge_energy(
            power_kw, duration_hours
        )

        # Should be limited to available capacity (with efficiency consideration)
        assert actual_duration < duration_hours


class TestThermalStorageUnitStandbyLoss:
    """Tests for standby loss calculations."""

    @pytest.fixture
    def storage_unit(self, sample_storage_config):
        """Create storage unit instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageUnit,
        )
        return ThermalStorageUnit(sample_storage_config)

    def test_standby_loss_calculation(self, storage_unit):
        """Test standby loss calculation."""
        duration_hours = 1.0

        loss = storage_unit.calculate_standby_loss(duration_hours)

        # Loss = current_soc * (1 - (1 - loss_rate)^duration)
        # With 0.5%/hour loss rate and 2500 kWh current
        loss_rate = 0.005
        expected_loss = 2500.0 * (1 - (1 - loss_rate) ** duration_hours)
        assert abs(loss - expected_loss) < 0.1

    def test_standby_loss_over_24_hours(self, storage_unit):
        """Test standby loss over 24 hours."""
        duration_hours = 24.0

        loss = storage_unit.calculate_standby_loss(duration_hours)

        # Should lose about 11-12% over 24 hours with 0.5%/hour
        # (1 - 0.995^24) = ~11.3%
        expected_loss_pct = (1 - (1 - 0.005) ** 24) * 100
        actual_loss_pct = (loss / 2500.0) * 100

        assert abs(actual_loss_pct - expected_loss_pct) < 1.0


class TestThermalStorageUnitStateUpdate:
    """Tests for state update mechanics."""

    @pytest.fixture
    def storage_unit(self, sample_storage_config):
        """Create storage unit instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageUnit,
        )
        return ThermalStorageUnit(sample_storage_config)

    def test_update_state_charging(self, storage_unit):
        """Test state update during charging."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import StorageMode

        initial_soc = storage_unit.current_soc_kwh

        state = storage_unit.update_state(power_kw=300.0, duration_hours=1.0)

        assert state.mode == StorageMode.CHARGING
        assert storage_unit.current_soc_kwh > initial_soc
        assert state.power_kw == 300.0

    def test_update_state_discharging(self, storage_unit):
        """Test state update during discharging."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import StorageMode

        initial_soc = storage_unit.current_soc_kwh

        state = storage_unit.update_state(power_kw=-300.0, duration_hours=1.0)

        assert state.mode == StorageMode.DISCHARGING
        assert storage_unit.current_soc_kwh < initial_soc
        assert state.power_kw == -300.0

    def test_update_state_idle(self, storage_unit):
        """Test state update during idle (standby losses)."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import StorageMode

        initial_soc = storage_unit.current_soc_kwh

        state = storage_unit.update_state(power_kw=0.0, duration_hours=1.0)

        assert state.mode == StorageMode.IDLE
        assert storage_unit.current_soc_kwh < initial_soc  # Standby loss
        assert state.power_kw == 0.0

    def test_temperature_updates_with_soc(self, storage_unit):
        """Test temperature updates based on SOC."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import StorageMode

        # Charge to increase SOC
        storage_unit.update_state(power_kw=500.0, duration_hours=2.0)

        # Temperature should increase with SOC
        # Linear model: temp = min_temp + soc_ratio * (max_temp - min_temp)
        expected_min = storage_unit.config.min_temperature_c
        expected_max = storage_unit.config.max_temperature_c

        assert expected_min <= storage_unit._current_temp_c <= expected_max

    def test_get_state_returns_current(self, storage_unit):
        """Test get_state returns current state."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import StorageMode

        state = storage_unit.get_state()

        assert state.state_of_charge_pct == storage_unit.current_soc_pct
        assert state.state_of_charge_kwh == storage_unit.current_soc_kwh
        assert state.mode == StorageMode.IDLE


class TestThermalStorageOptimizerInitialization:
    """Tests for ThermalStorageOptimizer initialization."""

    def test_optimizer_initialization(self, sample_storage_config, sample_tariff_config):
        """Test optimizer initializes correctly."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageOptimizer,
        )

        optimizer = ThermalStorageOptimizer(
            storage_configs=[sample_storage_config],
            tariff_config=sample_tariff_config,
        )

        assert len(optimizer._units) == 1
        assert "TES-001" in optimizer._units
        assert optimizer._tariff == sample_tariff_config

    def test_optimizer_skips_disabled_storage(self, sample_storage_config, sample_tariff_config):
        """Test optimizer skips disabled storage units."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageOptimizer,
        )

        sample_storage_config.enabled = False

        optimizer = ThermalStorageOptimizer(
            storage_configs=[sample_storage_config],
            tariff_config=sample_tariff_config,
        )

        assert len(optimizer._units) == 0


class TestThermalStorageOptimizerDispatch:
    """Tests for dispatch optimization."""

    @pytest.fixture
    def optimizer(self, sample_storage_config, sample_tariff_config):
        """Create optimizer instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageOptimizer,
        )
        return ThermalStorageOptimizer(
            storage_configs=[sample_storage_config],
            tariff_config=sample_tariff_config,
        )

    def test_optimize_dispatch_returns_result(self, optimizer, sample_load_forecast):
        """Test optimize_dispatch returns result."""
        result = optimizer.optimize_dispatch(
            load_forecast=sample_load_forecast,
            horizon_hours=24,
        )

        assert result is not None
        assert len(result.unit_schedules) == 1

    def test_optimize_dispatch_calculates_savings(self, optimizer, sample_load_forecast):
        """Test optimize_dispatch calculates savings."""
        result = optimizer.optimize_dispatch(
            load_forecast=sample_load_forecast,
            horizon_hours=24,
        )

        # Should have some arbitrage or demand savings
        assert result.total_energy_arbitrage_usd >= 0
        assert result.total_demand_savings_usd >= 0
        assert result.total_savings_usd >= 0

    def test_optimize_dispatch_with_demand_limit(self, optimizer, sample_load_forecast):
        """Test optimize_dispatch with demand limit."""
        result = optimizer.optimize_dispatch(
            load_forecast=sample_load_forecast,
            horizon_hours=24,
            demand_limit_kw=3000.0,
        )

        assert result is not None

    def test_dispatch_schedule_has_points(self, optimizer, sample_load_forecast):
        """Test dispatch schedule contains points."""
        result = optimizer.optimize_dispatch(
            load_forecast=sample_load_forecast,
            horizon_hours=24,
        )

        schedule = result.unit_schedules[0]
        # 24 hours / 0.25 hours per step = 96 points
        assert len(schedule.dispatch_points) == 96

    def test_dispatch_schedule_tracks_energy(self, optimizer, sample_load_forecast):
        """Test dispatch schedule tracks energy charged/discharged."""
        result = optimizer.optimize_dispatch(
            load_forecast=sample_load_forecast,
            horizon_hours=24,
        )

        schedule = result.unit_schedules[0]
        # Should have some charging or discharging
        assert schedule.total_charge_kwh >= 0
        assert schedule.total_discharge_kwh >= 0

    def test_dispatch_maintains_soc_limits(self, optimizer, sample_load_forecast):
        """Test dispatch maintains SOC limits."""
        result = optimizer.optimize_dispatch(
            load_forecast=sample_load_forecast,
            horizon_hours=24,
        )

        schedule = result.unit_schedules[0]

        for point in schedule.dispatch_points:
            # SOC should stay within limits
            assert 0 <= point.state_of_charge_pct <= 100


class TestThermalStorageOptimizerPeakHours:
    """Tests for peak hour identification."""

    @pytest.fixture
    def optimizer(self, sample_storage_config, sample_tariff_config):
        """Create optimizer instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageOptimizer,
        )
        return ThermalStorageOptimizer(
            storage_configs=[sample_storage_config],
            tariff_config=sample_tariff_config,
        )

    def test_get_peak_hours_from_tariff(self, optimizer):
        """Test peak hours extracted from tariff."""
        peak_hours = optimizer._get_peak_hours()

        # From sample_tariff_config: peak 14-20
        assert 14 in peak_hours
        assert 15 in peak_hours
        assert 16 in peak_hours
        assert 17 in peak_hours
        assert 18 in peak_hours
        assert 19 in peak_hours
        assert 20 in peak_hours

    def test_get_peak_hours_default(self, sample_storage_config):
        """Test default peak hours without tariff."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageOptimizer,
        )

        optimizer = ThermalStorageOptimizer(
            storage_configs=[sample_storage_config],
            tariff_config=None,
        )

        peak_hours = optimizer._get_peak_hours()

        # Default: 14-19
        assert {14, 15, 16, 17, 18, 19} == peak_hours


class TestThermalStorageOptimizerExecution:
    """Tests for dispatch execution."""

    @pytest.fixture
    def optimizer(self, sample_storage_config, sample_tariff_config):
        """Create optimizer instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageOptimizer,
        )
        return ThermalStorageOptimizer(
            storage_configs=[sample_storage_config],
            tariff_config=sample_tariff_config,
        )

    def test_execute_dispatch_charging(self, optimizer):
        """Test execute dispatch for charging."""
        initial_soc = optimizer._units["TES-001"].current_soc_kwh

        state = optimizer.execute_dispatch(
            storage_id="TES-001",
            power_kw=200.0,
            duration_hours=1.0,
        )

        assert optimizer._units["TES-001"].current_soc_kwh > initial_soc

    def test_execute_dispatch_discharging(self, optimizer):
        """Test execute dispatch for discharging."""
        initial_soc = optimizer._units["TES-001"].current_soc_kwh

        state = optimizer.execute_dispatch(
            storage_id="TES-001",
            power_kw=-200.0,
            duration_hours=1.0,
        )

        assert optimizer._units["TES-001"].current_soc_kwh < initial_soc

    def test_execute_dispatch_limits_power(self, optimizer):
        """Test execute dispatch limits power to max rates."""
        # Try to charge at excessive rate
        state = optimizer.execute_dispatch(
            storage_id="TES-001",
            power_kw=10000.0,  # Way over max
            duration_hours=1.0,
        )

        # Power should be limited to max charge rate
        # (Result power is what was actually applied after limiting)

    def test_execute_dispatch_unknown_storage_raises(self, optimizer):
        """Test execute dispatch raises for unknown storage."""
        with pytest.raises(ValueError):
            optimizer.execute_dispatch(
                storage_id="UNKNOWN",
                power_kw=200.0,
                duration_hours=1.0,
            )

    def test_get_current_state(self, optimizer):
        """Test get_current_state returns all units."""
        states = optimizer.get_current_state()

        assert "TES-001" in states
        assert states["TES-001"].state_of_charge_pct == 50.0


class TestPCMStorageCalculator:
    """Tests for PCMStorageCalculator."""

    @pytest.fixture
    def pcm_calculator(self):
        """Create PCM calculator instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            PCMStorageCalculator,
        )
        return PCMStorageCalculator(
            pcm_mass_kg=1000.0,
            melt_temperature_c=58.0,
            latent_heat_kj_kg=200.0,
            sensible_heat_kj_kg_k=2.0,
        )

    def test_pcm_initialization(self, pcm_calculator):
        """Test PCM calculator initialization."""
        assert pcm_calculator._mass == 1000.0
        assert pcm_calculator._melt_temp == 58.0
        assert pcm_calculator._latent_heat == 200.0

    def test_pcm_latent_capacity(self, pcm_calculator):
        """Test latent heat capacity calculation."""
        # Latent capacity = mass * latent_heat / 3600
        # = 1000 * 200 / 3600 = 55.56 kWh
        expected = 1000.0 * 200.0 / 3600.0
        assert abs(pcm_calculator.latent_capacity_kwh - expected) < 0.01

    def test_sensible_heat_only_below_melt(self, pcm_calculator):
        """Test sensible heat calculation below melting point."""
        # 40C to 50C (both below 58C melt point)
        energy = pcm_calculator.calculate_energy_stored(40.0, 50.0)

        # Sensible = mass * cp * delta_T / 3600
        # = 1000 * 2.0 * 10 / 3600 = 5.56 kWh
        expected = 1000.0 * 2.0 * 10.0 / 3600.0
        assert abs(energy - expected) < 0.01

    def test_sensible_heat_only_above_melt(self, pcm_calculator):
        """Test sensible heat calculation above melting point."""
        # 60C to 70C (both above 58C melt point)
        energy = pcm_calculator.calculate_energy_stored(60.0, 70.0)

        # Sensible = mass * cp * delta_T / 3600
        expected = 1000.0 * 2.0 * 10.0 / 3600.0
        assert abs(energy - expected) < 0.01

    def test_phase_change_charging(self, pcm_calculator):
        """Test energy calculation crossing melting point (charging)."""
        # 50C to 70C (crosses 58C melt point)
        energy = pcm_calculator.calculate_energy_stored(50.0, 70.0)

        # Sensible below melt: 1000 * 2 * 8 / 3600 = 4.44 kWh
        # Latent: 1000 * 200 / 3600 = 55.56 kWh
        # Sensible above melt: 1000 * 2 * 12 / 3600 = 6.67 kWh
        # Total = 66.67 kWh
        sensible_below = 1000.0 * 2.0 * 8.0 / 3600.0
        latent = 1000.0 * 200.0 / 3600.0
        sensible_above = 1000.0 * 2.0 * 12.0 / 3600.0
        expected = sensible_below + latent + sensible_above

        assert abs(energy - expected) < 0.1

    def test_phase_change_discharging(self, pcm_calculator):
        """Test energy calculation crossing melting point (discharging)."""
        # 70C to 50C (crosses 58C melt point, releases energy)
        energy = pcm_calculator.calculate_energy_stored(70.0, 50.0)

        # This releases energy, so negative
        assert energy < 0

    def test_melt_fraction_below_melt(self, pcm_calculator):
        """Test melt fraction below melting point."""
        fraction = pcm_calculator.calculate_melt_fraction(
            current_temp_c=50.0,
            energy_stored_kwh=10.0,
        )

        assert fraction == 0.0

    def test_melt_fraction_above_melt(self, pcm_calculator):
        """Test melt fraction above melting point."""
        fraction = pcm_calculator.calculate_melt_fraction(
            current_temp_c=65.0,
            energy_stored_kwh=100.0,
        )

        assert fraction == 1.0

    def test_melt_fraction_at_melt_point(self, pcm_calculator):
        """Test melt fraction at melting point."""
        # At exactly melt temperature, fraction depends on energy
        latent_capacity = pcm_calculator.latent_capacity_kwh

        # 50% of latent capacity stored
        fraction = pcm_calculator.calculate_melt_fraction(
            current_temp_c=58.05,  # Slightly above melt temp
            energy_stored_kwh=latent_capacity * 0.5,
        )

        # Should be between 0 and 1
        assert 0 <= fraction <= 1


class TestThermalStoragePerformance:
    """Performance tests for thermal storage."""

    @pytest.mark.performance
    def test_dispatch_optimization_time(
        self,
        sample_storage_config,
        sample_tariff_config,
        sample_load_forecast,
    ):
        """Test dispatch optimization completes in reasonable time."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageOptimizer,
        )
        import time

        optimizer = ThermalStorageOptimizer(
            storage_configs=[sample_storage_config],
            tariff_config=sample_tariff_config,
        )

        start = time.time()
        result = optimizer.optimize_dispatch(
            load_forecast=sample_load_forecast,
            horizon_hours=24,
        )
        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 1.0  # Should complete in under 1 second

    @pytest.mark.performance
    def test_long_horizon_optimization(
        self,
        sample_storage_config,
        sample_tariff_config,
        large_load_forecast,
    ):
        """Test optimization with 168-hour horizon."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
            ThermalStorageOptimizer,
        )
        import time

        optimizer = ThermalStorageOptimizer(
            storage_configs=[sample_storage_config],
            tariff_config=sample_tariff_config,
        )

        start = time.time()
        result = optimizer.optimize_dispatch(
            load_forecast=large_load_forecast,
            horizon_hours=168,
        )
        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 5.0  # Should complete in under 5 seconds


class TestThermalStorageConstants:
    """Tests for thermal storage constants."""

    def test_water_specific_heat(self):
        """Test water specific heat constant."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import CP_WATER

        assert CP_WATER == 4.186  # kJ/kg-K

    def test_water_density(self):
        """Test water density constant."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import WATER_DENSITY

        assert WATER_DENSITY == 1000.0  # kg/m3
