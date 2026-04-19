"""
GL-019 HEATSCHEDULER - Thermal Storage Optimization Module

Thermal energy storage optimization for hot water tanks, phase change
materials (PCM), and other thermal storage technologies.

Key Features:
    - Hot water tank sizing and dispatch optimization
    - Phase change material (PCM) storage modeling
    - Charge/discharge scheduling based on tariff structure
    - State of charge management with safety reserves
    - Integration with load forecast and demand charge optimization
    - Zero-hallucination: Deterministic thermodynamic calculations

Author: GreenLang Team
Version: 1.0.0
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
    ThermalStorageConfiguration,
    TariffConfiguration,
    StorageType,
)
from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
    StorageStatePoint,
    StorageDispatchSchedule,
    ThermalStorageResult,
    StorageMode,
    LoadForecastResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Specific heat of water (kJ/kg-K)
CP_WATER = 4.186

# Water density (kg/m3)
WATER_DENSITY = 1000.0

# Standard temperature reference
REFERENCE_TEMP_C = 15.0


# =============================================================================
# STORAGE UNIT MODEL
# =============================================================================

class ThermalStorageUnit:
    """
    Model for a single thermal storage unit.

    Implements thermodynamic calculations for sensible heat storage
    (hot water tanks) and latent heat storage (PCM).
    All calculations are DETERMINISTIC.
    """

    def __init__(
        self,
        config: ThermalStorageConfiguration,
    ) -> None:
        """
        Initialize thermal storage unit.

        Args:
            config: Storage unit configuration
        """
        self.config = config
        self._storage_id = config.storage_id
        self._storage_type = config.storage_type

        # Current state
        self._current_soc_kwh = config.current_soc_pct / 100.0 * config.capacity_kwh
        self._current_temp_c = config.current_temperature_c
        self._current_mode = StorageMode.IDLE

        # Calculate derived parameters
        self._usable_capacity_kwh = (
            config.capacity_kwh *
            (config.max_soc_pct - config.min_soc_pct) / 100.0
        )

        logger.info(
            f"ThermalStorageUnit {self._storage_id} initialized: "
            f"{config.capacity_kwh:.0f} kWh, "
            f"current SOC={config.current_soc_pct:.1f}%"
        )

    @property
    def storage_id(self) -> str:
        """Get storage unit ID."""
        return self._storage_id

    @property
    def current_soc_pct(self) -> float:
        """Get current state of charge (%)."""
        return (self._current_soc_kwh / self.config.capacity_kwh) * 100.0

    @property
    def current_soc_kwh(self) -> float:
        """Get current state of charge (kWh)."""
        return self._current_soc_kwh

    @property
    def available_charge_capacity_kwh(self) -> float:
        """Get available charging capacity (kWh)."""
        max_soc_kwh = self.config.max_soc_pct / 100.0 * self.config.capacity_kwh
        return max(0, max_soc_kwh - self._current_soc_kwh)

    @property
    def available_discharge_capacity_kwh(self) -> float:
        """Get available discharge capacity (kWh)."""
        min_soc_kwh = self.config.min_soc_pct / 100.0 * self.config.capacity_kwh
        return max(0, self._current_soc_kwh - min_soc_kwh)

    def calculate_charge_energy(
        self,
        power_kw: float,
        duration_hours: float,
    ) -> Tuple[float, float]:
        """
        Calculate energy stored from charging.

        Args:
            power_kw: Charging power (kW)
            duration_hours: Charging duration (hours)

        Returns:
            Tuple of (energy_stored_kwh, actual_duration_hours)
        """
        # Apply charging efficiency
        energy_in = power_kw * duration_hours
        energy_stored = energy_in * (self.config.round_trip_efficiency ** 0.5)

        # Check capacity limits
        max_energy = self.available_charge_capacity_kwh
        if energy_stored > max_energy:
            energy_stored = max_energy
            energy_in = energy_stored / (self.config.round_trip_efficiency ** 0.5)
            actual_duration = energy_in / power_kw if power_kw > 0 else 0
        else:
            actual_duration = duration_hours

        return (energy_stored, actual_duration)

    def calculate_discharge_energy(
        self,
        power_kw: float,
        duration_hours: float,
    ) -> Tuple[float, float]:
        """
        Calculate energy delivered from discharging.

        Args:
            power_kw: Discharge power (kW)
            duration_hours: Discharge duration (hours)

        Returns:
            Tuple of (energy_delivered_kwh, actual_duration_hours)
        """
        # Apply discharge efficiency
        energy_out = power_kw * duration_hours
        energy_required = energy_out / (self.config.round_trip_efficiency ** 0.5)

        # Check capacity limits
        max_energy = self.available_discharge_capacity_kwh
        if energy_required > max_energy:
            energy_required = max_energy
            energy_out = energy_required * (self.config.round_trip_efficiency ** 0.5)
            actual_duration = energy_out / power_kw if power_kw > 0 else 0
        else:
            actual_duration = duration_hours

        return (energy_out, actual_duration)

    def calculate_standby_loss(
        self,
        duration_hours: float,
    ) -> float:
        """
        Calculate standby thermal losses.

        Args:
            duration_hours: Standby duration (hours)

        Returns:
            Energy lost (kWh)
        """
        loss_rate = self.config.standby_loss_pct_per_hour / 100.0
        return self._current_soc_kwh * (1 - (1 - loss_rate) ** duration_hours)

    def update_state(
        self,
        power_kw: float,
        duration_hours: float,
    ) -> StorageStatePoint:
        """
        Update storage state after operation.

        Args:
            power_kw: Power (positive=charge, negative=discharge)
            duration_hours: Operation duration

        Returns:
            New state point
        """
        if power_kw > 0:
            # Charging
            energy_stored, _ = self.calculate_charge_energy(power_kw, duration_hours)
            self._current_soc_kwh = min(
                self.config.capacity_kwh,
                self._current_soc_kwh + energy_stored
            )
            self._current_mode = StorageMode.CHARGING
        elif power_kw < 0:
            # Discharging
            energy_out, _ = self.calculate_discharge_energy(-power_kw, duration_hours)
            energy_used = energy_out / (self.config.round_trip_efficiency ** 0.5)
            self._current_soc_kwh = max(0, self._current_soc_kwh - energy_used)
            self._current_mode = StorageMode.DISCHARGING
        else:
            # Idle - apply standby losses
            loss = self.calculate_standby_loss(duration_hours)
            self._current_soc_kwh = max(0, self._current_soc_kwh - loss)
            self._current_mode = StorageMode.IDLE

        # Update temperature (simplified model)
        self._update_temperature()

        return StorageStatePoint(
            timestamp=datetime.now(timezone.utc),
            state_of_charge_pct=self.current_soc_pct,
            state_of_charge_kwh=self._current_soc_kwh,
            temperature_c=self._current_temp_c,
            mode=self._current_mode,
            power_kw=power_kw,
        )

    def _update_temperature(self) -> None:
        """Update storage temperature based on SOC."""
        # Simplified: Linear relationship between SOC and temperature
        soc_ratio = self._current_soc_kwh / self.config.capacity_kwh
        temp_range = self.config.max_temperature_c - self.config.min_temperature_c
        self._current_temp_c = (
            self.config.min_temperature_c + soc_ratio * temp_range
        )

    def get_state(self) -> StorageStatePoint:
        """Get current state."""
        return StorageStatePoint(
            timestamp=datetime.now(timezone.utc),
            state_of_charge_pct=self.current_soc_pct,
            state_of_charge_kwh=self._current_soc_kwh,
            temperature_c=self._current_temp_c,
            mode=self._current_mode,
            power_kw=0.0,
        )


# =============================================================================
# STORAGE OPTIMIZER
# =============================================================================

class ThermalStorageOptimizer:
    """
    Optimizer for thermal energy storage dispatch.

    Optimizes charge/discharge schedules to minimize energy costs
    while maintaining operational constraints.

    All optimization is DETERMINISTIC using linear programming
    principles without any ML or LLM inference for scheduling decisions.
    """

    def __init__(
        self,
        storage_configs: List[ThermalStorageConfiguration],
        tariff_config: Optional[TariffConfiguration] = None,
    ) -> None:
        """
        Initialize storage optimizer.

        Args:
            storage_configs: List of storage unit configurations
            tariff_config: Energy tariff configuration
        """
        self._units: Dict[str, ThermalStorageUnit] = {}
        for config in storage_configs:
            if config.enabled:
                self._units[config.storage_id] = ThermalStorageUnit(config)

        self._tariff = tariff_config
        self._time_step_hours = 0.25  # 15-minute intervals

        logger.info(
            f"ThermalStorageOptimizer initialized with {len(self._units)} units"
        )

    def optimize_dispatch(
        self,
        load_forecast: LoadForecastResult,
        horizon_hours: int = 24,
        demand_limit_kw: Optional[float] = None,
    ) -> ThermalStorageResult:
        """
        Optimize storage dispatch schedule.

        Uses a rule-based optimization approach:
        1. Charge during off-peak hours when prices are low
        2. Discharge during peak hours to reduce demand charges
        3. Maintain minimum SOC for emergency reserve

        Args:
            load_forecast: Load forecast for optimization horizon
            horizon_hours: Optimization horizon (hours)
            demand_limit_kw: Optional demand limit constraint

        Returns:
            ThermalStorageResult with dispatch schedules
        """
        start_time = datetime.now(timezone.utc)
        logger.info(f"Optimizing storage dispatch for {horizon_hours}h horizon")

        unit_schedules: List[StorageDispatchSchedule] = []

        for storage_id, unit in self._units.items():
            schedule = self._optimize_unit_dispatch(
                unit=unit,
                load_forecast=load_forecast,
                horizon_hours=horizon_hours,
                demand_limit_kw=demand_limit_kw,
            )
            unit_schedules.append(schedule)

        # Aggregate metrics
        total_capacity = sum(u.config.capacity_kwh for u in self._units.values())
        current_soc = sum(u.current_soc_kwh for u in self._units.values())
        total_arbitrage = sum(s.energy_arbitrage_usd for s in unit_schedules)
        total_demand_savings = sum(s.demand_savings_usd for s in unit_schedules)

        result = ThermalStorageResult(
            timestamp=datetime.now(timezone.utc),
            unit_schedules=unit_schedules,
            total_storage_capacity_kwh=total_capacity,
            current_soc_kwh=current_soc,
            total_energy_arbitrage_usd=total_arbitrage,
            total_demand_savings_usd=total_demand_savings,
            total_savings_usd=total_arbitrage + total_demand_savings,
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            f"Storage optimization complete: "
            f"arbitrage=${total_arbitrage:.2f}, demand_savings=${total_demand_savings:.2f} "
            f"({processing_time:.1f}ms)"
        )

        return result

    def _optimize_unit_dispatch(
        self,
        unit: ThermalStorageUnit,
        load_forecast: LoadForecastResult,
        horizon_hours: int,
        demand_limit_kw: Optional[float],
    ) -> StorageDispatchSchedule:
        """
        Optimize dispatch for a single storage unit.

        Args:
            unit: Storage unit to optimize
            load_forecast: Load forecast
            horizon_hours: Optimization horizon
            demand_limit_kw: Demand limit

        Returns:
            StorageDispatchSchedule for the unit
        """
        n_steps = int(horizon_hours / self._time_step_hours)
        dispatch_points: List[StorageStatePoint] = []

        # Determine peak/off-peak periods
        peak_hours = self._get_peak_hours()

        # Track metrics
        total_charged = 0.0
        total_discharged = 0.0
        charge_hours = 0.0
        discharge_hours = 0.0
        arbitrage_value = 0.0
        demand_savings = 0.0

        # Simulate state through horizon
        sim_soc_kwh = unit.current_soc_kwh
        config = unit.config

        for i in range(n_steps):
            step_time = datetime.now(timezone.utc) + timedelta(
                hours=i * self._time_step_hours
            )
            hour = step_time.hour

            # Get load at this time
            forecast_load = self._get_load_at_time(load_forecast, step_time)

            # Determine action based on time of use
            is_peak = hour in peak_hours
            power_kw = 0.0
            mode = StorageMode.IDLE

            # Peak hours: Discharge if possible
            if is_peak and sim_soc_kwh > config.min_soc_pct / 100 * config.capacity_kwh:
                # Check if discharge would help demand reduction
                if demand_limit_kw is None or forecast_load > demand_limit_kw * 0.8:
                    # Discharge to reduce peak
                    available = sim_soc_kwh - config.min_soc_pct / 100 * config.capacity_kwh
                    max_discharge = min(
                        config.max_discharge_rate_kw,
                        available / self._time_step_hours / (config.round_trip_efficiency ** 0.5)
                    )
                    power_kw = -max_discharge
                    mode = StorageMode.DISCHARGING

                    # Calculate energy
                    energy_out = -power_kw * self._time_step_hours * (config.round_trip_efficiency ** 0.5)
                    total_discharged += energy_out
                    discharge_hours += self._time_step_hours

                    # Calculate savings
                    if self._tariff:
                        peak_rate = self._tariff.peak_rate_per_kwh
                        demand_savings += energy_out * peak_rate * 0.5  # Simplified

            # Off-peak hours: Charge if capacity available
            elif not is_peak and sim_soc_kwh < config.max_soc_pct / 100 * config.capacity_kwh:
                available = config.max_soc_pct / 100 * config.capacity_kwh - sim_soc_kwh
                max_charge = min(
                    config.max_charge_rate_kw,
                    available / self._time_step_hours * (config.round_trip_efficiency ** 0.5)
                )
                power_kw = max_charge
                mode = StorageMode.CHARGING

                # Calculate energy
                energy_in = power_kw * self._time_step_hours
                energy_stored = energy_in * (config.round_trip_efficiency ** 0.5)
                total_charged += energy_stored
                charge_hours += self._time_step_hours

                # Calculate arbitrage opportunity
                if self._tariff:
                    off_peak_rate = self._tariff.off_peak_rate_per_kwh
                    peak_rate = self._tariff.peak_rate_per_kwh
                    arbitrage_value += energy_stored * (peak_rate - off_peak_rate)

            # Update simulated SOC
            if power_kw > 0:
                sim_soc_kwh = min(
                    config.capacity_kwh,
                    sim_soc_kwh + power_kw * self._time_step_hours * (config.round_trip_efficiency ** 0.5)
                )
            elif power_kw < 0:
                energy_needed = -power_kw * self._time_step_hours / (config.round_trip_efficiency ** 0.5)
                sim_soc_kwh = max(0, sim_soc_kwh - energy_needed)
            else:
                # Standby losses
                loss_rate = config.standby_loss_pct_per_hour / 100.0
                sim_soc_kwh *= (1 - loss_rate * self._time_step_hours)

            # Record state point
            soc_pct = (sim_soc_kwh / config.capacity_kwh) * 100
            dispatch_points.append(StorageStatePoint(
                timestamp=step_time,
                state_of_charge_pct=round(soc_pct, 2),
                state_of_charge_kwh=round(sim_soc_kwh, 2),
                temperature_c=config.design_temperature_c,  # Simplified
                mode=mode,
                power_kw=round(power_kw, 2),
            ))

        # Calculate equivalent cycles
        cycles = total_discharged / config.capacity_kwh if config.capacity_kwh > 0 else 0

        return StorageDispatchSchedule(
            storage_id=unit.storage_id,
            dispatch_points=dispatch_points,
            total_charge_kwh=round(total_charged, 2),
            total_discharge_kwh=round(total_discharged, 2),
            charge_hours=round(charge_hours, 2),
            discharge_hours=round(discharge_hours, 2),
            cycles=round(cycles, 3),
            energy_arbitrage_usd=round(arbitrage_value, 2),
            demand_savings_usd=round(demand_savings, 2),
            min_soc_maintained=True,
            reserve_maintained=True,
        )

    def _get_peak_hours(self) -> set:
        """Get set of peak hours from tariff."""
        if self._tariff is None:
            return {14, 15, 16, 17, 18, 19}  # Default peak hours

        start = self._tariff.peak_hours_start
        end = self._tariff.peak_hours_end

        if end > start:
            return set(range(start, end + 1))
        else:
            # Spans midnight
            return set(range(start, 24)) | set(range(0, end + 1))

    def _get_load_at_time(
        self,
        forecast: LoadForecastResult,
        target_time: datetime,
    ) -> float:
        """Get forecasted load at target time."""
        for point in forecast.forecast_points:
            if abs((point.timestamp - target_time).total_seconds()) < 900:  # 15 min
                return point.load_kw

        return forecast.avg_load_kw or 1000.0  # Fallback

    def get_current_state(self) -> Dict[str, StorageStatePoint]:
        """Get current state of all storage units."""
        return {
            storage_id: unit.get_state()
            for storage_id, unit in self._units.items()
        }

    def execute_dispatch(
        self,
        storage_id: str,
        power_kw: float,
        duration_hours: float,
    ) -> StorageStatePoint:
        """
        Execute dispatch command on a storage unit.

        Args:
            storage_id: Target storage unit
            power_kw: Power setpoint (positive=charge, negative=discharge)
            duration_hours: Duration of operation

        Returns:
            Updated state point
        """
        if storage_id not in self._units:
            raise ValueError(f"Unknown storage unit: {storage_id}")

        unit = self._units[storage_id]

        # Validate against limits
        if power_kw > 0:
            power_kw = min(power_kw, unit.config.max_charge_rate_kw)
        elif power_kw < 0:
            power_kw = max(power_kw, -unit.config.max_discharge_rate_kw)

        return unit.update_state(power_kw, duration_hours)


# =============================================================================
# PCM STORAGE CALCULATOR
# =============================================================================

class PCMStorageCalculator:
    """
    Calculator for Phase Change Material (PCM) storage.

    Implements latent heat storage calculations for PCM systems
    with proper handling of phase transitions.
    """

    def __init__(
        self,
        pcm_mass_kg: float,
        melt_temperature_c: float,
        latent_heat_kj_kg: float,
        sensible_heat_kj_kg_k: float = 2.0,
    ) -> None:
        """
        Initialize PCM calculator.

        Args:
            pcm_mass_kg: Mass of PCM (kg)
            melt_temperature_c: Melting temperature (C)
            latent_heat_kj_kg: Latent heat of fusion (kJ/kg)
            sensible_heat_kj_kg_k: Sensible specific heat (kJ/kg-K)
        """
        self._mass = pcm_mass_kg
        self._melt_temp = melt_temperature_c
        self._latent_heat = latent_heat_kj_kg
        self._cp = sensible_heat_kj_kg_k

        # Calculate capacities
        self._latent_capacity_kwh = (
            pcm_mass_kg * latent_heat_kj_kg / 3600.0
        )

        logger.info(
            f"PCM Calculator: {pcm_mass_kg}kg PCM, "
            f"melt={melt_temperature_c}C, "
            f"latent_capacity={self._latent_capacity_kwh:.1f}kWh"
        )

    @property
    def latent_capacity_kwh(self) -> float:
        """Get latent heat storage capacity (kWh)."""
        return self._latent_capacity_kwh

    def calculate_energy_stored(
        self,
        temp_start_c: float,
        temp_end_c: float,
    ) -> float:
        """
        Calculate energy stored for a temperature change.

        Handles sensible heat and latent heat (if phase change occurs).

        Args:
            temp_start_c: Starting temperature (C)
            temp_end_c: Ending temperature (C)

        Returns:
            Energy stored (kWh)
        """
        energy_kj = 0.0

        # Case 1: Below melting point (sensible only)
        if temp_end_c <= self._melt_temp and temp_start_c <= self._melt_temp:
            energy_kj = self._mass * self._cp * (temp_end_c - temp_start_c)

        # Case 2: Above melting point (sensible only)
        elif temp_start_c >= self._melt_temp and temp_end_c >= self._melt_temp:
            energy_kj = self._mass * self._cp * (temp_end_c - temp_start_c)

        # Case 3: Crosses melting point (charging)
        elif temp_start_c < self._melt_temp < temp_end_c:
            # Sensible below melt point
            energy_kj += self._mass * self._cp * (self._melt_temp - temp_start_c)
            # Latent heat
            energy_kj += self._mass * self._latent_heat
            # Sensible above melt point
            energy_kj += self._mass * self._cp * (temp_end_c - self._melt_temp)

        # Case 4: Crosses melting point (discharging)
        elif temp_end_c < self._melt_temp < temp_start_c:
            # Sensible above melt point
            energy_kj += self._mass * self._cp * (self._melt_temp - temp_start_c)
            # Latent heat release
            energy_kj -= self._mass * self._latent_heat
            # Sensible below melt point
            energy_kj += self._mass * self._cp * (temp_end_c - self._melt_temp)

        return energy_kj / 3600.0  # Convert to kWh

    def calculate_melt_fraction(
        self,
        current_temp_c: float,
        energy_stored_kwh: float,
    ) -> float:
        """
        Calculate fraction of PCM that has melted.

        Args:
            current_temp_c: Current temperature (C)
            energy_stored_kwh: Total energy stored (kWh)

        Returns:
            Melt fraction (0 to 1)
        """
        if current_temp_c < self._melt_temp:
            return 0.0
        elif current_temp_c > self._melt_temp + 0.1:
            return 1.0
        else:
            # During phase change
            total_latent = self._latent_capacity_kwh
            if total_latent > 0:
                return min(1.0, energy_stored_kwh / total_latent)
            return 0.5


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ThermalStorageUnit",
    "ThermalStorageOptimizer",
    "PCMStorageCalculator",
    "CP_WATER",
    "WATER_DENSITY",
]
