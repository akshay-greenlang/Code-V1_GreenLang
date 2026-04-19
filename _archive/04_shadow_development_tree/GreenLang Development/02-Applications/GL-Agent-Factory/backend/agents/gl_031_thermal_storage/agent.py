"""GL-031 THERMALSTORAGE: Thermal Energy Storage Agent.

Manages thermal energy storage for load shifting, grid support, and
process integration optimization.

Standards: IEC 62933, ASHRAE
"""
import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ThermalStorageInput(BaseModel):
    """Input for thermal storage optimization."""

    equipment_id: str = Field(..., description="Storage system ID")
    storage_type: str = Field("hot_water", description="hot_water, molten_salt, ice, pcm")

    # Storage state
    capacity_mwh: float = Field(..., ge=0, description="Total capacity (MWh)")
    current_soc_pct: float = Field(..., ge=0, le=100, description="State of charge (%)")
    min_soc_pct: float = Field(10, description="Minimum SOC (%)")
    max_soc_pct: float = Field(95, description="Maximum SOC (%)")

    # Thermal parameters
    storage_temp_c: float = Field(..., description="Current storage temperature (°C)")
    min_temp_c: float = Field(..., description="Minimum useful temperature (°C)")
    max_temp_c: float = Field(..., description="Maximum temperature (°C)")
    ambient_temp_c: float = Field(20, description="Ambient temperature (°C)")
    heat_loss_pct_per_hour: float = Field(0.5, description="Standby heat loss (%/hr)")

    # Charge/discharge rates
    max_charge_rate_mw: float = Field(..., ge=0, description="Max charge rate (MW)")
    max_discharge_rate_mw: float = Field(..., ge=0, description="Max discharge rate (MW)")
    charge_efficiency_pct: float = Field(95, description="Charging efficiency (%)")
    discharge_efficiency_pct: float = Field(95, description="Discharge efficiency (%)")

    # Demand and supply
    heat_demand_mw: float = Field(..., ge=0, description="Current heat demand (MW)")
    available_heat_supply_mw: float = Field(..., ge=0, description="Available supply (MW)")
    demand_forecast_24h: List[float] = Field(default_factory=list, description="24-hr demand forecast")

    # Economics
    energy_price_current: float = Field(0.05, description="Current energy price ($/kWh)")
    energy_price_forecast_24h: List[float] = Field(default_factory=list, description="24-hr price forecast")
    demand_charge_per_kw: float = Field(10, description="Demand charge ($/kW)")

    # Grid services
    grid_services_enabled: bool = Field(False, description="Participating in grid services")
    demand_response_price: float = Field(0.20, description="DR payment ($/kWh)")

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StorageDispatch(BaseModel):
    """Storage dispatch recommendation."""
    action: str  # CHARGE, DISCHARGE, IDLE
    rate_mw: float
    duration_hours: float
    target_soc_pct: float
    reason: str


class ThermalStorageOutput(BaseModel):
    """Output from thermal storage agent."""

    # Current status
    available_energy_mwh: float = Field(..., description="Available discharge energy")
    available_capacity_mwh: float = Field(..., description="Available charge capacity")
    estimated_standby_loss_mwh_day: float = Field(..., description="Daily standby loss")

    # Dispatch recommendation
    dispatch: StorageDispatch
    optimal_schedule_24h: List[Dict[str, Any]] = Field(default_factory=list)

    # Economic value
    energy_arbitrage_value_day: float = Field(..., description="Daily arbitrage value ($)")
    demand_charge_savings_day: float = Field(..., description="Daily demand savings ($)")
    grid_services_revenue_day: float = Field(0, description="DR revenue ($)")
    total_daily_value: float = Field(..., description="Total daily value ($)")

    # Efficiency metrics
    round_trip_efficiency_pct: float = Field(..., description="Round-trip efficiency")
    capacity_utilization_pct: float = Field(..., description="Capacity utilization")

    # Health monitoring
    cycles_today: int = Field(0, description="Charge/discharge cycles today")
    estimated_degradation_pct_year: float = Field(..., description="Annual degradation estimate")

    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    calculation_hash: str
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


def calculate_available_energy(
    capacity_mwh: float,
    soc_pct: float,
    min_soc_pct: float,
    discharge_efficiency: float
) -> float:
    """Calculate available discharge energy."""
    usable_soc = max(0, soc_pct - min_soc_pct) / 100
    return round(capacity_mwh * usable_soc * discharge_efficiency / 100, 2)


def calculate_available_capacity(
    capacity_mwh: float,
    soc_pct: float,
    max_soc_pct: float,
    charge_efficiency: float
) -> float:
    """Calculate available charge capacity."""
    headroom_soc = max(0, max_soc_pct - soc_pct) / 100
    # Need more input to reach storage target due to efficiency losses
    return round(capacity_mwh * headroom_soc / (charge_efficiency / 100), 2)


def optimize_dispatch(
    soc_pct: float,
    heat_demand_mw: float,
    heat_supply_mw: float,
    capacity_mwh: float,
    max_charge_mw: float,
    max_discharge_mw: float,
    min_soc: float,
    max_soc: float,
    energy_price: float,
    price_forecast: List[float]
) -> StorageDispatch:
    """Determine optimal storage dispatch."""
    # Simple dispatch logic
    supply_surplus = heat_supply_mw - heat_demand_mw

    # If we have excess supply and capacity, charge
    if supply_surplus > 0 and soc_pct < max_soc:
        charge_rate = min(supply_surplus, max_charge_mw)
        return StorageDispatch(
            action="CHARGE",
            rate_mw=round(charge_rate, 2),
            duration_hours=1.0,
            target_soc_pct=min(soc_pct + 10, max_soc),
            reason="Storing excess heat supply"
        )

    # If demand exceeds supply and we have stored energy, discharge
    supply_deficit = heat_demand_mw - heat_supply_mw
    if supply_deficit > 0 and soc_pct > min_soc:
        discharge_rate = min(supply_deficit, max_discharge_mw)
        return StorageDispatch(
            action="DISCHARGE",
            rate_mw=round(discharge_rate, 2),
            duration_hours=1.0,
            target_soc_pct=max(soc_pct - 10, min_soc),
            reason="Meeting demand deficit"
        )

    # Price-based arbitrage if we have forecast
    if price_forecast and len(price_forecast) >= 4:
        future_avg = sum(price_forecast[:4]) / 4
        if energy_price < future_avg * 0.8 and soc_pct < max_soc - 10:
            return StorageDispatch(
                action="CHARGE",
                rate_mw=round(max_charge_mw * 0.5, 2),
                duration_hours=2.0,
                target_soc_pct=min(soc_pct + 20, max_soc),
                reason="Charging during low-price period"
            )
        elif energy_price > future_avg * 1.2 and soc_pct > min_soc + 10:
            return StorageDispatch(
                action="DISCHARGE",
                rate_mw=round(max_discharge_mw * 0.5, 2),
                duration_hours=2.0,
                target_soc_pct=max(soc_pct - 20, min_soc),
                reason="Discharging during high-price period"
            )

    return StorageDispatch(
        action="IDLE",
        rate_mw=0,
        duration_hours=1.0,
        target_soc_pct=soc_pct,
        reason="No economic dispatch opportunity"
    )


class ThermalEnergyStorageAgent:
    """Thermal energy storage optimization agent."""

    AGENT_ID = "GL-031"
    AGENT_NAME = "THERMALSTORAGE"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = ThermalStorageInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: ThermalStorageInput) -> ThermalStorageOutput:
        recommendations = []
        warnings = []

        # Calculate available energy and capacity
        available_energy = calculate_available_energy(
            inp.capacity_mwh, inp.current_soc_pct,
            inp.min_soc_pct, inp.discharge_efficiency_pct
        )

        available_capacity = calculate_available_capacity(
            inp.capacity_mwh, inp.current_soc_pct,
            inp.max_soc_pct, inp.charge_efficiency_pct
        )

        # Standby losses
        daily_loss = inp.capacity_mwh * (inp.current_soc_pct / 100) * (inp.heat_loss_pct_per_hour / 100) * 24

        # Get dispatch recommendation
        dispatch = optimize_dispatch(
            inp.current_soc_pct,
            inp.heat_demand_mw,
            inp.available_heat_supply_mw,
            inp.capacity_mwh,
            inp.max_charge_rate_mw,
            inp.max_discharge_rate_mw,
            inp.min_soc_pct,
            inp.max_soc_pct,
            inp.energy_price_current,
            inp.energy_price_forecast_24h
        )

        # Economic calculations
        rt_efficiency = (inp.charge_efficiency_pct / 100) * (inp.discharge_efficiency_pct / 100) * 100

        # Estimate arbitrage value
        if inp.energy_price_forecast_24h:
            price_spread = max(inp.energy_price_forecast_24h) - min(inp.energy_price_forecast_24h)
            # One cycle per day at max rates
            arbitrage_mwh = min(inp.max_charge_rate_mw, inp.max_discharge_rate_mw) * 4  # 4 hours
            arbitrage_value = arbitrage_mwh * price_spread * (rt_efficiency / 100) * 1000  # Convert to $
        else:
            arbitrage_value = 0

        # Demand charge savings
        if inp.demand_forecast_24h:
            peak_demand = max(inp.demand_forecast_24h)
            shaved_peak = min(inp.max_discharge_rate_mw, peak_demand * 0.2)
            demand_savings = shaved_peak * inp.demand_charge_per_kw * 1000
        else:
            demand_savings = 0

        # Grid services revenue
        grid_revenue = 0
        if inp.grid_services_enabled:
            grid_revenue = available_energy * inp.demand_response_price * 1000

        total_value = arbitrage_value + demand_savings + grid_revenue

        # Capacity utilization
        utilization = (available_energy / inp.capacity_mwh * 100) if inp.capacity_mwh > 0 else 0

        # Degradation estimate (simplified)
        degradation = 2.0  # % per year typical

        # Warnings
        if inp.current_soc_pct < 20:
            warnings.append(f"Low SOC ({inp.current_soc_pct:.0f}%) - limited discharge capacity")

        if inp.current_soc_pct > 90:
            warnings.append(f"High SOC ({inp.current_soc_pct:.0f}%) - limited charge headroom")

        if daily_loss > inp.capacity_mwh * 0.05:
            warnings.append(f"High standby losses ({daily_loss:.1f} MWh/day) - check insulation")

        # Recommendations
        if rt_efficiency < 80:
            recommendations.append(f"Round-trip efficiency {rt_efficiency:.1f}% is low - investigate losses")

        if arbitrage_value > 100:
            recommendations.append(f"Price arbitrage opportunity: ${arbitrage_value:.0f}/day potential")

        if demand_savings > 50:
            recommendations.append(f"Peak shaving can save ${demand_savings:.0f}/day in demand charges")

        # Provenance
        calc_hash = hashlib.sha256(json.dumps({
            "soc": inp.current_soc_pct,
            "dispatch": dispatch.action,
            "value": total_value
        }).encode()).hexdigest()

        return ThermalStorageOutput(
            available_energy_mwh=available_energy,
            available_capacity_mwh=available_capacity,
            estimated_standby_loss_mwh_day=round(daily_loss, 2),
            dispatch=dispatch,
            optimal_schedule_24h=[],  # Would be populated by full optimization
            energy_arbitrage_value_day=round(arbitrage_value, 2),
            demand_charge_savings_day=round(demand_savings, 2),
            grid_services_revenue_day=round(grid_revenue, 2),
            total_daily_value=round(total_value, 2),
            round_trip_efficiency_pct=round(rt_efficiency, 1),
            capacity_utilization_pct=round(utilization, 1),
            cycles_today=0,
            estimated_degradation_pct_year=degradation,
            recommendations=recommendations,
            warnings=warnings,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "category": "Energy Storage",
            "type": "Controller",
            "complexity": "High"
        }
