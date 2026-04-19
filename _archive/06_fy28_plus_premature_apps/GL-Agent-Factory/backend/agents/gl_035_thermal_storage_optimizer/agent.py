"""
GL-035: Thermal Storage Optimizer Agent (THERMAL-STORAGE-OPTIMIZER)

Optimizes thermal energy storage systems for peak shaving and cost reduction.

Standards: ASHRAE, IEA-ECES (Energy Conservation through Energy Storage)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DemandProfile(BaseModel):
    """Hourly demand profile data."""
    hour: int = Field(..., ge=0, le=23)
    demand_kw: float = Field(..., ge=0)


class EnergyPrice(BaseModel):
    """Hourly energy price data."""
    hour: int = Field(..., ge=0, le=23)
    price_per_kwh: float = Field(..., ge=0)


class ThermalStorageOptimizerInput(BaseModel):
    """Input for ThermalStorageOptimizerAgent."""
    system_id: str = Field(..., description="Storage system identifier")
    storage_capacity_kwh: float = Field(..., gt=0, description="Total storage capacity")
    max_charge_rate_kw: float = Field(..., gt=0, description="Maximum charge rate")
    max_discharge_rate_kw: float = Field(..., gt=0, description="Maximum discharge rate")
    round_trip_efficiency: float = Field(default=0.90, gt=0, le=1)
    current_state_of_charge: float = Field(default=0.5, ge=0, le=1)
    demand_profile: List[DemandProfile] = Field(...)
    energy_prices: List[EnergyPrice] = Field(...)
    demand_charge_per_kw: float = Field(default=15.0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HourlySchedule(BaseModel):
    """Hourly charge/discharge schedule."""
    hour: int
    action: str  # CHARGE, DISCHARGE, IDLE
    power_kw: float
    state_of_charge: float
    grid_demand_kw: float
    cost: float


class ThermalStorageOptimizerOutput(BaseModel):
    """Output from ThermalStorageOptimizerAgent."""
    analysis_id: str
    system_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    optimal_schedule: List[HourlySchedule]
    total_cost_without_storage: float
    total_cost_with_storage: float
    cost_savings: float
    cost_savings_percent: float
    peak_demand_without_storage_kw: float
    peak_demand_with_storage_kw: float
    peak_shaving_kw: float
    storage_utilization_percent: float
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class ThermalStorageOptimizerAgent:
    """GL-035: Thermal Storage Optimizer Agent."""

    AGENT_ID = "GL-035"
    AGENT_NAME = "THERMAL-STORAGE-OPTIMIZER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"ThermalStorageOptimizerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: ThermalStorageOptimizerInput) -> ThermalStorageOptimizerOutput:
        """Execute thermal storage optimization."""
        start_time = datetime.utcnow()
        logger.info(f"Starting storage optimization for {input_data.system_id}")

        # Build demand and price arrays
        demand_by_hour = {d.hour: d.demand_kw for d in input_data.demand_profile}
        price_by_hour = {p.hour: p.price_per_kwh for p in input_data.energy_prices}

        # Simple optimization: charge during low prices, discharge during high prices
        avg_price = sum(price_by_hour.values()) / max(len(price_by_hour), 1)
        charge_hours = [h for h, p in price_by_hour.items() if p < avg_price * 0.8]
        discharge_hours = [h for h, p in price_by_hour.items() if p > avg_price * 1.2]

        schedule = []
        soc = input_data.current_state_of_charge
        total_charged = 0
        total_discharged = 0

        for hour in range(24):
            demand = demand_by_hour.get(hour, 0)
            price = price_by_hour.get(hour, avg_price)

            if hour in charge_hours and soc < 0.95:
                # Charge
                charge_kw = min(
                    input_data.max_charge_rate_kw,
                    (0.95 - soc) * input_data.storage_capacity_kwh
                )
                soc += (charge_kw * input_data.round_trip_efficiency) / input_data.storage_capacity_kwh
                total_charged += charge_kw
                grid_demand = demand + charge_kw
                action = "CHARGE"
                power = charge_kw
            elif hour in discharge_hours and soc > 0.1:
                # Discharge
                discharge_kw = min(
                    input_data.max_discharge_rate_kw,
                    (soc - 0.1) * input_data.storage_capacity_kwh,
                    demand
                )
                soc -= discharge_kw / input_data.storage_capacity_kwh
                total_discharged += discharge_kw
                grid_demand = max(0, demand - discharge_kw)
                action = "DISCHARGE"
                power = discharge_kw
            else:
                action = "IDLE"
                power = 0
                grid_demand = demand

            cost = grid_demand * price

            schedule.append(HourlySchedule(
                hour=hour,
                action=action,
                power_kw=round(power, 1),
                state_of_charge=round(soc, 3),
                grid_demand_kw=round(grid_demand, 1),
                cost=round(cost, 2)
            ))

        # Calculate costs
        demands_without = [demand_by_hour.get(h, 0) for h in range(24)]
        demands_with = [s.grid_demand_kw for s in schedule]

        peak_without = max(demands_without)
        peak_with = max(demands_with)

        energy_cost_without = sum(
            demand_by_hour.get(h, 0) * price_by_hour.get(h, avg_price)
            for h in range(24)
        )
        energy_cost_with = sum(s.cost for s in schedule)

        demand_cost_without = peak_without * input_data.demand_charge_per_kw
        demand_cost_with = peak_with * input_data.demand_charge_per_kw

        total_without = energy_cost_without + demand_cost_without
        total_with = energy_cost_with + demand_cost_with

        savings = total_without - total_with
        savings_pct = (savings / total_without * 100) if total_without > 0 else 0

        utilization = (total_discharged / input_data.storage_capacity_kwh * 100) if input_data.storage_capacity_kwh > 0 else 0

        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "system": input_data.system_id,
                        "timestamp": datetime.utcnow().isoformat()}, sort_keys=True).encode()
        ).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ThermalStorageOptimizerOutput(
            analysis_id=f"TSO-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            system_id=input_data.system_id,
            optimal_schedule=schedule,
            total_cost_without_storage=round(total_without, 2),
            total_cost_with_storage=round(total_with, 2),
            cost_savings=round(savings, 2),
            cost_savings_percent=round(savings_pct, 1),
            peak_demand_without_storage_kw=round(peak_without, 1),
            peak_demand_with_storage_kw=round(peak_with, 1),
            peak_shaving_kw=round(peak_without - peak_with, 1),
            storage_utilization_percent=round(utilization, 1),
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS"
        )


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-035",
    "name": "THERMAL-STORAGE-OPTIMIZER",
    "version": "1.0.0",
    "summary": "Thermal storage optimization for peak shaving and cost reduction",
    "tags": ["thermal-storage", "peak-shaving", "energy-optimization", "ASHRAE"],
    "owners": ["process-heat-optimization-team"],
    "standards": [
        {"ref": "ASHRAE", "description": "ASHRAE Standards"},
        {"ref": "IEA-ECES", "description": "Energy Conservation through Energy Storage"}
    ]
}
