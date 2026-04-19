"""
GL-036: CHP Coordinator Agent (CHP-COORDINATOR)

Combined Heat and Power optimization for economic dispatch and efficiency.

Standards: EPA CHP, ISO 50001
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CHPCoordinatorInput(BaseModel):
    """Input for CHPCoordinatorAgent."""
    system_id: str = Field(...)
    electrical_demand_kw: float = Field(..., ge=0)
    thermal_demand_kw: float = Field(..., ge=0)
    fuel_price_per_mmbtu: float = Field(default=5.0, ge=0)
    grid_price_per_kwh: float = Field(default=0.10, ge=0)
    grid_sellback_per_kwh: float = Field(default=0.05, ge=0)
    chp_electrical_capacity_kw: float = Field(default=1000, gt=0)
    chp_thermal_capacity_kw: float = Field(default=1500, gt=0)
    chp_electrical_efficiency: float = Field(default=0.35, gt=0, le=1)
    chp_thermal_efficiency: float = Field(default=0.45, gt=0, le=1)
    boiler_efficiency: float = Field(default=0.85, gt=0, le=1)
    equipment_available: bool = Field(default=True)
    co2_factor_grid_kg_per_kwh: float = Field(default=0.4)
    co2_factor_gas_kg_per_mmbtu: float = Field(default=53.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CHPCoordinatorOutput(BaseModel):
    """Output from CHPCoordinatorAgent."""
    analysis_id: str
    system_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    chp_electrical_output_kw: float
    chp_thermal_output_kw: float
    grid_import_kw: float
    grid_export_kw: float
    boiler_output_kw: float
    power_to_heat_ratio: float
    total_efficiency_percent: float
    operating_cost_per_hour: float
    baseline_cost_per_hour: float
    cost_savings_per_hour: float
    co2_emissions_kg_per_hour: float
    baseline_co2_kg_per_hour: float
    co2_reduction_percent: float
    dispatch_recommendation: str
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class CHPCoordinatorAgent:
    """GL-036: CHP Coordinator Agent."""

    AGENT_ID = "GL-036"
    AGENT_NAME = "CHP-COORDINATOR"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"CHPCoordinatorAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: CHPCoordinatorInput) -> CHPCoordinatorOutput:
        """Execute CHP optimization."""
        start_time = datetime.utcnow()
        logger.info(f"Starting CHP optimization for {input_data.system_id}")

        # Baseline: all electricity from grid, all heat from boiler
        baseline_elec_cost = input_data.electrical_demand_kw * input_data.grid_price_per_kwh
        baseline_heat_mmbtu = input_data.thermal_demand_kw * 0.003412 / input_data.boiler_efficiency
        baseline_heat_cost = baseline_heat_mmbtu * input_data.fuel_price_per_mmbtu
        baseline_cost = baseline_elec_cost + baseline_heat_cost

        baseline_co2_elec = input_data.electrical_demand_kw * input_data.co2_factor_grid_kg_per_kwh
        baseline_co2_heat = baseline_heat_mmbtu * input_data.co2_factor_gas_kg_per_mmbtu
        baseline_co2 = baseline_co2_elec + baseline_co2_heat

        if not input_data.equipment_available:
            # CHP not available - use baseline
            return self._create_baseline_output(input_data, start_time, baseline_cost, baseline_co2)

        # CHP dispatch optimization
        # Strategy: Run CHP to meet thermal demand, use grid for excess electrical
        total_efficiency = input_data.chp_electrical_efficiency + input_data.chp_thermal_efficiency

        # Thermal-led dispatch
        chp_thermal = min(input_data.thermal_demand_kw, input_data.chp_thermal_capacity_kw)
        thermal_fraction = chp_thermal / input_data.chp_thermal_capacity_kw if input_data.chp_thermal_capacity_kw > 0 else 0
        chp_electrical = thermal_fraction * input_data.chp_electrical_capacity_kw

        # Grid balance
        if chp_electrical >= input_data.electrical_demand_kw:
            grid_import = 0
            grid_export = chp_electrical - input_data.electrical_demand_kw
        else:
            grid_import = input_data.electrical_demand_kw - chp_electrical
            grid_export = 0

        # Supplemental boiler
        boiler_output = max(0, input_data.thermal_demand_kw - chp_thermal)

        # CHP fuel consumption
        chp_input_kw = (chp_electrical / input_data.chp_electrical_efficiency) if input_data.chp_electrical_efficiency > 0 else 0
        chp_fuel_mmbtu = chp_input_kw * 0.003412

        # Boiler fuel consumption
        boiler_fuel_mmbtu = (boiler_output * 0.003412 / input_data.boiler_efficiency) if input_data.boiler_efficiency > 0 else 0

        # Costs
        chp_fuel_cost = chp_fuel_mmbtu * input_data.fuel_price_per_mmbtu
        boiler_cost = boiler_fuel_mmbtu * input_data.fuel_price_per_mmbtu
        grid_cost = grid_import * input_data.grid_price_per_kwh
        grid_revenue = grid_export * input_data.grid_sellback_per_kwh

        total_cost = chp_fuel_cost + boiler_cost + grid_cost - grid_revenue

        # CO2 emissions
        chp_co2 = chp_fuel_mmbtu * input_data.co2_factor_gas_kg_per_mmbtu
        boiler_co2 = boiler_fuel_mmbtu * input_data.co2_factor_gas_kg_per_mmbtu
        grid_co2 = grid_import * input_data.co2_factor_grid_kg_per_kwh
        # Credit for exported power
        export_co2_credit = grid_export * input_data.co2_factor_grid_kg_per_kwh
        total_co2 = chp_co2 + boiler_co2 + grid_co2 - export_co2_credit

        # Power to heat ratio
        phr = chp_electrical / chp_thermal if chp_thermal > 0 else 0

        # Savings
        cost_savings = baseline_cost - total_cost
        co2_reduction = ((baseline_co2 - total_co2) / baseline_co2 * 100) if baseline_co2 > 0 else 0

        # Dispatch recommendation
        if chp_electrical > 0:
            recommendation = f"Run CHP at {thermal_fraction*100:.0f}% load (thermal-led)"
        else:
            recommendation = "CHP offline - use grid and boiler"

        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "system": input_data.system_id,
                        "timestamp": datetime.utcnow().isoformat()}, sort_keys=True).encode()
        ).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return CHPCoordinatorOutput(
            analysis_id=f"CHP-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            system_id=input_data.system_id,
            chp_electrical_output_kw=round(chp_electrical, 1),
            chp_thermal_output_kw=round(chp_thermal, 1),
            grid_import_kw=round(grid_import, 1),
            grid_export_kw=round(grid_export, 1),
            boiler_output_kw=round(boiler_output, 1),
            power_to_heat_ratio=round(phr, 2),
            total_efficiency_percent=round(total_efficiency * 100, 1),
            operating_cost_per_hour=round(total_cost, 2),
            baseline_cost_per_hour=round(baseline_cost, 2),
            cost_savings_per_hour=round(cost_savings, 2),
            co2_emissions_kg_per_hour=round(total_co2, 1),
            baseline_co2_kg_per_hour=round(baseline_co2, 1),
            co2_reduction_percent=round(co2_reduction, 1),
            dispatch_recommendation=recommendation,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS"
        )

    def _create_baseline_output(self, input_data, start_time, baseline_cost, baseline_co2):
        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "system": input_data.system_id}, sort_keys=True).encode()
        ).hexdigest()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        return CHPCoordinatorOutput(
            analysis_id=f"CHP-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            system_id=input_data.system_id,
            chp_electrical_output_kw=0,
            chp_thermal_output_kw=0,
            grid_import_kw=input_data.electrical_demand_kw,
            grid_export_kw=0,
            boiler_output_kw=input_data.thermal_demand_kw,
            power_to_heat_ratio=0,
            total_efficiency_percent=0,
            operating_cost_per_hour=round(baseline_cost, 2),
            baseline_cost_per_hour=round(baseline_cost, 2),
            cost_savings_per_hour=0,
            co2_emissions_kg_per_hour=round(baseline_co2, 1),
            baseline_co2_kg_per_hour=round(baseline_co2, 1),
            co2_reduction_percent=0,
            dispatch_recommendation="CHP unavailable - baseline operation",
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS"
        )


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-036",
    "name": "CHP-COORDINATOR",
    "version": "1.0.0",
    "summary": "Combined heat and power optimization",
    "tags": ["CHP", "cogeneration", "dispatch", "efficiency", "EPA-CHP", "ISO-50001"],
    "owners": ["process-heat-optimization-team"],
    "standards": [
        {"ref": "EPA CHP", "description": "EPA Combined Heat and Power Partnership"},
        {"ref": "ISO 50001", "description": "Energy Management Systems"}
    ]
}
