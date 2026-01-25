"""Input/Output models for GL-023 Heat Load Balancer Agent."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EquipmentUnit(BaseModel):
    """Individual heat generating equipment unit."""

    unit_id: str = Field(..., description="Unique equipment identifier")
    unit_type: str = Field(..., description="BOILER, FURNACE, HEATER, CHP")
    current_load_mw: float = Field(..., ge=0, description="Current thermal load (MW)")
    min_load_mw: float = Field(..., ge=0, description="Minimum stable load (MW)")
    max_load_mw: float = Field(..., ge=0, description="Maximum capacity (MW)")
    current_efficiency_pct: float = Field(..., ge=0, le=100, description="Current efficiency (%)")

    # Efficiency curve coefficients: η = a + b*L + c*L²
    efficiency_curve_a: float = Field(0.0, description="Efficiency intercept")
    efficiency_curve_b: float = Field(0.0, description="Efficiency linear term")
    efficiency_curve_c: float = Field(0.0, description="Efficiency quadratic term")

    # Operating status
    is_available: bool = Field(True, description="Unit available for loading")
    is_running: bool = Field(True, description="Unit currently running")
    startup_time_min: float = Field(30, description="Cold start time (min)")
    ramp_rate_mw_per_min: float = Field(1.0, description="Load ramp rate (MW/min)")

    # Costs
    fuel_cost_per_mwh: float = Field(..., description="Fuel cost ($/MWh)")
    maintenance_cost_per_mwh: float = Field(0.0, description="Variable maintenance ($/MWh)")
    startup_cost: float = Field(0.0, description="Cost per startup ($)")

    # Constraints
    min_run_time_hr: float = Field(1.0, description="Minimum run time (hr)")
    emissions_factor_kg_co2_mwh: float = Field(200, description="CO2 emissions (kg/MWh)")


class LoadBalancerInput(BaseModel):
    """Input parameters for heat load balancing."""

    # Equipment fleet
    equipment: List[EquipmentUnit] = Field(..., min_length=1, description="Equipment units")

    # Total demand
    total_heat_demand_mw: float = Field(..., ge=0, description="Total heat demand (MW)")
    demand_forecast_1hr_mw: Optional[float] = Field(None, description="1-hour demand forecast")
    demand_forecast_4hr_mw: Optional[float] = Field(None, description="4-hour demand forecast")

    # Optimization objectives
    optimization_mode: str = Field("COST", description="COST, EFFICIENCY, EMISSIONS, or BALANCED")
    cost_weight: float = Field(1.0, ge=0, le=1, description="Cost optimization weight")
    efficiency_weight: float = Field(0.0, ge=0, le=1, description="Efficiency optimization weight")
    emissions_weight: float = Field(0.0, ge=0, le=1, description="Emissions optimization weight")

    # Constraints
    min_spinning_reserve_pct: float = Field(10.0, ge=0, le=50, description="Min spinning reserve (%)")
    max_units_starting: int = Field(1, ge=0, description="Max simultaneous startups")

    # Energy prices
    electricity_price_per_mwh: Optional[float] = Field(None, description="Grid price for CHP export")
    natural_gas_price_per_mmbtu: Optional[float] = Field(None, description="Natural gas price")
    carbon_price_per_ton: float = Field(0.0, ge=0, description="Carbon price ($/ton CO2)")

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LoadAllocation(BaseModel):
    """Load allocation for a single unit."""

    unit_id: str
    target_load_mw: float
    load_change_mw: float
    action: str  # INCREASE, DECREASE, MAINTAIN, START, STOP
    efficiency_at_load_pct: float
    fuel_consumption_mw: float
    hourly_cost: float
    hourly_emissions_kg_co2: float


class LoadBalancerOutput(BaseModel):
    """Output from heat load balancer agent."""

    # Allocation results
    allocations: List[LoadAllocation] = Field(..., description="Load allocations per unit")

    # Fleet metrics
    total_capacity_mw: float = Field(..., description="Total available capacity")
    total_allocated_mw: float = Field(..., description="Total allocated load")
    spinning_reserve_mw: float = Field(..., description="Available spinning reserve")
    spinning_reserve_pct: float = Field(..., description="Spinning reserve percentage")

    # Efficiency metrics
    fleet_efficiency_pct: float = Field(..., description="Weighted average efficiency")
    efficiency_vs_equal_load_pct: float = Field(..., description="Improvement vs equal loading")

    # Cost metrics
    total_hourly_cost: float = Field(..., description="Total hourly operating cost ($)")
    cost_per_mwh: float = Field(..., description="Blended cost per MWh")
    cost_savings_vs_equal_pct: float = Field(..., description="Savings vs equal loading (%)")

    # Emissions metrics
    total_hourly_emissions_kg: float = Field(..., description="Total hourly CO2 (kg)")
    emissions_intensity_kg_mwh: float = Field(..., description="CO2 intensity (kg/MWh)")

    # Operating status
    units_running: int = Field(..., description="Number of units running")
    units_starting: int = Field(..., description="Number of units starting")
    units_stopping: int = Field(..., description="Number of units stopping")

    # Constraints status
    constraints_satisfied: bool = Field(..., description="All constraints met")
    constraint_violations: List[str] = Field(default_factory=list)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")
    optimization_method: str = Field("MILP", description="Optimization method used")
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")
