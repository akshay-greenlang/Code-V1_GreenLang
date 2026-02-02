"""Input/Output models for GL-022 Superheater Control Agent."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SuperheaterInput(BaseModel):
    """Input parameters for superheater temperature control."""

    # Steam conditions
    inlet_steam_temp_c: float = Field(..., ge=100, le=700, description="Inlet steam temperature (°C)")
    outlet_steam_temp_c: float = Field(..., ge=100, le=700, description="Current outlet steam temperature (°C)")
    target_steam_temp_c: float = Field(..., ge=200, le=650, description="Target outlet steam temperature (°C)")
    steam_pressure_bar: float = Field(..., ge=1, le=200, description="Steam pressure (bar)")
    steam_flow_kg_s: float = Field(..., ge=0, description="Steam mass flow rate (kg/s)")

    # Desuperheater spray
    spray_water_temp_c: float = Field(..., ge=10, le=200, description="Spray water temperature (°C)")
    current_spray_flow_kg_s: float = Field(0.0, ge=0, description="Current spray water flow (kg/s)")
    max_spray_flow_kg_s: float = Field(..., ge=0, description="Maximum spray flow capacity (kg/s)")
    spray_valve_position_pct: float = Field(0.0, ge=0, le=100, description="Current spray valve position (%)")

    # Firing conditions
    burner_load_pct: float = Field(..., ge=0, le=100, description="Burner load (%)")
    flue_gas_temp_c: float = Field(..., ge=100, le=1500, description="Flue gas temperature (°C)")
    excess_air_pct: float = Field(..., ge=0, le=100, description="Excess air (%)")

    # Process requirements
    process_temp_tolerance_c: float = Field(5.0, ge=1, le=20, description="Allowable temperature deviation (°C)")
    min_superheat_c: float = Field(20.0, ge=5, description="Minimum required superheat (°C)")

    # Equipment limits
    max_tube_metal_temp_c: float = Field(600.0, ge=400, le=800, description="Maximum tube metal temperature (°C)")
    current_tube_metal_temp_c: Optional[float] = Field(None, description="Current tube metal temperature (°C)")

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    equipment_id: str = Field(..., description="Superheater equipment ID")


class SprayControlAction(BaseModel):
    """Spray water control recommendation."""

    target_spray_flow_kg_s: float = Field(..., description="Recommended spray flow (kg/s)")
    valve_position_pct: float = Field(..., description="Target valve position (%)")
    rate_of_change_pct_per_min: float = Field(..., description="Maximum rate of change")
    action_type: str = Field(..., description="INCREASE, DECREASE, or MAINTAIN")


class ControlParameters(BaseModel):
    """Control loop parameters."""

    kp: float = Field(..., description="Proportional gain")
    ki: float = Field(..., description="Integral gain")
    kd: float = Field(..., description="Derivative gain")
    deadband_c: float = Field(..., description="Control deadband (°C)")
    max_rate_c_per_min: float = Field(..., description="Max temperature rate (°C/min)")


class SuperheaterOutput(BaseModel):
    """Output from superheater control agent."""

    # Control actions
    spray_control: SprayControlAction
    control_parameters: ControlParameters

    # Analysis results
    current_superheat_c: float = Field(..., description="Current superheat above saturation")
    saturation_temp_c: float = Field(..., description="Saturation temperature at pressure")
    temperature_deviation_c: float = Field(..., description="Deviation from target")
    within_tolerance: bool = Field(..., description="Temperature within tolerance")

    # Energy metrics
    spray_energy_loss_kw: float = Field(..., description="Energy loss from spray water")
    spray_water_cost_per_hour: float = Field(0.0, description="Cost of spray water usage")

    # Safety status
    tube_metal_margin_c: float = Field(..., description="Margin below max tube temp")
    safety_status: str = Field(..., description="SAFE, WARNING, or CRITICAL")

    # Efficiency
    thermal_efficiency_impact_pct: float = Field(..., description="Impact on thermal efficiency")

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 hash of calculation")
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
