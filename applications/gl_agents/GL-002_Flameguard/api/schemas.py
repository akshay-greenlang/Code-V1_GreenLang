"""
GL-002 FLAMEGUARD - API Schemas

Pydantic models for API request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class OptimizationMode(str, Enum):
    """Optimization mode selection."""
    EFFICIENCY = "efficiency"
    EMISSIONS = "emissions"
    COST = "cost"
    BALANCED = "balanced"


class BoilerState(str, Enum):
    """Boiler operating state."""
    OFFLINE = "offline"
    STANDBY = "standby"
    FIRING = "firing"
    PURGE = "purge"
    LOCKOUT = "lockout"


class SafetyState(str, Enum):
    """Safety system state."""
    NORMAL = "normal"
    ALARM = "alarm"
    TRIP = "trip"
    BYPASSED = "bypassed"


# ============================================================
# Request Models
# ============================================================

class ProcessDataInput(BaseModel):
    """Process data input for calculations."""
    drum_pressure_psig: float = Field(..., ge=0, le=300, description="Steam drum pressure")
    drum_level_inches: float = Field(..., ge=-12, le=12, description="Steam drum level")
    steam_flow_klb_hr: float = Field(..., ge=0, le=1000, description="Steam flow rate")
    steam_temperature_f: float = Field(..., ge=0, le=1200, description="Steam temperature")
    feedwater_temperature_f: float = Field(..., ge=32, le=500, description="Feedwater temperature")
    flue_gas_temperature_f: float = Field(..., ge=0, le=1000, description="Flue gas temperature")
    o2_percent: float = Field(..., ge=0, le=21, description="Flue gas O2 percentage")
    co_ppm: float = Field(..., ge=0, le=2000, description="Flue gas CO concentration")
    fuel_flow_scfh: float = Field(..., ge=0, description="Fuel flow rate")
    fuel_pressure_psig: float = Field(..., ge=0, le=100, description="Fuel header pressure")
    air_flow_scfm: float = Field(..., ge=0, description="Combustion air flow")
    ambient_temperature_f: float = Field(default=70.0, description="Ambient temperature")
    relative_humidity_percent: float = Field(default=50.0, ge=0, le=100, description="Relative humidity")

    class Config:
        schema_extra = {
            "example": {
                "drum_pressure_psig": 125.5,
                "drum_level_inches": 0.5,
                "steam_flow_klb_hr": 150.0,
                "steam_temperature_f": 450.0,
                "feedwater_temperature_f": 220.0,
                "flue_gas_temperature_f": 350.0,
                "o2_percent": 3.5,
                "co_ppm": 25.0,
                "fuel_flow_scfh": 25000.0,
                "fuel_pressure_psig": 15.0,
                "air_flow_scfm": 8500.0,
            }
        }


class OptimizationRequest(BaseModel):
    """Optimization request parameters."""
    boiler_id: str = Field(..., description="Boiler identifier")
    mode: OptimizationMode = Field(default=OptimizationMode.BALANCED, description="Optimization mode")
    target_load_percent: Optional[float] = Field(None, ge=0, le=100, description="Target load percentage")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Additional constraints")
    process_data: Optional[ProcessDataInput] = Field(None, description="Current process data")

    class Config:
        schema_extra = {
            "example": {
                "boiler_id": "BOILER-001",
                "mode": "balanced",
                "target_load_percent": 75.0,
            }
        }


class SetpointCommand(BaseModel):
    """Setpoint adjustment command."""
    boiler_id: str = Field(..., description="Boiler identifier")
    setpoint_type: str = Field(..., description="Type of setpoint")
    value: float = Field(..., description="Setpoint value")
    operator: str = Field(..., description="Operator ID")
    reason: Optional[str] = Field(None, description="Reason for change")

    @validator("setpoint_type")
    def validate_setpoint_type(cls, v):
        valid_types = ["o2_setpoint", "load_demand", "steam_pressure_setpoint", "excess_air_bias"]
        if v not in valid_types:
            raise ValueError(f"Invalid setpoint type. Must be one of: {valid_types}")
        return v


class SafetyBypassRequest(BaseModel):
    """Safety bypass request."""
    boiler_id: str = Field(..., description="Boiler identifier")
    interlock_tag: str = Field(..., description="Interlock tag to bypass")
    reason: str = Field(..., min_length=10, description="Detailed bypass reason")
    duration_minutes: int = Field(..., ge=1, le=480, description="Bypass duration")
    operator: str = Field(..., description="Operator ID")
    supervisor_approval: str = Field(..., description="Supervisor approval ID")


class MultiBoilerLoadRequest(BaseModel):
    """Multi-boiler load dispatch request."""
    boiler_ids: List[str] = Field(..., min_items=1, description="List of boiler IDs")
    total_demand_klb_hr: float = Field(..., ge=0, description="Total steam demand")
    optimization_mode: OptimizationMode = Field(default=OptimizationMode.COST, description="Dispatch mode")


# ============================================================
# Response Models
# ============================================================

class CombustionStatus(BaseModel):
    """Combustion analysis status."""
    o2_percent: float
    o2_setpoint: float
    o2_error: float
    co_ppm: float
    excess_air_percent: float
    stoichiometric_ratio: float
    combustion_quality: str


class EfficiencyMetrics(BaseModel):
    """Efficiency calculation metrics."""
    gross_efficiency_percent: float
    net_efficiency_percent: float
    stack_loss_percent: float
    radiation_loss_percent: float
    blowdown_loss_percent: float
    unaccounted_loss_percent: float
    calculation_method: str
    timestamp: datetime


class EmissionsMetrics(BaseModel):
    """Emissions calculation metrics."""
    nox_lb_hr: float
    nox_ppm: float
    co_lb_hr: float
    co_ppm: float
    co2_ton_hr: float
    so2_lb_hr: float = 0.0
    pm_lb_hr: float = 0.0
    ghg_mtco2e_hr: float
    timestamp: datetime


class SafetyInterlock(BaseModel):
    """Individual safety interlock status."""
    tag: str
    description: str
    value: float
    unit: str
    status: SafetyState
    trip_high: Optional[float] = None
    trip_low: Optional[float] = None
    bypassed: bool = False


class BoilerStatusResponse(BaseModel):
    """Complete boiler status response."""
    boiler_id: str
    name: str
    state: BoilerState
    timestamp: datetime

    # Operating parameters
    load_percent: float
    steam_flow_klb_hr: float
    drum_pressure_psig: float
    drum_level_inches: float
    steam_temperature_f: float

    # Combustion
    combustion: CombustionStatus

    # Efficiency
    efficiency: EfficiencyMetrics

    # Emissions
    emissions: EmissionsMetrics

    # Safety
    safety_status: SafetyState
    active_alarms: int
    active_trips: int

    class Config:
        schema_extra = {
            "example": {
                "boiler_id": "BOILER-001",
                "name": "Main Boiler #1",
                "state": "firing",
                "timestamp": "2024-01-15T10:30:00Z",
                "load_percent": 75.5,
                "steam_flow_klb_hr": 150.0,
                "drum_pressure_psig": 125.5,
                "drum_level_inches": 0.5,
                "steam_temperature_f": 450.0,
                "combustion": {
                    "o2_percent": 3.5,
                    "o2_setpoint": 3.0,
                    "o2_error": 0.5,
                    "co_ppm": 25.0,
                    "excess_air_percent": 20.0,
                    "stoichiometric_ratio": 1.2,
                    "combustion_quality": "good",
                },
                "efficiency": {
                    "gross_efficiency_percent": 85.5,
                    "net_efficiency_percent": 82.1,
                    "stack_loss_percent": 10.5,
                    "radiation_loss_percent": 1.5,
                    "blowdown_loss_percent": 2.0,
                    "unaccounted_loss_percent": 0.4,
                    "calculation_method": "indirect",
                    "timestamp": "2024-01-15T10:30:00Z",
                },
                "emissions": {
                    "nox_lb_hr": 15.5,
                    "nox_ppm": 45.0,
                    "co_lb_hr": 2.5,
                    "co_ppm": 25.0,
                    "co2_ton_hr": 8.5,
                    "ghg_mtco2e_hr": 7.7,
                    "timestamp": "2024-01-15T10:30:00Z",
                },
                "safety_status": "normal",
                "active_alarms": 0,
                "active_trips": 0,
            }
        }


class OptimizationResponse(BaseModel):
    """Optimization result response."""
    boiler_id: str
    mode: OptimizationMode
    timestamp: datetime
    success: bool

    # Current values
    current_efficiency: float
    current_emissions_mtco2e_hr: float
    current_cost_usd_hr: float

    # Recommendations
    recommended_o2_setpoint: float
    recommended_excess_air: float
    recommended_load_percent: Optional[float] = None

    # Predicted improvements
    predicted_efficiency: float
    predicted_emissions_mtco2e_hr: float
    predicted_cost_usd_hr: float

    # Deltas
    efficiency_improvement_percent: float
    emissions_reduction_percent: float
    cost_savings_usd_hr: float

    # Provenance
    calculation_hash: str
    model_version: str


class EfficiencyResponse(BaseModel):
    """Efficiency calculation response."""
    boiler_id: str
    timestamp: datetime

    # Efficiency values
    gross_efficiency_percent: float
    net_efficiency_percent: float
    fuel_efficiency_percent: float

    # Loss breakdown (ASME PTC 4.1)
    losses: Dict[str, float]

    # Heat balance
    heat_input_mmbtu_hr: float
    heat_output_mmbtu_hr: float
    heat_loss_mmbtu_hr: float

    # Calculation metadata
    calculation_method: str
    standard: str = "ASME PTC 4.1"
    calculation_hash: str


class EmissionsResponse(BaseModel):
    """Emissions calculation response."""
    boiler_id: str
    timestamp: datetime

    # Stack emissions
    nox_lb_hr: float
    nox_ppm_corrected: float
    co_lb_hr: float
    co_ppm: float
    co2_ton_hr: float
    so2_lb_hr: float
    pm_lb_hr: float
    voc_lb_hr: float

    # GHG totals
    ghg_mtco2e_hr: float
    annual_ghg_projection_mtco2e: float

    # Regulatory status
    permit_limit_status: Dict[str, str]
    exceedance_risk: str

    # Calculation metadata
    emission_factors_source: str
    calculation_hash: str


class SafetyStatusResponse(BaseModel):
    """Safety system status response."""
    boiler_id: str
    timestamp: datetime

    # Overall status
    bms_state: str
    safety_state: SafetyState
    flame_proven: bool

    # Interlocks
    interlocks: List[SafetyInterlock]
    bypassed_count: int
    alarm_count: int
    trip_count: int

    # Trip history
    last_trip_time: Optional[datetime] = None
    last_trip_reason: Optional[str] = None

    # Permissives
    permissives_satisfied: Dict[str, bool]


class LoadDispatchResponse(BaseModel):
    """Multi-boiler load dispatch response."""
    timestamp: datetime
    total_demand_klb_hr: float
    optimization_mode: OptimizationMode

    # Dispatch results
    boiler_allocations: Dict[str, float]  # boiler_id -> load_klb_hr
    total_allocated_klb_hr: float

    # Performance metrics
    total_efficiency_percent: float
    total_emissions_mtco2e_hr: float
    total_cost_usd_hr: float

    # Comparison to baseline
    efficiency_improvement: float
    emissions_reduction: float
    cost_savings_usd_hr: float


class HealthCheckResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    timestamp: datetime
    components: Dict[str, str]
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    code: int
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
