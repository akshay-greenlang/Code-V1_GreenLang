"""
GL-019 HEATSCHEDULER - Schema Definitions

Pydantic models for heat scheduling inputs, outputs, and results.
These schemas define the data contracts for the HeatSchedulerAgent.

Key Features:
    - Load forecast results with confidence intervals
    - Thermal storage state and dispatch schedules
    - Demand charge optimization results
    - Production schedule integration
    - Weather forecast data models
    - Complete scheduling output with provenance

Author: GreenLang Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class ScheduleStatus(Enum):
    """Schedule optimization status."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"


class LoadForecastStatus(Enum):
    """Load forecast status."""
    SUCCESS = "success"
    DEGRADED = "degraded"  # Using fallback model
    FAILED = "failed"


class StorageMode(Enum):
    """Thermal storage operating mode."""
    CHARGING = "charging"
    DISCHARGING = "discharging"
    IDLE = "idle"
    STANDBY = "standby"


class DemandAlertLevel(Enum):
    """Demand alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ScheduleAction(Enum):
    """Types of schedule actions."""
    START = "start"
    STOP = "stop"
    RAMP_UP = "ramp_up"
    RAMP_DOWN = "ramp_down"
    SETPOINT_CHANGE = "setpoint_change"
    STORAGE_CHARGE = "storage_charge"
    STORAGE_DISCHARGE = "storage_discharge"
    LOAD_SHIFT = "load_shift"


class ProductionStatus(Enum):
    """Production schedule status."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    CANCELLED = "cancelled"


# =============================================================================
# LOAD FORECASTING SCHEMAS
# =============================================================================

class LoadForecastPoint(BaseModel):
    """Single point in load forecast."""

    timestamp: datetime = Field(..., description="Forecast timestamp")
    load_kw: float = Field(..., ge=0, description="Predicted load (kW)")
    lower_bound_kw: float = Field(..., ge=0, description="Lower confidence bound")
    upper_bound_kw: float = Field(..., ge=0, description="Upper confidence bound")
    confidence: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence level"
    )


class LoadForecastResult(BaseModel):
    """Complete load forecast result."""

    forecast_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Forecast identifier"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Forecast generation timestamp"
    )
    status: LoadForecastStatus = Field(
        default=LoadForecastStatus.SUCCESS,
        description="Forecast status"
    )

    # Forecast data
    forecast_points: List[LoadForecastPoint] = Field(
        default_factory=list,
        description="Forecast time series"
    )
    forecast_horizon_hours: int = Field(..., description="Forecast horizon")
    resolution_minutes: int = Field(default=15, description="Time resolution")

    # Metrics
    model_used: str = Field(default="ensemble", description="Model used")
    mape_pct: Optional[float] = Field(None, ge=0, description="MAPE (%)")
    rmse_kw: Optional[float] = Field(None, ge=0, description="RMSE (kW)")
    mae_kw: Optional[float] = Field(None, ge=0, description="MAE (kW)")

    # Feature importance (for explainability)
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None,
        description="Feature importance scores"
    )

    # Aggregates
    peak_load_kw: Optional[float] = Field(None, ge=0, description="Peak load in forecast")
    peak_load_time: Optional[datetime] = Field(None, description="Time of peak load")
    min_load_kw: Optional[float] = Field(None, ge=0, description="Minimum load")
    avg_load_kw: Optional[float] = Field(None, ge=0, description="Average load")
    total_energy_kwh: Optional[float] = Field(None, ge=0, description="Total energy (kWh)")

    # Data quality
    data_quality_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Input data quality score"
    )
    missing_data_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of missing input data"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# THERMAL STORAGE SCHEMAS
# =============================================================================

class StorageStatePoint(BaseModel):
    """State of thermal storage at a point in time."""

    timestamp: datetime = Field(..., description="State timestamp")
    state_of_charge_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="State of charge (%)"
    )
    state_of_charge_kwh: float = Field(
        ...,
        ge=0,
        description="State of charge (kWh)"
    )
    temperature_c: Optional[float] = Field(
        None,
        description="Storage temperature (C)"
    )
    mode: StorageMode = Field(
        default=StorageMode.IDLE,
        description="Operating mode"
    )
    power_kw: float = Field(
        default=0.0,
        description="Charge/discharge power (kW), positive=charging"
    )

    class Config:
        use_enum_values = True


class StorageDispatchSchedule(BaseModel):
    """Dispatch schedule for thermal storage unit."""

    storage_id: str = Field(..., description="Storage unit identifier")
    schedule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Schedule identifier"
    )

    # Schedule
    dispatch_points: List[StorageStatePoint] = Field(
        default_factory=list,
        description="Scheduled dispatch time series"
    )

    # Summary
    total_charge_kwh: float = Field(
        default=0.0,
        ge=0,
        description="Total energy charged (kWh)"
    )
    total_discharge_kwh: float = Field(
        default=0.0,
        ge=0,
        description="Total energy discharged (kWh)"
    )
    charge_hours: float = Field(
        default=0.0,
        ge=0,
        description="Total charging hours"
    )
    discharge_hours: float = Field(
        default=0.0,
        ge=0,
        description="Total discharging hours"
    )
    cycles: float = Field(
        default=0.0,
        ge=0,
        description="Equivalent full cycles"
    )

    # Economics
    energy_arbitrage_usd: float = Field(
        default=0.0,
        description="Energy arbitrage value ($)"
    )
    demand_savings_usd: float = Field(
        default=0.0,
        ge=0,
        description="Demand charge savings ($)"
    )

    # Constraints satisfaction
    min_soc_maintained: bool = Field(
        default=True,
        description="Minimum SOC constraint met"
    )
    reserve_maintained: bool = Field(
        default=True,
        description="Emergency reserve maintained"
    )


class ThermalStorageResult(BaseModel):
    """Complete thermal storage optimization result."""

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Result identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result timestamp"
    )

    # Unit schedules
    unit_schedules: List[StorageDispatchSchedule] = Field(
        default_factory=list,
        description="Per-unit dispatch schedules"
    )

    # Aggregated metrics
    total_storage_capacity_kwh: float = Field(
        default=0.0,
        ge=0,
        description="Total storage capacity"
    )
    current_soc_kwh: float = Field(
        default=0.0,
        ge=0,
        description="Current total SOC"
    )
    total_energy_arbitrage_usd: float = Field(
        default=0.0,
        description="Total energy arbitrage ($)"
    )
    total_demand_savings_usd: float = Field(
        default=0.0,
        ge=0,
        description="Total demand savings ($)"
    )
    total_savings_usd: float = Field(
        default=0.0,
        ge=0,
        description="Total savings ($)"
    )


# =============================================================================
# DEMAND CHARGE SCHEMAS
# =============================================================================

class DemandPeriod(BaseModel):
    """Demand measurement period."""

    period_start: datetime = Field(..., description="Period start time")
    period_end: datetime = Field(..., description="Period end time")
    avg_demand_kw: float = Field(..., ge=0, description="Average demand (kW)")
    peak_demand_kw: float = Field(..., ge=0, description="Peak demand (kW)")
    is_on_peak: bool = Field(default=False, description="Is on-peak period")


class DemandChargeResult(BaseModel):
    """Demand charge optimization result."""

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Result identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result timestamp"
    )

    # Peak demand
    baseline_peak_kw: float = Field(
        ...,
        ge=0,
        description="Baseline peak demand (kW)"
    )
    optimized_peak_kw: float = Field(
        ...,
        ge=0,
        description="Optimized peak demand (kW)"
    )
    peak_reduction_kw: float = Field(
        default=0.0,
        ge=0,
        description="Peak reduction (kW)"
    )
    peak_reduction_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Peak reduction (%)"
    )

    # Timing
    peak_time_baseline: Optional[datetime] = Field(
        None,
        description="Time of baseline peak"
    )
    peak_time_optimized: Optional[datetime] = Field(
        None,
        description="Time of optimized peak"
    )

    # Costs
    baseline_demand_charge_usd: float = Field(
        ...,
        ge=0,
        description="Baseline demand charge ($)"
    )
    optimized_demand_charge_usd: float = Field(
        ...,
        ge=0,
        description="Optimized demand charge ($)"
    )
    demand_charge_savings_usd: float = Field(
        default=0.0,
        ge=0,
        description="Demand charge savings ($)"
    )

    # Ratchet
    annual_ratchet_peak_kw: float = Field(
        default=0.0,
        ge=0,
        description="Annual ratchet peak (kW)"
    )
    ratchet_impact_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual ratchet cost impact ($)"
    )

    # Load shifting
    load_shifted_kwh: float = Field(
        default=0.0,
        ge=0,
        description="Energy shifted (kWh)"
    )
    load_shift_savings_usd: float = Field(
        default=0.0,
        ge=0,
        description="Load shifting savings ($)"
    )

    # Alerts
    peak_limit_exceeded: bool = Field(
        default=False,
        description="Peak limit exceeded"
    )
    alert_level: Optional[DemandAlertLevel] = Field(
        default=None,
        description="Alert level if limit exceeded"
    )
    alert_message: Optional[str] = Field(
        default=None,
        description="Alert message"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# PRODUCTION PLANNING SCHEMAS
# =============================================================================

class ProductionOrder(BaseModel):
    """Production order for scheduling."""

    order_id: str = Field(..., description="Order identifier")
    product_id: Optional[str] = Field(None, description="Product identifier")
    product_name: Optional[str] = Field(None, description="Product name")

    # Timing
    scheduled_start: datetime = Field(..., description="Scheduled start time")
    scheduled_end: datetime = Field(..., description="Scheduled end time")
    duration_hours: float = Field(..., gt=0, description="Duration (hours)")

    # Heat requirements
    heat_load_kw: float = Field(..., gt=0, description="Heat load (kW)")
    temperature_c: Optional[float] = Field(None, description="Required temperature")
    ramp_up_time_minutes: int = Field(
        default=30,
        ge=0,
        description="Ramp-up time (minutes)"
    )

    # Priority and flexibility
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority (1=lowest, 10=highest)"
    )
    is_flexible: bool = Field(
        default=False,
        description="Can be rescheduled"
    )
    earliest_start: Optional[datetime] = Field(
        None,
        description="Earliest allowed start"
    )
    latest_end: Optional[datetime] = Field(
        None,
        description="Latest allowed end"
    )

    # Status
    status: ProductionStatus = Field(
        default=ProductionStatus.SCHEDULED,
        description="Order status"
    )

    class Config:
        use_enum_values = True


class ProductionScheduleResult(BaseModel):
    """Production schedule integration result."""

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Result identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result timestamp"
    )

    # Orders
    scheduled_orders: List[ProductionOrder] = Field(
        default_factory=list,
        description="Scheduled production orders"
    )
    rescheduled_orders: List[ProductionOrder] = Field(
        default_factory=list,
        description="Orders that were rescheduled"
    )

    # Metrics
    total_orders: int = Field(default=0, ge=0, description="Total orders")
    orders_on_time: int = Field(default=0, ge=0, description="Orders on time")
    orders_rescheduled: int = Field(default=0, ge=0, description="Orders rescheduled")
    total_heat_load_kwh: float = Field(
        default=0.0,
        ge=0,
        description="Total heat energy (kWh)"
    )

    # Cost impact
    scheduling_cost_savings_usd: float = Field(
        default=0.0,
        ge=0,
        description="Cost savings from rescheduling ($)"
    )


# =============================================================================
# WEATHER SCHEMAS
# =============================================================================

class WeatherForecastPoint(BaseModel):
    """Single point in weather forecast."""

    timestamp: datetime = Field(..., description="Forecast timestamp")
    temperature_c: float = Field(..., description="Temperature (C)")
    humidity_pct: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Relative humidity (%)"
    )
    solar_radiation_w_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Solar radiation (W/m2)"
    )
    wind_speed_m_s: Optional[float] = Field(
        None,
        ge=0,
        description="Wind speed (m/s)"
    )
    cloud_cover_pct: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Cloud cover (%)"
    )
    precipitation_mm: Optional[float] = Field(
        None,
        ge=0,
        description="Precipitation (mm)"
    )

    # Derived metrics
    heating_degree_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Heating degree hours"
    )
    cooling_degree_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Cooling degree hours"
    )


class WeatherForecastResult(BaseModel):
    """Complete weather forecast result."""

    forecast_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Forecast identifier"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Forecast generation time"
    )
    provider: str = Field(default="openweathermap", description="Data provider")

    # Location
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")

    # Forecast
    forecast_points: List[WeatherForecastPoint] = Field(
        default_factory=list,
        description="Weather forecast time series"
    )
    forecast_horizon_hours: int = Field(..., description="Forecast horizon")

    # Aggregates
    avg_temperature_c: Optional[float] = Field(None, description="Average temperature")
    max_temperature_c: Optional[float] = Field(None, description="Maximum temperature")
    min_temperature_c: Optional[float] = Field(None, description="Minimum temperature")
    total_heating_degree_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Total heating degree hours"
    )
    total_cooling_degree_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Total cooling degree hours"
    )


# =============================================================================
# SCHEDULE ACTION SCHEMAS
# =============================================================================

class ScheduleActionItem(BaseModel):
    """Single action in the optimized schedule."""

    action_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Action identifier"
    )
    timestamp: datetime = Field(..., description="Action timestamp")
    action_type: ScheduleAction = Field(..., description="Action type")

    # Target
    equipment_id: Optional[str] = Field(None, description="Target equipment")
    storage_id: Optional[str] = Field(None, description="Target storage")

    # Setpoints
    power_setpoint_kw: Optional[float] = Field(
        None,
        ge=0,
        description="Power setpoint (kW)"
    )
    temperature_setpoint_c: Optional[float] = Field(
        None,
        description="Temperature setpoint (C)"
    )

    # Duration
    duration_minutes: Optional[int] = Field(
        None,
        ge=0,
        description="Action duration (minutes)"
    )

    # Rationale
    reason: str = Field(default="", description="Action rationale")
    expected_savings_usd: float = Field(
        default=0.0,
        ge=0,
        description="Expected savings ($)"
    )

    # Priority
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Action priority"
    )
    is_mandatory: bool = Field(
        default=False,
        description="Mandatory action"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# MAIN INPUT/OUTPUT SCHEMAS
# =============================================================================

class HeatSchedulerInput(BaseModel):
    """Input data for heat scheduling optimization."""

    # Identity
    facility_id: str = Field(..., description="Facility identifier")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp"
    )

    # Time horizon
    optimization_horizon_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Optimization horizon (hours)"
    )
    time_step_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Time step resolution (minutes)"
    )

    # Current state
    current_load_kw: float = Field(
        ...,
        ge=0,
        description="Current heat load (kW)"
    )
    current_storage_soc_pct: Optional[Dict[str, float]] = Field(
        default=None,
        description="Current storage SOC by unit (%)"
    )
    current_equipment_status: Optional[Dict[str, str]] = Field(
        default=None,
        description="Current equipment status by ID"
    )

    # Forecasts (optional, will be generated if not provided)
    load_forecast: Optional[LoadForecastResult] = Field(
        default=None,
        description="Load forecast"
    )
    weather_forecast: Optional[WeatherForecastResult] = Field(
        default=None,
        description="Weather forecast"
    )

    # Production schedule
    production_orders: Optional[List[ProductionOrder]] = Field(
        default=None,
        description="Production orders to schedule"
    )

    # Real-time pricing (if available)
    rtp_prices: Optional[List[Tuple[datetime, float]]] = Field(
        default=None,
        description="Real-time prices (timestamp, $/kWh)"
    )

    # Constraints
    max_peak_demand_kw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum peak demand constraint"
    )
    must_run_periods: Optional[List[Tuple[datetime, datetime]]] = Field(
        default=None,
        description="Periods with mandatory operation"
    )
    blackout_periods: Optional[List[Tuple[datetime, datetime]]] = Field(
        default=None,
        description="Periods where scheduling is not allowed"
    )

    # Preferences
    prefer_storage_discharge_during_peak: bool = Field(
        default=True,
        description="Prefer storage discharge during peak"
    )
    allow_production_rescheduling: bool = Field(
        default=True,
        description="Allow flexible order rescheduling"
    )


class HeatSchedulerOutput(BaseModel):
    """Complete output from heat scheduling optimization."""

    # Identity
    facility_id: str = Field(..., description="Facility identifier")
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Output timestamp"
    )

    # Status
    status: ScheduleStatus = Field(
        default=ScheduleStatus.OPTIMAL,
        description="Optimization status"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing time (ms)"
    )
    solver_gap_pct: Optional[float] = Field(
        default=None,
        ge=0,
        description="Solver optimality gap (%)"
    )

    # Schedule
    schedule_horizon_hours: int = Field(..., description="Schedule horizon")
    schedule_actions: List[ScheduleActionItem] = Field(
        default_factory=list,
        description="Scheduled actions"
    )

    # Component results
    load_forecast: LoadForecastResult = Field(
        ...,
        description="Load forecast used"
    )
    storage_result: Optional[ThermalStorageResult] = Field(
        default=None,
        description="Storage optimization result"
    )
    demand_result: Optional[DemandChargeResult] = Field(
        default=None,
        description="Demand charge result"
    )
    production_result: Optional[ProductionScheduleResult] = Field(
        default=None,
        description="Production schedule result"
    )
    weather_forecast: Optional[WeatherForecastResult] = Field(
        default=None,
        description="Weather forecast used"
    )

    # Cost summary
    baseline_cost_usd: float = Field(
        ...,
        ge=0,
        description="Baseline energy cost ($)"
    )
    optimized_cost_usd: float = Field(
        ...,
        ge=0,
        description="Optimized energy cost ($)"
    )
    total_savings_usd: float = Field(
        default=0.0,
        ge=0,
        description="Total savings ($)"
    )
    savings_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Savings by category"
    )

    # Energy metrics
    total_energy_kwh: float = Field(
        default=0.0,
        ge=0,
        description="Total energy consumption (kWh)"
    )
    peak_demand_kw: float = Field(
        default=0.0,
        ge=0,
        description="Peak demand (kW)"
    )
    average_load_kw: float = Field(
        default=0.0,
        ge=0,
        description="Average load (kW)"
    )
    load_factor_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Load factor (%)"
    )

    # Emissions
    co2_emissions_kg: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO2 emissions (kg)"
    )
    emissions_reduction_kg: Optional[float] = Field(
        default=None,
        ge=0,
        description="Emissions reduction (kg)"
    )

    # KPIs
    kpis: Dict[str, float] = Field(
        default_factory=dict,
        description="Key performance indicators"
    )

    # Alerts and warnings
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active alerts"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Optimization warnings"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )
    input_hash: Optional[str] = Field(
        default=None,
        description="Input data hash"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ScheduleStatus",
    "LoadForecastStatus",
    "StorageMode",
    "DemandAlertLevel",
    "ScheduleAction",
    "ProductionStatus",
    # Load Forecasting
    "LoadForecastPoint",
    "LoadForecastResult",
    # Thermal Storage
    "StorageStatePoint",
    "StorageDispatchSchedule",
    "ThermalStorageResult",
    # Demand Charge
    "DemandPeriod",
    "DemandChargeResult",
    # Production
    "ProductionOrder",
    "ProductionScheduleResult",
    # Weather
    "WeatherForecastPoint",
    "WeatherForecastResult",
    # Schedule Actions
    "ScheduleActionItem",
    # Main I/O
    "HeatSchedulerInput",
    "HeatSchedulerOutput",
]
