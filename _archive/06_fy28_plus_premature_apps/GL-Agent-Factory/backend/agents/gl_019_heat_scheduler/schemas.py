"""
Pydantic Schemas for GL-019 HEATSCHEDULER Agent

This module defines all input/output data models for the ProcessHeatingSchedulerAgent.
Models include production schedules, energy tariffs, equipment availability,
weather forecasts, demand predictions, and optimized heating schedules.

All models use Pydantic for validation and serialization, ensuring
data integrity at all boundaries.

Standards:
- ISO 50001 Energy Management Systems
- IEC 61970 Energy Management System Integration
- OpenADR 2.0 Demand Response Standards
"""

from datetime import datetime, date, time
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class TariffType(str, Enum):
    """Energy tariff structure types."""
    FLAT = "flat"
    TOU = "time_of_use"  # Time-of-Use
    RTP = "real_time_pricing"  # Real-Time Pricing
    CPP = "critical_peak_pricing"  # Critical Peak Pricing
    DEMAND_CHARGE = "demand_charge"


class EquipmentStatus(str, Enum):
    """Equipment availability status."""
    AVAILABLE = "available"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    STANDBY = "standby"


class StorageMode(str, Enum):
    """Thermal storage operating mode."""
    CHARGING = "charging"
    DISCHARGING = "discharging"
    IDLE = "idle"
    MAINTENANCE = "maintenance"


class SchedulePriority(str, Enum):
    """Schedule optimization priority."""
    COST = "cost"  # Minimize energy cost
    EMISSIONS = "emissions"  # Minimize carbon emissions
    RELIABILITY = "reliability"  # Maximize equipment reliability
    BALANCED = "balanced"  # Balance all factors


class ForecastConfidence(str, Enum):
    """Confidence level for forecasts."""
    HIGH = "high"  # >90% confidence
    MEDIUM = "medium"  # 70-90% confidence
    LOW = "low"  # <70% confidence


class GridSignalType(str, Enum):
    """Grid demand response signal types."""
    NORMAL = "normal"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# =============================================================================
# Supporting Models
# =============================================================================

class TimeSlot(BaseModel):
    """Represents a time slot in the schedule."""

    start_time: datetime = Field(..., description="Start of time slot")
    end_time: datetime = Field(..., description="End of time slot")
    duration_minutes: int = Field(default=60, ge=1, le=1440, description="Slot duration in minutes")

    @root_validator(skip_on_failure=True)
    def validate_times(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure end_time is after start_time."""
        start = values.get('start_time')
        end = values.get('end_time')
        if start and end and end <= start:
            raise ValueError("end_time must be after start_time")
        return values


class ProductionOrder(BaseModel):
    """Production order requiring process heat."""

    order_id: str = Field(..., min_length=1, description="Unique order identifier")
    product_type: str = Field(..., description="Type of product being produced")
    quantity: float = Field(..., gt=0, description="Production quantity (units)")
    heat_requirement_kwh: float = Field(..., ge=0, description="Total heat energy required (kWh)")
    temperature_c: float = Field(..., ge=0, le=2000, description="Required process temperature (C)")
    earliest_start: datetime = Field(..., description="Earliest allowable start time")
    latest_end: datetime = Field(..., description="Latest allowable completion time")
    priority: int = Field(default=5, ge=1, le=10, description="Priority (1=highest, 10=lowest)")
    can_interrupt: bool = Field(default=False, description="Whether order can be interrupted")

    @root_validator(skip_on_failure=True)
    def validate_timeline(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate earliest_start before latest_end."""
        start = values.get('earliest_start')
        end = values.get('latest_end')
        if start and end and end <= start:
            raise ValueError("latest_end must be after earliest_start")
        return values


class TariffPeriod(BaseModel):
    """Time-of-use tariff period definition."""

    period_name: str = Field(..., description="Period identifier (e.g., 'peak', 'off_peak')")
    start_hour: int = Field(..., ge=0, le=23, description="Start hour (0-23)")
    end_hour: int = Field(..., ge=0, le=23, description="End hour (0-23)")
    days_of_week: List[int] = Field(
        default=[0, 1, 2, 3, 4],  # Monday-Friday
        description="Days when period applies (0=Monday, 6=Sunday)"
    )
    energy_rate_per_kwh: float = Field(..., ge=0, description="Energy rate ($/kWh)")
    demand_rate_per_kw: float = Field(default=0, ge=0, description="Demand charge ($/kW)")

    @validator('days_of_week')
    def validate_days(cls, v: List[int]) -> List[int]:
        """Validate days are in valid range."""
        for day in v:
            if day < 0 or day > 6:
                raise ValueError(f"Day {day} must be between 0 and 6")
        return v


class EnergyTariff(BaseModel):
    """Complete energy tariff structure."""

    tariff_id: str = Field(..., description="Tariff identifier")
    tariff_type: TariffType = Field(..., description="Type of tariff structure")
    utility_name: str = Field(..., description="Utility provider name")
    effective_date: date = Field(..., description="Date tariff becomes effective")

    # Flat rate (if applicable)
    flat_rate_per_kwh: Optional[float] = Field(None, ge=0, description="Flat rate ($/kWh)")

    # TOU periods
    tou_periods: List[TariffPeriod] = Field(
        default_factory=list,
        description="Time-of-use periods"
    )

    # Demand charges
    demand_charge_per_kw: float = Field(default=0, ge=0, description="Monthly demand charge ($/kW)")
    ratchet_percentage: float = Field(
        default=0, ge=0, le=100,
        description="Demand ratchet percentage"
    )

    # Carbon pricing
    carbon_price_per_tonne: float = Field(default=0, ge=0, description="Carbon price ($/tonne CO2)")
    grid_carbon_intensity_kg_per_kwh: float = Field(
        default=0.4, ge=0, le=2,
        description="Grid carbon intensity (kg CO2/kWh)"
    )


class Equipment(BaseModel):
    """Process heating equipment definition."""

    equipment_id: str = Field(..., min_length=1, description="Unique equipment identifier")
    equipment_type: str = Field(..., description="Type of equipment (furnace, boiler, etc.)")
    capacity_kw: float = Field(..., gt=0, description="Maximum heating capacity (kW)")
    min_load_percent: float = Field(default=20, ge=0, le=100, description="Minimum load percentage")
    efficiency_percent: float = Field(default=85, ge=50, le=100, description="Thermal efficiency (%)")
    ramp_rate_kw_per_min: float = Field(default=10, gt=0, description="Ramp rate (kW/min)")
    startup_time_min: int = Field(default=30, ge=0, description="Startup time (minutes)")
    shutdown_time_min: int = Field(default=15, ge=0, description="Shutdown time (minutes)")

    status: EquipmentStatus = Field(default=EquipmentStatus.AVAILABLE)
    maintenance_windows: List[TimeSlot] = Field(
        default_factory=list,
        description="Scheduled maintenance windows"
    )

    # Operating costs
    fixed_cost_per_hour: float = Field(default=0, ge=0, description="Fixed operating cost ($/hour)")
    variable_cost_per_kwh: float = Field(default=0, ge=0, description="Variable cost ($/kWh)")


class ThermalStorage(BaseModel):
    """Thermal energy storage system."""

    storage_id: str = Field(..., description="Storage system identifier")
    capacity_kwh: float = Field(..., gt=0, description="Total storage capacity (kWh)")
    current_state_of_charge: float = Field(
        ..., ge=0, le=100,
        description="Current state of charge (%)"
    )
    min_soc_percent: float = Field(default=10, ge=0, le=100, description="Minimum SOC (%)")
    max_soc_percent: float = Field(default=95, ge=0, le=100, description="Maximum SOC (%)")

    charge_rate_kw: float = Field(..., gt=0, description="Maximum charge rate (kW)")
    discharge_rate_kw: float = Field(..., gt=0, description="Maximum discharge rate (kW)")
    round_trip_efficiency: float = Field(default=0.85, ge=0.5, le=1, description="Round-trip efficiency")

    standby_losses_percent_per_hour: float = Field(
        default=0.5, ge=0, le=10,
        description="Standby heat loss (%/hour)"
    )

    mode: StorageMode = Field(default=StorageMode.IDLE)


class WeatherForecast(BaseModel):
    """Weather forecast data point."""

    timestamp: datetime = Field(..., description="Forecast timestamp")
    temperature_c: float = Field(..., ge=-50, le=60, description="Ambient temperature (C)")
    humidity_percent: float = Field(default=50, ge=0, le=100, description="Relative humidity (%)")
    wind_speed_ms: float = Field(default=0, ge=0, le=100, description="Wind speed (m/s)")
    solar_irradiance_w_per_m2: float = Field(
        default=0, ge=0, le=1400,
        description="Solar irradiance (W/m2)"
    )
    cloud_cover_percent: float = Field(default=50, ge=0, le=100, description="Cloud cover (%)")


class HistoricalDemand(BaseModel):
    """Historical demand data point."""

    timestamp: datetime = Field(..., description="Measurement timestamp")
    demand_kw: float = Field(..., ge=0, description="Measured demand (kW)")
    temperature_c: Optional[float] = Field(None, description="Ambient temperature at time")
    production_rate: Optional[float] = Field(None, ge=0, description="Production rate at time")


class GridSignal(BaseModel):
    """Grid demand response signal."""

    timestamp: datetime = Field(..., description="Signal timestamp")
    signal_type: GridSignalType = Field(..., description="Signal severity level")
    duration_minutes: int = Field(default=60, ge=1, description="Signal duration")
    price_multiplier: float = Field(default=1.0, ge=0, description="Price multiplier")
    requested_reduction_kw: float = Field(default=0, ge=0, description="Requested load reduction")


class UncertaintyBounds(BaseModel):
    """Uncertainty bounds for predictions."""

    point_estimate: float = Field(..., description="Point estimate (mean/median)")
    lower_bound_95: float = Field(..., description="95% confidence lower bound")
    upper_bound_95: float = Field(..., description="95% confidence upper bound")
    lower_bound_80: float = Field(..., description="80% confidence lower bound")
    upper_bound_80: float = Field(..., description="80% confidence upper bound")
    confidence_level: ForecastConfidence = Field(..., description="Overall confidence")

    @root_validator(skip_on_failure=True)
    def validate_bounds(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate bounds are in correct order."""
        l95 = values.get('lower_bound_95', 0)
        l80 = values.get('lower_bound_80', 0)
        point = values.get('point_estimate', 0)
        u80 = values.get('upper_bound_80', 0)
        u95 = values.get('upper_bound_95', 0)

        if not (l95 <= l80 <= point <= u80 <= u95):
            logger.warning(f"Uncertainty bounds not in expected order: {l95}, {l80}, {point}, {u80}, {u95}")
        return values


class DemandPrediction(BaseModel):
    """Demand prediction for a time slot."""

    timestamp: datetime = Field(..., description="Prediction timestamp")
    predicted_demand_kw: UncertaintyBounds = Field(..., description="Predicted demand with uncertainty")

    # Feature contributions (SHAP values)
    temperature_contribution: float = Field(default=0, description="Temperature contribution to prediction")
    production_contribution: float = Field(default=0, description="Production contribution to prediction")
    time_of_day_contribution: float = Field(default=0, description="Time-of-day contribution")
    historical_contribution: float = Field(default=0, description="Historical pattern contribution")

    model_version: str = Field(default="1.0.0", description="Forecasting model version")


class ScheduledOperation(BaseModel):
    """Single scheduled heating operation."""

    operation_id: str = Field(..., description="Unique operation identifier")
    equipment_id: str = Field(..., description="Equipment to use")
    order_id: Optional[str] = Field(None, description="Associated production order")

    start_time: datetime = Field(..., description="Scheduled start time")
    end_time: datetime = Field(..., description="Scheduled end time")

    setpoint_kw: float = Field(..., ge=0, description="Power setpoint (kW)")
    expected_cost: float = Field(..., ge=0, description="Expected energy cost ($)")
    expected_emissions_kg: float = Field(default=0, ge=0, description="Expected CO2 emissions (kg)")

    can_shift: bool = Field(default=False, description="Whether operation can be time-shifted")
    shift_window_minutes: int = Field(default=0, ge=0, description="Allowable shift window")


class StorageDispatchPlan(BaseModel):
    """Thermal storage dispatch plan for a time period."""

    timestamp: datetime = Field(..., description="Plan timestamp")
    mode: StorageMode = Field(..., description="Operating mode")
    power_kw: float = Field(..., description="Charge (+) or discharge (-) power")
    expected_soc_percent: float = Field(..., ge=0, le=100, description="Expected SOC after operation")
    cost_saving: float = Field(default=0, description="Cost saving from storage operation ($)")


class ExplainabilityReport(BaseModel):
    """Explainability report for schedule decisions."""

    decision_id: str = Field(..., description="Decision identifier")
    decision_type: str = Field(..., description="Type of decision (schedule, storage, shift)")

    # Top factors influencing decision
    key_factors: Dict[str, float] = Field(
        ...,
        description="Key factors and their importance (0-1)"
    )

    # Natural language explanation
    explanation: str = Field(..., description="Human-readable explanation")

    # Counterfactual analysis
    what_if_scenarios: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="What-if scenario analyses"
    )

    # Model confidence
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence in decision")


# =============================================================================
# Main Input/Output Models
# =============================================================================

class SchedulerInput(BaseModel):
    """Input data model for ProcessHeatingSchedulerAgent."""

    # Request metadata
    request_id: str = Field(..., min_length=1, description="Unique request identifier")
    facility_id: str = Field(..., description="Facility identifier")
    planning_horizon_hours: int = Field(
        default=24, ge=1, le=168,
        description="Planning horizon (hours)"
    )
    schedule_resolution_minutes: int = Field(
        default=15, ge=5, le=60,
        description="Schedule time resolution (minutes)"
    )

    # Optimization preferences
    optimization_priority: SchedulePriority = Field(
        default=SchedulePriority.BALANCED,
        description="Optimization priority"
    )
    cost_weight: float = Field(default=0.4, ge=0, le=1, description="Cost optimization weight")
    emissions_weight: float = Field(default=0.3, ge=0, le=1, description="Emissions weight")
    reliability_weight: float = Field(default=0.3, ge=0, le=1, description="Reliability weight")

    # Production requirements
    production_orders: List[ProductionOrder] = Field(
        default_factory=list,
        description="Production orders requiring heat"
    )

    # Energy tariff
    energy_tariff: EnergyTariff = Field(..., description="Current energy tariff structure")

    # Equipment
    equipment_list: List[Equipment] = Field(
        ...,
        min_items=1,
        description="Available heating equipment"
    )

    # Thermal storage (optional)
    thermal_storage: Optional[ThermalStorage] = Field(
        None,
        description="Thermal storage system if available"
    )

    # Weather forecast
    weather_forecast: List[WeatherForecast] = Field(
        default_factory=list,
        description="Weather forecast for planning horizon"
    )

    # Historical demand data
    historical_demand: List[HistoricalDemand] = Field(
        default_factory=list,
        description="Historical demand data for forecasting"
    )

    # Grid signals
    grid_signals: List[GridSignal] = Field(
        default_factory=list,
        description="Grid demand response signals"
    )

    # Constraints
    max_demand_kw: Optional[float] = Field(
        None, gt=0,
        description="Maximum allowed demand (kW)"
    )
    min_reliability_percent: float = Field(
        default=95, ge=0, le=100,
        description="Minimum required reliability (%)"
    )

    @root_validator(skip_on_failure=True)
    def validate_weights(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate optimization weights sum to approximately 1."""
        cost = values.get('cost_weight', 0)
        emissions = values.get('emissions_weight', 0)
        reliability = values.get('reliability_weight', 0)
        total = cost + emissions + reliability

        if abs(total - 1.0) > 0.01:
            logger.warning(f"Optimization weights sum to {total}, normalizing")
            if total > 0:
                values['cost_weight'] = cost / total
                values['emissions_weight'] = emissions / total
                values['reliability_weight'] = reliability / total

        return values


class SchedulerOutput(BaseModel):
    """Output data model for ProcessHeatingSchedulerAgent."""

    # Request tracking
    request_id: str = Field(..., description="Original request identifier")
    schedule_timestamp: datetime = Field(..., description="When schedule was generated")

    # Optimized schedule
    scheduled_operations: List[ScheduledOperation] = Field(
        ...,
        description="List of scheduled heating operations"
    )

    # Storage dispatch plan
    storage_dispatch: List[StorageDispatchPlan] = Field(
        default_factory=list,
        description="Thermal storage dispatch plan"
    )

    # Demand predictions
    demand_predictions: List[DemandPrediction] = Field(
        ...,
        description="Demand predictions with uncertainty"
    )

    # Cost analysis
    total_expected_cost: float = Field(..., ge=0, description="Total expected energy cost ($)")
    baseline_cost: float = Field(..., ge=0, description="Baseline cost without optimization ($)")
    cost_savings: float = Field(..., ge=0, description="Expected cost savings ($)")
    cost_savings_percent: float = Field(..., ge=0, le=100, description="Cost savings (%)")

    # Emissions analysis
    total_expected_emissions_kg: float = Field(..., ge=0, description="Total expected CO2 (kg)")
    baseline_emissions_kg: float = Field(..., ge=0, description="Baseline CO2 without optimization (kg)")
    emissions_reduction_kg: float = Field(..., ge=0, description="Emissions reduction (kg)")

    # Peak demand management
    peak_demand_kw: float = Field(..., ge=0, description="Scheduled peak demand (kW)")
    peak_reduction_kw: float = Field(default=0, ge=0, description="Peak reduction vs baseline (kW)")

    # Demand response
    dr_events_scheduled: int = Field(default=0, ge=0, description="Number of DR events scheduled")
    dr_capacity_available_kw: float = Field(default=0, ge=0, description="DR capacity available (kW)")

    # Explainability reports
    explainability_reports: List[ExplainabilityReport] = Field(
        default_factory=list,
        description="Explainability reports for key decisions"
    )

    # Schedule quality metrics
    schedule_feasibility: float = Field(..., ge=0, le=1, description="Schedule feasibility score")
    schedule_robustness: float = Field(..., ge=0, le=1, description="Schedule robustness to uncertainty")

    # Provenance and audit
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., ge=0, description="Processing time (ms)")
    validation_status: str = Field(
        ...,
        pattern="^(PASS|FAIL)$",
        description="PASS or FAIL"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation error messages"
    )

    # Model versions
    forecasting_model_version: str = Field(default="1.0.0", description="Forecasting model version")
    optimization_solver_version: str = Field(default="1.0.0", description="Optimization solver version")


class SSEScheduleUpdate(BaseModel):
    """Server-Sent Event (SSE) schedule update message."""

    event_type: str = Field(..., description="Event type (update, alert, complete)")
    request_id: str = Field(..., description="Associated request ID")
    timestamp: datetime = Field(..., description="Event timestamp")

    # Update content (depends on event_type)
    updated_operations: List[ScheduledOperation] = Field(
        default_factory=list,
        description="Updated operations"
    )

    # Alert information
    alert_message: Optional[str] = Field(None, description="Alert message if applicable")
    alert_severity: Optional[str] = Field(None, description="Alert severity (info, warning, critical)")

    # Progress
    progress_percent: float = Field(default=0, ge=0, le=100, description="Optimization progress (%)")

    # Sequence number for ordering
    sequence_number: int = Field(..., ge=0, description="Event sequence number")


class AgentConfig(BaseModel):
    """Configuration for ProcessHeatingSchedulerAgent."""

    agent_id: str = Field(default="GL-019", description="Agent identifier")
    agent_name: str = Field(default="HEATSCHEDULER", description="Agent name")
    version: str = Field(default="1.0.0", description="Agent version")

    # Forecasting settings
    forecast_model: str = Field(default="gradient_boosting", description="Forecasting model type")
    forecast_horizon_hours: int = Field(default=48, ge=1, le=168, description="Forecast horizon")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level for bounds")

    # Optimization settings
    solver_type: str = Field(default="milp", description="Optimization solver type")
    solver_time_limit_seconds: int = Field(default=60, ge=10, le=600, description="Solver time limit")
    optimality_gap: float = Field(default=0.01, ge=0, le=0.1, description="Acceptable optimality gap")

    # Storage optimization
    storage_optimization_enabled: bool = Field(default=True, description="Enable storage optimization")

    # SSE streaming
    sse_enabled: bool = Field(default=True, description="Enable SSE streaming updates")
    sse_update_interval_seconds: int = Field(default=5, ge=1, le=60, description="SSE update interval")

    # Explainability
    generate_explanations: bool = Field(default=True, description="Generate explainability reports")
    shap_enabled: bool = Field(default=True, description="Enable SHAP explanations")
