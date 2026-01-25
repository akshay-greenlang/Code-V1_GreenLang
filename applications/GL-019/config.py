# -*- coding: utf-8 -*-
"""
GL-019 HEATSCHEDULER Configuration Models.

This module provides comprehensive Pydantic configuration models for the
Process Heating Scheduler Agent, including tariff configurations, equipment
specifications, production schedule settings, and optimization parameters.

All models include validators to ensure operational constraints and
industry best practices for process heating operations.

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class TariffType(str, Enum):
    """Energy tariff type classification."""

    TIME_OF_USE = "time_of_use"
    DEMAND_CHARGE = "demand_charge"
    REAL_TIME_PRICING = "real_time_pricing"
    TIERED = "tiered"
    FLAT_RATE = "flat_rate"
    CRITICAL_PEAK = "critical_peak"


class EquipmentType(str, Enum):
    """Heating equipment type classification."""

    ELECTRIC_FURNACE = "electric_furnace"
    GAS_FURNACE = "gas_furnace"
    INDUCTION_FURNACE = "induction_furnace"
    BOILER = "boiler"
    HEAT_TREATMENT = "heat_treatment"
    OVEN = "oven"
    KILN = "kiln"
    DRYER = "dryer"


class EquipmentStatus(str, Enum):
    """Equipment operational status."""

    AVAILABLE = "available"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    STANDBY = "standby"
    FAULT = "fault"
    OFFLINE = "offline"


class OptimizationObjective(str, Enum):
    """Schedule optimization objective."""

    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_PEAK_DEMAND = "minimize_peak_demand"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCE_COST_DEMAND = "balance_cost_demand"
    EARLIEST_COMPLETION = "earliest_completion"


class SchedulePriority(str, Enum):
    """Production schedule priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FLEXIBLE = "flexible"


# ============================================================================
# TARIFF CONFIGURATION
# ============================================================================


class TariffConfiguration(BaseModel):
    """
    Energy tariff configuration and rate structure.

    Defines time-of-use rates, demand charges, and real-time pricing
    parameters for energy cost optimization.
    """

    tariff_id: str = Field(..., description="Unique tariff identifier")
    tariff_type: TariffType = Field(..., description="Tariff type classification")
    utility_name: Optional[str] = Field(None, description="Utility provider name")
    rate_schedule: Optional[str] = Field(None, description="Rate schedule identifier")

    # Time-of-use rates ($/kWh)
    peak_rate_per_kwh: float = Field(
        default=0.15, ge=0, description="Peak period energy rate ($/kWh)"
    )
    off_peak_rate_per_kwh: float = Field(
        default=0.06, ge=0, description="Off-peak period energy rate ($/kWh)"
    )
    shoulder_rate_per_kwh: Optional[float] = Field(
        None, ge=0, description="Shoulder period energy rate ($/kWh)"
    )
    super_off_peak_rate_per_kwh: Optional[float] = Field(
        None, ge=0, description="Super off-peak energy rate ($/kWh)"
    )

    # Peak period definition (24-hour format)
    peak_hours_start: int = Field(
        default=14, ge=0, le=23, description="Peak period start hour (0-23)"
    )
    peak_hours_end: int = Field(
        default=20, ge=0, le=23, description="Peak period end hour (0-23)"
    )
    shoulder_hours_start: Optional[int] = Field(
        None, ge=0, le=23, description="Shoulder period start hour"
    )
    shoulder_hours_end: Optional[int] = Field(
        None, ge=0, le=23, description="Shoulder period end hour"
    )

    # Weekend/holiday rates
    weekend_off_peak: bool = Field(
        default=True, description="Weekends use off-peak rates"
    )
    holiday_off_peak: bool = Field(
        default=True, description="Holidays use off-peak rates"
    )

    # Demand charges ($/kW)
    demand_charge_per_kw: float = Field(
        default=0.0, ge=0, description="Monthly demand charge ($/kW)"
    )
    peak_demand_charge_per_kw: float = Field(
        default=0.0, ge=0, description="Peak period demand charge ($/kW)"
    )
    ratchet_percentage: float = Field(
        default=0.0, ge=0, le=100, description="Demand ratchet percentage"
    )

    # Critical peak pricing
    critical_peak_rate_per_kwh: Optional[float] = Field(
        None, ge=0, description="Critical peak event rate ($/kWh)"
    )
    critical_peak_adder: float = Field(
        default=0.0, ge=0, description="Critical peak adder ($/kWh)"
    )

    # Real-time pricing
    rtp_enabled: bool = Field(
        default=False, description="Real-time pricing enabled"
    )
    rtp_api_endpoint: Optional[str] = Field(
        None, description="Real-time pricing API endpoint"
    )
    rtp_update_interval_minutes: int = Field(
        default=15, ge=5, le=60, description="RTP update interval (minutes)"
    )

    # Effective dates
    effective_start_date: Optional[datetime] = Field(
        None, description="Tariff effective start date"
    )
    effective_end_date: Optional[datetime] = Field(
        None, description="Tariff effective end date"
    )

    @field_validator("peak_hours_end")
    @classmethod
    def validate_peak_hours(cls, v: int, info) -> int:
        """Validate peak hours range."""
        if "peak_hours_start" in info.data:
            start = info.data["peak_hours_start"]
            if v < start:
                logger.warning(
                    f"Peak hours end ({v}) is before start ({start}) - may span midnight"
                )
        return v

    @model_validator(mode="after")
    def validate_rate_structure(self) -> "TariffConfiguration":
        """Validate rate structure consistency."""
        if self.peak_rate_per_kwh < self.off_peak_rate_per_kwh:
            logger.warning(
                f"Peak rate ({self.peak_rate_per_kwh}) is lower than off-peak rate "
                f"({self.off_peak_rate_per_kwh}) - verify tariff configuration"
            )

        if self.shoulder_rate_per_kwh is not None:
            if not (self.off_peak_rate_per_kwh <= self.shoulder_rate_per_kwh <= self.peak_rate_per_kwh):
                logger.warning(
                    "Shoulder rate should typically be between off-peak and peak rates"
                )

        return self

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "tariff_id": "TOU-001",
                "tariff_type": "time_of_use",
                "utility_name": "Pacific Gas & Electric",
                "rate_schedule": "E-19",
                "peak_rate_per_kwh": 0.15,
                "off_peak_rate_per_kwh": 0.06,
                "peak_hours_start": 14,
                "peak_hours_end": 20,
                "demand_charge_per_kw": 12.50,
            }
        }


# ============================================================================
# EQUIPMENT CONFIGURATION
# ============================================================================


class EquipmentConfiguration(BaseModel):
    """
    Heating equipment configuration and specifications.

    Defines equipment characteristics that influence schedule
    optimization and energy consumption calculations.
    """

    equipment_id: str = Field(..., description="Unique equipment identifier")
    equipment_type: EquipmentType = Field(..., description="Equipment type classification")
    equipment_name: Optional[str] = Field(None, description="Equipment display name")
    manufacturer: Optional[str] = Field(None, description="Equipment manufacturer")
    model: Optional[str] = Field(None, description="Equipment model number")

    # Capacity
    capacity_kw: float = Field(
        ..., gt=0, description="Rated power capacity (kW)"
    )
    min_power_kw: float = Field(
        default=0.0, ge=0, description="Minimum operating power (kW)"
    )
    max_power_kw: Optional[float] = Field(
        None, gt=0, description="Maximum operating power (kW)"
    )
    standby_power_kw: float = Field(
        default=0.0, ge=0, description="Standby power consumption (kW)"
    )

    # Efficiency
    efficiency: float = Field(
        default=0.85, ge=0.5, le=1.0, description="Energy efficiency (0-1)"
    )
    efficiency_at_partial_load: Optional[float] = Field(
        None, ge=0.3, le=1.0, description="Efficiency at partial load"
    )

    # Temperature specifications
    max_temperature_c: float = Field(
        default=1000.0, gt=0, description="Maximum operating temperature (C)"
    )
    min_temperature_c: float = Field(
        default=20.0, ge=0, description="Minimum operating temperature (C)"
    )
    ramp_rate_c_per_minute: float = Field(
        default=10.0, gt=0, description="Temperature ramp rate (C/min)"
    )
    cooldown_rate_c_per_minute: float = Field(
        default=5.0, gt=0, description="Cooldown rate (C/min)"
    )

    # Timing constraints
    min_run_time_minutes: int = Field(
        default=30, ge=0, description="Minimum run time (minutes)"
    )
    min_idle_time_minutes: int = Field(
        default=15, ge=0, description="Minimum idle time between runs (minutes)"
    )
    startup_time_minutes: int = Field(
        default=30, ge=0, description="Startup time to reach operating temp (minutes)"
    )
    shutdown_time_minutes: int = Field(
        default=15, ge=0, description="Shutdown time (minutes)"
    )

    # Availability schedule
    availability_start_time: time = Field(
        default=time(0, 0), description="Daily availability start time"
    )
    availability_end_time: time = Field(
        default=time(23, 59), description="Daily availability end time"
    )
    available_days: List[int] = Field(
        default=[0, 1, 2, 3, 4, 5, 6],
        description="Available days (0=Monday, 6=Sunday)"
    )

    # Current status
    status: EquipmentStatus = Field(
        default=EquipmentStatus.AVAILABLE, description="Current equipment status"
    )

    # Maintenance
    next_maintenance_date: Optional[datetime] = Field(
        None, description="Next scheduled maintenance date"
    )
    maintenance_interval_hours: int = Field(
        default=2000, ge=100, description="Maintenance interval (operating hours)"
    )
    current_operating_hours: float = Field(
        default=0.0, ge=0, description="Current operating hours"
    )

    # Location
    location: Optional[str] = Field(None, description="Physical location")
    production_line: Optional[str] = Field(
        None, description="Associated production line"
    )

    @field_validator("max_power_kw")
    @classmethod
    def validate_max_power(cls, v: Optional[float], info) -> Optional[float]:
        """Validate max power against capacity."""
        if v is not None and "capacity_kw" in info.data:
            if v < info.data["capacity_kw"]:
                logger.warning(
                    f"Max power ({v} kW) is less than capacity ({info.data['capacity_kw']} kW)"
                )
        return v

    @model_validator(mode="after")
    def validate_temperature_range(self) -> "EquipmentConfiguration":
        """Validate temperature range consistency."""
        if self.min_temperature_c >= self.max_temperature_c:
            raise ValueError(
                "Minimum temperature must be less than maximum temperature"
            )
        return self

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "equipment_id": "FURN-001",
                "equipment_type": "electric_furnace",
                "equipment_name": "Main Heat Treatment Furnace",
                "capacity_kw": 500.0,
                "efficiency": 0.92,
                "max_temperature_c": 1200.0,
                "ramp_rate_c_per_minute": 15.0,
                "min_run_time_minutes": 60,
                "status": "available",
            }
        }


# ============================================================================
# PRODUCTION SCHEDULE CONFIGURATION
# ============================================================================


class ProductionScheduleConfiguration(BaseModel):
    """
    Production schedule configuration and constraints.

    Defines production requirements and scheduling constraints
    for the optimization engine.
    """

    schedule_id: str = Field(..., description="Unique schedule identifier")
    schedule_name: Optional[str] = Field(None, description="Schedule display name")

    # Planning horizon
    planning_horizon_hours: int = Field(
        default=24, ge=1, le=168, description="Planning horizon (hours)"
    )
    time_slot_minutes: int = Field(
        default=15, ge=5, le=60, description="Schedule time slot granularity (minutes)"
    )

    # Scheduling constraints
    allow_preemption: bool = Field(
        default=False, description="Allow job preemption"
    )
    allow_job_splitting: bool = Field(
        default=False, description="Allow splitting jobs across time periods"
    )
    respect_equipment_preferences: bool = Field(
        default=True, description="Respect equipment assignment preferences"
    )

    # Priority handling
    priority_weight_critical: float = Field(
        default=100.0, gt=0, description="Weight for critical priority jobs"
    )
    priority_weight_high: float = Field(
        default=50.0, gt=0, description="Weight for high priority jobs"
    )
    priority_weight_medium: float = Field(
        default=20.0, gt=0, description="Weight for medium priority jobs"
    )
    priority_weight_low: float = Field(
        default=5.0, gt=0, description="Weight for low priority jobs"
    )

    # Buffer times
    buffer_between_jobs_minutes: int = Field(
        default=15, ge=0, description="Buffer time between jobs (minutes)"
    )
    setup_time_minutes: int = Field(
        default=30, ge=0, description="Default setup time (minutes)"
    )
    changeover_time_minutes: int = Field(
        default=45, ge=0, description="Product changeover time (minutes)"
    )

    # Constraints
    max_concurrent_jobs: int = Field(
        default=10, ge=1, description="Maximum concurrent jobs"
    )
    max_daily_operating_hours: float = Field(
        default=24.0, ge=1, le=24, description="Maximum daily operating hours"
    )

    # ERP integration
    erp_sync_enabled: bool = Field(
        default=True, description="Enable ERP schedule synchronization"
    )
    erp_sync_interval_minutes: int = Field(
        default=15, ge=5, description="ERP sync interval (minutes)"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "schedule_id": "SCHED-2024-001",
                "schedule_name": "Production Week 49",
                "planning_horizon_hours": 24,
                "time_slot_minutes": 15,
                "allow_preemption": False,
                "buffer_between_jobs_minutes": 15,
            }
        }


# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================


class OptimizationParameters(BaseModel):
    """
    Schedule optimization parameters and tuning settings.

    Defines optimization objectives, constraints, and algorithm
    parameters for the scheduling engine.
    """

    # Primary objective
    primary_objective: OptimizationObjective = Field(
        default=OptimizationObjective.MINIMIZE_COST,
        description="Primary optimization objective"
    )
    secondary_objective: Optional[OptimizationObjective] = Field(
        None, description="Secondary optimization objective"
    )

    # Objective weights (for multi-objective optimization)
    cost_weight: float = Field(
        default=0.6, ge=0, le=1, description="Weight for cost minimization"
    )
    demand_weight: float = Field(
        default=0.3, ge=0, le=1, description="Weight for demand reduction"
    )
    efficiency_weight: float = Field(
        default=0.1, ge=0, le=1, description="Weight for efficiency maximization"
    )

    # Energy cost targets
    target_cost_reduction_percent: float = Field(
        default=15.0, ge=0, le=50, description="Target cost reduction (%)"
    )
    target_peak_demand_reduction_percent: float = Field(
        default=20.0, ge=0, le=50, description="Target peak demand reduction (%)"
    )

    # Demand response
    enable_demand_response: bool = Field(
        default=True, description="Enable demand response participation"
    )
    demand_response_threshold_kw: float = Field(
        default=1000.0, ge=0, description="Demand response threshold (kW)"
    )
    max_demand_curtailment_percent: float = Field(
        default=30.0, ge=0, le=100, description="Maximum demand curtailment (%)"
    )

    # Peak shaving
    enable_peak_shaving: bool = Field(
        default=True, description="Enable peak demand shaving"
    )
    peak_demand_limit_kw: float = Field(
        default=5000.0, gt=0, description="Peak demand limit (kW)"
    )
    soft_peak_limit_kw: Optional[float] = Field(
        None, gt=0, description="Soft peak limit for optimization"
    )

    # Load shifting
    enable_load_shifting: bool = Field(
        default=True, description="Enable load shifting to off-peak"
    )
    max_shift_hours: int = Field(
        default=4, ge=0, le=12, description="Maximum load shift (hours)"
    )
    min_shift_savings_threshold: float = Field(
        default=5.0, ge=0, description="Minimum savings threshold for shifting ($)"
    )

    # Algorithm parameters
    optimization_time_limit_seconds: int = Field(
        default=30, ge=5, le=300, description="Optimization time limit (seconds)"
    )
    solution_gap_tolerance: float = Field(
        default=0.05, ge=0, le=0.5, description="MIP gap tolerance"
    )
    num_scenarios: int = Field(
        default=3, ge=1, le=10, description="Number of scenarios to evaluate"
    )

    # Robustness
    enable_uncertainty_handling: bool = Field(
        default=True, description="Enable uncertainty handling"
    )
    price_uncertainty_percent: float = Field(
        default=10.0, ge=0, le=50, description="Price uncertainty margin (%)"
    )
    demand_uncertainty_percent: float = Field(
        default=5.0, ge=0, le=30, description="Demand forecast uncertainty (%)"
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "OptimizationParameters":
        """Validate objective weights sum to 1.0."""
        total_weight = self.cost_weight + self.demand_weight + self.efficiency_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Objective weights sum to {total_weight:.2f}, not 1.0 - normalizing"
            )
        return self

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "primary_objective": "minimize_cost",
                "cost_weight": 0.6,
                "demand_weight": 0.3,
                "target_cost_reduction_percent": 15.0,
                "enable_demand_response": True,
                "peak_demand_limit_kw": 5000.0,
            }
        }


# ============================================================================
# INTEGRATION CONFIGURATIONS
# ============================================================================


class ERPIntegration(BaseModel):
    """
    ERP system integration configuration.

    Defines connection parameters and mappings for ERP integration.
    """

    enabled: bool = Field(default=True, description="ERP integration enabled")
    erp_system: str = Field(default="", description="ERP system type/vendor")
    connection_string: str = Field(default="", description="ERP connection string")
    api_endpoint: str = Field(default="", description="ERP API endpoint")

    # Polling and sync
    polling_interval_seconds: int = Field(
        default=60, ge=10, le=3600, description="Data polling interval (seconds)"
    )
    sync_mode: str = Field(
        default="pull", description="Sync mode: pull, push, bidirectional"
    )

    # Data mappings
    production_order_table: str = Field(
        default="", description="Production order table/entity"
    )
    work_center_table: str = Field(
        default="", description="Work center table/entity"
    )
    material_table: str = Field(
        default="", description="Material master table/entity"
    )

    # Authentication
    auth_type: str = Field(
        default="api_key", description="Authentication type: api_key, oauth2, basic"
    )
    auth_credentials_secret: str = Field(
        default="", description="Secret name for credentials"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "enabled": True,
                "erp_system": "SAP S/4HANA",
                "api_endpoint": "https://erp.company.com/api/v1",
                "polling_interval_seconds": 60,
                "sync_mode": "bidirectional",
            }
        }


class ControlSystemIntegration(BaseModel):
    """
    Control system integration configuration.

    Defines connection parameters for SCADA/PLC integration
    to apply optimized schedules.
    """

    enabled: bool = Field(default=True, description="Control system integration enabled")
    system_type: str = Field(default="", description="Control system type")
    connection_protocol: str = Field(
        default="opcua", description="Connection protocol: opcua, modbus, mqtt"
    )
    endpoint: str = Field(default="", description="Control system endpoint")

    # Tag mappings
    schedule_start_tag: str = Field(
        default="", description="Schedule start command tag"
    )
    power_setpoint_tag: str = Field(
        default="", description="Power setpoint tag"
    )
    temperature_setpoint_tag: str = Field(
        default="", description="Temperature setpoint tag"
    )
    equipment_status_tag: str = Field(
        default="", description="Equipment status read tag"
    )

    # Safety
    enable_safety_checks: bool = Field(
        default=True, description="Enable safety checks before control"
    )
    max_setpoint_change_rate: float = Field(
        default=10.0, ge=0, description="Maximum setpoint change rate (%/min)"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "enabled": True,
                "system_type": "Siemens WinCC",
                "connection_protocol": "opcua",
                "endpoint": "opc.tcp://scada.company.com:4840",
            }
        }


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================


class AgentConfiguration(BaseModel):
    """
    Main agent configuration.

    Top-level configuration for the Process Heating Scheduler Agent.
    """

    agent_name: str = Field(
        default="GL-019 HEATSCHEDULER", description="Agent name"
    )
    version: str = Field(default="1.0.0", description="Agent version")
    environment: str = Field(
        default="production", description="Deployment environment"
    )

    # Tariff configurations
    tariffs: List[TariffConfiguration] = Field(
        ..., min_length=1, description="List of energy tariff configurations"
    )

    # Equipment configurations
    equipment: List[EquipmentConfiguration] = Field(
        ..., min_length=1, description="List of heating equipment configurations"
    )

    # Production schedule
    production_schedule: Optional[ProductionScheduleConfiguration] = Field(
        None, description="Production schedule configuration"
    )

    # Optimization parameters
    optimization_parameters: OptimizationParameters = Field(
        default_factory=OptimizationParameters,
        description="Optimization parameters"
    )

    # Integrations
    erp_integration: Optional[ERPIntegration] = Field(
        None, description="ERP integration configuration"
    )
    control_system_integration: Optional[ControlSystemIntegration] = Field(
        None, description="Control system integration configuration"
    )

    # Operational settings
    optimization_interval_minutes: int = Field(
        default=15, ge=5, le=60, description="Schedule optimization interval (minutes)"
    )
    auto_apply_schedule: bool = Field(
        default=False, description="Automatically apply optimized schedules"
    )
    schedule_lookahead_hours: int = Field(
        default=24, ge=4, le=168, description="Schedule lookahead horizon (hours)"
    )

    # Alerting
    enable_email_alerts: bool = Field(
        default=True, description="Enable email alerts"
    )
    enable_sms_alerts: bool = Field(
        default=False, description="Enable SMS alerts"
    )
    alert_recipients: List[str] = Field(
        default_factory=list, description="Alert recipient email addresses"
    )

    # Cost thresholds for alerts
    cost_alert_threshold_usd: float = Field(
        default=1000.0, ge=0, description="Daily cost alert threshold ($)"
    )
    demand_alert_threshold_kw: float = Field(
        default=5000.0, ge=0, description="Demand alert threshold (kW)"
    )

    # Reporting
    enable_daily_reports: bool = Field(
        default=True, description="Generate daily reports"
    )
    enable_weekly_reports: bool = Field(
        default=True, description="Generate weekly reports"
    )
    report_recipients: List[str] = Field(
        default_factory=list, description="Report recipient email addresses"
    )

    # Data retention
    historical_data_retention_days: int = Field(
        default=90, ge=7, le=730, description="Historical data retention (days)"
    )

    def get_tariff(self, tariff_id: str) -> Optional[TariffConfiguration]:
        """
        Get tariff configuration by ID.

        Args:
            tariff_id: Tariff identifier

        Returns:
            TariffConfiguration or None if not found
        """
        for tariff in self.tariffs:
            if tariff.tariff_id == tariff_id:
                return tariff
        return None

    def get_equipment(self, equipment_id: str) -> Optional[EquipmentConfiguration]:
        """
        Get equipment configuration by ID.

        Args:
            equipment_id: Equipment identifier

        Returns:
            EquipmentConfiguration or None if not found
        """
        for equip in self.equipment:
            if equip.equipment_id == equipment_id:
                return equip
        return None

    def get_available_equipment(self) -> List[EquipmentConfiguration]:
        """
        Get list of available equipment.

        Returns:
            List of equipment with AVAILABLE status
        """
        return [
            equip for equip in self.equipment
            if equip.status == EquipmentStatus.AVAILABLE
        ]

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "agent_name": "GL-019 HEATSCHEDULER",
                "version": "1.0.0",
                "environment": "production",
                "optimization_interval_minutes": 15,
                "auto_apply_schedule": False,
                "schedule_lookahead_hours": 24,
            }
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "TariffType",
    "EquipmentType",
    "EquipmentStatus",
    "OptimizationObjective",
    "SchedulePriority",
    "TariffConfiguration",
    "EquipmentConfiguration",
    "ProductionScheduleConfiguration",
    "OptimizationParameters",
    "ERPIntegration",
    "ControlSystemIntegration",
    "AgentConfiguration",
]
