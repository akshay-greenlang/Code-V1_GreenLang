"""
GL-019 HEATSCHEDULER Pydantic Schemas

Request and response models for the ProcessHeatingScheduler REST API.
Implements comprehensive validation rules and example values for documentation.

Author: GL-APIDeveloper
Version: 1.0.0
"""

from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator, EmailStr


# =============================================================================
# Enums
# =============================================================================

class ScheduleStatus(str, Enum):
    """Schedule execution status."""
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OptimizationObjective(str, Enum):
    """Optimization objective for scheduling."""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_EMISSIONS = "minimize_emissions"
    MINIMIZE_PEAK_DEMAND = "minimize_peak_demand"
    BALANCED = "balanced"


class EquipmentType(str, Enum):
    """Heating equipment types."""
    FURNACE = "furnace"
    BOILER = "boiler"
    HEAT_EXCHANGER = "heat_exchanger"
    ELECTRIC_HEATER = "electric_heater"
    STEAM_GENERATOR = "steam_generator"
    HEAT_PUMP = "heat_pump"


class EquipmentStatus(str, Enum):
    """Equipment operational status."""
    AVAILABLE = "available"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    FAULT = "fault"


class TariffType(str, Enum):
    """Energy tariff types."""
    FIXED = "fixed"
    TIME_OF_USE = "time_of_use"
    DEMAND_BASED = "demand_based"
    REAL_TIME_PRICING = "real_time_pricing"
    TIERED = "tiered"


class DemandResponseEventType(str, Enum):
    """Demand response event types."""
    CURTAILMENT = "curtailment"
    LOAD_SHIFT = "load_shift"
    EMERGENCY = "emergency"
    ECONOMIC = "economic"
    CAPACITY = "capacity"


class DemandResponseStatus(str, Enum):
    """Demand response participation status."""
    IDLE = "idle"
    PENDING = "pending"
    ACTIVE = "active"
    RESPONDING = "responding"
    COMPLETED = "completed"
    OPTED_OUT = "opted_out"


class BatchPriority(str, Enum):
    """Production batch priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    FLEXIBLE = "flexible"


class EnergyUnit(str, Enum):
    """Energy measurement units."""
    KWH = "kWh"
    MWH = "MWh"
    THERM = "therm"
    MMBTU = "MMBtu"
    GJ = "GJ"


# =============================================================================
# Base Models
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            time: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
            UUID: lambda v: str(v)
        }


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., ge=0, description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")


# =============================================================================
# Temperature and Energy Models
# =============================================================================

class TemperatureProfile(BaseModel):
    """Temperature profile for heating operations."""
    initial_temp: float = Field(..., description="Initial temperature in Celsius")
    target_temp: float = Field(..., description="Target temperature in Celsius")
    ramp_rate: float = Field(..., ge=0, description="Temperature ramp rate in C/minute")
    hold_duration_minutes: int = Field(..., ge=0, description="Hold duration at target temp")
    tolerance: float = Field(2.0, ge=0, description="Temperature tolerance in Celsius")

    class Config:
        schema_extra = {
            "example": {
                "initial_temp": 25.0,
                "target_temp": 850.0,
                "ramp_rate": 5.0,
                "hold_duration_minutes": 120,
                "tolerance": 2.0
            }
        }


class EnergyConsumption(BaseModel):
    """Energy consumption data."""
    value: float = Field(..., ge=0, description="Energy consumption value")
    unit: EnergyUnit = Field(EnergyUnit.KWH, description="Energy unit")
    peak_demand_kw: Optional[float] = Field(None, ge=0, description="Peak demand in kW")
    timestamp: Optional[datetime] = Field(None, description="Measurement timestamp")


# =============================================================================
# Schedule Models
# =============================================================================

class HeatingOperation(BaseModel):
    """Individual heating operation within a schedule."""
    operation_id: str = Field(..., description="Operation identifier")
    equipment_id: str = Field(..., description="Equipment to use")
    batch_id: Optional[str] = Field(None, description="Associated production batch")
    temperature_profile: TemperatureProfile = Field(..., description="Temperature profile")
    start_time: datetime = Field(..., description="Scheduled start time")
    end_time: datetime = Field(..., description="Scheduled end time")
    estimated_energy_kwh: float = Field(..., ge=0, description="Estimated energy consumption")
    estimated_cost: float = Field(..., ge=0, description="Estimated energy cost")
    priority: BatchPriority = Field(BatchPriority.NORMAL, description="Operation priority")

    class Config:
        schema_extra = {
            "example": {
                "operation_id": "op_001",
                "equipment_id": "furnace_01",
                "batch_id": "batch_20231109_001",
                "temperature_profile": {
                    "initial_temp": 25.0,
                    "target_temp": 850.0,
                    "ramp_rate": 5.0,
                    "hold_duration_minutes": 120,
                    "tolerance": 2.0
                },
                "start_time": "2025-11-09T02:00:00Z",
                "end_time": "2025-11-09T05:30:00Z",
                "estimated_energy_kwh": 450.5,
                "estimated_cost": 45.05,
                "priority": "normal"
            }
        }


class ScheduleOptimizeRequest(BaseModel):
    """Request to create an optimized heating schedule."""
    name: str = Field(..., min_length=1, max_length=255, description="Schedule name")
    description: Optional[str] = Field(None, max_length=1000, description="Schedule description")
    start_date: date = Field(..., description="Schedule start date")
    end_date: date = Field(..., description="Schedule end date")
    facility_id: str = Field(..., description="Facility identifier")
    batch_ids: List[str] = Field(..., min_items=1, description="Production batch IDs to schedule")
    objective: OptimizationObjective = Field(
        OptimizationObjective.MINIMIZE_COST,
        description="Optimization objective"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional scheduling constraints"
    )
    demand_response_enabled: bool = Field(
        True,
        description="Enable demand response participation"
    )
    max_peak_demand_kw: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum allowed peak demand in kW"
    )

    @validator("end_date")
    def end_date_after_start(cls, v, values):
        if "start_date" in values and v < values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v

    class Config:
        schema_extra = {
            "example": {
                "name": "Weekly Production Schedule",
                "description": "Optimized heating schedule for Week 45",
                "start_date": "2025-11-09",
                "end_date": "2025-11-15",
                "facility_id": "facility_01",
                "batch_ids": ["batch_001", "batch_002", "batch_003"],
                "objective": "minimize_cost",
                "constraints": {
                    "avoid_peak_hours": True,
                    "min_equipment_utilization": 0.7
                },
                "demand_response_enabled": True,
                "max_peak_demand_kw": 500.0
            }
        }


class ScheduleResponse(BaseModel):
    """Heating schedule response."""
    schedule_id: str = Field(..., description="Unique schedule identifier")
    name: str = Field(..., description="Schedule name")
    description: Optional[str] = Field(None, description="Schedule description")
    status: ScheduleStatus = Field(..., description="Current schedule status")
    start_date: date = Field(..., description="Schedule start date")
    end_date: date = Field(..., description="Schedule end date")
    facility_id: str = Field(..., description="Facility identifier")
    objective: OptimizationObjective = Field(..., description="Optimization objective used")
    operations: List[HeatingOperation] = Field(..., description="Scheduled heating operations")
    total_energy_kwh: float = Field(..., ge=0, description="Total estimated energy")
    total_cost: float = Field(..., ge=0, description="Total estimated cost")
    baseline_cost: float = Field(..., ge=0, description="Baseline cost without optimization")
    savings: float = Field(..., ge=0, description="Estimated cost savings")
    savings_percent: float = Field(..., ge=0, le=100, description="Savings percentage")
    peak_demand_kw: float = Field(..., ge=0, description="Peak demand in kW")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        schema_extra = {
            "example": {
                "schedule_id": "sched_abc123",
                "name": "Weekly Production Schedule",
                "description": "Optimized heating schedule for Week 45",
                "status": "approved",
                "start_date": "2025-11-09",
                "end_date": "2025-11-15",
                "facility_id": "facility_01",
                "objective": "minimize_cost",
                "operations": [],
                "total_energy_kwh": 15000.0,
                "total_cost": 1200.00,
                "baseline_cost": 1500.00,
                "savings": 300.00,
                "savings_percent": 20.0,
                "peak_demand_kw": 450.0,
                "created_at": "2025-11-09T10:00:00Z",
                "updated_at": "2025-11-09T10:00:00Z"
            }
        }


class ScheduleUpdateRequest(BaseModel):
    """Request to update a schedule."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Schedule name")
    description: Optional[str] = Field(None, max_length=1000, description="Schedule description")
    status: Optional[ScheduleStatus] = Field(None, description="New schedule status")
    operations: Optional[List[HeatingOperation]] = Field(None, description="Updated operations")


class ScheduleListParams(BaseModel):
    """Parameters for listing schedules."""
    facility_id: Optional[str] = Field(None, description="Filter by facility")
    status: Optional[ScheduleStatus] = Field(None, description="Filter by status")
    start_date_from: Optional[date] = Field(None, description="Filter by start date from")
    start_date_to: Optional[date] = Field(None, description="Filter by start date to")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")


class ScheduleListResponse(BaseModel):
    """Paginated list of schedules."""
    items: List[ScheduleResponse] = Field(..., description="List of schedules")
    total: int = Field(..., ge=0, description="Total number of schedules")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total pages")


# =============================================================================
# Production Models
# =============================================================================

class ProductionBatch(BaseModel):
    """Production batch data."""
    batch_id: str = Field(..., description="Unique batch identifier")
    product_id: str = Field(..., description="Product identifier")
    product_name: str = Field(..., description="Product name")
    quantity: int = Field(..., ge=1, description="Batch quantity")
    priority: BatchPriority = Field(..., description="Batch priority")
    required_temp: float = Field(..., description="Required processing temperature")
    hold_duration_minutes: int = Field(..., ge=0, description="Required hold duration")
    earliest_start: datetime = Field(..., description="Earliest possible start time")
    latest_end: datetime = Field(..., description="Latest acceptable end time")
    equipment_types: List[EquipmentType] = Field(..., description="Compatible equipment types")
    estimated_energy_kwh: float = Field(..., ge=0, description="Estimated energy consumption")
    status: str = Field(..., description="Batch status")

    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_20231109_001",
                "product_id": "prod_steel_alloy",
                "product_name": "Steel Alloy A36",
                "quantity": 500,
                "priority": "high",
                "required_temp": 850.0,
                "hold_duration_minutes": 120,
                "earliest_start": "2025-11-09T06:00:00Z",
                "latest_end": "2025-11-10T18:00:00Z",
                "equipment_types": ["furnace"],
                "estimated_energy_kwh": 450.0,
                "status": "pending"
            }
        }


class ProductionBatchListResponse(BaseModel):
    """Response containing list of production batches."""
    items: List[ProductionBatch] = Field(..., description="List of production batches")
    total: int = Field(..., ge=0, description="Total number of batches")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, description="Items per page")


class ProductionSyncRequest(BaseModel):
    """Request to sync production schedule from ERP."""
    erp_system: str = Field(..., description="ERP system identifier")
    facility_id: str = Field(..., description="Facility identifier")
    date_from: date = Field(..., description="Sync data from this date")
    date_to: date = Field(..., description="Sync data to this date")
    sync_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional sync options"
    )

    class Config:
        schema_extra = {
            "example": {
                "erp_system": "sap_s4hana",
                "facility_id": "facility_01",
                "date_from": "2025-11-09",
                "date_to": "2025-11-15",
                "sync_options": {
                    "include_pending": True,
                    "auto_assign_equipment": True
                }
            }
        }


class ProductionSyncResponse(BaseModel):
    """Response from production sync operation."""
    sync_id: str = Field(..., description="Sync operation identifier")
    status: str = Field(..., description="Sync status")
    batches_synced: int = Field(..., ge=0, description="Number of batches synced")
    batches_created: int = Field(..., ge=0, description="Number of new batches created")
    batches_updated: int = Field(..., ge=0, description="Number of batches updated")
    errors: List[str] = Field(default_factory=list, description="Sync errors if any")
    synced_at: datetime = Field(..., description="Sync completion timestamp")


# =============================================================================
# Tariff Models
# =============================================================================

class TariffPeriod(BaseModel):
    """Tariff rate for a specific time period."""
    start_time: time = Field(..., description="Period start time")
    end_time: time = Field(..., description="Period end time")
    rate_per_kwh: float = Field(..., ge=0, description="Rate per kWh")
    demand_charge_per_kw: Optional[float] = Field(None, ge=0, description="Demand charge per kW")
    period_name: Optional[str] = Field(None, description="Period name (e.g., 'Off-Peak')")

    class Config:
        schema_extra = {
            "example": {
                "start_time": "00:00:00",
                "end_time": "06:00:00",
                "rate_per_kwh": 0.08,
                "demand_charge_per_kw": 5.00,
                "period_name": "Off-Peak"
            }
        }


class TariffResponse(BaseModel):
    """Energy tariff information."""
    tariff_id: str = Field(..., description="Tariff identifier")
    name: str = Field(..., description="Tariff name")
    tariff_type: TariffType = Field(..., description="Tariff type")
    utility_provider: str = Field(..., description="Utility provider name")
    currency: str = Field("USD", description="Currency code")
    effective_date: date = Field(..., description="Tariff effective date")
    expiration_date: Optional[date] = Field(None, description="Tariff expiration date")
    periods: List[TariffPeriod] = Field(..., description="Tariff periods")
    demand_charge_per_kw: Optional[float] = Field(None, ge=0, description="Monthly demand charge")

    class Config:
        schema_extra = {
            "example": {
                "tariff_id": "tariff_tou_2024",
                "name": "Time-of-Use Industrial Rate",
                "tariff_type": "time_of_use",
                "utility_provider": "Pacific Gas & Electric",
                "currency": "USD",
                "effective_date": "2024-01-01",
                "expiration_date": "2024-12-31",
                "periods": [
                    {
                        "start_time": "00:00:00",
                        "end_time": "06:00:00",
                        "rate_per_kwh": 0.08,
                        "period_name": "Off-Peak"
                    },
                    {
                        "start_time": "06:00:00",
                        "end_time": "14:00:00",
                        "rate_per_kwh": 0.12,
                        "period_name": "Mid-Peak"
                    },
                    {
                        "start_time": "14:00:00",
                        "end_time": "20:00:00",
                        "rate_per_kwh": 0.25,
                        "period_name": "On-Peak"
                    },
                    {
                        "start_time": "20:00:00",
                        "end_time": "00:00:00",
                        "rate_per_kwh": 0.12,
                        "period_name": "Mid-Peak"
                    }
                ],
                "demand_charge_per_kw": 15.00
            }
        }


class TariffForecastPoint(BaseModel):
    """Single point in tariff forecast."""
    timestamp: datetime = Field(..., description="Forecast timestamp")
    rate_per_kwh: float = Field(..., ge=0, description="Forecasted rate")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level 0-1")


class TariffForecastResponse(BaseModel):
    """Tariff rate forecast."""
    facility_id: str = Field(..., description="Facility identifier")
    forecast_generated_at: datetime = Field(..., description="When forecast was generated")
    forecast_horizon_hours: int = Field(..., ge=1, description="Forecast horizon in hours")
    currency: str = Field("USD", description="Currency code")
    forecast: List[TariffForecastPoint] = Field(..., description="Forecast data points")
    avg_rate: float = Field(..., ge=0, description="Average forecasted rate")
    min_rate: float = Field(..., ge=0, description="Minimum forecasted rate")
    max_rate: float = Field(..., ge=0, description="Maximum forecasted rate")


class TariffUploadRequest(BaseModel):
    """Request to upload custom tariff data."""
    name: str = Field(..., min_length=1, max_length=255, description="Tariff name")
    tariff_type: TariffType = Field(..., description="Tariff type")
    utility_provider: str = Field(..., description="Utility provider")
    facility_id: str = Field(..., description="Facility this tariff applies to")
    effective_date: date = Field(..., description="Effective date")
    expiration_date: Optional[date] = Field(None, description="Expiration date")
    periods: List[TariffPeriod] = Field(..., min_items=1, description="Tariff periods")
    demand_charge_per_kw: Optional[float] = Field(None, ge=0, description="Demand charge")

    class Config:
        schema_extra = {
            "example": {
                "name": "Custom Industrial Rate",
                "tariff_type": "time_of_use",
                "utility_provider": "Local Utility Co",
                "facility_id": "facility_01",
                "effective_date": "2024-01-01",
                "periods": [
                    {
                        "start_time": "00:00:00",
                        "end_time": "07:00:00",
                        "rate_per_kwh": 0.07,
                        "period_name": "Night"
                    },
                    {
                        "start_time": "07:00:00",
                        "end_time": "00:00:00",
                        "rate_per_kwh": 0.15,
                        "period_name": "Day"
                    }
                ],
                "demand_charge_per_kw": 12.00
            }
        }


# =============================================================================
# Equipment Models
# =============================================================================

class EquipmentSpecs(BaseModel):
    """Equipment technical specifications."""
    max_temp: float = Field(..., description="Maximum operating temperature")
    min_temp: float = Field(0.0, description="Minimum operating temperature")
    capacity_kg: Optional[float] = Field(None, ge=0, description="Capacity in kg")
    power_rating_kw: float = Field(..., ge=0, description="Power rating in kW")
    efficiency: float = Field(..., ge=0, le=1, description="Energy efficiency 0-1")
    ramp_rate_max: float = Field(..., ge=0, description="Max ramp rate C/min")


class Equipment(BaseModel):
    """Heating equipment data."""
    equipment_id: str = Field(..., description="Equipment identifier")
    name: str = Field(..., description="Equipment name")
    equipment_type: EquipmentType = Field(..., description="Equipment type")
    status: EquipmentStatus = Field(..., description="Current status")
    facility_id: str = Field(..., description="Facility location")
    specs: EquipmentSpecs = Field(..., description="Technical specifications")
    current_temp: Optional[float] = Field(None, description="Current temperature")
    current_power_kw: Optional[float] = Field(None, ge=0, description="Current power draw")
    last_maintenance: Optional[datetime] = Field(None, description="Last maintenance date")
    next_maintenance: Optional[datetime] = Field(None, description="Next scheduled maintenance")

    class Config:
        schema_extra = {
            "example": {
                "equipment_id": "furnace_01",
                "name": "Heat Treatment Furnace #1",
                "equipment_type": "furnace",
                "status": "available",
                "facility_id": "facility_01",
                "specs": {
                    "max_temp": 1200.0,
                    "min_temp": 0.0,
                    "capacity_kg": 5000.0,
                    "power_rating_kw": 250.0,
                    "efficiency": 0.85,
                    "ramp_rate_max": 10.0
                },
                "current_temp": 25.0,
                "current_power_kw": 0.0,
                "last_maintenance": "2025-10-15T08:00:00Z",
                "next_maintenance": "2025-12-15T08:00:00Z"
            }
        }


class EquipmentListResponse(BaseModel):
    """List of equipment."""
    items: List[Equipment] = Field(..., description="List of equipment")
    total: int = Field(..., ge=0, description="Total count")


class EquipmentAvailabilitySlot(BaseModel):
    """Equipment availability time slot."""
    start_time: datetime = Field(..., description="Slot start time")
    end_time: datetime = Field(..., description="Slot end time")
    is_available: bool = Field(..., description="Whether equipment is available")
    scheduled_batch_id: Optional[str] = Field(None, description="Scheduled batch if any")


class EquipmentAvailabilityResponse(BaseModel):
    """Equipment availability response."""
    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_name: str = Field(..., description="Equipment name")
    status: EquipmentStatus = Field(..., description="Current status")
    availability_slots: List[EquipmentAvailabilitySlot] = Field(
        ...,
        description="Availability time slots"
    )
    utilization_percent: float = Field(..., ge=0, le=100, description="Utilization percentage")


class EquipmentStatusUpdateRequest(BaseModel):
    """Request to update equipment status."""
    status: EquipmentStatus = Field(..., description="New equipment status")
    reason: Optional[str] = Field(None, max_length=500, description="Reason for status change")
    expected_duration_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Expected duration if temporary (e.g., maintenance)"
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "maintenance",
                "reason": "Scheduled preventive maintenance",
                "expected_duration_hours": 8.0
            }
        }


# =============================================================================
# Analytics Models
# =============================================================================

class SavingsBreakdown(BaseModel):
    """Breakdown of savings by category."""
    time_shifting: float = Field(..., description="Savings from shifting to off-peak")
    demand_reduction: float = Field(..., description="Savings from demand charge reduction")
    efficiency_improvement: float = Field(..., description="Savings from efficiency gains")
    demand_response: float = Field(..., description="Revenue from demand response")


class SavingsReportRequest(BaseModel):
    """Request for savings report."""
    facility_id: Optional[str] = Field(None, description="Filter by facility")
    start_date: date = Field(..., description="Report start date")
    end_date: date = Field(..., description="Report end date")
    include_breakdown: bool = Field(True, description="Include savings breakdown")


class SavingsReportResponse(BaseModel):
    """Savings analytics report."""
    report_id: str = Field(..., description="Report identifier")
    facility_id: Optional[str] = Field(None, description="Facility (if filtered)")
    period_start: date = Field(..., description="Report period start")
    period_end: date = Field(..., description="Report period end")
    total_energy_kwh: float = Field(..., ge=0, description="Total energy consumed")
    total_cost: float = Field(..., ge=0, description="Total actual cost")
    baseline_cost: float = Field(..., ge=0, description="Cost without optimization")
    total_savings: float = Field(..., ge=0, description="Total savings")
    savings_percent: float = Field(..., ge=0, description="Savings percentage")
    breakdown: Optional[SavingsBreakdown] = Field(None, description="Savings breakdown")
    schedules_optimized: int = Field(..., ge=0, description="Number of schedules optimized")
    co2_avoided_kg: float = Field(..., ge=0, description="CO2 emissions avoided")
    generated_at: datetime = Field(..., description="Report generation timestamp")

    class Config:
        schema_extra = {
            "example": {
                "report_id": "rpt_sav_202311",
                "facility_id": "facility_01",
                "period_start": "2025-11-01",
                "period_end": "2025-11-30",
                "total_energy_kwh": 150000.0,
                "total_cost": 12000.00,
                "baseline_cost": 15000.00,
                "total_savings": 3000.00,
                "savings_percent": 20.0,
                "breakdown": {
                    "time_shifting": 1800.00,
                    "demand_reduction": 600.00,
                    "efficiency_improvement": 300.00,
                    "demand_response": 300.00
                },
                "schedules_optimized": 45,
                "co2_avoided_kg": 1500.0,
                "generated_at": "2025-12-01T08:00:00Z"
            }
        }


class CostForecastPoint(BaseModel):
    """Single point in cost forecast."""
    date: date = Field(..., description="Forecast date")
    forecasted_cost: float = Field(..., ge=0, description="Forecasted cost")
    forecasted_energy_kwh: float = Field(..., ge=0, description="Forecasted energy")
    confidence_lower: float = Field(..., ge=0, description="Lower confidence bound")
    confidence_upper: float = Field(..., ge=0, description="Upper confidence bound")


class CostForecastResponse(BaseModel):
    """Cost forecast response."""
    facility_id: str = Field(..., description="Facility identifier")
    forecast_horizon_days: int = Field(..., ge=1, description="Forecast horizon in days")
    forecast_generated_at: datetime = Field(..., description="When forecast was generated")
    currency: str = Field("USD", description="Currency code")
    forecast: List[CostForecastPoint] = Field(..., description="Forecast data points")
    total_forecasted_cost: float = Field(..., ge=0, description="Total forecasted cost")
    total_forecasted_energy_kwh: float = Field(..., ge=0, description="Total forecasted energy")


class WhatIfScenario(BaseModel):
    """What-if scenario parameters."""
    scenario_name: str = Field(..., min_length=1, max_length=255, description="Scenario name")
    schedule_id: Optional[str] = Field(None, description="Base schedule to modify")
    tariff_change_percent: Optional[float] = Field(
        None,
        ge=-100,
        le=500,
        description="Tariff rate change percentage"
    )
    demand_change_percent: Optional[float] = Field(
        None,
        ge=-100,
        le=500,
        description="Demand change percentage"
    )
    equipment_efficiency_change: Optional[float] = Field(
        None,
        ge=-0.5,
        le=0.5,
        description="Equipment efficiency change"
    )
    new_equipment_ids: Optional[List[str]] = Field(
        None,
        description="Additional equipment to add"
    )
    remove_equipment_ids: Optional[List[str]] = Field(
        None,
        description="Equipment to remove"
    )

    class Config:
        schema_extra = {
            "example": {
                "scenario_name": "10% Tariff Increase Impact",
                "schedule_id": "sched_abc123",
                "tariff_change_percent": 10.0
            }
        }


class WhatIfResult(BaseModel):
    """What-if analysis result."""
    scenario_name: str = Field(..., description="Scenario name")
    baseline_cost: float = Field(..., ge=0, description="Baseline cost")
    scenario_cost: float = Field(..., ge=0, description="Cost under scenario")
    cost_difference: float = Field(..., description="Cost difference")
    cost_difference_percent: float = Field(..., description="Cost difference percentage")
    baseline_energy_kwh: float = Field(..., ge=0, description="Baseline energy")
    scenario_energy_kwh: float = Field(..., ge=0, description="Energy under scenario")
    recommendations: List[str] = Field(..., description="Recommendations based on analysis")
    analyzed_at: datetime = Field(..., description="Analysis timestamp")


# =============================================================================
# Demand Response Models
# =============================================================================

class DemandResponseEventRequest(BaseModel):
    """Demand response event notification."""
    event_id: str = Field(..., description="Event identifier from grid operator")
    event_type: DemandResponseEventType = Field(..., description="Type of DR event")
    facility_id: str = Field(..., description="Target facility")
    start_time: datetime = Field(..., description="Event start time")
    end_time: datetime = Field(..., description="Event end time")
    required_reduction_kw: float = Field(..., ge=0, description="Required load reduction in kW")
    incentive_rate: Optional[float] = Field(
        None,
        ge=0,
        description="Incentive rate per kWh reduced"
    )
    penalty_rate: Optional[float] = Field(
        None,
        ge=0,
        description="Penalty rate for non-compliance"
    )
    mandatory: bool = Field(False, description="Whether event is mandatory")
    notification_lead_time_minutes: int = Field(..., ge=0, description="Lead time in minutes")

    class Config:
        schema_extra = {
            "example": {
                "event_id": "dr_evt_20231109_001",
                "event_type": "curtailment",
                "facility_id": "facility_01",
                "start_time": "2025-11-09T14:00:00Z",
                "end_time": "2025-11-09T18:00:00Z",
                "required_reduction_kw": 200.0,
                "incentive_rate": 0.50,
                "penalty_rate": 1.00,
                "mandatory": False,
                "notification_lead_time_minutes": 60
            }
        }


class DemandResponseEventResponse(BaseModel):
    """Response to demand response event."""
    event_id: str = Field(..., description="Event identifier")
    facility_id: str = Field(..., description="Facility identifier")
    participation_status: str = Field(..., description="Participation decision")
    committed_reduction_kw: float = Field(..., ge=0, description="Committed reduction")
    estimated_revenue: float = Field(..., ge=0, description="Estimated incentive revenue")
    rescheduled_operations: int = Field(..., ge=0, description="Number of operations rescheduled")
    response_received_at: datetime = Field(..., description="Response timestamp")


class DemandResponseStatusResponse(BaseModel):
    """Current demand response status."""
    facility_id: str = Field(..., description="Facility identifier")
    status: DemandResponseStatus = Field(..., description="Current DR status")
    active_events: List[Dict[str, Any]] = Field(..., description="Currently active events")
    pending_events: List[Dict[str, Any]] = Field(..., description="Upcoming events")
    current_load_kw: float = Field(..., ge=0, description="Current load in kW")
    available_reduction_kw: float = Field(..., ge=0, description="Available load reduction")
    ytd_participation_count: int = Field(..., ge=0, description="Year-to-date participations")
    ytd_revenue: float = Field(..., ge=0, description="Year-to-date DR revenue")
    last_updated: datetime = Field(..., description="Status last updated")

    class Config:
        schema_extra = {
            "example": {
                "facility_id": "facility_01",
                "status": "idle",
                "active_events": [],
                "pending_events": [],
                "current_load_kw": 350.0,
                "available_reduction_kw": 150.0,
                "ytd_participation_count": 12,
                "ytd_revenue": 5400.00,
                "last_updated": "2025-11-09T10:30:00Z"
            }
        }


# =============================================================================
# Error Response Models
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""
    field: Optional[str] = Field(None, description="Field with error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed error info")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid request parameters",
                "details": [
                    {
                        "field": "start_date",
                        "message": "start_date must be before end_date",
                        "code": "invalid_date_range"
                    }
                ],
                "request_id": "req_abc123",
                "timestamp": "2025-11-09T10:30:00Z"
            }
        }


# =============================================================================
# API Key and Auth Models
# =============================================================================

class TokenResponse(BaseModel):
    """OAuth2 token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiry in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    scope: str = Field(..., description="Token scope")


class APIKeyInfo(BaseModel):
    """API key information."""
    key_id: str = Field(..., description="API key identifier")
    name: str = Field(..., description="API key name")
    prefix: str = Field(..., description="Key prefix (first 8 chars)")
    scopes: List[str] = Field(..., description="Authorized scopes")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
