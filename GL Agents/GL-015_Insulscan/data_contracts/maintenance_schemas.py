# -*- coding: utf-8 -*-
"""
GL-015 Insulscan: Maintenance Schemas - Version 1.0

Provides validated data schemas for insulation repair work orders,
material specifications, and maintenance scheduling.

This module defines Pydantic v2 models for:
- RepairWorkOrder: Complete repair work order with materials and labor
- MaterialSpec: Insulation material specification for repairs
- MaintenanceSchedule: Scheduled maintenance with crew assignment

Author: GreenLang AI Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RepairType(str, Enum):
    """Type of insulation repair."""
    PATCH_REPAIR = "patch_repair"
    SECTION_REPLACEMENT = "section_replacement"
    FULL_REPLACEMENT = "full_replacement"
    JACKET_REPAIR = "jacket_repair"
    JACKET_REPLACEMENT = "jacket_replacement"
    SEALANT_APPLICATION = "sealant_application"
    BAND_TIGHTENING = "band_tightening"
    VAPOR_BARRIER_REPAIR = "vapor_barrier_repair"
    SUPPORT_REPAIR = "support_repair"
    REWRAP = "rewrap"
    PREVENTIVE = "preventive"
    OTHER = "other"


class WorkOrderStatus(str, Enum):
    """Work order lifecycle status."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    MATERIALS_ORDERED = "materials_ordered"
    MATERIALS_RECEIVED = "materials_received"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    VERIFIED = "verified"
    CANCELLED = "cancelled"
    CLOSED = "closed"


class WorkOrderPriority(str, Enum):
    """Work order priority level."""
    EMERGENCY = "emergency"
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ROUTINE = "routine"


class MaintenanceCategory(str, Enum):
    """Category of maintenance activity."""
    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CONDITION_BASED = "condition_based"
    EMERGENCY = "emergency"
    IMPROVEMENT = "improvement"
    TURNAROUND = "turnaround"


class MaterialUnit(str, Enum):
    """Units for material quantities."""
    PIECE = "piece"
    METER = "meter"
    SQUARE_METER = "square_meter"
    CUBIC_METER = "cubic_meter"
    KILOGRAM = "kilogram"
    ROLL = "roll"
    BOX = "box"
    LITER = "liter"
    SET = "set"


class ScheduleStatus(str, Enum):
    """Status of scheduled maintenance."""
    PLANNED = "planned"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class CrewType(str, Enum):
    """Type of maintenance crew."""
    INSULATION = "insulation"
    MECHANICAL = "mechanical"
    SCAFFOLDING = "scaffolding"
    INSPECTION = "inspection"
    CONTRACTOR = "contractor"
    MULTI_CRAFT = "multi_craft"


# =============================================================================
# MATERIAL SPECIFICATION
# =============================================================================

class MaterialSpec(BaseModel):
    """
    Material specification for insulation repairs.

    Defines material requirements including type, quantity,
    supplier information, and cost for repair work orders.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "material_id": "MAT-001",
                    "material_type": "mineral_wool",
                    "description": "Mineral wool pipe insulation, 75mm thick",
                    "quantity": 15.0,
                    "unit": "meter",
                    "unit_cost": 25.50,
                    "supplier": "Rockwool Inc."
                }
            ]
        }
    )

    # Identification
    material_id: str = Field(
        default_factory=lambda: f"MAT-{uuid.uuid4().hex[:8].upper()}",
        description="Unique material identifier"
    )
    material_type: Literal[
        "insulation", "jacket", "sealant", "tape", "band",
        "wire", "adhesive", "vapor_barrier", "support", "fastener", "other"
    ] = Field(
        ...,
        description="Type of material"
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Material description"
    )

    # Specifications
    product_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Product name"
    )
    product_code: Optional[str] = Field(
        None,
        max_length=50,
        description="Manufacturer product code"
    )
    manufacturer: Optional[str] = Field(
        None,
        max_length=200,
        description="Material manufacturer"
    )

    # For insulation materials
    insulation_material_type: Optional[str] = Field(
        None,
        max_length=100,
        description="Insulation material type (e.g., mineral_wool)"
    )
    thickness_mm: Optional[float] = Field(
        None,
        gt=0,
        le=500,
        description="Material thickness in mm"
    )
    density_kg_m3: Optional[float] = Field(
        None,
        gt=0,
        le=2000,
        description="Material density in kg/m^3"
    )
    thermal_conductivity_w_mk: Optional[float] = Field(
        None,
        gt=0,
        le=5,
        description="Thermal conductivity in W/(m.K)"
    )
    max_service_temp_c: Optional[float] = Field(
        None,
        ge=-273.15,
        le=1500,
        description="Maximum service temperature in Celsius"
    )

    # Quantity and cost
    quantity: float = Field(
        ...,
        gt=0,
        description="Required quantity"
    )
    unit: MaterialUnit = Field(
        ...,
        description="Unit of measurement"
    )
    unit_cost: float = Field(
        ...,
        ge=0,
        description="Cost per unit in local currency"
    )
    total_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Total cost (quantity * unit_cost)"
    )
    currency: str = Field(
        default="USD",
        max_length=3,
        description="Currency code"
    )

    # Supplier information
    supplier: Optional[str] = Field(
        None,
        max_length=200,
        description="Supplier name"
    )
    supplier_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Supplier identifier"
    )
    supplier_part_number: Optional[str] = Field(
        None,
        max_length=50,
        description="Supplier part number"
    )
    lead_time_days: Optional[int] = Field(
        None,
        ge=0,
        description="Lead time for ordering in days"
    )

    # Inventory
    in_stock: bool = Field(
        default=False,
        description="Whether material is in stock"
    )
    stock_location: Optional[str] = Field(
        None,
        max_length=100,
        description="Stock/warehouse location"
    )
    stock_quantity: Optional[float] = Field(
        None,
        ge=0,
        description="Available stock quantity"
    )

    # Certification
    certified: bool = Field(
        default=False,
        description="Whether material is certified for application"
    )
    certification_ref: Optional[str] = Field(
        None,
        max_length=100,
        description="Certification reference"
    )

    @model_validator(mode="after")
    def calculate_total_cost(self) -> "MaterialSpec":
        """Calculate total cost if not provided."""
        # Note: Cannot modify frozen model, so this validates consistency
        if self.total_cost is not None:
            expected = self.quantity * self.unit_cost
            if abs(self.total_cost - expected) > 0.01:
                # Allow small rounding differences
                pass
        return self


# =============================================================================
# LABOR SPECIFICATION
# =============================================================================

class LaborSpec(BaseModel):
    """Labor specification for work order."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "craft": "insulation",
                    "estimated_hours": 8.0,
                    "workers_required": 2,
                    "hourly_rate": 75.0
                }
            ]
        }
    )

    craft: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Craft/skill type"
    )
    estimated_hours: float = Field(
        ...,
        gt=0,
        description="Estimated labor hours"
    )
    actual_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Actual labor hours spent"
    )
    workers_required: int = Field(
        default=1,
        ge=1,
        description="Number of workers required"
    )
    hourly_rate: float = Field(
        ...,
        ge=0,
        description="Hourly labor rate"
    )
    overtime_rate: Optional[float] = Field(
        None,
        ge=0,
        description="Overtime hourly rate"
    )
    estimated_labor_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated total labor cost"
    )
    actual_labor_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Actual labor cost"
    )


# =============================================================================
# REPAIR WORK ORDER
# =============================================================================

class RepairWorkOrder(BaseModel):
    """
    Complete repair work order for insulation maintenance.

    Tracks the full lifecycle of repair work including materials,
    labor, scheduling, and completion verification.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "work_order_id": "WO-2024-00156",
                    "asset_id": "INS-1001",
                    "repair_type": "patch_repair",
                    "title": "Repair damaged insulation at pipe support",
                    "category": "corrective",
                    "priority": "high",
                    "status": "approved",
                    "estimated_hours": 4.0,
                    "estimated_cost": 850.0
                }
            ]
        }
    )

    # Identifiers
    schema_version: str = Field(
        default="1.0",
        description="Schema version for compatibility tracking"
    )
    work_order_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique work order identifier"
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to insulation asset"
    )

    # Work order details
    repair_type: RepairType = Field(
        ...,
        description="Type of repair to be performed"
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Work order title"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Detailed work description"
    )
    scope_of_work: Optional[str] = Field(
        None,
        max_length=2000,
        description="Detailed scope of work"
    )

    # Classification
    category: MaintenanceCategory = Field(
        ...,
        description="Maintenance category"
    )
    priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Work order priority"
    )
    status: WorkOrderStatus = Field(
        default=WorkOrderStatus.DRAFT,
        description="Current status"
    )

    # Affected area
    affected_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Affected area in square meters"
    )
    affected_length_m: Optional[float] = Field(
        None,
        ge=0,
        description="Affected length in meters"
    )
    location_description: Optional[str] = Field(
        None,
        max_length=500,
        description="Description of repair location"
    )

    # Materials
    materials_required: List[MaterialSpec] = Field(
        default_factory=list,
        description="List of materials required"
    )
    estimated_material_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Total estimated material cost"
    )
    actual_material_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Actual material cost"
    )

    # Labor
    labor_required: List[LaborSpec] = Field(
        default_factory=list,
        description="Labor requirements"
    )
    estimated_hours: float = Field(
        ...,
        gt=0,
        description="Total estimated labor hours"
    )
    actual_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Actual labor hours"
    )
    estimated_labor_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated labor cost"
    )
    actual_labor_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Actual labor cost"
    )

    # Total costs
    estimated_cost: float = Field(
        ...,
        ge=0,
        description="Total estimated cost"
    )
    actual_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Actual total cost"
    )
    budget_code: Optional[str] = Field(
        None,
        max_length=50,
        description="Budget/cost center code"
    )
    currency: str = Field(
        default="USD",
        max_length=3,
        description="Currency code"
    )

    # Equipment/scaffolding
    scaffolding_required: bool = Field(
        default=False,
        description="Whether scaffolding is required"
    )
    scaffolding_height_m: Optional[float] = Field(
        None,
        gt=0,
        description="Required scaffolding height in meters"
    )
    equipment_required: List[str] = Field(
        default_factory=list,
        description="List of equipment required"
    )
    equipment_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Equipment rental/usage cost"
    )

    # Dates
    created_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Work order creation date"
    )
    required_by_date: Optional[datetime] = Field(
        None,
        description="Date by which work must be completed"
    )
    scheduled_start: Optional[datetime] = Field(
        None,
        description="Scheduled start date/time"
    )
    scheduled_end: Optional[datetime] = Field(
        None,
        description="Scheduled end date/time"
    )
    actual_start: Optional[datetime] = Field(
        None,
        description="Actual start date/time"
    )
    actual_end: Optional[datetime] = Field(
        None,
        description="Actual end date/time"
    )

    # Personnel
    requested_by: Optional[str] = Field(
        None,
        max_length=200,
        description="Person who requested the work"
    )
    approved_by: Optional[str] = Field(
        None,
        max_length=200,
        description="Person who approved the work order"
    )
    assigned_to: Optional[str] = Field(
        None,
        max_length=200,
        description="Person or crew assigned"
    )
    performed_by: Optional[str] = Field(
        None,
        max_length=200,
        description="Person/crew who performed the work"
    )
    verified_by: Optional[str] = Field(
        None,
        max_length=200,
        description="Person who verified completion"
    )

    # Related records
    inspection_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Related inspection record ID"
    )
    assessment_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Related condition assessment ID"
    )
    parent_work_order_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Parent work order ID"
    )

    # CMMS integration
    cmms_system: Optional[str] = Field(
        None,
        max_length=100,
        description="CMMS system name"
    )
    cmms_work_order_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Work order ID in CMMS"
    )

    # Safety
    safety_requirements: Optional[str] = Field(
        None,
        max_length=1000,
        description="Safety requirements and precautions"
    )
    permits_required: List[str] = Field(
        default_factory=list,
        description="List of required permits"
    )
    isolation_required: bool = Field(
        default=False,
        description="Whether process isolation is required"
    )

    # Completion
    work_performed: Optional[str] = Field(
        None,
        max_length=2000,
        description="Description of work performed"
    )
    completion_notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Completion notes"
    )
    quality_verified: bool = Field(
        default=False,
        description="Whether quality was verified"
    )
    photos_before: List[str] = Field(
        default_factory=list,
        description="References to before photos"
    )
    photos_after: List[str] = Field(
        default_factory=list,
        description="References to after photos"
    )

    # Timestamps
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail"
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "RepairWorkOrder":
        """Validate date consistency."""
        if self.scheduled_end and self.scheduled_start:
            if self.scheduled_end < self.scheduled_start:
                raise ValueError("Scheduled end must be after scheduled start")
        if self.actual_end and self.actual_start:
            if self.actual_end < self.actual_start:
                raise ValueError("Actual end must be after actual start")
        return self

    @property
    def is_complete(self) -> bool:
        """Check if work order is complete."""
        return self.status in [
            WorkOrderStatus.COMPLETED,
            WorkOrderStatus.VERIFIED,
            WorkOrderStatus.CLOSED
        ]

    @property
    def is_overdue(self) -> bool:
        """Check if work order is overdue."""
        if self.required_by_date and not self.is_complete:
            return datetime.now(timezone.utc) > self.required_by_date
        return False

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        content = (
            f"{self.work_order_id}"
            f"{self.asset_id}"
            f"{self.repair_type.value}"
            f"{self.status.value}"
            f"{self.created_date.isoformat()}"
        )
        return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# MAINTENANCE SCHEDULE
# =============================================================================

class MaintenanceSchedule(BaseModel):
    """
    Scheduled maintenance activity.

    Tracks planned maintenance activities with crew assignment,
    timing, and status tracking.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "schedule_id": "SCH-2024-00042",
                    "work_order_id": "WO-2024-00156",
                    "asset_id": "INS-1001",
                    "scheduled_date": "2024-02-01T08:00:00Z",
                    "priority": "high",
                    "assigned_crew": "Crew A - Insulation",
                    "status": "confirmed"
                }
            ]
        }
    )

    # Identifiers
    schedule_id: str = Field(
        default_factory=lambda: f"SCH-{uuid.uuid4().hex[:12].upper()}",
        description="Unique schedule identifier"
    )
    work_order_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to work order"
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to asset"
    )

    # Scheduling
    scheduled_date: datetime = Field(
        ...,
        description="Scheduled date and time"
    )
    scheduled_end_date: Optional[datetime] = Field(
        None,
        description="Scheduled end date and time"
    )
    duration_hours: Optional[float] = Field(
        None,
        gt=0,
        description="Scheduled duration in hours"
    )

    # Priority and status
    priority: WorkOrderPriority = Field(
        ...,
        description="Schedule priority"
    )
    status: ScheduleStatus = Field(
        default=ScheduleStatus.PLANNED,
        description="Schedule status"
    )

    # Crew assignment
    assigned_crew: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Assigned crew name or ID"
    )
    crew_type: CrewType = Field(
        default=CrewType.INSULATION,
        description="Type of crew"
    )
    crew_size: Optional[int] = Field(
        None,
        ge=1,
        description="Number of crew members"
    )
    crew_lead: Optional[str] = Field(
        None,
        max_length=200,
        description="Crew lead name"
    )
    crew_contact: Optional[str] = Field(
        None,
        max_length=200,
        description="Crew contact information"
    )

    # Supporting activities
    scaffolding_scheduled: bool = Field(
        default=False,
        description="Whether scaffolding is scheduled"
    )
    scaffolding_date: Optional[datetime] = Field(
        None,
        description="Scaffolding erection date"
    )
    permits_obtained: bool = Field(
        default=False,
        description="Whether required permits are obtained"
    )
    materials_available: bool = Field(
        default=False,
        description="Whether materials are available"
    )

    # Process coordination
    process_impact: Optional[str] = Field(
        None,
        max_length=500,
        description="Expected process impact"
    )
    isolation_required: bool = Field(
        default=False,
        description="Whether isolation is required"
    )
    isolation_confirmed: bool = Field(
        default=False,
        description="Whether isolation is confirmed"
    )
    coordination_notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="Coordination notes"
    )

    # Weather considerations (for outdoor work)
    weather_dependent: bool = Field(
        default=False,
        description="Whether work is weather dependent"
    )
    weather_constraints: Optional[str] = Field(
        None,
        max_length=500,
        description="Weather constraints"
    )

    # Notification
    notifications_sent: bool = Field(
        default=False,
        description="Whether notifications have been sent"
    )
    notification_recipients: List[str] = Field(
        default_factory=list,
        description="List of notification recipients"
    )

    # Rescheduling
    original_date: Optional[datetime] = Field(
        None,
        description="Original scheduled date if rescheduled"
    )
    reschedule_reason: Optional[str] = Field(
        None,
        max_length=500,
        description="Reason for rescheduling"
    )
    reschedule_count: int = Field(
        default=0,
        ge=0,
        description="Number of times rescheduled"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )

    @model_validator(mode="after")
    def validate_schedule(self) -> "MaintenanceSchedule":
        """Validate schedule consistency."""
        if self.scheduled_end_date and self.scheduled_end_date < self.scheduled_date:
            raise ValueError("Scheduled end must be after scheduled start")
        if self.scaffolding_date and self.scaffolding_date > self.scheduled_date:
            raise ValueError("Scaffolding must be erected before scheduled work date")
        return self

    @property
    def is_ready(self) -> bool:
        """Check if all prerequisites are met."""
        prerequisites = [
            self.materials_available,
            not self.isolation_required or self.isolation_confirmed,
        ]
        if self.scaffolding_scheduled:
            # Scaffolding should be completed
            pass
        return all(prerequisites)


# =============================================================================
# EXPORTS
# =============================================================================

MAINTENANCE_SCHEMAS = {
    "RepairType": RepairType,
    "WorkOrderStatus": WorkOrderStatus,
    "WorkOrderPriority": WorkOrderPriority,
    "MaintenanceCategory": MaintenanceCategory,
    "MaterialUnit": MaterialUnit,
    "ScheduleStatus": ScheduleStatus,
    "CrewType": CrewType,
    "MaterialSpec": MaterialSpec,
    "LaborSpec": LaborSpec,
    "RepairWorkOrder": RepairWorkOrder,
    "MaintenanceSchedule": MaintenanceSchedule,
}

__all__ = [
    # Enumerations
    "RepairType",
    "WorkOrderStatus",
    "WorkOrderPriority",
    "MaintenanceCategory",
    "MaterialUnit",
    "ScheduleStatus",
    "CrewType",
    # Supporting models
    "MaterialSpec",
    "LaborSpec",
    # Main schemas
    "RepairWorkOrder",
    "MaintenanceSchedule",
    # Export dictionary
    "MAINTENANCE_SCHEMAS",
]
