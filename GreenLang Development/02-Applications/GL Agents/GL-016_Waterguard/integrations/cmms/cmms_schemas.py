"""
GL-016 Waterguard CMMS Schemas

Data models for CMMS integration including work orders, assets,
and maintenance tasks.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================

class WorkOrderType(str, Enum):
    """Types of maintenance work orders."""
    CORRECTIVE = "corrective"  # Fix something broken
    PREVENTIVE = "preventive"  # Scheduled maintenance
    PREDICTIVE = "predictive"  # AI-recommended maintenance
    CALIBRATION = "calibration"  # Analyzer calibration
    INSPECTION = "inspection"  # Visual or functional inspection
    EMERGENCY = "emergency"  # Urgent safety-related


class WorkOrderPriority(str, Enum):
    """Work order priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class WorkOrderStatus(str, Enum):
    """Work order status."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    CLOSED = "closed"


class AssetType(str, Enum):
    """Types of assets in water treatment system."""
    ANALYZER = "analyzer"
    PUMP = "pump"
    VALVE = "valve"
    TANK = "tank"
    SENSOR = "sensor"
    CONTROLLER = "controller"
    PIPING = "piping"
    HEAT_EXCHANGER = "heat_exchanger"
    OTHER = "other"


class EquipmentStatus(str, Enum):
    """Equipment operational status."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    CALIBRATING = "calibrating"


# =============================================================================
# Asset
# =============================================================================

class AssetLocation(BaseModel):
    """Physical location of an asset."""
    site: str = Field(..., description="Site/plant name")
    building: Optional[str] = Field(default=None, description="Building")
    floor: Optional[str] = Field(default=None, description="Floor")
    area: Optional[str] = Field(default=None, description="Process area")
    position: Optional[str] = Field(default=None, description="Specific position")


class Asset(BaseModel):
    """
    Represents an equipment asset in the CMMS.

    Assets include analyzers, pumps, valves, and other equipment
    that requires maintenance tracking.
    """

    # Identification
    asset_id: str = Field(..., description="Unique asset ID")
    asset_tag: str = Field(..., description="Physical asset tag number")
    name: str = Field(..., description="Asset name")
    description: str = Field(default="", description="Asset description")

    # Classification
    asset_type: AssetType = Field(..., description="Type of asset")
    manufacturer: Optional[str] = Field(default=None, description="Manufacturer")
    model: Optional[str] = Field(default=None, description="Model number")
    serial_number: Optional[str] = Field(default=None, description="Serial number")

    # Location
    location: AssetLocation = Field(..., description="Physical location")

    # Status
    status: EquipmentStatus = Field(
        default=EquipmentStatus.OPERATIONAL,
        description="Current status"
    )
    last_status_change: Optional[datetime] = Field(default=None)

    # Maintenance
    installation_date: Optional[datetime] = Field(default=None)
    last_maintenance_date: Optional[datetime] = Field(default=None)
    next_maintenance_date: Optional[datetime] = Field(default=None)
    maintenance_interval_days: Optional[int] = Field(default=None)

    # Calibration (for analyzers)
    last_calibration_date: Optional[datetime] = Field(default=None)
    next_calibration_date: Optional[datetime] = Field(default=None)
    calibration_interval_days: Optional[int] = Field(default=None)

    # OPC-UA mapping
    opc_node_ids: List[str] = Field(
        default_factory=list,
        description="Associated OPC-UA node IDs"
    )

    # Metadata
    properties: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    @property
    def is_calibration_due(self) -> bool:
        """Check if calibration is due."""
        if self.next_calibration_date is None:
            return False
        return datetime.utcnow() >= self.next_calibration_date

    @property
    def is_maintenance_due(self) -> bool:
        """Check if maintenance is due."""
        if self.next_maintenance_date is None:
            return False
        return datetime.utcnow() >= self.next_maintenance_date

    @property
    def days_until_calibration(self) -> Optional[int]:
        """Days until next calibration."""
        if self.next_calibration_date is None:
            return None
        delta = self.next_calibration_date - datetime.utcnow()
        return max(0, delta.days)


# =============================================================================
# Maintenance Task
# =============================================================================

class TaskType(str, Enum):
    """Types of maintenance tasks."""
    INSPECTION = "inspection"
    CLEANING = "cleaning"
    LUBRICATION = "lubrication"
    ADJUSTMENT = "adjustment"
    REPLACEMENT = "replacement"
    CALIBRATION = "calibration"
    TESTING = "testing"
    REPAIR = "repair"
    OTHER = "other"


class MaintenanceTask(BaseModel):
    """
    Individual maintenance task within a work order.
    """

    task_id: UUID = Field(default_factory=uuid4, description="Task ID")
    sequence: int = Field(..., description="Task sequence number")
    task_type: TaskType = Field(..., description="Type of task")
    description: str = Field(..., description="Task description")
    instructions: str = Field(default="", description="Detailed instructions")

    # Time
    estimated_duration_minutes: int = Field(default=30, description="Estimated duration")
    actual_duration_minutes: Optional[int] = Field(default=None)

    # Status
    completed: bool = Field(default=False)
    completed_at: Optional[datetime] = Field(default=None)
    completed_by: Optional[str] = Field(default=None)

    # Results
    result_notes: Optional[str] = Field(default=None)
    measurements: Dict[str, float] = Field(default_factory=dict)
    pass_fail: Optional[bool] = Field(default=None)

    # Parts and materials
    required_parts: List[str] = Field(default_factory=list)
    used_parts: List[str] = Field(default_factory=list)


# =============================================================================
# Work Order
# =============================================================================

class WorkOrderContext(BaseModel):
    """Context information for work order creation."""

    # Trigger information
    trigger_type: str = Field(..., description="What triggered the WO")
    trigger_timestamp: datetime = Field(default_factory=datetime.utcnow)
    trigger_source: str = Field(default="system", description="Source system")

    # Related data
    alarm_id: Optional[str] = Field(default=None, description="Related alarm ID")
    recommendation_id: Optional[UUID] = Field(default=None)
    trace_id: Optional[UUID] = Field(default=None)

    # Measurements at time of trigger
    readings: Dict[str, float] = Field(
        default_factory=dict,
        description="Relevant readings at trigger time"
    )

    # Additional context
    notes: str = Field(default="", description="Additional notes")
    attachments: List[str] = Field(default_factory=list, description="Attachment URLs")


class WorkOrder(BaseModel):
    """
    CMMS work order for maintenance activities.

    Represents a complete maintenance work order with all tracking
    information for asset maintenance.
    """

    # Identification
    work_order_id: UUID = Field(default_factory=uuid4, description="Work order ID")
    work_order_number: Optional[str] = Field(
        default=None,
        description="CMMS work order number"
    )
    external_id: Optional[str] = Field(
        default=None,
        description="External system reference ID"
    )

    # Classification
    work_order_type: WorkOrderType = Field(..., description="Type of work order")
    priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Priority level"
    )
    status: WorkOrderStatus = Field(
        default=WorkOrderStatus.DRAFT,
        description="Current status"
    )

    # Asset
    asset_id: str = Field(..., description="Target asset ID")
    asset_name: Optional[str] = Field(default=None, description="Asset name")
    asset_location: Optional[str] = Field(default=None, description="Asset location")

    # Description
    title: str = Field(..., description="Work order title")
    description: str = Field(..., description="Detailed description")
    symptoms: Optional[str] = Field(default=None, description="Observed symptoms")
    root_cause: Optional[str] = Field(default=None, description="Identified root cause")

    # Context
    context: WorkOrderContext = Field(..., description="Creation context")

    # Tasks
    tasks: List[MaintenanceTask] = Field(
        default_factory=list,
        description="Maintenance tasks"
    )

    # Scheduling
    requested_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Requested completion date"
    )
    scheduled_start: Optional[datetime] = Field(default=None)
    scheduled_end: Optional[datetime] = Field(default=None)
    actual_start: Optional[datetime] = Field(default=None)
    actual_end: Optional[datetime] = Field(default=None)

    # Assignment
    assigned_to: Optional[str] = Field(default=None, description="Assigned technician")
    assigned_team: Optional[str] = Field(default=None, description="Assigned team")
    created_by: str = Field(default="system", description="Creator")

    # Resolution
    resolution_notes: Optional[str] = Field(default=None)
    actions_taken: Optional[str] = Field(default=None)
    parts_used: List[str] = Field(default_factory=list)
    total_cost: Optional[float] = Field(default=None)
    downtime_minutes: Optional[int] = Field(default=None)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    approved_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    closed_at: Optional[datetime] = Field(default=None)

    # Idempotency
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Key for preventing duplicate work orders"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str,
        }

    @property
    def is_open(self) -> bool:
        """Check if work order is still open."""
        return self.status not in [
            WorkOrderStatus.COMPLETED,
            WorkOrderStatus.CANCELLED,
            WorkOrderStatus.CLOSED,
        ]

    @property
    def is_overdue(self) -> bool:
        """Check if work order is overdue."""
        if not self.is_open:
            return False
        if self.requested_date:
            return datetime.utcnow() > self.requested_date
        return False

    @property
    def completion_percentage(self) -> float:
        """Calculate task completion percentage."""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.completed)
        return (completed / len(self.tasks)) * 100

    @property
    def estimated_duration_minutes(self) -> int:
        """Total estimated duration."""
        return sum(t.estimated_duration_minutes for t in self.tasks)

    def generate_idempotency_key(self) -> str:
        """Generate idempotency key for duplicate prevention."""
        import hashlib
        key_parts = [
            self.asset_id,
            self.work_order_type.value,
            self.context.trigger_type,
            self.context.trigger_timestamp.strftime("%Y%m%d%H"),  # Hour precision
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]


# =============================================================================
# Work Order Templates
# =============================================================================

class WorkOrderTemplate(BaseModel):
    """Template for creating work orders."""

    template_id: str = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    work_order_type: WorkOrderType = Field(..., description="Work order type")
    default_priority: WorkOrderPriority = Field(default=WorkOrderPriority.MEDIUM)

    # Content
    title_template: str = Field(..., description="Title template with placeholders")
    description_template: str = Field(..., description="Description template")

    # Default tasks
    default_tasks: List[MaintenanceTask] = Field(default_factory=list)

    # Applicability
    applicable_asset_types: List[AssetType] = Field(default_factory=list)
    applicable_triggers: List[str] = Field(default_factory=list)

    def create_work_order(
        self,
        asset: Asset,
        context: WorkOrderContext,
        **overrides
    ) -> WorkOrder:
        """Create work order from template."""
        # Format title and description with asset info
        title = self.title_template.format(
            asset_name=asset.name,
            asset_tag=asset.asset_tag,
            asset_type=asset.asset_type.value,
        )
        description = self.description_template.format(
            asset_name=asset.name,
            asset_tag=asset.asset_tag,
            trigger_type=context.trigger_type,
        )

        # Copy tasks with new IDs
        tasks = [
            MaintenanceTask(
                sequence=i + 1,
                task_type=t.task_type,
                description=t.description,
                instructions=t.instructions,
                estimated_duration_minutes=t.estimated_duration_minutes,
                required_parts=t.required_parts.copy(),
            )
            for i, t in enumerate(self.default_tasks)
        ]

        return WorkOrder(
            work_order_type=self.work_order_type,
            priority=self.default_priority,
            asset_id=asset.asset_id,
            asset_name=asset.name,
            asset_location=f"{asset.location.site}/{asset.location.area}",
            title=title,
            description=description,
            context=context,
            tasks=tasks,
            **overrides
        )


# =============================================================================
# Default Templates
# =============================================================================

def get_calibration_template() -> WorkOrderTemplate:
    """Get calibration work order template."""
    return WorkOrderTemplate(
        template_id="calibration_due",
        name="Analyzer Calibration",
        work_order_type=WorkOrderType.CALIBRATION,
        default_priority=WorkOrderPriority.HIGH,
        title_template="{asset_name} Calibration Due",
        description_template=(
            "Scheduled calibration for {asset_name} ({asset_tag}). "
            "Triggered by: {trigger_type}. Follow standard calibration procedure."
        ),
        default_tasks=[
            MaintenanceTask(
                sequence=1,
                task_type=TaskType.INSPECTION,
                description="Visual inspection of analyzer",
                instructions="Check for leaks, damage, and proper operation",
                estimated_duration_minutes=10,
            ),
            MaintenanceTask(
                sequence=2,
                task_type=TaskType.CLEANING,
                description="Clean analyzer probe/sensor",
                instructions="Use approved cleaning solution per manufacturer specs",
                estimated_duration_minutes=15,
            ),
            MaintenanceTask(
                sequence=3,
                task_type=TaskType.CALIBRATION,
                description="Perform two-point calibration",
                instructions="Use certified calibration standards",
                estimated_duration_minutes=30,
            ),
            MaintenanceTask(
                sequence=4,
                task_type=TaskType.TESTING,
                description="Verify calibration accuracy",
                instructions="Test with QC standard, document results",
                estimated_duration_minutes=15,
            ),
        ],
        applicable_asset_types=[AssetType.ANALYZER],
        applicable_triggers=["calibration_due", "calibration_failed", "drift_detected"],
    )


def get_analyzer_fault_template() -> WorkOrderTemplate:
    """Get analyzer fault work order template."""
    return WorkOrderTemplate(
        template_id="analyzer_fault",
        name="Analyzer Fault Investigation",
        work_order_type=WorkOrderType.CORRECTIVE,
        default_priority=WorkOrderPriority.HIGH,
        title_template="{asset_name} Fault - Investigation Required",
        description_template=(
            "Fault detected on {asset_name} ({asset_tag}). "
            "Triggered by: {trigger_type}. Investigate and repair as needed."
        ),
        default_tasks=[
            MaintenanceTask(
                sequence=1,
                task_type=TaskType.INSPECTION,
                description="Diagnose fault condition",
                instructions="Review alarm history and perform diagnostic checks",
                estimated_duration_minutes=30,
            ),
            MaintenanceTask(
                sequence=2,
                task_type=TaskType.REPAIR,
                description="Repair or replace faulty components",
                instructions="Follow manufacturer service procedures",
                estimated_duration_minutes=60,
            ),
            MaintenanceTask(
                sequence=3,
                task_type=TaskType.CALIBRATION,
                description="Recalibrate after repair",
                instructions="Perform full calibration after any repair",
                estimated_duration_minutes=30,
            ),
        ],
        applicable_asset_types=[AssetType.ANALYZER],
        applicable_triggers=["analyzer_fault", "sensor_failure", "comm_failure"],
    )


def get_reagent_low_template() -> WorkOrderTemplate:
    """Get reagent low work order template."""
    return WorkOrderTemplate(
        template_id="reagent_low",
        name="Reagent Replenishment",
        work_order_type=WorkOrderType.PREVENTIVE,
        default_priority=WorkOrderPriority.MEDIUM,
        title_template="{asset_name} Reagent Low - Replenishment Required",
        description_template=(
            "Reagent level low on {asset_name} ({asset_tag}). "
            "Replenish reagents to ensure continuous operation."
        ),
        default_tasks=[
            MaintenanceTask(
                sequence=1,
                task_type=TaskType.INSPECTION,
                description="Check reagent containers",
                instructions="Inspect all reagent bottles for level and condition",
                estimated_duration_minutes=10,
            ),
            MaintenanceTask(
                sequence=2,
                task_type=TaskType.REPLACEMENT,
                description="Replace reagents",
                instructions="Replace with fresh reagents, document lot numbers",
                estimated_duration_minutes=20,
            ),
            MaintenanceTask(
                sequence=3,
                task_type=TaskType.TESTING,
                description="Verify analyzer operation",
                instructions="Run test sample to verify proper operation",
                estimated_duration_minutes=15,
            ),
        ],
        applicable_asset_types=[AssetType.ANALYZER],
        applicable_triggers=["reagent_low", "reagent_empty"],
    )


def get_pump_mismatch_template() -> WorkOrderTemplate:
    """Get pump output mismatch work order template."""
    return WorkOrderTemplate(
        template_id="pump_mismatch",
        name="Pump Output Mismatch Investigation",
        work_order_type=WorkOrderType.CORRECTIVE,
        default_priority=WorkOrderPriority.HIGH,
        title_template="{asset_name} Output Mismatch - Investigation Required",
        description_template=(
            "Pump output does not match setpoint on {asset_name} ({asset_tag}). "
            "Investigate pump, tubing, and controls."
        ),
        default_tasks=[
            MaintenanceTask(
                sequence=1,
                task_type=TaskType.INSPECTION,
                description="Inspect pump and tubing",
                instructions="Check for leaks, blockages, and worn tubing",
                estimated_duration_minutes=20,
            ),
            MaintenanceTask(
                sequence=2,
                task_type=TaskType.TESTING,
                description="Verify pump calibration",
                instructions="Perform volumetric test at multiple speeds",
                estimated_duration_minutes=30,
            ),
            MaintenanceTask(
                sequence=3,
                task_type=TaskType.ADJUSTMENT,
                description="Adjust or recalibrate pump",
                instructions="Recalibrate pump output per procedure",
                estimated_duration_minutes=20,
            ),
        ],
        applicable_asset_types=[AssetType.PUMP],
        applicable_triggers=["pump_mismatch", "flow_deviation", "output_error"],
    )
