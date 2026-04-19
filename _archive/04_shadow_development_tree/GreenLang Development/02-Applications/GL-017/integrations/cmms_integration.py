"""
GL-017 CONDENSYNC CMMS Integration Module

Computerized Maintenance Management System integration for work order
management, equipment history, maintenance scheduling, and reliability tracking.

Features:
- Work order creation for tube cleaning
- Equipment history retrieval
- Maintenance scheduling
- Spare parts inventory check
- Reliability metrics tracking

Author: GreenLang AI Platform
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class CMMSError(Exception):
    """Base exception for CMMS integration."""
    pass


class CMMSConnectionError(CMMSError):
    """Raised when CMMS connection fails."""
    pass


class CMMSWorkOrderError(CMMSError):
    """Raised when work order operation fails."""
    pass


# =============================================================================
# Enums
# =============================================================================

class WorkOrderPriority(Enum):
    """Work order priority levels."""
    EMERGENCY = 1
    URGENT = 2
    HIGH = 3
    MEDIUM = 4
    LOW = 5
    PLANNED = 6


class WorkOrderStatus(Enum):
    """Work order status."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class WorkOrderType(Enum):
    """Work order types."""
    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"
    INSPECTION = "inspection"
    CLEANING = "cleaning"
    CALIBRATION = "calibration"


class MaintenanceType(Enum):
    """Maintenance activity types."""
    TUBE_CLEANING = "tube_cleaning"
    TUBE_PLUGGING = "tube_plugging"
    TUBE_REPLACEMENT = "tube_replacement"
    VALVE_MAINTENANCE = "valve_maintenance"
    PUMP_MAINTENANCE = "pump_maintenance"
    GASKET_REPLACEMENT = "gasket_replacement"
    INSPECTION = "inspection"
    EDDY_CURRENT_TEST = "eddy_current_test"


# =============================================================================
# Data Models
# =============================================================================

class CMMSConfig(BaseModel):
    """Configuration for CMMS integration."""

    api_base_url: str = Field(
        ...,
        description="CMMS API base URL"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication"
    )
    username: Optional[str] = Field(
        default=None,
        description="Username for authentication"
    )
    password: Optional[str] = Field(
        default=None,
        description="Password for authentication"
    )
    connection_timeout: float = Field(
        default=10.0,
        description="Connection timeout in seconds"
    )
    request_timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds"
    )
    plant_id: str = Field(
        default="PLANT-001",
        description="Plant identifier"
    )
    area_id: str = Field(
        default="CONDENSER",
        description="Area identifier"
    )
    default_cost_center: str = Field(
        default="CC-MAINT-001",
        description="Default cost center for work orders"
    )


@dataclass
class WorkOrder:
    """Work order data model."""

    work_order_id: str
    title: str
    description: str
    equipment_id: str
    equipment_name: str
    work_order_type: WorkOrderType
    priority: WorkOrderPriority
    status: WorkOrderStatus

    # Dates
    created_date: datetime
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None

    # Assignment
    assigned_to: Optional[str] = None
    assigned_crew: Optional[str] = None
    supervisor: Optional[str] = None

    # Cost tracking
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    cost_center: Optional[str] = None

    # Additional info
    failure_code: Optional[str] = None
    cause_code: Optional[str] = None
    remedy_code: Optional[str] = None
    notes: str = ""
    attachments: List[str] = field(default_factory=list)

    # Related work orders
    parent_wo_id: Optional[str] = None
    child_wo_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "work_order_id": self.work_order_id,
            "title": self.title,
            "description": self.description,
            "equipment_id": self.equipment_id,
            "equipment_name": self.equipment_name,
            "work_order_type": self.work_order_type.value,
            "priority": self.priority.name,
            "status": self.status.value,
            "created_date": self.created_date.isoformat(),
            "scheduled_start": self.scheduled_start.isoformat() if self.scheduled_start else None,
            "scheduled_end": self.scheduled_end.isoformat() if self.scheduled_end else None,
            "actual_start": self.actual_start.isoformat() if self.actual_start else None,
            "actual_end": self.actual_end.isoformat() if self.actual_end else None,
            "assigned_to": self.assigned_to,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "notes": self.notes
        }


@dataclass
class EquipmentHistory:
    """Equipment maintenance history."""

    equipment_id: str
    equipment_name: str
    equipment_type: str
    install_date: datetime
    manufacturer: str
    model: str
    serial_number: str

    # Maintenance records
    total_work_orders: int = 0
    completed_work_orders: int = 0
    corrective_actions: int = 0
    preventive_actions: int = 0

    # Reliability metrics
    mtbf_hours: float = 0.0  # Mean Time Between Failures
    mttr_hours: float = 0.0  # Mean Time To Repair
    availability_percent: float = 100.0
    total_downtime_hours: float = 0.0

    # Recent history
    last_maintenance_date: Optional[datetime] = None
    last_failure_date: Optional[datetime] = None
    next_scheduled_maintenance: Optional[datetime] = None

    # Cost tracking
    total_maintenance_cost: float = 0.0
    ytd_maintenance_cost: float = 0.0

    work_order_history: List[WorkOrder] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "equipment_id": self.equipment_id,
            "equipment_name": self.equipment_name,
            "equipment_type": self.equipment_type,
            "install_date": self.install_date.isoformat(),
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "total_work_orders": self.total_work_orders,
            "mtbf_hours": self.mtbf_hours,
            "mttr_hours": self.mttr_hours,
            "availability_percent": self.availability_percent,
            "last_maintenance_date": (
                self.last_maintenance_date.isoformat()
                if self.last_maintenance_date else None
            ),
            "next_scheduled_maintenance": (
                self.next_scheduled_maintenance.isoformat()
                if self.next_scheduled_maintenance else None
            ),
            "total_maintenance_cost": self.total_maintenance_cost
        }


@dataclass
class MaintenanceSchedule:
    """Scheduled maintenance activity."""

    schedule_id: str
    equipment_id: str
    equipment_name: str
    maintenance_type: MaintenanceType
    description: str
    frequency_days: int
    last_performed: Optional[datetime]
    next_due: datetime
    estimated_duration_hours: float
    required_skills: List[str] = field(default_factory=list)
    required_parts: List[str] = field(default_factory=list)
    is_overdue: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schedule_id": self.schedule_id,
            "equipment_id": self.equipment_id,
            "equipment_name": self.equipment_name,
            "maintenance_type": self.maintenance_type.value,
            "description": self.description,
            "frequency_days": self.frequency_days,
            "last_performed": (
                self.last_performed.isoformat() if self.last_performed else None
            ),
            "next_due": self.next_due.isoformat(),
            "estimated_duration_hours": self.estimated_duration_hours,
            "is_overdue": self.is_overdue
        }


@dataclass
class SparePartStatus:
    """Spare part inventory status."""

    part_number: str
    description: str
    quantity_on_hand: int
    quantity_reserved: int
    quantity_available: int
    reorder_point: int
    reorder_quantity: int
    unit_cost: float
    lead_time_days: int
    location: str
    last_used: Optional[datetime] = None
    associated_equipment: List[str] = field(default_factory=list)

    def needs_reorder(self) -> bool:
        """Check if part needs to be reordered."""
        return self.quantity_available <= self.reorder_point

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "part_number": self.part_number,
            "description": self.description,
            "quantity_on_hand": self.quantity_on_hand,
            "quantity_available": self.quantity_available,
            "reorder_point": self.reorder_point,
            "unit_cost": self.unit_cost,
            "lead_time_days": self.lead_time_days,
            "needs_reorder": self.needs_reorder()
        }


# =============================================================================
# CMMS Integration Class
# =============================================================================

class CMMSIntegration:
    """
    CMMS integration for maintenance management.

    Provides:
    - Work order creation and management
    - Equipment history retrieval
    - Maintenance scheduling
    - Spare parts inventory check
    - Reliability metrics tracking
    """

    def __init__(self, config: CMMSConfig):
        """
        Initialize CMMS integration.

        Args:
            config: CMMS configuration
        """
        self.config = config

        self._client = None
        self._connected = False

        # Caches
        self._work_orders: Dict[str, WorkOrder] = {}
        self._equipment_history: Dict[str, EquipmentHistory] = {}
        self._schedules: Dict[str, MaintenanceSchedule] = {}
        self._spare_parts: Dict[str, SparePartStatus] = {}

        # Statistics
        self._stats = {
            "work_orders_created": 0,
            "work_orders_completed": 0,
            "queries_total": 0,
            "last_sync": None
        }

        logger.info(f"CMMS Integration initialized for {config.api_base_url}")

    @property
    def is_connected(self) -> bool:
        """Check if connected to CMMS."""
        return self._connected

    async def connect(self) -> None:
        """
        Establish connection to CMMS.

        Raises:
            CMMSConnectionError: If connection fails
        """
        logger.info(f"Connecting to CMMS at {self.config.api_base_url}")

        try:
            await self._create_client()
            await self._load_initial_data()

            self._connected = True
            logger.info("Successfully connected to CMMS")

        except Exception as e:
            logger.error(f"Failed to connect to CMMS: {e}")
            raise CMMSConnectionError(f"Connection failed: {e}")

    async def _create_client(self) -> None:
        """Create CMMS API client."""
        self._client = {
            "base_url": self.config.api_base_url,
            "connected": False
        }

        await asyncio.sleep(0.1)
        self._client["connected"] = True

    async def _load_initial_data(self) -> None:
        """Load initial data from CMMS."""
        # Load condenser equipment
        await self._load_condenser_equipment()
        # Load maintenance schedules
        await self._load_maintenance_schedules()
        # Load spare parts
        await self._load_spare_parts()

    async def _load_condenser_equipment(self) -> None:
        """Load condenser equipment data."""
        equipment_list = [
            ("EQ-COND-001", "Main Condenser", "Shell-and-Tube Condenser"),
            ("EQ-CWPUMP-001", "CW Pump A", "Centrifugal Pump"),
            ("EQ-CWPUMP-002", "CW Pump B", "Centrifugal Pump"),
            ("EQ-CPUMP-001", "Condensate Pump A", "Centrifugal Pump"),
            ("EQ-CPUMP-002", "Condensate Pump B", "Centrifugal Pump"),
            ("EQ-EJECT-001", "Air Ejector A", "Steam Jet Ejector"),
            ("EQ-EJECT-002", "Air Ejector B", "Steam Jet Ejector"),
        ]

        for eq_id, eq_name, eq_type in equipment_list:
            self._equipment_history[eq_id] = EquipmentHistory(
                equipment_id=eq_id,
                equipment_name=eq_name,
                equipment_type=eq_type,
                install_date=datetime(2018, 1, 1),
                manufacturer="Industrial Equipment Co",
                model="IEC-2000",
                serial_number=f"SN-{eq_id}",
                total_work_orders=25,
                completed_work_orders=24,
                mtbf_hours=4380.0,
                mttr_hours=8.0,
                availability_percent=99.5,
                last_maintenance_date=datetime.utcnow() - timedelta(days=30)
            )

    async def _load_maintenance_schedules(self) -> None:
        """Load maintenance schedules."""
        schedules = [
            MaintenanceSchedule(
                schedule_id="SCH-001",
                equipment_id="EQ-COND-001",
                equipment_name="Main Condenser",
                maintenance_type=MaintenanceType.TUBE_CLEANING,
                description="High pressure water cleaning of condenser tubes",
                frequency_days=180,
                last_performed=datetime.utcnow() - timedelta(days=90),
                next_due=datetime.utcnow() + timedelta(days=90),
                estimated_duration_hours=24.0,
                required_skills=["Mechanical", "Cleaning Specialist"],
                required_parts=["Cleaning nozzles", "Gaskets"]
            ),
            MaintenanceSchedule(
                schedule_id="SCH-002",
                equipment_id="EQ-COND-001",
                equipment_name="Main Condenser",
                maintenance_type=MaintenanceType.EDDY_CURRENT_TEST,
                description="Eddy current tube inspection",
                frequency_days=365,
                last_performed=datetime.utcnow() - timedelta(days=200),
                next_due=datetime.utcnow() + timedelta(days=165),
                estimated_duration_hours=48.0,
                required_skills=["NDT Specialist", "Mechanical"]
            ),
            MaintenanceSchedule(
                schedule_id="SCH-003",
                equipment_id="EQ-CWPUMP-001",
                equipment_name="CW Pump A",
                maintenance_type=MaintenanceType.PUMP_MAINTENANCE,
                description="Pump bearing inspection and lubrication",
                frequency_days=90,
                last_performed=datetime.utcnow() - timedelta(days=85),
                next_due=datetime.utcnow() + timedelta(days=5),
                estimated_duration_hours=4.0,
                required_skills=["Mechanical"]
            )
        ]

        for schedule in schedules:
            self._schedules[schedule.schedule_id] = schedule

    async def _load_spare_parts(self) -> None:
        """Load spare parts inventory."""
        parts = [
            SparePartStatus(
                part_number="SP-TUBE-001",
                description="Condenser Tube - Titanium Grade 2",
                quantity_on_hand=50,
                quantity_reserved=5,
                quantity_available=45,
                reorder_point=20,
                reorder_quantity=50,
                unit_cost=250.00,
                lead_time_days=60,
                location="Warehouse A-12",
                associated_equipment=["EQ-COND-001"]
            ),
            SparePartStatus(
                part_number="SP-GASKET-001",
                description="Waterbox Gasket - Main Condenser",
                quantity_on_hand=4,
                quantity_reserved=0,
                quantity_available=4,
                reorder_point=2,
                reorder_quantity=4,
                unit_cost=1500.00,
                lead_time_days=30,
                location="Warehouse A-05",
                associated_equipment=["EQ-COND-001"]
            ),
            SparePartStatus(
                part_number="SP-SEAL-001",
                description="Mechanical Seal - CW Pump",
                quantity_on_hand=2,
                quantity_reserved=0,
                quantity_available=2,
                reorder_point=2,
                reorder_quantity=2,
                unit_cost=3500.00,
                lead_time_days=45,
                location="Warehouse B-03",
                associated_equipment=["EQ-CWPUMP-001", "EQ-CWPUMP-002"]
            )
        ]

        for part in parts:
            self._spare_parts[part.part_number] = part

    async def disconnect(self) -> None:
        """Disconnect from CMMS."""
        logger.info("Disconnecting from CMMS")

        if self._client:
            self._client["connected"] = False
            self._client = None

        self._connected = False
        logger.info("Disconnected from CMMS")

    async def create_work_order(
        self,
        title: str,
        description: str,
        equipment_id: str,
        work_order_type: WorkOrderType,
        priority: WorkOrderPriority,
        scheduled_start: Optional[datetime] = None,
        estimated_hours: float = 0.0
    ) -> WorkOrder:
        """
        Create a new work order.

        Args:
            title: Work order title
            description: Detailed description
            equipment_id: Equipment identifier
            work_order_type: Type of work order
            priority: Priority level
            scheduled_start: Scheduled start date
            estimated_hours: Estimated hours to complete

        Returns:
            Created WorkOrder

        Raises:
            CMMSWorkOrderError: If creation fails
        """
        if not self._connected:
            raise CMMSWorkOrderError("Not connected to CMMS")

        equipment = self._equipment_history.get(equipment_id)
        if not equipment:
            raise CMMSWorkOrderError(f"Equipment not found: {equipment_id}")

        work_order_id = f"WO-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

        work_order = WorkOrder(
            work_order_id=work_order_id,
            title=title,
            description=description,
            equipment_id=equipment_id,
            equipment_name=equipment.equipment_name,
            work_order_type=work_order_type,
            priority=priority,
            status=WorkOrderStatus.PENDING_APPROVAL,
            created_date=datetime.utcnow(),
            scheduled_start=scheduled_start,
            estimated_hours=estimated_hours,
            cost_center=self.config.default_cost_center
        )

        self._work_orders[work_order_id] = work_order
        self._stats["work_orders_created"] += 1

        logger.info(f"Created work order: {work_order_id} - {title}")
        return work_order

    async def create_tube_cleaning_work_order(
        self,
        cleanliness_factor: float,
        ttd: float,
        urgency_score: float
    ) -> WorkOrder:
        """
        Create a tube cleaning work order based on performance metrics.

        Args:
            cleanliness_factor: Current cleanliness factor (0-1)
            ttd: Terminal temperature difference (degC)
            urgency_score: Calculated urgency (0-100)

        Returns:
            Created WorkOrder
        """
        # Determine priority based on urgency
        if urgency_score >= 80:
            priority = WorkOrderPriority.URGENT
        elif urgency_score >= 60:
            priority = WorkOrderPriority.HIGH
        elif urgency_score >= 40:
            priority = WorkOrderPriority.MEDIUM
        else:
            priority = WorkOrderPriority.LOW

        description = (
            f"Condenser tube cleaning required based on performance degradation.\n\n"
            f"Current Metrics:\n"
            f"- Cleanliness Factor: {cleanliness_factor:.2%}\n"
            f"- Terminal Temperature Difference: {ttd:.1f} degC\n"
            f"- Urgency Score: {urgency_score:.0f}/100\n\n"
            f"Recommended Action: High-pressure water cleaning of condenser tubes.\n"
            f"Isolation and lockout/tagout procedures required."
        )

        return await self.create_work_order(
            title="Condenser Tube Cleaning - Performance Degradation",
            description=description,
            equipment_id="EQ-COND-001",
            work_order_type=WorkOrderType.CLEANING,
            priority=priority,
            estimated_hours=24.0
        )

    async def get_work_order(self, work_order_id: str) -> Optional[WorkOrder]:
        """Get work order by ID."""
        self._stats["queries_total"] += 1
        return self._work_orders.get(work_order_id)

    async def get_equipment_history(
        self,
        equipment_id: str
    ) -> Optional[EquipmentHistory]:
        """
        Get equipment maintenance history.

        Args:
            equipment_id: Equipment identifier

        Returns:
            EquipmentHistory or None
        """
        self._stats["queries_total"] += 1
        return self._equipment_history.get(equipment_id)

    async def get_maintenance_schedule(
        self,
        equipment_id: Optional[str] = None
    ) -> List[MaintenanceSchedule]:
        """
        Get maintenance schedules.

        Args:
            equipment_id: Optional equipment filter

        Returns:
            List of MaintenanceSchedule
        """
        self._stats["queries_total"] += 1

        schedules = list(self._schedules.values())

        if equipment_id:
            schedules = [s for s in schedules if s.equipment_id == equipment_id]

        # Update overdue status
        now = datetime.utcnow()
        for schedule in schedules:
            schedule.is_overdue = schedule.next_due < now

        return sorted(schedules, key=lambda s: s.next_due)

    async def get_overdue_maintenance(self) -> List[MaintenanceSchedule]:
        """Get all overdue maintenance schedules."""
        schedules = await self.get_maintenance_schedule()
        return [s for s in schedules if s.is_overdue]

    async def check_spare_parts(
        self,
        part_numbers: List[str]
    ) -> Dict[str, SparePartStatus]:
        """
        Check spare parts availability.

        Args:
            part_numbers: List of part numbers to check

        Returns:
            Dictionary mapping part numbers to status
        """
        self._stats["queries_total"] += 1

        results = {}
        for part_number in part_numbers:
            part = self._spare_parts.get(part_number)
            if part:
                results[part_number] = part

        return results

    async def get_parts_needing_reorder(self) -> List[SparePartStatus]:
        """Get all parts that need to be reordered."""
        return [p for p in self._spare_parts.values() if p.needs_reorder()]

    async def get_reliability_metrics(
        self,
        equipment_id: str
    ) -> Dict[str, Any]:
        """
        Get reliability metrics for equipment.

        Args:
            equipment_id: Equipment identifier

        Returns:
            Reliability metrics dictionary
        """
        equipment = self._equipment_history.get(equipment_id)
        if not equipment:
            raise CMMSError(f"Equipment not found: {equipment_id}")

        return {
            "equipment_id": equipment_id,
            "equipment_name": equipment.equipment_name,
            "mtbf_hours": equipment.mtbf_hours,
            "mttr_hours": equipment.mttr_hours,
            "availability_percent": equipment.availability_percent,
            "total_downtime_hours": equipment.total_downtime_hours,
            "total_work_orders": equipment.total_work_orders,
            "corrective_ratio": (
                equipment.corrective_actions / equipment.total_work_orders
                if equipment.total_work_orders > 0 else 0
            ),
            "maintenance_cost_ytd": equipment.ytd_maintenance_cost,
            "last_failure": (
                equipment.last_failure_date.isoformat()
                if equipment.last_failure_date else None
            )
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "work_orders_count": len(self._work_orders),
            "equipment_count": len(self._equipment_history),
            "schedules_count": len(self._schedules),
            "spare_parts_count": len(self._spare_parts)
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "connected": self._connected,
            "timestamp": datetime.utcnow().isoformat()
        }

        if not self._connected:
            health["status"] = "unhealthy"
            health["reason"] = "Not connected to CMMS"
            return health

        # Check for overdue maintenance
        overdue = await self.get_overdue_maintenance()
        if overdue:
            health["status"] = "degraded"
            health["overdue_maintenance_count"] = len(overdue)

        # Check parts needing reorder
        reorder_parts = await self.get_parts_needing_reorder()
        if reorder_parts:
            health["parts_needing_reorder"] = len(reorder_parts)

        return health
