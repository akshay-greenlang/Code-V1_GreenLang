# -*- coding: utf-8 -*-
"""
MaintenanceSystemConnector for GL-008 TRAPCATCHER

Provides integration with CMMS (Computerized Maintenance Management System)
and EAM (Enterprise Asset Management) systems for work order generation,
asset tracking, and maintenance history.

Supported Systems:
- SAP PM (Plant Maintenance)
- IBM Maximo
- Infor EAM
- eMaint CMMS
- Fiix CMMS
- Generic REST API systems

Features:
- Work order creation and updates
- Asset hierarchy management
- Parts/inventory queries
- Labor scheduling
- Maintenance history retrieval
- Compliance documentation

Author: GL-DataIntegrationEngineer
Date: December 2025
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class CMMSType(str, Enum):
    """Supported CMMS/EAM system types."""
    SAP_PM = "sap_pm"
    IBM_MAXIMO = "ibm_maximo"
    INFOR_EAM = "infor_eam"
    EMAINT = "emaint"
    FIIX = "fiix"
    GENERIC_REST = "generic_rest"


class WorkOrderStatus(str, Enum):
    """Work order status states."""
    DRAFT = "draft"
    OPEN = "open"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class WorkOrderPriority(str, Enum):
    """Work order priority levels."""
    EMERGENCY = "emergency"  # P0: Immediate
    URGENT = "urgent"  # P1: Same day
    HIGH = "high"  # P2: Within 24 hours
    MEDIUM = "medium"  # P3: Within 7 days
    LOW = "low"  # P4: Within 30 days
    ROUTINE = "routine"  # P5: Scheduled maintenance


class WorkOrderType(str, Enum):
    """Work order types."""
    CORRECTIVE = "corrective"  # Repair/fix
    PREVENTIVE = "preventive"  # Scheduled PM
    PREDICTIVE = "predictive"  # Condition-based
    INSPECTION = "inspection"  # Inspection/testing
    MODIFICATION = "modification"  # Changes/upgrades
    EMERGENCY = "emergency"  # Emergency repair


class ConnectionState(str, Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MaintenanceSystemConfig:
    """
    Configuration for maintenance system connector.

    Attributes:
        connector_id: Unique connector identifier
        connector_name: Human-readable name
        cmms_type: Type of CMMS system
        base_url: API base URL
        api_key: API authentication key
        client_id: OAuth client ID (for SAP/Maximo)
        plant_code: Plant/facility code
        default_priority: Default work order priority
        auto_approve: Auto-approve work orders below threshold
        approval_threshold_usd: Cost threshold for auto-approval
    """
    connector_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connector_name: str = "MaintenanceSystemConnector"
    cmms_type: CMMSType = CMMSType.GENERIC_REST
    base_url: str = "https://cmms.example.com/api"
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    plant_code: str = "PLANT01"
    default_priority: WorkOrderPriority = WorkOrderPriority.MEDIUM
    auto_approve: bool = False
    approval_threshold_usd: float = 1000.0
    timeout_seconds: float = 30.0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "connector_id": self.connector_id,
            "connector_name": self.connector_name,
            "cmms_type": self.cmms_type.value,
            "base_url": self.base_url,
            "plant_code": self.plant_code,
            "auto_approve": self.auto_approve,
        }


@dataclass
class WorkOrderData:
    """
    Work order data structure.

    Standard work order format compatible with major CMMS systems.
    """
    work_order_id: str
    work_order_number: str
    trap_id: str
    trap_tag: str
    asset_id: str
    location: str
    title: str
    description: str
    work_type: WorkOrderType
    priority: WorkOrderPriority
    status: WorkOrderStatus

    # Timing
    created_at: datetime
    due_date: datetime
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None

    # Assignment
    assigned_to: Optional[str] = None
    assigned_craft: str = "Pipefitter"
    crew_size: int = 1

    # Estimates
    estimated_hours: float = 2.0
    actual_hours: float = 0.0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0

    # Parts
    parts_required: List[str] = field(default_factory=list)
    parts_issued: List[str] = field(default_factory=list)

    # Safety
    safety_requirements: List[str] = field(default_factory=list)
    permits_required: List[str] = field(default_factory=list)

    # Documentation
    failure_code: str = ""
    cause_code: str = ""
    action_code: str = ""
    notes: str = ""

    # References
    parent_wo_id: Optional[str] = None
    reference_documents: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API submission."""
        return {
            "work_order_id": self.work_order_id,
            "work_order_number": self.work_order_number,
            "trap_id": self.trap_id,
            "trap_tag": self.trap_tag,
            "asset_id": self.asset_id,
            "location": self.location,
            "title": self.title,
            "description": self.description,
            "work_type": self.work_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat(),
            "scheduled_start": (
                self.scheduled_start.isoformat()
                if self.scheduled_start else None
            ),
            "scheduled_end": (
                self.scheduled_end.isoformat()
                if self.scheduled_end else None
            ),
            "assigned_to": self.assigned_to,
            "assigned_craft": self.assigned_craft,
            "crew_size": self.crew_size,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "estimated_cost_usd": self.estimated_cost_usd,
            "actual_cost_usd": self.actual_cost_usd,
            "parts_required": self.parts_required,
            "parts_issued": self.parts_issued,
            "safety_requirements": self.safety_requirements,
            "permits_required": self.permits_required,
            "failure_code": self.failure_code,
            "cause_code": self.cause_code,
            "action_code": self.action_code,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkOrderData":
        """Create WorkOrderData from dictionary."""
        # Parse datetime fields
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        due_date = data.get("due_date")
        if isinstance(due_date, str):
            due_date = datetime.fromisoformat(due_date)
        elif due_date is None:
            due_date = datetime.now(timezone.utc) + timedelta(days=7)

        return cls(
            work_order_id=data.get("work_order_id", str(uuid.uuid4())),
            work_order_number=data.get("work_order_number", ""),
            trap_id=data.get("trap_id", ""),
            trap_tag=data.get("trap_tag", ""),
            asset_id=data.get("asset_id", ""),
            location=data.get("location", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            work_type=WorkOrderType(data.get("work_type", "corrective")),
            priority=WorkOrderPriority(data.get("priority", "medium")),
            status=WorkOrderStatus(data.get("status", "open")),
            created_at=created_at,
            due_date=due_date,
            estimated_hours=data.get("estimated_hours", 2.0),
            estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
            parts_required=data.get("parts_required", []),
            safety_requirements=data.get("safety_requirements", []),
        )


@dataclass
class AssetData:
    """
    Asset/equipment data from CMMS.

    Represents a steam trap or related equipment.
    """
    asset_id: str
    asset_tag: str
    description: str
    asset_class: str
    location: str
    parent_asset_id: Optional[str]
    manufacturer: str
    model: str
    serial_number: str
    install_date: Optional[datetime]
    warranty_end_date: Optional[datetime]
    criticality: str
    status: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "asset_id": self.asset_id,
            "asset_tag": self.asset_tag,
            "description": self.description,
            "asset_class": self.asset_class,
            "location": self.location,
            "parent_asset_id": self.parent_asset_id,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "install_date": (
                self.install_date.isoformat()
                if self.install_date else None
            ),
            "criticality": self.criticality,
            "status": self.status,
            "attributes": self.attributes,
        }


@dataclass
class MaintenanceHistory:
    """
    Maintenance history record.

    Historical work order and maintenance activity.
    """
    record_id: str
    asset_id: str
    work_order_id: str
    work_type: str
    completion_date: datetime
    description: str
    failure_code: str
    cause_code: str
    action_code: str
    total_hours: float
    total_cost_usd: float
    technician: str
    parts_used: List[str]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "asset_id": self.asset_id,
            "work_order_id": self.work_order_id,
            "work_type": self.work_type,
            "completion_date": self.completion_date.isoformat(),
            "description": self.description,
            "failure_code": self.failure_code,
            "cause_code": self.cause_code,
            "action_code": self.action_code,
            "total_hours": self.total_hours,
            "total_cost_usd": self.total_cost_usd,
            "technician": self.technician,
            "parts_used": self.parts_used,
            "notes": self.notes,
        }


@dataclass
class CreateWorkOrderResult:
    """Result of work order creation."""
    success: bool
    work_order_id: str
    work_order_number: str
    message: str
    cmms_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "work_order_id": self.work_order_id,
            "work_order_number": self.work_order_number,
            "message": self.message,
        }


# ============================================================================
# MAINTENANCE SYSTEM CONNECTOR
# ============================================================================

class MaintenanceSystemConnector:
    """
    Connector for CMMS/EAM system integration.

    Provides work order management, asset tracking, and maintenance
    history integration with enterprise maintenance systems.

    Features:
    - Work order creation and lifecycle management
    - Asset hierarchy queries
    - Maintenance history retrieval
    - Parts/inventory queries
    - Failure coding and analytics

    Example:
        >>> config = MaintenanceSystemConfig(
        ...     cmms_type=CMMSType.IBM_MAXIMO,
        ...     base_url="https://maximo.example.com/api"
        ... )
        >>> connector = MaintenanceSystemConnector(config)
        >>> await connector.connect()
        >>> result = await connector.create_work_order(work_order_data)
    """

    # Failure codes for steam traps
    FAILURE_CODES = {
        "seat_erosion": "STTRAP-001",
        "disc_worn": "STTRAP-002",
        "bellows_rupture": "STTRAP-003",
        "dirt_blocked": "STTRAP-004",
        "corrosion": "STTRAP-005",
        "valve_stuck": "STTRAP-006",
        "unknown": "STTRAP-999",
    }

    # Cause codes
    CAUSE_CODES = {
        "wear": "CAUSE-001",
        "contamination": "CAUSE-002",
        "overpressure": "CAUSE-003",
        "corrosion": "CAUSE-004",
        "improper_install": "CAUSE-005",
        "age": "CAUSE-006",
        "unknown": "CAUSE-999",
    }

    # Action codes
    ACTION_CODES = {
        "replace": "ACT-001",
        "repair": "ACT-002",
        "clean": "ACT-003",
        "adjust": "ACT-004",
        "inspect": "ACT-005",
        "no_action": "ACT-006",
    }

    def __init__(self, config: MaintenanceSystemConfig):
        """
        Initialize maintenance system connector.

        Args:
            config: Connector configuration
        """
        self.config = config
        self._state = ConnectionState.DISCONNECTED
        self._session: Optional[Any] = None

        # Work order tracking
        self._created_work_orders: List[str] = []
        self._work_order_cache: Dict[str, WorkOrderData] = {}

        # Metrics
        self._request_count = 0
        self._error_count = 0

        logger.info(
            f"MaintenanceSystemConnector initialized: "
            f"{config.cmms_type.value} at {config.base_url}"
        )

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    async def connect(self) -> bool:
        """
        Establish connection to CMMS system.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        if self._state == ConnectionState.CONNECTED:
            logger.warning("Already connected to CMMS")
            return True

        self._state = ConnectionState.CONNECTING
        logger.info(f"Connecting to {self.config.cmms_type.value} CMMS")

        try:
            # In production: establish actual API session
            # For testing: simulate connection

            self._session = {
                "type": self.config.cmms_type.value,
                "base_url": self.config.base_url,
                "connected": True,
                "authenticated": bool(self.config.api_key),
            }

            self._state = ConnectionState.CONNECTED
            logger.info("Successfully connected to CMMS")
            return True

        except Exception as e:
            self._state = ConnectionState.ERROR
            logger.error(f"Failed to connect to CMMS: {e}")
            raise ConnectionError(f"CMMS connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from CMMS system."""
        logger.info("Disconnecting from CMMS")
        self._session = None
        self._state = ConnectionState.DISCONNECTED

    async def create_work_order(
        self,
        work_order: WorkOrderData
    ) -> CreateWorkOrderResult:
        """
        Create a new work order in CMMS.

        Args:
            work_order: Work order data

        Returns:
            CreateWorkOrderResult with status and IDs

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to CMMS")

        self._request_count += 1
        logger.info(
            f"Creating work order for trap {work_order.trap_tag}: "
            f"{work_order.title}"
        )

        try:
            # Generate work order number if not provided
            if not work_order.work_order_number:
                work_order.work_order_number = self._generate_wo_number()

            # In production: call CMMS API
            # For testing: simulate creation

            # Check auto-approval
            if (self.config.auto_approve and
                work_order.estimated_cost_usd <= self.config.approval_threshold_usd):
                work_order.status = WorkOrderStatus.APPROVED

            # Store in cache
            self._work_order_cache[work_order.work_order_id] = work_order
            self._created_work_orders.append(work_order.work_order_id)

            logger.info(
                f"Work order created: {work_order.work_order_number} "
                f"(status={work_order.status.value})"
            )

            return CreateWorkOrderResult(
                success=True,
                work_order_id=work_order.work_order_id,
                work_order_number=work_order.work_order_number,
                message="Work order created successfully",
                cmms_response={"id": work_order.work_order_id},
            )

        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to create work order: {e}")
            return CreateWorkOrderResult(
                success=False,
                work_order_id="",
                work_order_number="",
                message=f"Failed to create work order: {e}",
            )

    def _generate_wo_number(self) -> str:
        """Generate unique work order number."""
        timestamp = datetime.now().strftime("%Y%m%d")
        sequence = len(self._created_work_orders) + 1
        return f"WO-{self.config.plant_code}-{timestamp}-{sequence:04d}"

    async def update_work_order(
        self,
        work_order_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update an existing work order.

        Args:
            work_order_id: Work order ID to update
            updates: Dictionary of fields to update

        Returns:
            True if update successful
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to CMMS")

        self._request_count += 1

        if work_order_id not in self._work_order_cache:
            logger.warning(f"Work order not found: {work_order_id}")
            return False

        try:
            wo = self._work_order_cache[work_order_id]

            # Apply updates
            for key, value in updates.items():
                if hasattr(wo, key):
                    if key == "status":
                        value = WorkOrderStatus(value)
                    elif key == "priority":
                        value = WorkOrderPriority(value)
                    setattr(wo, key, value)

            logger.info(f"Work order updated: {work_order_id}")
            return True

        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to update work order: {e}")
            return False

    async def get_work_order(
        self,
        work_order_id: str
    ) -> Optional[WorkOrderData]:
        """
        Retrieve work order by ID.

        Args:
            work_order_id: Work order ID

        Returns:
            WorkOrderData or None if not found
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to CMMS")

        self._request_count += 1
        return self._work_order_cache.get(work_order_id)

    async def get_asset_work_orders(
        self,
        asset_id: str,
        status_filter: Optional[List[WorkOrderStatus]] = None,
        limit: int = 100
    ) -> List[WorkOrderData]:
        """
        Get work orders for an asset.

        Args:
            asset_id: Asset/trap ID
            status_filter: Optional status filter
            limit: Maximum results

        Returns:
            List of WorkOrderData
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to CMMS")

        self._request_count += 1

        # Filter from cache
        results = []
        for wo in self._work_order_cache.values():
            if wo.trap_id == asset_id or wo.asset_id == asset_id:
                if status_filter is None or wo.status in status_filter:
                    results.append(wo)
                    if len(results) >= limit:
                        break

        return results

    async def get_asset(self, asset_id: str) -> Optional[AssetData]:
        """
        Get asset information from CMMS.

        Args:
            asset_id: Asset identifier

        Returns:
            AssetData or None if not found
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to CMMS")

        self._request_count += 1

        # In production: query CMMS
        # For testing: generate simulated asset
        import random
        random.seed(hash(asset_id))

        return AssetData(
            asset_id=asset_id,
            asset_tag=f"ST-{asset_id}",
            description=f"Steam Trap {asset_id}",
            asset_class="STEAM_TRAP",
            location=f"Building A, Line {(hash(asset_id) % 5) + 1}",
            parent_asset_id=None,
            manufacturer=random.choice(["Armstrong", "Spirax Sarco", "TLV"]),
            model=random.choice(["CD-30", "FT-14", "J3X"]),
            serial_number=f"SN{random.randint(100000, 999999)}",
            install_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
            warranty_end_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            criticality=random.choice(["critical", "high", "medium", "low"]),
            status="active",
            attributes={
                "trap_type": random.choice(["disc", "inverted_bucket", "float"]),
                "size_inches": random.choice([0.5, 0.75, 1.0, 1.5]),
                "pressure_rating_bar": 16.0,
            },
        )

    async def get_maintenance_history(
        self,
        asset_id: str,
        months: int = 24
    ) -> List[MaintenanceHistory]:
        """
        Get maintenance history for an asset.

        Args:
            asset_id: Asset identifier
            months: History period in months

        Returns:
            List of MaintenanceHistory records
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to CMMS")

        self._request_count += 1

        # In production: query CMMS historian
        # For testing: generate simulated history
        import random
        random.seed(hash(asset_id))

        history = []
        current_date = datetime.now(timezone.utc)

        # Generate 0-5 historical records
        num_records = random.randint(0, 5)
        for i in range(num_records):
            days_ago = random.randint(30, months * 30)
            completion_date = current_date - timedelta(days=days_ago)

            record = MaintenanceHistory(
                record_id=f"MH-{asset_id}-{i:03d}",
                asset_id=asset_id,
                work_order_id=f"WO-{random.randint(10000, 99999)}",
                work_type=random.choice(["corrective", "preventive", "inspection"]),
                completion_date=completion_date,
                description=random.choice([
                    "Replaced steam trap",
                    "Cleaned trap internals",
                    "Adjusted valve",
                    "Inspection - no action",
                    "Replaced gaskets",
                ]),
                failure_code=random.choice(list(self.FAILURE_CODES.values())),
                cause_code=random.choice(list(self.CAUSE_CODES.values())),
                action_code=random.choice(list(self.ACTION_CODES.values())),
                total_hours=random.uniform(0.5, 4.0),
                total_cost_usd=random.uniform(100, 500),
                technician=f"Tech-{random.randint(1, 10):03d}",
                parts_used=random.sample(
                    ["Trap assembly", "Gasket kit", "Screen", "Valve disc"],
                    k=random.randint(0, 2)
                ),
                notes="Maintenance completed per standard procedure",
            )
            history.append(record)

        # Sort by date descending
        history.sort(key=lambda x: x.completion_date, reverse=True)

        return history

    async def close_work_order(
        self,
        work_order_id: str,
        completion_data: Dict[str, Any]
    ) -> bool:
        """
        Close a completed work order.

        Args:
            work_order_id: Work order ID
            completion_data: Completion details

        Returns:
            True if closed successfully
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to CMMS")

        self._request_count += 1

        if work_order_id not in self._work_order_cache:
            logger.warning(f"Work order not found: {work_order_id}")
            return False

        try:
            wo = self._work_order_cache[work_order_id]

            # Update completion data
            wo.status = WorkOrderStatus.COMPLETED
            wo.actual_end = datetime.now(timezone.utc)
            wo.actual_hours = completion_data.get("actual_hours", wo.estimated_hours)
            wo.actual_cost_usd = completion_data.get("actual_cost_usd", wo.estimated_cost_usd)
            wo.notes = completion_data.get("notes", "")

            if "failure_code" in completion_data:
                wo.failure_code = completion_data["failure_code"]
            if "cause_code" in completion_data:
                wo.cause_code = completion_data["cause_code"]
            if "action_code" in completion_data:
                wo.action_code = completion_data["action_code"]

            logger.info(f"Work order closed: {work_order_id}")
            return True

        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to close work order: {e}")
            return False

    def get_failure_code(self, failure_mode: str) -> str:
        """
        Get CMMS failure code from failure mode.

        Args:
            failure_mode: Failure mode string

        Returns:
            CMMS failure code
        """
        return self.FAILURE_CODES.get(failure_mode.lower(), self.FAILURE_CODES["unknown"])

    def get_cause_code(self, cause: str) -> str:
        """
        Get CMMS cause code.

        Args:
            cause: Cause string

        Returns:
            CMMS cause code
        """
        return self.CAUSE_CODES.get(cause.lower(), self.CAUSE_CODES["unknown"])

    def get_action_code(self, action: str) -> str:
        """
        Get CMMS action code.

        Args:
            action: Action string

        Returns:
            CMMS action code
        """
        return self.ACTION_CODES.get(action.lower(), self.ACTION_CODES["no_action"])

    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics."""
        return {
            "connector_id": self.config.connector_id,
            "cmms_type": self.config.cmms_type.value,
            "state": self._state.value,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "work_orders_created": len(self._created_work_orders),
            "cached_work_orders": len(self._work_order_cache),
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_maintenance_connector(
    cmms_type: CMMSType = CMMSType.GENERIC_REST,
    base_url: str = "https://cmms.example.com/api",
    plant_code: str = "PLANT01",
    **kwargs
) -> MaintenanceSystemConnector:
    """
    Factory function to create MaintenanceSystemConnector.

    Args:
        cmms_type: Type of CMMS system
        base_url: API base URL
        plant_code: Plant/facility code
        **kwargs: Additional configuration options

    Returns:
        Configured MaintenanceSystemConnector
    """
    config = MaintenanceSystemConfig(
        cmms_type=cmms_type,
        base_url=base_url,
        plant_code=plant_code,
        **kwargs
    )
    return MaintenanceSystemConnector(config)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "MaintenanceSystemConnector",
    "MaintenanceSystemConfig",
    "WorkOrderData",
    "WorkOrderStatus",
    "WorkOrderPriority",
    "WorkOrderType",
    "AssetData",
    "MaintenanceHistory",
    "CreateWorkOrderResult",
    "CMMSType",
    "ConnectionState",
    "create_maintenance_connector",
]
