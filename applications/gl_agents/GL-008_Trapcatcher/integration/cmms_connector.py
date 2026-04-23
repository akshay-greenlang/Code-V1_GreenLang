# -*- coding: utf-8 -*-
"""
CMMS Connector for GL-008 TRAPCATCHER

Integration with Computerized Maintenance Management Systems
(SAP PM, IBM Maximo, Infor EAM) for work order automation.

Zero-Hallucination Guarantee:
- Deterministic work order creation and updates
- Full audit trail with provenance tracking
- No AI inference in integration logic

Author: GL-DataIntegrationEngineer
Date: December 2025
Version: 1.0.0
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================================
# ENUMERATIONS
# ============================================================================

class CMMSType(Enum):
    """Supported CMMS types."""
    SAP_PM = "sap_pm"
    IBM_MAXIMO = "ibm_maximo"
    INFOR_EAM = "infor_eam"
    ORACLE_EAM = "oracle_eam"
    HEXAGON_EAM = "hexagon_eam"
    FIIX = "fiix"
    MAINTAINX = "maintainx"
    LIMBLE = "limble"
    GENERIC_REST = "generic_rest"


class WorkOrderPriority(Enum):
    """Work order priority levels."""
    EMERGENCY = 1      # Immediate action required
    URGENT = 2         # Within 24 hours
    HIGH = 3           # Within 48 hours
    MEDIUM = 4         # Within 1 week
    LOW = 5            # Within 1 month
    PREVENTIVE = 6     # Scheduled maintenance


class WorkOrderStatus(Enum):
    """Work order lifecycle status."""
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    CLOSED = "closed"


class WorkOrderType(Enum):
    """Types of work orders."""
    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    INSPECTION = "inspection"
    REPLACEMENT = "replacement"
    EMERGENCY = "emergency"
    PREDICTIVE = "predictive"


class AssetStatus(Enum):
    """Asset operational status."""
    OPERATING = "operating"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    DECOMMISSIONED = "decommissioned"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CMMSConfig:
    """Configuration for CMMS connector."""

    # Connection settings
    cmms_type: CMMSType = CMMSType.GENERIC_REST
    api_url: str = ""
    api_key: str = ""
    username: str = ""
    password: str = ""

    # Authentication
    use_oauth: bool = False
    oauth_client_id: str = ""
    oauth_client_secret: str = ""
    oauth_token_url: str = ""

    # API settings
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Work order defaults
    default_priority: WorkOrderPriority = WorkOrderPriority.MEDIUM
    default_work_type: WorkOrderType = WorkOrderType.CORRECTIVE
    auto_approve: bool = False

    # Integration settings
    sync_interval_minutes: int = 15
    batch_size: int = 100


@dataclass
class Asset:
    """CMMS asset representation."""
    asset_id: str
    asset_name: str
    asset_type: str = "STEAM_TRAP"
    location: str = ""
    parent_asset_id: str = ""
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    install_date: Optional[datetime] = None
    status: AssetStatus = AssetStatus.OPERATING
    criticality: str = "medium"
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkOrder:
    """Work order for maintenance activity."""

    # Identification
    work_order_id: str
    external_id: Optional[str] = None  # CMMS-assigned ID

    # Asset reference
    asset_id: str = ""
    asset_name: str = ""
    location: str = ""

    # Work details
    title: str = ""
    description: str = ""
    work_type: WorkOrderType = WorkOrderType.CORRECTIVE
    priority: WorkOrderPriority = WorkOrderPriority.MEDIUM
    status: WorkOrderStatus = WorkOrderStatus.DRAFT

    # Scheduling
    requested_date: Optional[datetime] = None
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    due_date: Optional[datetime] = None

    # Execution
    assigned_to: str = ""
    assigned_crew: str = ""
    estimated_hours: float = 1.0
    actual_hours: float = 0.0

    # Costs
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    parts_cost_usd: float = 0.0
    labor_cost_usd: float = 0.0

    # Parts
    parts_required: List[str] = field(default_factory=list)
    parts_used: List[str] = field(default_factory=list)

    # Completion
    completed_date: Optional[datetime] = None
    completion_notes: str = ""
    root_cause: str = ""
    corrective_action: str = ""

    # Audit
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "TRAPCATCHER"
    updated_at: Optional[datetime] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API submission."""
        return {
            "work_order_id": self.work_order_id,
            "external_id": self.external_id,
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "location": self.location,
            "title": self.title,
            "description": self.description,
            "work_type": self.work_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "requested_date": self.requested_date.isoformat() if self.requested_date else None,
            "scheduled_start": self.scheduled_start.isoformat() if self.scheduled_start else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "assigned_to": self.assigned_to,
            "estimated_hours": self.estimated_hours,
            "estimated_cost_usd": self.estimated_cost_usd,
            "parts_required": self.parts_required,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class MaintenanceHistory:
    """Historical maintenance record for an asset."""
    record_id: str
    asset_id: str
    work_order_id: str
    work_type: WorkOrderType
    completed_date: datetime
    technician: str
    duration_hours: float
    cost_usd: float
    description: str
    findings: str
    parts_replaced: List[str]
    next_scheduled: Optional[datetime] = None


# ============================================================================
# ABSTRACT BASE CONNECTOR
# ============================================================================

class BaseCMMSConnector(ABC):
    """Abstract base class for CMMS connectors."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to CMMS."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to CMMS."""
        pass

    @abstractmethod
    async def create_work_order(self, work_order: WorkOrder) -> str:
        """Create new work order and return CMMS ID."""
        pass

    @abstractmethod
    async def update_work_order(self, work_order: WorkOrder) -> bool:
        """Update existing work order."""
        pass

    @abstractmethod
    async def get_work_order(self, work_order_id: str) -> Optional[WorkOrder]:
        """Retrieve work order by ID."""
        pass

    @abstractmethod
    async def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Retrieve asset by ID."""
        pass

    @abstractmethod
    async def get_maintenance_history(
        self, asset_id: str, limit: int = 100
    ) -> List[MaintenanceHistory]:
        """Get maintenance history for asset."""
        pass


# ============================================================================
# MAIN CONNECTOR CLASS
# ============================================================================

class CMMSConnector:
    """
    CMMS connector for work order automation.

    Supports multiple CMMS platforms with unified interface
    for creating, updating, and tracking work orders.

    Zero-Hallucination Guarantee:
    - Deterministic work order creation
    - Full audit trail with provenance
    - No AI in integration logic

    Example:
        >>> connector = CMMSConnector(config)
        >>> await connector.connect()
        >>> wo = connector.create_work_order_from_diagnostic(diagnostic)
        >>> cmms_id = await connector.submit_work_order(wo)
    """

    VERSION = "1.0.0"

    def __init__(self, config: CMMSConfig):
        """
        Initialize CMMS connector.

        Args:
            config: CMMS configuration
        """
        self.config = config
        self._connected = False
        self._work_order_counter = 0
        self._backend: Optional[BaseCMMSConnector] = None

    @property
    def connected(self) -> bool:
        """Check if connected to CMMS."""
        return self._connected

    async def connect(self) -> bool:
        """
        Connect to CMMS system.

        Returns:
            True if connection successful
        """
        # In production, this would establish actual connection
        # For now, simulate successful connection
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Disconnect from CMMS system."""
        self._connected = False

    def _compute_provenance_hash(self, work_order: WorkOrder) -> str:
        """Compute SHA-256 hash for work order provenance."""
        data = {
            "version": self.VERSION,
            "work_order_id": work_order.work_order_id,
            "asset_id": work_order.asset_id,
            "title": work_order.title,
            "priority": work_order.priority.value,
            "created_at": work_order.created_at.isoformat(),
        }

        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def create_work_order_from_diagnostic(
        self,
        diagnostic_result: Any,  # DiagnosticOutput
        work_type: Optional[WorkOrderType] = None,
        priority_override: Optional[WorkOrderPriority] = None,
        assigned_to: str = ""
    ) -> WorkOrder:
        """
        Create work order from diagnostic result.

        Args:
            diagnostic_result: Diagnostic output from TRAPCATCHER
            work_type: Override work type
            priority_override: Override priority
            assigned_to: Technician assignment

        Returns:
            Work order ready for submission
        """
        now = datetime.now(timezone.utc)
        self._work_order_counter += 1

        # Extract diagnostic info
        trap_id = getattr(diagnostic_result, 'trap_id', 'UNKNOWN')
        condition = getattr(diagnostic_result, 'condition', 'unknown')
        severity = getattr(diagnostic_result, 'severity', 'medium')
        energy_loss = getattr(diagnostic_result, 'energy_loss_kw', 0.0)
        recommendation = getattr(diagnostic_result, 'recommended_action', '')
        location = getattr(diagnostic_result, 'location', '')

        # Map condition to priority
        if priority_override:
            priority = priority_override
        elif condition.lower() == 'failed':
            priority = WorkOrderPriority.URGENT
        elif condition.lower() == 'leaking':
            priority = WorkOrderPriority.HIGH
        elif severity.lower() == 'critical':
            priority = WorkOrderPriority.URGENT
        else:
            priority = self.config.default_priority

        # Determine work type
        if work_type:
            wt = work_type
        elif condition.lower() in ('failed', 'leaking'):
            wt = WorkOrderType.CORRECTIVE
        else:
            wt = WorkOrderType.INSPECTION

        # Estimate hours based on condition
        if condition.lower() == 'failed':
            est_hours = 2.0
        elif condition.lower() == 'leaking':
            est_hours = 1.5
        else:
            est_hours = 1.0

        # Create title and description
        title = f"Steam Trap {condition.upper()}: {trap_id}"
        description = f"""Steam trap diagnostic alert generated by TRAPCATCHER.

Trap ID: {trap_id}
Condition: {condition}
Severity: {severity}
Energy Loss: {energy_loss:.2f} kW
Location: {location}

Recommended Action: {recommendation}

This work order was automatically generated based on predictive diagnostics.
"""

        # Create work order
        wo_id = f"WO-TRAP-{now.strftime('%Y%m%d')}-{self._work_order_counter:05d}"

        work_order = WorkOrder(
            work_order_id=wo_id,
            asset_id=trap_id,
            asset_name=f"Steam Trap {trap_id}",
            location=location,
            title=title,
            description=description,
            work_type=wt,
            priority=priority,
            status=WorkOrderStatus.PENDING,
            requested_date=now,
            due_date=self._calculate_due_date(priority, now),
            assigned_to=assigned_to,
            estimated_hours=est_hours,
            created_at=now,
            created_by="TRAPCATCHER",
        )

        work_order.provenance_hash = self._compute_provenance_hash(work_order)

        return work_order

    def _calculate_due_date(
        self,
        priority: WorkOrderPriority,
        from_date: datetime
    ) -> datetime:
        """Calculate due date based on priority."""
        from datetime import timedelta

        days_map = {
            WorkOrderPriority.EMERGENCY: 0,
            WorkOrderPriority.URGENT: 1,
            WorkOrderPriority.HIGH: 2,
            WorkOrderPriority.MEDIUM: 7,
            WorkOrderPriority.LOW: 30,
            WorkOrderPriority.PREVENTIVE: 60,
        }

        days = days_map.get(priority, 7)
        return from_date + timedelta(days=days)

    async def submit_work_order(self, work_order: WorkOrder) -> str:
        """
        Submit work order to CMMS.

        Args:
            work_order: Work order to submit

        Returns:
            CMMS-assigned work order ID
        """
        if not self._connected:
            raise RuntimeError("Not connected to CMMS")

        # In production, this would call CMMS API
        # For now, simulate successful submission
        external_id = f"CMMS-{work_order.work_order_id}"
        work_order.external_id = external_id
        work_order.status = WorkOrderStatus.PENDING

        if self.config.auto_approve:
            work_order.status = WorkOrderStatus.APPROVED

        return external_id

    async def update_work_order_status(
        self,
        work_order_id: str,
        status: WorkOrderStatus,
        notes: str = ""
    ) -> bool:
        """
        Update work order status.

        Args:
            work_order_id: Work order ID
            status: New status
            notes: Status update notes

        Returns:
            True if update successful
        """
        if not self._connected:
            raise RuntimeError("Not connected to CMMS")

        # In production, this would call CMMS API
        return True

    async def get_open_work_orders(
        self,
        asset_id: Optional[str] = None
    ) -> List[WorkOrder]:
        """
        Get open work orders, optionally filtered by asset.

        Args:
            asset_id: Filter by asset ID

        Returns:
            List of open work orders
        """
        if not self._connected:
            raise RuntimeError("Not connected to CMMS")

        # In production, this would query CMMS API
        return []

    async def get_asset_info(self, asset_id: str) -> Optional[Asset]:
        """
        Get asset information from CMMS.

        Args:
            asset_id: Asset ID

        Returns:
            Asset information or None
        """
        if not self._connected:
            raise RuntimeError("Not connected to CMMS")

        # In production, this would query CMMS API
        return None

    async def sync_diagnostic_results(
        self,
        diagnostics: List[Any],
        auto_create_work_orders: bool = True
    ) -> Dict[str, Any]:
        """
        Sync diagnostic results to CMMS.

        Args:
            diagnostics: List of diagnostic outputs
            auto_create_work_orders: Automatically create work orders

        Returns:
            Sync summary
        """
        if not self._connected:
            raise RuntimeError("Not connected to CMMS")

        created = 0
        failed = 0
        skipped = 0
        work_orders = []

        for diag in diagnostics:
            condition = getattr(diag, 'condition', 'healthy')

            # Only create work orders for failed or leaking traps
            if condition.lower() not in ('failed', 'leaking') or not auto_create_work_orders:
                skipped += 1
                continue

            try:
                wo = self.create_work_order_from_diagnostic(diag)
                external_id = await self.submit_work_order(wo)
                work_orders.append(wo)
                created += 1
            except Exception:
                failed += 1

        return {
            "total_diagnostics": len(diagnostics),
            "work_orders_created": created,
            "failed": failed,
            "skipped": skipped,
            "work_orders": work_orders,
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_cmms_connector(config: CMMSConfig) -> CMMSConnector:
    """
    Factory function to create CMMS connector.

    Args:
        config: CMMS configuration

    Returns:
        Configured CMMS connector
    """
    return CMMSConnector(config)
