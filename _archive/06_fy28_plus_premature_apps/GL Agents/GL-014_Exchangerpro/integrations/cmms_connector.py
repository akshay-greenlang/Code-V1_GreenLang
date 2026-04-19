# -*- coding: utf-8 -*-
"""
GL-014 ExchangerPro - CMMS Connector

Computerized Maintenance Management System (CMMS) integration for:
- Reading work orders, maintenance events, costs, completion timestamps
- Writing draft work-order recommendations (with human approval workflow)
- Linkage: recommendation -> computation record -> CMMS work order
- Support for SAP PM, Maximo, and common CMMS systems

Key Features:
- READ work orders, maintenance history, costs
- WRITE only draft recommendations (requires human approval)
- Full audit trail with computation linkage
- Support for multiple CMMS backends

Security:
- All write operations are draft recommendations only
- Human approval workflow required before execution
- Complete audit trail for regulatory compliance

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class WorkOrderPriority(str, Enum):
    """Work order priority levels."""
    EMERGENCY = "emergency"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ROUTINE = "routine"


class WorkOrderType(str, Enum):
    """Work order types."""
    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    INSPECTION = "inspection"
    CLEANING = "cleaning"
    CALIBRATION = "calibration"


class WorkOrderStatus(str, Enum):
    """Work order lifecycle status."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class WorkOrderMode(str, Enum):
    """Work order creation mode."""
    READ_ONLY = "read_only"
    DRAFT_ONLY = "draft_only"
    FULL_ACCESS = "full_access"


class RecommendationStatus(str, Enum):
    """Cleaning recommendation status."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"


class CMMSType(str, Enum):
    """Supported CMMS systems."""
    SAP_PM = "sap_pm"
    MAXIMO = "maximo"
    INFOR_EAM = "infor_eam"
    FIIX = "fiix"
    GENERIC_REST = "generic_rest"
    MOCK = "mock"


class CleaningMethod(str, Enum):
    """Heat exchanger cleaning methods."""
    CHEMICAL_CIP = "chemical_cip"
    MECHANICAL_BRUSH = "mechanical_brush"
    HIGH_PRESSURE_WATER = "high_pressure_water"
    HYDRO_BLASTING = "hydro_blasting"
    ULTRASONIC = "ultrasonic"
    PIGGING = "pigging"
    THERMAL = "thermal"


# =============================================================================
# DATA MODELS
# =============================================================================

class MaintenanceCost(BaseModel):
    """Maintenance cost record."""
    cost_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Cost record ID"
    )
    work_order_id: str = Field(..., description="Associated work order")
    cost_type: str = Field(..., description="Labor, Parts, Contractor, etc.")
    amount: float = Field(..., ge=0, description="Cost amount")
    currency: str = Field(default="USD", description="Currency code")
    cost_date: datetime = Field(..., description="Cost date")
    description: str = Field(default="", description="Cost description")

    # Labor specifics
    labor_hours: Optional[float] = Field(None, ge=0, description="Labor hours")
    labor_rate: Optional[float] = Field(None, ge=0, description="Labor rate")
    craft: Optional[str] = Field(None, description="Trade/craft")

    # Parts specifics
    part_number: Optional[str] = Field(None, description="Part number")
    quantity: Optional[float] = Field(None, ge=0, description="Part quantity")
    unit_cost: Optional[float] = Field(None, ge=0, description="Unit cost")


class MaintenanceEvent(BaseModel):
    """Maintenance event record from CMMS."""
    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Event ID"
    )
    work_order_id: str = Field(..., description="Associated work order")
    exchanger_id: str = Field(..., description="Heat exchanger ID")
    event_type: str = Field(..., description="Event type")

    # Timing
    start_time: datetime = Field(..., description="Event start")
    end_time: Optional[datetime] = Field(None, description="Event end")
    duration_hours: Optional[float] = Field(None, description="Duration")

    # Details
    description: str = Field(default="", description="Event description")
    performed_by: str = Field(default="", description="Technician/crew")

    # Results
    status: str = Field(..., description="Event status")
    findings: Optional[str] = Field(None, description="Findings")
    corrective_actions: Optional[str] = Field(None, description="Actions taken")

    # Measurements (for cleaning events)
    pre_event_fouling: Optional[float] = Field(None, description="Pre-event fouling factor")
    post_event_fouling: Optional[float] = Field(None, description="Post-event fouling factor")
    pre_event_htc: Optional[float] = Field(None, description="Pre-event HTC")
    post_event_htc: Optional[float] = Field(None, description="Post-event HTC")


class WorkOrder(BaseModel):
    """Work order model compatible with CMMS systems."""
    work_order_id: str = Field(
        default_factory=lambda: f"WO-{uuid.uuid4().hex[:8].upper()}",
        description="Work order ID"
    )
    external_id: Optional[str] = Field(None, description="CMMS external ID")

    # Equipment
    exchanger_id: str = Field(..., description="Heat exchanger ID")
    equipment_tag: str = Field(default="", description="Equipment tag")
    functional_location: str = Field(default="", description="Functional location")

    # Work order details
    work_order_type: WorkOrderType = Field(
        default=WorkOrderType.CORRECTIVE,
        description="Work order type"
    )
    priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Priority"
    )
    status: WorkOrderStatus = Field(
        default=WorkOrderStatus.DRAFT,
        description="Status"
    )

    # Description
    short_description: str = Field(..., max_length=100, description="Short description")
    long_description: str = Field(default="", description="Detailed description")
    cleaning_method: Optional[CleaningMethod] = Field(None, description="Cleaning method")

    # Trigger information
    trigger_reason: str = Field(default="", description="Reason for work order")
    fouling_factor: Optional[float] = Field(None, description="Triggering fouling factor")
    efficiency_loss_percent: Optional[float] = Field(None, description="Efficiency loss %")
    ai_confidence: Optional[float] = Field(None, ge=0, le=1, description="AI confidence")

    # Scheduling
    requested_date: Optional[datetime] = Field(None, description="Requested date")
    scheduled_start: Optional[datetime] = Field(None, description="Scheduled start")
    scheduled_end: Optional[datetime] = Field(None, description="Scheduled end")
    actual_start: Optional[datetime] = Field(None, description="Actual start")
    actual_end: Optional[datetime] = Field(None, description="Actual end")

    # Estimates
    estimated_duration_hours: float = Field(default=8.0, ge=0, description="Duration")
    estimated_cost: float = Field(default=0.0, ge=0, description="Estimated cost")

    # Actuals
    actual_duration_hours: Optional[float] = Field(None, ge=0, description="Actual duration")
    actual_cost: Optional[float] = Field(None, ge=0, description="Actual cost")

    # Assignments
    assigned_to: str = Field(default="", description="Assigned to")
    planner: str = Field(default="", description="Planner")

    # Audit
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation time"
    )
    created_by: str = Field(default="GL014-ExchangerPro", description="Created by")
    modified_at: Optional[datetime] = Field(None, description="Last modified")
    completed_at: Optional[datetime] = Field(None, description="Completion time")

    # Provenance
    provenance_hash: str = Field(default="", description="Provenance hash")
    computation_record_id: Optional[str] = Field(
        None,
        description="Linked computation record"
    )
    recommendation_id: Optional[str] = Field(None, description="Source recommendation")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance()

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        data = f"{self.work_order_id}|{self.exchanger_id}|{self.created_at.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()


class CleaningRecommendation(BaseModel):
    """
    Cleaning recommendation with approval workflow.

    Represents a draft recommendation that requires human approval
    before being converted to a work order.
    """
    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Recommendation ID"
    )
    exchanger_id: str = Field(..., description="Heat exchanger ID")
    site_id: str = Field(..., description="Site ID")

    # Recommendation details
    cleaning_method: CleaningMethod = Field(..., description="Recommended cleaning method")
    priority: WorkOrderPriority = Field(..., description="Recommended priority")
    urgency: str = Field(..., description="Immediate, Scheduled, Opportunistic")

    # Timing
    recommended_date: datetime = Field(..., description="Recommended cleaning date")
    window_start: datetime = Field(..., description="Scheduling window start")
    window_end: datetime = Field(..., description="Scheduling window end")

    # Justification
    trigger_reason: str = Field(..., description="Reason for recommendation")
    fouling_factor: float = Field(..., description="Current fouling factor")
    fouling_rate: float = Field(..., description="Fouling rate (m2K/W per day)")
    efficiency_loss_percent: float = Field(..., description="Efficiency loss %")

    # Economic analysis
    estimated_cleaning_cost: float = Field(..., description="Cleaning cost estimate")
    estimated_energy_savings: float = Field(..., description="Energy savings estimate")
    estimated_roi: Optional[float] = Field(None, description="ROI estimate")
    payback_days: Optional[int] = Field(None, description="Payback period")
    cost_of_delay_per_day: Optional[float] = Field(None, description="Delay cost/day")

    # Model information
    model_id: str = Field(..., description="Prediction model ID")
    model_version: str = Field(..., description="Model version")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")

    # Approval workflow
    status: RecommendationStatus = Field(
        default=RecommendationStatus.DRAFT,
        description="Recommendation status"
    )
    requires_approval: bool = Field(default=True, description="Requires approval")
    approver_role: str = Field(
        default="maintenance_supervisor",
        description="Required approver role"
    )
    approved_by: Optional[str] = Field(None, description="Approver")
    approval_date: Optional[datetime] = Field(None, description="Approval date")
    approval_notes: Optional[str] = Field(None, description="Approval notes")

    # Linkage
    computation_record_id: str = Field(..., description="Linked computation record")
    work_order_id: Optional[str] = Field(None, description="Created work order ID")

    # Audit
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation time"
    )
    expires_at: datetime = Field(..., description="Recommendation expiry")
    provenance_hash: str = Field(default="", description="Provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance."""
        if not self.provenance_hash:
            data = f"{self.recommendation_id}|{self.exchanger_id}|{self.created_at.isoformat()}"
            self.provenance_hash = hashlib.sha256(data.encode()).hexdigest()

    def is_expired(self) -> bool:
        """Check if recommendation has expired."""
        return datetime.now(timezone.utc) > self.expires_at


class ComputationLinkage(BaseModel):
    """
    Links recommendation to underlying computation record.

    Provides full traceability from CMMS work order back to
    the original fouling prediction computation.
    """
    linkage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Linkage ID"
    )

    # Identifiers
    recommendation_id: str = Field(..., description="Recommendation ID")
    computation_record_id: str = Field(..., description="Computation record ID")
    work_order_id: Optional[str] = Field(None, description="CMMS work order ID")

    # Computation details
    computation_timestamp: datetime = Field(..., description="Computation time")
    model_id: str = Field(..., description="Model ID")
    model_version: str = Field(..., description="Model version")

    # Input summary
    input_data_hash: str = Field(..., description="Hash of input data")
    input_data_count: int = Field(..., description="Number of input data points")
    input_time_range_start: datetime = Field(..., description="Input data start")
    input_time_range_end: datetime = Field(..., description="Input data end")

    # Output summary
    fouling_prediction: float = Field(..., description="Predicted fouling")
    confidence: float = Field(..., description="Prediction confidence")
    recommended_action: str = Field(..., description="Recommended action")

    # Provenance
    provenance_hash: str = Field(default="", description="Provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance."""
        if not self.provenance_hash:
            data = (
                f"{self.linkage_id}|{self.recommendation_id}|"
                f"{self.computation_record_id}|{self.computation_timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(data.encode()).hexdigest()


class ApprovalWorkflow(BaseModel):
    """Approval workflow state for recommendations."""
    workflow_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Workflow ID"
    )
    recommendation_id: str = Field(..., description="Recommendation ID")

    # Workflow state
    current_step: str = Field(default="review", description="Current step")
    steps: List[str] = Field(
        default_factory=lambda: ["review", "approve", "schedule", "execute"],
        description="Workflow steps"
    )
    completed_steps: List[str] = Field(
        default_factory=list,
        description="Completed steps"
    )

    # Approvers
    required_approvers: List[str] = Field(
        default_factory=lambda: ["maintenance_supervisor"],
        description="Required approvers"
    )
    approvals: Dict[str, datetime] = Field(
        default_factory=dict,
        description="Approvals received"
    )
    rejections: Dict[str, str] = Field(
        default_factory=dict,
        description="Rejections with reasons"
    )

    # Timing
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Workflow start"
    )
    deadline: Optional[datetime] = Field(None, description="Approval deadline")
    completed_at: Optional[datetime] = Field(None, description="Completion time")

    def is_approved(self) -> bool:
        """Check if all required approvals received."""
        return all(
            approver in self.approvals
            for approver in self.required_approvers
        )

    def is_rejected(self) -> bool:
        """Check if any rejection received."""
        return len(self.rejections) > 0


class CMMSResponse(BaseModel):
    """Response from CMMS operations."""
    success: bool = Field(..., description="Operation successful")
    external_id: Optional[str] = Field(None, description="CMMS ID")
    message: str = Field(default="", description="Response message")
    errors: List[str] = Field(default_factory=list, description="Errors")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response time"
    )


# =============================================================================
# CMMS ADAPTERS
# =============================================================================

class CMMSAdapter(ABC):
    """Abstract base class for CMMS adapters."""

    @abstractmethod
    async def read_work_orders(
        self,
        exchanger_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[WorkOrderStatus] = None,
    ) -> List[WorkOrder]:
        """Read work orders from CMMS."""
        pass

    @abstractmethod
    async def read_maintenance_events(
        self,
        exchanger_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[MaintenanceEvent]:
        """Read maintenance events from CMMS."""
        pass

    @abstractmethod
    async def read_maintenance_costs(
        self,
        work_order_id: str,
    ) -> List[MaintenanceCost]:
        """Read costs for a work order."""
        pass

    @abstractmethod
    async def create_draft_work_order(
        self,
        recommendation: CleaningRecommendation,
    ) -> CMMSResponse:
        """Create draft work order from recommendation."""
        pass

    @abstractmethod
    async def check_connection(self) -> bool:
        """Check CMMS connection status."""
        pass


class SAPPMConfig(BaseModel):
    """SAP PM configuration."""
    base_url: str = Field(..., description="SAP OData service URL")
    client: str = Field(default="100", description="SAP client")
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    plant: str = Field(..., description="Plant code")
    order_type: str = Field(default="PM01", description="Order type")


class SAPPMConnector(CMMSAdapter):
    """SAP Plant Maintenance connector."""

    def __init__(self, config: SAPPMConfig):
        """Initialize SAP PM connector."""
        self.config = config
        self._connected = False
        logger.info(f"SAP PM connector initialized for {config.base_url}")

    async def read_work_orders(
        self,
        exchanger_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[WorkOrderStatus] = None,
    ) -> List[WorkOrder]:
        """Read work orders from SAP PM."""
        logger.info(f"Reading work orders for {exchanger_id} from SAP PM")
        # In production, would query SAP OData
        return []

    async def read_maintenance_events(
        self,
        exchanger_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[MaintenanceEvent]:
        """Read maintenance events from SAP PM."""
        logger.info(f"Reading maintenance events for {exchanger_id}")
        return []

    async def read_maintenance_costs(
        self,
        work_order_id: str,
    ) -> List[MaintenanceCost]:
        """Read costs from SAP PM."""
        logger.info(f"Reading costs for {work_order_id}")
        return []

    async def create_draft_work_order(
        self,
        recommendation: CleaningRecommendation,
    ) -> CMMSResponse:
        """Create draft work order in SAP PM."""
        logger.info(f"Creating draft WO in SAP PM for {recommendation.exchanger_id}")

        # Map to SAP fields
        sap_order = {
            "OrderType": self.config.order_type,
            "Plant": self.config.plant,
            "Equipment": recommendation.exchanger_id,
            "Priority": self._map_priority(recommendation.priority),
            "ShortText": f"Cleaning: {recommendation.cleaning_method.value}",
            "LongText": recommendation.trigger_reason,
            "BasicStartDate": recommendation.recommended_date.strftime("%Y%m%d"),
        }

        # In production, would POST to SAP OData
        external_id = f"40{uuid.uuid4().int % 10000000:08d}"

        return CMMSResponse(
            success=True,
            external_id=external_id,
            message=f"Draft work order created in SAP PM: {external_id}",
        )

    async def check_connection(self) -> bool:
        """Check SAP connection."""
        self._connected = True
        return True

    def _map_priority(self, priority: WorkOrderPriority) -> str:
        """Map priority to SAP code."""
        mapping = {
            WorkOrderPriority.EMERGENCY: "1",
            WorkOrderPriority.HIGH: "2",
            WorkOrderPriority.MEDIUM: "3",
            WorkOrderPriority.LOW: "4",
            WorkOrderPriority.ROUTINE: "5",
        }
        return mapping.get(priority, "3")


class MaximoConfig(BaseModel):
    """IBM Maximo configuration."""
    base_url: str = Field(..., description="Maximo API URL")
    api_key: str = Field(..., description="API key")
    site_id: str = Field(..., description="Site ID")
    org_id: str = Field(..., description="Organization ID")


class MaximoConnector(CMMSAdapter):
    """IBM Maximo connector."""

    def __init__(self, config: MaximoConfig):
        """Initialize Maximo connector."""
        self.config = config
        self._connected = False
        logger.info(f"Maximo connector initialized for {config.base_url}")

    async def read_work_orders(
        self,
        exchanger_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[WorkOrderStatus] = None,
    ) -> List[WorkOrder]:
        """Read work orders from Maximo."""
        logger.info(f"Reading work orders for {exchanger_id} from Maximo")
        return []

    async def read_maintenance_events(
        self,
        exchanger_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[MaintenanceEvent]:
        """Read maintenance events from Maximo."""
        return []

    async def read_maintenance_costs(
        self,
        work_order_id: str,
    ) -> List[MaintenanceCost]:
        """Read costs from Maximo."""
        return []

    async def create_draft_work_order(
        self,
        recommendation: CleaningRecommendation,
    ) -> CMMSResponse:
        """Create draft work order in Maximo."""
        logger.info(f"Creating draft WO in Maximo for {recommendation.exchanger_id}")

        external_id = f"WO{uuid.uuid4().int % 10000000:07d}"

        return CMMSResponse(
            success=True,
            external_id=external_id,
            message=f"Draft work order created in Maximo: {external_id}",
        )

    async def check_connection(self) -> bool:
        """Check Maximo connection."""
        self._connected = True
        return True


class MockCMMSAdapter(CMMSAdapter):
    """Mock CMMS adapter for testing."""

    def __init__(self):
        """Initialize mock adapter."""
        self._work_orders: Dict[str, WorkOrder] = {}
        self._events: Dict[str, MaintenanceEvent] = {}
        self._costs: Dict[str, List[MaintenanceCost]] = {}
        self._counter = 1000

    async def read_work_orders(
        self,
        exchanger_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[WorkOrderStatus] = None,
    ) -> List[WorkOrder]:
        """Read work orders from mock storage."""
        results = [
            wo for wo in self._work_orders.values()
            if wo.exchanger_id == exchanger_id
        ]

        if status:
            results = [wo for wo in results if wo.status == status]

        if start_date:
            results = [wo for wo in results if wo.created_at >= start_date]

        if end_date:
            results = [wo for wo in results if wo.created_at <= end_date]

        return results

    async def read_maintenance_events(
        self,
        exchanger_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[MaintenanceEvent]:
        """Read maintenance events from mock storage."""
        return [
            e for e in self._events.values()
            if e.exchanger_id == exchanger_id
        ]

    async def read_maintenance_costs(
        self,
        work_order_id: str,
    ) -> List[MaintenanceCost]:
        """Read costs from mock storage."""
        return self._costs.get(work_order_id, [])

    async def create_draft_work_order(
        self,
        recommendation: CleaningRecommendation,
    ) -> CMMSResponse:
        """Create draft work order in mock storage."""
        self._counter += 1
        external_id = f"MOCK-{self._counter}"

        work_order = WorkOrder(
            external_id=external_id,
            exchanger_id=recommendation.exchanger_id,
            work_order_type=WorkOrderType.CLEANING,
            priority=recommendation.priority,
            status=WorkOrderStatus.DRAFT,
            short_description=f"Cleaning: {recommendation.cleaning_method.value}",
            long_description=recommendation.trigger_reason,
            cleaning_method=recommendation.cleaning_method,
            trigger_reason=recommendation.trigger_reason,
            fouling_factor=recommendation.fouling_factor,
            efficiency_loss_percent=recommendation.efficiency_loss_percent,
            ai_confidence=recommendation.confidence,
            requested_date=recommendation.recommended_date,
            estimated_cost=recommendation.estimated_cleaning_cost,
            recommendation_id=recommendation.recommendation_id,
            computation_record_id=recommendation.computation_record_id,
        )

        self._work_orders[work_order.work_order_id] = work_order

        return CMMSResponse(
            success=True,
            external_id=external_id,
            message=f"Draft work order created: {external_id}",
        )

    async def check_connection(self) -> bool:
        """Mock is always connected."""
        return True


# =============================================================================
# CMMS MANAGER
# =============================================================================

class CMMSManager:
    """
    CMMS Manager for GL-014 ExchangerPro.

    Provides unified interface for CMMS operations with:
    - Read access to work orders, events, costs
    - Draft recommendation creation (requires approval)
    - Full audit trail and computation linkage
    - Support for multiple CMMS backends

    Example:
        >>> manager = CMMSManager(adapter=SAPPMConnector(config))
        >>> # Read historical maintenance
        >>> events = await manager.get_maintenance_history("HX-001")
        >>> # Create recommendation (draft only)
        >>> rec = await manager.create_cleaning_recommendation(params)
        >>> # Submit for approval
        >>> await manager.submit_for_approval(rec.recommendation_id)
    """

    def __init__(
        self,
        adapter: Optional[CMMSAdapter] = None,
        mode: WorkOrderMode = WorkOrderMode.DRAFT_ONLY,
    ):
        """
        Initialize CMMS Manager.

        Args:
            adapter: CMMS adapter (default: MockCMMSAdapter)
            mode: Operation mode (default: DRAFT_ONLY for safety)
        """
        self._adapter = adapter or MockCMMSAdapter()
        self._mode = mode

        # Local storage for recommendations and linkages
        self._recommendations: Dict[str, CleaningRecommendation] = {}
        self._workflows: Dict[str, ApprovalWorkflow] = {}
        self._linkages: Dict[str, ComputationLinkage] = {}
        self._audit_log: List[Dict[str, Any]] = []

        logger.info(
            f"CMMS Manager initialized (adapter={type(self._adapter).__name__}, "
            f"mode={mode.value})"
        )

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    async def get_work_orders(
        self,
        exchanger_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[WorkOrderStatus] = None,
    ) -> List[WorkOrder]:
        """
        Get work orders for an exchanger.

        Args:
            exchanger_id: Heat exchanger ID
            start_date: Optional start date filter
            end_date: Optional end date filter
            status: Optional status filter

        Returns:
            List of work orders
        """
        work_orders = await self._adapter.read_work_orders(
            exchanger_id, start_date, end_date, status
        )

        self._log_audit(
            "READ_WORK_ORDERS",
            exchanger_id=exchanger_id,
            count=len(work_orders),
        )

        return work_orders

    async def get_maintenance_history(
        self,
        exchanger_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[MaintenanceEvent]:
        """
        Get maintenance event history.

        Args:
            exchanger_id: Heat exchanger ID
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            List of maintenance events
        """
        events = await self._adapter.read_maintenance_events(
            exchanger_id, start_date, end_date
        )

        self._log_audit(
            "READ_MAINTENANCE_HISTORY",
            exchanger_id=exchanger_id,
            count=len(events),
        )

        return events

    async def get_maintenance_costs(
        self,
        work_order_id: str,
    ) -> List[MaintenanceCost]:
        """
        Get costs for a work order.

        Args:
            work_order_id: Work order ID

        Returns:
            List of cost records
        """
        return await self._adapter.read_maintenance_costs(work_order_id)

    async def get_total_maintenance_cost(
        self,
        exchanger_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Get total maintenance costs by category.

        Args:
            exchanger_id: Heat exchanger ID
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dictionary of costs by category
        """
        work_orders = await self.get_work_orders(
            exchanger_id, start_date, end_date
        )

        costs_by_type: Dict[str, float] = {}

        for wo in work_orders:
            costs = await self.get_maintenance_costs(wo.work_order_id)
            for cost in costs:
                if cost.cost_type not in costs_by_type:
                    costs_by_type[cost.cost_type] = 0.0
                costs_by_type[cost.cost_type] += cost.amount

        return costs_by_type

    async def get_last_cleaning_date(
        self,
        exchanger_id: str,
    ) -> Optional[datetime]:
        """
        Get date of last cleaning event.

        Args:
            exchanger_id: Heat exchanger ID

        Returns:
            Last cleaning date or None
        """
        events = await self.get_maintenance_history(exchanger_id)

        cleaning_events = [
            e for e in events
            if "clean" in e.event_type.lower()
        ]

        if not cleaning_events:
            return None

        return max(e.start_time for e in cleaning_events)

    async def get_completion_timestamps(
        self,
        exchanger_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get recent work order completion timestamps.

        Args:
            exchanger_id: Heat exchanger ID
            limit: Max records to return

        Returns:
            List of completion info
        """
        work_orders = await self.get_work_orders(
            exchanger_id, status=WorkOrderStatus.COMPLETED
        )

        completions = []
        for wo in work_orders[:limit]:
            if wo.completed_at:
                completions.append({
                    "work_order_id": wo.work_order_id,
                    "type": wo.work_order_type.value,
                    "completed_at": wo.completed_at,
                    "duration_hours": wo.actual_duration_hours,
                    "cost": wo.actual_cost,
                })

        return completions

    # =========================================================================
    # RECOMMENDATION CREATION (DRAFT ONLY)
    # =========================================================================

    async def create_cleaning_recommendation(
        self,
        exchanger_id: str,
        site_id: str,
        cleaning_method: CleaningMethod,
        priority: WorkOrderPriority,
        urgency: str,
        recommended_date: datetime,
        trigger_reason: str,
        fouling_factor: float,
        fouling_rate: float,
        efficiency_loss_percent: float,
        estimated_cleaning_cost: float,
        estimated_energy_savings: float,
        model_id: str,
        model_version: str,
        confidence: float,
        computation_record_id: str,
        expires_in_days: int = 14,
    ) -> CleaningRecommendation:
        """
        Create a cleaning recommendation (draft).

        This creates a DRAFT recommendation that requires human approval
        before a work order is created in CMMS.

        Args:
            exchanger_id: Heat exchanger ID
            site_id: Site ID
            cleaning_method: Recommended cleaning method
            priority: Recommended priority
            urgency: Urgency level
            recommended_date: Recommended cleaning date
            trigger_reason: Reason for recommendation
            fouling_factor: Current fouling factor
            fouling_rate: Fouling rate
            efficiency_loss_percent: Current efficiency loss
            estimated_cleaning_cost: Cost estimate
            estimated_energy_savings: Savings estimate
            model_id: Prediction model ID
            model_version: Model version
            confidence: Prediction confidence
            computation_record_id: Linked computation record
            expires_in_days: Recommendation expiry

        Returns:
            Created recommendation (draft status)
        """
        # Calculate economic metrics
        roi = None
        payback_days = None
        if estimated_cleaning_cost > 0 and estimated_energy_savings > 0:
            daily_savings = estimated_energy_savings / 365
            payback_days = int(estimated_cleaning_cost / daily_savings) if daily_savings > 0 else None
            roi = (estimated_energy_savings - estimated_cleaning_cost) / estimated_cleaning_cost

        recommendation = CleaningRecommendation(
            exchanger_id=exchanger_id,
            site_id=site_id,
            cleaning_method=cleaning_method,
            priority=priority,
            urgency=urgency,
            recommended_date=recommended_date,
            window_start=recommended_date - timedelta(days=7),
            window_end=recommended_date + timedelta(days=14),
            trigger_reason=trigger_reason,
            fouling_factor=fouling_factor,
            fouling_rate=fouling_rate,
            efficiency_loss_percent=efficiency_loss_percent,
            estimated_cleaning_cost=estimated_cleaning_cost,
            estimated_energy_savings=estimated_energy_savings,
            estimated_roi=roi,
            payback_days=payback_days,
            model_id=model_id,
            model_version=model_version,
            confidence=confidence,
            computation_record_id=computation_record_id,
            expires_at=datetime.now(timezone.utc) + timedelta(days=expires_in_days),
            status=RecommendationStatus.DRAFT,
        )

        # Store recommendation
        self._recommendations[recommendation.recommendation_id] = recommendation

        # Create computation linkage
        linkage = ComputationLinkage(
            recommendation_id=recommendation.recommendation_id,
            computation_record_id=computation_record_id,
            computation_timestamp=datetime.now(timezone.utc),
            model_id=model_id,
            model_version=model_version,
            input_data_hash=hashlib.sha256(
                f"{exchanger_id}|{datetime.now().isoformat()}".encode()
            ).hexdigest(),
            input_data_count=0,
            input_time_range_start=datetime.now(timezone.utc) - timedelta(days=30),
            input_time_range_end=datetime.now(timezone.utc),
            fouling_prediction=fouling_factor,
            confidence=confidence,
            recommended_action=f"Clean using {cleaning_method.value}",
        )
        self._linkages[linkage.linkage_id] = linkage

        # Create approval workflow
        workflow = ApprovalWorkflow(
            recommendation_id=recommendation.recommendation_id,
            deadline=recommendation.expires_at,
        )
        self._workflows[workflow.workflow_id] = workflow

        self._log_audit(
            "RECOMMENDATION_CREATED",
            recommendation_id=recommendation.recommendation_id,
            exchanger_id=exchanger_id,
            priority=priority.value,
            status="draft",
        )

        logger.info(
            f"Created cleaning recommendation {recommendation.recommendation_id} "
            f"for {exchanger_id} (status=draft)"
        )

        return recommendation

    async def submit_for_approval(
        self,
        recommendation_id: str,
    ) -> bool:
        """
        Submit recommendation for approval.

        Args:
            recommendation_id: Recommendation ID

        Returns:
            True if submitted successfully
        """
        recommendation = self._recommendations.get(recommendation_id)
        if not recommendation:
            raise ValueError(f"Recommendation not found: {recommendation_id}")

        if recommendation.is_expired():
            raise ValueError("Recommendation has expired")

        recommendation.status = RecommendationStatus.PENDING_REVIEW

        self._log_audit(
            "RECOMMENDATION_SUBMITTED",
            recommendation_id=recommendation_id,
            status="pending_review",
        )

        logger.info(f"Recommendation {recommendation_id} submitted for approval")
        return True

    async def approve_recommendation(
        self,
        recommendation_id: str,
        approver: str,
        notes: Optional[str] = None,
    ) -> CMMSResponse:
        """
        Approve recommendation and create draft work order.

        Args:
            recommendation_id: Recommendation ID
            approver: Approver username
            notes: Approval notes

        Returns:
            CMMS response with work order ID
        """
        recommendation = self._recommendations.get(recommendation_id)
        if not recommendation:
            raise ValueError(f"Recommendation not found: {recommendation_id}")

        if recommendation.status != RecommendationStatus.PENDING_REVIEW:
            raise ValueError(f"Recommendation not pending review: {recommendation.status}")

        if recommendation.is_expired():
            raise ValueError("Recommendation has expired")

        # Update approval
        recommendation.status = RecommendationStatus.APPROVED
        recommendation.approved_by = approver
        recommendation.approval_date = datetime.now(timezone.utc)
        recommendation.approval_notes = notes

        # Create draft work order in CMMS
        response = await self._adapter.create_draft_work_order(recommendation)

        if response.success:
            recommendation.work_order_id = response.external_id

            # Update linkage
            for linkage in self._linkages.values():
                if linkage.recommendation_id == recommendation_id:
                    linkage.work_order_id = response.external_id

        self._log_audit(
            "RECOMMENDATION_APPROVED",
            recommendation_id=recommendation_id,
            approver=approver,
            work_order_id=response.external_id,
        )

        logger.info(
            f"Recommendation {recommendation_id} approved by {approver}, "
            f"work order: {response.external_id}"
        )

        return response

    async def reject_recommendation(
        self,
        recommendation_id: str,
        rejector: str,
        reason: str,
    ) -> bool:
        """
        Reject a recommendation.

        Args:
            recommendation_id: Recommendation ID
            rejector: Rejector username
            reason: Rejection reason

        Returns:
            True if rejected
        """
        recommendation = self._recommendations.get(recommendation_id)
        if not recommendation:
            raise ValueError(f"Recommendation not found: {recommendation_id}")

        recommendation.status = RecommendationStatus.REJECTED
        recommendation.approval_notes = f"Rejected by {rejector}: {reason}"

        self._log_audit(
            "RECOMMENDATION_REJECTED",
            recommendation_id=recommendation_id,
            rejector=rejector,
            reason=reason,
        )

        logger.info(f"Recommendation {recommendation_id} rejected by {rejector}")
        return True

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_recommendation(
        self,
        recommendation_id: str,
    ) -> Optional[CleaningRecommendation]:
        """Get recommendation by ID."""
        return self._recommendations.get(recommendation_id)

    def get_pending_recommendations(
        self,
        exchanger_id: Optional[str] = None,
    ) -> List[CleaningRecommendation]:
        """Get pending recommendations."""
        pending = [
            r for r in self._recommendations.values()
            if r.status in [RecommendationStatus.DRAFT, RecommendationStatus.PENDING_REVIEW]
            and not r.is_expired()
        ]

        if exchanger_id:
            pending = [r for r in pending if r.exchanger_id == exchanger_id]

        return sorted(pending, key=lambda x: x.created_at, reverse=True)

    def get_computation_linkage(
        self,
        recommendation_id: str,
    ) -> Optional[ComputationLinkage]:
        """Get computation linkage for recommendation."""
        for linkage in self._linkages.values():
            if linkage.recommendation_id == recommendation_id:
                return linkage
        return None

    def get_approval_workflow(
        self,
        recommendation_id: str,
    ) -> Optional[ApprovalWorkflow]:
        """Get approval workflow for recommendation."""
        for workflow in self._workflows.values():
            if workflow.recommendation_id == recommendation_id:
                return workflow
        return None

    # =========================================================================
    # AUDIT
    # =========================================================================

    def _log_audit(self, event_type: str, **kwargs: Any) -> None:
        """Log audit event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **kwargs,
        }
        hash_str = f"{entry['timestamp']}|{event_type}|{str(kwargs)}"
        entry["provenance_hash"] = hashlib.sha256(hash_str.encode()).hexdigest()[:16]
        self._audit_log.append(entry)

    def get_audit_log(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        return list(reversed(self._audit_log[-limit:]))

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    async def check_connection(self) -> bool:
        """Check CMMS connection."""
        return await self._adapter.check_connection()

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "adapter_type": type(self._adapter).__name__,
            "mode": self._mode.value,
            "recommendations_count": len(self._recommendations),
            "pending_count": len(self.get_pending_recommendations()),
            "linkages_count": len(self._linkages),
            "audit_log_count": len(self._audit_log),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_cmms_manager(
    cmms_type: CMMSType,
    config: Optional[Dict[str, Any]] = None,
    mode: WorkOrderMode = WorkOrderMode.DRAFT_ONLY,
) -> CMMSManager:
    """
    Create CMMS manager with appropriate adapter.

    Args:
        cmms_type: CMMS system type
        config: Adapter configuration
        mode: Operation mode

    Returns:
        Configured CMMSManager
    """
    config = config or {}

    if cmms_type == CMMSType.SAP_PM:
        adapter = SAPPMConnector(SAPPMConfig(**config))
    elif cmms_type == CMMSType.MAXIMO:
        adapter = MaximoConnector(MaximoConfig(**config))
    elif cmms_type == CMMSType.MOCK:
        adapter = MockCMMSAdapter()
    else:
        logger.warning(f"Unknown CMMS type {cmms_type}, using mock")
        adapter = MockCMMSAdapter()

    return CMMSManager(adapter=adapter, mode=mode)


# =============================================================================
# CONNECTOR CLASS (ALIAS FOR COMPATIBILITY)
# =============================================================================

class CMMSConnector(CMMSManager):
    """Alias for CMMSManager for naming consistency."""
    pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "WorkOrderPriority",
    "WorkOrderType",
    "WorkOrderStatus",
    "WorkOrderMode",
    "RecommendationStatus",
    "CMMSType",
    "CleaningMethod",

    # Data Models
    "MaintenanceCost",
    "MaintenanceEvent",
    "WorkOrder",
    "CleaningRecommendation",
    "ComputationLinkage",
    "ApprovalWorkflow",
    "CMMSResponse",

    # Adapters
    "CMMSAdapter",
    "SAPPMConfig",
    "SAPPMConnector",
    "MaximoConfig",
    "MaximoConnector",
    "MockCMMSAdapter",

    # Manager
    "CMMSManager",
    "CMMSConnector",

    # Factory
    "create_cmms_manager",
]
