"""
GL-016 Waterguard CMMS Integration

Connector for creating and managing work orders in Computerized Maintenance
Management Systems. Supports multiple CMMS backends with idempotent writes
to prevent duplicate work orders.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, Field, SecretStr
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from integrations.cmms.cmms_schemas import (
    Asset,
    AssetType,
    EquipmentStatus,
    MaintenanceTask,
    WorkOrder,
    WorkOrderContext,
    WorkOrderPriority,
    WorkOrderStatus,
    WorkOrderTemplate,
    WorkOrderType,
    get_analyzer_fault_template,
    get_calibration_template,
    get_pump_mismatch_template,
    get_reagent_low_template,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Work Order Triggers
# =============================================================================

class WorkOrderTrigger(str, Enum):
    """Standard triggers for work order creation."""
    CALIBRATION_DUE = "calibration_due"
    CALIBRATION_FAILED = "calibration_failed"
    DRIFT_DETECTED = "drift_detected"
    ANALYZER_FAULT = "analyzer_fault"
    SENSOR_FAILURE = "sensor_failure"
    COMM_FAILURE = "comm_failure"
    REAGENT_LOW = "reagent_low"
    REAGENT_EMPTY = "reagent_empty"
    PUMP_MISMATCH = "pump_mismatch"
    FLOW_DEVIATION = "flow_deviation"
    OUTPUT_ERROR = "output_error"
    VALVE_FAULT = "valve_fault"
    MANUAL_REQUEST = "manual_request"
    SCHEDULED = "scheduled"


# =============================================================================
# Configuration
# =============================================================================

class CMMSType(str, Enum):
    """Supported CMMS types."""
    SAP_PM = "sap_pm"
    MAXIMO = "maximo"
    MAINTENANCE_CONNECTION = "maintenance_connection"
    FIIX = "fiix"
    UPKEEP = "upkeep"
    GENERIC_REST = "generic_rest"
    MOCK = "mock"


class CMMSConfig(BaseModel):
    """CMMS connection configuration."""

    # Connection
    cmms_type: CMMSType = Field(..., description="CMMS type")
    base_url: str = Field(..., description="CMMS API base URL")
    api_version: str = Field(default="v1", description="API version")

    # Authentication
    auth_type: str = Field(default="oauth2", description="Authentication type")
    client_id: Optional[str] = Field(default=None, description="OAuth client ID")
    client_secret: Optional[SecretStr] = Field(default=None, description="OAuth secret")
    api_key: Optional[SecretStr] = Field(default=None, description="API key")
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[SecretStr] = Field(default=None, description="Password")
    token_url: Optional[str] = Field(default=None, description="OAuth token URL")

    # Behavior
    timeout_seconds: int = Field(default=30, description="Request timeout")
    max_retries: int = Field(default=3, description="Max retry attempts")
    retry_delay_seconds: float = Field(default=1.0, description="Initial retry delay")

    # Idempotency
    idempotency_window_hours: int = Field(
        default=24,
        description="Hours to prevent duplicate work orders"
    )

    # Mapping
    site_code: str = Field(default="PLANT1", description="Site code for work orders")
    work_center: Optional[str] = Field(default=None, description="Default work center")
    cost_center: Optional[str] = Field(default=None, description="Default cost center")

    class Config:
        extra = "forbid"


# =============================================================================
# Idempotency Tracking
# =============================================================================

@dataclass
class IdempotencyRecord:
    """Record for tracking work order creation."""
    idempotency_key: str
    work_order_id: UUID
    external_id: Optional[str]
    created_at: datetime
    trigger: WorkOrderTrigger
    asset_id: str


class IdempotencyTracker:
    """
    Tracks work order creation to prevent duplicates.

    Uses idempotency keys to ensure the same condition doesn't
    create multiple work orders within the configured window.
    """

    def __init__(self, window_hours: int = 24):
        self._records: Dict[str, IdempotencyRecord] = {}
        self._window = timedelta(hours=window_hours)
        self._lock = asyncio.Lock()

    async def check_and_register(
        self,
        idempotency_key: str,
        work_order_id: UUID,
        external_id: Optional[str],
        trigger: WorkOrderTrigger,
        asset_id: str,
    ) -> Optional[IdempotencyRecord]:
        """
        Check if work order already exists for this key.

        Args:
            idempotency_key: Unique key for this condition
            work_order_id: New work order ID
            external_id: External CMMS ID
            trigger: Trigger type
            asset_id: Asset ID

        Returns:
            Existing record if duplicate, None if new
        """
        async with self._lock:
            # Clean old records
            self._cleanup()

            # Check for existing
            if idempotency_key in self._records:
                existing = self._records[idempotency_key]
                logger.info(
                    f"Duplicate work order prevented: key={idempotency_key}, "
                    f"existing_wo={existing.work_order_id}"
                )
                return existing

            # Register new
            record = IdempotencyRecord(
                idempotency_key=idempotency_key,
                work_order_id=work_order_id,
                external_id=external_id,
                created_at=datetime.utcnow(),
                trigger=trigger,
                asset_id=asset_id,
            )
            self._records[idempotency_key] = record
            return None

    def _cleanup(self) -> None:
        """Remove expired records."""
        cutoff = datetime.utcnow() - self._window
        expired = [
            key for key, record in self._records.items()
            if record.created_at < cutoff
        ]
        for key in expired:
            del self._records[key]

    def get_recent_for_asset(self, asset_id: str) -> List[IdempotencyRecord]:
        """Get recent work orders for an asset."""
        return [
            record for record in self._records.values()
            if record.asset_id == asset_id
        ]


# =============================================================================
# CMMS Connector
# =============================================================================

class CMMSConnector:
    """
    CMMS connector for work order management.

    Features:
    - Work order creation with templates
    - Asset synchronization
    - Idempotent writes to prevent duplicates
    - Multiple CMMS backend support
    - Retry logic with exponential backoff

    Example:
        config = CMMSConfig(
            cmms_type=CMMSType.GENERIC_REST,
            base_url="https://cmms.example.com/api/v1",
            api_key=SecretStr("your-api-key"),
        )

        connector = CMMSConnector(config)
        await connector.initialize()

        # Create work order
        result = await connector.create_work_order(
            asset_id="ANALYZER_001",
            trigger=WorkOrderTrigger.CALIBRATION_DUE,
            description="Analyzer calibration due per schedule",
            context={"readings": {"phosphate": 5.2}},
        )
    """

    # Template registry
    TEMPLATES: Dict[str, WorkOrderTemplate] = {
        "calibration_due": get_calibration_template(),
        "calibration_failed": get_calibration_template(),
        "drift_detected": get_calibration_template(),
        "analyzer_fault": get_analyzer_fault_template(),
        "sensor_failure": get_analyzer_fault_template(),
        "comm_failure": get_analyzer_fault_template(),
        "reagent_low": get_reagent_low_template(),
        "reagent_empty": get_reagent_low_template(),
        "pump_mismatch": get_pump_mismatch_template(),
        "flow_deviation": get_pump_mismatch_template(),
        "output_error": get_pump_mismatch_template(),
    }

    def __init__(
        self,
        config: CMMSConfig,
        on_work_order_created: Optional[Callable[[WorkOrder], None]] = None,
    ):
        """
        Initialize CMMS connector.

        Args:
            config: CMMS configuration
            on_work_order_created: Callback when work order is created
        """
        self.config = config
        self._on_work_order_created = on_work_order_created

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

        # Asset cache
        self._assets: Dict[str, Asset] = {}

        # Idempotency tracking
        self._idempotency_tracker = IdempotencyTracker(
            window_hours=config.idempotency_window_hours
        )

        # Metrics
        self._work_orders_created = 0
        self._work_orders_duplicates_prevented = 0

    async def initialize(self) -> None:
        """Initialize the connector."""
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            headers={"Content-Type": "application/json"},
        )

        # Authenticate if needed
        if self.config.auth_type == "oauth2":
            await self._authenticate_oauth2()
        elif self.config.auth_type == "api_key":
            self._client.headers["X-API-Key"] = self.config.api_key.get_secret_value()

        logger.info(f"CMMS connector initialized: {self.config.cmms_type.value}")

    async def close(self) -> None:
        """Close the connector."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("CMMS connector closed")

    async def __aenter__(self) -> "CMMSConnector":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _authenticate_oauth2(self) -> None:
        """Authenticate using OAuth2."""
        if not self.config.token_url:
            raise ValueError("token_url required for OAuth2 authentication")

        try:
            response = await self._client.post(
                self.config.token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret.get_secret_value(),
                },
            )
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

            self._client.headers["Authorization"] = f"Bearer {self._access_token}"
            logger.info("CMMS OAuth2 authentication successful")

        except Exception as e:
            logger.error(f"CMMS authentication failed: {e}")
            raise

    async def _ensure_authenticated(self) -> None:
        """Ensure authentication is valid."""
        if self.config.auth_type != "oauth2":
            return

        if self._token_expires_at and datetime.utcnow() >= self._token_expires_at:
            await self._authenticate_oauth2()

    # =========================================================================
    # Asset Management
    # =========================================================================

    def register_asset(self, asset: Asset) -> None:
        """Register an asset for work order creation."""
        self._assets[asset.asset_id] = asset
        logger.debug(f"Registered asset: {asset.asset_id}")

    def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Get registered asset."""
        return self._assets.get(asset_id)

    async def sync_assets_from_cmms(self) -> int:
        """
        Sync assets from CMMS.

        Returns:
            Number of assets synced
        """
        await self._ensure_authenticated()

        try:
            response = await self._client.get(
                f"{self.config.base_url}/assets",
                params={"site": self.config.site_code},
            )
            response.raise_for_status()

            assets_data = response.json()
            count = 0

            for asset_data in assets_data.get("results", []):
                # Map CMMS data to Asset model
                # This mapping would be specific to each CMMS type
                asset = self._map_cmms_asset(asset_data)
                if asset:
                    self._assets[asset.asset_id] = asset
                    count += 1

            logger.info(f"Synced {count} assets from CMMS")
            return count

        except Exception as e:
            logger.error(f"Failed to sync assets: {e}")
            raise

    def _map_cmms_asset(self, data: Dict[str, Any]) -> Optional[Asset]:
        """Map CMMS asset data to Asset model."""
        # This would be customized per CMMS type
        try:
            from integrations.cmms.cmms_schemas import AssetLocation

            return Asset(
                asset_id=data.get("assetId", data.get("id")),
                asset_tag=data.get("assetTag", data.get("tag")),
                name=data.get("name", data.get("description")),
                asset_type=AssetType(data.get("type", "other").lower()),
                location=AssetLocation(
                    site=data.get("site", self.config.site_code),
                    area=data.get("area"),
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to map asset: {e}")
            return None

    # =========================================================================
    # Work Order Creation
    # =========================================================================

    async def create_work_order(
        self,
        asset_id: str,
        trigger: WorkOrderTrigger,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        priority: Optional[WorkOrderPriority] = None,
        custom_title: Optional[str] = None,
        additional_tasks: Optional[List[MaintenanceTask]] = None,
        force_create: bool = False,
    ) -> Optional[WorkOrder]:
        """
        Create a work order for an asset.

        Implements idempotent write pattern to prevent duplicate work orders
        for the same condition within the configured time window.

        Args:
            asset_id: Target asset ID
            trigger: What triggered the work order
            description: Work order description
            context: Additional context data
            priority: Override default priority
            custom_title: Custom work order title
            additional_tasks: Additional maintenance tasks
            force_create: Force creation even if duplicate exists

        Returns:
            Created WorkOrder or None if duplicate prevented
        """
        # Get asset
        asset = self._assets.get(asset_id)
        if not asset:
            logger.warning(f"Asset not found: {asset_id}")
            # Create minimal asset
            from integrations.cmms.cmms_schemas import AssetLocation
            asset = Asset(
                asset_id=asset_id,
                asset_tag=asset_id,
                name=asset_id,
                asset_type=AssetType.OTHER,
                location=AssetLocation(site=self.config.site_code),
            )

        # Create context
        wo_context = WorkOrderContext(
            trigger_type=trigger.value,
            trigger_source="waterguard-gl016",
            readings=context.get("readings", {}) if context else {},
            notes=context.get("notes", "") if context else "",
        )

        # Get template
        template = self.TEMPLATES.get(trigger.value)
        if not template:
            # Create generic work order
            work_order = WorkOrder(
                work_order_type=WorkOrderType.CORRECTIVE,
                priority=priority or WorkOrderPriority.MEDIUM,
                asset_id=asset_id,
                asset_name=asset.name,
                title=custom_title or f"{asset.name} - {trigger.value.replace('_', ' ').title()}",
                description=description,
                context=wo_context,
                tasks=additional_tasks or [],
            )
        else:
            # Create from template
            work_order = template.create_work_order(
                asset=asset,
                context=wo_context,
                priority=priority or template.default_priority,
            )
            if custom_title:
                work_order.title = custom_title
            if additional_tasks:
                work_order.tasks.extend(additional_tasks)

        # Generate idempotency key
        work_order.idempotency_key = work_order.generate_idempotency_key()

        # Check for duplicates (unless force_create)
        if not force_create:
            existing = await self._idempotency_tracker.check_and_register(
                idempotency_key=work_order.idempotency_key,
                work_order_id=work_order.work_order_id,
                external_id=None,
                trigger=trigger,
                asset_id=asset_id,
            )
            if existing:
                self._work_orders_duplicates_prevented += 1
                logger.info(
                    f"Duplicate work order prevented for {asset_id}/{trigger.value}"
                )
                return None

        # Submit to CMMS
        try:
            external_id = await self._submit_to_cmms(work_order)
            work_order.work_order_number = external_id
            work_order.external_id = external_id
            work_order.status = WorkOrderStatus.PENDING_APPROVAL

            self._work_orders_created += 1

            logger.info(
                f"Created work order {work_order.work_order_id} "
                f"(external: {external_id}) for {asset_id}"
            )

            # Notify callback
            if self._on_work_order_created:
                try:
                    self._on_work_order_created(work_order)
                except Exception as e:
                    logger.error(f"Error in work order callback: {e}")

            return work_order

        except Exception as e:
            logger.error(f"Failed to create work order: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    async def _submit_to_cmms(self, work_order: WorkOrder) -> str:
        """
        Submit work order to CMMS backend.

        Returns:
            External work order ID
        """
        await self._ensure_authenticated()

        # Build payload based on CMMS type
        payload = self._build_cmms_payload(work_order)

        if self.config.cmms_type == CMMSType.MOCK:
            # Mock mode for testing
            await asyncio.sleep(0.1)
            return f"WO-{work_order.work_order_id.hex[:8].upper()}"

        response = await self._client.post(
            f"{self.config.base_url}/work-orders",
            json=payload,
        )
        response.raise_for_status()

        result = response.json()
        return result.get("workOrderNumber", result.get("id", str(uuid4())))

    def _build_cmms_payload(self, work_order: WorkOrder) -> Dict[str, Any]:
        """Build CMMS-specific payload."""
        # Base payload
        payload = {
            "title": work_order.title,
            "description": work_order.description,
            "assetId": work_order.asset_id,
            "priority": work_order.priority.value.upper(),
            "type": work_order.work_order_type.value.upper(),
            "site": self.config.site_code,
            "requestedDate": work_order.requested_date.isoformat(),
            "tasks": [
                {
                    "sequence": task.sequence,
                    "description": task.description,
                    "estimatedMinutes": task.estimated_duration_minutes,
                }
                for task in work_order.tasks
            ],
        }

        # Add optional fields
        if self.config.work_center:
            payload["workCenter"] = self.config.work_center
        if self.config.cost_center:
            payload["costCenter"] = self.config.cost_center

        # CMMS-specific customizations
        if self.config.cmms_type == CMMSType.SAP_PM:
            payload = self._transform_for_sap(payload, work_order)
        elif self.config.cmms_type == CMMSType.MAXIMO:
            payload = self._transform_for_maximo(payload, work_order)

        return payload

    def _transform_for_sap(
        self,
        payload: Dict[str, Any],
        work_order: WorkOrder,
    ) -> Dict[str, Any]:
        """Transform payload for SAP PM."""
        return {
            "OrderType": "PM01",
            "FunctionalLocation": work_order.asset_id,
            "ShortText": payload["title"][:40],
            "LongText": payload["description"],
            "Priority": {"1": "1", "2": "2", "3": "3", "4": "4"}.get(
                payload["priority"].lower(), "3"
            ),
            "MainWorkCenter": self.config.work_center,
            "Operations": [
                {
                    "OperationNumber": str(task["sequence"] * 10).zfill(4),
                    "Description": task["description"][:40],
                    "Duration": task["estimatedMinutes"],
                    "DurationUnit": "MIN",
                }
                for task in payload["tasks"]
            ],
        }

    def _transform_for_maximo(
        self,
        payload: Dict[str, Any],
        work_order: WorkOrder,
    ) -> Dict[str, Any]:
        """Transform payload for IBM Maximo."""
        return {
            "siteid": self.config.site_code,
            "assetnum": work_order.asset_id,
            "description": payload["title"],
            "description_longdescription": payload["description"],
            "wopriority": {"low": 4, "medium": 3, "high": 2, "critical": 1}.get(
                payload["priority"].lower(), 3
            ),
            "worktype": payload["type"],
            "woclass": "WORKORDER",
        }

    # =========================================================================
    # Work Order Status
    # =========================================================================

    async def get_work_order_status(
        self,
        work_order_id: Union[UUID, str],
    ) -> Optional[WorkOrderStatus]:
        """
        Get current status of a work order.

        Args:
            work_order_id: Work order ID (internal or external)

        Returns:
            Current status or None if not found
        """
        await self._ensure_authenticated()

        try:
            response = await self._client.get(
                f"{self.config.base_url}/work-orders/{work_order_id}",
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            status_str = data.get("status", "").lower()
            status_map = {
                "draft": WorkOrderStatus.DRAFT,
                "pending": WorkOrderStatus.PENDING_APPROVAL,
                "approved": WorkOrderStatus.APPROVED,
                "scheduled": WorkOrderStatus.SCHEDULED,
                "inprogress": WorkOrderStatus.IN_PROGRESS,
                "in_progress": WorkOrderStatus.IN_PROGRESS,
                "onhold": WorkOrderStatus.ON_HOLD,
                "on_hold": WorkOrderStatus.ON_HOLD,
                "completed": WorkOrderStatus.COMPLETED,
                "cancelled": WorkOrderStatus.CANCELLED,
                "closed": WorkOrderStatus.CLOSED,
            }
            return status_map.get(status_str, WorkOrderStatus.PENDING_APPROVAL)

        except Exception as e:
            logger.error(f"Failed to get work order status: {e}")
            return None

    async def update_work_order_status(
        self,
        work_order_id: Union[UUID, str],
        new_status: WorkOrderStatus,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Update work order status.

        Args:
            work_order_id: Work order ID
            new_status: New status
            notes: Optional status change notes

        Returns:
            True if successful
        """
        await self._ensure_authenticated()

        try:
            payload = {
                "status": new_status.value.upper(),
            }
            if notes:
                payload["statusNotes"] = notes

            response = await self._client.patch(
                f"{self.config.base_url}/work-orders/{work_order_id}/status",
                json=payload,
            )
            response.raise_for_status()

            logger.info(f"Updated work order {work_order_id} status to {new_status}")
            return True

        except Exception as e:
            logger.error(f"Failed to update work order status: {e}")
            return False

    # =========================================================================
    # Metrics
    # =========================================================================

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get connector metrics."""
        return {
            "work_orders_created": self._work_orders_created,
            "duplicates_prevented": self._work_orders_duplicates_prevented,
            "registered_assets": len(self._assets),
        }
