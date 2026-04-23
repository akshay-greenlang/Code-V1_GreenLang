"""
CMMS Integration for FurnacePulse

Computerized Maintenance Management System integration providing:
- Asset hierarchy mapping (furnace -> zones -> burners -> tubes)
- Work order creation with evidence attachments
- Closure feedback loop for model retraining
- Maintenance history retrieval for RUL calculations

Supports common CMMS platforms: SAP PM, Maximo, Infor EAM, etc.
"""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field, HttpUrl, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Asset Hierarchy Models
# =============================================================================

class AssetType(str, Enum):
    """Asset types in furnace hierarchy."""
    SITE = "site"
    PLANT = "plant"
    FURNACE = "furnace"
    ZONE = "zone"
    BURNER = "burner"
    TUBE = "tube"
    SENSOR = "sensor"
    DAMPER = "damper"
    FAN = "fan"
    REFRACTORY = "refractory"


class AssetStatus(str, Enum):
    """Asset operational status."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    DECOMMISSIONED = "decommissioned"


class Asset(BaseModel):
    """Asset in the CMMS hierarchy."""
    asset_id: str = Field(..., description="CMMS asset ID")
    asset_type: AssetType
    name: str
    description: str = ""
    parent_id: Optional[str] = Field(None, description="Parent asset ID")
    children_ids: List[str] = Field(default_factory=list)
    status: AssetStatus = AssetStatus.OPERATIONAL
    location: str = ""
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    installation_date: Optional[str] = None
    last_maintenance_date: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class AssetHierarchy(BaseModel):
    """
    Complete asset hierarchy for a furnace.

    Represents the physical structure:
    Site -> Plant -> Furnace -> Zones -> Components (Burners, Tubes, etc.)
    """
    root_asset: Asset
    assets: Dict[str, Asset] = Field(default_factory=dict)

    def add_asset(self, asset: Asset) -> None:
        """Add asset to hierarchy."""
        self.assets[asset.asset_id] = asset

    def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Get asset by ID."""
        return self.assets.get(asset_id)

    def get_children(self, asset_id: str) -> List[Asset]:
        """Get child assets."""
        asset = self.assets.get(asset_id)
        if not asset:
            return []
        return [self.assets[cid] for cid in asset.children_ids if cid in self.assets]

    def get_parent(self, asset_id: str) -> Optional[Asset]:
        """Get parent asset."""
        asset = self.assets.get(asset_id)
        if not asset or not asset.parent_id:
            return None
        return self.assets.get(asset.parent_id)

    def get_path_to_root(self, asset_id: str) -> List[Asset]:
        """Get path from asset to root."""
        path = []
        current_id = asset_id

        while current_id:
            asset = self.assets.get(current_id)
            if not asset:
                break
            path.append(asset)
            current_id = asset.parent_id

        return path

    def get_all_of_type(self, asset_type: AssetType) -> List[Asset]:
        """Get all assets of a specific type."""
        return [a for a in self.assets.values() if a.asset_type == asset_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "root_asset": self.root_asset.dict(),
            "assets": {k: v.dict() for k, v in self.assets.items()}
        }


# =============================================================================
# Work Order Models
# =============================================================================

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
    COMPLETED = "completed"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class WorkOrderType(str, Enum):
    """Work order type."""
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    INSPECTION = "inspection"


class Attachment(BaseModel):
    """Work order attachment (evidence)."""
    attachment_id: str
    filename: str
    content_type: str = Field(..., description="MIME type")
    description: str = ""
    url: Optional[str] = Field(None, description="URL if stored in object storage")
    data_base64: Optional[str] = Field(None, description="Base64 encoded data if inline")
    size_bytes: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @classmethod
    def from_file(cls, file_path: Path, description: str = "") -> "Attachment":
        """Create attachment from file."""
        import mimetypes
        import uuid

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

        with open(file_path, "rb") as f:
            data = f.read()

        return cls(
            attachment_id=str(uuid.uuid4()),
            filename=file_path.name,
            content_type=content_type,
            description=description,
            data_base64=base64.b64encode(data).decode("utf-8"),
            size_bytes=len(data)
        )


class WorkOrderRequest(BaseModel):
    """
    Work order creation request.

    Contains all information needed to create a maintenance work order
    with supporting evidence from the predictive maintenance system.
    """
    # Asset identification
    asset_id: str = Field(..., description="CMMS asset ID")
    asset_name: str = Field("", description="Asset name for reference")

    # Alert linkage
    alert_id: str = Field(..., description="FurnacePulse alert ID")
    alert_type: str = Field(..., description="Alert type code")

    # Work order details
    title: str = Field(..., description="Work order title")
    description: str = Field(..., description="Detailed description")
    work_order_type: WorkOrderType = Field(WorkOrderType.PREDICTIVE)
    priority: WorkOrderPriority = Field(WorkOrderPriority.MEDIUM)

    # Recommended actions
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="List of recommended maintenance actions"
    )

    # RUL and timing
    estimated_rul_days: Optional[float] = Field(
        None,
        description="Remaining useful life estimate in days"
    )
    recommended_completion_date: Optional[str] = Field(
        None,
        description="Recommended completion date"
    )

    # Evidence attachments
    attachments: List[Attachment] = Field(
        default_factory=list,
        description="Supporting evidence (plots, IR images, SHAP explanations)"
    )

    # Model information
    model_id: Optional[str] = Field(None, description="Model that generated prediction")
    model_version: Optional[str] = Field(None)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    feature_importance: Optional[Dict[str, float]] = Field(None, description="SHAP values")

    # Additional context
    related_tags: List[str] = Field(default_factory=list)
    additional_notes: str = ""


class WorkOrder(BaseModel):
    """Work order created in CMMS."""
    work_order_id: str
    work_order_number: str  # Human-readable number
    asset_id: str
    alert_id: str
    title: str
    description: str
    work_order_type: WorkOrderType
    priority: WorkOrderPriority
    status: WorkOrderStatus
    recommended_actions: List[str]
    attachments: List[Attachment]
    created_at: str
    updated_at: str
    scheduled_start: Optional[str] = None
    scheduled_end: Optional[str] = None
    actual_start: Optional[str] = None
    actual_end: Optional[str] = None
    assigned_to: Optional[str] = None
    completed_by: Optional[str] = None
    closure_notes: Optional[str] = None
    failure_code: Optional[str] = None
    root_cause: Optional[str] = None
    labor_hours: Optional[float] = None
    parts_used: List[Dict[str, Any]] = Field(default_factory=list)


class WorkOrderClosureReport(BaseModel):
    """
    Work order closure report for model feedback.

    Used to feed maintenance outcomes back to ML models for retraining.
    """
    work_order_id: str
    work_order_number: str
    asset_id: str
    alert_id: str

    # Timing
    scheduled_completion: Optional[str] = None
    actual_completion: str

    # Outcome
    was_failure_found: bool = Field(..., description="Was the predicted failure confirmed?")
    failure_mode: Optional[str] = Field(None, description="Actual failure mode if found")
    failure_severity: Optional[str] = Field(None)
    root_cause_analysis: Optional[str] = Field(None)

    # Prediction accuracy feedback
    prediction_accuracy: str = Field(
        ...,
        description="true_positive, false_positive, true_negative, false_negative"
    )
    time_to_failure_actual_days: Optional[float] = Field(
        None,
        description="Actual time to failure if predicted"
    )

    # Maintenance details
    actions_performed: List[str] = Field(default_factory=list)
    parts_replaced: List[str] = Field(default_factory=list)
    labor_hours: float = 0
    downtime_hours: float = 0

    # Notes for model improvement
    technician_notes: str = ""
    recommended_model_improvements: Optional[str] = None


class MaintenanceHistoryRecord(BaseModel):
    """Historical maintenance record for RUL calculations."""
    record_id: str
    asset_id: str
    work_order_id: Optional[str] = None
    maintenance_type: str
    description: str
    performed_date: str
    performed_by: Optional[str] = None
    failure_mode: Optional[str] = None
    root_cause: Optional[str] = None
    actions_performed: List[str] = Field(default_factory=list)
    parts_replaced: List[str] = Field(default_factory=list)
    labor_hours: float = 0
    downtime_hours: float = 0
    cost: float = 0
    notes: str = ""


# =============================================================================
# CMMS Configuration
# =============================================================================

class CMMSType(str, Enum):
    """Supported CMMS platforms."""
    SAP_PM = "sap_pm"
    MAXIMO = "maximo"
    INFOR_EAM = "infor_eam"
    FIIX = "fiix"
    UPKEEP = "upkeep"
    GENERIC_REST = "generic_rest"


class CMMSConfig(BaseModel):
    """CMMS integration configuration."""
    cmms_type: CMMSType = Field(..., description="CMMS platform type")
    base_url: HttpUrl = Field(..., description="CMMS API base URL")

    # Authentication
    auth_type: str = Field("oauth2", description="oauth2, api_key, basic")
    client_id: Optional[str] = Field(None)
    client_secret: Optional[str] = Field(None)  # From vault
    api_key: Optional[str] = Field(None)  # From vault
    username: Optional[str] = Field(None)
    password: Optional[str] = Field(None)  # From vault
    oauth_token_url: Optional[str] = Field(None)

    # Connection settings
    timeout_seconds: int = Field(30)
    max_retries: int = Field(3)
    retry_backoff_seconds: float = Field(1.0)

    # Feature flags
    enable_attachment_upload: bool = Field(True)
    enable_closure_feedback: bool = Field(True)
    max_attachment_size_mb: int = Field(10)


# =============================================================================
# CMMS Integrator
# =============================================================================

class CMMSIntegrator:
    """
    CMMS Integration for FurnacePulse.

    Provides:
    - Asset hierarchy mapping (furnace -> zones -> burners -> tubes)
    - Work order creation with evidence attachments
    - Closure feedback loop for model retraining
    - Maintenance history retrieval for RUL

    Usage:
        config = CMMSConfig(
            cmms_type=CMMSType.SAP_PM,
            base_url="https://cmms.example.com/api"
        )

        integrator = CMMSIntegrator(config)
        await integrator.connect()

        # Get asset hierarchy
        hierarchy = await integrator.get_asset_hierarchy("FURNACE-001")

        # Create work order
        work_order = await integrator.create_work_order(request)

        # Get maintenance history
        history = await integrator.get_maintenance_history("TUBE-001")
    """

    def __init__(
        self,
        config: CMMSConfig,
        vault_client=None,
        closure_callback: Optional[Callable[[WorkOrderClosureReport], None]] = None
    ):
        """
        Initialize CMMS integrator.

        Args:
            config: CMMS configuration
            vault_client: Vault client for secrets
            closure_callback: Callback for work order closures (for model retraining)
        """
        self.config = config
        self.vault_client = vault_client
        self.closure_callback = closure_callback

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

        # Cache
        self._asset_cache: Dict[str, Asset] = {}
        self._hierarchy_cache: Dict[str, AssetHierarchy] = {}

        logger.info(f"CMMSIntegrator initialized for {config.cmms_type}")

    async def connect(self) -> None:
        """Establish connection to CMMS."""
        # Retrieve secrets from vault
        if self.vault_client:
            if self.config.auth_type == "oauth2":
                self.config.client_secret = await self.vault_client.get_secret(
                    "cmms_client_secret"
                )
            elif self.config.auth_type == "api_key":
                self.config.api_key = await self.vault_client.get_secret("cmms_api_key")

        # Initialize HTTP client
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            limits=httpx.Limits(max_connections=10)
        )

        # Authenticate
        await self._authenticate()

        logger.info("Connected to CMMS")

    async def disconnect(self) -> None:
        """Close CMMS connection."""
        if self._client:
            await self._client.aclose()
            self._client = None

        logger.info("Disconnected from CMMS")

    async def _authenticate(self) -> None:
        """Authenticate with CMMS."""
        if self.config.auth_type == "oauth2":
            await self._oauth2_authenticate()
        elif self.config.auth_type == "api_key":
            # API key is sent in headers, no separate auth needed
            pass
        elif self.config.auth_type == "basic":
            # Basic auth is sent in headers, no separate auth needed
            pass

    async def _oauth2_authenticate(self) -> None:
        """Perform OAuth2 authentication."""
        if not self.config.oauth_token_url:
            raise ValueError("OAuth token URL required for oauth2 auth")

        token_data = {
            "grant_type": "client_credentials",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret
        }

        try:
            response = await self._client.post(
                self.config.oauth_token_url,
                data=token_data
            )
            response.raise_for_status()

            token_response = response.json()
            self._access_token = token_response["access_token"]
            expires_in = token_response.get("expires_in", 3600)
            self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

            logger.info("CMMS OAuth2 authentication successful")

        except Exception as e:
            logger.error(f"CMMS authentication failed: {e}")
            raise

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {"Content-Type": "application/json"}

        if self.config.auth_type == "oauth2" and self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        elif self.config.auth_type == "api_key" and self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        elif self.config.auth_type == "basic":
            import base64
            credentials = base64.b64encode(
                f"{self.config.username}:{self.config.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to CMMS."""
        if not self._client:
            raise RuntimeError("Not connected to CMMS")

        # Check token expiration
        if (
            self.config.auth_type == "oauth2" and
            self._token_expires_at and
            datetime.utcnow() >= self._token_expires_at - timedelta(minutes=5)
        ):
            await self._oauth2_authenticate()

        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()

        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=headers
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    # Re-authenticate and retry
                    await self._authenticate()
                    continue
                elif e.response.status_code >= 500:
                    # Server error - retry with backoff
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(
                            self.config.retry_backoff_seconds * (attempt + 1)
                        )
                        continue
                raise

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(
                        self.config.retry_backoff_seconds * (attempt + 1)
                    )
                    continue
                raise

        raise RuntimeError(f"Max retries exceeded for {endpoint}")

    # =========================================================================
    # Asset Hierarchy
    # =========================================================================

    async def get_asset_hierarchy(
        self,
        root_asset_id: str,
        max_depth: int = 10
    ) -> AssetHierarchy:
        """
        Get complete asset hierarchy starting from a root asset.

        Args:
            root_asset_id: Root asset ID (typically furnace)
            max_depth: Maximum hierarchy depth to traverse

        Returns:
            AssetHierarchy containing all descendant assets
        """
        # Check cache
        if root_asset_id in self._hierarchy_cache:
            return self._hierarchy_cache[root_asset_id]

        # Get root asset
        root = await self.get_asset(root_asset_id)
        if not root:
            raise ValueError(f"Asset not found: {root_asset_id}")

        hierarchy = AssetHierarchy(root_asset=root)
        hierarchy.add_asset(root)

        # Recursively get children
        await self._load_children(hierarchy, root_asset_id, current_depth=0, max_depth=max_depth)

        # Cache
        self._hierarchy_cache[root_asset_id] = hierarchy

        logger.info(
            f"Loaded asset hierarchy for {root_asset_id}: "
            f"{len(hierarchy.assets)} assets"
        )

        return hierarchy

    async def _load_children(
        self,
        hierarchy: AssetHierarchy,
        parent_id: str,
        current_depth: int,
        max_depth: int
    ) -> None:
        """Recursively load child assets."""
        if current_depth >= max_depth:
            return

        try:
            # Get children from CMMS
            response = await self._request(
                "GET",
                f"/assets/{parent_id}/children"
            )

            children = response.get("children", [])

            for child_data in children:
                child = Asset(
                    asset_id=child_data["id"],
                    asset_type=AssetType(child_data.get("type", "sensor")),
                    name=child_data["name"],
                    description=child_data.get("description", ""),
                    parent_id=parent_id,
                    status=AssetStatus(child_data.get("status", "operational")),
                    location=child_data.get("location", ""),
                    attributes=child_data.get("attributes", {})
                )

                hierarchy.add_asset(child)

                # Update parent's children list
                parent = hierarchy.get_asset(parent_id)
                if parent and child.asset_id not in parent.children_ids:
                    parent.children_ids.append(child.asset_id)

                # Recursively load this child's children
                await self._load_children(
                    hierarchy, child.asset_id, current_depth + 1, max_depth
                )

        except Exception as e:
            logger.warning(f"Failed to load children for {parent_id}: {e}")

    async def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Get single asset by ID."""
        # Check cache
        if asset_id in self._asset_cache:
            return self._asset_cache[asset_id]

        try:
            response = await self._request("GET", f"/assets/{asset_id}")

            asset = Asset(
                asset_id=response["id"],
                asset_type=AssetType(response.get("type", "sensor")),
                name=response["name"],
                description=response.get("description", ""),
                parent_id=response.get("parent_id"),
                status=AssetStatus(response.get("status", "operational")),
                location=response.get("location", ""),
                manufacturer=response.get("manufacturer"),
                model=response.get("model"),
                serial_number=response.get("serial_number"),
                installation_date=response.get("installation_date"),
                last_maintenance_date=response.get("last_maintenance_date"),
                attributes=response.get("attributes", {})
            )

            self._asset_cache[asset_id] = asset
            return asset

        except Exception as e:
            logger.error(f"Failed to get asset {asset_id}: {e}")
            return None

    async def map_furnace_to_cmms_assets(
        self,
        furnace_id: str,
        zone_ids: List[str],
        burner_ids: List[str],
        tube_ids: List[str]
    ) -> Dict[str, str]:
        """
        Map FurnacePulse component IDs to CMMS asset IDs.

        Args:
            furnace_id: FurnacePulse furnace ID
            zone_ids: List of zone IDs
            burner_ids: List of burner IDs
            tube_ids: List of tube IDs

        Returns:
            Dictionary mapping FurnacePulse IDs to CMMS asset IDs
        """
        mapping = {}

        # Query CMMS for assets matching these IDs
        try:
            response = await self._request(
                "POST",
                "/assets/search",
                data={
                    "external_ids": [furnace_id] + zone_ids + burner_ids + tube_ids
                }
            )

            for asset_data in response.get("assets", []):
                external_id = asset_data.get("external_id")
                if external_id:
                    mapping[external_id] = asset_data["id"]

        except Exception as e:
            logger.error(f"Failed to map assets: {e}")

        return mapping

    # =========================================================================
    # Work Order Management
    # =========================================================================

    async def create_work_order(self, request: WorkOrderRequest) -> WorkOrder:
        """
        Create a maintenance work order with evidence attachments.

        Args:
            request: Work order request with details and attachments

        Returns:
            Created work order

        Raises:
            Exception: If work order creation fails
        """
        logger.info(
            f"Creating work order for asset {request.asset_id}, "
            f"alert {request.alert_id}"
        )

        # Build work order payload
        payload = {
            "asset_id": request.asset_id,
            "title": request.title,
            "description": self._build_description(request),
            "type": request.work_order_type.value,
            "priority": request.priority.value,
            "external_reference": request.alert_id,
            "recommended_actions": request.recommended_actions,
            "metadata": {
                "alert_id": request.alert_id,
                "alert_type": request.alert_type,
                "model_id": request.model_id,
                "model_version": request.model_version,
                "confidence": request.confidence,
                "estimated_rul_days": request.estimated_rul_days
            }
        }

        if request.recommended_completion_date:
            payload["due_date"] = request.recommended_completion_date

        # Create work order
        try:
            response = await self._request("POST", "/work-orders", data=payload)

            work_order_id = response["id"]
            work_order_number = response["number"]

            logger.info(f"Created work order {work_order_number} (ID: {work_order_id})")

            # Upload attachments
            if request.attachments and self.config.enable_attachment_upload:
                await self._upload_attachments(work_order_id, request.attachments)

            # Construct work order object
            work_order = WorkOrder(
                work_order_id=work_order_id,
                work_order_number=work_order_number,
                asset_id=request.asset_id,
                alert_id=request.alert_id,
                title=request.title,
                description=request.description,
                work_order_type=request.work_order_type,
                priority=request.priority,
                status=WorkOrderStatus.PENDING_APPROVAL,
                recommended_actions=request.recommended_actions,
                attachments=request.attachments,
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat()
            )

            return work_order

        except Exception as e:
            logger.error(f"Failed to create work order: {e}")
            raise

    def _build_description(self, request: WorkOrderRequest) -> str:
        """Build detailed work order description."""
        lines = [
            request.description,
            "",
            "--- FurnacePulse Predictive Maintenance ---",
            f"Alert ID: {request.alert_id}",
            f"Alert Type: {request.alert_type}",
        ]

        if request.estimated_rul_days is not None:
            lines.append(f"Estimated RUL: {request.estimated_rul_days:.1f} days")

        if request.confidence is not None:
            lines.append(f"Model Confidence: {request.confidence:.1%}")

        if request.model_id:
            lines.append(f"Model: {request.model_id} v{request.model_version}")

        if request.recommended_actions:
            lines.append("")
            lines.append("Recommended Actions:")
            for i, action in enumerate(request.recommended_actions, 1):
                lines.append(f"  {i}. {action}")

        if request.related_tags:
            lines.append("")
            lines.append(f"Related Sensors: {', '.join(request.related_tags)}")

        if request.feature_importance:
            lines.append("")
            lines.append("Key Contributing Factors (SHAP):")
            sorted_features = sorted(
                request.feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            for feature, importance in sorted_features:
                lines.append(f"  - {feature}: {importance:.3f}")

        if request.additional_notes:
            lines.append("")
            lines.append("Additional Notes:")
            lines.append(request.additional_notes)

        return "\n".join(lines)

    async def _upload_attachments(
        self,
        work_order_id: str,
        attachments: List[Attachment]
    ) -> None:
        """Upload attachments to work order."""
        for attachment in attachments:
            try:
                # Check size limit
                if attachment.size_bytes > self.config.max_attachment_size_mb * 1024 * 1024:
                    logger.warning(
                        f"Attachment {attachment.filename} exceeds size limit, skipping"
                    )
                    continue

                if attachment.url:
                    # Reference external URL
                    await self._request(
                        "POST",
                        f"/work-orders/{work_order_id}/attachments",
                        data={
                            "filename": attachment.filename,
                            "content_type": attachment.content_type,
                            "description": attachment.description,
                            "url": attachment.url
                        }
                    )
                elif attachment.data_base64:
                    # Upload inline data
                    await self._request(
                        "POST",
                        f"/work-orders/{work_order_id}/attachments",
                        data={
                            "filename": attachment.filename,
                            "content_type": attachment.content_type,
                            "description": attachment.description,
                            "data": attachment.data_base64
                        }
                    )

                logger.debug(f"Uploaded attachment: {attachment.filename}")

            except Exception as e:
                logger.error(f"Failed to upload attachment {attachment.filename}: {e}")

    async def get_work_order(self, work_order_id: str) -> Optional[WorkOrder]:
        """Get work order by ID."""
        try:
            response = await self._request("GET", f"/work-orders/{work_order_id}")
            return self._parse_work_order(response)
        except Exception as e:
            logger.error(f"Failed to get work order {work_order_id}: {e}")
            return None

    async def get_work_orders_for_asset(
        self,
        asset_id: str,
        status: Optional[WorkOrderStatus] = None,
        limit: int = 100
    ) -> List[WorkOrder]:
        """Get work orders for an asset."""
        try:
            params = {"asset_id": asset_id, "limit": limit}
            if status:
                params["status"] = status.value

            response = await self._request(
                "GET",
                "/work-orders",
                params=params
            )

            return [self._parse_work_order(wo) for wo in response.get("work_orders", [])]

        except Exception as e:
            logger.error(f"Failed to get work orders for {asset_id}: {e}")
            return []

    def _parse_work_order(self, data: Dict[str, Any]) -> WorkOrder:
        """Parse work order from API response."""
        return WorkOrder(
            work_order_id=data["id"],
            work_order_number=data["number"],
            asset_id=data["asset_id"],
            alert_id=data.get("external_reference", ""),
            title=data["title"],
            description=data.get("description", ""),
            work_order_type=WorkOrderType(data.get("type", "predictive")),
            priority=WorkOrderPriority(data.get("priority", "medium")),
            status=WorkOrderStatus(data.get("status", "draft")),
            recommended_actions=data.get("recommended_actions", []),
            attachments=[],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            scheduled_start=data.get("scheduled_start"),
            scheduled_end=data.get("scheduled_end"),
            actual_start=data.get("actual_start"),
            actual_end=data.get("actual_end"),
            assigned_to=data.get("assigned_to"),
            completed_by=data.get("completed_by"),
            closure_notes=data.get("closure_notes"),
            failure_code=data.get("failure_code"),
            root_cause=data.get("root_cause"),
            labor_hours=data.get("labor_hours"),
            parts_used=data.get("parts_used", [])
        )

    # =========================================================================
    # Closure Feedback Loop
    # =========================================================================

    async def process_work_order_closure(
        self,
        work_order_id: str
    ) -> Optional[WorkOrderClosureReport]:
        """
        Process work order closure and extract feedback for model retraining.

        Args:
            work_order_id: Closed work order ID

        Returns:
            Closure report for model feedback
        """
        if not self.config.enable_closure_feedback:
            return None

        work_order = await self.get_work_order(work_order_id)
        if not work_order:
            return None

        if work_order.status != WorkOrderStatus.CLOSED:
            logger.warning(f"Work order {work_order_id} is not closed")
            return None

        # Determine prediction accuracy
        prediction_accuracy = self._determine_prediction_accuracy(work_order)

        report = WorkOrderClosureReport(
            work_order_id=work_order.work_order_id,
            work_order_number=work_order.work_order_number,
            asset_id=work_order.asset_id,
            alert_id=work_order.alert_id,
            scheduled_completion=work_order.scheduled_end,
            actual_completion=work_order.actual_end or datetime.utcnow().isoformat(),
            was_failure_found=work_order.failure_code is not None,
            failure_mode=work_order.failure_code,
            root_cause_analysis=work_order.root_cause,
            prediction_accuracy=prediction_accuracy,
            actions_performed=work_order.recommended_actions,
            parts_replaced=[p.get("name", "") for p in work_order.parts_used],
            labor_hours=work_order.labor_hours or 0,
            technician_notes=work_order.closure_notes or ""
        )

        # Invoke callback for model retraining
        if self.closure_callback:
            try:
                self.closure_callback(report)
            except Exception as e:
                logger.error(f"Closure callback failed: {e}")

        logger.info(
            f"Processed closure for WO {work_order.work_order_number}: "
            f"accuracy={prediction_accuracy}"
        )

        return report

    def _determine_prediction_accuracy(self, work_order: WorkOrder) -> str:
        """Determine if prediction was accurate based on work order outcome."""
        # True Positive: Predicted failure, failure was found
        # False Positive: Predicted failure, no failure found
        # We don't have True/False Negative in this context since
        # work orders are only created when we predict something

        if work_order.failure_code:
            return "true_positive"
        else:
            # Check closure notes for false positive indicators
            notes = (work_order.closure_notes or "").lower()
            if "no issue" in notes or "no problem" in notes or "false" in notes:
                return "false_positive"
            # Default to true positive if maintenance was performed
            return "true_positive"

    async def subscribe_to_closures(
        self,
        callback: Callable[[WorkOrderClosureReport], None],
        poll_interval_seconds: int = 60
    ) -> asyncio.Task:
        """
        Subscribe to work order closures for feedback loop.

        Args:
            callback: Callback for each closure
            poll_interval_seconds: Polling interval

        Returns:
            Background task
        """
        self.closure_callback = callback

        async def poll_loop():
            last_check = datetime.utcnow()

            while True:
                try:
                    await asyncio.sleep(poll_interval_seconds)

                    # Query recently closed work orders
                    response = await self._request(
                        "GET",
                        "/work-orders",
                        params={
                            "status": "closed",
                            "closed_after": last_check.isoformat()
                        }
                    )

                    for wo_data in response.get("work_orders", []):
                        await self.process_work_order_closure(wo_data["id"])

                    last_check = datetime.utcnow()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error polling for closures: {e}")

        task = asyncio.create_task(poll_loop())
        logger.info("Started work order closure subscription")
        return task

    # =========================================================================
    # Maintenance History for RUL
    # =========================================================================

    async def get_maintenance_history(
        self,
        asset_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[MaintenanceHistoryRecord]:
        """
        Get maintenance history for RUL calculations.

        Args:
            asset_id: Asset ID
            start_date: Start of date range
            end_date: End of date range
            limit: Maximum records to return

        Returns:
            List of maintenance history records
        """
        try:
            params = {"asset_id": asset_id, "limit": limit}

            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()

            response = await self._request(
                "GET",
                "/maintenance-history",
                params=params
            )

            records = []
            for record_data in response.get("records", []):
                record = MaintenanceHistoryRecord(
                    record_id=record_data["id"],
                    asset_id=record_data["asset_id"],
                    work_order_id=record_data.get("work_order_id"),
                    maintenance_type=record_data.get("type", "unknown"),
                    description=record_data.get("description", ""),
                    performed_date=record_data["performed_date"],
                    performed_by=record_data.get("performed_by"),
                    failure_mode=record_data.get("failure_mode"),
                    root_cause=record_data.get("root_cause"),
                    actions_performed=record_data.get("actions", []),
                    parts_replaced=record_data.get("parts", []),
                    labor_hours=record_data.get("labor_hours", 0),
                    downtime_hours=record_data.get("downtime_hours", 0),
                    cost=record_data.get("cost", 0),
                    notes=record_data.get("notes", "")
                )
                records.append(record)

            logger.info(f"Retrieved {len(records)} maintenance records for {asset_id}")
            return records

        except Exception as e:
            logger.error(f"Failed to get maintenance history for {asset_id}: {e}")
            return []

    async def get_last_maintenance_date(self, asset_id: str) -> Optional[datetime]:
        """Get date of last maintenance for an asset."""
        history = await self.get_maintenance_history(asset_id, limit=1)

        if history:
            return datetime.fromisoformat(history[0].performed_date)

        return None

    async def get_failure_history(
        self,
        asset_id: str,
        failure_mode: Optional[str] = None
    ) -> List[MaintenanceHistoryRecord]:
        """Get failure history for failure mode analysis."""
        history = await self.get_maintenance_history(asset_id)

        # Filter to only failure records
        failures = [r for r in history if r.failure_mode]

        if failure_mode:
            failures = [r for r in failures if r.failure_mode == failure_mode]

        return failures
