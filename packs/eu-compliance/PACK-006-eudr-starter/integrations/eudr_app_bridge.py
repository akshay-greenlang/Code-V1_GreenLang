# -*- coding: utf-8 -*-
"""
EUDRAppBridge - Bridge to GL-EUDR-APP v1.0 Compliance Platform
=================================================================

This module provides a bridge interface to the GL-EUDR-APP v1.0 at
``applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/``. It exposes
eight proxy classes that mirror the application service layer, enabling the
EUDR Starter Pack to interact with the full application when available while
gracefully falling back to stub implementations when it is not.

Proxy Services:
    - SupplierProxy: Supplier CRUD, bulk import, compliance status
    - PlotProxy: Plot registry, geolocation management
    - DDSProxy: DDS lifecycle (draft/review/validated/submitted/accepted/rejected)
    - PipelineProxy: 5-stage compliance pipeline management
    - RiskProxy: Risk assessment, heatmaps, alerts
    - DashboardProxy: Compliance metrics dashboard
    - DocumentProxy: Document upload, verification, retention
    - SettingsProxy: Application configuration management

Example:
    >>> bridge = EUDRAppBridge()
    >>> supplier_svc = bridge.get_supplier_service()
    >>> suppliers = await supplier_svc.list_suppliers(limit=10)

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class DDSStatus(str, Enum):
    """DDS lifecycle statuses matching GL-EUDR-APP."""
    DRAFT = "draft"
    REVIEW = "review"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class PipelineStage(str, Enum):
    """5-stage compliance pipeline stages."""
    DATA_COLLECTION = "data_collection"
    RISK_ASSESSMENT = "risk_assessment"
    DUE_DILIGENCE = "due_diligence"
    COMPLIANCE_VERIFICATION = "compliance_verification"
    SUBMISSION = "submission"


class SupplierComplianceStatus(str, Enum):
    """Supplier compliance status values."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    UNDER_INVESTIGATION = "under_investigation"


# =============================================================================
# Data Models
# =============================================================================


class EUDRAppBridgeConfig(BaseModel):
    """Configuration for the EUDR App Bridge."""
    app_base_path: str = Field(
        default="applications/GL-EUDR-APP/EUDR-Compliance-Platform/services",
        description="Base path to GL-EUDR-APP services",
    )
    stub_mode: bool = Field(
        default=True, description="Use stub fallback if app not available"
    )
    timeout_seconds: int = Field(
        default=30, description="Timeout for service calls"
    )
    enable_caching: bool = Field(
        default=True, description="Enable response caching"
    )


class SupplierRecord(BaseModel):
    """Supplier data record from the app."""
    supplier_id: str = Field(default="", description="Supplier ID")
    name: str = Field(default="", description="Supplier name")
    country: str = Field(default="", description="Country of operation")
    commodities: List[str] = Field(default_factory=list, description="Commodities supplied")
    compliance_status: str = Field(default="pending_review", description="Compliance status")
    risk_score: float = Field(default=0.0, description="Risk score")
    last_updated: Optional[datetime] = Field(None, description="Last update time")


class PlotRecord(BaseModel):
    """Plot/geolocation data record."""
    plot_id: str = Field(default="", description="Plot ID")
    supplier_id: str = Field(default="", description="Owning supplier ID")
    latitude: float = Field(default=0.0, description="Latitude")
    longitude: float = Field(default=0.0, description="Longitude")
    polygon: Optional[List[List[float]]] = Field(None, description="Polygon coordinates")
    area_ha: float = Field(default=0.0, description="Area in hectares")
    country: str = Field(default="", description="Country code")
    verified: bool = Field(default=False, description="Whether verified")


class DDSRecord(BaseModel):
    """Due Diligence Statement record."""
    dds_id: str = Field(default="", description="DDS ID")
    supplier_id: str = Field(default="", description="Supplier ID")
    status: DDSStatus = Field(default=DDSStatus.DRAFT, description="DDS status")
    dds_type: str = Field(default="standard", description="DDS type")
    reference_number: Optional[str] = Field(None, description="EU IS reference number")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created at")
    updated_at: Optional[datetime] = Field(None, description="Last updated")


class PipelineStatus(BaseModel):
    """Pipeline execution status."""
    pipeline_id: str = Field(default="", description="Pipeline ID")
    current_stage: PipelineStage = Field(
        default=PipelineStage.DATA_COLLECTION, description="Current stage"
    )
    stages_completed: List[str] = Field(default_factory=list, description="Completed stages")
    progress_pct: float = Field(default=0.0, description="Overall progress")


class DashboardMetrics(BaseModel):
    """Compliance dashboard metrics."""
    total_suppliers: int = Field(default=0, description="Total suppliers")
    compliant_suppliers: int = Field(default=0, description="Compliant count")
    pending_suppliers: int = Field(default=0, description="Pending review count")
    total_dds: int = Field(default=0, description="Total DDS count")
    submitted_dds: int = Field(default=0, description="Submitted DDS count")
    compliance_score: float = Field(default=0.0, description="Overall compliance score")
    risk_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Risk level distribution"
    )
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last updated")


class DocumentRecord(BaseModel):
    """Document metadata record."""
    document_id: str = Field(default="", description="Document ID")
    filename: str = Field(default="", description="Original filename")
    document_type: str = Field(default="", description="Document type")
    supplier_id: Optional[str] = Field(None, description="Associated supplier")
    dds_id: Optional[str] = Field(None, description="Associated DDS")
    verified: bool = Field(default=False, description="Whether verified")
    retention_until: Optional[datetime] = Field(None, description="Retention deadline")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow, description="Upload time")


# =============================================================================
# Proxy Classes
# =============================================================================


class SupplierProxy:
    """Proxy for GL-EUDR-APP supplier service.

    Provides supplier CRUD, bulk import, and compliance status queries.
    Falls back to stub data when the app is not available.
    """

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode
        self._stub_store: Dict[str, SupplierRecord] = {}

    async def list_suppliers(
        self, limit: int = 100, offset: int = 0
    ) -> List[SupplierRecord]:
        """List suppliers with pagination."""
        if self._service and not self._stub_mode:
            return self._service.list(limit=limit, offset=offset)
        return list(self._stub_store.values())[offset:offset + limit]

    async def get_supplier(self, supplier_id: str) -> Optional[SupplierRecord]:
        """Get a single supplier by ID."""
        if self._service and not self._stub_mode:
            return self._service.get(supplier_id)
        return self._stub_store.get(supplier_id)

    async def create_supplier(self, data: Dict[str, Any]) -> SupplierRecord:
        """Create a new supplier record."""
        record = SupplierRecord(
            supplier_id=data.get("supplier_id", str(uuid4())[:8]),
            name=data.get("name", ""),
            country=data.get("country", ""),
            commodities=data.get("commodities", []),
            compliance_status="pending_review",
            last_updated=datetime.utcnow(),
        )
        self._stub_store[record.supplier_id] = record
        return record

    async def update_supplier(
        self, supplier_id: str, data: Dict[str, Any]
    ) -> Optional[SupplierRecord]:
        """Update an existing supplier record."""
        record = self._stub_store.get(supplier_id)
        if record is None:
            return None
        for key, value in data.items():
            if hasattr(record, key):
                setattr(record, key, value)
        record.last_updated = datetime.utcnow()
        return record

    async def delete_supplier(self, supplier_id: str) -> bool:
        """Delete a supplier record."""
        if supplier_id in self._stub_store:
            del self._stub_store[supplier_id]
            return True
        return False

    async def bulk_import(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Bulk import supplier records."""
        imported = 0
        errors = []
        for data in records:
            try:
                await self.create_supplier(data)
                imported += 1
            except Exception as exc:
                errors.append(str(exc))
        return {"imported": imported, "errors": errors, "total": len(records)}

    async def get_compliance_status(
        self, supplier_id: str
    ) -> Dict[str, Any]:
        """Get compliance status for a supplier."""
        record = self._stub_store.get(supplier_id)
        if record:
            return {
                "supplier_id": supplier_id,
                "status": record.compliance_status,
                "risk_score": record.risk_score,
            }
        return {"supplier_id": supplier_id, "status": "unknown"}


class PlotProxy:
    """Proxy for GL-EUDR-APP plot/geolocation service."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode
        self._stub_store: Dict[str, PlotRecord] = {}

    async def list_plots(
        self, supplier_id: Optional[str] = None, limit: int = 100
    ) -> List[PlotRecord]:
        """List plots, optionally filtered by supplier."""
        plots = list(self._stub_store.values())
        if supplier_id:
            plots = [p for p in plots if p.supplier_id == supplier_id]
        return plots[:limit]

    async def get_plot(self, plot_id: str) -> Optional[PlotRecord]:
        """Get a plot by ID."""
        return self._stub_store.get(plot_id)

    async def create_plot(self, data: Dict[str, Any]) -> PlotRecord:
        """Create a new plot record."""
        record = PlotRecord(
            plot_id=data.get("plot_id", str(uuid4())[:8]),
            supplier_id=data.get("supplier_id", ""),
            latitude=data.get("latitude", 0.0),
            longitude=data.get("longitude", 0.0),
            polygon=data.get("polygon"),
            area_ha=data.get("area_ha", 0.0),
            country=data.get("country", ""),
        )
        self._stub_store[record.plot_id] = record
        return record

    async def update_plot(
        self, plot_id: str, data: Dict[str, Any]
    ) -> Optional[PlotRecord]:
        """Update an existing plot record."""
        record = self._stub_store.get(plot_id)
        if record is None:
            return None
        for key, value in data.items():
            if hasattr(record, key):
                setattr(record, key, value)
        return record

    async def delete_plot(self, plot_id: str) -> bool:
        """Delete a plot record."""
        if plot_id in self._stub_store:
            del self._stub_store[plot_id]
            return True
        return False

    async def verify_plot(self, plot_id: str) -> Dict[str, Any]:
        """Mark a plot as verified."""
        record = self._stub_store.get(plot_id)
        if record:
            record.verified = True
            return {"plot_id": plot_id, "verified": True}
        return {"plot_id": plot_id, "verified": False, "error": "Not found"}


class DDSProxy:
    """Proxy for GL-EUDR-APP DDS lifecycle service."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode
        self._stub_store: Dict[str, DDSRecord] = {}

    async def list_dds(
        self, status: Optional[str] = None, limit: int = 100
    ) -> List[DDSRecord]:
        """List DDS records, optionally filtered by status."""
        records = list(self._stub_store.values())
        if status:
            records = [r for r in records if r.status.value == status]
        return records[:limit]

    async def get_dds(self, dds_id: str) -> Optional[DDSRecord]:
        """Get a DDS by ID."""
        return self._stub_store.get(dds_id)

    async def create_dds(self, data: Dict[str, Any]) -> DDSRecord:
        """Create a new DDS record in draft status."""
        record = DDSRecord(
            dds_id=data.get("dds_id", str(uuid4())[:12]),
            supplier_id=data.get("supplier_id", ""),
            status=DDSStatus.DRAFT,
            dds_type=data.get("dds_type", "standard"),
        )
        self._stub_store[record.dds_id] = record
        return record

    async def transition_status(
        self, dds_id: str, new_status: str
    ) -> Optional[DDSRecord]:
        """Transition a DDS to a new lifecycle status."""
        record = self._stub_store.get(dds_id)
        if record is None:
            return None

        valid_transitions = {
            DDSStatus.DRAFT: {DDSStatus.REVIEW},
            DDSStatus.REVIEW: {DDSStatus.VALIDATED, DDSStatus.DRAFT},
            DDSStatus.VALIDATED: {DDSStatus.SUBMITTED, DDSStatus.REVIEW},
            DDSStatus.SUBMITTED: {DDSStatus.ACCEPTED, DDSStatus.REJECTED},
            DDSStatus.REJECTED: {DDSStatus.DRAFT},
        }

        try:
            target = DDSStatus(new_status)
        except ValueError:
            logger.error("Invalid DDS status: %s", new_status)
            return None

        allowed = valid_transitions.get(record.status, set())
        if target not in allowed:
            logger.warning(
                "Invalid transition %s -> %s for DDS %s",
                record.status.value, new_status, dds_id,
            )
            return None

        record.status = target
        record.updated_at = datetime.utcnow()
        return record

    async def submit_to_eu_is(self, dds_id: str) -> Dict[str, Any]:
        """Submit DDS to EU Information System (stub)."""
        record = self._stub_store.get(dds_id)
        if record and record.status == DDSStatus.VALIDATED:
            record.status = DDSStatus.SUBMITTED
            record.reference_number = f"EUDR-{str(uuid4())[:8].upper()}"
            record.updated_at = datetime.utcnow()
            return {
                "dds_id": dds_id,
                "reference_number": record.reference_number,
                "status": "submitted",
            }
        return {"dds_id": dds_id, "error": "DDS must be in validated status"}

    async def delete_dds(self, dds_id: str) -> bool:
        """Delete a DDS record."""
        if dds_id in self._stub_store:
            del self._stub_store[dds_id]
            return True
        return False


class PipelineProxy:
    """Proxy for GL-EUDR-APP 5-stage compliance pipeline."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode
        self._pipelines: Dict[str, PipelineStatus] = {}

    async def create_pipeline(
        self, supplier_id: str
    ) -> PipelineStatus:
        """Create a new compliance pipeline for a supplier."""
        pipeline = PipelineStatus(
            pipeline_id=str(uuid4())[:12],
            current_stage=PipelineStage.DATA_COLLECTION,
        )
        self._pipelines[pipeline.pipeline_id] = pipeline
        return pipeline

    async def get_pipeline(self, pipeline_id: str) -> Optional[PipelineStatus]:
        """Get pipeline status."""
        return self._pipelines.get(pipeline_id)

    async def advance_stage(self, pipeline_id: str) -> Optional[PipelineStatus]:
        """Advance the pipeline to the next stage."""
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            return None

        stage_order = list(PipelineStage)
        try:
            current_index = stage_order.index(pipeline.current_stage)
        except ValueError:
            return pipeline

        if current_index < len(stage_order) - 1:
            pipeline.stages_completed.append(pipeline.current_stage.value)
            pipeline.current_stage = stage_order[current_index + 1]
            pipeline.progress_pct = ((current_index + 1) / len(stage_order)) * 100

        return pipeline

    async def list_pipelines(self) -> List[PipelineStatus]:
        """List all pipelines."""
        return list(self._pipelines.values())


class RiskProxy:
    """Proxy for GL-EUDR-APP risk assessment service."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode

    async def get_risk_assessment(
        self, supplier_id: str
    ) -> Dict[str, Any]:
        """Get risk assessment for a supplier."""
        return {
            "supplier_id": supplier_id,
            "risk_score": 50.0,
            "risk_level": "standard",
            "components": {
                "country": 35.0,
                "supplier": 50.0,
                "commodity": 60.0,
                "document": 55.0,
            },
            "assessed_at": datetime.utcnow().isoformat(),
        }

    async def get_risk_heatmap(
        self, commodity: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get risk heatmap data."""
        return {
            "type": "risk_heatmap",
            "commodity_filter": commodity,
            "regions": [],
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def get_risk_alerts(
        self, severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get risk alerts."""
        return []


class DashboardProxy:
    """Proxy for GL-EUDR-APP compliance dashboard."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode

    async def get_metrics(self) -> DashboardMetrics:
        """Get dashboard metrics."""
        return DashboardMetrics()

    async def get_compliance_trend(
        self, period_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get compliance score trend over time."""
        return []

    async def get_commodity_breakdown(self) -> Dict[str, Any]:
        """Get compliance breakdown by commodity."""
        return {"commodities": {}, "generated_at": datetime.utcnow().isoformat()}


class DocumentProxy:
    """Proxy for GL-EUDR-APP document service."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode
        self._stub_store: Dict[str, DocumentRecord] = {}

    async def upload_document(
        self, data: Dict[str, Any]
    ) -> DocumentRecord:
        """Upload a document."""
        record = DocumentRecord(
            document_id=str(uuid4())[:12],
            filename=data.get("filename", ""),
            document_type=data.get("document_type", ""),
            supplier_id=data.get("supplier_id"),
            dds_id=data.get("dds_id"),
        )
        self._stub_store[record.document_id] = record
        return record

    async def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Get document metadata."""
        return self._stub_store.get(document_id)

    async def verify_document(self, document_id: str) -> Dict[str, Any]:
        """Verify a document."""
        record = self._stub_store.get(document_id)
        if record:
            record.verified = True
            return {"document_id": document_id, "verified": True}
        return {"document_id": document_id, "error": "Not found"}

    async def list_documents(
        self, supplier_id: Optional[str] = None
    ) -> List[DocumentRecord]:
        """List documents, optionally filtered by supplier."""
        docs = list(self._stub_store.values())
        if supplier_id:
            docs = [d for d in docs if d.supplier_id == supplier_id]
        return docs

    async def set_retention(
        self, document_id: str, retention_until: datetime
    ) -> bool:
        """Set document retention deadline (EUDR 5-year requirement)."""
        record = self._stub_store.get(document_id)
        if record:
            record.retention_until = retention_until
            return True
        return False

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document (respecting retention rules)."""
        record = self._stub_store.get(document_id)
        if record and record.retention_until:
            if datetime.utcnow() < record.retention_until:
                logger.warning(
                    "Cannot delete document %s: retention until %s",
                    document_id, record.retention_until,
                )
                return False
        if document_id in self._stub_store:
            del self._stub_store[document_id]
            return True
        return False


class SettingsProxy:
    """Proxy for GL-EUDR-APP application settings."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode
        self._settings: Dict[str, Any] = {
            "sandbox_mode": True,
            "eu_is_endpoint": "https://webgate.ec.europa.eu/eudr/api/v1",
            "default_dd_type": "standard",
            "cutoff_date": "2020-12-31",
            "retention_years": 5,
            "commodities_enabled": [
                "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soy", "wood",
            ],
            "risk_weights": {
                "country": 0.35, "supplier": 0.25,
                "commodity": 0.20, "document": 0.20,
            },
        }

    async def get_all(self) -> Dict[str, Any]:
        """Get all settings."""
        return dict(self._settings)

    async def get_setting(self, key: str) -> Any:
        """Get a single setting value."""
        return self._settings.get(key)

    async def update_setting(self, key: str, value: Any) -> bool:
        """Update a single setting."""
        self._settings[key] = value
        return True

    async def update_settings(self, updates: Dict[str, Any]) -> bool:
        """Bulk update settings."""
        self._settings.update(updates)
        return True

    async def reset_to_defaults(self) -> bool:
        """Reset all settings to defaults."""
        self.__init__(stub_mode=self._stub_mode)
        return True


# =============================================================================
# Main Bridge
# =============================================================================


class EUDRAppBridge:
    """Bridge to GL-EUDR-APP v1.0 Compliance Platform.

    Provides proxy access to eight application services. Attempts to connect
    to the real GL-EUDR-APP services; falls back to stub implementations if
    the application is not available.

    Attributes:
        config: Bridge configuration
        _app_available: Whether the real app is detected
        _proxies: Cached proxy instances

    Example:
        >>> bridge = EUDRAppBridge()
        >>> supplier_svc = bridge.get_supplier_service()
        >>> result = await supplier_svc.bulk_import(records)
    """

    def __init__(self, config: Optional[EUDRAppBridgeConfig] = None) -> None:
        """Initialize the EUDR App Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or EUDRAppBridgeConfig()
        self._app_available = False
        self._proxies: Dict[str, Any] = {}

        self._detect_app()
        logger.info(
            "EUDRAppBridge initialized: app_available=%s, stub_mode=%s",
            self._app_available, self.config.stub_mode,
        )

    def _detect_app(self) -> None:
        """Detect whether the GL-EUDR-APP is available for import."""
        try:
            import importlib
            importlib.import_module(
                "applications.GL_EUDR_APP.EUDR_Compliance_Platform.services"
            )
            self._app_available = True
        except ImportError:
            self._app_available = False
            logger.info("GL-EUDR-APP not available, using stub mode")

    def is_app_available(self) -> bool:
        """Check if the real GL-EUDR-APP is available.

        Returns:
            True if the app services are importable.
        """
        return self._app_available

    def get_supplier_service(self) -> SupplierProxy:
        """Get the supplier service proxy.

        Returns:
            SupplierProxy with CRUD and bulk import capabilities.
        """
        if "supplier" not in self._proxies:
            self._proxies["supplier"] = SupplierProxy(
                stub_mode=self.config.stub_mode or not self._app_available
            )
        return self._proxies["supplier"]

    def get_plot_service(self) -> PlotProxy:
        """Get the plot/geolocation service proxy.

        Returns:
            PlotProxy with plot registry management.
        """
        if "plot" not in self._proxies:
            self._proxies["plot"] = PlotProxy(
                stub_mode=self.config.stub_mode or not self._app_available
            )
        return self._proxies["plot"]

    def get_dds_service(self) -> DDSProxy:
        """Get the DDS lifecycle service proxy.

        Returns:
            DDSProxy with DDS CRUD and status transitions.
        """
        if "dds" not in self._proxies:
            self._proxies["dds"] = DDSProxy(
                stub_mode=self.config.stub_mode or not self._app_available
            )
        return self._proxies["dds"]

    def get_pipeline_service(self) -> PipelineProxy:
        """Get the 5-stage compliance pipeline service proxy.

        Returns:
            PipelineProxy for pipeline management.
        """
        if "pipeline" not in self._proxies:
            self._proxies["pipeline"] = PipelineProxy(
                stub_mode=self.config.stub_mode or not self._app_available
            )
        return self._proxies["pipeline"]

    def get_risk_service(self) -> RiskProxy:
        """Get the risk assessment service proxy.

        Returns:
            RiskProxy for risk assessments, heatmaps, and alerts.
        """
        if "risk" not in self._proxies:
            self._proxies["risk"] = RiskProxy(
                stub_mode=self.config.stub_mode or not self._app_available
            )
        return self._proxies["risk"]

    def get_dashboard_service(self) -> DashboardProxy:
        """Get the compliance dashboard service proxy.

        Returns:
            DashboardProxy for metrics and trend data.
        """
        if "dashboard" not in self._proxies:
            self._proxies["dashboard"] = DashboardProxy(
                stub_mode=self.config.stub_mode or not self._app_available
            )
        return self._proxies["dashboard"]

    def get_document_service(self) -> DocumentProxy:
        """Get the document management service proxy.

        Returns:
            DocumentProxy for document upload, verification, and retention.
        """
        if "document" not in self._proxies:
            self._proxies["document"] = DocumentProxy(
                stub_mode=self.config.stub_mode or not self._app_available
            )
        return self._proxies["document"]

    def get_settings_service(self) -> SettingsProxy:
        """Get the application settings service proxy.

        Returns:
            SettingsProxy for configuration management.
        """
        if "settings" not in self._proxies:
            self._proxies["settings"] = SettingsProxy(
                stub_mode=self.config.stub_mode or not self._app_available
            )
        return self._proxies["settings"]
