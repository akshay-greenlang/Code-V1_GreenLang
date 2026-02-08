# -*- coding: utf-8 -*-
"""
ERP/Finance Connector Service Setup - AGENT-DATA-003: ERP Connector

Provides ``configure_erp_connector(app)`` which wires up the ERP/Finance
Connector SDK (connection manager, spend extractor, purchase order engine,
inventory tracker, Scope 3 mapper, emissions calculator, currency converter,
provenance tracker) and mounts the REST API.

Also exposes ``get_erp_connector(app)`` for programmatic access
and the ``ERPConnectorService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.erp_connector.setup import configure_erp_connector
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_erp_connector(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-003 ERP/Finance Connector
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.erp_connector.config import ERPConnectorConfig, get_config
from greenlang.erp_connector.metrics import (
    PROMETHEUS_AVAILABLE,
    record_connection,
    record_spend_record,
    record_purchase_order,
    record_scope3_mapping,
    record_emissions_calculated,
    record_sync_error,
    record_currency_conversion,
    record_inventory_item,
    record_batch_sync,
    update_active_connections,
    update_sync_queue_size,
    erp_sync_duration_seconds,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# Lightweight Pydantic models used by the facade
# ===================================================================


class ERPConnection(BaseModel):
    """Record representing a registered ERP connection.

    Attributes:
        connection_id: Unique identifier for this connection.
        erp_system: ERP system type (sap, oracle, netsuite, dynamics, simulated).
        host: ERP host address or URL.
        port: ERP connection port.
        username: Authentication username.
        tenant_id: Owning tenant identifier.
        database_name: ERP database or instance name.
        connection_params: Additional connection parameters.
        status: Connection status (active, inactive, error, testing).
        last_tested_at: Timestamp of last connectivity test.
        provenance_hash: SHA-256 provenance hash.
        created_at: Timestamp of registration.
    """
    connection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    erp_system: str = Field(default="simulated")
    host: str = Field(default="")
    port: int = Field(default=443)
    username: str = Field(default="")
    tenant_id: str = Field(default="default")
    database_name: Optional[str] = Field(default=None)
    connection_params: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="active")
    last_tested_at: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class ConnectionTestResult(BaseModel):
    """Result of an ERP connectivity test.

    Attributes:
        connection_id: Tested connection identifier.
        success: Whether the test passed.
        latency_ms: Round-trip latency in milliseconds.
        message: Human-readable result message.
        tested_at: Timestamp of the test.
        provenance_hash: SHA-256 provenance hash.
    """
    connection_id: str = Field(default="")
    success: bool = Field(default=True)
    latency_ms: float = Field(default=0.0)
    message: str = Field(default="Connection successful")
    tested_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class ConnectionRemovalResult(BaseModel):
    """Result of removing an ERP connection.

    Attributes:
        connection_id: Removed connection identifier.
        removed: Whether the connection was removed.
        message: Human-readable result message.
        provenance_hash: SHA-256 provenance hash.
    """
    connection_id: str = Field(default="")
    removed: bool = Field(default=True)
    message: str = Field(default="Connection removed")
    provenance_hash: str = Field(default="")


class SpendRecord(BaseModel):
    """Single spend record synced from ERP.

    Attributes:
        record_id: Unique spend record identifier.
        connection_id: Source ERP connection identifier.
        vendor_id: Vendor identifier.
        vendor_name: Vendor display name.
        amount: Spend amount in original currency.
        currency: Currency code.
        spend_category: Spend classification category.
        transaction_date: Date of the transaction.
        description: Transaction description.
        po_number: Linked purchase order number.
        scope3_category: Mapped Scope 3 category.
        provenance_hash: SHA-256 provenance hash.
    """
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    connection_id: str = Field(default="")
    vendor_id: str = Field(default="")
    vendor_name: str = Field(default="")
    amount: float = Field(default=0.0)
    currency: str = Field(default="USD")
    spend_category: str = Field(default="general")
    transaction_date: str = Field(default="")
    description: str = Field(default="")
    po_number: Optional[str] = Field(default=None)
    scope3_category: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")


class SyncResult(BaseModel):
    """Result of a data sync operation.

    Attributes:
        sync_id: Unique sync operation identifier.
        connection_id: Source ERP connection identifier.
        sync_type: Type of sync (spend, purchase_orders, inventory).
        records_synced: Number of records synced.
        records_new: Number of new records created.
        records_updated: Number of existing records updated.
        duration_seconds: Sync duration in seconds.
        status: Sync status (completed, partial, failed).
        errors: List of error messages encountered.
        started_at: Timestamp of sync start.
        completed_at: Timestamp of sync completion.
        provenance_hash: SHA-256 provenance hash.
    """
    sync_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    connection_id: str = Field(default="")
    sync_type: str = Field(default="spend")
    records_synced: int = Field(default=0)
    records_new: int = Field(default=0)
    records_updated: int = Field(default=0)
    duration_seconds: float = Field(default=0.0)
    status: str = Field(default="completed")
    errors: List[str] = Field(default_factory=list)
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    completed_at: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")


class SpendSummary(BaseModel):
    """Aggregated spend summary for a connection.

    Attributes:
        connection_id: Source ERP connection identifier.
        total_spend: Total spend amount in base currency.
        currency: Base currency code.
        record_count: Number of spend records.
        by_category: Spend breakdown by spend category.
        by_vendor: Spend breakdown by top vendors.
        period_start: Summary period start date.
        period_end: Summary period end date.
        provenance_hash: SHA-256 provenance hash.
    """
    connection_id: str = Field(default="")
    total_spend: float = Field(default=0.0)
    currency: str = Field(default="USD")
    record_count: int = Field(default=0)
    by_category: Dict[str, float] = Field(default_factory=dict)
    by_vendor: Dict[str, float] = Field(default_factory=dict)
    period_start: Optional[str] = Field(default=None)
    period_end: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")


class PurchaseOrder(BaseModel):
    """Purchase order record synced from ERP.

    Attributes:
        po_number: Purchase order number.
        connection_id: Source ERP connection identifier.
        vendor_id: Vendor identifier.
        vendor_name: Vendor display name.
        total_amount: Total PO amount.
        currency: Currency code.
        status: PO status (open, closed, cancelled, pending).
        order_date: Date the PO was created.
        delivery_date: Expected delivery date.
        line_items: PO line items.
        scope3_category: Mapped Scope 3 category.
        provenance_hash: SHA-256 provenance hash.
    """
    po_number: str = Field(default_factory=lambda: f"PO-{uuid.uuid4().hex[:8].upper()}")
    connection_id: str = Field(default="")
    vendor_id: str = Field(default="")
    vendor_name: str = Field(default="")
    total_amount: float = Field(default=0.0)
    currency: str = Field(default="USD")
    status: str = Field(default="open")
    order_date: str = Field(default="")
    delivery_date: Optional[str] = Field(default=None)
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    scope3_category: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")


class InventoryItem(BaseModel):
    """Inventory item record synced from ERP.

    Attributes:
        item_id: Unique inventory item identifier.
        connection_id: Source ERP connection identifier.
        material_id: Material identifier.
        material_name: Material display name.
        material_group: Material group classification.
        warehouse_id: Warehouse identifier.
        quantity_on_hand: Current stock quantity.
        unit_of_measure: Unit of measure.
        unit_cost: Unit cost in original currency.
        currency: Currency code.
        last_receipt_date: Date of last stock receipt.
        provenance_hash: SHA-256 provenance hash.
    """
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    connection_id: str = Field(default="")
    material_id: str = Field(default="")
    material_name: str = Field(default="")
    material_group: str = Field(default="general")
    warehouse_id: str = Field(default="")
    quantity_on_hand: float = Field(default=0.0)
    unit_of_measure: str = Field(default="EA")
    unit_cost: float = Field(default=0.0)
    currency: str = Field(default="USD")
    last_receipt_date: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")


class VendorMapping(BaseModel):
    """Mapping of a vendor to a Scope 3 category.

    Attributes:
        mapping_id: Unique mapping identifier.
        vendor_id: Vendor identifier from ERP.
        vendor_name: Vendor display name.
        category: Scope 3 category.
        spend_category: Spend classification category.
        emission_factor: Custom emission factor (kgCO2e per unit currency).
        provenance_hash: SHA-256 provenance hash.
        created_at: Timestamp of creation.
    """
    mapping_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vendor_id: str = Field(default="")
    vendor_name: str = Field(default="")
    category: str = Field(default="")
    spend_category: str = Field(default="general")
    emission_factor: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class MaterialMapping(BaseModel):
    """Mapping of a material to a Scope 3 category.

    Attributes:
        mapping_id: Unique mapping identifier.
        material_id: Material identifier from ERP.
        material_name: Material display name.
        category: Scope 3 category.
        spend_category: Spend classification category.
        emission_factor: Custom emission factor (kgCO2e per unit).
        provenance_hash: SHA-256 provenance hash.
        created_at: Timestamp of creation.
    """
    mapping_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    material_id: str = Field(default="")
    material_name: str = Field(default="")
    category: str = Field(default="")
    spend_category: str = Field(default="general")
    emission_factor: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class EmissionsResult(BaseModel):
    """Result of a Scope 3 emissions calculation.

    Attributes:
        calculation_id: Unique calculation identifier.
        connection_id: Source ERP connection identifier.
        methodology: Methodology used for calculation.
        total_emissions_kgco2e: Total calculated emissions in kgCO2e.
        by_category: Emissions breakdown by Scope 3 category.
        by_vendor: Emissions breakdown by top vendors.
        record_count: Number of spend records used.
        total_spend: Total spend amount used in calculation.
        currency: Currency of the spend amounts.
        period_start: Calculation period start date.
        period_end: Calculation period end date.
        provenance_hash: SHA-256 provenance hash.
        calculated_at: Timestamp of calculation.
    """
    calculation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    connection_id: str = Field(default="")
    methodology: str = Field(default="eeio")
    total_emissions_kgco2e: float = Field(default=0.0)
    by_category: Dict[str, float] = Field(default_factory=dict)
    by_vendor: Dict[str, float] = Field(default_factory=dict)
    record_count: int = Field(default=0)
    total_spend: float = Field(default=0.0)
    currency: str = Field(default="USD")
    period_start: Optional[str] = Field(default=None)
    period_end: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")
    calculated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class EmissionsSummary(BaseModel):
    """Aggregated emissions summary for a connection.

    Attributes:
        connection_id: Source ERP connection identifier.
        total_emissions_kgco2e: Total emissions in kgCO2e.
        total_emissions_tco2e: Total emissions in tCO2e.
        calculation_count: Number of calculations performed.
        by_category: Emissions breakdown by Scope 3 category.
        by_methodology: Emissions breakdown by methodology.
        period_start: Summary period start date.
        period_end: Summary period end date.
        provenance_hash: SHA-256 provenance hash.
    """
    connection_id: str = Field(default="")
    total_emissions_kgco2e: float = Field(default=0.0)
    total_emissions_tco2e: float = Field(default=0.0)
    calculation_count: int = Field(default=0)
    by_category: Dict[str, float] = Field(default_factory=dict)
    by_methodology: Dict[str, float] = Field(default_factory=dict)
    period_start: Optional[str] = Field(default=None)
    period_end: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")


class ERPStatistics(BaseModel):
    """Aggregate statistics for the ERP connector service.

    Attributes:
        total_connections: Total ERP connections registered.
        active_connections: Currently active connections.
        total_spend_records: Total spend records synced.
        total_purchase_orders: Total purchase orders synced.
        total_inventory_items: Total inventory items synced.
        total_vendor_mappings: Total vendor mappings created.
        total_material_mappings: Total material mappings created.
        total_emissions_calculations: Total emissions calculations performed.
        total_sync_operations: Total sync operations performed.
        total_sync_errors: Total sync errors encountered.
        total_currency_conversions: Total currency conversions performed.
        avg_sync_duration_seconds: Average sync duration in seconds.
    """
    total_connections: int = Field(default=0)
    active_connections: int = Field(default=0)
    total_spend_records: int = Field(default=0)
    total_purchase_orders: int = Field(default=0)
    total_inventory_items: int = Field(default=0)
    total_vendor_mappings: int = Field(default=0)
    total_material_mappings: int = Field(default=0)
    total_emissions_calculations: int = Field(default=0)
    total_sync_operations: int = Field(default=0)
    total_sync_errors: int = Field(default=0)
    total_currency_conversions: int = Field(default=0)
    avg_sync_duration_seconds: float = Field(default=0.0)


# ===================================================================
# Provenance helper
# ===================================================================


class _ProvenanceTracker:
    """Minimal provenance tracker recording SHA-256 audit entries.

    Attributes:
        entries: List of provenance entries.
        entry_count: Number of entries recorded.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self.entry_count: int = 0

    def record(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
        user_id: str = "system",
    ) -> str:
        """Record a provenance entry and return its hash.

        Args:
            entity_type: Type of entity (connection, spend, po, inventory, etc.).
            entity_id: Entity identifier.
            action: Action performed (register, sync, calculate, map, etc.).
            data_hash: SHA-256 hash of associated data.
            user_id: User or system that performed the action.

        Returns:
            SHA-256 hash of the provenance entry itself.
        """
        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "data_hash": data_hash,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entry_hash = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode()
        ).hexdigest()
        entry["entry_hash"] = entry_hash
        self._entries.append(entry)
        self.entry_count += 1
        return entry_hash


# ===================================================================
# ERPConnectorService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional[ERPConnectorService] = None


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class ERPConnectorService:
    """Unified facade over the ERP/Finance Connector SDK.

    Aggregates all connector engines (connection manager, spend extractor,
    purchase order engine, inventory tracker, Scope 3 mapper, emissions
    calculator, currency converter, provenance tracker) through a single
    entry point with convenience methods for common operations.

    Each method records provenance and updates self-monitoring metrics.

    Attributes:
        config: ERPConnectorConfig instance.
        provenance: _ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = ERPConnectorService()
        >>> conn = service.register_connection(
        ...     erp_system="simulated", host="localhost",
        ...     port=443, username="admin",
        ... )
        >>> print(conn.connection_id, conn.status)
    """

    def __init__(
        self,
        config: Optional[ERPConnectorConfig] = None,
    ) -> None:
        """Initialize the ERP Connector Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - ConnectionManager
        - SpendExtractor
        - PurchaseOrderEngine
        - InventoryTracker
        - Scope3Mapper
        - EmissionsCalculator
        - CurrencyConverter

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Provenance tracker
        self.provenance = _ProvenanceTracker()

        # Engine placeholders -- real implementations are injected by the
        # respective SDK modules at import time. We use a lazy-init approach
        # so that setup.py can be imported without the full SDK installed.
        self._connection_manager: Any = None
        self._spend_extractor: Any = None
        self._purchase_order_engine: Any = None
        self._inventory_tracker: Any = None
        self._scope3_mapper: Any = None
        self._emissions_calculator: Any = None
        self._currency_converter: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._connections: Dict[str, ERPConnection] = {}
        self._spend_records: Dict[str, SpendRecord] = {}
        self._purchase_orders: Dict[str, PurchaseOrder] = {}
        self._inventory_items: Dict[str, InventoryItem] = {}
        self._vendor_mappings: Dict[str, VendorMapping] = {}
        self._material_mappings: Dict[str, MaterialMapping] = {}
        self._sync_results: Dict[str, SyncResult] = {}
        self._emissions_results: Dict[str, EmissionsResult] = {}

        # Statistics
        self._stats = ERPStatistics()
        self._started = False

        logger.info("ERPConnectorService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def connection_manager(self) -> Any:
        """Get the ConnectionManager engine instance.

        Returns:
            ConnectionManager or None if not available.
        """
        return self._connection_manager

    @property
    def spend_extractor(self) -> Any:
        """Get the SpendExtractor engine instance.

        Returns:
            SpendExtractor or None if not available.
        """
        return self._spend_extractor

    @property
    def purchase_order_engine(self) -> Any:
        """Get the PurchaseOrderEngine engine instance.

        Returns:
            PurchaseOrderEngine or None if not available.
        """
        return self._purchase_order_engine

    @property
    def inventory_tracker(self) -> Any:
        """Get the InventoryTracker engine instance.

        Returns:
            InventoryTracker or None if not available.
        """
        return self._inventory_tracker

    @property
    def scope3_mapper(self) -> Any:
        """Get the Scope3Mapper engine instance.

        Returns:
            Scope3Mapper or None if not available.
        """
        return self._scope3_mapper

    @property
    def emissions_calculator(self) -> Any:
        """Get the EmissionsCalculator engine instance.

        Returns:
            EmissionsCalculator or None if not available.
        """
        return self._emissions_calculator

    @property
    def currency_converter(self) -> Any:
        """Get the CurrencyConverter engine instance.

        Returns:
            CurrencyConverter or None if not available.
        """
        return self._currency_converter

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are optional; missing imports are logged as warnings and
        the service continues in degraded mode.
        """
        try:
            from greenlang.erp_connector.connection_manager import ConnectionManager
            self._connection_manager = ConnectionManager(self.config)
        except ImportError:
            logger.warning("ConnectionManager not available; using stub")

        try:
            from greenlang.erp_connector.spend_extractor import SpendExtractor
            self._spend_extractor = SpendExtractor(self.config)
        except ImportError:
            logger.warning("SpendExtractor not available; using stub")

        try:
            from greenlang.erp_connector.purchase_order_engine import PurchaseOrderEngine
            self._purchase_order_engine = PurchaseOrderEngine(self.config)
        except ImportError:
            logger.warning("PurchaseOrderEngine not available; using stub")

        try:
            from greenlang.erp_connector.inventory_tracker import InventoryTracker
            self._inventory_tracker = InventoryTracker(self.config)
        except ImportError:
            logger.warning("InventoryTracker not available; using stub")

        try:
            from greenlang.erp_connector.scope3_mapper import Scope3Mapper
            self._scope3_mapper = Scope3Mapper(self.config)
        except ImportError:
            logger.warning("Scope3Mapper not available; using stub")

        try:
            from greenlang.erp_connector.emissions_calculator import EmissionsCalculator
            self._emissions_calculator = EmissionsCalculator(self.config)
        except ImportError:
            logger.warning("EmissionsCalculator not available; using stub")

        try:
            from greenlang.erp_connector.currency_converter import CurrencyConverter
            self._currency_converter = CurrencyConverter(self.config)
        except ImportError:
            logger.warning("CurrencyConverter not available; using stub")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def register_connection(
        self,
        erp_system: str,
        host: str,
        port: int = 443,
        username: str = "",
        password: Optional[str] = None,
        tenant_id: str = "default",
        database_name: Optional[str] = None,
        connection_params: Optional[Dict[str, Any]] = None,
    ) -> ERPConnection:
        """Register a new ERP system connection.

        Args:
            erp_system: ERP system type (sap, oracle, netsuite, dynamics, simulated).
            host: ERP host address or URL.
            port: ERP connection port.
            username: Authentication username.
            password: Authentication password (stored encrypted in production).
            tenant_id: Tenant identifier.
            database_name: ERP database or instance name.
            connection_params: Additional connection parameters.

        Returns:
            ERPConnection with registration details.

        Raises:
            ValueError: If erp_system or host is empty.
        """
        start_time = time.time()

        if not erp_system.strip():
            raise ValueError("erp_system must not be empty")
        if not host.strip():
            raise ValueError("host must not be empty")

        # Validate max connections
        active_count = sum(
            1 for c in self._connections.values() if c.status == "active"
        )
        if active_count >= self.config.max_connections:
            raise ValueError(
                f"Maximum connections ({self.config.max_connections}) reached"
            )

        connection = ERPConnection(
            erp_system=erp_system,
            host=host,
            port=port,
            username=username,
            tenant_id=tenant_id,
            database_name=database_name,
            connection_params=connection_params or {},
            status="active",
        )

        # Compute provenance hash
        connection.provenance_hash = _compute_hash(connection)

        # Store connection
        self._connections[connection.connection_id] = connection

        # Record metrics
        record_connection(erp_system, tenant_id)
        update_active_connections(1)

        # Record provenance
        self.provenance.record(
            entity_type="connection",
            entity_id=connection.connection_id,
            action="register",
            data_hash=connection.provenance_hash,
        )

        # Update statistics
        self._stats.total_connections += 1
        self._stats.active_connections = sum(
            1 for c in self._connections.values() if c.status == "active"
        )

        duration = time.time() - start_time
        logger.info(
            "Registered ERP connection %s (%s@%s:%d, tenant=%s) in %.2fs",
            connection.connection_id, erp_system, host, port,
            tenant_id, duration,
        )
        return connection

    def test_connection(
        self,
        connection_id: str,
    ) -> ConnectionTestResult:
        """Test an ERP connection's connectivity.

        Args:
            connection_id: ID of the connection to test.

        Returns:
            ConnectionTestResult with test outcome.

        Raises:
            ValueError: If connection ID is not found.
        """
        start_time = time.time()

        conn = self._connections.get(connection_id)
        if conn is None:
            raise ValueError(f"Connection {connection_id} not found")

        # Delegate to engine if available
        if self._connection_manager is not None:
            try:
                success, msg = self._connection_manager.test(conn)
            except Exception as exc:
                success = False
                msg = str(exc)
                record_sync_error(conn.erp_system, "connection")
        else:
            # Stub: simulated connections always succeed
            success = True
            msg = "Connection successful (simulated)"

        latency = (time.time() - start_time) * 1000
        conn.last_tested_at = datetime.now(timezone.utc).isoformat()

        result = ConnectionTestResult(
            connection_id=connection_id,
            success=success,
            latency_ms=latency,
            message=msg,
        )
        result.provenance_hash = _compute_hash(result)

        # Record provenance
        self.provenance.record(
            entity_type="connection",
            entity_id=connection_id,
            action="test",
            data_hash=result.provenance_hash,
        )

        logger.info(
            "Tested connection %s: success=%s, latency=%.1fms",
            connection_id, success, latency,
        )
        return result

    def remove_connection(
        self,
        connection_id: str,
    ) -> ConnectionRemovalResult:
        """Remove an ERP connection.

        Args:
            connection_id: ID of the connection to remove.

        Returns:
            ConnectionRemovalResult with removal outcome.

        Raises:
            ValueError: If connection ID is not found.
        """
        conn = self._connections.get(connection_id)
        if conn is None:
            raise ValueError(f"Connection {connection_id} not found")

        # Remove from store
        del self._connections[connection_id]
        update_active_connections(-1)

        result = ConnectionRemovalResult(
            connection_id=connection_id,
            removed=True,
            message=f"Connection {connection_id} removed",
        )
        result.provenance_hash = _compute_hash(result)

        # Record provenance
        self.provenance.record(
            entity_type="connection",
            entity_id=connection_id,
            action="remove",
            data_hash=result.provenance_hash,
        )

        # Update statistics
        self._stats.active_connections = sum(
            1 for c in self._connections.values() if c.status == "active"
        )

        logger.info("Removed connection %s", connection_id)
        return result

    def list_connections(
        self,
        tenant_id: Optional[str] = None,
    ) -> List[ERPConnection]:
        """List registered ERP connections with optional tenant filter.

        Args:
            tenant_id: Optional tenant filter.

        Returns:
            List of ERPConnection instances.
        """
        conns = list(self._connections.values())
        if tenant_id is not None:
            conns = [c for c in conns if c.tenant_id == tenant_id]
        return conns

    def get_connection(
        self,
        connection_id: str,
    ) -> Optional[ERPConnection]:
        """Get an ERP connection by ID.

        Args:
            connection_id: Connection identifier.

        Returns:
            ERPConnection or None if not found.
        """
        return self._connections.get(connection_id)

    # ------------------------------------------------------------------
    # Spend data
    # ------------------------------------------------------------------

    def sync_spend(
        self,
        connection_id: str,
        start_date: str,
        end_date: str,
        vendor_ids: Optional[List[str]] = None,
        spend_categories: Optional[List[str]] = None,
        incremental: bool = True,
    ) -> SyncResult:
        """Sync spend data from an ERP connection.

        Args:
            connection_id: ERP connection identifier.
            start_date: Sync period start date (ISO 8601).
            end_date: Sync period end date (ISO 8601).
            vendor_ids: Optional vendor ID filter.
            spend_categories: Optional spend category filter.
            incremental: Whether to use incremental sync mode.

        Returns:
            SyncResult with sync outcome.

        Raises:
            ValueError: If connection ID is not found.
        """
        start_time = time.time()

        conn = self._connections.get(connection_id)
        if conn is None:
            raise ValueError(f"Connection {connection_id} not found")

        update_sync_queue_size(1)
        record_batch_sync("submitted")

        try:
            # Delegate to engine if available
            if self._spend_extractor is not None:
                raw_records = self._spend_extractor.extract(
                    connection=conn,
                    start_date=start_date,
                    end_date=end_date,
                    vendor_ids=vendor_ids,
                    spend_categories=spend_categories,
                    incremental=incremental,
                )
                records_new = 0
                records_updated = 0
                for raw in raw_records:
                    sr = SpendRecord(
                        connection_id=connection_id,
                        **raw,
                    )
                    sr.provenance_hash = _compute_hash(sr)
                    if sr.record_id in self._spend_records:
                        records_updated += 1
                    else:
                        records_new += 1
                    self._spend_records[sr.record_id] = sr
                    record_spend_record(sr.spend_category)
                records_synced = len(raw_records)
            else:
                # Stub: no records synced
                records_synced = 0
                records_new = 0
                records_updated = 0

            duration = time.time() - start_time

            # Observe sync duration
            if PROMETHEUS_AVAILABLE and erp_sync_duration_seconds is not None:
                erp_sync_duration_seconds.observe(duration)

            sync_result = SyncResult(
                connection_id=connection_id,
                sync_type="spend",
                records_synced=records_synced,
                records_new=records_new,
                records_updated=records_updated,
                duration_seconds=duration,
                status="completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            sync_result.provenance_hash = _compute_hash(sync_result)
            self._sync_results[sync_result.sync_id] = sync_result

            # Record provenance
            self.provenance.record(
                entity_type="sync",
                entity_id=sync_result.sync_id,
                action="sync_spend",
                data_hash=sync_result.provenance_hash,
            )

            # Update statistics
            self._stats.total_spend_records += records_synced
            self._stats.total_sync_operations += 1
            self._update_avg_sync_duration(duration)

            record_batch_sync("completed")

            logger.info(
                "Spend sync %s completed: %d records (%d new, %d updated) in %.2fs",
                sync_result.sync_id, records_synced, records_new,
                records_updated, duration,
            )
            return sync_result

        except Exception as exc:
            record_sync_error(conn.erp_system, "data")
            record_batch_sync("failed")
            self._stats.total_sync_errors += 1
            logger.error("Spend sync failed: %s", exc, exc_info=True)
            raise
        finally:
            update_sync_queue_size(0)

    def get_spend(
        self,
        connection_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        vendor_ids: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[SpendRecord]:
        """Query synced spend records with filters.

        Args:
            connection_id: ERP connection identifier.
            start_date: Optional period start date filter.
            end_date: Optional period end date filter.
            vendor_ids: Optional vendor ID filter.
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List of SpendRecord instances.
        """
        records = [
            r for r in self._spend_records.values()
            if r.connection_id == connection_id
        ]

        if start_date is not None:
            records = [
                r for r in records if r.transaction_date >= start_date
            ]
        if end_date is not None:
            records = [
                r for r in records if r.transaction_date <= end_date
            ]
        if vendor_ids:
            records = [
                r for r in records if r.vendor_id in vendor_ids
            ]

        return records[offset:offset + limit]

    def get_spend_summary(
        self,
        connection_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> SpendSummary:
        """Get aggregated spend summary for a connection.

        Args:
            connection_id: ERP connection identifier.
            start_date: Optional period start date filter.
            end_date: Optional period end date filter.

        Returns:
            SpendSummary with aggregated totals.
        """
        records = self.get_spend(
            connection_id=connection_id,
            start_date=start_date,
            end_date=end_date,
            limit=self.config.sync_max_records,
        )

        total_spend = sum(r.amount for r in records)
        by_category: Dict[str, float] = {}
        by_vendor: Dict[str, float] = {}

        for r in records:
            by_category[r.spend_category] = (
                by_category.get(r.spend_category, 0.0) + r.amount
            )
            by_vendor[r.vendor_name] = (
                by_vendor.get(r.vendor_name, 0.0) + r.amount
            )

        summary = SpendSummary(
            connection_id=connection_id,
            total_spend=total_spend,
            currency=self.config.default_currency,
            record_count=len(records),
            by_category=by_category,
            by_vendor=by_vendor,
            period_start=start_date,
            period_end=end_date,
        )
        summary.provenance_hash = _compute_hash(summary)

        # Record provenance
        self.provenance.record(
            entity_type="spend_summary",
            entity_id=connection_id,
            action="summarize",
            data_hash=summary.provenance_hash,
        )

        return summary

    # ------------------------------------------------------------------
    # Purchase orders
    # ------------------------------------------------------------------

    def sync_purchase_orders(
        self,
        connection_id: str,
        start_date: str,
        end_date: str,
        statuses: Optional[List[str]] = None,
        incremental: bool = True,
    ) -> SyncResult:
        """Sync purchase orders from an ERP connection.

        Args:
            connection_id: ERP connection identifier.
            start_date: Sync period start date (ISO 8601).
            end_date: Sync period end date (ISO 8601).
            statuses: Optional PO status filter.
            incremental: Whether to use incremental sync mode.

        Returns:
            SyncResult with sync outcome.

        Raises:
            ValueError: If connection ID is not found.
        """
        start_time = time.time()

        conn = self._connections.get(connection_id)
        if conn is None:
            raise ValueError(f"Connection {connection_id} not found")

        update_sync_queue_size(1)
        record_batch_sync("submitted")

        try:
            # Delegate to engine if available
            if self._purchase_order_engine is not None:
                raw_orders = self._purchase_order_engine.sync(
                    connection=conn,
                    start_date=start_date,
                    end_date=end_date,
                    statuses=statuses,
                    incremental=incremental,
                )
                records_new = 0
                records_updated = 0
                for raw in raw_orders:
                    po = PurchaseOrder(
                        connection_id=connection_id,
                        **raw,
                    )
                    po.provenance_hash = _compute_hash(po)
                    if po.po_number in self._purchase_orders:
                        records_updated += 1
                    else:
                        records_new += 1
                    self._purchase_orders[po.po_number] = po
                    record_purchase_order(po.status)
                records_synced = len(raw_orders)
            else:
                records_synced = 0
                records_new = 0
                records_updated = 0

            duration = time.time() - start_time

            if PROMETHEUS_AVAILABLE and erp_sync_duration_seconds is not None:
                erp_sync_duration_seconds.observe(duration)

            sync_result = SyncResult(
                connection_id=connection_id,
                sync_type="purchase_orders",
                records_synced=records_synced,
                records_new=records_new,
                records_updated=records_updated,
                duration_seconds=duration,
                status="completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            sync_result.provenance_hash = _compute_hash(sync_result)
            self._sync_results[sync_result.sync_id] = sync_result

            self.provenance.record(
                entity_type="sync",
                entity_id=sync_result.sync_id,
                action="sync_purchase_orders",
                data_hash=sync_result.provenance_hash,
            )

            self._stats.total_purchase_orders += records_synced
            self._stats.total_sync_operations += 1
            self._update_avg_sync_duration(duration)

            record_batch_sync("completed")

            logger.info(
                "PO sync %s completed: %d orders (%d new, %d updated) in %.2fs",
                sync_result.sync_id, records_synced, records_new,
                records_updated, duration,
            )
            return sync_result

        except Exception as exc:
            record_sync_error(conn.erp_system, "data")
            record_batch_sync("failed")
            self._stats.total_sync_errors += 1
            logger.error("PO sync failed: %s", exc, exc_info=True)
            raise
        finally:
            update_sync_queue_size(0)

    def get_purchase_orders(
        self,
        connection_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[PurchaseOrder]:
        """Query synced purchase orders with filters.

        Args:
            connection_id: ERP connection identifier.
            start_date: Optional period start date filter.
            end_date: Optional period end date filter.
            limit: Maximum number of orders to return.
            offset: Number of orders to skip.

        Returns:
            List of PurchaseOrder instances.
        """
        orders = [
            po for po in self._purchase_orders.values()
            if po.connection_id == connection_id
        ]

        if start_date is not None:
            orders = [
                po for po in orders if po.order_date >= start_date
            ]
        if end_date is not None:
            orders = [
                po for po in orders if po.order_date <= end_date
            ]

        return orders[offset:offset + limit]

    def get_purchase_order(
        self,
        po_number: str,
    ) -> Optional[PurchaseOrder]:
        """Get a single purchase order by PO number.

        Args:
            po_number: Purchase order number.

        Returns:
            PurchaseOrder or None if not found.
        """
        return self._purchase_orders.get(po_number)

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def sync_inventory(
        self,
        connection_id: str,
        warehouse_ids: Optional[List[str]] = None,
        material_groups: Optional[List[str]] = None,
    ) -> SyncResult:
        """Sync inventory data from an ERP connection.

        Args:
            connection_id: ERP connection identifier.
            warehouse_ids: Optional warehouse ID filter.
            material_groups: Optional material group filter.

        Returns:
            SyncResult with sync outcome.

        Raises:
            ValueError: If connection ID is not found.
        """
        start_time = time.time()

        conn = self._connections.get(connection_id)
        if conn is None:
            raise ValueError(f"Connection {connection_id} not found")

        update_sync_queue_size(1)
        record_batch_sync("submitted")

        try:
            if self._inventory_tracker is not None:
                raw_items = self._inventory_tracker.sync(
                    connection=conn,
                    warehouse_ids=warehouse_ids,
                    material_groups=material_groups,
                )
                records_new = 0
                records_updated = 0
                for raw in raw_items:
                    item = InventoryItem(
                        connection_id=connection_id,
                        **raw,
                    )
                    item.provenance_hash = _compute_hash(item)
                    if item.item_id in self._inventory_items:
                        records_updated += 1
                    else:
                        records_new += 1
                    self._inventory_items[item.item_id] = item
                    record_inventory_item(item.material_group)
                records_synced = len(raw_items)
            else:
                records_synced = 0
                records_new = 0
                records_updated = 0

            duration = time.time() - start_time

            if PROMETHEUS_AVAILABLE and erp_sync_duration_seconds is not None:
                erp_sync_duration_seconds.observe(duration)

            sync_result = SyncResult(
                connection_id=connection_id,
                sync_type="inventory",
                records_synced=records_synced,
                records_new=records_new,
                records_updated=records_updated,
                duration_seconds=duration,
                status="completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            sync_result.provenance_hash = _compute_hash(sync_result)
            self._sync_results[sync_result.sync_id] = sync_result

            self.provenance.record(
                entity_type="sync",
                entity_id=sync_result.sync_id,
                action="sync_inventory",
                data_hash=sync_result.provenance_hash,
            )

            self._stats.total_inventory_items += records_synced
            self._stats.total_sync_operations += 1
            self._update_avg_sync_duration(duration)

            record_batch_sync("completed")

            logger.info(
                "Inventory sync %s completed: %d items (%d new, %d updated) in %.2fs",
                sync_result.sync_id, records_synced, records_new,
                records_updated, duration,
            )
            return sync_result

        except Exception as exc:
            record_sync_error(conn.erp_system, "data")
            record_batch_sync("failed")
            self._stats.total_sync_errors += 1
            logger.error("Inventory sync failed: %s", exc, exc_info=True)
            raise
        finally:
            update_sync_queue_size(0)

    def get_inventory(
        self,
        connection_id: str,
        warehouse_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[InventoryItem]:
        """Query synced inventory items with filters.

        Args:
            connection_id: ERP connection identifier.
            warehouse_id: Optional warehouse ID filter.
            limit: Maximum number of items to return.
            offset: Number of items to skip.

        Returns:
            List of InventoryItem instances.
        """
        items = [
            it for it in self._inventory_items.values()
            if it.connection_id == connection_id
        ]

        if warehouse_id is not None:
            items = [it for it in items if it.warehouse_id == warehouse_id]

        return items[offset:offset + limit]

    # ------------------------------------------------------------------
    # Vendor mapping
    # ------------------------------------------------------------------

    def map_vendor(
        self,
        vendor_id: str,
        vendor_name: str,
        category: str,
        spend_category: str = "general",
        emission_factor: Optional[float] = None,
    ) -> VendorMapping:
        """Map a vendor to a Scope 3 category with optional emission factor.

        Args:
            vendor_id: Vendor identifier from ERP.
            vendor_name: Vendor display name.
            category: Scope 3 category (e.g. cat1_purchased_goods).
            spend_category: Spend classification category.
            emission_factor: Custom emission factor (kgCO2e per unit currency).

        Returns:
            VendorMapping with mapping details.

        Raises:
            ValueError: If vendor_id or category is empty.
        """
        if not vendor_id.strip():
            raise ValueError("vendor_id must not be empty")
        if not category.strip():
            raise ValueError("category must not be empty")

        mapping = VendorMapping(
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            category=category,
            spend_category=spend_category,
            emission_factor=emission_factor,
        )
        mapping.provenance_hash = _compute_hash(mapping)

        self._vendor_mappings[mapping.mapping_id] = mapping

        # Record metrics
        record_scope3_mapping(category)

        # Record provenance
        self.provenance.record(
            entity_type="vendor_mapping",
            entity_id=mapping.mapping_id,
            action="map_vendor",
            data_hash=mapping.provenance_hash,
        )

        # Update statistics
        self._stats.total_vendor_mappings += 1

        logger.info(
            "Mapped vendor %s (%s) to category %s",
            vendor_id, vendor_name, category,
        )
        return mapping

    def list_vendor_mappings(self) -> List[VendorMapping]:
        """List all vendor-to-Scope-3 mappings.

        Returns:
            List of VendorMapping instances.
        """
        return list(self._vendor_mappings.values())

    # ------------------------------------------------------------------
    # Material mapping
    # ------------------------------------------------------------------

    def map_material(
        self,
        material_id: str,
        material_name: str,
        category: str,
        spend_category: str = "general",
        emission_factor: Optional[float] = None,
    ) -> MaterialMapping:
        """Map a material to a Scope 3 category with optional emission factor.

        Args:
            material_id: Material identifier from ERP.
            material_name: Material display name.
            category: Scope 3 category (e.g. cat1_purchased_goods).
            spend_category: Spend classification category.
            emission_factor: Custom emission factor (kgCO2e per unit).

        Returns:
            MaterialMapping with mapping details.

        Raises:
            ValueError: If material_id or category is empty.
        """
        if not material_id.strip():
            raise ValueError("material_id must not be empty")
        if not category.strip():
            raise ValueError("category must not be empty")

        mapping = MaterialMapping(
            material_id=material_id,
            material_name=material_name,
            category=category,
            spend_category=spend_category,
            emission_factor=emission_factor,
        )
        mapping.provenance_hash = _compute_hash(mapping)

        self._material_mappings[mapping.mapping_id] = mapping

        # Record metrics
        record_scope3_mapping(category)

        # Record provenance
        self.provenance.record(
            entity_type="material_mapping",
            entity_id=mapping.mapping_id,
            action="map_material",
            data_hash=mapping.provenance_hash,
        )

        # Update statistics
        self._stats.total_material_mappings += 1

        logger.info(
            "Mapped material %s (%s) to category %s",
            material_id, material_name, category,
        )
        return mapping

    # ------------------------------------------------------------------
    # Emissions calculation
    # ------------------------------------------------------------------

    def calculate_emissions(
        self,
        connection_id: str,
        start_date: str,
        end_date: str,
        methodology: str = "eeio",
        scope3_categories: Optional[List[str]] = None,
    ) -> EmissionsResult:
        """Calculate Scope 3 emissions from synced ERP spend data.

        This method uses ZERO-HALLUCINATION deterministic calculation
        only. Emission factors are looked up from the vendor/material
        mappings or the default EEIO factors. No LLM is used for
        numeric computation.

        Args:
            connection_id: ERP connection identifier with synced data.
            start_date: Calculation period start date (ISO 8601).
            end_date: Calculation period end date (ISO 8601).
            methodology: Emission calculation methodology.
            scope3_categories: Optional Scope 3 category filter.

        Returns:
            EmissionsResult with calculated emissions.

        Raises:
            ValueError: If connection ID is not found.
        """
        start_time = time.time()

        conn = self._connections.get(connection_id)
        if conn is None:
            raise ValueError(f"Connection {connection_id} not found")

        # Get relevant spend records
        spend_records = self.get_spend(
            connection_id=connection_id,
            start_date=start_date,
            end_date=end_date,
            limit=self.config.sync_max_records,
        )

        # Filter by Scope 3 categories if specified
        if scope3_categories:
            spend_records = [
                r for r in spend_records
                if r.scope3_category in scope3_categories
            ]

        # Delegate to engine if available
        if self._emissions_calculator is not None:
            calc_result = self._emissions_calculator.calculate(
                records=spend_records,
                vendor_mappings=self._vendor_mappings,
                material_mappings=self._material_mappings,
                methodology=methodology,
            )
            total_emissions = calc_result.get("total_emissions_kgco2e", 0.0)
            by_category = calc_result.get("by_category", {})
            by_vendor = calc_result.get("by_vendor", {})
        else:
            # Stub: deterministic EEIO-based estimation
            total_emissions, by_category, by_vendor = self._stub_calculate_emissions(
                spend_records, methodology,
            )

        total_spend = sum(r.amount for r in spend_records)
        duration = time.time() - start_time

        result = EmissionsResult(
            connection_id=connection_id,
            methodology=methodology,
            total_emissions_kgco2e=total_emissions,
            by_category=by_category,
            by_vendor=by_vendor,
            record_count=len(spend_records),
            total_spend=total_spend,
            currency=self.config.default_currency,
            period_start=start_date,
            period_end=end_date,
        )
        result.provenance_hash = _compute_hash(result)

        self._emissions_results[result.calculation_id] = result

        # Record metrics
        record_emissions_calculated(methodology)

        # Record provenance
        self.provenance.record(
            entity_type="emissions",
            entity_id=result.calculation_id,
            action="calculate",
            data_hash=result.provenance_hash,
        )

        # Update statistics
        self._stats.total_emissions_calculations += 1

        logger.info(
            "Emissions calculation %s completed: %.2f kgCO2e from %d records "
            "($%.2f spend, %s methodology) in %.2fs",
            result.calculation_id, total_emissions, len(spend_records),
            total_spend, methodology, duration,
        )
        return result

    def get_emissions_summary(
        self,
        connection_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> EmissionsSummary:
        """Get aggregated emissions summary for a connection.

        Args:
            connection_id: ERP connection identifier.
            start_date: Optional period start date filter.
            end_date: Optional period end date filter.

        Returns:
            EmissionsSummary with aggregated totals.
        """
        results = [
            r for r in self._emissions_results.values()
            if r.connection_id == connection_id
        ]

        if start_date is not None:
            results = [
                r for r in results
                if r.period_start is not None and r.period_start >= start_date
            ]
        if end_date is not None:
            results = [
                r for r in results
                if r.period_end is not None and r.period_end <= end_date
            ]

        total_kgco2e = sum(r.total_emissions_kgco2e for r in results)
        by_category: Dict[str, float] = {}
        by_methodology: Dict[str, float] = {}

        for r in results:
            for cat, val in r.by_category.items():
                by_category[cat] = by_category.get(cat, 0.0) + val
            by_methodology[r.methodology] = (
                by_methodology.get(r.methodology, 0.0) + r.total_emissions_kgco2e
            )

        summary = EmissionsSummary(
            connection_id=connection_id,
            total_emissions_kgco2e=total_kgco2e,
            total_emissions_tco2e=total_kgco2e / 1000.0,
            calculation_count=len(results),
            by_category=by_category,
            by_methodology=by_methodology,
            period_start=start_date,
            period_end=end_date,
        )
        summary.provenance_hash = _compute_hash(summary)

        self.provenance.record(
            entity_type="emissions_summary",
            entity_id=connection_id,
            action="summarize",
            data_hash=summary.provenance_hash,
        )

        return summary

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> ERPStatistics:
        """Get aggregated ERP connector statistics.

        Returns:
            ERPStatistics summary.
        """
        return self._stats

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    def get_provenance(self) -> _ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            _ProvenanceTracker used by this service.
        """
        return self.provenance

    def get_metrics(self) -> Dict[str, Any]:
        """Get ERP connector service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "total_connections": self._stats.total_connections,
            "active_connections": self._stats.active_connections,
            "total_spend_records": self._stats.total_spend_records,
            "total_purchase_orders": self._stats.total_purchase_orders,
            "total_inventory_items": self._stats.total_inventory_items,
            "total_vendor_mappings": self._stats.total_vendor_mappings,
            "total_material_mappings": self._stats.total_material_mappings,
            "total_emissions_calculations": self._stats.total_emissions_calculations,
            "total_sync_operations": self._stats.total_sync_operations,
            "total_sync_errors": self._stats.total_sync_errors,
            "provenance_entries": self.provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stub_calculate_emissions(
        self,
        spend_records: List[SpendRecord],
        methodology: str,
    ) -> tuple:
        """Stub emissions calculation using deterministic EEIO factors.

        This is the zero-hallucination fallback when the EmissionsCalculator
        engine is not available. Uses a conservative EEIO factor of
        0.42 kgCO2e per USD (EPA USEEIO average).

        Args:
            spend_records: List of spend records.
            methodology: Methodology name (for logging).

        Returns:
            Tuple of (total_emissions, by_category, by_vendor).
        """
        # Default EEIO factor: 0.42 kgCO2e per USD
        default_ef = 0.42
        total_emissions = 0.0
        by_category: Dict[str, float] = {}
        by_vendor: Dict[str, float] = {}

        for record in spend_records:
            # Look up vendor-specific emission factor
            ef = default_ef
            for vm in self._vendor_mappings.values():
                if vm.vendor_id == record.vendor_id and vm.emission_factor is not None:
                    ef = vm.emission_factor
                    break

            # Deterministic calculation: spend * emission_factor
            emission = record.amount * ef
            total_emissions += emission

            cat = record.scope3_category or "unclassified"
            by_category[cat] = by_category.get(cat, 0.0) + emission
            by_vendor[record.vendor_name] = (
                by_vendor.get(record.vendor_name, 0.0) + emission
            )

        return total_emissions, by_category, by_vendor

    def _update_avg_sync_duration(self, duration: float) -> None:
        """Update running average sync duration.

        Args:
            duration: Latest sync duration in seconds.
        """
        total = self._stats.total_sync_operations
        if total <= 0:
            self._stats.avg_sync_duration_seconds = duration
            return
        prev_avg = self._stats.avg_sync_duration_seconds
        self._stats.avg_sync_duration_seconds = (
            (prev_avg * (total - 1) + duration) / total
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the ERP connector service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("ERPConnectorService already started; skipping")
            return

        logger.info("ERPConnectorService starting up...")
        self._started = True
        logger.info("ERPConnectorService startup complete")

    def shutdown(self) -> None:
        """Shutdown the ERP connector service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("ERPConnectorService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> ERPConnectorService:
    """Get or create the singleton ERPConnectorService instance.

    Returns:
        The singleton ERPConnectorService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = ERPConnectorService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_erp_connector(
    app: Any,
    config: Optional[ERPConnectorConfig] = None,
) -> ERPConnectorService:
    """Configure the ERP Connector Service on a FastAPI application.

    Creates the ERPConnectorService, stores it in app.state, mounts
    the ERP connector API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional ERP connector config.

    Returns:
        ERPConnectorService instance.
    """
    global _singleton_instance

    service = ERPConnectorService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.erp_connector_service = service

    # Mount ERP connector API router
    try:
        from greenlang.erp_connector.api.router import router as erp_router
        if erp_router is not None:
            app.include_router(erp_router)
            logger.info("ERP connector service API router mounted")
    except ImportError:
        logger.warning("ERP connector router not available; API not mounted")

    # Start service
    service.startup()

    logger.info("ERP connector service configured on app")
    return service


def get_erp_connector(app: Any) -> ERPConnectorService:
    """Get the ERPConnectorService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        ERPConnectorService instance.

    Raises:
        RuntimeError: If ERP connector service not configured.
    """
    service = getattr(app.state, "erp_connector_service", None)
    if service is None:
        raise RuntimeError(
            "ERP connector service not configured. "
            "Call configure_erp_connector(app) first."
        )
    return service


def get_router(service: Optional[ERPConnectorService] = None) -> Any:
    """Get the ERP connector API router.

    Args:
        service: Optional service instance (unused, kept for API compat).

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.erp_connector.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "ERPConnectorService",
    "configure_erp_connector",
    "get_erp_connector",
    "get_router",
    # Models
    "ERPConnection",
    "ConnectionTestResult",
    "ConnectionRemovalResult",
    "SpendRecord",
    "SyncResult",
    "SpendSummary",
    "PurchaseOrder",
    "InventoryItem",
    "VendorMapping",
    "MaterialMapping",
    "EmissionsResult",
    "EmissionsSummary",
    "ERPStatistics",
]
