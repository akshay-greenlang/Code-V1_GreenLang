# -*- coding: utf-8 -*-
"""
GL-DATA-X-004: GreenLang ERP/Finance Connector Service SDK
===========================================================

This package provides ERP system connectivity, spend data extraction,
purchase order processing, inventory tracking, Scope 3 category mapping,
spend-based emissions calculation, currency conversion, and provenance
tracking SDK for the GreenLang framework. It supports:

- 10 ERP systems (SAP S/4HANA, SAP ECC, Oracle Cloud, Oracle EBS,
  NetSuite, Dynamics 365, Workday, Sage, QuickBooks, Simulated)
- Spend data extraction with vendor and cost center filtering
- Purchase order ingestion with line item detail
- Inventory and materials tracking
- GHG Protocol Scope 3 category mapping (16 categories)
- Spend-based emissions calculation (EEIO, hybrid, process, supplier)
- Multi-currency conversion with rate tracking
- Batch sync processing with parallel workers
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_ERP_CONNECTOR_ env prefix

Key Components:
    - config: ERPConnectorConfig with GL_ERP_CONNECTOR_ env prefix
    - models: Pydantic v2 models for all data structures
    - connection_manager: ERP connection lifecycle management
    - spend_extractor: Spend data extraction engine
    - purchase_order_engine: Purchase order processing engine
    - inventory_tracker: Inventory and materials tracking engine
    - scope3_mapper: GHG Protocol Scope 3 category mapping engine
    - emissions_calculator: Spend-based emissions calculation engine
    - currency_converter: Multi-currency conversion engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: ERPConnectorService facade

Example:
    >>> from greenlang.erp_connector import ERPConnectorService
    >>> service = ERPConnectorService()
    >>> result = service.register_connection(request)
    >>> print(result.status)
    connected

Agent ID: GL-DATA-X-004
Agent Name: ERP/Finance Connector Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-X-004"
__agent_name__ = "ERP/Finance Connector Agent"

# SDK availability flag
ERP_CONNECTOR_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.erp_connector.config import (
    ERPConnectorConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, Layer 1, SDK)
# ---------------------------------------------------------------------------
from greenlang.erp_connector.models import (
    # Layer 1 enumerations
    ERPSystem,
    Scope3Category,
    TransactionType,
    SpendCategory,
    # Layer 1 models
    ERPConnectionConfig,
    VendorMapping,
    MaterialMapping,
    PurchaseOrderLine,
    PurchaseOrder,
    SpendRecord,
    InventoryItem,
    ERPQueryInput,
    ERPQueryOutput,
    # Layer 1 constants
    SPEND_TO_SCOPE3_MAPPING,
    DEFAULT_EMISSION_FACTORS,
    # New enumerations
    ConnectionStatus,
    SyncMode,
    EmissionMethodology,
    # SDK models
    ConnectionRecord,
    SyncJob,
    SpendSummary,
    Scope3Summary,
    EmissionResult,
    CurrencyRate,
    ERPStatistics,
    # Request models
    RegisterConnectionRequest,
    SyncSpendRequest,
    MapVendorRequest,
    CalculateEmissionsRequest,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.erp_connector.connection_manager import ConnectionManager
from greenlang.erp_connector.spend_extractor import SpendExtractor
from greenlang.erp_connector.purchase_order_engine import PurchaseOrderEngine
from greenlang.erp_connector.inventory_tracker import InventoryTracker
from greenlang.erp_connector.scope3_mapper import Scope3Mapper
from greenlang.erp_connector.emissions_calculator import EmissionsCalculator
from greenlang.erp_connector.currency_converter import CurrencyConverter
from greenlang.erp_connector.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.erp_connector.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    erp_connections_total,
    erp_sync_duration_seconds,
    erp_spend_records_total,
    erp_purchase_orders_total,
    erp_scope3_mappings_total,
    erp_emissions_calculated_total,
    erp_sync_errors_total,
    erp_currency_conversions_total,
    erp_inventory_items_total,
    erp_batch_syncs_total,
    erp_active_connections,
    erp_sync_queue_size,
    # Helper functions
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
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.erp_connector.setup import (
    ERPConnectorService,
    configure_erp_connector,
    get_erp_connector,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "ERP_CONNECTOR_SDK_AVAILABLE",
    # Configuration
    "ERPConnectorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Layer 1 enumerations
    "ERPSystem",
    "Scope3Category",
    "TransactionType",
    "SpendCategory",
    # Layer 1 models
    "ERPConnectionConfig",
    "VendorMapping",
    "MaterialMapping",
    "PurchaseOrderLine",
    "PurchaseOrder",
    "SpendRecord",
    "InventoryItem",
    "ERPQueryInput",
    "ERPQueryOutput",
    # Layer 1 constants
    "SPEND_TO_SCOPE3_MAPPING",
    "DEFAULT_EMISSION_FACTORS",
    # New enumerations
    "ConnectionStatus",
    "SyncMode",
    "EmissionMethodology",
    # SDK models
    "ConnectionRecord",
    "SyncJob",
    "SpendSummary",
    "Scope3Summary",
    "EmissionResult",
    "CurrencyRate",
    "ERPStatistics",
    # Request models
    "RegisterConnectionRequest",
    "SyncSpendRequest",
    "MapVendorRequest",
    "CalculateEmissionsRequest",
    # Core engines
    "ConnectionManager",
    "SpendExtractor",
    "PurchaseOrderEngine",
    "InventoryTracker",
    "Scope3Mapper",
    "EmissionsCalculator",
    "CurrencyConverter",
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "erp_connections_total",
    "erp_sync_duration_seconds",
    "erp_spend_records_total",
    "erp_purchase_orders_total",
    "erp_scope3_mappings_total",
    "erp_emissions_calculated_total",
    "erp_sync_errors_total",
    "erp_currency_conversions_total",
    "erp_inventory_items_total",
    "erp_batch_syncs_total",
    "erp_active_connections",
    "erp_sync_queue_size",
    # Metric helper functions
    "record_connection",
    "record_spend_record",
    "record_purchase_order",
    "record_scope3_mapping",
    "record_emissions_calculated",
    "record_sync_error",
    "record_currency_conversion",
    "record_inventory_item",
    "record_batch_sync",
    "update_active_connections",
    "update_sync_queue_size",
    # Service setup facade
    "ERPConnectorService",
    "configure_erp_connector",
    "get_erp_connector",
    "get_router",
]
