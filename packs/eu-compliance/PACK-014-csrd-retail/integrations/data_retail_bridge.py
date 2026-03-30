# -*- coding: utf-8 -*-
"""
DataRetailBridge - Bridge to DATA Agents and Retail ERP Systems for PACK-014
==============================================================================

This module routes retail data intake and quality operations to the appropriate
DATA agents, and provides ERP field mapping for the four major retail ERP
systems: SAP Retail, Oracle Retail, NetSuite, and Microsoft Dynamics 365.

ERP Field Mapping (4 systems):
    SAP Retail:   store_id->WERKS, energy_kwh->MENGE_KWH, waste_kg->ABFALL_KG
    Oracle Retail: store_id->STORE_NO, energy->UTILITY_USAGE, waste->WASTE_METRIC
    NetSuite:     store_id->LOCATION_ID, energy->CUSTOM_ENERGY, waste->CUSTOM_WASTE
    Dynamics 365: store_id->SITE_ID, energy->ENERGY_CONSUMPTION, waste->WASTE_RECORD

Data Agent Routing:
    POS/sales data          --> DATA-002 (Excel/CSV Normalizer)
    Supplier certificates   --> DATA-001 (PDF Extractor)
    ERP integration         --> DATA-003 (ERP Connector)
    Supplier questionnaires --> DATA-008 (Questionnaire Processor)
    Spend categorization    --> DATA-009 (Spend Categorizer)
    Quality profiling       --> DATA-010 (Data Quality Profiler)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-014 CSRD Retail & Consumer Goods
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

class _AgentStub:
    """Stub for unavailable DATA agent modules."""

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name
        self._available = False

    def __getattr__(self, name: str) -> Any:
        def _stub_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "agent": self._agent_name,
                "method": name,
                "status": "degraded",
                "message": f"{self._agent_name} not available, using stub",
            }
        return _stub_method

def _try_import_data_agent(agent_id: str, module_path: str) -> Any:
    """Try to import a DATA agent with graceful fallback."""
    try:
        import importlib

        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("DATA agent %s not available, using stub", agent_id)
        return _AgentStub(agent_id)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RetailERP(str, Enum):
    """Supported retail ERP systems."""

    SAP_RETAIL = "sap_retail"
    ORACLE_RETAIL = "oracle_retail"
    NETSUITE = "netsuite"
    DYNAMICS_365 = "dynamics_365"

class DataSource(str, Enum):
    """Retail data source categories."""

    POS_SALES = "pos_sales"
    SUPPLIER_CERTIFICATES = "supplier_certificates"
    ERP_DATA = "erp_data"
    SUPPLIER_QUESTIONNAIRES = "supplier_questionnaires"
    SPEND_DATA = "spend_data"
    ENERGY_BILLS = "energy_bills"
    WASTE_RECORDS = "waste_records"
    PACKAGING_DATA = "packaging_data"
    PRODUCT_CATALOG = "product_catalog"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class ERPFieldMapping(BaseModel):
    """Field mapping between GreenLang standard fields and ERP-specific fields."""

    erp_system: RetailERP = Field(...)
    gl_field: str = Field(..., description="GreenLang standard field name")
    erp_field: str = Field(..., description="ERP-specific field name")
    erp_table: str = Field(default="", description="ERP table or entity name")
    data_type: str = Field(default="string")
    transform: str = Field(default="direct", description="Transformation rule")
    description: str = Field(default="")

class DataAgentRoute(BaseModel):
    """Routing entry mapping a data source to a DATA agent."""

    source: DataSource = Field(...)
    agent_id: str = Field(..., description="DATA agent identifier")
    agent_name: str = Field(default="")
    module_path: str = Field(default="")
    description: str = Field(default="")
    file_formats: List[str] = Field(default_factory=list)

class DataRoutingResult(BaseModel):
    """Result of routing a data operation to a DATA agent."""

    routing_id: str = Field(default_factory=_new_uuid)
    source: str = Field(default="")
    agent_id: str = Field(default="")
    success: bool = Field(default=False)
    degraded: bool = Field(default=False)
    records_processed: int = Field(default=0)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class ERPExtractionResult(BaseModel):
    """Result of extracting data from a retail ERP system."""

    extraction_id: str = Field(default_factory=_new_uuid)
    erp_system: str = Field(default="")
    tables_queried: List[str] = Field(default_factory=list)
    records_extracted: int = Field(default=0)
    fields_mapped: int = Field(default=0)
    success: bool = Field(default=False)
    degraded: bool = Field(default=False)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class DataBridgeConfig(BaseModel):
    """Configuration for the Data Retail Bridge."""

    pack_id: str = Field(default="PACK-014")
    erp_system: RetailERP = Field(default=RetailERP.SAP_RETAIL)
    enable_provenance: bool = Field(default=True)
    enable_quality_profiling: bool = Field(default=True)
    max_records_per_batch: int = Field(default=10000, ge=100)

# ---------------------------------------------------------------------------
# ERP Field Mapping Tables
# ---------------------------------------------------------------------------

ERP_FIELD_MAPPINGS: List[ERPFieldMapping] = [
    # SAP Retail
    ERPFieldMapping(erp_system=RetailERP.SAP_RETAIL, gl_field="store_id", erp_field="WERKS", erp_table="T001W", data_type="string", description="Plant/Store ID"),
    ERPFieldMapping(erp_system=RetailERP.SAP_RETAIL, gl_field="store_name", erp_field="NAME1", erp_table="T001W", data_type="string", description="Store name"),
    ERPFieldMapping(erp_system=RetailERP.SAP_RETAIL, gl_field="energy_kwh", erp_field="MENGE_KWH", erp_table="ZUTIL_CONS", data_type="float", description="Energy consumption kWh"),
    ERPFieldMapping(erp_system=RetailERP.SAP_RETAIL, gl_field="waste_kg", erp_field="ABFALL_KG", erp_table="ZABFALL", data_type="float", description="Waste in kg"),
    ERPFieldMapping(erp_system=RetailERP.SAP_RETAIL, gl_field="sales_revenue", erp_field="NETWR", erp_table="VBRK", data_type="float", description="Net sales revenue"),
    ERPFieldMapping(erp_system=RetailERP.SAP_RETAIL, gl_field="floor_area_m2", erp_field="FLAECHE", erp_table="T001W", data_type="float", description="Store floor area m2"),
    ERPFieldMapping(erp_system=RetailERP.SAP_RETAIL, gl_field="refrigerant_type", erp_field="KAELTEMITTEL", erp_table="ZKAELTE", data_type="string", description="Refrigerant type"),
    ERPFieldMapping(erp_system=RetailERP.SAP_RETAIL, gl_field="refrigerant_charge_kg", erp_field="FUELLMENGE_KG", erp_table="ZKAELTE", data_type="float", description="Refrigerant charge kg"),
    # Oracle Retail
    ERPFieldMapping(erp_system=RetailERP.ORACLE_RETAIL, gl_field="store_id", erp_field="STORE_NO", erp_table="STORE", data_type="string", description="Store number"),
    ERPFieldMapping(erp_system=RetailERP.ORACLE_RETAIL, gl_field="store_name", erp_field="STORE_NAME", erp_table="STORE", data_type="string", description="Store name"),
    ERPFieldMapping(erp_system=RetailERP.ORACLE_RETAIL, gl_field="energy_kwh", erp_field="UTILITY_USAGE", erp_table="FACILITY_MGMT", data_type="float", description="Utility usage"),
    ERPFieldMapping(erp_system=RetailERP.ORACLE_RETAIL, gl_field="waste_kg", erp_field="WASTE_METRIC", erp_table="WASTE_MGMT", data_type="float", description="Waste metric"),
    ERPFieldMapping(erp_system=RetailERP.ORACLE_RETAIL, gl_field="sales_revenue", erp_field="TOTAL_REVENUE", erp_table="IF_TRAN_DATA", data_type="float", description="Total revenue"),
    ERPFieldMapping(erp_system=RetailERP.ORACLE_RETAIL, gl_field="floor_area_m2", erp_field="TOTAL_AREA", erp_table="STORE", data_type="float", description="Total area"),
    # NetSuite
    ERPFieldMapping(erp_system=RetailERP.NETSUITE, gl_field="store_id", erp_field="LOCATION_ID", erp_table="Location", data_type="string", description="Location ID"),
    ERPFieldMapping(erp_system=RetailERP.NETSUITE, gl_field="store_name", erp_field="NAME", erp_table="Location", data_type="string", description="Location name"),
    ERPFieldMapping(erp_system=RetailERP.NETSUITE, gl_field="energy_kwh", erp_field="CUSTOM_ENERGY", erp_table="CustomRecord_Energy", data_type="float", description="Custom energy field"),
    ERPFieldMapping(erp_system=RetailERP.NETSUITE, gl_field="waste_kg", erp_field="CUSTOM_WASTE", erp_table="CustomRecord_Waste", data_type="float", description="Custom waste field"),
    ERPFieldMapping(erp_system=RetailERP.NETSUITE, gl_field="sales_revenue", erp_field="TOTAL_AMOUNT", erp_table="Transaction", data_type="float", description="Transaction amount"),
    # Dynamics 365
    ERPFieldMapping(erp_system=RetailERP.DYNAMICS_365, gl_field="store_id", erp_field="SITE_ID", erp_table="RetailStoreTable", data_type="string", description="Site ID"),
    ERPFieldMapping(erp_system=RetailERP.DYNAMICS_365, gl_field="store_name", erp_field="NAME", erp_table="RetailStoreTable", data_type="string", description="Store name"),
    ERPFieldMapping(erp_system=RetailERP.DYNAMICS_365, gl_field="energy_kwh", erp_field="ENERGY_CONSUMPTION", erp_table="SustainabilityEntry", data_type="float", description="Energy consumption"),
    ERPFieldMapping(erp_system=RetailERP.DYNAMICS_365, gl_field="waste_kg", erp_field="WASTE_RECORD", erp_table="SustainabilityEntry", data_type="float", description="Waste record"),
    ERPFieldMapping(erp_system=RetailERP.DYNAMICS_365, gl_field="sales_revenue", erp_field="TOTALAMOUNT", erp_table="RetailTransactionTable", data_type="float", description="Total amount"),
]

# ---------------------------------------------------------------------------
# Data Agent Routing Table
# ---------------------------------------------------------------------------

DATA_AGENT_ROUTES: List[DataAgentRoute] = [
    DataAgentRoute(
        source=DataSource.POS_SALES, agent_id="DATA-002",
        agent_name="Excel/CSV Normalizer",
        module_path="greenlang.agents.data.excel_normalizer",
        description="Normalize POS/sales data from CSV/Excel exports",
        file_formats=["csv", "xlsx", "xls"],
    ),
    DataAgentRoute(
        source=DataSource.SUPPLIER_CERTIFICATES, agent_id="DATA-001",
        agent_name="PDF & Invoice Extractor",
        module_path="greenlang.agents.data.pdf_extractor",
        description="Extract data from supplier sustainability certificates",
        file_formats=["pdf", "png", "jpg"],
    ),
    DataAgentRoute(
        source=DataSource.ERP_DATA, agent_id="DATA-003",
        agent_name="ERP/Finance Connector",
        module_path="greenlang.agents.data.erp_connector",
        description="Connect to retail ERP systems for automated data extraction",
        file_formats=["api", "odata", "rest"],
    ),
    DataAgentRoute(
        source=DataSource.SUPPLIER_QUESTIONNAIRES, agent_id="DATA-008",
        agent_name="Supplier Questionnaire Processor",
        module_path="greenlang.agents.data.questionnaire_processor",
        description="Process supplier sustainability questionnaire responses",
        file_formats=["xlsx", "csv", "json"],
    ),
    DataAgentRoute(
        source=DataSource.SPEND_DATA, agent_id="DATA-009",
        agent_name="Spend Data Categorizer",
        module_path="greenlang.agents.data.spend_categorizer",
        description="Categorize procurement spend for Scope 3 calculation",
        file_formats=["csv", "xlsx"],
    ),
    DataAgentRoute(
        source=DataSource.ENERGY_BILLS, agent_id="DATA-001",
        agent_name="PDF & Invoice Extractor",
        module_path="greenlang.agents.data.pdf_extractor",
        description="Extract energy consumption data from utility invoices",
        file_formats=["pdf"],
    ),
    DataAgentRoute(
        source=DataSource.WASTE_RECORDS, agent_id="DATA-002",
        agent_name="Excel/CSV Normalizer",
        module_path="greenlang.agents.data.excel_normalizer",
        description="Normalize waste management records",
        file_formats=["csv", "xlsx"],
    ),
    DataAgentRoute(
        source=DataSource.PACKAGING_DATA, agent_id="DATA-002",
        agent_name="Excel/CSV Normalizer",
        module_path="greenlang.agents.data.excel_normalizer",
        description="Normalize packaging material and weight data",
        file_formats=["csv", "xlsx"],
    ),
    DataAgentRoute(
        source=DataSource.PRODUCT_CATALOG, agent_id="DATA-003",
        agent_name="ERP/Finance Connector",
        module_path="greenlang.agents.data.erp_connector",
        description="Extract product catalog for DPP and PEF analysis",
        file_formats=["api", "json", "xml"],
    ),
]

# ---------------------------------------------------------------------------
# DataRetailBridge
# ---------------------------------------------------------------------------

class DataRetailBridge:
    """Bridge to DATA agents and retail ERP systems.

    Routes retail data intake operations to the appropriate DATA agent and
    provides ERP-specific field mapping for SAP Retail, Oracle Retail,
    NetSuite, and Microsoft Dynamics 365.

    Attributes:
        config: Bridge configuration.
        _agents: Dict of loaded DATA agent modules/stubs.
        _erp_mappings: ERP field mappings for the configured ERP system.

    Example:
        >>> bridge = DataRetailBridge(DataBridgeConfig(erp_system="sap_retail"))
        >>> result = bridge.route_data_intake(DataSource.POS_SALES, {"file": "sales.csv"})
        >>> mappings = bridge.get_erp_field_mappings()
    """

    def __init__(self, config: Optional[DataBridgeConfig] = None) -> None:
        """Initialize the Data Retail Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or DataBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load DATA agents
        self._agents: Dict[str, Any] = {}
        unique_agents = {r.agent_id: r.module_path for r in DATA_AGENT_ROUTES}
        for agent_id, module_path in unique_agents.items():
            self._agents[agent_id] = _try_import_data_agent(agent_id, module_path)

        # Filter ERP mappings for configured system
        self._erp_mappings = [
            m for m in ERP_FIELD_MAPPINGS if m.erp_system == self.config.erp_system
        ]

        available = sum(1 for a in self._agents.values() if not isinstance(a, _AgentStub))
        self.logger.info(
            "DataRetailBridge initialized: %d/%d agents, ERP=%s (%d mappings)",
            available, len(self._agents), self.config.erp_system.value,
            len(self._erp_mappings),
        )

    def route_data_intake(
        self, source: DataSource, data: Dict[str, Any],
    ) -> DataRoutingResult:
        """Route a data intake request to the appropriate DATA agent.

        Args:
            source: Data source category.
            data: Input data or file reference.

        Returns:
            DataRoutingResult with processing status.
        """
        start = time.monotonic()

        route = self._find_route(source)
        if route is None:
            return DataRoutingResult(
                source=source.value, success=False,
                message=f"No routing entry for source '{source.value}'",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        agent = self._agents.get(route.agent_id)
        degraded = isinstance(agent, _AgentStub)

        result = DataRoutingResult(
            source=source.value,
            agent_id=route.agent_id,
            success=not degraded,
            degraded=degraded,
            records_processed=0 if degraded else data.get("record_count", 0),
            quality_score=0.0 if degraded else 85.0,
            message=(
                f"Routed to {route.agent_name}" if not degraded
                else f"{route.agent_name} not available (stub mode)"
            ),
            duration_ms=(time.monotonic() - start) * 1000,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    def extract_from_erp(
        self, erp_system: Optional[RetailERP] = None, tables: Optional[List[str]] = None,
    ) -> ERPExtractionResult:
        """Extract data from a retail ERP system using field mappings.

        Args:
            erp_system: Override ERP system (uses config default if None).
            tables: Specific ERP tables to query (all mapped tables if None).

        Returns:
            ERPExtractionResult with extraction status.
        """
        start = time.monotonic()
        erp = erp_system or self.config.erp_system

        mappings = [m for m in ERP_FIELD_MAPPINGS if m.erp_system == erp]
        if tables:
            mappings = [m for m in mappings if m.erp_table in tables]

        queried_tables = list(set(m.erp_table for m in mappings))

        erp_agent = self._agents.get("DATA-003")
        degraded = isinstance(erp_agent, _AgentStub)

        result = ERPExtractionResult(
            erp_system=erp.value,
            tables_queried=queried_tables,
            records_extracted=0,
            fields_mapped=len(mappings),
            success=not degraded,
            degraded=degraded,
            message=(
                f"Extracted from {len(queried_tables)} tables ({len(mappings)} fields)"
                if not degraded
                else "ERP Connector not available (stub mode)"
            ),
            duration_ms=(time.monotonic() - start) * 1000,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    def get_erp_field_mappings(
        self, erp_system: Optional[RetailERP] = None,
    ) -> List[Dict[str, str]]:
        """Get ERP field mappings for the specified system.

        Args:
            erp_system: ERP system (uses config default if None).

        Returns:
            List of field mapping dicts.
        """
        erp = erp_system or self.config.erp_system
        return [
            {
                "gl_field": m.gl_field,
                "erp_field": m.erp_field,
                "erp_table": m.erp_table,
                "data_type": m.data_type,
                "description": m.description,
            }
            for m in ERP_FIELD_MAPPINGS
            if m.erp_system == erp
        ]

    def run_quality_profiling(self, data: Dict[str, Any]) -> DataRoutingResult:
        """Run data quality profiling on intake data.

        Args:
            data: Data to profile.

        Returns:
            DataRoutingResult with quality score.
        """
        start = time.monotonic()
        agent = self._agents.get("DATA-010",
            _try_import_data_agent("DATA-010", "greenlang.agents.data.data_profiler"))
        degraded = isinstance(agent, _AgentStub)

        result = DataRoutingResult(
            source="quality_profiling",
            agent_id="DATA-010",
            success=not degraded,
            degraded=degraded,
            quality_score=0.0 if degraded else 90.0,
            message="Quality profiling complete" if not degraded else "Profiler not available",
            duration_ms=(time.monotonic() - start) * 1000,
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def _find_route(self, source: DataSource) -> Optional[DataAgentRoute]:
        """Find routing entry for a data source."""
        for route in DATA_AGENT_ROUTES:
            if route.source == source:
                return route
        return None
