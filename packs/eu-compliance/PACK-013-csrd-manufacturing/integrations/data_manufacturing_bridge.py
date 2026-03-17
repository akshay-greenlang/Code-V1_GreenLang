"""
PACK-013 CSRD Manufacturing Pack - Data Manufacturing Bridge.

Routes data-intake and quality operations to AGENT-DATA agents with
manufacturing-specific transformations.  Supports ERP/MES/SCADA data
sources and maps them through the GreenLang data-quality pipeline.
"""

import hashlib
import importlib
import logging
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class DataBridgeConfig(BaseModel):
    """Configuration for the data manufacturing bridge."""
    enabled_agents: List[str] = Field(
        default_factory=lambda: [
            "DATA-001", "DATA-002", "DATA-003", "DATA-004", "DATA-005",
            "DATA-008", "DATA-009", "DATA-010", "DATA-011", "DATA-012",
            "DATA-013", "DATA-014", "DATA-015", "DATA-018", "DATA-019",
        ]
    )
    erp_system: str = Field(
        default="sap",
        description="ERP system type: sap, oracle, workday, dynamics",
    )
    mes_integration: bool = Field(
        default=False,
        description="Enable MES (Manufacturing Execution System) data",
    )
    scada_enabled: bool = Field(
        default=False,
        description="Enable SCADA/IIoT sensor data ingestion",
    )
    agent_module_prefix: str = Field(
        default="greenlang.agents.data"
    )
    fallback_to_stubs: bool = Field(default=True)
    batch_size: int = Field(default=5000, ge=100)
    encoding: str = Field(default="utf-8")


class DataRouting(BaseModel):
    """Routing entry mapping a data type to a DATA agent."""
    data_type: str
    agent_id: str
    agent_module: str
    transformation: str = Field(
        default="passthrough",
        description="Transformation to apply before routing",
    )
    description: str = Field(default="")


# ---------------------------------------------------------------------------
# Internal stub
# ---------------------------------------------------------------------------

class _DataAgentStub:
    """Stub when a DATA agent cannot be imported."""

    def __init__(self, agent_id: str, reason: str = "not installed") -> None:
        self._agent_id = agent_id
        self._reason = reason

    def __getattr__(self, name: str) -> Callable[..., Dict[str, Any]]:
        def _stub(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "unavailable",
                "agent_id": self._agent_id,
                "method": name,
                "reason": self._reason,
                "record_count": 0,
                "data": {},
            }
        return _stub


# ---------------------------------------------------------------------------
# Default routing table
# ---------------------------------------------------------------------------

DEFAULT_DATA_ROUTES: List[Dict[str, str]] = [
    {
        "data_type": "pdf_invoice",
        "agent_id": "DATA-001",
        "agent_module": "agent_001_pdf_extractor",
        "transformation": "ocr_extract",
        "description": "PDF and invoice extraction",
    },
    {
        "data_type": "excel_csv",
        "agent_id": "DATA-002",
        "agent_module": "agent_002_excel_csv",
        "transformation": "normalise_columns",
        "description": "Excel/CSV normalisation",
    },
    {
        "data_type": "erp_export",
        "agent_id": "DATA-003",
        "agent_module": "agent_003_erp_connector",
        "transformation": "erp_mapping",
        "description": "ERP/Finance data connector",
    },
    {
        "data_type": "api_payload",
        "agent_id": "DATA-004",
        "agent_module": "agent_004_api_gateway",
        "transformation": "passthrough",
        "description": "API gateway data ingestion",
    },
    {
        "data_type": "supplier_questionnaire",
        "agent_id": "DATA-008",
        "agent_module": "agent_008_supplier_questionnaire",
        "transformation": "questionnaire_normalise",
        "description": "Supplier questionnaire processing",
    },
    {
        "data_type": "spend_data",
        "agent_id": "DATA-009",
        "agent_module": "agent_009_spend_categoriser",
        "transformation": "categorise",
        "description": "Spend data categorisation",
    },
    {
        "data_type": "quality_profile",
        "agent_id": "DATA-010",
        "agent_module": "agent_010_data_quality_profiler",
        "transformation": "passthrough",
        "description": "Data quality profiling",
    },
    {
        "data_type": "dedup_check",
        "agent_id": "DATA-011",
        "agent_module": "agent_011_duplicate_detection",
        "transformation": "passthrough",
        "description": "Duplicate detection",
    },
    {
        "data_type": "imputation",
        "agent_id": "DATA-012",
        "agent_module": "agent_012_missing_value_imputer",
        "transformation": "passthrough",
        "description": "Missing value imputation",
    },
    {
        "data_type": "outlier_check",
        "agent_id": "DATA-013",
        "agent_module": "agent_013_outlier_detection",
        "transformation": "passthrough",
        "description": "Outlier detection",
    },
    {
        "data_type": "timeseries_gap",
        "agent_id": "DATA-014",
        "agent_module": "agent_014_timeseries_gap",
        "transformation": "passthrough",
        "description": "Time series gap filling",
    },
    {
        "data_type": "reconciliation",
        "agent_id": "DATA-015",
        "agent_module": "agent_015_cross_source_recon",
        "transformation": "passthrough",
        "description": "Cross-source reconciliation",
    },
    {
        "data_type": "lineage",
        "agent_id": "DATA-018",
        "agent_module": "agent_018_data_lineage",
        "transformation": "passthrough",
        "description": "Data lineage tracking",
    },
    {
        "data_type": "validation",
        "agent_id": "DATA-019",
        "agent_module": "agent_019_validation_rule_engine",
        "transformation": "passthrough",
        "description": "Validation rule engine",
    },
]

# Manufacturing-specific ERP field mappings
ERP_FIELD_MAP: Dict[str, Dict[str, str]] = {
    "sap": {
        "MATNR": "material_number",
        "WERKS": "plant_code",
        "MENGE": "quantity",
        "MEINS": "unit_of_measure",
        "BUKRS": "company_code",
        "BWART": "movement_type",
        "BUDAT": "posting_date",
        "EBELN": "purchase_order",
        "LIFNR": "supplier_id",
        "NETWR": "net_value",
        "WAERS": "currency",
    },
    "oracle": {
        "ITEM_NUMBER": "material_number",
        "ORGANIZATION_CODE": "plant_code",
        "QUANTITY": "quantity",
        "UOM_CODE": "unit_of_measure",
        "LEDGER_ID": "company_code",
        "TRANSACTION_TYPE": "movement_type",
        "TRANSACTION_DATE": "posting_date",
        "PO_NUMBER": "purchase_order",
        "VENDOR_ID": "supplier_id",
        "AMOUNT": "net_value",
        "CURRENCY_CODE": "currency",
    },
    "workday": {
        "Item_ID": "material_number",
        "Location": "plant_code",
        "Qty": "quantity",
        "Unit": "unit_of_measure",
        "Company": "company_code",
        "Type": "movement_type",
        "Date": "posting_date",
        "PO_Ref": "purchase_order",
        "Supplier_Ref": "supplier_id",
        "Amount": "net_value",
        "Currency": "currency",
    },
    "dynamics": {
        "ItemNumber": "material_number",
        "Site": "plant_code",
        "TransactionQuantity": "quantity",
        "InventoryUnitOfMeasure": "unit_of_measure",
        "DataArea": "company_code",
        "TransactionType": "movement_type",
        "TransDate": "posting_date",
        "PurchaseOrderNumber": "purchase_order",
        "VendorAccountNumber": "supplier_id",
        "NetAmount": "net_value",
        "CurrencyCode": "currency",
    },
}


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class DataManufacturingBridge:
    """
    Route manufacturing data intake and quality operations to the
    appropriate AGENT-DATA agents.

    Supports SAP, Oracle, Workday, and Dynamics ERP field mappings,
    as well as MES and SCADA data sources.
    """

    def __init__(
        self, config: Optional[DataBridgeConfig] = None
    ) -> None:
        self.config = config or DataBridgeConfig()
        self._agents: Dict[str, Any] = {}
        self._routing: Dict[str, DataRouting] = {}
        self._build_routing()
        self._load_agents()

    # -- setup ---------------------------------------------------------------

    def _build_routing(self) -> None:
        for entry in DEFAULT_DATA_ROUTES:
            self._routing[entry["data_type"]] = DataRouting(**entry)

    def _load_agents(self) -> None:
        seen: Dict[str, Any] = {}
        for route in self._routing.values():
            if route.agent_id not in self.config.enabled_agents:
                continue
            if route.agent_module in seen:
                self._agents[route.agent_id] = seen[route.agent_module]
                continue
            full = (
                f"{self.config.agent_module_prefix}.{route.agent_module}"
            )
            try:
                mod = importlib.import_module(full)
                self._agents[route.agent_id] = mod
                seen[route.agent_module] = mod
            except ImportError as exc:
                if self.config.fallback_to_stubs:
                    stub = _DataAgentStub(route.agent_id, str(exc))
                    self._agents[route.agent_id] = stub
                    seen[route.agent_module] = stub

    def _get_agent(self, agent_id: str) -> Any:
        return self._agents.get(
            agent_id, _DataAgentStub(agent_id, "not loaded")
        )

    @staticmethod
    def _compute_hash(data: Any) -> str:
        raw = str(data).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    # -- ERP field mapping ---------------------------------------------------

    def _map_erp_fields(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Map ERP-specific field names to canonical GreenLang names."""
        field_map = ERP_FIELD_MAP.get(
            self.config.erp_system, ERP_FIELD_MAP["sap"]
        )
        mapped: List[Dict[str, Any]] = []
        for rec in records:
            new_rec: Dict[str, Any] = {}
            for erp_field, gl_field in field_map.items():
                if erp_field in rec:
                    new_rec[gl_field] = rec[erp_field]
            # Carry over unmapped fields
            for k, v in rec.items():
                if k not in field_map:
                    new_rec[k] = v
            mapped.append(new_rec)
        return mapped

    # -- public API ----------------------------------------------------------

    def ingest_erp_data(
        self, erp_export: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ingest ERP export data via DATA-003.

        The raw ERP field names are mapped to canonical GreenLang names
        before passing to the agent.
        """
        agent = self._get_agent("DATA-003")
        records = erp_export.get("records", [])
        mapped = self._map_erp_fields(records)

        result = agent.ingest({
            "records": mapped,
            "erp_system": self.config.erp_system,
            "batch_size": self.config.batch_size,
        })
        result["field_mapping_applied"] = True
        result["erp_system"] = self.config.erp_system
        result["provenance_hash"] = self._compute_hash(mapped[:10])
        return result

    def ingest_production_data(
        self, excel_file: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ingest production data from Excel/CSV files via DATA-002.

        Handles merged cells, multi-sheet workbooks, and unit conversion.
        """
        agent = self._get_agent("DATA-002")
        result = agent.ingest(excel_file)
        result["source_type"] = "production_data"
        result["provenance_hash"] = self._compute_hash(
            excel_file.get("file_path", "unknown")
        )
        return result

    def ingest_supplier_data(
        self, questionnaire: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ingest supplier questionnaire data via DATA-008.

        Parses structured questionnaire responses and normalises them
        for downstream emission-factor lookups.
        """
        agent = self._get_agent("DATA-008")
        result = agent.process(questionnaire)
        result["source_type"] = "supplier_questionnaire"
        result["provenance_hash"] = self._compute_hash(
            questionnaire.get("supplier_id", "unknown")
        )
        return result

    def validate_data(
        self,
        data: Dict[str, Any],
        rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate data against a rule set via DATA-019.

        Manufacturing-specific rules include unit checks, emission-factor
        bounds, and temporal consistency.
        """
        agent = self._get_agent("DATA-019")

        # Add manufacturing-specific rules
        mfg_rules = {
            "emission_factor_bounds": {
                "min": 0.0,
                "max": 100.0,
                "unit": "tCO2e/unit",
            },
            "energy_consumption_bounds": {
                "min": 0.0,
                "max": 1_000_000.0,
                "unit": "MWh",
            },
            "water_withdrawal_bounds": {
                "min": 0.0,
                "max": 10_000_000.0,
                "unit": "m3",
            },
        }
        combined_rules = {**mfg_rules, **rules}

        result = agent.validate(data, combined_rules)
        result["provenance_hash"] = self._compute_hash(combined_rules)
        return result

    def profile_data_quality(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Profile data quality via DATA-010.

        Returns completeness, consistency, validity, and uniqueness
        scores for the manufacturing dataset.
        """
        agent = self._get_agent("DATA-010")
        result = agent.profile(data)
        result["provenance_hash"] = self._compute_hash(
            str(len(data.get("records", [])))
        )
        return result

    def track_lineage(
        self, data_id: str
    ) -> Dict[str, Any]:
        """
        Track data lineage for a given data asset via DATA-018.

        Returns the full transformation chain from source to destination.
        """
        agent = self._get_agent("DATA-018")
        result = agent.track(data_id)
        result["provenance_hash"] = self._compute_hash(data_id)
        return result

    def detect_duplicates(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect duplicate records via DATA-011."""
        agent = self._get_agent("DATA-011")
        return agent.detect(records)

    def detect_outliers(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect statistical outliers via DATA-013."""
        agent = self._get_agent("DATA-013")
        return agent.detect(records)

    def reconcile_sources(
        self,
        source_a: Dict[str, Any],
        source_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Cross-source reconciliation via DATA-015."""
        agent = self._get_agent("DATA-015")
        return agent.reconcile(source_a, source_b)
