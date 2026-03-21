# -*- coding: utf-8 -*-
"""
DataBuildingBridge - Bridge to DATA Agents for Building Energy Data Intake
===========================================================================

This module routes building energy assessment data intake and quality operations
to the appropriate DATA agents. It handles utility bill spreadsheets, BMS exports,
ERP energy procurement records, time series gap filling, data quality profiling,
and validation rule enforcement for building-specific data formats.

Data Agent Routing:
    Utility bill CSV/Excel    --> DATA-002 (Excel/CSV Normalizer)
    ERP energy procurement    --> DATA-003 (ERP/Finance Connector)
    Building data quality     --> DATA-010 (Data Quality Profiler)
    BMS time series gaps      --> DATA-014 (Time Series Gap Filler)
    Energy data validation    --> DATA-019 (Validation Rule Engine)

Building-Specific Formats:
    - Utility bill CSV (electricity, gas, water, oil, district heating)
    - BMS data exports (BACnet point logs, Modbus registers, OPC-UA nodes)
    - DXF/IFC building geometry (floor area, volumes, envelope areas)
    - TM54 operational energy data format
    - DEC (Display Energy Certificate) historical data

ERP Field Mapping (3 systems):
    SAP RE-FX:  property_id->IMKEY, cost_center->KOSTL, energy_kwh->MENGE
    Oracle PMS: building_id->BUILDING_ID, meter->METER_ID, reading->VALUE
    Dynamics:   facility_id->FACILITY_NO, consumption->CONSUMPTION_QTY

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-032 Building Energy Assessment
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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class BuildingERP(str, Enum):
    """Supported ERP systems for building energy data."""

    SAP_RE_FX = "sap_re_fx"
    ORACLE_PMS = "oracle_pms"
    DYNAMICS_FACILITIES = "dynamics_facilities"
    YARDI = "yardi"
    MRI_SOFTWARE = "mri_software"


class BuildingDataSource(str, Enum):
    """Types of building energy data sources."""

    UTILITY_BILL_CSV = "utility_bill_csv"
    UTILITY_BILL_EXCEL = "utility_bill_excel"
    BMS_EXPORT_CSV = "bms_export_csv"
    BMS_EXPORT_JSON = "bms_export_json"
    DXF_GEOMETRY = "dxf_geometry"
    IFC_MODEL = "ifc_model"
    GBXML_MODEL = "gbxml_model"
    EPC_REGISTER = "epc_register"
    DEC_HISTORICAL = "dec_historical"
    TM54_DATA = "tm54_data"
    METER_AMR_DATA = "meter_amr_data"
    WEATHER_TMY = "weather_tmy"
    OCCUPANCY_SCHEDULE = "occupancy_schedule"


class DataAgentRoute(str, Enum):
    """DATA agent identifiers for routing."""

    EXCEL_CSV = "DATA-002"
    ERP_FINANCE = "DATA-003"
    QUALITY_PROFILER = "DATA-010"
    TIME_SERIES_GAP = "DATA-014"
    VALIDATION_RULES = "DATA-019"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class ERPFieldMapping(BaseModel):
    """Field mapping for ERP-to-building data extraction."""

    erp_system: BuildingERP = Field(...)
    property_id_field: str = Field(default="")
    building_id_field: str = Field(default="")
    meter_id_field: str = Field(default="")
    reading_value_field: str = Field(default="")
    reading_date_field: str = Field(default="")
    cost_field: str = Field(default="")
    cost_center_field: str = Field(default="")
    energy_carrier_field: str = Field(default="")
    unit_field: str = Field(default="")


class DataRoutingResult(BaseModel):
    """Result of routing a data operation to a DATA agent."""

    routing_id: str = Field(default_factory=_new_uuid)
    data_source: str = Field(default="")
    agent_id: str = Field(default="")
    agent_name: str = Field(default="")
    success: bool = Field(default=False)
    degraded: bool = Field(default=False)
    records_processed: int = Field(default=0)
    records_valid: int = Field(default=0)
    records_invalid: int = Field(default=0)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ERPExtractionResult(BaseModel):
    """Result of extracting building energy data from ERP."""

    extraction_id: str = Field(default_factory=_new_uuid)
    erp_system: str = Field(default="")
    building_id: str = Field(default="")
    records_extracted: int = Field(default=0)
    date_range_start: str = Field(default="")
    date_range_end: str = Field(default="")
    energy_carriers_found: List[str] = Field(default_factory=list)
    total_energy_kwh: float = Field(default=0.0)
    total_cost: float = Field(default=0.0)
    currency: str = Field(default="GBP")
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


class UtilityBillSchema(BaseModel):
    """Expected schema for utility bill data imports."""

    date_column: str = Field(default="date", description="Invoice/period date column")
    end_date_column: str = Field(default="end_date", description="Period end date")
    meter_id_column: str = Field(default="meter_id")
    energy_carrier_column: str = Field(default="energy_carrier")
    consumption_column: str = Field(default="consumption_kwh")
    unit_column: str = Field(default="unit")
    cost_column: str = Field(default="cost")
    currency_column: str = Field(default="currency")
    supplier_column: str = Field(default="supplier")
    tariff_column: str = Field(default="tariff")


class BuildingDataQualityReport(BaseModel):
    """Data quality assessment report for building energy data."""

    report_id: str = Field(default_factory=_new_uuid)
    building_id: str = Field(default="")
    total_records: int = Field(default=0)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    accuracy_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    consistency_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    timeliness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    overall_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    gaps_detected: int = Field(default=0)
    outliers_detected: int = Field(default=0)
    duplicates_detected: int = Field(default=0)
    validation_errors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class DataBuildingBridgeConfig(BaseModel):
    """Configuration for the Data Building Bridge."""

    pack_id: str = Field(default="PACK-032")
    enable_provenance: bool = Field(default=True)
    building_type: str = Field(default="commercial_office")
    default_unit: str = Field(default="kWh")
    default_currency: str = Field(default="GBP")
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    quality_threshold: float = Field(default=70.0, ge=0.0, le=100.0)
    gap_fill_method: str = Field(default="linear_interpolation")
    outlier_detection_method: str = Field(default="iqr")
    outlier_threshold_sigma: float = Field(default=3.0, ge=1.0, le=5.0)


# ---------------------------------------------------------------------------
# ERP Field Mapping Presets
# ---------------------------------------------------------------------------

ERP_FIELD_MAPPINGS: Dict[str, ERPFieldMapping] = {
    "sap_re_fx": ERPFieldMapping(
        erp_system=BuildingERP.SAP_RE_FX,
        property_id_field="IMKEY",
        building_id_field="BUILDING_NO",
        meter_id_field="METER_NO",
        reading_value_field="MENGE",
        reading_date_field="ABLDT",
        cost_field="WRBTR",
        cost_center_field="KOSTL",
        energy_carrier_field="ENTYPE",
        unit_field="MEINS",
    ),
    "oracle_pms": ERPFieldMapping(
        erp_system=BuildingERP.ORACLE_PMS,
        property_id_field="PROPERTY_ID",
        building_id_field="BUILDING_ID",
        meter_id_field="METER_ID",
        reading_value_field="METER_VALUE",
        reading_date_field="READING_DATE",
        cost_field="AMOUNT",
        cost_center_field="COST_CENTER",
        energy_carrier_field="UTILITY_TYPE",
        unit_field="UOM",
    ),
    "dynamics_facilities": ERPFieldMapping(
        erp_system=BuildingERP.DYNAMICS_FACILITIES,
        property_id_field="PROPERTY_NO",
        building_id_field="FACILITY_NO",
        meter_id_field="RESOURCE_NO",
        reading_value_field="CONSUMPTION_QTY",
        reading_date_field="POSTING_DATE",
        cost_field="AMOUNT_LCY",
        cost_center_field="DIMENSION_1",
        energy_carrier_field="RESOURCE_TYPE",
        unit_field="UNIT_OF_MEASURE",
    ),
    "yardi": ERPFieldMapping(
        erp_system=BuildingERP.YARDI,
        property_id_field="hPropCode",
        building_id_field="hBldgCode",
        meter_id_field="hMeterCode",
        reading_value_field="dConsumption",
        reading_date_field="dtReadDate",
        cost_field="dCharge",
        cost_center_field="hAcctCode",
        energy_carrier_field="hUtilType",
        unit_field="hUOM",
    ),
    "mri_software": ERPFieldMapping(
        erp_system=BuildingERP.MRI_SOFTWARE,
        property_id_field="PropertyID",
        building_id_field="BuildingID",
        meter_id_field="MeterID",
        reading_value_field="ReadingValue",
        reading_date_field="ReadingDate",
        cost_field="InvoiceAmount",
        cost_center_field="CostCode",
        energy_carrier_field="ServiceType",
        unit_field="UnitOfMeasure",
    ),
}

# Data source to agent routing
DATA_SOURCE_ROUTING: Dict[str, DataAgentRoute] = {
    BuildingDataSource.UTILITY_BILL_CSV.value: DataAgentRoute.EXCEL_CSV,
    BuildingDataSource.UTILITY_BILL_EXCEL.value: DataAgentRoute.EXCEL_CSV,
    BuildingDataSource.BMS_EXPORT_CSV.value: DataAgentRoute.EXCEL_CSV,
    BuildingDataSource.BMS_EXPORT_JSON.value: DataAgentRoute.EXCEL_CSV,
    BuildingDataSource.METER_AMR_DATA.value: DataAgentRoute.TIME_SERIES_GAP,
    BuildingDataSource.DEC_HISTORICAL.value: DataAgentRoute.EXCEL_CSV,
    BuildingDataSource.TM54_DATA.value: DataAgentRoute.VALIDATION_RULES,
    BuildingDataSource.OCCUPANCY_SCHEDULE.value: DataAgentRoute.VALIDATION_RULES,
    BuildingDataSource.WEATHER_TMY.value: DataAgentRoute.TIME_SERIES_GAP,
}

# Unit conversion factors to kWh
UNIT_CONVERSION_TO_KWH: Dict[str, float] = {
    "kWh": 1.0,
    "MWh": 1000.0,
    "GJ": 277.778,
    "MJ": 0.277778,
    "therms": 29.3071,
    "m3_gas": 10.55,  # Natural gas at standard conditions
    "litres_oil": 10.35,  # Heating oil
    "litres_lpg": 7.11,  # LPG
    "kg_pellets": 4.8,  # Wood pellets
    "kW": 0.0,  # Power, not energy
    "tonnes_steam": 694.44,
    "tonnes_co2": 0.0,  # Not energy
}


# ---------------------------------------------------------------------------
# DataBuildingBridge
# ---------------------------------------------------------------------------


class DataBuildingBridge:
    """Routes building energy data operations to DATA agents.

    Handles utility bill imports, BMS data processing, ERP extraction,
    data quality profiling, gap filling, and validation for building
    energy assessment purposes.

    Attributes:
        config: Bridge configuration.
        _agents: Loaded DATA agent instances (or stubs).

    Example:
        >>> bridge = DataBuildingBridge()
        >>> result = bridge.route_data_import("utility_bill_csv", {"file_path": "bills.csv"})
        >>> assert result.success
    """

    def __init__(self, config: Optional[DataBuildingBridgeConfig] = None) -> None:
        """Initialize the Data Building Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or DataBuildingBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._agents: Dict[str, Any] = {}
        self._load_agents()

        self.logger.info(
            "DataBuildingBridge initialized: building_type=%s, "
            "quality_threshold=%.1f",
            self.config.building_type,
            self.config.quality_threshold,
        )

    def _load_agents(self) -> None:
        """Load DATA agents with graceful fallback to stubs."""
        agent_modules = {
            DataAgentRoute.EXCEL_CSV.value: "greenlang.agents.data.excel_csv_normalizer",
            DataAgentRoute.ERP_FINANCE.value: "greenlang.agents.data.erp_finance_connector",
            DataAgentRoute.QUALITY_PROFILER.value: "greenlang.agents.data.data_quality_profiler",
            DataAgentRoute.TIME_SERIES_GAP.value: "greenlang.agents.data.time_series_gap_filler",
            DataAgentRoute.VALIDATION_RULES.value: "greenlang.agents.data.validation_rule_engine",
        }
        for agent_id, module_path in agent_modules.items():
            self._agents[agent_id] = _try_import_data_agent(agent_id, module_path)

    def route_data_import(
        self,
        source_type: str,
        data: Dict[str, Any],
    ) -> DataRoutingResult:
        """Route a data import operation to the appropriate DATA agent.

        Args:
            source_type: Data source type (see BuildingDataSource enum).
            data: Import data including file paths and configuration.

        Returns:
            DataRoutingResult with processing details.
        """
        start_time = time.monotonic()
        result = DataRoutingResult(data_source=source_type)

        agent_route = DATA_SOURCE_ROUTING.get(source_type)
        if agent_route is None:
            result.message = f"No DATA route for source '{source_type}'"
            result.duration_ms = (time.monotonic() - start_time) * 1000
            return result

        result.agent_id = agent_route.value
        result.agent_name = agent_route.name

        agent = self._agents.get(agent_route.value)
        if agent is None:
            result.message = f"Agent {agent_route.value} not loaded"
            result.duration_ms = (time.monotonic() - start_time) * 1000
            return result

        try:
            if isinstance(agent, _AgentStub):
                result.degraded = True
                result.success = True
                result.records_processed = data.get("record_count", 0)
                result.records_valid = result.records_processed
                result.quality_score = 75.0
                result.message = f"Degraded: stub processing for {agent_route.value}"
            else:
                agent_result = agent.process(data)
                result.records_processed = agent_result.get("records_processed", 0)
                result.records_valid = agent_result.get("records_valid", 0)
                result.records_invalid = agent_result.get("records_invalid", 0)
                result.quality_score = agent_result.get("quality_score", 0.0)
                result.success = True
                result.message = f"Processed via {agent_route.value}"

        except Exception as exc:
            result.message = f"Agent {agent_route.value} failed: {exc}"
            self.logger.error(
                "DATA routing failed: source=%s, agent=%s, error=%s",
                source_type, agent_route.value, exc,
            )

        result.duration_ms = (time.monotonic() - start_time) * 1000

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    def extract_erp_data(
        self,
        erp_system: str,
        building_id: str,
        date_range: Optional[Dict[str, str]] = None,
    ) -> ERPExtractionResult:
        """Extract building energy data from ERP system.

        Args:
            erp_system: ERP system identifier (see BuildingERP enum).
            building_id: Building identifier in the ERP system.
            date_range: Optional dict with 'start' and 'end' dates.

        Returns:
            ERPExtractionResult with extracted data summary.
        """
        start_time = time.monotonic()
        result = ERPExtractionResult(
            erp_system=erp_system,
            building_id=building_id,
        )

        mapping = ERP_FIELD_MAPPINGS.get(erp_system)
        if mapping is None:
            self.logger.warning("No ERP mapping for system '%s'", erp_system)
            return result

        agent = self._agents.get(DataAgentRoute.ERP_FINANCE.value)
        if agent is None or isinstance(agent, _AgentStub):
            result.records_extracted = 0
            result.quality_score = 0.0
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result)
            return result

        try:
            extraction_data = {
                "erp_system": erp_system,
                "building_id": building_id,
                "field_mapping": mapping.model_dump(),
                "date_range": date_range or {},
            }
            agent_result = agent.extract(extraction_data)
            result.records_extracted = agent_result.get("records_extracted", 0)
            result.date_range_start = agent_result.get("date_range_start", "")
            result.date_range_end = agent_result.get("date_range_end", "")
            result.energy_carriers_found = agent_result.get("carriers", [])
            result.total_energy_kwh = agent_result.get("total_energy_kwh", 0.0)
            result.total_cost = agent_result.get("total_cost", 0.0)
            result.quality_score = agent_result.get("quality_score", 75.0)

        except Exception as exc:
            self.logger.error(
                "ERP extraction failed: system=%s, building=%s, error=%s",
                erp_system, building_id, exc,
            )

        elapsed = (time.monotonic() - start_time) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "ERP extraction complete: system=%s, building=%s, records=%d, "
            "duration=%.1fms",
            erp_system, building_id, result.records_extracted, elapsed,
        )
        return result

    def profile_data_quality(
        self,
        building_id: str,
        data: Dict[str, Any],
    ) -> BuildingDataQualityReport:
        """Profile the quality of building energy data.

        Args:
            building_id: Building identifier.
            data: Data to profile.

        Returns:
            BuildingDataQualityReport with quality metrics.
        """
        start_time = time.monotonic()
        report = BuildingDataQualityReport(building_id=building_id)

        agent = self._agents.get(DataAgentRoute.QUALITY_PROFILER.value)
        if agent is None or isinstance(agent, _AgentStub):
            # Deterministic default quality assessment
            records = data.get("records", [])
            total = len(records) if isinstance(records, list) else data.get("record_count", 0)
            report.total_records = total
            report.completeness_pct = data.get("completeness", 80.0)
            report.accuracy_pct = data.get("accuracy", 85.0)
            report.consistency_pct = data.get("consistency", 82.0)
            report.timeliness_pct = data.get("timeliness", 90.0)
            report.overall_quality_score = round(
                (report.completeness_pct + report.accuracy_pct
                 + report.consistency_pct + report.timeliness_pct) / 4.0,
                1,
            )
            report.recommendations.append(
                "Quality profiler agent not available; default assessment used"
            )
        else:
            try:
                result = agent.profile(data)
                report.total_records = result.get("total_records", 0)
                report.completeness_pct = result.get("completeness", 0.0)
                report.accuracy_pct = result.get("accuracy", 0.0)
                report.consistency_pct = result.get("consistency", 0.0)
                report.timeliness_pct = result.get("timeliness", 0.0)
                report.overall_quality_score = result.get("overall_score", 0.0)
                report.gaps_detected = result.get("gaps", 0)
                report.outliers_detected = result.get("outliers", 0)
                report.duplicates_detected = result.get("duplicates", 0)
                report.validation_errors = result.get("errors", [])
                report.recommendations = result.get("recommendations", [])
            except Exception as exc:
                self.logger.error(
                    "Data quality profiling failed: building=%s, error=%s",
                    building_id, exc,
                )
                report.recommendations.append(f"Profiling error: {exc}")

        elapsed = (time.monotonic() - start_time) * 1000
        if self.config.enable_provenance:
            report.provenance_hash = _compute_hash(report)

        self.logger.info(
            "Data quality profiling complete: building=%s, score=%.1f, "
            "duration=%.1fms",
            building_id, report.overall_quality_score, elapsed,
        )
        return report

    def convert_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str = "kWh",
    ) -> Dict[str, Any]:
        """Convert energy units to kWh (or between supported units).

        Zero-hallucination: deterministic unit conversion factors.

        Args:
            value: Numeric value to convert.
            from_unit: Source unit.
            to_unit: Target unit (default kWh).

        Returns:
            Dict with converted value and conversion details.
        """
        from_factor = UNIT_CONVERSION_TO_KWH.get(from_unit)
        to_factor = UNIT_CONVERSION_TO_KWH.get(to_unit)

        if from_factor is None:
            return {"error": f"Unknown source unit '{from_unit}'", "converted_value": 0.0}
        if to_factor is None or to_factor == 0:
            return {"error": f"Unknown or invalid target unit '{to_unit}'", "converted_value": 0.0}

        kwh_value = value * from_factor
        converted_value = kwh_value / to_factor

        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": round(converted_value, 6),
            "target_unit": to_unit,
            "intermediate_kwh": round(kwh_value, 6),
            "from_factor_to_kwh": from_factor,
            "to_factor_from_kwh": to_factor,
        }

    def get_utility_bill_schema(self) -> Dict[str, Any]:
        """Return the expected schema for utility bill imports.

        Returns:
            Dict describing column names and expected formats.
        """
        schema = UtilityBillSchema()
        return schema.model_dump()

    def get_erp_field_mapping(self, erp_system: str) -> Optional[Dict[str, Any]]:
        """Get the field mapping for an ERP system.

        Args:
            erp_system: ERP system identifier.

        Returns:
            Field mapping dict or None if not found.
        """
        mapping = ERP_FIELD_MAPPINGS.get(erp_system)
        return mapping.model_dump() if mapping else None

    def get_supported_units(self) -> List[str]:
        """Return list of supported energy units for conversion.

        Returns:
            List of unit identifiers.
        """
        return list(UNIT_CONVERSION_TO_KWH.keys())
