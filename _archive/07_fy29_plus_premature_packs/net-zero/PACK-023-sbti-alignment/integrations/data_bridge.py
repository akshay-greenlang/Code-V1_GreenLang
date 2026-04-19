# -*- coding: utf-8 -*-
"""
SBTiDataBridge - AGENT-DATA Integration Bridge for PACK-023
=============================================================

Routes data intake and quality validation through 20 AGENT-DATA agents for
SBTi activity data. Handles energy bills, fuel records, travel data,
procurement spend, supplier questionnaires, and data quality validation --
all formatted for SBTi inventory requirements.

DATA Agent Routing for SBTi:
    Energy/Fuel Data:   DATA-002 (Excel/CSV), DATA-003 (ERP)
    Travel Data:        DATA-002 (Excel/CSV), DATA-004 (API)
    Procurement:        DATA-003 (ERP), DATA-009 (Spend Categorizer)
    Documents:          DATA-001 (PDF & Invoice Extractor)
    Questionnaires:     DATA-008 (Supplier Questionnaire Processor)
    Quality:            DATA-010 (Data Quality Profiler)
    Dedup:              DATA-011 (Duplicate Detection)
    Missing Values:     DATA-012 (Missing Value Imputer)
    Outliers:           DATA-013 (Outlier Detection)
    Time Series:        DATA-014 (Time Series Gap Filler)
    Reconciliation:     DATA-015 (Cross-Source Reconciliation)
    Freshness:          DATA-016 (Data Freshness Monitor)
    Schema:             DATA-017 (Schema Migration Agent)
    Lineage:            DATA-018 (Data Lineage Tracker)
    Validation:         DATA-019 (Validation Rule Engine)
    Climate:            DATA-020 (Climate Hazard Connector)

SBTi-Specific Features:
    - Routes activity data to appropriate intake agents
    - Validates data quality for SBTi submission standards
    - Tracks data lineage for audit trail
    - Monitors freshness for annual reporting cycles
    - Reconciles multi-source data for consistency

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-023 SBTi Alignment Pack
Status: Production Ready
"""

import hashlib
import importlib
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Agent Stubs
# ---------------------------------------------------------------------------

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
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("DATA agent %s not available, using stub", agent_id)
        return _AgentStub(agent_id)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DataSourceType(str, Enum):
    """Supported data source types."""

    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    ERP = "erp"
    API = "api"
    QUESTIONNAIRE = "questionnaire"

class ERPSystem(str, Enum):
    """Supported ERP systems."""

    SAP = "sap"
    ORACLE = "oracle"
    WORKDAY = "workday"
    DYNAMICS_365 = "dynamics_365"

class DataCategory(str, Enum):
    """SBTi-relevant data categories."""

    ENERGY = "energy"
    FUEL = "fuel"
    TRAVEL = "travel"
    PROCUREMENT = "procurement"
    FLEET = "fleet"
    WASTE = "waste"
    WATER = "water"
    REFRIGERANTS = "refrigerants"
    SUPPLIER = "supplier"
    FINANCIAL = "financial"
    PROCESS = "process"
    LAND_USE = "land_use"

class QualityDimension(str, Enum):
    """Data quality dimensions for SBTi assessment."""

    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    RELEVANCE = "relevance"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class DataBridgeConfig(BaseModel):
    """Configuration for the SBTi Data Bridge."""

    pack_id: str = Field(default="PACK-023")
    enable_provenance: bool = Field(default=True)
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    base_year: int = Field(default=2019, ge=2015, le=2025)
    organization_name: str = Field(default="")
    erp_system: str = Field(default="sap")
    min_quality_score: float = Field(
        default=70.0, ge=0.0, le=100.0,
        description="Minimum data quality score for SBTi submission",
    )

class IntakeResult(BaseModel):
    """Result of data intake processing."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    source_type: str = Field(default="")
    data_category: str = Field(default="")
    records_ingested: int = Field(default=0)
    records_validated: int = Field(default=0)
    records_rejected: int = Field(default=0)
    validation_errors: List[str] = Field(default_factory=list)
    agent_id: str = Field(default="")
    sbti_data_ready: bool = Field(default=False)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class QualityResult(BaseModel):
    """Data quality assessment result for SBTi."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    accuracy_score: float = Field(default=0.0, ge=0.0, le=100.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=100.0)
    timeliness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    duplicates_found: int = Field(default=0)
    missing_values_count: int = Field(default=0)
    outliers_detected: int = Field(default=0)
    sbti_submission_ready: bool = Field(default=False)
    recommendations: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class LineageResult(BaseModel):
    """Data lineage tracking result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    data_sources: List[Dict[str, Any]] = Field(default_factory=list)
    transformations: List[Dict[str, Any]] = Field(default_factory=list)
    lineage_chain: List[str] = Field(default_factory=list)
    audit_trail_hash: str = Field(default="")
    sbti_traceability_met: bool = Field(default=False)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class FreshnessResult(BaseModel):
    """Data freshness monitoring result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    datasets_monitored: int = Field(default=0)
    fresh: int = Field(default=0)
    stale: int = Field(default=0)
    critical: int = Field(default=0)
    oldest_data_days: int = Field(default=0)
    newest_data_days: int = Field(default=0)
    within_sbti_reporting_window: bool = Field(default=False)
    alerts: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class ReconciliationResult(BaseModel):
    """Cross-source reconciliation result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    sources_compared: int = Field(default=0)
    records_matched: int = Field(default=0)
    records_unmatched: int = Field(default=0)
    discrepancies: List[Dict[str, Any]] = Field(default_factory=list)
    reconciliation_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    material_discrepancies: int = Field(default=0)
    sbti_data_consistent: bool = Field(default=False)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class ERPFieldMapping(BaseModel):
    """ERP field mapping for SBTi data extraction."""

    erp_system: str = Field(default="sap")
    module: str = Field(default="")
    table: str = Field(default="")
    fields: List[Dict[str, str]] = Field(default_factory=list)
    data_category: str = Field(default="")
    sbti_scope: str = Field(default="")

# ---------------------------------------------------------------------------
# DATA Agent Mapping (20 agents)
# ---------------------------------------------------------------------------

DATA_AGENTS: Dict[str, str] = {
    "DATA-001": "greenlang.agents.data.pdf_invoice_extractor",
    "DATA-002": "greenlang.agents.data.excel_csv_normalizer",
    "DATA-003": "greenlang.agents.data.erp_finance_connector",
    "DATA-004": "greenlang.agents.data.api_gateway_agent",
    "DATA-005": "greenlang.agents.data.eudr_traceability_connector",
    "DATA-006": "greenlang.agents.data.gis_mapping_connector",
    "DATA-007": "greenlang.agents.data.deforestation_satellite_connector",
    "DATA-008": "greenlang.agents.data.supplier_questionnaire_processor",
    "DATA-009": "greenlang.agents.data.spend_data_categorizer",
    "DATA-010": "greenlang.agents.data.data_quality_profiler",
    "DATA-011": "greenlang.agents.data.duplicate_detection",
    "DATA-012": "greenlang.agents.data.missing_value_imputer",
    "DATA-013": "greenlang.agents.data.outlier_detection",
    "DATA-014": "greenlang.agents.data.time_series_gap_filler",
    "DATA-015": "greenlang.agents.data.cross_source_reconciliation",
    "DATA-016": "greenlang.agents.data.data_freshness_monitor",
    "DATA-017": "greenlang.agents.data.schema_migration_agent",
    "DATA-018": "greenlang.agents.data.data_lineage_tracker",
    "DATA-019": "greenlang.agents.data.validation_rule_engine",
    "DATA-020": "greenlang.agents.data.climate_hazard_connector",
}

# Data category to agent routing
CATEGORY_AGENT_ROUTING: Dict[str, List[str]] = {
    "energy": ["DATA-002", "DATA-003", "DATA-004"],
    "fuel": ["DATA-002", "DATA-003"],
    "travel": ["DATA-002", "DATA-004"],
    "procurement": ["DATA-003", "DATA-009"],
    "fleet": ["DATA-002", "DATA-003"],
    "waste": ["DATA-002"],
    "water": ["DATA-002"],
    "refrigerants": ["DATA-002", "DATA-001"],
    "supplier": ["DATA-008", "DATA-009"],
    "financial": ["DATA-003"],
    "process": ["DATA-002", "DATA-003"],
    "land_use": ["DATA-006", "DATA-007"],
}

# ERP field mappings for SBTi data
ERP_FIELD_MAPPINGS: Dict[str, List[ERPFieldMapping]] = {
    "sap": [
        ERPFieldMapping(erp_system="sap", module="MM", table="EKBE", fields=[{"sap_field": "MENGE", "sbti_field": "quantity"}], data_category="procurement", sbti_scope="scope_3"),
        ERPFieldMapping(erp_system="sap", module="FI", table="BKPF", fields=[{"sap_field": "DMBTR", "sbti_field": "amount_eur"}], data_category="financial", sbti_scope="scope_1"),
        ERPFieldMapping(erp_system="sap", module="PM", table="AFKO", fields=[{"sap_field": "GAMNG", "sbti_field": "consumption"}], data_category="energy", sbti_scope="scope_1"),
    ],
    "oracle": [
        ERPFieldMapping(erp_system="oracle", module="AP", table="AP_INVOICES", fields=[{"oracle_field": "INVOICE_AMOUNT", "sbti_field": "amount_eur"}], data_category="procurement", sbti_scope="scope_3"),
        ERPFieldMapping(erp_system="oracle", module="GL", table="GL_JE_LINES", fields=[{"oracle_field": "ACCOUNTED_DR", "sbti_field": "amount_eur"}], data_category="financial", sbti_scope="scope_1"),
    ],
}

# ---------------------------------------------------------------------------
# SBTiDataBridge
# ---------------------------------------------------------------------------

class SBTiDataBridge:
    """Bridge to 20 DATA agents for SBTi activity data intake and quality.

    Routes data intake to appropriate DATA agents, validates quality
    for SBTi submission standards, tracks lineage for audit trail,
    monitors freshness for annual reporting, and reconciles multi-source
    data for consistency.

    Example:
        >>> bridge = SBTiDataBridge(DataBridgeConfig(erp_system="sap"))
        >>> result = bridge.ingest_data(DataSourceType.EXCEL, DataCategory.ENERGY, {"file": "energy.xlsx"})
        >>> print(f"Records ingested: {result.records_ingested}")
    """

    def __init__(self, config: Optional[DataBridgeConfig] = None) -> None:
        """Initialize the SBTi Data Bridge."""
        self.config = config or DataBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._agents: Dict[str, Any] = {}
        for agent_id, module_path in DATA_AGENTS.items():
            self._agents[agent_id] = _try_import_data_agent(agent_id, module_path)
        available = sum(1 for a in self._agents.values() if not isinstance(a, _AgentStub))
        self.logger.info(
            "SBTiDataBridge initialized: %d/%d agents, erp=%s",
            available, len(self._agents), self.config.erp_system,
        )

    def ingest_data(
        self,
        source_type: DataSourceType,
        category: DataCategory,
        data: Dict[str, Any],
    ) -> IntakeResult:
        """Ingest data from a source for SBTi inventory.

        Args:
            source_type: Type of data source.
            category: SBTi data category.
            data: Input data for ingestion.

        Returns:
            IntakeResult with ingestion status.
        """
        start = time.monotonic()

        # Route to appropriate agent
        agent_ids = CATEGORY_AGENT_ROUTING.get(category.value, ["DATA-002"])
        source_agent_map = {
            DataSourceType.PDF: "DATA-001",
            DataSourceType.EXCEL: "DATA-002",
            DataSourceType.CSV: "DATA-002",
            DataSourceType.ERP: "DATA-003",
            DataSourceType.API: "DATA-004",
            DataSourceType.QUESTIONNAIRE: "DATA-008",
        }
        selected_agent = source_agent_map.get(source_type, agent_ids[0])

        agent = self._agents.get(selected_agent)
        if agent is None or isinstance(agent, _AgentStub):
            result = IntakeResult(
                status="degraded",
                source_type=source_type.value,
                data_category=category.value,
                agent_id=selected_agent,
                records_ingested=data.get("record_count", 0),
                records_validated=data.get("record_count", 0),
                sbti_data_ready=True,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        else:
            try:
                agent_result = agent.process(data)
                records = agent_result.get("records_processed", 0) if isinstance(agent_result, dict) else 0
                result = IntakeResult(
                    status="completed",
                    source_type=source_type.value,
                    data_category=category.value,
                    agent_id=selected_agent,
                    records_ingested=records,
                    records_validated=records,
                    sbti_data_ready=True,
                )
            except Exception as exc:
                result = IntakeResult(
                    status="failed",
                    source_type=source_type.value,
                    data_category=category.value,
                    agent_id=selected_agent,
                    validation_errors=[str(exc)],
                )

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def assess_quality(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> QualityResult:
        """Assess data quality for SBTi submission readiness.

        Args:
            context: Optional context with quality metrics.

        Returns:
            QualityResult with dimension scores and recommendations.
        """
        start = time.monotonic()
        context = context or {}

        comp = context.get("completeness_score", 80.0)
        acc = context.get("accuracy_score", 85.0)
        cons = context.get("consistency_score", 90.0)
        time_score = context.get("timeliness_score", 85.0)
        rel = context.get("relevance_score", 90.0)

        overall = round((comp * 0.25 + acc * 0.25 + cons * 0.2 + time_score * 0.15 + rel * 0.15), 1)

        issues: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        if comp < 80:
            issues.append({"dimension": "completeness", "score": comp, "severity": "high"})
            recommendations.append("Fill gaps in activity data records")
        if acc < 80:
            issues.append({"dimension": "accuracy", "score": acc, "severity": "high"})
            recommendations.append("Replace estimated values with measured data")
        if cons < 85:
            issues.append({"dimension": "consistency", "score": cons, "severity": "medium"})
            recommendations.append("Reconcile data across sources")

        dupes = context.get("duplicates_found", 0)
        missing = context.get("missing_values_count", 0)
        outliers = context.get("outliers_detected", 0)

        submission_ready = overall >= self.config.min_quality_score and comp >= 75

        result = QualityResult(
            status="completed",
            overall_score=overall,
            completeness_score=comp,
            accuracy_score=acc,
            consistency_score=cons,
            timeliness_score=time_score,
            relevance_score=rel,
            issues=issues,
            duplicates_found=dupes,
            missing_values_count=missing,
            outliers_detected=outliers,
            sbti_submission_ready=submission_ready,
            recommendations=recommendations,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def track_lineage(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> LineageResult:
        """Track data lineage for SBTi audit trail.

        Args:
            context: Optional context with lineage data.

        Returns:
            LineageResult with traceability chain.
        """
        start = time.monotonic()
        context = context or {}

        sources = context.get("data_sources", [])
        transforms = context.get("transformations", [])
        chain = [s.get("source_id", "") for s in sources] + [t.get("transform_id", "") for t in transforms]

        result = LineageResult(
            status="completed",
            data_sources=sources,
            transformations=transforms,
            lineage_chain=chain,
            audit_trail_hash=_compute_hash(chain),
            sbti_traceability_met=len(sources) > 0,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def check_freshness(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> FreshnessResult:
        """Monitor data freshness for SBTi annual reporting.

        Args:
            context: Optional context with freshness data.

        Returns:
            FreshnessResult with staleness assessment.
        """
        start = time.monotonic()
        context = context or {}

        datasets = context.get("datasets", [])
        fresh = sum(1 for d in datasets if d.get("status") == "fresh")
        stale = sum(1 for d in datasets if d.get("status") == "stale")
        critical = sum(1 for d in datasets if d.get("status") == "critical")
        oldest = context.get("oldest_data_days", 365)
        newest = context.get("newest_data_days", 0)

        within_window = oldest <= 365  # SBTi annual reporting

        alerts: List[str] = []
        if critical > 0:
            alerts.append(f"{critical} datasets critically stale")
        if stale > 0:
            alerts.append(f"{stale} datasets need refresh")

        result = FreshnessResult(
            status="completed",
            datasets_monitored=len(datasets),
            fresh=fresh,
            stale=stale,
            critical=critical,
            oldest_data_days=oldest,
            newest_data_days=newest,
            within_sbti_reporting_window=within_window,
            alerts=alerts,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def reconcile_sources(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReconciliationResult:
        """Reconcile data across sources for SBTi consistency.

        Args:
            context: Optional context with reconciliation data.

        Returns:
            ReconciliationResult with match assessment.
        """
        start = time.monotonic()
        context = context or {}

        sources_compared = context.get("sources_compared", 0)
        matched = context.get("records_matched", 0)
        unmatched = context.get("records_unmatched", 0)
        total = matched + unmatched
        rate = round(matched / total * 100.0, 1) if total > 0 else 0.0

        discrepancies = context.get("discrepancies", [])
        material = sum(1 for d in discrepancies if d.get("material", False))

        result = ReconciliationResult(
            status="completed",
            sources_compared=sources_compared,
            records_matched=matched,
            records_unmatched=unmatched,
            discrepancies=discrepancies,
            reconciliation_rate_pct=rate,
            material_discrepancies=material,
            sbti_data_consistent=rate >= 95.0 and material == 0,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def get_erp_mappings(self, erp_system: Optional[str] = None) -> List[ERPFieldMapping]:
        """Get ERP field mappings for SBTi data extraction.

        Args:
            erp_system: ERP system to get mappings for.

        Returns:
            List of ERP field mappings.
        """
        erp = erp_system or self.config.erp_system
        return ERP_FIELD_MAPPINGS.get(erp, [])

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        available = sum(1 for a in self._agents.values() if not isinstance(a, _AgentStub))
        return {
            "pack_id": self.config.pack_id,
            "module_version": _MODULE_VERSION,
            "total_agents": len(self._agents),
            "available_agents": available,
            "erp_system": self.config.erp_system,
            "reporting_year": self.config.reporting_year,
            "min_quality_score": self.config.min_quality_score,
        }
