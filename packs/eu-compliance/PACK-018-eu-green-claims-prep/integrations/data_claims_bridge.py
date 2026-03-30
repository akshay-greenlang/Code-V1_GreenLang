# -*- coding: utf-8 -*-
"""
DataClaimsBridge - AGENT-DATA Evidence Gathering Bridge for PACK-018
=======================================================================

This module routes evidence gathering requests to the appropriate DATA
agents (001-020) for green claims substantiation. It maps evidence
requirements to specific data intake and quality agents, ensuring that
all claims are backed by high-quality, validated source data.

DATA Agent Routing:
    Intake Agents (001-007):
        DATA-001: PDF & Invoice Extractor    --> document-based evidence
        DATA-002: Excel/CSV Normalizer       --> spreadsheet evidence
        DATA-003: ERP/Finance Connector      --> financial evidence
        DATA-004: API Gateway Agent          --> external API evidence
        DATA-005: EUDR Traceability Connector --> supply chain evidence
        DATA-006: GIS/Mapping Connector      --> geospatial evidence
        DATA-007: Deforestation Satellite    --> satellite imagery evidence

    Quality Agents (008-019):
        DATA-008: Supplier Questionnaire     --> supplier response data
        DATA-009: Spend Data Categorizer     --> spend categorization
        DATA-010: Data Quality Profiler      --> quality assessment
        DATA-011: Duplicate Detection        --> deduplication
        DATA-012: Missing Value Imputer      --> gap filling
        DATA-013: Outlier Detection          --> anomaly screening
        DATA-014: Time Series Gap Filler     --> temporal completeness
        DATA-015: Cross-Source Reconciliation --> multi-source validation
        DATA-016: Data Freshness Monitor     --> data currency check
        DATA-017: Schema Migration Agent     --> schema compatibility
        DATA-018: Data Lineage Tracker       --> provenance tracking
        DATA-019: Validation Rule Engine     --> rule-based validation

    Geo Agent (020):
        DATA-020: Climate Hazard Connector   --> climate risk evidence

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-018 EU Green Claims Prep Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

__all__ = [
    "EvidenceSourceType",
    "DataQualityLevel",
    "RoutingStatus",
    "DataRoutingConfig",
    "DataRoutingEntry",
    "DataRoutingResult",
    "DataClaimsBridge",
]

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvidenceSourceType(str, Enum):
    """Types of evidence sources for green claims."""

    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    ERP_SYSTEM = "erp_system"
    EXTERNAL_API = "external_api"
    SUPPLY_CHAIN = "supply_chain"
    GEOSPATIAL = "geospatial"
    SATELLITE = "satellite"
    QUESTIONNAIRE = "questionnaire"
    FINANCIAL = "financial"
    CLIMATE_DATA = "climate_data"

class DataQualityLevel(str, Enum):
    """Quality level of evidence data."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class RoutingStatus(str, Enum):
    """Status of a data routing operation."""

    ROUTED = "routed"
    NO_AGENT = "no_agent_found"
    FAILED = "failed"
    QUALITY_CHECK_NEEDED = "quality_check_needed"

# ---------------------------------------------------------------------------
# Agent Routing Tables
# ---------------------------------------------------------------------------

INTAKE_AGENTS: Dict[str, str] = {
    "DATA-001": "pdf_invoice_extractor",
    "DATA-002": "excel_csv_normalizer",
    "DATA-003": "erp_finance_connector",
    "DATA-004": "api_gateway_agent",
    "DATA-005": "eudr_traceability_connector",
    "DATA-006": "gis_mapping_connector",
    "DATA-007": "deforestation_satellite_connector",
}

QUALITY_AGENTS: Dict[str, str] = {
    "DATA-008": "supplier_questionnaire_processor",
    "DATA-009": "spend_data_categorizer",
    "DATA-010": "data_quality_profiler",
    "DATA-011": "duplicate_detection",
    "DATA-012": "missing_value_imputer",
    "DATA-013": "outlier_detection",
    "DATA-014": "time_series_gap_filler",
    "DATA-015": "cross_source_reconciliation",
    "DATA-016": "data_freshness_monitor",
    "DATA-017": "schema_migration",
    "DATA-018": "data_lineage_tracker",
    "DATA-019": "validation_rule_engine",
}

GEO_AGENTS: Dict[str, str] = {
    "DATA-020": "climate_hazard_connector",
}

EVIDENCE_TO_AGENT_MAP: Dict[str, List[str]] = {
    "emission_certificates": ["DATA-001", "DATA-003", "DATA-010", "DATA-019"],
    "lifecycle_assessment": ["DATA-001", "DATA-002", "DATA-010", "DATA-015"],
    "supplier_declarations": ["DATA-008", "DATA-010", "DATA-011", "DATA-019"],
    "energy_purchase_records": ["DATA-003", "DATA-002", "DATA-010"],
    "waste_manifests": ["DATA-001", "DATA-002", "DATA-010"],
    "water_usage_records": ["DATA-002", "DATA-003", "DATA-010"],
    "supply_chain_traceability": ["DATA-005", "DATA-006", "DATA-018"],
    "satellite_verification": ["DATA-007", "DATA-006"],
    "financial_audit_data": ["DATA-003", "DATA-009", "DATA-010"],
    "climate_risk_data": ["DATA-020", "DATA-006"],
    "product_composition": ["DATA-001", "DATA-002", "DATA-008", "DATA-019"],
    "certification_records": ["DATA-001", "DATA-004", "DATA-016"],
    "biodiversity_surveys": ["DATA-006", "DATA-007", "DATA-020"],
    "carbon_offset_records": ["DATA-001", "DATA-003", "DATA-004", "DATA-015"],
    "recycling_reports": ["DATA-001", "DATA-002", "DATA-010"],
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class DataRoutingConfig(BaseModel):
    """Configuration for DATA agent evidence routing."""

    pack_id: str = Field(default="PACK-018")
    enable_intake: bool = Field(default=True)
    enable_quality: bool = Field(default=True)
    enable_geo: bool = Field(default=True)
    enable_provenance: bool = Field(default=True)
    auto_quality_check: bool = Field(
        default=True,
        description="Automatically route through quality agents after intake",
    )
    freshness_threshold_days: int = Field(
        default=90, ge=1, le=365,
        description="Maximum age of evidence data in days",
    )

class DataRoutingEntry(BaseModel):
    """A single agent routing entry for evidence gathering."""

    agent_id: str = Field(..., description="DATA agent ID (e.g., DATA-001)")
    agent_name: str = Field(default="")
    agent_category: str = Field(default="intake")
    purpose: str = Field(default="")
    priority: int = Field(default=1, ge=1, le=10)

class DataRoutingResult(BaseModel):
    """Result of a DATA evidence routing operation."""

    routing_id: str = Field(default_factory=_new_uuid)
    evidence_type: str = Field(default="")
    source_type: EvidenceSourceType = Field(default=EvidenceSourceType.DOCUMENT)
    status: RoutingStatus = Field(default=RoutingStatus.ROUTED)
    intake_agents: List[DataRoutingEntry] = Field(default_factory=list)
    quality_agents: List[DataRoutingEntry] = Field(default_factory=list)
    total_agents: int = Field(default=0)
    quality_level: DataQualityLevel = Field(default=DataQualityLevel.UNKNOWN)
    timestamp: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")
    duration_ms: float = Field(default=0.0)

# ---------------------------------------------------------------------------
# DataClaimsBridge
# ---------------------------------------------------------------------------

class DataClaimsBridge:
    """Routes evidence gathering requests to DATA agents for green claims.

    Maps evidence requirements to the appropriate AGENT-DATA agents
    (001-020) to collect, validate, and track provenance of data that
    substantiates environmental marketing claims.

    Attributes:
        config: Data routing configuration.

    Example:
        >>> config = DataRoutingConfig()
        >>> bridge = DataClaimsBridge(config)
        >>> result = bridge.route_evidence_request("lifecycle_assessment", "document")
        >>> assert result["status"] == "routed"
    """

    def __init__(self, config: Optional[DataRoutingConfig] = None) -> None:
        """Initialize DataClaimsBridge.

        Args:
            config: Routing configuration. Defaults used if None.
        """
        self.config = config or DataRoutingConfig()
        logger.info(
            "DataClaimsBridge initialized (intake=%s, quality=%s, geo=%s)",
            self.config.enable_intake,
            self.config.enable_quality,
            self.config.enable_geo,
        )

    def route_evidence_request(
        self,
        evidence_type: str,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route an evidence gathering request to appropriate DATA agents.

        Args:
            evidence_type: Type of evidence needed (e.g., "lifecycle_assessment").
            source: Optional source type hint (e.g., "document", "erp_system").

        Returns:
            Dict with routing result including agents, quality level, and hash.
        """
        start = utcnow()
        source_type = self._resolve_source_type(source)
        result = DataRoutingResult(
            evidence_type=evidence_type,
            source_type=source_type,
        )

        agent_ids = EVIDENCE_TO_AGENT_MAP.get(evidence_type, [])
        if not agent_ids:
            result.status = RoutingStatus.NO_AGENT
            logger.warning("No DATA agents found for evidence type: %s", evidence_type)
        else:
            intake_entries, quality_entries = self._build_routing_entries(agent_ids)
            result.intake_agents = intake_entries
            result.quality_agents = quality_entries
            result.total_agents = len(intake_entries) + len(quality_entries)
            result.status = RoutingStatus.ROUTED

            if self.config.auto_quality_check and not quality_entries:
                result.status = RoutingStatus.QUALITY_CHECK_NEEDED

        result.duration_ms = (utcnow() - start).total_seconds() * 1000

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        logger.info(
            "DataClaimsBridge routed '%s' to %d agents (intake=%d, quality=%d)",
            evidence_type,
            result.total_agents,
            len(result.intake_agents),
            len(result.quality_agents),
        )

        return result.model_dump(mode="json")

    def get_supported_evidence_types(self) -> List[str]:
        """Return list of all supported evidence types."""
        return list(EVIDENCE_TO_AGENT_MAP.keys())

    def get_routing_summary(self) -> Dict[str, Any]:
        """Get summary of DATA routing configuration.

        Returns:
            Dict with agent counts by category and evidence type count.
        """
        return {
            "intake_agents": len(INTAKE_AGENTS),
            "quality_agents": len(QUALITY_AGENTS),
            "geo_agents": len(GEO_AGENTS),
            "total_agents": len(INTAKE_AGENTS) + len(QUALITY_AGENTS) + len(GEO_AGENTS),
            "supported_evidence_types": len(EVIDENCE_TO_AGENT_MAP),
            "evidence_types": list(EVIDENCE_TO_AGENT_MAP.keys()),
        }

    def route_pdf_extraction(self, document_path: str) -> Dict[str, Any]:
        """Route a PDF document to the PDF & Invoice Extractor (DATA-001).

        Delegates document-based evidence extraction for green claims
        substantiation to AGENT-DATA-001.

        Args:
            document_path: Path or reference to the PDF document.

        Returns:
            Dict with routing result to DATA-001 and provenance hash.
        """
        return self.route_evidence_request(
            "emission_certificates", "document",
        )

    def route_excel_normalization(self, file_path: str) -> Dict[str, Any]:
        """Route a spreadsheet to the Excel/CSV Normalizer (DATA-002).

        Delegates tabular evidence normalization for green claims
        substantiation to AGENT-DATA-002.

        Args:
            file_path: Path or reference to the Excel/CSV file.

        Returns:
            Dict with routing result to DATA-002 and provenance hash.
        """
        return self.route_evidence_request(
            "lifecycle_assessment", "spreadsheet",
        )

    def route_supplier_questionnaire(self, questionnaire_data: Dict[str, Any]) -> Dict[str, Any]:
        """Route supplier questionnaire data to DATA-008.

        Delegates supplier declaration processing for green claims
        evidence to AGENT-DATA-008 (Supplier Questionnaire Processor).

        Args:
            questionnaire_data: Dict with supplier responses.

        Returns:
            Dict with routing result to DATA-008 and provenance hash.
        """
        return self.route_evidence_request(
            "supplier_declarations", "questionnaire",
        )

    def route_data_quality_check(self, dataset_id: str) -> Dict[str, Any]:
        """Route a dataset through the quality assurance pipeline.

        Runs the dataset through DATA-010 (Data Quality Profiler),
        DATA-011 (Duplicate Detection), DATA-013 (Outlier Detection),
        and DATA-019 (Validation Rule Engine).

        Args:
            dataset_id: Identifier of the dataset to quality-check.

        Returns:
            Dict with quality pipeline routing and provenance hash.
        """
        pipeline = self.get_quality_pipeline()
        result: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "pipeline_steps": len(pipeline),
            "pipeline": pipeline,
            "status": "routed",
            "provenance_hash": _compute_hash({"dataset_id": dataset_id}),
        }
        logger.info("DataClaimsBridge quality check routed for dataset '%s'", dataset_id)
        return result

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get overall bridge operational status.

        Returns:
            Dict with bridge health, agent counts, and capabilities.
        """
        summary = self.get_routing_summary()
        return {
            "bridge": "DataClaimsBridge",
            "status": "operational",
            "intake_enabled": self.config.enable_intake,
            "quality_enabled": self.config.enable_quality,
            "geo_enabled": self.config.enable_geo,
            "auto_quality_check": self.config.auto_quality_check,
            "freshness_threshold_days": self.config.freshness_threshold_days,
            **summary,
            "provenance_hash": _compute_hash(summary),
        }

    def get_quality_pipeline(self) -> List[Dict[str, str]]:
        """Get the default quality assurance pipeline order.

        Returns:
            Ordered list of quality agents to run after intake.
        """
        pipeline = [
            {"agent_id": "DATA-010", "step": "profiling", "purpose": "Assess data quality"},
            {"agent_id": "DATA-011", "step": "dedup", "purpose": "Remove duplicates"},
            {"agent_id": "DATA-013", "step": "outlier", "purpose": "Detect anomalies"},
            {"agent_id": "DATA-012", "step": "impute", "purpose": "Fill missing values"},
            {"agent_id": "DATA-016", "step": "freshness", "purpose": "Check data age"},
            {"agent_id": "DATA-019", "step": "validate", "purpose": "Run validation rules"},
            {"agent_id": "DATA-018", "step": "lineage", "purpose": "Track provenance"},
        ]
        return pipeline

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _resolve_source_type(self, source: Optional[str]) -> EvidenceSourceType:
        """Resolve a source hint string to an EvidenceSourceType enum."""
        if not source:
            return EvidenceSourceType.DOCUMENT
        source_map = {
            "document": EvidenceSourceType.DOCUMENT,
            "pdf": EvidenceSourceType.DOCUMENT,
            "spreadsheet": EvidenceSourceType.SPREADSHEET,
            "excel": EvidenceSourceType.SPREADSHEET,
            "csv": EvidenceSourceType.SPREADSHEET,
            "erp": EvidenceSourceType.ERP_SYSTEM,
            "erp_system": EvidenceSourceType.ERP_SYSTEM,
            "api": EvidenceSourceType.EXTERNAL_API,
            "external_api": EvidenceSourceType.EXTERNAL_API,
            "supply_chain": EvidenceSourceType.SUPPLY_CHAIN,
            "geospatial": EvidenceSourceType.GEOSPATIAL,
            "satellite": EvidenceSourceType.SATELLITE,
            "questionnaire": EvidenceSourceType.QUESTIONNAIRE,
            "financial": EvidenceSourceType.FINANCIAL,
            "climate": EvidenceSourceType.CLIMATE_DATA,
        }
        return source_map.get(source.lower(), EvidenceSourceType.DOCUMENT)

    def _build_routing_entries(
        self, agent_ids: List[str]
    ) -> tuple:
        """Split agent IDs into intake and quality routing entries."""
        intake_entries: List[DataRoutingEntry] = []
        quality_entries: List[DataRoutingEntry] = []

        for agent_id in agent_ids:
            category = self._categorize_agent(agent_id)
            if category == "intake" and not self.config.enable_intake:
                continue
            if category == "quality" and not self.config.enable_quality:
                continue
            if category == "geo" and not self.config.enable_geo:
                continue

            name = self._resolve_agent_name(agent_id)
            entry = DataRoutingEntry(
                agent_id=agent_id,
                agent_name=name,
                agent_category=category,
                purpose=f"{category} processing via {name}",
            )

            if category == "intake" or category == "geo":
                intake_entries.append(entry)
            else:
                quality_entries.append(entry)

        return intake_entries, quality_entries

    def _categorize_agent(self, agent_id: str) -> str:
        """Categorize an agent as intake, quality, or geo."""
        if agent_id in INTAKE_AGENTS:
            return "intake"
        if agent_id in QUALITY_AGENTS:
            return "quality"
        if agent_id in GEO_AGENTS:
            return "geo"
        return "unknown"

    def _resolve_agent_name(self, agent_id: str) -> str:
        """Resolve an agent ID to its human-readable name."""
        all_agents = {**INTAKE_AGENTS, **QUALITY_AGENTS, **GEO_AGENTS}
        return all_agents.get(agent_id, "unknown")
