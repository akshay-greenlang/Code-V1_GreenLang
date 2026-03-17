# -*- coding: utf-8 -*-
"""
CSRDPackBridge - ESRS E1/E2/E5 to Battery Regulation Bridge for PACK-020
============================================================================

Maps ESRS environmental disclosures (E1 Climate Change, E2 Pollution,
E5 Resource Use and Circular Economy) to EU Battery Regulation requirements.
Imports climate data for carbon footprint, pollution data for hazardous
substance declarations, and resource/circularity data for recycled content
and end-of-life obligations.

Methods:
    - map_esrs_to_battery_reg()  -- Map all relevant ESRS DRs to Battery Reg articles
    - get_e1_climate_data()      -- Import E1 climate data for carbon footprint (Art 7)
    - get_e2_pollution_data()    -- Import E2 pollution data for substance restrictions
    - get_e5_resource_data()     -- Import E5 circularity data for recycled content (Art 8)

ESRS to Battery Regulation Mapping:
    E1-1 (Transition plan)     -> Art 7 (carbon footprint reduction targets)
    E1-4 (GHG emissions)       -> Art 7 (carbon footprint calculation)
    E1-5 (Energy consumption)  -> Art 7 (manufacturing energy lifecycle stage)
    E1-6 (Scope 1/2/3)        -> Art 7 (lifecycle stage emissions)
    E2-2 (Pollution policies)  -> Art 6 (substance restrictions)
    E2-4 (Pollution of soil)   -> Art 71 (contaminated site management)
    E5-1 (Resource policies)   -> Art 8 (recycled content policies)
    E5-3 (Resource outflows)   -> Art 8, Art 57 (recycled content, recycling)
    E5-4 (Circular economy)    -> Art 11 (removability and replaceability)

Legal References:
    - Regulation (EU) 2023/1542 (EU Battery Regulation)
    - Directive (EU) 2022/2464 (CSRD)
    - ESRS E1, E2, E5 (environmental standards)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-020 Battery Passport Prep Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ESRSStandard(str, Enum):
    """ESRS environmental standards relevant to Battery Regulation."""

    E1_CLIMATE = "E1"
    E2_POLLUTION = "E2"
    E5_CIRCULAR = "E5"


class MappingRelevance(str, Enum):
    """Relevance level of ESRS-to-Battery Regulation mapping."""

    DIRECT = "direct"
    SUPPORTIVE = "supportive"
    CONTEXTUAL = "contextual"


class ImportStatus(str, Enum):
    """Data import operation status."""

    PENDING = "pending"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class CSRDBridgeConfig(BaseModel):
    """Configuration for the CSRD Pack Bridge."""

    pack_id: str = Field(default="PACK-020")
    source_pack_id: str = Field(default="PACK-017")
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    import_e1: bool = Field(default=True, description="Import E1 climate data")
    import_e2: bool = Field(default=True, description="Import E2 pollution data")
    import_e5: bool = Field(default=True, description="Import E5 resource data")


class ESRSDisclosureMapping(BaseModel):
    """Mapping of an ESRS disclosure requirement to Battery Regulation article."""

    esrs_standard: ESRSStandard = Field(default=ESRSStandard.E1_CLIMATE)
    esrs_dr_id: str = Field(default="", description="e.g. E1-4")
    esrs_dr_name: str = Field(default="")
    battery_reg_article: str = Field(default="", description="e.g. Art 7")
    battery_reg_requirement: str = Field(default="")
    relevance: MappingRelevance = Field(default=MappingRelevance.DIRECT)
    data_fields: List[str] = Field(default_factory=list)


class ESRSImportResult(BaseModel):
    """Result of an ESRS standard data import."""

    operation_id: str = Field(default_factory=_new_uuid)
    standard: ESRSStandard = Field(default=ESRSStandard.E1_CLIMATE)
    status: ImportStatus = Field(default=ImportStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    datapoints_imported: int = Field(default=0)
    datapoints_mapped: int = Field(default=0)
    disclosure_requirements_covered: List[str] = Field(default_factory=list)
    battery_articles_mapped: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class BridgeResult(BaseModel):
    """Complete bridge mapping result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: ImportStatus = Field(default=ImportStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    mappings: List[ESRSDisclosureMapping] = Field(default_factory=list)
    total_mappings: int = Field(default=0)
    direct_mappings: int = Field(default=0)
    supportive_mappings: int = Field(default=0)
    contextual_mappings: int = Field(default=0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# ESRS to Battery Regulation Mapping Table
# ---------------------------------------------------------------------------

ESRS_BATTERY_MAPPINGS: List[ESRSDisclosureMapping] = [
    # E1 Climate Change -> Carbon Footprint (Art 7)
    ESRSDisclosureMapping(
        esrs_standard=ESRSStandard.E1_CLIMATE,
        esrs_dr_id="E1-1", esrs_dr_name="Transition plan for climate change mitigation",
        battery_reg_article="Art 7",
        battery_reg_requirement="Carbon footprint reduction trajectory",
        relevance=MappingRelevance.SUPPORTIVE,
        data_fields=["transition_targets", "reduction_pathway"],
    ),
    ESRSDisclosureMapping(
        esrs_standard=ESRSStandard.E1_CLIMATE,
        esrs_dr_id="E1-4", esrs_dr_name="GHG emission reduction targets",
        battery_reg_article="Art 7",
        battery_reg_requirement="Carbon footprint performance classes",
        relevance=MappingRelevance.DIRECT,
        data_fields=["scope1_target", "scope2_target", "scope3_target"],
    ),
    ESRSDisclosureMapping(
        esrs_standard=ESRSStandard.E1_CLIMATE,
        esrs_dr_id="E1-5", esrs_dr_name="Energy consumption and mix",
        battery_reg_article="Art 7",
        battery_reg_requirement="Manufacturing energy in lifecycle stage",
        relevance=MappingRelevance.DIRECT,
        data_fields=["total_energy_mwh", "renewable_pct", "electricity_mix"],
    ),
    ESRSDisclosureMapping(
        esrs_standard=ESRSStandard.E1_CLIMATE,
        esrs_dr_id="E1-6", esrs_dr_name="Gross Scope 1/2/3 and total GHG emissions",
        battery_reg_article="Art 7",
        battery_reg_requirement="Lifecycle stage carbon footprint breakdown",
        relevance=MappingRelevance.DIRECT,
        data_fields=["scope1_tco2e", "scope2_tco2e", "scope3_tco2e", "total_tco2e"],
    ),
    ESRSDisclosureMapping(
        esrs_standard=ESRSStandard.E1_CLIMATE,
        esrs_dr_id="E1-7", esrs_dr_name="GHG removals and mitigation projects",
        battery_reg_article="Art 7",
        battery_reg_requirement="Carbon footprint offsets (not allowed for passport)",
        relevance=MappingRelevance.CONTEXTUAL,
        data_fields=["removals_tco2e", "offset_projects"],
    ),
    # E2 Pollution -> Substance restrictions
    ESRSDisclosureMapping(
        esrs_standard=ESRSStandard.E2_POLLUTION,
        esrs_dr_id="E2-2", esrs_dr_name="Actions related to pollution",
        battery_reg_article="Art 6",
        battery_reg_requirement="Substance of concern restrictions (Annex VI)",
        relevance=MappingRelevance.DIRECT,
        data_fields=["pollution_policies", "substance_management"],
    ),
    ESRSDisclosureMapping(
        esrs_standard=ESRSStandard.E2_POLLUTION,
        esrs_dr_id="E2-4", esrs_dr_name="Pollution of air, water, soil",
        battery_reg_article="Art 6",
        battery_reg_requirement="Hazardous substance content declaration",
        relevance=MappingRelevance.DIRECT,
        data_fields=["mercury_content", "cadmium_content", "lead_content"],
    ),
    ESRSDisclosureMapping(
        esrs_standard=ESRSStandard.E2_POLLUTION,
        esrs_dr_id="E2-5", esrs_dr_name="Substances of concern and very high concern",
        battery_reg_article="Art 6",
        battery_reg_requirement="SVHC in batteries per REACH/Battery Reg Annex VI",
        relevance=MappingRelevance.DIRECT,
        data_fields=["svhc_list", "concentration_limits"],
    ),
    # E5 Resource Use and Circular Economy -> Recycled Content & End-of-Life
    ESRSDisclosureMapping(
        esrs_standard=ESRSStandard.E5_CIRCULAR,
        esrs_dr_id="E5-1", esrs_dr_name="Policies on resource use and circular economy",
        battery_reg_article="Art 8",
        battery_reg_requirement="Recycled content policies and targets",
        relevance=MappingRelevance.SUPPORTIVE,
        data_fields=["circular_economy_policy", "recycled_content_strategy"],
    ),
    ESRSDisclosureMapping(
        esrs_standard=ESRSStandard.E5_CIRCULAR,
        esrs_dr_id="E5-3", esrs_dr_name="Resource outflows (waste)",
        battery_reg_article="Art 57",
        battery_reg_requirement="Collection and recycling targets",
        relevance=MappingRelevance.DIRECT,
        data_fields=["waste_generated_tonnes", "recycled_tonnes", "recycling_rate_pct"],
    ),
    ESRSDisclosureMapping(
        esrs_standard=ESRSStandard.E5_CIRCULAR,
        esrs_dr_id="E5-4", esrs_dr_name="Resource inflows",
        battery_reg_article="Art 8",
        battery_reg_requirement="Recycled content percentage by material",
        relevance=MappingRelevance.DIRECT,
        data_fields=[
            "cobalt_recycled_pct", "lithium_recycled_pct",
            "nickel_recycled_pct", "lead_recycled_pct",
        ],
    ),
    ESRSDisclosureMapping(
        esrs_standard=ESRSStandard.E5_CIRCULAR,
        esrs_dr_id="E5-5", esrs_dr_name="Resource outflows other than waste",
        battery_reg_article="Art 11",
        battery_reg_requirement="Removability and replaceability of batteries",
        relevance=MappingRelevance.SUPPORTIVE,
        data_fields=["design_for_disassembly", "removability_score"],
    ),
]


# ---------------------------------------------------------------------------
# CSRDPackBridge
# ---------------------------------------------------------------------------


class CSRDPackBridge:
    """ESRS E1/E2/E5 to Battery Regulation mapping bridge for PACK-020.

    Maps CSRD/ESRS environmental disclosures to EU Battery Regulation
    requirements. Imports E1 climate data for carbon footprint, E2
    pollution data for substance restrictions, and E5 circularity data
    for recycled content and end-of-life obligations.

    Attributes:
        config: Bridge configuration.
        _mappings: Cached disclosure mappings.

    Example:
        >>> bridge = CSRDPackBridge(CSRDBridgeConfig())
        >>> result = bridge.map_esrs_to_battery_reg()
        >>> assert result.total_mappings >= 12
    """

    def __init__(self, config: Optional[CSRDBridgeConfig] = None) -> None:
        """Initialize CSRDPackBridge."""
        self.config = config or CSRDBridgeConfig()
        self._mappings: List[ESRSDisclosureMapping] = list(ESRS_BATTERY_MAPPINGS)
        logger.info(
            "CSRDPackBridge initialized (source=%s, mappings=%d)",
            self.config.source_pack_id,
            len(self._mappings),
        )

    def map_esrs_to_battery_reg(self) -> BridgeResult:
        """Map all relevant ESRS disclosure requirements to Battery Regulation articles.

        Returns:
            BridgeResult with mapping counts by relevance level.
        """
        result = BridgeResult(started_at=_utcnow())

        try:
            result.mappings = list(self._mappings)
            result.total_mappings = len(result.mappings)
            result.direct_mappings = sum(
                1 for m in result.mappings
                if m.relevance == MappingRelevance.DIRECT
            )
            result.supportive_mappings = sum(
                1 for m in result.mappings
                if m.relevance == MappingRelevance.SUPPORTIVE
            )
            result.contextual_mappings = sum(
                1 for m in result.mappings
                if m.relevance == MappingRelevance.CONTEXTUAL
            )
            result.status = ImportStatus.COMPLETED

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "total": result.total_mappings,
                    "direct": result.direct_mappings,
                })

            logger.info(
                "ESRS mapping: %d total (%d direct, %d supportive, %d contextual)",
                result.total_mappings,
                result.direct_mappings,
                result.supportive_mappings,
                result.contextual_mappings,
            )

        except Exception as exc:
            result.status = ImportStatus.FAILED
            logger.error("ESRS mapping failed: %s", str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def get_e1_climate_data(
        self,
        context: Dict[str, Any],
    ) -> ESRSImportResult:
        """Import E1 climate data for battery carbon footprint (Art 7).

        Args:
            context: Pipeline context with ESRS E1 disclosure data.

        Returns:
            ESRSImportResult with GHG emissions and energy data.
        """
        result = ESRSImportResult(
            standard=ESRSStandard.E1_CLIMATE, started_at=_utcnow()
        )

        try:
            e1_data = context.get("esrs_e1_data", {})

            result.data = {
                "scope1_tco2e": e1_data.get("scope1_tco2e", 0.0),
                "scope2_tco2e": e1_data.get("scope2_tco2e", 0.0),
                "scope3_tco2e": e1_data.get("scope3_tco2e", 0.0),
                "total_tco2e": e1_data.get("total_tco2e", 0.0),
                "total_energy_mwh": e1_data.get("total_energy_mwh", 0.0),
                "renewable_pct": e1_data.get("renewable_pct", 0.0),
                "transition_targets": e1_data.get("transition_targets", {}),
                "electricity_mix": e1_data.get("electricity_mix", {}),
            }
            result.disclosure_requirements_covered = [
                "E1-1", "E1-4", "E1-5", "E1-6", "E1-7",
            ]
            result.battery_articles_mapped = ["Art 7"]
            result.datapoints_imported = len(result.data)
            result.datapoints_mapped = sum(
                1 for v in result.data.values() if v
            )
            result.status = ImportStatus.COMPLETED

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result.data)

            logger.info(
                "E1 import: %d datapoints, %d mapped to Art 7",
                result.datapoints_imported, result.datapoints_mapped,
            )

        except Exception as exc:
            result.status = ImportStatus.FAILED
            result.errors.append(str(exc))
            logger.error("E1 import failed: %s", str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def get_e2_pollution_data(
        self,
        context: Dict[str, Any],
    ) -> ESRSImportResult:
        """Import E2 pollution data for substance restriction compliance.

        Args:
            context: Pipeline context with ESRS E2 disclosure data.

        Returns:
            ESRSImportResult with hazardous substance data.
        """
        result = ESRSImportResult(
            standard=ESRSStandard.E2_POLLUTION, started_at=_utcnow()
        )

        try:
            e2_data = context.get("esrs_e2_data", {})

            result.data = {
                "mercury_content_ppm": e2_data.get("mercury_content_ppm", 0.0),
                "cadmium_content_ppm": e2_data.get("cadmium_content_ppm", 0.0),
                "lead_content_ppm": e2_data.get("lead_content_ppm", 0.0),
                "svhc_substances": e2_data.get("svhc_substances", []),
                "pollution_policies": e2_data.get("pollution_policies", []),
                "substance_management_plan": e2_data.get(
                    "substance_management_plan", False
                ),
                "reach_compliance": e2_data.get("reach_compliance", False),
            }
            result.disclosure_requirements_covered = ["E2-2", "E2-4", "E2-5"]
            result.battery_articles_mapped = ["Art 6"]
            result.datapoints_imported = len(result.data)
            result.datapoints_mapped = sum(
                1 for v in result.data.values() if v
            )
            result.status = ImportStatus.COMPLETED

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result.data)

            logger.info(
                "E2 import: %d datapoints, %d mapped to Art 6",
                result.datapoints_imported, result.datapoints_mapped,
            )

        except Exception as exc:
            result.status = ImportStatus.FAILED
            result.errors.append(str(exc))
            logger.error("E2 import failed: %s", str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def get_e5_resource_data(
        self,
        context: Dict[str, Any],
    ) -> ESRSImportResult:
        """Import E5 resource and circularity data for recycled content (Art 8).

        Args:
            context: Pipeline context with ESRS E5 disclosure data.

        Returns:
            ESRSImportResult with recycled content and circularity data.
        """
        result = ESRSImportResult(
            standard=ESRSStandard.E5_CIRCULAR, started_at=_utcnow()
        )

        try:
            e5_data = context.get("esrs_e5_data", {})

            result.data = {
                "cobalt_recycled_pct": e5_data.get("cobalt_recycled_pct", 0.0),
                "lithium_recycled_pct": e5_data.get("lithium_recycled_pct", 0.0),
                "nickel_recycled_pct": e5_data.get("nickel_recycled_pct", 0.0),
                "lead_recycled_pct": e5_data.get("lead_recycled_pct", 0.0),
                "waste_generated_tonnes": e5_data.get("waste_generated_tonnes", 0.0),
                "recycled_tonnes": e5_data.get("recycled_tonnes", 0.0),
                "recycling_rate_pct": e5_data.get("recycling_rate_pct", 0.0),
                "circular_economy_policy": e5_data.get(
                    "circular_economy_policy", False
                ),
                "design_for_disassembly": e5_data.get(
                    "design_for_disassembly", False
                ),
                "removability_score": e5_data.get("removability_score", 0.0),
            }
            result.disclosure_requirements_covered = [
                "E5-1", "E5-3", "E5-4", "E5-5",
            ]
            result.battery_articles_mapped = ["Art 8", "Art 11", "Art 57"]
            result.datapoints_imported = len(result.data)
            result.datapoints_mapped = sum(
                1 for v in result.data.values() if v
            )
            result.status = ImportStatus.COMPLETED

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result.data)

            logger.info(
                "E5 import: %d datapoints, %d mapped to Art 8/11/57",
                result.datapoints_imported, result.datapoints_mapped,
            )

        except Exception as exc:
            result.status = ImportStatus.FAILED
            result.errors.append(str(exc))
            logger.error("E5 import failed: %s", str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def get_mappings_for_article(self, article: str) -> List[ESRSDisclosureMapping]:
        """Get all ESRS mappings for a given Battery Regulation article.

        Args:
            article: Battery Regulation article (e.g. "Art 7").

        Returns:
            List of ESRSDisclosureMapping for the article.
        """
        return [
            m for m in self._mappings
            if m.battery_reg_article == article
        ]

    def get_mappings_for_standard(
        self,
        standard: ESRSStandard,
    ) -> List[ESRSDisclosureMapping]:
        """Get all mappings for a given ESRS standard.

        Args:
            standard: ESRS standard to filter by.

        Returns:
            List of ESRSDisclosureMapping for the standard.
        """
        return [
            m for m in self._mappings
            if m.esrs_standard == standard
        ]

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status.

        Returns:
            Dict with bridge status information.
        """
        return {
            "pack_id": self.config.pack_id,
            "source_pack_id": self.config.source_pack_id,
            "reporting_year": self.config.reporting_year,
            "total_mappings": len(self._mappings),
            "standards_covered": list({m.esrs_standard.value for m in self._mappings}),
            "articles_mapped": list({m.battery_reg_article for m in self._mappings}),
            "import_e1": self.config.import_e1,
            "import_e2": self.config.import_e2,
            "import_e5": self.config.import_e5,
        }
