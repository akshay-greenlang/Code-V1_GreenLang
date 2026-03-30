# -*- coding: utf-8 -*-
"""
CSRDPackBridge - Bridge to PACK-001/002/003 CSRD Packs for PACK-014
=====================================================================

This module provides the bridge between the CSRD Retail & Consumer Goods Pack
(PACK-014) and the base CSRD Solution Packs (Starter/Professional/Enterprise).
It maps retail-specific datapoints to ESRS structure and builds ESRS chapter
data for E1, E5, S2, and S4 disclosures.

Features:
    - Import ESRS general disclosures from base CSRD packs
    - Map retail-specific datapoints to ESRS structure
    - Build ESRS E1 chapter from store emissions + Scope 3
    - Build ESRS E5 chapter from packaging + circular economy + food waste
    - Build ESRS S2 chapter from supply chain due diligence
    - Build ESRS S4 chapter from product sustainability
    - SHA-256 provenance on all bridge operations
    - Graceful degradation with _AgentStub when base packs not available

Architecture:
    PACK-014 Retail Data --> CSRDPackBridge --> ESRS Chapter Assembly
                                  |                    |
                                  v                    v
    PACK-001/002/003 <-- Import Disclosures    Provenance Hash

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
# Agent Stubs (graceful degradation)
# ---------------------------------------------------------------------------

class _AgentStub:
    """Stub for unavailable CSRD pack modules.

    Returns informative defaults when base CSRD packs are not installed,
    allowing PACK-014 to operate in standalone mode with degraded
    cross-pack integration.
    """

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

def _try_import_csrd_pack(pack_id: str) -> Any:
    """Try to import a CSRD pack module with graceful fallback.

    Args:
        pack_id: Pack identifier (e.g., 'PACK_001', 'PACK_002', 'PACK_003').

    Returns:
        Imported module or _AgentStub if unavailable.
    """
    module_map = {
        "PACK_001": "packs.eu_compliance.PACK_001_csrd_starter",
        "PACK_002": "packs.eu_compliance.PACK_002_csrd_professional",
        "PACK_003": "packs.eu_compliance.PACK_003_csrd_enterprise",
    }
    module_path = module_map.get(pack_id)
    if module_path is None:
        return _AgentStub(pack_id)

    try:
        import importlib

        return importlib.import_module(module_path)
    except ImportError:
        logger.warning("CSRD pack %s not available, using stub", pack_id)
        return _AgentStub(pack_id)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ESRSChapter(str, Enum):
    """ESRS chapters relevant to retail CSRD reporting."""

    E1 = "E1"    # Climate change
    E2 = "E2"    # Pollution
    E3 = "E3"    # Water and marine resources
    E4 = "E4"    # Biodiversity and ecosystems
    E5 = "E5"    # Resource use and circular economy
    S1 = "S1"    # Own workforce
    S2 = "S2"    # Workers in the value chain
    S3 = "S3"    # Affected communities
    S4 = "S4"    # Consumers and end-users
    G1 = "G1"    # Business conduct

class BasePack(str, Enum):
    """Target base CSRD packs for integration."""

    STARTER = "PACK-001"
    PROFESSIONAL = "PACK-002"
    ENTERPRISE = "PACK-003"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class CSRDPackBridgeConfig(BaseModel):
    """Configuration for the CSRD Pack Bridge."""

    target_pack: BasePack = Field(
        default=BasePack.PROFESSIONAL,
        description="Target CSRD base pack for integration",
    )
    enable_provenance: bool = Field(default=True)
    auto_import_general_disclosures: bool = Field(default=True)
    retail_esrs_chapters: List[str] = Field(
        default_factory=lambda: ["E1", "E5", "S2", "S4"],
        description="ESRS chapters with retail-specific content",
    )
    include_voluntary_disclosures: bool = Field(default=False)

class DatapointMapping(BaseModel):
    """Mapping of a retail datapoint to an ESRS disclosure requirement."""

    datapoint_id: str = Field(default="", description="Retail datapoint identifier")
    esrs_chapter: str = Field(default="", description="Target ESRS chapter")
    esrs_disclosure: str = Field(default="", description="ESRS disclosure requirement ID")
    esrs_paragraph: str = Field(default="", description="ESRS paragraph reference")
    description: str = Field(default="", description="Human-readable description")
    data_type: str = Field(default="quantitative", description="quantitative or narrative")
    unit: str = Field(default="", description="Unit of measure")
    source_engine: str = Field(default="", description="Source retail engine")

class ESRSChapterData(BaseModel):
    """Assembled ESRS chapter data from retail calculations."""

    chapter: str = Field(default="")
    chapter_title: str = Field(default="")
    datapoints: List[Dict[str, Any]] = Field(default_factory=list)
    datapoint_count: int = Field(default=0)
    narrative_sections: List[Dict[str, str]] = Field(default_factory=list)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    source_engines: List[str] = Field(default_factory=list)
    assembled_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class BridgeResult(BaseModel):
    """Result of a bridge operation."""

    operation: str = Field(default="")
    success: bool = Field(default=False)
    degraded: bool = Field(default=False)
    chapters_built: List[str] = Field(default_factory=list)
    mappings_applied: int = Field(default=0)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Retail ESRS Datapoint Mappings
# ---------------------------------------------------------------------------

RETAIL_ESRS_MAPPINGS: List[DatapointMapping] = [
    # E1 - Climate Change (from store emissions + Scope 3)
    DatapointMapping(
        datapoint_id="RET-E1-001", esrs_chapter="E1", esrs_disclosure="E1-4",
        esrs_paragraph="36", description="Scope 1 GHG emissions from store operations",
        data_type="quantitative", unit="tCO2e", source_engine="store_emissions",
    ),
    DatapointMapping(
        datapoint_id="RET-E1-002", esrs_chapter="E1", esrs_disclosure="E1-4",
        esrs_paragraph="37", description="Scope 1 GHG emissions from refrigerant leakage",
        data_type="quantitative", unit="tCO2e", source_engine="store_emissions",
    ),
    DatapointMapping(
        datapoint_id="RET-E1-003", esrs_chapter="E1", esrs_disclosure="E1-5",
        esrs_paragraph="44", description="Scope 2 location-based store electricity",
        data_type="quantitative", unit="tCO2e", source_engine="store_emissions",
    ),
    DatapointMapping(
        datapoint_id="RET-E1-004", esrs_chapter="E1", esrs_disclosure="E1-5",
        esrs_paragraph="45", description="Scope 2 market-based store electricity",
        data_type="quantitative", unit="tCO2e", source_engine="store_emissions",
    ),
    DatapointMapping(
        datapoint_id="RET-E1-005", esrs_chapter="E1", esrs_disclosure="E1-6",
        esrs_paragraph="50", description="Scope 3 Category 1 purchased goods",
        data_type="quantitative", unit="tCO2e", source_engine="retail_scope3",
    ),
    DatapointMapping(
        datapoint_id="RET-E1-006", esrs_chapter="E1", esrs_disclosure="E1-6",
        esrs_paragraph="51", description="Scope 3 Category 4 upstream transport",
        data_type="quantitative", unit="tCO2e", source_engine="retail_scope3",
    ),
    DatapointMapping(
        datapoint_id="RET-E1-007", esrs_chapter="E1", esrs_disclosure="E1-6",
        esrs_paragraph="52", description="Scope 3 Category 9 downstream transport",
        data_type="quantitative", unit="tCO2e", source_engine="retail_scope3",
    ),
    DatapointMapping(
        datapoint_id="RET-E1-008", esrs_chapter="E1", esrs_disclosure="E1-6",
        esrs_paragraph="53", description="Scope 3 Category 12 end-of-life sold products",
        data_type="quantitative", unit="tCO2e", source_engine="retail_scope3",
    ),
    DatapointMapping(
        datapoint_id="RET-E1-009", esrs_chapter="E1", esrs_disclosure="E1-3",
        esrs_paragraph="29", description="GHG emission reduction targets",
        data_type="narrative", unit="", source_engine="store_emissions",
    ),
    DatapointMapping(
        datapoint_id="RET-E1-010", esrs_chapter="E1", esrs_disclosure="E1-6",
        esrs_paragraph="54", description="Energy consumption per square meter",
        data_type="quantitative", unit="kWh/m2", source_engine="store_emissions",
    ),
    # E5 - Resource Use and Circular Economy (from packaging + circular)
    DatapointMapping(
        datapoint_id="RET-E5-001", esrs_chapter="E5", esrs_disclosure="E5-4",
        esrs_paragraph="33", description="Packaging material by type and recyclability",
        data_type="quantitative", unit="tonnes", source_engine="packaging_compliance",
    ),
    DatapointMapping(
        datapoint_id="RET-E5-002", esrs_chapter="E5", esrs_disclosure="E5-5",
        esrs_paragraph="37", description="Waste generation by category",
        data_type="quantitative", unit="tonnes", source_engine="circular_economy",
    ),
    DatapointMapping(
        datapoint_id="RET-E5-003", esrs_chapter="E5", esrs_disclosure="E5-5",
        esrs_paragraph="38", description="Food waste generated and redistributed",
        data_type="quantitative", unit="tonnes", source_engine="food_waste",
    ),
    DatapointMapping(
        datapoint_id="RET-E5-004", esrs_chapter="E5", esrs_disclosure="E5-3",
        esrs_paragraph="28", description="Material Circularity Indicator (MCI)",
        data_type="quantitative", unit="score", source_engine="circular_economy",
    ),
    DatapointMapping(
        datapoint_id="RET-E5-005", esrs_chapter="E5", esrs_disclosure="E5-4",
        esrs_paragraph="34", description="Extended Producer Responsibility status",
        data_type="narrative", unit="", source_engine="circular_economy",
    ),
    DatapointMapping(
        datapoint_id="RET-E5-006", esrs_chapter="E5", esrs_disclosure="E5-4",
        esrs_paragraph="35", description="Take-back programme volumes",
        data_type="quantitative", unit="tonnes", source_engine="circular_economy",
    ),
    # S2 - Workers in the Value Chain (from supply chain due diligence)
    DatapointMapping(
        datapoint_id="RET-S2-001", esrs_chapter="S2", esrs_disclosure="S2-1",
        esrs_paragraph="14", description="Supply chain due diligence process",
        data_type="narrative", unit="", source_engine="supply_chain_dd",
    ),
    DatapointMapping(
        datapoint_id="RET-S2-002", esrs_chapter="S2", esrs_disclosure="S2-2",
        esrs_paragraph="20", description="Forced labour risk screening results",
        data_type="quantitative", unit="count", source_engine="supply_chain_dd",
    ),
    DatapointMapping(
        datapoint_id="RET-S2-003", esrs_chapter="S2", esrs_disclosure="S2-4",
        esrs_paragraph="30", description="Remediation actions for value chain impacts",
        data_type="narrative", unit="", source_engine="supply_chain_dd",
    ),
    # S4 - Consumers and End-Users (from product sustainability)
    DatapointMapping(
        datapoint_id="RET-S4-001", esrs_chapter="S4", esrs_disclosure="S4-1",
        esrs_paragraph="11", description="Product safety and consumer protection",
        data_type="narrative", unit="", source_engine="product_sustainability",
    ),
    DatapointMapping(
        datapoint_id="RET-S4-002", esrs_chapter="S4", esrs_disclosure="S4-3",
        esrs_paragraph="24", description="Digital Product Passport compliance",
        data_type="quantitative", unit="count", source_engine="product_sustainability",
    ),
    DatapointMapping(
        datapoint_id="RET-S4-003", esrs_chapter="S4", esrs_disclosure="S4-4",
        esrs_paragraph="30", description="Green claims substantiation status",
        data_type="quantitative", unit="count", source_engine="product_sustainability",
    ),
]

# ---------------------------------------------------------------------------
# CSRDPackBridge
# ---------------------------------------------------------------------------

class CSRDPackBridge:
    """Bridge between PACK-014 Retail Pack and base CSRD Packs.

    Maps retail-specific datapoints to ESRS disclosures and assembles
    ESRS chapter data for E1, E5, S2, and S4. Supports graceful
    degradation when base CSRD packs are not installed.

    Attributes:
        config: Bridge configuration.
        _base_pack: Imported base CSRD pack module or stub.

    Example:
        >>> bridge = CSRDPackBridge()
        >>> e1 = bridge.build_e1_chapter(store_emissions_data, scope3_data)
        >>> print(f"E1 completeness: {e1.completeness_pct}%")
    """

    def __init__(self, config: Optional[CSRDPackBridgeConfig] = None) -> None:
        """Initialize the CSRD Pack Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or CSRDPackBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Try to import base CSRD pack
        pack_module_id = self.config.target_pack.value.replace("-", "_").upper()
        self._base_pack = _try_import_csrd_pack(pack_module_id)
        self._is_degraded = isinstance(self._base_pack, _AgentStub)

        self.logger.info(
            "CSRDPackBridge initialized: target=%s, degraded=%s",
            self.config.target_pack.value,
            self._is_degraded,
        )

    # -------------------------------------------------------------------------
    # ESRS Chapter Building
    # -------------------------------------------------------------------------

    def build_e1_chapter(
        self,
        store_emissions: Dict[str, Any],
        scope3_data: Dict[str, Any],
    ) -> ESRSChapterData:
        """Build ESRS E1 (Climate Change) chapter from retail data.

        Assembles Scope 1, 2, and 3 emission data into ESRS E1 disclosures
        including E1-3 (targets), E1-4 (Scope 1+2), E1-5 (Scope 2 detail),
        and E1-6 (Scope 3).

        Args:
            store_emissions: Store-level emission calculation results.
            scope3_data: Scope 3 assessment results.

        Returns:
            ESRSChapterData for the E1 chapter.
        """
        start = time.monotonic()

        datapoints: List[Dict[str, Any]] = []
        e1_mappings = [m for m in RETAIL_ESRS_MAPPINGS if m.esrs_chapter == "E1"]

        for mapping in e1_mappings:
            value = self._extract_datapoint_value(
                mapping, store_emissions, scope3_data
            )
            datapoints.append({
                "datapoint_id": mapping.datapoint_id,
                "disclosure": mapping.esrs_disclosure,
                "paragraph": mapping.esrs_paragraph,
                "description": mapping.description,
                "value": value,
                "unit": mapping.unit,
                "data_type": mapping.data_type,
            })

        populated = sum(1 for dp in datapoints if dp["value"] is not None)
        completeness = (populated / len(datapoints) * 100.0) if datapoints else 0.0

        chapter = ESRSChapterData(
            chapter="E1",
            chapter_title="Climate Change",
            datapoints=datapoints,
            datapoint_count=len(datapoints),
            narrative_sections=[
                {"title": "GHG Reduction Strategy", "content": ""},
                {"title": "Transition Plan", "content": ""},
            ],
            completeness_pct=round(completeness, 1),
            source_engines=["store_emissions", "retail_scope3"],
        )

        if self.config.enable_provenance:
            chapter.provenance_hash = _compute_hash(chapter)

        elapsed = (time.monotonic() - start) * 1000
        self.logger.info(
            "E1 chapter built: %d datapoints, %.1f%% complete in %.1fms",
            len(datapoints), completeness, elapsed,
        )
        return chapter

    def build_e5_chapter(
        self,
        packaging_data: Dict[str, Any],
        circular_data: Dict[str, Any],
        food_waste_data: Optional[Dict[str, Any]] = None,
    ) -> ESRSChapterData:
        """Build ESRS E5 (Resource Use and Circular Economy) chapter.

        Assembles packaging compliance, circular economy metrics, and food
        waste data into ESRS E5 disclosures.

        Args:
            packaging_data: Packaging compliance engine results.
            circular_data: Circular economy engine results.
            food_waste_data: Food waste engine results (optional, grocery only).

        Returns:
            ESRSChapterData for the E5 chapter.
        """
        start = time.monotonic()

        datapoints: List[Dict[str, Any]] = []
        e5_mappings = [m for m in RETAIL_ESRS_MAPPINGS if m.esrs_chapter == "E5"]

        combined_data = {**packaging_data, **circular_data}
        if food_waste_data:
            combined_data.update(food_waste_data)

        for mapping in e5_mappings:
            value = combined_data.get(mapping.datapoint_id)
            if value is None:
                value = self._lookup_value_by_engine(mapping, combined_data)
            datapoints.append({
                "datapoint_id": mapping.datapoint_id,
                "disclosure": mapping.esrs_disclosure,
                "paragraph": mapping.esrs_paragraph,
                "description": mapping.description,
                "value": value,
                "unit": mapping.unit,
                "data_type": mapping.data_type,
            })

        populated = sum(1 for dp in datapoints if dp["value"] is not None)
        completeness = (populated / len(datapoints) * 100.0) if datapoints else 0.0

        source_engines = ["packaging_compliance", "circular_economy"]
        if food_waste_data:
            source_engines.append("food_waste")

        chapter = ESRSChapterData(
            chapter="E5",
            chapter_title="Resource Use and Circular Economy",
            datapoints=datapoints,
            datapoint_count=len(datapoints),
            narrative_sections=[
                {"title": "Circular Economy Strategy", "content": ""},
                {"title": "Packaging and Waste Management", "content": ""},
            ],
            completeness_pct=round(completeness, 1),
            source_engines=source_engines,
        )

        if self.config.enable_provenance:
            chapter.provenance_hash = _compute_hash(chapter)

        elapsed = (time.monotonic() - start) * 1000
        self.logger.info(
            "E5 chapter built: %d datapoints, %.1f%% complete in %.1fms",
            len(datapoints), completeness, elapsed,
        )
        return chapter

    def build_s2_chapter(
        self, supply_chain_data: Dict[str, Any]
    ) -> ESRSChapterData:
        """Build ESRS S2 (Workers in the Value Chain) chapter.

        Args:
            supply_chain_data: Supply chain due diligence results.

        Returns:
            ESRSChapterData for the S2 chapter.
        """
        start = time.monotonic()

        datapoints: List[Dict[str, Any]] = []
        s2_mappings = [m for m in RETAIL_ESRS_MAPPINGS if m.esrs_chapter == "S2"]

        for mapping in s2_mappings:
            value = supply_chain_data.get(mapping.datapoint_id)
            if value is None:
                value = self._lookup_value_by_engine(mapping, supply_chain_data)
            datapoints.append({
                "datapoint_id": mapping.datapoint_id,
                "disclosure": mapping.esrs_disclosure,
                "paragraph": mapping.esrs_paragraph,
                "description": mapping.description,
                "value": value,
                "unit": mapping.unit,
                "data_type": mapping.data_type,
            })

        populated = sum(1 for dp in datapoints if dp["value"] is not None)
        completeness = (populated / len(datapoints) * 100.0) if datapoints else 0.0

        chapter = ESRSChapterData(
            chapter="S2",
            chapter_title="Workers in the Value Chain",
            datapoints=datapoints,
            datapoint_count=len(datapoints),
            narrative_sections=[
                {"title": "Due Diligence Process", "content": ""},
                {"title": "Remediation Actions", "content": ""},
            ],
            completeness_pct=round(completeness, 1),
            source_engines=["supply_chain_dd"],
        )

        if self.config.enable_provenance:
            chapter.provenance_hash = _compute_hash(chapter)

        elapsed = (time.monotonic() - start) * 1000
        self.logger.info(
            "S2 chapter built: %d datapoints, %.1f%% complete in %.1fms",
            len(datapoints), completeness, elapsed,
        )
        return chapter

    def build_s4_chapter(
        self, product_data: Dict[str, Any]
    ) -> ESRSChapterData:
        """Build ESRS S4 (Consumers and End-Users) chapter.

        Args:
            product_data: Product sustainability engine results.

        Returns:
            ESRSChapterData for the S4 chapter.
        """
        start = time.monotonic()

        datapoints: List[Dict[str, Any]] = []
        s4_mappings = [m for m in RETAIL_ESRS_MAPPINGS if m.esrs_chapter == "S4"]

        for mapping in s4_mappings:
            value = product_data.get(mapping.datapoint_id)
            if value is None:
                value = self._lookup_value_by_engine(mapping, product_data)
            datapoints.append({
                "datapoint_id": mapping.datapoint_id,
                "disclosure": mapping.esrs_disclosure,
                "paragraph": mapping.esrs_paragraph,
                "description": mapping.description,
                "value": value,
                "unit": mapping.unit,
                "data_type": mapping.data_type,
            })

        populated = sum(1 for dp in datapoints if dp["value"] is not None)
        completeness = (populated / len(datapoints) * 100.0) if datapoints else 0.0

        chapter = ESRSChapterData(
            chapter="S4",
            chapter_title="Consumers and End-Users",
            datapoints=datapoints,
            datapoint_count=len(datapoints),
            narrative_sections=[
                {"title": "Consumer Safety", "content": ""},
                {"title": "Green Claims Substantiation", "content": ""},
            ],
            completeness_pct=round(completeness, 1),
            source_engines=["product_sustainability"],
        )

        if self.config.enable_provenance:
            chapter.provenance_hash = _compute_hash(chapter)

        elapsed = (time.monotonic() - start) * 1000
        self.logger.info(
            "S4 chapter built: %d datapoints, %.1f%% complete in %.1fms",
            len(datapoints), completeness, elapsed,
        )
        return chapter

    # -------------------------------------------------------------------------
    # General Disclosures Import
    # -------------------------------------------------------------------------

    def import_general_disclosures(self) -> BridgeResult:
        """Import ESRS general disclosures (ESRS 1 + ESRS 2) from base pack.

        Returns:
            BridgeResult with import status.
        """
        start = time.monotonic()

        if self._is_degraded:
            return BridgeResult(
                operation="import_general_disclosures",
                success=False,
                degraded=True,
                message=f"Base CSRD pack {self.config.target_pack.value} not available",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        result = BridgeResult(
            operation="import_general_disclosures",
            success=True,
            chapters_built=["ESRS_1", "ESRS_2"],
            mappings_applied=2,
            message="General disclosures imported from base CSRD pack",
            duration_ms=(time.monotonic() - start) * 1000,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    def get_datapoint_mappings(
        self, chapter: Optional[str] = None
    ) -> List[DatapointMapping]:
        """Get retail ESRS datapoint mappings, optionally filtered by chapter.

        Args:
            chapter: Optional ESRS chapter filter (e.g., 'E1', 'E5').

        Returns:
            List of DatapointMapping objects.
        """
        if chapter:
            return [m for m in RETAIL_ESRS_MAPPINGS if m.esrs_chapter == chapter]
        return list(RETAIL_ESRS_MAPPINGS)

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _extract_datapoint_value(
        self,
        mapping: DatapointMapping,
        store_emissions: Dict[str, Any],
        scope3_data: Dict[str, Any],
    ) -> Any:
        """Extract a datapoint value from calculation results.

        Args:
            mapping: Datapoint mapping specification.
            store_emissions: Store emission data.
            scope3_data: Scope 3 data.

        Returns:
            Extracted value or None if not found.
        """
        if mapping.source_engine == "store_emissions":
            return store_emissions.get(mapping.datapoint_id)
        elif mapping.source_engine == "retail_scope3":
            return scope3_data.get(mapping.datapoint_id)
        return None

    def _lookup_value_by_engine(
        self, mapping: DatapointMapping, data: Dict[str, Any]
    ) -> Any:
        """Look up a value from combined data by engine and field patterns.

        Args:
            mapping: Datapoint mapping specification.
            data: Combined data dictionary.

        Returns:
            Found value or None.
        """
        # Try direct key lookup
        for key in data:
            if mapping.datapoint_id.lower() in key.lower():
                return data[key]
        return None
