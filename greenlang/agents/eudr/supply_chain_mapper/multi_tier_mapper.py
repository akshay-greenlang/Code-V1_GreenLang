# -*- coding: utf-8 -*-
"""
GL-EUDR-SCM-001 Feature 2: Multi-Tier Recursive Mapper
=======================================================

Implements recursive supply chain discovery from Tier 1 through Tier N down
to production plots for all seven EUDR-regulated commodities (cattle, cocoa,
coffee, palm oil, rubber, soya, wood).

This module integrates with:
    - AGENT-DATA-003 ERP/Finance Connector for Tier 1 procurement data
    - AGENT-DATA-008 Supplier Questionnaire Processor for sub-tier declarations
    - AGENT-DATA-001 PDF Invoice Extractor for custody documents
    - AGENT-DATA-002 Excel/CSV Normalizer for bulk supplier imports

Capabilities:
    - Recursive supply chain discovery (Tier 1 -> Tier N -> Production Plots)
    - Tier depth tracking and completeness metrics
    - Opaque segment identification (missing sub-tier visibility)
    - Support for all 7 EUDR commodity supply chain archetypes
    - Incremental mapping (add tiers without rebuilding entire graph)
    - Tier-depth distribution reporting
    - Comprehensive Prometheus metrics and async patterns

Zero-Hallucination Guarantees:
    - All graph mutations are deterministic
    - All completeness calculations are pure arithmetic
    - NO LLM involvement in mapping logic
    - Complete provenance hash for every mapping operation

Target: Map >= 4 tiers deep for 80%+ of supply chains within 30 days.

PRD: PRD-AGENT-EUDR-001, Feature 2
Agent ID: GL-EUDR-SCM-001
Regulation: EU 2023/1115 (EUDR), Articles 4(2), 9, 10
Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

from pydantic import Field, field_validator, model_validator
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class EUDRCommodity(str, Enum):
    """Seven EUDR-regulated commodities per Regulation (EU) 2023/1115."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class NodeType(str, Enum):
    """Supply chain actor types in EUDR context."""

    PRODUCER = "producer"
    COLLECTOR = "collector"
    PROCESSOR = "processor"
    TRADER = "trader"
    IMPORTER = "importer"
    CERTIFIER = "certifier"
    WAREHOUSE = "warehouse"
    PORT = "port"


class ComplianceStatus(str, Enum):
    """Compliance verification status of a supply chain node."""

    VERIFIED = "verified"
    PENDING_VERIFICATION = "pending_verification"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """EUDR country risk classification per Article 29."""

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


class MappingSourceType(str, Enum):
    """Source from which a supply chain tier was discovered."""

    ERP_PROCUREMENT = "erp_procurement"
    SUPPLIER_QUESTIONNAIRE = "supplier_questionnaire"
    PDF_INVOICE = "pdf_invoice"
    BULK_IMPORT = "bulk_import"
    MANUAL_ENTRY = "manual_entry"
    API_INTEGRATION = "api_integration"


class OpaqueReason(str, Enum):
    """Reason why a supply chain segment is opaque (sub-tier not visible)."""

    SUPPLIER_REFUSED = "supplier_refused"
    NO_QUESTIONNAIRE_RESPONSE = "no_questionnaire_response"
    DATA_UNAVAILABLE = "data_unavailable"
    AGGREGATION_POINT = "aggregation_point"
    CONFIDENTIALITY_CLAIM = "confidentiality_claim"
    UNKNOWN = "unknown"


class DiscoveryStatus(str, Enum):
    """Status of a tier discovery operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


# =============================================================================
# COMMODITY ARCHETYPE CONFIGURATION
# =============================================================================


class CommodityArchetype(GreenLangBase):
    """
    Supply chain archetype for a specific EUDR commodity.

    Each commodity has a typical supply chain structure with expected
    actor types at each tier level. This is used to validate discovered
    tiers and estimate completeness.
    """

    commodity: EUDRCommodity = Field(..., description="EUDR commodity type")
    typical_depth: int = Field(
        ..., ge=2, le=10, description="Typical supply chain depth (tiers)"
    )
    min_depth: int = Field(
        ..., ge=1, description="Minimum expected depth for compliance"
    )
    max_depth: int = Field(
        ..., le=15, description="Maximum realistic depth"
    )
    expected_actor_sequence: List[List[NodeType]] = Field(
        ...,
        description=(
            "Expected actor types at each tier level (from importer backward). "
            "Each tier position is a list of acceptable node types."
        ),
    )
    mapping_complexity: str = Field(
        ..., description="Relative mapping complexity rating"
    )
    key_challenges: List[str] = Field(
        default_factory=list,
        description="Key challenges specific to this commodity chain",
    )


# EUDR commodity archetypes per PRD Section 5.2
COMMODITY_ARCHETYPES: Dict[EUDRCommodity, CommodityArchetype] = {
    EUDRCommodity.CATTLE: CommodityArchetype(
        commodity=EUDRCommodity.CATTLE,
        typical_depth=5,
        min_depth=3,
        max_depth=8,
        expected_actor_sequence=[
            [NodeType.IMPORTER],
            [NodeType.TRADER],
            [NodeType.PROCESSOR],  # Slaughterhouse / Packer
            [NodeType.COLLECTOR],  # Feedlot
            [NodeType.PRODUCER],   # Ranch
        ],
        mapping_complexity="High",
        key_challenges=[
            "Animal movement between ranches",
            "Pasture rotation across plots",
            "Multi-farm feedlot aggregation",
        ],
    ),
    EUDRCommodity.COCOA: CommodityArchetype(
        commodity=EUDRCommodity.COCOA,
        typical_depth=6,
        min_depth=4,
        max_depth=9,
        expected_actor_sequence=[
            [NodeType.IMPORTER],
            [NodeType.TRADER],
            [NodeType.PROCESSOR],      # Cocoa processor / grinder
            [NodeType.COLLECTOR],      # Local collector / buying station
            [NodeType.COLLECTOR],      # Cooperative
            [NodeType.PRODUCER],       # Smallholder farmer
        ],
        mapping_complexity="Very High",
        key_challenges=[
            "Thousands of smallholders per cooperative",
            "Seasonal aggregation destroys traceability",
            "Informal collector networks",
        ],
    ),
    EUDRCommodity.COFFEE: CommodityArchetype(
        commodity=EUDRCommodity.COFFEE,
        typical_depth=5,
        min_depth=3,
        max_depth=8,
        expected_actor_sequence=[
            [NodeType.IMPORTER],
            [NodeType.TRADER],
            [NodeType.PROCESSOR],  # Dry mill / Exporter
            [NodeType.PROCESSOR],  # Wet mill
            [NodeType.PRODUCER],   # Coffee farm
        ],
        mapping_complexity="High",
        key_challenges=[
            "Altitude/origin segregation requirements",
            "Wet mill aggregation from multiple farms",
            "Seasonal harvest batching",
        ],
    ),
    EUDRCommodity.PALM_OIL: CommodityArchetype(
        commodity=EUDRCommodity.PALM_OIL,
        typical_depth=4,
        min_depth=3,
        max_depth=7,
        expected_actor_sequence=[
            [NodeType.IMPORTER],
            [NodeType.TRADER],
            [NodeType.PROCESSOR],  # Refinery
            [NodeType.PROCESSOR],  # Palm oil mill
            [NodeType.PRODUCER],   # Plantation / smallholder
        ],
        mapping_complexity="High",
        key_challenges=[
            "RSPO mass balance complexity",
            "Smallholder supply to mills is opaque",
            "FFB aggregation at mills",
        ],
    ),
    EUDRCommodity.RUBBER: CommodityArchetype(
        commodity=EUDRCommodity.RUBBER,
        typical_depth=5,
        min_depth=3,
        max_depth=8,
        expected_actor_sequence=[
            [NodeType.IMPORTER],
            [NodeType.TRADER],
            [NodeType.PROCESSOR],  # Rubber processor
            [NodeType.COLLECTOR],  # Latex collector
            [NodeType.PRODUCER],   # Smallholder
        ],
        mapping_complexity="High",
        key_challenges=[
            "Latex aggregation destroys traceability",
            "Informal collector networks",
            "Smallholder plot identification",
        ],
    ),
    EUDRCommodity.SOYA: CommodityArchetype(
        commodity=EUDRCommodity.SOYA,
        typical_depth=4,
        min_depth=3,
        max_depth=7,
        expected_actor_sequence=[
            [NodeType.IMPORTER],
            [NodeType.TRADER],
            [NodeType.PROCESSOR],  # Soya crusher
            [NodeType.COLLECTOR, NodeType.WAREHOUSE],  # Silo / warehouse
            [NodeType.PRODUCER],   # Farm
        ],
        mapping_complexity="Medium-High",
        key_challenges=[
            "Large volume co-mingling at silos",
            "Multiple farm sources per silo",
            "Deforestation risk in Cerrado biome",
        ],
    ),
    EUDRCommodity.WOOD: CommodityArchetype(
        commodity=EUDRCommodity.WOOD,
        typical_depth=6,
        min_depth=4,
        max_depth=10,
        expected_actor_sequence=[
            [NodeType.IMPORTER],
            [NodeType.TRADER],
            [NodeType.PROCESSOR],  # Furniture / final product
            [NodeType.PROCESSOR],  # Veneer / plywood
            [NodeType.PROCESSOR],  # Sawmill
            [NodeType.PRODUCER],   # Forest concession
        ],
        mapping_complexity="Very High",
        key_challenges=[
            "Multi-step processing chain (5-8 intermediaries)",
            "Species mixing at sawmills",
            "Batch mixing across forest origins",
        ],
    ),
}


# =============================================================================
# DATA MODELS -- INPUT / OUTPUT
# =============================================================================


class SupplierRecord(GreenLangBase):
    """A supplier record discovered from any data source."""

    supplier_id: str = Field(..., description="Unique supplier identifier")
    supplier_name: str = Field(..., description="Legal name of the supplier")
    country_code: str = Field(
        ..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code"
    )
    region: Optional[str] = Field(None, description="Sub-national region")
    node_type: NodeType = Field(..., description="Type of supply chain actor")
    commodities: List[EUDRCommodity] = Field(
        ..., min_length=1, description="Commodities handled by this supplier"
    )
    coordinates: Optional[Tuple[float, float]] = Field(
        None, description="(latitude, longitude) if known"
    )
    certifications: List[str] = Field(
        default_factory=list, description="Certification IDs (FSC, RSPO, etc.)"
    )
    parent_supplier_id: Optional[str] = Field(
        None, description="ID of the supplier this one supplies to"
    )
    plot_ids: List[str] = Field(
        default_factory=list,
        description="Linked production plot IDs (for producers only)",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.UNKNOWN,
        description="Current compliance verification status",
    )
    source_type: MappingSourceType = Field(
        ..., description="How this supplier was discovered"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the accuracy of this record (0.0-1.0)",
    )
    raw_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Raw metadata from the source system"
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country code is uppercase."""
        return v.upper()

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, v: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Validate coordinates are within valid WGS84 ranges."""
        if v is not None:
            lat, lon = v
            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude {lat} out of range [-90, 90]")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude {lon} out of range [-180, 180]")
        return v


class TierMappingResult(GreenLangBase):
    """Result of mapping a single tier level in the supply chain."""

    tier_depth: int = Field(..., ge=0, description="Tier depth (0 = importer)")
    suppliers_discovered: int = Field(
        ..., ge=0, description="Number of suppliers discovered at this tier"
    )
    suppliers_expected: int = Field(
        ..., ge=0, description="Estimated number of suppliers expected at this tier"
    )
    completeness_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of expected suppliers mapped at this tier",
    )
    source_types: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of discovery source types",
    )
    opaque_segments: int = Field(
        default=0,
        ge=0,
        description="Number of opaque segments at this tier",
    )
    avg_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average confidence score across discovered suppliers",
    )
    node_types_found: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of each node type discovered at this tier",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Time spent mapping this tier in milliseconds"
    )


class OpaqueSegment(GreenLangBase):
    """Represents a segment of the supply chain where sub-tier visibility is missing."""

    segment_id: str = Field(..., description="Unique identifier for this opaque segment")
    parent_node_id: str = Field(
        ..., description="ID of the last known node before the opaque segment"
    )
    parent_node_name: str = Field(
        ..., description="Name of the last known node"
    )
    tier_depth: int = Field(
        ..., ge=0, description="Tier depth where opacity begins"
    )
    commodity: EUDRCommodity = Field(
        ..., description="Commodity flowing through this segment"
    )
    reason: OpaqueReason = Field(
        ..., description="Why this segment is opaque"
    )
    estimated_missing_tiers: int = Field(
        default=1,
        ge=1,
        description="Estimated number of missing tiers in this segment",
    )
    risk_impact: RiskLevel = Field(
        default=RiskLevel.HIGH,
        description="Risk level impact of this opaque segment",
    )
    remediation_action: str = Field(
        default="",
        description="Recommended action to resolve the opacity",
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this opaque segment was detected",
    )


class TierDepthDistribution(GreenLangBase):
    """Distribution report showing supply chain depth across all chains."""

    commodity: EUDRCommodity = Field(..., description="Commodity")
    total_chains: int = Field(
        ..., ge=0, description="Total number of distinct supply chains"
    )
    depth_histogram: Dict[int, int] = Field(
        default_factory=dict,
        description="Count of chains at each depth level",
    )
    median_depth: float = Field(
        default=0.0, description="Median supply chain depth"
    )
    mean_depth: float = Field(
        default=0.0, description="Mean supply chain depth"
    )
    max_depth: int = Field(
        default=0, description="Maximum depth reached"
    )
    min_depth: int = Field(
        default=0, description="Minimum depth found"
    )
    pct_at_target_depth: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of chains reaching the target depth (4+ tiers)",
    )
    target_depth: int = Field(
        default=4, description="Target tier depth for compliance"
    )


class MultiTierMappingInput(GreenLangBase):
    """Input model for multi-tier mapping operation."""

    graph_id: str = Field(..., description="Target supply chain graph ID")
    operator_id: str = Field(..., description="Operator (importer) ID")
    commodity: EUDRCommodity = Field(..., description="Commodity to map")
    max_depth: int = Field(
        default=10,
        ge=1,
        le=15,
        description="Maximum recursion depth to attempt",
    )
    target_depth: int = Field(
        default=4,
        ge=1,
        le=15,
        description="Target depth for compliance (default 4 per PRD)",
    )
    timeout_seconds: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Maximum time for recursive discovery (seconds)",
    )
    incremental: bool = Field(
        default=False,
        description="If True, add to existing graph without rebuilding",
    )
    start_tier: int = Field(
        default=0,
        ge=0,
        description="Tier to start discovery from (for incremental mapping)",
    )
    source_filters: List[MappingSourceType] = Field(
        default_factory=lambda: [
            MappingSourceType.ERP_PROCUREMENT,
            MappingSourceType.SUPPLIER_QUESTIONNAIRE,
            MappingSourceType.PDF_INVOICE,
            MappingSourceType.BULK_IMPORT,
            MappingSourceType.MANUAL_ENTRY,
        ],
        description="Which data sources to query for discovery",
    )


class MultiTierMappingOutput(GreenLangBase):
    """Output model for multi-tier mapping operation."""

    graph_id: str = Field(..., description="Supply chain graph ID")
    operator_id: str = Field(..., description="Operator ID")
    commodity: EUDRCommodity = Field(..., description="Commodity mapped")
    status: DiscoveryStatus = Field(
        ..., description="Overall discovery status"
    )

    # Tier results
    tiers_mapped: int = Field(
        ..., ge=0, description="Number of tiers successfully mapped"
    )
    tier_results: List[TierMappingResult] = Field(
        default_factory=list, description="Per-tier mapping results"
    )

    # Totals
    total_nodes_added: int = Field(
        default=0, ge=0, description="Total new nodes added to graph"
    )
    total_edges_added: int = Field(
        default=0, ge=0, description="Total new edges added to graph"
    )
    total_nodes_in_graph: int = Field(
        default=0, ge=0, description="Total nodes in graph after mapping"
    )

    # Completeness
    overall_completeness_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall supply chain mapping completeness",
    )
    reached_plot_level: bool = Field(
        default=False,
        description="Whether mapping reached production plot level",
    )

    # Opaque segments
    opaque_segments: List[OpaqueSegment] = Field(
        default_factory=list,
        description="Identified opaque segments in the supply chain",
    )

    # Depth distribution
    depth_distribution: Optional[TierDepthDistribution] = Field(
        None, description="Tier depth distribution report"
    )

    # Provenance
    provenance_hash: str = Field(
        ..., description="SHA-256 hash of the mapping operation"
    )
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the mapping operation",
    )

    # Errors / warnings
    warnings: List[str] = Field(
        default_factory=list, description="Non-fatal warnings during mapping"
    )
    errors: List[str] = Field(
        default_factory=list, description="Errors encountered during mapping"
    )


# =============================================================================
# INTEGRATION PROTOCOLS
# =============================================================================


@runtime_checkable
class ERPConnectorProtocol(Protocol):
    """Protocol for AGENT-DATA-003 ERP/Finance Connector integration."""

    async def fetch_procurement_records(
        self,
        operator_id: str,
        commodity: str,
        *,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch Tier 1 procurement records from ERP system."""
        ...


@runtime_checkable
class QuestionnaireProcessorProtocol(Protocol):
    """Protocol for AGENT-DATA-008 Supplier Questionnaire Processor integration."""

    async def fetch_supplier_declarations(
        self,
        supplier_id: str,
        commodity: str,
    ) -> List[Dict[str, Any]]:
        """Fetch sub-tier supplier declarations from questionnaire responses."""
        ...


@runtime_checkable
class PDFExtractorProtocol(Protocol):
    """Protocol for AGENT-DATA-001 PDF Invoice Extractor integration."""

    async def extract_custody_records(
        self,
        supplier_id: str,
        commodity: str,
    ) -> List[Dict[str, Any]]:
        """Extract supplier relationships from custody/invoice documents."""
        ...


@runtime_checkable
class BulkImporterProtocol(Protocol):
    """Protocol for AGENT-DATA-002 Excel/CSV Normalizer integration."""

    async def fetch_bulk_supplier_data(
        self,
        operator_id: str,
        commodity: str,
    ) -> List[Dict[str, Any]]:
        """Fetch bulk-imported supplier data from CSV/Excel files."""
        ...


@runtime_checkable
class GraphStorageProtocol(Protocol):
    """Protocol for the supply chain graph storage backend."""

    async def get_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a supply chain graph by ID."""
        ...

    async def get_nodes_at_tier(
        self, graph_id: str, tier_depth: int
    ) -> List[Dict[str, Any]]:
        """Get all nodes at a specific tier depth in the graph."""
        ...

    async def get_children(
        self, graph_id: str, node_id: str
    ) -> List[Dict[str, Any]]:
        """Get child nodes (suppliers) of a given node."""
        ...

    async def node_exists(self, graph_id: str, supplier_id: str) -> bool:
        """Check if a node with the given supplier_id exists in the graph."""
        ...

    async def add_node(
        self, graph_id: str, node_data: Dict[str, Any]
    ) -> str:
        """Add a node to the graph. Returns the node_id."""
        ...

    async def add_edge(
        self, graph_id: str, edge_data: Dict[str, Any]
    ) -> str:
        """Add an edge to the graph. Returns the edge_id."""
        ...

    async def get_node_count(self, graph_id: str) -> int:
        """Get total node count in graph."""
        ...

    async def get_edge_count(self, graph_id: str) -> int:
        """Get total edge count in graph."""
        ...

    async def get_leaf_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """Get leaf nodes (nodes with no upstream suppliers)."""
        ...

    async def get_all_chain_depths(self, graph_id: str) -> List[int]:
        """Get the depth of every distinct chain (root-to-leaf path) in the graph."""
        ...

    async def update_graph_metadata(
        self, graph_id: str, metadata: Dict[str, Any]
    ) -> None:
        """Update graph-level metadata (total_nodes, max_tier_depth, etc.)."""
        ...


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

try:
    from greenlang.agents.eudr.supply_chain_mapper.metrics import (
        PROMETHEUS_AVAILABLE as _METRICS_AVAILABLE,
        record_node_added as _metrics_record_node_added,
        record_edge_added as _metrics_record_edge_added,
        record_tier_discovery as _metrics_record_tier_discovery,
        record_error as _metrics_record_error,
        observe_processing_duration as _metrics_observe_duration,
    )
except ImportError:
    _METRICS_AVAILABLE = False
    logger.debug(
        "Supply chain mapper metrics module not available; "
        "Prometheus metrics disabled for multi-tier mapper"
    )


# =============================================================================
# MULTI-TIER MAPPER
# =============================================================================


class MultiTierMapper:
    """
    Multi-Tier Recursive Mapper for EUDR supply chain discovery.

    This class implements PRD Feature 2: recursive supply chain discovery from
    Tier 1 (direct suppliers) through Tier N down to production plots. It
    integrates with four AGENT-DATA connectors to discover suppliers from
    multiple data sources, tracks tier depth and completeness, identifies
    opaque segments, and supports incremental mapping.

    The mapper is designed for all seven EUDR-regulated commodities and
    respects each commodity's typical supply chain archetype.

    Attributes:
        graph_storage: Graph storage backend
        erp_connector: AGENT-DATA-003 ERP/Finance Connector
        questionnaire_processor: AGENT-DATA-008 Supplier Questionnaire Processor
        pdf_extractor: AGENT-DATA-001 PDF Invoice Extractor
        bulk_importer: AGENT-DATA-002 Excel/CSV Normalizer
        commodity_archetypes: Commodity archetype definitions

    Example:
        >>> mapper = MultiTierMapper(
        ...     graph_storage=storage,
        ...     erp_connector=erp,
        ...     questionnaire_processor=questionnaire,
        ... )
        >>> result = await mapper.discover_supply_chain(
        ...     MultiTierMappingInput(
        ...         graph_id="graph-123",
        ...         operator_id="op-456",
        ...         commodity=EUDRCommodity.COCOA,
        ...     )
        ... )
        >>> assert result.tiers_mapped >= 4
    """

    AGENT_ID = "GL-EUDR-SCM-001-F2"
    AGENT_NAME = "Multi-Tier Recursive Mapper"
    VERSION = "1.0.0"

    # Target: Map >= 4 tiers deep for 80%+ of supply chains
    DEFAULT_TARGET_DEPTH = 4
    DEFAULT_TARGET_COVERAGE_PCT = 80.0

    def __init__(
        self,
        graph_storage: GraphStorageProtocol,
        erp_connector: Optional[ERPConnectorProtocol] = None,
        questionnaire_processor: Optional[QuestionnaireProcessorProtocol] = None,
        pdf_extractor: Optional[PDFExtractorProtocol] = None,
        bulk_importer: Optional[BulkImporterProtocol] = None,
    ):
        """
        Initialize MultiTierMapper.

        At least one data source connector must be provided. The graph_storage
        backend is required for reading/writing the supply chain graph.

        Args:
            graph_storage: Graph storage backend (required)
            erp_connector: AGENT-DATA-003 ERP/Finance Connector
            questionnaire_processor: AGENT-DATA-008 Supplier Questionnaire Processor
            pdf_extractor: AGENT-DATA-001 PDF Invoice Extractor
            bulk_importer: AGENT-DATA-002 Excel/CSV Normalizer

        Raises:
            ValueError: If no data source connectors are provided
        """
        self._graph_storage = graph_storage
        self._erp_connector = erp_connector
        self._questionnaire_processor = questionnaire_processor
        self._pdf_extractor = pdf_extractor
        self._bulk_importer = bulk_importer

        # Validate at least one source is available
        available_sources = sum([
            erp_connector is not None,
            questionnaire_processor is not None,
            pdf_extractor is not None,
            bulk_importer is not None,
        ])
        if available_sources == 0:
            raise ValueError(
                "At least one data source connector must be provided "
                "(erp_connector, questionnaire_processor, pdf_extractor, "
                "or bulk_importer)"
            )

        self._commodity_archetypes = COMMODITY_ARCHETYPES
        logger.info(
            "%s v%s initialized with %d data source connector(s)",
            self.AGENT_NAME,
            self.VERSION,
            available_sources,
        )

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    async def discover_supply_chain(
        self, input_data: MultiTierMappingInput
    ) -> MultiTierMappingOutput:
        """
        Execute recursive multi-tier supply chain discovery.

        This is the main entry point. It discovers suppliers tier by tier,
        starting from the operator's Tier 1 suppliers and recursing deeper
        until either the max depth is reached, production plots are found,
        or all segments become opaque.

        Args:
            input_data: Validated input parameters for the mapping operation

        Returns:
            MultiTierMappingOutput with tier results, opaque segments,
            and depth distribution

        Raises:
            ValueError: If input validation fails
            TimeoutError: If discovery exceeds the configured timeout
        """
        start_time = time.monotonic()
        start_dt = datetime.now(timezone.utc)
        tier_results: List[TierMappingResult] = []
        opaque_segments: List[OpaqueSegment] = []
        warnings: List[str] = []
        errors: List[str] = []
        total_nodes_added = 0
        total_edges_added = 0

        commodity = input_data.commodity
        archetype = self._commodity_archetypes.get(commodity)
        if archetype is None:
            raise ValueError(f"No archetype defined for commodity: {commodity}")

        logger.info(
            "Starting multi-tier discovery for graph=%s operator=%s commodity=%s "
            "max_depth=%d target_depth=%d incremental=%s",
            input_data.graph_id,
            input_data.operator_id,
            commodity.value,
            input_data.max_depth,
            input_data.target_depth,
            input_data.incremental,
        )

        # Track visited supplier IDs to avoid cycles
        visited_supplier_ids: Set[str] = set()
        status = DiscoveryStatus.IN_PROGRESS

        try:
            # If incremental, load existing visited set from graph
            if input_data.incremental:
                visited_supplier_ids = await self._load_existing_supplier_ids(
                    input_data.graph_id
                )
                logger.info(
                    "Incremental mode: loaded %d existing supplier IDs",
                    len(visited_supplier_ids),
                )

            # Determine the starting tier for discovery
            current_tier = input_data.start_tier

            # Tier 1 special case: discover from ERP/bulk sources
            if current_tier == 0:
                tier_result, tier_nodes, tier_edges, tier_opaque = (
                    await self._discover_tier_one(
                        input_data, archetype, visited_supplier_ids
                    )
                )
                tier_results.append(tier_result)
                opaque_segments.extend(tier_opaque)
                total_nodes_added += tier_nodes
                total_edges_added += tier_edges
                current_tier = 1

            # Recursive discovery: Tier 2 -> Tier N
            for depth in range(
                max(current_tier, 1), input_data.max_depth
            ):
                elapsed = time.monotonic() - start_time
                if elapsed >= input_data.timeout_seconds:
                    warnings.append(
                        f"Discovery timed out at tier {depth} "
                        f"after {elapsed:.1f}s"
                    )
                    status = DiscoveryStatus.TIMED_OUT
                    if _METRICS_AVAILABLE:
                        _metrics_record_error("tier_discover")
                    break

                # Get frontier nodes (nodes at current tier with no children)
                frontier_nodes = await self._get_frontier_nodes(
                    input_data.graph_id, depth
                )

                if not frontier_nodes:
                    logger.info(
                        "No frontier nodes at tier %d; discovery complete",
                        depth,
                    )
                    break

                tier_result, tier_nodes, tier_edges, tier_opaque = (
                    await self._discover_sub_tier(
                        input_data,
                        depth + 1,
                        frontier_nodes,
                        archetype,
                        visited_supplier_ids,
                    )
                )
                tier_results.append(tier_result)
                opaque_segments.extend(tier_opaque)
                total_nodes_added += tier_nodes
                total_edges_added += tier_edges

                # Stop if no new suppliers were found at this tier
                if tier_result.suppliers_discovered == 0:
                    logger.info(
                        "No new suppliers at tier %d; stopping recursion",
                        depth + 1,
                    )
                    break

            # Mark status
            if status == DiscoveryStatus.IN_PROGRESS:
                if opaque_segments:
                    status = DiscoveryStatus.PARTIAL
                else:
                    status = DiscoveryStatus.COMPLETED

        except TimeoutError:
            status = DiscoveryStatus.TIMED_OUT
            errors.append("Discovery operation timed out")
            if _METRICS_AVAILABLE:
                _metrics_record_error("tier_discover")

        except Exception as exc:
            status = DiscoveryStatus.FAILED
            errors.append(f"Discovery failed: {str(exc)}")
            logger.error(
                "Multi-tier discovery failed for graph=%s: %s",
                input_data.graph_id,
                exc,
                exc_info=True,
            )
            if _METRICS_AVAILABLE:
                _metrics_record_error("tier_discover")

        # Compute outputs
        elapsed_ms = (time.monotonic() - start_time) * 1000
        total_nodes_in_graph = await self._safe_get_node_count(
            input_data.graph_id
        )
        overall_completeness = self._calculate_overall_completeness(
            tier_results
        )
        tiers_mapped = len(tier_results)
        reached_plot = self._check_reached_plot_level(tier_results)

        # Build depth distribution
        depth_distribution = await self._build_depth_distribution(
            input_data.graph_id, commodity, input_data.target_depth
        )

        # Update graph metadata
        await self._safe_update_graph_metadata(
            input_data.graph_id,
            total_nodes_in_graph,
            tiers_mapped,
            overall_completeness,
        )

        # Compute provenance hash
        provenance_hash = self._compute_provenance_hash(
            input_data, tier_results, total_nodes_added, total_edges_added
        )

        # Record Prometheus metrics
        self._record_metrics(
            commodity, status, tiers_mapped, elapsed_ms / 1000,
            total_nodes_added, total_edges_added, opaque_segments,
            input_data.graph_id, tier_results,
        )

        output = MultiTierMappingOutput(
            graph_id=input_data.graph_id,
            operator_id=input_data.operator_id,
            commodity=commodity,
            status=status,
            tiers_mapped=tiers_mapped,
            tier_results=tier_results,
            total_nodes_added=total_nodes_added,
            total_edges_added=total_edges_added,
            total_nodes_in_graph=total_nodes_in_graph,
            overall_completeness_pct=overall_completeness,
            reached_plot_level=reached_plot,
            opaque_segments=opaque_segments,
            depth_distribution=depth_distribution,
            provenance_hash=provenance_hash,
            processing_time_ms=elapsed_ms,
            timestamp=start_dt,
            warnings=warnings,
            errors=errors,
        )

        logger.info(
            "Multi-tier discovery completed: graph=%s commodity=%s status=%s "
            "tiers=%d nodes_added=%d edges_added=%d completeness=%.1f%% "
            "opaque=%d elapsed=%.1fms",
            input_data.graph_id,
            commodity.value,
            status.value,
            tiers_mapped,
            total_nodes_added,
            total_edges_added,
            overall_completeness,
            len(opaque_segments),
            elapsed_ms,
        )

        return output

    async def add_tier_incrementally(
        self,
        graph_id: str,
        parent_node_id: str,
        suppliers: List[SupplierRecord],
        commodity: EUDRCommodity,
    ) -> Tuple[int, int]:
        """
        Add new suppliers to an existing graph without rebuilding.

        This supports the incremental mapping requirement: new tiers can be
        added to an existing graph as data becomes available.

        Args:
            graph_id: Target graph ID
            parent_node_id: Node ID that these suppliers supply to
            suppliers: List of supplier records to add
            commodity: Commodity being tracked

        Returns:
            Tuple of (nodes_added, edges_added)

        Raises:
            ValueError: If graph_id or parent_node_id is invalid
        """
        nodes_added = 0
        edges_added = 0

        for supplier in suppliers:
            exists = await self._graph_storage.node_exists(
                graph_id, supplier.supplier_id
            )
            if exists:
                logger.debug(
                    "Supplier %s already exists in graph %s; skipping",
                    supplier.supplier_id,
                    graph_id,
                )
                continue

            # Determine tier depth from parent
            parent_children = await self._graph_storage.get_children(
                graph_id, parent_node_id
            )
            # Parent's tier_depth + 1
            # We need to look up parent tier depth
            parent_nodes = await self._graph_storage.get_nodes_at_tier(
                graph_id, -1  # Fetch all to find parent
            )
            parent_tier = 0
            for node in parent_nodes:
                if node.get("node_id") == parent_node_id or node.get("supplier_id") == parent_node_id:
                    parent_tier = node.get("tier_depth", 0)
                    break
            new_tier = parent_tier + 1

            node_id = await self._add_supplier_node(
                graph_id, supplier, new_tier
            )
            nodes_added += 1

            edge_id = await self._add_supply_edge(
                graph_id, node_id, parent_node_id, supplier, commodity
            )
            edges_added += 1

        logger.info(
            "Incremental mapping: added %d nodes and %d edges to graph %s",
            nodes_added,
            edges_added,
            graph_id,
        )
        return nodes_added, edges_added

    async def get_tier_depth_report(
        self,
        graph_id: str,
        commodity: EUDRCommodity,
        target_depth: int = 4,
    ) -> TierDepthDistribution:
        """
        Generate a tier-depth distribution report for a supply chain graph.

        Args:
            graph_id: Supply chain graph ID
            commodity: Commodity to report on
            target_depth: Target depth for compliance measurement

        Returns:
            TierDepthDistribution with histogram and statistics
        """
        return await self._build_depth_distribution(
            graph_id, commodity, target_depth
        )

    async def identify_opaque_segments(
        self,
        graph_id: str,
        commodity: EUDRCommodity,
    ) -> List[OpaqueSegment]:
        """
        Identify all opaque segments in the supply chain graph.

        An opaque segment is a point in the supply chain where sub-tier
        visibility is missing and no further upstream data is available.

        Args:
            graph_id: Supply chain graph ID
            commodity: Commodity to check

        Returns:
            List of OpaqueSegment instances
        """
        archetype = self._commodity_archetypes.get(commodity)
        if archetype is None:
            raise ValueError(f"No archetype for commodity: {commodity}")

        leaf_nodes = await self._graph_storage.get_leaf_nodes(graph_id)
        opaque_segments: List[OpaqueSegment] = []

        for leaf in leaf_nodes:
            node_type = leaf.get("node_type", "")
            tier_depth = leaf.get("tier_depth", 0)

            # A leaf node is opaque if it is NOT a producer (production plot)
            # and its tier depth is less than the minimum expected depth
            if node_type != NodeType.PRODUCER.value:
                if tier_depth < archetype.min_depth:
                    segment = OpaqueSegment(
                        segment_id=self._generate_segment_id(
                            graph_id, leaf.get("node_id", "")
                        ),
                        parent_node_id=leaf.get("node_id", ""),
                        parent_node_name=leaf.get("operator_name", "Unknown"),
                        tier_depth=tier_depth,
                        commodity=commodity,
                        reason=OpaqueReason.DATA_UNAVAILABLE,
                        estimated_missing_tiers=max(
                            1, archetype.min_depth - tier_depth
                        ),
                        risk_impact=RiskLevel.HIGH,
                        remediation_action=(
                            f"Send sub-tier discovery questionnaire to "
                            f"{leaf.get('operator_name', 'Unknown')} "
                            f"(tier {tier_depth})"
                        ),
                    )
                    opaque_segments.append(segment)

        return opaque_segments

    # -------------------------------------------------------------------------
    # TIER 1 DISCOVERY (ERP + Bulk Import)
    # -------------------------------------------------------------------------

    async def _discover_tier_one(
        self,
        input_data: MultiTierMappingInput,
        archetype: CommodityArchetype,
        visited: Set[str],
    ) -> Tuple[TierMappingResult, int, int, List[OpaqueSegment]]:
        """
        Discover Tier 1 suppliers from ERP procurement and bulk imports.

        Tier 1 suppliers are the direct suppliers of the operator. They are
        typically discovered from:
        - AGENT-DATA-003 ERP/Finance Connector (purchase orders, invoices)
        - AGENT-DATA-002 Excel/CSV Normalizer (manually uploaded supplier lists)
        - AGENT-DATA-001 PDF Invoice Extractor (scanned custody documents)

        Args:
            input_data: Mapping input parameters
            archetype: Commodity archetype configuration
            visited: Set of already-visited supplier IDs

        Returns:
            Tuple of (TierMappingResult, nodes_added, edges_added, opaque_segments)
        """
        tier_start = time.monotonic()
        suppliers: List[SupplierRecord] = []
        source_counts: Dict[str, int] = defaultdict(int)
        tier_depth = 1

        # Source 1: ERP procurement records
        if (
            self._erp_connector is not None
            and MappingSourceType.ERP_PROCUREMENT in input_data.source_filters
        ):
            erp_suppliers = await self._fetch_erp_tier_one(
                input_data.operator_id, input_data.commodity
            )
            for s in erp_suppliers:
                if s.supplier_id not in visited:
                    suppliers.append(s)
                    source_counts[MappingSourceType.ERP_PROCUREMENT.value] += 1
                    visited.add(s.supplier_id)

        # Source 2: Bulk imports (CSV/Excel)
        if (
            self._bulk_importer is not None
            and MappingSourceType.BULK_IMPORT in input_data.source_filters
        ):
            bulk_suppliers = await self._fetch_bulk_tier_one(
                input_data.operator_id, input_data.commodity
            )
            for s in bulk_suppliers:
                if s.supplier_id not in visited:
                    suppliers.append(s)
                    source_counts[MappingSourceType.BULK_IMPORT.value] += 1
                    visited.add(s.supplier_id)

        # Source 3: PDF invoice extraction
        if (
            self._pdf_extractor is not None
            and MappingSourceType.PDF_INVOICE in input_data.source_filters
        ):
            pdf_suppliers = await self._fetch_pdf_tier_one(
                input_data.operator_id, input_data.commodity
            )
            for s in pdf_suppliers:
                if s.supplier_id not in visited:
                    suppliers.append(s)
                    source_counts[MappingSourceType.PDF_INVOICE.value] += 1
                    visited.add(s.supplier_id)

        # Add discovered suppliers to graph
        nodes_added = 0
        edges_added = 0
        for supplier in suppliers:
            node_id = await self._add_supplier_node(
                input_data.graph_id, supplier, tier_depth
            )
            nodes_added += 1

            # Edge from this supplier to the operator (importer)
            edge_id = await self._add_supply_edge(
                input_data.graph_id,
                node_id,
                input_data.operator_id,
                supplier,
                input_data.commodity,
            )
            edges_added += 1

        # Calculate completeness estimate for Tier 1
        expected_t1 = self._estimate_expected_suppliers(
            archetype, tier_depth, len(suppliers)
        )
        completeness = self._compute_tier_completeness(
            len(suppliers), expected_t1
        )

        # Build node type counts
        node_types: Dict[str, int] = defaultdict(int)
        for s in suppliers:
            node_types[s.node_type.value] += 1

        # Average confidence
        avg_conf = self._compute_average_confidence(suppliers)

        tier_elapsed = (time.monotonic() - tier_start) * 1000

        tier_result = TierMappingResult(
            tier_depth=tier_depth,
            suppliers_discovered=len(suppliers),
            suppliers_expected=expected_t1,
            completeness_pct=completeness,
            source_types=dict(source_counts),
            opaque_segments=0,
            avg_confidence=avg_conf,
            node_types_found=dict(node_types),
            processing_time_ms=tier_elapsed,
        )

        logger.info(
            "Tier 1 discovery: found %d suppliers (expected ~%d), "
            "completeness=%.1f%%, elapsed=%.1fms",
            len(suppliers),
            expected_t1,
            completeness,
            tier_elapsed,
        )

        return tier_result, nodes_added, edges_added, []

    # -------------------------------------------------------------------------
    # SUB-TIER DISCOVERY (Tier 2+)
    # -------------------------------------------------------------------------

    async def _discover_sub_tier(
        self,
        input_data: MultiTierMappingInput,
        tier_depth: int,
        frontier_nodes: List[Dict[str, Any]],
        archetype: CommodityArchetype,
        visited: Set[str],
    ) -> Tuple[TierMappingResult, int, int, List[OpaqueSegment]]:
        """
        Discover sub-tier suppliers (Tier 2+) using questionnaires and documents.

        For each frontier node (a node at the current boundary with no
        upstream suppliers), this method queries:
        - AGENT-DATA-008 Supplier Questionnaire Processor for sub-tier declarations
        - AGENT-DATA-001 PDF Invoice Extractor for custody documents
        - AGENT-DATA-002 Excel/CSV Normalizer for bulk declarations

        Args:
            input_data: Mapping input parameters
            tier_depth: Tier depth being discovered
            frontier_nodes: Nodes at the current frontier
            archetype: Commodity archetype configuration
            visited: Set of already-visited supplier IDs

        Returns:
            Tuple of (TierMappingResult, nodes_added, edges_added, opaque_segments)
        """
        tier_start = time.monotonic()
        all_suppliers: List[SupplierRecord] = []
        source_counts: Dict[str, int] = defaultdict(int)
        opaque_segments: List[OpaqueSegment] = []
        nodes_added = 0
        edges_added = 0

        for frontier_node in frontier_nodes:
            frontier_id = frontier_node.get("node_id", "")
            frontier_supplier_id = frontier_node.get(
                "supplier_id", frontier_id
            )
            frontier_name = frontier_node.get("operator_name", "Unknown")

            # Collect sub-tier suppliers from all available sources
            sub_suppliers: List[SupplierRecord] = []

            # Source 1: Supplier questionnaire declarations
            if (
                self._questionnaire_processor is not None
                and MappingSourceType.SUPPLIER_QUESTIONNAIRE
                in input_data.source_filters
            ):
                q_suppliers = await self._fetch_questionnaire_sub_tier(
                    frontier_supplier_id, input_data.commodity
                )
                for s in q_suppliers:
                    if s.supplier_id not in visited:
                        s.parent_supplier_id = frontier_id
                        sub_suppliers.append(s)
                        source_counts[
                            MappingSourceType.SUPPLIER_QUESTIONNAIRE.value
                        ] += 1
                        visited.add(s.supplier_id)

            # Source 2: PDF custody documents
            if (
                self._pdf_extractor is not None
                and MappingSourceType.PDF_INVOICE in input_data.source_filters
            ):
                pdf_suppliers = await self._fetch_pdf_sub_tier(
                    frontier_supplier_id, input_data.commodity
                )
                for s in pdf_suppliers:
                    if s.supplier_id not in visited:
                        s.parent_supplier_id = frontier_id
                        sub_suppliers.append(s)
                        source_counts[
                            MappingSourceType.PDF_INVOICE.value
                        ] += 1
                        visited.add(s.supplier_id)

            # Source 3: Bulk import data
            if (
                self._bulk_importer is not None
                and MappingSourceType.BULK_IMPORT in input_data.source_filters
            ):
                bulk_suppliers = await self._fetch_bulk_sub_tier(
                    frontier_supplier_id, input_data.commodity
                )
                for s in bulk_suppliers:
                    if s.supplier_id not in visited:
                        s.parent_supplier_id = frontier_id
                        sub_suppliers.append(s)
                        source_counts[
                            MappingSourceType.BULK_IMPORT.value
                        ] += 1
                        visited.add(s.supplier_id)

            # If no sub-tier suppliers found, mark as opaque
            if not sub_suppliers:
                node_type = frontier_node.get("node_type", "")
                if node_type != NodeType.PRODUCER.value:
                    opaque = OpaqueSegment(
                        segment_id=self._generate_segment_id(
                            input_data.graph_id, frontier_id
                        ),
                        parent_node_id=frontier_id,
                        parent_node_name=frontier_name,
                        tier_depth=tier_depth,
                        commodity=input_data.commodity,
                        reason=OpaqueReason.NO_QUESTIONNAIRE_RESPONSE,
                        estimated_missing_tiers=max(
                            1, archetype.min_depth - tier_depth
                        ),
                        risk_impact=RiskLevel.HIGH,
                        remediation_action=(
                            f"Send sub-tier discovery questionnaire to "
                            f"{frontier_name} (tier {tier_depth - 1})"
                        ),
                    )
                    opaque_segments.append(opaque)
                continue

            # Add discovered suppliers to graph
            for supplier in sub_suppliers:
                node_id = await self._add_supplier_node(
                    input_data.graph_id, supplier, tier_depth
                )
                nodes_added += 1

                edge_id = await self._add_supply_edge(
                    input_data.graph_id,
                    node_id,
                    frontier_id,
                    supplier,
                    input_data.commodity,
                )
                edges_added += 1

            all_suppliers.extend(sub_suppliers)

        # Calculate tier metrics
        expected = self._estimate_expected_suppliers(
            archetype, tier_depth, len(all_suppliers)
        )
        completeness = self._compute_tier_completeness(
            len(all_suppliers), expected
        )

        node_types: Dict[str, int] = defaultdict(int)
        for s in all_suppliers:
            node_types[s.node_type.value] += 1

        avg_conf = self._compute_average_confidence(all_suppliers)
        tier_elapsed = (time.monotonic() - tier_start) * 1000

        tier_result = TierMappingResult(
            tier_depth=tier_depth,
            suppliers_discovered=len(all_suppliers),
            suppliers_expected=expected,
            completeness_pct=completeness,
            source_types=dict(source_counts),
            opaque_segments=len(opaque_segments),
            avg_confidence=avg_conf,
            node_types_found=dict(node_types),
            processing_time_ms=tier_elapsed,
        )

        logger.info(
            "Tier %d discovery: found %d suppliers from %d frontier nodes, "
            "opaque=%d, completeness=%.1f%%, elapsed=%.1fms",
            tier_depth,
            len(all_suppliers),
            len(frontier_nodes),
            len(opaque_segments),
            completeness,
            tier_elapsed,
        )

        return tier_result, nodes_added, edges_added, opaque_segments

    # -------------------------------------------------------------------------
    # DATA SOURCE ADAPTERS
    # -------------------------------------------------------------------------

    async def _fetch_erp_tier_one(
        self, operator_id: str, commodity: EUDRCommodity
    ) -> List[SupplierRecord]:
        """
        Fetch Tier 1 suppliers from AGENT-DATA-003 ERP/Finance Connector.

        Converts ERP procurement records into SupplierRecord instances.

        Args:
            operator_id: Operator ID to query procurement for
            commodity: EUDR commodity type

        Returns:
            List of SupplierRecord instances from ERP data
        """
        if self._erp_connector is None:
            return []

        try:
            raw_records = await self._erp_connector.fetch_procurement_records(
                operator_id, commodity.value
            )
            suppliers = []
            for record in raw_records:
                supplier = self._parse_erp_record(record, commodity)
                if supplier is not None:
                    suppliers.append(supplier)
            logger.debug(
                "ERP connector returned %d Tier 1 suppliers for operator %s",
                len(suppliers),
                operator_id,
            )
            return suppliers

        except Exception as exc:
            logger.warning(
                "ERP connector fetch failed for operator %s: %s",
                operator_id,
                exc,
            )
            if _METRICS_AVAILABLE:
                _metrics_record_error("tier_discover")
            return []

    async def _fetch_bulk_tier_one(
        self, operator_id: str, commodity: EUDRCommodity
    ) -> List[SupplierRecord]:
        """
        Fetch Tier 1 suppliers from AGENT-DATA-002 bulk imports.

        Args:
            operator_id: Operator ID
            commodity: EUDR commodity type

        Returns:
            List of SupplierRecord instances from bulk imports
        """
        if self._bulk_importer is None:
            return []

        try:
            raw_records = await self._bulk_importer.fetch_bulk_supplier_data(
                operator_id, commodity.value
            )
            suppliers = []
            for record in raw_records:
                supplier = self._parse_bulk_record(record, commodity)
                if supplier is not None:
                    suppliers.append(supplier)
            logger.debug(
                "Bulk importer returned %d Tier 1 suppliers",
                len(suppliers),
            )
            return suppliers

        except Exception as exc:
            logger.warning(
                "Bulk importer fetch failed for operator %s: %s",
                operator_id,
                exc,
            )
            if _METRICS_AVAILABLE:
                _metrics_record_error("tier_discover")
            return []

    async def _fetch_pdf_tier_one(
        self, operator_id: str, commodity: EUDRCommodity
    ) -> List[SupplierRecord]:
        """
        Fetch Tier 1 suppliers from AGENT-DATA-001 PDF Invoice Extractor.

        Args:
            operator_id: Operator ID
            commodity: EUDR commodity type

        Returns:
            List of SupplierRecord instances from PDF documents
        """
        if self._pdf_extractor is None:
            return []

        try:
            raw_records = await self._pdf_extractor.extract_custody_records(
                operator_id, commodity.value
            )
            suppliers = []
            for record in raw_records:
                supplier = self._parse_pdf_record(record, commodity)
                if supplier is not None:
                    suppliers.append(supplier)
            logger.debug(
                "PDF extractor returned %d Tier 1 suppliers",
                len(suppliers),
            )
            return suppliers

        except Exception as exc:
            logger.warning(
                "PDF extractor fetch failed for operator %s: %s",
                operator_id,
                exc,
            )
            if _METRICS_AVAILABLE:
                _metrics_record_error("tier_discover")
            return []

    async def _fetch_questionnaire_sub_tier(
        self, supplier_id: str, commodity: EUDRCommodity
    ) -> List[SupplierRecord]:
        """
        Fetch sub-tier suppliers from AGENT-DATA-008 Supplier Questionnaire.

        Args:
            supplier_id: Parent supplier ID
            commodity: EUDR commodity type

        Returns:
            List of SupplierRecord instances from questionnaire responses
        """
        if self._questionnaire_processor is None:
            return []

        try:
            raw_records = (
                await self._questionnaire_processor.fetch_supplier_declarations(
                    supplier_id, commodity.value
                )
            )
            suppliers = []
            for record in raw_records:
                supplier = self._parse_questionnaire_record(record, commodity)
                if supplier is not None:
                    suppliers.append(supplier)
            logger.debug(
                "Questionnaire processor returned %d sub-tier suppliers "
                "for supplier %s",
                len(suppliers),
                supplier_id,
            )
            return suppliers

        except Exception as exc:
            logger.warning(
                "Questionnaire fetch failed for supplier %s: %s",
                supplier_id,
                exc,
            )
            if _METRICS_AVAILABLE:
                _metrics_record_error("tier_discover")
            return []

    async def _fetch_pdf_sub_tier(
        self, supplier_id: str, commodity: EUDRCommodity
    ) -> List[SupplierRecord]:
        """
        Fetch sub-tier suppliers from AGENT-DATA-001 PDF documents
        associated with a specific supplier.

        Args:
            supplier_id: Parent supplier ID
            commodity: EUDR commodity type

        Returns:
            List of SupplierRecord instances from PDF custody records
        """
        if self._pdf_extractor is None:
            return []

        try:
            raw_records = await self._pdf_extractor.extract_custody_records(
                supplier_id, commodity.value
            )
            suppliers = []
            for record in raw_records:
                supplier = self._parse_pdf_record(record, commodity)
                if supplier is not None:
                    suppliers.append(supplier)
            return suppliers

        except Exception as exc:
            logger.warning(
                "PDF sub-tier fetch failed for supplier %s: %s",
                supplier_id,
                exc,
            )
            return []

    async def _fetch_bulk_sub_tier(
        self, supplier_id: str, commodity: EUDRCommodity
    ) -> List[SupplierRecord]:
        """
        Fetch sub-tier suppliers from bulk imports for a specific supplier.

        Args:
            supplier_id: Parent supplier ID
            commodity: EUDR commodity type

        Returns:
            List of SupplierRecord instances from bulk data
        """
        if self._bulk_importer is None:
            return []

        try:
            raw_records = await self._bulk_importer.fetch_bulk_supplier_data(
                supplier_id, commodity.value
            )
            suppliers = []
            for record in raw_records:
                supplier = self._parse_bulk_record(record, commodity)
                if supplier is not None:
                    suppliers.append(supplier)
            return suppliers

        except Exception as exc:
            logger.warning(
                "Bulk sub-tier fetch failed for supplier %s: %s",
                supplier_id,
                exc,
            )
            return []

    # -------------------------------------------------------------------------
    # RECORD PARSERS
    # -------------------------------------------------------------------------

    def _parse_erp_record(
        self, record: Dict[str, Any], commodity: EUDRCommodity
    ) -> Optional[SupplierRecord]:
        """
        Parse an ERP procurement record into a SupplierRecord.

        Expected ERP record fields:
            - supplier_id (str): Unique supplier identifier
            - supplier_name (str): Legal name
            - country_code (str): ISO 3166-1 alpha-2
            - node_type (str, optional): Actor type
            - region (str, optional): Sub-national region
            - latitude (float, optional): GPS latitude
            - longitude (float, optional): GPS longitude
            - certifications (list, optional): Certification IDs

        Args:
            record: Raw ERP record dictionary
            commodity: EUDR commodity type

        Returns:
            SupplierRecord or None if record is invalid
        """
        try:
            supplier_id = record.get("supplier_id", "")
            supplier_name = record.get("supplier_name", "")
            country_code = record.get("country_code", "")

            if not supplier_id or not supplier_name or not country_code:
                logger.debug(
                    "Skipping ERP record with missing required fields: %s",
                    record,
                )
                return None

            node_type_str = record.get("node_type", NodeType.TRADER.value)
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                node_type = NodeType.TRADER

            coordinates = None
            lat = record.get("latitude")
            lon = record.get("longitude")
            if lat is not None and lon is not None:
                coordinates = (float(lat), float(lon))

            return SupplierRecord(
                supplier_id=supplier_id,
                supplier_name=supplier_name,
                country_code=country_code,
                region=record.get("region"),
                node_type=node_type,
                commodities=[commodity],
                coordinates=coordinates,
                certifications=record.get("certifications", []),
                plot_ids=record.get("plot_ids", []),
                compliance_status=ComplianceStatus(
                    record.get(
                        "compliance_status",
                        ComplianceStatus.UNKNOWN.value,
                    )
                ),
                source_type=MappingSourceType.ERP_PROCUREMENT,
                confidence_score=float(
                    record.get("confidence_score", 0.8)
                ),
                raw_metadata=record,
            )

        except Exception as exc:
            logger.warning("Failed to parse ERP record: %s", exc)
            return None

    def _parse_questionnaire_record(
        self, record: Dict[str, Any], commodity: EUDRCommodity
    ) -> Optional[SupplierRecord]:
        """
        Parse a supplier questionnaire response into a SupplierRecord.

        Args:
            record: Raw questionnaire record dictionary
            commodity: EUDR commodity type

        Returns:
            SupplierRecord or None if record is invalid
        """
        try:
            supplier_id = record.get("supplier_id", "")
            supplier_name = record.get("supplier_name", "")
            country_code = record.get("country_code", "")

            if not supplier_id or not supplier_name or not country_code:
                return None

            node_type_str = record.get("node_type", NodeType.COLLECTOR.value)
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                node_type = NodeType.COLLECTOR

            coordinates = None
            lat = record.get("latitude")
            lon = record.get("longitude")
            if lat is not None and lon is not None:
                coordinates = (float(lat), float(lon))

            return SupplierRecord(
                supplier_id=supplier_id,
                supplier_name=supplier_name,
                country_code=country_code,
                region=record.get("region"),
                node_type=node_type,
                commodities=[commodity],
                coordinates=coordinates,
                certifications=record.get("certifications", []),
                plot_ids=record.get("plot_ids", []),
                compliance_status=ComplianceStatus(
                    record.get(
                        "compliance_status",
                        ComplianceStatus.UNKNOWN.value,
                    )
                ),
                source_type=MappingSourceType.SUPPLIER_QUESTIONNAIRE,
                confidence_score=float(
                    record.get("confidence_score", 0.6)
                ),
                raw_metadata=record,
            )

        except Exception as exc:
            logger.warning("Failed to parse questionnaire record: %s", exc)
            return None

    def _parse_pdf_record(
        self, record: Dict[str, Any], commodity: EUDRCommodity
    ) -> Optional[SupplierRecord]:
        """
        Parse a PDF-extracted custody record into a SupplierRecord.

        Args:
            record: Raw PDF-extracted record dictionary
            commodity: EUDR commodity type

        Returns:
            SupplierRecord or None if record is invalid
        """
        try:
            supplier_id = record.get("supplier_id", "")
            supplier_name = record.get("supplier_name", "")
            country_code = record.get("country_code", "")

            if not supplier_id or not supplier_name or not country_code:
                return None

            node_type_str = record.get("node_type", NodeType.TRADER.value)
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                node_type = NodeType.TRADER

            coordinates = None
            lat = record.get("latitude")
            lon = record.get("longitude")
            if lat is not None and lon is not None:
                coordinates = (float(lat), float(lon))

            return SupplierRecord(
                supplier_id=supplier_id,
                supplier_name=supplier_name,
                country_code=country_code,
                region=record.get("region"),
                node_type=node_type,
                commodities=[commodity],
                coordinates=coordinates,
                certifications=record.get("certifications", []),
                plot_ids=record.get("plot_ids", []),
                compliance_status=ComplianceStatus(
                    record.get(
                        "compliance_status",
                        ComplianceStatus.UNKNOWN.value,
                    )
                ),
                source_type=MappingSourceType.PDF_INVOICE,
                confidence_score=float(
                    record.get("confidence_score", 0.5)
                ),
                raw_metadata=record,
            )

        except Exception as exc:
            logger.warning("Failed to parse PDF record: %s", exc)
            return None

    def _parse_bulk_record(
        self, record: Dict[str, Any], commodity: EUDRCommodity
    ) -> Optional[SupplierRecord]:
        """
        Parse a bulk-imported (CSV/Excel) record into a SupplierRecord.

        Args:
            record: Raw bulk import record dictionary
            commodity: EUDR commodity type

        Returns:
            SupplierRecord or None if record is invalid
        """
        try:
            supplier_id = record.get("supplier_id", "")
            supplier_name = record.get("supplier_name", "")
            country_code = record.get("country_code", "")

            if not supplier_id or not supplier_name or not country_code:
                return None

            node_type_str = record.get("node_type", NodeType.TRADER.value)
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                node_type = NodeType.TRADER

            coordinates = None
            lat = record.get("latitude")
            lon = record.get("longitude")
            if lat is not None and lon is not None:
                coordinates = (float(lat), float(lon))

            return SupplierRecord(
                supplier_id=supplier_id,
                supplier_name=supplier_name,
                country_code=country_code,
                region=record.get("region"),
                node_type=node_type,
                commodities=[commodity],
                coordinates=coordinates,
                certifications=record.get("certifications", []),
                plot_ids=record.get("plot_ids", []),
                compliance_status=ComplianceStatus(
                    record.get(
                        "compliance_status",
                        ComplianceStatus.UNKNOWN.value,
                    )
                ),
                source_type=MappingSourceType.BULK_IMPORT,
                confidence_score=float(
                    record.get("confidence_score", 0.7)
                ),
                raw_metadata=record,
            )

        except Exception as exc:
            logger.warning("Failed to parse bulk record: %s", exc)
            return None

    # -------------------------------------------------------------------------
    # GRAPH OPERATIONS
    # -------------------------------------------------------------------------

    async def _add_supplier_node(
        self,
        graph_id: str,
        supplier: SupplierRecord,
        tier_depth: int,
    ) -> str:
        """
        Add a supplier as a node in the supply chain graph.

        Args:
            graph_id: Target graph ID
            supplier: Supplier record to add
            tier_depth: Tier depth of this supplier

        Returns:
            Node ID of the added node
        """
        node_data = {
            "supplier_id": supplier.supplier_id,
            "node_type": supplier.node_type.value,
            "operator_id": supplier.supplier_id,
            "operator_name": supplier.supplier_name,
            "country_code": supplier.country_code,
            "region": supplier.region,
            "commodities": [c.value for c in supplier.commodities],
            "tier_depth": tier_depth,
            "risk_score": 0.0,
            "risk_level": RiskLevel.STANDARD.value,
            "compliance_status": supplier.compliance_status.value,
            "certifications": supplier.certifications,
            "plot_ids": supplier.plot_ids,
            "source_type": supplier.source_type.value,
            "confidence_score": supplier.confidence_score,
            "metadata": supplier.raw_metadata,
        }

        if supplier.coordinates is not None:
            node_data["latitude"] = supplier.coordinates[0]
            node_data["longitude"] = supplier.coordinates[1]

        node_id = await self._graph_storage.add_node(graph_id, node_data)

        if _METRICS_AVAILABLE:
            _metrics_record_node_added(supplier.node_type.value)

        logger.debug(
            "Added node %s (%s) at tier %d in graph %s",
            supplier.supplier_name,
            supplier.node_type.value,
            tier_depth,
            graph_id,
        )

        return node_id

    async def _add_supply_edge(
        self,
        graph_id: str,
        source_node_id: str,
        target_node_id: str,
        supplier: SupplierRecord,
        commodity: EUDRCommodity,
    ) -> str:
        """
        Add an edge (supply relationship) between two nodes.

        The edge direction is upstream-to-downstream:
        source (supplier) -> target (buyer).

        Args:
            graph_id: Target graph ID
            source_node_id: Upstream supplier node ID
            target_node_id: Downstream buyer node ID
            supplier: Supplier record providing edge metadata
            commodity: Commodity flowing through this edge

        Returns:
            Edge ID of the added edge
        """
        provenance_str = (
            f"{graph_id}:{source_node_id}:{target_node_id}:"
            f"{commodity.value}:{supplier.supplier_id}"
        )
        edge_hash = hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()

        edge_data = {
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "commodity": commodity.value,
            "product_description": f"{commodity.value} supply from {supplier.supplier_name}",
            "quantity": 0,  # Quantity populated from custody records later
            "unit": "kg",
            "custody_model": "segregated",
            "provenance_hash": edge_hash,
        }

        edge_id = await self._graph_storage.add_edge(graph_id, edge_data)

        if _METRICS_AVAILABLE:
            _metrics_record_edge_added()

        return edge_id

    # -------------------------------------------------------------------------
    # GRAPH QUERIES
    # -------------------------------------------------------------------------

    async def _get_frontier_nodes(
        self, graph_id: str, tier_depth: int
    ) -> List[Dict[str, Any]]:
        """
        Get frontier nodes at the specified tier depth.

        Frontier nodes are nodes that have no upstream suppliers (i.e., they
        are at the current boundary of the mapped supply chain). These are
        the nodes whose sub-tier suppliers need to be discovered next.

        Args:
            graph_id: Supply chain graph ID
            tier_depth: Tier depth to query

        Returns:
            List of node dictionaries at the frontier
        """
        nodes_at_tier = await self._graph_storage.get_nodes_at_tier(
            graph_id, tier_depth
        )

        frontier = []
        for node in nodes_at_tier:
            node_id = node.get("node_id", "")
            children = await self._graph_storage.get_children(
                graph_id, node_id
            )
            # A frontier node has no children (upstream suppliers)
            if not children:
                frontier.append(node)

        return frontier

    async def _load_existing_supplier_ids(
        self, graph_id: str
    ) -> Set[str]:
        """
        Load all existing supplier IDs from the graph for deduplication.

        Args:
            graph_id: Supply chain graph ID

        Returns:
            Set of supplier IDs already in the graph
        """
        visited: Set[str] = set()
        try:
            consecutive_empty = 0
            # Query all tiers to collect supplier IDs
            for tier in range(0, 20):  # Reasonable max depth
                nodes = await self._graph_storage.get_nodes_at_tier(
                    graph_id, tier
                )
                if not nodes:
                    consecutive_empty += 1
                    # Stop after 3 consecutive empty tiers to handle
                    # sparse graphs where tier 0 (importer) may not
                    # be stored as a node
                    if consecutive_empty >= 3:
                        break
                    continue
                consecutive_empty = 0
                for node in nodes:
                    sid = node.get("supplier_id", node.get("node_id", ""))
                    if sid:
                        visited.add(sid)
        except Exception as exc:
            logger.warning(
                "Failed to load existing supplier IDs for graph %s: %s",
                graph_id,
                exc,
            )
        return visited

    # -------------------------------------------------------------------------
    # COMPLETENESS CALCULATIONS (ZERO-HALLUCINATION / DETERMINISTIC)
    # -------------------------------------------------------------------------

    def _estimate_expected_suppliers(
        self,
        archetype: CommodityArchetype,
        tier_depth: int,
        discovered_count: int,
    ) -> int:
        """
        Estimate the expected number of suppliers at a given tier depth.

        Uses the commodity archetype's typical structure. For deeper tiers,
        applies a fan-out multiplier based on the commodity type.

        This is a DETERMINISTIC calculation -- same inputs always produce
        the same output. No LLM involvement.

        Args:
            archetype: Commodity archetype configuration
            tier_depth: Tier depth being estimated
            discovered_count: Number of suppliers actually discovered

        Returns:
            Estimated expected supplier count at this tier
        """
        # Base estimate: at least what was discovered
        base = max(discovered_count, 1)

        # Apply commodity-specific fan-out multiplier
        # Deeper tiers tend to have more actors (many-to-one aggregation)
        fan_out_multipliers: Dict[EUDRCommodity, float] = {
            EUDRCommodity.CATTLE: 1.5,
            EUDRCommodity.COCOA: 3.0,   # Many smallholders per cooperative
            EUDRCommodity.COFFEE: 2.0,
            EUDRCommodity.PALM_OIL: 2.5,
            EUDRCommodity.RUBBER: 2.0,
            EUDRCommodity.SOYA: 1.5,
            EUDRCommodity.WOOD: 1.8,
        }
        multiplier = fan_out_multipliers.get(archetype.commodity, 2.0)

        # For Tier 1, the discovered count IS the expected count
        if tier_depth <= 1:
            return max(discovered_count, 1)

        # For deeper tiers, estimate based on parent tier and multiplier
        # The expectation grows with depth for commodities with many producers
        if discovered_count > 0:
            return max(
                discovered_count,
                int(discovered_count * multiplier * 0.5),
            )

        return max(1, int(base * multiplier))

    def _compute_tier_completeness(
        self, discovered: int, expected: int
    ) -> float:
        """
        Compute completeness percentage for a tier.

        Deterministic calculation: (discovered / expected) * 100,
        capped at 100%.

        Args:
            discovered: Number of suppliers discovered
            expected: Number of suppliers expected

        Returns:
            Completeness percentage (0.0 - 100.0)
        """
        if expected <= 0:
            return 100.0 if discovered > 0 else 0.0
        return min(100.0, round((discovered / expected) * 100.0, 1))

    def _calculate_overall_completeness(
        self, tier_results: List[TierMappingResult]
    ) -> float:
        """
        Calculate overall supply chain mapping completeness.

        Weighted average across all tiers, with deeper tiers weighted
        more heavily (they are harder to map and more important for
        EUDR compliance).

        Args:
            tier_results: List of per-tier mapping results

        Returns:
            Overall completeness percentage (0.0 - 100.0)
        """
        if not tier_results:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for result in tier_results:
            # Deeper tiers get higher weight
            weight = 1.0 + (result.tier_depth * 0.5)
            weighted_sum += result.completeness_pct * weight
            total_weight += weight

        if total_weight <= 0:
            return 0.0

        return min(100.0, round(weighted_sum / total_weight, 1))

    def _compute_average_confidence(
        self, suppliers: List[SupplierRecord]
    ) -> float:
        """
        Compute average confidence score across a list of suppliers.

        Args:
            suppliers: List of supplier records

        Returns:
            Average confidence (0.0 - 1.0)
        """
        if not suppliers:
            return 0.0
        total = sum(s.confidence_score for s in suppliers)
        return round(total / len(suppliers), 3)

    def _check_reached_plot_level(
        self, tier_results: List[TierMappingResult]
    ) -> bool:
        """
        Check if mapping reached production plot level (producers found).

        Args:
            tier_results: List of per-tier mapping results

        Returns:
            True if at least one producer node was found
        """
        for result in tier_results:
            if NodeType.PRODUCER.value in result.node_types_found:
                return True
        return False

    # -------------------------------------------------------------------------
    # DEPTH DISTRIBUTION
    # -------------------------------------------------------------------------

    async def _build_depth_distribution(
        self,
        graph_id: str,
        commodity: EUDRCommodity,
        target_depth: int,
    ) -> TierDepthDistribution:
        """
        Build a tier-depth distribution report from the graph.

        Queries all distinct root-to-leaf paths in the graph and computes
        the depth histogram, median, mean, and compliance percentage.

        Args:
            graph_id: Supply chain graph ID
            commodity: Commodity type
            target_depth: Target depth for compliance measurement

        Returns:
            TierDepthDistribution report
        """
        try:
            chain_depths = await self._graph_storage.get_all_chain_depths(
                graph_id
            )
        except Exception as exc:
            logger.warning(
                "Failed to get chain depths for graph %s: %s",
                graph_id,
                exc,
            )
            chain_depths = []

        if not chain_depths:
            return TierDepthDistribution(
                commodity=commodity,
                total_chains=0,
                target_depth=target_depth,
            )

        # Build histogram
        histogram: Dict[int, int] = defaultdict(int)
        for depth in chain_depths:
            histogram[depth] += 1

        # Statistics
        sorted_depths = sorted(chain_depths)
        total_chains = len(sorted_depths)
        mean_depth = sum(sorted_depths) / total_chains

        # Median
        mid = total_chains // 2
        if total_chains % 2 == 0:
            median_depth = (sorted_depths[mid - 1] + sorted_depths[mid]) / 2.0
        else:
            median_depth = float(sorted_depths[mid])

        # Percentage at or above target depth
        at_target = sum(1 for d in sorted_depths if d >= target_depth)
        pct_at_target = round((at_target / total_chains) * 100.0, 1)

        return TierDepthDistribution(
            commodity=commodity,
            total_chains=total_chains,
            depth_histogram=dict(histogram),
            median_depth=round(median_depth, 1),
            mean_depth=round(mean_depth, 1),
            max_depth=max(sorted_depths),
            min_depth=min(sorted_depths),
            pct_at_target_depth=pct_at_target,
            target_depth=target_depth,
        )

    # -------------------------------------------------------------------------
    # PROVENANCE (ZERO-HALLUCINATION)
    # -------------------------------------------------------------------------

    def _compute_provenance_hash(
        self,
        input_data: MultiTierMappingInput,
        tier_results: List[TierMappingResult],
        total_nodes: int,
        total_edges: int,
    ) -> str:
        """
        Compute SHA-256 provenance hash for the mapping operation.

        Ensures deterministic, reproducible hashing of all inputs and
        outputs for regulatory audit trail compliance (EUDR Article 31).

        Args:
            input_data: Mapping input parameters
            tier_results: Per-tier mapping results
            total_nodes: Total nodes added
            total_edges: Total edges added

        Returns:
            SHA-256 hex digest string
        """
        provenance_payload = {
            "graph_id": input_data.graph_id,
            "operator_id": input_data.operator_id,
            "commodity": input_data.commodity.value,
            "max_depth": input_data.max_depth,
            "target_depth": input_data.target_depth,
            "incremental": input_data.incremental,
            "tier_count": len(tier_results),
            "tier_depths": [t.tier_depth for t in tier_results],
            "tier_discovered": [t.suppliers_discovered for t in tier_results],
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        }

        payload_str = json.dumps(provenance_payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _generate_segment_id(
        self, graph_id: str, node_id: str
    ) -> str:
        """
        Generate a deterministic opaque segment ID.

        Args:
            graph_id: Supply chain graph ID
            node_id: Node ID where opacity begins

        Returns:
            Deterministic segment ID string
        """
        content = f"opaque:{graph_id}:{node_id}"
        hash_hex = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return f"opaque_{hash_hex}"

    async def _safe_get_node_count(self, graph_id: str) -> int:
        """Get node count with error handling."""
        try:
            return await self._graph_storage.get_node_count(graph_id)
        except Exception:
            return 0

    async def _safe_update_graph_metadata(
        self,
        graph_id: str,
        total_nodes: int,
        max_tier_depth: int,
        completeness: float,
    ) -> None:
        """Update graph metadata with error handling."""
        try:
            await self._graph_storage.update_graph_metadata(
                graph_id,
                {
                    "total_nodes": total_nodes,
                    "max_tier_depth": max_tier_depth,
                    "compliance_readiness": completeness,
                },
            )
        except Exception as exc:
            logger.warning(
                "Failed to update graph metadata for %s: %s",
                graph_id,
                exc,
            )

    # -------------------------------------------------------------------------
    # PROMETHEUS METRICS RECORDING
    # -------------------------------------------------------------------------

    def _record_metrics(
        self,
        commodity: EUDRCommodity,
        status: DiscoveryStatus,
        tiers_mapped: int,
        elapsed_seconds: float,
        nodes_added: int,
        edges_added: int,
        opaque_segments: List[OpaqueSegment],
        graph_id: str,
        tier_results: List[TierMappingResult],
    ) -> None:
        """
        Record Prometheus metrics for the mapping operation.

        Args:
            commodity: Commodity mapped
            status: Discovery status
            tiers_mapped: Number of tiers mapped
            elapsed_seconds: Total elapsed time in seconds
            nodes_added: Nodes added to graph
            edges_added: Edges added to graph
            opaque_segments: Detected opaque segments
            graph_id: Graph ID for labeling
            tier_results: Per-tier results for completeness gauges
        """
        if not _METRICS_AVAILABLE:
            return

        try:
            # Record tier discovery counter
            _metrics_record_tier_discovery()

            # Record mapping duration via canonical histogram
            _metrics_observe_duration("tier_discover", elapsed_seconds)

            # Log opaque segment details (no dedicated Prometheus metric
            # in the canonical metrics.py; structured logging is sufficient)
            if opaque_segments:
                logger.info(
                    "Opaque segments detected: graph=%s commodity=%s count=%d reasons=%s",
                    graph_id,
                    commodity.value,
                    len(opaque_segments),
                    [s.reason.value for s in opaque_segments],
                )

            # Log per-tier completeness (structured logging; fine-grained
            # per-tier gauges are tracked at the application layer)
            for tier_result in tier_results:
                logger.info(
                    "Tier completeness: graph=%s commodity=%s tier=%d "
                    "completeness=%.1f%% suppliers=%d",
                    graph_id,
                    commodity.value,
                    tier_result.tier_depth,
                    tier_result.completeness_pct,
                    tier_result.suppliers_found,
                )

        except Exception as exc:
            logger.debug("Failed to record Prometheus metrics: %s", exc)
