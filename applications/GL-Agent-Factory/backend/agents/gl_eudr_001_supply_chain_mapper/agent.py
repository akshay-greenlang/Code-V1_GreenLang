"""
GL-EUDR-001: Supply Chain Mapper Agent

This module implements the Supply Chain Mapper Agent for EUDR compliance,
providing complete traceability from production plots to EU market entry.

The agent supports:
- Multi-tier supply chain mapping (5-10+ tiers)
- Hybrid entity resolution (deterministic rules + ML)
- Coverage calculation with risk-weighted gates
- PostgreSQL primary storage with Neo4j graph cache
- Temporal tracking with as-of queries and snapshots
- LLM integration for entity extraction and NL queries

Regulatory Reference:
    EU Regulation 2023/1115 (EUDR)
    Enforcement Date: December 30, 2024 (Large Operators)
    SME Enforcement Date: June 30, 2025

Example:
    >>> agent = SupplyChainMapperAgent()
    >>> result = agent.run(SupplyChainMapperInput(
    ...     importer_id="uuid-here",
    ...     commodity=CommodityType.COFFEE,
    ...     operation=OperationType.MAP_SUPPLY_CHAIN
    ... ))
    >>> print(f"Nodes: {result.node_count}, Coverage: {result.coverage_percentage}%")
"""

import hashlib
import json
import logging
import re
import time
import uuid
from datetime import date, datetime
from decimal import Decimal
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class CommodityType(str, Enum):
    """EUDR regulated commodity categories."""
    CATTLE = "CATTLE"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    PALM_OIL = "PALM_OIL"
    RUBBER = "RUBBER"
    SOY = "SOY"
    WOOD = "WOOD"


class NodeType(str, Enum):
    """Types of supply chain nodes."""
    PRODUCER = "PRODUCER"
    PROCESSOR = "PROCESSOR"
    TRADER = "TRADER"
    IMPORTER = "IMPORTER"
    AGGREGATOR = "AGGREGATOR"


class EdgeType(str, Enum):
    """Types of supply chain relationships."""
    SUPPLIES = "SUPPLIES"
    PROCESSES = "PROCESSES"
    TRADES = "TRADES"
    IMPORTS = "IMPORTS"
    AGGREGATES = "AGGREGATES"


class DataSource(str, Enum):
    """Data provenance for edges."""
    SUPPLIER_DECLARED = "SUPPLIER_DECLARED"
    INFERRED_CUSTOMS = "INFERRED_CUSTOMS"
    INFERRED_SHIPPING = "INFERRED_SHIPPING"
    INFERRED_CERTIFICATION = "INFERRED_CERTIFICATION"
    INFERRED_SATELLITE = "INFERRED_SATELLITE"
    THIRD_PARTY_DATA = "THIRD_PARTY_DATA"


class VerificationStatus(str, Enum):
    """Verification status for nodes."""
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    PENDING = "PENDING"


class DisclosureStatus(str, Enum):
    """Disclosure status for suppliers."""
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    NONE = "NONE"


class OperatorSize(str, Enum):
    """EUDR operator classification."""
    LARGE = "LARGE"
    SME = "SME"


class GapType(str, Enum):
    """Types of supply chain gaps."""
    UNVERIFIED_SUPPLIER = "UNVERIFIED_SUPPLIER"
    MISSING_PLOT_DATA = "MISSING_PLOT_DATA"
    MISSING_COORDINATES = "MISSING_COORDINATES"
    PARTIAL_DISCLOSURE = "PARTIAL_DISCLOSURE"
    EXPIRED_CERTIFICATION = "EXPIRED_CERTIFICATION"
    CYCLE_DETECTED = "CYCLE_DETECTED"
    MISSING_TIER_DATA = "MISSING_TIER_DATA"
    UNVERIFIED_TRANSFORMATION = "UNVERIFIED_TRANSFORMATION"


class GapSeverity(str, Enum):
    """Severity levels for gaps."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ResolutionStatus(str, Enum):
    """Entity resolution candidate status."""
    PENDING = "PENDING"
    AUTO_MERGED = "AUTO_MERGED"
    REVIEWED_MERGE = "REVIEWED_MERGE"
    REVIEWED_NO_MERGE = "REVIEWED_NO_MERGE"


class ResolutionDecision(str, Enum):
    """Entity resolution decisions."""
    AUTO_MERGE = "AUTO_MERGE"
    REVIEW = "REVIEW"
    NO_MERGE = "NO_MERGE"


class SnapshotTrigger(str, Enum):
    """Triggers for snapshot creation."""
    SCHEDULED = "SCHEDULED"
    DDS_SUBMISSION = "DDS_SUBMISSION"
    COVERAGE_DROP = "COVERAGE_DROP"
    MANUAL = "MANUAL"


class RiskLevel(str, Enum):
    """Risk levels for coverage thresholds."""
    LOW = "LOW"
    STANDARD = "STANDARD"
    HIGH = "HIGH"


class OperationType(str, Enum):
    """Agent operation types."""
    MAP_SUPPLY_CHAIN = "MAP_SUPPLY_CHAIN"
    CALCULATE_COVERAGE = "CALCULATE_COVERAGE"
    RUN_ENTITY_RESOLUTION = "RUN_ENTITY_RESOLUTION"
    CREATE_SNAPSHOT = "CREATE_SNAPSHOT"
    QUERY_GRAPH = "QUERY_GRAPH"
    CHECK_GATES = "CHECK_GATES"
    NATURAL_LANGUAGE_QUERY = "NATURAL_LANGUAGE_QUERY"


# =============================================================================
# DATA MODELS - Address & Geolocation
# =============================================================================

class Address(BaseModel):
    """Structured address for supply chain nodes."""
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: str

    class Config:
        frozen = True


class GeoCoordinate(BaseModel):
    """Geographic coordinate (WGS-84)."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

    @validator('latitude', 'longitude')
    def validate_precision(cls, v):
        """Ensure 6 decimal precision for EUDR compliance."""
        return round(v, 6)


class PlotGeometry(BaseModel):
    """Geolocation for origin plots - supports point and polygon."""
    type: str = Field(..., pattern=r"^(Point|Polygon)$")
    coordinates: Union[List[float], List[List[List[float]]]]

    @validator('coordinates')
    def validate_coordinates(cls, v, values):
        if values.get('type') == 'Point':
            if len(v) != 2:
                raise ValueError("Point requires [longitude, latitude]")
        return v


# =============================================================================
# DATA MODELS - Supply Chain Nodes
# =============================================================================

class Certification(BaseModel):
    """Certification held by a supplier."""
    name: str
    issuer: str
    certificate_id: Optional[str] = None
    valid_from: date
    valid_to: Optional[date] = None
    commodities: List[CommodityType] = []

    @property
    def is_valid(self) -> bool:
        if self.valid_to is None:
            return True
        return self.valid_to >= date.today()


class SupplyChainNode(BaseModel):
    """A node in the supply chain graph (supplier, processor, etc.)."""
    node_id: UUID = Field(default_factory=uuid.uuid4)
    node_type: NodeType
    name: str = Field(..., min_length=1, max_length=500)
    country_code: str = Field(..., pattern=r"^[A-Z]{2}$")
    address: Optional[Address] = None

    # Identifiers
    tax_id: Optional[str] = None
    duns_number: Optional[str] = Field(None, pattern=r"^[0-9]{9}$")
    eori_number: Optional[str] = Field(None, max_length=17)

    # Certifications
    certifications: List[Certification] = []
    commodities: List[CommodityType]

    # Tier information (multi-tier support)
    tier: Optional[int] = Field(None, ge=0, description="tier_min - shortest path to importer")
    tier_max: Optional[int] = Field(None, ge=0)
    all_tiers: List[int] = Field(default_factory=list)

    # Risk and verification
    risk_score: Optional[Decimal] = Field(None, ge=0, le=1)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    disclosure_status: DisclosureStatus = DisclosureStatus.FULL

    # Operator classification
    operator_size: Optional[OperatorSize] = None

    # Golden record tracking
    field_sources: Dict[str, str] = Field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


# =============================================================================
# DATA MODELS - Supply Chain Edges
# =============================================================================

class EdgeContext(BaseModel):
    """Context for multi-tier edges."""
    is_direct_to_importer: bool = False
    observed_tier: Optional[int] = None
    relationship_path: List[str] = []


class InferenceEvidence(BaseModel):
    """Evidence for inferred edges."""
    source_type: str
    source_id: Optional[str] = None
    confidence: float = Field(..., ge=0, le=1)
    extracted_data: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SupplyChainEdge(BaseModel):
    """An edge (relationship) in the supply chain graph."""
    edge_id: UUID = Field(default_factory=uuid.uuid4)
    source_node_id: UUID
    target_node_id: UUID
    edge_type: EdgeType
    commodity: CommodityType

    # Transaction details
    quantity: Optional[Decimal] = Field(None, ge=0)
    quantity_unit: Optional[str] = None
    transaction_date: Optional[date] = None
    documents: List[Dict[str, Any]] = Field(default_factory=list)

    # Verification
    verified: bool = False
    confidence_score: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)

    # Data provenance (ENHANCED)
    data_source: DataSource = DataSource.SUPPLIER_DECLARED
    inference_method: Optional[str] = None
    inference_evidence: List[InferenceEvidence] = Field(default_factory=list)

    # Multi-tier context
    edge_context: EdgeContext = Field(default_factory=EdgeContext)

    # DDS eligibility
    dds_eligible: bool = True

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


# =============================================================================
# DATA MODELS - Origin Plots
# =============================================================================

class OriginPlot(BaseModel):
    """Production plot where commodity originates."""
    plot_id: UUID = Field(default_factory=uuid.uuid4)
    producer_node_id: UUID
    plot_identifier: Optional[str] = None
    geometry: PlotGeometry
    area_hectares: Optional[Decimal] = Field(None, gt=0)
    commodity: CommodityType
    country_code: str = Field(..., pattern=r"^[A-Z]{2}$")

    # Validation and risk
    validation_status: str = "PENDING"
    deforestation_risk_score: Optional[Decimal] = Field(None, ge=0, le=1)
    last_assessment_date: Optional[date] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class PlotProducer(BaseModel):
    """Producer relationship to plot (for cooperatives/shared plots)."""
    id: UUID = Field(default_factory=uuid.uuid4)
    plot_id: UUID
    producer_node_id: UUID
    share_percentage: Optional[Decimal] = Field(None, ge=0, le=100)
    tenure_type: Optional[str] = None  # OWNER, LEASE, COMMUNITY
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# DATA MODELS - Gaps & Coverage
# =============================================================================

class SupplyChainGap(BaseModel):
    """A gap identified in supply chain coverage."""
    gap_id: UUID = Field(default_factory=uuid.uuid4)
    snapshot_id: Optional[UUID] = None
    node_id: Optional[UUID] = None
    gap_type: GapType
    severity: GapSeverity
    description: str
    remediation_suggestion: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class CoverageThresholds(BaseModel):
    """Coverage thresholds by risk level."""
    mapping_completeness: float
    plot_coverage: float


COVERAGE_GATES = {
    RiskLevel.LOW: CoverageThresholds(mapping_completeness=0.90, plot_coverage=0.85),
    RiskLevel.STANDARD: CoverageThresholds(mapping_completeness=0.95, plot_coverage=0.90),
    RiskLevel.HIGH: CoverageThresholds(mapping_completeness=0.98, plot_coverage=0.95),
}


class CoverageReport(BaseModel):
    """Coverage analysis report."""
    importer_id: UUID
    commodity: CommodityType
    overall_coverage: float = Field(..., ge=0, le=100)
    mapping_completeness: float = Field(..., ge=0, le=100)
    plot_coverage: float = Field(..., ge=0, le=100)
    tier_coverage: Dict[int, float] = Field(default_factory=dict)
    volume_coverage: Optional[float] = None
    gaps: List[SupplyChainGap] = Field(default_factory=list)
    gap_summary: Dict[str, int] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class CoverageGateResult(BaseModel):
    """Result of coverage gate checks."""
    can_proceed_to_risk_assessment: bool
    can_submit_dds: bool
    blocking_gaps: List[SupplyChainGap] = []
    risk_level_applied: RiskLevel
    mapping_completeness: float
    plot_coverage: float
    required_mapping: float
    required_plot: float

    class Config:
        use_enum_values = True


# =============================================================================
# DATA MODELS - Snapshots
# =============================================================================

class SupplyChainSnapshot(BaseModel):
    """Immutable snapshot of supply chain state."""
    snapshot_id: UUID = Field(default_factory=uuid.uuid4)
    snapshot_date: datetime = Field(default_factory=datetime.utcnow)
    importer_node_id: UUID
    commodity: CommodityType
    graph_hash: str
    node_count: int
    edge_count: int
    plot_count: int
    coverage_percentage: float
    mapping_completeness: float
    plot_coverage: float
    snapshot_data: Dict[str, Any]
    trigger_type: SnapshotTrigger = SnapshotTrigger.MANUAL
    created_by: Optional[str] = None

    class Config:
        use_enum_values = True


class SnapshotDiff(BaseModel):
    """Diff between two snapshots."""
    base_snapshot_id: UUID
    compare_snapshot_id: UUID
    nodes_added: List[UUID] = []
    nodes_removed: List[UUID] = []
    nodes_modified: List[Dict[str, Any]] = []
    edges_added: List[UUID] = []
    edges_removed: List[UUID] = []
    coverage_change: Dict[str, float] = {}


# =============================================================================
# DATA MODELS - Entity Resolution
# =============================================================================

class MatchFeature(BaseModel):
    """A feature matched during entity resolution."""
    feature_name: str
    score: float = Field(..., ge=0, le=1)
    weight: float = Field(..., ge=0, le=1)
    is_strong: bool = False
    evidence: Optional[str] = None


class EntityResolutionCandidate(BaseModel):
    """Candidate pair for entity resolution."""
    candidate_id: UUID = Field(default_factory=uuid.uuid4)
    node_a_id: UUID
    node_b_id: UUID
    similarity_score: float = Field(..., ge=0, le=1)
    matching_features: List[MatchFeature] = []
    resolution_status: ResolutionStatus = ResolutionStatus.PENDING
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class MatchResult(BaseModel):
    """Result of matching two entities."""
    node_a_id: UUID
    node_b_id: UUID
    overall_score: float
    features: List[MatchFeature]
    decision: ResolutionDecision
    strong_feature_matched: bool

    class Config:
        use_enum_values = True


# =============================================================================
# DATA MODELS - Graph
# =============================================================================

class SupplyChainGraph(BaseModel):
    """Complete supply chain graph representation."""
    nodes: List[SupplyChainNode] = []
    edges: List[SupplyChainEdge] = []
    plots: List[OriginPlot] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def max_tier(self) -> int:
        if not self.nodes:
            return 0
        tiers = [n.tier for n in self.nodes if n.tier is not None]
        return max(tiers) if tiers else 0

    @property
    def has_cycles(self) -> bool:
        return self.metadata.get('has_cycles', False)

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of graph for provenance."""
        content = json.dumps({
            'nodes': [n.node_id.hex for n in sorted(self.nodes, key=lambda x: str(x.node_id))],
            'edges': [e.edge_id.hex for e in sorted(self.edges, key=lambda x: str(x.edge_id))]
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class SupplyChainMapperInput(BaseModel):
    """Input for Supply Chain Mapper Agent."""
    importer_id: UUID
    commodity: CommodityType
    operation: OperationType

    # Optional parameters by operation
    depth: int = Field(default=10, ge=1, le=20)
    include_inferred: bool = True
    risk_level: RiskLevel = RiskLevel.STANDARD

    # For entity resolution
    node_ids: Optional[List[UUID]] = None
    scope: str = "ALL"

    # For snapshots
    trigger_type: SnapshotTrigger = SnapshotTrigger.MANUAL
    snapshot_id: Optional[UUID] = None
    compare_snapshot_id: Optional[UUID] = None
    as_of: Optional[datetime] = None

    # For natural language queries
    query: Optional[str] = None

    class Config:
        use_enum_values = True


class SupplyChainMapperOutput(BaseModel):
    """Output from Supply Chain Mapper Agent."""
    success: bool
    operation: OperationType

    # Graph results
    graph: Optional[SupplyChainGraph] = None
    node_count: int = 0
    edge_count: int = 0

    # Coverage results
    coverage_report: Optional[CoverageReport] = None
    gate_result: Optional[CoverageGateResult] = None

    # Snapshot results
    snapshot: Optional[SupplyChainSnapshot] = None
    snapshot_diff: Optional[SnapshotDiff] = None

    # Entity resolution results
    resolution_candidates: List[EntityResolutionCandidate] = []
    auto_merged_count: int = 0
    review_queue_count: int = 0

    # NL query results
    nl_interpreted_query: Optional[str] = None
    nl_generated_filter: Optional[Dict[str, Any]] = None
    nl_results: List[SupplyChainNode] = []

    # Metadata
    processing_time_ms: int = 0
    provenance_hash: Optional[str] = None
    errors: List[str] = []
    warnings: List[str] = []

    class Config:
        use_enum_values = True


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class SupplyChainMapperAgent:
    """
    GL-EUDR-001: Supply Chain Mapper Agent

    Maps multi-tier supply chains from production plots to EU market entry
    for EUDR compliance. Provides entity resolution, coverage analysis,
    and temporal tracking with snapshot support.
    """

    # Entity resolution thresholds
    AUTO_MERGE_THRESHOLD = 0.98
    REVIEW_THRESHOLD = 0.85

    # Strong features for entity resolution
    STRONG_FEATURES = {
        "tax_id_match",
        "duns_match",
        "eori_match",
        "exact_address_country_match",
        "email_domain_match"
    }

    # Feature weights
    FEATURE_WEIGHTS = {
        "tax_id_match": 0.30,
        "duns_match": 0.25,
        "eori_match": 0.25,
        "name_similarity": 0.15,
        "address_similarity": 0.10,
        "country_match": 0.05,
        "phone_match": 0.05,
        "email_domain_match": 0.10,
        "certification_overlap": 0.05
    }

    def __init__(
        self,
        db_session: Optional[Any] = None,
        neo4j_driver: Optional[Any] = None,
        llm_service: Optional[Any] = None,
        embedding_service: Optional[Any] = None
    ) -> None:
        self.db = db_session
        self.neo4j = neo4j_driver
        self.llm = llm_service
        self.embeddings = embedding_service

        # In-memory storage for testing
        self._nodes: Dict[UUID, SupplyChainNode] = {}
        self._edges: Dict[UUID, SupplyChainEdge] = {}
        self._plots: Dict[UUID, OriginPlot] = {}
        self._snapshots: Dict[UUID, SupplyChainSnapshot] = {}
        self._resolution_candidates: Dict[UUID, EntityResolutionCandidate] = {}

    def run(self, input_data: SupplyChainMapperInput) -> SupplyChainMapperOutput:
        """Execute the requested operation."""
        start_time = time.time()

        try:
            if input_data.operation == OperationType.MAP_SUPPLY_CHAIN:
                result = self._map_supply_chain(input_data)
            elif input_data.operation == OperationType.CALCULATE_COVERAGE:
                result = self._calculate_coverage(input_data)
            elif input_data.operation == OperationType.RUN_ENTITY_RESOLUTION:
                result = self._run_entity_resolution(input_data)
            elif input_data.operation == OperationType.CREATE_SNAPSHOT:
                result = self._create_snapshot(input_data)
            elif input_data.operation == OperationType.QUERY_GRAPH:
                result = self._query_graph(input_data)
            elif input_data.operation == OperationType.CHECK_GATES:
                result = self._check_gates(input_data)
            elif input_data.operation == OperationType.NATURAL_LANGUAGE_QUERY:
                result = self._natural_language_query(input_data)
            else:
                result = SupplyChainMapperOutput(
                    success=False,
                    operation=input_data.operation,
                    errors=[f"Unknown operation: {input_data.operation}"]
                )

            result.processing_time_ms = int((time.time() - start_time) * 1000)
            return result

        except Exception as e:
            logger.exception(f"Error in operation {input_data.operation}")
            return SupplyChainMapperOutput(
                success=False,
                operation=input_data.operation,
                errors=[str(e)],
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

    # =========================================================================
    # SUPPLY CHAIN MAPPING
    # =========================================================================

    def _map_supply_chain(
        self,
        input_data: SupplyChainMapperInput
    ) -> SupplyChainMapperOutput:
        """Build supply chain graph for importer/commodity."""
        graph = self._build_graph(
            importer_id=input_data.importer_id,
            commodity=input_data.commodity,
            depth=input_data.depth,
            include_inferred=input_data.include_inferred
        )

        # Detect cycles
        cycles = self._detect_cycles(graph)
        graph.metadata['has_cycles'] = len(cycles) > 0
        graph.metadata['cycles'] = cycles

        # Calculate tiers
        self._calculate_tiers(graph, input_data.importer_id)

        return SupplyChainMapperOutput(
            success=True,
            operation=input_data.operation,
            graph=graph,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            provenance_hash=graph.compute_hash()
        )

    def _build_graph(
        self,
        importer_id: UUID,
        commodity: CommodityType,
        depth: int,
        include_inferred: bool
    ) -> SupplyChainGraph:
        """Build graph using BFS from importer."""
        visited_nodes = set()
        visited_edges = set()
        queue = [(importer_id, 0)]

        nodes = []
        edges = []
        plots = []

        while queue:
            node_id, current_depth = queue.pop(0)

            if node_id in visited_nodes or current_depth > depth:
                continue

            visited_nodes.add(node_id)

            # Get node
            node = self._get_node(node_id)
            if node and commodity in node.commodities:
                nodes.append(node)

                # Get plots for producers
                if node.node_type == NodeType.PRODUCER:
                    node_plots = self._get_plots_for_producer(node_id)
                    plots.extend(node_plots)

                # Get incoming edges
                incoming = self._get_incoming_edges(
                    node_id, commodity, include_inferred
                )
                for edge in incoming:
                    if edge.edge_id not in visited_edges:
                        visited_edges.add(edge.edge_id)
                        edges.append(edge)
                        queue.append((edge.source_node_id, current_depth + 1))

        return SupplyChainGraph(
            nodes=nodes,
            edges=edges,
            plots=plots,
            metadata={
                'importer_id': str(importer_id),
                'commodity': commodity.value,
                'depth': depth,
                'include_inferred': include_inferred
            }
        )

    def _detect_cycles(self, graph: SupplyChainGraph) -> List[List[UUID]]:
        """Detect cycles using DFS."""
        adj = {}
        for edge in graph.edges:
            if edge.source_node_id not in adj:
                adj[edge.source_node_id] = []
            adj[edge.source_node_id].append(edge.target_node_id)

        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node_id: UUID):
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor in adj.get(node_id, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:])

            path.pop()
            rec_stack.remove(node_id)
            return None

        for node in graph.nodes:
            if node.node_id not in visited:
                dfs(node.node_id)

        return cycles

    def _calculate_tiers(self, graph: SupplyChainGraph, importer_id: UUID):
        """Calculate tier levels for all nodes (distance from importer)."""
        # Build reverse adjacency (target -> sources)
        reverse_adj = {}
        for edge in graph.edges:
            if edge.target_node_id not in reverse_adj:
                reverse_adj[edge.target_node_id] = []
            reverse_adj[edge.target_node_id].append(edge.source_node_id)

        # BFS from importer
        tiers = {importer_id: 0}
        all_tiers = {importer_id: [0]}
        queue = [importer_id]

        while queue:
            current = queue.pop(0)
            current_tier = tiers[current]

            for source in reverse_adj.get(current, []):
                next_tier = current_tier + 1
                if source not in tiers:
                    tiers[source] = next_tier
                    all_tiers[source] = [next_tier]
                    queue.append(source)
                else:
                    if next_tier not in all_tiers[source]:
                        all_tiers[source].append(next_tier)

        # Update nodes
        node_map = {n.node_id: n for n in graph.nodes}
        for node_id, tier in tiers.items():
            if node_id in node_map:
                node = node_map[node_id]
                node.tier = min(all_tiers[node_id])
                node.tier_max = max(all_tiers[node_id])
                node.all_tiers = sorted(all_tiers[node_id])

    # =========================================================================
    # COVERAGE CALCULATION
    # =========================================================================

    def _calculate_coverage(
        self,
        input_data: SupplyChainMapperInput
    ) -> SupplyChainMapperOutput:
        """Calculate coverage metrics."""
        graph = self._build_graph(
            importer_id=input_data.importer_id,
            commodity=input_data.commodity,
            depth=input_data.depth,
            include_inferred=input_data.include_inferred
        )

        # Calculate metrics
        total_nodes = len(graph.nodes)
        verified_nodes = sum(1 for n in graph.nodes if n.verification_status == VerificationStatus.VERIFIED)

        producers = [n for n in graph.nodes if n.node_type == NodeType.PRODUCER]
        producers_with_plots = sum(1 for p in producers if self._has_plots(p.node_id))

        mapping_completeness = (verified_nodes / total_nodes * 100) if total_nodes > 0 else 0
        plot_coverage = (producers_with_plots / len(producers) * 100) if producers else 100
        overall_coverage = (mapping_completeness + plot_coverage) / 2

        # Identify gaps
        gaps = self._identify_gaps(graph)

        # Gap summary
        gap_summary = {
            'critical': sum(1 for g in gaps if g.severity == GapSeverity.CRITICAL),
            'high': sum(1 for g in gaps if g.severity == GapSeverity.HIGH),
            'medium': sum(1 for g in gaps if g.severity == GapSeverity.MEDIUM),
            'low': sum(1 for g in gaps if g.severity == GapSeverity.LOW),
        }

        # Tier coverage
        tier_coverage = {}
        for tier in set(n.tier for n in graph.nodes if n.tier is not None):
            tier_nodes = [n for n in graph.nodes if n.tier == tier]
            verified = sum(1 for n in tier_nodes if n.verification_status == VerificationStatus.VERIFIED)
            tier_coverage[tier] = (verified / len(tier_nodes) * 100) if tier_nodes else 0

        report = CoverageReport(
            importer_id=input_data.importer_id,
            commodity=input_data.commodity,
            overall_coverage=overall_coverage,
            mapping_completeness=mapping_completeness,
            plot_coverage=plot_coverage,
            tier_coverage=tier_coverage,
            gaps=gaps,
            gap_summary=gap_summary
        )

        return SupplyChainMapperOutput(
            success=True,
            operation=input_data.operation,
            coverage_report=report,
            node_count=total_nodes,
            edge_count=len(graph.edges)
        )

    def _identify_gaps(self, graph: SupplyChainGraph) -> List[SupplyChainGap]:
        """Identify coverage gaps in the graph."""
        gaps = []

        for node in graph.nodes:
            # Unverified suppliers
            if node.verification_status == VerificationStatus.UNVERIFIED:
                severity = GapSeverity.HIGH if node.tier and node.tier <= 2 else GapSeverity.MEDIUM
                gaps.append(SupplyChainGap(
                    node_id=node.node_id,
                    gap_type=GapType.UNVERIFIED_SUPPLIER,
                    severity=severity,
                    description=f"Supplier {node.name} is not verified",
                    remediation_suggestion="Request verification documents from supplier"
                ))

            # Missing plot data for producers
            if node.node_type == NodeType.PRODUCER and not self._has_plots(node.node_id):
                gaps.append(SupplyChainGap(
                    node_id=node.node_id,
                    gap_type=GapType.MISSING_PLOT_DATA,
                    severity=GapSeverity.CRITICAL,
                    description=f"Producer {node.name} has no plot data",
                    remediation_suggestion="Collect geolocation data for production plots"
                ))

            # Partial disclosure
            if node.disclosure_status == DisclosureStatus.PARTIAL:
                gaps.append(SupplyChainGap(
                    node_id=node.node_id,
                    gap_type=GapType.PARTIAL_DISCLOSURE,
                    severity=GapSeverity.HIGH,
                    description=f"Supplier {node.name} has partial disclosure only",
                    remediation_suggestion="Engage supplier for full upstream disclosure"
                ))

            # Expired certifications
            for cert in node.certifications:
                if not cert.is_valid:
                    gaps.append(SupplyChainGap(
                        node_id=node.node_id,
                        gap_type=GapType.EXPIRED_CERTIFICATION,
                        severity=GapSeverity.MEDIUM,
                        description=f"Certification {cert.name} expired on {cert.valid_to}",
                        remediation_suggestion="Request renewed certification from supplier"
                    ))

        return gaps

    # =========================================================================
    # COVERAGE GATES
    # =========================================================================

    def _check_gates(
        self,
        input_data: SupplyChainMapperInput
    ) -> SupplyChainMapperOutput:
        """Check if coverage meets gates for risk assessment and DDS submission."""
        coverage_result = self._calculate_coverage(input_data)
        report = coverage_result.coverage_report

        thresholds = COVERAGE_GATES[input_data.risk_level]

        # Check mapping completeness
        mapping_ok = report.mapping_completeness >= (thresholds.mapping_completeness * 100)

        # Check plot coverage
        plot_ok = report.plot_coverage >= (thresholds.plot_coverage * 100)

        # Check for HIGH severity gaps blocking DDS
        high_severity_gaps = [g for g in report.gaps if g.severity in [GapSeverity.CRITICAL, GapSeverity.HIGH]]

        can_risk_assessment = mapping_ok and plot_ok
        can_dds = can_risk_assessment and len(high_severity_gaps) == 0

        gate_result = CoverageGateResult(
            can_proceed_to_risk_assessment=can_risk_assessment,
            can_submit_dds=can_dds,
            blocking_gaps=high_severity_gaps,
            risk_level_applied=input_data.risk_level,
            mapping_completeness=report.mapping_completeness,
            plot_coverage=report.plot_coverage,
            required_mapping=thresholds.mapping_completeness * 100,
            required_plot=thresholds.plot_coverage * 100
        )

        return SupplyChainMapperOutput(
            success=True,
            operation=input_data.operation,
            coverage_report=report,
            gate_result=gate_result
        )

    # =========================================================================
    # ENTITY RESOLUTION
    # =========================================================================

    def _run_entity_resolution(
        self,
        input_data: SupplyChainMapperInput
    ) -> SupplyChainMapperOutput:
        """Run entity resolution to find duplicate suppliers."""
        nodes_to_check = list(self._nodes.values())
        if input_data.node_ids:
            nodes_to_check = [self._nodes[nid] for nid in input_data.node_ids if nid in self._nodes]

        candidates = []
        auto_merged = 0
        review_queue = 0

        # Find and score candidates
        checked_pairs = set()
        for node in nodes_to_check:
            potential_matches = self._find_resolution_candidates(node)

            for match_id in potential_matches:
                pair_key = tuple(sorted([str(node.node_id), str(match_id)]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                result = self._score_entity_pair(node.node_id, match_id)

                if result.decision == ResolutionDecision.AUTO_MERGE:
                    self._merge_entities(node.node_id, match_id)
                    auto_merged += 1
                elif result.decision == ResolutionDecision.REVIEW:
                    candidate = EntityResolutionCandidate(
                        node_a_id=node.node_id,
                        node_b_id=match_id,
                        similarity_score=result.overall_score,
                        matching_features=result.features,
                        resolution_status=ResolutionStatus.PENDING
                    )
                    candidates.append(candidate)
                    self._resolution_candidates[candidate.candidate_id] = candidate
                    review_queue += 1

        return SupplyChainMapperOutput(
            success=True,
            operation=input_data.operation,
            resolution_candidates=candidates,
            auto_merged_count=auto_merged,
            review_queue_count=review_queue
        )

    def _find_resolution_candidates(self, node: SupplyChainNode) -> List[UUID]:
        """Find potential duplicates using blocking strategy."""
        candidates = set()

        # Block 1: Same country + name prefix
        name_prefix = self._normalize_name(node.name)[:3].upper()
        for other in self._nodes.values():
            if other.node_id == node.node_id:
                continue
            if other.country_code == node.country_code:
                other_prefix = self._normalize_name(other.name)[:3].upper()
                if name_prefix == other_prefix:
                    candidates.add(other.node_id)

        # Block 2: Same tax ID
        if node.tax_id:
            for other in self._nodes.values():
                if other.node_id != node.node_id and other.tax_id:
                    if self._normalize_tax_id(node.tax_id) == self._normalize_tax_id(other.tax_id):
                        candidates.add(other.node_id)

        # Block 3: Same DUNS/EORI
        if node.duns_number:
            for other in self._nodes.values():
                if other.duns_number == node.duns_number and other.node_id != node.node_id:
                    candidates.add(other.node_id)

        if node.eori_number:
            for other in self._nodes.values():
                if other.eori_number == node.eori_number and other.node_id != node.node_id:
                    candidates.add(other.node_id)

        return list(candidates)

    def _score_entity_pair(self, node_a_id: UUID, node_b_id: UUID) -> MatchResult:
        """Calculate similarity score between two nodes."""
        node_a = self._nodes[node_a_id]
        node_b = self._nodes[node_b_id]
        features = []

        # Tax ID match
        if node_a.tax_id and node_b.tax_id:
            if self._normalize_tax_id(node_a.tax_id) == self._normalize_tax_id(node_b.tax_id):
                features.append(MatchFeature(
                    feature_name="tax_id_match",
                    score=1.0,
                    weight=self.FEATURE_WEIGHTS["tax_id_match"],
                    is_strong=True,
                    evidence=f"Tax ID: {node_a.tax_id}"
                ))

        # DUNS match
        if node_a.duns_number and node_b.duns_number:
            if node_a.duns_number == node_b.duns_number:
                features.append(MatchFeature(
                    feature_name="duns_match",
                    score=1.0,
                    weight=self.FEATURE_WEIGHTS["duns_match"],
                    is_strong=True,
                    evidence=f"DUNS: {node_a.duns_number}"
                ))

        # EORI match
        if node_a.eori_number and node_b.eori_number:
            if node_a.eori_number == node_b.eori_number:
                features.append(MatchFeature(
                    feature_name="eori_match",
                    score=1.0,
                    weight=self.FEATURE_WEIGHTS["eori_match"],
                    is_strong=True,
                    evidence=f"EORI: {node_a.eori_number}"
                ))

        # Name similarity
        name_sim = self._string_similarity(
            self._normalize_name(node_a.name),
            self._normalize_name(node_b.name)
        )
        if name_sim > 0.7:
            features.append(MatchFeature(
                feature_name="name_similarity",
                score=name_sim,
                weight=self.FEATURE_WEIGHTS["name_similarity"],
                evidence=f"Names: {node_a.name} vs {node_b.name}"
            ))

        # Country match
        if node_a.country_code == node_b.country_code:
            features.append(MatchFeature(
                feature_name="country_match",
                score=1.0,
                weight=self.FEATURE_WEIGHTS["country_match"]
            ))

        # Calculate overall score
        if not features:
            return MatchResult(
                node_a_id=node_a_id,
                node_b_id=node_b_id,
                overall_score=0.0,
                features=[],
                decision=ResolutionDecision.NO_MERGE,
                strong_feature_matched=False
            )

        total_weight = sum(f.weight for f in features)
        weighted_score = sum(f.score * f.weight for f in features) / total_weight
        strong_matched = any(f.is_strong for f in features)

        # Decision based on thresholds
        if weighted_score >= self.AUTO_MERGE_THRESHOLD and strong_matched:
            decision = ResolutionDecision.AUTO_MERGE
        elif weighted_score >= self.REVIEW_THRESHOLD:
            decision = ResolutionDecision.REVIEW
        else:
            decision = ResolutionDecision.NO_MERGE

        return MatchResult(
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            overall_score=weighted_score,
            features=features,
            decision=decision,
            strong_feature_matched=strong_matched
        )

    def _merge_entities(self, keep_id: UUID, merge_id: UUID):
        """Merge two entities using field-level golden record strategy."""
        keep_node = self._nodes[keep_id]
        merge_node = self._nodes[merge_id]

        # Field-level merge - keep most complete/recent data
        for field in ['tax_id', 'duns_number', 'eori_number', 'address']:
            keep_val = getattr(keep_node, field)
            merge_val = getattr(merge_node, field)
            if merge_val and not keep_val:
                setattr(keep_node, field, merge_val)
                keep_node.field_sources[field] = 'MERGE'

        # Merge certifications
        keep_node.certifications.extend(merge_node.certifications)

        # Update all edges pointing to merge_node
        for edge in self._edges.values():
            if edge.source_node_id == merge_id:
                edge.source_node_id = keep_id
            if edge.target_node_id == merge_id:
                edge.target_node_id = keep_id

        # Remove merged node
        del self._nodes[merge_id]

        keep_node.updated_at = datetime.utcnow()

    # =========================================================================
    # SNAPSHOTS
    # =========================================================================

    def _create_snapshot(
        self,
        input_data: SupplyChainMapperInput
    ) -> SupplyChainMapperOutput:
        """Create immutable snapshot of current supply chain state."""
        graph = self._build_graph(
            importer_id=input_data.importer_id,
            commodity=input_data.commodity,
            depth=input_data.depth,
            include_inferred=input_data.include_inferred
        )

        coverage_result = self._calculate_coverage(input_data)
        report = coverage_result.coverage_report

        snapshot = SupplyChainSnapshot(
            importer_node_id=input_data.importer_id,
            commodity=input_data.commodity,
            graph_hash=graph.compute_hash(),
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            plot_count=len(graph.plots),
            coverage_percentage=report.overall_coverage,
            mapping_completeness=report.mapping_completeness,
            plot_coverage=report.plot_coverage,
            snapshot_data={
                'nodes': [n.dict() for n in graph.nodes],
                'edges': [e.dict() for e in graph.edges],
                'plots': [p.dict() for p in graph.plots]
            },
            trigger_type=input_data.trigger_type
        )

        self._snapshots[snapshot.snapshot_id] = snapshot

        return SupplyChainMapperOutput(
            success=True,
            operation=input_data.operation,
            snapshot=snapshot,
            provenance_hash=snapshot.graph_hash
        )

    def _query_graph(
        self,
        input_data: SupplyChainMapperInput
    ) -> SupplyChainMapperOutput:
        """Query graph, optionally with as-of timestamp."""
        if input_data.as_of and input_data.snapshot_id:
            # Compare snapshots
            if input_data.compare_snapshot_id:
                diff = self._diff_snapshots(
                    input_data.snapshot_id,
                    input_data.compare_snapshot_id
                )
                return SupplyChainMapperOutput(
                    success=True,
                    operation=input_data.operation,
                    snapshot_diff=diff
                )
            else:
                # Return specific snapshot
                snapshot = self._snapshots.get(input_data.snapshot_id)
                return SupplyChainMapperOutput(
                    success=True,
                    operation=input_data.operation,
                    snapshot=snapshot
                )
        else:
            # Current state
            return self._map_supply_chain(input_data)

    def _diff_snapshots(self, base_id: UUID, compare_id: UUID) -> SnapshotDiff:
        """Calculate diff between two snapshots."""
        base = self._snapshots.get(base_id)
        compare = self._snapshots.get(compare_id)

        if not base or not compare:
            return SnapshotDiff(
                base_snapshot_id=base_id,
                compare_snapshot_id=compare_id
            )

        base_nodes = {n['node_id'] for n in base.snapshot_data.get('nodes', [])}
        compare_nodes = {n['node_id'] for n in compare.snapshot_data.get('nodes', [])}

        base_edges = {e['edge_id'] for e in base.snapshot_data.get('edges', [])}
        compare_edges = {e['edge_id'] for e in compare.snapshot_data.get('edges', [])}

        return SnapshotDiff(
            base_snapshot_id=base_id,
            compare_snapshot_id=compare_id,
            nodes_added=[UUID(n) for n in compare_nodes - base_nodes],
            nodes_removed=[UUID(n) for n in base_nodes - compare_nodes],
            edges_added=[UUID(e) for e in compare_edges - base_edges],
            edges_removed=[UUID(e) for e in base_edges - compare_edges],
            coverage_change={
                'previous': base.coverage_percentage,
                'current': compare.coverage_percentage,
                'delta': compare.coverage_percentage - base.coverage_percentage
            }
        )

    # =========================================================================
    # NATURAL LANGUAGE QUERIES
    # =========================================================================

    def _natural_language_query(
        self,
        input_data: SupplyChainMapperInput
    ) -> SupplyChainMapperOutput:
        """Process natural language query using LLM."""
        if not input_data.query:
            return SupplyChainMapperOutput(
                success=False,
                operation=input_data.operation,
                errors=["Query is required for natural language operation"]
            )

        # Parse query to structured filter
        parsed = self._parse_nl_query(input_data.query)

        # Execute filter
        results = self._execute_nl_filter(parsed, input_data.commodity)

        return SupplyChainMapperOutput(
            success=True,
            operation=input_data.operation,
            nl_interpreted_query=parsed.get('interpretation', input_data.query),
            nl_generated_filter=parsed,
            nl_results=results
        )

    def _parse_nl_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query to structured filter."""
        # Simple keyword-based parsing (would use LLM in production)
        filter_dict = {'interpretation': query}

        query_lower = query.lower()

        # Country detection
        countries = {
            'indonesia': 'ID', 'brazil': 'BR', 'colombia': 'CO',
            'vietnam': 'VN', 'peru': 'PE', 'ghana': 'GH',
            'ivory coast': 'CI', 'malaysia': 'MY', 'thailand': 'TH'
        }
        for country, code in countries.items():
            if country in query_lower:
                filter_dict['country_code'] = code
                break

        # Node type detection
        if 'producer' in query_lower or 'farm' in query_lower:
            filter_dict['node_type'] = NodeType.PRODUCER.value
        elif 'processor' in query_lower:
            filter_dict['node_type'] = NodeType.PROCESSOR.value
        elif 'trader' in query_lower:
            filter_dict['node_type'] = NodeType.TRADER.value

        # Verification status
        if 'unverified' in query_lower:
            filter_dict['verification_status'] = VerificationStatus.UNVERIFIED.value
        elif 'verified' in query_lower:
            filter_dict['verification_status'] = VerificationStatus.VERIFIED.value

        # Certification status
        if 'expired' in query_lower and 'certification' in query_lower:
            filter_dict['expired_certifications'] = True

        return filter_dict

    def _execute_nl_filter(
        self,
        filter_dict: Dict[str, Any],
        commodity: CommodityType
    ) -> List[SupplyChainNode]:
        """Execute parsed filter against nodes."""
        results = []

        for node in self._nodes.values():
            if commodity not in node.commodities:
                continue

            match = True

            if 'country_code' in filter_dict:
                if node.country_code != filter_dict['country_code']:
                    match = False

            if 'node_type' in filter_dict:
                if node.node_type.value != filter_dict['node_type']:
                    match = False

            if 'verification_status' in filter_dict:
                if node.verification_status.value != filter_dict['verification_status']:
                    match = False

            if filter_dict.get('expired_certifications'):
                has_expired = any(not c.is_valid for c in node.certifications)
                if not has_expired:
                    match = False

            if match:
                results.append(node)

        return results

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_node(self, node_id: UUID) -> Optional[SupplyChainNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def _get_incoming_edges(
        self,
        node_id: UUID,
        commodity: CommodityType,
        include_inferred: bool
    ) -> List[SupplyChainEdge]:
        """Get edges where node is the target."""
        edges = []
        for edge in self._edges.values():
            if edge.target_node_id == node_id and edge.commodity == commodity:
                if include_inferred or edge.data_source == DataSource.SUPPLIER_DECLARED:
                    edges.append(edge)
        return edges

    def _get_plots_for_producer(self, producer_id: UUID) -> List[OriginPlot]:
        """Get plots for a producer."""
        return [p for p in self._plots.values() if p.producer_node_id == producer_id]

    def _has_plots(self, producer_id: UUID) -> bool:
        """Check if producer has associated plots."""
        return any(p.producer_node_id == producer_id for p in self._plots.values())

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize company name for comparison."""
        name = name.upper()
        # Remove common suffixes
        for suffix in ['LTD', 'LLC', 'INC', 'CORP', 'GMBH', 'SA', 'SRL', 'PTE']:
            name = re.sub(rf'\b{suffix}\b\.?', '', name)
        # Remove punctuation and extra spaces
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    @staticmethod
    def _normalize_tax_id(tax_id: str) -> str:
        """Normalize tax ID for comparison."""
        return tax_id.upper().replace(' ', '').replace('-', '').replace('.', '')

    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """Calculate string similarity using SequenceMatcher."""
        return SequenceMatcher(None, s1, s2).ratio()

    # =========================================================================
    # NODE/EDGE CRUD OPERATIONS
    # =========================================================================

    def add_node(self, node: SupplyChainNode) -> SupplyChainNode:
        """Add a node to the graph."""
        self._nodes[node.node_id] = node
        return node

    def add_edge(self, edge: SupplyChainEdge) -> SupplyChainEdge:
        """Add an edge to the graph."""
        self._edges[edge.edge_id] = edge
        return edge

    def add_plot(self, plot: OriginPlot) -> OriginPlot:
        """Add an origin plot."""
        self._plots[plot.plot_id] = plot
        return plot

    def get_all_nodes(self) -> List[SupplyChainNode]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_all_edges(self) -> List[SupplyChainEdge]:
        """Get all edges."""
        return list(self._edges.values())

    def get_plot(self, plot_id: UUID) -> Optional[OriginPlot]:
        """Get plot by ID."""
        return self._plots.get(plot_id)

    def get_all_plots(self) -> List[OriginPlot]:
        """Get all plots."""
        return list(self._plots.values())

    def update_node(self, node_id: UUID, **kwargs: Any) -> Optional[SupplyChainNode]:
        """Update node fields."""
        node = self._nodes.get(node_id)
        if not node:
            return None
        for field, value in kwargs.items():
            if hasattr(node, field):
                setattr(node, field, value)
        node.updated_at = datetime.utcnow()
        return node

    def delete_node(self, node_id: UUID) -> bool:
        """Delete node by ID."""
        if node_id in self._nodes:
            del self._nodes[node_id]
            # Also remove related edges
            edges_to_remove = [
                eid for eid, e in self._edges.items()
                if e.source_node_id == node_id or e.target_node_id == node_id
            ]
            for eid in edges_to_remove:
                del self._edges[eid]
            return True
        return False

    def update_plot(self, plot_id: UUID, **kwargs: Any) -> Optional[OriginPlot]:
        """Update plot fields."""
        plot = self._plots.get(plot_id)
        if not plot:
            return None
        for field, value in kwargs.items():
            if hasattr(plot, field):
                setattr(plot, field, value)
        return plot

    def delete_plot(self, plot_id: UUID) -> bool:
        """Delete plot by ID."""
        if plot_id in self._plots:
            del self._plots[plot_id]
            return True
        return False

    # =========================================================================
    # LLM-ENHANCED OPERATIONS
    # =========================================================================

    def extract_entities_from_text(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract supply chain entities from unstructured text using LLM.

        Args:
            text: Input text (invoice, email, document)
            entity_types: Types to extract (SUPPLIER_NAME, ADDRESS, etc.)

        Returns:
            List of extracted entities
        """
        if not self.llm:
            logger.warning("LLM service not configured, skipping entity extraction")
            return []

        entities = self.llm.extract_entities(text, entity_types)
        return [
            {
                'entity_type': e.entity_type,
                'value': e.value,
                'confidence': e.confidence,
                'source_text': e.source_text,
                'metadata': e.metadata
            }
            for e in entities
        ]

    def assess_entity_match_with_llm(
        self,
        node_a_id: UUID,
        node_b_id: UUID
    ) -> Dict[str, Any]:
        """
        Use LLM to assess if two nodes refer to the same entity.

        Args:
            node_a_id: First node ID
            node_b_id: Second node ID

        Returns:
            Match assessment with reasoning
        """
        if not self.llm:
            logger.warning("LLM service not configured")
            return {'error': 'LLM not configured'}

        node_a = self._nodes.get(node_a_id)
        node_b = self._nodes.get(node_b_id)

        if not node_a or not node_b:
            return {'error': 'One or both nodes not found'}

        suggestion = self.llm.assess_entity_match(
            node_a.dict(),
            node_b.dict()
        )

        return {
            'is_same_entity': suggestion.is_same_entity,
            'confidence': suggestion.confidence,
            'reasoning': suggestion.reasoning,
            'key_factors': suggestion.key_factors
        }

    def parse_nl_query_with_llm(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query using LLM.

        Args:
            query: Natural language query

        Returns:
            Parsed query with filters
        """
        if not self.llm:
            # Fall back to basic parsing
            return self._parse_nl_query(query)

        parsed = self.llm.parse_natural_language_query(query)
        return {
            'interpretation': parsed.interpretation,
            'filters': parsed.filters,
            'confidence': parsed.confidence,
            'suggested_refinements': parsed.suggested_refinements
        }

    def assess_gap_materiality_with_llm(
        self,
        gap: SupplyChainGap,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to assess gap materiality.

        Args:
            gap: The supply chain gap
            context: Additional context

        Returns:
            Materiality assessment
        """
        if not self.llm:
            # Return basic assessment based on severity
            return {
                'is_material': gap.severity in [GapSeverity.CRITICAL, GapSeverity.HIGH],
                'severity_assessment': gap.severity.value,
                'confidence': 0.7
            }

        ctx = context or {}
        ctx['gap_id'] = str(gap.gap_id)

        assessment = self.llm.assess_gap_materiality(
            gap.gap_type.value,
            gap.description,
            ctx
        )

        return {
            'is_material': assessment.is_material,
            'severity_assessment': assessment.severity_assessment,
            'risk_factors': assessment.risk_factors,
            'recommended_action': assessment.recommended_action,
            'confidence': assessment.confidence
        }

    # =========================================================================
    # DATABASE-BACKED OPERATIONS (when DB is configured)
    # =========================================================================

    async def load_from_database(
        self,
        importer_id: UUID,
        commodity: CommodityType
    ) -> None:
        """
        Load supply chain data from database into memory.

        Args:
            importer_id: Importer to load data for
            commodity: Commodity to filter by
        """
        if not self.db:
            logger.warning("Database not configured, using in-memory storage only")
            return

        try:
            from .database import NodeRepository, EdgeRepository

            node_repo = NodeRepository(self.db)
            edge_repo = EdgeRepository(self.db)

            # Load nodes for commodity
            db_nodes = await node_repo.get_by_commodity(commodity.value, limit=10000)
            for db_node in db_nodes:
                node = SupplyChainNode(
                    node_id=db_node.node_id,
                    node_type=NodeType(db_node.node_type),
                    name=db_node.name,
                    country_code=db_node.country_code,
                    commodities=[CommodityType(c) for c in db_node.commodities or []],
                    tax_id=db_node.tax_id,
                    duns_number=db_node.duns_number,
                    eori_number=db_node.eori_number,
                    verification_status=VerificationStatus(db_node.verification_status or 'UNVERIFIED'),
                    disclosure_status=DisclosureStatus(db_node.disclosure_status or 'FULL'),
                    tier=db_node.tier,
                    metadata=db_node.metadata or {}
                )
                self._nodes[node.node_id] = node

            # Load edges for commodity
            for node_id in self._nodes.keys():
                incoming = await edge_repo.get_incoming_edges(node_id, commodity.value)
                for db_edge in incoming:
                    edge = SupplyChainEdge(
                        edge_id=db_edge.edge_id,
                        source_node_id=db_edge.source_node_id,
                        target_node_id=db_edge.target_node_id,
                        edge_type=EdgeType(db_edge.edge_type),
                        commodity=CommodityType(db_edge.commodity),
                        confidence_score=Decimal(str(db_edge.confidence_score or 0.5)),
                        data_source=DataSource(db_edge.data_source or 'SUPPLIER_DECLARED')
                    )
                    self._edges[edge.edge_id] = edge

            logger.info(
                f"Loaded {len(self._nodes)} nodes and {len(self._edges)} edges from database"
            )

        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            raise

    async def persist_to_database(self) -> Dict[str, int]:
        """
        Persist in-memory data to database.

        Returns:
            Count of persisted entities
        """
        if not self.db:
            logger.warning("Database not configured, cannot persist")
            return {'nodes': 0, 'edges': 0, 'plots': 0}

        try:
            from .database import NodeRepository, EdgeRepository
            from .models import SupplyChainNodeModel, SupplyChainEdgeModel

            node_repo = NodeRepository(self.db)
            edge_repo = EdgeRepository(self.db)

            nodes_persisted = 0
            edges_persisted = 0

            # Persist nodes
            for node in self._nodes.values():
                db_node = SupplyChainNodeModel(
                    node_id=node.node_id,
                    node_type=node.node_type.value if hasattr(node.node_type, 'value') else node.node_type,
                    name=node.name,
                    country_code=node.country_code,
                    commodities=[c.value if hasattr(c, 'value') else c for c in node.commodities],
                    tax_id=node.tax_id,
                    duns_number=node.duns_number,
                    eori_number=node.eori_number,
                    verification_status=node.verification_status.value if hasattr(node.verification_status, 'value') else node.verification_status,
                    disclosure_status=node.disclosure_status.value if hasattr(node.disclosure_status, 'value') else node.disclosure_status,
                    tier=node.tier,
                    metadata=node.metadata
                )
                await node_repo.create(db_node)
                nodes_persisted += 1

            # Persist edges
            for edge in self._edges.values():
                db_edge = SupplyChainEdgeModel(
                    edge_id=edge.edge_id,
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    edge_type=edge.edge_type.value if hasattr(edge.edge_type, 'value') else edge.edge_type,
                    commodity=edge.commodity.value if hasattr(edge.commodity, 'value') else edge.commodity,
                    confidence_score=float(edge.confidence_score),
                    data_source=edge.data_source.value if hasattr(edge.data_source, 'value') else edge.data_source
                )
                await edge_repo.create(db_edge)
                edges_persisted += 1

            logger.info(
                f"Persisted {nodes_persisted} nodes and {edges_persisted} edges to database"
            )

            return {
                'nodes': nodes_persisted,
                'edges': edges_persisted,
                'plots': 0  # TODO: implement plot persistence
            }

        except Exception as e:
            logger.error(f"Failed to persist to database: {e}")
            raise


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "pack_id": "greenlang/eudr-supply-chain-mapper",
    "version": "1.0.0",
    "name": "EUDR Supply Chain Mapper",
    "description": "Maps multi-tier supply chains for EUDR compliance",
    "agent_family": "EUDRTraceabilityFamily",
    "layer": "Supply Chain Traceability",
    "domains": ["supply_chain", "traceability", "eudr"],
    "inputs": {
        "importer_id": "UUID of the importing company",
        "commodity": "EUDR commodity type (CATTLE, COCOA, COFFEE, etc.)",
        "operation": "Operation to perform"
    },
    "outputs": {
        "graph": "Supply chain graph with nodes and edges",
        "coverage_report": "Coverage metrics and gaps",
        "snapshot": "Immutable snapshot of supply chain state"
    },
    "regulatory_reference": "EU Regulation 2023/1115",
    "enforcement_dates": {
        "large_operators": "2024-12-30",
        "sme": "2025-06-30"
    }
}
