# -*- coding: utf-8 -*-
"""
Supply Chain Mapper Data Models - AGENT-EUDR-001

Pydantic v2 data models for the Supply Chain Mapping Master Agent covering
graph-native supply chain modeling for EU Deforestation Regulation (EUDR)
compliance. Defines all enumerations, core graph models, gap analysis models,
risk propagation models, and API request/response wrappers.

The supply chain graph models represent actors (producers, collectors,
processors, traders, importers), custody transfers (edges), production plots,
and batch split/merge operations as a directed acyclic graph (DAG). Every
model is designed for deterministic serialization and SHA-256 provenance
hashing to ensure zero-hallucination, bit-perfect reproducibility.

Enumerations (5):
    - NodeType, CustodyModel, RiskLevel, ComplianceStatus, GapType

Core Models (5):
    - SupplyChainNode, SupplyChainEdge, SupplyChainGraph,
      SupplyChainGap, RiskPropagationResult

Query/Response Models (14):
    - CreateGraphRequest, CreateNodeRequest, CreateEdgeRequest,
      UpdateNodeRequest, GraphQueryParams, NodeQueryParams,
      EdgeQueryParams, TraceResult, TierDistribution,
      RiskSummary, GapAnalysisResult, DDSExportData,
      GraphLayoutData, SankeyData

Compatibility:
    Imports EUDRCommodity, CustodyModel (as BaseCustodyModel), RiskLevel
    (as BaseRiskLevel), and ComplianceStatus (as BaseComplianceStatus) from
    greenlang.agents.data.eudr_traceability.models for cross-agent consistency with
    AGENT-DATA-005 EUDR Traceability Connector.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001 Supply Chain Mapping Master (GL-EUDR-SCM-001)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from greenlang.schemas import (
    GreenLangBase,
    GreenLangRecord,
    GreenLangRequest,
    GreenLangResponse,
    ProvenanceMixin,
    utcnow,
    new_uuid,
)


# ---------------------------------------------------------------------------
# Helpers (local alias for backward compatibility)
# ---------------------------------------------------------------------------

_utcnow = utcnow


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: Maximum number of nodes in a single graph before sharding is recommended.
MAX_NODES_PER_GRAPH: int = 100_000

#: Maximum number of edges in a single graph.
MAX_EDGES_PER_GRAPH: int = 500_000

#: Maximum tier depth supported for recursive mapping.
MAX_TIER_DEPTH: int = 50

#: EUDR deforestation cutoff date (31 December 2020).
EUDR_DEFORESTATION_CUTOFF: str = "2020-12-31"

#: Default risk weights as specified in PRD Section 6.1 Feature 5.
DEFAULT_RISK_WEIGHTS: Dict[str, float] = {
    "country": 0.30,
    "commodity": 0.20,
    "supplier": 0.25,
    "deforestation": 0.25,
}


# =============================================================================
# Enumerations
# =============================================================================


class NodeType(str, Enum):
    """Supply chain actor type classification.

    Identifies the role an actor plays in the EUDR-regulated supply chain.
    Each node in the supply chain graph is assigned exactly one NodeType
    that determines its position and expected attributes.

    PRODUCER: Farm, plantation, forest concession, or ranch that
        produces the raw commodity on a specific plot of land.
    COLLECTOR: Cooperative, aggregation point, silo, or collection
        center that aggregates commodities from multiple producers.
    PROCESSOR: Mill, refinery, slaughterhouse, sawmill, or factory
        that transforms raw commodities into derived products.
    TRADER: Trading company or intermediary that buys and sells
        commodities without physical transformation.
    IMPORTER: EU-based operator placing the product on the EU market,
        responsible for filing the Due Diligence Statement.
    CERTIFIER: Certification body (FSC, RSPO, Rainforest Alliance)
        that verifies compliance standards.
    WAREHOUSE: Storage and logistics facility used for commodity
        holding between supply chain stages.
    PORT: Port of loading or unloading for international shipments,
        relevant for customs and DDS documentation.
    """

    PRODUCER = "producer"
    COLLECTOR = "collector"
    PROCESSOR = "processor"
    TRADER = "trader"
    IMPORTER = "importer"
    CERTIFIER = "certifier"
    WAREHOUSE = "warehouse"
    PORT = "port"


class EUDRCommodity(str, Enum):
    """EUDR-regulated commodities and their derived products.

    The EU Deforestation Regulation covers seven primary commodities
    and their key derived products as listed in Annex I of Regulation
    (EU) 2023/1115. This enum mirrors the definition in
    greenlang.agents.data.eudr_traceability.models for cross-agent consistency.
    """

    # Primary commodities
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

    # Derived products - cattle
    BEEF = "beef"
    LEATHER = "leather"

    # Derived products - cocoa
    CHOCOLATE = "chocolate"

    # Derived products - oil palm
    PALM_OIL = "palm_oil"

    # Derived products - rubber
    NATURAL_RUBBER = "natural_rubber"
    TYRES = "tyres"

    # Derived products - soya
    SOYBEAN_OIL = "soybean_oil"
    SOYBEAN_MEAL = "soybean_meal"

    # Derived products - wood
    TIMBER = "timber"
    FURNITURE = "furniture"
    PAPER = "paper"
    CHARCOAL = "charcoal"


class CustodyModel(str, Enum):
    """Chain of custody models for commodity traceability.

    Defines how EUDR-relevant commodities are tracked through the
    supply chain from plot of origin to final product placement.
    Per ISO 22095:2020 and EUDR Article 10(2)(f).

    IDENTITY_PRESERVED: Compliant material is physically separated
        throughout the entire supply chain. Full plot-to-product
        traceability is maintained.
    SEGREGATED: Compliant material is kept separate from non-compliant
        material but may be mixed with other compliant material from
        different certified sources.
    MASS_BALANCE: Compliant and non-compliant material may be
        physically mixed but quantities are tracked administratively
        to ensure the correct volume of compliant product is sold.
    """

    IDENTITY_PRESERVED = "identity_preserved"
    SEGREGATED = "segregated"
    MASS_BALANCE = "mass_balance"


class RiskLevel(str, Enum):
    """Risk classification levels for EUDR country benchmarking.

    Based on Article 29 of Regulation (EU) 2023/1115, the European
    Commission classifies countries or parts thereof into risk
    categories based on deforestation rates, agricultural expansion,
    commodity production trends, and stakeholder input.

    LOW: Simplified due diligence permitted; reduced verification
        requirements.
    STANDARD: Standard due diligence applies; full information
        collection and risk assessment required.
    HIGH: Enhanced due diligence required; additional risk mitigation
        measures mandatory, including independent audits.
    """

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


class ComplianceStatus(str, Enum):
    """EUDR compliance status for supply chain nodes and edges.

    Reflects the current verification state of a supply chain actor
    or custody transfer against EUDR requirements. Used for gap
    analysis and compliance readiness scoring.

    COMPLIANT: Fully verified and meets all EUDR requirements.
    NON_COMPLIANT: Verified and found to violate EUDR requirements.
    PENDING_VERIFICATION: Not yet verified; awaiting documentation
        or assessment.
    UNDER_REVIEW: Verification in progress; documentation submitted
        but assessment not yet complete.
    INSUFFICIENT_DATA: Cannot determine compliance due to missing
        or incomplete information.
    EXEMPTED: Exempt from EUDR requirements (e.g., recycled or
        pre-existing stock with valid proof).
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_VERIFICATION = "pending_verification"
    UNDER_REVIEW = "under_review"
    INSUFFICIENT_DATA = "insufficient_data"
    EXEMPTED = "exempted"


class GapType(str, Enum):
    """Classification of supply chain compliance gaps.

    Each gap type maps to a specific EUDR article and has an
    associated severity level and auto-remediation action as
    defined in PRD Section 6.1 Feature 6.

    MISSING_GEOLOCATION: Producer node lacks GPS coordinates
        (EUDR Article 9, Critical).
    MISSING_POLYGON: Plot > 4 hectares without polygon boundary
        data (EUDR Article 9(1)(d), Critical).
    BROKEN_CUSTODY_CHAIN: Product has no traceable link back to
        origin production plots (EUDR Article 4(2)(f), Critical).
    UNVERIFIED_ACTOR: Supply chain node without identity or
        compliance verification (EUDR Article 10, High).
    MISSING_TIER: Opaque segment where sub-tier visibility is
        absent (EUDR Article 4(2), High).
    MASS_BALANCE_DISCREPANCY: Output quantity exceeds input quantity
        beyond tolerance (EUDR Article 10(2)(f), High).
    MISSING_CERTIFICATION: Node lacks expected certification
        (EUDR Article 10, Medium).
    STALE_DATA: Data older than 12 months without refresh
        (EUDR Article 31, Medium).
    ORPHAN_NODE: Node with no incoming or outgoing edges
        (internal quality check, Low).
    MISSING_DOCUMENTATION: Node or edge missing required custody
        transfer documents (EUDR Article 4(2), Medium).
    """

    MISSING_GEOLOCATION = "missing_geolocation"
    MISSING_POLYGON = "missing_polygon"
    BROKEN_CUSTODY_CHAIN = "broken_custody_chain"
    UNVERIFIED_ACTOR = "unverified_actor"
    MISSING_TIER = "missing_tier"
    MASS_BALANCE_DISCREPANCY = "mass_balance_discrepancy"
    MISSING_CERTIFICATION = "missing_certification"
    STALE_DATA = "stale_data"
    ORPHAN_NODE = "orphan_node"
    MISSING_DOCUMENTATION = "missing_documentation"


class GapSeverity(str, Enum):
    """Severity classification for supply chain gaps.

    Determines the priority of remediation and the impact on
    compliance readiness scoring.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TransportMode(str, Enum):
    """Transport mode for custody transfer edges.

    Identifies how commodities are physically moved between
    supply chain nodes.
    """

    ROAD = "road"
    SEA = "sea"
    RAIL = "rail"
    AIR = "air"
    RIVER = "river"
    PIPELINE = "pipeline"
    MULTIMODAL = "multimodal"


#: Maps derived products back to their primary EUDR commodity.
DERIVED_TO_PRIMARY: Dict[EUDRCommodity, EUDRCommodity] = {
    EUDRCommodity.BEEF: EUDRCommodity.CATTLE,
    EUDRCommodity.LEATHER: EUDRCommodity.CATTLE,
    EUDRCommodity.CHOCOLATE: EUDRCommodity.COCOA,
    EUDRCommodity.PALM_OIL: EUDRCommodity.OIL_PALM,
    EUDRCommodity.NATURAL_RUBBER: EUDRCommodity.RUBBER,
    EUDRCommodity.TYRES: EUDRCommodity.RUBBER,
    EUDRCommodity.SOYBEAN_OIL: EUDRCommodity.SOYA,
    EUDRCommodity.SOYBEAN_MEAL: EUDRCommodity.SOYA,
    EUDRCommodity.TIMBER: EUDRCommodity.WOOD,
    EUDRCommodity.FURNITURE: EUDRCommodity.WOOD,
    EUDRCommodity.PAPER: EUDRCommodity.WOOD,
    EUDRCommodity.CHARCOAL: EUDRCommodity.WOOD,
}

#: Set of the seven primary EUDR commodities.
PRIMARY_COMMODITIES = frozenset({
    EUDRCommodity.CATTLE,
    EUDRCommodity.COCOA,
    EUDRCommodity.COFFEE,
    EUDRCommodity.OIL_PALM,
    EUDRCommodity.RUBBER,
    EUDRCommodity.SOYA,
    EUDRCommodity.WOOD,
})

#: Maps gap types to their default severity.
GAP_SEVERITY_MAP: Dict[GapType, GapSeverity] = {
    GapType.MISSING_GEOLOCATION: GapSeverity.CRITICAL,
    GapType.MISSING_POLYGON: GapSeverity.CRITICAL,
    GapType.BROKEN_CUSTODY_CHAIN: GapSeverity.CRITICAL,
    GapType.UNVERIFIED_ACTOR: GapSeverity.HIGH,
    GapType.MISSING_TIER: GapSeverity.HIGH,
    GapType.MASS_BALANCE_DISCREPANCY: GapSeverity.HIGH,
    GapType.MISSING_CERTIFICATION: GapSeverity.MEDIUM,
    GapType.STALE_DATA: GapSeverity.MEDIUM,
    GapType.ORPHAN_NODE: GapSeverity.LOW,
    GapType.MISSING_DOCUMENTATION: GapSeverity.MEDIUM,
}

#: Maps gap types to their EUDR article reference.
GAP_ARTICLE_MAP: Dict[GapType, str] = {
    GapType.MISSING_GEOLOCATION: "Article 9",
    GapType.MISSING_POLYGON: "Article 9(1)(d)",
    GapType.BROKEN_CUSTODY_CHAIN: "Article 4(2)(f)",
    GapType.UNVERIFIED_ACTOR: "Article 10",
    GapType.MISSING_TIER: "Article 4(2)",
    GapType.MASS_BALANCE_DISCREPANCY: "Article 10(2)(f)",
    GapType.MISSING_CERTIFICATION: "Article 10",
    GapType.STALE_DATA: "Article 31",
    GapType.ORPHAN_NODE: "Internal",
    GapType.MISSING_DOCUMENTATION: "Article 4(2)",
}


# =============================================================================
# Core Data Models
# =============================================================================


class SupplyChainNode(BaseModel):
    """A single actor (node) in the EUDR supply chain graph.

    Represents a supply chain participant such as a producer, collector,
    processor, trader, or importer. Each node has a type, geographic
    location, risk assessment, compliance status, and links to
    production plots (for producers).

    Attributes:
        node_id: Unique identifier for this node within the graph.
        node_type: Role of this actor in the supply chain.
        operator_id: Reference to the operator/company record ID.
        operator_name: Human-readable legal name of the operator.
        country_code: ISO 3166-1 alpha-2 country code where the
            operator is based or the node is located.
        region: Sub-national region or administrative division.
        coordinates: Optional GPS (latitude, longitude) tuple in
            WGS84 decimal degrees.
        commodities: List of EUDR commodities handled by this node.
        tier_depth: Distance from the importer node in the graph.
            0 = importer, 1 = direct supplier, etc.
        risk_score: Composite risk score on a 0-100 scale, computed
            by the risk propagation engine.
        risk_level: Categorical risk classification derived from
            risk_score.
        compliance_status: Current compliance verification state.
        certifications: List of certification identifiers held by
            this operator (e.g., FSC, RSPO, RA).
        plot_ids: List of linked production plot IDs (relevant only
            for PRODUCER nodes).
        metadata: Additional key-value metadata for extensibility.
        created_at: UTC timestamp of node creation.
        updated_at: UTC timestamp of last modification.
    """

    model_config = ConfigDict(from_attributes=True)

    node_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this node within the graph",
    )
    node_type: NodeType = Field(
        ...,
        description="Role of this actor in the supply chain",
    )
    operator_id: str = Field(
        ...,
        description="Reference to the operator/company record ID",
    )
    operator_name: str = Field(
        ...,
        description="Human-readable legal name of the operator",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    region: Optional[str] = Field(
        None,
        description="Sub-national region or administrative division",
    )
    coordinates: Optional[Tuple[float, float]] = Field(
        None,
        description="GPS coordinates as (latitude, longitude) in WGS84",
    )
    commodities: List[EUDRCommodity] = Field(
        default_factory=list,
        description="EUDR commodities handled by this node",
    )
    tier_depth: int = Field(
        default=0,
        ge=0,
        le=MAX_TIER_DEPTH,
        description="Distance from importer (0 = importer, 1 = direct supplier)",
    )
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Composite risk score (0-100)",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.STANDARD,
        description="Categorical risk classification",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING_VERIFICATION,
        description="Current compliance verification state",
    )
    certifications: List[str] = Field(
        default_factory=list,
        description="Certification identifiers (FSC, RSPO, RA, etc.)",
    )
    plot_ids: List[str] = Field(
        default_factory=list,
        description="Linked production plot IDs (producers only)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value metadata",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of node creation",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of last modification",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate and normalize country code to uppercase ISO alpha-2."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("operator_id")
    @classmethod
    def validate_operator_id(cls, v: str) -> str:
        """Validate operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("operator_id must be non-empty")
        return v

    @field_validator("operator_name")
    @classmethod
    def validate_operator_name(cls, v: str) -> str:
        """Validate operator_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("operator_name must be non-empty")
        return v

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(
        cls, v: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        """Validate GPS coordinates are within valid WGS84 ranges."""
        if v is None:
            return v
        lat, lon = v
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(
                f"Latitude must be between -90 and 90, got {lat}"
            )
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(
                f"Longitude must be between -180 and 180, got {lon}"
            )
        return v

    @model_validator(mode="after")
    def validate_producer_has_plots_info(self) -> SupplyChainNode:
        """Warn-level check: producers should have plot_ids or coordinates."""
        # This is intentionally not a hard error since gaps are detected
        # by the gap analyzer; here we only ensure structural correctness.
        return self


class SupplyChainEdge(BaseModel):
    """A directed edge (custody transfer) between two supply chain nodes.

    Represents a single transfer of custody where a commodity or product
    moves from one operator (source) to another (target). Includes
    quantity, batch tracking, custody model, transport details, and
    a SHA-256 provenance hash for audit integrity.

    Attributes:
        edge_id: Unique identifier for this edge.
        source_node_id: ID of the upstream (source) node.
        target_node_id: ID of the downstream (target) node.
        commodity: EUDR commodity being transferred.
        product_description: Human-readable description of the product.
        quantity: Quantity transferred, using Decimal for mass balance
            accuracy (no floating-point drift).
        unit: Unit of measurement (default: kg).
        batch_number: Optional batch or lot identifier for traceability.
        custody_model: Chain of custody model governing this transfer.
        transfer_date: Date and time when the custody transfer occurred.
        cn_code: Optional EU Combined Nomenclature code.
        hs_code: Optional Harmonized System code.
        transport_mode: Optional mode of transport used.
        provenance_hash: SHA-256 hash of the edge data for audit trail.
        metadata: Additional key-value metadata.
        created_at: UTC timestamp of edge creation.
    """

    model_config = ConfigDict(from_attributes=True)

    edge_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this edge",
    )
    source_node_id: str = Field(
        ...,
        description="ID of the upstream (source) node",
    )
    target_node_id: str = Field(
        ...,
        description="ID of the downstream (target) node",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity being transferred",
    )
    product_description: str = Field(
        ...,
        description="Human-readable description of the product",
    )
    quantity: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Quantity transferred (Decimal for mass balance accuracy)",
    )
    unit: str = Field(
        default="kg",
        description="Unit of measurement for quantity",
    )
    batch_number: Optional[str] = Field(
        None,
        description="Optional batch or lot identifier",
    )
    custody_model: CustodyModel = Field(
        default=CustodyModel.SEGREGATED,
        description="Chain of custody model for this transfer",
    )
    transfer_date: datetime = Field(
        default_factory=_utcnow,
        description="Date and time of the custody transfer",
    )
    cn_code: Optional[str] = Field(
        None,
        description="EU Combined Nomenclature code",
    )
    hs_code: Optional[str] = Field(
        None,
        description="Harmonized System code",
    )
    transport_mode: Optional[TransportMode] = Field(
        None,
        description="Mode of transport used for this transfer",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash for audit trail",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value metadata",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of edge creation",
    )

    @field_validator("source_node_id")
    @classmethod
    def validate_source_node_id(cls, v: str) -> str:
        """Validate source_node_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_node_id must be non-empty")
        return v

    @field_validator("target_node_id")
    @classmethod
    def validate_target_node_id(cls, v: str) -> str:
        """Validate target_node_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_node_id must be non-empty")
        return v

    @field_validator("product_description")
    @classmethod
    def validate_product_description(cls, v: str) -> str:
        """Validate product_description is non-empty."""
        if not v or not v.strip():
            raise ValueError("product_description must be non-empty")
        return v

    @model_validator(mode="after")
    def validate_no_self_loop(self) -> SupplyChainEdge:
        """Ensure the edge does not form a self-loop."""
        if self.source_node_id == self.target_node_id:
            raise ValueError(
                f"Self-loop detected: source_node_id and target_node_id "
                f"are both '{self.source_node_id}'"
            )
        return self


class SupplyChainGap(BaseModel):
    """A compliance gap identified in the supply chain graph.

    Represents a specific deficiency in the supply chain mapping that
    must be remediated to achieve EUDR compliance. Each gap has a type,
    severity, affected entity, EUDR article reference, and suggested
    remediation action.

    Attributes:
        gap_id: Unique identifier for this gap record.
        gap_type: Classification of the gap.
        severity: Severity level determining remediation priority.
        affected_node_id: Optional ID of the affected supply chain node.
        affected_edge_id: Optional ID of the affected supply chain edge.
        description: Human-readable description of the gap.
        remediation: Suggested remediation action.
        eudr_article: EUDR article that this gap violates.
        is_resolved: Whether this gap has been remediated.
        resolved_at: Timestamp when the gap was resolved.
        detected_at: Timestamp when the gap was detected.
    """

    model_config = ConfigDict(from_attributes=True)

    gap_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this gap record",
    )
    gap_type: GapType = Field(
        ...,
        description="Classification of the compliance gap",
    )
    severity: GapSeverity = Field(
        default=GapSeverity.MEDIUM,
        description="Severity level for remediation priority",
    )
    affected_node_id: Optional[str] = Field(
        None,
        description="ID of the affected supply chain node",
    )
    affected_edge_id: Optional[str] = Field(
        None,
        description="ID of the affected supply chain edge",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the gap",
    )
    remediation: Optional[str] = Field(
        None,
        description="Suggested remediation action",
    )
    eudr_article: str = Field(
        default="",
        description="EUDR article reference violated by this gap",
    )
    is_resolved: bool = Field(
        default=False,
        description="Whether this gap has been remediated",
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the gap was resolved",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the gap was detected",
    )

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate description is non-empty."""
        if not v or not v.strip():
            raise ValueError("description must be non-empty")
        return v

    @model_validator(mode="after")
    def auto_populate_severity_and_article(self) -> SupplyChainGap:
        """Auto-populate severity and article from gap type mappings."""
        if self.severity == GapSeverity.MEDIUM and self.gap_type in GAP_SEVERITY_MAP:
            self.severity = GAP_SEVERITY_MAP[self.gap_type]
        if not self.eudr_article and self.gap_type in GAP_ARTICLE_MAP:
            self.eudr_article = GAP_ARTICLE_MAP[self.gap_type]
        return self


class SupplyChainGraph(BaseModel):
    """Complete supply chain graph for one operator and commodity.

    The top-level container that holds all nodes, edges, compliance
    metrics, and gap analysis results for a single operator's supply
    chain view of one EUDR commodity. Supports versioning with
    immutable snapshots for audit trail compliance per Article 31.

    Attributes:
        graph_id: Unique identifier for this graph.
        operator_id: ID of the operator who owns this graph view.
        commodity: Primary EUDR commodity tracked by this graph.
        graph_name: Optional human-readable name for the graph.
        nodes: Dictionary of node_id -> SupplyChainNode.
        edges: Dictionary of edge_id -> SupplyChainEdge.
        total_nodes: Count of nodes in the graph.
        total_edges: Count of edges in the graph.
        max_tier_depth: Maximum tier depth found in the graph.
        traceability_score: Percentage of products traceable to
            origin plots (0-100).
        compliance_readiness: Overall compliance readiness score
            (0-100), accounting for gaps and verification status.
        risk_summary: Count of nodes by risk level.
        gaps: List of detected compliance gaps.
        created_at: UTC timestamp of graph creation.
        updated_at: UTC timestamp of last modification.
        version: Monotonically increasing version number,
            incremented on every mutation.
    """

    model_config = ConfigDict(from_attributes=True)

    graph_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this graph",
    )
    operator_id: str = Field(
        ...,
        description="ID of the operator who owns this graph view",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="Primary EUDR commodity tracked by this graph",
    )
    graph_name: Optional[str] = Field(
        None,
        description="Optional human-readable name for the graph",
    )
    nodes: Dict[str, SupplyChainNode] = Field(
        default_factory=dict,
        description="Nodes indexed by node_id",
    )
    edges: Dict[str, SupplyChainEdge] = Field(
        default_factory=dict,
        description="Edges indexed by edge_id",
    )
    total_nodes: int = Field(
        default=0,
        ge=0,
        description="Count of nodes in the graph",
    )
    total_edges: int = Field(
        default=0,
        ge=0,
        description="Count of edges in the graph",
    )
    max_tier_depth: int = Field(
        default=0,
        ge=0,
        description="Maximum tier depth found in the graph",
    )
    traceability_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of products traceable to origin plots (0-100)",
    )
    compliance_readiness: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall compliance readiness score (0-100)",
    )
    risk_summary: Dict[str, int] = Field(
        default_factory=lambda: {"low": 0, "standard": 0, "high": 0},
        description="Count of nodes by risk level",
    )
    gaps: List[SupplyChainGap] = Field(
        default_factory=list,
        description="Detected compliance gaps",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of graph creation",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of last modification",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Version number, incremented on every mutation",
    )

    @field_validator("operator_id")
    @classmethod
    def validate_operator_id(cls, v: str) -> str:
        """Validate operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("operator_id must be non-empty")
        return v


class RiskPropagationResult(BaseModel):
    """Result of risk propagation across the supply chain graph.

    Captures the risk score changes for a single node after
    propagation, including the contributing risk factors and
    the propagation source for audit purposes.

    Attributes:
        node_id: ID of the node whose risk was updated.
        previous_risk_score: Risk score before propagation.
        new_risk_score: Risk score after propagation.
        previous_risk_level: Risk level before propagation.
        new_risk_level: Risk level after propagation.
        propagation_source: What triggered the propagation
            (e.g., country_update, parent_risk_change).
        risk_factors: Breakdown of individual risk dimension scores.
        inherited_risk: Maximum risk inherited from parent nodes.
        calculated_at: Timestamp of the propagation.
    """

    model_config = ConfigDict(from_attributes=True)

    node_id: str = Field(
        ...,
        description="ID of the node whose risk was updated",
    )
    previous_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Risk score before propagation",
    )
    new_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Risk score after propagation",
    )
    previous_risk_level: RiskLevel = Field(
        default=RiskLevel.STANDARD,
        description="Risk level before propagation",
    )
    new_risk_level: RiskLevel = Field(
        default=RiskLevel.STANDARD,
        description="Risk level after propagation",
    )
    propagation_source: str = Field(
        default="manual",
        description="What triggered the propagation",
    )
    risk_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of individual risk dimension scores",
    )
    inherited_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Maximum risk inherited from parent nodes",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of the propagation",
    )


# =============================================================================
# Request Models
# =============================================================================


class CreateGraphRequest(BaseModel):
    """Request body for creating a new supply chain graph.

    Attributes:
        commodity: Primary EUDR commodity for this graph.
        graph_name: Optional human-readable name.
    """

    model_config = ConfigDict(extra="forbid")

    commodity: EUDRCommodity = Field(
        ...,
        description="Primary EUDR commodity for this graph",
    )
    graph_name: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional human-readable name for the graph",
    )


class CreateNodeRequest(BaseModel):
    """Request body for adding a node to a supply chain graph.

    Attributes:
        node_type: Role of this actor in the supply chain.
        operator_id: Reference to the operator/company record ID.
        operator_name: Human-readable legal name of the operator.
        country_code: ISO 3166-1 alpha-2 country code.
        region: Optional sub-national region.
        coordinates: Optional GPS (latitude, longitude) tuple.
        commodities: List of EUDR commodities handled.
        certifications: List of certification identifiers.
        plot_ids: Linked production plot IDs (producers only).
        metadata: Additional key-value metadata.
    """

    model_config = ConfigDict(extra="forbid")

    node_type: NodeType = Field(
        ...,
        description="Role of this actor in the supply chain",
    )
    operator_id: str = Field(
        ...,
        description="Reference to the operator/company record ID",
    )
    operator_name: str = Field(
        ...,
        description="Human-readable legal name of the operator",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    region: Optional[str] = Field(
        None,
        max_length=200,
        description="Sub-national region or administrative division",
    )
    coordinates: Optional[Tuple[float, float]] = Field(
        None,
        description="GPS coordinates as (latitude, longitude) in WGS84",
    )
    commodities: List[EUDRCommodity] = Field(
        default_factory=list,
        description="EUDR commodities handled by this node",
    )
    certifications: List[str] = Field(
        default_factory=list,
        description="Certification identifiers (FSC, RSPO, RA)",
    )
    plot_ids: List[str] = Field(
        default_factory=list,
        description="Linked production plot IDs (producers only)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value metadata",
    )

    @field_validator("operator_id")
    @classmethod
    def validate_operator_id(cls, v: str) -> str:
        """Validate operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("operator_id must be non-empty")
        return v

    @field_validator("operator_name")
    @classmethod
    def validate_operator_name(cls, v: str) -> str:
        """Validate operator_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("operator_name must be non-empty")
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate and normalize country code to uppercase."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(
        cls, v: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        """Validate GPS coordinates within WGS84 ranges."""
        if v is None:
            return v
        lat, lon = v
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180, got {lon}")
        return v


class CreateEdgeRequest(BaseModel):
    """Request body for adding a custody transfer edge to a graph.

    Attributes:
        source_node_id: ID of the upstream (source) node.
        target_node_id: ID of the downstream (target) node.
        commodity: EUDR commodity being transferred.
        product_description: Description of the product.
        quantity: Quantity transferred.
        unit: Unit of measurement.
        batch_number: Optional batch or lot identifier.
        custody_model: Chain of custody model.
        transfer_date: Date of the custody transfer.
        cn_code: Optional EU Combined Nomenclature code.
        hs_code: Optional Harmonized System code.
        transport_mode: Optional mode of transport.
    """

    model_config = ConfigDict(extra="forbid")

    source_node_id: str = Field(
        ...,
        description="ID of the upstream (source) node",
    )
    target_node_id: str = Field(
        ...,
        description="ID of the downstream (target) node",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity being transferred",
    )
    product_description: str = Field(
        ...,
        description="Human-readable description of the product",
    )
    quantity: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Quantity transferred",
    )
    unit: str = Field(
        default="kg",
        description="Unit of measurement for quantity",
    )
    batch_number: Optional[str] = Field(
        None,
        description="Optional batch or lot identifier",
    )
    custody_model: CustodyModel = Field(
        default=CustodyModel.SEGREGATED,
        description="Chain of custody model",
    )
    transfer_date: Optional[datetime] = Field(
        None,
        description="Date and time of the custody transfer",
    )
    cn_code: Optional[str] = Field(
        None,
        description="EU Combined Nomenclature code",
    )
    hs_code: Optional[str] = Field(
        None,
        description="Harmonized System code",
    )
    transport_mode: Optional[TransportMode] = Field(
        None,
        description="Mode of transport used",
    )

    @field_validator("source_node_id")
    @classmethod
    def validate_source_node_id(cls, v: str) -> str:
        """Validate source_node_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_node_id must be non-empty")
        return v

    @field_validator("target_node_id")
    @classmethod
    def validate_target_node_id(cls, v: str) -> str:
        """Validate target_node_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_node_id must be non-empty")
        return v

    @field_validator("product_description")
    @classmethod
    def validate_product_description(cls, v: str) -> str:
        """Validate product_description is non-empty."""
        if not v or not v.strip():
            raise ValueError("product_description must be non-empty")
        return v

    @model_validator(mode="after")
    def validate_no_self_loop(self) -> CreateEdgeRequest:
        """Ensure the edge does not form a self-loop."""
        if self.source_node_id == self.target_node_id:
            raise ValueError(
                "source_node_id and target_node_id must be different"
            )
        return self


class UpdateNodeRequest(BaseModel):
    """Request body for updating a supply chain node's attributes.

    All fields are optional; only provided fields are updated.

    Attributes:
        operator_name: Updated legal name.
        country_code: Updated ISO 3166-1 alpha-2 country code.
        region: Updated sub-national region.
        coordinates: Updated GPS coordinates.
        commodities: Updated commodity list.
        compliance_status: Updated compliance status.
        certifications: Updated certification list.
        plot_ids: Updated plot ID list.
        metadata: Updated metadata (merged with existing).
    """

    model_config = ConfigDict(extra="forbid")

    operator_name: Optional[str] = Field(None, description="Updated legal name")
    country_code: Optional[str] = Field(None, min_length=2, max_length=2)
    region: Optional[str] = Field(None, max_length=200)
    coordinates: Optional[Tuple[float, float]] = Field(None)
    commodities: Optional[List[EUDRCommodity]] = Field(None)
    compliance_status: Optional[ComplianceStatus] = Field(None)
    certifications: Optional[List[str]] = Field(None)
    plot_ids: Optional[List[str]] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize country code if provided."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(
        cls, v: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        """Validate GPS coordinates if provided."""
        if v is None:
            return v
        lat, lon = v
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180, got {lon}")
        return v


# =============================================================================
# Query Parameter Models
# =============================================================================


class GraphQueryParams(BaseModel):
    """Query parameters for listing supply chain graphs.

    Attributes:
        commodity: Filter by EUDR commodity.
        limit: Maximum number of results to return.
        offset: Number of results to skip for pagination.
    """

    model_config = ConfigDict(extra="forbid")

    commodity: Optional[EUDRCommodity] = Field(None)
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class NodeQueryParams(BaseModel):
    """Query parameters for listing nodes within a graph.

    Attributes:
        node_type: Filter by node type.
        country_code: Filter by country code.
        risk_level: Filter by risk level.
        compliance_status: Filter by compliance status.
        tier_depth: Filter by tier depth.
        limit: Maximum results to return.
        offset: Pagination offset.
    """

    model_config = ConfigDict(extra="forbid")

    node_type: Optional[NodeType] = Field(None)
    country_code: Optional[str] = Field(None, min_length=2, max_length=2)
    risk_level: Optional[RiskLevel] = Field(None)
    compliance_status: Optional[ComplianceStatus] = Field(None)
    tier_depth: Optional[int] = Field(None, ge=0)
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)


class EdgeQueryParams(BaseModel):
    """Query parameters for listing edges within a graph.

    Attributes:
        commodity: Filter by commodity.
        transfer_date_from: Filter edges after this date.
        transfer_date_to: Filter edges before this date.
        custody_model: Filter by custody model.
        limit: Maximum results to return.
        offset: Pagination offset.
    """

    model_config = ConfigDict(extra="forbid")

    commodity: Optional[EUDRCommodity] = Field(None)
    transfer_date_from: Optional[datetime] = Field(None)
    transfer_date_to: Optional[datetime] = Field(None)
    custody_model: Optional[CustodyModel] = Field(None)
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)


# =============================================================================
# Response Models
# =============================================================================


class TraceResult(BaseModel):
    """Result of a forward or backward trace through the supply chain.

    Attributes:
        trace_id: Unique identifier for this trace operation.
        direction: Trace direction (forward or backward).
        start_node_id: Node where the trace began.
        visited_nodes: Ordered list of node IDs visited during trace.
        visited_edges: Ordered list of edge IDs traversed during trace.
        origin_plot_ids: Plot IDs found at the terminal producer nodes
            (populated in backward traces).
        trace_depth: Maximum depth reached during trace.
        total_quantity: Aggregate quantity traced (Decimal).
        is_complete: Whether the trace reached terminal nodes without
            encountering broken chains.
        broken_at: List of node IDs where the trace was interrupted.
        processing_time_ms: Wall-clock time for the trace in ms.
    """

    model_config = ConfigDict(from_attributes=True)

    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this trace operation",
    )
    direction: str = Field(
        ...,
        description="Trace direction: 'forward' or 'backward'",
    )
    start_node_id: str = Field(
        ...,
        description="Node where the trace began",
    )
    visited_nodes: List[str] = Field(
        default_factory=list,
        description="Ordered list of visited node IDs",
    )
    visited_edges: List[str] = Field(
        default_factory=list,
        description="Ordered list of traversed edge IDs",
    )
    origin_plot_ids: List[str] = Field(
        default_factory=list,
        description="Plot IDs at terminal producer nodes",
    )
    trace_depth: int = Field(
        default=0,
        ge=0,
        description="Maximum depth reached during trace",
    )
    total_quantity: Optional[Decimal] = Field(
        None,
        description="Aggregate quantity traced",
    )
    is_complete: bool = Field(
        default=True,
        description="Whether trace reached terminal nodes",
    )
    broken_at: List[str] = Field(
        default_factory=list,
        description="Node IDs where the trace was interrupted",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock time for the trace in milliseconds",
    )


class TierDistribution(BaseModel):
    """Distribution of supply chain nodes by tier depth.

    Attributes:
        tier_counts: Dictionary mapping tier depth to node count.
        max_depth: Maximum tier depth in the graph.
        average_depth: Average tier depth across all nodes.
        median_depth: Median tier depth.
    """

    model_config = ConfigDict(from_attributes=True)

    tier_counts: Dict[int, int] = Field(
        default_factory=dict,
        description="Tier depth -> node count mapping",
    )
    max_depth: int = Field(default=0, ge=0)
    average_depth: float = Field(default=0.0, ge=0.0)
    median_depth: float = Field(default=0.0, ge=0.0)


class RiskSummary(BaseModel):
    """Aggregated risk summary for a supply chain graph.

    Attributes:
        graph_id: Graph this summary belongs to.
        total_nodes: Total nodes assessed.
        risk_distribution: Count by risk level.
        average_risk_score: Mean risk score across all nodes.
        max_risk_score: Highest individual node risk score.
        high_risk_nodes: List of node IDs with HIGH risk.
        risk_concentration: Top 5 nodes contributing most downstream risk.
        propagation_results: Individual node propagation results.
    """

    model_config = ConfigDict(from_attributes=True)

    graph_id: str = Field(..., description="Graph this summary belongs to")
    total_nodes: int = Field(default=0, ge=0)
    risk_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"low": 0, "standard": 0, "high": 0},
    )
    average_risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    max_risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    high_risk_nodes: List[str] = Field(default_factory=list)
    risk_concentration: List[Dict[str, Any]] = Field(default_factory=list)
    propagation_results: List[RiskPropagationResult] = Field(
        default_factory=list,
    )


class GapAnalysisResult(BaseModel):
    """Complete gap analysis result for a supply chain graph.

    Attributes:
        graph_id: Graph analyzed.
        total_gaps: Total number of gaps detected.
        gaps_by_severity: Count by severity level.
        gaps_by_type: Count by gap type.
        compliance_readiness: Compliance readiness score (0-100).
        gaps: List of individual gap records.
        remediation_priority: Ordered list of gap IDs by priority.
        analysis_timestamp: When the analysis was performed.
    """

    model_config = ConfigDict(from_attributes=True)

    graph_id: str = Field(..., description="Graph analyzed")
    total_gaps: int = Field(default=0, ge=0)
    gaps_by_severity: Dict[str, int] = Field(
        default_factory=lambda: {
            "critical": 0, "high": 0, "medium": 0, "low": 0,
        },
    )
    gaps_by_type: Dict[str, int] = Field(default_factory=dict)
    compliance_readiness: float = Field(default=0.0, ge=0.0, le=100.0)
    gaps: List[SupplyChainGap] = Field(default_factory=list)
    remediation_priority: List[str] = Field(default_factory=list)
    analysis_timestamp: datetime = Field(default_factory=_utcnow)


class DDSExportData(BaseModel):
    """Supply chain data formatted for Due Diligence Statement export.

    Contains the supply chain section of a DDS as required by
    EUDR Article 4(2)(f), including node summaries, traceability
    metrics, and provenance hashes.

    Attributes:
        graph_id: Source graph ID.
        operator_id: Operator filing the DDS.
        commodity: EUDR commodity covered.
        total_supply_chain_actors: Total actors in the chain.
        tier_depth: Maximum supply chain depth.
        traceability_score: Plot-to-product traceability percentage.
        origin_countries: List of origin country codes.
        origin_plot_count: Number of linked origin plots.
        custody_transfers_count: Number of custody transfers.
        risk_level: Overall supply chain risk level.
        compliance_readiness: Compliance readiness score.
        provenance_hash: SHA-256 hash of the export data.
        export_timestamp: When the export was generated.
        supply_chain_summary: Structured summary for DDS inclusion.
    """

    model_config = ConfigDict(from_attributes=True)

    graph_id: str = Field(..., description="Source graph ID")
    operator_id: str = Field(..., description="Operator filing the DDS")
    commodity: EUDRCommodity = Field(..., description="EUDR commodity")
    total_supply_chain_actors: int = Field(default=0, ge=0)
    tier_depth: int = Field(default=0, ge=0)
    traceability_score: float = Field(default=0.0, ge=0.0, le=100.0)
    origin_countries: List[str] = Field(default_factory=list)
    origin_plot_count: int = Field(default=0, ge=0)
    custody_transfers_count: int = Field(default=0, ge=0)
    risk_level: RiskLevel = Field(default=RiskLevel.STANDARD)
    compliance_readiness: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    export_timestamp: datetime = Field(default_factory=_utcnow)
    supply_chain_summary: Dict[str, Any] = Field(default_factory=dict)


class GraphLayoutData(BaseModel):
    """Graph layout data for frontend visualization rendering.

    Provides node positions, edge paths, and styling metadata for
    rendering the supply chain graph using D3.js, vis-network, or
    Cytoscape.js in the GL-EUDR-APP frontend.

    Attributes:
        graph_id: Source graph ID.
        layout_algorithm: Algorithm used (force-directed, hierarchical).
        node_positions: Mapping of node_id -> (x, y) coordinates.
        edge_paths: Mapping of edge_id -> list of (x, y) waypoints.
        node_styles: Mapping of node_id -> style metadata.
        edge_styles: Mapping of edge_id -> style metadata.
        viewport: Bounding box for the layout.
    """

    model_config = ConfigDict(from_attributes=True)

    graph_id: str = Field(..., description="Source graph ID")
    layout_algorithm: str = Field(
        default="force_directed",
        description="Layout algorithm used",
    )
    node_positions: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
    )
    edge_paths: Dict[str, List[Tuple[float, float]]] = Field(
        default_factory=dict,
    )
    node_styles: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
    )
    edge_styles: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
    )
    viewport: Dict[str, float] = Field(
        default_factory=lambda: {
            "min_x": 0.0, "min_y": 0.0, "max_x": 1000.0, "max_y": 1000.0,
        },
    )


class SankeyData(BaseModel):
    """Sankey diagram data for commodity flow visualization.

    Attributes:
        graph_id: Source graph ID.
        nodes: List of Sankey nodes with labels and values.
        links: List of Sankey links with source, target, and value.
    """

    model_config = ConfigDict(from_attributes=True)

    graph_id: str = Field(..., description="Source graph ID")
    nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sankey nodes with label and value",
    )
    links: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sankey links with source, target, value",
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Constants
    "VERSION",
    "MAX_NODES_PER_GRAPH",
    "MAX_EDGES_PER_GRAPH",
    "MAX_TIER_DEPTH",
    "EUDR_DEFORESTATION_CUTOFF",
    "DEFAULT_RISK_WEIGHTS",
    "DERIVED_TO_PRIMARY",
    "PRIMARY_COMMODITIES",
    "GAP_SEVERITY_MAP",
    "GAP_ARTICLE_MAP",
    # Enumerations
    "NodeType",
    "EUDRCommodity",
    "CustodyModel",
    "RiskLevel",
    "ComplianceStatus",
    "GapType",
    "GapSeverity",
    "TransportMode",
    # Core models
    "SupplyChainNode",
    "SupplyChainEdge",
    "SupplyChainGraph",
    "SupplyChainGap",
    "RiskPropagationResult",
    # Request models
    "CreateGraphRequest",
    "CreateNodeRequest",
    "CreateEdgeRequest",
    "UpdateNodeRequest",
    # Query parameter models
    "GraphQueryParams",
    "NodeQueryParams",
    "EdgeQueryParams",
    # Response models
    "TraceResult",
    "TierDistribution",
    "RiskSummary",
    "GapAnalysisResult",
    "DDSExportData",
    "GraphLayoutData",
    "SankeyData",
]
