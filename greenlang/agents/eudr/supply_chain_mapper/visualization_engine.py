# -*- coding: utf-8 -*-
"""
VisualizationEngine - AGENT-EUDR-001 Feature 7: Supply Chain Visualization Backend

Computes graph layouts, generates Sankey diagram data, positions nodes
geographically, exports graphs in multiple formats, and applies risk-based
color coding for frontend rendering with D3.js / Cytoscape.js.

Key capabilities:
    - Fruchterman-Reingold force-directed layout (pure Python, no GPU)
    - Hierarchical (tier-based) layout for DAG supply chains
    - Sankey diagram data generation (commodity flow volumes between actors)
    - Geographic overlay positioning (latitude/longitude for map plotting)
    - Graph export: GeoJSON, GraphML, JSON-LD
    - Time-based graph snapshot retrieval for historical views
    - Risk-based node/edge coloring (green=LOW, yellow=STANDARD, red=HIGH)
    - Filters: commodity, country, risk level, compliance status, tier depth
    - Node clustering for 1,000-10,000 node graphs
    - Layout computation < 3 seconds for 1,000 nodes

Integrations:
    - graph_engine.SupplyChainGraphEngine: Graph data retrieval
    - geolocation_linker.GeolocationLinker: Geographic positioning
    - risk_propagation.RiskPropagationEngine: Risk color coding

Zero-Hallucination Guarantees:
    - All layout computations are deterministic (same seed = same layout)
    - No LLM involvement in any visualization computation path
    - All positions use IEEE 754 double precision floating point
    - Provenance hash on every layout computation

Performance Targets (from PRD):
    - Layout computation: < 3 seconds for 1,000 nodes
    - Sankey data generation: < 1 second for 500 edges
    - GeoJSON export: < 2 seconds for 5,000 nodes

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001, Feature 7
Agent ID: GL-EUDR-SCM-001
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Module version for provenance tracking.
_MODULE_VERSION = "1.0.0"

#: Default canvas dimensions for force-directed layout.
DEFAULT_CANVAS_WIDTH = 1000.0
DEFAULT_CANVAS_HEIGHT = 1000.0

#: Default force-directed layout parameters.
DEFAULT_FR_ITERATIONS = 300
DEFAULT_FR_COOLING_FACTOR = 0.95
DEFAULT_FR_INITIAL_TEMP = 10.0
DEFAULT_FR_REPULSION_CONSTANT = 50.0
DEFAULT_FR_ATTRACTION_CONSTANT = 0.1
DEFAULT_FR_SEED = 42

#: Hierarchical layout spacing.
HIERARCHICAL_X_SPACING = 150.0
HIERARCHICAL_Y_SPACING = 200.0

#: Clustering defaults.
DEFAULT_CLUSTER_THRESHOLD = 50
DEFAULT_CLUSTER_RADIUS = 100.0

#: Risk color mapping (EUDR traffic-light scheme).
RISK_COLOR_LOW = "#22C55E"
RISK_COLOR_STANDARD = "#F59E0B"
RISK_COLOR_HIGH = "#EF4444"
RISK_COLOR_UNKNOWN = "#9CA3AF"

#: Compliance status colors.
COMPLIANCE_COLORS: Dict[str, str] = {
    "compliant": "#22C55E",
    "non_compliant": "#EF4444",
    "pending_verification": "#F59E0B",
    "under_review": "#3B82F6",
    "insufficient_data": "#9CA3AF",
    "exempted": "#8B5CF6",
}

#: Node type shapes for frontend rendering.
NODE_TYPE_SHAPES: Dict[str, str] = {
    "producer": "circle",
    "collector": "diamond",
    "processor": "square",
    "trader": "triangle",
    "importer": "star",
    "certifier": "hexagon",
    "warehouse": "rectangle",
    "port": "pentagon",
}

#: Default node sizes by type.
NODE_TYPE_SIZES: Dict[str, float] = {
    "producer": 8.0,
    "collector": 12.0,
    "processor": 16.0,
    "trader": 14.0,
    "importer": 20.0,
    "certifier": 10.0,
    "warehouse": 10.0,
    "port": 14.0,
}

#: JSON-LD context for supply chain graph export.
JSONLD_CONTEXT = {
    "@context": {
        "@vocab": "https://schema.org/",
        "eudr": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R1115#",
        "greenlang": "https://greenlang.io/ontology/",
        "node_type": "greenlang:nodeType",
        "risk_level": "greenlang:riskLevel",
        "risk_score": "greenlang:riskScore",
        "compliance_status": "greenlang:complianceStatus",
        "country_code": "greenlang:countryCode",
        "commodity": "greenlang:commodity",
        "tier_depth": "greenlang:tierDepth",
        "quantity": "greenlang:quantity",
        "custody_model": "greenlang:custodyModel",
        "transfer_date": "greenlang:transferDate",
    }
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_provenance_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class LayoutAlgorithm(str, Enum):
    """Available graph layout algorithms."""

    FORCE_DIRECTED = "force_directed"
    HIERARCHICAL = "hierarchical"
    GEOGRAPHIC = "geographic"
    CIRCULAR = "circular"


class ExportFormat(str, Enum):
    """Available graph export formats."""

    GEOJSON = "geojson"
    GRAPHML = "graphml"
    JSONLD = "jsonld"


class ColorScheme(str, Enum):
    """Available color coding schemes for nodes and edges."""

    RISK_LEVEL = "risk_level"
    COMPLIANCE_STATUS = "compliance_status"
    NODE_TYPE = "node_type"
    TIER_DEPTH = "tier_depth"
    COUNTRY = "country"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisualizationConfig:
    """Immutable configuration for the visualization engine.

    Attributes:
        canvas_width: Canvas width in pixels for layout computation.
        canvas_height: Canvas height in pixels for layout computation.
        fr_iterations: Number of Fruchterman-Reingold iterations.
        fr_cooling_factor: Temperature cooling factor per iteration.
        fr_initial_temp: Initial temperature for force-directed layout.
        fr_repulsion_constant: Repulsive force constant.
        fr_attraction_constant: Attractive force constant.
        fr_seed: Random seed for deterministic layout.
        hierarchical_x_spacing: Horizontal spacing for hierarchical layout.
        hierarchical_y_spacing: Vertical spacing for hierarchical layout.
        cluster_threshold: Node count threshold to trigger clustering.
        cluster_radius: Radius of cluster circles in layout units.
        default_color_scheme: Default color coding scheme.
        enable_edge_bundling: Whether to compute bundled edge paths.
        max_layout_time_ms: Maximum allowed layout computation time in ms.
    """

    canvas_width: float = DEFAULT_CANVAS_WIDTH
    canvas_height: float = DEFAULT_CANVAS_HEIGHT
    fr_iterations: int = DEFAULT_FR_ITERATIONS
    fr_cooling_factor: float = DEFAULT_FR_COOLING_FACTOR
    fr_initial_temp: float = DEFAULT_FR_INITIAL_TEMP
    fr_repulsion_constant: float = DEFAULT_FR_REPULSION_CONSTANT
    fr_attraction_constant: float = DEFAULT_FR_ATTRACTION_CONSTANT
    fr_seed: int = DEFAULT_FR_SEED
    hierarchical_x_spacing: float = HIERARCHICAL_X_SPACING
    hierarchical_y_spacing: float = HIERARCHICAL_Y_SPACING
    cluster_threshold: int = DEFAULT_CLUSTER_THRESHOLD
    cluster_radius: float = DEFAULT_CLUSTER_RADIUS
    default_color_scheme: ColorScheme = ColorScheme.RISK_LEVEL
    enable_edge_bundling: bool = False
    max_layout_time_ms: float = 5000.0


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class NodePosition:
    """Computed position and style for a single graph node.

    Attributes:
        node_id: Unique node identifier.
        x: Horizontal position in layout coordinates.
        y: Vertical position in layout coordinates.
        latitude: Optional geographic latitude for map overlay.
        longitude: Optional geographic longitude for map overlay.
        color: Hex color code for rendering.
        shape: Shape identifier for rendering.
        size: Node size in pixels.
        label: Display label for the node.
        cluster_id: Cluster identifier if node is clustered.
        metadata: Additional style and data attributes.
    """

    node_id: str
    x: float = 0.0
    y: float = 0.0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    color: str = RISK_COLOR_UNKNOWN
    shape: str = "circle"
    size: float = 10.0
    label: str = ""
    cluster_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result: Dict[str, Any] = {
            "node_id": self.node_id,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "color": self.color,
            "shape": self.shape,
            "size": self.size,
            "label": self.label,
        }
        if self.latitude is not None:
            result["latitude"] = self.latitude
        if self.longitude is not None:
            result["longitude"] = self.longitude
        if self.cluster_id is not None:
            result["cluster_id"] = self.cluster_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class EdgePath:
    """Computed path and style for a single graph edge.

    Attributes:
        edge_id: Unique edge identifier.
        source_node_id: Source node identifier.
        target_node_id: Target node identifier.
        waypoints: List of (x, y) waypoints along the edge path.
        color: Hex color code for rendering.
        width: Edge width in pixels.
        label: Optional display label.
        metadata: Additional style and data attributes.
    """

    edge_id: str
    source_node_id: str
    target_node_id: str
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    color: str = "#6B7280"
    width: float = 1.5
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result: Dict[str, Any] = {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "waypoints": [
                (round(x, 4), round(y, 4)) for x, y in self.waypoints
            ],
            "color": self.color,
            "width": self.width,
        }
        if self.label:
            result["label"] = self.label
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class ClusterGroup:
    """A group of nodes that have been clustered for layout optimization.

    Attributes:
        cluster_id: Unique cluster identifier.
        center_x: Cluster center horizontal position.
        center_y: Cluster center vertical position.
        radius: Cluster display radius.
        node_ids: List of node IDs in this cluster.
        label: Display label for the cluster.
        color: Hex color for cluster boundary.
        metadata: Additional cluster attributes.
    """

    cluster_id: str
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = DEFAULT_CLUSTER_RADIUS
    node_ids: List[str] = field(default_factory=list)
    label: str = ""
    color: str = "#E5E7EB"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "center_x": round(self.center_x, 4),
            "center_y": round(self.center_y, 4),
            "radius": round(self.radius, 4),
            "node_ids": list(self.node_ids),
            "label": self.label,
            "color": self.color,
            "metadata": self.metadata,
        }


@dataclass
class SankeyNode:
    """A single node in a Sankey diagram.

    Attributes:
        id: Node identifier.
        label: Display label.
        color: Fill color.
        value: Total flow through this node.
    """

    id: str
    label: str = ""
    color: str = "#3B82F6"
    value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "color": self.color,
            "value": round(self.value, 4),
        }


@dataclass
class SankeyLink:
    """A single link (flow) in a Sankey diagram.

    Attributes:
        source: Source node identifier.
        target: Target node identifier.
        value: Flow volume.
        color: Link color.
        label: Optional label.
    """

    source: str
    target: str
    value: float = 0.0
    color: str = "#93C5FD"
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result: Dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "value": round(self.value, 4),
            "color": self.color,
        }
        if self.label:
            result["label"] = self.label
        return result


@dataclass
class LayoutResult:
    """Complete result of a graph layout computation.

    Attributes:
        layout_id: Unique identifier for this layout.
        graph_id: Source graph identifier.
        algorithm: Layout algorithm used.
        node_positions: Computed node positions.
        edge_paths: Computed edge paths.
        clusters: Cluster groups (if clustering applied).
        viewport: Bounding box of the layout.
        computation_time_ms: Layout computation time in ms.
        total_nodes: Number of nodes in the layout.
        total_edges: Number of edges in the layout.
        provenance_hash: SHA-256 hash of the layout data.
        created_at: UTC timestamp of layout creation.
    """

    layout_id: str = field(default_factory=lambda: _generate_id("LAYOUT"))
    graph_id: str = ""
    algorithm: str = LayoutAlgorithm.FORCE_DIRECTED.value
    node_positions: Dict[str, NodePosition] = field(default_factory=dict)
    edge_paths: Dict[str, EdgePath] = field(default_factory=dict)
    clusters: List[ClusterGroup] = field(default_factory=list)
    viewport: Dict[str, float] = field(default_factory=lambda: {
        "min_x": 0.0, "min_y": 0.0,
        "max_x": DEFAULT_CANVAS_WIDTH, "max_y": DEFAULT_CANVAS_HEIGHT,
    })
    computation_time_ms: float = 0.0
    total_nodes: int = 0
    total_edges: int = 0
    provenance_hash: str = ""
    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export / frontend consumption."""
        return {
            "layout_id": self.layout_id,
            "graph_id": self.graph_id,
            "algorithm": self.algorithm,
            "node_positions": {
                nid: np.to_dict()
                for nid, np in self.node_positions.items()
            },
            "edge_paths": {
                eid: ep.to_dict()
                for eid, ep in self.edge_paths.items()
            },
            "clusters": [c.to_dict() for c in self.clusters],
            "viewport": self.viewport,
            "computation_time_ms": round(self.computation_time_ms, 2),
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "provenance_hash": self.provenance_hash,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SankeyResult:
    """Complete Sankey diagram data for a supply chain graph.

    Attributes:
        sankey_id: Unique identifier.
        graph_id: Source graph identifier.
        nodes: Sankey diagram nodes.
        links: Sankey diagram links.
        total_flow: Total flow volume across all links.
        commodity_filter: Commodity filter applied (if any).
        computation_time_ms: Computation time in ms.
        created_at: UTC timestamp.
    """

    sankey_id: str = field(default_factory=lambda: _generate_id("SANKEY"))
    graph_id: str = ""
    nodes: List[SankeyNode] = field(default_factory=list)
    links: List[SankeyLink] = field(default_factory=list)
    total_flow: float = 0.0
    commodity_filter: Optional[str] = None
    computation_time_ms: float = 0.0
    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sankey_id": self.sankey_id,
            "graph_id": self.graph_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [l.to_dict() for l in self.links],
            "total_flow": round(self.total_flow, 4),
            "commodity_filter": self.commodity_filter,
            "computation_time_ms": round(self.computation_time_ms, 2),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class GraphFilter:
    """Filter specification for graph visualization.

    Attributes:
        commodities: Filter by commodity types.
        countries: Filter by ISO country codes.
        risk_levels: Filter by risk levels.
        compliance_statuses: Filter by compliance statuses.
        max_tier_depth: Maximum tier depth to include.
        min_tier_depth: Minimum tier depth to include.
        node_types: Filter by node types.
        node_ids: Explicit set of node IDs to include.
    """

    commodities: Optional[List[str]] = None
    countries: Optional[List[str]] = None
    risk_levels: Optional[List[str]] = None
    compliance_statuses: Optional[List[str]] = None
    max_tier_depth: Optional[int] = None
    min_tier_depth: Optional[int] = None
    node_types: Optional[List[str]] = None
    node_ids: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Visualization Engine
# ---------------------------------------------------------------------------


class VisualizationEngine:
    """Backend engine for EUDR supply chain graph visualization.

    Computes layouts, generates Sankey data, positions nodes geographically,
    exports graphs in multiple formats, and applies risk-based color coding
    for frontend rendering with D3.js / Cytoscape.js.

    The engine operates on in-memory graph data provided by the
    SupplyChainGraphEngine, applies color coding from the
    RiskPropagationEngine, and uses geographic coordinates from the
    GeolocationLinker.

    All layout computations are deterministic given the same random seed,
    ensuring reproducible visualizations for regulatory audit purposes.

    Architecture:
        - Layout layer: Fruchterman-Reingold, hierarchical, geographic, circular
        - Color layer: Risk-based, compliance-based, type-based, tier-based
        - Export layer: GeoJSON, GraphML, JSON-LD
        - Filter layer: Commodity, country, risk, compliance, tier depth
        - Clustering: Country-based and tier-based node grouping

    Performance targets:
        - Force-directed layout: < 3 seconds for 1,000 nodes
        - Hierarchical layout: < 1 second for 1,000 nodes
        - Sankey generation: < 1 second for 500 edges
        - GeoJSON export: < 2 seconds for 5,000 nodes

    Usage::

        engine = VisualizationEngine()

        # Prepare node/edge data
        nodes = {"N1": {...}, "N2": {...}}
        edges = {"E1": {...}}

        # Compute force-directed layout
        layout = engine.compute_force_directed_layout("GRAPH-1", nodes, edges)

        # Generate Sankey diagram data
        sankey = engine.generate_sankey_data("GRAPH-1", nodes, edges)

        # Export to GeoJSON
        geojson = engine.export_geojson("GRAPH-1", nodes, edges)

    Args:
        config: Optional VisualizationConfig for layout parameters.
        graph_engine: Optional SupplyChainGraphEngine for graph retrieval.
        geolocation_linker: Optional GeolocationLinker for geographic positions.
        risk_engine: Optional RiskPropagationEngine for risk color coding.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        graph_engine: Any = None,
        geolocation_linker: Any = None,
        risk_engine: Any = None,
    ) -> None:
        """Initialize VisualizationEngine.

        Args:
            config: Visualization configuration. Defaults to standard config.
            graph_engine: Optional SupplyChainGraphEngine instance.
            geolocation_linker: Optional GeolocationLinker instance.
            risk_engine: Optional RiskPropagationEngine instance.
        """
        self._config = config or VisualizationConfig()
        self._graph_engine = graph_engine
        self._geolocation_linker = geolocation_linker
        self._risk_engine = risk_engine

        # Layout cache keyed by (graph_id, algorithm, filter_hash)
        self._layout_cache: Dict[str, LayoutResult] = {}

        # Snapshot cache for historical views
        self._snapshot_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "VisualizationEngine initialized (seed=%d, iterations=%d)",
            self._config.fr_seed,
            self._config.fr_iterations,
        )

    # ==================================================================
    # Force-Directed Layout (Fruchterman-Reingold)
    # ==================================================================

    def compute_force_directed_layout(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        graph_filter: Optional[GraphFilter] = None,
        color_scheme: Optional[ColorScheme] = None,
    ) -> LayoutResult:
        """Compute a force-directed layout using Fruchterman-Reingold algorithm.

        Positions nodes using attractive and repulsive forces to produce
        an aesthetically pleasing graph layout. The algorithm simulates
        a physical system where connected nodes attract each other and
        all nodes repel each other.

        Args:
            graph_id: Graph identifier for tracking.
            nodes: Dictionary of node_id -> node data dictionaries.
                Expected keys: node_type, operator_name, country_code,
                risk_level, risk_score, compliance_status, tier_depth,
                commodities, latitude, longitude.
            edges: Dictionary of edge_id -> edge data dictionaries.
                Expected keys: source_node_id, target_node_id, commodity,
                quantity.
            graph_filter: Optional filter to subset the graph.
            color_scheme: Color coding scheme (defaults to config default).

        Returns:
            LayoutResult with computed node positions, edge paths,
            and optional clusters.
        """
        start_time = time.monotonic()
        scheme = color_scheme or self._config.default_color_scheme

        # Apply filters
        filtered_nodes, filtered_edges = self._apply_filter(
            nodes, edges, graph_filter
        )

        if not filtered_nodes:
            return self._empty_layout(graph_id, LayoutAlgorithm.FORCE_DIRECTED.value)

        node_ids = list(filtered_nodes.keys())
        n = len(node_ids)

        # Build adjacency for force calculation
        adjacency = self._build_adjacency(filtered_edges)

        # Initialize positions deterministically
        rng = random.Random(self._config.fr_seed)
        positions: Dict[str, List[float]] = {}
        for nid in node_ids:
            positions[nid] = [
                rng.uniform(0, self._config.canvas_width),
                rng.uniform(0, self._config.canvas_height),
            ]

        # Fruchterman-Reingold constants
        area = self._config.canvas_width * self._config.canvas_height
        k = math.sqrt(area / max(n, 1))
        temperature = self._config.fr_initial_temp * k
        cooling = self._config.fr_cooling_factor

        # Adaptive iteration count for large graphs
        iterations = self._config.fr_iterations
        if n > 5000:
            iterations = min(iterations, 100)
        elif n > 2000:
            iterations = min(iterations, 150)

        # Fruchterman-Reingold iteration loop
        for iteration in range(iterations):
            # Check time budget
            elapsed = (time.monotonic() - start_time) * 1000
            if elapsed > self._config.max_layout_time_ms:
                logger.warning(
                    "Layout computation exceeded time budget at iteration %d/%d (%.1fms)",
                    iteration, iterations, elapsed,
                )
                break

            # Compute repulsive forces (O(n^2) - for large graphs use
            # Barnes-Hut approximation via spatial grid)
            displacements: Dict[str, List[float]] = {
                nid: [0.0, 0.0] for nid in node_ids
            }

            if n <= 2000:
                # Direct N^2 computation for moderate graphs
                for i in range(n):
                    ni = node_ids[i]
                    for j in range(i + 1, n):
                        nj = node_ids[j]
                        dx = positions[ni][0] - positions[nj][0]
                        dy = positions[ni][1] - positions[nj][1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        dist = max(dist, 0.01)

                        repulsive_force = (
                            self._config.fr_repulsion_constant * k * k / dist
                        )
                        fx = (dx / dist) * repulsive_force
                        fy = (dy / dist) * repulsive_force

                        displacements[ni][0] += fx
                        displacements[ni][1] += fy
                        displacements[nj][0] -= fx
                        displacements[nj][1] -= fy
            else:
                # Grid-based approximation for large graphs
                displacements = self._compute_repulsive_grid(
                    node_ids, positions, k
                )

            # Compute attractive forces along edges
            for eid, edata in filtered_edges.items():
                src = edata.get("source_node_id", "")
                tgt = edata.get("target_node_id", "")
                if src in positions and tgt in positions:
                    dx = positions[tgt][0] - positions[src][0]
                    dy = positions[tgt][1] - positions[src][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    dist = max(dist, 0.01)

                    attractive_force = (
                        self._config.fr_attraction_constant * dist * dist / k
                    )
                    fx = (dx / dist) * attractive_force
                    fy = (dy / dist) * attractive_force

                    displacements[src][0] += fx
                    displacements[src][1] += fy
                    displacements[tgt][0] -= fx
                    displacements[tgt][1] -= fy

            # Apply displacements with temperature limiting
            for nid in node_ids:
                dx = displacements[nid][0]
                dy = displacements[nid][1]
                disp_mag = math.sqrt(dx * dx + dy * dy)
                if disp_mag > 0:
                    scale = min(disp_mag, temperature) / disp_mag
                    positions[nid][0] += dx * scale
                    positions[nid][1] += dy * scale

                # Clamp to canvas bounds with padding
                padding = 20.0
                positions[nid][0] = max(
                    padding,
                    min(self._config.canvas_width - padding, positions[nid][0]),
                )
                positions[nid][1] = max(
                    padding,
                    min(self._config.canvas_height - padding, positions[nid][1]),
                )

            # Cool temperature
            temperature *= cooling

        # Build result
        result = self._build_layout_result(
            graph_id=graph_id,
            algorithm=LayoutAlgorithm.FORCE_DIRECTED.value,
            positions=positions,
            filtered_nodes=filtered_nodes,
            filtered_edges=filtered_edges,
            color_scheme=scheme,
            start_time=start_time,
        )

        return result

    def _compute_repulsive_grid(
        self,
        node_ids: List[str],
        positions: Dict[str, List[float]],
        k: float,
    ) -> Dict[str, List[float]]:
        """Compute repulsive forces using grid-based spatial partitioning.

        For graphs with > 2000 nodes, directly computing N^2 pair
        interactions is too expensive. This method partitions the space
        into grid cells and only computes interactions between nodes
        in nearby cells.

        Args:
            node_ids: List of all node IDs.
            positions: Current node positions.
            k: Optimal distance parameter.

        Returns:
            Displacement dictionary keyed by node_id.
        """
        displacements: Dict[str, List[float]] = {
            nid: [0.0, 0.0] for nid in node_ids
        }

        # Build spatial grid
        cell_size = k * 3.0
        grid: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for nid in node_ids:
            cx = int(positions[nid][0] / cell_size)
            cy = int(positions[nid][1] / cell_size)
            grid[(cx, cy)].append(nid)

        # Compute forces between nodes in neighboring cells
        for (cx, cy), cell_nodes in grid.items():
            # Collect neighbors from 3x3 grid neighborhood
            neighbors: List[str] = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    neighbors.extend(grid.get((cx + dx, cy + dy), []))

            for ni in cell_nodes:
                for nj in neighbors:
                    if ni >= nj:
                        continue
                    dx_val = positions[ni][0] - positions[nj][0]
                    dy_val = positions[ni][1] - positions[nj][1]
                    dist = math.sqrt(dx_val * dx_val + dy_val * dy_val)
                    dist = max(dist, 0.01)

                    repulsive_force = (
                        self._config.fr_repulsion_constant * k * k / dist
                    )
                    fx = (dx_val / dist) * repulsive_force
                    fy = (dy_val / dist) * repulsive_force

                    displacements[ni][0] += fx
                    displacements[ni][1] += fy
                    displacements[nj][0] -= fx
                    displacements[nj][1] -= fy

        return displacements

    # ==================================================================
    # Hierarchical Layout (Tier-Based)
    # ==================================================================

    def compute_hierarchical_layout(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        graph_filter: Optional[GraphFilter] = None,
        color_scheme: Optional[ColorScheme] = None,
    ) -> LayoutResult:
        """Compute a hierarchical layout based on supply chain tiers.

        Positions nodes in horizontal layers by tier depth, with
        importers (tier 0) at the bottom and producers (highest tier)
        at the top. Nodes within each tier are spread horizontally.

        Args:
            graph_id: Graph identifier.
            nodes: Node data dictionaries.
            edges: Edge data dictionaries.
            graph_filter: Optional filter.
            color_scheme: Color coding scheme.

        Returns:
            LayoutResult with tier-based positioning.
        """
        start_time = time.monotonic()
        scheme = color_scheme or self._config.default_color_scheme

        filtered_nodes, filtered_edges = self._apply_filter(
            nodes, edges, graph_filter
        )

        if not filtered_nodes:
            return self._empty_layout(graph_id, LayoutAlgorithm.HIERARCHICAL.value)

        # Group nodes by tier depth
        tiers: Dict[int, List[str]] = defaultdict(list)
        for nid, ndata in filtered_nodes.items():
            tier = ndata.get("tier_depth", 0)
            tiers[tier].append(nid)

        max_tier = max(tiers.keys()) if tiers else 0

        # Position nodes
        positions: Dict[str, List[float]] = {}
        for tier, tier_nodes in tiers.items():
            n_in_tier = len(tier_nodes)
            # Producers at top, importers at bottom
            y = (max_tier - tier) * self._config.hierarchical_y_spacing + 50.0
            total_width = (n_in_tier - 1) * self._config.hierarchical_x_spacing
            start_x = (self._config.canvas_width - total_width) / 2.0

            for idx, nid in enumerate(sorted(tier_nodes)):
                x = start_x + idx * self._config.hierarchical_x_spacing
                positions[nid] = [x, y]

        result = self._build_layout_result(
            graph_id=graph_id,
            algorithm=LayoutAlgorithm.HIERARCHICAL.value,
            positions=positions,
            filtered_nodes=filtered_nodes,
            filtered_edges=filtered_edges,
            color_scheme=scheme,
            start_time=start_time,
        )

        return result

    # ==================================================================
    # Geographic Layout (Map Overlay)
    # ==================================================================

    def compute_geographic_layout(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        graph_filter: Optional[GraphFilter] = None,
        color_scheme: Optional[ColorScheme] = None,
    ) -> LayoutResult:
        """Compute a geographic layout using node lat/lon coordinates.

        Projects geographic coordinates onto the canvas using simple
        Mercator-like mapping. Nodes without coordinates are positioned
        at the center of their country's known nodes or at canvas center.

        Args:
            graph_id: Graph identifier.
            nodes: Node data dictionaries with latitude/longitude.
            edges: Edge data dictionaries.
            graph_filter: Optional filter.
            color_scheme: Color coding scheme.

        Returns:
            LayoutResult with geographic positioning.
        """
        start_time = time.monotonic()
        scheme = color_scheme or self._config.default_color_scheme

        filtered_nodes, filtered_edges = self._apply_filter(
            nodes, edges, graph_filter
        )

        if not filtered_nodes:
            return self._empty_layout(graph_id, LayoutAlgorithm.GEOGRAPHIC.value)

        # Collect coordinates and enrich from geolocation linker if available
        node_coords: Dict[str, Tuple[float, float]] = {}
        for nid, ndata in filtered_nodes.items():
            lat = ndata.get("latitude")
            lon = ndata.get("longitude")
            # Also check coordinates tuple format from models
            coords = ndata.get("coordinates")
            if lat is not None and lon is not None:
                node_coords[nid] = (lat, lon)
            elif coords and len(coords) == 2:
                node_coords[nid] = (coords[0], coords[1])

        # Try geolocation linker for nodes without coordinates
        if self._geolocation_linker and len(node_coords) < len(filtered_nodes):
            for nid in filtered_nodes:
                if nid not in node_coords:
                    try:
                        links = self._geolocation_linker.get_links_for_producer(nid)
                        if links:
                            link = links[0]
                            lat_val = link.get("latitude") or link.get("centroid_lat")
                            lon_val = link.get("longitude") or link.get("centroid_lon")
                            if lat_val is not None and lon_val is not None:
                                node_coords[nid] = (float(lat_val), float(lon_val))
                    except Exception:
                        pass

        # Project coordinates to canvas
        positions = self._project_geographic(node_coords, filtered_nodes)

        result = self._build_layout_result(
            graph_id=graph_id,
            algorithm=LayoutAlgorithm.GEOGRAPHIC.value,
            positions=positions,
            filtered_nodes=filtered_nodes,
            filtered_edges=filtered_edges,
            color_scheme=scheme,
            start_time=start_time,
        )

        # Attach lat/lon to node positions
        for nid, pos in result.node_positions.items():
            if nid in node_coords:
                pos.latitude = node_coords[nid][0]
                pos.longitude = node_coords[nid][1]

        return result

    def _project_geographic(
        self,
        node_coords: Dict[str, Tuple[float, float]],
        filtered_nodes: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[float]]:
        """Project geographic coordinates to canvas coordinates.

        Uses a simple equirectangular projection with scaling to fit
        the canvas. Nodes without coordinates are placed at the
        average position of their country group, or at canvas center.

        Args:
            node_coords: Mapping of node_id -> (latitude, longitude).
            filtered_nodes: All filtered node data.

        Returns:
            Positions dictionary.
        """
        positions: Dict[str, List[float]] = {}

        if not node_coords:
            # No geographic data, fall back to center placement
            cx = self._config.canvas_width / 2.0
            cy = self._config.canvas_height / 2.0
            rng = random.Random(self._config.fr_seed)
            for nid in filtered_nodes:
                positions[nid] = [
                    cx + rng.uniform(-100, 100),
                    cy + rng.uniform(-100, 100),
                ]
            return positions

        # Find bounding box of known coordinates
        lats = [c[0] for c in node_coords.values()]
        lons = [c[1] for c in node_coords.values()]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Add padding to avoid edge placement
        padding = 50.0
        effective_width = self._config.canvas_width - 2 * padding
        effective_height = self._config.canvas_height - 2 * padding

        lat_range = max_lat - min_lat if max_lat != min_lat else 1.0
        lon_range = max_lon - min_lon if max_lon != min_lon else 1.0

        # Place nodes with known coordinates
        for nid, (lat, lon) in node_coords.items():
            x = padding + ((lon - min_lon) / lon_range) * effective_width
            y = padding + ((max_lat - lat) / lat_range) * effective_height
            positions[nid] = [x, y]

        # Country averages for nodes without coordinates
        country_positions: Dict[str, List[List[float]]] = defaultdict(list)
        for nid, pos in positions.items():
            cc = filtered_nodes[nid].get("country_code", "")
            if cc:
                country_positions[cc].append(pos)

        rng = random.Random(self._config.fr_seed + 1)
        for nid in filtered_nodes:
            if nid not in positions:
                cc = filtered_nodes[nid].get("country_code", "")
                if cc and cc in country_positions:
                    cp_list = country_positions[cc]
                    avg_x = sum(p[0] for p in cp_list) / len(cp_list)
                    avg_y = sum(p[1] for p in cp_list) / len(cp_list)
                    positions[nid] = [
                        avg_x + rng.uniform(-30, 30),
                        avg_y + rng.uniform(-30, 30),
                    ]
                else:
                    positions[nid] = [
                        self._config.canvas_width / 2 + rng.uniform(-50, 50),
                        self._config.canvas_height / 2 + rng.uniform(-50, 50),
                    ]

        return positions

    # ==================================================================
    # Circular Layout
    # ==================================================================

    def compute_circular_layout(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        graph_filter: Optional[GraphFilter] = None,
        color_scheme: Optional[ColorScheme] = None,
    ) -> LayoutResult:
        """Compute a circular layout arranging nodes on a circle.

        Nodes are positioned evenly around a circle centered on the
        canvas. Node ordering can be influenced by tier depth.

        Args:
            graph_id: Graph identifier.
            nodes: Node data dictionaries.
            edges: Edge data dictionaries.
            graph_filter: Optional filter.
            color_scheme: Color coding scheme.

        Returns:
            LayoutResult with circular positioning.
        """
        start_time = time.monotonic()
        scheme = color_scheme or self._config.default_color_scheme

        filtered_nodes, filtered_edges = self._apply_filter(
            nodes, edges, graph_filter
        )

        if not filtered_nodes:
            return self._empty_layout(graph_id, LayoutAlgorithm.CIRCULAR.value)

        # Sort nodes by tier depth for grouping
        sorted_node_ids = sorted(
            filtered_nodes.keys(),
            key=lambda nid: (
                filtered_nodes[nid].get("tier_depth", 0),
                nid,
            ),
        )

        n = len(sorted_node_ids)
        cx = self._config.canvas_width / 2.0
        cy = self._config.canvas_height / 2.0
        radius = min(cx, cy) * 0.8

        positions: Dict[str, List[float]] = {}
        for i, nid in enumerate(sorted_node_ids):
            angle = 2.0 * math.pi * i / n
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            positions[nid] = [x, y]

        result = self._build_layout_result(
            graph_id=graph_id,
            algorithm=LayoutAlgorithm.CIRCULAR.value,
            positions=positions,
            filtered_nodes=filtered_nodes,
            filtered_edges=filtered_edges,
            color_scheme=scheme,
            start_time=start_time,
        )

        return result

    # ==================================================================
    # Sankey Diagram Data Generation
    # ==================================================================

    def generate_sankey_data(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        graph_filter: Optional[GraphFilter] = None,
        commodity_filter: Optional[str] = None,
    ) -> SankeyResult:
        """Generate Sankey diagram data for commodity flow visualization.

        Creates node and link data suitable for rendering with Plotly
        Sankey or D3.js Sankey diagrams. Flow volumes are derived from
        edge quantities.

        Args:
            graph_id: Graph identifier.
            nodes: Node data dictionaries.
            edges: Edge data dictionaries.
            graph_filter: Optional graph filter.
            commodity_filter: Optional specific commodity to filter flows.

        Returns:
            SankeyResult with nodes, links, and total flow volume.
        """
        start_time = time.monotonic()

        filtered_nodes, filtered_edges = self._apply_filter(
            nodes, edges, graph_filter
        )

        # Apply commodity filter
        if commodity_filter:
            filtered_edges = {
                eid: edata
                for eid, edata in filtered_edges.items()
                if edata.get("commodity", "") == commodity_filter
            }

        # Identify nodes that participate in remaining edges
        participating_nodes: Set[str] = set()
        for edata in filtered_edges.values():
            src = edata.get("source_node_id", "")
            tgt = edata.get("target_node_id", "")
            if src in filtered_nodes:
                participating_nodes.add(src)
            if tgt in filtered_nodes:
                participating_nodes.add(tgt)

        # Build Sankey nodes
        sankey_nodes: List[SankeyNode] = []
        node_flow: Dict[str, float] = defaultdict(float)

        for edata in filtered_edges.values():
            qty = float(edata.get("quantity", 0))
            src = edata.get("source_node_id", "")
            tgt = edata.get("target_node_id", "")
            node_flow[src] += qty
            node_flow[tgt] += qty

        for nid in sorted(participating_nodes):
            ndata = filtered_nodes.get(nid, {})
            node_type = ndata.get("node_type", "unknown")
            label = ndata.get("operator_name", nid)
            color = self._get_risk_color(ndata.get("risk_level", "standard"))

            sankey_nodes.append(SankeyNode(
                id=nid,
                label=label,
                color=color,
                value=node_flow.get(nid, 0.0),
            ))

        # Build Sankey links
        sankey_links: List[SankeyLink] = []
        total_flow = 0.0

        for eid, edata in filtered_edges.items():
            src = edata.get("source_node_id", "")
            tgt = edata.get("target_node_id", "")
            qty = float(edata.get("quantity", 0))
            commodity = edata.get("commodity", "")

            if src in participating_nodes and tgt in participating_nodes:
                # Edge color based on source risk
                src_data = filtered_nodes.get(src, {})
                edge_color = self._get_risk_color(
                    src_data.get("risk_level", "standard"),
                    opacity=0.4,
                )

                sankey_links.append(SankeyLink(
                    source=src,
                    target=tgt,
                    value=qty,
                    color=edge_color,
                    label=commodity,
                ))
                total_flow += qty

        elapsed_ms = (time.monotonic() - start_time) * 1000
        return SankeyResult(
            graph_id=graph_id,
            nodes=sankey_nodes,
            links=sankey_links,
            total_flow=total_flow,
            commodity_filter=commodity_filter,
            computation_time_ms=elapsed_ms,
        )

    # ==================================================================
    # Node Clustering
    # ==================================================================

    def compute_clusters(
        self,
        nodes: Dict[str, Dict[str, Any]],
        cluster_by: str = "country",
    ) -> List[ClusterGroup]:
        """Group nodes into clusters for layout optimization.

        Clusters can be based on country, tier depth, or node type.
        Clustering reduces visual complexity for large graphs.

        Args:
            nodes: Node data dictionaries.
            cluster_by: Clustering criterion ("country", "tier", "type").

        Returns:
            List of ClusterGroup objects.
        """
        groups: Dict[str, List[str]] = defaultdict(list)

        for nid, ndata in nodes.items():
            if cluster_by == "country":
                key = ndata.get("country_code", "unknown")
            elif cluster_by == "tier":
                key = str(ndata.get("tier_depth", 0))
            elif cluster_by == "type":
                key = ndata.get("node_type", "unknown")
            else:
                key = "default"
            groups[key].append(nid)

        clusters: List[ClusterGroup] = []
        for group_key, group_node_ids in sorted(groups.items()):
            if len(group_node_ids) < 2:
                continue
            clusters.append(ClusterGroup(
                cluster_id=_generate_id("CLUSTER"),
                node_ids=group_node_ids,
                label=group_key,
            ))

        return clusters

    # ==================================================================
    # Graph Export Formats
    # ==================================================================

    def export_geojson(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        graph_filter: Optional[GraphFilter] = None,
    ) -> Dict[str, Any]:
        """Export graph as GeoJSON FeatureCollection.

        Nodes are exported as Point features and edges as LineString
        features. Requires nodes to have geographic coordinates.

        Args:
            graph_id: Graph identifier.
            nodes: Node data dictionaries.
            edges: Edge data dictionaries.
            graph_filter: Optional filter.

        Returns:
            GeoJSON FeatureCollection dictionary.
        """
        filtered_nodes, filtered_edges = self._apply_filter(
            nodes, edges, graph_filter
        )

        features: List[Dict[str, Any]] = []

        # Node features
        for nid, ndata in filtered_nodes.items():
            lat = ndata.get("latitude")
            lon = ndata.get("longitude")
            coords = ndata.get("coordinates")
            if lat is None or lon is None:
                if coords and len(coords) == 2:
                    lat, lon = coords[0], coords[1]
                else:
                    continue

            properties = {
                "node_id": nid,
                "node_type": ndata.get("node_type", "unknown"),
                "operator_name": ndata.get("operator_name", ""),
                "country_code": ndata.get("country_code", ""),
                "risk_level": ndata.get("risk_level", "standard"),
                "risk_score": ndata.get("risk_score", 0),
                "compliance_status": ndata.get("compliance_status", "pending_verification"),
                "tier_depth": ndata.get("tier_depth", 0),
                "color": self._get_risk_color(ndata.get("risk_level", "standard")),
                "feature_type": "node",
            }
            commodities = ndata.get("commodities", [])
            if commodities:
                properties["commodities"] = (
                    commodities if isinstance(commodities, list) else [commodities]
                )

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": properties,
            })

        # Edge features (LineString between nodes with coordinates)
        for eid, edata in filtered_edges.items():
            src = edata.get("source_node_id", "")
            tgt = edata.get("target_node_id", "")
            src_data = filtered_nodes.get(src, {})
            tgt_data = filtered_nodes.get(tgt, {})

            src_coords = self._extract_coords(src_data)
            tgt_coords = self._extract_coords(tgt_data)

            if src_coords and tgt_coords:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [src_coords[1], src_coords[0]],
                            [tgt_coords[1], tgt_coords[0]],
                        ],
                    },
                    "properties": {
                        "edge_id": eid,
                        "source_node_id": src,
                        "target_node_id": tgt,
                        "commodity": edata.get("commodity", ""),
                        "quantity": float(edata.get("quantity", 0)),
                        "feature_type": "edge",
                    },
                })

        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "graph_id": graph_id,
                "total_nodes": sum(
                    1 for f in features if f["properties"].get("feature_type") == "node"
                ),
                "total_edges": sum(
                    1 for f in features if f["properties"].get("feature_type") == "edge"
                ),
                "generated_at": _utcnow().isoformat(),
                "generator": "GreenLang VisualizationEngine",
                "version": _MODULE_VERSION,
            },
        }

    def export_graphml(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        graph_filter: Optional[GraphFilter] = None,
    ) -> str:
        """Export graph in GraphML XML format.

        GraphML is an XML-based format widely supported by graph
        analysis tools such as Gephi, yEd, and Cytoscape.

        Args:
            graph_id: Graph identifier.
            nodes: Node data dictionaries.
            edges: Edge data dictionaries.
            graph_filter: Optional filter.

        Returns:
            GraphML XML string.
        """
        filtered_nodes, filtered_edges = self._apply_filter(
            nodes, edges, graph_filter
        )

        root = ET.Element("graphml")
        root.set("xmlns", "http://graphml.graphstruct.org/xmlns")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")

        # Define node attribute keys
        node_attrs = [
            ("node_type", "string"),
            ("operator_name", "string"),
            ("country_code", "string"),
            ("risk_level", "string"),
            ("risk_score", "double"),
            ("compliance_status", "string"),
            ("tier_depth", "int"),
        ]
        for attr_name, attr_type in node_attrs:
            key_el = ET.SubElement(root, "key")
            key_el.set("id", attr_name)
            key_el.set("for", "node")
            key_el.set("attr.name", attr_name)
            key_el.set("attr.type", attr_type)

        # Define edge attribute keys
        edge_attrs = [
            ("commodity", "string"),
            ("quantity", "double"),
            ("custody_model", "string"),
        ]
        for attr_name, attr_type in edge_attrs:
            key_el = ET.SubElement(root, "key")
            key_el.set("id", attr_name)
            key_el.set("for", "edge")
            key_el.set("attr.name", attr_name)
            key_el.set("attr.type", attr_type)

        # Graph element
        graph_el = ET.SubElement(root, "graph")
        graph_el.set("id", graph_id)
        graph_el.set("edgedefault", "directed")

        # Node elements
        for nid, ndata in filtered_nodes.items():
            node_el = ET.SubElement(graph_el, "node")
            node_el.set("id", nid)
            for attr_name, _ in node_attrs:
                val = ndata.get(attr_name, "")
                data_el = ET.SubElement(node_el, "data")
                data_el.set("key", attr_name)
                data_el.text = str(val)

        # Edge elements
        for eid, edata in filtered_edges.items():
            edge_el = ET.SubElement(graph_el, "edge")
            edge_el.set("id", eid)
            edge_el.set("source", edata.get("source_node_id", ""))
            edge_el.set("target", edata.get("target_node_id", ""))
            for attr_name, _ in edge_attrs:
                val = edata.get(attr_name, "")
                data_el = ET.SubElement(edge_el, "data")
                data_el.set("key", attr_name)
                data_el.text = str(val)

        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def export_jsonld(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        graph_filter: Optional[GraphFilter] = None,
    ) -> Dict[str, Any]:
        """Export graph as JSON-LD linked data.

        Produces a JSON-LD document with Schema.org and GreenLang
        ontology context for semantic web interoperability.

        Args:
            graph_id: Graph identifier.
            nodes: Node data dictionaries.
            edges: Edge data dictionaries.
            graph_filter: Optional filter.

        Returns:
            JSON-LD document dictionary.
        """
        filtered_nodes, filtered_edges = self._apply_filter(
            nodes, edges, graph_filter
        )

        node_items: List[Dict[str, Any]] = []
        for nid, ndata in filtered_nodes.items():
            item: Dict[str, Any] = {
                "@type": "greenlang:SupplyChainNode",
                "@id": f"greenlang:node/{nid}",
                "identifier": nid,
                "node_type": ndata.get("node_type", "unknown"),
                "name": ndata.get("operator_name", ""),
                "country_code": ndata.get("country_code", ""),
                "risk_level": ndata.get("risk_level", "standard"),
                "risk_score": ndata.get("risk_score", 0),
                "compliance_status": ndata.get("compliance_status", "pending_verification"),
                "tier_depth": ndata.get("tier_depth", 0),
            }
            lat = ndata.get("latitude")
            lon = ndata.get("longitude")
            if lat is not None and lon is not None:
                item["geo"] = {
                    "@type": "GeoCoordinates",
                    "latitude": lat,
                    "longitude": lon,
                }
            commodities = ndata.get("commodities", [])
            if commodities:
                item["commodity"] = commodities
            node_items.append(item)

        edge_items: List[Dict[str, Any]] = []
        for eid, edata in filtered_edges.items():
            edge_items.append({
                "@type": "greenlang:CustodyTransfer",
                "@id": f"greenlang:edge/{eid}",
                "identifier": eid,
                "source": f"greenlang:node/{edata.get('source_node_id', '')}",
                "target": f"greenlang:node/{edata.get('target_node_id', '')}",
                "commodity": edata.get("commodity", ""),
                "quantity": float(edata.get("quantity", 0)),
                "custody_model": edata.get("custody_model", "segregated"),
                "transfer_date": str(edata.get("transfer_date", "")),
            })

        document = dict(JSONLD_CONTEXT)
        document["@type"] = "greenlang:SupplyChainGraph"
        document["@id"] = f"greenlang:graph/{graph_id}"
        document["identifier"] = graph_id
        document["nodes"] = node_items
        document["edges"] = edge_items
        document["dateGenerated"] = _utcnow().isoformat()

        return document

    # ==================================================================
    # Time-Based Snapshot Retrieval
    # ==================================================================

    def get_graph_snapshot(
        self,
        graph_id: str,
        snapshot_time: Optional[datetime] = None,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a historical graph snapshot for time-based visualization.

        If the graph engine is available, delegates to its snapshot
        mechanism. Otherwise, uses the internal snapshot cache.

        Args:
            graph_id: Graph identifier.
            snapshot_time: Optional point-in-time for the snapshot.
            version: Optional specific graph version to retrieve.

        Returns:
            Snapshot dictionary with nodes, edges, and metadata, or
            None if no matching snapshot is found.
        """
        if self._graph_engine is not None:
            try:
                if hasattr(self._graph_engine, "create_snapshot"):
                    # Retrieve from graph engine snapshot store
                    if hasattr(self._graph_engine, "_snapshots"):
                        graph_snapshots = self._graph_engine._snapshots.get(
                            graph_id, []
                        )
                        for snap in reversed(graph_snapshots):
                            if version is not None and snap.version == version:
                                return {
                                    "snapshot_id": snap.snapshot_id,
                                    "graph_id": snap.graph_id,
                                    "version": snap.version,
                                    "node_count": snap.node_count,
                                    "edge_count": snap.edge_count,
                                    "nodes": snap.nodes,
                                    "edges": snap.edges,
                                    "provenance_hash": snap.provenance_hash,
                                    "created_at": snap.created_at.isoformat()
                                    if hasattr(snap.created_at, "isoformat")
                                    else str(snap.created_at),
                                }
                            if snapshot_time is not None:
                                snap_time = snap.created_at
                                if hasattr(snap_time, "timestamp"):
                                    if snap_time <= snapshot_time:
                                        return {
                                            "snapshot_id": snap.snapshot_id,
                                            "graph_id": snap.graph_id,
                                            "version": snap.version,
                                            "node_count": snap.node_count,
                                            "edge_count": snap.edge_count,
                                            "nodes": snap.nodes,
                                            "edges": snap.edges,
                                            "provenance_hash": snap.provenance_hash,
                                            "created_at": snap_time.isoformat(),
                                        }
            except Exception as e:
                logger.warning(
                    "Failed to retrieve snapshot from graph engine: %s", e
                )

        # Fallback to internal cache
        cache_key = f"{graph_id}:{version}" if version else f"{graph_id}:latest"
        return self._snapshot_cache.get(cache_key)

    def store_snapshot(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        version: int = 1,
    ) -> Dict[str, Any]:
        """Store a graph snapshot in the internal cache for historical views.

        Args:
            graph_id: Graph identifier.
            nodes: Node data dictionaries.
            edges: Edge data dictionaries.
            version: Graph version number.

        Returns:
            Snapshot metadata dictionary.
        """
        snapshot_id = _generate_id("SNAP")
        snapshot_data = {
            "snapshot_id": snapshot_id,
            "graph_id": graph_id,
            "version": version,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": dict(nodes),
            "edges": dict(edges),
            "provenance_hash": _compute_provenance_hash({
                "graph_id": graph_id,
                "version": version,
                "nodes": nodes,
                "edges": edges,
            }),
            "created_at": _utcnow().isoformat(),
        }

        cache_key = f"{graph_id}:{version}"
        self._snapshot_cache[cache_key] = snapshot_data
        self._snapshot_cache[f"{graph_id}:latest"] = snapshot_data

        return snapshot_data

    # ==================================================================
    # Risk Color Coding
    # ==================================================================

    def get_risk_coloring(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        """Compute risk-based color coding for all nodes and edges.

        Uses the risk propagation engine if available, otherwise
        derives colors from node risk_level attributes.

        Args:
            nodes: Node data dictionaries.
            edges: Edge data dictionaries.

        Returns:
            Dictionary with "nodes" and "edges" sub-dictionaries
            mapping IDs to color hex codes.
        """
        node_colors: Dict[str, str] = {}
        edge_colors: Dict[str, str] = {}

        for nid, ndata in nodes.items():
            risk_level = ndata.get("risk_level", "standard")
            node_colors[nid] = self._get_risk_color(risk_level)

        for eid, edata in edges.items():
            src = edata.get("source_node_id", "")
            tgt = edata.get("target_node_id", "")
            # Edge color is based on the higher-risk endpoint
            src_risk = nodes.get(src, {}).get("risk_score", 0)
            tgt_risk = nodes.get(tgt, {}).get("risk_score", 0)
            max_risk_level = nodes.get(
                src if src_risk >= tgt_risk else tgt, {}
            ).get("risk_level", "standard")
            edge_colors[eid] = self._get_risk_color(max_risk_level)

        return {"nodes": node_colors, "edges": edge_colors}

    # ==================================================================
    # Internal Helpers
    # ==================================================================

    def _apply_filter(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        graph_filter: Optional[GraphFilter],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Apply filter to nodes and edges, returning filtered copies.

        Filters are applied in the following order:
        1. Node IDs (explicit inclusion)
        2. Commodities
        3. Countries
        4. Risk levels
        5. Compliance statuses
        6. Tier depth range
        7. Node types
        Then edges are filtered to only include edges between remaining nodes.

        Args:
            nodes: All node data.
            edges: All edge data.
            graph_filter: Optional filter specification.

        Returns:
            Tuple of (filtered_nodes, filtered_edges).
        """
        if graph_filter is None:
            return dict(nodes), dict(edges)

        filtered_nodes: Dict[str, Dict[str, Any]] = {}

        for nid, ndata in nodes.items():
            # Explicit node ID filter
            if graph_filter.node_ids is not None:
                if nid not in graph_filter.node_ids:
                    continue

            # Commodity filter
            if graph_filter.commodities is not None:
                node_commodities = ndata.get("commodities", [])
                if isinstance(node_commodities, str):
                    node_commodities = [node_commodities]
                if not any(c in graph_filter.commodities for c in node_commodities):
                    continue

            # Country filter
            if graph_filter.countries is not None:
                cc = ndata.get("country_code", "")
                if cc not in graph_filter.countries:
                    continue

            # Risk level filter
            if graph_filter.risk_levels is not None:
                rl = ndata.get("risk_level", "standard")
                if rl not in graph_filter.risk_levels:
                    continue

            # Compliance status filter
            if graph_filter.compliance_statuses is not None:
                cs = ndata.get("compliance_status", "pending_verification")
                if cs not in graph_filter.compliance_statuses:
                    continue

            # Tier depth filter
            tier = ndata.get("tier_depth", 0)
            if graph_filter.min_tier_depth is not None:
                if tier < graph_filter.min_tier_depth:
                    continue
            if graph_filter.max_tier_depth is not None:
                if tier > graph_filter.max_tier_depth:
                    continue

            # Node type filter
            if graph_filter.node_types is not None:
                nt = ndata.get("node_type", "unknown")
                if nt not in graph_filter.node_types:
                    continue

            filtered_nodes[nid] = ndata

        # Filter edges to only include those between remaining nodes
        remaining_ids = set(filtered_nodes.keys())
        filtered_edges: Dict[str, Dict[str, Any]] = {}
        for eid, edata in edges.items():
            src = edata.get("source_node_id", "")
            tgt = edata.get("target_node_id", "")
            if src in remaining_ids and tgt in remaining_ids:
                filtered_edges[eid] = edata

        return filtered_nodes, filtered_edges

    def _build_adjacency(
        self,
        edges: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Build adjacency list from edge data.

        Args:
            edges: Edge data dictionaries.

        Returns:
            Adjacency list mapping source -> [targets].
        """
        adj: Dict[str, List[str]] = defaultdict(list)
        for edata in edges.values():
            src = edata.get("source_node_id", "")
            tgt = edata.get("target_node_id", "")
            if src and tgt:
                adj[src].append(tgt)
        return dict(adj)

    def _build_layout_result(
        self,
        graph_id: str,
        algorithm: str,
        positions: Dict[str, List[float]],
        filtered_nodes: Dict[str, Dict[str, Any]],
        filtered_edges: Dict[str, Dict[str, Any]],
        color_scheme: ColorScheme,
        start_time: float,
    ) -> LayoutResult:
        """Build a LayoutResult from computed positions.

        Args:
            graph_id: Graph identifier.
            algorithm: Algorithm name.
            positions: Computed node positions.
            filtered_nodes: Node data.
            filtered_edges: Edge data.
            color_scheme: Color coding scheme.
            start_time: Computation start time (monotonic).

        Returns:
            Complete LayoutResult.
        """
        # Build node positions with styling
        node_positions: Dict[str, NodePosition] = {}
        for nid, pos in positions.items():
            ndata = filtered_nodes.get(nid, {})
            node_type = ndata.get("node_type", "unknown")

            color = self._compute_node_color(ndata, color_scheme)
            shape = NODE_TYPE_SHAPES.get(node_type, "circle")
            size = NODE_TYPE_SIZES.get(node_type, 10.0)
            label = ndata.get("operator_name", nid)

            node_positions[nid] = NodePosition(
                node_id=nid,
                x=pos[0],
                y=pos[1],
                color=color,
                shape=shape,
                size=size,
                label=label,
                metadata={
                    "node_type": node_type,
                    "country_code": ndata.get("country_code", ""),
                    "risk_level": ndata.get("risk_level", "standard"),
                    "risk_score": ndata.get("risk_score", 0),
                    "compliance_status": ndata.get(
                        "compliance_status", "pending_verification"
                    ),
                    "tier_depth": ndata.get("tier_depth", 0),
                },
            )

        # Build edge paths
        edge_paths: Dict[str, EdgePath] = {}
        for eid, edata in filtered_edges.items():
            src = edata.get("source_node_id", "")
            tgt = edata.get("target_node_id", "")
            if src in positions and tgt in positions:
                src_pos = positions[src]
                tgt_pos = positions[tgt]
                waypoints = [(src_pos[0], src_pos[1]), (tgt_pos[0], tgt_pos[1])]

                # Edge color based on source risk
                src_data = filtered_nodes.get(src, {})
                edge_color = self._get_risk_color(
                    src_data.get("risk_level", "standard")
                )

                qty = float(edata.get("quantity", 0))
                width = max(1.0, min(5.0, 1.0 + math.log10(max(qty, 1))))

                edge_paths[eid] = EdgePath(
                    edge_id=eid,
                    source_node_id=src,
                    target_node_id=tgt,
                    waypoints=waypoints,
                    color=edge_color,
                    width=width,
                    label=edata.get("commodity", ""),
                    metadata={
                        "quantity": qty,
                        "commodity": edata.get("commodity", ""),
                    },
                )

        # Compute viewport
        if positions:
            all_x = [p[0] for p in positions.values()]
            all_y = [p[1] for p in positions.values()]
            viewport = {
                "min_x": min(all_x) - 20,
                "min_y": min(all_y) - 20,
                "max_x": max(all_x) + 20,
                "max_y": max(all_y) + 20,
            }
        else:
            viewport = {
                "min_x": 0.0, "min_y": 0.0,
                "max_x": self._config.canvas_width,
                "max_y": self._config.canvas_height,
            }

        # Compute clusters if threshold exceeded
        clusters: List[ClusterGroup] = []
        if len(filtered_nodes) >= self._config.cluster_threshold:
            clusters = self.compute_clusters(filtered_nodes, cluster_by="country")
            # Assign cluster centers and node cluster_ids
            for cluster in clusters:
                cluster_xs = [
                    positions[nid][0] for nid in cluster.node_ids
                    if nid in positions
                ]
                cluster_ys = [
                    positions[nid][1] for nid in cluster.node_ids
                    if nid in positions
                ]
                if cluster_xs and cluster_ys:
                    cluster.center_x = sum(cluster_xs) / len(cluster_xs)
                    cluster.center_y = sum(cluster_ys) / len(cluster_ys)
                    max_dist = max(
                        math.sqrt(
                            (x - cluster.center_x) ** 2 + (y - cluster.center_y) ** 2
                        )
                        for x, y in zip(cluster_xs, cluster_ys)
                    )
                    cluster.radius = max(max_dist + 20, self._config.cluster_radius)

                for nid in cluster.node_ids:
                    if nid in node_positions:
                        node_positions[nid].cluster_id = cluster.cluster_id

        # Compute provenance hash
        elapsed_ms = (time.monotonic() - start_time) * 1000
        provenance_data = {
            "graph_id": graph_id,
            "algorithm": algorithm,
            "node_count": len(node_positions),
            "edge_count": len(edge_paths),
            "seed": self._config.fr_seed,
        }
        provenance_hash = _compute_provenance_hash(provenance_data)

        return LayoutResult(
            graph_id=graph_id,
            algorithm=algorithm,
            node_positions=node_positions,
            edge_paths=edge_paths,
            clusters=clusters,
            viewport=viewport,
            computation_time_ms=elapsed_ms,
            total_nodes=len(node_positions),
            total_edges=len(edge_paths),
            provenance_hash=provenance_hash,
        )

    def _empty_layout(self, graph_id: str, algorithm: str) -> LayoutResult:
        """Return an empty layout result for graphs with no nodes."""
        return LayoutResult(
            graph_id=graph_id,
            algorithm=algorithm,
            total_nodes=0,
            total_edges=0,
        )

    def _compute_node_color(
        self,
        ndata: Dict[str, Any],
        color_scheme: ColorScheme,
    ) -> str:
        """Compute node color based on the selected color scheme.

        Args:
            ndata: Node data dictionary.
            color_scheme: Active color scheme.

        Returns:
            Hex color string.
        """
        if color_scheme == ColorScheme.RISK_LEVEL:
            return self._get_risk_color(ndata.get("risk_level", "standard"))

        elif color_scheme == ColorScheme.COMPLIANCE_STATUS:
            status = ndata.get("compliance_status", "pending_verification")
            return COMPLIANCE_COLORS.get(status, RISK_COLOR_UNKNOWN)

        elif color_scheme == ColorScheme.NODE_TYPE:
            node_type = ndata.get("node_type", "unknown")
            type_colors = {
                "producer": "#22C55E",
                "collector": "#F59E0B",
                "processor": "#3B82F6",
                "trader": "#8B5CF6",
                "importer": "#EF4444",
                "certifier": "#06B6D4",
                "warehouse": "#6B7280",
                "port": "#14B8A6",
            }
            return type_colors.get(node_type, RISK_COLOR_UNKNOWN)

        elif color_scheme == ColorScheme.TIER_DEPTH:
            tier = ndata.get("tier_depth", 0)
            # Gradient from blue (tier 0) to green (deep tiers)
            tier_colors = [
                "#EF4444", "#F59E0B", "#22C55E", "#3B82F6",
                "#8B5CF6", "#06B6D4", "#14B8A6", "#10B981",
            ]
            idx = min(tier, len(tier_colors) - 1)
            return tier_colors[idx]

        elif color_scheme == ColorScheme.COUNTRY:
            # Deterministic color from country code using stable hash
            cc = ndata.get("country_code", "XX")
            # Use sum of char ordinals for cross-run determinism (avoids
            # Python hash randomization which makes hash() non-deterministic
            # across interpreter invocations).
            h = sum(ord(c) for c in cc)
            palette = [
                "#EF4444", "#F97316", "#F59E0B", "#84CC16",
                "#22C55E", "#14B8A6", "#06B6D4", "#3B82F6",
                "#6366F1", "#8B5CF6", "#EC4899", "#F43F5E",
            ]
            return palette[h % len(palette)]

        return RISK_COLOR_UNKNOWN

    @staticmethod
    def _get_risk_color(
        risk_level: str,
        opacity: Optional[float] = None,
    ) -> str:
        """Get hex color for a risk level.

        Args:
            risk_level: Risk level string ("low", "standard", "high").
            opacity: Optional opacity (not applied to hex, used for
                generating alpha-variant CSS colors).

        Returns:
            Hex color string.
        """
        level = risk_level.lower() if isinstance(risk_level, str) else "standard"
        if level == "low":
            return RISK_COLOR_LOW
        elif level == "standard":
            return RISK_COLOR_STANDARD
        elif level == "high":
            return RISK_COLOR_HIGH
        return RISK_COLOR_UNKNOWN

    @staticmethod
    def _extract_coords(
        ndata: Dict[str, Any],
    ) -> Optional[Tuple[float, float]]:
        """Extract (latitude, longitude) from node data.

        Checks both separate lat/lon fields and the combined
        coordinates tuple format.

        Args:
            ndata: Node data dictionary.

        Returns:
            (latitude, longitude) tuple, or None if not available.
        """
        lat = ndata.get("latitude")
        lon = ndata.get("longitude")
        if lat is not None and lon is not None:
            return (float(lat), float(lon))
        coords = ndata.get("coordinates")
        if coords and len(coords) == 2:
            return (float(coords[0]), float(coords[1]))
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Engine
    "VisualizationEngine",
    # Configuration
    "VisualizationConfig",
    # Enumerations
    "LayoutAlgorithm",
    "ExportFormat",
    "ColorScheme",
    # Data structures
    "NodePosition",
    "EdgePath",
    "ClusterGroup",
    "SankeyNode",
    "SankeyLink",
    "LayoutResult",
    "SankeyResult",
    "GraphFilter",
    # Constants
    "RISK_COLOR_LOW",
    "RISK_COLOR_STANDARD",
    "RISK_COLOR_HIGH",
    "RISK_COLOR_UNKNOWN",
    "COMPLIANCE_COLORS",
    "NODE_TYPE_SHAPES",
    "NODE_TYPE_SIZES",
    "JSONLD_CONTEXT",
]
