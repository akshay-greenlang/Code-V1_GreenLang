"""
Supply Chain Graph Analysis.

This module provides NetworkX-based graph modeling and analysis for
multi-tier supply chain mapping. It supports:

- Directed graph representation of supplier relationships
- Material flow tracking with quantities and emissions
- Risk propagation through supply chain tiers
- Path finding for traceability and due diligence
- Network analysis (centrality, clustering, critical paths)

Key Features:
- Tier-based supplier classification
- Material flow aggregation
- Emission allocation through supply chain
- Critical supplier identification (Pareto analysis)
- EUDR-compliant traceability paths

Example:
    >>> from greenlang.supply_chain.graph import SupplyChainGraph
    >>> graph = SupplyChainGraph()
    >>> graph.add_supplier(supplier)
    >>> graph.add_relationship(relationship)
    >>> paths = graph.find_traceability_paths("SUP001", "MATERIAL001")
    >>> metrics = graph.compute_network_metrics()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import (
    Optional, List, Dict, Any, Set, Tuple, Iterator, Generator
)
from collections import defaultdict

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from greenlang.supply_chain.models.entity import (
    Supplier,
    Facility,
    Material,
    Product,
    SupplierRelationship,
    RelationshipType,
    SupplierTier,
    CommodityType,
)

logger = logging.getLogger(__name__)


class FlowType(Enum):
    """Type of material/product flow in the supply chain."""
    MATERIAL = "material"
    PRODUCT = "product"
    SERVICE = "service"
    FINANCIAL = "financial"


@dataclass
class MaterialFlow:
    """
    Represents a material/product flow between suppliers.

    Attributes:
        source_id: Source supplier/facility ID
        target_id: Target supplier/facility ID
        material_id: Material or product ID
        material_name: Human-readable material name
        quantity: Flow quantity
        unit: Unit of measure
        emission_kg_co2e: Associated emissions in kg CO2e
        flow_type: Type of flow (material, product, service)
        period_start: Start of measurement period
        period_end: End of measurement period
        verified: Whether flow has been verified
        metadata: Additional flow attributes
    """
    source_id: str
    target_id: str
    material_id: str
    material_name: str
    quantity: Decimal
    unit: str = "kg"
    emission_kg_co2e: Optional[Decimal] = None
    flow_type: FlowType = FlowType.MATERIAL
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "material_id": self.material_id,
            "material_name": self.material_name,
            "quantity": str(self.quantity),
            "unit": self.unit,
            "emission_kg_co2e": str(self.emission_kg_co2e) if self.emission_kg_co2e else None,
            "flow_type": self.flow_type.value,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "verified": self.verified,
            "metadata": self.metadata,
        }


@dataclass
class SupplyChainPath:
    """
    Represents a path through the supply chain graph.

    Used for traceability analysis and EUDR due diligence.

    Attributes:
        nodes: Ordered list of supplier IDs in the path
        edges: List of relationships along the path
        total_length: Number of hops in the path
        materials: Materials traced through this path
        total_emission_kg_co2e: Cumulative emissions along path
        risk_score: Aggregated risk score for the path
        verified: Whether all nodes/edges are verified
    """
    nodes: List[str]
    edges: List[str] = field(default_factory=list)
    total_length: int = 0
    materials: List[str] = field(default_factory=list)
    total_emission_kg_co2e: Optional[Decimal] = None
    risk_score: Optional[float] = None
    verified: bool = False

    def __post_init__(self):
        """Compute path length."""
        self.total_length = len(self.nodes) - 1 if len(self.nodes) > 1 else 0

    @property
    def source(self) -> Optional[str]:
        """Get source node of the path."""
        return self.nodes[0] if self.nodes else None

    @property
    def target(self) -> Optional[str]:
        """Get target node of the path."""
        return self.nodes[-1] if self.nodes else None

    @property
    def tier_depth(self) -> int:
        """Get the tier depth of the path (same as length)."""
        return self.total_length

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "total_length": self.total_length,
            "materials": self.materials,
            "total_emission_kg_co2e": str(self.total_emission_kg_co2e) if self.total_emission_kg_co2e else None,
            "risk_score": self.risk_score,
            "verified": self.verified,
        }


@dataclass
class GraphMetrics:
    """
    Network analysis metrics for the supply chain graph.

    Attributes:
        total_nodes: Total number of suppliers
        total_edges: Total number of relationships
        density: Graph density (edges / possible edges)
        avg_degree: Average node degree
        tier_distribution: Count of suppliers per tier
        top_suppliers_by_degree: Most connected suppliers
        top_suppliers_by_spend: Highest spend suppliers
        clustering_coefficient: Network clustering coefficient
        connected_components: Number of disconnected subgraphs
        critical_suppliers: Suppliers with high centrality
    """
    total_nodes: int = 0
    total_edges: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    tier_distribution: Dict[int, int] = field(default_factory=dict)
    top_suppliers_by_degree: List[Tuple[str, int]] = field(default_factory=list)
    top_suppliers_by_spend: List[Tuple[str, Decimal]] = field(default_factory=list)
    clustering_coefficient: float = 0.0
    connected_components: int = 0
    critical_suppliers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "density": self.density,
            "avg_degree": self.avg_degree,
            "tier_distribution": self.tier_distribution,
            "top_suppliers_by_degree": [
                {"id": s[0], "degree": s[1]}
                for s in self.top_suppliers_by_degree
            ],
            "top_suppliers_by_spend": [
                {"id": s[0], "spend": str(s[1])}
                for s in self.top_suppliers_by_spend
            ],
            "clustering_coefficient": self.clustering_coefficient,
            "connected_components": self.connected_components,
            "critical_suppliers": self.critical_suppliers,
        }


class SupplyChainGraph:
    """
    NetworkX-based supply chain graph model.

    Provides comprehensive graph analysis capabilities for multi-tier
    supply chain mapping, supporting:
    - Supplier and relationship management
    - Material flow tracking
    - Traceability path finding
    - Network analysis and metrics
    - Risk propagation analysis

    Example:
        >>> graph = SupplyChainGraph(company_id="MYCOMPANY")
        >>> graph.add_supplier(tier1_supplier)
        >>> graph.add_supplier(tier2_supplier)
        >>> graph.add_relationship(relationship)
        >>>
        >>> # Find all paths to a material source
        >>> paths = graph.find_all_paths_to_source(
        ...     "MATERIAL001",
        ...     max_depth=5
        ... )
        >>>
        >>> # Compute network metrics
        >>> metrics = graph.compute_network_metrics()
        >>> print(f"Critical suppliers: {metrics.critical_suppliers}")
    """

    def __init__(
        self,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
    ):
        """
        Initialize the supply chain graph.

        Args:
            company_id: ID of the focal company (your organization)
            company_name: Name of the focal company
        """
        if not HAS_NETWORKX:
            raise ImportError(
                "NetworkX is required for supply chain graph analysis. "
                "Install with: pip install networkx"
            )

        # Create directed graph (supplier -> buyer direction)
        self._graph: nx.DiGraph = nx.DiGraph()

        # Focal company
        self.company_id = company_id
        self.company_name = company_name

        # Entity storage
        self._suppliers: Dict[str, Supplier] = {}
        self._facilities: Dict[str, Facility] = {}
        self._materials: Dict[str, Material] = {}
        self._products: Dict[str, Product] = {}
        self._relationships: Dict[str, SupplierRelationship] = {}
        self._flows: List[MaterialFlow] = []

        # Indexes for efficient lookup
        self._supplier_by_tier: Dict[SupplierTier, Set[str]] = defaultdict(set)
        self._suppliers_by_country: Dict[str, Set[str]] = defaultdict(set)
        self._suppliers_by_commodity: Dict[CommodityType, Set[str]] = defaultdict(set)

        logger.info(f"SupplyChainGraph initialized for company: {company_id}")

    @property
    def graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph."""
        return self._graph

    @property
    def supplier_count(self) -> int:
        """Get total number of suppliers."""
        return len(self._suppliers)

    @property
    def relationship_count(self) -> int:
        """Get total number of relationships."""
        return len(self._relationships)

    # =========================================================================
    # Entity Management
    # =========================================================================

    def add_supplier(self, supplier: Supplier) -> None:
        """
        Add a supplier to the graph.

        Args:
            supplier: Supplier entity to add
        """
        # Store supplier
        self._suppliers[supplier.id] = supplier

        # Add node to graph with attributes
        self._graph.add_node(
            supplier.id,
            entity_type="supplier",
            name=supplier.name,
            tier=supplier.tier.value,
            country=supplier.country_code,
            status=supplier.status.value,
            annual_spend=float(supplier.annual_spend) if supplier.annual_spend else 0,
        )

        # Update indexes
        self._supplier_by_tier[supplier.tier].add(supplier.id)
        if supplier.country_code:
            self._suppliers_by_country[supplier.country_code].add(supplier.id)
        for commodity in supplier.commodities:
            self._suppliers_by_commodity[commodity].add(supplier.id)

        logger.debug(f"Added supplier: {supplier.id} - {supplier.name}")

    def add_facility(self, facility: Facility) -> None:
        """
        Add a facility to the graph.

        Args:
            facility: Facility entity to add
        """
        self._facilities[facility.id] = facility

        # Add node to graph
        self._graph.add_node(
            facility.id,
            entity_type="facility",
            name=facility.name,
            supplier_id=facility.supplier_id,
            facility_type=facility.facility_type,
            latitude=facility.location.latitude if facility.location else None,
            longitude=facility.location.longitude if facility.location else None,
        )

        # Link to parent supplier
        if facility.supplier_id in self._suppliers:
            self._graph.add_edge(
                facility.id,
                facility.supplier_id,
                relationship_type="belongs_to",
            )

        logger.debug(f"Added facility: {facility.id} - {facility.name}")

    def add_material(self, material: Material) -> None:
        """
        Add a material to the entity store.

        Args:
            material: Material entity to add
        """
        self._materials[material.id] = material
        logger.debug(f"Added material: {material.id} - {material.name}")

    def add_product(self, product: Product) -> None:
        """
        Add a product to the entity store.

        Args:
            product: Product entity to add
        """
        self._products[product.id] = product
        logger.debug(f"Added product: {product.id} - {product.name}")

    def add_relationship(self, relationship: SupplierRelationship) -> None:
        """
        Add a supplier relationship as a graph edge.

        The edge direction is source (upstream) -> target (downstream),
        representing the flow of goods/materials.

        Args:
            relationship: Supplier relationship to add
        """
        # Store relationship
        self._relationships[relationship.id] = relationship

        # Add edge to graph (source -> target)
        self._graph.add_edge(
            relationship.source_supplier_id,
            relationship.target_supplier_id,
            relationship_id=relationship.id,
            relationship_type=relationship.relationship_type.value,
            materials=relationship.materials,
            products=relationship.products,
            annual_spend=float(relationship.annual_spend) if relationship.annual_spend else 0,
            annual_volume=float(relationship.annual_volume) if relationship.annual_volume else 0,
            active=relationship.active,
            verified=relationship.verified,
        )

        logger.debug(
            f"Added relationship: {relationship.source_supplier_id} -> "
            f"{relationship.target_supplier_id}"
        )

    def add_material_flow(self, flow: MaterialFlow) -> None:
        """
        Add a material flow between suppliers.

        Args:
            flow: Material flow to add
        """
        self._flows.append(flow)

        # Update edge attributes with flow data
        if self._graph.has_edge(flow.source_id, flow.target_id):
            edge_data = self._graph.edges[flow.source_id, flow.target_id]
            flows = edge_data.get("flows", [])
            flows.append(flow.to_dict())
            edge_data["flows"] = flows

    def get_supplier(self, supplier_id: str) -> Optional[Supplier]:
        """Get supplier by ID."""
        return self._suppliers.get(supplier_id)

    def get_facility(self, facility_id: str) -> Optional[Facility]:
        """Get facility by ID."""
        return self._facilities.get(facility_id)

    def get_relationship(self, relationship_id: str) -> Optional[SupplierRelationship]:
        """Get relationship by ID."""
        return self._relationships.get(relationship_id)

    def get_suppliers_by_tier(self, tier: SupplierTier) -> List[Supplier]:
        """Get all suppliers at a specific tier."""
        supplier_ids = self._supplier_by_tier.get(tier, set())
        return [self._suppliers[sid] for sid in supplier_ids if sid in self._suppliers]

    def get_suppliers_by_country(self, country_code: str) -> List[Supplier]:
        """Get all suppliers in a specific country."""
        supplier_ids = self._suppliers_by_country.get(country_code.upper(), set())
        return [self._suppliers[sid] for sid in supplier_ids if sid in self._suppliers]

    def get_suppliers_by_commodity(self, commodity: CommodityType) -> List[Supplier]:
        """Get all suppliers providing a specific commodity."""
        supplier_ids = self._suppliers_by_commodity.get(commodity, set())
        return [self._suppliers[sid] for sid in supplier_ids if sid in self._suppliers]

    # =========================================================================
    # Graph Traversal and Path Finding
    # =========================================================================

    def get_direct_suppliers(self, supplier_id: str) -> List[Supplier]:
        """
        Get direct (Tier 1) suppliers of a given entity.

        Args:
            supplier_id: Target supplier ID

        Returns:
            List of direct suppliers
        """
        if supplier_id not in self._graph:
            return []

        # Get predecessors (upstream suppliers)
        predecessor_ids = list(self._graph.predecessors(supplier_id))
        return [
            self._suppliers[pid]
            for pid in predecessor_ids
            if pid in self._suppliers
        ]

    def get_direct_customers(self, supplier_id: str) -> List[Supplier]:
        """
        Get direct customers (downstream) of a given supplier.

        Args:
            supplier_id: Source supplier ID

        Returns:
            List of direct customers
        """
        if supplier_id not in self._graph:
            return []

        # Get successors (downstream customers)
        successor_ids = list(self._graph.successors(supplier_id))
        return [
            self._suppliers[sid]
            for sid in successor_ids
            if sid in self._suppliers
        ]

    def get_upstream_suppliers(
        self,
        supplier_id: str,
        max_depth: Optional[int] = None,
    ) -> Dict[int, List[Supplier]]:
        """
        Get all upstream suppliers organized by tier depth.

        Args:
            supplier_id: Starting supplier ID
            max_depth: Maximum depth to traverse (None for unlimited)

        Returns:
            Dictionary mapping tier depth to list of suppliers
        """
        if supplier_id not in self._graph:
            return {}

        upstream: Dict[int, List[Supplier]] = defaultdict(list)
        visited: Set[str] = {supplier_id}
        current_level: Set[str] = {supplier_id}
        depth = 0

        while current_level:
            depth += 1
            if max_depth and depth > max_depth:
                break

            next_level: Set[str] = set()
            for node_id in current_level:
                for pred_id in self._graph.predecessors(node_id):
                    if pred_id not in visited:
                        visited.add(pred_id)
                        next_level.add(pred_id)
                        if pred_id in self._suppliers:
                            upstream[depth].append(self._suppliers[pred_id])

            current_level = next_level

        return dict(upstream)

    def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: Optional[int] = None,
    ) -> List[SupplyChainPath]:
        """
        Find all paths between two suppliers.

        Args:
            source_id: Source supplier ID
            target_id: Target supplier ID
            max_length: Maximum path length

        Returns:
            List of SupplyChainPath objects
        """
        if source_id not in self._graph or target_id not in self._graph:
            return []

        try:
            cutoff = max_length or 10  # Default max depth
            all_paths = nx.all_simple_paths(
                self._graph,
                source_id,
                target_id,
                cutoff=cutoff
            )

            paths = []
            for path_nodes in all_paths:
                # Get edge IDs along the path
                edge_ids = []
                for i in range(len(path_nodes) - 1):
                    edge_data = self._graph.edges.get(
                        (path_nodes[i], path_nodes[i + 1]), {}
                    )
                    rel_id = edge_data.get("relationship_id")
                    if rel_id:
                        edge_ids.append(rel_id)

                paths.append(SupplyChainPath(
                    nodes=list(path_nodes),
                    edges=edge_ids,
                ))

            return paths

        except nx.NetworkXError as e:
            logger.warning(f"Path finding error: {e}")
            return []

    def find_traceability_paths(
        self,
        target_id: str,
        material_id: Optional[str] = None,
        commodity: Optional[CommodityType] = None,
        max_depth: int = 10,
    ) -> List[SupplyChainPath]:
        """
        Find traceability paths from sources to a target supplier.

        For EUDR compliance, traces materials back to their origins.

        Args:
            target_id: Target supplier ID (usually your company)
            material_id: Specific material to trace (optional)
            commodity: Commodity type to trace (optional)
            max_depth: Maximum depth to search

        Returns:
            List of traceability paths
        """
        if target_id not in self._graph:
            return []

        paths: List[SupplyChainPath] = []

        # Find all source nodes (nodes with no incoming edges that match criteria)
        source_nodes = []
        for node_id in self._graph.nodes():
            # Check if it's a source (no predecessors)
            if self._graph.in_degree(node_id) == 0:
                supplier = self._suppliers.get(node_id)
                if supplier:
                    # Filter by commodity if specified
                    if commodity and commodity not in supplier.commodities:
                        continue
                    source_nodes.append(node_id)

        # Find paths from each source to target
        for source_id in source_nodes:
            source_paths = self.find_all_paths(
                source_id,
                target_id,
                max_length=max_depth
            )

            # Filter by material if specified
            if material_id:
                filtered_paths = []
                for path in source_paths:
                    # Check if any edge carries the material
                    for edge_id in path.edges:
                        rel = self._relationships.get(edge_id)
                        if rel and material_id in rel.materials:
                            path.materials.append(material_id)
                            filtered_paths.append(path)
                            break
                paths.extend(filtered_paths)
            else:
                paths.extend(source_paths)

        return paths

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[SupplyChainPath]:
        """
        Find the shortest path between two suppliers.

        Args:
            source_id: Source supplier ID
            target_id: Target supplier ID

        Returns:
            SupplyChainPath or None if no path exists
        """
        if source_id not in self._graph or target_id not in self._graph:
            return None

        try:
            path_nodes = nx.shortest_path(
                self._graph,
                source_id,
                target_id
            )

            edge_ids = []
            for i in range(len(path_nodes) - 1):
                edge_data = self._graph.edges.get(
                    (path_nodes[i], path_nodes[i + 1]), {}
                )
                rel_id = edge_data.get("relationship_id")
                if rel_id:
                    edge_ids.append(rel_id)

            return SupplyChainPath(
                nodes=path_nodes,
                edges=edge_ids,
            )

        except nx.NetworkXNoPath:
            return None

    # =========================================================================
    # Risk Propagation
    # =========================================================================

    def propagate_risk(
        self,
        risk_scores: Dict[str, float],
        decay_factor: float = 0.8,
    ) -> Dict[str, float]:
        """
        Propagate risk scores through the supply chain.

        Risk is propagated upstream (from suppliers to your company),
        with decay at each tier to reflect diminishing visibility.

        Args:
            risk_scores: Initial risk scores for suppliers
            decay_factor: Risk decay per tier (default 0.8)

        Returns:
            Dictionary of propagated risk scores
        """
        propagated_scores: Dict[str, float] = dict(risk_scores)

        # Process in reverse topological order (upstream to downstream)
        try:
            ordered_nodes = list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            # Graph has cycles, use simple iteration
            ordered_nodes = list(self._graph.nodes())

        for node_id in reversed(ordered_nodes):
            # Get risk from upstream suppliers
            predecessor_risks = []
            for pred_id in self._graph.predecessors(node_id):
                if pred_id in propagated_scores:
                    predecessor_risks.append(propagated_scores[pred_id])

            if predecessor_risks:
                # Propagate maximum risk with decay
                max_upstream_risk = max(predecessor_risks) * decay_factor
                current_risk = propagated_scores.get(node_id, 0.0)
                propagated_scores[node_id] = max(current_risk, max_upstream_risk)

        return propagated_scores

    def calculate_emission_allocation(
        self,
        target_id: str,
    ) -> Dict[str, Decimal]:
        """
        Calculate emission allocation from upstream suppliers.

        Allocates emissions based on spend proportion at each tier.

        Args:
            target_id: Target supplier ID

        Returns:
            Dictionary mapping supplier IDs to allocated emissions
        """
        allocations: Dict[str, Decimal] = {}

        # Get all upstream suppliers
        upstream = self.get_upstream_suppliers(target_id)

        for tier, suppliers in upstream.items():
            # Get total spend at this tier
            total_spend = Decimal("0")
            for supplier in suppliers:
                if supplier.annual_spend:
                    total_spend += supplier.annual_spend

            # Allocate emissions proportionally
            for supplier in suppliers:
                if supplier.annual_spend and total_spend > 0:
                    proportion = supplier.annual_spend / total_spend
                    if supplier.emission_factor_kg_co2e_per_usd:
                        emission = (
                            supplier.annual_spend *
                            supplier.emission_factor_kg_co2e_per_usd *
                            proportion
                        )
                        allocations[supplier.id] = emission

        return allocations

    # =========================================================================
    # Network Analysis
    # =========================================================================

    def compute_network_metrics(self) -> GraphMetrics:
        """
        Compute comprehensive network analysis metrics.

        Returns:
            GraphMetrics object with computed values
        """
        metrics = GraphMetrics()

        # Basic counts
        metrics.total_nodes = self._graph.number_of_nodes()
        metrics.total_edges = self._graph.number_of_edges()

        if metrics.total_nodes == 0:
            return metrics

        # Density
        metrics.density = nx.density(self._graph)

        # Average degree
        degrees = [d for _, d in self._graph.degree()]
        metrics.avg_degree = sum(degrees) / len(degrees) if degrees else 0

        # Tier distribution
        for tier, supplier_ids in self._supplier_by_tier.items():
            metrics.tier_distribution[tier.value] = len(supplier_ids)

        # Top suppliers by degree (most connections)
        degree_centrality = nx.degree_centrality(self._graph)
        sorted_by_degree = sorted(
            degree_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )
        metrics.top_suppliers_by_degree = [
            (node_id, self._graph.degree(node_id))
            for node_id, _ in sorted_by_degree[:10]
        ]

        # Top suppliers by spend
        spend_data = [
            (sid, s.annual_spend)
            for sid, s in self._suppliers.items()
            if s.annual_spend
        ]
        spend_data.sort(key=lambda x: x[1], reverse=True)
        metrics.top_suppliers_by_spend = spend_data[:10]

        # Clustering coefficient (for undirected view)
        undirected = self._graph.to_undirected()
        metrics.clustering_coefficient = nx.average_clustering(undirected)

        # Connected components
        metrics.connected_components = nx.number_weakly_connected_components(
            self._graph
        )

        # Critical suppliers (high betweenness centrality)
        betweenness = nx.betweenness_centrality(self._graph)
        sorted_by_betweenness = sorted(
            betweenness.items(),
            key=lambda x: x[1],
            reverse=True
        )
        # Consider top 10% as critical
        critical_count = max(1, len(sorted_by_betweenness) // 10)
        metrics.critical_suppliers = [
            node_id for node_id, _ in sorted_by_betweenness[:critical_count]
        ]

        return metrics

    def identify_pareto_suppliers(
        self,
        metric: str = "spend",
        threshold: float = 0.80,
    ) -> List[Tuple[str, Decimal]]:
        """
        Identify Pareto (80/20) suppliers by a given metric.

        Args:
            metric: Metric to analyze ("spend", "volume", "emissions")
            threshold: Cumulative proportion threshold (default 0.80)

        Returns:
            List of (supplier_id, metric_value) tuples
        """
        # Get metric values
        if metric == "spend":
            values = [
                (sid, s.annual_spend or Decimal("0"))
                for sid, s in self._suppliers.items()
            ]
        elif metric == "volume":
            # Aggregate volume from relationships
            volumes: Dict[str, Decimal] = defaultdict(Decimal)
            for rel in self._relationships.values():
                if rel.annual_volume:
                    volumes[rel.source_supplier_id] += rel.annual_volume
            values = list(volumes.items())
        else:
            # Default to spend
            values = [
                (sid, s.annual_spend or Decimal("0"))
                for sid, s in self._suppliers.items()
            ]

        # Sort by value descending
        values.sort(key=lambda x: x[1], reverse=True)

        # Find Pareto suppliers
        total = sum(v for _, v in values)
        if total == 0:
            return []

        cumulative = Decimal("0")
        pareto_suppliers = []

        for supplier_id, value in values:
            cumulative += value
            pareto_suppliers.append((supplier_id, value))
            if cumulative / total >= Decimal(str(threshold)):
                break

        return pareto_suppliers

    def find_concentration_risks(
        self,
        concentration_threshold: float = 0.30,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify concentration risks in the supply chain.

        Analyzes:
        - Supplier concentration (single supplier dominance)
        - Geographic concentration (country risk)
        - Commodity concentration (material risk)

        Args:
            concentration_threshold: Threshold for concentration warning

        Returns:
            Dictionary of risk categories with details
        """
        risks: Dict[str, List[Dict[str, Any]]] = {
            "supplier_concentration": [],
            "geographic_concentration": [],
            "commodity_concentration": [],
        }

        # Supplier concentration
        total_spend = sum(
            s.annual_spend or Decimal("0")
            for s in self._suppliers.values()
        )

        if total_spend > 0:
            for supplier in self._suppliers.values():
                if supplier.annual_spend:
                    proportion = float(supplier.annual_spend / total_spend)
                    if proportion >= concentration_threshold:
                        risks["supplier_concentration"].append({
                            "supplier_id": supplier.id,
                            "supplier_name": supplier.name,
                            "spend_proportion": proportion,
                            "spend_amount": str(supplier.annual_spend),
                        })

        # Geographic concentration
        country_spend: Dict[str, Decimal] = defaultdict(Decimal)
        for supplier in self._suppliers.values():
            if supplier.country_code and supplier.annual_spend:
                country_spend[supplier.country_code] += supplier.annual_spend

        for country, spend in country_spend.items():
            if total_spend > 0:
                proportion = float(spend / total_spend)
                if proportion >= concentration_threshold:
                    risks["geographic_concentration"].append({
                        "country": country,
                        "spend_proportion": proportion,
                        "spend_amount": str(spend),
                        "supplier_count": len(self._suppliers_by_country[country]),
                    })

        # Commodity concentration
        commodity_suppliers: Dict[CommodityType, int] = {}
        for commodity, supplier_ids in self._suppliers_by_commodity.items():
            count = len(supplier_ids)
            if count <= 2:  # Single or dual source risk
                commodity_suppliers[commodity] = count
                risks["commodity_concentration"].append({
                    "commodity": commodity.value,
                    "supplier_count": count,
                    "risk_level": "high" if count == 1 else "medium",
                })

        return risks

    # =========================================================================
    # Export and Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Export graph to dictionary representation.

        Returns:
            Dictionary with nodes, edges, and metadata
        """
        nodes = []
        for node_id, attrs in self._graph.nodes(data=True):
            node_data = {"id": node_id, **attrs}
            nodes.append(node_data)

        edges = []
        for source, target, attrs in self._graph.edges(data=True):
            edge_data = {"source": source, "target": target, **attrs}
            edges.append(edge_data)

        return {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "nodes": nodes,
            "edges": edges,
            "supplier_count": len(self._suppliers),
            "relationship_count": len(self._relationships),
        }

    def to_networkx(self) -> nx.DiGraph:
        """
        Get a copy of the underlying NetworkX graph.

        Returns:
            Copy of the NetworkX DiGraph
        """
        return self._graph.copy()

    def export_for_neo4j(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Export graph data for Neo4j import.

        Returns:
            Tuple of (node_records, relationship_records) for Neo4j import
        """
        nodes = []
        for supplier_id, supplier in self._suppliers.items():
            nodes.append({
                "id": supplier_id,
                "labels": ["Supplier"],
                "properties": {
                    "name": supplier.name,
                    "tier": supplier.tier.value,
                    "country": supplier.country_code,
                    "status": supplier.status.value,
                    "annual_spend": float(supplier.annual_spend) if supplier.annual_spend else None,
                }
            })

        relationships = []
        for rel_id, rel in self._relationships.items():
            relationships.append({
                "id": rel_id,
                "type": rel.relationship_type.value.upper(),
                "start_node": rel.source_supplier_id,
                "end_node": rel.target_supplier_id,
                "properties": {
                    "active": rel.active,
                    "verified": rel.verified,
                    "annual_spend": float(rel.annual_spend) if rel.annual_spend else None,
                }
            })

        return nodes, relationships

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SupplyChainGraph":
        """
        Create graph from dictionary representation.

        Args:
            data: Dictionary with nodes, edges, and metadata

        Returns:
            New SupplyChainGraph instance
        """
        graph = cls(
            company_id=data.get("company_id"),
            company_name=data.get("company_name"),
        )

        # Reconstruct suppliers from nodes
        for node_data in data.get("nodes", []):
            if node_data.get("entity_type") == "supplier":
                supplier = Supplier(
                    id=node_data["id"],
                    name=node_data.get("name", ""),
                    tier=SupplierTier(node_data.get("tier", 0)),
                    country_code=node_data.get("country"),
                )
                graph.add_supplier(supplier)

        # Reconstruct relationships from edges
        for edge_data in data.get("edges", []):
            if edge_data.get("relationship_type") not in ["belongs_to"]:
                relationship = SupplierRelationship(
                    id=edge_data.get("relationship_id", f"REL-{edge_data['source']}-{edge_data['target']}"),
                    source_supplier_id=edge_data["source"],
                    target_supplier_id=edge_data["target"],
                    relationship_type=RelationshipType(edge_data.get("relationship_type", "supplier")),
                    materials=edge_data.get("materials", []),
                    products=edge_data.get("products", []),
                    active=edge_data.get("active", True),
                    verified=edge_data.get("verified", False),
                )
                graph.add_relationship(relationship)

        return graph
