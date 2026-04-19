"""
Supply Chain Visualization Module.

This module provides data export utilities for various visualization
libraries and formats, enabling interactive supply chain dashboards
and analysis tools.

Supported Formats:
- D3.js network graphs (force-directed, hierarchical)
- Sankey diagrams for material/spend flows
- Geographic maps (GeoJSON, Leaflet-compatible)
- Heat maps for risk visualization
- Tree maps for spend/emission breakdown

Example:
    >>> from greenlang.supply_chain.visualization import SupplyChainVisualizer
    >>> viz = SupplyChainVisualizer(supply_chain_graph)
    >>>
    >>> # Export for D3.js force-directed graph
    >>> d3_data = viz.export_d3_network()
    >>>
    >>> # Export Sankey diagram data
    >>> sankey_data = viz.export_sankey_diagram(metric="spend")
    >>>
    >>> # Export geographic map data
    >>> geo_data = viz.export_geo_map()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Set
from collections import defaultdict

from greenlang.supply_chain.models.entity import (
    Supplier,
    Facility,
    SupplierTier,
    CommodityType,
)
from greenlang.supply_chain.graph.supply_chain_graph import (
    SupplyChainGraph,
    MaterialFlow,
)

logger = logging.getLogger(__name__)


class ColorScheme(Enum):
    """Predefined color schemes for visualizations."""
    TIER = "tier"
    RISK = "risk"
    COMMODITY = "commodity"
    COUNTRY = "country"
    EMISSION = "emission"


@dataclass
class D3Node:
    """
    Node for D3.js network visualization.

    Attributes:
        id: Unique node identifier
        name: Display name
        group: Group for coloring
        tier: Supply chain tier
        value: Size value (spend, emissions, etc.)
        country: Country code
        risk_score: Risk score (0-100)
        metadata: Additional attributes
    """
    id: str
    name: str
    group: int = 0
    tier: int = 0
    value: float = 1.0
    country: Optional[str] = None
    risk_score: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        result = {
            "id": self.id,
            "name": self.name,
            "group": self.group,
            "tier": self.tier,
            "value": self.value,
        }
        if self.country:
            result["country"] = self.country
        if self.risk_score is not None:
            result["riskScore"] = self.risk_score
        if self.latitude is not None:
            result["latitude"] = self.latitude
        if self.longitude is not None:
            result["longitude"] = self.longitude
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class D3Link:
    """
    Link/edge for D3.js network visualization.

    Attributes:
        source: Source node ID
        target: Target node ID
        value: Link strength/width
        type: Relationship type
        material: Material being transferred
        verified: Whether link is verified
    """
    source: str
    target: str
    value: float = 1.0
    type: str = "supplier"
    material: Optional[str] = None
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "source": self.source,
            "target": self.target,
            "value": self.value,
            "type": self.type,
            "material": self.material,
            "verified": self.verified,
        }


@dataclass
class D3NetworkData:
    """
    Complete D3.js network data structure.

    Compatible with D3 force-directed graph.
    """
    nodes: List[D3Node] = field(default_factory=list)
    links: List[D3Link] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [l.to_dict() for l in self.links],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class SankeyNode:
    """Node for Sankey diagram."""
    id: str
    name: str
    category: str = ""
    value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "value": self.value,
        }


@dataclass
class SankeyLink:
    """Link for Sankey diagram."""
    source: int  # Source node index
    target: int  # Target node index
    value: float
    material: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "source": self.source,
            "target": self.target,
            "value": self.value,
        }
        if self.material:
            result["material"] = self.material
        return result


@dataclass
class SankeyData:
    """
    Sankey diagram data structure.

    Compatible with D3-sankey and ECharts.
    """
    nodes: List[SankeyNode] = field(default_factory=list)
    links: List[SankeyLink] = field(default_factory=list)
    unit: str = "USD"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [l.to_dict() for l in self.links],
            "unit": self.unit,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class GeoFeature:
    """GeoJSON feature for map visualization."""
    id: str
    name: str
    latitude: float
    longitude: float
    feature_type: str = "supplier"  # supplier, facility, plot
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON feature."""
        return {
            "type": "Feature",
            "id": self.id,
            "geometry": {
                "type": "Point",
                "coordinates": [self.longitude, self.latitude]
            },
            "properties": {
                "name": self.name,
                "featureType": self.feature_type,
                **self.properties
            }
        }


@dataclass
class GeoMapData:
    """
    Geographic map data structure.

    Outputs GeoJSON FeatureCollection compatible with Leaflet,
    Mapbox, and other mapping libraries.
    """
    features: List[GeoFeature] = field(default_factory=list)
    center: Tuple[float, float] = (0.0, 0.0)  # (lat, lon)
    zoom: int = 2
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON FeatureCollection."""
        return {
            "type": "FeatureCollection",
            "features": [f.to_geojson() for f in self.features],
            "metadata": {
                "center": list(self.center),
                "zoom": self.zoom,
                "bounds": self.bounds,
                "featureCount": len(self.features),
            }
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_geojson(), indent=indent)


@dataclass
class HeatMapCell:
    """Cell for heat map visualization."""
    x: str  # X-axis label (e.g., country)
    y: str  # Y-axis label (e.g., risk category)
    value: float
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "value": self.value,
            "label": self.label,
        }


@dataclass
class HeatMapData:
    """
    Heat map data structure.

    Compatible with D3 heat maps and ECharts.
    """
    cells: List[HeatMapCell] = field(default_factory=list)
    x_labels: List[str] = field(default_factory=list)
    y_labels: List[str] = field(default_factory=list)
    min_value: float = 0.0
    max_value: float = 100.0
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": [c.to_dict() for c in self.cells],
            "xLabels": self.x_labels,
            "yLabels": self.y_labels,
            "minValue": self.min_value,
            "maxValue": self.max_value,
            "unit": self.unit,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class SupplyChainVisualizer:
    """
    Supply chain visualization data generator.

    Transforms supply chain graph data into formats suitable for
    various visualization libraries and dashboards.

    Example:
        >>> from greenlang.supply_chain.graph import SupplyChainGraph
        >>> from greenlang.supply_chain.visualization import SupplyChainVisualizer
        >>>
        >>> graph = SupplyChainGraph()
        >>> # ... add suppliers and relationships ...
        >>>
        >>> viz = SupplyChainVisualizer(graph)
        >>>
        >>> # Export for D3.js
        >>> d3_data = viz.export_d3_network(color_by="tier")
        >>> with open("network.json", "w") as f:
        ...     f.write(d3_data.to_json())
        >>>
        >>> # Export Sankey diagram
        >>> sankey = viz.export_sankey_diagram(metric="spend")
        >>>
        >>> # Export geographic map
        >>> geo = viz.export_geo_map()
    """

    # Color palettes
    TIER_COLORS = {
        1: "#1f77b4",  # Blue - Tier 1
        2: "#ff7f0e",  # Orange - Tier 2
        3: "#2ca02c",  # Green - Tier 3
        99: "#d62728",  # Red - Tier N
        0: "#7f7f7f",  # Gray - Unknown
    }

    RISK_COLORS = {
        "low": "#2ca02c",  # Green
        "standard": "#ffbb00",  # Yellow
        "high": "#d62728",  # Red
        "unknown": "#7f7f7f",  # Gray
    }

    COMMODITY_COLORS = {
        "cattle": "#8b4513",
        "cocoa": "#654321",
        "coffee": "#6f4e37",
        "oil_palm": "#ffa500",
        "rubber": "#2f4f4f",
        "soya": "#9acd32",
        "wood": "#deb887",
    }

    def __init__(
        self,
        supply_chain_graph: SupplyChainGraph,
        risk_scores: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the visualizer.

        Args:
            supply_chain_graph: Supply chain graph instance
            risk_scores: Optional risk scores by supplier ID
        """
        self.graph = supply_chain_graph
        self.risk_scores = risk_scores or {}

        logger.info("SupplyChainVisualizer initialized")

    def export_d3_network(
        self,
        color_by: str = "tier",
        size_by: str = "spend",
        include_facilities: bool = False,
        max_nodes: Optional[int] = None,
    ) -> D3NetworkData:
        """
        Export supply chain as D3.js network data.

        Args:
            color_by: Attribute for node coloring (tier, risk, country, commodity)
            size_by: Attribute for node sizing (spend, emissions, equal)
            include_facilities: Include facility nodes
            max_nodes: Maximum number of nodes (for performance)

        Returns:
            D3NetworkData structure
        """
        nodes: List[D3Node] = []
        links: List[D3Link] = []
        node_ids: Set[str] = set()

        # Get graph data
        graph_data = self.graph.to_dict()

        # Process nodes (suppliers)
        for node_data in graph_data.get("nodes", []):
            if node_data.get("entity_type") == "facility" and not include_facilities:
                continue

            if max_nodes and len(nodes) >= max_nodes:
                break

            supplier = self.graph.get_supplier(node_data["id"])
            if not supplier:
                continue

            # Determine group (color)
            group = self._get_node_group(supplier, color_by)

            # Determine size value
            value = self._get_node_value(supplier, size_by)

            # Get location if available
            lat, lon = None, None
            facilities = [
                f for f in self.graph._facilities.values()
                if f.supplier_id == supplier.id and f.location
            ]
            if facilities:
                lat = facilities[0].location.latitude
                lon = facilities[0].location.longitude

            node = D3Node(
                id=supplier.id,
                name=supplier.name,
                group=group,
                tier=supplier.tier.value,
                value=value,
                country=supplier.country_code,
                risk_score=self.risk_scores.get(supplier.id),
                latitude=lat,
                longitude=lon,
                metadata={
                    "status": supplier.status.value,
                    "commodities": [c.value for c in supplier.commodities],
                }
            )
            nodes.append(node)
            node_ids.add(supplier.id)

        # Process edges (relationships)
        for edge_data in graph_data.get("edges", []):
            source = edge_data.get("source")
            target = edge_data.get("target")

            if source not in node_ids or target not in node_ids:
                continue

            # Get spend or volume for link strength
            value = edge_data.get("annual_spend", 1) or 1

            link = D3Link(
                source=source,
                target=target,
                value=float(value),
                type=edge_data.get("relationship_type", "supplier"),
                verified=edge_data.get("verified", False),
            )
            links.append(link)

        return D3NetworkData(
            nodes=nodes,
            links=links,
            metadata={
                "colorBy": color_by,
                "sizeBy": size_by,
                "nodeCount": len(nodes),
                "linkCount": len(links),
                "generatedAt": datetime.utcnow().isoformat(),
            }
        )

    def export_d3_hierarchy(
        self,
        root_id: Optional[str] = None,
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        """
        Export supply chain as hierarchical tree data.

        Suitable for D3 tree, treemap, or sunburst visualizations.

        Args:
            root_id: Root node ID (default: focal company)
            max_depth: Maximum tree depth

        Returns:
            Hierarchical tree structure
        """
        root_id = root_id or self.graph.company_id

        if not root_id:
            return {"name": "Root", "children": []}

        def build_tree(node_id: str, depth: int) -> Dict[str, Any]:
            supplier = self.graph.get_supplier(node_id)
            if not supplier:
                return {"name": node_id, "children": []}

            node = {
                "name": supplier.name,
                "id": supplier.id,
                "tier": supplier.tier.value,
                "value": float(supplier.annual_spend or 0),
                "country": supplier.country_code,
            }

            if depth < max_depth:
                children = []
                for pred in self.graph.get_direct_suppliers(node_id):
                    child = build_tree(pred.id, depth + 1)
                    children.append(child)
                if children:
                    node["children"] = children

            return node

        return build_tree(root_id, 0)

    def export_sankey_diagram(
        self,
        metric: str = "spend",
        group_by: str = "tier",
        min_value: float = 0,
    ) -> SankeyData:
        """
        Export supply chain flows as Sankey diagram data.

        Args:
            metric: Flow metric (spend, volume, emissions)
            group_by: Node grouping (tier, country, commodity)
            min_value: Minimum value to include

        Returns:
            SankeyData structure
        """
        nodes: List[SankeyNode] = []
        links: List[SankeyLink] = []
        node_index: Dict[str, int] = {}

        # Create nodes based on grouping
        if group_by == "tier":
            categories = ["Tier 1", "Tier 2", "Tier 3", "Tier N", "Company"]
            for i, cat in enumerate(categories):
                nodes.append(SankeyNode(
                    id=f"tier_{i}",
                    name=cat,
                    category="tier",
                ))
                node_index[f"tier_{i}"] = i
        else:
            # Create nodes for individual suppliers
            for supplier in self.graph._suppliers.values():
                idx = len(nodes)
                nodes.append(SankeyNode(
                    id=supplier.id,
                    name=supplier.name,
                    category=f"tier_{supplier.tier.value}",
                    value=float(supplier.annual_spend or 0),
                ))
                node_index[supplier.id] = idx

        # Create links
        if group_by == "tier":
            # Aggregate flows between tiers
            tier_flows: Dict[Tuple[int, int], float] = defaultdict(float)

            for rel in self.graph._relationships.values():
                source_supplier = self.graph.get_supplier(rel.source_supplier_id)
                target_supplier = self.graph.get_supplier(rel.target_supplier_id)

                if source_supplier and target_supplier:
                    source_tier = source_supplier.tier.value
                    target_tier = target_supplier.tier.value

                    if metric == "spend" and rel.annual_spend:
                        value = float(rel.annual_spend)
                    elif metric == "volume" and rel.annual_volume:
                        value = float(rel.annual_volume)
                    else:
                        value = 1.0

                    tier_flows[(source_tier, target_tier)] += value

            for (s_tier, t_tier), value in tier_flows.items():
                if value >= min_value:
                    s_idx = node_index.get(f"tier_{s_tier}", 0)
                    t_idx = node_index.get(f"tier_{t_tier}", 0)
                    links.append(SankeyLink(
                        source=s_idx,
                        target=t_idx,
                        value=value,
                    ))
        else:
            # Individual supplier flows
            for rel in self.graph._relationships.values():
                if rel.source_supplier_id in node_index and rel.target_supplier_id in node_index:
                    if metric == "spend" and rel.annual_spend:
                        value = float(rel.annual_spend)
                    elif metric == "volume" and rel.annual_volume:
                        value = float(rel.annual_volume)
                    else:
                        value = 1.0

                    if value >= min_value:
                        links.append(SankeyLink(
                            source=node_index[rel.source_supplier_id],
                            target=node_index[rel.target_supplier_id],
                            value=value,
                        ))

        return SankeyData(
            nodes=nodes,
            links=links,
            unit="USD" if metric == "spend" else metric,
        )

    def export_geo_map(
        self,
        include_facilities: bool = True,
        filter_country: Optional[str] = None,
    ) -> GeoMapData:
        """
        Export supplier locations as GeoJSON.

        Args:
            include_facilities: Include facility locations
            filter_country: Filter by country code

        Returns:
            GeoMapData structure
        """
        features: List[GeoFeature] = []
        lats, lons = [], []

        # Add facilities
        if include_facilities:
            for facility in self.graph._facilities.values():
                if not facility.location:
                    continue

                if filter_country:
                    supplier = self.graph.get_supplier(facility.supplier_id)
                    if supplier and supplier.country_code != filter_country:
                        continue

                feature = GeoFeature(
                    id=facility.id,
                    name=facility.name,
                    latitude=facility.location.latitude,
                    longitude=facility.location.longitude,
                    feature_type="facility",
                    properties={
                        "supplierId": facility.supplier_id,
                        "facilityType": facility.facility_type,
                        "certifications": facility.certifications,
                    }
                )
                features.append(feature)
                lats.append(facility.location.latitude)
                lons.append(facility.location.longitude)

        # Add suppliers (if they have location data from facilities)
        supplier_locations: Dict[str, Tuple[float, float]] = {}
        for facility in self.graph._facilities.values():
            if facility.location and facility.supplier_id not in supplier_locations:
                supplier_locations[facility.supplier_id] = (
                    facility.location.latitude,
                    facility.location.longitude
                )

        for supplier_id, (lat, lon) in supplier_locations.items():
            supplier = self.graph.get_supplier(supplier_id)
            if not supplier:
                continue

            if filter_country and supplier.country_code != filter_country:
                continue

            feature = GeoFeature(
                id=supplier_id,
                name=supplier.name,
                latitude=lat,
                longitude=lon,
                feature_type="supplier",
                properties={
                    "tier": supplier.tier.value,
                    "country": supplier.country_code,
                    "riskScore": self.risk_scores.get(supplier_id),
                    "commodities": [c.value for c in supplier.commodities],
                }
            )
            features.append(feature)

        # Calculate center and bounds
        center = (0.0, 0.0)
        bounds = None
        if lats and lons:
            center = (sum(lats) / len(lats), sum(lons) / len(lons))
            bounds = ((min(lats), min(lons)), (max(lats), max(lons)))

        return GeoMapData(
            features=features,
            center=center,
            zoom=3,
            bounds=bounds,
        )

    def export_risk_heatmap(
        self,
        risk_categories: Optional[List[str]] = None,
        group_by: str = "country",
    ) -> HeatMapData:
        """
        Export risk data as heat map.

        Args:
            risk_categories: Risk categories to include
            group_by: Grouping dimension (country, tier, commodity)

        Returns:
            HeatMapData structure
        """
        default_categories = [
            "environmental",
            "social",
            "governance",
            "geographic",
            "concentration",
        ]
        risk_categories = risk_categories or default_categories

        cells: List[HeatMapCell] = []
        x_labels: Set[str] = set()
        y_labels = risk_categories

        # Group suppliers
        if group_by == "country":
            groups = self.graph._suppliers_by_country
        elif group_by == "tier":
            groups = {
                f"Tier {t.value}": ids
                for t, ids in self.graph._supplier_by_tier.items()
            }
        else:
            groups = {"All": set(self.graph._suppliers.keys())}

        # Generate heat map data
        for group_name, supplier_ids in groups.items():
            x_labels.add(group_name)

            for risk_cat in risk_categories:
                # Calculate average risk for this group/category
                # In a real implementation, this would use actual risk scores
                risk_values = []
                for sid in supplier_ids:
                    base_risk = self.risk_scores.get(sid, 50)
                    # Simulate category-specific variations
                    category_factor = {
                        "environmental": 1.0,
                        "social": 0.8,
                        "governance": 0.7,
                        "geographic": 1.2,
                        "concentration": 0.9,
                    }.get(risk_cat, 1.0)
                    risk_values.append(base_risk * category_factor)

                avg_risk = sum(risk_values) / len(risk_values) if risk_values else 50

                cells.append(HeatMapCell(
                    x=group_name,
                    y=risk_cat,
                    value=avg_risk,
                    label=f"{avg_risk:.1f}",
                ))

        return HeatMapData(
            cells=cells,
            x_labels=sorted(x_labels),
            y_labels=y_labels,
            min_value=0.0,
            max_value=100.0,
            unit="Risk Score",
        )

    def export_treemap(
        self,
        metric: str = "spend",
        group_by: str = "tier",
    ) -> Dict[str, Any]:
        """
        Export data for treemap visualization.

        Args:
            metric: Size metric (spend, emissions)
            group_by: Grouping dimension (tier, country, commodity)

        Returns:
            Hierarchical treemap data
        """
        root = {
            "name": "Supply Chain",
            "children": [],
        }

        # Group data
        if group_by == "tier":
            for tier in [SupplierTier.TIER_1, SupplierTier.TIER_2, SupplierTier.TIER_3, SupplierTier.TIER_N]:
                suppliers = self.graph.get_suppliers_by_tier(tier)
                if suppliers:
                    tier_node = {
                        "name": f"Tier {tier.value}",
                        "children": [],
                    }
                    for supplier in suppliers:
                        value = float(supplier.annual_spend or 0) if metric == "spend" else 1
                        tier_node["children"].append({
                            "name": supplier.name,
                            "value": value,
                            "id": supplier.id,
                            "country": supplier.country_code,
                        })
                    root["children"].append(tier_node)

        elif group_by == "country":
            for country, supplier_ids in self.graph._suppliers_by_country.items():
                country_node = {
                    "name": country,
                    "children": [],
                }
                for sid in supplier_ids:
                    supplier = self.graph.get_supplier(sid)
                    if supplier:
                        value = float(supplier.annual_spend or 0) if metric == "spend" else 1
                        country_node["children"].append({
                            "name": supplier.name,
                            "value": value,
                            "id": supplier.id,
                            "tier": supplier.tier.value,
                        })
                root["children"].append(country_node)

        return root

    def _get_node_group(self, supplier: Supplier, color_by: str) -> int:
        """Determine node group for coloring."""
        if color_by == "tier":
            return supplier.tier.value
        elif color_by == "risk":
            risk = self.risk_scores.get(supplier.id, 50)
            if risk < 30:
                return 0  # Low
            elif risk < 70:
                return 1  # Medium
            else:
                return 2  # High
        elif color_by == "country":
            # Hash country code to consistent group
            if supplier.country_code:
                return hash(supplier.country_code) % 10
            return 0
        return 0

    def _get_node_value(self, supplier: Supplier, size_by: str) -> float:
        """Determine node value for sizing."""
        if size_by == "spend":
            return float(supplier.annual_spend or 1)
        elif size_by == "emissions":
            # Would need emission data
            return float(supplier.annual_spend or 1) * 0.5
        elif size_by == "risk":
            return self.risk_scores.get(supplier.id, 50)
        return 1.0

    def export_dashboard_data(self) -> Dict[str, Any]:
        """
        Export comprehensive dashboard data package.

        Returns:
            Complete dashboard data with all visualizations
        """
        return {
            "generatedAt": datetime.utcnow().isoformat(),
            "summary": {
                "supplierCount": self.graph.supplier_count,
                "relationshipCount": self.graph.relationship_count,
                "facilityCount": len(self.graph._facilities),
            },
            "network": self.export_d3_network().to_dict(),
            "hierarchy": self.export_d3_hierarchy(),
            "sankey": self.export_sankey_diagram().to_dict(),
            "geoMap": self.export_geo_map().to_geojson(),
            "riskHeatmap": self.export_risk_heatmap().to_dict(),
            "treemap": self.export_treemap(),
        }
