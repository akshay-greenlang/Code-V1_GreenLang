"""
CombustionCausalGraph - Causal graph builder for combustion systems.

This module implements causal graph construction and validation for combustion
process analysis.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class EdgeType(str, Enum):
    PHYSICS_BASED = "physics_based"
    LEARNED = "learned"
    DOMAIN_EXPERT = "domain_expert"
    HYBRID = "hybrid"


class NodeType(str, Enum):
    INPUT = "input"
    INTERMEDIATE = "intermediate"
    OUTPUT = "output"
    LATENT = "latent"


class CausalNode(BaseModel):
    name: str = Field(..., description="Unique node identifier")
    node_type: NodeType = Field(..., description="Type of node")
    description: str = Field("", description="Human-readable description")
    unit: str = Field("", description="Measurement unit")
    bounds: Optional[Tuple[float, float]] = Field(None, description="(min, max) bounds")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class CausalEdge(BaseModel):
    source: str = Field(..., description="Source node (cause)")
    target: str = Field(..., description="Target node (effect)")
    edge_type: EdgeType = Field(..., description="Type of edge")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Causal strength")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Edge confidence")
    mechanism: str = Field("", description="Causal mechanism description")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class GraphValidationResult(BaseModel):
    is_valid: bool = Field(..., description="Overall validation status")
    is_acyclic: bool = Field(..., description="No cycles present")
    is_connected: bool = Field(..., description="Graph is connected")
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field("", description="SHA-256 hash of validation")


class CausalGraph(BaseModel):
    name: str = Field(..., description="Graph name")
    nodes: Dict[str, CausalNode] = Field(default_factory=dict)
    edges: List[CausalEdge] = Field(default_factory=list)
    domain_knowledge: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field("", description="SHA-256 hash")

    def to_networkx(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for node_name, node in self.nodes.items():
            G.add_node(node_name, **node.dict())
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, weight=edge.weight,
                       confidence=edge.confidence, edge_type=edge.edge_type,
                       mechanism=edge.mechanism)
        return G

    def get_parents(self, node: str) -> List[str]:
        return [e.source for e in self.edges if e.target == node]

    def get_children(self, node: str) -> List[str]:
        return [e.target for e in self.edges if e.source == node]

    def get_ancestors(self, node: str) -> Set[str]:
        G = self.to_networkx()
        return nx.ancestors(G, node)

    def get_descendants(self, node: str) -> Set[str]:
        G = self.to_networkx()
        return nx.descendants(G, node)


class CombustionCausalGraph:
    DEFAULT_NODES: Dict[str, Dict[str, Any]] = {
        "fuel_flow": {"node_type": NodeType.INPUT, "description": "Fuel mass flow rate", "unit": "kg/s", "bounds": (0.0, 100.0)},
        "air_flow": {"node_type": NodeType.INPUT, "description": "Combustion air flow rate", "unit": "kg/s", "bounds": (0.0, 1000.0)},
        "O2": {"node_type": NodeType.INTERMEDIATE, "description": "Flue gas oxygen", "unit": "%", "bounds": (0.0, 21.0)},
        "CO": {"node_type": NodeType.OUTPUT, "description": "Carbon monoxide", "unit": "ppm", "bounds": (0.0, 10000.0)},
        "NOx": {"node_type": NodeType.OUTPUT, "description": "Nitrogen oxides", "unit": "ppm", "bounds": (0.0, 1000.0)},
        "flame_temp": {"node_type": NodeType.INTERMEDIATE, "description": "Flame temperature", "unit": "K", "bounds": (300.0, 2500.0)},
        "stability": {"node_type": NodeType.OUTPUT, "description": "Flame stability index", "unit": "dimensionless", "bounds": (0.0, 1.0)},
        "efficiency": {"node_type": NodeType.OUTPUT, "description": "Combustion efficiency", "unit": "%", "bounds": (0.0, 100.0)}
    }

    PHYSICS_EDGES: List[Dict[str, Any]] = [
        {"source": "fuel_flow", "target": "flame_temp", "mechanism": "Fuel energy release determines flame temperature", "weight": 0.9},
        {"source": "air_flow", "target": "flame_temp", "mechanism": "Air-fuel ratio affects combustion temperature", "weight": 0.8},
        {"source": "fuel_flow", "target": "O2", "mechanism": "Fuel consumption depletes oxygen", "weight": 0.85},
        {"source": "air_flow", "target": "O2", "mechanism": "Air supplies oxygen for combustion", "weight": 0.95},
        {"source": "flame_temp", "target": "NOx", "mechanism": "Thermal NOx formation exponential with temperature", "weight": 0.95},
        {"source": "flame_temp", "target": "CO", "mechanism": "Temperature affects CO oxidation rate", "weight": 0.7},
        {"source": "O2", "target": "CO", "mechanism": "Excess O2 promotes CO burnout", "weight": 0.8},
        {"source": "O2", "target": "efficiency", "mechanism": "O2 level indicates excess air losses", "weight": 0.75},
        {"source": "fuel_flow", "target": "stability", "mechanism": "Fuel flow affects flame anchoring", "weight": 0.7},
        {"source": "air_flow", "target": "stability", "mechanism": "Air velocity affects flame stability", "weight": 0.75},
        {"source": "flame_temp", "target": "stability", "mechanism": "Temperature affects combustion intensity", "weight": 0.6},
        {"source": "CO", "target": "efficiency", "mechanism": "CO represents unburned fuel loss", "weight": 0.6},
        {"source": "flame_temp", "target": "efficiency", "mechanism": "Higher temp improves heat transfer", "weight": 0.5},
        {"source": "stability", "target": "efficiency", "mechanism": "Unstable flames have poor efficiency", "weight": 0.4}
    ]

    def __init__(self):
        self._nx_graph: Optional[nx.DiGraph] = None
        logger.info("CombustionCausalGraph initialized")

    def build_graph(self, variables: List[str], domain_knowledge: Dict[str, Any]) -> CausalGraph:
        start_time = datetime.now()
        logger.info(f"Building causal graph for {len(variables)} variables")
        if not variables:
            raise ValueError("Variables list cannot be empty")
        
        nodes: Dict[str, CausalNode] = {}
        for var in variables:
            if var in self.DEFAULT_NODES:
                node_config = self.DEFAULT_NODES[var]
                nodes[var] = CausalNode(name=var, node_type=node_config["node_type"],
                    description=node_config["description"], unit=node_config["unit"],
                    bounds=node_config.get("bounds"))
            else:
                logger.warning(f"Unknown variable {var}, creating with default settings")
                nodes[var] = CausalNode(name=var, node_type=NodeType.INTERMEDIATE,
                    description=f"Custom variable: {var}")
        
        edges: List[CausalEdge] = []
        for edge_def in self.PHYSICS_EDGES:
            if edge_def["source"] in variables and edge_def["target"] in variables:
                edges.append(CausalEdge(source=edge_def["source"], target=edge_def["target"],
                    edge_type=EdgeType.PHYSICS_BASED, weight=edge_def.get("weight", 1.0),
                    confidence=1.0, mechanism=edge_def.get("mechanism", "")))
        
        edges = self._apply_domain_knowledge(edges, domain_knowledge, variables)
        graph_data = f"{variables}{domain_knowledge}{len(edges)}"
        provenance_hash = hashlib.sha256(graph_data.encode()).hexdigest()
        
        graph = CausalGraph(name="CombustionCausalGraph", nodes=nodes, edges=edges,
            domain_knowledge=domain_knowledge,
            metadata={"build_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                      "variable_count": len(variables), "edge_count": len(edges)},
            provenance_hash=provenance_hash)
        
        logger.info(f"Built causal graph with {len(nodes)} nodes and {len(edges)} edges")
        return graph

    def _apply_domain_knowledge(self, edges: List[CausalEdge], domain_knowledge: Dict[str, Any],
                                 variables: List[str]) -> List[CausalEdge]:
        fuel_type = domain_knowledge.get("fuel_type", "natural_gas")
        if fuel_type == "coal":
            for edge in edges:
                if edge.target == "efficiency":
                    edge.weight *= 0.95
        
        burner_type = domain_knowledge.get("burner_type", "standard")
        if burner_type == "low_NOx":
            for edge in edges:
                if edge.target == "NOx":
                    edge.mechanism += " (modified for low-NOx burner)"
                    edge.weight *= 0.8
        
        additional_edges = domain_knowledge.get("additional_edges", [])
        for edge_def in additional_edges:
            if edge_def["source"] in variables and edge_def["target"] in variables:
                edges.append(CausalEdge(source=edge_def["source"], target=edge_def["target"],
                    edge_type=EdgeType.DOMAIN_EXPERT, weight=edge_def.get("weight", 0.5),
                    confidence=edge_def.get("confidence", 0.8),
                    mechanism=edge_def.get("mechanism", "Domain expert defined")))
        
        remove_edges = domain_knowledge.get("remove_edges", [])
        edges = [e for e in edges if (e.source, e.target) not in
                 [(r["source"], r["target"]) for r in remove_edges]]
        return edges

    def add_learned_edges(self, graph: CausalGraph, data: pd.DataFrame, method: str = "pc") -> CausalGraph:
        start_time = datetime.now()
        logger.info(f"Learning edges from {len(data)} observations using {method}")
        existing_edges = {(e.source, e.target) for e in graph.edges}
        new_edges = []
        
        if method == "correlation":
            new_edges = self._learn_from_correlation(data, graph, existing_edges)
        elif method == "pc":
            new_edges = self._learn_pc_algorithm(data, graph, existing_edges)
        elif method == "ges":
            new_edges = self._learn_ges_algorithm(data, graph, existing_edges)
        else:
            raise ValueError(f"Unknown learning method: {method}")
        
        updated_edges = list(graph.edges) + new_edges
        graph_data = f"{graph.name}{len(updated_edges)}{method}"
        provenance_hash = hashlib.sha256(graph_data.encode()).hexdigest()
        
        updated_graph = CausalGraph(name=graph.name, nodes=graph.nodes, edges=updated_edges,
            domain_knowledge=graph.domain_knowledge,
            metadata={**graph.metadata, "learned_edges_count": len(new_edges),
                      "learning_method": method,
                      "learning_time_ms": (datetime.now() - start_time).total_seconds() * 1000},
            provenance_hash=provenance_hash)
        
        logger.info(f"Added {len(new_edges)} learned edges to graph")
        return updated_graph

    def _learn_from_correlation(self, data: pd.DataFrame, graph: CausalGraph,
                                  existing_edges: Set[Tuple[str, str]]) -> List[CausalEdge]:
        new_edges = []
        variables = list(graph.nodes.keys())
        corr_matrix = data[variables].corr()
        threshold = 0.3
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i >= j:
                    continue
                corr = abs(corr_matrix.loc[var1, var2])
                if corr > threshold:
                    if (var1, var2) not in existing_edges and (var2, var1) not in existing_edges:
                        node1 = graph.nodes[var1]
                        node2 = graph.nodes[var2]
                        if node1.node_type == NodeType.INPUT:
                            source, target = var1, var2
                        elif node2.node_type == NodeType.INPUT:
                            source, target = var2, var1
                        elif node1.node_type == NodeType.OUTPUT:
                            source, target = var2, var1
                        else:
                            source, target = var1, var2
                        new_edges.append(CausalEdge(source=source, target=target,
                            edge_type=EdgeType.LEARNED, weight=corr,
                            confidence=min(corr, 0.8),
                            mechanism=f"Learned from correlation (r={corr:.3f})"))
        return new_edges

    def _learn_pc_algorithm(self, data: pd.DataFrame, graph: CausalGraph,
                             existing_edges: Set[Tuple[str, str]]) -> List[CausalEdge]:
        new_edges = []
        variables = list(graph.nodes.keys())
        n = len(data)
        alpha = 0.05
        skeleton = {(v1, v2) for i, v1 in enumerate(variables) for v2 in variables[i+1:]}
        
        for v1, v2 in list(skeleton):
            try:
                from scipy import stats
                corr = data[[v1, v2]].corr().iloc[0, 1]
                z = 0.5 * np.log((1 + corr) / (1 - corr + 1e-10))
                se = 1 / np.sqrt(n - 3)
                p_value = 2 * (1 - stats.norm.cdf(abs(z) / se))
                if p_value > alpha:
                    skeleton.discard((v1, v2))
            except Exception as e:
                logger.warning(f"PC test failed for {v1}-{v2}: {e}")
        
        for v1, v2 in skeleton:
            if (v1, v2) not in existing_edges and (v2, v1) not in existing_edges:
                node1, node2 = graph.nodes[v1], graph.nodes[v2]
                if node1.node_type == NodeType.INPUT:
                    source, target = v1, v2
                elif node2.node_type == NodeType.INPUT:
                    source, target = v2, v1
                else:
                    source, target = v1, v2
                corr = abs(data[[source, target]].corr().iloc[0, 1])
                new_edges.append(CausalEdge(source=source, target=target,
                    edge_type=EdgeType.LEARNED, weight=corr, confidence=0.7,
                    mechanism="Learned via PC algorithm"))
        return new_edges

    def _learn_ges_algorithm(self, data: pd.DataFrame, graph: CausalGraph,
                              existing_edges: Set[Tuple[str, str]]) -> List[CausalEdge]:
        new_edges = []
        variables = list(graph.nodes.keys())
        
        for i, v1 in enumerate(variables):
            for v2 in variables[i+1:]:
                if (v1, v2) in existing_edges or (v2, v1) in existing_edges:
                    continue
                try:
                    from sklearn.linear_model import LinearRegression
                    X = data[[v1]].values
                    y = data[v2].values
                    model = LinearRegression()
                    model.fit(X, y)
                    r_squared = model.score(X, y)
                    if r_squared > 0.1:
                        node1, node2 = graph.nodes[v1], graph.nodes[v2]
                        if node1.node_type == NodeType.INPUT:
                            source, target = v1, v2
                        elif node2.node_type == NodeType.INPUT:
                            source, target = v2, v1
                        else:
                            source, target = v1, v2
                        new_edges.append(CausalEdge(source=source, target=target,
                            edge_type=EdgeType.LEARNED, weight=np.sqrt(r_squared),
                            confidence=0.65, mechanism=f"Learned via GES (R2={r_squared:.3f})"))
                except Exception as e:
                    logger.warning(f"GES edge learning failed for {v1}-{v2}: {e}")
        return new_edges

    def validate_graph(self, graph: CausalGraph) -> GraphValidationResult:
        start_time = datetime.now()
        logger.info("Validating causal graph")
        errors: List[str] = []
        warnings: List[str] = []
        G = graph.to_networkx()
        
        is_acyclic = nx.is_directed_acyclic_graph(G)
        if not is_acyclic:
            cycles = list(nx.simple_cycles(G))
            errors.append(f"Graph contains {len(cycles)} cycle(s): {cycles[:3]}")
        
        is_connected = nx.is_weakly_connected(G) if len(G) > 0 else True
        if not is_connected:
            components = list(nx.weakly_connected_components(G))
            warnings.append(f"Graph has {len(components)} disconnected components")
        
        isolated = list(nx.isolates(G))
        if isolated:
            warnings.append(f"Graph has {len(isolated)} isolated node(s): {isolated}")
        
        for edge in graph.edges:
            if edge.source not in graph.nodes:
                errors.append(f"Edge source {edge.source} not in nodes")
            if edge.target not in graph.nodes:
                errors.append(f"Edge target {edge.target} not in nodes")
            if edge.source == edge.target:
                errors.append(f"Self-loop detected: {edge.source}")
        
        input_nodes = [n for n, node in graph.nodes.items() if node.node_type == NodeType.INPUT]
        output_nodes = [n for n, node in graph.nodes.items() if node.node_type == NodeType.OUTPUT]
        if not input_nodes:
            warnings.append("No INPUT nodes defined")
        if not output_nodes:
            warnings.append("No OUTPUT nodes defined")
        
        statistics = {
            "node_count": G.number_of_nodes(), "edge_count": G.number_of_edges(),
            "density": nx.density(G),
            "avg_in_degree": sum(d for n, d in G.in_degree()) / max(G.number_of_nodes(), 1),
            "avg_out_degree": sum(d for n, d in G.out_degree()) / max(G.number_of_nodes(), 1),
            "max_path_length": nx.dag_longest_path_length(G) if is_acyclic and len(G) > 0 else None,
            "input_nodes": input_nodes, "output_nodes": output_nodes}
        
        is_valid = len(errors) == 0 and is_acyclic
        validation_data = f"{is_valid}{errors}{warnings}"
        provenance_hash = hashlib.sha256(validation_data.encode()).hexdigest()
        
        result = GraphValidationResult(is_valid=is_valid, is_acyclic=is_acyclic,
            is_connected=is_connected, errors=errors, warnings=warnings,
            statistics=statistics, provenance_hash=provenance_hash)
        
        logger.info(f"Graph validation complete: valid={is_valid}")
        return result

    def visualize_graph(self, graph: CausalGraph, figsize: Tuple[int, int] = (12, 8),
                        show_weights: bool = True, highlight_path: Optional[List[str]] = None):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed, cannot visualize")
            return None
        
        G = graph.to_networkx()
        fig, ax = plt.subplots(figsize=figsize)
        
        color_map = {NodeType.INPUT: "#90EE90", NodeType.INTERMEDIATE: "#87CEEB",
                     NodeType.OUTPUT: "#FFB6C1", NodeType.LATENT: "#DDA0DD"}
        
        node_colors = []
        for node in G.nodes():
            if node in graph.nodes:
                node_type = graph.nodes[node].node_type
                node_colors.append(color_map.get(node_type, "#FFFFFF"))
            else:
                node_colors.append("#FFFFFF")
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=2000, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")
        
        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            if highlight_path and u in highlight_path and v in highlight_path:
                idx_u, idx_v = highlight_path.index(u), highlight_path.index(v)
                if abs(idx_u - idx_v) == 1:
                    edge_colors.append("red")
                    edge_widths.append(3)
                    continue
            edge_type = data.get("edge_type", EdgeType.PHYSICS_BASED)
            if edge_type == EdgeType.PHYSICS_BASED:
                edge_colors.append("#2E8B57")
            elif edge_type == EdgeType.LEARNED:
                edge_colors.append("#4169E1")
            else:
                edge_colors.append("#696969")
            edge_widths.append(data.get("weight", 0.5) * 2 + 0.5)
        
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths,
            alpha=0.7, arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
        
        if show_weights:
            edge_labels = {(u, v): f"{data.get('weight', 1.0):.2f}" for u, v, data in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)
        
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#90EE90", markersize=10, label="Input"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#87CEEB", markersize=10, label="Intermediate"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFB6C1", markersize=10, label="Output"),
            plt.Line2D([0], [0], color="#2E8B57", linewidth=2, label="Physics-based"),
            plt.Line2D([0], [0], color="#4169E1", linewidth=2, label="Learned")]
        ax.legend(handles=legend_elements, loc="upper left")
        ax.set_title(f"Combustion Causal Graph: {graph.name}")
        ax.axis("off")
        plt.tight_layout()
        return fig
