# -*- coding: utf-8 -*-
"""
Graph Visualizer - Export dependency graph to DOT and Mermaid formats.

Renders the dependency graph with styling based on agent type and edge
type. Supports node coloring, edge styling, cluster grouping, and
critical path highlighting.

Example:
    >>> visualizer = GraphVisualizer()
    >>> dot_str = visualizer.to_dot(graph)
    >>> mermaid_str = visualizer.to_mermaid(graph)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from greenlang.infrastructure.agent_factory.dependencies.graph import (
    DependencyEdge,
    DependencyGraph,
    EdgeType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Agent type to color mapping for DOT/Mermaid
AGENT_TYPE_COLORS: Dict[str, str] = {
    "deterministic": "#4A90D9",  # Blue
    "reasoning": "#50C878",      # Green
    "insight": "#FF8C42",        # Orange
}

AGENT_TYPE_DOT_COLORS: Dict[str, str] = {
    "deterministic": "lightblue",
    "reasoning": "lightgreen",
    "insight": "lightsalmon",
}

# Edge type to DOT style mapping
EDGE_STYLES: Dict[str, str] = {
    EdgeType.RUNTIME.value: "solid",
    EdgeType.BUILD.value: "bold",
    EdgeType.TEST.value: "dotted",
    EdgeType.OPTIONAL.value: "dashed",
}

EDGE_MERMAID_STYLES: Dict[str, str] = {
    EdgeType.RUNTIME.value: "-->",
    EdgeType.BUILD.value: "==>",
    EdgeType.TEST.value: "-.->",
    EdgeType.OPTIONAL.value: "-..->",
}


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------


class GraphVisualizer:
    """Export dependency graphs to visual formats.

    Supports:
      - DOT (Graphviz) format with typed colors and edge styles
      - Mermaid diagram format for Markdown rendering
      - Critical path highlighting
      - Cluster grouping by agent category
    """

    def to_dot(
        self,
        graph: DependencyGraph,
        title: str = "Agent Dependency Graph",
        highlight_critical_path: bool = False,
    ) -> str:
        """Export the graph to DOT (Graphviz) format.

        Args:
            graph: The dependency graph to visualize.
            title: Graph title.
            highlight_critical_path: If True, bold the longest dependency chain.

        Returns:
            DOT format string.
        """
        lines: List[str] = []
        lines.append(f'digraph "{title}" {{')
        lines.append('    rankdir=TB;')
        lines.append('    node [shape=box, style=filled, fontname="Helvetica"];')
        lines.append('    edge [fontname="Helvetica", fontsize=10];')
        lines.append("")

        # Critical path detection
        critical_edges: Set[tuple[str, str]] = set()
        if highlight_critical_path:
            critical_edges = self._find_critical_path_edges(graph)

        # Cluster nodes by agent type
        clusters = self._cluster_by_type(graph)
        cluster_idx = 0

        for agent_type, agents in clusters.items():
            color = AGENT_TYPE_DOT_COLORS.get(agent_type, "white")
            lines.append(f"    subgraph cluster_{cluster_idx} {{")
            lines.append(f'        label="{agent_type}";')
            lines.append(f"        style=dashed;")
            lines.append(f'        color="{color}";')

            for agent_key in sorted(agents):
                meta = graph.get_metadata(agent_key)
                version = meta.get("version", "")
                label = f"{agent_key}\\n{version}" if version else agent_key
                lines.append(
                    f'        "{agent_key}" [label="{label}", fillcolor="{color}"];'
                )

            lines.append("    }")
            cluster_idx += 1

        # Handle unclustered nodes
        unclustered = set(graph.all_agents())
        for agents in clusters.values():
            unclustered -= agents
        for agent_key in sorted(unclustered):
            meta = graph.get_metadata(agent_key)
            version = meta.get("version", "")
            label = f"{agent_key}\\n{version}" if version else agent_key
            lines.append(f'    "{agent_key}" [label="{label}"];')

        lines.append("")

        # Edges
        for agent_key in graph.all_agents():
            for edge in graph.get_dependency_edges(agent_key):
                style = EDGE_STYLES.get(edge.edge_type.value, "solid")
                attrs = [f'style="{style}"']

                if edge.version_constraint and edge.version_constraint != "*":
                    attrs.append(f'label="{edge.version_constraint}"')

                if (edge.from_key, edge.to_key) in critical_edges:
                    attrs.append('color="red"')
                    attrs.append('penwidth=2.0')

                attr_str = ", ".join(attrs)
                lines.append(
                    f'    "{edge.from_key}" -> "{edge.to_key}" [{attr_str}];'
                )

        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(
        self,
        graph: DependencyGraph,
        title: str = "Agent Dependency Graph",
        highlight_critical_path: bool = False,
    ) -> str:
        """Export the graph to Mermaid diagram format.

        Args:
            graph: The dependency graph to visualize.
            title: Diagram title.
            highlight_critical_path: If True, highlight the longest chain.

        Returns:
            Mermaid diagram format string.
        """
        lines: List[str] = []
        lines.append(f"---")
        lines.append(f"title: {title}")
        lines.append(f"---")
        lines.append("graph TD")

        critical_edges: Set[tuple[str, str]] = set()
        if highlight_critical_path:
            critical_edges = self._find_critical_path_edges(graph)

        # Node definitions with styling
        for agent_key in sorted(graph.all_agents()):
            meta = graph.get_metadata(agent_key)
            version = meta.get("version", "")
            agent_type = meta.get("type", "deterministic")
            safe_key = self._mermaid_safe_id(agent_key)

            label = f"{agent_key} v{version}" if version else agent_key

            if agent_type == "deterministic":
                lines.append(f"    {safe_key}[{label}]")
            elif agent_type == "reasoning":
                lines.append(f"    {safe_key}({label})")
            elif agent_type == "insight":
                lines.append(f"    {safe_key}{{{{{label}}}}}")
            else:
                lines.append(f"    {safe_key}[{label}]")

        lines.append("")

        # Edges
        for agent_key in sorted(graph.all_agents()):
            for edge in graph.get_dependency_edges(agent_key):
                from_id = self._mermaid_safe_id(edge.from_key)
                to_id = self._mermaid_safe_id(edge.to_key)
                arrow = EDGE_MERMAID_STYLES.get(
                    edge.edge_type.value, "-->"
                )

                constraint = ""
                if edge.version_constraint and edge.version_constraint != "*":
                    constraint = f"|{edge.version_constraint}|"

                if (edge.from_key, edge.to_key) in critical_edges:
                    lines.append(f"    {from_id} =={constraint}==> {to_id}")
                else:
                    lines.append(f"    {from_id} {arrow}{constraint} {to_id}")

        # Add styling
        lines.append("")
        clusters = self._cluster_by_type(graph)
        for agent_type, agents in clusters.items():
            color = AGENT_TYPE_COLORS.get(agent_type, "#FFFFFF")
            for agent_key in sorted(agents):
                safe_key = self._mermaid_safe_id(agent_key)
                lines.append(f"    style {safe_key} fill:{color},stroke:#333,stroke-width:2px")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cluster_by_type(
        self, graph: DependencyGraph
    ) -> Dict[str, Set[str]]:
        """Group agents by their type metadata."""
        clusters: Dict[str, Set[str]] = {}
        for agent_key in graph.all_agents():
            meta = graph.get_metadata(agent_key)
            agent_type = meta.get("type", "unknown")
            clusters.setdefault(agent_type, set()).add(agent_key)
        return clusters

    def _find_critical_path_edges(
        self, graph: DependencyGraph
    ) -> Set[tuple[str, str]]:
        """Find the longest dependency chain (critical path) in the graph.

        Returns:
            Set of (from_key, to_key) tuples on the critical path.
        """
        all_agents = graph.all_agents()
        if not all_agents:
            return set()

        # Compute longest path using dynamic programming
        longest_dist: Dict[str, int] = {key: 0 for key in all_agents}
        predecessor: Dict[str, Optional[str]] = {key: None for key in all_agents}

        # Process in topological order (BFS)
        from collections import deque

        in_degree: Dict[str, int] = {key: 0 for key in all_agents}
        for key in all_agents:
            for dep in graph.get_dependencies(key):
                if dep in in_degree:
                    in_degree[key] += 1

        queue: deque[str] = deque()
        for key in all_agents:
            if in_degree[key] == 0:
                queue.append(key)

        topo_order: List[str] = []
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for dependent in graph.get_dependents(node):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Compute longest distances
        for node in topo_order:
            for dep in graph.get_dependencies(node):
                if dep in longest_dist:
                    new_dist = longest_dist[dep] + 1
                    if new_dist > longest_dist[node]:
                        longest_dist[node] = new_dist
                        predecessor[node] = dep

        # Find the node with the longest distance
        if not longest_dist:
            return set()

        end_node = max(longest_dist, key=lambda k: longest_dist[k])

        # Trace back the critical path
        critical_edges: Set[tuple[str, str]] = set()
        current: Optional[str] = end_node
        while current and predecessor.get(current):
            prev = predecessor[current]
            if prev:
                critical_edges.add((current, prev))
            current = prev

        return critical_edges

    @staticmethod
    def _mermaid_safe_id(key: str) -> str:
        """Convert an agent key to a Mermaid-safe node ID."""
        return key.replace("-", "_").replace(".", "_")
