"""
Agent Factory Dependencies - INFRA-010 Phase 3

Directed acyclic graph (DAG) management for GreenLang agent dependencies.
Provides dependency graph construction, topological sorting with parallel
group detection, cycle detection, and graph visualization in DOT/Mermaid.

Public API:
    - DependencyGraph: DAG with agent nodes and dependency edges.
    - TopologicalResolver: Topological sort and parallel execution planning.
    - CycleDetector: Circular dependency detection with diagnostic paths.
    - GraphVisualizer: Export to DOT (Graphviz) and Mermaid diagram formats.

Example:
    >>> from greenlang.infrastructure.agent_factory.dependencies import (
    ...     DependencyGraph, TopologicalResolver, CycleDetector,
    ... )
    >>> graph = DependencyGraph()
    >>> graph.add_agent("intake", {"version": "1.0.0", "type": "deterministic"})
    >>> graph.add_agent("calc", {"version": "1.0.0", "type": "deterministic"})
    >>> graph.add_dependency("calc", "intake", "^1.0.0")
    >>> plan = TopologicalResolver().resolve(graph)
"""

from __future__ import annotations

from greenlang.infrastructure.agent_factory.dependencies.graph import (
    DependencyEdge,
    DependencyGraph,
    EdgeType,
)
from greenlang.infrastructure.agent_factory.dependencies.resolver import (
    ExecutionGroup,
    ExecutionPlan,
    TopologicalResolver,
)
from greenlang.infrastructure.agent_factory.dependencies.cycle_detector import (
    CycleDetectionResult,
    CycleDetector,
)
from greenlang.infrastructure.agent_factory.dependencies.visualizer import (
    GraphVisualizer,
)

__all__ = [
    "CycleDetectionResult",
    "CycleDetector",
    "DependencyEdge",
    "DependencyGraph",
    "EdgeType",
    "ExecutionGroup",
    "ExecutionPlan",
    "GraphVisualizer",
    "TopologicalResolver",
]
