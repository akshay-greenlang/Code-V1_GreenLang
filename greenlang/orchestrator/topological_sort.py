# -*- coding: utf-8 -*-
"""
Topological Sort - AGENT-FOUND-001: GreenLang DAG Orchestrator

Kahn's algorithm implementation with deterministic tie-breaking for
DAG scheduling. Also provides level-based grouping for parallel
execution.

Key Features:
- Kahn's algorithm with sorted tie-breaking (alphabetical by node_id)
- Level grouping: nodes at same depth execute in parallel
- Root and sink node discovery
- Deterministic ordering for replay guarantees

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Set

from greenlang.orchestrator.models import DAGWorkflow

logger = logging.getLogger(__name__)


def get_roots(dag: DAGWorkflow) -> List[str]:
    """Find root nodes (nodes with no predecessors).

    Root nodes have empty depends_on lists (considering only nodes
    that exist in the DAG).

    Args:
        dag: DAGWorkflow to analyze.

    Returns:
        Sorted list of root node IDs.
    """
    roots: List[str] = []
    for nid, node in dag.nodes.items():
        effective_deps = [d for d in node.depends_on if d in dag.nodes]
        if not effective_deps:
            roots.append(nid)
    return sorted(roots)


def get_sinks(dag: DAGWorkflow) -> List[str]:
    """Find sink nodes (nodes with no successors).

    Sink nodes are not listed in any other node's depends_on.

    Args:
        dag: DAGWorkflow to analyze.

    Returns:
        Sorted list of sink node IDs.
    """
    # Build set of all nodes that have at least one successor
    has_successor: Set[str] = set()
    for node in dag.nodes.values():
        for dep in node.depends_on:
            if dep in dag.nodes:
                has_successor.add(dep)

    sinks = sorted(nid for nid in dag.nodes if nid not in has_successor)
    return sinks


def topological_sort(dag: DAGWorkflow) -> List[str]:
    """Topological sort using Kahn's algorithm with deterministic tie-breaking.

    Tie-breaking is done alphabetically by node_id to ensure deterministic
    ordering across runs with identical DAG definitions.

    Args:
        dag: DAGWorkflow to sort.

    Returns:
        List of node IDs in topological order.

    Raises:
        ValueError: If the DAG contains cycles.
    """
    if not dag.nodes:
        return []

    # Calculate in-degree for each node
    in_degree: Dict[str, int] = {nid: 0 for nid in dag.nodes}
    for nid, node in dag.nodes.items():
        for dep in node.depends_on:
            if dep in dag.nodes:
                in_degree[nid] += 1

    # Initialize queue with all root nodes (in-degree 0), sorted
    queue: deque[str] = deque(
        sorted(nid for nid, deg in in_degree.items() if deg == 0)
    )

    result: List[str] = []

    # Build adjacency list (predecessor -> successors)
    adjacency: Dict[str, List[str]] = {nid: [] for nid in dag.nodes}
    for nid, node in dag.nodes.items():
        for dep in node.depends_on:
            if dep in adjacency:
                adjacency[dep].append(nid)

    while queue:
        node_id = queue.popleft()
        result.append(node_id)

        # Collect all newly ready nodes, then sort for determinism
        newly_ready: List[str] = []
        for successor in adjacency.get(node_id, []):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                newly_ready.append(successor)

        # Add sorted newly-ready nodes to maintain deterministic order
        for nid in sorted(newly_ready):
            queue.append(nid)

    if len(result) != len(dag.nodes):
        logger.error(
            "Topological sort incomplete: processed %d of %d nodes "
            "(DAG contains cycles)",
            len(result), len(dag.nodes),
        )
        raise ValueError(
            f"DAG contains cycles: could only sort {len(result)} of "
            f"{len(dag.nodes)} nodes"
        )

    logger.debug(
        "Topological sort for dag_id=%s: %s",
        dag.dag_id, " -> ".join(result),
    )
    return result


def level_grouping(dag: DAGWorkflow) -> List[List[str]]:
    """Group nodes into parallel execution levels.

    Each level contains nodes whose predecessors are all in earlier
    levels. Nodes within a level can execute concurrently. Nodes
    within each level are sorted alphabetically for determinism.

    Args:
        dag: DAGWorkflow to group.

    Returns:
        List of levels, each level is a sorted list of node IDs.

    Raises:
        ValueError: If the DAG contains cycles.
    """
    if not dag.nodes:
        return []

    # Calculate in-degree for each node
    in_degree: Dict[str, int] = {nid: 0 for nid in dag.nodes}
    for nid, node in dag.nodes.items():
        for dep in node.depends_on:
            if dep in dag.nodes:
                in_degree[nid] += 1

    # Build adjacency list (predecessor -> successors)
    adjacency: Dict[str, List[str]] = {nid: [] for nid in dag.nodes}
    for nid, node in dag.nodes.items():
        for dep in node.depends_on:
            if dep in adjacency:
                adjacency[dep].append(nid)

    # BFS level by level
    current_level = sorted(
        nid for nid, deg in in_degree.items() if deg == 0
    )

    levels: List[List[str]] = []
    processed_count = 0

    while current_level:
        levels.append(current_level)
        processed_count += len(current_level)

        next_level_set: set[str] = set()
        for node_id in current_level:
            for successor in adjacency.get(node_id, []):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    next_level_set.add(successor)

        current_level = sorted(next_level_set)

    if processed_count != len(dag.nodes):
        raise ValueError(
            f"DAG contains cycles: could only group {processed_count} of "
            f"{len(dag.nodes)} nodes into levels"
        )

    logger.debug(
        "Level grouping for dag_id=%s: %d levels, sizes=%s",
        dag.dag_id, len(levels), [len(lvl) for lvl in levels],
    )
    return levels


__all__ = [
    "topological_sort",
    "level_grouping",
    "get_roots",
    "get_sinks",
]
