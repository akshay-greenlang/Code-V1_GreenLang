# -*- coding: utf-8 -*-
"""
DAG Validator - AGENT-FOUND-001: GreenLang DAG Orchestrator

Structural validation for DAG workflows including:
- Cycle detection using DFS-based algorithm
- Unreachable node detection (BFS from roots)
- Missing dependency validation
- Duplicate node ID detection
- Self-dependency detection

All validators return a list of ValidationError dataclasses. An empty
list indicates a valid DAG.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set

from greenlang.orchestrator.models import DAGWorkflow

logger = logging.getLogger(__name__)


# ===================================================================
# Validation error model
# ===================================================================


@dataclass
class ValidationError:
    """Describes a single validation issue found in a DAG.

    Attributes:
        error_type: Category of the error (cycle, unreachable, missing,
            duplicate, self_dep).
        message: Human-readable description of the error.
        nodes: Node IDs involved in the error.
    """
    error_type: str = ""
    message: str = ""
    nodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Serialize to dictionary."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "nodes": list(self.nodes),
        }


# ===================================================================
# Individual validators
# ===================================================================


def detect_cycles(dag: DAGWorkflow) -> List[List[str]]:
    """Detect cycles in a DAG using DFS-based algorithm.

    Args:
        dag: DAGWorkflow to validate.

    Returns:
        List of cycle paths. Each path is a list of node IDs forming
        a cycle. An empty list means no cycles were detected.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {nid: WHITE for nid in dag.nodes}
    parent: Dict[str, str] = {}
    cycles: List[List[str]] = []

    # Build adjacency list (node -> successors)
    adjacency: Dict[str, List[str]] = {nid: [] for nid in dag.nodes}
    for nid, node in dag.nodes.items():
        for dep in node.depends_on:
            if dep in adjacency:
                adjacency[dep].append(nid)

    def _dfs(node_id: str) -> None:
        color[node_id] = GRAY
        for successor in sorted(adjacency.get(node_id, [])):
            if successor not in color:
                continue
            if color[successor] == GRAY:
                # Back edge found - reconstruct cycle
                cycle = [successor]
                current = node_id
                while current != successor:
                    cycle.append(current)
                    current = parent.get(current, successor)
                cycle.append(successor)
                cycle.reverse()
                cycles.append(cycle)
            elif color[successor] == WHITE:
                parent[successor] = node_id
                _dfs(successor)
        color[node_id] = BLACK

    for nid in sorted(dag.nodes.keys()):
        if color.get(nid) == WHITE:
            _dfs(nid)

    return cycles


def find_unreachable_nodes(dag: DAGWorkflow) -> List[str]:
    """Find nodes unreachable from any root node using BFS.

    Root nodes are those with no dependencies (in-degree = 0).

    Args:
        dag: DAGWorkflow to validate.

    Returns:
        List of unreachable node IDs.
    """
    if not dag.nodes:
        return []

    # Find root nodes (nodes with no depends_on)
    roots: List[str] = []
    for nid, node in dag.nodes.items():
        effective_deps = [d for d in node.depends_on if d in dag.nodes]
        if not effective_deps:
            roots.append(nid)

    if not roots:
        # If no roots, all nodes are potentially in cycles
        return sorted(dag.nodes.keys())

    # BFS from all roots
    visited: Set[str] = set()
    queue: deque[str] = deque(roots)

    # Build adjacency list (node -> successors)
    adjacency: Dict[str, List[str]] = {nid: [] for nid in dag.nodes}
    for nid, node in dag.nodes.items():
        for dep in node.depends_on:
            if dep in adjacency:
                adjacency[dep].append(nid)

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for successor in adjacency.get(current, []):
            if successor not in visited:
                queue.append(successor)

    unreachable = sorted(set(dag.nodes.keys()) - visited)
    return unreachable


def check_missing_dependencies(dag: DAGWorkflow) -> List[ValidationError]:
    """Check for nodes referencing non-existent predecessors.

    Args:
        dag: DAGWorkflow to validate.

    Returns:
        List of ValidationErrors for missing dependencies.
    """
    errors: List[ValidationError] = []
    for nid, node in sorted(dag.nodes.items()):
        for dep in node.depends_on:
            if dep not in dag.nodes:
                errors.append(ValidationError(
                    error_type="missing",
                    message=(
                        f"Node '{nid}' depends on '{dep}' which does "
                        f"not exist in the DAG"
                    ),
                    nodes=[nid, dep],
                ))
    return errors


def check_duplicate_nodes(dag: DAGWorkflow) -> List[ValidationError]:
    """Check for duplicate node IDs.

    Since nodes are stored in a Dict, true duplicates are impossible
    at the data structure level. This validator checks that node_id
    fields match their dictionary keys.

    Args:
        dag: DAGWorkflow to validate.

    Returns:
        List of ValidationErrors for mismatched node IDs.
    """
    errors: List[ValidationError] = []
    for key, node in dag.nodes.items():
        if node.node_id and node.node_id != key:
            errors.append(ValidationError(
                error_type="duplicate",
                message=(
                    f"Node key '{key}' does not match node_id "
                    f"'{node.node_id}'"
                ),
                nodes=[key, node.node_id],
            ))
    return errors


def check_self_dependencies(dag: DAGWorkflow) -> List[ValidationError]:
    """Check for nodes that depend on themselves.

    Args:
        dag: DAGWorkflow to validate.

    Returns:
        List of ValidationErrors for self-dependencies.
    """
    errors: List[ValidationError] = []
    for nid, node in sorted(dag.nodes.items()):
        if nid in node.depends_on:
            errors.append(ValidationError(
                error_type="self_dep",
                message=f"Node '{nid}' depends on itself",
                nodes=[nid],
            ))
    return errors


# ===================================================================
# Aggregate validator
# ===================================================================


def validate_dag(dag: DAGWorkflow) -> List[ValidationError]:
    """Run all DAG validation checks and return combined error list.

    An empty returned list indicates a valid DAG.

    Args:
        dag: DAGWorkflow to validate.

    Returns:
        Combined list of all validation errors found.
    """
    errors: List[ValidationError] = []

    # 1. Self-dependencies
    errors.extend(check_self_dependencies(dag))

    # 2. Duplicate/mismatched node IDs
    errors.extend(check_duplicate_nodes(dag))

    # 3. Missing dependencies
    errors.extend(check_missing_dependencies(dag))

    # 4. Cycle detection
    cycles = detect_cycles(dag)
    for cycle in cycles:
        errors.append(ValidationError(
            error_type="cycle",
            message=f"Cycle detected: {' -> '.join(cycle)}",
            nodes=cycle,
        ))

    # 5. Unreachable nodes
    unreachable = find_unreachable_nodes(dag)
    for nid in unreachable:
        errors.append(ValidationError(
            error_type="unreachable",
            message=f"Node '{nid}' is unreachable from any root node",
            nodes=[nid],
        ))

    if errors:
        logger.warning(
            "DAG validation found %d error(s) for dag_id=%s",
            len(errors), dag.dag_id,
        )
    else:
        logger.debug("DAG validation passed for dag_id=%s", dag.dag_id)

    return errors


__all__ = [
    "ValidationError",
    "validate_dag",
    "detect_cycles",
    "find_unreachable_nodes",
    "check_missing_dependencies",
    "check_duplicate_nodes",
    "check_self_dependencies",
]
