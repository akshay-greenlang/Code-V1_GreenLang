# -*- coding: utf-8 -*-
"""
DAG Builder - AGENT-FOUND-001: GreenLang DAG Orchestrator

Fluent builder API for programmatic DAG construction, plus YAML/JSON
loading and export.

Example:
    >>> dag = (
    ...     DAGBuilder("emissions-calc", "Scope 1+2+3 calculation")
    ...     .add_node("intake", "intake_agent")
    ...     .add_node("validate", "validation_agent", depends_on=["intake"])
    ...     .add_node("scope1", "scope1_agent", depends_on=["validate"])
    ...     .add_node("scope2", "scope2_agent", depends_on=["validate"])
    ...     .add_node("aggregate", "agg_agent", depends_on=["scope1", "scope2"])
    ...     .with_max_parallel(5)
    ...     .build()
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from greenlang.orchestrator.dag_validator import validate_dag
from greenlang.orchestrator.models import (
    DAGNode,
    DAGOnFailure,
    DAGWorkflow,
    OnFailure,
    RetryPolicy,
    TimeoutPolicy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional YAML import
# ---------------------------------------------------------------------------

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional clock import
# ---------------------------------------------------------------------------

try:
    from greenlang.utilities.determinism.clock import DeterministicClock
    _CLOCK_AVAILABLE = True
except ImportError:
    DeterministicClock = None  # type: ignore[assignment, misc]
    _CLOCK_AVAILABLE = False


def _now():
    """Get current timestamp."""
    if _CLOCK_AVAILABLE and DeterministicClock is not None:
        return DeterministicClock.now()
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


# ===================================================================
# DAGBuilder
# ===================================================================


class DAGBuilder:
    """Fluent builder for constructing DAGWorkflow instances.

    Provides a step-by-step API for adding nodes, edges, and policies,
    then validates and builds the final DAGWorkflow.

    Example:
        >>> dag = (
        ...     DAGBuilder("my-dag", "My workflow")
        ...     .add_node("a", "agent_a")
        ...     .add_node("b", "agent_b", depends_on=["a"])
        ...     .build()
        ... )
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        dag_id: Optional[str] = None,
    ) -> None:
        """Initialize DAGBuilder.

        Args:
            name: Human-readable workflow name.
            description: Workflow description.
            dag_id: Optional explicit DAG ID (auto-generated if None).
        """
        self._name = name
        self._description = description
        self._dag_id = dag_id or ""
        self._nodes: Dict[str, DAGNode] = {}
        self._default_retry: Optional[RetryPolicy] = None
        self._default_timeout: Optional[TimeoutPolicy] = None
        self._on_failure: DAGOnFailure = DAGOnFailure.FAIL_FAST
        self._max_parallel: int = 10
        self._metadata: Dict[str, Any] = {}
        self._version: str = "1.0.0"

    def add_node(
        self,
        node_id: str,
        agent_id: str,
        depends_on: Optional[List[str]] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_key: str = "",
        condition: Optional[str] = None,
        retry_policy: Optional[RetryPolicy] = None,
        timeout_policy: Optional[TimeoutPolicy] = None,
        on_failure: OnFailure = OnFailure.STOP,
        compensation_handler: Optional[str] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DAGBuilder:
        """Add a node to the DAG.

        Args:
            node_id: Unique node identifier.
            agent_id: Agent to execute for this node.
            depends_on: List of predecessor node IDs.
            input_mapping: Input mapping from context to node inputs.
            output_key: Key for storing output in context.
            condition: Conditional execution expression.
            retry_policy: Node-level retry override.
            timeout_policy: Node-level timeout override.
            on_failure: Failure strategy for this node.
            compensation_handler: Agent ID for compensation.
            priority: Priority for tie-breaking.
            metadata: Arbitrary metadata.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If node_id already exists.
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists in DAG")

        self._nodes[node_id] = DAGNode(
            node_id=node_id,
            agent_id=agent_id,
            depends_on=list(depends_on or []),
            input_mapping=dict(input_mapping or {}),
            output_key=output_key or node_id,
            condition=condition,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            on_failure=on_failure,
            compensation_handler=compensation_handler,
            priority=priority,
            metadata=dict(metadata or {}),
        )
        return self

    def add_edge(self, from_node: str, to_node: str) -> DAGBuilder:
        """Add a dependency edge from one node to another.

        This adds ``from_node`` to the ``to_node``'s depends_on list.

        Args:
            from_node: Predecessor node ID.
            to_node: Successor node ID.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If either node does not exist.
        """
        if from_node not in self._nodes:
            raise ValueError(f"Source node '{from_node}' not found")
        if to_node not in self._nodes:
            raise ValueError(f"Target node '{to_node}' not found")

        if from_node not in self._nodes[to_node].depends_on:
            self._nodes[to_node].depends_on.append(from_node)
        return self

    def with_default_retry(self, policy: RetryPolicy) -> DAGBuilder:
        """Set the DAG-level default retry policy.

        Args:
            policy: Default RetryPolicy.

        Returns:
            Self for chaining.
        """
        self._default_retry = policy
        return self

    def with_default_timeout(self, policy: TimeoutPolicy) -> DAGBuilder:
        """Set the DAG-level default timeout policy.

        Args:
            policy: Default TimeoutPolicy.

        Returns:
            Self for chaining.
        """
        self._default_timeout = policy
        return self

    def with_max_parallel(self, n: int) -> DAGBuilder:
        """Set maximum parallel nodes per level.

        Args:
            n: Maximum parallel node count.

        Returns:
            Self for chaining.
        """
        self._max_parallel = max(1, n)
        return self

    def with_on_failure(self, strategy: DAGOnFailure) -> DAGBuilder:
        """Set DAG-level failure strategy.

        Args:
            strategy: DAG failure strategy.

        Returns:
            Self for chaining.
        """
        self._on_failure = strategy
        return self

    def with_metadata(self, meta: Dict[str, Any]) -> DAGBuilder:
        """Set DAG metadata.

        Args:
            meta: Metadata dictionary.

        Returns:
            Self for chaining.
        """
        self._metadata.update(meta)
        return self

    def with_version(self, version: str) -> DAGBuilder:
        """Set DAG version string.

        Args:
            version: Semantic version string.

        Returns:
            Self for chaining.
        """
        self._version = version
        return self

    def build(self, validate: bool = True) -> DAGWorkflow:
        """Build and validate the DAGWorkflow.

        Args:
            validate: Whether to validate the DAG before returning.

        Returns:
            Validated DAGWorkflow instance.

        Raises:
            ValueError: If validation fails.
        """
        dag = DAGWorkflow(
            dag_id=self._dag_id or self._generate_id(),
            name=self._name,
            description=self._description,
            version=self._version,
            nodes=dict(self._nodes),
            default_retry_policy=self._default_retry or RetryPolicy(),
            default_timeout_policy=self._default_timeout or TimeoutPolicy(),
            on_failure=self._on_failure,
            max_parallel_nodes=self._max_parallel,
            metadata=dict(self._metadata),
            created_at=_now(),
        )
        dag.hash = dag.calculate_hash()

        if validate:
            errors = validate_dag(dag)
            if errors:
                error_msgs = [e.message for e in errors]
                raise ValueError(
                    f"DAG validation failed with {len(errors)} error(s): "
                    + "; ".join(error_msgs)
                )

        logger.info(
            "Built DAGWorkflow: id=%s name='%s' nodes=%d",
            dag.dag_id, dag.name, len(dag.nodes),
        )
        return dag

    def _generate_id(self) -> str:
        """Generate a DAG ID based on name and node structure."""
        content = f"{self._name}:{sorted(self._nodes.keys())}"
        hash_hex = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"dag_{hash_hex}"

    # ------------------------------------------------------------------
    # Class-level loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> DAGWorkflow:
        """Load a DAGWorkflow from a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            DAGWorkflow instance.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        return DAGWorkflow.from_yaml(path)

    @classmethod
    def from_json(cls, path: str) -> DAGWorkflow:
        """Load a DAGWorkflow from a JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            DAGWorkflow instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DAGWorkflow.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DAGWorkflow:
        """Load a DAGWorkflow from a dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            DAGWorkflow instance.
        """
        return DAGWorkflow.from_dict(data)

    # ------------------------------------------------------------------
    # Instance-level exporters
    # ------------------------------------------------------------------

    def to_yaml(self, path: str) -> None:
        """Build and export to YAML file.

        Args:
            path: Output YAML file path.
        """
        dag = self.build(validate=False)
        dag.to_yaml(path)

    def to_json(self, path: str) -> None:
        """Build and export to JSON file.

        Args:
            path: Output JSON file path.
        """
        dag = self.build(validate=False)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dag.to_dict(), f, indent=2, default=str)
        logger.info("DAG exported to JSON: %s", path)


__all__ = [
    "DAGBuilder",
]
