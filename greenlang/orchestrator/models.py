# -*- coding: utf-8 -*-
"""
Orchestrator Data Models - AGENT-FOUND-001: GreenLang DAG Orchestrator

Core data models for the DAG execution engine including:
- Enums for execution status, node status, and failure strategies
- RetryPolicy and TimeoutPolicy dataclasses
- DAGNode and DAGWorkflow definitions
- NodeExecutionResult and NodeProvenance records
- ExecutionTrace for full audit trails
- DAGCheckpoint for resume capability
- Serialization (to_dict/from_dict, to_yaml/from_yaml)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional YAML support
# ---------------------------------------------------------------------------

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    YAML_AVAILABLE = False


# ===================================================================
# Enums
# ===================================================================


class ExecutionStatus(str, Enum):
    """Status of a complete DAG execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPENSATING = "compensating"


class NodeStatus(str, Enum):
    """Status of an individual DAG node execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPENSATING = "compensating"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class OnFailure(str, Enum):
    """Per-node failure strategy."""
    STOP = "stop"
    SKIP = "skip"
    COMPENSATE = "compensate"


class DAGOnFailure(str, Enum):
    """DAG-level failure strategy."""
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"
    COMPENSATE = "compensate"


class RetryStrategyType(str, Enum):
    """Backoff strategy for retries."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"


class OnTimeout(str, Enum):
    """Action to take when a node times out."""
    FAIL = "fail"
    SKIP = "skip"
    COMPENSATE = "compensate"


# ===================================================================
# Policy dataclasses
# ===================================================================


@dataclass
class RetryPolicy:
    """Per-node retry configuration.

    Attributes:
        max_retries: Maximum number of retry attempts.
        strategy: Backoff strategy type.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap in seconds.
        jitter: Whether to add random jitter.
        jitter_range: Jitter range as fraction of delay.
        retryable_exceptions: Exception class names that are retryable.
    """
    max_retries: int = 3
    strategy: RetryStrategyType = RetryStrategyType.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    jitter_range: float = 0.1
    retryable_exceptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = asdict(self)
        data["strategy"] = self.strategy.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RetryPolicy:
        """Deserialize from dictionary."""
        if not data:
            return cls()
        strategy = data.get("strategy", "exponential")
        if isinstance(strategy, str):
            strategy = RetryStrategyType(strategy)
        return cls(
            max_retries=data.get("max_retries", 3),
            strategy=strategy,
            base_delay=data.get("base_delay", 1.0),
            max_delay=data.get("max_delay", 60.0),
            jitter=data.get("jitter", True),
            jitter_range=data.get("jitter_range", 0.1),
            retryable_exceptions=data.get("retryable_exceptions", []),
        )


@dataclass
class TimeoutPolicy:
    """Per-node timeout configuration.

    Attributes:
        timeout_seconds: Timeout duration in seconds.
        on_timeout: Action to take when timeout occurs.
    """
    timeout_seconds: float = 60.0
    on_timeout: OnTimeout = OnTimeout.FAIL

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timeout_seconds": self.timeout_seconds,
            "on_timeout": self.on_timeout.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TimeoutPolicy:
        """Deserialize from dictionary."""
        if not data:
            return cls()
        on_timeout = data.get("on_timeout", "fail")
        if isinstance(on_timeout, str):
            on_timeout = OnTimeout(on_timeout)
        return cls(
            timeout_seconds=data.get("timeout_seconds", 60.0),
            on_timeout=on_timeout,
        )


# ===================================================================
# DAG node and workflow
# ===================================================================


@dataclass
class DAGNode:
    """A single node in a DAG workflow.

    Attributes:
        node_id: Unique identifier for this node.
        agent_id: Reference to the agent that executes this node.
        depends_on: List of predecessor node IDs.
        input_mapping: Mapping from context keys to node input keys.
        output_key: Key under which node output is stored in context.
        condition: Optional condition expression for conditional execution.
        retry_policy: Node-level retry policy override.
        timeout_policy: Node-level timeout policy override.
        on_failure: Failure strategy for this node.
        compensation_handler: Optional agent_id for compensation.
        priority: Priority for tie-breaking within parallel levels.
        metadata: Arbitrary metadata attached to this node.
    """
    node_id: str = ""
    agent_id: str = ""
    depends_on: List[str] = field(default_factory=list)
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_key: str = ""
    condition: Optional[str] = None
    retry_policy: Optional[RetryPolicy] = None
    timeout_policy: Optional[TimeoutPolicy] = None
    on_failure: OnFailure = OnFailure.STOP
    compensation_handler: Optional[str] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data: Dict[str, Any] = {
            "node_id": self.node_id,
            "agent_id": self.agent_id,
            "depends_on": list(self.depends_on),
            "input_mapping": dict(self.input_mapping),
            "output_key": self.output_key,
            "condition": self.condition,
            "retry_policy": self.retry_policy.to_dict() if self.retry_policy else None,
            "timeout_policy": (
                self.timeout_policy.to_dict() if self.timeout_policy else None
            ),
            "on_failure": self.on_failure.value,
            "compensation_handler": self.compensation_handler,
            "priority": self.priority,
            "metadata": dict(self.metadata),
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DAGNode:
        """Deserialize from dictionary."""
        retry = data.get("retry_policy")
        timeout = data.get("timeout_policy")
        on_fail = data.get("on_failure", "stop")
        if isinstance(on_fail, str):
            on_fail = OnFailure(on_fail)

        return cls(
            node_id=data.get("node_id", ""),
            agent_id=data.get("agent_id", ""),
            depends_on=data.get("depends_on", []),
            input_mapping=data.get("input_mapping", {}),
            output_key=data.get("output_key", ""),
            condition=data.get("condition"),
            retry_policy=RetryPolicy.from_dict(retry) if retry else None,
            timeout_policy=TimeoutPolicy.from_dict(timeout) if timeout else None,
            on_failure=on_fail,
            compensation_handler=data.get("compensation_handler"),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DAGWorkflow:
    """Complete DAG workflow definition.

    Attributes:
        dag_id: Unique identifier for this workflow.
        name: Human-readable name.
        description: Description of workflow purpose.
        version: Semantic version string.
        nodes: Dictionary mapping node_id to DAGNode.
        default_retry_policy: Default retry policy for all nodes.
        default_timeout_policy: Default timeout policy for all nodes.
        on_failure: DAG-level failure strategy.
        max_parallel_nodes: Max concurrent nodes per level.
        metadata: Arbitrary metadata.
        created_at: Creation timestamp.
        hash: SHA-256 hash of the workflow definition.
    """
    dag_id: str = ""
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    default_retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    default_timeout_policy: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    on_failure: DAGOnFailure = DAGOnFailure.FAIL_FAST
    max_parallel_nodes: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    hash: str = ""

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the workflow definition."""
        definition = {
            "dag_id": self.dag_id,
            "name": self.name,
            "version": self.version,
            "nodes": {
                nid: node.to_dict() for nid, node in sorted(self.nodes.items())
            },
            "on_failure": self.on_failure.value,
            "max_parallel_nodes": self.max_parallel_nodes,
        }
        content = json.dumps(definition, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "dag_id": self.dag_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "nodes": {
                nid: node.to_dict() for nid, node in self.nodes.items()
            },
            "default_retry_policy": self.default_retry_policy.to_dict(),
            "default_timeout_policy": self.default_timeout_policy.to_dict(),
            "on_failure": self.on_failure.value,
            "max_parallel_nodes": self.max_parallel_nodes,
            "metadata": dict(self.metadata),
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DAGWorkflow:
        """Deserialize from dictionary."""
        nodes_data = data.get("nodes", {})
        nodes: Dict[str, DAGNode] = {}
        for nid, ndata in nodes_data.items():
            if isinstance(ndata, dict):
                ndata.setdefault("node_id", nid)
                nodes[nid] = DAGNode.from_dict(ndata)
            else:
                nodes[nid] = ndata

        on_fail = data.get("on_failure", "fail_fast")
        if isinstance(on_fail, str):
            on_fail = DAGOnFailure(on_fail)

        default_retry = data.get("default_retry_policy")
        default_timeout = data.get("default_timeout_policy")

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        dag = cls(
            dag_id=data.get("dag_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            nodes=nodes,
            default_retry_policy=(
                RetryPolicy.from_dict(default_retry) if default_retry else RetryPolicy()
            ),
            default_timeout_policy=(
                TimeoutPolicy.from_dict(default_timeout)
                if default_timeout
                else TimeoutPolicy()
            ),
            on_failure=on_fail,
            max_parallel_nodes=data.get("max_parallel_nodes", 10),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            hash=data.get("hash", ""),
        )
        if not dag.hash:
            dag.hash = dag.calculate_hash()
        return dag

    def to_yaml(self, path: Optional[str] = None) -> str:
        """Serialize to YAML string, optionally writing to file.

        Args:
            path: Optional file path to write YAML.

        Returns:
            YAML string representation.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML serialization")
        data = self.to_dict()
        yaml_str: str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_str)
            logger.info("DAGWorkflow written to %s", path)
        return yaml_str

    @classmethod
    def from_yaml(cls, yaml_str_or_path: str) -> DAGWorkflow:
        """Deserialize from YAML string or file path.

        Args:
            yaml_str_or_path: YAML content string or file path.

        Returns:
            DAGWorkflow instance.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML deserialization")
        # Try reading as file first
        try:
            with open(yaml_str_or_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except (FileNotFoundError, OSError):
            data = yaml.safe_load(yaml_str_or_path)
        return cls.from_dict(data)


# ===================================================================
# Execution result models
# ===================================================================


@dataclass
class NodeExecutionResult:
    """Result of executing a single DAG node.

    Attributes:
        node_id: Node that was executed.
        status: Final status after execution.
        outputs: Output data from the node.
        output_hash: SHA-256 hash of outputs for provenance.
        duration_ms: Total execution duration in milliseconds.
        attempt_count: Number of attempts (1 = no retries).
        error: Error message if failed.
        started_at: Execution start timestamp.
        completed_at: Execution completion timestamp.
    """
    node_id: str = ""
    status: NodeStatus = NodeStatus.PENDING
    outputs: Dict[str, Any] = field(default_factory=dict)
    output_hash: str = ""
    duration_ms: float = 0.0
    attempt_count: int = 1
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "outputs": self.outputs,
            "output_hash": self.output_hash,
            "duration_ms": self.duration_ms,
            "attempt_count": self.attempt_count,
            "error": self.error,
            "started_at": (
                self.started_at.isoformat() if self.started_at else None
            ),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NodeExecutionResult:
        """Deserialize from dictionary."""
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = NodeStatus(status)
        started = data.get("started_at")
        if isinstance(started, str):
            started = datetime.fromisoformat(started)
        completed = data.get("completed_at")
        if isinstance(completed, str):
            completed = datetime.fromisoformat(completed)
        return cls(
            node_id=data.get("node_id", ""),
            status=status,
            outputs=data.get("outputs", {}),
            output_hash=data.get("output_hash", ""),
            duration_ms=data.get("duration_ms", 0.0),
            attempt_count=data.get("attempt_count", 1),
            error=data.get("error"),
            started_at=started,
            completed_at=completed,
        )


@dataclass
class NodeProvenance:
    """Provenance record for a single node execution.

    Attributes:
        node_id: Node that was executed.
        input_hash: SHA-256 hash of the node input.
        output_hash: SHA-256 hash of the node output.
        duration_ms: Execution duration in milliseconds.
        attempt_count: Number of attempts.
        parent_hashes: Chain hashes of predecessor node provenances.
        chain_hash: SHA-256 hash linking this provenance to its parents.
    """
    node_id: str = ""
    input_hash: str = ""
    output_hash: str = ""
    duration_ms: float = 0.0
    attempt_count: int = 1
    parent_hashes: List[str] = field(default_factory=list)
    chain_hash: str = ""

    def calculate_chain_hash(self) -> str:
        """Calculate the chain hash for this provenance record."""
        data = (
            f"{self.node_id}:{self.input_hash}:{self.output_hash}:"
            f"{self.duration_ms}:{self.attempt_count}:"
            f"{','.join(sorted(self.parent_hashes))}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "duration_ms": self.duration_ms,
            "attempt_count": self.attempt_count,
            "parent_hashes": list(self.parent_hashes),
            "chain_hash": self.chain_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NodeProvenance:
        """Deserialize from dictionary."""
        return cls(
            node_id=data.get("node_id", ""),
            input_hash=data.get("input_hash", ""),
            output_hash=data.get("output_hash", ""),
            duration_ms=data.get("duration_ms", 0.0),
            attempt_count=data.get("attempt_count", 1),
            parent_hashes=data.get("parent_hashes", []),
            chain_hash=data.get("chain_hash", ""),
        )


@dataclass
class ExecutionTrace:
    """Complete execution trace for a DAG run.

    Attributes:
        execution_id: Unique identifier for this execution.
        dag_id: DAG that was executed.
        status: Final execution status.
        node_traces: Per-node execution results.
        topology_levels: Level grouping used for execution order.
        start_time: Execution start timestamp.
        end_time: Execution end timestamp.
        provenance_chain_hash: SHA-256 hash of the full provenance chain.
        input_data: Original input data.
        errors: List of error messages encountered.
    """
    execution_id: str = ""
    dag_id: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING
    node_traces: Dict[str, NodeExecutionResult] = field(default_factory=dict)
    topology_levels: List[List[str]] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    provenance_chain_hash: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "execution_id": self.execution_id,
            "dag_id": self.dag_id,
            "status": self.status.value,
            "node_traces": {
                nid: trace.to_dict()
                for nid, trace in self.node_traces.items()
            },
            "topology_levels": self.topology_levels,
            "start_time": (
                self.start_time.isoformat() if self.start_time else None
            ),
            "end_time": (
                self.end_time.isoformat() if self.end_time else None
            ),
            "provenance_chain_hash": self.provenance_chain_hash,
            "input_data": self.input_data,
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionTrace:
        """Deserialize from dictionary."""
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = ExecutionStatus(status)

        node_traces: Dict[str, NodeExecutionResult] = {}
        for nid, tdata in data.get("node_traces", {}).items():
            if isinstance(tdata, dict):
                node_traces[nid] = NodeExecutionResult.from_dict(tdata)
            else:
                node_traces[nid] = tdata

        start = data.get("start_time")
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        end = data.get("end_time")
        if isinstance(end, str):
            end = datetime.fromisoformat(end)

        return cls(
            execution_id=data.get("execution_id", ""),
            dag_id=data.get("dag_id", ""),
            status=status,
            node_traces=node_traces,
            topology_levels=data.get("topology_levels", []),
            start_time=start,
            end_time=end,
            provenance_chain_hash=data.get("provenance_chain_hash", ""),
            input_data=data.get("input_data", {}),
            errors=data.get("errors", []),
        )

    def to_yaml(self, path: Optional[str] = None) -> str:
        """Serialize to YAML string."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML serialization")
        data = self.to_dict()
        yaml_str: str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str

    @classmethod
    def from_yaml(cls, yaml_str_or_path: str) -> ExecutionTrace:
        """Deserialize from YAML."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML deserialization")
        try:
            with open(yaml_str_or_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except (FileNotFoundError, OSError):
            data = yaml.safe_load(yaml_str_or_path)
        return cls.from_dict(data)


@dataclass
class DAGCheckpoint:
    """Checkpoint data for a DAG node execution.

    Attributes:
        execution_id: Parent execution identifier.
        node_id: Node that was checkpointed.
        status: Node status at checkpoint time.
        outputs: Node output data.
        output_hash: SHA-256 hash of outputs.
        attempt_count: Number of attempts at checkpoint.
        created_at: Checkpoint creation timestamp.
    """
    execution_id: str = ""
    node_id: str = ""
    status: NodeStatus = NodeStatus.COMPLETED
    outputs: Dict[str, Any] = field(default_factory=dict)
    output_hash: str = ""
    attempt_count: int = 1
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "execution_id": self.execution_id,
            "node_id": self.node_id,
            "status": self.status.value,
            "outputs": self.outputs,
            "output_hash": self.output_hash,
            "attempt_count": self.attempt_count,
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DAGCheckpoint:
        """Deserialize from dictionary."""
        status = data.get("status", "completed")
        if isinstance(status, str):
            status = NodeStatus(status)
        created = data.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        return cls(
            execution_id=data.get("execution_id", ""),
            node_id=data.get("node_id", ""),
            status=status,
            outputs=data.get("outputs", {}),
            output_hash=data.get("output_hash", ""),
            attempt_count=data.get("attempt_count", 1),
            created_at=created,
        )


__all__ = [
    # Enums
    "ExecutionStatus",
    "NodeStatus",
    "OnFailure",
    "DAGOnFailure",
    "RetryStrategyType",
    "OnTimeout",
    # Policy dataclasses
    "RetryPolicy",
    "TimeoutPolicy",
    # DAG definitions
    "DAGNode",
    "DAGWorkflow",
    # Execution results
    "NodeExecutionResult",
    "NodeProvenance",
    "ExecutionTrace",
    "DAGCheckpoint",
    # Availability flags
    "YAML_AVAILABLE",
]
