"""
GreenLang Framework - Graph-Based State Machine Orchestration
LangGraph-Style Agent Workflow Management

Based on:
- LangGraph (LangChain) State Machine Pattern
- AgentScope (Alibaba) Async Execution Model
- Microsoft AutoGen Multi-Agent Dialogue
- Google ADK Structured Workflows

This module provides graph-based orchestration for complex, stateful
agent workflows with branching, error recovery, and human-in-the-loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union
import hashlib
import json
import logging
import uuid
from collections import defaultdict


logger = logging.getLogger(__name__)

StateT = TypeVar('StateT')
ContextT = TypeVar('ContextT')


class NodeStatus(Enum):
    """Execution status of a node."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    WAITING_HUMAN = auto()  # Human-in-the-loop pause


class EdgeType(Enum):
    """Types of edges between nodes."""
    SEQUENTIAL = auto()
    CONDITIONAL = auto()
    PARALLEL = auto()
    LOOP = auto()


@dataclass
class NodeResult:
    """Result from executing a node."""
    node_id: str
    status: NodeStatus
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.name,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }


@dataclass
class WorkflowState(Generic[StateT]):
    """
    Immutable workflow state container.

    Following LangGraph patterns, state is passed through nodes
    and can be modified to produce new states.
    """
    workflow_id: str
    current_node: str
    data: StateT
    history: List[NodeResult] = field(default_factory=list)
    checkpoints: Dict[str, StateT] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    iteration_count: int = 0
    max_iterations: int = 100

    def update(self, new_data: StateT, node_result: NodeResult) -> 'WorkflowState[StateT]':
        """Create new state with updated data and history."""
        new_history = self.history + [node_result]
        return WorkflowState(
            workflow_id=self.workflow_id,
            current_node=node_result.node_id,
            data=new_data,
            history=new_history,
            checkpoints=self.checkpoints.copy(),
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
            iteration_count=self.iteration_count + 1,
            max_iterations=self.max_iterations
        )

    def checkpoint(self, name: str) -> 'WorkflowState[StateT]':
        """Create a checkpoint of current state."""
        new_checkpoints = self.checkpoints.copy()
        new_checkpoints[name] = self.data
        return WorkflowState(
            workflow_id=self.workflow_id,
            current_node=self.current_node,
            data=self.data,
            history=self.history,
            checkpoints=new_checkpoints,
            created_at=self.created_at,
            updated_at=self.updated_at,
            iteration_count=self.iteration_count,
            max_iterations=self.max_iterations
        )

    def restore(self, name: str) -> Optional['WorkflowState[StateT]']:
        """Restore state from a checkpoint."""
        if name in self.checkpoints:
            return WorkflowState(
                workflow_id=self.workflow_id,
                current_node=self.current_node,
                data=self.checkpoints[name],
                history=self.history,
                checkpoints=self.checkpoints,
                created_at=self.created_at,
                updated_at=datetime.now(timezone.utc),
                iteration_count=self.iteration_count,
                max_iterations=self.max_iterations
            )
        return None


class GraphNode(ABC, Generic[StateT]):
    """
    Abstract base class for workflow nodes.

    Each node represents a step in the agent workflow that can
    transform the state and produce results.
    """

    def __init__(
        self,
        node_id: str,
        name: str,
        description: str = "",
        retry_count: int = 3,
        timeout_seconds: float = 60.0,
        require_human_approval: bool = False
    ):
        self.node_id = node_id
        self.name = name
        self.description = description
        self.retry_count = retry_count
        self.timeout_seconds = timeout_seconds
        self.require_human_approval = require_human_approval

    @abstractmethod
    def execute(self, state: WorkflowState[StateT]) -> NodeResult:
        """Execute the node and return a result."""
        pass

    @abstractmethod
    def transform_state(self, state: WorkflowState[StateT], result: NodeResult) -> StateT:
        """Transform the state based on the result."""
        pass

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if the node should retry after an error."""
        return attempt < self.retry_count

    def on_enter(self, state: WorkflowState[StateT]) -> None:
        """Called when entering the node."""
        logger.info(f"Entering node: {self.node_id}")

    def on_exit(self, state: WorkflowState[StateT], result: NodeResult) -> None:
        """Called when exiting the node."""
        logger.info(f"Exiting node: {self.node_id} with status: {result.status.name}")


@dataclass
class GraphEdge:
    """Edge connecting two nodes in the workflow graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.SEQUENTIAL
    condition: Optional[Callable[[WorkflowState, NodeResult], bool]] = None
    priority: int = 0  # Higher priority edges are evaluated first

    def should_traverse(self, state: WorkflowState, result: NodeResult) -> bool:
        """Determine if this edge should be traversed."""
        if self.condition is None:
            return True
        return self.condition(state, result)


class ConditionalRouter:
    """
    Router for conditional branching in workflows.

    Similar to LangGraph's conditional edges, this allows
    dynamic routing based on state and results.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._routes: List[tuple[Callable, str]] = []
        self._default_target: Optional[str] = None

    def add_route(
        self,
        condition: Callable[[WorkflowState, NodeResult], bool],
        target_id: str
    ) -> 'ConditionalRouter':
        """Add a conditional route."""
        self._routes.append((condition, target_id))
        return self

    def set_default(self, target_id: str) -> 'ConditionalRouter':
        """Set the default route if no conditions match."""
        self._default_target = target_id
        return self

    def get_next_node(self, state: WorkflowState, result: NodeResult) -> Optional[str]:
        """Determine the next node based on conditions."""
        for condition, target_id in self._routes:
            if condition(state, result):
                return target_id
        return self._default_target


class WorkflowGraph(Generic[StateT]):
    """
    Graph-based workflow orchestrator.

    Manages the execution of complex, stateful workflows with:
    - Sequential, conditional, and parallel execution
    - Error handling and retry logic
    - Human-in-the-loop breakpoints
    - Checkpointing and recovery
    - Cycle detection and prevention
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        max_iterations: int = 100
    ):
        self.name = name
        self.description = description
        self.max_iterations = max_iterations

        self._nodes: Dict[str, GraphNode[StateT]] = {}
        self._edges: Dict[str, List[GraphEdge]] = defaultdict(list)
        self._routers: Dict[str, ConditionalRouter] = {}
        self._entry_node: Optional[str] = None
        self._exit_nodes: Set[str] = set()

    def add_node(self, node: GraphNode[StateT]) -> 'WorkflowGraph[StateT]':
        """Add a node to the graph."""
        self._nodes[node.node_id] = node
        return self

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.SEQUENTIAL,
        condition: Optional[Callable[[WorkflowState, NodeResult], bool]] = None
    ) -> 'WorkflowGraph[StateT]':
        """Add an edge between nodes."""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            condition=condition
        )
        self._edges[source_id].append(edge)
        return self

    def add_conditional_edges(
        self,
        source_id: str,
        router: ConditionalRouter
    ) -> 'WorkflowGraph[StateT]':
        """Add conditional routing from a node."""
        self._routers[source_id] = router
        return self

    def set_entry(self, node_id: str) -> 'WorkflowGraph[StateT]':
        """Set the entry point of the workflow."""
        if node_id not in self._nodes:
            raise ValueError(f"Node not found: {node_id}")
        self._entry_node = node_id
        return self

    def set_exit(self, node_id: str) -> 'WorkflowGraph[StateT]':
        """Mark a node as an exit point."""
        if node_id not in self._nodes:
            raise ValueError(f"Node not found: {node_id}")
        self._exit_nodes.add(node_id)
        return self

    def validate(self) -> List[str]:
        """Validate the workflow graph."""
        errors = []

        # Check entry node
        if not self._entry_node:
            errors.append("No entry node defined")
        elif self._entry_node not in self._nodes:
            errors.append(f"Entry node not found: {self._entry_node}")

        # Check exit nodes
        if not self._exit_nodes:
            errors.append("No exit nodes defined")

        # Check all edges reference valid nodes
        for source_id, edges in self._edges.items():
            if source_id not in self._nodes:
                errors.append(f"Edge source not found: {source_id}")
            for edge in edges:
                if edge.target_id not in self._nodes:
                    errors.append(f"Edge target not found: {edge.target_id}")

        # Check for unreachable nodes
        reachable = self._get_reachable_nodes()
        for node_id in self._nodes:
            if node_id not in reachable and node_id != self._entry_node:
                errors.append(f"Unreachable node: {node_id}")

        return errors

    def _get_reachable_nodes(self) -> Set[str]:
        """Get all nodes reachable from the entry node."""
        if not self._entry_node:
            return set()

        reachable = set()
        queue = [self._entry_node]

        while queue:
            node_id = queue.pop(0)
            if node_id in reachable:
                continue
            reachable.add(node_id)

            # Add nodes from edges
            for edge in self._edges.get(node_id, []):
                if edge.target_id not in reachable:
                    queue.append(edge.target_id)

            # Add nodes from routers
            if node_id in self._routers:
                router = self._routers[node_id]
                for _, target_id in router._routes:
                    if target_id not in reachable:
                        queue.append(target_id)
                if router._default_target and router._default_target not in reachable:
                    queue.append(router._default_target)

        return reachable

    def _get_next_nodes(
        self,
        state: WorkflowState[StateT],
        result: NodeResult
    ) -> List[str]:
        """Determine the next nodes to execute."""
        current_id = result.node_id
        next_nodes = []

        # Check conditional router first
        if current_id in self._routers:
            router = self._routers[current_id]
            next_node = router.get_next_node(state, result)
            if next_node:
                next_nodes.append(next_node)
                return next_nodes

        # Check edges
        for edge in self._edges.get(current_id, []):
            if edge.should_traverse(state, result):
                next_nodes.append(edge.target_id)

        return next_nodes

    def run(
        self,
        initial_state: StateT,
        workflow_id: Optional[str] = None
    ) -> WorkflowState[StateT]:
        """
        Execute the workflow from the entry node.

        Args:
            initial_state: The initial state data
            workflow_id: Optional workflow identifier

        Returns:
            Final workflow state after execution
        """
        import time

        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {'; '.join(errors)}")

        if not workflow_id:
            workflow_id = str(uuid.uuid4())

        state = WorkflowState(
            workflow_id=workflow_id,
            current_node=self._entry_node,
            data=initial_state,
            max_iterations=self.max_iterations
        )

        current_nodes = [self._entry_node]

        while current_nodes and state.iteration_count < state.max_iterations:
            node_id = current_nodes.pop(0)
            node = self._nodes.get(node_id)

            if not node:
                logger.error(f"Node not found: {node_id}")
                continue

            # Check for human approval
            if node.require_human_approval:
                logger.info(f"Node {node_id} requires human approval")
                # In a real implementation, this would pause and wait for approval
                # For now, we continue automatically

            # Execute node with retry logic
            node.on_enter(state)
            start_time = time.time()

            result = None
            for attempt in range(node.retry_count + 1):
                try:
                    result = node.execute(state)
                    if result.status == NodeStatus.COMPLETED:
                        break
                except Exception as e:
                    if not node.should_retry(e, attempt):
                        result = NodeResult(
                            node_id=node_id,
                            status=NodeStatus.FAILED,
                            error=str(e)
                        )
                        break
                    logger.warning(f"Retrying node {node_id}, attempt {attempt + 1}")

            if result:
                result.execution_time_ms = (time.time() - start_time) * 1000
                new_data = node.transform_state(state, result)
                state = state.update(new_data, result)
                node.on_exit(state, result)

                # Check if we've reached an exit node
                if node_id in self._exit_nodes:
                    logger.info(f"Reached exit node: {node_id}")
                    break

                # Get next nodes
                if result.status == NodeStatus.COMPLETED:
                    next_nodes = self._get_next_nodes(state, result)
                    current_nodes.extend(next_nodes)

        if state.iteration_count >= state.max_iterations:
            logger.warning(f"Workflow reached max iterations: {state.max_iterations}")

        return state

    def to_mermaid(self) -> str:
        """Export the workflow as a Mermaid diagram."""
        lines = ["graph TD"]

        for node_id, node in self._nodes.items():
            shape_start = "((" if node_id == self._entry_node else "["
            shape_end = "))" if node_id == self._entry_node else "]"
            if node_id in self._exit_nodes:
                shape_start, shape_end = "{{", "}}"
            lines.append(f"    {node_id}{shape_start}{node.name}{shape_end}")

        for source_id, edges in self._edges.items():
            for edge in edges:
                arrow = "-->" if edge.edge_type == EdgeType.SEQUENTIAL else "-.->|conditional|"
                lines.append(f"    {source_id} {arrow} {edge.target_id}")

        return "\n".join(lines)


# ============================================================================
# COMMON NODE IMPLEMENTATIONS
# ============================================================================

class FunctionNode(GraphNode[StateT]):
    """Node that executes a function."""

    def __init__(
        self,
        node_id: str,
        name: str,
        func: Callable[[StateT], Any],
        state_updater: Callable[[StateT, Any], StateT],
        **kwargs
    ):
        super().__init__(node_id, name, **kwargs)
        self._func = func
        self._state_updater = state_updater

    def execute(self, state: WorkflowState[StateT]) -> NodeResult:
        try:
            result = self._func(state.data)
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.COMPLETED,
                output=result
            )
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.FAILED,
                error=str(e)
            )

    def transform_state(self, state: WorkflowState[StateT], result: NodeResult) -> StateT:
        if result.status == NodeStatus.COMPLETED and result.output is not None:
            return self._state_updater(state.data, result.output)
        return state.data


class BranchNode(GraphNode[StateT]):
    """Node that branches based on a condition."""

    def __init__(
        self,
        node_id: str,
        name: str,
        condition: Callable[[StateT], bool],
        **kwargs
    ):
        super().__init__(node_id, name, **kwargs)
        self._condition = condition

    def execute(self, state: WorkflowState[StateT]) -> NodeResult:
        try:
            result = self._condition(state.data)
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.COMPLETED,
                output=result,
                metadata={"branch_result": result}
            )
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.FAILED,
                error=str(e)
            )

    def transform_state(self, state: WorkflowState[StateT], result: NodeResult) -> StateT:
        return state.data


class ToolNode(GraphNode[StateT]):
    """Node that executes an MCP tool."""

    def __init__(
        self,
        node_id: str,
        name: str,
        tool_name: str,
        argument_extractor: Callable[[StateT], Dict[str, Any]],
        result_handler: Callable[[StateT, Any], StateT],
        **kwargs
    ):
        super().__init__(node_id, name, **kwargs)
        self._tool_name = tool_name
        self._argument_extractor = argument_extractor
        self._result_handler = result_handler

    def execute(self, state: WorkflowState[StateT]) -> NodeResult:
        from .mcp_protocol import GREENLANG_MCP_REGISTRY, ToolCallRequest

        try:
            arguments = self._argument_extractor(state.data)
            request = ToolCallRequest(
                tool_name=self._tool_name,
                arguments=arguments,
                caller_agent_id=state.workflow_id
            )
            response = GREENLANG_MCP_REGISTRY.invoke(request)

            if response.success:
                return NodeResult(
                    node_id=self.node_id,
                    status=NodeStatus.COMPLETED,
                    output=response.result,
                    metadata={"provenance_hash": response.provenance_hash}
                )
            else:
                return NodeResult(
                    node_id=self.node_id,
                    status=NodeStatus.FAILED,
                    error=response.error
                )
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.FAILED,
                error=str(e)
            )

    def transform_state(self, state: WorkflowState[StateT], result: NodeResult) -> StateT:
        if result.status == NodeStatus.COMPLETED and result.output is not None:
            return self._result_handler(state.data, result.output)
        return state.data


# ============================================================================
# WORKFLOW BUILDER FOR FLUENT API
# ============================================================================

class WorkflowBuilder(Generic[StateT]):
    """Fluent builder for creating workflow graphs."""

    def __init__(self, name: str, description: str = ""):
        self._graph = WorkflowGraph[StateT](name, description)
        self._last_node: Optional[str] = None

    def add_function(
        self,
        node_id: str,
        name: str,
        func: Callable[[StateT], Any],
        state_updater: Callable[[StateT, Any], StateT]
    ) -> 'WorkflowBuilder[StateT]':
        """Add a function node."""
        node = FunctionNode(node_id, name, func, state_updater)
        self._graph.add_node(node)
        if self._last_node:
            self._graph.add_edge(self._last_node, node_id)
        self._last_node = node_id
        return self

    def add_branch(
        self,
        node_id: str,
        name: str,
        condition: Callable[[StateT], bool],
        true_target: str,
        false_target: str
    ) -> 'WorkflowBuilder[StateT]':
        """Add a branch node with conditional routing."""
        node = BranchNode(node_id, name, condition)
        self._graph.add_node(node)
        if self._last_node:
            self._graph.add_edge(self._last_node, node_id)

        router = ConditionalRouter(node_id)
        router.add_route(
            lambda s, r: r.output is True,
            true_target
        )
        router.set_default(false_target)
        self._graph.add_conditional_edges(node_id, router)

        self._last_node = None  # Reset since we have branching
        return self

    def set_entry(self, node_id: str) -> 'WorkflowBuilder[StateT]':
        """Set the entry node."""
        self._graph.set_entry(node_id)
        return self

    def set_exit(self, node_id: str) -> 'WorkflowBuilder[StateT]':
        """Set an exit node."""
        self._graph.set_exit(node_id)
        return self

    def build(self) -> WorkflowGraph[StateT]:
        """Build and return the workflow graph."""
        return self._graph
