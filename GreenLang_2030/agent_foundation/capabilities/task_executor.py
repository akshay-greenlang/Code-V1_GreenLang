"""
Task Executor - Multi-step task execution with state management.

This module implements comprehensive task execution capabilities including:
- Task decomposition into subtasks
- Multiple execution strategies (waterfall, parallel, iterative, adaptive)
- State management and checkpointing
- Progress tracking and reporting

Example:
    >>> executor = TaskExecutor(config)
    >>> result = await executor.execute_task(task, strategy="parallel")
    >>> progress = executor.get_progress(task.task_id)
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import pickle

from pydantic import BaseModel, Field, validator
import aiofiles
from asyncio import Queue, Semaphore
import networkx as nx

logger = logging.getLogger(__name__)


class TaskGranularity(str, Enum):
    """Granularity levels for tasks."""

    ATOMIC = "atomic"        # Single indivisible operation
    COMPOSITE = "composite"  # Multiple operations
    WORKFLOW = "workflow"    # Complete process


class ExecutionStrategyType(str, Enum):
    """Types of execution strategies."""

    WATERFALL = "waterfall"  # Sequential completion
    PARALLEL = "parallel"    # Concurrent execution
    ITERATIVE = "iterative" # Repeated refinement
    ADAPTIVE = "adaptive"   # Dynamic adjustment


class TaskStatus(str, Enum):
    """Status of task execution."""

    PENDING = "pending"
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class Task:
    """Representation of a task."""

    task_id: str
    name: str
    description: str
    granularity: TaskGranularity
    dependencies: Set[str] = field(default_factory=set)
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_resources: Dict[str, float] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    """Task execution record."""

    task_id: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    checkpoint: Optional[Dict[str, Any]] = None


class TaskResult(BaseModel):
    """Result from task execution."""

    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float
    subtask_results: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    checkpoint_id: Optional[str] = None


class TaskDecomposer:
    """Decompose complex tasks into subtasks."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize task decomposer."""
        self.config = config or {}
        self.decomposition_strategies = {
            "hierarchical": self._hierarchical_decomposition,
            "sequential": self._sequential_decomposition,
            "parallel": self._parallel_decomposition,
            "conditional": self._conditional_decomposition
        }

    async def decompose(
        self,
        task: Task,
        strategy: str = "hierarchical"
    ) -> List[Task]:
        """Decompose task into subtasks."""
        if task.granularity == TaskGranularity.ATOMIC:
            return [task]  # Already atomic, no decomposition needed

        decomposer = self.decomposition_strategies.get(
            strategy,
            self._hierarchical_decomposition
        )

        subtasks = await decomposer(task)
        return subtasks

    async def _hierarchical_decomposition(self, task: Task) -> List[Task]:
        """Hierarchically decompose task."""
        subtasks = []

        # Analyze task to identify components
        components = self._identify_components(task)

        for i, component in enumerate(components):
            subtask = Task(
                task_id=f"{task.task_id}_sub_{i}",
                name=component["name"],
                description=component["description"],
                granularity=TaskGranularity.ATOMIC,
                dependencies=component.get("dependencies", set()),
                parameters=component.get("parameters", {}),
                required_resources=self._estimate_resources(component),
                timeout_seconds=task.timeout_seconds / len(components),
                priority=task.priority
            )
            subtasks.append(subtask)

        # Set up dependencies between subtasks
        subtasks = self._setup_dependencies(subtasks, task)

        return subtasks

    async def _sequential_decomposition(self, task: Task) -> List[Task]:
        """Sequentially decompose task."""
        subtasks = []
        steps = self._identify_sequential_steps(task)

        prev_task_id = None
        for i, step in enumerate(steps):
            dependencies = {prev_task_id} if prev_task_id else set()

            subtask = Task(
                task_id=f"{task.task_id}_seq_{i}",
                name=step["name"],
                description=step["description"],
                granularity=TaskGranularity.ATOMIC,
                dependencies=dependencies,
                parameters=step.get("parameters", {}),
                timeout_seconds=task.timeout_seconds / len(steps),
                priority=task.priority
            )
            subtasks.append(subtask)
            prev_task_id = subtask.task_id

        return subtasks

    async def _parallel_decomposition(self, task: Task) -> List[Task]:
        """Decompose into parallel subtasks."""
        subtasks = []
        parallel_components = self._identify_parallel_components(task)

        for i, component in enumerate(parallel_components):
            subtask = Task(
                task_id=f"{task.task_id}_par_{i}",
                name=component["name"],
                description=component["description"],
                granularity=TaskGranularity.ATOMIC,
                dependencies=set(),  # No dependencies for parallel tasks
                parameters=component.get("parameters", {}),
                timeout_seconds=task.timeout_seconds,  # All get full timeout
                priority=task.priority
            )
            subtasks.append(subtask)

        return subtasks

    async def _conditional_decomposition(self, task: Task) -> List[Task]:
        """Decompose with conditional branches."""
        subtasks = []
        branches = self._identify_conditional_branches(task)

        for i, branch in enumerate(branches):
            subtask = Task(
                task_id=f"{task.task_id}_cond_{i}",
                name=branch["name"],
                description=branch["description"],
                granularity=TaskGranularity.ATOMIC,
                dependencies=branch.get("dependencies", set()),
                parameters={
                    **branch.get("parameters", {}),
                    "condition": branch["condition"]
                },
                timeout_seconds=task.timeout_seconds / len(branches),
                priority=task.priority
            )
            subtasks.append(subtask)

        return subtasks

    def _identify_components(self, task: Task) -> List[Dict[str, Any]]:
        """Identify components in task."""
        # Analyze task description and parameters
        components = []

        # Simple heuristic decomposition
        if "steps" in task.metadata:
            for step in task.metadata["steps"]:
                components.append({
                    "name": step.get("name", "Step"),
                    "description": step.get("description", ""),
                    "parameters": step.get("parameters", {}),
                    "dependencies": step.get("dependencies", [])
                })
        else:
            # Default decomposition into 3 phases
            components = [
                {"name": "Initialize", "description": "Setup phase"},
                {"name": "Process", "description": "Main processing"},
                {"name": "Finalize", "description": "Cleanup phase"}
            ]

        return components

    def _identify_sequential_steps(self, task: Task) -> List[Dict[str, Any]]:
        """Identify sequential steps in task."""
        # Extract sequential steps from task
        if "sequence" in task.metadata:
            return task.metadata["sequence"]

        # Default sequential steps
        return [
            {"name": "Prepare", "description": "Prepare inputs"},
            {"name": "Execute", "description": "Execute main logic"},
            {"name": "Validate", "description": "Validate results"}
        ]

    def _identify_parallel_components(self, task: Task) -> List[Dict[str, Any]]:
        """Identify parallel components."""
        if "parallel" in task.metadata:
            return task.metadata["parallel"]

        # Default parallel components based on data
        if "data_items" in task.parameters:
            items = task.parameters["data_items"]
            return [
                {"name": f"Process_{i}", "description": f"Process item {i}"}
                for i in range(len(items))
            ]

        return []

    def _identify_conditional_branches(self, task: Task) -> List[Dict[str, Any]]:
        """Identify conditional branches."""
        if "conditions" in task.metadata:
            return task.metadata["conditions"]

        # Default conditional branches
        return [
            {
                "name": "Main",
                "description": "Main execution path",
                "condition": "default"
            },
            {
                "name": "Alternative",
                "description": "Alternative path",
                "condition": "fallback"
            }
        ]

    def _estimate_resources(self, component: Dict[str, Any]) -> Dict[str, float]:
        """Estimate resource requirements for component."""
        # Simple resource estimation
        return {
            "cpu": component.get("cpu_estimate", 0.1),
            "memory_mb": component.get("memory_estimate", 100),
            "disk_mb": component.get("disk_estimate", 10)
        }

    def _setup_dependencies(
        self,
        subtasks: List[Task],
        parent_task: Task
    ) -> List[Task]:
        """Setup dependencies between subtasks."""
        # If parent has dependency info, use it
        if "dependency_graph" in parent_task.metadata:
            graph = parent_task.metadata["dependency_graph"]
            for subtask in subtasks:
                task_name = subtask.name
                if task_name in graph:
                    deps = graph[task_name]
                    # Convert names to IDs
                    dep_ids = set()
                    for dep_name in deps:
                        for st in subtasks:
                            if st.name == dep_name:
                                dep_ids.add(st.task_id)
                    subtask.dependencies = dep_ids

        return subtasks


class ExecutionStrategy:
    """Base class for execution strategies."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize execution strategy."""
        self.config = config or {}

    async def execute(
        self,
        tasks: List[Task],
        executor: Callable
    ) -> Dict[str, TaskResult]:
        """Execute tasks according to strategy."""
        raise NotImplementedError


class WaterfallStrategy(ExecutionStrategy):
    """Sequential waterfall execution strategy."""

    async def execute(
        self,
        tasks: List[Task],
        executor: Callable
    ) -> Dict[str, TaskResult]:
        """Execute tasks sequentially."""
        results = {}

        # Sort tasks by dependencies
        sorted_tasks = self._topological_sort(tasks)

        for task in sorted_tasks:
            # Wait for dependencies
            await self._wait_for_dependencies(task, results)

            # Execute task
            result = await executor(task)
            results[task.task_id] = result

            # Stop on failure if configured
            if result.status == TaskStatus.FAILED:
                if self.config.get("stop_on_failure", True):
                    break

        return results

    def _topological_sort(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks topologically based on dependencies."""
        graph = nx.DiGraph()

        # Add nodes
        for task in tasks:
            graph.add_node(task.task_id, task=task)

        # Add edges
        for task in tasks:
            for dep in task.dependencies:
                if graph.has_node(dep):
                    graph.add_edge(dep, task.task_id)

        # Topological sort
        try:
            sorted_ids = list(nx.topological_sort(graph))
            return [graph.nodes[tid]["task"] for tid in sorted_ids]
        except nx.NetworkXError:
            # Has cycles, return original order
            return tasks

    async def _wait_for_dependencies(
        self,
        task: Task,
        results: Dict[str, TaskResult]
    ) -> None:
        """Wait for task dependencies to complete."""
        for dep_id in task.dependencies:
            while dep_id not in results:
                await asyncio.sleep(0.1)

            # Check if dependency succeeded
            if results[dep_id].status == TaskStatus.FAILED:
                raise Exception(f"Dependency {dep_id} failed")


class ParallelStrategy(ExecutionStrategy):
    """Parallel execution strategy."""

    async def execute(
        self,
        tasks: List[Task],
        executor: Callable
    ) -> Dict[str, TaskResult]:
        """Execute tasks in parallel."""
        max_workers = self.config.get("max_workers", 10)
        semaphore = Semaphore(max_workers)

        async def execute_with_semaphore(task):
            async with semaphore:
                return await executor(task)

        # Group tasks by dependency level
        levels = self._group_by_dependency_level(tasks)
        results = {}

        # Execute each level in parallel
        for level_tasks in levels:
            level_results = await asyncio.gather(
                *[execute_with_semaphore(task) for task in level_tasks],
                return_exceptions=True
            )

            for task, result in zip(level_tasks, level_results):
                if isinstance(result, Exception):
                    results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error=str(result),
                        execution_time_ms=0
                    )
                else:
                    results[task.task_id] = result

        return results

    def _group_by_dependency_level(self, tasks: List[Task]) -> List[List[Task]]:
        """Group tasks by dependency level for parallel execution."""
        levels = []
        task_dict = {t.task_id: t for t in tasks}
        completed = set()

        while len(completed) < len(tasks):
            level = []
            for task in tasks:
                if task.task_id not in completed:
                    # Check if all dependencies are completed
                    if task.dependencies.issubset(completed):
                        level.append(task)

            if not level:
                # No progress possible, add remaining tasks
                level = [t for t in tasks if t.task_id not in completed]

            levels.append(level)
            completed.update(t.task_id for t in level)

        return levels


class IterativeStrategy(ExecutionStrategy):
    """Iterative refinement execution strategy."""

    async def execute(
        self,
        tasks: List[Task],
        executor: Callable
    ) -> Dict[str, TaskResult]:
        """Execute tasks with iterative refinement."""
        max_iterations = self.config.get("max_iterations", 3)
        convergence_threshold = self.config.get("convergence_threshold", 0.95)

        results = {}
        quality_scores = {}

        for iteration in range(max_iterations):
            iteration_results = {}

            for task in tasks:
                # Execute task
                result = await executor(task)
                iteration_results[task.task_id] = result

                # Calculate quality score
                quality = self._calculate_quality(result)
                quality_scores[task.task_id] = quality

            # Check convergence
            avg_quality = sum(quality_scores.values()) / len(quality_scores)
            if avg_quality >= convergence_threshold:
                results = iteration_results
                break

            # Update tasks for next iteration
            tasks = self._refine_tasks(tasks, iteration_results)
            results = iteration_results

        return results

    def _calculate_quality(self, result: TaskResult) -> float:
        """Calculate quality score for result."""
        if result.status == TaskStatus.COMPLETED:
            # Use metrics if available
            if "quality" in result.metrics:
                return result.metrics["quality"]
            return 0.8  # Default quality for completed tasks
        return 0.0  # Failed tasks have zero quality

    def _refine_tasks(
        self,
        tasks: List[Task],
        results: Dict[str, TaskResult]
    ) -> List[Task]:
        """Refine tasks based on previous results."""
        refined = []

        for task in tasks:
            result = results.get(task.task_id)
            if result and result.status == TaskStatus.FAILED:
                # Adjust parameters for retry
                refined_task = Task(
                    task_id=task.task_id,
                    name=task.name,
                    description=task.description,
                    granularity=task.granularity,
                    dependencies=task.dependencies,
                    parameters={
                        **task.parameters,
                        "retry_adjustment": True
                    },
                    timeout_seconds=task.timeout_seconds * 1.5,  # Increase timeout
                    retry_count=task.retry_count + 1,
                    max_retries=task.max_retries,
                    priority=task.priority
                )
                refined.append(refined_task)
            else:
                refined.append(task)

        return refined


class AdaptiveStrategy(ExecutionStrategy):
    """Adaptive execution strategy that adjusts dynamically."""

    async def execute(
        self,
        tasks: List[Task],
        executor: Callable
    ) -> Dict[str, TaskResult]:
        """Execute tasks with adaptive adjustments."""
        results = {}
        execution_queue = asyncio.Queue()

        # Initialize queue with tasks
        for task in tasks:
            await execution_queue.put(task)

        # Adaptive execution loop
        while not execution_queue.empty():
            # Get next task
            task = await execution_queue.get()

            # Check system resources
            resources = await self._check_resources()

            # Adapt execution based on resources
            adapted_task = self._adapt_task(task, resources)

            # Execute task
            try:
                result = await executor(adapted_task)
                results[task.task_id] = result

                # Analyze result and adapt strategy
                await self._adapt_strategy(result, execution_queue)

            except Exception as e:
                # Handle failure adaptively
                retry_task = self._create_retry_task(task, str(e))
                if retry_task:
                    await execution_queue.put(retry_task)
                else:
                    results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error=str(e),
                        execution_time_ms=0
                    )

        return results

    async def _check_resources(self) -> Dict[str, float]:
        """Check available system resources."""
        import psutil

        return {
            "cpu_available": 100 - psutil.cpu_percent(),
            "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 1.0
        }

    def _adapt_task(self, task: Task, resources: Dict[str, float]) -> Task:
        """Adapt task based on available resources."""
        adapted = task

        # Reduce parallelism if resources are low
        if resources["cpu_available"] < 20:
            adapted.parameters["parallelism"] = 1
        elif resources["cpu_available"] < 50:
            adapted.parameters["parallelism"] = 2
        else:
            adapted.parameters["parallelism"] = 4

        # Adjust timeout based on load
        if resources.get("load_average", 1.0) > 2.0:
            adapted.timeout_seconds *= 1.5

        return adapted

    async def _adapt_strategy(
        self,
        result: TaskResult,
        queue: asyncio.Queue
    ) -> None:
        """Adapt strategy based on execution results."""
        if result.status == TaskStatus.FAILED:
            # Add diagnostic task
            diagnostic_task = self._create_diagnostic_task(result)
            if diagnostic_task:
                await queue.put(diagnostic_task)

        elif result.execution_time_ms > 10000:  # Slow execution
            # Consider decomposing into smaller tasks
            pass

    def _create_retry_task(self, task: Task, error: str) -> Optional[Task]:
        """Create retry task if within retry limit."""
        if task.retry_count < task.max_retries:
            return Task(
                task_id=f"{task.task_id}_retry_{task.retry_count + 1}",
                name=task.name,
                description=task.description,
                granularity=task.granularity,
                dependencies=task.dependencies,
                parameters={
                    **task.parameters,
                    "previous_error": error
                },
                timeout_seconds=task.timeout_seconds * 1.5,
                retry_count=task.retry_count + 1,
                max_retries=task.max_retries,
                priority=task.priority - 1  # Increase priority
            )
        return None

    def _create_diagnostic_task(self, failed_result: TaskResult) -> Optional[Task]:
        """Create diagnostic task for failure analysis."""
        return Task(
            task_id=f"{failed_result.task_id}_diagnostic",
            name="Diagnose Failure",
            description=f"Diagnose failure of {failed_result.task_id}",
            granularity=TaskGranularity.ATOMIC,
            dependencies=set(),
            parameters={
                "failed_task_id": failed_result.task_id,
                "error": failed_result.error
            },
            timeout_seconds=10,
            priority=1  # High priority
        )


class StateManager:
    """Manage task execution state."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize state manager."""
        self.storage_path = storage_path or Path("task_state.json")
        self.state: Dict[str, TaskExecution] = {}
        self._load_state()

    def save_state(self, task_id: str, execution: TaskExecution) -> None:
        """Save task execution state."""
        self.state[task_id] = execution
        self._persist_state()

    def get_state(self, task_id: str) -> Optional[TaskExecution]:
        """Get task execution state."""
        return self.state.get(task_id)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Any] = None,
        error: Optional[str] = None
    ) -> None:
        """Update task status."""
        if task_id in self.state:
            execution = self.state[task_id]
            execution.status = status
            if result is not None:
                execution.result = result
            if error is not None:
                execution.error = error
            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                execution.end_time = datetime.now()
            self._persist_state()

    def _persist_state(self) -> None:
        """Persist state to storage."""
        try:
            state_dict = {
                task_id: {
                    "status": execution.status,
                    "start_time": execution.start_time.isoformat() if execution.start_time else None,
                    "end_time": execution.end_time.isoformat() if execution.end_time else None,
                    "error": execution.error,
                    "execution_time_ms": execution.execution_time_ms
                }
                for task_id, execution in self.state.items()
            }

            with open(self.storage_path, "w") as f:
                json.dump(state_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist state: {str(e)}")

    def _load_state(self) -> None:
        """Load state from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    state_dict = json.load(f)

                for task_id, data in state_dict.items():
                    execution = TaskExecution(
                        task_id=task_id,
                        status=data["status"],
                        start_time=datetime.fromisoformat(data["start_time"])
                        if data["start_time"] else None,
                        end_time=datetime.fromisoformat(data["end_time"])
                        if data["end_time"] else None,
                        error=data.get("error"),
                        execution_time_ms=data.get("execution_time_ms", 0)
                    )
                    self.state[task_id] = execution
            except Exception as e:
                logger.error(f"Failed to load state: {str(e)}")
                self.state = {}


class CheckpointManager:
    """Manage task execution checkpoints."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize checkpoint manager."""
        self.storage_path = storage_path or Path("checkpoints")
        self.storage_path.mkdir(exist_ok=True)

    async def save_checkpoint(
        self,
        task_id: str,
        checkpoint_data: Dict[str, Any]
    ) -> str:
        """Save checkpoint for task."""
        checkpoint_id = self._generate_checkpoint_id(task_id)
        checkpoint_path = self.storage_path / f"{checkpoint_id}.pkl"

        try:
            async with aiofiles.open(checkpoint_path, "wb") as f:
                await f.write(pickle.dumps(checkpoint_data))

            logger.info(f"Saved checkpoint {checkpoint_id} for task {task_id}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint by ID."""
        checkpoint_path = self.storage_path / f"{checkpoint_id}.pkl"

        if not checkpoint_path.exists():
            return None

        try:
            async with aiofiles.open(checkpoint_path, "rb") as f:
                data = await f.read()
                return pickle.loads(data)

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return None

    def list_checkpoints(self, task_id: Optional[str] = None) -> List[str]:
        """List available checkpoints."""
        checkpoints = []

        for checkpoint_file in self.storage_path.glob("*.pkl"):
            checkpoint_id = checkpoint_file.stem
            if task_id is None or task_id in checkpoint_id:
                checkpoints.append(checkpoint_id)

        return checkpoints

    def _generate_checkpoint_id(self, task_id: str) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{task_id}{timestamp}"
        return f"ckpt_{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"


class ProgressTracker:
    """Track task execution progress."""

    def __init__(self):
        """Initialize progress tracker."""
        self.progress: Dict[str, Dict[str, Any]] = {}

    def update_progress(
        self,
        task_id: str,
        completed_steps: int,
        total_steps: int,
        current_step: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Update task progress."""
        self.progress[task_id] = {
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "percentage": (completed_steps / total_steps * 100) if total_steps > 0 else 0,
            "current_step": current_step,
            "metrics": metrics or {},
            "last_updated": datetime.now()
        }

    def get_progress(self, task_id: str) -> Dict[str, Any]:
        """Get task progress."""
        return self.progress.get(task_id, {
            "completed_steps": 0,
            "total_steps": 0,
            "percentage": 0,
            "current_step": None,
            "metrics": {}
        })

    def get_overall_progress(self, task_ids: List[str]) -> Dict[str, Any]:
        """Get overall progress for multiple tasks."""
        if not task_ids:
            return {"percentage": 0, "tasks_completed": 0, "tasks_total": 0}

        completed = 0
        total_percentage = 0

        for task_id in task_ids:
            progress = self.get_progress(task_id)
            total_percentage += progress["percentage"]
            if progress["percentage"] >= 100:
                completed += 1

        return {
            "percentage": total_percentage / len(task_ids),
            "tasks_completed": completed,
            "tasks_total": len(task_ids),
            "individual_progress": {
                task_id: self.get_progress(task_id)["percentage"]
                for task_id in task_ids
            }
        }


class TaskExecutor:
    """Main task executor with comprehensive capabilities."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize task executor."""
        self.config = config or {}
        self.decomposer = TaskDecomposer(config)
        self.state_manager = StateManager()
        self.checkpoint_manager = CheckpointManager()
        self.progress_tracker = ProgressTracker()

        # Initialize execution strategies
        self.strategies = {
            ExecutionStrategyType.WATERFALL: WaterfallStrategy(config),
            ExecutionStrategyType.PARALLEL: ParallelStrategy(config),
            ExecutionStrategyType.ITERATIVE: IterativeStrategy(config),
            ExecutionStrategyType.ADAPTIVE: AdaptiveStrategy(config)
        }

        # Execution queue
        self.execution_queue = asyncio.Queue()
        self.active_tasks: Dict[str, Task] = {}

    async def execute_task(
        self,
        task: Task,
        strategy: ExecutionStrategyType = ExecutionStrategyType.ADAPTIVE,
        executor_func: Optional[Callable] = None
    ) -> TaskResult:
        """Execute a task with specified strategy."""
        # Record start
        self.state_manager.save_state(
            task.task_id,
            TaskExecution(
                task_id=task.task_id,
                status=TaskStatus.EXECUTING,
                start_time=datetime.now()
            )
        )

        self.active_tasks[task.task_id] = task

        try:
            # Decompose if needed
            if task.granularity != TaskGranularity.ATOMIC:
                subtasks = await self.decomposer.decompose(task)
            else:
                subtasks = [task]

            # Update progress
            self.progress_tracker.update_progress(
                task.task_id,
                completed_steps=0,
                total_steps=len(subtasks)
            )

            # Execute with strategy
            execution_strategy = self.strategies[strategy]
            executor = executor_func or self._default_executor

            results = await execution_strategy.execute(subtasks, executor)

            # Aggregate results
            final_result = self._aggregate_results(task.task_id, results)

            # Update state
            self.state_manager.update_status(
                task.task_id,
                TaskStatus.COMPLETED,
                result=final_result
            )

            return final_result

        except Exception as e:
            logger.error(f"Task {task.task_id} execution failed: {str(e)}")

            # Update state
            self.state_manager.update_status(
                task.task_id,
                TaskStatus.FAILED,
                error=str(e)
            )

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time_ms=0
            )

        finally:
            del self.active_tasks[task.task_id]

    async def _default_executor(self, task: Task) -> TaskResult:
        """Default task executor."""
        start_time = time.time()

        try:
            # Simulate task execution
            await asyncio.sleep(0.1)  # Placeholder for actual execution

            # Create checkpoint
            checkpoint_id = await self.checkpoint_manager.save_checkpoint(
                task.task_id,
                {"task": task, "timestamp": datetime.now()}
            )

            execution_time = (time.time() - start_time) * 1000

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result={"executed": True},
                execution_time_ms=execution_time,
                checkpoint_id=checkpoint_id
            )

        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def _aggregate_results(
        self,
        parent_id: str,
        subtask_results: Dict[str, TaskResult]
    ) -> TaskResult:
        """Aggregate subtask results."""
        # Check overall status
        all_completed = all(
            r.status == TaskStatus.COMPLETED
            for r in subtask_results.values()
        )

        overall_status = TaskStatus.COMPLETED if all_completed else TaskStatus.FAILED

        # Aggregate execution time
        total_time = sum(r.execution_time_ms for r in subtask_results.values())

        # Aggregate metrics
        aggregated_metrics = {}
        for result in subtask_results.values():
            for metric, value in result.metrics.items():
                if metric not in aggregated_metrics:
                    aggregated_metrics[metric] = 0
                aggregated_metrics[metric] += value

        return TaskResult(
            task_id=parent_id,
            status=overall_status,
            result={"subtask_count": len(subtask_results)},
            execution_time_ms=total_time,
            subtask_results=subtask_results,
            metrics=aggregated_metrics
        )

    async def pause_task(self, task_id: str) -> bool:
        """Pause task execution."""
        if task_id in self.active_tasks:
            self.state_manager.update_status(task_id, TaskStatus.PAUSED)
            return True
        return False

    async def resume_task(self, task_id: str) -> bool:
        """Resume paused task."""
        state = self.state_manager.get_state(task_id)
        if state and state.status == TaskStatus.PAUSED:
            # Load checkpoint if available
            checkpoints = self.checkpoint_manager.list_checkpoints(task_id)
            if checkpoints:
                checkpoint_data = await self.checkpoint_manager.load_checkpoint(
                    checkpoints[-1]  # Most recent
                )
                # Resume from checkpoint
                # Implementation depends on task type

            self.state_manager.update_status(task_id, TaskStatus.EXECUTING)
            return True
        return False

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel task execution."""
        if task_id in self.active_tasks:
            self.state_manager.update_status(task_id, TaskStatus.CANCELLED)
            del self.active_tasks[task_id]
            return True
        return False

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current task status."""
        state = self.state_manager.get_state(task_id)
        return state.status if state else None

    def get_progress(self, task_id: str) -> Dict[str, Any]:
        """Get task progress."""
        return self.progress_tracker.get_progress(task_id)

    def get_active_tasks(self) -> List[str]:
        """Get list of active task IDs."""
        return list(self.active_tasks.keys())

    async def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """Clean up old checkpoints."""
        cutoff = datetime.now() - timedelta(days=days)
        cleaned = 0

        for checkpoint_file in self.checkpoint_manager.storage_path.glob("*.pkl"):
            if checkpoint_file.stat().st_mtime < cutoff.timestamp():
                checkpoint_file.unlink()
                cleaned += 1

        logger.info(f"Cleaned up {cleaned} old checkpoints")
        return cleaned