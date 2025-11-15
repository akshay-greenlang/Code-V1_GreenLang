"""
CoordinatorAgent - Swarm orchestration and coordination agent.

This module implements the CoordinatorAgent for managing distributed agent swarms,
task allocation, workflow orchestration, and collective intelligence coordination.

Example:
    >>> agent = CoordinatorAgent(config)
    >>> result = await agent.execute(CoordinationInput(
    ...     task="process_sustainability_data",
    ...     worker_count=10,
    ...     data_chunks=data_list,
    ...     strategy="parallel"
    ... ))
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent, AgentConfig, ExecutionContext

logger = logging.getLogger(__name__)


class CoordinationStrategy(str, Enum):
    """Coordination strategies for agent swarms."""

    PARALLEL = "parallel"  # All workers process simultaneously
    SEQUENTIAL = "sequential"  # Workers process one after another
    PIPELINE = "pipeline"  # Data flows through worker stages
    MAP_REDUCE = "map_reduce"  # Map phase then reduce phase
    SCATTER_GATHER = "scatter_gather"  # Distribute and collect
    HIERARCHICAL = "hierarchical"  # Multi-level coordination
    DYNAMIC = "dynamic"  # Adaptive strategy based on load
    CONSENSUS = "consensus"  # Workers reach agreement


class WorkerStatus(str, Enum):
    """Status of worker agents."""

    IDLE = "idle"
    ASSIGNED = "assigned"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class CoordinationInput(BaseModel):
    """Input data model for CoordinatorAgent."""

    task: str = Field(..., description="Task to coordinate")
    strategy: CoordinationStrategy = Field(
        CoordinationStrategy.PARALLEL,
        description="Coordination strategy"
    )
    worker_count: int = Field(5, ge=1, le=100, description="Number of worker agents")
    data_chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Data chunks to process"
    )
    worker_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for worker agents"
    )
    timeout_seconds: int = Field(300, ge=10, le=3600, description="Task timeout")
    retry_failed: bool = Field(True, description="Retry failed tasks")
    consensus_threshold: float = Field(
        0.8, ge=0.5, le=1.0,
        description="Threshold for consensus strategy"
    )
    priority_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Priority weights for task allocation"
    )

    @validator('data_chunks')
    def validate_chunks(cls, v, values):
        """Validate data chunks match worker count for certain strategies."""
        strategy = values.get('strategy')
        if strategy == CoordinationStrategy.PIPELINE and len(v) < 2:
            raise ValueError("Pipeline strategy requires at least 2 data chunks")
        return v


class CoordinationOutput(BaseModel):
    """Output data model for CoordinatorAgent."""

    success: bool = Field(..., description="Coordination success status")
    task: str = Field(..., description="Coordinated task")
    strategy: CoordinationStrategy = Field(..., description="Strategy used")
    workers_deployed: int = Field(..., description="Number of workers deployed")
    workers_succeeded: int = Field(..., description="Number of successful workers")
    workers_failed: int = Field(..., description="Number of failed workers")
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Aggregated results from workers"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics"
    )
    worker_stats: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Individual worker statistics"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., description="Total processing duration")
    parallel_efficiency: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Parallel processing efficiency"
    )


class CoordinatorAgent(BaseAgent):
    """
    CoordinatorAgent implementation for swarm orchestration.

    This agent manages distributed processing across multiple worker agents,
    handles task allocation, monitors progress, and aggregates results.

    Attributes:
        config: Agent configuration
        worker_pool: Pool of available worker agents
        task_queue: Queue of pending tasks
        active_tasks: Currently executing tasks

    Example:
        >>> config = AgentConfig(name="swarm_coordinator", version="1.0.0")
        >>> agent = CoordinatorAgent(config)
        >>> await agent.initialize()
        >>> result = await agent.execute(coordination_input)
        >>> print(f"Workers succeeded: {result.result.workers_succeeded}/{result.result.workers_deployed}")
    """

    def __init__(self, config: AgentConfig):
        """Initialize CoordinatorAgent."""
        super().__init__(config)
        self.worker_pool: List[WorkerAgent] = []
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, TaskInfo] = {}
        self.completed_tasks: List[TaskInfo] = []
        self.coordination_history: List[CoordinationOutput] = []

    async def _initialize_core(self) -> None:
        """Initialize coordinator resources."""
        self._logger.info("Initializing CoordinatorAgent resources")

        # Initialize task queue
        self.task_queue = asyncio.Queue()

        # Initialize monitoring
        self.performance_monitor = PerformanceMonitor()

        self._logger.info("Coordinator initialized")

    async def _execute_core(self, input_data: CoordinationInput, context: ExecutionContext) -> CoordinationOutput:
        """
        Core execution logic for coordination.

        This method orchestrates task distribution and result aggregation.
        """
        start_time = datetime.now(timezone.utc)
        results = []
        worker_stats = []

        try:
            # Step 1: Create worker pool
            self._logger.info(f"Creating worker pool with {input_data.worker_count} workers")
            worker_pool = await self._create_worker_pool(
                input_data.worker_count,
                input_data.worker_config
            )

            # Step 2: Prepare tasks
            tasks = self._prepare_tasks(input_data)
            self._logger.info(f"Prepared {len(tasks)} tasks for coordination")

            # Step 3: Execute coordination strategy
            if input_data.strategy == CoordinationStrategy.PARALLEL:
                results, worker_stats = await self._execute_parallel(
                    tasks, worker_pool, input_data.timeout_seconds
                )

            elif input_data.strategy == CoordinationStrategy.SEQUENTIAL:
                results, worker_stats = await self._execute_sequential(
                    tasks, worker_pool, input_data.timeout_seconds
                )

            elif input_data.strategy == CoordinationStrategy.PIPELINE:
                results, worker_stats = await self._execute_pipeline(
                    tasks, worker_pool, input_data.timeout_seconds
                )

            elif input_data.strategy == CoordinationStrategy.MAP_REDUCE:
                results, worker_stats = await self._execute_map_reduce(
                    tasks, worker_pool, input_data.timeout_seconds
                )

            elif input_data.strategy == CoordinationStrategy.SCATTER_GATHER:
                results, worker_stats = await self._execute_scatter_gather(
                    tasks, worker_pool, input_data.timeout_seconds
                )

            elif input_data.strategy == CoordinationStrategy.CONSENSUS:
                results, worker_stats = await self._execute_consensus(
                    tasks, worker_pool,
                    input_data.consensus_threshold,
                    input_data.timeout_seconds
                )

            else:
                raise ValueError(f"Unsupported strategy: {input_data.strategy}")

            # Step 4: Handle failed tasks
            failed_tasks = [s for s in worker_stats if s["status"] == WorkerStatus.FAILED]
            if failed_tasks and input_data.retry_failed:
                self._logger.info(f"Retrying {len(failed_tasks)} failed tasks")
                retry_results = await self._retry_failed_tasks(
                    failed_tasks, worker_pool, input_data.timeout_seconds
                )
                results.extend(retry_results)

            # Step 5: Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                worker_stats, start_time
            )

            # Step 6: Calculate parallel efficiency
            parallel_efficiency = self._calculate_parallel_efficiency(
                worker_stats, input_data.worker_count
            )

            # Step 7: Cleanup worker pool
            await self._cleanup_worker_pool(worker_pool)

            # Step 8: Generate provenance hash
            provenance_hash = self._calculate_provenance_hash(
                input_data.dict(),
                results,
                context.execution_id
            )

            # Step 9: Calculate total processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Step 10: Create output
            output = CoordinationOutput(
                success=len(failed_tasks) == 0,
                task=input_data.task,
                strategy=input_data.strategy,
                workers_deployed=input_data.worker_count,
                workers_succeeded=sum(1 for s in worker_stats if s["status"] == WorkerStatus.COMPLETED),
                workers_failed=len(failed_tasks),
                results=results,
                performance_metrics=performance_metrics,
                worker_stats=worker_stats,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time,
                parallel_efficiency=parallel_efficiency
            )

            # Store in history
            self.coordination_history.append(output)
            if len(self.coordination_history) > 50:
                self.coordination_history.pop(0)

            return output

        except Exception as e:
            self._logger.error(f"Coordination failed: {str(e)}", exc_info=True)
            raise

    async def _create_worker_pool(self, count: int, config: Dict) -> List['WorkerAgent']:
        """Create pool of worker agents."""
        workers = []
        for i in range(count):
            worker_config = AgentConfig(
                name=f"worker_{i}",
                version="1.0.0",
                **config
            )
            # In production, would create actual WorkerAgent instances
            # For now, create mock workers
            worker = MockWorker(worker_config, i)
            workers.append(worker)

        return workers

    def _prepare_tasks(self, input_data: CoordinationInput) -> List['TaskInfo']:
        """Prepare tasks for distribution."""
        tasks = []

        if input_data.data_chunks:
            # Create task for each data chunk
            for i, chunk in enumerate(input_data.data_chunks):
                task = TaskInfo(
                    task_id=str(uuid.uuid4()),
                    task_name=f"{input_data.task}_{i}",
                    data=chunk,
                    priority=input_data.priority_weights.get(str(i), 1.0)
                )
                tasks.append(task)
        else:
            # Create single task
            task = TaskInfo(
                task_id=str(uuid.uuid4()),
                task_name=input_data.task,
                data={},
                priority=1.0
            )
            tasks.append(task)

        return tasks

    async def _execute_parallel(self, tasks: List['TaskInfo'], workers: List, timeout: int) -> tuple:
        """Execute tasks in parallel across workers."""
        self._logger.info(f"Executing {len(tasks)} tasks in parallel")

        # Assign tasks to workers
        assignments = self._assign_tasks_to_workers(tasks, workers)

        # Execute all tasks concurrently
        async_tasks = []
        for worker, task in assignments:
            async_task = asyncio.create_task(
                self._execute_worker_task(worker, task, timeout)
            )
            async_tasks.append(async_task)

        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*async_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self._logger.error("Parallel execution timeout")
            results = []

        # Process results
        worker_stats = []
        processed_results = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                worker_stats.append({
                    "worker_id": i,
                    "status": WorkerStatus.FAILED,
                    "error": str(result),
                    "execution_time": 0
                })
            else:
                worker_stats.append(result["stats"])
                processed_results.append(result["data"])

        return processed_results, worker_stats

    async def _execute_sequential(self, tasks: List['TaskInfo'], workers: List, timeout: int) -> tuple:
        """Execute tasks sequentially."""
        self._logger.info(f"Executing {len(tasks)} tasks sequentially")

        results = []
        worker_stats = []

        for i, task in enumerate(tasks):
            worker = workers[i % len(workers)]
            try:
                result = await asyncio.wait_for(
                    self._execute_worker_task(worker, task, timeout),
                    timeout=timeout // len(tasks)
                )
                worker_stats.append(result["stats"])
                results.append(result["data"])
            except Exception as e:
                worker_stats.append({
                    "worker_id": i,
                    "status": WorkerStatus.FAILED,
                    "error": str(e),
                    "execution_time": 0
                })

        return results, worker_stats

    async def _execute_pipeline(self, tasks: List['TaskInfo'], workers: List, timeout: int) -> tuple:
        """Execute tasks in pipeline fashion."""
        self._logger.info(f"Executing pipeline with {len(workers)} stages")

        results = []
        worker_stats = []

        # Process data through pipeline stages
        pipeline_data = tasks[0].data if tasks else {}

        for i, worker in enumerate(workers):
            stage_task = TaskInfo(
                task_id=str(uuid.uuid4()),
                task_name=f"pipeline_stage_{i}",
                data=pipeline_data,
                priority=1.0
            )

            try:
                result = await asyncio.wait_for(
                    self._execute_worker_task(worker, stage_task, timeout),
                    timeout=timeout // len(workers)
                )
                pipeline_data = result["data"]  # Pass to next stage
                worker_stats.append(result["stats"])
            except Exception as e:
                worker_stats.append({
                    "worker_id": i,
                    "status": WorkerStatus.FAILED,
                    "error": str(e),
                    "execution_time": 0
                })
                break

        results.append(pipeline_data)
        return results, worker_stats

    async def _execute_map_reduce(self, tasks: List['TaskInfo'], workers: List, timeout: int) -> tuple:
        """Execute map-reduce pattern."""
        self._logger.info("Executing map-reduce pattern")

        # Map phase - distribute tasks
        map_results, map_stats = await self._execute_parallel(
            tasks, workers[:len(tasks)], timeout // 2
        )

        # Reduce phase - aggregate results
        reduce_task = TaskInfo(
            task_id=str(uuid.uuid4()),
            task_name="reduce",
            data={"map_results": map_results},
            priority=1.0
        )

        reducer = workers[-1]  # Use last worker as reducer
        try:
            reduce_result = await asyncio.wait_for(
                self._execute_worker_task(reducer, reduce_task, timeout // 2),
                timeout=timeout // 2
            )
            final_result = [reduce_result["data"]]
            reduce_stats = [reduce_result["stats"]]
        except Exception as e:
            final_result = map_results
            reduce_stats = [{
                "worker_id": len(workers) - 1,
                "status": WorkerStatus.FAILED,
                "error": str(e),
                "execution_time": 0
            }]

        all_stats = map_stats + reduce_stats
        return final_result, all_stats

    async def _execute_scatter_gather(self, tasks: List['TaskInfo'], workers: List, timeout: int) -> tuple:
        """Execute scatter-gather pattern."""
        self._logger.info("Executing scatter-gather pattern")

        # Scatter phase - distribute
        scatter_results, scatter_stats = await self._execute_parallel(
            tasks, workers, timeout * 0.8
        )

        # Gather phase - collect and aggregate
        gathered_result = {
            "gathered_count": len(scatter_results),
            "gathered_data": scatter_results
        }

        return [gathered_result], scatter_stats

    async def _execute_consensus(
        self,
        tasks: List['TaskInfo'],
        workers: List,
        threshold: float,
        timeout: int
    ) -> tuple:
        """Execute consensus-based coordination."""
        self._logger.info(f"Executing consensus with threshold {threshold}")

        # Each worker processes same task
        task = tasks[0] if tasks else TaskInfo(
            task_id=str(uuid.uuid4()),
            task_name="consensus_task",
            data={},
            priority=1.0
        )

        # Get results from all workers
        async_tasks = []
        for worker in workers:
            async_task = asyncio.create_task(
                self._execute_worker_task(worker, task, timeout)
            )
            async_tasks.append(async_task)

        results = await asyncio.gather(*async_tasks, return_exceptions=True)

        # Analyze consensus
        worker_stats = []
        valid_results = []

        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                worker_stats.append(result["stats"])
                valid_results.append(result["data"])
            else:
                worker_stats.append({
                    "worker_id": i,
                    "status": WorkerStatus.FAILED,
                    "error": str(result),
                    "execution_time": 0
                })

        # Determine consensus
        consensus_ratio = len(valid_results) / len(workers)
        consensus_reached = consensus_ratio >= threshold

        consensus_result = {
            "consensus_reached": consensus_reached,
            "agreement_ratio": consensus_ratio,
            "results": valid_results
        }

        return [consensus_result], worker_stats

    def _assign_tasks_to_workers(self, tasks: List['TaskInfo'], workers: List) -> List[tuple]:
        """Assign tasks to workers based on load balancing."""
        assignments = []

        # Round-robin assignment
        for i, task in enumerate(tasks):
            worker = workers[i % len(workers)]
            assignments.append((worker, task))

        return assignments

    async def _execute_worker_task(self, worker: Any, task: 'TaskInfo', timeout: int) -> Dict:
        """Execute a single task on a worker."""
        start_time = datetime.now(timezone.utc)

        try:
            # Simulate worker execution (in production, would call actual worker)
            result = await worker.process(task.data)

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            return {
                "data": result,
                "stats": {
                    "worker_id": worker.worker_id,
                    "task_id": task.task_id,
                    "status": WorkerStatus.COMPLETED,
                    "execution_time": execution_time,
                    "error": None
                }
            }
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {
                "data": None,
                "stats": {
                    "worker_id": worker.worker_id,
                    "task_id": task.task_id,
                    "status": WorkerStatus.FAILED,
                    "execution_time": execution_time,
                    "error": str(e)
                }
            }

    async def _retry_failed_tasks(self, failed_tasks: List, workers: List, timeout: int) -> List:
        """Retry failed tasks."""
        retry_results = []

        for failed in failed_tasks:
            # Find available worker
            worker = workers[0]  # Simplified selection

            # Create retry task
            task = TaskInfo(
                task_id=failed.get("task_id", str(uuid.uuid4())),
                task_name="retry",
                data={},
                priority=2.0  # Higher priority for retries
            )

            try:
                result = await asyncio.wait_for(
                    self._execute_worker_task(worker, task, timeout // 2),
                    timeout=timeout // 2
                )
                if result["stats"]["status"] == WorkerStatus.COMPLETED:
                    retry_results.append(result["data"])
            except:
                pass  # Skip if retry also fails

        return retry_results

    def _calculate_performance_metrics(self, worker_stats: List, start_time: datetime) -> Dict:
        """Calculate performance metrics."""
        if not worker_stats:
            return {}

        execution_times = [s["execution_time"] for s in worker_stats if "execution_time" in s]

        return {
            "total_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
            "average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "min_execution_time": min(execution_times) if execution_times else 0,
            "max_execution_time": max(execution_times) if execution_times else 0,
            "success_rate": sum(1 for s in worker_stats if s["status"] == WorkerStatus.COMPLETED) / len(worker_stats)
        }

    def _calculate_parallel_efficiency(self, worker_stats: List, worker_count: int) -> float:
        """Calculate parallel processing efficiency."""
        if not worker_stats or worker_count == 0:
            return 0.0

        execution_times = [s["execution_time"] for s in worker_stats if "execution_time" in s]
        if not execution_times:
            return 0.0

        # Ideal time if perfectly parallel
        total_work = sum(execution_times)
        ideal_time = total_work / worker_count

        # Actual time (max of all workers)
        actual_time = max(execution_times)

        # Efficiency ratio
        efficiency = ideal_time / actual_time if actual_time > 0 else 0
        return min(1.0, efficiency)  # Cap at 1.0

    async def _cleanup_worker_pool(self, workers: List) -> None:
        """Cleanup worker pool."""
        for worker in workers:
            try:
                await worker.cleanup()
            except Exception as e:
                self._logger.error(f"Error cleaning up worker: {e}")

    def _calculate_provenance_hash(self, inputs: Dict, results: List, execution_id: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "agent": self.config.name,
            "version": self.config.version,
            "execution_id": execution_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": inputs.get("task"),
            "strategy": inputs.get("strategy"),
            "result_count": len(results)
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def _terminate_core(self) -> None:
        """Cleanup coordinator resources."""
        self._logger.info("Cleaning up CoordinatorAgent resources")
        self.active_tasks.clear()
        self.completed_tasks.clear()
        self.coordination_history.clear()

    async def _collect_custom_metrics(self) -> Dict[str, Any]:
        """Collect coordinator-specific metrics."""
        if not self.coordination_history:
            return {}

        recent = self.coordination_history[-50:]
        return {
            "total_coordinations": len(self.coordination_history),
            "strategies_used": list(set(c.strategy for c in recent)),
            "average_workers": sum(c.workers_deployed for c in recent) / len(recent),
            "average_efficiency": sum(c.parallel_efficiency for c in recent) / len(recent),
            "success_rate": sum(1 for c in recent if c.success) / len(recent),
            "total_tasks_processed": sum(len(c.results) for c in recent)
        }


class TaskInfo:
    """Information about a task."""

    def __init__(self, task_id: str, task_name: str, data: Dict, priority: float):
        """Initialize task info."""
        self.task_id = task_id
        self.task_name = task_name
        self.data = data
        self.priority = priority
        self.created_at = datetime.now(timezone.utc)
        self.status = WorkerStatus.IDLE


class MockWorker:
    """Mock worker for testing."""

    def __init__(self, config: AgentConfig, worker_id: int):
        """Initialize mock worker."""
        self.config = config
        self.worker_id = worker_id

    async def process(self, data: Dict) -> Dict:
        """Process data (simulated)."""
        # Simulate processing time
        await asyncio.sleep(0.1)

        # Return processed data
        return {
            "processed": True,
            "worker_id": self.worker_id,
            "input_size": len(str(data)),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def cleanup(self) -> None:
        """Cleanup worker."""
        pass


class PerformanceMonitor:
    """Monitor performance of coordination."""

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}

    def record(self, metric: str, value: float) -> None:
        """Record a metric."""
        if metric not in self.metrics:
            self.metrics[metric] = []
        self.metrics[metric].append(value)

    def get_average(self, metric: str) -> float:
        """Get average value for metric."""
        if metric in self.metrics and self.metrics[metric]:
            return sum(self.metrics[metric]) / len(self.metrics[metric])
        return 0.0