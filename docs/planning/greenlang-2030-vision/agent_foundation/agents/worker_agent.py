# -*- coding: utf-8 -*-
"""
WorkerAgent - Distributed task execution worker agent.

This module implements the WorkerAgent for executing specific tasks as part
of a distributed swarm, processing data chunks, and reporting results.

Example:
    >>> agent = WorkerAgent(config)
    >>> result = await agent.execute(WorkerInput(
    ...     task_type="calculate_emissions",
    ...     data=activity_data,
    ...     parameters={"method": "GHG_Protocol"}
    ... ))
"""

import asyncio
import hashlib
import logging
import random
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

import sys
import os
from greenlang.determinism import DeterministicClock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent, AgentConfig, ExecutionContext

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks workers can execute."""

    CALCULATION = "calculation"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    EXTRACTION = "extraction"
    AGGREGATION = "aggregation"
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    INTEGRATION = "integration"


class WorkerCapability(str, Enum):
    """Capabilities of worker agents."""

    DATA_PROCESSING = "data_processing"
    CALCULATION = "calculation"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    API_INTEGRATION = "api_integration"
    FILE_PROCESSING = "file_processing"
    REPORTING = "reporting"
    MACHINE_LEARNING = "machine_learning"


class WorkerInput(BaseModel):
    """Input data model for WorkerAgent."""

    task_type: TaskType = Field(..., description="Type of task to execute")
    task_id: Optional[str] = Field(None, description="Unique task identifier")
    data: Dict[str, Any] = Field(..., description="Data to process")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task parameters"
    )
    coordinator_id: Optional[str] = Field(None, description="Coordinator agent ID")
    batch_size: int = Field(100, ge=1, le=10000, description="Batch processing size")
    timeout_seconds: int = Field(60, ge=1, le=3600, description="Task timeout")
    priority: int = Field(5, ge=1, le=10, description="Task priority (1=highest)")
    checkpoint_enabled: bool = Field(False, description="Enable task checkpointing")

    @validator('batch_size')
    def validate_batch_size(cls, v, values):
        """Validate batch size is appropriate for task."""
        task_type = values.get('task_type')
        if task_type == TaskType.CALCULATION and v > 1000:
            raise ValueError("Calculation tasks limited to batch size 1000")
        return v


class WorkerOutput(BaseModel):
    """Output data model for WorkerAgent."""

    success: bool = Field(..., description="Task execution success")
    task_type: TaskType = Field(..., description="Type of task executed")
    task_id: Optional[str] = Field(None, description="Task identifier")
    result: Dict[str, Any] = Field(..., description="Processing result")
    records_processed: int = Field(0, ge=0, description="Number of records processed")
    errors: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Processing errors"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Processing warnings"
    )
    performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics"
    )
    checkpoints: List[str] = Field(
        default_factory=list,
        description="Checkpoint IDs created"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., description="Processing duration")
    resource_usage: Dict[str, float] = Field(
        default_factory=dict,
        description="Resource usage metrics"
    )


class WorkerAgent(BaseAgent):
    """
    WorkerAgent implementation for distributed task execution.

    This agent executes specific tasks as part of a swarm, processing data
    chunks efficiently with checkpointing and error handling.

    Attributes:
        config: Agent configuration
        capabilities: Worker capabilities
        task_registry: Registry of executable tasks
        checkpoint_manager: Checkpoint management

    Example:
        >>> config = AgentConfig(name="worker_01", version="1.0.0")
        >>> agent = WorkerAgent(config)
        >>> await agent.initialize()
        >>> result = await agent.execute(worker_input)
        >>> print(f"Records processed: {result.result.records_processed}")
    """

    def __init__(self, config: AgentConfig):
        """Initialize WorkerAgent."""
        super().__init__(config)
        self.capabilities: Set[WorkerCapability] = set()
        self.task_registry: Dict[TaskType, callable] = {}
        self.checkpoint_manager = CheckpointManager()
        self.resource_monitor = ResourceMonitor()
        self.task_history: List[WorkerOutput] = []

    async def _initialize_core(self) -> None:
        """Initialize worker resources."""
        self._logger.info("Initializing WorkerAgent resources")

        # Register capabilities
        self._register_capabilities()

        # Register task handlers
        self._register_task_handlers()

        # Initialize resource monitoring
        self.resource_monitor.start()

        self._logger.info(f"Worker initialized with {len(self.capabilities)} capabilities")

    def _register_capabilities(self) -> None:
        """Register worker capabilities."""
        self.capabilities = {
            WorkerCapability.DATA_PROCESSING,
            WorkerCapability.CALCULATION,
            WorkerCapability.VALIDATION,
            WorkerCapability.TRANSFORMATION,
            WorkerCapability.REPORTING
        }

    def _register_task_handlers(self) -> None:
        """Register handlers for different task types."""
        self.task_registry = {
            TaskType.CALCULATION: self._handle_calculation,
            TaskType.VALIDATION: self._handle_validation,
            TaskType.TRANSFORMATION: self._handle_transformation,
            TaskType.EXTRACTION: self._handle_extraction,
            TaskType.AGGREGATION: self._handle_aggregation,
            TaskType.ANALYSIS: self._handle_analysis,
            TaskType.REPORTING: self._handle_reporting,
            TaskType.INTEGRATION: self._handle_integration
        }

    async def _execute_core(self, input_data: WorkerInput, context: ExecutionContext) -> WorkerOutput:
        """
        Core execution logic for worker tasks.

        This method processes tasks with checkpointing and resource monitoring.
        """
        start_time = datetime.now(timezone.utc)
        errors = []
        warnings = []
        checkpoints = []

        try:
            # Step 1: Validate task type is supported
            if input_data.task_type not in self.task_registry:
                raise ValueError(f"Unsupported task type: {input_data.task_type}")

            # Step 2: Check resource availability
            if not self.resource_monitor.check_availability():
                warnings.append("Resource constraints detected, performance may be impacted")

            # Step 3: Create initial checkpoint if enabled
            if input_data.checkpoint_enabled:
                checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                    input_data.task_id or context.execution_id,
                    input_data.data
                )
                checkpoints.append(checkpoint_id)

            # Step 4: Get task handler
            handler = self.task_registry[input_data.task_type]

            # Step 5: Process data in batches
            self._logger.info(f"Processing task {input_data.task_type} with batch size {input_data.batch_size}")

            result = await asyncio.wait_for(
                self._process_with_batching(
                    handler,
                    input_data.data,
                    input_data.parameters,
                    input_data.batch_size,
                    checkpoints if input_data.checkpoint_enabled else None
                ),
                timeout=input_data.timeout_seconds
            )

            # Step 6: Validate results
            validation_result = self._validate_result(result, input_data.task_type)
            if not validation_result["valid"]:
                warnings.extend(validation_result["warnings"])

            # Step 7: Calculate performance metrics
            performance = self._calculate_performance(
                start_time,
                result.get("records_processed", 0)
            )

            # Step 8: Get resource usage
            resource_usage = self.resource_monitor.get_usage()

            # Step 9: Generate provenance hash
            provenance_hash = self._calculate_provenance_hash(
                input_data.dict(),
                result,
                context.execution_id
            )

            # Step 10: Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Step 11: Create output
            output = WorkerOutput(
                success=True,
                task_type=input_data.task_type,
                task_id=input_data.task_id,
                result=result,
                records_processed=result.get("records_processed", 0),
                errors=errors,
                warnings=warnings,
                performance=performance,
                checkpoints=checkpoints,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time,
                resource_usage=resource_usage
            )

            # Store in history
            self.task_history.append(output)
            if len(self.task_history) > 100:
                self.task_history.pop(0)

            # Report to coordinator if specified
            if input_data.coordinator_id:
                await self._report_to_coordinator(
                    input_data.coordinator_id,
                    output
                )

            return output

        except asyncio.TimeoutError:
            self._logger.error(f"Task timeout after {input_data.timeout_seconds}s")
            errors.append(f"Task timeout: {input_data.timeout_seconds}s exceeded")

        except Exception as e:
            self._logger.error(f"Task execution failed: {str(e)}", exc_info=True)
            errors.append(str(e))

        # Create error output
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return WorkerOutput(
            success=False,
            task_type=input_data.task_type,
            task_id=input_data.task_id,
            result={},
            records_processed=0,
            errors=errors,
            warnings=warnings,
            performance={},
            checkpoints=checkpoints,
            provenance_hash=self._calculate_provenance_hash(
                input_data.dict(),
                {},
                context.execution_id
            ),
            processing_time_ms=processing_time,
            resource_usage=self.resource_monitor.get_usage()
        )

    async def _process_with_batching(
        self,
        handler: callable,
        data: Dict,
        parameters: Dict,
        batch_size: int,
        checkpoints: Optional[List[str]]
    ) -> Dict:
        """Process data in batches with optional checkpointing."""
        # Check if data contains list for batching
        batch_key = self._find_batch_key(data)

        if batch_key and isinstance(data[batch_key], list):
            # Process in batches
            items = data[batch_key]
            total_items = len(items)
            processed_items = []

            for i in range(0, total_items, batch_size):
                batch = items[i:i + batch_size]
                batch_data = {**data, batch_key: batch}

                # Process batch
                batch_result = await handler(batch_data, parameters)

                if "processed" in batch_result:
                    processed_items.extend(batch_result["processed"])

                # Create checkpoint after each batch
                if checkpoints is not None and i + batch_size < total_items:
                    checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                        f"batch_{i}",
                        {"processed": i + batch_size, "total": total_items}
                    )
                    checkpoints.append(checkpoint_id)

            return {
                "processed": processed_items,
                "records_processed": len(processed_items),
                "batches_processed": (total_items + batch_size - 1) // batch_size
            }
        else:
            # Process as single item
            return await handler(data, parameters)

    def _find_batch_key(self, data: Dict) -> Optional[str]:
        """Find the key containing list data for batching."""
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                return key
        return None

    # Task handlers

    async def _handle_calculation(self, data: Dict, parameters: Dict) -> Dict:
        """Handle calculation tasks."""
        self._logger.debug("Executing calculation task")

        # Perform calculations (example)
        results = {}

        if "values" in data:
            values = data["values"]
            if isinstance(values, list):
                results["sum"] = sum(values)
                results["average"] = results["sum"] / len(values) if values else 0
                results["min"] = min(values) if values else 0
                results["max"] = max(values) if values else 0

        # Apply method from parameters
        method = parameters.get("method", "default")
        if method == "GHG_Protocol" and "activity_data" in data:
            # Simulate GHG calculation
            activity = data.get("activity_data", 0)
            factor = parameters.get("emission_factor", 2.5)
            results["emissions"] = activity * factor

        return {
            "calculated": True,
            "results": results,
            "records_processed": len(data.get("values", []))
        }

    async def _handle_validation(self, data: Dict, parameters: Dict) -> Dict:
        """Handle validation tasks."""
        self._logger.debug("Executing validation task")

        validation_errors = []
        validated_records = []

        # Validate data based on rules
        rules = parameters.get("rules", {})

        for key, value in data.items():
            if key in rules:
                rule = rules[key]
                if "min" in rule and value < rule["min"]:
                    validation_errors.append(f"{key} below minimum: {value} < {rule['min']}")
                if "max" in rule and value > rule["max"]:
                    validation_errors.append(f"{key} above maximum: {value} > {rule['max']}")
                if "required" in rule and rule["required"] and not value:
                    validation_errors.append(f"{key} is required")

            if not validation_errors:
                validated_records.append({key: value})

        return {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "validated_records": validated_records,
            "records_processed": len(data)
        }

    async def _handle_transformation(self, data: Dict, parameters: Dict) -> Dict:
        """Handle data transformation tasks."""
        self._logger.debug("Executing transformation task")

        transformed = {}
        mapping = parameters.get("mapping", {})

        # Apply transformations
        for key, value in data.items():
            if key in mapping:
                new_key = mapping[key]
                transformed[new_key] = value
            else:
                transformed[key] = value

        # Apply formatting
        if "format" in parameters:
            format_type = parameters["format"]
            if format_type == "uppercase":
                transformed = {k: v.upper() if isinstance(v, str) else v
                             for k, v in transformed.items()}

        return {
            "transformed": transformed,
            "records_processed": len(data)
        }

    async def _handle_extraction(self, data: Dict, parameters: Dict) -> Dict:
        """Handle data extraction tasks."""
        self._logger.debug("Executing extraction task")

        extracted = []
        fields = parameters.get("fields", [])

        # Extract specified fields
        if fields:
            for item in data.get("items", [data]):
                extracted_item = {field: item.get(field) for field in fields if field in item}
                extracted.append(extracted_item)
        else:
            extracted = data.get("items", [data])

        return {
            "extracted": extracted,
            "records_processed": len(extracted)
        }

    async def _handle_aggregation(self, data: Dict, parameters: Dict) -> Dict:
        """Handle aggregation tasks."""
        self._logger.debug("Executing aggregation task")

        aggregated = {}
        group_by = parameters.get("group_by")
        aggregate_field = parameters.get("field")
        operation = parameters.get("operation", "sum")

        if "items" in data and isinstance(data["items"], list):
            # Group and aggregate
            groups = {}
            for item in data["items"]:
                if group_by and group_by in item:
                    key = item[group_by]
                    if key not in groups:
                        groups[key] = []
                    if aggregate_field and aggregate_field in item:
                        groups[key].append(item[aggregate_field])

            # Apply aggregation operation
            for key, values in groups.items():
                if operation == "sum":
                    aggregated[key] = sum(values)
                elif operation == "avg":
                    aggregated[key] = sum(values) / len(values) if values else 0
                elif operation == "count":
                    aggregated[key] = len(values)
                elif operation == "min":
                    aggregated[key] = min(values) if values else 0
                elif operation == "max":
                    aggregated[key] = max(values) if values else 0

        return {
            "aggregated": aggregated,
            "groups": len(aggregated),
            "records_processed": len(data.get("items", []))
        }

    async def _handle_analysis(self, data: Dict, parameters: Dict) -> Dict:
        """Handle analysis tasks."""
        self._logger.debug("Executing analysis task")

        analysis = {
            "summary": {},
            "insights": [],
            "metrics": {}
        }

        # Perform basic analysis
        if "values" in data:
            values = data["values"]
            if isinstance(values, list) and values:
                analysis["summary"] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "median": sorted(values)[len(values) // 2]
                }

                # Generate insights
                if analysis["summary"]["mean"] > 100:
                    analysis["insights"].append("High average value detected")
                if max(values) / min(values) > 10:
                    analysis["insights"].append("Large value range detected")

        return {
            "analysis": analysis,
            "records_processed": len(data.get("values", []))
        }

    async def _handle_reporting(self, data: Dict, parameters: Dict) -> Dict:
        """Handle reporting tasks."""
        self._logger.debug("Executing reporting task")

        report = {
            "title": parameters.get("title", "Worker Report"),
            "sections": [],
            "data": data,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

        # Generate report sections
        for key, value in data.items():
            section = {
                "name": key,
                "content": str(value),
                "type": type(value).__name__
            }
            report["sections"].append(section)

        return {
            "report": report,
            "records_processed": len(data)
        }

    async def _handle_integration(self, data: Dict, parameters: Dict) -> Dict:
        """Handle integration tasks."""
        self._logger.debug("Executing integration task")

        # Simulate API integration
        endpoint = parameters.get("endpoint", "unknown")
        method = parameters.get("method", "GET")

        integration_result = {
            "endpoint": endpoint,
            "method": method,
            "status": "simulated",
            "response": {"message": "Integration simulated"},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return {
            "integration": integration_result,
            "records_processed": 1
        }

    def _validate_result(self, result: Dict, task_type: TaskType) -> Dict:
        """Validate task execution result."""
        validation = {
            "valid": True,
            "warnings": []
        }

        # Check for required fields based on task type
        if task_type == TaskType.CALCULATION:
            if "results" not in result:
                validation["warnings"].append("Missing calculation results")

        elif task_type == TaskType.VALIDATION:
            if "valid" not in result:
                validation["warnings"].append("Missing validation status")

        # Check for empty results
        if not result or (isinstance(result, dict) and len(result) == 0):
            validation["warnings"].append("Empty result returned")
            validation["valid"] = False

        return validation

    def _calculate_performance(self, start_time: datetime, records: int) -> Dict:
        """Calculate performance metrics."""
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        return {
            "elapsed_seconds": elapsed,
            "records_per_second": records / elapsed if elapsed > 0 else 0,
            "throughput": "high" if records / max(1, elapsed) > 100 else "normal"
        }

    async def _report_to_coordinator(self, coordinator_id: str, output: WorkerOutput) -> None:
        """Report results to coordinator."""
        try:
            # In production, would send actual message to coordinator
            self._logger.info(f"Reporting to coordinator {coordinator_id}")
            # Simulate reporting delay
            await asyncio.sleep(0.01)
        except Exception as e:
            self._logger.error(f"Failed to report to coordinator: {e}")

    def _calculate_provenance_hash(self, inputs: Dict, result: Dict, execution_id: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "agent": self.config.name,
            "version": self.config.version,
            "execution_id": execution_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_type": inputs.get("task_type"),
            "result_hash": hashlib.md5(str(result).encode()).hexdigest() if result else None
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def _terminate_core(self) -> None:
        """Cleanup worker resources."""
        self._logger.info("Cleaning up WorkerAgent resources")
        self.resource_monitor.stop()
        self.task_registry.clear()
        self.task_history.clear()

    async def _collect_custom_metrics(self) -> Dict[str, Any]:
        """Collect worker-specific metrics."""
        if not self.task_history:
            return {}

        recent = self.task_history[-100:]
        return {
            "total_tasks": len(self.task_history),
            "task_types": list(set(t.task_type for t in recent)),
            "success_rate": sum(1 for t in recent if t.success) / len(recent),
            "average_records": sum(t.records_processed for t in recent) / len(recent),
            "capabilities": list(self.capabilities),
            "checkpoint_count": sum(len(t.checkpoints) for t in recent)
        }


class CheckpointManager:
    """Manage task checkpoints."""

    def __init__(self):
        """Initialize checkpoint manager."""
        self.checkpoints = {}

    async def create_checkpoint(self, task_id: str, data: Dict) -> str:
        """Create a checkpoint."""
        checkpoint_id = f"checkpoint_{task_id}_{DeterministicClock.now().timestamp()}"
        self.checkpoints[checkpoint_id] = {
            "task_id": task_id,
            "data": data,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        return checkpoint_id

    async def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """Restore from checkpoint."""
        return self.checkpoints.get(checkpoint_id)


class ResourceMonitor:
    """Monitor worker resource usage."""

    def __init__(self):
        """Initialize resource monitor."""
        self.monitoring = False
        self.usage = {}

    def start(self) -> None:
        """Start monitoring."""
        self.monitoring = True
        self.usage = {
            "cpu_percent": random.uniform(10, 50),  # Simulated
            "memory_mb": random.uniform(100, 500),  # Simulated
            "threads": 5
        }

    def stop(self) -> None:
        """Stop monitoring."""
        self.monitoring = False

    def check_availability(self) -> bool:
        """Check if resources are available."""
        # Simple check - in production would check actual resources
        return self.usage.get("cpu_percent", 0) < 80

    def get_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        if self.monitoring:
            # Update with simulated values
            self.usage["cpu_percent"] = random.uniform(10, 50)
            self.usage["memory_mb"] = random.uniform(100, 500)
        return self.usage