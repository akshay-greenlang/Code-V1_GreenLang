"""
Advanced Pipeline Executor for GreenLang Runtime

This module provides a production-ready executor with sophisticated features including:
- Conditional execution evaluation
- Reference resolution
- Retry logic with deterministic backoff
- Comprehensive error handling
- Step isolation and cleanup
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    from ..sdk.pipeline_spec import PipelineSpec, StepSpec
except ImportError:
    # Fallback for direct imports
    from greenlang.sdk.pipeline_spec import PipelineSpec, StepSpec

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Execution context that tracks state and provides step isolation."""

    # Core identification
    run_id: str
    pipeline_name: str

    # Runtime state
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)

    # Execution metadata
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_step: Optional[str] = None
    completed_steps: Set[str] = field(default_factory=set)
    failed_steps: Set[str] = field(default_factory=set)
    skipped_steps: Set[str] = field(default_factory=set)

    # Configuration
    artifacts_dir: Path = field(default_factory=lambda: Path("out"))
    deterministic: bool = False
    max_retries: int = 3
    base_backoff_seconds: float = 1.0

    # Resource tracking
    temp_dirs: List[Path] = field(default_factory=list)
    open_files: List[Any] = field(default_factory=list)


class ExecutionError(Exception):
    """Base exception for execution errors."""

    def __init__(self, message: str, step_name: Optional[str] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.step_name = step_name
        self.original_error = original_error


class ConditionalExecutionError(ExecutionError):
    """Raised when conditional expression evaluation fails."""
    pass


class ReferenceResolutionError(ExecutionError):
    """Raised when reference resolution fails."""
    pass


class StepExecutionError(ExecutionError):
    """Raised when step execution fails."""
    pass


class PipelineExecutor:
    """
    Production-ready pipeline executor with advanced features.

    Features:
    - Conditional execution with expression evaluation
    - Reference resolution for dynamic inputs
    - Deterministic retry with exponential backoff
    - Step isolation and resource cleanup
    - Comprehensive error handling and recovery
    """

    def __init__(
        self,
        max_parallel_steps: int = 4,
        default_timeout: float = 300.0,
        cleanup_on_failure: bool = True
    ):
        """
        Initialize the executor.

        Args:
            max_parallel_steps: Maximum concurrent step execution
            default_timeout: Default step timeout in seconds
            cleanup_on_failure: Whether to clean up resources on failure
        """
        self.max_parallel_steps = max_parallel_steps
        self.default_timeout = default_timeout
        self.cleanup_on_failure = cleanup_on_failure
        self._thread_pool = ThreadPoolExecutor(max_workers=max_parallel_steps)

        logger.info(f"PipelineExecutor initialized with {max_parallel_steps} max parallel steps")

    def execute_pipeline(
        self,
        spec: PipelineSpec,
        context: ExecutionContext,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a complete pipeline.

        Args:
            spec: Pipeline specification
            context: Execution context
            dry_run: If True, validate but don't execute

        Returns:
            Execution results dictionary

        Raises:
            ExecutionError: If pipeline execution fails
        """
        logger.info(f"Starting pipeline execution: {spec.name} (run_id: {context.run_id})")

        if dry_run:
            logger.info("Dry run mode - validating pipeline without execution")
            return self._validate_pipeline(spec, context)

        start_time = time.time()
        results = {
            "run_id": context.run_id,
            "pipeline_name": spec.name,
            "started_at": context.started_at.isoformat(),
            "status": "running",
            "steps": {},
            "summary": {
                "total_steps": len(spec.steps),
                "completed": 0,
                "failed": 0,
                "skipped": 0
            }
        }

        try:
            # Setup artifacts directory
            context.artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Execute steps in order or parallel as specified
            parallel_steps = []
            sequential_steps = []

            for step in spec.steps:
                if step.parallel:
                    parallel_steps.append(step)
                else:
                    # Execute any pending parallel steps first
                    if parallel_steps:
                        self._execute_parallel_steps(parallel_steps, context, results)
                        parallel_steps = []

                    # Execute sequential step
                    sequential_steps.append(step)
                    self._execute_sequential_steps(sequential_steps, context, results)
                    sequential_steps = []

            # Execute any remaining parallel steps
            if parallel_steps:
                self._execute_parallel_steps(parallel_steps, context, results)

            # Execute any remaining sequential steps
            if sequential_steps:
                self._execute_sequential_steps(sequential_steps, context, results)

            # Finalize results
            execution_time = time.time() - start_time
            results.update({
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "execution_time_seconds": execution_time,
                "outputs": context.outputs.copy(),
                "summary": {
                    "total_steps": len(spec.steps),
                    "completed": len(context.completed_steps),
                    "failed": len(context.failed_steps),
                    "skipped": len(context.skipped_steps)
                }
            })

            logger.info(f"Pipeline execution completed in {execution_time:.2f}s")
            return results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            results.update({
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "execution_time_seconds": time.time() - start_time
            })

            if self.cleanup_on_failure:
                self._cleanup_context(context)

            raise ExecutionError(f"Pipeline execution failed: {e}") from e

        finally:
            self._cleanup_context(context)

    def _execute_sequential_steps(
        self,
        steps: List[StepSpec],
        context: ExecutionContext,
        results: Dict[str, Any]
    ) -> None:
        """Execute steps sequentially."""
        for step in steps:
            if self._should_skip_step(step, context):
                context.skipped_steps.add(step.name)
                results["steps"][step.name] = {
                    "status": "skipped",
                    "reason": "condition not met"
                }
                logger.info(f"Step '{step.name}' skipped due to condition")
                continue

            try:
                step_result = self.execute_step(step, context)
                results["steps"][step.name] = step_result
                context.completed_steps.add(step.name)
                logger.info(f"Step '{step.name}' completed successfully")

            except Exception as e:
                context.failed_steps.add(step.name)
                results["steps"][step.name] = {
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.now(timezone.utc).isoformat()
                }
                logger.error(f"Step '{step.name}' failed: {e}")

                # Handle error policy
                if not self._handle_step_error(step, e, context):
                    raise StepExecutionError(f"Step '{step.name}' failed and pipeline stopped") from e

    def _execute_parallel_steps(
        self,
        steps: List[StepSpec],
        context: ExecutionContext,
        results: Dict[str, Any]
    ) -> None:
        """Execute steps in parallel."""
        futures = {}

        for step in steps:
            if self._should_skip_step(step, context):
                context.skipped_steps.add(step.name)
                results["steps"][step.name] = {
                    "status": "skipped",
                    "reason": "condition not met"
                }
                continue

            future = self._thread_pool.submit(self.execute_step, step, context)
            futures[future] = step

        # Wait for completion
        for future in as_completed(futures):
            step = futures[future]
            try:
                step_result = future.result()
                results["steps"][step.name] = step_result
                context.completed_steps.add(step.name)
                logger.info(f"Parallel step '{step.name}' completed successfully")

            except Exception as e:
                context.failed_steps.add(step.name)
                results["steps"][step.name] = {
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.now(timezone.utc).isoformat()
                }
                logger.error(f"Parallel step '{step.name}' failed: {e}")

                if not self._handle_step_error(step, e, context):
                    # Cancel remaining futures
                    for remaining_future in futures:
                        if not remaining_future.done():
                            remaining_future.cancel()
                    raise StepExecutionError(f"Parallel step '{step.name}' failed and pipeline stopped") from e

    def execute_step(
        self,
        step: StepSpec,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute a single pipeline step with retry logic and error handling.

        Args:
            step: Step specification
            context: Execution context

        Returns:
            Step execution results

        Raises:
            StepExecutionError: If step execution fails after retries
        """
        logger.debug(f"Executing step: {step.name}")
        context.current_step = step.name

        start_time = time.time()
        step_result = {
            "name": step.name,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "agent": step.agent,
            "action": step.action,
            "attempts": []
        }

        # Resolve inputs
        try:
            resolved_inputs = self._resolve_step_inputs(step, context)
        except Exception as e:
            raise StepExecutionError(f"Failed to resolve inputs for step '{step.name}': {e}") from e

        # Determine retry configuration
        max_retries = self._get_max_retries(step, context)
        backoff_seconds = self._get_backoff_seconds(step, context)

        # Execute with retry logic
        last_error = None
        for attempt in range(max_retries + 1):
            attempt_start = time.time()
            attempt_result = {
                "attempt": attempt + 1,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "running"
            }

            try:
                # Create isolated execution environment
                with self._create_step_environment(step, context) as step_env:
                    # Execute the actual step
                    outputs = self._execute_step_action(
                        step,
                        resolved_inputs,
                        step_env,
                        timeout=step.timeout or self.default_timeout
                    )

                    # Store results
                    context.step_results[step.name] = outputs
                    if step.outputs:
                        # Map outputs according to step specification
                        mapped_outputs = self._map_step_outputs(outputs, step.outputs)
                        context.outputs.update(mapped_outputs)

                    # Success
                    attempt_time = time.time() - attempt_start
                    attempt_result.update({
                        "status": "success",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "execution_time_seconds": attempt_time,
                        "outputs": outputs
                    })
                    step_result["attempts"].append(attempt_result)

                    # Update final step result
                    total_time = time.time() - start_time
                    step_result.update({
                        "status": "success",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "execution_time_seconds": total_time,
                        "outputs": outputs,
                        "total_attempts": attempt + 1
                    })

                    logger.debug(f"Step '{step.name}' completed in {total_time:.2f}s after {attempt + 1} attempts")
                    return step_result

            except Exception as e:
                last_error = e
                attempt_time = time.time() - attempt_start
                attempt_result.update({
                    "status": "failed",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "execution_time_seconds": attempt_time,
                    "error": str(e)
                })
                step_result["attempts"].append(attempt_result)

                logger.warning(f"Step '{step.name}' attempt {attempt + 1} failed: {e}")

                # Don't retry on the last attempt
                if attempt < max_retries:
                    # Calculate backoff with deterministic jitter
                    backoff_time = self._calculate_backoff(
                        attempt,
                        backoff_seconds,
                        context.deterministic,
                        step.name
                    )

                    logger.info(f"Retrying step '{step.name}' in {backoff_time:.2f}s (attempt {attempt + 2}/{max_retries + 1})")
                    time.sleep(backoff_time)

        # All retries exhausted
        total_time = time.time() - start_time
        step_result.update({
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "execution_time_seconds": total_time,
            "error": str(last_error),
            "total_attempts": max_retries + 1
        })

        raise StepExecutionError(
            f"Step '{step.name}' failed after {max_retries + 1} attempts: {last_error}",
            step_name=step.name,
            original_error=last_error
        )

    def _eval_when(self, condition: str, context: ExecutionContext) -> bool:
        """
        Evaluate conditional expression for step execution.

        Args:
            condition: Conditional expression string
            context: Execution context with variables and step results

        Returns:
            Boolean result of condition evaluation

        Raises:
            ConditionalExecutionError: If evaluation fails
        """
        if not condition:
            return True

        try:
            # Create evaluation context
            eval_context = {
                "steps": context.step_results.copy(),
                "vars": context.variables.copy(),
                "inputs": context.inputs.copy(),
                "env": dict(os.environ),
                # Add utility functions
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "isinstance": isinstance,
            }

            # Sanitize condition - only allow safe operations
            if not self._is_safe_expression(condition):
                raise ConditionalExecutionError(f"Unsafe expression detected: {condition}")

            # Evaluate condition
            result = eval(condition, {"__builtins__": {}}, eval_context)

            if not isinstance(result, bool):
                # Try to convert to boolean
                result = bool(result)

            logger.debug(f"Condition '{condition}' evaluated to {result}")
            return result

        except Exception as e:
            raise ConditionalExecutionError(f"Failed to evaluate condition '{condition}': {e}") from e

    def _is_safe_expression(self, expression: str) -> bool:
        """Check if expression is safe for evaluation."""
        # Block dangerous keywords and patterns
        dangerous_patterns = [
            r'\b(import|exec|eval|compile|open|file|input|raw_input)\b',
            r'__[a-zA-Z_]+__',  # Dunder methods
            r'\bgetattr\b',
            r'\bsetattr\b',
            r'\bdelattr\b',
            r'\bglobals\b',
            r'\blocals\b',
            r'\bvars\b',
            r'\bdir\b',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return False

        return True

    def _resolve_refs(self, data: Any, context: ExecutionContext) -> Any:
        """
        Resolve references in data structures.

        Args:
            data: Data to resolve references in
            context: Execution context

        Returns:
            Data with references resolved

        Raises:
            ReferenceResolutionError: If reference resolution fails
        """
        if isinstance(data, str):
            return self._resolve_string_refs(data, context)
        elif isinstance(data, dict):
            return {key: self._resolve_refs(value, context) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._resolve_refs(item, context) for item in data]
        else:
            return data

    def _resolve_string_refs(self, text: str, context: ExecutionContext) -> Any:
        """Resolve string references like $steps.step_name.output."""
        if not text.startswith('$'):
            return text

        try:
            # Parse reference
            if text.startswith('$steps.'):
                # Step output reference
                parts = text[7:].split('.')  # Remove '$steps.'
                if not parts:
                    raise ReferenceResolutionError(f"Invalid step reference: {text}")

                step_name = parts[0]
                if step_name not in context.step_results:
                    raise ReferenceResolutionError(f"Step '{step_name}' not found or not executed yet")

                result = context.step_results[step_name]

                # Navigate nested path
                for part in parts[1:]:
                    if isinstance(result, dict) and part in result:
                        result = result[part]
                    else:
                        raise ReferenceResolutionError(f"Path not found in step result: {text}")

                return result

            elif text.startswith('$vars.'):
                # Variable reference
                var_name = text[6:]  # Remove '$vars.'
                if var_name not in context.variables:
                    raise ReferenceResolutionError(f"Variable '{var_name}' not found")
                return context.variables[var_name]

            elif text.startswith('$inputs.'):
                # Input reference
                input_name = text[8:]  # Remove '$inputs.'
                if input_name not in context.inputs:
                    raise ReferenceResolutionError(f"Input '{input_name}' not found")
                return context.inputs[input_name]

            elif text.startswith('$env.'):
                # Environment variable reference
                env_name = text[5:]  # Remove '$env.'
                return os.environ.get(env_name, '')

            else:
                raise ReferenceResolutionError(f"Unknown reference type: {text}")

        except Exception as e:
            if isinstance(e, ReferenceResolutionError):
                raise
            raise ReferenceResolutionError(f"Failed to resolve reference '{text}': {e}") from e

    def _calculate_backoff(
        self,
        attempt: int,
        base_seconds: float,
        deterministic: bool,
        step_name: str
    ) -> float:
        """
        Calculate backoff delay with deterministic or random jitter.

        Args:
            attempt: Attempt number (0-based)
            base_seconds: Base backoff time
            deterministic: Whether to use deterministic jitter
            step_name: Step name for deterministic seed

        Returns:
            Backoff time in seconds
        """
        # Exponential backoff: base * 2^attempt
        backoff = base_seconds * (2 ** attempt)

        # Add jitter to prevent thundering herd
        if deterministic:
            # Use step name and attempt for deterministic jitter
            seed_str = f"{step_name}-{attempt}"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
            # Use seed to generate consistent pseudo-random jitter
            jitter_factor = (seed % 1000) / 1000.0  # 0.0 to 0.999
            jitter = backoff * 0.1 * jitter_factor  # 0-10% jitter
        else:
            # Random jitter
            jitter = backoff * 0.1 * random.random()  # 0-10% jitter

        total_backoff = backoff + jitter

        # Cap at reasonable maximum
        return min(total_backoff, 60.0)  # Max 60 seconds

    def _should_skip_step(self, step: StepSpec, context: ExecutionContext) -> bool:
        """Check if step should be skipped based on condition."""
        if not step.condition:
            return False

        try:
            return not self._eval_when(step.condition, context)
        except ConditionalExecutionError:
            # If condition evaluation fails, don't skip (safer)
            logger.warning(f"Condition evaluation failed for step '{step.name}', executing anyway")
            return False

    def _resolve_step_inputs(self, step: StepSpec, context: ExecutionContext) -> Dict[str, Any]:
        """Resolve step inputs, handling references."""
        if step.inputsRef:
            # Use reference to get inputs
            return self._resolve_refs(step.inputsRef, context)
        elif step.inputs:
            # Resolve references in inputs
            return self._resolve_refs(step.inputs, context)
        else:
            return {}

    def _get_max_retries(self, step: StepSpec, context: ExecutionContext) -> int:
        """Get maximum retries for step."""
        if hasattr(step.on_error, 'retry') and step.on_error.retry:
            return step.on_error.retry.max
        return context.max_retries

    def _get_backoff_seconds(self, step: StepSpec, context: ExecutionContext) -> float:
        """Get backoff seconds for step."""
        if hasattr(step.on_error, 'retry') and step.on_error.retry:
            return step.on_error.retry.backoff_seconds
        return context.base_backoff_seconds

    def _handle_step_error(self, step: StepSpec, error: Exception, context: ExecutionContext) -> bool:
        """
        Handle step error according to error policy.

        Returns:
            True if pipeline should continue, False if it should stop
        """
        error_policy = step.on_error

        if hasattr(error_policy, 'policy'):
            policy = error_policy.policy
        else:
            policy = error_policy

        if policy == "continue":
            logger.info(f"Step '{step.name}' failed but pipeline continues due to error policy")
            return True
        elif policy == "skip":
            logger.info(f"Step '{step.name}' failed, marking as skipped")
            context.skipped_steps.add(step.name)
            return True
        elif policy == "stop":
            logger.error(f"Step '{step.name}' failed, stopping pipeline")
            return False
        elif policy == "fail":
            logger.error(f"Step '{step.name}' failed, failing pipeline")
            return False
        else:
            # Default to stop
            return False

    @contextmanager
    def _create_step_environment(self, step: StepSpec, context: ExecutionContext):
        """Create isolated environment for step execution."""
        step_dir = context.artifacts_dir / step.name
        step_dir.mkdir(parents=True, exist_ok=True)

        env = {
            "step_name": step.name,
            "step_dir": str(step_dir),
            "artifacts_dir": str(context.artifacts_dir),
            "run_id": context.run_id
        }

        try:
            yield env
        finally:
            # Cleanup if needed
            pass

    def _execute_step_action(
        self,
        step: StepSpec,
        inputs: Dict[str, Any],
        environment: Dict[str, Any],
        timeout: float
    ) -> Dict[str, Any]:
        """Execute the actual step action (placeholder implementation)."""
        # This is where you would integrate with the actual agent execution system
        # For now, return a mock result

        logger.debug(f"Executing step action: {step.agent}.{step.action}")

        # Simulate execution time
        execution_time = random.uniform(0.1, 2.0) if not environment.get("deterministic") else 1.0
        time.sleep(execution_time)

        # Mock result
        return {
            "status": "success",
            "message": f"Step {step.name} executed successfully",
            "inputs_received": inputs,
            "execution_time": execution_time
        }

    def _map_step_outputs(self, outputs: Dict[str, Any], output_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Map step outputs according to output specification."""
        # Simple mapping for now
        mapped = {}
        for key, mapping in output_spec.items():
            if isinstance(mapping, str) and mapping in outputs:
                mapped[key] = outputs[mapping]
            else:
                mapped[key] = mapping
        return mapped

    def _validate_pipeline(self, spec: PipelineSpec, context: ExecutionContext) -> Dict[str, Any]:
        """Validate pipeline without executing it."""
        validation_results = {
            "status": "validated",
            "pipeline_name": spec.name,
            "validation_time": datetime.now(timezone.utc).isoformat(),
            "steps_validated": len(spec.steps),
            "issues": []
        }

        # Validate each step
        for step in spec.steps:
            try:
                # Check if condition is valid
                if step.condition:
                    self._eval_when(step.condition, context)

                # Check if inputs can be resolved
                self._resolve_step_inputs(step, context)

                logger.debug(f"Step '{step.name}' validation passed")

            except Exception as e:
                validation_results["issues"].append({
                    "step": step.name,
                    "type": "validation_error",
                    "message": str(e)
                })

        return validation_results

    def _cleanup_context(self, context: ExecutionContext) -> None:
        """Clean up execution context resources."""
        # Close any open files
        for file_obj in context.open_files:
            try:
                file_obj.close()
            except Exception as e:
                logger.warning(f"Failed to close file: {e}")

        # Remove temporary directories
        for temp_dir in context.temp_dirs:
            try:
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temp directory {temp_dir}: {e}")

        logger.debug(f"Cleaned up execution context for run {context.run_id}")

    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        logger.info("Shutting down PipelineExecutor")
        self._thread_pool.shutdown(wait=True)


# Utility functions for creating execution contexts
def create_execution_context(
    pipeline_name: str,
    inputs: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    deterministic: bool = False,
    artifacts_dir: Optional[Union[str, Path]] = None
) -> ExecutionContext:
    """
    Create an execution context for pipeline execution.

    Args:
        pipeline_name: Name of the pipeline
        inputs: Input data for the pipeline
        run_id: Unique run identifier (generated if not provided)
        deterministic: Whether to use deterministic execution
        artifacts_dir: Directory for artifacts (default: "out")

    Returns:
        ExecutionContext instance
    """
    if run_id is None:
        run_id = f"{pipeline_name}-{int(time.time())}-{random.randint(1000, 9999)}"

    if artifacts_dir is None:
        artifacts_dir = Path("out") / run_id
    else:
        artifacts_dir = Path(artifacts_dir)

    return ExecutionContext(
        run_id=run_id,
        pipeline_name=pipeline_name,
        inputs=inputs or {},
        deterministic=deterministic,
        artifacts_dir=artifacts_dir
    )