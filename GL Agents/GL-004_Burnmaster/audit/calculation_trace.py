"""
CalculationTrace - Deterministic calculation tracing for BURNMASTER.

This module implements the CalculationTrace for GL-004 BURNMASTER, providing
complete tracing of all calculations, optimizations, and model inferences.

Supports zero-hallucination verification by enabling replay and determinism
checking of any calculation in the system.

Example:
    >>> tracer = CalculationTrace(config)
    >>> trace = tracer.trace_calculation("efficiency_calc", inputs, outputs)
    >>> replay = tracer.replay_calculation(trace)
    >>> check = tracer.verify_determinism(trace)
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import uuid
import copy
import time

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class CalculationType(str, Enum):
    """Types of calculations that can be traced."""
    EFFICIENCY = "efficiency"
    EMISSIONS = "emissions"
    OPTIMIZATION = "optimization"
    INFERENCE = "inference"
    VALIDATION = "validation"
    AGGREGATION = "aggregation"
    TRANSFORMATION = "transformation"


class DeterminismStatus(str, Enum):
    """Status of determinism verification."""
    DETERMINISTIC = "deterministic"
    NON_DETERMINISTIC = "non_deterministic"
    UNKNOWN = "unknown"
    ERROR = "error"


# =============================================================================
# Input/Output Models
# =============================================================================

class OptimizationResult(BaseModel):
    """Result from an optimization operation."""

    optimizer_id: str = Field(..., description="Optimizer identifier")
    status: str = Field(..., description="Optimization status")
    optimal_value: float = Field(..., description="Optimal objective value")
    optimal_solution: Dict[str, float] = Field(..., description="Optimal solution variables")
    iterations: int = Field(..., ge=0, description="Number of iterations")
    convergence_info: Dict[str, Any] = Field(default_factory=dict, description="Convergence information")
    constraints_satisfied: bool = Field(..., description="All constraints satisfied")
    computation_time_ms: float = Field(..., ge=0, description="Computation time in ms")


class CalcTrace(BaseModel):
    """Trace record for a calculation."""

    trace_id: str = Field(..., description="Unique trace identifier")
    timestamp: datetime = Field(..., description="Trace creation timestamp")
    calculation_name: str = Field(..., description="Name of the calculation")
    calculation_type: CalculationType = Field(..., description="Type of calculation")

    # Input/Output snapshots
    inputs: Dict[str, Any] = Field(..., description="Input values")
    outputs: Dict[str, Any] = Field(..., description="Output values")

    # Hashes for verification
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(..., description="SHA-256 hash of outputs")
    combined_hash: str = Field(..., description="Combined input/output hash")

    # Execution metadata
    execution_time_ms: float = Field(..., ge=0, description="Execution time in ms")
    memory_usage_bytes: Optional[int] = Field(None, description="Memory usage if tracked")

    # Formula/function reference
    formula_id: Optional[str] = Field(None, description="Formula identifier if applicable")
    function_signature: Optional[str] = Field(None, description="Function signature")
    code_version: Optional[str] = Field(None, description="Code version at execution")

    class Config:
        """Pydantic configuration."""
        frozen = True


class OptTrace(BaseModel):
    """Trace record for an optimization operation."""

    trace_id: str = Field(..., description="Unique trace identifier")
    timestamp: datetime = Field(..., description="Trace creation timestamp")
    optimizer: str = Field(..., description="Optimizer identifier")

    # Input/Output snapshots
    inputs: Dict[str, Any] = Field(..., description="Optimization inputs")
    result: Dict[str, Any] = Field(..., description="Optimization result")

    # Hashes for verification
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    result_hash: str = Field(..., description="SHA-256 hash of result")
    combined_hash: str = Field(..., description="Combined hash")

    # Optimization metadata
    objective_function: str = Field(..., description="Objective function description")
    constraints: List[str] = Field(default_factory=list, description="Constraint descriptions")
    iterations: int = Field(..., ge=0, description="Number of iterations")
    convergence_achieved: bool = Field(..., description="Whether convergence was achieved")
    execution_time_ms: float = Field(..., ge=0, description="Execution time in ms")

    # Intermediate states for debugging
    iteration_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of intermediate states"
    )

    class Config:
        """Pydantic configuration."""
        frozen = True


class InferenceTrace(BaseModel):
    """Trace record for a model inference."""

    trace_id: str = Field(..., description="Unique trace identifier")
    timestamp: datetime = Field(..., description="Trace creation timestamp")
    model: str = Field(..., description="Model identifier")

    # Input/Output snapshots
    features: Dict[str, Any] = Field(..., description="Input features")
    prediction: Dict[str, Any] = Field(..., description="Model prediction")

    # Hashes for verification
    feature_hash: str = Field(..., description="SHA-256 hash of features")
    prediction_hash: str = Field(..., description="SHA-256 hash of prediction")
    combined_hash: str = Field(..., description="Combined hash")

    # Model metadata
    model_version: str = Field(..., description="Model version")
    model_hash: str = Field(..., description="Model weights hash")
    preprocessing_applied: List[str] = Field(default_factory=list, description="Preprocessing steps")

    # Inference metadata
    execution_time_ms: float = Field(..., ge=0, description="Inference time in ms")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Prediction confidence")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance")

    class Config:
        """Pydantic configuration."""
        frozen = True


class ReplayResult(BaseModel):
    """Result of replaying a calculation."""

    original_trace_id: str = Field(..., description="Original trace ID")
    replay_timestamp: datetime = Field(..., description="Replay timestamp")

    # Replay outputs
    replayed_outputs: Dict[str, Any] = Field(..., description="Outputs from replay")
    replayed_output_hash: str = Field(..., description="Hash of replayed outputs")

    # Comparison
    outputs_match: bool = Field(..., description="Whether outputs match original")
    hash_match: bool = Field(..., description="Whether hashes match")

    # Differences if any
    differences: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed differences if outputs don't match"
    )

    # Execution metadata
    replay_execution_time_ms: float = Field(..., ge=0, description="Replay execution time")
    original_execution_time_ms: float = Field(..., ge=0, description="Original execution time")


class DeterminismCheck(BaseModel):
    """Result of determinism verification."""

    trace_id: str = Field(..., description="Trace ID being verified")
    check_timestamp: datetime = Field(..., description="Check timestamp")
    status: DeterminismStatus = Field(..., description="Determinism status")

    # Verification details
    num_replays: int = Field(..., ge=1, description="Number of replay attempts")
    all_outputs_identical: bool = Field(..., description="All replay outputs identical")
    all_hashes_identical: bool = Field(..., description="All output hashes identical")

    # Statistics across replays
    output_hashes: List[str] = Field(..., description="List of output hashes from replays")
    unique_hashes: int = Field(..., ge=1, description="Number of unique hashes")

    # Timing statistics
    execution_times_ms: List[float] = Field(..., description="Execution times for each replay")
    avg_execution_time_ms: float = Field(..., ge=0, description="Average execution time")
    std_execution_time_ms: float = Field(..., ge=0, description="Standard deviation of times")

    # Error information
    errors: List[str] = Field(default_factory=list, description="Errors during verification")


# =============================================================================
# Configuration
# =============================================================================

class CalculationTraceConfig(BaseModel):
    """Configuration for CalculationTrace."""

    storage_backend: str = Field("memory", description="Storage backend: memory, file, database")
    storage_path: str = Field("./calc_traces", description="Path for file-based storage")
    max_traces: int = Field(100000, ge=1000, description="Maximum traces to store")
    enable_iteration_history: bool = Field(True, description="Store optimization iteration history")
    max_iteration_history: int = Field(100, ge=10, description="Max iterations to store")
    determinism_replays: int = Field(3, ge=2, le=10, description="Number of replays for determinism check")


# =============================================================================
# Registered Calculations (for replay)
# =============================================================================

# Global registry for calculation functions
_calculation_registry: Dict[str, Callable] = {}


def register_calculation(name: str):
    """Decorator to register a calculation function for replay."""
    def decorator(func: Callable) -> Callable:
        _calculation_registry[name] = func
        return func
    return decorator


# =============================================================================
# CalculationTrace Implementation
# =============================================================================

class CalculationTrace:
    """
    CalculationTrace implementation for BURNMASTER.

    This class provides complete tracing of all calculations, optimizations,
    and model inferences, supporting zero-hallucination verification through
    replay and determinism checking.

    Attributes:
        config: Trace configuration
        _traces: Storage for calculation traces
        _opt_traces: Storage for optimization traces
        _inference_traces: Storage for inference traces

    Example:
        >>> config = CalculationTraceConfig()
        >>> tracer = CalculationTrace(config)
        >>> trace = tracer.trace_calculation("efficiency_calc", inputs, outputs)
        >>> replay = tracer.replay_calculation(trace)
    """

    def __init__(self, config: CalculationTraceConfig):
        """
        Initialize CalculationTrace.

        Args:
            config: Trace configuration
        """
        self.config = config
        self._traces: Dict[str, CalcTrace] = {}
        self._opt_traces: Dict[str, OptTrace] = {}
        self._inference_traces: Dict[str, InferenceTrace] = {}

        logger.info(
            f"CalculationTrace initialized with backend={config.storage_backend}, "
            f"max_traces={config.max_traces}"
        )

    def trace_calculation(
        self,
        calc_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        calculation_type: CalculationType = CalculationType.TRANSFORMATION,
        execution_time_ms: Optional[float] = None,
        formula_id: Optional[str] = None,
        function_signature: Optional[str] = None,
        code_version: Optional[str] = None
    ) -> CalcTrace:
        """
        Trace a calculation with inputs and outputs.

        Args:
            calc_name: Name of the calculation
            inputs: Input values
            outputs: Output values
            calculation_type: Type of calculation
            execution_time_ms: Execution time if known
            formula_id: Formula identifier if applicable
            function_signature: Function signature
            code_version: Code version at execution

        Returns:
            Calculation trace record

        Raises:
            ValueError: If inputs or outputs are invalid
        """
        start_time = datetime.now(timezone.utc)
        trace_start = time.perf_counter()

        try:
            trace_id = str(uuid.uuid4())

            # Deep copy inputs/outputs to ensure immutability
            inputs_copy = copy.deepcopy(inputs)
            outputs_copy = copy.deepcopy(outputs)

            # Compute hashes
            input_hash = self._compute_hash(inputs_copy)
            output_hash = self._compute_hash(outputs_copy)
            combined_hash = self._compute_hash({
                "inputs": inputs_copy,
                "outputs": outputs_copy
            })

            # Calculate execution time if not provided
            if execution_time_ms is None:
                execution_time_ms = (time.perf_counter() - trace_start) * 1000

            trace = CalcTrace(
                trace_id=trace_id,
                timestamp=start_time,
                calculation_name=calc_name,
                calculation_type=calculation_type,
                inputs=inputs_copy,
                outputs=outputs_copy,
                input_hash=input_hash,
                output_hash=output_hash,
                combined_hash=combined_hash,
                execution_time_ms=execution_time_ms,
                memory_usage_bytes=None,
                formula_id=formula_id,
                function_signature=function_signature,
                code_version=code_version
            )

            # Store trace
            self._store_trace(trace)

            processing_time_ms = (time.perf_counter() - trace_start) * 1000
            logger.debug(
                f"Traced calculation {calc_name} (id={trace_id[:8]}) "
                f"in {processing_time_ms:.2f}ms"
            )

            return trace

        except Exception as e:
            logger.error(f"Failed to trace calculation: {str(e)}", exc_info=True)
            raise

    def trace_optimization(
        self,
        optimizer: str,
        inputs: Dict[str, Any],
        result: OptimizationResult,
        objective_function: str = "minimize",
        constraints: Optional[List[str]] = None,
        iteration_history: Optional[List[Dict[str, Any]]] = None
    ) -> OptTrace:
        """
        Trace an optimization operation.

        Args:
            optimizer: Optimizer identifier
            inputs: Optimization inputs
            result: Optimization result
            objective_function: Objective function description
            constraints: Constraint descriptions
            iteration_history: History of intermediate states

        Returns:
            Optimization trace record

        Raises:
            ValueError: If inputs or result are invalid
        """
        start_time = datetime.now(timezone.utc)
        trace_start = time.perf_counter()

        try:
            trace_id = str(uuid.uuid4())

            # Deep copy data
            inputs_copy = copy.deepcopy(inputs)
            result_dict = result.dict()

            # Compute hashes
            input_hash = self._compute_hash(inputs_copy)
            result_hash = self._compute_hash(result_dict)
            combined_hash = self._compute_hash({
                "inputs": inputs_copy,
                "result": result_dict
            })

            # Limit iteration history if needed
            if iteration_history and self.config.enable_iteration_history:
                if len(iteration_history) > self.config.max_iteration_history:
                    # Keep first and last iterations, sample middle
                    iteration_history = (
                        iteration_history[:5] +
                        iteration_history[-5:]
                    )

            trace = OptTrace(
                trace_id=trace_id,
                timestamp=start_time,
                optimizer=optimizer,
                inputs=inputs_copy,
                result=result_dict,
                input_hash=input_hash,
                result_hash=result_hash,
                combined_hash=combined_hash,
                objective_function=objective_function,
                constraints=constraints or [],
                iterations=result.iterations,
                convergence_achieved=result.status == "optimal",
                execution_time_ms=result.computation_time_ms,
                iteration_history=iteration_history or []
            )

            # Store trace
            self._opt_traces[trace_id] = trace
            self._enforce_storage_limits()

            processing_time_ms = (time.perf_counter() - trace_start) * 1000
            logger.info(
                f"Traced optimization {optimizer} (id={trace_id[:8]}), "
                f"iterations={result.iterations} in {processing_time_ms:.2f}ms"
            )

            return trace

        except Exception as e:
            logger.error(f"Failed to trace optimization: {str(e)}", exc_info=True)
            raise

    def trace_model_inference(
        self,
        model: str,
        features: Dict[str, Any],
        prediction: Dict[str, Any],
        model_version: str = "1.0.0",
        model_hash: Optional[str] = None,
        preprocessing_applied: Optional[List[str]] = None,
        execution_time_ms: Optional[float] = None,
        confidence_score: Optional[float] = None,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> InferenceTrace:
        """
        Trace a model inference.

        Args:
            model: Model identifier
            features: Input features
            prediction: Model prediction
            model_version: Model version
            model_hash: Hash of model weights
            preprocessing_applied: Preprocessing steps applied
            execution_time_ms: Inference time
            confidence_score: Prediction confidence
            feature_importance: Feature importance scores

        Returns:
            Inference trace record

        Raises:
            ValueError: If features or prediction are invalid
        """
        start_time = datetime.now(timezone.utc)
        trace_start = time.perf_counter()

        try:
            trace_id = str(uuid.uuid4())

            # Deep copy data
            features_copy = copy.deepcopy(features)
            prediction_copy = copy.deepcopy(prediction)

            # Compute hashes
            feature_hash = self._compute_hash(features_copy)
            prediction_hash = self._compute_hash(prediction_copy)
            combined_hash = self._compute_hash({
                "features": features_copy,
                "prediction": prediction_copy
            })

            # Generate model hash if not provided
            if model_hash is None:
                model_hash = self._compute_hash({
                    "model": model,
                    "version": model_version
                })

            # Calculate execution time if not provided
            if execution_time_ms is None:
                execution_time_ms = (time.perf_counter() - trace_start) * 1000

            trace = InferenceTrace(
                trace_id=trace_id,
                timestamp=start_time,
                model=model,
                features=features_copy,
                prediction=prediction_copy,
                feature_hash=feature_hash,
                prediction_hash=prediction_hash,
                combined_hash=combined_hash,
                model_version=model_version,
                model_hash=model_hash,
                preprocessing_applied=preprocessing_applied or [],
                execution_time_ms=execution_time_ms,
                confidence_score=confidence_score,
                feature_importance=feature_importance
            )

            # Store trace
            self._inference_traces[trace_id] = trace
            self._enforce_storage_limits()

            processing_time_ms = (time.perf_counter() - trace_start) * 1000
            logger.debug(
                f"Traced inference {model} (id={trace_id[:8]}) "
                f"in {processing_time_ms:.2f}ms"
            )

            return trace

        except Exception as e:
            logger.error(f"Failed to trace inference: {str(e)}", exc_info=True)
            raise

    def replay_calculation(self, trace: CalcTrace) -> ReplayResult:
        """
        Replay a calculation using stored inputs.

        Args:
            trace: Calculation trace to replay

        Returns:
            Replay result with comparison

        Raises:
            ValueError: If calculation function not registered
            RuntimeError: If replay fails
        """
        start_time = datetime.now(timezone.utc)
        replay_start = time.perf_counter()

        try:
            # Check if calculation is registered
            if trace.calculation_name not in _calculation_registry:
                # If not registered, we can only verify hash consistency
                logger.warning(
                    f"Calculation {trace.calculation_name} not registered for replay, "
                    "performing hash verification only"
                )
                return self._verify_hashes_only(trace, start_time, replay_start)

            # Get the registered function
            calc_func = _calculation_registry[trace.calculation_name]

            # Execute the calculation with original inputs
            replayed_outputs = calc_func(**trace.inputs)

            # Ensure outputs are a dict
            if not isinstance(replayed_outputs, dict):
                replayed_outputs = {"result": replayed_outputs}

            # Compute hash of replayed outputs
            replayed_output_hash = self._compute_hash(replayed_outputs)

            # Compare outputs
            outputs_match = self._compare_outputs(trace.outputs, replayed_outputs)
            hash_match = trace.output_hash == replayed_output_hash

            # Calculate differences if outputs don't match
            differences = []
            if not outputs_match:
                differences = self._calculate_differences(trace.outputs, replayed_outputs)

            replay_time_ms = (time.perf_counter() - replay_start) * 1000

            result = ReplayResult(
                original_trace_id=trace.trace_id,
                replay_timestamp=start_time,
                replayed_outputs=replayed_outputs,
                replayed_output_hash=replayed_output_hash,
                outputs_match=outputs_match,
                hash_match=hash_match,
                differences=differences,
                replay_execution_time_ms=replay_time_ms,
                original_execution_time_ms=trace.execution_time_ms
            )

            logger.info(
                f"Replayed calculation {trace.calculation_name}: "
                f"match={outputs_match}, hash_match={hash_match}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to replay calculation: {str(e)}", exc_info=True)
            raise RuntimeError(f"Replay failed: {str(e)}") from e

    def verify_determinism(
        self,
        trace: CalcTrace,
        num_replays: Optional[int] = None
    ) -> DeterminismCheck:
        """
        Verify a calculation is deterministic.

        Runs multiple replays and checks that all outputs are identical.

        Args:
            trace: Calculation trace to verify
            num_replays: Number of replay attempts (overrides config)

        Returns:
            Determinism check result

        Raises:
            ValueError: If calculation function not registered
        """
        start_time = datetime.now(timezone.utc)

        try:
            num_replays = num_replays or self.config.determinism_replays

            # Check if calculation is registered
            if trace.calculation_name not in _calculation_registry:
                return DeterminismCheck(
                    trace_id=trace.trace_id,
                    check_timestamp=start_time,
                    status=DeterminismStatus.UNKNOWN,
                    num_replays=0,
                    all_outputs_identical=False,
                    all_hashes_identical=False,
                    output_hashes=[trace.output_hash],
                    unique_hashes=1,
                    execution_times_ms=[trace.execution_time_ms],
                    avg_execution_time_ms=trace.execution_time_ms,
                    std_execution_time_ms=0.0,
                    errors=[f"Calculation {trace.calculation_name} not registered for replay"]
                )

            # Get the registered function
            calc_func = _calculation_registry[trace.calculation_name]

            output_hashes: List[str] = []
            execution_times: List[float] = []
            errors: List[str] = []

            # Run multiple replays
            for i in range(num_replays):
                try:
                    replay_start = time.perf_counter()
                    replayed_outputs = calc_func(**trace.inputs)

                    if not isinstance(replayed_outputs, dict):
                        replayed_outputs = {"result": replayed_outputs}

                    execution_time = (time.perf_counter() - replay_start) * 1000
                    output_hash = self._compute_hash(replayed_outputs)

                    output_hashes.append(output_hash)
                    execution_times.append(execution_time)

                except Exception as e:
                    errors.append(f"Replay {i+1} failed: {str(e)}")

            # Analyze results
            unique_hashes = len(set(output_hashes))
            all_hashes_identical = unique_hashes == 1 if output_hashes else False
            all_outputs_identical = all_hashes_identical  # Hash identity implies output identity

            # Calculate timing statistics
            avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
            if len(execution_times) > 1:
                variance = sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)
                std_time = variance ** 0.5
            else:
                std_time = 0.0

            # Determine status
            if errors:
                status = DeterminismStatus.ERROR
            elif all_hashes_identical:
                status = DeterminismStatus.DETERMINISTIC
            else:
                status = DeterminismStatus.NON_DETERMINISTIC

            result = DeterminismCheck(
                trace_id=trace.trace_id,
                check_timestamp=start_time,
                status=status,
                num_replays=len(output_hashes),
                all_outputs_identical=all_outputs_identical,
                all_hashes_identical=all_hashes_identical,
                output_hashes=output_hashes,
                unique_hashes=unique_hashes,
                execution_times_ms=execution_times,
                avg_execution_time_ms=avg_time,
                std_execution_time_ms=std_time,
                errors=errors
            )

            logger.info(
                f"Determinism check for {trace.calculation_name}: "
                f"status={status.value}, unique_hashes={unique_hashes}/{num_replays}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to verify determinism: {str(e)}", exc_info=True)
            raise

    def get_trace(self, trace_id: str) -> Optional[CalcTrace]:
        """Get a calculation trace by ID."""
        return self._traces.get(trace_id)

    def get_opt_trace(self, trace_id: str) -> Optional[OptTrace]:
        """Get an optimization trace by ID."""
        return self._opt_traces.get(trace_id)

    def get_inference_trace(self, trace_id: str) -> Optional[InferenceTrace]:
        """Get an inference trace by ID."""
        return self._inference_traces.get(trace_id)

    def _store_trace(self, trace: CalcTrace) -> None:
        """Store a calculation trace."""
        self._traces[trace.trace_id] = trace
        self._enforce_storage_limits()

    def _enforce_storage_limits(self) -> None:
        """Enforce maximum storage limits by removing oldest traces."""
        total_traces = len(self._traces) + len(self._opt_traces) + len(self._inference_traces)

        if total_traces > self.config.max_traces:
            # Remove oldest traces proportionally
            excess = total_traces - self.config.max_traces

            # Remove from each storage proportionally
            for storage in [self._traces, self._opt_traces, self._inference_traces]:
                if storage and excess > 0:
                    sorted_keys = sorted(
                        storage.keys(),
                        key=lambda k: storage[k].timestamp
                    )
                    to_remove = min(len(sorted_keys) // 3, excess)
                    for key in sorted_keys[:to_remove]:
                        del storage[key]
                    excess -= to_remove

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _compare_outputs(self, original: Dict[str, Any], replayed: Dict[str, Any]) -> bool:
        """Compare two output dictionaries for equality."""
        try:
            # Compare JSON representations for deep equality
            original_json = json.dumps(original, sort_keys=True, default=str)
            replayed_json = json.dumps(replayed, sort_keys=True, default=str)
            return original_json == replayed_json
        except Exception:
            return False

    def _calculate_differences(
        self,
        original: Dict[str, Any],
        replayed: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Calculate detailed differences between outputs."""
        differences = []

        all_keys = set(original.keys()) | set(replayed.keys())

        for key in all_keys:
            orig_val = original.get(key)
            repl_val = replayed.get(key)

            if key not in original:
                differences.append({
                    "key": key,
                    "type": "added",
                    "original": None,
                    "replayed": repl_val
                })
            elif key not in replayed:
                differences.append({
                    "key": key,
                    "type": "removed",
                    "original": orig_val,
                    "replayed": None
                })
            elif orig_val != repl_val:
                differences.append({
                    "key": key,
                    "type": "changed",
                    "original": orig_val,
                    "replayed": repl_val
                })

        return differences

    def _verify_hashes_only(
        self,
        trace: CalcTrace,
        start_time: datetime,
        replay_start: float
    ) -> ReplayResult:
        """Verify trace by hash comparison only (when replay not possible)."""
        # Recompute hashes from stored data
        recomputed_output_hash = self._compute_hash(trace.outputs)
        hash_match = trace.output_hash == recomputed_output_hash

        replay_time_ms = (time.perf_counter() - replay_start) * 1000

        return ReplayResult(
            original_trace_id=trace.trace_id,
            replay_timestamp=start_time,
            replayed_outputs=trace.outputs,  # Return original outputs
            replayed_output_hash=recomputed_output_hash,
            outputs_match=hash_match,  # If hash matches, data is consistent
            hash_match=hash_match,
            differences=[],
            replay_execution_time_ms=replay_time_ms,
            original_execution_time_ms=trace.execution_time_ms
        )
