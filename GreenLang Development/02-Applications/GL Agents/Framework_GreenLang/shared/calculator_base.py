"""
GreenLang Framework - Base Calculator Classes

Abstract base classes for deterministic calculators.
All GreenLang agent calculators should inherit from these.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Optional, TypeVar
import hashlib
import json

from .provenance import ProvenanceTracker


InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


@dataclass
class CalculationResult(Generic[OutputT]):
    """
    Standard result wrapper for all calculations.

    Includes:
    - The calculated result
    - Provenance hash
    - Execution metadata
    - Validation status
    """
    result: OutputT
    computation_hash: str
    inputs_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: float = 0.0
    calculator_name: str = ""
    calculator_version: str = ""
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result": self._serialize_result(self.result),
            "computation_hash": self.computation_hash,
            "inputs_hash": self.inputs_hash,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "calculator_name": self.calculator_name,
            "calculator_version": self.calculator_version,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def _serialize_result(self, result: Any) -> Any:
        """Serialize result for JSON."""
        if hasattr(result, 'dict'):
            return result.dict()
        elif hasattr(result, '__dict__'):
            return result.__dict__
        else:
            return result


class DeterministicCalculator(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for deterministic calculators.

    All GreenLang calculators must:
    1. Be deterministic (same inputs = same outputs)
    2. Track provenance with SHA-256 hashes
    3. Validate inputs before calculation
    4. Return CalculationResult with metadata

    Example:
        class PinchCalculator(DeterministicCalculator[PinchInputs, PinchResult]):
            NAME = "PinchAnalysis"
            VERSION = "1.0.0"

            def _validate_inputs(self, inputs: PinchInputs) -> List[str]:
                errors = []
                if not inputs.hot_streams:
                    errors.append("At least one hot stream required")
                return errors

            def _calculate(self, inputs: PinchInputs) -> PinchResult:
                # Deterministic calculation
                return result
    """

    NAME: str = "BaseCalculator"
    VERSION: str = "1.0.0"

    def __init__(
        self,
        agent_id: str = "GL-000",
        track_provenance: bool = True,
    ):
        """
        Initialize calculator.

        Args:
            agent_id: Agent identifier for provenance
            track_provenance: Whether to track calculation provenance
        """
        self.agent_id = agent_id
        self.track_provenance = track_provenance
        self._provenance_tracker = ProvenanceTracker(
            agent_id=agent_id,
            version=self.VERSION,
            store_records=track_provenance,
        )

    @property
    def provenance_tracker(self) -> ProvenanceTracker:
        """Get provenance tracker."""
        return self._provenance_tracker

    def calculate(
        self,
        inputs: InputT,
        **kwargs: Any,
    ) -> CalculationResult[OutputT]:
        """
        Perform calculation with provenance tracking.

        Args:
            inputs: Calculation inputs
            **kwargs: Additional parameters

        Returns:
            CalculationResult with result and provenance
        """
        start_time = datetime.now(timezone.utc)

        # Validate inputs
        validation_errors = self._validate_inputs(inputs)
        if validation_errors:
            return CalculationResult(
                result=None,
                computation_hash="",
                inputs_hash=self._compute_hash(inputs),
                timestamp=start_time,
                calculator_name=self.NAME,
                calculator_version=self.VERSION,
                is_valid=False,
                warnings=validation_errors,
            )

        # Perform calculation
        result = self._calculate(inputs, **kwargs)

        end_time = datetime.now(timezone.utc)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Compute hashes
        inputs_hash = self._compute_hash(inputs)
        outputs_hash = self._compute_hash(result)
        computation_hash = self._compute_combined_hash(
            inputs_hash, outputs_hash, kwargs
        )

        # Track provenance if enabled
        if self.track_provenance:
            self._provenance_tracker.create_record(
                computation_type=self.NAME,
                inputs=inputs,
                outputs=result,
                execution_time_ms=execution_time_ms,
                parameters=kwargs,
            )

        return CalculationResult(
            result=result,
            computation_hash=computation_hash,
            inputs_hash=inputs_hash,
            timestamp=start_time,
            execution_time_ms=execution_time_ms,
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            is_valid=True,
        )

    @abstractmethod
    def _validate_inputs(self, inputs: InputT) -> List[str]:
        """
        Validate calculation inputs.

        Args:
            inputs: Inputs to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        pass

    @abstractmethod
    def _calculate(self, inputs: InputT, **kwargs: Any) -> OutputT:
        """
        Perform the actual calculation.

        Must be deterministic - same inputs always produce same outputs.

        Args:
            inputs: Validated inputs
            **kwargs: Additional parameters

        Returns:
            Calculation result
        """
        pass

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data."""
        return self._provenance_tracker.compute_hash(data)

    def _compute_combined_hash(
        self,
        inputs_hash: str,
        outputs_hash: str,
        parameters: Dict[str, Any],
    ) -> str:
        """Compute combined hash for full provenance."""
        combined = {
            "inputs_hash": inputs_hash,
            "outputs_hash": outputs_hash,
            "calculator": self.NAME,
            "version": self.VERSION,
            "parameters": parameters,
        }
        return self._provenance_tracker.compute_hash(combined)


class BatchCalculator(DeterministicCalculator[List[InputT], List[OutputT]]):
    """
    Base class for batch calculations.

    Processes multiple inputs and tracks provenance for each.
    """

    def _calculate(
        self,
        inputs: List[InputT],
        **kwargs: Any,
    ) -> List[OutputT]:
        """Process batch of inputs."""
        results = []
        for input_item in inputs:
            result = self._calculate_single(input_item, **kwargs)
            results.append(result)
        return results

    @abstractmethod
    def _calculate_single(self, input_item: InputT, **kwargs: Any) -> OutputT:
        """Calculate for single input."""
        pass


class CachedCalculator(DeterministicCalculator[InputT, OutputT]):
    """
    Calculator with result caching.

    Caches results by input hash to avoid redundant calculations.
    """

    def __init__(
        self,
        agent_id: str = "GL-000",
        track_provenance: bool = True,
        cache_size: int = 1000,
    ):
        super().__init__(agent_id, track_provenance)
        self._cache: Dict[str, OutputT] = {}
        self._cache_size = cache_size

    def calculate(
        self,
        inputs: InputT,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> CalculationResult[OutputT]:
        """Calculate with optional caching."""
        if use_cache:
            inputs_hash = self._compute_hash(inputs)
            if inputs_hash in self._cache:
                return CalculationResult(
                    result=self._cache[inputs_hash],
                    computation_hash=inputs_hash,
                    inputs_hash=inputs_hash,
                    calculator_name=self.NAME,
                    calculator_version=self.VERSION,
                    is_valid=True,
                    metadata={"cached": True},
                )

        result = super().calculate(inputs, **kwargs)

        if use_cache and result.is_valid:
            self._add_to_cache(result.inputs_hash, result.result)

        return result

    def _add_to_cache(self, key: str, value: OutputT) -> None:
        """Add result to cache."""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
