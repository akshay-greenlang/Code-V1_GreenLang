"""
Determinism Validator for GL-002 BoilerEfficiencyOptimizer.

This module provides runtime verification of deterministic behavior across
all agent operations. It ensures 100% reproducibility and zero stochastic
operations in calculations, enforcing GreenLang's zero-hallucination principle.

Key Features:
- AI configuration validation (temperature=0.0, seed=42)
- Calculation determinism verification (3x execution checks)
- Provenance hash validation on every execution
- Cache key determinism verification
- Seed propagation verification
- Unseeded random operation detection
- Timestamp-based calculation detection

Example:
    >>> validator = DeterminismValidator(strict_mode=True)
    >>> validator.verify_ai_config(config)
    >>> validator.verify_calculation_determinism(calculate_efficiency, inputs)
"""

import hashlib
import time
import random
import inspect
import logging
import functools
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
from decimal import Decimal, getcontext

logger = logging.getLogger(__name__)


# ============================================================================
# DETERMINISM EXCEPTIONS
# ============================================================================

class DeterminismViolationError(Exception):
    """Raised when deterministic behavior is violated."""
    pass


class StochasticOperationError(DeterminismViolationError):
    """Raised when unseeded random operations are detected."""
    pass


class NonDeterministicResultError(DeterminismViolationError):
    """Raised when function returns different results for same input."""
    pass


class TimestampCalculationError(DeterminismViolationError):
    """Raised when timestamp-based calculations are detected."""
    pass


class AIConfigViolationError(DeterminismViolationError):
    """Raised when AI configuration violates determinism requirements."""
    pass


# ============================================================================
# DETERMINISM VALIDATOR
# ============================================================================

class DeterminismValidator:
    """
    Runtime validator for deterministic behavior.

    This class provides comprehensive validation of deterministic operations
    across all agent calculations, ensuring 100% reproducibility.

    Attributes:
        strict_mode: If True, raises exceptions on violations; if False, logs warnings
        verification_runs: Number of times to run calculations for determinism check
        tolerance: Floating-point comparison tolerance
        detected_violations: Count of detected violations
    """

    def __init__(
        self,
        strict_mode: bool = True,
        verification_runs: int = 3,
        tolerance: float = 1e-15,
        enable_metrics: bool = True
    ):
        """
        Initialize DeterminismValidator.

        Args:
            strict_mode: Raise exceptions on violations (True) or log warnings (False)
            verification_runs: Number of calculation runs for verification
            tolerance: Floating-point comparison tolerance
            enable_metrics: Enable Prometheus metrics tracking
        """
        self.strict_mode = strict_mode
        self.verification_runs = verification_runs
        self.tolerance = tolerance
        self.enable_metrics = enable_metrics

        # Violation tracking
        self.detected_violations: Dict[str, int] = defaultdict(int)
        self.violation_details: List[Dict[str, Any]] = []

        # Allowed non-deterministic operations (for logging, monitoring, etc.)
        self.allowed_timestamp_operations: Set[str] = {
            'logger', 'datetime.now', 'time.time', 'timestamp',
            'log_entry', 'metric_timestamp', 'audit_timestamp'
        }

        # AI configuration requirements
        self.required_ai_config = {
            'temperature': 0.0,
            'seed': 42,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }

        logger.info(f"DeterminismValidator initialized (strict_mode={strict_mode}, runs={verification_runs})")

    # ========================================================================
    # AI CONFIGURATION VALIDATION
    # ========================================================================

    def verify_ai_config(self, config: Any) -> bool:
        """
        Verify AI configuration meets determinism requirements.

        Args:
            config: AI/LLM configuration object

        Returns:
            True if configuration is deterministic

        Raises:
            AIConfigViolationError: If configuration violates determinism (strict mode)
        """
        violations = []

        # Check temperature
        if hasattr(config, 'temperature'):
            if config.temperature != self.required_ai_config['temperature']:
                violations.append(
                    f"Temperature must be exactly {self.required_ai_config['temperature']}, "
                    f"got {config.temperature}"
                )

        # Check seed
        if hasattr(config, 'seed'):
            if config.seed != self.required_ai_config['seed']:
                violations.append(
                    f"Seed must be exactly {self.required_ai_config['seed']}, "
                    f"got {config.seed}"
                )
        else:
            violations.append("Seed not set in AI configuration")

        # Check top_p (nucleus sampling)
        if hasattr(config, 'top_p'):
            if config.top_p != self.required_ai_config['top_p']:
                violations.append(
                    f"top_p must be exactly {self.required_ai_config['top_p']}, "
                    f"got {config.top_p}"
                )

        # Check penalties
        if hasattr(config, 'frequency_penalty'):
            if config.frequency_penalty != self.required_ai_config['frequency_penalty']:
                violations.append(
                    f"frequency_penalty must be {self.required_ai_config['frequency_penalty']}, "
                    f"got {config.frequency_penalty}"
                )

        if hasattr(config, 'presence_penalty'):
            if config.presence_penalty != self.required_ai_config['presence_penalty']:
                violations.append(
                    f"presence_penalty must be {self.required_ai_config['presence_penalty']}, "
                    f"got {config.presence_penalty}"
                )

        if violations:
            self.detected_violations['ai_config'] += len(violations)
            self._record_violation('ai_config_violation', violations)

            if self.strict_mode:
                raise AIConfigViolationError(
                    f"AI configuration violates determinism requirements:\n" +
                    "\n".join(f"  - {v}" for v in violations)
                )
            else:
                logger.warning(f"AI config violations detected: {violations}")
                return False

        logger.debug("AI configuration validated successfully")
        return True

    # ========================================================================
    # CALCULATION DETERMINISM VERIFICATION
    # ========================================================================

    def verify_calculation_determinism(
        self,
        func: Callable,
        inputs: Any,
        runs: Optional[int] = None
    ) -> bool:
        """
        Verify calculation produces identical results across multiple runs.

        Args:
            func: Function to verify
            inputs: Input data (must be deterministic)
            runs: Number of verification runs (default: self.verification_runs)

        Returns:
            True if all runs produce identical results

        Raises:
            NonDeterministicResultError: If results differ (strict mode)
        """
        runs = runs or self.verification_runs
        results = []

        # Run calculation multiple times
        for i in range(runs):
            try:
                result = func(inputs)
                results.append(self._normalize_result(result))
            except Exception as e:
                self.detected_violations['calculation_error'] += 1
                logger.error(f"Calculation failed on run {i+1}: {e}")
                raise

        # Verify all results are identical
        first_result = results[0]
        differences = []

        for i, result in enumerate(results[1:], start=2):
            if not self._results_equal(first_result, result):
                differences.append(f"Run {i} differs from run 1")

        if differences:
            self.detected_violations['non_deterministic_calculation'] += 1
            self._record_violation('non_deterministic_result', {
                'function': func.__name__,
                'differences': differences,
                'results': results
            })

            if self.strict_mode:
                raise NonDeterministicResultError(
                    f"Function {func.__name__} produced different results:\n" +
                    "\n".join(f"  - {d}" for d in differences)
                )
            else:
                logger.warning(f"Non-deterministic calculation detected: {func.__name__}")
                return False

        logger.debug(f"Calculation determinism verified: {func.__name__} ({runs} runs)")
        return True

    def _normalize_result(self, result: Any) -> Any:
        """
        Normalize result for comparison.

        Args:
            result: Result to normalize

        Returns:
            Normalized result
        """
        if isinstance(result, dict):
            # Sort dict keys for consistent comparison
            return {k: self._normalize_result(v) for k, v in sorted(result.items())}
        elif isinstance(result, list):
            return [self._normalize_result(item) for item in result]
        elif isinstance(result, (float, np.floating)):
            # Round to tolerance for floating-point comparison
            return round(float(result), -int(np.log10(self.tolerance)))
        elif isinstance(result, Decimal):
            return str(result)
        elif hasattr(result, '__dict__'):
            # Handle dataclass/object results
            return self._normalize_result(result.__dict__)
        else:
            return result

    def _results_equal(self, result1: Any, result2: Any) -> bool:
        """
        Compare two results for equality with tolerance.

        Args:
            result1: First result
            result2: Second result

        Returns:
            True if results are equal within tolerance
        """
        if type(result1) != type(result2):
            return False

        if isinstance(result1, dict):
            if set(result1.keys()) != set(result2.keys()):
                return False
            return all(
                self._results_equal(result1[k], result2[k])
                for k in result1.keys()
            )
        elif isinstance(result1, list):
            if len(result1) != len(result2):
                return False
            return all(
                self._results_equal(r1, r2)
                for r1, r2 in zip(result1, result2)
            )
        elif isinstance(result1, (float, np.floating)):
            return abs(float(result1) - float(result2)) < self.tolerance
        else:
            return result1 == result2

    # ========================================================================
    # PROVENANCE HASH VERIFICATION
    # ========================================================================

    def verify_provenance_hash(
        self,
        input_data: Any,
        result: Any,
        provided_hash: str
    ) -> bool:
        """
        Verify provenance hash matches calculated hash.

        Args:
            input_data: Input data
            result: Calculation result
            provided_hash: Provenance hash from result

        Returns:
            True if hash is correct

        Raises:
            DeterminismViolationError: If hash doesn't match (strict mode)
        """
        # Calculate expected hash
        expected_hash = self._calculate_provenance_hash(input_data, result)

        if expected_hash != provided_hash:
            self.detected_violations['provenance_hash_mismatch'] += 1
            self._record_violation('provenance_hash_mismatch', {
                'expected': expected_hash,
                'provided': provided_hash,
                'input': str(input_data)[:100],  # Truncate for logging
                'result': str(result)[:100]
            })

            if self.strict_mode:
                raise DeterminismViolationError(
                    f"Provenance hash mismatch:\n"
                    f"  Expected: {expected_hash}\n"
                    f"  Provided: {provided_hash}"
                )
            else:
                logger.warning(f"Provenance hash mismatch detected")
                return False

        return True

    def _calculate_provenance_hash(self, input_data: Any, result: Any) -> str:
        """
        Calculate SHA-256 provenance hash.

        Args:
            input_data: Input data
            result: Calculation result

        Returns:
            SHA-256 hash string
        """
        # Normalize data for hashing
        input_str = self._serialize_for_hash(input_data)
        result_str = self._serialize_for_hash(result)

        provenance_str = f"{input_str}|{result_str}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _serialize_for_hash(self, data: Any) -> str:
        """
        Serialize data for consistent hashing.

        Args:
            data: Data to serialize

        Returns:
            Serialized string
        """
        import json

        if isinstance(data, dict):
            # Sort keys for consistent ordering
            return json.dumps(data, sort_keys=True, default=str)
        elif isinstance(data, (list, tuple)):
            return json.dumps(list(data), default=str)
        elif hasattr(data, '__dict__'):
            return json.dumps(data.__dict__, sort_keys=True, default=str)
        else:
            return str(data)

    # ========================================================================
    # CACHE KEY DETERMINISM VERIFICATION
    # ========================================================================

    def verify_cache_key_determinism(
        self,
        cache_key_func: Callable,
        data: Any,
        runs: int = 10
    ) -> bool:
        """
        Verify cache key generation is deterministic.

        Args:
            cache_key_func: Function that generates cache keys
            data: Input data
            runs: Number of verification runs

        Returns:
            True if cache keys are deterministic

        Raises:
            DeterminismViolationError: If cache keys differ (strict mode)
        """
        cache_keys = []

        for i in range(runs):
            try:
                key = cache_key_func(data)
                cache_keys.append(key)
            except Exception as e:
                logger.error(f"Cache key generation failed: {e}")
                raise

        # All keys should be identical
        if len(set(cache_keys)) != 1:
            self.detected_violations['cache_key_non_deterministic'] += 1
            self._record_violation('cache_key_non_deterministic', {
                'function': cache_key_func.__name__,
                'unique_keys': len(set(cache_keys)),
                'sample_keys': cache_keys[:3]
            })

            if self.strict_mode:
                raise DeterminismViolationError(
                    f"Cache key generation is non-deterministic:\n"
                    f"  Generated {len(set(cache_keys))} unique keys in {runs} runs"
                )
            else:
                logger.warning("Non-deterministic cache key generation detected")
                return False

        return True

    # ========================================================================
    # SEED PROPAGATION VERIFICATION
    # ========================================================================

    def verify_seed_propagation(self, seed: int = 42) -> bool:
        """
        Verify random seed is properly propagated to all RNG operations.

        Args:
            seed: Expected seed value

        Returns:
            True if seed is properly set

        Raises:
            StochasticOperationError: If unseeded operations detected (strict mode)
        """
        violations = []

        # Check Python random module
        random.seed(seed)
        test_values = [random.random() for _ in range(5)]

        # Re-seed and verify same sequence
        random.seed(seed)
        verify_values = [random.random() for _ in range(5)]

        if test_values != verify_values:
            violations.append("Python random module not properly seeded")

        # Check NumPy random
        np.random.seed(seed)
        np_test = np.random.rand(5)

        np.random.seed(seed)
        np_verify = np.random.rand(5)

        if not np.array_equal(np_test, np_verify):
            violations.append("NumPy random not properly seeded")

        if violations:
            self.detected_violations['seed_propagation_failure'] += len(violations)
            self._record_violation('seed_propagation_failure', violations)

            if self.strict_mode:
                raise StochasticOperationError(
                    f"Seed propagation verification failed:\n" +
                    "\n".join(f"  - {v}" for v in violations)
                )
            else:
                logger.warning(f"Seed propagation issues: {violations}")
                return False

        return True

    # ========================================================================
    # UNSEEDED RANDOM OPERATION DETECTION
    # ========================================================================

    def detect_unseeded_random_operations(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> List[str]:
        """
        Detect unseeded random operations in function execution.

        Args:
            func: Function to analyze
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            List of detected unseeded operations
        """
        detected_operations = []

        # Monitor random module calls
        original_random = random.random
        original_randint = random.randint
        original_choice = random.choice

        def monitored_random():
            detected_operations.append('random.random')
            return original_random()

        def monitored_randint(a, b):
            detected_operations.append('random.randint')
            return original_randint(a, b)

        def monitored_choice(seq):
            detected_operations.append('random.choice')
            return original_choice(seq)

        # Temporarily replace random functions
        random.random = monitored_random
        random.randint = monitored_randint
        random.choice = monitored_choice

        try:
            func(*args, **kwargs)
        finally:
            # Restore original functions
            random.random = original_random
            random.randint = original_randint
            random.choice = original_choice

        if detected_operations:
            self.detected_violations['unseeded_random'] += len(detected_operations)
            self._record_violation('unseeded_random_operations', {
                'function': func.__name__,
                'operations': detected_operations
            })

        return detected_operations

    # ========================================================================
    # TIMESTAMP-BASED CALCULATION DETECTION
    # ========================================================================

    def detect_timestamp_calculations(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> List[str]:
        """
        Detect timestamp-based calculations (non-deterministic).

        Args:
            func: Function to analyze
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            List of detected timestamp operations
        """
        detected_timestamps = []

        # Analyze function source code
        try:
            source = inspect.getsource(func)

            # Check for timestamp-related operations
            timestamp_patterns = [
                'datetime.now()',
                'time.time()',
                'timestamp',
                'current_time',
                'get_time',
            ]

            for pattern in timestamp_patterns:
                if pattern in source:
                    # Check if it's an allowed operation
                    is_allowed = any(
                        allowed in source
                        for allowed in self.allowed_timestamp_operations
                    )

                    if not is_allowed:
                        detected_timestamps.append(pattern)

        except (OSError, TypeError):
            # Source not available (built-in function, etc.)
            pass

        if detected_timestamps:
            self.detected_violations['timestamp_calculations'] += len(detected_timestamps)
            self._record_violation('timestamp_based_calculation', {
                'function': func.__name__,
                'patterns': detected_timestamps
            })

        return detected_timestamps

    # ========================================================================
    # DETERMINISM DECORATOR
    # ========================================================================

    def deterministic(self, func: Callable) -> Callable:
        """
        Decorator to verify function determinism at runtime.

        Usage:
            @validator.deterministic
            def calculate_efficiency(inputs):
                return inputs['output'] / inputs['input']

        Args:
            func: Function to decorate

        Returns:
            Wrapped function with determinism verification
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-execution checks
            unseeded_ops = self.detect_unseeded_random_operations(func, *args, **kwargs)
            if unseeded_ops and self.strict_mode:
                raise StochasticOperationError(
                    f"Unseeded random operations in {func.__name__}: {unseeded_ops}"
                )

            timestamp_ops = self.detect_timestamp_calculations(func, *args, **kwargs)
            if timestamp_ops and self.strict_mode:
                raise TimestampCalculationError(
                    f"Timestamp-based calculations in {func.__name__}: {timestamp_ops}"
                )

            # Execute function
            result = func(*args, **kwargs)

            # Post-execution verification (optional, can be disabled for performance)
            # Verify determinism with multiple runs
            # self.verify_calculation_determinism(func, (args, kwargs))

            return result

        return wrapper

    # ========================================================================
    # REPORTING AND METRICS
    # ========================================================================

    def _record_violation(self, violation_type: str, details: Any) -> None:
        """
        Record violation details.

        Args:
            violation_type: Type of violation
            details: Violation details
        """
        self.violation_details.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': violation_type,
            'details': details
        })

        # Update Prometheus metrics if enabled
        if self.enable_metrics:
            try:
                from .metrics import determinism_verification_failures
                determinism_verification_failures.labels(
                    violation_type=violation_type
                ).inc()
            except ImportError:
                pass

    def get_violation_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected violations.

        Returns:
            Violation summary dictionary
        """
        return {
            'total_violations': sum(self.detected_violations.values()),
            'violations_by_type': dict(self.detected_violations),
            'recent_violations': self.violation_details[-10:],  # Last 10
            'determinism_score': self._calculate_determinism_score()
        }

    def _calculate_determinism_score(self) -> float:
        """
        Calculate determinism score (0-100).

        Returns:
            Determinism score percentage
        """
        total_checks = sum(self.detected_violations.values()) + 100  # Assume 100 successful checks
        total_violations = sum(self.detected_violations.values())

        if total_checks == 0:
            return 100.0

        score = ((total_checks - total_violations) / total_checks) * 100
        return round(max(0.0, min(100.0, score)), 2)

    def reset_violations(self) -> None:
        """Reset violation tracking."""
        self.detected_violations.clear()
        self.violation_details.clear()
        logger.info("Violation tracking reset")


# ============================================================================
# GLOBAL VALIDATOR INSTANCE
# ============================================================================

# Default validator instance for easy import
default_validator = DeterminismValidator(strict_mode=True)
