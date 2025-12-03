"""
Evaluation Harness - Golden test execution and determinism validation.

This module implements comprehensive evaluation capabilities for LLM models:
- Golden test execution (expected vs actual comparison)
- Determinism validation (bit-perfect reproducibility)
- Performance metrics collection (latency, tokens, cost)
- Evaluation report generation

Example:
    >>> executor = GoldenTestExecutor(model_registry)
    >>> report = await executor.run_golden_tests("claude-sonnet-4", test_suite)
    >>> assert report.all_tests_passed
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import hashlib
import logging
import time
import statistics
from enum import Enum
import json

from greenlang.registry.model_registry import model_registry, ModelMetadata

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GoldenTest(BaseModel):
    """Golden test definition."""

    id: str = Field(..., description="Unique test ID")
    name: str = Field(..., description="Test name")
    prompt: str = Field(..., description="Input prompt")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    expected_response: str = Field(..., description="Expected model response")
    temperature: float = Field(0.0, description="Temperature (0.0 for determinism)")
    max_tokens: Optional[int] = Field(None, description="Max tokens")
    tags: List[str] = Field(default_factory=list, description="Test tags")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GoldenTestResult(BaseModel):
    """Result of a single golden test execution."""

    test_id: str
    test_name: str
    status: TestStatus
    passed: bool = Field(..., description="Whether test passed")
    exact_match: bool = Field(..., description="Exact string match")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity (0-1)")

    # Actual vs expected
    expected_response: str
    actual_response: str
    diff: Optional[str] = Field(None, description="Diff between expected and actual")

    # Performance
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float

    # Provenance
    provenance_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None


class DeterminismCheckResult(BaseModel):
    """Result of determinism validation (multiple runs)."""

    test_id: str
    num_runs: int = Field(..., description="Number of test runs")
    all_identical: bool = Field(..., description="All responses bit-perfect identical")
    unique_responses: int = Field(..., description="Number of unique responses")
    responses: List[str] = Field(..., description="All responses")
    hashes: List[str] = Field(..., description="SHA-256 hashes of responses")
    variance_detected: bool = Field(..., description="Any variance detected")


class PerformanceMetrics(BaseModel):
    """Aggregated performance metrics."""

    total_tests: int
    total_tokens: int
    total_cost_usd: float

    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    avg_input_tokens: float
    avg_output_tokens: float


class EvaluationReport(BaseModel):
    """Complete evaluation report."""

    model_id: str
    test_suite_name: str
    execution_id: str

    # Summary
    total_tests: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    pass_rate: float = Field(..., ge=0.0, le=1.0)

    # Results
    test_results: List[GoldenTestResult]
    determinism_results: Optional[List[DeterminismCheckResult]] = None

    # Performance
    performance_metrics: PerformanceMetrics

    # Metadata
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    all_tests_passed: bool = Field(..., description="All tests passed flag")


# ============================================================================
# GOLDEN TEST EXECUTOR
# ============================================================================

class GoldenTestExecutor:
    """
    Executes golden tests against LLM models.

    Golden tests compare actual model responses to expected responses
    to ensure model behavior matches expectations (zero-hallucination validation).

    Example:
        >>> executor = GoldenTestExecutor(model_registry)
        >>> tests = [GoldenTest(...), GoldenTest(...)]
        >>> report = await executor.run_golden_tests("claude-sonnet-4", tests)
    """

    def __init__(self, registry=None):
        """Initialize golden test executor."""
        self.registry = registry or model_registry

    async def run_golden_tests(
        self,
        model_id: str,
        tests: List[GoldenTest],
        test_suite_name: str = "default",
        check_determinism: bool = True,
        determinism_runs: int = 3
    ) -> EvaluationReport:
        """
        Run golden tests against a model.

        Args:
            model_id: Model to test
            tests: List of golden tests
            test_suite_name: Name of test suite
            check_determinism: Whether to run determinism checks
            determinism_runs: Number of runs for determinism validation

        Returns:
            Complete evaluation report
        """
        start_time = datetime.utcnow()
        execution_id = hashlib.sha256(
            f"{model_id}|{test_suite_name}|{start_time}".encode()
        ).hexdigest()[:16]

        logger.info(
            f"Starting golden test execution: {test_suite_name} on {model_id} "
            f"({len(tests)} tests)"
        )

        # Execute all tests
        test_results = []
        for test in tests:
            result = await self._execute_test(model_id, test)
            test_results.append(result)

        # Run determinism checks if requested
        determinism_results = None
        if check_determinism:
            logger.info(f"Running determinism checks ({determinism_runs} runs per test)")
            determinism_results = await self._check_determinism(
                model_id,
                tests,
                runs=determinism_runs
            )

        # Calculate metrics
        performance_metrics = self._calculate_performance_metrics(test_results)

        # Generate summary
        tests_passed = sum(1 for r in test_results if r.passed)
        tests_failed = sum(1 for r in test_results if r.status == TestStatus.FAILED)
        tests_skipped = sum(1 for r in test_results if r.status == TestStatus.SKIPPED)

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        report = EvaluationReport(
            model_id=model_id,
            test_suite_name=test_suite_name,
            execution_id=execution_id,
            total_tests=len(tests),
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            pass_rate=tests_passed / len(tests) if tests else 0.0,
            test_results=test_results,
            determinism_results=determinism_results,
            performance_metrics=performance_metrics,
            started_at=start_time,
            completed_at=end_time,
            duration_seconds=duration,
            all_tests_passed=tests_passed == len(tests)
        )

        logger.info(
            f"Golden test execution complete: {tests_passed}/{len(tests)} passed "
            f"({report.pass_rate*100:.1f}%) in {duration:.2f}s"
        )

        return report

    async def _execute_test(
        self,
        model_id: str,
        test: GoldenTest
    ) -> GoldenTestResult:
        """Execute a single golden test."""
        start_time = time.time()

        try:
            # Invoke model (mock for now)
            actual_response, input_tokens, output_tokens = await self._invoke_model(
                model_id=model_id,
                prompt=test.prompt,
                system_prompt=test.system_prompt,
                temperature=test.temperature,
                max_tokens=test.max_tokens
            )

            latency_ms = (time.time() - start_time) * 1000

            # Compare responses
            exact_match = actual_response == test.expected_response
            similarity_score = self._calculate_similarity(
                test.expected_response,
                actual_response
            )

            # Calculate cost
            model = self.registry.get_model(model_id)
            cost_usd = 0.0
            if model and model.avg_cost_per_1k_tokens:
                total_tokens = input_tokens + output_tokens
                cost_usd = (total_tokens / 1000.0) * model.avg_cost_per_1k_tokens

            # Generate diff if not exact match
            diff = None
            if not exact_match:
                diff = self._generate_diff(test.expected_response, actual_response)

            # Determine pass/fail
            passed = exact_match or similarity_score >= 0.95

            # Calculate provenance hash
            provenance_hash = hashlib.sha256(
                f"{test.id}|{test.prompt}|{actual_response}".encode()
            ).hexdigest()

            return GoldenTestResult(
                test_id=test.id,
                test_name=test.name,
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                passed=passed,
                exact_match=exact_match,
                similarity_score=similarity_score,
                expected_response=test.expected_response,
                actual_response=actual_response,
                diff=diff,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            logger.error(f"Test {test.id} failed: {str(e)}", exc_info=True)
            return GoldenTestResult(
                test_id=test.id,
                test_name=test.name,
                status=TestStatus.FAILED,
                passed=False,
                exact_match=False,
                similarity_score=0.0,
                expected_response=test.expected_response,
                actual_response="",
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                provenance_hash="",
                error_message=str(e)
            )

    async def _check_determinism(
        self,
        model_id: str,
        tests: List[GoldenTest],
        runs: int = 3
    ) -> List[DeterminismCheckResult]:
        """Check determinism by running tests multiple times."""
        results = []

        for test in tests:
            responses = []
            hashes = []

            # Run test multiple times
            for _ in range(runs):
                response, _, _ = await self._invoke_model(
                    model_id=model_id,
                    prompt=test.prompt,
                    system_prompt=test.system_prompt,
                    temperature=test.temperature,
                    max_tokens=test.max_tokens
                )
                responses.append(response)
                hash_value = hashlib.sha256(response.encode()).hexdigest()
                hashes.append(hash_value)

            # Check if all identical
            unique_responses = len(set(responses))
            all_identical = unique_responses == 1

            results.append(DeterminismCheckResult(
                test_id=test.id,
                num_runs=runs,
                all_identical=all_identical,
                unique_responses=unique_responses,
                responses=responses,
                hashes=hashes,
                variance_detected=not all_identical
            ))

        return results

    async def _invoke_model(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, int, int]:
        """
        Invoke model (mock implementation).

        Returns:
            Tuple of (response, input_tokens, output_tokens)
        """
        # TODO: Replace with actual LLM SDK calls
        model = self.registry.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        # Mock response
        response = f"[{model.name}] {prompt[:50]}... processed"

        # Mock token counts
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4

        return response, input_tokens, output_tokens

    def _calculate_similarity(self, expected: str, actual: str) -> float:
        """
        Calculate similarity score between expected and actual.

        Uses simple character-level similarity for now.
        """
        if not expected and not actual:
            return 1.0
        if not expected or not actual:
            return 0.0

        # Simple character overlap ratio
        expected_chars = set(expected)
        actual_chars = set(actual)
        overlap = len(expected_chars & actual_chars)
        total = len(expected_chars | actual_chars)

        return overlap / total if total > 0 else 0.0

    def _generate_diff(self, expected: str, actual: str) -> str:
        """Generate simple diff string."""
        return f"Expected length: {len(expected)}, Actual length: {len(actual)}"

    def _calculate_performance_metrics(
        self,
        results: List[GoldenTestResult]
    ) -> PerformanceMetrics:
        """Calculate aggregated performance metrics."""
        if not results:
            return PerformanceMetrics(
                total_tests=0,
                total_tokens=0,
                total_cost_usd=0.0,
                avg_latency_ms=0.0,
                min_latency_ms=0.0,
                max_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                avg_input_tokens=0.0,
                avg_output_tokens=0.0
            )

        latencies = [r.latency_ms for r in results]
        input_tokens_list = [r.input_tokens for r in results]
        output_tokens_list = [r.output_tokens for r in results]

        return PerformanceMetrics(
            total_tests=len(results),
            total_tokens=sum(r.input_tokens + r.output_tokens for r in results),
            total_cost_usd=sum(r.cost_usd for r in results),
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=statistics.median(latencies),
            p95_latency_ms=self._percentile(latencies, 95),
            p99_latency_ms=self._percentile(latencies, 99),
            avg_input_tokens=statistics.mean(input_tokens_list),
            avg_output_tokens=statistics.mean(output_tokens_list)
        )

    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100.0))
        return sorted_data[min(index, len(sorted_data) - 1)]


# ============================================================================
# DETERMINISM VALIDATOR
# ============================================================================

class DeterminismValidator:
    """
    Validates bit-perfect reproducibility of model responses.

    Ensures that models produce identical outputs for identical inputs
    (critical for zero-hallucination guarantees in regulatory contexts).
    """

    def __init__(self, registry=None):
        """Initialize determinism validator."""
        self.registry = registry or model_registry

    async def validate_determinism(
        self,
        model_id: str,
        prompt: str,
        runs: int = 5,
        temperature: float = 0.0
    ) -> DeterminismCheckResult:
        """
        Validate determinism for a single prompt.

        Args:
            model_id: Model to test
            prompt: Input prompt
            runs: Number of test runs
            temperature: Temperature (must be 0.0 for determinism)

        Returns:
            Determinism check result
        """
        if temperature != 0.0:
            logger.warning(
                f"Temperature {temperature} is non-zero. "
                "Determinism not guaranteed for temperature > 0.0"
            )

        responses = []
        hashes = []

        logger.info(f"Running determinism check: {runs} runs")

        for i in range(runs):
            # Invoke model
            executor = GoldenTestExecutor(self.registry)
            response, _, _ = await executor._invoke_model(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature
            )

            responses.append(response)
            hash_value = hashlib.sha256(response.encode()).hexdigest()
            hashes.append(hash_value)

            logger.debug(f"Run {i+1}/{runs}: hash={hash_value[:16]}...")

        # Check if all identical
        unique_responses = len(set(responses))
        all_identical = unique_responses == 1

        if all_identical:
            logger.info("Determinism check PASSED: All responses identical")
        else:
            logger.warning(
                f"Determinism check FAILED: {unique_responses} unique responses detected"
            )

        return DeterminismCheckResult(
            test_id="determinism_check",
            num_runs=runs,
            all_identical=all_identical,
            unique_responses=unique_responses,
            responses=responses,
            hashes=hashes,
            variance_detected=not all_identical
        )


# ============================================================================
# METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    """
    Collects and aggregates performance metrics.

    Tracks:
    - Latency (p50, p95, p99)
    - Token usage (input, output, total)
    - Cost (per request, total)
    - Success rate
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}

    def record_invocation(
        self,
        model_id: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        success: bool = True
    ):
        """Record a single model invocation."""
        if model_id not in self.metrics:
            self.metrics[model_id] = []

        self.metrics[model_id].append({
            "timestamp": datetime.utcnow(),
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "success": success
        })

    def get_aggregated_metrics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get aggregated metrics for a model."""
        if model_id not in self.metrics or not self.metrics[model_id]:
            return None

        data = self.metrics[model_id]
        latencies = [d["latency_ms"] for d in data]
        costs = [d["cost_usd"] for d in data]
        successes = [d["success"] for d in data]

        return {
            "model_id": model_id,
            "total_invocations": len(data),
            "success_rate": sum(successes) / len(successes),
            "avg_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": GoldenTestExecutor._percentile(latencies, 95),
            "p99_latency_ms": GoldenTestExecutor._percentile(latencies, 99),
            "total_cost_usd": sum(costs),
            "avg_cost_per_request": statistics.mean(costs),
            "total_input_tokens": sum(d["input_tokens"] for d in data),
            "total_output_tokens": sum(d["output_tokens"] for d in data)
        }

    def export_metrics(self, output_file: str):
        """Export metrics to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

        logger.info(f"Metrics exported to {output_file}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_golden_test_suite(name: str, tests: List[Dict[str, Any]]) -> List[GoldenTest]:
    """
    Create a golden test suite from test definitions.

    Example:
        >>> tests = [
        ...     {"id": "test1", "name": "Basic", "prompt": "Hello", "expected_response": "Hi"}
        ... ]
        >>> suite = create_golden_test_suite("basic_tests", tests)
    """
    return [GoldenTest(**test) for test in tests]
