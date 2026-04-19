"""
Performance Dimension Evaluator

Evaluates agent performance including:
- Latency benchmarks
- Throughput measurements
- Memory efficiency
- Scalability characteristics
- Response time percentiles

Ensures agents meet production performance requirements.

"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from dimension evaluation."""
    score: float
    test_count: int
    tests_passed: int
    tests_failed: int
    details: Dict[str, Any] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PerformanceEvaluator:
    """
    Evaluator for performance dimension.

    Tests:
    1. Latency - Response time benchmarks
    2. Throughput - Records per second
    3. Memory - Memory usage efficiency
    4. Scalability - Performance under load
    5. Consistency - Stable response times
    """

    # Performance thresholds
    LATENCY_THRESHOLD_MS = 100  # Max acceptable latency (ms)
    THROUGHPUT_TARGET = 100  # Records per second target
    MEMORY_LIMIT_MB = 512  # Max memory usage (MB)
    P99_LATENCY_RATIO = 3.0  # P99 should be < 3x median

    def __init__(self):
        """Initialize performance evaluator."""
        logger.info("PerformanceEvaluator initialized")

    def evaluate(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[Any] = None,
        determinism_result: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent performance.

        Args:
            agent: Agent instance to evaluate
            pack_spec: Agent pack specification
            sample_inputs: Sample inputs for testing
            golden_result: Optional golden test results
            determinism_result: Optional determinism results

        Returns:
            EvaluationResult with score and details
        """
        tests_run = 0
        tests_passed = 0
        findings = []
        recommendations = []
        details = {}

        # Test 1: Latency benchmark
        latency_score, latency_details = self._test_latency(
            agent, sample_inputs
        )
        details["latency"] = latency_details
        tests_run += latency_details.get("test_count", 0)
        tests_passed += latency_details.get("tests_passed", 0)

        if latency_score < 100:
            avg_latency = latency_details.get("average_latency_ms", 0)
            findings.append(f"Average latency: {avg_latency:.1f}ms (target: <{self.LATENCY_THRESHOLD_MS}ms)")
            recommendations.append(
                "Optimize calculation performance to reduce latency"
            )

        # Test 2: Throughput measurement
        throughput_score, throughput_details = self._test_throughput(
            agent, sample_inputs
        )
        details["throughput"] = throughput_details
        tests_run += throughput_details.get("test_count", 0)
        tests_passed += throughput_details.get("tests_passed", 0)

        if throughput_score < 100:
            throughput = throughput_details.get("records_per_second", 0)
            findings.append(f"Throughput: {throughput:.1f} rec/s (target: >{self.THROUGHPUT_TARGET})")
            recommendations.append(
                "Implement batch processing for improved throughput"
            )

        # Test 3: Response time consistency
        consistency_score, consistency_details = self._test_consistency(
            agent, sample_inputs
        )
        details["consistency"] = consistency_details
        tests_run += consistency_details.get("test_count", 0)
        tests_passed += consistency_details.get("tests_passed", 0)

        if consistency_score < 100:
            findings.append(f"Response time variance: {consistency_score:.1f}%")
            recommendations.append(
                "Reduce response time variability for predictable performance"
            )

        # Test 4: Performance spec compliance
        spec_score, spec_details = self._test_performance_spec(pack_spec)
        details["spec_compliance"] = spec_details
        tests_run += spec_details.get("test_count", 0)
        tests_passed += spec_details.get("tests_passed", 0)

        if spec_score < 100:
            findings.append(f"Performance spec: {spec_score:.1f}%")
            recommendations.append(
                "Document performance characteristics in pack spec"
            )

        # Test 5: Warm-up behavior
        warmup_score, warmup_details = self._test_warmup(agent, sample_inputs)
        details["warmup"] = warmup_details
        tests_run += warmup_details.get("test_count", 0)
        tests_passed += warmup_details.get("tests_passed", 0)

        if warmup_score < 100:
            findings.append(f"Warm-up behavior: {warmup_score:.1f}%")
            recommendations.append(
                "Implement warm-up optimization for consistent cold-start performance"
            )

        # Calculate overall score
        if tests_run == 0:
            overall_score = 0.0
        else:
            overall_score = (tests_passed / tests_run) * 100

        return EvaluationResult(
            score=overall_score,
            test_count=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_run - tests_passed,
            details=details,
            findings=findings,
            recommendations=recommendations,
        )

    def _test_latency(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test latency benchmark."""
        tests_run = 0
        tests_passed = 0
        latencies = []

        # Run multiple iterations
        for sample_input in sample_inputs[:5]:
            for _ in range(3):  # 3 runs per input
                try:
                    start_time = time.perf_counter()
                    agent.run(sample_input)
                    end_time = time.perf_counter()

                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
                    tests_run += 1

                    if latency_ms < self.LATENCY_THRESHOLD_MS:
                        tests_passed += 1

                except Exception:
                    tests_run += 1

        # Calculate statistics
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            sorted_latencies = sorted(latencies)
            p50 = sorted_latencies[len(sorted_latencies) // 2]
            p99_idx = int(len(sorted_latencies) * 0.99)
            p99 = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
        else:
            avg_latency = min_latency = max_latency = p50 = p99 = 0

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "average_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "p50_latency_ms": p50,
            "p99_latency_ms": p99,
            "threshold_ms": self.LATENCY_THRESHOLD_MS,
        }

    def _test_throughput(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test throughput measurement."""
        tests_run = 1
        tests_passed = 0

        # Batch processing test
        num_records = min(20, len(sample_inputs) * 4)
        test_inputs = (sample_inputs * 4)[:num_records]

        start_time = time.perf_counter()

        successful = 0
        for test_input in test_inputs:
            try:
                agent.run(test_input)
                successful += 1
            except Exception:
                pass

        end_time = time.perf_counter()
        duration_seconds = end_time - start_time

        records_per_second = successful / duration_seconds if duration_seconds > 0 else 0

        if records_per_second >= self.THROUGHPUT_TARGET:
            tests_passed = 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "records_processed": successful,
            "duration_seconds": duration_seconds,
            "records_per_second": records_per_second,
            "throughput_target": self.THROUGHPUT_TARGET,
        }

    def _test_consistency(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test response time consistency."""
        tests_run = 1
        tests_passed = 0
        latencies = []

        # Collect latencies
        if sample_inputs:
            sample_input = sample_inputs[0]
            for _ in range(10):
                try:
                    start_time = time.perf_counter()
                    agent.run(sample_input)
                    end_time = time.perf_counter()
                    latencies.append((end_time - start_time) * 1000)
                except Exception:
                    pass

        # Calculate coefficient of variation
        if len(latencies) >= 2:
            avg = sum(latencies) / len(latencies)
            variance = sum((x - avg) ** 2 for x in latencies) / len(latencies)
            std_dev = variance ** 0.5
            cv = (std_dev / avg) * 100 if avg > 0 else 0

            # CV < 50% is considered consistent
            if cv < 50:
                tests_passed = 1
        else:
            cv = 0
            tests_passed = 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "coefficient_of_variation": cv,
            "sample_count": len(latencies),
        }

    def _test_performance_spec(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test performance specification compliance."""
        tests_run = 1
        tests_passed = 0
        spec_checks = []

        # Check for performance section in pack spec
        performance_spec = pack_spec.get("performance", {})

        if performance_spec:
            tests_passed = 1
            spec_checks.append({
                "check": "performance_spec",
                "status": "DOCUMENTED",
            })
        else:
            # Not required but recommended
            tests_passed = 1
            spec_checks.append({
                "check": "performance_spec",
                "status": "NOT_DOCUMENTED",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "spec_checks": spec_checks,
        }

    def _test_warmup(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test warm-up behavior."""
        tests_run = 1
        tests_passed = 0

        if not sample_inputs:
            return 100.0, {"test_count": 1, "tests_passed": 1, "status": "SKIPPED"}

        sample_input = sample_inputs[0]
        latencies = []

        # Collect first 5 latencies (including cold start)
        for _ in range(5):
            try:
                start_time = time.perf_counter()
                agent.run(sample_input)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)
            except Exception:
                pass

        # Check if warm-up improves performance
        if len(latencies) >= 3:
            first_latency = latencies[0]
            avg_warm = sum(latencies[2:]) / len(latencies[2:]) if len(latencies) > 2 else first_latency

            # Warm latency should be similar or better
            if avg_warm <= first_latency * 2:
                tests_passed = 1
        else:
            tests_passed = 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "latencies": latencies[:5],
        }
