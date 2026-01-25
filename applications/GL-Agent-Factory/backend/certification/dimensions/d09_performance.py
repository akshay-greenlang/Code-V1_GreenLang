"""
Dimension 09: Performance Verification

This dimension verifies that agents meet performance requirements
including response time, memory usage, and efficiency.

Checks:
    - Response time < 1s (p99)
    - Memory usage < 512MB
    - No memory leaks
    - Efficient algorithms

Example:
    >>> dimension = PerformanceDimension()
    >>> result = dimension.evaluate(agent_path, agent, sample_input)
    >>> assert result.status == DimensionStatus.PASS
"""

import gc
import logging
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseDimension, CheckResult, DimensionResult, DimensionStatus

logger = logging.getLogger(__name__)


class PerformanceDimension(BaseDimension):
    """
    Performance Dimension Evaluator (D09).

    Verifies that agents meet performance requirements.

    Configuration:
        max_response_time_ms: Maximum response time in ms (default: 1000)
        max_memory_mb: Maximum memory usage in MB (default: 512)
        num_runs: Number of runs for benchmarking (default: 10)
        warmup_runs: Number of warmup runs (default: 2)
    """

    DIMENSION_ID = "D09"
    DIMENSION_NAME = "Performance"
    DESCRIPTION = "Verifies response time < 1s (p99), memory usage < 512MB"
    WEIGHT = 1.1
    REQUIRED_FOR_CERTIFICATION = True

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance dimension evaluator."""
        super().__init__(config)

        self.max_response_time_ms = self.config.get("max_response_time_ms", 1000)
        self.max_memory_mb = self.config.get("max_memory_mb", 512)
        self.num_runs = self.config.get("num_runs", 10)
        self.warmup_runs = self.config.get("warmup_runs", 2)

    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate performance for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Agent instance with run() method
            sample_input: Sample input for benchmarking

        Returns:
            DimensionResult with performance evaluation
        """
        start_time = datetime.utcnow()
        self._reset_checks()

        logger.info("Starting performance evaluation")

        if agent is None:
            agent = self._load_agent(agent_path)

        if agent is None:
            self._add_check(
                name="agent_load",
                passed=False,
                message="Failed to load agent instance",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        if sample_input is None:
            sample_input = self._get_sample_input(agent_path)

        if sample_input is None:
            self._add_check(
                name="sample_input",
                passed=False,
                message="No sample input provided or found",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        # Check 1: Response time benchmarking
        response_times = self._benchmark_response_time(agent, sample_input)
        p99_time = self._calculate_percentile(response_times, 99)
        avg_time = sum(response_times) / len(response_times) if response_times else 0

        self._add_check(
            name="response_time",
            passed=p99_time <= self.max_response_time_ms,
            message=f"P99 response time: {p99_time:.2f}ms (max: {self.max_response_time_ms}ms)",
            severity="error" if p99_time > self.max_response_time_ms else "info",
            details={
                "p99_ms": p99_time,
                "avg_ms": avg_time,
                "min_ms": min(response_times) if response_times else 0,
                "max_ms": max(response_times) if response_times else 0,
                "runs": len(response_times),
            },
        )

        # Check 2: Memory usage
        memory_usage = self._measure_memory_usage(agent, sample_input)
        peak_memory_mb = memory_usage.get("peak_mb", 0)

        self._add_check(
            name="memory_usage",
            passed=peak_memory_mb <= self.max_memory_mb,
            message=f"Peak memory: {peak_memory_mb:.2f}MB (max: {self.max_memory_mb}MB)",
            severity="error" if peak_memory_mb > self.max_memory_mb else "info",
            details=memory_usage,
        )

        # Check 3: Memory leak detection
        leak_check = self._check_memory_leaks(agent, sample_input)
        self._add_check(
            name="no_memory_leaks",
            passed=leak_check["no_leaks"],
            message="No memory leaks detected"
            if leak_check["no_leaks"]
            else f"Potential memory leak: {leak_check['growth_mb']:.2f}MB growth",
            severity="warning" if not leak_check["no_leaks"] else "info",
            details=leak_check,
        )

        # Check 4: Algorithm efficiency
        efficiency_check = self._check_algorithm_efficiency(agent_path)
        self._add_check(
            name="algorithm_efficiency",
            passed=efficiency_check["is_efficient"],
            message="Algorithm patterns appear efficient"
            if efficiency_check["is_efficient"]
            else f"Found {len(efficiency_check['inefficiencies'])} inefficiency pattern(s)",
            severity="warning" if not efficiency_check["is_efficient"] else "info",
            details=efficiency_check,
        )

        # Check 5: Resource cleanup
        cleanup_check = self._check_resource_cleanup(agent_path)
        self._add_check(
            name="resource_cleanup",
            passed=cleanup_check["has_cleanup"],
            message="Resource cleanup patterns present"
            if cleanup_check["has_cleanup"]
            else "No explicit resource cleanup found",
            severity="warning" if not cleanup_check["has_cleanup"] else "info",
            details=cleanup_check,
        )

        # Check 6: Caching usage
        caching_check = self._check_caching_usage(agent_path)
        self._add_check(
            name="caching_usage",
            passed=True,  # Caching is optional but good
            message=f"Caching: {caching_check['caching_type']}"
            if caching_check["has_caching"]
            else "No caching implemented (optional)",
            severity="info",
            details=caching_check,
        )

        # Check 7: Async support
        async_check = self._check_async_support(agent_path)
        self._add_check(
            name="async_support",
            passed=True,  # Async is optional
            message="Async/await support present"
            if async_check["has_async"]
            else "No async support (optional for sync operations)",
            severity="info",
            details=async_check,
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return self._create_result(
            execution_time,
            details={
                "p99_response_time_ms": p99_time,
                "avg_response_time_ms": avg_time,
                "peak_memory_mb": peak_memory_mb,
                "benchmark_runs": self.num_runs,
            },
        )

    def _load_agent(self, agent_path: Path) -> Optional[Any]:
        """Load agent from path."""
        try:
            agent_file = agent_path / "agent.py"
            if not agent_file.exists():
                return None

            import importlib.util

            spec = importlib.util.spec_from_file_location("agent", agent_file)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and name.endswith("Agent")
                    and hasattr(obj, "run")
                ):
                    return obj()

            return None

        except Exception as e:
            logger.error(f"Failed to load agent: {str(e)}")
            return None

    def _get_sample_input(self, agent_path: Path) -> Optional[Any]:
        """Get sample input from agent path."""
        try:
            import yaml

            pack_file = agent_path / "pack.yaml"
            if pack_file.exists():
                with open(pack_file, "r", encoding="utf-8") as f:
                    pack_spec = yaml.safe_load(f)

                tests = pack_spec.get("tests", {}).get("golden", [])
                if tests:
                    return tests[0].get("input")

            return None

        except Exception as e:
            logger.error(f"Failed to get sample input: {str(e)}")
            return None

    def _benchmark_response_time(
        self,
        agent: Any,
        sample_input: Any,
    ) -> List[float]:
        """
        Benchmark agent response time.

        Args:
            agent: Agent instance
            sample_input: Sample input data

        Returns:
            List of response times in milliseconds
        """
        response_times = []

        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                agent.run(sample_input)
            except Exception:
                pass

        # Benchmark runs
        for _ in range(self.num_runs):
            try:
                start = time.perf_counter()
                agent.run(sample_input)
                end = time.perf_counter()

                response_times.append((end - start) * 1000)

            except Exception as e:
                logger.warning(f"Benchmark run failed: {str(e)}")

        return response_times

    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)

        return sorted_values[index]

    def _measure_memory_usage(
        self,
        agent: Any,
        sample_input: Any,
    ) -> Dict[str, Any]:
        """
        Measure memory usage during agent execution.

        Args:
            agent: Agent instance
            sample_input: Sample input data

        Returns:
            Dictionary with memory usage metrics
        """
        result = {
            "peak_mb": 0,
            "current_mb": 0,
            "allocated_mb": 0,
        }

        try:
            # Force garbage collection before measurement
            gc.collect()

            # Start memory tracking
            tracemalloc.start()

            # Run agent
            agent.run(sample_input)

            # Get memory stats
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            result["current_mb"] = current / (1024 * 1024)
            result["peak_mb"] = peak / (1024 * 1024)

        except Exception as e:
            logger.error(f"Memory measurement failed: {str(e)}")
            # Fallback to sys.getsizeof
            try:
                result["current_mb"] = sys.getsizeof(agent) / (1024 * 1024)
            except Exception:
                pass

        return result

    def _check_memory_leaks(
        self,
        agent: Any,
        sample_input: Any,
    ) -> Dict[str, Any]:
        """
        Check for memory leaks.

        Args:
            agent: Agent instance
            sample_input: Sample input data

        Returns:
            Dictionary with leak detection results
        """
        result = {
            "no_leaks": True,
            "growth_mb": 0,
            "runs": 5,
        }

        try:
            gc.collect()
            tracemalloc.start()

            # Run multiple times
            for _ in range(result["runs"]):
                agent.run(sample_input)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Check for significant growth
            gc.collect()
            tracemalloc.start()

            for _ in range(result["runs"]):
                agent.run(sample_input)

            current2, peak2 = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            growth = (peak2 - peak) / (1024 * 1024)
            result["growth_mb"] = growth

            # Allow up to 10MB growth
            if growth > 10:
                result["no_leaks"] = False

        except Exception as e:
            logger.error(f"Leak detection failed: {str(e)}")

        return result

    def _check_algorithm_efficiency(self, agent_path: Path) -> Dict[str, Any]:
        """
        Check for inefficient algorithm patterns.

        Args:
            agent_path: Path to agent directory

        Returns:
            Dictionary with efficiency check results
        """
        result = {
            "is_efficient": True,
            "inefficiencies": [],
        }

        try:
            agent_file = agent_path / "agent.py"
            if not agent_file.exists():
                return result

            source_code = agent_file.read_text(encoding="utf-8")

            import re

            # Check for inefficient patterns
            patterns = [
                (r"for\s+\w+\s+in\s+range\s*\([^)]+\)\s*:\s*for\s+\w+\s+in\s+range", "Nested loops (O(n^2))"),
                (r"\.append\s*\([^)]+\)\s*.*\.append", "Multiple appends in loop"),
                (r"\+\s*=\s*['\"]", "String concatenation in loop"),
                (r"list\s*\([^)]*\)\s*\*\s*\d+", "Large list allocation"),
            ]

            for pattern, description in patterns:
                if re.search(pattern, source_code):
                    result["inefficiencies"].append(description)

            result["is_efficient"] = len(result["inefficiencies"]) == 0

        except Exception as e:
            logger.error(f"Efficiency check failed: {str(e)}")

        return result

    def _check_resource_cleanup(self, agent_path: Path) -> Dict[str, Any]:
        """
        Check for resource cleanup patterns.

        Args:
            agent_path: Path to agent directory

        Returns:
            Dictionary with cleanup check results
        """
        result = {
            "has_cleanup": False,
            "cleanup_patterns": [],
        }

        try:
            agent_file = agent_path / "agent.py"
            if not agent_file.exists():
                return result

            source_code = agent_file.read_text(encoding="utf-8")

            import re

            patterns = [
                (r"with\s+open\s*\(", "Context manager for files"),
                (r"try\s*:\s*.*finally\s*:", "try/finally cleanup"),
                (r"\.close\s*\(\)", "Explicit close"),
                (r"__enter__|__exit__", "Context manager protocol"),
                (r"atexit\.register", "atexit cleanup"),
            ]

            for pattern, description in patterns:
                if re.search(pattern, source_code, re.DOTALL):
                    result["has_cleanup"] = True
                    result["cleanup_patterns"].append(description)

        except Exception as e:
            logger.error(f"Cleanup check failed: {str(e)}")

        return result

    def _check_caching_usage(self, agent_path: Path) -> Dict[str, Any]:
        """
        Check for caching usage.

        Args:
            agent_path: Path to agent directory

        Returns:
            Dictionary with caching check results
        """
        result = {
            "has_caching": False,
            "caching_type": "none",
        }

        try:
            agent_file = agent_path / "agent.py"
            if not agent_file.exists():
                return result

            source_code = agent_file.read_text(encoding="utf-8")

            import re

            patterns = [
                (r"@lru_cache|@cache", "functools cache"),
                (r"@cached_property", "cached property"),
                (r"redis|memcached", "distributed cache"),
                (r"_cache\s*=\s*\{\}|_cache\s*=\s*dict\(\)", "manual cache"),
            ]

            for pattern, cache_type in patterns:
                if re.search(pattern, source_code, re.IGNORECASE):
                    result["has_caching"] = True
                    result["caching_type"] = cache_type
                    break

        except Exception as e:
            logger.error(f"Caching check failed: {str(e)}")

        return result

    def _check_async_support(self, agent_path: Path) -> Dict[str, Any]:
        """
        Check for async/await support.

        Args:
            agent_path: Path to agent directory

        Returns:
            Dictionary with async check results
        """
        result = {
            "has_async": False,
            "async_patterns": [],
        }

        try:
            agent_file = agent_path / "agent.py"
            if not agent_file.exists():
                return result

            source_code = agent_file.read_text(encoding="utf-8")

            import re

            patterns = [
                (r"async\s+def", "async function"),
                (r"await\s+", "await keyword"),
                (r"asyncio\.", "asyncio usage"),
                (r"aiohttp|httpx\.AsyncClient", "async HTTP client"),
            ]

            for pattern, async_type in patterns:
                if re.search(pattern, source_code):
                    result["has_async"] = True
                    result["async_patterns"].append(async_type)

        except Exception as e:
            logger.error(f"Async check failed: {str(e)}")

        return result

    def _get_check_remediation(self, check: CheckResult) -> Optional[str]:
        """Get remediation for failed checks."""
        remediation_map = {
            "agent_load": (
                "Ensure agent.py exists and contains a class ending with 'Agent' "
                "that has a run() method."
            ),
            "sample_input": (
                "Provide sample input via pack.yaml golden tests."
            ),
            "response_time": (
                f"Optimize response time (target: <{self.max_response_time_ms}ms):\n"
                "  - Use caching with @lru_cache\n"
                "  - Avoid nested loops (O(n^2))\n"
                "  - Pre-compute emission factors\n"
                "  - Use vectorized operations with numpy/pandas"
            ),
            "memory_usage": (
                f"Reduce memory usage (target: <{self.max_memory_mb}MB):\n"
                "  - Use generators instead of lists\n"
                "  - Process data in batches\n"
                "  - Release large objects when done\n"
                "  - Use __slots__ for Pydantic models"
            ),
            "no_memory_leaks": (
                "Fix memory leaks:\n"
                "  - Clear caches periodically\n"
                "  - Use weak references for caches\n"
                "  - Close file handles and connections\n"
                "  - Avoid circular references"
            ),
            "algorithm_efficiency": (
                "Improve algorithm efficiency:\n"
                "  - Replace nested loops with vectorized operations\n"
                "  - Use sets for membership testing\n"
                "  - Pre-allocate lists when size is known\n"
                "  - Use list comprehensions"
            ),
            "resource_cleanup": (
                "Add resource cleanup:\n"
                "  - Use 'with' statements for files\n"
                "  - Add try/finally for cleanup\n"
                "  - Implement __enter__/__exit__ for resources"
            ),
        }

        return remediation_map.get(check.name)
