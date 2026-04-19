# Benchmarking Framework

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** Active
**Owner:** GreenLang Quality Engineering & Performance Team

---

## Executive Summary

This document defines the comprehensive benchmarking framework for evaluating GreenLang agent performance across four dimensions: **Performance** (latency, throughput), **Accuracy** (calculation correctness), **Cost** (LLM token usage), and **Quality** (explanation, reasoning). Benchmarking ensures agents meet production targets and enables data-driven optimization.

**Core Principle:** Measure everything, optimize what matters.

---

## Benchmark Categories

### 1. Performance Benchmarks (Speed & Scale)
### 2. Accuracy Benchmarks (Correctness)
### 3. Cost Benchmarks (Economics)
### 4. Quality Benchmarks (User Experience)

---

## 1. Performance Benchmarks

### Objective

Measure agent latency, throughput, and scalability to ensure sub-4-second response times and >100 req/s sustained throughput.

### Metrics

| Metric | Definition | Target | Measurement |
|--------|------------|--------|-------------|
| P50 Latency | Median response time | <2.0s | 50th percentile over 1000 runs |
| P95 Latency | 95th percentile response time | <4.0s | 95th percentile over 1000 runs |
| P99 Latency | 99th percentile response time | <6.0s | 99th percentile over 1000 runs |
| Throughput (Steady) | Sustained requests/second | >100 req/s | 1 hour load test |
| Throughput (Peak) | Peak requests/second | >500 req/s | 5 minute spike test |
| Time to First Byte (TTFB) | Time until first response data | <500ms | WebSocket/streaming |
| Memory Usage | RAM consumption per request | <512 MB | Pod monitoring |
| CPU Usage | CPU cores per request | <1 core | Pod monitoring |

### Benchmark Implementation

```python
# benchmarks/boiler_efficiency_optimizer_performance.py

import pytest
import time
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from greenlang.agents.boiler_efficiency_optimizer import BoilerEfficiencyOptimizer


class PerformanceBenchmark:
    """Performance benchmark suite for agents."""

    def __init__(self, agent_class, agent_config: Dict[str, Any]):
        self.agent_class = agent_class
        self.agent_config = agent_config
        self.results = []

    def measure_latency(
        self,
        test_input: Dict[str, Any],
        num_runs: int = 1000
    ) -> Dict[str, float]:
        """
        Measure agent latency distribution (P50, P95, P99).

        Args:
            test_input: Input parameters for agent
            num_runs: Number of runs for statistical significance

        Returns:
            Dict with P50, P95, P99 latency in seconds
        """
        agent = self.agent_class(**self.agent_config)
        latencies = []

        print(f"Running {num_runs} latency measurements...")

        for i in range(num_runs):
            start_time = time.perf_counter()

            result = agent.calculate_boiler_efficiency(**test_input)

            end_time = time.perf_counter()
            latency_seconds = end_time - start_time
            latencies.append(latency_seconds)

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{num_runs} runs completed")

        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[int(0.50 * num_runs)]
        p95 = latencies_sorted[int(0.95 * num_runs)]
        p99 = latencies_sorted[int(0.99 * num_runs)]

        results = {
            "p50_latency_seconds": p50,
            "p95_latency_seconds": p95,
            "p99_latency_seconds": p99,
            "mean_latency_seconds": statistics.mean(latencies),
            "stdev_latency_seconds": statistics.stdev(latencies),
            "min_latency_seconds": min(latencies),
            "max_latency_seconds": max(latencies)
        }

        print("\nLatency Results:")
        print(f"  P50: {p50:.3f}s")
        print(f"  P95: {p95:.3f}s (Target: <4.0s)")
        print(f"  P99: {p99:.3f}s (Target: <6.0s)")
        print(f"  Mean: {results['mean_latency_seconds']:.3f}s")
        print(f"  StdDev: {results['stdev_latency_seconds']:.3f}s")

        return results

    def measure_throughput_steady_state(
        self,
        test_input: Dict[str, Any],
        target_rps: int = 100,
        duration_seconds: int = 3600
    ) -> Dict[str, float]:
        """
        Measure sustained throughput (steady state load test).

        Args:
            test_input: Input parameters for agent
            target_rps: Target requests per second
            duration_seconds: Test duration in seconds (default: 1 hour)

        Returns:
            Dict with actual RPS, success rate, error rate
        """
        agent = self.agent_class(**self.agent_config)

        start_time = time.time()
        end_time = start_time + duration_seconds

        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        latencies = []

        print(f"Running steady state load test: {target_rps} req/s for {duration_seconds}s...")

        with ThreadPoolExecutor(max_workers=target_rps) as executor:
            while time.time() < end_time:
                batch_start = time.time()

                # Submit batch of requests
                futures = []
                for _ in range(target_rps):
                    future = executor.submit(
                        self._execute_request,
                        agent,
                        test_input
                    )
                    futures.append(future)

                # Wait for batch to complete
                for future in as_completed(futures):
                    total_requests += 1
                    try:
                        result, latency = future.result()
                        successful_requests += 1
                        latencies.append(latency)
                    except Exception as e:
                        failed_requests += 1
                        print(f"  Request failed: {e}")

                # Sleep to maintain target RPS
                batch_duration = time.time() - batch_start
                sleep_time = max(0, 1.0 - batch_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Progress update every 60 seconds
                if total_requests % (target_rps * 60) == 0:
                    elapsed = time.time() - start_time
                    print(f"  Progress: {elapsed:.0f}s / {duration_seconds}s, "
                          f"{total_requests} requests")

        actual_duration = time.time() - start_time
        actual_rps = total_requests / actual_duration
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0

        results = {
            "target_rps": target_rps,
            "actual_rps": actual_rps,
            "duration_seconds": actual_duration,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "p50_latency_seconds": statistics.median(latencies) if latencies else 0,
            "p95_latency_seconds": sorted(latencies)[int(0.95*len(latencies))] if latencies else 0
        }

        print("\nSteady State Throughput Results:")
        print(f"  Target RPS: {target_rps}")
        print(f"  Actual RPS: {actual_rps:.1f}")
        print(f"  Success Rate: {success_rate*100:.1f}%")
        print(f"  Error Rate: {error_rate*100:.1f}%")
        print(f"  P50 Latency: {results['p50_latency_seconds']:.3f}s")
        print(f"  P95 Latency: {results['p95_latency_seconds']:.3f}s")

        return results

    def measure_throughput_spike_test(
        self,
        test_input: Dict[str, Any],
        spike_rps: int = 500,
        duration_seconds: int = 300
    ) -> Dict[str, float]:
        """
        Measure peak throughput (spike load test).

        Ramps from 0 → spike_rps → 0 over duration_seconds.
        """
        print(f"Running spike test: 0 → {spike_rps} → 0 req/s over {duration_seconds}s...")

        # Implementation similar to steady state but with ramping load
        # Omitted for brevity
        pass

    def _execute_request(
        self,
        agent,
        test_input: Dict[str, Any]
    ) -> tuple[Dict[str, Any], float]:
        """Execute a single agent request and measure latency."""
        start_time = time.perf_counter()

        result = agent.calculate_boiler_efficiency(**test_input)

        end_time = time.perf_counter()
        latency = end_time - start_time

        return result, latency


# Pytest integration
@pytest.fixture
def performance_benchmark():
    """Create performance benchmark suite."""
    return PerformanceBenchmark(
        agent_class=BoilerEfficiencyOptimizer,
        agent_config={"temperature": 0.0, "seed": 42}
    )


@pytest.mark.performance
def test_latency_under_4_seconds(performance_benchmark):
    """Test that P95 latency is <4 seconds."""
    test_input = {
        "fuel_type": "natural_gas",
        "firing_rate_mmbtu_hr": 15.0,
        "flue_gas_temp_f": 350.0,
        "ambient_temp_f": 70.0
    }

    results = performance_benchmark.measure_latency(
        test_input=test_input,
        num_runs=1000
    )

    # Assert P95 latency <4s
    assert results['p95_latency_seconds'] < 4.0, \
        f"P95 latency {results['p95_latency_seconds']:.3f}s exceeds 4.0s target"

    # Assert P99 latency <6s
    assert results['p99_latency_seconds'] < 6.0, \
        f"P99 latency {results['p99_latency_seconds']:.3f}s exceeds 6.0s target"


@pytest.mark.performance
@pytest.mark.slow
def test_throughput_100_rps_sustained(performance_benchmark):
    """Test sustained throughput of 100 req/s for 1 hour."""
    test_input = {
        "fuel_type": "natural_gas",
        "firing_rate_mmbtu_hr": 15.0,
        "flue_gas_temp_f": 350.0,
        "ambient_temp_f": 70.0
    }

    results = performance_benchmark.measure_throughput_steady_state(
        test_input=test_input,
        target_rps=100,
        duration_seconds=3600  # 1 hour
    )

    # Assert actual RPS >100
    assert results['actual_rps'] >= 100, \
        f"Actual RPS {results['actual_rps']:.1f} below 100 req/s target"

    # Assert success rate >99%
    assert results['success_rate'] >= 0.99, \
        f"Success rate {results['success_rate']*100:.1f}% below 99% target"
```

### Load Testing with Locust

```python
# benchmarks/locustfile_boiler_efficiency.py

from locust import HttpUser, task, between
import json


class BoilerEfficiencyUser(HttpUser):
    """Locust user for load testing BoilerEfficiencyOptimizer agent."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    @task
    def calculate_boiler_efficiency(self):
        """Task: Calculate boiler efficiency."""
        payload = {
            "fuel_type": "natural_gas",
            "firing_rate_mmbtu_hr": 15.0,
            "flue_gas_temp_f": 350.0,
            "ambient_temp_f": 70.0,
            "feedwater_temp_f": 180.0,
            "steam_pressure_psig": 150.0,
            "blowdown_rate": 0.05
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.environment.parsed_options.api_token}"
        }

        with self.client.post(
            "/agents/boiler-efficiency-optimizer/calculate",
            data=json.dumps(payload),
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if 'efficiency_percent' in result:
                    response.success()
                else:
                    response.failure("Missing efficiency_percent in response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
```

Run locust:
```bash
# Steady state: 100 req/s for 1 hour
locust -f benchmarks/locustfile_boiler_efficiency.py \
       --host=https://api.greenlang.com \
       --users=100 \
       --spawn-rate=10 \
       --run-time=1h \
       --headless \
       --api-token=$GREENLANG_API_TOKEN

# Spike test: 0 → 500 → 0 req/s over 5 minutes
locust -f benchmarks/locustfile_boiler_efficiency.py \
       --host=https://api.greenlang.com \
       --users=500 \
       --spawn-rate=100 \
       --run-time=5m \
       --headless \
       --api-token=$GREENLANG_API_TOKEN
```

---

## 2. Accuracy Benchmarks

### Objective

Validate that agent calculations match known correct answers within acceptable tolerances across diverse scenarios.

### Metrics

| Metric | Definition | Target | Measurement |
|--------|------------|--------|-------------|
| Golden Test Pass Rate | % of golden tests passing | 100% | 25+ golden tests |
| Mean Absolute Error (MAE) | Average error vs. ground truth | <1% | Golden test results |
| Root Mean Square Error (RMSE) | RMS error vs. ground truth | <2% | Golden test results |
| Max Absolute Error | Worst-case error | <5% | Golden test results |
| Energy Balance Error | Conservation of energy violation | <0.1% | Thermodynamic tests |
| Mass Balance Error | Conservation of mass violation | <0.1% | Chemical process tests |

### Benchmark Implementation

```python
# benchmarks/boiler_efficiency_optimizer_accuracy.py

import pytest
import numpy as np
from typing import List, Dict, Any
import json

from greenlang.agents.boiler_efficiency_optimizer import BoilerEfficiencyOptimizer


class AccuracyBenchmark:
    """Accuracy benchmark suite for agents."""

    def __init__(self, agent_class, golden_tests_file: str):
        self.agent_class = agent_class
        self.golden_tests = self.load_golden_tests(golden_tests_file)

    def load_golden_tests(self, file_path: str) -> List[Dict[str, Any]]:
        """Load golden test cases from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['golden_tests']

    def run_all_golden_tests(self) -> Dict[str, Any]:
        """
        Run all golden tests and calculate accuracy metrics.

        Returns:
            Dict with pass rate, MAE, RMSE, max error
        """
        agent = self.agent_class(temperature=0.0, seed=42)

        results = []
        errors = []

        print(f"Running {len(self.golden_tests)} golden tests...")

        for i, test in enumerate(self.golden_tests):
            print(f"  Golden Test {i+1}/{len(self.golden_tests)}: {test['test_id']}")

            # Execute agent
            actual_result = agent.calculate_boiler_efficiency(**test['inputs'])

            # Compare to expected output
            expected_efficiency = test['expected_outputs']['efficiency_percent']
            actual_efficiency = actual_result['efficiency_percent']

            # Calculate error
            absolute_error = abs(actual_efficiency - expected_efficiency)
            relative_error = absolute_error / expected_efficiency

            test_passed = relative_error < test['tolerance']['efficiency_percent']['value']

            results.append({
                "test_id": test['test_id'],
                "expected": expected_efficiency,
                "actual": actual_efficiency,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "passed": test_passed
            })

            errors.append(relative_error)

        # Calculate aggregate metrics
        pass_count = sum(1 for r in results if r['passed'])
        pass_rate = pass_count / len(results) if results else 0

        mae = np.mean([r['absolute_error'] for r in results])
        rmse = np.sqrt(np.mean([r['absolute_error']**2 for r in results]))
        max_error = max([r['absolute_error'] for r in results])
        mean_relative_error = np.mean(errors)

        summary = {
            "total_tests": len(results),
            "passed": pass_count,
            "failed": len(results) - pass_count,
            "pass_rate": pass_rate,
            "mean_absolute_error": mae,
            "rmse": rmse,
            "max_absolute_error": max_error,
            "mean_relative_error": mean_relative_error,
            "results": results
        }

        print("\nAccuracy Benchmark Results:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']} ({pass_rate*100:.1f}%)")
        print(f"  Failed: {summary['failed']}")
        print(f"  Mean Absolute Error: {mae:.4f}%")
        print(f"  RMSE: {rmse:.4f}%")
        print(f"  Max Absolute Error: {max_error:.4f}%")
        print(f"  Mean Relative Error: {mean_relative_error*100:.2f}%")

        return summary


@pytest.mark.accuracy
def test_golden_tests_100_percent_pass_rate():
    """Test that all golden tests pass (100% pass rate)."""
    benchmark = AccuracyBenchmark(
        agent_class=BoilerEfficiencyOptimizer,
        golden_tests_file="tests/agents/fixtures/boiler_efficiency_optimizer_golden_results.json"
    )

    results = benchmark.run_all_golden_tests()

    # Assert 100% pass rate
    assert results['pass_rate'] == 1.0, \
        f"Golden test pass rate {results['pass_rate']*100:.1f}% is not 100%"

    # Assert MAE <1%
    assert results['mean_absolute_error'] < 1.0, \
        f"Mean absolute error {results['mean_absolute_error']:.2f}% exceeds 1% target"

    # Assert max error <5%
    assert results['max_absolute_error'] < 5.0, \
        f"Max absolute error {results['max_absolute_error']:.2f}% exceeds 5% target"
```

---

## 3. Cost Benchmarks

### Objective

Measure LLM token usage and cost per analysis to ensure agents remain economically viable (<$0.15 per analysis).

### Metrics

| Metric | Definition | Target | Measurement |
|--------|------------|--------|-------------|
| Cost per Analysis | Average cost per agent execution | <$0.15 | Token counting + pricing |
| Prompt Tokens | Tokens in input prompt | <5,000 | Token counting |
| Completion Tokens | Tokens in output completion | <2,000 | Token counting |
| Total Tokens | Prompt + completion tokens | <7,000 | Token counting |
| Cost per Tool | Cost per tool execution | <$0.03 | Tool-level tracking |
| Tool Call Count | Number of tools called | <8 | Tool call counting |

### Anthropic Claude Pricing (as of 2025-12-03)

| Model | Prompt Tokens | Completion Tokens |
|-------|---------------|-------------------|
| Claude Sonnet 4 | $3.00 / 1M tokens | $15.00 / 1M tokens |
| Claude Haiku 3.5 | $0.80 / 1M tokens | $4.00 / 1M tokens |

### Benchmark Implementation

```python
# benchmarks/boiler_efficiency_optimizer_cost.py

import pytest
from typing import Dict, Any
import tiktoken

from greenlang.agents.boiler_efficiency_optimizer import BoilerEfficiencyOptimizer


class CostBenchmark:
    """Cost benchmark suite for agents."""

    # Anthropic Claude pricing ($/1M tokens)
    PRICING = {
        "claude-sonnet-4": {
            "prompt": 3.00,
            "completion": 15.00
        },
        "claude-haiku-3.5": {
            "prompt": 0.80,
            "completion": 4.00
        }
    }

    def __init__(self, agent_class, model: str = "claude-sonnet-4"):
        self.agent_class = agent_class
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Approximation

    def measure_cost_per_analysis(
        self,
        test_input: Dict[str, Any],
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Measure average cost per agent analysis.

        Args:
            test_input: Input parameters for agent
            num_runs: Number of runs for averaging

        Returns:
            Dict with cost metrics (cost/analysis, tokens, tool calls)
        """
        agent = self.agent_class(
            temperature=0.0,
            seed=42,
            model=self.model,
            track_usage=True
        )

        total_cost = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tool_calls = 0

        print(f"Running {num_runs} cost measurements...")

        for i in range(num_runs):
            result = agent.calculate_boiler_efficiency(**test_input)

            # Extract usage from result
            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            tool_calls = usage.get('tool_calls', 0)

            # Calculate cost
            cost = self.calculate_cost(prompt_tokens, completion_tokens)

            total_cost += cost
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tool_calls += tool_calls

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_runs} runs completed")

        avg_cost = total_cost / num_runs
        avg_prompt_tokens = total_prompt_tokens / num_runs
        avg_completion_tokens = total_completion_tokens / num_runs
        avg_total_tokens = avg_prompt_tokens + avg_completion_tokens
        avg_tool_calls = total_tool_calls / num_runs

        results = {
            "cost_per_analysis_usd": avg_cost,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "avg_total_tokens": avg_total_tokens,
            "avg_tool_calls": avg_tool_calls,
            "model": self.model
        }

        print("\nCost Benchmark Results:")
        print(f"  Cost per Analysis: ${avg_cost:.4f} (Target: <$0.15)")
        print(f"  Avg Prompt Tokens: {avg_prompt_tokens:.0f}")
        print(f"  Avg Completion Tokens: {avg_completion_tokens:.0f}")
        print(f"  Avg Total Tokens: {avg_total_tokens:.0f}")
        print(f"  Avg Tool Calls: {avg_tool_calls:.1f}")
        print(f"  Model: {self.model}")

        return results

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate cost in USD based on token counts."""
        pricing = self.PRICING[self.model]

        prompt_cost = (prompt_tokens / 1_000_000) * pricing['prompt']
        completion_cost = (completion_tokens / 1_000_000) * pricing['completion']

        return prompt_cost + completion_cost


@pytest.mark.cost
def test_cost_under_15_cents(cost_benchmark):
    """Test that cost per analysis is <$0.15."""
    test_input = {
        "fuel_type": "natural_gas",
        "firing_rate_mmbtu_hr": 15.0,
        "flue_gas_temp_f": 350.0,
        "ambient_temp_f": 70.0
    }

    results = cost_benchmark.measure_cost_per_analysis(
        test_input=test_input,
        num_runs=100
    )

    # Assert cost <$0.15
    assert results['cost_per_analysis_usd'] < 0.15, \
        f"Cost ${results['cost_per_analysis_usd']:.4f} exceeds $0.15 target"

    # Assert total tokens <10,000 (safety margin)
    assert results['avg_total_tokens'] < 10000, \
        f"Total tokens {results['avg_total_tokens']:.0f} exceeds 10,000 safety limit"


@pytest.fixture
def cost_benchmark():
    """Create cost benchmark suite."""
    return CostBenchmark(
        agent_class=BoilerEfficiencyOptimizer,
        model="claude-sonnet-4"
    )
```

---

## 4. Quality Benchmarks

### Objective

Assess the quality of agent explanations, reasoning, and user experience through automated metrics and human evaluation.

### Metrics

| Metric | Definition | Target | Measurement |
|--------|------------|--------|-------------|
| Explanation Clarity | Readability score (Flesch-Kincaid) | >60 (college) | NLP analysis |
| Citation Coverage | % of claims with citations | >90% | Citation counting |
| Reasoning Consistency | Logical consistency score | >9.0/10 | LLM evaluation |
| Recommendation Actionability | Specificity score | >8.0/10 | Human evaluation |
| Error Message Clarity | Error clarity score | >8.0/10 | Human evaluation |
| Determinism Rate | % of identical outputs for identical inputs | 100% | Determinism tests |

### Benchmark Implementation

```python
# benchmarks/boiler_efficiency_optimizer_quality.py

import pytest
from typing import Dict, Any
import textstat
import re

from greenlang.agents.boiler_efficiency_optimizer import BoilerEfficiencyOptimizer


class QualityBenchmark:
    """Quality benchmark suite for agents."""

    def __init__(self, agent_class):
        self.agent_class = agent_class

    def measure_explanation_clarity(
        self,
        test_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Measure explanation clarity using readability metrics.

        Returns:
            Dict with Flesch-Kincaid score, grade level, etc.
        """
        agent = self.agent_class(temperature=0.0, seed=42)

        result = agent.calculate_boiler_efficiency(**test_input)
        explanation = result.get('explanation', '')

        # Calculate readability metrics
        flesch_reading_ease = textstat.flesch_reading_ease(explanation)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(explanation)
        smog_index = textstat.smog_index(explanation)
        coleman_liau_index = textstat.coleman_liau_index(explanation)

        # Count sentences, words, syllables
        sentence_count = textstat.sentence_count(explanation)
        word_count = textstat.lexicon_count(explanation)
        syllable_count = textstat.syllable_count(explanation)

        results = {
            "flesch_reading_ease": flesch_reading_ease,
            "flesch_kincaid_grade": flesch_kincaid_grade,
            "smog_index": smog_index,
            "coleman_liau_index": coleman_liau_index,
            "sentence_count": sentence_count,
            "word_count": word_count,
            "syllable_count": syllable_count
        }

        print("\nExplanation Clarity Results:")
        print(f"  Flesch Reading Ease: {flesch_reading_ease:.1f} (Target: >60)")
        print(f"  Flesch-Kincaid Grade Level: {flesch_kincaid_grade:.1f}")
        print(f"  Word Count: {word_count}")
        print(f"  Sentence Count: {sentence_count}")

        return results

    def measure_citation_coverage(
        self,
        test_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Measure citation coverage (% of claims with citations).

        Returns:
            Dict with citation count, claim count, coverage %
        """
        agent = self.agent_class(temperature=0.0, seed=42)

        result = agent.calculate_boiler_efficiency(**test_input)
        explanation = result.get('explanation', '')

        # Count citations (e.g., "[1]", "(Smith 2020)", "ASME PTC 4-2013")
        citation_pattern = r'\[[\d]+\]|\([A-Za-z]+ \d{4}\)|[A-Z]+ [A-Z0-9\-]+'
        citations = re.findall(citation_pattern, explanation)
        citation_count = len(citations)

        # Count claims (sentences ending with factual statements)
        # Simple heuristic: count sentences (actual NLP would be more accurate)
        claim_count = textstat.sentence_count(explanation)

        # Calculate coverage
        coverage = citation_count / claim_count if claim_count > 0 else 0

        results = {
            "citation_count": citation_count,
            "claim_count": claim_count,
            "citation_coverage": coverage
        }

        print("\nCitation Coverage Results:")
        print(f"  Citations: {citation_count}")
        print(f"  Claims: {claim_count}")
        print(f"  Coverage: {coverage*100:.1f}% (Target: >90%)")

        return results


@pytest.mark.quality
def test_explanation_clarity_above_60(quality_benchmark):
    """Test that explanation clarity score is >60 (college level)."""
    test_input = {
        "fuel_type": "natural_gas",
        "firing_rate_mmbtu_hr": 15.0,
        "flue_gas_temp_f": 350.0,
        "ambient_temp_f": 70.0
    }

    results = quality_benchmark.measure_explanation_clarity(test_input)

    # Assert Flesch Reading Ease >60
    assert results['flesch_reading_ease'] >= 60, \
        f"Flesch Reading Ease {results['flesch_reading_ease']:.1f} below 60 target"


@pytest.mark.quality
def test_citation_coverage_above_90_percent(quality_benchmark):
    """Test that citation coverage is >90%."""
    test_input = {
        "fuel_type": "natural_gas",
        "firing_rate_mmbtu_hr": 15.0,
        "flue_gas_temp_f": 350.0,
        "ambient_temp_f": 70.0
    }

    results = quality_benchmark.measure_citation_coverage(test_input)

    # Assert citation coverage >90%
    assert results['citation_coverage'] >= 0.90, \
        f"Citation coverage {results['citation_coverage']*100:.1f}% below 90% target"


@pytest.fixture
def quality_benchmark():
    """Create quality benchmark suite."""
    return QualityBenchmark(agent_class=BoilerEfficiencyOptimizer)
```

---

## Comparative Evaluation Across Models

### Objective

Compare agent performance across different LLM models (Claude Sonnet 4, Haiku 3.5, GPT-4) to identify optimal model for each use case.

### Model Comparison Matrix

| Model | Cost | Latency | Accuracy | Quality | Best Use Case |
|-------|------|---------|----------|---------|---------------|
| Claude Sonnet 4 | $$$ | Medium | Excellent | Excellent | Complex analysis, regulatory compliance |
| Claude Haiku 3.5 | $ | Fast | Good | Good | High-throughput, simple calculations |
| GPT-4 Turbo | $$ | Medium | Excellent | Excellent | Competitive benchmark |

### Implementation

```python
@pytest.mark.parametrize("model", [
    "claude-sonnet-4",
    "claude-haiku-3.5",
    "gpt-4-turbo"
])
def test_model_comparison(model):
    """Compare agent performance across different LLM models."""
    performance_benchmark = PerformanceBenchmark(
        agent_class=BoilerEfficiencyOptimizer,
        agent_config={"temperature": 0.0, "seed": 42, "model": model}
    )

    cost_benchmark = CostBenchmark(
        agent_class=BoilerEfficiencyOptimizer,
        model=model
    )

    accuracy_benchmark = AccuracyBenchmark(
        agent_class=BoilerEfficiencyOptimizer,
        golden_tests_file="tests/agents/fixtures/boiler_efficiency_optimizer_golden_results.json"
    )

    test_input = {
        "fuel_type": "natural_gas",
        "firing_rate_mmbtu_hr": 15.0,
        "flue_gas_temp_f": 350.0,
        "ambient_temp_f": 70.0
    }

    # Measure performance
    perf_results = performance_benchmark.measure_latency(test_input, num_runs=100)

    # Measure cost
    cost_results = cost_benchmark.measure_cost_per_analysis(test_input, num_runs=100)

    # Measure accuracy
    accuracy_results = accuracy_benchmark.run_all_golden_tests()

    print(f"\n{'='*70}")
    print(f"Model Comparison: {model}")
    print(f"{'='*70}")
    print(f"  P95 Latency: {perf_results['p95_latency_seconds']:.3f}s")
    print(f"  Cost per Analysis: ${cost_results['cost_per_analysis_usd']:.4f}")
    print(f"  Golden Test Pass Rate: {accuracy_results['pass_rate']*100:.1f}%")
    print(f"  Mean Absolute Error: {accuracy_results['mean_absolute_error']:.4f}%")
    print(f"{'='*70}")
```

---

## Benchmark Reporting

### Report Format

```json
{
  "agent": "BoilerEfficiencyOptimizer",
  "version": "1.0.0",
  "model": "claude-sonnet-4",
  "timestamp": "2025-12-03T10:30:00Z",
  "performance": {
    "p50_latency_seconds": 1.85,
    "p95_latency_seconds": 3.42,
    "p99_latency_seconds": 5.12,
    "throughput_steady_state_rps": 105.3,
    "throughput_peak_rps": 523.7,
    "success_rate": 0.998,
    "error_rate": 0.002
  },
  "accuracy": {
    "golden_test_pass_rate": 1.0,
    "mean_absolute_error": 0.42,
    "rmse": 0.65,
    "max_absolute_error": 2.1
  },
  "cost": {
    "cost_per_analysis_usd": 0.123,
    "avg_prompt_tokens": 3240,
    "avg_completion_tokens": 1850,
    "avg_total_tokens": 5090,
    "avg_tool_calls": 3.2
  },
  "quality": {
    "explanation_clarity_score": 72.3,
    "citation_coverage": 0.94,
    "reasoning_consistency_score": 9.2,
    "recommendation_actionability_score": 8.7
  },
  "pass_fail": {
    "performance_pass": true,
    "accuracy_pass": true,
    "cost_pass": true,
    "quality_pass": true,
    "overall_pass": true
  }
}
```

### Benchmark Dashboard

Create Grafana dashboard with:
- Performance metrics (latency percentiles, throughput)
- Cost metrics (cost/analysis, token usage)
- Accuracy metrics (golden test pass rate, MAE)
- Quality metrics (explanation clarity, citation coverage)
- Trend analysis (performance over time, cost over time)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-03 | GL-TestEngineer | Initial benchmarking framework |

---

**END OF DOCUMENT**
