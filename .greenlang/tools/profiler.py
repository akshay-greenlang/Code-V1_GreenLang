#!/usr/bin/env python3
"""
Performance Profiler

Profile infrastructure usage and identify performance bottlenecks.
Measures execution time, cache performance, and LLM costs.
"""

import argparse
import cProfile
import pstats
import io
import time
import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    total_time: float = 0.0
    function_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    llm_calls: int = 0
    llm_tokens: int = 0
    llm_cost: float = 0.0
    slow_functions: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.slow_functions is None:
            self.slow_functions = []


class InfrastructureProfiler:
    """Profile infrastructure usage."""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.profiler = cProfile.Profile()
        self.start_time = None

    @contextmanager
    def profile(self):
        """Context manager for profiling."""
        self.start_time = time.time()
        self.profiler.enable()

        try:
            yield self
        finally:
            self.profiler.disable()
            self.metrics.total_time = time.time() - self.start_time
            self._analyze_results()

    def _analyze_results(self):
        """Analyze profiling results."""
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)

        # Count function calls
        self.metrics.function_calls = ps.total_calls

        # Find slow functions
        stats = ps.stats
        for func, (cc, nc, tt, ct, callers) in list(stats.items())[:10]:
            if ct > 0.1:  # Functions taking more than 0.1s
                self.metrics.slow_functions.append({
                    'function': f"{func[0]}:{func[1]}:{func[2]}",
                    'calls': nc,
                    'total_time': ct,
                    'per_call': ct / nc if nc > 0 else 0
                })

    def get_report(self) -> str:
        """Generate performance report."""
        output = []
        output.append("=" * 80)
        output.append("PERFORMANCE PROFILING REPORT")
        output.append("=" * 80)

        output.append(f"\nTotal execution time: {self.metrics.total_time:.2f}s")
        output.append(f"Function calls: {self.metrics.function_calls:,}")

        if self.metrics.cache_hits > 0 or self.metrics.cache_misses > 0:
            total = self.metrics.cache_hits + self.metrics.cache_misses
            hit_rate = (self.metrics.cache_hits / total * 100) if total > 0 else 0
            output.append(f"\nCache Performance:")
            output.append(f"  Hits: {self.metrics.cache_hits}")
            output.append(f"  Misses: {self.metrics.cache_misses}")
            output.append(f"  Hit rate: {hit_rate:.1f}%")

        if self.metrics.llm_calls > 0:
            output.append(f"\nLLM Usage:")
            output.append(f"  API calls: {self.metrics.llm_calls}")
            output.append(f"  Tokens used: {self.metrics.llm_tokens:,}")
            output.append(f"  Estimated cost: ${self.metrics.llm_cost:.4f}")

        if self.metrics.slow_functions:
            output.append(f"\nSlowest Functions:")
            output.append("-" * 80)
            for func in self.metrics.slow_functions:
                output.append(f"\n  {func['function']}")
                output.append(f"    Calls: {func['calls']}")
                output.append(f"    Total time: {func['total_time']:.2f}s")
                output.append(f"    Per call: {func['per_call']:.4f}s")

        output.append("\n" + "=" * 80)
        output.append("\nRECOMMENDATIONS:")
        output.append("-" * 80)

        # Generate recommendations
        if self.metrics.cache_hit_rate < 50:
            output.append("\n⚠ Low cache hit rate - consider adjusting cache TTL or warming cache")

        if self.metrics.llm_calls > 10:
            output.append("\n⚠ High LLM usage - consider caching responses or batching requests")

        if self.metrics.slow_functions:
            output.append("\n⚠ Slow functions detected - consider optimization or async execution")

        output.append("\n" + "=" * 80)
        return "\n".join(output)


class CacheProfiler:
    """Profile cache performance."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.operations = []

    def record_hit(self, key: str):
        """Record cache hit."""
        self.hits += 1
        self.operations.append({
            'type': 'hit',
            'key': key,
            'timestamp': time.time()
        })

    def record_miss(self, key: str):
        """Record cache miss."""
        self.misses += 1
        self.operations.append({
            'type': 'miss',
            'key': key,
            'timestamp': time.time()
        })

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_operations': total,
            'hit_rate': hit_rate,
            'recent_operations': self.operations[-10:]
        }


class LLMProfiler:
    """Profile LLM usage and costs."""

    # Token costs (per 1000 tokens)
    COSTS = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'claude-3-opus': {'input': 0.015, 'output': 0.075},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
    }

    def __init__(self):
        self.calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.by_model = {}

    def record_call(self, model: str, input_tokens: int, output_tokens: int):
        """Record an LLM call."""
        self.calls += 1
        total = input_tokens + output_tokens
        self.total_tokens += total

        # Calculate cost
        if model in self.COSTS:
            cost = (
                (input_tokens / 1000) * self.COSTS[model]['input'] +
                (output_tokens / 1000) * self.COSTS[model]['output']
            )
            self.total_cost += cost

            # Track by model
            if model not in self.by_model:
                self.by_model[model] = {
                    'calls': 0,
                    'tokens': 0,
                    'cost': 0.0
                }

            self.by_model[model]['calls'] += 1
            self.by_model[model]['tokens'] += total
            self.by_model[model]['cost'] += cost

    def get_report(self) -> str:
        """Generate LLM usage report."""
        output = []
        output.append("=" * 80)
        output.append("LLM USAGE REPORT")
        output.append("=" * 80)

        output.append(f"\nTotal API calls: {self.calls}")
        output.append(f"Total tokens: {self.total_tokens:,}")
        output.append(f"Total cost: ${self.total_cost:.4f}")

        if self.by_model:
            output.append("\nBy Model:")
            output.append("-" * 80)

            for model, stats in self.by_model.items():
                output.append(f"\n{model}:")
                output.append(f"  Calls: {stats['calls']}")
                output.append(f"  Tokens: {stats['tokens']:,}")
                output.append(f"  Cost: ${stats['cost']:.4f}")

        output.append("\n" + "=" * 80)
        return "\n".join(output)


def profile_file(file_path: str):
    """Profile a Python file."""
    print(f"Profiling {file_path}...")

    profiler = InfrastructureProfiler()

    with profiler.profile():
        # Execute the file
        with open(file_path, 'r') as f:
            code = f.read()

        try:
            exec(code, {'__name__': '__main__'})
        except Exception as e:
            print(f"Error executing file: {e}")

    return profiler.get_report()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Profile infrastructure performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile a file
  greenlang profile my_agent.py

  # Profile with detailed output
  greenlang profile my_agent.py --detailed

  # Generate JSON report
  greenlang profile my_agent.py --format json --output profile.json
        """
    )

    parser.add_argument('file', help='Python file to profile')
    parser.add_argument('--detailed', action='store_true', help='Show detailed statistics')
    parser.add_argument('--format', choices=['text', 'json'], default='text')
    parser.add_argument('--output', help='Output file')

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    # Profile the file
    report = profile_file(args.file)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()
