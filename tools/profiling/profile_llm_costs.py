"""
LLM Cost Profiler
================

Track LLM token usage and costs across your application.

Features:
- Token counting for all LLM calls
- Cost calculation by model
- Cost breakdown by operation
- Savings from caching
- Optimization recommendations
- Budget tracking and alerts

Usage:
    from profile_llm_costs import LLMCostTracker

    tracker = LLMCostTracker()

    # Track a completion
    with tracker.track("user_query"):
        response = llm.complete(prompt)

    # Generate report
    tracker.generate_report()

Author: Performance Engineering Team
Date: 2025-11-09
Version: 1.0.0
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager


# Model pricing (per 1M tokens)
MODEL_PRICING = {
    "gpt-4": {
        "input": 30.00,
        "output": 60.00
    },
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00
    },
    "gpt-3.5-turbo": {
        "input": 0.50,
        "output": 1.50
    },
    "claude-3-opus": {
        "input": 15.00,
        "output": 75.00
    },
    "claude-3-sonnet": {
        "input": 3.00,
        "output": 15.00
    },
    "claude-3-haiku": {
        "input": 0.25,
        "output": 1.25
    }
}


@dataclass
class LLMCall:
    """Record of a single LLM call."""
    timestamp: float
    operation: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool
    latency_ms: float
    cost_usd: float


class LLMCostTracker:
    """Track and analyze LLM costs."""

    def __init__(self, output_dir: Path = None, budget_usd: Optional[float] = None):
        """
        Initialize LLM cost tracker.

        Args:
            output_dir: Directory for output files
            budget_usd: Optional budget limit in USD
        """
        self.output_dir = output_dir or Path("./profiles")
        self.output_dir.mkdir(exist_ok=True)

        self.budget_usd = budget_usd
        self.calls: List[LLMCall] = []
        self.active_operation: Optional[str] = None
        self.operation_start: Optional[float] = None

    @contextmanager
    def track(self, operation: str):
        """
        Context manager to track an LLM operation.

        Args:
            operation: Operation name

        Example:
            >>> with tracker.track("user_query"):
            ...     response = llm.complete(prompt)
        """
        self.active_operation = operation
        self.operation_start = time.perf_counter()

        try:
            yield self
        finally:
            self.active_operation = None
            self.operation_start = None

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False
    ):
        """
        Record an LLM call.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached: Whether response was cached
        """
        # Calculate latency
        latency_ms = 0
        if self.operation_start:
            latency_ms = (time.perf_counter() - self.operation_start) * 1000

        # Calculate cost
        cost_usd = self._calculate_cost(model, input_tokens, output_tokens, cached)

        call = LLMCall(
            timestamp=time.time(),
            operation=self.active_operation or "unknown",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=cached,
            latency_ms=latency_ms,
            cost_usd=cost_usd
        )

        self.calls.append(call)

        # Check budget
        if self.budget_usd:
            total_cost = self.get_total_cost()
            if total_cost > self.budget_usd:
                print(f"⚠️  Budget exceeded! Spent: ${total_cost:.2f}, Budget: ${self.budget_usd:.2f}")

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool
    ) -> float:
        """Calculate cost for an LLM call."""
        if model not in MODEL_PRICING:
            return 0.0

        pricing = MODEL_PRICING[model]

        # If cached, only count input tokens at reduced rate (typically 10%)
        if cached:
            return (input_tokens / 1_000_000) * pricing["input"] * 0.1

        # Normal pricing
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def get_total_cost(self) -> float:
        """Get total cost across all calls."""
        return sum(call.cost_usd for call in self.calls)

    def get_total_tokens(self) -> Dict[str, int]:
        """Get total token counts."""
        return {
            "input": sum(call.input_tokens for call in self.calls),
            "output": sum(call.output_tokens for call in self.calls),
            "total": sum(call.input_tokens + call.output_tokens for call in self.calls)
        }

    def get_cost_by_operation(self) -> Dict[str, float]:
        """Get cost breakdown by operation."""
        cost_by_op = {}

        for call in self.calls:
            if call.operation not in cost_by_op:
                cost_by_op[call.operation] = 0.0
            cost_by_op[call.operation] += call.cost_usd

        return dict(sorted(cost_by_op.items(), key=lambda x: x[1], reverse=True))

    def get_cost_by_model(self) -> Dict[str, float]:
        """Get cost breakdown by model."""
        cost_by_model = {}

        for call in self.calls:
            if call.model not in cost_by_model:
                cost_by_model[call.model] = 0.0
            cost_by_model[call.model] += call.cost_usd

        return dict(sorted(cost_by_model.items(), key=lambda x: x[1], reverse=True))

    def get_cache_savings(self) -> Dict[str, float]:
        """Calculate savings from caching."""
        cached_calls = [c for c in self.calls if c.cached]
        uncached_calls = [c for c in self.calls if not c.cached]

        # Estimate what cached calls would have cost without caching
        estimated_full_cost = sum(
            self._calculate_cost(c.model, c.input_tokens, c.output_tokens, cached=False)
            for c in cached_calls
        )

        actual_cached_cost = sum(c.cost_usd for c in cached_calls)
        savings = estimated_full_cost - actual_cached_cost

        return {
            "cached_calls": len(cached_calls),
            "total_calls": len(self.calls),
            "cache_hit_rate": len(cached_calls) / len(self.calls) if self.calls else 0,
            "savings_usd": savings,
            "savings_percent": (savings / estimated_full_cost * 100) if estimated_full_cost > 0 else 0
        }

    def get_optimization_recommendations(self) -> List[Dict]:
        """Generate optimization recommendations."""
        recommendations = []

        # Check model usage
        cost_by_model = self.get_cost_by_model()
        if "gpt-4" in cost_by_model and cost_by_model["gpt-4"] > 10:
            gpt4_calls = len([c for c in self.calls if c.model == "gpt-4"])
            recommendations.append({
                "type": "model_selection",
                "severity": "high",
                "message": f"Consider using GPT-3.5-Turbo for {gpt4_calls} simpler tasks to reduce costs by ~95%",
                "potential_savings": cost_by_model["gpt-4"] * 0.95
            })

        # Check caching
        cache_stats = self.get_cache_savings()
        if cache_stats["cache_hit_rate"] < 0.3:
            recommendations.append({
                "type": "caching",
                "severity": "high",
                "message": f"Cache hit rate is only {cache_stats['cache_hit_rate']*100:.1f}%. Enable semantic caching.",
                "potential_savings": self.get_total_cost() * 0.3
            })

        # Check token usage
        tokens = self.get_total_tokens()
        avg_tokens_per_call = tokens["total"] / len(self.calls) if self.calls else 0

        if avg_tokens_per_call > 2000:
            recommendations.append({
                "type": "prompt_optimization",
                "severity": "medium",
                "message": f"Average {avg_tokens_per_call:.0f} tokens/call. Consider prompt compression.",
                "potential_savings": self.get_total_cost() * 0.2
            })

        return recommendations

    def generate_html_report(self, output_file: str = "llm_cost_report.html"):
        """
        Generate comprehensive HTML cost report.

        Args:
            output_file: Output filename
        """
        total_cost = self.get_total_cost()
        total_tokens = self.get_total_tokens()
        cost_by_operation = self.get_cost_by_operation()
        cost_by_model = self.get_cost_by_model()
        cache_savings = self.get_cache_savings()
        recommendations = self.get_optimization_recommendations()

        # Calculate potential savings
        potential_savings = sum(r["potential_savings"] for r in recommendations)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Cost Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #9b59b6;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .summary-item {{
            display: inline-block;
            margin-right: 30px;
        }}
        .summary-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        .summary-value {{
            font-size: 1.5em;
            color: #2c3e50;
        }}
        .cost-highlight {{
            color: #9b59b6;
            font-weight: bold;
        }}
        .savings-highlight {{
            color: #27ae60;
            font-weight: bold;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }}
        .success {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background: #9b59b6;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .chart {{
            margin: 20px 0;
            height: 400px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Cost Analysis Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="summary">
            <div class="summary-item">
                <div class="summary-label">Total Cost</div>
                <div class="summary-value cost-highlight">${total_cost:.2f}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Total Tokens</div>
                <div class="summary-value">{total_tokens['total']:,}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Total Calls</div>
                <div class="summary-value">{len(self.calls)}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Cache Hit Rate</div>
                <div class="summary-value">{cache_savings['cache_hit_rate']*100:.1f}%</div>
            </div>
        </div>
        """

        # Cache savings
        if cache_savings["savings_usd"] > 0:
            html += f"""
        <div class="success">
            <strong>Cache Savings:</strong> ${cache_savings['savings_usd']:.2f}
            ({cache_savings['savings_percent']:.1f}% saved through caching)
        </div>
        """

        # Budget status
        if self.budget_usd:
            budget_used = (total_cost / self.budget_usd) * 100
            status_class = "warning" if budget_used > 80 else "success"
            html += f"""
        <div class="{status_class}">
            <strong>Budget Status:</strong> ${total_cost:.2f} / ${self.budget_usd:.2f}
            ({budget_used:.1f}% used)
        </div>
        """

        # Cost by operation chart
        if cost_by_operation:
            operations = list(cost_by_operation.keys())
            costs = list(cost_by_operation.values())

            html += f"""
        <h2>Cost by Operation</h2>
        <div id="cost-by-operation" class="chart"></div>
        <script>
            var data = [{{
                x: {json.dumps(operations)},
                y: {json.dumps(costs)},
                type: 'bar',
                marker: {{color: '#9b59b6'}}
            }}];

            var layout = {{
                title: 'Cost Breakdown by Operation',
                xaxis: {{title: 'Operation'}},
                yaxis: {{title: 'Cost (USD)'}}
            }};

            Plotly.newPlot('cost-by-operation', data, layout);
        </script>
        """

        # Cost by model
        html += """
        <h2>Cost by Model</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Cost (USD)</th>
                    <th>% of Total</th>
                </tr>
            </thead>
            <tbody>
        """

        for model, cost in cost_by_model.items():
            percent = (cost / total_cost * 100) if total_cost > 0 else 0
            html += f"""
                <tr>
                    <td>{model}</td>
                    <td class="cost-highlight">${cost:.2f}</td>
                    <td>{percent:.1f}%</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        # Optimization recommendations
        if recommendations:
            html += "<h2>Optimization Recommendations</h2>"

            for rec in recommendations:
                severity_class = "warning" if rec["severity"] == "high" else "success"
                html += f"""
                <div class="{severity_class}">
                    <strong>{rec['type'].replace('_', ' ').title()}:</strong> {rec['message']}<br>
                    <span class="savings-highlight">Potential savings: ${rec['potential_savings']:.2f}</span>
                </div>
                """

            html += f"""
            <div class="success">
                <strong>Total Potential Savings:</strong>
                <span class="savings-highlight">${potential_savings:.2f}</span>
                ({potential_savings / total_cost * 100:.1f}% of current spend)
            </div>
            """

        html += """
    </div>
</body>
</html>
        """

        output_path = self.output_dir / output_file
        with open(output_path, "w") as f:
            f.write(html)

        print(f"LLM cost report saved to: {output_path}")

    def save_data(self, filename: str = None) -> Path:
        """
        Save tracking data to JSON.

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_costs_{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "total_cost": self.get_total_cost(),
            "total_tokens": self.get_total_tokens(),
            "cost_by_operation": self.get_cost_by_operation(),
            "cost_by_model": self.get_cost_by_model(),
            "cache_savings": self.get_cache_savings(),
            "calls": [asdict(call) for call in self.calls]
        }

        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path

    def print_summary(self):
        """Print cost summary to console."""
        print("\n" + "=" * 80)
        print("LLM COST SUMMARY")
        print("=" * 80)

        print(f"\nTotal Cost: ${self.get_total_cost():.2f}")
        print(f"Total Calls: {len(self.calls)}")

        tokens = self.get_total_tokens()
        print(f"\nTokens:")
        print(f"  Input:  {tokens['input']:,}")
        print(f"  Output: {tokens['output']:,}")
        print(f"  Total:  {tokens['total']:,}")

        cache_savings = self.get_cache_savings()
        print(f"\nCache Performance:")
        print(f"  Hit Rate: {cache_savings['cache_hit_rate']*100:.1f}%")
        print(f"  Savings:  ${cache_savings['savings_usd']:.2f}")

        print("\nCost by Operation:")
        for operation, cost in list(self.get_cost_by_operation().items())[:5]:
            print(f"  {operation:<30} ${cost:.2f}")

        print("\n" + "=" * 80)


# Global tracker instance
_global_tracker = None


def get_tracker() -> LLMCostTracker:
    """Get global cost tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = LLMCostTracker()
    return _global_tracker


if __name__ == "__main__":
    # Demo usage
    tracker = LLMCostTracker(budget_usd=100.0)

    # Simulate some LLM calls
    with tracker.track("user_query"):
        tracker.record_call("gpt-4", input_tokens=500, output_tokens=200)

    with tracker.track("data_extraction"):
        tracker.record_call("gpt-3.5-turbo", input_tokens=1000, output_tokens=500, cached=True)

    with tracker.track("user_query"):
        tracker.record_call("gpt-4", input_tokens=500, output_tokens=200, cached=True)

    # Print summary
    tracker.print_summary()

    # Generate report
    tracker.generate_html_report()
