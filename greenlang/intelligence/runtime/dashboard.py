"""
Intelligence Layer Cost Tracking Dashboard

Real-time monitoring dashboard for:
- Cost by provider/model
- Budget utilization and alerts
- Request volume and success rates
- Token usage and cost per request
- Circuit breaker status
- Tool invocation statistics

Provides both CLI and web-based views.
"""

from __future__ import annotations
import time
from typing import Dict, List, Any
from datetime import datetime, timedelta
from greenlang.intelligence.runtime.monitoring import get_metrics_collector


class CostDashboard:
    """
    Cost tracking and visualization dashboard

    Displays:
    - Real-time cost by provider
    - Budget burn rate and remaining
    - Cost per request
    - Token efficiency
    - Alert status
    """

    def __init__(self):
        self.collector = get_metrics_collector()

    def get_cost_by_provider(self) -> Dict[str, float]:
        """Get total cost broken down by provider"""
        costs = {}

        for key, value in self.collector._counters.items():
            if "intelligence_cost_usd" in key:
                # Extract provider from key
                if "{provider=" in key:
                    provider = key.split("provider=")[1].split(",")[0].split("}")[0]
                    costs[provider] = costs.get(provider, 0) + value

        return costs

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        spent = self.collector.get_gauge("intelligence_budget_spent_usd") or 0
        remaining = self.collector.get_gauge("intelligence_budget_remaining_usd") or 0
        max_budget = self.collector.get_gauge("intelligence_budget_max_usd") or 0
        remaining_pct = self.collector.get_gauge("intelligence_budget_remaining_pct") or 0

        # Calculate burn rate (cost per hour)
        # Note: This is simplified - production should track time windows
        total_requests = self.collector.get_counter("intelligence_requests_total") or 1
        burn_rate_per_request = spent / total_requests if total_requests > 0 else 0

        return {
            "spent_usd": spent,
            "remaining_usd": remaining,
            "max_usd": max_budget,
            "remaining_pct": remaining_pct,
            "burn_rate_per_request_usd": burn_rate_per_request,
            "status": self._get_budget_status_label(remaining_pct)
        }

    def _get_budget_status_label(self, remaining_pct: float) -> str:
        """Get budget status label"""
        if remaining_pct >= 50:
            return "✅ HEALTHY"
        elif remaining_pct >= 20:
            return "⚠️  MODERATE"
        elif remaining_pct >= 10:
            return "🟠 LOW"
        else:
            return "🔴 CRITICAL"

    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics by provider"""
        providers = {}

        # Get unique providers from counters
        for key in self.collector._counters.keys():
            if "provider=" in key:
                provider = key.split("provider=")[1].split(",")[0].split("}")[0]
                if provider not in providers:
                    providers[provider] = {
                        "total_requests": 0,
                        "failed_requests": 0,
                        "total_cost_usd": 0.0,
                        "total_tokens": 0,
                        "avg_latency_ms": 0.0
                    }

        # Populate stats
        for provider in providers.keys():
            labels = {"provider": provider}

            providers[provider]["total_requests"] = self.collector.get_counter(
                "intelligence_requests_total", labels
            )
            providers[provider]["failed_requests"] = self.collector.get_counter(
                "intelligence_requests_failed", labels
            )

            # Get latency stats
            latency_stats = self.collector.get_histogram_stats(
                "intelligence_request_duration_ms", labels
            )
            providers[provider]["avg_latency_ms"] = latency_stats.get("avg", 0.0)

            # Calculate success rate
            total = providers[provider]["total_requests"]
            failed = providers[provider]["failed_requests"]
            providers[provider]["success_rate_pct"] = (
                ((total - failed) / total * 100) if total > 0 else 0.0
            )

        return providers

    def get_circuit_breaker_status(self) -> Dict[str, str]:
        """Get circuit breaker status for all providers"""
        status = {}

        for key, value in self.collector._gauges.items():
            if "circuit_breaker_state" in key and "provider=" in key:
                provider = key.split("provider=")[1].split(",")[0].split("}")[0]
                state_map = {0: "🟢 CLOSED", 1: "🟡 HALF_OPEN", 2: "🔴 OPEN"}
                status[provider] = state_map.get(int(value), "UNKNOWN")

        return status

    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get tool invocation statistics"""
        tools = {}

        for key in self.collector._counters.keys():
            if "tool_invocations_total" in key and "tool=" in key:
                tool_name = key.split("tool=")[1].split(",")[0].split("}")[0]
                if tool_name not in tools:
                    tools[tool_name] = {
                        "total_invocations": 0,
                        "failed_invocations": 0,
                        "avg_duration_ms": 0.0
                    }

        # Populate stats
        for tool_name in tools.keys():
            labels = {"tool": tool_name}

            tools[tool_name]["total_invocations"] = self.collector.get_counter(
                "intelligence_tool_invocations_total", labels
            )
            tools[tool_name]["failed_invocations"] = self.collector.get_counter(
                "intelligence_tool_invocations_failed", labels
            )

            # Get duration stats
            duration_stats = self.collector.get_histogram_stats(
                "intelligence_tool_duration_ms", labels
            )
            tools[tool_name]["avg_duration_ms"] = duration_stats.get("avg", 0.0)

            # Calculate success rate
            total = tools[tool_name]["total_invocations"]
            failed = tools[tool_name]["failed_invocations"]
            tools[tool_name]["success_rate_pct"] = (
                ((total - failed) / total * 100) if total > 0 else 0.0
            )

        return tools

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        return [alert.to_dict() for alert in self.collector.get_active_alerts()]

    def print_cli_dashboard(self):
        """Print dashboard to CLI"""
        print("\n" + "="*80)
        print("  GREENLANG INTELLIGENCE LAYER - COST DASHBOARD")
        print("="*80)
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # Budget Status
        budget = self.get_budget_status()
        print("\n📊 BUDGET STATUS")
        print("-"*80)
        print(f"  Status: {budget['status']}")
        print(f"  Spent: ${budget['spent_usd']:.4f} / ${budget['max_usd']:.2f}")
        print(f"  Remaining: ${budget['remaining_usd']:.4f} ({budget['remaining_pct']:.1f}%)")
        print(f"  Burn Rate: ${budget['burn_rate_per_request_usd']:.6f} per request")

        # Provider Stats
        print("\n🔌 PROVIDER STATISTICS")
        print("-"*80)
        providers = self.get_provider_stats()
        if providers:
            for provider, stats in providers.items():
                print(f"\n  {provider.upper()}:")
                print(f"    Requests: {stats['total_requests']} (Success: {stats['success_rate_pct']:.1f}%)")
                print(f"    Avg Latency: {stats['avg_latency_ms']:.0f}ms")
        else:
            print("  No provider data yet")

        # Circuit Breaker Status
        print("\n⚡ CIRCUIT BREAKER STATUS")
        print("-"*80)
        cb_status = self.get_circuit_breaker_status()
        if cb_status:
            for provider, state in cb_status.items():
                print(f"  {provider}: {state}")
        else:
            print("  All circuits operational")

        # Tool Stats
        print("\n🔧 TOOL INVOCATIONS")
        print("-"*80)
        tools = self.get_tool_stats()
        if tools:
            for tool_name, stats in sorted(tools.items(), key=lambda x: x[1]['total_invocations'], reverse=True)[:10]:
                print(f"  {tool_name}: {stats['total_invocations']} calls ({stats['success_rate_pct']:.1f}% success)")
        else:
            print("  No tool invocations yet")

        # Active Alerts
        print("\n🚨 ACTIVE ALERTS")
        print("-"*80)
        alerts = self.get_active_alerts()
        if alerts:
            for alert in alerts:
                severity_icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🔴"}.get(alert['severity'], "")
                print(f"  {severity_icon} [{alert['severity'].upper()}] {alert['name']}")
                print(f"     {alert['message']}")
        else:
            print("  ✅ No active alerts")

        print("\n" + "="*80 + "\n")

    def export_json_dashboard(self) -> Dict[str, Any]:
        """Export dashboard data as JSON"""
        return {
            "timestamp": datetime.now().isoformat(),
            "budget": self.get_budget_status(),
            "providers": self.get_provider_stats(),
            "circuit_breakers": self.get_circuit_breaker_status(),
            "tools": self.get_tool_stats(),
            "alerts": self.get_active_alerts()
        }


# Global dashboard instance
_dashboard: Optional['CostDashboard'] = None


def get_dashboard() -> CostDashboard:
    """Get global dashboard (singleton)"""
    global _dashboard
    if _dashboard is None:
        _dashboard = CostDashboard()
    return _dashboard


def print_dashboard():
    """Convenience function to print dashboard"""
    dashboard = get_dashboard()
    dashboard.print_cli_dashboard()


if __name__ == "__main__":
    # Demo dashboard
    print_dashboard()
