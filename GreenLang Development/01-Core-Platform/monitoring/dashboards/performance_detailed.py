# -*- coding: utf-8 -*-
"""
Real-Time Performance Dashboard
===============================

Comprehensive real-time performance monitoring dashboard for GreenLang infrastructure.

Features:
- Live SLO compliance tracking
- P50/P95/P99 latency charts
- Throughput trends
- Error rate monitoring
- Resource utilization
- Cost per operation
- Performance regression alerts

Updates every 10 seconds.

Usage:
    # Run dashboard
    python performance_detailed.py

    # Run with custom refresh rate
    python performance_detailed.py --refresh-interval 5

    # Export metrics
    python performance_detailed.py --export metrics.json

Author: Performance Engineering Team
Date: 2025-11-09
Version: 1.0.0
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path
from greenlang.utilities.determinism import deterministic_random, DeterministicClock


try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Install 'rich' for enhanced dashboard: pip install rich")


class PerformanceDashboard:
    """Real-time performance monitoring dashboard."""

    def __init__(self, refresh_interval: int = 10):
        """
        Initialize dashboard.

        Args:
            refresh_interval: Update interval in seconds
        """
        self.refresh_interval = refresh_interval
        self.console = Console() if RICH_AVAILABLE else None

        # Mock data generators (in production, fetch from monitoring system)
        self.start_time = time.time()
        self.metrics_history = []

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.

        In production, this would fetch from:
        - Prometheus
        - Grafana
        - CloudWatch
        - Application metrics endpoint
        """
        import random

        # Simulate realistic metrics (replace with actual data source)
        runtime = time.time() - self.start_time

        # Add some variation
        base_latency = 85 + deterministic_random().randint(-20, 40)
        base_throughput = 1200 + deterministic_random().randint(-200, 300)

        metrics = {
            "timestamp": DeterministicClock.now().isoformat(),
            "runtime_seconds": runtime,

            # Infrastructure
            "llm": {
                "p50_latency_ms": base_latency * 20 + deterministic_random().randint(-50, 100),
                "p95_latency_ms": base_latency * 22 + deterministic_random().randint(-50, 150),
                "p99_latency_ms": base_latency * 25 + deterministic_random().randint(-100, 200),
                "cache_hit_rate": 0.45 + random.uniform(0, 0.2),
                "requests_per_sec": 15 + deterministic_random().randint(-5, 10),
                "error_rate": 0.002 + random.uniform(0, 0.01),
                "cost_per_1m_tokens": 25 + random.uniform(-5, 5),
                "slo_target_p95": 2000,
                "slo_target_cache": 0.30
            },

            "cache": {
                "l1_p95_latency_us": 5 + deterministic_random().randint(-2, 5),
                "l2_p95_latency_ms": 2 + random.uniform(-0.5, 1),
                "l3_p95_latency_ms": 80 + deterministic_random().randint(-20, 30),
                "overall_hit_rate": 0.55 + random.uniform(-0.1, 0.15),
                "throughput_ops_sec": 250000 + deterministic_random().randint(-50000, 100000),
                "slo_target_l1": 100,
                "slo_target_l2": 5,
                "slo_target_hit_rate": 0.50
            },

            "database": {
                "p50_latency_ms": base_latency * 0.6,
                "p95_latency_ms": base_latency,
                "p99_latency_ms": base_latency * 3 + deterministic_random().randint(0, 100),
                "pool_utilization": 0.45 + random.uniform(-0.1, 0.2),
                "queries_per_sec": 850 + deterministic_random().randint(-100, 200),
                "slow_queries_pct": 0.005 + random.uniform(0, 0.01),
                "error_rate": 0.0005 + random.uniform(0, 0.001),
                "slo_target_p95": 100,
                "slo_target_pool": 0.80
            },

            "factor_broker": {
                "p50_latency_ms": 18 + deterministic_random().randint(-5, 10),
                "p95_latency_ms": 35 + deterministic_random().randint(-10, 20),
                "p99_latency_ms": 90 + deterministic_random().randint(-20, 30),
                "cache_hit_rate": 0.78 + random.uniform(-0.05, 0.10),
                "accuracy": 0.96 + random.uniform(-0.02, 0.02),
                "slo_target_p95": 50,
                "slo_target_cache": 0.70
            },

            # Applications
            "cbam": {
                "single_p95_ms": 750 + deterministic_random().randint(-100, 200),
                "batch_throughput_sec": base_throughput * 4,
                "memory_10k_mb": 280 + deterministic_random().randint(-30, 50),
                "error_rate": 0.003 + random.uniform(0, 0.005),
                "slo_target_latency": 1000,
                "slo_target_throughput": 1000
            },

            "csrd": {
                "materiality_ms": 3800 + deterministic_random().randint(-500, 1000),
                "full_pipeline_ms": 7500 + deterministic_random().randint(-1000, 2000),
                "accuracy": 0.97 + random.uniform(-0.01, 0.01),
                "error_rate": 0.008 + random.uniform(0, 0.01),
                "slo_target_materiality": 5000,
                "slo_target_pipeline": 10000
            },

            "vcci": {
                "scope3_10k_sec": 150 + deterministic_random().randint(-30, 50),
                "throughput_sec": 65 + deterministic_random().randint(-15, 25),
                "confidence": 0.87 + random.uniform(-0.05, 0.05),
                "error_rate": 0.004 + random.uniform(0, 0.005),
                "slo_target_time": 60,
                "slo_target_throughput": 100
            },

            # System resources
            "resources": {
                "cpu_percent": 45 + deterministic_random().randint(-10, 20),
                "memory_percent": 62 + deterministic_random().randint(-5, 10),
                "disk_percent": 38 + deterministic_random().randint(-2, 5),
                "network_mbps": 125 + deterministic_random().randint(-20, 40)
            }
        }

        # Track history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 60:  # Keep last 10 minutes (10s interval)
            self.metrics_history.pop(0)

        return metrics

    def check_slo_compliance(self, metrics: Dict) -> Dict[str, Dict]:
        """Check SLO compliance for all components."""
        compliance = {}

        # LLM SLO checks
        compliance["llm"] = {
            "p95_latency": {
                "value": metrics["llm"]["p95_latency_ms"],
                "target": metrics["llm"]["slo_target_p95"],
                "compliant": metrics["llm"]["p95_latency_ms"] < metrics["llm"]["slo_target_p95"],
                "status": "✓" if metrics["llm"]["p95_latency_ms"] < metrics["llm"]["slo_target_p95"] else "✗"
            },
            "cache_hit_rate": {
                "value": metrics["llm"]["cache_hit_rate"],
                "target": metrics["llm"]["slo_target_cache"],
                "compliant": metrics["llm"]["cache_hit_rate"] > metrics["llm"]["slo_target_cache"],
                "status": "✓" if metrics["llm"]["cache_hit_rate"] > metrics["llm"]["slo_target_cache"] else "✗"
            }
        }

        # Cache SLO checks
        compliance["cache"] = {
            "l1_latency": {
                "value": metrics["cache"]["l1_p95_latency_us"],
                "target": metrics["cache"]["slo_target_l1"],
                "compliant": metrics["cache"]["l1_p95_latency_us"] < metrics["cache"]["slo_target_l1"],
                "status": "✓" if metrics["cache"]["l1_p95_latency_us"] < metrics["cache"]["slo_target_l1"] else "✗"
            },
            "hit_rate": {
                "value": metrics["cache"]["overall_hit_rate"],
                "target": metrics["cache"]["slo_target_hit_rate"],
                "compliant": metrics["cache"]["overall_hit_rate"] > metrics["cache"]["slo_target_hit_rate"],
                "status": "✓" if metrics["cache"]["overall_hit_rate"] > metrics["cache"]["slo_target_hit_rate"] else "✗"
            }
        }

        # Database SLO checks
        compliance["database"] = {
            "p95_latency": {
                "value": metrics["database"]["p95_latency_ms"],
                "target": metrics["database"]["slo_target_p95"],
                "compliant": metrics["database"]["p95_latency_ms"] < metrics["database"]["slo_target_p95"],
                "status": "✓" if metrics["database"]["p95_latency_ms"] < metrics["database"]["slo_target_p95"] else "✗"
            },
            "pool_utilization": {
                "value": metrics["database"]["pool_utilization"],
                "target": metrics["database"]["slo_target_pool"],
                "compliant": metrics["database"]["pool_utilization"] < metrics["database"]["slo_target_pool"],
                "status": "✓" if metrics["database"]["pool_utilization"] < metrics["database"]["slo_target_pool"] else "✗"
            }
        }

        # Application SLO checks
        compliance["cbam"] = {
            "latency": {
                "value": metrics["cbam"]["single_p95_ms"],
                "target": metrics["cbam"]["slo_target_latency"],
                "compliant": metrics["cbam"]["single_p95_ms"] < metrics["cbam"]["slo_target_latency"],
                "status": "✓" if metrics["cbam"]["single_p95_ms"] < metrics["cbam"]["slo_target_latency"] else "✗"
            },
            "throughput": {
                "value": metrics["cbam"]["batch_throughput_sec"],
                "target": metrics["cbam"]["slo_target_throughput"],
                "compliant": metrics["cbam"]["batch_throughput_sec"] > metrics["cbam"]["slo_target_throughput"],
                "status": "✓" if metrics["cbam"]["batch_throughput_sec"] > metrics["cbam"]["slo_target_throughput"] else "✗"
            }
        }

        return compliance

    def render_slo_table(self, metrics: Dict, compliance: Dict) -> Table:
        """Render SLO compliance table."""
        table = Table(title="SLO Compliance Status", show_header=True)

        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Metric", style="white")
        table.add_column("Current", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status", justify="center")

        # LLM
        table.add_row(
            "LLM Services",
            "P95 Latency",
            f"{metrics['llm']['p95_latency_ms']:.0f}ms",
            f"< {metrics['llm']['slo_target_p95']}ms",
            f"[green]{compliance['llm']['p95_latency']['status']}[/green]" if compliance['llm']['p95_latency']['compliant'] else f"[red]{compliance['llm']['p95_latency']['status']}[/red]"
        )
        table.add_row(
            "",
            "Cache Hit Rate",
            f"{metrics['llm']['cache_hit_rate']*100:.1f}%",
            f"> {metrics['llm']['slo_target_cache']*100:.0f}%",
            f"[green]{compliance['llm']['cache_hit_rate']['status']}[/green]" if compliance['llm']['cache_hit_rate']['compliant'] else f"[red]{compliance['llm']['cache_hit_rate']['status']}[/red]"
        )

        # Cache
        table.add_row(
            "Cache (L1)",
            "P95 Latency",
            f"{metrics['cache']['l1_p95_latency_us']:.0f}µs",
            f"< {metrics['cache']['slo_target_l1']}µs",
            f"[green]{compliance['cache']['l1_latency']['status']}[/green]" if compliance['cache']['l1_latency']['compliant'] else f"[red]{compliance['cache']['l1_latency']['status']}[/red]"
        )

        # Database
        table.add_row(
            "Database",
            "P95 Latency",
            f"{metrics['database']['p95_latency_ms']:.0f}ms",
            f"< {metrics['database']['slo_target_p95']}ms",
            f"[green]{compliance['database']['p95_latency']['status']}[/green]" if compliance['database']['p95_latency']['compliant'] else f"[red]{compliance['database']['p95_latency']['status']}[/red]"
        )
        table.add_row(
            "",
            "Pool Utilization",
            f"{metrics['database']['pool_utilization']*100:.0f}%",
            f"< {metrics['database']['slo_target_pool']*100:.0f}%",
            f"[green]{compliance['database']['pool_utilization']['status']}[/green]" if compliance['database']['pool_utilization']['compliant'] else f"[red]{compliance['database']['pool_utilization']['status']}[/red]"
        )

        # CBAM
        table.add_row(
            "CBAM App",
            "P95 Latency",
            f"{metrics['cbam']['single_p95_ms']:.0f}ms",
            f"< {metrics['cbam']['slo_target_latency']}ms",
            f"[green]{compliance['cbam']['latency']['status']}[/green]" if compliance['cbam']['latency']['compliant'] else f"[red]{compliance['cbam']['latency']['status']}[/red]"
        )
        table.add_row(
            "",
            "Throughput",
            f"{metrics['cbam']['batch_throughput_sec']:.0f}/sec",
            f"> {metrics['cbam']['slo_target_throughput']}/sec",
            f"[green]{compliance['cbam']['throughput']['status']}[/green]" if compliance['cbam']['throughput']['compliant'] else f"[red]{compliance['cbam']['throughput']['status']}[/red]"
        )

        return table

    def render_performance_table(self, metrics: Dict) -> Table:
        """Render detailed performance metrics table."""
        table = Table(title="Performance Metrics", show_header=True)

        table.add_column("Component", style="cyan")
        table.add_column("P50", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("P99", justify="right")
        table.add_column("Throughput", justify="right")

        # LLM
        table.add_row(
            "LLM Services",
            f"{metrics['llm']['p50_latency_ms']:.0f}ms",
            f"{metrics['llm']['p95_latency_ms']:.0f}ms",
            f"{metrics['llm']['p99_latency_ms']:.0f}ms",
            f"{metrics['llm']['requests_per_sec']:.0f}/sec"
        )

        # Cache
        table.add_row(
            "Cache (L1)",
            f"{metrics['cache']['l1_p95_latency_us']:.0f}µs",
            f"{metrics['cache']['l1_p95_latency_us']:.0f}µs",
            "-",
            f"{metrics['cache']['throughput_ops_sec']/1000:.0f}K/sec"
        )

        # Database
        table.add_row(
            "Database",
            f"{metrics['database']['p50_latency_ms']:.0f}ms",
            f"{metrics['database']['p95_latency_ms']:.0f}ms",
            f"{metrics['database']['p99_latency_ms']:.0f}ms",
            f"{metrics['database']['queries_per_sec']:.0f}/sec"
        )

        # Factor Broker
        table.add_row(
            "Factor Broker",
            f"{metrics['factor_broker']['p50_latency_ms']:.0f}ms",
            f"{metrics['factor_broker']['p95_latency_ms']:.0f}ms",
            f"{metrics['factor_broker']['p99_latency_ms']:.0f}ms",
            "-"
        )

        # CBAM
        table.add_row(
            "CBAM (single)",
            "-",
            f"{metrics['cbam']['single_p95_ms']:.0f}ms",
            "-",
            "-"
        )
        table.add_row(
            "CBAM (batch)",
            "-",
            "-",
            "-",
            f"{metrics['cbam']['batch_throughput_sec']:.0f}/sec"
        )

        # CSRD
        table.add_row(
            "CSRD",
            "-",
            f"{metrics['csrd']['materiality_ms']:.0f}ms",
            "-",
            "-"
        )

        return table

    def render_resources_table(self, metrics: Dict) -> Table:
        """Render resource utilization table."""
        table = Table(title="Resource Utilization", show_header=True)

        table.add_column("Resource", style="cyan")
        table.add_column("Usage", justify="right")
        table.add_column("Bar", width=30)

        resources = metrics["resources"]

        # CPU
        cpu_color = "green" if resources["cpu_percent"] < 70 else "yellow" if resources["cpu_percent"] < 85 else "red"
        table.add_row(
            "CPU",
            f"{resources['cpu_percent']:.0f}%",
            f"[{cpu_color}]{'█' * int(resources['cpu_percent'] / 3.33)}[/{cpu_color}]"
        )

        # Memory
        mem_color = "green" if resources["memory_percent"] < 70 else "yellow" if resources["memory_percent"] < 85 else "red"
        table.add_row(
            "Memory",
            f"{resources['memory_percent']:.0f}%",
            f"[{mem_color}]{'█' * int(resources['memory_percent'] / 3.33)}[/{mem_color}]"
        )

        # Disk
        disk_color = "green" if resources["disk_percent"] < 80 else "yellow" if resources["disk_percent"] < 90 else "red"
        table.add_row(
            "Disk",
            f"{resources['disk_percent']:.0f}%",
            f"[{disk_color}]{'█' * int(resources['disk_percent'] / 3.33)}[/{disk_color}]"
        )

        # Network
        table.add_row(
            "Network",
            f"{resources['network_mbps']:.0f} Mbps",
            f"[blue]{'█' * int(resources['network_mbps'] / 10)}[/blue]"
        )

        return table

    def render_dashboard(self, metrics: Dict, compliance: Dict) -> Layout:
        """Render complete dashboard layout."""
        layout = Layout()

        # Header
        header = Panel(
            f"[bold cyan]GreenLang Performance Dashboard[/bold cyan]\n"
            f"Updated: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Refresh: {self.refresh_interval}s | "
            f"Uptime: {int(metrics['runtime_seconds'])}s",
            style="white on blue"
        )

        # Calculate overall SLO compliance
        all_checks = []
        for component in compliance.values():
            for check in component.values():
                all_checks.append(check["compliant"])

        compliance_rate = sum(all_checks) / len(all_checks) if all_checks else 0
        compliance_color = "green" if compliance_rate >= 0.95 else "yellow" if compliance_rate >= 0.80 else "red"

        summary = Panel(
            f"[bold]Overall SLO Compliance: [{compliance_color}]{compliance_rate*100:.0f}%[/{compliance_color}][/bold]",
            style="white"
        )

        # Split layout
        layout.split_column(
            Layout(header, size=3),
            Layout(summary, size=3),
            Layout(name="main", ratio=2)
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        layout["left"].split_column(
            Layout(self.render_slo_table(metrics, compliance)),
            Layout(self.render_performance_table(metrics))
        )

        layout["right"].split_column(
            Layout(self.render_resources_table(metrics)),
            Layout(Panel("Charts not available in terminal. Access Grafana for visualizations:\nhttp://grafana.greenlang.ai/d/performance", title="Charts"))
        )

        return layout

    async def run(self):
        """Run the dashboard with live updates."""
        if not RICH_AVAILABLE:
            print("Rich library not installed. Running in basic mode.")
            while True:
                metrics = self.get_current_metrics()
                print(json.dumps(metrics, indent=2))
                await asyncio.sleep(self.refresh_interval)
            return

        with Live(console=self.console, refresh_per_second=1) as live:
            while True:
                metrics = self.get_current_metrics()
                compliance = self.check_slo_compliance(metrics)
                dashboard = self.render_dashboard(metrics, compliance)

                live.update(dashboard)
                await asyncio.sleep(self.refresh_interval)

    def export_metrics(self, output_file: str = "metrics.json"):
        """Export current metrics to JSON file."""
        metrics = self.get_current_metrics()

        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics exported to: {output_path}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GreenLang Performance Dashboard")
    parser.add_argument("--refresh-interval", type=int, default=10, help="Refresh interval in seconds")
    parser.add_argument("--export", help="Export metrics to JSON file and exit")

    args = parser.parse_args()

    dashboard = PerformanceDashboard(refresh_interval=args.refresh_interval)

    if args.export:
        dashboard.export_metrics(args.export)
    else:
        print("Starting GreenLang Performance Dashboard...")
        print(f"Refresh interval: {args.refresh_interval} seconds")
        print("Press Ctrl+C to exit\n")

        try:
            await dashboard.run()
        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")


if __name__ == "__main__":
    asyncio.run(main())
