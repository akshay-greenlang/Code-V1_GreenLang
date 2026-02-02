# -*- coding: utf-8 -*-
"""
GreenLang Telemetry CLI Commands
"""

import click
import json
import yaml
import time
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from greenlang.utilities.determinism import DeterministicClock
from greenlang.telemetry import (
    get_monitoring_service,
    get_metrics_collector,
    get_health_checker,
    get_log_aggregator,
    get_performance_monitor,
    HealthStatus,
    CheckType,
    AlertSeverity,
)

console = Console()


@click.group()
def telemetry():
    """Monitoring and observability commands"""


# Health commands
@telemetry.group()
def health():
    """Health check commands"""


@health.command()
@click.option(
    "--check-type",
    type=click.Choice(["liveness", "readiness", "all"]),
    default="all",
    help="Type of health check",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
def status(check_type: str, output: str):
    """Check health status"""
    try:
        checker = get_health_checker()

        # Map check type
        if check_type == "liveness":
            check = CheckType.LIVENESS
        elif check_type == "readiness":
            check = CheckType.READINESS
        else:
            check = None

        # Get health report
        report = checker.check_health(check)

        if output == "json":
            click.echo(json.dumps(report.to_dict(), indent=2))
        elif output == "yaml":
            click.echo(yaml.dump(report.to_dict()))
        else:
            # Display as table
            status_color = {
                HealthStatus.HEALTHY: "green",
                HealthStatus.DEGRADED: "yellow",
                HealthStatus.UNHEALTHY: "red",
                HealthStatus.UNKNOWN: "white",
            }.get(report.status, "white")

            console.print(
                Panel(
                    f"[{status_color}]Overall Status: {report.status.value}[/{status_color}]\n"
                    f"Version: {report.version}\n"
                    f"Uptime: {report.uptime_seconds:.0f} seconds",
                    title="Health Status",
                )
            )

            # Show individual checks
            table = Table(title="Health Checks")
            table.add_column("Check", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Message", style="white")
            table.add_column("Duration (ms)", style="yellow")

            for check in report.checks:
                status_style = {
                    HealthStatus.HEALTHY: "green",
                    HealthStatus.DEGRADED: "yellow",
                    HealthStatus.UNHEALTHY: "red",
                    HealthStatus.UNKNOWN: "white",
                }.get(check.status, "white")

                table.add_row(
                    check.name,
                    f"[{status_style}]{check.status.value}[/{status_style}]",
                    check.message,
                    f"{check.duration_ms:.1f}",
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error checking health: {e}[/red]")
        raise click.ClickException(str(e))


@health.command()
@click.option("--port", type=int, default=8080, help="Port for health endpoint")
def serve(port: int):
    """Start health check HTTP server"""
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler

        checker = get_health_checker()

        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    report = checker.check_health()
                    status_code = 200 if report.status == HealthStatus.HEALTHY else 503

                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(report.to_json().encode())

                elif self.path == "/health/live":
                    report = checker.check_health(CheckType.LIVENESS)
                    status_code = 200 if report.status == HealthStatus.HEALTHY else 503

                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(report.to_json().encode())

                elif self.path == "/health/ready":
                    report = checker.check_health(CheckType.READINESS)
                    status_code = 200 if report.status == HealthStatus.HEALTHY else 503

                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(report.to_json().encode())

                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress logs

        server = HTTPServer(("", port), HealthHandler)

        console.print(f"[green]Health check server started on port {port}[/green]")
        console.print("Endpoints:")
        console.print(f"  - http://localhost:{port}/health")
        console.print(f"  - http://localhost:{port}/health/live")
        console.print(f"  - http://localhost:{port}/health/ready")

        server.serve_forever()

    except KeyboardInterrupt:
        console.print("\n[yellow]Health server stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting health server: {e}[/red]")
        raise click.ClickException(str(e))


# Metrics commands
@telemetry.group()
def metrics():
    """Metrics commands"""


@metrics.command()
@click.option("--port", type=int, default=8000, help="Port for metrics endpoint")
def serve(port: int):
    """Start metrics HTTP server"""
    try:
        collector = get_metrics_collector()
        collector.start_collection(port)

        console.print(f"[green]Metrics server started on port {port}[/green]")
        console.print(
            f"Prometheus metrics available at: http://localhost:{port}/metrics"
        )

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            collector.stop_collection()
            console.print("\n[yellow]Metrics server stopped[/yellow]")

    except Exception as e:
        console.print(f"[red]Error starting metrics server: {e}[/red]")
        raise click.ClickException(str(e))


@metrics.command()
@click.option("--metric", help="Specific metric to show")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
def list(metric: str, output: str):
    """List current metrics"""
    try:
        collector = get_metrics_collector()

        # Collect current system metrics
        collector.collect_system_metrics()

        # Get metrics data
        metrics_data = collector.get_metrics()

        if output == "json":
            # Parse Prometheus format to JSON
            metrics_dict = {}
            for line in metrics_data.decode().split("\n"):
                if line and not line.startswith("#"):
                    parts = line.split(" ")
                    if len(parts) == 2:
                        metrics_dict[parts[0]] = float(parts[1])

            if metric:
                filtered = {k: v for k, v in metrics_dict.items() if metric in k}
                click.echo(json.dumps(filtered, indent=2))
            else:
                click.echo(json.dumps(metrics_dict, indent=2))

        else:
            # Display as table
            table = Table(title="Current Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            for line in metrics_data.decode().split("\n"):
                if line and not line.startswith("#"):
                    parts = line.split(" ")
                    if len(parts) == 2:
                        if not metric or metric in parts[0]:
                            table.add_row(parts[0], parts[1])

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing metrics: {e}[/red]")
        raise click.ClickException(str(e))


# Alerts commands
@telemetry.group()
def alerts():
    """Alert management commands"""


@alerts.command("list")
@click.option(
    "--status",
    type=click.Choice(["active", "history", "all"]),
    default="active",
    help="Alert status to show",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
def list_alerts(status: str, output: str):
    """List alerts"""
    try:
        service = get_monitoring_service()
        alert_manager = service.alert_manager

        if status == "active":
            alerts = alert_manager.get_active_alerts()
        elif status == "history":
            alerts = alert_manager.get_alert_history(24)
        else:
            alerts = (
                alert_manager.get_active_alerts() + alert_manager.get_alert_history(24)
            )

        if output == "json":
            data = [a.to_dict() for a in alerts]
            click.echo(json.dumps(data, indent=2))
        elif output == "yaml":
            data = [a.to_dict() for a in alerts]
            click.echo(yaml.dump(data))
        else:
            # Display as table
            table = Table(title="Alerts")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Severity", style="white")
            table.add_column("Status", style="white")
            table.add_column("Message", style="white")
            table.add_column("Fired At", style="yellow")

            for alert in alerts:
                severity_style = {
                    AlertSeverity.INFO: "blue",
                    AlertSeverity.WARNING: "yellow",
                    AlertSeverity.ERROR: "red",
                    AlertSeverity.CRITICAL: "red bold",
                }.get(alert.severity, "white")

                table.add_row(
                    alert.alert_id[:12] + "...",
                    alert.name,
                    f"[{severity_style}]{alert.severity.value}[/{severity_style}]",
                    alert.status.value,
                    (
                        alert.message[:50] + "..."
                        if len(alert.message) > 50
                        else alert.message
                    ),
                    alert.fired_at.strftime("%Y-%m-%d %H:%M:%S"),
                )

            console.print(table)
            console.print(f"\nTotal: {len(alerts)} alerts")

    except Exception as e:
        console.print(f"[red]Error listing alerts: {e}[/red]")
        raise click.ClickException(str(e))


@alerts.command()
@click.argument("alert_id")
@click.argument("user")
def acknowledge(alert_id: str, user: str):
    """Acknowledge an alert"""
    try:
        service = get_monitoring_service()
        service.alert_manager.acknowledge_alert(alert_id, user)
        console.print(f"[green]Alert {alert_id} acknowledged by {user}[/green]")

    except Exception as e:
        console.print(f"[red]Error acknowledging alert: {e}[/red]")
        raise click.ClickException(str(e))


# Logs commands
@telemetry.group()
def logs():
    """Log analysis commands"""


@logs.command("tail")
@click.option(
    "--level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Filter by log level",
)
@click.option("--component", help="Filter by component")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
def tail(level, component, follow):
    """Tail logs"""
    try:
        aggregator = get_log_aggregator()

        if follow:
            console.print("[cyan]Following logs... (Ctrl+C to stop)[/cyan]")

            last_timestamp = DeterministicClock.utcnow()

            try:
                while True:
                    # Get new logs
                    logs = aggregator.get_logs(
                        level=level,
                        component=component,
                        start_time=last_timestamp,
                        limit=100,
                    )

                    for log in logs:
                        # Format log entry
                        level_style = {
                            "DEBUG": "white",
                            "INFO": "green",
                            "WARNING": "yellow",
                            "ERROR": "red",
                            "CRITICAL": "red bold",
                        }.get(log.level.value, "white")

                        console.print(
                            f"[dim]{log.timestamp.strftime('%H:%M:%S')}[/dim] "
                            f"[{level_style}]{log.level.value:8}[/{level_style}] "
                            f"[cyan]{log.context.component or 'unknown':15}[/cyan] "
                            f"{log.message}"
                        )

                        last_timestamp = max(last_timestamp, log.timestamp)

                    time.sleep(1)

            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped following logs[/yellow]")
        else:
            # Get recent logs
            logs = aggregator.get_logs(level=level, component=component, limit=50)

            for log in logs:
                level_style = {
                    "DEBUG": "white",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red bold",
                }.get(log.level.value, "white")

                console.print(
                    f"[dim]{log.timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/dim] "
                    f"[{level_style}]{log.level.value:8}[/{level_style}] "
                    f"[cyan]{log.context.component or 'unknown':15}[/cyan] "
                    f"{log.message}"
                )

    except Exception as e:
        console.print(f"[red]Error tailing logs: {e}[/red]")
        raise click.ClickException(str(e))


@logs.command("stats")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
def stats(output: str):
    """Show log statistics"""
    try:
        aggregator = get_log_aggregator()
        stats = aggregator.get_statistics()

        if output == "json":
            click.echo(json.dumps(stats, indent=2))
        elif output == "yaml":
            click.echo(yaml.dump(stats))
        else:
            console.print(
                Panel(
                    f"Total Logs: {stats['total_logs']}\n"
                    f"Time Range: {stats.get('time_range', {}).get('span_hours', 0):.1f} hours",
                    title="Log Statistics",
                )
            )

            # Log counts by level
            if stats.get("log_counts"):
                table = Table(title="Logs by Level")
                table.add_column("Level", style="cyan")
                table.add_column("Component", style="white")
                table.add_column("Count", style="yellow")

                for level, components in stats["log_counts"].items():
                    for component, count in components.items():
                        table.add_row(level, component, str(count))

                console.print(table)

            # Error patterns
            if stats.get("error_patterns"):
                table = Table(title="Error Patterns")
                table.add_column("Pattern", style="cyan")
                table.add_column("Count", style="red")

                for pattern, count in stats["error_patterns"].items():
                    table.add_row(pattern, str(count))

                console.print(table)

    except Exception as e:
        console.print(f"[red]Error getting log stats: {e}[/red]")
        raise click.ClickException(str(e))


# Performance commands
@telemetry.group()
def performance():
    """Performance monitoring commands"""


@performance.command()
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
def status(output: str):
    """Show performance status"""
    try:
        monitor = get_performance_monitor()

        # Get current stats
        memory = monitor.get_memory_usage()
        cpu = monitor.get_cpu_usage()

        data = {"memory": memory, "cpu": cpu}

        if output == "json":
            click.echo(json.dumps(data, indent=2))
        elif output == "yaml":
            click.echo(yaml.dump(data))
        else:
            console.print(
                Panel(
                    f"[cyan]Memory Usage[/cyan]\n"
                    f"RSS: {memory['rss_mb']:.1f} MB\n"
                    f"VMS: {memory['vms_mb']:.1f} MB\n"
                    f"Percent: {memory['percent']:.1f}%\n"
                    f"Available: {memory['available_mb']:.1f} MB\n\n"
                    f"[cyan]CPU Usage[/cyan]\n"
                    f"Process: {cpu['percent']:.1f}%\n"
                    f"System: {cpu['system_percent']:.1f}%\n"
                    f"Threads: {cpu['threads']}",
                    title="Performance Status",
                )
            )

    except Exception as e:
        console.print(f"[red]Error getting performance status: {e}[/red]")
        raise click.ClickException(str(e))


@performance.command()
@click.option("--duration", type=int, default=60, help="Profiling duration in seconds")
def profile(duration: int):
    """Profile application performance"""
    try:
        monitor = get_performance_monitor()

        console.print(f"[cyan]Starting CPU profiling for {duration} seconds...[/cyan]")

        monitor.start_profiling()

        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Profiling for {duration} seconds...", total=duration
            )

            for i in range(duration):
                time.sleep(1)
                progress.update(task, advance=1)

        # Stop profiling
        results = monitor.stop_profiling()

        console.print("\n[green]Profiling complete![/green]\n")
        console.print(results)

    except KeyboardInterrupt:
        monitor.stop_profiling()
        console.print("\n[yellow]Profiling cancelled[/yellow]")
    except Exception as e:
        console.print(f"[red]Error profiling: {e}[/red]")
        raise click.ClickException(str(e))


# Dashboard commands
@telemetry.group()
def dashboard():
    """Dashboard management commands"""


@dashboard.command("list")
def list_dashboards():
    """List available dashboards"""
    try:
        service = get_monitoring_service()

        table = Table(title="Available Dashboards")
        table.add_column("Name", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Description", style="white")
        table.add_column("Panels", style="yellow")

        for name, dash in service.dashboards.items():
            table.add_row(
                dash.name,
                dash.title,
                (
                    dash.description[:50] + "..."
                    if len(dash.description) > 50
                    else dash.description
                ),
                str(len(dash.panels)),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing dashboards: {e}[/red]")
        raise click.ClickException(str(e))


@dashboard.command("export")
@click.argument("name")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "yaml", "grafana"]),
    default="json",
    help="Export format",
)
@click.option("--output", "-o", help="Output file")
def export_dashboard(name: str, format: str, output: str):
    """Export dashboard configuration"""
    try:
        service = get_monitoring_service()

        config = service.export_dashboard(name, format)

        if output:
            Path(output).write_text(config)
            console.print(f"[green]Dashboard exported to {output}[/green]")
        else:
            click.echo(config)

    except Exception as e:
        console.print(f"[red]Error exporting dashboard: {e}[/red]")
        raise click.ClickException(str(e))


# Main monitoring command
@telemetry.command()
@click.option("--metrics-port", type=int, default=8000, help="Metrics server port")
@click.option("--health-port", type=int, default=8080, help="Health check port")
def start(metrics_port: int, health_port: int):
    """Start monitoring service"""
    try:
        service = get_monitoring_service()

        console.print(
            Panel(
                f"[green]Starting GreenLang Monitoring Service[/green]\n\n"
                f"Metrics: http://localhost:{metrics_port}/metrics\n"
                f"Health: http://localhost:{health_port}/health\n\n"
                f"Press Ctrl+C to stop",
                title="Monitoring Service",
            )
        )

        # Start service
        service.start(metrics_port)

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            service.stop()
            console.print("\n[yellow]Monitoring service stopped[/yellow]")

    except Exception as e:
        console.print(f"[red]Error starting monitoring service: {e}[/red]")
        raise click.ClickException(str(e))
