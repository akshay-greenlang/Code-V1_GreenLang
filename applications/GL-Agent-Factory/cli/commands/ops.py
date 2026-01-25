"""
GL-Agent-Factory CLI Operations Commands

Provides operational commands for deployed agents including:
- Real-time monitoring dashboard
- Log viewing and filtering
- Agent rollback management
- Health checks and diagnostics
- Alert management
- Deployment history
- Metrics collection
- Status reporting

Example:
    $ glang ops monitor GL-020 --env production
    $ glang ops logs GL-020 --tail 100 --level error
    $ glang ops rollback GL-020 --to-version 1.2.0
"""

import json
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

app = typer.Typer(name="ops", help="Agent operations and monitoring commands")
console = Console()


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# =============================================================================
# MONITOR COMMAND
# =============================================================================

@app.command()
def monitor(
    agent_id: str = typer.Argument(..., help="Agent ID to monitor (e.g., GL-020)"),
    env: Environment = typer.Option(Environment.PRODUCTION, "--env", "-e", help="Environment"),
    interval: int = typer.Option(5, "--interval", "-i", help="Refresh interval in seconds"),
    metrics: List[str] = typer.Option(
        ["cpu", "memory", "latency", "throughput"],
        "--metrics", "-m",
        help="Metrics to display"
    ),
):
    """Real-time monitoring dashboard for deployed agents."""
    console.print(f"[bold green]Starting monitor for {agent_id} in {env.value}...[/bold green]")

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                # Generate mock metrics (replace with actual data collection)
                metrics_data = _collect_metrics(agent_id, env.value, metrics)
                dashboard = _build_dashboard(agent_id, env.value, metrics_data)
                live.update(dashboard)
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped.[/yellow]")


def _collect_metrics(agent_id: str, env: str, metric_names: List[str]) -> dict:
    """Collect metrics for an agent (mock implementation)."""
    import random
    return {
        "cpu": round(random.uniform(10, 80), 1),
        "memory": round(random.uniform(200, 800), 0),
        "latency": round(random.uniform(50, 200), 1),
        "throughput": round(random.uniform(100, 500), 0),
        "requests_total": random.randint(10000, 100000),
        "error_rate": round(random.uniform(0, 2), 2),
        "uptime": "99.95%",
        "last_restart": "2 days ago",
    }


def _build_dashboard(agent_id: str, env: str, metrics: dict) -> Panel:
    """Build the monitoring dashboard panel."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_column("Status", justify="center")

    # Add metric rows
    cpu_status = "🟢" if metrics["cpu"] < 70 else "🟡" if metrics["cpu"] < 90 else "🔴"
    table.add_row("CPU Usage", f"{metrics['cpu']}%", cpu_status)

    mem_status = "🟢" if metrics["memory"] < 700 else "🟡" if metrics["memory"] < 900 else "🔴"
    table.add_row("Memory", f"{metrics['memory']} MB", mem_status)

    lat_status = "🟢" if metrics["latency"] < 100 else "🟡" if metrics["latency"] < 200 else "🔴"
    table.add_row("Latency (p95)", f"{metrics['latency']} ms", lat_status)

    table.add_row("Throughput", f"{metrics['throughput']} req/s", "🟢")
    table.add_row("Total Requests", f"{metrics['requests_total']:,}", "")

    err_status = "🟢" if metrics["error_rate"] < 1 else "🟡" if metrics["error_rate"] < 5 else "🔴"
    table.add_row("Error Rate", f"{metrics['error_rate']}%", err_status)

    table.add_row("Uptime", metrics["uptime"], "🟢")
    table.add_row("Last Restart", metrics["last_restart"], "")

    return Panel(
        table,
        title=f"[bold]{agent_id}[/bold] - {env.upper()}",
        subtitle=f"Updated: {datetime.now().strftime('%H:%M:%S')}",
        border_style="green",
    )


# =============================================================================
# LOGS COMMAND
# =============================================================================

@app.command()
def logs(
    agent_id: str = typer.Argument(..., help="Agent ID to view logs for"),
    env: Environment = typer.Option(Environment.PRODUCTION, "--env", "-e"),
    tail: int = typer.Option(100, "--tail", "-n", help="Number of lines to show"),
    level: Optional[LogLevel] = typer.Option(None, "--level", "-l", help="Filter by log level"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    since: Optional[str] = typer.Option(None, "--since", help="Show logs since (e.g., '1h', '30m')"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search pattern"),
):
    """View and filter agent logs."""
    console.print(f"[dim]Fetching logs for {agent_id} ({env.value})...[/dim]")

    # Generate mock log entries
    log_entries = _generate_mock_logs(agent_id, tail, level)

    if search:
        log_entries = [l for l in log_entries if search.lower() in l["message"].lower()]

    for entry in log_entries:
        _print_log_entry(entry)

    if follow:
        console.print("\n[dim]Following logs (Ctrl+C to stop)...[/dim]")
        try:
            while True:
                time.sleep(2)
                new_entry = _generate_mock_logs(agent_id, 1, level)[0]
                _print_log_entry(new_entry)
        except KeyboardInterrupt:
            pass


def _generate_mock_logs(agent_id: str, count: int, level_filter: Optional[LogLevel]) -> List[dict]:
    """Generate mock log entries."""
    import random
    levels = ["debug", "info", "info", "info", "warning", "error"]
    messages = [
        "Request processed successfully",
        "Calculation completed: efficiency=92.5%",
        "Cache hit for emission factor lookup",
        "Database query executed in 45ms",
        "Rate limit approaching threshold",
        "Connection pool exhausted, retrying...",
        "Validation error in input data",
        "External API timeout, using cached data",
    ]

    entries = []
    for i in range(count):
        lvl = random.choice(levels)
        if level_filter and lvl != level_filter.value:
            continue
        entries.append({
            "timestamp": (datetime.now() - timedelta(minutes=count-i)).isoformat(),
            "level": lvl,
            "message": random.choice(messages),
            "agent_id": agent_id,
        })
    return entries


def _print_log_entry(entry: dict):
    """Print a formatted log entry."""
    level_colors = {
        "debug": "dim",
        "info": "blue",
        "warning": "yellow",
        "error": "red",
        "critical": "bold red",
    }
    color = level_colors.get(entry["level"], "white")
    console.print(
        f"[dim]{entry['timestamp']}[/dim] [{color}]{entry['level'].upper():8}[/{color}] {entry['message']}"
    )


# =============================================================================
# ROLLBACK COMMAND
# =============================================================================

@app.command()
def rollback(
    agent_id: str = typer.Argument(..., help="Agent ID to rollback"),
    to_version: str = typer.Option(..., "--to-version", "-v", help="Target version"),
    env: Environment = typer.Option(Environment.PRODUCTION, "--env", "-e"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Rollback an agent to a previous version."""
    console.print(f"\n[bold]Rollback Plan for {agent_id}[/bold]")
    console.print(f"  Environment: {env.value}")
    console.print(f"  Target Version: {to_version}")

    # Show deployment history
    history = [
        {"version": "1.3.0", "deployed": "2024-01-15", "status": "current"},
        {"version": "1.2.0", "deployed": "2024-01-10", "status": "available"},
        {"version": "1.1.0", "deployed": "2024-01-05", "status": "available"},
    ]

    table = Table(title="Deployment History")
    table.add_column("Version")
    table.add_column("Deployed")
    table.add_column("Status")

    for h in history:
        style = "green" if h["status"] == "current" else "dim"
        table.add_row(h["version"], h["deployed"], h["status"], style=style)

    console.print(table)

    if dry_run:
        console.print("\n[yellow]DRY RUN - No changes will be made[/yellow]")
        return

    if not force:
        confirm = typer.confirm(f"\nRollback {agent_id} to version {to_version}?")
        if not confirm:
            console.print("[yellow]Rollback cancelled.[/yellow]")
            raise typer.Abort()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Rolling back...", total=None)
        time.sleep(2)

    console.print(f"\n[bold green]✓ Successfully rolled back {agent_id} to {to_version}[/bold green]")


# =============================================================================
# HEALTH COMMAND
# =============================================================================

@app.command()
def health(
    agent_id: Optional[str] = typer.Argument(None, help="Agent ID (optional, shows all if omitted)"),
    env: Environment = typer.Option(Environment.PRODUCTION, "--env", "-e"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed health info"),
):
    """Check health status of deployed agents."""
    if agent_id:
        agents = [agent_id]
    else:
        agents = ["GL-020", "GL-021", "GL-022", "GL-031", "GL-046"]

    table = Table(title=f"Agent Health Status ({env.value})")
    table.add_column("Agent ID", style="cyan")
    table.add_column("Status")
    table.add_column("Version")
    table.add_column("Uptime")
    table.add_column("Last Check")

    for aid in agents:
        import random
        status = random.choice(["🟢 Healthy", "🟢 Healthy", "🟢 Healthy", "🟡 Degraded", "🔴 Unhealthy"])
        version = f"1.{random.randint(0,5)}.{random.randint(0,10)}"
        uptime = f"{random.randint(1,30)} days"
        table.add_row(aid, status, version, uptime, "Just now")

    console.print(table)


# =============================================================================
# ALERTS COMMAND
# =============================================================================

@app.command()
def alerts(
    agent_id: Optional[str] = typer.Argument(None, help="Filter by agent ID"),
    env: Environment = typer.Option(Environment.PRODUCTION, "--env", "-e"),
    active_only: bool = typer.Option(True, "--active/--all", help="Show only active alerts"),
    severity: Optional[str] = typer.Option(None, "--severity", help="Filter by severity"),
):
    """View and manage agent alerts."""
    alerts_data = [
        {"id": "ALT-001", "agent": "GL-046", "severity": "warning", "message": "High memory usage", "time": "5m ago", "status": "active"},
        {"id": "ALT-002", "agent": "GL-031", "severity": "info", "message": "Scheduled maintenance", "time": "1h ago", "status": "active"},
        {"id": "ALT-003", "agent": "GL-020", "severity": "error", "message": "Connection timeout", "time": "2h ago", "status": "resolved"},
    ]

    if agent_id:
        alerts_data = [a for a in alerts_data if a["agent"] == agent_id]
    if active_only:
        alerts_data = [a for a in alerts_data if a["status"] == "active"]
    if severity:
        alerts_data = [a for a in alerts_data if a["severity"] == severity]

    if not alerts_data:
        console.print("[green]No alerts found.[/green]")
        return

    table = Table(title="Active Alerts")
    table.add_column("ID")
    table.add_column("Agent")
    table.add_column("Severity")
    table.add_column("Message")
    table.add_column("Time")

    severity_colors = {"info": "blue", "warning": "yellow", "error": "red", "critical": "bold red"}

    for alert in alerts_data:
        color = severity_colors.get(alert["severity"], "white")
        table.add_row(
            alert["id"],
            alert["agent"],
            Text(alert["severity"].upper(), style=color),
            alert["message"],
            alert["time"],
        )

    console.print(table)


# =============================================================================
# HISTORY COMMAND
# =============================================================================

@app.command()
def history(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    env: Environment = typer.Option(Environment.PRODUCTION, "--env", "-e"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of entries"),
):
    """View deployment history for an agent."""
    history_data = [
        {"version": "1.3.0", "action": "deploy", "user": "ci-bot", "time": "2024-01-15 10:30", "status": "success"},
        {"version": "1.2.5", "action": "rollback", "user": "ops-team", "time": "2024-01-14 15:45", "status": "success"},
        {"version": "1.2.5", "action": "deploy", "user": "ci-bot", "time": "2024-01-14 14:00", "status": "failed"},
        {"version": "1.2.0", "action": "deploy", "user": "ci-bot", "time": "2024-01-10 09:15", "status": "success"},
    ]

    table = Table(title=f"Deployment History: {agent_id}")
    table.add_column("Version")
    table.add_column("Action")
    table.add_column("User")
    table.add_column("Time")
    table.add_column("Status")

    status_styles = {"success": "green", "failed": "red", "pending": "yellow"}

    for h in history_data[:limit]:
        style = status_styles.get(h["status"], "white")
        table.add_row(h["version"], h["action"], h["user"], h["time"], Text(h["status"], style=style))

    console.print(table)


# =============================================================================
# METRICS COMMAND
# =============================================================================

@app.command()
def metrics(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    env: Environment = typer.Option(Environment.PRODUCTION, "--env", "-e"),
    period: str = typer.Option("1h", "--period", "-p", help="Time period (e.g., 1h, 24h, 7d)"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, csv)"),
):
    """Retrieve metrics for an agent."""
    import random

    metrics_data = {
        "agent_id": agent_id,
        "environment": env.value,
        "period": period,
        "metrics": {
            "requests_total": random.randint(50000, 200000),
            "requests_per_second_avg": round(random.uniform(50, 200), 2),
            "latency_p50_ms": round(random.uniform(30, 80), 1),
            "latency_p95_ms": round(random.uniform(100, 250), 1),
            "latency_p99_ms": round(random.uniform(200, 500), 1),
            "error_rate_percent": round(random.uniform(0, 2), 2),
            "cpu_avg_percent": round(random.uniform(20, 60), 1),
            "memory_avg_mb": round(random.uniform(200, 600), 0),
        }
    }

    if format == "json":
        console.print_json(json.dumps(metrics_data))
    else:
        table = Table(title=f"Metrics: {agent_id} ({period})")
        table.add_column("Metric")
        table.add_column("Value", justify="right")

        for key, value in metrics_data["metrics"].items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)


# =============================================================================
# STATUS COMMAND
# =============================================================================

@app.command()
def status(
    env: Environment = typer.Option(Environment.PRODUCTION, "--env", "-e"),
    format: str = typer.Option("table", "--format", "-f", help="Output format"),
):
    """Show overall system status."""
    status_data = {
        "environment": env.value,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_agents": 21,
            "healthy": 19,
            "degraded": 1,
            "unhealthy": 1,
        },
        "services": {
            "api_gateway": "🟢 Healthy",
            "message_queue": "🟢 Healthy",
            "database": "🟢 Healthy",
            "cache": "🟢 Healthy",
            "ml_pipeline": "🟡 Degraded",
        }
    }

    # Summary panel
    summary = Table(show_header=False, box=None)
    summary.add_column(style="bold")
    summary.add_column(justify="right")
    summary.add_row("Total Agents", str(status_data["summary"]["total_agents"]))
    summary.add_row("Healthy", f"[green]{status_data['summary']['healthy']}[/green]")
    summary.add_row("Degraded", f"[yellow]{status_data['summary']['degraded']}[/yellow]")
    summary.add_row("Unhealthy", f"[red]{status_data['summary']['unhealthy']}[/red]")

    console.print(Panel(summary, title=f"System Status - {env.value.upper()}", border_style="cyan"))

    # Services table
    services_table = Table(title="Core Services")
    services_table.add_column("Service")
    services_table.add_column("Status")

    for service, status in status_data["services"].items():
        services_table.add_row(service.replace("_", " ").title(), status)

    console.print(services_table)


if __name__ == "__main__":
    app()
