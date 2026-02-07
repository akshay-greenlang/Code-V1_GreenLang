# -*- coding: utf-8 -*-
"""
gl agent status - Display the status of GreenLang agents.

Shows a table (or JSON) of agent key, version, status, health, uptime,
and last execution time.  Supports filtering by agent key or status.

Example:
    gl agent status --all
    gl agent status --agent-key carbon-calc
    gl agent status --all --format json --filter-status running

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Mock registry (replaced by real registry in production)
# ---------------------------------------------------------------------------

def _fetch_agent_statuses(
    agent_key: Optional[str] = None,
    filter_status: Optional[str] = None,
) -> list[dict]:
    """Fetch agent statuses from the lifecycle manager.

    In production this queries the Agent Factory registry database.
    Returns a list of agent status dictionaries.
    """
    # Placeholder data -- production would query the registry
    all_agents = [
        {
            "agent_key": "carbon-calc",
            "version": "2.1.0",
            "status": "running",
            "health": "healthy",
            "uptime": "14d 3h 22m",
            "last_execution": "2026-02-05T09:14:02Z",
            "env": "prod",
        },
        {
            "agent_key": "eudr-compliance",
            "version": "1.3.0",
            "status": "running",
            "health": "healthy",
            "uptime": "7d 12h 5m",
            "last_execution": "2026-02-05T09:12:41Z",
            "env": "prod",
        },
        {
            "agent_key": "csrd-disclosure",
            "version": "0.9.1",
            "status": "degraded",
            "health": "degraded",
            "uptime": "2d 1h 11m",
            "last_execution": "2026-02-05T08:59:13Z",
            "env": "staging",
        },
        {
            "agent_key": "scope3-mapper",
            "version": "1.0.0",
            "status": "failed",
            "health": "unhealthy",
            "uptime": "0d 0h 0m",
            "last_execution": "2026-02-04T22:00:05Z",
            "env": "dev",
        },
        {
            "agent_key": "sbti-validator",
            "version": "1.2.0",
            "status": "running",
            "health": "healthy",
            "uptime": "30d 5h 8m",
            "last_execution": "2026-02-05T09:10:00Z",
            "env": "prod",
        },
    ]

    if agent_key:
        all_agents = [a for a in all_agents if a["agent_key"] == agent_key]

    if filter_status:
        all_agents = [a for a in all_agents if a["status"] == filter_status]

    return all_agents


_STATUS_COLORS = {
    "running": "green",
    "degraded": "yellow",
    "failed": "red",
    "stopped": "dim",
    "deploying": "cyan",
    "draining": "magenta",
}

_HEALTH_COLORS = {
    "healthy": "green",
    "degraded": "yellow",
    "unhealthy": "red",
    "unknown": "dim",
}


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

def status(
    agent_key: Optional[str] = typer.Option(
        None,
        "--agent-key", "-k",
        help="Show status for a specific agent.",
    ),
    all_agents: bool = typer.Option(
        False,
        "--all", "-a",
        help="List status for all registered agents.",
    ),
    format_: str = typer.Option(
        "table",
        "--format", "-f",
        help="Output format: table or json.",
    ),
    filter_status: Optional[str] = typer.Option(
        None,
        "--filter-status",
        help="Filter by status (running, failed, degraded, etc.).",
    ),
) -> None:
    """
    Display the status of one or all GreenLang agents.

    Shows key, version, lifecycle status, health, uptime, and last
    execution timestamp.

    Example:
        gl agent status --all
        gl agent status --agent-key carbon-calc --format json
    """
    if not agent_key and not all_agents:
        console.print("[yellow]Specify --agent-key or --all to view agent status.[/yellow]")
        raise typer.Exit(1)

    agents = _fetch_agent_statuses(agent_key=agent_key, filter_status=filter_status)

    if not agents:
        console.print("[yellow]No agents found matching the criteria.[/yellow]")
        raise typer.Exit(0)

    # JSON output
    if format_ == "json":
        console.print(json.dumps(agents, indent=2))
        return

    # Table output
    console.print(Panel(
        f"[bold cyan]Agent Status[/bold cyan]  "
        f"({len(agents)} agent{'s' if len(agents) != 1 else ''})",
        border_style="cyan",
    ))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Agent Key", style="cyan")
    table.add_column("Version", justify="right")
    table.add_column("Env")
    table.add_column("Status")
    table.add_column("Health")
    table.add_column("Uptime")
    table.add_column("Last Execution")

    for agent in agents:
        s_color = _STATUS_COLORS.get(agent["status"], "white")
        h_color = _HEALTH_COLORS.get(agent["health"], "white")

        table.add_row(
            agent["agent_key"],
            agent["version"],
            agent["env"],
            f"[{s_color}]{agent['status']}[/{s_color}]",
            f"[{h_color}]{agent['health']}[/{h_color}]",
            agent["uptime"],
            agent["last_execution"],
        )

    console.print(table)
