# -*- coding: utf-8 -*-
"""
gl agent inspect - Display comprehensive information about a GreenLang agent.

Shows agent metadata, dependency tree, configuration (with redacted secrets),
and recent execution metrics (last 24 hours).

Example:
    gl agent inspect --agent-key carbon-calc
    gl agent inspect --agent-key eudr --deps --config --metrics
    gl agent inspect --agent-key my-agent --verbose

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

_SECRET_KEYS = re.compile(
    r"(password|secret|token|api_key|private_key|credentials)",
    re.IGNORECASE,
)


def _redact_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the config dict with secret values redacted."""
    redacted: dict[str, Any] = {}
    for key, value in config.items():
        if _SECRET_KEYS.search(key):
            redacted[key] = "***REDACTED***"
        elif isinstance(value, dict):
            redacted[key] = _redact_config(value)
        else:
            redacted[key] = value
    return redacted


# ---------------------------------------------------------------------------
# Mock data providers (replaced by real registry in production)
# ---------------------------------------------------------------------------

def _fetch_agent_info(agent_key: str) -> dict[str, Any]:
    """Fetch agent metadata from the registry."""
    return {
        "agent_key": agent_key,
        "version": "2.1.0",
        "type": "deterministic",
        "description": f"GreenLang agent: {agent_key}",
        "author": "GreenLang Platform Team",
        "license": "Apache-2.0",
        "created_at": "2025-11-01T10:00:00Z",
        "updated_at": "2026-02-04T14:30:00Z",
        "status": "running",
        "env": "prod",
        "entry_point": "agent.py",
        "python_version": ">=3.11",
    }


def _fetch_dependencies(agent_key: str) -> dict[str, Any]:
    """Fetch agent dependency tree."""
    return {
        "python": [
            {"name": "pydantic", "version": "2.6.0", "required": ">=2.0"},
            {"name": "httpx", "version": "0.27.0", "required": ">=0.25"},
            {"name": "numpy", "version": "1.26.4", "required": ">=1.24"},
        ],
        "agents": [
            {
                "name": "emission-factor-lookup",
                "version": "1.5.0",
                "required": ">=1.0",
                "deps": [
                    {"name": "unit-converter", "version": "1.0.0", "required": ">=1.0"},
                ],
            },
            {
                "name": "data-validator",
                "version": "1.2.0",
                "required": ">=1.0",
                "deps": [],
            },
        ],
    }


def _fetch_config(agent_key: str) -> dict[str, Any]:
    """Fetch agent configuration."""
    return {
        "runtime": {
            "memory_mb": 512,
            "timeout_seconds": 30,
            "max_retries": 3,
        },
        "connections": {
            "database_url": "postgresql://agent_user@db:5432/greenlang",
            "database_password": "super-secret-password",
            "redis_url": "redis://cache:6379/0",
            "api_key": "sk-live-abc123xyz",
        },
        "feature_flags": {
            "enable_batch_processing": True,
            "enable_caching": True,
            "max_batch_size": 5000,
        },
    }


def _fetch_metrics(agent_key: str) -> dict[str, Any]:
    """Fetch recent execution metrics (last 24h)."""
    return {
        "execution_count": 12847,
        "success_count": 12801,
        "error_count": 46,
        "success_rate": "99.64%",
        "avg_duration_ms": 142.3,
        "p50_duration_ms": 98.0,
        "p95_duration_ms": 310.5,
        "p99_duration_ms": 892.1,
        "total_cost_usd": 4.82,
        "queue_depth": 3,
        "active_workers": 4,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _build_dep_tree(deps: dict[str, Any], agent_key: str) -> Tree:
    """Build a rich Tree representing the dependency graph."""
    tree = Tree(f"[bold cyan]{agent_key}[/bold cyan]")

    # Python dependencies
    py_branch = tree.add("[bold]Python Dependencies[/bold]")
    for dep in deps.get("python", []):
        py_branch.add(
            f"[green]{dep['name']}[/green] {dep['version']} "
            f"[dim](requires {dep['required']})[/dim]"
        )

    # Agent dependencies (recursive)
    agent_branch = tree.add("[bold]Agent Dependencies[/bold]")
    for dep in deps.get("agents", []):
        node = agent_branch.add(
            f"[cyan]{dep['name']}[/cyan] {dep['version']} "
            f"[dim](requires {dep['required']})[/dim]"
        )
        for sub in dep.get("deps", []):
            node.add(
                f"[cyan]{sub['name']}[/cyan] {sub['version']} "
                f"[dim](requires {sub['required']})[/dim]"
            )

    return tree


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

def inspect_agent(
    agent_key: str = typer.Option(
        ...,
        "--agent-key", "-k",
        help="Agent key to inspect.",
    ),
    deps: bool = typer.Option(
        False,
        "--deps",
        help="Show dependency tree.",
    ),
    config: bool = typer.Option(
        False,
        "--config",
        help="Show configuration (secrets redacted).",
    ),
    metrics: bool = typer.Option(
        False,
        "--metrics",
        help="Show recent execution metrics (last 24h).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show all sections.",
    ),
) -> None:
    """
    Display comprehensive information about a GreenLang agent.

    Without flags shows basic metadata.  Use --deps, --config, or --metrics
    to show additional sections, or --verbose for everything.

    Example:
        gl agent inspect --agent-key carbon-calc --deps --metrics
    """
    show_all = verbose

    # Basic info
    info = _fetch_agent_info(agent_key)

    console.print(Panel(
        f"[bold cyan]{info['agent_key']}[/bold cyan]  "
        f"v{info['version']}  [{info['status']}]",
        title="Agent Inspector",
        border_style="cyan",
    ))

    info_table = Table(title="Agent Metadata", show_header=False)
    info_table.add_column("Field", style="cyan", width=20)
    info_table.add_column("Value")

    for key in (
        "agent_key", "version", "type", "description", "author",
        "license", "status", "env", "entry_point", "python_version",
        "created_at", "updated_at",
    ):
        info_table.add_row(key, str(info.get(key, "N/A")))

    console.print(info_table)

    # Dependencies
    if deps or show_all:
        console.print("")
        dep_data = _fetch_dependencies(agent_key)
        tree = _build_dep_tree(dep_data, agent_key)
        console.print(tree)

    # Configuration
    if config or show_all:
        console.print("")
        raw_config = _fetch_config(agent_key)
        safe_config = _redact_config(raw_config)

        console.print(Panel(
            json.dumps(safe_config, indent=2),
            title="Configuration (secrets redacted)",
            border_style="yellow",
        ))

    # Metrics
    if metrics or show_all:
        console.print("")
        m = _fetch_metrics(agent_key)

        metrics_table = Table(title="Execution Metrics (last 24h)")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right")

        metrics_table.add_row("Executions", f"{m['execution_count']:,}")
        metrics_table.add_row("Successes", f"[green]{m['success_count']:,}[/green]")
        metrics_table.add_row("Errors", f"[red]{m['error_count']:,}[/red]")
        metrics_table.add_row("Success Rate", m["success_rate"])
        metrics_table.add_row("Avg Duration", f"{m['avg_duration_ms']:.1f} ms")
        metrics_table.add_row("P50 Duration", f"{m['p50_duration_ms']:.1f} ms")
        metrics_table.add_row("P95 Duration", f"{m['p95_duration_ms']:.1f} ms")
        metrics_table.add_row("P99 Duration", f"{m['p99_duration_ms']:.1f} ms")
        metrics_table.add_row("Total Cost", f"${m['total_cost_usd']:.2f}")
        metrics_table.add_row("Queue Depth", str(m["queue_depth"]))
        metrics_table.add_row("Active Workers", str(m["active_workers"]))

        console.print(metrics_table)
