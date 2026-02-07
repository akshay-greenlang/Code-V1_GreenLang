# -*- coding: utf-8 -*-
"""
gl agent logs - Fetch and display logs for a GreenLang agent.

Supports tailing a fixed number of lines, streaming new log entries with
--follow, filtering by severity level and time window, and colorizing
output by log level.

Example:
    gl agent logs --agent-key carbon-calc
    gl agent logs --agent-key carbon-calc --tail 50 --level error
    gl agent logs --agent-key eudr --follow --since 1h

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import typer
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Log level styling
# ---------------------------------------------------------------------------

_LEVEL_STYLES = {
    "DEBUG": "dim",
    "INFO": "green",
    "WARN": "yellow",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold red",
}

_LEVEL_PRIORITY = {
    "debug": 0,
    "info": 1,
    "warn": 2,
    "warning": 2,
    "error": 3,
    "critical": 4,
}


def _parse_since(since: str) -> datetime:
    """Parse a relative time string like '1h', '30m', '2d' into a UTC datetime.

    Args:
        since: Relative time string.  Supported suffixes: s, m, h, d.

    Returns:
        UTC datetime representing now minus the offset.

    Raises:
        typer.BadParameter: If the format is not recognized.
    """
    units = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
    if not since:
        return datetime.now(timezone.utc) - timedelta(hours=1)
    suffix = since[-1].lower()
    if suffix not in units:
        raise typer.BadParameter(
            f"Invalid time suffix '{suffix}'. Use s, m, h, or d."
        )
    try:
        value = int(since[:-1])
    except ValueError:
        raise typer.BadParameter(f"Cannot parse time value from '{since}'.")
    delta = timedelta(**{units[suffix]: value})
    return datetime.now(timezone.utc) - delta


# ---------------------------------------------------------------------------
# Mock log source (replaced by Loki/log aggregation in production)
# ---------------------------------------------------------------------------

_SAMPLE_MESSAGES = [
    ("INFO", "Processing batch 42 of emission data"),
    ("INFO", "Successfully calculated Scope 1 emissions: 1234.56 tCO2e"),
    ("DEBUG", "Cache hit for emission_factor:electricity:US-WEST"),
    ("WARN", "Slow query detected: 2.3s for supplier lookup"),
    ("ERROR", "Failed to connect to ERP endpoint, retrying (attempt 2/3)"),
    ("INFO", "Agent health check passed: all probes OK"),
    ("INFO", "Provenance hash computed: sha256:a1b2c3..."),
    ("DEBUG", "Queue depth: 12, active workers: 4"),
    ("WARN", "Memory usage at 78%, approaching threshold"),
    ("INFO", "Batch complete: 1000 records in 4.2s"),
    ("ERROR", "Validation failed for input record #8837: missing facility_id"),
    ("CRITICAL", "Circuit breaker OPEN for redis-cache after 5 failures"),
    ("INFO", "Rollback checkpoint saved at version 2.0.3"),
]


def _generate_log_lines(
    agent_key: str,
    count: int,
    min_level: int,
    since_dt: datetime,
) -> list[dict]:
    """Generate simulated log lines for demonstration.

    In production, this would query Loki or the logging infrastructure via
    the greenlang.infrastructure.logging module.
    """
    lines: list[dict] = []
    now = datetime.now(timezone.utc)
    span = (now - since_dt).total_seconds()
    if span <= 0:
        span = 3600.0

    for i in range(count):
        level, msg = random.choice(_SAMPLE_MESSAGES)
        if _LEVEL_PRIORITY.get(level.lower(), 0) < min_level:
            continue
        ts = since_dt + timedelta(seconds=random.uniform(0, span))
        lines.append({
            "timestamp": ts.isoformat(),
            "level": level,
            "agent_key": agent_key,
            "message": msg,
        })

    lines.sort(key=lambda x: x["timestamp"])
    return lines


def _print_log_line(entry: dict) -> None:
    """Print a single log entry with color-coded level."""
    level = entry["level"]
    style = _LEVEL_STYLES.get(level, "white")
    ts = entry["timestamp"][:23]  # Trim to ms precision
    console.print(
        f"[dim]{ts}[/dim] [{style}]{level:<8}[/{style}] "
        f"[cyan]{entry['agent_key']}[/cyan] {entry['message']}"
    )


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

def logs(
    agent_key: str = typer.Option(
        ...,
        "--agent-key", "-k",
        help="Agent key to fetch logs for.",
    ),
    tail: int = typer.Option(
        100,
        "--tail", "-n",
        help="Number of recent log lines to display.",
    ),
    follow: bool = typer.Option(
        False,
        "--follow", "-f",
        help="Stream new log entries continuously.",
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since", "-s",
        help="Show logs since relative time (e.g. 1h, 30m, 2d).",
    ),
    level: str = typer.Option(
        "info",
        "--level", "-l",
        help="Minimum log level: debug, info, warn, error, critical.",
    ),
) -> None:
    """
    Fetch and display logs for a GreenLang agent.

    Retrieves log entries from the log aggregation system, optionally
    streaming new entries with --follow.

    Example:
        gl agent logs --agent-key carbon-calc --tail 50 --level error
    """
    level_lower = level.lower()
    if level_lower not in _LEVEL_PRIORITY:
        console.print(
            f"[red]Invalid level '{level}'. "
            f"Choose from: debug, info, warn, error, critical[/red]"
        )
        raise typer.Exit(1)

    min_level = _LEVEL_PRIORITY[level_lower]
    since_dt = _parse_since(since) if since else (datetime.now(timezone.utc) - timedelta(hours=1))

    console.print(
        f"[dim]Fetching logs for [bold]{agent_key}[/bold] "
        f"(tail={tail}, level>={level}, since={since_dt.isoformat()})[/dim]\n"
    )

    # Fetch historical lines
    entries = _generate_log_lines(agent_key, tail, min_level, since_dt)
    for entry in entries:
        _print_log_line(entry)

    if not follow:
        console.print(f"\n[dim]Showing {len(entries)} log entries.[/dim]")
        return

    # Follow / streaming mode
    console.print("\n[dim]Streaming new log entries (Ctrl+C to stop)...[/dim]\n")
    try:
        while True:
            time.sleep(random.uniform(0.5, 2.0))
            new_entries = _generate_log_lines(
                agent_key,
                count=random.randint(1, 3),
                min_level=min_level,
                since_dt=datetime.now(timezone.utc) - timedelta(seconds=5),
            )
            for entry in new_entries:
                _print_log_line(entry)
    except KeyboardInterrupt:
        console.print("\n[dim]Log stream stopped.[/dim]")
