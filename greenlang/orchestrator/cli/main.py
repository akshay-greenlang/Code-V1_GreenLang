# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator CLI Main Module
=======================================

Implements the 'gl' command-line tool for interacting with
the GreenLang Orchestrator.

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    click = None
    RICH_AVAILABLE = False

# Only define CLI if click is available
if click is None:
    def create_app():
        raise ImportError("CLI requires 'click' and 'rich'. Install with: pip install click rich")

    def main():
        print("CLI requires 'click' and 'rich'. Install with: pip install click rich")
        sys.exit(1)
else:
    from greenlang.orchestrator.pipeline_schema import (
        PipelineDefinition,
        load_pipeline_file,
    )

    # Console for rich output
    console = Console()

    # Default API endpoint
    DEFAULT_API_URL = os.environ.get("GL_API_URL", "http://localhost:8000")
    DEFAULT_API_KEY = os.environ.get("GL_API_KEY", "")


    class CLIContext:
        """Context object for CLI commands."""

        def __init__(self, api_url: str, api_key: str, output_format: str):
            self.api_url = api_url
            self.api_key = api_key
            self.output_format = output_format
            self._client = None

        async def get_client(self):
            """Get or create HTTP client."""
            if self._client is None:
                try:
                    import httpx
                    headers = {}
                    if self.api_key:
                        headers["X-API-Key"] = self.api_key
                    self._client = httpx.AsyncClient(
                        base_url=self.api_url,
                        headers=headers,
                        timeout=30.0,
                    )
                except ImportError:
                    raise ImportError("HTTP client requires 'httpx'. Install with: pip install httpx")
            return self._client

        async def close(self):
            """Close the HTTP client."""
            if self._client:
                await self._client.aclose()
                self._client = None


    @click.group()
    @click.option(
        "--api-url",
        envvar="GL_API_URL",
        default=DEFAULT_API_URL,
        help="Orchestrator API URL",
    )
    @click.option(
        "--api-key",
        envvar="GL_API_KEY",
        default=DEFAULT_API_KEY,
        help="API key for authentication",
    )
    @click.option(
        "--output", "-o",
        type=click.Choice(["table", "json", "yaml"]),
        default="table",
        help="Output format",
    )
    @click.version_option(version="2.0.0", prog_name="gl")
    @click.pass_context
    def cli(ctx, api_url: str, api_key: str, output: str):
        """
        GreenLang Orchestrator CLI

        Manage pipelines, runs, and agents from the command line.
        """
        ctx.ensure_object(dict)
        ctx.obj["ctx"] = CLIContext(api_url, api_key, output)


    # ==========================================================================
    # Run Commands
    # ==========================================================================

    @cli.command("run")
    @click.argument("pipeline_file", type=click.Path(exists=True))
    @click.option(
        "--tenant", "-t",
        required=True,
        help="Tenant ID for the run",
    )
    @click.option(
        "--param", "-p",
        multiple=True,
        help="Pipeline parameter (key=value format)",
    )
    @click.option(
        "--dry-run", is_flag=True,
        help="Validate without executing",
    )
    @click.option(
        "--wait", "-w", is_flag=True,
        help="Wait for run to complete",
    )
    @click.pass_context
    def run_pipeline(ctx, pipeline_file: str, tenant: str, param: tuple, dry_run: bool, wait: bool):
        """Submit a pipeline run.

        Example:
            gl run pipeline.yaml --tenant acme-corp -p reporting_year=2024
        """
        async def _run():
            cli_ctx = ctx.obj["ctx"]

            # Load and validate pipeline
            try:
                pipeline = load_pipeline_file(pipeline_file)
                console.print(f"[green]Loaded pipeline:[/green] {pipeline.metadata.name}")
            except Exception as e:
                console.print(f"[red]Error loading pipeline:[/red] {e}")
                return 1

            # Parse parameters
            params = {}
            for p in param:
                if "=" in p:
                    key, value = p.split("=", 1)
                    params[key] = value

            # Submit run
            try:
                client = await cli_ctx.get_client()
                response = await client.post(
                    "/api/v1/runs",
                    json={
                        "pipeline": pipeline.model_dump(),
                        "tenant_id": tenant,
                        "parameters": params,
                        "dry_run": dry_run,
                    }
                )
                response.raise_for_status()
                result = response.json()
            except Exception as e:
                console.print(f"[red]Error submitting run:[/red] {e}")
                await cli_ctx.close()
                return 1

            run_id = result.get("run_id")

            if dry_run:
                console.print(Panel(
                    f"[yellow]Dry run completed[/yellow]\n\n"
                    f"Plan ID: {result.get('plan', {}).get('plan_id', 'N/A')}\n"
                    f"Steps: {len(result.get('plan', {}).get('steps', []))}",
                    title="Validation Result"
                ))
            else:
                console.print(f"[green]Run submitted:[/green] {run_id}")

                if wait:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task(f"Waiting for run {run_id[:8]}...", total=None)

                        while True:
                            await asyncio.sleep(2)
                            response = await client.get(f"/api/v1/runs/{run_id}")
                            status_data = response.json()
                            status = status_data.get("status")

                            progress.update(task, description=f"Run {run_id[:8]} - {status}")

                            if status in ("success", "failed", "cancelled", "timeout"):
                                break

                        progress.stop()
                        _print_run_status(status_data, cli_ctx.output_format)

            await cli_ctx.close()
            return 0

        return asyncio.run(_run())


    @cli.command("status")
    @click.argument("run_id")
    @click.option("--watch", "-w", is_flag=True, help="Watch for updates")
    @click.pass_context
    def get_status(ctx, run_id: str, watch: bool):
        """Get status of a run.

        Example:
            gl status abc123-def456
        """
        async def _status():
            cli_ctx = ctx.obj["ctx"]

            try:
                client = await cli_ctx.get_client()

                if watch:
                    while True:
                        response = await client.get(f"/api/v1/runs/{run_id}")
                        response.raise_for_status()
                        data = response.json()

                        console.clear()
                        _print_run_status(data, cli_ctx.output_format)

                        if data.get("status") in ("success", "failed", "cancelled", "timeout"):
                            break

                        await asyncio.sleep(2)
                else:
                    response = await client.get(f"/api/v1/runs/{run_id}")
                    response.raise_for_status()
                    _print_run_status(response.json(), cli_ctx.output_format)

            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                return 1
            finally:
                await cli_ctx.close()

            return 0

        return asyncio.run(_status())


    @cli.command("logs")
    @click.argument("run_id")
    @click.option("--step", "-s", help="Filter by step ID")
    @click.option("--follow", "-f", is_flag=True, help="Follow log output")
    @click.option("--tail", "-n", default=100, help="Number of lines to show")
    @click.pass_context
    def get_logs(ctx, run_id: str, step: Optional[str], follow: bool, tail: int):
        """View logs for a run.

        Example:
            gl logs abc123 --follow
        """
        async def _logs():
            cli_ctx = ctx.obj["ctx"]

            try:
                client = await cli_ctx.get_client()

                params = {"tail": tail}
                if step:
                    params["step_id"] = step

                response = await client.get(f"/api/v1/runs/{run_id}/logs", params=params)
                response.raise_for_status()
                data = response.json()

                for log_entry in data.get("logs", []):
                    timestamp = log_entry.get("timestamp", "")
                    level = log_entry.get("level", "INFO")
                    message = log_entry.get("message", "")
                    step_id = log_entry.get("step_id", "")

                    color = {"ERROR": "red", "WARN": "yellow", "INFO": "white"}.get(level, "white")
                    console.print(f"[dim]{timestamp}[/dim] [{color}]{level:5}[/{color}] {step_id:12} {message}")

            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                return 1
            finally:
                await cli_ctx.close()

            return 0

        return asyncio.run(_logs())


    @cli.command("cancel")
    @click.argument("run_id")
    @click.option("--force", "-f", is_flag=True, help="Force cancellation")
    @click.confirmation_option(prompt="Are you sure you want to cancel this run?")
    @click.pass_context
    def cancel_run(ctx, run_id: str, force: bool):
        """Cancel a running pipeline.

        Example:
            gl cancel abc123 --force
        """
        async def _cancel():
            cli_ctx = ctx.obj["ctx"]

            try:
                client = await cli_ctx.get_client()
                response = await client.post(
                    f"/api/v1/runs/{run_id}/cancel",
                    json={"force": force}
                )
                response.raise_for_status()
                result = response.json()

                console.print(f"[green]Cancellation requested for run {run_id}[/green]")

            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                return 1
            finally:
                await cli_ctx.close()

            return 0

        return asyncio.run(_cancel())


    @cli.command("list")
    @click.option("--tenant", "-t", help="Filter by tenant")
    @click.option("--status", "-s", help="Filter by status")
    @click.option("--limit", "-n", default=20, help="Number of runs to show")
    @click.pass_context
    def list_runs(ctx, tenant: Optional[str], status: Optional[str], limit: int):
        """List recent pipeline runs.

        Example:
            gl list --tenant acme-corp --status running
        """
        async def _list():
            cli_ctx = ctx.obj["ctx"]

            try:
                client = await cli_ctx.get_client()

                params = {"limit": limit}
                if tenant:
                    params["tenant_id"] = tenant
                if status:
                    params["status"] = status

                response = await client.get("/api/v1/runs", params=params)
                response.raise_for_status()
                data = response.json()

                if cli_ctx.output_format == "json":
                    console.print_json(json.dumps(data))
                else:
                    table = Table(title="Recent Runs")
                    table.add_column("Run ID", style="cyan")
                    table.add_column("Pipeline")
                    table.add_column("Status")
                    table.add_column("Started")
                    table.add_column("Duration")

                    for run in data.get("runs", []):
                        status_color = {
                            "success": "green",
                            "failed": "red",
                            "running": "yellow",
                            "pending": "blue",
                        }.get(run.get("status", ""), "white")

                        table.add_row(
                            run.get("run_id", "")[:12],
                            run.get("pipeline_id", "")[:20],
                            f"[{status_color}]{run.get('status', '')}[/{status_color}]",
                            run.get("started_at", "")[:19],
                            f"{run.get('duration_ms', 0):.0f}ms",
                        )

                    console.print(table)

            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                return 1
            finally:
                await cli_ctx.close()

            return 0

        return asyncio.run(_list())


    # ==========================================================================
    # Pipeline Commands
    # ==========================================================================

    @cli.command("validate")
    @click.argument("pipeline_file", type=click.Path(exists=True))
    @click.pass_context
    def validate_pipeline(ctx, pipeline_file: str):
        """Validate a pipeline YAML file.

        Example:
            gl validate pipeline.yaml
        """
        try:
            pipeline = load_pipeline_file(pipeline_file)

            console.print(Panel(
                f"[green]Pipeline is valid![/green]\n\n"
                f"Name: {pipeline.metadata.name}\n"
                f"Version: {pipeline.metadata.version}\n"
                f"Steps: {len(pipeline.spec.steps)}\n"
                f"Parameters: {len(pipeline.spec.parameters or [])}",
                title="Validation Result"
            ))

            # Show step graph
            table = Table(title="Step Dependencies")
            table.add_column("Step")
            table.add_column("Agent")
            table.add_column("Depends On")

            for step in pipeline.spec.steps:
                table.add_row(
                    step.id,
                    step.agent,
                    ", ".join(step.depends_on) if step.depends_on else "-",
                )

            console.print(table)
            return 0

        except Exception as e:
            console.print(f"[red]Validation failed:[/red] {e}")
            return 1


    # ==========================================================================
    # Agent Commands
    # ==========================================================================

    @cli.command("agents")
    @click.option("--layer", "-l", help="Filter by layer")
    @click.option("--capability", "-c", help="Filter by capability")
    @click.pass_context
    def list_agents(ctx, layer: Optional[str], capability: Optional[str]):
        """List registered agents.

        Example:
            gl agents --layer mrv
        """
        async def _agents():
            cli_ctx = ctx.obj["ctx"]

            try:
                client = await cli_ctx.get_client()

                params = {}
                if layer:
                    params["layer"] = layer
                if capability:
                    params["capability"] = capability

                response = await client.get("/api/v1/agents", params=params)
                response.raise_for_status()
                data = response.json()

                if cli_ctx.output_format == "json":
                    console.print_json(json.dumps(data))
                else:
                    table = Table(title="Registered Agents")
                    table.add_column("Agent ID", style="cyan")
                    table.add_column("Name")
                    table.add_column("Version")
                    table.add_column("Layer")
                    table.add_column("Mode")
                    table.add_column("Health")

                    for agent in data.get("agents", []):
                        health_color = {
                            "healthy": "green",
                            "degraded": "yellow",
                            "unhealthy": "red",
                        }.get(agent.get("health_status", ""), "dim")

                        table.add_row(
                            agent.get("agent_id", ""),
                            agent.get("name", "")[:30],
                            agent.get("version", ""),
                            agent.get("layer", ""),
                            agent.get("execution_mode", ""),
                            f"[{health_color}]{agent.get('health_status', 'unknown')}[/{health_color}]",
                        )

                    console.print(table)

            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                return 1
            finally:
                await cli_ctx.close()

            return 0

        return asyncio.run(_agents())


    # ==========================================================================
    # Audit Commands
    # ==========================================================================

    @cli.command("audit")
    @click.argument("run_id")
    @click.option("--verify", is_flag=True, help="Verify hash chain integrity")
    @click.pass_context
    def get_audit(ctx, run_id: str, verify: bool):
        """View audit trail for a run.

        Example:
            gl audit abc123 --verify
        """
        async def _audit():
            cli_ctx = ctx.obj["ctx"]

            try:
                client = await cli_ctx.get_client()

                params = {"verify": verify}
                response = await client.get(f"/api/v1/runs/{run_id}/audit", params=params)
                response.raise_for_status()
                data = response.json()

                if verify:
                    is_valid = data.get("chain_valid", False)
                    if is_valid:
                        console.print("[green]Hash chain integrity verified![/green]")
                    else:
                        console.print("[red]Hash chain integrity FAILED![/red]")
                        console.print(f"Error: {data.get('verification_error', 'Unknown')}")
                        return 1

                table = Table(title=f"Audit Trail for {run_id[:12]}")
                table.add_column("Event ID", style="dim")
                table.add_column("Type")
                table.add_column("Timestamp")
                table.add_column("Hash", style="dim")

                for event in data.get("events", []):
                    table.add_row(
                        event.get("event_id", "")[:8],
                        event.get("event_type", ""),
                        event.get("timestamp", "")[:19],
                        event.get("event_hash", "")[:12] + "...",
                    )

                console.print(table)

            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                return 1
            finally:
                await cli_ctx.close()

            return 0

        return asyncio.run(_audit())


    # ==========================================================================
    # Helper Functions
    # ==========================================================================

    def _print_run_status(data: Dict[str, Any], output_format: str):
        """Print run status in the specified format."""
        if output_format == "json":
            console.print_json(json.dumps(data))
            return

        status = data.get("status", "unknown")
        status_color = {
            "success": "green",
            "failed": "red",
            "running": "yellow",
            "pending": "blue",
            "cancelled": "dim",
            "timeout": "red",
        }.get(status, "white")

        panel_content = f"""
[bold]Run ID:[/bold] {data.get('run_id', 'N/A')}
[bold]Pipeline:[/bold] {data.get('pipeline_id', 'N/A')}
[bold]Status:[/bold] [{status_color}]{status}[/{status_color}]

[bold]Started:[/bold] {data.get('started_at', 'N/A')}
[bold]Duration:[/bold] {data.get('duration_ms', 0):.0f}ms

[bold]Steps:[/bold]
  Total: {data.get('agents_total', 0)}
  Succeeded: [green]{data.get('agents_succeeded', 0)}[/green]
  Failed: [red]{data.get('agents_failed', 0)}[/red]
  Skipped: {data.get('agents_skipped', 0)}
"""

        if data.get("errors"):
            panel_content += "\n[bold red]Errors:[/bold red]\n"
            for err in data["errors"][:3]:
                panel_content += f"  - {err.get('agent_id', 'unknown')}: {err.get('error', 'Unknown error')[:50]}\n"

        console.print(Panel(panel_content, title="Run Status"))


    def create_app():
        """Create the CLI application."""
        return cli


    def main():
        """Entry point for the CLI."""
        cli()


if __name__ == "__main__":
    main()
