"""
GreenLang CLI - Registry Commands
Commands for managing agents in the registry: publish, list, info
"""

import sys
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from ..registry.client import RegistryClient, RegistryError, AgentNotFoundError, AgentAlreadyExistsError

# Create CLI app
app = typer.Typer(name="agent", help="Agent registry operations")
console = Console()


# ============================================================================
# Registry Configuration
# ============================================================================

def get_registry_client(
    registry_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> RegistryClient:
    """
    Get configured registry client

    Args:
        registry_url: Override registry URL
        api_key: Override API key

    Returns:
        RegistryClient instance
    """
    # Try environment variables first
    import os
    base_url = registry_url or os.getenv("GL_REGISTRY_URL", "http://localhost:8000")
    api_key = api_key or os.getenv("GL_REGISTRY_API_KEY")

    return RegistryClient(base_url=base_url, api_key=api_key)


# ============================================================================
# CLI Commands
# ============================================================================

@app.command("publish")
def publish_agent(
    pack_path: Path = typer.Argument(..., help="Path to .glpack file or agent directory"),
    version: str = typer.Option(..., "--version", "-v", help="Semantic version (e.g., 1.0.0)"),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Agent namespace"),
    author: Optional[str] = typer.Option(None, "--author", "-a", help="Author name"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Agent description"),
    registry_url: Optional[str] = typer.Option(None, "--registry", help="Registry API URL"),
    force: bool = typer.Option(False, "--force", "-f", help="Force publish even if version exists")
):
    """
    Publish an agent to the registry

    Example:
        gl agent publish thermosync.glpack --version 1.0.0 --namespace greenlang
        gl agent publish ./agents/thermosync/ --version 1.0.1 --author "GreenLang Team"
    """
    console.print(f"\n[bold cyan]Publishing Agent to Registry[/bold cyan]")
    console.print(f"Pack: {pack_path}")
    console.print(f"Version: {version}\n")

    if not pack_path.exists():
        console.print(f"[bold red]Error:[/bold red] Pack not found: {pack_path}")
        raise typer.Exit(code=1)

    try:
        # Initialize registry client
        client = get_registry_client(registry_url=registry_url)

        # Check API health
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Connecting to registry...", total=None)

            try:
                health = client.health_check()
                if health.get("status") != "healthy":
                    console.print("[bold red]Error:[/bold red] Registry is not healthy")
                    raise typer.Exit(code=1)
            except RegistryError as e:
                console.print(f"[bold red]Error:[/bold red] Cannot connect to registry: {e}")
                raise typer.Exit(code=1)

            progress.update(task, description="Connected to registry")

        # Load pack manifest if available
        manifest_path = pack_path / "manifest.yml" if pack_path.is_dir() else None
        agent_name = None
        agent_description = description
        agent_author = author

        if manifest_path and manifest_path.exists():
            import yaml
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
                agent_name = manifest.get("name")
                agent_description = agent_description or manifest.get("description")
                agent_author = agent_author or manifest.get("author")

        if not agent_name:
            agent_name = pack_path.stem

        console.print(f"\n[bold]Agent:[/bold] {namespace}/{agent_name}")

        # Check if agent exists, if not register it
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Checking agent registration...", total=None)

            try:
                # Try to find existing agent
                agents = client.list_agents(namespace=namespace, search=agent_name)
                existing_agent = None

                for agent in agents.get("agents", []):
                    if agent["name"] == agent_name and agent["namespace"] == namespace:
                        existing_agent = agent
                        break

                if existing_agent:
                    agent_id = existing_agent["id"]
                    console.print(f"[green]✓[/green] Agent already registered (ID: {agent_id})")
                else:
                    # Register new agent
                    progress.update(task, description="Registering new agent...")
                    agent = client.register(
                        name=agent_name,
                        namespace=namespace,
                        description=agent_description,
                        author=agent_author
                    )
                    agent_id = agent["id"]
                    console.print(f"[green]✓[/green] Registered new agent (ID: {agent_id})")

            except AgentAlreadyExistsError:
                console.print(f"[bold red]Error:[/bold red] Agent already exists")
                raise typer.Exit(code=1)

        # Publish version
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Publishing version {version}...", total=None)

            try:
                # Get pack path (use directory if not a .glpack file)
                pack_file_path = str(pack_path)

                version_data = client.publish_version(
                    agent_id=agent_id,
                    version=version,
                    pack_path=pack_file_path,
                    published_by=agent_author
                )

                console.print(f"[green]✓[/green] Published version {version}")

            except RegistryError as e:
                if "already exists" in str(e).lower():
                    if force:
                        console.print(f"[yellow]Warning:[/yellow] Version exists, but --force not implemented for updates")
                    else:
                        console.print(f"[bold red]Error:[/bold red] Version {version} already exists. Use --force to overwrite.")
                    raise typer.Exit(code=1)
                else:
                    console.print(f"[bold red]Error:[/bold red] {e}")
                    raise typer.Exit(code=1)

        # Display success
        console.print("\n[bold green]✓ Agent Published Successfully![/bold green]\n")

        info_table = Table(show_header=False, box=None)
        info_table.add_row("[bold]Agent ID:[/bold]", agent_id)
        info_table.add_row("[bold]Name:[/bold]", f"{namespace}/{agent_name}")
        info_table.add_row("[bold]Version:[/bold]", version)
        info_table.add_row("[bold]Pack:[/bold]", str(pack_path))

        console.print(info_table)
        console.print(f"\n[dim]View agent: gl agent info {agent_id}[/dim]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted by user[/yellow]")
        raise typer.Exit(code=130)


@app.command("list")
def list_agents(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Filter by namespace"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search in name/description"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    page_size: int = typer.Option(20, "--page-size", help="Results per page"),
    registry_url: Optional[str] = typer.Option(None, "--registry", help="Registry API URL"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """
    List agents in the registry

    Example:
        gl agent list
        gl agent list --namespace greenlang
        gl agent list --search "temperature"
        gl agent list --json
    """
    try:
        # Initialize registry client
        client = get_registry_client(registry_url=registry_url)

        # Fetch agents
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Fetching agents...", total=None)

            response = client.list_agents(
                namespace=namespace,
                status=status,
                search=search,
                page=page,
                page_size=page_size
            )

        agents = response.get("agents", [])
        total = response.get("total", 0)

        if json_output:
            console.print(json.dumps(response, indent=2))
            return

        # Display results
        if not agents:
            console.print("\n[yellow]No agents found[/yellow]\n")
            return

        console.print(f"\n[bold cyan]Agents in Registry[/bold cyan] (Total: {total})\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Name", style="cyan", width=30)
        table.add_column("Namespace", style="blue", width=15)
        table.add_column("Description", width=40)
        table.add_column("Status", justify="center", width=10)
        table.add_column("ID", style="dim", width=36)

        for agent in agents:
            name = agent.get("name", "")
            namespace = agent.get("namespace", "")
            description = agent.get("description", "")[:40] + "..." if agent.get("description", "") else ""
            status = agent.get("status", "")
            agent_id = agent.get("id", "")

            # Color status
            status_colored = {
                "active": f"[green]{status}[/green]",
                "deprecated": f"[yellow]{status}[/yellow]",
                "archived": f"[red]{status}[/red]"
            }.get(status, status)

            table.add_row(name, namespace, description, status_colored, agent_id)

        console.print(table)
        console.print(f"\n[dim]Page {page} of {(total + page_size - 1) // page_size}[/dim]")
        console.print(f"[dim]View details: gl agent info <agent_id>[/dim]\n")

    except RegistryError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted by user[/yellow]")
        raise typer.Exit(code=130)


@app.command("info")
def agent_info(
    agent_id: str = typer.Argument(..., help="Agent ID or namespace/name"),
    registry_url: Optional[str] = typer.Option(None, "--registry", help="Registry API URL"),
    show_versions: bool = typer.Option(True, "--versions/--no-versions", help="Show version history"),
    show_certs: bool = typer.Option(True, "--certs/--no-certs", help="Show certifications"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """
    Show detailed agent information

    Example:
        gl agent info 12345678-1234-1234-1234-123456789abc
        gl agent info greenlang/thermosync
        gl agent info 12345678-1234-1234-1234-123456789abc --json
    """
    try:
        # Initialize registry client
        client = get_registry_client(registry_url=registry_url)

        # Handle namespace/name format
        if "/" in agent_id:
            namespace, name = agent_id.split("/", 1)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Searching for agent...", total=None)
                response = client.list_agents(namespace=namespace, search=name)
                agents = response.get("agents", [])

                found = None
                for agent in agents:
                    if agent["name"] == name and agent["namespace"] == namespace:
                        found = agent
                        break

                if not found:
                    console.print(f"[bold red]Error:[/bold red] Agent not found: {agent_id}")
                    raise typer.Exit(code=1)

                agent_id = found["id"]

        # Fetch agent details
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Fetching agent details...", total=None)

            agent = client.get(agent_id)
            versions = client.list_versions(agent_id) if show_versions else []
            certifications = client.list_certifications(agent_id) if show_certs else []

        if json_output:
            output = {
                "agent": agent,
                "versions": versions,
                "certifications": certifications
            }
            console.print(json.dumps(output, indent=2, default=str))
            return

        # Display agent info
        console.print(f"\n[bold cyan]Agent Details[/bold cyan]\n")

        # Basic info panel
        info_table = Table(show_header=False, box=None)
        info_table.add_row("[bold]ID:[/bold]", agent["id"])
        info_table.add_row("[bold]Name:[/bold]", f"{agent['namespace']}/{agent['name']}")
        info_table.add_row("[bold]Status:[/bold]", agent["status"])
        if agent.get("description"):
            info_table.add_row("[bold]Description:[/bold]", agent["description"])
        if agent.get("author"):
            info_table.add_row("[bold]Author:[/bold]", agent["author"])
        if agent.get("repository_url"):
            info_table.add_row("[bold]Repository:[/bold]", agent["repository_url"])
        info_table.add_row("[bold]Created:[/bold]", str(agent["created_at"]))
        info_table.add_row("[bold]Updated:[/bold]", str(agent["updated_at"]))

        console.print(Panel(info_table, title="Agent Information", border_style="cyan"))

        # Versions
        if show_versions and versions:
            console.print(f"\n[bold cyan]Version History[/bold cyan] ({len(versions)} versions)\n")

            version_table = Table(show_header=True, header_style="bold magenta")
            version_table.add_column("Version", style="cyan", width=15)
            version_table.add_column("Status", justify="center", width=12)
            version_table.add_column("Published", width=20)
            version_table.add_column("Size", justify="right", width=12)
            version_table.add_column("Published By", width=20)

            for v in versions[:10]:  # Show last 10 versions
                version = v.get("version", "")
                status = v.get("status", "")
                published = str(v.get("published_at", ""))[:19]
                size = f"{v.get('size_bytes', 0) / 1024:.1f} KB" if v.get("size_bytes") else "N/A"
                publisher = v.get("published_by", "Unknown")

                status_colored = {
                    "published": f"[green]{status}[/green]",
                    "yanked": f"[red]{status}[/red]",
                    "deprecated": f"[yellow]{status}[/yellow]"
                }.get(status, status)

                version_table.add_row(version, status_colored, published, size, publisher)

            console.print(version_table)

        # Certifications
        if show_certs and certifications:
            console.print(f"\n[bold cyan]Certifications[/bold cyan] ({len(certifications)} total)\n")

            cert_table = Table(show_header=True, header_style="bold magenta")
            cert_table.add_column("Dimension", style="cyan", width=15)
            cert_table.add_column("Version", width=12)
            cert_table.add_column("Status", justify="center", width=10)
            cert_table.add_column("Score", justify="right", width=8)
            cert_table.add_column("Certified By", width=15)
            cert_table.add_column("Date", width=20)

            for cert in certifications[:10]:  # Show last 10 certifications
                dimension = cert.get("dimension", "")
                version = cert.get("version", "")
                status = cert.get("status", "")
                score = f"{cert.get('score', 0):.1f}" if cert.get("score") is not None else "N/A"
                certified_by = cert.get("certified_by", "")
                date = str(cert.get("certification_date", ""))[:19]

                status_colored = {
                    "passed": f"[green]{status}[/green]",
                    "failed": f"[red]{status}[/red]",
                    "pending": f"[yellow]{status}[/yellow]"
                }.get(status, status)

                cert_table.add_row(dimension, version, status_colored, score, certified_by, date)

            console.print(cert_table)

        console.print()

    except AgentNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Agent not found: {agent_id}")
        raise typer.Exit(code=1)
    except RegistryError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted by user[/yellow]")
        raise typer.Exit(code=130)


@app.command("certify")
def certify_agent(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    version: str = typer.Option(..., "--version", "-v", help="Version to certify"),
    dimension: str = typer.Option(..., "--dimension", "-d", help="Certification dimension"),
    status: str = typer.Option("passed", "--status", "-s", help="Certification status (passed/failed/pending)"),
    score: Optional[float] = typer.Option(None, "--score", help="Certification score (0-100)"),
    certified_by: str = typer.Option("GL-CERT", "--certified-by", help="Certifying authority"),
    registry_url: Optional[str] = typer.Option(None, "--registry", help="Registry API URL")
):
    """
    Submit certification for an agent version

    Example:
        gl agent certify <agent_id> --version 1.0.0 --dimension security --status passed --score 95.0
        gl agent certify <agent_id> -v 1.0.0 -d performance -s passed --score 88.5
    """
    console.print(f"\n[bold cyan]Submitting Certification[/bold cyan]")
    console.print(f"Agent: {agent_id}")
    console.print(f"Version: {version}")
    console.print(f"Dimension: {dimension}")
    console.print(f"Status: {status}\n")

    try:
        # Initialize registry client
        client = get_registry_client(registry_url=registry_url)

        # Submit certification
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Submitting certification...", total=None)

            cert = client.certify(
                agent_id=agent_id,
                version=version,
                dimension=dimension,
                status=status,
                score=score,
                certified_by=certified_by
            )

        console.print(f"[bold green]✓ Certification Submitted![/bold green]\n")

        cert_table = Table(show_header=False, box=None)
        cert_table.add_row("[bold]Certification ID:[/bold]", cert["id"])
        cert_table.add_row("[bold]Dimension:[/bold]", cert["dimension"])
        cert_table.add_row("[bold]Status:[/bold]", cert["status"])
        if cert.get("score") is not None:
            cert_table.add_row("[bold]Score:[/bold]", f"{cert['score']:.1f}/100")
        cert_table.add_row("[bold]Certified By:[/bold]", cert["certified_by"])
        cert_table.add_row("[bold]Date:[/bold]", str(cert["certification_date"]))

        console.print(cert_table)
        console.print()

    except RegistryError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted by user[/yellow]")
        raise typer.Exit(code=130)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    app()
