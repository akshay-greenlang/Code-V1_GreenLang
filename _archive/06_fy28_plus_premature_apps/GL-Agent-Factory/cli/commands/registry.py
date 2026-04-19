"""
Registry Management Commands

Commands for interacting with the agent registry.
"""

import typer
from typing import Optional
from pathlib import Path

from cli.utils.console import (
    console,
    print_error,
    print_success,
    print_info,
    print_warning,
    create_agent_table,
)
from cli.utils.config import load_config


# Create registry command group
app = typer.Typer(
    help="Registry management commands",
    no_args_is_help=True,
)


@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help="Search query for agents",
    ),
    filter_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by agent type",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of results",
    ),
):
    """
    Search for agents in the registry.

    Example:
        gl registry search "furnace compliance" --type regulatory
    """
    try:
        console.print(f"\n[bold cyan]Searching registry for:[/bold cyan] {query}\n")

        # Placeholder results
        agents = [
            {
                "id": "nfpa86-furnace-agent",
                "name": "NFPA 86 Furnace Compliance Agent",
                "version": "1.2.0",
                "type": "regulatory",
                "status": "active",
                "updated": "2024-12-01",
            },
            {
                "id": "process-heat-agent",
                "name": "Process Heat Optimization Agent",
                "version": "1.0.0",
                "type": "optimization",
                "status": "active",
                "updated": "2024-11-28",
            },
        ]

        if agents:
            table = create_agent_table(agents)
            console.print(table)
            console.print(f"\n[dim]Found {len(agents)} agents[/dim]\n")
        else:
            print_info("No agents found matching your query")

    except Exception as e:
        print_error(f"Search failed: {str(e)}")
        raise typer.Exit(1)


@app.command()
def pull(
    agent_ref: str = typer.Argument(
        ...,
        help="Agent reference (id:version)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for pulled agent",
    ),
):
    """
    Pull an agent from the registry.

    Example:
        gl registry pull nfpa86-furnace-agent:1.2.0
    """
    try:
        console.print(f"\n[bold cyan]Pulling agent:[/bold cyan] {agent_ref}\n")

        # Parse reference
        if ":" in agent_ref:
            agent_id, version = agent_ref.split(":")
        else:
            agent_id = agent_ref
            version = "latest"

        if output_dir is None:
            output_dir = Path.cwd() / "agents" / agent_id

        with console.status(f"[bold green]Downloading {agent_id}:{version}...", spinner="dots"):
            # Simulate download
            import time
            time.sleep(1)

        print_success(f"Agent pulled successfully: {output_dir}")
        console.print(f"\nAgent location: [cyan]{output_dir}[/cyan]\n")

    except Exception as e:
        print_error(f"Failed to pull agent: {str(e)}")
        raise typer.Exit(1)


@app.command()
def push(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Version tag to push",
    ),
):
    """
    Push an agent to the registry.

    Example:
        gl registry push agents/nfpa86 --tag v1.2.0
    """
    try:
        console.print(f"\n[bold cyan]Pushing agent:[/bold cyan] {agent_path}\n")

        with console.status("[bold green]Uploading to registry...", spinner="dots"):
            # Simulate upload
            import time
            time.sleep(1)

        print_success("Agent pushed successfully to registry")

    except Exception as e:
        print_error(f"Failed to push agent: {str(e)}")
        raise typer.Exit(1)


@app.command()
def login(
    registry_url: Optional[str] = typer.Option(
        None,
        "--registry",
        "-r",
        help="Registry URL",
    ),
):
    """
    Login to the agent registry.

    Example:
        gl registry login --registry https://registry.greenlang.io
    """
    try:
        config = load_config()
        url = registry_url or config.get("registry", {}).get("url", "https://registry.greenlang.io")

        console.print(f"\n[bold cyan]Logging in to registry:[/bold cyan] {url}\n")

        username = console.input("[bold]Username:[/bold] ")
        password = console.input("[bold]Password:[/bold] ", password=True)

        with console.status("[bold green]Authenticating...", spinner="dots"):
            # Simulate authentication
            import time
            time.sleep(1)

        print_success("Successfully logged in to registry")

    except Exception as e:
        print_error(f"Login failed: {str(e)}")
        raise typer.Exit(1)


@app.command()
def logout():
    """
    Logout from the agent registry.

    Example:
        gl registry logout
    """
    try:
        print_success("Successfully logged out from registry")

    except Exception as e:
        print_error(f"Logout failed: {str(e)}")
        raise typer.Exit(1)
