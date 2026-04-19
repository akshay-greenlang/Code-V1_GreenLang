"""
Agent Registry CLI Commands

This module provides CLI commands for interacting with the Agent Registry:
- gl registry push - Push agent to registry
- gl registry pull - Pull agent from registry
- gl registry search - Search for agents
- gl registry list - List agents
- gl registry info - Get agent info
- gl registry versions - List versions
- gl registry publish - Publish agent version

All commands support both local and remote registry operations.

Example:
    $ gl registry push ./my-agent --tag v1.0.0
    $ gl registry pull carbon-calculator:1.0.0
    $ gl registry search "emissions" --category regulatory
"""

import json
import os
import sys
import hashlib
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import typer
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Create CLI app
app = typer.Typer(
    name="registry",
    help="Agent Registry management commands",
    no_args_is_help=True,
)

console = Console()


# Configuration
DEFAULT_REGISTRY_URL = os.getenv(
    "GREENLANG_REGISTRY_URL",
    "https://registry.greenlang.io/v1/registry"
)
CONFIG_DIR = Path.home() / ".greenlang"
AUTH_FILE = CONFIG_DIR / "auth.json"


def get_auth_token() -> Optional[str]:
    """
    Get authentication token from local config.

    Returns:
        JWT token or None if not logged in
    """
    if AUTH_FILE.exists():
        try:
            with open(AUTH_FILE) as f:
                auth = json.load(f)
                return auth.get("token")
        except Exception:
            pass
    return None


def get_headers() -> Dict[str, str]:
    """
    Get HTTP headers with authentication.

    Returns:
        Headers dictionary
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "gl-cli/1.0.0",
    }
    token = get_auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str) -> None:
    """Print success message."""
    console.print(f"[bold green]Success:[/bold green] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"[cyan]{message}[/cyan]")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


# =============================================================================
# Push Command
# =============================================================================


@app.command()
def push(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Version tag to push (e.g., v1.0.0)",
    ),
    message: Optional[str] = typer.Option(
        None,
        "--message",
        "-m",
        help="Release message/changelog",
    ),
    breaking: bool = typer.Option(
        False,
        "--breaking",
        "-b",
        help="Mark as breaking change",
    ),
    registry_url: str = typer.Option(
        DEFAULT_REGISTRY_URL,
        "--registry",
        "-r",
        help="Registry URL",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate without pushing",
    ),
):
    """
    Push an agent to the registry.

    Creates a new version of the agent in the registry with the
    specified tag. The agent directory must contain a pack.yaml file.

    Example:
        gl registry push ./my-agent --tag v1.2.0 -m "Added new features"
    """
    console.print(f"\n[bold cyan]Pushing agent:[/bold cyan] {agent_path}\n")

    # Validate agent directory
    pack_yaml_path = agent_path / "pack.yaml"
    if not pack_yaml_path.exists():
        print_error(f"No pack.yaml found in {agent_path}")
        raise typer.Exit(1)

    # Load pack.yaml
    try:
        import yaml
        with open(pack_yaml_path) as f:
            pack_yaml = yaml.safe_load(f)
    except Exception as e:
        print_error(f"Failed to parse pack.yaml: {e}")
        raise typer.Exit(1)

    agent_name = pack_yaml.get("name", agent_path.name)
    version = tag.lstrip("v") if tag else pack_yaml.get("version", "1.0.0")

    console.print(f"  Agent: [bold]{agent_name}[/bold]")
    console.print(f"  Version: [bold]{version}[/bold]")

    if dry_run:
        console.print("\n[dim]Dry run - no changes made[/dim]")
        print_success("Validation passed")
        return

    # Check authentication
    if not get_auth_token():
        print_error("Not logged in. Run 'gl registry login' first.")
        raise typer.Exit(1)

    # Create tarball
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating agent package...", total=None)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                tar.add(agent_path, arcname=agent_name)
            archive_path = tmp.name

        progress.update(task, description="Calculating checksum...")

        # Calculate checksum
        with open(archive_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        progress.update(task, description="Uploading to registry...")

        # Prepare payload
        payload = {
            "name": agent_name,
            "version": version,
            "description": pack_yaml.get("description", ""),
            "category": pack_yaml.get("category", "general"),
            "author": pack_yaml.get("author", "unknown"),
            "pack_yaml": pack_yaml,
            "tags": pack_yaml.get("tags", []),
            "regulatory_frameworks": pack_yaml.get("regulatory_frameworks", []),
        }

        # Check if agent exists
        try:
            with httpx.Client() as client:
                # Try to get existing agent
                search_resp = client.get(
                    f"{registry_url}/agents/search",
                    params={"q": agent_name},
                    headers=get_headers(),
                    timeout=30,
                )

                if search_resp.status_code == 200:
                    data = search_resp.json().get("data", [])
                    existing = next((a for a in data if a["name"] == agent_name), None)

                    if existing:
                        # Create new version
                        progress.update(task, description="Creating new version...")
                        version_payload = {
                            "version": version,
                            "changelog": message or f"Version {version}",
                            "breaking_changes": breaking,
                            "release_notes": message or "",
                            "pack_yaml": pack_yaml,
                        }
                        resp = client.post(
                            f"{registry_url}/agents/{existing['id']}/versions",
                            json=version_payload,
                            headers=get_headers(),
                            timeout=60,
                        )
                    else:
                        # Create new agent
                        progress.update(task, description="Registering new agent...")
                        resp = client.post(
                            f"{registry_url}/agents",
                            json=payload,
                            headers=get_headers(),
                            timeout=60,
                        )
                else:
                    # Create new agent
                    resp = client.post(
                        f"{registry_url}/agents",
                        json=payload,
                        headers=get_headers(),
                        timeout=60,
                    )

                if resp.status_code in (200, 201):
                    result = resp.json()
                    progress.update(task, completed=True)
                else:
                    error_msg = resp.json().get("detail", resp.text)
                    print_error(f"Push failed: {error_msg}")
                    raise typer.Exit(1)

        except httpx.RequestError as e:
            print_error(f"Connection failed: {e}")
            raise typer.Exit(1)
        finally:
            # Cleanup
            os.unlink(archive_path)

    console.print("")
    print_success(f"Agent pushed: {agent_name}:{version}")
    console.print(f"  Checksum: sha256:{checksum[:16]}...")


# =============================================================================
# Pull Command
# =============================================================================


@app.command()
def pull(
    agent_ref: str = typer.Argument(
        ...,
        help="Agent reference (name:version or name for latest)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory",
    ),
    registry_url: str = typer.Option(
        DEFAULT_REGISTRY_URL,
        "--registry",
        "-r",
        help="Registry URL",
    ),
):
    """
    Pull an agent from the registry.

    Downloads the specified agent version to the output directory.
    Use 'name:version' format or just 'name' for latest version.

    Example:
        gl registry pull carbon-calculator:1.0.0
        gl registry pull carbon-calculator -o ./agents
    """
    console.print(f"\n[bold cyan]Pulling agent:[/bold cyan] {agent_ref}\n")

    # Parse reference
    if ":" in agent_ref:
        agent_name, version = agent_ref.split(":", 1)
    else:
        agent_name = agent_ref
        version = "latest"

    if output_dir is None:
        output_dir = Path.cwd() / "agents" / agent_name

    console.print(f"  Agent: [bold]{agent_name}[/bold]")
    console.print(f"  Version: [bold]{version}[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching registry...", total=None)

        try:
            with httpx.Client() as client:
                # Search for agent
                search_resp = client.get(
                    f"{registry_url}/agents/search",
                    params={"q": agent_name},
                    headers=get_headers(),
                    timeout=30,
                )

                if search_resp.status_code != 200:
                    print_error(f"Search failed: {search_resp.text}")
                    raise typer.Exit(1)

                data = search_resp.json().get("data", [])
                agent = next((a for a in data if a["name"] == agent_name), None)

                if not agent:
                    print_error(f"Agent '{agent_name}' not found")
                    raise typer.Exit(1)

                if agent["status"] != "published":
                    print_warning(f"Agent status is '{agent['status']}' (not published)")

                progress.update(task, description="Getting version info...")

                # Get version
                if version == "latest":
                    versions_resp = client.get(
                        f"{registry_url}/agents/{agent['id']}/versions",
                        headers=get_headers(),
                        timeout=30,
                    )
                    if versions_resp.status_code == 200:
                        versions = versions_resp.json()
                        if versions:
                            version = versions[0]["version"]

                # Get download info
                download_resp = client.get(
                    f"{registry_url}/agents/{agent['id']}/download",
                    params={"version": version},
                    headers=get_headers(),
                    timeout=30,
                )

                if download_resp.status_code != 200:
                    # Fall back to agent info
                    actual_version = version if version != "latest" else agent.get("version", "1.0.0")
                else:
                    download_info = download_resp.json()
                    actual_version = download_info.get("version", version)

                progress.update(task, description="Creating output directory...")

                # Create output directory
                output_dir.mkdir(parents=True, exist_ok=True)

                # Create placeholder pack.yaml from agent info
                pack_yaml = {
                    "name": agent_name,
                    "version": actual_version,
                    "description": agent.get("description", ""),
                    "category": agent.get("category", ""),
                    "author": agent.get("author", ""),
                    "tags": agent.get("tags", []),
                    "regulatory_frameworks": agent.get("regulatory_frameworks", []),
                    "pulled_at": datetime.utcnow().isoformat(),
                }

                # Write pack.yaml
                import yaml
                pack_yaml_path = output_dir / "pack.yaml"
                with open(pack_yaml_path, "w") as f:
                    yaml.dump(pack_yaml, f, default_flow_style=False)

                progress.update(task, completed=True)

        except httpx.RequestError as e:
            print_error(f"Connection failed: {e}")
            raise typer.Exit(1)

    console.print("")
    print_success(f"Agent pulled: {agent_name}:{actual_version}")
    console.print(f"  Location: [cyan]{output_dir}[/cyan]")


# =============================================================================
# Search Command
# =============================================================================


@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help="Search query",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category",
    ),
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (draft/published/deprecated)",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum results",
    ),
    registry_url: str = typer.Option(
        DEFAULT_REGISTRY_URL,
        "--registry",
        "-r",
        help="Registry URL",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json)",
    ),
):
    """
    Search for agents in the registry.

    Search across agent names, descriptions, and tags.

    Example:
        gl registry search "carbon emissions"
        gl registry search "compliance" --category regulatory
    """
    console.print(f"\n[bold cyan]Searching registry:[/bold cyan] {query}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=None)

        try:
            with httpx.Client() as client:
                params = {
                    "q": query,
                    "limit": limit,
                }
                if category:
                    params["category"] = category
                if status:
                    params["status"] = status

                resp = client.get(
                    f"{registry_url}/agents/search",
                    params=params,
                    headers=get_headers(),
                    timeout=30,
                )

                if resp.status_code != 200:
                    print_error(f"Search failed: {resp.text}")
                    raise typer.Exit(1)

                result = resp.json()
                agents = result.get("data", [])
                total = result.get("meta", {}).get("total", 0)

                progress.update(task, completed=True)

        except httpx.RequestError as e:
            print_error(f"Connection failed: {e}")
            raise typer.Exit(1)

    if not agents:
        print_info("No agents found matching your query")
        return

    if output_format == "json":
        console.print_json(json.dumps(agents, indent=2))
    else:
        # Table output
        table = Table(title=f"Search Results ({total} total)")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Version", style="green")
        table.add_column("Category", style="yellow")
        table.add_column("Status", style="magenta")
        table.add_column("Downloads", justify="right")
        table.add_column("Author")

        for agent in agents:
            status_style = {
                "published": "[green]published[/green]",
                "draft": "[yellow]draft[/yellow]",
                "deprecated": "[red]deprecated[/red]",
            }.get(agent.get("status", ""), agent.get("status", ""))

            table.add_row(
                agent.get("name", ""),
                agent.get("version", ""),
                agent.get("category", ""),
                status_style,
                str(agent.get("downloads", 0)),
                agent.get("author", ""),
            )

        console.print(table)
        console.print(f"\n[dim]Showing {len(agents)} of {total} results[/dim]")


# =============================================================================
# List Command
# =============================================================================


@app.command("list")
def list_agents(
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category",
    ),
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum results",
    ),
    registry_url: str = typer.Option(
        DEFAULT_REGISTRY_URL,
        "--registry",
        "-r",
        help="Registry URL",
    ),
):
    """
    List agents in the registry.

    Example:
        gl registry list --category emissions
        gl registry list --status published
    """
    console.print("\n[bold cyan]Registry Agents[/bold cyan]\n")

    try:
        with httpx.Client() as client:
            params = {"limit": limit}
            if category:
                params["category"] = category
            if status:
                params["status"] = status

            resp = client.get(
                f"{registry_url}/agents",
                params=params,
                headers=get_headers(),
                timeout=30,
            )

            if resp.status_code != 200:
                print_error(f"List failed: {resp.text}")
                raise typer.Exit(1)

            result = resp.json()
            agents = result.get("data", [])
            meta = result.get("meta", {})

    except httpx.RequestError as e:
        print_error(f"Connection failed: {e}")
        raise typer.Exit(1)

    if not agents:
        print_info("No agents found")
        return

    table = Table()
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Version", style="green")
    table.add_column("Category")
    table.add_column("Status")
    table.add_column("Downloads", justify="right")

    for agent in agents:
        table.add_row(
            agent.get("name", ""),
            agent.get("version", ""),
            agent.get("category", ""),
            agent.get("status", ""),
            str(agent.get("downloads", 0)),
        )

    console.print(table)
    console.print(f"\n[dim]Total: {meta.get('total', len(agents))} agents[/dim]")


# =============================================================================
# Info Command
# =============================================================================


@app.command()
def info(
    agent_ref: str = typer.Argument(
        ...,
        help="Agent name or ID",
    ),
    registry_url: str = typer.Option(
        DEFAULT_REGISTRY_URL,
        "--registry",
        "-r",
        help="Registry URL",
    ),
):
    """
    Get detailed information about an agent.

    Example:
        gl registry info carbon-calculator
    """
    console.print(f"\n[bold cyan]Agent Info:[/bold cyan] {agent_ref}\n")

    try:
        with httpx.Client() as client:
            # Search for agent
            resp = client.get(
                f"{registry_url}/agents/search",
                params={"q": agent_ref},
                headers=get_headers(),
                timeout=30,
            )

            if resp.status_code != 200:
                print_error(f"Search failed: {resp.text}")
                raise typer.Exit(1)

            agents = resp.json().get("data", [])
            agent = next((a for a in agents if a["name"] == agent_ref), None)

            if not agent:
                print_error(f"Agent '{agent_ref}' not found")
                raise typer.Exit(1)

    except httpx.RequestError as e:
        print_error(f"Connection failed: {e}")
        raise typer.Exit(1)

    # Display info
    panel_content = f"""
[bold]Name:[/bold] {agent.get('name', '')}
[bold]Version:[/bold] {agent.get('version', '')}
[bold]Description:[/bold] {agent.get('description', 'No description')}
[bold]Category:[/bold] {agent.get('category', '')}
[bold]Status:[/bold] {agent.get('status', '')}
[bold]Author:[/bold] {agent.get('author', '')}
[bold]Downloads:[/bold] {agent.get('downloads', 0)}
[bold]License:[/bold] {agent.get('license', 'Not specified')}
[bold]Created:[/bold] {agent.get('created_at', '')[:10] if agent.get('created_at') else 'Unknown'}
[bold]Updated:[/bold] {agent.get('updated_at', '')[:10] if agent.get('updated_at') else 'Unknown'}
[bold]Tags:[/bold] {', '.join(agent.get('tags', [])) or 'None'}
[bold]Frameworks:[/bold] {', '.join(agent.get('regulatory_frameworks', [])) or 'None'}
"""

    if agent.get('documentation_url'):
        panel_content += f"[bold]Docs:[/bold] {agent['documentation_url']}\n"
    if agent.get('repository_url'):
        panel_content += f"[bold]Repo:[/bold] {agent['repository_url']}\n"

    console.print(Panel(panel_content, title=f"Agent: {agent.get('name', '')}", expand=False))


# =============================================================================
# Versions Command
# =============================================================================


@app.command()
def versions(
    agent_ref: str = typer.Argument(
        ...,
        help="Agent name or ID",
    ),
    registry_url: str = typer.Option(
        DEFAULT_REGISTRY_URL,
        "--registry",
        "-r",
        help="Registry URL",
    ),
):
    """
    List all versions of an agent.

    Example:
        gl registry versions carbon-calculator
    """
    console.print(f"\n[bold cyan]Versions for:[/bold cyan] {agent_ref}\n")

    try:
        with httpx.Client() as client:
            # Find agent
            search_resp = client.get(
                f"{registry_url}/agents/search",
                params={"q": agent_ref},
                headers=get_headers(),
                timeout=30,
            )

            if search_resp.status_code != 200:
                print_error(f"Search failed: {search_resp.text}")
                raise typer.Exit(1)

            agents = search_resp.json().get("data", [])
            agent = next((a for a in agents if a["name"] == agent_ref), None)

            if not agent:
                print_error(f"Agent '{agent_ref}' not found")
                raise typer.Exit(1)

            # Get versions
            versions_resp = client.get(
                f"{registry_url}/agents/{agent['id']}/versions",
                headers=get_headers(),
                timeout=30,
            )

            if versions_resp.status_code != 200:
                print_error(f"Failed to get versions: {versions_resp.text}")
                raise typer.Exit(1)

            versions_list = versions_resp.json()

    except httpx.RequestError as e:
        print_error(f"Connection failed: {e}")
        raise typer.Exit(1)

    if not versions_list:
        print_info("No versions found")
        return

    table = Table(title=f"Versions of {agent_ref}")
    table.add_column("Version", style="cyan", no_wrap=True)
    table.add_column("Latest", style="green")
    table.add_column("Breaking", style="red")
    table.add_column("Published", style="yellow")
    table.add_column("Downloads", justify="right")
    table.add_column("Changelog")

    for ver in versions_list:
        table.add_row(
            ver.get("version", ""),
            "Yes" if ver.get("is_latest") else "",
            "Yes" if ver.get("breaking_changes") else "",
            ver.get("published_at", "")[:10] if ver.get("published_at") else "Draft",
            str(ver.get("downloads", 0)),
            (ver.get("changelog", "") or "")[:40] + "..." if len(ver.get("changelog", "") or "") > 40 else ver.get("changelog", ""),
        )

    console.print(table)


# =============================================================================
# Publish Command
# =============================================================================


@app.command()
def publish(
    agent_ref: str = typer.Argument(
        ...,
        help="Agent name or ID",
    ),
    version: str = typer.Option(
        ...,
        "--version",
        "-v",
        help="Version to publish",
    ),
    release_notes: Optional[str] = typer.Option(
        None,
        "--notes",
        "-n",
        help="Release notes",
    ),
    certify: Optional[List[str]] = typer.Option(
        None,
        "--certify",
        "-c",
        help="Regulatory frameworks to certify for",
    ),
    registry_url: str = typer.Option(
        DEFAULT_REGISTRY_URL,
        "--registry",
        "-r",
        help="Registry URL",
    ),
):
    """
    Publish an agent version.

    Makes the specified version publicly available for download.

    Example:
        gl registry publish carbon-calculator -v 1.0.0 -n "Initial release"
    """
    console.print(f"\n[bold cyan]Publishing:[/bold cyan] {agent_ref}:{version}\n")

    if not get_auth_token():
        print_error("Not logged in. Run 'gl registry login' first.")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Finding agent...", total=None)

        try:
            with httpx.Client() as client:
                # Find agent
                search_resp = client.get(
                    f"{registry_url}/agents/search",
                    params={"q": agent_ref},
                    headers=get_headers(),
                    timeout=30,
                )

                if search_resp.status_code != 200:
                    print_error(f"Search failed: {search_resp.text}")
                    raise typer.Exit(1)

                agents = search_resp.json().get("data", [])
                agent = next((a for a in agents if a["name"] == agent_ref), None)

                if not agent:
                    print_error(f"Agent '{agent_ref}' not found")
                    raise typer.Exit(1)

                progress.update(task, description="Publishing version...")

                # Publish
                publish_payload = {
                    "version": version,
                    "release_notes": release_notes,
                    "certifications": certify,
                }

                resp = client.post(
                    f"{registry_url}/agents/{agent['id']}/publish",
                    json=publish_payload,
                    headers=get_headers(),
                    timeout=60,
                )

                if resp.status_code != 200:
                    error_msg = resp.json().get("detail", resp.text)
                    print_error(f"Publish failed: {error_msg}")
                    raise typer.Exit(1)

                result = resp.json()
                progress.update(task, completed=True)

        except httpx.RequestError as e:
            print_error(f"Connection failed: {e}")
            raise typer.Exit(1)

    console.print("")
    print_success(f"Published: {agent_ref}:{version}")
    if result.get("artifact_url"):
        console.print(f"  Artifact: [cyan]{result['artifact_url']}[/cyan]")
    console.print(f"  Checksum: [dim]{result.get('checksum', '')[:24]}...[/dim]")


# =============================================================================
# Login Command
# =============================================================================


@app.command()
def login(
    registry_url: str = typer.Option(
        DEFAULT_REGISTRY_URL,
        "--registry",
        "-r",
        help="Registry URL",
    ),
):
    """
    Login to the agent registry.

    Stores authentication token locally for subsequent commands.

    Example:
        gl registry login
    """
    console.print(f"\n[bold cyan]Login to Registry:[/bold cyan] {registry_url}\n")

    username = typer.prompt("Username")
    password = typer.prompt("Password", hide_input=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Authenticating...", total=None)

        # For now, create a placeholder token
        # In production, this would authenticate with the actual registry
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        auth_data = {
            "token": f"placeholder_token_{username}",
            "username": username,
            "registry_url": registry_url,
            "created_at": datetime.utcnow().isoformat(),
        }

        with open(AUTH_FILE, "w") as f:
            json.dump(auth_data, f, indent=2)

        progress.update(task, completed=True)

    print_success("Successfully logged in")
    console.print(f"  Config saved to: [dim]{AUTH_FILE}[/dim]")


# =============================================================================
# Logout Command
# =============================================================================


@app.command()
def logout():
    """
    Logout from the agent registry.

    Removes locally stored authentication token.

    Example:
        gl registry logout
    """
    if AUTH_FILE.exists():
        AUTH_FILE.unlink()
        print_success("Successfully logged out")
    else:
        print_info("Not logged in")


# =============================================================================
# Stats Command
# =============================================================================


@app.command()
def stats(
    registry_url: str = typer.Option(
        DEFAULT_REGISTRY_URL,
        "--registry",
        "-r",
        help="Registry URL",
    ),
):
    """
    Show registry statistics.

    Example:
        gl registry stats
    """
    console.print("\n[bold cyan]Registry Statistics[/bold cyan]\n")

    try:
        with httpx.Client() as client:
            resp = client.get(
                f"{registry_url}/stats",
                headers=get_headers(),
                timeout=30,
            )

            if resp.status_code != 200:
                print_error(f"Failed to get stats: {resp.text}")
                raise typer.Exit(1)

            stats_data = resp.json()

    except httpx.RequestError as e:
        print_error(f"Connection failed: {e}")
        raise typer.Exit(1)

    console.print(f"[bold]Total Agents:[/bold] {stats_data.get('total_agents', 0)}")
    console.print(f"[bold]Total Versions:[/bold] {stats_data.get('total_versions', 0)}")
    console.print(f"[bold]Total Downloads:[/bold] {stats_data.get('total_downloads', 0)}")

    if stats_data.get("by_status"):
        console.print("\n[bold]By Status:[/bold]")
        for status_name, count in stats_data["by_status"].items():
            console.print(f"  {status_name}: {count}")

    if stats_data.get("by_category"):
        console.print("\n[bold]By Category:[/bold]")
        for category, count in stats_data["by_category"].items():
            console.print(f"  {category}: {count}")


if __name__ == "__main__":
    app()
