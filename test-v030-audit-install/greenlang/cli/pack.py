"""
Pack Management CLI Commands for GreenLang
Maps user-friendly pack commands to hub operations
"""

import click
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from greenlang.hub import (
    HubClient,
    HubAuth,
    PackIndex,
    SearchFilters,
    SortOrder,
    create_manifest,
    load_manifest,
)

console = Console()


@click.group()
def pack():
    """Pack management commands for GreenLang"""


@pack.command()
@click.argument("pack_path", type=click.Path(exists=True))
@click.option("--tag", "-t", multiple=True, help="Pack tags")
@click.option("--description", "-d", help="Pack description")
@click.option("--registry", default="https://hub.greenlang.io", help="Registry URL")
@click.option(
    "--skip-policy", is_flag=True, help="Skip policy checks (not recommended)"
)
def publish(
    pack_path: str, tag: tuple, description: str, registry: str, skip_policy: bool
):
    """Publish a pack to the registry

    Example:
        gl pack publish my-pack
        gl pack publish ./dist/my-pack.tar.gz --tag carbon --tag emissions
    """
    try:
        console.print(f"[cyan]Publishing pack from {pack_path}...[/cyan]")

        if skip_policy:
            console.print(
                "[bold yellow]⚠️  WARNING: Skipping policy checks for publishing![/bold yellow]"
            )
            import os

            os.environ["GL_SKIP_PUBLISH_POLICY"] = "1"

        # Check authentication
        auth = HubAuth()
        if not auth.token and not auth.api_key:
            console.print("[yellow]Not authenticated. Please login first.[/yellow]")
            username = click.prompt("Username")
            password = click.prompt("Password", hide_input=True)

            if not auth.login(username, password, registry):
                console.print("[red]✗ Authentication failed[/red]")
                sys.exit(1)
            console.print("[green]✓ Logged in successfully[/green]")

        # Publish pack
        with HubClient(registry, auth) as client:
            result = client.push(
                Path(pack_path),
                tags=list(tag) if tag else None,
                description=description,
            )

            pack_id = result.get("id", "unknown")
            pack_url = result.get("url", f"{registry}/packs/{pack_id}")

            console.print(
                Panel.fit(
                    f"[bold green]✓ Pack published successfully![/bold green]\n\n"
                    f"ID: {pack_id}\n"
                    f"URL: {pack_url}",
                    title="Success",
                )
            )

    except Exception as e:
        console.print(f"[red]✗ Publish failed: {e}[/red]")
        sys.exit(1)


@pack.command()
@click.argument("pack_ref")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--registry", default="https://hub.greenlang.io", help="Registry URL")
@click.option("--no-verify", is_flag=True, help="Skip signature verification")
@click.option("--allow-unsigned", is_flag=True, help="Allow unsigned packs (dev only)")
def add(
    pack_ref: str,
    output: str,
    registry: str,
    no_verify: bool,
    allow_unsigned: bool,
):
    """Add (install) a pack from the registry

    Example:
        gl pack add greenlang/emissions-core@1.0.0
        gl pack add carbon-calculator
        gl pack add user/pack-name@2.0.0 --output ./packs
    """
    try:
        console.print(f"[cyan]Installing pack: {pack_ref}...[/cyan]")

        # Warn about dangerous flags
        if allow_unsigned:
            console.print(
                "[bold yellow]⚠️  WARNING: --allow-unsigned flag used. Security checks relaxed![/bold yellow]"
            )

        # Setup policy environment
        import os

        if allow_unsigned:
            os.environ["GL_ALLOW_UNSIGNED"] = "1"

        # Optional authentication for private packs
        auth = HubAuth()

        # Pull pack
        with HubClient(registry, auth if auth.token else None) as client:
            output_dir = Path(output) if output else None
            pack_dir = client.pull(pack_ref, output_dir, verify_signature=not no_verify)

            # Load manifest to show info
            manifest = load_manifest(pack_dir)

            console.print(
                Panel.fit(
                    f"[bold green]✓ Pack installed successfully![/bold green]\n\n"
                    f"Name: {manifest.name}\n"
                    f"Version: {manifest.version}\n"
                    f"Location: {pack_dir}",
                    title="Success",
                )
            )

            # Show dependencies if any
            if manifest.dependencies:
                console.print("\n[yellow]Dependencies:[/yellow]")
                for dep in manifest.dependencies:
                    console.print(f"  • {dep.name}@{dep.version}")

    except Exception as e:
        console.print(f"[red]✗ Installation failed: {e}[/red]")
        sys.exit(1)


@pack.command()
@click.argument("query", required=False)
@click.option("--category", "-c", help="Filter by category")
@click.option("--tag", "-t", multiple=True, help="Filter by tag")
@click.option("--author", "-a", help="Filter by author")
@click.option("--verified", is_flag=True, help="Verified packs only")
@click.option("--limit", "-l", default=20, help="Maximum results")
@click.option(
    "--sort",
    type=click.Choice(["relevance", "downloads", "stars", "updated"]),
    default="relevance",
    help="Sort order",
)
def search(
    query: str,
    category: str,
    tag: tuple,
    author: str,
    verified: bool,
    limit: int,
    sort: str,
):
    """Search for packs in the registry

    Example:
        gl pack search carbon
        gl pack search "emissions calculator" --verified
        gl pack search --category monitoring --tag cloud
        gl pack search --author greenlang --sort downloads
    """
    try:
        if not query and not category and not tag and not author:
            console.print(
                "[yellow]Showing featured packs (use search terms for specific results)[/yellow]\n"
            )
            # Show featured packs if no search criteria
            index = PackIndex()
            packs = index.get_featured(limit=limit)
        else:
            # Build search filters
            filters = SearchFilters(
                categories=[category] if category else None,
                tags=list(tag) if tag else None,
                author=author,
                verified_only=verified,
            )

            # Search
            index = PackIndex()
            sort_order = SortOrder(sort)
            packs = index.search(
                query=query, filters=filters, sort=sort_order, limit=limit
            )

        if not packs:
            console.print("[yellow]No packs found matching your criteria[/yellow]")
            return

        # Display results in a table
        table = Table(title=f"Pack Search Results ({len(packs)} found)")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Description", style="white")
        table.add_column("Author", style="blue")
        table.add_column("Stats", style="yellow")

        for pack in packs:
            # Build stats string
            stats = []
            if pack.downloads > 0:
                stats.append(f"↓{pack.downloads:,}")
            if pack.stars > 0:
                stats.append(f"★{pack.stars}")

            # Add badges
            badges = []
            if pack.verified:
                badges.append("✓")
            if pack.official:
                badges.append("Official")

            name_display = pack.name
            if badges:
                name_display += f" [{' '.join(badges)}]"

            table.add_row(
                name_display,
                pack.version,
                (
                    pack.description[:50] + "..."
                    if len(pack.description) > 50
                    else pack.description
                ),
                pack.author.get("name", "unknown"),
                " ".join(stats),
            )

        console.print(table)

        # Show installation hint
        if packs:
            console.print(
                f"\n[dim]To install a pack, use: gl pack add <pack-name>@<version>[/dim]"
            )

    except Exception as e:
        console.print(f"[red]✗ Search failed: {e}[/red]")
        sys.exit(1)


@pack.command()
@click.argument("pack_ref")
def info(pack_ref: str):
    """Show detailed information about a pack

    Example:
        gl pack info emissions-core
        gl pack info greenlang/carbon-calculator@1.0.0
    """
    try:
        console.print(f"[cyan]Fetching information for {pack_ref}...[/cyan]\n")

        with HubClient() as client:
            info = client.get_pack_info(pack_ref)

            manifest = info.get("manifest", {})
            stats = info.get("stats", {})

            # Display detailed info
            console.print(
                Panel.fit(
                    f"[bold]{manifest.get('name', 'unknown')}[/bold] v{manifest.get('version', '')}\n"
                    f"{manifest.get('description', 'No description')}\n\n"
                    f"[bold]Author:[/bold] {manifest.get('author', {}).get('name', 'Unknown')}\n"
                    f"[bold]License:[/bold] {manifest.get('license', 'Not specified')}\n"
                    f"[bold]Homepage:[/bold] {manifest.get('homepage', 'Not specified')}\n\n"
                    f"[bold]Statistics:[/bold]\n"
                    f"  Downloads: {stats.get('downloads', 0):,}\n"
                    f"  Stars: {stats.get('stars', 0):,}\n"
                    f"  Published: {info.get('published_at', 'unknown')}",
                    title=f"Pack Information",
                )
            )

            # Show dependencies
            deps = manifest.get("dependencies", [])
            if deps:
                console.print("\n[bold]Dependencies:[/bold]")
                for dep in deps:
                    console.print(f"  • {dep.get('name')}@{dep.get('version')}")

            # Show installation command
            console.print(f"\n[dim]To install: gl pack add {pack_ref}[/dim]")

    except Exception as e:
        console.print(f"[red]✗ Failed to get pack info: {e}[/red]")
        sys.exit(1)


@pack.command()
@click.option("--user", "-u", help="List packs by user")
@click.option("--installed", is_flag=True, help="List installed packs")
@click.option("--limit", "-l", default=50, help="Maximum results")
def list(user: str, installed: bool, limit: int):
    """List packs from registry or installed locally

    Example:
        gl pack list                    # List all packs
        gl pack list --user greenlang   # List packs by user
        gl pack list --installed        # List locally installed packs
    """
    try:
        if installed:
            # List locally installed packs
            packs_dir = Path.home() / ".greenlang" / "packs"
            if not packs_dir.exists():
                console.print("[yellow]No packs installed yet[/yellow]")
                return

            console.print("[bold]Installed Packs:[/bold]\n")

            for pack_dir in packs_dir.iterdir():
                if pack_dir.is_dir():
                    try:
                        manifest = load_manifest(pack_dir)
                        console.print(
                            f"  • {manifest.name}@{manifest.version} - {manifest.description}"
                        )
                    except:
                        console.print(f"  • {pack_dir.name} (manifest not found)")
        else:
            # List from registry
            with HubClient() as client:
                packs = client.list_packs(user=user, limit=limit)

                if not packs:
                    console.print("[yellow]No packs found[/yellow]")
                    return

                title = f"Packs by {user}" if user else "Available Packs"
                table = Table(title=f"{title} ({len(packs)} shown)")
                table.add_column("Name", style="cyan")
                table.add_column("Version", style="green")
                table.add_column("Downloads", style="yellow")
                table.add_column("Stars", style="yellow")

                for pack in packs:
                    table.add_row(
                        pack.get("name", "unknown"),
                        pack.get("version", ""),
                        f"{pack.get('downloads', 0):,}",
                        f"{pack.get('stars', 0):,}",
                    )

                console.print(table)

    except Exception as e:
        console.print(f"[red]✗ List failed: {e}[/red]")
        sys.exit(1)


@pack.command()
@click.argument("pack_ref")
@click.option("--confirm", is_flag=True, help="Skip confirmation")
def remove(pack_ref: str, confirm: bool):
    """Remove an installed pack

    Example:
        gl pack remove emissions-core
        gl pack remove carbon-calculator@1.0.0 --confirm
    """
    try:
        # Parse pack reference
        if "@" in pack_ref:
            pack_name, version = pack_ref.split("@")
        else:
            pack_name = pack_ref
            version = None

        # Find installed pack
        packs_dir = Path.home() / ".greenlang" / "packs"
        pack_dir = packs_dir / pack_name

        if not pack_dir.exists():
            console.print(f"[yellow]Pack '{pack_name}' is not installed[/yellow]")
            return

        # Confirm removal
        if not confirm:
            if not click.confirm(f"Remove pack '{pack_name}'?"):
                console.print("Cancelled")
                return

        # Remove pack directory
        import shutil

        shutil.rmtree(pack_dir)

        console.print(f"[green]✓ Pack '{pack_name}' removed successfully[/green]")

    except Exception as e:
        console.print(f"[red]✗ Remove failed: {e}[/red]")
        sys.exit(1)


@pack.command()
@click.argument("pack_dir", type=click.Path(exists=True), required=False)
@click.option("--name", "-n", help="Pack name")
@click.option("--version", "-v", default="0.1.0", help="Pack version")
@click.option("--description", "-d", help="Pack description")
@click.option("--author", "-a", help="Author name")
def init(pack_dir: str, name: str, version: str, description: str, author: str):
    """Initialize a new pack with manifest

    Example:
        gl pack init                    # Initialize in current directory
        gl pack init my-pack --name "My Pack" --author "John Doe"
    """
    try:
        pack_path = Path(pack_dir) if pack_dir else Path.cwd()

        # Check if manifest already exists
        manifest_files = ["manifest.json", "manifest.yaml", "greenlang.json"]
        for mf in manifest_files:
            if (pack_path / mf).exists():
                if not click.confirm(f"{mf} already exists. Overwrite?"):
                    console.print("Cancelled")
                    return

        # Create manifest
        manifest = create_manifest(
            pack_path,
            name=name or pack_path.name,
            version=version,
            description=description or f"{name or pack_path.name} GreenLang pack",
            author=author,
        )

        # Save manifest
        from greenlang.hub.manifest import save_manifest

        manifest_file = save_manifest(pack_path, manifest)

        console.print(
            Panel.fit(
                f"[bold green]✓ Pack initialized successfully![/bold green]\n\n"
                f"Name: {manifest.name}\n"
                f"Version: {manifest.version}\n"
                f"Manifest: {manifest_file}\n\n"
                f"Next steps:\n"
                f"1. Add your code and resources\n"
                f"2. Update manifest.json as needed\n"
                f"3. Run 'gl pack publish {pack_path}' to publish",
                title="Pack Initialized",
            )
        )

    except Exception as e:
        console.print(f"[red]✗ Init failed: {e}[/red]")
        sys.exit(1)


def register_pack_commands(cli):
    """Register pack commands with main CLI"""
    cli.add_command(pack)
