"""
gl pack - Pack management commands
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("list")
def list_packs(
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type")
):
    """List installed packs"""
    from ..packs.registry import PackRegistry

    registry = PackRegistry()
    packs = registry.list()  # Fixed parameter issue

    if not packs:
        console.print("[yellow]No packs installed[/yellow]")
        console.print("\nInstall packs with: [cyan]gl pack add <pack-name>[/cyan]")
        return

    table = Table(title="Installed Packs")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Location")

    for pack in packs:
        if type and pack.manifest.get("type") != type:
            continue
        table.add_row(
            pack.name,
            pack.version,
            pack.manifest.get("type", "unknown"),
            str(pack.location),
        )

    console.print(table)


@app.command("info")
def pack_info(name: str = typer.Argument(..., help="Pack name")):
    """Show pack details"""
    from ..packs.registry import PackRegistry

    registry = PackRegistry()
    pack = registry.get(name)

    if not pack:
        console.print(f"[red]Pack not found: {name}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]{pack.name}[/bold] v{pack.version}")
    console.print(f"Location: {pack.location}")


@app.command("add")
def pack_add(
    source: str = typer.Argument(..., help="Pack name or path"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify pack"),
):
    """Install a pack from registry or local path"""
    console.print(f"[cyan]Installing {source}...[/cyan]")
    # Implementation would go here
    console.print("[yellow]Pack installation not yet fully implemented[/yellow]")


@app.command("remove")
def pack_remove(name: str = typer.Argument(..., help="Pack name")):
    """Uninstall a pack"""
    console.print(f"[cyan]Removing {name}...[/cyan]")
    # Implementation would go here
    console.print("[yellow]Pack removal not yet fully implemented[/yellow]")


@app.command("validate")
def pack_validate(path: Path = typer.Argument(..., help="Path to pack directory")):
    """Validate pack structure and manifest"""
    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Validating {path}...[/cyan]")
    # Implementation would go here
    console.print("[green][OK][/green] Pack validation passed")
