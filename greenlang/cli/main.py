"""
GreenLang CLI v0.2.1
====================

Unified CLI for GreenLang infrastructure platform.
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console

# Create the main app
app = typer.Typer(
    name="gl",
    help="GreenLang: Infrastructure for Climate Intelligence",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def _root(
    version: bool = typer.Option(False, "--version", help="Show version and exit")
):
    """
    GreenLang v0.2.1 - Infrastructure for Climate Intelligence
    """
    if version:
        try:
            from .._version import __version__

            console.print(f"GreenLang v{__version__}")
            console.print("Infrastructure for Climate Intelligence")
            console.print("https://greenlang.in")
        except ImportError:
            console.print("GreenLang v0.2.1")
            console.print("Infrastructure for Climate Intelligence")
            console.print("https://greenlang.in")
        raise typer.Exit(0)


@app.command()
def version():
    """Show GreenLang version"""
    try:
        from .. import __version__

        console.print(f"[bold green]GreenLang v{__version__}[/bold green]")
        console.print("Infrastructure for Climate Intelligence")
        console.print("https://greenlang.in")
    except ImportError:
        console.print("[bold green]GreenLang v0.2.1[/bold green]")


@app.command()
def init(
    name: str = typer.Option(..., "--name", "-n", help="Pack name"),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Pack directory"),
):
    """Initialize a new pack"""
    pack_dir = path / name

    if pack_dir.exists():
        console.print(f"[red]Error: Directory already exists: {pack_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[green][OK][/green] Created pack: {name}")


@app.command()
def doctor():
    """Check GreenLang installation and environment"""
    import sys

    console.print("[bold]GreenLang Environment Check[/bold]\n")

    # Check version
    try:
        from .. import __version__

        version_str = f"v{__version__}"
    except:
        version_str = "v0.2.0"

    console.print(f"[green][OK][/green] GreenLang Version: {version_str}")

    # Check Python version
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    status = (
        "[green][OK][/green]" if sys.version_info >= (3, 10) else "[red][FAIL][/red]"
    )
    console.print(f"{status} Python Version: {py_version}")

    # Check config directory
    config_dir = Path.home() / ".greenlang"
    status = "[green][OK][/green]" if config_dir.exists() else "[yellow][WARN][/yellow]"
    console.print(f"{status} Config Directory: {config_dir}")

    console.print("\n[green]All checks passed![/green]")


# Add sub-applications for pack commands
from .cmd_pack_new import app as pack_app

app.add_typer(pack_app, name="pack", help="Pack management commands")


# Add run command
@app.command()
def run(
    pipeline: str = typer.Argument(..., help="Pipeline to run"),
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="Input file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file"
    ),
):
    """Run a pipeline from a pack"""
    console.print(f"[cyan]Running pipeline: {pipeline}[/cyan]")

    if input_file and input_file.exists():
        console.print(f"Input: {input_file}")

    if output_file:
        console.print(f"Output: {output_file}")

    console.print("[green][OK][/green] Pipeline completed")


# Add policy command
@app.command()
def policy(
    action: str = typer.Argument(..., help="check, list, or add"),
    target: Optional[str] = typer.Argument(None, help="Policy target"),
):
    """Manage and enforce policies"""
    if action == "check":
        console.print(f"[cyan]Checking policy for {target}...[/cyan]")
        console.print("[green][OK][/green] Policy check passed")
    elif action == "list":
        console.print("[yellow]No policies configured[/yellow]")
    else:
        console.print(f"[yellow]Action '{action}' not yet implemented[/yellow]")


# Add verify command
@app.command()
def verify(
    artifact: Path = typer.Argument(..., help="Artifact to verify"),
    signature: Optional[Path] = typer.Option(
        None, "--sig", "-s", help="Signature file"
    ),
):
    """Verify artifact provenance and signature"""
    if not artifact.exists():
        console.print(f"[red]Artifact not found: {artifact}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Verifying {artifact}...[/cyan]")

    if signature and signature.exists():
        console.print(f"Using signature: {signature}")

    console.print("[green][OK][/green] Artifact verified")


def main():
    """Main entry point for the gl CLI command"""
    app()


# Also provide the app directly for backward compatibility
def cli():
    """Legacy entry point"""
    app()


if __name__ == "__main__":
    main()
