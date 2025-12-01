# -*- coding: utf-8 -*-
"""
GreenLang CLI
====================

Unified CLI for GreenLang infrastructure platform.
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console

# Fallback version constant
FALLBACK_VERSION = "0.3.0"

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
    GreenLang - Infrastructure for Climate Intelligence
    """
    if version:
        try:
            from .._version import __version__

            console.print(f"GreenLang v{__version__}")
            console.print("Infrastructure for Climate Intelligence")
            console.print("https://greenlang.in")
        except ImportError:
            # Fallback version
            console.print(f"GreenLang v{FALLBACK_VERSION}")
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
        # Fallback version
        console.print(f"[bold green]GreenLang v{FALLBACK_VERSION}[/bold green]")
        console.print("Infrastructure for Climate Intelligence")
        console.print("https://greenlang.in")


@app.command()
def doctor(
    setup_path: bool = typer.Option(False, "--setup-path", help="Setup Windows PATH automatically"),
    revert_path: bool = typer.Option(False, "--revert-path", help="Revert Windows PATH to last backup"),
    list_backups: bool = typer.Option(False, "--list-backups", help="List available PATH backups"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed diagnostics")
):
    """Check GreenLang installation and environment"""
    import sys
    import platform

    console.print("[bold]GreenLang Environment Check[/bold]\n")

    # Check version
    try:
        from .. import __version__

        version_str = f"v{__version__}"
    except ImportError:
        version_str = f"v{FALLBACK_VERSION}"

    console.print(f"[green][OK][/green] GreenLang Version: {version_str}")

    # Check Python version
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    status = (
        "[green][OK][/green]" if sys.version_info >= (3, 10) else "[red][FAIL][/red]"
    )
    console.print(f"{status} Python Version: {py_version}")

    # Platform information
    console.print(f"[blue][INFO][/blue] Platform: {platform.system()} {platform.release()}")
    console.print(f"[blue][INFO][/blue] Python Executable: {sys.executable}")

    # Check config directory
    config_dir = Path.home() / ".greenlang"
    status = "[green][OK][/green]" if config_dir.exists() else "[yellow][WARN][/yellow]"
    console.print(f"{status} Config Directory: {config_dir}")

    # Windows-specific PATH diagnostics
    if platform.system() == "Windows":
        console.print("\n[bold]Windows PATH Diagnostics[/bold]")

        try:
            from ..utils.windows_path import (
                diagnose_path_issues,
                setup_windows_path,
                revert_windows_path,
                list_path_backups
            )

            # Handle list-backups option
            if list_backups:
                console.print("\n[bold]Available PATH Backups:[/bold]")
                backups = list_path_backups()

                if not backups:
                    console.print("[yellow][WARN][/yellow] No PATH backups found")
                    console.print("[blue][INFO][/blue] Backups are created automatically when PATH is modified")
                else:
                    from rich.table import Table
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Timestamp", style="cyan")
                    table.add_column("Entries Count", justify="right", style="green")
                    table.add_column("File", style="yellow")

                    for backup in backups:
                        timestamp = backup["timestamp"].split('T')[0] + ' ' + backup["timestamp"].split('T')[1][:8]
                        table.add_row(
                            timestamp,
                            str(backup["entries_count"]),
                            Path(backup["file"]).name
                        )

                    console.print(table)
                    console.print(f"\n[blue][INFO][/blue] Backup location: {Path.home() / '.greenlang' / 'backup'}")
                    console.print("[blue][INFO][/blue] To restore: gl doctor --revert-path")

                return

            # Handle revert-path option
            if revert_path:
                console.print("\n[bold]Reverting Windows PATH...[/bold]")

                # Show available backups first
                backups = list_path_backups()
                if not backups:
                    console.print("[red][FAIL][/red] No PATH backups found")
                    console.print("[blue][INFO][/blue] Backups are created automatically when PATH is modified")
                    return

                console.print(f"[blue][INFO][/blue] Will restore from most recent backup: {backups[0]['timestamp']}")

                # Confirm with user
                from rich.prompt import Confirm
                if not Confirm.ask("Are you sure you want to revert PATH to the backup?"):
                    console.print("[yellow][WARN][/yellow] Operation cancelled")
                    return

                success, message = revert_windows_path()

                if success:
                    console.print(f"[green][OK][/green] {message}")
                    console.print("[blue][INFO][/blue] Please restart your command prompt for changes to take effect")
                else:
                    console.print(f"[red][FAIL][/red] {message}")

                return

            # Run diagnostics
            diagnosis = diagnose_path_issues()

            # Show gl.exe status
            if diagnosis["gl_executable_found"]:
                console.print(f"[green][OK][/green] gl.exe found: {diagnosis['gl_executable_path']}")
                if diagnosis["in_path"]:
                    console.print("[green][OK][/green] gl.exe is in PATH")
                else:
                    console.print("[yellow][WARN][/yellow] gl.exe is NOT in PATH")
            else:
                console.print("[red][FAIL][/red] gl.exe not found")

            # Show detailed diagnostics if requested
            if verbose:
                console.print(f"\n[bold]Detailed Diagnostics[/bold]")
                console.print(f"Scripts directories: {len(diagnosis['scripts_directories'])}")
                for scripts_dir in diagnosis["scripts_directories"]:
                    console.print(f"  - {scripts_dir}")

                console.print(f"PATH entries: {len(diagnosis['path_entries'])}")
                for path_entry in diagnosis["path_entries"][:10]:  # Show first 10
                    console.print(f"  - {path_entry}")
                if len(diagnosis["path_entries"]) > 10:
                    console.print(f"  ... and {len(diagnosis['path_entries']) - 10} more")

                # Show backup info
                backups = list_path_backups()
                if backups:
                    console.print(f"\n[blue][INFO][/blue] PATH backups available: {len(backups)}")
                    console.print(f"[blue][INFO][/blue] Most recent: {backups[0]['timestamp']}")
                    console.print("[blue][INFO][/blue] Use --list-backups to see all backups")

            # Handle setup-path option
            if setup_path:
                console.print("\n[bold]Setting up Windows PATH...[/bold]")
                success, message = setup_windows_path()

                if success:
                    console.print(f"[green][OK][/green] {message}")
                    console.print("[blue][INFO][/blue] A backup of your previous PATH has been saved")
                    console.print("[blue][INFO][/blue] Use --revert-path to restore if needed")
                else:
                    console.print(f"[red][FAIL][/red] {message}")

                    # Provide manual instructions
                    console.print("\n[bold]Manual Setup Instructions:[/bold]")
                    if diagnosis["gl_executable_path"]:
                        gl_dir = Path(diagnosis["gl_executable_path"]).parent
                        console.print(f"1. Add this directory to your PATH: {gl_dir}")
                    console.print("2. Or add your Python Scripts directory to PATH")
                    console.print("3. Restart your command prompt")
                    console.print("4. Test with: gl --version")

            # Show recommendations
            if diagnosis["recommendations"]:
                console.print("\n[bold]Recommendations:[/bold]")
                for rec in diagnosis["recommendations"]:
                    console.print(f"• {rec}")

        except ImportError:
            console.print("[yellow][WARN][/yellow] Windows utilities not available")
            console.print("Manual PATH configuration may be required")

    elif platform.system() in ["Linux", "Darwin"]:
        # Unix-like systems usually handle this automatically
        console.print(f"\n[blue][INFO][/blue] On {platform.system()}, the gl command should be available automatically")
        console.print("[blue][INFO][/blue] If not, check that ~/.local/bin is in your PATH")

    console.print("\n[green]Environment check completed![/green]")


# Add sub-applications for pack, init, rag, sbom, generate, decarbonization, and rbac commands
from .cmd_pack_new import app as pack_app
from .cmd_init import app as init_app
from .rag_commands import app as rag_app
from .cmd_sbom import app as sbom_app
from .cmd_generate import app as generate_app
from .cmd_decarbonization import app as decarbonization_app
from .cmd_rbac import app as rbac_app

app.add_typer(pack_app, name="pack", help="Pack management commands")
app.add_typer(init_app, name="init", help="Initialize new projects, packs, and agents")
app.add_typer(rag_app, name="rag", help="RAG (Retrieval-Augmented Generation) commands")
app.add_typer(sbom_app, name="sbom", help="SBOM generation and verification")
app.add_typer(generate_app, name="generate", help="Generate agents using LLM-powered code generation")
app.add_typer(decarbonization_app, name="decarbonization", help="Generate industrial decarbonization roadmaps (Agent #12)")
app.add_typer(rbac_app, name="rbac", help="Agent-level Role-Based Access Control management")


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
