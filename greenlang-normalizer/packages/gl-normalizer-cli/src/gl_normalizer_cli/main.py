"""
GL-FOUND-X-003: GreenLang Normalizer CLI - Main Entry Point

This module provides the main CLI application using Typer, with support for
unit normalization, batch processing, vocabulary management, and configuration.

Example:
    >>> # From command line
    >>> glnorm normalize 100 kg --to metric_ton
    >>> glnorm batch input.csv --output output.json
    >>> glnorm vocab list
    >>> glnorm config show
"""

from typing import Optional

import typer
from rich.console import Console

from gl_normalizer_cli import __version__
from gl_normalizer_cli.commands import normalize, batch, vocab, config

# Initialize console for rich output
console = Console()

# Create the main Typer application
app = typer.Typer(
    name="glnorm",
    help=(
        "GreenLang Unit & Reference Normalizer CLI\n\n"
        "A command-line tool for normalizing units, converting measurements, "
        "and resolving reference data for sustainability applications.\n\n"
        "For more information, visit: https://docs.greenlang.io/normalizer/cli"
    ),
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register command groups
app.add_typer(vocab.app, name="vocab", help="Manage and search vocabularies")
app.add_typer(config.app, name="config", help="Manage CLI configuration")

# Register top-level commands
app.command(name="normalize", help="Normalize a single value with unit")(normalize.normalize_value)
app.command(name="batch", help="Process a batch of records from a file")(batch.process_batch)


def version_callback(value: bool) -> None:
    """Display version information and exit."""
    if value:
        console.print(f"[bold blue]glnorm[/bold blue] version [green]{__version__}[/green]")
        console.print(f"GreenLang Unit & Reference Normalizer CLI")
        console.print(f"Agent: GL-FOUND-X-003")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version information and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output with additional details.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-essential output.",
    ),
) -> None:
    """
    GreenLang Unit & Reference Normalizer CLI.

    A command-line tool for normalizing units, converting measurements,
    and resolving reference data for sustainability applications.

    [bold]Quick Start:[/bold]

        glnorm normalize 100 kg --to metric_ton

        glnorm batch input.csv --output output.json

        glnorm vocab search "natural gas"

    [bold]Configuration:[/bold]

        glnorm config init

        glnorm config set api_key YOUR_KEY

    For detailed help on any command, use: glnorm COMMAND --help
    """
    # Store verbose/quiet settings in context for subcommands
    ctx = typer.Context
    # Note: Context state handled via callback pattern


@app.command(name="version")
def show_version() -> None:
    """
    Display detailed version information.

    Shows version numbers for the CLI tool, core library, and agent identifier.

    [bold]Example:[/bold]

        glnorm version
    """
    console.print()
    console.print("[bold blue]GreenLang Normalizer CLI[/bold blue]")
    console.print(f"  CLI Version:   [green]{__version__}[/green]")

    # Try to get core library version
    try:
        from gl_normalizer_core import __version__ as core_version
        console.print(f"  Core Version:  [green]{core_version}[/green]")
    except ImportError:
        console.print("  Core Version:  [yellow]Not installed[/yellow]")

    console.print(f"  Agent ID:      [cyan]GL-FOUND-X-003[/cyan]")
    console.print()

    # Show configuration status
    from gl_normalizer_cli.commands.config import get_config_path, load_config

    config_path = get_config_path()
    if config_path.exists():
        console.print(f"  Config File:   [green]{config_path}[/green]")
        cfg = load_config()
        if cfg.get("api_key"):
            console.print("  API Mode:      [green]Configured[/green]")
        else:
            console.print("  API Mode:      [yellow]Not configured[/yellow]")
    else:
        console.print(f"  Config File:   [yellow]Not initialized[/yellow]")
        console.print("  Run 'glnorm config init' to create configuration.")

    console.print()


if __name__ == "__main__":
    app()
