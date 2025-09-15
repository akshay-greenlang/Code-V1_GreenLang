"""GreenLang CLI - Main entry point."""
import typer
from rich.console import Console

# Import command modules
from .cmd_demo import app as demo_app
from .cmd_validate import app as validate_app

console = Console()

# Create main app
app = typer.Typer(
    name="gl",
    help="GreenLang: Infrastructure for Climate Intelligence",
    no_args_is_help=True,
    rich_markup_mode="rich"
)

# Add subcommands
app.add_typer(demo_app, name="demo", help="Run demo pipelines")
app.add_typer(validate_app, name="validate", help="Validate pipeline and pack files")


def main():
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# Legacy support
cli = app

__all__ = ["app", "main", "cli"]