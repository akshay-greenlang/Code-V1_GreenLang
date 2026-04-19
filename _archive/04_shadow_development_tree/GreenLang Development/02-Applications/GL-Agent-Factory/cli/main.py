"""
GreenLang Agent Factory CLI - Main Entry Point

This module provides the main CLI application using Typer.
"""

import typer
from typing import Optional
from pathlib import Path

from cli.commands import agent, template, registry, formula, standards, ontology, scaffold, shadow
from cli.utils.console import console, print_banner, print_error

# Create main Typer app
app = typer.Typer(
    name="gl",
    help="GreenLang Agent Factory CLI - Generate, validate, test, and publish agents",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Add command groups
app.add_typer(
    agent.app,
    name="agent",
    help="Agent management commands (create, validate, test, publish)",
)
app.add_typer(
    template.app,
    name="template",
    help="Template management commands (list, init)",
)
app.add_typer(
    registry.app,
    name="registry",
    help="Registry management commands (search, pull, push)",
)
app.add_typer(
    formula.app,
    name="formula",
    help="Formula management (search, validate, test, list)",
)
app.add_typer(
    standards.app,
    name="standards",
    help="Standards management (search, equipment, section)",
)
app.add_typer(
    ontology.app,
    name="ontology",
    help="Ontology management (query, equipment, export)",
)
app.add_typer(
    scaffold.app,
    name="scaffold",
    help="Scaffolding (agent, test, docs from templates)",
)
app.add_typer(
    shadow.app,
    name="shadow",
    help="Shadow mode testing (record, replay, compare, report)",
)


@app.callback()
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        is_eager=True,
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-essential output",
    ),
):
    """
    GreenLang Agent Factory CLI

    A powerful command-line tool for managing AI agents in the GreenLang ecosystem.
    """
    if version:
        from cli import __version__
        console.print(f"[bold green]GreenLang Agent Factory CLI[/bold green] version [cyan]{__version__}[/cyan]")
        raise typer.Exit(0)

    # Store quiet flag in context for commands to access
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet


@app.command()
def init(
    directory: Path = typer.Argument(
        Path.cwd(),
        help="Directory to initialize",
    ),
    template: Optional[str] = typer.Option(
        None,
        "--template",
        "-t",
        help="Template to use (default: basic)",
    ),
):
    """
    Initialize a new Agent Factory project in the current directory.
    """
    from cli.utils.console import print_success, print_info

    console.print("\n[bold cyan]Initializing GreenLang Agent Factory project...[/bold cyan]\n")

    # Create basic structure
    directories = [
        "agents",
        "specs",
        "tests",
        "templates",
        "config",
    ]

    for dir_name in directories:
        dir_path = directory / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print_success(f"Created directory: {dir_path}")

    # Create initial config file
    config_file = directory / "config" / "factory.yaml"
    if not config_file.exists():
        config_content = """# GreenLang Agent Factory Configuration
version: "1.0"

# Default settings for agent generation
defaults:
  output_dir: "agents"
  test_dir: "tests"

# Registry settings
registry:
  url: "https://registry.greenlang.io"

# Generator settings
generator:
  enable_validation: true
  enable_tests: true
  enable_documentation: true
"""
        config_file.write_text(config_content)
        print_success(f"Created config file: {config_file}")

    print_info("\n[bold green]Project initialized successfully![/bold green]")
    console.print("\nNext steps:")
    console.print("  1. Create an agent spec: [cyan]gl agent create specs/my-agent.yaml[/cyan]")
    console.print("  2. Validate the spec: [cyan]gl agent validate specs/my-agent.yaml[/cyan]")
    console.print("  3. Run tests: [cyan]gl agent test agents/my-agent[/cyan]")


def cli_main():
    """Entry point for the CLI when installed via pip."""
    try:
        app()
    except KeyboardInterrupt:
        print_error("\n\nOperation cancelled by user")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    cli_main()
