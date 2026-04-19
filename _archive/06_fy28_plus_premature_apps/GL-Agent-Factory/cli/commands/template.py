"""
Template Management Commands

Commands for managing agent templates.
"""

import typer
from typing import Optional
from pathlib import Path
import yaml
import shutil

from cli.utils.console import (
    console,
    print_error,
    print_success,
    print_info,
    create_agent_table,
    display_yaml,
)


# Create template command group
app = typer.Typer(
    help="Template management commands",
    no_args_is_help=True,
)


@app.command()
def list(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed template information",
    ),
):
    """
    List all available agent templates.

    Example:
        gl template list --verbose
    """
    try:
        console.print("\n[bold cyan]Available Templates:[/bold cyan]\n")

        templates = [
            {
                "id": "basic",
                "name": "Basic Agent",
                "version": "1.0.0",
                "type": "template",
                "status": "active",
                "updated": "2024-12-01",
            },
            {
                "id": "regulatory",
                "name": "Regulatory Compliance Agent",
                "version": "1.0.0",
                "type": "template",
                "status": "active",
                "updated": "2024-12-01",
            },
            {
                "id": "api",
                "name": "API Integration Agent",
                "version": "1.0.0",
                "type": "template",
                "status": "active",
                "updated": "2024-12-01",
            },
        ]

        table = create_agent_table(templates)
        console.print(table)

        console.print(f"\n[dim]Total templates: {len(templates)}[/dim]\n")

    except Exception as e:
        print_error(f"Failed to list templates: {str(e)}")
        raise typer.Exit(1)


@app.command()
def init(
    template_name: str = typer.Argument(
        ...,
        help="Template name to initialize from",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for initialized template",
    ),
    agent_id: Optional[str] = typer.Option(
        None,
        "--id",
        help="Agent ID for the new agent",
    ),
):
    """
    Initialize a new agent from a template.

    Example:
        gl template init regulatory --id nfpa86-agent
    """
    try:
        console.print(f"\n[bold cyan]Initializing from template:[/bold cyan] {template_name}\n")

        if output_dir is None:
            output_dir = Path.cwd() / "agents" / (agent_id or "new-agent")

        # Create directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate template spec
        spec = {
            "metadata": {
                "id": agent_id or "new-agent",
                "name": f"New Agent from {template_name}",
                "version": "0.1.0",
                "type": "agent",
                "template": template_name,
            },
            "capabilities": [],
            "architecture": {
                "framework": "greenlang",
            },
        }

        spec_file = output_dir / "agent.yaml"
        spec_file.write_text(yaml.dump(spec, default_flow_style=False))

        print_success(f"Template initialized: {output_dir}")
        console.print(f"\nEdit the specification: [cyan]{spec_file}[/cyan]")
        console.print(f"Generate agent: [cyan]gl agent create {spec_file}[/cyan]\n")

    except Exception as e:
        print_error(f"Failed to initialize template: {str(e)}")
        raise typer.Exit(1)


@app.command()
def show(
    template_name: str = typer.Argument(
        ...,
        help="Template name to show",
    ),
):
    """
    Show details of a specific template.

    Example:
        gl template show regulatory
    """
    try:
        console.print(f"\n[bold cyan]Template:[/bold cyan] {template_name}\n")

        template_info = {
            "id": template_name,
            "name": f"{template_name.title()} Agent Template",
            "description": "A template for creating agents",
            "version": "1.0.0",
            "capabilities": ["validation", "testing", "deployment"],
        }

        console.print(yaml.dump(template_info, default_flow_style=False))

    except Exception as e:
        print_error(f"Failed to show template: {str(e)}")
        raise typer.Exit(1)
