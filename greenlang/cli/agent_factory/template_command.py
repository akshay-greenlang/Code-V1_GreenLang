# -*- coding: utf-8 -*-
"""
Template Management
===================

Manage agent templates for rapid development:
- List available templates
- Create custom templates
- Validate templates
- Version templates
- Share templates

Usage:
    gl agent template list
    gl agent template create my-template --from existing-agent
    gl agent template validate ./templates/my-template
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from pydantic import BaseModel, Field

# Create sub-app for template commands
template_app = typer.Typer(
    name="template",
    help="Agent template management",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# Template Models
# =============================================================================

class TemplateMetadata(BaseModel):
    """Template metadata."""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "GreenLang Team"
    category: str = "custom"
    tags: List[str] = Field(default_factory=list)
    created: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated: str = Field(default_factory=lambda: datetime.now().isoformat())


class TemplateFile(BaseModel):
    """Template file definition."""
    filename: str
    content: str
    description: str = ""


class Template(BaseModel):
    """Complete template definition."""
    metadata: TemplateMetadata
    files: List[TemplateFile] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    hooks: Dict[str, str] = Field(default_factory=dict)


# =============================================================================
# Built-in Templates
# =============================================================================

BUILTIN_TEMPLATES = {
    "calculator": {
        "id": "calculator",
        "name": "Emission Calculator",
        "description": "Template for emission calculation agents (Scope 1, 2, 3)",
        "version": "1.0.0",
        "author": "GreenLang Team",
        "category": "calculator",
        "tags": ["emissions", "ghg", "scope1", "scope2", "scope3"],
        "files": ["agent.py", "tools.py", "pack.yaml", "test_agent.py"],
        "variables": {
            "emission_type": "scope1",
            "calculation_method": "activity-based",
        },
    },
    "validator": {
        "id": "validator",
        "name": "Data Validator",
        "description": "Template for data validation agents",
        "version": "1.0.0",
        "author": "GreenLang Team",
        "category": "validator",
        "tags": ["validation", "data-quality", "compliance"],
        "files": ["agent.py", "tools.py", "pack.yaml", "test_agent.py"],
        "variables": {
            "validation_type": "schema",
            "strict_mode": True,
        },
    },
    "reporter": {
        "id": "reporter",
        "name": "Report Generator",
        "description": "Template for report generation agents",
        "version": "1.0.0",
        "author": "GreenLang Team",
        "category": "reporter",
        "tags": ["reports", "disclosure", "documentation"],
        "files": ["agent.py", "tools.py", "pack.yaml", "test_agent.py"],
        "variables": {
            "output_format": "pdf",
            "template_engine": "jinja2",
        },
    },
    "regulatory": {
        "id": "regulatory",
        "name": "Regulatory Compliance",
        "description": "Template for regulatory compliance agents",
        "version": "1.0.0",
        "author": "GreenLang Team",
        "category": "regulatory",
        "tags": ["compliance", "regulations", "csrd", "eudr", "sb253"],
        "files": ["agent.py", "tools.py", "pack.yaml", "test_agent.py"],
        "variables": {
            "regulation": "CSRD",
            "jurisdiction": "EU",
        },
    },
    "classifier": {
        "id": "classifier",
        "name": "Transaction Classifier",
        "description": "Template for AI-powered classification agents",
        "version": "1.0.0",
        "author": "GreenLang Team",
        "category": "classifier",
        "tags": ["classification", "ai", "ml", "categorization"],
        "files": ["agent.py", "tools.py", "pack.yaml", "test_agent.py"],
        "variables": {
            "model_type": "llm",
            "confidence_threshold": 0.8,
        },
    },
    "aggregator": {
        "id": "aggregator",
        "name": "Data Aggregator",
        "description": "Template for data aggregation agents",
        "version": "1.0.0",
        "author": "GreenLang Team",
        "category": "aggregator",
        "tags": ["aggregation", "consolidation", "summary"],
        "files": ["agent.py", "tools.py", "pack.yaml", "test_agent.py"],
        "variables": {
            "aggregation_level": "organization",
            "time_period": "annual",
        },
    },
    "custom": {
        "id": "custom",
        "name": "Custom Agent",
        "description": "Minimal starter template for custom agents",
        "version": "1.0.0",
        "author": "GreenLang Team",
        "category": "custom",
        "tags": ["starter", "minimal", "custom"],
        "files": ["agent.py", "tools.py", "pack.yaml", "test_agent.py"],
        "variables": {},
    },
}


# =============================================================================
# Template Commands
# =============================================================================

@template_app.command("list")
def template_list_command(
    category: Optional[str] = typer.Option(
        None,
        "--category", "-c",
        help="Filter by category",
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag", "-t",
        help="Filter by tag",
    ),
    include_custom: bool = typer.Option(
        True,
        "--include-custom/--builtin-only",
        help="Include custom templates",
    ),
    format: str = typer.Option(
        "table",
        "--format", "-f",
        help="Output format: table, json",
    ),
):
    """
    List available agent templates.

    Example:
        gl agent template list
        gl agent template list --category regulatory
        gl agent template list --tag emissions
    """
    console.print(Panel(
        "[bold cyan]Available Agent Templates[/bold cyan]",
        border_style="cyan"
    ))

    templates = list(BUILTIN_TEMPLATES.values())

    # Load custom templates
    if include_custom:
        custom_templates = _load_custom_templates()
        templates.extend(custom_templates)

    # Filter by category
    if category:
        templates = [t for t in templates if t.get("category") == category]

    # Filter by tag
    if tag:
        templates = [t for t in templates if tag in t.get("tags", [])]

    if not templates:
        console.print("[yellow]No templates found matching criteria[/yellow]")
        return

    if format == "json":
        console.print_json(data=templates)
        return

    # Table display
    table = Table(title=f"Templates ({len(templates)})")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Category")
    table.add_column("Version")
    table.add_column("Description", width=40)

    for tmpl in templates:
        desc = tmpl.get("description", "")
        if len(desc) > 40:
            desc = desc[:37] + "..."
        table.add_row(
            tmpl["id"],
            tmpl["name"],
            tmpl.get("category", "custom"),
            tmpl.get("version", "1.0.0"),
            desc,
        )

    console.print(table)

    console.print("\n[bold]Create an agent:[/bold]")
    console.print("  gl agent new <template-id> --name <agent-name>")


@template_app.command("info")
def template_info_command(
    template_id: str = typer.Argument(
        ...,
        help="Template ID to show info for",
    ),
):
    """
    Show detailed information about a template.

    Example:
        gl agent template info calculator
        gl agent template info regulatory
    """
    # Find template
    tmpl = BUILTIN_TEMPLATES.get(template_id)
    if not tmpl:
        # Check custom templates
        custom_templates = _load_custom_templates()
        for ct in custom_templates:
            if ct["id"] == template_id:
                tmpl = ct
                break

    if not tmpl:
        console.print(f"[red]Template not found: {template_id}[/red]")
        console.print(f"Available templates: {', '.join(BUILTIN_TEMPLATES.keys())}")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold cyan]{tmpl['name']}[/bold cyan]\n"
        f"ID: {tmpl['id']}",
        border_style="cyan"
    ))

    # Info table
    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Description", tmpl.get("description", "N/A"))
    table.add_row("Category", tmpl.get("category", "custom"))
    table.add_row("Version", tmpl.get("version", "1.0.0"))
    table.add_row("Author", tmpl.get("author", "Unknown"))
    table.add_row("Tags", ", ".join(tmpl.get("tags", [])))
    table.add_row("Files", ", ".join(tmpl.get("files", [])))

    console.print(table)

    # Variables
    if tmpl.get("variables"):
        console.print("\n[bold]Variables:[/bold]")
        for var, default in tmpl["variables"].items():
            console.print(f"  - {var}: {default}")

    console.print("\n[bold]Create an agent:[/bold]")
    console.print(f"  gl agent new {template_id} --name my-agent")


@template_app.command("create")
def template_create_command(
    template_id: str = typer.Argument(
        ...,
        help="ID for the new template",
    ),
    from_agent: Optional[Path] = typer.Option(
        None,
        "--from",
        help="Create template from existing agent",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name", "-n",
        help="Template display name",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description", "-d",
        help="Template description",
    ),
    category: str = typer.Option(
        "custom",
        "--category", "-c",
        help="Template category",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory",
    ),
):
    """
    Create a new template.

    Example:
        gl agent template create my-template --name "My Template"
        gl agent template create my-template --from ./agents/carbon
    """
    console.print(Panel(
        f"[bold cyan]Creating Template: {template_id}[/bold cyan]",
        border_style="cyan"
    ))

    # Determine output directory
    if output is None:
        output = Path.home() / ".greenlang" / "templates" / template_id
    output.mkdir(parents=True, exist_ok=True)

    # Set defaults
    if name is None:
        name = template_id.replace("-", " ").replace("_", " ").title()
    if description is None:
        description = f"Custom template: {name}"

    # Create metadata
    metadata = TemplateMetadata(
        id=template_id,
        name=name,
        description=description,
        category=category,
    )

    if from_agent:
        # Create template from existing agent
        _create_template_from_agent(from_agent, output, metadata)
    else:
        # Create minimal template
        _create_minimal_template(output, metadata)

    console.print(f"\n[green]Template created at:[/green] {output}")
    console.print(f"\n[bold]Use template:[/bold]")
    console.print(f"  gl agent new {template_id} --name my-agent")


@template_app.command("validate")
def template_validate_command(
    template_path: Path = typer.Argument(
        ...,
        help="Path to template directory",
        exists=True,
    ),
):
    """
    Validate a template.

    Example:
        gl agent template validate ./templates/my-template
    """
    console.print(Panel(
        "[bold cyan]Template Validation[/bold cyan]\n"
        f"Path: {template_path}",
        border_style="cyan"
    ))

    errors = []
    warnings = []

    # Check for template.yaml
    template_yaml = template_path / "template.yaml"
    if not template_yaml.exists():
        errors.append("Missing template.yaml")
    else:
        try:
            import yaml
            with open(template_yaml) as f:
                config = yaml.safe_load(f)
            if not config.get("id"):
                errors.append("template.yaml missing 'id' field")
            if not config.get("name"):
                errors.append("template.yaml missing 'name' field")
        except Exception as e:
            errors.append(f"Invalid template.yaml: {str(e)}")

    # Check for required files
    required_files = ["agent.py.template", "pack.yaml.template"]
    for req in required_files:
        if not (template_path / req).exists():
            warnings.append(f"Missing recommended file: {req}")

    # Display results
    if errors:
        console.print("\n[red]Validation Failed[/red]")
        for err in errors:
            console.print(f"  [red]x[/red] {err}")
    else:
        console.print("\n[green]Validation Passed[/green]")

    if warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warn in warnings:
            console.print(f"  [yellow]![/yellow] {warn}")


@template_app.command("export")
def template_export_command(
    template_id: str = typer.Argument(
        ...,
        help="Template ID to export",
    ),
    output: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Output file path (.tar.gz or .zip)",
    ),
):
    """
    Export a template for sharing.

    Example:
        gl agent template export my-template --output my-template.tar.gz
    """
    console.print(f"[cyan]Exporting template: {template_id}...[/cyan]")

    # Find template
    template_path = Path.home() / ".greenlang" / "templates" / template_id
    if not template_path.exists():
        console.print(f"[red]Template not found: {template_id}[/red]")
        raise typer.Exit(1)

    # Create archive
    if str(output).endswith(".zip"):
        shutil.make_archive(str(output).replace(".zip", ""), "zip", template_path)
    else:
        shutil.make_archive(str(output).replace(".tar.gz", ""), "gztar", template_path)

    console.print(f"[green]Template exported to:[/green] {output}")


@template_app.command("import")
def template_import_command(
    archive: Path = typer.Argument(
        ...,
        help="Path to template archive",
        exists=True,
    ),
    template_id: Optional[str] = typer.Option(
        None,
        "--id",
        help="Override template ID",
    ),
):
    """
    Import a template from archive.

    Example:
        gl agent template import my-template.tar.gz
    """
    console.print(f"[cyan]Importing template from: {archive}...[/cyan]")

    # Determine template ID
    if template_id is None:
        template_id = archive.stem.replace(".tar", "")

    # Extract to templates directory
    templates_dir = Path.home() / ".greenlang" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    output_dir = templates_dir / template_id

    # Extract archive
    shutil.unpack_archive(archive, output_dir)

    console.print(f"[green]Template imported:[/green] {template_id}")
    console.print(f"[bold]Use template:[/bold]")
    console.print(f"  gl agent new {template_id} --name my-agent")


@template_app.command("delete")
def template_delete_command(
    template_id: str = typer.Argument(
        ...,
        help="Template ID to delete",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Skip confirmation",
    ),
):
    """
    Delete a custom template.

    Example:
        gl agent template delete my-template
    """
    # Prevent deletion of built-in templates
    if template_id in BUILTIN_TEMPLATES:
        console.print(f"[red]Cannot delete built-in template: {template_id}[/red]")
        raise typer.Exit(1)

    template_path = Path.home() / ".greenlang" / "templates" / template_id
    if not template_path.exists():
        console.print(f"[red]Template not found: {template_id}[/red]")
        raise typer.Exit(1)

    if not force and not Confirm.ask(f"Delete template '{template_id}'?"):
        console.print("[yellow]Cancelled[/yellow]")
        raise typer.Exit(0)

    shutil.rmtree(template_path)
    console.print(f"[green]Template deleted:[/green] {template_id}")


# =============================================================================
# Helper Functions
# =============================================================================

def _load_custom_templates() -> List[Dict[str, Any]]:
    """Load custom templates from user directory."""
    templates = []
    templates_dir = Path.home() / ".greenlang" / "templates"

    if not templates_dir.exists():
        return templates

    for template_dir in templates_dir.iterdir():
        if not template_dir.is_dir():
            continue

        template_yaml = template_dir / "template.yaml"
        if template_yaml.exists():
            try:
                import yaml
                with open(template_yaml) as f:
                    config = yaml.safe_load(f)
                    config["id"] = template_dir.name
                    templates.append(config)
            except Exception as e:
                logger.warning(f"Failed to load template {template_dir.name}: {e}")

    return templates


def _create_template_from_agent(agent_path: Path, output: Path, metadata: TemplateMetadata) -> None:
    """Create template from existing agent."""
    console.print(f"[cyan]Creating template from agent: {agent_path}[/cyan]")

    # Copy and templatize files
    for py_file in agent_path.glob("*.py"):
        content = py_file.read_text()
        # Replace specific values with template variables
        content = _templatize_content(content, agent_path.name)
        (output / f"{py_file.name}.template").write_text(content)

    # Copy pack.yaml if exists
    pack_yaml = agent_path / "pack.yaml"
    if pack_yaml.exists():
        content = pack_yaml.read_text()
        content = _templatize_content(content, agent_path.name)
        (output / "pack.yaml.template").write_text(content)

    # Create template.yaml
    _create_template_yaml(output, metadata)


def _create_minimal_template(output: Path, metadata: TemplateMetadata) -> None:
    """Create minimal template structure."""
    # Create template.yaml
    _create_template_yaml(output, metadata)

    # Create agent.py.template
    agent_template = '''# -*- coding: utf-8 -*-
"""
{{ agent_display_name }}
{{ "=" * agent_display_name|length }}

{{ description }}
"""

import hashlib
import logging
from typing import Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

__version__ = "{{ version }}"
__author__ = "{{ author }}"

logger = logging.getLogger(__name__)


class AgentInput(BaseModel):
    """Input data model."""
    data: Any = Field(..., description="Input data")


class AgentOutput(BaseModel):
    """Output data model."""
    result: Any = Field(..., description="Processing result")
    provenance_hash: str = Field(..., description="SHA-256 hash")
    processing_time_ms: float = Field(..., description="Processing duration")


class {{ agent_class_name }}Agent:
    """{{ agent_display_name }} implementation."""

    def __init__(self):
        self.name = "{{ agent_display_name }}"
        self.version = "{{ version }}"

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """Execute agent workflow."""
        start = datetime.now()

        # Process (implement your logic here)
        result = self._process(input_data)

        # Calculate provenance
        provenance_hash = hashlib.sha256(
            str(input_data.model_dump()).encode()
        ).hexdigest()

        duration = (datetime.now() - start).total_seconds() * 1000

        return AgentOutput(
            result=result,
            provenance_hash=provenance_hash,
            processing_time_ms=duration,
        )

    def _process(self, input_data: AgentInput) -> Any:
        """Core processing logic."""
        raise NotImplementedError()


def create_agent():
    """Factory function."""
    return {{ agent_class_name }}Agent()
'''
    (output / "agent.py.template").write_text(agent_template)

    # Create pack.yaml.template
    pack_template = '''# GreenLang AgentSpec
id: {{ agent_id }}
name: {{ agent_display_name }}
version: {{ version }}
license: Apache-2.0

metadata:
  author: {{ author }}
  category: {{ category }}
  description: {{ description }}

tools:
  - name: process
    type: deterministic

inputs:
  - name: data
    type: any
    required: true

outputs:
  - name: result
    type: any
  - name: provenance_hash
    type: string

tests:
  golden:
    - name: basic_test
      input:
        data: "test"
      expected_output:
        status: success
'''
    (output / "pack.yaml.template").write_text(pack_template)


def _create_template_yaml(output: Path, metadata: TemplateMetadata) -> None:
    """Create template.yaml file."""
    import yaml

    config = {
        "id": metadata.id,
        "name": metadata.name,
        "description": metadata.description,
        "version": metadata.version,
        "author": metadata.author,
        "category": metadata.category,
        "tags": metadata.tags,
        "created": metadata.created,
        "variables": {
            "agent_id": "{{ name }}",
            "agent_class_name": "{{ name|replace('-', '')|replace('_', '')|title }}",
            "agent_display_name": "{{ name|replace('-', ' ')|replace('_', ' ')|title }}",
            "version": "0.1.0",
            "author": "GreenLang Team",
            "description": "Custom agent description",
            "category": metadata.category,
        },
    }

    with open(output / "template.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def _templatize_content(content: str, agent_name: str) -> str:
    """Replace agent-specific values with template variables."""
    # Replace agent name variations
    content = content.replace(agent_name, "{{ agent_id }}")
    content = content.replace(
        agent_name.replace("-", "_"),
        "{{ agent_id|replace('-', '_') }}"
    )
    content = content.replace(
        agent_name.replace("-", " ").title(),
        "{{ agent_display_name }}"
    )
    return content
