# -*- coding: utf-8 -*-
"""
Agent Creation Wizard
=====================

Interactive agent creation from templates with:
- Template selection and customization
- Parameter prompting
- Blueprint generation
- Code scaffolding
- Test generation
- Documentation generation
- Git initialization

Usage:
    gl agent create calculator --name carbon-calc --pack eudr
    gl agent create regulatory --name csrd-validator --interactive
"""

import os
import sys
import json
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from pydantic import BaseModel, Field

# Create sub-app for create commands
create_app = typer.Typer(
    name="create",
    help="Agent creation wizard with templates",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# Templates Configuration
# =============================================================================

AGENT_TEMPLATES = {
    "calculator": {
        "name": "Emission Calculator",
        "description": "Agent for calculating emissions (Scope 1, 2, 3)",
        "category": "calculator",
        "tools": ["calculate_emissions", "validate_input", "lookup_factor"],
        "inputs": ["activity_data", "emission_factors", "methodology"],
        "outputs": ["emissions_value", "unit", "provenance_hash"],
        "golden_tests": 25,
        "color": "green",
    },
    "validator": {
        "name": "Data Validator",
        "description": "Agent for validating sustainability data",
        "category": "validator",
        "tools": ["validate_schema", "check_completeness", "verify_ranges"],
        "inputs": ["data_record", "validation_rules", "threshold_config"],
        "outputs": ["is_valid", "errors", "warnings", "confidence"],
        "golden_tests": 30,
        "color": "blue",
    },
    "reporter": {
        "name": "Report Generator",
        "description": "Agent for generating compliance reports",
        "category": "reporter",
        "tools": ["generate_report", "format_output", "aggregate_data"],
        "inputs": ["report_data", "template", "format_options"],
        "outputs": ["report_content", "metadata", "attachments"],
        "golden_tests": 15,
        "color": "cyan",
    },
    "regulatory": {
        "name": "Regulatory Compliance",
        "description": "Agent for checking regulatory compliance",
        "category": "regulatory",
        "tools": ["check_compliance", "map_requirements", "generate_evidence"],
        "inputs": ["entity_data", "regulation_id", "reporting_period"],
        "outputs": ["compliance_status", "gaps", "recommendations"],
        "golden_tests": 40,
        "color": "yellow",
    },
    "custom": {
        "name": "Custom Agent",
        "description": "Minimal starter template for custom agents",
        "category": "custom",
        "tools": ["process", "validate", "respond"],
        "inputs": ["input_data"],
        "outputs": ["result", "provenance_hash"],
        "golden_tests": 10,
        "color": "magenta",
    },
}


# =============================================================================
# Pydantic Models for Agent Configuration
# =============================================================================

class AgentConfig(BaseModel):
    """Configuration for agent generation."""

    name: str = Field(..., description="Agent name (kebab-case)")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Agent description")
    template: str = Field(..., description="Template type")
    pack: Optional[str] = Field(None, description="Pack to add to")
    version: str = Field("0.1.0", description="Initial version")
    author: str = Field("GreenLang Team", description="Author name")
    license: str = Field("Apache-2.0", description="License")
    category: str = Field(..., description="Agent category")
    tools: List[str] = Field(default_factory=list, description="Tools")
    inputs: List[str] = Field(default_factory=list, description="Input fields")
    outputs: List[str] = Field(default_factory=list, description="Output fields")
    generate_tests: bool = Field(True, description="Generate test files")
    generate_docs: bool = Field(True, description="Generate documentation")
    git_init: bool = Field(True, description="Initialize git repository")


# =============================================================================
# Create Commands
# =============================================================================

@create_app.command("from-template")
def create_from_template(
    template: str = typer.Argument(
        ...,
        help="Template type: calculator, validator, reporter, regulatory, custom",
    ),
    name: str = typer.Option(
        ...,
        "--name", "-n",
        help="Agent name in kebab-case (e.g., carbon-calculator)",
    ),
    pack: Optional[str] = typer.Option(
        None,
        "--pack", "-p",
        help="Pack to add agent to",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description", "-d",
        help="Agent description",
    ),
    author: str = typer.Option(
        "GreenLang Team",
        "--author",
        help="Author name",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing files",
    ),
    no_tests: bool = typer.Option(
        False,
        "--no-tests",
        help="Skip test generation",
    ),
    no_docs: bool = typer.Option(
        False,
        "--no-docs",
        help="Skip documentation generation",
    ),
    no_git: bool = typer.Option(
        False,
        "--no-git",
        help="Skip git initialization",
    ),
):
    """
    Create a new agent from a template.

    Example:
        gl agent create from-template calculator --name carbon-calc
        gl agent create from-template regulatory --name eudr-validator --pack eudr
    """
    create_agent_from_template(
        template=template,
        name=name,
        pack=pack,
        output_dir=output_dir,
        description=description,
        author=author,
        force=force,
        generate_tests=not no_tests,
        generate_docs=not no_docs,
        git_init=not no_git,
    )


@create_app.command("wizard")
def create_wizard():
    """
    Interactive wizard for creating a new agent.

    Guides you through:
    - Template selection
    - Naming and description
    - Tool configuration
    - Input/Output definition
    - Test and docs generation
    """
    console.print(Panel(
        "[bold cyan]GreenLang Agent Creation Wizard[/bold cyan]\n"
        "Create a production-ready agent in minutes",
        border_style="cyan"
    ))

    # Step 1: Template selection
    console.print("\n[bold]Step 1: Select Template[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Template", style="green")
    table.add_column("Description")
    table.add_column("Category")
    table.add_column("Tools")

    for key, tmpl in AGENT_TEMPLATES.items():
        table.add_row(
            key,
            tmpl["description"],
            tmpl["category"],
            str(len(tmpl["tools"])),
        )

    console.print(table)

    template = Prompt.ask(
        "\nChoose template",
        choices=list(AGENT_TEMPLATES.keys()),
        default="calculator",
    )

    # Step 2: Naming
    console.print("\n[bold]Step 2: Agent Identity[/bold]")
    name = Prompt.ask("Agent name (kebab-case)", default="my-agent")
    display_name = Prompt.ask(
        "Display name",
        default=name.replace("-", " ").replace("_", " ").title()
    )
    description = Prompt.ask(
        "Description",
        default=AGENT_TEMPLATES[template]["description"]
    )

    # Step 3: Pack association
    console.print("\n[bold]Step 3: Pack Association[/bold]")
    use_pack = Confirm.ask("Add to an existing pack?", default=False)
    pack = None
    if use_pack:
        pack = Prompt.ask("Pack name (e.g., eudr, sb253)")

    # Step 4: Configuration
    console.print("\n[bold]Step 4: Configuration[/bold]")
    author = Prompt.ask("Author", default="GreenLang Team")
    version = Prompt.ask("Initial version", default="0.1.0")
    license_type = Prompt.ask("License", default="Apache-2.0")

    # Step 5: Generation options
    console.print("\n[bold]Step 5: Generation Options[/bold]")
    generate_tests = Confirm.ask("Generate test files?", default=True)
    generate_docs = Confirm.ask("Generate documentation?", default=True)
    git_init = Confirm.ask("Initialize git repository?", default=True)

    # Step 6: Confirmation
    console.print("\n[bold]Step 6: Review Configuration[/bold]")
    config_table = Table(show_header=False, border_style="blue")
    config_table.add_column("Property", style="cyan")
    config_table.add_column("Value")

    config_table.add_row("Template", template)
    config_table.add_row("Name", name)
    config_table.add_row("Display Name", display_name)
    config_table.add_row("Description", description)
    config_table.add_row("Pack", pack or "(none)")
    config_table.add_row("Author", author)
    config_table.add_row("Version", version)
    config_table.add_row("License", license_type)
    config_table.add_row("Generate Tests", str(generate_tests))
    config_table.add_row("Generate Docs", str(generate_docs))
    config_table.add_row("Git Init", str(git_init))

    console.print(config_table)

    if not Confirm.ask("\nProceed with agent creation?", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        raise typer.Exit(0)

    # Create the agent
    create_agent_from_template(
        template=template,
        name=name,
        pack=pack,
        description=description,
        author=author,
        generate_tests=generate_tests,
        generate_docs=generate_docs,
        git_init=git_init,
    )


@create_app.command("from-spec")
def create_from_spec(
    spec_path: Path = typer.Argument(
        ...,
        help="Path to AgentSpec YAML file",
        exists=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing files",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed output",
    ),
):
    """
    Create an agent from an existing AgentSpec YAML file.

    Parses the spec and generates:
    - Agent implementation
    - Tool implementations
    - Test files
    - Documentation

    Example:
        gl agent create from-spec pack.yaml
        gl agent create from-spec spec.yaml --output ./generated
    """
    console.print(Panel(
        "[bold cyan]Creating Agent from Spec[/bold cyan]\n"
        f"Source: {spec_path}",
        border_style="cyan"
    ))

    try:
        import yaml

        with open(spec_path, "r") as f:
            spec = yaml.safe_load(f)

        # Extract info from spec
        agent_id = spec.get("id", spec.get("name", "unknown"))
        agent_name = spec.get("name", agent_id)

        console.print(f"\n[bold]Agent:[/bold] {agent_name}")
        console.print(f"[bold]ID:[/bold] {agent_id}")

        if output_dir is None:
            output_dir = Path("./generated") / agent_id.replace("/", "_")

        # Check for existing
        if output_dir.exists() and not force:
            if not Confirm.ask(f"\nOutput directory exists. Overwrite?"):
                console.print("[yellow]Cancelled[/yellow]")
                raise typer.Exit(0)

        # Generate with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating agent...", total=5)

            # Create directory structure
            output_dir.mkdir(parents=True, exist_ok=True)
            progress.update(task, advance=1, description="Created directories")

            # Generate agent code
            agent_code = _generate_agent_code_from_spec(spec)
            (output_dir / "agent.py").write_text(agent_code)
            progress.update(task, advance=1, description="Generated agent.py")

            # Generate tools
            tools_code = _generate_tools_code_from_spec(spec)
            (output_dir / "tools.py").write_text(tools_code)
            progress.update(task, advance=1, description="Generated tools.py")

            # Generate tests
            test_code = _generate_test_code_from_spec(spec)
            tests_dir = output_dir / "tests"
            tests_dir.mkdir(exist_ok=True)
            (tests_dir / "test_agent.py").write_text(test_code)
            progress.update(task, advance=1, description="Generated tests")

            # Generate __init__.py
            init_code = _generate_init_code(spec)
            (output_dir / "__init__.py").write_text(init_code)
            progress.update(task, advance=1, description="Generated __init__.py")

        console.print(f"\n[bold green]Agent generated successfully![/bold green]")
        console.print(f"[bold]Output:[/bold] {output_dir}")

        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  1. cd {output_dir}")
        console.print("  2. gl agent test .")
        console.print("  3. gl agent certify .")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


# =============================================================================
# Core Implementation Functions
# =============================================================================

def create_agent_from_template(
    template: str,
    name: str,
    pack: Optional[str] = None,
    output_dir: Optional[Path] = None,
    description: Optional[str] = None,
    author: str = "GreenLang Team",
    force: bool = False,
    generate_tests: bool = True,
    generate_docs: bool = True,
    git_init: bool = True,
    interactive: bool = False,
) -> Path:
    """
    Create a new agent from a template.

    Args:
        template: Template type (calculator, validator, etc.)
        name: Agent name in kebab-case
        pack: Optional pack to add agent to
        output_dir: Output directory (default: ./agents/<name>)
        description: Agent description
        author: Author name
        force: Overwrite existing files
        generate_tests: Generate test files
        generate_docs: Generate documentation
        git_init: Initialize git repository

    Returns:
        Path to created agent directory
    """
    console.print(Panel(
        f"[bold cyan]Creating Agent: {name}[/bold cyan]\n"
        f"Template: {template}",
        border_style="cyan"
    ))

    # Validate template
    if template not in AGENT_TEMPLATES:
        console.print(f"[red]Unknown template: {template}[/red]")
        console.print(f"Available templates: {', '.join(AGENT_TEMPLATES.keys())}")
        raise typer.Exit(1)

    tmpl = AGENT_TEMPLATES[template]

    # Determine output directory
    if output_dir is None:
        if pack:
            output_dir = Path(f"./packs/{pack}/agents/{name}")
        else:
            output_dir = Path(f"./agents/{name}")

    # Check if exists
    if output_dir.exists() and not force:
        console.print(f"[yellow]Directory exists: {output_dir}[/yellow]")
        if not Confirm.ask("Overwrite?"):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    # Set description
    if description is None:
        description = tmpl["description"]

    # Generate agent config
    config = AgentConfig(
        name=name,
        display_name=name.replace("-", " ").replace("_", " ").title(),
        description=description,
        template=template,
        pack=pack,
        author=author,
        category=tmpl["category"],
        tools=tmpl["tools"],
        inputs=tmpl["inputs"],
        outputs=tmpl["outputs"],
        generate_tests=generate_tests,
        generate_docs=generate_docs,
        git_init=git_init,
    )

    # Generate with progress
    files_created = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        total_steps = 7 if git_init else 6
        task = progress.add_task("Creating agent...", total=total_steps)

        # 1. Create directory structure
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tests").mkdir(exist_ok=True)
        progress.update(task, advance=1, description="Created directories")

        # 2. Generate agent.py
        agent_code = _generate_agent_code(config, tmpl)
        agent_file = output_dir / "agent.py"
        agent_file.write_text(agent_code)
        files_created.append(agent_file)
        progress.update(task, advance=1, description="Generated agent.py")

        # 3. Generate tools.py
        tools_code = _generate_tools_code(config, tmpl)
        tools_file = output_dir / "tools.py"
        tools_file.write_text(tools_code)
        files_created.append(tools_file)
        progress.update(task, advance=1, description="Generated tools.py")

        # 4. Generate __init__.py
        init_code = _generate_init_code_from_config(config)
        init_file = output_dir / "__init__.py"
        init_file.write_text(init_code)
        files_created.append(init_file)
        progress.update(task, advance=1, description="Generated __init__.py")

        # 5. Generate pack.yaml
        pack_yaml = _generate_pack_yaml(config, tmpl)
        pack_file = output_dir / "pack.yaml"
        pack_file.write_text(pack_yaml)
        files_created.append(pack_file)
        progress.update(task, advance=1, description="Generated pack.yaml")

        # 6. Generate tests (if requested)
        if generate_tests:
            test_code = _generate_test_code(config, tmpl)
            test_file = output_dir / "tests" / "test_agent.py"
            test_file.write_text(test_code)
            files_created.append(test_file)

            conftest_code = _generate_conftest()
            conftest_file = output_dir / "tests" / "conftest.py"
            conftest_file.write_text(conftest_code)
            files_created.append(conftest_file)

        # Generate docs (if requested)
        if generate_docs:
            readme = _generate_readme(config, tmpl)
            readme_file = output_dir / "README.md"
            readme_file.write_text(readme)
            files_created.append(readme_file)

        progress.update(task, advance=1, description="Generated tests and docs")

        # 7. Initialize git (if requested)
        if git_init:
            _init_git_repo(output_dir)
            progress.update(task, advance=1, description="Initialized git")

    # Print summary
    console.print("\n[bold green]Agent created successfully![/bold green]\n")

    # Files table
    file_table = Table(title="Generated Files")
    file_table.add_column("File", style="cyan")
    file_table.add_column("Lines", justify="right", style="green")
    file_table.add_column("Purpose")

    file_purposes = {
        "agent.py": "Main agent implementation",
        "tools.py": "Tool implementations",
        "__init__.py": "Package initialization",
        "pack.yaml": "AgentSpec definition",
        "test_agent.py": "Unit and golden tests",
        "conftest.py": "Test fixtures",
        "README.md": "Documentation",
    }

    for f in files_created:
        content = f.read_text()
        lines = content.count("\n") + 1
        purpose = file_purposes.get(f.name, "Generated file")
        file_table.add_row(str(f.relative_to(output_dir)), str(lines), purpose)

    console.print(file_table)

    # Summary
    console.print(f"\n[bold]Output:[/bold] {output_dir}")
    console.print(f"[bold]Template:[/bold] {template}")
    console.print(f"[bold]Tools:[/bold] {len(config.tools)}")
    console.print(f"[bold]Golden Tests:[/bold] {tmpl['golden_tests']}")

    # Next steps
    console.print("\n[bold]Next steps:[/bold]")
    console.print(f"  1. cd {output_dir}")
    console.print("  2. Implement tool logic in tools.py")
    console.print("  3. Add golden test cases in tests/")
    console.print("  4. gl agent validate pack.yaml")
    console.print("  5. gl agent test .")
    console.print("  6. gl agent certify .")

    return output_dir


# =============================================================================
# Code Generation Functions
# =============================================================================

def _generate_agent_code(config: AgentConfig, tmpl: dict) -> str:
    """Generate agent.py code."""
    tools_import = ", ".join(config.tools)
    module_name = config.name.replace("-", "_")

    return f'''# -*- coding: utf-8 -*-
"""
{config.display_name}
{"=" * len(config.display_name)}

{config.description}

This agent is auto-generated by GreenLang Agent Factory.
Template: {config.template}
Category: {config.category}

Usage:
    >>> agent = {module_name.title().replace("_", "")}Agent()
    >>> result = await agent.run(input_data)
"""

import hashlib
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .tools import {tools_import}

__version__ = "{config.version}"
__author__ = "{config.author}"

logger = logging.getLogger(__name__)


# =============================================================================
# Input/Output Models
# =============================================================================

class AgentInput(BaseModel):
    """Input data model for {config.display_name}."""
{_generate_input_fields(config.inputs)}


class AgentOutput(BaseModel):
    """Output data model for {config.display_name}."""
{_generate_output_fields(config.outputs)}
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., description="Processing duration")
    timestamp: str = Field(..., description="ISO timestamp")


# =============================================================================
# Agent Implementation
# =============================================================================

class {module_name.title().replace("_", "")}Agent:
    """
    {config.display_name} implementation.

    This agent provides {config.category} capabilities for GreenLang.

    Attributes:
        name: Agent display name
        version: Agent version
        category: Agent category

    Example:
        >>> agent = {module_name.title().replace("_", "")}Agent()
        >>> input_data = AgentInput(...)
        >>> result = await agent.run(input_data)
        >>> assert result.provenance_hash is not None
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize agent with optional configuration."""
        self.name = "{config.display_name}"
        self.version = "{config.version}"
        self.category = "{config.category}"
        self.config = config or {{}}
        logger.info(f"Initialized {{self.name}} v{{self.version}}")

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute the agent's main workflow.

        Args:
            input_data: Validated input data

        Returns:
            AgentOutput with results and provenance

        Raises:
            ValueError: If input validation fails
            ProcessingError: If processing fails
        """
        start_time = datetime.now()
        logger.info(f"Starting {{self.name}} execution")

        try:
            # Step 1: Validate input
            self._validate_input(input_data)

            # Step 2: Execute tools (zero-hallucination - deterministic only)
            result = await self._execute_workflow(input_data)

            # Step 3: Calculate provenance hash
            provenance_hash = self._calculate_provenance(input_data, result)

            # Step 4: Create output
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            output = AgentOutput(
                **result,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat(),
            )

            logger.info(f"Completed {{self.name}} in {{processing_time:.2f}}ms")
            return output

        except Exception as e:
            logger.error(f"{{self.name}} execution failed: {{str(e)}}", exc_info=True)
            raise

    def _validate_input(self, input_data: AgentInput) -> None:
        """Validate input data meets requirements."""
        # Pydantic handles basic validation
        # Add custom validation logic here
        pass

    async def _execute_workflow(self, input_data: AgentInput) -> Dict[str, Any]:
        """
        Execute the main workflow.

        IMPORTANT: This method uses DETERMINISTIC processing only.
        No LLM calls allowed for numeric calculations.
        """
        # TODO: Implement workflow using tools
        # Example:
        # result = await {config.tools[0]}(input_data)
        # return result

        raise NotImplementedError("Implement workflow logic")

    def _calculate_provenance(self, input_data: AgentInput, result: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for complete audit trail."""
        provenance_str = f"{{input_data.model_dump_json()}}{{result}}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return agent metadata."""
        return {{
            "name": self.name,
            "version": self.version,
            "category": self.category,
            "tools": {config.tools},
        }}


# =============================================================================
# Factory Function
# =============================================================================

def create_agent(config: Optional[Dict[str, Any]] = None) -> {module_name.title().replace("_", "")}Agent:
    """
    Factory function to create agent instance.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured agent instance
    """
    return {module_name.title().replace("_", "")}Agent(config)
'''


def _generate_tools_code(config: AgentConfig, tmpl: dict) -> str:
    """Generate tools.py code."""
    tools_code = f'''# -*- coding: utf-8 -*-
"""
{config.display_name} - Tool Implementations
{"=" * (len(config.display_name) + 23)}

Deterministic tool implementations for zero-hallucination processing.

IMPORTANT: These tools must be DETERMINISTIC:
- No LLM calls for numeric calculations
- Database lookups only
- Pure Python arithmetic
- YAML/JSON formula evaluation

Usage:
    >>> result = await calculate_emissions(activity_data, factors)
"""

import hashlib
import logging
from typing import Dict, List, Any, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


'''

    for tool in config.tools:
        tools_code += f'''
async def {tool}(
    input_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    {tool.replace("_", " ").title()} tool implementation.

    Args:
        input_data: Input data dictionary
        config: Optional configuration

    Returns:
        Result dictionary

    Note:
        This tool must be DETERMINISTIC - no LLM calls for calculations.
    """
    logger.info(f"Executing {tool}")

    # TODO: Implement deterministic logic
    # Example:
    # - Database lookup: value = db.lookup(key)
    # - Arithmetic: result = a * b + c
    # - Formula evaluation: result = evaluate_formula(formula_id, inputs)

    raise NotImplementedError("Implement {tool} logic")

'''

    return tools_code


def _generate_init_code_from_config(config: AgentConfig) -> str:
    """Generate __init__.py code from config."""
    module_name = config.name.replace("-", "_")
    class_name = module_name.title().replace("_", "")

    return f'''# -*- coding: utf-8 -*-
"""
{config.display_name}
{"=" * len(config.display_name)}

{config.description}
"""

__version__ = "{config.version}"
__author__ = "{config.author}"
__license__ = "{config.license}"

from .agent import {class_name}Agent, AgentInput, AgentOutput, create_agent
from .tools import {", ".join(config.tools)}

__all__ = [
    "{class_name}Agent",
    "AgentInput",
    "AgentOutput",
    "create_agent",
    {", ".join(f'"{t}"' for t in config.tools)},
]
'''


def _generate_pack_yaml(config: AgentConfig, tmpl: dict) -> str:
    """Generate pack.yaml AgentSpec."""
    return f'''# GreenLang AgentSpec
# Generated by Agent Factory CLI
# Template: {config.template}

id: {config.name}
name: {config.display_name}
version: {config.version}
license: {config.license}

metadata:
  author: {config.author}
  category: {config.category}
  description: |
    {config.description}
  created: {datetime.now().strftime("%Y-%m-%d")}
  pack: {config.pack or "standalone"}

# Tool Definitions
tools:
{_generate_tools_yaml(config.tools)}

# Input Schema
inputs:
{_generate_io_yaml(config.inputs, "input")}

# Output Schema
outputs:
{_generate_io_yaml(config.outputs, "output")}

# Golden Tests
tests:
  golden:
{_generate_golden_tests_yaml(tmpl["golden_tests"])}

# Provenance Configuration
provenance:
  algorithm: sha256
  tracking: full
  audit_log: true

# Safety Constraints
safety:
  max_tokens: 4096
  rate_limit: 100
  timeout_seconds: 30
  allowed_tools: {config.tools}

# Explainability
explainability:
  audit_level: comprehensive
  citation_required: true
  confidence_threshold: 0.8
'''


def _generate_test_code(config: AgentConfig, tmpl: dict) -> str:
    """Generate test_agent.py code."""
    module_name = config.name.replace("-", "_")
    class_name = module_name.title().replace("_", "")

    return f'''# -*- coding: utf-8 -*-
"""
Tests for {config.display_name}
{"=" * (len(config.display_name) + 10)}

Test types:
- Unit tests: Basic functionality
- Golden tests: Determinism verification (100 runs)
- Property tests: Input validation
"""

import pytest
import hashlib
import asyncio
from typing import Dict, Any

# Import agent
from ..agent import {class_name}Agent, AgentInput, AgentOutput, create_agent
from ..tools import {", ".join(config.tools)}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Create agent instance for testing."""
    return create_agent()


@pytest.fixture
def sample_input():
    """Create sample input data."""
    return AgentInput(
{_generate_fixture_data(config.inputs)}
    )


# =============================================================================
# Unit Tests
# =============================================================================

class Test{class_name}Agent:
    """Unit tests for {class_name}Agent."""

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "{config.display_name}"
        assert agent.version == "{config.version}"
        assert agent.category == "{config.category}"

    def test_agent_metadata(self, agent):
        """Test agent metadata is correct."""
        metadata = agent.metadata
        assert "name" in metadata
        assert "version" in metadata
        assert "tools" in metadata
        assert len(metadata["tools"]) == {len(config.tools)}

    @pytest.mark.asyncio
    async def test_agent_run_returns_output(self, agent, sample_input):
        """Test agent run returns valid output."""
        # Skip if not implemented
        with pytest.raises(NotImplementedError):
            await agent.run(sample_input)


# =============================================================================
# Golden Tests (Determinism)
# =============================================================================

class TestGoldenTests:
    """Golden tests for determinism verification."""

    @pytest.mark.parametrize("run_number", range(5))  # Run 5 times for CI, 100 for full
    @pytest.mark.asyncio
    async def test_determinism(self, agent, sample_input, run_number):
        """
        Test that agent produces identical outputs for identical inputs.

        IMPORTANT: This test verifies the zero-hallucination principle.
        Every run with the same input MUST produce byte-identical output.
        """
        # Skip if not implemented
        pytest.skip("Implement agent workflow first")

        # Run twice with same input
        result1 = await agent.run(sample_input)
        result2 = await agent.run(sample_input)

        # Compare provenance hashes (must be identical)
        assert result1.provenance_hash == result2.provenance_hash, \\
            f"Run {{run_number}}: Provenance hash mismatch - agent is non-deterministic!"


# =============================================================================
# Tool Tests
# =============================================================================

class TestTools:
    """Tests for individual tools."""

{_generate_tool_tests(config.tools)}


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests with external systems."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, agent, sample_input):
        """Test complete workflow from input to output."""
        pytest.skip("Implement integration tests when ready")
'''


def _generate_conftest() -> str:
    """Generate conftest.py for pytest configuration."""
    return '''# -*- coding: utf-8 -*-
"""
Pytest Configuration for Agent Tests
====================================

Provides fixtures and configuration for testing.
"""

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "golden: marks tests as golden/determinism tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
'''


def _generate_readme(config: AgentConfig, tmpl: dict) -> str:
    """Generate README.md documentation."""
    return f'''# {config.display_name}

{config.description}

## Overview

- **Category:** {config.category}
- **Template:** {config.template}
- **Version:** {config.version}
- **License:** {config.license}
- **Author:** {config.author}

## Installation

```bash
# From registry
gl agent pull {config.name}

# From source
pip install -e .
```

## Usage

```python
from {config.name.replace("-", "_")} import create_agent, AgentInput

# Create agent
agent = create_agent()

# Prepare input
input_data = AgentInput(
    # ... your input data
)

# Run agent
result = await agent.run(input_data)

# Access results
print(f"Result: {{result}}")
print(f"Provenance: {{result.provenance_hash}}")
```

## Tools

{_generate_tools_markdown(config.tools)}

## Testing

```bash
# Run all tests
gl agent test .

# Run golden tests (determinism)
gl agent test . --golden

# Run with coverage
gl agent test . --coverage
```

## Certification

```bash
# Full certification
gl agent certify .

# Target gold certification
gl agent certify . --level gold
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `gl agent test .`
5. Submit a pull request

## License

{config.license}
'''


# =============================================================================
# Helper Functions
# =============================================================================

def _generate_input_fields(inputs: List[str]) -> str:
    """Generate Pydantic fields for inputs."""
    if not inputs:
        return "    pass"

    lines = []
    for inp in inputs:
        field_name = inp.replace("-", "_")
        lines.append(f'    {field_name}: Any = Field(..., description="{inp.replace("_", " ").title()}")')
    return "\n".join(lines)


def _generate_output_fields(outputs: List[str]) -> str:
    """Generate Pydantic fields for outputs."""
    if not outputs:
        return "    result: Any = Field(..., description='Processing result')"

    lines = []
    for out in outputs:
        field_name = out.replace("-", "_")
        lines.append(f'    {field_name}: Any = Field(..., description="{out.replace("_", " ").title()}")')
    return "\n".join(lines)


def _generate_tools_yaml(tools: List[str]) -> str:
    """Generate YAML for tools."""
    lines = []
    for tool in tools:
        lines.append(f'''  - name: {tool}
    type: deterministic
    description: "{tool.replace("_", " ").title()} implementation"
    inputs:
      - name: input_data
        type: object
    outputs:
      - name: result
        type: object''')
    return "\n".join(lines)


def _generate_io_yaml(items: List[str], io_type: str) -> str:
    """Generate YAML for inputs/outputs."""
    lines = []
    for item in items:
        lines.append(f'''  - name: {item}
    type: any
    required: true
    description: "{item.replace("_", " ").title()}"''')
    return "\n".join(lines)


def _generate_golden_tests_yaml(count: int) -> str:
    """Generate YAML for golden tests."""
    lines = []
    for i in range(min(3, count)):  # Generate 3 sample tests
        lines.append(f'''    - name: golden_test_{i+1}
      input:
        sample: "test_value_{i+1}"
      expected_output:
        status: "success"
      tolerance: 0.0001''')
    return "\n".join(lines)


def _generate_fixture_data(inputs: List[str]) -> str:
    """Generate fixture data for tests."""
    lines = []
    for inp in inputs:
        field_name = inp.replace("-", "_")
        lines.append(f'        {field_name}="sample_value",')
    return "\n".join(lines)


def _generate_tool_tests(tools: List[str]) -> str:
    """Generate test methods for each tool."""
    lines = []
    for tool in tools:
        lines.append(f'''
    @pytest.mark.asyncio
    async def test_{tool}(self):
        """Test {tool.replace("_", " ")} tool."""
        input_data = {{"test": "value"}}

        with pytest.raises(NotImplementedError):
            await {tool}(input_data)
''')
    return "".join(lines)


def _generate_tools_markdown(tools: List[str]) -> str:
    """Generate markdown for tools documentation."""
    lines = []
    for tool in tools:
        lines.append(f"- **{tool}**: {tool.replace('_', ' ').title()} implementation")
    return "\n".join(lines)


def _generate_agent_code_from_spec(spec: dict) -> str:
    """Generate agent code from parsed spec."""
    # Simplified implementation
    name = spec.get("name", "Agent")
    return f'''# -*- coding: utf-8 -*-
"""
{name} - Generated from AgentSpec
"""

class Agent:
    """Agent implementation."""

    def __init__(self):
        self.name = "{name}"
        self.version = "{spec.get("version", "0.1.0")}"

    async def run(self, input_data):
        """Execute agent workflow."""
        raise NotImplementedError()
'''


def _generate_tools_code_from_spec(spec: dict) -> str:
    """Generate tools code from parsed spec."""
    tools = spec.get("tools", [])
    code = '''# -*- coding: utf-8 -*-
"""Tool implementations generated from spec."""

'''
    for tool in tools:
        tool_name = tool.get("name", "tool") if isinstance(tool, dict) else tool
        code += f'''
async def {tool_name}(input_data, config=None):
    """Tool implementation."""
    raise NotImplementedError()
'''
    return code


def _generate_test_code_from_spec(spec: dict) -> str:
    """Generate test code from parsed spec."""
    return '''# -*- coding: utf-8 -*-
"""Tests generated from spec."""

import pytest

class TestAgent:
    """Agent tests."""

    def test_placeholder(self):
        """Placeholder test."""
        assert True
'''


def _generate_init_code(spec: dict) -> str:
    """Generate __init__.py from spec."""
    name = spec.get("name", "Agent")
    return f'''# -*- coding: utf-8 -*-
"""
{name}
"""

__version__ = "{spec.get("version", "0.1.0")}"

from .agent import Agent
from .tools import *
'''


def _init_git_repo(output_dir: Path) -> None:
    """Initialize git repository."""
    import subprocess

    try:
        # Initialize git
        subprocess.run(
            ["git", "init"],
            cwd=output_dir,
            capture_output=True,
            check=True,
        )

        # Create .gitignore
        gitignore = output_dir / ".gitignore"
        gitignore.write_text('''# Python
__pycache__/
*.py[cod]
*$py.class
.Python
*.so
.eggs/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Build
dist/
build/
*.egg-info/

# Local config
.env
.env.local
*.local.yaml
''')

        # Initial commit
        subprocess.run(
            ["git", "add", "."],
            cwd=output_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial agent scaffold from GreenLang Agent Factory"],
            cwd=output_dir,
            capture_output=True,
            check=True,
        )

    except subprocess.CalledProcessError as e:
        logger.warning(f"Git initialization failed: {e}")
    except FileNotFoundError:
        logger.warning("Git not found - skipping git initialization")
