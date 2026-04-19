"""
Scaffolding Commands

Commands for scaffolding new agents, tests, and documentation
using Cookiecutter-style templates for rapid development.
"""

import typer
from typing import Optional, List
from pathlib import Path
import yaml
import json
import shutil
from datetime import datetime
from enum import Enum

from cli.utils.console import (
    console,
    print_error,
    print_success,
    print_warning,
    print_info,
    create_info_panel,
    create_progress_bar,
    print_generation_summary,
    confirm_action,
)
from cli.utils.config import load_config, get_config_value


class AgentTemplate(str, Enum):
    """Agent template enumeration."""
    BASIC = "basic"
    REGULATORY = "regulatory"
    CALCULATOR = "calculator"
    INTEGRATION = "integration"
    PIPELINE = "pipeline"
    CUSTOM = "custom"


class TestType(str, Enum):
    """Test type enumeration."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    COMPLIANCE = "compliance"
    GOLDEN = "golden"
    ALL = "all"


class DocType(str, Enum):
    """Documentation type enumeration."""
    README = "readme"
    API = "api"
    USER_GUIDE = "user-guide"
    DEPLOYMENT = "deployment"
    ALL = "all"


# Create scaffold command group
app = typer.Typer(
    help="Scaffolding commands - generate agents, tests, and documentation",
    no_args_is_help=True,
)


@app.command("agent")
def scaffold_agent(
    template: AgentTemplate = typer.Option(
        AgentTemplate.BASIC,
        "--template",
        "-t",
        help="Template to use for scaffolding",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Agent name",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Agent description",
    ),
    author: Optional[str] = typer.Option(
        None,
        "--author",
        "-a",
        help="Author name",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        "-i/-I",
        help="Interactive mode for prompts",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be created without creating files",
    ),
):
    """
    Create a new agent from a template.

    Scaffolds a complete agent structure with all necessary files:
    - Agent implementation (agent.py)
    - Pydantic models (models.py)
    - Configuration (agent.yaml)
    - Tests (tests/)
    - Documentation (README.md)
    - Deployment (Dockerfile)

    Templates available:
    - basic: Simple agent with minimal setup
    - regulatory: Compliance-focused agent with audit trails
    - calculator: Emissions/carbon calculator agent
    - integration: API integration agent
    - pipeline: Multi-step pipeline agent

    Examples:
        gl scaffold agent --template regulatory
        gl scaffold agent -t calculator -n carbon-calc
        gl scaffold agent --template basic --no-interactive
    """
    try:
        console.print("\n[bold cyan]Agent Scaffolding[/bold cyan]\n")

        # Interactive prompts
        if interactive:
            if not name:
                name = console.input("[bold]Agent name[/bold]: ").strip()
                if not name:
                    print_error("Agent name is required")
                    raise typer.Exit(1)

            if not description:
                description = console.input("[bold]Description[/bold] (optional): ").strip() or None

            if not author:
                author = console.input("[bold]Author[/bold] (optional): ").strip() or None

            # Template selection if not specified
            console.print("\n[bold]Available templates:[/bold]")
            for t in AgentTemplate:
                if t != AgentTemplate.CUSTOM:
                    console.print(f"  - {t.value}")

            template_input = console.input(f"\n[bold]Template[/bold] [{template.value}]: ").strip()
            if template_input and template_input in [t.value for t in AgentTemplate]:
                template = AgentTemplate(template_input)

        elif not name:
            print_error("Agent name is required (use --name or --interactive)")
            raise typer.Exit(1)

        # Resolve output directory
        if output_dir is None:
            config = load_config()
            agents_dir = Path(get_config_value("defaults.output_dir", "agents", config))
            output_dir = agents_dir / name

        # Agent ID
        agent_id = name.lower().replace(" ", "-").replace("_", "-")
        agent_name = " ".join(word.title() for word in name.replace("-", " ").replace("_", " ").split())

        # Display configuration
        console.print(create_info_panel("Scaffold Configuration", {
            "Template": template.value,
            "Agent ID": agent_id,
            "Agent Name": agent_name,
            "Output": str(output_dir),
            "Description": description or "Not specified",
            "Author": author or "GreenLang Team",
        }))
        console.print()

        if dry_run:
            print_info("Dry run mode - showing what would be created:")
            _show_scaffold_preview(template, agent_id, output_dir)
            raise typer.Exit(0)

        # Confirm
        if interactive and not confirm_action("Create agent scaffold?", default=True):
            print_info("Scaffolding cancelled")
            raise typer.Exit(0)

        # Create scaffold
        files_created = _create_agent_scaffold(
            template=template,
            agent_id=agent_id,
            agent_name=agent_name,
            output_dir=output_dir,
            description=description,
            author=author,
        )

        # Summary
        print_generation_summary(output_dir, files_created)

        # Next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  1. Navigate to: [cyan]cd {output_dir}[/cyan]")
        console.print(f"  2. Install dependencies: [cyan]pip install -r requirements.txt[/cyan]")
        console.print(f"  3. Implement business logic in: [cyan]agent.py[/cyan]")
        console.print(f"  4. Run tests: [cyan]gl agent test {agent_id}[/cyan]")
        console.print(f"  5. Deploy: [cyan]gl agent deploy {agent_id}[/cyan]\n")

        print_success(f"Agent scaffold created: {agent_id}")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Scaffolding failed: {str(e)}")
        raise typer.Exit(1)


@app.command("test")
def scaffold_tests(
    agent: str = typer.Option(
        ...,
        "--agent",
        "-a",
        help="Agent name or path to generate tests for",
    ),
    test_type: TestType = typer.Option(
        TestType.ALL,
        "--type",
        "-t",
        help="Type of tests to generate",
    ),
    coverage_target: int = typer.Option(
        85,
        "--coverage",
        "-c",
        help="Target coverage percentage",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (defaults to agent/tests)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "-f",
        help="Overwrite existing tests",
    ),
):
    """
    Generate test suite for an agent.

    Creates comprehensive test files based on the agent implementation:
    - Unit tests for individual methods
    - Integration tests for agent workflows
    - Golden tests for determinism validation
    - Compliance tests for regulatory requirements

    Examples:
        gl scaffold test --agent carbon-calculator
        gl scaffold test --agent my-agent --type unit
        gl scaffold test --agent my-agent --coverage 90
    """
    try:
        console.print(f"\n[bold cyan]Test Scaffolding[/bold cyan]\n")

        # Resolve agent path
        agent_path = _resolve_agent_path(agent)

        if not agent_path.exists():
            print_error(f"Agent not found: {agent}")
            raise typer.Exit(1)

        # Determine output directory
        if output_dir is None:
            output_dir = agent_path / "tests"

        # Check for existing tests
        if output_dir.exists() and not overwrite:
            existing_tests = list(output_dir.glob("test_*.py"))
            if existing_tests:
                print_warning(f"Tests already exist in {output_dir}")
                if not confirm_action("Overwrite existing tests?", default=False):
                    print_info("Test scaffolding cancelled")
                    raise typer.Exit(0)

        # Display configuration
        console.print(create_info_panel("Test Configuration", {
            "Agent": agent_path.name,
            "Test Type": test_type.value,
            "Coverage Target": f"{coverage_target}%",
            "Output": str(output_dir),
        }))
        console.print()

        # Generate tests
        files_created = []

        with create_progress_bar() as progress:
            task = progress.add_task("Generating tests...", total=100)

            # Create test directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate conftest.py
            if test_type in [TestType.ALL, TestType.UNIT]:
                files_created.extend(_generate_conftest(agent_path, output_dir))
                progress.update(task, advance=20)

            # Generate unit tests
            if test_type in [TestType.ALL, TestType.UNIT]:
                files_created.extend(_generate_unit_tests(agent_path, output_dir))
                progress.update(task, advance=20)

            # Generate integration tests
            if test_type in [TestType.ALL, TestType.INTEGRATION]:
                files_created.extend(_generate_integration_tests(agent_path, output_dir))
                progress.update(task, advance=20)

            # Generate golden tests
            if test_type in [TestType.ALL, TestType.GOLDEN]:
                files_created.extend(_generate_golden_tests(agent_path, output_dir))
                progress.update(task, advance=20)

            # Generate compliance tests
            if test_type in [TestType.ALL, TestType.COMPLIANCE]:
                files_created.extend(_generate_compliance_tests(agent_path, output_dir))
                progress.update(task, advance=20)

        # Summary
        console.print()
        print_generation_summary(output_dir, files_created)

        # Usage
        console.print("\n[bold]Run tests:[/bold]")
        console.print(f"  [cyan]gl agent test {agent}[/cyan]")
        console.print(f"  [cyan]pytest {output_dir} --cov={agent_path} --cov-report=term-missing[/cyan]\n")

        print_success(f"Test suite generated: {len(files_created)} files")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Test scaffolding failed: {str(e)}")
        raise typer.Exit(1)


@app.command("docs")
def scaffold_docs(
    agent: str = typer.Option(
        ...,
        "--agent",
        "-a",
        help="Agent name or path to generate documentation for",
    ),
    doc_type: DocType = typer.Option(
        DocType.ALL,
        "--type",
        "-t",
        help="Type of documentation to generate",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (defaults to agent/docs)",
    ),
    format_type: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format (markdown/rst)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing documentation",
    ),
):
    """
    Generate documentation for an agent.

    Creates comprehensive documentation from agent code:
    - README with usage examples
    - API documentation from docstrings
    - User guide with configuration details
    - Deployment guide

    Examples:
        gl scaffold docs --agent carbon-calculator
        gl scaffold docs --agent my-agent --type api
        gl scaffold docs --agent my-agent --format rst
    """
    try:
        console.print(f"\n[bold cyan]Documentation Scaffolding[/bold cyan]\n")

        # Resolve agent path
        agent_path = _resolve_agent_path(agent)

        if not agent_path.exists():
            print_error(f"Agent not found: {agent}")
            raise typer.Exit(1)

        # Determine output directory
        if output_dir is None:
            output_dir = agent_path

        # Load agent metadata
        metadata = _load_agent_metadata(agent_path)

        # Display configuration
        console.print(create_info_panel("Documentation Configuration", {
            "Agent": agent_path.name,
            "Doc Type": doc_type.value,
            "Format": format_type,
            "Output": str(output_dir),
        }))
        console.print()

        # Generate documentation
        files_created = []

        with create_progress_bar() as progress:
            task = progress.add_task("Generating documentation...", total=100)

            # Generate README
            if doc_type in [DocType.ALL, DocType.README]:
                files_created.extend(_generate_readme(agent_path, output_dir, metadata, format_type))
                progress.update(task, advance=25)

            # Generate API docs
            if doc_type in [DocType.ALL, DocType.API]:
                files_created.extend(_generate_api_docs(agent_path, output_dir, format_type))
                progress.update(task, advance=25)

            # Generate user guide
            if doc_type in [DocType.ALL, DocType.USER_GUIDE]:
                files_created.extend(_generate_user_guide(agent_path, output_dir, metadata, format_type))
                progress.update(task, advance=25)

            # Generate deployment guide
            if doc_type in [DocType.ALL, DocType.DEPLOYMENT]:
                files_created.extend(_generate_deployment_guide(agent_path, output_dir, metadata, format_type))
                progress.update(task, advance=25)

        # Summary
        console.print()
        print_generation_summary(output_dir, files_created)

        print_success(f"Documentation generated: {len(files_created)} files")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Documentation scaffolding failed: {str(e)}")
        raise typer.Exit(1)


@app.command("templates")
def list_templates(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed template information",
    ),
):
    """
    List available scaffold templates.

    Display all available templates for agents, tests, and documentation.

    Examples:
        gl scaffold templates
        gl scaffold templates --verbose
    """
    try:
        console.print("\n[bold cyan]Available Templates[/bold cyan]\n")

        # Agent templates
        console.print("[bold]Agent Templates:[/bold]\n")

        from rich.table import Table

        table = Table(show_header=True)
        table.add_column("Template", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Features", style="dim")

        templates = {
            "basic": {
                "description": "Simple agent with minimal setup",
                "features": "Pydantic models, basic tests, README",
            },
            "regulatory": {
                "description": "Compliance-focused agent with audit trails",
                "features": "Provenance tracking, audit logging, compliance tests",
            },
            "calculator": {
                "description": "Emissions/carbon calculator agent",
                "features": "Formula engine, unit conversion, golden tests",
            },
            "integration": {
                "description": "API integration agent",
                "features": "HTTP client, retry logic, mock tests",
            },
            "pipeline": {
                "description": "Multi-step pipeline agent",
                "features": "Step orchestration, error handling, integration tests",
            },
        }

        for name, info in templates.items():
            table.add_row(name, info["description"], info["features"])

        console.print(table)

        # Test types
        console.print("\n[bold]Test Types:[/bold]\n")
        test_types = {
            "unit": "Unit tests for individual methods and functions",
            "integration": "Integration tests for agent workflows",
            "golden": "Golden tests for determinism validation",
            "compliance": "Compliance tests for regulatory requirements",
            "e2e": "End-to-end tests for full agent lifecycle",
        }

        for name, desc in test_types.items():
            console.print(f"  [cyan]{name}[/cyan]: {desc}")

        # Doc types
        console.print("\n[bold]Documentation Types:[/bold]\n")
        doc_types = {
            "readme": "README with usage examples",
            "api": "API documentation from docstrings",
            "user-guide": "User guide with configuration details",
            "deployment": "Deployment and operations guide",
        }

        for name, desc in doc_types.items():
            console.print(f"  [cyan]{name}[/cyan]: {desc}")

        console.print()

    except Exception as e:
        print_error(f"Failed to list templates: {str(e)}")
        raise typer.Exit(1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _resolve_agent_path(name: str) -> Path:
    """Resolve agent name or path to full path."""
    path = Path(name)

    if path.exists() and path.is_dir():
        return path.resolve()

    config = load_config()
    agents_dir = Path(get_config_value("defaults.output_dir", "agents", config))
    agent_path = agents_dir / name

    if agent_path.exists():
        return agent_path.resolve()

    return Path(name)


def _load_agent_metadata(agent_path: Path) -> dict:
    """Load agent metadata from agent.yaml."""
    metadata_file = agent_path / "agent.yaml"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            return yaml.safe_load(f) or {}
    return {"id": agent_path.name, "name": agent_path.name}


def _show_scaffold_preview(template: AgentTemplate, agent_id: str, output_dir: Path):
    """Show preview of files that would be created."""
    console.print("\n[bold]Files to be created:[/bold]\n")

    files = [
        f"{output_dir}/agent.py",
        f"{output_dir}/models.py",
        f"{output_dir}/__init__.py",
        f"{output_dir}/agent.yaml",
        f"{output_dir}/requirements.txt",
        f"{output_dir}/README.md",
        f"{output_dir}/Dockerfile",
        f"{output_dir}/.dockerignore",
        f"{output_dir}/tests/__init__.py",
        f"{output_dir}/tests/conftest.py",
        f"{output_dir}/tests/test_agent.py",
        f"{output_dir}/tests/test_models.py",
    ]

    for f in files:
        console.print(f"  [green]CREATE[/green] {f}")


def _create_agent_scaffold(
    template: AgentTemplate,
    agent_id: str,
    agent_name: str,
    output_dir: Path,
    description: Optional[str],
    author: Optional[str],
) -> List[Path]:
    """Create complete agent scaffold."""
    files_created = []

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tests").mkdir(exist_ok=True)

    # Generate files based on template
    from cli.commands.agent import (
        _generate_agent_module,
        _generate_models,
        _generate_config,
        _generate_test_suite,
        _generate_docs,
        _generate_deployment,
        AgentType,
    )

    # Map template to agent type
    type_map = {
        AgentTemplate.BASIC: AgentType.BASIC,
        AgentTemplate.REGULATORY: AgentType.REGULATORY,
        AgentTemplate.CALCULATOR: AgentType.CALCULATOR,
        AgentTemplate.INTEGRATION: AgentType.INTEGRATION,
        AgentTemplate.PIPELINE: AgentType.PIPELINE,
    }
    agent_type = type_map.get(template, AgentType.BASIC)

    # Generate all files
    files_created.extend(_generate_agent_module(
        output_dir, agent_id, agent_name, agent_type, description, author
    ))
    files_created.extend(_generate_models(
        output_dir, agent_id, agent_name, agent_type
    ))
    files_created.extend(_generate_config(
        output_dir, agent_id, agent_name, agent_type, description, author
    ))
    files_created.extend(_generate_test_suite(
        output_dir, agent_id, agent_name, agent_type
    ))
    files_created.extend(_generate_docs(
        output_dir, agent_id, agent_name, agent_type, description
    ))
    files_created.extend(_generate_deployment(
        output_dir, agent_id, agent_name
    ))

    return files_created


def _generate_conftest(agent_path: Path, output_dir: Path) -> List[Path]:
    """Generate conftest.py with fixtures."""
    files = []

    agent_name = agent_path.name.replace("-", "_").title().replace("_", "")

    conftest_content = f'''"""
Test configuration and fixtures.

Auto-generated by GreenLang Agent Factory CLI.
"""

import pytest
from pathlib import Path
import sys

# Add agent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def agent():
    """Create agent instance for testing."""
    try:
        from agent import {agent_name}Agent
        return {agent_name}Agent({{}})
    except ImportError:
        pytest.skip("Agent module not found")


@pytest.fixture
def sample_input():
    """Create sample input data."""
    try:
        from models import {agent_name}Input
        return {agent_name}Input(data="test_data")
    except ImportError:
        return {{"data": "test_data"}}


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {{
        "provenance_enabled": True,
        "logging_level": "DEBUG",
    }}


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path / "output"
'''

    conftest_file = output_dir / "conftest.py"
    conftest_file.write_text(conftest_content)
    files.append(conftest_file)

    # __init__.py
    init_file = output_dir / "__init__.py"
    init_file.write_text('"""Test suite."""\n')
    files.append(init_file)

    return files


def _generate_unit_tests(agent_path: Path, output_dir: Path) -> List[Path]:
    """Generate unit test file."""
    files = []

    agent_name = agent_path.name.replace("-", "_").title().replace("_", "")

    test_content = f'''"""
Unit tests for {agent_path.name} agent.

Auto-generated by GreenLang Agent Factory CLI.
"""

import pytest


class TestAgentInit:
    """Tests for agent initialization."""

    def test_init_default(self, agent):
        """Test agent initializes with defaults."""
        assert agent is not None

    def test_init_with_config(self, mock_config):
        """Test agent initializes with custom config."""
        try:
            from agent import {agent_name}Agent
            agent = {agent_name}Agent(mock_config)
            assert agent.config == mock_config
        except ImportError:
            pytest.skip("Agent module not found")


class TestAgentProcess:
    """Tests for agent processing."""

    def test_process_valid_input(self, agent, sample_input):
        """Test processing with valid input."""
        result = agent.process(sample_input)
        assert result is not None
        assert result.validation_status in ["PASS", "FAIL", "WARNING"]

    def test_process_tracks_time(self, agent, sample_input):
        """Test that processing time is tracked."""
        result = agent.process(sample_input)
        assert result.processing_time_ms >= 0


class TestAgentValidation:
    """Tests for input validation."""

    def test_validates_input(self, agent, sample_input):
        """Test input validation."""
        errors = agent._validate_input(sample_input)
        assert isinstance(errors, list)


class TestAgentHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_returns_status(self, agent):
        """Test health check returns expected format."""
        health = agent.health_check()
        assert "status" in health
        assert health["status"] == "healthy"


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self, agent, sample_input):
        """Test that same input produces consistent output."""
        result1 = agent.process(sample_input)
        result2 = agent.process(sample_input)

        assert result1.result == result2.result
        assert result1.provenance_hash == result2.provenance_hash
'''

    test_file = output_dir / "test_agent.py"
    test_file.write_text(test_content)
    files.append(test_file)

    return files


def _generate_integration_tests(agent_path: Path, output_dir: Path) -> List[Path]:
    """Generate integration test file."""
    files = []

    test_content = f'''"""
Integration tests for {agent_path.name} agent.

Auto-generated by GreenLang Agent Factory CLI.
"""

import pytest


class TestAgentWorkflow:
    """Integration tests for agent workflows."""

    def test_full_processing_workflow(self, agent, sample_input):
        """Test complete processing workflow."""
        # Process
        result = agent.process(sample_input)

        # Verify output
        assert result is not None
        assert result.validation_status == "PASS"
        assert result.provenance_hash != ""

    def test_error_handling_workflow(self, agent):
        """Test error handling in workflow."""
        # This should be customized based on agent behavior
        pass


class TestAgentConfiguration:
    """Integration tests for configuration."""

    def test_config_affects_behavior(self, mock_config):
        """Test that configuration affects agent behavior."""
        pass
'''

    test_file = output_dir / "test_integration.py"
    test_file.write_text(test_content)
    files.append(test_file)

    return files


def _generate_golden_tests(agent_path: Path, output_dir: Path) -> List[Path]:
    """Generate golden test file for determinism validation."""
    files = []

    test_content = f'''"""
Golden tests for {agent_path.name} agent.

These tests validate deterministic behavior by comparing outputs
against known expected values.

Auto-generated by GreenLang Agent Factory CLI.
"""

import pytest
import json
from pathlib import Path


# Golden test data - update with actual expected values
GOLDEN_TEST_CASES = [
    {{
        "name": "basic_input",
        "input": {{"data": "test"}},
        "expected_status": "PASS",
    }},
]


class TestGoldenOutputs:
    """Golden tests for deterministic outputs."""

    @pytest.mark.parametrize("test_case", GOLDEN_TEST_CASES)
    def test_golden_output(self, agent, test_case):
        """Test output matches expected golden value."""
        try:
            from models import {agent_path.name.replace("-", "_").title().replace("_", "")}Input
            input_data = {agent_path.name.replace("-", "_").title().replace("_", "")}Input(**test_case["input"])
        except ImportError:
            pytest.skip("Models not available")

        result = agent.process(input_data)

        assert result.validation_status == test_case["expected_status"]


class TestProvenanceConsistency:
    """Tests for provenance hash consistency."""

    def test_provenance_hash_deterministic(self, agent, sample_input):
        """Test that provenance hash is deterministic."""
        results = [agent.process(sample_input) for _ in range(3)]
        hashes = [r.provenance_hash for r in results]

        # All hashes should be identical
        assert len(set(hashes)) == 1
'''

    test_file = output_dir / "test_golden.py"
    test_file.write_text(test_content)
    files.append(test_file)

    return files


def _generate_compliance_tests(agent_path: Path, output_dir: Path) -> List[Path]:
    """Generate compliance test file."""
    files = []

    test_content = f'''"""
Compliance tests for {agent_path.name} agent.

These tests validate regulatory compliance requirements.

Auto-generated by GreenLang Agent Factory CLI.
"""

import pytest


class TestComplianceRequirements:
    """Tests for regulatory compliance."""

    def test_provenance_tracking(self, agent, sample_input):
        """Test that provenance is tracked for audit trail."""
        result = agent.process(sample_input)

        # Provenance hash should be generated
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_no_hallucination_in_calculations(self, agent):
        """Test that calculations are deterministic (no LLM)."""
        # Agent should not use LLM for numeric calculations
        # This is validated by the golden tests
        pass

    def test_input_validation(self, agent, sample_input):
        """Test that inputs are properly validated."""
        # Validation should be performed
        errors = agent._validate_input(sample_input)
        assert isinstance(errors, list)

    def test_output_validation(self, agent, sample_input):
        """Test that outputs meet requirements."""
        result = agent.process(sample_input)

        # Output should have required fields
        assert hasattr(result, "result")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "validation_status")
        assert hasattr(result, "processing_time_ms")


class TestAuditTrail:
    """Tests for audit trail requirements."""

    def test_timestamp_recorded(self, agent, sample_input):
        """Test that processing timestamp is recorded."""
        result = agent.process(sample_input)
        assert hasattr(result, "timestamp")
        assert result.timestamp != ""

    def test_processing_time_recorded(self, agent, sample_input):
        """Test that processing time is recorded."""
        result = agent.process(sample_input)
        assert result.processing_time_ms >= 0
'''

    test_file = output_dir / "test_compliance.py"
    test_file.write_text(test_content)
    files.append(test_file)

    return files


def _generate_readme(
    agent_path: Path,
    output_dir: Path,
    metadata: dict,
    format_type: str,
) -> List[Path]:
    """Generate README documentation."""
    files = []

    agent_name = metadata.get("name", agent_path.name)
    description = metadata.get("description", f"GreenLang agent for {agent_name} operations.")

    readme_content = f'''# {agent_name}

{description}

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from agent import process

result = process({{"data": "your_data"}})
print(result)
```

## Features

- Zero-hallucination deterministic processing
- SHA-256 provenance tracking for audit trails
- Pydantic model validation
- Health check endpoint

## Configuration

See `agent.yaml` for configuration options.

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

## License

MIT License
'''

    readme_file = output_dir / "README.md"
    readme_file.write_text(readme_content)
    files.append(readme_file)

    return files


def _generate_api_docs(agent_path: Path, output_dir: Path, format_type: str) -> List[Path]:
    """Generate API documentation."""
    files = []

    docs_dir = output_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    api_content = f'''# API Reference

## Agent Class

### `process(input_data)`

Process input data and return result.

**Parameters:**
- `input_data`: Input data model

**Returns:**
- Output data model with result and provenance

### `health_check()`

Return health status.

**Returns:**
- Dictionary with status, agent name, version, timestamp

## Models

### Input Model

```python
class Input(BaseModel):
    data: Any
    options: Optional[Dict]
    metadata: Optional[Dict]
```

### Output Model

```python
class Output(BaseModel):
    result: Any
    provenance_hash: str
    processing_time_ms: float
    validation_status: str
    timestamp: str
```
'''

    api_file = docs_dir / "API.md"
    api_file.write_text(api_content)
    files.append(api_file)

    return files


def _generate_user_guide(
    agent_path: Path,
    output_dir: Path,
    metadata: dict,
    format_type: str,
) -> List[Path]:
    """Generate user guide documentation."""
    files = []

    docs_dir = output_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    guide_content = f'''# User Guide

## Overview

This guide explains how to use the {metadata.get("name", "agent")} agent.

## Getting Started

1. Install dependencies
2. Configure the agent
3. Process your data

## Configuration

Edit `agent.yaml` to configure:

- `provenance_enabled`: Enable/disable provenance tracking
- `logging_level`: Set logging verbosity

## Examples

### Basic Usage

```python
from agent import process

result = process({{"data": "example"}})
```

### With Configuration

```python
from agent import Agent

agent = Agent({{"provenance_enabled": True}})
result = agent.process(input_data)
```

## Troubleshooting

Common issues and solutions.
'''

    guide_file = docs_dir / "USER_GUIDE.md"
    guide_file.write_text(guide_content)
    files.append(guide_file)

    return files


def _generate_deployment_guide(
    agent_path: Path,
    output_dir: Path,
    metadata: dict,
    format_type: str,
) -> List[Path]:
    """Generate deployment guide documentation."""
    files = []

    docs_dir = output_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    deploy_content = f'''# Deployment Guide

## Docker Deployment

Build and run with Docker:

```bash
docker build -t {metadata.get("id", "agent")}:latest .
docker run -p 8000:8000 {metadata.get("id", "agent")}:latest
```

## Kubernetes Deployment

Deploy to Kubernetes:

```bash
kubectl apply -f k8s/
```

## Environment Variables

- `AGENT_ID`: Agent identifier
- `LOG_LEVEL`: Logging level

## Health Checks

The agent exposes a health check endpoint for monitoring.

## Scaling

Configure replicas based on load requirements.
'''

    deploy_file = docs_dir / "DEPLOYMENT.md"
    deploy_file.write_text(deploy_content)
    files.append(deploy_file)

    return files
