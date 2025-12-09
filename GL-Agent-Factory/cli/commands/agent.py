"""
Agent Management Commands

Commands for creating, validating, testing, and publishing agents.
"""

import typer
from typing import Optional
from pathlib import Path
import yaml
import json
from datetime import datetime

from cli.utils.console import (
    console,
    print_error,
    print_success,
    print_warning,
    print_info,
    create_agent_table,
    create_progress_bar,
    create_info_panel,
    display_yaml,
    confirm_action,
    print_validation_results,
    print_test_results,
    print_generation_summary,
)
from cli.utils.config import load_config, get_config_value


# Create agent command group
app = typer.Typer(
    help="Agent management commands",
    no_args_is_help=True,
)


@app.command()
def create(
    spec_file: Path = typer.Argument(
        ...,
        help="Path to agent specification YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for generated agent",
    ),
    template: Optional[str] = typer.Option(
        None,
        "--template",
        "-t",
        help="Template to use for generation",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate spec without generating files",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    skip_tests: bool = typer.Option(
        False,
        "--skip-tests",
        help="Skip test generation",
    ),
    skip_docs: bool = typer.Option(
        False,
        "--skip-docs",
        help="Skip documentation generation",
    ),
):
    """
    Generate an agent from a specification file.

    This command reads an agent specification YAML file and generates
    a complete agent implementation with tests, documentation, and
    deployment configurations.

    Example:
        gl agent create specs/nfpa86-agent.yaml -o agents/nfpa86
    """
    try:
        # Load configuration
        config = load_config()

        # Determine output directory
        if output_dir is None:
            output_dir = Path(get_config_value("defaults.output_dir", "agents", config))

        console.print(f"\n[bold cyan]Creating agent from specification:[/bold cyan] {spec_file}\n")

        # Load and validate spec
        with open(spec_file, "r") as f:
            spec = yaml.safe_load(f)

        if verbose:
            console.print("[bold]Agent Specification:[/bold]")
            display_yaml(yaml.dump(spec, default_flow_style=False))
            console.print()

        # Extract agent info
        agent_id = spec.get("metadata", {}).get("id", "unknown")
        agent_name = spec.get("metadata", {}).get("name", "Unknown Agent")
        agent_version = spec.get("metadata", {}).get("version", "0.1.0")

        # Display agent info
        info = {
            "ID": agent_id,
            "Name": agent_name,
            "Version": agent_version,
            "Type": spec.get("metadata", {}).get("type", "unknown"),
            "Output": str(output_dir / agent_id),
        }
        console.print(create_info_panel("Agent Information", info))
        console.print()

        # Validate spec first
        validation_results = validate_spec(spec, verbose=verbose)
        print_validation_results(validation_results)

        if not validation_results.get("valid", False):
            print_error("Cannot generate agent with invalid specification")
            raise typer.Exit(1)

        if dry_run:
            print_info("Dry run complete - no files generated")
            raise typer.Exit(0)

        # Generate agent with progress tracking
        agent_output_dir = output_dir / agent_id
        agent_output_dir.mkdir(parents=True, exist_ok=True)

        files_created = []

        with create_progress_bar() as progress:
            # Core agent implementation
            task1 = progress.add_task("Generating core agent code...", total=100)
            files_created.extend(generate_core_agent(spec, agent_output_dir, verbose))
            progress.update(task1, completed=100)

            # Configuration files
            task2 = progress.add_task("Creating configuration files...", total=100)
            files_created.extend(generate_config_files(spec, agent_output_dir, verbose))
            progress.update(task2, completed=100)

            # Tests
            if not skip_tests:
                task3 = progress.add_task("Generating test suite...", total=100)
                files_created.extend(generate_tests(spec, agent_output_dir, verbose))
                progress.update(task3, completed=100)

            # Documentation
            if not skip_docs:
                task4 = progress.add_task("Creating documentation...", total=100)
                files_created.extend(generate_documentation(spec, agent_output_dir, verbose))
                progress.update(task4, completed=100)

            # Deployment configs
            task5 = progress.add_task("Generating deployment configs...", total=100)
            files_created.extend(generate_deployment_configs(spec, agent_output_dir, verbose))
            progress.update(task5, completed=100)

        # Print summary
        print_generation_summary(agent_output_dir, files_created)

        # Next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  1. Review the generated code: [cyan]cd {agent_output_dir}[/cyan]")
        console.print(f"  2. Run tests: [cyan]gl agent test {agent_output_dir}[/cyan]")
        console.print(f"  3. Customize the implementation as needed")
        console.print(f"  4. Publish: [cyan]gl agent publish {agent_output_dir}[/cyan]\n")

    except Exception as e:
        print_error(f"Failed to create agent: {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def validate(
    spec_file: Path = typer.Argument(
        ...,
        help="Path to agent specification YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Enable strict validation mode",
    ),
):
    """
    Validate an agent specification file without generating code.

    This command checks that the specification is valid, complete,
    and follows best practices.

    Example:
        gl agent validate specs/nfpa86-agent.yaml --strict
    """
    try:
        console.print(f"\n[bold cyan]Validating specification:[/bold cyan] {spec_file}\n")

        # Load spec
        with open(spec_file, "r") as f:
            spec = yaml.safe_load(f)

        if verbose:
            console.print("[bold]Agent Specification:[/bold]")
            display_yaml(yaml.dump(spec, default_flow_style=False))
            console.print()

        # Validate
        results = validate_spec(spec, strict=strict, verbose=verbose)
        print_validation_results(results)

        if not results.get("valid", False):
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"Validation failed: {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def test(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose test output",
    ),
    coverage: bool = typer.Option(
        True,
        "--coverage/--no-coverage",
        help="Generate coverage report",
    ),
    parallel: bool = typer.Option(
        True,
        "--parallel/--serial",
        help="Run tests in parallel",
    ),
):
    """
    Run tests for an agent.

    This command executes the test suite for an agent and displays
    the results with coverage information.

    Example:
        gl agent test agents/nfpa86 --verbose --coverage
    """
    try:
        console.print(f"\n[bold cyan]Running tests for agent:[/bold cyan] {agent_path}\n")

        # Check for test directory
        test_dir = agent_path / "tests"
        if not test_dir.exists():
            print_error(f"No tests found in {agent_path}")
            print_info("Generate tests with: gl agent create --with-tests")
            raise typer.Exit(1)

        # Run tests with progress
        with console.status("[bold green]Running tests...", spinner="dots"):
            results = run_agent_tests(agent_path, verbose=verbose, coverage=coverage, parallel=parallel)

        # Display results
        console.print()
        print_test_results(results)

        # Exit with appropriate code
        if results.get("failed", 0) > 0:
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"Test execution failed: {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def publish(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    registry: Optional[str] = typer.Option(
        None,
        "--registry",
        "-r",
        help="Registry URL (overrides config)",
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Version tag to publish",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force publish even if version exists",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate publish without uploading",
    ),
):
    """
    Publish an agent to the registry.

    This command packages and uploads an agent to the GreenLang
    Agent Registry for reuse and sharing.

    Example:
        gl agent publish agents/nfpa86 --tag v1.0.0
    """
    try:
        console.print(f"\n[bold cyan]Publishing agent:[/bold cyan] {agent_path}\n")

        # Load agent metadata
        metadata_file = agent_path / "agent.yaml"
        if not metadata_file.exists():
            print_error(f"No agent.yaml found in {agent_path}")
            raise typer.Exit(1)

        with open(metadata_file, "r") as f:
            metadata = yaml.safe_load(f)

        agent_id = metadata.get("id", agent_path.name)
        agent_version = tag or metadata.get("version", "0.1.0")

        # Display info
        info = {
            "Agent": agent_id,
            "Version": agent_version,
            "Registry": registry or load_config().get("registry", {}).get("url", "default"),
            "Mode": "DRY RUN" if dry_run else "PUBLISH",
        }
        console.print(create_info_panel("Publishing Information", info))
        console.print()

        # Confirm if not dry run
        if not dry_run and not force:
            if not confirm_action(f"Publish {agent_id}:{agent_version} to registry?", default=False):
                print_info("Publish cancelled")
                raise typer.Exit(0)

        # Package and publish
        with create_progress_bar() as progress:
            task1 = progress.add_task("Packaging agent...", total=100)
            package_path = package_agent(agent_path, agent_version)
            progress.update(task1, completed=100)

            if not dry_run:
                task2 = progress.add_task("Uploading to registry...", total=100)
                result = upload_to_registry(package_path, registry, force=force)
                progress.update(task2, completed=100)

        console.print()
        if dry_run:
            print_success(f"Dry run complete - agent would be published as {agent_id}:{agent_version}")
        else:
            print_success(f"Agent published successfully: {agent_id}:{agent_version}")
            console.print(f"\nPull with: [cyan]gl registry pull {agent_id}:{agent_version}[/cyan]")

    except Exception as e:
        print_error(f"Failed to publish agent: {str(e)}")
        raise typer.Exit(1)


@app.command()
def list(
    filter_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by agent type",
    ),
    filter_status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (active/inactive)",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json/yaml)",
    ),
):
    """
    List all agents in the local workspace.

    Example:
        gl agent list --type regulatory --status active
    """
    try:
        console.print("\n[bold cyan]Listing agents...[/bold cyan]\n")

        # Find all agents
        config = load_config()
        agents_dir = Path(get_config_value("defaults.output_dir", "agents", config))

        if not agents_dir.exists():
            print_warning("No agents directory found")
            raise typer.Exit(0)

        agents = []
        for agent_dir in agents_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            metadata_file = agent_dir / "agent.yaml"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = yaml.safe_load(f)

                # Apply filters
                if filter_type and metadata.get("type") != filter_type:
                    continue
                if filter_status and metadata.get("status") != filter_status:
                    continue

                agents.append({
                    "id": metadata.get("id", agent_dir.name),
                    "name": metadata.get("name", "Unknown"),
                    "version": metadata.get("version", "0.1.0"),
                    "type": metadata.get("type", "unknown"),
                    "status": metadata.get("status", "unknown"),
                    "updated": metadata.get("updated", "N/A"),
                })

        if not agents:
            print_info("No agents found")
            raise typer.Exit(0)

        # Display based on format
        if output_format == "table":
            table = create_agent_table(agents)
            console.print(table)
        elif output_format == "json":
            console.print_json(data=agents)
        elif output_format == "yaml":
            console.print(yaml.dump(agents, default_flow_style=False))

        console.print(f"\n[dim]Total agents: {len(agents)}[/dim]\n")

    except Exception as e:
        print_error(f"Failed to list agents: {str(e)}")
        raise typer.Exit(1)


@app.command()
def info(
    agent_id: str = typer.Argument(
        ...,
        help="Agent ID to show information for",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information",
    ),
):
    """
    Show detailed information about an agent.

    Example:
        gl agent info nfpa86-furnace-agent --verbose
    """
    try:
        console.print(f"\n[bold cyan]Agent Information:[/bold cyan] {agent_id}\n")

        # Find agent
        config = load_config()
        agents_dir = Path(get_config_value("defaults.output_dir", "agents", config))
        agent_dir = agents_dir / agent_id

        if not agent_dir.exists():
            print_error(f"Agent not found: {agent_id}")
            raise typer.Exit(1)

        # Load metadata
        metadata_file = agent_dir / "agent.yaml"
        if not metadata_file.exists():
            print_error(f"No agent.yaml found for {agent_id}")
            raise typer.Exit(1)

        with open(metadata_file, "r") as f:
            metadata = yaml.safe_load(f)

        # Display information
        info = {
            "ID": metadata.get("id", agent_id),
            "Name": metadata.get("name", "Unknown"),
            "Version": metadata.get("version", "0.1.0"),
            "Type": metadata.get("type", "unknown"),
            "Status": metadata.get("status", "unknown"),
            "Description": metadata.get("description", "No description"),
            "Created": metadata.get("created", "N/A"),
            "Updated": metadata.get("updated", "N/A"),
            "Path": str(agent_dir),
        }

        console.print(create_info_panel("Agent Details", info))

        if verbose and metadata.get("capabilities"):
            console.print("\n[bold]Capabilities:[/bold]")
            for cap in metadata["capabilities"]:
                console.print(f"  • {cap}")

        if verbose and metadata.get("dependencies"):
            console.print("\n[bold]Dependencies:[/bold]")
            for dep in metadata["dependencies"]:
                console.print(f"  • {dep}")

    except Exception as e:
        print_error(f"Failed to get agent info: {str(e)}")
        raise typer.Exit(1)


# Helper functions for agent generation

def validate_spec(spec: dict, strict: bool = False, verbose: bool = False) -> dict:
    """Validate an agent specification."""
    errors = []
    warnings = []

    # Check required fields
    if "metadata" not in spec:
        errors.append("Missing 'metadata' section")
    else:
        metadata = spec["metadata"]
        if "id" not in metadata:
            errors.append("Missing 'metadata.id'")
        if "name" not in metadata:
            errors.append("Missing 'metadata.name'")
        if "version" not in metadata:
            warnings.append("Missing 'metadata.version' (will default to 0.1.0)")

    # Check capabilities
    if "capabilities" not in spec:
        warnings.append("No capabilities defined")

    # Check architecture
    if "architecture" not in spec:
        warnings.append("No architecture specified")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def generate_core_agent(spec: dict, output_dir: Path, verbose: bool) -> list:
    """Generate core agent implementation files."""
    files = []

    # Create main agent file
    agent_file = output_dir / "agent.py"
    agent_code = f'''"""
{spec.get("metadata", {}).get("name", "Agent")}

Auto-generated by GreenLang Agent Factory
"""

class Agent:
    """Main agent class."""

    def __init__(self):
        self.id = "{spec.get("metadata", {}).get("id", "unknown")}"
        self.version = "{spec.get("metadata", {}).get("version", "0.1.0")}"

    def execute(self, task):
        """Execute a task."""
        pass
'''
    agent_file.write_text(agent_code)
    files.append(agent_file)

    return files


def generate_config_files(spec: dict, output_dir: Path, verbose: bool) -> list:
    """Generate configuration files."""
    files = []

    # Create agent.yaml
    config_file = output_dir / "agent.yaml"
    config = {
        "id": spec.get("metadata", {}).get("id", "unknown"),
        "name": spec.get("metadata", {}).get("name", "Unknown"),
        "version": spec.get("metadata", {}).get("version", "0.1.0"),
        "type": spec.get("metadata", {}).get("type", "unknown"),
        "status": "active",
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
    }
    config_file.write_text(yaml.dump(config, default_flow_style=False))
    files.append(config_file)

    return files


def generate_tests(spec: dict, output_dir: Path, verbose: bool) -> list:
    """Generate test files."""
    files = []

    test_dir = output_dir / "tests"
    test_dir.mkdir(exist_ok=True)

    # Create basic test
    test_file = test_dir / "test_agent.py"
    test_code = '''"""
Tests for agent
"""

def test_agent_creation():
    """Test agent creation."""
    pass
'''
    test_file.write_text(test_code)
    files.append(test_file)

    return files


def generate_documentation(spec: dict, output_dir: Path, verbose: bool) -> list:
    """Generate documentation files."""
    files = []

    # Create README
    readme_file = output_dir / "README.md"
    readme = f'''# {spec.get("metadata", {}).get("name", "Agent")}

{spec.get("metadata", {}).get("description", "No description provided")}

## Version

{spec.get("metadata", {}).get("version", "0.1.0")}

## Usage

```python
from agent import Agent

agent = Agent()
agent.execute(task)
```
'''
    readme_file.write_text(readme)
    files.append(readme_file)

    return files


def generate_deployment_configs(spec: dict, output_dir: Path, verbose: bool) -> list:
    """Generate deployment configuration files."""
    files = []

    # Create Dockerfile
    dockerfile = output_dir / "Dockerfile"
    dockerfile_content = '''FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "agent.py"]
'''
    dockerfile.write_text(dockerfile_content)
    files.append(dockerfile)

    return files


def run_agent_tests(agent_path: Path, verbose: bool, coverage: bool, parallel: bool) -> dict:
    """Run tests for an agent."""
    # Placeholder implementation
    return {
        "total": 10,
        "passed": 9,
        "failed": 1,
        "skipped": 0,
        "failures": [
            {
                "name": "test_example",
                "message": "AssertionError: Expected True but got False",
            }
        ],
    }


def package_agent(agent_path: Path, version: str) -> Path:
    """Package an agent for distribution."""
    # Placeholder implementation
    return agent_path / f"agent-{version}.tar.gz"


def upload_to_registry(package_path: Path, registry: Optional[str], force: bool) -> dict:
    """Upload package to registry."""
    # Placeholder implementation
    return {"success": True}


@app.command()
def certify(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output report file (supports .pdf, .html, .json, .md)",
    ),
    dimension: Optional[str] = typer.Option(
        None,
        "--dimension",
        "-d",
        help="Evaluate single dimension (e.g., D01, D02)",
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet",
        "-v/-q",
        help="Verbose output",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
):
    """
    Certify an agent against the 12-dimension certification framework.

    This command evaluates an agent across all certification dimensions:
    - D01: Determinism (100 runs, byte-identical outputs)
    - D02: Provenance (SHA-256 hash generation)
    - D03: Zero-Hallucination (no LLM in calculations)
    - D04: Accuracy (golden test pass rate)
    - D05: Source Verification (traceable emission factors)
    - D06: Unit Consistency (input/output validation)
    - D07: Regulatory Compliance (GHG Protocol, ISO 14064)
    - D08: Security (no secrets, injection prevention)
    - D09: Performance (response time, memory)
    - D10: Documentation (docstrings, API docs)
    - D11: Test Coverage (>90% coverage)
    - D12: Production Readiness (logging, health checks)

    Certification Levels:
    - GOLD: 100% score, all required dimensions pass
    - SILVER: 95%+ score, all required dimensions pass
    - BRONZE: 85%+ score, all required dimensions pass
    - FAIL: Below 85% or required dimension fails

    Example:
        gl agent certify agents/carbon_calculator
        gl agent certify agents/carbon_calculator --output report.pdf
        gl agent certify agents/carbon_calculator --dimension D01
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from backend.certification.engine import CertificationEngine
        from backend.certification.report import ReportGenerator

        engine = CertificationEngine()

        # Single dimension evaluation
        if dimension:
            console.print(f"\n[cyan]Evaluating dimension {dimension} for:[/cyan] {agent_path}\n")

            try:
                result = engine.evaluate_single_dimension(agent_path, dimension.upper())

                status_color = "green" if result.passed else "red"
                status_text = "PASS" if result.passed else "FAIL"

                console.print(create_info_panel(f"Dimension {result.dimension_id}", {
                    "Name": result.dimension_name,
                    "Status": f"[{status_color}]{status_text}[/{status_color}]",
                    "Score": f"{result.score:.1f}/100",
                    "Checks": f"{result.checks_passed}/{result.checks_total} passed",
                    "Time": f"{result.execution_time_ms:.2f}ms",
                }))

                if verbose and result.check_results:
                    console.print("\n[bold]Check Results:[/bold]")
                    for check in result.check_results:
                        icon = "[green]v[/green]" if check.passed else "[red]x[/red]"
                        console.print(f"  {icon} {check.name}: {check.message}")

                if result.remediation:
                    console.print("\n[bold yellow]Remediation:[/bold yellow]")
                    for suggestion in result.remediation[:3]:
                        console.print(f"  - {suggestion[:100]}...")

                if not result.passed:
                    raise typer.Exit(1)

            except ValueError as e:
                print_error(str(e))
                raise typer.Exit(1)

            return

        # Full certification
        if not json_output:
            console.print("\n[bold cyan]GreenLang Agent Certification[/bold cyan]")
            console.print("[dim]12-Dimension Evaluation Framework[/dim]\n")
            console.print(f"Agent: [cyan]{agent_path}[/cyan]\n")

        with console.status("[bold green]Running certification..."):
            result = engine.evaluate_agent(agent_path, verbose=False)

        # Output results
        if json_output:
            import json
            console.print_json(data=result.to_dict())
        else:
            _print_certification_results(result)

        # Generate report if requested
        if output:
            generator = ReportGenerator()
            suffix = output.suffix.lower()

            if suffix == ".pdf":
                generator.generate_pdf(result, output)
            elif suffix == ".html":
                generator.generate_html(result, output)
            elif suffix == ".json":
                generator.generate_json(result, output)
            elif suffix in [".md", ".markdown"]:
                generator.generate_markdown(result, output)
            else:
                generator.generate_html(result, output)

            print_success(f"Report generated: {output}")

        # Exit with appropriate code
        if not result.certified:
            raise typer.Exit(1)

    except ImportError as e:
        print_error(f"Certification module not available: {str(e)}")
        print_info("Ensure backend/certification is in your Python path")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Certification failed: {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _print_certification_results(result) -> None:
    """Print certification results."""
    from backend.certification.dimensions.base import DimensionStatus

    # Summary
    status_color = "green" if result.certified else "red"
    status_text = "CERTIFIED" if result.certified else "NOT CERTIFIED"

    console.print(create_info_panel("Certification Summary", {
        "Agent ID": result.agent_id,
        "Version": result.agent_version,
        "Certification ID": result.certification_id,
        "Status": f"[{status_color}]{status_text}[/{status_color}]",
        "Level": f"[{status_color}]{result.level.value}[/{status_color}]",
        "Overall Score": f"{result.overall_score:.1f}/100",
        "Weighted Score": f"{result.weighted_score:.1f}/100",
        "Dimensions": f"{result.dimensions_passed}/{result.dimensions_total} passed",
    }))

    # Dimension results table
    from rich.table import Table

    table = Table(title="Dimension Results")
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Dimension", width=25)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Checks", justify="right", width=10)

    for dim in result.dimension_results:
        if dim.status == DimensionStatus.PASS:
            status = "[green]PASS[/green]"
        elif dim.status == DimensionStatus.FAIL:
            status = "[red]FAIL[/red]"
        elif dim.status == DimensionStatus.WARNING:
            status = "[yellow]WARN[/yellow]"
        else:
            status = "[dim]N/A[/dim]"

        table.add_row(
            dim.dimension_id,
            dim.dimension_name,
            status,
            f"{dim.score:.1f}",
            f"{dim.checks_passed}/{dim.checks_total}",
        )

    console.print()
    console.print(table)

    # Remediation if needed
    if result.remediation_summary:
        console.print("\n[bold yellow]Remediation Required:[/bold yellow]")
        for dimension, suggestions in result.remediation_summary.items():
            console.print(f"\n  [cyan]{dimension}[/cyan]")
            for suggestion in suggestions[:2]:
                console.print(f"    - {suggestion[:80]}...")

    console.print()


@app.command("certify-dimensions")
def certify_dimensions():
    """
    List all certification dimensions.

    Displays information about each of the 12 certification dimensions
    including their requirements and weights.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from backend.certification.engine import CertificationEngine

        engine = CertificationEngine()
        dims = engine.get_dimension_info()

        from rich.table import Table

        table = Table(title="GreenLang Certification Dimensions")
        table.add_column("ID", style="cyan", width=6)
        table.add_column("Name", style="green", width=25)
        table.add_column("Weight", justify="right", width=8)
        table.add_column("Required", justify="center", width=10)
        table.add_column("Description", width=45)

        for dim in dims:
            required = "[green]Yes[/green]" if dim["required"] else "[yellow]No[/yellow]"
            desc = dim["description"][:45] + "..." if len(dim["description"]) > 45 else dim["description"]
            table.add_row(
                dim["id"],
                dim["name"],
                f"{dim['weight']:.1f}",
                required,
                desc,
            )

        console.print()
        console.print(table)
        console.print()

    except ImportError as e:
        print_error(f"Certification module not available: {str(e)}")
        raise typer.Exit(1)
