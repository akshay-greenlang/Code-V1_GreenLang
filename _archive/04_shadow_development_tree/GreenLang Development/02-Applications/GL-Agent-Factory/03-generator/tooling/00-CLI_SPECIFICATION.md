# Agent Generator CLI Specification

**Version**: 1.0.0
**Status**: Design
**Last Updated**: 2025-12-03
**Owner**: GL Backend Developer

---

## Executive Summary

This document defines the complete CLI interface for the GreenLang Agent Generator. The CLI provides intuitive commands for creating, validating, testing, updating, and publishing agent packs from AgentSpec v2 YAML definitions.

**CLI Tool**: `gl agent`
**Framework**: Typer (Python)
**Features**: Rich formatting, progress bars, interactive prompts, dry-run mode

---

## 1. CLI Architecture

### 1.1 Command Structure

```
gl agent
├── create          Create new agent pack from spec
├── update          Update existing agent pack
├── validate        Validate AgentSpec YAML
├── test            Run agent test suite
├── publish         Publish agent pack to registry
├── list            List available agents
├── info            Show agent information
└── init            Initialize new AgentSpec template
```

### 1.2 Implementation Framework

```python
# cli/agent_commands.py

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="agent",
    help="GreenLang Agent Generator CLI",
    add_completion=True
)

console = Console()
```

---

## 2. Command: `gl agent create`

### 2.1 Description

Create a new agent pack from AgentSpec v2 YAML.

### 2.2 Signature

```bash
gl agent create [OPTIONS] SPEC_PATH
```

### 2.3 Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `SPEC_PATH` | Path | Yes | Path to AgentSpec v2 YAML file |

### 2.4 Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output`, `-o` | Path | `./agents/{agent_name}` | Output directory for agent pack |
| `--overwrite` | Flag | False | Overwrite existing pack if present |
| `--skip-tests` | Flag | False | Skip running generated tests |
| `--dry-run` | Flag | False | Preview generation without writing files |
| `--template` | Path | None | Use custom template directory |
| `--verbose`, `-v` | Flag | False | Enable verbose logging |
| `--quiet`, `-q` | Flag | False | Suppress non-error output |

### 2.5 Examples

```bash
# Basic usage
gl agent create specs/fuel_agent.yaml

# Specify output directory
gl agent create specs/fuel_agent.yaml --output agents/fuel_agent

# Overwrite existing pack
gl agent create specs/fuel_agent.yaml --overwrite

# Dry run (preview only)
gl agent create specs/fuel_agent.yaml --dry-run

# Use custom templates
gl agent create specs/fuel_agent.yaml --template ./custom_templates

# Verbose output
gl agent create specs/fuel_agent.yaml -v
```

### 2.6 Implementation

```python
@app.command("create")
def create_agent(
    spec_path: Path = typer.Argument(
        ...,
        help="Path to AgentSpec v2 YAML file",
        exists=True,
        dir_okay=False,
        readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for agent pack"
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing pack if present"
    ),
    skip_tests: bool = typer.Option(
        False,
        "--skip-tests",
        help="Skip running generated tests"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview generation without writing files"
    ),
    template: Optional[Path] = typer.Option(
        None,
        "--template",
        help="Use custom template directory"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress non-error output"
    )
):
    """
    Create new agent pack from AgentSpec v2 YAML.

    Generates:
    - Agent class with type-safe I/O models
    - Tool wrappers for calculators
    - Complete test suite (unit, integration, determinism)
    - Documentation (README, ARCHITECTURE, API)
    - Deployment configs (Docker, K8s)
    - Monitoring configs (Grafana, Prometheus)

    Example:
        gl agent create specs/fuel_agent.yaml
        gl agent create specs/fuel_agent.yaml --output agents/fuel --overwrite
    """
    # Setup logging
    if verbose:
        setup_logging(level=logging.DEBUG)
    elif quiet:
        setup_logging(level=logging.ERROR)
    else:
        setup_logging(level=logging.INFO)

    try:
        if dry_run:
            # Dry run mode
            console.print("[yellow]Dry run mode - no files will be written[/yellow]")
            result = create_agent_dry_run(spec_path)

            if result.success:
                console.print(f"[green]✓[/green] Validation passed")
                console.print(f"\n[bold]Would generate:[/bold]")
                console.print(f"  Files: {len(result.files)}")
                console.print(f"  Lines of code: {result.lines_of_code}")
                console.print(f"\n[bold]Files:[/bold]")
                for file_path in result.files:
                    console.print(f"  - {file_path}")
            else:
                console.print("[red]✗[/red] Validation failed")
                for error in result.errors:
                    console.print(f"  [red]•[/red] {error}")
                raise typer.Exit(code=1)

        else:
            # Real generation
            console.print(f"[bold]Creating agent pack from {spec_path}[/bold]\n")

            # Determine output directory
            if output is None:
                spec_data = load_spec(spec_path)
                agent_name = spec_data.get('id', 'agent').replace('/', '_')
                output = Path(f"./agents/{agent_name}")

            # Check for existing pack
            if output.exists() and not overwrite:
                console.print(f"[red]Error:[/red] Pack already exists at {output}")
                console.print("Use --overwrite to replace existing pack")
                raise typer.Exit(code=1)

            # Run creation workflow with progress
            pack = create_agent_with_progress(
                spec_path=spec_path,
                output_dir=output,
                skip_tests=skip_tests,
                template_dir=template
            )

            # Success message
            console.print(f"\n[green]✓[/green] Agent pack created successfully!")
            console.print(f"\n[bold]Location:[/bold] {pack.pack_dir}")
            console.print(f"[bold]Agent ID:[/bold] {pack.spec_id}")
            console.print(f"[bold]Version:[/bold] {pack.version}")
            console.print(f"\n[bold]Next steps:[/bold]")
            console.print(f"  1. cd {pack.pack_dir}")
            console.print(f"  2. pip install -r requirements.txt")
            console.print(f"  3. pytest tests/")

    except ValidationError as e:
        console.print(f"[red]✗ Validation failed:[/red]")
        for error in e.errors:
            console.print(f"  [red]•[/red] {error}")
        raise typer.Exit(code=1)

    except GenerationError as e:
        console.print(f"[red]✗ Generation failed:[/red] {e}")
        raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]✗ Unexpected error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)
```

### 2.7 Output Format

```
Creating agent pack from specs/fuel_agent.yaml

⠹ Loading spec...
✓ Loading spec...
⠹ Validating spec...
✓ Validating spec...
⠹ Generating code...
✓ Generating code...
⠹ Running tests...
✓ Running tests...
⠹ Generating docs...
✓ Generating docs...

✓ Agent pack created successfully!

Location: ./agents/emissions_fuel_agent_v1
Agent ID: emissions/fuel_agent_v1
Version: 1.0.0

Files generated:
  - agent.py (245 lines)
  - tools.py (180 lines)
  - tests/test_agent.py (320 lines)
  - README.md (150 lines)
  - Total: 12 files, 1,245 lines

Next steps:
  1. cd ./agents/emissions_fuel_agent_v1
  2. pip install -r requirements.txt
  3. pytest tests/
```

---

## 3. Command: `gl agent update`

### 3.1 Description

Update existing agent pack from modified AgentSpec.

### 3.2 Signature

```bash
gl agent update [OPTIONS] PACK_DIR
```

### 3.3 Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `PACK_DIR` | Path | Yes | Path to existing agent pack directory |

### 3.4 Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--spec` | Path | `{PACK_DIR}/pack.yaml` | Path to updated AgentSpec |
| `--backup` | Flag | True | Backup existing pack before update |
| `--no-backup` | Flag | False | Skip backup |
| `--diff` | Flag | False | Show diff before applying |
| `--verbose`, `-v` | Flag | False | Enable verbose logging |

### 3.5 Examples

```bash
# Update from pack.yaml
gl agent update agents/fuel_agent

# Update from external spec
gl agent update agents/fuel_agent --spec specs/fuel_agent_v2.yaml

# Show diff before updating
gl agent update agents/fuel_agent --diff

# Update without backup
gl agent update agents/fuel_agent --no-backup
```

### 3.6 Implementation

```python
@app.command("update")
def update_agent(
    pack_dir: Path = typer.Argument(
        ...,
        help="Path to existing agent pack directory",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    spec: Optional[Path] = typer.Option(
        None,
        "--spec",
        help="Path to updated AgentSpec (defaults to pack.yaml)"
    ),
    backup: bool = typer.Option(
        True,
        "--backup/--no-backup",
        help="Backup existing pack before update"
    ),
    diff: bool = typer.Option(
        False,
        "--diff",
        help="Show diff before applying"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """
    Update existing agent pack from modified AgentSpec.

    Detects changes between existing and new spec, regenerates
    affected files, and merges custom code where possible.

    Example:
        gl agent update agents/fuel_agent
        gl agent update agents/fuel_agent --spec specs/fuel_v2.yaml --diff
    """
    # Setup logging
    if verbose:
        setup_logging(level=logging.DEBUG)

    try:
        console.print(f"[bold]Updating agent pack at {pack_dir}[/bold]\n")

        # Determine spec path
        if spec is None:
            spec = pack_dir / "pack.yaml"

        if not spec.exists():
            console.print(f"[red]Error:[/red] Spec file not found: {spec}")
            raise typer.Exit(code=1)

        # Run update workflow
        result = update_agent_workflow(
            spec_path=spec,
            pack_dir=pack_dir,
            backup=backup
        )

        if not result.changes:
            console.print("[green]✓[/green] No changes detected")
            return

        # Show changes
        console.print(f"[bold]Changes detected:[/bold]")
        for change in result.changes:
            console.print(f"  [yellow]•[/yellow] {change.description}")

        # Show diff if requested
        if diff:
            console.print(f"\n[bold]Diff:[/bold]")
            for change in result.changes:
                console.print(change.diff)

            # Confirm before applying
            if not typer.confirm("Apply these changes?"):
                console.print("[yellow]Update cancelled[/yellow]")
                return

        console.print(f"\n[green]✓[/green] Agent pack updated successfully!")

        if backup:
            console.print(f"[dim]Backup saved to: {result.backup_dir}[/dim]")

    except UpdateError as e:
        console.print(f"[red]✗ Update failed:[/red] {e}")
        raise typer.Exit(code=1)
```

---

## 4. Command: `gl agent validate`

### 4.1 Description

Validate AgentSpec v2 YAML against schema and business rules.

### 4.2 Signature

```bash
gl agent validate [OPTIONS] SPEC_PATH
```

### 4.3 Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--strict` | Flag | False | Enable strict validation (fail on warnings) |
| `--fix` | Flag | False | Attempt to auto-fix common issues |
| `--format` | Choice | `text` | Output format: text, json, yaml |
| `--verbose`, `-v` | Flag | False | Show detailed validation report |

### 4.4 Examples

```bash
# Basic validation
gl agent validate specs/fuel_agent.yaml

# Strict validation (fail on warnings)
gl agent validate specs/fuel_agent.yaml --strict

# Auto-fix common issues
gl agent validate specs/fuel_agent.yaml --fix

# JSON output
gl agent validate specs/fuel_agent.yaml --format json

# Verbose report
gl agent validate specs/fuel_agent.yaml -v
```

### 4.5 Implementation

```python
@app.command("validate")
def validate_agent(
    spec_path: Path = typer.Argument(
        ...,
        help="Path to AgentSpec v2 YAML file",
        exists=True
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail on warnings"
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Attempt to auto-fix common issues"
    ),
    format: str = typer.Option(
        "text",
        "--format",
        help="Output format: text, json, yaml"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed validation report"
    )
):
    """
    Validate AgentSpec v2 YAML.

    Validation stages:
    1. YAML syntax
    2. AgentSpec v2 schema
    3. Semantic rules
    4. Dependencies
    5. Compliance
    6. Best practices

    Example:
        gl agent validate specs/fuel_agent.yaml
        gl agent validate specs/fuel_agent.yaml --strict --verbose
    """
    try:
        console.print(f"[bold]Validating {spec_path}[/bold]\n")

        # Run validation workflow
        report = validate_workflow(spec_path)

        # Display results based on format
        if format == "json":
            print(report.to_json())
        elif format == "yaml":
            print(report.to_yaml())
        else:
            # Text format
            display_validation_report(report, verbose=verbose)

        # Exit code
        if report.has_errors():
            raise typer.Exit(code=1)
        elif strict and report.has_warnings():
            console.print("\n[yellow]Warnings present (strict mode)[/yellow]")
            raise typer.Exit(code=1)
        else:
            console.print("\n[green]✓ Validation passed[/green]")

    except Exception as e:
        console.print(f"[red]✗ Validation failed:[/red] {e}")
        raise typer.Exit(code=1)


def display_validation_report(report: ValidationReport, verbose: bool = False):
    """Display validation report in text format."""
    # Summary
    table = Table(title="Validation Summary")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    for stage in report.stages:
        status = "[green]✓[/green]" if stage.passed else "[red]✗[/red]"
        table.add_row(stage.name, status, stage.message)

    console.print(table)

    # Errors
    if report.has_errors():
        console.print("\n[bold red]Errors:[/bold red]")
        for error in report.errors:
            console.print(f"  [red]•[/red] {error}")

    # Warnings
    if report.has_warnings():
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in report.warnings:
            console.print(f"  [yellow]•[/yellow] {warning}")

    # Detailed report (if verbose)
    if verbose and report.details:
        console.print("\n[bold]Detailed Report:[/bold]")
        console.print(report.details)
```

---

## 5. Command: `gl agent test`

### 5.1 Description

Run test suite for agent pack.

### 5.2 Signature

```bash
gl agent test [OPTIONS] PACK_DIR
```

### 5.3 Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--coverage` | Flag | True | Generate coverage report |
| `--markers`, `-m` | String | None | Run tests with specific markers |
| `--verbose`, `-v` | Flag | False | Verbose pytest output |
| `--html-report` | Flag | False | Generate HTML coverage report |

### 5.4 Examples

```bash
# Run all tests
gl agent test agents/fuel_agent

# Run only unit tests
gl agent test agents/fuel_agent -m unit

# Run with coverage
gl agent test agents/fuel_agent --coverage

# Generate HTML report
gl agent test agents/fuel_agent --html-report
```

### 5.5 Implementation

```python
@app.command("test")
def test_agent(
    pack_dir: Path = typer.Argument(
        ...,
        help="Path to agent pack directory",
        exists=True
    ),
    coverage: bool = typer.Option(
        True,
        "--coverage/--no-coverage",
        help="Generate coverage report"
    ),
    markers: Optional[str] = typer.Option(
        None,
        "--markers", "-m",
        help="Run tests with specific markers (unit, integration, etc.)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose pytest output"
    ),
    html_report: bool = typer.Option(
        False,
        "--html-report",
        help="Generate HTML coverage report"
    )
):
    """
    Run test suite for agent pack.

    Test categories:
    - unit: Unit tests for agent methods
    - integration: Integration tests with real calculators
    - determinism: Determinism verification tests
    - performance: Performance benchmarks

    Example:
        gl agent test agents/fuel_agent
        gl agent test agents/fuel_agent -m unit --coverage
    """
    try:
        console.print(f"[bold]Running tests for {pack_dir}[/bold]\n")

        # Run test workflow
        report = test_workflow(
            pack_dir=pack_dir,
            coverage=coverage,
            markers=markers,
            verbose=verbose,
            html_report=html_report
        )

        # Display results
        display_test_report(report)

        # Exit code
        if not report.all_passed():
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]✗ Tests failed:[/red] {e}")
        raise typer.Exit(code=1)


def display_test_report(report: TestReport):
    """Display test report."""
    # Summary table
    table = Table(title="Test Results")
    table.add_column("Category", style="cyan")
    table.add_column("Passed", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Skipped", style="yellow")
    table.add_column("Duration")

    for category, result in report.results.items():
        table.add_row(
            category,
            str(result.passed),
            str(result.failed),
            str(result.skipped),
            f"{result.duration:.2f}s"
        )

    console.print(table)

    # Coverage
    if report.coverage:
        console.print(f"\n[bold]Coverage:[/bold] {report.coverage.percentage:.1f}%")
        if report.coverage.percentage < 85:
            console.print("[yellow]Warning: Coverage below 85% target[/yellow]")
```

---

## 6. Command: `gl agent publish`

### 6.1 Description

Publish agent pack to registry.

### 6.2 Signature

```bash
gl agent publish [OPTIONS] PACK_DIR
```

### 6.3 Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--registry` | String | `https://registry.greenlang.io` | Registry URL |
| `--dry-run` | Flag | False | Preview without publishing |
| `--tag` | String | None | Add tag to package |
| `--private` | Flag | False | Publish as private package |

### 6.4 Examples

```bash
# Publish to default registry
gl agent publish agents/fuel_agent

# Dry run
gl agent publish agents/fuel_agent --dry-run

# Publish with tag
gl agent publish agents/fuel_agent --tag production

# Publish to custom registry
gl agent publish agents/fuel_agent --registry https://custom.registry.io
```

### 6.5 Implementation

```python
@app.command("publish")
def publish_agent(
    pack_dir: Path = typer.Argument(
        ...,
        help="Path to agent pack directory",
        exists=True
    ),
    registry: str = typer.Option(
        "https://registry.greenlang.io",
        "--registry",
        help="Registry URL"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview without publishing"
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        help="Add tag to package"
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Publish as private package"
    )
):
    """
    Publish agent pack to registry.

    Publishing stages:
    1. Validate pack completeness
    2. Run all tests
    3. Generate SBOM
    4. Sign package
    5. Upload to registry

    Example:
        gl agent publish agents/fuel_agent
        gl agent publish agents/fuel_agent --dry-run
    """
    try:
        console.print(f"[bold]Publishing {pack_dir} to {registry}[/bold]\n")

        # Run publish workflow
        result = publish_workflow(
            pack_dir=pack_dir,
            registry_url=registry,
            dry_run=dry_run,
            tag=tag,
            private=private
        )

        if dry_run:
            console.print("[green]✓[/green] Dry run complete")
            console.print(f"\nWould publish:")
            console.print(f"  Pack: {result.pack_id}")
            console.print(f"  Tag: {tag or 'latest'}")
            console.print(f"  Visibility: {'private' if private else 'public'}")
        else:
            console.print(f"[green]✓[/green] Published successfully!")
            console.print(f"\nPack URL: {result.pack_url}")
            console.print(f"Pack ID: {result.pack_id}")

    except PublishError as e:
        console.print(f"[red]✗ Publish failed:[/red] {e}")
        raise typer.Exit(code=1)
```

---

## 7. Command: `gl agent list`

### 7.1 Description

List available agents in local workspace or registry.

### 7.2 Signature

```bash
gl agent list [OPTIONS]
```

### 7.3 Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--local` | Flag | True | List local agents |
| `--registry` | String | None | List agents from registry |
| `--tag` | String | None | Filter by tag |
| `--format` | Choice | `table` | Output format: table, json, yaml |

### 7.4 Examples

```bash
# List local agents
gl agent list

# List from registry
gl agent list --registry https://registry.greenlang.io

# Filter by tag
gl agent list --tag emissions

# JSON output
gl agent list --format json
```

---

## 8. Command: `gl agent info`

### 8.1 Description

Show detailed information about an agent.

### 8.2 Signature

```bash
gl agent info [OPTIONS] PACK_DIR
```

### 8.3 Examples

```bash
# Show agent info
gl agent info agents/fuel_agent

# Show as JSON
gl agent info agents/fuel_agent --format json
```

---

## 9. Command: `gl agent init`

### 9.1 Description

Initialize new AgentSpec template with interactive prompts.

### 9.2 Signature

```bash
gl agent init [OPTIONS]
```

### 9.3 Implementation

```python
@app.command("init")
def init_agent():
    """
    Initialize new AgentSpec template (interactive).

    Creates a starter AgentSpec v2 YAML with prompts for:
    - Agent name and ID
    - Input/output fields
    - Tool selections
    - Compliance requirements
    """
    console.print("[bold]Initialize new AgentSpec v2[/bold]\n")

    # Interactive prompts
    agent_id = typer.prompt("Agent ID (e.g., emissions/fuel_v1)")
    agent_name = typer.prompt("Agent name")
    summary = typer.prompt("Summary (one line)")

    # Deterministic?
    deterministic = typer.confirm("Is this agent deterministic?", default=True)

    # Generate template
    template = generate_agentspec_template(
        agent_id=agent_id,
        name=agent_name,
        summary=summary,
        deterministic=deterministic
    )

    # Write template
    output_path = Path(f"specs/{agent_id.replace('/', '_')}.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template)

    console.print(f"\n[green]✓[/green] AgentSpec template created: {output_path}")
    console.print(f"\nNext steps:")
    console.print(f"  1. Edit {output_path}")
    console.print(f"  2. gl agent validate {output_path}")
    console.print(f"  3. gl agent create {output_path}")
```

---

## 10. Global Options

### 10.1 All Commands Support

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--version` | Show version |
| `--config` | Path to config file |
| `--log-level` | Set log level: DEBUG, INFO, WARNING, ERROR |
| `--no-color` | Disable colored output |

---

## 11. Configuration File

### 11.1 Location

```
~/.greenlang/config.yaml
```

### 11.2 Example Config

```yaml
# GreenLang CLI Configuration

# Default registry
registry_url: "https://registry.greenlang.io"

# Default template directory
template_dir: "~/.greenlang/templates"

# Default output directory
output_dir: "./agents"

# Generator settings
generator:
  skip_tests: false
  coverage_threshold: 85
  type_check: true
  lint: true

# CLI preferences
cli:
  color: true
  progress_bar: true
  verbose: false
```

---

## 12. Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Validation error |
| 2 | Generation error |
| 3 | Test failure |
| 4 | Publish error |
| 127 | Unexpected error |

---

## 13. Shell Completion

### 13.1 Install Completion

```bash
# Bash
gl --install-completion bash

# Zsh
gl --install-completion zsh

# Fish
gl --install-completion fish
```

---

## Summary

The CLI provides comprehensive commands for:

1. **create**: Generate new agent packs with rich progress
2. **update**: Update existing packs with diff preview
3. **validate**: Multi-stage validation with auto-fix
4. **test**: Run test suites with coverage reports
5. **publish**: Publish to registry with signing
6. **list**: Browse available agents
7. **info**: Show detailed agent information
8. **init**: Interactive spec template creation

**Features**:
- Rich terminal UI with progress bars
- Interactive prompts
- Dry-run mode
- Multiple output formats (text, JSON, YAML)
- Comprehensive error messages
- Shell completion

---

**Document Status**: Design Complete
**Implementation Status**: Pending
**Next Step**: Implement CLI with Typer
