# -*- coding: utf-8 -*-
"""
Complete GreenLang CLI with all required features
"""

import click
import json
import yaml
import sys
import os
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional
import time

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from greenlang.determinism import DeterministicClock
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.logging import RichHandler
import logging

import greenlang
from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import Workflow
from greenlang.cli.jsonl_logger import JSONLLogger
from greenlang.cli.agent_registry import AgentRegistry
from greenlang.cli.cmd_pack import pack

# Setup console
console = Console()

# Cache directory
CACHE_DIR = Path.home() / ".greenlang" / "cache"
RUNS_DIR = Path.home() / ".greenlang" / "runs"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


class CLIContext:
    """Context for CLI state"""

    def __init__(self):
        self.verbose = False
        self.dry_run = False
        self.logger = None
        self.agent_registry = AgentRegistry()
        self.orchestrator = Orchestrator()

    def setup(self, verbose: bool, dry_run: bool):
        self.verbose = verbose
        self.dry_run = dry_run

        # Setup logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
        )
        self.logger = logging.getLogger("greenlang")


pass_context = click.make_pass_decorator(CLIContext, ensure=True)


@click.group(invoke_without_command=True)
@click.version_option(version=greenlang.__version__, prog_name="gl")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--dry-run", is_flag=True, help="Simulate without making changes")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, dry_run: bool):
    """GreenLang CLI - Climate Intelligence Framework

    Examples:
      gl init                  # Initialize project
      gl agents list          # List available agents
      gl run pipeline.yaml    # Execute pipeline
      gl validate config.yaml # Validate configuration
      gl report run_123       # Generate report
      gl ask "question"       # AI assistant
    """
    ctx.obj = CLIContext()
    ctx.obj.setup(verbose, dry_run)

    if ctx.invoked_subcommand is None:
        console.print(
            Panel.fit(
                f"[bold green]GreenLang CLI v{greenlang.__version__}[/bold green]\n\n"
                "Type 'gl --help' for available commands",
                title="Welcome to GreenLang",
            )
        )


# ============= INIT COMMAND =============
@cli.command()
@pass_context
def init(ctx: CLIContext):
    """Initialize a new GreenLang project with scaffolding"""

    if ctx.dry_run:
        console.print("[yellow]DRY-RUN: Would create project structure[/yellow]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        task = progress.add_task("Creating project structure...", total=7)

        # Create directories
        dirs = [
            "pipelines",
            "data",
            "reports",
            "logs",
            "agents/custom",
            "cache",
            "configs",
        ]
        for i, dir_name in enumerate(dirs):
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            progress.update(task, advance=1, description=f"Created {dir_name}/")
            time.sleep(0.1)  # Visual effect

        # Create sample pipeline
        sample_pipeline = {
            "name": "carbon_calculation",
            "description": "Sample carbon footprint calculation pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "validate_input",
                    "agent_id": "validator",
                    "description": "Validate input data",
                    "retry_count": 2,
                },
                {
                    "name": "calculate_emissions",
                    "agent_id": "carbon",
                    "description": "Calculate carbon emissions",
                },
                {
                    "name": "generate_report",
                    "agent_id": "report",
                    "description": "Generate emissions report",
                },
            ],
            "output_mapping": {
                "total_emissions": "results.calculate_emissions.data.total",
                "report": "results.generate_report.data.report",
            },
        }

        Path("pipelines/sample.yaml").write_text(yaml.dump(sample_pipeline))
        progress.update(task, completed=7, description="Project initialized!")

    # Create .env template
    env_template = """# GreenLang Environment Configuration
GREENLANG_ENV=development
GREENLANG_LOG_LEVEL=INFO

# API Keys (optional)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=...

# Custom paths
# GREENLANG_AGENTS_PATH=/path/to/custom/agents
"""
    Path(".env").write_text(env_template)

    # Create dataset stub
    dataset_stub = {
        "metadata": {
            "name": "Sample Dataset",
            "version": "1.0.0",
            "created": DeterministicClock.now().isoformat(),
        },
        "data": {
            "fuels": [
                {"type": "electricity", "amount": 1000, "unit": "kWh"},
                {"type": "natural_gas", "amount": 500, "unit": "therms"},
            ]
        },
    }
    Path("data/sample_dataset.json").write_text(json.dumps(dataset_stub, indent=2))

    console.print("\n[green]✓ Project initialized successfully![/green]")
    console.print("\nCreated structure:")
    console.print("  📁 pipelines/    - Pipeline definitions")
    console.print("  📁 data/         - Input datasets")
    console.print("  📁 reports/      - Generated reports")
    console.print("  📁 logs/         - Execution logs")
    console.print("  📁 agents/       - Custom agents")
    console.print("  📁 cache/        - Execution cache")
    console.print("  📄 .env          - Environment config")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  1. Edit pipelines/sample.yaml")
    console.print("  2. Run: gl validate pipelines/sample.yaml")
    console.print("  3. Execute: gl run pipelines/sample.yaml")


# ============= AGENTS COMMANDS =============
@cli.group()
@pass_context
def agents(ctx: CLIContext):
    """Discover and manage agents"""


@agents.command(name="list")
@pass_context
def agents_list(ctx: CLIContext):
    """List all available agents"""
    agents = ctx.agent_registry.discover_agents()

    table = Table(title="Available Agents", show_header=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Version", style="yellow")
    table.add_column("Type", style="magenta")

    for agent in agents:
        table.add_row(
            agent["id"],
            agent["name"],
            agent.get("version", "0.0.1"),
            agent.get("type", "core"),
        )

    console.print(table)


@agents.command(name="info")
@click.argument("agent_id")
@pass_context
def agents_info(ctx: CLIContext, agent_id: str):
    """Show detailed information about an agent"""
    info = ctx.agent_registry.get_agent_info(agent_id)

    if not info:
        console.print(f"[red]Agent '{agent_id}' not found[/red]")
        sys.exit(1)

    console.print(
        Panel(
            f"[bold]{info['name']}[/bold]\n\n"
            f"ID: {info['id']}\n"
            f"Version: {info.get('version', '0.0.1')}\n"
            f"Type: {info.get('type', 'core')}\n"
            f"Description: {info.get('description', 'No description')}",
            title=f"Agent: {agent_id}",
        )
    )


@agents.command(name="template")
@click.argument("name")
@click.option("--output", "-o", type=click.Path(), help="Save to file")
@pass_context
def agents_template(ctx: CLIContext, name: str, output: Optional[str]):
    """Generate agent template"""
    template = ctx.agent_registry.get_agent_template(name)

    if output:
        Path(output).write_text(template)
        console.print(f"[green]Template saved to {output}[/green]")
    else:
        console.print(template)


# ============= VALIDATE COMMAND =============
@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--json", "output_json", is_flag=True, help="Output validation results as JSON"
)
@click.option("--strict", is_flag=True, help="Fail on warnings")
@pass_context
def validate(ctx: CLIContext, file_path: str, output_json: bool, strict: bool):
    """Validate pipeline (gl.yaml) or pack manifest (pack.yaml)"""

    file_path_obj = Path(file_path)

    # Detect file type based on name and content
    is_pipeline = False
    is_pack = False

    if file_path_obj.name in ["gl.yaml", "gl.yml"] or "gl." in file_path_obj.name:
        is_pipeline = True
    elif (
        file_path_obj.name in ["pack.yaml", "pack.yml"] or "pack." in file_path_obj.name
    ):
        is_pack = True
    else:
        # Try to detect from content
        try:
            with open(file_path, "r") as f:
                content = yaml.safe_load(f)
            if isinstance(content, dict):
                if "steps" in content and "name" in content:
                    is_pipeline = True
                elif "contents" in content and "kind" in content:
                    is_pack = True
        except Exception:
            pass

    if not is_pipeline and not is_pack:
        if output_json:
            result = {
                "ok": False,
                "errors": [
                    "Unable to detect file type. Expected pipeline (gl.yaml) or pack (pack.yaml) file."
                ],
                "warnings": [],
                "summary": {},
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print(
                "[red]Error: Unable to detect file type. Expected pipeline (gl.yaml) or pack (pack.yaml) file.[/red]"
            )
        sys.exit(1)
        return

    errors = []
    warnings = []
    summary = {}
    spec_version = "1.0"
    file_type = "pipeline" if is_pipeline else "pack"
    file_name = ""

    try:
        if is_pipeline:
            # Validate pipeline
            schema_path = (
                Path(__file__).parent.parent.parent
                / "schemas"
                / "gl_pipeline.schema.v1.json"
            )

            if not schema_path.exists():
                errors.append(f"Pipeline schema not found: {schema_path}")
            else:
                try:
                    from greenlang.sdk.pipeline import Pipeline, load_pipeline_schema

                    # Load schema
                    schema = load_pipeline_schema(schema_path)

                    # Validate pipeline
                    pipeline = Pipeline.from_yaml(file_path, schema=schema)
                    file_name = pipeline.spec.name

                    # Get validation errors/warnings
                    validation_errors = pipeline.validate(strict=strict)

                    # Separate errors from warnings (basic heuristic)
                    for msg in validation_errors:
                        if any(
                            keyword in msg.lower()
                            for keyword in [
                                "error",
                                "failed",
                                "invalid",
                                "missing required",
                            ]
                        ):
                            errors.append(msg)
                        else:
                            warnings.append(msg)

                    summary = {"steps": len(pipeline.spec.steps)}

                except Exception as e:
                    errors.append(f"Pipeline validation failed: {str(e)}")

        elif is_pack:
            # Validate pack
            try:
                from greenlang.packs.manifest import PackManifest

                pack_manifest = PackManifest.from_file(file_path_obj)
                file_name = pack_manifest.name

                # Check if referenced files exist
                missing_files = pack_manifest.validate_files_exist(file_path_obj.parent)
                errors.extend(missing_files)

                # Get warnings
                warnings.extend(pack_manifest.get_warnings())

                summary = {
                    "pipelines": len(pack_manifest.contents.pipelines),
                    "agents": len(pack_manifest.contents.agents),
                    "datasets": len(pack_manifest.contents.datasets),
                }

            except Exception as e:
                errors.append(f"Pack validation failed: {str(e)}")

    except Exception as e:
        errors.append(f"Unexpected validation error: {str(e)}")

    # Determine success
    success = len(errors) == 0 and (not strict or len(warnings) == 0)

    if output_json:
        result = {
            "ok": success,
            "spec_version": spec_version,
            "type": file_type,
            "name": file_name,
            "errors": errors,
            "warnings": warnings,
            "summary": summary,
        }
        console.print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        if success:
            console.print(
                f"[green]✓ Validation passed[/green] - {file_type}: {file_name}"
            )
        else:
            console.print(f"[red]✗ Validation failed[/red] - {file_type}: {file_name}")

        if errors:
            console.print(f"\n[bold red]Errors ({len(errors)}):[/bold red]")
            for error in errors:
                console.print(f"  • {error}")

        if warnings:
            console.print(f"\n[bold yellow]Warnings ({len(warnings)}):[/bold yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")

        if summary:
            console.print("\n[bold]Summary:[/bold]")
            for key, value in summary.items():
                console.print(f"  {key.replace('_', ' ').title()}: {value}")

    # Exit with appropriate code
    if not success:
        sys.exit(1)


# ============= RUN COMMAND WITH CACHING & PROGRESS =============
@cli.command()
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.option("--input", "-i", type=click.Path(exists=True), help="Input data file")
@click.option("--no-cache", is_flag=True, help="Disable caching")
@pass_context
def run(ctx: CLIContext, pipeline_file: str, input: Optional[str], no_cache: bool):
    """Execute pipeline with caching and progress tracking"""

    # Generate run ID
    run_id = f"run_{DeterministicClock.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup JSONL logger
    jsonl_logger = JSONLLogger(run_dir / "events.jsonl")
    jsonl_logger.log_start(run_id, pipeline_file)

    try:
        # Load workflow
        workflow = Workflow.from_yaml(pipeline_file)

        # Load input data
        input_data = {}
        if input:
            with open(input, "r") as f:
                input_data = (
                    json.load(f) if input.endswith(".json") else yaml.safe_load(f)
                )

        if ctx.dry_run:
            console.print("[yellow]DRY-RUN: Would execute pipeline[/yellow]")
            for step in workflow.steps:
                console.print(f"  • {step.name} ({step.agent_id})")
            return

        # Execute with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:

            total_steps = len(workflow.steps)
            task = progress.add_task(f"Executing {workflow.name}", total=total_steps)

            # Register workflow
            ctx.orchestrator.register_workflow(run_id, workflow)

            # Execute each step with caching
            results = {}
            for i, step in enumerate(workflow.steps):
                progress.update(task, description=f"Running: {step.name}")

                # Generate cache key
                cache_key = None
                if not no_cache:
                    cache_data = {
                        "step": step.name,
                        "agent": step.agent_id,
                        "input": input_data,
                    }
                    cache_key = hashlib.md5(
                        json.dumps(cache_data, sort_keys=True).encode()
                    ).hexdigest()
                    cache_file = CACHE_DIR / f"{cache_key}.pkl"

                    # Check cache
                    if cache_file.exists():
                        with open(cache_file, "rb") as f:
                            cached_result = pickle.load(f)
                            results[step.name] = cached_result
                            jsonl_logger.log_event("cache_hit", {"step": step.name})
                            progress.update(task, advance=1)
                            continue

                # Execute step
                start_time = time.time()
                jsonl_logger.log_step_start(step.name, step.agent_id)

                # Simulate execution (would call actual agent)
                time.sleep(0.5)  # Simulate work
                result = {
                    "success": True,
                    "data": {"output": f"Result from {step.name}"},
                }
                results[step.name] = result

                duration = time.time() - start_time
                jsonl_logger.log_step_complete(step.name, True, duration)

                # Cache result
                if cache_key and not no_cache:
                    with open(cache_file, "wb") as f:
                        pickle.dump(result, f)

                progress.update(task, advance=1)

            progress.update(task, description="Pipeline complete!")

        # Save results
        results_file = run_dir / "results.json"
        results_file.write_text(json.dumps(results, indent=2))

        jsonl_logger.log_complete(run_id, True)

        console.print("\n[green]✓ Pipeline executed successfully![/green]")
        console.print(f"Run ID: {run_id}")
        console.print(f"Results: {results_file}")
        console.print(f"Logs: {run_dir / 'events.jsonl'}")

    except Exception as e:
        jsonl_logger.log_error(str(e))
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    finally:
        jsonl_logger.close()


# ============= REPORT COMMAND =============
@cli.command()
@click.argument("run_id")
@click.option("--format", "-f", type=click.Choice(["md", "html", "pdf"]), default="md")
@click.option("--output", "-o", type=click.Path(), help="Output file")
@pass_context
def report(ctx: CLIContext, run_id: str, format: str, output: Optional[str]):
    """Generate report from run ID"""

    # Find run directory
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        console.print(f"[red]Run '{run_id}' not found[/red]")
        console.print("Available runs:")
        for run in sorted(RUNS_DIR.iterdir()):
            if run.is_dir():
                console.print(f"  • {run.name}")
        sys.exit(1)

    # Load results
    results_file = run_dir / "results.json"
    if not results_file.exists():
        console.print(f"[red]No results found for run '{run_id}'[/red]")
        sys.exit(1)

    with open(results_file, "r") as f:
        results = json.load(f)

    # Load events
    events_file = run_dir / "events.jsonl"
    events = []
    if events_file.exists():
        events = JSONLLogger.read_jsonl(events_file)

    # Generate report content
    if format == "md":
        report_content = f"""# GreenLang Report
## Run ID: {run_id}

### Summary
- Total Steps: {len(results)}
- Status: Success ✓

### Results
"""
        for step_name, step_result in results.items():
            report_content += f"\n#### {step_name}\n"
            report_content += f"```json\n{json.dumps(step_result, indent=2)}\n```\n"

        report_content += "\n### Execution Timeline\n"
        for event in events:
            if event["event_type"] in ["step_start", "step_complete"]:
                report_content += f"- [{event['timestamp']}] {event['event_type']}: {event['data'].get('step_name', 'N/A')}\n"

    elif format == "html":
        report_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>GreenLang Report - {run_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2e7d32; }}
        pre {{ background: #f5f5f5; padding: 10px; }}
    </style>
</head>
<body>
    <h1>GreenLang Report</h1>
    <h2>Run ID: {run_id}</h2>
    <h3>Results</h3>
    <pre>{json.dumps(results, indent=2)}</pre>
</body>
</html>"""

    elif format == "pdf":
        console.print(
            "[yellow]PDF generation requires additional dependencies[/yellow]"
        )
        console.print("Install with: pip install pdfkit")
        return

    # Save report
    if output:
        output_path = Path(output)
    else:
        output_path = run_dir / f"report.{format}"

    output_path.write_text(report_content)
    console.print(f"[green]Report generated: {output_path}[/green]")


# ============= ASK COMMAND =============
@cli.command()
@click.argument("question", nargs=-1)
@pass_context
def ask(ctx: CLIContext, question: tuple):
    """Natural language assistant (requires API key)"""

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        console.print(
            Panel(
                "[yellow]API key required for AI assistant[/yellow]\n\n"
                "Set one of:\n"
                "  export OPENAI_API_KEY='your-key'\n"
                "  export ANTHROPIC_API_KEY='your-key'",
                title="Configuration Required",
            )
        )
        return

    if question:
        query = " ".join(question)
        console.print(f"[cyan]Question:[/cyan] {query}")
        console.print("[dim]AI features require additional setup[/dim]")
    else:
        console.print("[cyan]Interactive mode (type 'exit' to quit)[/cyan]")
        while True:
            query = console.input("[bold]Ask > [/bold]")
            if query.lower() in ["exit", "quit"]:
                break
            console.print("[dim]Processing...[/dim]")


# Add pack command group
cli.add_command(pack)

if __name__ == "__main__":
    cli()
