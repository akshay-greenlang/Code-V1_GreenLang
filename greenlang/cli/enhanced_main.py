# -*- coding: utf-8 -*-
"""
Enhanced GreenLang CLI with global options and improved DX
"""

import click
import json
import yaml
import os
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler

import greenlang
from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import Workflow
from greenlang.cli.jsonl_logger import JSONLLogger
from greenlang.cli.agent_registry import AgentRegistry
from greenlang.determinism import DeterministicClock

# Initialize console and logging
console = Console()


# Configure logging based on verbosity
def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    return logging.getLogger("greenlang")


# Global context for CLI state
class CLIContext:
    """Context object for CLI state management"""

    def __init__(self):
        self.verbose = False
        self.dry_run = False
        self.logger = None
        self.jsonl_logger = None
        self.agent_registry = None
        self.orchestrator = None

    def setup(self, verbose: bool, dry_run: bool):
        """Initialize CLI context"""
        self.verbose = verbose
        self.dry_run = dry_run
        self.logger = setup_logging(verbose)
        self.jsonl_logger = JSONLLogger()
        self.agent_registry = AgentRegistry()
        self.orchestrator = Orchestrator()

        if verbose:
            self.logger.debug(f"GreenLang CLI v{greenlang.__version__}")
            self.logger.debug(f"Verbose mode: {verbose}")
            self.logger.debug(f"Dry-run mode: {dry_run}")

        if dry_run:
            console.print("[yellow]DRY-RUN MODE: No changes will be made[/yellow]")


# Pass context through Click
pass_cli_context = click.make_pass_decorator(CLIContext, ensure=True)


@click.group(invoke_without_command=True)
@click.version_option(version=greenlang.__version__, prog_name="GreenLang")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--dry-run", is_flag=True, help="Simulate execution without making changes"
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, dry_run: bool) -> None:
    """GreenLang - Climate Intelligence Framework

    Global options:
      --verbose, -v  : Show detailed output
      --dry-run      : Simulate without making changes
    """
    # Initialize context
    ctx.obj = CLIContext()
    ctx.obj.setup(verbose, dry_run)

    if ctx.invoked_subcommand is None:
        # Show welcome message
        console.print(
            Panel.fit(
                f"[bold green]GreenLang CLI v{greenlang.__version__}[/bold green]\n\n"
                "Climate Intelligence Framework\n\n"
                "Use 'gl --help' for available commands",
                style="green",
            )
        )


# ============= AGENTS COMMAND WITH PLUGIN DISCOVERY =============


@cli.group()
@pass_cli_context
def agents(ctx: CLIContext) -> None:
    """Manage and discover climate intelligence agents"""


@agents.command(name="list")
@pass_cli_context
def agents_list(ctx: CLIContext) -> None:
    """List all available agents (including plugins)"""
    if ctx.verbose:
        ctx.logger.debug("Discovering agents...")

    agents = ctx.agent_registry.discover_agents()

    table = Table(title="Available Agents")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Version")
    table.add_column("Type")
    table.add_column("Source")

    for agent_info in agents:
        source = "plugin" if agent_info.get("is_plugin") else "core"
        table.add_row(
            agent_info["id"],
            agent_info["name"],
            agent_info.get("version", "0.0.1"),
            agent_info.get("type", "standard"),
            source,
        )

    console.print(table)

    if ctx.verbose:
        ctx.logger.info(f"Found {len(agents)} agents")


@agents.command(name="info")
@click.argument("agent_id")
@pass_cli_context
def agents_info(ctx: CLIContext, agent_id: str) -> None:
    """Show detailed information about an agent"""
    agent_info = ctx.agent_registry.get_agent_info(agent_id)

    if not agent_info:
        console.print(f"[red]Agent '{agent_id}' not found[/red]")
        return

    console.print(
        Panel(
            f"[bold]{agent_info['name']}[/bold]\n\n"
            f"ID: {agent_info['id']}\n"
            f"Version: {agent_info.get('version', '0.0.1')}\n"
            f"Description: {agent_info.get('description', 'No description')}\n"
            f"Type: {agent_info.get('type', 'standard')}\n"
            f"Source: {'plugin' if agent_info.get('is_plugin') else 'core'}",
            title=f"Agent: {agent_id}",
        )
    )


@agents.command(name="template")
@click.argument("agent_id")
@click.option("--output", "-o", type=click.Path(), help="Save template to file")
@pass_cli_context
def agents_template(ctx: CLIContext, agent_id: str, output: Optional[str]) -> None:
    """Generate a template for creating a new agent"""
    if ctx.dry_run:
        console.print("[yellow]DRY-RUN: Would generate template[/yellow]")
        return

    template = ctx.agent_registry.get_agent_template(agent_id)

    if not template:
        console.print(f"[red]Cannot generate template for '{agent_id}'[/red]")
        return

    if output:
        Path(output).write_text(template)
        console.print(f"[green]Template saved to {output}[/green]")
    else:
        console.print(template)


# ============= RUN COMMAND WITH JSONL LOGGING =============


@cli.command()
@click.argument("workflow_file", type=click.Path(exists=True))
@click.option("--input", "-i", type=click.Path(exists=True), help="Input data file")
@click.option("--output", "-o", type=click.Path(), help="Output directory for results")
@click.option("--log-dir", type=click.Path(), help="Directory for JSONL logs")
@pass_cli_context
def run(
    ctx: CLIContext,
    workflow_file: str,
    input: Optional[str],
    output: Optional[str],
    log_dir: Optional[str],
) -> None:
    """Run a workflow pipeline with JSONL logging"""

    # Generate run ID
    run_id = f"run_{DeterministicClock.now().strftime('%Y%m%d_%H%M%S')}"

    # Setup JSONL logging
    log_path = Path(log_dir or "logs") / f"{run_id}.jsonl"
    if not ctx.dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.jsonl_logger.set_output(log_path)

    # Log start event
    ctx.jsonl_logger.log_event(
        "start",
        {
            "run_id": run_id,
            "workflow": workflow_file,
            "input": input,
            "dry_run": ctx.dry_run,
        },
    )

    if ctx.verbose:
        ctx.logger.info(f"Starting run: {run_id}")
        ctx.logger.debug(f"Workflow: {workflow_file}")

    try:
        # Load workflow
        workflow = Workflow.from_yaml(workflow_file)
        ctx.jsonl_logger.log_event(
            "workflow_loaded", {"name": workflow.name, "steps": len(workflow.steps)}
        )

        # Load input data
        input_data = {}
        if input:
            with open(input, "r") as f:
                input_data = (
                    json.load(f) if input.endswith(".json") else yaml.safe_load(f)
                )
            ctx.jsonl_logger.log_event("input_loaded", {"file": input})

        if ctx.dry_run:
            console.print("[yellow]DRY-RUN: Would execute workflow[/yellow]")
            for step in workflow.steps:
                console.print(f"  - Step: {step.name} (agent: {step.agent_id})")
        else:
            # Execute workflow
            ctx.orchestrator.register_workflow(run_id, workflow)
            result = ctx.orchestrator.execute_workflow(run_id, input_data)

            # Log completion
            ctx.jsonl_logger.log_event(
                "complete",
                {
                    "run_id": run_id,
                    "success": result.get("success", False),
                    "duration": ctx.jsonl_logger.get_duration(),
                },
            )

            # Save output
            if output:
                output_path = Path(output) / f"{run_id}_result.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result, indent=2))
                console.print(f"[green]Results saved to {output_path}[/green]")

    except Exception as e:
        ctx.jsonl_logger.log_event("error", {"run_id": run_id, "error": str(e)})
        console.print(f"[red]Error: {e}[/red]")
        if ctx.verbose:
            import traceback

            traceback.print_exc()


# ============= REPORT COMMAND WITH FORMAT OPTIONS =============


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "--out", type=click.Path(), help="Output directory")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["md", "html", "pdf", "json"], case_sensitive=False),
    default="md",
    help="Output format",
)
@pass_cli_context
def report(
    ctx: CLIContext, input_file: str, output: Optional[str], format: str
) -> None:
    """Generate reports in various formats"""

    if ctx.verbose:
        ctx.logger.info(f"Generating {format.upper()} report from {input_file}")

    # Load input data
    with open(input_file, "r") as f:
        data = json.load(f) if input_file.endswith(".json") else yaml.safe_load(f)

    # Import report agent
    from greenlang.agents import ReportAgent

    agent = ReportAgent()

    # Prepare output path
    output_dir = Path(output or "reports")
    if not ctx.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = DeterministicClock.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.{format}"
    output_path = output_dir / filename

    if ctx.dry_run:
        console.print(
            f"[yellow]DRY-RUN: Would generate {format.upper()} report to {output_path}[/yellow]"
        )
        return

    # Generate report based on format
    try:
        if format == "md":
            result = agent.run({"carbon_data": data, "format": "markdown"})
            output_path.write_text(result.get("report", ""))

        elif format == "html":
            result = agent.run({"carbon_data": data, "format": "html"})
            output_path.write_text(result.get("report", ""))

        elif format == "pdf":
            # Check if PDF generation is available
            try:
                import pdfkit

                result = agent.run({"carbon_data": data, "format": "html"})
                pdfkit.from_string(result.get("report", ""), str(output_path))
            except ImportError:
                console.print("[red]PDF generation requires 'pdfkit' package[/red]")
                console.print("Install with: pip install pdfkit")
                return

        elif format == "json":
            output_path.write_text(json.dumps(data, indent=2))

        console.print(f"[green]Report generated: {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error generating {format.upper()} report: {e}[/red]")
        if ctx.verbose:
            import traceback

            traceback.print_exc()


# ============= ASK COMMAND WITH API KEY HANDLING =============


@cli.command()
@click.argument("question", nargs=-1, required=False)
@pass_cli_context
def ask(ctx: CLIContext, question: tuple) -> None:
    """AI assistant for climate intelligence queries (requires API key)"""

    # Check for API keys
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        console.print(
            Panel(
                "[yellow]AI Assistant requires an API key[/yellow]\n\n"
                "Please set one of the following environment variables:\n"
                "  - OPENAI_API_KEY (for OpenAI)\n"
                "  - ANTHROPIC_API_KEY (for Claude)\n\n"
                "Example:\n"
                "  export OPENAI_API_KEY='your-key-here'\n"
                "  gl ask 'How to calculate emissions?'",
                title="API Key Required",
                style="yellow",
            )
        )
        return

    # Check if assistant module is available
    try:
        from greenlang.cli.assistant import AIAssistant

        assistant = AIAssistant()
    except ImportError:
        console.print("[red]AI Assistant module not available[/red]")
        return

    if question:
        # Process direct question
        query = " ".join(question)
        if ctx.verbose:
            ctx.logger.info(f"Asking: {query}")

        if ctx.dry_run:
            console.print(f"[yellow]DRY-RUN: Would ask AI: {query}[/yellow]")
            return

        response = assistant.ask(query)
        console.print(Panel(response, title="AI Response"))
    else:
        # Interactive mode
        console.print("[cyan]AI Assistant Interactive Mode[/cyan]")
        console.print("Type 'exit' to quit\n")

        while True:
            query = console.input("[bold]Ask > [/bold]")
            if query.lower() in ["exit", "quit"]:
                break

            if ctx.dry_run:
                console.print(f"[yellow]DRY-RUN: Would ask AI: {query}[/yellow]")
                continue

            response = assistant.ask(query)
            console.print(Panel(response))


# ============= OTHER COMMANDS =============


@cli.command()
@pass_cli_context
def init(ctx: CLIContext) -> None:
    """Initialize a new GreenLang project"""
    if ctx.dry_run:
        console.print("[yellow]DRY-RUN: Would initialize project[/yellow]")
        return

    # Create project structure
    directories = ["workflows", "data", "reports", "logs", "agents/custom"]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        if ctx.verbose:
            ctx.logger.debug(f"Created directory: {dir_path}")

    # Create sample workflow
    sample_workflow = {
        "name": "sample_workflow",
        "description": "Sample GreenLang workflow",
        "version": "0.0.1",
        "steps": [
            {
                "name": "validate",
                "agent_id": "validator",
                "description": "Validate input data",
            },
            {
                "name": "calculate",
                "agent_id": "carbon",
                "description": "Calculate emissions",
            },
            {"name": "report", "agent_id": "report", "description": "Generate report"},
        ],
    }

    workflow_path = Path("workflows/sample.yaml")
    workflow_path.write_text(yaml.dump(sample_workflow))

    # Create config file
    config = {
        "project": "My GreenLang Project",
        "version": "0.0.1",
        "agents": {"custom_path": "agents/custom"},
        "logging": {"level": "INFO", "jsonl": True},
    }

    config_path = Path("greenlang.yaml")
    config_path.write_text(yaml.dump(config))

    console.print("[green]✓ Project initialized successfully![/green]")
    console.print("\nCreated:")
    console.print("  - workflows/     : Workflow definitions")
    console.print("  - data/          : Input data files")
    console.print("  - reports/       : Generated reports")
    console.print("  - logs/          : Execution logs")
    console.print("  - agents/custom/ : Custom agent plugins")
    console.print("  - greenlang.yaml : Project configuration")
    console.print("\nNext steps:")
    console.print("  1. Edit workflows/sample.yaml")
    console.print("  2. Run: gl run workflows/sample.yaml")


if __name__ == "__main__":
    cli()
