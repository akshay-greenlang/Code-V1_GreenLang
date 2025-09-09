"""
gl run - Execute pipelines and packs
"""

import typer
import json
import yaml
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.panel import Panel

app = typer.Typer(invoke_without_command=True)
console = Console()


@app.callback()
def run(
    ctx: typer.Context,
    pipeline: str = typer.Argument(..., help="Pipeline file or pack reference"),
    inputs: Optional[str] = typer.Option(None, "--inputs", "-i", help="Input data file (JSON/YAML)"),
    artifacts: str = typer.Option("out", "--artifacts", "-a", help="Artifacts directory"),
    backend: str = typer.Option("local", "--backend", "-b", help="Execution backend (local|k8s)"),
    profile: str = typer.Option("dev", "--profile", "-p", help="Configuration profile")
):
    """
    Execute pipelines deterministically
    
    Examples:
        gl run gl.yaml
        gl run pipeline.yaml --inputs data.json
        gl run calc --backend k8s --profile prod
    
    Produces stable run.json and artifacts in output directory.
    """
    # Don't run if a subcommand was invoked
    if ctx.invoked_subcommand is not None:
        return
    
    from pathlib import Path
    from ..runtime.executor import Executor
    from ..provenance.ledger import write_run_ledger
    
    # Simplified implementation for PR2
    pipeline_path = Path(pipeline)
    if not pipeline_path.exists() and not pipeline.endswith('.yaml'):
        pipeline_path = Path(f"{pipeline}.yaml")
    
    if not pipeline_path.exists():
        console.print(f"[red]Pipeline not found: {pipeline}[/red]")
        raise typer.Exit(1)
    
    # Load input data
    input_data = {}
    if inputs:
        inputs_path = Path(inputs)
        if inputs_path.suffix == ".json":
            with open(inputs_path) as f:
                input_data = json.load(f)
        elif inputs_path.suffix in [".yaml", ".yml"]:
            with open(inputs_path) as f:
                input_data = yaml.safe_load(f)
        else:
            console.print(f"[red]Unsupported input format: {inputs_path.suffix}[/red]")
            raise typer.Exit(1)
    
    # Create artifacts directory
    artifacts_path = Path(artifacts)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    # Execute pipeline - simplified for PR2
    try:
        # Load pipeline YAML
        with open(pipeline_path) as f:
            pipeline_data = yaml.safe_load(f)
        
        # Create executor
        exec = Executor(backend=backend)
        
        # Create context
        ctx = {
            "artifacts": str(artifacts_path),
            "pipeline": pipeline_path.name,
            "backend": backend,
            "profile": profile
        }
        
        # Execute (simplified - no policy check in PR2)
        console.print(f"[cyan]Executing {pipeline_path.name}...[/cyan]")
        res = exec.execute(pipeline_data, inputs=input_data)
        
        # Write run ledger
        write_run_ledger(res, ctx)
        
        console.print(f"[green]Artifacts -> {artifacts}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("list")
def list_pipelines():
    """List available pipelines"""
    from greenlang.packs.registry import PackRegistry
    
    registry = PackRegistry()
    pipelines = registry.list_pipelines()
    
    if not pipelines:
        console.print("[yellow]No pipelines found[/yellow]")
        console.print("\nInstall packs with pipelines: [cyan]gl pack add <pack-name>[/cyan]")
        return
    
    console.print("[bold]Available Pipelines:[/bold]\n")
    
    for pack_name, pack_pipelines in pipelines.items():
        console.print(f"[cyan]{pack_name}:[/cyan]")
        for pipeline in pack_pipelines:
            console.print(f"  • {pipeline['name']}: {pipeline.get('description', 'No description')}")
    
    console.print(f"\n[dim]Run with: gl run <pack>/<pipeline>[/dim]")


@app.command("info")
def pipeline_info(
    pipeline: str = typer.Argument(..., help="Pipeline name or reference")
):
    """Show pipeline details"""
    from ..sdk.pipeline import Pipeline
    from ..packs.registry import PackRegistry
    
    # Try to load pipeline
    if "/" in pipeline:
        pack_name, pipeline_name = pipeline.split("/", 1)
        registry = PackRegistry()
        pack = registry.get(pack_name)
        if not pack:
            console.print(f"[red]Pack not found: {pack_name}[/red]")
            raise typer.Exit(1)
        pipe_info = pack.get_pipeline_info(pipeline_name)
    else:
        # Try local file
        if Path(f"{pipeline}.yaml").exists():
            pipe = Pipeline.from_yaml(f"{pipeline}.yaml")
            pipe_info = pipe.to_dict()
        else:
            console.print(f"[red]Pipeline not found: {pipeline}[/red]")
            raise typer.Exit(1)
    
    # Display pipeline info
    console.print(Panel.fit(
        f"[bold]{pipe_info['name']}[/bold]\n"
        f"{pipe_info.get('description', 'No description')}\n\n"
        f"Version: {pipe_info.get('version', '1.0')}\n"
        f"Steps: {len(pipe_info.get('steps', []))}\n",
        title="Pipeline Information"
    ))
    
    # Show inputs
    if pipe_info.get('inputs'):
        console.print("\n[bold]Inputs:[/bold]")
        for name, spec in pipe_info['inputs'].items():
            console.print(f"  • {name}: {spec.get('type', 'any')} ({spec.get('unit', 'none')})")
    
    # Show outputs
    if pipe_info.get('outputs'):
        console.print("\n[bold]Outputs:[/bold]")
        for name, spec in pipe_info['outputs'].items():
            console.print(f"  • {name}: {spec.get('type', 'any')} ({spec.get('unit', 'none')})")
    
    # Show steps
    if pipe_info.get('steps'):
        console.print("\n[bold]Steps:[/bold]")
        for i, step in enumerate(pipe_info['steps'], 1):
            console.print(f"  {i}. {step['name']} ({step.get('agent', 'unknown')})")