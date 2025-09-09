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

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def run(
    ctx: typer.Context,
    pipeline: str = typer.Argument(..., help="Pipeline or pack to run"),
    inputs: Optional[Path] = typer.Option(None, "--inputs", "-i", help="Input data file (JSON/YAML)"),
    artifacts: Path = typer.Option(Path("out"), "--artifacts", "-a", help="Artifacts directory"),
    backend: str = typer.Option("local", "--backend", "-b", help="Execution backend (local|k8s)"),
    profile: str = typer.Option("dev", "--profile", "-p", help="Configuration profile"),
    policy: Optional[Path] = typer.Option(None, "--policy", help="Policy file to enforce"),
    explain: bool = typer.Option(False, "--explain", help="Explain execution steps"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate execution without running")
):
    """
    Execute pipelines deterministically
    
    Examples:
        gl run my-pipeline
        gl run my-pack/pipeline --inputs data.json
        gl run calc --backend k8s --profile prod
    
    Produces stable run.json and artifacts in output directory.
    """
    if ctx.invoked_subcommand is not None:
        return
    
    # Parse pipeline reference
    if "/" in pipeline:
        pack_name, pipeline_name = pipeline.split("/", 1)
    else:
        pack_name = None
        pipeline_name = pipeline
    
    # Load input data
    input_data = {}
    if inputs:
        if inputs.suffix == ".json":
            with open(inputs) as f:
                input_data = json.load(f)
        elif inputs.suffix in [".yaml", ".yml"]:
            with open(inputs) as f:
                input_data = yaml.safe_load(f)
        else:
            console.print(f"[red]Unsupported input format: {inputs.suffix}[/red]")
            raise typer.Exit(1)
    
    # Apply policy check if provided
    if policy:
        console.print(f"[cyan]Checking policy: {policy}...[/cyan]")
        from ..policy.enforcer import check_run
        
        policy_context = {
            "pipeline": pipeline,
            "inputs": input_data,
            "backend": backend,
            "profile": profile
        }
        
        allowed, reasons = check_run(policy, policy_context)
        
        if not allowed:
            console.print("[red]✗ Policy check failed[/red]")
            if explain:
                console.print("\n[yellow]Policy violations:[/yellow]")
                for reason in reasons:
                    console.print(f"  • {reason}")
            raise typer.Exit(1)
        
        console.print("[green]✓ Policy check passed[/green]")
    
    # Create artifacts directory
    artifacts.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        console.print("\n[yellow]DRY RUN - Would execute:[/yellow]")
        console.print(f"  Pipeline: {pipeline}")
        console.print(f"  Backend: {backend}")
        console.print(f"  Profile: {profile}")
        console.print(f"  Artifacts: {artifacts}")
        if input_data:
            console.print(f"  Inputs: {len(input_data)} parameters")
        return
    
    # Execute pipeline
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Running {pipeline}...", total=None)
        
        try:
            from ..runtime.executor import Executor
            from ..sdk.pipeline import Pipeline
            from ..provenance.ledger import write_run_ledger
            
            # Load pipeline
            if pack_name:
                from ..packs.loader import PackLoader
                loader = PackLoader()
                pack = loader.load(pack_name)
                pipe = pack.get_pipeline(pipeline_name)
            else:
                # Load from file or registry
                if Path(f"{pipeline}.yaml").exists():
                    pipe = Pipeline.from_yaml(f"{pipeline}.yaml")
                elif Path("gl.yaml").exists():
                    pipe = Pipeline.from_yaml("gl.yaml")
                else:
                    # Try to load from registered packs
                    from ..packs.registry import PackRegistry
                    registry = PackRegistry()
                    pipe = registry.get_pipeline(pipeline)
            
            # Create executor with context
            executor = Executor(backend=backend, profile=profile)
            ctx = executor.create_context(artifacts)
            
            # Execute pipeline
            progress.update(task, description=f"Executing {pipeline}...")
            result = executor.execute(pipe, input_data, context=ctx)
            
            # Write deterministic run ledger
            write_run_ledger(result, ctx, artifacts / "run.json")
            
            progress.update(task, completed=True)
            
            if result.success:
                console.print(f"\n[green]✓[/green] Pipeline completed successfully")
                console.print(f"Artifacts → {artifacts}")
                
                # Display results
                if result.outputs:
                    console.print("\n[bold]Outputs:[/bold]")
                    syntax = Syntax(
                        json.dumps(result.outputs, indent=2),
                        "json",
                        theme="monokai"
                    )
                    console.print(syntax)
                
                # Show metrics if available
                if result.metrics:
                    console.print("\n[bold]Metrics:[/bold]")
                    for key, value in result.metrics.items():
                        console.print(f"  {key}: {value}")
            else:
                console.print(f"\n[red]✗ Pipeline failed[/red]")
                console.print(f"Error: {result.error}")
                if explain and result.trace:
                    console.print("\n[yellow]Execution trace:[/yellow]")
                    for step in result.trace:
                        console.print(f"  {step}")
                raise typer.Exit(1)
                
        except Exception as e:
            progress.update(task, completed=True)
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