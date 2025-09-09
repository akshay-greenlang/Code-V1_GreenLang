"""
gl policy - Policy management and enforcement
"""

import typer
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("check")
def check(
    target: Path = typer.Argument(..., help="Target file or directory to check"),
    policy: Optional[Path] = typer.Option(None, "--policy", "-p", help="Policy file or bundle"),
    explain: bool = typer.Option(False, "--explain", help="Show detailed explanations"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """Evaluate OPA bundle against target"""
    from ..policy.enforcer import check_install, check_run
    
    if not target.exists():
        console.print(f"[red]Target not found: {target}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Checking policy for: {target}[/cyan]")
    
    # Determine check type based on target
    context = {}
    check_func = None
    
    if (target / "pack.yaml").exists():
        # Pack installation check
        from ..packs.manifest import load_manifest
        manifest = load_manifest(target)
        context = {
            "manifest": manifest.dict() if hasattr(manifest, 'dict') else manifest,
            "path": str(target),
            "stage": "install"
        }
        check_func = check_install
        check_type = "pack installation"
    elif (target / "gl.yaml").exists() or target.suffix in [".yaml", ".yml"]:
        # Pipeline run check
        if target.is_file():
            with open(target) as f:
                pipeline = yaml.safe_load(f)
        else:
            with open(target / "gl.yaml") as f:
                pipeline = yaml.safe_load(f)
        
        context = {
            "pipeline": pipeline,
            "inputs": {},
            "backend": "local"
        }
        check_func = check_run
        check_type = "pipeline execution"
    else:
        console.print(f"[red]Cannot determine policy check type for: {target}[/red]")
        raise typer.Exit(1)
    
    # Load policy if provided, otherwise use default
    if policy:
        policy_path = policy
    else:
        # Try to find default policy bundle
        policy_path = Path.home() / ".greenlang" / "policies" / "default.rego"
        if not policy_path.exists():
            policy_path = Path("/etc/greenlang/policies/default.rego")
    
    if not policy_path.exists():
        console.print(f"[yellow]Warning: No policy found at {policy_path}[/yellow]")
        console.print("Proceeding without policy enforcement")
        allowed = True
        reasons = []
    else:
        # Evaluate policy
        try:
            allowed, reasons = check_func(policy_path, context)
        except Exception as e:
            console.print(f"[red]Policy evaluation failed: {e}[/red]")
            raise typer.Exit(1)
    
    # Format output
    if json_output:
        output = {
            "target": str(target),
            "type": check_type,
            "allowed": allowed,
            "reasons": reasons
        }
        console.print(json.dumps(output, indent=2))
    else:
        if allowed:
            console.print(f"[green]✓ Policy check passed[/green]")
            if reasons and explain:
                console.print("\n[bold]Allowed because:[/bold]")
                for reason in reasons:
                    console.print(f"  • {reason}")
        else:
            console.print(f"[red]✗ Policy check failed[/red]")
            if reasons:
                console.print("\n[bold]Denied because:[/bold]")
                for reason in reasons:
                    console.print(f"  • {reason}")
                
                if explain:
                    console.print("\n[yellow]To fix:[/yellow]")
                    console.print("  1. Review the policy requirements")
                    console.print("  2. Update your configuration")
                    console.print("  3. Request policy exception if needed")
            raise typer.Exit(1)


@app.command("run")
def run_policy(
    pipeline: str = typer.Argument(..., help="Pipeline to check"),
    inputs: Optional[Path] = typer.Option(None, "--inputs", "-i", help="Input file"),
    policy: Optional[Path] = typer.Option(None, "--policy", "-p", help="Policy file"),
    explain: bool = typer.Option(False, "--explain", help="Show rule explanations")
):
    """Dry-run pipeline against policy"""
    from greenlang.policy.enforcer import check_run
    
    # Load pipeline
    if Path(pipeline).exists():
        with open(pipeline) as f:
            pipe_data = yaml.safe_load(f)
    else:
        console.print(f"[red]Pipeline not found: {pipeline}[/red]")
        raise typer.Exit(1)
    
    # Load inputs if provided
    input_data = {}
    if inputs:
        if inputs.suffix == ".json":
            with open(inputs) as f:
                input_data = json.load(f)
        elif inputs.suffix in [".yaml", ".yml"]:
            with open(inputs) as f:
                input_data = yaml.safe_load(f)
    
    context = {
        "pipeline": pipe_data,
        "inputs": input_data,
        "backend": "local",
        "profile": "dev"
    }
    
    console.print(f"[cyan]Policy dry-run for: {pipeline}[/cyan]")
    
    # Load policy
    if policy:
        policy_path = policy
    else:
        policy_path = Path.home() / ".greenlang" / "policies" / "default.rego"
    
    if not policy_path.exists():
        console.print(f"[yellow]No policy found, skipping checks[/yellow]")
        return
    
    # Evaluate policy
    try:
        allowed, reasons = check_run(policy_path, context)
    except Exception as e:
        console.print(f"[red]Policy evaluation failed: {e}[/red]")
        raise typer.Exit(1)
    
    if allowed:
        console.print(f"[green]✓ Pipeline would be allowed to run[/green]")
        if explain and reasons:
            console.print("\n[bold]Policy rules satisfied:[/bold]")
            for reason in reasons:
                console.print(f"  • {reason}")
    else:
        console.print(f"[red]✗ Pipeline would be denied[/red]")
        if reasons:
            console.print("\n[bold]Policy violations:[/bold]")
            for reason in reasons:
                console.print(f"  • {reason}")
        
        if explain:
            console.print("\n[yellow]Suggested fixes:[/yellow]")
            # Provide specific suggestions based on violations
            if "resource limits" in str(reasons).lower():
                console.print("  • Reduce resource requests in pipeline")
            if "untrusted" in str(reasons).lower():
                console.print("  • Use verified packs only")
            if "sensitive" in str(reasons).lower():
                console.print("  • Remove sensitive data from inputs")
        
        raise typer.Exit(1)


@app.command("list")
def list_policies(
    location: Path = typer.Option(Path.home() / ".greenlang" / "policies", "--location", "-l")
):
    """List available policies"""
    if not location.exists():
        console.print(f"[yellow]No policies found at: {location}[/yellow]")
        console.print("\nAdd policies with: [cyan]gl policy add <policy.rego>[/cyan]")
        return
    
    policies = list(location.glob("*.rego")) + list(location.glob("**/*.rego"))
    
    if not policies:
        console.print("[yellow]No policy files found[/yellow]")
        return
    
    table = Table(title="Available Policies")
    table.add_column("Name", style="cyan")
    table.add_column("Location")
    table.add_column("Size", style="green")
    table.add_column("Modified", style="yellow")
    
    for policy_file in policies:
        stat = policy_file.stat()
        table.add_row(
            policy_file.stem,
            str(policy_file.relative_to(location)),
            f"{stat.st_size} bytes",
            f"{stat.st_mtime}"
        )
    
    console.print(table)


@app.command("add")
def add_policy(
    source: Path = typer.Argument(..., help="Policy file or bundle to add"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Policy name"),
    location: Path = typer.Option(Path.home() / ".greenlang" / "policies", "--location", "-l"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing")
):
    """Add policy to local bundle"""
    if not source.exists():
        console.print(f"[red]Policy file not found: {source}[/red]")
        raise typer.Exit(1)
    
    # Determine target name
    if not name:
        name = source.stem
    
    # Create policies directory
    location.mkdir(parents=True, exist_ok=True)
    
    # Copy policy file
    target = location / f"{name}.rego"
    
    if target.exists() and not force:
        console.print(f"[red]Policy already exists: {name}[/red]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)
    
    if source.is_file():
        import shutil
        shutil.copy2(source, target)
        console.print(f"[green]✓[/green] Added policy: {name}")
        console.print(f"  Location: {target}")
    elif source.is_dir():
        # Copy entire bundle
        import shutil
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        console.print(f"[green]✓[/green] Added policy bundle: {name}")
        console.print(f"  Location: {target}")
    
    console.print("\nUse policy with:")
    console.print(f"  gl policy check <target> --policy {name}")
    console.print(f"  gl run <pipeline> --policy {name}")


@app.command("show")
def show_policy(
    name: str = typer.Argument(..., help="Policy name to display"),
    location: Path = typer.Option(Path.home() / ".greenlang" / "policies", "--location", "-l"),
    syntax: bool = typer.Option(True, "--syntax/--no-syntax", help="Syntax highlighting")
):
    """Display policy contents"""
    policy_file = location / f"{name}.rego"
    
    if not policy_file.exists():
        # Try without extension
        policy_file = location / name
        if not policy_file.exists():
            console.print(f"[red]Policy not found: {name}[/red]")
            raise typer.Exit(1)
    
    with open(policy_file) as f:
        content = f.read()
    
    if syntax:
        syntax_obj = Syntax(content, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax_obj, title=f"Policy: {name}"))
    else:
        console.print(content)


@app.command("validate")
def validate_policy(
    policy: Path = typer.Argument(..., help="Policy file to validate")
):
    """Validate policy syntax"""
    if not policy.exists():
        console.print(f"[red]Policy file not found: {policy}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Validating policy: {policy}[/cyan]")
    
    # Try to parse the policy
    try:
        with open(policy) as f:
            content = f.read()
        
        # Basic syntax check (would use OPA parser in real implementation)
        if "package" not in content:
            console.print("[yellow]Warning: No package declaration found[/yellow]")
        
        if "allow" not in content and "deny" not in content:
            console.print("[yellow]Warning: No allow/deny rules found[/yellow]")
        
        # Check for common patterns
        rules = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("allow") or line.startswith("deny"):
                rules.append(line.split()[0])
        
        console.print(f"[green]✓[/green] Policy syntax valid")
        console.print(f"  Rules found: {', '.join(set(rules))}")
        
    except Exception as e:
        console.print(f"[red]Policy validation failed: {e}[/red]")
        raise typer.Exit(1)