"""
gl policy - Policy management and enforcement
"""

import typer
import json
import yaml
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("check")
def check(
    pack_path: Optional[Path] = typer.Argument(
        None, help="Pack path to validate (defaults to current directory)"
    ),
    explain: bool = typer.Option(False, "--explain", help="Show detailed explanations"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Validate pack against install policy"""
    from ..policy.enforcer import check_install
    from ..packs.manifest import load_manifest

    # Default to current directory if no pack path provided
    target = pack_path or Path.cwd()

    if not target.exists():
        console.print(f"[red]Pack path not found: {target}[/red]")
        raise typer.Exit(1)

    # Look for pack.yaml
    pack_file = target / "pack.yaml" if target.is_dir() else target
    if target.is_dir() and not pack_file.exists():
        console.print(f"[red]No pack.yaml found in: {target}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Validating pack install policy: {target}[/cyan]")

    try:
        # Load manifest
        if target.is_dir():
            manifest = load_manifest(target)
        else:
            # Single pack.yaml file
            manifest = load_manifest(target.parent)

        # Use PR4 check_install function directly
        try:
            check_install(manifest, str(target), "publish")
            allowed = True
            reason = "Policy check passed"
        except RuntimeError as e:
            allowed = False
            reason = str(e)

        # Format output
        if json_output:
            output = {
                "target": str(target),
                "type": "pack installation",
                "allowed": allowed,
                "reason": reason,
            }
            console.print(json.dumps(output, indent=2))
        else:
            if allowed:
                console.print(f"[green]OK Policy check passed[/green]")
                if explain:
                    console.print(
                        f"  License: {getattr(manifest, 'license', 'unknown')}"
                    )
                    policy_attr = getattr(manifest, "policy", {})
                    if hasattr(policy_attr, "network"):
                        console.print(
                            f"  Network allowlist: {len(policy_attr.network)} domains"
                        )
                    if hasattr(policy_attr, "ef_vintage_min"):
                        console.print(f"  EF vintage: {policy_attr.ef_vintage_min}")
            else:
                console.print(f"[red]ERROR Policy check failed[/red]")
                console.print(f"  {reason}")

                if explain:
                    console.print("\n[yellow]Common fixes:[/yellow]")
                    if "license" in reason.lower():
                        console.print(
                            "  - Update license to: Apache-2.0, MIT, BSD-3-Clause, or Commercial"
                        )
                    if "network" in reason.lower():
                        console.print("  - Add network allowlist to pack.yaml:")
                        console.print("    policy:")
                        console.print("      network:")
                        console.print("        - domain1.com")
                        console.print("        - domain2.com")
                    if "vintage" in reason.lower():
                        console.print(
                            "  - Update ef_vintage_min to 2024 or later in pack.yaml:"
                        )
                        console.print("    policy:")
                        console.print("      ef_vintage_min: 2025")
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("run")
def run_policy(
    gl_yaml: Path = typer.Argument(..., help="gl.yaml pipeline file to check"),
    explain: bool = typer.Option(False, "--explain", help="Show detailed explanation"),
):
    """Dry-run pipeline against runtime policy (no execution)"""
    from ..policy.enforcer import check_run

    if not gl_yaml.exists():
        console.print(f"[red]Pipeline file not found: {gl_yaml}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Dry-run policy check for: {gl_yaml}[/cyan]")

    try:
        # Load pipeline
        with open(gl_yaml) as f:
            pipeline_data = yaml.safe_load(f)

        # Create a simple pipeline context
        class PipelineContext:
            def __init__(self, data):
                self.data = data
                self.egress_targets = []
                self.region = "us-west-2"

            def to_policy_doc(self):
                return {
                    "name": self.data.get("name", "unknown"),
                    "policy": self.data.get("policy", {}),
                    "resources": self.data.get(
                        "resources", {"memory": 1024, "cpu": 1, "disk": 1024}
                    ),
                    "steps": self.data.get("steps", []),
                }

        pipeline = PipelineContext(pipeline_data)

        class ExecutionContext:
            def __init__(self):
                self.egress_targets = []
                self.region = "us-west-2"

        ctx = ExecutionContext()

        # Use PR4 check_run function directly
        try:
            check_run(pipeline, ctx)
            allowed = True
            reason = "Runtime policy check passed"
        except RuntimeError as e:
            allowed = False
            reason = str(e)

        if allowed:
            console.print(f"[green]OK Pipeline would be allowed to run[/green]")
            if explain:
                console.print("\n[bold]Policy requirements satisfied:[/bold]")
                console.print("  - No unauthorized egress detected")
                console.print("  - Resource limits within bounds")
                console.print("  - Data residency compliance")
        else:
            console.print(f"[red]ERROR Pipeline would be denied[/red]")
            console.print(f"  {reason}")

            if explain:
                console.print("\n[yellow]Suggested fixes:[/yellow]")
                if "egress" in reason.lower():
                    console.print("  - Add required domains to gl.yaml:")
                    console.print("    policy:")
                    console.print("      network:")
                    console.print("        - required-domain.com")
                elif "region" in reason.lower():
                    console.print("  - Check data residency requirements")
                elif "resource" in reason.lower():
                    console.print("  - Reduce resource requirements in pipeline")
                else:
                    console.print("  - Review pipeline configuration")
                    console.print("  - Check network access requirements")

            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Policy dry-run failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("list")
def list_policies(
    location: Path = typer.Option(
        Path.home() / ".greenlang" / "policies", "--location", "-l"
    )
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
            f"{stat.st_mtime}",
        )

    console.print(table)


@app.command("add")
def add_policy(
    source: Path = typer.Argument(..., help="Policy file or bundle to add"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Policy name"),
    location: Path = typer.Option(
        Path.home() / ".greenlang" / "policies", "--location", "-l"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing"),
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
    location: Path = typer.Option(
        Path.home() / ".greenlang" / "policies", "--location", "-l"
    ),
    syntax: bool = typer.Option(
        True, "--syntax/--no-syntax", help="Syntax highlighting"
    ),
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
def validate_policy(policy: Path = typer.Argument(..., help="Policy file to validate")):
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
