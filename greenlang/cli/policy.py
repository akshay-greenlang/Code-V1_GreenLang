"""
Policy Management CLI Commands for GreenLang
"""

import click
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

# Import policy modules
try:
    from greenlang.policy.opa import evaluate, validate_policy
    from greenlang.policy.enforcer import PolicyEnforcer

    POLICY_ENABLED = True
except ImportError:
    console.print("[yellow]Policy module not available[/yellow]")
    POLICY_ENABLED = False


@click.group()
def policy():
    """Policy management and testing commands"""
    if not POLICY_ENABLED:
        console.print(
            "[red]Policy features are not available. Please install OPA.[/red]"
        )
        sys.exit(1)


@policy.command()
@click.argument("policy_file", type=click.Path(exists=True))
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.Path(exists=True),
    help="Input JSON file for policy evaluation",
)
@click.option(
    "--data", "-d", type=click.Path(exists=True), help="Additional data file for policy"
)
@click.option(
    "--policy",
    "-p",
    type=click.Path(exists=True),
    default="./policies/default",
    help="Policy bundle directory",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def test(policy_file: str, input_file: str, data: str, policy: str, verbose: bool):
    """Test a policy against input data

    Example:
        gl policy test allowlists.rego --input test_input.json
        gl policy test policies/default --input pack_manifest.json
    """
    try:
        # Load input document
        if not input_file:
            console.print("[red]Error: --input file is required[/red]")
            sys.exit(1)

        with open(input_file, "r") as f:
            input_doc = json.load(f)

        # Load optional data document
        data_doc = None
        if data:
            with open(data, "r") as f:
                data_doc = json.load(f)

        # Evaluate policy
        console.print(f"[cyan]Evaluating policy: {policy_file}[/cyan]")
        console.print(f"[cyan]Input: {input_file}[/cyan]")

        result = evaluate(policy_file, input_doc, data_doc)

        # Display result
        if result.get("allow", False):
            console.print(
                Panel.fit(
                    f"[bold green]✓ ALLOWED[/bold green]\n\n"
                    f"Reason: {result.get('reason', 'Policy conditions met')}",
                    title="Policy Decision",
                )
            )
        else:
            console.print(
                Panel.fit(
                    f"[bold red]✗ DENIED[/bold red]\n\n"
                    f"Reason: {result.get('reason', 'Policy conditions not met')}",
                    title="Policy Decision",
                )
            )

        # Verbose output
        if verbose:
            console.print("\n[yellow]Full Decision Document:[/yellow]")
            console.print(Syntax(json.dumps(result, indent=2), "json"))

        # Show deny reasons if available
        if "deny_reasons" in result and result["deny_reasons"]:
            console.print("\n[yellow]Denial Reasons:[/yellow]")
            for reason in result["deny_reasons"]:
                console.print(f"  • {reason}")

        # Show capabilities if evaluated
        if "capabilities" in result:
            console.print("\n[yellow]Capability Evaluation:[/yellow]")
            caps = result["capabilities"]
            if isinstance(caps, dict):
                for cap, allowed in caps.items():
                    status = "[green]✓[/green]" if allowed else "[red]✗[/red]"
                    console.print(f"  {status} {cap}")
            elif isinstance(caps, list):
                for cap in caps:
                    console.print(f"  [green]✓[/green] {cap}")

        # Exit with appropriate code
        sys.exit(0 if result.get("allow", False) else 1)

    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Policy evaluation error: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


@policy.command()
@click.argument("policy_file", type=click.Path(exists=True))
def validate(policy_file: str):
    """Validate a policy file for syntax errors

    Example:
        gl policy validate policies/default/allowlists.rego
    """
    try:
        console.print(f"[cyan]Validating policy: {policy_file}[/cyan]")

        is_valid, errors = validate_policy(policy_file)

        if is_valid:
            console.print("[bold green]✓ Policy is valid[/bold green]")
            sys.exit(0)
        else:
            console.print("[bold red]✗ Policy has errors:[/bold red]")
            for error in errors:
                console.print(f"  • {error}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        sys.exit(1)


@policy.command()
def list():
    """List all available policies

    Example:
        gl policy list
    """
    try:
        enforcer = PolicyEnforcer()
        policies = enforcer.list_policies()

        if not policies:
            console.print("[yellow]No policies found[/yellow]")
            return

        table = Table(title="Available Policies")
        table.add_column("Name", style="cyan")
        table.add_column("Location", style="green")

        for policy_name in policies:
            policy_path = enforcer.policy_dir / f"{policy_name}.rego"
            table.add_row(policy_name, str(policy_path))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing policies: {e}[/red]")
        sys.exit(1)


@policy.command()
def init():
    """Initialize default security policies

    Example:
        gl policy init
    """
    try:
        console.print("[cyan]Initializing default security policies...[/cyan]")

        enforcer = PolicyEnforcer()
        enforcer.create_default_policies()

        # Also copy the new default policies if they exist
        default_dir = Path("policies/default")
        if default_dir.exists():
            import shutil

            for policy_file in default_dir.glob("*.rego"):
                dest = enforcer.policy_dir / policy_file.name
                shutil.copy2(policy_file, dest)
                console.print(f"  [green]✓[/green] Copied {policy_file.name}")

        console.print("[bold green]✓ Default policies initialized[/bold green]")
        console.print(f"\nPolicies location: {enforcer.policy_dir}")
        console.print("\nTo test policies, run:")
        console.print("  gl policy test <policy> --input <input.json>")

    except Exception as e:
        console.print(f"[red]Error initializing policies: {e}[/red]")
        sys.exit(1)


@policy.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--policy",
    "-p",
    type=click.Path(exists=True),
    default="policies/default/allowlists.rego",
    help="Policy file to check against",
)
@click.option(
    "--stage",
    "-s",
    type=click.Choice(["install", "run"]),
    default="install",
    help="Policy stage to check",
)
def check(input_file: str, policy: str, stage: str):
    """Check if an operation would be allowed by policy

    Example:
        gl policy check pack_manifest.json
        gl policy check pipeline.yaml --stage run
    """
    try:
        # Load input
        with open(input_file, "r") as f:
            if input_file.endswith(".json"):
                input_data = json.load(f)
            else:
                import yaml

                input_data = yaml.safe_load(f)

        # Prepare input document based on stage
        if stage == "install":
            input_doc = {
                "stage": "install",
                "pack": input_data,
                "org": {
                    "allowed_publishers": ["greenlang-official", "verified"],
                    "allowed_regions": ["US", "EU"],
                    "allowed_capabilities": ["fs"],
                },
            }
        else:  # run
            input_doc = {
                "stage": "run",
                "pipeline": input_data,
                "user": {"authenticated": True, "role": "developer"},
                "env": {"region": "US"},
                "org": {
                    "allowed_publishers": ["greenlang-official", "verified"],
                    "allowed_regions": ["US", "EU"],
                    "allowed_capabilities": ["fs"],
                },
            }

        # Evaluate
        result = evaluate(policy, input_doc)

        # Display result
        if result.get("allow", False):
            console.print(f"[bold green]✓ Operation would be ALLOWED[/bold green]")
        else:
            console.print(f"[bold red]✗ Operation would be DENIED[/bold red]")
            console.print(f"Reason: {result.get('reason', 'Unknown')}")

        sys.exit(0 if result.get("allow", False) else 1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    policy()
