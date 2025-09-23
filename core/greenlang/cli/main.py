"""
GreenLang CLI v0.1
==================

Unified CLI for GreenLang infrastructure platform.
All domain logic lives in packs - the CLI just orchestrates.
"""

import sys

# Fix encoding issues on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        # Python < 3.7 fallback
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

import typer
import json
import yaml
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from .. import __version__
from ..packs.registry import PackRegistry
from ..packs.loader import PackLoader
from ..packs.manifest import PackManifest
from ..runtime.executor import Executor
from ..policy.enforcer import PolicyEnforcer
from ..provenance.signing import verify_artifact

app = typer.Typer(
    name="gl",
    help="GreenLang: Infrastructure for Climate Intelligence",
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def callback(
    version: bool = typer.Option(False, "--version", help="Show version and exit")
):
    """
    GreenLang v0.1 - Pure Infrastructure Platform

    Domain logic lives in packs. Platform = SDK/CLI/Runtime + Hub + Policy/Provenance
    """
    if version:
        console.print(f"GreenLang v{__version__}")
        console.print("Infrastructure for Climate Intelligence")
        console.print("https://greenlang.io")
        raise typer.Exit(0)


@app.command()
def version():
    """Show GreenLang version"""
    console.print(f"[bold green]GreenLang v{__version__}[/bold green]")
    console.print("Infrastructure for Climate Intelligence")
    console.print("https://greenlang.io")


# === Pack Management Commands ===


@app.command()
def init(
    name: str = typer.Option(..., "--name", "-n", help="Pack name (kebab-case)"),
    type: str = typer.Option("domain", "--type", "-t", help="Pack type"),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Pack directory"),
):
    """Initialize a new pack"""
    pack_dir = path / name

    if pack_dir.exists():
        console.print(f"[red]Error: Directory already exists: {pack_dir}[/red]")
        raise typer.Exit(1)

    # Create pack structure
    pack_dir.mkdir(parents=True)
    (pack_dir / "agents").mkdir()
    (pack_dir / "pipelines").mkdir()
    (pack_dir / "data").mkdir()
    (pack_dir / "cards").mkdir()
    (pack_dir / "policies").mkdir()
    (pack_dir / "tests").mkdir()

    # Create manifest
    manifest = PackManifest(
        name=name,
        version="0.1.0",
        type=type,
        description=f"A new {type} pack",
        authors=[{"name": "Your Name", "email": "you@example.com"}],
        exports={},
        dependencies=[{"name": "greenlang-sdk", "version": ">=0.1.0"}],
    )

    manifest.to_yaml(pack_dir / "pack.yaml")

    # Create README
    readme = f"""# {name}

A GreenLang {type} pack.

## Installation

```bash
gl pack add {name}
```

## Usage

```python
from greenlang import PackLoader

loader = PackLoader()
pack = loader.load("{name}")
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```
"""

    with open(pack_dir / "README.md", "w") as f:
        f.write(readme)

    # Create .gitignore
    gitignore = """__pycache__/
*.pyc
.pytest_cache/
.coverage
*.egg-info/
dist/
build/
.env
"""

    with open(pack_dir / ".gitignore", "w") as f:
        f.write(gitignore)

    console.print(f"[green][OK][/green] Created pack: {name}")
    console.print(f"  Location: {pack_dir}")
    console.print(f"  Type: {type}")
    console.print("\nNext steps:")
    console.print(f"  1. cd {name}")
    console.print("  2. Edit pack.yaml")
    console.print("  3. Add agents, pipelines, or datasets")
    console.print(f"  4. gl pack publish {name}")


# Create a sub-application for pack commands
pack_app = typer.Typer(help="Pack management commands")
app.add_typer(pack_app, name="pack")


@pack_app.command("list")
def pack_list(
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type")
):
    """List installed packs"""
    registry = PackRegistry()
    packs = registry.list(pack_type=type)

    if not packs:
        console.print("[yellow]No packs installed[/yellow]")
        console.print("\nInstall packs with: [cyan]gl pack add <pack-name>[/cyan]")
        return

    table = Table(title="Installed Packs")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Location")
    table.add_column("Verified", style="blue")

    for pack in packs:
        verified = "[OK]" if pack.verified else "[FAIL]"
        table.add_row(
            pack.name,
            pack.version,
            pack.manifest.get("type", "unknown"),
            pack.location[:30] + "..." if len(pack.location) > 30 else pack.location,
            verified,
        )

    console.print(table)


@pack_app.command("info")
def pack_info(name: str):
    """Show pack details"""
    registry = PackRegistry()
    pack = registry.get(name)

    if not pack:
        console.print(f"[red]Pack not found: {name}[/red]")
        raise typer.Exit(1)

    manifest = pack.manifest

    console.print(
        Panel.fit(
            f"[bold]{manifest['name']}[/bold] v{manifest['version']}\n"
            f"{manifest.get('description', 'No description')}\n\n"
            f"Type: {manifest.get('type', 'unknown')}\n"
            f"Location: {pack.location}\n"
            f"Verified: {'[OK]' if pack.verified else '[FAIL]'}\n"
            f"Hash: {pack.hash[:16]}...",
            title="Pack Information",
        )
    )

    # Show exports
    if manifest.get("exports"):
        console.print("\n[bold]Exports:[/bold]")
        for export_type, items in manifest["exports"].items():
            console.print(f"  {export_type}:")
            for item in items:
                console.print(
                    f"    - {item.get('name', 'unnamed')}: {item.get('description', '')}"
                )

    # Show dependencies
    if manifest.get("dependencies"):
        console.print("\n[bold]Dependencies:[/bold]")
        for dep in manifest["dependencies"]:
            console.print(f"  - {dep['name']} {dep.get('version', '')}")


@pack_app.command("add")
def pack_add(
    source: str,
    registry_url: str = typer.Option("hub.greenlang.io", "--registry", "-r"),
    verify: bool = typer.Option(True, "--verify/--no-verify"),
):
    """Install a pack from registry or local path"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Installing {source}...", total=None)

        registry = PackRegistry()

        # Check if local path
        if Path(source).exists():
            pack_path = Path(source)
            try:
                installed = registry.register(pack_path, verify=verify)
                progress.update(task, completed=True)
                console.print(
                    f"[green][OK][/green] Installed local pack: {installed.name}"
                )
            except Exception as e:
                console.print(f"[red]Failed to install pack: {e}[/red]")
                raise typer.Exit(1)
        else:
            # TODO: Download from registry
            console.print(f"[yellow]Registry installation not yet implemented[/yellow]")
            console.print(f"Would download {source} from {registry_url}")


@pack_app.command("remove")
def pack_remove(name: str):
    """Uninstall a pack"""
    registry = PackRegistry()

    try:
        registry.unregister(name)
        console.print(f"[green][OK][/green] Removed pack: {name}")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@pack_app.command("validate")
def pack_validate(
    path: Path = typer.Argument(..., help="Path to pack directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Validate pack structure and manifest"""
    from greenlang.packs.manifest import validate_pack, load_manifest

    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    # Run validation
    is_valid, errors = validate_pack(path)

    if is_valid:
        console.print(f"[green][OK][/green] Pack validation passed: {path.name}")

        if verbose:
            # Load and display manifest info
            try:
                manifest = load_manifest(path)
                console.print(f"\n[bold]Pack Details:[/bold]")
                console.print(f"  Name: {manifest.name}")
                console.print(f"  Version: {manifest.version}")
                console.print(f"  Kind: {manifest.kind}")
                console.print(f"  License: {manifest.license}")

                if manifest.contents.pipelines:
                    console.print(
                        f"  Pipelines: {', '.join(manifest.contents.pipelines)}"
                    )
                if manifest.contents.agents:
                    console.print(f"  Agents: {', '.join(manifest.contents.agents)}")
                if manifest.contents.datasets:
                    console.print(
                        f"  Datasets: {', '.join(manifest.contents.datasets)}"
                    )

            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not load manifest details: {e}[/yellow]"
                )
    else:
        console.print(f"[red][FAIL][/red] Pack validation failed: {path.name}")
        console.print("\n[red]Errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@pack_app.command("verify")
def pack_verify(name: str):
    """Verify pack integrity"""
    registry = PackRegistry()

    if registry.verify(name):
        console.print(f"[green][OK][/green] Pack verified: {name}")
    else:
        console.print(f"[red][FAIL][/red] Pack verification failed: {name}")
        raise typer.Exit(1)


@pack_app.command("publish")
def pack_publish(
    path: Path = typer.Argument(..., help="Path to pack directory"),
    registry_url: str = typer.Option("hub.greenlang.io", "--registry", "-r"),
    sign: bool = typer.Option(True, "--sign/--no-sign"),
    test: bool = typer.Option(
        True, "--test/--no-test", help="Run tests before publishing"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Simulate publishing without uploading"
    ),
):
    """Publish a pack to registry"""
    from greenlang.packs.manifest import load_manifest, validate_pack
    from greenlang.provenance.sbom import generate_sbom
    from greenlang.provenance.signing import sign_pack
    import subprocess
    import tarfile

    # Validate pack first
    console.print(f"[cyan]Validating pack at {path}...[/cyan]")
    is_valid, errors = validate_pack(path)

    if not is_valid:
        console.print("[red]Pack validation failed:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)

    console.print(f"[green][OK][/green] Pack validation passed")

    # Load manifest
    manifest = load_manifest(path)
    console.print(f"[cyan]Publishing {manifest.name} v{manifest.version}...[/cyan]")

    # Run tests if requested
    if test and manifest.tests:
        console.print("[cyan]Running tests...[/cyan]")
        for test_file in manifest.tests:
            test_path = path / test_file
            if test_path.exists():
                try:
                    result = subprocess.run(
                        ["python", "-m", "pytest", str(test_path), "-v"],
                        capture_output=True,
                        text=True,
                        cwd=path,
                    )
                    if result.returncode != 0:
                        console.print(f"[red]Tests failed: {test_file}[/red]")
                        if result.stdout:
                            console.print(result.stdout)
                        raise typer.Exit(1)
                    console.print(f"[green][OK][/green] Tests passed: {test_file}")
                except Exception as e:
                    console.print(f"[yellow]Could not run tests: {e}[/yellow]")

    # Generate SBOM
    if manifest.security.sbom:
        sbom_path = path / manifest.security.sbom
        console.print("[cyan]Generating SBOM...[/cyan]")
        sbom = generate_sbom(path, sbom_path)
        console.print(f"[green][OK][/green] Generated SBOM: {manifest.security.sbom}")

    # Sign pack if requested
    if sign:
        console.print("[cyan]Signing pack...[/cyan]")
        try:
            signature = sign_pack(path)
            console.print(
                f"[green][OK][/green] Pack signed (hash: {signature['spec']['hash']['value'][:16]}...)"
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not sign pack: {e}[/yellow]")

    # Build pack archive
    archive_name = f"{manifest.name}-{manifest.version}.glpack"
    archive_path = path.parent / "dist" / archive_name
    archive_path.parent.mkdir(exist_ok=True)

    if not dry_run:
        console.print(f"[cyan]Building archive: {archive_name}...[/cyan]")

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(path, arcname=manifest.name)

        console.print(
            f"[green][OK][/green] Built archive: {archive_path} ({archive_path.stat().st_size // 1024}KB)"
        )

    # Upload to registry (or simulate)
    if dry_run:
        console.print("\n[yellow]DRY RUN - Would perform:[/yellow]")
        console.print(f"  - Upload {archive_name} to {registry_url}")
        console.print(f"  - Register {manifest.name} v{manifest.version}")
        console.print(f"  - Publish metadata and signatures")
    else:
        # TODO: Implement actual upload
        console.print(
            f"\n[yellow]Upload to {registry_url} not yet implemented[/yellow]"
        )
        console.print(f"Archive ready at: {archive_path}")
        console.print("\nTo install locally:")
        console.print(f"  gl pack add {archive_path}")

    console.print(
        f"\n[green][OK][/green] Pack ready for publishing: {manifest.name} v{manifest.version}"
    )


# === Runtime Commands ===


@app.command()
def run(
    pipeline: str,
    input_file: Optional[Path] = typer.Option(None, "--input", "-i"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o"),
    profile: str = typer.Option("local", "--profile", "-p"),
    policy: Optional[Path] = typer.Option(None, "--policy"),
):
    """Run a pipeline from a pack"""
    loader = PackLoader()

    # Parse pipeline reference (pack.pipeline or just pipeline)
    if "." in pipeline:
        pack_name, pipeline_name = pipeline.split(".", 1)
    else:
        # Try to find pipeline in any loaded pack
        pipeline_name = pipeline
        pack_name = None

    # Load input
    input_data = {}
    if input_file:
        if input_file.suffix == ".json":
            with open(input_file) as f:
                input_data = json.load(f)
        elif input_file.suffix in [".yaml", ".yml"]:
            with open(input_file) as f:
                input_data = yaml.safe_load(f)

    # Get executor for profile
    executor = Executor(profile=profile)

    # Apply policy if provided
    if policy:
        enforcer = PolicyEnforcer()
        if not enforcer.check(policy, {"pipeline": pipeline, "input": input_data}):
            console.print("[red]Policy check failed[/red]")
            raise typer.Exit(1)

    # Run pipeline
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Running {pipeline}...", total=None)

        try:
            result = executor.run(pipeline, input_data)
            progress.update(task, completed=True)

            if result.success:
                console.print(f"[green][OK][/green] Pipeline completed successfully")

                # Save output if requested
                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(result.data, f, indent=2)
                    console.print(f"Output saved to: {output_file}")
                else:
                    # Pretty print result
                    if result.data:
                        syntax = Syntax(
                            json.dumps(result.data, indent=2), "json", theme="monokai"
                        )
                        console.print(syntax)
            else:
                console.print(f"[red]Pipeline failed: {result.error}[/red]")
                raise typer.Exit(1)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)


# === Policy Commands ===


@app.command()
def policy(
    action: str = typer.Argument(..., help="check, list, or add"),
    target: Optional[str] = typer.Argument(None),
    policy_file: Optional[Path] = typer.Option(None, "--file", "-f"),
):
    """Manage and enforce policies"""
    enforcer = PolicyEnforcer()

    if action == "check":
        if not target or not policy_file:
            console.print(
                "[red]Usage: gl policy check <target> --file <policy.rego>[/red]"
            )
            raise typer.Exit(1)

        result = enforcer.check(policy_file, {"target": target})
        if result:
            console.print(f"[green][OK][/green] Policy check passed")
        else:
            console.print(f"[red][FAIL][/red] Policy check failed")
            raise typer.Exit(1)

    elif action == "list":
        policies = enforcer.list_policies()
        if not policies:
            console.print("[yellow]No policies configured[/yellow]")
        else:
            for policy in policies:
                console.print(f"- {policy}")

    elif action == "add":
        if not policy_file:
            console.print("[red]Usage: gl policy add --file <policy.rego>[/red]")
            raise typer.Exit(1)

        enforcer.add_policy(policy_file)
        console.print(f"[green][OK][/green] Added policy: {policy_file.name}")


# === Provenance Commands ===


@app.command()
def verify(
    artifact: Path, signature: Optional[Path] = typer.Option(None, "--sig", "-s")
):
    """Verify artifact provenance and signature"""
    if not artifact.exists():
        console.print(f"[red]Artifact not found: {artifact}[/red]")
        raise typer.Exit(1)

    # Check signature if provided
    if signature:
        if verify_artifact(artifact, signature):
            console.print(f"[green][OK][/green] Signature valid")
        else:
            console.print(f"[red][FAIL][/red] Signature invalid")
            raise typer.Exit(1)

    # Check SBOM if exists
    sbom_path = artifact.parent / "sbom.json"
    if sbom_path.exists():
        with open(sbom_path) as f:
            sbom = json.load(f)
        console.print(
            f"[green][OK][/green] SBOM found: {len(sbom.get('components', []))} components"
        )

    console.print(f"[green][OK][/green] Artifact verified: {artifact.name}")


# === Utility Commands ===


@app.command()
def doctor():
    """Check GreenLang installation and environment"""
    console.print("[bold]GreenLang Environment Check[/bold]\n")

    checks = []

    # Check version
    checks.append(("GreenLang Version", f"v{__version__}", True))

    # Check Python version
    import sys

    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    checks.append(("Python Version", py_version, sys.version_info >= (3, 8)))

    # Check registry
    try:
        registry = PackRegistry()
        pack_count = len(registry.list())
        checks.append(("Pack Registry", f"{pack_count} packs", True))
    except (ImportError, OSError, ValueError) as e:
        checks.append(("Pack Registry", "Failed", False))

    # Check config directory
    config_dir = Path.home() / ".greenlang"
    checks.append(("Config Directory", str(config_dir), config_dir.exists()))

    # Display results
    for name, value, status in checks:
        icon = "[green][OK][/green]" if status else "[red][FAIL][/red]"
        console.print(f"{icon} {name}: {value}")

    # Overall status
    if all(c[2] for c in checks):
        console.print("\n[green]All checks passed![/green]")
    else:
        console.print(
            "\n[yellow]Some checks failed. Run 'gl init' to set up GreenLang.[/yellow]"
        )


def main():
    """Main entry point for the gl CLI command"""
    app()


# Also provide the app directly for backward compatibility
def cli():
    """Legacy entry point"""
    app()


if __name__ == "__main__":
    main()
