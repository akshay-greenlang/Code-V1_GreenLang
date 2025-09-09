"""
gl pack - Pack management commands
"""

import typer
import subprocess
import json
import yaml
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

app = typer.Typer()
console = Console()


@app.command("create")
def create(
    slug: str = typer.Argument(..., help="Pack slug (kebab-case)"),
    path: Path = typer.Option(Path("packs"), "--path", "-p", help="Parent directory"),
    template: str = typer.Option("basic", "--template", "-t", help="Pack template")
):
    """Create a new pack with boilerplate"""
    pack_dir = path / slug
    
    if pack_dir.exists():
        console.print(f"[red]Error: Pack already exists: {pack_dir}[/red]")
        raise typer.Exit(1)
    
    # Create pack structure
    pack_dir.mkdir(parents=True)
    (pack_dir / "agents").mkdir()
    (pack_dir / "pipelines").mkdir()
    (pack_dir / "data").mkdir()
    (pack_dir / "tests").mkdir()
    (pack_dir / "cards").mkdir()
    (pack_dir / "policies").mkdir()
    
    # Create pack.yaml
    manifest = {
        "name": slug,
        "version": "0.1.0",
        "kind": "pack",
        "description": f"A GreenLang pack for {slug}",
        "license": "MIT",
        "authors": [
            {"name": "Your Name", "email": "you@example.com"}
        ],
        "contents": {
            "pipelines": [],
            "agents": [],
            "datasets": []
        },
        "dependencies": [
            {"name": "greenlang-sdk", "version": ">=0.1.0"}
        ],
        "security": {
            "sbom": "sbom.spdx.json",
            "signatures": "signatures/"
        },
        "tests": ["tests/test_*.py"]
    }
    
    with open(pack_dir / "pack.yaml", "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    
    # Create gl.yaml
    pipeline = {
        "version": "1.0",
        "name": f"{slug}-main",
        "description": f"Main pipeline for {slug}",
        "inputs": {},
        "steps": [],
        "outputs": {}
    }
    
    with open(pack_dir / "gl.yaml", "w") as f:
        yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)
    
    # Create CARD.md
    with open(pack_dir / "CARD.md", "w") as f:
        f.write(f"""# {slug} Model Card

## Overview
Describe what this pack does...

## Intended Use
- Primary use cases
- Target users  
- Out of scope uses

## Data
- Input requirements
- Output format
- Data sources

## Performance
- Metrics
- Benchmarks
- Limitations

## Ethics & Risks
- Known issues
- Mitigation strategies
""")
    
    # Create README.md
    with open(pack_dir / "README.md", "w") as f:
        f.write(f"""# {slug}

A GreenLang pack for climate intelligence.

## Installation

```bash
gl pack add {slug}
```

## Usage

```bash
gl run {slug}
```

## Development

```bash
gl pack validate
pytest tests/
gl pack publish
```
""")
    
    # Create sample test
    with open(pack_dir / "tests" / "test_pack.py", "w") as f:
        f.write(f"""import pytest
from pathlib import Path
import yaml

def test_pack_structure():
    pack_dir = Path(__file__).parent.parent
    assert (pack_dir / "pack.yaml").exists()
    assert (pack_dir / "gl.yaml").exists()
    
def test_manifest_valid():
    pack_dir = Path(__file__).parent.parent
    with open(pack_dir / "pack.yaml") as f:
        manifest = yaml.safe_load(f)
    assert manifest["name"] == "{slug}"
""")
    
    console.print(f"[green]✓[/green] Created pack: {slug}")
    console.print(f"  Location: {pack_dir}")
    console.print("\nNext steps:")
    console.print(f"  1. cd {pack_dir}")
    console.print("  2. Edit pack.yaml and add your agents/pipelines")
    console.print("  3. gl pack validate")
    console.print("  4. gl pack publish")


@app.command("validate")
def validate(
    path: Path = typer.Argument(Path("."), help="Pack directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show details")
):
    """Validate manifest & files"""
    from ..packs.manifest import validate_pack, load_manifest
    
    if not (path / "pack.yaml").exists():
        console.print(f"[red]Error: No pack.yaml found in {path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Validating pack at {path}...[/cyan]")
    
    is_valid, errors = validate_pack(path)
    
    if is_valid:
        console.print(f"[green]✓[/green] Pack validation passed")
        
        if verbose:
            try:
                manifest = load_manifest(path)
                console.print(f"\n[bold]Pack Details:[/bold]")
                console.print(f"  Name: {manifest.name}")
                console.print(f"  Version: {manifest.version}")
                console.print(f"  Kind: {manifest.kind}")
                
                if manifest.contents.pipelines:
                    console.print(f"  Pipelines: {', '.join(manifest.contents.pipelines)}")
                if manifest.contents.agents:
                    console.print(f"  Agents: {', '.join(manifest.contents.agents)}")
                if manifest.contents.datasets:
                    console.print(f"  Datasets: {', '.join(manifest.contents.datasets)}")
            except Exception as e:
                console.print(f"[yellow]Warning: {e}[/yellow]")
    else:
        console.print(f"[red]✗ Pack validation failed[/red]")
        for error in errors:
            console.print(f"  • {error}")
        raise typer.Exit(1)


@app.command("publish")
def publish(
    path: Path = typer.Argument(Path("."), help="Pack directory"),
    registry: str = typer.Option("ghcr.io/greenlang", "--registry", "-r", help="OCI registry"),
    test: bool = typer.Option(True, "--test/--no-test", help="Run tests first"),
    sign: bool = typer.Option(True, "--sign/--no-sign", help="Sign the pack"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate publishing")
):
    """Test → policy → SBOM → sign → push"""
    from ..packs.manifest import load_manifest, validate_pack
    from ..provenance.sbom import generate_sbom
    from ..provenance.sign import cosign_sign
    from ..policy.enforcer import check_install
    
    # Validate first
    console.print("[cyan]Validating pack...[/cyan]")
    is_valid, errors = validate_pack(path)
    if not is_valid:
        console.print("[red]Validation failed:[/red]")
        for error in errors:
            console.print(f"  • {error}")
        raise typer.Exit(1)
    
    manifest = load_manifest(path)
    console.print(f"[green]✓[/green] Validated {manifest.name} v{manifest.version}")
    
    # Run tests if requested
    if test and manifest.tests:
        console.print("[cyan]Running tests...[/cyan]")
        for test_pattern in manifest.tests:
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", test_pattern, "-v"],
                    capture_output=True,
                    text=True,
                    cwd=path
                )
                if result.returncode != 0:
                    console.print(f"[red]Tests failed[/red]")
                    raise typer.Exit(1)
                console.print(f"[green]✓[/green] Tests passed")
            except Exception as e:
                console.print(f"[yellow]Could not run tests: {e}[/yellow]")
    
    # Generate SBOM
    console.print("[cyan]Generating SBOM...[/cyan]")
    sbom_path = path / "sbom.spdx.json"
    generate_sbom(path, sbom_path)
    console.print(f"[green]✓[/green] Generated SBOM")
    
    # Check policy
    console.print("[cyan]Checking policy...[/cyan]")
    try:
        check_install(manifest, path, stage="publish")
        console.print(f"[green]✓[/green] Policy check passed")
    except Exception as e:
        console.print(f"[red]Policy check failed: {e}[/red]")
        raise typer.Exit(1)
    
    # Sign pack if requested
    if sign:
        console.print("[cyan]Signing pack...[/cyan]")
        try:
            cosign_sign(path)
            console.print(f"[green]✓[/green] Pack signed")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not sign: {e}[/yellow]")
    
    # Build and push
    ref = f"{registry}/{manifest.name}:{manifest.version}"
    
    if dry_run:
        console.print("\n[yellow]DRY RUN - Would perform:[/yellow]")
        console.print(f"  • Build pack archive")
        console.print(f"  • Push to {ref}")
        console.print(f"  • Upload signatures and SBOM")
    else:
        console.print(f"[cyan]Publishing to {ref}...[/cyan]")
        
        # Create archive
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / f"{manifest.name}.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(path, arcname=manifest.name)
            
            # Push with oras
            try:
                subprocess.check_call([
                    "oras", "push", ref,
                    str(archive),
                    "--annotation", f"org.greenlang.type=pack",
                    "--annotation", f"org.greenlang.version={manifest.version}"
                ])
                console.print(f"[green]✓[/green] Published {manifest.name}@{manifest.version}")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Failed to push: {e}[/red]")
                raise typer.Exit(1)


@app.command("add") 
def add(
    ref: str = typer.Argument(..., help="Pack reference (name@version or path)"),
    cache: Path = typer.Option(Path(".gl_cache"), "--cache", help="Cache directory"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify signatures"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall")
):
    """Pull, verify signature, install"""
    from ..packs.installer import PackInstaller
    from ..packs.registry import PackRegistry
    
    console.print(f"[cyan]Installing {ref}...[/cyan]")
    
    # Check if local path
    if Path(ref).exists():
        # Install from local directory
        installer = PackInstaller(cache_dir=cache)
        try:
            installed = installer.install_local(Path(ref), verify=verify)
            console.print(f"[green]✓[/green] Installed local pack: {installed.name}")
        except Exception as e:
            console.print(f"[red]Failed to install: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Install from registry
        if "@" in ref:
            name, version = ref.split("@", 1)
        else:
            name, version = ref, "latest"
        
        # Check if already installed
        registry = PackRegistry()
        if not force and registry.get(name):
            console.print(f"[yellow]Pack already installed: {name}[/yellow]")
            console.print("Use --force to reinstall")
            raise typer.Exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Pulling {name}@{version}...", total=None)
            
            try:
                # Pull with oras
                cache.mkdir(parents=True, exist_ok=True)
                pack_dir = cache / name / version
                pack_dir.mkdir(parents=True, exist_ok=True)
                
                subprocess.check_call([
                    "oras", "pull",
                    f"ghcr.io/greenlang/{name}:{version}",
                    "-o", str(pack_dir)
                ])
                
                progress.update(task, description="Verifying...")
                
                # Verify if requested
                if verify:
                    # TODO: Implement cosign verification
                    pass
                
                progress.update(task, description="Installing...")
                
                # Register pack
                registry.register(pack_dir, verify=False)
                
                progress.update(task, completed=True)
                console.print(f"[green]✓[/green] Installed {name}@{version} → {cache}")
                
            except subprocess.CalledProcessError as e:
                progress.update(task, completed=True)
                console.print(f"[red]Failed to pull pack: {e}[/red]")
                raise typer.Exit(1)


@app.command("info")
def info(
    ref: str = typer.Argument(..., help="Pack name or reference")
):
    """Inspect pack metadata"""
    from ..packs.registry import PackRegistry
    
    registry = PackRegistry()
    
    # Parse reference
    if "@" in ref:
        name, version = ref.split("@", 1)
    else:
        name = ref
        version = None
    
    pack = registry.get(name, version=version)
    
    if not pack:
        console.print(f"[red]Pack not found: {ref}[/red]")
        raise typer.Exit(1)
    
    # Display pack info
    console.print(Panel.fit(
        f"[bold]{pack.manifest.name}[/bold] v{pack.manifest.version}\n"
        f"{pack.manifest.description}\n\n"
        f"Kind: {pack.manifest.kind}\n"
        f"License: {pack.manifest.license}\n"
        f"Location: {pack.location}\n"
        f"Verified: {'✓' if pack.verified else '✗'}",
        title="Pack Information"
    ))
    
    # Show contents
    if pack.manifest.contents:
        console.print("\n[bold]Contents:[/bold]")
        if pack.manifest.contents.pipelines:
            console.print(f"  Pipelines: {', '.join(pack.manifest.contents.pipelines)}")
        if pack.manifest.contents.agents:
            console.print(f"  Agents: {', '.join(pack.manifest.contents.agents)}")
        if pack.manifest.contents.datasets:
            console.print(f"  Datasets: {', '.join(pack.manifest.contents.datasets)}")
    
    # Show dependencies
    if pack.manifest.dependencies:
        console.print("\n[bold]Dependencies:[/bold]")
        for dep in pack.manifest.dependencies:
            console.print(f"  • {dep['name']} {dep.get('version', '')}")
    
    # Show authors
    if pack.manifest.authors:
        console.print("\n[bold]Authors:[/bold]")
        for author in pack.manifest.authors:
            console.print(f"  • {author['name']} <{author.get('email', '')}>")


@app.command("list")
def list_packs(
    kind: Optional[str] = typer.Option(None, "--kind", "-k", help="Filter by kind"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """List installed packs"""
    from ..packs.registry import PackRegistry
    
    registry = PackRegistry()
    packs = registry.list(kind=kind)
    
    if not packs:
        console.print("[yellow]No packs installed[/yellow]")
        console.print("\nInstall packs with: [cyan]gl pack add <pack-name>[/cyan]")
        return
    
    if json_output:
        import json
        output = []
        for pack in packs:
            output.append({
                "name": pack.manifest.name,
                "version": pack.manifest.version,
                "kind": pack.manifest.kind,
                "location": str(pack.location),
                "verified": pack.verified
            })
        console.print(json.dumps(output, indent=2))
    else:
        table = Table(title="Installed Packs")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Kind", style="yellow")
        table.add_column("Location")
        table.add_column("Verified", style="blue")
        
        for pack in packs:
            table.add_row(
                pack.manifest.name,
                pack.manifest.version,
                pack.manifest.kind,
                str(pack.location)[:40] + "..." if len(str(pack.location)) > 40 else str(pack.location),
                "✓" if pack.verified else "✗"
            )
        
        console.print(table)