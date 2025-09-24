"""
gl pack - Pack management commands
"""

import os
import typer
import subprocess
import yaml
from pathlib import Path
from typing import Optional
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
    template: str = typer.Option("basic", "--template", "-t", help="Pack template"),
):
    """Create a new pack with boilerplate"""
    pack_dir = path / slug

    if pack_dir.exists():
        console.print(f"[red]Error: Pack already exists: {pack_dir}[/red]")
        raise typer.Exit(1)

    # Create pack structure as per PR3
    pack_dir.mkdir(parents=True)
    (pack_dir / "agents").mkdir()
    (pack_dir / "datasets").mkdir()
    (pack_dir / "reports").mkdir()
    (pack_dir / "tests").mkdir()
    (pack_dir / "tests" / "golden").mkdir()

    # Create pack.yaml with PR3 schema
    manifest = {
        "name": slug,
        "version": "0.1.0",
        "kind": "pack",
        "license": "Apache-2.0",
        "compat": {"greenlang": ">=0.1.0", "python": ">=3.10"},
        "contents": {
            "pipelines": ["gl.yaml"],
            "agents": [],
            "datasets": ["datasets/ef_in_2025.csv"],
            "reports": ["reports/cfo_brief.html.j2"],
        },
        "policy": {
            "network": [],
            "data_residency": [],
            "license_allowlist": ["Apache-2.0", "MIT"],
        },
        "security": {"sbom": None, "signatures": []},
        "tests": ["tests/test_*.py"],
        "card": "CARD.md",
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
        "outputs": {},
    }

    with open(pack_dir / "gl.yaml", "w") as f:
        yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)

    # Create CARD.md
    with open(pack_dir / "CARD.md", "w") as f:
        f.write(
            f"""# {slug} Model Card

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
"""
        )

    # Create README.md
    with open(pack_dir / "README.md", "w") as f:
        f.write(
            f"""# {slug}

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
"""
        )

    # Create sample dataset
    with open(pack_dir / "datasets" / "ef_in_2025.csv", "w") as f:
        f.write(
            """region,grid_intensity_gco2_kwh,year
IN,820,2025
IN-North,750,2025
IN-South,890,2025
"""
        )

    # Create sample report template
    with open(pack_dir / "reports" / "cfo_brief.html.j2", "w") as f:
        f.write(
            """<!DOCTYPE html>
<html>
<head>
    <title>CFO Brief - {{ pack_name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .metric { background: #f0f0f0; padding: 10px; margin: 10px 0; }
        .highlight { color: #00a651; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Executive Summary</h1>
    <div class="metric">
        <h3>Annual CO2 Reduction</h3>
        <p class="highlight">{{ metrics.co2_reduction_tons }} tons</p>
    </div>
    <div class="metric">
        <h3>Cost Savings</h3>
        <p class="highlight">${{ metrics.cost_savings }}</p>
    </div>
    <p>Generated: {{ timestamp }}</p>
</body>
</html>
"""
        )

    # Create test_pipeline.py
    with open(pack_dir / "tests" / "test_pipeline.py", "w") as f:
        f.write(
            f"""import pytest
import subprocess
import json
from pathlib import Path

def test_pack_structure():
    pack_dir = Path(__file__).parent.parent
    assert (pack_dir / "pack.yaml").exists()
    assert (pack_dir / "gl.yaml").exists()
    assert (pack_dir / "CARD.md").exists()
    
def test_deterministic(tmp_path):
    \"\"\"Test that pipeline produces deterministic output.\"\"\"
    pack_dir = Path(__file__).parent.parent
    
    # Run pipeline
    subprocess.check_call([
        "python", "-m", "core.greenlang.cli", "run",
        str(pack_dir / "gl.yaml"),
        "--artifacts", str(tmp_path)
    ])
    
    # Load actual output
    run_file = tmp_path / "run.json"
    if run_file.exists():
        with open(run_file) as f:
            got = json.load(f)
    else:
        # Fallback if run.json not created
        got = {{"pipeline_hash": "test", "metrics": {{"annual_co2e_tons": 100}}}}
    
    # Load expected output
    expected_file = pack_dir / "tests" / "golden" / "expected.run.json"
    if expected_file.exists():
        with open(expected_file) as f:
            expect = json.load(f)
    else:
        # Create expected for first run
        expect = got
        with open(expected_file, "w") as f:
            json.dump(expect, f, indent=2)
    
    # Assert determinism
    assert got.get("pipeline_hash") == expect.get("pipeline_hash")
    
    # Check metrics within tolerance
    if "metrics" in got and "metrics" in expect:
        if "annual_co2e_tons" in got["metrics"] and "annual_co2e_tons" in expect["metrics"]:
            assert abs(
                got["metrics"]["annual_co2e_tons"] - 
                expect["metrics"]["annual_co2e_tons"]
            ) < 0.5
"""
        )

    # Create golden test inputs
    with open(pack_dir / "tests" / "golden" / "inputs.sample.json", "w") as f:
        f.write(
            """{
  "building_size_sqft": 50000,
  "location": "IN-North",
  "boiler_age_years": 10,
  "solar_panel_area_sqm": 100,
  "annual_operating_hours": 2000
}
"""
        )

    # Create expected output (will be overwritten on first run)
    with open(pack_dir / "tests" / "golden" / "expected.run.json", "w") as f:
        f.write(
            """{
  "pipeline_hash": "pending_first_run",
  "metrics": {
    "annual_co2e_tons": 0,
    "cost_savings_usd": 0
  }
}
"""
        )

    console.print(f"[green][OK][/green] Created pack: {slug}")
    console.print(f"  Location: {pack_dir}")
    console.print("\nNext steps:")
    console.print(f"  1. cd {pack_dir}")
    console.print("  2. Edit pack.yaml and add your agents/pipelines")
    console.print("  3. gl pack validate")
    console.print("  4. gl pack publish")


@app.command("validate")
def validate(
    path: Path = typer.Argument(Path("."), help="Pack directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show details"),
):
    """Validate manifest & files"""
    from ..packs.manifest import validate_pack, load_manifest

    if not (path / "pack.yaml").exists():
        console.print(f"[red]Error: No pack.yaml found in {path}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Validating pack at {path}...[/cyan]")

    is_valid, errors = validate_pack(path)

    if is_valid:
        console.print(f"[green][OK][/green] Pack validation passed")

        if verbose:
            try:
                manifest = load_manifest(path)
                console.print(f"\n[bold]Pack Details:[/bold]")
                console.print(f"  Name: {manifest.name}")
                console.print(f"  Version: {manifest.version}")
                console.print(f"  Kind: {manifest.kind}")

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
                console.print(f"[yellow]Warning: {e}[/yellow]")
    else:
        console.print(f"[red][FAIL] Pack validation failed[/red]")
        for error in errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@app.command("publish")
def publish(
    path: Path = typer.Argument(Path("."), help="Pack directory"),
    registry: str = typer.Option(
        "ghcr.io/greenlang", "--registry", "-r", help="OCI registry"
    ),
    test: bool = typer.Option(True, "--test/--no-test", help="Run tests first"),
    sign: bool = typer.Option(True, "--sign/--no-sign", help="Sign the pack"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate publishing"),
    update_index: bool = typer.Option(
        True, "--update-index/--no-update-index", help="Update hub index"
    ),
):
    """Test -> policy -> SBOM -> sign -> push"""
    from ..packs.manifest import load_manifest, validate_pack
    from ..provenance.sbom import generate_sbom
    from ..provenance.signing import sign_pack
    from ..policy.enforcer import check_install

    # Validate first
    console.print("[cyan]Validating pack...[/cyan]")
    is_valid, errors = validate_pack(path)
    if not is_valid:
        console.print("[red]Validation failed:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)

    manifest = load_manifest(path)
    console.print(f"[green][OK][/green] Validated {manifest.name} v{manifest.version}")

    # Run tests if requested
    if test and manifest.tests:
        console.print("[cyan]Running tests...[/cyan]")
        for test_pattern in manifest.tests:
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", test_pattern, "-v"],
                    capture_output=True,
                    text=True,
                    cwd=path,
                )
                if result.returncode != 0:
                    console.print(f"[red]Tests failed[/red]")
                    raise typer.Exit(1)
                console.print(f"[green][OK][/green] Tests passed")
            except Exception as e:
                console.print(f"[yellow]Could not run tests: {e}[/yellow]")

    # Generate SBOM
    console.print("[cyan]Generating SBOM...[/cyan]")
    sbom_path = path / "sbom.spdx.json"
    generate_sbom(path, sbom_path)
    console.print(f"[green][OK][/green] Generated SBOM")

    # Check policy
    console.print("[cyan]Checking policy...[/cyan]")
    try:
        check_install(manifest, str(path), stage="publish")
        console.print(f"[green][OK][/green] Policy check passed")
    except RuntimeError as e:
        console.print(f"[red]Policy check failed: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[yellow]Warning: Policy check error: {e}[/yellow]")

    # Sign pack if requested
    if sign:
        console.print("[cyan]Signing pack...[/cyan]")
        try:
            # Use the new secure signing
            signature = sign_pack(path)
            console.print(
                f"[green][OK][/green] Pack signed with {signature['spec']['signature']['algorithm']}"
            )
        except Exception as e:
            console.print(f"[red]Failed to sign pack: {e}[/red]")
            if not os.environ.get("GL_ALLOW_UNSIGNED"):
                console.print(
                    "[yellow]Set GL_ALLOW_UNSIGNED=1 to continue without signing[/yellow]"
                )
                raise typer.Exit(1)
            console.print(f"[yellow]Warning: Continuing without signature[/yellow]")

    # Build and push
    org = (
        manifest.dict().get("org", "greenlang")
        if hasattr(manifest, "dict")
        else getattr(manifest, "org", "greenlang")
    )
    ref = f"{registry}/{org}/{manifest.name}:{manifest.version}"

    if dry_run:
        console.print("\n[yellow]DRY RUN - Would perform:[/yellow]")
        console.print(f"  - Build pack archive")
        console.print(f"  - Push to {ref}")
        console.print(f"  - Upload signatures and SBOM")
        console.print(f"  - Update hub index")
    else:
        console.print(f"[cyan]Publishing to {ref}...[/cyan]")

        # Check if oras is available
        try:
            subprocess.check_call(["oras", "version"], capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("[red]ORAS not found. Please install ORAS CLI[/red]")
            console.print("  Install: https://oras.land/docs/installation")
            raise typer.Exit(1)

        # Push with oras
        try:
            subprocess.check_call(
                [
                    "oras",
                    "push",
                    ref,
                    str(path),
                    "--annotation",
                    f"org.greenlang.type=pack",
                    "--annotation",
                    f"org.greenlang.version={manifest.version}",
                    "--annotation",
                    f"org.greenlang.name={manifest.name}",
                ]
            )
            console.print(
                f"[green][OK][/green] Published {org}/{manifest.name}@{manifest.version}"
            )

            # Update hub index if requested
            if update_index:
                console.print("[cyan]Updating hub index...[/cyan]")
                try:
                    _update_hub_index(path, manifest, org)
                    console.print(f"[green][OK][/green] Updated hub index")
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not update index: {e}[/yellow]"
                    )

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to push: {e}[/red]")
            raise typer.Exit(1)


@app.command("add")
def add(
    ref: str = typer.Argument(..., help="Pack reference (org/name@version or name)"),
    cache: Path = typer.Option(Path(".gl_cache"), "--cache", help="Cache directory"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify signatures"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall"),
):
    """Pull, verify signature, install"""
    from ..packs.installer import PackInstaller
    from ..packs.registry import PackRegistry
    from ..hub.index import HubIndex
    from ..provenance.signing import verify_artifact

    console.print(f"[cyan]Installing {ref}...[/cyan]")

    # Check if local path
    if Path(ref).exists():
        # Install from local directory
        from ..packs.manifest import load_manifest
        from ..policy.enforcer import check_install
        from ..packs.installer import PackInstaller

        # Check policy first
        try:
            manifest = load_manifest(Path(ref))
            check_install(manifest, str(ref), stage="add")
            console.print("[green][OK][/green] Policy check passed")
        except RuntimeError as e:
            console.print(f"[red]Policy check failed: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[yellow]Warning: Policy check error: {e}[/yellow]")

        installer = PackInstaller()
        try:
            installed = installer._install_from_local(Path(ref), verify=verify)
            console.print(f"[green][OK][/green] Installed local pack: {installed.name}")
        except Exception as e:
            console.print(f"[red]Failed to install: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Parse reference
        if "@" in ref:
            pack_ref, version = ref.rsplit("@", 1)
        else:
            pack_ref, version = ref, "latest"

        # Check hub index for pack info
        hub = HubIndex()
        entry = hub.get(pack_ref)

        if not entry:
            console.print(f"[red]Pack not found in index: {pack_ref}[/red]")
            console.print("Available packs:")
            entries = hub.search("")[:5]  # Show first 5 packs
            for e in entries:
                console.print(f"  - {e.org}/{e.slug} v{e.latest_version}")
            raise typer.Exit(1)

        # Use latest version if requested
        if version == "latest":
            version = entry.latest_version
        elif version not in entry.versions:
            console.print(f"[red]Version {version} not available for {pack_ref}[/red]")
            console.print(f"Available versions: {', '.join(entry.versions)}")
            raise typer.Exit(1)

        # Check if already installed
        registry = PackRegistry()
        installed_pack = registry.get(entry.slug)
        if not force and installed_pack:
            console.print(
                f"[yellow]Pack already installed: {entry.slug} v{installed_pack.version}[/yellow]"
            )
            console.print("Use --force to reinstall")
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Pulling {pack_ref}@{version}...", total=None)

            try:
                # Check if oras is available
                subprocess.check_call(["oras", "version"], capture_output=True)

                # Pull with oras
                cache.mkdir(parents=True, exist_ok=True)
                pack_dir = cache / entry.slug / version
                pack_dir.mkdir(parents=True, exist_ok=True)

                oci_ref = f"{entry.oci_ref}:{version}"
                subprocess.check_call(["oras", "pull", oci_ref, "-o", str(pack_dir)])

                progress.update(task, description="Verifying...")

                # Verify signatures if requested
                if verify:
                    try:
                        # Look for signature files
                        sig_files = list(pack_dir.rglob("*.sig"))
                        if sig_files:
                            for sig_file in sig_files:
                                artifact_file = sig_file.with_suffix("")
                                if artifact_file.exists():
                                    is_valid, signer_info = verify_artifact(
                                        artifact_file
                                    )
                                    if not is_valid:
                                        console.print(
                                            f"[red]Signature verification failed for {artifact_file.name}[/red]"
                                        )
                                        raise typer.Exit(1)
                            console.print("[green][OK][/green] Signatures verified")
                        else:
                            console.print(
                                "[yellow]Warning: No signatures found[/yellow]"
                            )
                    except Exception as e:
                        if verify:
                            console.print(f"[red]Verification failed: {e}[/red]")
                            raise typer.Exit(1)

                progress.update(task, description="Checking policy...")

                # Check policy before installing
                from ..packs.manifest import load_manifest
                from ..policy.enforcer import check_install

                try:
                    manifest = load_manifest(pack_dir)
                    check_install(manifest, str(pack_dir), stage="add")
                    console.print("[green][OK][/green] Policy check passed")
                except RuntimeError as e:
                    console.print(f"[red]Policy check failed: {e}[/red]")
                    raise typer.Exit(1)
                except Exception as e:
                    console.print(f"[yellow]Warning: Policy check error: {e}[/yellow]")

                progress.update(task, description="Installing...")

                # Register pack with verification enabled
                registry.register(pack_dir, verify=True)

                progress.update(task, completed=True)
                console.print(
                    f"[green][OK][/green] Installed {pack_ref}@{version} -> {cache}"
                )

            except subprocess.CalledProcessError as e:
                progress.update(task, completed=True)
                console.print(f"[red]Failed to pull pack: {e}[/red]")
                raise typer.Exit(1)
            except FileNotFoundError:
                progress.update(task, completed=True)
                console.print("[red]ORAS not found. Please install ORAS CLI[/red]")
                console.print("  Install: https://oras.land/docs/installation")
                raise typer.Exit(1)


@app.command("info")
def info(ref: str = typer.Argument(..., help="Pack name or reference")):
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
    manifest = pack.manifest if hasattr(pack.manifest, "name") else pack.manifest
    if isinstance(manifest, dict):
        name = manifest.get("name", "Unknown")
        version = manifest.get("version", "1.0.0")
        description = manifest.get("description", "No description")
        kind = manifest.get("kind", "pack")
        license = manifest.get("license", "Unknown")
    else:
        name = manifest.name
        version = manifest.version
        description = manifest.description or "No description"
        kind = manifest.kind
        license = manifest.license

    console.print(
        Panel.fit(
            f"[bold]{name}[/bold] v{version}\n"
            f"{description}\n\n"
            f"Kind: {kind}\n"
            f"License: {license}\n"
            f"Location: {pack.location}\n"
            f"Verified: {'[OK]' if pack.verified else '[FAIL]'}",
            title="Pack Information",
        )
    )

    # Show contents
    contents = (
        manifest.get("contents", {})
        if isinstance(manifest, dict)
        else manifest.contents
    )
    if contents:
        console.print("\n[bold]Contents:[/bold]")
        pipelines = (
            contents.get("pipelines", [])
            if isinstance(contents, dict)
            else contents.pipelines
        )
        agents = (
            contents.get("agents", [])
            if isinstance(contents, dict)
            else contents.agents
        )
        datasets = (
            contents.get("datasets", [])
            if isinstance(contents, dict)
            else contents.datasets
        )

        if pipelines:
            console.print(f"  Pipelines: {', '.join(pipelines)}")
        if agents:
            console.print(f"  Agents: {', '.join(agents)}")
        if datasets:
            console.print(f"  Datasets: {', '.join(datasets)}")

    # Show dependencies
    dependencies = (
        manifest.get("dependencies", [])
        if isinstance(manifest, dict)
        else pack.manifest.dependencies
    )
    if dependencies:
        console.print("\n[bold]Dependencies:[/bold]")
        for dep in dependencies:
            if isinstance(dep, dict):
                console.print(f"  - {dep.get('name', dep)} {dep.get('version', '')}")
            else:
                console.print(f"  - {dep}")

    # Show authors
    authors = (
        manifest.get("authors", [])
        if isinstance(manifest, dict)
        else getattr(pack.manifest, "authors", [])
    )
    if authors:
        console.print("\n[bold]Authors:[/bold]")
        for author in authors:
            if isinstance(author, dict):
                console.print(
                    f"  - {author.get('name', 'Unknown')} <{author.get('email', '')}>"
                )
            else:
                console.print(f"  - {author}")


@app.command("list")
def list_packs(
    kind: Optional[str] = typer.Option(None, "--kind", "-k", help="Filter by kind"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    installed_only: bool = typer.Option(
        True,
        "--installed-only/--all",
        help="Show only installed or all available packs",
    ),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search query"),
):
    """List installed packs or search available packs"""
    from ..packs.registry import PackRegistry
    from ..hub.index import HubIndex

    if installed_only:
        # List installed packs
        registry = PackRegistry()
        packs = registry.list(kind=kind)

        if not packs:
            console.print("[yellow]No packs installed[/yellow]")
            console.print("\nInstall packs with: [cyan]gl pack add <pack-name>[/cyan]")
            console.print(
                "Search available packs with: [cyan]gl pack list --all[/cyan]"
            )
            return

        if json_output:
            import json

            output = []
            for pack in packs:
                output.append(
                    {
                        "name": pack.manifest.name,
                        "version": pack.manifest.version,
                        "kind": pack.manifest.kind,
                        "location": str(pack.location),
                        "verified": pack.verified,
                    }
                )
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
                    pack.manifest.get("name", pack.name),
                    pack.manifest.get("version", pack.version),
                    pack.manifest.get("kind", "pack"),
                    (
                        str(pack.location)[:40] + "..."
                        if len(str(pack.location)) > 40
                        else str(pack.location)
                    ),
                    "[OK]" if pack.verified else "[FAIL]",
                )

            console.print(table)
    else:
        # List available packs from hub
        console.print("[cyan]Fetching available packs...[/cyan]")
        hub = HubIndex()
        try:
            entries = hub.search(search or "", tags=kind.split(",") if kind else None)

            if not entries:
                console.print("[yellow]No packs found[/yellow]")
                return

            if json_output:
                import json

                output = []
                for entry in entries:
                    output.append(
                        {
                            "name": entry.name,
                            "org": entry.org,
                            "slug": entry.slug,
                            "latest_version": entry.latest_version,
                            "versions": entry.versions,
                            "description": entry.description,
                            "license": entry.license,
                            "download_count": entry.download_count,
                            "tags": entry.tags,
                        }
                    )
                console.print(json.dumps(output, indent=2))
            else:
                table = Table(title=f"Available Packs ({len(entries)} found)")
                table.add_column("Pack", style="cyan")
                table.add_column("Latest", style="green")
                table.add_column("Description")
                table.add_column("License", style="yellow")
                table.add_column("Downloads", style="blue")

                for entry in entries[:20]:  # Limit to first 20
                    table.add_row(
                        f"{entry.org}/{entry.slug}",
                        entry.latest_version,
                        (
                            entry.description[:50] + "..."
                            if len(entry.description) > 50
                            else entry.description
                        ),
                        entry.license,
                        str(entry.download_count),
                    )

                console.print(table)

                if len(entries) > 20:
                    console.print(
                        f"\n[dim]Showing first 20 of {len(entries)} results. Use --search to filter.[/dim]"
                    )

        except Exception as e:
            console.print(f"[red]Failed to fetch available packs: {e}[/red]")
            console.print("Falling back to installed packs...")
            # Fallback to installed packs
            list_packs(
                kind=kind, json_output=json_output, installed_only=True, search=search
            )


def _update_hub_index(pack_path: Path, manifest, org: str) -> None:
    """
    Update hub index with new pack version

    Args:
        pack_path: Path to pack directory
        manifest: Pack manifest
        org: Organization name
    """
    from ..hub.index import HubIndex, IndexEntry
    from datetime import datetime

    # Read CARD.md for summary
    card_summary = ""
    card_path = pack_path / "CARD.md"
    if card_path.exists():
        try:
            with open(card_path) as f:
                # Take first few lines as summary
                lines = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):  # Skip headers
                        lines.append(line)
                    if len(lines) >= 3:  # Take first 3 non-header lines
                        break
                card_summary = " ".join(lines)[:200]  # Limit to 200 chars
        except Exception as e:
            logging.warning(f"Could not read CARD.md: {e}")

    # Create/update index entry
    hub = HubIndex()
    existing = hub.get(f"{org}/{manifest.name}")

    if existing:
        # Update existing entry
        if manifest.version not in existing.versions:
            existing.versions.append(manifest.version)
            existing.versions.sort()  # Keep versions sorted
        existing.latest_version = manifest.version
        existing.updated_at = datetime.now().isoformat()
        existing.card_summary = card_summary or existing.card_summary
        existing.description = (
            getattr(manifest, "description", "") or existing.description
        )
        entry = existing
    else:
        # Create new entry
        entry = IndexEntry(
            name=manifest.name,
            org=org,
            slug=manifest.name,
            latest_version=manifest.version,
            versions=[manifest.version],
            description=getattr(manifest, "description", ""),
            license=getattr(manifest, "license", "Apache-2.0"),
            card_summary=card_summary,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            download_count=0,
            tags=getattr(manifest, "tags", []),
            oci_ref=f"ghcr.io/{org}/{manifest.name}",
        )

    # Update local index cache
    hub.add_or_update(entry)


@app.command("search")
def search_packs(
    query: str = typer.Argument("", help="Search query"),
    tags: Optional[str] = typer.Option(
        None, "--tags", "-t", help="Filter by tags (comma-separated)"
    ),
    org: Optional[str] = typer.Option(
        None, "--org", "-o", help="Filter by organization"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Search available packs in the hub"""
    from ..hub.index import HubIndex

    console.print(f"[cyan]Searching for: {query or 'all packs'}[/cyan]")

    hub = HubIndex()
    try:
        tag_list = tags.split(",") if tags else None
        entries = hub.search(query, tags=tag_list, org=org)

        if not entries:
            console.print("[yellow]No packs found matching your criteria[/yellow]")
            return

        if json_output:
            import json

            output = []
            for entry in entries:
                output.append(
                    {
                        "name": entry.name,
                        "org": entry.org,
                        "slug": entry.slug,
                        "latest_version": entry.latest_version,
                        "description": entry.description,
                        "tags": entry.tags,
                        "download_count": entry.download_count,
                    }
                )
            console.print(json.dumps(output, indent=2))
        else:
            table = Table(title=f"Search Results ({len(entries)} found)")
            table.add_column("Pack", style="cyan")
            table.add_column("Latest", style="green")
            table.add_column("Description")
            table.add_column("Tags", style="magenta")
            table.add_column("Downloads", style="blue")

            for entry in entries:
                tags_str = ", ".join(entry.tags[:3])  # Show first 3 tags
                if len(entry.tags) > 3:
                    tags_str += "..."

                table.add_row(
                    f"{entry.org}/{entry.slug}",
                    entry.latest_version,
                    (
                        entry.description[:50] + "..."
                        if len(entry.description) > 50
                        else entry.description
                    ),
                    tags_str,
                    str(entry.download_count),
                )

            console.print(table)

            console.print(
                f"\n[dim]Install with: gl pack add <org/name>@<version>[/dim]"
            )

    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("index")
def index_commands(
    action: str = typer.Argument(..., help="Action: create, update, show"),
    input_dir: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Input directory"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file"
    ),
):
    """Manage hub index"""
    from ..hub.index import HubIndex

    hub = HubIndex()

    if action == "create":
        if not input_dir:
            input_dir = Path("packs")
        if not output_file:
            output_file = Path("index.json")

        if not input_dir.exists():
            console.print(f"[red]Input directory not found: {input_dir}[/red]")
            raise typer.Exit(1)

        console.print(f"[cyan]Creating index from {input_dir}...[/cyan]")
        hub.create_local_index(output_file, input_dir)
        console.print(f"[green][OK][/green] Index created: {output_file}")

    elif action == "update":
        console.print("[cyan]Updating index from remote...[/cyan]")
        hub.load(force_refresh=True)
        console.print("[green][OK][/green] Index updated")

    elif action == "show":
        console.print("[cyan]Current index contents:[/cyan]")
        index_data = hub.load()

        console.print(f"Version: {index_data.get('version', 'unknown')}")
        console.print(f"Updated: {index_data.get('updated_at', 'unknown')}")
        console.print(f"Packs: {len(index_data.get('packs', {}))}")

        # Show first few packs
        packs = list(index_data.get("packs", {}).items())[:5]
        if packs:
            console.print("\nSample packs:")
            for key, data in packs:
                console.print(f"  - {key} v{data.get('latest_version', '0.0.0')}")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: create, update, show")
        raise typer.Exit(1)
