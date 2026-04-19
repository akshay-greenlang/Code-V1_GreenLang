# -*- coding: utf-8 -*-
"""
SBOM generation and verification commands for GreenLang CLI.

Provides local SBOM generation capabilities for developers.
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Optional
import json

app = typer.Typer(help="SBOM generation and verification commands")
console = Console()


@app.command("generate")
def generate_sbom(
    pack_path: Path = typer.Argument(..., help="Path to pack directory"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output SBOM file path"),
    format: str = typer.Option("cyclonedx", "-f", "--format", help="SBOM format: cyclonedx or spdx"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output")
):
    """
    Generate SBOM for a pack locally.

    Examples:
        gl sbom generate packs/my-pack
        gl sbom generate packs/my-pack -o sbom.json -f cyclonedx
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    if not pack_path.exists():
        console.print(f"[red][ERROR][/red] Pack not found: {pack_path}")
        raise typer.Exit(1)

    if not pack_path.is_dir():
        console.print(f"[red][ERROR][/red] Path is not a directory: {pack_path}")
        raise typer.Exit(1)

    # Default output path
    if output is None:
        output = pack_path / f"sbom.{format}.json"

    console.print(f"[bold]Generating SBOM for: {pack_path}[/bold]")
    console.print(f"Format: {format}")
    console.print(f"Output: {output}\n")

    try:
        from greenlang.provenance import sbom

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating SBOM...", total=None)

            if format == "cyclonedx":
                sbom_data = sbom._generate_manual_sbom(str(pack_path))
            elif format == "spdx":
                sbom_data = sbom._generate_spdx_sbom(str(pack_path))
            else:
                console.print(f"[red][ERROR][/red] Unknown format: {format}")
                raise typer.Exit(1)

            # Write SBOM to file
            with open(output, 'w') as f:
                json.dump(sbom_data, f, indent=2)

            progress.update(task, completed=True)

        console.print(f"\n[green][OK][/green] SBOM generated successfully")
        console.print(f"[blue][INFO][/blue] Saved to: {output}")

        if verbose:
            # Show SBOM stats
            if format == "cyclonedx":
                components_count = len(sbom_data.get("components", []))
                console.print(f"\n[bold]SBOM Statistics:[/bold]")
                console.print(f"  Components: {components_count}")
                console.print(f"  Spec Version: {sbom_data.get('specVersion', 'unknown')}")
            elif format == "spdx":
                packages_count = len(sbom_data.get("packages", []))
                console.print(f"\n[bold]SBOM Statistics:[/bold]")
                console.print(f"  Packages: {packages_count}")
                console.print(f"  SPDX Version: {sbom_data.get('spdxVersion', 'unknown')}")

    except ImportError:
        console.print("[red][ERROR][/red] SBOM module not available")
        console.print("[blue][INFO][/blue] Install with: pip install greenlang-cli[sbom]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red][ERROR][/red] Failed to generate SBOM: {str(e)}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command("verify")
def verify_sbom(
    sbom_file: Path = typer.Argument(..., help="Path to SBOM file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output")
):
    """
    Verify SBOM integrity and validity.

    Examples:
        gl sbom verify sbom.json
        gl sbom verify sbom.cyclonedx.json -v
    """
    if not sbom_file.exists():
        console.print(f"[red][ERROR][/red] SBOM file not found: {sbom_file}")
        raise typer.Exit(1)

    console.print(f"[bold]Verifying SBOM: {sbom_file}[/bold]\n")

    try:
        with open(sbom_file, 'r') as f:
            sbom_data = json.load(f)

        # Detect SBOM format
        if "bomFormat" in sbom_data:
            format_type = "CycloneDX"
            spec_version = sbom_data.get("specVersion", "unknown")
        elif "spdxVersion" in sbom_data:
            format_type = "SPDX"
            spec_version = sbom_data.get("spdxVersion", "unknown")
        else:
            console.print("[red][ERROR][/red] Unknown SBOM format")
            raise typer.Exit(1)

        console.print(f"[green][OK][/green] SBOM format: {format_type}")
        console.print(f"[green][OK][/green] Spec version: {spec_version}")

        # Validate structure
        checks = []

        if format_type == "CycloneDX":
            checks.append(("serialNumber" in sbom_data, "Serial number present"))
            checks.append(("components" in sbom_data, "Components defined"))
            checks.append((len(sbom_data.get("components", [])) > 0, "Components not empty"))
            checks.append(("metadata" in sbom_data, "Metadata present"))
        elif format_type == "SPDX":
            checks.append(("SPDXID" in sbom_data, "SPDX ID present"))
            checks.append(("packages" in sbom_data, "Packages defined"))
            checks.append((len(sbom_data.get("packages", [])) > 0, "Packages not empty"))
            checks.append(("creationInfo" in sbom_data, "Creation info present"))

        # Display results
        console.print("\n[bold]Validation Results:[/bold]")
        table = Table(show_header=False)
        table.add_column("Status", width=6)
        table.add_column("Check")

        all_passed = True
        for passed, check_name in checks:
            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            table.add_row(status, check_name)
            if not passed:
                all_passed = False

        console.print(table)

        if all_passed:
            console.print("\n[green][OK][/green] SBOM is valid")
        else:
            console.print("\n[red][ERROR][/red] SBOM validation failed")
            raise typer.Exit(1)

        # Show details if verbose
        if verbose:
            console.print("\n[bold]SBOM Details:[/bold]")

            if format_type == "CycloneDX":
                console.print(f"  Serial Number: {sbom_data.get('serialNumber', 'N/A')}")
                console.print(f"  Version: {sbom_data.get('version', 'N/A')}")
                console.print(f"  Components: {len(sbom_data.get('components', []))}")

                if sbom_data.get("metadata"):
                    metadata = sbom_data["metadata"]
                    console.print(f"  Component Name: {metadata.get('component', {}).get('name', 'N/A')}")
                    console.print(f"  Component Version: {metadata.get('component', {}).get('version', 'N/A')}")

            elif format_type == "SPDX":
                console.print(f"  SPDX ID: {sbom_data.get('SPDXID', 'N/A')}")
                console.print(f"  Document Name: {sbom_data.get('name', 'N/A')}")
                console.print(f"  Packages: {len(sbom_data.get('packages', []))}")

                if sbom_data.get("creationInfo"):
                    creation_info = sbom_data["creationInfo"]
                    console.print(f"  Created: {creation_info.get('created', 'N/A')}")
                    console.print(f"  Creators: {', '.join(creation_info.get('creators', []))}")

    except json.JSONDecodeError:
        console.print("[red][ERROR][/red] Invalid JSON in SBOM file")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red][ERROR][/red] Verification failed: {str(e)}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command("list")
def list_sboms(
    directory: Path = typer.Argument(Path.cwd(), help="Directory to search for SBOMs"),
    recursive: bool = typer.Option(False, "-r", "--recursive", help="Search recursively")
):
    """
    List all SBOM files in a directory.

    Examples:
        gl sbom list
        gl sbom list packs/ -r
    """
    if not directory.exists():
        console.print(f"[red][ERROR][/red] Directory not found: {directory}")
        raise typer.Exit(1)

    console.print(f"[bold]Searching for SBOMs in: {directory}[/bold]\n")

    # Search for SBOM files
    patterns = ["*.sbom.json", "sbom.*.json", "*cyclonedx*.json", "*spdx*.json"]
    sbom_files = []

    for pattern in patterns:
        if recursive:
            sbom_files.extend(directory.rglob(pattern))
        else:
            sbom_files.extend(directory.glob(pattern))

    # Remove duplicates
    sbom_files = list(set(sbom_files))

    if not sbom_files:
        console.print("[yellow][WARN][/yellow] No SBOM files found")
        return

    # Display results
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Format", style="green")
    table.add_column("Size", justify="right", style="yellow")

    for sbom_file in sorted(sbom_files):
        try:
            with open(sbom_file, 'r') as f:
                sbom_data = json.load(f)

            if "bomFormat" in sbom_data:
                format_type = "CycloneDX"
            elif "spdxVersion" in sbom_data:
                format_type = "SPDX"
            else:
                format_type = "Unknown"

            file_size = sbom_file.stat().st_size
            size_str = f"{file_size:,} bytes"

            table.add_row(
                str(sbom_file.relative_to(directory)),
                format_type,
                size_str
            )

        except Exception:
            # Skip invalid files
            pass

    console.print(table)
    console.print(f"\n[blue][INFO][/blue] Found {len(sbom_files)} SBOM file(s)")


@app.command("diff")
def diff_sboms(
    sbom1: Path = typer.Argument(..., help="First SBOM file"),
    sbom2: Path = typer.Argument(..., help="Second SBOM file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output")
):
    """
    Compare two SBOM files and show differences.

    Examples:
        gl sbom diff sbom-v1.json sbom-v2.json
        gl sbom diff old.json new.json -v
    """
    if not sbom1.exists():
        console.print(f"[red][ERROR][/red] SBOM file not found: {sbom1}")
        raise typer.Exit(1)

    if not sbom2.exists():
        console.print(f"[red][ERROR][/red] SBOM file not found: {sbom2}")
        raise typer.Exit(1)

    console.print(f"[bold]Comparing SBOMs:[/bold]")
    console.print(f"  File 1: {sbom1}")
    console.print(f"  File 2: {sbom2}\n")

    try:
        with open(sbom1, 'r') as f:
            sbom_data1 = json.load(f)

        with open(sbom2, 'r') as f:
            sbom_data2 = json.load(f)

        # Extract components/packages
        if "components" in sbom_data1:
            components1 = {c.get("name", ""): c for c in sbom_data1.get("components", [])}
            components2 = {c.get("name", ""): c for c in sbom_data2.get("components", [])}
        elif "packages" in sbom_data1:
            components1 = {p.get("name", ""): p for p in sbom_data1.get("packages", [])}
            components2 = {p.get("name", ""): p for p in sbom_data2.get("packages", [])}
        else:
            console.print("[red][ERROR][/red] Unable to extract components from SBOMs")
            raise typer.Exit(1)

        # Find differences
        only_in_1 = set(components1.keys()) - set(components2.keys())
        only_in_2 = set(components2.keys()) - set(components1.keys())
        common = set(components1.keys()) & set(components2.keys())

        # Display results
        console.print(f"[bold]Comparison Results:[/bold]")
        console.print(f"  Only in {sbom1.name}: {len(only_in_1)}")
        console.print(f"  Only in {sbom2.name}: {len(only_in_2)}")
        console.print(f"  Common: {len(common)}\n")

        if verbose:
            if only_in_1:
                console.print(f"[red]Components only in {sbom1.name}:[/red]")
                for comp in sorted(only_in_1):
                    console.print(f"  - {comp}")

            if only_in_2:
                console.print(f"[green]Components only in {sbom2.name}:[/green]")
                for comp in sorted(only_in_2):
                    console.print(f"  + {comp}")

        if len(only_in_1) == 0 and len(only_in_2) == 0:
            console.print("[green][OK][/green] SBOMs are identical")
        else:
            console.print("[yellow][WARN][/yellow] SBOMs have differences")

    except json.JSONDecodeError as e:
        console.print(f"[red][ERROR][/red] Invalid JSON: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red][ERROR][/red] Comparison failed: {str(e)}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
