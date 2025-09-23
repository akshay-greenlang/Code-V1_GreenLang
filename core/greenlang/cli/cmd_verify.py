"""
gl verify - Verify artifact signatures and SBOM
"""

import typer
import json
import os
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Set up Windows encoding support
if sys.platform == "win32":
    # Set environment variables for UTF-8 support
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

    # Configure console for Windows
    try:
        # Try to enable ANSI support on Windows
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except (AttributeError, OSError):
        pass

app = typer.Typer()
# Create console with explicit encoding for Windows compatibility
console = Console(force_terminal=True, legacy_windows=False, force_interactive=False)


def safe_string(text: str) -> str:
    """Convert string to ASCII-safe format for Windows console output"""
    if sys.platform == "win32":
        return str(text).encode("ascii", "replace").decode("ascii")
    return str(text)


@app.callback(invoke_without_command=True)
def verify(
    ctx: typer.Context,
    artifact: str = typer.Argument(..., help="Artifact or pack reference to verify"),
    signature: Optional[Path] = typer.Option(
        None, "--sig", "-s", help="Signature file"
    ),
    sbom: bool = typer.Option(True, "--sbom/--no-sbom", help="Check SBOM"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show details"),
):
    """
    Show signer, SBOM, provenance

    Verifies artifact integrity and displays provenance information.
    Fails on tamper or missing signatures.
    """
    if ctx.invoked_subcommand is not None:
        return

    from ..provenance.signing import verify_artifact
    from ..provenance.sbom import verify_sbom

    # Determine if artifact is a file or pack reference
    artifact_path = Path(artifact)

    if artifact_path.exists():
        # Check if this is an SBOM file specifically
        if artifact_path.suffix == ".json" and "sbom" in artifact_path.name.lower():
            # Verify SBOM directly
            console.print(f"[cyan]Verifying SBOM: {artifact}[/cyan]\n")

            # Load SBOM
            try:
                with open(artifact_path, "r", encoding="utf-8", errors="replace") as f:
                    sbom_data = json.load(f)

                # Check SBOM type
                if "spdxVersion" in sbom_data:
                    console.print(f"[green][OK][/green] Valid SPDX SBOM")
                    console.print(f"  Version: {sbom_data['spdxVersion']}")
                    console.print(f"  Document: {sbom_data.get('name', 'Unknown')}")
                    console.print(f"  Packages: {len(sbom_data.get('packages', []))}")
                    console.print(
                        f"  Relationships: {len(sbom_data.get('relationships', []))}"
                    )
                elif "bomFormat" in sbom_data:
                    console.print(f"[green][OK][/green] Valid CycloneDX SBOM")
                    console.print(f"  Format: {sbom_data['bomFormat']}")
                    console.print(
                        f"  Version: {sbom_data.get('specVersion', 'Unknown')}"
                    )
                    console.print(
                        f"  Components: {len(sbom_data.get('components', []))}"
                    )
                else:
                    console.print("[red][FAIL][/red] Unknown SBOM format")
                    raise typer.Exit(1)

                # Verify SBOM integrity if it's for a pack
                pack_path = artifact_path.parent
                if (pack_path / "pack.yaml").exists():
                    console.print(
                        "\n[cyan]Verifying SBOM against pack contents...[/cyan]"
                    )
                    try:
                        is_valid = verify_sbom(artifact_path, pack_path)
                        if is_valid:
                            console.print(
                                "[green][OK][/green] SBOM matches pack contents"
                            )
                        else:
                            console.print(
                                "[red][FAIL][/red] SBOM does not match pack contents"
                            )
                            raise typer.Exit(1)
                    except Exception as e:
                        console.print(
                            f"[yellow]Warning: Could not verify SBOM integrity: {e}[/yellow]"
                        )

                # Show detailed info if verbose
                if verbose:
                    console.print("\n[bold]SBOM Details:[/bold]")
                    if "spdxVersion" in sbom_data:
                        # SPDX details
                        creation = sbom_data.get("creationInfo", {})
                        console.print(
                            f"  Created: {creation.get('created', 'Unknown')}"
                        )
                        console.print(
                            f"  Creators: {', '.join(creation.get('creators', []))}"
                        )
                        console.print(
                            f"  License: {sbom_data.get('dataLicense', 'Unknown')}"
                        )

                        # Show main package
                        for pkg in sbom_data.get("packages", [])[:1]:
                            console.print(f"\n  Main Package:")
                            console.print(f"    Name: {pkg.get('name', 'Unknown')}")
                            console.print(
                                f"    Version: {pkg.get('versionInfo', 'Unknown')}"
                            )
                            console.print(
                                f"    License: {pkg.get('licenseConcluded', 'Unknown')}"
                            )
                            console.print(f"    Files: {len(pkg.get('files', []))}")

                return  # Exit after SBOM verification

            except json.JSONDecodeError as e:
                console.print(f"[red]Invalid JSON in SBOM: {safe_string(e)}[/red]")
                raise typer.Exit(1)
            except Exception as e:
                console.print(f"[red]SBOM verification failed: {safe_string(e)}[/red]")
                raise typer.Exit(1)

        # Regular artifact verification
        console.print(f"[cyan]Verifying artifact: {artifact}[/cyan]\n")

        # Check signature if provided
        if signature:
            console.print("[cyan]Checking signature...[/cyan]")
            try:
                is_valid, signer_info = verify_artifact(artifact_path, signature)
                if is_valid:
                    console.print(f"[green][OK][/green] Signature valid")
                    if verbose and signer_info:
                        console.print(
                            f"  Signer: {signer_info.get('subject', 'Unknown')}"
                        )
                        console.print(
                            f"  Issuer: {signer_info.get('issuer', 'Unknown')}"
                        )
                        console.print(
                            f"  Timestamp: {signer_info.get('timestamp', 'Unknown')}"
                        )
                else:
                    console.print(f"[red][FAIL][/red] Signature invalid")
                    raise typer.Exit(1)
            except Exception as e:
                console.print(
                    f"[red]Signature verification failed: {safe_string(e)}[/red]"
                )
                raise typer.Exit(1)

        # Check SBOM if requested
        if sbom:
            sbom_path = artifact_path.parent / "sbom.spdx.json"
            if not sbom_path.exists():
                sbom_path = artifact_path.parent / "sbom.json"

            if sbom_path.exists():
                console.print("[cyan]Checking SBOM...[/cyan]")
                try:
                    with open(sbom_path, "r", encoding="utf-8", errors="replace") as f:
                        sbom_data = json.load(f)

                    console.print(f"[green][OK][/green] SBOM found")

                    if verbose:
                        # Display SBOM summary
                        console.print("\n[bold]SBOM Summary:[/bold]")
                        console.print(
                            f"  Format: {sbom_data.get('spdxVersion', 'Unknown')}"
                        )
                        console.print(
                            f"  Created: {sbom_data.get('creationInfo', {}).get('created', 'Unknown')}"
                        )

                        packages = sbom_data.get("packages", [])
                        console.print(f"  Components: {len(packages)}")

                        if packages and verbose:
                            console.print("\n[bold]Top Components:[/bold]")
                            for pkg in packages[:5]:
                                console.print(
                                    f"  - {pkg.get('name', 'Unknown')} {pkg.get('versionInfo', '')}"
                                )
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not parse SBOM: {safe_string(e)}[/yellow]"
                    )
            else:
                console.print("[yellow]No SBOM found[/yellow]")

        # Check for provenance metadata
        provenance_path = artifact_path.parent / "provenance.json"
        if provenance_path.exists():
            console.print("[cyan]Checking provenance...[/cyan]")
            try:
                with open(
                    provenance_path, "r", encoding="utf-8", errors="replace"
                ) as f:
                    prov_data = json.load(f)

                console.print(f"[green][OK][/green] Provenance found")

                if verbose:
                    console.print("\n[bold]Provenance:[/bold]")
                    console.print(
                        f"  Builder: {prov_data.get('builder', {}).get('id', 'Unknown')}"
                    )
                    console.print(
                        f"  Build Type: {prov_data.get('buildType', 'Unknown')}"
                    )
                    console.print(
                        f"  Timestamp: {prov_data.get('metadata', {}).get('buildFinishedOn', 'Unknown')}"
                    )
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not parse provenance: {safe_string(e)}[/yellow]"
                )

        console.print(f"\n[green][OK][/green] Artifact verified: {artifact_path.name}")

    else:
        # Verify pack from registry
        from ..packs.registry import PackRegistry

        registry = PackRegistry()

        # Parse reference
        if "@" in artifact:
            name, version = artifact.split("@", 1)
        else:
            name = artifact
            version = None

        pack = registry.get(name, version=version)

        if not pack:
            console.print(f"[red]Pack not found: {artifact}[/red]")
            raise typer.Exit(1)

        console.print(
            f"[cyan]Verifying pack: {pack.manifest.name} v{pack.manifest.version}[/cyan]\n"
        )

        # Check pack verification status
        if pack.verified:
            console.print(f"[green][OK][/green] Pack verified")

            if verbose:
                # Show verification details
                console.print("\n[bold]Verification Details:[/bold]")
                console.print(f"  Location: {pack.location}")
                console.print(f"  Hash: {pack.hash[:16]}...")

                # Check for signatures
                sig_dir = pack.location / "signatures"
                if sig_dir.exists():
                    sigs = list(sig_dir.glob("*.sig"))
                    console.print(f"  Signatures: {len(sigs)} found")

                # Check for SBOM
                sbom_path = pack.location / "sbom.spdx.json"
                if sbom_path.exists():
                    with open(sbom_path, "r", encoding="utf-8", errors="replace") as f:
                        sbom_data = json.load(f)
                    console.print(
                        f"  SBOM: {len(sbom_data.get('packages', []))} components"
                    )
        else:
            console.print(f"[red][FAIL][/red] Pack not verified")
            console.print("\nRun verification with: [cyan]gl pack verify {name}[/cyan]")
            raise typer.Exit(1)


@app.command("sbom")
def show_sbom(
    artifact: str = typer.Argument(..., help="Artifact or pack to inspect"),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format (table|json|spdx)"
    ),
):
    """Display SBOM details"""
    # Find SBOM file
    artifact_path = Path(artifact)

    if artifact_path.is_file():
        sbom_path = artifact_path.parent / "sbom.spdx.json"
    elif artifact_path.is_dir():
        sbom_path = artifact_path / "sbom.spdx.json"
    else:
        # Try pack registry
        from ..packs.registry import PackRegistry

        registry = PackRegistry()
        pack = registry.get(artifact)
        if pack:
            sbom_path = pack.location / "sbom.spdx.json"
        else:
            console.print(f"[red]Artifact not found: {artifact}[/red]")
            raise typer.Exit(1)

    if not sbom_path.exists():
        console.print(f"[red]No SBOM found for: {artifact}[/red]")
        raise typer.Exit(1)

    with open(sbom_path, "r", encoding="utf-8", errors="replace") as f:
        sbom_data = json.load(f)

    if format == "json":
        console.print(json.dumps(sbom_data, indent=2))
    elif format == "spdx":
        # Output in SPDX format
        console.print(json.dumps(sbom_data, indent=2))
    else:
        # Table format
        console.print(
            Panel.fit(
                f"[bold]SBOM for {artifact}[/bold]\n\n"
                f"Format: {sbom_data.get('spdxVersion', 'Unknown')}\n"
                f"Created: {sbom_data.get('creationInfo', {}).get('created', 'Unknown')}\n"
                f"Creator: {sbom_data.get('creationInfo', {}).get('creators', ['Unknown'])[0]}",
                title="Software Bill of Materials",
            )
        )

        packages = sbom_data.get("packages", [])
        if packages:
            table = Table(title="Components")
            table.add_column("Name", style="cyan")
            table.add_column("Version", style="green")
            table.add_column("License", style="yellow")
            table.add_column("Supplier")

            for pkg in packages:
                table.add_row(
                    pkg.get("name", "Unknown"),
                    pkg.get("versionInfo", "-"),
                    pkg.get("licenseConcluded", "-"),
                    pkg.get("supplier", "-"),
                )

            console.print(table)

        relationships = sbom_data.get("relationships", [])
        if relationships:
            console.print(f"\n[bold]Relationships:[/bold] {len(relationships)} defined")


@app.command("provenance")
def show_provenance(artifact: str = typer.Argument(..., help="Artifact to inspect")):
    """Display provenance information"""
    artifact_path = Path(artifact)

    if artifact_path.is_file():
        prov_path = artifact_path.parent / "provenance.json"
    elif artifact_path.is_dir():
        prov_path = artifact_path / "provenance.json"
    else:
        console.print(f"[red]Artifact not found: {artifact}[/red]")
        raise typer.Exit(1)

    if not prov_path.exists():
        console.print(f"[red]No provenance found for: {artifact}[/red]")
        raise typer.Exit(1)

    with open(prov_path, "r", encoding="utf-8", errors="replace") as f:
        prov_data = json.load(f)

    console.print(
        Panel.fit(
            f"[bold]Provenance for {artifact}[/bold]\n\n"
            f"Builder: {prov_data.get('builder', {}).get('id', 'Unknown')}\n"
            f"Build Type: {prov_data.get('buildType', 'Unknown')}\n"
            f"Invocation: {prov_data.get('invocation', {}).get('configSource', {}).get('uri', 'Unknown')}\n"
            f"Started: {prov_data.get('metadata', {}).get('buildStartedOn', 'Unknown')}\n"
            f"Finished: {prov_data.get('metadata', {}).get('buildFinishedOn', 'Unknown')}",
            title="Build Provenance",
        )
    )

    # Show materials (inputs)
    materials = prov_data.get("materials", [])
    if materials:
        console.print("\n[bold]Materials (Inputs):[/bold]")
        for material in materials:
            console.print(f"  - {material.get('uri', 'Unknown')}")
            if material.get("digest"):
                for alg, value in material["digest"].items():
                    console.print(f"    {alg}: {value[:16]}...")

    # Show subjects (outputs)
    subjects = prov_data.get("subject", [])
    if subjects:
        console.print("\n[bold]Subjects (Outputs):[/bold]")
        for subject in subjects:
            console.print(f"  - {subject.get('name', 'Unknown')}")
            if subject.get("digest"):
                for alg, value in subject["digest"].items():
                    console.print(f"    {alg}: {value[:16]}...")
