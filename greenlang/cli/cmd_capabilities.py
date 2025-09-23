"""
Capability Management CLI Commands

Commands for managing and inspecting pack capabilities.
"""

import click
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from greenlang.packs.manifest import PackManifest
from greenlang.packs.installer import PackInstaller

console = Console()


@click.group()
def capabilities():
    """Manage and inspect pack capabilities"""


@capabilities.command()
@click.argument("pack_path", type=click.Path(exists=True))
def lint(pack_path: str) -> None:
    """Lint and validate pack capabilities"""
    pack_path = Path(pack_path)

    # Find manifest file
    manifest_path = pack_path / "pack.yaml"
    if not manifest_path.exists():
        manifest_path = pack_path / "manifest.yaml"
        if not manifest_path.exists():
            console.print("[red]Error: No pack.yaml or manifest.yaml found[/red]")
            return

    installer = PackInstaller()
    report = installer.lint_capabilities(manifest_path)
    console.print(report)


@capabilities.command()
@click.argument("pack_path", type=click.Path(exists=True))
def show(pack_path: str) -> None:
    """Show capabilities requested by a pack"""
    pack_path = Path(pack_path)

    # Find manifest file
    manifest_path = pack_path / "pack.yaml"
    if not manifest_path.exists():
        manifest_path = pack_path / "manifest.yaml"
        if not manifest_path.exists():
            console.print("[red]Error: No pack.yaml or manifest.yaml found[/red]")
            return

    try:
        manifest = PackManifest.from_file(manifest_path)
    except Exception as e:
        console.print(f"[red]Error loading manifest: {e}[/red]")
        return

    console.print(
        Panel.fit(
            f"[bold]Pack: {manifest.name} v{manifest.version}[/bold]", style="cyan"
        )
    )

    if not manifest.capabilities:
        console.print("\n✓ No capabilities requested (deny-all by default)")
        return

    caps = manifest.capabilities

    # Create capabilities table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Capability", style="cyan", width=15)
    table.add_column("Status", style="green", width=10)
    table.add_column("Configuration", style="white")

    # Network
    net_status = "✓ Enabled" if caps.net and caps.net.allow else "✗ Disabled"
    net_config = ""
    if caps.net and caps.net.allow and caps.net.outbound:
        domains = caps.net.outbound.get("allowlist", [])
        net_config = f"Domains: {', '.join(domains[:3])}"
        if len(domains) > 3:
            net_config += f" (+{len(domains)-3} more)"
    table.add_row("Network", net_status, net_config)

    # Filesystem
    fs_status = "✓ Enabled" if caps.fs and caps.fs.allow else "✗ Disabled"
    fs_config = ""
    if caps.fs and caps.fs.allow:
        read_paths = caps.fs.read.get("allowlist", []) if caps.fs.read else []
        write_paths = caps.fs.write.get("allowlist", []) if caps.fs.write else []
        fs_config = f"Read: {len(read_paths)} paths, Write: {len(write_paths)} paths"
    table.add_row("Filesystem", fs_status, fs_config)

    # Clock
    clock_status = "✓ Real-time" if caps.clock and caps.clock.allow else "✗ Frozen"
    clock_config = (
        "Access to system clock"
        if caps.clock and caps.clock.allow
        else "Deterministic time"
    )
    table.add_row("Clock", clock_status, clock_config)

    # Subprocess
    sub_status = (
        "✓ Enabled" if caps.subprocess and caps.subprocess.allow else "✗ Disabled"
    )
    sub_config = ""
    if caps.subprocess and caps.subprocess.allow:
        binaries = caps.subprocess.allowlist
        sub_config = f"Binaries: {len(binaries)}"
    table.add_row("Subprocess", sub_status, sub_config)

    console.print(table)

    # Show detailed configuration if verbose
    if caps.net and caps.net.allow and caps.net.outbound:
        console.print("\n[bold]Network Allowlist:[/bold]")
        for domain in caps.net.outbound.get("allowlist", []):
            console.print(f"  - {domain}")

    if caps.fs and caps.fs.allow:
        if caps.fs.read:
            console.print("\n[bold]Filesystem Read Paths:[/bold]")
            for path in caps.fs.read.get("allowlist", []):
                console.print(f"  - {path}")
        if caps.fs.write:
            console.print("\n[bold]Filesystem Write Paths:[/bold]")
            for path in caps.fs.write.get("allowlist", []):
                console.print(f"  - {path}")

    if caps.subprocess and caps.subprocess.allow:
        console.print("\n[bold]Allowed Binaries:[/bold]")
        for binary in caps.subprocess.allowlist:
            console.print(f"  - {binary}")


@capabilities.command()
@click.argument("pack_path", type=click.Path(exists=True))
def validate(pack_path: str) -> None:
    """Validate pack capabilities against security policies"""
    pack_path = Path(pack_path)

    # Find manifest file
    manifest_path = pack_path / "pack.yaml"
    if not manifest_path.exists():
        manifest_path = pack_path / "manifest.yaml"
        if not manifest_path.exists():
            console.print("[red]Error: No pack.yaml or manifest.yaml found[/red]")
            return

    installer = PackInstaller()
    is_valid, issues = installer.validate_manifest(manifest_path)

    console.print(Panel.fit("[bold]Capability Validation Report[/bold]", style="cyan"))

    if not issues:
        console.print("\n✅ All capability declarations are valid")
    else:
        # Separate errors, warnings, and info
        errors = [
            i
            for i in issues
            if not i.startswith("Warning:") and not i.startswith("Info:")
        ]
        warnings = [i for i in issues if i.startswith("Warning:")]
        infos = [i for i in issues if i.startswith("Info:")]

        if errors:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in errors:
                console.print(f"  ❌ {error}")

        if warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in warnings:
                console.print(f"  ⚠️  {warning}")

        if infos:
            console.print("\n[bold blue]Information:[/bold blue]")
            for info in infos:
                console.print(f"  ℹ️  {info}")

    if is_valid:
        console.print(
            "\n[bold green]✅ Pack is valid and can be installed[/bold green]"
        )
    else:
        console.print(
            "\n[bold red]❌ Pack has errors and cannot be installed[/bold red]"
        )


@capabilities.command()
def template() -> None:
    """Show example capability declarations"""

    example = """# Example capability declarations in pack.yaml

name: my-secure-pack
version: 1.0.0
license: MIT
contents:
  pipelines:
    - pipeline.yaml

# Security capabilities (all default to false/deny)
capabilities:
  # Network access
  net:
    allow: true
    outbound:
      allowlist:
        - https://api.company.com/*
        - https://*.climatenza.com/*

  # Filesystem access
  fs:
    allow: true
    read:
      allowlist:
        - ${INPUT_DIR}/**      # User input files
        - ${PACK_DATA_DIR}/**  # Pack's bundled data
    write:
      allowlist:
        - ${RUN_TMP}/**        # Temporary workspace

  # Real-time clock access
  clock:
    allow: false  # Use frozen time for determinism

  # Subprocess execution
  subprocess:
    allow: true
    allowlist:
      - /usr/bin/exiftool
      - /usr/local/bin/ffmpeg
"""

    syntax = Syntax(example, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Capability Template", style="cyan"))

    console.print("\n[bold]Key Principles:[/bold]")
    console.print("1. All capabilities default to [red]deny[/red] when not specified")
    console.print("2. Be as restrictive as possible - only request what you need")
    console.print(
        "3. Use environment variables for paths (${INPUT_DIR}, ${PACK_DATA_DIR}, ${RUN_TMP})"
    )
    console.print("4. Specify absolute paths for binaries")
    console.print("5. Use glob patterns for domains and paths")


@capabilities.command()
def policy() -> None:
    """Show organization capability policy"""

    config_file = Path.home() / ".greenlang" / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            return
    else:
        config = {}

    capability_policy = config.get("capability_policy", {})

    if not capability_policy:
        console.print("[yellow]No organization capability policy configured[/yellow]")
        console.print("\nDefault policy: [red]DENY ALL[/red]")
        console.print("\nTo configure a policy, create ~/.greenlang/config.json with:")

        example = """{
  "capability_policy": {
    "net": {
      "allow": false,
      "allowed_domains": []
    },
    "fs": {
      "allow": true
    },
    "subprocess": {
      "allow": false,
      "allowed_binaries": []
    },
    "clock": {
      "allow": false
    }
  }
}"""
        syntax = Syntax(example, "json", theme="monokai")
        console.print(syntax)
        return

    console.print(
        Panel.fit("[bold]Organization Capability Policy[/bold]", style="cyan")
    )

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Capability", style="cyan", width=15)
    table.add_column("Policy", style="green", width=10)
    table.add_column("Restrictions", style="white")

    for cap_name in ["net", "fs", "subprocess", "clock"]:
        cap_policy = capability_policy.get(cap_name, {})
        allowed = cap_policy.get("allow", False)
        status = "✓ Allowed" if allowed else "✗ Denied"

        restrictions = ""
        if cap_name == "net" and allowed:
            domains = cap_policy.get("allowed_domains", [])
            restrictions = f"Domains: {len(domains)}"
        elif cap_name == "subprocess" and allowed:
            binaries = cap_policy.get("allowed_binaries", [])
            restrictions = f"Binaries: {len(binaries)}"

        table.add_row(cap_name.title(), status, restrictions)

    console.print(table)


# Register commands with main CLI
def register_commands(cli):
    """Register capability commands with the main CLI"""
    cli.add_command(capabilities, name="capabilities")
    cli.add_command(capabilities, name="caps")  # Short alias
