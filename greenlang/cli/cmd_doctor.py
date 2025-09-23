"""
gl doctor - Environment diagnostics
"""

import typer
import sys
import os
import subprocess
import shutil
import platform
from pathlib import Path
from typing import Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import importlib.metadata as md

app = typer.Typer()
console = Console()


def check_command(command: str) -> Tuple[bool, str]:
    """Check if a command is available and get its version"""
    try:
        if shutil.which(command):
            # Try to get version
            try:
                result = subprocess.run(
                    [command, "--version"], capture_output=True, text=True, timeout=5
                )
                version = result.stdout.strip().split("\n")[0]
                return True, version
            except:
                return True, "installed"
        return False, "not found"
    except:
        return False, "error"


def check_python_package(package: str) -> Tuple[bool, str]:
    """Check if a Python package is installed"""
    try:
        version = md.version(package)
        return True, version
    except:
        return False, "not installed"


def check_directory(path: Path) -> Tuple[bool, str]:
    """Check if a directory exists and is writable"""
    if path.exists():
        if path.is_dir():
            # Check if writable
            test_file = path / ".gl_test"
            try:
                test_file.touch()
                test_file.unlink()
                return True, "OK writable"
            except:
                return True, "read-only"
        else:
            return False, "not a directory"
    else:
        # Try to create it
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True, "OK created"
        except:
            return False, "cannot create"


def check_kubernetes() -> Tuple[bool, str]:
    """Check Kubernetes connectivity"""
    try:
        result = subprocess.run(
            ["kubectl", "cluster-info"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Extract cluster name
            for line in result.stdout.split("\n"):
                if "is running at" in line:
                    return True, "OK connected"
            return True, "connected"
        else:
            return False, "not connected"
    except FileNotFoundError:
        return False, "kubectl not found"
    except:
        return False, "error"


def check_hub_auth() -> Tuple[bool, str]:
    """Check Hub authentication"""
    # Check for credentials in various locations
    cred_locations = [
        Path.home() / ".greenlang" / "credentials",
        Path.home() / ".config" / "greenlang" / "auth.json",
    ]

    for loc in cred_locations:
        if loc.exists():
            return True, f"OK {loc.name}"

    # Check environment variable
    if os.getenv("GL_HUB_TOKEN"):
        return True, "OK via GL_HUB_TOKEN"

    return False, "not configured"


@app.callback(invoke_without_command=True)
def doctor(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Check GreenLang installation and environment

    Checks:
    - Python version and packages
    - CLI tools (cosign, oras, kubectl)
    - Configuration and cache directories
    - Hub authentication
    - Policy bundles
    """
    if ctx.invoked_subcommand is not None:
        return

    console.print(
        Panel.fit(
            "[bold]GreenLang Environment Check[/bold]\n"
            "Checking system requirements and configuration...",
            title="gl doctor",
        )
    )

    checks = []

    # === Core Requirements ===
    console.print("\n[bold cyan]Core Requirements:[/bold cyan]")

    # Python version
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    py_ok = sys.version_info >= (3, 8)
    checks.append(("Python Version", py_version, py_ok))

    # GreenLang version
    try:
        gl_version = md.version("greenlang")
        checks.append(("GreenLang Version", gl_version, True))
    except:
        checks.append(("GreenLang Version", "development", True))

    # Typer
    typer_ok, typer_version = check_python_package("typer")
    checks.append(("Typer CLI Framework", typer_version, typer_ok))

    # === Supply Chain Tools ===
    console.print("\n[bold cyan]Supply Chain Tools:[/bold cyan]")

    # cosign
    cosign_ok, cosign_version = check_command("cosign")
    checks.append(("cosign (signatures)", cosign_version, cosign_ok))

    # oras
    oras_ok, oras_version = check_command("oras")
    checks.append(("oras (OCI registry)", oras_version, oras_ok))

    # git
    git_ok, git_version = check_command("git")
    checks.append(("git", git_version, git_ok))

    # === Runtime Backends ===
    console.print("\n[bold cyan]Runtime Backends:[/bold cyan]")

    # Docker
    docker_ok, docker_version = check_command("docker")
    checks.append(("Docker", docker_version, docker_ok))

    # Kubernetes
    k8s_ok, k8s_status = check_kubernetes()
    checks.append(("Kubernetes", k8s_status, k8s_ok))

    # === Configuration ===
    console.print("\n[bold cyan]Configuration:[/bold cyan]")

    # Config directory
    config_dir = Path.home() / ".greenlang"
    config_ok, config_status = check_directory(config_dir)
    checks.append(("Config Directory", str(config_dir), config_ok))

    # Cache directory
    cache_dir = Path(
        os.getenv("GL_CACHE_DIR", str(Path.home() / ".greenlang" / "cache"))
    )
    cache_ok, cache_status = check_directory(cache_dir)
    checks.append(("Cache Directory", f"{cache_dir} ({cache_status})", cache_ok))

    # Hub authentication
    hub_ok, hub_status = check_hub_auth()
    checks.append(("Hub Authentication", hub_status, hub_ok))

    # Policy bundle
    policy_dir = Path.home() / ".greenlang" / "policies"
    policy_ok = policy_dir.exists() and list(policy_dir.glob("*.rego"))
    policy_count = len(list(policy_dir.glob("*.rego"))) if policy_dir.exists() else 0
    checks.append(("Policy Bundle", f"{policy_count} policies", policy_ok))

    # === Platform Detection ===
    console.print("\n[bold cyan]Platform:[/bold cyan]")
    checks.append(("Operating System", platform.system(), True))
    checks.append(("Architecture", platform.machine(), True))
    checks.append(("Platform", platform.platform(), True))

    # === Display Results ===
    if json_output:
        import json

        output = []
        for name, value, status in checks:
            output.append(
                {"check": name, "value": value, "status": "pass" if status else "fail"}
            )
        console.print(json.dumps(output, indent=2))
    else:
        # Create table
        table = Table(show_header=False, box=None)
        table.add_column("Status", width=3)
        table.add_column("Check", style="cyan")
        table.add_column("Value")

        for name, value, status in checks:
            icon = "[green]OK[/green]" if status else "[red]FAIL[/red]"
            table.add_row(icon, name, value)

        console.print(table)

        # Overall summary
        passed = sum(1 for _, _, status in checks if status)
        total = len(checks)

        console.print(f"\n[bold]Summary:[/bold] {passed}/{total} checks passed")

        if passed == total:
            console.print(
                "[green]OK All checks passed! GreenLang is ready to use.[/green]"
            )
        else:
            console.print(
                "[yellow]WARNING: Some checks failed. See details above.[/yellow]"
            )

            # Provide fix suggestions
            if verbose:
                console.print("\n[bold]Suggested Fixes:[/bold]")

                if not cosign_ok:
                    console.print(
                        "  - Install cosign: https://docs.sigstore.dev/cosign/installation/"
                    )

                if not oras_ok:
                    console.print(
                        "  - Install oras: https://oras.land/docs/installation"
                    )

                if not docker_ok:
                    console.print(
                        "  - Install Docker: https://docs.docker.com/get-docker/"
                    )

                if not hub_ok:
                    console.print("  - Configure Hub auth: gl auth login")

                if not policy_ok:
                    console.print("  - Add policies: gl policy add <policy.rego>")

        # Environment variables
        if verbose:
            console.print("\n[bold]Environment Variables:[/bold]")
            env_vars = [
                ("GL_PROFILE", os.getenv("GL_PROFILE", "not set")),
                ("GL_REGION", os.getenv("GL_REGION", "not set")),
                ("GL_HUB", os.getenv("GL_HUB", "hub.greenlang.io")),
                ("GL_TELEMETRY", os.getenv("GL_TELEMETRY", "on")),
                ("GL_POLICY_BUNDLE", os.getenv("GL_POLICY_BUNDLE", "not set")),
                ("GL_CACHE_DIR", os.getenv("GL_CACHE_DIR", "~/.greenlang/cache")),
            ]

            for var, value in env_vars:
                console.print(f"  {var}: {value}")


@app.command("fix")
def fix(
    component: Optional[str] = typer.Argument(
        None, help="Component to fix (cosign|oras|policies)"
    )
):
    """Auto-fix common issues"""
    console.print("[cyan]Running auto-fix...[/cyan]\n")

    fixed = []

    # Fix config directory
    config_dir = Path.home() / ".greenlang"
    if not config_dir.exists():
        config_dir.mkdir(parents=True)
        fixed.append("Created config directory")

    # Fix cache directory
    cache_dir = config_dir / "cache"
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
        fixed.append("Created cache directory")

    # Fix policies directory
    policies_dir = config_dir / "policies"
    if not policies_dir.exists():
        policies_dir.mkdir(parents=True)
        fixed.append("Created policies directory")

        # Add default policy
        default_policy = """package greenlang.default

# Default allow-all policy
default allow = true

# Example deny rule (commented out)
# deny[msg] {
#     input.pipeline.backend == "k8s"
#     not input.profile == "prod"
#     msg := "K8s backend requires prod profile"
# }
"""
        with open(policies_dir / "default.rego", "w") as f:
            f.write(default_policy)
        fixed.append("Added default policy")

    # Component-specific fixes
    if component == "cosign":
        console.print("[cyan]Installing cosign...[/cyan]")
        # Platform-specific installation
        if platform.system() == "Darwin":
            subprocess.run(["brew", "install", "cosign"])
        elif platform.system() == "Linux":
            console.print(
                "Run: wget -O - https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64 | sudo tee /usr/local/bin/cosign > /dev/null && sudo chmod +x /usr/local/bin/cosign"
            )
        else:
            console.print("Visit: https://docs.sigstore.dev/cosign/installation/")

    elif component == "oras":
        console.print("[cyan]Installing oras...[/cyan]")
        if platform.system() == "Darwin":
            subprocess.run(["brew", "install", "oras"])
        elif platform.system() == "Linux":
            console.print(
                "Run: curl -LO https://github.com/oras-project/oras/releases/download/v1.0.0/oras_1.0.0_linux_amd64.tar.gz && tar -xzf oras_1.0.0_linux_amd64.tar.gz && sudo mv oras /usr/local/bin/"
            )
        else:
            console.print("Visit: https://oras.land/docs/installation")

    # Display results
    if fixed:
        console.print("[green]OK Fixed issues:[/green]")
        for fix in fixed:
            console.print(f"  - {fix}")
    else:
        console.print("[yellow]No issues to fix[/yellow]")

    console.print("\nRun [cyan]gl doctor[/cyan] to verify")
