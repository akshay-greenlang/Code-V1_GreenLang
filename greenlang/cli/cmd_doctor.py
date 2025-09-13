"""
Environment validation and diagnostic commands for GreenLang CLI
"""

import click
import json
import sys
import os
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
import time

console = Console()


@click.group()
def doctor():
    """System health and environment validation

    Check environment setup, dependencies, and configuration.
    """
    pass


@doctor.command(name="check")
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
@click.option("--fix", is_flag=True, help="Attempt to fix issues automatically")
def check_environment(output_json: bool, verbose: bool, fix: bool):
    """Check environment setup and dependencies

    Examples:
        gl doctor check                 # Basic health check
        gl doctor check --verbose       # Detailed diagnostics
        gl doctor check --json         # JSON output
        gl doctor check --fix          # Auto-fix issues
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task = progress.add_task("Running diagnostics...", total=None)

        # Run all checks
        results = {
            "system": check_system_info(),
            "python": check_python_environment(),
            "dependencies": check_dependencies(),
            "greenlang": check_greenlang_installation(),
            "project": check_project_structure(),
            "configuration": check_configuration(),
            "permissions": check_permissions(),
            "network": check_network_connectivity()
        }

        progress.update(task, description="Analysis complete!")

    # Analyze results
    all_issues = []
    all_warnings = []
    fixes_applied = []

    for category, result in results.items():
        all_issues.extend(result.get("issues", []))
        all_warnings.extend(result.get("warnings", []))

        # Apply fixes if requested
        if fix and "fixes" in result:
            for fix_item in result["fixes"]:
                try:
                    success = apply_fix(fix_item)
                    if success:
                        fixes_applied.append(fix_item["description"])
                except Exception as e:
                    all_issues.append(f"Failed to apply fix: {e}")

    # Determine overall health
    health_status = "HEALTHY" if not all_issues else "UNHEALTHY"
    if all_warnings and not all_issues:
        health_status = "WARNING"

    # Generate summary
    summary = {
        "status": health_status,
        "total_issues": len(all_issues),
        "total_warnings": len(all_warnings),
        "fixes_applied": len(fixes_applied),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if output_json:
        output = {
            "summary": summary,
            "results": results,
            "issues": all_issues,
            "warnings": all_warnings,
            "fixes_applied": fixes_applied
        }
        console.print(json.dumps(output, indent=2))
    else:
        display_health_report(summary, results, all_issues, all_warnings, fixes_applied, verbose)

    # Exit with appropriate code
    sys.exit(0 if health_status == "HEALTHY" else 1)


@doctor.command(name="deps")
@click.option("--install", is_flag=True, help="Install missing dependencies")
@click.option("--upgrade", is_flag=True, help="Upgrade existing dependencies")
def check_deps(install: bool, upgrade: bool):
    """Check and manage Python dependencies

    Examples:
        gl doctor deps                  # Check dependencies
        gl doctor deps --install        # Install missing ones
        gl doctor deps --upgrade        # Upgrade existing ones
    """
    console.print("[bold blue]Checking Python Dependencies[/bold blue]\n")

    # Core dependencies
    core_deps = [
        ("click", ">=8.0.0"),
        ("rich", ">=12.0.0"),
        ("pydantic", ">=1.8.0"),
        ("requests", ">=2.25.0"),
        ("PyYAML", ">=6.0"),
        ("jsonschema", ">=4.0.0"),
    ]

    # Optional dependencies
    optional_deps = [
        ("pandas", ">=1.3.0", "Data analysis"),
        ("numpy", ">=1.21.0", "Numerical computing"),
        ("matplotlib", ">=3.5.0", "Plotting"),
        ("jupyter", ">=1.0.0", "Notebook support"),
        ("pytest", ">=6.0.0", "Testing"),
        ("black", ">=21.0.0", "Code formatting"),
    ]

    missing_core = []
    missing_optional = []
    installed = []

    # Check core dependencies
    for package, version in core_deps:
        status = check_package(package, version)
        if status["installed"]:
            installed.append((package, status["version"], "core"))
        else:
            missing_core.append((package, version))

    # Check optional dependencies
    for item in optional_deps:
        package, version = item[0], item[1]
        description = item[2] if len(item) > 2 else ""
        status = check_package(package, version)
        if status["installed"]:
            installed.append((package, status["version"], "optional"))
        else:
            missing_optional.append((package, version, description))

    # Display results
    if installed:
        table = Table(title="Installed Dependencies")
        table.add_column("Package", style="green")
        table.add_column("Version", style="cyan")
        table.add_column("Type", style="yellow")

        for package, version, dep_type in installed:
            table.add_row(package, version, dep_type)

        console.print(table)

    if missing_core:
        console.print("\n[bold red]Missing Core Dependencies:[/bold red]")
        for package, version in missing_core:
            console.print(f"  [red]x[/red] {package} {version}")

    if missing_optional:
        console.print("\n[bold yellow]Missing Optional Dependencies:[/bold yellow]")
        for item in missing_optional:
            package, version = item[0], item[1]
            description = item[2] if len(item) > 2 else ""
            console.print(f"  [yellow]o[/yellow] {package} {version} - {description}")

    # Install missing dependencies
    if install and (missing_core or missing_optional):
        console.print("\n[bold blue]Installing Dependencies...[/bold blue]")

        to_install = [pkg for pkg, _ in missing_core]
        if click.confirm("Also install optional dependencies?"):
            to_install.extend([pkg for pkg, _, _ in missing_optional])

        for package in to_install:
            try:
                console.print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                                    capture_output=True)
                console.print(f"  [green]+[/green] {package} installed")
            except subprocess.CalledProcessError:
                console.print(f"  [red]x[/red] Failed to install {package}")

    # Upgrade dependencies
    if upgrade and installed:
        console.print("\n[bold blue]Upgrading Dependencies...[/bold blue]")

        for package, current_version, _ in installed:
            try:
                console.print(f"Upgrading {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package],
                                    capture_output=True)
                console.print(f"  [green]+[/green] {package} upgraded")
            except subprocess.CalledProcessError:
                console.print(f"  [yellow]o[/yellow] {package} already up to date")


@doctor.command(name="config")
def check_config():
    """Validate GreenLang configuration

    Examples:
        gl doctor config               # Check configuration
    """
    console.print("[bold blue]Configuration Validation[/bold blue]\n")

    config_files = [
        ".env",
        "greenlang.yaml",
        "greenlang.yml",
        "config/greenlang.yaml"
    ]

    found_configs = []
    issues = []

    # Check for configuration files
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            found_configs.append(str(path))
            try:
                validate_config_file(path)
            except Exception as e:
                issues.append(f"{config_file}: {e}")

    if found_configs:
        console.print("[green]+[/green] Found configuration files:")
        for config in found_configs:
            console.print(f"  • {config}")
    else:
        console.print("[yellow]o[/yellow] No configuration files found")
        console.print("  Consider creating a .env file with your settings")

    # Check environment variables
    env_vars = [
        ("GREENLANG_ENV", "Environment setting"),
        ("GREENLANG_LOG_LEVEL", "Logging level"),
        ("OPENAI_API_KEY", "OpenAI API key (optional)"),
        ("ANTHROPIC_API_KEY", "Anthropic API key (optional)"),
    ]

    console.print("\n[bold]Environment Variables:[/bold]")
    for var, description in env_vars:
        value = os.environ.get(var)
        if value:
            # Mask sensitive values
            if "key" in var.lower() or "token" in var.lower():
                masked_value = value[:8] + "..." if len(value) > 8 else "***"
                console.print(f"  [green]+[/green] {var} = {masked_value}")
            else:
                console.print(f"  [green]+[/green] {var} = {value}")
        else:
            console.print(f"  [dim]o[/dim] {var} (not set) - {description}")

    if issues:
        console.print(f"\n[bold red]Configuration Issues:[/bold red]")
        for issue in issues:
            console.print(f"  [red]x[/red] {issue}")


@doctor.command(name="network")
def check_network():
    """Test network connectivity

    Examples:
        gl doctor network              # Test network connections
    """
    console.print("[bold blue]Network Connectivity Check[/bold blue]\n")

    endpoints = [
        ("github.com", 443, "GitHub (for packages)"),
        ("pypi.org", 443, "PyPI (for Python packages)"),
        ("api.openai.com", 443, "OpenAI API"),
        ("api.anthropic.com", 443, "Anthropic API"),
    ]

    for host, port, description in endpoints:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                console.print(f"  [green]+[/green] {host}:{port} - {description}")
            else:
                console.print(f"  [red]x[/red] {host}:{port} - {description} (unreachable)")

        except Exception as e:
            console.print(f"  [red]x[/red] {host}:{port} - {description} (error: {e})")


def check_system_info() -> Dict[str, Any]:
    """Check system information"""
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "issues": [],
        "warnings": []
    }


def check_python_environment() -> Dict[str, Any]:
    """Check Python environment"""
    issues = []
    warnings = []

    # Check Python version
    python_version = tuple(map(int, platform.python_version().split('.')))
    if python_version < (3, 8):
        issues.append("Python 3.8 or higher is required")
    elif python_version < (3, 9):
        warnings.append("Python 3.9+ recommended for better performance")

    # Check virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        warnings.append("Not running in a virtual environment (recommended)")

    # Check pip
    pip_available = shutil.which('pip') is not None
    if not pip_available:
        issues.append("pip is not available")

    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "virtual_env": in_venv,
        "pip_available": pip_available,
        "issues": issues,
        "warnings": warnings
    }


def check_dependencies() -> Dict[str, Any]:
    """Check Python dependencies"""
    required_packages = ["click", "rich", "pydantic", "requests", "PyYAML"]
    optional_packages = ["pandas", "numpy", "matplotlib", "jsonschema"]

    missing_required = []
    missing_optional = []
    installed = []

    for package in required_packages:
        try:
            __import__(package)
            installed.append(package)
        except ImportError:
            missing_required.append(package)

    for package in optional_packages:
        try:
            __import__(package)
            installed.append(package)
        except ImportError:
            missing_optional.append(package)

    issues = []
    warnings = []
    fixes = []

    if missing_required:
        issues.extend([f"Required package missing: {pkg}" for pkg in missing_required])
        fixes.extend([{"type": "install_package", "package": pkg, "description": f"Install {pkg}"}
                     for pkg in missing_required])

    if missing_optional:
        warnings.extend([f"Optional package missing: {pkg}" for pkg in missing_optional])

    return {
        "installed": installed,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "issues": issues,
        "warnings": warnings,
        "fixes": fixes
    }


def check_greenlang_installation() -> Dict[str, Any]:
    """Check GreenLang installation"""
    issues = []
    warnings = []

    try:
        import greenlang
        version = getattr(greenlang, '__version__', 'unknown')
        installed = True
    except ImportError:
        issues.append("GreenLang package not found")
        installed = False
        version = None

    # Check for CLI availability
    cli_available = shutil.which('gl') is not None
    if not cli_available:
        warnings.append("GreenLang CLI (gl) not found in PATH")

    return {
        "installed": installed,
        "version": version,
        "cli_available": cli_available,
        "issues": issues,
        "warnings": warnings
    }


def check_project_structure() -> Dict[str, Any]:
    """Check project structure"""
    warnings = []
    suggestions = []

    # Check for common directories
    expected_dirs = ["pipelines", "data", "reports", "logs", "cache"]
    missing_dirs = [d for d in expected_dirs if not Path(d).exists()]

    if missing_dirs:
        warnings.extend([f"Directory not found: {d}" for d in missing_dirs])
        suggestions.append("Run 'gl init project' to create project structure")

    # Check for configuration files
    config_files = [".env", "greenlang.yaml", "greenlang.yml"]
    has_config = any(Path(f).exists() for f in config_files)

    if not has_config:
        warnings.append("No configuration file found")
        suggestions.append("Create .env file with your settings")

    return {
        "missing_directories": missing_dirs,
        "has_configuration": has_config,
        "issues": [],
        "warnings": warnings,
        "suggestions": suggestions
    }


def check_configuration() -> Dict[str, Any]:
    """Check configuration settings"""
    issues = []
    warnings = []

    # Check environment variables
    important_vars = ["GREENLANG_ENV", "GREENLANG_LOG_LEVEL"]
    missing_vars = [var for var in important_vars if not os.environ.get(var)]

    if missing_vars:
        warnings.extend([f"Environment variable not set: {var}" for var in missing_vars])

    # Check for API keys if needed
    api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    has_api_key = any(os.environ.get(key) for key in api_keys)

    if not has_api_key:
        warnings.append("No API keys configured (needed for AI features)")

    return {
        "missing_env_vars": missing_vars,
        "has_api_keys": has_api_key,
        "issues": issues,
        "warnings": warnings
    }


def check_permissions() -> Dict[str, Any]:
    """Check file system permissions"""
    issues = []
    warnings = []

    # Check write permissions for common directories
    dirs_to_check = [
        Path.home() / ".greenlang",
        Path.cwd(),
        Path("/tmp") if Path("/tmp").exists() else Path.cwd() / "tmp"
    ]

    for directory in dirs_to_check:
        if directory.exists():
            if not os.access(directory, os.W_OK):
                issues.append(f"No write permission: {directory}")
        else:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                issues.append(f"Cannot create directory: {directory}")

    return {
        "issues": issues,
        "warnings": warnings
    }


def check_network_connectivity() -> Dict[str, Any]:
    """Check network connectivity"""
    issues = []
    warnings = []

    # Test basic connectivity
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        network_available = True
    except OSError:
        network_available = False
        warnings.append("No internet connectivity detected")

    return {
        "network_available": network_available,
        "issues": issues,
        "warnings": warnings
    }


def check_package(package: str, version_req: Optional[str] = None) -> Dict[str, Any]:
    """Check if a package is installed with version requirements"""
    try:
        import importlib.metadata
        version = importlib.metadata.version(package)
        installed = True

        # Simple version checking (could be enhanced)
        version_ok = True
        if version_req and version_req.startswith(">="):
            required = version_req[2:]
            version_ok = version >= required

        return {
            "installed": installed,
            "version": version,
            "version_ok": version_ok
        }
    except importlib.metadata.PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
            "version_ok": False
        }


def validate_config_file(config_path: Path):
    """Validate a configuration file"""
    if config_path.suffix in ['.yaml', '.yml']:
        import yaml
        with open(config_path, 'r') as f:
            yaml.safe_load(f)
    elif config_path.name == '.env':
        # Basic .env validation
        with open(config_path, 'r') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#') and '=' not in line:
                    raise ValueError(f"Invalid .env format at line {line_no}")


def apply_fix(fix_item: Dict[str, Any]) -> bool:
    """Apply an automatic fix"""
    if fix_item["type"] == "install_package":
        package = fix_item["package"]
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                                capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    return False


def display_health_report(summary: Dict, results: Dict, issues: List[str], warnings: List[str], fixes_applied: List[str], verbose: bool):
    """Display health report in formatted output"""

    # Status panel
    status_color = "green" if summary["status"] == "HEALTHY" else "red" if summary["status"] == "UNHEALTHY" else "yellow"
    status_panel = Panel(
        f"[bold {status_color}]{summary['status']}[/bold {status_color}]\n\n"
        f"Issues: {summary['total_issues']}\n"
        f"Warnings: {summary['total_warnings']}\n"
        f"Fixes Applied: {summary['fixes_applied']}\n"
        f"Checked: {summary['timestamp']}",
        title="System Health",
        expand=False
    )
    console.print(status_panel)

    # Issues
    if issues:
        console.print(f"\n[bold red]Issues ({len(issues)}):[/bold red]")
        for issue in issues:
            console.print(f"  [red]x[/red] {issue}")

    # Warnings
    if warnings:
        console.print(f"\n[bold yellow]Warnings ({len(warnings)}):[/bold yellow]")
        for warning in warnings:
            console.print(f"  [yellow]o[/yellow] {warning}")

    # Applied fixes
    if fixes_applied:
        console.print(f"\n[bold green]Fixes Applied ({len(fixes_applied)}):[/bold green]")
        for fix in fixes_applied:
            console.print(f"  [green]+[/green] {fix}")

    # Detailed results if verbose
    if verbose:
        console.print("\n[bold]Detailed Results:[/bold]")
        for category, result in results.items():
            console.print(f"\n[cyan]{category.title()}:[/cyan]")
            for key, value in result.items():
                if key not in ["issues", "warnings", "fixes"]:
                    console.print(f"  {key}: {value}")

    # Suggestions
    console.print("\n[bold cyan]Recommendations:[/bold cyan]")
    if summary["status"] == "HEALTHY":
        console.print("  [green]+[/green] Your environment is healthy!")
    else:
        console.print("  • Run 'gl doctor check --fix' to auto-fix issues")
        console.print("  • Check documentation for manual fixes")
        console.print("  • Use 'gl doctor deps --install' for missing packages")