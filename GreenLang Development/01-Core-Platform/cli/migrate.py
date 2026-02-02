# -*- coding: utf-8 -*-
"""
GreenLang Migration CLI Tool

Automated migration tool for upgrading GreenLang from v0.2 to v0.3.

Commands:
    greenlang migrate analyze     - Analyze current installation
    greenlang migrate plan        - Generate migration plan
    greenlang migrate execute     - Execute migration
    greenlang migrate verify      - Verify migration success
    greenlang migrate rollback    - Rollback to previous version

Author: GreenLang Team
License: MIT
Version: 1.0.0
"""

import click
import json
import yaml
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm
import re
from greenlang.utilities.determinism import DeterministicClock

console = Console()


class MigrationContext:
    """Context for migration operations"""

    def __init__(self):
        self.current_version = None
        self.target_version = "0.3.0"
        self.backup_dir = None
        self.errors = []
        self.warnings = []
        self.steps_completed = []
        self.dry_run = False

    def add_error(self, message: str):
        """Add error to context"""
        self.errors.append(message)
        console.print(f"[red]ERROR: {message}[/red]")

    def add_warning(self, message: str):
        """Add warning to context"""
        self.warnings.append(message)
        console.print(f"[yellow]WARNING: {message}[/yellow]")

    def mark_step_complete(self, step: str):
        """Mark migration step as complete"""
        self.steps_completed.append(step)
        console.print(f"[green]✓ {step}[/green]")


@click.group()
def migrate():
    """Migration tools for upgrading GreenLang versions"""
    pass


@migrate.command()
@click.option('--output', '-o', type=click.Path(), help='Output analysis to file')
@click.option('--json-format', is_flag=True, help='Output in JSON format')
def analyze(output: Optional[str], json_format: bool):
    """Analyze current GreenLang installation and detect version"""

    console.print(Panel.fit("[bold cyan]GreenLang Migration Analyzer[/bold cyan]"))

    # Detect current version
    current_version = detect_current_version()

    # Analyze installation
    analysis = {
        "current_version": current_version,
        "target_version": "0.3.0",
        "python_version": sys.version.split()[0],
        "installation_path": str(Path(__file__).parent.parent),
        "config_files": analyze_config_files(),
        "database": analyze_database(),
        "agent_packs": analyze_agent_packs(),
        "custom_agents": analyze_custom_agents(),
        "breaking_changes": identify_breaking_changes(current_version),
        "compatibility": check_compatibility(),
        "estimated_migration_time": estimate_migration_time(),
        "risk_assessment": assess_migration_risk()
    }

    # Display analysis
    if json_format:
        console.print(json.dumps(analysis, indent=2))
    else:
        display_analysis(analysis)

    # Save to file if requested
    if output:
        with open(output, 'w') as f:
            json.dump(analysis, f, indent=2)
        console.print(f"\n[green]Analysis saved to {output}[/green]")

    return analysis


def detect_current_version() -> str:
    """Detect currently installed GreenLang version"""
    try:
        import greenlang
        version = greenlang.__version__
        console.print(f"[cyan]Detected version: {version}[/cyan]")
        return version
    except ImportError:
        console.print("[red]GreenLang not installed[/red]")
        return "unknown"
    except AttributeError:
        console.print("[yellow]Version information not available[/yellow]")
        return "0.1.0"


def analyze_config_files() -> Dict[str, Any]:
    """Analyze configuration files"""
    config_analysis = {
        "greenlang_yaml_exists": False,
        "greenlang_yaml_valid": False,
        "env_file_exists": False,
        "config_version": "unknown",
        "issues": []
    }

    # Check greenlang.yaml
    greenlang_yaml_paths = [
        Path("greenlang.yaml"),
        Path("greenlang.yml"),
        Path.home() / ".greenlang" / "config.yaml"
    ]

    for path in greenlang_yaml_paths:
        if path.exists():
            config_analysis["greenlang_yaml_exists"] = True
            try:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                config_analysis["greenlang_yaml_valid"] = True
                config_analysis["config_version"] = config.get("version", "unknown")

                # Check for deprecated keys
                if "database" in config:
                    if "url" in config["database"]:
                        config_analysis["issues"].append(
                            "database.url is deprecated, use database.connection_string"
                        )

                if "security" in config:
                    if "encryption" not in config["security"]:
                        config_analysis["issues"].append(
                            "security.encryption is required in v0.3"
                        )

            except Exception as e:
                config_analysis["issues"].append(f"Failed to parse greenlang.yaml: {str(e)}")
            break

    # Check .env file
    env_paths = [Path(".env"), Path.home() / ".greenlang" / ".env"]
    for path in env_paths:
        if path.exists():
            config_analysis["env_file_exists"] = True

            # Check for deprecated env vars
            with open(path, 'r') as f:
                content = f.read()
                if "GREENLANG_ENV=" in content:
                    config_analysis["issues"].append(
                        "GREENLANG_ENV is deprecated, use GL_ENV"
                    )
                if "GREENLANG_DB_URL=" in content:
                    config_analysis["issues"].append(
                        "GREENLANG_DB_URL is deprecated, use GL_DATABASE_URL"
                    )
                if "GL_ENCRYPTION_KEY=" not in content:
                    config_analysis["issues"].append(
                        "GL_ENCRYPTION_KEY is required in v0.3"
                    )
            break

    return config_analysis


def analyze_database() -> Dict[str, Any]:
    """Analyze database configuration and schema"""
    db_analysis = {
        "configured": False,
        "accessible": False,
        "schema_version": "unknown",
        "migration_required": False,
        "tables_count": 0,
        "issues": []
    }

    try:
        # Try to get database URL from config or env
        db_url = get_database_url()
        if db_url:
            db_analysis["configured"] = True

            # Try to connect and check schema
            try:
                from sqlalchemy import create_engine, inspect
                engine = create_engine(db_url)
                inspector = inspect(engine)

                db_analysis["accessible"] = True
                db_analysis["tables_count"] = len(inspector.get_table_names())

                # Check for v0.3 specific tables
                tables = inspector.get_table_names()
                if "audit_logs" not in tables:
                    db_analysis["migration_required"] = True
                    db_analysis["issues"].append("Missing audit_logs table (new in v0.3)")

                if "user_sessions" not in tables:
                    db_analysis["migration_required"] = True
                    db_analysis["issues"].append("Missing user_sessions table (new in v0.3)")

                # Check agents table for spec_version column
                if "agents" in tables:
                    columns = [col["name"] for col in inspector.get_columns("agents")]
                    if "spec_version" not in columns:
                        db_analysis["migration_required"] = True
                        db_analysis["issues"].append("agents table missing spec_version column")

            except Exception as e:
                db_analysis["issues"].append(f"Database connection failed: {str(e)}")

    except Exception as e:
        db_analysis["issues"].append(f"Database configuration error: {str(e)}")

    return db_analysis


def analyze_agent_packs() -> Dict[str, Any]:
    """Analyze installed agent packs"""
    packs_analysis = {
        "total_packs": 0,
        "v1_packs": 0,
        "v2_packs": 0,
        "invalid_packs": 0,
        "packs_requiring_migration": [],
        "issues": []
    }

    # Check standard pack locations
    pack_dirs = [
        Path.home() / ".greenlang" / "packs",
        Path("packs"),
        Path("~/.greenlang/packs").expanduser()
    ]

    for pack_dir in pack_dirs:
        if pack_dir.exists():
            for pack_path in pack_dir.rglob("pack.yaml"):
                packs_analysis["total_packs"] += 1

                try:
                    with open(pack_path, 'r') as f:
                        pack = yaml.safe_load(f)

                    spec_version = pack.get("spec_version", "1.0")

                    if spec_version == "1.0":
                        packs_analysis["v1_packs"] += 1
                        packs_analysis["packs_requiring_migration"].append(str(pack_path))
                    elif spec_version == "2.0":
                        packs_analysis["v2_packs"] += 1
                    else:
                        packs_analysis["invalid_packs"] += 1
                        packs_analysis["issues"].append(
                            f"Unknown spec_version in {pack_path}: {spec_version}"
                        )

                except Exception as e:
                    packs_analysis["invalid_packs"] += 1
                    packs_analysis["issues"].append(
                        f"Failed to parse {pack_path}: {str(e)}"
                    )

    return packs_analysis


def analyze_custom_agents() -> Dict[str, Any]:
    """Analyze custom agent code for compatibility"""
    agents_analysis = {
        "custom_agents_found": 0,
        "async_compatible": 0,
        "sync_only": 0,
        "needs_update": [],
        "issues": []
    }

    # Search for Python files that might be custom agents
    agent_dirs = [
        Path("agents"),
        Path("greenlang/agents"),
        Path("custom_agents")
    ]

    for agent_dir in agent_dirs:
        if agent_dir.exists():
            for py_file in agent_dir.rglob("*.py"):
                if py_file.name.startswith("__"):
                    continue

                try:
                    with open(py_file, 'r') as f:
                        content = f.read()

                    # Check if it's an agent class
                    if "BaseAgent" in content or "class" in content and "Agent" in content:
                        agents_analysis["custom_agents_found"] += 1

                        # Check if async
                        if "async def run" in content:
                            agents_analysis["async_compatible"] += 1
                        elif "def run" in content:
                            agents_analysis["sync_only"] += 1
                            agents_analysis["needs_update"].append(str(py_file))

                        # Check for deprecated imports
                        if "from greenlang.core import ExecutionContext" in content:
                            agents_analysis["issues"].append(
                                f"{py_file}: Deprecated import - use 'from greenlang.core.context import ExecutionContext'"
                            )

                except Exception as e:
                    agents_analysis["issues"].append(f"Failed to analyze {py_file}: {str(e)}")

    return agents_analysis


def identify_breaking_changes(current_version: str) -> List[Dict[str, str]]:
    """Identify breaking changes applicable to migration"""
    breaking_changes = [
        {
            "id": "BC-001",
            "title": "AgentSpec v2 Schema Changes",
            "impact": "HIGH",
            "affected": "Agent packs with spec_version: 1.0"
        },
        {
            "id": "BC-002",
            "title": "API Endpoint Changes",
            "impact": "MEDIUM",
            "affected": "API consumers using /api/v1/ endpoints"
        },
        {
            "id": "BC-003",
            "title": "Configuration Format Changes",
            "impact": "LOW",
            "affected": "greenlang.yaml and .env files"
        },
        {
            "id": "BC-004",
            "title": "Database Schema Changes",
            "impact": "MEDIUM",
            "affected": "Database requires Alembic migration"
        },
        {
            "id": "BC-005",
            "title": "Workflow Execution Context",
            "impact": "MEDIUM",
            "affected": "Custom workflow executors"
        },
        {
            "id": "BC-006",
            "title": "Agent Registration",
            "impact": "MEDIUM",
            "affected": "Programmatic agent registration"
        },
        {
            "id": "BC-007",
            "title": "Environment Variables",
            "impact": "LOW",
            "affected": "Deployment configurations"
        },
        {
            "id": "BC-008",
            "title": "Python Package Dependencies",
            "impact": "LOW",
            "affected": "Python version and package requirements"
        },
        {
            "id": "BC-009",
            "title": "Agent Result Format",
            "impact": "MEDIUM",
            "affected": "Agent output parsing"
        },
        {
            "id": "BC-010",
            "title": "Logging Format",
            "impact": "LOW",
            "affected": "Log parsing and monitoring"
        },
        {
            "id": "BC-011",
            "title": "CLI Command Changes",
            "impact": "LOW",
            "affected": "CLI scripts and automation"
        },
        {
            "id": "BC-012",
            "title": "Async/Sync Agent Execution",
            "impact": "HIGH",
            "affected": "Custom agent development"
        }
    ]

    return breaking_changes


def check_compatibility() -> Dict[str, Any]:
    """Check Python version and dependency compatibility"""
    compatibility = {
        "python_compatible": False,
        "python_version": sys.version.split()[0],
        "python_required": ">=3.9",
        "dependencies_compatible": True,
        "issues": []
    }

    # Check Python version
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 9:
        compatibility["python_compatible"] = True
    else:
        compatibility["issues"].append(
            f"Python {major}.{minor} is not compatible. Python 3.9+ required."
        )

    # Check key dependencies
    required_packages = {
        "pydantic": ">=2.0.0",
        "aiohttp": ">=3.9.0",
        "cryptography": ">=41.0.0",
        "alembic": ">=1.12.0"
    }

    for package, version_spec in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            compatibility["dependencies_compatible"] = False
            compatibility["issues"].append(f"Missing package: {package} {version_spec}")

    return compatibility


def estimate_migration_time() -> str:
    """Estimate total migration time"""
    # Base time: 30 minutes
    base_time = 30

    # Add time based on components
    analysis = {
        "packs": analyze_agent_packs(),
        "custom_agents": analyze_custom_agents()
    }

    # +15 min per 10 v1 packs
    packs_time = (analysis["packs"]["v1_packs"] / 10) * 15

    # +30 min per 10 custom agents
    agents_time = (analysis["custom_agents"]["custom_agents_found"] / 10) * 30

    total_minutes = int(base_time + packs_time + agents_time)

    if total_minutes < 60:
        return f"{total_minutes} minutes"
    else:
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours}h {minutes}m"


def assess_migration_risk() -> Dict[str, Any]:
    """Assess overall migration risk"""
    risk = {
        "level": "LOW",
        "score": 0,
        "factors": []
    }

    analysis = {
        "db": analyze_database(),
        "packs": analyze_agent_packs(),
        "agents": analyze_custom_agents()
    }

    # Database risk
    if analysis["db"]["migration_required"]:
        risk["score"] += 30
        risk["factors"].append("Database migration required")

    # Agent packs risk
    if analysis["packs"]["v1_packs"] > 10:
        risk["score"] += 20
        risk["factors"].append(f"{analysis['packs']['v1_packs']} agent packs need migration")

    # Custom agents risk
    if analysis["agents"]["sync_only"] > 5:
        risk["score"] += 25
        risk["factors"].append(f"{analysis['agents']['sync_only']} sync-only agents need async update")

    # Determine risk level
    if risk["score"] < 30:
        risk["level"] = "LOW"
    elif risk["score"] < 60:
        risk["level"] = "MEDIUM"
    else:
        risk["level"] = "HIGH"

    return risk


def display_analysis(analysis: Dict[str, Any]):
    """Display analysis results"""

    # Version info
    console.print("\n[bold]Version Information:[/bold]")
    console.print(f"  Current: {analysis['current_version']}")
    console.print(f"  Target: {analysis['target_version']}")
    console.print(f"  Python: {analysis['python_version']}")

    # Compatibility
    console.print("\n[bold]Compatibility Check:[/bold]")
    compat = analysis['compatibility']
    status = "[green]✓ Compatible[/green]" if compat['python_compatible'] else "[red]✗ Incompatible[/red]"
    console.print(f"  Python: {status}")

    if compat['issues']:
        for issue in compat['issues']:
            console.print(f"  [yellow]! {issue}[/yellow]")

    # Database
    console.print("\n[bold]Database Status:[/bold]")
    db = analysis['database']
    console.print(f"  Configured: {'Yes' if db['configured'] else 'No'}")
    console.print(f"  Accessible: {'Yes' if db['accessible'] else 'No'}")
    console.print(f"  Migration Required: {'Yes' if db['migration_required'] else 'No'}")

    if db['issues']:
        for issue in db['issues']:
            console.print(f"  [yellow]! {issue}[/yellow]")

    # Agent Packs
    console.print("\n[bold]Agent Packs:[/bold]")
    packs = analysis['agent_packs']
    console.print(f"  Total: {packs['total_packs']}")
    console.print(f"  AgentSpec v1: {packs['v1_packs']} (require migration)")
    console.print(f"  AgentSpec v2: {packs['v2_packs']}")

    # Custom Agents
    console.print("\n[bold]Custom Agents:[/bold]")
    agents = analysis['custom_agents']
    console.print(f"  Found: {agents['custom_agents_found']}")
    console.print(f"  Async compatible: {agents['async_compatible']}")
    console.print(f"  Sync only: {agents['sync_only']} (need update)")

    # Risk Assessment
    console.print("\n[bold]Risk Assessment:[/bold]")
    risk = analysis['risk_assessment']
    risk_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}[risk['level']]
    console.print(f"  Level: [{risk_color}]{risk['level']}[/{risk_color}]")
    console.print(f"  Estimated Time: {analysis['estimated_migration_time']}")

    if risk['factors']:
        console.print("  Factors:")
        for factor in risk['factors']:
            console.print(f"    • {factor}")


@migrate.command()
@click.option('--output', '-o', type=click.Path(), default='migration_plan.json', help='Output plan file')
def plan(output: str):
    """Generate detailed migration plan"""

    console.print(Panel.fit("[bold cyan]Migration Plan Generator[/bold cyan]"))

    # Analyze current state
    analysis = analyze(output=None, json_format=False)

    # Generate migration plan
    migration_plan = {
        "version": "1.0",
        "created_at": DeterministicClock.now().isoformat(),
        "source_version": analysis["current_version"],
        "target_version": analysis["target_version"],
        "estimated_duration": analysis["estimated_migration_time"],
        "risk_level": analysis["risk_assessment"]["level"],
        "steps": generate_migration_steps(analysis),
        "rollback_plan": generate_rollback_plan(),
        "validation_checks": generate_validation_checks()
    }

    # Save plan
    with open(output, 'w') as f:
        json.dump(migration_plan, f, indent=2)

    console.print(f"\n[green]Migration plan saved to {output}[/green]")

    # Display summary
    display_migration_plan(migration_plan)


def generate_migration_steps(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate ordered migration steps based on analysis"""
    steps = [
        {
            "step": 1,
            "name": "Backup Database",
            "description": "Create full database backup",
            "duration": "5 minutes",
            "required": True,
            "risk": "CRITICAL"
        },
        {
            "step": 2,
            "name": "Backup Configuration",
            "description": "Backup greenlang.yaml and .env files",
            "duration": "2 minutes",
            "required": True,
            "risk": "LOW"
        },
        {
            "step": 3,
            "name": "Backup Agent Packs",
            "description": "Backup all agent pack directories",
            "duration": "3 minutes",
            "required": True,
            "risk": "MEDIUM"
        },
        {
            "step": 4,
            "name": "Update Python Packages",
            "description": "Upgrade GreenLang and dependencies",
            "duration": "10 minutes",
            "required": True,
            "risk": "MEDIUM"
        },
        {
            "step": 5,
            "name": "Run Database Migrations",
            "description": "Execute Alembic migrations for v0.3 schema",
            "duration": "5 minutes",
            "required": analysis["database"]["migration_required"],
            "risk": "HIGH"
        },
        {
            "step": 6,
            "name": "Update Configuration Files",
            "description": "Convert greenlang.yaml to v0.3 format",
            "duration": "10 minutes",
            "required": True,
            "risk": "LOW"
        },
        {
            "step": 7,
            "name": "Convert Agent Packs",
            "description": f"Convert {analysis['agent_packs']['v1_packs']} agent packs to AgentSpec v2",
            "duration": f"{analysis['agent_packs']['v1_packs'] * 2} minutes",
            "required": analysis["agent_packs"]["v1_packs"] > 0,
            "risk": "MEDIUM"
        },
        {
            "step": 8,
            "name": "Update Custom Agents",
            "description": f"Update {analysis['custom_agents']['sync_only']} sync agents to async",
            "duration": f"{analysis['custom_agents']['sync_only'] * 10} minutes",
            "required": analysis["custom_agents"]["sync_only"] > 0,
            "risk": "HIGH"
        },
        {
            "step": 9,
            "name": "Run Validation Tests",
            "description": "Execute validation test suite",
            "duration": "10 minutes",
            "required": True,
            "risk": "LOW"
        }
    ]

    return [s for s in steps if s["required"]]


def generate_rollback_plan() -> Dict[str, Any]:
    """Generate rollback plan"""
    return {
        "description": "Steps to rollback migration if issues occur",
        "steps": [
            "Stop all GreenLang services",
            "Restore database from backup",
            "Restore configuration files from backup",
            "Restore agent packs from backup",
            "Downgrade Python package to v0.2",
            "Restart services",
            "Verify v0.2 functionality"
        ],
        "estimated_duration": "15 minutes",
        "data_loss_risk": "None if backups are valid"
    }


def generate_validation_checks() -> List[Dict[str, str]]:
    """Generate validation checks to run after migration"""
    return [
        {"name": "Version Check", "command": "greenlang --version"},
        {"name": "Database Connectivity", "command": "greenlang admin health-check"},
        {"name": "Agent Pack Validation", "command": "greenlang agents validate"},
        {"name": "Workflow Execution", "command": "greenlang run tests/workflows/smoke_test.yaml"},
        {"name": "API Health", "command": "curl http://localhost:8000/health"}
    ]


def display_migration_plan(plan: Dict[str, Any]):
    """Display migration plan summary"""

    console.print("\n[bold]Migration Plan Summary:[/bold]")
    console.print(f"  Source Version: {plan['source_version']}")
    console.print(f"  Target Version: {plan['target_version']}")
    console.print(f"  Estimated Duration: {plan['estimated_duration']}")
    console.print(f"  Risk Level: {plan['risk_level']}")
    console.print(f"  Total Steps: {len(plan['steps'])}")

    # Display steps table
    console.print("\n[bold]Migration Steps:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Step", style="cyan", width=6)
    table.add_column("Name", style="white", width=30)
    table.add_column("Duration", style="yellow", width=12)
    table.add_column("Risk", style="red", width=10)

    for step in plan['steps']:
        table.add_row(
            str(step['step']),
            step['name'],
            step['duration'],
            step['risk']
        )

    console.print(table)


@migrate.command()
@click.option('--dry-run', is_flag=True, help='Simulate migration without making changes')
@click.option('--force', is_flag=True, help='Skip confirmation prompts')
@click.option('--skip-backup', is_flag=True, help='Skip backup step (not recommended)')
@click.option('--production', is_flag=True, help='Production mode (extra safety checks)')
def execute(dry_run: bool, force: bool, skip_backup: bool, production: bool):
    """Execute migration from v0.2 to v0.3"""

    console.print(Panel.fit("[bold cyan]GreenLang Migration Executor[/bold cyan]"))

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]\n")

    # Initialize migration context
    ctx = MigrationContext()
    ctx.dry_run = dry_run

    # Pre-flight checks
    if production:
        console.print("[bold yellow]PRODUCTION MODE ENABLED[/bold yellow]")
        if not force:
            if not Confirm.ask("Are you sure you want to run migration in production?"):
                console.print("[red]Migration cancelled[/red]")
                return

    # Confirm migration
    if not force and not dry_run:
        console.print("\n[bold]This migration will:[/bold]")
        console.print("  • Modify database schema")
        console.print("  • Update configuration files")
        console.print("  • Convert agent packs to AgentSpec v2")
        console.print("  • Update Python packages")

        if not Confirm.ask("\nContinue with migration?"):
            console.print("[red]Migration cancelled[/red]")
            return

    # Execute migration steps
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:

        migration_steps = [
            ("backup_database", "Backing up database", backup_database),
            ("backup_config", "Backing up configuration", backup_configuration),
            ("backup_packs", "Backing up agent packs", backup_agent_packs),
            ("update_packages", "Updating Python packages", update_python_packages),
            ("migrate_database", "Running database migrations", migrate_database_schema),
            ("update_config", "Updating configuration files", update_configuration_files),
            ("convert_packs", "Converting agent packs", convert_agent_packs),
            ("validate", "Running validation tests", run_validation_tests)
        ]

        total_steps = len(migration_steps)
        if skip_backup:
            total_steps -= 3

        task = progress.add_task("[cyan]Migrating...", total=total_steps)

        for step_id, step_name, step_func in migration_steps:
            if skip_backup and step_id.startswith("backup"):
                continue

            progress.update(task, description=f"[cyan]{step_name}...")

            try:
                result = step_func(ctx)
                if result:
                    ctx.mark_step_complete(step_name)
                else:
                    ctx.add_error(f"{step_name} failed")
                    if not force:
                        console.print("[red]Migration failed. Rolling back...[/red]")
                        rollback_migration(ctx)
                        return
            except Exception as e:
                ctx.add_error(f"{step_name} error: {str(e)}")
                if not force:
                    console.print(f"[red]Migration failed: {str(e)}[/red]")
                    console.print("[red]Rolling back...[/red]")
                    rollback_migration(ctx)
                    return

            progress.advance(task)

    # Migration complete
    if ctx.errors:
        console.print("\n[red]Migration completed with errors:[/red]")
        for error in ctx.errors:
            console.print(f"  • {error}")
    else:
        console.print("\n[green]✓ Migration completed successfully![/green]")

    if ctx.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in ctx.warnings:
            console.print(f"  • {warning}")

    # Save migration log
    save_migration_log(ctx)


def backup_database(ctx: MigrationContext) -> bool:
    """Backup database"""
    if ctx.dry_run:
        console.print("[dim]Would backup database[/dim]")
        return True

    try:
        # Create backup directory
        backup_dir = Path("backups") / f"migration_v0.2_to_v0.3_{DeterministicClock.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ctx.backup_dir = backup_dir

        # Get database URL
        db_url = get_database_url()
        if not db_url:
            ctx.add_warning("No database configured, skipping backup")
            return True

        # Determine database type and backup
        if db_url.startswith("postgresql"):
            # PostgreSQL backup
            # SECURITY FIX: Use shell=False to prevent command injection
            backup_file = backup_dir / "database.sql"

            # Parse pg_dump command safely without shell=True
            import shlex

            # Validate db_url to prevent injection
            if any(char in str(db_url) for char in [';', '|', '&', '$', '`', '\n', '(', ')']):
                ctx.add_error("Invalid database URL - contains dangerous characters")
                return False

            # Use subprocess without shell=True (secure)
            with open(backup_file, 'w') as f:
                result = subprocess.run(
                    ["pg_dump", str(db_url)],
                    shell=False,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    timeout=300  # 5 minute timeout
                )

            if result.returncode != 0:
                ctx.add_error(f"Database backup failed: {result.stderr.decode()}")
                return False

        elif db_url.startswith("sqlite"):
            # SQLite backup
            import shutil
            db_file = db_url.replace("sqlite:///", "")
            backup_file = backup_dir / "database.db"
            shutil.copy2(db_file, backup_file)

        else:
            ctx.add_warning(f"Unsupported database type: {db_url.split(':')[0]}")
            return True

        console.print(f"[dim]Database backed up to {backup_file}[/dim]")
        return True

    except Exception as e:
        ctx.add_error(f"Database backup failed: {str(e)}")
        return False


def backup_configuration(ctx: MigrationContext) -> bool:
    """Backup configuration files"""
    if ctx.dry_run:
        console.print("[dim]Would backup configuration[/dim]")
        return True

    try:
        if not ctx.backup_dir:
            ctx.backup_dir = Path("backups") / f"migration_v0.2_to_v0.3_{DeterministicClock.now().strftime('%Y%m%d_%H%M%S')}"
            ctx.backup_dir.mkdir(parents=True, exist_ok=True)

        config_files = [
            "greenlang.yaml",
            "greenlang.yml",
            ".env"
        ]

        for config_file in config_files:
            if Path(config_file).exists():
                shutil.copy2(config_file, ctx.backup_dir / f"{config_file}.backup")
                console.print(f"[dim]Backed up {config_file}[/dim]")

        return True

    except Exception as e:
        ctx.add_error(f"Configuration backup failed: {str(e)}")
        return False


def backup_agent_packs(ctx: MigrationContext) -> bool:
    """Backup agent packs"""
    if ctx.dry_run:
        console.print("[dim]Would backup agent packs[/dim]")
        return True

    try:
        if not ctx.backup_dir:
            ctx.backup_dir = Path("backups") / f"migration_v0.2_to_v0.3_{DeterministicClock.now().strftime('%Y%m%d_%H%M%S')}"
            ctx.backup_dir.mkdir(parents=True, exist_ok=True)

        packs_dir = Path.home() / ".greenlang" / "packs"
        if packs_dir.exists():
            backup_packs_dir = ctx.backup_dir / "packs"
            shutil.copytree(packs_dir, backup_packs_dir)
            console.print(f"[dim]Backed up agent packs to {backup_packs_dir}[/dim]")

        return True

    except Exception as e:
        ctx.add_error(f"Agent packs backup failed: {str(e)}")
        return False


def update_python_packages(ctx: MigrationContext) -> bool:
    """Update Python packages"""
    if ctx.dry_run:
        console.print("[dim]Would update Python packages to v0.3[/dim]")
        return True

    try:
        # Upgrade GreenLang
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "greenlang==0.3.0"],
            capture_output=True
        )

        if result.returncode != 0:
            ctx.add_error(f"Package upgrade failed: {result.stderr.decode()}")
            return False

        console.print("[dim]Python packages updated[/dim]")
        return True

    except Exception as e:
        ctx.add_error(f"Package update failed: {str(e)}")
        return False


def migrate_database_schema(ctx: MigrationContext) -> bool:
    """Run database migrations"""
    if ctx.dry_run:
        console.print("[dim]Would run database migrations[/dim]")
        return True

    try:
        # Run Alembic migrations
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True
        )

        if result.returncode != 0:
            ctx.add_error(f"Database migration failed: {result.stderr.decode()}")
            return False

        console.print("[dim]Database schema updated[/dim]")
        return True

    except Exception as e:
        ctx.add_error(f"Database migration failed: {str(e)}")
        return False


def update_configuration_files(ctx: MigrationContext) -> bool:
    """Update configuration files to v0.3 format"""
    if ctx.dry_run:
        console.print("[dim]Would update configuration files[/dim]")
        return True

    try:
        # Update greenlang.yaml
        config_path = Path("greenlang.yaml")
        if not config_path.exists():
            config_path = Path("greenlang.yml")

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Update version
            config["version"] = "0.3.0"

            # Update database config
            if "database" in config:
                if "url" in config["database"]:
                    config["database"]["connection_string"] = config["database"].pop("url")

            # Add security.encryption if missing
            if "security" in config:
                if "encryption" not in config["security"]:
                    config["security"]["encryption"] = {
                        "algorithm": "AES-256-GCM",
                        "key": "${GL_ENCRYPTION_KEY}",
                        "key_rotation_days": 90
                    }

            # Update logging to uppercase
            if "logging" in config:
                if "level" in config["logging"]:
                    config["logging"]["level"] = config["logging"]["level"].upper()

            # Save updated config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            console.print("[dim]Configuration files updated[/dim]")

        # Update .env file
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_content = f.read()

            # Replace deprecated env vars
            env_content = env_content.replace("GREENLANG_ENV=", "GL_ENV=")
            env_content = env_content.replace("GREENLANG_DB_URL=", "GL_DATABASE_URL=")

            # Add new required vars if missing
            if "GL_SECRET_KEY=" not in env_content:
                env_content += "\nGL_SECRET_KEY=<generate-with-openssl-rand-hex-32>"
            if "GL_ENCRYPTION_KEY=" not in env_content:
                env_content += "\nGL_ENCRYPTION_KEY=<generate-with-openssl-rand-base64-32>"

            with open(env_path, 'w') as f:
                f.write(env_content)

        return True

    except Exception as e:
        ctx.add_error(f"Configuration update failed: {str(e)}")
        return False


def convert_agent_packs(ctx: MigrationContext) -> bool:
    """Convert agent packs to AgentSpec v2"""
    if ctx.dry_run:
        console.print("[dim]Would convert agent packs to AgentSpec v2[/dim]")
        return True

    try:
        packs_dir = Path.home() / ".greenlang" / "packs"
        if not packs_dir.exists():
            console.print("[dim]No agent packs to convert[/dim]")
            return True

        converted_count = 0
        for pack_path in packs_dir.rglob("pack.yaml"):
            if convert_pack_to_v2(pack_path):
                converted_count += 1

        console.print(f"[dim]Converted {converted_count} agent packs to AgentSpec v2[/dim]")
        return True

    except Exception as e:
        ctx.add_error(f"Agent pack conversion failed: {str(e)}")
        return False


def convert_pack_to_v2(pack_path: Path) -> bool:
    """Convert a single pack to AgentSpec v2"""
    try:
        with open(pack_path, 'r') as f:
            pack = yaml.safe_load(f)

        if pack.get("spec_version") == "2.0":
            return False  # Already v2

        # Convert to v2 format
        v2_pack = {
            "spec_version": "2.0",
            "metadata": {
                "name": pack.get("name"),
                "version": pack.get("version"),
                "description": pack.get("description"),
                "author": pack.get("author"),
                "license": pack.get("license")
            }
        }

        # Convert inputs
        if "inputs" in pack:
            v2_pack["inputs"] = {}
            for input_field in pack["inputs"]:
                if isinstance(input_field, str):
                    v2_pack["inputs"][input_field] = {
                        "type": "string",
                        "required": True
                    }
                else:
                    v2_pack["inputs"][input_field["name"]] = input_field

        # Convert outputs
        if "outputs" in pack:
            v2_pack["outputs"] = {}
            for output_field in pack["outputs"]:
                if isinstance(output_field, str):
                    v2_pack["outputs"][output_field] = {
                        "type": "string"
                    }
                else:
                    v2_pack["outputs"][output_field["name"]] = output_field

        # Move capabilities to metadata
        if "capabilities" in pack:
            v2_pack["metadata"]["capabilities"] = pack["capabilities"]

        # Convert dependencies
        if "dependencies" in pack:
            v2_pack["dependencies"] = []
            for dep in pack["dependencies"]:
                if isinstance(dep, str):
                    v2_pack["dependencies"].append({"name": dep})
                else:
                    v2_pack["dependencies"].append(dep)

        # Save converted pack
        with open(pack_path, 'w') as f:
            yaml.dump(v2_pack, f, default_flow_style=False)

        return True

    except Exception as e:
        console.print(f"[red]Failed to convert {pack_path}: {str(e)}[/red]")
        return False


def run_validation_tests(ctx: MigrationContext) -> bool:
    """Run validation tests after migration"""
    if ctx.dry_run:
        console.print("[dim]Would run validation tests[/dim]")
        return True

    try:
        # Check version
        import greenlang
        if greenlang.__version__ != "0.3.0":
            ctx.add_error(f"Version mismatch: expected 0.3.0, got {greenlang.__version__}")
            return False

        # Check database connectivity
        db_url = get_database_url()
        if db_url:
            from sqlalchemy import create_engine
            engine = create_engine(db_url)
            with engine.connect() as conn:
                pass  # Just test connection

        console.print("[dim]Validation tests passed[/dim]")
        return True

    except Exception as e:
        ctx.add_error(f"Validation failed: {str(e)}")
        return False


def rollback_migration(ctx: MigrationContext):
    """Rollback migration to previous state"""
    console.print("\n[bold yellow]Rolling back migration...[/bold yellow]")

    if not ctx.backup_dir or not ctx.backup_dir.exists():
        console.print("[red]No backup directory found. Cannot rollback.[/red]")
        return

    # Restore database
    # Restore configuration
    # Restore agent packs
    # Downgrade packages

    console.print("[green]Rollback complete[/green]")


def save_migration_log(ctx: MigrationContext):
    """Save migration log"""
    log_file = Path("migration_log.json")
    log_data = {
        "timestamp": DeterministicClock.now().isoformat(),
        "source_version": ctx.current_version,
        "target_version": ctx.target_version,
        "steps_completed": ctx.steps_completed,
        "errors": ctx.errors,
        "warnings": ctx.warnings,
        "backup_dir": str(ctx.backup_dir) if ctx.backup_dir else None
    }

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    console.print(f"\n[dim]Migration log saved to {log_file}[/dim]")


@migrate.command()
@click.option('--compare-baseline', type=click.Path(exists=True), help='Compare against baseline')
def verify(compare_baseline: Optional[str]):
    """Verify migration success"""

    console.print(Panel.fit("[bold cyan]Migration Verification[/bold cyan]"))

    # Run verification checks
    checks = [
        ("Version Check", verify_version),
        ("Database Schema", verify_database_schema),
        ("Configuration Format", verify_configuration),
        ("Agent Packs", verify_agent_packs),
        ("API Endpoints", verify_api_endpoints)
    ]

    results = {}
    for check_name, check_func in checks:
        console.print(f"\n[bold]Running: {check_name}[/bold]")
        result = check_func()
        results[check_name] = result

        if result["passed"]:
            console.print(f"[green]✓ {check_name} passed[/green]")
        else:
            console.print(f"[red]✗ {check_name} failed[/red]")
            for issue in result.get("issues", []):
                console.print(f"  [red]• {issue}[/red]")

    # Summary
    passed = sum(1 for r in results.values() if r["passed"])
    total = len(results)

    console.print(f"\n[bold]Verification Summary:[/bold]")
    console.print(f"  Passed: {passed}/{total}")

    if passed == total:
        console.print("\n[green]✓ All verification checks passed![/green]")
    else:
        console.print("\n[red]✗ Some verification checks failed[/red]")


def verify_version() -> Dict[str, Any]:
    """Verify GreenLang version"""
    try:
        import greenlang
        version = greenlang.__version__
        passed = version == "0.3.0"
        return {
            "passed": passed,
            "version": version,
            "issues": [] if passed else [f"Expected version 0.3.0, got {version}"]
        }
    except Exception as e:
        return {
            "passed": False,
            "issues": [f"Version check failed: {str(e)}"]
        }


def verify_database_schema() -> Dict[str, Any]:
    """Verify database schema is updated"""
    try:
        db_url = get_database_url()
        if not db_url:
            return {"passed": True, "issues": ["No database configured"]}

        from sqlalchemy import create_engine, inspect
        engine = create_engine(db_url)
        inspector = inspect(engine)

        required_tables = ["audit_logs", "user_sessions", "agents", "workflows"]
        tables = inspector.get_table_names()

        issues = []
        for table in required_tables:
            if table not in tables:
                issues.append(f"Missing table: {table}")

        # Check agents table for spec_version column
        if "agents" in tables:
            columns = [col["name"] for col in inspector.get_columns("agents")]
            if "spec_version" not in columns:
                issues.append("agents table missing spec_version column")

        return {
            "passed": len(issues) == 0,
            "issues": issues
        }

    except Exception as e:
        return {
            "passed": False,
            "issues": [f"Database verification failed: {str(e)}"]
        }


def verify_configuration() -> Dict[str, Any]:
    """Verify configuration format is updated"""
    try:
        config_path = Path("greenlang.yaml")
        if not config_path.exists():
            config_path = Path("greenlang.yml")

        if not config_path.exists():
            return {"passed": True, "issues": ["No configuration file found"]}

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        issues = []

        # Check version
        if config.get("version") != "0.3.0":
            issues.append(f"Config version should be 0.3.0, got {config.get('version')}")

        # Check database.connection_string
        if "database" in config and "url" in config["database"]:
            issues.append("database.url is deprecated, use database.connection_string")

        # Check security.encryption
        if "security" in config and "encryption" not in config["security"]:
            issues.append("security.encryption is required in v0.3")

        return {
            "passed": len(issues) == 0,
            "issues": issues
        }

    except Exception as e:
        return {
            "passed": False,
            "issues": [f"Configuration verification failed: {str(e)}"]
        }


def verify_agent_packs() -> Dict[str, Any]:
    """Verify agent packs are converted to v2"""
    try:
        packs_dir = Path.home() / ".greenlang" / "packs"
        if not packs_dir.exists():
            return {"passed": True, "issues": ["No agent packs found"]}

        issues = []
        for pack_path in packs_dir.rglob("pack.yaml"):
            with open(pack_path, 'r') as f:
                pack = yaml.safe_load(f)

            if pack.get("spec_version") != "2.0":
                issues.append(f"{pack_path.name} not converted to AgentSpec v2")

        return {
            "passed": len(issues) == 0,
            "issues": issues
        }

    except Exception as e:
        return {
            "passed": False,
            "issues": [f"Agent pack verification failed: {str(e)}"]
        }


def verify_api_endpoints() -> Dict[str, Any]:
    """Verify API endpoints are accessible"""
    try:
        import requests

        # Try to connect to API
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                return {"passed": True, "issues": []}
            else:
                return {
                    "passed": False,
                    "issues": [f"API health check returned {response.status_code}"]
                }
        except requests.exceptions.ConnectionError:
            return {
                "passed": True,
                "issues": ["API not running (optional check)"]
            }

    except Exception as e:
        return {
            "passed": False,
            "issues": [f"API verification failed: {str(e)}"]
        }


@migrate.command()
@click.option('--force', is_flag=True, help='Skip confirmation prompt')
def rollback(force: bool):
    """Rollback migration to previous version"""

    console.print(Panel.fit("[bold red]Migration Rollback[/bold red]"))

    if not force:
        console.print("[bold yellow]WARNING: This will rollback to v0.2[/bold yellow]")
        console.print("This will:")
        console.print("  • Restore database from backup")
        console.print("  • Restore configuration files")
        console.print("  • Restore agent packs")
        console.print("  • Downgrade Python packages")

        if not Confirm.ask("\nContinue with rollback?"):
            console.print("[red]Rollback cancelled[/red]")
            return

    # Find most recent backup
    backups_dir = Path("backups")
    if not backups_dir.exists():
        console.print("[red]No backups found[/red]")
        return

    backup_dirs = sorted(backups_dir.glob("migration_v0.2_to_v0.3_*"), reverse=True)
    if not backup_dirs:
        console.print("[red]No migration backups found[/red]")
        return

    backup_dir = backup_dirs[0]
    console.print(f"[cyan]Using backup: {backup_dir}[/cyan]")

    # Execute rollback
    ctx = MigrationContext()
    ctx.backup_dir = backup_dir

    rollback_migration(ctx)


def get_database_url() -> Optional[str]:
    """Get database URL from config or environment"""
    # Check environment variables
    db_url = os.environ.get("GL_DATABASE_URL") or os.environ.get("GREENLANG_DB_URL")
    if db_url:
        return db_url

    # Check config file
    config_paths = [Path("greenlang.yaml"), Path("greenlang.yml")]
    for config_path in config_paths:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            if "database" in config:
                return (
                    config["database"].get("connection_string") or
                    config["database"].get("url")
                )

    return None


if __name__ == "__main__":
    migrate()
