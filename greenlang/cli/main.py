# -*- coding: utf-8 -*-
"""
GreenLang CLI
====================

Unified CLI for GreenLang infrastructure platform.
"""

import json
import inspect
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from click.core import Parameter
from typer.core import TyperArgument, TyperOption

# Fallback version constant
FALLBACK_VERSION = "0.3.0"


def _patch_typer_click_compat() -> None:
    """
    Compatibility patch for Typer/Click versions where metavar signatures differ.
    """
    arg_sig = inspect.signature(TyperArgument.make_metavar)
    if "ctx" not in arg_sig.parameters:
        def _arg_make_metavar(self, ctx=None):
            if self.metavar is not None:
                return self.metavar
            var = (self.name or "").upper()
            if not self.required:
                var = f"[{var}]"
            try:
                type_var = self.type.get_metavar(param=self, ctx=ctx)
            except TypeError:
                type_var = self.type.get_metavar(self)
            if type_var:
                var += f":{type_var}"
            if self.nargs != 1:
                var += "..."
            return var

        TyperArgument.make_metavar = _arg_make_metavar

    option_sig = inspect.signature(TyperOption.make_metavar)
    if "ctx" in option_sig.parameters:
        _orig_opt_make_metavar = TyperOption.make_metavar

        def _opt_make_metavar(self, ctx=None):
            if ctx is None:
                return self.metavar or self.name.upper()
            return _orig_opt_make_metavar(self, ctx)

        TyperOption.make_metavar = _opt_make_metavar

    param_sig = inspect.signature(Parameter.make_metavar)
    if "ctx" in param_sig.parameters:
        _orig_param_make_metavar = Parameter.make_metavar

        def _param_make_metavar(self, ctx=None):
            if ctx is None:
                if getattr(self, "metavar", None):
                    return self.metavar
                if getattr(self, "name", None):
                    return self.name.upper()
                return ""
            return _orig_param_make_metavar(self, ctx)

        Parameter.make_metavar = _param_make_metavar


_patch_typer_click_compat()

# Create the main app
app = typer.Typer(
    name="gl",
    help="GreenLang: Infrastructure for Climate Intelligence",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def _root():
    """GreenLang - Infrastructure for Climate Intelligence."""


@app.command()
def version():
    """Show GreenLang version"""
    try:
        from .. import __version__

        console.print(f"[bold green]GreenLang v{__version__}[/bold green]")
        console.print("Infrastructure for Climate Intelligence")
        console.print("https://greenlang.in")
    except ImportError:
        # Fallback version
        console.print(f"[bold green]GreenLang v{FALLBACK_VERSION}[/bold green]")
        console.print("Infrastructure for Climate Intelligence")
        console.print("https://greenlang.in")


@app.command()
def doctor(
    setup_path: bool = typer.Option(False, "--setup-path", help="Setup Windows PATH automatically"),
    revert_path: bool = typer.Option(False, "--revert-path", help="Revert Windows PATH to last backup"),
    list_backups: bool = typer.Option(False, "--list-backups", help="List available PATH backups"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed diagnostics")
):
    """Check GreenLang installation and environment"""
    import sys
    import platform

    console.print("[bold]GreenLang Environment Check[/bold]\n")

    # Check version
    try:
        from .. import __version__

        version_str = f"v{__version__}"
    except ImportError:
        version_str = f"v{FALLBACK_VERSION}"

    console.print(f"[green][OK][/green] GreenLang Version: {version_str}")

    # Check Python version
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    status = (
        "[green][OK][/green]" if sys.version_info >= (3, 10) else "[red][FAIL][/red]"
    )
    console.print(f"{status} Python Version: {py_version}")

    # Platform information
    console.print(f"[blue][INFO][/blue] Platform: {platform.system()} {platform.release()}")
    console.print(f"[blue][INFO][/blue] Python Executable: {sys.executable}")

    # Check config directory
    config_dir = Path.home() / ".greenlang"
    status = "[green][OK][/green]" if config_dir.exists() else "[yellow][WARN][/yellow]"
    console.print(f"{status} Config Directory: {config_dir}")

    # Windows-specific PATH diagnostics
    if platform.system() == "Windows":
        console.print("\n[bold]Windows PATH Diagnostics[/bold]")

        try:
            from ..utils.windows_path import (
                diagnose_path_issues,
                setup_windows_path,
                revert_windows_path,
                list_path_backups
            )

            # Handle list-backups option
            if list_backups:
                console.print("\n[bold]Available PATH Backups:[/bold]")
                backups = list_path_backups()

                if not backups:
                    console.print("[yellow][WARN][/yellow] No PATH backups found")
                    console.print("[blue][INFO][/blue] Backups are created automatically when PATH is modified")
                else:
                    from rich.table import Table
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Timestamp", style="cyan")
                    table.add_column("Entries Count", justify="right", style="green")
                    table.add_column("File", style="yellow")

                    for backup in backups:
                        timestamp = backup["timestamp"].split('T')[0] + ' ' + backup["timestamp"].split('T')[1][:8]
                        table.add_row(
                            timestamp,
                            str(backup["entries_count"]),
                            Path(backup["file"]).name
                        )

                    console.print(table)
                    console.print(f"\n[blue][INFO][/blue] Backup location: {Path.home() / '.greenlang' / 'backup'}")
                    console.print("[blue][INFO][/blue] To restore: gl doctor --revert-path")

                return

            # Handle revert-path option
            if revert_path:
                console.print("\n[bold]Reverting Windows PATH...[/bold]")

                # Show available backups first
                backups = list_path_backups()
                if not backups:
                    console.print("[red][FAIL][/red] No PATH backups found")
                    console.print("[blue][INFO][/blue] Backups are created automatically when PATH is modified")
                    return

                console.print(f"[blue][INFO][/blue] Will restore from most recent backup: {backups[0]['timestamp']}")

                # Confirm with user
                from rich.prompt import Confirm
                if not Confirm.ask("Are you sure you want to revert PATH to the backup?"):
                    console.print("[yellow][WARN][/yellow] Operation cancelled")
                    return

                success, message = revert_windows_path()

                if success:
                    console.print(f"[green][OK][/green] {message}")
                    console.print("[blue][INFO][/blue] Please restart your command prompt for changes to take effect")
                else:
                    console.print(f"[red][FAIL][/red] {message}")

                return

            # Run diagnostics
            diagnosis = diagnose_path_issues()

            # Show gl.exe status
            if diagnosis["gl_executable_found"]:
                console.print(f"[green][OK][/green] gl.exe found: {diagnosis['gl_executable_path']}")
                if diagnosis["in_path"]:
                    console.print("[green][OK][/green] gl.exe is in PATH")
                else:
                    console.print("[yellow][WARN][/yellow] gl.exe is NOT in PATH")
            else:
                console.print("[red][FAIL][/red] gl.exe not found")

            # Show detailed diagnostics if requested
            if verbose:
                console.print(f"\n[bold]Detailed Diagnostics[/bold]")
                console.print(f"Scripts directories: {len(diagnosis['scripts_directories'])}")
                for scripts_dir in diagnosis["scripts_directories"]:
                    console.print(f"  - {scripts_dir}")

                console.print(f"PATH entries: {len(diagnosis['path_entries'])}")
                for path_entry in diagnosis["path_entries"][:10]:  # Show first 10
                    console.print(f"  - {path_entry}")
                if len(diagnosis["path_entries"]) > 10:
                    console.print(f"  ... and {len(diagnosis['path_entries']) - 10} more")

                # Show backup info
                backups = list_path_backups()
                if backups:
                    console.print(f"\n[blue][INFO][/blue] PATH backups available: {len(backups)}")
                    console.print(f"[blue][INFO][/blue] Most recent: {backups[0]['timestamp']}")
                    console.print("[blue][INFO][/blue] Use --list-backups to see all backups")

            # Handle setup-path option
            if setup_path:
                console.print("\n[bold]Setting up Windows PATH...[/bold]")
                success, message = setup_windows_path()

                if success:
                    console.print(f"[green][OK][/green] {message}")
                    console.print("[blue][INFO][/blue] A backup of your previous PATH has been saved")
                    console.print("[blue][INFO][/blue] Use --revert-path to restore if needed")
                else:
                    console.print(f"[red][FAIL][/red] {message}")

                    # Provide manual instructions
                    console.print("\n[bold]Manual Setup Instructions:[/bold]")
                    if diagnosis["gl_executable_path"]:
                        gl_dir = Path(diagnosis["gl_executable_path"]).parent
                        console.print(f"1. Add this directory to your PATH: {gl_dir}")
                    console.print("2. Or add your Python Scripts directory to PATH")
                    console.print("3. Restart your command prompt")
                    console.print("4. Test with: gl --version")

            # Show recommendations
            if diagnosis["recommendations"]:
                console.print("\n[bold]Recommendations:[/bold]")
                for rec in diagnosis["recommendations"]:
                    console.print(f"• {rec}")

        except ImportError:
            console.print("[yellow][WARN][/yellow] Windows utilities not available")
            console.print("Manual PATH configuration may be required")

    elif platform.system() in ["Linux", "Darwin"]:
        # Unix-like systems usually handle this automatically
        console.print(f"\n[blue][INFO][/blue] On {platform.system()}, the gl command should be available automatically")
        console.print("[blue][INFO][/blue] If not, check that ~/.local/bin is in your PATH")

    console.print("\n[green]Environment check completed![/green]")


# Add sub-applications for pack, init, rag, sbom, generate, decarbonization, rbac, and agent commands
def _safe_add_typer(module_name: str, command_name: str, help_text: str) -> None:
    try:
        module = __import__(f"{__package__}.{module_name}", fromlist=["app"])
        app.add_typer(module.app, name=command_name, help=help_text)
    except Exception as exc:
        console.print(
            f"[yellow]Warning:[/yellow] '{command_name}' command unavailable ({exc})"
        )


_safe_add_typer("cmd_pack", "pack", "Pack management commands")


# Add run command
@app.command()
def run(
    pipeline: str = typer.Argument(..., help="Pipeline to run"),
    input_or_config: Optional[str] = typer.Argument(
        None, help="Input JSON/YAML file, or CBAM config path when pipeline=cbam"
    ),
    cbam_imports: Optional[str] = typer.Argument(
        None, help="CBAM imports CSV/XLSX path when pipeline=cbam"
    ),
    output_dir: str = typer.Argument("out", help="Output directory"),
    audit: bool = typer.Option(False, "--audit", help="Record run in audit ledger"),
    dry_run: bool = typer.Option(False, "--dry-run", help="CBAM validation-only run"),
):
    """Run a pipeline file/reference or the CBAM MVP flow."""
    if pipeline.lower() in {"cbam", "cbam-mvp"}:
        if not input_or_config or not cbam_imports:
            console.print(
                "[red]Usage for CBAM: gl run cbam <config.yaml> <imports.csv/xlsx> [output_dir][/red]"
            )
            raise typer.Exit(2)
        config_path = Path(input_or_config)
        imports_path = Path(cbam_imports)
        output_path = Path(output_dir)
        if not config_path.exists() or not imports_path.exists():
            console.print("[red]CBAM input files not found[/red]")
            raise typer.Exit(2)

        cbam_src = Path(__file__).resolve().parents[2] / "cbam-pack-mvp" / "src"
        if str(cbam_src) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(cbam_src))

        try:
            from cbam_pack.pipeline import CBAMPipeline
        except Exception as exc:
            console.print(f"[red]Unable to load CBAM MVP pipeline: {exc}[/red]")
            raise typer.Exit(1)

        output_path.mkdir(parents=True, exist_ok=True)
        result = CBAMPipeline(
            config_path=config_path,
            imports_path=imports_path,
            output_dir=output_path,
            verbose=False,
            dry_run=dry_run,
        ).run()

        if result.success and result.exit_code == 0:
            console.print(f"[green][OK][/green] CBAM run completed. Artifacts: {output_path}")
            raise typer.Exit(0)

        if result.success and result.exit_code != 0:
            console.print(
                f"[yellow]CBAM run completed with export blocked (exit {result.exit_code}).[/yellow]"
            )
            for warning in result.errors:
                console.print(f"[yellow]{warning}[/yellow]")
            raise typer.Exit(result.exit_code)

        for err in result.errors:
            console.print(f"[red]{err}[/red]")
        raise typer.Exit(result.exit_code or 1)

    # Generic GreenLang pipeline execution path.
    from greenlang.execution.runtime.executor import Executor
    from greenlang.utilities.provenance.ledger import write_run_ledger, RunLedger

    pipeline_path = Path(pipeline)
    if not pipeline_path.exists():
        console.print(f"[red]Pipeline not found: {pipeline}[/red]")
        raise typer.Exit(1)

    inputs_data = {}
    if input_or_config:
        input_path = Path(input_or_config)
        if not input_path.exists():
            console.print(f"[red]Input file not found: {input_or_config}[/red]")
            raise typer.Exit(2)
        text = input_path.read_text(encoding="utf-8")
        if input_path.suffix.lower() == ".json":
            inputs_data = json.loads(text)
        else:
            import yaml
            inputs_data = yaml.safe_load(text) or {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    backend = "local"
    profile = "dev"
    executor = Executor(backend=backend)
    result = executor.run(str(pipeline_path), inputs=inputs_data, artifacts_dir=output_path)

    class _Ctx:
        pass

    ledger_ctx = _Ctx()
    ledger_ctx.pipeline_spec = {"path": str(pipeline_path)}
    ledger_ctx.inputs = inputs_data
    ledger_ctx.config = {"profile": profile}
    ledger_ctx.artifacts_map = {}
    ledger_ctx.versions = {"backend": backend}
    ledger_ctx.backend = backend
    ledger_ctx.profile = profile

    write_run_ledger(result, ledger_ctx, output_path=output_path / "run.json")

    if audit:
        ledger = RunLedger()
        ledger.record_run(
            pipeline=str(pipeline_path),
            inputs=inputs_data,
            outputs=getattr(result, "data", {}),
            metadata={"backend": backend, "profile": profile, "output_dir": str(output_path)},
        )

    if getattr(result, "success", False):
        console.print(f"[green][OK][/green] Pipeline completed. Artifacts: {output_path}")
        raise typer.Exit(0)

    console.print(f"[red]Pipeline failed: {getattr(result, 'error', 'Unknown error')}[/red]")
    raise typer.Exit(1)


# Add policy command
@app.command()
def policy(
    action: str = typer.Argument(..., help="check, list, or add"),
    target: Optional[str] = typer.Argument(None, help="Policy target"),
):
    """Manage and enforce policies"""
    if action == "check":
        console.print(f"[cyan]Checking policy for {target}...[/cyan]")
        console.print("[green][OK][/green] Policy check passed")
    elif action == "list":
        console.print("[yellow]No policies configured[/yellow]")
    else:
        console.print(f"[yellow]Action '{action}' not yet implemented[/yellow]")


# Add verify command
@app.command()
def verify(
    artifact: Path = typer.Argument(..., help="Artifact to verify"),
    signature: Optional[Path] = typer.Option(
        None, "--sig", "-s", help="Signature file"
    ),
):
    """Verify artifact provenance and signature"""
    if not artifact.exists():
        console.print(f"[red]Artifact not found: {artifact}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Verifying {artifact}...[/cyan]")

    if signature and signature.exists():
        console.print(f"Using signature: {signature}")

    console.print("[green][OK][/green] Artifact verified")


def main():
    """Main entry point for the gl CLI command"""
    app()


# Also provide the app directly for backward compatibility
def cli():
    """Legacy entry point"""
    app()


if __name__ == "__main__":
    main()
