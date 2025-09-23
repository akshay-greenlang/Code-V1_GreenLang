"""
GreenLang Unified CLI (gl)
==========================

A modern, fast CLI built with Typer for the GreenLang infrastructure platform.
All domain logic lives in packs - the CLI just orchestrates.
"""

import typer
from typing import Optional
import importlib.metadata as md

app = typer.Typer(
    name="gl",
    no_args_is_help=True,
    add_completion=True,
    help="GreenLang: Infrastructure for Climate Intelligence",
)

from . import cmd_init, cmd_run, cmd_pack, cmd_verify, cmd_policy, cmd_doctor

app.add_typer(cmd_init.app, name="init", help="Initialize new projects/packs")
app.add_typer(cmd_run.app, name="run", help="Execute pipelines and packs")
app.add_typer(cmd_pack.app, name="pack", help="Manage packs")
app.add_typer(cmd_verify.app, name="verify", help="Verify signatures & SBOM")
app.add_typer(cmd_policy.app, name="policy", help="Policy management")
app.add_typer(
    cmd_doctor.app,
    name="doctor",
    help="Environment diagnostics",
    invoke_without_command=True,
)


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version and exit"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output"),
    json: bool = typer.Option(False, "--json", help="Output in JSON format"),
    profile: Optional[str] = typer.Option(
        None, "--profile", help="Configuration profile"
    ),
    region: Optional[str] = typer.Option(None, "--region", help="Target region"),
):
    """
    GreenLang v0.1 - Infrastructure for Climate Intelligence

    Domain logic lives in packs. Platform = SDK/CLI/Runtime + Hub + Policy/Provenance

    Environment Variables:
        GL_PROFILE: Default profile
        GL_REGION: Default region
        GL_HUB: Hub URL
        GL_TELEMETRY: on|off
        GL_POLICY_BUNDLE: Policy bundle path
        GL_CACHE_DIR: Cache directory
    """
    if version:
        try:
            gl_version = md.version("greenlang")
        except:
            gl_version = "0.1.0"
        typer.echo(f"gl {gl_version}")
        raise typer.Exit()

    # Store options in context for use by subcommands
    # Note: Context is passed automatically to commands that need it


def main():
    """Main entry point for gl CLI"""
    app()


def deprecated_main():
    """Deprecated greenlang entry point - forwards to gl with warning"""
    import sys

    print(
        "WARNING: 'greenlang' is deprecated. Use 'gl' instead. Forwarding...",
        file=sys.stderr,
    )
    app()


if __name__ == "__main__":
    main()
