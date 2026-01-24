"""
CBAM Pack CLI

Command-line interface for running the CBAM Compliance Pack.
Implements: gl run cbam --config <config.yaml> --imports <imports.xlsx> --out <output_dir>
"""

import sys
from pathlib import Path

import click

from cbam_pack import __version__
from cbam_pack.pipeline import CBAMPipeline, PipelineResult


@click.group()
@click.version_option(version=__version__, prog_name="GreenLang CBAM Pack")
def cli():
    """GreenLang CBAM Compliance Pack - EU Carbon Border Adjustment Mechanism reporting."""
    pass


@cli.command("run")
@click.argument("pack", default="cbam")
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to CBAM config YAML file",
)
@click.option(
    "--imports", "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to import ledger XLSX/CSV file",
)
@click.option(
    "--out", "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for generated artifacts",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate without producing final outputs",
)
def run_pack(
    pack: str,
    config: Path,
    imports: Path,
    out: Path,
    verbose: bool,
    dry_run: bool,
):
    """
    Run the CBAM Compliance Pack to generate a quarterly report.

    \b
    Examples:
        gl-cbam run cbam -c cbam.yaml -i imports.xlsx -o ./q1_2025/
        gl-cbam run cbam --config config.yaml --imports data.csv --out ./output/ --verbose
    """
    if pack.lower() != "cbam":
        click.echo(f"Error: Unknown pack '{pack}'. Only 'cbam' is supported.", err=True)
        sys.exit(1)

    click.echo(f"GreenLang CBAM Pack v{__version__}")
    click.echo("=" * 50)

    if dry_run:
        click.echo("Running in DRY-RUN mode (validation only)")
        click.echo()

    # Create and run pipeline
    pipeline = CBAMPipeline(
        config_path=config,
        imports_path=imports,
        output_dir=out,
        verbose=verbose,
        dry_run=dry_run,
    )

    try:
        result = pipeline.run()
    except Exception as e:
        click.echo(f"\nUnexpected error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Display result
    click.echo()
    if result.success:
        click.echo(click.style("SUCCESS", fg="green", bold=True))
        click.echo(f"  Lines processed: {result.statistics.get('total_lines', 0)}")
        click.echo(f"  Total emissions: {result.statistics.get('total_emissions_tco2e', 0):.2f} tCO2e")
        click.echo(f"  Default factor usage: {result.statistics.get('default_usage_percent', 0):.1f}%")
        click.echo()
        click.echo(f"Output directory: {out}")
        click.echo("Generated artifacts:")
        for artifact in result.artifacts:
            click.echo(f"  - {artifact}")
        sys.exit(0)
    else:
        click.echo(click.style("FAILED", fg="red", bold=True))
        click.echo()
        for error in result.errors:
            click.echo(error, err=True)
            click.echo()
        sys.exit(result.exit_code)


@cli.command("validate")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to CBAM config YAML file to validate",
)
@click.option(
    "--imports", "-i",
    type=click.Path(exists=True, path_type=Path),
    help="Path to import ledger XLSX/CSV file to validate",
)
def validate(config: Path | None, imports: Path | None):
    """
    Validate input files without running the full pipeline.

    \b
    Examples:
        gl-cbam validate -c cbam.yaml
        gl-cbam validate -i imports.xlsx
        gl-cbam validate -c cbam.yaml -i imports.csv
    """
    from cbam_pack.validators import InputValidator

    if not config and not imports:
        click.echo("Error: Provide at least one of --config or --imports", err=True)
        sys.exit(1)

    validator = InputValidator(fail_fast=False)
    has_errors = False

    if config:
        click.echo(f"Validating config: {config}")
        result = validator.validate_config(config)
        if result.is_valid:
            click.echo(click.style("  Config: VALID", fg="green"))
        else:
            click.echo(click.style("  Config: INVALID", fg="red"))
            for error in result.errors:
                click.echo(f"    {error.format_error()}")
            has_errors = True

    if imports:
        click.echo(f"Validating imports: {imports}")
        result = validator.validate_imports(imports)
        if result.is_valid:
            click.echo(click.style(f"  Imports: VALID ({len(result.validated_lines)} lines)", fg="green"))
        else:
            click.echo(click.style("  Imports: INVALID", fg="red"))
            for error in result.errors:
                click.echo(f"    {error.format_error()}")
            has_errors = True

    sys.exit(1 if has_errors else 0)


@cli.command("version")
def show_version():
    """Show version information."""
    click.echo(f"GreenLang CBAM Pack v{__version__}")
    click.echo("License: Apache-2.0")
    click.echo("https://greenlang.in")


@cli.command("web")
@click.option(
    "--port", "-p",
    default=8000,
    type=int,
    help="Port to run the web server on (default: 8000)",
)
@click.option(
    "--host", "-h",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload for development",
)
def run_web(port: int, host: str, reload: bool):
    """
    Start the CBAM Pack web interface.

    \b
    Examples:
        gl-cbam web
        gl-cbam web --port 8080
        gl-cbam web --host 0.0.0.0 --port 8000
    """
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: Web dependencies not installed.", err=True)
        click.echo("Install with: pip install greenlang-cbam-pack[web]", err=True)
        sys.exit(1)

    click.echo(f"GreenLang CBAM Pack v{__version__} - Web Interface")
    click.echo("=" * 50)
    click.echo(f"Starting server at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")
    click.echo()

    uvicorn.run(
        "cbam_pack.web.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
