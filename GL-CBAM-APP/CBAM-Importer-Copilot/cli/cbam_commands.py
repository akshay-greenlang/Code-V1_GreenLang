"""
CBAM Importer Copilot - GreenLang CLI Commands

Provides CLI commands for CBAM reporting integrated with GreenLang.

Commands:
    gl cbam report  - Generate CBAM Transitional Registry report
    gl cbam config  - Manage CBAM configuration
    gl cbam validate - Validate shipment data

Version: 1.0.0
Author: GreenLang CBAM Team
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Rich console for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback to standard print
    def rprint(*args, **kwargs):
        print(*args)

# Import pipeline components
import os
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cbam_pipeline import CBAMPipeline


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load CBAM configuration from file.

    Search order:
    1. Provided config_path
    2. .cbam.yaml in current directory
    3. config/cbam_config.yaml in pack directory
    4. Environment variables

    Args:
        config_path: Optional path to config file

    Returns:
        Configuration dictionary
    """
    import yaml

    config = {}

    # Try provided path first
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        return config

    # Try .cbam.yaml in current directory
    local_config = Path(".cbam.yaml")
    if local_config.exists():
        with open(local_config) as f:
            config = yaml.safe_load(f) or {}
        return config

    # Try default config in pack directory
    default_config = Path(__file__).parent.parent / "config" / "cbam_config.yaml"
    if default_config.exists():
        with open(default_config) as f:
            config = yaml.safe_load(f) or {}
        return config

    # Environment variables as fallback
    config = {
        "importer": {
            "name": os.getenv("CBAM_IMPORTER_NAME"),
            "country": os.getenv("CBAM_IMPORTER_COUNTRY"),
            "eori": os.getenv("CBAM_IMPORTER_EORI"),
        },
        "declarant": {
            "name": os.getenv("CBAM_DECLARANT_NAME"),
            "position": os.getenv("CBAM_DECLARANT_POSITION"),
        }
    }

    return config


def save_config(config: Dict[str, Any], config_path: str = ".cbam.yaml"):
    """
    Save CBAM configuration to file.

    Args:
        config: Configuration dictionary
        config_path: Path to save config (default: .cbam.yaml)
    """
    import yaml

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    if RICH_AVAILABLE:
        console = Console()
        console.print(f"[green]✓[/green] Configuration saved to {config_path}")
    else:
        print(f"✓ Configuration saved to {config_path}")


# ============================================================================
# CLI COMMAND: gl cbam report
# ============================================================================

def cbam_report(args: Optional[argparse.Namespace] = None):
    """
    Generate CBAM Transitional Registry report.

    Usage:
        gl cbam report --input shipments.csv --output report.json

    Args:
        args: Command-line arguments (optional, for testing)
    """
    if RICH_AVAILABLE:
        console = Console()

    # Parse arguments if not provided
    if args is None:
        parser = create_report_parser()
        args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config if hasattr(args, 'config') else None)
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[yellow]⚠[/yellow] Could not load config: {e}")
        config = {}

    # Merge command-line args with config (CLI takes precedence)
    importer_info = {
        "importer_name": args.importer_name or config.get("importer", {}).get("name"),
        "importer_country": args.importer_country or config.get("importer", {}).get("country"),
        "importer_eori": args.importer_eori or config.get("importer", {}).get("eori"),
        "declarant_name": args.declarant_name or config.get("declarant", {}).get("name"),
        "declarant_position": args.declarant_position or config.get("declarant", {}).get("position"),
    }

    # Validate required fields
    missing_fields = [k for k, v in importer_info.items() if not v]
    if missing_fields:
        if RICH_AVAILABLE:
            console.print(f"[red]✗[/red] Missing required fields: {', '.join(missing_fields)}")
            console.print("\n[yellow]Hint:[/yellow] Set these via:")
            console.print("  1. Command-line flags (--importer-name, etc.)")
            console.print("  2. Configuration file (.cbam.yaml)")
            console.print("  3. Run 'gl cbam config' to create a config file")
        else:
            print(f"✗ Missing required fields: {', '.join(missing_fields)}")
            print("\nHint: Set these via command-line flags or .cbam.yaml config file")
        sys.exit(1)

    # Display banner
    if RICH_AVAILABLE:
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]CBAM Importer Copilot[/bold cyan]\n"
            "[dim]Zero-Hallucination EU CBAM Compliance Reporting[/dim]",
            border_style="cyan"
        ))
        console.print("\n")
    else:
        print("\n" + "="*60)
        print("CBAM IMPORTER COPILOT")
        print("Zero-Hallucination EU CBAM Compliance Reporting")
        print("="*60 + "\n")

    # Initialize pipeline
    try:
        if RICH_AVAILABLE:
            console.print("[cyan]Initializing pipeline...[/cyan]")

        pipeline = CBAMPipeline(
            cn_codes_path=args.cn_codes,
            cbam_rules_path=args.rules,
            suppliers_path=args.suppliers
        )

        if RICH_AVAILABLE:
            console.print("[green]✓[/green] Pipeline initialized\n")
        else:
            print("✓ Pipeline initialized\n")

    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]✗[/red] Pipeline initialization failed: {e}")
        else:
            print(f"✗ Pipeline initialization failed: {e}")
        sys.exit(1)

    # Run pipeline with progress indicator
    try:
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Generating CBAM report...", total=None)

                report = pipeline.run(
                    input_file=args.input,
                    importer_info=importer_info,
                    output_report_path=args.output,
                    output_summary_path=args.summary,
                    intermediate_output_dir=args.intermediate
                )

                progress.update(task, completed=True)
        else:
            print("Generating CBAM report...")
            report = pipeline.run(
                input_file=args.input,
                importer_info=importer_info,
                output_report_path=args.output,
                output_summary_path=args.summary,
                intermediate_output_dir=args.intermediate
            )
            print("✓ Report generation complete")

    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n[red]✗[/red] Report generation failed: {e}")
        else:
            print(f"\n✗ Report generation failed: {e}")
        sys.exit(1)

    # Display results
    display_report_summary(report, console if RICH_AVAILABLE else None)

    # Success message
    if RICH_AVAILABLE:
        console.print("\n[bold green]✨ Report generation complete![/bold green]")
        if args.output:
            console.print(f"[dim]Report saved to: {args.output}[/dim]")
        if args.summary:
            console.print(f"[dim]Summary saved to: {args.summary}[/dim]")
    else:
        print("\n✨ Report generation complete!")
        if args.output:
            print(f"Report saved to: {args.output}")
        if args.summary:
            print(f"Summary saved to: {args.summary}")

    return report


def display_report_summary(report: Dict[str, Any], console: Optional[Any] = None):
    """
    Display a beautiful summary of the report.

    Args:
        report: CBAM report dictionary
        console: Rich console (optional)
    """
    metadata = report.get("report_metadata", {})
    goods = report.get("goods_summary", {})
    emissions = report.get("emissions_summary", {})
    validation = report.get("validation_results", {})

    if console:  # Rich output
        # Create summary table
        table = Table(title="CBAM Report Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Report ID", metadata.get("report_id", "N/A"))
        table.add_row("Quarter", metadata.get("quarter", "N/A"))
        table.add_row("Generated", metadata.get("generated_at", "N/A"))
        table.add_row("", "")  # Separator
        table.add_row("Total Shipments", str(goods.get("total_shipments", 0)))
        table.add_row("Total Mass (tonnes)", f"{goods.get('total_mass_tonnes', 0):.2f}")
        table.add_row("Total Emissions (tCO2)", f"{emissions.get('total_embedded_emissions_tco2', 0):.2f}")
        table.add_row("", "")  # Separator

        # Validation status
        is_valid = validation.get("is_valid", False)
        status_text = "[green]PASS ✅[/green]" if is_valid else "[red]FAIL ❌[/red]"
        table.add_row("Validation Status", status_text)

        if not is_valid:
            table.add_row("Errors", str(len(validation.get("errors", []))))

        console.print("\n")
        console.print(table)

    else:  # Plain output
        print("\n" + "="*60)
        print("CBAM REPORT SUMMARY")
        print("="*60)
        print(f"Report ID: {metadata.get('report_id', 'N/A')}")
        print(f"Quarter: {metadata.get('quarter', 'N/A')}")
        print(f"Generated: {metadata.get('generated_at', 'N/A')}")
        print(f"\nTotal Shipments: {goods.get('total_shipments', 0)}")
        print(f"Total Mass (tonnes): {goods.get('total_mass_tonnes', 0):.2f}")
        print(f"Total Emissions (tCO2): {emissions.get('total_embedded_emissions_tco2', 0):.2f}")
        print(f"\nValidation: {'PASS ✅' if validation.get('is_valid') else 'FAIL ❌'}")
        print("="*60 + "\n")


# ============================================================================
# CLI COMMAND: gl cbam config
# ============================================================================

def cbam_config(args: Optional[argparse.Namespace] = None):
    """
    Manage CBAM configuration.

    Usage:
        gl cbam config --init                    # Create new config
        gl cbam config --show                    # Show current config
        gl cbam config --set importer.name "..."  # Set config value
    """
    if RICH_AVAILABLE:
        console = Console()

    # Parse arguments if not provided
    if args is None:
        parser = create_config_parser()
        args = parser.parse_args()

    # Initialize new config
    if args.init:
        config = {
            "importer": {
                "name": input("EU Importer Legal Name: "),
                "country": input("EU Country Code (e.g., NL, DE, FR): "),
                "eori": input("EORI Number: "),
            },
            "declarant": {
                "name": input("Declarant Name: "),
                "position": input("Declarant Position: "),
            },
            "paths": {
                "cn_codes": "data/cn_codes.json",
                "rules": "rules/cbam_rules.yaml",
                "suppliers": "examples/demo_suppliers.yaml"
            }
        }

        save_config(config)

        if RICH_AVAILABLE:
            console.print("\n[green]✓[/green] Configuration created successfully!")
            console.print("[dim]Run 'gl cbam config --show' to view[/dim]")
        else:
            print("\n✓ Configuration created successfully!")

        return

    # Show current config
    if args.show:
        try:
            config = load_config()

            if RICH_AVAILABLE:
                console.print_json(data=config)
            else:
                print(json.dumps(config, indent=2))

        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]✗[/red] Could not load config: {e}")
            else:
                print(f"✗ Could not load config: {e}")

        return


# ============================================================================
# CLI COMMAND: gl cbam validate
# ============================================================================

def cbam_validate(args: Optional[argparse.Namespace] = None):
    """
    Validate shipment data without generating report.

    Usage:
        gl cbam validate --input shipments.csv
    """
    if RICH_AVAILABLE:
        console = Console()
        console.print("\n[cyan]Validating shipment data...[/cyan]\n")
    else:
        print("\nValidating shipment data...\n")

    # Parse arguments if not provided
    if args is None:
        parser = create_validate_parser()
        args = parser.parse_args()

    # Import validation agent
    from agents import ShipmentIntakeAgent

    # Initialize agent
    agent = ShipmentIntakeAgent(
        cn_codes_path=args.cn_codes,
        cbam_rules_path=args.rules,
        suppliers_path=args.suppliers
    )

    # Process and validate
    try:
        result = agent.process(args.input)

        # Display results
        metadata = result.get("metadata", {})

        if RICH_AVAILABLE:
            table = Table(title="Validation Results", show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="dim")
            table.add_column("Count", justify="right")

            table.add_row("Total Records", str(metadata.get("total_records", 0)))
            table.add_row("[green]Valid Records[/green]", str(metadata.get("valid_records", 0)))
            table.add_row("[red]Invalid Records[/red]", str(metadata.get("invalid_records", 0)))
            table.add_row("[yellow]Warnings[/yellow]", str(metadata.get("warnings", 0)))

            console.print(table)

            if metadata.get("invalid_records", 0) == 0:
                console.print("\n[bold green]✓ All shipments are valid![/bold green]")
            else:
                console.print("\n[bold yellow]⚠ Some shipments have errors[/bold yellow]")
                console.print("[dim]Check the error details for more information[/dim]")
        else:
            print("Validation Results:")
            print(f"Total Records: {metadata.get('total_records', 0)}")
            print(f"Valid Records: {metadata.get('valid_records', 0)}")
            print(f"Invalid Records: {metadata.get('invalid_records', 0)}")
            print(f"Warnings: {metadata.get('warnings', 0)}")

            if metadata.get("invalid_records", 0) == 0:
                print("\n✓ All shipments are valid!")
            else:
                print("\n⚠ Some shipments have errors")

    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]✗[/red] Validation failed: {e}")
        else:
            print(f"✗ Validation failed: {e}")
        sys.exit(1)


# ============================================================================
# ARGUMENT PARSERS
# ============================================================================

def create_report_parser() -> argparse.ArgumentParser:
    """Create argument parser for 'gl cbam report' command."""
    parser = argparse.ArgumentParser(
        prog="gl cbam report",
        description="Generate CBAM Transitional Registry report",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output
    parser.add_argument("--input", "-i", required=True, help="Input shipments file (CSV/JSON/Excel)")
    parser.add_argument("--output", "-o", help="Output report JSON path")
    parser.add_argument("--summary", "-s", help="Output summary Markdown path")
    parser.add_argument("--intermediate", help="Directory for intermediate outputs")

    # Configuration
    parser.add_argument("--config", "-c", help="Path to configuration file (.cbam.yaml)")

    # Reference data
    parser.add_argument("--cn-codes", default="data/cn_codes.json", help="Path to CN codes JSON")
    parser.add_argument("--rules", default="rules/cbam_rules.yaml", help="Path to CBAM rules YAML")
    parser.add_argument("--suppliers", default="examples/demo_suppliers.yaml", help="Path to suppliers YAML")

    # Importer information (can override config)
    parser.add_argument("--importer-name", help="EU importer legal name")
    parser.add_argument("--importer-country", help="EU country code (e.g., NL, DE, FR)")
    parser.add_argument("--importer-eori", help="EORI number")
    parser.add_argument("--declarant-name", help="Person making declaration")
    parser.add_argument("--declarant-position", help="Declarant position/title")

    return parser


def create_config_parser() -> argparse.ArgumentParser:
    """Create argument parser for 'gl cbam config' command."""
    parser = argparse.ArgumentParser(
        prog="gl cbam config",
        description="Manage CBAM configuration"
    )

    parser.add_argument("--init", action="store_true", help="Create new configuration file")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Set configuration value")

    return parser


def create_validate_parser() -> argparse.ArgumentParser:
    """Create argument parser for 'gl cbam validate' command."""
    parser = argparse.ArgumentParser(
        prog="gl cbam validate",
        description="Validate shipment data"
    )

    parser.add_argument("--input", "-i", required=True, help="Input shipments file (CSV/JSON/Excel)")
    parser.add_argument("--cn-codes", default="data/cn_codes.json", help="Path to CN codes JSON")
    parser.add_argument("--rules", default="rules/cbam_rules.yaml", help="Path to CBAM rules YAML")
    parser.add_argument("--suppliers", default="examples/demo_suppliers.yaml", help="Path to suppliers YAML")

    return parser


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main CLI entry point for direct execution."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: gl cbam <command> [options]")
        print("\nCommands:")
        print("  report    Generate CBAM Transitional Registry report")
        print("  config    Manage CBAM configuration")
        print("  validate  Validate shipment data")
        sys.exit(1)

    command = sys.argv[1]

    if command == "report":
        cbam_report()
    elif command == "config":
        cbam_config()
    elif command == "validate":
        cbam_validate()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
