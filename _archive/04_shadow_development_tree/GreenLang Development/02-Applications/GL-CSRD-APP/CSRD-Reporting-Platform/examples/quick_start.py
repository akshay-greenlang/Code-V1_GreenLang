# -*- coding: utf-8 -*-
"""
CSRD Platform - Quick Start Example

Get started with the CSRD Reporting Platform in 5 minutes!

This example demonstrates the simplest way to generate a complete CSRD report
using the one-function SDK API.

What this script does:
1. Loads demo ESG data and company profile
2. Configures the CSRD pipeline
3. Generates a complete CSRD report with all 6 agents
4. Saves outputs to the output directory

Processing time: ~2-5 minutes for demo data
Output: Complete CSRD report package with XBRL, PDF, and audit trail

Prerequisites:
- Python 3.11+
- All dependencies installed (pip install -r requirements.txt)
- OpenAI API key set in environment (for materiality assessment)

Usage:
    python examples/quick_start.py

    # With custom data
    python examples/quick_start.py --data your_data.csv --company your_company.json

Author: GreenLang CSRD Team
Version: 1.0.0
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdk.csrd_sdk import (
    csrd_build_report,
    CSRDConfig,
    CSRDReport
)

# Rich console for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for beautiful terminal output: pip install rich")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Get the base directory (CSRD-Reporting-Platform/)
BASE_DIR = Path(__file__).parent.parent

# Demo data files (included in the repository)
DEMO_ESG_DATA = BASE_DIR / "examples" / "demo_esg_data.csv"
DEMO_COMPANY_PROFILE = BASE_DIR / "examples" / "demo_company_profile.json"

# Output directory
OUTPUT_DIR = BASE_DIR / "output" / "quick_start_demo"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_banner():
    """Display welcome banner."""
    if RICH_AVAILABLE:
        console = Console()
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]CSRD Platform - Quick Start Demo[/bold cyan]\n"
            "[dim]Generate a complete CSRD report in 5 minutes[/dim]",
            border_style="cyan"
        ))
        console.print("\n")
    else:
        print("\n" + "=" * 80)
        print("CSRD PLATFORM - QUICK START DEMO")
        print("Generate a complete CSRD report in 5 minutes")
        print("=" * 80 + "\n")


def print_step(step_num: int, title: str, description: str):
    """Print a step in the process."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"\n[bold cyan]Step {step_num}:[/bold cyan] {title}")
        console.print(f"[dim]{description}[/dim]")
    else:
        print(f"\nStep {step_num}: {title}")
        print(f"{description}")


def print_success(message: str):
    """Print success message."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"[bold green]✓[/bold green] {message}")
    else:
        print(f"✓ {message}")


def print_warning(message: str):
    """Print warning message."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"[bold yellow]⚠[/bold yellow] {message}")
    else:
        print(f"⚠ {message}")


def print_error(message: str):
    """Print error message."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"[bold red]✗[/bold red] {message}")
    else:
        print(f"✗ {message}")


def display_report_summary(report: CSRDReport):
    """Display a beautiful summary of the generated report."""
    if RICH_AVAILABLE:
        console = Console()

        # Create summary table
        table = Table(
            title="CSRD Report Summary",
            show_header=True,
            header_style="bold cyan"
        )
        table.add_column("Category", style="dim", width=30)
        table.add_column("Result", justify="left", width=40)

        # Company info
        table.add_row(
            "Company",
            report.company_info.get('legal_name', 'N/A')
        )
        table.add_row(
            "Reporting Year",
            str(report.reporting_period.get('year', 'N/A'))
        )

        # Materiality
        table.add_row("", "")  # Separator
        table.add_row(
            "[bold]Materiality Assessment[/bold]",
            ""
        )
        table.add_row(
            "Material Topics",
            f"{report.materiality.material_topics_count} topics identified"
        )
        table.add_row(
            "Material ESRS Standards",
            ", ".join(report.material_standards) if report.material_standards else "None"
        )
        table.add_row(
            "AI Confidence",
            f"{report.materiality.average_confidence:.0%}"
        )

        # Metrics
        table.add_row("", "")  # Separator
        table.add_row(
            "[bold]ESRS Metrics Calculated[/bold]",
            ""
        )
        table.add_row(
            "Total Metrics",
            str(report.metrics.total_metrics_calculated)
        )
        if report.metrics.total_ghg_emissions_tco2e:
            table.add_row(
                "Total GHG Emissions",
                f"{report.metrics.total_ghg_emissions_tco2e:,.2f} tCO2e"
            )
        if report.metrics.total_workforce:
            table.add_row(
                "Total Workforce",
                f"{report.metrics.total_workforce:,} employees"
            )
        table.add_row(
            "Zero Hallucination",
            "[green]100% Guaranteed ✓[/green]"
        )

        # Compliance
        table.add_row("", "")  # Separator
        table.add_row(
            "[bold]Compliance Status[/bold]",
            ""
        )
        status_color = "green" if report.is_compliant else "red"
        table.add_row(
            "Overall Status",
            f"[{status_color}]{report.compliance_status.compliance_status}[/{status_color}]"
        )
        table.add_row(
            "Rules Checked",
            f"{report.compliance_status.rules_passed}/{report.compliance_status.total_rules_checked} passed"
        )
        table.add_row(
            "Audit Ready",
            "[green]Yes ✓[/green]" if report.is_audit_ready else "[red]No ✗[/red]"
        )

        # Performance
        table.add_row("", "")  # Separator
        table.add_row(
            "Total Processing Time",
            f"{report.processing_time_total_minutes:.1f} minutes"
        )

        console.print("\n")
        console.print(table)
        console.print("\n")

    else:
        # Plain text summary
        print("\n" + "=" * 80)
        print("CSRD REPORT SUMMARY")
        print("=" * 80)
        print(f"Company: {report.company_info.get('legal_name', 'N/A')}")
        print(f"Reporting Year: {report.reporting_period.get('year', 'N/A')}")
        print(f"\nMaterial Topics: {report.materiality.material_topics_count}")
        print(f"ESRS Standards: {', '.join(report.material_standards) if report.material_standards else 'None'}")
        print(f"Metrics Calculated: {report.metrics.total_metrics_calculated}")
        print(f"Compliance Status: {report.compliance_status.compliance_status}")
        print(f"Processing Time: {report.processing_time_total_minutes:.1f} minutes")
        print("=" * 80 + "\n")


# ============================================================================
# MAIN QUICK START FUNCTION
# ============================================================================

def run_quick_start(
    esg_data_path: str = None,
    company_profile_path: str = None,
    output_dir: str = None,
    skip_materiality: bool = False
):
    """
    Run the quick start demo.

    Args:
        esg_data_path: Path to ESG data file (default: demo data)
        company_profile_path: Path to company profile (default: demo profile)
        output_dir: Output directory (default: output/quick_start_demo)
        skip_materiality: Skip materiality assessment (faster, for testing)
    """
    print_banner()

    # Use defaults if not provided
    esg_data = esg_data_path or str(DEMO_ESG_DATA)
    company_profile = company_profile_path or str(DEMO_COMPANY_PROFILE)
    output = output_dir or str(OUTPUT_DIR)

    # ========================================================================
    # STEP 1: Verify Input Files
    # ========================================================================
    print_step(
        1,
        "Verify Input Files",
        "Checking that demo data files exist and are accessible"
    )

    if not Path(esg_data).exists():
        print_error(f"ESG data file not found: {esg_data}")
        print(f"Please ensure the file exists or run with --data <your_file.csv>")
        return False

    if not Path(company_profile).exists():
        print_error(f"Company profile not found: {company_profile}")
        print(f"Please ensure the file exists or run with --company <your_file.json>")
        return False

    print_success(f"ESG data: {Path(esg_data).name}")
    print_success(f"Company profile: {Path(company_profile).name}")

    # ========================================================================
    # STEP 2: Create Configuration
    # ========================================================================
    print_step(
        2,
        "Create Configuration",
        "Setting up CSRD configuration with company information"
    )

    # Load company profile to extract info
    with open(company_profile, 'r') as f:
        company_data = json.load(f)

    # Create CSRD configuration
    config = CSRDConfig(
        company_name=company_data.get('legal_name', 'Demo Company'),
        company_lei=company_data.get('lei_code', '549300ABC123DEF456GH'),
        reporting_year=company_data.get('reporting_period', {}).get('fiscal_year', 2024),
        sector=company_data.get('sector', {}).get('industry', 'Manufacturing'),
        country=company_data.get('country', 'NL'),
        employee_count=company_data.get('company_size', {}).get('employee_count'),
        revenue=company_data.get('company_size', {}).get('revenue_eur'),

        # LLM configuration (for materiality assessment)
        llm_provider="openai",
        llm_model="gpt-4o",
        llm_api_key=os.getenv("OPENAI_API_KEY"),  # Set via environment variable

        # Quality thresholds
        quality_threshold=0.80,
        impact_materiality_threshold=5.0,
        financial_materiality_threshold=5.0
    )

    print_success(f"Company: {config.company_name}")
    print_success(f"Reporting Year: {config.reporting_year}")
    print_success(f"Sector: {config.sector}")

    # Check for API key if not skipping materiality
    if not skip_materiality and not config.llm_api_key:
        print_warning("OPENAI_API_KEY not set in environment")
        print("  Materiality assessment will be skipped")
        print("  To enable: export OPENAI_API_KEY='your-key-here'")
        skip_materiality = True

    # ========================================================================
    # STEP 3: Generate CSRD Report
    # ========================================================================
    print_step(
        3,
        "Generate CSRD Report",
        "Running the complete 6-agent pipeline (this may take a few minutes)"
    )

    print("\nExecuting pipeline agents:")
    print("  1. IntakeAgent - Data validation and enrichment")
    print("  2. MaterialityAgent - Double materiality assessment (AI-powered)")
    print("  3. CalculatorAgent - ESRS metrics calculation (ZERO HALLUCINATION)")
    print("  4. AggregatorAgent - Multi-framework integration")
    print("  5. ReportingAgent - XBRL/ESEF report generation")
    print("  6. AuditAgent - Compliance validation (200+ rules)")
    print()

    try:
        # Generate report using the one-function API
        report = csrd_build_report(
            esg_data=esg_data,
            company_profile=company_profile,
            config=config,
            output_dir=output,
            skip_materiality=skip_materiality,
            skip_audit=False,  # Always run audit
            verbose=True  # Show progress
        )

        print_success("Report generation complete!")

    except Exception as e:
        print_error(f"Report generation failed: {str(e)}")
        print(f"\nError details: {e}")
        return False

    # ========================================================================
    # STEP 4: Display Results
    # ========================================================================
    print_step(
        4,
        "Review Results",
        "Summary of the generated CSRD report"
    )

    display_report_summary(report)

    # Show warnings if any
    if report.warnings:
        print("\n[bold yellow]Warnings:[/bold yellow]" if RICH_AVAILABLE else "\nWarnings:")
        for warning in report.warnings:
            print_warning(warning)

    # ========================================================================
    # STEP 5: Access Output Files
    # ========================================================================
    print_step(
        5,
        "Access Output Files",
        f"All outputs have been saved to: {output}"
    )

    output_path = Path(output)
    if output_path.exists():
        print("\nGenerated files:")
        for file in sorted(output_path.glob("*.json")):
            size = file.stat().st_size
            size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
            print(f"  • {file.name} ({size_str})")

        for file in sorted(output_path.glob("*.md")):
            print(f"  • {file.name}")

    print_success(f"\nAll outputs saved to: {output_path.absolute()}")

    # ========================================================================
    # STEP 6: Next Steps
    # ========================================================================
    if RICH_AVAILABLE:
        console = Console()
        console.print("\n")
        console.print(Panel.fit(
            "[bold green]Quick Start Complete![/bold green]\n\n"
            "[bold]Next Steps:[/bold]\n"
            "1. Review the summary report: [cyan]output/quick_start_demo/00_summary.md[/cyan]\n"
            "2. Check compliance status: [cyan]output/quick_start_demo/06_audit_compliance.json[/cyan]\n"
            "3. Explore the complete report: [cyan]output/quick_start_demo/00_complete_report.json[/cyan]\n"
            "4. Try the full pipeline example: [cyan]python examples/full_pipeline_example.py[/cyan]\n"
            "5. Read the user guide: [cyan]docs/USER_GUIDE.md[/cyan]",
            border_style="green"
        ))
        console.print("\n")
    else:
        print("\n" + "=" * 80)
        print("QUICK START COMPLETE!")
        print("=" * 80)
        print("\nNext Steps:")
        print("1. Review the summary report: output/quick_start_demo/00_summary.md")
        print("2. Check compliance status: output/quick_start_demo/06_audit_compliance.json")
        print("3. Explore the complete report: output/quick_start_demo/00_complete_report.json")
        print("4. Try the full pipeline example: python examples/full_pipeline_example.py")
        print("5. Read the user guide: docs/USER_GUIDE.md")
        print("=" * 80 + "\n")

    return True


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for quick start demo."""
    parser = argparse.ArgumentParser(
        description="CSRD Platform Quick Start - Generate a CSRD report in 5 minutes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with demo data
  python examples/quick_start.py

  # Run with your own data
  python examples/quick_start.py --data your_esg_data.csv --company your_company.json

  # Skip materiality assessment (faster, for testing)
  python examples/quick_start.py --skip-materiality

  # Custom output directory
  python examples/quick_start.py --output output/my_report

For more information, see docs/USER_GUIDE.md
        """
    )

    parser.add_argument(
        '--data',
        type=str,
        help='Path to ESG data file (CSV/JSON/Excel) [default: demo data]'
    )

    parser.add_argument(
        '--company',
        type=str,
        help='Path to company profile JSON [default: demo profile]'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output directory [default: output/quick_start_demo]'
    )

    parser.add_argument(
        '--skip-materiality',
        action='store_true',
        help='Skip materiality assessment (faster, for testing)'
    )

    args = parser.parse_args()

    # Run quick start
    success = run_quick_start(
        esg_data_path=args.data,
        company_profile_path=args.company,
        output_dir=args.output,
        skip_materiality=args.skip_materiality
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
