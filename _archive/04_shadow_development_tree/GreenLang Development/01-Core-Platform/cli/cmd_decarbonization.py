# -*- coding: utf-8 -*-
"""
gl decarbonization - Generate comprehensive industrial decarbonization roadmaps

This command executes Agent #12: DecarbonizationRoadmapAgent_AI to create
strategic decarbonization plans with financial analysis, technology assessment,
and compliance roadmaps.
"""

import typer
import json
import yaml
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def decarbonization(
    ctx: typer.Context,
    facility_id: Optional[str] = typer.Option(
        None, "--facility-id", "-f", help="Unique facility identifier"
    ),
    facility_name: Optional[str] = typer.Option(
        None, "--facility-name", help="Facility name"
    ),
    industry_type: Optional[str] = typer.Option(
        None, "--industry", "-i", help="Industry type (Food & Beverage, Chemicals, etc.)"
    ),
    input_file: Optional[str] = typer.Option(
        None, "--input", "-I", help="Input data file (JSON/YAML)"
    ),
    output: str = typer.Option(
        "roadmap.json", "--output", "-o", help="Output file path"
    ),
    budget: float = typer.Option(
        2.0, "--budget", "-b", help="AI budget in USD (default: $2.00)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose output with detailed logs"
    ),
):
    """
    Generate comprehensive decarbonization roadmap for industrial facilities

    Examples:
        # From JSON input file
        gl decarbonization --input facility.json --output roadmap.json

        # With CLI parameters
        gl decarbonization --facility-id PLANT-001 \\
            --facility-name "Food Processing Plant" \\
            --industry "Food & Beverage" \\
            --input facility_data.json

        # Quick demo with sample data
        gl decarbonization demo

    The roadmap includes:
        - GHG inventory (Scope 1, 2, 3)
        - Technology assessment (solar, heat pumps, WHR, etc.)
        - Financial analysis (NPV, IRR, payback, LCOA)
        - 3-phase implementation plan
        - Risk assessment
        - Compliance analysis (CBAM, CSRD, SEC)
    """
    # Check if this is a subcommand call
    if ctx.invoked_subcommand is not None:
        return

    # Import agent
    try:
        from greenlang.agents import DecarbonizationRoadmapAgentAI
    except ImportError as e:
        console.print(f"[red]Error importing DecarbonizationRoadmapAgentAI: {e}[/red]")
        console.print("[yellow]Ensure greenlang.agents is properly installed[/yellow]")
        raise typer.Exit(1)

    # Load input data
    input_data = {}

    if input_file:
        # Load from file
        input_path = Path(input_file)
        if not input_path.exists():
            console.print(f"[red]Input file not found: {input_file}[/red]")
            raise typer.Exit(1)

        if input_path.suffix == ".json":
            with open(input_path) as f:
                input_data = json.load(f)
        elif input_path.suffix in [".yaml", ".yml"]:
            with open(input_path) as f:
                input_data = yaml.safe_load(f)
        else:
            console.print(f"[red]Unsupported input format: {input_path.suffix}[/red]")
            console.print("[yellow]Use .json or .yaml files[/yellow]")
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Loaded input from {input_file}")

    elif facility_id and facility_name and industry_type:
        # Build from CLI parameters
        console.print("[yellow]⚠[/yellow] Using CLI parameters (requires additional data in input file)")
        input_data = {
            "facility_id": facility_id,
            "facility_name": facility_name,
            "industry_type": industry_type,
        }
        console.print("[yellow]Note: Fuel consumption, electricity, and other parameters required[/yellow]")
        console.print("[yellow]Please provide complete input via --input file[/yellow]")
        raise typer.Exit(1)

    else:
        # No input provided
        console.print("[red]Error: No input data provided[/red]")
        console.print("\nUsage:")
        console.print("  gl decarbonization --input facility.json")
        console.print("  gl decarbonization demo")
        console.print("\nFor help:")
        console.print("  gl decarbonization --help")
        raise typer.Exit(1)

    # Validate required fields
    required_fields = [
        "facility_id",
        "facility_name",
        "industry_type",
        "latitude",
        "fuel_consumption",
        "electricity_consumption_kwh",
        "grid_region",
        "capital_budget_usd",
    ]

    missing_fields = [field for field in required_fields if field not in input_data]
    if missing_fields:
        console.print(f"[red]Error: Missing required fields: {', '.join(missing_fields)}[/red]")
        console.print("\nRequired fields:")
        for field in required_fields:
            console.print(f"  - {field}")
        raise typer.Exit(1)

    # Display input summary
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]DECARBONIZATION ROADMAP GENERATION[/bold cyan]")
    console.print("=" * 70 + "\n")

    summary_table = Table(box=box.ROUNDED, show_header=False)
    summary_table.add_column("Parameter", style="cyan")
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Facility ID", input_data["facility_id"])
    summary_table.add_row("Facility Name", input_data["facility_name"])
    summary_table.add_row("Industry", input_data["industry_type"])
    summary_table.add_row("Capital Budget", f"${input_data['capital_budget_usd']:,.0f}")
    summary_table.add_row("AI Budget", f"${budget:.2f}")

    console.print(summary_table)
    console.print()

    # Initialize agent
    if verbose:
        console.print("[cyan]Initializing DecarbonizationRoadmapAgentAI...[/cyan]")

    agent = DecarbonizationRoadmapAgentAI(budget_usd=budget)

    # Execute
    console.print("[cyan]Executing decarbonization analysis...[/cyan]")
    console.print("[dim]This may take 30-60 seconds for comprehensive analysis...[/dim]\n")

    try:
        result = agent.run(input_data)
    except Exception as e:
        console.print(f"[red]Error executing agent: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)

    # Check result
    if not result.get("success", False):
        console.print(f"[red]Agent execution failed: {result.get('error', 'Unknown error')}[/red]")
        raise typer.Exit(1)

    data = result["data"]
    metadata = result.get("metadata", {})

    # Display results summary
    console.print("[green]✓[/green] Roadmap generated successfully!\n")

    # Executive Summary
    exec_panel = Panel(
        data.get("executive_summary", "Comprehensive decarbonization roadmap generated"),
        title="[bold green]Executive Summary[/bold green]",
        border_style="green",
    )
    console.print(exec_panel)
    console.print()

    # Key Metrics
    metrics_table = Table(title="Key Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="white", justify="right")

    metrics_table.add_row(
        "Baseline Emissions",
        f"{data.get('baseline_emissions_kg_co2e', 0):,.0f} kg CO2e"
    )
    metrics_table.add_row(
        "Total Reduction Potential",
        f"{data.get('total_reduction_potential_kg_co2e', 0):,.0f} kg CO2e"
    )
    metrics_table.add_row(
        "Reduction Percentage",
        f"{data.get('target_reduction_percent', 0):.1f}%"
    )
    metrics_table.add_row(
        "Total CAPEX",
        f"${data.get('total_capex_required_usd', 0):,.0f}"
    )
    metrics_table.add_row(
        "Federal Incentives",
        f"${data.get('federal_incentives_usd', 0):,.0f}"
    )
    metrics_table.add_row(
        "Net Investment",
        f"${data.get('total_capex_required_usd', 0) - data.get('federal_incentives_usd', 0):,.0f}"
    )
    metrics_table.add_row(
        "NPV (20 years)",
        f"${data.get('npv_usd', 0):,.0f}"
    )
    metrics_table.add_row(
        "IRR",
        f"{data.get('irr_percent', 0):.1f}%"
    )
    metrics_table.add_row(
        "Simple Payback",
        f"{data.get('simple_payback_years', 0):.1f} years"
    )
    metrics_table.add_row(
        "LCOA",
        f"${data.get('lcoa_usd_per_ton', 0):.2f}/ton CO2e"
    )

    console.print(metrics_table)
    console.print()

    # Recommended Pathway
    pathway_panel = Panel(
        f"[bold]{data.get('recommended_pathway', 'Not specified')}[/bold]",
        title="[bold cyan]Recommended Pathway[/bold cyan]",
        border_style="cyan",
    )
    console.print(pathway_panel)
    console.print()

    # AI Metadata
    if verbose and metadata:
        metadata_table = Table(title="AI Execution Metadata", box=box.SIMPLE)
        metadata_table.add_column("Property", style="dim")
        metadata_table.add_column("Value", style="white")

        metadata_table.add_row("Model", metadata.get("model", "Unknown"))
        metadata_table.add_row("Provider", metadata.get("provider", "Unknown"))
        metadata_table.add_row("Tokens", str(metadata.get("tokens", 0)))
        metadata_table.add_row("Cost", f"${metadata.get('cost_usd', 0):.4f}")
        metadata_table.add_row("Deterministic", str(metadata.get("deterministic", True)))

        console.print(metadata_table)
        console.print()

    # Save output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    console.print(f"[green]✓[/green] Full roadmap saved to: [bold]{output}[/bold]")

    # Success message
    console.print("\n" + "=" * 70)
    console.print("[bold green]SUCCESS[/bold green]")
    console.print("=" * 70)
    console.print(f"\nRecommended next steps:")
    for i, step in enumerate(data.get("next_steps", []), 1):
        step_text = step if isinstance(step, str) else step.get("description", str(step))
        console.print(f"  {i}. {step_text}")

    console.print()


@app.command()
def demo():
    """
    Run with sample facility data (Food Processing Plant example)
    """
    console.print("[bold cyan]Running Demo: Food Processing Plant[/bold cyan]\n")

    # Sample data from README example
    sample_data = {
        "facility_id": "DEMO-001",
        "facility_name": "Sample Food Processing Plant",
        "industry_type": "Food & Beverage",
        "latitude": 35.0,
        "fuel_consumption": {
            "natural_gas": 50000,  # MMBtu/year
            "fuel_oil": 5000,
        },
        "electricity_consumption_kwh": 15000000,
        "grid_region": "CAISO",
        "capital_budget_usd": 10000000,
        "target_year": 2030,
        "target_reduction_percent": 50,
        "risk_tolerance": "moderate",
        "facility_sqft": 100000,
    }

    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_data, f, indent=2)
        temp_path = f.name

    console.print(f"[dim]Using sample data from: {temp_path}[/dim]\n")

    # Import and run
    from greenlang.agents import DecarbonizationRoadmapAgentAI

    agent = DecarbonizationRoadmapAgentAI(budget_usd=2.0)

    console.print("[cyan]Executing analysis...[/cyan]\n")

    try:
        result = agent.run(sample_data)

        if result.get("success"):
            data = result["data"]

            console.print("[green]✓[/green] Demo completed successfully!\n")

            # Show key results
            console.print(f"[bold]Baseline Emissions:[/bold] {data.get('baseline_emissions_kg_co2e', 0):,.0f} kg CO2e")
            console.print(f"[bold]Reduction Potential:[/bold] {data.get('total_reduction_potential_kg_co2e', 0):,.0f} kg CO2e ({data.get('target_reduction_percent', 0):.1f}%)")
            console.print(f"[bold]NPV:[/bold] ${data.get('npv_usd', 0):,.0f}")
            console.print(f"[bold]IRR:[/bold] {data.get('irr_percent', 0):.1f}%")
            console.print(f"[bold]Payback:[/bold] {data.get('simple_payback_years', 0):.1f} years")
            console.print(f"[bold]Pathway:[/bold] {data.get('recommended_pathway', 'Not specified')}")

            # Save demo output
            demo_output = Path("demo_roadmap.json")
            with open(demo_output, 'w') as f:
                json.dump(result, f, indent=2)

            console.print(f"\n[green]✓[/green] Full results saved to: [bold]{demo_output}[/bold]")

        else:
            console.print(f"[red]Demo failed: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Cleanup temp file
    import os
    os.unlink(temp_path)


@app.command()
def example():
    """
    Generate example input file template
    """
    console.print("[bold cyan]Generating example input template[/bold cyan]\n")

    example_data = {
        "facility_id": "PLANT-001",
        "facility_name": "Your Facility Name",
        "industry_type": "Food & Beverage",
        "latitude": 40.0,
        "longitude": -95.0,

        "fuel_consumption": {
            "natural_gas": 50000,  # MMBtu/year
            "fuel_oil": 5000,
            "diesel": 1000,
        },

        "electricity_consumption_kwh": 15000000,  # kWh/year
        "grid_region": "US_AVERAGE",

        "capital_budget_usd": 10000000,
        "target_year": 2030,
        "target_reduction_percent": 50,
        "risk_tolerance": "moderate",  # conservative|moderate|aggressive

        "facility_sqft": 50000,
        "renewable_tech_percentage": 40,  # % of CAPEX for solar/renewables

        "value_chain_activities": {
            "purchased_goods_usd": 5000000,  # Optional Scope 3
            "business_travel_miles": 100000,
            "waste_tons": 500,
        },

        "export_markets": ["US", "EU"],  # For compliance analysis
    }

    output_file = "example_facility.json"
    with open(output_file, 'w') as f:
        json.dump(example_data, f, indent=2)

    console.print(f"[green]✓[/green] Example template saved to: [bold]{output_file}[/bold]")
    console.print("\nEdit this file with your facility data, then run:")
    console.print(f"  gl decarbonization --input {output_file}")


if __name__ == "__main__":
    app()
