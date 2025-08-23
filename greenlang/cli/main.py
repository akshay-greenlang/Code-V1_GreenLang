import click
import json
import yaml
import os
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import Workflow
from greenlang.cli.assistant import AIAssistant

console = Console()

# Try to import new agents, fallback to basic if not available
try:
    from greenlang.agents import (
        FuelAgent, CarbonAgent, InputValidatorAgent, 
        ReportAgent, BenchmarkAgent, GridFactorAgent,
        BuildingProfileAgent, IntensityAgent, RecommendationAgent
    )
    ENHANCED_MODE = True
except ImportError:
    from greenlang.agents import (
        FuelAgent, CarbonAgent, InputValidatorAgent, 
        ReportAgent, BenchmarkAgent
    )
    ENHANCED_MODE = False


@click.group(invoke_without_command=True)
@click.version_option(version="0.0.1", prog_name="GreenLang")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """GreenLang - Global Climate Intelligence Framework for Commercial Buildings"""
    if ctx.invoked_subcommand is None:
        # Show welcome message when no subcommand is provided
        version_text = "v0.0.1 - Global Commercial Building Edition" if ENHANCED_MODE else "v0.0.1"
        console.print(Panel.fit(
            f"[bold green]GreenLang Climate Intelligence Framework {version_text}[/bold green]\n\n"
            "Use 'greenlang --help' for available commands\n"
            "Use 'greenlang dev' for interactive developer interface\n"
            "Use 'greenlang calc' for quick emissions calculation\n"
            "Use 'greenlang calc --building' for commercial building analysis", 
            style="green"
        ))
        console.print("\n[dim]Quick Start Commands:[/dim]")
        console.print("  greenlang calc     - Interactive calculator")
        if ENHANCED_MODE:
            console.print("  greenlang calc --building - Commercial building mode")
            console.print("  greenlang analyze  - Analyze building from file")
            console.print("  greenlang benchmark - View regional benchmarks")
            console.print("  greenlang recommend - Get recommendations")
        console.print("  greenlang dev      - Developer interface")
        console.print("  greenlang ask      - AI assistant")
        console.print("  greenlang init     - Initialize project")
        console.print("  greenlang --help   - Show all commands")
        console.print("  greenlang --version - Show version")


@cli.command()
@click.option("--building", is_flag=True, help="Commercial building mode (enhanced)")
@click.option("--country", type=click.Choice(["US", "IN", "EU", "CN", "JP", "BR", "KR", "UK", "DE", "CA", "AU"]), 
              default=None, help="Country/Region for emission factors")
@click.option("--input", "-i", type=click.Path(exists=True), help="Load data from JSON file")
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
def calc(building: bool, country: Optional[str], input: Optional[str], output: Optional[str]) -> None:
    """Interactive emissions calculator with optional commercial building mode"""
    
    # If building mode requested and enhanced features available
    if building and ENHANCED_MODE:
        if input:
            # Load from file
            with open(input, 'r') as f:
                data = json.load(f)
            _process_building_data(data, country, output)
        else:
            _interactive_building_calculator(country, output)
    else:
        # Use simple calculator (original or fallback)
        _simple_calculator(country, output)


def _interactive_building_calculator(country: Optional[str], output_path: Optional[str]) -> None:
    """Interactive commercial building emissions calculator"""
    console.print(Panel("Commercial Building Emissions Calculator", style="cyan"))
    
    try:
        from greenlang.sdk.enhanced_client import GreenLangClient
        from greenlang.agents import GridFactorAgent
    except ImportError:
        console.print("[yellow]Enhanced features not available. Using simple calculator.[/yellow]")
        _simple_calculator(country, output_path)
        return
    
    # Get country/region
    if not country:
        country = Prompt.ask(
            "Select country/region",
            choices=["US", "IN", "EU", "CN", "JP", "BR", "KR", "UK", "DE", "CA", "AU"],
            default="US"
        )
    
    # Get building metadata
    console.print("\n[bold]Building Information:[/bold]")
    
    building_types = ["commercial_office", "hospital", "data_center", "retail", 
                     "warehouse", "hotel", "education", "restaurant", "industrial"]
    building_type = Prompt.ask(
        "Building type",
        choices=building_types,
        default="commercial_office"
    )
    
    area = float(Prompt.ask("Building area (sqft)", default="50000"))
    occupancy = int(Prompt.ask("Average daily occupancy", default="200"))
    floor_count = int(Prompt.ask("Number of floors", default="10"))
    building_age = int(Prompt.ask("Building age (years)", default="15"))
    
    climate_zones = ["1A", "2A", "3A", "4A", "5A", "6A", "7", "8", "tropical", "temperate", "dry"]
    climate_zone = Prompt.ask("Climate zone (optional)", choices=climate_zones + ["skip"], default="skip")
    if climate_zone == "skip":
        climate_zone = None
    
    # Get energy consumption
    console.print("\n[bold]Annual Energy Consumption:[/bold]")
    console.print("[dim]Enter 0 or press Enter to skip[/dim]\n")
    
    energy_data = {}
    
    # Electricity
    electricity = float(Prompt.ask("Electricity (kWh/year)", default="0"))
    if electricity > 0:
        energy_data["electricity"] = {"value": electricity, "unit": "kWh"}
    
    # Natural Gas
    gas_units = ["therms", "m3", "MMBtu", "kWh"]
    gas_value = float(Prompt.ask("Natural gas consumption", default="0"))
    if gas_value > 0:
        gas_unit = Prompt.ask("Natural gas unit", choices=gas_units, default="therms")
        energy_data["natural_gas"] = {"value": gas_value, "unit": gas_unit}
    
    # Diesel (for backup generators)
    diesel_value = float(Prompt.ask("Diesel (liters/year)", default="0"))
    if diesel_value > 0:
        energy_data["diesel"] = {"value": diesel_value, "unit": "liters"}
    
    # District Heating (for EU/China)
    if country in ["EU", "CN", "DE", "KR"]:
        district = float(Prompt.ask("District heating (kWh/year)", default="0"))
        if district > 0:
            energy_data["district_heating"] = {"value": district, "unit": "kWh"}
    
    # Solar PV generation
    solar = float(Prompt.ask("Solar PV generation (kWh/year)", default="0"))
    if solar > 0:
        energy_data["solar_pv_generation"] = {"value": solar, "unit": "kWh"}
    
    # Process the data
    building_data = {
        "metadata": {
            "building_type": building_type,
            "area": area,
            "area_unit": "sqft",
            "location": {"country": country},
            "occupancy": occupancy,
            "floor_count": floor_count,
            "building_age": building_age,
            "climate_zone": climate_zone
        },
        "energy_consumption": energy_data
    }
    
    _process_building_data(building_data, country, output_path)


def _process_building_data(data: Dict[str, Any], country: Optional[str], output_path: Optional[str]) -> None:
    """Process building data through all enhanced agents"""
    
    try:
        from greenlang.sdk.enhanced_client import GreenLangClient
        
        # Override country if specified
        if country:
            data["metadata"]["location"]["country"] = country
        
        client = GreenLangClient(region=data["metadata"]["location"]["country"])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Analyzing building...", total=None)
            
            # Use enhanced client's analyze_building method
            results = client.analyze_building(data)
            
            progress.update(task, description="[cyan]Generating report...")
        
        if results["success"]:
            _display_enhanced_results(results["data"], data["metadata"])
            
            # Save if requested
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                console.print(f"\n[green]Results saved to {output_path}[/green]")
        else:
            console.print(f"[red]Error: {results.get('error', 'Unknown error')}[/red]")
            
    except ImportError:
        console.print("[yellow]Enhanced features not available. Using simple calculation.[/yellow]")
        _simple_calculator(country, output_path)


def _display_enhanced_results(results: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Display comprehensive results from enhanced analysis"""
    
    console.print("\n" + "="*60)
    console.print(Panel.fit("[bold green]Commercial Building Emissions Analysis[/bold green]", style="green"))
    
    # Building Profile
    if "profile" in results:
        profile = results["profile"]
        console.print("\n[bold]Building Profile:[/bold]")
        console.print(f"  Type: {metadata['building_type']}")
        console.print(f"  Location: {metadata['location']['country']}")
        console.print(f"  Area: {metadata['area']:,.0f} sqft")
        console.print(f"  Age Category: {profile.get('age_category', 'N/A')}")
        console.print(f"  Expected EUI: {profile.get('expected_eui', 'N/A')} kWh/sqft/year")
    
    # Emissions Summary
    if "emissions" in results:
        emissions = results["emissions"]
        console.print("\n[bold]Total Emissions:[/bold]")
        console.print(f"  Annual: {emissions.get('total_co2e_tons', 0):.2f} metric tons CO2e")
        console.print(f"  Annual: {emissions.get('total_co2e_kg', 0):,.0f} kg CO2e")
        
        if "emissions_breakdown" in emissions:
            console.print("\n[bold]Emissions by Source:[/bold]")
            for item in emissions["emissions_breakdown"]:
                console.print(f"  {item['source']}: {item['co2e_tons']:.2f} tons ({item['percentage']}%)")
    
    # Intensity Metrics
    if "intensity" in results:
        intensity = results["intensity"]["intensities"]
        rating = results["intensity"].get("performance_rating", "N/A")
        
        console.print("\n[bold]Intensity Metrics:[/bold]")
        console.print(f"  Per sqft: {intensity.get('per_sqft_year', 0):.2f} kgCO2e/sqft/year")
        console.print(f"  Per person: {intensity.get('per_person_year', 0):.0f} kgCO2e/person/year")
        if "energy_use_intensity_kwh_sqft" in intensity:
            console.print(f"  Energy Use Intensity: {intensity['energy_use_intensity_kwh_sqft']:.1f} kWh/sqft/year")
        console.print(f"\n[bold]Performance Rating:[/bold] {rating}")
    
    # Benchmark Comparison
    if "benchmark" in results:
        benchmark = results["benchmark"]
        console.print(f"\n[bold]Benchmark Rating:[/bold] {benchmark.get('rating', 'N/A')}")
        console.print(f"  Category: {benchmark.get('performance_category', 'N/A')}")
        if "comparison_message" in benchmark:
            console.print(f"  {benchmark['comparison_message']}")
    
    # Top Recommendations
    if "recommendations" in results:
        recs = results["recommendations"]
        console.print("\n[bold]Top Recommendations:[/bold]")
        
        # Quick wins
        if "quick_wins" in recs:
            console.print("\n  [cyan]Quick Wins (Low Cost, High Impact):[/cyan]")
            for i, rec in enumerate(recs["quick_wins"][:3], 1):
                console.print(f"    {i}. {rec['action']}")
                console.print(f"       Impact: {rec['impact']}, Payback: {rec['payback']}")
        
        # Potential savings
        if "potential_emissions_reduction" in recs:
            savings = recs["potential_emissions_reduction"]
            console.print(f"\n  [green]Potential Savings:[/green] {savings.get('percentage_range', 'N/A')}")


def _simple_calculator(country: Optional[str], output_path: Optional[str]) -> None:
    """Simple emissions calculator (original functionality)"""
    console.print(Panel("Simple Emissions Calculator", style="cyan"))
    
    # Try to use enhanced client if available
    try:
        from greenlang.sdk.enhanced_client import GreenLangClient
        if not country:
            country = Prompt.ask("Country/Region (optional)", 
                               choices=["US", "IN", "EU", "CN", "JP", "BR", "KR", "skip"],
                               default="skip")
            if country == "skip":
                country = "US"
        client = GreenLangClient(region=country)
        use_enhanced = True
    except ImportError:
        from greenlang.sdk import GreenLangClient
        client = GreenLangClient()
        use_enhanced = False
        country = "US"
    
    # Collect inputs
    fuels: List[Dict[str, Any]] = []
    
    console.print("\n[bold]Enter fuel consumption data:[/bold]")
    console.print("[dim]Press Enter with 0 or empty value to skip[/dim]\n")
    
    # Electricity
    electricity = Prompt.ask("Electricity consumption (kWh)", default="0")
    try:
        if float(electricity) > 0:
            fuels.append({
                "fuel_type": "electricity",
                "consumption": float(electricity),
                "unit": "kWh"
            })
    except ValueError:
        pass
    
    # Natural Gas
    gas = Prompt.ask("Natural gas (therms)", default="0")
    try:
        if float(gas) > 0:
            fuels.append({
                "fuel_type": "natural_gas",
                "consumption": float(gas),
                "unit": "therms"
            })
    except ValueError:
        pass
    
    # Diesel
    diesel = Prompt.ask("Diesel (gallons)", default="0")
    try:
        if float(diesel) > 0:
            fuels.append({
                "fuel_type": "diesel",
                "consumption": float(diesel),
                "unit": "gallons"
            })
    except ValueError:
        pass
    
    if not fuels:
        console.print("[red]No fuel data entered[/red]")
        return
    
    # Building info (optional)
    console.print("\n[bold]Building information (optional):[/bold]")
    area = Prompt.ask("Building area (sqft)", default="0")
    
    # Calculate emissions
    console.print("\n[cyan]Calculating emissions...[/cyan]")
    
    emissions_list = []
    total_kg = 0
    
    for fuel in fuels:
        if use_enhanced:
            result = client.calculate_emissions(
                fuel["fuel_type"],
                fuel["consumption"],
                fuel["unit"],
                region=country
            )
        else:
            result = client.calculate_emissions(
                fuel["fuel_type"],
                fuel["consumption"],
                fuel["unit"]
            )
        
        if result["success"]:
            emissions_list.append(result["data"])
            total_kg += result["data"]["co2e_emissions_kg"]
            console.print(f"  ✓ {fuel['fuel_type']}: {result['data']['co2e_emissions_kg']:.2f} kg CO2e")
    
    # Aggregate
    if emissions_list:
        agg_result = client.aggregate_emissions(emissions_list)
        
        console.print("\n[bold green]Results:[/bold green]")
        console.print(f"Total Emissions: {agg_result['data']['total_co2e_tons']:.3f} metric tons CO2e")
        console.print(f"Total Emissions: {agg_result['data']['total_co2e_kg']:.2f} kg CO2e")
        
        if use_enhanced and country != "US":
            console.print(f"Region: {country}")
        
        # Show breakdown
        if "emissions_breakdown" in agg_result["data"]:
            console.print("\n[bold]Breakdown by Source:[/bold]")
            for item in agg_result["data"]["emissions_breakdown"]:
                console.print(f"  - {item['source']}: {item['co2e_tons']:.3f} tons ({item['percentage']}%)")
        
        # Benchmark if area provided
        if float(area) > 0:
            benchmark_result = client.benchmark_emissions(
                agg_result["data"]["total_co2e_kg"],
                float(area),
                "commercial_office",
                12  # annual
            )
            if benchmark_result["success"]:
                console.print(f"\n[bold]Benchmark Rating:[/bold] {benchmark_result['data']['rating']}")
                console.print(f"Carbon Intensity: {benchmark_result['data']['carbon_intensity']:.2f} kg CO2e/sqft/year")
        
        # Save results if requested
        if output_path:
            results = {
                "emissions": agg_result["data"],
                "inputs": fuels,
                "country": country if use_enhanced else "US"
            }
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]Results saved to {output_path}[/green]")


@cli.command()
@click.argument("building_file", type=click.Path(exists=True))
@click.option("--country", type=click.Choice(["US", "IN", "EU", "CN", "JP", "BR", "KR"]))
def analyze(building_file: str, country: Optional[str]) -> None:
    """Analyze an existing building from JSON file"""
    
    if not ENHANCED_MODE:
        console.print("[red]This command requires enhanced features. Please ensure all agents are installed.[/red]")
        return
    
    with open(building_file, 'r') as f:
        data = json.load(f)
    
    if country:
        data["metadata"]["location"]["country"] = country
    
    _process_building_data(data, country, None)


@cli.command()
@click.option("--type", "building_type", 
              type=click.Choice(["commercial_office", "hospital", "data_center", "retail", "warehouse", "hotel", "education"]),
              default="commercial_office")
@click.option("--country", type=click.Choice(["US", "IN", "EU", "CN", "JP", "BR", "KR"]), default="US")
@click.option("--list", "list_all", is_flag=True, help="List all available benchmarks")
def benchmark(building_type: str, country: str, list_all: bool) -> None:
    """Show benchmark data for building types by country"""
    
    if not ENHANCED_MODE:
        console.print("[yellow]Enhanced benchmarks not available. Showing basic benchmarks.[/yellow]")
        # Show basic benchmarks
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rating", style="cyan")
        table.add_column("Carbon Intensity", style="green")
        table.add_row("Excellent", "< 5.0 kg CO2e/sqft/year")
        table.add_row("Good", "5.0 - 8.0 kg CO2e/sqft/year")
        table.add_row("Average", "8.0 - 12.0 kg CO2e/sqft/year")
        table.add_row("Below Average", "12.0 - 15.0 kg CO2e/sqft/year")
        table.add_row("Poor", "> 15.0 kg CO2e/sqft/year")
        console.print(table)
        return
    
    import json
    benchmarks_path = Path(__file__).parent.parent / "data" / "global_benchmarks.json"
    
    if not benchmarks_path.exists():
        console.print("[red]Benchmark data file not found.[/red]")
        return
    
    with open(benchmarks_path, 'r') as f:
        benchmarks = json.load(f)
    
    if list_all:
        console.print(Panel.fit("[bold]Available Benchmarks[/bold]", style="cyan"))
        for btype in benchmarks:
            if btype != "metadata":
                console.print(f"\n[bold]{btype}:[/bold]")
                for country_code in benchmarks[btype]:
                    console.print(f"  - {country_code}")
        return
    
    if building_type in benchmarks and country in benchmarks[building_type]:
        data = benchmarks[building_type][country]
        
        console.print(Panel.fit(f"[bold]Benchmarks: {building_type} in {country}[/bold]", style="cyan"))
        console.print(f"\nSource: {data.get('source', 'N/A')}")
        console.print(f"Year: {data.get('year', 'N/A')}")
        console.print(f"Unit: {data.get('unit', 'kgCO2e/sqft/year')}\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Performance Level", style="cyan")
        table.add_column("Threshold", style="green")
        
        table.add_row("Excellent", f"< {data['excellent']}")
        table.add_row("Good", f"{data['excellent']} - {data['good']}")
        table.add_row("Average", f"{data['good']} - {data['average']}")
        table.add_row("Below Average", f"{data['average']} - {data['poor']}")
        table.add_row("Poor", f"> {data['poor']}")
        
        console.print(table)
    else:
        console.print(f"[red]No benchmark data for {building_type} in {country}[/red]")


@cli.command()
def recommend() -> None:
    """Interactive recommendation generator"""
    
    if not ENHANCED_MODE:
        console.print("[yellow]Enhanced recommendations not available.[/yellow]")
        console.print("\n[bold]General Recommendations:[/bold]")
        console.print("1. Upgrade to LED lighting (50-70% reduction)")
        console.print("2. Install smart thermostats (10-15% reduction)")
        console.print("3. Improve insulation (10-20% reduction)")
        console.print("4. Consider solar PV installation (30-70% reduction)")
        return
    
    console.print(Panel("Building Optimization Recommendations", style="cyan"))
    
    try:
        from greenlang.agents import RecommendationAgent
    except ImportError:
        console.print("[red]Recommendation agent not available.[/red]")
        return
    
    # Get basic info
    building_type = Prompt.ask("Building type", 
                              choices=["commercial_office", "hospital", "data_center", "retail", "warehouse"],
                              default="commercial_office")
    
    country = Prompt.ask("Country", 
                        choices=["US", "IN", "EU", "CN", "JP", "BR", "KR"],
                        default="US")
    
    building_age = int(Prompt.ask("Building age (years)", default="15"))
    
    performance = Prompt.ask("Current performance rating",
                            choices=["Excellent", "Good", "Average", "Below Average", "Poor"],
                            default="Average")
    
    # Initialize recommendation agent
    rec_agent = RecommendationAgent()
    
    result = rec_agent.run({
        "building_type": building_type,
        "country": country,
        "building_age": building_age,
        "performance_rating": performance
    })
    
    if result.success:
        recs = result.data
        
        console.print("\n[bold green]Optimization Recommendations:[/bold green]\n")
        
        # Quick wins
        if "quick_wins" in recs:
            console.print("[bold]Quick Wins:[/bold]")
            for i, rec in enumerate(recs["quick_wins"][:3], 1):
                console.print(f"  {i}. {rec['action']}")
                console.print(f"     Impact: {rec['impact']}, Payback: {rec['payback']}")
        
        # Implementation roadmap
        if "implementation_roadmap" in recs:
            console.print("\n[bold]Implementation Roadmap:[/bold]")
            for phase in recs["implementation_roadmap"]:
                console.print(f"\n  [cyan]{phase['phase']}[/cyan]")
                console.print(f"  Estimated Cost: {phase['estimated_cost']}")
                console.print(f"  Expected Impact: {phase['expected_impact']}")


# Import existing commands from original main.py
@cli.command()
@click.argument("workflow_file", type=click.Path(exists=True))
@click.option("--input", "-i", type=click.Path(exists=True), help="Input data file (JSON)")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "-f", type=click.Choice(["json", "text", "markdown"]), default="text", help="Output format")
def run(workflow_file: str, input: Optional[str], output: Optional[str], format: str) -> None:
    """Run a workflow with the specified input data"""
    
    console.print(Panel.fit("GreenLang Climate Intelligence", style="green bold"))
    
    orchestrator = Orchestrator()
    
    console.print("Loading agents...", style="dim")
    orchestrator.register_agent("validator", InputValidatorAgent())
    orchestrator.register_agent("fuel", FuelAgent())
    orchestrator.register_agent("carbon", CarbonAgent())
    orchestrator.register_agent("report", ReportAgent())
    orchestrator.register_agent("benchmark", BenchmarkAgent())
    
    # Register enhanced agents if available
    if ENHANCED_MODE:
        orchestrator.register_agent("grid_factor", GridFactorAgent())
        orchestrator.register_agent("building_profile", BuildingProfileAgent())
        orchestrator.register_agent("intensity", IntensityAgent())
        orchestrator.register_agent("recommendation", RecommendationAgent())
    
    console.print("Loading workflow...", style="dim")
    if workflow_file.endswith(".yaml") or workflow_file.endswith(".yml"):
        workflow = Workflow.from_yaml(workflow_file)
    else:
        workflow = Workflow.from_json(workflow_file)
    
    orchestrator.register_workflow("main", workflow)
    
    input_data = {}
    if input:
        console.print(f"Loading input from {input}...", style="dim")
        with open(input, 'r') as f:
            input_data = json.load(f)
    
    console.print("Executing workflow...\n", style="bold cyan")
    
    result = orchestrator.execute_workflow("main", input_data)
    
    if result["success"]:
        console.print("[OK] Workflow completed successfully!\n", style="bold green")
    else:
        console.print("[!] Workflow completed with errors\n", style="bold yellow")
        for error in result["errors"]:
            console.print(f"  [X] {error['step']}: {error['error']}", style="red")
    
    # Format and display/save output
    if format == "json":
        output_content = json.dumps(result, indent=2)
    else:
        output_content = str(result)
    
    if output:
        with open(output, 'w') as f:
            f.write(output_content)
        console.print(f"\nResults saved to {output}", style="green")
    else:
        console.print("\nResults:", style="bold")
        console.print(output_content)


@cli.command()
def agents() -> None:
    """List available agents"""
    
    agents_info = [
        ("validator", "InputValidatorAgent", "Validates input data for emissions calculations"),
        ("fuel", "FuelAgent", "Calculates emissions based on fuel consumption"),
        ("carbon", "CarbonAgent", "Aggregates emissions and provides carbon footprint"),
        ("report", "ReportAgent", "Generates carbon footprint reports"),
        ("benchmark", "BenchmarkAgent", "Compares emissions against industry benchmarks"),
    ]
    
    if ENHANCED_MODE:
        agents_info.extend([
            ("grid_factor", "GridFactorAgent", "Retrieves country-specific emission factors"),
            ("building_profile", "BuildingProfileAgent", "Categorizes buildings and expected performance"),
            ("intensity", "IntensityAgent", "Calculates emission intensity metrics"),
            ("recommendation", "RecommendationAgent", "Provides optimization recommendations"),
        ])
    
    table = Table(title="Available Agents", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Class", style="green")
    table.add_column("Description", style="white")
    
    for agent_id, class_name, description in agents_info:
        table.add_row(agent_id, class_name, description)
    
    console.print(table)


@cli.command()
@click.argument("agent_id")
def agent(agent_id: str) -> None:
    """Show details about a specific agent"""
    
    agents_map = {
        "validator": InputValidatorAgent(),
        "fuel": FuelAgent(),
        "carbon": CarbonAgent(),
        "report": ReportAgent(),
        "benchmark": BenchmarkAgent(),
    }
    
    if ENHANCED_MODE:
        agents_map.update({
            "grid_factor": GridFactorAgent(),
            "building_profile": BuildingProfileAgent(),
            "intensity": IntensityAgent(),
            "recommendation": RecommendationAgent(),
        })
    
    if agent_id not in agents_map:
        console.print(f"[X] Agent '{agent_id}' not found", style="red")
        return
    
    agent_instance = agents_map[agent_id]
    
    console.print(Panel.fit(f"Agent: {agent_id}", style="cyan bold"))
    console.print(f"Class: {agent_instance.__class__.__name__}")
    console.print(f"Description: {agent_instance.config.description}")
    console.print(f"Version: {agent_instance.config.version}")
    console.print(f"Enabled: {agent_instance.config.enabled}")


@cli.command()
@click.argument("question", nargs=-1, required=False)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed agent execution")
def ask(question: Tuple[str, ...], verbose: bool) -> None:
    """Ask the AI assistant about emissions (Natural Language Interface)"""
    
    if not question:
        question_text = Prompt.ask("Enter your question")
    else:
        question_text = " ".join(question)
    
    console.print(Panel.fit("GreenLang AI Assistant", style="cyan bold"))
    console.print(f"\nQuestion: {question_text}\n", style="dim")
    
    try:
        assistant = AIAssistant()
        result = assistant.process_query(question_text, verbose=verbose)
        
        if result["success"]:
            console.print("[OK] Analysis Complete\n", style="bold green")
            console.print(result["response"])
            
            if verbose and "agent_results" in result:
                console.print("\nAgent Execution Details:", style="bold")
                for agent_name, agent_result in result["agent_results"].items():
                    console.print(f"\n  {agent_name}:", style="cyan")
                    console.print(f"    Success: {agent_result.get('success', False)}")
                    if "data" in agent_result:
                        console.print(f"    Data: {json.dumps(agent_result['data'], indent=6)}")
        else:
            console.print("[X] Error processing query", style="red")
            console.print(result.get("error", "Unknown error"))
            
    except Exception as e:
        console.print(f"[X] Error: {str(e)}", style="red")


@cli.command()
def dev() -> None:
    """Launch the developer interface (VS Code-like terminal UI)"""
    from greenlang.cli.dev_interface import main as dev_main
    dev_main()


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="workflow.yaml", help="Output file path")
def init(output: str) -> None:
    """Create a sample workflow configuration"""
    
    # Create enhanced workflow if in enhanced mode
    if ENHANCED_MODE:
        sample_workflow = {
            "name": "commercial_building_analysis",
            "description": "Comprehensive building emissions analysis",
            "version": "0.0.1",
            "steps": [
                {
                    "name": "validate_input",
                    "agent_id": "validator",
                    "description": "Validate input data"
                },
                {
                    "name": "profile_building",
                    "agent_id": "building_profile",
                    "description": "Analyze building characteristics"
                },
                {
                    "name": "calculate_emissions",
                    "agent_id": "grid_factor",
                    "description": "Calculate emissions with regional factors"
                },
                {
                    "name": "calculate_intensity",
                    "agent_id": "intensity",
                    "description": "Calculate intensity metrics"
                },
                {
                    "name": "benchmark",
                    "agent_id": "benchmark",
                    "description": "Compare to standards"
                },
                {
                    "name": "recommendations",
                    "agent_id": "recommendation",
                    "description": "Generate recommendations"
                },
                {
                    "name": "report",
                    "agent_id": "report",
                    "description": "Generate final report"
                }
            ]
        }
    else:
        # Basic workflow
        sample_workflow = {
            "name": "carbon_footprint_calculation",
            "description": "Calculate carbon footprint for a commercial building",
            "version": "0.0.1",
            "steps": [
                {
                    "name": "validate_input",
                    "agent_id": "validator",
                    "description": "Validate input data"
                },
                {
                    "name": "calculate_emissions",
                    "agent_id": "fuel",
                    "description": "Calculate emissions"
                },
                {
                    "name": "aggregate",
                    "agent_id": "carbon",
                    "description": "Aggregate emissions"
                },
                {
                    "name": "benchmark",
                    "agent_id": "benchmark",
                    "description": "Benchmark performance"
                },
                {
                    "name": "report",
                    "agent_id": "report",
                    "description": "Generate report"
                }
            ]
        }
    
    with open(output, 'w') as f:
        yaml.dump(sample_workflow, f, default_flow_style=False)
    
    console.print(f"[OK] Sample workflow created: {output}", style="green")
    
    # Create sample input
    if ENHANCED_MODE:
        sample_input = {
            "metadata": {
                "building_type": "commercial_office",
                "area": 50000,
                "area_unit": "sqft",
                "location": {
                    "country": "US",
                    "city": "New York"
                },
                "occupancy": 200,
                "floor_count": 10,
                "building_age": 15
            },
            "energy_consumption": {
                "electricity": {"value": 1500000, "unit": "kWh"},
                "natural_gas": {"value": 30000, "unit": "therms"}
            }
        }
    else:
        sample_input = {
            "fuels": [
                {"type": "electricity", "consumption": 1000, "unit": "kWh"},
                {"type": "natural_gas", "consumption": 500, "unit": "therms"}
            ],
            "building_info": {
                "type": "commercial_office",
                "area": 10000,
                "area_unit": "sqft"
            }
        }
    
    input_file = output.replace(".yaml", "_input.json").replace(".yml", "_input.json")
    with open(input_file, 'w') as f:
        json.dump(sample_input, f, indent=2)
    
    console.print(f"[OK] Sample input created: {input_file}", style="green")
    console.print(f"\nTo run the workflow, use:", style="dim")
    console.print(f"  greenlang run {output} --input {input_file}", style="cyan")


if __name__ == "__main__":
    cli()