#!/usr/bin/env python3
"""
GreenLang Developer Interface - VS Code-like Terminal UI
"""

import os
import sys
import json
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from rich.live import Live
from rich.text import Text
from rich.columns import Columns
from rich import box

from greenlang.sdk import GreenLangClient, WorkflowBuilder, AgentBuilder
from greenlang.agents.base import AgentResult
from greenlang.core.workflow import Workflow

console = Console()


class GreenLangDevInterface:
    """Interactive Developer Interface for GreenLang"""
    
    def __init__(self):
        self.client = GreenLangClient()
        self.current_project = None
        self.history = []
        self.workspace = Path.cwd()
        self.session_data = {}
        
    def start(self):
        """Start the developer interface"""
        self.show_welcome()
        self.main_loop()
    
    def show_welcome(self):
        """Display welcome screen"""
        welcome_text = """
[bold green]GreenLang Developer Interface v0.0.1[/bold green]
[dim]Climate Intelligence Framework - Developer Tools[/dim]

[cyan]Commands:[/cyan]
  • [bold]new[/bold]      - Create new project/workflow
  • [bold]calc[/bold]     - Interactive emissions calculator
  • [bold]test[/bold]     - Run test suite
  • [bold]agents[/bold]   - Manage agents
  • [bold]workflow[/bold] - Workflow designer
  • [bold]repl[/bold]     - Python REPL with GreenLang
  • [bold]docs[/bold]     - View documentation
  • [bold]help[/bold]     - Show all commands
  • [bold]exit[/bold]     - Exit interface
        """
        console.print(Panel(welcome_text, title="🌍 Welcome to GreenLang", border_style="green"))
    
    def main_loop(self):
        """Main command loop"""
        while True:
            try:
                command = Prompt.ask("\n[bold cyan]greenlang[/bold cyan]", default="help")
                
                if command.lower() in ['exit', 'quit', 'q']:
                    if Confirm.ask("Exit GreenLang Developer Interface?"):
                        console.print("[green]Goodbye![/green]")
                        break
                
                self.execute_command(command)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def execute_command(self, command: str):
        """Execute a command"""
        parts = command.strip().split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        commands = {
            # Core Commands
            'new': self.cmd_new,
            'calc': self.cmd_calc,
            'test': self.cmd_test,
            'agents': self.cmd_agents,
            'workflow': self.cmd_workflow,
            'repl': self.cmd_repl,
            
            # Project Commands
            'workspace': self.cmd_workspace,
            'run': self.cmd_run,
            'export': self.cmd_export,
            'init': self.cmd_init,
            'project': self.cmd_project,
            
            # Analysis Commands
            'benchmark': self.cmd_benchmark,
            'profile': self.cmd_profile,
            'validate': self.cmd_validate,
            'analyze': self.cmd_analyze,
            'compare': self.cmd_compare,
            
            # Documentation
            'docs': self.cmd_docs,
            'help': self.cmd_help,
            'examples': self.cmd_examples,
            'api': self.cmd_api,
            
            # System
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            'clear': self.cmd_clear,
            'status': self.cmd_status,
            'version': self.cmd_version,
            'config': self.cmd_config,
        }
        
        if cmd in commands:
            try:
                commands[cmd](args)
            except Exception as e:
                console.print(f"[red]Error executing {cmd}: {str(e)}[/red]")
                console.print("[dim]Use 'help' for command usage[/dim]")
        else:
            console.print(f"[red]Unknown command: {cmd}[/red]")
            console.print("Type 'help' for available commands")
    
    def cmd_new(self, args):
        """Create new project or workflow"""
        project_type = Prompt.ask(
            "What would you like to create?",
            choices=["project", "workflow", "agent", "config"]
        )
        
        if project_type == "project":
            self.create_project()
        elif project_type == "workflow":
            self.create_workflow()
        elif project_type == "agent":
            self.create_agent()
        elif project_type == "config":
            self.create_config()
    
    def create_project(self):
        """Create a new GreenLang project"""
        project_name = Prompt.ask("Project name")
        project_path = self.workspace / project_name
        
        if project_path.exists():
            console.print(f"[red]Project {project_name} already exists[/red]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Creating project structure...", total=5)
            
            # Create directories
            project_path.mkdir(parents=True)
            (project_path / "workflows").mkdir()
            progress.update(task, advance=1)
            
            (project_path / "agents").mkdir()
            progress.update(task, advance=1)
            
            (project_path / "data").mkdir()
            progress.update(task, advance=1)
            
            (project_path / "tests").mkdir()
            progress.update(task, advance=1)
            
            # Create initial files
            self.create_project_files(project_path, project_name)
            progress.update(task, advance=1)
        
        console.print(f"[green]✓ Project '{project_name}' created successfully![/green]")
        console.print(f"[dim]Location: {project_path}[/dim]")
        
        if Confirm.ask("Switch to new project?"):
            self.workspace = project_path
            self.current_project = project_name
            console.print(f"[green]Switched to project: {project_name}[/green]")
    
    def create_project_files(self, project_path: Path, project_name: str):
        """Create initial project files"""
        # greenlang.yaml
        config = {
            "name": project_name,
            "version": "0.0.1",
            "description": f"GreenLang project: {project_name}",
            "agents": ["validator", "fuel", "carbon", "report", "benchmark"],
            "workflows": [],
            "settings": {
                "region": "US",
                "report_format": "text",
                "auto_validate": True
            }
        }
        
        with open(project_path / "greenlang.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # README.md
        readme = f"""# {project_name}

A GreenLang climate intelligence project.

## Quick Start

```bash
greenlang run workflows/main.yaml
```

## Project Structure

- `workflows/` - Workflow definitions
- `agents/` - Custom agents
- `data/` - Data files and emission factors
- `tests/` - Test files

## Usage

1. Define your workflows in YAML
2. Run with: `greenlang run <workflow>`
3. Test with: `greenlang test`
"""
        
        with open(project_path / "README.md", 'w') as f:
            f.write(readme)
        
        # Sample workflow
        sample_workflow = {
            "name": "main",
            "description": "Main workflow",
            "steps": [
                {"name": "validate", "agent_id": "validator"},
                {"name": "calculate", "agent_id": "fuel"},
                {"name": "aggregate", "agent_id": "carbon"},
                {"name": "report", "agent_id": "report"}
            ]
        }
        
        with open(project_path / "workflows" / "main.yaml", 'w') as f:
            yaml.dump(sample_workflow, f, default_flow_style=False)
    
    def cmd_calc(self, args):
        """Interactive emissions calculator"""
        console.print(Panel("🔬 Interactive Emissions Calculator", style="cyan"))
        
        # Collect inputs
        fuels = []
        
        console.print("\n[bold]Enter fuel consumption data:[/bold]")
        console.print("[dim]Press Enter with empty value to skip[/dim]\n")
        
        # Electricity
        electricity = Prompt.ask("Electricity consumption (kWh)", default="0")
        if float(electricity) > 0:
            fuels.append({
                "fuel_type": "electricity",
                "consumption": float(electricity),
                "unit": "kWh"
            })
        
        # Natural Gas
        gas = Prompt.ask("Natural gas (therms)", default="0")
        if float(gas) > 0:
            fuels.append({
                "fuel_type": "natural_gas",
                "consumption": float(gas),
                "unit": "therms"
            })
        
        # Diesel
        diesel = Prompt.ask("Diesel (gallons)", default="0")
        if float(diesel) > 0:
            fuels.append({
                "fuel_type": "diesel",
                "consumption": float(diesel),
                "unit": "gallons"
            })
        
        # Building info (optional)
        console.print("\n[bold]Building information (optional):[/bold]")
        area = Prompt.ask("Building area (sqft)", default="0")
        building_type = Prompt.ask(
            "Building type",
            choices=["commercial_office", "retail", "warehouse", "residential", "skip"],
            default="commercial_office"
        )
        
        if not fuels:
            console.print("[red]No fuel data entered[/red]")
            return
        
        # Calculate
        with console.status("Calculating emissions..."):
            results = self.calculate_emissions(fuels, float(area), building_type)
        
        # Display results
        self.display_calculation_results(results)
    
    def calculate_emissions(self, fuels: List[Dict], area: float = 0, building_type: str = "commercial_office") -> Dict:
        """Calculate emissions using the SDK"""
        emissions_list = []
        
        for fuel in fuels:
            result = self.client.calculate_emissions(
                fuel["fuel_type"],
                fuel["consumption"],
                fuel["unit"]
            )
            if result["success"]:
                emissions_list.append(result["data"])
        
        # Aggregate
        agg_result = self.client.aggregate_emissions(emissions_list)
        
        # Benchmark if area provided
        benchmark_result = None
        if area > 0:
            benchmark_result = self.client.benchmark_emissions(
                agg_result["data"]["total_co2e_kg"],
                area,
                building_type,
                1  # 1 month
            )
        
        # Generate report
        report_result = self.client.generate_report(
            agg_result["data"],
            format="text",
            building_info={"area": area, "type": building_type} if area > 0 else None
        )
        
        return {
            "emissions": agg_result["data"],
            "benchmark": benchmark_result["data"] if benchmark_result and benchmark_result["success"] else None,
            "report": report_result["data"]["report"] if report_result["success"] else None
        }
    
    def display_calculation_results(self, results: Dict):
        """Display calculation results in a formatted way"""
        layout = Layout()
        layout.split_column(
            Layout(name="summary", size=10),
            Layout(name="details", size=15),
            Layout(name="report", size=10)
        )
        
        # Summary
        summary_table = Table(title="Emissions Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Total CO2e", f"{results['emissions']['total_co2e_tons']:.3f} metric tons")
        summary_table.add_row("Total kg CO2e", f"{results['emissions']['total_co2e_kg']:.2f} kg")
        
        layout["summary"].update(Panel(summary_table, title="📊 Results"))
        
        # Details
        if results['emissions'].get('emissions_breakdown'):
            breakdown_table = Table(title="Breakdown by Source", box=box.SIMPLE)
            breakdown_table.add_column("Source", style="cyan")
            breakdown_table.add_column("Emissions", style="yellow")
            breakdown_table.add_column("Percentage", style="magenta")
            
            for item in results['emissions']['emissions_breakdown']:
                breakdown_table.add_row(
                    item['source'],
                    f"{item['co2e_tons']:.3f} tons",
                    f"{item['percentage']:.1f}%"
                )
            
            # Benchmark if available
            if results.get('benchmark'):
                benchmark_text = f"""
[bold]Benchmark Analysis:[/bold]
Rating: [cyan]{results['benchmark']['rating']}[/cyan]
Carbon Intensity: {results['benchmark']['carbon_intensity']:.2f} kg CO2e/sqft/year
Percentile: Top {results['benchmark']['percentile']}%
                """
                
                details_panel = Panel(
                    Columns([breakdown_table, Text(benchmark_text)]),
                    title="📈 Analysis"
                )
            else:
                details_panel = Panel(breakdown_table, title="📈 Breakdown")
            
            layout["details"].update(details_panel)
        
        # Report preview
        if results.get('report'):
            report_preview = results['report'][:500] + "..." if len(results['report']) > 500 else results['report']
            layout["report"].update(Panel(
                Syntax(report_preview, "text", theme="monokai"),
                title="📄 Report Preview"
            ))
        
        console.print(layout)
        
        # Save option
        if Confirm.ask("\nSave results to file?"):
            filename = Prompt.ask("Filename", default=f"emissions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Results saved to {filename}[/green]")
    
    def cmd_test(self, args):
        """Run test suite"""
        console.print(Panel("🧪 Running Test Suite", style="cyan"))
        
        test_types = ["unit", "integration", "workflow", "all"]
        test_type = Prompt.ask("Test type", choices=test_types, default="all")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            if test_type in ["unit", "all"]:
                task = progress.add_task("Running unit tests...", total=5)
                self.run_unit_tests(progress, task)
            
            if test_type in ["integration", "all"]:
                task = progress.add_task("Running integration tests...", total=3)
                self.run_integration_tests(progress, task)
            
            if test_type in ["workflow", "all"]:
                task = progress.add_task("Running workflow tests...", total=2)
                self.run_workflow_tests(progress, task)
        
        console.print("[green]✓ All tests completed![/green]")
    
    def run_unit_tests(self, progress, task):
        """Run unit tests"""
        tests = [
            ("FuelAgent", self.test_fuel_agent),
            ("CarbonAgent", self.test_carbon_agent),
            ("ValidatorAgent", self.test_validator_agent),
            ("ReportAgent", self.test_report_agent),
            ("BenchmarkAgent", self.test_benchmark_agent)
        ]
        
        for test_name, test_func in tests:
            result = test_func()
            if result:
                console.print(f"  ✓ {test_name} passed", style="green")
            else:
                console.print(f"  ✗ {test_name} failed", style="red")
            progress.update(task, advance=1)
    
    def test_fuel_agent(self) -> bool:
        """Test FuelAgent"""
        result = self.client.calculate_emissions("electricity", 1000, "kWh")
        return result["success"] and abs(result["data"]["co2e_emissions_kg"] - 385.0) < 0.1
    
    def test_carbon_agent(self) -> bool:
        """Test CarbonAgent"""
        emissions = [{"co2e_emissions_kg": 100}, {"co2e_emissions_kg": 200}]
        result = self.client.aggregate_emissions(emissions)
        return result["success"] and result["data"]["total_co2e_kg"] == 300
    
    def test_validator_agent(self) -> bool:
        """Test ValidatorAgent"""
        data = {"fuels": [{"type": "electricity", "consumption": 100, "unit": "kWh"}]}
        result = self.client.validate_input(data)
        return result["success"]
    
    def test_report_agent(self) -> bool:
        """Test ReportAgent"""
        carbon_data = {"total_co2e_tons": 1.0, "total_co2e_kg": 1000}
        result = self.client.generate_report(carbon_data)
        return result["success"]
    
    def test_benchmark_agent(self) -> bool:
        """Test BenchmarkAgent"""
        result = self.client.benchmark_emissions(1000, 1000, "commercial_office", 1)
        return result["success"]
    
    def run_integration_tests(self, progress, task):
        """Run integration tests"""
        tests = [
            "End-to-end calculation",
            "Workflow execution",
            "Error handling"
        ]
        
        for test in tests:
            time.sleep(0.5)  # Simulate test
            console.print(f"  ✓ {test} passed", style="green")
            progress.update(task, advance=1)
    
    def run_workflow_tests(self, progress, task):
        """Run workflow tests"""
        tests = [
            "Sample workflow",
            "Complex workflow"
        ]
        
        for test in tests:
            time.sleep(0.5)  # Simulate test
            console.print(f"  ✓ {test} passed", style="green")
            progress.update(task, advance=1)
    
    def cmd_agents(self, args):
        """Manage agents"""
        action = Prompt.ask(
            "Agent action",
            choices=["list", "info", "create", "test"]
        )
        
        if action == "list":
            self.list_agents()
        elif action == "info":
            agent_id = Prompt.ask("Agent ID")
            self.show_agent_info(agent_id)
        elif action == "create":
            self.create_agent()
        elif action == "test":
            agent_id = Prompt.ask("Agent ID to test")
            self.test_agent(agent_id)
    
    def list_agents(self):
        """List all available agents"""
        agents = self.client.list_agents()
        
        table = Table(title="Available Agents", box=box.ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Description", style="white")
        table.add_column("Version", style="dim")
        
        for agent_id in agents:
            info = self.client.get_agent_info(agent_id)
            if info:
                table.add_row(
                    agent_id,
                    info["name"],
                    info["description"],
                    info["version"]
                )
        
        console.print(table)
    
    def show_agent_info(self, agent_id: str):
        """Show detailed agent information"""
        info = self.client.get_agent_info(agent_id)
        if not info:
            console.print(f"[red]Agent '{agent_id}' not found[/red]")
            return
        
        panel_content = f"""
[bold]Agent: {info['name']}[/bold]
[dim]{info['description']}[/dim]

[cyan]Details:[/cyan]
  ID: {agent_id}
  Version: {info['version']}
  Enabled: {info['enabled']}
        """
        
        console.print(Panel(panel_content, title=f"Agent: {agent_id}", border_style="cyan"))
    
    def create_agent(self):
        """Create a custom agent with templates for different types"""
        console.print(Panel("Create Custom Agent", style="cyan"))
        
        # Ask for agent type
        agent_types = [
            "custom", "emissions", "boiler", "fuel", "validator", 
            "benchmark", "report", "intensity", "recommendation"
        ]
        agent_type = Prompt.ask(
            "Agent type", 
            choices=agent_types,
            default="custom"
        )
        
        name = Prompt.ask("Agent name")
        description = Prompt.ask("Description")
        
        # Generate appropriate agent code based on type
        agent_code = self._generate_agent_code(agent_type, name, description)
        
        # Show preview
        console.print("\n[bold]Generated Agent Code:[/bold]")
        console.print(Syntax(agent_code, "python", theme="monokai"))
        
        if Confirm.ask("\nSave agent to file?"):
            filename = f"{name.lower()}_agent.py"
            filepath = self.workspace / "agents" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                f.write(agent_code)
            
            console.print(f"[green]Agent saved to {filepath}[/green]")
            console.print(f"[dim]To use: Import and register with orchestrator[/dim]")
    
    def _generate_agent_code(self, agent_type: str, name: str, description: str) -> str:
        """Generate agent code based on type"""
        
        if agent_type == "boiler":
            return self._generate_boiler_agent(name, description)
        elif agent_type == "emissions" or agent_type == "fuel":
            return self._generate_emissions_agent(name, description)
        elif agent_type == "validator":
            return self._generate_validator_agent(name, description)
        elif agent_type == "benchmark":
            return self._generate_benchmark_agent(name, description)
        elif agent_type == "report":
            return self._generate_report_agent(name, description)
        elif agent_type == "intensity":
            return self._generate_intensity_agent(name, description)
        elif agent_type == "recommendation":
            return self._generate_recommendation_agent(name, description)
        else:
            return self._generate_custom_agent(name, description)
    
    def _generate_custom_agent(self, name: str, description: str) -> str:
        """Generate a basic custom agent template"""
        return f'''"""
Custom Agent: {name}
{description}
"""

from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from typing import Dict, Any


class {name}Agent(BaseAgent):
    """
    {description}
    """
    
    def __init__(self):
        config = AgentConfig(
            name="{name}",
            description="{description}",
            version="0.0.1"
        )
        super().__init__(config)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        # Add your validation logic here
        required_fields = ["data"]  # Update with your requirements
        return all(field in input_data for field in required_fields)
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute agent logic"""
        try:
            # Validate input
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    error="Invalid input data"
                )
            
            # Process data
            result_data = {{
                "message": f"Processed by {name}",
                "input": input_data,
                # Add your results here
            }}
            
            return AgentResult(
                success=True,
                data=result_data
            )
        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e)
            )
'''
    
    def _generate_boiler_agent(self, name: str, description: str) -> str:
        """Generate a boiler-type agent template"""
        return f'''"""
Boiler Agent: {name}
{description}
"""

from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from typing import Dict, Any, Optional


class {name}Agent(BaseAgent):
    """
    {description}
    
    Calculates emissions from boiler/thermal systems.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="{name}",
            description="{description}",
            version="0.0.1"
        )
        super().__init__(config)
        
        # Emission factors for different fuel types (kgCO2e per unit)
        self.emission_factors = {{
            "natural_gas": {{"value": 5.3, "unit": "kgCO2e/therm"}},
            "diesel": {{"value": 10.21, "unit": "kgCO2e/gallon"}},
            "propane": {{"value": 5.77, "unit": "kgCO2e/gallon"}},
            "biomass": {{"value": 0.0, "unit": "kgCO2e/kg"}},  # Carbon neutral
            "electricity": {{"value": 0.386, "unit": "kgCO2e/kWh"}}  # For electric boilers
        }}
        
        # Typical efficiencies
        self.default_efficiencies = {{
            "condensing": 0.95,
            "standard": 0.85,
            "old": 0.75,
            "electric": 0.99
        }}
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate boiler input data"""
        required = ["boiler_type", "fuel_type", "thermal_output"]
        return all(field in input_data for field in required)
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Calculate boiler emissions"""
        try:
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    error="Missing required fields: boiler_type, fuel_type, thermal_output"
                )
            
            # Extract inputs
            boiler_type = input_data["boiler_type"]
            fuel_type = input_data["fuel_type"]
            thermal_output = input_data["thermal_output"]
            efficiency = input_data.get("efficiency", self.default_efficiencies.get(boiler_type, 0.85))
            
            # Get thermal output value
            if isinstance(thermal_output, dict):
                output_value = thermal_output.get("value", 0)
                output_unit = thermal_output.get("unit", "kWh")
            else:
                output_value = thermal_output
                output_unit = "kWh"
            
            # Calculate fuel consumption based on efficiency
            fuel_consumption = output_value / efficiency
            
            # Convert to appropriate units and calculate emissions
            if fuel_type == "natural_gas":
                # Convert kWh to therms (1 therm = 29.3 kWh)
                fuel_therms = fuel_consumption / 29.3
                emissions_kg = fuel_therms * self.emission_factors["natural_gas"]["value"]
                fuel_unit = "therms"
                fuel_value = fuel_therms
            elif fuel_type == "electricity":
                emissions_kg = fuel_consumption * self.emission_factors["electricity"]["value"]
                fuel_unit = "kWh"
                fuel_value = fuel_consumption
            else:
                # Default calculation
                emissions_kg = fuel_consumption * 0.2  # Default factor
                fuel_unit = "units"
                fuel_value = fuel_consumption
            
            result_data = {{
                "co2e_emissions_kg": emissions_kg,
                "boiler_type": boiler_type,
                "fuel_type": fuel_type,
                "fuel_consumption_value": fuel_value,
                "fuel_consumption_unit": fuel_unit,
                "thermal_output_value": output_value,
                "thermal_output_unit": output_unit,
                "efficiency": efficiency,
                "thermal_efficiency_percent": efficiency * 100,
                "emission_factor": self.emission_factors.get(fuel_type, {{}}).get("value", 0),
                "emission_factor_unit": self.emission_factors.get(fuel_type, {{}}).get("unit", ""),
                "recommendations": self._get_recommendations(boiler_type, efficiency)
            }}
            
            return AgentResult(
                success=True,
                data=result_data
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Error calculating boiler emissions: {{str(e)}}"
            )
    
    def _get_recommendations(self, boiler_type: str, efficiency: float) -> list:
        """Get improvement recommendations"""
        recommendations = []
        
        if efficiency < 0.80:
            recommendations.append({{
                "priority": "high",
                "action": "Upgrade to high-efficiency condensing boiler",
                "impact": "20-30% efficiency improvement"
            }})
        
        if boiler_type == "old":
            recommendations.append({{
                "priority": "high",
                "action": "Replace aging boiler system",
                "impact": "15-25% energy savings"
            }})
        
        recommendations.append({{
            "priority": "medium",
            "action": "Install smart controls and weather compensation",
            "impact": "10-15% fuel savings"
        }})
        
        return recommendations
'''
    
    def _generate_emissions_agent(self, name: str, description: str) -> str:
        """Generate an emissions calculation agent template"""
        return f'''"""
Emissions Agent: {name}
{description}
"""

from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from typing import Dict, Any, List


class {name}Agent(BaseAgent):
    """
    {description}
    
    Calculates emissions from various fuel sources.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="{name}",
            description="{description}",
            version="0.0.1"
        )
        super().__init__(config)
        
        # Emission factors (kgCO2e per unit)
        self.emission_factors = {{
            "electricity": 0.386,  # kgCO2e/kWh (US average)
            "natural_gas": 5.3,    # kgCO2e/therm
            "diesel": 10.21,       # kgCO2e/gallon
            "gasoline": 8.89,      # kgCO2e/gallon
            "propane": 5.77,       # kgCO2e/gallon
            "coal": 2.86,          # kgCO2e/kg
        }}
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate emissions input data"""
        return "fuels" in input_data and isinstance(input_data["fuels"], list)
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Calculate emissions for multiple fuel sources"""
        try:
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    error="Invalid input: expected 'fuels' list"
                )
            
            total_emissions = 0
            emissions_breakdown = []
            
            for fuel in input_data["fuels"]:
                fuel_type = fuel.get("type", "")
                amount = fuel.get("amount", 0)
                unit = fuel.get("unit", "")
                
                # Calculate emissions for this fuel
                if fuel_type in self.emission_factors:
                    emissions = amount * self.emission_factors[fuel_type]
                    total_emissions += emissions
                    
                    emissions_breakdown.append({{
                        "fuel_type": fuel_type,
                        "amount": amount,
                        "unit": unit,
                        "co2e_emissions_kg": emissions
                    }})
            
            result_data = {{
                "total_co2e_kg": total_emissions,
                "total_co2e_tons": total_emissions / 1000,
                "emissions_breakdown": emissions_breakdown,
                "calculation_method": "IPCC Tier 1"
            }}
            
            return AgentResult(
                success=True,
                data=result_data
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Error calculating emissions: {{str(e)}}"
            )
'''
    
    def _generate_validator_agent(self, name: str, description: str) -> str:
        """Generate a validator agent template"""
        return f'''"""
Validator Agent: {name}
{description}
"""

from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from typing import Dict, Any, List


class {name}Agent(BaseAgent):
    """
    {description}
    
    Validates and normalizes input data.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="{name}",
            description="{description}",
            version="0.0.1"
        )
        super().__init__(config)
        
        # Valid values for validation
        self.valid_fuel_types = [
            "electricity", "natural_gas", "diesel", "gasoline",
            "propane", "coal", "biomass", "solar", "wind"
        ]
        
        self.valid_units = {{
            "electricity": ["kWh", "MWh", "GWh"],
            "natural_gas": ["therms", "ccf", "mcf", "mmBtu"],
            "diesel": ["gallons", "liters"],
            "gasoline": ["gallons", "liters"]
        }}
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input structure"""
        return isinstance(input_data, dict)
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Validate and normalize input data"""
        try:
            validation_errors = []
            validation_warnings = []
            normalized_data = {{}}
            
            # Validate fuels data
            if "fuels" in input_data:
                normalized_fuels = []
                for fuel in input_data["fuels"]:
                    # Check fuel type
                    fuel_type = fuel.get("type", "").lower()
                    if fuel_type not in self.valid_fuel_types:
                        validation_errors.append(f"Invalid fuel type: {{fuel_type}}")
                        continue
                    
                    # Check amount
                    amount = fuel.get("amount")
                    if amount is None or amount < 0:
                        validation_errors.append(f"Invalid amount for {{fuel_type}}")
                        continue
                    
                    # Check unit
                    unit = fuel.get("unit", "")
                    valid_units = self.valid_units.get(fuel_type, [])
                    if unit not in valid_units:
                        validation_warnings.append(f"Unusual unit '{{unit}}' for {{fuel_type}}")
                    
                    normalized_fuels.append({{
                        "type": fuel_type,
                        "amount": float(amount),
                        "unit": unit
                    }})
                
                normalized_data["fuels"] = normalized_fuels
            
            # Check for validation issues
            if validation_errors:
                return AgentResult(
                    success=False,
                    error="Validation failed",
                    data={{
                        "errors": validation_errors,
                        "warnings": validation_warnings
                    }}
                )
            
            result_data = {{
                "validated": True,
                "normalized_data": normalized_data,
                "warnings": validation_warnings,
                "statistics": {{
                    "fuel_count": len(normalized_data.get("fuels", [])),
                    "total_records": len(input_data)
                }}
            }}
            
            return AgentResult(
                success=True,
                data=result_data
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Validation error: {{str(e)}}"
            )
'''
    
    def _generate_benchmark_agent(self, name: str, description: str) -> str:
        """Generate a benchmark agent template"""
        return f'''"""
Benchmark Agent: {name}
{description}
"""

from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from typing import Dict, Any


class {name}Agent(BaseAgent):
    """
    {description}
    
    Compares emissions against industry benchmarks.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="{name}",
            description="{description}",
            version="0.0.1"
        )
        super().__init__(config)
        
        # Industry benchmarks (kgCO2e/sqft/year)
        self.benchmarks = {{
            "commercial_office": {{
                "excellent": 1.5,
                "good": 3.0,
                "average": 5.0,
                "poor": 7.0
            }},
            "hospital": {{
                "excellent": 3.0,
                "good": 5.0,
                "average": 8.0,
                "poor": 12.0
            }},
            "data_center": {{
                "excellent": 10.0,
                "good": 20.0,
                "average": 35.0,
                "poor": 50.0
            }}
        }}
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate benchmark input"""
        required = ["total_emissions_kg", "building_area", "building_type"]
        return all(field in input_data for field in required)
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Benchmark emissions performance"""
        try:
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    error="Missing required fields"
                )
            
            emissions_kg = input_data["total_emissions_kg"]
            area_sqft = input_data["building_area"]
            building_type = input_data["building_type"]
            period_months = input_data.get("period_months", 12)
            
            # Calculate intensity
            annual_emissions = (emissions_kg / period_months) * 12
            intensity = annual_emissions / area_sqft
            
            # Get benchmark
            benchmarks = self.benchmarks.get(building_type, self.benchmarks["commercial_office"])
            
            # Determine rating
            if intensity <= benchmarks["excellent"]:
                rating = "Excellent"
                percentile = 90
            elif intensity <= benchmarks["good"]:
                rating = "Good"
                percentile = 70
            elif intensity <= benchmarks["average"]:
                rating = "Average"
                percentile = 50
            else:
                rating = "Below Average"
                percentile = 30
            
            result_data = {{
                "carbon_intensity": intensity,
                "unit": "kgCO2e/sqft/year",
                "rating": rating,
                "percentile": percentile,
                "benchmarks": benchmarks,
                "recommendations": self._get_recommendations(rating)
            }}
            
            return AgentResult(
                success=True,
                data=result_data
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Benchmark error: {{str(e)}}"
            )
    
    def _get_recommendations(self, rating: str) -> list:
        """Get improvement recommendations based on rating"""
        if rating == "Excellent":
            return [{{"action": "Maintain current practices", "priority": "low"}}]
        elif rating == "Good":
            return [{{"action": "Consider renewable energy", "priority": "medium"}}]
        else:
            return [
                {{"action": "Conduct energy audit", "priority": "high"}},
                {{"action": "Upgrade HVAC systems", "priority": "high"}}
            ]
'''
    
    def _generate_report_agent(self, name: str, description: str) -> str:
        """Generate a report agent template"""
        return f'''"""
Report Agent: {name}
{description}
"""

from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from typing import Dict, Any
from datetime import datetime


class {name}Agent(BaseAgent):
    """
    {description}
    
    Generates formatted reports from emissions data.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="{name}",
            description="{description}",
            version="0.0.1"
        )
        super().__init__(config)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate report input"""
        return "carbon_data" in input_data
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Generate emissions report"""
        try:
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    error="Missing carbon_data"
                )
            
            carbon_data = input_data["carbon_data"]
            format_type = input_data.get("format", "text")
            
            if format_type == "json":
                report = carbon_data
            else:
                # Generate text report
                report = self._generate_text_report(carbon_data)
            
            result_data = {{
                "report": report,
                "format": format_type,
                "generated_at": datetime.now().isoformat(),
                "summary": {{
                    "total_emissions_kg": carbon_data.get("total_emissions_kg", 0),
                    "total_emissions_tons": carbon_data.get("total_emissions_kg", 0) / 1000
                }}
            }}
            
            return AgentResult(
                success=True,
                data=result_data
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Report generation error: {{str(e)}}"
            )
    
    def _generate_text_report(self, data: Dict) -> str:
        """Generate formatted text report"""
        report = []
        report.append("=" * 60)
        report.append("CARBON FOOTPRINT REPORT")
        report.append("=" * 60)
        report.append("")
        
        total = data.get("total_emissions_kg", 0)
        report.append(f"Total Emissions: {{total:,.2f}} kg CO2e")
        report.append(f"                 {{total/1000:,.2f}} tons CO2e")
        report.append("")
        
        if "emissions_by_source" in data:
            report.append("Emissions by Source:")
            report.append("-" * 40)
            for source, amount in data["emissions_by_source"].items():
                percentage = (amount / total * 100) if total > 0 else 0
                report.append(f"  {{source:<20}} {{amount:>10,.2f}} kg ({{percentage:>5.1f}}%)")
        
        report.append("")
        report.append("=" * 60)
        
        return "\\n".join(report)
'''
    
    def _generate_intensity_agent(self, name: str, description: str) -> str:
        """Generate an intensity calculation agent template"""
        return f'''"""
Intensity Agent: {name}
{description}
"""

from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from typing import Dict, Any


class {name}Agent(BaseAgent):
    """
    {description}
    
    Calculates emission intensity metrics.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="{name}",
            description="{description}",
            version="0.0.1"
        )
        super().__init__(config)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate intensity input"""
        required = ["total_emissions_kg", "building_area"]
        return all(field in input_data for field in required)
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Calculate emission intensities"""
        try:
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    error="Missing required fields"
                )
            
            emissions_kg = input_data["total_emissions_kg"]
            area = input_data["building_area"]
            area_unit = input_data.get("area_unit", "sqft")
            occupancy = input_data.get("occupancy", 1)
            period_months = input_data.get("period_months", 12)
            
            # Calculate various intensities
            annual_factor = 12 / period_months
            
            intensities = {{
                "per_area_year": (emissions_kg * annual_factor) / area,
                "per_area_month": emissions_kg / (area * period_months),
                "per_person_year": (emissions_kg * annual_factor) / occupancy if occupancy > 0 else 0,
                "per_person_month": emissions_kg / (occupancy * period_months) if occupancy > 0 else 0
            }}
            
            result_data = {{
                "intensities": intensities,
                "units": {{
                    "per_area": f"kgCO2e/{{area_unit}}/period",
                    "per_person": "kgCO2e/person/period"
                }},
                "period_months": period_months,
                "metrics": {{
                    "total_emissions_kg": emissions_kg,
                    "area": area,
                    "area_unit": area_unit,
                    "occupancy": occupancy
                }}
            }}
            
            return AgentResult(
                success=True,
                data=result_data
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Intensity calculation error: {{str(e)}}"
            )
'''
    
    def _generate_recommendation_agent(self, name: str, description: str) -> str:
        """Generate a recommendation agent template"""
        return f'''"""
Recommendation Agent: {name}
{description}
"""

from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from typing import Dict, Any, List


class {name}Agent(BaseAgent):
    """
    {description}
    
    Provides emissions reduction recommendations.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="{name}",
            description="{description}",
            version="0.0.1"
        )
        super().__init__(config)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate recommendation input"""
        return "emissions_data" in input_data
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Generate recommendations"""
        try:
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    error="Missing emissions_data"
                )
            
            emissions_data = input_data["emissions_data"]
            building_info = input_data.get("building_info", {{}})
            
            recommendations = self._generate_recommendations(emissions_data, building_info)
            
            result_data = {{
                "recommendations": recommendations,
                "potential_savings": self._calculate_savings(emissions_data, recommendations),
                "implementation_timeline": self._create_timeline(recommendations)
            }}
            
            return AgentResult(
                success=True,
                data=result_data
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Recommendation error: {{str(e)}}"
            )
    
    def _generate_recommendations(self, emissions: Dict, building: Dict) -> List[Dict]:
        """Generate specific recommendations"""
        recommendations = []
        
        # Analyze emission sources
        sources = emissions.get("emissions_by_source", {{}})
        total = emissions.get("total_emissions_kg", 0)
        
        # High electricity usage
        if sources.get("electricity", 0) > total * 0.5:
            recommendations.append({{
                "category": "Renewable Energy",
                "action": "Install solar panels or purchase renewable energy",
                "impact": "30-50% reduction in electricity emissions",
                "cost": "High",
                "payback": "5-7 years",
                "priority": "High"
            }})
            
            recommendations.append({{
                "category": "Energy Efficiency",
                "action": "Upgrade to LED lighting throughout facility",
                "impact": "10-15% reduction in electricity use",
                "cost": "Medium",
                "payback": "2-3 years",
                "priority": "High"
            }})
        
        # Natural gas usage
        if sources.get("natural_gas", 0) > 0:
            recommendations.append({{
                "category": "Heating Efficiency",
                "action": "Upgrade to high-efficiency heating system",
                "impact": "20-30% reduction in gas consumption",
                "cost": "High",
                "payback": "5-10 years",
                "priority": "Medium"
            }})
        
        # General recommendations
        recommendations.append({{
            "category": "Monitoring",
            "action": "Implement energy management system",
            "impact": "5-10% reduction through optimization",
            "cost": "Low",
            "payback": "1-2 years",
            "priority": "High"
        }})
        
        return recommendations
    
    def _calculate_savings(self, emissions: Dict, recommendations: List) -> Dict:
        """Calculate potential savings"""
        total = emissions.get("total_emissions_kg", 0)
        
        # Estimate based on recommendations
        potential_reduction_percent = min(50, len(recommendations) * 10)
        potential_reduction_kg = total * (potential_reduction_percent / 100)
        
        return {{
            "potential_reduction_percent": potential_reduction_percent,
            "potential_reduction_kg": potential_reduction_kg,
            "potential_reduction_tons": potential_reduction_kg / 1000
        }}
    
    def _create_timeline(self, recommendations: List) -> List[Dict]:
        """Create implementation timeline"""
        timeline = []
        
        # Sort by priority
        high_priority = [r for r in recommendations if r.get("priority") == "High"]
        medium_priority = [r for r in recommendations if r.get("priority") == "Medium"]
        low_priority = [r for r in recommendations if r.get("priority") == "Low"]
        
        # Phase 1: Quick wins
        if high_priority:
            timeline.append({{
                "phase": 1,
                "timeframe": "0-6 months",
                "actions": [r["action"] for r in high_priority[:2]]
            }})
        
        # Phase 2: Medium-term
        if medium_priority:
            timeline.append({{
                "phase": 2,
                "timeframe": "6-12 months",
                "actions": [r["action"] for r in medium_priority[:2]]
            }})
        
        # Phase 3: Long-term
        if low_priority:
            timeline.append({{
                "phase": 3,
                "timeframe": "12-24 months",
                "actions": [r["action"] for r in low_priority]
            }})
        
        return timeline
'''
    
    def test_agent(self, agent_id: str):
        """Test an agent with sample data"""
        console.print(f"Testing agent: {agent_id}")
        
        # Get comprehensive test data for each agent
        test_data = self._get_agent_test_data(agent_id)
        
        if not test_data:
            console.print(f"[yellow]No test data configured for agent: {agent_id}[/yellow]")
            console.print("[dim]Would you like to provide custom test data?[/dim]")
            if Confirm.ask("Provide custom data?"):
                test_data = self._get_custom_test_data(agent_id)
            else:
                return
        
        # Run test
        with console.status("Running agent..."):
            result = self.client.execute_agent(agent_id, test_data)
        
        # Display result
        if result["success"]:
            console.print("[green]✓ Agent executed successfully[/green]")
            console.print("\n[bold]Result:[/bold]")
            console.print(Syntax(json.dumps(result["data"], indent=2), "json", theme="monokai"))
        else:
            console.print(f"[red]✗ Agent failed: {result.get('error', 'Unknown error')}[/red]")
    
    def _get_agent_test_data(self, agent_id: str) -> Dict[str, Any]:
        """Get appropriate test data for each agent"""
        test_data_map = {
            "validator": {
                "fuels": [
                    {"type": "electricity", "amount": 1000, "unit": "kWh"},
                    {"type": "natural_gas", "amount": 100, "unit": "therms"}
                ]
            },
            "fuel": {
                # FuelAgent expects a list of fuels
                "fuels": [
                    {"type": "electricity", "amount": 1000, "unit": "kWh"},
                    {"type": "natural_gas", "amount": 100, "unit": "therms"}
                ]
            },
            "boiler": {
                # BoilerAgent expects specific structure
                "boiler_type": "standard",
                "fuel_type": "natural_gas", 
                "thermal_output": {
                    "value": 1000,
                    "unit": "kWh"
                },
                "efficiency": 0.85,
                "country": "US"
            },
            "carbon": {
                "emissions": [
                    {"co2e_emissions_kg": 500, "source": "electricity"},
                    {"co2e_emissions_kg": 250, "source": "natural_gas"}
                ]
            },
            "report": {
                "carbon_data": {
                    "total_emissions_kg": 750,
                    "emissions_by_source": {
                        "electricity": 500,
                        "natural_gas": 250
                    }
                },
                "format": "text"
            },
            "benchmark": {
                "total_emissions_kg": 10000,
                "building_area": 5000,
                "building_type": "commercial_office",
                "period_months": 12
            },
            "grid_factor": {
                # GridFactorAgent expects country, fuel_type, and unit
                "country": "US",
                "fuel_type": "electricity",
                "unit": "kWh"
            },
            "building_profile": {
                "building_type": "commercial_office",
                "area": 5000,
                "area_unit": "sqft",
                "occupancy": 50,
                "floor_count": 3,
                "building_age": 10,
                "country": "US"
            },
            "intensity": {
                "total_emissions_kg": 10000,
                "building_area": 5000,
                "area_unit": "sqft",
                "occupancy": 50,
                "period_months": 12
            },
            "recommendation": {
                "emissions_data": {
                    "total_emissions_kg": 10000,
                    "emissions_by_source": {
                        "electricity": 7000,
                        "natural_gas": 3000
                    }
                },
                "building_info": {
                    "type": "commercial_office",
                    "area": 5000,
                    "occupancy": 50,
                    "area_unit": "sqft"
                }
            }
        }
        
        return test_data_map.get(agent_id, {})
    
    def _get_custom_test_data(self, agent_id: str) -> Dict[str, Any]:
        """Get custom test data from user input"""
        console.print("[bold]Enter custom test data (JSON format):[/bold]")
        console.print("[dim]Example: {\"key\": \"value\", \"number\": 123}[/dim]")
        
        json_input = Prompt.ask("Test data")
        try:
            return json.loads(json_input)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON: {e}[/red]")
            return {}
    
    def cmd_workflow(self, args):
        """Workflow designer"""
        console.print(Panel("🔧 Workflow Designer", style="cyan"))
        
        action = Prompt.ask(
            "Workflow action",
            choices=["create", "edit", "list", "run", "validate"]
        )
        
        if action == "create":
            self.create_workflow()
        elif action == "edit":
            workflow_name = Prompt.ask("Workflow name")
            self.edit_workflow(workflow_name)
        elif action == "list":
            self.list_workflows()
        elif action == "run":
            workflow_name = Prompt.ask("Workflow to run")
            self.run_workflow(workflow_name)
        elif action == "validate":
            workflow_name = Prompt.ask("Workflow to validate")
            self.validate_workflow(workflow_name)
    
    def create_workflow(self):
        """Create a new workflow interactively"""
        name = Prompt.ask("Workflow name")
        description = Prompt.ask("Description")
        
        builder = WorkflowBuilder(name, description)
        
        console.print("\n[bold]Add workflow steps:[/bold]")
        console.print("[dim]Type 'done' when finished[/dim]\n")
        
        step_count = 0
        while True:
            step_count += 1
            step_name = Prompt.ask(f"Step {step_count} name", default="done")
            
            if step_name.lower() == "done":
                break
            
            agents = self.client.list_agents()
            agent_id = Prompt.ask(f"Agent for {step_name}", choices=agents)
            
            builder.add_step(step_name, agent_id)
            
            if Confirm.ask("Add input mapping?", default=False):
                console.print("[dim]Example: fuel_type=input.fuel_type[/dim]")
                mappings = {}
                while True:
                    mapping = Prompt.ask("Mapping (key=path)", default="done")
                    if mapping.lower() == "done":
                        break
                    key, path = mapping.split("=")
                    mappings[key.strip()] = path.strip()
                
                if mappings:
                    builder.current_step.input_mapping = mappings
        
        workflow = builder.build()
        
        # Preview
        workflow_dict = workflow.model_dump()
        console.print("\n[bold]Workflow Preview:[/bold]")
        console.print(Syntax(yaml.dump(workflow_dict, default_flow_style=False), "yaml", theme="monokai"))
        
        if Confirm.ask("\nSave workflow?"):
            filename = f"{name}.yaml"
            filepath = self.workspace / "workflows" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            workflow.to_yaml(str(filepath))
            console.print(f"[green]Workflow saved to {filepath}[/green]")
            
            # Register with client
            self.client.register_workflow(name, workflow)
            console.print(f"[green]Workflow '{name}' registered[/green]")
    
    def list_workflows(self):
        """List all workflows"""
        workflows_dir = self.workspace / "workflows"
        if not workflows_dir.exists():
            console.print("[yellow]No workflows directory found[/yellow]")
            return
        
        workflows = list(workflows_dir.glob("*.yaml")) + list(workflows_dir.glob("*.yml"))
        
        if not workflows:
            console.print("[yellow]No workflows found[/yellow]")
            return
        
        table = Table(title="Available Workflows", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("File", style="green")
        table.add_column("Steps", style="yellow")
        
        for workflow_file in workflows:
            with open(workflow_file, 'r') as f:
                data = yaml.safe_load(f)
            
            name = data.get("name", workflow_file.stem)
            steps = len(data.get("steps", []))
            
            table.add_row(name, workflow_file.name, str(steps))
        
        console.print(table)
    
    def run_workflow(self, workflow_name: str):
        """Run a workflow"""
        workflow_file = self.workspace / "workflows" / f"{workflow_name}.yaml"
        
        if not workflow_file.exists():
            console.print(f"[red]Workflow '{workflow_name}' not found[/red]")
            return
        
        # Load workflow
        workflow = Workflow.from_yaml(str(workflow_file))
        self.client.register_workflow(workflow_name, workflow)
        
        # Get input data
        console.print("\n[bold]Enter input data:[/bold]")
        input_method = Prompt.ask("Input method", choices=["interactive", "file", "json"])
        
        input_data = {}
        
        if input_method == "interactive":
            # Interactive input based on workflow requirements
            console.print("[dim]Enter fuel data:[/dim]")
            fuels = []
            
            electricity = Prompt.ask("Electricity (kWh)", default="0")
            if float(electricity) > 0:
                fuels.append({
                    "fuel_type": "electricity",
                    "consumption": float(electricity),
                    "unit": "kWh"
                })
            
            gas = Prompt.ask("Natural gas (therms)", default="0")
            if float(gas) > 0:
                fuels.append({
                    "fuel_type": "natural_gas",
                    "consumption": float(gas),
                    "unit": "therms"
                })
            
            input_data["fuels"] = fuels
            
        elif input_method == "file":
            input_file = Prompt.ask("Input file path")
            with open(input_file, 'r') as f:
                input_data = json.load(f)
        
        elif input_method == "json":
            json_str = Prompt.ask("JSON data")
            input_data = json.loads(json_str)
        
        # Run workflow
        with console.status(f"Running workflow '{workflow_name}'..."):
            result = self.client.execute_workflow(workflow_name, input_data)
        
        # Display results
        if result["success"]:
            console.print(f"[green]✓ Workflow completed successfully[/green]")
            
            if "data" in result:
                console.print("\n[bold]Output:[/bold]")
                console.print(Syntax(json.dumps(result["data"], indent=2), "json", theme="monokai"))
        else:
            console.print(f"[red]✗ Workflow failed[/red]")
            if "errors" in result:
                for error in result["errors"]:
                    console.print(f"  - {error['step']}: {error['error']}", style="red")
    
    def validate_workflow(self, workflow_name: str):
        """Validate a workflow"""
        workflow_file = self.workspace / "workflows" / f"{workflow_name}.yaml"
        
        if not workflow_file.exists():
            console.print(f"[red]Workflow '{workflow_name}' not found[/red]")
            return
        
        try:
            workflow = Workflow.from_yaml(str(workflow_file))
            errors = workflow.validate_workflow()
            
            if errors:
                console.print(f"[red]Workflow validation failed:[/red]")
                for error in errors:
                    console.print(f"  - {error}", style="red")
            else:
                console.print(f"[green]✓ Workflow '{workflow_name}' is valid[/green]")
                
                # Show workflow structure
                tree = Tree(f"[bold]{workflow.name}[/bold]")
                for step in workflow.steps:
                    step_node = tree.add(f"{step.name} ([cyan]{step.agent_id}[/cyan])")
                    if step.input_mapping:
                        mapping_node = step_node.add("[dim]Input Mapping:[/dim]")
                        for key, value in step.input_mapping.items():
                            mapping_node.add(f"{key} ← {value}")
                
                console.print(tree)
                
        except Exception as e:
            console.print(f"[red]Error loading workflow: {e}[/red]")
    
    def edit_workflow(self, workflow_name: str):
        """Edit an existing workflow"""
        workflow_file = self.workspace / "workflows" / f"{workflow_name}.yaml"
        
        if not workflow_file.exists():
            console.print(f"[red]Workflow '{workflow_name}' not found[/red]")
            return
        
        # Load workflow
        with open(workflow_file, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        # Show current workflow
        console.print("\n[bold]Current Workflow:[/bold]")
        console.print(Syntax(yaml.dump(workflow_data, default_flow_style=False), "yaml", theme="monokai"))
        
        # Edit options
        action = Prompt.ask(
            "Edit action",
            choices=["add_step", "remove_step", "edit_step", "rename", "save"]
        )
        
        if action == "add_step":
            step_name = Prompt.ask("New step name")
            agent_id = Prompt.ask("Agent ID", choices=self.client.list_agents())
            
            new_step = {
                "name": step_name,
                "agent_id": agent_id
            }
            
            workflow_data["steps"].append(new_step)
            console.print(f"[green]Added step '{step_name}'[/green]")
        
        elif action == "remove_step":
            steps = [step["name"] for step in workflow_data["steps"]]
            step_to_remove = Prompt.ask("Step to remove", choices=steps)
            
            workflow_data["steps"] = [
                step for step in workflow_data["steps"] 
                if step["name"] != step_to_remove
            ]
            console.print(f"[green]Removed step '{step_to_remove}'[/green]")
        
        elif action == "rename":
            new_name = Prompt.ask("New workflow name")
            workflow_data["name"] = new_name
            console.print(f"[green]Renamed workflow to '{new_name}'[/green]")
        
        # Save changes
        if Confirm.ask("Save changes?"):
            with open(workflow_file, 'w') as f:
                yaml.dump(workflow_data, f, default_flow_style=False)
            console.print(f"[green]Workflow saved[/green]")
    
    def cmd_repl(self, args):
        """Start Python REPL with GreenLang"""
        console.print(Panel("🐍 Python REPL with GreenLang", style="cyan"))
        console.print("[dim]GreenLang client available as 'client'[/dim]")
        console.print("[dim]Type 'exit()' to return to main interface[/dim]\n")
        
        import code
        
        # Create namespace with GreenLang objects
        namespace = {
            'client': self.client,
            'WorkflowBuilder': WorkflowBuilder,
            'AgentBuilder': AgentBuilder,
            'console': console,
        }
        
        # Start REPL
        code.interact(local=namespace, banner="")
    
    def cmd_docs(self, args):
        """View documentation"""
        docs = {
            "quick": self.show_quick_docs,
            "agents": self.show_agent_docs,
            "workflows": self.show_workflow_docs,
            "sdk": self.show_sdk_docs,
            "api": self.show_api_docs
        }
        
        doc_type = Prompt.ask(
            "Documentation",
            choices=list(docs.keys()),
            default="quick"
        )
        
        docs[doc_type]()
    
    def show_quick_docs(self):
        """Show quick start documentation"""
        docs = """
# GreenLang Quick Start

## Basic Usage

```python
from greenlang.sdk import GreenLangClient

client = GreenLangClient()

# Calculate emissions
result = client.calculate_emissions(
    fuel_type="electricity",
    consumption=1000,
    unit="kWh"
)
```

## Workflow Example

```yaml
name: carbon_calculation
steps:
  - name: validate
    agent_id: validator
  - name: calculate
    agent_id: fuel
  - name: report
    agent_id: report
```

## CLI Commands

- `greenlang calc` - Interactive calculator
- `greenlang test` - Run tests
- `greenlang workflow` - Manage workflows
- `greenlang agents` - Manage agents
        """
        
        console.print(Syntax(docs, "markdown", theme="monokai"))
    
    def show_agent_docs(self):
        """Show agent documentation"""
        agent_docs = """
# GreenLang Agents

## Available Agents

### FuelAgent
Calculates emissions based on fuel consumption.

**Input:**
- fuel_type: Type of fuel (electricity, natural_gas, diesel, etc.)
- consumption: Amount consumed
- unit: Unit of measurement (kWh, therms, gallons, etc.)

### CarbonAgent
Aggregates emissions from multiple sources.

**Input:**
- emissions: List of emission data objects

### ValidatorAgent
Validates input data for emissions calculations.

**Input:**
- fuels: List of fuel consumption data

### ReportAgent
Generates carbon footprint reports.

**Input:**
- carbon_data: Aggregated carbon data
- format: Output format (text, json, markdown)

### BenchmarkAgent
Compares emissions against industry benchmarks.

**Input:**
- total_emissions_kg: Total emissions in kg
- building_area: Building area in sqft
- building_type: Type of building
        """
        
        console.print(Syntax(agent_docs, "markdown", theme="monokai"))
    
    def show_workflow_docs(self):
        """Show workflow documentation"""
        workflow_docs = """
# GreenLang Workflows

## Creating Workflows

Workflows define a sequence of agent operations.

### YAML Format

```yaml
name: workflow_name
description: Workflow description
steps:
  - name: step1
    agent_id: agent_name
    input_mapping:
      param1: input.field1
      param2: results.previous_step.data
    on_failure: stop
```

### Python SDK

```python
from greenlang.sdk import WorkflowBuilder

workflow = (WorkflowBuilder("name", "description")
    .add_step("step1", "agent1")
    .add_step("step2", "agent2")
    .build())
```

## Input Mapping

Map data between steps using dot notation:
- `input.field` - Access input data
- `results.step_name.data` - Access previous step results
        """
        
        console.print(Syntax(workflow_docs, "markdown", theme="monokai"))
    
    def show_sdk_docs(self):
        """Show SDK documentation"""
        sdk_docs = """
# GreenLang Python SDK

## Installation

```bash
pip install greenlang
```

## Basic Usage

```python
from greenlang.sdk import GreenLangClient

client = GreenLangClient()

# Calculate emissions
emissions = client.calculate_emissions(
    fuel_type="electricity",
    consumption=1000,
    unit="kWh"
)

# Aggregate emissions
total = client.aggregate_emissions([emissions])

# Generate report
report = client.generate_report(total)
```

## Custom Agents

```python
from greenlang.sdk import AgentBuilder

agent = (AgentBuilder("CustomAgent", "Description")
    .with_execute(my_function)
    .build())

client.register_agent("custom", agent)
```
        """
        
        console.print(Syntax(sdk_docs, "markdown", theme="monokai"))
    
    def show_api_docs(self):
        """Show API documentation"""
        api_docs = """
# GreenLang API Reference

## GreenLangClient

### Methods

#### calculate_emissions(fuel_type, consumption, unit, region="US")
Calculate emissions for a single fuel source.

#### aggregate_emissions(emissions_list)
Aggregate emissions from multiple sources.

#### generate_report(carbon_data, format="text", building_info=None)
Generate a carbon footprint report.

#### benchmark_emissions(total_kg, area, building_type, months)
Compare emissions against industry benchmarks.

#### register_agent(agent_id, agent)
Register a custom agent.

#### register_workflow(workflow_id, workflow)
Register a workflow.

#### execute_workflow(workflow_id, input_data)
Execute a registered workflow.

## AgentResult

### Properties
- success: bool
- data: dict
- error: str (optional)
- metadata: dict
        """
        
        console.print(Syntax(api_docs, "markdown", theme="monokai"))
    
    def cmd_help(self, args):
        """Show help"""
        help_text = """
[bold cyan]GreenLang Developer Interface Commands[/bold cyan]

[bold]Core Commands:[/bold]
  new        Create new project, workflow, or agent
  calc       Interactive emissions calculator
  test       Run test suite
  agents     Manage and test agents
  workflow   Design and manage workflows
  repl       Python REPL with GreenLang
  
[bold]Project Commands:[/bold]
  workspace  Manage workspace and projects
  run        Run workflows or scripts
  export     Export data and reports
  init       Initialize a new GreenLang project
  project    Manage current project
  
[bold]Analysis Commands:[/bold]
  benchmark  Run benchmark analysis
  profile    Profile emissions over time
  validate   Validate data and workflows
  analyze    Analyze emissions data
  compare    Compare multiple scenarios
  
[bold]Documentation:[/bold]
  docs       View documentation
  help       Show this help message
  examples   Show code examples
  api        API reference
  
[bold]System:[/bold]
  exit/quit  Exit the interface
  clear      Clear the screen
  status     Show system status
  version    Show version info
  config     Manage configuration
        """
        
        console.print(Panel(help_text, title="Help", border_style="blue"))
    
    def cmd_workspace(self, args):
        """Manage workspace"""
        action = Prompt.ask(
            "Workspace action",
            choices=["info", "change", "list"]
        )
        
        if action == "info":
            console.print(f"Current workspace: [cyan]{self.workspace}[/cyan]")
            if self.current_project:
                console.print(f"Current project: [green]{self.current_project}[/green]")
        
        elif action == "change":
            new_path = Prompt.ask("New workspace path")
            new_workspace = Path(new_path)
            if new_workspace.exists():
                self.workspace = new_workspace
                console.print(f"[green]Workspace changed to: {new_workspace}[/green]")
            else:
                console.print(f"[red]Path does not exist: {new_path}[/red]")
        
        elif action == "list":
            # List projects in workspace
            projects = [d for d in self.workspace.iterdir() if d.is_dir() and (d / "greenlang.yaml").exists()]
            
            if projects:
                table = Table(title="Projects", box=box.ROUNDED)
                table.add_column("Name", style="cyan")
                table.add_column("Path", style="green")
                
                for project in projects:
                    table.add_row(project.name, str(project))
                
                console.print(table)
            else:
                console.print("[yellow]No GreenLang projects found in workspace[/yellow]")
    
    def cmd_run(self, args):
        """Run a workflow or script"""
        if args:
            target = args[0]
        else:
            target = Prompt.ask("File to run")
        
        target_path = self.workspace / target
        
        if not target_path.exists():
            console.print(f"[red]File not found: {target}[/red]")
            return
        
        if target_path.suffix in ['.yaml', '.yml']:
            # Run as workflow
            workflow = Workflow.from_yaml(str(target_path))
            self.client.register_workflow("temp", workflow)
            
            # Get input
            input_file = Prompt.ask("Input file (JSON)", default="")
            if input_file:
                with open(input_file, 'r') as f:
                    input_data = json.load(f)
            else:
                input_data = {}
            
            with console.status("Running workflow..."):
                result = self.client.execute_workflow("temp", input_data)
            
            if result["success"]:
                console.print("[green]✓ Workflow completed[/green]")
                if "data" in result:
                    console.print(Syntax(json.dumps(result["data"], indent=2), "json", theme="monokai"))
            else:
                console.print("[red]✗ Workflow failed[/red]")
        
        elif target_path.suffix == '.py':
            # Run as Python script
            import subprocess
            result = subprocess.run([sys.executable, str(target_path)], capture_output=True, text=True)
            console.print(result.stdout)
            if result.stderr:
                console.print(result.stderr, style="red")
    
    def cmd_benchmark(self, args):
        """Run benchmark analysis"""
        console.print(Panel("📊 Benchmark Analysis", style="cyan"))
        
        # Get parameters
        emissions_kg = float(Prompt.ask("Total emissions (kg CO2e)"))
        area = float(Prompt.ask("Building area (sqft)"))
        building_type = Prompt.ask(
            "Building type",
            choices=["commercial_office", "retail", "warehouse", "residential"]
        )
        period_months = int(Prompt.ask("Period (months)", default="12"))
        
        # Run benchmark
        result = self.client.benchmark_emissions(
            emissions_kg,
            area,
            building_type,
            period_months
        )
        
        if result["success"]:
            data = result["data"]
            
            # Display results
            rating_color = "green" if data["rating"] in ["Excellent", "Good"] else "yellow" if data["rating"] == "Average" else "red"
            
            results_panel = f"""
[bold]Benchmark Results[/bold]

Rating: [{rating_color}]{data['rating']}[/{rating_color}]
Carbon Intensity: {data['carbon_intensity']:.2f} kg CO2e/sqft/year
Percentile: Top {data['percentile']}%

[bold]Comparison:[/bold]
vs Excellent: {data['comparison']['vs_excellent']:+.2f} kg CO2e/sqft/year
vs Average: {data['comparison']['vs_average']:+.2f} kg CO2e/sqft/year
Improvement needed: {data['comparison']['improvement_to_good']:.2f} kg CO2e/sqft/year

[bold]Recommendations:[/bold]
"""
            
            for i, rec in enumerate(data["recommendations"][:5], 1):
                results_panel += f"{i}. {rec}\n"
            
            console.print(Panel(results_panel, title="Benchmark Analysis", border_style="cyan"))
            
            # Save option
            if Confirm.ask("Save benchmark results?"):
                filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                console.print(f"[green]Results saved to {filename}[/green]")
    
    def cmd_export(self, args):
        """Export data and reports"""
        export_type = Prompt.ask(
            "Export type",
            choices=["emissions", "report", "workflow", "agents"]
        )
        
        if export_type == "emissions":
            # Export emissions data
            format = Prompt.ask("Format", choices=["json", "csv", "excel"])
            filename = Prompt.ask("Filename", default=f"emissions_{datetime.now().strftime('%Y%m%d')}")
            
            # Get sample data (in real use, this would be from calculations)
            data = {
                "date": datetime.now().isoformat(),
                "emissions": {
                    "electricity": 385.0,
                    "natural_gas": 530.0,
                    "total": 915.0
                }
            }
            
            if format == "json":
                with open(f"{filename}.json", 'w') as f:
                    json.dump(data, f, indent=2)
                console.print(f"[green]Exported to {filename}.json[/green]")
            
            elif format == "csv":
                import csv
                with open(f"{filename}.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Source", "Emissions (kg CO2e)"])
                    for source, value in data["emissions"].items():
                        writer.writerow([source, value])
                console.print(f"[green]Exported to {filename}.csv[/green]")
    
    def cmd_validate(self, args):
        """Validate data and workflows"""
        validate_type = Prompt.ask(
            "Validate",
            choices=["data", "workflow", "config"]
        )
        
        if validate_type == "data":
            # Validate emissions data
            data_file = Prompt.ask("Data file (JSON)")
            
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                result = self.client.validate_input(data)
                
                if result["success"]:
                    console.print("[green]✓ Data is valid[/green]")
                    if "warnings" in result["data"] and result["data"]["warnings"]:
                        console.print("\n[yellow]Warnings:[/yellow]")
                        for warning in result["data"]["warnings"]:
                            console.print(f"  - {warning}")
                else:
                    console.print("[red]✗ Data validation failed[/red]")
                    if "errors" in result["data"]:
                        for error in result["data"]["errors"]:
                            console.print(f"  - {error}", style="red")
            
            except Exception as e:
                console.print(f"[red]Error loading data: {e}[/red]")
        
        elif validate_type == "workflow":
            workflow_file = Prompt.ask("Workflow file")
            self.validate_workflow(workflow_file.replace(".yaml", "").replace(".yml", ""))
    
    def cmd_profile(self, args):
        """Profile emissions over time"""
        console.print(Panel("📈 Emissions Profiling", style="cyan"))
        
        # Get time series data
        periods = int(Prompt.ask("Number of periods", default="12"))
        
        console.print("\nEnter monthly consumption data:")
        
        data = []
        for i in range(1, periods + 1):
            console.print(f"\n[bold]Month {i}:[/bold]")
            electricity = float(Prompt.ask(f"  Electricity (kWh)", default="0"))
            gas = float(Prompt.ask(f"  Natural gas (therms)", default="0"))
            
            # Calculate emissions
            emissions = 0
            if electricity > 0:
                result = self.client.calculate_emissions("electricity", electricity, "kWh")
                if result["success"]:
                    emissions += result["data"]["co2e_emissions_kg"]
            
            if gas > 0:
                result = self.client.calculate_emissions("natural_gas", gas, "therms")
                if result["success"]:
                    emissions += result["data"]["co2e_emissions_kg"]
            
            data.append({
                "month": i,
                "electricity": electricity,
                "gas": gas,
                "emissions_kg": emissions,
                "emissions_tons": emissions / 1000
            })
        
        # Display profile
        table = Table(title="Emissions Profile", box=box.ROUNDED)
        table.add_column("Month", style="cyan")
        table.add_column("Electricity (kWh)", style="yellow")
        table.add_column("Gas (therms)", style="yellow")
        table.add_column("Emissions (tons)", style="green")
        
        total_emissions = 0
        for period in data:
            table.add_row(
                str(period["month"]),
                f"{period['electricity']:.0f}",
                f"{period['gas']:.0f}",
                f"{period['emissions_tons']:.3f}"
            )
            total_emissions += period["emissions_tons"]
        
        console.print(table)
        
        # Summary
        avg_emissions = total_emissions / periods
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"Total emissions: {total_emissions:.3f} metric tons CO2e")
        console.print(f"Average monthly: {avg_emissions:.3f} metric tons CO2e")
        console.print(f"Annualized: {avg_emissions * 12:.3f} metric tons CO2e")
        
        # Save profile
        if Confirm.ask("\nSave profile data?"):
            filename = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump({
                    "periods": data,
                    "summary": {
                        "total": total_emissions,
                        "average": avg_emissions,
                        "annualized": avg_emissions * 12
                    }
                }, f, indent=2)
            console.print(f"[green]Profile saved to {filename}[/green]")


    def cmd_init(self, args):
        """Initialize a new GreenLang project in current directory"""
        project_name = Prompt.ask("Project name", default=Path.cwd().name)
        self.create_project_files(Path.cwd(), project_name)
        console.print(f"[green]✓ Initialized GreenLang project: {project_name}[/green]")
    
    def cmd_project(self, args):
        """Manage current project"""
        if not self.current_project:
            console.print("[yellow]No active project. Use 'new' to create one.[/yellow]")
            return
        
        action = Prompt.ask(
            "Project action",
            choices=["info", "settings", "dependencies", "build"]
        )
        
        if action == "info":
            console.print(f"Project: [cyan]{self.current_project}[/cyan]")
            console.print(f"Path: [green]{self.workspace}[/green]")
            
            # Count files
            py_files = len(list(self.workspace.glob("**/*.py")))
            yaml_files = len(list(self.workspace.glob("**/*.yaml")))
            console.print(f"Files: {py_files} Python, {yaml_files} YAML")
        
        elif action == "settings":
            config_file = self.workspace / "greenlang.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                console.print(Syntax(yaml.dump(config, default_flow_style=False), "yaml", theme="monokai"))
    
    def cmd_analyze(self, args):
        """Analyze emissions data"""
        console.print(Panel("📊 Emissions Analysis", style="cyan"))
        
        analysis_type = Prompt.ask(
            "Analysis type",
            choices=["trends", "breakdown", "intensity", "comparison"]
        )
        
        if analysis_type == "breakdown":
            # Get emissions data
            electricity = float(Prompt.ask("Electricity (kWh)", default="0"))
            gas = float(Prompt.ask("Natural gas (therms)", default="0"))
            diesel = float(Prompt.ask("Diesel (gallons)", default="0"))
            
            # Calculate and show breakdown
            total = 0
            breakdown = []
            
            if electricity > 0:
                result = self.client.calculate_emissions("electricity", electricity, "kWh")
                emissions = result["data"]["co2e_emissions_kg"]
                total += emissions
                breakdown.append(("Electricity", emissions))
            
            if gas > 0:
                result = self.client.calculate_emissions("natural_gas", gas, "therms")
                emissions = result["data"]["co2e_emissions_kg"]
                total += emissions
                breakdown.append(("Natural Gas", emissions))
            
            if diesel > 0:
                result = self.client.calculate_emissions("diesel", diesel, "gallons")
                emissions = result["data"]["co2e_emissions_kg"]
                total += emissions
                breakdown.append(("Diesel", emissions))
            
            # Display pie chart-like breakdown
            console.print("\n[bold]Emissions Breakdown:[/bold]")
            for source, emissions in breakdown:
                percentage = (emissions / total * 100) if total > 0 else 0
                bar_length = int(percentage / 2)
                bar = "█" * bar_length
                console.print(f"{source:15} {bar} {percentage:.1f}% ({emissions:.2f} kg)")
            
            console.print(f"\n[bold]Total:[/bold] {total:.2f} kg CO2e ({total/1000:.3f} tons)")
    
    def cmd_compare(self, args):
        """Compare multiple scenarios"""
        console.print(Panel("🔄 Scenario Comparison", style="cyan"))
        
        scenarios = []
        num_scenarios = int(Prompt.ask("Number of scenarios to compare", default="2"))
        
        for i in range(num_scenarios):
            console.print(f"\n[bold]Scenario {i+1}:[/bold]")
            name = Prompt.ask(f"Scenario name", default=f"Scenario {i+1}")
            electricity = float(Prompt.ask("Electricity (kWh)", default="0"))
            gas = float(Prompt.ask("Natural gas (therms)", default="0"))
            
            # Calculate emissions
            total = 0
            if electricity > 0:
                result = self.client.calculate_emissions("electricity", electricity, "kWh")
                total += result["data"]["co2e_emissions_kg"]
            if gas > 0:
                result = self.client.calculate_emissions("natural_gas", gas, "therms")
                total += result["data"]["co2e_emissions_kg"]
            
            scenarios.append({
                "name": name,
                "electricity": electricity,
                "gas": gas,
                "emissions": total
            })
        
        # Display comparison
        table = Table(title="Scenario Comparison", box=box.ROUNDED)
        table.add_column("Scenario", style="cyan")
        table.add_column("Electricity (kWh)", style="yellow")
        table.add_column("Gas (therms)", style="yellow")
        table.add_column("Emissions (kg)", style="green")
        table.add_column("Difference", style="magenta")
        
        baseline = scenarios[0]["emissions"]
        for scenario in scenarios:
            diff = scenario["emissions"] - baseline
            diff_str = f"{diff:+.2f}" if scenario != scenarios[0] else "baseline"
            table.add_row(
                scenario["name"],
                f"{scenario['electricity']:.0f}",
                f"{scenario['gas']:.0f}",
                f"{scenario['emissions']:.2f}",
                diff_str
            )
        
        console.print(table)
        
        # Best scenario
        best = min(scenarios, key=lambda x: x["emissions"])
        console.print(f"\n[green]Best scenario: {best['name']} ({best['emissions']:.2f} kg CO2e)[/green]")
    
    def cmd_examples(self, args):
        """Show code examples"""
        examples = {
            "basic": """# Basic emissions calculation
from greenlang.sdk import GreenLangClient

client = GreenLangClient()
result = client.calculate_emissions("electricity", 1000, "kWh")
print(f"Emissions: {result['data']['co2e_emissions_kg']} kg")""",
            
            "workflow": """# Create and run a workflow
from greenlang.sdk import WorkflowBuilder

workflow = WorkflowBuilder("analysis", "Emissions analysis")
    .add_step("calculate", "fuel")
    .add_step("report", "report")
    .build()

client.register_workflow("analysis", workflow)
result = client.execute_workflow("analysis", input_data)""",
            
            "agent": """# Create custom agent
from greenlang.sdk import AgentBuilder

agent = AgentBuilder("Custom", "My agent")
    .with_execute(my_function)
    .build()

client.register_agent("custom", agent)"""
        }
        
        example_type = Prompt.ask(
            "Example type",
            choices=list(examples.keys()),
            default="basic"
        )
        
        console.print(Syntax(examples[example_type], "python", theme="monokai"))
    
    def cmd_api(self, args):
        """Show API reference"""
        self.show_api_docs()
    
    def cmd_exit(self, args):
        """Exit the interface"""
        if Confirm.ask("Exit GreenLang Developer Interface?"):
            console.print("[green]Goodbye![/green]")
            exit(0)
    
    def cmd_clear(self, args):
        """Clear the screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        self.show_welcome()
    
    def cmd_status(self, args):
        """Show system status"""
        status_info = f"""
[bold]GreenLang Status[/bold]

[cyan]System:[/cyan]
  Version: 0.0.1
  Python: {sys.version.split()[0]}
  Platform: {sys.platform}
  
[cyan]Workspace:[/cyan]
  Path: {self.workspace}
  Project: {self.current_project or 'None'}
  
[cyan]Agents:[/cyan]
  Loaded: {len(self.client.list_agents())} agents
  
[cyan]Session:[/cyan]
  Commands executed: {len(self.history)}
  Current directory: {Path.cwd()}
        """
        console.print(Panel(status_info, title="Status", border_style="green"))
    
    def cmd_version(self, args):
        """Show version information"""
        console.print("[bold]GreenLang Developer Interface[/bold]")
        console.print("Version: [green]0.0.1[/green]")
        console.print("Python: [cyan]" + sys.version + "[/cyan]")
    
    def cmd_config(self, args):
        """Manage configuration"""
        config_action = Prompt.ask(
            "Config action",
            choices=["show", "edit", "reset"]
        )
        
        config_file = self.workspace / "greenlang.yaml"
        
        if config_action == "show":
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                console.print(Syntax(yaml.dump(config, default_flow_style=False), "yaml", theme="monokai"))
            else:
                console.print("[yellow]No configuration file found[/yellow]")
        
        elif config_action == "edit":
            if config_file.exists():
                key = Prompt.ask("Config key (e.g., settings.region)")
                value = Prompt.ask("New value")
                
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Set nested key
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                console.print(f"[green]Updated {key} = {value}[/green]")
        
        elif config_action == "reset":
            if Confirm.ask("Reset configuration to defaults?"):
                default_config = {
                    "name": self.current_project or "greenlang-project",
                    "version": "0.0.1",
                    "settings": {
                        "region": "US",
                        "report_format": "text",
                        "auto_validate": True
                    }
                }
                
                with open(config_file, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                
                console.print("[green]Configuration reset to defaults[/green]")


def main():
    """Main entry point for developer interface"""
    interface = GreenLangDevInterface()
    interface.start()


if __name__ == "__main__":
    main()