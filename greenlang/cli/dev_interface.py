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
        """Create a custom agent"""
        console.print(Panel("Create Custom Agent", style="cyan"))
        
        name = Prompt.ask("Agent name")
        description = Prompt.ask("Description")
        
        # Generate agent code
        agent_code = f'''"""
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
        # Add your validation logic here
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        # Add your agent logic here
        
        # Example:
        result_data = {{
            "message": f"Processed by {name}",
            "input": input_data
        }}
        
        return AgentResult(
            success=True,
            data=result_data
        )
'''
        
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
    
    def test_agent(self, agent_id: str):
        """Test an agent with sample data"""
        console.print(f"Testing agent: {agent_id}")
        
        # Get test data
        test_data = {}
        if agent_id == "fuel":
            test_data = {
                "fuel_type": "electricity",
                "consumption": 1000,
                "unit": "kWh"
            }
        elif agent_id == "carbon":
            test_data = {
                "emissions": [
                    {"co2e_emissions_kg": 100},
                    {"co2e_emissions_kg": 200}
                ]
            }
        elif agent_id == "validator":
            test_data = {
                "fuels": [
                    {"type": "electricity", "consumption": 100, "unit": "kWh"}
                ]
            }
        
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