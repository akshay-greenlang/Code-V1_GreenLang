"""
Enhanced initialization commands for GreenLang CLI
"""

import click
import json
import yaml
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

console = Console()

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"

# Available pipeline templates
PIPELINE_TEMPLATES = {
    "basic": {
        "description": "Basic pipeline template with validation and reporting",
        "agents": ["validator", "carbon", "report"]
    },
    "carbon-calculation": {
        "description": "Comprehensive carbon footprint calculation pipeline",
        "agents": ["validator", "carbon", "intensity", "report"]
    },
    "energy-analysis": {
        "description": "Energy consumption analysis and optimization",
        "agents": ["validator", "energy_balance", "solar_resource", "report"]
    },
    "building-assessment": {
        "description": "Building performance and efficiency assessment",
        "agents": ["validator", "building_profile", "energy_balance", "boiler", "report"]
    },
    "grid-analysis": {
        "description": "Electricity grid factor analysis",
        "agents": ["validator", "grid_factor", "intensity", "report"]
    },
    "fuel-analysis": {
        "description": "Fuel consumption and emissions analysis",
        "agents": ["validator", "fuel", "carbon", "report"]
    },
    "benchmark": {
        "description": "Performance benchmarking pipeline",
        "agents": ["validator", "benchmark", "report"]
    }
}

# Available pack templates
PACK_TEMPLATES = {
    "basic": {
        "description": "Basic pack with pipeline and data",
        "kind": "pack",
        "contents": ["pipelines", "datasets"]
    },
    "agent-pack": {
        "description": "Pack containing custom agents",
        "kind": "agent-pack",
        "contents": ["agents", "pipelines"]
    },
    "dataset": {
        "description": "Dataset pack with multiple data sources",
        "kind": "dataset",
        "contents": ["datasets", "schemas"]
    },
    "connector": {
        "description": "Data connector pack for external APIs",
        "kind": "connector",
        "contents": ["agents", "pipelines", "configs"]
    },
    "ml-model": {
        "description": "Machine learning model pack",
        "kind": "ml-model",
        "contents": ["models", "pipelines", "datasets"]
    },
    "report-template": {
        "description": "Report template pack",
        "kind": "report-template",
        "contents": ["reports", "templates"]
    }
}


@click.group()
def init():
    """Initialize GreenLang projects and components

    Create new projects, pipelines, and packs from templates.
    """
    pass


@init.command(name="project")
@click.argument("name", required=False)
@click.option("--template", "-t", type=click.Choice(["basic", "advanced", "minimal"]), default="basic", help="Project template")
@click.option("--path", "-p", type=click.Path(), help="Directory to create project in")
@click.option("--author", help="Author name")
@click.option("--description", help="Project description")
def init_project(name: Optional[str], template: str, path: Optional[str], author: Optional[str], description: Optional[str]):
    """Initialize a new GreenLang project

    Examples:
        gl init project my-project                    # Basic project setup
        gl init project --template advanced my-proj     # Advanced template
        gl init project --path ./projects my-proj       # Specific directory
    """
    if not name:
        name = click.prompt("Project name", type=str)

    # Validate project name
    if not name.replace("-", "").replace("_", "").isalnum():
        console.print(f"[red]Invalid project name '{name}'[/red]")
        console.print("Project name should contain only letters, numbers, hyphens, and underscores")
        sys.exit(1)

    # Determine target directory
    if path:
        target_dir = Path(path) / name
    else:
        target_dir = Path.cwd() / name

    if target_dir.exists():
        console.print(f"[red]Directory '{target_dir}' already exists[/red]")
        sys.exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task = progress.add_task("Creating project structure...", total=None)

        # Create project directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create directory structure based on template
        if template == "minimal":
            dirs = ["pipelines", "data"]
        elif template == "advanced":
            dirs = [
                "pipelines", "data", "reports", "logs", "agents/custom",
                "cache", "configs", "schemas", "templates", "tests",
                "docs", "scripts", "notebooks"
            ]
        else:  # basic
            dirs = [
                "pipelines", "data", "reports", "logs", "agents/custom",
                "cache", "configs", "schemas"
            ]

        for dir_name in dirs:
            (target_dir / dir_name).mkdir(parents=True, exist_ok=True)
            progress.update(task, description=f"Created {dir_name}/")
            time.sleep(0.05)

        # Create project config
        project_config = {
            "name": name,
            "version": "1.0.0",
            "description": description or f"GreenLang project: {name}",
            "author": author or "Unknown",
            "created": datetime.now().isoformat(),
            "template": template,
            "greenlang_version": ">=1.0.0"
        }

        (target_dir / "greenlang.yaml").write_text(yaml.dump(project_config, default_flow_style=False))

        # Create sample pipeline based on template
        if template != "minimal":
            sample_pipeline = create_sample_pipeline(name, template)
            (target_dir / "pipelines" / "sample.yaml").write_text(yaml.dump(sample_pipeline, default_flow_style=False))

        # Create environment file
        env_content = create_env_template(template)
        (target_dir / ".env").write_text(env_content)

        # Create gitignore
        gitignore_content = create_gitignore_template(template)
        (target_dir / ".gitignore").write_text(gitignore_content)

        # Create README if advanced template
        if template == "advanced":
            readme_content = create_readme_template(name, description)
            (target_dir / "README.md").write_text(readme_content)

        # Create sample data
        if template != "minimal":
            sample_data = create_sample_dataset(template)
            (target_dir / "data" / "sample.json").write_text(json.dumps(sample_data, indent=2))

        progress.update(task, description="Project created successfully!")

    console.print(f"\n[green]+ Project '{name}' created successfully![/green]")
    console.print(f"Location: {target_dir}")
    console.print(f"Template: {template}")

    console.print("\n[cyan]Next steps:[/cyan]")
    console.print(f"  1. cd {target_dir}")
    console.print(f"  2. gl validate pipelines/sample.yaml")
    console.print(f"  3. gl run pipelines/sample.yaml")

    if template == "advanced":
        console.print(f"  4. Explore the docs/ directory for more information")


@init.command(name="pipeline")
@click.argument("name")
@click.option("--template", "-t", type=click.Choice(list(PIPELINE_TEMPLATES.keys())), default="basic", help="Pipeline template")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--author", help="Author name")
@click.option("--description", help="Pipeline description")
def init_pipeline(name: str, template: str, output: Optional[str], author: Optional[str], description: Optional[str]):
    """Create a new pipeline from template

    Examples:
        gl init pipeline my-pipeline                           # Basic pipeline
        gl init pipeline carbon-calc --template carbon-calculation  # Specific template
        gl init pipeline analysis --output pipelines/analysis.yaml  # Custom output path
    """
    # Validate pipeline name
    if not name.replace("-", "").replace("_", "").isalnum():
        console.print(f"[red]Invalid pipeline name '{name}'[/red]")
        console.print("Pipeline name should contain only letters, numbers, hyphens, and underscores")
        sys.exit(1)

    # Determine output file
    if output:
        output_path = Path(output)
    else:
        # Default to pipelines directory if it exists, otherwise current directory
        if Path("pipelines").exists():
            output_path = Path("pipelines") / f"{name}.yaml"
        else:
            output_path = Path(f"{name}.yaml")

    if output_path.exists():
        console.print(f"[red]File '{output_path}' already exists[/red]")
        sys.exit(1)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template_info = PIPELINE_TEMPLATES[template]

    # Create pipeline configuration
    pipeline_config = {
        "name": name,
        "description": description or template_info["description"],
        "version": "1.0.0",
        "author": author or "Unknown",
        "created": datetime.now().isoformat(),
        "template": template,
        "metadata": {
            "tags": [template, "generated"],
            "category": get_pipeline_category(template)
        },
        "steps": []
    }

    # Add steps based on template
    step_number = 1
    for agent_id in template_info["agents"]:
        step = {
            "name": f"step_{step_number:02d}_{agent_id}",
            "agent_id": agent_id,
            "description": f"Execute {agent_id} agent",
            "timeout": 300,
            "retry_count": 2 if agent_id != "validator" else 1
        }

        # Add agent-specific configuration
        if agent_id == "validator":
            step["config"] = {
                "strict_mode": True,
                "schema_validation": True
            }
        elif agent_id == "report":
            step["config"] = {
                "format": ["json", "html"],
                "include_charts": True
            }

        pipeline_config["steps"].append(step)
        step_number += 1

    # Add output mapping
    pipeline_config["output_mapping"] = create_output_mapping(template_info["agents"])

    # Add environment and dependencies
    pipeline_config["environment"] = {
        "python_version": ">=3.8",
        "dependencies": get_template_dependencies(template)
    }

    # Write pipeline file
    output_path.write_text(yaml.dump(pipeline_config, default_flow_style=False, sort_keys=False))

    console.print(f"[green]+ Pipeline '{name}' created successfully![/green]")
    console.print(f"Template: {template}")
    console.print(f"File: {output_path}")
    console.print(f"Agents: {', '.join(template_info['agents'])}")

    console.print("\n[cyan]Next steps:[/cyan]")
    console.print(f"  1. gl validate {output_path}")
    console.print(f"  2. gl run {output_path}")
    console.print(f"  3. Customize the pipeline configuration as needed")


@init.command(name="pack")
@click.argument("name")
@click.option("--template", "-t", type=click.Choice(list(PACK_TEMPLATES.keys())), default="basic", help="Pack template")
@click.option("--path", "-p", type=click.Path(), help="Directory to create pack in")
@click.option("--author", help="Author name")
@click.option("--description", help="Pack description")
@click.option("--license", default="MIT", help="License for the pack")
def init_pack(name: str, template: str, path: Optional[str], author: Optional[str], description: Optional[str], license: str):
    """Create a new pack from template (enhanced version)

    Examples:
        gl init pack my-pack                          # Basic pack
        gl init pack ml-models --template ml-model     # ML model pack
        gl init pack connectors --template connector   # Connector pack
    """
    # Validate pack name (DNS-safe)
    import re
    if not re.match(r'^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$', name):
        console.print(f"[red]Invalid pack name '{name}'[/red]")
        console.print("Pack name must be DNS-safe: lowercase, alphanumeric, hyphens only")
        sys.exit(1)

    # Determine target directory
    target_dir = Path(path) / name if path else Path(name)

    if target_dir.exists():
        console.print(f"[red]Directory '{target_dir}' already exists[/red]")
        sys.exit(1)

    template_info = PACK_TEMPLATES[template]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task = progress.add_task("Creating pack structure...", total=None)

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create directory structure based on template
        for content_type in template_info["contents"]:
            content_dir = target_dir / content_type
            content_dir.mkdir(exist_ok=True)
            progress.update(task, description=f"Created {content_type}/")
            time.sleep(0.05)

        # Create standard directories
        for dir_name in ["input", "output", "tests", "docs"]:
            (target_dir / dir_name).mkdir(exist_ok=True)

        # Create pack manifest
        manifest = create_pack_manifest(name, template, template_info, author, description, license)
        (target_dir / "pack.yaml").write_text(yaml.dump(manifest, default_flow_style=False, sort_keys=False))

        # Create main pipeline if pipeline content is included
        if "pipelines" in template_info["contents"]:
            main_pipeline = create_pack_pipeline(name, template)
            (target_dir / "gl.yaml").write_text(yaml.dump(main_pipeline, default_flow_style=False))

        # Create template-specific content
        create_template_content(target_dir, template, template_info, name)

        # Create documentation
        create_pack_documentation(target_dir, name, template, template_info)

        # Create gitignore
        gitignore_content = """# GreenLang Pack
output/
*.pyc
__pycache__/
.env
.venv/
*.log
.DS_Store
*.tmp
.pytest_cache/
coverage.xml
htmlcov/
"""
        (target_dir / ".gitignore").write_text(gitignore_content)

        progress.update(task, description="Pack created successfully!")

    console.print(f"\n[green]+ Pack '{name}' created successfully![/green]")
    console.print(f"Template: {template}")
    console.print(f"Location: {target_dir}")
    console.print(f"Kind: {template_info['kind']}")

    console.print("\n[cyan]Next steps:[/cyan]")
    console.print(f"  1. cd {target_dir}")
    console.print(f"  2. gl pack validate")
    if "pipelines" in template_info["contents"]:
        console.print(f"  3. gl run gl.yaml")
    console.print(f"  4. Customize the pack contents as needed")


def create_sample_pipeline(project_name: str, template: str) -> Dict[str, Any]:
    """Create a sample pipeline for the project"""
    if template == "advanced":
        return {
            "name": f"{project_name}_comprehensive",
            "description": f"Comprehensive analysis pipeline for {project_name}",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "validate_input",
                    "agent_id": "validator",
                    "description": "Validate input data structure and content",
                    "retry_count": 2
                },
                {
                    "name": "calculate_carbon",
                    "agent_id": "carbon",
                    "description": "Calculate carbon footprint"
                },
                {
                    "name": "analyze_intensity",
                    "agent_id": "intensity",
                    "description": "Analyze carbon intensity factors"
                },
                {
                    "name": "generate_report",
                    "agent_id": "report",
                    "description": "Generate comprehensive analysis report"
                }
            ],
            "output_mapping": {
                "total_emissions": "results.calculate_carbon.data.total",
                "intensity_data": "results.analyze_intensity.data",
                "report": "results.generate_report.data.report"
            }
        }
    else:
        return {
            "name": f"{project_name}_basic",
            "description": f"Basic carbon calculation pipeline for {project_name}",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "validate_input",
                    "agent_id": "validator",
                    "description": "Validate input data",
                    "retry_count": 2
                },
                {
                    "name": "calculate_emissions",
                    "agent_id": "carbon",
                    "description": "Calculate carbon emissions"
                },
                {
                    "name": "generate_report",
                    "agent_id": "report",
                    "description": "Generate emissions report"
                }
            ],
            "output_mapping": {
                "total_emissions": "results.calculate_emissions.data.total",
                "report": "results.generate_report.data.report"
            }
        }


def create_env_template(template: str) -> str:
    """Create environment template"""
    base_env = """# GreenLang Environment Configuration
GREENLANG_ENV=development
GREENLANG_LOG_LEVEL=INFO

# API Keys (optional)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=...

# Custom paths
# GREENLANG_AGENTS_PATH=/path/to/custom/agents
"""

    if template == "advanced":
        base_env += """
# Advanced Configuration
GREENLANG_CACHE_DIR=./cache
GREENLANG_REPORTS_DIR=./reports
GREENLANG_SCHEMAS_DIR=./schemas

# Performance Settings
GREENLANG_MAX_WORKERS=4
GREENLANG_TIMEOUT=300

# External Services
# DATABASE_URL=postgresql://user:pass@localhost/db
# REDIS_URL=redis://localhost:6379
"""

    return base_env


def create_gitignore_template(template: str) -> str:
    """Create gitignore template"""
    base_ignore = """# GreenLang Project
cache/
output/
reports/*.html
reports/*.pdf
logs/
.env
.venv/
venv/

# Python
*.pyc
__pycache__/
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
"""

    if template == "advanced":
        base_ignore += """
# Advanced template additions
notebooks/.ipynb_checkpoints/
docs/_build/
coverage.xml
htmlcov/
.pytest_cache/
.coverage
*.tmp
"""

    return base_ignore


def create_readme_template(name: str, description: Optional[str]) -> str:
    """Create README template for advanced projects"""
    return f"""# {name.title().replace('-', ' ').replace('_', ' ')}

{description or f'GreenLang project for climate intelligence analysis.'}

## Overview

This project provides comprehensive climate analysis capabilities using the GreenLang framework.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- GreenLang CLI installed

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Usage

1. Validate your pipeline:
   ```bash
   gl validate pipelines/sample.yaml
   ```

2. Run the analysis:
   ```bash
   gl run pipelines/sample.yaml --input data/sample.json
   ```

3. View results:
   ```bash
   gl report <run-id>
   ```

## Project Structure

- `pipelines/` - Analysis pipeline definitions
- `data/` - Input datasets
- `reports/` - Generated reports
- `agents/` - Custom analysis agents
- `schemas/` - Data validation schemas
- `configs/` - Configuration files
- `docs/` - Documentation
- `tests/` - Test suite

## Documentation

See the `docs/` directory for detailed documentation.

## License

This project is licensed under the MIT License.
"""


def create_sample_dataset(template: str) -> Dict[str, Any]:
    """Create sample dataset based on template"""
    base_data = {
        "metadata": {
            "name": "Sample Dataset",
            "version": "1.0.0",
            "created": datetime.now().isoformat(),
            "source": "Generated sample data"
        }
    }

    if template == "advanced":
        base_data["data"] = {
            "facilities": [
                {
                    "id": "facility_001",
                    "name": "Manufacturing Plant A",
                    "location": {"lat": 40.7128, "lng": -74.0060},
                    "fuels": [
                        {"type": "electricity", "amount": 50000, "unit": "kWh"},
                        {"type": "natural_gas", "amount": 2500, "unit": "therms"},
                        {"type": "diesel", "amount": 1000, "unit": "gallons"}
                    ],
                    "processes": [
                        {"name": "production", "energy_consumption": 35000, "unit": "kWh"},
                        {"name": "heating", "energy_consumption": 15000, "unit": "kWh"}
                    ]
                }
            ],
            "time_period": {
                "start": "2024-01-01",
                "end": "2024-12-31"
            }
        }
    else:
        base_data["data"] = {
            "fuels": [
                {"type": "electricity", "amount": 1000, "unit": "kWh"},
                {"type": "natural_gas", "amount": 500, "unit": "therms"}
            ]
        }

    return base_data


def get_pipeline_category(template: str) -> str:
    """Get category for pipeline template"""
    categories = {
        "basic": "general",
        "carbon-calculation": "emissions",
        "energy-analysis": "energy",
        "building-assessment": "buildings",
        "grid-analysis": "grid",
        "fuel-analysis": "fuels",
        "benchmark": "performance"
    }
    return categories.get(template, "general")


def create_output_mapping(agents: List[str]) -> Dict[str, str]:
    """Create output mapping for pipeline"""
    mapping = {}

    for agent in agents:
        if agent == "validator":
            mapping["validation_results"] = f"results.{agent}.data.validation"
        elif agent == "carbon":
            mapping["carbon_emissions"] = f"results.{agent}.data.total_emissions"
        elif agent == "report":
            mapping["final_report"] = f"results.{agent}.data.report"
        else:
            mapping[f"{agent}_results"] = f"results.{agent}.data"

    return mapping


def get_template_dependencies(template: str) -> List[str]:
    """Get dependencies for pipeline template"""
    base_deps = ["greenlang"]

    template_deps = {
        "carbon-calculation": ["pandas", "numpy"],
        "energy-analysis": ["pandas", "numpy", "scipy"],
        "building-assessment": ["pandas", "numpy", "matplotlib"],
        "grid-analysis": ["pandas", "requests"],
        "fuel-analysis": ["pandas", "numpy"],
        "benchmark": ["pandas", "numpy", "matplotlib"]
    }

    return base_deps + template_deps.get(template, [])


def create_pack_manifest(name: str, template: str, template_info: Dict, author: Optional[str], description: Optional[str], license: str) -> Dict[str, Any]:
    """Create pack manifest"""
    return {
        "name": name,
        "version": "1.0.0",
        "description": description or template_info["description"],
        "kind": template_info["kind"],
        "license": license,
        "author": author or "Unknown",
        "created": datetime.now().isoformat(),
        "template": template,
        "spec_version": "1.0",
        "contents": {
            content_type: [] for content_type in template_info["contents"]
        },
        "compat": {
            "greenlang": ">=1.0.0",
            "python": ">=3.8"
        }
    }


def create_pack_pipeline(name: str, template: str) -> Dict[str, Any]:
    """Create main pipeline for pack"""
    return {
        "name": f"{name}_main",
        "description": f"Main pipeline for {name} pack",
        "version": "1.0.0",
        "steps": [
            {
                "name": "validate_input",
                "agent_id": "validator",
                "description": "Validate input data"
            },
            {
                "name": "process_data",
                "agent_id": "carbon",
                "description": "Process data using pack logic"
            },
            {
                "name": "generate_output",
                "agent_id": "report",
                "description": "Generate pack output"
            }
        ],
        "output_mapping": {
            "results": "results.process_data.data",
            "report": "results.generate_output.data.report"
        }
    }


def create_template_content(target_dir: Path, template: str, template_info: Dict, name: str):
    """Create template-specific content"""

    if template == "agent-pack":
        # Create sample agent
        agent_dir = target_dir / "agents"
        agent_dir.mkdir(exist_ok=True)

        agent_content = f"""# Custom Agent for {name}

from greenlang.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    \"\"\"Custom agent implementation\"\"\"

    def __init__(self):
        super().__init__(
            agent_id="custom_{name.replace('-', '_')}",
            name="Custom {name.title()} Agent",
            description="Custom agent for {name} pack"
        )

    def execute(self, input_data):
        \"\"\"Execute agent logic\"\"\"
        # Implement your custom logic here
        return {{
            "success": True,
            "data": {{
                "processed": input_data,
                "agent": self.agent_id
            }}
        }}
"""
        (agent_dir / f"{name.replace('-', '_')}_agent.py").write_text(agent_content)

    elif template == "dataset":
        # Create sample schema
        schemas_dir = target_dir / "schemas"
        schemas_dir.mkdir(exist_ok=True)

        schema_content = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": f"{name.title()} Dataset Schema",
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "version": {"type": "string"},
                        "created": {"type": "string", "format": "date-time"}
                    },
                    "required": ["name", "version"]
                },
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object"
                    }
                }
            },
            "required": ["metadata", "data"]
        }

        (schemas_dir / f"{name}.schema.json").write_text(json.dumps(schema_content, indent=2))


def create_pack_documentation(target_dir: Path, name: str, template: str, template_info: Dict):
    """Create pack documentation"""
    docs_dir = target_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    readme_content = f"""# {name.title().replace('-', ' ')} Pack

{template_info['description']}

## Type

{template_info['kind']}

## Contents

This pack includes:

{chr(10).join(f'- {content.title()}' for content in template_info['contents'])}

## Usage

1. Install the pack:
   ```bash
   gl pack install {name}
   ```

2. Use in your pipeline:
   ```yaml
   # In your pipeline configuration
   dependencies:
     - {name}
   ```

## Development

To modify this pack:

1. Make your changes
2. Validate: `gl pack validate`
3. Test: `gl run gl.yaml`

## License

This pack is distributed under the MIT License.
"""

    (docs_dir / "README.md").write_text(readme_content)