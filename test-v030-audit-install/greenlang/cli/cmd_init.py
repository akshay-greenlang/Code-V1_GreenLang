"""
gl init - Initialize new projects, packs, and datasets
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
import yaml
import json

from ..cards.generator import (
    generate_pack_card,
    generate_dataset_card,
)

app = typer.Typer()
console = Console()


@app.command()
def pack_basic(
    name: str = typer.Argument(..., help="Pack name"),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Target directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
):
    """
    Initialize a basic pack structure
    """
    create_pack("pack-basic", name, path, force)


@app.callback(invoke_without_command=False)
def init(
    ctx: typer.Context,
):
    """
    Initialize a new GreenLang project/pack/dataset

    Templates:
        - pack-basic: Basic pack structure
        - dataset-basic: Dataset with versioning
        - example-hvac: HVAC optimization example
        - example-boiler-solar: Boiler + solar example
    """


def create_pack(template: str, name: str, path: Path, force: bool):

    # Determine name from template if not provided
    if not name:
        if template.startswith("example-"):
            name = template.replace("example-", "")
        else:
            console.print("[red]Error: --name is required[/red]")
            raise typer.Exit(1)

    target_dir = path / name

    # Check if directory exists
    if target_dir.exists() and not force:
        console.print(f"[red]Error: Directory already exists: {target_dir}[/red]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)

    # Create directory structure based on template
    if template == "pack-basic":
        create_basic_pack(target_dir, name)
    elif template == "dataset-basic":
        create_basic_dataset(target_dir, name)
    elif template == "example-hvac":
        create_hvac_example(target_dir, name)
    elif template == "example-boiler-solar":
        create_boiler_solar_example(target_dir, name)
    else:
        console.print(f"[red]Unknown template: {template}[/red]")
        raise typer.Exit(1)

    # Success message
    console.print(
        Panel.fit(
            f"[green][OK][/green] Created {template} project: [cyan]{name}[/cyan]\n\n"
            f"Location: {target_dir}\n\n"
            f"Next steps:\n"
            f"  1. cd {name}\n"
            f"  2. Edit pack.yaml\n"
            f"  3. gl pack validate\n"
            f"  4. gl pack publish",
            title="Project Initialized",
        )
    )


def create_basic_pack(path: Path, name: str):
    """Create a basic pack structure"""
    path.mkdir(parents=True, exist_ok=True)

    # Create directories
    (path / "agents").mkdir(exist_ok=True)
    (path / "pipelines").mkdir(exist_ok=True)
    (path / "data").mkdir(exist_ok=True)
    (path / "tests").mkdir(exist_ok=True)
    (path / "policies").mkdir(exist_ok=True)

    # Create pack.yaml
    manifest = {
        "name": name,
        "version": "0.1.0",
        "kind": "pack",
        "description": f"A GreenLang pack for {name}",
        "license": "MIT",
        "authors": [{"name": "Your Name", "email": "you@example.com"}],
        "contents": {"pipelines": [], "agents": [], "datasets": []},
        "dependencies": [{"name": "greenlang-sdk", "version": ">=0.1.0"}],
        "security": {"sbom": "sbom.spdx.json", "signatures": "signatures/"},
        "tests": ["tests/test_*.py"],
    }

    with open(path / "pack.yaml", "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    # Create gl.yaml (pipeline definition)
    pipeline = {
        "version": "1.0",
        "name": f"{name}-pipeline",
        "description": f"Main pipeline for {name}",
        "inputs": {},
        "steps": [],
        "outputs": {},
    }

    with open(path / "gl.yaml", "w") as f:
        yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)

    # Create CARD.md using generator
    card = generate_pack_card(
        name=name,
        version="0.1.0",
        description=f"A GreenLang pack for {name}",
        purpose=f"This pack provides climate intelligence functionality for {name}",
        author="Your Name",
        license="MIT",
        tags=["climate", "greenlang", "sustainability"],
        minimal=False,  # Use full template
        # Additional details
        assumptions="- Input data is properly formatted\n- Environment variables are configured",
        limitations="- Maximum input size: 10MB\n- Requires Python 3.8+",
        carbon_footprint="Estimated: 0.1 kg CO2 per 1000 runs",
        # Examples
        quick_start=f"""from {name.replace('-', '_')} import main

# Run the main pipeline
result = main.run({{"input": "data"}})
print(result)""",
        dependencies="- greenlang>=0.1.0\n- numpy>=1.19.0",
        ethical_considerations="This pack is designed with environmental sustainability in mind. All algorithms are optimized for energy efficiency.",
    )

    with open(path / "CARD.md", "w") as f:
        f.write(card)

    # Create README.md
    readme = f"""# {name}

A GreenLang pack for climate intelligence.

## Installation

```bash
gl pack add {name}
```

## Usage

```bash
gl run {name}
```

## Development

```bash
# Validate pack
gl pack validate

# Run tests
pytest tests/

# Publish
gl pack publish
```
"""

    with open(path / "README.md", "w") as f:
        f.write(readme)

    # Create .gitignore
    gitignore = """__pycache__/
*.pyc
.pytest_cache/
.coverage
*.egg-info/
dist/
build/
.env
.gl_cache/
sbom.spdx.json
signatures/
"""

    with open(path / ".gitignore", "w") as f:
        f.write(gitignore)

    # Create sample test
    test_code = f"""import pytest
from pathlib import Path

def test_pack_structure():
    \"\"\"Test that pack has required structure\"\"\"
    pack_dir = Path(__file__).parent.parent
    assert (pack_dir / "pack.yaml").exists()
    assert (pack_dir / "gl.yaml").exists()
    assert (pack_dir / "CARD.md").exists()

def test_manifest_valid():
    \"\"\"Test that manifest is valid YAML\"\"\"
    import yaml
    pack_dir = Path(__file__).parent.parent
    with open(pack_dir / "pack.yaml") as f:
        manifest = yaml.safe_load(f)
    assert manifest["name"] == "{name}"
    assert "version" in manifest
"""

    with open(path / "tests" / "test_pack.py", "w") as f:
        f.write(test_code)


def create_basic_dataset(path: Path, name: str):
    """Create a basic dataset structure"""
    path.mkdir(parents=True, exist_ok=True)

    # Create directories
    (path / "raw").mkdir(exist_ok=True)
    (path / "processed").mkdir(exist_ok=True)
    (path / "schemas").mkdir(exist_ok=True)
    (path / "docs").mkdir(exist_ok=True)

    # Create dataset.yaml
    dataset_manifest = {
        "name": name,
        "version": "1.0.0",
        "kind": "dataset",
        "description": f"A GreenLang dataset for {name}",
        "license": "CC-BY-4.0",
        "sources": [{"name": "primary", "url": "https://example.com/data"}],
        "schema": "schemas/schema.json",
        "files": {"raw": ["raw/*.csv"], "processed": ["processed/*.parquet"]},
        "metadata": {
            "created": "2024-01-01",
            "updated": "2024-01-01",
            "rows": 0,
            "columns": 0,
        },
    }

    with open(path / "dataset.yaml", "w") as f:
        yaml.dump(dataset_manifest, f, default_flow_style=False, sort_keys=False)

    # Create schema
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {},
        "required": [],
    }

    with open(path / "schemas" / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # Create CARD.md using generator
    card = generate_dataset_card(
        name=name,
        format="csv",
        size="TBD",
        samples=0,
        features=["timestamp", "value", "metadata"],
        license="CC-BY-4.0",
        summary=f"A climate dataset for {name}",
        minimal=False,
        # Additional details
        purpose="This dataset supports climate intelligence research and analysis",
        supported_tasks="- Time series analysis\n- Anomaly detection\n- Predictive modeling",
        data_collection="Data collected from environmental sensors and public databases",
        annotation_process="Automated validation with manual quality checks",
        social_impact="Enables better understanding of climate patterns and impacts",
        carbon_emissions="Minimal - efficient data collection and storage",
        quality_checks="- Schema validation\n- Range checks\n- Temporal consistency",
        load_example=f"""from greenlang.datasets import load_dataset

# Load the dataset
dataset = load_dataset("{name}")

# Access data
for sample in dataset:
    print(sample)""",
    )

    with open(path / "CARD.md", "w") as f:
        f.write(card)

    # Create README
    readme = f"""# {name} Dataset

## Description
{name} dataset for GreenLang climate intelligence.

## Structure
- `raw/`: Original data files
- `processed/`: Cleaned and transformed data
- `schemas/`: Data schemas and validation
- `docs/`: Documentation

## Usage

```python
from greenlang.data import DatasetLoader

loader = DatasetLoader()
dataset = loader.load("{name}")
```

## License
CC-BY-4.0
"""

    with open(path / "README.md", "w") as f:
        f.write(readme)


def create_hvac_example(path: Path, name: str):
    """Create HVAC optimization example"""
    create_basic_pack(path, name)

    # Add HVAC-specific pipeline
    hvac_pipeline = {
        "version": "1.0",
        "name": "hvac-optimization",
        "description": "HVAC energy optimization pipeline",
        "inputs": {
            "building_area": {"type": "number", "unit": "sqft"},
            "occupancy": {"type": "number", "unit": "people"},
            "climate_zone": {"type": "string"},
            "hvac_type": {"type": "string"},
        },
        "steps": [
            {
                "name": "load_profile",
                "agent": "building-profile",
                "inputs": ["building_area", "occupancy"],
            },
            {
                "name": "calculate_load",
                "agent": "hvac-load",
                "inputs": ["load_profile", "climate_zone"],
            },
            {
                "name": "optimize",
                "agent": "hvac-optimizer",
                "inputs": ["calculate_load", "hvac_type"],
            },
        ],
        "outputs": {
            "energy_use": {"type": "number", "unit": "kWh"},
            "cost": {"type": "number", "unit": "USD"},
            "emissions": {"type": "number", "unit": "kgCO2e"},
        },
    }

    with open(path / "pipelines" / "hvac.yaml", "w") as f:
        yaml.dump(hvac_pipeline, f, default_flow_style=False, sort_keys=False)

    # Update manifest
    manifest_path = path / "pack.yaml"
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    manifest["contents"]["pipelines"] = ["hvac"]
    manifest["description"] = "HVAC optimization example pack"

    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)


def create_boiler_solar_example(path: Path, name: str):
    """Create boiler + solar integration example"""
    create_basic_pack(path, name)

    # Add boiler-solar pipeline
    boiler_solar_pipeline = {
        "version": "1.0",
        "name": "boiler-solar-integration",
        "description": "Boiler and solar system integration analysis",
        "inputs": {
            "boiler_capacity": {"type": "number", "unit": "kW"},
            "solar_capacity": {"type": "number", "unit": "kW"},
            "location": {"type": "string"},
            "annual_demand": {"type": "number", "unit": "kWh"},
        },
        "steps": [
            {
                "name": "solar_generation",
                "agent": "solar-estimator",
                "inputs": ["solar_capacity", "location"],
            },
            {
                "name": "boiler_efficiency",
                "agent": "boiler-analyzer",
                "inputs": ["boiler_capacity", "annual_demand"],
            },
            {
                "name": "integration",
                "agent": "system-integrator",
                "inputs": ["solar_generation", "boiler_efficiency"],
            },
            {
                "name": "optimization",
                "agent": "energy-optimizer",
                "inputs": ["integration", "annual_demand"],
            },
        ],
        "outputs": {
            "total_energy": {"type": "number", "unit": "kWh"},
            "solar_contribution": {"type": "number", "unit": "%"},
            "emissions_saved": {"type": "number", "unit": "kgCO2e"},
            "roi": {"type": "number", "unit": "years"},
        },
    }

    with open(path / "pipelines" / "boiler_solar.yaml", "w") as f:
        yaml.dump(boiler_solar_pipeline, f, default_flow_style=False, sort_keys=False)

    # Update manifest
    manifest_path = path / "pack.yaml"
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    manifest["contents"]["pipelines"] = ["boiler_solar"]
    manifest["description"] = "Boiler and solar system integration example"

    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
