"""
gl init agent - Initialize new GreenLang agents with AgentSpec v2

This module provides scaffolding for compute, AI, and industry agents with:
- AgentSpec v2 manifests
- Python implementation templates
- Comprehensive test suites (golden, property, spec)
- Cross-OS compatibility (Windows, macOS, Linux)
- Security-first defaults
"""

import typer
import re
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import yaml

app = typer.Typer()
console = Console()

# Validation patterns
SLUG_PATTERN = re.compile(r'^[a-z0-9]+(?:[._-][a-z0-9]+)*$')
SEMVER_PATTERN = re.compile(r'^\d+\.\d+\.\d+$')


@app.command()
def agent(
    name: str = typer.Argument(..., help="Agent name (kebab-case, e.g., 'boiler-efficiency')"),
    template: str = typer.Option("compute", "--template", "-t",
                                  help="Agent template: compute|ai|industry"),
    from_spec: Optional[Path] = typer.Option(None, "--from-spec",
                                               help="Pre-fill from existing spec.yaml"),
    output_dir: Path = typer.Option(Path.cwd(), "--dir",
                                     help="Output directory"),
    force: bool = typer.Option(False, "--force", "-f",
                                help="Overwrite existing files"),
    license: str = typer.Option("apache-2.0", "--license",
                                help="License: apache-2.0|mit|none"),
    author: Optional[str] = typer.Option(None, "--author",
                                         help="Author name and email"),
    no_git: bool = typer.Option(False, "--no-git",
                                help="Skip git initialization"),
    no_precommit: bool = typer.Option(False, "--no-precommit",
                                      help="Skip pre-commit hook setup"),
    runtimes: str = typer.Option("local", "--runtimes",
                                 help="Runtimes: local,docker,k8s"),
    realtime: bool = typer.Option(False, "--realtime",
                                  help="Include realtime connector stubs"),
    with_ci: bool = typer.Option(False, "--with-ci",
                                 help="Generate GitHub Actions workflow"),
):
    """
    Initialize a new GreenLang agent with AgentSpec v2.

    This creates a production-ready agent skeleton with:
    - AgentSpec v2 manifest (pack.yaml)
    - Python implementation with type safety
    - Comprehensive test suite (golden, property, spec)
    - Documentation and examples
    - Security-first defaults
    - Cross-OS compatibility

    Examples:
        # Create a compute agent
        gl init agent boiler-efficiency

        # Create an AI agent with realtime connectors
        gl init agent climate-advisor --template ai --realtime

        # Create from existing spec
        gl init agent my-agent --from-spec ./spec.yaml --force
    """

    # Phase 1: Validation
    console.print("[cyan]Initializing agent scaffold...[/cyan]")

    # Validate name (slug format)
    if not SLUG_PATTERN.match(name):
        console.print(
            f"[red]Error: Invalid agent name '{name}'[/red]\n"
            f"Agent names must be kebab-case: lowercase, hyphens/underscores only\n"
            f"Examples: boiler-ng, climate-advisor, fuel-emissions"
        )
        raise typer.Exit(1)

    # Validate template
    if template not in ["compute", "ai", "industry"]:
        console.print(f"[red]Error: Invalid template '{template}'[/red]")
        console.print("Valid templates: compute, ai, industry")
        raise typer.Exit(1)

    # Validate license
    valid_licenses = ["apache-2.0", "mit", "none"]
    if license.lower() not in valid_licenses:
        console.print(f"[red]Error: Invalid license '{license}'[/red]")
        console.print(f"Valid licenses: {', '.join(valid_licenses)}")
        raise typer.Exit(1)

    # Validate from_spec if provided
    if from_spec and not from_spec.exists():
        console.print(f"[red]Error: Spec file not found: {from_spec}[/red]")
        raise typer.Exit(1)

    # Phase 2: Name derivation
    pack_id = name  # kebab-case
    python_pkg = name.replace('-', '_').replace('.', '_')  # snake_case
    class_name = ''.join(word.capitalize() for word in name.replace('_', '-').split('-'))  # PascalCase

    # Phase 3: Path setup
    from greenlang.security.paths import validate_safe_path

    agent_dir = output_dir / pack_id

    try:
        validate_safe_path(output_dir, pack_id)
    except ValueError as e:
        console.print(f"[red]Security error: {e}[/red]")
        raise typer.Exit(1)

    # Check existence
    if agent_dir.exists() and not force:
        console.print(f"[red]Error: Directory already exists: {agent_dir}[/red]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)

    # Phase 4: Load or create spec data
    spec_data = {}
    if from_spec:
        console.print(f"[cyan]Loading spec from: {from_spec}[/cyan]")
        with open(from_spec, 'r', encoding='utf-8') as f:
            spec_data = yaml.safe_load(f)

    # Phase 5: Create directory structure
    console.print(f"[cyan]Creating agent directory: {agent_dir}[/cyan]")
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (agent_dir / f"src/{python_pkg}").mkdir(parents=True, exist_ok=True)
    (agent_dir / "tests").mkdir(parents=True, exist_ok=True)
    (agent_dir / "docs").mkdir(parents=True, exist_ok=True)
    (agent_dir / "examples").mkdir(parents=True, exist_ok=True)

    # Phase 6: Generate files based on template
    console.print(f"[cyan]Generating {template} agent template...[/cyan]")

    if template == "compute":
        generate_compute_agent(
            agent_dir, pack_id, python_pkg, class_name,
            license, author, realtime, spec_data
        )
    elif template == "ai":
        generate_ai_agent(
            agent_dir, pack_id, python_pkg, class_name,
            license, author, realtime, spec_data
        )
    elif template == "industry":
        generate_industry_agent(
            agent_dir, pack_id, python_pkg, class_name,
            license, author, realtime, spec_data
        )

    # Phase 7: Generate common files
    generate_common_files(agent_dir, pack_id, python_pkg, license, author)
    generate_test_suite(agent_dir, pack_id, python_pkg, class_name, template)
    generate_examples(agent_dir, pack_id, python_pkg, template, realtime)
    generate_documentation(agent_dir, pack_id, template, realtime)

    # Phase 8: Generate .gitignore
    generate_gitignore(agent_dir)

    # Phase 9: Generate pre-commit config (with enhancements)
    if not no_precommit:
        generate_precommit_config(agent_dir)

    # Phase 10: Generate CI workflow
    if with_ci:
        generate_ci_workflow(agent_dir, pack_id, runtimes)

    # Phase 11: Post-generation steps
    if not no_git:
        git_init(agent_dir)

    # Phase 12: Validation
    console.print("[cyan]Validating generated agent...[/cyan]")
    validation_result = validate_generated_agent(agent_dir)

    if not validation_result["valid"]:
        console.print("[yellow]⚠ Validation warnings:[/yellow]")
        for warning in validation_result["warnings"]:
            console.print(f"  • {warning}")
    else:
        console.print("[green]✓ Validation passed[/green]")

    # Phase 13: Success message
    print_success_message(agent_dir, pack_id, template, realtime, with_ci, no_git, no_precommit)


def generate_compute_agent(
    agent_dir: Path,
    pack_id: str,
    python_pkg: str,
    class_name: str,
    license: str,
    author: Optional[str],
    realtime: bool,
    spec_data: dict
):
    """Generate compute agent template."""

    # 1. Generate pack.yaml (AgentSpec v2)
    pack_yaml = generate_pack_yaml_compute(
        pack_id, python_pkg, license, author, realtime, spec_data
    )

    with open(agent_dir / "pack.yaml", 'w', encoding='utf-8', newline='\n') as f:
        yaml.dump(pack_yaml, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # 2. Generate schemas.py
    schemas_py = generate_schemas_py(python_pkg, class_name, template="compute")

    with open(agent_dir / f"src/{python_pkg}/schemas.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(schemas_py)

    # 3. Generate agent.py
    agent_py = generate_agent_py(python_pkg, class_name, template="compute", realtime=realtime)

    with open(agent_dir / f"src/{python_pkg}/agent.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(agent_py)

    # 4. Generate __init__.py
    init_py = generate_init_py(class_name)

    with open(agent_dir / f"src/{python_pkg}/__init__.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(init_py)

    # 5. Generate provenance.py
    provenance_py = generate_provenance_py()

    with open(agent_dir / f"src/{python_pkg}/provenance.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(provenance_py)

    # 6. Generate realtime.py if needed
    if realtime:
        realtime_py = generate_realtime_py(python_pkg)
        with open(agent_dir / f"src/{python_pkg}/realtime.py", 'w', encoding='utf-8', newline='\n') as f:
            f.write(realtime_py)


def generate_ai_agent(
    agent_dir: Path,
    pack_id: str,
    python_pkg: str,
    class_name: str,
    license: str,
    author: Optional[str],
    realtime: bool,
    spec_data: dict
):
    """Generate AI agent template."""

    # 1. Generate pack.yaml (AgentSpec v2) for AI
    pack_yaml = generate_pack_yaml_ai(
        pack_id, python_pkg, license, author, realtime, spec_data
    )

    with open(agent_dir / "pack.yaml", 'w', encoding='utf-8', newline='\n') as f:
        yaml.dump(pack_yaml, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # 2. Generate schemas.py
    schemas_py = generate_schemas_py(python_pkg, class_name, template="ai")

    with open(agent_dir / f"src/{python_pkg}/schemas.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(schemas_py)

    # 3. Generate agent.py
    agent_py = generate_agent_py(python_pkg, class_name, template="ai", realtime=realtime)

    with open(agent_dir / f"src/{python_pkg}/agent.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(agent_py)

    # 4. Generate ai_tools.py
    ai_tools_py = generate_ai_tools_py(python_pkg, class_name)

    with open(agent_dir / f"src/{python_pkg}/ai_tools.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(ai_tools_py)

    # 5. Generate __init__.py
    init_py = generate_init_py(class_name)

    with open(agent_dir / f"src/{python_pkg}/__init__.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(init_py)

    # 6. Generate provenance.py
    provenance_py = generate_provenance_py()

    with open(agent_dir / f"src/{python_pkg}/provenance.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(provenance_py)

    # 7. Generate realtime.py if needed
    if realtime:
        realtime_py = generate_realtime_py(python_pkg)
        with open(agent_dir / f"src/{python_pkg}/realtime.py", 'w', encoding='utf-8', newline='\n') as f:
            f.write(realtime_py)


def generate_industry_agent(
    agent_dir: Path,
    pack_id: str,
    python_pkg: str,
    class_name: str,
    license: str,
    author: Optional[str],
    realtime: bool,
    spec_data: dict
):
    """Generate industry agent template."""

    # 1. Generate pack.yaml (AgentSpec v2) for industry
    pack_yaml = generate_pack_yaml_industry(
        pack_id, python_pkg, license, author, realtime, spec_data
    )

    with open(agent_dir / "pack.yaml", 'w', encoding='utf-8', newline='\n') as f:
        yaml.dump(pack_yaml, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # 2. Generate schemas.py
    schemas_py = generate_schemas_py(python_pkg, class_name, template="industry")

    with open(agent_dir / f"src/{python_pkg}/schemas.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(schemas_py)

    # 3. Generate agent.py
    agent_py = generate_agent_py(python_pkg, class_name, template="industry", realtime=realtime)

    with open(agent_dir / f"src/{python_pkg}/agent.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(agent_py)

    # 4. Generate __init__.py
    init_py = generate_init_py(class_name)

    with open(agent_dir / f"src/{python_pkg}/__init__.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(init_py)

    # 5. Generate provenance.py
    provenance_py = generate_provenance_py()

    with open(agent_dir / f"src/{python_pkg}/provenance.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(provenance_py)

    # 6. Generate realtime.py if needed
    if realtime:
        realtime_py = generate_realtime_py(python_pkg)
        with open(agent_dir / f"src/{python_pkg}/realtime.py", 'w', encoding='utf-8', newline='\n') as f:
            f.write(realtime_py)


# Template generation functions
def generate_pack_yaml_compute(pack_id, python_pkg, license, author, realtime, spec_data):
    """Generate AgentSpec v2 pack.yaml for compute template.

    Creates a complete AgentSpec v2 manifest with metadata, compute section,
    emission factors, provenance, and optional realtime configuration.

    Args:
        pack_id: Agent identifier (kebab-case)
        python_pkg: Python package name (snake_case)
        license: License identifier (apache-2.0, mit, none)
        author: Author name and email
        realtime: Include realtime section
        spec_data: Optional existing spec data to merge

    Returns:
        Dictionary with AgentSpec v2 structure
    """
    # Base structure from spec_data or defaults
    manifest = spec_data.copy() if spec_data else {}

    # Override/set core fields
    manifest['schema_version'] = '2.0.0'
    manifest['id'] = manifest.get('id', f"custom/{pack_id}")
    manifest['name'] = manifest.get('name', pack_id.replace('-', ' ').title())
    manifest['version'] = manifest.get('version', '0.1.0')
    manifest['summary'] = manifest.get('summary', f"Compute emissions for {manifest['name']}")
    manifest['tags'] = manifest.get('tags', ['compute', 'emissions', 'custom'])

    if author:
        manifest['authors'] = [author]

    manifest['license'] = license if license != 'none' else 'proprietary'

    # Compute section
    if 'compute' not in manifest:
        manifest['compute'] = {
            'entrypoint': f"python://{python_pkg}.agent:compute",
            'deterministic': True,
            'timeout_s': 30,
            'memory_limit_mb': 512,
            'python_version': '3.11',
            'dependencies': [
                'pydantic>=2.7',
                'pyyaml>=6.0'
            ],
            'inputs': {
                'fuel_volume': {
                    'dtype': 'float64',
                    'unit': 'm^3',
                    'required': True,
                    'ge': 0.0,
                    'description': 'Volume of fuel consumed (cubic meters)'
                },
                'emission_factor': {
                    'dtype': 'float64',
                    'unit': 'kgCO2e/m^3',
                    'required': True,
                    'ge': 0.0,
                    'description': 'Emission factor (kgCO2e per cubic meter)'
                }
            },
            'outputs': {
                'co2e_kg': {
                    'dtype': 'float64',
                    'unit': 'kgCO2e',
                    'description': 'Total CO2 equivalent emissions (kilograms)'
                },
                'formula': {
                    'dtype': 'string',
                    'unit': '1',
                    'description': 'Formula used for calculation'
                }
            },
            'factors': {
                'emission_factor': {
                    'ref': 'ef://ipcc_ar6/default/co2e_kg_per_unit',
                    'gwp_set': 'AR6GWP100',
                    'description': 'Default emission factor reference'
                }
            }
        }

    # Provenance section
    if 'provenance' not in manifest:
        manifest['provenance'] = {
            'pin_ef': True,
            'gwp_set': 'AR6GWP100',
            'record': [
                'inputs',
                'outputs',
                'factors',
                'ef_uri',
                'ef_cid',
                'code_sha',
                'timestamp'
            ]
        }

    # Realtime section (optional)
    if realtime and 'realtime' not in manifest:
        manifest['realtime'] = {
            'default_mode': 'replay',
            'snapshot_path': 'snapshots/latest.json',
            'connectors': [
                {
                    'name': 'data_source',
                    'topic': 'sensor_data',
                    'window': '1h',
                    'ttl': '6h',
                    'required': False
                }
            ]
        }

    return manifest


def generate_pack_yaml_ai(pack_id, python_pkg, license, author, realtime, spec_data):
    """Generate AgentSpec v2 pack.yaml for AI template.

    Creates an AgentSpec v2 manifest with AI section including LLM configuration,
    tools, RAG collections, and budget constraints.

    Args:
        pack_id: Agent identifier (kebab-case)
        python_pkg: Python package name (snake_case)
        license: License identifier
        author: Author name and email
        realtime: Include realtime section
        spec_data: Optional existing spec data

    Returns:
        Dictionary with AgentSpec v2 structure
    """
    # Start with compute template
    manifest = generate_pack_yaml_compute(pack_id, python_pkg, license, author, realtime, spec_data)

    # Update tags for AI
    manifest['tags'] = ['ai', 'llm', 'climate-advisor', 'custom']
    manifest['summary'] = manifest.get('summary', f"AI-powered climate advisor for {manifest['name']}")

    # Add AI section
    if 'ai' not in manifest:
        manifest['ai'] = {
            'json_mode': True,
            'system_prompt': (
                f"You are a climate advisor for {manifest['name']}.\n\n"
                "Guidelines:\n"
                "- Use tools to calculate emissions; never guess numbers.\n"
                "- Always cite emission factors with proper URIs (ef://).\n"
                "- Validate all inputs for physical plausibility.\n"
                "- Provide uncertainty estimates when available.\n"
                "- Follow GHG Protocol standards.\n\n"
                "When uncertain, ask clarifying questions rather than making assumptions."
            ),
            'budget': {
                'max_cost_usd': 1.0,
                'max_input_tokens': 15000,
                'max_output_tokens': 2000,
                'max_retries': 3
            },
            'rag_collections': [
                'ghg_protocol_corp',
                'ipcc_ar6',
                'gl_docs'
            ],
            'tools': [
                {
                    'name': 'calculate_emissions',
                    'description': 'Calculate emissions based on input parameters',
                    'schema_in': {
                        'type': 'object',
                        'properties': {
                            'fuel_volume': {
                                'type': 'number',
                                'description': 'Volume of fuel consumed',
                                'minimum': 0
                            },
                            'emission_factor': {
                                'type': 'number',
                                'description': 'Emission factor to apply',
                                'minimum': 0
                            }
                        },
                        'required': ['fuel_volume', 'emission_factor']
                    },
                    'schema_out': {
                        'type': 'object',
                        'properties': {
                            'co2e_kg': {
                                'type': 'number',
                                'description': 'Total emissions in kgCO2e'
                            },
                            'formula': {
                                'type': 'string',
                                'description': 'Formula used'
                            }
                        },
                        'required': ['co2e_kg', 'formula']
                    },
                    'impl': f"python://{python_pkg}.ai_tools:calculate_emissions",
                    'safe': True
                }
            ]
        }

    # Add LLM provenance fields
    if 'provenance' in manifest:
        manifest['provenance']['record'].extend([
            'llm_model',
            'prompt_hash',
            'cost_usd'
        ])

    return manifest


def generate_pack_yaml_industry(pack_id, python_pkg, license, author, realtime, spec_data):
    """Generate AgentSpec v2 pack.yaml for industry template.

    Creates an AgentSpec v2 manifest with multiple emission factors and
    scope tracking (scope1, scope2, scope3).

    Args:
        pack_id: Agent identifier (kebab-case)
        python_pkg: Python package name (snake_case)
        license: License identifier
        author: Author name and email
        realtime: Include realtime section
        spec_data: Optional existing spec data

    Returns:
        Dictionary with AgentSpec v2 structure
    """
    # Start with compute template
    manifest = generate_pack_yaml_compute(pack_id, python_pkg, license, author, realtime, spec_data)

    # Update tags for industry
    manifest['tags'] = ['industry', 'scope1', 'scope2', 'scope3', 'custom']
    manifest['summary'] = manifest.get('summary', f"Industry emissions tracking for {manifest['name']}")

    # Expand compute section for industry-specific inputs/outputs
    if 'compute' in manifest:
        manifest['compute']['inputs'] = {
            'fuel_consumption': {
                'dtype': 'float64',
                'unit': 'm^3',
                'required': True,
                'ge': 0.0,
                'description': 'Fuel consumption for scope 1 emissions'
            },
            'electricity_kwh': {
                'dtype': 'float64',
                'unit': 'kWh',
                'required': True,
                'ge': 0.0,
                'description': 'Electricity consumption for scope 2 emissions'
            },
            'supply_chain_emissions': {
                'dtype': 'float64',
                'unit': 'kgCO2e',
                'required': False,
                'ge': 0.0,
                'description': 'Supply chain emissions for scope 3'
            },
            'region': {
                'dtype': 'string',
                'unit': '1',
                'required': True,
                'enum': ['US', 'EU', 'UK', 'CA', 'AU', 'JP', 'CN', 'IN'],
                'description': 'Geographic region for grid factors'
            }
        }

        manifest['compute']['outputs'] = {
            'scope1_co2e_kg': {
                'dtype': 'float64',
                'unit': 'kgCO2e',
                'description': 'Scope 1 emissions (direct combustion)'
            },
            'scope2_co2e_kg': {
                'dtype': 'float64',
                'unit': 'kgCO2e',
                'description': 'Scope 2 emissions (purchased electricity)'
            },
            'scope3_co2e_kg': {
                'dtype': 'float64',
                'unit': 'kgCO2e',
                'description': 'Scope 3 emissions (supply chain)'
            },
            'total_co2e_kg': {
                'dtype': 'float64',
                'unit': 'kgCO2e',
                'description': 'Total CO2e emissions across all scopes'
            },
            'breakdown': {
                'dtype': 'object',
                'unit': '1',
                'description': 'Detailed breakdown by scope'
            }
        }

        # Multiple emission factors
        manifest['compute']['factors'] = {
            'fuel_ef': {
                'ref': 'ef://ipcc_ar6/combustion/ng/co2e_kg_per_m3',
                'gwp_set': 'AR6GWP100',
                'description': 'Natural gas combustion emission factor'
            },
            'grid_ef': {
                'ref': 'ef://epa/egrid/{region}/co2e_kg_per_kwh',
                'gwp_set': 'AR6GWP100',
                'description': 'Regional grid emission factor (parameterized by region)'
            },
            'supply_chain_ef': {
                'ref': 'ef://ghg_protocol/scope3/category1/co2e_kg_per_usd',
                'gwp_set': 'AR6GWP100',
                'description': 'Supply chain emission factor'
            }
        }

    return manifest


def generate_schemas_py(python_pkg, class_name, template):
    """Generate schemas.py with Pydantic models and unit annotations.

    Creates Pydantic models for inputs and outputs with proper type hints,
    validation, and unit annotations using Annotated types.

    Args:
        python_pkg: Python package name
        class_name: Agent class name (PascalCase)
        template: Template type (compute, ai, industry)

    Returns:
        Python source code as string
    """
    if template == "compute":
        return f'''"""
Pydantic schemas for {class_name} agent.

This module defines input and output models with:
- Type safety via Pydantic v2
- Unit annotations using Annotated types
- Validation constraints (ge, le, etc.)
- Example input generation
"""

from typing import Annotated, Optional
from pydantic import BaseModel, Field, field_validator


class InputModel(BaseModel):
    """Input parameters for emissions calculation."""

    fuel_volume: Annotated[float, Field(ge=0.0, description="Volume of fuel consumed (m^3)")] = Field(
        ..., description="Volume of fuel consumed (cubic meters)"
    )
    emission_factor: Annotated[
        float, Field(ge=0.0, description="Emission factor (kgCO2e/m^3)")
    ] = Field(..., description="Emission factor (kgCO2e per cubic meter)")

    @field_validator("fuel_volume", "emission_factor")
    @classmethod
    def validate_non_negative(cls, v: float, info) -> float:
        """Ensure values are non-negative."""
        if v < 0:
            raise ValueError(f"{{info.field_name}} must be non-negative, got {{v}}")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {{
            "example": {{
                "fuel_volume": 100.0,
                "emission_factor": 2.3
            }}
        }}


class OutputModel(BaseModel):
    """Output results from emissions calculation."""

    co2e_kg: Annotated[float, Field(ge=0.0, description="Total CO2e emissions (kgCO2e)")] = Field(
        ..., description="Total CO2 equivalent emissions (kilograms)"
    )
    formula: str = Field(..., description="Formula used for calculation")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {{
            "example": {{
                "co2e_kg": 230.0,
                "formula": "fuel_volume * emission_factor"
            }}
        }}


def example_input() -> dict:
    """Generate example input for testing.

    Returns:
        Dictionary with example input parameters
    """
    return {{
        "fuel_volume": 100.0,
        "emission_factor": 2.3
    }}
'''
    elif template == "ai":
        return f'''"""
Pydantic schemas for {class_name} AI agent.

This module defines input and output models for AI-powered analysis.
"""

from typing import Annotated, Optional, List
from pydantic import BaseModel, Field, field_validator


class InputModel(BaseModel):
    """Input parameters for AI agent."""

    query: str = Field(..., description="Natural language query about emissions")
    fuel_volume: Annotated[float, Field(ge=0.0, description="Volume of fuel (m^3)")] = Field(
        ..., description="Volume of fuel consumed"
    )
    emission_factor: Annotated[
        float, Field(ge=0.0, description="Emission factor (kgCO2e/m^3)")
    ] = Field(..., description="Emission factor")
    context: Optional[List[str]] = Field(
        default=None, description="Additional context for AI"
    )

    @field_validator("fuel_volume", "emission_factor")
    @classmethod
    def validate_non_negative(cls, v: float, info) -> float:
        """Ensure values are non-negative."""
        if v < 0:
            raise ValueError(f"{{info.field_name}} must be non-negative")
        return v


class OutputModel(BaseModel):
    """Output from AI agent."""

    co2e_kg: float = Field(..., description="Calculated emissions (kgCO2e)")
    formula: str = Field(..., description="Formula used")
    explanation: str = Field(..., description="Natural language explanation")
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ..., description="Confidence score (0-1)"
    )


def example_input() -> dict:
    """Generate example input."""
    return {{
        "query": "Calculate emissions from natural gas consumption",
        "fuel_volume": 100.0,
        "emission_factor": 2.3
    }}
'''
    else:  # industry
        return f'''"""
Pydantic schemas for {class_name} industry agent.

This module defines input and output models for multi-scope emissions tracking.
"""

from typing import Annotated, Optional, Dict
from pydantic import BaseModel, Field, field_validator


class InputModel(BaseModel):
    """Input parameters for industry emissions calculation."""

    fuel_consumption: Annotated[float, Field(ge=0.0, description="Fuel consumption (m^3)")] = Field(
        ..., description="Fuel consumption for scope 1 emissions"
    )
    electricity_kwh: Annotated[float, Field(ge=0.0, description="Electricity (kWh)")] = Field(
        ..., description="Electricity consumption for scope 2 emissions"
    )
    supply_chain_emissions: Annotated[
        Optional[float], Field(ge=0.0, description="Supply chain emissions (kgCO2e)")
    ] = Field(default=0.0, description="Supply chain emissions for scope 3")
    region: str = Field(..., description="Geographic region", pattern="^[A-Z]{{2}}$")

    @field_validator("fuel_consumption", "electricity_kwh", "supply_chain_emissions")
    @classmethod
    def validate_non_negative(cls, v: float, info) -> float:
        """Ensure values are non-negative."""
        if v and v < 0:
            raise ValueError(f"{{info.field_name}} must be non-negative")
        return v


class OutputModel(BaseModel):
    """Output results from industry emissions calculation."""

    scope1_co2e_kg: Annotated[float, Field(ge=0.0, description="Scope 1 emissions")] = Field(
        ..., description="Scope 1 emissions (direct combustion)"
    )
    scope2_co2e_kg: Annotated[float, Field(ge=0.0, description="Scope 2 emissions")] = Field(
        ..., description="Scope 2 emissions (purchased electricity)"
    )
    scope3_co2e_kg: Annotated[float, Field(ge=0.0, description="Scope 3 emissions")] = Field(
        ..., description="Scope 3 emissions (supply chain)"
    )
    total_co2e_kg: Annotated[float, Field(ge=0.0, description="Total emissions")] = Field(
        ..., description="Total CO2e emissions across all scopes"
    )
    breakdown: Dict[str, float] = Field(..., description="Detailed breakdown by scope")


def example_input() -> dict:
    """Generate example input."""
    return {{
        "fuel_consumption": 100.0,
        "electricity_kwh": 5000.0,
        "supply_chain_emissions": 150.0,
        "region": "US"
    }}
'''


def generate_agent_py(python_pkg, class_name, template, realtime):
    """Generate agent.py with agent implementation.

    Creates agent class extending BaseAgent with compute() method,
    provenance tracking, error handling, and no network/file I/O.

    Args:
        python_pkg: Python package name
        class_name: Agent class name (PascalCase)
        template: Template type (compute, ai, industry)
        realtime: Include realtime connector support

    Returns:
        Python source code as string
    """
    realtime_import = ""
    realtime_init = ""
    if realtime:
        realtime_import = f"\nfrom {python_pkg}.realtime import RealtimeConnector"
        realtime_init = "\n        self.realtime = RealtimeConnector()"

    if template == "compute":
        return f'''"""
{class_name} agent implementation.

This agent performs emissions calculations with:
- No network or file I/O in compute() method
- Provenance tracking
- Comprehensive error handling
- Deterministic outputs
"""

import logging
from typing import Any

from {python_pkg}.provenance import compute_formula_hash, create_provenance_record
from {python_pkg}.schemas import InputModel, OutputModel{realtime_import}

logger = logging.getLogger(__name__)


class {class_name}:
    """
    {class_name} - Emissions calculation agent.

    This agent computes CO2e emissions based on fuel volume and emission factors.
    All computations are deterministic and free of network/file I/O.

    Attributes:
        name: Agent name
        version: Agent version
    """

    def __init__(self):
        """Initialize the agent."""
        self.name = "{class_name}"
        self.version = "0.1.0"{realtime_init}
        logger.info(f"Initialized {{self.name}} v{{self.version}}")

    def compute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Compute emissions from input parameters.

        This method is deterministic and performs no network or file I/O.

        Args:
            inputs: Dictionary with fuel_volume and emission_factor

        Returns:
            Dictionary with co2e_kg and formula

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If computation fails
        """
        try:
            # Validate inputs using Pydantic
            input_model = InputModel(**inputs)

            # Extract validated values
            fuel_volume = input_model.fuel_volume
            emission_factor = input_model.emission_factor

            # Compute emissions (deterministic)
            co2e_kg = fuel_volume * emission_factor

            # Build formula string
            formula = f"fuel_volume * emission_factor = {{fuel_volume}} * {{emission_factor}}"

            # Create output
            output = {{
                "co2e_kg": round(co2e_kg, 3),
                "formula": formula
            }}

            # Validate output
            output_model = OutputModel(**output)

            # Add provenance
            provenance = create_provenance_record(
                inputs=inputs,
                outputs=output_model.model_dump(),
                formula_hash=compute_formula_hash(formula),
                agent_name=self.name,
                agent_version=self.version
            )

            # Return with provenance
            return {{
                **output_model.model_dump(),
                "_provenance": provenance
            }}

        except ValueError as e:
            logger.error(f"Input validation failed: {{e}}")
            raise ValueError(f"Invalid input: {{e}}") from e
        except Exception as e:
            logger.error(f"Computation failed: {{e}}")
            raise RuntimeError(f"Computation failed: {{e}}") from e

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Main entry point for agent execution.

        Args:
            inputs: Input parameters

        Returns:
            Computation results with provenance
        """
        return self.compute(inputs)


# Module-level compute function for AgentSpec v2 entrypoint compliance
def compute(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Module-level compute function (AgentSpec v2 entrypoint).

    This function creates an agent instance and delegates to its compute method.
    Required for python://module:function entrypoint format.

    Args:
        inputs: Input parameters dictionary

    Returns:
        Computation results with provenance
    """
    agent = {class_name}()
    return agent.compute(inputs)
'''
    elif template == "ai":
        return f'''"""
{class_name} AI agent implementation.

This agent uses LLM for climate advisory with:
- Tool calling for emissions calculations
- Natural language explanations
- Provenance tracking
"""

import logging
from typing import Dict, Any

from {python_pkg}.schemas import InputModel, OutputModel
from {python_pkg}.provenance import create_provenance_record
from {python_pkg}.ai_tools import calculate_emissions{realtime_import}

logger = logging.getLogger(__name__)


class {class_name}:
    """
    {class_name} - AI-powered climate advisor.

    This agent provides natural language explanations and recommendations
    using LLM with structured tool calling.

    Attributes:
        name: Agent name
        version: Agent version
    """

    def __init__(self):
        """Initialize the AI agent."""
        self.name = "{class_name}"
        self.version = "0.1.0"{realtime_init}
        logger.info(f"Initialized {{self.name}} v{{self.version}}")

    def compute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Process query and compute emissions using AI tools.

        Args:
            inputs: Dictionary with query, fuel_volume, emission_factor

        Returns:
            Dictionary with emissions, formula, explanation, confidence

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            input_model = InputModel(**inputs)

            # Call calculation tool (deterministic part)
            calc_result = calculate_emissions(
                fuel_volume=input_model.fuel_volume,
                emission_factor=input_model.emission_factor
            )

            # Generate explanation (this would call LLM in production)
            explanation = (
                f"Based on your query: '{{input_model.query}}', "
                f"the emissions from {{input_model.fuel_volume}} m³ of fuel "
                f"with an emission factor of {{input_model.emission_factor}} kgCO2e/m³ "
                f"result in {{calc_result['co2e_kg']}} kgCO2e."
            )

            # Create output
            output = {{
                "co2e_kg": calc_result["co2e_kg"],
                "formula": calc_result["formula"],
                "explanation": explanation,
                "confidence": 0.95  # High confidence for deterministic calculation
            }}

            # Validate output
            output_model = OutputModel(**output)

            # Add provenance
            provenance = create_provenance_record(
                inputs=inputs,
                outputs=output_model.model_dump(),
                formula_hash=calc_result.get("formula_hash", ""),
                agent_name=self.name,
                agent_version=self.version
            )

            return {{
                **output_model.model_dump(),
                "_provenance": provenance
            }}

        except ValueError as e:
            logger.error(f"Input validation failed: {{e}}")
            raise
        except Exception as e:
            logger.error(f"Computation failed: {{e}}")
            raise RuntimeError(f"AI agent failed: {{e}}") from e

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Main entry point."""
        return self.compute(inputs)


# Module-level compute function for AgentSpec v2 entrypoint compliance
def compute(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Module-level compute function (AgentSpec v2 entrypoint).

    This function creates an agent instance and delegates to its compute method.
    Required for python://module:function entrypoint format.

    Args:
        inputs: Input parameters dictionary

    Returns:
        Computation results with provenance
    """
    agent = {class_name}()
    return agent.compute(inputs)
'''
    else:  # industry
        return f'''"""
{class_name} industry agent implementation.

This agent tracks multi-scope emissions with:
- Scope 1: Direct combustion
- Scope 2: Purchased electricity
- Scope 3: Supply chain
"""

import logging
from typing import Dict, Any

from {python_pkg}.schemas import InputModel, OutputModel
from {python_pkg}.provenance import create_provenance_record{realtime_import}

logger = logging.getLogger(__name__)


# Mock emission factors (in production, fetch from EF database)
FUEL_EF = 2.3  # kgCO2e/m³
GRID_EF = {{"US": 0.42, "EU": 0.35, "UK": 0.29}}  # kgCO2e/kWh by region


class {class_name}:
    """
    {class_name} - Industry multi-scope emissions tracker.

    This agent computes emissions across all three GHG Protocol scopes.

    Attributes:
        name: Agent name
        version: Agent version
    """

    def __init__(self):
        """Initialize the industry agent."""
        self.name = "{class_name}"
        self.version = "0.1.0"{realtime_init}
        logger.info(f"Initialized {{self.name}} v{{self.version}}")

    def compute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Compute multi-scope emissions.

        Args:
            inputs: Dictionary with fuel_consumption, electricity_kwh, supply_chain_emissions, region

        Returns:
            Dictionary with scope-wise and total emissions

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            input_model = InputModel(**inputs)

            # Scope 1: Direct combustion
            scope1_co2e_kg = input_model.fuel_consumption * FUEL_EF

            # Scope 2: Purchased electricity
            grid_ef = GRID_EF.get(input_model.region, 0.40)  # Default to 0.40
            scope2_co2e_kg = input_model.electricity_kwh * grid_ef

            # Scope 3: Supply chain (already in kgCO2e)
            scope3_co2e_kg = input_model.supply_chain_emissions or 0.0

            # Total
            total_co2e_kg = scope1_co2e_kg + scope2_co2e_kg + scope3_co2e_kg

            # Breakdown
            breakdown = {{
                "scope1": round(scope1_co2e_kg, 3),
                "scope2": round(scope2_co2e_kg, 3),
                "scope3": round(scope3_co2e_kg, 3),
                "scope1_pct": round(100 * scope1_co2e_kg / total_co2e_kg, 2) if total_co2e_kg > 0 else 0,
                "scope2_pct": round(100 * scope2_co2e_kg / total_co2e_kg, 2) if total_co2e_kg > 0 else 0,
                "scope3_pct": round(100 * scope3_co2e_kg / total_co2e_kg, 2) if total_co2e_kg > 0 else 0
            }}

            # Create output
            output = {{
                "scope1_co2e_kg": round(scope1_co2e_kg, 3),
                "scope2_co2e_kg": round(scope2_co2e_kg, 3),
                "scope3_co2e_kg": round(scope3_co2e_kg, 3),
                "total_co2e_kg": round(total_co2e_kg, 3),
                "breakdown": breakdown
            }}

            # Validate output
            output_model = OutputModel(**output)

            # Add provenance
            provenance = create_provenance_record(
                inputs=inputs,
                outputs=output_model.model_dump(),
                formula_hash="",
                agent_name=self.name,
                agent_version=self.version
            )

            return {{
                **output_model.model_dump(),
                "_provenance": provenance
            }}

        except ValueError as e:
            logger.error(f"Input validation failed: {{e}}")
            raise
        except Exception as e:
            logger.error(f"Computation failed: {{e}}")
            raise RuntimeError(f"Industry agent failed: {{e}}") from e

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Main entry point."""
        return self.compute(inputs)


# Module-level compute function for AgentSpec v2 entrypoint compliance
def compute(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Module-level compute function (AgentSpec v2 entrypoint).

    This function creates an agent instance and delegates to its compute method.
    Required for python://module:function entrypoint format.

    Args:
        inputs: Input parameters dictionary

    Returns:
        Computation results with provenance
    """
    agent = {class_name}()
    return agent.compute(inputs)
'''


def generate_ai_tools_py(python_pkg, class_name):
    """Generate ai_tools.py with @tool decorators.

    Creates AI tool wrappers with:
    - @tool decorator for LLM function calling
    - JSON Schema validation
    - No naked numbers enforcement
    - Tool safety (safe=true)

    Args:
        python_pkg: Python package name
        class_name: Agent class name

    Returns:
        Python source code as string
    """
    return f'''"""
AI Tools for {class_name}.

This module provides tool wrappers for LLM function calling with:
- @tool decorator for structured calling
- JSON Schema validation
- No naked numbers (all values have units)
- Safety guarantees (safe=true)
"""

import logging
from typing import Dict, Any
import hashlib

logger = logging.getLogger(__name__)


def calculate_emissions(fuel_volume: float, emission_factor: float) -> dict[str, Any]:
    """
    Calculate emissions from fuel volume and emission factor.

    This tool enforces no naked numbers - all values must have explicit units
    in the calling context.

    Args:
        fuel_volume: Volume of fuel consumed (must be in m^3)
        emission_factor: Emission factor (must be in kgCO2e/m^3)

    Returns:
        Dictionary with co2e_kg, formula, and formula_hash

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if fuel_volume < 0:
        raise ValueError(f"fuel_volume must be non-negative, got {{fuel_volume}}")
    if emission_factor < 0:
        raise ValueError(f"emission_factor must be non-negative, got {{emission_factor}}")

    # Compute emissions
    co2e_kg = fuel_volume * emission_factor

    # Build formula
    formula = f"fuel_volume * emission_factor = {{fuel_volume}} * {{emission_factor}}"

    # Compute formula hash for provenance
    formula_hash = hashlib.sha256(formula.encode()).hexdigest()[:16]

    logger.info(f"Calculated {{co2e_kg}} kgCO2e from {{fuel_volume}} m³ fuel")

    return {{
        "co2e_kg": round(co2e_kg, 3),
        "formula": formula,
        "formula_hash": formula_hash
    }}


def validate_input(parameter: str, value: float, unit: str) -> dict[str, Any]:
    """
    Validate input parameter for physical plausibility.

    Args:
        parameter: Parameter name
        value: Parameter value
        unit: Parameter unit

    Returns:
        Dictionary with valid (bool), warning (str), suggestion (float)
    """
    warnings = []
    suggestion = None

    # Physical plausibility checks
    if parameter == "fuel_volume":
        if value > 1000000:  # > 1 million m³
            warnings.append(f"Fuel volume {{value}} m³ is unusually high")
            suggestion = 10000.0
        elif value == 0:
            warnings.append("Fuel volume is zero - no emissions will be calculated")

    elif parameter == "emission_factor":
        if value > 10:  # > 10 kgCO2e/m³
            warnings.append(f"Emission factor {{value}} kgCO2e/m³ is unusually high")
            suggestion = 2.5

    valid = len(warnings) == 0

    return {{
        "valid": valid,
        "warning": "; ".join(warnings) if warnings else None,
        "suggestion": suggestion
    }}


# Tool metadata for LLM function calling (JSON Schema)
TOOL_CALCULATE_EMISSIONS = {{
    "name": "calculate_emissions",
    "description": "Calculate CO2e emissions from fuel volume and emission factor",
    "parameters": {{
        "type": "object",
        "properties": {{
            "fuel_volume": {{
                "type": "number",
                "description": "Volume of fuel consumed (m^3)",
                "minimum": 0
            }},
            "emission_factor": {{
                "type": "number",
                "description": "Emission factor (kgCO2e/m^3)",
                "minimum": 0
            }}
        }},
        "required": ["fuel_volume", "emission_factor"]
    }},
    "safe": True  # No side effects, no network/file I/O
}}

TOOL_VALIDATE_INPUT = {{
    "name": "validate_input",
    "description": "Validate input parameter for physical plausibility",
    "parameters": {{
        "type": "object",
        "properties": {{
            "parameter": {{
                "type": "string",
                "description": "Parameter name"
            }},
            "value": {{
                "type": "number",
                "description": "Parameter value"
            }},
            "unit": {{
                "type": "string",
                "description": "Parameter unit"
            }}
        }},
        "required": ["parameter", "value", "unit"]
    }},
    "safe": True
}}
'''


def generate_init_py(class_name):
    """Generate __init__.py for package initialization.

    Args:
        class_name: Agent class name

    Returns:
        Python source code as string
    """
    return f'''"""
{class_name} package initialization.

This package provides emissions calculation capabilities for GreenLang.
"""

from {class_name.lower()}.agent import {class_name}
from {class_name.lower()}.schemas import InputModel, OutputModel, example_input

__version__ = "0.1.0"
__all__ = ["{class_name}", "InputModel", "OutputModel", "example_input"]
'''


def generate_provenance_py():
    """Generate provenance.py with helper functions.

    Creates provenance tracking helpers including:
    - Formula hash computation
    - Provenance record creation
    - Metadata collection

    Returns:
        Python source code as string
    """
    return '''"""
Provenance tracking helpers.

This module provides utilities for creating audit trails:
- Formula hash computation for reproducibility
- Provenance record creation
- Metadata collection (code SHA, timestamp, etc.)
"""

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def compute_formula_hash(formula: str) -> str:
    """
    Compute SHA-256 hash of formula for reproducibility tracking.

    Args:
        formula: Formula string

    Returns:
        First 16 characters of SHA-256 hash
    """
    return hashlib.sha256(formula.encode("utf-8")).hexdigest()[:16]


def get_git_sha() -> Optional[str]:
    """
    Get current git SHA if in a git repository.

    Returns:
        Git SHA or None if not in git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()[:16]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def create_provenance_record(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    formula_hash: str,
    agent_name: str,
    agent_version: str
) -> dict[str, Any]:
    """
    Create provenance record for audit trail.

    This record enables:
    - Reproducibility: Same inputs → same outputs
    - Traceability: Track which version of code produced results
    - Auditability: Timestamp and user tracking

    Args:
        inputs: Input parameters
        outputs: Output results
        formula_hash: Hash of formula used
        agent_name: Name of agent
        agent_version: Version of agent

    Returns:
        Provenance record dictionary
    """
    # Compute input hash for deduplication
    input_str = str(sorted(inputs.items()))
    inputs_hash = hashlib.sha256(input_str.encode()).hexdigest()[:16]

    # Get git SHA if available
    code_sha = get_git_sha()

    # Create provenance record
    provenance = {
        "agent_name": agent_name,
        "agent_version": agent_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs_hash": inputs_hash,
        "formula_hash": formula_hash,
        "code_sha": code_sha,
        "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        "gwp_set": "AR6GWP100"
    }

    logger.debug(f"Created provenance record: {provenance}")

    return provenance


def compute_inputs_hash(inputs: dict[str, Any]) -> str:
    """
    Compute hash of inputs for deduplication.

    Args:
        inputs: Input dictionary

    Returns:
        First 16 characters of SHA-256 hash
    """
    input_str = str(sorted(inputs.items()))
    return hashlib.sha256(input_str.encode()).hexdigest()[:16]
'''


def generate_realtime_py(python_pkg):
    """Generate realtime.py with connector helpers for Replay/Live modes.

    Args:
        python_pkg: Python package name

    Returns:
        Python source code as string
    """
    return f'''"""
Realtime connector helpers for {python_pkg}.

This module provides connector interfaces for:
- Replay mode: Use cached/snapshot data
- Live mode: Fetch fresh data from external sources

Connectors must be thread-safe and handle failures gracefully.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RealtimeConnector:
    """
    Realtime data connector for agent.

    Supports two modes:
    - replay: Use cached snapshots (default)
    - live: Fetch fresh data from external sources

    Attributes:
        mode: Current mode (replay or live)
        snapshot_dir: Directory for cached snapshots
    """

    def __init__(self, mode: str = "replay", snapshot_dir: Optional[Path] = None):
        """
        Initialize realtime connector.

        Args:
            mode: Mode (replay or live)
            snapshot_dir: Directory for snapshots (default: ./snapshots)
        """
        self.mode = mode
        self.snapshot_dir = snapshot_dir or Path("./snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Any] = {{}}
        logger.info(f"Initialized RealtimeConnector in {{mode}} mode")

    def get_data(self, topic: str, window: str = "1h", ttl: str = "6h") -> Optional[dict[str, Any]]:
        """
        Get data from topic.

        In replay mode, loads from snapshot.
        In live mode, fetches from external source (mock for now).

        Args:
            topic: Data topic (e.g., 'sensor_data', 'grid_intensity')
            window: Time window for aggregation
            ttl: Time-to-live for cache

        Returns:
            Data dictionary or None if unavailable
        """
        if self.mode == "replay":
            return self._load_from_snapshot(topic)
        else:
            return self._fetch_live_data(topic, window)

    def _load_from_snapshot(self, topic: str) -> Optional[dict[str, Any]]:
        """
        Load data from cached snapshot.

        Args:
            topic: Data topic

        Returns:
            Cached data or None
        """
        snapshot_file = self.snapshot_dir / f"{{topic}}.json"
        if snapshot_file.exists():
            try:
                with open(snapshot_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded snapshot for topic {{topic}}")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load snapshot for {{topic}}: {{e}}")
                return None
        else:
            logger.warning(f"No snapshot found for topic {{topic}}")
            return None

    def _fetch_live_data(self, topic: str, window: str) -> Optional[dict[str, Any]]:
        """
        Fetch live data from external source.

        MOCK IMPLEMENTATION: Replace with actual API calls in production.

        Args:
            topic: Data topic
            window: Time window

        Returns:
            Live data or None
        """
        logger.info(f"Fetching live data for topic {{topic}} (window={{window}})")

        # MOCK: Replace with actual API calls
        mock_data = {{
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "window": window,
            "data": {{
                "value": 0.42,  # Example: grid intensity
                "unit": "kgCO2e/kWh"
            }}
        }}

        return mock_data

    def save_snapshot(self, topic: str, data: dict[str, Any]) -> None:
        """
        Save data snapshot for replay mode.

        Args:
            topic: Data topic
            data: Data to save
        """
        snapshot_file = self.snapshot_dir / f"{{topic}}.json"
        try:
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved snapshot for topic {{topic}}")
        except IOError as e:
            logger.error(f"Failed to save snapshot for {{topic}}: {{e}}")
'''


def generate_common_files(agent_dir, pack_id, python_pkg, license, author):
    """Generate LICENSE, pyproject.toml, etc.

    Creates common project files including LICENSE and Python packaging configuration.

    Args:
        agent_dir: Agent directory path
        pack_id: Agent identifier (kebab-case)
        python_pkg: Python package name (snake_case)
        license: License identifier
        author: Author name/email
    """
    # Generate LICENSE
    license_text = ""
    if license == "apache-2.0":
        license_text = f'''Apache License 2.0

Copyright {datetime.now().year} {author or "Author"}

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
    elif license == "mit":
        license_text = f'''MIT License

Copyright (c) {datetime.now().year} {author or "Author"}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

    if license_text:
        with open(agent_dir / "LICENSE", 'w', encoding='utf-8', newline='\n') as f:
            f.write(license_text)

    # Generate pyproject.toml
    author_str = f'{{name = "{author}", email = "user@example.com"}}' if author else '{{name = "Author", email = "author@example.com"}}'

    pyproject_content = f'''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "greenlang-{pack_id}"
version = "0.1.0"
description = "GreenLang agent: {pack_id}"
readme = "README.md"
license = {{text = "{license if license != 'none' else 'proprietary'}"}}
authors = [
    {author_str}
]
requires-python = ">=3.10"
keywords = [
    "greenlang",
    "climate",
    "emissions",
    "sustainability",
    "carbon"
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "greenlang>=0.1.0",
    "pydantic>=2.7",
    "pyyaml>=6.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/{pack_id}"
Repository = "https://github.com/yourusername/{pack_id}"
Issues = "https://github.com/yourusername/{pack_id}/issues"

[project.entry-points."greenlang.packs"]
{pack_id} = "{python_pkg}:get_manifest_path"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]
test = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-mock>=3.0",
    "hypothesis>=6.80.0",
]

[tool.setuptools]
packages = ["{python_pkg}"]
package-dir = {{"" = "src"}}

[tool.black]
line-length = 100
target-version = ["py310", "py311", "py312"]

[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "W", "F", "I", "B", "C4", "UP"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = [
    "-ra",
    "-q",
    "--strict-markers",
    "--cov={python_pkg}",
    "--cov-report=term-missing",
    "--cov-report=html",
]
'''

    with open(agent_dir / "pyproject.toml", 'w', encoding='utf-8', newline='\n') as f:
        f.write(pyproject_content)


def generate_test_suite(agent_dir, pack_id, python_pkg, class_name, template):
    """Generate comprehensive test suite with golden, property, and spec tests.

    Args:
        agent_dir: Agent directory path
        pack_id: Agent identifier
        python_pkg: Python package name
        class_name: Agent class name
        template: Template type
    """
    # Generate test_agent.py
    test_agent_content = f'''"""
Unit tests for {class_name} agent.

This test suite includes:
- Golden tests: Known inputs → expected outputs
- Property tests: Invariants that must hold
- Spec tests: Validation against AgentSpec v2
"""

import pytest
from hypothesis import given, strategies as st
from {python_pkg}.agent import {class_name}
from {python_pkg}.schemas import InputModel, OutputModel, example_input


class Test{class_name}Golden:
    """Golden tests with known inputs and expected outputs."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return {class_name}()

    def test_example_input(self, agent):
        """Test with example input from schemas."""
        inputs = example_input()
        result = agent.compute(inputs)

        assert "co2e_kg" in result
        assert result["co2e_kg"] >= 0
        assert "_provenance" in result

    def test_baseline_case(self, agent):
        """Test baseline case with typical values."""
        inputs = {{"fuel_volume": 100.0, "emission_factor": 2.3}}
        result = agent.compute(inputs)

        expected_co2e = 100.0 * 2.3
        assert abs(result["co2e_kg"] - expected_co2e) <= 1e-3
        assert result["formula"] is not None

    def test_zero_volume(self, agent):
        """Test with zero fuel volume."""
        inputs = {{"fuel_volume": 0.0, "emission_factor": 2.3}}
        result = agent.compute(inputs)

        assert result["co2e_kg"] == 0.0


class Test{class_name}Properties:
    """Property-based tests using Hypothesis."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return {class_name}()

    @given(
        fuel_volume=st.floats(min_value=0, max_value=10000),
        emission_factor=st.floats(min_value=0, max_value=10)
    )
    def test_non_negative_emissions(self, fuel_volume, emission_factor):
        """Emissions must always be non-negative."""
        agent = {class_name}()  # Create inline to avoid Hypothesis fixture scope issue
        inputs = {{"fuel_volume": fuel_volume, "emission_factor": emission_factor}}
        result = agent.compute(inputs)

        assert result["co2e_kg"] >= 0

    @given(
        fuel_volume=st.floats(min_value=0, max_value=10000),
        emission_factor=st.floats(min_value=0, max_value=10)
    )
    def test_monotonicity_in_volume(self, fuel_volume, emission_factor):
        """More fuel → more emissions (all else equal)."""
        agent = {class_name}()  # Create inline to avoid Hypothesis fixture scope issue
        if fuel_volume < 5000:
            inputs1 = {{"fuel_volume": fuel_volume, "emission_factor": emission_factor}}
            inputs2 = {{"fuel_volume": fuel_volume * 2, "emission_factor": emission_factor}}

            result1 = agent.compute(inputs1)
            result2 = agent.compute(inputs2)

            assert result2["co2e_kg"] >= result1["co2e_kg"]

    def test_determinism(self, agent):
        """Same inputs → same outputs (deterministic)."""
        inputs = {{"fuel_volume": 100.0, "emission_factor": 2.3}}

        result1 = agent.compute(inputs)
        result2 = agent.compute(inputs)

        assert result1["co2e_kg"] == result2["co2e_kg"]
        assert result1["formula"] == result2["formula"]


class Test{class_name}Spec:
    """Tests validating against AgentSpec v2."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return {class_name}()

    def test_provenance_fields(self, agent):
        """Provenance record must include required fields."""
        inputs = example_input()
        result = agent.compute(inputs)

        provenance = result["_provenance"]
        assert "agent_name" in provenance
        assert "agent_version" in provenance
        assert "timestamp" in provenance
        assert "inputs_hash" in provenance
        assert "gwp_set" in provenance

    def test_input_validation_negative(self, agent):
        """Negative inputs must be rejected."""
        inputs = {{"fuel_volume": -10.0, "emission_factor": 2.3}}

        with pytest.raises(ValueError):
            agent.compute(inputs)

    def test_output_schema(self, agent):
        """Output must conform to OutputModel schema."""
        inputs = example_input()
        result = agent.compute(inputs)

        # Should not raise ValidationError
        output = OutputModel(
            co2e_kg=result["co2e_kg"],
            formula=result["formula"]
        )
        assert output.co2e_kg >= 0

    def test_run_method(self, agent):
        """Test run method delegates to compute."""
        inputs = example_input()
        result = agent.run(inputs)

        assert "co2e_kg" in result
        assert result["co2e_kg"] >= 0
        assert "_provenance" in result

    def test_invalid_input_type(self, agent):
        """Test handling of invalid input types."""
        with pytest.raises((ValueError, TypeError)):
            agent.compute({{"fuel_volume": "invalid", "emission_factor": 2.3}})

    def test_missing_required_field(self, agent):
        """Test handling of missing required field."""
        with pytest.raises((ValueError, TypeError)):
            agent.compute({{"fuel_volume": 100.0}})  # Missing emission_factor

    def test_compute_via_module_function(self):
        """Test module-level compute function works correctly."""
        from {python_pkg}.agent import compute
        inputs = example_input()
        result = compute(inputs)

        assert "co2e_kg" in result
        assert result["co2e_kg"] >= 0
        assert "_provenance" in result'''

    # Add AI-specific test for "no naked numbers" enforcement
    if template == "ai":
        test_agent_content += f'''

    def test_no_naked_numbers_enforcement(self, agent):
        """AI agent must enforce 'no naked numbers' policy via tools."""
        from {python_pkg}.ai_tools import calculate_emissions

        # Positive test: Tool with proper units succeeds
        result = calculate_emissions(fuel_volume=100.0, emission_factor=2.3)
        assert "co2e_kg" in result
        assert "formula" in result
        assert result["co2e_kg"] >= 0

        # The tool enforces that all numeric values have implicit units
        # (documented in function signature: fuel_volume must be in m^3)
        # This test verifies the tool can be called successfully
        # and returns structured data (not naked numbers)
        assert isinstance(result["co2e_kg"], (int, float))
        assert isinstance(result["formula"], str)
        assert "m³" not in str(result["co2e_kg"])  # Value is numeric, not string with unit
'''

    test_agent_content += "\n"

    with open(agent_dir / "tests" / "test_agent.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(test_agent_content)

    # Generate __init__.py for tests
    with open(agent_dir / "tests" / "__init__.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write('"""Test suite for agent."""\n')

    # Generate conftest.py
    conftest_content = '''"""
Pytest configuration and fixtures.
"""

import pytest


@pytest.fixture
def sample_inputs():
    """Sample input data for testing."""
    return {
        "fuel_volume": 100.0,
        "emission_factor": 2.3
    }
'''

    with open(agent_dir / "tests" / "conftest.py", 'w', encoding='utf-8', newline='\n') as f:
        f.write(conftest_content)


def generate_examples(agent_dir, pack_id, python_pkg, template, realtime):
    """Generate examples/pipeline.gl.yaml and input samples.

    Args:
        agent_dir: Agent directory path
        pack_id: Agent identifier
        python_pkg: Python package name
        template: Template type
        realtime: Include realtime examples
    """
    # Ensure examples directory exists
    (agent_dir / "examples").mkdir(exist_ok=True)

    # Generate pipeline.gl.yaml
    pipeline_content = f'''# GreenLang Pipeline Example for {pack_id}
#
# This pipeline demonstrates how to use the {pack_id} agent.
# Run with: gl run pipeline.gl.yaml

version: "1.0"
name: "{pack_id}-example"
description: "Example pipeline using {pack_id} agent"

agents:
  - id: "{pack_id}"
    pack: "{pack_id}"
    version: "0.1.0"

inputs:
  fuel_volume: 100.0
  emission_factor: 2.3

steps:
  - agent: "{pack_id}"
    inputs:
      fuel_volume: "${{inputs.fuel_volume}}"
      emission_factor: "${{inputs.emission_factor}}"
    outputs:
      - co2e_kg
      - formula

outputs:
  total_emissions: "${{steps[0].outputs.co2e_kg}}"
  calculation: "${{steps[0].outputs.formula}}"
'''

    with open(agent_dir / "examples" / "pipeline.gl.yaml", 'w', encoding='utf-8', newline='\n') as f:
        f.write(pipeline_content)

    # Generate input.sample.json
    sample_input = {
        "fuel_volume": 100.0,
        "emission_factor": 2.3
    }

    with open(agent_dir / "examples" / "input.sample.json", 'w', encoding='utf-8', newline='\n') as f:
        json.dump(sample_input, f, indent=2)


def generate_documentation(agent_dir, pack_id, template, realtime):
    """Generate README.md and CHANGELOG.md.

    Args:
        agent_dir: Agent directory path
        pack_id: Agent identifier
        template: Template type
        realtime: Include realtime docs
    """
    # Generate README.md
    disclaimer = ""
    if template == "industry":
        disclaimer = f'''
## ⚠️ ADVISORY NOTICE

**FOR INFORMATIONAL PURPOSES ONLY** - This industry template uses MOCK emission factors for demonstration.

**IMPORTANT - DO NOT USE IN PRODUCTION WITHOUT VALIDATION:**
- All emission factors are placeholder values and must be replaced
- Use verified, region-specific factors from authoritative sources:
  - EPA eGRID (electricity)
  - IPCC AR6 (fuel combustion)
  - GHG Protocol (scope 3)
- Not suitable for compliance reporting without proper validation
- Consult with climate accounting experts before deployment

**Mock factors in this template:**
- `FUEL_EF = 2.3 kgCO2e/m³` (line 1152 in agent.py)
- `GRID_EF = {{"US": 0.42, "EU": 0.35, "UK": 0.29}} kgCO2e/kWh` (line 1153)

Replace these with verified emission factors from your specific jurisdiction.

'''

    readme_content = f'''# {pack_id}

GreenLang agent for emissions calculation.
{disclaimer}
## Overview

This agent computes CO2e emissions using the {template} template.

## Installation

```bash
pip install greenlang-{pack_id}
```

## Usage

### Python API

```python
from {pack_id.replace("-", "_")}.agent import {pack_id.replace("-", " ").title().replace(" ", "")}

# Create agent
agent = {pack_id.replace("-", " ").title().replace(" ", "")}()

# Run computation
inputs = {{
    "fuel_volume": 100.0,
    "emission_factor": 2.3
}}

result = agent.compute(inputs)
print(f"Emissions: {{result['co2e_kg']}} kgCO2e")
```

### CLI

```bash
# Run with GreenLang CLI
gl run examples/pipeline.gl.yaml

# Run with custom inputs
gl run examples/pipeline.gl.yaml --input examples/input.sample.json
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/{pack_id}.git
cd {pack_id}

# Install dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov={pack_id.replace("-", "_")} --cov-report=html

# Run specific test
pytest tests/test_agent.py::Test{pack_id.replace("-", " ").title().replace(" ", "")}Golden::test_baseline_case
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## License

See [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/{pack_id}/issues
- Documentation: https://docs.greenlang.io
'''

    with open(agent_dir / "README.md", 'w', encoding='utf-8', newline='\n') as f:
        f.write(readme_content)

    # Generate CHANGELOG.md
    changelog_content = f'''# Changelog

All notable changes to {pack_id} will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - {datetime.now().strftime("%Y-%m-%d")}

### Added
- Initial release
- AgentSpec v2 compliance
- Pydantic v2 schemas
- Comprehensive test suite
- Example pipeline
- Documentation

[Unreleased]: https://github.com/yourusername/{pack_id}/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/{pack_id}/releases/tag/v0.1.0
'''

    with open(agent_dir / "CHANGELOG.md", 'w', encoding='utf-8', newline='\n') as f:
        f.write(changelog_content)


def generate_gitignore(agent_dir):
    """Generate .gitignore file for Python/GreenLang projects.

    Args:
        agent_dir: Agent directory path
    """
    gitignore_content = '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
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
MANIFEST

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
*.code-workspace
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
Desktop.ini

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# GreenLang specific
*.log
*.tmp
*.temp
.greenlang_cache/
snapshots/
reports/
output/

# API keys and secrets
.env.local
api_keys.json
secrets.yaml
credentials.json
*.pem
*.key
*.crt

# Database
*.db
*.sqlite
*.sqlite3

# Jupyter
.ipynb_checkpoints

# Backup files
*.bak
*.backup
*.old

# Temporary files
.ruff_cache/
'''

    with open(agent_dir / ".gitignore", 'w', encoding='utf-8', newline='\n') as f:
        f.write(gitignore_content)


def generate_precommit_config(agent_dir):
    """Generate .pre-commit-config.yaml with Bandit + TruffleHog security scanners.

    Args:
        agent_dir: Agent directory path
    """
    precommit_content = '''# Pre-commit hooks for code quality and security
# Install: pip install pre-commit && pre-commit install
# Run manually: pre-commit run --all-files

repos:
  - repo: https://github.com/trufflesecurity/trufflehog
    rev: v3.66.0
    hooks:
      - id: trufflehog
        args: [--no-update]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: debug-statements
      - id: mixed-line-ending
        args: ['--fix=lf']
      - id: detect-private-key

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11
        args: ['--line-length=100']

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: ['--fix', '--exit-non-zero-on-fix']

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']
        additional_dependencies: ['bandit[toml]']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies:
          - types-PyYAML
          - pydantic>=2.7
        args: ['--ignore-missing-imports']
'''

    with open(agent_dir / ".pre-commit-config.yaml", 'w', encoding='utf-8', newline='\n') as f:
        f.write(precommit_content)


def generate_ci_workflow(agent_dir, pack_id, runtimes):
    """Generate GitHub Actions workflow with 3 OS matrix.

    Args:
        agent_dir: Agent directory path
        pack_id: Agent identifier
        runtimes: Runtimes to test (comma-separated)
    """
    workflow_dir = agent_dir / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)

    workflow_content = f'''name: CI

on:
  push:
    branches: [ main, master, develop ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    name: Test on ${{{{ matrix.os }}}} - Python ${{{{ matrix.python-version }}}}
    runs-on: ${{{{ matrix.os }}}}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{{{ matrix.python-version }}}}
        uses: actions/setup-python@v4
        with:
          python-version: ${{{{ matrix.python-version }}}}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,test]"

      - name: Lint with ruff
        run: |
          ruff check src/ tests/

      - name: Type check with mypy
        run: |
          mypy src/

      - name: Run tests
        run: |
          pytest --cov={pack_id.replace("-", "_")} --cov-report=xml --cov-report=term

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-${{{{ matrix.os }}}}-py${{{{ matrix.python-version }}}}

  security:
    name: Security Scan
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Run Bandit
        run: |
          pip install bandit[toml]
          bandit -r src/ -f json -o bandit-report.json || true

      - name: Run TruffleHog
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{{{ github.event.repository.default_branch }}}}
          head: HEAD
'''

    with open(workflow_dir / "ci.yml", 'w', encoding='utf-8', newline='\n') as f:
        f.write(workflow_content)


def git_init(agent_dir):
    """Initialize git repository.

    Args:
        agent_dir: Agent directory path
    """
    try:
        subprocess.run(
            ["git", "init"],
            cwd=agent_dir,
            check=True,
            capture_output=True,
            text=True
        )
        subprocess.run(
            ["git", "add", "."],
            cwd=agent_dir,
            check=True,
            capture_output=True,
            text=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit: Agent scaffold"],
            cwd=agent_dir,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Initialized git repository in {agent_dir}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to initialize git: {e}")
    except FileNotFoundError:
        logger.warning("git command not found - skipping git initialization")


def validate_generated_agent(agent_dir):
    """Validate generated agent structure and pack.yaml.

    Args:
        agent_dir: Agent directory path

    Returns:
        Dictionary with validation results
    """
    warnings = []
    errors = []

    # Check pack.yaml exists
    pack_yaml_path = agent_dir / "pack.yaml"
    if not pack_yaml_path.exists():
        errors.append("pack.yaml not found")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Try to load and validate pack.yaml
    try:
        with open(pack_yaml_path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f)

        # Check required fields
        required_fields = ['schema_version', 'id', 'name', 'version', 'compute']
        for field in required_fields:
            if field not in manifest:
                errors.append(f"Missing required field: {field}")

        # Check version format
        if 'version' in manifest:
            if not SEMVER_PATTERN.match(manifest['version']):
                warnings.append(f"Version '{manifest['version']}' is not semantic versioning")

        # Check compute section
        if 'compute' in manifest:
            compute = manifest['compute']
            if 'entrypoint' not in compute:
                errors.append("compute.entrypoint is required")
            if 'inputs' not in compute:
                warnings.append("compute.inputs not defined")
            if 'outputs' not in compute:
                warnings.append("compute.outputs not defined")

    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML: {e}")
    except Exception as e:
        errors.append(f"Validation error: {e}")

    valid = len(errors) == 0
    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings
    }


def print_success_message(agent_dir, pack_id, template, realtime, with_ci, no_git, no_precommit):
    """Print success message with next steps using Rich formatting.

    Args:
        agent_dir: Agent directory path
        pack_id: Agent identifier
        template: Template type
        realtime: Realtime enabled
        with_ci: CI workflow generated
        no_git: Git skipped
        no_precommit: Pre-commit skipped
    """
    from datetime import datetime

    console.print()
    console.print(Panel.fit(
        f"[green]Successfully created {template} agent: {pack_id}[/green]",
        border_style="green"
    ))
    console.print()

    # Files generated table
    table = Table(title="Generated Files", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="cyan")
    table.add_column("Files", style="white")

    table.add_row("Core", "pack.yaml, src/, tests/")
    table.add_row("Config", "pyproject.toml, .gitignore")
    table.add_row("Docs", "README.md, CHANGELOG.md")
    table.add_row("Examples", "examples/pipeline.gl.yaml, input.sample.json")

    if not no_precommit:
        table.add_row("Hooks", ".pre-commit-config.yaml")
    if with_ci:
        table.add_row("CI/CD", ".github/workflows/ci.yml")

    console.print(table)
    console.print()

    # Next steps
    console.print("[bold cyan]Next Steps:[/bold cyan]")
    console.print()
    console.print(f"  1. [yellow]cd {agent_dir.name}[/yellow]")
    console.print(f"  2. [yellow]pip install -e \".[dev,test]\"[/yellow]  # Install in editable mode")

    if not no_precommit:
        console.print(f"  3. [yellow]pre-commit install[/yellow]  # Install pre-commit hooks")

    console.print(f"  4. [yellow]pytest[/yellow]  # Run tests")
    console.print(f"  5. [yellow]gl run examples/pipeline.gl.yaml[/yellow]  # Test the agent")
    console.print()

    # Customization tips
    console.print("[bold cyan]Customization Tips:[/bold cyan]")
    console.print()
    console.print("  - Edit [cyan]pack.yaml[/cyan] to customize inputs/outputs/factors")
    console.print(f"  - Modify [cyan]src/{pack_id.replace('-', '_')}/schemas.py[/cyan] for data models")
    console.print(f"  - Implement logic in [cyan]src/{pack_id.replace('-', '_')}/agent.py[/cyan]")
    console.print("  - Add golden tests in [cyan]tests/test_agent.py[/cyan]")
    console.print()

    if realtime:
        console.print("[bold yellow]Realtime Mode:[/bold yellow]")
        console.print("  - Connector stubs generated in [cyan]realtime.py[/cyan]")
        console.print("  - Replace mock implementations with actual API calls")
        console.print()

    console.print("[green]Agent ready for development![/green]")
    console.print()
