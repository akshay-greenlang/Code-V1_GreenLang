"""
GreenLang Framework - Base Agent Template

Standard template structure for all GreenLang agents.
Provides common patterns for agent organization and implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
import json


class AgentCategory(Enum):
    """Categories of GreenLang agents."""
    THERMAL = "thermal"
    COMBUSTION = "combustion"
    STEAM = "steam"
    EMISSIONS = "emissions"
    OPTIMIZATION = "optimization"
    COMPLIANCE = "compliance"
    MONITORING = "monitoring"


class AgentType(Enum):
    """Types of agent functionality."""
    CALCULATOR = "calculator"
    OPTIMIZER = "optimizer"
    ANALYZER = "analyzer"
    PREDICTOR = "predictor"
    ORCHESTRATOR = "orchestrator"
    HYBRID = "hybrid"


@dataclass
class AgentConfig:
    """
    Configuration for a GreenLang agent.

    Defines all metadata and structure for agent generation.
    """
    agent_id: str  # e.g., "GL-006"
    name: str  # e.g., "HEATRECLAIM"
    full_name: str  # e.g., "Industrial Heat Recovery Optimization"
    description: str
    version: str = "1.0.0"
    category: AgentCategory = AgentCategory.THERMAL
    agent_type: AgentType = AgentType.CALCULATOR

    # Domain specifications
    domain: str = ""
    standards: List[str] = field(default_factory=list)
    regulations: List[str] = field(default_factory=list)

    # Technical specifications
    primary_calculations: List[str] = field(default_factory=list)
    input_data_types: List[str] = field(default_factory=list)
    output_data_types: List[str] = field(default_factory=list)

    # Dependencies
    core_dependencies: List[str] = field(default_factory=lambda: [
        "pydantic>=2.0",
        "numpy>=1.24",
    ])
    optional_dependencies: Dict[str, List[str]] = field(default_factory=dict)

    # Features to include
    include_api: bool = True
    include_cli: bool = True
    include_streaming: bool = False
    include_optimization: bool = False
    include_explainability: bool = True
    include_monitoring: bool = True

    # Quality targets
    test_coverage_target: float = 85.0
    documentation_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "full_name": self.full_name,
            "description": self.description,
            "version": self.version,
            "category": self.category.value,
            "agent_type": self.agent_type.value,
            "domain": self.domain,
            "standards": self.standards,
            "regulations": self.regulations,
            "primary_calculations": self.primary_calculations,
            "input_data_types": self.input_data_types,
            "output_data_types": self.output_data_types,
            "core_dependencies": self.core_dependencies,
            "optional_dependencies": self.optional_dependencies,
            "include_api": self.include_api,
            "include_cli": self.include_cli,
            "include_streaming": self.include_streaming,
            "include_optimization": self.include_optimization,
            "include_explainability": self.include_explainability,
            "include_monitoring": self.include_monitoring,
            "test_coverage_target": self.test_coverage_target,
            "documentation_required": self.documentation_required,
        }


class BaseAgentTemplate:
    """
    Base template for generating GreenLang agents.

    Provides standard directory structure and file templates.
    """

    # Standard directory structure
    DIRECTORY_STRUCTURE = {
        "core": "Core agent logic and business rules",
        "calculators": "Deterministic calculation modules",
        "models": "Pydantic data models",
        "api": "REST/GraphQL API endpoints",
        "cli": "Command-line interface",
        "explainability": "Audit trail and explanation generation",
        "monitoring": "Metrics and health checks",
        "tests": "Test suites",
        "tests/unit": "Unit tests",
        "tests/integration": "Integration tests",
        "tests/golden": "Golden master tests",
        "deployment": "Kubernetes and Docker configs",
        "docs": "Documentation",
    }

    # Files to generate in each directory
    STANDARD_FILES = {
        "": [  # Root
            "__init__.py",
            "agent.py",
            "config.py",
            "pyproject.toml",
            "README.md",
        ],
        "core": [
            "__init__.py",
            "engine.py",
            "orchestrator.py",
        ],
        "calculators": [
            "__init__.py",
        ],
        "models": [
            "__init__.py",
            "inputs.py",
            "outputs.py",
            "domain.py",
        ],
        "api": [
            "__init__.py",
            "routes.py",
            "schemas.py",
        ],
        "explainability": [
            "__init__.py",
            "explainer.py",
            "audit.py",
        ],
        "monitoring": [
            "__init__.py",
            "metrics.py",
            "health.py",
        ],
        "tests": [
            "__init__.py",
            "conftest.py",
        ],
        "tests/unit": [
            "__init__.py",
        ],
        "tests/integration": [
            "__init__.py",
        ],
        "deployment": [
            "Dockerfile",
            "docker-compose.yml",
        ],
    }

    def __init__(self, config: AgentConfig):
        """Initialize template with agent configuration."""
        self.config = config

    def get_directory_structure(self) -> Dict[str, str]:
        """Get directory structure based on config."""
        structure = dict(self.DIRECTORY_STRUCTURE)

        if self.config.include_optimization:
            structure["optimization"] = "Optimization algorithms"

        if self.config.include_streaming:
            structure["streaming"] = "Real-time data streaming"

        return structure

    def generate_init_py(self) -> str:
        """Generate root __init__.py content."""
        return f'''"""
GreenLang Agent: {self.config.agent_id}_{self.config.name}

{self.config.full_name}
{self.config.description}

Version: {self.config.version}
Category: {self.config.category.value}
Type: {self.config.agent_type.value}
"""

__version__ = "{self.config.version}"
__agent_id__ = "{self.config.agent_id}"
__agent_name__ = "{self.config.name}"

from .core import {self.config.name}Engine
from .models import *

__all__ = [
    "{self.config.name}Engine",
    "__version__",
    "__agent_id__",
    "__agent_name__",
]
'''

    def generate_agent_py(self) -> str:
        """Generate main agent.py content."""
        return f'''"""
{self.config.agent_id}_{self.config.name} - Main Agent Entry Point

{self.config.description}
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass

from .core import {self.config.name}Engine
from .models import AgentInput, AgentOutput


@dataclass
class {self.config.name}Agent:
    """
    {self.config.full_name}

    Agent ID: {self.config.agent_id}
    Version: {self.config.version}

    Standards:
    {chr(10).join(f"    - {s}" for s in self.config.standards) if self.config.standards else "    - None specified"}

    Primary Calculations:
    {chr(10).join(f"    - {c}" for c in self.config.primary_calculations) if self.config.primary_calculations else "    - None specified"}
    """

    AGENT_ID = "{self.config.agent_id}"
    VERSION = "{self.config.version}"
    NAME = "{self.config.name}"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        track_provenance: bool = True,
    ):
        """
        Initialize {self.config.name} agent.

        Args:
            config: Optional configuration overrides
            track_provenance: Enable SHA-256 provenance tracking
        """
        self.config = config or {{}}
        self.engine = {self.config.name}Engine(
            agent_id=self.AGENT_ID,
            track_provenance=track_provenance,
        )

    def process(self, inputs: AgentInput) -> AgentOutput:
        """
        Process inputs through the agent.

        Args:
            inputs: Validated agent inputs

        Returns:
            AgentOutput with results and provenance
        """
        return self.engine.execute(inputs)

    def explain(self, result: AgentOutput) -> Dict[str, Any]:
        """
        Generate explanation for a result.

        Args:
            result: Agent output to explain

        Returns:
            Explanation dictionary with methodology and breakdown
        """
        return self.engine.generate_explanation(result)

    def validate_inputs(self, inputs: Dict[str, Any]) -> AgentInput:
        """
        Validate and parse raw inputs.

        Args:
            inputs: Raw input dictionary

        Returns:
            Validated AgentInput model
        """
        return AgentInput.model_validate(inputs)
'''

    def generate_config_py(self) -> str:
        """Generate config.py content."""
        return f'''"""
{self.config.agent_id}_{self.config.name} - Configuration

Agent configuration and environment settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class AgentSettings(BaseSettings):
    """Agent configuration from environment."""

    # Agent identification
    agent_id: str = "{self.config.agent_id}"
    agent_name: str = "{self.config.name}"
    agent_version: str = "{self.config.version}"

    # Logging
    log_level: str = "INFO"

    # Provenance tracking
    track_provenance: bool = True
    provenance_store_path: Optional[str] = None

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Performance
    max_concurrent_calculations: int = 10
    calculation_timeout_seconds: int = 300

    # Feature flags
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_tracing: bool = False

    class Config:
        env_prefix = "{self.config.agent_id.replace('-', '_')}_"
        env_file = ".env"


@lru_cache
def get_settings() -> AgentSettings:
    """Get cached settings instance."""
    return AgentSettings()
'''

    def generate_core_engine(self) -> str:
        """Generate core engine template."""
        return f'''"""
{self.config.agent_id}_{self.config.name} - Core Engine

Main calculation engine with provenance tracking.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import hashlib
import json

from ..models import AgentInput, AgentOutput


class {self.config.name}Engine:
    """
    Core calculation engine for {self.config.full_name}.

    Provides:
    - Deterministic calculations
    - SHA-256 provenance tracking
    - Explainability generation
    - Audit trail
    """

    VERSION = "{self.config.version}"

    def __init__(
        self,
        agent_id: str = "{self.config.agent_id}",
        track_provenance: bool = True,
    ):
        """Initialize engine."""
        self.agent_id = agent_id
        self.track_provenance = track_provenance
        self._calculation_history: List[Dict[str, Any]] = []

    def execute(self, inputs: AgentInput) -> AgentOutput:
        """
        Execute calculation with provenance tracking.

        Args:
            inputs: Validated input data

        Returns:
            AgentOutput with results and metadata
        """
        start_time = datetime.now(timezone.utc)

        # Compute input hash
        inputs_hash = self._compute_hash(inputs.model_dump())

        # Perform calculation
        result = self._calculate(inputs)

        # Compute output hash
        outputs_hash = self._compute_hash(result)

        end_time = datetime.now(timezone.utc)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Build output
        output = AgentOutput(
            **result,
            computation_hash=self._compute_combined_hash(inputs_hash, outputs_hash),
            inputs_hash=inputs_hash,
            timestamp=start_time,
            execution_time_ms=execution_time_ms,
            agent_id=self.agent_id,
            agent_version=self.VERSION,
        )

        # Track provenance
        if self.track_provenance:
            self._record_calculation(inputs, output)

        return output

    def _calculate(self, inputs: AgentInput) -> Dict[str, Any]:
        """
        Perform the actual calculation.

        Override this method in subclasses.

        Args:
            inputs: Validated inputs

        Returns:
            Result dictionary
        """
        raise NotImplementedError("Subclasses must implement _calculate")

    def generate_explanation(self, output: AgentOutput) -> Dict[str, Any]:
        """
        Generate human-readable explanation.

        Args:
            output: Calculation output

        Returns:
            Explanation with methodology and breakdown
        """
        return {{
            "summary": "Calculation completed successfully",
            "methodology": [],
            "key_parameters": {{}},
            "assumptions": [],
            "limitations": [],
            "provenance": {{
                "computation_hash": output.computation_hash,
                "inputs_hash": output.inputs_hash,
                "timestamp": output.timestamp.isoformat(),
            }},
        }}

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _compute_combined_hash(self, inputs_hash: str, outputs_hash: str) -> str:
        """Compute combined hash for provenance."""
        combined = {{
            "inputs_hash": inputs_hash,
            "outputs_hash": outputs_hash,
            "agent_id": self.agent_id,
            "version": self.VERSION,
        }}
        return self._compute_hash(combined)

    def _record_calculation(
        self,
        inputs: AgentInput,
        output: AgentOutput,
    ) -> None:
        """Record calculation in history."""
        self._calculation_history.append({{
            "timestamp": output.timestamp.isoformat(),
            "inputs_hash": output.inputs_hash,
            "computation_hash": output.computation_hash,
            "execution_time_ms": output.execution_time_ms,
        }})

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get calculation audit trail."""
        return self._calculation_history.copy()
'''

    def generate_models_inputs(self) -> str:
        """Generate inputs model template."""
        return f'''"""
{self.config.agent_id}_{self.config.name} - Input Models

Pydantic v2 models for agent inputs with validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class AgentInput(BaseModel):
    """
    Input model for {self.config.name} agent.

    Attributes:
        name: Descriptive name for this calculation
        # Add domain-specific fields
    """

    name: str = Field(
        ...,
        description="Descriptive name for this calculation",
        min_length=1,
        max_length=200,
    )

    # TODO: Add domain-specific input fields
    # Example:
    # temperature: float = Field(..., ge=-273.15, description="Temperature in Celsius")
    # pressure: float = Field(..., gt=0, description="Pressure in Pa")

    model_config = {{
        "json_schema_extra": {{
            "examples": [
                {{
                    "name": "Example calculation",
                }}
            ]
        }}
    }}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name field."""
        return v.strip()
'''

    def generate_models_outputs(self) -> str:
        """Generate outputs model template."""
        return f'''"""
{self.config.agent_id}_{self.config.name} - Output Models

Pydantic v2 models for agent outputs with provenance.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class AgentOutput(BaseModel):
    """
    Output model for {self.config.name} agent.

    Includes calculation results and provenance metadata.
    """

    # Results - TODO: Add domain-specific output fields
    # Example:
    # efficiency: float = Field(..., description="Calculated efficiency")
    # energy_recovered: float = Field(..., description="Energy recovered in kW")

    # Provenance metadata
    computation_hash: str = Field(
        ...,
        description="SHA-256 hash of computation for audit"
    )
    inputs_hash: str = Field(
        ...,
        description="SHA-256 hash of inputs"
    )
    timestamp: datetime = Field(
        ...,
        description="Calculation timestamp (UTC)"
    )
    execution_time_ms: float = Field(
        ...,
        description="Execution time in milliseconds"
    )
    agent_id: str = Field(
        default="{self.config.agent_id}",
        description="Agent identifier"
    )
    agent_version: str = Field(
        default="{self.config.version}",
        description="Agent version"
    )

    # Optional metadata
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings generated during calculation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
'''

    def generate_pyproject_toml(self) -> str:
        """Generate pyproject.toml content."""
        deps = '",\n    "'.join(self.config.core_dependencies)
        return f'''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{self.config.agent_id.lower()}-{self.config.name.lower()}"
version = "{self.config.version}"
description = "{self.config.full_name}"
readme = "README.md"
requires-python = ">=3.10"
license = {{text = "Proprietary"}}
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "{deps}"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.21",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
]
api = [
    "fastapi>=0.100",
    "uvicorn>=0.23",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["{self.config.name.lower()}"]
branch = true

[tool.coverage.report]
fail_under = {self.config.test_coverage_target}

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.10"
strict = true
'''

    def generate_dockerfile(self) -> str:
        """Generate Dockerfile content."""
        return f'''# {self.config.agent_id}_{self.config.name} Docker Image
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir build && \\
    pip wheel --no-cache-dir --wheel-dir /wheels -e ".[api]"

# Production image
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash agent
USER agent

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --user /wheels/*

# Copy application
COPY --chown=agent:agent . .

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Environment
ENV PYTHONUNBUFFERED=1
ENV {self.config.agent_id.replace("-", "_")}_LOG_LEVEL=INFO

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

LABEL org.opencontainers.image.title="{self.config.agent_id}_{self.config.name}"
LABEL org.opencontainers.image.description="{self.config.full_name}"
LABEL org.opencontainers.image.version="{self.config.version}"
'''

    def generate_readme(self) -> str:
        """Generate README.md content."""
        return f'''# {self.config.agent_id}_{self.config.name}

{self.config.full_name}

## Overview

{self.config.description}

**Agent ID:** {self.config.agent_id}
**Version:** {self.config.version}
**Category:** {self.config.category.value}
**Type:** {self.config.agent_type.value}

## Standards Compliance

{chr(10).join(f"- {s}" for s in self.config.standards) if self.config.standards else "- None specified"}

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from {self.config.name.lower()} import {self.config.name}Agent

agent = {self.config.name}Agent()
result = agent.process(inputs)
```

## API

Start the API server:

```bash
uvicorn api:app --reload
```

API documentation available at `http://localhost:8000/docs`

## Testing

```bash
pytest tests/ -v --cov
```

## Provenance

All calculations include SHA-256 provenance tracking:
- `computation_hash`: Combined hash of inputs + outputs + parameters
- `inputs_hash`: Hash of input data
- `timestamp`: UTC timestamp of calculation

## License

Proprietary - GreenLang Platform
'''

    def get_all_templates(self) -> Dict[str, str]:
        """Get all template contents."""
        return {
            "__init__.py": self.generate_init_py(),
            "agent.py": self.generate_agent_py(),
            "config.py": self.generate_config_py(),
            "core/engine.py": self.generate_core_engine(),
            "models/inputs.py": self.generate_models_inputs(),
            "models/outputs.py": self.generate_models_outputs(),
            "pyproject.toml": self.generate_pyproject_toml(),
            "deployment/Dockerfile": self.generate_dockerfile(),
            "README.md": self.generate_readme(),
        }
