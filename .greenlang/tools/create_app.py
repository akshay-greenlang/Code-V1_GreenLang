#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

logger = logging.getLogger(__name__)
GreenLang Application Generator

Create complete GreenLang applications with a single command.
Generates full project structure, infrastructure setup, tests, and CI/CD.
"""

import logging
import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class AppTemplate:
    """Application template configuration."""
    name: str
    description: str
    features: List[str]
    agents: List[str]


class AppTemplates:
    """Predefined application templates."""

    TEMPLATES = {
        'data-intake': AppTemplate(
            name='Data Intake Application',
            description='Validates and processes incoming data files (CSV, Excel, JSON)',
            features=['validation', 'database', 'monitoring'],
            agents=['DataValidatorAgent', 'DataTransformerAgent', 'DataLoaderAgent']
        ),
        'calculation': AppTemplate(
            name='Calculation Engine',
            description='Performs complex sustainability calculations',
            features=['validation', 'cache', 'monitoring', 'batch'],
            agents=['CalculatorAgent', 'ValidatorAgent', 'AggregatorAgent']
        ),
        'llm-analysis': AppTemplate(
            name='LLM Analysis Application',
            description='Uses AI to analyze sustainability reports and data',
            features=['llm', 'cache', 'validation', 'monitoring'],
            agents=['LLMAnalyzerAgent', 'SummarizerAgent', 'ClassifierAgent']
        ),
        'pipeline': AppTemplate(
            name='Data Pipeline',
            description='Multi-stage data processing pipeline',
            features=['validation', 'database', 'cache', 'monitoring', 'batch'],
            agents=['IngestAgent', 'ProcessAgent', 'ValidateAgent', 'ExportAgent']
        ),
        'reporting': AppTemplate(
            name='Reporting Application',
            description='Generates sustainability reports and dashboards',
            features=['database', 'llm', 'validation', 'monitoring'],
            agents=['DataCollectorAgent', 'ReportGeneratorAgent', 'ExportAgent']
        ),
        'api-service': AppTemplate(
            name='API Service',
            description='REST API for sustainability data processing',
            features=['api', 'database', 'cache', 'validation', 'monitoring'],
            agents=['RequestValidatorAgent', 'ProcessorAgent', 'ResponseFormatterAgent']
        )
    }

    @classmethod
    def get_template(cls, name: str) -> Optional[AppTemplate]:
        """Get template by name."""
        return cls.TEMPLATES.get(name)

    @classmethod
    def list_templates(cls) -> List[str]:
        """List available templates."""
        return list(cls.TEMPLATES.keys())


class FileGenerator:
    """Generates project files."""

    @staticmethod
    def generate_readme(app_name: str, template: AppTemplate, config: Dict[str, Any]) -> str:
        """Generate README.md."""

        features_list = '\n'.join([f'- {feature.upper()}' for feature in template.features])
        agents_list = '\n'.join([f'- {agent}' for agent in template.agents])

        return f"""# {app_name}

{template.description}

Generated with GreenLang Application Generator

## Features

{features_list}

## Architecture

### Agents

{agents_list}

### Infrastructure

This application uses the GreenLang infrastructure framework:

- **BaseAgent**: All agents inherit from the standardized base class
- **Validation**: Built-in input/output validation
- **Logging**: Structured logging with context
- **Error Handling**: Comprehensive error handling and recovery
{'- **LLM Integration**: ChatSession for AI-powered analysis' if config.get('use_llm') else ''}
{'- **Caching**: CacheManager for performance optimization' if config.get('use_cache') else ''}
{'- **Database**: Database connectivity and ORM support' if config.get('use_database') else ''}
{'- **Monitoring**: Prometheus metrics and health checks' if config.get('use_monitoring') else ''}

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd {app_name}
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Configuration

Environment variables are defined in `.env`:

```bash
# Application
APP_NAME={app_name}
ENVIRONMENT=development
LOG_LEVEL=INFO

{'# LLM Configuration' if config.get('use_llm') else ''}
{'LLM_PROVIDER=' + config.get('llm_provider', 'openai') if config.get('use_llm') else ''}
{'OPENAI_API_KEY=your-key' if config.get('use_llm') and config.get('llm_provider') == 'openai' else ''}
{'ANTHROPIC_API_KEY=your-key' if config.get('use_llm') and config.get('llm_provider') == 'anthropic' else ''}

{'# Cache Configuration' if config.get('use_cache') else ''}
{'CACHE_TYPE=' + config.get('cache_type', 'memory') if config.get('use_cache') else ''}
{'REDIS_URL=redis://localhost:6379' if config.get('use_cache') and config.get('cache_type') in ['redis', 'both'] else ''}

{'# Database Configuration' if config.get('use_database') else ''}
{'DATABASE_URL=postgresql://user:pass@localhost/db' if config.get('use_database') else ''}
```

## Usage

### Run the application:

```bash
python app/main.py
```

### Example usage:

```python
from app.agents.{template.agents[0].lower()} import {template.agents[0]}

# Initialize agent
agent = {template.agents[0]}()

# Execute
input_data = {{
    "data": "your input data"
}}

result = agent.execute(input_data)
print(result)
```

### Batch Processing:

```python
batch_data = [
    {{"data": "input1"}},
    {{"data": "input2"}},
    {{"data": "input3"}}
]

results = agent.batch_execute(batch_data)
```

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_{template.agents[0].lower()}.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=app --cov-report=html
```

## Development

### Project Structure

```
{app_name}/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── agents/
│   │   ├── __init__.py
{chr(10).join([f'│   │   └── {agent.lower()}.py' for agent in template.agents])}
│   └── utils/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
{chr(10).join([f'│   └── test_{agent.lower()}.py' for agent in template.agents])}
├── config.yaml
├── .env.example
├── .env
├── requirements.txt
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
└── .github/
    └── workflows/
        └── ci.yml
```

### Code Style

This project follows:
- PEP 8 style guide
- Black code formatter
- isort for import sorting
- mypy for type checking
- pylint for linting

Run linters:
```bash
black app/ tests/
isort app/ tests/
mypy app/
pylint app/
```

## Docker

Build and run with Docker:

```bash
# Build
docker-compose build

# Run
docker-compose up

# Run in background
docker-compose up -d

# Stop
docker-compose down
```

## CI/CD

This project uses GitHub Actions for CI/CD:

- Linting and code quality checks
- Unit tests with coverage
- Integration tests
- Security scanning
- Docker image building
- Automated deployment (on main branch)

See `.github/workflows/ci.yml` for details.

## Monitoring

{'Access metrics at: http://localhost:9090/metrics' if config.get('use_monitoring') else ''}

{'Health check endpoint: http://localhost:8000/health' if config.get('use_monitoring') else ''}

## License

Copyright (c) 2025 GreenLang

## Support

For issues and questions, please open a GitHub issue.

---

Generated by GreenLang Application Generator
"""

    @staticmethod
    def generate_main(app_name: str, template: AppTemplate, config: Dict[str, Any]) -> str:
        """Generate main.py."""

        imports = [
            'import sys',
            'import os',
            'from pathlib import Path',
            'import json',
            'from app.config import Config',
            'from shared.infrastructure.logging import Logger'
        ]

        # Add agent imports
        for agent in template.agents:
            imports.append(f'from app.agents.{agent.lower()} import {agent}')

        return f'''"""
{app_name} - Main Application Entry Point

{template.description}

Generated by GreenLang Application Generator
"""

{chr(10).join(imports)}


class {app_name.replace('-', '_').title().replace('_', '')}App:
    """Main application class."""

    def __init__(self):
        """Initialize application."""
        self.config = Config()
        self.logger = Logger(name=__name__)

        # Initialize agents
        self.agents = {{
{chr(10).join([f'            "{agent}": {agent}(),' for agent in template.agents])}
        }}

        self.logger.info("Application initialized successfully")

    def run(self, input_data: dict) -> dict:
        """
        Run the application.

        Args:
            input_data: Input data dictionary

        Returns:
            Result dictionary
        """
        self.logger.info("Starting application execution")

        try:
            # Example: Run first agent
            agent = self.agents["{template.agents[0]}"]
            result = agent.execute(input_data)

            self.logger.info("Application execution completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Application execution failed: {{e}}", exc_info=True)
            return {{
                "status": "error",
                "error": str(e)
            }}

    def run_pipeline(self, input_data: dict) -> dict:
        """
        Run all agents in sequence (pipeline mode).

        Args:
            input_data: Input data dictionary

        Returns:
            Final result dictionary
        """
        self.logger.info("Starting pipeline execution")

        data = input_data
        results = []

        try:
            for agent_name, agent in self.agents.items():
                self.logger.info(f"Executing agent: {{agent_name}}")
                result = agent.execute(data)
                results.append(result)

                # Pass output to next agent
                if result.get("status") == "success":
                    data = result.get("result", {{}})
                else:
                    raise Exception(f"Agent {{agent_name}} failed: {{result.get('error')}}")

            self.logger.info("Pipeline execution completed successfully")
            return {{
                "status": "success",
                "results": results,
                "final_output": data
            }}

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {{e}}", exc_info=True)
            return {{
                "status": "error",
                "error": str(e),
                "completed_agents": len(results)
            }}


def main():
    """Main entry point."""
    # Initialize application
    app = {app_name.replace('-', '_').title().replace('_', '')}App()

    # Example input data
    input_data = {{
        "data": "sample input",
        "source": "command line"
    }}

    # Run application
    result = app.run(input_data)

    # Print result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
'''

    @staticmethod
    def generate_config(app_name: str, template: AppTemplate, config: Dict[str, Any]) -> str:
        """Generate config.py."""

        return f'''"""
Configuration Management for {app_name}

Generated by GreenLang Application Generator
"""

import os
from pathlib import Path
from typing import Any, Optional
from shared.infrastructure.config import ConfigManager


class Config:
    """Application configuration."""

    def __init__(self):
        """Initialize configuration."""
        self.config_manager = ConfigManager()
        self.config_manager.load()

        # Application settings
        self.app_name = self.get("APP_NAME", default="{app_name}")
        self.environment = self.get("ENVIRONMENT", default="development")
        self.log_level = self.get("LOG_LEVEL", default="INFO")

        # Validate configuration
        self._validate()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value
        """
        return self.config_manager.get(key, default=default)

    def _validate(self):
        """Validate required configuration."""
        required = ["APP_NAME"]

{'        # LLM configuration' if config.get('use_llm') else ''}
{'        if self.get("USE_LLM", default="false").lower() == "true":' if config.get('use_llm') else ''}
{'            required.extend(["LLM_PROVIDER"])' if config.get('use_llm') else ''}

{'        # Database configuration' if config.get('use_database') else ''}
{'        if self.get("USE_DATABASE", default="false").lower() == "true":' if config.get('use_database') else ''}
{'            required.extend(["DATABASE_URL"])' if config.get('use_database') else ''}

        missing = [key for key in required if not self.get(key)]
        if missing:
            raise ValueError(f"Missing required configuration: {{missing}}")

{'    def get_llm_config(self) -> dict:' if config.get('use_llm') else ''}
{'        """Get LLM configuration."""' if config.get('use_llm') else ''}
{'        provider = self.get("LLM_PROVIDER", default="openai")' if config.get('use_llm') else ''}
{'        return {{' if config.get('use_llm') else ''}
{'            "provider": provider,' if config.get('use_llm') else ''}
{'            "model": self.get("LLM_MODEL", default="gpt-4" if provider == "openai" else "claude-3-opus-20240229"),' if config.get('use_llm') else ''}
{'            "api_key": self.get(f"{{provider.upper()}}_API_KEY"),' if config.get('use_llm') else ''}
{'            "temperature": float(self.get("LLM_TEMPERATURE", default="0.7"))' if config.get('use_llm') else ''}
{'        }}' if config.get('use_llm') else ''}

{'    def get_cache_config(self) -> dict:' if config.get('use_cache') else ''}
{'        """Get cache configuration."""' if config.get('use_cache') else ''}
{'        return {{' if config.get('use_cache') else ''}
{'            "type": self.get("CACHE_TYPE", default="memory"),' if config.get('use_cache') else ''}
{'            "ttl": int(self.get("CACHE_TTL", default="3600")),' if config.get('use_cache') else ''}
{'            "redis_url": self.get("REDIS_URL", default="redis://localhost:6379")' if config.get('use_cache') else ''}
{'        }}' if config.get('use_cache') else ''}

{'    def get_database_config(self) -> dict:' if config.get('use_database') else ''}
{'        """Get database configuration."""' if config.get('use_database') else ''}
{'        return {{' if config.get('use_database') else ''}
{'            "url": self.get("DATABASE_URL"),' if config.get('use_database') else ''}
{'            "pool_size": int(self.get("DB_POOL_SIZE", default="10")),' if config.get('use_database') else ''}
{'            "echo": self.environment == "development"' if config.get('use_database') else ''}
{'        }}' if config.get('use_database') else ''}

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
'''

    @staticmethod
    def generate_agent(agent_name: str, config: Dict[str, Any]) -> str:
        """Generate agent file."""

        imports = ['from shared.infrastructure.agents import BaseAgent']

        if config.get('use_llm'):
            imports.append('from shared.infrastructure.llm import ChatSession')
        if config.get('use_cache'):
            imports.append('from shared.infrastructure.cache import CacheManager')

        imports.extend([
            'from shared.infrastructure.logging import Logger',
            'from shared.infrastructure.validation import ValidationFramework, Field',
            'from typing import Dict, Any, List'
        ])

        return f'''"""
{agent_name}

Generated by GreenLang Application Generator
"""

{chr(10).join(imports)}


class {agent_name}(BaseAgent):
    """
    {agent_name} - Part of the application processing pipeline.
    """

    def __init__(self):
        """Initialize agent."""
        super().__init__()
        self.logger = Logger(name=__name__)
{'        self.chat_session = ChatSession(provider="openai", model="gpt-4")' if config.get('use_llm') else ''}
{'        self.cache = CacheManager(ttl=3600)' if config.get('use_cache') else ''}

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent.

        Args:
            input_data: Input data dictionary

        Returns:
            Result dictionary
        """
        self.logger.info(f"Executing {{self.__class__.__name__}}", extra={{"input": input_data}})

        # Validate input
        if not self.validate_input(input_data):
            return {{
                "status": "error",
                "error": "Invalid input data"
            }}

        try:
            # Process data
            result = self._process(input_data)

            # Validate output
            if not self.validate_output({{"result": result}}):
                raise ValueError("Invalid output data")

            self.logger.info("Execution completed successfully")
            return {{
                "status": "success",
                "result": result
            }}

        except Exception as e:
            self.logger.error(f"Execution failed: {{e}}", exc_info=True)
            return {{
                "status": "error",
                "error": str(e)
            }}

    def _process(self, data: Dict[str, Any]) -> Any:
        """
        Core processing logic.

        Args:
            data: Input data

        Returns:
            Processed result
        """
        # TODO: Implement your processing logic here

        result = {{
            "processed": True,
            "data": data
        }}

        return result

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data.

        Args:
            input_data: Data to validate

        Returns:
            True if valid
        """
        # TODO: Customize validation rules
        required_fields = ["data"]

        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"Missing required field: {{field}}")
                return False

        return True

    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """
        Validate output data.

        Args:
            output_data: Data to validate

        Returns:
            True if valid
        """
        return "result" in output_data
'''

    @staticmethod
    def generate_test(agent_name: str, config: Dict[str, Any]) -> str:
        """Generate test file."""

        return f'''"""
Tests for {agent_name}

Generated by GreenLang Application Generator
"""

import pytest
import json
from app.agents.{agent_name.lower()} import {agent_name}


class Test{agent_name}:
    """Test suite for {agent_name}."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return {agent_name}()

    def test_execute_success(self, agent):
        """Test successful execution."""
        input_data = {{
            "data": "test input"
        }}

        result = agent.execute(input_data)

        assert result["status"] == "success"
        assert "result" in result

    def test_execute_invalid_input(self, agent):
        """Test execution with invalid input."""
        input_data = {{}}

        result = agent.execute(input_data)

        assert result["status"] == "error"

    def test_validate_input_valid(self, agent):
        """Test input validation with valid data."""
        input_data = {{
            "data": "valid"
        }}

        assert agent.validate_input(input_data) is True

    def test_validate_input_invalid(self, agent):
        """Test input validation with invalid data."""
        input_data = {{}}

        assert agent.validate_input(input_data) is False

    def test_batch_execute(self, agent):
        """Test batch execution."""
        batch_data = [
            {{"data": "input1"}},
            {{"data": "input2"}},
            {{"data": "input3"}}
        ]

        results = agent.batch_execute(batch_data)

        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)


class Test{agent_name}Integration:
    """Integration tests for {agent_name}."""

    def test_end_to_end(self):
        """Test complete workflow."""
        agent = {agent_name}()

        # Real data
        real_data = {{
            "data": "production data"
        }}

        result = agent.execute(real_data)

        assert result["status"] == "success"
'''

    @staticmethod
    def generate_requirements(config: Dict[str, Any]) -> str:
        """Generate requirements.txt."""

        reqs = [
            '# Core dependencies',
            'pydantic>=2.0.0',
            'python-dotenv>=1.0.0',
            'pyyaml>=6.0',
            '',
            '# Testing',
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-asyncio>=0.21.0',
            '',
            '# Code quality',
            'black>=23.0.0',
            'isort>=5.12.0',
            'mypy>=1.0.0',
            'pylint>=2.17.0',
            '',
            '# Logging',
            'structlog>=23.0.0',
            ''
        ]

        if config.get('use_llm'):
            reqs.extend([
                '# LLM Integration',
                'openai>=1.0.0' if config.get('llm_provider') in ['openai', 'all'] else '',
                'anthropic>=0.18.0' if config.get('llm_provider') in ['anthropic', 'all'] else '',
                ''
            ])

        if config.get('use_cache'):
            reqs.extend([
                '# Caching',
                'redis>=5.0.0' if config.get('cache_type') in ['redis', 'both'] else '',
                ''
            ])

        if config.get('use_database'):
            reqs.extend([
                '# Database',
                'sqlalchemy>=2.0.0' if config.get('database_type') in ['postgresql', 'both'] else '',
                'psycopg2-binary>=2.9.0' if config.get('database_type') in ['postgresql', 'both'] else '',
                'pymongo>=4.0.0' if config.get('database_type') in ['mongodb', 'both'] else '',
                ''
            ])

        if config.get('use_monitoring'):
            reqs.extend([
                '# Monitoring',
                'prometheus-client>=0.18.0',
                ''
            ])

        return '\n'.join([r for r in reqs if r is not None])

    @staticmethod
    def generate_env_example(app_name: str, config: Dict[str, Any]) -> str:
        """Generate .env.example."""

        lines = [
            '# Application Configuration',
            f'APP_NAME={app_name}',
            'ENVIRONMENT=development',
            'LOG_LEVEL=INFO',
            ''
        ]

        if config.get('use_llm'):
            lines.extend([
                '# LLM Configuration',
                f'LLM_PROVIDER={config.get("llm_provider", "openai")}',
                'LLM_MODEL=gpt-4',
                'LLM_TEMPERATURE=0.7',
                'OPENAI_API_KEY=your-openai-key',
                'ANTHROPIC_API_KEY=your-anthropic-key',
                ''
            ])

        if config.get('use_cache'):
            lines.extend([
                '# Cache Configuration',
                f'CACHE_TYPE={config.get("cache_type", "memory")}',
                'CACHE_TTL=3600',
                'REDIS_URL=redis://localhost:6379',
                ''
            ])

        if config.get('use_database'):
            lines.extend([
                '# Database Configuration',
                'DATABASE_URL=postgresql://user:password@localhost:5432/dbname',
                'DB_POOL_SIZE=10',
                ''
            ])

        if config.get('use_monitoring'):
            lines.extend([
                '# Monitoring',
                'METRICS_PORT=9090',
                'ENABLE_METRICS=true',
                ''
            ])

        return '\n'.join(lines)

    @staticmethod
    def generate_gitignore() -> str:
        """Generate .gitignore."""

        return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
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

# Virtual environments
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
.env.local
.env.*.local

# Testing
.coverage
htmlcov/
.pytest_cache/
.tox/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Application specific
data/
output/
temp/
*.db
*.sqlite
'''

    @staticmethod
    def generate_dockerfile(app_name: str) -> str:
        """Generate Dockerfile."""

        return f'''FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_NAME={app_name}

# Run application
CMD ["python", "app/main.py"]
'''

    @staticmethod
    def generate_docker_compose(app_name: str, config: Dict[str, Any]) -> str:
        """Generate docker-compose.yml."""

        services = {
            'app': {
                'build': '.',
                'container_name': app_name,
                'environment': [
                    'ENVIRONMENT=development'
                ],
                'volumes': [
                    '.:/app'
                ],
                'ports': [
                    '8000:8000'
                ]
            }
        }

        if config.get('use_cache') and config.get('cache_type') in ['redis', 'both']:
            services['redis'] = {
                'image': 'redis:7-alpine',
                'container_name': f'{app_name}-redis',
                'ports': ['6379:6379']
            }
            services['app']['depends_on'] = ['redis']

        if config.get('use_database') and config.get('database_type') == 'postgresql':
            services['postgres'] = {
                'image': 'postgres:15-alpine',
                'container_name': f'{app_name}-postgres',
                'environment': [
                    'POSTGRES_USER=greenlang',
                    'POSTGRES_PASSWORD=greenlang',
                    'POSTGRES_DB=greenlang'
                ],
                'ports': ['5432:5432'],
                'volumes': ['postgres_data:/var/lib/postgresql/data']
            }
            if 'depends_on' not in services['app']:
                services['app']['depends_on'] = []
            services['app']['depends_on'].append('postgres')

        compose = {
            'version': '3.8',
            'services': services
        }

        if config.get('use_database'):
            compose['volumes'] = {'postgres_data': {}}

        import yaml
        return yaml.dump(compose, default_flow_style=False, sort_keys=False)


class AppGenerator:
    """Main application generator."""

    def __init__(self):
        self.file_gen = FileGenerator()

    def interactive_mode(self) -> Dict[str, Any]:
        """Interactive configuration wizard."""

        print("\n" + "="*70)
        print("GreenLang Application Generator - Interactive Mode")
        print("="*70 + "\n")

        # App name
        app_name = input("Application name: ").strip()
        if not app_name:
            logger.error(f" Application name is required")
            sys.exit(1)

        # Template selection
        print("\nAvailable templates:")
        for i, template_name in enumerate(AppTemplates.list_templates(), 1):
            template = AppTemplates.get_template(template_name)
            print(f"  {i}. {template_name}: {template.description}")

        template_choice = input("\nSelect template (1-6): ").strip()
        template_name = AppTemplates.list_templates()[int(template_choice) - 1]
        template = AppTemplates.get_template(template_name)

        # LLM
        use_llm = input("\nUse LLM integration? (y/n): ").lower().strip() == 'y'
        llm_provider = None
        if use_llm:
            print("  1. OpenAI")
            print("  2. Anthropic")
            print("  3. Both")
            llm_choice = input("Select LLM provider (1-3): ").strip()
            llm_provider = ['openai', 'anthropic', 'all'][int(llm_choice) - 1]

        # Cache
        use_cache = input("\nUse caching? (y/n): ").lower().strip() == 'y'
        cache_type = None
        if use_cache:
            print("  1. Memory (in-process)")
            print("  2. Redis")
            print("  3. Both")
            cache_choice = input("Select cache type (1-3): ").strip()
            cache_type = ['memory', 'redis', 'both'][int(cache_choice) - 1]

        # Database
        use_database = input("\nUse database? (y/n): ").lower().strip() == 'y'
        database_type = None
        if use_database:
            print("  1. PostgreSQL")
            print("  2. MongoDB")
            print("  3. Both")
            db_choice = input("Select database (1-3): ").strip()
            database_type = ['postgresql', 'mongodb', 'both'][int(db_choice) - 1]

        # Tests
        generate_tests = input("\nGenerate tests? (y/n): ").lower().strip() == 'y'

        # CI/CD
        generate_cicd = input("\nGenerate CI/CD? (y/n): ").lower().strip() == 'y'

        # Monitoring
        use_monitoring = input("\nAdd monitoring? (y/n): ").lower().strip() == 'y'

        return {
            'app_name': app_name,
            'template': template,
            'use_llm': use_llm,
            'llm_provider': llm_provider,
            'use_cache': use_cache,
            'cache_type': cache_type,
            'use_database': use_database,
            'database_type': database_type,
            'generate_tests': generate_tests,
            'generate_cicd': generate_cicd,
            'use_monitoring': use_monitoring
        }

    def generate(self, config: Dict[str, Any], output_dir: str = None):
        """Generate application."""

        app_name = config['app_name']
        template = config['template']

        if output_dir is None:
            output_dir = f"./{app_name}"

        output_path = Path(output_dir)

        print(f"\nGenerating application: {app_name}")
        print(f"Template: {template.name}")
        print(f"Output directory: {output_path.absolute()}")
        print("\nGenerating files...\n")

        # Create directory structure
        dirs = [
            output_path,
            output_path / 'app',
            output_path / 'app' / 'agents',
            output_path / 'app' / 'utils',
            output_path / 'tests',
            output_path / 'docs'
        ]

        if config.get('generate_cicd'):
            dirs.extend([
                output_path / '.github',
                output_path / '.github' / 'workflows'
            ])

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")

        # Generate files
        files = {
            'README.md': self.file_gen.generate_readme(app_name, template, config),
            'app/main.py': self.file_gen.generate_main(app_name, template, config),
            'app/config.py': self.file_gen.generate_config(app_name, template, config),
            'requirements.txt': self.file_gen.generate_requirements(config),
            '.env.example': self.file_gen.generate_env_example(app_name, config),
            '.gitignore': self.file_gen.generate_gitignore(),
            'Dockerfile': self.file_gen.generate_dockerfile(app_name),
            'docker-compose.yml': self.file_gen.generate_docker_compose(app_name, config),
            'app/__init__.py': '',
            'app/agents/__init__.py': '',
            'app/utils/__init__.py': '',
            'tests/__init__.py': '',
        }

        # Generate agent files
        for agent_name in template.agents:
            files[f'app/agents/{agent_name.lower()}.py'] = self.file_gen.generate_agent(agent_name, config)

            if config.get('generate_tests'):
                files[f'tests/test_{agent_name.lower()}.py'] = self.file_gen.generate_test(agent_name, config)

        # Write all files
        for file_path, content in files.items():
            full_path = output_path / file_path
            full_path.write_text(content, encoding='utf-8')
            print(f"Generated: {file_path}")

        print(f"\n{'='*70}")
        print(f"Application generated successfully!")
        print(f"{'='*70}\n")

        # Post-generation steps
        print("Next steps:")
        print(f"  1. cd {app_name}")
        print(f"  2. python -m venv venv")
        print(f"  3. source venv/bin/activate  (or venv\\Scripts\\activate on Windows)")
        print(f"  4. pip install -r requirements.txt")
        print(f"  5. cp .env.example .env")
        print(f"  6. Edit .env with your configuration")
        print(f"  7. python app/main.py")

        if config.get('generate_tests'):
            print(f"  8. pytest tests/ -v")

        print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='GreenLang Application Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  greenlang create-app my-app

  # Quick start with template
  greenlang create-app my-app --template data-intake

  # Full configuration
  greenlang create-app my-app --template llm-analysis --llm openai --cache redis --database postgresql
        """
    )

    parser.add_argument('name', nargs='?', help='Application name')
    parser.add_argument('--template', choices=AppTemplates.list_templates(), help='Application template')
    parser.add_argument('--llm', choices=['openai', 'anthropic', 'all'], help='LLM provider')
    parser.add_argument('--cache', choices=['memory', 'redis', 'both'], help='Cache type')
    parser.add_argument('--database', choices=['postgresql', 'mongodb', 'both'], help='Database type')
    parser.add_argument('--no-tests', action='store_true', help='Skip test generation')
    parser.add_argument('--no-cicd', action='store_true', help='Skip CI/CD generation')
    parser.add_argument('--no-monitoring', action='store_true', help='Skip monitoring')
    parser.add_argument('--output', help='Output directory')

    args = parser.parse_args()

    generator = AppGenerator()

    # Interactive mode or command-line mode
    if not args.name or not args.template:
        config = generator.interactive_mode()
    else:
        template = AppTemplates.get_template(args.template)
        config = {
            'app_name': args.name,
            'template': template,
            'use_llm': args.llm is not None,
            'llm_provider': args.llm,
            'use_cache': args.cache is not None,
            'cache_type': args.cache,
            'use_database': args.database is not None,
            'database_type': args.database,
            'generate_tests': not args.no_tests,
            'generate_cicd': not args.no_cicd,
            'use_monitoring': not args.no_monitoring
        }

    # Generate application
    generator.generate(config, output_dir=args.output)


if __name__ == '__main__':
    main()
