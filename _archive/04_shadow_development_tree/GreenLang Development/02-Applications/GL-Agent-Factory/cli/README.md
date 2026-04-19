# GreenLang Agent Factory CLI

A powerful command-line interface for generating, validating, testing, and publishing AI agents in the GreenLang ecosystem.

## Installation

```bash
# Install from source
cd GL-Agent-Factory/cli
pip install -e .

# Or install from PyPI (when published)
pip install greenlang-agent-factory-cli
```

## Quick Start

### Initialize a Project

```bash
# Initialize a new Agent Factory project
gl init

# Initialize with a specific template
gl init --template regulatory
```

### Create an Agent

```bash
# Generate agent from specification
gl agent create specs/nfpa86-agent.yaml

# Generate to specific directory
gl agent create specs/nfpa86-agent.yaml --output agents/nfpa86

# Dry run (validate without generating)
gl agent create specs/nfpa86-agent.yaml --dry-run

# Skip tests/docs generation
gl agent create specs/nfpa86-agent.yaml --skip-tests --skip-docs
```

### Validate Specifications

```bash
# Validate an agent specification
gl agent validate specs/nfpa86-agent.yaml

# Strict validation mode
gl agent validate specs/nfpa86-agent.yaml --strict

# Verbose output
gl agent validate specs/nfpa86-agent.yaml --verbose
```

### Test Agents

```bash
# Run tests for an agent
gl agent test agents/nfpa86

# Run with coverage report
gl agent test agents/nfpa86 --coverage

# Verbose test output
gl agent test agents/nfpa86 --verbose

# Run tests serially (not in parallel)
gl agent test agents/nfpa86 --serial
```

### Publish Agents

```bash
# Publish agent to registry
gl agent publish agents/nfpa86

# Publish with specific version tag
gl agent publish agents/nfpa86 --tag v1.2.0

# Force publish (overwrite existing version)
gl agent publish agents/nfpa86 --tag v1.2.0 --force

# Dry run (simulate publish)
gl agent publish agents/nfpa86 --dry-run
```

### List Agents

```bash
# List all local agents
gl agent list

# Filter by type
gl agent list --type regulatory

# Filter by status
gl agent list --status active

# Output as JSON
gl agent list --format json

# Output as YAML
gl agent list --format yaml
```

### Agent Information

```bash
# Show agent details
gl agent info nfpa86-furnace-agent

# Show detailed information
gl agent info nfpa86-furnace-agent --verbose
```

## Template Commands

### List Templates

```bash
# List all available templates
gl template list

# Show detailed template information
gl template list --verbose
```

### Initialize from Template

```bash
# Initialize new agent from template
gl template init regulatory --id my-agent

# Initialize to specific directory
gl template init regulatory --id my-agent --output agents/my-agent
```

### Show Template Details

```bash
# Show template information
gl template show regulatory
```

## Registry Commands

### Search Registry

```bash
# Search for agents
gl registry search "furnace compliance"

# Filter by type
gl registry search "compliance" --type regulatory

# Limit results
gl registry search "compliance" --limit 10
```

### Pull Agents

```bash
# Pull agent from registry
gl registry pull nfpa86-furnace-agent:1.2.0

# Pull to specific directory
gl registry pull nfpa86-furnace-agent:1.2.0 --output agents/nfpa86

# Pull latest version
gl registry pull nfpa86-furnace-agent:latest
```

### Push Agents

```bash
# Push agent to registry
gl registry push agents/nfpa86

# Push with specific tag
gl registry push agents/nfpa86 --tag v1.2.0
```

### Registry Authentication

```bash
# Login to registry
gl registry login

# Login to specific registry
gl registry login --registry https://registry.greenlang.io

# Logout
gl registry logout
```

## Configuration

The CLI uses a configuration file (`config/factory.yaml`) for default settings:

```yaml
# GreenLang Agent Factory Configuration
version: "1.0"

# Default settings for agent generation
defaults:
  output_dir: "agents"
  test_dir: "tests"
  spec_dir: "specs"

# Registry settings
registry:
  url: "https://registry.greenlang.io"
  timeout: 30

# Generator settings
generator:
  enable_validation: true
  enable_tests: true
  enable_documentation: true
  template: "basic"

# Validation settings
validation:
  strict_mode: false
  allow_unknown_fields: true

# Testing settings
testing:
  parallel: true
  verbose: false
  coverage: true
```

## Global Options

All commands support these global options:

- `--version, -v`: Show version and exit
- `--quiet, -q`: Suppress non-essential output
- `--help`: Show help message

## Agent Specification Format

Agent specifications are defined in YAML:

```yaml
metadata:
  id: "nfpa86-furnace-agent"
  name: "NFPA 86 Furnace Compliance Agent"
  version: "1.2.0"
  type: "regulatory"
  description: "Validates industrial furnaces against NFPA 86 standards"

capabilities:
  - "code-compliance-validation"
  - "safety-analysis"
  - "documentation-generation"

architecture:
  framework: "greenlang"
  components:
    - name: "validator"
      type: "compliance"
    - name: "analyzer"
      type: "safety"

dependencies:
  - "greenlang-sdk>=1.0.0"
  - "pydantic>=2.5.0"

configuration:
  standards:
    - "NFPA 86-2023"
  validation_level: "strict"
```

## Examples

### Complete Workflow

```bash
# 1. Initialize project
gl init

# 2. Create agent specification (manually edit specs/my-agent.yaml)

# 3. Validate specification
gl agent validate specs/my-agent.yaml --strict

# 4. Generate agent
gl agent create specs/my-agent.yaml

# 5. Run tests
gl agent test agents/my-agent --coverage

# 6. Publish to registry
gl agent publish agents/my-agent --tag v1.0.0
```

### Using Templates

```bash
# List available templates
gl template list

# Initialize from regulatory template
gl template init regulatory --id nfpa86-agent

# Edit the generated specification
# vim specs/nfpa86-agent.yaml

# Generate the agent
gl agent create specs/nfpa86-agent.yaml
```

### Registry Workflow

```bash
# Login to registry
gl registry login

# Search for existing agents
gl registry search "compliance" --type regulatory

# Pull an agent to use as reference
gl registry pull nfpa86-base:1.0.0 --output reference/nfpa86

# Create your custom agent
gl agent create specs/my-compliance-agent.yaml

# Test thoroughly
gl agent test agents/my-compliance-agent --coverage

# Publish to registry
gl registry push agents/my-compliance-agent --tag v1.0.0
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/greenlang/agent-factory.git
cd agent-factory/cli

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black .
ruff check .
mypy cli
```

### Project Structure

```
cli/
├── __init__.py           # Package initialization
├── main.py              # CLI entry point
├── commands/            # Command implementations
│   ├── __init__.py
│   ├── agent.py         # Agent commands
│   ├── template.py      # Template commands
│   └── registry.py      # Registry commands
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── console.py       # Rich console helpers
│   └── config.py        # Configuration management
├── templates/           # Agent templates
├── pyproject.toml       # Project metadata
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run linting and tests
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- Documentation: https://docs.greenlang.io/agent-factory
- Issues: https://github.com/greenlang/agent-factory/issues
- Discussions: https://github.com/greenlang/agent-factory/discussions

## Version History

- **0.1.0** (2024-12-09)
  - Initial release
  - Agent generation commands
  - Template management
  - Registry integration
  - Rich terminal UI
