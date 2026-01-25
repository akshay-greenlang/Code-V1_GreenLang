# Installation Guide

## Quick Install

### From Source (Recommended for Development)

```bash
# Navigate to CLI directory
cd C:\Users\aksha\Code-V1_GreenLang\GL-Agent-Factory\cli

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install in development mode
pip install -e .

# Verify installation
gl --version
```

### From PyPI (When Published)

```bash
# Install from PyPI
pip install greenlang-agent-factory-cli

# Verify installation
gl --version
```

## Installation Options

### Standard Installation

```bash
pip install greenlang-agent-factory-cli
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/greenlang/agent-factory.git
cd agent-factory/cli

# Install with development dependencies
pip install -e ".[dev]"
```

### Minimal Installation

```bash
# Install without optional dependencies
pip install --no-deps greenlang-agent-factory-cli
pip install typer rich pyyaml pydantic
```

## System Requirements

### Python Version

- Python 3.11 or higher
- pip 23.0 or higher (recommended)

### Operating Systems

- Windows 10/11
- macOS 12+
- Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+)

### Dependencies

Core dependencies (automatically installed):
- typer[all] >= 0.12.0
- rich >= 13.7.0
- pyyaml >= 6.0.1
- pydantic >= 2.5.0
- requests >= 2.31.0
- jinja2 >= 3.1.2
- click >= 8.1.7

Development dependencies (optional):
- pytest >= 7.4.3
- pytest-cov >= 4.1.0
- black >= 23.12.0
- ruff >= 0.1.8
- mypy >= 1.7.1

## Post-Installation Setup

### Verify Installation

```bash
# Check version
gl --version

# Show help
gl --help

# Test basic command
gl agent list
```

### Initialize Project

```bash
# Create new project directory
mkdir my-agent-project
cd my-agent-project

# Initialize Agent Factory project
gl init

# Verify structure
ls -R
```

### Configure CLI

```bash
# Edit configuration
vim config/factory.yaml

# Or use default configuration
# No configuration needed for basic usage
```

## Troubleshooting

### Command Not Found

If `gl` command is not found after installation:

```bash
# Check if CLI is installed
pip show greenlang-agent-factory-cli

# Check Python scripts directory is in PATH
# On Windows:
echo %PATH% | findstr Python

# On Linux/Mac:
echo $PATH | grep python

# Add to PATH if needed
# On Windows (PowerShell):
$env:Path += ";C:\Python311\Scripts"

# On Linux/Mac:
export PATH="$HOME/.local/bin:$PATH"
```

### Import Errors

If you get import errors:

```bash
# Reinstall with all dependencies
pip uninstall greenlang-agent-factory-cli
pip install --upgrade greenlang-agent-factory-cli

# Or install missing dependencies manually
pip install typer[all] rich pyyaml pydantic requests jinja2
```

### Permission Errors

On Linux/Mac, if you get permission errors:

```bash
# Install for current user only
pip install --user greenlang-agent-factory-cli

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install greenlang-agent-factory-cli
```

### Version Conflicts

If you have version conflicts:

```bash
# Create fresh virtual environment
python -m venv fresh-venv
source fresh-venv/bin/activate  # On Windows: fresh-venv\Scripts\activate
pip install greenlang-agent-factory-cli
```

## Upgrading

### Upgrade to Latest Version

```bash
# Upgrade from PyPI
pip install --upgrade greenlang-agent-factory-cli

# Or upgrade development installation
cd cli
git pull
pip install -e .
```

### Upgrade Dependencies

```bash
# Upgrade all dependencies
pip install --upgrade typer rich pyyaml pydantic requests jinja2

# Or reinstall with latest dependencies
pip install --force-reinstall greenlang-agent-factory-cli
```

## Uninstallation

### Remove CLI

```bash
# Uninstall package
pip uninstall greenlang-agent-factory-cli

# Remove configuration (optional)
rm -rf ~/.config/greenlang-cli

# Remove virtual environment (if used)
rm -rf venv
```

### Clean Uninstall

```bash
# Remove all traces
pip uninstall greenlang-agent-factory-cli
rm -rf ~/.config/greenlang-cli
rm -rf ~/.cache/greenlang-cli
rm -rf venv
```

## Docker Installation

### Using Docker

```dockerfile
# Create Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install CLI
RUN pip install greenlang-agent-factory-cli

# Set entrypoint
ENTRYPOINT ["gl"]
```

```bash
# Build image
docker build -t greenlang-cli .

# Run CLI in container
docker run --rm greenlang-cli --version

# Use with local files
docker run --rm -v $(pwd):/app greenlang-cli agent create specs/my-agent.yaml
```

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  greenlang-cli:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - .:/app
    command: sh -c "pip install greenlang-agent-factory-cli && gl agent list"
```

```bash
# Run with Docker Compose
docker-compose run greenlang-cli agent create specs/my-agent.yaml
```

## Alternative Installation Methods

### Using pipx (Isolated Installation)

```bash
# Install pipx
python -m pip install --user pipx
python -m pipx ensurepath

# Install CLI with pipx
pipx install greenlang-agent-factory-cli

# Verify
gl --version
```

### Using conda

```bash
# Create conda environment
conda create -n greenlang python=3.11
conda activate greenlang

# Install CLI
pip install greenlang-agent-factory-cli

# Verify
gl --version
```

### Building from Source

```bash
# Clone repository
git clone https://github.com/greenlang/agent-factory.git
cd agent-factory/cli

# Install build tools
pip install build

# Build package
python -m build

# Install built package
pip install dist/greenlang_agent_factory_cli-0.1.0-py3-none-any.whl
```

## IDE Integration

### VS Code

Add to `.vscode/settings.json`:

```json
{
  "python.terminal.activateEnvironment": true,
  "python.defaultInterpreterPath": "./venv/bin/python",
  "terminal.integrated.env.windows": {
    "PATH": "${workspaceFolder}\\venv\\Scripts;${env:PATH}"
  }
}
```

### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add → Virtual Environment → Existing
3. Select `venv/bin/python`
4. Mark `cli` directory as Sources Root

## Verification Checklist

After installation, verify:

- [ ] `gl --version` shows correct version
- [ ] `gl --help` displays help message
- [ ] `gl agent list` runs without errors
- [ ] `gl init` creates project structure
- [ ] Configuration file can be created and edited
- [ ] All dependencies are installed correctly

## Getting Help

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section above
2. Search existing issues: https://github.com/greenlang/agent-factory/issues
3. Create new issue with:
   - Python version (`python --version`)
   - CLI version (`gl --version`)
   - Operating system
   - Full error message
   - Steps to reproduce

## Next Steps

After successful installation:

1. Read the [README.md](README.md) for usage instructions
2. Try the Quick Start guide
3. Initialize your first project: `gl init`
4. Create an agent: `gl agent create specs/my-agent.yaml`
5. Explore available templates: `gl template list`
