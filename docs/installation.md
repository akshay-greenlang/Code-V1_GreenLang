# Installation Guide

## Quick Start

Install GreenLang CLI using pip:

```bash
pip install greenlang-cli
```

This installs the minimal core platform without ML dependencies, saving 250MB of disk space.

## Requirements

- Python 3.10 or higher (3.11+ recommended)
- pip (Python package manager)
- Optional: Docker for containerized execution
- Optional: cosign for signature verification

## Installation Methods

### 1. Install from PyPI (Recommended)

#### Minimal Installation (No ML - Saves 250MB)

```bash
# Basic installation without ML/vector databases
pip install greenlang-cli
```

This is perfect for:
- Calculation engines
- Regulatory compliance frameworks
- Data intake and validation
- Reporting and exports
- Basic LLM integration (OpenAI, Anthropic)

#### ML-Enhanced Installations

```bash
# Add ML capabilities (PyTorch, transformers, sentence-transformers)
pip install greenlang-cli[ml]

# Add vector databases (Weaviate, ChromaDB, Pinecone, Qdrant, FAISS)
pip install greenlang-cli[vector-db]

# Full AI capabilities (LLM + ML + Vector DBs)
pip install greenlang-cli[ai-full]
```

#### Feature-Specific Installations

```bash
# Analytics and data processing
pip install greenlang-cli[analytics]

# Server components (FastAPI, Celery, Redis)
pip install greenlang-cli[server]

# Security features (Sigstore/cosign support)
pip install greenlang-cli[security]

# Development tools
pip install greenlang-cli[dev]

# Testing framework
pip install greenlang-cli[test]

# Documentation tools
pip install greenlang-cli[doc]
```

#### Complete Installations

```bash
# Full features (excludes ML by default)
pip install greenlang-cli[full]

# Everything including ML/AI
pip install greenlang-cli[all]
```

### 2. Install from Source

```bash
# Clone the repository
git clone https://github.com/greenlang/greenlang.git
cd greenlang

# Install in development mode
pip install -e .

# Or build and install
python -m build
pip install dist/greenlang_cli-*.whl
```

### 3. Docker Installation

```bash
# Pull the official image
docker pull ghcr.io/greenlang/greenlang:latest

# Or use specific version
docker pull ghcr.io/greenlang/greenlang:0.3.0

# Run with Docker
docker run --rm -it ghcr.io/greenlang/greenlang:latest gl --help
```

### 4. Install in Virtual Environment

```bash
# Create virtual environment
python -m venv gl-env

# Activate environment
# On Linux/Mac:
source gl-env/bin/activate
# On Windows:
gl-env\Scripts\activate

# Install GreenLang
pip install greenlang-cli
```

## Verify Installation

After installation, verify GreenLang is working:

```bash
# Check version
gl --version

# Run doctor command
gl doctor

# View help
gl --help
```

## Platform-Specific Notes

### Windows

On Windows, you may need to add Python Scripts to PATH:

```batch
# Add to PATH (adjust path as needed)
set PATH=%PATH%;%USERPROFILE%\AppData\Roaming\Python\Python39\Scripts
```

### macOS

On macOS with Apple Silicon, you might need Rosetta for some dependencies:

```bash
# Install Rosetta if needed
softwareupdate --install-rosetta
```

### Linux

On Linux, ensure you have Python development headers:

```bash
# Debian/Ubuntu
sudo apt-get install python3-dev

# RHEL/CentOS/Fedora
sudo yum install python3-devel

# Alpine
apk add python3-dev
```

## Security Setup

### Install Cosign for Signature Verification

```bash
# Install cosign
# On Linux/Mac:
curl -O -L https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64
chmod +x cosign-linux-amd64
sudo mv cosign-linux-amd64 /usr/local/bin/cosign

# On Windows (PowerShell):
Invoke-WebRequest -Uri https://github.com/sigstore/cosign/releases/latest/download/cosign-windows-amd64.exe -OutFile cosign.exe

# Verify cosign installation
cosign version
```

### Install Sigstore Python (Alternative)

```bash
pip install sigstore
```

## Configuration

### Global Configuration

Create a configuration file at `~/.greenlang/config.yaml`:

```yaml
# Registry settings
registry:
  url: https://hub.greenlang.ai
  verify_signatures: true

# Security settings
security:
  default_deny: true
  require_signatures: true
  dev_mode: false

# Telemetry (opt-in)
telemetry:
  enabled: false
  endpoint: https://telemetry.greenlang.ai
```

### Environment Variables

```bash
# Set development mode (disables some security checks)
export GREENLANG_DEV_MODE=true

# Set custom config path
export GREENLANG_CONFIG=/path/to/config.yaml

# Set log level
export GREENLANG_LOG_LEVEL=DEBUG
```

## Troubleshooting

### Common Issues

**Command not found after installation:**
```bash
# Check if gl is in PATH
which gl

# If not found, check Python scripts directory
python -m site --user-base

# Add to PATH (replace with your path)
export PATH=$PATH:~/.local/bin
```

**Permission denied errors:**
```bash
# Install for current user only
pip install --user greenlang-cli
```

**SSL Certificate errors:**
```bash
# Update certificates
pip install --upgrade certifi

# Or disable SSL verification (NOT RECOMMENDED)
pip install --trusted-host pypi.org greenlang-cli
```

**Missing ML dependencies error:**
```
MissingDependencyError: Missing Optional Dependency: torch
```

This means you're trying to use ML features without installing ML dependencies:
```bash
# Install ML capabilities
pip install greenlang-cli[ml]

# Or install all AI features
pip install greenlang-cli[ai-full]
```

**Check available ML features:**
```bash
# Run the ML feature check utility
python -m greenlang.utils.ml_imports

# Output shows which ML components are installed
```

### Getting Help

```bash
# Run diagnostic command
gl doctor

# Check documentation
gl help <command>

# Report issues
# https://github.com/greenlang/greenlang/issues
```

## Next Steps

- [Getting Started Guide](getting-started.md)
- [Core Concepts](concepts.md)
- [CLI Commands Reference](cli/commands.md)
- [Examples](../examples/README.md)