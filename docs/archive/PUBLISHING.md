# GreenLang PyPI Publishing Guide

## Overview

This guide documents the process for publishing GreenLang to PyPI (Python Package Index). GreenLang is published as `greenlang-cli` and supports both automated (GitHub Actions) and manual publishing workflows.

## Package Information

- **Package Name:** `greenlang-cli`
- **Current Version:** `0.3.0`
- **Python Requirements:** >= 3.11
- **License:** MIT
- **Homepage:** https://greenlang.io
- **Repository:** https://github.com/greenlang/greenlang

## Prerequisites

### 1. PyPI Account Setup

1. Create accounts on:
   - [Test PyPI](https://test.pypi.org/account/register/) - For testing
   - [PyPI](https://pypi.org/account/register/) - For production

2. Enable 2FA (Two-Factor Authentication) on both accounts

3. Generate API tokens:
   - Go to Account Settings → API tokens
   - Create tokens with appropriate scope
   - Save tokens securely

### 2. Environment Setup

```bash
# Install required tools
pip install --upgrade pip build twine wheel setuptools
pip install --upgrade check-wheel-contents pip-audit

# For dependency compilation
pip install --upgrade pip-tools pipdeptree
```

### 3. Set API Tokens (Optional, for automation)

```bash
# Linux/Mac
export TESTPYPI_API_TOKEN="pypi-..."
export PYPI_API_TOKEN="pypi-..."

# Windows PowerShell
$env:TESTPYPI_API_TOKEN="pypi-..."
$env:PYPI_API_TOKEN="pypi-..."
```

## Dependency Management

### Compile Dependencies with Hashes

We use `pip-compile` to generate reproducible requirements with SHA256 hashes:

```bash
# Compile all requirement files
python scripts/compile_requirements.py

# This generates:
# - requirements-compiled.txt (from requirements.in)
# - requirements-dev-compiled.txt (from requirements-dev.in)
# - requirements-docs-compiled.txt (from requirements-docs.in)
```

### Verify Dependencies

```bash
# Check for CVEs
pip-audit

# Verify package hashes
python scripts/verify_hashes.py

# Generate dependency graph
pipdeptree --json > docs/dependency-graph.json
```

### Pin Dependencies in pyproject.toml

All dependencies in `pyproject.toml` are pinned to exact versions:

```toml
dependencies = [
    "typer==0.9.0",
    "pydantic==2.5.3",
    "pyyaml==6.0.1",
    # ... etc
]
```

## Building the Package

### 1. Update Version

Edit `pyproject.toml`:

```toml
[project]
version = "0.3.0"  # Update this
```

### 2. Clean Previous Builds

```bash
rm -rf build/ dist/ *.egg-info
```

### 3. Build Distributions

```bash
# Build both source and wheel distributions
python -m build

# This creates:
# - dist/greenlang_cli-0.3.0.tar.gz (source)
# - dist/greenlang_cli-0.3.0-py3-none-any.whl (wheel)
```

### 4. Verify Build

```bash
# Check wheel contents
check-wheel-contents dist/*.whl

# Validate distributions
python -m twine check dist/*

# Generate checksums
cd dist && sha256sum * > SHA256SUMS && cd ..
```

## Publishing Workflows

### Option 1: Automated via GitHub Actions (Recommended)

The `.github/workflows/publish-pypi.yml` workflow handles the entire process:

#### For Testing (TestPyPI)

```bash
# Trigger manually via GitHub UI
# Go to Actions → Publish to PyPI → Run workflow
# Select: environment = testpypi
```

#### For Production (PyPI)

```bash
# Option 1: Create a GitHub release
git tag v0.3.0
git push origin v0.3.0
# Then create release on GitHub

# Option 2: Manual trigger
# Go to Actions → Publish to PyPI → Run workflow
# Select: environment = pypi
```

### Option 2: Manual Publishing

#### Linux/Mac

```bash
# Test PyPI (default)
./scripts/publish.sh

# Production PyPI
./scripts/publish.sh pypi
```

#### Windows

```powershell
# Test PyPI (default)
.\scripts\publish.ps1

# Production PyPI
.\scripts\publish.ps1 pypi
```

### Option 3: Direct Twine Commands

#### Test PyPI

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ greenlang-cli
```

#### Production PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*

# Install from PyPI
pip install greenlang-cli
```

## Testing the Published Package

### 1. Create Test Environment

```bash
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# or
test_env\Scripts\activate  # Windows
```

### 2. Install and Test

```bash
# From TestPyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ greenlang-cli

# From PyPI
pip install greenlang-cli

# Test import
python -c "import greenlang; print(greenlang.__version__)"

# Test CLI
gl --version
greenlang --version
```

### 3. Test Optional Dependencies

```bash
# Install with extras
pip install greenlang-cli[ai-full]
pip install greenlang-cli[server]
pip install greenlang-cli[dev,test]
```

## Docker Integration

After publishing to PyPI, update Docker images:

### Update Dockerfile

```dockerfile
# Change from local install
# RUN pip install -e .

# To PyPI install
RUN pip install greenlang-cli==0.3.0
```

### Build and Test Docker

```bash
docker build -t greenlang:0.3.0 .
docker run --rm greenlang:0.3.0 gl --version
```

## Security Checklist

- [ ] All dependencies pinned to exact versions
- [ ] SHA256 hashes generated for all packages
- [ ] CVE scan completed (`pip-audit`)
- [ ] API tokens stored securely (never in code)
- [ ] 2FA enabled on PyPI accounts
- [ ] Package signed with GPG (optional)
- [ ] SBOM generated (`cyclonedx-bom`)

## Release Checklist

1. **Pre-release**
   - [ ] Update version in `pyproject.toml`
   - [ ] Update CHANGELOG.md
   - [ ] Run full test suite
   - [ ] Compile and pin dependencies
   - [ ] Security audit (`pip-audit`)

2. **Build**
   - [ ] Clean previous builds
   - [ ] Build distributions
   - [ ] Check wheel contents
   - [ ] Validate with twine

3. **Test Release**
   - [ ] Upload to TestPyPI
   - [ ] Test installation from TestPyPI
   - [ ] Verify all extras work

4. **Production Release**
   - [ ] Create git tag
   - [ ] Upload to PyPI
   - [ ] Create GitHub release
   - [ ] Upload release assets

5. **Post-release**
   - [ ] Update Docker images
   - [ ] Update documentation
   - [ ] Announce release

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Ensure API token is correct
   - Token must start with `pypi-`
   - Check token scope (project vs user)

2. **Version Already Exists**
   - PyPI doesn't allow overwriting versions
   - Increment version number
   - Use TestPyPI for testing

3. **Missing Dependencies**
   - Ensure all dependencies are in pyproject.toml
   - Run `pip install -e .[all]` locally to test

4. **Import Errors After Installation**
   - Check MANIFEST.in includes all necessary files
   - Verify package structure with `check-wheel-contents`

### Getting Help

- PyPI Documentation: https://packaging.python.org/
- Twine Documentation: https://twine.readthedocs.io/
- GitHub Actions: https://docs.github.com/en/actions
- GreenLang Issues: https://github.com/greenlang/greenlang/issues

## Package Metadata

The package metadata in `pyproject.toml` includes:

- **Classifiers:** Development status, intended audience, license
- **Keywords:** climate, emissions, sustainability, etc.
- **URLs:** Homepage, documentation, repository, bug tracker
- **Entry Points:** `gl` and `greenlang` CLI commands
- **Optional Dependencies:** Groups for different use cases

## Dependency Summary

### Core Dependencies (13 packages)
- Framework: typer, pydantic, rich
- Web/API: httpx, requests
- Data: pyyaml, jsonschema, packaging
- Utils: python-dotenv, typing-extensions, networkx, tenacity, psutil

### Optional Groups
- **analytics:** pandas, numpy
- **llm:** openai, langchain, anthropic
- **ml:** torch, transformers, sentence-transformers
- **vector-db:** weaviate, chromadb, pinecone, faiss, qdrant
- **server:** fastapi, uvicorn, redis, celery
- **security:** cryptography, PyJWT
- **test:** pytest suite, coverage, hypothesis
- **dev:** mypy, ruff, black, isort, bandit
- **doc:** mkdocs, mkdocs-material

## Version History

- **0.3.0** (Current) - Production-ready release with full infrastructure
- **0.2.x** - Beta releases with core features
- **0.1.x** - Alpha releases for testing

---

**Last Updated:** 2025-01-15
**Maintainer:** GreenLang Team