# Developer Guide

This guide covers development setup, architecture, and contribution guidelines for the GreenLang Normalizer.

## Contents

- `getting-started.md` - Development environment setup
- `architecture.md` - System architecture overview
- `contributing.md` - Contribution guidelines
- `testing.md` - Testing guide
- `debugging.md` - Debugging tips

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Poetry or pip

### Setup

```bash
# Clone repository
git clone https://github.com/greenlang/greenlang-normalizer.git
cd greenlang-normalizer

# Install core library
pip install -e "packages/gl-normalizer-core[dev]"

# Install service
pip install -e "packages/gl-normalizer-service[dev]"

# Run tests
pytest

# Start service
uvicorn gl_normalizer_service.main:app --reload
```

## Architecture

```
                    ┌─────────────────┐
                    │   API Gateway   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Normalizer API │
                    │   (FastAPI)     │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
   ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
   │    Parser     │ │   Converter   │ │   Resolver    │
   │   (Pint)      │ │               │ │  (RapidFuzz)  │
   └───────────────┘ └───────────────┘ └───────────────┘
                             │
                    ┌────────▼────────┐
                    │   Vocabulary    │
                    │    Service      │
                    └─────────────────┘
```

## Code Style

- Follow PEP 8
- Use type hints on all functions
- Write docstrings for all public methods
- Run `ruff check` before committing
- Run `mypy` for type checking

## Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Property-based tests
pytest tests/property/

# Golden file tests
pytest tests/golden/

# Coverage report
pytest --cov=gl_normalizer_core --cov-report=html
```

## Documentation

```bash
# Build docs
cd docs
mkdocs build

# Serve docs locally
mkdocs serve
```
