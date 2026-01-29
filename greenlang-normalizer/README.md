# GL-FOUND-X-003: GreenLang Unit & Reference Normalizer

A comprehensive unit conversion, reference data resolution, and vocabulary management system for the GreenLang sustainability platform.

## Project Overview

The GreenLang Normalizer provides enterprise-grade unit conversion and reference data normalization capabilities for sustainability and ESG reporting. It ensures consistent measurement units across all GreenLang pipelines while maintaining full audit trails and regulatory compliance.

## Architecture

```
greenlang-normalizer/
├── packages/
│   ├── gl-normalizer-core/      # Core Python library
│   ├── gl-normalizer-service/   # FastAPI microservice
│   ├── gl-normalizer-sdk/       # Python SDK for clients
│   └── gl-normalizer-cli/       # Command-line interface
├── vocab/                        # Vocabulary repository
├── config/                       # Configuration files
├── tests/                        # Cross-package tests
├── docs/                         # Documentation
├── infrastructure/               # Deployment configs
└── review-console/               # Admin UI
```

## Packages

### gl-normalizer-core
The core library providing:
- **Parser**: Parse quantity strings (e.g., "100 kg CO2e")
- **Converter**: Unit conversion with Pint integration
- **Resolver**: Reference data resolution and matching
- **Dimension**: Dimensional analysis for unit compatibility
- **Policy**: Conversion policies and compliance rules
- **Audit**: Provenance tracking and audit trails
- **Vocab**: Vocabulary management and versioning
- **Errors**: Structured error handling

### gl-normalizer-service
FastAPI-based microservice providing:
- REST API for conversions and resolutions
- Kafka integration for async processing
- Admin endpoints for vocabulary management
- Audit logging and compliance reporting

### gl-normalizer-sdk
Python SDK for GreenLang clients:
- Synchronous and async client interfaces
- Vocabulary provider abstraction
- Local caching for performance

### gl-normalizer-cli
Command-line interface for:
- Batch conversions
- Vocabulary management
- Compliance validation
- Development utilities

## Quick Start

### Installation

```bash
# Install core library
pip install gl-normalizer-core

# Install with service dependencies
pip install gl-normalizer-service

# Install SDK
pip install gl-normalizer-sdk

# Install CLI
pip install gl-normalizer-cli
```

### Basic Usage

```python
from gl_normalizer_core import UnitParser, UnitConverter

# Parse a quantity
parser = UnitParser()
quantity = parser.parse("100 kg CO2e")

# Convert units
converter = UnitConverter()
result = converter.convert(quantity, target_unit="t CO2e")
print(result)  # 0.1 t CO2e
```

## Development

### Prerequisites
- Python 3.11+
- Poetry or pip
- Docker (for service development)

### Setup

```bash
# Clone repository
git clone https://github.com/greenlang/greenlang-normalizer.git
cd greenlang-normalizer

# Install development dependencies
pip install -e "packages/gl-normalizer-core[dev]"
pip install -e "packages/gl-normalizer-service[dev]"
pip install -e "packages/gl-normalizer-sdk[dev]"
pip install -e "packages/gl-normalizer-cli[dev]"

# Run tests
pytest tests/
```

### Running the Service

```bash
# Development mode
uvicorn gl_normalizer_service.main:app --reload

# With Docker
docker-compose up -d
```

## Compliance

The normalizer supports compliance with:
- GHG Protocol
- ISO 14064
- CSRD / ESRS
- SEC Climate Disclosure
- ISSB Standards

## License

Copyright (c) 2024-2026 GreenLang. All rights reserved.
