# gl-normalizer-core

Core library for the GreenLang Unit & Reference Normalizer.

## Overview

This package provides the foundational components for unit conversion, reference data resolution, and vocabulary management in the GreenLang sustainability platform.

## Features

- **Unit Parsing**: Parse quantity strings like "100 kg CO2e", "1.5 MWh", "500 m3"
- **Unit Conversion**: Convert between compatible units with dimensional analysis
- **Reference Resolution**: Match and resolve reference data (fuels, materials, processes)
- **Vocabulary Management**: Version-controlled vocabularies with governance
- **Policy Enforcement**: Compliance-aware conversion policies
- **Audit Trails**: Full provenance tracking for regulatory compliance

## Installation

```bash
pip install gl-normalizer-core
```

## Quick Start

```python
from gl_normalizer_core import UnitParser, UnitConverter, ReferenceResolver

# Parse a quantity string
parser = UnitParser()
quantity = parser.parse("100 kg CO2e")
print(quantity)  # Quantity(magnitude=100, unit=kg_CO2e)

# Convert to different unit
converter = UnitConverter()
result = converter.convert(quantity, "t CO2e")
print(result)  # Quantity(magnitude=0.1, unit=t_CO2e)

# Resolve a reference
resolver = ReferenceResolver()
match = resolver.resolve("natural gas", vocabulary="fuels")
print(match)  # ResolvedReference(id="FUEL_001", name="Natural Gas", ...)
```

## Modules

### parser
Parse quantity strings with support for various formats and unit aliases.

### converter
Unit conversion engine built on Pint with GreenLang extensions.

### resolver
Reference data resolution with fuzzy matching and confidence scoring.

### dimension
Dimensional analysis for unit compatibility checking.

### policy
Conversion policies and compliance rules.

### audit
Provenance tracking and audit trail generation.

### vocab
Vocabulary loading, versioning, and management.

### errors
Structured error types for the normalizer.

## Configuration

```python
from gl_normalizer_core import NormalizerConfig

config = NormalizerConfig(
    vocab_path="/path/to/vocab",
    cache_enabled=True,
    strict_mode=True,
    compliance_profile="ghg_protocol",
)
```

## License

Copyright (c) 2024-2026 GreenLang. All rights reserved.
