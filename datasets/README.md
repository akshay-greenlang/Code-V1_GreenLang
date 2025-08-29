# GreenLang Datasets

This directory contains datasets, emission factors, and knowledge bases for GreenLang.

## Structure

```
datasets/
├── README.md
├── emission_factors/     # Global emission factor databases
│   ├── global/          # Country-specific factors
│   ├── regional/        # Regional/state-level factors
│   └── industry/        # Industry-specific factors
├── benchmarks/          # Industry benchmark data
│   ├── buildings/       # Building performance benchmarks
│   ├── industrial/      # Industrial process benchmarks
│   └── transport/       # Transportation benchmarks
├── knowledge_base/      # RAG knowledge base (moved from root)
│   ├── documents/       # Source documents
│   └── vector_store/    # Indexed vectors
└── reference/           # Reference datasets and standards
    ├── ghg_protocol/    # GHG Protocol data
    ├── ipcc/           # IPCC emission factors
    └── iso/            # ISO standards data
```

## Available Datasets

### Emission Factors
- **Global Grid Factors**: Electricity emission factors for 195+ countries
- **Fuel Factors**: Emission factors for 50+ fuel types
- **Process Factors**: Industry-specific process emissions

### Benchmarks
- **Building Performance**: Energy use intensity by building type
- **Industrial Standards**: Emissions per unit production
- **Transport Metrics**: Emissions per passenger/ton-km

### Knowledge Base
- Climate science fundamentals
- GHG Protocol guidelines
- Renewable energy technologies
- Building efficiency standards

## Data Sources

All data is sourced from authoritative organizations:
- IPCC (Intergovernmental Panel on Climate Change)
- IEA (International Energy Agency)
- EPA (Environmental Protection Agency)
- GHG Protocol
- National grid operators

## Usage

```python
from greenlang.data import EmissionFactors

# Load emission factors
factors = EmissionFactors()
grid_factor = factors.get_grid_factor("US", "CA")  # California grid
fuel_factor = factors.get_fuel_factor("natural_gas")
```

## Updates

Datasets are versioned and updated quarterly. Check [CHANGELOG.md](../CHANGELOG.md) for updates.

## Contributing

To contribute data:
1. Ensure data is from authoritative sources
2. Include metadata and citations
3. Add validation tests
4. Submit via pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.