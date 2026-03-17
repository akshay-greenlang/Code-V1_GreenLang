# PACK-004: CBAM Readiness Pack

**Complete EU Carbon Border Adjustment Mechanism compliance solution powered by GreenLang AI agents**

## Overview

The CBAM Readiness Pack bundles GreenLang's AI agents, applications, and infrastructure into a ready-to-deploy solution for EU Carbon Border Adjustment Mechanism (CBAM) compliance. It covers the full CBAM lifecycle: import data collection, embedded emissions calculation, quarterly reporting, annual declarations, certificate management, supplier data collection, de minimis monitoring, and verification coordination.

| Metric | Value |
|--------|-------|
| Goods Categories | 6 (cement, iron & steel, aluminium, fertilizers, electricity, hydrogen) |
| CN Codes Covered | 50+ from CBAM Annex I |
| Calculation Methods | Actual, Country Default, EU Default |
| Compliance Rules | 187 automated checks |
| Engines | 7 (calculation, certificate, quarterly, supplier, de minimis, verification, policy) |
| Workflows | 7 (quarterly, annual, onboarding, certificate, verification, de minimis, collection) |
| Templates | 8 (reports, dashboards, scorecards, projections) |
| Presets | 9 (6 commodity + 3 sector) |
| Calculation Accuracy | 100% (zero hallucination - deterministic only) |

## Regulatory Coverage

- **CBAM Regulation**: Regulation (EU) 2023/956
- **CBAM Implementing Regulation**: Implementing Regulation (EU) 2023/1773
- **EU ETS**: Directive 2003/87/EC (as amended by Directive (EU) 2023/959)
- **CBAM Delegated Acts**: Commission Delegated Regulation (EU) 2024/XXX

## CBAM Timeline

| Period | Dates | Obligations |
|--------|-------|-------------|
| Transitional | Oct 2023 - Dec 2025 | Quarterly reports (no certificates) |
| Definitive | Jan 2026 onwards | Annual declarations + CBAM certificates |
| Free allocation phase-out | 2026 - 2034 | Gradual reduction from 97.5% to 0% |

## Quick Start

```python
from packs.eu_compliance.pack_004_cbam_readiness.config.pack_config import CBAMPackConfig

# 1. Load from a preset
config = CBAMPackConfig.from_preset("steel_importer")

# 2. Or load from YAML
config = CBAMPackConfig.from_yaml("config/demo/demo_config.yaml")

# 3. Or load demo configuration
config = CBAMPackConfig.from_demo()

# 4. Validate configuration
issues = config.validate_config()
if not issues:
    print("Configuration is valid")

# 5. Check summary
print(config.summary())

# 6. Estimate certificate cost
cost = config.estimate_certificate_cost(
    embedded_emissions_tco2e=10000.0,
    year=2026,
    scenario=None,  # uses default
    country_carbon_price_eur=8.50,
)
print(f"Net CBAM cost: EUR {cost['net_cost_eur']:,.2f}")
```

## Pack Structure

```
PACK-004-cbam-readiness/
├── pack.yaml                         # Pack manifest (components, engines, workflows)
├── README.md                         # This file
├── config/
│   ├── __init__.py                   # Configuration module exports
│   ├── pack_config.py                # Pydantic configuration manager
│   ├── presets/                       # Commodity presets
│   │   ├── steel_importer.yaml       # Iron & steel (CN 72xx/73xx)
│   │   ├── aluminum_importer.yaml    # Aluminium (CN 76xx)
│   │   ├── cement_importer.yaml      # Cement (CN 2523)
│   │   ├── fertilizer_importer.yaml  # Fertilizers (CN 28xx/31xx)
│   │   ├── multi_commodity.yaml      # All 6 categories
│   │   └── small_importer.yaml       # De minimis eligible
│   ├── sectors/                       # Sector presets
│   │   ├── heavy_industry.yaml       # Steel + cement + aluminium
│   │   ├── chemicals.yaml            # Fertilizers + hydrogen
│   │   └── energy_trading.yaml       # Electricity imports
│   └── demo/                          # Demo configuration
│       ├── demo_config.yaml          # EuroSteel Imports GmbH demo
│       ├── demo_imports.csv          # 150 sample import records
│       └── demo_supplier_data.json   # 3 demo suppliers
```

## Goods Categories

| Category | CN Codes | Key Products | Typical Emission Intensity |
|----------|----------|--------------|---------------------------|
| Cement | 2523 | Clinker, Portland, Aluminous | 0.5 - 1.0 tCO2e/t |
| Iron & Steel | 7201-7326 | HRC, CRC, bars, tubes, structures | 0.45 - 2.5 tCO2e/t |
| Aluminium | 7601-7616 | Ingots, sheets, profiles, foil | 0.5 - 14.0 tCO2e/t |
| Fertilizers | 2808, 2814, 2834, 3102-3105 | Ammonia, urea, AN, NPK | 1.5 - 4.5 tCO2e/t |
| Electricity | 2716 | Electrical energy | 0.01 - 1.2 tCO2e/MWh |
| Hydrogen | 2804.10 | Hydrogen gas | 0.5 - 12.0 tCO2e/t |

## Configuration

### Presets

Choose from 6 commodity presets and 3 sector presets:

```python
# Single commodity
config = CBAMPackConfig.from_preset("steel_importer")
config = CBAMPackConfig.from_preset("cement_importer")
config = CBAMPackConfig.from_preset("aluminum_importer")
config = CBAMPackConfig.from_preset("fertilizer_importer")

# Multi-commodity
config = CBAMPackConfig.from_preset("multi_commodity")

# Small importer (de minimis)
config = CBAMPackConfig.from_preset("small_importer")

# With sector overlay
config = CBAMPackConfig.from_preset("steel_importer", sector_name="heavy_industry")
```

### Environment Variables

Override configuration via environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `CBAM_PACK_REPORTING_YEAR` | Reporting year | `2027` |
| `CBAM_PACK_DEMO_MODE` | Enable demo mode | `true` |
| `CBAM_PACK_IMPORTER_COMPANY_NAME` | Company name | `Acme GmbH` |
| `CBAM_PACK_IMPORTER_EORI` | EORI number | `DE123456789` |
| `CBAM_PACK_EMISSION_METHOD` | Calculation method | `actual` |
| `CBAM_PACK_SUPPLIER_QUALITY_THRESHOLD` | Quality threshold | `75.0` |

## Standalone Design

PACK-004 is fully standalone. It does **not** depend on PACK-001, PACK-002, or PACK-003 (CSRD packs). It can be deployed independently for organizations that need CBAM compliance without CSRD reporting.

## Components

- **Application**: GL-CBAM-APP v1.1 (CBAM compliance UI)
- **Foundation Agents**: Orchestrator, Schema, Units, Citations, Reproducibility, Observability
- **Data Agents**: PDF Extractor, Excel Normalizer, ERP Connector, Supplier Questionnaire
- **Quality Agents**: Data Profiler, Duplicate Detection, Validation Rules
- **Industry Calculators**: Steel, Cement, Aluminium embedded emissions
- **Industrial MRV**: Steel, Cement, Aluminium monitoring agents
- **Policy Agent**: CBAM compliance rule enforcement (187 rules)

## Dependencies

- Python >= 3.11
- PostgreSQL >= 16
- Redis >= 7
- pydantic >= 2.0
- pyyaml >= 6.0
- pandas >= 2.0

## Development

```bash
# Run configuration validation
python -c "from config.pack_config import CBAMPackConfig; c = CBAMPackConfig(); print(c.validate_config())"

# Load and inspect demo
python -c "from config.pack_config import CBAMPackConfig; c = CBAMPackConfig.from_demo(); print(c.summary())"

# Run tests
pytest tests/packs/eu_compliance/pack_004_cbam_readiness/ -v
```
