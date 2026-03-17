# PACK-006: EUDR Starter Pack

**Complete EU Deforestation Regulation compliance solution powered by 34+ GreenLang AI agents**

## Overview

The EUDR Starter Pack bundles GreenLang's AI agents, data connectors, and infrastructure into a ready-to-deploy solution for EU Deforestation Regulation (EUDR) compliance. It covers due diligence statement generation, geolocation verification, supply chain traceability, risk assessment, and EU Information System integration.

| Metric | Value |
|--------|-------|
| EUDR Agents | 18 (of 40 total) |
| Data Agents | 6 |
| Foundation Agents | 10 |
| Commodities | 7 (cattle, cocoa, coffee, oil palm, rubber, soya, wood) |
| Annex I CN Codes | 400+ product classifications |
| Country Risk Database | 200+ countries benchmarked |
| Compliance Rules | 10 automated checks |
| Calculation Accuracy | 100% (zero hallucination) |
| Report Templates | 7 pre-built templates |

## Regulatory Coverage

- **EUDR**: Regulation (EU) 2023/1115
- **Article 3**: Prohibition on non-compliant products
- **Articles 4, 8-12**: Due diligence obligations
- **Article 9**: Information collection (geolocation, quantities, dates)
- **Article 13**: Simplified due diligence for low-risk countries
- **Article 29**: Country benchmarking system
- **Article 33**: EU Information System

## Quick Start

```python
from packs.eu_compliance.pack_006_eudr_starter.config.pack_config import PackConfig

# 1. Load configuration with presets
config = PackConfig.load(
    size_preset="mid_market",
    sector_preset="palm_oil",
)

# 2. Check country risk
from packs.eu_compliance.pack_006_eudr_starter.config.pack_config import get_country_risk
risk = get_country_risk("IDN")  # Returns CountryBenchmark.HIGH_RISK

# 3. Check CN code scope
from packs.eu_compliance.pack_006_eudr_starter.config.pack_config import is_eudr_commodity
in_scope, commodity = is_eudr_commodity("1511")  # (True, "oil_palm")

# 4. View active agents
print(config.active_agents)
```

## Pack Structure

```
PACK-006-eudr-starter/
|-- pack.yaml                    # Pack manifest (1,600+ lines)
|-- README.md                    # This documentation
|-- config/                      # Configuration (16 files)
|   |-- __init__.py              # Exports all config classes
|   |-- pack_config.py           # Configuration manager (2,200+ lines)
|   |-- presets/                 # Size presets
|   |   |-- large_operator.yaml  # Full deployment for large operators
|   |   |-- mid_market.yaml      # Mid-market optimized
|   |   |-- sme.yaml             # SME/trader simplified
|   |   |-- first_time.yaml      # First-time declarant guided
|   |-- sectors/                 # Commodity sector presets
|   |   |-- palm_oil.yaml        # Palm oil supply chain
|   |   |-- timber_wood.yaml     # Timber and wood products
|   |   |-- cocoa_coffee.yaml    # Cocoa and coffee
|   |   |-- soy_cattle.yaml      # Soy and cattle
|   |   |-- rubber.yaml          # Natural rubber
|   |-- demo/                    # Demo mode data
|       |-- demo_config.yaml     # Demo settings
|       |-- demo_suppliers.json  # 10 sample suppliers
|       |-- demo_plots.geojson   # 20 sample plot geolocations
```

## Configuration Presets

### By Company Size

| Preset | Type | Tier Depth | DDS Type | Guided Mode |
|--------|------|------------|----------|-------------|
| Large Operator | OPERATOR | 5 tiers | Standard | No |
| Mid-Market | OPERATOR | 3 tiers | Standard | No |
| SME | TRADER | 2 tiers | Simplified eligible | Yes |
| First-Time | Any | 2 tiers | Standard (guided) | Yes |

### By Commodity Sector

| Preset | Primary Commodities | Key Certifications | Focus Regions |
|--------|--------------------|--------------------|---------------|
| Palm Oil | Oil palm | RSPO, ISCC, MSPO | SEA, West Africa |
| Timber & Wood | Wood | FSC, PEFC, SFI | Global |
| Cocoa & Coffee | Cocoa, Coffee | RA, UTZ, 4C, Fairtrade | West Africa, LAM |
| Soy & Cattle | Soya, Cattle | RTRS, ProTerra | South America |
| Rubber | Rubber | FSC, PEFC | SEA |

## Commodity Configuration

All 7 EUDR commodities are configurable. Enable/disable per commodity:

```yaml
# In preset or override
commodities:
  oil_palm:
    enabled: true
    priority: 1
    certification_schemes: [RSPO, ISCC]
  cattle:
    enabled: false
```

## Risk Assessment

Composite risk scoring with configurable weights (must sum to 1.0):

| Factor | Default Weight | Source |
|--------|---------------|--------|
| Country Risk | 35% | Article 29 benchmarking |
| Supplier Risk | 25% | Supplier scoring engine |
| Commodity Risk | 20% | Commodity-specific factors |
| Documentation | 20% | Document completeness |

Risk levels: LOW (0-25), MEDIUM (25-50), HIGH (50-75), CRITICAL (75-100)

## Country Risk Database

200+ countries classified per Article 29 benchmarking:

- **HIGH RISK** (28 countries): Brazil, Indonesia, DRC, Myanmar, etc.
- **LOW RISK** (38 countries): EU-27, EEA/EFTA, select OECD
- **STANDARD RISK**: All remaining countries (default)

## Key Agents Included

### EUDR Agents (18 of 40)
- Supply chain mapping and traceability (8 agents)
- Risk assessment and scoring (3 agents)
- Due diligence orchestration (3 agents)
- Documentation and EU IS submission (4 agents)

### Data Agents (6)
- PDF extraction, Excel normalization, EUDR traceability connector
- Data quality profiling, deduplication, validation rules

### Foundation Agents (10)
- Orchestrator, schema validation, unit normalization
- Provenance tracking, reproducibility, QA test harness

## Environment Variables

Override configuration via `EUDR_PACK_*` environment variables:

```bash
export EUDR_PACK_SANDBOX_MODE=true
export EUDR_PACK_DEMO_MODE=false
export EUDR_PACK_COORDINATE_PRECISION=8
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | >= 3.11 | 3.12 |
| PostgreSQL | >= 14 | 16 |
| Redis | >= 7 | 7.2 |
| CPU | 4 cores | 8 cores |
| Memory | 16 GB | 32 GB |
| Storage | 100 GB | 500 GB |

Required extensions: pgvector, TimescaleDB, PostGIS

## License

Proprietary - GreenLang Platform Team
