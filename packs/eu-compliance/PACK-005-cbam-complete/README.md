# PACK-005: CBAM Complete Pack

**Enterprise-grade EU Carbon Border Adjustment Mechanism compliance solution powered by GreenLang AI agents**

## Overview

PACK-005 extends PACK-004 (CBAM Readiness) to deliver the complete CBAM compliance platform for enterprise importers, customs brokers, and industrial groups. It adds certificate trading optimization, multi-entity group management, CBAM registry API integration, advanced Monte Carlo analytics, customs automation, cross-regulation alignment, NCA audit management, and deep precursor chain analysis.

| Metric | PACK-004 (Readiness) | PACK-005 (Complete) |
|--------|---------------------|---------------------|
| Engines | 7 | 15 (7 inherited + 8 new) |
| Workflows | 7 | 13 (7 inherited + 6 new) |
| Templates | 8 | 14 (8 inherited + 6 new) |
| Integrations | 6 | 13 (6 inherited + 7 new) |
| Presets | 9 | 7 (4 pack + 3 sector) |
| CN Codes | 50+ | 167 (full Annex I) |
| Compliance Rules | 187 | 187 + 94 cross-regulation mappings |
| Goods Calculators | 3 (steel, cement, aluminium) | 6 (+ fertilizer, electricity, hydrogen) |
| Precursor Depth | 1-2 tiers | Up to 5 tiers |
| Entity Support | Single entity | Multi-entity group |

## What PACK-005 Adds Over PACK-004

### 8 New Engines

1. **Certificate Trading** -- Buying strategy optimization (DCA, price-triggered, budget-paced), FIFO/WAVG valuation, portfolio management
2. **Entity Group** -- Multi-entity hierarchy, delegated compliance, cost allocation, consolidated reporting
3. **Registry Integration** -- Bidirectional CBAM registry API with mTLS, sandbox mode, retry logic
4. **Advanced Analytics** -- Monte Carlo simulation, scenario analysis, supplier portfolio optimization
5. **Customs Automation** -- TARIC validation, SAD parsing, AEO checks, anti-circumvention detection
6. **Cross-Regulation** -- Maps CBAM data to CSRD, CDP, SBTi, EU Taxonomy, EU ETS, EUDR
7. **Audit Management** -- NCA readiness, evidence packaging, encrypted data rooms, penalty tracking
8. **Precursor Chain** -- Deep precursor chain analysis (5 tiers), mass/economic/energy allocation

### 3 Additional Calculators

- **Fertilizer Calculator** -- Ammonia, urea, nitric acid, ammonium nitrate, NPK
- **Electricity Calculator** -- Grid average, interconnector, PPA-based
- **Hydrogen Calculator** -- Grey, blue, green, turquoise, pink production routes

### 6 New Workflows

- Certificate Trading Cycle
- Group Consolidation
- Registry Synchronization
- Anti-Circumvention Monitoring
- Cross-Regulation Data Sync
- NCA Audit Preparation

## Quick Start

```python
from packs.eu_compliance.pack_005_cbam_complete.config import (
    CBAMCompleteConfig,
    load_preset,
)

# 1. Load enterprise importer preset
config = CBAMCompleteConfig.from_preset("enterprise_importer")

# 2. Or load from YAML
config = CBAMCompleteConfig.from_yaml("config/presets/steel_group.yaml")

# 3. Validate configuration
issues = config.validate_config()
if not issues:
    print("Configuration is valid")

# 4. Check summary
print(config.summary())

# 5. Compute provenance hash
print(f"Config hash: {config.compute_provenance_hash()[:16]}...")
```

## Presets

### Pack-Level Presets

| Preset | Use Case | Key Features |
|--------|----------|--------------|
| `enterprise_importer` | Large multi-entity group | All features, 3+ entities, full analytics |
| `customs_broker` | Delegated compliance | Multi-client, simplified configs, consolidated billing |
| `steel_group` | Steel industry focus | Deep precursor chains (5 tiers), BF-BOF/EAF optimization |
| `multi_commodity_group` | Diversified importer | All 6 categories, commodity-level cost allocation |

### Sector Presets

| Sector | Goods Focus | Cross-Regulation |
|--------|-------------|------------------|
| `automotive_oem` | Steel + aluminium | SBTi, CSRD E1 |
| `construction` | Cement + steel | EU Taxonomy, local NCA |
| `chemical_manufacturing` | Fertilizers + hydrogen | EUDR, CDP |

## Pack Structure

```
PACK-005-cbam-complete/
  pack.yaml                              # Pack manifest (1800+ lines)
  README.md                              # This file
  config/
    __init__.py                          # Configuration exports
    pack_config.py                       # Pydantic config models (2500+ lines)
    presets/
      enterprise_importer.yaml           # Multi-entity group preset
      customs_broker.yaml                # Delegated compliance preset
      steel_group.yaml                   # Steel-focused preset
      multi_commodity_group.yaml         # Diversified importer preset
    sectors/
      automotive_oem.yaml                # Automotive sector config
      construction.yaml                  # Construction sector config
      chemical_manufacturing.yaml        # Chemical sector config
    demo/
      demo_config.yaml                   # Demo configuration
      demo_group_structure.json          # 3-entity group hierarchy
      demo_import_portfolio.csv          # 500-row import dataset
```

## Dependencies

### Required

- **PACK-004-cbam-readiness >= 1.0.0** -- Base CBAM pack (7 engines, 7 workflows, core config)

### Optional (Cross-Regulation)

- PACK-001-csrd-starter >= 1.0.0
- PACK-002-csrd-professional >= 1.0.0
- PACK-003-csrd-enterprise >= 1.0.0

### Runtime

- Python >= 3.11
- PostgreSQL >= 16
- Redis >= 7
- pydantic >= 2.0
- scipy >= 1.11 (Monte Carlo and optimization)
- cryptography >= 41.0 (evidence encryption)

## Configuration Reference

All PACK-005 configuration is managed through `CBAMCompleteConfig`, which extends PACK-004's `CBAMPackConfig` with 8 new sub-config blocks:

- `CertificateTradingConfig` -- Buying strategies, price alerts, valuation
- `EntityGroupConfig` -- Multi-entity hierarchy, cost allocation
- `RegistryAPIConfig` -- Registry connection, polling, certificates
- `AdvancedAnalyticsConfig` -- Monte Carlo, optimization solver
- `CustomsAutomationConfig` -- TARIC, SAD, anti-circumvention rules
- `CrossRegulationConfig` -- Framework toggles, sync frequency
- `AuditManagementConfig` -- Retention, data rooms, NCA deadlines
- `PrecursorChainConfig` -- Chain depth, allocation methods, fallback

## Anti-Circumvention Monitoring

PACK-005 includes Article 27 anti-circumvention detection covering:

- **Origin Change** -- Detects goods re-routed through non-CBAM countries
- **CN Reclassification** -- Flags suspicious tariff heading changes
- **Scrap Ratio** -- Monitors unusual scrap-to-primary ratios in EAF production
- **Restructuring** -- Detects supply chain restructuring to avoid CBAM
- **Minor Processing** -- Identifies minimal transformation to change classification

## Development

```bash
# Load and inspect demo
python -c "
from config.pack_config import CBAMCompleteConfig
c = CBAMCompleteConfig.from_demo()
print(c.summary())
"

# Run tests
pytest tests/packs/eu_compliance/pack_005_cbam_complete/ -v
```
