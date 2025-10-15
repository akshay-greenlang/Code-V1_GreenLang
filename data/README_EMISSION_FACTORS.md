# Emission Factors Registry - Quick Reference

## Overview

Comprehensive emission factors database with URIs for provenance and audit compliance.

- **Registry Files:**
  - `emission_factors_registry.yaml` (source of truth)
  - `emission_factors_registry.json` (generated from YAML)

- **Query Tool:** `scripts/query_emission_factors.py`
- **Documentation:** `docs/EMISSION_FACTORS_SOURCES.md`

## Registry Statistics

- **18 Fuel Types** (Scope 1)
  - Fossil fuels (natural gas, coal, diesel, gasoline, etc.)
  - Alternative fuels (hydrogen grey/blue/green, biodiesel, ethanol)

- **21 Electricity Grids** (Scope 2)
  - 10 US regional grids (eGRID subregions)
  - 11 international grids (UK, EU, China, India, Japan, etc.)

- **19 Industrial Processes** (Scope 1 & 3)
  - Thermal processes (pasteurization, sterilization, steam)
  - Manufacturing (cement, steel, aluminum)
  - Refrigeration (HFC leakage)
  - Transportation (freight, air, ocean)
  - Waste treatment
  - Agriculture

- **Additional Categories:**
  - District energy (heating/cooling)
  - Renewable generation (lifecycle emissions)
  - Water usage
  - Business travel

## Quick Start

### Python API

```python
from scripts.query_emission_factors import EmissionFactorRegistry

# Initialize
registry = EmissionFactorRegistry()

# Get fuel factor
factor = registry.get_fuel_factor("natural_gas", unit="kwh")
print(f"{factor['emission_factor']} kg CO2e/kWh")
print(f"Source: {factor['uri']}")

# Get grid factor
grid = registry.get_grid_factor("US_WECC_CA")
print(f"{grid['emission_factor_kwh']} kg CO2e/kWh")

# Search
results = registry.search("coal")
for r in results:
    print(f"{r['name']}: {r['uri']}")
```

### Command Line

```bash
# List all available factors
python scripts/query_emission_factors.py list

# Get specific fuel factor
python scripts/query_emission_factors.py get-fuel --fuel natural_gas --unit kwh

# Get grid factor
python scripts/query_emission_factors.py get-grid --grid US_WECC_CA

# Search
python scripts/query_emission_factors.py search --query "coal"

# Export audit report
python scripts/query_emission_factors.py export-audit --output audit.csv
```

## Key Features

### 1. Complete Provenance
Every emission factor includes:
- Source URI (verified and accessible)
- Source organization
- Publication/update date
- Standard compliance (GHG Protocol, ISO 14064, IPCC)
- Data quality tier

### 2. Multiple Units
Fuels support multiple units:
```yaml
natural_gas:
  emission_factor_kg_co2e_per_kwh: 0.202
  emission_factor_kg_co2e_per_m3: 1.89
  emission_factor_kg_co2e_per_mmbtu: 53.06
  emission_factor_kg_co2e_per_therm: 5.30
```

### 3. Geographic Coverage
- **US:** National + 10 regional grids (eGRID subregions)
- **International:** UK, EU, Germany, France, China, India, Japan, Brazil, South Korea, Canada, Australia

### 4. Standards Compliance
- GHG Protocol Corporate Standard
- ISO 14064-1:2018
- IPCC 2021 Guidelines
- EPA GHG Reporting Program (40 CFR Part 98)

### 5. Audit-Ready
- All URIs verified
- Data quality tiers documented
- Uncertainty ranges provided
- Update dates tracked

## Data Sources

### Primary Sources

1. **EPA eGRID** (US electricity)
   - https://www.epa.gov/egrid
   - Annual updates
   - Current: eGRID 2023

2. **EPA GHG Factors Hub** (US fuels)
   - https://www.epa.gov/climateleadership/ghg-emission-factors-hub
   - Annual updates

3. **IPCC 2021 Guidelines**
   - https://www.ipcc-nggip.iges.or.jp/
   - Industrial processes, agriculture, waste

4. **UK DEFRA** (UK factors, business travel)
   - https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024
   - Annual updates (June)

5. **National Grid Authorities**
   - India CEA, China MEE, Japan METI, etc.
   - Country-specific updates

## Example Emission Factors

### Fuels (kg CO2e)

| Fuel | Per kWh | Per Liter | Per kg |
|------|---------|-----------|--------|
| Natural Gas | 0.202 | - | - |
| Diesel | - | 2.68 | 3.16 |
| Coal (Bituminous) | - | - | 2.40 |
| Hydrogen (Grey) | 0.315 | - | 10.5 |
| Hydrogen (Green) | 0.015 | - | 0.5 |

### Grids (kg CO2e/kWh)

| Region | Factor | Renewable % |
|--------|--------|-------------|
| France | 0.052 | 28% (+ 65% nuclear) |
| Brazil | 0.120 | 83% |
| UK | 0.212 | 43% |
| US California | 0.234 | 58% |
| US National | 0.385 | 21% |
| China | 0.554 | 36% |
| India | 0.710 | 23% |

### Processes

- **Pasteurization:** 0.080 kWh/kg (72°C for 15s)
- **Cement Production:** 0.865 kg CO2e/kg cement
- **Steel (Primary):** 2.30 kg CO2e/kg
- **Steel (Recycled):** 0.45 kg CO2e/kg
- **Aluminum (Primary):** 12.5 kg CO2e/kg

## Integration with Agents

The registry is designed for seamless integration with all 84+ GreenLang agents:

```python
# In your agent
class CarbonCalculationAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.emission_registry = EmissionFactorRegistry()

    def calculate_fuel_emissions(self, fuel_type, quantity, unit):
        factor = self.emission_registry.get_fuel_factor(fuel_type, unit)

        return {
            'emissions_kg_co2e': quantity * factor['emission_factor'],
            'source': factor['source'],
            'uri': factor['uri'],
            'scope': factor['scope'],
            'data_quality': factor['data_quality']
        }
```

## Update Schedule

- **Quarterly Review:** Check for new data releases
- **Annual Major Update:** January (full registry update)
- **Ad-hoc Updates:** Critical corrections and new standards

## Support

- **Documentation:** `docs/EMISSION_FACTORS_SOURCES.md` (24KB comprehensive guide)
- **Issues:** Report via GitHub
- **Updates:** Subscribe to data team notifications

## File Structure

```
data/
├── emission_factors_registry.yaml      # 34KB - Source of truth
├── emission_factors_registry.json      # 38KB - Auto-generated
└── README_EMISSION_FACTORS.md          # This file

scripts/
└── query_emission_factors.py           # 19KB - Query utility

docs/
└── EMISSION_FACTORS_SOURCES.md         # 24KB - Full documentation
```

## License and Attribution

All emission factors are sourced from public domain or open access sources:
- EPA (Public Domain - US Government)
- IPCC (Open Access - CC BY)
- National authorities (Public data)

When using in publications or reports, cite original sources using the URIs provided.

---

**Version:** 1.0.0
**Last Updated:** 2025-01-15
**Next Review:** 2025-04-15
