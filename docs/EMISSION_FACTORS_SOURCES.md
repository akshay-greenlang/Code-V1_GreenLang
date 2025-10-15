# Emission Factors Sources Documentation

## Overview

This document provides comprehensive information about the emission factors used in the GreenLang system, their sources, standards compliance, and update procedures. All emission factors are maintained in the central registry with full URI provenance for audit compliance.

**Registry Location:** `data/emission_factors_registry.yaml` (and `.json`)

**Last Updated:** 2025-01-15

**Version:** 1.0.0

---

## Table of Contents

1. [Standards and Methodologies](#standards-and-methodologies)
2. [Data Sources](#data-sources)
3. [Emission Factor Categories](#emission-factor-categories)
4. [Data Quality and Uncertainty](#data-quality-and-uncertainty)
5. [Update Procedures](#update-procedures)
6. [Usage Guidelines](#usage-guidelines)
7. [API and Query Tools](#api-and-query-tools)
8. [Audit Compliance](#audit-compliance)

---

## Standards and Methodologies

All emission factors in the registry comply with the following international standards:

### Primary Standards

1. **GHG Protocol Corporate Standard**
   - URL: https://ghgprotocol.org/corporate-standard
   - Scope: Corporate GHG accounting and reporting
   - Application: All Scope 1, 2, and 3 emissions

2. **ISO 14064-1:2018**
   - URL: https://www.iso.org/standard/66453.html
   - Scope: Specification with guidance for quantification and reporting of GHG emissions
   - Application: Organizational-level GHG inventories

3. **IPCC 2021 Guidelines for National Greenhouse Gas Inventories**
   - URL: https://www.ipcc-nggip.iges.or.jp/public/2019rf/index.html
   - Scope: National GHG inventory methodologies
   - Application: Energy, industrial processes, agriculture, waste

4. **EPA GHG Reporting Program (40 CFR Part 98)**
   - URL: https://www.epa.gov/ghgreporting
   - Scope: US facility-level GHG reporting
   - Application: US-specific emission factors

### Global Warming Potential (GWP)

- **Basis:** IPCC Sixth Assessment Report (AR6) - 2021
- **Time Horizon:** 100 years (GWP100)
- **Key Values:**
  - CO2: 1 (by definition)
  - CH4 (fossil): 29.8
  - CH4 (biogenic): 27.2
  - N2O: 273
  - HFC-134a: 1,430
  - R-410A: 2,088

---

## Data Sources

### 1. EPA (United States Environmental Protection Agency)

#### eGRID (Emissions & Generation Resource Integrated Database)

- **URL:** https://www.epa.gov/egrid
- **Scope:** US electricity grid emission factors by region
- **Update Frequency:** Annual (typically released 18 months after data year)
- **Current Version:** eGRID 2023 (released November 2024)
- **Coverage:**
  - 26 eGRID subregions
  - Location-based Scope 2 factors
  - Includes renewable generation data

#### EPA GHG Emission Factors Hub

- **URL:** https://www.epa.gov/climateleadership/ghg-emission-factors-hub
- **Scope:** Stationary combustion, mobile combustion, fugitive emissions
- **Update Frequency:** Annual
- **Last Update:** November 2024
- **Application:** US corporate GHG inventories

#### EPA 40 CFR Part 98

- **URL:** https://www.ecfr.gov/current/title-40/chapter-I/subchapter-C/part-98
- **Scope:** Facility-level mandatory GHG reporting
- **Update Frequency:** Annual regulatory updates
- **Application:** Coal, petroleum products, chemicals

### 2. IPCC (Intergovernmental Panel on Climate Change)

#### 2021 Refinement to the 2006 Guidelines

- **URL:** https://www.ipcc-nggip.iges.or.jp/public/2019rf/index.html
- **Scope:** Comprehensive national GHG inventory guidelines
- **Coverage:**
  - Volume 1: General Guidance
  - Volume 2: Energy
  - Volume 3: Industrial Processes
  - Volume 4: Agriculture, Forestry, Land Use
  - Volume 5: Waste
- **Application:** Global default factors, fuel properties

#### IPCC Special Report on Renewable Energy Sources (SRREN)

- **URL:** https://www.ipcc.ch/report/renewable-energy-sources-and-climate-change-mitigation/
- **Scope:** Lifecycle emissions from renewable energy
- **Coverage:** Solar, wind, hydro, biomass, geothermal
- **Application:** Renewable energy lifecycle assessments

### 3. UK DEFRA (Department for Environment, Food & Rural Affairs)

#### GHG Conversion Factors for Company Reporting

- **URL:** https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024
- **Update Frequency:** Annual (June)
- **Current Version:** 2024
- **Scope:**
  - UK electricity grid
  - Business travel (air, rail, road)
  - Fuels and energy
  - Water and waste
  - Material use

### 4. IEA (International Energy Agency)

#### Global Energy Review

- **URL:** https://www.iea.org/reports/global-energy-review-2024
- **Scope:** Global energy sector emissions and trends
- **Application:** International grid factors, energy trends

#### Global Hydrogen Review

- **URL:** https://www.iea.org/reports/global-hydrogen-review-2024
- **Scope:** Hydrogen production pathways and emissions
- **Application:** Grey, blue, and green hydrogen factors

### 5. National Grid Authorities

#### India - Central Electricity Authority (CEA)

- **URL:** https://cea.nic.in/baseline-carbon-dioxide-emissions-from-power-sector/?lang=en
- **Update Frequency:** Annual (August)
- **Current Version:** 2023 baseline (released August 2024)
- **Factor:** 0.710 kg CO2e/kWh (2023)

#### China - Ministry of Ecology and Environment (MEE)

- **URL:** https://www.mee.gov.cn/ywgz/ydqhbh/wsqtkz/
- **Update Frequency:** Annual
- **Current Version:** 2023 (released December 2024)
- **Factor:** 0.554 kg CO2e/kWh (2023)

#### Japan - METI (Ministry of Economy, Trade and Industry)

- **URL:** https://www.meti.go.jp/english/policy/energy_environment/global_warming/index.html
- **Update Frequency:** Annual
- **Factor:** 0.450 kg CO2e/kWh (2023)

#### Germany - Umweltbundesamt (UBA)

- **URL:** https://www.umweltbundesamt.de/en/topics/climate-energy/renewable-energies
- **Update Frequency:** Annual
- **Factor:** 0.380 kg CO2e/kWh (2024)

#### France - RTE (Réseau de Transport d'Électricité)

- **URL:** https://www.rte-france.com/en/eco2mix/co2-emissions
- **Update Frequency:** Real-time and annual average
- **Factor:** 0.052 kg CO2e/kWh (2024)
- **Note:** Low due to nuclear dominance (65%)

#### Brazil - MCTI (Ministry of Science, Technology and Innovation)

- **URL:** https://www.gov.br/mcti/pt-br/acompanhe-o-mcti/sirene/emissoes/fatores-de-emissao
- **Update Frequency:** Annual
- **Factor:** 0.120 kg CO2e/kWh (2023)
- **Note:** Low due to hydro dominance (62%)

#### Canada - Environment and Climate Change Canada (ECCC)

- **URL:** https://www.canada.ca/en/environment-climate-change/services/climate-change/
- **Factor:** 0.130 kg CO2e/kWh (2023)
- **Note:** Varies significantly by province (Quebec: 0.001, Alberta: 0.810)

#### Australia - DCCEEW (Department of Climate Change, Energy, Environment and Water)

- **URL:** https://www.dcceew.gov.au/climate-change/publications/national-greenhouse-accounts-factors-2024
- **Update Frequency:** Annual (August)
- **Factor:** 0.660 kg CO2e/kWh (2023)

### 6. Industry-Specific Sources

#### World Steel Association

- **URL:** https://worldsteel.org/steel-topics/climate-change/
- **Standard:** ISO 14404-1, ISO 14404-2
- **Scope:** Steel production lifecycle emissions

#### International Aluminium Institute (IAI)

- **URL:** https://international-aluminium.org/statistics/greenhouse-gas-emissions-data/
- **Update Frequency:** Annual
- **Scope:** Primary and secondary aluminum production

#### Food and Agriculture Organization (FAO)

- **URL:** https://www.fao.org/energy/agrifood-chains/en/
- **Scope:** Agricultural processes, food processing
- **Application:** Pasteurization, sterilization, refrigeration

#### National Ready Mixed Concrete Association (NRMCA)

- **URL:** https://www.nrmca.org/sustainability/epd-program/
- **Standard:** ISO 14025 Environmental Product Declarations
- **Scope:** Concrete production emissions

#### International Maritime Organization (IMO)

- **URL:** https://www.imo.org/en/OurWork/Environment/Pages/Fourth-IMO-Greenhouse-Gas-Study-2020.aspx
- **Scope:** Maritime shipping emissions
- **Application:** Ocean freight factors

#### International Civil Aviation Organization (ICAO)

- **URL:** https://www.icao.int/environmental-protection/CarbonOffset/Pages/default.aspx
- **Program:** CORSIA (Carbon Offsetting and Reduction Scheme)
- **Scope:** Aviation emissions including radiative forcing

---

## Emission Factor Categories

### 1. Fuels (Scope 1)

#### Fossil Fuels

- **Natural Gas:** 0.202 kg CO2e/kWh
  - Source: EPA GHG Factors Hub
  - Standard: GHG Protocol

- **Coal (Bituminous):** 2.40 kg CO2e/kg
  - Source: EPA 40 CFR Part 98
  - Standard: EPA GHG Reporting Program

- **Diesel:** 2.68 kg CO2e/liter
  - Source: EPA GHG Factors Hub
  - Standard: GHG Protocol

- **Gasoline:** 2.31 kg CO2e/liter
  - Source: EPA GHG Factors Hub
  - Standard: GHG Protocol

#### Alternative Fuels

- **Hydrogen (Grey):** 10.5 kg CO2e/kg H2
  - Source: IEA Hydrogen Review 2024
  - Method: Steam Methane Reforming (SMR)

- **Hydrogen (Blue):** 3.5 kg CO2e/kg H2
  - Source: IEA Hydrogen Review 2024
  - Method: SMR + 85% CCS

- **Hydrogen (Green):** 0.5 kg CO2e/kg H2
  - Source: IEA Hydrogen Review 2024
  - Method: Renewable electrolysis

- **Biodiesel (B100):** 1.35 kg CO2e/liter
  - Source: EPA Renewable Fuel Standard
  - Includes lifecycle land-use change

### 2. Electricity Grids (Scope 2)

#### US Regional Grids (eGRID 2023)

| Region | Factor (kg CO2e/kWh) | Renewable % |
|--------|---------------------|-------------|
| US National | 0.385 | 21% |
| WECC California (CAMX) | 0.234 | 58% |
| WECC Northwest (NWPP) | 0.306 | 49% |
| Texas ERCOT | 0.398 | 33% |
| PJM (Mid-Atlantic) | 0.385 | 11% |
| MISO (Midwest) | 0.531 | - |
| SPP (Plains) | 0.564 | - |

#### International Grids

| Country | Factor (kg CO2e/kWh) | Year | Renewable % |
|---------|---------------------|------|-------------|
| France | 0.052 | 2024 | 28% (+ 65% nuclear) |
| Brazil | 0.120 | 2023 | 83% |
| Canada | 0.130 | 2023 | 68% |
| UK | 0.212 | 2024 | 43% |
| EU-27 | 0.230 | 2023 | 44% |
| Germany | 0.380 | 2024 | 52% |
| Japan | 0.450 | 2023 | 22% |
| South Korea | 0.490 | 2023 | 8% |
| China | 0.554 | 2023 | 36% |
| Australia | 0.660 | 2023 | 37% |
| India | 0.710 | 2023 | 23% |

### 3. Industrial Processes

#### Thermal Processes

- **Milk Pasteurization (HTST):** 0.080 kWh/kg
  - Temperature: 72°C for 15 seconds
  - Source: FAO

- **UHT Sterilization:** 0.120 kWh/kg
  - Temperature: 140°C for 4 seconds
  - Source: FAO

- **Steam Generation:** 2.95 MJ/kg steam
  - Conditions: 180°C, 10 bar
  - Efficiency: 85%
  - Source: US DOE

#### Manufacturing

- **Cement Production:** 0.865 kg CO2e/kg cement
  - 60% process emissions (calcination)
  - 40% energy emissions
  - Source: IPCC 2021

- **Steel (Blast Furnace):** 2.30 kg CO2e/kg steel
  - Primary production from iron ore
  - Source: World Steel Association

- **Steel (Electric Arc):** 0.45 kg CO2e/kg steel
  - Recycled steel (90% scrap)
  - Source: World Steel Association

- **Aluminum (Primary):** 12.5 kg CO2e/kg aluminum
  - Includes smelting energy
  - Source: IAI

- **Aluminum (Secondary):** 0.60 kg CO2e/kg aluminum
  - 95% recycled content
  - Source: IAI

### 4. Transportation (Scope 3)

- **Freight Truck (Diesel):** 0.062 kg CO2e/ton-km
- **Ocean Freight (Container):** 0.011 kg CO2e/ton-km
- **Air Freight (Long-haul):** 0.602 kg CO2e/ton-km
- **Rail Freight:** 0.022 kg CO2e/ton-km

### 5. Renewable Energy (Lifecycle)

| Technology | Lifecycle Emissions (g CO2e/kWh) |
|-----------|----------------------------------|
| Wind (Onshore) | 11 |
| Nuclear | 12 |
| Hydro (Reservoir) | 24 |
| Solar PV | 41 |
| Geothermal | 38 |
| Biomass | 230 |

Source: IPCC Special Report on Renewable Energy

---

## Data Quality and Uncertainty

### Data Quality Tiers

The registry uses a three-tier classification system:

#### Tier 1: National/Industry Averages
- **Uncertainty:** ±5-10%
- **Application:** Initial screening, low-materiality sources
- **Example:** US national grid factor (0.385 kg CO2e/kWh)

#### Tier 2: Technology/Region-Specific
- **Uncertainty:** ±3-7%
- **Application:** Material sources, location-specific reporting
- **Example:** WECC California grid factor (0.234 kg CO2e/kWh)

#### Tier 3: Facility/Process-Specific
- **Uncertainty:** ±1-5%
- **Application:** High-materiality sources, detailed inventories
- **Example:** Supplier-specific emission factors, direct measurement

### Uncertainty Ranges

| Fuel/Source | Typical Uncertainty |
|------------|-------------------|
| Natural Gas | ±4-5% |
| Diesel/Gasoline | ±4% |
| Coal | ±8% |
| Electricity Grids | ±5-10% |
| Industrial Processes | ±10-15% |
| Transportation | ±10-20% |

### Geographic Scope

- **US Factors:** EPA sources - US-specific
- **International Factors:** IPCC, IEA - Global applicability
- **Regional Factors:** National authorities - Country/region-specific

---

## Update Procedures

### Update Schedule

1. **Quarterly Review (January, April, July, October)**
   - Check for new EPA eGRID releases
   - Review IPCC updates
   - Check national grid authority updates

2. **Annual Major Update (January)**
   - Full registry update
   - URI validation
   - Standards compliance review
   - Documentation update

3. **Ad-hoc Updates**
   - Critical corrections
   - New standards releases
   - Regulatory changes

### Update Process

1. **Identification**
   - Monitor source websites for updates
   - Track regulatory announcements
   - Review industry publications

2. **Validation**
   - Verify new factor values
   - Check URI accessibility
   - Compare with previous values
   - Document significant changes

3. **Implementation**
   - Update YAML registry
   - Regenerate JSON version
   - Update documentation
   - Run validation tests

4. **Communication**
   - Changelog documentation
   - User notifications
   - Agent compatibility testing

### Change Log Format

```yaml
changes:
  - date: "2025-01-15"
    category: "grids"
    item: "US_WECC_CA"
    field: "emission_factor_kg_co2e_per_kwh"
    old_value: 0.247
    new_value: 0.234
    reason: "EPA eGRID 2023 update"
    uri: "https://www.epa.gov/egrid/download-data"
```

---

## Usage Guidelines

### For Agent Developers

#### Basic Usage

```python
from scripts.query_emission_factors import EmissionFactorRegistry

# Initialize registry
registry = EmissionFactorRegistry()

# Get fuel factor
factor = registry.get_fuel_factor("natural_gas", unit="kwh")
print(f"Natural Gas: {factor['emission_factor']} kg CO2e/kWh")
print(f"Source: {factor['source']}")
print(f"URI: {factor['uri']}")

# Get grid factor
grid = registry.get_grid_factor("US_WECC_CA")
print(f"California Grid: {grid['emission_factor_kwh']} kg CO2e/kWh")
```

#### Search and Discovery

```python
# Search for factors
results = registry.search("coal")
for result in results:
    print(f"{result['name']}: {result['uri']}")

# List all categories
categories = registry.list_categories()
print(f"Available fuels: {categories['fuels']}")
```

#### URI Validation

```python
# Validate specific factor
factor = registry.get_fuel_factor("diesel", validate_uri=True)
if factor['uri_valid']:
    print("URI is accessible")

# Validate all URIs
validation = registry.validate_all_uris()
print(f"Valid: {validation['valid']}/{validation['total']}")
```

### For Reporting and Audits

#### Export Audit Report

```python
# Export CSV report
registry.export_audit_report("audit_report.csv")

# Export JSON report
registry.export_audit_report("audit_report.json")
```

#### Command-Line Usage

```bash
# Get fuel factor
python scripts/query_emission_factors.py get-fuel --fuel natural_gas --unit kwh

# Get grid factor
python scripts/query_emission_factors.py get-grid --grid US_WECC_CA

# Search
python scripts/query_emission_factors.py search --query "coal"

# List all categories
python scripts/query_emission_factors.py list

# Validate all URIs
python scripts/query_emission_factors.py validate-uris

# Export audit report
python scripts/query_emission_factors.py export-audit --output audit.csv
```

### Best Practices

1. **Geographic Specificity**
   - Use most geographically specific factor available
   - Example: Use "US_WECC_CA" instead of "US_NATIONAL" for California

2. **Temporal Consistency**
   - Use factors from the same year when possible
   - Document reporting year in outputs

3. **Scope Alignment**
   - Ensure factor scope matches reporting requirements
   - Location-based vs. market-based for Scope 2

4. **Data Quality**
   - Prioritize Tier 3 > Tier 2 > Tier 1
   - Document tier used in reports

5. **URI Provenance**
   - Always include source URI in outputs
   - Validate URIs before audit submissions

6. **Uncertainty**
   - Consider uncertainty ranges in sensitivity analysis
   - Document data quality limitations

---

## API and Query Tools

### EmissionFactorRegistry Class

The `EmissionFactorRegistry` class provides programmatic access to all emission factors.

#### Key Methods

1. **`get_fuel_factor(fuel_type, unit, validate_uri)`**
   - Returns fuel emission factor with metadata
   - Supports multiple units per fuel

2. **`get_grid_factor(grid_region, validate_uri)`**
   - Returns electricity grid emission factor
   - Includes renewable share and regional info

3. **`get_process_factor(process_name, validate_uri)`**
   - Returns industrial process emissions
   - Includes energy intensity and parameters

4. **`search(query, category)`**
   - Searches across all categories
   - Case-insensitive keyword matching

5. **`list_categories()`**
   - Lists all available factors by category

6. **`validate_all_uris()`**
   - Validates accessibility of all source URIs
   - Returns validation report

7. **`export_audit_report(output_path)`**
   - Exports audit-ready report (CSV or JSON)

### Integration Examples

#### Agent Integration

```python
class EmissionCalculationAgent:
    def __init__(self):
        self.registry = EmissionFactorRegistry()

    def calculate_fuel_emissions(self, fuel_type, quantity, unit):
        factor = self.registry.get_fuel_factor(fuel_type, unit)
        emissions = quantity * factor['emission_factor']

        return {
            'emissions_kg_co2e': emissions,
            'source': factor['source'],
            'uri': factor['uri'],
            'data_quality': factor['data_quality']
        }
```

#### Workflow Integration

```yaml
# In GreenLang workflow
steps:
  - name: get_emission_factor
    agent_id: emission_factor_lookup
    inputs:
      fuel_type: "natural_gas"
      unit: "kwh"

  - name: calculate_emissions
    agent_id: emission_calculator
    inputs:
      quantity: 10000
      emission_factor: results.get_emission_factor.emission_factor
```

---

## Audit Compliance

### Documentation Requirements

For audit compliance, the following must be documented:

1. **Emission Factor Details**
   - Value and unit
   - Source organization
   - Source URI (accessible)
   - Publication date
   - Standard/methodology

2. **Data Quality**
   - Quality tier (1, 2, or 3)
   - Uncertainty range
   - Geographic scope
   - Temporal scope

3. **Calculation Method**
   - Formula used
   - Activity data source
   - Emission factor source
   - Any adjustments or assumptions

### Audit Trail

The registry provides built-in audit trail support:

```python
# Generate audit report with full provenance
registry = EmissionFactorRegistry()
audit_data = registry.export_audit_report("audit_trail.json")

# Each factor includes:
# - Source URI
# - Last updated date
# - Standard compliance
# - Data quality tier
# - Uncertainty range
```

### Compliance Checklist

- [ ] All emission factors have source URIs
- [ ] All URIs are accessible and verified
- [ ] Standards compliance documented (GHG Protocol, ISO 14064, etc.)
- [ ] Data quality tiers assigned
- [ ] Update dates tracked
- [ ] Uncertainty ranges documented
- [ ] Geographic and temporal scope clear
- [ ] Calculation methods documented
- [ ] Audit reports generated and archived

---

## Frequently Asked Questions

### Q: How often should I update emission factors?

**A:** Update annually at minimum. For material sources, update quarterly or when authoritative sources release new data (e.g., EPA eGRID).

### Q: Should I use location-based or market-based Scope 2 factors?

**A:**
- **Location-based:** Use grid factors from this registry
- **Market-based:** Use supplier-specific factors or renewable energy certificate (REC) attributes
- **Best Practice:** Report both per GHG Protocol Scope 2 Guidance

### Q: What if a URI becomes inaccessible?

**A:**
1. Check for domain changes or page moves
2. Use Internet Archive (Wayback Machine) for historical access
3. Contact source organization
4. Document in registry with archived URI

### Q: How do I choose between multiple fuel types?

**A:** Use the most specific match:
- Coal → Bituminous, Subbituminous, or Anthracite based on actual fuel
- Natural Gas → LNG if liquefied, otherwise pipeline natural gas

### Q: Can I use these factors for carbon credits?

**A:** These factors are suitable for corporate GHG inventories and CDP reporting. For carbon offset projects or voluntary markets (VCS, Gold Standard), project-specific factors may be required.

### Q: What about Scope 3 emissions?

**A:** The registry includes Scope 3 factors for:
- Transportation (freight, business travel)
- Water and waste
- Upstream/downstream processes
- Refer to GHG Protocol Scope 3 Standard for calculation guidance

---

## Contact and Support

### Registry Maintenance

- **Owner:** GreenLang Data Team
- **Email:** data@greenlang.io
- **GitHub:** https://github.com/greenlang/emissions-registry

### Report Issues

- **Bug Reports:** GitHub Issues
- **Data Corrections:** Email with source documentation
- **New Factor Requests:** Submit via GitHub with source URIs

### Contributing

Contributions welcome! Please:
1. Provide authoritative source URI
2. Document standard compliance
3. Include data quality assessment
4. Follow registry YAML structure

---

## References

1. GHG Protocol. (2004). *Corporate Accounting and Reporting Standard*. World Resources Institute.

2. IPCC. (2021). *2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories*. IPCC, Switzerland.

3. ISO. (2018). *ISO 14064-1:2018 - Greenhouse gases — Part 1: Specification with guidance at the organization level for quantification and reporting of greenhouse gas emissions and removals*. International Organization for Standardization.

4. US EPA. (2024). *Emission Factors for Greenhouse Gas Inventories*. Environmental Protection Agency.

5. UK DEFRA. (2024). *Greenhouse Gas Reporting: Conversion Factors 2024*. Department for Environment, Food & Rural Affairs.

6. IEA. (2024). *Global Energy Review 2024*. International Energy Agency.

7. IPCC. (2011). *Special Report on Renewable Energy Sources and Climate Change Mitigation*. Cambridge University Press.

---

**Document Version:** 1.0.0
**Last Updated:** 2025-01-15
**Next Review:** 2025-04-15
