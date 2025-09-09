# Global Emission Factors Dataset Card

## Overview
- **Name**: emission_factors
- **Version**: 2025.1
- **Type**: Dataset
- **Format**: CSV, JSON
- **License**: CC BY 4.0
- **Author**: GreenLang Data Team
- **Created**: 2024-01-01
- **Updated**: 2025-01-01

## Description
Comprehensive dataset of emission factors for electricity grids, fuels, and industrial processes worldwide. This dataset provides the fundamental coefficients needed to convert energy consumption and activity data into greenhouse gas emissions (CO₂, CH₄, N₂O) for carbon accounting, LCA studies, and climate impact assessments.

## Coverage

### Geographic Scope
- **Countries**: 195+ countries and territories
- **Grid Regions**: 300+ electricity grid zones and balancing areas
- **Subnational**: Major provinces/states with distinct energy profiles
- **Update Frequency**: Annual, with quarterly updates for major grids

### Emission Sources
- **Electricity**: Grid emission factors by region and time
- **Fuels**: Natural gas, coal, oil products, biomass, hydrogen
- **Transportation**: Road, rail, marine, aviation by vehicle type
- **Industrial Processes**: Cement, steel, aluminum, chemicals
- **Refrigerants**: GWP factors for HFCs, CFCs, natural refrigerants
- **Land Use**: Forestry, agriculture, soil carbon factors

### Greenhouse Gases
- **CO₂**: Carbon dioxide (primary focus)
- **CH₄**: Methane (28x GWP over 100 years)
- **N₂O**: Nitrous oxide (265x GWP over 100 years)
- **F-gases**: HFCs, PFCs, SF₆ with GWP factors

## Data Sources

### Primary Sources
- **Grid Electricity**: National energy agencies, grid operators (ISO, RTO)
- **Fuels**: IEA, EPA, IPCC 2019 Refinement
- **Transport**: ICAO, IMO, national transport agencies
- **Industrial**: Industry associations, peer-reviewed studies
- **Carbon Intensity**: Real-time APIs where available

### Data Vintage
- **Electricity**: 2024 data (latest available)
- **Fuels**: 2023-2024 heating values and emission factors
- **GWP Values**: IPCC AR6 (2021) with 100-year timeframe
- **Update Policy**: Annual refresh, emergency updates for major changes

## File Structure

```
emission_factors/
├── grid_electricity/
│   ├── grid_factors_2025.csv          # Main grid emission factors
│   ├── hourly_factors_major_grids.csv # Time-varying factors
│   └── renewable_energy_certificates.csv
├── fuels/
│   ├── fossil_fuels.csv               # Coal, oil, gas
│   ├── renewable_fuels.csv            # Biomass, hydrogen, synthetic
│   └── heating_values.csv             # Energy content (LHV/HHV)
├── transport/
│   ├── road_transport.csv             # Cars, trucks, buses
│   ├── aviation.csv                   # Domestic/international flights
│   ├── marine.csv                     # Shipping emission factors
│   └── rail.csv                       # Passenger and freight rail
├── industrial/
│   ├── process_emissions.csv          # Non-energy industrial emissions
│   ├── cement.csv                     # Cement production factors
│   └── steel.csv                      # Steel production factors
└── metadata/
    ├── data_quality.csv               # Quality scores and uncertainty
    ├── sources.csv                    # Data provenance
    └── changelog.csv                  # Version history
```

## Key Fields

### Grid Electricity (grid_factors_2025.csv)
- **region_code**: ISO 3166-1 alpha-3 country code + grid zone
- **grid_name**: Human-readable grid operator name
- **co2_kg_mwh**: Grid average CO₂ emission factor (kg CO₂/MWh)
- **co2eq_kg_mwh**: Full GHG factor including CH₄, N₂O (kg CO₂eq/MWh)
- **renewable_fraction**: Share of renewable generation (0-1)
- **data_year**: Year of underlying generation data
- **data_quality**: Quality score (A=high, B=medium, C=low, D=estimated)
- **marginal_factor**: Marginal emission factor for demand response
- **uncertainty_pct**: Uncertainty range as percentage (±%)

### Fuels (fossil_fuels.csv, renewable_fuels.csv)
- **fuel_type**: Standardized fuel name (e.g., "natural_gas", "diesel")
- **co2_kg_gj**: CO₂ emission factor (kg CO₂/GJ)
- **ch4_g_gj**: CH₄ emission factor (g CH₄/GJ)
- **n2o_g_gj**: N₂O emission factor (g N₂O/GJ)
- **co2eq_kg_gj**: Total GHG factor with GWP (kg CO₂eq/GJ)
- **heating_value_lhv**: Lower heating value (MJ/kg or MJ/m³)
- **heating_value_hhv**: Higher heating value (MJ/kg or MJ/m³)
- **density_kg_m3**: Density for volume conversions (kg/m³)
- **carbon_content_pct**: Carbon content by mass (%)
- **biogenic_fraction**: Share of biogenic carbon (0-1)

## Usage Examples

### Python Integration
```python
import pandas as pd
from pathlib import Path

# Load grid emission factors
grid_ef = pd.read_csv('datasets/emission_factors/grid_electricity/grid_factors_2025.csv')

# Get emission factor for California grid
ca_factor = grid_ef[grid_ef['region_code'] == 'US-CA']['co2eq_kg_mwh'].iloc[0]
print(f"California grid: {ca_factor:.1f} kg CO₂eq/MWh")

# Calculate emissions from electricity consumption
kwh_consumed = 1000  # kWh
emissions_kg = kwh_consumed * ca_factor / 1000  # Convert MWh to kWh
print(f"Emissions: {emissions_kg:.1f} kg CO₂eq")
```

### CLI Usage
```bash
# Search for emission factors
gl data query emission_factors --region US-TX --fuel "natural_gas"

# Convert units
gl data convert --from "1000 kWh" --to "kg_co2eq" --region "DE"

# Validate data quality
gl data validate emission_factors --check-completeness --check-consistency
```

## Data Quality

### Quality Scores
- **A (High)**: Direct measurement, official government data, <5% uncertainty
- **B (Medium)**: Industry average, peer-reviewed studies, 5-15% uncertainty  
- **C (Low)**: Regional proxy, older data (>3 years), 15-30% uncertainty
- **D (Estimated)**: Modeled values, significant assumptions, >30% uncertainty

### Completeness
- **Electricity**: 98% coverage by generation volume
- **Fuels**: 100% coverage for major fossil fuels
- **Transport**: 95% coverage by transport mode
- **Industrial**: 80% coverage by industrial output

### Validation
- **Cross-validation**: Multiple sources compared where available
- **Trend Analysis**: Year-over-year consistency checks
- **Outlier Detection**: Statistical methods to identify anomalies
- **Expert Review**: Annual review by domain experts

## Limitations

### Temporal Resolution
- Annual average factors (not hourly/seasonal variation)
- Some data may be 1-2 years behind current year
- Future projections not included (separate dataset)

### Geographic Resolution
- Country-level for smaller nations (no subnational data)
- Grid zones may not match actual electricity delivery
- Remote areas may use regional proxies

### Scope Boundaries
- Lifecycle emissions included where available
- Upstream emissions (fuel extraction, transport) vary by source
- Land use change impacts not systematically included
- Indirect effects (e.g., grid interactions) not captured

## Technical Specifications

### File Formats
- **Primary**: CSV with UTF-8 encoding
- **Alternative**: JSON, Parquet for large datasets
- **Compression**: gzip compression for bandwidth efficiency

### Units and Conventions
- **Emissions**: kg CO₂eq (carbon dioxide equivalent)
- **Energy**: GJ (gigajoules), MWh (megawatt-hours)
- **Mass**: kg (kilograms), tonnes (1000 kg)
- **GWP**: IPCC AR6 100-year values
- **Precision**: 3 significant figures typical

### API Access
- **REST API**: https://api.greenlang.io/emission-factors/v1/
- **Authentication**: API key required for bulk access
- **Rate Limits**: 1000 requests/hour (free tier)
- **GraphQL**: Advanced querying capabilities

## Changelog

### v2025.1 (2025-01-01)
- Updated all grid factors with 2024 generation data
- Added marginal emission factors for 50 major grids
- Improved data quality scores and uncertainty estimates
- Added hydrogen emission factors (green, blue, grey)

### v2024.4 (2024-10-01)
- Quarterly update for major electricity grids
- Added aviation sustainable fuel emission factors
- Updated GWP factors to IPCC AR6 values
- Improved geographic coverage in Asia-Pacific

### v2024.1 (2024-01-01)
- Annual release with 2023 electricity data
- Expanded industrial process coverage
- Added uncertainty quantification
- New transport mode: electric vehicles by region

## Citation
```bibtex
@dataset{emission_factors_2025,
  title = {Global Emission Factors Dataset},
  author = {GreenLang Data Team},
  year = {2025},
  version = {2025.1},
  publisher = {GreenLang},
  doi = {10.5281/zenodo.emission-factors.2025.1},
  url = {https://datasets.greenlang.io/emission-factors}
}
```

## License & Attribution
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Attribution Required**: Cite this dataset card and version
- **Commercial Use**: Permitted with attribution
- **Distribution**: Permitted with attribution and license preservation

## Support & Contact
- **Documentation**: https://docs.greenlang.io/datasets/emission-factors
- **Data Issues**: https://github.com/greenlang/datasets/issues
- **Email**: data-team@greenlang.io
- **Updates**: Subscribe to data-updates@greenlang.io

## Compliance & Standards
- **GHG Protocol**: Corporate and Product Life Cycle Standards
- **ISO 14064**: Greenhouse Gas Accounting and Verification
- **IPCC Guidelines**: 2006 IPCC Guidelines with 2019 Refinement
- **Science-Based Targets**: SBTi methodology alignment
- **EU Taxonomy**: Alignment with EU environmental objectives