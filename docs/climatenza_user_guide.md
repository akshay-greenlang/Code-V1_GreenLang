# Climatenza AI User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [CLI Commands](#cli-commands)
4. [SDK Usage](#sdk-usage)
5. [Configuration](#configuration)
6. [Agents](#agents)
7. [Workflows](#workflows)
8. [Examples](#examples)
9. [API Reference](#api-reference)

## Introduction

Climatenza AI is a comprehensive solar thermal feasibility analysis application built on the GreenLang Climate Intelligence Framework. It provides industrial facilities with data-driven insights for solar thermal system deployment.

### Key Features
- **8760-hour simulation**: Hourly energy balance calculations for full-year analysis
- **Modular architecture**: Reusable agents for different calculation components
- **Data validation**: Robust Pydantic schemas ensure data integrity
- **Multiple output formats**: JSON, YAML, HTML reports
- **SDK integration**: Python API for programmatic access

## Quick Start

### Installation
Climatenza AI is included with GreenLang. Ensure you have GreenLang installed:

```bash
pip install -r requirements.txt
```

### Basic Usage

#### CLI
```bash
# Run with default example
greenlang climatenza

# Run with custom site configuration
greenlang climatenza --site my_site.yaml --output report.json

# Generate HTML report
greenlang climatenza --site my_site.yaml --output report.html --format html
```

#### Python SDK
```python
from greenlang.sdk import GreenLangClient

client = GreenLangClient()
result = client.run_solar_feasibility("climatenza_app/examples/dairy_hotwater_site.yaml")

print(f"Solar Fraction: {result['data']['solar_fraction']:.1%}")
print(f"Required Area: {result['data']['required_aperture_area_m2']:,.0f} m²")
```

## CLI Commands

### `gl climatenza`
Run a complete solar thermal feasibility analysis.

**Options:**
- `--site PATH`: Path to site configuration YAML file
- `--output PATH`: Output file path for results
- `--format [json|yaml|html]`: Output format (default: json)

**Examples:**
```bash
# Use default dairy plant example
greenlang climatenza

# Custom site with JSON output
greenlang climatenza --site industrial_site.yaml --output results.json

# Generate HTML report
greenlang climatenza --site site.yaml --output report.html --format html
```

## SDK Usage

### Basic Example
```python
from greenlang.sdk import GreenLangClient

# Initialize client
client = GreenLangClient()

# Run complete feasibility analysis
result = client.run_solar_feasibility("path/to/site_config.yaml")

# Access results
if result["success"]:
    data = result["data"]
    print(f"Solar Fraction: {data['solar_fraction']:.1%}")
    print(f"Annual Solar Yield: {data['total_solar_yield_gwh']:.3f} GWh")
    print(f"Collectors Required: {data['num_collectors']}")
```

### Advanced Usage
```python
# Get solar resource for specific location
solar_data = client.get_solar_resource(lat=16.506, lon=80.648)

# Calculate solar field size
field_size = client.calculate_solar_field_size(
    annual_demand_gwh=1.5,
    solar_config={
        "tech": "ASC",
        "orientation": "N-S",
        "row_spacing_factor": 2.2,
        "tracking": "1-axis"
    }
)

# Run energy balance simulation
simulation = client.simulate_energy_balance(
    solar_data=solar_json,
    load_data=load_json,
    aperture_area=800
)
```

## Configuration

### Site Configuration Schema

```yaml
# Site Information
site:
  name: "Facility Name"
  country: "IN"  # ISO country code
  lat: 16.506    # Latitude
  lon: 80.648    # Longitude
  tz: "Asia/Kolkata"
  elevation_m: 23
  land_area_m2: 50000
  roof_area_m2: 15000
  ambient_dust_level: "high"  # low, med, high

# Process Requirements
process_demand:
  medium: "hot_water"  # hot_water or steam
  temp_in_C: 60
  temp_out_C: 85
  flow_profile: "path/to/hourly_load.csv"
  schedule:
    workdays: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    shutdown_days: ["2024-12-25"]

# Existing System
boiler:
  type: "NG"  # NG, HSD, coal, biomass
  rated_steam_tph: 10
  efficiency_pct: 84

# Solar Configuration
solar_config:
  tech: "ASC"  # ASC or T160
  orientation: "N-S"  # N-S or E-W
  row_spacing_factor: 2.2
  tracking: "1-axis"  # 1-axis or none

# Financial Parameters
finance:
  currency: "INR"
  discount_rate_pct: 10
  capex_breakdown:
    collector: 32000
    bos: 18000
    epc: 8000
  opex_pct_of_capex: 2
  tariff_fuel_per_kWh: 7.5
  tariff_elec_per_kWh: 9.0
  escalation_fuel_pct: 4
  escalation_elec_pct: 3

# Technical Assumptions
assumptions:
  cleaning_cycle_days: 7
  soiling_loss_pct: 5
  availability_pct: 96
  parasitic_kWh_per_m2yr: 12
```

### Load Profile CSV Format

```csv
timestamp,flow_kg_s
2024-01-01 00:00:00,1.2
2024-01-01 01:00:00,1.1
2024-01-01 02:00:00,1.0
...
```

## Agents

### SiteInputAgent
Loads and validates site configuration from YAML files.

**Input:** `site_file` (path to YAML)
**Output:** Validated site data dictionary

### SolarResourceAgent
Fetches TMY solar radiation data for the specified location.

**Input:** `lat`, `lon` (coordinates)
**Output:** Hourly DNI and temperature data (8760 hours)

### LoadProfileAgent
Calculates hourly thermal energy demand from flow profiles.

**Input:** `process_demand` dictionary
**Output:** Hourly load profile and annual demand

### FieldLayoutAgent
Sizes the solar collector field based on demand and target solar fraction.

**Input:** `total_annual_demand_gwh`, `solar_config`
**Output:** Required aperture area, number of collectors, land area

### EnergyBalanceAgent
Performs hourly energy balance simulation (8760 hours).

**Input:** Solar resource, load profile, field size
**Output:** Solar fraction, hourly performance data

## Workflows

### Feasibility Base Workflow
Location: `climatenza_app/gl_workflows/feasibility_base.yaml`

**Pipeline:**
1. Load and validate site data (SiteInputAgent)
2. Fetch solar resource data (SolarResourceAgent)
3. Calculate load profile (LoadProfileAgent)
4. Size solar field (FieldLayoutAgent)
5. Run energy simulation (EnergyBalanceAgent)

## Examples

### Example 1: Dairy Plant Analysis
```python
from greenlang.sdk import GreenLangClient

client = GreenLangClient()

# Use provided dairy example
result = client.run_solar_feasibility(
    "climatenza_app/examples/dairy_hotwater_site.yaml"
)

if result["success"]:
    print("Feasibility Analysis Results:")
    print(f"  Solar Fraction: {result['data']['solar_fraction']:.1%}")
    print(f"  Annual Demand: {result['data']['total_annual_demand_gwh']:.3f} GWh")
    print(f"  Solar Yield: {result['data']['total_solar_yield_gwh']:.3f} GWh")
    print(f"  Collectors: {result['data']['num_collectors']}")
    print(f"  Land Required: {result['data']['required_land_area_m2']:,.0f} m²")
```

### Example 2: Custom Industrial Site
```python
import yaml
from greenlang.sdk import GreenLangClient

# Create custom configuration
config = {
    "site": {
        "name": "Textile Factory Mumbai",
        "country": "IN",
        "lat": 19.076,
        "lon": 72.877,
        "tz": "Asia/Kolkata"
    },
    "process_demand": {
        "medium": "steam",
        "temp_in_C": 80,
        "temp_out_C": 150,
        "flow_profile": "textile_load.csv",
        "schedule": {"workdays": ["Mon", "Tue", "Wed", "Thu", "Fri"]}
    },
    # ... additional configuration
}

# Save configuration
with open("textile_site.yaml", "w") as f:
    yaml.dump(config, f)

# Run analysis
client = GreenLangClient()
result = client.run_solar_feasibility("textile_site.yaml")
```

## API Reference

### GreenLangClient Methods

#### `run_solar_feasibility(site_config_path: str) -> Dict[str, Any]`
Run complete solar thermal feasibility analysis.

**Parameters:**
- `site_config_path`: Path to site configuration YAML

**Returns:**
- Dictionary with success status and results data

#### `calculate_solar_field_size(annual_demand_gwh: float, solar_config: Dict) -> Dict[str, Any]`
Calculate required solar collector field size.

**Parameters:**
- `annual_demand_gwh`: Annual thermal demand in GWh
- `solar_config`: Solar system configuration dictionary

**Returns:**
- Field sizing results

#### `simulate_energy_balance(solar_data: str, load_data: str, aperture_area: float) -> Dict[str, Any]`
Run hourly energy balance simulation.

**Parameters:**
- `solar_data`: JSON string of solar resource data
- `load_data`: JSON string of load profile data
- `aperture_area`: Total collector aperture area in m²

**Returns:**
- Simulation results including solar fraction

#### `get_solar_resource(lat: float, lon: float) -> Dict[str, Any]`
Fetch solar resource data for location.

**Parameters:**
- `lat`: Latitude in decimal degrees
- `lon`: Longitude in decimal degrees

**Returns:**
- Solar resource data (DNI and temperature)

## Troubleshooting

### Common Issues

1. **Module not found error**
   - Ensure you're in the GreenLang project root directory
   - Check that all dependencies are installed: `pip install -r requirements.txt`

2. **File not found for flow profile**
   - Use relative paths from the current working directory
   - Ensure CSV file exists at specified location

3. **Low solar fraction results**
   - Check that load profile timing matches solar availability
   - Verify temperature requirements are achievable
   - Consider increasing collector area

4. **Workflow execution errors**
   - Validate YAML configuration against schema
   - Check that all required fields are provided
   - Review agent error messages for specific issues

## Support

For issues or questions:
- GitHub Issues: [GreenLang Repository](https://github.com/greenlang/greenlang)
- Documentation: [GreenLang Docs](https://docs.greenlang.io)
- Email: support@greenlang.io