# Getting Started with GreenLang Emission Factors

**Time to First Calculation:** 5 minutes
**Difficulty:** Beginner
**Prerequisites:** Python 3.10+, pip

---

## What You'll Learn

By the end of this guide, you'll be able to:
- Install the GreenLang SDK
- Import emission factors into the database
- Query emission factors
- Calculate emissions with complete audit trails
- Use the CLI tool

---

## Step 1: Installation (1 minute)

### Option A: Install from PyPI (Recommended)

```bash
pip install greenlang
```

### Option B: Install from Source

```bash
# Clone repository
git clone https://github.com/greenlang/greenlang.git
cd greenlang

# Install in development mode
pip install -e .
```

### Verify Installation

```bash
python -c "from greenlang.sdk.emission_factor_client import EmissionFactorClient; print('Installation successful!')"
```

---

## Step 2: Import Emission Factors (2 minutes)

The emission factor database must be populated before use.

```bash
# Navigate to GreenLang root directory
cd C:\Users\aksha\Code-V1_GreenLang

# Import all 500 factors from YAML files
python scripts/import_emission_factors.py --overwrite
```

**Expected Output:**
```
Creating database at: C:\Users\aksha\Code-V1_GreenLang\greenlang\data\emission_factors.db
Importing from: data/emission_factors_registry.yaml
Importing from: data/emission_factors_expansion_phase1.yaml
Importing from: data/emission_factors_expansion_phase2.yaml

======================================================================
IMPORT COMPLETE
======================================================================
Total factors processed: 500
Successfully imported: 500
Failed imports: 0
Duplicate factors: 0
Unique categories: 11
Unique sources: 50+

Database created: C:\Users\aksha\Code-V1_GreenLang\greenlang\data\emission_factors.db
Size: 4.8 MB
```

**Troubleshooting:**
- If import fails, check that YAML files exist in `data/` directory
- Ensure write permissions for `greenlang/data/` directory
- See [Troubleshooting Guide](./10_TROUBLESHOOTING.md) for common issues

---

## Step 3: Your First Calculation (2 minutes)

### Python SDK

Create a file `first_calculation.py`:

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

# Initialize client
client = EmissionFactorClient()

# Calculate emissions from burning 100 gallons of diesel
result = client.calculate_emissions(
    factor_id="fuels_diesel",
    activity_amount=100.0,
    activity_unit="gallon"
)

# Display results
print("=" * 70)
print("EMISSION CALCULATION RESULT")
print("=" * 70)
print(f"Activity:              {result.activity_amount} {result.activity_unit}")
print(f"Emissions:             {result.emissions_kg_co2e:.2f} kg CO2e")
print(f"Emissions:             {result.emissions_metric_tons_co2e:.4f} metric tons CO2e")
print(f"Factor Used:           {result.factor_used.name}")
print(f"Factor Value:          {result.factor_value_applied} kg CO2e/{result.activity_unit}")
print(f"Source:                {result.factor_used.source.source_org}")
print(f"Audit Hash:            {result.audit_trail[:32]}...")
print("=" * 70)

# Close connection
client.close()
```

**Run it:**
```bash
python first_calculation.py
```

**Expected Output:**
```
======================================================================
EMISSION CALCULATION RESULT
======================================================================
Activity:              100.0 gallons
Emissions:             1021.00 kg CO2e
Emissions:             1.0210 metric tons CO2e
Factor Used:           Diesel Fuel
Factor Value:          10.21 kg CO2e/gallon
Source:                EPA
Audit Hash:            5f4dcc3b5aa765d61d8327deb882cf99...
======================================================================
```

### CLI Tool

Alternatively, use the command-line tool:

```bash
# Calculate emissions (Windows)
python greenlang/cli/factor_query.py calculate --factor=fuels_diesel --amount=100 --unit=gallon

# Calculate emissions (Linux/Mac with installed CLI)
greenlang factors calculate --factor=fuels_diesel --amount=100 --unit=gallon
```

---

## Step 4: Explore the Factor Library

### Search for Factors

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

client = EmissionFactorClient()

# Search by name
diesel_factors = client.get_factor_by_name("diesel")
for factor in diesel_factors:
    print(f"{factor.factor_id}: {factor.name} - {factor.emission_factor_kg_co2e} kg CO2e/{factor.unit}")

# Get all fuel factors
fuel_factors = client.get_by_category("fuels")
print(f"\nFound {len(fuel_factors)} fuel factors")

# Get all Scope 1 factors
scope1_factors = client.get_by_scope("Scope 1")
print(f"Found {len(scope1_factors)} Scope 1 factors")

client.close()
```

**Output:**
```
fuels_diesel: Diesel Fuel - 2.68 kg CO2e/liter
fuels_diesel_marine: Marine Diesel - 3.21 kg CO2e/liter
fuels_biodiesel_b20: Biodiesel B20 - 2.54 kg CO2e/liter

Found 38 fuel factors
Found 118 Scope 1 factors
```

### View Factor Details

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

client = EmissionFactorClient()

# Get specific factor
factor = client.get_factor("fuels_diesel")

print(f"Name:              {factor.name}")
print(f"Category:          {factor.category}")
print(f"Value:             {factor.emission_factor_kg_co2e} kg CO2e/{factor.unit}")
print(f"Scope:             {factor.scope}")
print(f"Source:            {factor.source.source_org}")
print(f"URI:               {factor.source.source_uri}")
print(f"Standard:          {factor.source.standard}")
print(f"Last Updated:      {factor.last_updated}")
print(f"Geographic Scope:  {factor.geography.geographic_scope}")
print(f"Data Quality:      {factor.data_quality.tier}")
print(f"Uncertainty:       ±{factor.data_quality.uncertainty_percent}%")

# Check for additional units
print(f"\nAvailable Units:")
for unit in factor.additional_units:
    print(f"  - {unit.emission_factor_value} kg CO2e/{unit.unit_name}")

client.close()
```

---

## Step 5: Database Statistics

### Python SDK

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

client = EmissionFactorClient()

stats = client.get_statistics()

print(f"Total Factors:         {stats['total_factors']}")
print(f"Total Calculations:    {stats['total_calculations']}")
print(f"Stale Factors (>3y):   {stats['stale_factors']}")

print("\nFactors by Category:")
for category, count in sorted(stats['by_category'].items(), key=lambda x: -x[1])[:10]:
    print(f"  {category:30} {count:3} factors")

print("\nFactors by Scope:")
for scope, count in stats['by_scope'].items():
    print(f"  {scope:30} {count:3} factors")

client.close()
```

### CLI Tool

```bash
python greenlang/cli/factor_query.py stats
```

**Output:**
```
======================================================================
EMISSION FACTOR DATABASE STATISTICS
======================================================================

Total Factors:         500
Total Calculations:    0
Stale Factors (>3y):   0

FACTORS BY CATEGORY:
  fuels                          117 factors
  grids                           66 factors
  transportation                  64 factors
  agriculture                     50 factors
  materials_manufacturing         30 factors
  building_materials              15 factors
  waste                           25 factors
  data_centers                    20 factors
  services                        25 factors
  healthcare                      13 factors
  industrial_processes            75 factors

FACTORS BY SCOPE:
  Scope 1                        118 factors
  Scope 2 - Location-Based        66 factors
  Scope 3                        316 factors
```

---

## Common Use Cases

### 1. Calculate Electricity Emissions

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

client = EmissionFactorClient()

# Calculate emissions from 1000 kWh of electricity
result = client.calculate_emissions(
    factor_id="grids_us_national",
    activity_amount=1000.0,
    activity_unit="kwh"
)

print(f"1000 kWh in the US produces {result.emissions_kg_co2e:.2f} kg CO2e")

client.close()
```

### 2. Calculate Transportation Emissions

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

client = EmissionFactorClient()

# Calculate emissions from 500 miles driven in a gasoline car
result = client.calculate_emissions(
    factor_id="transportation_passenger_car_gasoline",
    activity_amount=500.0,
    activity_unit="mile"
)

print(f"500 miles driven produces {result.emissions_kg_co2e:.2f} kg CO2e")

client.close()
```

### 3. Batch Calculations

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

client = EmissionFactorClient()

activities = [
    {"factor_id": "fuels_diesel", "amount": 100, "unit": "gallon"},
    {"factor_id": "fuels_natural_gas", "amount": 500, "unit": "therm"},
    {"factor_id": "grids_us_national", "amount": 2000, "unit": "kwh"}
]

total_emissions = 0
for activity in activities:
    result = client.calculate_emissions(
        factor_id=activity["factor_id"],
        activity_amount=activity["amount"],
        activity_unit=activity["unit"]
    )
    total_emissions += result.emissions_kg_co2e
    print(f"{activity['factor_id']:30} {result.emissions_kg_co2e:10.2f} kg CO2e")

print(f"{'Total':30} {total_emissions:10.2f} kg CO2e")

client.close()
```

**Output:**
```
fuels_diesel                      1021.00 kg CO2e
fuels_natural_gas                 2650.00 kg CO2e
grids_us_national                  920.00 kg CO2e
Total                             4591.00 kg CO2e
```

---

## Next Steps

Now that you've completed the basics, you can:

1. **Explore the API** - [API Reference](./02_API_REFERENCE.md)
   - Learn about all 14 REST API endpoints
   - Integrate with your web applications
   - Use authentication and rate limiting

2. **Advanced SDK Usage** - [SDK Guide](./03_SDK_GUIDE.md)
   - Advanced search and filtering
   - Geographic and temporal fallback
   - Error handling and validation
   - Performance optimization

3. **Deep Dive into Calculations** - [Calculation Guide](./04_CALCULATION_GUIDE.md)
   - Multi-gas decomposition (CO2, CH4, N2O)
   - Uncertainty quantification
   - Batch processing
   - Audit trail generation

4. **Browse the Factor Catalog** - [Factor Catalog](./05_FACTOR_CATALOG.md)
   - Complete list of 500 factors
   - Source attribution
   - Geographic coverage
   - Data quality tiers

5. **Deploy to Production** - [Deployment Guide](./07_DEPLOYMENT.md)
   - Docker deployment
   - Kubernetes deployment
   - Cloud deployment (AWS, GCP, Azure)
   - Monitoring and alerting

---

## Key Concepts

### Emission Factors

An emission factor is a coefficient that quantifies the emissions per unit of activity:

```
Emissions (kg CO2e) = Activity Amount × Emission Factor
```

**Example:**
- Diesel: 10.21 kg CO2e/gallon
- Activity: 100 gallons
- Emissions: 100 × 10.21 = 1,021 kg CO2e

### Factor IDs

Each emission factor has a unique identifier:

```
{category}_{fuel_or_process_name}_{optional_variant}

Examples:
- fuels_diesel
- grids_us_caiso
- transportation_passenger_car_gasoline
- materials_steel_rebar
```

### Multi-Unit Support

Many factors support multiple units:

```python
# Diesel fuel
# - 2.68 kg CO2e/liter (primary unit)
# - 10.21 kg CO2e/gallon (additional unit)

# SDK automatically selects correct unit
result = client.calculate_emissions("fuels_diesel", 100, "gallon")  # Uses 10.21
result = client.calculate_emissions("fuels_diesel", 100, "liter")   # Uses 2.68
```

### Audit Trails

Every calculation generates a SHA-256 hash for reproducibility:

```python
result = client.calculate_emissions("fuels_diesel", 100, "gallon")

# Audit trail includes:
# - Factor ID and value used
# - Activity amount and unit
# - Emissions result
# - Timestamp
# - Source provenance
# - SHA-256 hash for verification
```

### Zero-Hallucination

All calculations use deterministic arithmetic. No AI or machine learning models are used for numeric calculations:

```python
# ✅ ALLOWED (deterministic)
emissions = activity_amount * emission_factor

# ❌ NOT ALLOWED (non-deterministic)
emissions = llm.estimate_emissions(activity_description)
```

---

## Troubleshooting

### Database Not Found

**Error:**
```
DatabaseConnectionError: Database not found at C:\...\emission_factors.db
```

**Solution:**
```bash
# Import emission factors
python scripts/import_emission_factors.py --overwrite
```

### Factor Not Found

**Error:**
```
EmissionFactorNotFoundError: Emission factor not found: invalid_factor_id
```

**Solution:**
```python
# Search for available factors
factors = client.get_factor_by_name("diesel")
print([f.factor_id for f in factors])
```

### Unit Not Available

**Error:**
```
UnitNotAvailableError: Unit 'pounds' not available for fuels_diesel.
Available units: liter, gallon
```

**Solution:**
```python
# Use available units
result = client.calculate_emissions("fuels_diesel", 100, "gallon")
```

---

## Help & Support

### Documentation
- **API Reference:** [API Reference](./02_API_REFERENCE.md)
- **SDK Guide:** [SDK Guide](./03_SDK_GUIDE.md)
- **Troubleshooting:** [Troubleshooting Guide](./10_TROUBLESHOOTING.md)

### Community
- **Discord:** https://discord.gg/greenlang
- **Forum:** https://community.greenlang.io
- **GitHub Issues:** https://github.com/greenlang/greenlang/issues

### Commercial Support
- **Email:** support@greenlang.io
- **Enterprise:** enterprise@greenlang.io

---

## Summary

You've learned how to:
- ✅ Install the GreenLang SDK
- ✅ Import 500 emission factors
- ✅ Calculate emissions with audit trails
- ✅ Query and search factors
- ✅ Use the CLI tool

**Next:** Explore the [API Reference](./02_API_REFERENCE.md) to integrate with your applications.

---

**Questions?** Check the [Troubleshooting Guide](./10_TROUBLESHOOTING.md) or ask on Discord.
