# How to Run GreenLang SDK - Complete Guide

## 🚀 THREE WAYS TO RUN SDK CODE

### Method 1: Direct Python Command (One-Liners)
Run SDK commands directly from CMD:

```bash
# Navigate to GreenLang directory
cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"

# Run SDK commands directly
python -c "from greenlang.sdk import GreenLangClient; c = GreenLangClient(); print(c.calculate_emissions('electricity', 1000, 'kWh'))"

# Calculate emissions
python -c "from greenlang.sdk import GreenLangClient; c = GreenLangClient('US'); r = c.calculate_emissions('electricity', 5000, 'kWh'); print(f'Emissions: {r[\"data\"][\"co2e_emissions_kg\"]} kg CO2e')"

# List agents
python -c "from greenlang.sdk import GreenLangClient; c = GreenLangClient(); print('Agents:', c.list_agents())"
```

### Method 2: Python Script Files
Create `.py` files and run them:

```bash
# Create a script file (e.g., sdk_demo.py)
# Then run it:
python sdk_demo.py
```

### Method 3: Interactive Python Shell
```bash
# Start Python interactive mode
python

# Then type SDK commands:
>>> from greenlang.sdk import GreenLangClient
>>> client = GreenLangClient()
>>> result = client.calculate_emissions("electricity", 1000, "kWh")
>>> print(result)
>>> exit()
```

## 📝 COMPLETE SDK SCRIPT EXAMPLES

### 1. Basic Emissions Calculator (save as `sdk_emissions.py`)
```python
#!/usr/bin/env python
"""Calculate emissions using GreenLang SDK"""

from greenlang.sdk import GreenLangClient

# Initialize client
client = GreenLangClient(region="US")

# Calculate emissions
electricity = client.calculate_emissions("electricity", 5000, "kWh")
natural_gas = client.calculate_emissions("natural_gas", 200, "therms")
diesel = client.calculate_emissions("diesel", 50, "gallons")

# Display results
print("EMISSIONS CALCULATION RESULTS")
print("="*40)
print(f"Electricity (5000 kWh): {electricity['data']['co2e_emissions_kg']:.2f} kg CO2e")
print(f"Natural Gas (200 therms): {natural_gas['data']['co2e_emissions_kg']:.2f} kg CO2e")
print(f"Diesel (50 gallons): {diesel['data']['co2e_emissions_kg']:.2f} kg CO2e")

# Aggregate
emissions_list = [
    electricity['data'],
    natural_gas['data'],
    diesel['data']
]
total = client.aggregate_emissions(emissions_list)
print(f"\nTOTAL: {total['data']['total_co2e_tons']:.3f} metric tons CO2e")
```

**Run it:** `python sdk_emissions.py`

### 2. Building Analysis (save as `sdk_building.py`)
```python
#!/usr/bin/env python
"""Analyze building emissions using SDK"""

from greenlang.sdk import GreenLangClient
import sys

# Get country from command line (default: US)
country = sys.argv[1] if len(sys.argv) > 1 else "US"

# Initialize client
client = GreenLangClient(region=country)

# Define building
building = {
    "metadata": {
        "building_type": "hospital",
        "area": 100000,
        "area_unit": "sqft",
        "location": {"country": country},
        "occupancy": 500,
        "floor_count": 5,
        "building_age": 10
    },
    "energy_consumption": {
        "electricity": {"value": 3500000, "unit": "kWh"},
        "natural_gas": {"value": 10000, "unit": "therms"},
        "diesel": {"value": 5000, "unit": "gallons"}
    }
}

# Analyze
result = client.analyze_building(building)

if result["success"]:
    data = result["data"]
    print(f"\nBUILDING ANALYSIS - {country}")
    print("="*40)
    print(f"Building Type: Hospital")
    print(f"Area: 100,000 sqft")
    print(f"Country: {country}")
    print(f"\nTotal Emissions: {data['emissions']['total_co2e_tons']:.2f} tons/year")
    print(f"Intensity: {data['intensity']['intensities']['per_sqft_year']:.2f} kgCO2e/sqft/year")
    print(f"Rating: {data['benchmark']['rating']}")
    
    print("\nEmissions Breakdown:")
    for item in data['emissions']['emissions_breakdown']:
        print(f"  - {item['source']}: {item['co2e_tons']:.2f} tons ({item['percentage']}%)")
```

**Run it:** 
```bash
python sdk_building.py        # Uses US
python sdk_building.py IN     # Uses India
python sdk_building.py EU     # Uses Europe
```

### 3. Multi-Country Comparison (save as `sdk_compare.py`)
```python
#!/usr/bin/env python
"""Compare emissions across countries"""

from greenlang.sdk import GreenLangClient

countries = ["US", "IN", "CN", "EU", "JP", "BR"]
fuel_type = "electricity"
amount = 1000
unit = "kWh"

print(f"\nCOMPARING {amount} {unit} OF {fuel_type.upper()}")
print("="*50)

for country in countries:
    client = GreenLangClient(region=country)
    result = client.calculate_emissions(fuel_type, amount, unit)
    if result["success"]:
        emissions = result['data']['co2e_emissions_kg']
        factor = result['data']['emission_factor']
        print(f"{country:3} : {emissions:6.1f} kg CO2e (factor: {factor} kgCO2e/{unit})")
```

**Run it:** `python sdk_compare.py`

### 4. Boiler Calculator (save as `sdk_boiler.py`)
```python
#!/usr/bin/env python
"""Calculate boiler emissions"""

from greenlang.sdk import GreenLangClient

client = GreenLangClient(region="US")

# Natural gas boiler
ng_boiler = client.calculate_boiler_emissions(
    fuel_type="natural_gas",
    thermal_output=1000,
    output_unit="kWh",
    efficiency=0.85,
    boiler_type="condensing"
)

# Diesel boiler
diesel_boiler = client.calculate_boiler_emissions(
    fuel_type="diesel",
    thermal_output=1000,
    output_unit="kWh",
    efficiency=0.75,
    boiler_type="standard"
)

print("\nBOILER EMISSIONS COMPARISON")
print("="*40)
print("Thermal Output: 1000 kWh\n")

print("Natural Gas Boiler (85% efficiency):")
print(f"  Emissions: {ng_boiler['data']['co2e_emissions_kg']:.2f} kg CO2e")
print(f"  Fuel Used: {ng_boiler['data']['fuel_consumption_value']:.2f} {ng_boiler['data']['fuel_consumption_unit']}")

print("\nDiesel Boiler (75% efficiency):")
print(f"  Emissions: {diesel_boiler['data']['co2e_emissions_kg']:.2f} kg CO2e")
print(f"  Fuel Used: {diesel_boiler['data']['fuel_consumption_value']:.2f} {diesel_boiler['data']['fuel_consumption_unit']}")
```

**Run it:** `python sdk_boiler.py`

### 5. Complete SDK Demo (save as `sdk_full_demo.py`)
```python
#!/usr/bin/env python
"""Complete SDK demonstration"""

from greenlang.sdk import GreenLangClient
import json

def main():
    # Initialize
    client = GreenLangClient(region="US")
    
    print("\n" + "="*60)
    print("GREENLANG SDK - COMPLETE DEMONSTRATION")
    print("="*60)
    
    # 1. Basic Emissions
    print("\n1. BASIC EMISSIONS CALCULATION")
    print("-"*40)
    result = client.calculate_emissions("electricity", 1000, "kWh")
    print(f"1000 kWh electricity: {result['data']['co2e_emissions_kg']:.2f} kg CO2e")
    
    # 2. Boiler Emissions
    print("\n2. BOILER EMISSIONS")
    print("-"*40)
    boiler = client.calculate_boiler_emissions(
        "natural_gas", 1000, "kWh", 0.85, "condensing"
    )
    print(f"Natural gas boiler: {boiler['data']['co2e_emissions_kg']:.2f} kg CO2e")
    
    # 3. Building Analysis
    print("\n3. BUILDING ANALYSIS")
    print("-"*40)
    building = {
        "metadata": {
            "building_type": "office",
            "area": 50000,
            "location": {"country": "US"},
            "occupancy": 200
        },
        "energy_consumption": {
            "electricity": {"value": 1500000, "unit": "kWh"}
        }
    }
    analysis = client.analyze_building(building)
    if analysis["success"]:
        print(f"Office emissions: {analysis['data']['emissions']['total_co2e_tons']:.2f} tons/year")
        print(f"Rating: {analysis['data']['benchmark']['rating']}")
    
    # 4. Benchmarking
    print("\n4. BENCHMARKING")
    print("-"*40)
    benchmark = client.benchmark_emissions(50000, 10000, "office", 12)
    print(f"Benchmark rating: {benchmark['data']['rating']}")
    
    # 5. Intensity Metrics
    print("\n5. INTENSITY METRICS")
    print("-"*40)
    intensity = client.calculate_intensity(100000, 50000, "sqft", 200)
    print(f"Per sqft: {intensity['data']['intensities']['per_sqft_year']:.2f} kgCO2e/sqft/year")
    print(f"Per person: {intensity['data']['intensities']['per_person_year']:.0f} kgCO2e/person/year")
    
    # 6. Grid Factors
    print("\n6. GRID EMISSION FACTORS")
    print("-"*40)
    for country in ["US", "IN", "EU"]:
        factor = client.get_emission_factor("electricity", country, "kWh")
        print(f"{country}: {factor['data']['emission_factor']} kgCO2e/kWh")
    
    # 7. List Utilities
    print("\n7. AVAILABLE RESOURCES")
    print("-"*40)
    print(f"Agents: {len(client.list_agents())} available")
    print(f"Countries: {len(client.get_supported_countries())} supported")
    
    print("\n" + "="*60)
    print("SDK DEMONSTRATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
```

**Run it:** `python sdk_full_demo.py`

## 🔧 BATCH FILES FOR WINDOWS

### Create `run_sdk.bat`:
```batch
@echo off
echo ========================================
echo GreenLang SDK Runner
echo ========================================
echo.
echo Choose an option:
echo 1. Calculate Emissions
echo 2. Analyze Building
echo 3. Compare Countries
echo 4. Boiler Calculator
echo 5. Full Demo
echo 6. Custom Python Script
echo.
set /p choice="Enter choice (1-6): "

if %choice%==1 python sdk_emissions.py
if %choice%==2 (
    set /p country="Enter country (US/IN/EU/CN/JP/BR): "
    python sdk_building.py %country%
)
if %choice%==3 python sdk_compare.py
if %choice%==4 python sdk_boiler.py
if %choice%==5 python sdk_full_demo.py
if %choice%==6 (
    set /p script="Enter script name: "
    python %script%
)

pause
```

## 📊 ONE-LINE CMD COMMANDS

### Quick SDK Commands for CMD:
```bash
# Calculate emissions
python -c "from greenlang.sdk import GreenLangClient as G; c=G(); print(c.calculate_emissions('electricity',1000,'kWh')['data']['co2e_emissions_kg'], 'kg CO2e')"

# List all agents
python -c "from greenlang.sdk import GreenLangClient as G; c=G(); print('Agents:', c.list_agents())"

# Get emission factor
python -c "from greenlang.sdk import GreenLangClient as G; c=G(); print('US Grid:', c.get_emission_factor('electricity','US','kWh')['data']['emission_factor'], 'kgCO2e/kWh')"

# Compare countries
python -c "from greenlang.sdk import GreenLangClient as G; [print(f'{c}: {G(c).calculate_emissions(\"electricity\",1000,\"kWh\")[\"data\"][\"co2e_emissions_kg\"]:.0f} kg') for c in ['US','IN','EU','CN','JP','BR']]"

# Boiler calculation
python -c "from greenlang.sdk import GreenLangClient as G; c=G(); r=c.calculate_boiler_emissions('natural_gas',1000,'kWh',0.85); print(f'Boiler: {r[\"data\"][\"co2e_emissions_kg\"]:.2f} kg CO2e')"
```

## 🎯 QUICK START STEPS

1. **Navigate to GreenLang directory:**
```bash
cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"
```

2. **Create a simple test script (`test_sdk.py`):**
```python
from greenlang.sdk import GreenLangClient
client = GreenLangClient()
result = client.calculate_emissions("electricity", 1000, "kWh")
print(f"Result: {result['data']['co2e_emissions_kg']} kg CO2e")
```

3. **Run it:**
```bash
python test_sdk.py
```

## 📚 COMPLETE COMMAND REFERENCE

| What You Want | Command to Run |
|--------------|---------------|
| Calculate emissions | `python -c "from greenlang.sdk import GreenLangClient; print(GreenLangClient().calculate_emissions('electricity', 1000, 'kWh'))"` |
| List agents | `python -c "from greenlang.sdk import GreenLangClient; print(GreenLangClient().list_agents())"` |
| Get countries | `python -c "from greenlang.sdk import GreenLangClient; print(GreenLangClient().get_supported_countries())"` |
| Run script | `python your_script.py` |
| Interactive mode | `python` then type commands |
| One-liner calc | `python -c "exec('from greenlang.sdk import GreenLangClient\\nc=GreenLangClient()\\nprint(c.calculate_emissions(\"electricity\",1000,\"kWh\"))')"` |

---

**The SDK can be run from CMD in multiple ways: direct commands, script files, or interactive mode!**