# Building Analysis - Complete Guide ✅

## ✅ FIXED: Building Analysis Commands

### Working Commands

```bash
# Analyze building from JSON file
greenlang analyze building_data.json

# Analyze with country override
greenlang analyze building_data.json --country IN
greenlang analyze building_data.json --country EU
greenlang analyze building_data.json --country CN
```

## Sample Building Data Files

### 1. Hospital (building_data.json)
```json
{
  "metadata": {
    "building_type": "hospital",
    "area": 100000,
    "area_unit": "sqft",
    "location": {
      "country": "US",
      "city": "New York"
    },
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
```

### 2. Office Building (office_building.json)
```json
{
  "metadata": {
    "building_type": "commercial_office",
    "area": 50000,
    "area_unit": "sqft",
    "location": {"country": "US"},
    "occupancy": 200,
    "floor_count": 10,
    "building_age": 5
  },
  "energy_consumption": {
    "electricity": {"value": 1500000, "unit": "kWh"},
    "natural_gas": {"value": 5000, "unit": "therms"}
  }
}
```

### 3. Data Center
```json
{
  "metadata": {
    "building_type": "data_center",
    "area": 20000,
    "area_unit": "sqft",
    "location": {"country": "US"},
    "occupancy": 50,
    "floor_count": 2,
    "building_age": 3
  },
  "energy_consumption": {
    "electricity": {"value": 10000000, "unit": "kWh"},
    "diesel": {"value": 10000, "unit": "gallons"}
  }
}
```

## Analysis Output

The analysis provides:
1. **Building Profile** - Type, location, area, expected performance
2. **Total Emissions** - Annual CO2e in kg and metric tons
3. **Emissions Breakdown** - By fuel source with percentages
4. **Intensity Metrics** - Per sqft, per person, EUI
5. **Performance Rating** - Excellent/Good/Average/Poor
6. **Benchmark Rating** - Industry comparison
7. **Recommendations** - Quick wins and long-term strategies

## Example Results

### US Hospital (100,000 sqft)
```
Total Emissions: 1,451.55 metric tons CO2e/year
Intensity: 14.52 kgCO2e/sqft/year
Rating: Excellent
Breakdown:
  - Electricity: 92.83%
  - Natural Gas: 3.65%
  - Diesel: 3.52%
```

### India Hospital (Same building, different grid)
```
Total Emissions: 2,589.15 metric tons CO2e/year
Intensity: 25.89 kgCO2e/sqft/year
Rating: Good
Breakdown:
  - Electricity: 95.98% (higher grid factor)
  - Natural Gas: 2.05%
  - Diesel: 1.98%
```

### EU Office (50,000 sqft)
```
Total Emissions: 370.00 metric tons CO2e/year
Intensity: 7.40 kgCO2e/sqft/year
Rating: Excellent
Note: EU has cleaner grid (0.23 kgCO2e/kWh vs 0.385 US)
```

## Building Types Supported

- `commercial_office` - Office buildings
- `hospital` - Healthcare facilities
- `data_center` - Data centers
- `retail` - Retail stores
- `warehouse` - Storage facilities
- `hotel` - Hospitality
- `education` - Schools/universities
- `restaurant` - Food service
- `industrial` - Manufacturing

## Energy Units Supported

### Electricity
- `kWh` - Kilowatt-hours
- `MWh` - Megawatt-hours
- `GWh` - Gigawatt-hours

### Natural Gas
- `therms` - Thermal units
- `m3` - Cubic meters
- `MMBtu` - Million BTU
- `ccf` - Hundred cubic feet

### Liquid Fuels
- `gallons` - US gallons
- `liters` - Liters
- `kg` - Kilograms

## Country Codes

- `US` - United States
- `IN` - India
- `CN` - China
- `EU` - European Union
- `JP` - Japan
- `BR` - Brazil
- `KR` - South Korea
- `UK` - United Kingdom
- `DE` - Germany
- `CA` - Canada
- `AU` - Australia

## Advanced Features

### Save Results to File
```bash
# Not yet implemented in CLI, but available in SDK:
from greenlang.sdk import GreenLangClient
client = GreenLangClient()
results = client.analyze_building(building_data)
# Save to JSON
import json
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Batch Analysis
```python
# Analyze multiple buildings
buildings = ["building1.json", "building2.json", "building3.json"]
for building_file in buildings:
    with open(building_file) as f:
        data = json.load(f)
    result = client.analyze_building(data)
    print(f"{building_file}: {result['data']['emissions']['total_co2e_tons']} tons")
```

## Troubleshooting

### If analysis fails:
1. Check JSON format is valid
2. Ensure all required fields are present
3. Verify units are supported
4. Check country code is valid

### Required Fields:
- `metadata.building_type`
- `metadata.area`
- `metadata.location.country`
- `energy_consumption` (at least one fuel type)

### Optional Fields:
- `metadata.occupancy`
- `metadata.floor_count`
- `metadata.building_age`
- `metadata.climate_zone`
- `hvac_system`
- `building_envelope`
- `appliance_loads`

---

**The building analysis command is now fully operational!**