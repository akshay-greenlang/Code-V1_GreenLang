# GreenLang API Documentation

## Overview
GreenLang provides a comprehensive REST API and Python SDK for calculating carbon emissions, benchmarking performance, and generating climate intelligence reports.

## Base URL
```
http://localhost:5000
```

## Authentication
Currently, the API does not require authentication. Future versions will support API key authentication.

## API Endpoints

### 1. Calculate Emissions
Calculate carbon emissions for multiple fuel sources.

**Endpoint:** `POST /api/calculate`

**Request Body:**
```json
{
  "fuels": [
    {
      "type": "electricity",
      "consumption": 1000,
      "unit": "kWh"
    },
    {
      "type": "natural_gas", 
      "consumption": 50,
      "unit": "therms"
    }
  ],
  "region": "US",
  "building_info": {
    "area": 10000,
    "type": "commercial_office"
  },
  "period_months": 1
}
```

**Response:**
```json
{
  "success": true,
  "emissions": {
    "total_co2e_kg": 823.45,
    "total_co2e_tons": 0.82345,
    "emissions_breakdown": [
      {
        "fuel_type": "electricity",
        "co2e_kg": 456.78,
        "co2e_tons": 0.45678
      },
      {
        "fuel_type": "natural_gas",
        "co2e_kg": 366.67,
        "co2e_tons": 0.36667
      }
    ]
  },
  "report": "Carbon Footprint Report...",
  "benchmark": {
    "performance": "above_average",
    "comparison": "15% better than industry average",
    "intensity": 8.23
  }
}
```

### 2. Quick Calculation
Perform a quick emissions calculation with minimal parameters.

**Endpoint:** `GET /api/quick-calc`

**Query Parameters:**
- `electricity` (float): Electricity consumption in kWh
- `gas` (float): Natural gas consumption in therms

**Example:**
```
GET /api/quick-calc?electricity=1000&gas=50
```

**Response:**
```json
{
  "success": true,
  "total_co2e_tons": 0.82345,
  "total_co2e_kg": 823.45,
  "breakdown": [
    {
      "fuel_type": "electricity",
      "co2e_kg": 456.78
    },
    {
      "fuel_type": "natural_gas",
      "co2e_kg": 366.67
    }
  ]
}
```

### 3. List Available Agents
Get information about available calculation agents.

**Endpoint:** `GET /api/agents`

**Response:**
```json
[
  {
    "id": "validator",
    "name": "Input Validator Agent",
    "description": "Validates input data for emissions calculations"
  },
  {
    "id": "fuel",
    "name": "Fuel Agent",
    "description": "Calculates emissions from fuel consumption"
  },
  {
    "id": "carbon",
    "name": "Carbon Agent",
    "description": "Aggregates carbon emissions data"
  },
  {
    "id": "report",
    "name": "Report Agent",
    "description": "Generates emissions reports"
  },
  {
    "id": "benchmark",
    "name": "Benchmark Agent",
    "description": "Compares emissions to industry benchmarks"
  }
]
```

## Python SDK

### Installation
```bash
pip install greenlang
```

### Basic Usage

```python
from greenlang.sdk import GreenLangClient

# Initialize client
client = GreenLangClient()

# Calculate emissions for a single fuel
result = client.calculate_emissions(
    fuel_type="electricity",
    consumption=1000,
    unit="kWh",
    region="US"
)

print(f"CO2 emissions: {result['data']['co2e_kg']} kg")
```

### Advanced Usage

```python
from greenlang.sdk import GreenLangClient

client = GreenLangClient()

# Calculate emissions for multiple fuels
emissions_list = []

# Electricity
elec_result = client.calculate_emissions(
    fuel_type="electricity",
    consumption=5000,
    unit="kWh",
    region="US"
)
emissions_list.append(elec_result['data'])

# Natural Gas
gas_result = client.calculate_emissions(
    fuel_type="natural_gas",
    consumption=100,
    unit="therms",
    region="US"
)
emissions_list.append(gas_result['data'])

# Aggregate emissions
agg_result = client.aggregate_emissions(emissions_list)
total_emissions = agg_result['data']['total_co2e_kg']

# Generate report
report_result = client.generate_report(
    carbon_data=agg_result['data'],
    format='text',
    building_info={
        'name': 'Office Building A',
        'area': 10000,
        'type': 'commercial_office'
    }
)

# Benchmark emissions
benchmark_result = client.benchmark_emissions(
    total_emissions_kg=total_emissions,
    building_area=10000,
    building_type='commercial_office',
    period_months=12
)

print(f"Total emissions: {total_emissions} kg CO2e")
print(f"Performance: {benchmark_result['data']['performance']}")
print(f"Report: {report_result['data']['report']}")
```

### SDK Methods

#### `calculate_emissions(fuel_type, consumption, unit, region='US')`
Calculate emissions for a specific fuel type.

**Parameters:**
- `fuel_type` (str): Type of fuel (electricity, natural_gas, diesel, gasoline)
- `consumption` (float): Amount consumed
- `unit` (str): Unit of measurement (kWh, therms, gallons, liters)
- `region` (str): Geographic region (default: 'US')

**Returns:** Dictionary with emissions data

#### `aggregate_emissions(emissions_list)`
Aggregate emissions from multiple sources.

**Parameters:**
- `emissions_list` (list): List of emissions dictionaries

**Returns:** Dictionary with total emissions and breakdown

#### `generate_report(carbon_data, format='text', building_info=None)`
Generate an emissions report.

**Parameters:**
- `carbon_data` (dict): Aggregated emissions data
- `format` (str): Report format ('text', 'json', 'html')
- `building_info` (dict): Optional building information

**Returns:** Dictionary with formatted report

#### `benchmark_emissions(total_emissions_kg, building_area, building_type, period_months)`
Compare emissions to industry benchmarks.

**Parameters:**
- `total_emissions_kg` (float): Total emissions in kg CO2e
- `building_area` (float): Building area in square feet
- `building_type` (str): Type of building
- `period_months` (int): Time period in months

**Returns:** Dictionary with benchmark comparison

### Workflow Execution

```python
from greenlang.sdk import GreenLangClient
from greenlang.core.workflow import WorkflowBuilder

client = GreenLangClient()

# Create custom workflow
builder = WorkflowBuilder("custom_workflow", "My Custom Workflow")
builder.add_step("validate", "validator")
builder.add_step("calculate", "fuel")
builder.add_step("aggregate", "carbon")
builder.add_step("report", "report")
workflow = builder.build()

# Register workflow
client.register_workflow("custom", workflow)

# Execute workflow
input_data = {
    "fuels": [
        {"type": "electricity", "consumption": 1000, "unit": "kWh"}
    ],
    "region": "US"
}

result = client.execute_workflow("custom", input_data)
print(result)
```

### Custom Agents

```python
from greenlang.sdk import GreenLangClient
from greenlang.agents import BaseAgent
from greenlang.data.models import AgentResult

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="custom",
            name="Custom Agent",
            description="Custom emissions calculation"
        )
    
    def execute(self, input_data):
        # Custom logic here
        result_data = {
            "custom_metric": input_data.get("value", 0) * 2
        }
        return AgentResult(
            success=True,
            data=result_data,
            message="Custom calculation complete"
        )

# Register custom agent
client = GreenLangClient()
client.register_agent("custom", CustomAgent())

# Use custom agent
result = client.execute_agent("custom", {"value": 100})
print(result)
```

## Error Handling

All API endpoints return consistent error responses:

```json
{
  "success": false,
  "error": "Error message describing the issue"
}
```

HTTP Status Codes:
- `200 OK`: Successful request
- `400 Bad Request`: Invalid input data
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## Rate Limiting

Currently no rate limiting is implemented. Future versions will include:
- 100 requests per minute per IP
- 10,000 requests per day per API key

## Supported Fuel Types

- `electricity`: Measured in kWh
- `natural_gas`: Measured in therms or cubic meters
- `diesel`: Measured in gallons or liters
- `gasoline`: Measured in gallons or liters
- `propane`: Measured in gallons or liters
- `coal`: Measured in tons or kg
- `fuel_oil`: Measured in gallons or liters

## Supported Regions

- `US`: United States (default)
- `EU`: European Union
- `UK`: United Kingdom
- `IN`: India
- `CN`: China
- `JP`: Japan
- `AU`: Australia
- `CA`: Canada

## Building Types

- `commercial_office`: Commercial office building
- `retail`: Retail store
- `warehouse`: Warehouse/storage
- `residential`: Residential building
- `industrial`: Industrial facility
- `hospital`: Healthcare facility
- `school`: Educational institution
- `hotel`: Hospitality
- `restaurant`: Food service
- `data_center`: Data center

## Examples

### cURL Examples

#### Calculate emissions:
```bash
curl -X POST http://localhost:5000/api/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "fuels": [
      {"type": "electricity", "consumption": 1000, "unit": "kWh"},
      {"type": "natural_gas", "consumption": 50, "unit": "therms"}
    ],
    "region": "US"
  }'
```

#### Quick calculation:
```bash
curl "http://localhost:5000/api/quick-calc?electricity=1000&gas=50"
```

#### List agents:
```bash
curl http://localhost:5000/api/agents
```

### JavaScript/Fetch Examples

```javascript
// Calculate emissions
async function calculateEmissions() {
  const response = await fetch('http://localhost:5000/api/calculate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      fuels: [
        { type: 'electricity', consumption: 1000, unit: 'kWh' },
        { type: 'natural_gas', consumption: 50, unit: 'therms' }
      ],
      region: 'US',
      building_info: {
        area: 10000,
        type: 'commercial_office'
      }
    })
  });
  
  const data = await response.json();
  console.log('Total emissions:', data.emissions.total_co2e_kg, 'kg CO2e');
  return data;
}

// Quick calculation
async function quickCalc(electricity, gas) {
  const response = await fetch(
    `http://localhost:5000/api/quick-calc?electricity=${electricity}&gas=${gas}`
  );
  const data = await response.json();
  return data;
}

// List agents
async function listAgents() {
  const response = await fetch('http://localhost:5000/api/agents');
  const agents = await response.json();
  return agents;
}
```

## Webhook Support (Coming Soon)

Future versions will support webhooks for async calculations:

```json
{
  "webhook_url": "https://your-server.com/webhook",
  "fuels": [...],
  "async": true
}
```

## Batch Processing (Coming Soon)

Future versions will support batch processing:

```json
{
  "batch": [
    {
      "id": "building_1",
      "fuels": [...]
    },
    {
      "id": "building_2", 
      "fuels": [...]
    }
  ]
}
```

## Support

For issues, questions, or feature requests:
- GitHub: https://github.com/greenlang/greenlang
- Email: support@greenlang.io
- Documentation: https://docs.greenlang.io