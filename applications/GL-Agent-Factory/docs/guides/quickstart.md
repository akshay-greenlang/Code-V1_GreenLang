# Getting Started with GreenLang

## Introduction

GreenLang provides a comprehensive suite of APIs and tools for carbon accounting, emissions calculations, and regulatory compliance. This guide will help you get up and running in under 10 minutes.

**What you can do with GreenLang:**

- Calculate greenhouse gas emissions from fuel, energy, and transportation
- Generate regulatory compliance reports (CBAM, EUDR, SEC Climate)
- Track Scope 1, 2, and 3 emissions across your organization
- Calculate product carbon footprints (PCF)
- Integrate sustainability data into your existing systems

**Who should use this guide:**

- **Developers** integrating emissions calculations into applications
- **Sustainability teams** automating carbon accounting workflows
- **Compliance officers** generating regulatory reports
- **Data engineers** building emissions data pipelines

---

## Prerequisites

Before you begin, you will need:

1. A GreenLang account (sign up at https://app.greenlang.io/signup)
2. API credentials (client ID and client secret)
3. Basic familiarity with REST APIs

---

## Installation

### Python SDK

```bash
pip install greenlang-sdk
```

### JavaScript/Node.js SDK

```bash
npm install @greenlang/sdk
```

### Go SDK

```bash
go get github.com/greenlang/sdk-go
```

### Direct API Access

No installation required. Use any HTTP client (curl, Postman, etc.) to call the API directly.

---

## Step 1: Get Your API Credentials

1. Log in to the [GreenLang Dashboard](https://app.greenlang.io)
2. Navigate to **Settings** > **API Keys**
3. Click **Create New API Key**
4. Copy your **Client ID** and **Client Secret**

**Important:** Store your credentials securely. Never commit them to version control or expose them in client-side code.

---

## Step 2: Authenticate

### Using the Python SDK

```python
from greenlang import Client

# Initialize the client with your credentials
client = Client(
    client_id="your_client_id",
    client_secret="your_client_secret"
)

# The SDK handles token management automatically
print("Connected to GreenLang API!")
```

### Using the JavaScript SDK

```javascript
const { GreenLangClient } = require('@greenlang/sdk');

const client = new GreenLangClient({
  clientId: 'your_client_id',
  clientSecret: 'your_client_secret'
});

// Authenticate (tokens are managed automatically)
await client.authenticate();
console.log('Connected to GreenLang API!');
```

### Using cURL (Direct API)

```bash
# Step 1: Get an access token
TOKEN=$(curl -s -X POST "https://api.greenlang.io/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=your_client_id" \
  -d "client_secret=your_client_secret" | jq -r '.access_token')

# Step 2: Use the token in subsequent requests
echo "Access Token: $TOKEN"
```

### Using Python (Direct API)

```python
import requests

# Get access token
response = requests.post(
    "https://api.greenlang.io/v1/auth/token",
    data={
        "grant_type": "client_credentials",
        "client_id": "your_client_id",
        "client_secret": "your_client_secret"
    }
)

token_data = response.json()
access_token = token_data["access_token"]

# Create headers for API requests
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

print(f"Authenticated! Token expires in {token_data['expires_in']} seconds")
```

---

## Step 3: Make Your First API Call

Let's calculate the CO2 emissions from burning 1,000 liters of diesel fuel.

### Using the Python SDK

```python
from greenlang import Client

client = Client(
    client_id="your_client_id",
    client_secret="your_client_secret"
)

# Calculate fuel emissions
result = client.calculate.fuel(
    fuel_type="diesel",
    quantity=1000,
    unit="liters",
    options={
        "include_breakdown": True
    }
)

print(f"CO2 Emissions: {result.emissions.co2:.2f} kg")
print(f"CH4 Emissions: {result.emissions.ch4:.4f} kg")
print(f"N2O Emissions: {result.emissions.n2o:.4f} kg")
print(f"Total CO2e: {result.emissions.co2e:.2f} kg")
```

**Output:**

```
CO2 Emissions: 2680.50 kg
CH4 Emissions: 0.1200 kg
N2O Emissions: 0.0800 kg
Total CO2e: 2705.20 kg
```

### Using the JavaScript SDK

```javascript
const { GreenLangClient } = require('@greenlang/sdk');

async function calculateFuelEmissions() {
  const client = new GreenLangClient({
    clientId: 'your_client_id',
    clientSecret: 'your_client_secret'
  });

  await client.authenticate();

  const result = await client.calculate.fuel({
    fuelType: 'diesel',
    quantity: 1000,
    unit: 'liters',
    options: {
      includeBreakdown: true
    }
  });

  console.log(`CO2 Emissions: ${result.emissions.co2.toFixed(2)} kg`);
  console.log(`Total CO2e: ${result.emissions.co2e.toFixed(2)} kg`);
}

calculateFuelEmissions();
```

### Using cURL

```bash
curl -X POST "https://api.greenlang.io/v1/calculate/fuel" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "fuel_type": "diesel",
    "quantity": 1000,
    "unit": "liters",
    "options": {
      "include_breakdown": true
    }
  }'
```

### Using Python (Direct API)

```python
import requests

data = {
    "fuel_type": "diesel",
    "quantity": 1000,
    "unit": "liters",
    "options": {
        "include_breakdown": True
    }
}

response = requests.post(
    "https://api.greenlang.io/v1/calculate/fuel",
    headers=headers,
    json=data
)

result = response.json()["data"]

print(f"CO2 Emissions: {result['emissions']['co2']:.2f} kg")
print(f"Total CO2e: {result['emissions']['co2e']:.2f} kg")
print(f"Calculation ID: {result['calculation_id']}")
```

---

## Step 4: Explore Available Agents

Agents are pre-configured calculation engines for specific use cases.

### List All Agents

**Python SDK:**

```python
agents = client.agents.list()

for agent in agents:
    print(f"{agent.id}: {agent.name}")
    print(f"  Category: {agent.category}")
    print(f"  Status: {agent.status}")
    print()
```

**cURL:**

```bash
curl -X GET "https://api.greenlang.io/v1/agents" \
  -H "Authorization: Bearer $TOKEN"
```

### Execute an Agent

**Python SDK:**

```python
# Execute the fuel emissions agent
execution = client.agents.execute(
    agent_id="fuel_emissions",
    inputs={
        "fuel_type": "natural_gas",
        "quantity": 10000,
        "unit": "cubic_meters"
    },
    options={
        "include_breakdown": True
    }
)

print(f"Execution ID: {execution.execution_id}")
print(f"Status: {execution.status}")
print(f"CO2e: {execution.outputs.co2e} kg")
```

---

## Common Use Cases

### Use Case 1: Fleet Emissions Tracking

Calculate emissions for a vehicle fleet consuming multiple fuel types.

```python
from greenlang import Client

client = Client(
    client_id="your_client_id",
    client_secret="your_client_secret"
)

# Fleet fuel consumption data
fleet_data = [
    {"fuel_type": "diesel", "quantity": 50000, "unit": "liters", "category": "trucks"},
    {"fuel_type": "gasoline", "quantity": 25000, "unit": "liters", "category": "cars"},
    {"fuel_type": "lpg", "quantity": 5000, "unit": "liters", "category": "forklifts"}
]

total_emissions = 0
results = []

for fuel in fleet_data:
    result = client.calculate.fuel(
        fuel_type=fuel["fuel_type"],
        quantity=fuel["quantity"],
        unit=fuel["unit"]
    )

    results.append({
        "category": fuel["category"],
        "fuel_type": fuel["fuel_type"],
        "co2e": result.emissions.co2e
    })

    total_emissions += result.emissions.co2e

print("Fleet Emissions Report")
print("=" * 40)
for r in results:
    print(f"{r['category']:15} ({r['fuel_type']:10}): {r['co2e']:>10,.2f} kg CO2e")
print("=" * 40)
print(f"{'Total':27}: {total_emissions:>10,.2f} kg CO2e")
print(f"{'Total (tonnes)':27}: {total_emissions/1000:>10,.2f} tCO2e")
```

**Output:**

```
Fleet Emissions Report
========================================
trucks          (diesel    ):  135,260.00 kg CO2e
cars            (gasoline  ):   57,750.00 kg CO2e
forklifts       (lpg       ):    7,975.00 kg CO2e
========================================
Total                       :  200,985.00 kg CO2e
Total (tonnes)              :      200.99 tCO2e
```

### Use Case 2: Building Energy Analysis

Calculate Scope 1 and Scope 2 emissions for a building.

```python
result = client.calculate.building(
    building={
        "name": "Corporate Headquarters",
        "type": "office",
        "location": {
            "country": "US",
            "state": "CA",
            "grid_region": "CAMX"
        },
        "area": {"value": 100000, "unit": "sqft"}
    },
    period={
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    },
    energy_consumption={
        "electricity": {
            "quantity": 2500000,
            "unit": "kwh",
            "renewable_percentage": 30
        },
        "natural_gas": {
            "quantity": 100000,
            "unit": "therms"
        }
    },
    options={
        "include_intensity_metrics": True,
        "market_based_accounting": True
    }
)

print(f"Scope 1 Emissions: {result.emissions.scope1.total:.2f} tCO2e")
print(f"Scope 2 (Location-Based): {result.emissions.scope2.location_based.total:.2f} tCO2e")
print(f"Scope 2 (Market-Based): {result.emissions.scope2.market_based.total:.2f} tCO2e")
print(f"Energy Use Intensity: {result.intensity_metrics.energy_use_intensity.value:.1f} kBtu/sqft")
```

### Use Case 3: CBAM Quarterly Report

Generate a CBAM report for EU imports.

```python
result = client.calculate.cbam(
    reporting_period={"year": 2025, "quarter": 1},
    imports=[
        {
            "product_id": "steel_import_001",
            "cn_code": "7208510091",
            "product_category": "iron_steel",
            "description": "Hot-rolled steel sheets",
            "quantity": 1000,
            "unit": "tonnes",
            "country_of_origin": "CN",
            "emission_data": {
                "type": "default",
                "use_country_default": True
            }
        }
    ],
    options={
        "include_certificate_estimate": True,
        "carbon_price_eur": 85.00,
        "generate_xml": True
    }
)

print(f"Total Embedded Emissions: {result.summary.total_embedded_emissions.total} tCO2e")
print(f"Estimated Certificate Cost: EUR {result.certificate_estimate.net_cost_eur:,.2f}")

# Download XML report
if result.xml_report.available:
    print(f"XML Report: {result.xml_report.download_url}")
```

### Use Case 4: Product Carbon Footprint

Calculate the carbon footprint of a product.

```python
result = client.calculate.pcf(
    product={
        "name": "Aluminum Can (330ml)",
        "sku": "CAN-AL-330",
        "category": "packaging",
        "functional_unit": {
            "description": "One 330ml aluminum beverage can",
            "quantity": 1,
            "unit": "piece"
        }
    },
    boundary="cradle_to_gate",
    lifecycle_stages={
        "raw_materials": {
            "items": [
                {"material": "aluminum_primary", "quantity": 0.015, "unit": "kg"},
                {"material": "aluminum_recycled", "quantity": 0.010, "unit": "kg"}
            ]
        },
        "manufacturing": {
            "location": {"country": "DE"},
            "processes": [
                {"process": "can_forming", "electricity_kwh": 0.02}
            ]
        }
    },
    options={
        "include_breakdown": True
    }
)

print(f"Product Carbon Footprint: {result.pcf_result.total:.3f} kgCO2e per can")
print(f"Recycled content benefit: {result.breakdown_by_stage.raw_materials.details[1].emissions:.4f} kgCO2e")
```

---

## Error Handling

Always handle potential errors gracefully:

### Python SDK

```python
from greenlang import Client
from greenlang.exceptions import (
    AuthenticationError,
    ValidationError,
    RateLimitError,
    APIError
)

client = Client(
    client_id="your_client_id",
    client_secret="your_client_secret"
)

try:
    result = client.calculate.fuel(
        fuel_type="diesel",
        quantity=1000,
        unit="liters"
    )
    print(f"CO2e: {result.emissions.co2e} kg")

except AuthenticationError as e:
    print(f"Authentication failed: {e.message}")
    print("Check your client_id and client_secret")

except ValidationError as e:
    print(f"Invalid input: {e.message}")
    for detail in e.details:
        print(f"  - {detail.field}: {detail.message}")

except RateLimitError as e:
    print(f"Rate limit exceeded. Retry after {e.retry_after} seconds")

except APIError as e:
    print(f"API error: {e.message}")
    print(f"Request ID: {e.request_id}")
```

### JavaScript SDK

```javascript
const { GreenLangClient } = require('@greenlang/sdk');

async function calculateWithErrorHandling() {
  const client = new GreenLangClient({
    clientId: 'your_client_id',
    clientSecret: 'your_client_secret'
  });

  try {
    await client.authenticate();

    const result = await client.calculate.fuel({
      fuelType: 'diesel',
      quantity: 1000,
      unit: 'liters'
    });

    console.log(`CO2e: ${result.emissions.co2e} kg`);

  } catch (error) {
    if (error.code === 'authentication_error') {
      console.error('Authentication failed:', error.message);
    } else if (error.code === 'validation_error') {
      console.error('Invalid input:', error.message);
      error.details.forEach(d => console.error(`  - ${d.field}: ${d.message}`));
    } else if (error.code === 'rate_limit_exceeded') {
      console.error(`Rate limited. Retry after ${error.retryAfter} seconds`);
    } else {
      console.error('API error:', error.message);
    }
  }
}
```

---

## Environment Configuration

### Using Environment Variables

Store your credentials in environment variables for security:

```bash
# .env file (never commit this!)
GREENLANG_CLIENT_ID=your_client_id
GREENLANG_CLIENT_SECRET=your_client_secret
GREENLANG_API_URL=https://api.greenlang.io/v1
```

**Python:**

```python
import os
from greenlang import Client

client = Client(
    client_id=os.environ["GREENLANG_CLIENT_ID"],
    client_secret=os.environ["GREENLANG_CLIENT_SECRET"]
)
```

**JavaScript:**

```javascript
require('dotenv').config();
const { GreenLangClient } = require('@greenlang/sdk');

const client = new GreenLangClient({
  clientId: process.env.GREENLANG_CLIENT_ID,
  clientSecret: process.env.GREENLANG_CLIENT_SECRET
});
```

### Using Different Environments

```python
# Production
client = Client(
    client_id=os.environ["GREENLANG_CLIENT_ID"],
    client_secret=os.environ["GREENLANG_CLIENT_SECRET"],
    base_url="https://api.greenlang.io/v1"
)

# Staging
staging_client = Client(
    client_id=os.environ["GREENLANG_STAGING_CLIENT_ID"],
    client_secret=os.environ["GREENLANG_STAGING_CLIENT_SECRET"],
    base_url="https://staging-api.greenlang.io/v1"
)

# Sandbox (for testing)
sandbox_client = Client(
    client_id=os.environ["GREENLANG_SANDBOX_CLIENT_ID"],
    client_secret=os.environ["GREENLANG_SANDBOX_CLIENT_SECRET"],
    base_url="https://sandbox-api.greenlang.io/v1"
)
```

---

## Best Practices

### 1. Reuse Client Instances

Create a single client instance and reuse it across your application:

```python
# Good: Single client instance
from greenlang import Client

client = Client(...)

def calculate_fleet_emissions(fleet_data):
    return [client.calculate.fuel(**fuel) for fuel in fleet_data]

def calculate_building_emissions(building_data):
    return client.calculate.building(**building_data)
```

### 2. Use Batch Operations

For multiple calculations, use batch endpoints when available:

```python
# Efficient: Batch calculation
results = client.calculate.batch([
    {"type": "fuel", "inputs": {"fuel_type": "diesel", "quantity": 1000, "unit": "liters"}},
    {"type": "fuel", "inputs": {"fuel_type": "gasoline", "quantity": 500, "unit": "liters"}},
    {"type": "fuel", "inputs": {"fuel_type": "natural_gas", "quantity": 2000, "unit": "cubic_meters"}}
])
```

### 3. Cache Results When Appropriate

Cache calculation results for identical inputs:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_emission_factor(fuel_type, source="ipcc_2023"):
    result = client.calculate.fuel(
        fuel_type=fuel_type,
        quantity=1,
        unit="liters",
        emission_factor_source=source,
        options={"include_breakdown": True}
    )
    return result.breakdown.emission_factors
```

### 4. Handle Rate Limits

Implement retry logic with exponential backoff:

```python
import time
from greenlang.exceptions import RateLimitError

def calculate_with_retry(client, **kwargs):
    max_retries = 5

    for attempt in range(max_retries):
        try:
            return client.calculate.fuel(**kwargs)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = e.retry_after * (2 ** attempt)  # Exponential backoff
            print(f"Rate limited. Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
```

---

## Next Steps

Now that you have completed the quickstart, explore these resources:

- **[API Reference](../api/README.md)** - Complete API documentation
- **[Agents API](../api/agents.md)** - Work with calculation agents
- **[Calculation Endpoints](../api/calculations.md)** - All calculation APIs
- **[EUDR Compliance Guide](./eudr_compliance.md)** - EU Deforestation Regulation
- **[CBAM Reporting Guide](./cbam_reporting.md)** - Carbon Border Adjustment Mechanism

---

## Support

- **Documentation:** https://docs.greenlang.io
- **API Status:** https://status.greenlang.io
- **Community Forum:** https://community.greenlang.io
- **Support Email:** support@greenlang.io
- **GitHub Issues:** https://github.com/greenlang/sdk-python/issues
