# Agents API

## Overview

Agents are pre-configured calculation engines that perform specific emissions calculations and compliance tasks. Each agent encapsulates domain expertise, calculation methodologies, and regulatory requirements.

**Base URL:** `https://api.greenlang.io/v1`

---

## Available Agents

| Agent ID | Name | Description |
|----------|------|-------------|
| `fuel_emissions` | Fuel Emissions Calculator | Calculate GHG emissions from fuel combustion |
| `cbam_reporter` | CBAM Reporter | EU Carbon Border Adjustment Mechanism reporting |
| `building_energy` | Building Energy Analyzer | Building energy consumption and emissions |
| `eudr_compliance` | EUDR Compliance | EU Deforestation Regulation compliance |
| `scope3_calculator` | Scope 3 Calculator | Value chain emissions calculation |
| `pcf_generator` | PCF Generator | Product Carbon Footprint calculation |

---

## Endpoints

### List Agents

Retrieve all available agents.

```http
GET /v1/agents
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `category` | string | No | Filter by category: `emissions`, `compliance`, `reporting` |
| `status` | string | No | Filter by status: `active`, `beta`, `deprecated` |
| `page` | integer | No | Page number (default: 1) |
| `per_page` | integer | No | Items per page (default: 20, max: 100) |

**Response (200 OK):**

```json
{
  "data": [
    {
      "id": "fuel_emissions",
      "name": "Fuel Emissions Calculator",
      "description": "Calculate greenhouse gas emissions from fuel combustion using IPCC, EPA, or DEFRA emission factors",
      "version": "2.1.0",
      "category": "emissions",
      "status": "active",
      "supported_inputs": ["fuel_type", "quantity", "unit", "emission_factor_source"],
      "supported_outputs": ["co2", "ch4", "n2o", "co2e", "breakdown"],
      "pricing_tier": "starter",
      "created_at": "2024-01-15T00:00:00Z",
      "updated_at": "2025-01-10T14:30:00Z"
    },
    {
      "id": "cbam_reporter",
      "name": "CBAM Reporter",
      "description": "Generate EU Carbon Border Adjustment Mechanism quarterly reports with embedded emissions calculations",
      "version": "1.5.0",
      "category": "compliance",
      "status": "active",
      "supported_inputs": ["imports", "production_data", "emission_factors"],
      "supported_outputs": ["quarterly_report", "xml_export", "certificate_estimate"],
      "pricing_tier": "professional",
      "created_at": "2024-03-01T00:00:00Z",
      "updated_at": "2025-01-08T09:15:00Z"
    },
    {
      "id": "eudr_compliance",
      "name": "EUDR Compliance",
      "description": "EU Deforestation Regulation compliance verification with geolocation validation and due diligence statements",
      "version": "1.2.0",
      "category": "compliance",
      "status": "active",
      "supported_inputs": ["commodities", "geolocation", "supplier_data"],
      "supported_outputs": ["risk_assessment", "due_diligence_statement", "compliance_report"],
      "pricing_tier": "professional",
      "created_at": "2024-06-01T00:00:00Z",
      "updated_at": "2025-01-05T11:20:00Z"
    }
  ],
  "meta": {
    "request_id": "req_abc123xyz789",
    "timestamp": "2025-01-15T10:30:00Z"
  },
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total_items": 6,
    "total_pages": 1,
    "has_next": false,
    "has_prev": false
  }
}
```

**Code Examples:**

**Python:**

```python
import requests

headers = {
    "Authorization": f"Bearer {access_token}"
}

# List all active agents
response = requests.get(
    "https://api.greenlang.io/v1/agents",
    headers=headers,
    params={"status": "active", "category": "emissions"}
)

agents = response.json()["data"]

for agent in agents:
    print(f"{agent['id']}: {agent['name']} (v{agent['version']})")
```

**cURL:**

```bash
curl -X GET "https://api.greenlang.io/v1/agents?status=active" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

**JavaScript:**

```javascript
const response = await fetch(
  'https://api.greenlang.io/v1/agents?status=active',
  {
    headers: {
      'Authorization': `Bearer ${accessToken}`
    }
  }
);

const { data: agents } = await response.json();

agents.forEach(agent => {
  console.log(`${agent.id}: ${agent.name} (v${agent.version})`);
});
```

---

### Get Agent

Retrieve details for a specific agent.

```http
GET /v1/agents/{agent_id}
```

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | string | Yes | Unique agent identifier |

**Response (200 OK):**

```json
{
  "data": {
    "id": "fuel_emissions",
    "name": "Fuel Emissions Calculator",
    "description": "Calculate greenhouse gas emissions from fuel combustion using IPCC, EPA, or DEFRA emission factors",
    "version": "2.1.0",
    "category": "emissions",
    "status": "active",
    "methodology": {
      "name": "GHG Protocol",
      "version": "2023",
      "source": "https://ghgprotocol.org/calculation-tools"
    },
    "supported_inputs": [
      {
        "name": "fuel_type",
        "type": "string",
        "required": true,
        "description": "Type of fuel (e.g., diesel, gasoline, natural_gas)",
        "allowed_values": ["diesel", "gasoline", "natural_gas", "lpg", "coal", "fuel_oil", "kerosene", "biodiesel", "ethanol"]
      },
      {
        "name": "quantity",
        "type": "number",
        "required": true,
        "description": "Amount of fuel consumed",
        "minimum": 0
      },
      {
        "name": "unit",
        "type": "string",
        "required": true,
        "description": "Unit of measurement",
        "allowed_values": ["liters", "gallons", "kg", "tonnes", "cubic_meters", "mmbtu", "therms"]
      },
      {
        "name": "emission_factor_source",
        "type": "string",
        "required": false,
        "description": "Source for emission factors",
        "default": "ipcc_2023",
        "allowed_values": ["ipcc_2023", "epa_2024", "defra_2024", "custom"]
      },
      {
        "name": "custom_emission_factor",
        "type": "object",
        "required": false,
        "description": "Custom emission factor (required if emission_factor_source is 'custom')"
      }
    ],
    "supported_outputs": [
      {
        "name": "co2",
        "type": "number",
        "unit": "kg",
        "description": "Carbon dioxide emissions"
      },
      {
        "name": "ch4",
        "type": "number",
        "unit": "kg",
        "description": "Methane emissions"
      },
      {
        "name": "n2o",
        "type": "number",
        "unit": "kg",
        "description": "Nitrous oxide emissions"
      },
      {
        "name": "co2e",
        "type": "number",
        "unit": "kg",
        "description": "Total CO2 equivalent emissions"
      }
    ],
    "rate_limits": {
      "requests_per_minute": 100,
      "requests_per_day": 10000
    },
    "pricing": {
      "tier": "starter",
      "cost_per_execution": 0.001
    },
    "documentation_url": "https://docs.greenlang.io/agents/fuel-emissions",
    "created_at": "2024-01-15T00:00:00Z",
    "updated_at": "2025-01-10T14:30:00Z"
  },
  "meta": {
    "request_id": "req_def456uvw123",
    "timestamp": "2025-01-15T10:32:00Z"
  }
}
```

**Code Examples:**

**Python:**

```python
import requests

response = requests.get(
    "https://api.greenlang.io/v1/agents/fuel_emissions",
    headers={"Authorization": f"Bearer {access_token}"}
)

agent = response.json()["data"]

print(f"Agent: {agent['name']}")
print(f"Version: {agent['version']}")
print(f"Methodology: {agent['methodology']['name']}")

print("\nSupported Inputs:")
for input_field in agent["supported_inputs"]:
    required = "required" if input_field["required"] else "optional"
    print(f"  - {input_field['name']} ({input_field['type']}, {required})")
```

**cURL:**

```bash
curl -X GET "https://api.greenlang.io/v1/agents/fuel_emissions" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

**JavaScript:**

```javascript
const response = await fetch(
  'https://api.greenlang.io/v1/agents/fuel_emissions',
  {
    headers: {
      'Authorization': `Bearer ${accessToken}`
    }
  }
);

const { data: agent } = await response.json();

console.log(`Agent: ${agent.name}`);
console.log(`Methodology: ${agent.methodology.name}`);
console.log('Supported fuel types:', agent.supported_inputs
  .find(i => i.name === 'fuel_type')
  .allowed_values
  .join(', ')
);
```

**Error Response (404 Not Found):**

```json
{
  "error": {
    "code": "not_found",
    "message": "Agent 'unknown_agent' not found",
    "request_id": "req_xyz789abc456"
  }
}
```

---

### Execute Agent

Execute an agent with the provided input data.

```http
POST /v1/agents/{agent_id}/execute
```

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | string | Yes | Unique agent identifier |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | Bearer token or API key |
| `Content-Type` | Yes | Must be `application/json` |
| `X-Idempotency-Key` | No | Unique key for idempotent execution |

**Request Body:**

```json
{
  "inputs": {
    "fuel_type": "diesel",
    "quantity": 1000,
    "unit": "liters",
    "emission_factor_source": "ipcc_2023"
  },
  "options": {
    "include_breakdown": true,
    "output_unit": "kg",
    "async": false
  },
  "metadata": {
    "reference_id": "fleet-report-2025-q1",
    "tags": ["fleet", "transport", "q1-2025"]
  }
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `inputs` | object | Yes | Agent-specific input parameters |
| `options.include_breakdown` | boolean | No | Include detailed emissions breakdown (default: false) |
| `options.output_unit` | string | No | Output unit: `kg`, `tonnes`, `lbs` (default: `kg`) |
| `options.async` | boolean | No | Execute asynchronously (default: false) |
| `metadata.reference_id` | string | No | Your internal reference ID |
| `metadata.tags` | array | No | Tags for categorization |

**Response (200 OK - Synchronous):**

```json
{
  "data": {
    "execution_id": "exec_abc123xyz789",
    "agent_id": "fuel_emissions",
    "agent_version": "2.1.0",
    "status": "completed",
    "inputs": {
      "fuel_type": "diesel",
      "quantity": 1000,
      "unit": "liters",
      "emission_factor_source": "ipcc_2023"
    },
    "outputs": {
      "co2": 2680.5,
      "ch4": 0.12,
      "n2o": 0.08,
      "co2e": 2705.2,
      "unit": "kg"
    },
    "breakdown": {
      "emission_factor": {
        "co2": 2.68,
        "ch4": 0.00012,
        "n2o": 0.00008,
        "unit": "kg/liter",
        "source": "IPCC 2023 Guidelines",
        "gwp_values": {
          "ch4": 28,
          "n2o": 265,
          "source": "AR6"
        }
      },
      "calculation_steps": [
        {
          "step": "CO2 calculation",
          "formula": "quantity * emission_factor_co2",
          "values": "1000 * 2.68",
          "result": 2680.5
        },
        {
          "step": "CH4 to CO2e",
          "formula": "ch4 * gwp_ch4",
          "values": "0.12 * 28",
          "result": 3.36
        },
        {
          "step": "N2O to CO2e",
          "formula": "n2o * gwp_n2o",
          "values": "0.08 * 265",
          "result": 21.34
        }
      ]
    },
    "metadata": {
      "reference_id": "fleet-report-2025-q1",
      "tags": ["fleet", "transport", "q1-2025"]
    },
    "processing_time_ms": 45,
    "created_at": "2025-01-15T10:35:00Z",
    "completed_at": "2025-01-15T10:35:00Z"
  },
  "meta": {
    "request_id": "req_ghi789jkl012",
    "timestamp": "2025-01-15T10:35:00Z"
  }
}
```

**Response (202 Accepted - Asynchronous):**

```json
{
  "data": {
    "execution_id": "exec_def456uvw789",
    "agent_id": "cbam_reporter",
    "status": "processing",
    "progress": 0,
    "estimated_completion": "2025-01-15T10:40:00Z",
    "status_url": "/v1/executions/exec_def456uvw789"
  },
  "meta": {
    "request_id": "req_mno345pqr678",
    "timestamp": "2025-01-15T10:35:00Z"
  }
}
```

**Code Examples:**

**Python:**

```python
import requests

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

# Execute fuel emissions calculation
data = {
    "inputs": {
        "fuel_type": "diesel",
        "quantity": 1000,
        "unit": "liters",
        "emission_factor_source": "ipcc_2023"
    },
    "options": {
        "include_breakdown": True
    },
    "metadata": {
        "reference_id": "fleet-report-2025-q1"
    }
}

response = requests.post(
    "https://api.greenlang.io/v1/agents/fuel_emissions/execute",
    headers=headers,
    json=data
)

result = response.json()["data"]

print(f"Execution ID: {result['execution_id']}")
print(f"CO2e Emissions: {result['outputs']['co2e']} {result['outputs']['unit']}")
print(f"Processing Time: {result['processing_time_ms']}ms")
```

**cURL:**

```bash
curl -X POST "https://api.greenlang.io/v1/agents/fuel_emissions/execute" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "fuel_type": "diesel",
      "quantity": 1000,
      "unit": "liters",
      "emission_factor_source": "ipcc_2023"
    },
    "options": {
      "include_breakdown": true
    }
  }'
```

**JavaScript:**

```javascript
const response = await fetch(
  'https://api.greenlang.io/v1/agents/fuel_emissions/execute',
  {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${accessToken}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      inputs: {
        fuel_type: 'diesel',
        quantity: 1000,
        unit: 'liters',
        emission_factor_source: 'ipcc_2023'
      },
      options: {
        include_breakdown: true
      },
      metadata: {
        reference_id: 'fleet-report-2025-q1'
      }
    })
  }
);

const { data: result } = await response.json();

console.log(`Execution ID: ${result.execution_id}`);
console.log(`CO2e Emissions: ${result.outputs.co2e} ${result.outputs.unit}`);
```

**Error Response (422 Unprocessable Entity):**

```json
{
  "error": {
    "code": "validation_error",
    "message": "Input validation failed",
    "request_id": "req_err123xyz",
    "details": [
      {
        "field": "inputs.fuel_type",
        "message": "Invalid fuel type 'petrol'. Allowed values: diesel, gasoline, natural_gas, lpg, coal, fuel_oil, kerosene, biodiesel, ethanol",
        "code": "invalid_enum_value"
      },
      {
        "field": "inputs.quantity",
        "message": "Quantity must be a positive number",
        "code": "invalid_value"
      }
    ]
  }
}
```

---

### Get Execution

Retrieve the status and results of an agent execution.

```http
GET /v1/executions/{execution_id}
```

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `execution_id` | string | Yes | Unique execution identifier |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `include_inputs` | boolean | No | Include original inputs (default: false) |
| `include_breakdown` | boolean | No | Include calculation breakdown (default: false) |

**Response (200 OK - Completed):**

```json
{
  "data": {
    "execution_id": "exec_abc123xyz789",
    "agent_id": "fuel_emissions",
    "agent_version": "2.1.0",
    "status": "completed",
    "outputs": {
      "co2": 2680.5,
      "ch4": 0.12,
      "n2o": 0.08,
      "co2e": 2705.2,
      "unit": "kg"
    },
    "metadata": {
      "reference_id": "fleet-report-2025-q1",
      "tags": ["fleet", "transport", "q1-2025"]
    },
    "processing_time_ms": 45,
    "created_at": "2025-01-15T10:35:00Z",
    "completed_at": "2025-01-15T10:35:00Z"
  },
  "meta": {
    "request_id": "req_stu901vwx234",
    "timestamp": "2025-01-15T10:36:00Z"
  }
}
```

**Response (200 OK - Processing):**

```json
{
  "data": {
    "execution_id": "exec_def456uvw789",
    "agent_id": "cbam_reporter",
    "agent_version": "1.5.0",
    "status": "processing",
    "progress": 65,
    "current_step": "Calculating embedded emissions for 150 products",
    "steps_completed": 3,
    "steps_total": 5,
    "estimated_completion": "2025-01-15T10:38:00Z",
    "created_at": "2025-01-15T10:35:00Z"
  },
  "meta": {
    "request_id": "req_yza567bcd890",
    "timestamp": "2025-01-15T10:36:30Z"
  }
}
```

**Response (200 OK - Failed):**

```json
{
  "data": {
    "execution_id": "exec_ghi789jkl012",
    "agent_id": "eudr_compliance",
    "agent_version": "1.2.0",
    "status": "failed",
    "error": {
      "code": "calculation_error",
      "message": "Geolocation validation failed for 3 entries",
      "details": [
        {
          "row": 45,
          "field": "coordinates",
          "message": "Coordinates fall outside declared production region"
        },
        {
          "row": 89,
          "field": "coordinates",
          "message": "Invalid coordinate format"
        }
      ]
    },
    "created_at": "2025-01-15T10:35:00Z",
    "failed_at": "2025-01-15T10:35:45Z"
  },
  "meta": {
    "request_id": "req_efg123hij456",
    "timestamp": "2025-01-15T10:36:00Z"
  }
}
```

**Execution Status Values:**

| Status | Description |
|--------|-------------|
| `pending` | Execution queued, not yet started |
| `processing` | Execution in progress |
| `completed` | Execution completed successfully |
| `failed` | Execution failed with error |
| `cancelled` | Execution cancelled by user |
| `timeout` | Execution exceeded time limit |

**Code Examples:**

**Python:**

```python
import requests
import time

def wait_for_execution(execution_id, timeout=300, poll_interval=5):
    """Poll execution status until complete or timeout."""
    headers = {"Authorization": f"Bearer {access_token}"}
    start_time = time.time()

    while time.time() - start_time < timeout:
        response = requests.get(
            f"https://api.greenlang.io/v1/executions/{execution_id}",
            headers=headers,
            params={"include_breakdown": True}
        )

        result = response.json()["data"]
        status = result["status"]

        if status == "completed":
            return result
        elif status == "failed":
            raise Exception(f"Execution failed: {result['error']['message']}")
        elif status == "processing":
            progress = result.get("progress", 0)
            print(f"Processing... {progress}%")
            time.sleep(poll_interval)
        else:
            time.sleep(poll_interval)

    raise TimeoutError("Execution timed out")

# Usage
result = wait_for_execution("exec_def456uvw789")
print(f"Completed! CO2e: {result['outputs']['co2e']} kg")
```

**cURL:**

```bash
curl -X GET "https://api.greenlang.io/v1/executions/exec_abc123xyz789?include_breakdown=true" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

**JavaScript:**

```javascript
async function waitForExecution(executionId, timeout = 300000, pollInterval = 5000) {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const response = await fetch(
      `https://api.greenlang.io/v1/executions/${executionId}?include_breakdown=true`,
      {
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      }
    );

    const { data: result } = await response.json();

    if (result.status === 'completed') {
      return result;
    } else if (result.status === 'failed') {
      throw new Error(`Execution failed: ${result.error.message}`);
    }

    console.log(`Processing... ${result.progress || 0}%`);
    await new Promise(resolve => setTimeout(resolve, pollInterval));
  }

  throw new Error('Execution timed out');
}

// Usage
const result = await waitForExecution('exec_def456uvw789');
console.log(`Completed! CO2e: ${result.outputs.co2e} kg`);
```

---

### List Executions

Retrieve execution history for your account.

```http
GET /v1/executions
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | string | No | Filter by agent ID |
| `status` | string | No | Filter by status |
| `from` | string | No | Start date (ISO 8601) |
| `to` | string | No | End date (ISO 8601) |
| `reference_id` | string | No | Filter by reference ID |
| `tags` | string | No | Filter by tags (comma-separated) |
| `page` | integer | No | Page number (default: 1) |
| `per_page` | integer | No | Items per page (default: 20) |

**Response (200 OK):**

```json
{
  "data": [
    {
      "execution_id": "exec_abc123xyz789",
      "agent_id": "fuel_emissions",
      "status": "completed",
      "outputs": {
        "co2e": 2705.2,
        "unit": "kg"
      },
      "metadata": {
        "reference_id": "fleet-report-2025-q1"
      },
      "created_at": "2025-01-15T10:35:00Z",
      "completed_at": "2025-01-15T10:35:00Z"
    },
    {
      "execution_id": "exec_def456uvw789",
      "agent_id": "cbam_reporter",
      "status": "completed",
      "outputs": {
        "total_embedded_emissions": 15420.5,
        "products_processed": 150
      },
      "metadata": {
        "reference_id": "cbam-q4-2024"
      },
      "created_at": "2025-01-14T14:20:00Z",
      "completed_at": "2025-01-14T14:25:00Z"
    }
  ],
  "meta": {
    "request_id": "req_klm789nop012",
    "timestamp": "2025-01-15T10:40:00Z"
  },
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total_items": 156,
    "total_pages": 8,
    "has_next": true,
    "has_prev": false
  }
}
```

**Example - List recent fuel emissions calculations:**

```python
import requests

response = requests.get(
    "https://api.greenlang.io/v1/executions",
    headers={"Authorization": f"Bearer {access_token}"},
    params={
        "agent_id": "fuel_emissions",
        "status": "completed",
        "from": "2025-01-01T00:00:00Z",
        "per_page": 50
    }
)

executions = response.json()["data"]
total_emissions = sum(e["outputs"]["co2e"] for e in executions)
print(f"Total CO2e this month: {total_emissions:,.2f} kg")
```

---

### Cancel Execution

Cancel a pending or processing execution.

```http
POST /v1/executions/{execution_id}/cancel
```

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `execution_id` | string | Yes | Unique execution identifier |

**Response (200 OK):**

```json
{
  "data": {
    "execution_id": "exec_def456uvw789",
    "status": "cancelled",
    "cancelled_at": "2025-01-15T10:37:00Z"
  },
  "meta": {
    "request_id": "req_qrs345tuv678",
    "timestamp": "2025-01-15T10:37:00Z"
  }
}
```

**Error Response (409 Conflict):**

```json
{
  "error": {
    "code": "invalid_state",
    "message": "Cannot cancel execution with status 'completed'",
    "request_id": "req_wxy901zab234"
  }
}
```

---

## Batch Execution

Execute an agent with multiple input sets in a single request.

```http
POST /v1/agents/{agent_id}/batch
```

**Request Body:**

```json
{
  "items": [
    {
      "id": "item_001",
      "inputs": {
        "fuel_type": "diesel",
        "quantity": 500,
        "unit": "liters"
      }
    },
    {
      "id": "item_002",
      "inputs": {
        "fuel_type": "gasoline",
        "quantity": 300,
        "unit": "liters"
      }
    },
    {
      "id": "item_003",
      "inputs": {
        "fuel_type": "natural_gas",
        "quantity": 1000,
        "unit": "cubic_meters"
      }
    }
  ],
  "options": {
    "continue_on_error": true
  }
}
```

**Response (200 OK):**

```json
{
  "data": {
    "batch_id": "batch_xyz789abc123",
    "status": "completed",
    "total_items": 3,
    "successful": 3,
    "failed": 0,
    "results": [
      {
        "id": "item_001",
        "status": "completed",
        "outputs": {
          "co2e": 1352.6,
          "unit": "kg"
        }
      },
      {
        "id": "item_002",
        "status": "completed",
        "outputs": {
          "co2e": 693.0,
          "unit": "kg"
        }
      },
      {
        "id": "item_003",
        "status": "completed",
        "outputs": {
          "co2e": 1980.5,
          "unit": "kg"
        }
      }
    ],
    "summary": {
      "total_co2e": 4026.1,
      "unit": "kg"
    },
    "processing_time_ms": 125
  },
  "meta": {
    "request_id": "req_cde567fgh890",
    "timestamp": "2025-01-15T10:45:00Z"
  }
}
```

---

## Webhooks for Agent Executions

Subscribe to execution events:

```http
POST /v1/webhooks
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/greenlang",
  "events": [
    "execution.started",
    "execution.completed",
    "execution.failed"
  ],
  "filters": {
    "agent_ids": ["cbam_reporter", "eudr_compliance"]
  }
}
```

**Webhook Payload:**

```json
{
  "event": "execution.completed",
  "timestamp": "2025-01-15T10:35:00Z",
  "data": {
    "execution_id": "exec_abc123xyz789",
    "agent_id": "fuel_emissions",
    "status": "completed",
    "outputs": {
      "co2e": 2705.2,
      "unit": "kg"
    },
    "metadata": {
      "reference_id": "fleet-report-2025-q1"
    }
  }
}
```

---

## Next Steps

- [Calculation Endpoints](./calculations.md) - Direct calculation APIs
- [EUDR Compliance Guide](../guides/eudr_compliance.md) - EU Deforestation Regulation
- [CBAM Reporting Guide](../guides/cbam_reporting.md) - Carbon Border Adjustment Mechanism
