# Quickstart Guide - Get Started in 15 Minutes

## GL-VCCI Platform Integration

This quickstart guide will get you calculating Scope 3 emissions in under 15 minutes.

---

## Prerequisites

- GL-VCCI account (sign up at https://vcci.greenlang.io)
- API key or OAuth credentials
- Basic knowledge of REST APIs
- Development environment (Python, Node.js, or cURL)

---

## Step 1: Get Your API Key (2 minutes)

### Via Web Dashboard

1. Log in to https://app.vcci.greenlang.io
2. Navigate to **Settings** → **API Keys**
3. Click **Create API Key**
4. Name it "Quickstart Test"
5. Copy the key (starts with `sk_live_` or `sk_test_`)

⚠️ **Important:** Store your API key securely. It will only be shown once.

### Environment Setup

```bash
# Create .env file
echo "VCCI_API_KEY=sk_live_your_key_here" > .env

# Add to .gitignore
echo ".env" >> .gitignore
```

---

## Step 2: Install Dependencies (1 minute)

### Python

```bash
pip install requests python-dotenv
```

### Node.js

```bash
npm install axios dotenv
```

### No Installation (cURL)

Skip to Step 3.

---

## Step 3: Make Your First API Call (2 minutes)

### Test Connection

#### Python

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('VCCI_API_KEY')
BASE_URL = "https://api.vcci.greenlang.io/v2"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Health check
response = requests.get(f"{BASE_URL}/health")
print("API Status:", response.json())

# List suppliers (should be empty initially)
response = requests.get(f"{BASE_URL}/suppliers", headers=headers)
print("Suppliers:", response.json())
```

#### Node.js

```javascript
require('dotenv').config();
const axios = require('axios');

const API_KEY = process.env.VCCI_API_KEY;
const BASE_URL = 'https://api.vcci.greenlang.io/v2';

const headers = {
  'X-API-Key': API_KEY,
  'Content-Type': 'application/json'
};

// Health check
axios.get(`${BASE_URL}/health`)
  .then(response => console.log('API Status:', response.data));

// List suppliers
axios.get(`${BASE_URL}/suppliers`, { headers })
  .then(response => console.log('Suppliers:', response.data));
```

#### cURL

```bash
export VCCI_API_KEY="sk_live_your_key_here"

# Health check
curl https://api.vcci.greenlang.io/v2/health

# List suppliers
curl -H "X-API-Key: $VCCI_API_KEY" \
  https://api.vcci.greenlang.io/v2/suppliers
```

---

## Step 4: Create a Supplier (2 minutes)

Create your first supplier entity:

#### Python

```python
supplier_data = {
    "canonical_name": "Acme Steel Corporation",
    "aliases": ["ACME Steel", "Acme Corp"],
    "identifiers": {
        "duns": "123456789"
    },
    "enrichment": {
        "headquarters": {
            "country": "US",
            "city": "Pittsburgh"
        },
        "industry": "Steel Manufacturing"
    }
}

response = requests.post(
    f"{BASE_URL}/suppliers",
    headers=headers,
    json=supplier_data
)

supplier = response.json()
print(f"Created supplier: {supplier['id']}")
print(f"Name: {supplier['canonical_name']}")
```

#### Node.js

```javascript
const supplierData = {
  canonical_name: "Acme Steel Corporation",
  aliases: ["ACME Steel", "Acme Corp"],
  identifiers: {
    duns: "123456789"
  },
  enrichment: {
    headquarters: {
      country: "US",
      city: "Pittsburgh"
    },
    industry: "Steel Manufacturing"
  }
};

axios.post(`${BASE_URL}/suppliers`, supplierData, { headers })
  .then(response => {
    const supplier = response.data;
    console.log(`Created supplier: ${supplier.id}`);
    console.log(`Name: ${supplier.canonical_name}`);
  });
```

#### cURL

```bash
curl -X POST https://api.vcci.greenlang.io/v2/suppliers \
  -H "X-API-Key: $VCCI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "canonical_name": "Acme Steel Corporation",
    "aliases": ["ACME Steel", "Acme Corp"],
    "identifiers": {
      "duns": "123456789"
    }
  }'
```

---

## Step 5: Create a Procurement Transaction (3 minutes)

Record a purchase from the supplier:

#### Python

```python
transaction_data = {
    "transaction_type": "purchase_order",
    "transaction_id": "PO-2024-001",
    "transaction_date": "2024-11-01",
    "supplier_name": "Acme Steel Corporation",
    "product_name": "Hot rolled steel coil",
    "quantity": 1000,
    "unit": "kg",
    "spend_usd": 5000.00,
    "currency": "USD"
}

response = requests.post(
    f"{BASE_URL}/procurement/transactions",
    headers=headers,
    json=transaction_data
)

transaction = response.json()
print(f"Created transaction: {transaction['id']}")
print(f"Status: {transaction['status']}")
```

#### Node.js

```javascript
const transactionData = {
  transaction_type: "purchase_order",
  transaction_id: "PO-2024-001",
  transaction_date: "2024-11-01",
  supplier_name: "Acme Steel Corporation",
  product_name: "Hot rolled steel coil",
  quantity: 1000,
  unit: "kg",
  spend_usd: 5000.00,
  currency: "USD"
};

axios.post(`${BASE_URL}/procurement/transactions`, transactionData, { headers })
  .then(response => {
    const transaction = response.data;
    console.log(`Created transaction: ${transaction.id}`);
    console.log(`Status: ${transaction.status}`);
  });
```

#### cURL

```bash
curl -X POST https://api.vcci.greenlang.io/v2/procurement/transactions \
  -H "X-API-Key: $VCCI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_type": "purchase_order",
    "transaction_id": "PO-2024-001",
    "transaction_date": "2024-11-01",
    "supplier_name": "Acme Steel Corporation",
    "product_name": "Hot rolled steel coil",
    "quantity": 1000,
    "unit": "kg",
    "spend_usd": 5000.00,
    "currency": "USD"
  }'
```

---

## Step 6: Calculate Emissions (3 minutes)

Calculate Scope 3 emissions for the transaction:

#### Python

```python
calculation_request = {
    "transactions": [
        {"transaction_id": transaction['id']}
    ],
    "options": {
        "gwp_standard": "AR6",
        "uncertainty_method": "monte_carlo",
        "monte_carlo_iterations": 10000
    }
}

response = requests.post(
    f"{BASE_URL}/emissions/calculate",
    headers=headers,
    json=calculation_request
)

results = response.json()
calc = results['results'][0]

print("\n🎉 Emission Calculation Results:")
print(f"  Emissions: {calc['emissions_kg_co2e']:.2f} kg CO2e")
print(f"  Category: {calc['category']}")
print(f"  Tier: {calc['tier']}")
print(f"  Uncertainty: ±{calc['uncertainty']['range']['lower_bound']:.2f} to {calc['uncertainty']['range']['upper_bound']:.2f}")
print(f"  DQI Score: {calc['data_quality']['dqi_score']:.2f}/5.0")
print(f"  Pedigree: {calc['data_quality']['pedigree']}")
```

#### Node.js

```javascript
const calculationRequest = {
  transactions: [
    { transaction_id: transaction.id }
  ],
  options: {
    gwp_standard: "AR6",
    uncertainty_method: "monte_carlo",
    monte_carlo_iterations: 10000
  }
};

axios.post(`${BASE_URL}/emissions/calculate`, calculationRequest, { headers })
  .then(response => {
    const calc = response.data.results[0];

    console.log('\n🎉 Emission Calculation Results:');
    console.log(`  Emissions: ${calc.emissions_kg_co2e.toFixed(2)} kg CO2e`);
    console.log(`  Category: ${calc.category}`);
    console.log(`  Tier: ${calc.tier}`);
    console.log(`  Uncertainty: ±${calc.uncertainty.range.lower_bound.toFixed(2)} to ${calc.uncertainty.range.upper_bound.toFixed(2)}`);
    console.log(`  DQI Score: ${calc.data_quality.dqi_score.toFixed(2)}/5.0`);
    console.log(`  Pedigree: ${calc.data_quality.pedigree}`);
  });
```

---

## Step 7: View Aggregated Emissions (2 minutes)

Get aggregated emissions data:

#### Python

```python
params = {
    "date_from": "2024-01-01",
    "date_to": "2024-12-31",
    "group_by": "category,supplier"
}

response = requests.get(
    f"{BASE_URL}/emissions/aggregate",
    headers=headers,
    params=params
)

aggregate = response.json()
print("\n📊 Aggregated Emissions:")
print(f"  Total: {aggregate['summary']['total_emissions_kg_co2e']:.2f} kg CO2e")
print(f"  Period: {aggregate['summary']['date_from']} to {aggregate['summary']['date_to']}")
```

---

## Complete Example Script

Save this as `quickstart.py`:

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('VCCI_API_KEY')
BASE_URL = "https://api.vcci.greenlang.io/v2"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def main():
    print("GL-VCCI Platform Quickstart\n")

    # 1. Create supplier
    print("1. Creating supplier...")
    supplier_data = {
        "canonical_name": "Acme Steel Corporation",
        "aliases": ["ACME Steel"],
        "enrichment": {
            "headquarters": {"country": "US", "city": "Pittsburgh"},
            "industry": "Steel Manufacturing"
        }
    }
    supplier = requests.post(
        f"{BASE_URL}/suppliers",
        headers=headers,
        json=supplier_data
    ).json()
    print(f"   ✓ Created: {supplier['canonical_name']} ({supplier['id']})\n")

    # 2. Create transaction
    print("2. Creating procurement transaction...")
    transaction_data = {
        "transaction_type": "purchase_order",
        "transaction_id": "PO-2024-001",
        "transaction_date": "2024-11-01",
        "supplier_name": "Acme Steel Corporation",
        "product_name": "Hot rolled steel coil",
        "quantity": 1000,
        "unit": "kg",
        "spend_usd": 5000.00
    }
    transaction = requests.post(
        f"{BASE_URL}/procurement/transactions",
        headers=headers,
        json=transaction_data
    ).json()
    print(f"   ✓ Created: {transaction['id']}\n")

    # 3. Calculate emissions
    print("3. Calculating emissions...")
    calc_request = {
        "transactions": [{"transaction_id": transaction['id']}],
        "options": {"gwp_standard": "AR6", "uncertainty_method": "monte_carlo"}
    }
    results = requests.post(
        f"{BASE_URL}/emissions/calculate",
        headers=headers,
        json=calc_request
    ).json()

    calc = results['results'][0]
    print(f"   ✓ Emissions: {calc['emissions_kg_co2e']:.2f} kg CO2e")
    print(f"   ✓ Tier: {calc['tier']}")
    print(f"   ✓ DQI Score: {calc['data_quality']['dqi_score']:.2f}/5.0\n")

    print("🎉 Quickstart complete! You've successfully:")
    print("   - Created a supplier")
    print("   - Recorded a procurement transaction")
    print("   - Calculated Scope 3 emissions")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python quickstart.py
```

---

## Next Steps

### Learn More

- [API Reference](../API_REFERENCE.md) - Complete API documentation
- [Authentication Guide](../AUTHENTICATION.md) - OAuth and API keys
- [Python SDK](./PYTHON_SDK.md) - Official Python SDK
- [JavaScript SDK](./JAVASCRIPT_SDK.md) - Official JS/Node SDK

### Build Your Integration

1. **Bulk Import:** Upload CSV/Excel files with procurement data
2. **ERP Integration:** Connect SAP, Oracle, or Workday
3. **Reporting:** Generate ESRS, CDP, IFRS S2 reports
4. **Supplier Engagement:** Launch PCF request campaigns
5. **Webhooks:** Set up real-time event notifications

### Get Support

- **Documentation:** https://docs.vcci.greenlang.io
- **API Status:** https://status.vcci.greenlang.io
- **Support Email:** support@greenlang.io
- **Community Slack:** https://community.greenlang.io

---

**Last Updated:** November 6, 2025
**Version:** 2.0.0
