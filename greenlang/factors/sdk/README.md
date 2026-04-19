# GreenLang Factors Python SDK

Python SDK for the [GreenLang Emission Factors API](https://greenlang.io). Query, search, match, and calculate with 327+ audited emission factors across US, EU, UK, and more -- all with full provenance tracking and data quality scoring.

**Zero dependencies** -- uses only the Python standard library.

## Installation

```bash
pip install greenlang-factors-sdk
```

Requires Python 3.9 or later.

## Quick Start

```python
from greenlang_factors_sdk import FactorsClient, FactorsConfig

# Initialize the client
client = FactorsClient(FactorsConfig(
    base_url="https://api.greenlang.io/api/v1",
    api_key="gl_your_api_key_here",
))

# Search for emission factors
results = client.search("diesel combustion US")
for factor in results["factors"]:
    print(f"{factor['factor_id']}: {factor['co2e_per_unit']} kgCO2e/{factor['unit']}")
```

## Authentication

The SDK supports two authentication methods:

### API Key (recommended)

```python
client = FactorsClient(FactorsConfig(
    base_url="https://api.greenlang.io/api/v1",
    api_key="gl_your_api_key_here",
))
```

### JWT Token

```python
client = FactorsClient(FactorsConfig(
    base_url="https://api.greenlang.io/api/v1",
    api_key="eyJhbGciOiJIUzI1NiIs...",  # JWT token
))
```

## Configuration

```python
config = FactorsConfig(
    base_url="https://api.greenlang.io/api/v1",  # API base URL
    api_key="gl_...",          # API key or JWT token
    edition="2026-Q1",         # Pin to a specific catalog edition
    timeout=60,                # Request timeout in seconds (default: 60)
    max_retries=3,             # Number of retries on transient errors (default: 3)
    retry_backoff=1.0,         # Initial backoff in seconds (default: 1.0)
)
```

## Common Operations

### Search Factors

```python
# Basic search
results = client.search("natural gas")

# Search with filters
results = client.search(
    "electricity grid",
    geography="EU",
    limit=10,
    include_preview=False,
)
```

### Get Factor Details

```python
# Get a specific factor by ID
factor = client.get_factor("ef_us_diesel_scope1_v2")

# Get provenance and license info
provenance = client.get_provenance("ef_us_diesel_scope1_v2")

# Get deprecation replacement chain
replacements = client.get_replacements("ef_us_diesel_scope1_v1")
```

### Match Activity to Factors

```python
# Find the best matching factors for an activity description
matches = client.match(
    activity_description="Burning 500 gallons of diesel fuel in a generator",
    geography="US",
    scope="1",
    limit=5,
)
for candidate in matches["candidates"]:
    print(f"{candidate['factor_id']} (score: {candidate['score']:.2f})")
```

### Calculate Emissions

```python
# Single calculation
result = client.calculate(
    fuel_type="diesel",
    activity_amount=1000.0,
    activity_unit="gallons",
    geography="US",
    scope="1",
    boundary="combustion",
)
print(f"Emissions: {result['emissions_tonnes_co2e']:.2f} tCO2e")

# Batch calculation
batch_result = client.calculate_batch([
    {
        "fuel_type": "diesel",
        "activity_amount": 500.0,
        "activity_unit": "gallons",
        "geography": "US",
        "scope": "1",
    },
    {
        "fuel_type": "natural_gas",
        "activity_amount": 10000.0,
        "activity_unit": "therms",
        "geography": "US",
        "scope": "1",
    },
])
print(f"Total: {batch_result['total_emissions_tonnes_co2e']:.2f} tCO2e")
```

### List and Compare Editions

```python
# List available catalog editions
editions = client.list_editions()
for ed in editions["editions"]:
    print(f"{ed['edition_id']} ({ed['status']})")

# Compare two editions
diff = client.compare_editions("2025-Q4", "2026-Q1")
print(f"Added: {len(diff.get('added', []))}")
print(f"Removed: {len(diff.get('removed', []))}")
print(f"Changed: {len(diff.get('changed', []))}")
```

### Edition Pinning

Pin your application to a specific catalog edition for reproducible results:

```python
# Pin at client level
client = FactorsClient(FactorsConfig(
    base_url="https://api.greenlang.io/api/v1",
    api_key="gl_...",
    edition="2026-Q1",  # All requests use this edition
))

# The X-Factors-Edition header is sent automatically
results = client.search("diesel US")
```

### System and Coverage

```python
# Health check
health = client.health()
print(f"Status: {health['status']}")

# API statistics
stats = client.stats()
print(f"Total factors: {stats['total_factors']}")

# Coverage breakdown
coverage = client.coverage()
print(f"Geographies: {coverage['geographies']}")
print(f"Fuel types: {coverage['fuel_types']}")
```

## Error Handling

```python
from greenlang_factors_sdk import FactorsClient, FactorsConfig, FactorsApiError, FactorsConnectionError

client = FactorsClient(FactorsConfig(
    base_url="https://api.greenlang.io/api/v1",
    api_key="gl_...",
))

try:
    factor = client.get_factor("nonexistent_id")
except FactorsApiError as e:
    print(f"API error: HTTP {e.status_code} - {e.message}")
    if e.status_code == 404:
        print("Factor not found")
    elif e.status_code == 429:
        print("Rate limit exceeded, try again later")
except FactorsConnectionError as e:
    print(f"Connection error: {e}")
```

The SDK automatically retries on transient errors (HTTP 429, 500, 502, 503, 504) with exponential backoff.

## Tiers

| Tier | Price | Monthly Quota | Overage |
|------|-------|---------------|---------|
| Community | Free | 1,000 requests | N/A |
| Pro | $299/mo | 50,000 requests | $0.01/req |
| Enterprise | $999/mo | 500,000 requests | $0.005/req |

## License

MIT License. See [LICENSE](LICENSE) for details.
