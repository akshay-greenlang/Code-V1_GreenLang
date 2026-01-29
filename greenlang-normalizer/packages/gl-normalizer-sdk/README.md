# GreenLang Normalizer SDK

Python SDK for the GreenLang Unit & Reference Normalizer (GL-FOUND-X-003).

## Installation

```bash
pip install gl-normalizer-sdk
```

## Quick Start

### Basic Usage

```python
from gl_normalizer import NormalizerClient

# Initialize the client
client = NormalizerClient(api_key="your-api-key")

# Normalize a single value
result = client.normalize(100, "kWh", target_unit="MJ")
print(result.canonical_value)  # 360.0
print(result.canonical_unit)   # "MJ"

# With context
result = client.normalize(
    value=1500,
    unit="kWh",
    target_unit="MJ",
    expected_dimension="energy",
    field="energy_consumption"
)
```

### Batch Normalization

```python
from gl_normalizer import NormalizerClient, NormalizeRequest, BatchMode

client = NormalizerClient(api_key="your-api-key")

# Create batch requests
requests = [
    NormalizeRequest(value=100, unit="kWh", expected_dimension="energy"),
    NormalizeRequest(value=50, unit="kg", expected_dimension="mass"),
    NormalizeRequest(value=1000, unit="L", expected_dimension="volume"),
]

# Process batch (up to 10K items)
result = client.normalize_batch(requests, mode=BatchMode.PARTIAL)

print(f"Success: {result.summary.success}")
print(f"Failed: {result.summary.failed}")

for item in result.results:
    if item.status == "success":
        print(f"{item.field}: {item.canonical_value} {item.canonical_unit}")
```

### Async Jobs for Large Datasets

```python
from gl_normalizer import NormalizerClient, NormalizeRequest
import time

client = NormalizerClient(api_key="your-api-key")

# Create job for 100K+ items
requests = [NormalizeRequest(value=i, unit="kWh") for i in range(100000)]
job = client.create_job(requests)

print(f"Job created: {job.job_id}")

# Poll for completion
while job.status not in ("completed", "failed"):
    time.sleep(5)
    job = client.get_job(job.job_id)
    print(f"Progress: {job.progress}%")

# Get results
if job.status == "completed":
    results = job.results
```

### Async Client

```python
import asyncio
from gl_normalizer import AsyncNormalizerClient

async def main():
    async with AsyncNormalizerClient(api_key="your-api-key") as client:
        # Normalize single value
        result = await client.normalize(100, "kWh", target_unit="MJ")
        print(result.canonical_value)

        # Batch normalize
        requests = [
            NormalizeRequest(value=100, unit="kWh"),
            NormalizeRequest(value=50, unit="kg"),
        ]
        batch_result = await client.normalize_batch(requests)

asyncio.run(main())
```

### List Available Vocabularies

```python
from gl_normalizer import NormalizerClient

client = NormalizerClient(api_key="your-api-key")

vocabularies = client.list_vocabularies()
for vocab in vocabularies:
    print(f"{vocab.name} v{vocab.version} - {vocab.entity_count} entities")
```

## Configuration

### Client Options

```python
from gl_normalizer import NormalizerClient, ClientConfig

config = ClientConfig(
    timeout=30.0,           # Request timeout in seconds
    max_retries=3,          # Maximum retry attempts
    retry_delay=1.0,        # Initial retry delay
    retry_max_delay=30.0,   # Maximum retry delay
    enable_cache=True,      # Enable response caching
    cache_ttl=300,          # Cache TTL in seconds
)

client = NormalizerClient(
    api_key="your-api-key",
    base_url="https://api.greenlang.io",  # Custom base URL
    config=config
)
```

### Vocabulary Version Pinning

```python
# Pin to specific vocabulary version for reproducibility
result = client.normalize(
    100, "kWh",
    vocabulary_version="2026.01.0"
)
```

### Policy Modes

```python
from gl_normalizer import PolicyMode

# STRICT mode - fail on any ambiguity
result = client.normalize(
    100, "kWh",
    policy_mode=PolicyMode.STRICT
)

# LENIENT mode - return warnings instead of failures
result = client.normalize(
    100, "kWh",
    policy_mode=PolicyMode.LENIENT
)
```

## Error Handling

```python
from gl_normalizer import (
    NormalizerClient,
    NormalizerError,
    ValidationError,
    ConversionError,
    RateLimitError,
    APIError,
)

client = NormalizerClient(api_key="your-api-key")

try:
    result = client.normalize(100, "invalid_unit")
except ValidationError as e:
    print(f"Validation failed: {e.code} - {e.message}")
    print(f"Hint: {e.hint}")
except ConversionError as e:
    print(f"Conversion failed: {e.code} - {e.message}")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after} seconds")
except APIError as e:
    print(f"API error: {e.status_code} - {e.message}")
except NormalizerError as e:
    print(f"General error: {e}")
```

## Features

- **Automatic Retry**: Exponential backoff with jitter for transient failures
- **Connection Pooling**: Efficient HTTP connection reuse
- **Response Caching**: Optional caching for repeated requests
- **Type Safety**: Full type hints and mypy compliance
- **Async Support**: Native async/await with `AsyncNormalizerClient`
- **Batch Processing**: Process up to 10K items in a single request
- **Async Jobs**: Handle 100K+ items with job-based processing

## API Reference

See the [API Documentation](https://docs.greenlang.io/normalizer/sdk/api) for complete reference.

## License

Proprietary - GreenLang Technologies
