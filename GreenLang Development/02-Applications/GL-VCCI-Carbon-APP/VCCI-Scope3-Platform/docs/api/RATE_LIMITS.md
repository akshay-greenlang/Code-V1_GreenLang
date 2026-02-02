# Rate Limits & Best Practices

## GL-VCCI API Rate Limiting Guide

This document covers rate limiting policies, optimization strategies, and best practices for efficient API usage.

---

## Rate Limit Tiers

### Standard Plan
- **Rate:** 1,000 requests/hour per tenant
- **Burst:** 20 requests/second
- **Best for:** Small to medium deployments

### Premium Plan
- **Rate:** 10,000 requests/hour per tenant
- **Burst:** 50 requests/second
- **Best for:** Large enterprises

### Enterprise Plan
- **Rate:** Custom limits (contact sales)
- **Burst:** Custom
- **Best for:** High-volume integrations

---

## Rate Limit Headers

Every API response includes rate limit information:

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1699275600
```

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests allowed per hour |
| `X-RateLimit-Remaining` | Requests remaining in current window |
| `X-RateLimit-Reset` | Unix timestamp when limit resets |

---

## Handling Rate Limits

### 429 Too Many Requests Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please retry after 60 seconds."
  }
}
```

Response headers:
```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1699275600
Retry-After: 60
```

### Exponential Backoff Strategy

```python
import time
import requests

def make_request_with_backoff(url, headers, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)

        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            wait_time = min(retry_after, 2 ** attempt)
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
            continue

        return response

    raise Exception("Max retries exceeded")
```

---

## Optimization Strategies

### 1. Use Batch Endpoints

Instead of:
```python
# ❌ Bad: Individual requests (1000 requests)
for transaction in transactions[:1000]:
    create_transaction(transaction)
```

Do this:
```python
# ✅ Good: Batch request (1 request)
batch_create_transactions(transactions[:1000])
```

**Savings:** 1000 requests → 1 request (99.9% reduction)

### 2. Implement Caching

```python
import redis
import json
import hashlib

cache = redis.Redis(host='localhost', port=6379, db=0)

def get_supplier_cached(supplier_id, ttl=3600):
    cache_key = f"supplier:{supplier_id}"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Fetch from API
    supplier = api.get_supplier(supplier_id)

    # Cache result
    cache.setex(cache_key, ttl, json.dumps(supplier))

    return supplier
```

### 3. Pagination Best Practices

```python
# ✅ Good: Fetch all suppliers efficiently
def get_all_suppliers():
    all_suppliers = []
    offset = 0
    limit = 1000  # Maximum allowed

    while True:
        response = api.list_suppliers(limit=limit, offset=offset)
        all_suppliers.extend(response['data'])

        if not response['pagination']['has_more']:
            break

        offset += limit

    return all_suppliers
```

### 4. Webhooks Instead of Polling

Instead of polling for job status:
```python
# ❌ Bad: Polling (many requests)
while True:
    status = api.get_job_status(job_id)
    if status['status'] == 'completed':
        break
    time.sleep(30)  # Poll every 30 seconds
```

Use webhooks:
```python
# ✅ Good: Webhook (zero polling requests)
@app.route('/webhooks/job-completed', methods=['POST'])
def job_completed():
    job_data = request.json
    process_completed_job(job_data)
    return '', 200
```

### 5. Parallel Requests with Rate Limiting

```python
import asyncio
import aiohttp
from asyncio import Semaphore

async def fetch_with_semaphore(session, url, headers, semaphore):
    async with semaphore:
        async with session.get(url, headers=headers) as response:
            return await response.json()

async def fetch_all_suppliers(supplier_ids):
    # Limit to 20 concurrent requests
    semaphore = Semaphore(20)

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_with_semaphore(
                session,
                f"{BASE_URL}/suppliers/{sid}",
                headers,
                semaphore
            )
            for sid in supplier_ids
        ]

        return await asyncio.gather(*tasks)

# Usage
supplier_ids = ['sup_1', 'sup_2', ..., 'sup_1000']
suppliers = asyncio.run(fetch_all_suppliers(supplier_ids))
```

---

## Request Optimization

### Minimize Response Payload

Use field selection (when available):
```http
GET /v2/suppliers?fields=id,canonical_name,status
```

### Use Compression

```python
headers = {
    'Accept-Encoding': 'gzip, deflate',
    'X-API-Key': API_KEY
}
```

### Conditional Requests

```python
# First request
response = requests.get(url, headers=headers)
etag = response.headers.get('ETag')

# Subsequent requests
headers['If-None-Match'] = etag
response = requests.get(url, headers=headers)

if response.status_code == 304:
    # Not modified - use cached data
    pass
```

---

## Monitoring Usage

### Track Your Rate Limit

```python
class RateLimitTracker:
    def __init__(self):
        self.limit = None
        self.remaining = None
        self.reset_time = None

    def update(self, headers):
        self.limit = int(headers.get('X-RateLimit-Limit', 0))
        self.remaining = int(headers.get('X-RateLimit-Remaining', 0))
        self.reset_time = int(headers.get('X-RateLimit-Reset', 0))

    def get_usage_percentage(self):
        if self.limit:
            return ((self.limit - self.remaining) / self.limit) * 100
        return 0

    def should_throttle(self, threshold=90):
        return self.get_usage_percentage() > threshold

# Usage
tracker = RateLimitTracker()

response = requests.get(url, headers=headers)
tracker.update(response.headers)

if tracker.should_throttle():
    print("Approaching rate limit. Throttling requests...")
    time.sleep(10)
```

---

## Best Practices Summary

### DO ✅

- Use batch endpoints for bulk operations
- Implement exponential backoff for retries
- Cache frequently accessed data
- Use webhooks instead of polling
- Monitor rate limit headers
- Request only the data you need
- Use compression
- Parallelize requests with controlled concurrency

### DON'T ❌

- Make individual requests in loops
- Poll continuously without backoff
- Ignore `Retry-After` headers
- Request unnecessary data
- Create new connections for each request
- Exceed burst limits
- Ignore rate limit warnings

---

## Rate Limit Exemptions

Certain endpoints have different limits:

| Endpoint Category | Rate Limit |
|------------------|-----------|
| Health checks | Unlimited |
| Webhook delivery | Not counted |
| Batch uploads | 10/hour (file size limits apply) |
| Report generation | 100/day |

---

## Upgrading Limits

If you consistently hit rate limits:

1. **Optimize First:** Implement caching and batching
2. **Review Usage:** Analyze request patterns
3. **Upgrade Plan:** Contact sales for higher limits
4. **Request Increase:** Submit support ticket with justification

Contact: sales@greenlang.io

---

## Example: Complete Rate-Limit-Aware Client

```python
import time
import requests
from typing import Optional

class RateLimitedClient:
    def __init__(self, api_key: str, requests_per_second: int = 10):
        self.api_key = api_key
        self.base_url = "https://api.vcci.greenlang.io/v2"
        self.requests_per_second = requests_per_second
        self.last_request_time = 0
        self.rate_limit_tracker = RateLimitTracker()

    def _throttle(self):
        """Throttle requests to stay within rate limits"""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.requests_per_second

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self.last_request_time = time.time()

    def _make_request(self, method: str, endpoint: str, **kwargs):
        """Make rate-limited request with automatic retry"""
        url = f"{self.base_url}{endpoint}"
        headers = kwargs.pop('headers', {})
        headers['X-API-Key'] = self.api_key

        max_retries = 5
        for attempt in range(max_retries):
            # Throttle before request
            self._throttle()

            try:
                response = requests.request(method, url, headers=headers, **kwargs)

                # Update rate limit tracking
                self.rate_limit_tracker.update(response.headers)

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.json() if response.content else {}

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"Request failed. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise

        raise Exception("Max retries exceeded")

    def get(self, endpoint: str, **kwargs):
        return self._make_request('GET', endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs):
        return self._make_request('POST', endpoint, **kwargs)

# Usage
client = RateLimitedClient(api_key="YOUR_API_KEY", requests_per_second=10)

# Safe, rate-limited requests
suppliers = client.get("/suppliers", params={'limit': 100})
```

---

**Last Updated:** November 6, 2025
**Version:** 2.0.0
