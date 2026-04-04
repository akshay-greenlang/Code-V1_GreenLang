# GreenLang API Rate Limiting

## Overview

GreenLang enforces rate limiting at multiple layers to ensure fair usage,
protect downstream services, and maintain system stability.  The platform
provides both **in-memory** and **Redis-backed distributed** rate limiters
with four algorithm choices and configurable scoping.

---

## Rate Limit Headers

Every API response includes standard rate limit headers:

| Header | Description | Example |
|--------|-------------|---------|
| `X-RateLimit-Limit` | Maximum requests allowed in the current window | `100` |
| `X-RateLimit-Remaining` | Remaining requests in the current window | `87` |
| `X-RateLimit-Reset` | Unix timestamp when the window resets | `1743782400` |
| `Retry-After` | Seconds until the client should retry (only present when rate limited) | `42` |

### Example Response Headers

Normal response:

```
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1743782400
Content-Type: application/json
```

Rate-limited response:

```
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1743782400
Retry-After: 42
Content-Type: application/json

{"error": "Rate limit exceeded"}
```

---

## Rate Limit Tiers

### Per Authentication Method

| Authentication | Default Limit | Window | Scope |
|----------------|---------------|--------|-------|
| JWT Bearer token | 100 requests | 1 minute | Per user |
| API Key | 1000 requests | 1 hour | Per key |
| Unauthenticated | 10 requests | 1 minute | Per IP |

### Per Endpoint Category

| Endpoint Pattern | Limit | Period | Notes |
|------------------|-------|--------|-------|
| `POST /api/v1/apps/*/run` | 10 | minute | Pipeline execution (expensive compute) |
| `GET /api/v1/runs` | 100 | minute | List runs (read-only) |
| `GET /api/v1/runs/*/bundle` | 30 | minute | Download bundles (I/O intensive) |
| `POST /api/v1/auth/token` | 5 | minute | Login attempts (brute-force protection) |
| `POST /api/v1/auth/api-keys` | 10 | hour | Key creation |
| `GET /api/v1/agents` | 100 | minute | Registry reads |
| `POST /api/v1/agents` | 20 | minute | Agent creation |
| `GET /health` | Unlimited | -- | Health checks excluded from rate limiting |
| `GET /metrics` | Unlimited | -- | Metrics endpoint excluded from rate limiting |

---

## Rate Limiting Algorithms

GreenLang supports four rate limiting strategies, selectable via configuration.

### 1. Token Bucket (Default)

The token bucket algorithm provides burst-friendly rate limiting.  Tokens are
added to a bucket at a constant rate; each request consumes one token.  When
the bucket is empty, requests are rejected.

**Advantages:** Allows short bursts of traffic while maintaining a long-term average rate.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tokens_per_second` | 10.0 | Rate at which tokens are replenished |
| `bucket_size` | 100 | Maximum burst capacity |

### 2. Sliding Window

The sliding window algorithm counts requests within a moving time window.
It is more accurate at window boundaries than the fixed window approach.

**Advantages:** No boundary burst problem; accurate request counting.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size_seconds` | 60 | Length of the sliding window |
| `max_requests` | 100 | Maximum requests per window |

### 3. Fixed Window

The fixed window algorithm aligns time windows to fixed boundaries (e.g.,
every minute on the minute) and counts requests within each window.

**Advantages:** Memory efficient (single counter per window), predictable reset times.

**Disadvantage:** Clients can briefly send up to 2x the limit at window boundaries.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size_seconds` | 60 | Window duration |
| `max_requests` | 100 | Maximum requests per window |

### 4. Leaky Bucket

The leaky bucket algorithm enforces a smooth, constant output rate.  Requests
fill a bucket that "leaks" at a fixed rate.  When the bucket is full, new
requests are rejected.

**Advantages:** Constant output rate, good for protecting downstream systems with fixed throughput.

**Disadvantage:** Adds latency during sustained traffic, no burst allowance.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `leak_rate` (tokens_per_second) | 10.0 | Rate at which requests are processed |
| `bucket_size` | 100 | Maximum queue depth |

---

## Rate Limit Scoping

Rate limits can be scoped to different granularity levels:

| Scope | Key Format | Description |
|-------|------------|-------------|
| `global` | `global` | Single limit shared across all clients |
| `per_client` | `client:<id>` | Independent limit per authenticated client |
| `per_route` | `route:<path>` | Independent limit per API endpoint |
| `per_client_route` | `client:<id>:route:<path>` | Independent limit per client per endpoint |

The default scope is `per_client`.

### Client Identification

The client is identified in this order:

1. `X-API-Key` header (if present).
2. Authenticated user ID from JWT (via `request.state.user_id`).
3. Client IP address (fallback).

---

## Distributed Rate Limiting (Redis)

For multi-instance deployments, GreenLang uses **Redis** as the coordination
backend.  Atomic Lua scripts ensure consistency across application instances.

### Architecture

```
Client --> Kong API Gateway --> App Instance 1 --+
                            --> App Instance 2 --+--> Redis (rate limit state)
                            --> App Instance 3 --+
```

### Redis Key Structure

| Strategy | Redis Keys | TTL |
|----------|-----------|-----|
| Token Bucket | `ratelimit:client:<id>:tokens`, `ratelimit:client:<id>:ts` | 3600s |
| Sliding Window | `ratelimit:client:<id>:window` (sorted set) | 3600s |
| Fixed Window | `ratelimit:client:<id>:<window_start>` | window_size + 1s |
| Leaky Bucket | `ratelimit:client:<id>:level`, `ratelimit:client:<id>:ts` | 3600s |

### Fallback Behavior

If Redis is unavailable, the rate limiter automatically falls back to
in-memory limiting per application instance.  This is controlled by the
`fallback_to_memory` configuration option (default: `true`).

### Redis Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `redis_url` | `redis://localhost:6379` | Redis connection URL |
| `redis_prefix` | `ratelimit:` | Key prefix |
| `redis_db` | 0 | Redis database number |
| `pool_max_connections` | 10 | Connection pool size |
| `connection_timeout` | 5.0s | Connection timeout |
| `socket_timeout` | 5.0s | Socket timeout |
| `key_ttl_seconds` | 3600 | TTL for all rate limit keys |

---

## API Key Rate Limiting

Each API key has its own rate limit independent of the global limits:

| Setting | Default | Description |
|---------|---------|-------------|
| Per-key rate limit | 1000 requests/hour | Configurable at key creation |
| Rate limit window | 3600 seconds | Rolling window |

When a key exceeds its individual rate limit, the request is rejected with
a `RateLimitExceededError` (HTTP 429) before the global rate limiter is
consulted.

---

## Handling Rate Limits

### Best Practices

1. **Respect `Retry-After`** -- Always wait the number of seconds indicated
   before retrying.
2. **Implement exponential backoff** -- If no `Retry-After` header is present,
   use exponential backoff starting at 1 second.
3. **Monitor `X-RateLimit-Remaining`** -- Proactively slow down requests when
   remaining quota is low.
4. **Batch requests** -- Combine multiple operations into a single request
   when the API supports it.
5. **Cache responses** -- Avoid unnecessary duplicate requests.

### Python Example

```python
import time
import requests

def call_with_retry(url, headers, max_retries=3):
    """Make an API call with rate limit retry handling."""
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            remaining = response.headers.get("X-RateLimit-Remaining", "0")
            print(f"Rate limited. Remaining: {remaining}. Retrying in {retry_after}s...")
            time.sleep(retry_after)
            continue

        response.raise_for_status()
        return response.json()

    raise Exception("Max retries exceeded")


# Usage
headers = {"Authorization": f"Bearer {token}"}
data = call_with_retry(
    "https://api.greenlang.io/api/v1/runs",
    headers
)
```

### cURL Example

```bash
# Check remaining quota from response headers
curl -s -D - "https://api.greenlang.io/api/v1/runs" \
  -H "Authorization: Bearer $TOKEN" \
  -o /dev/null 2>&1 | grep -i "x-ratelimit"

# Output:
# X-RateLimit-Limit: 100
# X-RateLimit-Remaining: 87
# X-RateLimit-Reset: 1743782400
```

---

## Middleware Integration

### FastAPI Middleware

Rate limiting is applied as FastAPI middleware, automatically covering all
endpoints:

```python
from greenlang.execution.infrastructure.api.rate_limiter import (
    RateLimiter,
    RateLimiterConfig,
    RateLimitMiddleware,
    RateLimitStrategy,
    RateLimitScope,
)

# Configure
config = RateLimiterConfig(
    strategy=RateLimitStrategy.TOKEN_BUCKET,
    scope=RateLimitScope.PER_CLIENT,
    tokens_per_second=10.0,
    bucket_size=100,
    enable_headers=True,
)

limiter = RateLimiter(config)

# Apply as middleware
app.add_middleware(
    RateLimitMiddleware,
    limiter=limiter,
    exclude_paths=["/health", "/metrics"]
)
```

### Per-Endpoint Decorator

For fine-grained control, individual endpoints can specify their own limits:

```python
from greenlang.integration.api.security.rate_limiting import RateLimiter, RateLimitConfig

limiter = RateLimiter(RateLimitConfig(default_limit=100))

@app.post("/api/v1/apps/cbam/run")
@limiter.limit("10/minute")
async def run_cbam(request: Request):
    ...

@app.get("/api/v1/runs")
@limiter.limit("100/minute")
async def list_runs(request: Request):
    ...
```

The decorator format is `"<count>/<period>"` where period is `second`,
`minute`, `hour`, or `day`.

---

## Excluded Paths

The following paths are excluded from rate limiting by default:

| Path | Reason |
|------|--------|
| `/health` | Kubernetes liveness and readiness probes |
| `/metrics` | Prometheus metrics scraping |
| `/docs` | OpenAPI documentation UI |
| `/openapi.json` | OpenAPI schema |

---

## Source Files

| File | Purpose |
|------|---------|
| `greenlang/execution/infrastructure/api/rate_limiter.py` | Core rate limiter: TokenBucket, SlidingWindow, FixedWindow, LeakyBucket, RateLimiter, RateLimitMiddleware |
| `greenlang/execution/infrastructure/api/rate_limiter_redis.py` | Redis-backed distributed rate limiter with Lua scripts, factory function, async context manager |
| `greenlang/integration/api/security/rate_limiting.py` | Integration API rate limiter with per-endpoint decorator, Redis pipeline support |
| `greenlang/auth/api_key_manager.py` | Per-API-key rate limiting (1000 req/hour default) |
| `greenlang/agents/tools/rate_limiting.py` | Agent tool rate limiting |
| `greenlang/execution/resilience/rate_limit_handler.py` | Resilience pattern rate limit handling |
