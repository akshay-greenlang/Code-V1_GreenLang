# Rate Limits

GreenLang Factors uses a **sliding-window** rate limiter keyed on the caller's `user_id` (or API-key fingerprint for machine credentials). Each tier has a per-minute cap, a short-term burst allowance, and a separate per-15-minute export budget.

**Implementation:** `greenlang/factors/middleware/rate_limiter.py`

---

## Tier table (requests per minute)

| Tier | RPM | Burst | Exports / 15 min |
|---|---|---|---|
| `community` | 60 | 10 | 1 |
| `pro` | 600 | 50 | 5 |
| `enterprise` | 6000 | 200 | 20 |
| `internal` | 60000 | 1000 | 200 |

`consulting` shares the `pro` quota by default; contact sales for custom ceilings. Full specs are hard-coded in `_TIER_SPECS` (`rate_limiter.py`, line 72).

### Windows

- **General window:** 60 seconds (`_GENERAL_WINDOW_SECONDS = 60`). Every request counts toward the per-minute cap.
- **Export window:** 15 minutes (`_EXPORT_WINDOW_SECONDS = 900`). Only `/factors/export` and `/factors/{id}/audit-bundle` count.

Both windows are implemented as sliding (sorted-set style). Bursts are absorbed by the `burst` allowance; sustained traffic above the RPM returns 429.

---

## Headers on every response

```
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 587
X-RateLimit-Reset: 1745328000
```

| Header | Meaning |
|---|---|
| `X-RateLimit-Limit` | Your tier's RPM cap. |
| `X-RateLimit-Remaining` | Requests left in the current sliding window. |
| `X-RateLimit-Reset` | UTC epoch seconds when the oldest tracked request expires from the window. |

Export calls also carry `X-RateLimit-Export-Limit` / `Remaining` / `Reset`.

---

## 429 Too Many Requests

Response:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 42
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1745328042
Content-Type: application/json

{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Try again in 42 seconds.",
  "details": {
    "tier": "pro",
    "limit_type": "rpm",
    "limit": 600,
    "retry_after_seconds": 42
  }
}
```

`Retry-After` is the recommended wait before the next request. Honor it — the server will continue 429'ing until the oldest tracked call exits the window.

---

## Best practices

### Respect `Retry-After`

Simple exponential backoff on other 5xx responses, but on 429 use the server-provided `Retry-After` value. The server already knows when the window opens up.

```python
import time, httpx

def call_with_retry(req_fn, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return req_fn()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = int(e.response.headers.get("Retry-After", "5"))
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("exhausted retries")
```

Both SDKs implement this automatically up to `max_retries` (default 3). See `greenlang/factors/sdk/python/transport.py` and `sdk/ts/src/transport.ts`.

### Prefer batch for bulk

A 10,000-row inventory should not make 10,000 `/resolve-explain` calls — use `/factors/batch/submit`. Batch jobs consume **one** RPM slot each, regardless of payload size, and process asynchronously.

### Pin `default_edition`

Pinning does not itself affect rate limits, but it makes your calls idempotent, which means you can use ETag caching aggressively:

```python
client = FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_pk_...",
    default_edition="2027.Q1-electricity",
    # ETagCache is on by default
)
```

Cache hits return 304 and do not consume your RPM budget.

### Cache at your edge

For read-heavy workloads, put a cache in front of the SDK. Factor data is idempotent per-edition, so a caching layer keyed on `(edition_id, factor_id)` or `(edition_id, resolution_request_hash)` is safe.

### Avoid clock-synchronized retry storms

When many workers retry at exactly `Retry-After: N`, they synchronise and hammer the server as a wave. Add jitter:

```python
import random
time.sleep(retry_after + random.uniform(0, retry_after * 0.25))
```

---

## Bursts

The `burst` column in the tier table is a short-term budget above the RPM cap that resets opportunistically. It exists to absorb page-load spikes (e.g. 50 concurrent calls when a dashboard opens). You cannot rely on it for sustained traffic; sustained traffic is governed by RPM.

Concretely: Pro tier lets you do 50 requests in 1 second, then throttles the next 49 seconds down to ~10 req/sec until the window balances out. The middleware enforces this via the sliding-window algorithm (not a token bucket), so bursts are naturally smoothed.

---

## Export budget

The export endpoint (`/factors/export`) and audit-bundle endpoint (`/factors/{id}/audit-bundle`) consume a **separate** 15-minute budget. This protects upstream source licensing (many licensed sources cap daily pulls).

| Tier | Exports / 15 min |
|---|---|
| community | 1 |
| pro | 5 |
| enterprise | 20 |
| internal | 200 |

When exceeded, the server returns 429 with `details.limit_type = "exports"`. Rising from `pro` to `enterprise` is the usual remedy — sales can also provision temporary bursts.

---

## Storage backends

### In-memory (default)

Suitable for single-instance deployments. State is held in a `deque` per-user, protected by a `threading.Lock`. Restarts reset the counters.

### Redis

For multi-instance deployments, pass a `redis.Redis` client via `RateLimitConfig.redis_client`. The limiter uses a sorted-set sliding window (`ZADD ... NX`, `ZCOUNT`, `ZREMRANGEBYSCORE`). All instances share state.

```python
import redis
from greenlang.factors.middleware.rate_limiter import RateLimitConfig, configure

configure(RateLimitConfig(redis_client=redis.Redis(host="redis-prod", port=6379)))
```

---

## Kill switch

`RateLimitConfig(enabled=False)` disables all limiting. Use for load-test environments only.

```python
configure(RateLimitConfig(enabled=False))
```

Also available via env: `GL_FACTORS_RATE_LIMIT_ENABLED=false`.

---

## See also

- [Authentication](./authentication.md) — how tier is derived from the credential.
- [Errors](./errors.md) — full 429 semantics and sibling error codes.
- [cURL recipes — recipe 9](../quickstart/curl-recipes.md) — shell backoff template.

---

## File citations

| Piece | File |
|---|---|
| Sliding-window limiter | `greenlang/factors/middleware/rate_limiter.py` |
| Tier specs (`_TIER_SPECS`) | `greenlang/factors/middleware/rate_limiter.py` (line 72) |
| Window constants | `greenlang/factors/middleware/rate_limiter.py` (lines 80-81) |
| `apply_rate_limit` / `apply_export_rate_limit` | `greenlang/factors/middleware/rate_limiter.py` |
| SDK retry-on-429 | `greenlang/factors/sdk/python/transport.py`, `greenlang/factors/sdk/ts/src/transport.ts` |
