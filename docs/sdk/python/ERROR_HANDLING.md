# Error Handling

All SDK errors inherit from `FactorsAPIError`, which in turn extends `greenlang.utilities.exceptions.base.GreenLangException`. That means you can catch SDK errors in your existing GreenLang exception middleware without special-casing the SDK.

## Exception hierarchy

```
GreenLangException                  (greenlang.utilities.exceptions.base)
 └── FactorsAPIError                (sdk.python.errors)
      ├── AuthError                 (401)
      ├── TierError                 (403, generic)
      ├── LicenseError              (403, connector_only / redistribution)
      ├── FactorNotFoundError       (404 on /factors/{id})
      ├── ValidationError           (400 / 422)
      └── RateLimitError            (429, carries retry_after)
```

## Common shape

Every SDK exception exposes:

```python
try:
    client.get_factor("bad_id")
except FactorsAPIError as exc:
    print(exc.status_code)     # 404
    print(exc.response_body)   # {"detail": "Factor 'bad_id' not found ..."}
    print(exc.request_id)      # X-Request-ID, if server returned one
    print(exc.error_code)      # Auto-generated GreenLang error code
    print(exc.context)         # dict with status_code/response_body/request_id
```

## Mapping HTTP status to SDK exception

| Status | Exception class | Typical cause |
|:------:|-----------------|---------------|
| 400, 422 | `ValidationError` | Bad input (missing query, typo in filter) |
| 401 | `AuthError` | Missing/invalid JWT or API key |
| 403 (tier) | `TierError` | Endpoint requires a higher tier |
| 403 (license) | `LicenseError` | Factor `connector_only` + redistribution blocked |
| 404 (`/factors/{id}`) | `FactorNotFoundError` | factor_id not in edition |
| 404 (other) | `FactorsAPIError` | Unknown endpoint |
| 429 | `RateLimitError(retry_after=...)` | Tier quota exceeded |
| 5xx | `FactorsAPIError` | Server error (retried automatically) |

## Retry strategy

The transport retries automatically on `429` and `5xx` up to `max_retries` times (default 3) with exponential backoff (capped at 30 s).  Specifically:

- **429**: If `Retry-After` is present, honors it exactly (clamped to ≤60 s); otherwise falls back to exponential backoff.
- **5xx**: Pure exponential backoff: `2^(attempt-1)` seconds.
- **Network errors** (`httpx.RequestError`): Retried with exponential backoff.
- **4xx (other)**: Not retried — raised immediately.

After exhausting the retry budget, the appropriate typed exception is raised. User code should catch these (not HTTP status codes) for compatibility with future transport changes:

```python
import time

from greenlang.factors.sdk.python import FactorsClient, RateLimitError, TierError

with FactorsClient(base_url="...", api_key="...") as c:
    try:
        hits = c.search("diesel")
    except RateLimitError as exc:
        # Transport already retried; we're out of budget.  Slow down.
        time.sleep(exc.retry_after or 60)
    except TierError:
        # Upgrade tier or drop the advanced filters.
        hits = c.search("diesel", include_preview=False)
```

## Raw HTTP error access

If you need the raw response body or status, it's all on the exception object:

```python
try:
    c.search_v2("x", sort_by="bogus_field")
except ValidationError as exc:
    raw = exc.response_body
    assert exc.status_code in (400, 422)
    print("server said:", raw)
```

## Catching everything

For defensive wrappers, catch the base class so you also cover new subclasses added in future SDK versions:

```python
from greenlang.factors.sdk.python import FactorsAPIError

try:
    ...
except FactorsAPIError as exc:
    logger.error("factors call failed: status=%s err=%s", exc.status_code, exc)
```
