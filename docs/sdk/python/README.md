# GreenLang Factors — Python SDK

Production-grade Python client for the GreenLang Factors REST API.

- Sync + async clients (`FactorsClient`, `AsyncFactorsClient`)
- JWT + API key auth, optional HMAC request signing (Pro+)
- Auto-retry on 429/5xx with exponential backoff
- Transparent ETag response cache
- Typed Pydantic v2 response models
- Cursor + offset pagination helpers
- HMAC-SHA256 webhook signature verifier
- `greenlang-factors` CLI

## Installation

The SDK ships inside the `greenlang` distribution. To use it standalone, make sure the required runtime dependencies are installed:

```bash
pip install "httpx>=0.27" "pydantic>=2.0" "tenacity>=8.0"
```

The SDK itself lives at `greenlang.factors.sdk.python`.

## Quickstart (sync)

```python
from greenlang.factors.sdk.python import FactorsClient

with FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_fac_...",                 # or jwt_token=...
    default_edition="ef_2026_q1",         # optional: pin every call
) as client:
    hits = client.search("diesel US Scope 1", limit=5)
    for f in hits.factors:
        print(f.factor_id, f.co2e_per_unit)
```

## Quickstart (async)

```python
import asyncio
from greenlang.factors.sdk.python import AsyncFactorsClient

async def main():
    async with AsyncFactorsClient(base_url="https://api.greenlang.io", api_key="gl_fac_...") as c:
        hits = await c.search("diesel")
        async for factor in c.paginate_search("diesel", page_size=50):
            print(factor.factor_id)

asyncio.run(main())
```

## Public surface

Importable from `greenlang.factors.sdk.python`:

| Symbol | Kind | Description |
|---|---|---|
| `FactorsClient` / `AsyncFactorsClient` | class | Primary API client |
| `APIKeyAuth`, `JWTAuth`, `HMACAuth` | class | Authentication providers |
| `Factor`, `Edition`, `Source`, `MethodPack` | model | Entity response models |
| `ResolvedFactor`, `ResolutionRequest` | model | Resolution I/O |
| `QualityScore`, `Uncertainty`, `GasBreakdown` | model | Nested details |
| `SearchResponse`, `FactorMatch`, `FactorDiff` | model | List/search/diff wrappers |
| `AuditBundle`, `Override`, `CoverageReport` | model | Enterprise + tenant features |
| `BatchJobHandle` | model | Async batch job handle |
| `FactorsAPIError`, `RateLimitError`, `TierError`, `FactorNotFoundError`, `LicenseError`, `ValidationError`, `AuthError` | exceptions | Error hierarchy |
| `verify_webhook`, `verify_webhook_bytes`, `verify_webhook_strict` | functions | Webhook HMAC verifier |

See companion docs in this directory for deep dives.

## CLI

Installed as `greenlang-factors`:

```bash
export GREENLANG_FACTORS_BASE_URL=https://api.greenlang.io
export GREENLANG_FACTORS_API_KEY=gl_fac_...

greenlang-factors search "natural gas"
greenlang-factors get-factor EF:US:diesel:2024:v1 --pretty
greenlang-factors resolve "diesel combustion" \
  --method-profile corporate_scope1 --jurisdiction US
greenlang-factors explain EF:US:diesel:2024:v1 --alternates 5
greenlang-factors list-editions --include-pending
```

Set `GREENLANG_LOG_LEVEL=INFO` for verbose client logs.

## Examples

Runnable scripts are under `examples/factors_sdk/python/`:

- `01_basic_search.py` — search + search_v2
- `02_resolve_with_explain.py` — full cascade + explain payload
- `03_batch_resolution.py` — async batch + polling
- `04_tenant_override.py` — Consulting-tier override flow
- `05_audit_export.py` — Enterprise-tier audit bundle export

## Related docs

- `AUTHENTICATION.md` — JWT, API key, HMAC signing
- `RESOLUTION.md` — 7-step resolution cascade, explain, alternates
- `ERROR_HANDLING.md` — exception hierarchy and retry strategy
- `VERSION_PINNING.md` — edition pinning for reproducibility

## Tests

```bash
python -m pytest tests/factors/sdk/test_python_sdk.py -v
```
