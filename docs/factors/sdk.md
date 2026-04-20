# GreenLang Factors SDK

> **What.** The official Python + TypeScript client libraries for the GreenLang Factors API.
> **Why.** Zero-dependency HTTP clients with type hints, retries, and pagination. Saves you 200 lines per project.

---

## 1. Python SDK

### 1.1 Install

```bash
pip install greenlang-factors-sdk
```

Zero runtime dependencies (stdlib only). Python 3.9+.

### 1.2 Quick start

```python
from greenlang_factors_sdk import FactorsClient

client = FactorsClient(
    base_url="https://factors.greenlang.io",
    api_key="gl_live_sk_...",     # or token="<jwt>" for JWT auth
)

# Search
results = client.search("diesel combustion", geography="US")
for factor in results.factors:
    print(factor.factor_id, factor.co2e_per_unit, factor.source)

# Match (Pro tier+)
match = client.match(
    activity="stationary combustion of natural gas, boiler",
    unit="kWh",
    geography="GB",
)
print(match.factor_id, match.score, match.explanation)

# Single factor
factor = client.get_factor("ef.defra.natgas.2024.stat-comb")
print(factor.co2_per_unit, factor.ch4_per_unit, factor.dqs)

# Edition pinning (reproducibility)
client.edition = "defra-2024-q1"
```

### 1.3 Examples

| Use-case | Module |
|---|---|
| Basic search + iterate | `examples/quickstart.py` |
| Edition pinning for reproducible runs | `examples/edition_pinning.py` |
| Tenant factor overlay | `examples/tenant_overlay.py` |
| Export to JSON / Parquet | `examples/bulk_export.py` |
| Audit bundle retrieval | `examples/audit_bundle.py` |

(See `greenlang/factors/sdk/README.md` for the full list of 20+ examples.)

### 1.4 Config

| Env var | Purpose | Default |
|---|---|---|
| `GL_FACTORS_URL` | Base API URL | `https://factors.greenlang.io` |
| `GL_FACTORS_API_KEY` | API key (overrides kwarg) | — |
| `GL_FACTORS_JWT` | JWT token (overrides kwarg) | — |
| `GL_FACTORS_EDITION` | Default edition | server default |
| `GL_FACTORS_TIMEOUT` | HTTP timeout seconds | 30 |
| `GL_FACTORS_RETRIES` | Retry attempts on 429/5xx | 3 |

## 2. TypeScript SDK

### 2.1 Install

```bash
npm install @greenlang/factors-sdk
# or
pnpm add @greenlang/factors-sdk
# or
yarn add @greenlang/factors-sdk
```

### 2.2 Quick start

```ts
import { FactorsClient } from "@greenlang/factors-sdk";

const client = new FactorsClient({
  baseUrl: "https://factors.greenlang.io",
  apiKey: process.env.GL_FACTORS_API_KEY!,
});

const results = await client.search({ q: "diesel combustion", geography: "US" });
for (const factor of results.factors) {
  console.log(factor.factor_id, factor.co2e_per_unit, factor.source);
}
```

Package ships as ESM + CJS dual build; types included.

## 3. Publishing

Both SDKs publish through `.github/workflows/factors-sdk-publish.yml`, triggered on tag `factors-sdk-v*`:

```bash
# Python
cd greenlang/factors/sdk
# version is read from pyproject.toml (currently 1.0.0)
# bump + tag: factors-sdk-v1.0.1

# TypeScript
cd greenlang/factors/sdk/ts
# version is read from package.json
```

The workflow:

1. Validates version from tag matches the package manifest.
2. Runs `pytest` for Python SDK + `vitest` for TypeScript SDK.
3. Builds sdist + wheel (Python) and ESM + CJS (TypeScript).
4. Publishes to PyPI using `PYPI_API_TOKEN`.
5. Publishes to npm using `NPM_TOKEN`.
6. Creates a GitHub Release with changelog notes.

## 4. Versioning

- **Python SDK:** semver starting at 1.0.0.
- **TypeScript SDK:** semver starting at 1.0.0.
- SDKs are deliberately de-coupled from the API server version (the API is backwards-compatible within v1).

## 5. Support

- GitHub issues: https://github.com/greenlang/greenlang/issues (label `sdk`)
- Docs: https://docs.greenlang.io/factors-sdk
- Email: support@greenlang.io

---

*Last updated: 2026-04-20. Source: `greenlang/factors/sdk/` + `.github/workflows/factors-sdk-publish.yml`.*
