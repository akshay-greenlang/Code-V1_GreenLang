---
title: Quickstart
description: From sign-up to your first /resolve call in five minutes.
---

# Quickstart

This guide takes you from "I have nothing" to "I called the resolve endpoint and got a fully-explained factor back" in under five minutes.

## 1. Get an API key

Visit [greenlang.ai/pricing](https://greenlang.ai/pricing) and sign up for the **Community** plan (free) or **Developer Pro**.

The success page shows your API key exactly once. Save it to your secret manager:

```sh
export GL_FACTORS_API_KEY="gl_fac_..."
```

## 2. Install an SDK (optional but recommended)

```sh
# Python
pip install greenlang-factors==1.0.0

# Node
npm install @greenlang/factors@1.0.0
```

You can also call the API directly via curl -- see step 4 below.

## 3. Make a search call

```python
from greenlang_factors import FactorsClient

with FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_fac_...",
) as client:
    hits = client.search("natural gas US Scope 1", limit=3)
    for f in hits.factors:
        print(f.factor_id, f.co2e_per_unit, f.unit)
```

```ts
import { FactorsClient } from "@greenlang/factors";

const client = new FactorsClient({
  baseUrl: "https://api.greenlang.io",
  apiKey: process.env.GL_FACTORS_API_KEY!,
});
const hits = await client.search("natural gas US Scope 1", { limit: 3 });
for (const f of hits.factors) {
  console.log(f.factor_id, f.co2e_per_unit, f.unit);
}
```

## 4. Make a resolve call (with explain)

```python
resolved = client.resolve({
    "activity": "natural gas combustion",
    "method_profile": "corporate_scope1",
    "jurisdiction": "US",
    "reporting_date": "2026-04-01",
    "quantity": 1000,
    "unit": "therm",
})
print("chosen factor:", resolved.chosen.factor_id)
print("co2e total:", resolved.computed_total)
print("alternates considered:", len(resolved.alternates))
```

Or via curl:

```sh
curl -s "https://api.greenlang.io/api/v1/factors/resolve-explain" \
  -H "X-API-Key: $GL_FACTORS_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{"activity":"natural gas combustion","jurisdiction":"US","method_profile":"corporate_scope1","quantity":1000,"unit":"therm"}'
```

## 5. Pin an edition for reproducibility

For audited or regulated reporting, **pin every request to a specific catalog edition** so the same input always produces the same output, even after the catalog updates:

```python
with client.with_edition("2027.Q1-electricity") as scoped:
    resolved = scoped.resolve({"activity": "electricity", ...})
```

If the server returns a different edition than the pin, an `EditionMismatchError` is raised -- we never silently accept drift.

See [Editions](./concepts/editions.md) for the full mental model.

## 6. Verify the signed receipt

Pro+ responses carry a signed receipt. Verify it offline so audit packages remain self-contained:

```python
summary = client.verify_receipt(response.model_dump())
print("verified by", summary["key_id"], "at", summary["signed_at"])
```

## What next?

* [Concepts -- factor record](./concepts/factor-record.md): what every field actually means.
* [API reference -- resolve](./api/resolve.md): full endpoint contract.
* [SDK reference -- Python](./sdk/python.md): every method, every option.
* [Gold set](./gold-set.md): regression-test your integration against our published vectors.
