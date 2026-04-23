---
title: "Gold set"
description: Public regression vectors for integrators to validate their Factors integration.
---

# Gold set

The **gold set** is a public collection of validated request / response vectors. Use it to:

* Smoke-test your integration against a known-good baseline.
* Catch regressions when you upgrade SDK or pin to a new edition.
* Compare numeric outputs across method profiles or jurisdictions.

## Where to get it

```sh
# Latest gold set (always tracks the current Certified edition)
curl -O https://api.greenlang.io/.well-known/gold-set/latest.tar.gz

# A specific edition
curl -O https://api.greenlang.io/.well-known/gold-set/2027.Q1.tar.gz
```

The bundle is signed (`.tar.gz.sig`) so you can verify provenance:

```sh
gpg --verify latest.tar.gz.sig latest.tar.gz
```

## Bundle layout

```
gold-set/
  manifest.json                 # bundle metadata (edition, generated_at, sha256)
  vectors/
    scope1/                     # method_profile == corporate_scope1
      001-natgas-us.json        # request + expected response
      002-diesel-eu.json
      ...
    scope2-lb/
      ...
    freight-iso14083/
      ...
  README.md
```

Each `*.json` file is a complete fixture:

```json
{
  "id": "001-natgas-us",
  "request": { /* exact body to POST to /resolve-explain */ },
  "expected": {
    "chosen_factor_id": "ef:co2:natgas:us:2026",
    "computed_total": { "value": 5302.0, "unit": "kg CO2e" },
    "edition_id": "2027.Q1"
  }
}
```

## Running the gold set against your integration

The Python SDK ships a runner:

```sh
gl-factors gold-set run \
  --bundle latest.tar.gz \
  --base-url https://api.greenlang.io \
  --tolerance 0.01
```

Exits non-zero on any divergence, with a unified-diff report.

You can also wire it into your CI:

```yaml
- name: Validate against gold set
  run: |
    pip install greenlang-factors
    gl-factors gold-set run --bundle gold-set/latest.tar.gz --tolerance 0.01
```

## How often it changes

* The Certified gold set is regenerated on every Certified edition cut (typically quarterly).
* The Preview gold set tracks the in-flight edition and may change daily.
* Both are immutable once published -- a gold-set release is never mutated, only superseded.
