# GreenLang Factors — Developer One-Pager

> **Audience:** Product-led growth. Individual developers, small engineering
> teams, startups shipping carbon features.
> **Pricing proposal v1 — subject to CTO / Commercial approval.**

## The API for climate-reference data

Shipping a carbon feature without a climate data layer is like shipping
a payments product without an ACH provider. GreenLang Factors is the
climate-data equivalent of Stripe: one API, signed receipts, full
provenance, regulated coverage.

## 10-minute quickstart

```bash
pip install greenlang-factors
export GL_FACTORS_API_KEY=sk_test_...
python -c "from greenlang.factors import resolve; print(resolve('electricity_us_grid_ca', kwh=1000))"
```

That's it. You have a signed emission-factor response with full
provenance, a method-pack reference, and a reproducible explain block.

## What ships with every plan

- **Certified factors** — regulator-grade, SLA-backed, reproducible
- **Signed receipts** — Ed25519 signatures on every response
- **Method packs** — Corporate GHG, Electricity (location + market), Freight ISO 14083, Land / Removals
- **Python + TypeScript SDKs** — idiomatic, typed, batteries-included
- **CLI** (`gl factors ...`) — for notebooks, CI, audit workflows
- **Batch API** — `/export` for tens of thousands of line items in one call
- **Explain blocks** — every response includes reproducible calculation trace

## Developer Pro tier — $299 / mo ($2,990 / yr, ~17% off)

| Feature | Included |
|---|---|
| API calls | 50,000 / month |
| Overage rate | $0.01 per call |
| Batch export | 10,000 rows / day |
| Method packs | Corporate, Electricity, Freight, Land |
| SLA | 99.5% uptime |
| Support | Email, 1 business day |
| Private overrides | 50 entries / project |

## What's NOT in Developer Pro

- Premium data packs (CBAM, Product Carbon / LCI, PCAF, EPD, Agrifood)
  — available as add-ons starting at $399 / mo each
- Multi-tenant / OEM redistribution — that's the Platform tier
- SSO / SCIM / VPC — that's the Enterprise tier

## Why not just price per factor?

We explicitly don't. Every tier gets the same catalog. You pay for the
usage and assurance guarantees your product needs — not for features
you never touch.

## Get started

```bash
curl -X POST https://api.greenlang.io/v1/billing/checkout/developer_pro \
  -H "Content-Type: application/json" \
  -d '{"success_url":"https://yourapp.com/success","cancel_url":"https://yourapp.com/cancel"}'
```

Or visit [https://greenlang.io/pricing](https://greenlang.io/pricing).

**Questions?** [developer-relations@greenlang.io](mailto:developer-relations@greenlang.io)
