# GreenLang Factors — Platform / OEM One-Pager

> **Audience:** ESG SaaS platforms, carbon accounting software vendors,
> enterprise consulting platforms embedding climate data.
> **Pricing proposal v1 — subject to CTO / Commercial approval.**

## The climate-data layer for your SaaS

You've built the product. Your customers need the emission factors.
Don't become a data company. Use ours.

GreenLang Factors Platform is the **OEM-grade climate reference layer**
for SaaS vendors. White-label, redistribute, sub-tenant, and embed.

## What Platform tier unlocks

### OEM white-label
- **Redistribute factors** to your customers under your brand
- **Signed responses** carry both GreenLang and your OEM signature
- **Custom domain** (`factors.yourplatform.com`)
- **Logo + color theming** on the developer docs, hosted portal, and
  audit bundle PDFs
- **White-label SDK** — `pip install yourplatform-factors`

### Sub-tenants
- **100 sub-tenants included** — one per end-customer
- **Isolated overrides** — each sub-tenant has its own overlay
- **Quota per sub-tenant** — meter end-customers within your quota
- **Sub-tenant provisioning API** — programmatic onboarding

### Redistribution controls
- **Per-tenant license enforcement** — `customer_private`, `restricted`,
  `connector_only` license classes respected automatically
- **Audit trail** — every redistribution is logged + signed
- **Customer-attributable usage** — per-sub-tenant Stripe metering so
  you can pass through costs cleanly

### Signed responses
- Every `/resolve`, `/search`, `/match`, `/export` response carries an
  Ed25519 signature
- Verifiable by your customers without needing to trust us
- Supports rotation, revocation, and multi-signer flows

## Platform tier — $4,999 / mo ($50,000 / yr)

| Feature | Included |
|---|---|
| API calls | 5,000,000 / month |
| Overage rate | $0.0005 per call |
| Sub-tenants | 100 (add more for $35 / mo each) |
| OEM sites | 3 (add more for $500 / mo each) |
| Premium packs | 3 bundled + all available as add-on |
| SLA | 99.9% uptime |
| Support | Slack-Connect + named Technical Account Manager |

## Who's using it?

We're onboarding our first 3 Platform partners through Q2 2026. Ask
for references during your qualification call.

## Technical integration

- **Self-serve OEM portal** — provision sub-tenants, issue API keys, see
  usage per end-customer
- **Webhook-driven billing** — Stripe Connect supported for pass-through
  revenue sharing
- **API Gateway** — your rate limits, your brand, our data
- **Bring-your-own-credentials** — for ecoinvent / IEA / GLEC premium
  sources, your customers use their own licenses (we're transport-only)

## Get started

Email [platform@greenlang.io](mailto:platform@greenlang.io) with:
1. Your platform's name + category
2. Estimated end-customer count in year one
3. Target go-live date

Typical Platform close: 4-6 weeks through a structured technical +
commercial diligence. First $50k-$100k ACV typical for a design partner.
