# GreenLang Factors — Open-Core Boundary

Non-negotiable #7 from the CTO spec: *"open-core must not blur into
enterprise entitlements."* This document is the canonical contract for
what ships in the open-source distribution, what requires a commercial
subscription, and what is always licensed (pass-through cost) regardless
of plan.

## Three lanes

| Lane | What it means | License |
| --- | --- | --- |
| **Open core (OSS)** | Shipped in the public repository under the Apache 2.0 License. No entitlement check. Runs on any laptop with a free Community tier API key (or offline with bundled open editions). | Apache-2.0 |
| **Commercial add-ons** | Shipped in the same repository but gated by runtime entitlement checks. Source is readable; execution requires a Pro / Consulting / Enterprise subscription. | GreenLang Commercial Terms |
| **Licensed packs** | Upstream data redistributed under a third-party license. Never included in bulk public exports; enabled per tenant via a pass-through licensing fee. | Data-provider license (Green-e, Ecoinvent, IEA, AIB, …) |

The runtime enforces the boundary in three places:

- `greenlang/factors/tier_enforcement.py` — visibility + export row caps.
- `greenlang/factors/entitlements.py` — premium pack SKUs, OEM rights.
- `greenlang/factors/approval_gate.py` — registry-level license blocks.

If a feature is not in the tables below, it is **open core**.

## Feature matrix

### Data model + registry

| Feature | OSS | Commercial | Licensed |
| --- | --- | --- | --- |
| Gas-component storage (CO2/CH4/N2O/HFCs/PFCs/SF6/NF3/biogenic_CO2) | ✅ | | |
| AR4/AR5/AR6 GWP sets (100-yr and 20-yr) | ✅ | | |
| Validity-date model (`valid_from` / `valid_to`) | ✅ | | |
| SQLite catalog repository | ✅ | | |
| PostgreSQL + pgvector repository | | ✅ | |
| Edition manifest + cryptographic signing | ✅ | | |
| Tenant overlay / private registry | | ✅ (Consulting+) | |
| Unit ontology (unit graph, conversions) | ✅ | | |
| NAICS / ISIC / NACE / HS-CN / GICS classifications | ✅ | | |
| Industry crosswalk YAML taxonomies | ✅ | | |
| Regulatory-framework tagger (GHG Protocol, TCFD, CSRD, ESRS E1, SBTi, CDP, IFRS S2) | ✅ | | |
| CBAM selector + EU Taxonomy tagging | | ✅ (Pro+) | |
| CA SB 253 / SB 261 applicability | | ✅ (Pro+) | |

### Ingestion + sources

| Feature | OSS | Commercial | Licensed |
| --- | --- | --- | --- |
| DESNZ / UK GHG conversion factors | ✅ | | |
| EPA GHG Emission Factors Hub | ✅ | | |
| eGRID (US) | ✅ | | |
| IPCC 2006 default factors | ✅ | | |
| Australian NGA factors | ✅ | | |
| TCR / GRP defaults | ✅ | | |
| IEA / Ecoinvent / Green-e | | | 🔒 |
| Residual-mix factors (EU AIB, US Green-e) | | | 🔒 |
| ERP/procurement connectors | | ✅ (Consulting+) | |

### Resolution + method packs

| Feature | OSS | Commercial | Licensed |
| --- | --- | --- | --- |
| 7-step deterministic resolution cascade | ✅ | | |
| Method-profile indirection enforced (non-negotiable #6) | ✅ | | |
| Explain endpoint with `why_chosen` + alternates | ✅ | | |
| GHG Protocol Corporate S1/S2-LB/S2-MB/S3 packs | ✅ | | |
| ISO 14064-1 universal pack | ✅ | | |
| ISO 14083 freight pack | ✅ | | |
| Product carbon pack (ISO 14067 / GHG PS / PACT) | | ✅ (Pro+) | |
| PCAF financed-emissions packs (7 asset-class variants) | | ✅ (Pro+) | |
| GHG LSR land-sector packs | | ✅ (Pro+) | |
| CBAM method pack | | ✅ (Pro+) | |
| PAS 2050 / PEF / OEF packs | | ✅ (Pro+) | |

### API + developer platform

| Feature | OSS | Commercial | Licensed |
| --- | --- | --- | --- |
| REST `/factors`, `/search`, `/match`, `/coverage` | ✅ | | |
| REST `/resolve`, `/explain`, `/resolve-explain`, `/alternates` | | ✅ (Pro+) | |
| Batch resolution (queued jobs) | | ✅ (Pro+) | |
| Audit-bundle endpoint | | ✅ (Enterprise) | |
| GraphQL Factors schema (queries) | ✅ | | |
| GraphQL resolveFactor / resolveFactorExplain | | ✅ (Pro+) | |
| GraphQL setFactorOverride | | ✅ (Consulting+) | |
| Python SDK (`greenlang.factors.sdk.python`) | ✅ | | |
| TypeScript SDK (`@greenlang/factors-sdk`) | ✅ | | |
| CLI (`glfactors`) | ✅ | | |
| Webhook registry + HMAC signing + retry delivery | ✅ | | |
| Customer webhook subscriptions > 5 active | | ✅ (Pro+) | |

### Governance + operations

| Feature | OSS | Commercial | Licensed |
| --- | --- | --- | --- |
| Source registry + legal sign-off flags | ✅ | | |
| Approval gate for certified promotion | ✅ | | |
| Regulatory-watch polling + change detection | ✅ | | |
| Customer-specific policy overlays | | ✅ (Consulting+) | |
| Edition rollback tooling | ✅ | | |
| SLA timer / review workflow | | ✅ (Pro+) | |
| Prometheus metrics | ✅ | | |
| Loki log aggregation | ✅ | | |
| White-label / OEM deployment rights | | ✅ (Enterprise) | |

## Entitlement check — one source of truth

Runtime enforcement is a single decision point. Every request that could
touch a commercial or licensed surface passes through
`greenlang.factors.tier_enforcement.enforce_tier()` which reads the
tenant record, checks tier membership, and confirms per-pack SKU
entitlement:

```
request
  → tenant_resolver  (tenant_id + api_key)
  → tier_enforcement (plan in {community, pro, consulting, enterprise})
  → entitlements      (PackSKU, OEM rights, seat cap, volume cap)
  → approval_gate     (registry-level redistribution guards)
```

Any code path that bypasses this chain is a boundary violation and is
blocked in CI (`tests/factors/test_tier_enforcement.py`,
`tests/factors/test_approval_gate.py`).

## Commitments we don't break

1. **No tier flip on existing OSS code.** Once a feature ships as open
   core, it stays open core. New commercial features arrive in new
   modules or behind new flags, never by gating an existing OSS code
   path.
2. **No telemetry as a gate.** OSS users are free to run fully offline.
   Entitlement checks short-circuit to "community" when the telemetry
   endpoint is unreachable.
3. **Separation of license classes stays strict.** Licensed packs live
   behind `connector_only=true` rows that never enter public bulk
   exports and never merge with open data in a single edition.
4. **Explainability is universal.** Every factor the engine returns —
   OSS, commercial, or licensed — carries the same `why_chosen`,
   `assumptions`, and `alternates` payload. Transparency is not a
   premium feature.
5. **Documentation parity.** OSS features are documented under
   `docs/factors/`; commercial features are documented alongside the
   same surfaces with a `@premium` marker. No "please contact sales"
   black holes.

## Requesting a change

Boundary moves require CTO approval. Open a ticket in the
`greenlang-factors` repository tagged `open-core-boundary` with:

- Which feature is moving (and which lane it moves from/to).
- Which commitments above would be affected.
- A migration plan for customers whose contracts referenced the old
  lane.

The CTO approves in writing; the approval ticket is linked from this
document's next commit.
