# GreenLang Factors — OEM Deployment Guide

This guide is for partners embedding GreenLang Factors inside their own
product (white-label, private deployment, or managed-on-behalf-of
customers). The platform supports four deployment modes; pick one and
follow the matching checklist.

## Deployment modes at a glance

| Mode | Who operates it | Tenant model | Typical use |
| --- | --- | --- | --- |
| **Managed multi-tenant** | GreenLang | Shared cluster, per-tenant overlays | Direct Pro / Consulting / Enterprise customers |
| **White-label shared** | GreenLang | Shared cluster, OEM-branded UI + API domain | SaaS integrators / ERP vendors |
| **OEM single-tenant** | GreenLang | Dedicated per-OEM cluster | Regulated industries, data-residency requirements |
| **Self-hosted** | Partner | Partner operates GreenLang binaries on partner infrastructure | Banks, defense, sovereign clouds |

All four share the same code base. What differs is tenancy scope,
operator, and which parts of the licence matrix apply.

## OEM rights — the contract surface

`greenlang/factors/entitlements.py` exposes a tri-state `OEMRights`
enum that controls what partners can do with factor data returned to
their end users:

| `OEMRights` value | What it means | Typical SKU |
| --- | --- | --- |
| `FORBIDDEN` | Partner may call the API but may not redistribute factor rows to their end users. Useful for internal analytics. | Community, Developer Pro |
| `INTERNAL_ONLY` | Partner may surface factors inside their own product UI, but factor rows may not be exported or syndicated further. | Consulting |
| `REDISTRIBUTABLE` | Partner may embed factor rows inside reports their customers receive. No downstream syndication without written addenda. | Enterprise / OEM |

The enforcement point is
`greenlang.factors.tier_enforcement.enforce_oem_rights()`. Every
export endpoint (`/factors/export`, `/factors/{id}` with
`include_values=true`, batch jobs that dump CSV/Parquet) checks the
tenant's OEMRights grant before emitting a row.

## Licensed data pass-through

Licensed packs (Green-e residual mix, AIB European residual mix,
Ecoinvent, IEA) never ship inside the OEM deployment by default. A
licensed pack is enabled per tenant, per OEM deployment, after three
things happen in order:

1. Partner signs the data-provider addendum (GreenLang facilitates; fee
   passes through with a small integration margin).
2. Ops flips `entitlements.packs[tenant_id]` to include the relevant
   `PackSKU`.
3. Runtime sees the entitlement on the next request and begins routing
   traffic to the connector-only rows.

Bulk exports of licensed data are blocked at the approval-gate layer
regardless of tier — the registry row sets `connector_only=true` and
`redistribution_allowed=false`, and `approval_gate.public_bulk_export_allowed_for_factor()`
short-circuits to "no".

## White-label topology

```
  ┌──────────────────────────────┐
  │ Partner-branded web UI       │  (factors.partner.example.com)
  └────────────┬─────────────────┘
               │ (HTTPS, partner JWT)
               ▼
  ┌──────────────────────────────┐
  │ Kong Gateway + TLS 1.3       │  — INFRA-006
  │  • HMAC auth per tenant      │
  │  • Rate limit per plan       │
  │  • Request signing           │
  └────────────┬─────────────────┘
               │
               ▼
  ┌──────────────────────────────┐
  │ greenlang-factors-api        │  (REST + GraphQL)
  │  • tenant_overlay_reader     │
  │  • tier_enforcement          │
  │  • entitlements              │
  │  • approval_gate             │
  └─────┬──────────┬─────────────┘
        │          │
        │          ▼
        │   ┌──────────────────┐
        │   │ Tenant overlays   │  (per-OEM SQLite / Postgres)
        │   └──────────────────┘
        ▼
  ┌──────────────────────────────┐
  │ PostgreSQL + pgvector        │  — INFRA-002 / INFRA-005
  │  • factor_catalog             │
  │  • factor_editions            │
  │  • factor_regulatory_events   │
  │  • factor_webhook_subscriptions
  └──────────────────────────────┘
```

Key properties:

- Every query carries `tenant_id`. Queries without one are rejected at
  the middleware layer — there is no "global" query path in OEM mode.
- Tenant overlays are strictly scoped; Postgres row-level security
  enforces isolation even from a compromised application role.
- Edition manifests include an OEM-specific signing key when
  `entitlements.packs[tenant_id].requires_private_signing = True`, so
  auditors can verify an edition was served inside a specific OEM
  deployment.

## Single-tenant / self-hosted checklist

### 1. Infrastructure baseline

| Component | Minimum | Recommended |
| --- | --- | --- |
| Compute | 4 vCPU × 2 (API + worker) | 8 vCPU × 3 + 4 vCPU worker pool |
| Memory | 16 GB | 64 GB |
| Postgres | 15.x, 100 GB SSD | 16.x, 500 GB SSD + pgvector 0.5+ |
| Redis | 7.x, 2 GB | 7.x, 8 GB for ~1M cached resolutions |
| TLS | 1.2 | 1.3 with mutual auth |
| Secrets | env vars | HashiCorp Vault (SEC-006) |

### 2. Bootstrap steps

```bash
# 1. Provision the stack
helm install gl-factors ./deploy/helm/factors \
    --set image.tag=$VERSION \
    --set tenant.mode=oem \
    --set tenant.oem_id=partner-acme \
    --set postgres.url=$DATABASE_URL \
    --set signing.secret_ref=vault:factors/partner-acme/signing-key

# 2. Seed the factor catalog
gl-factors ingest-builtin \
    --sources "epa_hub,desnz_ghg_conversion,ipcc_defaults" \
    --edition-id 2026.05.0

# 3. Promote edition to stable
gl-factors release-publish \
    --edition-id 2026.05.0 \
    --approved-by ops@partner.example

# 4. Wire the regulatory watch CronJob
kubectl apply -f deploy/k8s/cronjob-regulatory-watch.yaml
```

### 3. Tenant onboarding

```python
from greenlang.factors.entitlements import Entitlement, PackSKU, OEMRights
from greenlang.factors.entitlements_registry import EntitlementRegistry

registry = EntitlementRegistry.from_env()
registry.grant(
    tenant_id="customer-1234",
    entitlement=Entitlement(
        plan="enterprise",
        packs=[PackSKU.CBAM, PackSKU.FREIGHT, PackSKU.FINANCE_PCAF],
        oem_rights=OEMRights.INTERNAL_ONLY,
        seat_cap=200,
        volume_cap_per_month=5_000_000,
        expires_at="2027-12-31T23:59:59Z",
    ),
)
```

### 4. Regulatory-watch + webhooks

In OEM deployments each OEM runs its own watch cluster so that
regulatory-event streams are isolated per partner. The pipeline entry
point is `greenlang.factors.watch.pipeline.run_regulatory_watch_cycle`;
schedule it daily at 06:00 UTC. Events land in
`factor_regulatory_events` and fan out to any customer webhook
subscriptions owned by that OEM's tenants.

Private webhook endpoints inside partner infrastructure are required to
accept `X-GL-Signature` HMAC headers. See
`greenlang.factors.webhooks.sign_webhook_payload` for the canonical
signing scheme.

### 5. Observability

Every OEM deployment exports the same Prometheus metric names —
`factors_resolution_latency_ms`, `factors_factor_updates_total`,
`factors_webhook_delivery_failed_total` — prefixed by `oem_id` so that
central GreenLang SRE can maintain a unified Grafana (OBS-002) with
per-OEM breakdowns.

### 6. Disaster recovery

- Edition artifacts (factor dumps + manifests) are written to an S3
  bucket in the OEM tenant's region with 90-day versioning. Rollback is
  `gl-factors rollback-edition --to 2026.04.1`.
- The regulatory-event store is append-only; restoring the service
  from a Postgres backup preserves downstream replay semantics.
- Webhook deliveries retry with exponential backoff (4 attempts by
  default). A failed delivery never stops the pipeline; it logs to
  `factors_webhook_delivery_failed_total` and is available in the
  receipts table for manual replay.

## Branding and surface customization

- **Domain:** partners point their chosen DNS to the OEM ingress.
- **SDK package name:** `npm install @partner-acme/factors-sdk` is a
  thin re-export of `@greenlang/factors-sdk` with the partner UA string
  and base URL baked in. Reach out to GreenLang ops to mint the
  scoped package.
- **Explain payloads:** `why_chosen` and `assumptions` text templates
  are per-tenant. Override them via `method_pack.audit_text_template`
  when you register a custom method pack.
- **Logo + palette:** the hosted factor-explorer UI (`packages/factor-explorer`)
  reads `THEME_CONFIG` at bootstrap; replace the file at build time.
- **Support contact:** `entitlements.support_routing` controls whether
  a tenant's /support endpoint routes to partner-acme or GreenLang's
  pooled queue.

## Security + compliance baseline

OEM deployments inherit every control from SEC-001 through SEC-011
(JWT, RBAC, TLS 1.3, audit logging, Vault secrets, PII detection). The
extra OEM-specific controls are:

- Partner signs the GreenLang OEM DPA — covers sub-processor notice,
  breach response SLA, and data-localisation obligations.
- Tenant isolation drills run quarterly; proof is an audit-bundle
  query that returns only tenant-owned factors when scoped to the
  tenant role.
- Signing keys are partner-specific; GreenLang never reuses a signing
  key across OEM deployments.
- SOC 2 Type II evidence (SEC-009) is partitioned per OEM so partners
  can reuse the control narrative for their own audits.

## Support boundary

| Issue | Partner ops | GreenLang ops |
| --- | --- | --- |
| Ingress / DNS / TLS certs | ✅ | |
| Tenant provisioning | ✅ | |
| Container image / helm chart | | ✅ |
| Upstream source change | | ✅ (regulatory-watch) |
| Custom method-pack approval | joint | joint |
| Licensed-pack entitlement | | ✅ |
| Breach response | joint (DPA) | joint (DPA) |

Escalation contacts live in `docs/factors/support_boundaries_and_severity.md`;
OEM deployments get the same Sev 1 SLA as managed Enterprise tenants
(15-minute acknowledgement, 4-hour mitigation target).
