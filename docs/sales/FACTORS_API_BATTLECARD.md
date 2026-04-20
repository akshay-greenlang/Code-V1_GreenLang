# GreenLang Factors API — Sales Battlecard

> **One-liner.** 25,000+ emission factors across EPA, DESNZ, eGRID, IEA, IPCC, Green-e, TCR, and CBAM — with version pinning, source lineage, and a three-label coverage system (Certified / Preview / Connector-only). Open-core with an Enterprise tier.

**Target buyer.** Developers building climate apps; consultants tired of maintaining Excel factor libraries; ESG SaaS vendors who want to stop curating factors themselves and plug into ours.

---

## Why buyers care

| Pain | How the Factors API fixes it |
|---|---|
| "Which factor did we actually use, and where did it come from?" | Every factor carries source, edition ID, license class, publication date. `factor_source` + `factor_vintage` are serialized into every emission result. |
| "Our factor catalog is 3 Excel workbooks and someone's head." | 25,000+ factors in the hosted catalog + SDK (Python + TypeScript). No spreadsheet. |
| "The auditor wants to see the factor's source PDF." | Evidence Vault stores it. Bundle export returns raw source + parser log. |
| "Regulators just changed the default factor for our fuel." | Source-watch pipeline detects the change within 24 hours of publication and tags a new edition. Customers pinned to old edition stay reproducible; new runs opt-in. |
| "We need a factor for a niche supplier that nobody publishes." | Connector-only coverage + tenant overlays: customer can attest their own factor. The label is honest about what's Certified vs what's a customer-provided estimate. |

## Product shape

- **Hosted API** — auth (API key / JWT), rate limits, usage metering, OpenAPI spec. Backed by Postgres + pgvector for semantic search.
- **Python SDK** — `greenlang-factors-sdk` on PyPI. Zero runtime dependencies.
- **TypeScript SDK** — `@greenlang/factors` on npm.
- **Catalog explorer UI** — `/factors/explorer` (Phase 5 deliverable).
- **Public catalog status** — `GET /api/v1/factors/status/summary` returns counts by label (Phase 5 deliverable).

## Coverage labels (FY27 proposal discipline)

| Label | Meaning |
|---|---|
| **Certified** | Signed off by curation + license review; safe for regulatory filings. |
| **Preview** | In review / pending signoff; usable with explicit disclosure. |
| **Connector-only** | Rights-restricted; accessible only through pre-licensed customer connectors. |

Count three **separate** numbers publicly (not one vanity metric):

1. Factors **catalogued**
2. Factors **QA-certified**
3. Factors **usable through the API** in production

## Pricing

| Tier | Price | Good for | Inclusions |
|---|---|---|---|
| **Developer** (open-core) | Free | Individual dev or ≤ 5-person team | 10k API calls/month, Certified factors, rate-limited, community support |
| **Startup** | $500/month | ≤ 50-person climate SaaS | 100k API calls/month, Certified + Preview, email support |
| **Enterprise** | from $50,000/year | Customers building on GreenLang | 10M+ API calls/month, all labels, tenant overlays, SLA 99.9%, dedicated support, on-prem option |
| **Consultant** | Custom | White-label reseller | Volume discount, co-branded, reseller margin |

The Factors API is also **bundled** into every Comply subscription — Comply customers never buy it separately.

## Proof points (repo assets)

- `greenlang/factors/source_registry.py` — 17-field source registry (license class, watch cadence, legal signoff, redistribution rights).
- `greenlang/factors/ingestion/parsers/` — 8 parsers (EPA GHG Hub, eGRID, DESNZ, IPCC, Green-e, TCR, GHG Protocol, CBAM).
- `greenlang/factors/matching/` — semantic embedding + pgvector + LLM rerank + gold-set evaluation (511 gold cases).
- `greenlang/factors/quality/` — dedupe, cross-source checks, review queue, release signoff, audit export.
- `greenlang/factors/watch/` — per-source change detection + classification + rollback editions.
- `deployment/helm/greenlang-factors/` — Helm chart for the hosted API (API + worker + migration job + HPA + ingress).
- 57 passing factor tests in `tests/factors/`.

## Competitive landscape

| Category | Examples | Our edge |
|---|---|---|
| **Open academic data** | IPCC AR6 + EPA eGRID public downloads | We do the parsing, QA, versioning, and API for you. Don't build this in-house. |
| **Commercial factor library** | Ecoinvent, Sphera Gabi | Much larger (Ecoinvent: 20k+ LCIs). Our edge is breadth across regulatory sources + CBAM-native data + transparent labeling. |
| **SaaS vendor bundled data** | Watershed, Persefoni | Black-box factor choices. Our factors are named, source-linked, and license-labeled per record. |
| **LLM-generated factors** | Various startups | Hallucinated. We don't use LLMs for factor generation — only for matching activity text to candidate factors, with human review. |

## FAQs

**Q: Is the API really "open-core"?** Developer tier is free and covers individual development + small teams. Production traffic or Preview-label access needs a paid tier. SDKs are open-source (Apache 2.0).

**Q: Can we self-host the catalog?** Yes, Enterprise tier. Ship the Helm chart + weekly edition-delta updates.

**Q: What's in the public explorer?** All Certified + Preview factors, with filters by source, region, sector, scope, and coverage label. Connector-only factors are listed with their access path but not their values.

**Q: How often is the catalog updated?** Per-source cadence: weekly (eGRID, DESNZ), monthly (EPA GHG Hub, Green-e, TCR), on-publication (IPCC, CBAM rule updates). Every change tagged as a new edition.

**Q: Licensing worries?** Every factor row has an explicit `license_class` and `redistribution_allowed` boolean. Our own redistribution complies with source terms; customer overlays stay on customer infrastructure.

---

*Last updated: 2026-04-20. Owner: GreenLang product (Factors). Source: `GreenLang_Factors_FY27_Product_Development_Proposal.pdf` + `FY27_vs_Reality_Analysis.md` §3.1.*
