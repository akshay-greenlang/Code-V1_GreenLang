# Factors Coverage Dashboard

The Coverage Dashboard is the public view of what the catalog contains, at what quality, and under which channel. Accessible at `https://app.greenlang.io/coverage` or programmatically at `GET /v1/coverage`.

The dashboard exists so prospective customers can answer "do you cover my use case" before signing up, and so existing customers can monitor the catalog for new releases.

---

## Channel semantics

Every factor belongs to exactly one channel. The dashboard counts factors per channel per jurisdiction.

| Channel | Definition | What it counts |
|---|---|---|
| **Certified** | Factors promoted into a Certified edition after methodology + QA + legal signoff. Full SLA. Eligible for regulated disclosures. | Factors with `status == "active"` in the latest `builtin-v<x.y>` Certified edition. |
| **Preview** | Factors published but not Certified. Usable for exploratory work; NO SLA; NOT eligible for regulatory filing. Licensed-connector sources that have not cleared Certified gating stay here. | Factors with `status == "under_review"` or in `preview-v<x.y>` editions. |
| **Connector-only** | Factors NOT bundled into any edition. Resolution routes through a live BYO-credentials connector (ecoinvent, IEA, Electricity Maps, EC3, pre-contract Green-e / GLEC / TCR). Reproducibility depends on the customer's upstream subscription cadence. | Connector endpoints registered in the source catalog; not counted as catalog factors. |

A single jurisdiction may have factors in all three channels — e.g., India Scope 2 has Certified (CEA), Preview (some state-level provisional factors), and Connector-only (Electricity Maps real-time grid intensity).

---

## Dimensions counted

For each `factor_family` and `jurisdiction.country`, the dashboard shows:

- Total factor count by channel.
- FQS band distribution (high >=80, medium 60-79, low <60).
- Minimum / median / maximum `valid_to` date (data freshness signal).
- Distinct `source_id` count.
- Licensing class breakdown (`open` vs `licensed_embedded` vs `customer_private` vs `oem_redistributable`).

---

## What's counted and what isn't

| Counted | Not counted |
|---|---|
| Active Certified factors | Draft factors (not yet approved) |
| Active Preview factors | Retired factors (archive only) |
| Deprecated factors (still servable) | Connector-only factor values (no registry entry) |
| Customer-private factors (visible to tenant only) | Superseded factor versions (subsumed by latest) |

`customer_private` factors appear only in the tenant's private view of the dashboard. They are never aggregated into public counts.

---

## Example output

```json
GET /v1/coverage?country=IN&factor_family=electricity
{
  "country": "IN",
  "factor_family": "electricity",
  "channels": {
    "certified": { "count": 12, "fqs_bands": { "high": 8, "medium": 4, "low": 0 } },
    "preview":   { "count": 3,  "fqs_bands": { "high": 1, "medium": 2, "low": 0 } },
    "connector_only": { "sources_available": ["electricity_maps"] }
  },
  "freshness": { "oldest_valid_to": "2023-03-31", "newest_valid_to": "2025-03-31" },
  "sources": ["india_cea_co2_baseline", "india_ccts_baselines"],
  "licensing_breakdown": { "open": 15, "licensed_embedded": 0, "customer_private": 0 }
}
```

---

## How partners should use this

- **Pre-sales**: "Do you cover Indonesia Scope 2?" → `GET /v1/coverage?country=ID&factor_family=electricity`.
- **Audit readiness**: "How many of my Scope 3 activities land on Certified vs Preview?" → batch resolve, count by `release_version`.
- **Data quality gate**: "Accept only Certified + FQS >= 80" → filter by channel + composite_fqs.

---

## Relationship to the roadmap

A jurisdiction tagged "connector-only at v1" in [`roadmap.md`](roadmap.md) will show only the connector entry until a contract closes or open-class coverage is added. The dashboard and roadmap are the paired source of truth for coverage commitments.

---

## Related

- [`concepts/edition.md`](concepts/edition.md), [`licensing.md`](licensing.md), [`roadmap.md`](roadmap.md).
- [`api-reference/sources.md`](api-reference/sources.md).
