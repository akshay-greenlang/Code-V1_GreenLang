# GreenLang Scope Engine — Sales Battlecard

> **One-liner.** One deterministic engine for Scope 1, 2, and 3 GHG emissions, with five native framework projections (GHG Protocol, ISO 14064, SBTi, CSRD E1, CBAM). Same activity data in, multiple framework views out.

**Target buyer.** Any enterprise doing Scope 1+2+3 accounting — especially those reporting under *multiple* frameworks simultaneously. California SB 253 filers for Aug 2026 are the highest-intent FY27 segment.

---

## Why it exists

Enterprises run into **two calculators** problems:

1. One tool for Scope 1/2 and another for Scope 3 — numbers don't reconcile.
2. One calculation for GHG Protocol and a separate, re-done calculation for CSRD E1, a third for CBAM. The same activity gets recomputed with subtly different conventions and nobody can explain the delta.

GreenLang Scope Engine eliminates both. One `ComputationRequest`, one `ScopeComputation` output, N framework projections — each with the same underlying numbers, differences only in **presentation**, never in calculation.

## What makes it defensible

| Property | How |
|---|---|
| **Deterministic** | Python + Decimal arithmetic, no floats, no LLMs in the calc path. |
| **Framework-aware** | 5 adapters (GHG Protocol, ISO 14064, SBTi, CSRD E1, CBAM) project from one computation to framework-native rows. |
| **Cross-framework reconciled** | Same `ScopeComputation` backs every framework view. If CBAM and CSRD show different totals, it's a presentation bug — not a disagreement between numbers. |
| **GWP-pinned** | AR4 / AR5 / AR6 100-year and 20-year horizons selectable per run. Factor + GWP are both captured in the formula hash. |
| **Consolidation-aware** | Equity share vs operational control vs financial control — selectable per entity. |
| **Auditable** | Every `EmissionResult` carries a SHA-256 `formula_hash`. The computation itself has a `computation_hash` that the Climate Ledger signs. |
| **Fast** | Built-in dispatcher routes activities to the right Scope 1/2/3 primitive agents; parallelizes per-activity calculation. |

## Who uses it

| Segment | Typical run |
|---|---|
| **California SB 253** filer ($1B+ revenue) | Scope 1+2 via GHG Protocol view; Scope 3 from FY27 onward. First filing deadline Aug 2026. |
| **EU CBAM** importer | Scope 1+2+3 for embedded emissions in covered goods; projected into the CBAM adapter for XML output. |
| **CSRD E1** filer | Scope 1+2+3 projected into CSRD E1 rows for double-materiality disclosures. |
| **SBTi** target setter | Scope 1+2+3 projected into SBTi tracking rows (near-term + net-zero trajectory). |
| **ISO 14064** verification candidate | Same numbers, ISO-native structure for external verifier. |

## Competitive landscape

| Competitor category | Typical gap | Scope Engine advantage |
|---|---|---|
| **Spreadsheet + consultant** | Every quarter rebuilt from scratch | Deterministic rerun from same inputs; hash-verified reproducibility |
| **Point Scope 3 tool** | Scope 1+2 lives elsewhere, numbers diverge | One engine, one ScopeComputation for all scopes |
| **ESG SaaS (Watershed, etc.)** | Black box; auditors see a PDF, not a derivation | `EmissionResult.formula_hash` exposes the full derivation per-gas |
| **In-house analyst** | ~$120k/yr fully loaded + slow | Engine + 5 adapters + CLI + Python SDK — analyst reallocated to strategy |

## Proof points (repo assets)

- `greenlang/scope_engine/service.py` — the orchestrator (~200 lines, deterministic).
- `greenlang/scope_engine/adapters/` — 5 framework adapters (`ghg_protocol`, `iso_14064`, `sbti`, `csrd_e1`, `cbam`).
- `greenlang/scope_engine/factor_adapter.py` — pulls factors from the GreenLang Factors catalog with tenant-overlay support.
- `tests/scope_engine/test_framework_matrix.py` — activity × 5-framework matrix test (8 passing).
- `gl scope compute <request.json>` — the CLI entry point (see `docs/CLI_REFERENCE.md`).
- Postgres migration `V437__scope_engine_computations.sql` — multi-tenant persistence.

## Integration with the rest of the substrate

- **Factors:** Scope Engine reads factor values from `greenlang/factors/` (the FY27 flagship). Every computation is pinned to a specific factor edition.
- **Policy Graph:** The Comply orchestrator calls `PolicyGraph.applies_to()` *first*, then runs Scope Engine only for the frameworks that actually apply.
- **Climate Ledger:** Every `ScopeComputation.computation_hash` is appended to the ledger. Reproducibility means anyone can re-run later and re-verify.
- **Evidence Vault:** Each scope computation is written into the case's evidence bundle, so auditors get the numbers and the derivation in one package.

## Pricing

- **Sold inside Comply tiers.** Scope Engine is included in every Comply subscription (Essentials +).
- **Standalone pack** (`PACK-041-scope-1-2-complete` + `PACK-042-scope-3-starter`): $50k/yr Essentials, scaling with Comply tier.
- **API metering (Factors-first buyers):** included up to 1M activity computations/month; overage per `pricing_model.md`.

## Typical pilot (1 week standalone)

- **Day 1:** Map customer's activity taxonomy to `ActivityRecord` schema.
- **Day 2:** `gl scope compute` against one prior quarter.
- **Day 3–4:** Reconcile against the customer's existing tool — explain every delta via `formula_hash` trace.
- **Day 5:** Framework views demo (GHG Protocol, ISO 14064, CSRD E1, SBTi, CBAM) from the same computation. Sign-off.

Scope Engine is typically *not* sold standalone — it's a substrate piece. Customers discover it during a CBAM or Comply pilot and end up using it for everything.

## FAQs

**Q: Can we swap in our own factors?** Yes — tenant overlay in the FactorAdapter layer.

**Q: What if our activity data is dirty?** Use `gl connect extract --dry-run` for intake validation, then `gl scope compute`. The engine is strict about units, years, and region codes (raises rather than silently accepting).

**Q: How do we handle consolidated group reporting?** Set `consolidation: financial_control` (or `equity_share` / `operational_control`) on the request. The engine applies the correct boundary.

**Q: Can the engine output directly to disclosure templates?** Framework views are structured rows. Downstream packs (CSRD, CBAM, SBTi) project those into XML / XBRL / questionnaire formats.

---

*Last updated: 2026-04-20. Owner: GreenLang sales (Scope Engine). Complement: `docs/sales/COMPLY_BATTLECARD.md`, `docs/sales/CBAM_BATTLECARD.md`.*
