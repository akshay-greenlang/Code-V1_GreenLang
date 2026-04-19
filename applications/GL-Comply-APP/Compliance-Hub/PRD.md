# GL-Comply-APP — Unified Compliance Hub (PRD)

Single API/SDK/CLI orchestrating the 10 framework-specific GreenLang apps
behind one experience: customer uploads data once, Comply-Hub routes to all
applicable frameworks, produces a unified report.

## Scope (FY27)

In: applicability rules, parallel framework dispatch, result aggregation,
unified PDF/XBRL/JSON report, provenance via climate_ledger + evidence_vault.

Out (v1): custom framework authoring, tenant self-service report templates.

## Frameworks (10)

CSRD, CBAM, EUDR, GHG Protocol, ISO 14064, California SB 253, SBTi, EU
Taxonomy, TCFD, CDP.

## Public API surface (v1)

- `POST /api/v1/compliance/intake` — run all applicable frameworks
- `POST /api/v1/compliance/applicability` — rule-based framework selection
- `GET  /api/v1/compliance/frameworks` — registered adapters
- `GET  /api/v1/compliance/results/{id}` — fetch job result (COMPLY-APP 4)
- `GET  /api/v1/compliance/reports/{id}` — fetch unified report artifact

## Status (2026-04-19)

- COMPLY-APP 1 (scaffold): done
- COMPLY-APP 2 (adapter registry — 10 adapters): pending (task #16)
- COMPLY-APP 3 (orchestrator + applicability + normalizer): stub done, production wiring pending (task #17)
- COMPLY-APP 4 (FastAPI + SDK + CLI): API stub done, SDK/CLI pending (task #18)
- COMPLY-APP 5 (unified report agent): pending (task #19)
- COMPLY-APP 6 (migrations + deployment + tests): pending (task #20)
