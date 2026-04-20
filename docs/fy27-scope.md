# FY27 Scope Decisions

> **Purpose.** Record the *de jure* scope of FY27 as opposed to what happens to be built in the repo. When a PM or sales engineer asks "do we sell X in FY27?", this doc is the authoritative answer.
> **Audience.** PMs, sales, engineers shipping the first paid pilots.
> **Source.** `GreenLang_Climate_OS_v3_Business_Plan_2026_2031.pdf` (§FY27 scope) + `FY27_vs_Reality_Analysis.md` §6 (Phase 6 decisions).

---

## 1. FY27 launch scope (authoritative)

**Core Platform (L1–L3 substrate, must ship):**
Factors, Connect, Entity Graph, Climate Ledger, Evidence Vault, Policy Graph, Agent Runtime + Eval, SDK/API/CLI.

**Compliance Cloud (L4, commercial wedge):**
Comply (CSRD / ESRS + TCFD + SBTi + ISO 14064 + CDP + EU Taxonomy), CBAM, Scope Engine.

**FY27 business targets (from v3):**
$0.7 M end ARR · $0.3 M recognized · 8 paying logos · 12 headcount · 4,000 developers · 80 contributors · $7 M Seed.

**Beachhead:** India-linked manufacturers and exporters selling into EU / global OEM supply chains.

---

## 2. What is **NOT** FY27 (Phase 6 decisions, 2026-04-20)

### 2.1 EUDR — off FY27 scope

**Status:** out of scope. EUDR is not in v3's FY27 product list.

**Current repo footprint (for context, not for sale):**

| Artefact | Size | Role |
|---|---|---|
| `applications/GL-EUDR-APP/` | 1.4 MB | Compliance platform skeleton |
| `greenlang/agents/eudr/` | 31 MB | 40 EUDR agents (supply chain, risk, due diligence, workflow) |
| `greenlang/agents/data/eudr_traceability/` | 673 KB | Traceability connector |
| PACK-006, PACK-007 (EUDR starter/professional) | archived | Already in `_archive/07_fy29_plus_premature_packs/` |

**Decision and rationale:**

- **Keep in tree.** The 31 MB of agent code is sunk investment. The EUDR app is self-contained — zero FY27-active products (CBAM, CSRD, Comply, Scope Engine, Factors) import from `greenlang.agents.eudr` or depend on it.
- **Do not sell.** No EUDR price page, no EUDR battlecard, no EUDR entry in the Comply umbrella (`packs/eu-compliance/PACK-Comply-000-unified/`). If a pilot requires EUDR, route the deal through a Big-4 partner rather than blocking FY27 launch.
- **Revisit at FY28 Q1.** If Indian exporters pull for EUDR alongside CBAM (real signal: > 3 design-partner requests), promote it in the FY28 scope doc. Otherwise the app and agents continue to sit dormant.

**Deletion criteria (future):** If by FY28 Q2 there are still zero paid EUDR logos and zero design-partner activity, archive to `_archive/06_fy28_plus_premature_apps/` under the Phase 6 archival note.

### 2.2 Legacy v1/v2 runtime — fully retired

**Status:** the `greenlang/v1/` and `greenlang/v2/` runtime packages were archived during Phase 0 cleanup (see `_archive/08_legacy_v1_v2_runtime/`).

**Phase 6 completion of the purge:**

- Archived `greenlang/cli/cmd_v1.py` and `cmd_v2.py` (the CLI wrappers that imported the archived modules) to `_archive/08_legacy_v1_v2_runtime/cli/`.
- Removed the `_safe_add_typer("cmd_v1", ...)` and `_safe_add_typer("cmd_v2", ...)` entries from `greenlang/cli/main.py`.
- Removed the defensive `try/except ImportError` shims for `greenlang.v1.backends` / `greenlang.v2.backends` in `main.py` (no callers remain).
- Archived `.github/workflows/greenlang-v2-platform-ci.yml` and `v2-release-train.yml` to `_archive/08_legacy_v1_v2_runtime/workflows/` (they reference the archived modules).

**Out of scope for this retirement:** the per-app `v2/runtime_backend.py` files inside `GL-CBAM-APP/` and `GL-CSRD-APP/` are a **different** artefact — these are application-internal v2 backends, not the archived platform runtime. They stay.

**Re-checking gates:** if someone proposes resurrecting v1/v2, point them at `_archive/08_legacy_v1_v2_runtime/README.md` and the v3 substrate layer (`docs/REPO_TOUR.md` §2 "Core platform").

### 2.3 IoT schemas — defer to FY28

**Status:** v3 lists IoT schemas as an FY27 L1 deliverable. After reviewing the **pilot-profile matrix** in `docs/sales/CBAM_BATTLECARD.md` and `docs/sales/COMPLY_BATTLECARD.md`, no FY27 customer ingests real-time IoT streams:

| FY27 pilot profile | Data sources | Needs IoT schemas? |
|---|---|---|
| EU CBAM importer (Indian steel/aluminum) | Shipment CSV + supplier YAML + CN-code registry | No |
| CSRD filer (mid-market EU) | ERP extract (SAP/Oracle), utility bills (PDF), spend data | No |
| SB 253 Scope 1+2 (California) | Meter reads (monthly CSV), natural gas bills | No (monthly CSV, not OPC-UA) |
| Scope 3 Cat 1 (purchased goods) | AWS Cost Explorer, Snowflake spend | No |

**Decision and rationale:**

- **Defer the production IoT surface to FY28** (PlantOS). Shipping IoT schemas in FY27 would be premature and distract from the CBAM + Comply pilot push.
- **Create a placeholder module** (`greenlang/iot_schemas/`) with the OPC-UA / MQTT / Modbus canonical event schemas as Pydantic stubs. This reserves the API surface and gives FY28 PlantOS work a place to land without forcing a module creation.
- **Existing FY27-active intake paths** that touch IoT-adjacent data (BMS, SCADA, meter management in `greenlang/agents/data/`) continue to work as-is. Their schemas are legacy agent-local Pydantic models and will be aligned to `greenlang.iot_schemas` during FY28 PlantOS work.

**Trigger to accelerate:** if a design partner is explicitly conditioning a CBAM or SB 253 pilot on live OPC-UA / SCADA ingestion (not periodic CSV exports), flag it to the PM and revisit.

---

## 3. Scope matrix (authoritative)

| Module | v3 plan says | FY27 decision | Owner |
|---|---|---|---|
| Factors | FY27 | **In scope** — Phase 4 shipped hosted API | Factors team |
| Connect | FY27 | In scope — SAP/Workday/Snowflake/Databricks/AWS Cost (Phase 2.5) | Platform team |
| Entity Graph | FY27 | In scope — Phase 2.3 | Platform team |
| Climate Ledger | FY27 | In scope — Phase 2.1 | Platform team |
| Evidence Vault | FY27 | In scope — Phase 2.2 | Platform team |
| Policy Graph | FY27 | In scope — Phase 2.4 | Platform team |
| Agent Runtime + Eval | FY27 | In scope — Phase 1 | Platform team |
| SDK/API/CLI | FY27 | In scope — Phase 4 | Platform team |
| Comply (CSRD + TCFD + SBTi + ISO 14064 + CDP + Taxonomy + SB 253) | FY27 | In scope — Phase 3.1 | Applications team |
| CBAM | FY27 | In scope — Phase 3.3 (strongest wedge) | Applications team |
| Scope Engine | FY27 | In scope — Phase 3.2 | Applications team |
| **EUDR** | — | **OUT of scope** (§2.1) | — |
| **IoT schemas** | FY27 | **DEFERRED to FY28** (§2.3) | FY28 PlantOS |
| Legacy v1 / v2 runtime | — | **RETIRED** (§2.2) | archived |

---

## 4. References

- `GreenLang_Climate_OS_v3_Business_Plan_2026_2031.pdf` (root).
- `FY27_vs_Reality_Analysis.md` (root) — full gap analysis, Phase 6.
- `docs/REPO_TOUR.md` — v3 layer map.
- `_archive/README.md` — archive bucket index.
- `_archive/08_legacy_v1_v2_runtime/` — archived runtime + CLI + workflows.

---

*Last updated: 2026-04-20. Source: Phase 6 decisions executed on 2026-04-20. Owner: Akshay.*
