# GreenLang Repository Tour

**Last updated:** 2026-04-20
**Purpose:** In 5 minutes, a new engineer, design partner, or investor should understand where every part of the product lives and which v3 layer it implements.

> For the historical record of *what was moved out* of the main tree, see [`_archive/README.md`](../_archive/README.md) (12 archive buckets covering ~1.3GB of previous sprawl).
> For v3 product strategy, see `GreenLang_Climate_OS_v3_Business_Plan_2026_2031.pdf` at repo root.
> For the FY27 reality gap analysis, see [`FY27_vs_Reality_Analysis.md`](../FY27_vs_Reality_Analysis.md).

---

## 1. The v3 four-layer architecture

| Layer | What it is | FY27 products |
|---|---|---|
| **L1 — Data foundation** | Normalized climate data model | Factors, Connect, Entity Graph, IoT schemas |
| **L2 — System of record** | Auditable activity/emissions/evidence records | Climate Ledger, Evidence Vault |
| **L3 — Intelligence** | Regulation-aware AI workflows | Policy Graph, Agent Runtime + Eval |
| **L4 — Sector clouds** | Commercial applications on the substrate | Comply, CBAM, Scope Engine |

The map below ties each top-level directory to the layer it implements and the entry point that exposes it.

---

## 2. Top-level directory map

### Core platform — `greenlang/`

| Subdirectory | v3 layer | Purpose | Entry point |
|---|---|---|---|
| `factors/` | L1 | **FY27 flagship.** Source registry, ingestion + parsers (EPA, eGRID, DESNZ, IPCC, Green-e, TCR, GHG Protocol, CBAM), semantic matching (pgvector + LLM rerank), quality pipeline, watch/release engine, ETL, GA billing, tier enforcement, SDK. | `gl factors *` |
| `connect/` | L1 | Enterprise system connectors (SAP S/4HANA, Snowflake, AWS Cost Explorer — currently stubs; Phase 2.5). | *(pending `gl connect`)* |
| `entity_graph/` | L1 | Entity → facility → asset → meter graph. In-memory only; Phase 2.3 adds Postgres. | *(pending `gl entity`)* |
| `data/` | L1 | Emission-factor YAML (~25.6k lines across 7 expansion phases) + built-in factor DB. | used by `factors/` |
| `data_commons/` | L1 | Shared data utilities (provenance, metrics, config, router, hash). | library |
| `climate_ledger/` | L2 | Append-only signed ledger for activities + emissions. Memory backend today; SQLite/Postgres land in Phase 2.1. | *(pending `gl ledger`)* |
| `evidence_vault/` | L2 | Raw source + parser log + reviewer decision bundle. Bundle export + S3 backend land in Phase 2.2. | *(pending `gl evidence`)* |
| `provenance/` | L2 | Lower-level provenance primitives reused by Ledger + Vault. | library |
| `policy_graph/` | L3 | `applies_to(entity, activity, jurisdiction, date)` API. Rules wiring to packs is Phase 2.4. | *(pending `gl policy applies-to`)* |
| `governance/` | L3 | OPA policy enforcer, EU/EPA compliance rule bundles. Source of rules the Policy Graph will read. | library |
| `scope_engine/` | L3 | Unified Scope 1/2/3 engine with adapters for GHG Protocol, ISO 14064, SBTi, CSRD E1, CBAM. Typer CLI exists but is not yet mounted to `gl` (Phase 3.2). | *(pending `gl scope compute`)* |
| `agents/` | L3 | ~45 active agents (post-cleanup) + 6 base classes slated for consolidation to `IntelligentAgentBase` in Phase 1.2. | library |
| `agent_runtime/`, `execution/`, `core/` | L3 | DAG orchestrator, async runtime, pipeline resilience. | library |
| `intelligence/` | L3 | LLM providers (OpenAI, Anthropic), RAG engine, budget/session runtime. | library |
| `cli/` | all | Typer apps under `cmd_*.py` + `main.py`. See [`CLI_REFERENCE.md`](CLI_REFERENCE.md). | `gl` |
| `sdk/`, `factors/sdk/` | L1 | Python SDK (v1.0.0, publish-ready). TypeScript SDK under `sdk/ts/`. | `pip install greenlang-factors-sdk` |
| `infrastructure/` | infra | Logging, audit service, storage, secrets, middleware. | library |
| `security/` | infra | JWT auth, RBAC, encryption, PII detection — SEC-001..011 complete. | library |
| `monitoring/`, `telemetry/` | infra | Prometheus, Grafana, OpenTelemetry integration — OBS-001..005 complete. | library |
| `db/` | infra | DB access layer. Migrations live in `deployment/db/migrations/` (128 today; Phase 2.1–2.3 add V406–V408). | library |
| `schemas/`, `specs/`, `validation/` | infra | Shared base classes (`GreenLangBase` etc.), JSON schemas, validators. | library |
| `exceptions/` | infra | 60 centralized exception classes across 11 domain modules. | library |
| `commercial/`, `ecosystem/`, `integration/`, `io/`, `cache/`, `auth/`, `config/`, `utilities/`, `utils/`, `testing/`, `tests/`, `templates/`, `extensions/` | infra | Supporting libraries. | library |

### Commercial applications — `applications/`

All FY27 L4 Compliance Cloud apps. Plant/heat/FY28+ apps were archived to `_archive/06_fy28_plus_premature_apps/`.

| App | FY27 role | Status |
|---|---|---|
| `GL-CBAM-APP/` | Primary commercial wedge | **Ready 70%** — 332MB with K8s manifests, tests, demo data. Phase 3.3 adds battlecard + runbook. |
| `GL-CSRD-APP/` | Comply core | Production |
| `GL-Comply-APP/` | Comply umbrella | **Stub (361KB)** — Phase 3.1 expands this to bundle CSRD + CBAM + SB253 + supporting apps |
| `GL-GHG-APP/` | Scope Engine source | Production |
| `GL-SB253-APP/` | US Comply wedge (CA SB 253, Aug 2026 deadline) | Built |
| `GL-SBTi-APP/`, `GL-TCFD-APP/`, `GL-ISO14064-APP/`, `GL-CDP-APP/`, `GL-Taxonomy-APP/` | Comply satellites | Built — Phase 3.1 merges under Comply |
| `GL-EUDR-APP/` | Off v3 scope | **Phase 6.1 decision pending** — archive or officially add to Comply |

### Compliance packs — `packs/`

Down from 50 packs to 14 FY27-active packs. Remaining 36 are in `_archive/07_fy29_plus_premature_packs/`.

| Subdirectory | Active packs | Purpose |
|---|---|---|
| `packs/eu-compliance/` | PACK-001..005, 012..017 (11) | CSRD starter/pro/enterprise, sector CSRD (finance, manufacturing, retail), double materiality, ESRS E1 climate, ESRS full coverage, CBAM readiness, CBAM complete |
| `packs/ghg-accounting/` | PACK-041, 042, 043 (3) | Scope 1-2 complete, Scope 3 starter, Scope 3 complete |

### Deployment — `deployment/`

- **Docker:** `deployment/docker/` (Dockerfiles incl. `Dockerfile.factors-service`, `Dockerfile.api`)
- **Helm charts:** `deployment/helm/greenlang-factors/` (migration job, API, worker, ingress, HPA, NetworkPolicy) — Phase 4.3 tests end-to-end
- **K8s manifests:** per-application, also inside each `GL-*-APP/`
- **Terraform IaC:** `deployment/terraform/`
- **DB migrations:** `deployment/db/migrations/` (128 today; new V406–V408 land in Phase 2)

### Frontend — `frontend/`

React/TypeScript governance UI (AdminPage, GovernancePage, RunsPage, WorkspacePage). **No Factor Explorer today** — Phase 5.1 adds it.

### Tests — `tests/`

Root-level test suites (unit, integration, load). Factors tests (57 files) live under `tests/factors/`. Gold-label evaluation set at `tests/factors/fixtures/gold_eval_full.json` (511 cases; curation is Phase 5.2).

### Scripts — `scripts/`

Migration and utility scripts: shared-schema migration, config-enum consolidation, requirements consolidation, logging standardization, etc.

### Other top-level directories

| Directory | Purpose |
|---|---|
| `config/` | Runtime and environment configs |
| `datasets/` | Reference datasets |
| `docs/` | Docs hub — this file, `CLI_REFERENCE.md`, `ARCHITECTURE.md`, business/pricing docs, etc. |
| `examples/` | Example pipelines |
| `tasks/` | Task specs |
| `tools/` | Developer tooling (archived binaries moved to `_archive/12_large_binaries/`) |
| `greenlang-normalizer/` | Standalone normalizer module (its own pyproject.toml + tests) |
| `_archive/` | 12 buckets of archived content (~1.3GB) — see [`_archive/README.md`](../_archive/README.md) |

---

## 3. FY27 product → code path cheat sheet

| FY27 product (v3) | Code path | Entry point (today / planned) |
|---|---|---|
| **Factors** | `greenlang/factors/` | `gl factors inventory/manifest/ingest-*` |
| **Connect** | `greenlang/connect/` | *pending* `gl connect` |
| **Entity Graph** | `greenlang/entity_graph/` | *pending* `gl entity` |
| **Climate Ledger** | `greenlang/climate_ledger/` | *pending* `gl ledger` |
| **Evidence Vault** | `greenlang/evidence_vault/` | *pending* `gl evidence bundle --case <id>` |
| **Policy Graph** | `greenlang/policy_graph/` + `greenlang/governance/` + pack rules | *pending* `gl policy applies-to` |
| **Agent Runtime + Eval** | `greenlang/agents/` + `agent_runtime/` + `execution/` + `intelligence/` | library |
| **SDK / API / CLI** | `greenlang/factors/sdk/`, `sdk/`, `cli/` | `gl`, `pip install greenlang-factors-sdk` |
| **Comply** | `applications/GL-Comply-APP/` + `applications/GL-CSRD-APP/` + related | *pending* `gl comply run` |
| **CBAM** | `applications/GL-CBAM-APP/` + packs 004/005 | `gl run cbam` |
| **Scope Engine** | `greenlang/scope_engine/` + packs 041/042/043 | *pending* `gl scope compute` |

---

## 4. Root-level files worth knowing

| File | Purpose |
|---|---|
| `README.md` | Top-level pitch (Phase 0.1 rewrite updates this to v3 positioning) |
| `pyproject.toml` | Single source for all deps (21 optional groups). Entry points: `gl`, `greenlang` |
| `FY27_vs_Reality_Analysis.md` | 19 Apr 2026 reality gap analysis (this tour is its companion) |
| `GreenLang_Climate_OS_v3_Business_Plan_2026_2031.pdf` | v3 strategy doc |
| `GreenLang_Factors_FY27_Product_Development_Proposal.pdf` | FY27 Factors proposal |
| `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE` | Standard repo metadata |
| `Dockerfile`, `docker-compose.yml`, `Makefile` | Developer tooling |
| `.pre-commit-config.yaml` | Consolidated pre-commit config (single source; 20 duplicate per-module configs were archived) |

---

## 5. Where to go next

- **Want to add a CLI command?** See [`CLI_REFERENCE.md`](CLI_REFERENCE.md) §6.
- **Want to understand the deployment stack?** See `docs/ARCHITECTURE.md` and `deployment/helm/`.
- **Want to know what's archived and why?** See `_archive/README.md`.
- **Want the roadmap?** See [`FY27_vs_Reality_Analysis.md`](../FY27_vs_Reality_Analysis.md) for phased to-do and v3 PDF for the FY31 endpoint.

> This tour is a living document. When you add, rename, or delete a top-level directory, update this file.
