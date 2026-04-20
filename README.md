# GreenLang

**Climate intelligence substrate — Factors, Ledger, Policy Graph, Agent Runtime — plus CBAM, CSRD, and Scope Engine applications on top.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://github.com/akshay-greenlang/Code-V1_GreenLang/releases)
[![Climate OS](https://img.shields.io/badge/Climate%20OS-Substrate%20%2B%20Applications-brightgreen.svg)](#2-the-four-layers)
[![FY27](https://img.shields.io/badge/FY27-Compliance%20Cloud-blue.svg)](#3-fy27-products)

---

## 1. What is GreenLang

GreenLang is the **climate operating system** for regulated enterprises: one auditable substrate for emission factors, activity data, and policy logic, with sector applications — CBAM, CSRD, Scope Engine — built on top.

Climate compliance is a moving target. CBAM's definitive period is live, CSRD is in year one for thousands of companies, California SB 253 hits **10 August 2026**, and India's CCTS is notifying its first ~490 obligated entities. Enterprises are trying to meet this with spreadsheets, consultants, and disconnected point tools. Factor choices are not defended, evidence is not auditable, and every new regulation forces a manual rebuild.

GreenLang is the substrate underneath, not another report generator:

- Every activity and emission is written once, **signed, and reproducible**.
- Every factor carries **source lineage, license, and review history**.
- Every claim is backed by a retrievable **evidence bundle**.
- Every regulation — CBAM, CSRD, SB 253, TCFD, SBTi, ISO 14064 — is expressed as **applicability logic**, not a frozen report template.

**Vision.** Make climate compliance as automated and reliable as financial accounting.

---

## 2. The four layers

GreenLang is organized into four layers. Layer 1 powers Layer 2, which grounds Layer 3, which powers the Layer 4 sector applications customers buy.

```
+========================================================================+
|  L4 — Sector Clouds (applications)                                      |
|    FY27: Comply · CBAM · Scope Engine                                   |
|    FY28+: SupplierOS · PCF Studio · DPP Hub · PlantOS · BuildingOS ...  |
+========================================================================+
|  L3 — Intelligence                                                      |
|    Policy Graph  ·  Agent Runtime + Eval  ·  Scenario/Benchmark         |
+========================================================================+
|  L2 — System of record                                                  |
|    Climate Ledger  ·  Evidence Vault  ·  Proof Hub (FY29)               |
+========================================================================+
|  L1 — Data foundation                                                   |
|    Factors  ·  Connect  ·  Entity Graph  ·  IoT Schemas                 |
+========================================================================+
```

### L1 — Data foundation

- **Factors** — Versioned emission-factor catalog across EPA GHG Hub, eGRID, DESNZ, IPCC, Green-e, TCR, GHG Protocol, and CBAM. Source registry with license class, redistribution rights, and watch cadence. Semantic matching (pgvector + LLM rerank). Three coverage labels: **Certified / Preview / Connector-only**.
- **Connect** — Enterprise system connectors (SAP S/4HANA, Snowflake, AWS Cost Explorer, with Workday and Databricks on the roadmap) for procurement, utility, and IoT data intake.
- **Entity Graph** — Multi-tier organization model (entity → facility → asset → meter) that every Ledger write and Policy Graph evaluation resolves against.
- **IoT Schemas** — Canonical event schemas for OPC-UA / MQTT / Modbus streams.

### L2 — System of record

- **Climate Ledger** — Append-only, content-addressed, signed record of every activity and emission. Reproducible line back to the exact Factors edition and Policy Graph decision used.
- **Evidence Vault** — Customer-facing vault for raw source artifacts, parser logs, reviewer decisions, and attached documents. One command returns a signed auditor bundle.

### L3 — Intelligence

- **Policy Graph** — Given `(entity, activity, jurisdiction, date)`, the Policy Graph returns which regulations apply, which factor classes are required, and what the reporting deadline is. Rules for CBAM, CSRD, SB 253, TCFD, SBTi, and ISO 14064 are expressed as applicability logic, not report templates.
- **Agent Runtime + Eval** — Deterministic calculation where numbers must clear audit, AI reasoning where it helps (source parsing, factor matching, policy diff). Single canonical base class, versioned agent specs, reproducible eval harness.

### L4 — Sector clouds

- **Comply** — Unified CSRD + ESRS + TCFD + SBTi + ISO 14064 + CDP + EU Taxonomy reporting. One substrate, many frameworks.
- **CBAM** — EU Carbon Border Adjustment reporting with embedded emissions calculation, XML export, and audit-ready evidence.
- **Scope Engine** — Unified Scope 1/2/3 engine with adapters for GHG Protocol, ISO 14064, SBTi, CSRD E1, and CBAM.

FY28 onward adds Supply Chain (SupplierOS, PCF Studio, DPP Hub), Operations (PlantOS, BuildingOS, DataCenter CarbonOps), Mobility (FleetOS, Freight Carbon API), Land/Water/Nature, Risk, and Finance & Markets clouds — 36 modules across 8 clouds by FY31.

---

## 3. FY27 products

FY27 launches the substrate plus the first commercial wedge: Compliance Cloud.

| Product | Status | Role |
|---|---|---|
| **Factors** | Ready | Versioned catalog + semantic matching + source watch. Hosted API with auth, rate limits, and three-label coverage dashboard is the FY27 go-to-market vehicle. |
| **CBAM** | Ready | Largest application in the repo. Operator path: `gl run cbam`. The strongest wedge for Indian exporters into EU supply chains. |
| **Comply (CSRD / ESRS + SB 253 + TCFD + SBTi + ISO 14064)** | Partial → bundling | Strong ESRS bones across 9 CSRD packs and 6 supporting apps; being unified into one Comply umbrella that shares the Ledger, Vault, and Policy Graph. |
| **Scope Engine** | Partial → packaging | Engine + adapters built; unified `gl scope compute` CLI and pack wiring in progress. |
| **Climate Ledger · Evidence Vault · Entity Graph · Policy Graph** | Stubs → hardening | Modules exist; production SQLite/Postgres backends, signed writes, bundle export, and `applies_to()` API are the substrate work for the launch. |
| **Connect** | Stubs → implementations | SAP / Snowflake / AWS connectors scaffolded; real integrations and credential-store wiring in progress. |
| **SDK / API / CLI** | Ready → consolidating | Python + TypeScript SDK publish-ready. `gl` CLI consolidation is tracked in [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md). |

**FY27 go-to-market.** India-linked EU exporters (steel, aluminium, cement, fertilizers) facing CBAM; Indian subsidiaries of EU parents facing CSRD cascade; California SB 253 Scope 1+2 reporters for August 2026; consultancies buying a Factors API instead of maintaining spreadsheets internally.

---

## 4. Quick start

### Prerequisites

- Python 3.10, 3.11, or 3.12
- pip (latest)
- (Optional) Docker for containerized deployment

### Install

```bash
# From PyPI
pip install greenlang-cli

# Or from source
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang.git
cd Code-V1_GreenLang
pip install -e ".[full]"

# Verify
gl --version
```

### Run the CBAM flagship loop

Canonical operator path: `gl run cbam`.

```bash
python -m greenlang.cli.main run cbam \
  "applications/GL-CBAM-APP/examples/sample_config.yaml" \
  "applications/GL-CBAM-APP/examples/sample_imports.csv" \
  out
```

Outputs in `out/`:

- `cbam_report.xml` — CBAM XML report
- `report_summary.xlsx` — operator-readable summary
- `audit/*` — claims, lineage, assumptions, policy validation, run manifest, checksums
- `evidence/*` — immutable copies of inputs

### Use the Factors SDK

```python
from greenlang.factors.sdk import FactorsClient

client = FactorsClient(api_key="...")
factor = client.match(
    activity="diesel combustion, stationary",
    geography="IN",
    quantity=1000, unit="L",
)
print(factor.co2e_kg, factor.source, factor.license, factor.edition)
```

The SDK and its TypeScript counterpart are zero-dependency and FY27 target the hosted Factors API.

### Repository tour

Pick one file to orient yourself: [`docs/REPO_TOUR.md`](docs/REPO_TOUR.md) maps every top-level directory to the v3 layer it implements and to its entry point.

---

## 5. Architecture at a glance

```
+------------------------------------------------------------------------+
|  Applications                                                           |
|    GL-CBAM-APP  ·  GL-CSRD-APP  ·  GL-Comply-APP  ·  GL-GHG-APP         |
|    GL-SB253-APP · GL-SBTi-APP   · GL-TCFD-APP     · GL-ISO14064-APP     |
+------------------------------------------------------------------------+
|  greenlang/                                                             |
|    L3: policy_graph/  scope_engine/  agents/  agent_runtime/  intel/    |
|    L2: climate_ledger/  evidence_vault/  provenance/                    |
|    L1: factors/  connect/  entity_graph/  data/  data_commons/          |
+------------------------------------------------------------------------+
|  Infrastructure                                                         |
|    security/  infrastructure/  monitoring/  telemetry/  db/             |
|    K8s + Helm (deployment/helm/greenlang-factors/)                      |
|    PostgreSQL + TimescaleDB + pgvector  ·  Redis  ·  S3  ·  Kong        |
+------------------------------------------------------------------------+
```

**Technology stack.** Python 3.10–3.12 · FastAPI · Pydantic v2 · PostgreSQL + TimescaleDB + pgvector · Redis · Kubernetes (EKS) · Helm · Terraform · Prometheus / Grafana / OpenTelemetry · JWT / RBAC / AES-256-GCM / TLS 1.3 / HashiCorp Vault · pytest · OpenAI & Anthropic LLM integration.

---

## 6. FY31 roadmap

By FY31 the roadmap is **36 modules across 8 clouds**, $95M end ARR target, 520 paying logos. Launch cadence:

| FY | Focus | New products |
|---|---|---|
| **FY27** | Substrate + Compliance wedge | Factors, Connect, Entity Graph, Climate Ledger, Evidence Vault, Policy Graph, Agent Runtime + Eval, SDK/API/CLI, Comply, CBAM, Scope Engine |
| **FY28** | Supply Chain + Operations | SupplierOS, PCF Studio, DPP Hub, PlantOS, BuildingOS |
| **FY29** | Ops + Mobility + Risk + Finance | DataCenter CarbonOps, PowerOS, FleetOS, Proof Hub, RiskOS, FinanceOS, MRV Studio |
| **FY30** | Land/Water/Nature + Advanced ops | AgriLandOS, WaterOS, Methane & Nitrogen Monitor, FlexOS, Microgrid Planner, Freight Carbon API, Transition Finance Studio, Carbon Markets Hub, Adaptation Planner |
| **FY31** | Closing the portfolio | Nature/TNFD Hub, CityOS, CDR Portfolio Manager, Circularity Hub |

For strategy detail, see the `GreenLang_Climate_OS_v3_Business_Plan_2026_2031.pdf` at repo root. For the current reality gap, see [`FY27_vs_Reality_Analysis.md`](FY27_vs_Reality_Analysis.md).

---

## 7. Development

```bash
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang.git
cd Code-V1_GreenLang
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install -e ".[dev]"

# Tests
pytest tests/
pytest tests/factors/       # Factors-specific
pytest --cov=greenlang      # With coverage

# Code quality
ruff check .
black greenlang/
mypy greenlang/
```

All dependencies live in a single root `pyproject.toml` (21 optional groups: `analytics`, `cli`, `data`, `pdf`, `visualization`, `nlp`, `graph`, `llm`, `ml`, `vector-db`, `ai-full`, `server`, `security`, `sbom`, `supply-chain`, `monitoring`, `test`, `dev`, `doc`, `full`, `all`). Pre-commit is configured at repo root (`.pre-commit-config.yaml`).

---

## 8. License & contact

- **License:** Apache 2.0 — see [LICENSE](LICENSE).
- **Issues:** [GitHub Issues](https://github.com/akshay-greenlang/Code-V1_GreenLang/issues)
- **Security:** security@greenlang.io
- **Homepage:** https://greenlang.io

---

**Version:** 0.3.0 · **Stage:** Pre-Seed · **Last updated:** 20 Apr 2026

**GreenLang** — Measure what matters. Write it once. Clear audit.
