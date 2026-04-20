# CBAM Pilot Runbook

**Purpose.** Step-by-step guide for a 2-week CBAM pilot engagement against a real customer quarter.
**Audience.** Implementation engineer + customer compliance officer.
**Assumes.** GreenLang installed (`pip install -e ".[full]"`) and `gl --version` works.

---

## 1. Prerequisites

Collect these from the customer *before* Day 0:

| Item | Format | Why |
|---|---|---|
| **Covered-goods shipment list** | CSV or XLSX | Each row = one import declaration. Required columns: `cn_code`, `supplier_id`, `quantity`, `unit`, `country_of_origin`, `installation_id`. |
| **Supplier master data** | YAML | Per-supplier: GPS coords, production-process hints, electricity mix disclosures. |
| **Prior declarations** (if any) | XML | Used to reconcile deltas. |
| **Company info** | JSON | Entity identifier, EORI number, EU member-state of import, authorised-declarant status. |
| **Custom factors** (optional) | JSON | Customer-specific emission factors that must override catalog defaults. Loaded via tenant overlay. |

Sample files live under `applications/GL-CBAM-APP/CBAM-Importer-Copilot/examples/`.

## 2. 3-agent CBAM pipeline

`gl run cbam` dispatches a **3-agent pipeline**. Each agent writes to Ledger + Vault.

```
shipments.csv ─▶  [routing_agent]     ─ classifies CN codes, flags deminimis
                          │
                          ▼
                  [emissions_agent]   ─ factor lookup, deterministic CO2e
                          │
                          ▼
                  [reporting_agent]   ─ XML serialization, summary XLSX
                          │
                          ▼
                  cbam_report.xml + audit/* + evidence/*
```

Agent source lives in `applications/GL-CBAM-APP/CBAM-Importer-Copilot/agents/`. Ledger writes are recorded automatically when `--audit` is passed.

## 3. Daily plan

### Day 0 — Kick-off

- 60-min call with customer CFO/compliance, their consultant (if any), and our implementation engineer.
- Review prerequisites (section 1). Confirm EORI number + quarter under review.
- Schedule Day 10 sign-off review.

### Day 1-2 — Data load

1. Place customer files in `./pilot/<customer>/inputs/`.
2. Validate schemas:
   ```bash
   gl validate shipments ./pilot/<customer>/inputs/shipments.csv
   gl validate suppliers ./pilot/<customer>/inputs/suppliers.yaml
   ```
3. First-pass CBAM run:
   ```bash
   gl run cbam \
     ./pilot/<customer>/inputs/config.yaml \
     ./pilot/<customer>/inputs/shipments.csv \
     ./pilot/<customer>/out/run1
   ```
4. Open `./pilot/<customer>/out/run1/report_summary.xlsx` with customer. Highlight any rows flagged as **Preview** (factor class) — those need supplier data.

### Day 3-4 — Gap report

Produce `gap_report.md`:

- **Missing data:** shipments without GPS coords, CN-code refinements, or supplier process hints.
- **Low-quality factors:** any row using a Connector-only factor (flag for remediation).
- **Rule violations:** items the Policy Graph flagged as out-of-scope or needing a derogation.
- Run:
  ```bash
  gl policy-graph applies-to \
    '{"hq_country":"IN","operates_in":["EU"],...}' \
    '{"category":"cbam_covered_goods","goods":"steel"}' \
    EU 2026-06-30 \
    --output ./pilot/<customer>/out/applicability.json
  ```
  — this confirms CBAM applies and prints the quarter-end deadline.

### Day 5-8 — Remediation

- Customer provides missing data. Update `suppliers.yaml`.
- If any customer-specific factors need to override catalog defaults:
  ```bash
  gl factors ingest-paths \
    --sqlite ./pilot/<customer>/factors.sqlite \
    --edition-id "<customer>-q2-2026-overlay" \
    ./pilot/<customer>/inputs/custom_factors/*.json
  ```
- Re-run `gl run cbam` into `out/run2/`.
- Continue until gap list is zero / flagged-acceptable.

### Day 9 — Final run + evidence

```bash
gl run cbam \
  ./pilot/<customer>/inputs/config.yaml \
  ./pilot/<customer>/inputs/shipments.csv \
  ./pilot/<customer>/out/final \
  --audit

gl evidence bundle \
  --case "<customer>-q2-2026" \
  --sqlite ./pilot/<customer>/vault.sqlite \
  ./pilot/<customer>/out/final/evidence.zip

gl ledger export \
  --sqlite ./pilot/<customer>/ledger.sqlite \
  ./pilot/<customer>/out/final/ledger.json
```

### Day 10 — Sign-off review

Agenda:

- Walk the `final/cbam_report.xml` line-by-line against customer's prior declaration.
- Open the evidence bundle; show the raw source → parser log → reviewer decision chain for 3 sample rows (including one on a Preview factor).
- Show `ledger.json`: every activity traceable to its factor edition + policy-graph decision.
- Capture the customer's "sign-off" on the run.

### Day 11-14 — Handover

- Production tenant provisioned.
- Credentials stored in Vault (see `gl connect test` for each customer source system).
- Runbook (this document) checked-in to customer's team wiki.
- Year-1 pricing agreed (see `docs/sales/CBAM_BATTLECARD.md`).
- First production run scheduled for the next CBAM deadline.

## 4. Outputs

After a successful run, the `out/final/` directory contains:

| Artifact | Purpose |
|---|---|
| `cbam_report.xml` | The signed XML for submission to the importing member-state authority. |
| `report_summary.xlsx` | Operator-readable summary: shipment-level totals, factor sources, data-quality flags. |
| `audit/claims.json` | All claims the report makes, with citations. |
| `audit/lineage.json` | Activity → factor → emission trace. |
| `audit/policy_validation.json` | Policy-Graph applicability verdict + rule trace. |
| `audit/run_manifest.json` | Reproducibility manifest: pack versions, factor edition, software versions. |
| `evidence/*` | Immutable copies of the inputs. |
| `evidence.zip` | Signed Evidence Vault bundle (from `gl evidence bundle`). |
| `ledger.json` | Export of the Climate Ledger chain for this case. |

## 5. Troubleshooting

| Symptom | Likely cause | Remedy |
|---|---|---|
| `Preview factor used for N rows` | Factor for the customer's specific supplier/process is not Certified yet | Escalate to factor curation team or attach a customer-specific factor via tenant overlay |
| `Policy Graph: CBAM did not apply` | Date outside definitive period, or goods not in covered list | Double-check `cn_code` and `reporting_period_end` |
| `Ledger chain-hash drift` | Tampering or version skew between memory and SQLite | Re-run with a fresh SQLite path; investigate the source file's git diff |
| `Evidence bundle missing attachments` | `attach()` not called before `bundle()` | Verify pipeline calls `vault.attach()` for parser logs and raw inputs |

## 6. Common customer profiles (FY27)

- **Indian steel exporter to EU:** ~5,000 shipments/quarter; uses SAP S/4HANA; needs CBAM + CSRD.
- **EU-based cement importer:** Uses Oracle Fusion; needs CBAM + EU Taxonomy + CSRD.
- **Global aluminum trader:** 10,000+ shipments; Snowflake data warehouse; needs CBAM + SBTi + TCFD.

Each matches a different **Connect** connector path (SAP, Oracle, Snowflake). See `gl connect list`.

---

*Last updated: 2026-04-20. Owner: GreenLang implementation. Complement to `docs/sales/CBAM_BATTLECARD.md`.*
