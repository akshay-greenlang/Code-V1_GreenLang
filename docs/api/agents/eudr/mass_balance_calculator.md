# AGENT-EUDR-011: Mass Balance Calculator API

**Agent ID:** `GL-EUDR-MBC-011`
**Prefix:** `/v1/eudr-mbc`
**Version:** 1.0.0
**PRD:** PRD-AGENT-EUDR-011
**Regulation:** EU 2023/1115 (EUDR) -- Mass balance chain of custody per Article 4

## Purpose

The Mass Balance Calculator agent implements the mass balance chain of custody
model for EUDR-regulated commodities. It manages credit ledgers that track
compliant vs. non-compliant volumes, handles credit period management,
validates conversion factors for processed commodities, detects overdraft
conditions, tracks losses and waste, performs reconciliation, and consolidates
balances across multiple facilities.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/ledgers` | Create a ledger | JWT |
| GET | `/ledgers` | List ledgers | JWT |
| GET | `/ledgers/{ledger_id}` | Get ledger details | JWT |
| POST | `/ledgers/{ledger_id}/credit` | Add credit entry | JWT |
| POST | `/ledgers/{ledger_id}/debit` | Add debit entry | JWT |
| GET | `/ledgers/{ledger_id}/balance` | Get current balance | JWT |
| GET | `/ledgers/{ledger_id}/transactions` | List transactions | JWT |
| POST | `/periods` | Create credit period | JWT |
| GET | `/periods` | List credit periods | JWT |
| GET | `/periods/{period_id}` | Get period details | JWT |
| POST | `/periods/{period_id}/close` | Close credit period | JWT |
| GET | `/periods/{period_id}/summary` | Get period summary | JWT |
| POST | `/factors/validate` | Validate conversion factor | JWT |
| GET | `/factors` | List conversion factors | JWT |
| GET | `/factors/{commodity}` | Get factors for commodity | JWT |
| PUT | `/factors/{factor_id}` | Update conversion factor | JWT |
| POST | `/overdraft/detect` | Run overdraft detection | JWT |
| GET | `/overdraft/alerts` | List overdraft alerts | JWT |
| GET | `/overdraft/{alert_id}` | Get overdraft alert details | JWT |
| POST | `/overdraft/{alert_id}/resolve` | Resolve overdraft | JWT |
| GET | `/overdraft/thresholds` | Get overdraft thresholds | JWT |
| POST | `/losses/record` | Record loss or waste | JWT |
| GET | `/losses` | List loss records | JWT |
| GET | `/losses/{loss_id}` | Get loss record details | JWT |
| POST | `/losses/summary` | Get loss summary | JWT |
| POST | `/reconciliation/run` | Run reconciliation | JWT |
| GET | `/reconciliation` | List reconciliation results | JWT |
| GET | `/reconciliation/{recon_id}` | Get reconciliation details | JWT |
| POST | `/reconciliation/schedule` | Schedule reconciliation | JWT |
| POST | `/consolidation/run` | Run multi-facility consolidation | JWT |
| GET | `/consolidation` | List consolidation results | JWT |
| GET | `/consolidation/{consol_id}` | Get consolidation details | JWT |
| POST | `/consolidation/configure` | Configure consolidation rules | JWT |
| GET | `/consolidation/summary` | Get consolidation summary | JWT |
| POST | `/batch` | Submit batch job | JWT |
| DELETE | `/batch/{job_id}` | Cancel batch job | JWT |
| GET | `/health` | Health check | None |

**Total: 37 endpoints**

---

## Endpoints

### POST /v1/eudr-mbc/ledgers/{ledger_id}/credit

Add a credit entry to a mass balance ledger, representing compliant commodity
volume entering the facility.

**Request:**

```json
{
  "quantity_kg": 5000.0,
  "commodity": "cocoa_beans",
  "source_batch_id": "batch-001",
  "compliance_status": "eudr_compliant",
  "reference_dds": "DDS-2026-001",
  "timestamp": "2026-01-15T08:00:00Z"
}
```

**Response (201 Created):**

```json
{
  "transaction_id": "txn_001",
  "ledger_id": "ldg_001",
  "entry_type": "credit",
  "quantity_kg": 5000.0,
  "balance_after_kg": 15000.0,
  "compliance_status": "eudr_compliant",
  "provenance_hash": "sha256:a1b2c3d4...",
  "created_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/eudr-mbc/overdraft/detect

Run overdraft detection across ledgers to identify cases where debit entries
exceed available compliant credit balances.

**Request:**

```json
{
  "ledger_ids": ["ldg_001", "ldg_002"],
  "period_id": "per_2026_q1",
  "tolerance_pct": 1.0,
  "include_projected": true
}
```

**Response (200 OK):**

```json
{
  "detection_id": "od_001",
  "alerts": [
    {
      "alert_id": "odalert_001",
      "ledger_id": "ldg_002",
      "severity": "warning",
      "overdraft_kg": 250.0,
      "balance_kg": -250.0,
      "period": "2026-Q1",
      "recommendation": "Source additional compliant volume"
    }
  ],
  "total_alerts": 1,
  "checked_at": "2026-04-04T10:10:00Z"
}
```

---

### POST /v1/eudr-mbc/reconciliation/run

Run a mass balance reconciliation that compares total credits against total
debits and losses for a given period, flagging discrepancies.

**Request:**

```json
{
  "ledger_id": "ldg_001",
  "period_id": "per_2026_q1",
  "include_losses": true,
  "tolerance_pct": 2.0
}
```

**Response (200 OK):**

```json
{
  "recon_id": "recon_001",
  "ledger_id": "ldg_001",
  "period": "2026-Q1",
  "total_credit_kg": 50000.0,
  "total_debit_kg": 48500.0,
  "total_loss_kg": 800.0,
  "unaccounted_kg": 700.0,
  "variance_pct": 1.4,
  "within_tolerance": true,
  "status": "reconciled",
  "reconciled_at": "2026-04-04T10:15:00Z"
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_entry` | Credit/debit entry fails validation |
| 404 | `ledger_not_found` | Ledger ID not found |
| 409 | `period_closed` | Cannot modify a closed credit period |
| 422 | `invalid_factor` | Conversion factor out of valid range |
