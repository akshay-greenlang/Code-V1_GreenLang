# Validation Rule Engine API Reference

**Agent:** AGENT-DATA-019 (GL-DATA-X-022)
**Prefix:** `/api/v1/validation-rules`
**Source:** `greenlang/agents/data/validation_rule_engine/api/router.py`
**Status:** Production Ready

## Overview

The Validation Rule Engine manages validation rules and rule sets, evaluates them against datasets, detects rule conflicts, applies compliance rule packs (CSRD, GHG Protocol, CBAM), and generates validation reports. Supports individual and batch evaluation.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/rules` | Register a new validation rule | Yes |
| 2 | GET | `/rules` | List rules with filters | Yes |
| 3 | GET | `/rules/{rule_id}` | Get rule details | Yes |
| 4 | PUT | `/rules/{rule_id}` | Update a rule | Yes |
| 5 | DELETE | `/rules/{rule_id}` | Delete a rule (soft delete) | Yes |
| 6 | POST | `/rule-sets` | Create a rule set | Yes |
| 7 | GET | `/rule-sets` | List rule sets | Yes |
| 8 | GET | `/rule-sets/{set_id}` | Get rule set details | Yes |
| 9 | PUT | `/rule-sets/{set_id}` | Update a rule set | Yes |
| 10 | DELETE | `/rule-sets/{set_id}` | Delete a rule set | Yes |
| 11 | POST | `/evaluate` | Evaluate rules against data | Yes |
| 12 | POST | `/evaluate/batch` | Batch evaluate | Yes |
| 13 | GET | `/evaluations/{eval_id}` | Get evaluation result | Yes |
| 14 | POST | `/conflicts/detect` | Detect rule conflicts | Yes |
| 15 | GET | `/conflicts` | List detected conflicts | Yes |
| 16 | POST | `/packs/{pack_name}/apply` | Apply a rule pack | Yes |
| 17 | GET | `/packs` | List available rule packs | Yes |
| 18 | POST | `/reports` | Generate validation report | Yes |
| 19 | POST | `/pipeline` | Run full validation pipeline | Yes |
| 20 | GET | `/health` | Health check | No |

---

## Key Endpoints

### 11. Evaluate Rules Against Data

```http
POST /api/v1/validation-rules/evaluate
```

**Request Body:**

```json
{
  "data": [
    {"facility": "HQ", "emissions_kg": 1500.0, "scope": "1"},
    {"facility": "Plant A", "emissions_kg": -50.0, "scope": "1"}
  ],
  "rule_set_id": "rs_ghg_basic",
  "rule_ids": null
}
```

**Response:**

```json
{
  "evaluation_id": "eval_abc123",
  "rule_set_id": "rs_ghg_basic",
  "total_records": 2,
  "passed": 1,
  "failed": 1,
  "results": [
    {
      "record_index": 1,
      "rule_id": "rule_non_negative_emissions",
      "passed": false,
      "severity": "error",
      "message": "emissions_kg must be >= 0, got -50.0"
    }
  ]
}
```

### 16. Apply a Rule Pack

Apply a pre-built compliance rule pack that installs and activates domain-specific validation rules.

```http
POST /api/v1/validation-rules/packs/csrd_esrs_e1/apply
```

**Request Body:**

```json
{
  "tenant_id": "tenant_abc",
  "override_existing": false
}
```

Available packs include: `ghg_protocol`, `csrd_esrs_e1`, `cbam`, `iso_14064`, `cdp`, `tcfd`, `sbt`.
