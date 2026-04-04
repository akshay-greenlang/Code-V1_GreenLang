# AGENT-EUDR-024: Third-Party Audit Manager API

**Agent ID:** `GL-EUDR-TAM-024`
**Prefix:** `/v1/eudr-tam`
**Version:** 1.0.0
**PRD:** AGENT-EUDR-024
**Regulation:** EU 2023/1115 (EUDR) -- Third-party verification per Articles 10 and 14

## Purpose

The Third-Party Audit Manager agent orchestrates the full lifecycle of
third-party audits for EUDR compliance. It manages audit planning and
scheduling, auditor registry and qualification matching, EUDR-specific
checklists, evidence collection, non-conformance (NC) tracking, corrective
action request (CAR) management, certificate generation, report creation,
and competent authority communication.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/audits` | Create audit | JWT |
| GET | `/audits` | List audits | JWT |
| GET | `/audits/{audit_id}` | Get audit details | JWT |
| POST | `/audits/{audit_id}/schedule` | Schedule audit | JWT |
| POST | `/audits/{audit_id}/start` | Start audit | JWT |
| POST | `/audits/{audit_id}/complete` | Complete audit | JWT |
| POST | `/auditors/register` | Register auditor | JWT |
| GET | `/auditors` | List auditors | JWT |
| GET | `/auditors/{auditor_id}` | Get auditor details | JWT |
| POST | `/auditors/match` | Match auditor to audit | JWT |
| POST | `/auditors/{auditor_id}/qualification` | Record qualification | JWT |
| GET | `/checklists` | List EUDR checklists | JWT |
| POST | `/checklists/custom` | Create custom checklist | JWT |
| POST | `/checklists/{checklist_id}/progress` | Update progress | JWT |
| POST | `/evidence/upload` | Upload audit evidence | JWT |
| GET | `/evidence` | List evidence items | JWT |
| DELETE | `/evidence/{evidence_id}` | Remove evidence | JWT |
| POST | `/non-conformances` | Create NC finding | JWT |
| GET | `/non-conformances` | List NC findings | JWT |
| GET | `/non-conformances/{nc_id}` | Get NC details | JWT |
| POST | `/non-conformances/{nc_id}/classify` | Classify NC severity | JWT |
| POST | `/non-conformances/{nc_id}/root-cause` | Record root cause | JWT |
| POST | `/cars/issue` | Issue CAR | JWT |
| GET | `/cars` | List CARs | JWT |
| GET | `/cars/{car_id}` | Get CAR details | JWT |
| POST | `/cars/{car_id}/submit-plan` | Submit corrective plan | JWT |
| POST | `/cars/{car_id}/verify` | Verify corrective action | JWT |
| POST | `/cars/{car_id}/close` | Close CAR | JWT |
| POST | `/certificates/create` | Create audit certificate | JWT |
| GET | `/certificates` | List certificates | JWT |
| GET | `/certificates/supplier/{supplier_id}` | Get supplier certs | JWT |
| POST | `/certificates/validate-eudr` | Validate EUDR cert | JWT |
| POST | `/reports/generate` | Generate audit report | JWT |
| GET | `/reports` | List reports | JWT |
| GET | `/reports/{report_id}` | Get report details | JWT |
| GET | `/reports/{report_id}/download` | Download report | JWT |
| POST | `/authority/create` | Create authority comm | JWT |
| GET | `/authority` | List authority comms | JWT |
| POST | `/authority/{comm_id}/respond` | Respond to authority | JWT |
| GET | `/authority/compliance-rate` | Get compliance rate | JWT |
| GET | `/authority/nc-trends` | Get NC trends | JWT |
| GET | `/health` | Health check | None |
| GET | `/stats` | Service statistics | JWT |

**Total: 43 endpoints**

---

## Endpoints

### POST /v1/eudr-tam/audits

Create a new third-party audit for EUDR compliance verification.

**Request:**

```json
{
  "audit_type": "eudr_full",
  "operator_id": "OP-2024-001",
  "supplier_id": "sup-001",
  "commodity": "cocoa",
  "scope": "supply_chain_traceability",
  "standard": "eudr_2023_1115",
  "planned_date": "2026-05-15",
  "duration_days": 3
}
```

**Response (201 Created):**

```json
{
  "audit_id": "aud_001",
  "audit_type": "eudr_full",
  "operator_id": "OP-2024-001",
  "status": "planned",
  "planned_date": "2026-05-15",
  "checklist_id": "chk_eudr_full_v2",
  "created_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/eudr-tam/non-conformances

Record a non-conformance finding discovered during an audit.

**Request:**

```json
{
  "audit_id": "aud_001",
  "finding_type": "major",
  "category": "traceability",
  "clause_reference": "Article 9(1)(d)",
  "description": "Geolocation data missing for 5 of 12 production plots",
  "evidence_ids": ["ev_001", "ev_002"],
  "affected_entity": "sup-001"
}
```

**Response (201 Created):**

```json
{
  "nc_id": "nc_001",
  "audit_id": "aud_001",
  "finding_type": "major",
  "category": "traceability",
  "status": "open",
  "car_required": true,
  "remediation_deadline": "2026-06-15",
  "created_at": "2026-04-04T10:10:00Z"
}
```

---

### POST /v1/eudr-tam/cars/issue

Issue a Corrective Action Request (CAR) for a non-conformance finding.

**Request:**

```json
{
  "nc_id": "nc_001",
  "required_action": "Obtain GPS coordinates for all 12 production plots and submit updated geolocation data",
  "deadline": "2026-06-15",
  "assigned_to": "sup-001",
  "priority": "high"
}
```

**Response (201 Created):**

```json
{
  "car_id": "car_001",
  "nc_id": "nc_001",
  "status": "issued",
  "deadline": "2026-06-15",
  "priority": "high",
  "created_at": "2026-04-04T10:15:00Z"
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_audit` | Audit parameters are invalid |
| 404 | `audit_not_found` | Audit ID not found |
| 404 | `nc_not_found` | Non-conformance ID not found |
| 409 | `audit_already_completed` | Cannot modify completed audit |
| 422 | `invalid_status_transition` | Status change not allowed |
