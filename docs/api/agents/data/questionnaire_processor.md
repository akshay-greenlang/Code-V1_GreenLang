# Supplier Questionnaire Processor API Reference

**Agent:** AGENT-DATA-008 (GL-DATA-008)
**Prefix:** `/api/v1/questionnaires`
**Source:** `greenlang/agents/data/supplier_questionnaire/api/router.py`
**Status:** Production Ready

## Overview

The Supplier Questionnaire Processor agent provides 20 REST API endpoints for end-to-end supplier questionnaire lifecycle management. Capabilities include questionnaire template creation and versioning (CDP, EcoVadis, GRI, custom), distribution to suppliers via email/portal/API/bulk channels, response collection and multi-level validation (completeness, consistency, evidence, cross-field), framework-aware scoring, follow-up and escalation management, and campaign analytics with compliance gap identification.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/v1/templates` | Create questionnaire template | Yes |
| 2 | GET | `/v1/templates` | List templates | Yes |
| 3 | GET | `/v1/templates/{template_id}` | Get template details | Yes |
| 4 | PUT | `/v1/templates/{template_id}` | Update template | Yes |
| 5 | POST | `/v1/templates/{template_id}/clone` | Clone template | Yes |
| 6 | POST | `/v1/distribute` | Distribute questionnaire to supplier | Yes |
| 7 | GET | `/v1/distributions` | List distributions | Yes |
| 8 | GET | `/v1/distributions/{dist_id}` | Get distribution details | Yes |
| 9 | POST | `/v1/responses` | Submit questionnaire response | Yes |
| 10 | GET | `/v1/responses` | List responses | Yes |
| 11 | GET | `/v1/responses/{response_id}` | Get response details | Yes |
| 12 | POST | `/v1/responses/{response_id}/validate` | Validate response | Yes |
| 13 | POST | `/v1/score` | Score response | Yes |
| 14 | GET | `/v1/scores/{score_id}` | Get score details | Yes |
| 15 | GET | `/v1/scores/supplier/{supplier_id}` | Get all scores for supplier | Yes |
| 16 | POST | `/v1/followup` | Trigger follow-up action | Yes |
| 17 | GET | `/v1/followup/{campaign_id}` | Get follow-up status | Yes |
| 18 | GET | `/v1/analytics/{campaign_id}` | Get campaign analytics | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/v1/statistics` | Service statistics | Yes |

---

## Key Endpoints

### 1. Create Questionnaire Template

Create a new questionnaire template with sections and questions.

```http
POST /api/v1/questionnaires/v1/templates
```

**Request Body:**

```json
{
  "name": "CDP Climate Change 2026",
  "framework": "cdp",
  "version": "2026.1",
  "description": "CDP Climate Change questionnaire for supplier engagement",
  "language": "en",
  "tags": ["climate", "scope3", "supplier"],
  "sections": [
    {
      "section_id": "C1",
      "title": "Governance",
      "questions": [
        {
          "question_id": "C1.1",
          "text": "Is there board-level oversight of climate-related issues?",
          "type": "yes_no",
          "required": true
        }
      ]
    }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Template display name |
| `framework` | string | No | Framework: `cdp`, `ecovadis`, `gri`, `custom` (default: `custom`) |
| `version` | string | No | Version string (default: `1.0`) |
| `description` | string | No | Template description |
| `sections` | array | No | Ordered list of section definitions with questions |
| `language` | string | No | ISO 639-1 language code (default: `en`) |
| `tags` | array | No | Classification tags |

**Response (200):** Complete template record with generated template_id and initial `draft` status.

**Status Codes:** `200` Success | `400` Validation error | `503` Service not configured

---

### 6. Distribute Questionnaire

Distribute a questionnaire to a supplier.

```http
POST /api/v1/questionnaires/v1/distribute
```

**Request Body:**

```json
{
  "template_id": "tmpl_abc123",
  "supplier_id": "SUP-001",
  "supplier_name": "Green Materials Co",
  "supplier_email": "sustainability@greenmaterials.example.com",
  "channel": "email",
  "deadline": "2026-06-30T23:59:59Z"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `template_id` | string | Yes | Template ID to distribute |
| `supplier_id` | string | Yes | Target supplier identifier |
| `supplier_name` | string | Yes | Supplier display name |
| `supplier_email` | string | Yes | Contact email address |
| `campaign_id` | string | No | Campaign ID (auto-generated if omitted) |
| `channel` | string | No | Channel: `email`, `portal`, `api`, `bulk` (default: `email`) |
| `deadline` | string | No | Response deadline (ISO 8601) |

---

### 9. Submit Response

Submit a questionnaire response with answers and evidence.

```http
POST /api/v1/questionnaires/v1/responses
```

**Request Body:**

```json
{
  "distribution_id": "dist_abc123",
  "supplier_id": "SUP-001",
  "supplier_name": "Green Materials Co",
  "answers": {
    "C1.1": {"value": "yes", "comment": "Board sustainability committee meets quarterly"},
    "C1.2": {"value": "Chief Sustainability Officer"}
  },
  "evidence_files": ["board_minutes_q1_2026.pdf"],
  "channel": "portal"
}
```

---

### 12. Validate Response

Run validation on a submitted response at a specified level.

```http
POST /api/v1/questionnaires/v1/responses/{response_id}/validate
```

**Request Body:**

```json
{
  "level": "consistency"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `level` | string | No | Validation level: `completeness`, `consistency`, `evidence`, `cross_field` (default: `completeness`) |

---

### 13. Score Response

Score a questionnaire response using framework-specific scoring rules.

```http
POST /api/v1/questionnaires/v1/score
```

**Request Body:**

```json
{
  "response_id": "resp_abc123",
  "framework": "cdp"
}
```

---

### 16. Trigger Follow-Up

Trigger a reminder or escalation for a pending questionnaire distribution.

```http
POST /api/v1/questionnaires/v1/followup
```

**Request Body:**

```json
{
  "distribution_id": "dist_abc123",
  "action_type": "reminder",
  "message": "Your questionnaire response is due in 7 days."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `distribution_id` | string | Yes | Distribution to follow up on |
| `action_type` | string | No | `reminder` or `escalation` (default: `reminder`) |
| `message` | string | No | Custom message content |

---

### 18. Get Campaign Analytics

Retrieve analytics and compliance gap identification for a campaign.

```http
GET /api/v1/questionnaires/v1/analytics/{campaign_id}
```

**Response (200):** Campaign analytics including response rates, average scores, compliance gaps, and supplier performance.

---

## Error Responses

All error responses follow a standard format:

```json
{
  "detail": "Descriptive error message"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- validation error or malformed input |
| 404 | Not Found -- template, distribution, response, or score not found |
| 503 | Service Unavailable -- questionnaire service not configured |
