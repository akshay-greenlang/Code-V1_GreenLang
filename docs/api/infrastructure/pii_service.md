# PII Detection and Redaction Service API Reference (SEC-011)

## Overview

The PII Detection and Redaction Service provides PII detection, redaction, tokenization/detokenization, enforcement policies, allowlist management, quarantine operations, and metrics. Supports multiple PII types including email addresses, phone numbers, SSNs, credit card numbers, and custom patterns.

**Router Prefix:** `/api/v1/pii`
**Tags:** `pii`
**Source:** `greenlang/infrastructure/pii_service/api/pii_routes.py`

---

## Endpoint Summary

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/api/v1/pii/detect` | Detect PII in content | Yes |
| POST | `/api/v1/pii/redact` | Redact PII from content | Yes |
| POST | `/api/v1/pii/tokenize` | Create reversible token for PII | Yes |
| POST | `/api/v1/pii/detokenize` | Retrieve original from token | Yes |
| GET | `/api/v1/pii/policies` | List enforcement policies | Yes |
| PUT | `/api/v1/pii/policies/{pii_type}` | Update enforcement policy | Yes |
| GET | `/api/v1/pii/allowlist` | List allowlist entries | Yes |
| POST | `/api/v1/pii/allowlist` | Add allowlist entry | Yes |
| DELETE | `/api/v1/pii/allowlist/{id}` | Remove allowlist entry | Yes |
| GET | `/api/v1/pii/quarantine` | List quarantined items | Yes |
| POST | `/api/v1/pii/quarantine/{id}/release` | Release quarantined item | Yes |
| POST | `/api/v1/pii/quarantine/{id}/delete` | Delete quarantined item | Yes |
| GET | `/api/v1/pii/metrics` | Get PII detection metrics | Yes |

---

## Endpoints

### POST /api/v1/pii/detect

Detect PII in the provided content. Returns a list of detected PII entities with type, location, and confidence score.

**Request Body:**

```json
{
  "content": "Contact John Doe at john.doe@acme.com or call 555-0123.",
  "pii_types": ["email", "phone", "name"],
  "confidence_threshold": 0.8
}
```

**Response (200 OK):**

```json
{
  "entities": [
    {
      "type": "email",
      "value": "john.doe@acme.com",
      "start": 24,
      "end": 41,
      "confidence": 0.99
    },
    {
      "type": "phone",
      "value": "555-0123",
      "start": 50,
      "end": 58,
      "confidence": 0.92
    },
    {
      "type": "name",
      "value": "John Doe",
      "start": 8,
      "end": 16,
      "confidence": 0.85
    }
  ],
  "total_detected": 3,
  "processing_time_ms": 12.5
}
```

---

### POST /api/v1/pii/redact

Redact PII from content. Returns the redacted content with PII replaced by placeholders.

**Request Body:**

```json
{
  "content": "Contact John Doe at john.doe@acme.com or call 555-0123.",
  "redaction_strategy": "mask",
  "pii_types": ["email", "phone"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | Yes | Text content to redact |
| `redaction_strategy` | string | No | Strategy: `mask`, `hash`, `remove`, `placeholder` (default: `mask`) |
| `pii_types` | array | No | Specific PII types to redact (all if omitted) |

**Response (200 OK):**

```json
{
  "redacted_content": "Contact John Doe at [EMAIL_REDACTED] or call [PHONE_REDACTED].",
  "entities_redacted": 2,
  "processing_time_ms": 15.2
}
```

---

### POST /api/v1/pii/tokenize

Create a reversible token for a PII value. The original value can be recovered using the detokenize endpoint. Tokens are encrypted and stored securely.

**Request Body:**

```json
{
  "value": "john.doe@acme.com",
  "pii_type": "email",
  "tenant_id": "t-acme-corp"
}
```

**Response (200 OK):**

```json
{
  "token": "tok_abc123def456",
  "pii_type": "email",
  "created_at": "2026-04-04T12:00:00Z"
}
```

---

### POST /api/v1/pii/detokenize

Retrieve the original value from a PII token. Requires appropriate permissions and is audit-logged.

**Request Body:**

```json
{
  "token": "tok_abc123def456"
}
```

**Response (200 OK):**

```json
{
  "value": "john.doe@acme.com",
  "pii_type": "email"
}
```

---

### GET /api/v1/pii/policies

List all PII enforcement policies. Policies define how each PII type is handled (detect, redact, quarantine, or block).

**Response (200 OK):**

```json
{
  "policies": [
    {
      "pii_type": "email",
      "action": "redact",
      "enabled": true,
      "confidence_threshold": 0.8
    },
    {
      "pii_type": "ssn",
      "action": "quarantine",
      "enabled": true,
      "confidence_threshold": 0.9
    }
  ]
}
```

---

### GET /api/v1/pii/metrics

Get PII detection metrics including total detections, detections by type, and processing performance.

**Response (200 OK):**

```json
{
  "total_scans": 15420,
  "total_detections": 2340,
  "detections_by_type": {
    "email": 890,
    "phone": 450,
    "name": 620,
    "ssn": 12,
    "credit_card": 3
  },
  "quarantined_items": 15,
  "avg_processing_time_ms": 8.5,
  "period": "last_30_days"
}
```
