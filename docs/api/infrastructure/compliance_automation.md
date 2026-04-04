# Compliance Automation Service API Reference (SEC-010)

## Overview

The Compliance Automation Service provides multi-framework compliance management including ISO 27001, GDPR, PCI-DSS, CCPA, and LGPD. Features compliance dashboards, DSAR (Data Subject Access Request) processing, consent management, and framework-specific assessment endpoints.

**Tags:** `Compliance Automation`
**Source:** `greenlang/infrastructure/compliance_automation/api/compliance_routes.py`

---

## Endpoint Summary

### Compliance Status

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/compliance/status` | Overall compliance dashboard | Yes |
| GET | `/compliance/iso27001` | ISO 27001 compliance status | Yes |
| GET | `/compliance/iso27001/soa` | Statement of Applicability | Yes |
| GET | `/compliance/gdpr` | GDPR compliance status | Yes |
| GET | `/compliance/pci-dss` | PCI-DSS compliance status | Yes |

### DSAR Processing

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/dsar` | Submit DSAR request | No (public) |
| GET | `/dsar` | List DSAR requests | Yes |
| GET | `/dsar/{request_id}` | Get DSAR details | Yes |
| POST | `/dsar/{request_id}/verify` | Verify identity | Yes |
| POST | `/dsar/{request_id}/execute` | Execute DSAR | Yes |
| GET | `/dsar/{request_id}/download` | Download data export | Yes |

### Consent Management

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/consent` | Record consent | Yes |
| GET | `/consent/{user_id}` | Get user consent summary | Yes |
| DELETE | `/consent/{user_id}/{purpose}` | Revoke consent | Yes |
| GET | `/consent/{user_id}/audit` | Get consent audit trail | Yes |

---

## Compliance Status Endpoints

### GET /compliance/status

Get an overview of compliance status across all supported frameworks.

**Response (200 OK):**

```json
{
  "overall_score": 96.5,
  "last_assessed": "2026-04-01T10:00:00Z",
  "frameworks": {
    "iso27001": {
      "score": 96.5,
      "status": "compliant"
    },
    "gdpr": {
      "score": 95.0,
      "status": "compliant"
    },
    "pci_dss": {
      "score": 100.0,
      "status": "compliant"
    }
  },
  "pending_dsars": 3,
  "dsar_sla_compliance": 100.0,
  "next_assessment": "2026-07-01T10:00:00Z"
}
```

---

### GET /compliance/iso27001

Get detailed ISO/IEC 27001:2022 compliance status including control counts and gap analysis.

**Response (200 OK):**

```json
{
  "framework": "iso27001",
  "framework_name": "ISO/IEC 27001:2022",
  "version": "2022",
  "score": 96.5,
  "status": "compliant",
  "controls_total": 93,
  "controls_compliant": 90,
  "controls_non_compliant": 2,
  "controls_not_applicable": 1,
  "gaps_count": 2,
  "last_assessed": "2026-04-01T10:00:00Z",
  "next_assessment": "2026-07-01T10:00:00Z"
}
```

---

### GET /compliance/iso27001/soa

Get the Statement of Applicability (SoA) for ISO 27001. Lists all controls with their applicability status, implementation evidence, and justification for exclusions.

**Response (200 OK):**

```json
{
  "framework": "iso27001",
  "version": "2022",
  "generated_at": "2026-04-04T12:00:00Z",
  "summary": {
    "total_controls": 93,
    "applicable": 92,
    "not_applicable": 1,
    "implemented": 90,
    "partially_implemented": 2
  },
  "controls": [
    {
      "id": "A.5.1",
      "title": "Policies for information security",
      "applicable": true,
      "status": "implemented",
      "evidence": "SEC-008 Security Policies Documentation",
      "justification": null
    }
  ]
}
```

---

### GET /compliance/gdpr

Get GDPR compliance status based on DSAR processing performance and consent management.

**Response (200 OK):**

```json
{
  "framework": "gdpr",
  "framework_name": "EU General Data Protection Regulation",
  "version": "2018",
  "score": 100.0,
  "status": "compliant",
  "controls_total": 8,
  "controls_compliant": 8,
  "controls_non_compliant": 0,
  "controls_not_applicable": 0,
  "gaps_count": 0,
  "last_assessed": "2026-04-04T12:00:00Z",
  "next_assessment": null
}
```

---

### GET /compliance/pci-dss

Get PCI-DSS v4.0 compliance status including CDE scope analysis and encryption assessment.

**Response (200 OK):**

```json
{
  "framework": "pci_dss",
  "framework_name": "Payment Card Industry Data Security Standard",
  "version": "4.0",
  "score": 100.0,
  "status": "compliant",
  "controls_total": 12,
  "controls_compliant": 12,
  "controls_non_compliant": 0,
  "controls_not_applicable": 0,
  "gaps_count": 0,
  "last_assessed": "2026-04-04T12:00:00Z",
  "next_assessment": null
}
```

---

## DSAR Endpoints

### POST /dsar

Submit a new Data Subject Access Request. This is a public endpoint.

**Request Body:**

```json
{
  "request_type": "access",
  "subject_email": "user@example.com",
  "subject_name": "John Doe",
  "regulation": "gdpr",
  "description": "I would like to receive a copy of all personal data you hold about me."
}
```

**Request Types:** `access` (Article 15), `erasure` (Article 17), `portability` (Article 20), `rectification` (Article 16), `restriction` (Article 18), `objection` (Article 21)

**Response (201 Created):**

```json
{
  "id": "dsar-uuid-001",
  "request_number": "DSAR-2026-0015",
  "request_type": "access",
  "subject_email": "user@example.com",
  "status": "submitted",
  "submitted_at": "2026-04-04T12:00:00Z",
  "due_date": "2026-05-04T12:00:00Z",
  "days_remaining": 30,
  "is_overdue": false,
  "completed_at": null,
  "export_url": null
}
```

---

### POST /dsar/{request_id}/verify

Verify the identity of the data subject before processing the request.

**Request Body:**

```json
{
  "method": "email",
  "verification_data": null
}
```

**Response (200 OK):**

```json
{
  "request_id": "dsar-uuid-001",
  "verified": true,
  "method": "email",
  "confidence": 0.95,
  "verified_at": "2026-04-04T14:00:00Z"
}
```

---

### POST /dsar/{request_id}/execute

Execute the DSAR request (data access, erasure, or portability).

**Request Body:**

```json
{
  "export_format": "json"
}
```

**Response (200 OK):**

```json
{
  "request_id": "dsar-uuid-001",
  "success": true,
  "request_type": "access",
  "records_affected": 142,
  "export_url": "https://storage.greenlang.io/dsar-exports/dsar-uuid-001.json",
  "deletion_certificate_id": null,
  "completed_at": "2026-04-04T15:30:00Z",
  "errors": []
}
```

For erasure requests, a `deletion_certificate_id` is provided as proof of data deletion.

---

## Consent Management Endpoints

### POST /consent

Record a new consent grant for a user.

**Request Body:**

```json
{
  "user_id": "user-001",
  "purpose": "analytics",
  "source": "api"
}
```

**Consent Purposes:** `essential`, `analytics`, `marketing`, `third_party`, `profiling`, `research`

**Response (201 Created):**

```json
{
  "consent_id": "consent-uuid-001",
  "user_id": "user-001",
  "purpose": "analytics",
  "granted_at": "2026-04-04T12:00:00Z",
  "consent_version": "2.0"
}
```

---

### GET /consent/{user_id}

Get a summary of all consent grants for a user.

**Response (200 OK):**

```json
{
  "user_id": "user-001",
  "total_purposes": 6,
  "active_consents": 4,
  "revoked_consents": 2,
  "consents_by_purpose": {
    "essential": true,
    "analytics": true,
    "marketing": false,
    "third_party": false,
    "profiling": true,
    "research": true
  },
  "last_updated": "2026-04-04T12:00:00Z"
}
```

---

### DELETE /consent/{user_id}/{purpose}

Revoke consent for a specific purpose.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `reason` | string | No | Reason for revocation |

**Response (200 OK):**

```json
{
  "user_id": "user-001",
  "purpose": "marketing",
  "revoked": true,
  "revoked_at": "2026-04-04T12:00:00Z"
}
```

---

### GET /consent/{user_id}/audit

Get the audit trail of consent changes for a user.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `purpose` | string | - | Filter by consent purpose |
| `limit` | integer | 50 | Maximum results (max 200) |

**Response (200 OK):**

```json
{
  "user_id": "user-001",
  "entries": [
    {
      "id": "audit-001",
      "action": "granted",
      "purpose": "analytics",
      "timestamp": "2026-03-01T10:00:00Z",
      "performed_by": "user-001",
      "details": {"source": "web_form", "ip_address": "192.168.1.1"}
    },
    {
      "id": "audit-002",
      "action": "revoked",
      "purpose": "marketing",
      "timestamp": "2026-04-04T12:00:00Z",
      "performed_by": "user-001",
      "details": {"reason": "No longer wish to receive marketing emails"}
    }
  ],
  "total_entries": 2
}
```

---

## Supported Frameworks

| Framework | Version | Description |
|-----------|---------|-------------|
| ISO 27001 | 2022 | Information security management system |
| GDPR | 2018 | EU General Data Protection Regulation |
| PCI-DSS | 4.0 | Payment card data security |
| CCPA | 2020 | California Consumer Privacy Act |
| LGPD | 2020 | Brazil General Data Protection Law |

## DSAR SLA Timelines

| Regulation | SLA | Description |
|------------|-----|-------------|
| GDPR | 30 days | Articles 15-22 subject requests |
| CCPA | 45 days | Consumer data requests |
| LGPD | 15 days | Brazilian data subject requests |
