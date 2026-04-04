# Security Scanning Service API Reference (SEC-007)

## Overview

The Security Scanning Service provides automated security scan execution, vulnerability management, and a security dashboard. Supports SAST, DAST, dependency scanning, container scanning, and infrastructure-as-code scanning. Includes SLA tracking and trend analysis.

**Router Prefix:** `/api/v1/security`
**Tags:** `Security Scans`, `Vulnerabilities`, `Security Dashboard`
**Source:** `greenlang/infrastructure/security_scanning/api/`

---

## Endpoint Summary

### Scans

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/api/v1/security/scans` | Trigger a new security scan (async) | Yes |
| GET | `/api/v1/security/scans` | List scan runs with filters | Yes |
| GET | `/api/v1/security/scans/{id}` | Get scan run details | Yes |
| GET | `/api/v1/security/scans/{id}/findings` | Get findings for a scan run | Yes |

### Vulnerabilities

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/security/vulnerabilities` | List vulnerabilities with filters | Yes |
| GET | `/api/v1/security/vulnerabilities/{id}` | Get vulnerability details with findings | Yes |
| POST | `/api/v1/security/vulnerabilities/{id}/accept` | Risk acceptance workflow | Yes |
| POST | `/api/v1/security/vulnerabilities/{id}/remediate` | Mark vulnerability as fixed | Yes |
| GET | `/api/v1/security/vulnerabilities/stats` | Vulnerability statistics | Yes |

### Dashboard

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/security/dashboard` | Summary statistics | Yes |
| GET | `/api/v1/security/dashboard/trends` | 90-day trend data | Yes |
| GET | `/api/v1/security/dashboard/coverage` | Scanner coverage metrics | Yes |
| GET | `/api/v1/security/dashboard/sla` | SLA compliance metrics | Yes |

---

## Scan Endpoints

### POST /api/v1/security/scans

Trigger a new security scan. Runs asynchronously and returns a scan run ID for status polling.

**Request Body:**

```json
{
  "scan_type": "dependency",
  "target": "greenlang/agents/mrv",
  "severity_threshold": "medium",
  "notify_on_complete": true
}
```

**Response (202 Accepted):**

```json
{
  "scan_id": "scan-abc123",
  "scan_type": "dependency",
  "target": "greenlang/agents/mrv",
  "status": "running",
  "started_at": "2026-04-04T12:00:00Z"
}
```

---

### GET /api/v1/security/scans/{id}/findings

Get detailed findings for a completed scan run. Findings include CVE IDs, severity, affected components, and remediation guidance.

**Response (200 OK):**

```json
{
  "scan_id": "scan-abc123",
  "findings": [
    {
      "id": "finding-001",
      "cve_id": "CVE-2026-12345",
      "severity": "high",
      "title": "Remote Code Execution in dependency X",
      "affected_component": "package-x@1.2.3",
      "remediation": "Upgrade to package-x@1.2.4",
      "status": "open"
    }
  ],
  "total_findings": 1,
  "critical": 0,
  "high": 1,
  "medium": 0,
  "low": 0
}
```

---

## Vulnerability Endpoints

### POST /api/v1/security/vulnerabilities/{id}/accept

Submit a risk acceptance for a vulnerability. Requires justification and approval workflow.

**Request Body:**

```json
{
  "justification": "Compensating controls in place. WAF rule blocks exploitation path.",
  "accepted_by": "security-lead",
  "expires_at": "2026-07-01T00:00:00Z"
}
```

### POST /api/v1/security/vulnerabilities/{id}/remediate

Mark a vulnerability as remediated. Triggers a verification scan.

**Request Body:**

```json
{
  "remediation_notes": "Upgraded package-x to 1.2.4",
  "remediated_by": "dev-team-lead"
}
```

---

## Dashboard Endpoints

### GET /api/v1/security/dashboard

Summary statistics across all security scanning activities.

**Response (200 OK):**

```json
{
  "total_scans": 1250,
  "scans_last_7_days": 42,
  "open_vulnerabilities": 15,
  "critical_open": 0,
  "high_open": 3,
  "medium_open": 8,
  "low_open": 4,
  "mean_time_to_remediate_hours": 48.5,
  "sla_compliance_pct": 96.2
}
```

### GET /api/v1/security/dashboard/trends

90-day trend data for vulnerability counts by severity.

### GET /api/v1/security/dashboard/sla

SLA compliance metrics showing remediation times against defined SLA targets.
