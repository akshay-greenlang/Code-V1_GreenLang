# SOC 2 Preparation Service API Reference (SEC-009)

## Overview

The SOC 2 Preparation Service provides a comprehensive platform for SOC 2 Type II audit preparation, including project management, control assessments, evidence collection, finding tracking, attestation workflow, control testing, portal access, and dashboards. It covers all five Trust Service Criteria (Security, Availability, Processing Integrity, Confidentiality, Privacy).

**Router Prefixes (under `/api/v1/soc2`):**
- `/project` -- Project and control management
- `/assessment` -- Control assessments and evaluations
- `/evidence` -- Evidence collection and management
- `/findings` -- Finding tracking and remediation
- `/attestations` -- Attestation workflow
- `/tests` -- Control testing execution
- `/portal` -- Auditor portal and document management
- `/dashboard` -- Compliance dashboards and analytics

**Tags:** `soc2-project`, `soc2-assessment`, `soc2-evidence`, `soc2-findings`, `soc2-attestation`, `soc2-testing`, `soc2-portal`, `soc2-dashboard`
**Source:** `greenlang/infrastructure/soc2_preparation/api/`

---

## Endpoint Summary

### Project Management

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/project/` | List SOC 2 projects | Yes |
| POST | `/project/` | Create SOC 2 project | Yes |
| GET | `/project/{project_id}` | Get project details | Yes |
| GET | `/project/{project_id}/controls` | List controls for project | Yes |
| GET | `/project/{project_id}/timeline` | Get project timeline | Yes |
| POST | `/project/{project_id}/milestones` | Add project milestone | Yes |
| PUT | `/project/{project_id}/status` | Update project status | Yes |

### Assessments

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/assessment/` | List control assessments | Yes |
| POST | `/assessment/` | Create control assessment | Yes |
| GET | `/assessment/{assessment_id}` | Get assessment details | Yes |
| GET | `/assessment/{assessment_id}/controls` | List assessed controls | Yes |
| GET | `/assessment/{assessment_id}/gaps` | List assessment gaps | Yes |
| PUT | `/assessment/{assessment_id}/status` | Update assessment status | Yes |

### Evidence

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/evidence/` | List evidence items | Yes |
| GET | `/evidence/{evidence_id}` | Get evidence details | Yes |
| POST | `/evidence/` | Upload evidence | Yes |
| POST | `/evidence/link` | Link evidence to control | Yes |
| GET | `/evidence/coverage` | Evidence coverage report | Yes |
| GET | `/evidence/gaps` | Evidence gaps report | Yes |

### Findings

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/findings/` | List findings with filters | Yes |
| POST | `/findings/` | Create finding | Yes |
| GET | `/findings/{finding_id}` | Get finding details | Yes |
| GET | `/findings/stats` | Finding statistics | Yes |
| PUT | `/findings/{finding_id}` | Update finding | Yes |
| POST | `/findings/{finding_id}/remediation` | Add remediation plan | Yes |
| GET | `/findings/{finding_id}/timeline` | Get finding timeline | Yes |
| PUT | `/findings/{finding_id}/status` | Update finding status | Yes |

### Attestations

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/attestations/` | List attestations | Yes |
| POST | `/attestations/` | Create attestation | Yes |
| GET | `/attestations/{attestation_id}` | Get attestation details | Yes |
| POST | `/attestations/{attestation_id}/sign` | Sign attestation | Yes |
| POST | `/attestations/{attestation_id}/revoke` | Revoke attestation | Yes |
| GET | `/attestations/coverage` | Attestation coverage | Yes |
| POST | `/attestations/bulk-create` | Bulk create attestations | Yes |

### Control Testing

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/tests/` | List test executions | Yes |
| POST | `/tests/` | Create and run test | Yes |
| GET | `/tests/{test_id}` | Get test execution details | Yes |
| GET | `/tests/{test_id}/results` | Get test results | Yes |
| GET | `/tests/schedule` | Get test schedule | Yes |
| GET | `/tests/coverage` | Get test coverage | Yes |

### Portal

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/portal/` | Portal home (project summary) | Yes |
| POST | `/portal/documents` | Upload document | Yes |
| GET | `/portal/documents` | List documents | Yes |
| GET | `/portal/documents/{doc_id}` | Get document details | Yes |
| GET | `/portal/activity` | Activity log | Yes |
| GET | `/portal/comments` | List comments | Yes |
| GET | `/portal/status` | Compliance status overview | Yes |

### Dashboard

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/dashboard/` | Executive summary | Yes |
| GET | `/dashboard/controls` | Control status breakdown | Yes |
| GET | `/dashboard/progress` | Project progress over time | Yes |
| GET | `/dashboard/risks` | Risk heatmap data | Yes |

---

## Key Concepts

### Trust Service Criteria

The SOC 2 framework covers five criteria:

| Criteria | Code | Description |
|----------|------|-------------|
| Security | CC | Common Criteria -- required for all SOC 2 reports |
| Availability | A | System availability commitments |
| Processing Integrity | PI | Data processing accuracy and completeness |
| Confidentiality | C | Confidential information protection |
| Privacy | P | Personal information handling |

### Finding Lifecycle

Findings progress through the following states:

```
identified -> acknowledged -> remediation_planned -> in_progress -> remediated -> verified -> closed
```

### Attestation Workflow

Attestations are cryptographically signed statements that a control is operating effectively:

```
draft -> pending_review -> reviewed -> signed -> (revoked)
```
