# Security Training Service API Reference (SEC-010)

## Overview

The Security Training Service provides a platform for security awareness training, assessments, phishing simulation campaigns, and security scoring. Supports course management, curriculum mapping, certificate issuance, team compliance tracking, and gamified leaderboards.

**Router Prefix:** `/api/v1/secops`
**Tags:** `Security Training`
**Source:** `greenlang/infrastructure/security_training/api/training_routes.py`

---

## Endpoint Summary

### Training Management

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/secops/training/courses` | List training courses | Yes |
| GET | `/api/v1/secops/training/courses/{course_id}` | Get course content | Yes |
| GET | `/api/v1/secops/training/my-progress` | Get user training progress | Yes |
| GET | `/api/v1/secops/training/my-curriculum` | Get required curriculum | Yes |
| POST | `/api/v1/secops/training/courses/{course_id}/start` | Start a course | Yes |
| POST | `/api/v1/secops/training/courses/{course_id}/complete` | Mark course complete | Yes |
| POST | `/api/v1/secops/training/courses/{course_id}/assessment` | Submit assessment | Yes |
| GET | `/api/v1/secops/training/certificates` | List user certificates | Yes |
| GET | `/api/v1/secops/training/certificates/{code}/verify` | Verify certificate | Yes |
| GET | `/api/v1/secops/training/team-compliance` | Get team compliance stats | Yes |

### Phishing Campaigns

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/api/v1/secops/phishing/campaigns` | Create campaign | Yes |
| GET | `/api/v1/secops/phishing/campaigns` | List campaigns | Yes |
| GET | `/api/v1/secops/phishing/campaigns/{id}` | Get campaign details | Yes |
| PUT | `/api/v1/secops/phishing/campaigns/{id}` | Update campaign | Yes |
| POST | `/api/v1/secops/phishing/campaigns/{id}/send` | Send campaign emails | Yes |
| GET | `/api/v1/secops/phishing/campaigns/{id}/metrics` | Get campaign metrics | Yes |
| POST | `/api/v1/secops/phishing/track/{campaign_id}/{user_id}/open` | Track email open | Yes |
| POST | `/api/v1/secops/phishing/track/{campaign_id}/{user_id}/click` | Track link click | Yes |

### Security Score

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/secops/security-score` | Get user security score | Yes |
| GET | `/api/v1/secops/security-score/leaderboard` | Get leaderboard | Yes |

---

## Training Endpoints

### GET /api/v1/secops/training/courses

List available training courses with optional filters.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `role` | string | - | Filter by required role |
| `tag` | string | - | Filter by tag |
| `mandatory_only` | boolean | false | Only mandatory courses |
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page (max 100) |

**Response (200 OK):**

```json
{
  "items": [
    {
      "id": "course-sec-101",
      "title": "Security Awareness Fundamentals",
      "description": "Introduction to information security principles and best practices.",
      "duration_minutes": 45,
      "content_type": "interactive",
      "role_required": null,
      "passing_score": 80,
      "is_mandatory": true,
      "tags": ["awareness", "fundamentals"],
      "prerequisites": []
    }
  ],
  "total": 15,
  "page": 1,
  "page_size": 20
}
```

---

### POST /api/v1/secops/training/courses/{course_id}/assessment

Submit quiz answers for grading. Returns score, pass/fail, and optionally a certificate.

**Request Body:**

```json
{
  "answers": {
    "q1": 2,
    "q2": 0,
    "q3": 3,
    "q4": 1
  },
  "time_taken_seconds": 420
}
```

**Response (200 OK):**

```json
{
  "score": 85,
  "passed": true,
  "total_questions": 20,
  "correct_answers": 17,
  "attempt_number": 1,
  "certificate_id": "cert-abc123",
  "feedback": {
    "q2": "The correct answer is option 1: TLS 1.3 provides forward secrecy by default.",
    "q8": "Review the section on social engineering techniques.",
    "q15": "Multi-factor authentication requires something you know AND something you have."
  }
}
```

---

### GET /api/v1/secops/training/my-progress

Get the current user's overall training progress.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | User ID |

**Response (200 OK):**

```json
{
  "user_id": "user-001",
  "total_required": 8,
  "total_completed": 5,
  "total_in_progress": 1,
  "total_overdue": 1,
  "completion_rate": 62.5,
  "average_score": 88.5,
  "certificates": ["cert-001", "cert-002", "cert-003"],
  "expiring_soon": ["cert-001"]
}
```

---

### GET /api/v1/secops/training/team-compliance

Get training compliance statistics for a team.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `team_id` | string | Yes | Team ID |

**Response (200 OK):**

```json
{
  "team_id": "team-platform",
  "team_name": "Platform Engineering",
  "total_members": 12,
  "compliant_members": 10,
  "overdue_members": 2,
  "compliance_rate": 83.3,
  "average_score": 91.2
}
```

---

## Phishing Campaign Endpoints

### POST /api/v1/secops/phishing/campaigns

Create a new phishing simulation campaign.

**Request Body:**

```json
{
  "name": "Q2 2026 Phishing Exercise",
  "template_type": "credential_harvest",
  "target_user_ids": [],
  "target_roles": ["developer", "analyst"],
  "scheduled_at": "2026-04-15T09:00:00Z"
}
```

**Template Types:** `credential_harvest`, `malware_link`, `wire_transfer`, `data_request`, `urgency`

---

### POST /api/v1/secops/phishing/campaigns/{campaign_id}/send

Send phishing simulation emails for a campaign.

**Response (200 OK):**

```json
{
  "campaign_id": "camp-001",
  "emails_sent": 45,
  "sent_at": "2026-04-15T09:00:00Z"
}
```

---

### GET /api/v1/secops/phishing/campaigns/{campaign_id}/metrics

Get detailed metrics for a phishing campaign.

**Response (200 OK):**

```json
{
  "campaign_id": "camp-001",
  "total_sent": 45,
  "total_opened": 38,
  "total_clicked": 8,
  "total_credentials": 3,
  "total_reported": 15,
  "open_rate": 84.4,
  "click_rate": 17.8,
  "credential_rate": 6.7,
  "report_rate": 33.3
}
```

---

## Security Score Endpoints

### GET /api/v1/secops/security-score

Get a user's composite security score based on training completion, phishing performance, and compliance.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | User ID |

**Response (200 OK):**

```json
{
  "user_id": "user-001",
  "score": 82,
  "components": {
    "training_completion": 90.0,
    "phishing_resistance": 75.0,
    "policy_compliance": 85.0,
    "incident_reporting": 80.0
  },
  "calculated_at": "2026-04-04T12:00:00Z",
  "trend": "improving",
  "suggestions": [
    "Complete the 'Advanced Phishing Defense' course to improve phishing resistance.",
    "Review the updated password policy guidelines."
  ]
}
```

---

### GET /api/v1/secops/security-score/leaderboard

Get the security score leaderboard.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `team_id` | string | - | Filter by team |
| `limit` | integer | 10 | Number of entries (max 100) |

**Response (200 OK):**

```json
{
  "team_id": null,
  "entries": [
    {
      "rank": 1,
      "user_id": "user-042",
      "score": 98,
      "trend": "stable"
    },
    {
      "rank": 2,
      "user_id": "user-017",
      "score": 95,
      "trend": "improving"
    }
  ],
  "organization_average": 78.5
}
```
