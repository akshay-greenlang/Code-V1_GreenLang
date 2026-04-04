# WAF Management Service API Reference (SEC-010)

## Overview

The WAF Management Service provides Web Application Firewall rule management, attack detection, DDoS mitigation, and AWS Shield Advanced integration. Supports rule CRUD, testing, deployment, attack tracking, and real-time metrics.

**Router Prefix:** `/api/v1/secops/waf`
**Tags:** `WAF Management`
**Source:** `greenlang/infrastructure/waf_management/api/waf_routes.py`

---

## Endpoint Summary

### Rule Management

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/secops/waf/rules` | List WAF rules | Yes |
| POST | `/api/v1/secops/waf/rules` | Create WAF rule | Yes |
| GET | `/api/v1/secops/waf/rules/{rule_id}` | Get rule details | Yes |
| PUT | `/api/v1/secops/waf/rules/{rule_id}` | Update rule | Yes |
| DELETE | `/api/v1/secops/waf/rules/{rule_id}` | Delete rule | Yes |
| POST | `/api/v1/secops/waf/rules/{rule_id}/test` | Test rule against sample traffic | Yes |
| POST | `/api/v1/secops/waf/rules/{rule_id}/deploy` | Deploy rule to AWS WAF | Yes |
| POST | `/api/v1/secops/waf/rules/{rule_id}/disable` | Disable rule without deleting | Yes |

### Attack Detection

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/secops/waf/attacks` | List detected attacks | Yes |
| GET | `/api/v1/secops/waf/attacks/{attack_id}` | Get attack details | Yes |
| POST | `/api/v1/secops/waf/attacks/{attack_id}/mitigate` | Manual attack mitigation | Yes |

### Metrics and Status

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/secops/waf/metrics` | WAF/DDoS metrics | Yes |
| GET | `/api/v1/secops/waf/shield/status` | AWS Shield Advanced status | Yes |

---

## Rule Management Endpoints

### GET /api/v1/secops/waf/rules

List WAF rules with pagination and optional filters.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page (max 100) |
| `rule_type` | string | - | Filter by type: `rate_limit`, `geo_block`, `ip_block`, `regex`, `managed` |
| `status` | string | - | Filter by status: `draft`, `active`, `disabled`, `deployed` |
| `enabled` | boolean | - | Filter by enabled state |

**Response (200 OK):**

```json
{
  "items": [
    {
      "id": "rule-abc123",
      "name": "rate-limit-api",
      "rule_type": "rate_limit",
      "priority": 100,
      "action": "block",
      "description": "Rate limit API requests to 2000/5min",
      "enabled": true,
      "status": "deployed",
      "rate_limit_threshold": 2000,
      "rate_limit_window_seconds": 300,
      "blocked_countries": [],
      "ip_set_arn": null,
      "regex_pattern": null,
      "managed_rule_group": null,
      "aws_rule_id": "aws-waf-rule-xyz",
      "created_at": "2026-03-01T10:00:00Z",
      "updated_at": "2026-04-01T14:30:00Z",
      "deployed_at": "2026-04-01T14:35:00Z",
      "created_by": "security-admin",
      "metrics": {},
      "metadata": {}
    }
  ],
  "total": 12,
  "page": 1,
  "page_size": 20,
  "total_pages": 1,
  "has_next": false,
  "has_prev": false
}
```

---

### POST /api/v1/secops/waf/rules

Create a new WAF rule. Rule names must be unique.

**Request Body:**

```json
{
  "name": "geo-block-sanctioned",
  "rule_type": "geo_block",
  "priority": 50,
  "action": "block",
  "description": "Block traffic from sanctioned countries",
  "enabled": true,
  "blocked_countries": ["KP", "IR", "SY"],
  "metadata": {
    "compliance": "OFAC"
  }
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Rule name (2-128 chars, unique) |
| `rule_type` | string | Yes | `rate_limit`, `geo_block`, `ip_block`, `regex`, `managed` |
| `priority` | integer | No | Evaluation priority 0-10000 (default: 100) |
| `action` | string | No | `allow`, `block`, `count`, `captcha` (default: `block`) |
| `description` | string | No | Rule description (max 2048 chars) |
| `enabled` | boolean | No | Whether rule is active (default: true) |
| `rate_limit_threshold` | integer | No | Rate limit threshold (default: 2000) |
| `rate_limit_window_seconds` | integer | No | Rate limit window (default: 300) |
| `blocked_countries` | array | No | ISO country codes to block |
| `ip_set_arn` | string | No | AWS IP set ARN |
| `regex_pattern` | string | No | Custom regex pattern |
| `managed_rule_group` | string | No | AWS managed rule group name |
| `conditions` | array | No | Rule conditions (field, operator, values) |
| `metadata` | object | No | Additional metadata |

**Response (201 Created):** Returns the created `RuleResponse` object.

**Error Responses:**

| Status Code | Description |
|-------------|-------------|
| 409 | Conflict - Rule with same name already exists |
| 422 | Validation Error - Invalid rule_type or action |

---

### POST /api/v1/secops/waf/rules/{rule_id}/test

Test a WAF rule against sample traffic. Generates synthetic legitimate and malicious requests and evaluates the rule's detection accuracy.

**Request Body:**

```json
{
  "test_requests": null,
  "include_legitimate": true,
  "malicious_count": 100
}
```

**Response (200 OK):**

```json
{
  "rule_name": "rate-limit-api",
  "rule_type": "rate_limit",
  "total_requests": 200,
  "total_matched": 95,
  "true_positives": 90,
  "false_positives": 5,
  "false_negatives": 10,
  "detection_rate": 0.90,
  "false_positive_rate": 0.05,
  "accuracy": 0.925,
  "precision": 0.947,
  "recall": 0.90,
  "f1_score": 0.923,
  "average_latency_us": 45.2,
  "p99_latency_us": 120.5,
  "recommendations": [
    "Consider lowering rate_limit_threshold to reduce false negatives"
  ]
}
```

---

### POST /api/v1/secops/waf/rules/{rule_id}/deploy

Deploy a rule to AWS WAF. The rule is validated before deployment.

**Request Body:**

```json
{
  "web_acl_id": "arn:aws:wafv2:us-east-1:123456:webacl/prod-acl"
}
```

**Error Responses:**

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Deployment error |
| 422 | Validation Error - Rule validation failed |

---

## Attack Detection Endpoints

### GET /api/v1/secops/waf/attacks

List detected attacks with filtering and pagination.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page (max 100) |
| `attack_type` | string | - | Filter: `ddos_volumetric`, `ddos_protocol`, `ddos_application`, `sql_injection`, `xss`, `brute_force` |
| `status` | string | - | Filter: `pending`, `in_progress`, `mitigated`, `resolved` |
| `severity` | string | - | Filter: `critical`, `high`, `medium`, `low` |

**Response (200 OK):**

```json
{
  "items": [
    {
      "id": "atk-001",
      "attack_type": "ddos_volumetric",
      "severity": "high",
      "source_ips": ["203.0.113.50", "198.51.100.25"],
      "target_endpoints": ["/api/v1/cbam/intake"],
      "requests_per_second": 50000,
      "total_requests": 3000000,
      "bytes_per_second": 125000000,
      "started_at": "2026-04-04T10:00:00Z",
      "detected_at": "2026-04-04T10:00:15Z",
      "mitigated_at": "2026-04-04T10:01:00Z",
      "ended_at": null,
      "status": "mitigated",
      "detection_source": "anomaly_detector",
      "attack_signature": "SYN flood with spoofed source IPs",
      "geographic_distribution": {"CN": 60, "RU": 25, "BR": 15},
      "metadata": {}
    }
  ],
  "total": 5,
  "page": 1,
  "page_size": 20,
  "active_count": 1,
  "mitigated_count": 4
}
```

---

### POST /api/v1/secops/waf/attacks/{attack_id}/mitigate

Manually trigger attack mitigation actions.

**Request Body:**

```json
{
  "actions": ["block_ips", "rate_limit", "geo_block"],
  "block_ips": ["203.0.113.50", "198.51.100.25"],
  "geo_block_countries": ["CN"],
  "engage_shield_drt": false
}
```

**Response (200 OK):**

```json
{
  "attack_id": "atk-001",
  "status": "mitigated",
  "actions_taken": [
    {"action": "block_ips", "ips_blocked": 2, "success": true},
    {"action": "rate_limit", "threshold": 100, "success": true}
  ],
  "effectiveness_score": 0.95,
  "traffic_reduction_percent": 92.5,
  "duration_seconds": 3.2,
  "shield_engaged": false,
  "recommendations": [
    "Monitor for attack pattern changes in the next 24 hours"
  ]
}
```

**Error Responses:**

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Attack already mitigated |
| 404 | Not Found - Attack not found |

---

## Metrics and Status Endpoints

### GET /api/v1/secops/waf/metrics

Get current WAF and DDoS protection metrics.

**Response (200 OK):**

```json
{
  "timestamp": "2026-04-04T12:00:00Z",
  "waf": {
    "rules_total": 12,
    "rules_active": 10,
    "requests_evaluated_total": 1500000,
    "requests_blocked_total": 2500,
    "false_positives_total": 15
  },
  "ddos": {
    "attacks_total": 5,
    "attacks_active": 1,
    "attacks_mitigated": 4
  },
  "traffic": {
    "requests_per_second": 850.0,
    "blocked_per_second": 1.5,
    "latency_p99_ms": 12.5
  },
  "shield": {
    "subscription_active": true,
    "protections_count": 8
  }
}
```

---

### GET /api/v1/secops/waf/shield/status

Get AWS Shield Advanced subscription and protection status.

**Response (200 OK):**

```json
{
  "subscription_active": true,
  "subscription_start": "2025-01-01T00:00:00Z",
  "auto_renew": true,
  "protections_count": 8,
  "protections": [
    {
      "id": "prot-001",
      "resource_arn": "arn:aws:elasticloadbalancing:us-east-1:123456:loadbalancer/app/prod-alb",
      "name": "prod-load-balancer"
    }
  ],
  "proactive_engagement_enabled": true,
  "attack_statistics": {
    "attacks_last_30_days": 3,
    "average_duration_minutes": 12.5
  }
}
```

---

## Rule Types

| Type | Description |
|------|-------------|
| `rate_limit` | Throttle requests exceeding a threshold within a time window |
| `geo_block` | Block traffic from specific countries |
| `ip_block` | Block traffic from specific IP addresses or CIDR ranges |
| `regex` | Match requests using custom regex patterns |
| `managed` | AWS-managed rule groups (e.g., AWSManagedRulesCommonRuleSet) |

## Rule Actions

| Action | Description |
|--------|-------------|
| `allow` | Allow the request |
| `block` | Block the request (403 response) |
| `count` | Allow but count the request (monitoring mode) |
| `captcha` | Challenge with CAPTCHA before allowing |
