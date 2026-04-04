# EUDR Agents API Reference

## Overview

The GreenLang EUDR (EU Deforestation Regulation) Agent Suite provides 40 specialized
agents that collectively implement the full lifecycle of EU 2023/1115 compliance.
Each agent exposes a REST API (FastAPI) with JWT authentication (SEC-001) and
RBAC authorization (SEC-002).

**Base URL:** `https://api.greenlang.io/api`

**Authentication:** All endpoints (except `/health`) require a JWT Bearer token
obtained via the GreenLang OAuth2 token endpoint.

**Rate Limits:**
- Standard: 100 requests/minute per user
- Write: 30 requests/minute per user
- Heavy/Batch: 10 requests/minute per user
- Export: 5 requests/minute per user

**Common Error Responses:**

| Status | Description |
|--------|-------------|
| 401 | Authentication required -- missing or invalid JWT |
| 403 | Insufficient permissions -- RBAC check failed |
| 404 | Resource not found |
| 422 | Validation error -- invalid request body |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service unavailable -- upstream dependency down |

---

## Agent Categories

### Category 1 -- Supply Chain Traceability (Agents 001-015)

Agents that map, trace, verify, and document the physical supply chain from
production plot to EU border.

| # | Agent | Prefix | Endpoints | Doc |
|---|-------|--------|-----------|-----|
| 001 | [Supply Chain Mapper](supply_chain_mapper.md) | `/v1/eudr-scm` | 23 | Detailed |
| 002 | [Geolocation Verification](geolocation_verification.md) | `/v1/eudr-geo` | 24 | Detailed |
| 003 | [Satellite Monitoring](satellite_monitoring.md) | `/v1/eudr-sat` | 29 | Detailed |
| 004 | [Forest Cover Analysis](forest_cover_analysis.md) | `/v1/eudr-fca` | 32 | Detailed |
| 005 | GPS Coordinate Validator | `/v1/eudr-gcv` | 25 | -- |
| 006 | Plot Boundary Agent | `/v1/eudr-pba` | 26 | -- |
| 007 | Land Use Change Agent | `/v1/eudr-luc` | 28 | -- |
| 008 | Multi-Tier Supplier Agent | `/v1/eudr-mts` | 27 | -- |
| 009 | [Chain of Custody](chain_of_custody.md) | `/v1/eudr-coc` | 37 | Detailed |
| 010 | Segregation Verifier | `/v1/eudr-sgv` | 28 | -- |
| 011 | [Mass Balance Calculator](mass_balance_calculator.md) | `/v1/eudr-mbc` | 37 | Detailed |
| 012 | [Document Authentication](document_authentication.md) | `/v1/eudr-dav` | 37 | Detailed |
| 013 | [Blockchain Integration](blockchain_integration.md) | `/v1/eudr-bci` | 39 | Detailed |
| 014 | QR Code Generator | `/v1/eudr-qrg` | 30 | -- |
| 015 | Mobile Data Collector | `/v1/eudr-mdc` | 30 | -- |

### Category 2 -- Risk Assessment (Agents 016-020)

Agents that evaluate country, supplier, commodity, and deforestation risk in
compliance with EUDR Articles 10 and 29.

| # | Agent | Prefix | Endpoints | Doc |
|---|-------|--------|-----------|-----|
| 016 | [Country Risk Evaluator](country_risk_evaluator.md) | `/v1/eudr-cre` | 38 | Detailed |
| 017 | [Supplier Risk Scorer](supplier_risk_scorer.md) | `/v1/eudr-srs` | 43 | Detailed |
| 018 | [Commodity Risk Analyzer](commodity_risk_analyzer.md) | `/v1/eudr-cra` | 42 | Detailed |
| 019 | Corruption Index Monitor | `/v1/eudr-cim` | 32 | -- |
| 020 | [Deforestation Alert System](deforestation_alert_system.md) | `/v1/eudr-das` | 39 | Detailed |

### Category 3 -- Due Diligence Core (Agents 021-026)

Agents that execute the due diligence workflow mandated by EUDR Articles 8-12,
including legal compliance, third-party auditing, and orchestration.

| # | Agent | Prefix | Endpoints | Doc |
|---|-------|--------|-----------|-----|
| 021 | Protected Area Validator | `/v1/eudr-pav` | 30 | -- |
| 022 | Indigenous Rights Checker | `/v1/eudr-irc` | 28 | -- |
| 023 | [Legal Compliance Verifier](legal_compliance_verifier.md) | `/v1/eudr-lcv` | 38 | Detailed |
| 024 | [Third-Party Audit Manager](third_party_audit_manager.md) | `/v1/eudr-tam` | 43 | Detailed |
| 025 | Stakeholder Engagement | -- | 10+ | -- |
| 026 | [Due Diligence Orchestrator](due_diligence_orchestrator.md) | `/api/v1/eudr-ddo` | 35 | Detailed |

### Category 4 -- Support Agents (Agents 027-029)

Agents that support information gathering, risk assessment computation, and
risk mitigation advisory.

| # | Agent | Prefix | Endpoints | Doc |
|---|-------|--------|-----------|-----|
| 027 | Information Gathering | `/api/v1/eudr/information-gathering` | 11 | -- |
| 028 | [Risk Assessment Engine](risk_assessment_engine.md) | `/api/v1/eudr/risk-assessment-engine` | 12 | Detailed |
| 029 | Risk Mitigation Advisor | `/v1/eudr-rma` | 35 | -- |

### Category 5 -- Due Diligence Workflow (Agents 030-040)

Agents that handle documentation generation, continuous monitoring, DDS
creation, EU IS submission, and lifecycle management.

| # | Agent | Prefix | Endpoints | Doc |
|---|-------|--------|-----------|-----|
| 030 | Documentation Generator | `/api/v1/eudr/documentation-generator` | 12 | -- |
| 031 | Mitigation Measure Designer | `/api/v1/eudr/mitigation-measure-designer` | 12+ | -- |
| 032 | Improvement Plan Creator | `/api/v1/eudr/improvement-plan-creator` | 12+ | -- |
| 033 | Continuous Monitoring | `/api/v1/eudr/continuous-monitoring` | 30+ | -- |
| 034 | Grievance Mechanism Manager | -- | 12+ | -- |
| 035 | Authority Communication Manager | -- | 12+ | -- |
| 036 | EU Information System Interface | `/api/v1/eudr/eu-information-system-interface` | 12 | -- |
| 037 | [Due Diligence Statement Creator](due_diligence_statement_creator.md) | `/api/v1/eudr/dds-creator` | 30 | Detailed |
| 038 | Reference Number Generator | `/api/v1/eudr/reference-number-generator` | 25+ | -- |
| 039 | Annual Review Scheduler | -- | 12+ | -- |
| 040 | Customs Declaration Support | -- | 12+ | -- |

---

## Authentication

All EUDR agent APIs share the GreenLang platform authentication flow.

### Obtain Access Token

```http
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&
client_id=your_client_id&
client_secret=your_client_secret
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

### Using the Token

Include the token in the `Authorization` header of every request:

```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
```

### RBAC Permissions

Each agent defines its own permission namespace. Examples:

- `eudr-supply-chain:*` -- Supply Chain Mapper
- `eudr-geolocation:*` -- Geolocation Verification
- `eudr-satellite:*` -- Satellite Monitoring
- `eudr-risk-assessment-engine:*` -- Risk Assessment Engine
- `eudr-dds-creator:*` -- Due Diligence Statement Creator

Wildcard (`*`) grants all sub-permissions within the namespace.

---

## Pagination

Endpoints returning lists support pagination via query parameters:

```http
GET /v1/eudr-scm/graphs?page=1&per_page=50
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number (1-indexed) |
| `per_page` | integer | 20 | Items per page (max 100) |

---

## Health Checks

Every agent exposes a `GET /health` endpoint that requires no authentication
and returns the agent identity, version, and status:

```json
{
  "status": "healthy",
  "agent_id": "GL-EUDR-SCM-001",
  "agent_name": "EUDR Supply Chain Mapper Agent",
  "version": "1.0.0",
  "timestamp": "2026-04-04T12:00:00Z"
}
```

---

## Provenance Tracking

All write operations record a SHA-256 provenance hash in the response,
enabling immutable audit trails per EUDR Article 31. Batch job responses
include a `provenance_hash` field computed from the job parameters and
submitting user identity.

---

## SDK Libraries

```python
from greenlang import Client

client = Client(client_id="...", client_secret="...")

# Supply Chain Mapper
graph = client.eudr.scm.create_graph(operator_id="OP-001", ...)

# Risk Assessment Engine
assessment = client.eudr.rae.assess_risk(operator_id="OP-001", commodity="cocoa", ...)

# Due Diligence Statement Creator
dds = client.eudr.dds.create_dds(operator_id="OP-001", commodities=["cocoa"], ...)
```

---

## Support

- **Documentation:** https://docs.greenlang.io/eudr
- **API Status:** https://status.greenlang.io
- **Support:** support@greenlang.io
