# GreenLang API Reference

## Overview

GreenLang provides a comprehensive REST API platform for climate compliance, emissions calculation, and sustainability reporting. The platform exposes **3,936 endpoints** across 101 agents, 10 applications, and infrastructure services.

**Base URL:** `https://api.greenlang.ai/api/v1/` (production) | `http://localhost:8000/api/v1/` (development)

## Quick Links

| Resource | Description |
|----------|-------------|
| [API Catalog](API_CATALOG.md) | Complete endpoint inventory with 3,936 endpoints |
| [Authentication](authentication.md) | JWT (RS256) and API key auth guide |
| [Error Codes](error-codes.md) | Error code reference by domain |
| [Pagination](pagination.md) | Offset-based pagination patterns |
| [Rate Limiting](rate-limiting.md) | Rate limit policies and headers |
| [API Versioning](versioning.md) | v1/v2 versioning strategy |

## Agent APIs

### Foundation Agents (10 agents, 165 endpoints)

Core platform services for orchestration, schema validation, and reproducibility.

| Agent | Endpoints | Purpose |
|-------|-----------|---------|
| [Orchestrator](agents/foundation/orchestrator.md) | 19 | DAG workflow execution and management |
| [Schema Compiler](agents/foundation/schema_compiler.md) | 15+ | Schema validation and compilation |
| [Unit Normalizer](agents/foundation/unit_normalizer.md) | 15+ | Unit conversion and normalization |
| [Assumptions Registry](agents/foundation/assumptions_registry.md) | 15+ | Calculation assumption tracking |
| [Citations & Evidence](agents/foundation/citations_evidence.md) | 15+ | Source citation management |
| [Access & Policy Guard](agents/foundation/access_policy.md) | 15+ | Policy enforcement |
| [Agent Registry](agents/foundation/agent_registry.md) | 20+ | Agent discovery and management |
| [Reproducibility](agents/foundation/reproducibility.md) | 15+ | Calculation reproducibility |
| [QA Test Harness](agents/foundation/qa_harness.md) | 15+ | Quality assurance testing |
| [Observability](agents/foundation/observability.md) | 15+ | Telemetry and monitoring |

### Data Agents (20 agents, 441 endpoints)

Data intake, quality profiling, and transformation services.

| Category | Agents | Purpose |
|----------|--------|---------|
| [Intake](agents/data/index.md) | PDF, Excel, ERP, API, EUDR, GIS, Satellite | Data ingestion from multiple sources |
| [Quality](agents/data/index.md) | Questionnaire, Spend, Profiler, Dedup, Imputer, Outlier, GapFill, Recon, Freshness, Schema, Lineage, Validation | Data cleansing and validation |
| [Geo](agents/data/index.md) | Climate Hazard Connector | Climate and geospatial data |

### MRV Agents (30 agents, 604 endpoints)

GHG emissions measurement, reporting, and verification per GHG Protocol.

| Scope | Agents | Categories |
|-------|--------|------------|
| [Scope 1](agents/mrv/index.md) | 8 | Stationary, Mobile, Process, Fugitive, Refrigerant, Land Use, Waste, Agricultural |
| [Scope 2](agents/mrv/index.md) | 5 | Location-Based, Market-Based, Steam, Cooling, Dual Reporting |
| [Scope 3](agents/mrv/index.md) | 15 | Categories 1-15 (Purchased Goods through Investments) |
| [Cross-Cutting](agents/mrv/index.md) | 2 | Category Mapper, Audit Trail |

### EUDR Agents (40 agents, 1,270 endpoints)

EU Deforestation Regulation compliance and supply chain traceability.

| Category | Agents | Purpose |
|----------|--------|---------|
| [Traceability](agents/eudr/index.md) | 15 | Supply chain mapping, geolocation, satellite monitoring |
| [Risk Assessment](agents/eudr/index.md) | 5 | Country, supplier, commodity risk scoring |
| [Due Diligence](agents/eudr/index.md) | 6 | Legal compliance, audit management |
| [Support](agents/eudr/index.md) | 3 | Reporting, notifications, training |
| [Workflow](agents/eudr/index.md) | 11 | DDS creation, remediation, approval workflows |

## Infrastructure APIs

Platform services for authentication, authorization, and operational management.

| Service | Purpose |
|---------|---------|
| [Authentication (JWT)](infrastructure/auth_service.md) | Login, token management, MFA |
| [Auth Admin](infrastructure/auth_admin.md) | User and tenant administration |
| [Authorization (RBAC)](infrastructure/rbac_service.md) | Role and permission management |
| [Agent Factory](infrastructure/agent_factory.md) | Agent lifecycle management |
| [Feature Flags](infrastructure/feature_flags.md) | Feature toggle management |
| [Audit Logging](infrastructure/audit_service.md) | Audit event tracking |
| [Encryption](infrastructure/encryption_service.md) | Data encryption services |
| [Secrets Management](infrastructure/secrets_service.md) | Secrets vault integration |
| [Security Scanning](infrastructure/security_scanning.md) | Vulnerability scanning |
| [PII Detection](infrastructure/pii_service.md) | PII detection and redaction |
| [SOC 2](infrastructure/soc2_preparation.md) | SOC 2 Type II compliance |
| [Incident Response](infrastructure/incident_response.md) | Security incident management |

## Application APIs

Pre-built compliance applications with end-to-end workflows.

| Application | Regulation | Status |
|-------------|------------|--------|
| [GL-CSRD-APP](applications/gl-csrd-app.md) | EU Corporate Sustainability Reporting Directive | v1.1 |
| [GL-CBAM-APP](applications/gl-cbam-app.md) | EU Carbon Border Adjustment Mechanism | v1.1 |
| [GL-EUDR-APP](applications/gl-eudr-app.md) | EU Deforestation Regulation | v1.0 |
| [GL-GHG-APP](applications/gl-ghg-app.md) | GHG Protocol Corporate Standard | v1.0 |
| [GL-VCCI-APP](applications/gl-vcci-carbon-app.md) | Value Chain Carbon Intelligence | v1.1 |
| [GL-ISO14064-APP](applications/gl-iso14064-app.md) | ISO 14064 Compliance | v1.0 |
| [GL-CDP-APP](applications/gl-cdp-app.md) | CDP Climate Disclosure | v1.0 Beta |
| [GL-TCFD-APP](applications/gl-tcfd-app.md) | TCFD Climate Risk Disclosure | v1.0 Beta |
| [GL-SBTi-APP](applications/gl-sbti-app.md) | Science Based Targets | v1.0 Beta |
| [GL-Taxonomy-APP](applications/gl-taxonomy-app.md) | EU Taxonomy Alignment | v1.0 Alpha |
| [CBAM Pack MVP](cbam-pack/endpoints.md) | CBAM Quick-Start Pack | MVP |

## Authentication

All API endpoints require authentication via JWT Bearer token or API key. See the [Authentication Guide](authentication.md) for details.

```bash
# JWT Bearer Token
curl -H "Authorization: Bearer eyJhbGc..." https://api.greenlang.ai/api/v1/agents

# API Key
curl -H "X-API-Key: glk_abc123..." https://api.greenlang.ai/api/v1/agents
```

## Error Handling

All errors follow a consistent format. See [Error Codes](error-codes.md) for the complete reference.

```json
{
  "error": {
    "code": "GL_AGENT_VALIDATION_ERROR",
    "message": "Missing required field: fuel_type",
    "details": { "agent_name": "FuelAgent", "timestamp": "2026-04-04T10:30:00Z" }
  }
}
```

## Tools

| Tool | Purpose |
|------|---------|
| `scripts/generate_api_catalog.py` | Regenerate the API catalog from source code |
| `mkdocs serve` | Preview documentation locally |
| `mkdocs build` | Build static documentation site |
