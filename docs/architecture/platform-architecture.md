# GreenLang Carbon Intelligence Platform - Architecture Documentation

**Version:** 2.0.0
**Last Updated:** 2025-11-08

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Components](#system-components)
3. [Application Architecture](#application-architecture)
4. [Data Flow](#data-flow)
5. [Integration Patterns](#integration-patterns)
6. [Scalability Architecture](#scalability-architecture)
7. [Security Architecture](#security-architecture)
8. [Deployment Topologies](#deployment-topologies)

---

## Architecture Overview

### Platform Vision

The GreenLang Carbon Intelligence Platform is an enterprise-grade, cloud-native solution that unifies three critical carbon accounting and compliance applications into a single, integrated ecosystem.

### Design Principles

1. **Microservices Architecture**: Each application is independently deployable
2. **Shared Infrastructure**: Common databases, caching, and monitoring
3. **API-First**: All interactions through well-defined REST/GraphQL APIs
4. **Event-Driven**: Asynchronous communication via message queues
5. **Cloud-Native**: Designed for containerization and orchestration
6. **Observable**: Built-in monitoring, logging, and tracing
7. **Secure by Default**: Zero-trust security model

### High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────────────┐  │
│  │   Web UI     │    │   Web UI     │    │      React Frontend         │  │
│  │  (CBAM)      │    │  (CSRD)      │    │      (VCCI Scope 3)         │  │
│  │   Port N/A   │    │   Port N/A   │    │      Port 3000              │  │
│  └──────────────┘    └──────────────┘    └─────────────────────────────┘  │
│         │                   │                           │                   │
└─────────┼───────────────────┼───────────────────────────┼───────────────────┘
          │                   │                           │
          └───────────────────┴───────────────────────────┘
                                    │
┌────────────────────────────────────────────────────────────────────────────┐
│                          API GATEWAY LAYER                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                    ┌────────────────────────────┐                           │
│                    │     NGINX / Kong           │                           │
│                    │   API Gateway + LB         │                           │
│                    │   - Rate Limiting          │                           │
│                    │   - Authentication         │                           │
│                    │   - SSL Termination        │                           │
│                    │   - Request Routing        │                           │
│                    └────────────────────────────┘                           │
│                              │                                               │
└──────────────────────────────┼───────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐ │
│  │   GL-CBAM-APP    │  │  GL-CSRD-APP     │  │    GL-VCCI-APP           │ │
│  │   Port 8001      │  │  Port 8002       │  │    Port 8000             │ │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────────────┤ │
│  │ CBAM Copilot     │  │ CSRD Platform    │  │ Scope 3 Platform         │ │
│  │                  │  │                  │  │                          │ │
│  │ Agents:          │  │ Components:      │  │ Services:                │ │
│  │ • Intake         │  │ • LLM Agent      │  │ • Backend API            │ │
│  │ • Calculator     │  │ • XBRL Gen       │  │ • 5 Agent Services       │ │
│  │ • Reporter       │  │ • Validator      │  │   - Intake               │ │
│  │                  │  │ • Reporter       │  │   - Calculator           │ │
│  │ Tech Stack:      │  │                  │  │   - Hotspot              │ │
│  │ • Python/FastAPI │  │ Tech Stack:      │  │   - Engagement           │ │
│  │ • Pandas         │  │ • Python/FastAPI │  │   - Reporting            │ │
│  │ • Pydantic       │  │ • LangChain      │  │ • Entity MDM             │ │
│  │                  │  │ • Arelle (XBRL)  │  │ • PCF Exchange           │ │
│  │ Database:        │  │ • Celery         │  │ • Policy Engine (OPA)    │ │
│  │ • Optional       │  │                  │  │                          │ │
│  │   (audit logs)   │  │ Database:        │  │ Tech Stack:              │ │
│  │                  │  │ • PostgreSQL     │  │ • Python/FastAPI         │ │
│  │                  │  │ • Redis          │  │ • React Frontend         │ │
│  │                  │  │                  │  │ • Celery                 │ │
│  │                  │  │                  │  │ • Weaviate (Vector DB)   │ │
│  │                  │  │                  │  │                          │ │
│  │                  │  │                  │  │ Databases:               │ │
│  │                  │  │                  │  │ • PostgreSQL             │ │
│  │                  │  │                  │  │ • Redis                  │ │
│  │                  │  │                  │  │ • Weaviate               │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘ │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         INTEGRATION LAYER                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Message Queue (RabbitMQ / SQS)                 │   │
│  │  Exchanges:                                                         │   │
│  │  • greenlang.emissions.calculated                                   │   │
│  │  • greenlang.report.generated                                       │   │
│  │  • greenlang.cbam.submitted                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Celery Task Queue                              │   │
│  │  Workers:                                                           │   │
│  │  • CSRD Workers (LLM processing, XBRL generation)                   │   │
│  │  • VCCI Workers (emissions calc, entity resolution, PCF exchange)   │   │
│  │  • Shared Workers (cross-app sync, notifications)                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    PostgreSQL Cluster (Multi-AZ)                     │  │
│  │  Databases:                                                          │  │
│  │  • cbam_db     - CBAM audit logs (optional)                          │  │
│  │  • csrd_db     - CSRD reports, XBRL documents, LLM cache            │  │
│  │  • vcci_db     - Scope 3 emissions, entities, suppliers              │  │
│  │  • shared_db   - Users, orgs, cross-app sync                        │  │
│  │                                                                      │  │
│  │  Topology:                                                           │  │
│  │  • Primary (writes)                                                  │  │
│  │  • Replica 1 (reads)                                                 │  │
│  │  • Replica 2 (reads, backups)                                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Redis Cluster (High Availability)                 │  │
│  │  Use Cases:                                                          │  │
│  │  • DB 0: API response cache                                          │  │
│  │  • DB 1: Celery broker                                               │  │
│  │  • DB 2: Celery results                                              │  │
│  │  • DB 3: Session store                                               │  │
│  │  • DB 4: Rate limiting                                               │  │
│  │  • DB 5: LLM response cache                                          │  │
│  │                                                                      │  │
│  │  Topology:                                                           │  │
│  │  • 3 Masters (sharded)                                               │  │
│  │  • 3 Replicas (one per master)                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Weaviate Vector Database                          │  │
│  │  Collections:                                                        │  │
│  │  • Entity         - Company/supplier entity resolution               │  │
│  │  • Product        - Product carbon footprints                        │  │
│  │  • Supplier       - Supplier profiles and embeddings                 │  │
│  │  • EmissionFactor - Semantic search for emission factors             │  │
│  │                                                                      │  │
│  │  Used by: GL-VCCI-APP (Entity MDM, semantic search)                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Object Storage (S3 / Blob / GCS)                  │  │
│  │  Buckets:                                                            │  │
│  │  • greenlang-data         - Raw input files                          │  │
│  │  • greenlang-reports      - Generated reports (PDF, XBRL)            │  │
│  │  • greenlang-backups      - Database backups                         │  │
│  │  • greenlang-logs         - Application logs (long-term)             │  │
│  │  • greenlang-pcf-exchange - PCF exchange data                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY LAYER                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐      │
│  │ Prometheus  │  │   Grafana   │  │    Loki     │  │   Jaeger     │      │
│  │  (Metrics)  │  │ (Dashboards)│  │   (Logs)    │  │  (Tracing)   │      │
│  │  Port 9090  │  │  Port 3000  │  │  Port 3100  │  │  Port 16686  │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘      │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │              AlertManager (Alerting & Notifications)              │     │
│  │              Port 9093                                            │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │              Sentry (Error Tracking & Performance)                │     │
│  │              SaaS                                                 │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    EXTERNAL INTEGRATIONS                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        ERP Systems                                   │  │
│  │  • SAP S/4HANA (OData API)                                           │  │
│  │  • Oracle ERP Cloud (REST API)                                       │  │
│  │  • Workday (REST API)                                                │  │
│  │                                                                      │  │
│  │  Used by: GL-VCCI-APP (transaction data extraction)                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        LLM Providers                                 │  │
│  │  • OpenAI (GPT-4, GPT-3.5)                                           │  │
│  │  • Anthropic (Claude Opus, Sonnet, Haiku)                            │  │
│  │                                                                      │  │
│  │  Used by: GL-CSRD-APP (ESRS reporting), GL-VCCI-APP (analysis)      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Entity Master Data Providers                      │  │
│  │  • GLEIF (Legal Entity Identifiers)                                  │  │
│  │  • Dun & Bradstreet (DUNS numbers)                                   │  │
│  │  • OpenCorporates (Company data)                                     │  │
│  │                                                                      │  │
│  │  Used by: GL-VCCI-APP (Entity MDM)                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    PCF Exchange Networks                             │  │
│  │  • PACT Pathfinder (WBCSD)                                           │  │
│  │  • Catena-X (Automotive supply chain)                                │  │
│  │  • SAP Sustainability Data Exchange                                  │  │
│  │                                                                      │  │
│  │  Used by: GL-VCCI-APP (PCF data exchange)                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Identity Providers                                │  │
│  │  • Okta (SSO, MFA)                                                   │  │
│  │  • Azure AD (Enterprise SSO)                                         │  │
│  │  • Auth0 (OAuth 2.0, OIDC)                                           │  │
│  │                                                                      │  │
│  │  Used by: All apps (authentication)                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Notification Services                             │  │
│  │  • SendGrid (Email)                                                  │  │
│  │  • Twilio (SMS)                                                      │  │
│  │  • Slack (Webhooks)                                                  │  │
│  │                                                                      │  │
│  │  Used by: All apps (notifications)                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## System Components

### Component Inventory

| Component | Purpose | Technology | HA | Backups |
|-----------|---------|------------|----|------------|
| **API Gateway** | Request routing, rate limiting, SSL | NGINX / Kong | Yes (multi-node) | Config only |
| **GL-CBAM-APP** | CBAM compliance reporting | Python/FastAPI | Yes | Stateless |
| **GL-CSRD-APP** | CSRD/ESRS reporting | Python/FastAPI/Celery | Yes | Reports in S3 |
| **GL-VCCI-APP** | Scope 3 emissions platform | Python/FastAPI/React/Celery | Yes | Data in PostgreSQL |
| **PostgreSQL** | Primary data store | PostgreSQL 15 | Yes (Multi-AZ) | Daily + WAL |
| **Redis** | Cache, queue, session | Redis 7 | Yes (cluster) | AOF + RDB |
| **Weaviate** | Vector database | Weaviate 1.23 | Yes (cluster) | Snapshots |
| **RabbitMQ** | Message queue | RabbitMQ 3.12 | Yes (cluster) | Persistent queues |
| **Object Storage** | Files, backups | S3 / Blob / GCS | Yes (cloud-native) | Versioning |
| **Prometheus** | Metrics collection | Prometheus 2.45 | Optional | Config only |
| **Grafana** | Dashboards | Grafana 10.0 | Optional | Dashboard JSON |
| **Loki** | Log aggregation | Loki 2.8 | Optional | S3 backend |
| **Jaeger** | Distributed tracing | Jaeger 1.48 | Optional | Cassandra backend |

---

## Application Architecture

### GL-CBAM-APP Architecture

```
┌────────────────────────────────────────────────────┐
│              GL-CBAM-APP (Port 8001)               │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │           FastAPI Application                │ │
│  │  - REST API endpoints                        │ │
│  │  - OpenAPI/Swagger docs                      │ │
│  │  - Health checks                             │ │
│  └──────────────────────────────────────────────┘ │
│                     │                              │
│        ┌────────────┼────────────┐                 │
│        ▼            ▼            ▼                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────────┐          │
│  │ Agent 1 │ │ Agent 2 │ │  Agent 3    │          │
│  │ Intake  │ │Calc     │ │  Reporter   │          │
│  │         │ │         │ │             │          │
│  │ • Parse │ │ • Lookup│ │ • Aggregate │          │
│  │ • Valid │ │   EF    │ │ • Format    │          │
│  │ • Enrich│ │ • Calc  │ │   CBAM JSON │          │
│  │         │ │   (100%)│ │ • Validate  │          │
│  └─────────┘ └─────────┘ └─────────────┘          │
│       │           │              │                 │
│       ▼           ▼              ▼                 │
│  ┌──────────────────────────────────────────────┐ │
│  │           Data Layer (Static)                │ │
│  │  • CN codes (JSON)                           │ │
│  │  • Emission factors (Python dict)            │ │
│  │  • CBAM rules (YAML)                         │ │
│  │  • Supplier profiles (YAML)                  │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  Optional: PostgreSQL for audit logs              │
│                                                    │
└────────────────────────────────────────────────────┘

Key Characteristics:
• Zero Hallucination: No LLM for calculations
• Deterministic: Same input → same output
• Stateless: Can scale horizontally easily
• Fast: <10 min for 10K shipments
• Standalone: Minimal dependencies
```

### GL-CSRD-APP Architecture

```
┌─────────────────────────────────────────────────────────┐
│              GL-CSRD-APP (Port 8002)                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │           FastAPI Web Application                 │ │
│  │  - REST API (CRUD operations)                     │ │
│  │  - Authentication (JWT)                           │ │
│  │  - File uploads                                   │ │
│  └───────────────────────────────────────────────────┘ │
│                        │                                │
│        ┌───────────────┼───────────────┐                │
│        ▼               ▼               ▼                │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐        │
│  │ LLM Agent│  │ XBRL Gen │  │ Validator      │        │
│  │          │  │          │  │                │        │
│  │ • ESRS   │  │ • Arelle │  │ • 50+ rules    │        │
│  │   analysis│  │ • iXBRL  │  │ • Completeness │        │
│  │ • Data   │  │   gen    │  │ • Consistency  │        │
│  │   extract│  │ • Schema │  │                │        │
│  └──────────┘  └──────────┘  └────────────────┘        │
│       │               │               │                 │
│       ▼               ▼               ▼                 │
│  ┌───────────────────────────────────────────────────┐ │
│  │              Celery Task Queue                    │ │
│  │  - Async LLM calls                                │ │
│  │  - XBRL generation                                │ │
│  │  - Report processing                              │ │
│  └───────────────────────────────────────────────────┘ │
│       │                                                 │
│       ▼                                                 │
│  ┌───────────────────────────────────────────────────┐ │
│  │              Data Layer                           │ │
│  │                                                   │ │
│  │  PostgreSQL:                                      │ │
│  │  • Organizations                                  │ │
│  │  • CSRD reports                                   │ │
│  │  • ESRS disclosures                               │ │
│  │  • XBRL documents                                 │ │
│  │  • Audit trail                                    │ │
│  │                                                   │ │
│  │  Redis:                                           │ │
│  │  • LLM response cache                             │ │
│  │  • Celery broker                                  │ │
│  │  • Session store                                  │ │
│  │                                                   │ │
│  │  S3/Blob:                                         │ │
│  │  • Generated XBRL files                           │ │
│  │  • PDF reports                                    │ │
│  │  • Source documents                               │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘

Key Characteristics:
• LLM-powered: Uses GPT-4/Claude for analysis
• Async processing: Celery for long-running tasks
• Compliance-focused: XBRL/ESEF generation
• Cached: Aggressive LLM response caching
• Scalable: Web + worker separation
```

### GL-VCCI-APP Architecture

```
┌────────────────────────────────────────────────────────────────┐
│              GL-VCCI-APP (Ports 8000, 3000)                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │               React Frontend (Port 3000)                 │ │
│  │  - Dashboard                                             │ │
│  │  - Data visualization (Plotly)                           │ │
│  │  - Supplier engagement portal                            │ │
│  └──────────────────────────────────────────────────────────┘ │
│                            │                                   │
│                            │ REST API                          │
│                            ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │          FastAPI Backend (Port 8000)                     │ │
│  │  - REST API + WebSocket                                  │ │
│  │  - Authentication (JWT)                                  │ │
│  │  - File uploads                                          │ │
│  │  - Real-time updates                                     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                            │                                   │
│        ┌───────────────────┼─────────────────┐                 │
│        ▼                   ▼                 ▼                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              5 Agent Services                            │ │
│  │                                                          │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │ │
│  │  │ Intake   │  │Calculator│  │ Hotspot  │              │ │
│  │  │ Agent    │  │ Agent    │  │ Agent    │              │ │
│  │  │          │  │          │  │          │              │ │
│  │  │ • ERP    │  │ • GHG    │  │ • Pareto │              │ │
│  │  │   extract│  │   Protocol│  │   80/20  │              │ │
│  │  │ • Normalize│  │ • Factors│  │ • Priority│              │ │
│  │  └──────────┘  └──────────┘  └──────────┘              │ │
│  │                                                          │ │
│  │  ┌──────────┐  ┌──────────┐                            │ │
│  │  │Engagement│  │Reporting │                            │ │
│  │  │ Agent    │  │ Agent    │                            │ │
│  │  │          │  │          │                            │ │
│  │  │ • Survey │  │ • PPTX   │                            │ │
│  │  │   gen    │  │ • PDF    │                            │ │
│  │  │ • Track  │  │ • XBRL   │                            │ │
│  │  └──────────┘  └──────────┘                            │ │
│  └──────────────────────────────────────────────────────────┘ │
│                            │                                   │
│        ┌───────────────────┼─────────────────┐                 │
│        ▼                   ▼                 ▼                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              Advanced Services                           │ │
│  │                                                          │ │
│  │  ┌─────────────────┐  ┌─────────────────┐              │ │
│  │  │  Entity MDM     │  │ PCF Exchange    │              │ │
│  │  │                 │  │                 │              │ │
│  │  │ • 2-stage       │  │ • PACT API      │              │ │
│  │  │   resolution    │  │ • Catena-X      │              │ │
│  │  │ • LEI/DUNS      │  │ • SAP SDX       │              │ │
│  │  │ • Weaviate      │  │                 │              │ │
│  │  └─────────────────┘  └─────────────────┘              │ │
│  │                                                          │ │
│  │  ┌─────────────────┐  ┌─────────────────┐              │ │
│  │  │ Policy Engine   │  │ Factor Broker   │              │ │
│  │  │ (OPA)           │  │                 │              │ │
│  │  │                 │  │ • Runtime API   │              │ │
│  │  │ • Data quality  │  │ • Factor query  │              │ │
│  │  │ • Validation    │  │ • Version mgmt  │              │ │
│  │  │ • Governance    │  │                 │              │ │
│  │  └─────────────────┘  └─────────────────┘              │ │
│  └──────────────────────────────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              Celery Workers                              │ │
│  │  - Emissions calculations (async)                        │ │
│  │  - Entity resolution (batch)                             │ │
│  │  - PCF exchange (scheduled)                              │ │
│  │  - Report generation                                     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              Data Layer                                  │ │
│  │                                                          │ │
│  │  PostgreSQL:                                             │ │
│  │  • Organizations, users                                  │ │
│  │  • Transactions, spend data                              │ │
│  │  • Emissions calculations                                │ │
│  │  • Suppliers, entities                                   │ │
│  │  • PCF data                                              │ │
│  │  • Engagement campaigns                                  │ │
│  │                                                          │ │
│  │  Weaviate:                                               │ │
│  │  • Entity embeddings                                     │ │
│  │  • Supplier profiles (vector search)                     │ │
│  │  • Product footprints (semantic search)                  │ │
│  │                                                          │ │
│  │  Redis:                                                  │ │
│  │  • API cache                                             │ │
│  │  • Celery broker/results                                 │ │
│  │  • Session store                                         │ │
│  │  • Rate limiting                                         │ │
│  │                                                          │ │
│  │  S3/Blob:                                                │ │
│  │  • ERP extract files                                     │ │
│  │  • Generated reports                                     │ │
│  │  • PCF exchange payloads                                 │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Key Characteristics:
• Most complex: Full-stack with React frontend
• Agent-based: 5 specialized agents
• AI-powered: LLM + ML + vector search
• ERP integrated: SAP, Oracle, Workday connectors
• Scalable: Microservices + async workers
• Policy-driven: OPA for governance
```

---

## Data Flow

### CBAM Reporting Flow

```
1. User uploads shipment data (CSV/Excel)
                │
                ▼
2. GL-CBAM-APP: Intake Agent
   • Parse file
   • Validate schema
   • Enrich with CN codes
                │
                ▼
3. GL-CBAM-APP: Calculator Agent
   • Lookup emission factors (DB)
   • Calculate emissions (deterministic)
   • Audit trail
                │
                ▼
4. GL-CBAM-APP: Reporter Agent
   • Aggregate by product group
   • Generate CBAM JSON
   • Validate compliance
                │
                ▼
5. User downloads report
   • CBAM Transitional Registry JSON
   • Human-readable summary (Markdown)
```

### CSRD Reporting Flow

```
1. User initiates CSRD report
                │
                ▼
2. GL-CSRD-APP: Data Collection
   • User inputs ESRS data points
   • Optional: Import from GL-VCCI-APP
                │
                ▼
3. GL-CSRD-APP: LLM Agent (async)
   • Analyze completeness
   • Generate narratives
   • Suggest improvements
                │
                ▼
4. GL-CSRD-APP: XBRL Generator (async)
   • Convert to iXBRL
   • Validate against taxonomy
   • Generate ESEF package
                │
                ▼
5. GL-CSRD-APP: Report Generation
   • PDF report (human-readable)
   • XBRL package (machine-readable)
   • Store in S3
                │
                ▼
6. User downloads report
   • ESEF-compliant XBRL
   • PDF report
```

### VCCI Scope 3 Calculation Flow

```
1. ERP Connector extracts transaction data
   • SAP S/4HANA OData API
   • Oracle REST API
   • Workday API
                │
                ▼
2. GL-VCCI-APP: Intake Agent
   • Normalize transactions
   • Classify by Scope 3 category
   • Map to suppliers
                │
                ▼
3. GL-VCCI-APP: Entity MDM
   • Resolve supplier identities
   • 2-stage resolution (exact → semantic)
   • Enrich with LEI/DUNS
   • Store in Weaviate
                │
                ▼
4. GL-VCCI-APP: Calculator Agent (async)
   • Lookup emission factors
   • Calculate emissions per transaction
   • Uncertainty quantification
   • Store results in PostgreSQL
                │
                ▼
5. GL-VCCI-APP: Hotspot Agent
   • Pareto analysis (80/20)
   • Identify top emitters
   • Prioritize engagement
                │
                ▼
6. GL-VCCI-APP: Engagement Agent
   • Generate supplier surveys
   • Send via email (SendGrid)
   • Track responses
   • Collect actual PCF data
                │
                ▼
7. GL-VCCI-APP: PCF Exchange
   • Export to PACT Pathfinder
   • Import from suppliers
   • Update emission factors
                │
                ▼
8. GL-VCCI-APP: Reporting Agent
   • Generate dashboards
   • Export to PowerPoint
   • Export to XBRL (for CSRD)
                │
                ▼
9. Optional: Sync to GL-CSRD-APP
   • Publish to message queue
   • GL-CSRD-APP consumes
   • Use for E1 climate disclosures
```

### Cross-Application Integration Flow

```
GL-VCCI-APP                    Message Queue              GL-CSRD-APP
     │                              │                          │
     │ Calculate Scope 3            │                          │
     │ emissions                    │                          │
     │                              │                          │
     ├─────── Publish Event ───────>│                          │
     │  greenlang.emissions         │                          │
     │    .calculated               │                          │
     │                              │                          │
     │                              ├─────── Consume ─────────>│
     │                              │                          │
     │                              │                      Store in
     │                              │                      csrd_db
     │                              │                          │
     │                              │                      Use for E1
     │                              │                      reporting
     │                              │                          │
GL-CBAM-APP                         │                          │
     │                              │                          │
     │ Generate CBAM report         │                          │
     │                              │                          │
     ├─────── Publish Event ───────>│                          │
     │  greenlang.cbam              │                          │
     │    .submitted                │                          │
     │                              │                          │
     │                              ├─────── Consume ─────────>│
     │                              │                          │
     │                              │                      Use for CBAM
     │                              │                      disclosures
```

---

## Integration Patterns

### Pattern 1: Synchronous REST API

**Use Case:** Real-time queries, CRUD operations

```python
# GL-VCCI-APP calls GL-CSRD-APP
async with httpx.AsyncClient() as client:
    response = await client.get(
        "http://csrd-app:8002/api/v1/organizations/123",
        headers={"Authorization": f"Bearer {jwt_token}"}
    )
    org_data = response.json()
```

**Pros:**
- Simple, easy to understand
- Immediate response
- Good for small data transfers

**Cons:**
- Coupling between services
- Timeout issues with slow operations
- Not suitable for bulk data

### Pattern 2: Asynchronous Message Queue

**Use Case:** Event notifications, bulk data sync

```python
# GL-VCCI-APP publishes emission calculation event
await message_broker.publish(
    exchange="greenlang.emissions.calculated",
    routing_key=f"vcci.scope3.{org_id}",
    message={
        "org_id": org_id,
        "period": "2025-Q1",
        "total_emissions_tco2": 12345.67,
        "categories": {...}
    }
)

# GL-CSRD-APP consumes event
@message_broker.subscribe("greenlang.emissions.calculated")
async def handle_emissions_calculated(message):
    # Store in CSRD database for E1 reporting
    await csrd_db.store_scope3_emissions(message)
```

**Pros:**
- Loose coupling
- Resilient (retries, dead letter queues)
- Scalable

**Cons:**
- Eventual consistency
- More complex debugging
- Requires message broker infrastructure

### Pattern 3: Shared Database

**Use Case:** Cross-app user management, organization data

```sql
-- Shared database: shared_db
CREATE TABLE organizations (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    org_id UUID REFERENCES organizations(id),
    roles JSONB NOT NULL,  -- {"cbam": ["admin"], "csrd": ["analyst"], "vcci": ["viewer"]}
    created_at TIMESTAMP DEFAULT NOW()
);

-- All apps read/write to shared_db
-- Each app has its own database for app-specific data
```

**Pros:**
- Single source of truth
- No sync latency
- Strong consistency

**Cons:**
- Schema coupling
- Migration coordination required
- Potential performance bottleneck

### Pattern 4: API Composition (Backend for Frontend)

**Use Case:** Unified frontend aggregating data from multiple apps

```python
# Unified API endpoint that calls multiple apps
@app.get("/api/v1/dashboard")
async def get_unified_dashboard(org_id: str):
    async with httpx.AsyncClient() as client:
        # Call all 3 apps in parallel
        cbam_data, csrd_data, vcci_data = await asyncio.gather(
            client.get(f"http://cbam-app:8001/api/summary?org={org_id}"),
            client.get(f"http://csrd-app:8002/api/v1/reports?org={org_id}"),
            client.get(f"http://vcci-app:8000/api/emissions?org={org_id}")
        )

        # Combine and return
        return {
            "cbam": cbam_data.json(),
            "csrd": csrd_data.json(),
            "vcci": vcci_data.json()
        }
```

**Pros:**
- Frontend simplicity
- Parallel data fetching
- Single authentication point

**Cons:**
- Additional latency
- BFF becomes single point of failure
- Requires careful caching

---

## Scalability Architecture

### Horizontal Scaling Strategy

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │   (AWS ALB)     │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ VCCI-API-1   │ │ VCCI-API-2   │ │ VCCI-API-N   │
    │  (Stateless) │ │  (Stateless) │ │  (Stateless) │
    └──────────────┘ └──────────────┘ └──────────────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Redis Cache   │
                    │  (Shared State) │
                    └─────────────────┘
```

**Auto-Scaling Rules (AWS):**

```yaml
WebTier:
  MetricType: TargetTrackingScaling
  TargetValue:
    CPU: 70%
    RequestCountPerTarget: 1000
  ScaleUp:
    Cooldown: 300s
    Instances: +2
  ScaleDown:
    Cooldown: 600s
    Instances: -1
  MinInstances: 2
  MaxInstances: 20

WorkerTier:
  MetricType: QueueDepth
  TargetValue:
    QueueLength: 100 messages
  ScaleUp:
    Cooldown: 180s
    Instances: +4
  ScaleDown:
    Cooldown: 900s
    Instances: -2
  MinInstances: 2
  MaxInstances: 100
```

### Database Scaling

**Read Replicas:**

```
Primary (writes)
    │
    ├─── Replica 1 (reads) ────> Application (read traffic)
    ├─── Replica 2 (reads) ────> Application (read traffic)
    └─── Replica 3 (backups) ──> Backup process
```

**Sharding Strategy (if needed):**

```sql
-- Shard by org_id (hash sharding)
-- Shard 0: org_id % 4 == 0
-- Shard 1: org_id % 4 == 1
-- Shard 2: org_id % 4 == 2
-- Shard 3: org_id % 4 == 3

-- Application-level routing
def get_shard(org_id):
    return int(hashlib.md5(org_id.encode()).hexdigest(), 16) % 4

conn = get_db_connection(shard_id=get_shard(org_id))
```

### Caching Strategy

```yaml
L1: Browser Cache (static assets)
  TTL: 1 year
  Size: Unlimited

L2: CDN Edge Cache (CloudFront, Cloudflare)
  TTL: 1 hour (API), 1 week (static)
  Size: Unlimited

L3: Redis Application Cache
  TTL: Varies by data type
  Size: 64 GB (cluster)

L4: Database Query Cache (PostgreSQL)
  TTL: Auto-invalidated
  Size: 25% of shared_buffers

L5: LLM Response Cache (Redis)
  TTL: 24 hours
  Size: Dedicated 32 GB
```

---

## Security Architecture

### Zero-Trust Security Model

```
┌─────────────────────────────────────────────────────┐
│          Every Request Must Be Authenticated        │
│             and Authorized at Every Layer           │
└─────────────────────────────────────────────────────┘

User Request
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  1. TLS Termination (Load Balancer)                 │
│     • Verify certificate                            │
│     • Enforce TLS 1.3                               │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  2. WAF (Web Application Firewall)                  │
│     • SQL injection detection                       │
│     • XSS prevention                                │
│     • Rate limiting                                 │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  3. API Gateway Authentication                      │
│     • JWT validation                                │
│     • OAuth 2.0 / OIDC                              │
│     • API key verification                          │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  4. Application Authorization (RBAC)                │
│     • Role check: admin, analyst, viewer            │
│     • Resource ownership: org_id match              │
│     • Permission check: read, write, delete         │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  5. Data Access Layer                               │
│     • Row-level security (PostgreSQL RLS)           │
│     • Encrypted connections (SSL)                   │
│     • Audit logging                                 │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  6. Data Encryption at Rest                         │
│     • PostgreSQL TDE                                │
│     • S3 encryption (AES-256)                       │
│     • Fernet encryption for sensitive fields        │
└─────────────────────────────────────────────────────┘
```

### Network Security

```
┌────────────────────────────────────────────────────┐
│              VPC / Virtual Network                  │
├────────────────────────────────────────────────────┤
│                                                    │
│  Public Subnet (0.0.0.0/0)                         │
│  ┌──────────────────────────────────────────────┐ │
│  │  Load Balancer                               │ │
│  │  NAT Gateway                                 │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  Private Subnet (Application Tier)                 │
│  ┌──────────────────────────────────────────────┐ │
│  │  Application Servers                         │ │
│  │  • No direct internet access                 │ │
│  │  • Egress via NAT Gateway                    │ │
│  │  • Ingress only from Load Balancer           │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  Private Subnet (Database Tier)                    │
│  ┌──────────────────────────────────────────────┐ │
│  │  Databases (PostgreSQL, Redis, Weaviate)     │ │
│  │  • No internet access                        │ │
│  │  • Accessible only from Application Subnet   │ │
│  │  • Backup access via VPN/Bastion             │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## Deployment Topologies

### Topology 1: Single-Server (Development)

```
┌─────────────────────────────────────────┐
│         Single EC2 / VM                 │
│         (m5.2xlarge: 8 vCPU, 32 GB)     │
├─────────────────────────────────────────┤
│                                         │
│  Docker Compose:                        │
│  • CBAM-APP                             │
│  • CSRD-APP (web + worker)              │
│  • VCCI-APP (backend + worker + frontend)│
│  • PostgreSQL                           │
│  • Redis                                │
│  • Weaviate                             │
│  • NGINX                                │
│  • Prometheus + Grafana                 │
│                                         │
└─────────────────────────────────────────┘

Cost: ~$350/month (AWS)
Use Case: Development, demos, POCs
```

### Topology 2: Multi-Server (Small Production)

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  App Server 1    │  │  App Server 2    │  │  Database Server │
│  (t3.large)      │  │  (t3.large)      │  │  (db.m5.large)   │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│                  │  │                  │  │                  │
│  • CBAM-APP      │  │  • CBAM-APP      │  │  • PostgreSQL    │
│  • CSRD-APP      │  │  • CSRD-APP      │  │    (Multi-AZ)    │
│  • VCCI-APP      │  │  • VCCI-APP      │  │  • Redis         │
│    (backend)     │  │    (backend)     │  │    (cluster)     │
│                  │  │                  │  │  • Weaviate      │
│  • Workers       │  │  • Workers       │  │                  │
│  • NGINX         │  │  • NGINX         │  │                  │
│                  │  │                  │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
        │                     │                      │
        └─────────────────────┴──────────────────────┘
                              │
                   ┌──────────────────┐
                   │  Load Balancer   │
                   │  (AWS ALB)       │
                   └──────────────────┘

Cost: ~$600/month (AWS)
Use Case: Small production (<100 orgs)
```

### Topology 3: Kubernetes (Large Production)

```
┌─────────────────────────────────────────────────────────────┐
│              Kubernetes Cluster (EKS / AKS / GKE)           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Namespaces:                                                │
│  • greenlang-cbam                                           │
│  • greenlang-csrd                                           │
│  • greenlang-vcci                                           │
│  • greenlang-shared                                         │
│  • monitoring                                               │
│                                                             │
│  Pods (Auto-scaled):                                        │
│  • cbam-app: 2-10 replicas                                  │
│  • csrd-web: 2-10 replicas                                  │
│  • csrd-worker: 4-50 replicas                               │
│  • vcci-backend: 2-20 replicas                              │
│  • vcci-worker: 4-100 replicas                              │
│  • vcci-frontend: 2-10 replicas                             │
│                                                             │
│  Services:                                                  │
│  • ClusterIP: Internal communication                        │
│  • LoadBalancer: External access                            │
│  • Ingress: nginx-ingress-controller                        │
│                                                             │
│  Persistent Volumes:                                        │
│  • EBS / Azure Disk / GCE PD                                │
│  • StorageClass: gp3 (AWS), Premium_LRS (Azure)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  RDS         │      │ ElastiCache  │      │  S3          │
│  PostgreSQL  │      │  Redis       │      │  Buckets     │
│  (Multi-AZ)  │      │  (cluster)   │      │              │
└──────────────┘      └──────────────┘      └──────────────┘

Cost: ~$5,000-20,000/month (depending on scale)
Use Case: Large production (100+ orgs, 1000+ users)
```

---

## Summary

The GreenLang Carbon Intelligence Platform is designed as a **cloud-native, microservices-based architecture** that provides:

1. **Modularity**: Each application can be deployed independently
2. **Scalability**: Horizontal and vertical scaling at every layer
3. **Resilience**: High availability, auto-recovery, disaster recovery
4. **Security**: Zero-trust model, encryption, audit logging
5. **Observability**: Comprehensive monitoring, logging, tracing
6. **Integration**: Event-driven architecture for loose coupling
7. **Performance**: Caching, connection pooling, optimized queries

**Design Trade-offs:**

| Aspect | Benefit | Cost |
|--------|---------|------|
| Microservices | Scalability, independence | Complexity, ops overhead |
| Shared Database | Consistency, simplicity | Coupling, migration coordination |
| Event-Driven | Loose coupling, scalability | Eventual consistency, debugging |
| Multi-Cloud | Vendor independence | Higher complexity |
| Auto-Scaling | Cost optimization, performance | Configuration complexity |

**Next Steps:**

- Review deployment options in `GREENLANG_PLATFORM_DEPLOYMENT.md`
- Implement environment-specific configurations
- Set up monitoring and alerting
- Perform load testing
- Execute disaster recovery drills

---

**Document Maintained By:** Platform Architecture Team
**Last Updated:** 2025-11-08
**Next Review:** 2025-12-08
