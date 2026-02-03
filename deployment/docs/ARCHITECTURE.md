# GreenLang Platform - Architecture Documentation

**Document ID:** INFRA-001-ARCH
**Version:** 1.0.0
**Last Updated:** 2026-02-03
**Status:** Production

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Descriptions](#component-descriptions)
4. [Data Flow](#data-flow)
5. [Security Boundaries](#security-boundaries)
6. [Network Architecture](#network-architecture)
7. [Storage Architecture](#storage-architecture)
8. [Monitoring Architecture](#monitoring-architecture)

---

## Executive Summary

The GreenLang Platform is a unified carbon intelligence system comprising three core applications (CBAM, CSRD, VCCI) built on shared infrastructure. This document describes the system architecture, component interactions, data flows, and security boundaries.

### Key Architectural Principles

- **Shared Infrastructure:** Single PostgreSQL, Redis, RabbitMQ, and Weaviate instances serve all applications
- **Unified Authentication:** JWT-based authentication shared across all applications
- **Event-Driven Integration:** Cross-application communication via RabbitMQ message queues
- **Containerized Deployment:** Docker-based with Kubernetes support for production
- **Observable by Design:** Prometheus metrics and Grafana dashboards for all components

---

## High-Level Architecture

```
+==============================================================================+
|                        GreenLang Platform Architecture                        |
+==============================================================================+

                              +-----------------+
                              |   Load Balancer |
                              |  (nginx/ALB)    |
                              +--------+--------+
                                       |
            +--------------------------|---------------------------+
            |                          |                           |
            v                          v                           v
    +---------------+          +---------------+          +---------------+
    |   CBAM API    |          |   CSRD Web    |          | VCCI Backend  |
    |   Port 8001   |          |   Port 8002   |          |   Port 8000   |
    | (FastAPI)     |          | (FastAPI)     |          | (FastAPI)     |
    +-------+-------+          +-------+-------+          +-------+-------+
            |                          |                           |
            |                          |                           |
            +--------------------------|---------------------------+
                                       |
                              +--------v--------+
                              |  VCCI Worker    |
                              |  (Celery)       |
                              +-----------------+
                                       |
    +--------------------------|-------|------------------------------+
    |                          |       |                              |
    v                          v       v                              v
+-------+              +-------+   +-------+                  +----------+
|Postgres|<----------->| Redis |   |RabbitMQ|                 | Weaviate |
|  5432  |             | 6379  |   |  5672  |                 |   8080   |
+-------+              +-------+   +-------+                  +----------+
    |                      |           |                           |
    |   SHARED INFRASTRUCTURE LAYER    |                           |
    +----------------------------------+---------------------------+

                              MONITORING LAYER
    +---------------------------------------------------------------+
    |                                                               |
    |     +------------+              +------------+                |
    |     | Prometheus |------------->|  Grafana   |                |
    |     |    9090    |              |    3000    |                |
    |     +------------+              +------------+                |
    |                                                               |
    +---------------------------------------------------------------+

```

### Detailed Component Diagram

```
+=============================================================================+
|                           APPLICATION TIER                                   |
+=============================================================================+
|                                                                             |
|  +-------------------+  +-------------------+  +-------------------+         |
|  |    CBAM API       |  |    CSRD Web       |  |   VCCI Backend    |         |
|  |-------------------|  |-------------------|  |-------------------|         |
|  | - Import Copilot  |  | - Report Builder  |  | - Scope 3 Calc    |         |
|  | - Declaration Gen |  | - ESRS Mapping    |  | - Supply Chain    |         |
|  | - Compliance      |  | - AI Assistant    |  | - Emission Factor |         |
|  | - EU Submission   |  | - Audit Trail     |  | - Analytics       |         |
|  +--------+----------+  +--------+----------+  +--------+----------+         |
|           |                      |                      |                    |
|           | REST API             | REST API             | REST API           |
|           | (JSON/HTTP)          | (JSON/HTTP)          | (JSON/HTTP)        |
|           |                      |                      |                    |
+=============================================================================+
                                   |
+=============================================================================+
|                           DATA TIER                                          |
+=============================================================================+
|                                                                             |
|  +-------------------+  +-------------------+  +-------------------+         |
|  |    PostgreSQL     |  |      Redis        |  |    RabbitMQ       |         |
|  |-------------------|  |-------------------|  |-------------------|         |
|  | Schemas:          |  | Databases:        |  | Exchanges:        |         |
|  | - public (shared) |  | - 0: CBAM cache   |  | - cbam.events     |         |
|  | - cbam            |  | - 1: CSRD cache   |  | - csrd.events     |         |
|  | - csrd            |  | - 2: VCCI cache   |  | - vcci.events     |         |
|  | - vcci            |  | - 3: Celery       |  | - platform.events |         |
|  | - shared          |  | - 4: Results      |  |                   |         |
|  +-------------------+  +-------------------+  +-------------------+         |
|                                                                             |
|  +-------------------+  +-------------------+                               |
|  |    Weaviate       |  |   File Storage    |                               |
|  |-------------------|  |-------------------|                               |
|  | Collections:      |  | Volumes:          |                               |
|  | - Entity          |  | - cbam-uploads    |                               |
|  | - Product         |  | - csrd-output     |                               |
|  | - Supplier        |  | - vcci-data       |                               |
|  | - Regulation      |  | - logs            |                               |
|  +-------------------+  +-------------------+                               |
|                                                                             |
+=============================================================================+
```

---

## Component Descriptions

### Application Components

#### CBAM API (Port 8001)

The Carbon Border Adjustment Mechanism compliance application.

| Attribute | Value |
|-----------|-------|
| **Container Name** | cbam-api |
| **Image** | greenlang/cbam-app:latest |
| **Framework** | FastAPI (Python 3.11) |
| **Internal Port** | 8000 |
| **External Port** | 8001 |
| **Health Endpoint** | /health |
| **Metrics Endpoint** | /metrics |
| **API Docs** | /docs |
| **Database Schema** | cbam |
| **Redis DB** | 0 |

**Key Functions:**
- Import data from ERP systems (SAP, Oracle, Workday)
- Calculate embedded emissions for CBAM-covered goods
- Generate quarterly CBAM declarations
- Submit reports to EU CBAM registry

#### CSRD Web (Port 8002)

The Corporate Sustainability Reporting Directive reporting platform.

| Attribute | Value |
|-----------|-------|
| **Container Name** | csrd-web |
| **Image** | greenlang/csrd-app:latest |
| **Framework** | FastAPI (Python 3.11) |
| **Internal Port** | 8000 |
| **External Port** | 8002 |
| **Health Endpoint** | /health |
| **Metrics Endpoint** | /metrics |
| **API Docs** | /docs |
| **Database Schema** | csrd |
| **Redis DB** | 1 |

**Key Functions:**
- Map data to ESRS disclosure requirements
- AI-assisted report generation
- Document management and versioning
- Audit trail and compliance tracking

#### VCCI Backend (Port 8000)

The Value Chain Carbon Intelligence platform for Scope 3 emissions.

| Attribute | Value |
|-----------|-------|
| **Container Name** | vcci-backend |
| **Image** | greenlang/vcci-backend:latest |
| **Framework** | FastAPI (Python 3.11) |
| **Internal Port** | 8000 |
| **External Port** | 8000 |
| **Health Endpoint** | /health/live |
| **Metrics Endpoint** | /metrics |
| **API Docs** | /docs |
| **Database Schema** | vcci |
| **Redis DB** | 2 |

**Key Functions:**
- Calculate Scope 3 emissions across all 15 categories
- Supplier engagement and data collection
- Emission factor database management
- Carbon footprint analytics and reporting

#### VCCI Worker

Background task processor for asynchronous calculations.

| Attribute | Value |
|-----------|-------|
| **Container Name** | vcci-worker |
| **Image** | greenlang/vcci-worker:latest |
| **Framework** | Celery |
| **Broker** | Redis (DB 3) |
| **Result Backend** | Redis (DB 4) |

**Key Functions:**
- Process large emission calculations
- Generate PDF reports
- Handle data imports
- Execute scheduled tasks

### Infrastructure Components

#### PostgreSQL Database

Shared relational database for all applications.

| Attribute | Value |
|-----------|-------|
| **Container Name** | greenlang-postgres |
| **Image** | postgres:15-alpine |
| **Port** | 5432 |
| **Database** | greenlang_platform |
| **Max Connections** | 300 |
| **Shared Buffers** | 512MB |

**Schema Layout:**
```sql
greenlang_platform
├── public        -- Shared tables (users, organizations, roles)
├── cbam          -- CBAM-specific tables
├── csrd          -- CSRD-specific tables
├── vcci          -- VCCI-specific tables
└── shared        -- Cross-application views and functions
```

#### Redis Cache

Shared in-memory cache and message broker.

| Attribute | Value |
|-----------|-------|
| **Container Name** | greenlang-redis |
| **Image** | redis:7-alpine |
| **Port** | 6379 |
| **Max Memory** | 1GB |
| **Eviction Policy** | allkeys-lru |
| **Persistence** | AOF + RDB |

**Database Allocation:**
| DB | Application | Purpose |
|----|-------------|---------|
| 0 | CBAM | Session cache, API response cache |
| 1 | CSRD | Session cache, document cache |
| 2 | VCCI | Session cache, calculation cache |
| 3 | Celery | Task broker |
| 4 | Celery | Task results |

#### RabbitMQ Message Queue

Event-driven message broker for cross-application communication.

| Attribute | Value |
|-----------|-------|
| **Container Name** | greenlang-rabbitmq |
| **Image** | rabbitmq:3.12-management-alpine |
| **AMQP Port** | 5672 |
| **Management Port** | 15672 |
| **Virtual Host** | greenlang_platform |

**Exchange Configuration:**
```
greenlang_platform (vhost)
├── cbam.events       -- CBAM application events
├── csrd.events       -- CSRD application events
├── vcci.events       -- VCCI application events
└── platform.events   -- Cross-application events
```

#### Weaviate Vector Database

Vector database for semantic search and RAG capabilities.

| Attribute | Value |
|-----------|-------|
| **Container Name** | greenlang-weaviate |
| **Image** | semitechnologies/weaviate:1.23.0 |
| **Port** | 8080 |
| **Vectorizer** | text2vec-openai |

**Collections:**
- `Entity` - Company and supplier entities
- `Product` - Product and material embeddings
- `Regulation` - Regulatory text embeddings
- `Document` - Report document embeddings

### Monitoring Components

#### Prometheus

Metrics collection and alerting.

| Attribute | Value |
|-----------|-------|
| **Container Name** | greenlang-prometheus |
| **Image** | prom/prometheus:latest |
| **Port** | 9090 |
| **Retention** | 30 days |
| **Scrape Interval** | 15 seconds |

#### Grafana

Visualization and dashboards.

| Attribute | Value |
|-----------|-------|
| **Container Name** | greenlang-grafana |
| **Image** | grafana/grafana:latest |
| **Port** | 3000 |
| **Default User** | admin |
| **Data Source** | Prometheus |

---

## Data Flow

### Request Flow

```
                                    [Client Request]
                                           |
                                           v
                              +------------------------+
                              |     Load Balancer      |
                              |  (Round Robin/Health)  |
                              +------------------------+
                                           |
              +----------------------------+----------------------------+
              |                            |                            |
              v                            v                            v
     +----------------+           +----------------+           +----------------+
     |   CBAM API     |           |   CSRD Web     |           | VCCI Backend   |
     |   /api/v1/*    |           |   /api/v1/*    |           |   /api/v1/*    |
     +-------+--------+           +-------+--------+           +-------+--------+
             |                            |                            |
             |     +------------------------------------------+        |
             |     |           JWT Token Validation           |        |
             |     +------------------------------------------+        |
             |                            |                            |
             +----------------------------+----------------------------+
                                          |
                              +-----------+-----------+
                              |                       |
                              v                       v
                    +----------------+       +----------------+
                    |   PostgreSQL   |       |     Redis      |
                    |   (Primary)    |       |   (Cache)      |
                    +----------------+       +----------------+
```

### Event-Driven Data Flow

```
+-------------------+     Publish Event     +-------------------+
|   VCCI Backend    | -------------------> |    RabbitMQ       |
|                   |    "emissions.calc"   |                   |
+-------------------+                       +-------------------+
                                                   |
                                      +------------+------------+
                                      |                         |
                                      v                         v
                            +----------------+        +----------------+
                            |   CSRD Web     |        |   CBAM API     |
                            |  (Subscriber)  |        |  (Subscriber)  |
                            +----------------+        +----------------+
                                   |                         |
                         Update CSRD           Update CBAM
                         Report Data           Declaration
```

### Async Task Flow

```
+-------------------+     Submit Task      +-------------------+
|   VCCI Backend    | -------------------> |   Redis (Celery)  |
|   (API Handler)   |    DB=3 (Broker)     |                   |
+-------------------+                       +-------------------+
                                                   |
                                                   v
                                           +----------------+
                                           |  VCCI Worker   |
                                           |  (Celery)      |
                                           +-------+--------+
                                                   |
                              +--------------------+--------------------+
                              |                    |                    |
                              v                    v                    v
                    +----------------+    +----------------+    +----------------+
                    |   PostgreSQL   |    |   Weaviate     |    |  Redis DB=4    |
                    |  (Write Data)  |    | (Store Vector) |    | (Store Result) |
                    +----------------+    +----------------+    +----------------+
```

### Data Integration Flow

```
+=============================================================================+
|                        Cross-Application Data Flow                           |
+=============================================================================+

     +---------------+                                    +---------------+
     |   CBAM API    |                                    |   CSRD Web    |
     |               |                                    |               |
     | - Import data |                                    | - Pull CBAM   |
     | - Calculate   |                                    |   emissions   |
     | - Generate    |                                    | - Pull VCCI   |
     |   declarations|                                    |   Scope 3     |
     +-------+-------+                                    +-------+-------+
             |                                                    |
             |  1. Publish emissions                              |
             |     to RabbitMQ                                    |
             v                                                    v
     +---------------------------------------------------------------+
     |                       RabbitMQ                                 |
     |                    platform.events                             |
     +---------------------------------------------------------------+
             |                                                    ^
             | 2. Subscribe to                                    |
             |    emissions events                                |
             v                                                    |
     +---------------+                                    +-------+-------+
     |  VCCI Backend |                                    |               |
     |               |------------------------------------| 3. CSRD pulls |
     | - Scope 3     |      REST API call                 |    combined   |
     |   calculations|      /api/v1/vcci/emissions        |    emissions  |
     |               |                                    |    data       |
     +---------------+                                    +---------------+

```

---

## Security Boundaries

### Network Security Zones

```
+=============================================================================+
|                           SECURITY ZONES                                     |
+=============================================================================+

+-----------------------------------------------------------------------------+
|                         PUBLIC ZONE (DMZ)                                    |
|                                                                              |
|    Internet --> [WAF/CDN] --> [Load Balancer] --> [Ingress Controller]      |
|                                                                              |
+-----------------------------------------------------------------------------+
                                    |
                                    | HTTPS (TLS 1.3)
                                    v
+-----------------------------------------------------------------------------+
|                       APPLICATION ZONE                                       |
|                      (172.26.0.100-119)                                     |
|                                                                              |
|    +---------------+    +---------------+    +---------------+              |
|    |   CBAM API    |    |   CSRD Web    |    | VCCI Backend  |              |
|    |  172.26.0.101 |    |  172.26.0.102 |    |  172.26.0.103 |              |
|    +---------------+    +---------------+    +---------------+              |
|                                                                              |
|    Network Policy: Allow ingress from Load Balancer only                    |
|                    Allow egress to Data Zone only                           |
|                                                                              |
+-----------------------------------------------------------------------------+
                                    |
                                    | Internal Network (No Encryption)
                                    v
+-----------------------------------------------------------------------------+
|                         DATA ZONE                                            |
|                      (172.26.0.10-19)                                       |
|                                                                              |
|    +---------------+    +---------------+    +---------------+              |
|    |   PostgreSQL  |    |    Redis      |    |   RabbitMQ    |              |
|    |  172.26.0.10  |    |  172.26.0.11  |    |  172.26.0.12  |              |
|    +---------------+    +---------------+    +---------------+              |
|                                                                              |
|    +---------------+                                                        |
|    |   Weaviate    |                                                        |
|    |  172.26.0.13  |                                                        |
|    +---------------+                                                        |
|                                                                              |
|    Network Policy: Allow ingress from Application Zone only                 |
|                    Deny all internet egress                                 |
|                                                                              |
+-----------------------------------------------------------------------------+
                                    |
                                    | Monitoring Scrape
                                    v
+-----------------------------------------------------------------------------+
|                       MONITORING ZONE                                        |
|                      (172.26.0.200-219)                                     |
|                                                                              |
|    +---------------+    +---------------+                                   |
|    |  Prometheus   |    |   Grafana     |                                   |
|    |  172.26.0.201 |    |  172.26.0.202 |                                   |
|    +---------------+    +---------------+                                   |
|                                                                              |
|    Network Policy: Allow ingress from all zones (metrics scrape)            |
|                    Limited internet egress (Grafana plugins)                |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Authentication and Authorization

```
+=============================================================================+
|                    AUTHENTICATION FLOW                                       |
+=============================================================================+

    [User/Client]
          |
          | 1. POST /api/v1/auth/token
          |    (client_credentials or password grant)
          v
    +---------------+
    |   Auth Layer  |
    | (Any App API) |
    +-------+-------+
            |
            | 2. Validate credentials against
            |    shared users table (public.users)
            v
    +---------------+
    |   PostgreSQL  |
    | public.users  |
    +-------+-------+
            |
            | 3. Generate JWT with shared secret
            |    (SHARED_JWT_SECRET)
            v
    +---------------+
    |   JWT Token   |
    | - user_id     |
    | - org_id      |
    | - roles       |
    | - exp (30min) |
    +---------------+
            |
            | 4. Return token to client
            v
    [Client stores token]
            |
            | 5. Subsequent requests include
            |    Authorization: Bearer <token>
            v
    +---------------+
    | CBAM/CSRD/VCCI|
    |  (Any App)    |
    +-------+-------+
            |
            | 6. Validate JWT signature
            |    using SHARED_JWT_SECRET
            |    (token works on all apps)
            v
    [Request processed]
```

### Data Encryption

| Data State | Encryption Method | Key Management |
|------------|-------------------|----------------|
| **At Rest (PostgreSQL)** | AES-256 | AWS KMS / HashiCorp Vault |
| **At Rest (Redis)** | None (ephemeral) | N/A |
| **At Rest (Weaviate)** | AES-256 | Local key file |
| **At Rest (Files)** | AES-256 | Application-managed |
| **In Transit (External)** | TLS 1.3 | Let's Encrypt / ACM |
| **In Transit (Internal)** | Plaintext | N/A (network isolation) |
| **Secrets** | SOPS / Sealed Secrets | AWS KMS / GPG |

### Access Control Matrix

| Resource | CBAM | CSRD | VCCI | Admin |
|----------|------|------|------|-------|
| `public.users` | Read | Read | Read | Full |
| `public.organizations` | Read | Read | Read | Full |
| `cbam.*` | Full | Read | None | Full |
| `csrd.*` | None | Full | Read | Full |
| `vcci.*` | Read | Read | Full | Full |
| `shared.*` | Read | Read | Read | Full |
| Redis (all DBs) | Own DB | Own DB | Own DB | Full |
| RabbitMQ | Own exchanges | Own exchanges | Own exchanges | Full |
| Weaviate | Read | Read/Write | Read/Write | Full |

---

## Network Architecture

### Docker Network Configuration

```yaml
networks:
  greenlang-platform:
    driver: bridge
    ipam:
      config:
        - subnet: 172.26.0.0/16
          gateway: 172.26.0.1
```

### IP Address Allocation

| Component | IP Address | Port(s) |
|-----------|------------|---------|
| Gateway | 172.26.0.1 | - |
| PostgreSQL | 172.26.0.10 | 5432 |
| Redis | 172.26.0.11 | 6379 |
| RabbitMQ | 172.26.0.12 | 5672, 15672 |
| Weaviate | 172.26.0.13 | 8080 |
| CBAM API | 172.26.0.101 | 8000 |
| CSRD Web | 172.26.0.102 | 8000 |
| VCCI Backend | 172.26.0.103 | 8000 |
| VCCI Worker | 172.26.0.104 | - |
| Prometheus | 172.26.0.201 | 9090 |
| Grafana | 172.26.0.202 | 3000 |
| pgAdmin | 172.26.0.203 | 80 |

### Port Mapping (Host to Container)

| Service | Host Port | Container Port | Protocol |
|---------|-----------|----------------|----------|
| VCCI Backend | 8000 | 8000 | HTTP |
| CBAM API | 8001 | 8000 | HTTP |
| CSRD Web | 8002 | 8000 | HTTP |
| Grafana | 3000 | 3000 | HTTP |
| PostgreSQL | 5432 | 5432 | TCP |
| Redis | 6379 | 6379 | TCP |
| RabbitMQ AMQP | 5672 | 5672 | TCP |
| RabbitMQ Mgmt | 15672 | 15672 | HTTP |
| Weaviate | 8080 | 8080 | HTTP |
| Prometheus | 9090 | 9090 | HTTP |
| pgAdmin | 5050 | 80 | HTTP |

---

## Storage Architecture

### Volume Configuration

```
+=============================================================================+
|                         VOLUME ARCHITECTURE                                  |
+=============================================================================+

INFRASTRUCTURE VOLUMES (Persistent, Critical)
+---------------------------+---------------------------+
|  greenlang-postgres-data  |  greenlang-redis-data     |
|  - Database files         |  - AOF persistence        |
|  - WAL logs               |  - RDB snapshots          |
|  - Size: 100GB+           |  - Size: 10GB             |
+---------------------------+---------------------------+
|  greenlang-rabbitmq-data  |  greenlang-weaviate-data  |
|  - Queue persistence      |  - Vector storage         |
|  - Message store          |  - Index files            |
|  - Size: 20GB             |  - Size: 50GB             |
+---------------------------+---------------------------+

APPLICATION VOLUMES (Persistent, Important)
+---------------------------+---------------------------+---------------------------+
|       cbam-data           |       csrd-data           |       vcci-data           |
|  - Application data       |  - Application data       |  - Application data       |
|  - Size: 20GB             |  - Size: 20GB             |  - Size: 20GB             |
+---------------------------+---------------------------+---------------------------+
|       cbam-uploads        |       csrd-output         |       vcci-logs           |
|  - User uploads           |  - Generated reports      |  - Application logs       |
|  - Size: 50GB             |  - Size: 100GB            |  - Size: 10GB             |
+---------------------------+---------------------------+---------------------------+
|       cbam-output         |       csrd-logs           |                           |
|  - Generated files        |  - Application logs       |                           |
|  - Size: 50GB             |  - Size: 10GB             |                           |
+---------------------------+---------------------------+---------------------------+
|       cbam-logs           |                           |                           |
|  - Application logs       |                           |                           |
|  - Size: 10GB             |                           |                           |
+---------------------------+---------------------------+---------------------------+

MONITORING VOLUMES (Persistent, Important)
+---------------------------+---------------------------+
|  greenlang-prometheus-data|  greenlang-grafana-data   |
|  - Time-series metrics    |  - Dashboards             |
|  - Retention: 30 days     |  - User preferences       |
|  - Size: 50GB             |  - Size: 5GB              |
+---------------------------+---------------------------+
```

### Backup Strategy

| Volume Type | Backup Frequency | Retention | Method |
|-------------|------------------|-----------|--------|
| PostgreSQL | Continuous (WAL) + Daily | 30 days | pg_dump + S3 |
| Redis | Every 6 hours | 7 days | RDB + AOF to S3 |
| Weaviate | Daily | 30 days | API backup to S3 |
| Application Data | Daily | 14 days | tar + S3 |
| Logs | Daily | 7 days | S3 archival |

---

## Monitoring Architecture

### Metrics Collection

```
+=============================================================================+
|                        PROMETHEUS SCRAPE TARGETS                             |
+=============================================================================+

+-------------------+     +-------------------+     +-------------------+
|    cbam-api       |     |    csrd-web       |     |   vcci-backend    |
|  /metrics:8000    |     |  /metrics:8000    |     |  /metrics:8000    |
+-------------------+     +-------------------+     +-------------------+
         |                         |                         |
         +-----------+-------------+-----------+-------------+
                     |                         |
                     v                         v
              +-------------+           +-------------+
              |  Prometheus |           |   Grafana   |
              |    :9090    |---------->|    :3000    |
              +-------------+           +-------------+
                     ^
                     |
         +-----------+-------------+-----------+-------------+
         |                         |                         |
+-------------------+     +-------------------+     +-------------------+
|    postgres       |     |     redis         |     |    rabbitmq       |
|    :5432          |     |    :6379          |     |  /metrics:15692   |
+-------------------+     +-------------------+     +-------------------+
```

### Key Metrics

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| `http_requests_total` | Applications | N/A (trend) |
| `http_request_duration_seconds` | Applications | p99 > 5s |
| `up` | All components | == 0 for 2m |
| `pg_stat_database_numbackends` | PostgreSQL | > 250 |
| `redis_memory_used_bytes` | Redis | > 900MB |
| `rabbitmq_queue_messages` | RabbitMQ | > 10000 |

---

## Related Documentation

- [Operations Guide](./OPERATIONS.md) - Day-to-day operations and procedures
- [Troubleshooting Guide](./TROUBLESHOOTING.md) - Common issues and solutions
- [Deployment README](../README.md) - Quick start and configuration
- [Disaster Recovery](../platform-disaster-recovery.md) - Backup and recovery procedures
- [Security Audit](../security/SECURITY_AUDIT_EXECUTIVE_SUMMARY.md) - Security assessment

---

**Document Owner:** Platform Engineering
**Review Cycle:** Quarterly
**Next Review:** 2026-05-03
